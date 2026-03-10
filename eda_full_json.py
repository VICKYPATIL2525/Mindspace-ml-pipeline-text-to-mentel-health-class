"""
Comprehensive JSON EDA — Mental Health Synthetic Dataset (with Normal class)
=============================================================================
NO charts / images are produced.
All analysis results are saved in ONE single JSON file.

Run:
    python eda_full_json.py

Output → ./eda_full_report.json   (single file, created automatically)

Analyses performed
──────────────────
 01  Dataset overview (shape, dtypes, nulls, duplicates, memory)
 02  Class distribution (counts, proportions, entropy, imbalance)
 03  Descriptive statistics (mean/std/percentiles/skew/kurtosis per col)
 04  Normality tests — D'Agostino K² (all numeric cols)
 05  Outlier analysis — IQR & Z-score (all numeric cols)
 06  Per-class feature statistics (mean/median/std/min/max/IQR per class)
 07  Feature × class profile (class means, medians, stds per feature)
 08  Pearson correlation matrix (interpretable features)
 08b Highly correlated feature pairs (|r| ≥ threshold)
 09  Spearman correlation matrix (interpretable features)
 10  Kruskal-Wallis H-test (each feature vs target)
 11  One-way ANOVA F-test (each feature vs target)
 12  Eta-squared effect sizes (each feature vs target)
 13  Pairwise class comparisons — Mann-Whitney U + Cohen's d (top features)
 14  Fisher discriminant ratio (between-class vs within-class MS)
 15  Random Forest feature importance (MDI)
 16  Mutual Information scores (each feature vs target)
 17  PCA — Embedding features (variance explained)
 18  PCA — Interpretable features (variance explained + top loadings)
 19  Topic feature analysis per class
 20  Language distribution (if 'language' column exists)
 21  Categorical columns analysis
 22  Class centroid distances (Euclidean & cosine on scaled features)
 23  Feature variance decomposition (between-class % vs within-class %)
 24  Embedding statistics per class (mean/std vectors, centroid distances)
 25  Percentile profiles per class (p5–p95 for each linguistic feature)
 26  Feature ranking consensus (KW + RF + MI combined rank)
 27  Master summary report
"""

import os
import json
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
BASE        = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE, "mental_health_synthetic_dataset_with_normal.csv")
OUTPUT_FILE = os.path.join(BASE, "eda_full_report.json")

# Central report — all sections accumulate here
EDA_REPORT = {}

CORR_HIGH_THRESHOLD = 0.75   # flag pairs with |r| >= this
OUTLIER_Z_THRESHOLD = 3.0    # Z-score threshold
NORMALITY_SAMPLE    = 3000   # rows sampled per column for normality test
RF_SAMPLE           = 40000  # rows used for RF training (speed)
RF_N_ESTIMATORS     = 150
PCA_EMBED_N_COMP    = 10
PCA_FEAT_N_COMP     = 15
PAIRWISE_TOP_N      = 15     # top-N features for pairwise class comparison
PERCENTILES         = [5, 10, 25, 50, 75, 90, 95]


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def _cvt(obj):
    """Recursively convert numpy / pandas types to JSON-serialisable Python types."""
    if isinstance(obj, dict):
        return {k: _cvt(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_cvt(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_cvt(x) for x in obj.tolist()]
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def jstore(name, obj):
    """Store obj in the central EDA_REPORT under a clean section key."""
    key = name.replace(".json", "")
    EDA_REPORT[key] = _cvt(obj)
    print(f"  [stored] {key}")
    return obj

# Alias kept so no call-site needs changing
jdump = jstore


def cohens_d(a, b):
    """Pooled Cohen's d between two 1-D arrays."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    pooled = np.sqrt(((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2))
    return float((np.mean(a) - np.mean(b)) / pooled) if pooled > 0 else 0.0


def eta_squared(groups):
    """Eta-squared effect size (proportion of total variance explained by group)."""
    all_vals   = np.concatenate(groups)
    grand_mean = np.mean(all_vals)
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    ss_total   = np.sum((all_vals - grand_mean) ** 2)
    return float(ss_between / ss_total) if ss_total > 0 else 0.0


def effect_label_d(d):
    if d is None:
        return None
    return "large" if abs(d) >= 0.8 else "medium" if abs(d) >= 0.5 else "small" if abs(d) >= 0.2 else "negligible"


def effect_label_eta(e):
    return "large" if e >= 0.14 else "medium" if e >= 0.06 else "small" if e >= 0.01 else "negligible"


def rank_dict(feat_list):
    return {feat: (idx + 1) for idx, feat in enumerate(feat_list)}


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("\n══ Loading dataset ══")
df = pd.read_csv(DATA_PATH)
print(f"  Shape : {df.shape}")

TARGET_COL = "target"
CLASSES    = sorted(df[TARGET_COL].unique().tolist())
N_CLASSES  = len(CLASSES)

NUM_COLS = df.select_dtypes(include="number").columns.tolist()
CAT_COLS = [c for c in df.select_dtypes(include="object").columns if c != TARGET_COL]

# Column groups ----------------------------------------------------------------
EMB_COLS        = sorted([c for c in NUM_COLS if c.startswith("emb_")])
# pure topic features: topic_0, topic_1, … (digit after underscore)
TOPIC_COLS      = sorted([c for c in NUM_COLS if c.startswith("topic_") and c.split("_")[-1].isdigit()])
# everything else numeric = linguistic / other engineered features
LING_COLS       = [c for c in NUM_COLS if c not in EMB_COLS and c not in TOPIC_COLS]
# interpretable = linguistic + topic (used for most analyses)
FEAT_COLS       = LING_COLS + TOPIC_COLS

le = LabelEncoder()
y  = le.fit_transform(df[TARGET_COL])

print(f"  Classes      : {CLASSES}")
print(f"  Numeric cols : {len(NUM_COLS)} "
      f"(ling={len(LING_COLS)}, topic={len(TOPIC_COLS)}, emb={len(EMB_COLS)})")
print(f"  Categorical  : {list(CAT_COLS)}")

# ═════════════════════════════════════════════════════════════════════════════
# 01  DATASET OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
print("\n[01] Dataset Overview")

dtype_map   = {col: str(df[col].dtype) for col in df.columns}
null_counts = {col: int(cnt) for col, cnt in df.isnull().sum().items() if cnt > 0}
null_pct    = {col: round(float(cnt / len(df) * 100), 2) for col, cnt in df.isnull().sum().items() if cnt > 0}
n_dups      = int(df.duplicated().sum())
memory_mb   = round(float(df.memory_usage(deep=True).sum() / 1024 ** 2), 2)

overview = {
    "rows"               : int(df.shape[0]),
    "cols"               : int(df.shape[1]),
    "target_column"      : TARGET_COL,
    "classes"            : CLASSES,
    "n_classes"          : N_CLASSES,
    "numeric_cols_count" : len(NUM_COLS),
    "categorical_cols_count": len(CAT_COLS) + 1,     # +1 for target
    "column_dtypes"      : dtype_map,
    "null_counts"        : null_counts or "none",
    "null_percentages"   : null_pct    or "none",
    "total_null_cells"   : int(df.isnull().sum().sum()),
    "duplicate_rows"     : n_dups,
    "memory_usage_mb"    : memory_mb,
    "column_groups"      : {
        "linguistic"  : LING_COLS,
        "topic"       : TOPIC_COLS,
        "embedding"   : EMB_COLS,
        "categorical" : list(CAT_COLS),
    },
}
jdump("01_overview.json", overview)


# ═════════════════════════════════════════════════════════════════════════════
# 02  CLASS DISTRIBUTION
# ═════════════════════════════════════════════════════════════════════════════
print("\n[02] Class Distribution")

counts    = df[TARGET_COL].value_counts().sort_values(ascending=False)
props     = (counts / len(df) * 100).round(4)
imbalance = round(float(counts.max() / counts.min()), 4)

p         = (counts / counts.sum()).values
entropy   = float(-np.sum(p * np.log2(p + 1e-12)))
max_ent   = float(np.log2(N_CLASSES))

dist = {
    "counts"                : counts.to_dict(),
    "proportions_pct"       : props.to_dict(),
    "imbalance_ratio"       : imbalance,
    "most_frequent_class"   : str(counts.idxmax()),
    "least_frequent_class"  : str(counts.idxmin()),
    "shannon_entropy_bits"  : round(entropy, 4),
    "max_possible_entropy"  : round(max_ent, 4),
    "entropy_pct_of_max"    : round(entropy / max_ent * 100, 2),
    "note_entropy"          : "100% means perfectly balanced; lower = more skewed",
}
jdump("02_class_distribution.json", dist)


# ═════════════════════════════════════════════════════════════════════════════
# 03  DESCRIPTIVE STATISTICS — ALL NUMERIC COLUMNS
# ═════════════════════════════════════════════════════════════════════════════
print("\n[03] Descriptive Statistics")

desc_stats = {}
for col in NUM_COLS:
    s = df[col].dropna()
    if len(s) == 0:
        continue
    pcts = np.percentile(s, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    desc_stats[col] = {
        "count"    : int(s.count()),
        "mean"     : round(float(s.mean()), 6),
        "std"      : round(float(s.std()),  6),
        "variance" : round(float(s.var()),  6),
        "min"      : round(float(s.min()),  6),
        "p1"       : round(float(pcts[0]),  6),
        "p5"       : round(float(pcts[1]),  6),
        "p10"      : round(float(pcts[2]),  6),
        "p25"      : round(float(pcts[3]),  6),
        "p50"      : round(float(pcts[4]),  6),
        "p75"      : round(float(pcts[5]),  6),
        "p90"      : round(float(pcts[6]),  6),
        "p95"      : round(float(pcts[7]),  6),
        "p99"      : round(float(pcts[8]),  6),
        "max"      : round(float(s.max()),  6),
        "range"    : round(float(s.max() - s.min()), 6),
        "iqr"      : round(float(pcts[5] - pcts[3]), 6),
        "cv_pct"   : round(float(s.std() / (abs(s.mean()) + 1e-12) * 100), 4),
        "skewness" : round(float(sp_stats.skew(s)), 4),
        "kurtosis" : round(float(sp_stats.kurtosis(s)), 4),   # excess kurtosis
        "mode"     : round(float(s.mode().iloc[0]), 6) if len(s.mode()) > 0 else None,
    }
jdump("03_descriptive_statistics.json", desc_stats)


# ═════════════════════════════════════════════════════════════════════════════
# 04  NORMALITY TESTS — D'Agostino K² (sampled)
# ═════════════════════════════════════════════════════════════════════════════
print("\n[04] Normality Tests (D'Agostino K²)")

normality = {}
for col in NUM_COLS:
    s = df[col].dropna()
    samp = s.sample(n=min(NORMALITY_SAMPLE, len(s)), random_state=42).values
    try:
        stat, pval = sp_stats.normaltest(samp)
        normality[col] = {
            "test"          : "D'Agostino K²",
            "sample_size"   : len(samp),
            "statistic"     : round(float(stat), 4),
            "p_value"       : float(pval),
            "is_normal_p05" : bool(pval > 0.05),
            "skewness"      : round(float(sp_stats.skew(samp)), 4),
            "kurtosis"      : round(float(sp_stats.kurtosis(samp)), 4),
        }
    except Exception as exc:
        normality[col] = {"error": str(exc)}

n_normal    = sum(1 for v in normality.values() if v.get("is_normal_p05", False))
n_non_normal = len(normality) - n_normal

jdump("04_normality_tests.json", {
    "test"              : "D'Agostino K² (omnibus skew+kurtosis test)",
    "sample_per_column" : NORMALITY_SAMPLE,
    "summary"           : {"normal_p05": n_normal, "non_normal_p05": n_non_normal},
    "results"           : normality,
})


# ═════════════════════════════════════════════════════════════════════════════
# 05  OUTLIER ANALYSIS — IQR & Z-score
# ═════════════════════════════════════════════════════════════════════════════
print("\n[05] Outlier Analysis (IQR & Z-score)")

outliers = {}
for col in NUM_COLS:
    s = df[col].dropna()
    q25, q75 = np.percentile(s, [25, 75])
    iqr      = q75 - q25
    lo, hi   = q25 - 1.5 * iqr, q75 + 1.5 * iqr
    iqr_n    = int(((s < lo) | (s > hi)).sum())
    z        = np.abs((s - s.mean()) / (s.std() + 1e-12))
    z_n      = int((z > OUTLIER_Z_THRESHOLD).sum())
    outliers[col] = {
        "iqr_lower_fence"    : round(float(lo), 6),
        "iqr_upper_fence"    : round(float(hi), 6),
        "iqr_outliers_n"     : iqr_n,
        "iqr_outliers_pct"   : round(iqr_n / len(s) * 100, 3),
        "zscore_outliers_n"  : z_n,
        "zscore_outliers_pct": round(z_n / len(s) * 100, 3),
    }
jdump("05_outlier_analysis.json", {
    "iqr_multiplier"    : 1.5,
    "zscore_threshold"  : OUTLIER_Z_THRESHOLD,
    "results"           : outliers,
})


# ═════════════════════════════════════════════════════════════════════════════
# 06  PER-CLASS FEATURE STATISTICS
# ═════════════════════════════════════════════════════════════════════════════
print("\n[06] Per-Class Feature Statistics")

per_class = {}
for cls in CLASSES:
    sub = df[df[TARGET_COL] == cls][FEAT_COLS]
    cls_stats = {}
    for col in FEAT_COLS:
        s = sub[col].dropna()
        q25, q75 = float(np.percentile(s, 25)), float(np.percentile(s, 75))
        cls_stats[col] = {
            "n"      : int(len(s)),
            "mean"   : round(float(s.mean()),   6),
            "median" : round(float(s.median()), 6),
            "std"    : round(float(s.std()),    6),
            "min"    : round(float(s.min()),    6),
            "max"    : round(float(s.max()),    6),
            "p25"    : round(q25, 6),
            "p75"    : round(q75, 6),
            "iqr"    : round(q75 - q25, 6),
        }
    per_class[cls] = cls_stats
jdump("06_per_class_feature_statistics.json", per_class)


# ═════════════════════════════════════════════════════════════════════════════
# 07  FEATURE × CLASS PROFILE (flipped view: feature → each class)
# ═════════════════════════════════════════════════════════════════════════════
print("\n[07] Feature × Class Profile")

feat_profile = {}
for feat in FEAT_COLS:
    feat_profile[feat] = {
        "global_mean"    : round(float(df[feat].mean()), 6),
        "global_median"  : round(float(df[feat].median()), 6),
        "global_std"     : round(float(df[feat].std()),  6),
        "class_means"    : df.groupby(TARGET_COL)[feat].mean().round(6).to_dict(),
        "class_medians"  : df.groupby(TARGET_COL)[feat].median().round(6).to_dict(),
        "class_stds"     : df.groupby(TARGET_COL)[feat].std().round(6).to_dict(),
        "class_mins"     : df.groupby(TARGET_COL)[feat].min().round(6).to_dict(),
        "class_maxs"     : df.groupby(TARGET_COL)[feat].max().round(6).to_dict(),
    }
jdump("07_feature_class_profile.json", feat_profile)


# ═════════════════════════════════════════════════════════════════════════════
# 08  PEARSON CORRELATION MATRIX (interpretable features)
# ═════════════════════════════════════════════════════════════════════════════
print("\n[08] Pearson Correlation Matrix")

pearson = df[FEAT_COLS].corr(method="pearson").round(4)
jdump("08_correlation_pearson.json", {
    "method"  : "Pearson r",
    "features": FEAT_COLS,
    "matrix"  : pearson.to_dict(),
})

# 08b — highly correlated pairs
pairs = []
for i in range(len(FEAT_COLS)):
    for j in range(i + 1, len(FEAT_COLS)):
        val = float(pearson.iloc[i, j])
        if abs(val) >= CORR_HIGH_THRESHOLD:
            pairs.append({
                "feature_1" : FEAT_COLS[i],
                "feature_2" : FEAT_COLS[j],
                "pearson_r" : round(val, 4),
                "abs_r"     : round(abs(val), 4),
                "direction" : "positive" if val > 0 else "negative",
            })
pairs.sort(key=lambda x: -x["abs_r"])
jdump("08b_highly_correlated_pairs.json", {
    "threshold" : CORR_HIGH_THRESHOLD,
    "n_pairs"   : len(pairs),
    "pairs"     : pairs,
})


# ═════════════════════════════════════════════════════════════════════════════
# 09  SPEARMAN CORRELATION MATRIX
# ═════════════════════════════════════════════════════════════════════════════
print("\n[09] Spearman Correlation Matrix")

spearman = df[FEAT_COLS].corr(method="spearman").round(4)
jdump("09_correlation_spearman.json", {
    "method"  : "Spearman rho",
    "features": FEAT_COLS,
    "matrix"  : spearman.to_dict(),
})


# ═════════════════════════════════════════════════════════════════════════════
# 10  KRUSKAL-WALLIS H-TEST (each feature vs target)
# ═════════════════════════════════════════════════════════════════════════════
print("\n[10] Kruskal-Wallis H-Tests")

kw_raw = {}
for feat in FEAT_COLS:
    groups = [df.loc[df[TARGET_COL] == cls, feat].dropna().values for cls in CLASSES]
    if all(len(g) >= 2 for g in groups):
        stat, pval = sp_stats.kruskal(*groups)
        kw_raw[feat] = {
            "H_statistic" : round(float(stat), 4),
            "p_value"     : float(pval),
            "significant" : bool(pval < 0.05),
        }

kw_sorted  = dict(sorted(kw_raw.items(), key=lambda x: -x[1]["H_statistic"]))
sig_feats  = [f for f, v in kw_raw.items() if v["significant"]]
insig_feats = [f for f, v in kw_raw.items() if not v["significant"]]

jdump("10_kruskal_wallis.json", {
    "test"                    : "Kruskal-Wallis H-test (non-parametric one-way ANOVA)",
    "n_features_tested"       : len(kw_raw),
    "n_significant_p05"       : len(sig_feats),
    "n_non_significant"       : len(insig_feats),
    "significant_features"    : sig_feats,
    "non_significant_features": insig_feats,
    "results_sorted_by_H"     : kw_sorted,
})


# ═════════════════════════════════════════════════════════════════════════════
# 11  ONE-WAY ANOVA F-TEST (parametric, each feature vs target)
# ═════════════════════════════════════════════════════════════════════════════
print("\n[11] One-Way ANOVA")

anova_raw = {}
for feat in FEAT_COLS:
    groups = [df.loc[df[TARGET_COL] == cls, feat].dropna().values for cls in CLASSES]
    if all(len(g) >= 2 for g in groups):
        f_stat, pval = sp_stats.f_oneway(*groups)
        anova_raw[feat] = {
            "F_statistic" : round(float(f_stat), 4),
            "p_value"     : float(pval),
            "significant" : bool(pval < 0.05),
        }

anova_sorted = dict(sorted(anova_raw.items(), key=lambda x: -x[1]["F_statistic"]))
jdump("11_anova.json", {
    "test"              : "One-Way ANOVA (F-test) — parametric, assumes normality",
    "n_features_tested" : len(anova_raw),
    "n_significant_p05" : sum(1 for v in anova_raw.values() if v["significant"]),
    "results"           : anova_sorted,
})


# ═════════════════════════════════════════════════════════════════════════════
# 12  ETA-SQUARED EFFECT SIZES
# ═════════════════════════════════════════════════════════════════════════════
print("\n[12] Eta-Squared Effect Sizes")

eta2_raw = {}
for feat in FEAT_COLS:
    groups = [df.loc[df[TARGET_COL] == cls, feat].dropna().values for cls in CLASSES]
    if any(len(g) >= 1 for g in groups):
        e2 = eta_squared(groups)
        eta2_raw[feat] = {
            "eta_squared"          : round(e2, 6),
            "effect_interpretation": effect_label_eta(e2),
        }

eta2_sorted = dict(sorted(eta2_raw.items(), key=lambda x: -x[1]["eta_squared"]))
jdump("12_eta_squared.json", {
    "description"    : "Proportion of total feature variance explained by class membership.",
    "benchmarks"     : {"negligible": "<0.01", "small": "0.01-0.06", "medium": "0.06-0.14", "large": ">=0.14"},
    "n_large"        : sum(1 for v in eta2_raw.values() if v["effect_interpretation"] == "large"),
    "n_medium"       : sum(1 for v in eta2_raw.values() if v["effect_interpretation"] == "medium"),
    "n_small"        : sum(1 for v in eta2_raw.values() if v["effect_interpretation"] == "small"),
    "n_negligible"   : sum(1 for v in eta2_raw.values() if v["effect_interpretation"] == "negligible"),
    "results"        : eta2_sorted,
})


# ═════════════════════════════════════════════════════════════════════════════
# 13  PAIRWISE CLASS COMPARISONS — Mann-Whitney U + Cohen's d
#     Applied to top-N most discriminative features
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n[13] Pairwise Class Comparisons (top {PAIRWISE_TOP_N} features)")

top_feats    = list(kw_sorted.keys())[:PAIRWISE_TOP_N]
class_pairs  = list(combinations(CLASSES, 2))

pairwise = {}
for feat in top_feats:
    feat_res = {}
    for cls_a, cls_b in class_pairs:
        a = df.loc[df[TARGET_COL] == cls_a, feat].dropna().values
        b = df.loc[df[TARGET_COL] == cls_b, feat].dropna().values
        try:
            u_stat, pval = sp_stats.mannwhitneyu(a, b, alternative="two-sided")
            d = cohens_d(a, b)
            feat_res[f"{cls_a}_vs_{cls_b}"] = {
                "U_statistic"    : round(float(u_stat), 2),
                "p_value"        : float(pval),
                "significant_p05": bool(pval < 0.05),
                "cohens_d"       : round(d, 4) if d is not None else None,
                "effect_size"    : effect_label_d(d),
                "mean_a"         : round(float(np.mean(a)), 4),
                "mean_b"         : round(float(np.mean(b)), 4),
            }
        except Exception as exc:
            feat_res[f"{cls_a}_vs_{cls_b}"] = {"error": str(exc)}
    pairwise[feat] = feat_res

jdump("13_pairwise_class_comparisons.json", {
    "top_features_tested" : top_feats,
    "n_class_pairs"       : len(class_pairs),
    "class_pairs"         : [f"{a}_vs_{b}" for a, b in class_pairs],
    "test"                : "Mann-Whitney U (two-sided)",
    "effect_size_metric"  : "Cohen's d (pooled SD)",
    "results"             : pairwise,
})


# ═════════════════════════════════════════════════════════════════════════════
# 14  FISHER DISCRIMINANT RATIO (between-class / within-class MS)
# ═════════════════════════════════════════════════════════════════════════════
print("\n[14] Fisher Discriminant Ratio")

fisher = {}
for feat in FEAT_COLS:
    groups     = [df.loc[df[TARGET_COL] == cls, feat].dropna().values for cls in CLASSES]
    all_vals   = np.concatenate(groups)
    grand_mean = np.mean(all_vals)
    n_total    = len(all_vals)

    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    ss_within  = sum(np.sum((g - np.mean(g)) ** 2) for g in groups)

    df_between = N_CLASSES - 1
    df_within  = n_total - N_CLASSES
    ms_between = ss_between / df_between if df_between > 0 else 0.0
    ms_within  = ss_within  / df_within  if df_within  > 0 else 1e-12

    fisher[feat] = {
        "fisher_ratio"  : round(float(ms_between / (ms_within + 1e-12)), 4),
        "ms_between"    : round(float(ms_between), 4),
        "ms_within"     : round(float(ms_within),  4),
        "ss_between"    : round(float(ss_between), 4),
        "ss_within"     : round(float(ss_within),  4),
    }

fisher_sorted = dict(sorted(fisher.items(), key=lambda x: -x[1]["fisher_ratio"]))
jdump("14_fisher_discriminant_ratio.json", {
    "description" : "Higher ratio = stronger class separation. Equivalent to the F-statistic in ANOVA.",
    "results"     : fisher_sorted,
})


# ═════════════════════════════════════════════════════════════════════════════
# 15  RANDOM FOREST FEATURE IMPORTANCE (MDI)
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n[15] Random Forest Feature Importance (n_sample={RF_SAMPLE})")

rng_idx = np.random.default_rng(42).choice(len(df), size=min(RF_SAMPLE, len(df)), replace=False)
X_rf    = df.iloc[rng_idx][FEAT_COLS].fillna(df[FEAT_COLS].median())
y_rf    = y[rng_idx]

rf = RandomForestClassifier(
    n_estimators     = RF_N_ESTIMATORS,
    max_depth        = 14,
    min_samples_leaf = 5,
    n_jobs           = -1,
    random_state     = 42,
    class_weight     = "balanced",
)
rf.fit(X_rf, y_rf)
rf_imp   = pd.Series(rf.feature_importances_, index=FEAT_COLS).sort_values(ascending=False)
top10_rf = rf_imp.head(10).round(6).to_dict()

# Per-class feature importance from per-tree impurity (approximation via class proba feature importance)
jdump("15_rf_feature_importance.json", {
    "model"           : f"RandomForestClassifier(n_estimators={RF_N_ESTIMATORS}, max_depth=14)",
    "sample_size"     : int(len(X_rf)),
    "importance_type" : "Mean Decrease in Impurity (MDI)",
    "all_importances" : rf_imp.round(6).to_dict(),
    "top_10"          : top10_rf,
    "top_10_features" : list(top10_rf.keys()),
    "classes_order"   : list(le.classes_),
})


# ═════════════════════════════════════════════════════════════════════════════
# 16  MUTUAL INFORMATION SCORES
# ═════════════════════════════════════════════════════════════════════════════
print("\n[16] Mutual Information Scores")

X_mi      = df[FEAT_COLS].fillna(df[FEAT_COLS].median())
mi_scores = mutual_info_classif(X_mi, y, discrete_features=False, random_state=42)
mi_series = pd.Series(mi_scores, index=FEAT_COLS).sort_values(ascending=False)

jdump("16_mutual_information.json", {
    "description" : "Mutual information between each feature and the target label (higher = more informative).",
    "all_scores"  : mi_series.round(6).to_dict(),
    "top_10"      : mi_series.head(10).round(6).to_dict(),
    "top_10_features": list(mi_series.head(10).index),
})


# ═════════════════════════════════════════════════════════════════════════════
# 17  PCA — EMBEDDING FEATURES
# ═════════════════════════════════════════════════════════════════════════════
print("\n[17] PCA — Embedding Features")

pca_emb_result = {"note": "No embedding columns found."}
if EMB_COLS:
    X_emb   = df[EMB_COLS].fillna(0).values
    k_emb   = min(PCA_EMBED_N_COMP, len(EMB_COLS))
    pca_emb = PCA(n_components=k_emb, random_state=42)
    pca_emb.fit(X_emb)
    evr_emb = pca_emb.explained_variance_ratio_
    cum_emb = np.cumsum(evr_emb)

    pca_emb_result = {
        "n_embedding_features"         : len(EMB_COLS),
        "n_components_analysed"        : k_emb,
        "explained_variance_ratio"     : [round(float(v), 4) for v in evr_emb],
        "cumulative_variance_ratio"    : [round(float(v), 4) for v in cum_emb],
        "total_variance_top_k_pct"     : round(float(cum_emb[-1] * 100), 2),
        "components_for_80pct"         : int(np.argmax(cum_emb >= 0.80) + 1) if any(cum_emb >= 0.80) else None,
        "components_for_90pct"         : int(np.argmax(cum_emb >= 0.90) + 1) if any(cum_emb >= 0.90) else None,
        "per_component"                : [
            {
                "component"            : i + 1,
                "variance_explained_pct": round(float(evr_emb[i] * 100), 3),
                "cumulative_pct"       : round(float(cum_emb[i] * 100), 3),
            }
            for i in range(k_emb)
        ],
    }
jdump("17_pca_embeddings.json", pca_emb_result)


# ═════════════════════════════════════════════════════════════════════════════
# 18  PCA — INTERPRETABLE FEATURES (scaled)
# ═════════════════════════════════════════════════════════════════════════════
print("\n[18] PCA — Interpretable Features")

X_feat   = df[FEAT_COLS].fillna(df[FEAT_COLS].median()).values
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_feat)

k_feat    = min(PCA_FEAT_N_COMP, len(FEAT_COLS))
pca_feat  = PCA(n_components=k_feat, random_state=42)
pca_feat.fit(X_scaled)
evr_feat  = pca_feat.explained_variance_ratio_
cum_feat  = np.cumsum(evr_feat)

# Top-5 feature loadings per principal component
loadings = pd.DataFrame(
    pca_feat.components_,
    columns=FEAT_COLS,
    index=[f"PC{i + 1}" for i in range(k_feat)],
)
top_loadings = {}
for pc in loadings.index:
    row = loadings.loc[pc].abs().sort_values(ascending=False)
    top_loadings[pc] = {feat: round(float(loadings.loc[pc, feat]), 4) for feat in row.head(5).index}

jdump("18_pca_interpretable_features.json", {
    "n_features"               : len(FEAT_COLS),
    "n_components_analysed"    : k_feat,
    "scaling"                  : "StandardScaler (zero mean, unit variance)",
    "explained_variance_ratio" : [round(float(v), 4) for v in evr_feat],
    "cumulative_variance_ratio": [round(float(v), 4) for v in cum_feat],
    "components_for_80pct"     : int(np.argmax(cum_feat >= 0.80) + 1) if any(cum_feat >= 0.80) else None,
    "components_for_90pct"     : int(np.argmax(cum_feat >= 0.90) + 1) if any(cum_feat >= 0.90) else None,
    "top_5_loadings_per_pc"    : top_loadings,
    "per_component"            : [
        {
            "component"              : i + 1,
            "variance_explained_pct" : round(float(evr_feat[i] * 100), 3),
            "cumulative_pct"         : round(float(cum_feat[i] * 100), 3),
        }
        for i in range(k_feat)
    ],
})


# ═════════════════════════════════════════════════════════════════════════════
# 19  TOPIC FEATURE ANALYSIS PER CLASS
# ═════════════════════════════════════════════════════════════════════════════
print("\n[19] Topic Feature Analysis")

topic_per_feat = {}
for topic in TOPIC_COLS:
    topic_per_feat[topic] = {
        "global_mean" : round(float(df[topic].mean()), 6),
        "global_std"  : round(float(df[topic].std()),  6),
        "by_class"    : df.groupby(TARGET_COL)[topic].agg(["mean", "std", "min", "max"]).round(6).to_dict(),
    }

if TOPIC_COLS:
    topic_means   = df.groupby(TARGET_COL)[TOPIC_COLS].mean()
    topic_dom     = {cls: str(topic_means.loc[cls].idxmax()) for cls in CLASSES}
    topic_z       = (topic_means - topic_means.mean()) / (topic_means.std() + 1e-9)
    topic_z_dict  = topic_z.round(4).to_dict()
else:
    topic_dom    = {}
    topic_z_dict = {}

jdump("19_topic_analysis.json", {
    "topic_columns"           : TOPIC_COLS,
    "dominant_topic_per_class": topic_dom,
    "z_scored_class_means"    : topic_z_dict,
    "per_topic_stats"         : topic_per_feat,
})


# ═════════════════════════════════════════════════════════════════════════════
# 20  LANGUAGE DISTRIBUTION
# ═════════════════════════════════════════════════════════════════════════════
print("\n[20] Language Distribution")

if "language" in df.columns:
    lang_overall     = df["language"].value_counts().to_dict()
    lang_by_class    = df.groupby([TARGET_COL, "language"]).size().unstack(fill_value=0)
    lang_pct         = (lang_by_class.div(lang_by_class.sum(axis=1), axis=0) * 100).round(2)
    jdump("20_language_distribution.json", {
        "overall_language_counts"    : lang_overall,
        "counts_by_class"            : lang_by_class.to_dict(),
        "proportions_pct_by_class"   : lang_pct.to_dict(),
    })
else:
    jdump("20_language_distribution.json", {"note": "'language' column not present in dataset"})


# ═════════════════════════════════════════════════════════════════════════════
# 21  CATEGORICAL COLUMNS ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
print("\n[21] Categorical Columns Analysis")

cat_analysis = {}
for col in CAT_COLS:
    vc = df[col].value_counts()
    cross = df.groupby([TARGET_COL, col]).size().unstack(fill_value=0)
    cat_analysis[col] = {
        "n_unique"        : int(df[col].nunique()),
        "top_20_values"   : vc.head(20).to_dict(),
        "null_count"      : int(df[col].isnull().sum()),
        "cross_tab_by_class": cross.to_dict(),
    }
jdump("21_categorical_analysis.json", cat_analysis if cat_analysis else {"note": "No categorical columns found."})


# ═════════════════════════════════════════════════════════════════════════════
# 22  CLASS CENTROID DISTANCES (Euclidean & cosine on scaled features)
# ═════════════════════════════════════════════════════════════════════════════
print("\n[22] Class Centroid Distances")

X_all  = df[FEAT_COLS].fillna(df[FEAT_COLS].median()).values
X_norm = StandardScaler().fit_transform(X_all)

centroids      = {cls: X_norm[df[TARGET_COL] == cls].mean(axis=0) for cls in CLASSES}
centroid_mat   = np.array([centroids[cls] for cls in CLASSES])
euc_mat        = euclidean_distances(centroid_mat)
cos_mat        = cosine_distances(centroid_mat)

euc_pairs, cos_pairs = {}, {}
for i, ci in enumerate(CLASSES):
    for j, cj in enumerate(CLASSES):
        if i < j:
            key = f"{ci}_vs_{cj}"
            euc_pairs[key] = round(float(euc_mat[i, j]), 4)
            cos_pairs[key] = round(float(cos_mat[i, j]), 4)

jdump("22_class_centroid_distances.json", {
    "features_used"       : FEAT_COLS,
    "scaling"             : "StandardScaler",
    "euclidean_pairwise"  : euc_pairs,
    "cosine_pairwise"     : cos_pairs,
    "most_similar_pair"   : min(euc_pairs, key=euc_pairs.get) if euc_pairs else None,
    "most_distant_pair"   : max(euc_pairs, key=euc_pairs.get) if euc_pairs else None,
    "sorted_by_distance"  : dict(sorted(euc_pairs.items(), key=lambda x: x[1])),
})


# ═════════════════════════════════════════════════════════════════════════════
# 23  FEATURE VARIANCE DECOMPOSITION (between-class % vs within-class %)
# ═════════════════════════════════════════════════════════════════════════════
print("\n[23] Feature Variance Decomposition")

var_decomp = {}
for feat in FEAT_COLS:
    total_var = float(df[feat].var())
    if total_var < 1e-15:
        var_decomp[feat] = {"total_variance": 0.0, "between_class_pct": 0.0, "within_class_pct": 100.0}
        continue
    n_k       = df.groupby(TARGET_COL)[feat].count()
    cls_means = df.groupby(TARGET_COL)[feat].mean()
    grand_mean = float(df[feat].mean())
    between   = float((n_k * (cls_means - grand_mean) ** 2).sum() / len(df))
    within    = total_var - between
    var_decomp[feat] = {
        "total_variance"    : round(total_var, 6),
        "between_class_var" : round(between, 6),
        "within_class_var"  : round(within,  6),
        "between_class_pct" : round(between / total_var * 100, 2),
        "within_class_pct"  : round(within  / total_var * 100, 2),
    }

var_sorted = dict(sorted(var_decomp.items(), key=lambda x: -x[1]["between_class_pct"]))
jdump("23_feature_variance_decomposition.json", {
    "description" : "% of each feature's variance explained by class membership vs within-class spread.",
    "results"     : var_sorted,
})


# ═════════════════════════════════════════════════════════════════════════════
# 24  EMBEDDING STATISTICS PER CLASS
# ═════════════════════════════════════════════════════════════════════════════
print("\n[24] Embedding Statistics per Class")

emb_stats = {}
if EMB_COLS:
    for cls in CLASSES:
        sub = df[df[TARGET_COL] == cls][EMB_COLS]
        emb_stats[cls] = {
            "mean_vector" : [round(float(v), 6) for v in sub.mean().values],
            "std_vector"  : [round(float(v), 6) for v in sub.std().values],
            "mean_l2_norm": round(float(np.linalg.norm(sub.mean().values)), 6),
        }

    emb_centroid_mat = np.array([emb_stats[cls]["mean_vector"] for cls in CLASSES])
    euc_emb_mat      = euclidean_distances(emb_centroid_mat)
    emb_pair_dist    = {}
    for i, ci in enumerate(CLASSES):
        for j, cj in enumerate(CLASSES):
            if i < j:
                emb_pair_dist[f"{ci}_vs_{cj}"] = round(float(euc_emb_mat[i, j]), 4)

    jdump("24_embedding_class_statistics.json", {
        "embedding_columns"          : EMB_COLS,
        "per_class_vectors"          : emb_stats,
        "pairwise_centroid_distances": emb_pair_dist,
        "most_similar_pair"          : min(emb_pair_dist, key=emb_pair_dist.get) if emb_pair_dist else None,
        "most_distant_pair"          : max(emb_pair_dist, key=emb_pair_dist.get) if emb_pair_dist else None,
    })
else:
    jdump("24_embedding_class_statistics.json", {"note": "No embedding columns found."})


# ═════════════════════════════════════════════════════════════════════════════
# 25  PERCENTILE PROFILES PER CLASS (linguistic features)
# ═════════════════════════════════════════════════════════════════════════════
print("\n[25] Percentile Profiles per Class")

pct_profiles = {}
for cls in CLASSES:
    sub = df[df[TARGET_COL] == cls]
    cls_pcts = {}
    for feat in LING_COLS:
        vals = sub[feat].dropna().values
        cls_pcts[feat] = {f"p{p}": round(float(np.percentile(vals, p)), 4) for p in PERCENTILES}
    pct_profiles[cls] = cls_pcts

jdump("25_percentile_profiles_per_class.json", {
    "percentiles_computed": PERCENTILES,
    "features"            : LING_COLS,
    "profiles"            : pct_profiles,
})


# ═════════════════════════════════════════════════════════════════════════════
# 26  FEATURE RANKING CONSENSUS (KW + RF + MI combined)
# ═════════════════════════════════════════════════════════════════════════════
print("\n[26] Feature Ranking Consensus")

kw_rank = rank_dict(list(kw_sorted.keys()))
rf_rank = rank_dict(list(rf_imp.index))
mi_rank = rank_dict(list(mi_series.index))

consensus = []
for feat in FEAT_COLS:
    r_kw = kw_rank.get(feat, len(FEAT_COLS))
    r_rf = rf_rank.get(feat, len(FEAT_COLS))
    r_mi = mi_rank.get(feat, len(FEAT_COLS))
    avg  = (r_kw + r_rf + r_mi) / 3
    consensus.append({
        "feature"   : feat,
        "kw_rank"   : r_kw,
        "rf_rank"   : r_rf,
        "mi_rank"   : r_mi,
        "avg_rank"  : round(avg, 2),
    })
consensus.sort(key=lambda x: x["avg_rank"])

jdump("26_feature_ranking_consensus.json", {
    "description"    : "Consensus rank = average of Kruskal-Wallis, Random Forest MDI, and Mutual Information ranks.",
    "top_10_features": [c["feature"] for c in consensus[:10]],
    "full_ranking"   : consensus,
})


# ═════════════════════════════════════════════════════════════════════════════
# 27  MASTER SUMMARY REPORT
# ═════════════════════════════════════════════════════════════════════════════
print("\n[27] Master Summary Report")

top10_consensus = [c["feature"] for c in consensus[:10]]
top10_kw        = list(kw_sorted.keys())[:10]
top10_rf_list   = list(rf_imp.head(10).index)
top10_mi_list   = list(mi_series.head(10).index)

# Compute class-level summary
class_summary = {}
for cls in CLASSES:
    sub = df[df[TARGET_COL] == cls]
    class_summary[cls] = {
        "n_samples"       : int(len(sub)),
        "pct_of_dataset"  : round(len(sub) / len(df) * 100, 2),
        "top_feature_mean": {feat: round(float(sub[feat].mean()), 4) for feat in top10_consensus[:5]},
    }

summary = {
    "project"       : "Mindspace Voice Agent — Mental Health Classification EDA",
    "dataset_file"  : os.path.basename(DATA_PATH),
    "analysis_date" : str(pd.Timestamp.today().date()),

    "dataset": {
        "rows"             : int(df.shape[0]),
        "cols"             : int(df.shape[1]),
        "classes"          : CLASSES,
        "n_classes"        : N_CLASSES,
        "class_counts"     : dist["counts"],
        "imbalance_ratio"  : dist["imbalance_ratio"],
        "class_entropy_pct": dist["entropy_pct_of_max"],
        "total_null_cells" : int(df.isnull().sum().sum()),
        "duplicate_rows"   : n_dups,
        "memory_mb"        : memory_mb,
    },

    "feature_groups": {
        "n_linguistic" : len(LING_COLS),
        "n_topic"      : len(TOPIC_COLS),
        "n_embedding"  : len(EMB_COLS),
        "n_categorical": len(list(CAT_COLS)),
    },

    "statistical_tests": {
        "kruskal_wallis": {
            "n_features_tested" : len(kw_raw),
            "n_significant_p05" : len(sig_feats),
            "top_10_by_H_stat"  : top10_kw,
        },
        "anova": {
            "n_significant_p05": sum(1 for v in anova_raw.values() if v["significant"]),
        },
        "eta_squared": {
            "n_large_effect"   : sum(1 for v in eta2_raw.values() if v["effect_interpretation"] == "large"),
            "n_medium_effect"  : sum(1 for v in eta2_raw.values() if v["effect_interpretation"] == "medium"),
        },
    },

    "feature_importance": {
        "top_10_kruskal_wallis" : top10_kw,
        "top_10_random_forest"  : top10_rf_list,
        "top_10_mutual_info"    : top10_mi_list,
        "top_10_consensus"      : top10_consensus,
    },

    "correlations": {
        "n_highly_correlated_pairs": len(pairs),
        "threshold"               : CORR_HIGH_THRESHOLD,
    },

    "class_separability": {
        "most_similar_pair_euclidean": min(euc_pairs, key=euc_pairs.get) if euc_pairs else None,
        "most_distant_pair_euclidean": max(euc_pairs, key=euc_pairs.get) if euc_pairs else None,
        "top_discriminative_feature" : list(kw_sorted.keys())[0] if kw_sorted else None,
    },

    "pca_insights": {
        "embedding_top10_components_variance_pct": round(float(cum_emb[-1] * 100), 2) if EMB_COLS else None,
        "feat_pca_components_for_80pct"          : int(np.argmax(cum_feat >= 0.80) + 1) if any(cum_feat >= 0.80) else None,
        "feat_pca_components_for_90pct"          : int(np.argmax(cum_feat >= 0.90) + 1) if any(cum_feat >= 0.90) else None,
    },

    "per_class_summary": class_summary,

    "sections_in_this_report": sorted(EDA_REPORT.keys()),
}
jdump("27_MASTER_SUMMARY", summary)


# ─────────────────────────────────────────────────────────────────────────────
# Write the single combined report
with open(OUTPUT_FILE, "w") as _fh:
    json.dump(EDA_REPORT, _fh, indent=2)

print("\n\n══════════════════════════════════════════════════════════")
print(f"  EDA COMPLETE — {len(EDA_REPORT)} sections written to:")
print(f"  {OUTPUT_FILE}")
print("══════════════════════════════════════════════════════════\n")
