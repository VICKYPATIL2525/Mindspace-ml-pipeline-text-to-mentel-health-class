"""
Comprehensive EDA — Mental Health Classification Dataset
=========================================================
Run:  python eda.py
All charts  → eda/output/charts/
All stats   → eda/output/stats/
"""

import os, json, warnings
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use("Agg")                      # non-interactive backend — no window pop-ups
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams.update({"figure.dpi": 110, "font.size": 9})

# ── Paths ──────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
DATA   = os.path.join(BASE, "..", "data")
OUT    = os.path.join(BASE, "output")
CHARTS = os.path.join(OUT, "charts")
STATS  = os.path.join(OUT, "stats")
for d in [CHARTS, STATS]:
    os.makedirs(d, exist_ok=True)

# ── Load data (prefer balanced dataset if available) ───────────────────────
bal_path = os.path.join(DATA, "mental_health_balanced_dataset.csv")
raw_path = os.path.join(DATA, "mental_health_synthetic_dataset_10k.csv")
if os.path.exists(bal_path):
    df = pd.read_csv(bal_path)
    print(f"[INFO] Loaded balanced dataset: {df.shape}")
else:
    df = pd.read_csv(raw_path)
    print(f"[INFO] Loaded raw dataset: {df.shape}")

# ── Column groups ──────────────────────────────────────────────────────────
LING_COLS = [
    "total_word_count", "unique_word_count", "ttr",
    "positive_emotion_ratio", "negative_emotion_ratio",
    "fear_word_frequency", "sadness_word_frequency", "anger_word_frequency",
    "uncertainty_word_frequency", "filler_word_frequency", "repetition_rate",
    "rumination_phrase_frequency", "semantic_coherence_score",
    "language_model_perplexity", "overall_sentiment_score",
    "past_focus_ratio", "present_focus_ratio", "future_focus_ratio",
    "noun_ratio", "verb_ratio", "adjective_ratio", "adverb_ratio",
    "avg_sentence_length", "parse_tree_depth",
    "topic_shift_frequency", "self_reference_density",
]
TOPIC_COLS  = [c for c in df.columns if c.startswith("topic_")]
EMB_COLS    = [c for c in df.columns if c.startswith("emb_")]
TARGET_COL  = "target"
CLASSES     = sorted(df[TARGET_COL].unique())
PALETTE     = dict(zip(CLASSES, sns.color_palette("Set2", len(CLASSES))))


def save_fig(name):
    path = os.path.join(CHARTS, name)
    plt.savefig(path, bbox_inches="tight")
    plt.close("all")
    print(f"  [chart] {name}")


def save_json(obj, name):
    path = os.path.join(STATS, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"  [stats] {name}")


def save_csv(df_out, name):
    path = os.path.join(STATS, name)
    df_out.to_csv(path)
    print(f"  [stats] {name}")


# ══════════════════════════════════════════════════════════════════════════
# 01. Overview — shape, nulls, dtypes
# ══════════════════════════════════════════════════════════════════════════
print("\n[01] Overview")
overview = {
    "rows": int(df.shape[0]),
    "cols": int(df.shape[1]),
    "target_col": TARGET_COL,
    "classes": CLASSES,
    "numeric_cols": len(df.select_dtypes("number").columns),
    "categorical_cols": len(df.select_dtypes("object").columns),
    "null_counts": df.isnull().sum()[df.isnull().sum() > 0].to_dict(),
    "col_groups": {
        "linguistic": LING_COLS,
        "topic": TOPIC_COLS,
        "embedding": EMB_COLS,
    },
}
save_json(overview, "01_overview.json")

# numeric summary
num_summary = df[LING_COLS + TOPIC_COLS].describe().T.round(4)
save_csv(num_summary, "01_numeric_summary.csv")


# ══════════════════════════════════════════════════════════════════════════
# 02. Class Distribution
# ══════════════════════════════════════════════════════════════════════════
print("\n[02] Class Distribution")
counts  = df[TARGET_COL].value_counts().sort_values(ascending=False)
proportions = (counts / counts.sum() * 100).round(2)

dist_info = {
    "counts":         counts.to_dict(),
    "proportions_%":  proportions.to_dict(),
    "imbalance_ratio": round(float(counts.max() / counts.min()), 4),
}
save_json(dist_info, "02_class_distribution.json")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
colors = [PALETTE[c] for c in counts.index]
bars = axes[0].bar(counts.index, counts.values, color=colors)
axes[0].set_title("Class Counts")
axes[0].tick_params(axis="x", rotation=20)
for bar, v in zip(bars, counts.values):
    axes[0].text(bar.get_x() + bar.get_width() / 2, v + 50, str(v), ha="center", fontsize=8)
axes[1].pie(counts.values, labels=counts.index, autopct="%1.1f%%", colors=colors, startangle=140)
axes[1].set_title("Class Proportions")
plt.suptitle("Class Distribution", fontsize=12)
plt.tight_layout()
save_fig("02_class_distribution.png")


# ══════════════════════════════════════════════════════════════════════════
# 03. Per-Class Feature Statistics (mean & median)
# ══════════════════════════════════════════════════════════════════════════
print("\n[03] Per-Class Feature Stats")
means   = df.groupby(TARGET_COL)[LING_COLS].mean().round(4)
medians = df.groupby(TARGET_COL)[LING_COLS].median().round(4)
save_csv(means,   "03_feature_means_per_class.csv")
save_csv(medians, "03_feature_medians_per_class.csv")

# Z-scored median heatmap
medians_z = (medians - medians.mean()) / (medians.std() + 1e-9)
fig, ax = plt.subplots(figsize=(18, 5))
sns.heatmap(medians_z, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
            linewidths=0.4, annot_kws={"size": 7}, ax=ax)
ax.set_title("Z-scored Median Feature Values per Class", fontsize=12)
ax.set_ylabel("")
plt.tight_layout()
save_fig("03_feature_median_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════
# 04. KDE Distributions — top linguistic features per class
# ══════════════════════════════════════════════════════════════════════════
print("\n[04] KDE Distributions")
KEY_FEATS = [
    "sadness_word_frequency", "fear_word_frequency", "anger_word_frequency",
    "negative_emotion_ratio", "positive_emotion_ratio", "overall_sentiment_score",
    "future_focus_ratio", "self_reference_density",
    "rumination_phrase_frequency", "semantic_coherence_score",
    "topic_shift_frequency", "language_model_perplexity",
]
fig, axes = plt.subplots(4, 3, figsize=(16, 16))
axes = axes.flatten()
for ax, feat in zip(axes, KEY_FEATS):
    for cls in CLASSES:
        subset = df.loc[df[TARGET_COL] == cls, feat].dropna()
        subset.plot.kde(ax=ax, label=cls, color=PALETTE[cls], linewidth=1.6)
    ax.set_title(feat.replace("_", " ").title(), fontsize=9)
    ax.set_xlabel("")
    ax.legend(fontsize=6, ncol=2)
    ax.set_yticks([])
plt.suptitle("KDE Feature Distributions by Class", fontsize=13, y=1.01)
plt.tight_layout()
save_fig("04_kde_distributions.png")


# ══════════════════════════════════════════════════════════════════════════
# 05. Box Plots — key features by class
# ══════════════════════════════════════════════════════════════════════════
print("\n[05] Box Plots")
BOX_FEATS = [
    "overall_sentiment_score", "negative_emotion_ratio",
    "sadness_word_frequency", "fear_word_frequency",
    "self_reference_density", "rumination_phrase_frequency",
    "semantic_coherence_score", "topic_shift_frequency",
]
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()
for ax, feat in zip(axes, BOX_FEATS):
    data_by_class = [df.loc[df[TARGET_COL] == cls, feat].dropna().values for cls in CLASSES]
    bp = ax.boxplot(data_by_class, patch_artist=True, notch=False,
                    medianprops={"color": "black", "linewidth": 1.5})
    for patch, cls in zip(bp["boxes"], CLASSES):
        patch.set_facecolor(PALETTE[cls])
    ax.set_xticklabels(CLASSES, rotation=25, fontsize=7)
    ax.set_title(feat.replace("_", " ").title(), fontsize=9)
plt.suptitle("Box Plots: Key Features by Class", fontsize=13)
plt.tight_layout()
save_fig("05_boxplots_by_class.png")


# ══════════════════════════════════════════════════════════════════════════
# 06. Correlation Matrix — linguistic features
# ══════════════════════════════════════════════════════════════════════════
print("\n[06] Correlation Matrix")
corr = df[LING_COLS].corr().round(3)
save_csv(corr, "06_correlation_matrix.csv")

fig, ax = plt.subplots(figsize=(16, 13))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.3, annot_kws={"size": 6.5}, vmin=-1, vmax=1, ax=ax)
ax.set_title("Linguistic Feature Correlation Matrix", fontsize=12)
plt.tight_layout()
save_fig("06_correlation_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════
# 07. Kruskal-Wallis Statistical Tests (each feature vs target)
# ══════════════════════════════════════════════════════════════════════════
print("\n[07] Kruskal-Wallis Tests")
kw_results = {}
all_feat_cols = LING_COLS + TOPIC_COLS
for feat in all_feat_cols:
    groups = [df.loc[df[TARGET_COL] == cls, feat].dropna().values for cls in CLASSES]
    if all(len(g) > 1 for g in groups):
        stat, pval = sp_stats.kruskal(*groups)
        kw_results[feat] = {"H_statistic": round(float(stat), 4), "p_value": float(pval)}

kw_df = pd.DataFrame(kw_results).T.sort_values("H_statistic", ascending=False)
save_csv(kw_df, "07_kruskal_wallis_tests.csv")
save_json(kw_results, "07_kruskal_wallis_tests.json")

# Plot top-20 most discriminative features by H-statistic
top20 = kw_df.head(20)
fig, ax = plt.subplots(figsize=(12, 6))
colors_bar = ["tomato" if p < 0.05 else "steelblue" for p in top20["p_value"]]
bars = ax.barh(top20.index[::-1], top20["H_statistic"].values[::-1], color=colors_bar[::-1])
ax.set_xlabel("Kruskal-Wallis H Statistic")
ax.set_title("Top 20 Most Discriminative Features (KW H-Statistic)\nRed = statistically significant (p < 0.05)", fontsize=11)
plt.tight_layout()
save_fig("07_kruskal_wallis_top20.png")

# Significance summary
sig_feats   = [f for f, r in kw_results.items() if r["p_value"] < 0.05]
insig_feats = [f for f, r in kw_results.items() if r["p_value"] >= 0.05]
sig_summary = {
    "significant_features_p<0.05":     sig_feats,
    "non_significant_features":         insig_feats,
    "count_significant":                len(sig_feats),
    "count_non_significant":            len(insig_feats),
}
save_json(sig_summary, "07_significance_summary.json")


# ══════════════════════════════════════════════════════════════════════════
# 08. Random Forest Feature Importance
# ══════════════════════════════════════════════════════════════════════════
print("\n[08] Random Forest Feature Importance")
le      = LabelEncoder()
y       = le.fit_transform(df[TARGET_COL])
feat_cols = LING_COLS + TOPIC_COLS      # interpretable features only
X       = df[feat_cols].fillna(df[feat_cols].median())

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    n_jobs=-1,
    random_state=42,
    class_weight="balanced",
)
rf.fit(X, y)
importances = pd.Series(rf.feature_importances_, index=feat_cols).sort_values(ascending=False)
save_csv(importances.to_frame("importance"), "08_rf_feature_importance.csv")
save_json(importances.round(6).to_dict(), "08_rf_feature_importance.json")

fig, ax = plt.subplots(figsize=(12, 7))
colors_imp = sns.color_palette("viridis", len(importances))
ax.barh(importances.index[::-1], importances.values[::-1], color=colors_imp[::-1])
ax.set_xlabel("Feature Importance")
ax.set_title("Random Forest Feature Importance\n(predicting mental health class)", fontsize=11)
plt.tight_layout()
save_fig("08_rf_feature_importance.png")

# Top 10 summary
top10_imp = importances.head(10).round(4).to_dict()
save_json(top10_imp, "08_top10_important_features.json")
print(f"  Top 10 features: {list(top10_imp.keys())}")


# ══════════════════════════════════════════════════════════════════════════
# 09. PCA on Embedding Features — class separation
# ══════════════════════════════════════════════════════════════════════════
print("\n[09] PCA on Embeddings")
X_emb = df[EMB_COLS].fillna(0).values

# Sample 5000 rows for speed
rng = np.random.default_rng(42)
idx = rng.choice(len(X_emb), size=min(5000, len(X_emb)), replace=False)
X_s = X_emb[idx]
y_s = df[TARGET_COL].values[idx]

pca   = PCA(n_components=2, random_state=42)
X_2d  = pca.fit_transform(X_s)
exp_var = pca.explained_variance_ratio_

pca_info = {
    "explained_variance_ratio": [round(float(v), 4) for v in exp_var],
    "total_variance_explained_pct": round(float(exp_var.sum() * 100), 2),
}
save_json(pca_info, "09_pca_info.json")

fig, ax = plt.subplots(figsize=(10, 7))
for cls in CLASSES:
    mask_c = y_s == cls
    ax.scatter(X_2d[mask_c, 0], X_2d[mask_c, 1],
               label=cls, color=PALETTE[cls], alpha=0.5, s=12)
ax.set_xlabel(f"PC1 ({exp_var[0]*100:.1f}% var)")
ax.set_ylabel(f"PC2 ({exp_var[1]*100:.1f}% var)")
ax.set_title(f"PCA on Embedding Features (n=5,000 sample)\nTotal variance explained: {exp_var.sum()*100:.1f}%", fontsize=11)
ax.legend(fontsize=8, markerscale=2)
plt.tight_layout()
save_fig("09_pca_embeddings.png")


# ══════════════════════════════════════════════════════════════════════════
# 10. Topic Feature Analysis per Class
# ══════════════════════════════════════════════════════════════════════════
print("\n[10] Topic Features per Class")
topic_means = df.groupby(TARGET_COL)[TOPIC_COLS].mean().round(4)
save_csv(topic_means, "10_topic_means_per_class.csv")

fig, ax = plt.subplots(figsize=(10, 4))
topic_z = (topic_means - topic_means.mean()) / (topic_means.std() + 1e-9)
sns.heatmap(topic_z, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
            linewidths=0.5, annot_kws={"size": 9}, ax=ax)
ax.set_title("Z-scored Mean Topic Features per Class", fontsize=11)
plt.tight_layout()
save_fig("10_topic_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════
# 11. Language Distribution per Class
# ══════════════════════════════════════════════════════════════════════════
print("\n[11] Language Distribution")
lang_dist = df.groupby([TARGET_COL, "language"]).size().unstack(fill_value=0)
save_csv(lang_dist, "11_language_distribution.csv")
save_json(lang_dist.to_dict(), "11_language_distribution.json")

fig, ax = plt.subplots(figsize=(11, 5))
lang_dist.plot(kind="bar", ax=ax, colormap="Set3", edgecolor="white")
ax.set_title("Language Distribution per Class")
ax.set_xlabel("Class")
ax.set_ylabel("Count")
ax.tick_params(axis="x", rotation=20)
ax.legend(title="Language", fontsize=8)
plt.tight_layout()
save_fig("11_language_distribution.png")


# ══════════════════════════════════════════════════════════════════════════
# 12. Pairwise Feature Scatter — top 4 features coloured by class
# ══════════════════════════════════════════════════════════════════════════
print("\n[12] Pairplot — Top 4 Features")
top4 = list(importances.head(4).index)
sample_df = df[top4 + [TARGET_COL]].sample(n=min(3000, len(df)), random_state=42)

palette_list = [PALETTE[c] for c in CLASSES]
pair_palette  = dict(zip(CLASSES, palette_list))
pg = sns.pairplot(sample_df, hue=TARGET_COL, vars=top4,
                  palette=pair_palette, plot_kws={"alpha": 0.35, "s": 12},
                  diag_kind="kde")
pg.figure.suptitle(f"Pairplot — Top 4 Features by RF Importance", y=1.01, fontsize=11)
pg.figure.savefig(os.path.join(CHARTS, "12_pairplot_top4_features.png"),
                  bbox_inches="tight", dpi=110)
plt.close("all")
print("  [chart] 12_pairplot_top4_features.png")


# ══════════════════════════════════════════════════════════════════════════
# 13. Final Summary Report
# ══════════════════════════════════════════════════════════════════════════
print("\n[13] Summary Report")
summary = {
    "dataset": {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "classes": CLASSES,
        "imbalance_ratio": dist_info["imbalance_ratio"],
    },
    "top_10_features_by_rf_importance": top10_imp,
    "top_10_discriminative_features_kw": kw_df.head(10)["H_statistic"].round(2).to_dict(),
    "statistically_significant_features_count": len(sig_feats),
    "pca_embedding_variance_explained_pct": pca_info["total_variance_explained_pct"],
    "output_files": {
        "charts": sorted(os.listdir(CHARTS)),
        "stats":  sorted(os.listdir(STATS)),
    }
}
save_json(summary, "00_SUMMARY.json")

print("\n" + "=" * 60)
print("  EDA COMPLETE")
print(f"  Charts saved to : {CHARTS}")
print(f"  Stats  saved to : {STATS}")
print(f"  Total charts    : {len(os.listdir(CHARTS))}")
print(f"  Total stat files: {len(os.listdir(STATS))}")
print("=" * 60)
