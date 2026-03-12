# ML Pipeline Documentation

## Overview
Automated end-to-end ML pipeline: takes a CSV file, performs EDA, feature engineering, model selection, training, tuning, and saves artifacts. Designed to work on any tabular dataset.

**Dataset used:** `mental_health_synthetic_dataset_with_normal.csv` (50,000 rows, 66 columns)
**Target:** `profile` (7-class classification: Anxiety, Bipolar_Mania, Depression, Normal, Phobia, Stress, Suicidal_Tendency)
**Best model:** LightGBM (F1 macro ~0.88)

---

## Step-by-Step Breakdown

### Step 0 — Import Libraries
**What:** Imports pandas, numpy, sklearn, xgboost, lightgbm, optuna, scipy, matplotlib, seaborn.
**Method:** Standard Python imports.
**How it affects next steps:** Makes all tools available for the pipeline.

---

### Step 1 — Configuration
**What:** Sets `FILE_PATH`, `RANDOM_SEED=42`, `OUTPUT_DIR`. Creates `PIPELINE_STATE` dict to track decisions.
**How it affects next steps:** Every subsequent step reads these config values. The seed ensures reproducibility.

---

### Step 2 — Data Loading
**What:** Loads CSV with `pd.read_csv()`, shows shape, dtypes, memory usage, first 5 rows.
**Output:** 50,000 rows x 66 columns, 32.20 MB. Types: 61 float, 3 object, 2 int.
**How it affects next steps:** The `df` DataFrame flows into all subsequent steps. If loading fails, pipeline stops.

---

### Step 3 — Column Overview
**What:** Prints a table of all columns with dtype, null count, unique values, and sample value.
**Method:** Simple pandas column inspection.
**How it affects next steps:** User reviews this to decide which column to set as target in Step 4.

---

### Step 4 — Target Column Selection
**What:** User sets `TARGET_COLUMN = 'profile'`. Auto-detects task type.
**Method:** If dtype is `object` OR `nunique <= 20` -> classification. Else -> regression.
**Output:** 7 classes detected -> CLASSIFICATION.
**How it affects next steps:** `TASK_TYPE` controls ALL branching decisions for the rest of the pipeline:
- Classification -> stratified splits, Kruskal-Wallis tests, class balance handling, f1_macro scoring, confusion matrices
- Regression -> regular splits, Spearman tests, no balance handling, MSE scoring, scatter plots

---

### Step 5 — Data Profiling & Quality Audit
**What:** Checks data quality across 6 dimensions and flags columns for removal.
**Methods used:**
| Check | Method | Threshold | Result |
|-------|--------|-----------|--------|
| Null values | `isnull().sum()` | Flag if >50% null | No nulls found |
| Duplicates | `duplicated().sum()` | Remove if >30% | No duplicates |
| Constant cols | `nunique() <= 1` | Always remove | None found |
| ID-like cols | All-unique non-float + name patterns | Always remove | None found |
| High cardinality | `nunique/len > 0.5` for object cols | Always remove | None found |
| Target leakage | 1-to-1 mapping with target + binary indicator check | Always remove | **Caught `target` column!** |

**How it affects next steps:** Produces `columns_to_drop` list -> Step 6 drops these columns. If leakage columns were missed here, the model would cheat and give fake-high accuracy.

---

### Step 6 — Auto-Clean
**What:** Drops flagged columns from Step 5, imputes remaining nulls.
**Methods:**
- Numeric nulls -> fill with **median** (robust to outliers)
- Categorical nulls -> fill with **mode** (most frequent value)
- Target nulls -> drop those rows entirely
**Output:** Dropped `target` column (leakage). Shape: 50,000 x 65. No imputation needed.
**How it affects next steps:** Clean `df` with no nulls flows into Step 7. The leakage column is gone.

---

### Step 7 — Target Analysis, Class Balance & Train/Test Split
**What:** Three things happen in this step:

**7a. Class balance analysis:**
| Class | Count | Percentage |
|-------|-------|------------|
| Depression | 11,281 | 22.56% |
| Anxiety | 9,024 | 18.05% |
| Stress | 9,000 | 18.00% |
| Bipolar_Mania | 6,818 | 13.64% |
| Normal | 4,977 | 9.95% |
| Suicidal_Tendency | 4,461 | 8.92% |
| Phobia | 4,439 | 8.88% |

Imbalance ratio: **2.54:1** -> Moderate imbalance -> `class_weight="balanced"` strategy.

**7b. Target encoding:** LabelEncoder maps classes to 0-6.

**7c. Train/test split:** 80/20 stratified split **BEFORE** any transformations.
- Train: 40,000 samples
- Test: 10,000 samples
- Index reset applied for clean alignment.

**How it affects next steps:**
- `BALANCE_STRATEGY` -> controls whether models use class_weight/sample_weight (Steps 13-14)
- `X_train_df`, `X_test_df` -> all subsequent transforms fit on train only
- `y_train`, `y_test` -> used for model training and evaluation
- Splitting here prevents data leakage in Steps 8-12

---

### Step 8 — Outlier Handling (Train-Fit)
**What:** Detects outliers using IQR on **training data only**, tests 4 smoothing strategies, picks the one that minimizes skewness, applies same transform to both train and test.
**Methods tested per column:**
| Strategy | How it works | When applicable |
|----------|-------------|-----------------|
| Winsorize | Clip values to [Q1-1.5*IQR, Q3+1.5*IQR] | Always |
| Log1p | `log(1+x)` transform | Non-negative data only |
| Sqrt | Square root transform | Non-negative data only |
| Yeo-Johnson | Power transform (fitted on train) | Always |

**Selection:** Strategy with lowest `|skewness|` on training data wins.
**Output:** Yeo-Johnson won for most columns. All fitted transformers saved in `outlier_transformers` dict.
**How it affects next steps:**
- Transformed features flow into encoding (Step 9) and all later steps
- Saved transformers enable reproducing the same transform on new data at inference

---

### Step 9 — Feature Type Handling (Train-Fit)
**What:** Encodes categorical features. Fit on training data, applied identically to test.
**Methods by cardinality:**
| Cardinality | Method | Applied to |
|-------------|--------|------------|
| Binary (2 unique) | LabelEncoder (fit on train) | — not triggered — |
| Low (3-10 unique) | One-Hot (`drop_first=False`) | `language` -> 3 columns |
| High (11+ unique) | Frequency encoding (train frequencies) | — not triggered — |

**`drop_first=False`** used because tree models (RF, XGBoost, LightGBM) need full category representation.
**Column alignment:** Missing test categories get 0, extra test-only categories get dropped.
**Output:** 1 categorical column (`language`) -> 3 one-hot columns. Final: 66 features.
**How it affects next steps:** All features are now numeric -> ready for EDA, feature selection, and model training. Encoding artifacts saved for inference.

---

### Step 10 — EDA & Visualization (Training Data Only)
**What:** Statistical analysis of features vs target, using **training data only**.
**Methods:**
| Analysis | Method | Purpose |
|----------|--------|---------|
| Distribution | Skewness + Kurtosis per feature | Identify non-normal features |
| Correlation | Pearson correlation matrix | Find redundant feature pairs |
| Feature-Target test | **Kruskal-Wallis H-test** (classification) | Test if feature distributions differ across classes |
| Variance test | **Levene's test** | Test if feature spread differs across classes |

**Key results:**
- 1 correlated pair: `total_word_count` <-> `unique_word_count` (r=0.885)
- 26/66 features statistically significant (p < 0.05)
- Top features: `self_reference_density`, `overall_sentiment_score`, `future_focus_ratio`

**How it affects next steps:**
- `high_corr_pairs` -> Step 11 removes one from each correlated pair
- `stat_test_results` -> Step 11 uses H-statistics for consensus ranking
- `levene_results` -> Step 11 uses W-statistics as 4th ranking method

---

### Step 11 — Feature Selection (Multi-Method Consensus, Train Only)
**What:** Progressive feature elimination using multiple independent methods.

**Sub-steps:**
| Sub-step | Method | What it removes | Result |
|----------|--------|-----------------|--------|
| 11.1 Correlation filter | Remove one from pairs with \|r\| >= 0.85 (keep the one more related to target) | `unique_word_count` | 65 remain |
| 11.2 RF Importance | Train RF on training data, rank by MDI importance | Top-25% (16 features) **protected** from VIF removal | — |
| 11.3 VIF | Iteratively remove worst VIF > 10 among **unprotected** features | `language_english` (84.0), `adverb_ratio` (29.5), `present_focus_ratio` (13.8) | 62 remain |
| 11.4 Mutual Information | `mutual_info_classif` on training data | Ranking only, no removal | — |
| 11.5 Consensus ranking | Average rank across RF + MI + Kruskal-Wallis + Levene's (4 methods) | Ranks all features | — |
| 11.6 Pruning | Less aggressive of: MI < 0.01 threshold OR bottom 30% consensus | Drops low-importance features | **43 remain** |

**Key design decisions:**
- Top-25% RF importance features are **protected from VIF removal** -> prevents dropping highly predictive features just for multicollinearity
- Pruning uses the **less aggressive** of two methods -> avoids over-pruning
- VIF, RF, MI all computed on **training data only** -> no leakage

**How it affects next steps:**
- 43 selected features -> Step 12 scales these
- Feature list defines what the final model expects as input

---

### Step 12 — Feature Scaling
**What:** Scale features. Scaler fit on training data, applied to both.
**Method selection:**
| Condition | Scaler chosen | Reason |
|-----------|--------------|--------|
| >30% of features had outlier treatment | **RobustScaler** | Uses median/IQR, robust to remaining outliers |
| <=30% outlier treatment | StandardScaler | Zero mean, unit variance |

**Output:** RobustScaler selected (63 outlier columns > 30% of 43 features).
**How it affects next steps:** Scaled `X_train_scaled` and `X_test_scaled` numpy arrays go into model training. Scaler saved for inference.

---

### Step 13 — Model Shortlisting
**What:** Dynamically selects candidate models based on dataset properties.
**Selection logic:**
| Model | Condition | Selected? | Class balance handling |
|-------|-----------|-----------|----------------------|
| Random Forest | Always | Yes | `class_weight='balanced'` |
| LightGBM | Always | Yes | `is_unbalance=True` |
| Extra Trees | Always | Yes | `class_weight='balanced'` |
| XGBoost | n_samples <= 100K | Yes | `sample_weight` in fit() |
| Gradient Boosting | Always | Yes | `sample_weight` in fit() |
| Logistic Regression | n_feats < 100 AND n_samples <= 50K | Yes | `class_weight='balanced'` |
| SVM | n_feats < 50 AND n_samples <= 20K | No (40K > 20K) | — |
| KNN | n_samples <= 50K | Yes | No weighting |

**`compute_sample_weight('balanced')`** computed for XGBoost and Gradient Boosting (which don't accept `class_weight` parameter directly).

**How it affects next steps:**
- 7 candidates go into cross-validation (Step 14)
- `MODELS_NEEDING_SAMPLE_WEIGHT` set tells Step 14 which models need manual CV loop with `sample_weight`

---

### Step 14 — Model Training & Cross-Validation
**What:** 5-fold stratified CV on all 7 candidates, ranked by **f1_macro**.
**Method:** `StratifiedKFold(n_splits=5)` with `scoring='f1_macro'`.
- Models needing sample_weight -> manual CV loop with `model.fit(X, y, sample_weight=sw)`
- Other models -> `cross_val_score()` with `n_jobs=-1`

**Partial results (Gradient Boosting still running in latest execution):**
| Model | F1 Macro | Time |
|-------|----------|------|
| LightGBM | 0.9198 | 22.2s |
| XGBoost [sw] | 0.9194 | 31.4s |
| Random Forest | 0.9087 | 66.5s |
| Extra Trees | 0.8926 | 29.9s |

**How it affects next steps:** `ranked` list -> Step 15 picks top 2 for tuning.

---

### Step 15 — Top-K Model Selection
**What:** Picks top 2 models from ranking. Analyzes score gap.
**Logic:**
| Score gap | Interpretation |
|-----------|---------------|
| < 0.005 | Very close — tuning both is critical |
| 0.005 - 0.02 | Runner-up could close gap |
| > 0.02 | Clear leader |

**How it affects next steps:** Top 2 model names -> Step 16 tunes only these two.

---

### Step 16 — Hyperparameter Tuning (Optuna)
**What:** Bayesian optimization (TPE sampler) for top 2 models.
**Settings:** 15 trials max, 180s timeout per model, 5-fold CV (matching training).
**Hyperparameter spaces:**
| Model | Parameters tuned |
|-------|-----------------|
| LightGBM | n_estimators, max_depth, learning_rate, num_leaves, subsample, colsample_bytree, min_child_samples, reg_alpha, reg_lambda |
| XGBoost | n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_weight, gamma, reg_alpha, reg_lambda |

Models needing sample_weight -> manual CV loop inside Optuna objective.

**How it affects next steps:** Best model name + best parameters -> Step 17 trains final model.

---

### Step 17 — Final Evaluation
**What:** Train best tuned model on full training set, evaluate on held-out test set.
**Metrics reported:** Accuracy, F1 (macro + weighted), Precision, Recall, full classification report, confusion matrix.
**Runner-up** also evaluated for comparison.
**How it affects next steps:** Final metrics + trained model -> Step 18 saves everything.

---

### Step 18 — Save Artifacts
**What:** Saves everything needed for inference to a timestamped folder.
**Artifacts saved:**
| File | What it contains |
|------|-----------------|
| `best_model.joblib` | Trained LightGBM model |
| `scaler.joblib` | Fitted RobustScaler |
| `label_encoder.joblib` | Target class mapping (0-6 -> class names) |
| `outlier_transformers.joblib` | Fitted PowerTransformer objects + strategy per column |
| `encoding_artifacts.joblib` | One-hot categories, frequency maps, label encoders |
| `feature_names.json` | Ordered list of expected feature columns |
| `pipeline_state.json` | Full decision log from every step |
| `model_metadata.json` | Model name, params, metrics, config summary |

---

## Pipeline Flow Diagram

```
CSV File
  |
  v
[Step 2] Load Data  ──>  df (50K x 66)
  |
  v
[Step 4] Set Target  ──>  TASK_TYPE = classification
  |
  v
[Step 5] Quality Audit  ──>  columns_to_drop = ['target']
  |
  v
[Step 6] Auto-Clean  ──>  df (50K x 65, no leakage column)
  |
  v
[Step 7] Balance + SPLIT  ──>  X_train_df (40K x 64) + X_test_df (10K x 64)
  |                              BALANCE_STRATEGY = 'class_weight'
  v
[Step 8] Outlier Handling  ──>  Fit on train, apply to both (yeo-johnson mostly)
  |
  v
[Step 9] Encoding  ──>  Fit on train, apply to both (one-hot: language -> 3 cols)
  |                      66 features
  v
[Step 10] EDA  ──>  stat_test_results, high_corr_pairs, levene_results
  |                  (all from training data)
  v
[Step 11] Feature Selection  ──>  43 features kept (correlation + VIF + consensus)
  |                                 Protected top-25% from VIF
  v
[Step 12] Scaling  ──>  RobustScaler fit on train, apply to both
  |
  v
[Step 13] Model Shortlist  ──>  7 candidates + sample_weight for XGB/GB
  |
  v
[Step 14] 5-Fold CV  ──>  ranked by f1_macro
  |
  v
[Step 15] Top-2  ──>  LightGBM + XGBoost
  |
  v
[Step 16] Optuna Tuning  ──>  best: LightGBM (f1_macro = 0.8833)
  |
  v
[Step 17] Final Eval  ──>  Accuracy: 0.8856, F1 macro: 0.88
  |
  v
[Step 18] Save  ──>  7 artifact files in pipeline_output/LightGBM_DDMMYYYY_HHMMSS/
```

---

## Score: ~85% Well-Done

### What was fixed from the first audit (all good now):
- Train/test split moved BEFORE any transformations (no data leakage)
- Outlier handling fits on training data only, applies to both
- Encoding fits on training data only, applies to both
- EDA and feature selection use training data only
- Index reset after split prevents alignment bugs
- `drop_first=False` for one-hot encoding (better for tree models)
- Top-25% features protected from VIF removal
- Smarter feature pruning (less aggressive of two methods)
- `f1_macro` as scoring metric (equal weight to all classes)
- `sample_weight` computed for XGBoost and Gradient Boosting
- Outlier transformers and encoding artifacts saved for inference
- Feature names saved as JSON

### Remaining issues (15% gap):

**1. Stale execution — Cells 32-38 used old `ranked` variable**
Cell 30 (Step 14) was re-run with `f1_macro` but never finished (Gradient Boosting still training). Yet cells 32-38 have output — they used the **old** `ranked` from a previous run that used `f1_weighted`. This means:
- Step 15 selected top-2 based on old f1_weighted scores
- Step 16 tuned using f1_weighted scores, not f1_macro
- Step 17 reports f1_weighted as primary metric, not f1_macro
- **Fix:** Re-run cells 30 through 38 sequentially after Cell 30 finishes.

**2. Step 18 — Outlier transformers and encoding artifacts NOT actually saved**
The output shows only 5 files saved (model, scaler, encoder, pipeline_state, metadata). The `outlier_transformers.joblib` and `encoding_artifacts.joblib` lines are missing from the output. The `'outlier_transformers' in dir()` check likely failed because the variable was from a different cell execution context.
- **Fix:** Re-run after Cell 30 finishes — the variables should be in scope.

**3. Step 12 — Scaler selection still uses wrong count**
`n_outlier_cols = 63` (from PIPELINE_STATE, counting ALL original columns with outliers) vs `total_features = 43` (current features). 63 > 43*0.3 = 12.9 is true, but the comparison is apples-to-oranges. Should count outlier columns among the 43 current features only.

**4. Step 10 — Regression branch Spearman alignment still buggy**
`target_aligned = y_train[:len(vals)]` takes the first N values by position, not by index. If any training rows had NaN in that column, `vals` has fewer entries and the alignment is wrong. Not triggered for this dataset (classification), but would break on a regression dataset with nulls.

**5. Step 16 — Optuna tuning used 3-fold CV in code, not 5**
The `TUNING_CV_FOLDS = 5` comment says "Consistent with training CV" but looking at the Optuna output, trials completed in ~16s each (LightGBM) which is closer to 3-fold timing than 5-fold. Check if the variable was actually updated in the running notebook.

**6. Minor — `import copy` inside a for loop (Step 14, line 719)**
Importing inside a loop works but is inefficient. Should be at the top with other imports.
