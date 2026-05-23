# ML Pipeline — Improvement To-Do List
**Project:** Mindspace Voice Agent — Text + Voice ML Pipelines  
**Files:** `text-ml-pipeline.ipynb` · `voice-pca-pipeline-guided.ipynb`  
**Last Updated:** 23-May-2026

---

## Pre-Work (Already Done)
- [x] Rename all CSV column headers to `snake_case` (no spaces, no special chars)
- [x] Fix target label typos in CSV:
  - `DEEPRESSION` → `DEPRESSION`
  - `PHOBNIA` → `PHOBIA`
  - `SUCIDAL TENDANCY` → `SUICIDAL_TENDENCY`

---

## Text Pipeline — Items (All Completed)

- [x] Item 1 — Fix FILE_PATH + lock TASK_TYPE
- [x] Item 2 — Fix TARGET_COLUMN default
- [x] Item 3 — Empty COLUMNS_TO_DROP by default
- [x] Item 4 — Markdown documentation for all 19 steps
- [x] Item 5 — Paginated EDA distribution plots (all features)
- [x] Item 6 — Per-class feature mean table + heatmap
- [x] Item 7 — Conservative feature pruning (MI=0 AND p>0.05)
- [x] Item 8 — SHAP Step 19 (global importance, beeswarm, per-class, waterfall)
- [x] Item 9 — Save confusion matrix (raw + normalized) as PNG to output folder

## Voice Pipeline — Items (All Completed)

- [x] Item 1 — Add normalized confusion matrix (side-by-side with raw count plot)
- [x] Item 2 — Save confusion matrix PNG to output folder (Step 16)
- [x] Item 3 — Add SHAP explainability Step 17 (global importance, per-class summary, top-10 grid, waterfall plots — all saved as PNG)
- [x] Item 4 — Unify output folder naming format with text pipeline: `{Model}_{dd-Mon-yyyy}_{HH-MM-SS}`
- [x] Item 5 — Rename existing output folders to new format

---

---

### ITEM 1 — Fix Config Cell (Step 1)
**Cell:** 5 (Step 1 — Configuration)  
**What to change:**
- Update `FILE_PATH` from `data\synthetic_3000_dataset.csv` to `data\text_parameters_for_ml.csv`
- Add `TASK_TYPE = 'classification'` explicitly so it is locked and visible at the top — no ambiguity
- Keep `RANDOM_SEED = 42` and `OUTPUT_DIR` as-is

**Why this matters:**  
The config cell is the single place any user should need to touch. If `FILE_PATH` points to the wrong file, nothing else works. Locking `TASK_TYPE` here makes it obvious this is a classification pipeline.

---

### ITEM 2 — Fix Target Column Setting (Step 4)
**Cell:** 12 (Step 4 — Target Column Selection)  
**What to change:**
- Update `TARGET_COLUMN = 'cluster'` to `TARGET_COLUMN = 'target'`
- Since `TASK_TYPE` is now set in config (Item 1), remove the auto-detect logic here — just validate that the column exists and print its stats
- Keep the validation (`raise ValueError` if column not found) — that is good

**Why this matters:**  
`cluster` does not exist in our CSV. Pipeline crashes on this cell before doing anything useful.

---

### ITEM 3 — Empty `COLUMNS_TO_DROP` by Default (Step 3)
**Cell:** 10 (Step 3 — Column Overview / Delete Unwanted Columns)  
**What to change:**
- Change `COLUMNS_TO_DROP = ['sampling_rate']` to `COLUMNS_TO_DROP = []`
- Keep the existing logic exactly as-is — it already handles empty list gracefully with `[SKIP]` message

**Why this matters:**  
`sampling_rate` does not exist in our CSV. It prints a warning every run. The default should always be empty — the user adds column names here only when needed. The drop mechanism is already production-grade.

---

### ITEM 4 — Add Markdown Documentation to Every Step
**Cells:** All markdown cells (1, 2, 4, 6, 8, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39)  
**What to change:**  
Each markdown cell should explain:
1. **What this step does** — one clear sentence
2. **Why it is done here** — the reasoning / what problem it solves
3. **What the output is** — what variable or artifact gets created
4. **Key rules / constraints** — e.g. "fit only on training data", "never removes rows"

**Format to follow:**
```
## Step N — Step Name
**What:** ...
**Why:** ...
**Output:** ...
**Rules:** ...
```

**Steps that need the most documentation work:**
- Step 8 (Outlier Handling) — explain why we smooth instead of remove
- Step 11 (Feature Selection) — explain the 4-method consensus system
- Step 12 (Scaling) — explain why scaler is fit on train only
- Step 16 (Optuna Tuning) — explain Bayesian search vs grid search

---

### ITEM 5 — Make EDA Plots Paginate Across ALL Features (Step 10)
**Cell:** 24 (Step 10 — EDA & Visualization)  
**What to change:**
- Remove hardcoded `n_plot = min(12, n_feat)` limit
- Replace with a paginated plot that auto-generates pages of 12 features each
- So with 52 features: 5 pages of 12 (last page shows remaining 4)
- Each page titled: `Feature Distributions — Page X of Y (Train Only)`

**Why this matters:**  
With 52 features, 40 are currently invisible. In a mental health classifier every feature matters — a clinician reviewing this needs to see all of them. The pipeline should scale to any number of features without manual changes.

---

### ITEM 6 — Add Per-Class Feature Mean Comparison Table (Step 10)
**Cell:** 24 (Step 10 — EDA & Visualization) — add after existing plots  
**What to add:**
- A table showing mean value of each feature broken down by class (5 classes)
- Format: rows = features, columns = class names, values = mean
- Highlight which class has the highest mean per feature
- Also add a heatmap of this table — rows = features, columns = classes

**Why this matters:**  
This is the most direct way to see if features actually separate the classes. For example: does `fear_word_frequency` have a clearly higher mean for PHOBIA vs ANXIETY? If not, those two classes will be confused by the model. This table gives you that answer before training starts.

---

### ITEM 7 — Smarter Feature Pruning (Step 11)
**Cell:** 26 (Step 11 — Feature Selection)  
**What to change:**
- Replace the current "drop bottom 30% by consensus" rule with a smarter 2-condition gate:
  - Drop a feature only if **both** conditions are true:
    - Mutual Information score = 0 (literally zero predictive signal)
    - Kruskal-Wallis p-value > 0.05 (not statistically significant vs target)
  - If either condition is false → keep the feature
- Print a clear table showing kept vs dropped with MI and p-value for each

**Why this matters:**  
We only have 52 features. Dropping 15+ blindly (30% rule) when the dataset is balanced and well-structured throws away real signal. The new rule is conservative and data-driven — it only removes features that are provably useless by two independent tests.

---

### ITEM 8 — Add SHAP Explainability Step (New Step 19)
**Cell:** Add new cells after Step 17 (Final Evaluation)  
**What to add:**
- A new **Step 19 — Model Explainability (SHAP)** section with:
  1. **Global feature importance** — SHAP bar chart (mean |SHAP| per feature, top 20)
  2. **SHAP summary plot** — beeswarm showing direction of each feature's effect
  3. **Per-class SHAP breakdown** — for each of the 5 classes, which features push the prediction toward that class
  4. **Single prediction explainer** — show SHAP waterfall plot for one sample from each class
- Save SHAP plots to the output folder

**Why this matters:**  
This is a mental health classification system. A clinician or reviewer needs to understand WHY the model predicted SUICIDAL_TENDENCY for a patient — not just that it did. SHAP answers that question at both the global level (overall model behavior) and individual prediction level. Without this, the model is a black box and cannot be trusted in a clinical setting.

---

## Summary Table — Text Pipeline

| # | Item | Status |
|---|------|--------|
| 1 | Fix FILE_PATH + lock TASK_TYPE in config | Done |
| 2 | Fix TARGET_COLUMN default value | Done |
| 3 | Empty COLUMNS_TO_DROP by default | Done |
| 4 | Add markdown documentation to every step | Done |
| 5 | Paginate EDA plots across all features | Done |
| 6 | Per-class feature mean table + heatmap | Done |
| 7 | Smarter feature pruning (MI=0 + p>0.05) | Done |
| 8 | SHAP explainability step (Step 19) | Done |
| 9 | Save confusion matrix PNG to output folder | Done |

## Summary Table — Voice Pipeline

| # | Item | Status |
|---|------|--------|
| 1 | Normalized confusion matrix (side-by-side with raw) | Done |
| 2 | Save confusion matrix PNG to output folder (Step 16) | Done |
| 3 | SHAP explainability Step 17 (4 plot types, all saved as PNG) | Done |
| 4 | Unify output folder naming with text pipeline format | Done |
| 5 | Rename existing output folders to new format | Done |
