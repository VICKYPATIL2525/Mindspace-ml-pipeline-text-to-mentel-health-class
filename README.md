# Mindspace — Mental Health Profiling via NLP & Voice Features

> **Dual-pipeline ML system that screens for mental health profiles from text/linguistic features and voice/acoustic PCA features — achieving 99.15% accuracy (text) and 99.50% accuracy (voice) across 6 classes each.**

> ⚠️ **Disclaimer:** This project is a **screening / research tool for educational purposes only**. It is **not** a diagnostic tool and does **not** provide a clinical diagnosis. A positive screen is not a diagnosis — always consult a qualified mental-health professional.

---

## Table of Contents

- [Project Summary](#project-summary)
- [Datasets](#datasets)
- [Pipeline Architecture](#pipeline-architecture)
- [Algorithms & Models](#algorithms--models)
- [Key Results](#key-results)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [How to Run](#how-to-run)
- [Saved Artifacts](#saved-artifacts)
- [Tech Stack](#tech-stack)
- [Roadmap](#roadmap)

-----

## Project Summary

Mindspace is a mental health **screening** system with two independent, fully automated ML pipelines:

| Pipeline | Notebook | Model | Accuracy | Input |
|----------|----------|-------|----------|-------|
| **Text** | `text-ml-pipeline.ipynb` | Extra Trees | 99.15% | 20 linguistic/semantic features |
| **Voice** | `voice-pca-pipeline-guided.ipynb` | LightGBM | 99.50% | 23 PCA components from acoustic features |

Both pipelines are **fully automated and adaptive**: they handle EDA, feature engineering, model selection, hyperparameter tuning, and artifact saving without manual intervention.

### What Each Pipeline Does

**Text Pipeline**
- Screens for mental health profiles across **6 categories** based on linguistic, emotional, and semantic features extracted from speech/text
- Handles outliers, nulls, duplicates, leakage, encoding, feature selection, and scaling automatically
- Prevents data leakage — train/test split happens *before* any transformation
- Trains and compares up to 8 ML algorithms via 5-fold CV, tunes top 2 with Optuna

**Voice Pipeline**
- Screens for mental health profiles across **6 categories** based on PCA-reduced acoustic features
- Input: principal components extracted from OpenSMILE acoustic features
- Same anti-leakage pipeline design as the text pipeline
- Includes SHAP explainability (Step 17) and saves confusion matrix + SHAP plots to output folder

---

## Datasets

> **Note:** The datasets are not included in this repository (the `data/` folder is git-ignored — the CSVs exceed GitHub's file-size limits). Supply your own CSVs with the columns described below.

### Text Dataset

| Property | Value |
|----------|-------|
| **File** | `data/TEXT.csv` |
| **Columns** | 53 (52 linguistic/semantic features + target) |
| **Target** | `mental_health_label` (6 classes) |
| **Task** | Multi-class classification |
| **Train / Test Split** | 8,001 / 2,001 (80/20, stratified) |

**Text Target Classes**

| Class | Description |
|-------|-------------|
| ANXIETY | Anxiety-related speech patterns |
| BIPOLAR | Bipolar / manic episode indicators |
| DEPRESSION | Depressive speech markers |
| NORMAL | Baseline / healthy patterns |
| STRESS | Stress-related speech patterns |
| SUICIDAL | Suicidal ideation markers |

**Text Feature Categories**

- **Emotion ratios** — positive, negative, fear, sadness, anger, uncertainty word frequencies
- **Linguistic features** — word count, unique word count, TTR, avg sentence length, parse tree depth, POS ratios
- **Semantic features** — coherence score, sentiment score, self-reference density, rumination phrase frequency
- **Topic distributions** — 5 topic weights (`topic_0`–`topic_4`) + topic shift frequency
- **Embeddings** — 32-dimensional sentence embeddings (`emb_0`–`emb_31`)
- **Temporal focus** — past, present, future focus ratios
- **Paralinguistic** — language model perplexity, filler word frequency, repetition rate
- **Language** — multilingual indicator (English / Hindi / Marathi)

### Voice Dataset

| Property | Value |
|----------|-------|
| **File** | `data/features_pca.csv` |
| **Columns** | 25 (24 PCA features `PC1`–`PC24` + label) |
| **Target** | `label` (6 classes) |
| **Task** | Multi-class classification |
| **Train / Test Split** | 80/20, stratified |

**Voice Target Classes**

| Class | Description |
|-------|-------------|
| Anxiety | Anxiety disorder voice patterns |
| Bipolar | Bipolar / manic episode indicators |
| Depression | Depressive speech markers |
| Normal | Baseline / healthy voice patterns |
| Stress | Stress-related voice patterns |
| Suicidal | Suicidal ideation voice markers |

**Voice Features**: PCA components (`PC1`–`PC24`) derived from OpenSMILE acoustic features including MFCC coefficients, spectral features (entropy, rolloff, harmonicity, flux), pitch (F0), shimmer, jitter, voicing, RMS energy, zero-crossing rate, and HNR. Feature selection keeps 23 of the 24 components (drops `PC17`).

---

## Pipeline Architecture

> **Visual diagrams + step-by-step explanations** of each pipeline are in the `project flow diagrams/` folder:
> - Text pipeline → [`PIPELINE_FLOW_CLEAN.md`](project%20flow%20diagrams/PIPELINE_FLOW_CLEAN.md)
> - Voice pipeline → [`VOICE_PIPELINE_FLOW.md`](project%20flow%20diagrams/VOICE_PIPELINE_FLOW.md)

### Text Pipeline — Steps 0–19

Full diagrams + per-step "what & why" are in [`project flow diagrams/PIPELINE_FLOW_CLEAN.md`](project%20flow%20diagrams/PIPELINE_FLOW_CLEAN.md).

| Step | Stage | What Happens |
|------|-------|-------------|
| **0** | Import & Hardware Detection | Load libraries; detect GPU (CUDA) and CPU core count |
| **1** | Configuration | Set `FILE_PATH`, `TASK_TYPE`, random seed (42), output directory |
| **2** | Data Loading | Load CSV into `df` (keep untouched `df_raw`); preview shape, dtypes, head |
| **3** | Column Overview & Optional Drop | Per-column summary; drop columns in `COLUMNS_TO_DROP` (empty by default) |
| **4** | Target Selection | Set `TARGET_COLUMN = 'mental_health_label'` and validate it exists |
| **5** | Data Profiling | Detect nulls, duplicates, constants, ID-like columns, leakage (diagnostic) |
| **6** | Auto-Clean | Drop flagged columns, impute remaining nulls, drop duplicate rows |
| **7** | Target Analysis & Split | Analyze class balance → **train/test split (80/20, stratified)** before any transformation |
| **8** | Outlier Handling | Test 4 smoothing strategies per column (winsorize, log1p, sqrt, yeo-johnson); pick lowest skew. Fit on train only. |
| **9** | Feature Type Handling | Binary → Label Encoding; Low-cardinality → One-Hot; High-cardinality → Frequency Encoding. Fit on train only. |
| **10** | EDA & Visualization | Distribution plots, correlation heatmaps, Kruskal-Wallis H tests, Levene's W tests — training data only |
| **11** | Feature Selection | Correlation filter → VIF → RF importance + MI + stat tests consensus → conservative pruning. **52 → 20 features.** |
| **12** | Scaling | RobustScaler fit on train, transform both |
| **13** | Model Shortlisting | Dynamically select models based on dataset size and dimensionality |
| **14** | Training & CV | 5-fold stratified CV, scored by `f1_macro` |
| **15** | Top-K Selection | Pick top 2 models for tuning |
| **16** | Hyperparameter Tuning | Optuna TPE sampler, 5-fold CV per trial |
| **17** | Final Evaluation | Full test-set metrics + raw & normalized confusion matrix → **Extra Trees, 99.15% accuracy** |
| **18** | Save Artifacts | Model, scaler, encoders, transformers, feature names, metadata |
| **19** | Model Explainability (SHAP) | Global importance, per-class summary, per-class top-10, waterfall plots — all saved as PNG |

### Voice Pipeline — Steps 0–17

Full diagrams + per-step "what & why" are in [`project flow diagrams/VOICE_PIPELINE_FLOW.md`](project%20flow%20diagrams/VOICE_PIPELINE_FLOW.md).

| Step | Stage | What Happens |
|------|-------|-------------|
| **0** | Imports & Hardware Detection | Load libraries; detect GPU (CUDA) and CPU core count |
| **1** | Configuration | Build `CONFIG` (path, seed, output directory) |
| **2** | Data Loading | Load PCA CSV; preview shape, dtypes, head |
| **3** | Column Overview | Per-column summary of `PC1`–`PC24` + label |
| **3.5** | Manual Setup | Optional column drops + set `TARGET_COLUMN = 'label'` |
| **4** | Data Profiling | Detect nulls, duplicates, dtype issues (diagnostic) |
| **5** | Target Validation | Confirm label; analyze class balance (6 classes) |
| **6** | Auto-Clean | Drop duplicate rows / flagged columns |
| **7** | Target Analysis & Split | Stratified **train/test split (80/20)** before any transformation |
| **8** | Outlier Smoothing | Per-column lowest-skew smoothing. Fit on train only. |
| **9** | Feature Selection | MI + RF consensus → drop `PC17`. **24 → 23 features.** Fit on train only. |
| **10** | Scaling | RobustScaler fit on train, transform both *(note: selection comes before scaling here)* |
| **11** | Model Shortlisting | Dynamically select candidate models for the data/hardware |
| **12** | Training & CV | 5-fold stratified CV, scored by `f1_macro` |
| **13** | Top-K Selection | Pick top 2 models for tuning |
| **14** | Hyperparameter Tuning | Optuna TPE sampler, 5-fold CV per trial |
| **15** | Final Evaluation | Full test-set metrics + raw & normalized confusion matrix → **LightGBM, 99.50% accuracy** |
| **16** | Save Artifacts | Model, scaler, encoder, transformers, feature names, metadata |
| **17** | Model Explainability (SHAP) | Global importance, per-class summary, top-10 grid, waterfall plots — all saved as PNG |

### Anti-Leakage Design

In **both** pipelines every transformation that learns from data (outlier handling, encoding, scaling, feature selection) is **fit exclusively on training data** and applied identically to the test set. The train/test split (text Step 7 / voice Step 7) is a hard boundary — no test data information flows backward.

---

## Algorithms & Models

### Candidate Models

| Model | Type | GPU Support |
|-------|------|-------------|
| **Random Forest** | Ensemble (Bagging) | CPU (`n_jobs=-1`) |
| **LightGBM** | Gradient Boosting | `device='gpu'` when CUDA available |
| **Extra Trees** | Ensemble (Bagging) | CPU (`n_jobs=-1`) |
| **XGBoost** | Gradient Boosting | `device='cuda'` when available |
| **HistGradientBoosting** | Histogram-based GB | CPU (native multi-core) |
| **Logistic Regression** | Linear | CPU |
| **SVM (RBF)** | Kernel | CPU |
| **KNN** | Instance-based | CPU |

### Hyperparameter Tuning

- **Optimizer**: Optuna with TPE (Tree-structured Parzen Estimator) sampler
- **Trials**: 15 per model
- **Timeout**: 180 seconds per model
- **CV**: 5-fold stratified
- **Scoring**: `f1_macro`

---

## Key Results

### Text Model — Extra Trees

| Metric | Score |
|--------|-------|
| **Accuracy** | 0.9915 |
| **F1 (macro)** | 0.9915 |
| **F1 (weighted)** | 0.9915 |
| **Precision (macro)** | 0.9915 |
| **Recall (macro)** | 0.9915 |

Training: 8,001 samples | Test: 2,001 samples | Features: 20 (selected from 52)

**Tuned Hyperparameters**

| Parameter | Value |
|-----------|-------|
| `n_estimators` | 113 |
| `max_depth` | 28 |
| `min_samples_split` | 6 |
| `min_samples_leaf` | 7 |
| `max_features` | None |

**Top predictors** (from saved feature set): `overall_sentiment_score`, `self_reference_density`, `fear_word_frequency`, `emotional_volatility_score`, `catastrophizing_indicators`, `negative_emotion_spike_count`

### Voice Model — LightGBM

| Metric | Score |
|--------|-------|
| **Accuracy** | 0.9950 |
| **Balanced Accuracy** | 0.9950 |
| **F1 (macro)** | 0.9950 |
| **F1 (weighted)** | 0.9950 |
| **Precision (macro)** | 0.9950 |
| **Recall (macro)** | 0.9950 |

Test: 1,985 samples | Features: 23 PCA components (`PC1`–`PC24`, excluding `PC17`)

**Tuned Hyperparameters**

| Parameter | Value |
|-----------|-------|
| `n_estimators` | 362 |
| `max_depth` | 15 |
| `learning_rate` | 0.100 |
| `num_leaves` | 82 |
| `subsample` | 0.578 |
| `colsample_bytree` | 0.578 |
| `min_child_samples` | 62 |
| `reg_lambda` | 0.625 |

Runner-up: Extra Trees (F1 macro 0.9925).

---

## Project Structure

```
Mindspace-ml-pipeline/
├── text-ml-pipeline.ipynb               # Text ML pipeline (19 steps)
├── voice-pca-pipeline-guided.ipynb      # Voice/PCA ML pipeline (17 steps)
├── requirements.txt                     # Python dependencies
├── README.md                            # This file
│
├── data/                                # Datasets (git-ignored — not in repo)
│   ├── TEXT.csv                         # Text dataset (53 cols: 52 features + label)
│   └── features_pca.csv                 # Voice PCA dataset (25 cols: PC1–PC24 + label)
│
├── demo-api-input-data-sample/
│   ├── voice_depression_sample.json     # Sample voice request (PCA features)
│   └── voice_stress_sample.json         # Sample voice request (PCA features)
│
├── project flow diagrams/               # Visual documentation
│   ├── PIPELINE_FLOW_CLEAN.md           # Text pipeline — diagrams + step-by-step guide
│   └── VOICE_PIPELINE_FLOW.md           # Voice pipeline — diagrams + step-by-step guide
│
├── text_ml_pipeline_output/
│   ├── readme.md
│   └── Extra_Trees_18-May-2026_12-11-08/   # Text model artifacts (latest run)
│       ├── best_model.joblib                # Trained Extra Trees classifier
│       ├── scaler.joblib                    # RobustScaler (fit on train samples)
│       ├── label_encoder.joblib             # Target label encoder
│       ├── encoding_artifacts.joblib        # Categorical encoding maps
│       ├── outlier_transformers.joblib      # Per-column outlier smoothing transforms
│       ├── feature_names.json               # 20 selected feature names
│       ├── model_metadata.json              # Metrics, params, class names
│       ├── pipeline_state.json              # Full pipeline execution state
│       └── shap_*.png                       # SHAP global / per-class / waterfall plots
│
├── voice_ml_pipeline_output/
│   ├── readme.md
│   └── LightGBM_23-May-2026_14-03-40/      # Voice model artifacts (latest run)
│       ├── best_model.joblib                # Trained LightGBM classifier
│       ├── scaler.joblib                    # RobustScaler (fit on train)
│       ├── label_encoder.joblib             # Target label encoder
│       ├── encoding_artifacts.joblib        # Categorical encoding maps
│       ├── outlier_transformers.joblib      # Per-column outlier smoothing transforms
│       ├── feature_names.json               # 23 selected PCA feature names
│       ├── model_metadata.json              # Metrics, params, class names
│       └── pipeline_state.json              # Full pipeline execution state
│
└── myenv/                               # Python virtual environment (git-ignored)
```

---

## Installation & Setup

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA drivers (optional, for GPU acceleration)

### Install Dependencies

```bash
# Clone the repository
git clone <repo-url>
cd Mindspace-ml-pipeline-text-to-mentel-health-class

# Create virtual environment
python -m venv myenv

# Activate (Windows)
myenv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### Torch Installation Fix (Copy-Paste)

If `pip install -r requirements.txt` fails around `torch`, use this exact sequence:

```bash
# 1) Make sure no notebook/kernel is using the environment
# 2) Upgrade pip in the same venv
python -m pip install --upgrade pip

# 3) Install torch first
pip install torch==2.7.1

# 4) Install the rest
pip install -r requirements.txt
```

If you see a Windows file-lock error (WinError 32), close VS Code terminals/kernels using `myenv`, then run:

```bash
pip install -r requirements.txt
```

### Optional: GPU Support

```bash
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
```

When a CUDA-capable GPU is detected, XGBoost uses `device='cuda'` and LightGBM uses `device='gpu'` automatically.

---

## How to Run

### Text Pipeline

1. Open `text-ml-pipeline.ipynb` in Jupyter or VS Code
2. Set kernel to the `myenv` virtual environment
3. Run all cells sequentially

### Voice Pipeline

1. Open `voice-pca-pipeline-guided.ipynb` in Jupyter or VS Code
2. Set kernel to the `myenv` virtual environment
3. Run all cells sequentially

> The notebook expects the dataset at the path set in the **Step 1 — Configuration** cell. Place your CSV in `data/` and update `FILE_PATH` if your filename differs.

---

## Saved Artifacts

Each pipeline run creates a timestamped folder (`{Model}_{dd-Mon-yyyy}_{HH-MM-SS}/`) containing:

| File | Description |
|------|-------------|
| `best_model.joblib` | Trained model, ready for inference |
| `scaler.joblib` | Feature scaler (fit on training data only) |
| `label_encoder.joblib` | Target label encoder (class name ↔ integer) |
| `encoding_artifacts.joblib` | Categorical encoding mappings |
| `outlier_transformers.joblib` | Per-column outlier smoothing transformers |
| `feature_names.json` | Ordered list of selected feature names |
| `model_metadata.json` | Best model name, params, all test metrics, class names |
| `pipeline_state.json` | Complete pipeline state (every step's decisions and stats) |
| `confusion_matrix.png` | Raw count + normalized side-by-side confusion matrix |
| `shap_global_importance.png` | Top 20 features by mean \|SHAP\| value |
| `shap_summary_{class}.png` | SHAP beeswarm for the highest-confidence class |
| `shap_per_class_top10.png` | Top 10 features per class grid |
| `shap_waterfall_{class}.png` | Per-class waterfall plot (one sample each) |

---

## Tech Stack

| Category | Libraries |
|----------|-----------|
| **Data** | pandas, numpy, scipy |
| **ML** | scikit-learn, XGBoost, LightGBM |
| **Tuning** | Optuna (TPE Bayesian optimization) |
| **Visualization** | matplotlib, seaborn, plotly |
| **Statistics** | scipy.stats (Kruskal-Wallis, Levene's, Spearman), statsmodels (VIF) |
| **Explainability** | SHAP |
| **GPU** | PyTorch (CUDA detection), XGBoost CUDA, LightGBM GPU |
| **Persistence** | joblib, JSON |

---

## Roadmap

- [x] End-to-end text ML pipeline (19 steps, anti-leakage)
- [x] Text model training & tuning — Extra Trees, 20 features, 6 classes, 99.15% accuracy
- [x] SHAP explainability for text pipeline (Step 19) — global, per-class, waterfall plots saved to output
- [x] Voice PCA pipeline — LightGBM, 23 PCA features, 6 classes, 99.50% accuracy
- [x] SHAP explainability for voice pipeline (Step 17) — global, per-class, waterfall plots saved to output
- [x] Unified output folder naming format — `{Model}_{dd-Mon-yyyy}_{HH-MM-SS}` for both pipelines
- [x] Confusion matrix (raw + normalized) saved as PNG for both pipelines
- [ ] FastAPI inference servers for text and voice models
- [ ] Real-time voice agent — record audio → extract features → screen live
- [ ] Real-time voice agent — record audio → extract features → screen live
