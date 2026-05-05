# Mindspace — Mental Health Profiling via NLP & Voice Features

> **Dual-pipeline ML system that classifies mental health profiles from text/linguistic features and voice/acoustic PCA features — achieving 92% accuracy (text) and 99.3% accuracy (voice) across 7 and 6 classes respectively.**

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

---

## Project Summary

Mindspace is a mental health classification system with two independent ML pipelines and two production FastAPI inference servers:

| Pipeline | Notebook | Model | Accuracy | Input |
|----------|----------|-------|----------|-------|
| **Text** | `text-ml-pipeline.ipynb` | LightGBM | 92.0% | 43 linguistic/semantic features |
| **Voice** | `voice-pca-pipeline-guided.ipynb` | ExtraTrees | 99.3% | 13 PCA components from acoustic features |

Both pipelines are **fully automated and adaptive**: they handle EDA, feature engineering, model selection, hyperparameter tuning, and artifact saving without manual intervention.

### What Each Pipeline Does

**Text Pipeline**
- Classifies mental health profiles into **7 categories** based on linguistic, emotional, and semantic features extracted from speech/text
- Handles outliers, nulls, duplicates, leakage, encoding, feature selection, and scaling automatically
- Prevents data leakage — train/test split happens *before* any transformation
- Trains and compares up to 8 ML algorithms via 5-fold CV, tunes top 2 with Optuna

**Voice Pipeline**
- Classifies mental health profiles into **6 categories** based on PCA-reduced acoustic features
- Input: 13 principal components extracted from OpenSMILE acoustic features
- Same anti-leakage pipeline design as the text pipeline
- Trained on acoustic/prosodic feature dataset with pre-computed PCA reduction

---

## Datasets

### Text Dataset

| Property | Value |
|----------|-------|
| **File** | `data/mental_health_synthetic_dataset_with_normal.csv` |
| **Rows** | 50,000 |
| **Columns** | 66 (61 float, 3 object, 2 int) |
| **Target** | `profile` (7 classes) |
| **Task** | Multi-class classification |
| **Imbalance Ratio** | 2.54:1 |
| **Train / Test Split** | 40,000 / 10,000 (80/20, stratified) |

**Text Target Classes**

| Class | Description |
|-------|-------------|
| Anxiety | Anxiety-related speech patterns |
| Bipolar_Mania | Manic episode indicators |
| Depression | Depressive speech markers |
| Normal | Baseline / healthy patterns |
| Phobia | Phobia-related indicators |
| Stress | Stress-related speech patterns |
| Suicidal_Tendency | Suicidal ideation markers |

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
| **File** | `data/features_pca_dataset.csv` |
| **Rows** | ~3,000 |
| **Columns** | 14 (13 PCA features + label) |
| **Target** | `label` (6 classes) |
| **Task** | Multi-class classification |
| **Train / Test Split** | ~2,400 / ~600 (80/20, stratified) |

**Voice Target Classes**

| Class | Description |
|-------|-------------|
| Anxiety | Anxiety disorder voice patterns |
| Bipolar | Bipolar / manic episode indicators |
| Depression | Depressive speech markers |
| Normal | Baseline / healthy voice patterns |
| Stress | Stress-related voice patterns |
| Suicidal | Suicidal ideation voice markers |

**Voice Features**: 13 PCA components (PC1–PC13) derived from OpenSMILE acoustic features including MFCC coefficients, spectral features (entropy, rolloff, harmonicity, flux), pitch (F0), shimmer, jitter, voicing, RMS energy, zero-crossing rate, and HNR.

---

## Pipeline Architecture

> **Visual diagrams** of every flow (end-to-end pipeline, anti-leakage, feature selection, model tuning, outlier handling) are in [`project flow diagrams/PIPELINE_FLOW_CLEAN.md`](project%20flow%20diagrams/PIPELINE_FLOW_CLEAN.md)

### Text Pipeline — 18 Steps

| Step | Stage | What Happens |
|------|-------|-------------|
| **0** | Import & Hardware Detection | Load libraries; detect GPU (CUDA) and CPU core count |
| **1** | Configuration | Set CSV path, random seed (42), output directory |
| **2** | Data Loading | Load CSV, preview shape, dtypes, head |
| **3** | Column Overview | Print all columns for manual review |
| **4** | Target Selection | Set `TARGET_COLUMN = 'profile'` |
| **5** | Data Profiling | Detect nulls, duplicates, constants, ID-like columns, leakage |
| **6** | Auto-Clean | Drop flagged columns, impute remaining nulls |
| **7** | Target Analysis & Split | Analyze class balance → **train/test split (80/20, stratified)** before any transformation |
| **8** | Outlier Handling | Test 4 smoothing strategies per column (winsorize, log1p, sqrt, yeo-johnson); pick lowest skew. Fit on train only. |
| **9** | Encoding | Binary → Label Encoding; Low-cardinality → One-Hot; High-cardinality → Frequency Encoding. Fit on train only. |
| **10** | EDA & Visualization | Distribution plots, correlation heatmaps, Kruskal-Wallis H tests, Levene's W tests — training data only |
| **11** | Feature Selection | Correlation filter → VIF → RF importance + MI + stat tests consensus → prune MI < 0.01. **65 → 43 features.** |
| **12** | Scaling | RobustScaler fit on train, transform both |
| **13** | Model Shortlisting | Dynamically select models based on dataset size and dimensionality |
| **14** | Training & CV | 5-fold stratified CV, scored by `f1_macro` |
| **15** | Top-K Selection | Pick top 2 models for tuning |
| **16** | Hyperparameter Tuning | Optuna TPE, 15 trials, 3-min timeout, 5-fold CV per model |
| **17** | Final Evaluation | Full test-set metrics, confusion matrix, runner-up comparison |
| **18** | Save Artifacts | Model, scaler, encoder, transformers, feature names, metadata |

### Anti-Leakage Design

Every transformation (outlier handling, encoding, scaling, feature selection) is **fit exclusively on training data** and applied identically to the test set. The train/test split at Step 7 is a hard boundary — no test data information flows backward.

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

### Text Model — LightGBM

| Metric | Score |
|--------|-------|
| **Accuracy** | 0.920 |
| **F1 (macro)** | 0.918 |
| **F1 (weighted)** | 0.920 |
| **Precision (macro)** | 0.917 |
| **Recall (macro)** | 0.920 |

Training: 40,000 samples | Test: 10,000 samples | Features: 43 (selected from 65)

**Tuned Hyperparameters**

| Parameter | Value |
|-----------|-------|
| `n_estimators` | 250 |
| `max_depth` | 15 |
| `learning_rate` | 0.121 |
| `num_leaves` | 64 |
| `subsample` | 0.578 |
| `colsample_bytree` | 0.578 |
| `min_child_samples` | 7 |
| `reg_alpha` | 0.625 |
| `reg_lambda` | 0.003 |

**Top predictors**: `overall_sentiment_score`, `semantic_coherence_score`, `self_reference_density`, `future_focus_ratio`, `positive_emotion_ratio`, `fear_word_frequency`

### Voice Model — ExtraTrees

| Metric | Score |
|--------|-------|
| **Accuracy** | 0.993 |
| **F1 (macro)** | 0.993 |
| **F1 (weighted)** | 0.993 |
| **Precision (macro)** | 0.993 |
| **Recall (macro)** | 0.993 |

Training: ~2,400 samples | Test: ~600 samples | Features: 13 PCA components (PC1–PC13)

**Tuned Hyperparameters**

| Parameter | Value |
|-----------|-------|
| `n_estimators` | 400 |
| `max_depth` | 26 |
| `min_samples_split` | 2 |
| `min_samples_leaf` | 1 |
| `max_features` | sqrt |

---

## Project Structure

```
Mindspace-voice-agent/
├── text-ml-pipeline.ipynb               # Text ML pipeline (18 steps, 39 cells)
├── voice-pca-pipeline-guided.ipynb      # Voice/PCA ML pipeline
├── requirements.txt                     # Python dependencies
│
├── data/
│   └── features_pca_dataset.csv         # Voice PCA dataset (~3K rows, 13 PCA + label)
│
├── demo-api-input-data-sample/
│   └── voice_normal_sample_1.json       # Sample voice API request (13 PCA features)
│
├── Both_API_combined/                   # FastAPI inference servers
│   ├── api_text_to_sentiment.py         # Text API — LightGBM, 43 features, 7 classes (port 9000)
│   ├── api_voice_to_sentiment.py        # Voice API — ExtraTrees, 13 PCA features, 6 classes (port 9100)
│   ├── .env                             # API_KEY (never commit)
│   ├── .env.example                     # Template for .env
│   ├── requirements.txt                 # Deployment-only dependencies
│   └── README.md                        # API deployment documentation
│
├── project flow diagrams/               # Visual documentation
│   ├── PIPELINE_FLOW_CLEAN.md           # Mermaid diagrams of all pipeline flows
│   └── project_flow_diagram.md          # High-level project overview diagram
│
├── text_ml_pipeline_output/
│   └── LightGBM_13032026_110356/        # Text model artifacts
│       ├── best_model.joblib            # Trained LightGBM classifier (7.7 MB)
│       ├── scaler.joblib                # RobustScaler (fit on 40K train samples)
│       ├── label_encoder.joblib         # Target label encoder
│       ├── encoding_artifacts.joblib    # Categorical encoding maps
│       ├── outlier_transformers.joblib  # Per-column outlier smoothing transforms
│       ├── feature_names.json           # 43 selected feature names
│       ├── model_metadata.json          # Metrics, params, class names
│       └── pipeline_state.json          # Full pipeline execution state
│
├── voice_ml_pipeline_output/
│   └── ExtraTrees_25042026_142358/      # Voice model artifacts
│       ├── best_model.joblib            # Trained ExtraTrees classifier (11 MB)
│       ├── scaler.joblib                # RobustScaler (fit on train)
│       ├── label_encoder.joblib         # Target label encoder
│       ├── encoding_artifacts.joblib    # Categorical encoding maps
│       ├── outlier_transformers.joblib  # Per-column outlier smoothing transforms
│       ├── feature_names.json           # [PC1 … PC13]
│       ├── model_metadata.json          # Metrics, params, class names
│       └── pipeline_state.json          # Full pipeline execution state
│
└── myenv/                               # Python virtual environment
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
cd Mindspace-voice-agent

# Create virtual environment
python -m venv myenv

# Activate (Windows)
myenv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### Optional: GPU Support

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
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

### API Servers

```bash
# From project root:

# Text API (port 9000)
uvicorn Both_API_combined.api_text_to_sentiment:app --reload --port 9000

# Voice API (port 9100)
uvicorn Both_API_combined.api_voice_to_sentiment:app --reload --port 9100
```

Swagger docs: `http://localhost:9000/docs` and `http://localhost:9100/docs`

---

## Saved Artifacts

Each pipeline run creates a timestamped folder (`{Model}_{ddmmyyyy}_{hhmmss}/`) containing:

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

---

## Tech Stack

| Category | Libraries |
|----------|-----------|
| **Data** | pandas, numpy, scipy |
| **ML** | scikit-learn, XGBoost, LightGBM |
| **Tuning** | Optuna (TPE Bayesian optimization) |
| **Visualization** | matplotlib, seaborn, plotly |
| **Statistics** | scipy.stats (Kruskal-Wallis, Levene's, Spearman), statsmodels (VIF) |
| **GPU** | PyTorch (CUDA detection), XGBoost CUDA, LightGBM GPU |
| **API** | FastAPI, uvicorn, pydantic |
| **Persistence** | joblib, JSON |

---

## Roadmap

- [x] Synthetic text dataset generation (multilingual: English, Hindi, Marathi)
- [x] End-to-end text ML pipeline (18 steps, anti-leakage)
- [x] Text model training & tuning — LightGBM (92% accuracy, 43 features, 7 classes)
- [x] Voice PCA pipeline — ExtraTrees (99.3% accuracy, 13 PCA features, 6 classes)
- [x] Text API — FastAPI server, LightGBM, port 9000
- [x] Voice API — FastAPI server, ExtraTrees, port 9100
- [ ] Real-time voice agent — record audio → extract features → classify live
