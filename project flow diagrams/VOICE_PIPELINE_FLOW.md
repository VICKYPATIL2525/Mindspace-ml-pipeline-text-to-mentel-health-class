# Mindspace Voice Pipeline — Diagrams

## 1. End-to-End Pipeline Flow

```mermaid
flowchart LR
    %% ── Phase 1: Setup & Load ──
    subgraph P1[" "]
        direction TB
        S0(["⚙️ Step 0\nImports &\nHardware Detection"])
        S1(["📁 Step 1\nConfiguration"])
        S0 --> S1
    end

    subgraph P2[" "]
        direction TB
        S2(["📂 Step 2\nLoad CSV\n4,800 × 15"])
        S3(["🔍 Step 3\nColumn Overview\n14 PCA cols + label"])
        S35(["🎯 Step 3.5\nManual Setup\nTarget = label"])
        S2 --> S3 --> S35
    end

    subgraph P3[" "]
        direction TB
        S4(["🧪 Step 4\nData Profiling\n38 Duplicates · 0 Nulls"])
        S5(["✅ Step 5\nTarget Validation\n6 Classes · Balanced"])
        S6(["🧹 Step 6\nAuto-Clean\nDrop Duplicates → 4,762 rows"])
        S4 --> S5 --> S6
    end

    %% ── Phase 2: Split (the hard boundary) ──
    S7{{"✂️ Step 7\nTrain / Test Split\n3,809 / 953\nStratified 80 / 20"}}

    %% ── Phase 3: Train-only transforms ──
    subgraph P4["🔧 Transform (fitted on training set only)"]
        direction TB
        S8(["Step 8 · Outlier Smoothing\n4 strategies → lowest skewness\nYeo-Johnson · Winsorize"])
        S9(["Step 9 · Feature Selection\nMutual Information + Random Forest consensus\n14 → 13 features · PC14 dropped"])
        S10(["Step 10 · RobustScaler\nCenter on median · Scale by IQR"])
        S8 --> S9 --> S10
    end

    %% ── Phase 4: Model Training ──
    subgraph P5["🏋️ Train & Compare"]
        direction TB
        S11(["Step 11 · Shortlist\n8 candidate models"])
        S12(["Step 12 · 5-Fold Stratified Cross-Validation\nF1 Macro · parallel · 65.6 seconds"])
        S13(["Step 13 · Top-2 Selection\nLightGBM 0.9876 · ExtraTrees 0.9873"])
        S11 --> S12 --> S13
    end

    %% ── Phase 5: Tuning ──
    subgraph P6["🎛️ Tune"]
        S14(["Step 14 · Optuna Tree-structured Parzen Estimator\n30 trials · 300 second timeout · 5-Fold Cross-Validation\nExtraTrees improved to 0.9892"])
    end

    %% ── Phase 6: Evaluate & Save ──
    subgraph P7["✅ Deliver"]
        direction TB
        S15(["Step 15 · Final Evaluation\nAccuracy 99.3% · F1 Macro 0.9927\nConfusion Matrix · Per-Class Report"])
        S16(["Step 16 · Save\nmodel · scaler · encoder\ntransformers · metadata · state"])
        S15 --> S16
    end

    %% ── Connections ──
    P1 --> P2 --> P3
    P3 --> S7
    S7 -->|"Train 3,809 rows"| P4
    P4 --> P5 --> P6
    P6 --> P7

    %% ── Test set bypass (no leakage) ──
    S7 -.->|"Test 953 rows held out"| S15

    %% ── Styles ──
    style P1 fill:#1a1a2e,stroke:#e94560,color:#fff
    style P2 fill:#16213e,stroke:#0f3460,color:#fff
    style P3 fill:#16213e,stroke:#0f3460,color:#fff
    style S7 fill:#533483,stroke:#e94560,color:#fff
    style P4 fill:#0f3460,stroke:#53a8b6,color:#fff
    style P5 fill:#6a040f,stroke:#d00000,color:#fff
    style P6 fill:#ff6d00,stroke:#ff9e00,color:#fff
    style P7 fill:#2d6a4f,stroke:#40916c,color:#fff
```

---

## 2. Anti-Leakage Data Flow

```mermaid
flowchart LR
    RAW["Raw CSV\n4,800 rows × 15 cols"]
    CLEAN["Cleaned\n4,762 rows × 14 feature cols"]
    SPLIT["Train / Test Split\nStratified 80 / 20"]
    TRAINSET["Training Set\n3,809 rows"]
    TESTSET["Test Set\n953 rows\n(sealed until final evaluation)"]
    FIT["Fit Transforms on Training Set Only\n(Outlier Smoothing · Feature Selection · RobustScaler)"]
    APPLY_TRAIN["Apply → Training Set"]
    APPLY_TEST["Apply → Test Set"]
    MODEL["Train & Tune Models\non Training Set"]
    EVALUATE["Final Evaluation\non Test Set (once only)"]

    RAW --> CLEAN --> SPLIT
    SPLIT --> TRAINSET
    SPLIT --> TESTSET
    TRAINSET --> FIT
    FIT --> APPLY_TRAIN --> MODEL
    FIT --> APPLY_TEST --> EVALUATE
    MODEL --> EVALUATE

    style TRAINSET fill:#2d6a4f,stroke:#40916c,color:#fff
    style TESTSET fill:#9d0208,stroke:#d00000,color:#fff
    style FIT fill:#3a0ca3,stroke:#7209b7,color:#fff
    style EVALUATE fill:#ff6d00,stroke:#ff9e00,color:#fff
```

---

## 3. Feature Selection Pipeline

```mermaid
flowchart TD
    START["14 Principal Component Features\n(PC1 through PC14 — already PCA-reduced)"]
    NZV["Near-Zero Variance Check\nDrop if standard deviation less than 1e-6\n0 columns flagged"]
    MI["Mutual Information Scores\nComputed on training data\nBottom 2 percent threshold = 0.094"]
    RF["Random Forest Importance\n100 trees · max depth 8\nBottom 2 percent threshold = 0.021"]
    CONSENSUS["Consensus Rule\nDrop only if in bottom 2 percent by BOTH\nMutual Information AND Random Forest"]
    PRUNE["Drop PC14\n(only feature in bottom 2 percent by both metrics)"]
    FINAL["13 Selected Features\n(PC1 through PC13)"]

    START --> NZV --> MI
    NZV --> RF
    MI --> CONSENSUS
    RF --> CONSENSUS
    CONSENSUS --> PRUNE --> FINAL

    style START fill:#16213e,stroke:#0f3460,color:#fff
    style FINAL fill:#2d6a4f,stroke:#40916c,color:#fff
    style CONSENSUS fill:#ff6d00,stroke:#ff9e00,color:#fff
    style PRUNE fill:#9d0208,stroke:#d00000,color:#fff
```

---

## 4. Model Selection & Tuning Flow

```mermaid
flowchart TD
    POOL["8 Candidate Models"]
    RF2["Random Forest\n200 estimators"]
    ET["Extra Trees\n200 estimators"]
    LGB["LightGBM\n300 estimators · GPU enabled"]
    HGB["HistGradientBoosting\n200 iterations"]
    XGB["XGBoost\n200 estimators · CUDA GPU"]
    LR["Logistic Regression\nmax iterations 1000"]
    SVM["Support Vector Machine RBF\nprobability enabled"]
    KNN["K-Nearest Neighbors\nk = 7"]
    CV["5-Fold Stratified Cross-Validation\nF1 Macro Scoring · Parallel · 65.6 seconds"]
    RANK["Rank All 8 Models by Cross-Validation Score"]
    TOP2["Top 2 Selected\nLightGBM 0.9876 · ExtraTrees 0.9873"]
    OPTUNA["Optuna Tree-structured Parzen Estimator Tuning\n30 trials · 300 second timeout per model · 5-Fold Cross-Validation"]
    BEST["Best Model: ExtraTrees\nF1 Macro = 0.9927 · Accuracy = 99.3%"]
    RUNNER["Runner-up: LightGBM\nF1 Macro = 0.9905"]

    POOL --> RF2 & ET & LGB & HGB & XGB & LR & SVM & KNN
    RF2 & ET & LGB & HGB & XGB & LR & SVM & KNN --> CV
    CV --> RANK --> TOP2
    TOP2 --> OPTUNA
    OPTUNA --> BEST
    OPTUNA --> RUNNER

    style BEST fill:#2d6a4f,stroke:#40916c,color:#fff
    style OPTUNA fill:#ff6d00,stroke:#ff9e00,color:#fff
    style POOL fill:#3a0ca3,stroke:#7209b7,color:#fff
    style RUNNER fill:#16213e,stroke:#0f3460,color:#fff
```

---

## 5. Outlier Handling Strategy

```mermaid
flowchart TD
    COL["For Each Numeric Feature Column\n(14 columns · fitted on training data only)"]
    DETECT["Detect Outliers via Interquartile Range\nBounds = Q1 minus 1.5 times IQR  to  Q3 plus 1.5 times IQR"]
    CHECK{"> 0.3% of values are outliers?"}
    SKIP["Skip — within tolerance\n(no treatment needed)"]
    TEST["Test All 4 Smoothing Strategies"]
    W["Winsorize\nClip values to IQR bounds"]
    L["Log1p\nLog transform for right-skew\nnon-negative columns only"]
    SQ["Square Root\nModerate skew\nnon-negative columns only"]
    YJ["Yeo-Johnson Power Transform\nWorks on any distribution\nincluding negative values"]
    PICK["Pick Strategy with Lowest Absolute Skewness"]
    APPLY["Apply to Training and Test Sets\nusing training-fitted parameters"]

    COL --> DETECT --> CHECK
    CHECK -->|No| SKIP
    CHECK -->|Yes| TEST
    TEST --> W & L & SQ & YJ
    W & L & SQ & YJ --> PICK --> APPLY

    style PICK fill:#ff6d00,stroke:#ff9e00,color:#fff
    style APPLY fill:#2d6a4f,stroke:#40916c,color:#fff
    style SKIP fill:#16213e,stroke:#0f3460,color:#fff
```

**Results:** 13 of 14 columns treated — 11 columns used Yeo-Johnson, 2 columns used Winsorize, 1 column skipped.

---

## 6. Hardware Utilization

```mermaid
flowchart LR
    HW["Hardware Detection\n(on startup)"]
    GPU{"CUDA GPU Available?"}
    YES_GPU["NVIDIA GeForce GTX 1650 Ti detected\nXGBoost → device = cuda\nLightGBM → device = gpu"]
    NO_GPU["No GPU detected\nAll models → CPU\nn_jobs = -1 (all cores)"]
    CORES["CPU Detection\n8 cores detected\nN_JOBS = 7 (cores minus 1)"]
    PARALLEL["All Cross-Validation folds run in parallel\njoblib.Parallel for sample-weighted models"]

    HW --> GPU
    GPU -->|Yes| YES_GPU
    GPU -->|No| NO_GPU
    YES_GPU --> CORES
    NO_GPU --> CORES
    CORES --> PARALLEL

    style YES_GPU fill:#2d6a4f,stroke:#40916c,color:#fff
    style NO_GPU fill:#16213e,stroke:#0f3460,color:#fff
    style PARALLEL fill:#ff6d00,stroke:#ff9e00,color:#fff
```

---

## 7. Class Distribution & Target Encoding

```mermaid
flowchart TD
    RAW_LABEL["Raw Label Column\n6 Classes · 800 samples each\nTotal 4,800 rows · Perfectly Balanced"]
    DEDUP["After Duplicate Removal\n38 rows dropped\nTotal 4,762 rows"]
    ENCODE["LabelEncoder\nAnxiety=0 · Bipolar=1 · Depression=2\nNormal=3 · Stress=4 · Suicidal=5"]
    IMBALANCE["Imbalance Ratio Check\nMax class / Min class = 1.03x\nThreshold = 1.5x"]
    STRATEGY["Imbalance Strategy = NONE\nNo oversampling or class weighting needed\nAll models use default class weights"]
    SPLIT2["Stratified Split preserves\n~16.7% per class in both\nTraining Set and Test Set"]

    RAW_LABEL --> DEDUP --> ENCODE --> IMBALANCE --> STRATEGY --> SPLIT2

    style RAW_LABEL fill:#16213e,stroke:#0f3460,color:#fff
    style STRATEGY fill:#2d6a4f,stroke:#40916c,color:#fff
    style ENCODE fill:#3a0ca3,stroke:#7209b7,color:#fff
```

---

## 8. Final Model Performance

```mermaid
flowchart TD
    WINNER["Best Model: Extra Trees Classifier\n(after Optuna tuning)"]

    subgraph PARAMS["Tuned Hyperparameters"]
        P1_["n_estimators = 400"]
        P2_["max_depth = 26"]
        P3_["min_samples_split = 2"]
        P4_["min_samples_leaf = 1"]
        P5_["max_features = sqrt"]
    end

    subgraph METRICS["Test Set Performance\n(953 held-out samples · evaluated once only)"]
        M1["Accuracy = 99.27%"]
        M2["F1 Macro = 0.9927"]
        M3["Balanced Accuracy = 0.9927"]
        M4["F1 Weighted = 0.9927"]
    end

    subgraph PERCLASS["Per-Class F1 Scores"]
        C1["Anxiety → F1 = 1.00"]
        C2["Bipolar → F1 = 1.00"]
        C3["Depression → F1 = 0.98"]
        C4["Normal → F1 = 0.98"]
        C5["Stress → F1 = 1.00"]
        C6["Suicidal → F1 = 1.00"]
    end

    WINNER --> PARAMS
    WINNER --> METRICS
    WINNER --> PERCLASS

    style WINNER fill:#2d6a4f,stroke:#40916c,color:#fff
    style METRICS fill:#1b4332,stroke:#52b788,color:#fff
    style PERCLASS fill:#3a0ca3,stroke:#7209b7,color:#fff
    style PARAMS fill:#16213e,stroke:#0f3460,color:#fff
```

---

## 9. Saved Artifacts

```mermaid
flowchart LR
    PIPELINE["Pipeline Run Complete"]
    FOLDER["voice_ml_pipeline_output /\nExtraTrees_25042026_142358 /"]
    M["best_model.joblib\nTrained Extra Trees Classifier\n11,230.8 KB"]
    SC["scaler.joblib\nFitted RobustScaler\n0.6 KB"]
    LE["label_encoder.joblib\n6 class label mappings\n0.3 KB"]
    EA["encoding_artifacts.joblib\nNo categorical encoding needed\n(PCA-only pipeline)\n0.1 KB"]
    OT["outlier_transformers.joblib\nPer-column outlier transform parameters\n0.7 KB"]
    FN["feature_names.json\n13 selected PCA feature names\n0.1 KB"]
    MD["model_metadata.json\nMetrics · best params · class names\nhardware info · 8.3 KB"]
    PS["pipeline_state.json\nComplete run state from all steps\n12.7 KB"]

    PIPELINE --> FOLDER
    FOLDER --> M & SC & LE & EA & OT & FN & MD & PS

    style PIPELINE fill:#3a0ca3,stroke:#7209b7,color:#fff
    style FOLDER fill:#ff6d00,stroke:#ff9e00,color:#fff
    style M fill:#2d6a4f,stroke:#40916c,color:#fff
```
