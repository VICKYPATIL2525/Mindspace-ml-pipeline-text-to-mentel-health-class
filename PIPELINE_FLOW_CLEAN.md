# Mindspace ML Pipeline — Diagrams

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
        S2(["📂 Step 2\nLoad CSV\n50K × 66"])
        S3(["🔍 Step 3\nColumn Overview"])
        S4(["🎯 Step 4\nTarget = profile"])
        S2 --> S3 --> S4
    end

    subgraph P3[" "]
        direction TB
        S5(["🧪 Step 5\nData Profiling\nNulls · Leakage"])
        S6(["🧹 Step 6\nAuto-Clean\nDrop leakage col"])
        S5 --> S6
    end

    %% ── Phase 2: Split (the hard boundary) ──
    S7{{"✂️ Step 7\nTrain/Test Split\n40K / 10K\nStratified"}}

    %% ── Phase 3: Train-only transforms ──
    subgraph P4["🔧 Transform (fit on train)"]
        direction TB
        S8(["Step 8 · Outlier Smoothing\n4 strategies → lowest skew"])
        S9(["Step 9 · Encoding\nLabel · One-Hot · Frequency"])
        S8 --> S9
    end

    subgraph P5["📊 Analyze"]
        S10(["Step 10 · EDA\nDistributions · Correlations\nKruskal-Wallis · Levene's"])
    end

    subgraph P6["🎯 Select & Scale"]
        direction TB
        S11(["Step 11 · Feature Selection\nCorr → VIF → RF+MI+Stats\n65 → 43 features"])
        S12(["Step 12 · RobustScaler"])
        S11 --> S12
    end

    %% ── Phase 4: Model Training ──
    subgraph P7["🏋️ Train & Compare"]
        direction TB
        S13(["Step 13 · Shortlist\n8 candidate models"])
        S14(["Step 14 · 5-Fold CV\nf1_macro · parallel"])
        S15(["Step 15 · Top-2\nselection"])
        S13 --> S14 --> S15
    end

    %% ── Phase 5: Tuning ──
    subgraph P8["🎛️ Tune"]
        S16(["Step 16 · Optuna TPE\n15 trials · 3 min\n5-fold CV"])
    end

    %% ── Phase 6: Evaluate & Save ──
    subgraph P9["✅ Deliver"]
        direction TB
        S17(["Step 17 · Final Eval\nAccuracy 92% · F1 0.918\nConfusion Matrix"])
        S18(["Step 18 · Save\nmodel · scaler · encoder\ntransformers · metadata"])
        S17 --> S18
    end

    %% ── Connections ──
    P1 --> P2 --> P3
    P3 --> S7
    S7 -->|"Train 40K"| P4
    P4 --> P5 --> P6 --> P7 --> P8
    P8 --> P9

    %% ── Test set bypass (no leakage) ──
    S7 -.->|"Test 10K held out"| S17

    %% ── Styles ──
    style P1 fill:#1a1a2e,stroke:#e94560,color:#fff
    style P2 fill:#16213e,stroke:#0f3460,color:#fff
    style P3 fill:#16213e,stroke:#0f3460,color:#fff
    style S7 fill:#533483,stroke:#e94560,color:#fff
    style P4 fill:#0f3460,stroke:#53a8b6,color:#fff
    style P5 fill:#1b4332,stroke:#52b788,color:#fff
    style P6 fill:#3a0ca3,stroke:#7209b7,color:#fff
    style P7 fill:#6a040f,stroke:#d00000,color:#fff
    style P8 fill:#ff6d00,stroke:#ff9e00,color:#fff
    style P9 fill:#2d6a4f,stroke:#40916c,color:#fff
```

---

## 2. Anti-Leakage Data Flow

```mermaid
flowchart LR
    RAW["Raw CSV\n50K × 66"]
    CLEAN["Cleaned\n50K × 65"]
    SPLIT["Train/Test Split"]
    TRAINSET["Train Set\n40K rows"]
    TESTSET["Test Set\n10K rows"]
    FIT["Fit Transforms\n(outlier · encode · scale · select)"]
    APPLY_TRAIN["Apply → Train"]
    APPLY_TEST["Apply → Test"]
    MODEL["Train Models"]
    EVALUATE["Evaluate"]

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
```

---

## 3. Feature Selection Pipeline

```mermaid
flowchart TD
    START["65 Features\n(after encoding & EDA)"]
    CORR["Correlation Filter\nRemove one from pairs |r| ≥ 0.85"]
    VIF["VIF Iteration\nRemove VIF > 10\n(top-25% importance protected)"]
    RF["Random Forest\nMDI Importance Ranking"]
    MI["Mutual Information\nScores on train data"]
    STAT["Statistical Tests\nKruskal-Wallis H + Levene's W"]
    CONSENSUS["Consensus Ranking\nAverage ranks from RF + MI + Stats"]
    PRUNE["Prune by MI Threshold\nDrop features with MI < 0.01"]
    FINAL["43 Selected Features"]

    START --> CORR --> VIF
    VIF --> RF
    VIF --> MI
    VIF --> STAT
    RF --> CONSENSUS
    MI --> CONSENSUS
    STAT --> CONSENSUS
    CONSENSUS --> PRUNE --> FINAL

    style START fill:#16213e,stroke:#0f3460,color:#fff
    style FINAL fill:#2d6a4f,stroke:#40916c,color:#fff
    style CONSENSUS fill:#ff6d00,stroke:#ff9e00,color:#fff
```

---

## 4. Model Selection & Tuning Flow

```mermaid
flowchart TD
    POOL["8 Candidate Models"]
    RF["Random Forest"]
    LGB["LightGBM"]
    ET["Extra Trees"]
    XGB["XGBoost"]
    HGB["HistGradientBoosting"]
    LR["Logistic Regression"]
    SVM["SVM (RBF)"]
    KNN["KNN"]
    CV["5-Fold Stratified CV\nf1_macro scoring"]
    RANK["Rank by CV Score"]
    TOP2["Top 2 Selected"]
    OPTUNA["Optuna TPE Tuning\n15 trials · 3 min timeout · 5-fold CV"]
    BEST["Best Model: LightGBM\nF1 macro = 0.918"]
    RUNNER["Runner-up:\nHistGradientBoosting"]

    POOL --> RF & LGB & ET & XGB & HGB & LR & SVM & KNN
    RF & LGB & ET & XGB & HGB & LR & SVM & KNN --> CV
    CV --> RANK --> TOP2
    TOP2 --> OPTUNA
    OPTUNA --> BEST
    OPTUNA --> RUNNER

    style BEST fill:#2d6a4f,stroke:#40916c,color:#fff
    style OPTUNA fill:#ff6d00,stroke:#ff9e00,color:#fff
    style POOL fill:#3a0ca3,stroke:#7209b7,color:#fff
```

---

## 5. Outlier Handling Strategy

```mermaid
flowchart TD
    COL["For each numeric column\n(on training data)"]
    DETECT["Detect outliers via IQR\nQ1 - 1.5·IQR ... Q3 + 1.5·IQR"]
    CHECK{"> 0.3% outliers?"}
    SKIP["Skip — within tolerance"]
    TEST["Test 4 smoothing strategies"]
    W["Winsorize\n(cap to IQR bounds)"]
    L["Log1p\n(right-skew, non-negative)"]
    SQ["Sqrt\n(moderate skew, non-negative)"]
    YJ["Yeo-Johnson\n(any distribution)"]
    PICK["Pick strategy with\nlowest |skewness|"]
    APPLY["Apply to train & test\n(using train-fit params)"]

    COL --> DETECT --> CHECK
    CHECK -->|No| SKIP
    CHECK -->|Yes| TEST
    TEST --> W & L & SQ & YJ
    W & L & SQ & YJ --> PICK --> APPLY

    style PICK fill:#ff6d00,stroke:#ff9e00,color:#fff
    style APPLY fill:#2d6a4f,stroke:#40916c,color:#fff
```

---

## 6. Hardware Utilization

```mermaid
flowchart LR
    HW["Hardware Detection\n(on startup)"]
    GPU{"CUDA GPU\nAvailable?"}
    YES_GPU["XGBoost → device='cuda'\nLightGBM → device='gpu'\nFeature Importance → XGB GPU"]
    NO_GPU["All models → CPU\nn_jobs=-1 (all cores)"]
    PARALLEL["All CV folds parallel\njoblib.Parallel for\nsample_weight models"]

    HW --> GPU
    GPU -->|Yes| YES_GPU
    GPU -->|No| NO_GPU
    YES_GPU --> PARALLEL
    NO_GPU --> PARALLEL

    style YES_GPU fill:#2d6a4f,stroke:#40916c,color:#fff
    style NO_GPU fill:#16213e,stroke:#0f3460,color:#fff
    style PARALLEL fill:#ff6d00,stroke:#ff9e00,color:#fff
```

---

## 7. Saved Artifacts

```mermaid
flowchart LR
    PIPELINE["Pipeline Run Complete"]
    FOLDER["pipeline_output/\nLightGBM_13032026_110356/"]
    M["best_model.joblib\nTrained LightGBM"]
    SC["scaler.joblib\nRobustScaler"]
    LE["label_encoder.joblib\n7 class mappings"]
    EA["encoding_artifacts.joblib\nCategorical mappings"]
    OT["outlier_transformers.joblib\nPer-column transforms"]
    FN["feature_names.json\n43 selected features"]
    MD["model_metadata.json\nMetrics & params"]
    PS["pipeline_state.json\nFull run state"]

    PIPELINE --> FOLDER
    FOLDER --> M & SC & LE & EA & OT & FN & MD & PS

    style PIPELINE fill:#3a0ca3,stroke:#7209b7,color:#fff
    style FOLDER fill:#ff6d00,stroke:#ff9e00,color:#fff
```
