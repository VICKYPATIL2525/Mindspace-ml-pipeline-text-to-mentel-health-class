# Mindspace ML Pipeline — Diagrams

## 1. End-to-End Pipeline Flow

```mermaid
flowchart TD
    subgraph SETUP["⚙️ SETUP"]
        S0["Step 0\nImport Libraries\n& Hardware Detection\n(GPU / CPU)"]
        S1["Step 1\nConfiguration\nCSV Path · Seed · Output Dir"]
    end

    subgraph LOAD["📂 DATA LOADING & PROFILING"]
        S2["Step 2\nLoad CSV\n50K rows × 66 cols"]
        S3["Step 3\nColumn Overview\nList all 66 columns"]
        S4["Step 4\nTarget Selection\nTARGET = 'profile'"]
        S5["Step 5\nData Profiling\nNulls · Duplicates · Constants\nID-like · Leakage Detection"]
        S6["Step 6\nAuto-Clean\nDrop 'target' (leakage)\nImpute remaining nulls"]
    end

    subgraph SPLIT["✂️ TRAIN / TEST SPLIT"]
        S7["Step 7\nTarget Analysis & Split\n7 classes · Imbalance 2.54:1\n80/20 Stratified Split"]
    end

    subgraph TRANSFORM["🔧 FEATURE ENGINEERING (Train-Fit → Apply Both)"]
        S8["Step 8 — Outlier Handling\nTest 4 strategies per column\nPick lowest skew · No rows removed"]
        S9["Step 9 — Encoding\nBinary → Label · Low-card → One-Hot\nHigh-card → Frequency"]
    end

    subgraph EDA["📊 EDA (Training Data Only)"]
        S10["Step 10\nDistributions · Correlations\nKruskal-Wallis H · Levene's W"]
    end

    subgraph FEATURE["🎯 FEATURE SELECTION"]
        S11["Step 11 — Multi-Method Consensus\nCorrelation → VIF → RF + MI + Stats\n65 → 43 features"]
        S12["Step 12 — Scaling\nRobustScaler"]
    end

    subgraph TRAIN["🏋️ MODEL TRAINING"]
        S13["Step 13 — Shortlist 8 Models"]
        S14["Step 14 — 5-Fold CV\nf1_macro scoring"]
        S15["Step 15 — Top-2 Selection"]
    end

    subgraph TUNE["🎛️ TUNING"]
        S16["Step 16 — Optuna TPE\n15 trials · 3 min timeout"]
    end

    subgraph EVAL["✅ EVALUATION & SAVE"]
        S17["Step 17 — Final Test Metrics\nConfusion Matrix · Runner-up"]
        S18["Step 18 — Save All Artifacts"]
    end

    S0 --> S1 --> S2 --> S3 --> S4 --> S5 --> S6
    S6 --> S7
    S7 -->|"Train set only"| S8
    S8 --> S9 --> S10 --> S11 --> S12
    S12 --> S13 --> S14 --> S15 --> S16
    S16 --> S17 --> S18

    S7 -.->|"Test set held out (no leakage)"| S17
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

## 3. Feature Selection Pipeline (Step 11)

```mermaid
flowchart TD
    START["65 Features\n(post-encoding)"]
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

## 5. Outlier Handling Strategy (Step 8)

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
    HW["Hardware Detection\n(Step 0)"]
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
