# =============================================================================
# Mindspace Mental Health Classifier — FastAPI Inference Server
# =============================================================================
# This file is the entire backend API for the Mindspace project.
# It loads the trained LightGBM model and all preprocessing artifacts at
# startup, then serves predictions via HTTP endpoints.
#
# Flow for every prediction request:
#   1. Client sends 43 float features (extracted from speech/text)
#   2. API applies outlier smoothing (same transforms used during training)
#   3. API scales the features with RobustScaler (same scaler from training)
#   4. LightGBM model predicts one of 7 mental health profiles
#   5. API returns the label + confidence + all 7 class probabilities
# =============================================================================

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

# ─── Artifact paths ──────────────────────────────────────────────────────────
# All model artifacts were saved by the ML pipeline (full-final-pipeline.ipynb)
# into a timestamped folder. This path points to that folder.
# __file__ is deployment/main.py → .parent.parent is the project root.

ARTIFACTS_DIR = Path(__file__).parent.parent / "pipeline_output" / "LightGBM_13032026_110356"

# ─── Global state (loaded once at startup) ────────────────────────────────────
# We use a plain dict to hold all artifacts in memory so every request can
# reuse them without re-loading from disk each time (which would be very slow).

artifacts = {}


def load_artifacts():
    """
    Load all 7 ML artifacts from disk into the global `artifacts` dict.
    Called once when the server starts up (see lifespan below).

    Artifacts loaded:
      - best_model.joblib          → trained LightGBM classifier
      - scaler.joblib              → RobustScaler (fit on 40K training rows)
      - label_encoder.joblib       → maps integer predictions → class name strings
      - encoding_artifacts.joblib  → categorical encoding maps (not used at inference
                                     since language is already one-hot in input)
      - outlier_transformers.joblib → per-column smoothing params (winsorize bounds,
                                      yeo-johnson fitted transformer, sqrt shift)
      - feature_names.json         → ordered list of 43 feature names the model expects
      - model_metadata.json        → hyperparams, CV score, test metrics, class names
    """
    artifacts["model"]               = joblib.load(ARTIFACTS_DIR / "best_model.joblib")
    artifacts["scaler"]              = joblib.load(ARTIFACTS_DIR / "scaler.joblib")
    artifacts["label_encoder"]       = joblib.load(ARTIFACTS_DIR / "label_encoder.joblib")
    artifacts["encoding"]            = joblib.load(ARTIFACTS_DIR / "encoding_artifacts.joblib")
    artifacts["outlier_transformers"] = joblib.load(ARTIFACTS_DIR / "outlier_transformers.joblib")
    artifacts["feature_names"]       = json.loads((ARTIFACTS_DIR / "feature_names.json").read_text())
    artifacts["metadata"]            = json.loads((ARTIFACTS_DIR / "model_metadata.json").read_text())


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager — runs code at startup and shutdown.
    Everything before `yield` runs when the server starts.
    Everything after `yield` would run on shutdown (nothing needed here).
    This is the modern FastAPI replacement for @app.on_event("startup").
    """
    load_artifacts()
    yield


# ─── App ──────────────────────────────────────────────────────────────────────
# Create the FastAPI application instance.
# `lifespan=lifespan` wires up the startup artifact loading defined above.
# The title/description appear in the auto-generated Swagger UI at /docs.

app = FastAPI(
    title="Mindspace Mental Health Classifier",
    description="Predicts mental health profile from speech/text features.",
    version="1.0.0",
    lifespan=lifespan,
)


# ─── Request / Response schemas ───────────────────────────────────────────────
# Pydantic models define the exact shape of JSON that the API accepts and returns.
# FastAPI uses these to automatically validate incoming requests and generate
# the interactive docs at /docs — no extra work needed.

class PredictRequest(BaseModel):
    """
    The 43 features the model was trained on, after feature selection in the pipeline.
    The caller is responsible for extracting these from raw text before calling /predict.

    All values are floats. The only exception is language_hindi / language_marathi
    which must be exactly 0.0 or 1.0 (one-hot encoded language flags).

    Note: not all embedding dimensions (emb_0 to emb_31) are included — only the
    17 that survived the pipeline's feature selection step are required.
    """

    # ── Linguistic / Semantic features (19) ──────────────────────────────────
    # These capture the emotional and semantic content of the text.
    overall_sentiment_score: float      # tanh of (positive - negative) emotion ratio; range [-1, 1]
    semantic_coherence_score: float     # how logically connected sentences are; range [0, 1]
    self_reference_density: float       # proportion of first-person pronouns (I, me, my); range [0, 0.4]
    future_focus_ratio: float           # proportion of future-tense words; range [0, 0.3]
    positive_emotion_ratio: float       # proportion of positive emotion words; range [0, 0.12]
    fear_word_frequency: float          # proportion of fear-related words; range [0, 0.3]
    sadness_word_frequency: float       # proportion of sadness-related words; range [0, 0.4]
    negative_emotion_ratio: float       # proportion of negative emotion words; range [0, 0.5]
    uncertainty_word_frequency: float   # proportion of uncertainty words (maybe, perhaps...); range [0, 0.3]
    anger_word_frequency: float         # proportion of anger-related words; range [0, 0.2]
    rumination_phrase_frequency: float  # repetitive negative thought patterns; range [0, 0.3]
    filler_word_frequency: float        # um, uh, like, you know...; range [0, 0.25]
    topic_shift_frequency: float        # entropy of topic distribution (how much topics jump); range [0, 1]
    total_word_count: float             # total words spoken/written
    avg_sentence_length: float          # average words per sentence
    language_model_perplexity: float    # how "surprising"/unpredictable the text is; higher = more chaotic
    past_focus_ratio: float             # proportion of past-tense words; range [0, 0.4]
    repetition_rate: float              # how often words/phrases repeat; range [0, 0.2]
    adjective_ratio: float              # proportion of adjectives in the text; range [0, 0.25]

    # ── Topic model outputs (5) ───────────────────────────────────────────────
    # Weights from a topic model (e.g., LDA) trained on the corpus.
    # Each row sums to 1.0 (a probability distribution over 5 topics).
    topic_0: float
    topic_1: float
    topic_2: float
    topic_3: float
    topic_4: float

    # ── Sentence embedding dimensions (17 of 32) ──────────────────────────────
    # 32-dimensional sentence embeddings were generated for the text.
    # After feature selection, only these 17 dimensions were retained.
    # Non-contiguous indices (e.g., emb_2, emb_6 were dropped as low-importance).
    emb_1: float
    emb_3: float
    emb_4: float
    emb_5: float
    emb_7: float
    emb_8: float
    emb_10: float
    emb_11: float
    emb_12: float
    emb_14: float
    emb_15: float
    emb_21: float
    emb_22: float
    emb_25: float
    emb_28: float
    emb_29: float
    emb_30: float

    # ── Language flags (2) ────────────────────────────────────────────────────
    # One-hot encoding of the detected language of the input text.
    # language_hindi=1, language_marathi=0 → Hindi
    # language_hindi=0, language_marathi=1 → Marathi
    # language_hindi=0, language_marathi=0 → English (the baseline/dropped category)
    language_hindi: float    # 1.0 if Hindi, else 0.0
    language_marathi: float  # 1.0 if Marathi, else 0.0

    @field_validator("language_hindi", "language_marathi")
    @classmethod
    def language_must_be_binary(cls, v: float) -> float:
        # Enforce that language flags are strictly 0 or 1 — any other value is invalid.
        if v not in (0.0, 1.0):
            raise ValueError("language flags must be 0 or 1")
        return v


class PredictResponse(BaseModel):
    """
    What the API returns after a successful prediction.

      prediction   — the predicted mental health profile (e.g. "Depression")
      confidence   — probability assigned to the predicted class (0.0 to 1.0)
      probabilities — full probability distribution across all 7 classes
      model        — name of the model that made the prediction ("LightGBM")
      accuracy     — the model's test-set accuracy from training (0.92)
    """
    prediction: str
    confidence: float
    probabilities: dict[str, float]
    model: str
    accuracy: float


# ─── Preprocessing ────────────────────────────────────────────────────────────
# These two functions replicate exactly what the training pipeline did to the
# data before fitting the model. Applying the SAME transforms at inference is
# critical — if we skip them, the feature distributions won't match what the
# model was trained on and predictions will be wrong.

def apply_outlier_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply per-column outlier smoothing using the transformers saved during training.

    During training (pipeline Step 8), each column was tested with 4 strategies:
      - winsorize    → clamp values to [lower, upper] percentile bounds
      - sqrt         → square-root compress large values (shift to non-negative first)
      - yeo-johnson  → power transform that handles both positive and negative values
      - log1p        → log(1+x) compression (not used in this run, but supported)

    The strategy that produced the lowest skew was chosen per column and saved
    in outlier_transformers.joblib. We replay the same strategy here.
    """
    transformers = artifacts["outlier_transformers"]
    df = df.copy()

    for col, info in transformers.items():
        # Skip columns that aren't in the input (e.g., columns dropped by feature selection)
        if col not in df.columns:
            continue
        strategy = info["strategy"]

        if strategy == "yeo-johnson":
            # Use the sklearn PowerTransformer that was fit on training data
            pt = info["fitted_pt"]
            df[col] = pt.transform(df[[col]].values).ravel()

        elif strategy == "sqrt":
            # Shift negative values to zero before taking sqrt (sqrt of negative is undefined)
            min_val = df[col].min()
            shift = abs(min_val) + 1e-6 if min_val < 0 else 0
            df[col] = np.sqrt(df[col] + shift)

        elif strategy == "winsorize":
            # Clip values to the bounds calculated from training data percentiles
            lower = info["lower"]
            upper = info["upper"]
            df[col] = df[col].clip(lower=lower, upper=upper)

    return df


def preprocess(raw: dict) -> np.ndarray:
    """
    Full preprocessing pipeline for a single inference sample.
    Mirrors training pipeline Steps 8 (outlier) and 12 (scaling) exactly.

    Steps:
      1. Wrap the raw feature dict in a single-row DataFrame
      2. Apply per-column outlier smoothing (same strategies as training)
      3. Scale with RobustScaler (fit on training data — robust to outliers)
      4. Select and reorder to exactly the 43 features the model expects

    Returns a (1, 43) numpy array ready to pass to model.predict_proba().
    """
    feature_names = artifacts["feature_names"]  # ordered list of 43 feature names
    df = pd.DataFrame([raw])                     # single-row DataFrame

    # Step 1: Smooth outliers using the saved per-column transformers
    df = apply_outlier_transforms(df)

    # Step 2 & 3: Scale and select features in the exact order the model expects.
    # df[feature_names] reorders columns to match training order before scaling.
    scaler = artifacts["scaler"]
    return scaler.transform(df[feature_names].values)


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    """
    Service info endpoint — returns a summary of the running model.
    Useful as a quick sanity check that the right model is loaded.
    No authentication or input required.
    """
    meta = artifacts.get("metadata", {})
    return {
        "service": "Mindspace Mental Health Classifier",
        "model": meta.get("best_model_name"),
        "accuracy": meta.get("test_metrics", {}).get("accuracy"),
        "classes": meta.get("class_names"),
        "n_features": meta.get("n_features"),
    }


@app.get("/health")
def health():
    """
    Health check endpoint — confirms all artifacts are loaded and the server is ready.
    Returns { "status": "ok", "artifacts_loaded": true } when healthy.
    Returns artifacts_loaded: false if startup failed to load any artifact.
    Typically polled by load balancers or monitoring systems.
    """
    expected_keys = {"model", "scaler", "label_encoder", "encoding", "outlier_transformers", "feature_names", "metadata"}
    return {"status": "ok", "artifacts_loaded": expected_keys.issubset(artifacts.keys())}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Main prediction endpoint — the core of the API.

    Accepts 43 pre-extracted speech/text features as JSON and returns:
      - prediction    → the most likely mental health profile
      - confidence    → probability of the predicted class (0–1)
      - probabilities → full softmax distribution across all 7 classes
      - model / accuracy → metadata about the model that ran inference

    Two try/except blocks handle errors at different stages:
      - 422 Unprocessable Entity → something went wrong during preprocessing
        (bad feature values, unexpected column, etc.)
      - 500 Internal Server Error → model inference itself failed
        (should be very rare if preprocessing succeeded)
    """
    # ── Preprocessing ────────────────────────────────────────────────────────
    try:
        raw = request.model_dump()   # convert Pydantic model → plain Python dict
        X = preprocess(raw)          # outlier smooth → scale → (1, 43) numpy array
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Preprocessing failed: {e}")

    # ── Inference ────────────────────────────────────────────────────────────
    try:
        model = artifacts["model"]           # LightGBM classifier
        le    = artifacts["label_encoder"]   # LabelEncoder: int index → class name
        meta  = artifacts["metadata"]

        # predict_proba returns shape (1, 7) — one probability per class
        proba     = model.predict_proba(X)[0]
        pred_idx  = int(np.argmax(proba))                    # index of highest probability
        pred_label = le.inverse_transform([pred_idx])[0]    # e.g. 2 → "Depression"
        confidence = float(proba[pred_idx])

        # Build a readable dict: {"Anxiety": 0.02, "Depression": 0.94, ...}
        class_names   = le.classes_.tolist()
        probabilities = {cls: round(float(p), 4) for cls, p in zip(class_names, proba)}

        return PredictResponse(
            prediction=pred_label,
            confidence=round(confidence, 4),
            probabilities=probabilities,
            model=meta.get("best_model_name", "LightGBM"),
            accuracy=meta.get("test_metrics", {}).get("accuracy", 0.0),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.get("/model/info")
def model_info():
    """
    Full model metadata endpoint — returns everything saved about the trained model.
    Includes hyperparameters, cross-validation score, and all test-set metrics
    (accuracy, F1 macro/weighted, precision, recall).
    Useful for auditing what model is running and verifying its performance.
    """
    meta = artifacts.get("metadata", {})
    return meta
