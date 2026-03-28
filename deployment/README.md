# Mindspace Voice Agent — Deployment

FastAPI inference server for the Mindspace mental health classifier.

---

## What This Does

Serves the trained LightGBM model (92% accuracy, 7-class mental health classification) via a REST API.
Accepts 43 pre-extracted speech/text features, runs the full preprocessing pipeline internally, and returns a prediction with per-class probabilities.

---

## Project Structure

```
deployment/
├── api_text_to_sentiment.py   # FastAPI app (all routes + preprocessing logic)
├── requirements.txt  # Pinned dependencies for deployment
└── README.md         # This file

pipeline_output/LightGBM_13032026_110356/   # Model artifacts (loaded at startup)
├── best_model.joblib           # Trained LightGBM classifier
├── scaler.joblib               # RobustScaler (fit on 40K training rows)
├── label_encoder.joblib        # Integer → class name decoder
├── encoding_artifacts.joblib   # Categorical encoding maps
├── outlier_transformers.joblib # Per-column outlier smoothing transforms
├── feature_names.json          # Ordered list of 43 selected features
└── model_metadata.json         # Hyperparams, test metrics, class names
```

---

## API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| `GET` | `/` | Service info — model name, accuracy, output classes |
| `GET` | `/health` | Health check — confirms all 7 artifacts are loaded |
| `POST` | `/predict` | Main prediction — returns label + confidence + all probabilities |
| `GET` | `/model/info` | Full model metadata — hyperparams, CV score, test metrics |

---

## Running Locally

**Important:** Always run from the **project root** (`Mindspace-voice-agent/`), not from inside `deployment/`.
Ports `8000` and `8080` may be blocked on Windows — use `9000` instead.

```bash
# From: C:\Users\vicky\OneDrive\Desktop\SCS-projects\Mindspace-voice-agent\

myenv\Scripts\python -m uvicorn deployment.api_text_to_sentiment:app --host 0.0.0.0 --port 9000 --reload
```

Server starts at: `http://localhost:9000`
Swagger UI (interactive docs): `http://localhost:9000/docs`
ReDoc: `http://localhost:9000/redoc`

---

## Input Format

`POST /predict` accepts JSON with **43 float fields**:

### Linguistic / Semantic Scores (19 fields)
```
overall_sentiment_score, semantic_coherence_score, self_reference_density,
future_focus_ratio, positive_emotion_ratio, fear_word_frequency,
sadness_word_frequency, negative_emotion_ratio, uncertainty_word_frequency,
anger_word_frequency, rumination_phrase_frequency, filler_word_frequency,
topic_shift_frequency, total_word_count, avg_sentence_length,
language_model_perplexity, past_focus_ratio, repetition_rate, adjective_ratio
```

### Topic Model Outputs (5 fields)
```
topic_0, topic_1, topic_2, topic_3, topic_4
```

### Embedding Dimensions (17 fields)
```
emb_1, emb_3, emb_4, emb_5, emb_7, emb_8, emb_10, emb_11, emb_12,
emb_14, emb_15, emb_21, emb_22, emb_25, emb_28, emb_29, emb_30
```

### Language Flags (2 fields) — binary 0 or 1
```
language_hindi, language_marathi
```
> `language_hindi=0, language_marathi=0` → English
> `language_hindi=1, language_marathi=0` → Hindi
> `language_hindi=0, language_marathi=1` → Marathi

---

## Example Request

```bash
curl -X POST http://localhost:9000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "overall_sentiment_score": -0.45,
    "semantic_coherence_score": 0.32,
    "self_reference_density": 0.18,
    "future_focus_ratio": 0.05,
    "positive_emotion_ratio": 0.08,
    "fear_word_frequency": 0.12,
    "sadness_word_frequency": 0.21,
    "negative_emotion_ratio": 0.55,
    "uncertainty_word_frequency": 0.09,
    "anger_word_frequency": 0.03,
    "rumination_phrase_frequency": 0.14,
    "filler_word_frequency": 0.07,
    "topic_shift_frequency": 0.02,
    "total_word_count": 120.0,
    "avg_sentence_length": 15.2,
    "language_model_perplexity": 85.3,
    "past_focus_ratio": 0.42,
    "repetition_rate": 0.11,
    "adjective_ratio": 0.09,
    "topic_0": 0.1, "topic_1": 0.3, "topic_2": 0.2, "topic_3": 0.25, "topic_4": 0.15,
    "emb_1": 0.12, "emb_3": -0.05, "emb_4": 0.08, "emb_5": 0.03, "emb_7": -0.11,
    "emb_8": 0.07, "emb_10": 0.14, "emb_11": -0.02, "emb_12": 0.09, "emb_14": 0.05,
    "emb_15": -0.07, "emb_21": 0.11, "emb_22": 0.04, "emb_25": -0.08, "emb_28": 0.06,
    "emb_29": 0.01, "emb_30": -0.03,
    "language_hindi": 0,
    "language_marathi": 0
  }'
```

## Example Response

```json
{
  "prediction": "Depression",
  "confidence": 0.874,
  "probabilities": {
    "Anxiety": 0.042,
    "Bipolar_Mania": 0.011,
    "Depression": 0.874,
    "Normal": 0.008,
    "Phobia": 0.019,
    "Stress": 0.038,
    "Suicidal_Tendency": 0.008
  },
  "model": "LightGBM",
  "accuracy": 0.92
}
```

---

## Preprocessing Pipeline (inside the API)

The API replicates the exact pipeline steps from training — in the same order:

```
Raw JSON input (43 features)
    │
    ▼  Step 1: Outlier Smoothing (outlier_transformers.joblib)
    │   • yeo-johnson  → PowerTransformer.transform() — 58 columns
    │   • sqrt         → np.sqrt(x + shift)           — 3 columns
    │   • winsorize    → clip to [lower, upper]        — 2 columns
    │
    ▼  Step 2: Scaling (scaler.joblib)
    │   • RobustScaler.transform() on all 43 features
    │
    ▼  Step 3: Predict (best_model.joblib)
    │   • LightGBM.predict_proba() → 7-class probabilities
    │
    ▼  Step 4: Decode (label_encoder.joblib)
        • LabelEncoder.inverse_transform() → class name string
```

---

## Output Classes

| Class | Description |
|-------|-------------|
| `Anxiety` | Anxiety disorder indicators |
| `Bipolar_Mania` | Bipolar / manic episode indicators |
| `Depression` | Depressive episode indicators |
| `Normal` | No significant mental health concerns |
| `Phobia` | Phobia-related indicators |
| `Stress` | Stress-related indicators |
| `Suicidal_Tendency` | Suicidal ideation indicators |

---

## Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 0.920 |
| F1 Macro | 0.9181 |
| F1 Weighted | 0.920 |
| Precision Macro | 0.9167 |
| Recall Macro | 0.9202 |

Trained on: 40,000 samples | Tested on: 10,000 samples | Random seed: 42

---

## AWS EC2 Deployment (next steps)

1. Launch EC2 instance (Ubuntu 22.04, t3.medium or above)
2. Install Python 3.10+, clone repo, create venv, install `deployment/requirements.txt`
3. Copy `pipeline_output/` to EC2 (or use S3)
4. Run with gunicorn + uvicorn workers:
   ```bash
   gunicorn deployment.main:app -w 2 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8080
   ```
5. Configure security group to allow inbound TCP on port 8080
6. (Optional) Put Nginx in front as reverse proxy on port 80/443

---

## Changelog

| Date | Change |
|------|--------|
| 2026-03-18 | Initial FastAPI app created (`main.py`) — all 4 endpoints, full preprocessing pipeline, tested locally on port 8080 |
