# Mindspace Voice Agent — Deployment

Two FastAPI inference servers for mental health classification — one for text features, one for voice/acoustic features.

---

## What This Does

Serves two trained ML models via REST APIs:

| API | Model | Input | Output | Accuracy |
|-----|-------|-------|--------|----------|
| **Text API** (`api_text_to_sentiment.py`) | LightGBM | 43 linguistic/text features | 7-class prediction | 92.0% |
| **Voice API** (`api_voice_to_sentiment.py`) | XGBoost | 1,351 OpenSMILE acoustic features | 6-class prediction | 60.3% |

Both APIs accept pre-extracted features, run the full preprocessing pipeline internally (outlier smoothing → scaling), and return a prediction with per-class probabilities.

---

## Project Structure

```
deployment/
├── api_text_to_sentiment.py    # Text API — LightGBM, 43 features, 7 classes
├── api_voice_to_sentiment.py   # Voice API — XGBoost, 1351 features, 6 classes
├── .env                        # API_KEY (never commit to git)
├── requirements.txt            # Pinned dependencies for deployment
└── README.md                   # This file

pipeline_output/
├── LightGBM_13032026_110356/   # Text model artifacts
│   ├── best_model.joblib           # Trained LightGBM classifier
│   ├── scaler.joblib               # RobustScaler (fit on 40K training rows)
│   ├── label_encoder.joblib        # Integer → class name decoder
│   ├── encoding_artifacts.joblib   # Categorical encoding maps
│   ├── outlier_transformers.joblib # Per-column outlier smoothing transforms
│   ├── feature_names.json          # Ordered list of 43 selected features
│   └── model_metadata.json         # Hyperparams, test metrics, class names
│
└── XGBoost_27032026_152209/    # Voice model artifacts
    ├── best_model.joblib           # Trained XGBoost classifier
    ├── scaler.joblib               # RobustScaler (fit on training rows)
    ├── label_encoder.joblib        # Integer → class name decoder
    ├── encoding_artifacts.joblib   # Categorical encoding maps
    ├── outlier_transformers.joblib # Per-column outlier smoothing transforms
    ├── feature_names.json          # Ordered list of 1,351 selected features
    └── model_metadata.json         # Hyperparams, test metrics, class names
```

---

## API Endpoints

Both APIs share the same 4 endpoints:

| Method | Route | Auth | Description |
|--------|-------|------|-------------|
| `GET` | `/` | No | Service info — model name, accuracy, output classes |
| `GET` | `/health` | No | Health check — confirms all 7 artifacts are loaded |
| `POST` | `/predict` | `X-API-Key` | Main prediction — returns label + confidence + all probabilities |
| `GET` | `/model/info` | `X-API-Key` | Full model metadata — hyperparams, CV score, test metrics |

---

## Running Locally

**Important:** Always run from the **project root** (`Mindspace-voice-agent/`), not from inside `deployment/`.

### Text API (LightGBM — port 9000)

```bash
# From project root:
uvicorn deployment.api_text_to_sentiment:app --reload --port 9000
```

- Server: `http://localhost:9000`
- Swagger docs: `http://localhost:9000/docs`

### Voice API (XGBoost — port 9100)

```bash
# From project root:
uvicorn deployment.api_voice_to_sentiment:app --reload --port 9100
```

- Server: `http://localhost:9100`
- Swagger docs: `http://localhost:9100/docs`

### Running from inside `deployment/` folder

```bash
cd deployment
uvicorn api_text_to_sentiment:app --reload --port 9000
uvicorn api_voice_to_sentiment:app --reload --port 9100
```

> **Note:** On Windows, some ports (e.g. 9001) may be reserved by Hyper-V. If you get `[Errno 13] Permission denied`, pick a different port. Run `netsh interface ipv4 show excludedportrange protocol=tcp` to see reserved ports.

---

## Authentication

Both APIs require an `X-API-Key` header on protected routes (`/predict`, `/model/info`).

The key is loaded from `deployment/.env`:

```
API_KEY=your-secret-key-here
```

Example authenticated request:

```bash
curl -H "X-API-Key: YOUR_KEY" http://localhost:9000/model/info
```

---

## Text API — Input Format

`POST /predict` accepts JSON with **43 float fields** (flat body, no wrapper):

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

### Example Request (Text API)

```bash
curl -X POST http://localhost:9000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_KEY" \
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

### Example Response (Text API)

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

## Voice API — Input Format

`POST /predict` accepts JSON with a **`features` key** containing a flat dict of all 1,351 OpenSMILE acoustic feature values:

```json
{
  "features": {
    "pcm_fftMag_fband250-650_sma_de_pctlrange0-1": 0.0042,
    "mfcc_sma_de[11]_meanPeakDist": 12.34,
    "...": "... (all 1,351 features required)"
  }
}
```

The full list of required feature names can be retrieved via `GET /model/info` under the `feature_names` key.

Feature categories include: MFCC coefficients, spectral features (entropy, rolloff, harmonicity, kurtosis, slope, flux, variance), pitch (F0), shimmer, jitter, voicing, auditory spectrum (audSpec), RMS energy, zero-crossing rate, psychoacoustic sharpness, HNR, and their delta/statistical functionals.

### Example Response (Voice API)

```json
{
  "prediction": "depression",
  "confidence": 0.6521,
  "probabilities": {
    "anxiety": 0.0832,
    "bipolar": 0.0411,
    "depression": 0.6521,
    "normal": 0.0234,
    "stress": 0.1147,
    "suicidal": 0.0855
  },
  "model": "XGBoost",
  "accuracy": 0.6033
}
```

---

## Preprocessing Pipeline (inside both APIs)

Both APIs replicate the exact pipeline steps from training — in the same order:

```
Raw JSON input
    │
    ▼  Step 1: Outlier Smoothing (outlier_transformers.joblib)
    │   • yeo-johnson  → PowerTransformer.transform()
    │   • sqrt         → np.sqrt(clip(x, 0))
    │   • log1p        → np.log1p(clip(x, 0))
    │   • winsorize    → clip to [lower, upper]
    │
    ▼  Step 2: Scaling (scaler.joblib)
    │   • RobustScaler.transform() on all features
    │
    ▼  Step 3: Predict (best_model.joblib)
    │   • model.predict_proba() → class probabilities
    │
    ▼  Step 4: Decode (label_encoder.joblib)
        • LabelEncoder.inverse_transform() → class name string
```

---

## Output Classes

### Text API — 7 Classes

| Class | Description |
|-------|-------------|
| `Anxiety` | Anxiety disorder indicators |
| `Bipolar_Mania` | Bipolar / manic episode indicators |
| `Depression` | Depressive episode indicators |
| `Normal` | No significant mental health concerns |
| `Phobia` | Phobia-related indicators |
| `Stress` | Stress-related indicators |
| `Suicidal_Tendency` | Suicidal ideation indicators |

### Voice API — 6 Classes

| Class | Description |
|-------|-------------|
| `anxiety` | Anxiety disorder indicators |
| `bipolar` | Bipolar / manic episode indicators |
| `depression` | Depressive episode indicators |
| `normal` | No significant mental health concerns |
| `stress` | Stress-related indicators |
| `suicidal` | Suicidal ideation indicators |

---

## Model Performance

### Text API — LightGBM

| Metric | Score |
|--------|-------|
| Accuracy | 0.920 |
| F1 Macro | 0.918 |
| F1 Weighted | 0.920 |
| Precision Macro | 0.917 |
| Recall Macro | 0.920 |

Trained on: 40,000 samples | Tested on: 10,000 samples

### Voice API — XGBoost

| Metric | Score |
|--------|-------|
| Accuracy | 0.603 |
| F1 Macro | 0.563 |
| F1 Weighted | 0.550 |
| Precision Macro | 0.633 |
| Recall Macro | 0.619 |

Trained on: 2,400 samples | Tested on: 600 samples

---

## AWS EC2 Deployment (next steps)

1. Launch EC2 instance (Ubuntu 22.04, t3.medium or above)
2. Install Python 3.10+, clone repo, create venv, install `deployment/requirements.txt`
3. Copy `pipeline_output/` to EC2 (or use S3)
4. Run with gunicorn + uvicorn workers:
   ```bash
   # Text API
   gunicorn deployment.api_text_to_sentiment:app -w 2 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:9000
   # Voice API
   gunicorn deployment.api_voice_to_sentiment:app -w 2 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:9100
   ```
5. Configure security group to allow inbound TCP on ports 9000 and 9100
6. (Optional) Put Nginx in front as reverse proxy on port 80/443

---

## Changelog

| Date | Change |
|------|--------|
| 2026-03-18 | Text API created (`api_text_to_sentiment.py`) — LightGBM, 43 features, 7 classes, all 4 endpoints, tested on port 9000 |
| 2026-03-28 | Voice API created (`api_voice_to_sentiment.py`) — XGBoost, 1,351 acoustic features, 6 classes, all 4 endpoints, tested on port 9100 |
