# Mindspace — Backend API Integration Guide

> **Document Purpose:** Single source of truth for connecting all seven microservices to the Spring Boot backend.  
> **Target Audience:** Backend Developer  
> **Last Updated:** April 2026  
> **VPS Base Address:** `http://88.222.212.15`

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [API Index & Port Map](#2-api-index--port-map)
3. [Authentication Model](#3-authentication-model)
4. [API 1 — Facial Feature Extraction API (Port 8010)](#4-api-1--facial-feature-extraction-api-port-8010)
5. [API 2 — Facial Risk Scoring API (Port 8011)](#5-api-2--facial-risk-scoring-api-port-8011)
6. [API 3 — Psychological Text Analysis API (Port 8025)](#6-api-3--psychological-text-analysis-api-port-8025)
7. [API 4 — Mindspace Text Classifier API (Port 9000)](#7-api-4--mindspace-text-classifier-api-port-9000)
8. [API 5 — Mindspace Voice Classifier API (Port 9100)](#8-api-5--mindspace-voice-classifier-api-port-9100)
9. [API 6 — Multimodal Pipeline Inference API (Port 8000)](#9-api-6--multimodal-pipeline-inference-api-port-8000)
10. [API 7 — Audio Feature Extraction API (Port 8013)](#10-api-7--audio-feature-extraction-api-port-8013)
11. [End-to-End Integration Workflow](#11-end-to-end-integration-workflow)
12. [Common Error Reference](#12-common-error-reference)
13. [Quick Connectivity Checklist](#13-quick-connectivity-checklist)

---

## 1. Architecture Overview

The Mindspace platform is a **multimodal mental health analysis system**. It processes input from three modalities — **video (face)**, **text (conversation)**, and **voice (audio)** — and produces structured mental health risk assessments. All seven services run as isolated Docker containers on a single VPS and are accessed over HTTP.

### Modality Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        Spring Boot Backend                      │
└────────────┬──────────────────┬────────────────┬───────────────-┘
             │                  │                │

     ┌───────▼──────┐  ┌────────▼───────┐  ┌────▼──────────────┐
     │  FACE TRACK  │  │   TEXT TRACK   │  │   VOICE TRACK     │
     │              │  │                │  │                   │
     │  Port 8010   │  │   Port 8025    │  │   Port 8013       │
     │  Extraction  │  │  Psych. Text   │  │  Audio Feature    │
     │     API      │  │  Analysis API  │  │  Extraction API   │
     │      │       │  │       │        │  │        │          │
     │  Port 8011   │  │   Port 9000    │  │   Port 9100       │
     │   Scoring    │  │  Text Classif. │  │  Voice Classif.   │
     │     API      │  │     API        │  │     API           │
     └──────────────┘  └────────────────┘  └───────────────────┘
                                │
                        ┌───────▼────────┐
                        │   Port 8000    │
                        │  Multimodal    │
                        │  Pipeline API  │
                        └────────────────┘
```

### What Each Modality Does

| Modality | Services Involved | Output |
|---|---|---|
| **Video / Face** | API 1 (Port 8010) → API 2 (Port 8011) | Behavioural feature vector → Risk label + probabilities |
| **Text** | API 3 (Port 8025) extracts 49+ features → API 4 (Port 9000) classifies | Mental health category + confidence |
| **Voice** | API 7 (Port 8013) extracts 6000+ acoustic features → API 5 (Port 9100) classifies | Mental health category + confidence |
| **Multimodal Fusion** | API 6 (Port 8000) takes combined feature vector | Final dominant class across all modalities |

---

## 2. API Index & Port Map

| # | Service Name | VPS Base URL | Port | Swagger UI | Auth Required |
|---|---|---|---|---|---|
| 1 | Facial Feature Extraction API | `http://88.222.212.15:8010` | 8010 | `/docs` | Yes (`X-API-Key`) |
| 2 | Facial Risk Scoring API | `http://88.222.212.15:8011` | 8011 | `/docs` | Yes (`X-API-Key`) |
| 3 | Psychological Text Analysis API | `http://88.222.212.15:8025` | 8025 | `/docs` | Yes (`x-api-key`) |
| 4 | Mindspace Text Classifier API | `http://88.222.212.15:9000` | 9000 | `/docs` | Yes (`X-API-Key`) |
| 5 | Mindspace Voice Classifier API | `http://88.222.212.15:9100` | 9100 | `/docs` | Yes (`X-API-Key`) |
| 6 | Multimodal Pipeline Inference API | `http://88.222.212.15:8000` | 8000 | N/A | Yes (`X-API-Key`) |
| 7 | Audio Feature Extraction API | `http://88.222.212.15:8013` | 8013 | `/docs` | Yes (`x-api-key`) |

> **Swagger UI access pattern:** `http://88.222.212.15:{port}/docs`  
> Example: `http://88.222.212.15:8010/docs`

---

## 3. Authentication Model

All protected endpoints across **all seven services** use a **static API key** passed as an HTTP request header. There is no OAuth, JWT, or session-based auth.

### Header Name Variants

> ⚠️ Header names are **case-sensitive** in some HTTP clients. Be sure to match exactly:

| API | Header Name |
|---|---|
| API 1 — Face Extraction | `X-API-Key` |
| API 2 — Face Scoring | `X-API-Key` |
| API 3 — Text Analysis | `x-api-key` (lowercase) |
| API 4 — Text Classifier | `X-API-Key` |
| API 5 — Voice Classifier | `X-API-Key` |
| API 6 — Multimodal Pipeline | `X-API-Key` |
| API 7 — Audio Extraction | `x-api-key` (lowercase) |

### How Keys Are Set

Each Docker container reads its API key from a `.env` file at startup. The environment variable names differ per service:

| API | Env Variable Name |
|---|---|
| API 1 — Face Extraction | `EXTRACTION_API_KEY` |
| API 2 — Face Scoring | `SCORING_API_KEY` |
| API 3 — Text Analysis | `API_KEY` (+ `ANTHROPIC_API_KEY` for LLM calls) |
| API 4 — Text Classifier | `API_KEY` |
| API 5 — Voice Classifier | `API_KEY` |
| API 6 — Multimodal Pipeline | `MODEL_API_KEY` |
| API 7 — Audio Extraction | `API_KEY` |

> The backend should store all API keys securely (e.g., in `application.properties` or environment variables injected at deploy time) and never expose them in logs or responses.

---

## 4. API 1 — Facial Feature Extraction API (Port 8010)

### Purpose

Accepts an **uploaded video file**, runs it through a **MediaPipe Face Mesh** pipeline, and returns a **session-level ML feature vector** of 63 behavioural features. This vector represents how a person's face behaved during the video — not who they are (privacy-safe by design).

The 6 core behaviours extracted are:

| Feature | What It Represents |
|---|---|
| `S_AU12` | Smile intensity (lip corner elevation) |
| `S_AUVar` | Facial expressivity / animation level |
| `S_HeadVelocity` | Restlessness / head scanning behaviour |
| `S_EyeContact` | Frontal gaze / social engagement |
| `S_BlinkRate` | Cognitive load, stress, anxiety indicators |
| `S_ResponseLatency` | Hesitation / cognitive processing speed |

All values are **normalised to [0.0 – 1.0]** relative to the person's own 30-second session baseline. `0.5` = neutral/baseline; `> 0.5` = elevated; `< 0.5` = suppressed.

### Base URL

```
http://88.222.212.15:8010
```

### Endpoints

---

#### `GET /health`

Checks if the service is up. No auth required.

**Request:**
```http
GET http://88.222.212.15:8010/health
```

**Response:**
```json
{
  "status": "ok",
  "service": "Facial Extraction API"
}
```

---

#### `POST /extract/video`

Uploads a video file and extracts the behavioural feature vector.

**Auth:** `X-API-Key` header required  
**Content-Type:** `multipart/form-data`

**Form Field:**

| Field | Type | Required | Description |
|---|---|---|---|
| `video` | file | ✅ Yes | Video file to process (e.g., `.mp4`, `.avi`) |

**Query Parameters:**

| Parameter | Type | Default | Options / Notes |
|---|---|---|---|
| `mode` | string | `balanced` | `accurate` / `balanced` / `fast` — affects frame sampling density |
| `frame_stride` | int | `0` | `0` means use mode default; `>0` manually overrides frames to skip |
| `min_duration_seconds` | float | `150.0` | Minimum required video length in seconds |
| `allow_short` | bool | `false` | Set `true` to process videos shorter than `min_duration_seconds` |
| `model_dir` | string | `reports/model_training/run_20260324_171117` | Path to model artifacts (leave default unless retrained) |
| `label_col` | string | `condition_label` | Label column used in training (leave default) |

**Example Request (cURL):**
```bash
curl -X POST "http://88.222.212.15:8010/extract/video?mode=balanced&allow_short=true" \
  -H "X-API-Key: YOUR_EXTRACTION_API_KEY" \
  -F "video=@/path/to/session_video.mp4"
```

**Success Response (HTTP 200):**
```json
{
  "session_id": "8d0c5f7e62d14e2cbe64b70842d4f4da",
  "vector_feature_count": 63
}
```

> Store the `session_id` — you need it to retrieve the actual feature vector in the next step.

**Server-side Artifacts Created (inside container):**

| Path | Description |
|---|---|
| `reports/api_sessions/{session_id}.json` | Full session payload |
| `reports/api_vectors/{session_id}.json` | Vector-only payload |
| `reports/api_raw_features/api_raw_features_{session_id}.csv` | Raw feature CSV |

> Uploaded video is **not permanently stored** — deleted after extraction completes.

---

#### `GET /extract/session/{session_id}/vector`

Retrieves the previously computed feature vector for a session.

**Auth:** `X-API-Key` header required  
**Path Parameter:** `session_id` — obtained from `/extract/video` response

**Example Request:**
```bash
curl -X GET "http://88.222.212.15:8010/extract/session/8d0c5f7e62d14e2cbe64b70842d4f4da/vector" \
  -H "X-API-Key: YOUR_EXTRACTION_API_KEY"
```

**Success Response (HTTP 200):**
```json
{
  "session_id": "8d0c5f7e62d14e2cbe64b70842d4f4da",
  "vector": {
    "au12_mean_amplitude__mean": 0.000123,
    "au12_mean_amplitude__std": 0.000045,
    "au12_variance__slope": 0.000067
    // ... 63 features total
  }
}
```

> This `vector` object is the **direct input** for API 2 (`/score`).

### Recommended Backend Flow (Face Modality)

```
1. Backend receives video from client
2. POST /extract/video  →  get session_id
3. GET /extract/session/{session_id}/vector  →  get vector{}
4. Pass vector{} to API 2 POST /score
5. Receive risk label + probabilities
```

---

## 5. API 2 — Facial Risk Scoring API (Port 8011)

### Purpose

Takes the **63-feature behavioural vector** produced by API 1 and runs it through a trained classification model. Returns a **dominant mental health risk label** and a **probability distribution** across all risk classes.

**Classes:**

- `stress`
- `depression`
- `suicidal_tendency`
- *(model may include additional classes — check `/health` response)*

### Base URL

```
http://88.222.212.15:8011
```

### Endpoints

---

#### `GET /health`

No auth required.

**Response:**
```json
{
  "status": "ok",
  "service": "Facial Risk Scoring API",
  "model_loaded": true
}
```

> If `model_loaded` is `false`, the API is running but the ML model failed to load. Predictions will fail.

---

#### `POST /score`

Runs inference on a feature vector.

**Auth:** `X-API-Key` header required  
**Content-Type:** `application/json`

**Request Body:**
```json
{
  "vector": {
    "au12_mean_amplitude__slope": 0.000123,
    "au12_variance__min": 0.000045
    // ... all 63 features from API 1
  }
}
```

> The `vector` object here is exactly the `vector` field from the API 1 GET response. Pass it directly.

**Success Response (HTTP 200):**
```json
{
  "dominant_risk": {
    "label": "stress",
    "probability": 0.9475
  },
  "other_risks": [
    {
      "label": "suicidal_tendency",
      "probability": 0.0506
    },
    {
      "label": "depression",
      "probability": 0.0006
    }
  ]
}
```

**Example cURL:**
```bash
curl -X POST "http://88.222.212.15:8011/score" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_SCORING_API_KEY" \
  -d '{"vector": {"au12_mean_amplitude__slope": 0.000123, "au12_variance__min": 0.000045}}'
```

---

## 6. API 3 — Psychological Text Analysis API (Port 8025)

### Purpose

Accepts a **Client–Assistant conversation transcript** and performs deep NLP + LLM-based psychological analysis. Returns **49+ structured linguistic and emotional features** about the client's mental state derived from their speech patterns in the conversation.

This service uses **Anthropic Claude** (LLM) internally for psychological reasoning, combined with **Stanza NLP** and **Sentence Transformers** for feature extraction.

**Features Extracted (examples):**

| Feature | Description |
|---|---|
| `total_word_count` | Total words spoken by client |
| `semantic_coherence_score` | Logical consistency of client speech (0–1) |
| `catastrophizing_score` | Presence of catastrophic thinking patterns |
| Emotional patterns | Positive/negative sentiment ratios |
| Cognitive distortions | Helplessness, hopelessness indicators |
| Self-focus & time orientation | Pronoun usage patterns, past vs. future language |
| Topic shifts | Discourse coherence across conversation |

### Base URL

```
http://88.222.212.15:8025
```

### Endpoints

---

#### `POST /analyze`

Analyzes a conversation and returns psychological features.

**Auth:** `x-api-key` header required (lowercase)  
**Content-Type:** `application/json`

**Request Body:**
```json
{
  "conversation": "Client: I feel stressed and anxious Assistant: Tell me more about that feeling Client: I don't know, everything feels out of control"
}
```

**Input Format Rules (Critical):**
- The conversation must be a **single string** (no newlines in JSON — use space or `\n` escape)
- Must use explicit `Client:` and `Assistant:` labels
- Only the **Client** portions are analysed — Assistant text is used for context only
- Do **not** mix languages within a single input
- Supports **English, Marathi, and Hindi** (separately)

**Success Response (HTTP 200):**
```json
{
  "status": "success",
  "client_text": "I feel stressed and anxious ... everything feels out of control",
  "analysis": {
    "total_word_count": 120,
    "semantic_coherence_score": 0.82,
    "catastrophizing_score": 0.4
    // ... 49+ total features
  },
  "latency": {
    "total_time": 2.1
  }
}
```

**Example cURL:**
```bash
curl -X POST "http://88.222.212.15:8025/analyze" \
  -H "Content-Type: application/json" \
  -H "x-api-key: YOUR_TEXT_ANALYSIS_API_KEY" \
  -d '{"conversation": "Client: I feel very stressed and overwhelmed Assistant: Can you tell me more about what is happening?"}'
```

> ⚠️ **Performance Note:** This service requires 6 GB+ RAM. It uses heavy NLP models (Stanza, Sentence Transformers) plus live LLM calls. Response latency is typically 2–5 seconds per request. Plan for appropriate timeouts on the backend (suggest 30s minimum).

### Recommended Backend Flow (Text Modality — Option A)

```
1. Backend receives conversation transcript
2. POST /analyze  →  get 49+ feature dict from analysis{}
3. Pass features to API 4 POST /predict  →  get mental health class
```

---

## 7. API 4 — Mindspace Text Classifier API (Port 9000)

### Purpose

Takes **43 pre-extracted text/speech features** as input and classifies the user into one of **7 mental health categories** using a trained **LightGBM model** (reported accuracy: 92%).

> This API does **not** extract features from raw text. It classifies from a numeric feature vector. The features are expected to come from a prior text analysis step (e.g., API 3 output, or a pre-built feature extraction pipeline).

**Classes:**

| Label | Description |
|---|---|
| `Anxiety` | Generalised anxiety indicators |
| `Bipolar_Mania` | Manic episode indicators |
| `Depression` | Depressive state indicators |
| `Normal` | No significant mental health risk |
| `Phobia` | Phobia-related patterns |
| `Stress` | Acute or chronic stress indicators |
| `Suicidal_Tendency` | Suicidal ideation indicators |

### Base URL

```
http://88.222.212.15:9000
```

### Endpoints

---

#### `GET /`

Returns basic service info. No auth required.

**Response:**
```json
{
  "service": "Mindspace Mental Health Classifier",
  "model": "LightGBM",
  "accuracy": 0.92,
  "classes": ["Anxiety", "Bipolar_Mania", "Depression", "Normal", "Phobia", "Stress", "Suicidal_Tendency"],
  "n_features": 43
}
```

---

#### `GET /health`

Readiness check. No auth required.

**Response:**
```json
{
  "status": "ok",
  "artifacts_loaded": true
}
```

---

#### `POST /predict`

Main inference endpoint.

**Auth:** `X-API-Key` header required  
**Content-Type:** `application/json`

**Request Body:**

All 43 numeric features must be present. Key constraints:
- `language_hindi` and `language_marathi` must be `0.0` or `1.0`
- `topic_0` through `topic_4` must be in range `[0, 1]`

```json
{
  "feature_1": 0.234,
  "feature_2": -0.112,
  "language_hindi": 0.0,
  "language_marathi": 1.0,
  "topic_0": 0.312,
  "topic_1": 0.188
  // ... all 43 fields required
}
```

> For the exact list of 43 feature names, refer to the sample payloads in the `demo-api-input-data-sample/` directory of the repo (e.g., `depression_sample_1.json`).

**Success Response (HTTP 200):**
```json
{
  "prediction": "Depression",
  "confidence": 0.9412,
  "probabilities": {
    "Anxiety": 0.0101,
    "Bipolar_Mania": 0.0021,
    "Depression": 0.9412,
    "Normal": 0.015,
    "Phobia": 0.0061,
    "Stress": 0.022,
    "Suicidal_Tendency": 0.0035
  },
  "model": "LightGBM",
  "accuracy": 0.92
}
```

**Example cURL:**
```bash
curl -X POST "http://88.222.212.15:9000/predict" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_TEXT_CLASSIFIER_API_KEY" \
  --data-binary "@depression_sample_1.json"
```

---

#### `GET /model/info`

Returns full model metadata including training metrics.

**Auth:** `X-API-Key` header required

---

## 8. API 5 — Mindspace Voice Classifier API (Port 9100)

### Purpose

Takes **1,351 pre-extracted acoustic features** from a voice recording and classifies it into one of **6 mental health categories** using a trained **XGBoost model** (test accuracy: 60.33%).

> This API does **not** accept raw audio. It expects a numeric feature vector already extracted by an upstream acoustic feature extractor (e.g., API 7 output).

> ⚠️ The lower accuracy (60%) is expected for voice-only modality. Voice features contribute as one signal in a multimodal pipeline — not as the sole classifier.

**Classes:**

| Label | Description |
|---|---|
| `anxiety` | Anxiety acoustic patterns |
| `bipolar` | Bipolar disorder patterns |
| `depression` | Depressive state patterns |
| `normal` | No significant risk indicators |
| `stress` | Stress-related acoustic patterns |
| `suicidal` | Suicidal risk acoustic patterns |

### Base URL

```
http://88.222.212.15:9100
```

### Endpoints

---

#### `GET /`

No auth required.

**Response:**
```json
{
  "service": "Mindspace Mental Health Classifier — Voice",
  "model": "XGBoost",
  "accuracy": 0.6033,
  "classes": ["anxiety", "bipolar", "depression", "normal", "stress", "suicidal"],
  "n_features": 1351
}
```

---

#### `GET /health`

No auth required.

**Response:**
```json
{
  "status": "ok",
  "artifacts_loaded": true
}
```

---

#### `POST /predict`

Main inference endpoint.

**Auth:** `X-API-Key` header required  
**Content-Type:** `application/json`

**Request Body:**
```json
{
  "features": {
    "feature_name_1": 0.123,
    "feature_name_2": -0.456
    // ... all 1,351 acoustic feature names required
  }
}
```

> The `features` dict here maps directly to the output of API 7's `POST /extract` response — specifically the `features` object in that response.

**Success Response (HTTP 200):**
```json
{
  "prediction": "normal",
  "confidence": 0.8123,
  "probabilities": {
    "anxiety": 0.0312,
    "bipolar": 0.0441,
    "depression": 0.0527,
    "normal": 0.8123,
    "stress": 0.0419,
    "suicidal": 0.0178
  },
  "model": "XGBoost",
  "accuracy": 0.6033
}
```

**Example cURL:**
```bash
curl -X POST "http://88.222.212.15:9100/predict" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_VOICE_CLASSIFIER_API_KEY" \
  --data @voice_normal_sample_1.json
```

---

#### `GET /model/info`

Returns model metadata.

**Auth:** `X-API-Key` header required

> **Note:** CORS is currently open to all origins in this service. Recommend tightening before production.

---

## 9. API 6 — Multimodal Pipeline Inference API (Port 8000)

### Purpose

The **final fusion layer** of the Mindspace system. Takes a **combined feature vector** (up to 102 features selected from all modalities) and runs it through a **Gradient Boosting classifier** to produce a single unified mental health risk prediction across all modalities.

- **Model:** Gradient Boosting
- **Accuracy:** 78.85%
- **F1 Macro:** 78.81%
- **Input features:** 102 (selected by a two-stage feature selection pipeline from the full multimodal dataset)

**Classes:**

| Label |
|---|
| `Anxiety` |
| `Depression` |
| `Mania` |
| `Normal` |
| `Stress` |
| `Suicidal` |

### Base URL

```
http://88.222.212.15:8000
```

### Endpoints

---

#### `GET /health`

No auth required.

**Response:** Standard health check confirming service is running.

---

#### `GET /metadata`

Returns model metadata and required feature schema.

**Auth:** `X-API-Key` header required  
**Header:** `X-API-Key`

**Example cURL:**
```bash
curl -H "X-API-Key: YOUR_MULTIMODAL_API_KEY" http://88.222.212.15:8000/metadata
```

---

#### `GET /sample-payload`

Returns a full sample JSON payload showing all 102 required feature names and example values.

**Auth:** `X-API-Key` header required  

> **Important:** Always use this endpoint to get the exact feature schema before constructing `/predict` requests. Feature names and their expected ranges are listed here.

**Example cURL:**
```bash
curl -H "X-API-Key: YOUR_MULTIMODAL_API_KEY" http://88.222.212.15:8000/sample-payload
```

---

#### `POST /predict`

Main multimodal inference endpoint.

**Auth:** `X-API-Key` header required  
**Content-Type:** `application/json`

**Request Body:**

Use the output of `GET /sample-payload` to understand the feature schema. The body is a flat JSON object with all 102 feature keys filled in with numeric values.

```json
{
  "feature_1": 0.23,
  "feature_2": -0.11,
  // ... 102 total features selected by the training pipeline
}
```

**Success Response:**
```json
{
  "prediction": "Stress",
  "confidence": 0.7885,
  "probabilities": {
    "Anxiety": 0.04,
    "Depression": 0.07,
    "Mania": 0.02,
    "Normal": 0.06,
    "Stress": 0.7885,
    "Suicidal": 0.03
  }
}
```

---

#### `POST /predict-batch`

Batch version of `/predict` — accepts multiple feature vectors in one request.

**Auth:** `X-API-Key` header required  
**Content-Type:** `application/json`

```json
[
  { "feature_1": 0.23, "feature_2": -0.11 },
  { "feature_1": 0.55, "feature_2": 0.32 }
]
```

> Use this for processing multiple sessions at once to reduce round-trip overhead.

---

## 10. API 7 — Audio Feature Extraction API (Port 8013)

### Purpose

Accepts a raw `.wav` audio file, resamples it to 44,100 Hz, and runs it through the **OpenSMILE ComParE 2016** feature extraction pipeline to produce **6,000+ acoustic features** per audio file. These features are the direct input to API 5 (Voice Classifier).

Additionally, this service can apply **PCA + K-Means clustering** to assign the audio file to one of 6 acoustic mental health condition groups (cluster-based, not clinical diagnosis).

**Acoustic Features Extracted (examples):**

| Feature | Description |
|---|---|
| `mfcc_1` to `mfcc_N` | Mel-Frequency Cepstral Coefficients (voice timbre) |
| `pitch_mean` | Average fundamental frequency |
| `energy_rms` | Root mean square energy (loudness) |
| `zcr_mean` | Zero-crossing rate (voice roughness/tonality) |
| *(6,000+ total)* | Full ComParE 2016 functional set |

**Cluster Labels (for reference):**

| Cluster | Label | Acoustic Pattern |
|---|---|---|
| 0 | `normal` | Balanced pitch, energy, and speech rate |
| 1 | `depression_like` | Low energy, flat pitch, slow speech |
| 2 | `anxiety_like` | High pitch, fast speech, high ZCR |
| 3 | `stress_like` | High energy, irregular pitch, tense voice |
| 4 | `bipolar_like` | Highly variable pitch and energy |
| 5 | `suicidal_like` | Very low energy, monotone, long pauses |

> ⚠️ Cluster labels are **acoustic pattern interpretations only**, not clinical diagnoses.

### Base URL

```
http://88.222.212.15:8013
```

### Endpoints

---

#### `GET /`

Health check. No auth required.

**Response:**
```json
{
  "message": "API is running successfully"
}
```

---

#### `POST /extract`

Uploads a `.wav` file, extracts acoustic features, and saves them server-side.

**Auth:** `x-api-key` header required (lowercase)  
**Content-Type:** `multipart/form-data`

**Form Field:**

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | `.wav` file | ✅ Yes | Audio file to analyse. Must be `.wav` format. |

**Header:**
```
x-api-key: YOUR_AUDIO_API_KEY
```

**Success Response (HTTP 200):**
```json
{
  "status": "success",
  "file_id": "6428c45b",
  "filename": "sample_audio.wav",
  "sampling_rate": 44100,
  "feature_count": 6373,
  "csv_saved_to": "output/6428c45b_features.csv",
  "json_saved_to": "output/6428c45b_features.json",
  "features": {
    "mfcc_1": -243.56,
    "mfcc_2": 87.34,
    "pitch_mean": 187.43,
    "energy_rms": 0.032,
    "zcr_mean": 0.085
    // ... 6000+ features
  }
}
```

> Store the `file_id` for retrieval later. The `features` object in this response is the **direct input** for API 5's `/predict` endpoint (use the `features` dict as the `features` field in the request body). Note: API 5 expects only 1,351 features — ensure the feature set passed matches the model's training schema.

**Example cURL:**
```bash
curl -X POST "http://88.222.212.15:8013/extract" \
  -H "x-api-key: YOUR_AUDIO_API_KEY" \
  -F "file=@/path/to/session_audio.wav"
```

**Example Python:**
```python
import requests

API_URL = "http://88.222.212.15:8013"
API_KEY = "YOUR_AUDIO_API_KEY"
HEADERS = {"x-api-key": API_KEY}

with open("session_audio.wav", "rb") as f:
    files = {"file": ("session_audio.wav", f, "audio/wav")}
    response = requests.post(f"{API_URL}/extract", headers=HEADERS, files=files)

result = response.json()
file_id = result["file_id"]
features = result["features"]
```

---

#### `GET /features/{file_id}`

Retrieves previously extracted features by file ID.

**Auth:** `x-api-key` header required  
**Path Parameter:** `file_id` — obtained from `/extract` response

**Response:**
```json
{
  "status": "success",
  "file_id": "6428c45b",
  "features": {
    "mfcc_1": -243.56,
    "mfcc_2": 87.34
    // ... all extracted features
  }
}
```

**Error (file not found):**
```json
{
  "detail": "Feature output not found"
}
```

---

## 11. End-to-End Integration Workflow

This section defines how the Spring Boot backend should orchestrate all 7 APIs for a complete multimodal session.

### Full Session Flow

```
CLIENT SUBMITS:
  ├── Video file (.mp4)
  ├── Conversation transcript (text)
  └── Voice recording (.wav)

SPRING BOOT BACKEND:

  ══════════ FACE TRACK ══════════
  [1] POST http://88.222.212.15:8010/extract/video
      → session_id returned

  [2] GET  http://88.222.212.15:8010/extract/session/{session_id}/vector
      → facial_vector{} (63 features)

  [3] POST http://88.222.212.15:8011/score
      Body: { "vector": facial_vector{} }
      → face_risk_label, face_probabilities{}

  ══════════ TEXT TRACK ══════════
  [4] POST http://88.222.212.15:8025/analyze
      Body: { "conversation": "Client: ... Assistant: ..." }
      → text_features{} (49+ features)

  [5] POST http://88.222.212.15:9000/predict
      Body: text_features mapped to 43 required fields
      → text_risk_label, text_probabilities{}

  ══════════ VOICE TRACK ══════════
  [6] POST http://88.222.212.15:8013/extract
      Body: multipart .wav file
      → audio_file_id, audio_features{} (6000+ features)

  [7] POST http://88.222.212.15:9100/predict
      Body: { "features": audio_features{} } (1,351 features)
      → voice_risk_label, voice_probabilities{}

  ══════════ FUSION ══════════
  [8] POST http://88.222.212.15:8000/predict
      Body: combined feature vector (102 features from all modalities)
      → final_prediction, final_confidence, final_probabilities{}

BACKEND RETURNS TO CLIENT:
  {
    "session_result": {
      "face": { "label": "stress", "confidence": 0.94 },
      "text": { "label": "Depression", "confidence": 0.94 },
      "voice": { "label": "normal", "confidence": 0.81 },
      "final": { "label": "Stress", "confidence": 0.79 }
    }
  }
```

### Parallelisation Opportunity

Steps [1–3] (face), [4–5] (text), and [6–7] (voice) are **independent tracks**. The backend can run all three in parallel (using async calls or threads) and only wait for all three before executing step [8] (fusion). This reduces total latency significantly.

---

## 12. Common Error Reference

| HTTP Status | Meaning | Likely Cause | Fix |
|---|---|---|---|
| `400 Bad Request` | Invalid input | Missing required fields, wrong data types, video too short | Validate input before sending; check `allow_short=true` for short videos |
| `401 Unauthorized` | Auth failure (APIs 1, 2, 4, 5, 6) | Missing or wrong `X-API-Key` header | Double-check key name casing and value |
| `403 Forbidden` | Auth failure (API 4) | Missing or wrong `X-API-Key` | Same as above |
| `404 Not Found` | Resource not found | Invalid `session_id` or `file_id`, wrong endpoint path | Verify session_id/file_id from prior responses |
| `422 Unprocessable Entity` | Validation error | Missing features in request body, wrong data types | Check all required fields are present with correct types |
| `500 Internal Server Error` | Runtime failure | Model inference error, model artifact not loaded | Check `/health` endpoint; if `model_loaded: false`, the container needs restart |

### Auth Header Name Quick Reference (Repeat for convenience)

```
APIs 1, 2, 4, 5, 6: X-API-Key
APIs 3, 7:          x-api-key
```

---

## 13. Quick Connectivity Checklist

Before beginning integration, verify all services are reachable:

```bash
# API 1 — Face Extraction
curl http://88.222.212.15:8010/health

# API 2 — Face Scoring
curl http://88.222.212.15:8011/health

# API 3 — Text Analysis
curl http://88.222.212.15:8025/docs

# API 4 — Text Classifier
curl http://88.222.212.15:9000/health

# API 5 — Voice Classifier
curl http://88.222.212.15:9100/health

# API 6 — Multimodal Pipeline
curl -H "X-API-Key: YOUR_KEY" http://88.222.212.15:8000/health

# API 7 — Audio Feature Extraction
curl http://88.222.212.15:8013/
```

Expected responses for healthy services:

| API | Expected |
|---|---|
| API 1 | `{"status": "ok", "service": "Facial Extraction API"}` |
| API 2 | `{"status": "ok", ..., "model_loaded": true}` |
| API 3 | Swagger UI HTML (or `{"status": "ok"}`) |
| API 4 | `{"status": "ok", "artifacts_loaded": true}` |
| API 5 | `{"status": "ok", "artifacts_loaded": true}` |
| API 6 | Service health response |
| API 7 | `{"message": "API is running successfully"}` |

---

## Notes for the Backend Developer

1. **API Keys:** Obtain all API keys from the respective `.env` files on the VPS before starting integration. Store them in your Spring Boot `application.properties` or environment config — never hardcode in source.

2. **Feature Schema for Multimodal API (Port 8000):** The 102 features expected by the fusion API are a curated subset from all modalities. Always call `GET /sample-payload` on port 8000 to get the exact feature names expected. Do not assume they are the same as the raw outputs of other APIs without verification.

3. **Audio Format:** API 7 strictly requires `.wav` files. If the client submits audio in another format (`.mp3`, `.m4a`, etc.), the backend must transcode it before sending. Consider using FFmpeg on the backend server for this conversion.

4. **Text Conversation Format:** API 3 requires labeled turns (`Client:` / `Assistant:`). Ensure the backend formats the conversation string correctly before sending.

5. **Timeout Settings:** Set HTTP client timeouts appropriately — API 3 (text analysis with LLM) can take 3–10 seconds. API 1 (video processing) depends on video length. Recommended minimums: API 1 → 120s, API 3 → 30s, all others → 15s.

6. **CORS Note:** API 5 (Voice Classifier, port 9100) currently has CORS open to all origins. This should be locked down before going to production — coordinate with the ML team.

7. **Swagger UI:** All APIs (except API 6) expose interactive documentation at `http://88.222.212.15:{port}/docs`. Use these for manual testing before wiring into Spring Boot.

---

*End of Mindspace API Integration Guide*
