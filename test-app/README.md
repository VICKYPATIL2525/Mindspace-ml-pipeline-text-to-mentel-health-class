# Mindspace API Test Application

Simple web frontend to test the Mindspace Mental Health Classifier API.

## Features

- ✅ Load demo sample JSON files with one click
- ✅ Form interface to manually enter all 43 features
- ✅ Raw JSON editor for advanced users
- ✅ Real-time API calls with proper X-API-Key authentication
- ✅ Beautiful visualization of prediction results
- ✅ Probability distribution chart

## Setup

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Make sure the Mindspace API is running** on `http://127.0.0.1:9000`:
```bash
# From project root
.\myenv\Scripts\python -m uvicorn deployment.api_text_to_sentiment:app --host 0.0.0.0 --port 9000 --reload
```

3. **Start the test app**:
```bash
python app.py
```

4. **Open in browser**:
```
http://127.0.0.1:5000
```

## Usage

### Option 1: Load a Demo Sample
1. Go to the left panel "Load Demo Samples"
2. Click any sample (e.g., `anxiety_sample_1`)
3. The form fields automatically populate
4. Click "Predict" button
5. See results in the "Results" tab

### Option 2: Manual Form Entry
1. Fill in the form fields with 43 feature values
2. Click "Predict"
3. See results

### Option 3: JSON Editor
1. Click the "JSON" tab
2. Paste or edit raw JSON with 43 features
3. Click "Predict from JSON"
4. See results

## API

The test app forwards requests to:
- **Endpoint**: `POST http://127.0.0.1:9000/predict`
- **Authentication**: `X-API-Key` header (loaded from `../deployment/.env`)
- **Payload**: 43 float features

## Project Structure

```
test-app/
├── app.py                  # Flask backend
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── templates/
│   └── index.html         # Main HTML page
└── static/
    ├── style.css          # Styling
    └── script.js          # Frontend logic
```

## Features

| Feature | Type | Description |
|---------|------|-------------|
| overall_sentiment_score | float | Emotional sentiment [-1, 1] |
| semantic_coherence_score | float | Text coherence [0, 1] |
| self_reference_density | float | First-person pronoun ratio |
| topic_0 to topic_4 | float | Topic distribution weights |
| emb_1, emb_3, ... emb_30 | float | 17 embedding dimensions |
| language_hindi | 0 or 1 | Hindi language flag |
| language_marathi | 0 or 1 | Marathi language flag |

(43 total features)

## Expected Response

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

## Troubleshooting

**Cannot connect to API**: Make sure the Mindspace API is running on port 9000.

**API_KEY not found**: Ensure `../deployment/.env` exists with `API_KEY=...`.

**Form looks empty**: Open browser console (F12) and check for JavaScript errors.
