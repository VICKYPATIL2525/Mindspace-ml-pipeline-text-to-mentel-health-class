# =============================================================================
# Mindspace Test Application — Flask Frontend for API Testing
# =============================================================================
# Simple web app to test the Mindspace API with proper authentication.
# Allows uploading demo JSON payloads or manually filling in the form.
# Calls the API with X-API-Key header and displays prediction results.

import os
import json
import requests
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# ─── Configuration ────────────────────────────────────────────────────────────
load_dotenv(dotenv_path=Path(__file__).parent.parent / "deployment" / ".env")

API_URL = "http://127.0.0.1:9000/predict"
API_KEY = os.environ.get("API_KEY", "")

if not API_KEY:
    print("⚠️  WARNING: API_KEY not found in environment. API calls will fail.")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5MB max upload


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    """The main test page — HTML form to submit feature data."""
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Backend endpoint — receives JSON payload and forwards it to the Mindspace API.
    
    Request: JSON with 43 features
    Response: JSON with prediction + probabilities (from Mindspace API)
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON payload provided"}), 400
        
        # ── Call the Mindspace API ────────────────────────────────────────────
        headers = {
            "X-API-Key": API_KEY,
            "Content-Type": "application/json",
        }
        
        response = requests.post(API_URL, json=data, headers=headers, timeout=30)
        
        # ── Return the API response directly ──────────────────────────────────
        if response.status_code == 200:
            return jsonify(response.json()), 200
        else:
            return jsonify({
                "error": f"API returned {response.status_code}",
                "detail": response.text
            }), response.status_code
    
    except requests.exceptions.ConnectionError:
        return jsonify({
            "error": "Cannot connect to API",
            "detail": f"Make sure the API is running at {API_URL}"
        }), 503
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/load-sample/<sample_name>", methods=["GET"])
def load_sample(sample_name):
    """
    Load one of the demo sample JSON files.
    
    Example: GET /api/load-sample/anxiety_sample_1
    Returns: the JSON payload from demo-api-input-data-sample/anxiety_sample_1.json
    """
    try:
        sample_path = (
            Path(__file__).parent.parent / 
            "demo-api-input-data-sample" / 
            f"{sample_name}.json"
        )
        
        if not sample_path.exists():
            return jsonify({"error": f"Sample not found: {sample_name}"}), 404
        
        sample_data = json.loads(sample_path.read_text())
        return jsonify(sample_data), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/samples", methods=["GET"])
def list_samples():
    """
    List all available demo samples grouped by category.
    
    Returns: {"anxiety": [...], "depression": [...], ...}
    """
    try:
        samples_dir = Path(__file__).parent.parent / "demo-api-input-data-sample"
        
        if not samples_dir.exists():
            return jsonify({"error": "Demo samples directory not found"}), 404
        
        samples = {}
        for file in samples_dir.glob("*.json"):
            # Extract category (e.g., "anxiety_sample_1.json" → "anxiety")
            name = file.stem
            category = "_".join(name.split("_")[:-1])  # Remove trailing number
            
            if category not in samples:
                samples[category] = []
            samples[category].append(name)
        
        # Sort each category's samples
        for category in samples:
            samples[category].sort()
        
        return jsonify(samples), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print(f"Starting test app...")
    print(f"API URL: {API_URL}")
    print(f"API Key loaded: {bool(API_KEY)}")
    print(f"Open http://127.0.0.1:5000 in your browser")
    app.run(debug=True, host="127.0.0.1", port=5000)
