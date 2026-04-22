"""
Mindspace API Integration Test Suite
Tests all 7 microservices: health checks + authenticated endpoints.

Configure API keys via environment variables or edit the API_KEYS dict below.
"""

import os
import sys
import json
import time
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

VPS_BASE = "http://88.222.212.15"

SERVICES = {
    "API 1 - Face Extraction":   {"port": 8010, "health": "/health"},
    "API 2 - Face Scoring":      {"port": 8011, "health": "/health"},
    "API 3 - Text Analysis":     {"port": 8025, "health": "/docs"},
    "API 4 - Text Classifier":   {"port": 9000, "health": "/health"},
    "API 5 - Voice Classifier":  {"port": 9100, "health": "/health"},
    "API 6 - Multimodal Pipeline": {"port": 8000, "health": "/health"},
    "API 7 - Audio Extraction":  {"port": 8013, "health": "/"},
}

# API keys — set via env vars or edit directly
API_KEYS = {
    "API_1_KEY": os.environ.get("EXTRACTION_API_KEY", ""),
    "API_2_KEY": os.environ.get("SCORING_API_KEY", ""),
    "API_3_KEY": os.environ.get("TEXT_ANALYSIS_API_KEY", ""),
    "API_4_KEY": os.environ.get("ANTHROPIC_API_KEY", ""),
    "API_5_KEY": os.environ.get("ANTHROPIC_API_KEY", ""),
    "API_6_KEY": os.environ.get("MODEL_API_KEY", ""),
    "API_7_KEY": os.environ.get("AUDIO_API_KEY", ""),
}

TIMEOUT = 15  # seconds per request


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def log(status, message):
    symbols = {"PASS": "[PASS]", "FAIL": "[FAIL]", "SKIP": "[SKIP]", "INFO": "[INFO]"}
    print(f"  {symbols.get(status, '[????]')} {message}")


def make_url(port, path=""):
    return f"{VPS_BASE}:{port}{path}"


def safe_request(method, url, **kwargs):
    """Make an HTTP request and return (response, error_string)."""
    kwargs.setdefault("timeout", TIMEOUT)
    try:
        resp = getattr(requests, method)(url, **kwargs)
        return resp, None
    except requests.ConnectionError:
        return None, "Connection refused / unreachable"
    except requests.Timeout:
        return None, "Request timed out"
    except Exception as e:
        return None, str(e)


# ──────────────────────────────────────────────
# 1. Health Check Tests (no auth)
# ──────────────────────────────────────────────

def test_health_checks():
    print("\n" + "=" * 60)
    print("HEALTH CHECK TESTS (no auth required)")
    print("=" * 60)

    results = {}
    for name, cfg in SERVICES.items():
        url = make_url(cfg["port"], cfg["health"])
        resp, err = safe_request("get", url)

        if err:
            log("FAIL", f"{name} ({url}) -> {err}")
            results[name] = False
        elif resp.status_code == 200:
            # Try to show response body summary
            try:
                body = resp.json()
                log("PASS", f"{name} (:{cfg['port']}) -> {json.dumps(body, indent=None)[:120]}")
            except ValueError:
                log("PASS", f"{name} (:{cfg['port']}) -> HTTP 200 (HTML/non-JSON response, likely Swagger UI)")
            results[name] = True
        else:
            log("FAIL", f"{name} (:{cfg['port']}) -> HTTP {resp.status_code}")
            results[name] = False

    return results


# ──────────────────────────────────────────────
# 2. API 1 — Face Extraction (Port 8010)
# ──────────────────────────────────────────────

def test_api1_extract_video():
    print("\n" + "-" * 60)
    print("API 1 — Facial Feature Extraction (Port 8010)")
    print("-" * 60)

    key = API_KEYS["API_1_KEY"]
    if not key:
        log("SKIP", "POST /extract/video — no EXTRACTION_API_KEY set")
        return None

    # Test with allow_short=true since we may not have a 150s+ video
    test_video = os.environ.get("TEST_VIDEO_PATH", "")
    if not test_video or not os.path.isfile(test_video):
        log("SKIP", "POST /extract/video — no TEST_VIDEO_PATH set or file not found")
        return None

    url = make_url(8010, "/extract/video?mode=fast&allow_short=true")
    headers = {"X-API-Key": key}
    with open(test_video, "rb") as f:
        files = {"video": (os.path.basename(test_video), f, "video/mp4")}
        resp, err = safe_request("post", url, headers=headers, files=files, timeout=120)

    if err:
        log("FAIL", f"POST /extract/video -> {err}")
        return None

    if resp.status_code == 200:
        data = resp.json()
        session_id = data.get("session_id", "")
        log("PASS", f"POST /extract/video -> session_id={session_id}, features={data.get('vector_feature_count')}")

        # Follow up: GET vector
        vec_url = make_url(8010, f"/extract/session/{session_id}/vector")
        resp2, err2 = safe_request("get", vec_url, headers=headers)
        if err2:
            log("FAIL", f"GET /extract/session/.../vector -> {err2}")
        elif resp2.status_code == 200:
            vec_data = resp2.json()
            vec = vec_data.get("vector", {})
            log("PASS", f"GET /extract/session/.../vector -> {len(vec)} features retrieved")
            return vec
        else:
            log("FAIL", f"GET /extract/session/.../vector -> HTTP {resp2.status_code}: {resp2.text[:200]}")
    else:
        log("FAIL", f"POST /extract/video -> HTTP {resp.status_code}: {resp.text[:200]}")

    return None


# ──────────────────────────────────────────────
# 3. API 2 — Face Scoring (Port 8011)
# ──────────────────────────────────────────────

def test_api2_score(vector=None):
    print("\n" + "-" * 60)
    print("API 2 — Facial Risk Scoring (Port 8011)")
    print("-" * 60)

    key = API_KEYS["API_2_KEY"]
    if not key:
        log("SKIP", "POST /score — no SCORING_API_KEY set")
        return

    if vector is None:
        log("SKIP", "POST /score — no facial vector available (API 1 not run or failed)")
        return

    url = make_url(8011, "/score")
    headers = {"X-API-Key": key, "Content-Type": "application/json"}
    payload = {"vector": vector}

    resp, err = safe_request("post", url, headers=headers, json=payload)
    if err:
        log("FAIL", f"POST /score -> {err}")
        return

    if resp.status_code == 200:
        data = resp.json()
        dominant = data.get("dominant_risk", {})
        log("PASS", f"POST /score -> dominant_risk={dominant.get('label')} (p={dominant.get('probability')})")
    else:
        log("FAIL", f"POST /score -> HTTP {resp.status_code}: {resp.text[:200]}")


# ──────────────────────────────────────────────
# 4. API 3 — Text Analysis (Port 8025)
# ──────────────────────────────────────────────

def test_api3_analyze():
    print("\n" + "-" * 60)
    print("API 3 — Psychological Text Analysis (Port 8025)")
    print("-" * 60)

    key = API_KEYS["API_3_KEY"]
    if not key:
        log("SKIP", "POST /analyze — no TEXT_ANALYSIS_API_KEY set")
        return None

    url = make_url(8025, "/analyze")
    headers = {"x-api-key": key, "Content-Type": "application/json"}
    payload = {
        "conversation": (
            "Client: I have been feeling very stressed and overwhelmed lately "
            "Assistant: Can you tell me more about what has been causing this stress? "
            "Client: Work pressure and I can't sleep well at night, everything feels out of control "
            "Assistant: How long have you been experiencing these feelings? "
            "Client: For about three weeks now, it keeps getting worse"
        )
    }

    resp, err = safe_request("post", url, headers=headers, json=payload, timeout=30)
    if err:
        log("FAIL", f"POST /analyze -> {err}")
        return None

    if resp.status_code == 200:
        data = resp.json()
        analysis = data.get("analysis", {})
        log("PASS", f"POST /analyze -> status={data.get('status')}, features_count={len(analysis)}")
        log("INFO", f"  latency={data.get('latency', {}).get('total_time', '?')}s, "
                     f"word_count={analysis.get('total_word_count', '?')}, "
                     f"coherence={analysis.get('semantic_coherence_score', '?')}")
        return analysis
    else:
        log("FAIL", f"POST /analyze -> HTTP {resp.status_code}: {resp.text[:200]}")
        return None


# ──────────────────────────────────────────────
# 5. API 4 — Text Classifier (Port 9000)
# ──────────────────────────────────────────────

def test_api4_info():
    """Test the unauthenticated root endpoint."""
    url = make_url(9000, "/")
    resp, err = safe_request("get", url)
    if err:
        log("FAIL", f"GET / -> {err}")
    elif resp.status_code == 200:
        data = resp.json()
        log("PASS", f"GET / -> model={data.get('model')}, accuracy={data.get('accuracy')}, "
                     f"classes={data.get('classes')}")
    else:
        log("FAIL", f"GET / -> HTTP {resp.status_code}")


def test_api4_predict(text_features=None):
    print("\n" + "-" * 60)
    print("API 4 — Text Classifier (Port 9000)")
    print("-" * 60)

    test_api4_info()

    key = API_KEYS["API_4_KEY"]
    if not key:
        log("SKIP", "POST /predict — no TEXT_CLASSIFIER_API_KEY set")
        return

    if text_features is None:
        log("SKIP", "POST /predict — no text features available (API 3 not run or failed)")
        return

    url = make_url(9000, "/predict")
    headers = {"X-API-Key": key, "Content-Type": "application/json"}

    resp, err = safe_request("post", url, headers=headers, json=text_features)
    if err:
        log("FAIL", f"POST /predict -> {err}")
        return

    if resp.status_code == 200:
        data = resp.json()
        log("PASS", f"POST /predict -> prediction={data.get('prediction')}, "
                     f"confidence={data.get('confidence')}")
    else:
        log("FAIL", f"POST /predict -> HTTP {resp.status_code}: {resp.text[:200]}")


# ──────────────────────────────────────────────
# 6. API 5 — Voice Classifier (Port 9100)
# ──────────────────────────────────────────────

def test_api5_info():
    url = make_url(9100, "/")
    resp, err = safe_request("get", url)
    if err:
        log("FAIL", f"GET / -> {err}")
    elif resp.status_code == 200:
        data = resp.json()
        log("PASS", f"GET / -> model={data.get('model')}, accuracy={data.get('accuracy')}, "
                     f"n_features={data.get('n_features')}")
    else:
        log("FAIL", f"GET / -> HTTP {resp.status_code}")


def test_api5_predict(audio_features=None):
    print("\n" + "-" * 60)
    print("API 5 — Voice Classifier (Port 9100)")
    print("-" * 60)

    test_api5_info()

    key = API_KEYS["API_5_KEY"]
    if not key:
        log("SKIP", "POST /predict — no VOICE_CLASSIFIER_API_KEY set")
        return

    if audio_features is None:
        log("SKIP", "POST /predict — no audio features available (API 7 not run or failed)")
        return

    url = make_url(9100, "/predict")
    headers = {"X-API-Key": key, "Content-Type": "application/json"}
    payload = {"features": audio_features}

    resp, err = safe_request("post", url, headers=headers, json=payload)
    if err:
        log("FAIL", f"POST /predict -> {err}")
        return

    if resp.status_code == 200:
        data = resp.json()
        log("PASS", f"POST /predict -> prediction={data.get('prediction')}, "
                     f"confidence={data.get('confidence')}")
    else:
        log("FAIL", f"POST /predict -> HTTP {resp.status_code}: {resp.text[:200]}")


# ──────────────────────────────────────────────
# 7. API 6 — Multimodal Pipeline (Port 8000)
# ──────────────────────────────────────────────

def test_api6_metadata():
    print("\n" + "-" * 60)
    print("API 6 — Multimodal Pipeline (Port 8000)")
    print("-" * 60)

    key = API_KEYS["API_6_KEY"]
    if not key:
        log("SKIP", "GET /metadata — no MULTIMODAL_API_KEY set")
        return

    headers = {"X-API-Key": key}

    # /metadata
    url = make_url(8000, "/metadata")
    resp, err = safe_request("get", url, headers=headers)
    if err:
        log("FAIL", f"GET /metadata -> {err}")
    elif resp.status_code == 200:
        data = resp.json()
        log("PASS", f"GET /metadata -> {json.dumps(data, indent=None)[:150]}")
    else:
        log("FAIL", f"GET /metadata -> HTTP {resp.status_code}: {resp.text[:200]}")

    # /sample-payload
    url = make_url(8000, "/sample-payload")
    resp, err = safe_request("get", url, headers=headers)
    if err:
        log("FAIL", f"GET /sample-payload -> {err}")
    elif resp.status_code == 200:
        data = resp.json()
        feature_count = len(data) if isinstance(data, dict) else "?"
        log("PASS", f"GET /sample-payload -> {feature_count} features in sample")
    else:
        log("FAIL", f"GET /sample-payload -> HTTP {resp.status_code}: {resp.text[:200]}")


# ──────────────────────────────────────────────
# 8. API 7 — Audio Feature Extraction (Port 8013)
# ──────────────────────────────────────────────

def test_api7_extract():
    print("\n" + "-" * 60)
    print("API 7 — Audio Feature Extraction (Port 8013)")
    print("-" * 60)

    key = API_KEYS["API_7_KEY"]
    if not key:
        log("SKIP", "POST /extract — no AUDIO_EXTRACTION_API_KEY set")
        return None

    test_audio = os.environ.get("TEST_AUDIO_PATH", "")
    if not test_audio or not os.path.isfile(test_audio):
        log("SKIP", "POST /extract — no TEST_AUDIO_PATH set or file not found")
        return None

    url = make_url(8013, "/extract")
    headers = {"x-api-key": key}
    with open(test_audio, "rb") as f:
        files = {"file": (os.path.basename(test_audio), f, "audio/wav")}
        resp, err = safe_request("post", url, headers=headers, files=files, timeout=60)

    if err:
        log("FAIL", f"POST /extract -> {err}")
        return None

    if resp.status_code == 200:
        data = resp.json()
        features = data.get("features", {})
        log("PASS", f"POST /extract -> file_id={data.get('file_id')}, "
                     f"feature_count={data.get('feature_count')}, sr={data.get('sampling_rate')}")
        return features
    else:
        log("FAIL", f"POST /extract -> HTTP {resp.status_code}: {resp.text[:200]}")
        return None


# ──────────────────────────────────────────────
# Direct LLM / Upstream API Tests
# (bypasses VPS containers — calls provider APIs directly)
# ──────────────────────────────────────────────

def test_anthropic_direct():
    print("\n" + "-" * 60)
    print("DIRECT — Anthropic Claude API")
    print("-" * 60)

    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        log("SKIP", "No ANTHROPIC_API_KEY set")
        return

    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": "claude-3-haiku-20240307",
        "max_tokens": 64,
        "messages": [
            {"role": "user", "content": "Reply with exactly: API key is working"}
        ],
    }

    resp, err = safe_request("post", url, headers=headers, json=payload, timeout=20)
    if err:
        log("FAIL", f"Anthropic -> {err}")
        return

    if resp.status_code == 200:
        data = resp.json()
        reply = data.get("content", [{}])[0].get("text", "").strip()
        model = data.get("model", "?")
        log("PASS", f"Anthropic -> model={model}, reply='{reply}'")
    else:
        log("FAIL", f"Anthropic -> HTTP {resp.status_code}: {resp.text[:200]}")


def test_azure_openai_direct():
    print("\n" + "-" * 60)
    print("DIRECT — Azure OpenAI API")
    print("-" * 60)

    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    key = os.environ.get("AZURE_OPENAI_KEY", "")
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

    if not endpoint or not key:
        log("SKIP", "No AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_KEY set")
        return

    url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
    headers = {
        "api-key": key,
        "Content-Type": "application/json",
    }
    payload = {
        "messages": [
            {"role": "user", "content": "Reply with exactly: API key is working"}
        ],
        "max_tokens": 32,
    }

    resp, err = safe_request("post", url, headers=headers, json=payload, timeout=20)
    if err:
        log("FAIL", f"Azure OpenAI -> {err}")
        return

    if resp.status_code == 200:
        data = resp.json()
        reply = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        model = data.get("model", deployment)
        log("PASS", f"Azure OpenAI -> model={model}, reply='{reply}'")
    else:
        log("FAIL", f"Azure OpenAI -> HTTP {resp.status_code}: {resp.text[:200]}")


def test_sarvam_direct():
    print("\n" + "-" * 60)
    print("DIRECT — Sarvam AI API")
    print("-" * 60)

    key = os.environ.get("SARVAM_API_KEY", "")
    endpoint = os.environ.get("SARVAM_ENDPOINT", "https://api.sarvam.ai/v1")

    if not key:
        log("SKIP", "No SARVAM_API_KEY set")
        return

    # Test with the text analytics / chat endpoint
    url = f"{endpoint}/chat/completions"
    headers = {
        "api-subscription-key": key,
        "Content-Type": "application/json",
    }
    payload = {
        "model": "sarvam-m",
        "messages": [{"role": "user", "content": "Reply with exactly: API key is working"}],
        "max_tokens": 128,
    elif resp.status_code == 401:
        log("FAIL", f"Sarvam -> HTTP 401 Unauthorized — key rejected")
    else:
        log("FAIL", f"Sarvam -> HTTP {resp.status_code}: {resp.text[:200]}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Mindspace API Integration Test Suite")
    print(f"  VPS: {VPS_BASE}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Check which API keys are configured
    print("\nAPI Key Status:")
    for name, val in API_KEYS.items():
        status = "configured" if val else "NOT SET"
        print(f"  {name}: {status}")

    # ── Phase 1: Health checks (no auth) ──
    health_results = test_health_checks()

    passed = sum(1 for v in health_results.values() if v)
    total = len(health_results)
    print(f"\nHealth Check Summary: {passed}/{total} services reachable")

    # ── Phase 2: Authenticated endpoint tests ──
    print("\n" + "=" * 60)
    print("AUTHENTICATED ENDPOINT TESTS")
    print("=" * 60)

    # Face track: API 1 -> API 2
    facial_vector = test_api1_extract_video()
    test_api2_score(facial_vector)

    # Text track: API 3 -> API 4
    text_features = test_api3_analyze()
    test_api4_predict(text_features)

    # Voice track: API 7 -> API 5
    audio_features = test_api7_extract()
    test_api5_predict(audio_features)

    # Multimodal: API 6
    test_api6_metadata()

    # ── Phase 3: Direct LLM / upstream provider tests ──
    print("\n" + "=" * 60)
    print("DIRECT LLM / UPSTREAM PROVIDER TESTS")
    print("=" * 60)

    test_anthropic_direct()
    test_azure_openai_direct()
    test_sarvam_direct()

    # ── Summary ──
    print("\n" + "=" * 60)
    print("TEST RUN COMPLETE")
    print("=" * 60)
    print("\nNotes:")
    print("  - Health checks test connectivity only (no auth)")
    print("  - Authenticated tests require env vars for API keys")
    print("  - File-based tests (video/audio) require TEST_VIDEO_PATH / TEST_AUDIO_PATH env vars")
    print("  - Set env vars and re-run for full integration testing")


if __name__ == "__main__":
    main()
