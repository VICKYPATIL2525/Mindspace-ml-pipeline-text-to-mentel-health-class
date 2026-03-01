"""
Speech-to-Text — Streamlit App
Upload or record audio in Marathi, Hindi, or English → get text + stats.
"""

import os
import json
import time
import datetime
import tempfile
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from sarvamai import SarvamAI

# --- Config ---
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=ROOT_DIR / ".env")

client = SarvamAI(api_subscription_key=os.getenv("SARVAM_API_KEY"))

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

LANGUAGES = {"Hindi": "hi-IN", "English": "en-IN", "Marathi": "mr-IN"}

# MIME map to force audio type (Sarvam rejects video/mp4)
MIME_MAP = {
    ".wav": "audio/wav", ".mp3": "audio/mp3", ".mp4": "audio/mp4",
    ".m4a": "audio/x-m4a", ".ogg": "audio/ogg", ".flac": "audio/flac",
    ".webm": "audio/webm", ".aac": "audio/aac",
}

# Session state
if "stt_result" not in st.session_state:
    st.session_state.stt_result = None

# --- UI ---
st.set_page_config(page_title="Speech to Text", page_icon="🎙️", layout="centered")
st.title("🎙️ Speech to Text")
st.caption("Upload audio in Marathi, Hindi or English → get transcription")

col1, col2 = st.columns(2)
language = col1.selectbox("Language", list(LANGUAGES.keys()), index=2)

# Upload audio
uploaded = st.file_uploader("Upload audio file", type=["wav", "mp3", "mp4", "m4a", "ogg", "flac", "webm"])

# Demo file shortcut
demo_path = ROOT_DIR / "demo-audio" / "marathi-demo-audio.mp4"


def transcribe_rest(audio_path: str, lang_code: str):
    """Try REST API (fast, for audio ≤ 30s). Returns transcript or raises."""
    ext = Path(audio_path).suffix.lower()
    mime = MIME_MAP.get(ext, "application/octet-stream")
    with open(audio_path, "rb") as f:
        file_tuple = (Path(audio_path).name, f.read(), mime)
        response = client.speech_to_text.transcribe(
            file=file_tuple,
            model="saarika:v2.5",
            language_code=lang_code,
        )
    return response.transcript if hasattr(response, "transcript") else str(response)


def transcribe_batch(audio_path: str, lang_code: str):
    """Batch API (for audio > 30s). Returns transcript."""
    batch_dir = OUTPUT_DIR / "batch_output"
    batch_dir.mkdir(exist_ok=True)

    job = client.speech_to_text_job.create_job(
        model="saaras:v3",
        mode="transcribe",
        language_code=lang_code,
        with_diarization=False,
    )
    job.upload_files(file_paths=[audio_path])
    job.start()
    job.wait_until_complete()
    job.download_outputs(output_dir=str(batch_dir))

    # Read transcript from downloaded files
    transcript = ""
    for fname in sorted(os.listdir(batch_dir)):
        fpath = os.path.join(batch_dir, fname)
        if fname.endswith(".json"):
            raw = json.loads(Path(fpath).read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                transcript = raw.get("transcript", str(raw))
            elif isinstance(raw, list):
                transcript = " ".join(item.get("transcript", "") for item in raw if isinstance(item, dict))
        elif fname.endswith(".txt"):
            transcript = Path(fpath).read_text(encoding="utf-8")
    return transcript


def run_transcription(audio_path: str, file_name: str, file_size: int):
    """Try REST first, fall back to batch for long audio."""
    lang_code = LANGUAGES[language]
    start = time.time()
    method = "rest"

    try:
        transcript = transcribe_rest(audio_path, lang_code)
    except Exception as rest_err:
        err_msg = str(rest_err).lower()
        # If audio is too long for REST, use batch
        if any(kw in err_msg for kw in ["30", "duration", "too long", "limit", "exceeded"]):
            method = "batch"
            transcript = transcribe_batch(audio_path, lang_code)
        else:
            st.session_state.stt_result = {"error": str(rest_err)}
            return

    elapsed = round(time.time() - start, 3)

    result = {
        "timestamp": datetime.datetime.now().isoformat(),
        "file": file_name,
        "file_size_bytes": file_size,
        "language": language,
        "language_code": lang_code,
        "method": method,
        "transcript": transcript,
        "response_time_seconds": elapsed,
        "transcript_length": len(transcript),
    }

    # Save to JSON
    json_path = OUTPUT_DIR / "stt_results.json"
    history = []
    if json_path.exists():
        history = json.loads(json_path.read_text(encoding="utf-8"))
    history.append(result)
    json_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")

    st.session_state.stt_result = result


# --- Buttons ---
btn_col1, btn_col2 = st.columns(2)

with btn_col1:
    if demo_path.exists():
        if st.button("📂 Use Marathi Demo Audio", use_container_width=True):
            with st.spinner("Transcribing demo audio..."):
                run_transcription(
                    str(demo_path),
                    "marathi-demo-audio.mp4",
                    os.path.getsize(demo_path),
                )

with btn_col2:
    if uploaded:
        if st.button("🎯 Transcribe Uploaded", type="primary", use_container_width=True):
            with st.spinner("Transcribing..."):
                suffix = Path(uploaded.name).suffix
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tmp.write(uploaded.read())
                tmp.close()
                run_transcription(tmp.name, uploaded.name, os.path.getsize(tmp.name))
                os.unlink(tmp.name)

# --- Show Result ---
if st.session_state.stt_result:
    result = st.session_state.stt_result

    if "error" in result:
        st.error(f"Error: {result['error']}")
    else:
        st.success(f"✅ Transcribed in {result['response_time_seconds']}s")
        st.text_area("📝 Transcript", result["transcript"], height=200)

        st.subheader("📊 Stats")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Time", f"{result['response_time_seconds']}s")
        c2.metric("File Size", f"{result['file_size_bytes'] / 1024:.1f} KB")
        c3.metric("Characters", result["transcript_length"])
        c4.metric("Method", result.get("method", "—").upper())

        st.caption(f"💾 Saved to `output/stt_results.json`")

        with st.expander("🔍 Raw JSON"):
            st.json(result)

    if st.button("🔄 Clear"):
        st.session_state.stt_result = None
        st.rerun()
