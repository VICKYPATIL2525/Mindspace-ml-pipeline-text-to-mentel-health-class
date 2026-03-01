"""
Text-to-Speech — Streamlit App
Paste text or upload .txt → generate audio with all Sarvam TTS options.
"""

import os
import json
import time
import base64
import datetime
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

LANGUAGES = {
    "Hindi": "hi-IN",
    "English": "en-IN",
    "Marathi": "mr-IN",
    "Tamil": "ta-IN",
    "Telugu": "te-IN",
    "Kannada": "kn-IN",
    "Malayalam": "ml-IN",
    "Bengali": "bn-IN",
    "Gujarati": "gu-IN",
    "Punjabi": "pa-IN",
    "Odia": "od-IN",
}

MODELS = ["bulbul:v3", "bulbul:v2"]  # v3 default — better quality

# All 46 speakers from Sarvam SDK
SPEAKERS = [
    "anushka", "abhilash", "manisha", "vidya", "arya", "karun", "hitesh",
    "aditya", "ritu", "priya", "neha", "rahul", "pooja", "rohan",
    "simran", "kavya", "amit", "dev", "ishita", "shreya", "ratan",
    "varun", "manan", "sumit", "roopa", "kabir", "aayan", "shubh",
    "ashutosh", "advait", "amelia", "sophia", "anand", "tanya", "tarun",
    "sunny", "mani", "gokul", "vijay", "shruti", "suhani", "mohit",
    "kavitha", "rehan", "soham", "rupali",
]

# Recommended calm/warm speakers for mental health counselor voice
COUNSELOR_SPEAKERS = ["anushka", "manisha", "vidya", "priya", "ritu", "kavya", "tanya", "shruti"]

# --- Counselor Voice Defaults (calm, warm, doctor-like) ---
DEFAULT_PACE = 0.9        # slightly slower — calm, reassuring
DEFAULT_PITCH = -0.05     # slightly deeper — professional, grounded
DEFAULT_LOUDNESS = 1.2    # normal volume
DEFAULT_TEMPERATURE = 0.4 # consistent, professional tone
DEFAULT_SAMPLE_RATE = 22050
DEFAULT_SPEAKER = "anushka"  # warm female voice

# --- UI ---
st.set_page_config(page_title="Text to Speech", page_icon="🔊", layout="centered")
st.title("🔊 Text to Speech")
st.caption("Paste text or upload a .txt file → generate speech with Sarvam AI")

# Text input
tab_paste, tab_file = st.tabs(["📝 Paste Text", "📁 Upload .txt"])

text_input = ""
with tab_paste:
    text_input = st.text_area("Enter text", height=150, placeholder="Type or paste your text here...")

with tab_file:
    txt_file = st.file_uploader("Upload .txt file", type=["txt"])
    if txt_file:
        text_input = txt_file.read().decode("utf-8")
        st.text_area("File content", text_input, height=150, disabled=True)

st.divider()

# TTS Options
st.subheader("⚙️ Options")
col1, col2, col3 = st.columns(3)

language = col1.selectbox("Language", list(LANGUAGES.keys()))
model = col2.selectbox("Model", MODELS)

show_all = col3.checkbox("Show all speakers", value=False)
speaker_list = SPEAKERS if show_all else COUNSELOR_SPEAKERS
default_idx = speaker_list.index(DEFAULT_SPEAKER) if DEFAULT_SPEAKER in speaker_list else 0
speaker = col3.selectbox("Speaker", speaker_list, index=default_idx)

# Voice tuning — counselor-friendly defaults
with st.expander("🎛️ Voice Tuning (Counselor Preset)", expanded=False):
    st.caption("Pre-tuned for a calm, warm doctor/counselor voice for mental health conversations.")
    tc1, tc2 = st.columns(2)
    pace = tc1.slider("Pace", 0.5, 2.0, DEFAULT_PACE, 0.05, help="Lower = slower, calmer speech")
    pitch = tc2.slider("Pitch", -1.0, 1.0, DEFAULT_PITCH, 0.05, help="Lower = deeper, more grounded voice")
    tc3, tc4 = st.columns(2)
    loudness = tc3.slider("Loudness", 0.5, 3.0, DEFAULT_LOUDNESS, 0.1, help="Volume level")
    temperature = tc4.slider("Temperature", 0.0, 1.0, DEFAULT_TEMPERATURE, 0.05, help="Lower = more consistent tone")
    tc5, tc6 = st.columns(2)
    sample_rate = tc5.selectbox("Sample Rate (Hz)", [8000, 16000, 22050, 24000], index=2)
    enable_preprocessing = tc6.checkbox("Enable Preprocessing", value=True, help="Cleans up text for better speech")

# Generate
if text_input.strip() and st.button("🎵 Generate Speech", type="primary"):
    with st.spinner("Generating audio..."):
        start = time.time()
        lang_code = LANGUAGES[language]
        full_text = text_input.strip()

        # Chunk text to stay under 2500 char limit
        MAX_CHARS = 2400
        chunks = []
        while full_text:
            if len(full_text) <= MAX_CHARS:
                chunks.append(full_text)
                break
            # Find last sentence end within limit
            cut = full_text[:MAX_CHARS].rfind(".")
            if cut == -1:
                cut = full_text[:MAX_CHARS].rfind(" ")
            if cut == -1:
                cut = MAX_CHARS
            else:
                cut += 1
            chunks.append(full_text[:cut].strip())
            full_text = full_text[cut:].strip()

        try:
            all_audio = b""
            for i, chunk in enumerate(chunks):
                if len(chunks) > 1:
                    st.caption(f"Processing chunk {i+1}/{len(chunks)}...")
                response = client.text_to_speech.convert(
                    text=chunk,
                    target_language_code=lang_code,
                    speaker=speaker,
                    model=model,
                    pace=pace,
                    pitch=pitch,
                    loudness=loudness,
                    temperature=temperature,
                    speech_sample_rate=sample_rate,
                    enable_preprocessing=enable_preprocessing,
                )
                audio_b64 = response.audios[0] if response.audios else None
                if audio_b64:
                    all_audio += base64.b64decode(audio_b64)

            elapsed = round(time.time() - start, 3)

            if all_audio:
                audio_bytes = all_audio

                # Save wav
                wav_path = OUTPUT_DIR / f"tts_{speaker}_{lang_code}.wav"
                wav_path.write_bytes(audio_bytes)

                st.success(f"Generated in {elapsed}s" + (f" ({len(chunks)} chunks)" if len(chunks) > 1 else ""))

                # Play audio
                st.audio(audio_bytes, format="audio/wav")

                # Stats
                c1, c2, c3 = st.columns(3)
                c1.metric("Response Time", f"{elapsed}s")
                c2.metric("Audio Size", f"{len(audio_bytes) / 1024:.1f} KB")
                c3.metric("Input Chars", len(text_input.strip()))

                # Download button
                st.download_button("⬇️ Download WAV", audio_bytes, file_name=wav_path.name, mime="audio/wav")

                # Save to JSON
                result = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "text": text_input.strip()[:200] + ("..." if len(text_input.strip()) > 200 else ""),
                    "text_length": len(text_input.strip()),
                    "chunks": len(chunks),
                    "language": language,
                    "language_code": lang_code,
                    "model": model,
                    "speaker": speaker,
                    "pace": pace,
                    "pitch": pitch,
                    "loudness": loudness,
                    "temperature": temperature,
                    "sample_rate": sample_rate,
                    "enable_preprocessing": enable_preprocessing,
                    "response_time_seconds": elapsed,
                    "audio_size_bytes": len(audio_bytes),
                    "audio_file": str(wav_path),
                }

                json_path = OUTPUT_DIR / "tts_results.json"
                history = []
                if json_path.exists():
                    history = json.loads(json_path.read_text(encoding="utf-8"))
                history.append(result)
                json_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")

                st.caption(f"💾 Saved to `output/tts_results.json`")

                with st.expander("🔍 Raw JSON"):
                    st.json(result)
            else:
                st.error("No audio returned from API")

        except Exception as e:
            st.error(f"Error: {e}")
elif not text_input.strip():
    st.info("Enter some text above to generate speech.")
