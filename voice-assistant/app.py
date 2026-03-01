"""
Voice Assistant — Streamlit App
Sarvam female voice asks questions → user replies via audio → 
AI generates next question → continues conversation.
Uses Azure OpenAI for conversation + Sarvam for STT/TTS.
"""

import os
import json
import time
import base64
import tempfile
import datetime
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from openai import AzureOpenAI
from sarvamai import SarvamAI

# --- Config ---
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=ROOT_DIR / ".env")

sarvam = SarvamAI(api_subscription_key=os.getenv("SARVAM_API_KEY"))

azure = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

LANGUAGES = {"Hindi": "hi-IN", "English": "en-IN", "Marathi": "mr-IN"}
SPEAKER = "anushka"  # Warm female counselor voice
TTS_MODEL = "bulbul:v3"  # Better quality
STT_MODEL = "saarika:v2.5"  # Current model (v2 deprecated)

# Counselor voice tuning — calm, warm, doctor-like
TTS_PACE = 0.9          # slightly slower — reassuring
TTS_PITCH = -0.05       # slightly deeper — grounded, professional
TTS_LOUDNESS = 1.2      # normal
TTS_TEMPERATURE = 0.4   # consistent tone
TTS_SAMPLE_RATE = 22050

SYSTEM_PROMPT = """You are Mindspace — a compassionate, professional mental health counselor.
You speak in a calm, warm, and reassuring tone like a doctor talking to a patient.
You listen actively, validate feelings, and gently guide the conversation.
Keep responses short (2-3 sentences) since they will be spoken aloud.
Never diagnose — instead offer support, coping techniques, and ask thoughtful follow-up questions.
Always end with a caring question to keep the patient engaged."""

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "started" not in st.session_state:
    st.session_state.started = False

# --- Helpers ---
def ask_ai(messages):
    """Get response from Azure OpenAI."""
    start = time.time()
    resp = azure.chat.completions.create(
        model=DEPLOYMENT, messages=messages, temperature=0.7, max_tokens=256
    )
    elapsed = round(time.time() - start, 3)
    text = resp.choices[0].message.content
    tokens = resp.usage.total_tokens if resp.usage else 0
    return text, elapsed, tokens


def speak(text, lang_code):
    """Convert text to speech using Sarvam TTS with counselor voice settings."""
    start = time.time()
    resp = sarvam.text_to_speech.convert(
        text=text,
        target_language_code=lang_code,
        speaker=SPEAKER,
        model=TTS_MODEL,
        pace=TTS_PACE,
        pitch=TTS_PITCH,
        loudness=TTS_LOUDNESS,
        temperature=TTS_TEMPERATURE,
        speech_sample_rate=TTS_SAMPLE_RATE,
        enable_preprocessing=True,
    )
    elapsed = round(time.time() - start, 3)
    audio_b64 = resp.audios[0] if resp.audios else None
    if audio_b64:
        return base64.b64decode(audio_b64), elapsed
    return None, elapsed


def listen(audio_file, lang_code, model=STT_MODEL):
    """Transcribe audio using Sarvam STT. Returns text."""
    start = time.time()
    resp = sarvam.speech_to_text.transcribe(
        file=audio_file, model=model, language_code=lang_code
    )
    elapsed = round(time.time() - start, 3)
    text = resp.transcript if hasattr(resp, "transcript") else str(resp)
    return text, elapsed


def save_conversation():
    """Save conversation to JSON."""
    data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "turns": st.session_state.conversation,
        "total_turns": len(st.session_state.conversation),
    }
    json_path = OUTPUT_DIR / "voice_assistant_log.json"
    history = []
    if json_path.exists():
        history = json.loads(json_path.read_text(encoding="utf-8"))
    history.append(data)
    json_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")


# --- UI ---
st.set_page_config(page_title="Voice Assistant", page_icon="🤖", layout="centered")
st.title("� Mindspace Voice Counselor")
st.caption("Talk with Mindspace — your calm, supportive mental health companion")

language = st.selectbox("Language", list(LANGUAGES.keys()), index=0)
lang_code = LANGUAGES[language]

st.divider()

# Show conversation history
for turn in st.session_state.conversation:
    if turn["role"] == "assistant":
        with st.chat_message("assistant", avatar="🤖"):
            st.write(turn["text"])
            if turn.get("audio"):
                st.audio(turn["audio"], format="audio/wav")
    else:
        with st.chat_message("user", avatar="🎤"):
            st.write(turn["text"])

# Start / Continue conversation
if not st.session_state.started:
    if st.button("🎙️ Start Conversation", type="primary"):
        with st.spinner("Mindspace is thinking..."):
            # Get opening question from AI
            st.session_state.messages.append(
                {"role": "user", "content": f"Start a counseling session. Gently greet me and ask how I'm feeling today, in {language}. Be warm and professional like a therapist."}
            )
            ai_text, ai_time, tokens = ask_ai(st.session_state.messages)
            st.session_state.messages.append({"role": "assistant", "content": ai_text})

            # Convert to speech
            audio_bytes, tts_time = speak(ai_text, lang_code)

            turn = {
                "role": "assistant",
                "text": ai_text,
                "audio": audio_bytes,
                "ai_time": ai_time,
                "tts_time": tts_time,
                "tokens": tokens,
            }
            st.session_state.conversation.append(turn)
            st.session_state.started = True
            save_conversation()
            st.rerun()
else:
    # User replies via audio upload
    st.subheader("🎤 Your turn — upload your reply")
    user_audio = st.file_uploader("Record/upload your reply", type=["wav", "mp3", "mp4", "m4a", "ogg", "webm"], key=f"audio_{len(st.session_state.conversation)}")

    if user_audio and st.button("📤 Send Reply", type="primary"):
        with st.spinner("Listening..."):
            # Save temp file
            suffix = Path(user_audio.name).suffix
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(user_audio.read())
            tmp.close()

            # STT
            with open(tmp.name, "rb") as f:
                user_text, stt_time = listen(f, lang_code)
            os.unlink(tmp.name)

            st.session_state.conversation.append({
                "role": "user", "text": user_text, "stt_time": stt_time
            })
            st.session_state.messages.append({"role": "user", "content": user_text})

        with st.spinner("Mindspace is thinking..."):
            # AI reply
            ai_text, ai_time, tokens = ask_ai(st.session_state.messages)
            st.session_state.messages.append({"role": "assistant", "content": ai_text})

            # TTS
            audio_bytes, tts_time = speak(ai_text, lang_code)

            st.session_state.conversation.append({
                "role": "assistant",
                "text": ai_text,
                "audio": audio_bytes,
                "ai_time": ai_time,
                "tts_time": tts_time,
                "tokens": tokens,
            })
            save_conversation()
            st.rerun()

    # Reset button
    if st.button("🔄 Reset Conversation"):
        st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        st.session_state.conversation = []
        st.session_state.started = False
        st.rerun()

# Sidebar: stats
with st.sidebar:
    st.header("📊 Session Stats")
    turns = len([t for t in st.session_state.conversation if t["role"] == "user"])
    st.metric("Turns", turns)
    total_ai_time = sum(t.get("ai_time", 0) for t in st.session_state.conversation)
    total_tts_time = sum(t.get("tts_time", 0) for t in st.session_state.conversation)
    total_stt_time = sum(t.get("stt_time", 0) for t in st.session_state.conversation)
    st.metric("AI Response Time", f"{total_ai_time:.2f}s")
    st.metric("TTS Time", f"{total_tts_time:.2f}s")
    st.metric("STT Time", f"{total_stt_time:.2f}s")

    if st.session_state.conversation:
        with st.expander("🔍 Conversation JSON"):
            # Remove audio bytes for display
            display = []
            for t in st.session_state.conversation:
                d = {k: v for k, v in t.items() if k != "audio"}
                display.append(d)
            st.json(display)
