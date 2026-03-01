# Mindspace Voice Agent

A collection of production-grade apps demonstrating **Sarvam AI** (Indian-language voice) and **Azure OpenAI** (LLM) integration for speech-to-text, text-to-speech, and conversational AI — with web-based frontends for the Chatbot and Speech-to-Text apps.

---

## Project Structure

```
Mindspace-voice-agent/
├── .env                        # API keys and configuration
├── requirements.txt            # Python dependencies
├── README.md
│
├── chatbot/                    # App 1: Web Chatbot
│   ├── server.py               # FastAPI backend (port 8001)
│   ├── static/
│   │   ├── index.html          # Chat UI
│   │   ├── style.css
│   │   └── script.js
│   └── output/
│       └── chatbot_trace.json  # Auto-saved conversation trace
│
├── speech-to-text/             # App 2: Speech-to-Text
│   ├── server.py               # FastAPI backend (port 8002)
│   ├── static/
│   │   ├── index.html          # Upload & transcribe UI
│   │   ├── style.css
│   │   └── script.js
│   ├── uploads/                # Uploaded audio files
│   └── output/
│       └── stt_output.json     # Transcription report
│
├── topic-to-speech/            # App 3: Topic → TTS (terminal)
│   ├── app.py
│   └── output/
│       ├── tts_output.json
│       └── output_speech.wav
│
├── marathi-demo-audio.mp4      # Sample Marathi audio
└── myenv/                      # Virtual environment
```

---

## Setup

### 1. Create virtual environment

```bash
python -m venv myenv
myenv\Scripts\Activate.ps1       # Windows PowerShell
# or
source myenv/bin/activate        # macOS/Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
# Sarvam AI
SARVAM_API_KEY=your_sarvam_api_key
SARVAM_ENDPOINT=https://api.sarvam.ai/v1

# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_KEY=your_azure_openai_key
AZURE_OPENAI_DEPLOYMENT=gpt-4.1-mini
AZURE_OPENAI_API_VERSION=2024-12-01-preview
```

- **Sarvam AI key**: Get from [Sarvam AI Dashboard](https://dashboard.sarvam.ai/)
- **Azure OpenAI key**: Get from [Azure Portal](https://portal.azure.com/)

---

## Apps

### App 1: Chatbot (`chatbot/`)

Web-based multi-turn chatbot with a modern chat UI, powered by Azure OpenAI GPT-4.1 Mini.

```bash
cd chatbot
python server.py
# Open http://localhost:8001
```

- Real-time chat interface (HTML/CSS/JS)
- Per-turn timing and token usage displayed in the UI
- Auto-saves conversation trace to `output/chatbot_trace.json`
- Session reset via UI button

### App 2: Speech-to-Text (`speech-to-text/`)

Web UI for uploading audio files and getting transcription via Sarvam AI STT.

```bash
cd speech-to-text
python server.py
# Open http://localhost:8002
```

- Drag & drop or file picker for audio upload
- Supports WAV, MP3, MP4, M4A, OGG, FLAC, WebM, AAC
- Language selector (11 Indian languages)
- Model selector (Saarika v2, v2.5, Saaras v2)
- REST API for short audio (≤30s) with auto-fallback to Batch API for longer files
- Saves full transcription report to `output/stt_output.json`
- Copy transcript & download JSON from the UI

### App 3: Topic → AI Content → Speech (`topic-to-speech/`)

Terminal app: enter a topic → Azure OpenAI generates text → Sarvam AI converts to speech.

```bash
cd topic-to-speech
python app.py
```

- Generates content in any supported Indian language
- TTS via Sarvam SDK (`bulbul:v2`)
- Audio saved to `output/output_speech.wav`
- Pipeline report saved to `output/tts_output.json`

---

## Sarvam AI API Reference

### SDK: `sarvamai` (v0.1.25)

Install: `pip install sarvamai`

```python
from sarvamai import SarvamAI

client = SarvamAI(api_subscription_key="YOUR_API_KEY")
```

### Available SDK Methods

| Method | Description |
|--------|-------------|
| `client.speech_to_text.transcribe()` | REST STT — audio under 30s |
| `client.speech_to_text_job.create_job()` | Batch STT — audio up to 1 hour |
| `client.speech_to_text_streaming` | Streaming/WebSocket STT |
| `client.text_to_speech.convert()` | REST TTS — text to audio |
| `client.text_to_speech_streaming` | Streaming TTS |
| `client.text.translate()` | Text translation |
| `client.chat` | Chat completions |
| `client.document_intelligence` | Document processing |

### Speech-to-Text (REST)

For audio **under 30 seconds**. Use the Batch API for longer audio.

```python
response = client.speech_to_text.transcribe(
    file=open("audio.wav", "rb"),
    model="saarika:v2.5",          # or "saaras:v3"
    language_code="mr-IN",
    # mode="transcribe",           # saaras:v3 only: transcribe|translate|verbatim|translit|codemix
)
print(response.transcript)
```

**Endpoint:** `POST https://api.sarvam.ai/speech-to-text`

### Speech-to-Text (Batch API)

For audio **up to 1 hour**. Supports diarization.

```python
job = client.speech_to_text_job.create_job(
    model="saaras:v3",
    mode="transcribe",
    language_code="mr-IN",
    with_diarization=True,
    num_speakers=2,
)
job.upload_files(file_paths=["long_audio.mp4"])
job.start()
job.wait_until_complete()

results = job.get_file_results()
job.download_outputs(output_dir="./output")
```

### Text-to-Speech

```python
response = client.text_to_speech.convert(
    text="नमस्ते, कसे आहात?",
    target_language_code="mr-IN",
    speaker="anushka",              # bulbul:v2 speakers
    model="bulbul:v2",
)
# response.audios[0] → base64-encoded WAV audio
```

**Endpoint:** `POST https://api.sarvam.ai/text-to-speech`

### STT Models

| Model | Best For | Max Duration |
|-------|----------|-------------|
| `saarika:v2.5` | Quick REST transcription | 30 seconds |
| `saaras:v3` | Advanced features, batch processing | 1 hour (batch) |

#### `saaras:v3` Modes

| Mode | Output |
|------|--------|
| `transcribe` | Standard transcription in source language |
| `translate` | Transcribe + translate to English |
| `verbatim` | Word-for-word including fillers |
| `translit` | Romanized (Latin script) output |
| `codemix` | Code-mixed (English words in English, Indic in native script) |

### TTS Models

| Model | Speakers | Max Chars | Controls |
|-------|----------|-----------|----------|
| `bulbul:v2` | Anushka, Manisha, Vidya, Arya (F) / Abhilash, Karun, Hitesh (M) | 1500 | pitch, pace, loudness |
| `bulbul:v3` | 39 voices (Shubh default) | 2500 | pace, temperature |

### Supported Languages

| Code | Language | STT | TTS |
|------|----------|-----|-----|
| `hi-IN` | Hindi | ✅ | ✅ |
| `en-IN` | English | ✅ | ✅ |
| `mr-IN` | Marathi | ✅ | ✅ |
| `ta-IN` | Tamil | ✅ | ✅ |
| `te-IN` | Telugu | ✅ | ✅ |
| `kn-IN` | Kannada | ✅ | ✅ |
| `ml-IN` | Malayalam | ✅ | ✅ |
| `bn-IN` | Bengali | ✅ | ✅ |
| `gu-IN` | Gujarati | ✅ | ✅ |
| `pa-IN` | Punjabi | ✅ | ✅ |
| `od-IN` | Odia | ✅ | ✅ |
| `as-IN` | Assamese | ✅ (v3) | ❌ |
| `ur-IN` | Urdu | ✅ (v3) | ❌ |
| `ne-IN` | Nepali | ✅ (v3) | ❌ |

### Supported Audio Formats (STT)

WAV, MP3, AAC, AIFF, OGG, OPUS, FLAC, MP4/M4A, AMR, WMA, WebM, PCM

### REST API Base URL

```
https://api.sarvam.ai
```

> **Note:** Do NOT append `/v1` to the base URL for REST endpoints. The correct endpoint is `https://api.sarvam.ai/speech-to-text` (not `https://api.sarvam.ai/v1/speech-to-text`).

---

## JSON Output Examples

### `stt_output.json` (App 2)

```json
{
  "pipeline": "Speech-to-Text (Sarvam AI Batch SDK)",
  "timestamp": "2026-03-01T...",
  "input": { "source_file": "marathi-demo-audio.mp4", "language_code": "mr-IN" },
  "output": { "transcript": "..." },
  "tracing": {
    "job_create_time": 0.5,
    "upload_time": 2.1,
    "processing_time": 45.3,
    "total_time_seconds": 48.2,
    "status": "success"
  }
}
```

### `tts_output.json` (App 3)

```json
{
  "pipeline": "Topic → AI Content → Text-to-Speech",
  "timestamp": "2026-03-01T...",
  "input": { "topic": "bharat maza desh ahe", "language_code": "hi-IN" },
  "output": { "generated_text": "...", "audio_file": "output_speech.wav" },
  "tracing": {
    "total_time_seconds": 5.2,
    "steps": [
      { "step": "content_generation", "provider": "Azure OpenAI", "time_seconds": 2.1 },
      { "step": "text_to_speech", "provider": "Sarvam AI (SDK)", "time_seconds": 3.1 }
    ]
  }
}
```

---

## Tech Stack

- **Python 3.10+**
- **Azure OpenAI** — GPT-4.1 Mini for LLM/chat
- **Sarvam AI** — Indian-language STT & TTS
- **sarvamai SDK** (v0.1.25) — Official Python SDK
- **LangChain** — Framework for LLM apps
- **python-dotenv** — Environment variable management

---

## Links

- [Sarvam AI Docs](https://docs.sarvam.ai)
- [Sarvam AI Dashboard](https://dashboard.sarvam.ai)
- [Sarvam STT API](https://docs.sarvam.ai/api-reference-docs/speech-to-text)
- [Sarvam TTS API](https://docs.sarvam.ai/api-reference-docs/text-to-speech)
- [Sarvam Batch STT Guide](https://docs.sarvam.ai/api-reference-docs/api-guides-tutorials/speech-to-text/batch-api)
- [Azure OpenAI Docs](https://learn.microsoft.com/en-us/azure/ai-services/openai/)