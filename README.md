# 🧠 MindSpace — Voice-Based AI Mental Health Screening Agent

> **Disclaimer:** This is a prototype screening tool and is **NOT a medical diagnosis system**. If you or someone you know is in crisis, please contact a professional helpline immediately.

---

## 1. Project Overview

**MindSpace** is a voice-first AI agent that conducts a structured 3-step mental health screening conversation. It uses speech-to-text, LLM reasoning, and a state-machine conversation graph to guide users through a warm, empathetic well-being check-in.

The AI assistant **Asha** greets users, obtains consent, assesses emotional state, and explores well-being topics — all through natural voice interaction.

---

## 2. Architecture

### High-Level Flow

```
┌─────────────┐     audio      ┌──────────────┐    transcript    ┌─────────────────┐
│   Browser    │ ──────────────▶│  Sarvam STT  │ ──────────────▶ │   FastAPI        │
│  (mic input) │                │   Service    │                 │   Backend        │
└─────────────┘                └──────────────┘                 └────────┬────────┘
       ▲                                                                 │
       │  response text                                                  ▼
       │                                                        ┌─────────────────┐
       └────────────────────────────────────────────────────────│  LangGraph       │
                                                                │  Conversation    │
                                                                │  State Machine   │
                                                                └────────┬────────┘
                                                                         │
                                                                         ▼
                                                                ┌─────────────────┐
                                                                │  Azure OpenAI   │
                                                                │  GPT-4o-mini    │
                                                                └─────────────────┘
```

### Conversation State Machine

```
START ──▶ GREETING ──▶ CONSENT ──┬──▶ EMOTIONAL_CHECK ──▶ GUIDED_CONVERSATION ──▶ END
                                 │
                                 ├──▶ CONSENT (re-ask if ambiguous)
                                 │
                                 └──▶ END (if user declines)
```

### Project Structure

```
Mindspace-voice-agent/
├── backend/
│   ├── __init__.py
│   ├── main.py                    # FastAPI application & endpoints
│   ├── config.py                  # Environment variable loader
│   ├── conversation_agent.py      # High-level conversation orchestrator
│   ├── state.py                   # Pydantic state & enum definitions
│   │
│   ├── skills/                    # Markdown-based skill prompts
│   │   ├── greeting.md
│   │   ├── consent.md
│   │   ├── emotion_check.md
│   │   ├── guided_conversation.md
│   │   └── empathy.md
│   │
│   ├── prompts/                   # System & extraction prompts
│   │   ├── system_prompt.md
│   │   └── extraction_prompt.md
│   │
│   ├── services/                  # External service integrations
│   │   ├── __init__.py
│   │   ├── sarvam_stt.py          # Sarvam AI speech-to-text
│   │   └── llm_service.py         # Azure OpenAI chat & extraction
│   │
│   ├── graph/                     # LangGraph state machine
│   │   ├── __init__.py
│   │   └── mindspace_graph.py     # Graph nodes, edges, and routing
│   │
│   └── memory/                    # Session memory management
│       ├── __init__.py
│       └── session_memory.py
│
├── frontend/
│   ├── index.html                 # Minimal UI
│   ├── app.js                     # Frontend logic & audio recording
│   └── style.css                  # Dark-themed styling
│
├── .env                           # API keys (not committed to git)
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## 3. Setup Instructions

### Prerequisites

- Python 3.10+
- A microphone-enabled browser (Chrome recommended)
- API keys for:
  - **Azure OpenAI** (GPT-4o-mini deployment)
  - **Sarvam AI** (Speech-to-Text)

### Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd Mindspace-voice-agent

# 2. Create a virtual environment
python -m venv myenv

# 3. Activate it
# Windows:
myenv\Scripts\activate
# macOS/Linux:
source myenv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

---

## 4. Environment Variables

Create a `.env` file in the project root with the following:

```env
OPENAI_API_KEY=your_azure_openai_api_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_VERSION=2024-12-01-preview
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
SARVAM_API_KEY=your_sarvam_api_key
SARVAM_ENDPOINT=https://api.sarvam.ai/v1
```

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | Azure OpenAI API key |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI resource endpoint |
| `AZURE_OPENAI_VERSION` | API version string |
| `AZURE_OPENAI_DEPLOYMENT` | Deployment name (default: `gpt-4o-mini`) |
| `SARVAM_API_KEY` | Sarvam AI subscription key |
| `SARVAM_ENDPOINT` | Sarvam API base URL |

---

## 5. Running the Server

```bash
# From the project root, with the virtual environment activated:
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Then open your browser to:

- **Frontend UI:** [http://localhost:8000/app](http://localhost:8000/app)
- **API docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 6. System Workflow

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/start` | Start a new screening session |
| `POST` | `/api/chat` | Send text message to session |
| `POST` | `/api/voice` | Send audio for STT + conversation |
| `GET` | `/api/summary/{id}` | Get session memory summary |

### Conversation Flow

1. **User clicks "Start Screening"** → `POST /api/start`
   - Creates a new session
   - LangGraph runs `GREETING → CONSENT` nodes
   - Returns Asha's greeting and consent prompt

2. **User speaks** → Audio recorded in browser → `POST /api/voice`
   - Audio sent to Sarvam STT for transcription
   - Transcript processed by conversation agent
   - LLM generates response + extracts structured data
   - State machine advances to next phase

3. **Repeat** until all phases are complete (or user declines consent)

4. **Session ends** → Summary available via `/api/summary/{id}`

### Session Memory Schema

```json
{
    "consent": true,
    "mood": "anxious and tired",
    "sleep_quality": "poor, waking up multiple times",
    "stress_source": "work deadlines",
    "recent_events": "conflict with a colleague",
    "support_system": "close friend and partner"
}
```

---

## 7. Limitations

- **No persistent storage** — sessions are in-memory only (lost on server restart)
- **No voice emotion detection** — only text-based analysis
- **Single language per session** — STT language is selected at the start
- **No authentication** — prototype does not implement user auth
- **Not clinically validated** — this is a technology demonstration, not a medical tool
- **No TTS** — responses are text-only (no text-to-speech playback)

---

## 8. Future Improvements

### TODO

- [ ] Add **voice emotion detection** (prosody, tone analysis)
- [ ] Add **facial emotion analysis** via webcam
- [ ] Add **risk scoring engine** based on PHQ-9 / GAD-7 scales
- [ ] Add **Tele-MANAS escalation** for high-risk detection
- [ ] **Multilingual support** with dynamic language switching
- [ ] **Persistent user memory** with database backend
- [ ] **Text-to-speech** response playback (Sarvam TTS)
- [ ] **Production deployment** with Docker + cloud hosting
- [ ] **User authentication** and data encryption
- [ ] **Therapist dashboard** for reviewing session summaries

---

## 9. Safety

This system includes safety protocols:

- If a user expresses suicidal ideation or self-harm intent, the agent provides helpline numbers (**Tele-MANAS: 14416**, **iCall: 9152987821**) and stops the screening.
- Every session includes a clear disclaimer that this is not a medical diagnosis.

---

## License

This project is a prototype for educational and research purposes.

---

*Built with FastAPI, LangGraph, Azure OpenAI, and Sarvam AI.*
