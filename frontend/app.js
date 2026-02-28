/**
 * MindSpace Voice Agent — Frontend Application
 * Handles session lifecycle, audio recording, TTS playback, and API communication.
 */

const API_BASE = window.location.origin;

// ── DOM Elements ─────────────────────────────────────────────────────────────
const chatArea     = document.getElementById("chat");
const startBtn     = document.getElementById("startBtn");
const recordBtn    = document.getElementById("recordBtn");
const langSelect   = document.getElementById("langSelect");
const sessionInfo  = document.getElementById("sessionInfo");

// ── State ────────────────────────────────────────────────────────────────────
let sessionId   = null;
let isRecording = false;
let mediaRecorder = null;
let audioChunks = [];
let currentAudio = null;  // currently playing TTS audio

// ══════════════════════════════════════════════════════════════════════════════
// TTS Audio Playback
// ══════════════════════════════════════════════════════════════════════════════

function playBase64Audio(base64String) {
    // Stop any currently playing audio
    if (currentAudio) {
        currentAudio.pause();
        currentAudio = null;
    }
    if (!base64String) return;

    try {
        const audioSrc = `data:audio/wav;base64,${base64String}`;
        currentAudio = new Audio(audioSrc);
        currentAudio.play().catch(err => {
            console.warn("Audio playback failed:", err);
        });
    } catch (err) {
        console.warn("Could not play TTS audio:", err);
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Chat UI helpers
// ══════════════════════════════════════════════════════════════════════════════

function addMessage(role, text) {
    const bubble = document.createElement("div");
    bubble.className = `bubble bubble-${role}`;
    bubble.textContent = text;
    chatArea.appendChild(bubble);
    chatArea.scrollTop = chatArea.scrollHeight;
}

function addSystemMessage(text) {
    const bubble = document.createElement("div");
    bubble.className = "bubble bubble-system";
    bubble.textContent = text;
    chatArea.appendChild(bubble);
    chatArea.scrollTop = chatArea.scrollHeight;
}

function setLoading(on) {
    if (on) {
        const loader = document.createElement("div");
        loader.className = "bubble bubble-ai loading";
        loader.id = "loader";
        loader.textContent = "Asha is thinking…";
        chatArea.appendChild(loader);
        chatArea.scrollTop = chatArea.scrollHeight;
    } else {
        const loader = document.getElementById("loader");
        if (loader) loader.remove();
    }
}

function updateSessionInfo(memory) {
    sessionInfo.textContent = JSON.stringify(
        { session_id: sessionId, memory },
        null,
        2
    );
}

function handleSessionEnd() {
    addSystemMessage("Session ended. Thank you for participating.");
    recordBtn.disabled = true;
    startBtn.disabled = false;
    startBtn.textContent = "Start New Screening";
    sessionId = null;
}

// ══════════════════════════════════════════════════════════════════════════════
// API calls
// ══════════════════════════════════════════════════════════════════════════════

async function startSession() {
    setLoading(true);
    startBtn.disabled = true;
    const lang = langSelect.value;
    try {
        const res = await fetch(`${API_BASE}/api/start?language_code=${lang}`, { method: "POST" });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        sessionId = data.session_id;
        addMessage("ai", data.message);
        // Play greeting audio
        playBase64Audio(data.audio_base64);
        recordBtn.disabled = false;
        startBtn.textContent = "Session Active";
    } catch (err) {
        addSystemMessage(`Error starting session: ${err.message}`);
        startBtn.disabled = false;
    } finally {
        setLoading(false);
    }
}

async function sendAudio(blob) {
    if (!sessionId) return;
    setLoading(true);
    recordBtn.disabled = true;

    const formData = new FormData();
    formData.append("audio", blob, "recording.webm");
    formData.append("session_id", sessionId);
    formData.append("language_code", langSelect.value);

    try {
        const res = await fetch(`${API_BASE}/api/voice`, {
            method: "POST",
            body: formData,
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();

        // Show what the user said (transcript from STT)
        if (data.transcript) {
            addMessage("user", data.transcript);
        }

        // Show and speak AI response
        addMessage("ai", data.message);
        playBase64Audio(data.audio_base64);

        if (data.memory) updateSessionInfo(data.memory);
        if (data.is_ended) {
            handleSessionEnd();
        } else {
            recordBtn.disabled = false;
        }
    } catch (err) {
        addSystemMessage(`Error: ${err.message}`);
        recordBtn.disabled = false;
    } finally {
        setLoading(false);
    }
}

async function sendText(text) {
    if (!sessionId) return;
    setLoading(true);
    recordBtn.disabled = true;

    try {
        const res = await fetch(`${API_BASE}/api/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ session_id: sessionId, text }),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        addMessage("ai", data.message);
        playBase64Audio(data.audio_base64);
        if (data.memory) updateSessionInfo(data.memory);
        if (data.is_ended) {
            handleSessionEnd();
        } else {
            recordBtn.disabled = false;
        }
    } catch (err) {
        addSystemMessage(`Error: ${err.message}`);
        recordBtn.disabled = false;
    } finally {
        setLoading(false);
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Audio Recording (MediaRecorder API)
// ══════════════════════════════════════════════════════════════════════════════

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
        audioChunks = [];

        mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) audioChunks.push(e.data);
        };

        mediaRecorder.onstop = () => {
            const blob = new Blob(audioChunks, { type: "audio/webm" });
            addMessage("user", "🎙️ [sending voice…]");
            sendAudio(blob);
            // Stop all tracks to release the microphone
            stream.getTracks().forEach((t) => t.stop());
        };

        mediaRecorder.start();
        isRecording = true;
        recordBtn.classList.add("recording");
        recordBtn.textContent = "🔴 Recording…";
    } catch (err) {
        addSystemMessage(`Microphone error: ${err.message}`);
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
    }
    isRecording = false;
    recordBtn.classList.remove("recording");
    recordBtn.textContent = "🎙️ Hold to Speak";
}

// ══════════════════════════════════════════════════════════════════════════════
// Event Listeners
// ══════════════════════════════════════════════════════════════════════════════

startBtn.addEventListener("click", () => {
    chatArea.innerHTML = "";
    startSession();
});

// Hold-to-record (mouse)
recordBtn.addEventListener("mousedown", (e) => {
    e.preventDefault();
    if (!isRecording) startRecording();
});
recordBtn.addEventListener("mouseup", () => {
    if (isRecording) stopRecording();
});
recordBtn.addEventListener("mouseleave", () => {
    if (isRecording) stopRecording();
});

// Hold-to-record (touch)
recordBtn.addEventListener("touchstart", (e) => {
    e.preventDefault();
    if (!isRecording) startRecording();
});
recordBtn.addEventListener("touchend", () => {
    if (isRecording) stopRecording();
});

// Keyboard fallback — press Enter to type instead of speak
document.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && sessionId && !startBtn.disabled) {
        const text = prompt("Type your message:");
        if (text && text.trim()) {
            addMessage("user", text.trim());
            sendText(text.trim());
        }
    }
});
