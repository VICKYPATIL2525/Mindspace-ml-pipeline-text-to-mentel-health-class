/* Mindspace Test App — Frontend JavaScript */

// Feature field definitions (43 features)
const FEATURES = [
    // Linguistic / Semantic (19)
    "overall_sentiment_score", "semantic_coherence_score", "self_reference_density",
    "future_focus_ratio", "positive_emotion_ratio", "fear_word_frequency",
    "sadness_word_frequency", "negative_emotion_ratio", "uncertainty_word_frequency",
    "anger_word_frequency", "rumination_phrase_frequency", "filler_word_frequency",
    "topic_shift_frequency", "total_word_count", "avg_sentence_length",
    "language_model_perplexity", "past_focus_ratio", "repetition_rate", "adjective_ratio",
    // Topic model (5)
    "topic_0", "topic_1", "topic_2", "topic_3", "topic_4",
    // Embeddings (17)
    "emb_1", "emb_3", "emb_4", "emb_5", "emb_7", "emb_8", "emb_10", "emb_11", "emb_12",
    "emb_14", "emb_15", "emb_21", "emb_22", "emb_25", "emb_28", "emb_29", "emb_30",
    // Language flags (2)
    "language_hindi", "language_marathi"
];

// ─── DOM Ready ─────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", function () {
    renderFormFields();
    loadSamples();
    setupEventListeners();
});

// ─── Render Form Fields ────────────────────────────────────────────────────
function renderFormFields() {
    const container = document.getElementById("form-fields");
    container.innerHTML = "";

    FEATURES.forEach((feature) => {
        const group = document.createElement("div");
        group.className = "form-group";

        const label = document.createElement("label");
        label.textContent = formatFeatureName(feature);

        const input = document.createElement("input");
        input.type = "number";
        input.step = "any";
        input.name = feature;
        input.placeholder = "0.0";

        group.appendChild(label);
        group.appendChild(input);
        container.appendChild(group);
    });
}

// ─── Format Feature Name (snake_case → Title Case) ────────────────────────
function formatFeatureName(name) {
    return name
        .split("_")
        .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
        .join(" ");
}

// ─── Load Demo Samples ─────────────────────────────────────────────────────
function loadSamples() {
    fetch("/api/samples")
        .then((res) => res.json())
        .then((samples) => {
            const container = document.getElementById("samples-container");
            container.innerHTML = "";

            // Group by category and render buttons
            Object.entries(samples).forEach(([category, sampleNames]) => {
                const categoryDiv = document.createElement("div");
                categoryDiv.className = "sample-category";

                const title = document.createElement("div");
                title.className = "sample-category-title";
                title.textContent = formatFeatureName(category);
                categoryDiv.appendChild(title);

                sampleNames.forEach((sampleName) => {
                    const btn = document.createElement("button");
                    btn.className = "sample-btn";
                    btn.type = "button";
                    btn.textContent = sampleName;
                    btn.onclick = () => loadSampleData(sampleName);
                    categoryDiv.appendChild(btn);
                });

                container.appendChild(categoryDiv);
            });
        })
        .catch((err) => {
            document.getElementById("samples-container").innerHTML =
                `<p class="error">Failed to load samples: ${err.message}</p>`;
        });
}

// ─── Load Sample Data ──────────────────────────────────────────────────────
function loadSampleData(sampleName) {
    fetch(`/api/load-sample/${sampleName}`)
        .then((res) => res.json())
        .then((data) => {
            // Populate form fields
            Object.entries(data).forEach(([key, value]) => {
                const input = document.querySelector(`input[name="${key}"]`);
                if (input) {
                    input.value = value;
                }
            });

            // Populate JSON tab
            document.getElementById("json-input").value = JSON.stringify(data, null, 2);

            // Switch to form tab
            switchTab("form");
        })
        .catch((err) => {
            alert(`Failed to load sample: ${err.message}`);
        });
}

// ─── Setup Event Listeners ────────────────────────────────────────────────
function setupEventListeners() {
    // Tab switching
    document.querySelectorAll(".tab-btn").forEach((btn) => {
        btn.addEventListener("click", function () {
            const tabName = this.dataset.tab;
            switchTab(tabName);
        });
    });

    // Form submission
    document.getElementById("predict-form").addEventListener("submit", function (e) {
        e.preventDefault();
        submitForm();
    });
}

// ─── Switch Tabs ───────────────────────────────────────────────────────────
function switchTab(tabName) {
    // Deactivate all tabs
    document.querySelectorAll(".tab-content").forEach((tab) => {
        tab.classList.remove("active");
    });
    document.querySelectorAll(".tab-btn").forEach((btn) => {
        btn.classList.remove("active");
    });

    // Activate selected tab
    const tabId = `${tabName}-tab`;
    if (document.getElementById(tabId)) {
        document.getElementById(tabId).classList.add("active");
    }

    // Activate button
    document.querySelector(`[data-tab="${tabName}"]`).classList.add("active");
}

// ─── Submit Form ───────────────────────────────────────────────────────────
function submitForm() {
    const form = document.getElementById("predict-form");
    const formData = new FormData(form);

    // Convert to JSON object
    const payload = {};
    FEATURES.forEach((feature) => {
        const value = formData.get(feature);
        payload[feature] = value ? parseFloat(value) : 0;
    });

    // Update JSON tab
    document.getElementById("json-input").value = JSON.stringify(payload, null, 2);

    // Send to API
    sendPrediction(payload);
}

// ─── Send JSON ─────────────────────────────────────────────────────────────
function sendJSON() {
    try {
        const jsonText = document.getElementById("json-input").value;
        const payload = JSON.parse(jsonText);
        sendPrediction(payload);
    } catch (err) {
        showError(`Invalid JSON: ${err.message}`);
    }
}

// ─── Send Prediction Request ───────────────────────────────────────────────
function sendPrediction(payload) {
    // Validate required fields
    if (!payload || Object.keys(payload).length === 0) {
        showError("No data to send");
        return;
    }

    // Show loading state
    switchTab("results");
    const resultsContainer = document.getElementById("results-container");
    resultsContainer.innerHTML = '<p class="loading">Sending request...</p>';

    // Send POST request
    fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    })
        .then((res) => res.json())
        .then((data) => {
            if (data.error) {
                showError(data.error, data.detail);
            } else {
                displayResults(data);
            }
        })
        .catch((err) => {
            showError("Network error", err.message);
        });
}

// ─── Display Results ───────────────────────────────────────────────────────
function displayResults(result) {
    const container = document.getElementById("results-container");
    container.innerHTML = "";

    if (!result.prediction) {
        showError("Invalid response from API");
        return;
    }

    // Prediction card
    const predCard = document.createElement("div");
    predCard.className = "result-item";

    const header = document.createElement("div");
    header.style.marginBottom = "15px";

    const predLabel = document.createElement("span");
    predLabel.className = "result-label";
    predLabel.textContent = "🎯 Prediction: ";

    const predValue = document.createElement("span");
    predValue.className = "result-value";
    predValue.textContent = result.prediction;

    header.appendChild(predLabel);
    header.appendChild(predValue);

    // Confidence
    const confDiv = document.createElement("div");
    confDiv.style.marginBottom = "10px";

    const confLabel = document.createElement("span");
    confLabel.className = "result-label";
    confLabel.textContent = "📊 Confidence: ";

    const confValue = document.createElement("span");
    confValue.textContent = `${(result.confidence * 100).toFixed(2)}%`;

    confDiv.appendChild(confLabel);
    confDiv.appendChild(confValue);

    // Model info
    const modelDiv = document.createElement("div");
    const modelText = document.createElement("small");
    modelText.style.color = "#999";
    modelText.textContent = `${result.model} (${(result.accuracy * 100).toFixed(1)}% accuracy)`;
    modelDiv.appendChild(modelText);

    predCard.appendChild(header);
    predCard.appendChild(confDiv);
    predCard.appendChild(modelDiv);

    // Probabilities chart
    const chartDiv = document.createElement("div");
    chartDiv.className = "probabilities-chart";

    const chartTitle = document.createElement("div");
    chartTitle.className = "result-label";
    chartTitle.textContent = "Class Probabilities";
    chartDiv.appendChild(chartTitle);

    Object.entries(result.probabilities).forEach(([className, probability]) => {
        const probRow = document.createElement("div");
        probRow.className = "prob-row";

        const label = document.createElement("div");
        label.className = "prob-label";
        label.textContent = className;

        const bar = document.createElement("div");
        bar.className = "prob-bar";

        const fill = document.createElement("div");
        fill.className = "prob-fill";
        fill.style.width = `${probability * 100}%`;
        fill.textContent = `${(probability * 100).toFixed(1)}%`;

        bar.appendChild(fill);
        probRow.appendChild(label);
        probRow.appendChild(bar);
        chartDiv.appendChild(probRow);
    });

    predCard.appendChild(chartDiv);
    container.appendChild(predCard);
}

// ─── Show Error ────────────────────────────────────────────────────────────
function showError(message, detail = "") {
    switchTab("results");
    const container = document.getElementById("results-container");
    container.innerHTML = `
        <div class="error">
            <strong>${message}</strong>
            ${detail ? `<p>${detail}</p>` : ""}
        </div>
    `;
}
