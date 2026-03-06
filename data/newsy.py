import numpy as np
import pandas as pd
from scipy.stats import truncnorm
from tqdm import tqdm

np.random.seed(42)

ROWS = 100000
EMB_DIM = 32
TOPICS = 5

# -----------------------------
# Helper: generate ROWS values at once
# -----------------------------
def trunc(mean, sd, low, high):
    a, b = (low - mean) / sd, (high - mean) / sd
    return truncnorm.rvs(a, b, loc=mean, scale=sd, size=ROWS)

# -----------------------------
# Language distribution
# -----------------------------
languages = ["english", "hindi", "marathi"]
lang_prob = [0.45, 0.35, 0.20]

# -----------------------------
# Mental health prediction classes
# -----------------------------
profiles = {
    "Depression":        0.25,
    "Anxiety":           0.20,
    "Stress":            0.20,
    "Bipolar_Mania":     0.15,
    "Phobia":            0.10,
    "Suicidal_Tendency": 0.10,
}

profile_names = list(profiles.keys())
profile_prob = list(profiles.values())

# -----------------------------
# Generate all columns at once (vectorized)
# -----------------------------
profile_col  = np.random.choice(profile_names, size=ROWS, p=profile_prob)
language_col = np.random.choice(languages, size=ROWS, p=lang_prob)

pbar = tqdm(total=8, desc="Generating dataset", unit="step", ncols=80, colour="green")
pbar.set_postfix_str("class labels & language")
pbar.update(1)

total_words = trunc(300, 120, 50, 1200)
ttr         = trunc(0.5, 0.12, 0.2, 0.9)

positive    = trunc(0.04, 0.02, 0,   0.12)
negative    = trunc(0.05, 0.03, 0,   0.15)
fear        = trunc(0.02, 0.02, 0,   0.08)
sadness     = trunc(0.03, 0.02, 0,   0.12)
anger       = trunc(0.02, 0.02, 0,   0.08)
uncertainty = trunc(0.03, 0.02, 0,   0.12)
filler      = trunc(0.03, 0.02, 0,   0.10)
repetition  = trunc(0.05, 0.04, 0,   0.20)
rumination  = trunc(0.02, 0.02, 0,   0.10)
coherence   = trunc(0.75, 0.1,  0,   1.0)
perplexity  = trunc(80,   40,   20,  300)
self_ref    = trunc(0.05, 0.02, 0,   0.20)
pbar.set_postfix_str("base linguistic features")
pbar.update(1)

# -----------------------------
# Profile masks
# -----------------------------
dep_mask      = profile_col == "Depression"
anx_mask      = profile_col == "Anxiety"
stress_mask   = profile_col == "Stress"
mania_mask    = profile_col == "Bipolar_Mania"
phobia_mask   = profile_col == "Phobia"
suicidal_mask = profile_col == "Suicidal_Tendency"

# Depression: more sadness/rumination, less positive, more negative
sadness    = np.where(dep_mask, sadness    * 2.0, sadness)
rumination = np.where(dep_mask, rumination * 2.5, rumination)
negative   = np.where(dep_mask, negative   * 1.7, negative)
positive   = np.where(dep_mask, positive   * 0.5, positive)
self_ref   = np.where(dep_mask, self_ref   * 2.0, self_ref)

# Anxiety: more fear, uncertainty, filler
fear        = np.where(anx_mask, fear        * 2.5, fear)
uncertainty = np.where(anx_mask, uncertainty * 2.5, uncertainty)
filler      = np.where(anx_mask, filler      * 2.0, filler)

# Stress: more anger, higher perplexity
anger      = np.where(stress_mask, anger      * 2.5, anger)
perplexity = np.where(stress_mask, perplexity * 1.4, perplexity)

# Bipolar Mania: more words, lower coherence (topic shifts handled below)
total_words = np.where(mania_mask, total_words * 1.8, total_words)
coherence   = np.where(mania_mask, coherence   * 0.65, coherence)

# Phobia: strongly more fear, more uncertainty
fear        = np.where(phobia_mask, fear        * 3.5, fear)
uncertainty = np.where(phobia_mask, uncertainty * 2.0, uncertainty)

# Suicidal Tendency: strong sadness/negative, very low positive, high self-reference
sadness  = np.where(suicidal_mask, sadness  * 3.5, sadness)
negative = np.where(suicidal_mask, negative * 2.5, negative)
positive = np.where(suicidal_mask, positive * 0.3, positive)
self_ref = np.where(suicidal_mask, self_ref * 2.5, self_ref)

# Clip to natural bounds
fear        = np.clip(fear,        0, 0.30)
sadness     = np.clip(sadness,     0, 0.40)
anger       = np.clip(anger,       0, 0.20)
uncertainty = np.clip(uncertainty, 0, 0.30)
filler      = np.clip(filler,      0, 0.25)
rumination  = np.clip(rumination,  0, 0.30)
negative    = np.clip(negative,    0, 0.50)
positive    = np.clip(positive,    0, 0.12)
coherence   = np.clip(coherence,   0, 1.0)
perplexity  = np.clip(perplexity,  20, 500)
self_ref    = np.clip(self_ref,    0, 0.40)
pbar.set_postfix_str("class-specific adjustments")
pbar.update(1)

# Overall sentiment
sentiment = np.tanh((positive - negative) * 5)
# Suicidal: strongly suppress sentiment
sentiment = np.where(suicidal_mask, sentiment - 0.5, sentiment)
sentiment = np.clip(sentiment, -1.0, 1.0)

# Time focus
past    = trunc(0.18, 0.08, 0, 0.4)
present = trunc(0.50, 0.10, 0, 0.6)
future  = trunc(0.06, 0.04, 0, 0.3)
# Depression: decrease future focus
future = np.where(dep_mask,      future * 0.40,  future)
# Suicidal: strongly decrease future focus
future = np.where(suicidal_mask, future * 0.15,  future)
future = np.clip(future, 1e-6, 0.3)
total_time = past + present + future
past    = past    / total_time
present = present / total_time
future  = future  / total_time
pbar.set_postfix_str("time focus & sentiment")
pbar.update(1)

# Syntactic
noun_ratio   = trunc(0.22, 0.05, 0.10, 0.40)
verb_ratio   = trunc(0.18, 0.05, 0.10, 0.35)
adj_ratio    = trunc(0.09, 0.03, 0.05, 0.25)
adv_ratio    = trunc(0.05, 0.02, 0.02, 0.20)
sentence_len = trunc(12,   5,    5,    30)
parse_depth  = trunc(6,    2,    2,    12)
# Stress: slightly longer sentences; Bipolar Mania: longer sentences
sentence_len = np.where(stress_mask, sentence_len * 1.3, sentence_len)
sentence_len = np.where(mania_mask,  sentence_len * 1.5, sentence_len)
sentence_len = np.clip(sentence_len, 5, 60)
pbar.set_postfix_str("syntax features")
pbar.update(1)

# Unique word count derived after total_words adjustment
unique_words = (total_words * ttr).astype(int)
total_words  = total_words.astype(int)

# Topics: shape (ROWS, TOPICS)
topic_matrix = np.random.dirichlet(np.ones(TOPICS), size=ROWS)

# Bipolar Mania: high-entropy (spread) topic distribution to reflect topic shifting
mania_topics = np.random.dirichlet(np.ones(TOPICS) * 0.3, size=ROWS)
topic_matrix  = np.where(mania_mask[:, None], mania_topics, topic_matrix)

# Phobia: bias topic_3 toward fear-related content
topic_matrix[phobia_mask, 3] += 0.3

# Normalize so every row sums to 1
topic_matrix = topic_matrix / topic_matrix.sum(axis=1, keepdims=True)

# topic_shift_frequency: entropy of the topic distribution, normalized to [0, 1]
topic_entropy    = -np.sum(topic_matrix * np.log(topic_matrix + 1e-9), axis=1)
topic_shift_freq = topic_entropy / np.log(TOPICS)
topic_shift_freq = np.where(mania_mask, topic_shift_freq * 1.5, topic_shift_freq)
topic_shift_freq = np.clip(topic_shift_freq, 0, 1)
pbar.set_postfix_str("topic matrix")
pbar.update(1)

# Embeddings: shape (ROWS, EMB_DIM) — unchanged
emb_matrix = np.random.normal(0, 1, size=(ROWS, EMB_DIM))
pbar.set_postfix_str("embeddings")
pbar.update(1)

# -----------------------------
# Assemble DataFrame
# -----------------------------
df = pd.DataFrame({
    "profile":                     profile_col,
    "language":                    language_col,
    "total_word_count":            total_words,
    "unique_word_count":           unique_words,
    "ttr":                         ttr,
    "positive_emotion_ratio":      positive,
    "negative_emotion_ratio":      negative,
    "fear_word_frequency":         fear,
    "sadness_word_frequency":      sadness,
    "anger_word_frequency":        anger,
    "uncertainty_word_frequency":  uncertainty,
    "filler_word_frequency":       filler,
    "repetition_rate":             repetition,
    "rumination_phrase_frequency": rumination,
    "semantic_coherence_score":    coherence,
    "language_model_perplexity":   perplexity,
    "overall_sentiment_score":     sentiment,
    "past_focus_ratio":            past,
    "present_focus_ratio":         present,
    "future_focus_ratio":          future,
    "noun_ratio":                  noun_ratio,
    "verb_ratio":                  verb_ratio,
    "adjective_ratio":             adj_ratio,
    "adverb_ratio":                adv_ratio,
    "avg_sentence_length":         sentence_len,
    "parse_tree_depth":            parse_depth,
    "topic_shift_frequency":       topic_shift_freq,
    "self_reference_density":      self_ref,
})

for t in tqdm(range(TOPICS), desc="  topic cols", leave=False, ncols=80):
    df[f"topic_{t}"] = topic_matrix[:, t]

for e in tqdm(range(EMB_DIM), desc="  emb cols  ", leave=False, ncols=80):
    df[f"emb_{e}"] = emb_matrix[:, e]

df["target"] = profile_col

pbar.set_postfix_str("saving CSV")
pbar.update(1)
df.to_csv("mental_health_synthetic_dataset_10k.csv", index=False)
pbar.close()

print("\nDataset generated successfully")
print(df.shape)
print(df.head())