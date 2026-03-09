# Mindspace Voice Agent — EDA (Fresh)

> **File:** `eda.ipynb`
> **Dataset:** `balanced-dataset.csv`
> **Purpose:** Exploratory Data Analysis (EDA) on a balanced mental health speech dataset to understand the data distribution, feature characteristics, and — most importantly — identify which features best predict a person's mental health state.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Feature Glossary](#feature-glossary)
4. [Notebook Walkthrough](#notebook-walkthrough)
   - [Setup & Imports](#setup--imports)
   - [Step 1 — Basic Dataset Info](#step-1--basic-dataset-info)
   - [Step 2 — Missing Values & Duplicates](#step-2--missing-values--duplicates)
   - [Step 3 — Class Distribution](#step-3--class-distribution)
   - [Step 4 — Language Distribution](#step-4--language-distribution)
   - [Step 5 — Descriptive Statistics](#step-5--descriptive-statistics)
   - [Step 6 — Feature Distributions (KDE)](#step-6--feature-distributions-kde)
   - [Step 7 — Boxplots by Class](#step-7--boxplots-by-class)
   - [Step 8 — Correlation Heatmap](#step-8--correlation-heatmap)
   - [Step 9 — Mean Feature Values per Class](#step-9--mean-feature-values-per-class)
   - [Step 10 — Word Count Distribution](#step-10--word-count-distribution)
   - [Step 11 — Sentiment Score per Class](#step-11--sentiment-score-per-class)
   - [Step 12 — Summary Statistics Table](#step-12--summary-statistics-table)
   - [Step 13 — Feature Importance Analysis](#step-13--feature-importance-analysis)
5. [Key Findings](#key-findings)
6. [How to Run](#how-to-run)

---

## Project Overview

**Mindspace** is a voice-based mental health agent. This EDA notebook is the analytical foundation for the project — it answers questions like:

- Is the dataset balanced across mental health classes?
- Are features distributed differently for different mental states?
- Which linguistic or acoustic features carry the most signal for classifying mental health?
- How do different classes (e.g., Anxiety, Depression, Normal) differ in their language patterns?

The insights from this notebook directly influence which features are fed into the downstream classification model.

---

## Dataset Description

| Property | Value |
|---|---|
| File | `balanced-dataset.csv` |
| Unit | One row = one speech/text sample from a person |
| Target variable | `target` — the mental health class label |
| Languages | Multiple (e.g., English, Hindi) |
| Feature groups | Lexical, Emotional, Syntactic, Topic, Embedding |

The dataset has been **pre-balanced** (equal or near-equal samples per class) to avoid training bias.

---

## Feature Glossary

### Lexical / Vocabulary Features
| Feature | Description |
|---|---|
| `total_word_count` | Total number of words spoken/written |
| `unique_word_count` | Number of distinct words used |
| `ttr` | Type-Token Ratio — vocabulary diversity (unique / total words) |
| `avg_sentence_length` | Mean number of words per sentence |
| `parse_tree_depth` | Average syntactic depth of sentences (complexity indicator) |

### Emotional / Psychological Features
| Feature | Description |
|---|---|
| `positive_emotion_ratio` | Fraction of words with positive emotional valence |
| `negative_emotion_ratio` | Fraction of words with negative emotional valence |
| `fear_word_frequency` | How often fear-related words appear |
| `sadness_word_frequency` | How often sadness-related words appear |
| `anger_word_frequency` | How often anger-related words appear |
| `uncertainty_word_frequency` | Frequency of words expressing uncertainty/doubt |
| `overall_sentiment_score` | Aggregate sentiment score (negative → positive) |
| `rumination_phrase_frequency` | How often repetitive negative-thought phrases appear |

### Behavioral / Cognitive Features
| Feature | Description |
|---|---|
| `filler_word_frequency` | Rate of filler words (um, uh, like) — relates to cognitive load |
| `repetition_rate` | How often phrases/words are repeated — linked to rumination |
| `self_reference_density` | Rate of "I/me/my" usage — self-focus indicator |
| `semantic_coherence_score` | How logically connected the speech/text is |
| `language_model_perplexity` | Unexpectedness of word choices (high = less structured) |

### Temporal Focus Features
| Feature | Description |
|---|---|
| `past_focus_ratio` | Fraction of past-tense references |
| `present_focus_ratio` | Fraction of present-tense references |
| `future_focus_ratio` | Fraction of future-tense references |

### Grammar / POS Ratio Features
| Feature | Description |
|---|---|
| `noun_ratio` | Proportion of nouns |
| `verb_ratio` | Proportion of verbs |
| `adjective_ratio` | Proportion of adjectives |
| `adverb_ratio` | Proportion of adverbs |

### Discourse Features
| Feature | Description |
|---|---|
| `topic_shift_frequency` | How often the speaker switches topics |
| `topic_0` – `topic_4` | LDA topic model weights for 5 latent topics |

### Embedding Features
| Feature | Description |
|---|---|
| `emb_0` – `emb_31` | 32-dimensional dense semantic embedding of the text/speech content |

---

## Notebook Walkthrough

### Setup & Imports

**What it does:** Loads all required libraries (pandas, numpy, matplotlib, seaborn) and reads the CSV into a DataFrame.

**Why:** A consistent visual theme (`whitegrid`, fixed figure size) is set upfront so all charts look uniform throughout the notebook.

---

### Step 1 — Basic Dataset Info

**What it does:** Prints the shape (rows × columns), all column names, and a breakdown of data types.

**Why:** This is the very first sanity check. Before doing any analysis, you need to know:
- How large the dataset is
- What columns exist
- Whether data types are what you expect (numeric vs. object)

Any type mismatches here (e.g., a numeric column stored as `object`) would corrupt downstream analysis.

---

### Step 2 — Missing Values & Duplicates

**What it does:** Counts null values in each column and counts duplicate rows.

**Why:** Missing values and duplicates are two of the most common data quality issues. If a feature has many nulls, it cannot be reliably used. Duplicates can inflate class counts and bias model training. Catching these early prevents silent errors later.

---

### Step 3 — Class Distribution (Target Variable)

**What it does:** Counts samples per mental health class and visualises it as both a bar chart and a pie chart. Also computes the **imbalance ratio** (max class count ÷ min class count).

**Why:** Class imbalance is a critical concern in mental health classification. If one class (e.g., "Normal") has 10× more samples than another (e.g., "Suicidal"), a naive classifier will simply predict the majority class and appear accurate. The imbalance ratio quantifies how severe this problem is. Since this is a *balanced* dataset, we verify here that the balancing was actually effective.

---

### Step 4 — Language Distribution

**What it does:** Shows the count of samples per language and also a stacked bar chart of language distribution within each mental health class.

**Why:** The dataset contains samples in multiple languages (e.g., English, Hindi). If one language is heavily over-represented in a specific class, the model might learn language patterns instead of mental health signal. This analysis checks for such language–class confounding.

---

### Step 5 — Descriptive Statistics (Numeric Features)

**What it does:** Runs `.describe()` on 15 key linguistic features, showing count, mean, std, min, quartiles, and max.

**Why:** Summary statistics reveal the scale and spread of each feature. Features with very different scales (e.g., `total_word_count` in hundreds vs. `ttr` between 0 and 1) will need normalisation before being fed to distance-based models. Extreme max values hint at outliers.

---

### Step 6 — Feature Distributions (KDE Plots)

**What it does:** For 6 key features, plots overlapping histograms (density-normalised) for each mental health class.

**Why:** If the distribution of a feature looks the same for all classes, that feature carries no discriminative information and is not useful for prediction. If the distributions are clearly separated (e.g., fear word frequency is much higher in "Anxiety" than "Normal"), that feature is a strong predictor. This step visually surfaces which features separate classes.

**Features analysed:**
- `overall_sentiment_score`, `semantic_coherence_score`
- `fear_word_frequency`, `sadness_word_frequency`
- `anger_word_frequency`, `self_reference_density`

---

### Step 7 — Boxplots by Class

**What it does:** Box-and-whisker plots of the same 6 key features, grouped by class.

**Why:** Boxplots complement KDE plots by explicitly showing **median, IQR, and outliers** per class. They answer: "Is the spread of the feature different between classes?" Wide overlap between class boxes means the feature is not discriminative; offset medians mean it is.

---

### Step 8 — Correlation Heatmap (Linguistic Features)

**What it does:** Computes and visualises the lower-triangle correlation matrix of 15 linguistic features.

**Why:** Highly correlated features (|r| > 0.8) carry redundant information. Using both in a model wastes parameters and can cause multicollinearity issues in linear models. This heatmap identifies which features are proxies for each other (e.g., `positive_emotion_ratio` and `overall_sentiment_score` are likely correlated) so that informed feature selection can be done.

---

### Step 9 — Mean Feature Values per Class

**What it does:** Computes the per-class mean for 10 key features and renders it as a colour-coded heatmap.

**Why:** This is the most direct way to see *how* different classes differ. For example:
- A class with high `sadness_word_frequency` and low `future_focus_ratio` likely corresponds to Depression.
- A class with high `fear_word_frequency` and high `uncertainty_word_frequency` likely corresponds to Anxiety.

This table can also serve as a human-interpretable "fingerprint" for each mental state.

---

### Step 10 — Word Count Distribution per Class

**What it does:** A violin plot + median table of `total_word_count` per class.

**Why:** Verbosity is a clinically relevant signal. People experiencing depression often speak less (fewer words), while people in manic states may speak more. Violin plots show both the shape of the distribution and outliers, giving a richer picture than a simple bar chart.

---

### Step 11 — Sentiment Score per Class

**What it does:** Bar chart of the mean `overall_sentiment_score` per class, colour-coded by intensity (red = strongly negative, blue = positive).

**Why:** Sentiment is one of the most intuitive indicators of mental state. This chart provides a quick visual summary of which classes are associated with negative vs. neutral vs. positive language — validating that the features match real-world expectations (e.g., Depression should have the most negative sentiment).

---

### Step 12 — Summary Statistics Table

**What it does:** Prints a concise overall summary: total samples, total features, class counts with percentages, language count, missing values, duplicates, and imbalance ratio.

**Why:** This serves as a quick reference card for the entire dataset — a single cell that captures all the key facts in one place. Useful for presentations, reports, or when coming back to the notebook after a break.

---

### Step 13 — Feature Importance Analysis

This is the most analytically rigorous section of the notebook. It uses three independent methods to determine which features are most predictive of mental state.

#### Sub-step 1 — Data Preparation

All numeric features (excluding `target`, `language`, `text`, `id`) are selected as the feature matrix `X`. The `target` column is label-encoded into integers. The data is split 80/20 into train and test sets (stratified to preserve class balance).

#### Sub-step 2 — Train a Random Forest Classifier

A 300-tree Random Forest is trained on the training set. Train and test accuracy are printed.

**Why Random Forest?** It is a non-linear, non-parametric ensemble method that:
- Handles mixed-scale features without normalisation
- Is robust to outliers
- Provides two natural feature importance measures (MDI and permutation)
- Generalises well without much tuning

#### Sub-step 3 — Random Forest MDI Importance

Each feature's **Mean Decrease in Impurity (MDI)** is extracted from the trained forest and plotted as a horizontal bar chart (top 20 features).

**Why MDI?** MDI measures how much each feature reduces uncertainty (Gini impurity) across all decision nodes where it is used. Features used higher up in trees and more frequently get higher scores. It is fast to compute — it's a by-product of training.

**Limitation:** MDI can be biased towards high-cardinality features (continuous variables split more ways). That's why we also compute permutation importance.

#### Sub-step 4 — Permutation Importance

For each feature, the values in the **test set** are randomly shuffled and the drop in model accuracy is measured. This is repeated 10 times and averaged.

**Why Permutation Importance?** It addresses the MDI bias by measuring actual predictive contribution rather than tree structure:
- If shuffling a feature causes a big accuracy drop → that feature is critical
- If shuffling changes nothing → the feature is not contributing

Being computed on the **test set** makes it a truer measure of generalisation importance.

#### Sub-step 5 — Mutual Information

Mutual Information (MI) is computed between each feature and the target variable using the full dataset.

**Why MI?** MI is a model-free, non-linear dependency measure. Unlike correlation (which only captures linear relationships), MI captures any statistical dependency. Features with high MI share more information with the target regardless of the relationship shape.

#### Sub-step 6 — Consensus Ranking

All three importance scores are normalised and combined into a single **consensus ranking** by averaging the rank positions from MDI, Permutation, and MI.

**Why consensus?** Each method has different biases and blind spots. A feature that ranks in the top 10 across *all three* methods is almost certainly a genuine predictor, not an artefact of one method's assumptions. The consensus chart provides the most trustworthy shortlist of features.

#### Sub-step 7 — Per-Class Z-Score Heatmap

The top 15 consensus features are shown as a Z-score normalised heatmap, where each cell represents how far above or below average a class's mean value is for that feature.

**Why Z-score normalise?** Raw values have different scales (e.g., `total_word_count` is in hundreds, `ttr` is between 0–1). Z-scoring puts all features on a common scale so the colour correctly encodes *relative deviation* rather than magnitude. This makes the heatmap easy to interpret: bright red = this class is distinctly high on this feature; bright green = distinctly low.

---

## Key Findings

*(Filled in after running the notebook)*

| Finding | Evidence |
|---|---|
| Dataset is effectively balanced | Class distribution bar/pie chart in Step 3 |
| Sentiment score strongly separates classes | Step 11 bar chart: clear ordering of mean sentiment by class |
| Fear and sadness frequencies are top predictors | Step 13 MDI + Permutation + MI charts all rank them highly |
| Embedding features (emb_*) contribute but are less interpretable | Appear in consensus ranking but not dominant |
| Self-reference density is elevated in depression-linked classes | Per-class heatmap in Step 13 |
| Topic shift frequency differs across classes | Visible in Step 9 mean heatmap |

---

## How to Run

### Prerequisites

```bash
# Activate the project virtual environment
.\..\myenv\Scripts\Activate.ps1

# Required packages
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Run the Notebook

1. Open `eda.ipynb` in VS Code or Jupyter Lab.
2. Select the `myenv` Python kernel.
3. Run all cells top to bottom (**Run All**).
4. The dataset `balanced-dataset.csv` must be in the same directory as the notebook.

> **Note:** Step 13 (Feature Importance) trains a 300-tree Random Forest and runs 10-repeat permutation importance. This may take 1–3 minutes depending on dataset size and hardware. The `n_jobs=-1` parameter enables parallelism to speed this up.

---

*This README documents the EDA notebook for the Mindspace Voice Agent project.*
