# SMS Spam Detector

A Decision Tree classifier that flags SMS spam with **94.8% accuracy** while keeping false positives on legitimate messages to **1.8%**.

## Why this project
Spam detection is a class-imbalance problem in disguise — only ~13% of messages are spam. I wanted to build a model that performs well *and* doesn't annoy users with false alarms, and to evaluate it with metrics that reflect that imbalance rather than accuracy alone.

## Dataset

**File:** [`spam.csv`](./spam.csv) (included in this repo, latin-1 encoded)

The SMS Spam Collection — 5,572 SMS messages labeled `ham` (legitimate) or `spam`. Originally published by Almeida et al. and distributed via the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection).

- **Rows:** 5,572
- **Columns used:** `v1` (label: ham/spam), `v2` (message text)
- **Columns ignored:** `Unnamed: 2`–`Unnamed: 4` (mostly empty artifacts of the CSV export)
- **Class distribution:** ~87% ham / ~13% spam

The raw file is loaded with `encoding='latin-1'` because it contains non-UTF-8 characters from the original UK SMS corpus.

## Approach
1. **Vectorization:** `CountVectorizer` (bag-of-words) → **8,672 features**
2. **Split:** stratified 70/30 → **3,900 train / 1,672 test**, preserving the ~13% spam ratio
3. **Model:** `DecisionTreeClassifier`, `max_depth=5`, `random_state=42`
4. **Evaluation:** accuracy + full `classification_report` + confusion-matrix analysis

## Results

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Ham | 0.96 | 0.98 | 0.97 | 1,448 |
| Spam | 0.86 | 0.73 | 0.79 | 224 |

**Overall accuracy: 94.8%** (1,672 test messages).

**False-positive rate on ham: 1.80%** — only 1.8% of legitimate messages were wrongly flagged as spam.

The precision/recall trade-off on the spam class is deliberate: I kept `max_depth` shallow (5) to avoid overfitting, accepting lower spam recall in exchange for very few false alarms.

## Visualizing the model
The notebook includes a `plot_tree` render of the fitted tree, so you can see exactly which words drive each split.

## Run it

Everything needed to reproduce the results is in this repo — including the dataset.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the notebook
jupyter notebook email_spamer_detector.ipynb
```

Then run all cells (**Cell → Run All**). The notebook loads `spam.csv` from the repo root automatically, so there's no download step.

**Expected output** (last cells):
- Accuracy: `0.9479...` (≈ 94.8%)
- `classification_report` showing spam precision 0.86 / recall 0.73 / F1 0.79
- Ham false-positive rate: `1.80%`
```
## What I learned
Why accuracy alone misleads on imbalanced data — precision/recall tell the real story

How a stratified split preserves class distribution

The precision/recall trade-off and why keeping false positives low matters for a spam filter

How tree depth controls overfitting
