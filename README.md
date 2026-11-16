# SMS Spam Detector

A Decision Tree classifier that flags SMS spam with **94.8% accuracy** while keeping false positives on legitimate messages to **1.8%**.

## Why this project
Spam detection is a class-imbalance problem in disguise — only ~13% of messages are spam. I wanted to build a model that performs well *and* doesn't annoy users with false alarms, and to evaluate it with metrics that reflect that imbalance rather than accuracy alone.

## Dataset
`spam.csv` (latin-1 encoded) — 5,572 SMS messages labeled `ham` or `spam`. The raw file contains extra mostly-empty columns (`Unnamed: 2`–`4`); only `v1` (label) and `v2` (text) are used.

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
```bash
pip install -r requirements.txt
# place spam.csv in the repo root
jupyter notebook email_spamer_detector.ipynb
```
## What I learned
Why accuracy alone misleads on imbalanced data — precision/recall tell the real story

How a stratified split preserves class distribution

The precision/recall trade-off and why keeping false positives low matters for a spam filter

How tree depth controls overfitting
