# Wine Dataset — Train/Validation/Test Split Experiment

Logistic regression on the scikit-learn Wine dataset, comparing different
train/validation/test split ratios and their effect on model performance.

## Overview

This project explores how the size of the train/validation/test split
affects a logistic regression classifier's accuracy on the Wine dataset
(178 samples, 13 features, 3 classes). Two ratios were tested:

- **70:15:15** (train:validation:test)
- **60:20:20** (train:validation:test)

Both splits used `stratify=y` to preserve class balance, and
`random_state=42` for reproducibility.

## Results

| Split     | Train | Val | Test | Val Accuracy | Test Accuracy |
|-----------|-------|-----|------|--------------|----------------|
| 70:15:15  | 124   | 27  | 27   | 0.9630       | 0.9630         |
| 60:20:20  | 106   | 36  | 36   | 0.9444       | 0.9722         |

### Classification report — 70:15:15 (test set)

| Class    | Precision | Recall | F1-score | Support |
|----------|-----------|--------|----------|---------|
| class_0  | 1.00      | 1.00   | 1.00     | 9       |
| class_1  | 0.92      | 1.00   | 0.96     | 11      |
| class_2  | 1.00      | 0.86   | 0.92     | 7       |

### Classification report — 60:20:20 (test set)

| Class    | Precision | Recall | F1-score | Support |
|----------|-----------|--------|----------|---------|
| class_0  | 1.00      | 1.00   | 1.00     | 12      |
| class_1  | 0.93      | 1.00   | 0.97     | 14      |
| class_2  | 1.00      | 0.90   | 0.95     | 10      |

## Key findings

- Split ratio had a small effect on this dataset — Wine is small (178
  samples) and its 3 classes are close to linearly separable, so both
  splits reached ~94–97% accuracy.
- `class_2` had the lowest recall in both splits, suggesting it's the
  hardest class to separate regardless of split ratio.
- With test sets this small (27–36 samples), a single misclassified
  sample moves accuracy by several percentage points — the difference
  between the two splits is within that noise band.
- Both runs raised a `ConvergenceWarning` from `lbfgs` (max_iter=1000
  reached before convergence). Accuracy was unaffected here, but
  scaling the features (e.g. `StandardScaler`) or raising `max_iter`
  would resolve the warning.

## How to run

```bash
pip install scikit-learn jupyter
jupyter notebook Module_5_Data_Split.ipynb
```

## Repository structure

```
.
├── README.md
└── Module_5_Data_Split.ipynb   # both splits (70:15:15 and 60:20:20),
                                  # run end-to-end with saved outputs
```
