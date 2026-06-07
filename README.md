# Model-Robustness-and-Data-Integrity-Project-
# ML Model Robustness and Data Integrity

A study of data poisoning attacks on supervised machine learning models and defence using influence functions, applied to the breast cancer classification dataset.

## Results

| Stage | Accuracy |
|---|---|
| Clean baseline | 99.12% |
| After label-flip attack | 94.74% |
| After PGD adversarial attack | 90.35% |
| After influence function filtering | 95.61% |

The defence successfully recovered +5.26% above the PGD attack by identifying and removing poisoned training samples.

## Project Pipeline

1. Data loading — Breast cancer dataset, 455 training samples, 30 features
2. Clean baseline — Logistic regression trained on clean data
3. Label flip attack — 12% of training labels randomly flipped
4. PGD adversarial attack — Projected gradient descent crafts subtle boundary-targeted poison
5. Influence functions — Hessian-vector product + conjugate gradient to score every training sample
6. Multi-signal detection — Ensemble of influence scores, training loss, and spectral signatures
7. Recovery — Remove top suspicious samples and retrain

## Key Concepts

- Data poisoning — Injecting corrupted samples into training data to degrade model performance
- PGD attack — Smart adversarial attack targeting uncertain samples near the decision boundary
- Influence functions — Measures how much each training sample shaped the model weights
- Spectral signatures — SVD-based detection of outliers within each class cluster

## Requirements

pip install numpy scikit-learn scipy matplotlib

## Run

python ml_robustness_final.py
