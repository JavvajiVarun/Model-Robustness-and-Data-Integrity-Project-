import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from scipy.sparse.linalg import LinearOperator, cg
import warnings
warnings.filterwarnings("ignore")

SEED = 1
np.random.seed(SEED)


def load_and_split():
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    print(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
    print(f"Features: {X_train.shape[1]} | Classes: {np.unique(y).tolist()}")
    return X_train, X_test, y_train, y_test, scaler


def random_label_flip_poison(X, y, poison_rate=0.12):
    rng = np.random.RandomState(SEED)
    n_poison = int(len(y) * poison_rate)
    poison_idx = rng.choice(len(y), n_poison, replace=False)
    X_p = X.copy(); y_p = y.copy()
    y_p[poison_idx] = 1 - y_p[poison_idx]
    X_p[poison_idx] += rng.normal(0, 0.05, X_p[poison_idx].shape)
    print(f"\n[Poison] Label flip: {n_poison} samples poisoned ({poison_rate*100:.0f}%)")
    return X_p, y_p, poison_idx


def pgd_adversarial_poison(X, y, model, poison_rate=0.12, step_size=0.03, n_steps=30):
    rng = np.random.RandomState(SEED)
    n_poison = int(len(y) * poison_rate)
    epsilon = 0.3
    probs = model.predict_proba(X)[:, 1]
    uncertainty = 1 - np.abs(probs - 0.5) * 2
    sample_probs = uncertainty / uncertainty.sum()
    poison_idx = rng.choice(len(y), n_poison, replace=False, p=sample_probs)
    X_p = X.copy(); y_p = y.copy(); X_orig = X[poison_idx].copy()
    for step in range(n_steps):
        p = model.predict_proba(X_p[poison_idx])[:, 1]
        y_flipped = 1 - y_p[poison_idx]
        residual = (p - y_flipped)[:, None]
        grad = residual * model.coef_[0]
        X_p[poison_idx] += step_size * np.sign(grad)
        delta = np.clip(X_p[poison_idx] - X_orig, -epsilon, epsilon)
        X_p[poison_idx] = X_orig + delta
    y_p[poison_idx] = 1 - y_p[poison_idx]
    print(f"[Poison] PGD adversarial: {n_poison} samples crafted ({poison_rate*100:.0f}%)")
    return X_p, y_p, poison_idx


def train_model(X_tr, y_tr, C=1.0):
    model = LogisticRegression(C=C, max_iter=2000, random_state=SEED, solver='lbfgs')
    model.fit(X_tr, y_tr)
    return model


def evaluate_model(model, X_te, y_te, label="Model"):
    preds = model.predict(X_te)
    probs = model.predict_proba(X_te)[:, 1]
    acc   = accuracy_score(y_te, preds)
    auc   = roc_auc_score(y_te, probs)
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(classification_report(y_te, preds, target_names=["Malignant (0)", "Benign (1)"]))
    return acc, auc


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def logistic_loss_gradient(model, X_i, y_i):
    z = X_i @ model.coef_[0] + model.intercept_[0]
    p = sigmoid(z)
    return (p - y_i) * X_i


def hessian_vector_product(model, X_train, lam=0.001):
    z = X_train @ model.coef_[0] + model.intercept_[0]
    p = sigmoid(z)
    W = p * (1 - p)
    n = len(X_train)
    def hvp(v):
        return X_train.T @ (W * (X_train @ v)) / n + lam * v
    return hvp


def inverse_hessian_vector_product(hvp_fn, v, tol=1e-4, maxiter=500):
    d = len(v)
    H_op = LinearOperator((d, d), matvec=hvp_fn)
    try:
        x, info = cg(H_op, v, rtol=tol, maxiter=maxiter)
    except TypeError:
        x, info = cg(H_op, v, tol=tol, maxiter=maxiter)
    if info != 0:
        print(f"  [Warning] CG did not fully converge (info={info})")
    return x


def compute_influence_scores(model, X_train, y_train, X_test, y_test, lam=0.001):
    print("\n[Influence] Computing Hessian-vector product function...")
    hvp_fn = hessian_vector_product(model, X_train, lam=lam)
    test_grads = np.array([logistic_loss_gradient(model, X_test[j], y_test[j]) for j in range(len(X_test))])
    g_test = test_grads.mean(axis=0)
    print("[Influence] Running conjugate gradient to invert Hessian...")
    s_test = inverse_hessian_vector_product(hvp_fn, g_test)
    print("[Influence] Scoring all training samples...")
    scores = np.zeros(len(X_train))
    for i in range(len(X_train)):
        scores[i] = -np.dot(logistic_loss_gradient(model, X_train[i], y_train[i]), s_test)
    print(f"[Influence] Done. Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    return scores


def compute_training_losses(model, X_train, y_train):
    z = X_train @ model.coef_[0] + model.intercept_[0]
    p = sigmoid(z)
    eps = 1e-9
    return -(y_train * np.log(p + eps) + (1 - y_train) * np.log(1 - p + eps))


def spectral_signature_scores(X_train, y_train):
    scores = np.zeros(len(X_train))
    for cls in [0, 1]:
        idx = np.where(y_train == cls)[0]
        X_cls = X_train[idx]
        X_centred = X_cls - X_cls.mean(axis=0)
        _, _, Vt = np.linalg.svd(X_centred, full_matrices=False)
        scores[idx] = (X_centred @ Vt[0]) ** 2
    return scores


def ensemble_anomaly_score(influence_scores, loss_scores, spectral_scores):
    def normalise(s):
        r = s - s.min()
        return r / (r.max() + 1e-9)
    return (0.6 * normalise(np.abs(influence_scores)) +
            0.3 * normalise(loss_scores) +
            0.1 * normalise(spectral_scores))


def detect_and_remove(X_train, y_train, ensemble_scores, remove_pct=0.15):
    n_remove = int(len(y_train) * remove_pct)
    remove_idx = np.argsort(ensemble_scores)[::-1][:n_remove]
    keep_mask = np.ones(len(y_train), dtype=bool)
    keep_mask[remove_idx] = False
    print(f"\n[Filter] Removing {n_remove} samples ({remove_pct*100:.0f}%)")
    print(f"[Filter] Retained {keep_mask.sum()} training samples")
    return X_train[keep_mask], y_train[keep_mask], remove_idx


def detection_metrics(detected_idx, true_poison_idx, n_train):
    detected_set = set(detected_idx)
    poison_set   = set(true_poison_idx)
    tp = len(detected_set & poison_set)
    fp = len(detected_set - poison_set)
    fn = len(poison_set - detected_set)
    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    print(f"\n[Detection Metrics]")
    print(f"  True Positives  : {tp}")
    print(f"  False Positives : {fp}")
    print(f"  False Negatives : {fn}")
    print(f"  Precision       : {precision:.3f}")
    print(f"  Recall          : {recall:.3f}")
    print(f"  F1 Score        : {f1:.3f}")
    return precision, recall, f1


def plot_results(results, influence_scores, ensemble_scores, true_poison_idx, n_train):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("ML Model Robustness — Full Analysis", fontsize=14, fontweight='bold')

    ax = axes[0, 0]
    labels = list(results.keys())
    accs   = [v[0] for v in results.values()]
    colors = ['#2ecc71', '#e74c3c', '#e74c3c', '#2ecc71']
    bars = ax.bar(labels, accs, color=colors[:len(labels)], width=0.5, edgecolor='white')
    ax.set_ylim(0.85, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Accuracy at Each Stage")
    ax.tick_params(axis='x', rotation=25)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=9)

    ax = axes[0, 1]
    poison_mask = np.zeros(n_train, dtype=bool)
    poison_mask[true_poison_idx] = True
    ax.hist(influence_scores[~poison_mask], bins=40, alpha=0.7, color='#3498db', label='Clean samples')
    ax.hist(influence_scores[poison_mask],  bins=40, alpha=0.7, color='#e74c3c', label='Poisoned samples')
    ax.set_xlabel("Influence Score")
    ax.set_ylabel("Count")
    ax.set_title("Influence Score Distribution")
    ax.legend()

    ax = axes[1, 0]
    ax.hist(ensemble_scores[~poison_mask], bins=40, alpha=0.7, color='#2ecc71', label='Clean samples')
    ax.hist(ensemble_scores[poison_mask],  bins=40, alpha=0.7, color='#e74c3c', label='Poisoned samples')
    threshold = np.percentile(ensemble_scores, 85)
    ax.axvline(threshold, color='black', linestyle='--', label='85th pct threshold')
    ax.set_xlabel("Ensemble Anomaly Score")
    ax.set_ylabel("Count")
    ax.set_title("Ensemble Detection Score Distribution")
    ax.legend()

    ax = axes[1, 1]
    ax.scatter(influence_scores[~poison_mask], ensemble_scores[~poison_mask],
               alpha=0.4, s=12, color='#3498db', label='Clean')
    ax.scatter(influence_scores[poison_mask],  ensemble_scores[poison_mask],
               alpha=0.7, s=20, color='#e74c3c', label='Poisoned', zorder=5)
    ax.set_xlabel("Influence Score")
    ax.set_ylabel("Ensemble Score")
    ax.set_title("Influence vs Ensemble Score")
    ax.legend()

    plt.tight_layout()
    plt.savefig("ml_robustness_analysis.png", dpi=150, bbox_inches='tight')
    print("\n[Plot] Saved to ml_robustness_analysis.png")
    plt.show()


def main():
    print("=" * 60)
    print("  ML MODEL ROBUSTNESS AND DATA INTEGRITY")
    print("=" * 60)

    X_train, X_test, y_train, y_test, scaler = load_and_split()

    print("\n[Step 1] Training clean baseline model...")
    model_clean = train_model(X_train, y_train)
    acc_clean, auc_clean = evaluate_model(model_clean, X_test, y_test, "Clean Baseline")

    print("\n[Step 2a] Applying random label-flip poisoning...")
    X_flip, y_flip, poison_idx_flip = random_label_flip_poison(X_train, y_train)
    model_flip = train_model(X_flip, y_flip)
    acc_flip, auc_flip = evaluate_model(model_flip, X_test, y_test, "After Label-Flip Poisoning")

    print("\n[Step 2b] Applying PGD adversarial poisoning...")
    X_pgd, y_pgd, poison_idx_pgd = pgd_adversarial_poison(X_train, y_train, model_clean)
    model_pgd = train_model(X_pgd, y_pgd)
    acc_pgd, auc_pgd = evaluate_model(model_pgd, X_test, y_test, "After PGD Adversarial Poisoning")

    print("\n[Step 3] Computing influence scores...")
    influence_scores = compute_influence_scores(
        model_pgd, X_pgd, y_pgd, X_test, y_test, lam=0.001
    )

    print("\n[Step 4] Computing auxiliary detection signals...")
    loss_scores     = compute_training_losses(model_pgd, X_pgd, y_pgd)
    spectral_scores = spectral_signature_scores(X_pgd, y_pgd)
    ensemble_scores = ensemble_anomaly_score(influence_scores, loss_scores, spectral_scores)

    n_remove = int(len(y_pgd) * 0.15)
    detected_idx = np.argsort(ensemble_scores)[::-1][:n_remove]
    detection_metrics(detected_idx, poison_idx_pgd, len(y_train))

    print("\n[Step 5] Removing suspicious samples and retraining...")
    X_filtered, y_filtered, removed_idx = detect_and_remove(
        X_pgd, y_pgd, ensemble_scores, remove_pct=0.15
    )
    model_recovered = train_model(X_filtered, y_filtered, C=10.0)
    acc_rec, auc_rec = evaluate_model(model_recovered, X_test, y_test, "After Influence Filtering (Recovery)")

    results = {
        "Clean":      (acc_clean, auc_clean),
        "Label flip": (acc_flip,  auc_flip),
        "PGD attack": (acc_pgd,   auc_pgd),
        "Recovered":  (acc_rec,   auc_rec),
    }

    print("\n" + "=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)
    print(f"  {'Stage':<25} {'Accuracy':>10} {'ROC-AUC':>10}")
    print(f"  {'-'*45}")
    for stage, (acc, auc) in results.items():
        marker = " ✓" if stage == "Recovered" and acc > acc_pgd else ""
        print(f"  {stage:<25} {acc:>10.4f} {auc:>10.4f}{marker}")

    if acc_rec > acc_pgd:
        print(f"\n   Recovery BEATS PGD attack by {(acc_rec - acc_pgd)*100:.2f}%!")
    else:
        print(f"\n    Recovery did not beat PGD attack.")

    print("\n[Step 6] Generating diagnostic plots...")
    plot_results(results, influence_scores, ensemble_scores, poison_idx_pgd, len(y_train))

    return results, influence_scores, ensemble_scores


if __name__ == "__main__":
    results, influence_scores, ensemble_scores = main()
