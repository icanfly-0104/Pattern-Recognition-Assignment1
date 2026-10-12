"""Part 5: Iris experiment with 100 added Gaussian noise dimensions.

Experiments inspired by Parts 2-4, using only a Mahalanobis-distance-based
classifier. Compares (a) raw vs standardized original Iris features, (b) noise
feature count, (c) PCA projection / ridge whitening, (d) LDA / ridge-LDA, and
(e) noise variance levels.

Notes on fair evaluation:
- StandardScaler is fitted to the four original Iris features using TRAIN only.
- The added noise columns are not standardized; their variance is the parameter
  being studied.
- PCA and LDA projections are fitted using TRAIN only.
- Since Iris has only 150 samples, a 50/50 split has 75 training observations;
  centered PCA therefore has at most 74 nonzero dimensions. For the PCA 1..100
  sweep, this script uses an 80/20 split (120 train observations), which supports
  100 projected dimensions without using test samples to fit PCA.
- With >100 features but few Iris samples, ordinary LDA's within-class scatter
  is singular. gamma=0 uses a tiny numerical floor only so the generalized
  eigenproblem can be solved; gamma>0 is ridge-regularized LDA.
"""

from pathlib import Path
import csv
import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = Path(__file__).resolve().parent
RANDOM_SEED = 42
DEFAULT_REPEATS = 5
DEFAULT_NOISE_DIM = 100
DEFAULT_NOISE_VARIANCE = 0.5
MAHALANOBIS_RIDGE = 0.01
NOISE_VARIANCES = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
PCA_GAMMAS = [None, 0.0, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
LDA_GAMMAS = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0]


def load_iris_with_noise(noise_dim=100, noise_variance=0.5, seed=42):
    """Load sklearn's Iris data and append independent class-uninformative noise."""
    iris = load_iris()
    X_iris = iris.data.astype(float)
    y = iris.target.astype(int)

    if noise_variance < 0:
        raise ValueError("noise_variance must be non-negative")
    rng = np.random.default_rng(seed)
    X_noise = rng.normal(
        loc=0.0,
        scale=np.sqrt(noise_variance),
        size=(len(y), noise_dim),
    )
    return np.hstack([X_iris, X_noise]), y


def split_and_preprocess(X, y, train_ratio=0.5, seed=42, normalize_iris=False):
    """Stratified split; optionally standardize only the four Iris features."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=train_ratio,
        random_state=seed + 1000,
        stratify=y,
    )
    X_train = X_train.copy()
    X_test = X_test.copy()

    if normalize_iris:
        scaler = StandardScaler()
        X_train[:, :4] = scaler.fit_transform(X_train[:, :4])
        X_test[:, :4] = scaler.transform(X_test[:, :4])

    return X_train, X_test, y_train, y_test


class MahalanobisClassifier:
    """Classify by distance to class means using a regularized global covariance.

    The inverse covariance is estimated from all training rows (as in Parts 2-4),
    with ridge_alpha scaled by average feature variance for numerical stability.
    """

    def __init__(self, ridge_alpha=MAHALANOBIS_RIDGE):
        self.ridge_alpha = float(ridge_alpha)

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.class_means_ = np.vstack([X[y == cls].mean(axis=0) for cls in self.classes_])

        covariance = np.cov(X, rowvar=False, ddof=1)
        covariance = np.atleast_2d(covariance)
        mean_variance = max(float(np.trace(covariance) / covariance.shape[0]), 1e-12)
        regularized_covariance = covariance + (
            self.ridge_alpha * mean_variance * np.eye(covariance.shape[0])
        )
        # The relative ridge term ensures this is invertible even when features > samples.
        self.precision_ = np.linalg.inv(regularized_covariance)
        return self

    def predict(self, X):
        delta = X[:, None, :] - self.class_means_[None, :, :]
        dist2 = np.einsum("nci,ij,ncj->nc", delta, self.precision_, delta, optimize=True)
        return self.classes_[np.argmin(dist2, axis=1)]


def accuracy_pair(X_train, y_train, X_test, y_test):
    clf = MahalanobisClassifier().fit(X_train, y_train)
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))
    return float(train_acc), float(test_acc)


def write_csv(path, rows, fieldnames=None):
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_curves(record_list, keys, n_points):
    out = {key: np.zeros(n_points, dtype=float) for key in keys}
    for key in keys:
        values = [r[key] for r in record_list]
        out[key] = np.mean(np.asarray(values, dtype=float), axis=0)
    return out


# ============================================================
# Part 2 repetition: noise-dimension sweep and train/test splits
# ============================================================

def experiment_noise_dimensions(repeats=5, max_noise_dim=100, noise_variance=0.5):
    rows = []
    summary = {}
    dims = np.arange(1, max_noise_dim + 1)
    for normalize in (False, True):
        for train_ratio in (0.5, 0.2):
            train_curves, test_curves = [], []
            for rep in range(repeats):
                tr_curve = np.zeros(max_noise_dim)
                te_curve = np.zeros(max_noise_dim)
                seed = RANDOM_SEED + rep
                # Keep feature/noise generation paired across preprocessing settings.
                X_full, y = load_iris_with_noise(
                    noise_dim=max_noise_dim,
                    noise_variance=noise_variance,
                    seed=seed,
                )
                Xtr0, Xte0, ytr, yte = split_and_preprocess(
                    X_full, y, train_ratio=train_ratio, seed=seed,
                    normalize_iris=normalize,
                )
                # Extend noise dimensions one at a time using nested feature prefixes.
                # Every prefix uses the same train/test split and noise samples.
                for d in dims:
                    Xtr = Xtr0[:, :4 + d]
                    Xte = Xte0[:, :4 + d]
                    tr_acc, te_acc = accuracy_pair(Xtr, ytr, Xte, yte)
                    tr_curve[d - 1] = tr_acc
                    te_curve[d - 1] = te_acc
                    rows.append({
                        "experiment": "noise_dimensions",
                        "repeat": rep,
                        "normalize_iris_features": normalize,
                        "train_ratio": train_ratio,
                        "noise_variance": noise_variance,
                        "noise_dimensions": d,
                        "projected_dimensions": "",
                        "method": "Mahalanobis",
                        "gamma": "",
                        "train_accuracy": tr_acc,
                        "test_accuracy": te_acc,
                    })
                train_curves.append(tr_curve)
                test_curves.append(te_curve)
            summary[(normalize, train_ratio)] = {
                "train": np.mean(train_curves, axis=0),
                "test": np.mean(test_curves, axis=0),
                "train_std": np.std(train_curves, axis=0, ddof=1) if repeats > 1 else np.zeros(max_noise_dim),
                "test_std": np.std(test_curves, axis=0, ddof=1) if repeats > 1 else np.zeros(max_noise_dim),
            }
            print(
                f"[Noise dimensions] normalize={normalize}, train_ratio={train_ratio:.0%}: "
                f"test acc at d={max_noise_dim}: {summary[(normalize, train_ratio)]['test'][-1]:.3f}"
            )
    write_csv(OUTPUT_DIR / "part5_noise_dimension_results.csv", rows)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    for row, normalize in enumerate((False, True)):
        for col, train_ratio in enumerate((0.5, 0.2)):
            ax = axes[row, col]
            item = summary[(normalize, train_ratio)]
            ax.plot(dims, item["train"], label="Train accuracy", linewidth=1.8)
            ax.plot(dims, item["test"], label="Test accuracy", linewidth=1.8)
            ax.fill_between(dims, item["test"] - item["test_std"], item["test"] + item["test_std"], alpha=0.12)
            ax.set_title(
                f"{'Standardized Iris features' if normalize else 'Raw Iris features'}\n"
                f"Train/test = {train_ratio:.0%}/{1-train_ratio:.0%}"
            )
            ax.set_xlabel("Number of added noise dimensions")
            ax.set_ylabel("Accuracy")
            ax.set_ylim(0, 1.03)
            ax.grid(alpha=0.25)
            ax.legend()
    fig.suptitle(f"Mahalanobis accuracy vs noise dimensions (noise variance={noise_variance:g})")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "part5_noise_dimensions_accuracy.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return summary


# ============================================================
# Part 3 repetition: PCA spectrum, dimension sweep, ridge whitening
# ============================================================

def whiten_scores(scores, eigenvalues, gamma):
    if gamma is None:
        return scores
    return scores / np.sqrt(np.maximum(eigenvalues + gamma, 1e-12))[None, :]


def diagonal_mahalanobis_accuracy_curves(Z_train, y_train, Z_test, y_test, ridge_alpha=MAHALANOBIS_RIDGE):
    """Fast train/test Mahalanobis curves for all leading feature prefixes.

    PCA scores are uncorrelated over the training set. As in Part 3, use the
    diagonal covariance in projected coordinates plus ridge stabilization.
    """
    classes = np.unique(y_train)
    means = np.vstack([Z_train[y_train == c].mean(axis=0) for c in classes])
    variances = np.var(Z_train, axis=0, ddof=1)
    weights = 1.0 / np.maximum(variances + ridge_alpha, 1e-12)

    # Shape after broadcast: samples x classes x dimensions.
    dtr = (Z_train[:, None, :] - means[None, :, :]) ** 2 * weights[None, None, :]
    dte = (Z_test[:, None, :] - means[None, :, :]) ** 2 * weights[None, None, :]
    dtr_cumulative = np.cumsum(dtr, axis=2)
    dte_cumulative = np.cumsum(dte, axis=2)
    train_pred = classes[np.argmin(dtr_cumulative, axis=1)]  # samples x dims
    test_pred = classes[np.argmin(dte_cumulative, axis=1)]
    train_curve = np.mean(train_pred == y_train[:, None], axis=0)
    test_curve = np.mean(test_pred == y_test[:, None], axis=0)
    return train_curve, test_curve


def experiment_pca(repeats=5, noise_dim=100, noise_variance=0.5, train_ratio=0.8):
    """PCA sweep through 100 PCs; 80/20 split gives enough training rank on Iris."""
    gamma_labels = [("PCA only", None)] + [(f"gamma={g:g}", g) for g in PCA_GAMMAS if g is not None]
    max_dim = min(100, 4 + noise_dim)
    curves = {
        (normalize, label, metric): []
        for normalize in (False, True)
        for label, _ in gamma_labels
        for metric in ("train", "test")
    }
    spectrum_by_norm = {False: [], True: []}
    rows = []

    for rep in range(repeats):
        seed = RANDOM_SEED + rep
        X_full, y = load_iris_with_noise(noise_dim, noise_variance, seed)
        for normalize in (False, True):
            Xtr, Xte, ytr, yte = split_and_preprocess(
                X_full, y, train_ratio=train_ratio, seed=seed, normalize_iris=normalize
            )
            n_comp = min(max_dim, Xtr.shape[0] - 1, Xtr.shape[1])
            pca = PCA(n_components=n_comp, svd_solver="full")
            pca.fit(Xtr)
            scores_tr = pca.transform(Xtr)
            scores_te = pca.transform(Xte)
            eigenvalues = pca.explained_variance_
            spectrum_by_norm[normalize].append(eigenvalues.copy())

            for label, gamma in gamma_labels:
                Ztr = whiten_scores(scores_tr, eigenvalues, gamma)
                Zte = whiten_scores(scores_te, eigenvalues, gamma)
                tr_curve, te_curve = diagonal_mahalanobis_accuracy_curves(
                    Ztr, ytr, Zte, yte
                )
                curves[(normalize, label, "train")].append(tr_curve)
                curves[(normalize, label, "test")].append(te_curve)
                for k in range(1, n_comp + 1):
                    rows.append({
                        "experiment": "PCA_whitening",
                        "repeat": rep,
                        "normalize_iris_features": normalize,
                        "train_ratio": train_ratio,
                        "noise_variance": noise_variance,
                        "noise_dimensions": noise_dim,
                        "projected_dimensions": k,
                        "method": label,
                        "gamma": "" if gamma is None else gamma,
                        "train_accuracy": tr_curve[k - 1],
                        "test_accuracy": te_curve[k - 1],
                    })
    write_csv(OUTPUT_DIR / "part5_pca_results.csv", rows)

    # Plot eigenvalue spectrum averaged over repeated training splits.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7), sharey=True)
    for ax, normalize in zip(axes, (False, True)):
        eigs = np.mean(np.stack(spectrum_by_norm[normalize]), axis=0)
        ax.plot(np.arange(1, len(eigs) + 1), eigs, marker=".", markersize=3, linewidth=1.5)
        ax.set_title("Standardized 4 Iris features" if normalize else "Raw 4 Iris features")
        ax.set_xlabel("Principal component index")
        ax.set_ylabel("Eigenvalue")
        ax.grid(alpha=0.25)
    fig.suptitle(f"PCA eigenvalue spectrum (100 noise dims, variance={noise_variance:g}, train/test=80/20)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "part5_pca_eigenvalue_spectrum.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Train and test accuracy curves, normalization columns.
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True)
    colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.95, len(gamma_labels)))
    for col, normalize in enumerate((False, True)):
        for row, metric in enumerate(("train", "test")):
            ax = axes[row, col]
            for color, (label, _) in zip(colors, gamma_labels):
                series = np.mean(np.stack(curves[(normalize, label, metric)]), axis=0)
                ax.plot(np.arange(1, len(series) + 1), series, label=label, color=color, linewidth=1.4)
            ax.set_title(
                f"{'Standardized Iris features' if normalize else 'Raw Iris features'} — {metric} accuracy"
            )
            ax.set_xlabel("Number of retained principal components")
            ax.set_ylabel("Accuracy")
            ax.set_ylim(0, 1.03)
            ax.grid(alpha=0.25)
            if row == 0 and col == 0:
                ax.legend(fontsize=8, ncol=2)
    fig.suptitle("Mahalanobis accuracy after PCA / ridge whitening")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "part5_pca_train_test_accuracy.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return curves


# ============================================================
# Part 4 repetition: LDA and regularized LDA
# ============================================================

def fit_lda_projection(X_train, y_train, gamma=0.0):
    """Fit supervised LDA projection using a ridge-regularized within scatter."""
    classes = np.unique(y_train)
    n_samples, n_features = X_train.shape
    global_mean = X_train.mean(axis=0)
    Sw = np.zeros((n_features, n_features), dtype=float)
    Sb = np.zeros((n_features, n_features), dtype=float)

    for cls in classes:
        Xc = X_train[y_train == cls]
        mean_c = Xc.mean(axis=0)
        residual = Xc - mean_c
        Sw += residual.T @ residual
        diff = (mean_c - global_mean).reshape(-1, 1)
        Sb += len(Xc) * (diff @ diff.T)

    Sw /= max(n_samples - len(classes), 1)
    Sb /= max(n_samples, 1)
    scale = max(float(np.trace(Sw) / n_features), 1e-12)
    numerical_floor = 1e-8 * scale if gamma == 0 else 0.0
    ridge = max(gamma * scale, numerical_floor)
    Sw_reg = Sw + ridge * np.eye(n_features)

    vals, vecs = eigh(Sb, Sw_reg, check_finite=False)
    order = np.argsort(vals)[::-1]
    n_components = min(len(classes) - 1, n_features)
    W = vecs[:, order[:n_components]]
    return W


def experiment_lda(repeats=5, noise_dim=100, noise_variance=0.5, train_ratio=0.5):
    rows = []
    results = {
        (normalize, gamma, k, metric): []
        for normalize in (False, True)
        for gamma in LDA_GAMMAS
        for k in (1, 2)
        for metric in ("train", "test")
    }
    pca_baseline = {
        (normalize, k, metric): []
        for normalize in (False, True)
        for k in (1, 2)
        for metric in ("train", "test")
    }

    for rep in range(repeats):
        seed = RANDOM_SEED + rep
        X_full, y = load_iris_with_noise(noise_dim, noise_variance, seed)
        for normalize in (False, True):
            Xtr, Xte, ytr, yte = split_and_preprocess(
                X_full, y, train_ratio=train_ratio, seed=seed, normalize_iris=normalize
            )
            # PCA baseline at dimensions 1 and 2, using train-only PCA.
            pca = PCA(n_components=2, svd_solver="full").fit(Xtr)
            P_tr, P_te = pca.transform(Xtr), pca.transform(Xte)
            for k in (1, 2):
                tr_acc, te_acc = accuracy_pair(P_tr[:, :k], ytr, P_te[:, :k], yte)
                pca_baseline[(normalize, k, "train")].append(tr_acc)
                pca_baseline[(normalize, k, "test")].append(te_acc)
                rows.append({
                    "experiment": "PCA_baseline_for_LDA",
                    "repeat": rep,
                    "normalize_iris_features": normalize,
                    "train_ratio": train_ratio,
                    "noise_variance": noise_variance,
                    "noise_dimensions": noise_dim,
                    "projected_dimensions": k,
                    "method": "PCA only",
                    "gamma": "",
                    "train_accuracy": tr_acc,
                    "test_accuracy": te_acc,
                })

            for gamma in LDA_GAMMAS:
                W = fit_lda_projection(Xtr, ytr, gamma=gamma)
                Ltr, Lte = Xtr @ W, Xte @ W
                for k in (1, 2):
                    tr_acc, te_acc = accuracy_pair(Ltr[:, :k], ytr, Lte[:, :k], yte)
                    results[(normalize, gamma, k, "train")].append(tr_acc)
                    results[(normalize, gamma, k, "test")].append(te_acc)
                    rows.append({
                        "experiment": "LDA_regularization",
                        "repeat": rep,
                        "normalize_iris_features": normalize,
                        "train_ratio": train_ratio,
                        "noise_variance": noise_variance,
                        "noise_dimensions": noise_dim,
                        "projected_dimensions": k,
                        "method": "LDA" if gamma == 0 else "Regularized LDA",
                        "gamma": gamma,
                        "train_accuracy": tr_acc,
                        "test_accuracy": te_acc,
                    })
    write_csv(OUTPUT_DIR / "part5_lda_results.csv", rows)

    # Plot LDA regularization (test accuracy) and PCA dimension-1/2 baseline.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    x = np.array([1, 2])
    for ax, normalize in zip(axes, (False, True)):
        for gamma in LDA_GAMMAS:
            vals = [np.mean(results[(normalize, gamma, k, "test")]) for k in (1, 2)]
            ax.plot(x, vals, marker="o", linewidth=1.6, label=f"LDA gamma={gamma:g}")
        pca_vals = [np.mean(pca_baseline[(normalize, k, "test")]) for k in (1, 2)]
        ax.plot(x, pca_vals, marker="s", linestyle="--", linewidth=2.2, label="PCA only")
        ax.set_xticks([1, 2])
        ax.set_xlabel("Projected dimensions")
        ax.set_ylabel("Test accuracy")
        ax.set_ylim(0, 1.03)
        ax.grid(alpha=0.25)
        ax.set_title("Standardized Iris features" if normalize else "Raw Iris features")
        if not normalize:
            ax.legend(fontsize=8, ncol=2)
    fig.suptitle("PCA vs LDA / regularized LDA (Mahalanobis classifier, 50/50 split)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "part5_lda_vs_pca.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Train/test curves for every gamma, separated by normalization and metric.
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    for col, normalize in enumerate((False, True)):
        for row, metric in enumerate(("train", "test")):
            ax = axes[row, col]
            for gamma in LDA_GAMMAS:
                vals = [np.mean(results[(normalize, gamma, k, metric)]) for k in (1, 2)]
                ax.plot(x, vals, marker="o", linewidth=1.5, label=f"gamma={gamma:g}")
            ax.set_xticks([1, 2])
            ax.set_xlabel("LDA projected dimensions")
            ax.set_ylabel(f"{metric.capitalize()} accuracy")
            ax.set_ylim(0, 1.03)
            ax.grid(alpha=0.25)
            ax.set_title(f"{'Standardized' if normalize else 'Raw'} features — {metric}")
            if row == 0 and col == 0:
                ax.legend(fontsize=8, ncol=2)
    fig.suptitle("LDA and ridge-regularized LDA accuracy")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "part5_lda_train_test_accuracy.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return results, pca_baseline


# ============================================================
# Noise variance effect (100 added noise features)
# ============================================================

def experiment_noise_variance(repeats=5, noise_dim=100, variances=None):
    if variances is None:
        variances = NOISE_VARIANCES
    rows = []
    summary = {}
    for normalize in (False, True):
        for train_ratio in (0.5, 0.2):
            for variance in variances:
                train_scores, test_scores = [], []
                for rep in range(repeats):
                    seed = RANDOM_SEED + rep
                    X_full, y = load_iris_with_noise(noise_dim, variance, seed)
                    Xtr, Xte, ytr, yte = split_and_preprocess(
                        X_full, y, train_ratio=train_ratio, seed=seed, normalize_iris=normalize
                    )
                    tr_acc, te_acc = accuracy_pair(Xtr, ytr, Xte, yte)
                    train_scores.append(tr_acc)
                    test_scores.append(te_acc)
                    rows.append({
                        "experiment": "noise_variance",
                        "repeat": rep,
                        "normalize_iris_features": normalize,
                        "train_ratio": train_ratio,
                        "noise_variance": variance,
                        "noise_dimensions": noise_dim,
                        "projected_dimensions": "",
                        "method": "Mahalanobis",
                        "gamma": "",
                        "train_accuracy": tr_acc,
                        "test_accuracy": te_acc,
                    })
                summary[(normalize, train_ratio, variance)] = {
                    "train_mean": float(np.mean(train_scores)),
                    "test_mean": float(np.mean(test_scores)),
                    "train_std": float(np.std(train_scores, ddof=1)) if repeats > 1 else 0.0,
                    "test_std": float(np.std(test_scores, ddof=1)) if repeats > 1 else 0.0,
                }
                print(
                    f"[Noise variance] norm={normalize}, train={train_ratio:.0%}, "
                    f"var={variance:g}, test acc={summary[(normalize, train_ratio, variance)]['test_mean']:.3f}"
                )
    write_csv(OUTPUT_DIR / "part5_noise_variance_results.csv", rows)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    for row, normalize in enumerate((False, True)):
        for col, train_ratio in enumerate((0.5, 0.2)):
            ax = axes[row, col]
            means = [summary[(normalize, train_ratio, v)]["test_mean"] for v in variances]
            stds = [summary[(normalize, train_ratio, v)]["test_std"] for v in variances]
            ax.errorbar(variances, means, yerr=stds, marker="o", capsize=3, linewidth=1.8)
            ax.set_title(
                f"{'Standardized Iris features' if normalize else 'Raw Iris features'}\n"
                f"Train/test = {train_ratio:.0%}/{1-train_ratio:.0%}"
            )
            ax.set_xlabel("Variance of each noise feature")
            ax.set_ylabel("Test accuracy")
            ax.set_ylim(0, 1.03)
            ax.grid(alpha=0.25)
    fig.suptitle("Noise variance vs Mahalanobis test accuracy (100 noise dimensions)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "part5_noise_variance_accuracy.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return summary


def main(repeats=DEFAULT_REPEATS, max_noise_dim=100, quick=False):
    if quick:
        repeats = min(repeats, 2)
        max_noise_dim = min(max_noise_dim, 20)
        pca_noise_dim = 100
        var_levels = [0.0, 0.1, 0.5, 2.0]
    else:
        pca_noise_dim = 100
        var_levels = NOISE_VARIANCES

    print("=" * 72)
    print("Part 5: Iris + 100 Gaussian noise dimensions; Mahalanobis classifier only")
    print(f"Repeats={repeats}; noise dimension sweep=1..{max_noise_dim}")
    print(f"Default noise variance={DEFAULT_NOISE_VARIANCE}; ridge alpha={MAHALANOBIS_RIDGE}")
    print("=" * 72)

    experiment_noise_dimensions(
        repeats=repeats,
        max_noise_dim=max_noise_dim,
        noise_variance=DEFAULT_NOISE_VARIANCE,
    )
    experiment_pca(
        repeats=repeats,
        noise_dim=pca_noise_dim,
        noise_variance=DEFAULT_NOISE_VARIANCE,
        train_ratio=0.8,
    )
    experiment_lda(
        repeats=repeats,
        noise_dim=100,
        noise_variance=DEFAULT_NOISE_VARIANCE,
        train_ratio=0.5,
    )
    experiment_noise_variance(
        repeats=repeats,
        noise_dim=100,
        variances=var_levels,
    )
    print("\nFinished. Output files were saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--max-noise-dim", type=int, default=100)
    parser.add_argument("--quick", action="store_true", help="run a smaller smoke test")
    args = parser.parse_args()
    main(repeats=args.repeats, max_noise_dim=args.max_noise_dim, quick=args.quick)
