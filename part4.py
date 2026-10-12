"""Part 4: Supervised dimensionality reduction with LDA and regularized LDA.

Compares LDA with PCA (Part 3) on the three-class synthetic dataset with
3 signal dimensions and 100 class-independent noise dimensions.

Default experiment: stratified 50/50 train/test split, repeated several times.
LDA has at most C - 1 = 2 useful discriminant dimensions for C classes.
PCA is fitted on training data only to avoid test-set leakage.
"""

from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# ============================================================
# 1. Dataset specification (same as part1.py / part2.py / part3.py)
# ============================================================

C = 3
N_PER_CLASS = 100
DSIGNAL = 3
NOISE_VARIANCE = 0.5
OUTPUT_DIR = Path(__file__).resolve().parent

means = np.array([
    [2.0, 2.0, 0.0],
    [-2.0, 2.0, 0.0],
    [0.0, -2.0, 0.0],
])

Sigma = np.array([
    [1.0, 0.6, 0.0],
    [0.6, 1.5, 0.0],
    [0.0, 0.0, 0.5],
])


def generate_synthetic_data(dnoise=100, seed=42):
    """Generate 3-class data with three signal dimensions and dnoise noise dimensions."""
    rng = np.random.default_rng(seed)
    X_signal_list = []
    y_list = []

    for c in range(C):
        Xc = rng.multivariate_normal(means[c], Sigma, size=N_PER_CLASS)
        X_signal_list.append(Xc)
        y_list.append(np.full(N_PER_CLASS, c, dtype=int))

    X_signal = np.vstack(X_signal_list)
    y = np.concatenate(y_list)
    X_noise = rng.normal(
        loc=0.0,
        scale=np.sqrt(NOISE_VARIANCE),
        size=(len(y), dnoise),
    )

    X = np.hstack([X_signal, X_noise])
    permutation = rng.permutation(len(y))
    return X[permutation], y[permutation]


# ============================================================
# 2. Shared classifier helpers
# ============================================================


def one_nn_accuracy_curves(Z_train, y_train, Z_test, y_test):
    """1-NN accuracy at every leading dimension; training uses leave-one-out."""
    n_train, n_dims = Z_train.shape
    n_test = Z_test.shape[0]

    train_dist2 = np.zeros((n_train, n_train), dtype=np.float64)
    test_dist2 = np.zeros((n_test, n_train), dtype=np.float64)
    train_curve = np.zeros(n_dims)
    test_curve = np.zeros(n_dims)

    for j in range(n_dims):
        train_delta = Z_train[:, j, None] - Z_train[None, :, j]
        test_delta = Z_test[:, j, None] - Z_train[None, :, j]
        train_dist2 += train_delta ** 2
        test_dist2 += test_delta ** 2

        train_dist_loo = train_dist2.copy()
        np.fill_diagonal(train_dist_loo, np.inf)
        train_nearest = np.argmin(train_dist_loo, axis=1)
        test_nearest = np.argmin(test_dist2, axis=1)

        train_curve[j] = accuracy_score(y_train, y_train[train_nearest])
        test_curve[j] = accuracy_score(y_test, y_train[test_nearest])

    return train_curve, test_curve


class DiagonalMahalanobisClassifier:
    """Diagonal covariance Mahalanobis classifier, consistent with Part 3."""

    def __init__(self, ridge_alpha=0.01):
        self.ridge_alpha = ridge_alpha

    def fit(self, Z, y):
        self.classes_ = np.unique(y)
        self.class_means_ = np.vstack([Z[y == c].mean(axis=0) for c in self.classes_])
        variances = np.var(Z, axis=0, ddof=1)
        self.precision_weights_ = 1.0 / np.maximum(variances + self.ridge_alpha, 1e-12)
        return self

    def predict(self, Z):
        differences = Z[:, None, :] - self.class_means_[None, :, :]
        distances_squared = np.sum(
            differences ** 2 * self.precision_weights_[None, None, :], axis=2
        )
        return self.classes_[np.argmin(distances_squared, axis=1)]


def mahalanobis_accuracy_curves(Z_train, y_train, Z_test, y_test):
    """Mahalanobis train/test accuracy for each leading dimension."""
    n_dims = Z_train.shape[1]
    train_curve = np.zeros(n_dims)
    test_curve = np.zeros(n_dims)

    for k in range(1, n_dims + 1):
        clf = DiagonalMahalanobisClassifier().fit(Z_train[:, :k], y_train)
        train_curve[k - 1] = accuracy_score(y_train, clf.predict(Z_train[:, :k]))
        test_curve[k - 1] = accuracy_score(y_test, clf.predict(Z_test[:, :k]))

    return train_curve, test_curve


def evaluate_low_dimensional_projection(Z_train, y_train, Z_test, y_test):
    """Evaluate 1-NN (LOO for train) and Mahalanobis for projected dims 1 and 2."""
    out = {}
    for k in range(1, Z_train.shape[1] + 1):
        ztr = Z_train[:, :k]
        zte = Z_test[:, :k]

        # 1-NN: training accuracy excludes each sample's self-distance.
        delta = ztr[:, None, :] - ztr[None, :, :]
        dtrain = np.sum(delta * delta, axis=2)
        np.fill_diagonal(dtrain, np.inf)
        train_nn_idx = np.argmin(dtrain, axis=1)
        test_delta = zte[:, None, :] - ztr[None, :, :]
        dtest = np.sum(test_delta * test_delta, axis=2)
        test_nn_idx = np.argmin(dtest, axis=1)
        out[("1NN_train", k)] = accuracy_score(y_train, y_train[train_nn_idx])
        out[("1NN_test", k)] = accuracy_score(y_test, y_train[test_nn_idx])

        # Same diagonal Mahalanobis classifier is used for PCA and LDA spaces.
        clf = DiagonalMahalanobisClassifier().fit(ztr, y_train)
        out[("Mahalanobis_train", k)] = accuracy_score(y_train, clf.predict(ztr))
        out[("Mahalanobis_test", k)] = accuracy_score(y_test, clf.predict(zte))

    return out


# ============================================================
# 3. LDA and ridge-regularized LDA
# ============================================================


def fit_lda_projection(X_train, y_train, gamma=0.0):
    """Fit LDA directions using S_B w = lambda (S_W + ridge I) w.

    gamma=0 is ordinary LDA. For gamma>0, the within-class covariance matrix
    is regularized by gamma * mean(within-class variance) * I. The scaling makes
    gamma more interpretable across datasets/features of different scale.

    Returns a projection matrix with C-1 (here, at most two) columns ordered by
    descending generalized eigenvalue.
    """
    classes = np.unique(y_train)
    n_samples, n_features = X_train.shape
    global_mean = X_train.mean(axis=0)

    within_scatter = np.zeros((n_features, n_features), dtype=np.float64)
    between_scatter = np.zeros((n_features, n_features), dtype=np.float64)

    for cls in classes:
        Xc = X_train[y_train == cls]
        class_mean = Xc.mean(axis=0)
        residual = Xc - class_mean
        within_scatter += residual.T @ residual

        mean_diff = (class_mean - global_mean).reshape(-1, 1)
        between_scatter += len(Xc) * (mean_diff @ mean_diff.T)

    # Pooled within-class covariance estimate. It is positive definite for the
    # default 50/50 split with 103 features, but ridge also stabilizes it.
    within_cov = within_scatter / max(n_samples - len(classes), 1)
    between_scatter /= n_samples
    mean_variance = max(float(np.trace(within_cov) / n_features), 1e-12)
    regularized_within = within_cov + gamma * mean_variance * np.eye(n_features)

    try:
        eigenvalues, eigenvectors = eigh(
            between_scatter, regularized_within, check_finite=False
        )
    except np.linalg.LinAlgError:
        # Defensive numerical fallback for unusual data splits.
        eps = 1e-8 * mean_variance
        eigenvalues, eigenvectors = eigh(
            between_scatter,
            regularized_within + eps * np.eye(n_features),
            check_finite=False,
        )

    order = np.argsort(eigenvalues)[::-1]
    n_components = min(len(classes) - 1, n_features)
    projection = eigenvectors[:, order[:n_components]]
    return projection


# ============================================================
# 4. Experiment configuration and evaluation
# ============================================================

# gamma is relative to the average within-class feature variance.
LDA_GAMMAS = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0]
PCA_CONFIGS = [
    ("PCA only", None),
    ("PCA whitening gamma=0", 0.0),
    ("PCA ridge gamma=0.01", 0.01),
    ("PCA ridge gamma=0.1", 0.1),
    ("PCA ridge gamma=1", 1.0),
    ("PCA ridge gamma=10", 10.0),
]


def scale_pca_scores(scores, eigenvalues, gamma):
    """Apply the same PCA spatial-whitening convention as Part 3."""
    if gamma is None:
        return scores
    scales = 1.0 / np.sqrt(np.maximum(eigenvalues + gamma, 1e-12))
    return scores * scales[None, :]


def evaluate_part4(repeats=5, noise_dim=100, train_ratio=0.5, seed_start=42):
    """Evaluate PCA (Part 3 baseline), ordinary LDA, and regularized LDA."""
    n_features = DSIGNAL + noise_dim
    # 150 training samples allow PCA up to 103 feature dimensions; requested max is 100.
    max_pca_dim = min(100, n_features)
    pca_metrics = ["1NN_train", "1NN_test", "Mahalanobis_train", "Mahalanobis_test"]
    pca_results = {
        label: {metric: np.zeros(max_pca_dim) for metric in pca_metrics}
        for label, _ in PCA_CONFIGS
    }

    lda_metrics = ["1NN_train", "1NN_test", "Mahalanobis_train", "Mahalanobis_test"]
    lda_results = {
        f"LDA gamma={gamma:g}": {
            metric: np.zeros(2) for metric in lda_metrics
        }
        for gamma in LDA_GAMMAS
    }

    first_eigenvalues = None

    for repeat in range(repeats):
        seed = seed_start + repeat
        X, y = generate_synthetic_data(dnoise=noise_dim, seed=seed)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            train_size=train_ratio,
            random_state=seed + 1000,
            stratify=y,
        )

        # ---------- PCA benchmark: fit only using training data ----------
        pca = PCA(n_components=min(X_train.shape[1], X_train.shape[0] - 1), svd_solver="full")
        pca.fit(X_train)
        eigvals = pca.explained_variance_[:max_pca_dim]
        scores_train = pca.transform(X_train)[:, :max_pca_dim]
        scores_test = pca.transform(X_test)[:, :max_pca_dim]

        if repeat == 0:
            first_eigenvalues = pca.explained_variance_.copy()

        for label, gamma in PCA_CONFIGS:
            Z_train = scale_pca_scores(scores_train, eigvals, gamma)
            Z_test = scale_pca_scores(scores_test, eigvals, gamma)

            nn_train, nn_test = one_nn_accuracy_curves(
                Z_train, y_train, Z_test, y_test
            )
            maha_train, maha_test = mahalanobis_accuracy_curves(
                Z_train, y_train, Z_test, y_test
            )

            pca_results[label]["1NN_train"] += nn_train / repeats
            pca_results[label]["1NN_test"] += nn_test / repeats
            pca_results[label]["Mahalanobis_train"] += maha_train / repeats
            pca_results[label]["Mahalanobis_test"] += maha_test / repeats

        # ---------- LDA and regularized LDA ----------
        for gamma in LDA_GAMMAS:
            W = fit_lda_projection(X_train, y_train, gamma=gamma)
            Z_train = (X_train - X_train.mean(axis=0)) @ W
            Z_test = (X_test - X_train.mean(axis=0)) @ W
            one_run = evaluate_low_dimensional_projection(
                Z_train, y_train, Z_test, y_test
            )
            label = f"LDA gamma={gamma:g}"
            for metric in lda_metrics:
                for k in (1, 2):
                    lda_results[label][metric][k - 1] += one_run[(metric, k)] / repeats

        print(f"Completed repeat {repeat + 1}/{repeats} ({train_ratio:.0%} training data)")

    return np.arange(1, max_pca_dim + 1), first_eigenvalues, pca_results, lda_results


# ============================================================
# 5. Plots
# ============================================================


def plot_pca_baseline(dimensions, pca_results, train_ratio):
    """PCA baseline curves for comparison with the supervised LDA subspace."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    for label, res in pca_results.items():
        axes[0].plot(dimensions, res["1NN_test"], linewidth=1.6, label=label)
        axes[1].plot(dimensions, res["Mahalanobis_test"], linewidth=1.6, label=label)

    axes[0].set_title("PCA: 1-NN test accuracy")
    axes[1].set_title("PCA: Mahalanobis test accuracy")
    for ax in axes:
        ax.set_xlabel("Number of leading principal components")
        ax.set_ylabel("Testing accuracy")
        ax.set_ylim(0.0, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"PCA Baseline (train/test = {train_ratio:.0%}/{1-train_ratio:.0%})")
    plt.tight_layout()
    path = OUTPUT_DIR / "part4_pca_baseline.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_lda_train_test(lda_results, train_ratio):
    """Train/test accuracies for LDA and regularized LDA at 1 and 2 dimensions."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    panels = [
        (axes[0, 0], "1NN_train", "1-NN training accuracy (LOO)"),
        (axes[0, 1], "1NN_test", "1-NN testing accuracy"),
        (axes[1, 0], "Mahalanobis_train", "Mahalanobis training accuracy"),
        (axes[1, 1], "Mahalanobis_test", "Mahalanobis testing accuracy"),
    ]
    x = np.array([1, 2])

    for ax, metric, title in panels:
        for label, res in lda_results.items():
            ax.plot(x, res[metric], marker="o", linewidth=1.5, label=label)
        ax.set_title(title)
        ax.set_xlabel("LDA projected dimensionality")
        ax.set_ylabel("Accuracy")
        ax.set_xticks([1, 2])
        ax.set_ylim(0.0, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=2)

    fig.suptitle(
        f"LDA vs Ridge-Regularized LDA (train/test = {train_ratio:.0%}/{1-train_ratio:.0%})"
    )
    plt.tight_layout()
    path = OUTPUT_DIR / "part4_lda_train_test_accuracy.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_pca_vs_lda(dimensions, pca_results, lda_results, train_ratio):
    """Compare PCA curves (1..100 dims) and LDA results (1 or 2 dims) in separate rows."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), sharey="row")

    # Top row: PCA accuracy curves over the full projected-dimensionality range.
    for label, res in pca_results.items():
        axes[0, 0].plot(dimensions, res["1NN_test"], linewidth=1.5, label=label)
        axes[0, 1].plot(dimensions, res["Mahalanobis_test"], linewidth=1.5, label=label)

    axes[0, 0].set_title("PCA: 1-NN testing accuracy")
    axes[0, 1].set_title("PCA: Mahalanobis testing accuracy")
    for ax in axes[0]:
        ax.set_xlabel("Number of leading principal components")
        ax.set_ylabel("Testing accuracy")
        ax.set_ylim(0.0, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=2)

    # Bottom row: supervised LDA and regularized LDA, whose maximum dimension is 2.
    x = np.array([1, 2])
    for label, res in lda_results.items():
        axes[1, 0].plot(x, res["1NN_test"], marker="o", linewidth=1.5, label=label)
        axes[1, 1].plot(x, res["Mahalanobis_test"], marker="o", linewidth=1.5, label=label)

    axes[1, 0].set_title("LDA / regularized LDA: 1-NN")
    axes[1, 1].set_title("LDA / regularized LDA: Mahalanobis")
    for ax in axes[1]:
        ax.set_xlabel("LDA projected dimensionality (maximum = C - 1 = 2)")
        ax.set_ylabel("Testing accuracy")
        ax.set_xticks([1, 2])
        ax.set_ylim(0.0, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=2)

    fig.suptitle(
        f"PCA vs Supervised LDA (train/test = {train_ratio:.0%}/{1-train_ratio:.0%})"
    )
    plt.tight_layout()
    path = OUTPUT_DIR / "part4_pca_vs_lda.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_lda_gamma_comparison(lda_results, pca_results, train_ratio):
    """Testing accuracy as a function of LDA ridge strength, for 1D and 2D."""
    gamma_labels = [f"{g:g}" for g in LDA_GAMMAS]
    x = np.arange(len(LDA_GAMMAS))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)

    for ax, metric, title in [
        (axes[0], "1NN_test", "1-NN testing accuracy"),
        (axes[1], "Mahalanobis_test", "Mahalanobis testing accuracy"),
    ]:
        for dim_idx, dim in enumerate((1, 2)):
            values = [lda_results[f"LDA gamma={g:g}"][metric][dim_idx] for g in LDA_GAMMAS]
            ax.plot(x, values, marker="o", linewidth=2, label=f"LDA dimension={dim}")

        # Horizontal references are the best PCA test accuracy over dimensions/configs.
        pca_metric = metric
        best_pca = max(
            float(np.max(res[pca_metric])) for res in pca_results.values()
        )
        ax.axhline(best_pca, linestyle=":", linewidth=1.8,
                   label=f"Best PCA test acc. = {best_pca:.3f}")
        ax.set_title(title)
        ax.set_xlabel("LDA ridge gamma (0 = ordinary LDA)")
        ax.set_ylabel("Testing accuracy")
        ax.set_xticks(x, gamma_labels)
        ax.set_ylim(0.0, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle(f"Effect of Ridge Regularization in LDA (train/test = {train_ratio:.0%}/{1-train_ratio:.0%})")
    plt.tight_layout()
    path = OUTPUT_DIR / "part4_lda_regularization_comparison.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def save_summary_csv(pca_results, lda_results):
    """Save averaged results for convenient reporting and plotting elsewhere."""
    path = OUTPUT_DIR / "part4_results_summary.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "metric", "projected_dimension", "accuracy"])
        for label, res in pca_results.items():
            for metric, curve in res.items():
                for k, value in enumerate(curve, start=1):
                    writer.writerow([label, metric, k, f"{value:.6f}"])
        for label, res in lda_results.items():
            for metric, values in res.items():
                for k, value in enumerate(values, start=1):
                    writer.writerow([label, metric, k, f"{value:.6f}"])
    return path


def print_best_results(pca_results, lda_results):
    print("\n=== Best PCA test accuracies across dimensions/configurations ===")
    for metric in ("1NN_test", "Mahalanobis_test"):
        best = (-np.inf, None, None)
        for label, res in pca_results.items():
            idx = int(np.argmax(res[metric]))
            value = float(res[metric][idx])
            if value > best[0]:
                best = (value, label, idx + 1)
        print(f"{metric}: {best[0]:.4f} with {best[1]}, dimension={best[2]}")

    print("\n=== LDA / regularized LDA test accuracies ===")
    for label, res in lda_results.items():
        print(
            f"{label:16s} | "
            f"1NN: dim1={res['1NN_test'][0]:.4f}, dim2={res['1NN_test'][1]:.4f} | "
            f"Mahalanobis: dim1={res['Mahalanobis_test'][0]:.4f}, "
            f"dim2={res['Mahalanobis_test'][1]:.4f}"
        )


# ============================================================
# 6. Main
# ============================================================

if __name__ == "__main__":
    TRAIN_RATIO = 0.5
    REPEATS = 5
    NOISE_DIM = 100

    dimensions, eigenvalues, pca_results, lda_results = evaluate_part4(
        repeats=REPEATS,
        noise_dim=NOISE_DIM,
        train_ratio=TRAIN_RATIO,
        seed_start=42,
    )

    # Eigenvalue spectrum is included for a consistent PCA reference.
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(np.arange(1, len(eigenvalues) + 1), eigenvalues, marker=".", linewidth=1.5)
    ax.set_xlabel("Principal component index")
    ax.set_ylabel("Eigenvalue (explained variance)")
    ax.set_title("PCA Eigenvalue Spectrum (training set only)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "part4_pca_eigenvalue_spectrum.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    output_paths = [
        plot_pca_baseline(dimensions, pca_results, TRAIN_RATIO),
        plot_lda_train_test(lda_results, TRAIN_RATIO),
        plot_pca_vs_lda(dimensions, pca_results, lda_results, TRAIN_RATIO),
        plot_lda_gamma_comparison(lda_results, pca_results, TRAIN_RATIO),
        save_summary_csv(pca_results, lda_results),
    ]

    print_best_results(pca_results, lda_results)
    print("\nSaved outputs:")
    print("  - part4_pca_eigenvalue_spectrum.png")
    for output_path in output_paths:
        print(f"  - {output_path.name}")
    print("\nExperiment completed.")
