import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from sklearn.metrics import accuracy_score


# ============================================================
# 1. Dataset specification (consistent with part1.py / part2.py)
# ============================================================

C = 3
N_PER_CLASS = 100
DSIGNAL = 3
NOISE_VARIANCE = 0.5

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
    """Generate three-class data with 3 signal dimensions and dnoise noise dimensions."""
    rng = np.random.default_rng(seed)

    X_signal_list = []
    y_list = []

    for c in range(C):
        Xc = rng.multivariate_normal(
            mean=means[c],
            cov=Sigma,
            size=N_PER_CLASS,
        )
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
# 2. Classifier 1: 1-Nearest Neighbor
# ============================================================

def one_nn_accuracy_curves(Z_train, y_train, Z_test, y_test):
    """
    Compute 1-NN accuracy for dimensions 1..D.

    Training accuracy uses leave-one-out (LOO): a training sample is not
    allowed to select itself as its nearest neighbor. Squared Euclidean
    distances can be used because squaring preserves distance rankings.
    Distances are accumulated one PC at a time for efficiency.
    """
    n_train, n_dims = Z_train.shape
    n_test = Z_test.shape[0]

    train_dist2 = np.zeros((n_train, n_train), dtype=np.float64)
    test_dist2 = np.zeros((n_test, n_train), dtype=np.float64)

    train_curve = np.zeros(n_dims, dtype=float)
    test_curve = np.zeros(n_dims, dtype=float)

    for j in range(n_dims):
        train_delta = Z_train[:, j, None] - Z_train[None, :, j]
        train_dist2 += train_delta ** 2

        test_delta = Z_test[:, j, None] - Z_train[None, :, j]
        test_dist2 += test_delta ** 2

        # Exclude the sample itself from its training nearest-neighbor search.
        train_dist_loo = train_dist2.copy()
        np.fill_diagonal(train_dist_loo, np.inf)

        train_nearest = np.argmin(train_dist_loo, axis=1)
        test_nearest = np.argmin(test_dist2, axis=1)

        train_curve[j] = accuracy_score(y_train, y_train[train_nearest])
        test_curve[j] = accuracy_score(y_test, y_train[test_nearest])

    return train_curve, test_curve


# ============================================================
# 3. Classifier 2: Mahalanobis distance classifier
# ============================================================

class MahalanobisClassifier:
    """Classify by distance to class means using a regularized shared precision matrix."""

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.class_means_ = np.vstack([
            X[y == c].mean(axis=0) for c in self.classes_
        ])

        # This follows the Part 2 implementation: estimate a shared covariance
        # from the training feature vectors, using Ledoit-Wolf shrinkage for stability.
        self.precision_ = LedoitWolf().fit(X).precision_
        return self

    def predict(self, X):
        # differences shape: (number of samples, number of classes, dimensions)
        differences = X[:, None, :] - self.class_means_[None, :, :]
        distances_squared = np.einsum(
            "nci,ij,ncj->nc",
            differences,
            self.precision_,
            differences,
        )
        return self.classes_[np.argmin(distances_squared, axis=1)]


def mahalanobis_accuracy_curves(Z_train, y_train, Z_test, y_test):
    """Fit a Mahalanobis classifier for each leading-PC dimensionality 1..D."""
    n_dims = Z_train.shape[1]
    train_curve = np.zeros(n_dims, dtype=float)
    test_curve = np.zeros(n_dims, dtype=float)

    for k in range(1, n_dims + 1):
        classifier = MahalanobisClassifier().fit(Z_train[:, :k], y_train)
        train_pred = classifier.predict(Z_train[:, :k])
        test_pred = classifier.predict(Z_test[:, :k])

        train_curve[k - 1] = accuracy_score(y_train, train_pred)
        test_curve[k - 1] = accuracy_score(y_test, test_pred)

    return train_curve, test_curve


# ============================================================
# 4. PCA projection and spatial whitening
# ============================================================

def whiten_pca_scores(scores, eigenvalues, gamma):
    """
    Transform PCA scores with spatial whitening.

    gamma=None : PCA projection only (no whitening)
    gamma=0    : standard PCA whitening, score_i / sqrt(lambda_i)
    gamma>0    : ridge-regularized whitening, score_i / sqrt(lambda_i + gamma)
    """
    if gamma is None:
        weights = np.ones_like(eigenvalues)
    else:
        # Clip only to prevent numerical problems if an eigenvalue is near zero.
        weights = 1.0 / np.sqrt(np.maximum(eigenvalues + gamma, 1e-12))

    return scores * weights[None, :]


# ============================================================
# 5. Run PCA / ridge-whitening experiments
# ============================================================

def evaluate_pca_experiment(
    train_ratio=0.5,
    repeats=5,
    max_projected_dim=100,
    noise_dim=100,
    seed_start=42,
):
    """
    Repeatedly split the synthetic dataset, fit PCA on training data only,
    and evaluate both classifiers for each leading-PC dimensionality.

    The available PCA dimension is limited by feature count and training rank.
    With a 50/50 split, 100 projected dimensions are available. With a 20/80
    split, the centered training data has rank at most 59, so only dimensions
    1..59 are available without fitting PCA on test data (which would leak information).
    """
    configs = [("PCA only (no whitening)", None)] + [
        (f"gamma={gamma:g}", gamma)
        for gamma in [0.0, 0.01, 0.1, 0.5, 1.0, 2.0]
    ]

    # Determine the maximum dimension from the first split.
    X0, y0 = generate_synthetic_data(dnoise=noise_dim, seed=seed_start)
    Xtr0, Xte0, ytr0, yte0 = train_test_split(
        X0,
        y0,
        train_size=train_ratio,
        random_state=seed_start + 1000,
        stratify=y0,
    )
    max_dim = min(max_projected_dim, Xtr0.shape[1], Xtr0.shape[0] - 1)
    if max_dim < max_projected_dim:
        print(
            f"Note: requested {max_projected_dim} PCs, but this split supports "
            f"only {max_dim} centered PCA dimensions."
        )

    metric_names = ["1NN_train", "1NN_test", "Mahalanobis_train", "Mahalanobis_test"]
    results = {
        label: {metric: np.zeros(max_dim, dtype=float) for metric in metric_names}
        for label, _ in configs
    }

    eigenvalues_for_plot = None
    seed = seed_start
    X, y = generate_synthetic_data(dnoise=noise_dim, seed=seed)
    
    for repeat in range(repeats):
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            train_size=train_ratio,
            random_state=seed + 1000,
            stratify=y,
        )

        # Fit PCA ONLY on the training subset to avoid test-set information leakage.
        n_pca_components = min(X_train.shape[1], X_train.shape[0])
        pca = PCA(n_components=n_pca_components, svd_solver="full")
        pca.fit(X_train)

        if repeat == 0:
            eigenvalues_for_plot = pca.explained_variance_.copy()

        scores_train = pca.transform(X_train)[:, :max_dim]
        scores_test = pca.transform(X_test)[:, :max_dim]
        eigenvalues = pca.explained_variance_[:max_dim]

        for label, gamma in configs:
            Z_train = whiten_pca_scores(scores_train, eigenvalues, gamma)
            Z_test = whiten_pca_scores(scores_test, eigenvalues, gamma)

            nn_train, nn_test = one_nn_accuracy_curves(
                Z_train, y_train, Z_test, y_test
            )
            maha_train, maha_test = mahalanobis_accuracy_curves(
                Z_train, y_train, Z_test, y_test
            )

            results[label]["1NN_train"] += nn_train / repeats
            results[label]["1NN_test"] += nn_test / repeats
            results[label]["Mahalanobis_train"] += maha_train / repeats
            results[label]["Mahalanobis_test"] += maha_test / repeats

        print(f"Completed repeat {repeat + 1}/{repeats} (train ratio={train_ratio:.0%})")

    return np.arange(1, max_dim + 1), eigenvalues_for_plot, results


# ============================================================
# 6. Plot eigenvalue spectrum
# ============================================================

def plot_eigenvalue_spectrum(eigenvalues, train_ratio):
    dimensions = np.arange(1, len(eigenvalues) + 1)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(dimensions, eigenvalues, marker=".", markersize=4, linewidth=1.5)
    ax.set_xlabel("Principal component index")
    ax.set_ylabel("Eigenvalue (explained variance)")
    ax.set_title(
        f"PCA Eigenvalue Spectrum (descending) — train ratio {train_ratio:.0%}"
    )
    ax.grid(alpha=0.3)
    ax.set_xlim(1, len(eigenvalues))
    plt.tight_layout()
    plt.savefig("part3_eigenvalue_spectrum.png", dpi=300, bbox_inches="tight")
    plt.show()


# ============================================================
# 7. Plot train/test curves for each gamma
# ============================================================

def plot_accuracy_curves(noise_dims, results, train_ratio, repeats):
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), sharex=True, sharey=True)

    panels = [
        (axes[0, 0], "1NN_train", "1-NN: training accuracy (LOO)"),
        (axes[0, 1], "1NN_test", "1-NN: testing accuracy"),
        (axes[1, 0], "Mahalanobis_train", "Mahalanobis: training accuracy"),
        (axes[1, 1], "Mahalanobis_test", "Mahalanobis: testing accuracy"),
    ]

    for ax, metric, title in panels:
        for label, metric_results in results.items():
            ax.plot(
                noise_dims,
                metric_results[metric],
                linewidth=1.8,
                label=label,
            )
        ax.set_title(title)
        ax.set_xlabel("Projected dimensionality (number of leading PCs)")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.0, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=2)

    fig.suptitle(
        f"PCA Spatial Whitening with Ridge Regularization "
        f"(train/test = {train_ratio:.0%}/{1 - train_ratio:.0%}, "
        f"{repeats} repetitions)",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig("part3_accuracy_by_gamma.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_testing_comparison(noise_dims, results, train_ratio):
    """Focused view of testing accuracy curves for easier gamma comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)

    for label, metric_results in results.items():
        axes[0].plot(
            noise_dims, metric_results["1NN_test"], linewidth=1.8, label=label
        )
        axes[1].plot(
            noise_dims,
            metric_results["Mahalanobis_test"],
            linewidth=1.8,
            label=label,
        )

    axes[0].set_title("1-NN testing accuracy")
    axes[1].set_title("Mahalanobis testing accuracy")
    for ax in axes:
        ax.set_xlabel("Projected dimensionality (number of leading PCs)")
        ax.set_ylabel("Testing accuracy")
        ax.set_ylim(0.0, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=2)

    fig.suptitle(
        f"Testing Accuracy by Ridge Parameter "
        f"(train/test = {train_ratio:.0%}/{1 - train_ratio:.0%})",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig("part3_testing_accuracy_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()


# ============================================================
# 8. Main
# ============================================================

if __name__ == "__main__":
    # Main assignment setting: dnoise=100 and a 50/50 training/testing split.
    # A 50/50 split leaves enough training samples to examine 100 PCs.
    TRAIN_RATIO = 0.5
    REPEATS = 5
    NOISE_DIM = 100
    MAX_PROJECTED_DIM = 100

    dimensions, eigenvalues, results = evaluate_pca_experiment(
        train_ratio=TRAIN_RATIO,
        repeats=REPEATS,
        max_projected_dim=MAX_PROJECTED_DIM,
        noise_dim=NOISE_DIM,
        seed_start=42,
    )

    plot_eigenvalue_spectrum(eigenvalues, train_ratio=TRAIN_RATIO)
    plot_accuracy_curves(
        dimensions,
        results,
        train_ratio=TRAIN_RATIO,
        repeats=REPEATS,
    )
    plot_testing_comparison(
        dimensions,
        results,
        train_ratio=TRAIN_RATIO,
    )

    print("\nExperiment completed.")
    print("Saved plots:")
    print("  - part3_eigenvalue_spectrum.png")
    print("  - part3_accuracy_by_gamma.png")
    print("  - part3_testing_accuracy_comparison.png")
