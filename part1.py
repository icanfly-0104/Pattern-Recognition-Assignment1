import numpy as np
import matplotlib.pyplot as plt

from itertools import combinations
from scipy.stats import chi2
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.model_selection import train_test_split

# =========================
# 1. Dataset specification
# =========================

C = 3
N_PER_CLASS = 100
DSIGNAL = 3
NOISE_VARIANCE = 0.5

means = np.array([
    [ 2.0,  2.0, 0.0],  # Class 1
    [-2.0,  2.0, 0.0],  # Class 2
    [ 0.0, -2.0, 0.0],  # Class 3
])

Sigma = np.array([
    [1.0, 0.6, 0.0],
    [0.6, 1.5, 0.0],
    [0.0, 0.0, 0.5],
])

def generate_synthetic_data(dnoise=10, seed=42):
    """
    Generate 3-class synthetic data.

    Returns
    -------
    X : (300, 3 + dnoise) array
    y : (300,) array, labels 0, 1, 2
    """
    rng = np.random.default_rng(seed)

    X_signal_list = []
    y_list = []

    for c in range(C):
        # Generate signal dimensions from a multivariate Gaussian
        Xc = rng.multivariate_normal(
            mean=means[c],
            cov=Sigma,
            size=N_PER_CLASS
        )

        X_signal_list.append(Xc)
        y_list.append(
            np.full(N_PER_CLASS, c, dtype=int)
        )

    X_signal = np.vstack(X_signal_list)
    y = np.concatenate(y_list)

    # Generate independent, class-uninformative noise
    X_noise = rng.normal(
        loc=0.0,
        scale=np.sqrt(NOISE_VARIANCE),
        size=(len(y), dnoise)
    )

    # Concatenate signal and noise features
    X = np.hstack([X_signal, X_noise])

    # Shuffle the samples
    perm = rng.permutation(len(y))
    X = X[perm]
    y = y[perm]

    return X, y

def plot_2d_projections(X, y):
    # Only use the first 3 signal dimensions
    Xs = X[:, :3]

    # All 2-D combinations of the 3 signal dimensions
    projections = list(combinations(range(3), 2))

    class_names = ["Class 1", "Class 2", "Class 3"]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    # Mahalanobis distance levels
    distance_levels = [1.0, 2.0, 3.0]

    fig, axes = plt.subplots(
        1, 3, figsize=(18, 5)
    )

    for ax, dims in zip(axes, projections):
        i, j = dims

        # Plot samples and theoretical contours
        for c in range(C):
            Xc = Xs[y == c][:, [i, j]]

            ax.scatter(
                Xc[:, 0], Xc[:, 1],
                s=18, alpha=0.55,
                color=colors[c],
                label=class_names[c]
            )

            # Projected mean and covariance
            mu_2d = means[c, [i, j]]
            cov_2d = Sigma[np.ix_([i, j], [i, j])]

            # Grid for Mahalanobis distance
            margin = 4.0
            x_min = min(Xs[:, i].min(), mu_2d[0]) - margin
            x_max = max(Xs[:, i].max(), mu_2d[0]) + margin
            y_min = min(Xs[:, j].min(), mu_2d[1]) - margin
            y_max = max(Xs[:, j].max(), mu_2d[1]) + margin

            gx, gy = np.meshgrid(
                np.linspace(x_min, x_max, 220),
                np.linspace(y_min, y_max, 220)
            )

            grid = np.stack(
                [gx - mu_2d[0], gy - mu_2d[1]],
                axis=-1
            )

            inv_cov = np.linalg.inv(cov_2d)

            # Squared Mahalanobis distance
            D2 = np.einsum(
                "...i,ij,...j->...",
                grid, inv_cov, grid
            )

            ax.contour(
                gx, gy, D2,
                levels=np.square(distance_levels),
                colors=[colors[c]],
                linewidths=1.2,
                linestyles="dashed"
            )

            # Mark theoretical mean
            ax.scatter(
                mu_2d[0], mu_2d[1],
                marker="x", s=100,
                color=colors[c],
                linewidths=2
            )

        ax.set_xlabel(f"Signal dimension {i + 1}")
        ax.set_ylabel(f"Signal dimension {j + 1}")
        ax.set_title(f"Projection: x{i + 1} vs x{j + 1}")
        ax.grid(alpha=0.2)
        ax.legend()

    plt.tight_layout()
    plt.show()

def plot_3d_ellipsoids(X, y):
    Xs = X[:, :3]

    class_names = ["Class 1", "Class 2", "Class 3"]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Chi-square threshold for a 95% probability region
    threshold = chi2.ppf(0.95, df=3)

    # Eigen-decomposition of the shared covariance
    eigenvalues, eigenvectors = np.linalg.eigh(Sigma)

    # Sort eigenvalues in descending order
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Unit sphere parameterization
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 30)

    sphere_x = np.outer(np.cos(u), np.sin(v))
    sphere_y = np.outer(np.sin(u), np.sin(v))
    sphere_z = np.outer(np.ones_like(u), np.cos(v))

    sphere = np.stack(
        [sphere_x, sphere_y, sphere_z],
        axis=-1
    )

    # Transform sphere into a covariance ellipsoid
    transform = eigenvectors @ np.diag(
        np.sqrt(eigenvalues * threshold)
    )

    for c in range(C):
        Xc = Xs[y == c]

        ax.scatter(
            Xc[:, 0], Xc[:, 1], Xc[:, 2],
            s=16, alpha=0.45,
            color=colors[c],
            label=class_names[c]
        )

        mu = means[c]

        ellipsoid = sphere @ transform.T + mu

        ax.plot_surface(
            ellipsoid[:, :, 0],
            ellipsoid[:, :, 1],
            ellipsoid[:, :, 2],
            color=colors[c],
            alpha=0.18,
            linewidth=0
        )

        ax.scatter(
            mu[0], mu[1], mu[2],
            color=colors[c],
            marker="X", s=100,
            edgecolors="black"
        )

    ax.set_xlabel("Signal dimension 1")
    ax.set_ylabel("Signal dimension 2")
    ax.set_zlabel("Signal dimension 3")
    ax.set_title("3-D Signal Data with 95% Confidence Ellipsoids")
    ax.legend()

    plt.tight_layout()
    plt.show()

def predict_1nn(X_train, y_train, X_test):

    predictions = []

    for x in X_test:

        # Calculate distance to every training sample
        distances = np.linalg.norm(
            X_train - x,
            axis=1
        )

        # Find nearest training sample
        nearest_index = np.argmin(distances)

        # Use its label
        predictions.append(
            y_train[nearest_index]
        )

    return np.array(predictions)

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

# Run
if __name__ == '__main__':
    X, y = generate_synthetic_data(dnoise=100, seed=42)
    plot_2d_projections(X, y)
    plot_3d_ellipsoids(X, y)
