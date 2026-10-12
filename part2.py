import numpy as np
import matplotlib.pyplot as plt

# ---------- 1. Generate data ----------
MEANS = np.array([
    [ 2.0,  2.0, 0.0],
    [-2.0,  2.0, 0.0],
    [ 0.0, -2.0, 0.0]
])

SIGMA = np.array([
    [1.0, 0.6, 0.0],
    [0.6, 1.5, 0.0],
    [0.0, 0.0, 0.5]
])

def generate_data(dnoise=100, seed=42):
    rng = np.random.default_rng(seed)
    X_list, y_list = [], []

    for c in range(3):
        Xc = rng.multivariate_normal(
            MEANS[c], SIGMA, size=100
        )
        X_list.append(Xc)
        y_list.append(np.full(100, c))

    X_signal = np.vstack(X_list)
    y = np.concatenate(y_list)

    X_noise = rng.normal(
        0, np.sqrt(0.5), size=(300, dnoise)
    )
    X = np.hstack([X_signal, X_noise])

    order = rng.permutation(len(y))
    return X[order], y[order]

# ---------- 2. Stratified split ----------
def split_data(X, y, train_ratio, seed=123):
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []

    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_train = int(len(idx) * train_ratio)

        train_idx.extend(idx[:n_train])
        test_idx.extend(idx[n_train:])

    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    return (X[train_idx], y[train_idx],
            X[test_idx], y[test_idx])

# ---------- 3. 1-NN classifier ----------
def predict_1nn(X_query, X_train, y_train):
    # Squared Euclidean distance
    dist2 = (
        (X_query ** 2).sum(axis=1, keepdims=True)
        + (X_train ** 2).sum(axis=1)[None, :]
        - 2 * X_query @ X_train.T
    )
    nearest = np.argmin(dist2, axis=1)
    return y_train[nearest]

# ---------- 4. Mahalanobis classifier ----------
def predict_mahalanobis(X_query, X_train, y_train):
    classes = np.unique(y_train)
    distances = []

    for c in classes:
        Xc = X_train[y_train == c]
        mean_c = Xc.mean(axis=0)
        cov_c = np.cov(Xc, rowvar=False)

        # Pseudoinverse for potentially singular covariance
        inv_cov = np.linalg.pinv(cov_c)

        diff = X_query - mean_c
        d2 = np.einsum(
            "bi,ij,bj->b", diff, inv_cov, diff
        )
        distances.append(d2)

    distances = np.column_stack(distances)
    return classes[np.argmin(distances, axis=1)]


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred) * 100


# ---------- 5. Run one experiment ----------
def run_experiment(train_ratio, seed=123):
    # Generate once so dimensions are nested and comparable
    X, y = generate_data(dnoise=100, seed=42)

    Xtr, ytr, Xte, yte = split_data(
        X, y, train_ratio=train_ratio, seed=seed
    )

    results = {
        "1NN_train": [], "1NN_test": [],
        "Mahalanobis_train": [],
        "Mahalanobis_test": []
    }

    for k in range(1, 101):
        # 3 signal dimensions + first k noise dimensions
        Xtr_k = Xtr[:, :3+k]
        Xte_k = Xte[:, :3+k]

        # 1-NN
        pred_tr = predict_1nn(Xtr_k, Xtr_k, ytr)
        pred_te = predict_1nn(Xte_k, Xtr_k, ytr)

        results["1NN_train"].append(accuracy(ytr, pred_tr))
        results["1NN_test"].append(accuracy(yte, pred_te))

        # Mahalanobis
        pred_tr_m = predict_mahalanobis(Xtr_k, Xtr_k, ytr)
        pred_te_m = predict_mahalanobis(Xte_k, Xtr_k, ytr)

        results["Mahalanobis_train"].append(accuracy(ytr, pred_tr_m))
        results["Mahalanobis_test"].append(accuracy(yte, pred_te_m))

    return results


# ---------- 6. Plot ----------
def plot_results(results, train_ratio):
    dims = np.arange(1, 101)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, name in zip(
        axes, ["1NN", "Mahalanobis"]
    ):
        ax.plot(
            dims, results[f"{name}_train"],
            label="Training accuracy"
        )
        ax.plot(
            dims, results[f"{name}_test"],
            label="Testing accuracy"
        )
        ax.set_xlabel("Number of noise dimensions")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"{name}: train ratio={train_ratio}")
        ax.set_ylim(0, 105)
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Experiment A: 50/50
    results_50 = run_experiment(train_ratio=0.5)
    plot_results(results_50, train_ratio=0.5)

    # Experiment B: 20/80
    results_20 = run_experiment(train_ratio=0.2)
    plot_results(results_20, train_ratio=0.2)
