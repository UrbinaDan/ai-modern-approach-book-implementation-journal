# KNN_numpy_digits.py
# NumPy-only kNN for the 32x32 ASCII digit dataset (no scikit-learn).
# - Robust filename parsing: "<label>_*.txt" -> int label 0..9
# - Vectorized distance (Euclidean or Manhattan) per test sample
# - Fast top-k with np.argpartition
# - Optional inverse-distance weighting
# - Saves confusion matrix and class-mean "weight" heatmaps

from __future__ import annotations
import os, argparse
import numpy as np
import matplotlib.pyplot as plt


# ----------------------- small utilities -----------------------

def resolve(path: str) -> str:
    """Make 'path' absolute relative to the current working directory."""
    return path if os.path.isabs(path) else os.path.join(os.getcwd(), path)

def parse_label_from_filename(fname: str) -> int:
    """Assumes filenames like '7_45.txt' or '3_123.txt' → label 7 or 3."""
    base = os.path.basename(fname)
    first = base.split('.')[0]      # '7_45'
    lbl = first.split('_')[0]       # '7'
    return int(lbl)

def img2vector(path: str) -> np.ndarray:
    """Read a 32x32 ASCII bitmap file -> 1x1024 float vector (0/1)."""
    arr = np.zeros((1, 1024), dtype=float)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i in range(32):
            line = f.readline()
            # accept '0'/'1' (or any digit chars) with possible line breaks
            for j in range(32):
                arr[0, 32 * i + j] = 1.0 if j < len(line) and line[j] != '0' and line[j] != ' ' else 0.0
    return arr

def load_digit_dir(root: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load all digit files in directory 'root'.
    Returns:
        X: (n_samples, 1024) float array of 0/1
        y: (n_samples,) integer labels 0..9
    """
    root = resolve(root)
    files = [os.path.join(root, f) for f in os.listdir(root) if f.endswith(".txt")]
    files.sort()
    X_list, y_list = [], []
    for fp in files:
        try:
            y_list.append(parse_label_from_filename(fp))
            X_list.append(img2vector(fp))
        except Exception:
            # skip malformed file
            pass
    if not X_list:
        raise FileNotFoundError(f"No .txt digit files found in: {root}")
    X = np.vstack(X_list)  # (n, 1024)
    y = np.asarray(y_list, dtype=int)
    return X, y

# ----------------------- KNN core (NumPy-only) -----------------------

def _distances(x: np.ndarray, X: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """
    Vectorized distances from single test vector x (1, d) to all rows of X (n, d).
    Returns shape (n,).
    """
    if metric == "euclidean":
        # ||x - Xi||_2
        d = np.linalg.norm(X - x, axis=1)
    elif metric == "manhattan":
        # L1 distance
        d = np.abs(X - x).sum(axis=1)
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    return d

def knn_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    k: int = 3,
    weighted: bool = True,
    metric: str = "euclidean",
    eps: float = 1e-12,
) -> np.ndarray:
    """
    NumPy-only kNN prediction.
    - For each test x, compute distances to all training points.
    - Take k smallest via argpartition.
    - Weighted vote (1 / (dist+eps)) if weighted=True, else majority count.
    Returns y_pred of shape (n_test,).
    """
    n_test = X_test.shape[0]
    y_pred = np.empty(n_test, dtype=int)

    for i in range(n_test):
        d = _distances(X_test[i : i + 1], X_train, metric=metric)

        # top-k indices without full sort: O(n)
        kth = k if k < len(d) else len(d)
        idx_k = np.argpartition(d, kth - 1)[:k]  # unsorted k smallest

        # vote
        if weighted:
            # avoid div by zero for exact duplicates
            w = 1.0 / (d[idx_k] + eps)
            labels = y_train[idx_k]
            # sum weights per class
            # classes are 0..9 → we can accumulate into length-10 buffer
            totals = np.zeros(10, dtype=float)
            np.add.at(totals, labels, w)
            y_pred[i] = int(totals.argmax())
        else:
            labels = y_train[idx_k]
            # simple majority (ties broken by smallest label)
            counts = np.bincount(labels, minlength=10)
            y_pred[i] = int(counts.argmax())

    return y_pred

# ----------------------- evaluation & plots -----------------------

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())

def per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 10) -> list[float]:
    accs = []
    for c in range(n_classes):
        mask = (y_true == c)
        if mask.any():
            accs.append(float((y_pred[mask] == c).mean()))
        else:
            accs.append(np.nan)
    return accs

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 10) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def save_confusion(cm: np.ndarray, path: str) -> None:
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    # annotate counts
    nC = cm.shape[0]
    for i in range(nC):
        for j in range(nC):
            val = cm[i, j]
            if val:
                plt.text(j, i, str(val), ha="center", va="center", fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(resolve(path)), exist_ok=True)
    plt.savefig(resolve(path), dpi=160)
    plt.close()

def save_class_means_grid(X_train: np.ndarray, y_train: np.ndarray, path: str) -> None:
    """
    Compute mean image per class (0..9) from training data and save a 2x5 grid.
    """
    means = []
    for c in range(10):
        rows = X_train[y_train == c]
        if len(rows) == 0:
            means.append(np.zeros((32, 32)))
        else:
            means.append(rows.mean(axis=0).reshape(32, 32))

    fig, axes = plt.subplots(2, 5, figsize=(8, 3.5))
    for c, ax in enumerate(axes.flat):
        ax.imshow(means[c], cmap="viridis", interpolation="nearest")
        ax.set_title(str(c))
        ax.axis("off")
    fig.suptitle("Class mean 'weights' (average bitmaps)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(resolve(path)), exist_ok=True)
    fig.savefig(resolve(path), dpi=160)
    plt.close(fig)

# ----------------------- CLI -----------------------

def main():
    ap = argparse.ArgumentParser(description="NumPy-only kNN for ASCII 32x32 digit dataset.")
    ap.add_argument("--train-dir", required=True, help="path to trainingDigits folder")
    ap.add_argument("--test-dir",  required=True, help="path to testDigits folder")
    ap.add_argument("--k", type=int, default=3, help="neighbors (k)")
    ap.add_argument("--metric", choices=["euclidean","manhattan"], default="euclidean")
    ap.add_argument("--weighted", action="store_true", help="inverse-distance weighting vote")
    ap.add_argument("--save-confusion", default="", help="filename to save confusion matrix PNG")
    ap.add_argument("--save-means", default="", help="filename to save class-mean heatmap grid PNG")
    args = ap.parse_args()

    # Load data
    Xtr, ytr = load_digit_dir(args.train_dir)
    Xte, yte = load_digit_dir(args.test_dir)

    # Predict
    yhat = knn_predict(
        Xtr, ytr, Xte,
        k=args.k,
        weighted=args.weighted,
        metric=args.metric
    )

    # Report
    acc = accuracy(yte, yhat)
    print(f"Accuracy: {acc:.4f}  (k={args.k}, weighted={args.weighted}, metric={args.metric})")
    pcs = per_class_accuracy(yte, yhat, n_classes=10)
    print("Per-class accuracy:")
    for c, a in enumerate(pcs):
        if np.isnan(a):
            print(f"  {c}: n/a (no samples)")
        else:
            print(f"  {c}: {a:.3f}")

    # Plots
    if args.save_confusion:
        cm = confusion_matrix(yte, yhat, n_classes=10)
        save_confusion(cm, args.save_confusion)
        print(f"[saved] {args.save_confusion}")

    if args.save_means:
        save_class_means_grid(Xtr, ytr, args.save_means)
        print(f"[saved] {args.save_means}")


if __name__ == "__main__":
    main()

#Run 
# python "19.7 NonParametric Algorithms/KNN/KNN Ex 2/Digit Recognition KNN/KNN_numpy_digits.py" \
#   --train-dir "19.7 NonParametric Algorithms/KNN/KNN Ex 2/Digit Recognition KNN/trainingDigits" \
#   --test-dir  "19.7 NonParametric Algorithms/KNN/KNN Ex 2/Digit Recognition KNN/testDigits" \
#   --k 3 --weighted --metric euclidean \
#   --save-confusion "19.7 NonParametric Algorithms/KNN/KNN Ex 2/Digit Recognition KNN/knn_confusion.png" \
#   --save-means    "19.7 NonParametric Algorithms/KNN/KNN Ex 2/Digit Recognition KNN/weights_heatmap.png"
