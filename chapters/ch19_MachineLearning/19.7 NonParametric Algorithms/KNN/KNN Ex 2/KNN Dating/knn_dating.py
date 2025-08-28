# knn_dating.py
# A clean, NumPy-only k-NN for the "dating" dataset with plotting & saving.
# - Loads tab-separated data with 3 features + label (didntLike/smallDoses/largeDoses)
# - Optional feature scaling: none | minmax | standard
# - Distance: euclidean | manhattan, with optional distance-weighted voting
# - Random train/test split
# - Saves pairwise scatter plots and a confusion-matrix heatmap if paths are given

from __future__ import annotations
import argparse, os, io
from typing import Tuple, List, Dict
import numpy as np
import matplotlib.pyplot as plt

# ------------------------- data loading & scaling -------------------------

LABEL_MAP = {
    "didntLike": 1,
    "smallDoses": 2,
    "largeDoses": 3,
}
LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()}

def load_dating_tsv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read a tab-separated file with 3 numeric columns and a string label.
    Returns:
        X: (n, 3) float array
        y: (n,)  int array in {1,2,3}
    """
    # Read text (handling possible UTF-8 BOM)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    if txt.startswith("\ufeff"):
        txt = txt.lstrip("\ufeff")
    # Parse lines
    X_rows: List[List[float]] = []
    y_rows: List[int] = []
    for ln in txt.strip().splitlines():
        parts = ln.strip().split("\t")
        if len(parts) < 4:
            continue
        try:
            x0, x1, x2 = float(parts[0]), float(parts[1]), float(parts[2])
        except ValueError:
            # skip header/nonnumeric rows
            continue
        label_str = parts[-1].strip()
        if label_str not in LABEL_MAP:
            # If file already uses numeric labels 1/2/3, try that.
            try:
                lab = int(label_str)
                if lab in (1,2,3):
                    y_rows.append(lab)
                    X_rows.append([x0, x1, x2])
                    continue
            except Exception:
                pass
            # Otherwise skip
            continue
        y_rows.append(LABEL_MAP[label_str])
        X_rows.append([x0, x1, x2])
    X = np.asarray(X_rows, dtype=float)
    y = np.asarray(y_rows, dtype=int)
    return X, y

def split_train_test(X: np.ndarray, y: np.ndarray, test_size: float, seed: int) -> Tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = max(1, int(round(test_size * n)))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def scale_fit(X: np.ndarray, mode: str):
    """
    Fit scaling stats. mode in {"none","minmax","standard"}.
    Returns a tuple (transform_func, inverse_info) where transform_func(X)->X_scaled.
    """
    mode = (mode or "none").lower()
    if mode == "none":
        return (lambda A: A, None)

    if mode == "minmax":
        mn = X.min(axis=0)
        mx = X.max(axis=0)
        rng = np.where(mx - mn == 0, 1.0, (mx - mn))
        def _tx(A): return (A - mn) / rng
        return _tx, ("minmax", mn, rng)

    if mode == "standard":
        mu = X.mean(axis=0)
        sd = X.std(axis=0, ddof=0)
        sd = np.where(sd == 0, 1.0, sd)
        def _tx(A): return (A - mu) / sd
        return _tx, ("standard", mu, sd)

    raise ValueError(f"Unknown scale mode: {mode}")

# ------------------------------ k-NN core --------------------------------

def pairwise_distance(a: np.ndarray, b: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """
    Compute distances from every row in a to every row in b.
    Returns D with shape (a.shape[0], b.shape[0]).
    """
    metric = metric.lower()
    if metric == "euclidean":
        # (a - b)^2 = a^2 + b^2 - 2ab
        a2 = (a*a).sum(axis=1, keepdims=True)           # (m,1)
        b2 = (b*b).sum(axis=1, keepdims=True).T         # (1,n)
        ab = a @ b.T                                    # (m,n)
        d2 = np.maximum(a2 + b2 - 2.0*ab, 0.0)
        return np.sqrt(d2)
    elif metric == "manhattan":
        # broadcast |a_i - b_j|
        return np.abs(a[:, None, :] - b[None, :, :]).sum(axis=2)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

def knn_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    k: int = 3,
    metric: str = "euclidean",
    weighted: bool = False,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Predict labels for X_test using k-NN.
    If weighted=True, votes are weighted by 1/(distance + eps).
    Returns y_pred (int) with same classes as y_train.
    """
    D = pairwise_distance(X_test, X_train, metric=metric)       # (m_test, m_train)
    # indices of neighbors sorted by distance ascending
    idx_sorted = np.argsort(D, axis=1)[:, :k]                   # (m_test, k)
    neighbor_labels = y_train[idx_sorted]                       # (m_test, k)
    if not weighted:
        # simple majority vote
        # count per class for each row
        classes = np.unique(y_train)
        votes = np.zeros((X_test.shape[0], classes.size), dtype=float)
        for ci, c in enumerate(classes):
            votes[:, ci] = (neighbor_labels == c).sum(axis=1)
        y_pred = classes[np.argmax(votes, axis=1)]
        return y_pred

    # distance-weighted vote
    neighbor_d = np.take_along_axis(D, idx_sorted, axis=1)      # (m_test, k)
    w = 1.0 / (neighbor_d + eps)                                # inverse distance weights
    classes = np.unique(y_train)
    scores = np.zeros((X_test.shape[0], classes.size), dtype=float)
    for ci, c in enumerate(classes):
        mask = (neighbor_labels == c).astype(float)
        scores[:, ci] = (w * mask).sum(axis=1)
    y_pred = classes[np.argmax(scores, axis=1)]
    return y_pred

# ------------------------------ evaluation -------------------------------

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: List[int]) -> np.ndarray:
    L = len(labels)
    cm = np.zeros((L, L), dtype=int)
    label_to_idx = {lab:i for i, lab in enumerate(labels)}
    for t, p in zip(y_true, y_pred):
        cm[label_to_idx[t], label_to_idx[p]] += 1
    return cm

def per_class_accuracy(cm: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        acc = np.diag(cm) / cm.sum(axis=1, keepdims=False)
        acc = np.nan_to_num(acc)
    return acc

# -------------------------------- plots ----------------------------------

def save_pairwise_scatter(X: np.ndarray, y: np.ndarray, path: str):
    """
    3 features → 3 pairwise scatter plots (0-1, 0-2, 1-2).
    Colors by label. Saves PNG to `path`.
    """
    labels = np.unique(y)
    colors = {1: "#1f77b4", 2: "#ff7f0e", 3: "#d62728"}
    names  = {1: "didntLike", 2: "smallDoses", 3: "largeDoses"}
    pairs = [(0,1), (0,2), (1,2)]
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (i,j) in zip(axs, pairs):
        for lab in labels:
            m = (y == lab)
            ax.scatter(X[m, i], X[m, j], s=18, alpha=0.7, label=names[lab], c=colors[lab])
        ax.set_xlabel(f"feature {i}")
        ax.set_ylabel(f"feature {j}")
        ax.grid(True, alpha=0.2)
    # single legend
    handles, llabels = axs[0].get_legend_handles_labels()
    fig.legend(handles, llabels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Pairwise feature scatter (colored by class)")
    fig.tight_layout(rect=[0,0,1,0.92])
    plt.savefig(path, dpi=160)
    plt.close(fig)
    print(f"[saved] {path}")

def save_confusion_heatmap(cm: np.ndarray, labels: List[int], path: str):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels([LABEL_NAMES[l] for l in labels], rotation=30, ha="right")
    ax.set_yticklabels([LABEL_NAMES[l] for l in labels])
    # annotate counts
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9)
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close(fig)
    print(f"[saved] {path}")

# --------------------------------- CLI -----------------------------------

def main():
    ap = argparse.ArgumentParser(description="From-scratch k-NN on the dating dataset with plotting.")
    ap.add_argument("--data", default="datingTestSet.txt", help="path to TSV file (3 features + label)")
    ap.add_argument("--test-size", type=float, default=0.10, help="fraction for test split (0-1)")
    ap.add_argument("--seed", type=int, default=42, help="random seed")
    ap.add_argument("--k", type=int, default=4, help="k neighbors")
    ap.add_argument("--metric", choices=["euclidean","manhattan"], default="euclidean")
    ap.add_argument("--weighted", action="store_true", help="use inverse-distance weighted vote")
    ap.add_argument("--scale", choices=["none","minmax","standard"], default="minmax",
                    help="feature scaling mode")
    ap.add_argument("--save-scatter", default="", help="path to save pairwise scatter PNG")
    ap.add_argument("--save-confusion", default="", help="path to save confusion-matrix PNG")
    args = ap.parse_args()

    # 1) Load
    X, y = load_dating_tsv(args.data)
    if X.size == 0:
        raise SystemExit(f"No data loaded from {args.data}. Check path/format.")

    # 2) Split
    Xtr, Xte, ytr, yte = split_train_test(X, y, test_size=args.test_size, seed=args.seed)

    # 3) Scale (fit on train, apply to both)
    tx, _ = scale_fit(Xtr, args.scale)
    Xtr_s = tx(Xtr)
    Xte_s = tx(Xte)

    # 4) Train+Predict (k-NN is lazy; prediction uses entire training set)
    yhat = knn_predict(Xtr_s, ytr, Xte_s, k=args.k, metric=args.metric, weighted=args.weighted)

    # 5) Metrics
    acc = (yhat == yte).mean()
    print(f"Accuracy: {acc:.4f}  (k={args.k}, weighted={args.weighted}, metric={args.metric}, scale={args.scale})")
    labels_sorted = sorted(np.unique(y))
    cm = confusion_matrix(yte, yhat, labels_sorted)
    pc = per_class_accuracy(cm)
    print("Per-class accuracy:")
    for lab, a in zip(labels_sorted, pc):
        print(f"  {LABEL_NAMES[lab]}: {a:.3f}")

    # 6) Plots (save if path provided)
    if args.save_confusion:
        os.makedirs(os.path.dirname(args.save_confusion) or ".", exist_ok=True)
        save_confusion_heatmap(cm, labels_sorted, args.save_confusion)

    if args.save_scatter:
        os.makedirs(os.path.dirname(args.save_scatter) or ".", exist_ok=True)
        # Use the **scaled** training set so axes are comparable to the fit
        save_pairwise_scatter(Xtr_s, ytr, args.save_scatter)

if __name__ == "__main__":
    main()
#Run this
# python "19.7 NonParametric Algorithms/KNN/KNN Ex 2/KNN Dating/knn_dating.py" \
#   --data "19.7 NonParametric Algorithms/KNN/KNN Ex 2/KNN Dating/datingTestSet.txt" \
#   --k 5 --weighted --metric manhattan --test-size 0.2 \
#   --save-scatter "19.7 NonParametric Algorithms/KNN/KNN Ex 2/KNN Dating/outputs/dating_scatter.png" \
#   --save-confusion "19.7 NonParametric Algorithms/KNN/KNN Ex 2/KNN Dating/outputs/dating_confusion.png"
