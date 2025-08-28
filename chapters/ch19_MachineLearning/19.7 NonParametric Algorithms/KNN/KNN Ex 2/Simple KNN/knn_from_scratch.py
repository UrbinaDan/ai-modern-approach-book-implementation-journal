# knn_from_scratch.py
# NumPy-only k-NN classifier with scaling, metrics, weighting, and CV
from __future__ import annotations
import argparse, os, math, json
from typing import Tuple, List, Dict
import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# ------------------------- small helpers -------------------------

def load_table(path: str, sep: str = None) -> np.ndarray:
    """
    Load a numeric table. If sep is None, np.loadtxt auto-detects whitespace.
    Assumes last column is the label (string or number). If labels are strings,
    load twice: once for X (float), once for y (str).
    """
    # Try loading everything as float; if it fails, do mixed pass
    try:
        arr = np.loadtxt(path, delimiter=sep)
        return arr
    except Exception:
        # mixed types: load raw strings then split
        raw = np.genfromtxt(path, delimiter=sep, dtype=None, encoding="utf-8")
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        # Convert all but last column to float; keep last as string
        X = np.asarray(raw[:, :-1], dtype=float)
        y = raw[:, -1].astype(str)
        # stitch back into an object array for the caller to split
        arr = np.empty((X.shape[0], X.shape[1] + 1), dtype=object)
        arr[:, :-1] = X
        arr[:, -1] = y
        return arr


def split_Xy(table: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(table[:, :-1], dtype=float)
    y = table[:, -1]
    # ensure 1D string/int labels
    if np.issubdtype(y.dtype, np.number):
        y = y.astype(int)
    else:
        y = y.astype(str)
    return X, y


def train_test_split(X, y, test_size=0.25, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(round(n * float(test_size)))
    te = idx[:n_test]
    tr = idx[n_test:]
    return X[tr], X[te], y[tr], y[te]


def zscore_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sigma = X.std(axis=0, ddof=0)
    sigma = np.where(sigma == 0, 1.0, sigma)
    return mu, sigma


def zscore_transform(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return (X - mu) / sigma


class LabelMap:
    """Simple label encoder/decoder that preserves insertion order."""
    def __init__(self):
        self._to_idx: Dict[str, int] = {}
        self._to_lbl: List[str] = []

    def fit(self, y: np.ndarray):
        for v in y:
            v = str(v)
            if v not in self._to_idx:
                self._to_idx[v] = len(self._to_lbl)
                self._to_lbl.append(v)
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        return np.array([self._to_idx[str(v)] for v in y], dtype=int)

    def inverse_transform(self, yi: np.ndarray) -> np.ndarray:
        return np.array([self._to_lbl[int(i)] for i in yi], dtype=object)

    @property
    def classes_(self) -> List[str]:
        return list(self._to_lbl)


# ------------------------- distances -------------------------

def pairwise_distances(Xa: np.ndarray, Xb: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """
    Return matrix D[i,j] = dist(Xa[i], Xb[j]).
    """
    if metric == "euclidean":
        # (a - b)^2 = a^2 + b^2 - 2ab
        a2 = np.sum(Xa * Xa, axis=1, keepdims=True)            # (na,1)
        b2 = np.sum(Xb * Xb, axis=1, keepdims=True).T          # (1,nb)
        D2 = np.maximum(a2 + b2 - 2.0 * (Xa @ Xb.T), 0.0)
        return np.sqrt(D2, dtype=float)
    elif metric == "manhattan":
        # broadcasting; may be heavy for huge sets
        return np.sum(np.abs(Xa[:, None, :] - Xb[None, :, :]), axis=2, dtype=float)
    elif metric == "cosine":
        # cosine distance = 1 - cosine similarity
        Xa_n = Xa / np.linalg.norm(Xa, axis=1, keepdims=True).clip(min=1e-12)
        Xb_n = Xb / np.linalg.norm(Xb, axis=1, keepdims=True).clip(min=1e-12)
        S = Xa_n @ Xb_n.T
        return 1.0 - S
    else:
        raise ValueError(f"Unknown metric: {metric}")


# ------------------------- k-NN core -------------------------

def knn_predict(
    X_train: np.ndarray,
    y_train_idx: np.ndarray,     # int labels starting at 0
    X_test: np.ndarray,
    k: int = 3,
    metric: str = "euclidean",
    weighted: bool = False,
    chunk: int = 4096,
) -> np.ndarray:
    """
    Vectorized k-NN. If memory is tight, computes distances in chunks of test points.
    """
    n_classes = int(y_train_idx.max()) + 1
    y_pred_idx = np.empty(X_test.shape[0], dtype=int)

    for start in range(0, X_test.shape[0], chunk):
        end = min(start + chunk, X_test.shape[0])
        D = pairwise_distances(X_test[start:end], X_train, metric=metric)  # (m, n_train)

        # indices of k smallest distances per row
        nn_idx = np.argpartition(D, kth=min(k, D.shape[1]-1), axis=1)[:, :k]  # (m, k)

        # gather neighbor labels
        neigh_labels = y_train_idx[nn_idx]  # (m, k)

        if weighted:
            # inverse-distance weights (protect zero)
            w = 1.0 / (D[np.arange(D.shape[0])[:, None], nn_idx] + 1e-12)  # (m, k)
            # vote with weights using bincount per row
            votes = np.zeros((end - start, n_classes), dtype=float)
            for i in range(end - start):
                votes[i] = np.bincount(neigh_labels[i], weights=w[i], minlength=n_classes)
        else:
            # unweighted majority using bincount per row
            votes = np.zeros((end - start, n_classes), dtype=float)
            for i in range(end - start):
                votes[i] = np.bincount(neigh_labels[i], minlength=n_classes)

        y_pred_idx[start:end] = np.argmax(votes, axis=1)

    return y_pred_idx


# ------------------------- metrics / reporting -------------------------

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def per_class_accuracy(cm: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        acc = np.diag(cm) / cm.sum(axis=1).clip(min=1)
    return acc


def save_confusion_plot(cm: np.ndarray, class_names: List[str], path: str):
    if plt is None:
        print("[warn] matplotlib not available; cannot save confusion plot.")
        return
    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar(fraction=0.046, pad=0.04)
    tick = np.arange(len(class_names))
    plt.xticks(tick, class_names, rotation=45, ha="right")
    plt.yticks(tick, class_names)
    # annotate
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i,j] > cm.max()/2 else "black")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    print(f"[saved] {path}")


# ------------------------- CV for k -------------------------

def kfold_indices(n: int, folds: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    return np.array_split(idx, folds)

def choose_k_cv(
    X: np.ndarray,
    y_idx: np.ndarray,
    k_grid: List[int],
    folds: int = 5,
    metric: str = "euclidean",
    weighted: bool = False,
    seed: int = 42,
) -> int:
    parts = kfold_indices(len(y_idx), folds, seed=seed)
    scores = []
    for k in k_grid:
        correct = 0
        total = 0
        for i in range(folds):
            te = parts[i]
            tr = np.concatenate([parts[j] for j in range(folds) if j != i])
            y_pred = knn_predict(X[tr], y_idx[tr], X[te], k=k, metric=metric, weighted=weighted)
            correct += int((y_pred == y_idx[te]).sum())
            total += len(te)
        acc = correct / total if total else 0.0
        scores.append((k, acc))
    best_k = max(scores, key=lambda t: (t[1], -t[0]))[0]
    return best_k


# ------------------------- CLI -------------------------

def main():
    ap = argparse.ArgumentParser(description="NumPy-only k-NN classifier (scaling, metrics, weighting, CV).")
    ap.add_argument("--data", default="", help="path to CSV/TSV (last column is label). If empty, use built-in toy data.")
    ap.add_argument("--sep", default=None, help="delimiter (default: auto whitespace; use ',' for CSV, '\\t' for TSV)")
    ap.add_argument("--scale", action="store_true", help="z-score features using train statistics")
    ap.add_argument("--metric", choices=["euclidean", "manhattan", "cosine"], default="euclidean")
    ap.add_argument("--weighted", action="store_true", help="inverse-distance weighted vote")
    ap.add_argument("--k", type=int, default=3, help="neighbors (ignored if --cv-k is given)")
    ap.add_argument("--cv-k", type=int, nargs="+", default=[], help="grid of k values to choose via CV (e.g. 1 3 5 7)")
    ap.add_argument("--folds", type=int, default=5, help="CV folds (only if --cv-k is set)")
    ap.add_argument("--test-size", type=float, default=0.25, help="holdout fraction for final report")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-confusion", default="", help="path to save confusion matrix PNG")
    ap.add_argument("--dump-pred", default="", help="optional JSON to dump predictions and settings")
    args = ap.parse_args()

    # 1) Load data
    if args.data:
        table = load_table(args.data, sep=args.sep)
        X, y = split_Xy(table)
    else:
        # tiny 2D toy dataset (same as the one you saw)
        X = np.array([[1,101],[5,89],[108,5],[115,8]], dtype=float)
        y = np.array(["romance","romance","action","action"], dtype=object)

    # map labels -> integers
    le = LabelMap().fit(y)
    y_idx = le.transform(y)

    # 2) Train/val split (final evaluation uses this holdout)
    Xtr, Xte, ytr, yte = train_test_split(X, y_idx, test_size=args.test_size, seed=args.seed)

    # 3) Optional scaling
    if args.scale:
        mu, sigma = zscore_fit(Xtr)
        Xtr = zscore_transform(Xtr, mu, sigma)
        Xte = zscore_transform(Xte, mu, sigma)

    # 4) Choose k via CV if requested
    k = args.k
    if args.cv_k:
        k = choose_k_cv(Xtr, ytr, k_grid=args.cv_k, folds=args.folds,
                        metric=args.metric, weighted=args.weighted, seed=args.seed)

    # 5) Fit = store training set; Predict on test
    ypred_te = knn_predict(Xtr, ytr, Xte, k=k, metric=args.metric, weighted=args.weighted)

    # 6) Report
    acc = float((ypred_te == yte).mean())
    cm = confusion_matrix(yte, ypred_te, n_classes=len(le.classes_))
    per_cls = per_class_accuracy(cm)

    print(f"Accuracy: {acc:.4f}  (k={k}, weighted={args.weighted}, metric={args.metric})")
    print("Per-class accuracy:")
    for i, c in enumerate(le.classes_):
        print(f"  {c}: {per_cls[i]:.3f}")

    if args.save_confusion:
        save_confusion_plot(cm, le.classes_, args.save_confusion)

    if args.dump_pred:
        out = {
            "k": k,
            "weighted": args.weighted,
            "metric": args.metric,
            "classes": le.classes_,
            "true": [le.classes_[i] for i in yte],
            "pred": [le.classes_[i] for i in ypred_te],
            "accuracy": acc,
        }
        with open(args.dump_pred, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"[saved] {args.dump_pred}")


if __name__ == "__main__":
    main()
# Run
# python "19.7 NonParametric Algorithms/KNN/KNN Ex 2/Simple KNN/knn_from_scratch.py" \
#   --scale --metric euclidean --weighted \
#   --cv-k 1 3 5 7 --folds 4 \
#   --save-confusion "19.7 NonParametric Algorithms/KNN/KNN Ex 2/Simple KNN/toy_knn_cm.png"
