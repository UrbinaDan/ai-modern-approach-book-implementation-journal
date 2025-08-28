# -*- coding: utf-8 -*-
"""
From-scratch k-NN for 32x32 ASCII digit images (like 'trainingDigits'/'testDigits').
- Loads each 32x32 text image (0/1 chars) into a 1x1024 vector
- Classifies test images with k-NN (Euclidean; optional distance-weighted vote)
- Prints accuracy and saves:
  * confusion matrix heatmap
  * "weight heatmap": per-class average image (class prototype)

Usage:
  python knn_digits.py \
    --train-dir trainingDigits --test-dir testDigits \
    --k 3 --weighted \
    --save-confusion knn_confusion.png \
    --save-means weights_heatmap.png
"""

import os, glob, argparse
import numpy as np
import matplotlib.pyplot as plt

# ---------- file / parsing utilities ----------

def resolve(path: str) -> str:
    """Return absolute path (handle relative nicely)."""
    return path if os.path.isabs(path) else os.path.join(os.getcwd(), path)

def parse_label_from_filename(path: str) -> int:
    """
    Filenames look like '7_45.txt' -> label 7.
    We only use the number before the underscore.
    """
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    return int(stem.split('_')[0])

def img32x32_to_vec(path: str) -> np.ndarray:
    """
    Load a 32x32 ASCII image of '0'/'1' chars into a flat 1x1024 float vector.
    """
    vec = np.zeros(1024, dtype=np.float32)
    with open(path, "r") as f:
        for i in range(32):
            line = f.readline().strip()
            # Convert each char ('0' or '1') to int and place
            for j in range(32):
                vec[32 * i + j] = 1.0 if line[j] != '0' else 0.0
    return vec

def load_digit_dir(dir_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Read all *.txt images from a directory, return (X, y).
    X shape: (n_samples, 1024), y shape: (n_samples,)
    """
    files = sorted(glob.glob(os.path.join(resolve(dir_path), "*.txt")))
    X = np.vstack([img32x32_to_vec(fp) for fp in files])
    y = np.array([parse_label_from_filename(fp) for fp in files], dtype=int)
    return X, y

# ---------- k-NN core ----------

def knn_predict_one(x: np.ndarray,
                    X_train: np.ndarray,
                    y_train: np.ndarray,
                    k: int = 3,
                    weighted: bool = False) -> int:
    """
    Classify one vector x using k-NN with Euclidean distance.
    If weighted=True, neighbors vote with weight 1/(d+1e-9).
    """
    # Euclidean distance to all train samples (vectorized)
    dists = np.linalg.norm(X_train - x, axis=1)
    # Get indices of k smallest distances (argpartition is O(n))
    nn_idx = np.argpartition(dists, k)[:k]
    nn_labels = y_train[nn_idx]

    if not weighted:
        # Majority vote
        vals, counts = np.unique(nn_labels, return_counts=True)
        return int(vals[np.argmax(counts)])

    # Distance-weighted vote
    nn_d = dists[nn_idx]
    weights = 1.0 / (nn_d + 1e-9)
    # Sum weights per class
    scores = {}
    for lab, w in zip(nn_labels, weights):
        scores[lab] = scores.get(lab, 0.0) + float(w)
    # Pick the class with highest total weight
    return max(scores.items(), key=lambda kv: kv[1])[0]

def knn_predict(X_test: np.ndarray,
                X_train: np.ndarray,
                y_train: np.ndarray,
                k: int = 3,
                weighted: bool = False) -> np.ndarray:
    """
    Predict labels for a matrix of test vectors.
    """
    preds = [knn_predict_one(x, X_train, y_train, k=k, weighted=weighted)
             for x in X_test]
    return np.array(preds, dtype=int)

# ---------- plotting helpers ----------

def save_confusion_matrix(cm: np.ndarray, labels: list[int], path: str, title: str = "Confusion matrix"):
    """
    Save a confusion matrix heatmap.
    cm: rows=true, cols=pred
    """
    plt.figure(figsize=(7.2, 6))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    # annotate counts
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]),
                     ha="center", va="center",
                     color="black" if cm[i, j] < cm.max()*0.7 else "white")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def save_class_mean_heatmaps(X: np.ndarray, y: np.ndarray, path: str):
    """
    Compute per-class mean image (prototype) and save as a grid heatmap.
    This acts as a 'weight heatmap' for k-NN (it has no parameters to plot).
    """
    classes = sorted(np.unique(y).tolist())  # typically 0..9
    n = len(classes)
    ncols = 5
    nrows = int(np.ceil(n / ncols))
    plt.figure(figsize=(1.8*ncols, 1.8*nrows))
    for idx, c in enumerate(classes, start=1):
        mean_img = X[y == c].mean(axis=0).reshape(32, 32)
        ax = plt.subplot(nrows, ncols, idx)
        ax.imshow(mean_img, cmap="gray_r", vmin=0, vmax=1)
        ax.set_title(f"{c}")
        ax.axis("off")
    plt.suptitle("Class mean images (\"weight\" heatmaps)")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(path, dpi=150)
    plt.close()

# ---------- metrics ----------

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> np.ndarray:
    """
    Simple confusion matrix without sklearn.
    """
    L = len(labels)
    lab2idx = {lab: i for i, lab in enumerate(labels)}
    cm = np.zeros((L, L), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[lab2idx[t], lab2idx[p]] += 1
    return cm

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="From-scratch k-NN for 32x32 ASCII digit images.")
    ap.add_argument("--train-dir", default="trainingDigits", help="folder with training *.txt")
    ap.add_argument("--test-dir",  default="testDigits",     help="folder with test *.txt")
    ap.add_argument("--k", type=int, default=3, help="number of neighbors")
    ap.add_argument("--weighted", action="store_true", help="distance-weighted vote (1/(d+ε))")
    ap.add_argument("--save-confusion", default="knn_confusion.png", help="path to save confusion heatmap")
    ap.add_argument("--save-means",     default="weights_heatmap.png", help="path to save class-mean heatmaps")
    args = ap.parse_args()

    # 1) Load data
    Xtr, ytr = load_digit_dir(args.train_dir)
    Xte, yte = load_digit_dir(args.test_dir)

    # 2) Predict
    yhat = knn_predict(Xte, Xtr, ytr, k=args.k, weighted=args.weighted)

    # 3) Report accuracy
    acc = (yhat == yte).mean()
    print(f"Accuracy: {acc:.4f}  (k={args.k}, weighted={args.weighted})")
    # per-class accuracy
    labels = sorted(np.unique(np.concatenate([ytr, yte])).tolist())
    cm = confusion_matrix(yte, yhat, labels)
    per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    print("Per-class accuracy:")
    for lab, a in zip(labels, per_class_acc):
        print(f"  {lab}: {a:.3f}")

    # 4) Save confusion matrix heatmap
    if args.save_confusion:
        save_confusion_matrix(cm, labels, args.save_confusion,
                              title=f"Confusion (k={args.k}, weighted={args.weighted})")
        print(f"[saved] {args.save_confusion}")

    # 5) Save "weight heatmap" (class mean images)
    if args.save_means:
        save_class_mean_heatmaps(Xtr, ytr, args.save_means)
        print(f"[saved] {args.save_means}")

if __name__ == "__main__":
    main()

#run 
# python "19.7 NonParametric Algorithms/KNN/KNN Ex 1/knn_digits.py" \
#   --train-dir "19.7 NonParametric Algorithms/KNN/KNN Ex 1/trainingDigits" \
#   --test-dir  "19.7 NonParametric Algorithms/KNN/KNN Ex 1/testDigits" \
#   --k 3 --weighted \
#   --save-confusion "19.7 NonParametric Algorithms/KNN/KNN Ex 1/knn_confusion.png" \
#   --save-means "19.7 NonParametric Algorithms/KNN/KNN Ex 1/weights_heatmap.png"
