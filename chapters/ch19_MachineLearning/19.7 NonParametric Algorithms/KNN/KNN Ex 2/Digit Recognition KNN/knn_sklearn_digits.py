# knn_sklearn_digits.py
# -*- coding: utf-8 -*-
"""
KNN on 32x32 ASCII digit files (0/1 chars). Uses scikit-learn.

- Train on a folder of files like "3_45.txt" (label is the number before '_')
- Each file has 32 lines of 32 characters in {'0','1'} forming a 32×32 bitmap
- Optionally tune k via cross-validation on the training set
- Optionally save a confusion matrix and a "mean digit" grid image

Example:
python knn_sklearn_digits.py \
  --train-dir "trainingDigits" \
  --test-dir  "testDigits" \
  --k 3 --weights distance --metric euclidean \
  --save-confusion knn_confusion.png \
  --save-means    knn_means.png
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score


# ---------------------------- IO helpers ----------------------------

def resolve(path: str) -> str:
    """Return absolute path; if relative, make it relative to current working dir."""
    return path if os.path.isabs(path) else os.path.join(os.getcwd(), path)

def parse_digit_file(path: str) -> np.ndarray:
    """
    Read a 32x32 ASCII '0'/'1' digit file and return a (1024,) uint8 array.
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()
    if len(lines) != 32:
        raise ValueError(f"{path}: expected 32 lines, got {len(lines)}")
    flat: List[int] = []
    for line in lines:
        if len(line) < 32:
            raise ValueError(f"{path}: each line must have 32 characters")
        # only take first 32 chars to be safe
        flat.extend(1 if c == '1' else 0 for c in line[:32])
    arr = np.fromiter(flat, dtype=np.uint8)
    if arr.size != 1024:
        raise ValueError(f"{path}: parsed size {arr.size}, expected 1024")
    return arr

def load_dir(dir_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load all .txt images in a directory.
    Returns X (n_samples, 1024) uint8 and y (n_samples,) int labels.
    """
    dir_path = resolve(dir_path)
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    files = sorted([f for f in os.listdir(dir_path) if f.endswith(".txt")])
    if not files:
        raise FileNotFoundError(f"No .txt images found in {dir_path}")

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    for fname in files:
        # label is the prefix before the underscore
        try:
            label = int(fname.split("_")[0])
        except Exception:
            raise ValueError(f"Cannot parse label from filename: {fname}")
        X_list.append(parse_digit_file(os.path.join(dir_path, fname)))
        y_list.append(label)

    X = np.vstack(X_list).astype(np.float32)  # float32 for distance math
    y = np.asarray(y_list, dtype=int)
    return X, y


# ---------------------------- Plotting ----------------------------

def save_confusion(cm: np.ndarray, class_names: List[str], out_path: str, normalize: bool = True) -> None:
    """
    Save a confusion matrix heatmap using matplotlib (no seaborn dependency).
    """
    if normalize:
        with np.errstate(invalid="ignore", divide="ignore"):
            cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            cm = np.nan_to_num(cm)

    plt.figure(figsize=(6.5, 5.5))
    im = plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix" + (" (normalized)" if normalize else ""))
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            txt = f"{val:.2f}" if normalize else f"{int(val)}"
            plt.text(j, i, txt,
                     ha="center", va="center",
                     color="white" if val > thresh else "black", fontsize=9)

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    os.makedirs(os.path.dirname(resolve(out_path)) or ".", exist_ok=True)
    plt.savefig(resolve(out_path), dpi=150)
    plt.close()

def save_class_means(X: np.ndarray, y: np.ndarray, out_path: str) -> None:
    """
    Compute the mean bitmap per class and save a grid image (2x5 for digits 0–9).
    Works with any subset of classes; empty panels will be blank.
    """
    labels = sorted(set(y.tolist()))
    means = {}
    for c in labels:
        Xc = X[y == c]
        means[c] = Xc.mean(axis=0).reshape(32, 32)

    # Determine grid (default for digits: 2x5)
    uniq = sorted(labels)
    rows, cols = (2, 5) if len(uniq) <= 10 else (int(np.ceil(len(uniq)/5)), 5)

    fig, axes = plt.subplots(rows, cols, figsize=(1.8*cols, 1.8*rows))
    axes = np.atleast_2d(axes)
    k = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            ax.axis("off")
            if k < len(uniq):
                label = uniq[k]
                img = means[label]
                ax.imshow(img, cmap="gray_r", vmin=0.0, vmax=1.0)
                ax.set_title(str(label), fontsize=10)
                k += 1

    plt.tight_layout()
    os.makedirs(os.path.dirname(resolve(out_path)) or ".", exist_ok=True)
    plt.savefig(resolve(out_path), dpi=150)
    plt.close()


# ---------------------------- Main ----------------------------

def main():
    ap = argparse.ArgumentParser(description="KNN on ASCII 32×32 digit images (scikit-learn).")
    ap.add_argument("--train-dir", default="trainingDigits", help="folder with training .txt images")
    ap.add_argument("--test-dir",  default="testDigits",     help="folder with test .txt images")
    ap.add_argument("--k", type=int, default=3, help="neighbors (ignored if --tune-k is used)")
    ap.add_argument("--tune-k", type=int, nargs="+", default=None,
                    help="if given, perform CV on these k values and pick the best")
    ap.add_argument("--weights", choices=["uniform", "distance"], default="distance",
                    help="vote weighting")
    ap.add_argument("--metric", choices=["euclidean", "manhattan", "minkowski"], default="euclidean")
    ap.add_argument("--p", type=int, default=2, help="Minkowski power parameter p (used if metric=minkowski)")
    ap.add_argument("--n-jobs", type=int, default=-1, help="threads for KNN (distance computations)")
    ap.add_argument("--cv-folds", type=int, default=5, help="folds for k tuning if --tune-k is set")
    ap.add_argument("--save-confusion", default="", help="path to save confusion matrix PNG")
    ap.add_argument("--save-means", default="", help="path to save class-mean grid PNG")
    args = ap.parse_args()

    # 1) Load data
    X_train, y_train = load_dir(args.train_dir)
    X_test,  y_test  = load_dir(args.test_dir)

    # 2) Optionally tune k via CV on the training set
    if args.tune_k:
        print(f"[cv] searching k in {args.tune_k} (weights={args.weights}, metric={args.metric})")
        best_k = None
        best_score = -np.inf
        for k in args.tune_k:
            clf = KNeighborsClassifier(
                n_neighbors=k,
                weights=args.weights,
                metric=("minkowski" if args.metric == "minkowski" else args.metric),
                p=args.p,
                n_jobs=args.n_jobs,
            )
            cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
            scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=args.n_jobs)
            mean_acc = scores.mean()
            print(f"  k={k:<2d} -> CV accuracy={mean_acc:.4f}")
            if mean_acc > best_score:
                best_score = mean_acc
                best_k = k
        print(f"[cv] best k = {best_k} (acc={best_score:.4f})")
        k = best_k
    else:
        k = args.k

    # 3) Train final KNN
    clf = KNeighborsClassifier(
        n_neighbors=k,
        weights=args.weights,
        metric=("minkowski" if args.metric == "minkowski" else args.metric),
        p=args.p,
        n_jobs=args.n_jobs,
    )
    clf.fit(X_train, y_train)

    # 4) Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}  (k={k}, weights={args.weights}, metric={args.metric})")

    # Per-class accuracy
    labels_sorted = sorted(set(y_test.tolist()) | set(y_train.tolist()))
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
    per_class = []
    for i, lab in enumerate(labels_sorted):
        tp = cm[i, i]
        total = cm[i].sum()
        per_class.append((lab, (tp / total) if total else np.nan))
    print("Per-class accuracy:")
    for lab, val in per_class:
        s = "n/a" if np.isnan(val) else f"{val:.3f}"
        print(f"  {lab}: {s}")

    # (Optional) full text report
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, labels=labels_sorted, digits=3, zero_division=0))

    # 5) Plots
    if args.save_confusion:
        save_confusion(cm, [str(c) for c in labels_sorted], args.save_confusion, normalize=True)
        print(f"[saved] {args.save_confusion}")
    if args.save_means:
        # Combine train+test for a cleaner average look (or just use train)
        X_all = np.vstack([X_train, X_test])
        y_all = np.hstack([y_train, y_test])
        save_class_means(X_all, y_all, args.save_means)
        print(f"[saved] {args.save_means}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)

#Run 
# python "19.7 NonParametric Algorithms/KNN/KNN Ex 2/Digit Recognition KNN/knn_sklearn_digits.py" \
#   --train-dir "19.7 NonParametric Algorithms/KNN/KNN Ex 2/Digit Recognition KNN/trainingDigits" \
#   --test-dir  "19.7 NonParametric Algorithms/KNN/KNN Ex 2/Digit Recognition KNN/testDigits" \
#   --tune-k 1 3 5 7 9 \
#   --weights distance --metric euclidean \
#   --save-confusion "19.7 NonParametric Algorithms/KNN/KNN Ex 2/Digit Recognition KNN/knn_confusion.png" \
#   --save-means    "19.7 NonParametric Algorithms/KNN/KNN Ex 2/Digit Recognition KNN/knn_means.png"