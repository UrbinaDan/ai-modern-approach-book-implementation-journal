# colic_logreg_modern.py
# -*- coding: utf-8 -*-
# This code implements logistic regression on the Horse Colic dataset using either
# custom batch gradient ascent, improved stochastic gradient ascent, or scikit-learn's implementation.
# It supports optional feature standardization and L2 regularization.
# The script evaluates model accuracy on a test set and can print a detailed classification report.
# Ensure you have numpy, scikit-learn installed to run this script.
# Example usage:
# with sklearn logistic regression:
# python colic_logreg_modern.py --method sklearn --solver saga --max-iter 5000 --C 1.0


# with standardization:
# python colic_logreg_modern.py --scale

# custom batch (with bias term handled internally):
# python colic_logreg_modern.py --method batch --alpha 0.001 --iters 800 --l2 0.0

# custom improved SGD:
# python colic_logreg_modern.py --method sgd --epochs 200 --l2 1e-4 --seed 7

# explicit data paths (any working dir):
# python colic_logreg_modern.py \
#   --train path/to/horseColicTraining.txt \
#   --test  path/to/horseColicTest.txt \
#   --method sklearn --scale --report


from __future__ import annotations

import os
import argparse
import numpy as np
from typing import Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# --------------------------- I/O ---------------------------

def resolve_path(path: str) -> str:
    """Resolve relative paths against this script's directory."""
    if os.path.isabs(path):
        return path
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, path)

def load_tsv_xy(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load tab-separated file where last column is the label (0/1)."""
    path = resolve_path(path)
    X, y = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            *feat, lbl = parts
            try:
                X.append([float(v) for v in feat])
                y.append(int(float(lbl)))   # handle "0.0"/"1.0"
            except ValueError:
                # skip malformed lines
                continue
    X = np.asarray(X, dtype=np.float64)              # (m, n)
    y = np.asarray(y, dtype=np.int64).reshape(-1, 1) # (m, 1)
    return X, y


# --------------------------- Math ---------------------------

def sigmoid_stable(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out


# --------------------------- Custom training ---------------------------

def train_batch(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 1e-3,
    iters: int = 500,
    l2: float = 0.0,
) -> np.ndarray:
    """
    Batch gradient ascent on logistic log-likelihood with optional L2.
    Returns weights (n,1).
    """
    m, n = X.shape
    w = np.ones((n, 1), dtype=np.float64)
    for _ in range(iters):
        p = sigmoid_stable(X @ w)            # (m,1)
        grad = X.T @ (y - p) - l2 * w        # ascent
        w += alpha * grad
    return w


def train_sgd_improved(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 150,
    l2: float = 0.0,
    seed: int | None = 42,
) -> np.ndarray:
    """
    Improved stochastic gradient ascent: one sample per step,
    decaying step size, shuffling each epoch. Returns (n,1).
    """
    rng = np.random.default_rng(seed)
    m, n = X.shape
    w = np.ones((n,), dtype=np.float64)

    for epoch in range(epochs):
        idx = rng.permutation(m)
        for i, k in enumerate(idx):
            # decaying step size
            alpha = 4.0 / (1.0 + epoch + i) + 0.01
            z = float(np.dot(X[k], w))           # scalar
            p = 1.0 / (1.0 + np.exp(-z))         # scalar is fine
            err = float(y[k, 0]) - p
            # ascent with L2 shrinkage
            w += alpha * (err * X[k] - l2 * w)

    return w.reshape(-1, 1)


# --------------------------- Evaluation helpers ---------------------------

def predict_proba(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return sigmoid_stable(X @ w).ravel()

def predict_label(X: np.ndarray, w: np.ndarray, thr: float = 0.5) -> np.ndarray:
    return (predict_proba(X, w) > thr).astype(np.int64)


# --------------------------- Main ---------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Horse Colic Logistic Regression (custom batch/SGD or scikit-learn)."
    )
    ap.add_argument("--train", default="horseColicTraining.txt", help="path to training TSV")
    ap.add_argument("--test",  default="horseColicTest.txt",     help="path to test TSV")

    ap.add_argument("--method", choices=["sklearn", "batch", "sgd"], default="sklearn",
                    help="which trainer to use")

    # Custom training params
    ap.add_argument("--alpha", type=float, default=1e-3, help="batch learning rate")
    ap.add_argument("--iters", type=int,   default=500,  help="batch iterations")
    ap.add_argument("--epochs", type=int,  default=150,  help="SGD epochs")
    ap.add_argument("--l2", type=float,    default=0.0,  help="L2 regularization (custom)")
    ap.add_argument("--seed", type=int,    default=42,   help="random seed for SGD shuffling")

    # Sklearn params
    ap.add_argument("--solver", default="saga", help="sklearn solver (e.g., liblinear, lbfgs, sag, saga)")
    ap.add_argument("--max-iter", type=int, default=5000, help="sklearn max_iter")
    ap.add_argument("--C", type=float, default=1.0, help="sklearn inverse regularization strength")

    # Common
    ap.add_argument("--scale", action="store_true", help="standardize features (train+test)")

    # Reporting
    ap.add_argument("--report", action="store_true", help="print classification report")

    args = ap.parse_args()

    # Load data
    X_tr, y_tr = load_tsv_xy(args.train)
    X_te, y_te = load_tsv_xy(args.test)

    # Optional standardization (fit on train, apply to test)
    if args.scale:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

    # Train
    if args.method == "sklearn":
        clf = LogisticRegression(
            solver=args.solver,
            max_iter=args.max_iter,
            C=args.C,
        )
        clf.fit(X_tr, y_tr.ravel())
        y_pred = clf.predict(X_te)
        acc = accuracy_score(y_te.ravel(), y_pred)
        cm = confusion_matrix(y_te.ravel(), y_pred)
        print(f"[sklearn] accuracy: {acc:.4f}")
        print("Confusion matrix:\n", cm)
        if args.report:
            print(classification_report(y_te.ravel(), y_pred, digits=4))

    elif args.method == "batch":
        # Add bias term column
        Xtr_b = np.hstack([np.ones((X_tr.shape[0], 1)), X_tr])
        Xte_b = np.hstack([np.ones((X_te.shape[0], 1)), X_te])
        w = train_batch(Xtr_b, y_tr, alpha=args.alpha, iters=args.iters, l2=args.l2)
        y_pred = predict_label(Xte_b, w)
        acc = accuracy_score(y_te.ravel(), y_pred)
        cm = confusion_matrix(y_te.ravel(), y_pred)
        print(f"[custom-batch] accuracy: {acc:.4f}")
        print("Confusion matrix:\n", cm)
        if args.report:
            print(classification_report(y_te.ravel(), y_pred, digits=4))

    else:  # args.method == "sgd"
        Xtr_b = np.hstack([np.ones((X_tr.shape[0], 1)), X_tr])
        Xte_b = np.hstack([np.ones((X_te.shape[0], 1)), X_te])
        w = train_sgd_improved(Xtr_b, y_tr, epochs=args.epochs, l2=args.l2, seed=args.seed)
        y_pred = predict_label(Xte_b, w)
        acc = accuracy_score(y_te.ravel(), y_pred)
        cm = confusion_matrix(y_te.ravel(), y_pred)
        print(f"[custom-sgd] accuracy: {acc:.4f}")
        print("Confusion matrix:\n", cm)
        if args.report:
            print(classification_report(y_te.ravel(), y_pred, digits=4))


if __name__ == "__main__":
    main()
