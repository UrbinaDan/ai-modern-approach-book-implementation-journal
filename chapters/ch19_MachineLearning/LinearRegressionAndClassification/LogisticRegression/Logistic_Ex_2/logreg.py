# logreg.py
# This code implements logistic regression using batch gradient ascent.
# It loads a dataset from a text file, trains the model, evaluates accuracy,
# and visualizes the decision boundary.
# Ensure you have numpy and matplotlib installed to run this script.
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import os

# --------------------------- Data ---------------------------

def load_dataset(path: str = "testSet.txt"):
    # resolve relative to this script’s directory
    if not os.path.isabs(path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(script_dir, path)

    X, y = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            x1, x2, lbl = float(parts[0]), float(parts[1]), int(parts[2])
            X.append([1.0, x1, x2])
            y.append(lbl)
    import numpy as np
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64).reshape(-1, 1)
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

# --------------------------- Training ---------------------------

def train_logreg_batch(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 1e-3,
    iters: int = 500,
    l2: float = 0.0,
) -> np.ndarray:
    """
    Batch gradient ascent on the log-likelihood with optional L2.
    Returns weights of shape (n, 1).
    """
    m, n = X.shape
    w = np.ones((n, 1), dtype=np.float64)
    for _ in range(iters):
        p = sigmoid_stable(X @ w)                 # (m,1)
        grad = X.T @ (y - p) - l2 * w            # ascent (y - p), minus L2
        w += alpha * grad
    return w

# --------------------------- Eval ---------------------------

def accuracy(X: np.ndarray, y: np.ndarray, w: np.ndarray, thr: float = 0.5) -> float:
    p = sigmoid_stable(X @ w).ravel()
    yhat = (p > thr).astype(np.int64)
    return float((yhat == y.ravel()).mean())

# --------------------------- Plots ---------------------------

def plot_dataset(X: np.ndarray, y: np.ndarray, title: str = "DataSet", save: str | None = None):
    pos = y.ravel() == 1
    neg = ~pos
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(X[pos, 1], X[pos, 2], s=20, c="tab:red", marker="s", alpha=0.6, label="1")
    ax.scatter(X[neg, 1], X[neg, 2], s=20, c="tab:green", alpha=0.6, label="0")
    ax.set_title(title)
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.legend()
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=180, bbox_inches="tight")
    else:
        plt.show()

def plot_decision_boundary(
    X: np.ndarray, y: np.ndarray, w: np.ndarray,
    title: str = "BestFit", save: str | None = None
):
    pos = y.ravel() == 1
    neg = ~pos
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(X[pos, 1], X[pos, 2], s=20, c="tab:red", marker="s", alpha=0.6, label="1")
    ax.scatter(X[neg, 1], X[neg, 2], s=20, c="tab:green", alpha=0.6, label="0")

    x_min, x_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xs = np.linspace(x_min, x_max, 200)
    # w0 + w1*x + w2*y = 0  =>  y = -(w0 + w1*x) / w2
    if abs(w[2, 0]) > 1e-12:
        ys = -(w[0, 0] + w[1, 0] * xs) / w[2, 0]
        ax.plot(xs, ys, lw=2, label="Decision boundary")

    ax.set_title(title)
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.legend()
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=180, bbox_inches="tight")
    else:
        plt.show()

# --------------------------- Demo: 1D gradient ascent ---------------------------

def gradient_ascent_1d_demo(alpha: float = 0.01, precision: float = 1e-8) -> float:
    """Maximize f(x) = -x^2 + 4x via gradient ascent."""
    def f_prime(x): return -2.0 * x + 4.0
    x_old, x_new = -1.0, 0.0
    while abs(x_new - x_old) > precision:
        x_old = x_new
        x_new = x_old + alpha * f_prime(x_old)
    return x_new  # ~ 2.0

# --------------------------- CLI ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Modern logistic regression demo (batch ascent).")
    ap.add_argument("--data", default="testSet.txt", help="path to dataset (x1 x2 label)")
    ap.add_argument("--alpha", type=float, default=1e-3, help="learning rate")
    ap.add_argument("--iters", type=int, default=500, help="iterations")
    ap.add_argument("--l2", type=float, default=0.0, help="L2 regularization strength")
    ap.add_argument("--save", action="store_true", help="save plots instead of showing")
    args = ap.parse_args()

    # Demo: 1D gradient ascent
    x_star = gradient_ascent_1d_demo()
    print(f"1D gradient-ascent demo maximizer ~ {x_star:.6f} (true argmax = 2.0)")

    # Load, train, report
    X, y = load_dataset(args.data)
    w = train_logreg_batch(X, y, alpha=args.alpha, iters=args.iters, l2=args.l2)
    acc = accuracy(X, y, w)
    print(f"Training accuracy: {acc:.4f}")
    print(f"Weights: {w.ravel()}")

    # Plot
    if args.save:
        plot_dataset(X, y, title="DataSet", save="dataset.png")
        plot_decision_boundary(X, y, w, title="BestFit", save="decision_boundary.png")
        print("Saved plots: dataset.png, decision_boundary.png")
    else:
        plot_dataset(X, y, title="DataSet")
        plot_decision_boundary(X, y, w, title="BestFit")

if __name__ == "__main__":
    main()
