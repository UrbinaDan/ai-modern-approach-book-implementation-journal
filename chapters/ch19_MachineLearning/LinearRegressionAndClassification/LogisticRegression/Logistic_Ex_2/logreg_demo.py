# logreg_demo.py
# This code demonstrates logistic regression using both batch gradient ascent and improved stochastic gradient ascent.
# It loads a simple 2D dataset from a text file, trains both models, evaluates their accuracy, and visualizes the results.
# The decision boundary learned by the SGD model is plotted, along with the weight trajectories for both methods.
# Ensure you have numpy and matplotlib installed to run this script.
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# --------------------------- Data ---------------------------

def load_dataset(path: str = "testSet.txt"):
    # resolve relative to this script file
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

def sigmoid_stable(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[neg])
    out[neg] = ex / (1.0 + ex)
    return out

@dataclass
class TrainConfig:
    alpha: float = 0.01     # learning rate (batch)
    iters: int = 500        # iterations (batch)
    l2: float = 0.0         # L2 regularization (batch & SGD)
    sgd_epochs: int = 150
    seed: int | None = 42

# --------------------------- Training ---------------------------

def train_batch(X: np.ndarray, y: np.ndarray, cfg: TrainConfig) -> tuple[np.ndarray, np.ndarray]:
    """Batch gradient ascent on log-likelihood."""
    m, n = X.shape
    w = np.ones((n, 1), dtype=np.float64)
    w_hist = np.empty((cfg.iters, n), dtype=np.float64)

    for t in range(cfg.iters):
        p = sigmoid_stable(X @ w)                  # (m,1)
        grad = X.T @ (y - p) - cfg.l2 * w          # ascent with L2
        w += cfg.alpha * grad
        w_hist[t] = w.ravel()

    return w, w_hist

def train_sgd_improved(X: np.ndarray, y: np.ndarray, cfg: TrainConfig) -> tuple[np.ndarray, np.ndarray]:
    """Improved stochastic gradient ascent with decaying step size and shuffle."""
    rng = np.random.default_rng(cfg.seed)
    m, n = X.shape
    w = np.ones((n,), dtype=np.float64)
    w_hist = []

    for epoch in range(cfg.sgd_epochs):
        idx = rng.permutation(m)
        for i, k in enumerate(idx):
            alpha = 4.0 / (1.0 + epoch + i) + 0.01   # decays over time
            z = np.dot(X[k], w)                      # scalar
            p = 1.0 / (1.0 + np.exp(-z))             # fine here (scalar)
            err = y[k, 0] - p
            # ascent step with optional L2
            w += alpha * (err * X[k] - cfg.l2 * w)
            w_hist.append(w.copy())

    return w.reshape(-1, 1), np.asarray(w_hist)

# --------------------------- Evaluation ---------------------------

def accuracy(X: np.ndarray, y: np.ndarray, w: np.ndarray, thr: float = 0.5) -> float:
    p = sigmoid_stable(X @ w).ravel()
    yhat = (p > thr).astype(np.int64)
    return float((yhat == y.ravel()).mean())

# --------------------------- Plotting ---------------------------

def plot_decision_boundary(X: np.ndarray, y: np.ndarray, w: np.ndarray,
                           title: str = "Decision Boundary", save: str | None = None):
    """Plot data points and linear boundary w0 + w1*x + w2*y = 0."""
    pos = y.ravel() == 1
    neg = ~pos
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(X[pos, 1], X[pos, 2], s=30, c="tab:red", marker="s", alpha=0.7, label="1")
    ax.scatter(X[neg, 1], X[neg, 2], s=30, c="tab:green", alpha=0.7, label="0")
    x = np.linspace(X[:,1].min()-0.5, X[:,1].max()+0.5, 200)
    # w0 + w1*x + w2*y = 0 -> y = -(w0 + w1*x)/w2
    if abs(w[2,0]) > 1e-12:
        y_line = -(w[0,0] + w[1,0]*x) / w[2,0]
        ax.plot(x, y_line, lw=2)
    ax.set_title(title)
    ax.set_xlabel("X1"); ax.set_ylabel("X2")
    ax.legend()
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=180, bbox_inches="tight")
    else:
        plt.show()

def plot_weight_trajectories(w_hist_sgd: np.ndarray, w_hist_batch: np.ndarray,
                             save: str | None = None):
    """Plot w0, w1, w2 over iterations for SGD (left) and batch (right)."""
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(12, 8), sharex=False, sharey=False)
    # left: SGD
    x1 = np.arange(len(w_hist_sgd))
    axs[0,0].plot(x1, w_hist_sgd[:,0]); axs[0,0].set_title("SGD: w0 vs iter"); axs[0,0].set_ylabel("w0")
    axs[1,0].plot(x1, w_hist_sgd[:,1]); axs[1,0].set_ylabel("w1")
    axs[2,0].plot(x1, w_hist_sgd[:,2]); axs[2,0].set_xlabel("iteration"); axs[2,0].set_ylabel("w2")
    # right: Batch
    x2 = np.arange(len(w_hist_batch))
    axs[0,1].plot(x2, w_hist_batch[:,0]); axs[0,1].set_title("Batch: w0 vs iter"); axs[0,1].set_ylabel("w0")
    axs[1,1].plot(x2, w_hist_batch[:,1]); axs[1,1].set_ylabel("w1")
    axs[2,1].plot(x2, w_hist_batch[:,2]); axs[2,1].set_xlabel("iteration"); axs[2,1].set_ylabel("w2")
    fig.suptitle("Logistic Regression – Weight Trajectories", y=0.98)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=180, bbox_inches="tight")
    else:
        plt.show()

# --------------------------- CLI ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Modern logistic regression demo (batch vs improved SGD).")
    ap.add_argument("--data", default="testSet.txt", help="path to dataset text file")
    ap.add_argument("--alpha", type=float, default=0.01, help="batch learning rate")
    ap.add_argument("--iters", type=int, default=500, help="batch iterations")
    ap.add_argument("--l2", type=float, default=0.0, help="L2 regularization (both methods)")
    ap.add_argument("--sgd-epochs", type=int, default=150, help="SGD epochs")
    ap.add_argument("--seed", type=int, default=42, help="random seed for SGD shuffling")
    ap.add_argument("--save", action="store_true", help="save plots to PNG instead of showing")
    args = ap.parse_args()

    X, y = load_dataset(args.data)
    cfg = TrainConfig(alpha=args.alpha, iters=args.iters, l2=args.l2,
                      sgd_epochs=args.sgd_epochs, seed=args.seed)

    # Train both ways
    w_sgd, w_hist_sgd = train_sgd_improved(X, y, cfg)
    w_batch, w_hist_batch = train_batch(X, y, cfg)

    # Report accuracies
    acc_sgd = accuracy(X, y, w_sgd)
    acc_batch = accuracy(X, y, w_batch)
    print(f"Accuracy (SGD):   {acc_sgd:.4f}")
    print(f"Accuracy (Batch): {acc_batch:.4f}")

    # Plots
    out_weights = "weights_trajectories.png" if args.save else None
    out_boundary = "decision_boundary_demo.png" if args.save else None
    plot_weight_trajectories(w_hist_sgd, w_hist_batch, save=out_weights)
    plot_decision_boundary(X, y, w_sgd, title="Decision Boundary (SGD)", save=out_boundary)

if __name__ == "__main__":
    main()
