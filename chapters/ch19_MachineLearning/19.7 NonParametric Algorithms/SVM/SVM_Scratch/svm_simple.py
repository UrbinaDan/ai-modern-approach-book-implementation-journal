# svm_simple.py
# ------------------------------------------------------------
# Simplified SMO for a LINEAR SVM on small 2-D data (no kernels).
# - Loads a tab-separated file with columns: x1, x2, label(±1)
# - Trains with a lightweight SMO loop (random partner j)
# - Computes w,b; plots decision boundary + support vectors (optional)
# Example:
#   python svm_simple.py --data testSet.txt --C 0.6 --toler 1e-3 --max-iter 40 \
#                        --save-plot linear_margin.png
# ------------------------------------------------------------

import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


def load_tsv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            a, b, c = ln.split()
            X.append([float(a), float(b)])
            y.append(float(c))
    return np.asarray(X, float), np.asarray(y, float)


def clip(a: float, L: float, H: float) -> float:
    return H if a > H else (L if a < L else a)


def smo_simple_linear(
    X: np.ndarray,
    y: np.ndarray,
    C: float = 0.6,
    toler: float = 1e-3,
    max_iter: int = 40,
    seed: int = 0,
):
    """Return (b, alphas) after simplified SMO for linear SVM."""
    rng = np.random.default_rng(seed)
    m, n = X.shape
    alphas = np.zeros(m, float)
    b = 0.0

    def f(i):
        # decision value for x_i
        return (alphas * y) @ (X @ X[i]) + b

    it = 0
    while it < max_iter:
        changed = 0
        for i in range(m):
            Ei = f(i) - y[i]
            if ((y[i] * Ei < -toler) and (alphas[i] < C)) or ((y[i] * Ei > toler) and (alphas[i] > 0)):
                # pick j != i at random
                j = i
                while j == i:
                    j = rng.integers(0, m)
                Ej = f(j) - y[j]

                ai_old, aj_old = alphas[i], alphas[j]
                # bounds
                if y[i] != y[j]:
                    L = max(0.0, aj_old - ai_old)
                    H = min(C, C + aj_old - ai_old)
                else:
                    L = max(0.0, ai_old + aj_old - C)
                    H = min(C, ai_old + aj_old)
                if L == H:
                    continue

                # eta
                eta = 2.0 * (X[i] @ X[j]) - (X[i] @ X[i]) - (X[j] @ X[j])
                if eta >= 0:
                    continue

                # update aj
                alphas[j] -= y[j] * (Ei - Ej) / eta
                alphas[j] = clip(alphas[j], L, H)
                if abs(alphas[j] - aj_old) < 1e-5:
                    continue

                # update ai
                alphas[i] += y[i] * y[j] * (aj_old - alphas[j])

                # update b
                b1 = b - Ei - y[i] * (alphas[i] - ai_old) * (X[i] @ X[i]) - y[j] * (alphas[j] - aj_old) * (X[i] @ X[j])
                b2 = b - Ej - y[i] * (alphas[i] - ai_old) * (X[i] @ X[j]) - y[j] * (alphas[j] - aj_old) * (X[j] @ X[j])
                if 0 < alphas[i] < C:
                    b = b1
                elif 0 < alphas[j] < C:
                    b = b2
                else:
                    b = 0.5 * (b1 + b2)

                changed += 1
        it = it + 1 if changed == 0 else 0
    return b, alphas


def compute_w(X: np.ndarray, y: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    return (alphas * y) @ X  # shape (2,)


def plot_linear_boundary(X, y, w, b, alphas, out_path=None):
    pos = y > 0
    neg = ~pos
    plt.figure(figsize=(6, 5))
    plt.scatter(X[pos, 0], X[pos, 1], s=30, label="+1")
    plt.scatter(X[neg, 0], X[neg, 1], s=30, label="-1")

    xmin, xmax = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    xs = np.linspace(xmin, xmax, 100)
    if abs(w[1]) < 1e-9:
        # vertical boundary
        x0 = -b / (w[0] + 1e-12)
        plt.axvline(x0, color="k")
    else:
        ys = (-(b + w[0] * xs) / w[1])
        plt.plot(xs, ys, "k-", label="decision f(x)=0")

    sv_mask = alphas > 1e-8
    plt.scatter(X[sv_mask, 0], X[sv_mask, 1], s=150, facecolors="none", edgecolors="r", linewidths=1.5, label="SVs")
    plt.legend()
    plt.title("Linear SVM (simplified SMO)")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=160)
        print(f"[saved] {out_path}")
    else:
        plt.show()
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="TSV file with x1, x2, label(±1)")
    ap.add_argument("--C", type=float, default=0.6)
    ap.add_argument("--toler", type=float, default=1e-3)
    ap.add_argument("--max-iter", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-plot", default="")
    args = ap.parse_args()

    X, y = load_tsv(args.data)
    b, alphas = smo_simple_linear(X, y, C=args.C, toler=args.toler, max_iter=args.max_iter, seed=args.seed)
    w = compute_w(X, y, alphas)
    preds = np.sign(X @ w + b)
    acc = (preds == y).mean()
    print(f"Train accuracy: {acc:.4f}  (C={args.C}, toler={args.toler}, max_iter={args.max_iter})")
    if args.save_plot:
        plot_linear_boundary(X, y, w, b, alphas, args.save_plot)


if __name__ == "__main__":
    main()
