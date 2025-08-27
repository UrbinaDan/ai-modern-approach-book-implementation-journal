# lwlr_ex0_demo.py
# Modern, commented LWLR demo for the classic ex0.txt dataset.
# Works in GitHub Codespaces (NumPy + Matplotlib only).

from __future__ import annotations
import os
import argparse
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt


# ---------- I/O helpers ----------

def resolve_here(path: str) -> str:
    """Resolve a path relative to this script’s folder if not absolute."""
    if os.path.isabs(path):
        return path
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, path)

def load_xy(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a whitespace/tab-separated file where *last* column is y and all
    previous columns are X. Returns X (m,n), y (m,).
    """
    path = resolve_here(path)
    arr = np.loadtxt(path, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    X, y = arr[:, :-1], arr[:, -1]
    return X, y

def maybe_drop_bias(X: np.ndarray, drop_bias: bool = True) -> np.ndarray:
    """
    If the first column looks like a constant ~1.0, drop it.
    Many versions of ex0.txt include a leading intercept column of 1.0.
    """
    if not drop_bias or X.shape[1] == 0:
        return X
    col = X[:, 0]
    if np.std(col) < 1e-9 and 0.8 <= np.mean(col) <= 1.2:
        return X[:, 1:]
    return X


# ---------- Math: LWLR & OLS ----------

def lwlr_predict(
    x_query: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    k: float = 1.0,
    ridge: float = 1e-8,
) -> float:
    """
    Predict y(x_query) by solving a weighted least squares centered at x_query.

    Weights: w_j = exp(-||x_j - x_query||^2 / (2 k^2))
    Solve:   (X^T W X + λI) w = X^T W y      (ridge λ stabilizes the solve)
    Return:  x_query^T w
    """
    # Row-wise squared distances to the query point
    diffs = X - x_query  # (m,n)
    # Avoid k=0 blow-up; treat very tiny k as tiny-but-not-zero
    k = max(float(k), 1e-12)
    w = np.exp(-np.sum(diffs * diffs, axis=1) / (2.0 * k * k))  # (m,)

    # Build weighted normal equations
    # A = X^T W X + λI ,  b = X^T W y
    Xw = X * w[:, None]                  # scale rows of X by weights
    A = X.T @ Xw + ridge * np.eye(X.shape[1])
    b = X.T @ (w * y)

    # Solve for local linear parameters and return prediction at x_query
    beta = np.linalg.solve(A, b)         # (n,)
    return float(x_query @ beta)


def lwlr_curve(
    X: np.ndarray, y: np.ndarray, k: float, ridge: float,
    x_grid: np.ndarray
) -> np.ndarray:
    """Vectorized loop over query points in x_grid (shape (m, n))."""
    preds = np.empty(len(x_grid), dtype=float)
    for i, xq in enumerate(x_grid):
        preds[i] = lwlr_predict(xq, X, y, k=k, ridge=ridge)
    return preds


def ols_fit(X: np.ndarray, y: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    """
    Global linear regression via (X^T X + λI)^{-1} X^T y.
    Returns coefficients beta (n,).
    """
    A = X.T @ X + ridge * np.eye(X.shape[1])
    b = X.T @ y
    return np.linalg.solve(A, b)


# ---------- Plotting ----------

def plot_three_k(
    X: np.ndarray,
    y: np.ndarray,
    ks: List[float],
    ridge: float,
    overlay_ols: bool,
    save: str | None = None,
):
    """
    Make a 3-row figure:
      - blue scatter of (x, y)   (assumes X is 1D for visualization)
      - red LWLR curve for each k
      - optional dashed global OLS line
    """
    if X.shape[1] != 1:
        raise ValueError("This demo plots only for 1D X. Drop bias/extra cols first.")

    # Sort by x for a nice monotone curve
    order = np.argsort(X[:, 0])
    Xs, ys = X[order], y[order]

    # Dense grid over the domain for smooth curves
    xmin = float(Xs[:, 0].min()) - 0.05
    xmax = float(Xs[:, 0].max()) + 0.05
    x_grid = np.linspace(xmin, xmax, 400).reshape(-1, 1)

    # Optional global OLS line
    beta = ols_fit(X, y) if overlay_ols else None
    if beta is not None:
        ols_line = x_grid @ beta

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(9, 10), sharex=True)
    fig.suptitle("LWLR on ex0.txt (modern demo)", y=0.98)

    for ax, k in zip(axes, ks):
        # Compute LWLR curve for this bandwidth
        y_curve = lwlr_curve(X, y, k=k, ridge=ridge, x_grid=x_grid)

        # Scatter the raw data
        ax.scatter(Xs[:, 0], ys, s=20, alpha=0.7, label="data")

        # Plot LWLR curve
        ax.plot(x_grid[:, 0], y_curve, lw=2, color="tab:red", label=f"LWLR (k={k:g})")

        # Optional OLS overlay
        if beta is not None:
            ax.plot(x_grid[:, 0], ols_line, lw=1.5, ls="--", color="tab:gray", label="OLS")

        ax.set_ylabel("y")
        ax.legend(loc="best")

    axes[-1].set_xlabel("x")
    fig.tight_layout()

    if save:
        os.makedirs(os.path.dirname(save) or ".", exist_ok=True)
        fig.savefig(save, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot to {save}")
    else:
        plt.show()


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Modern LWLR demo for ex0.txt")
    ap.add_argument("--data", default="ex0.txt", help="path to ex0.txt (whitespace/TSV)")
    ap.add_argument("--k", type=float, nargs="+", default=[1.0, 0.01, 0.003],
                    help="bandwidth(s) to plot")
    ap.add_argument("--ridge", type=float, default=1e-8,
                    help="ridge λ for numeric stability")
    ap.add_argument("--keep-bias", action="store_true",
                    help="do NOT drop a leading constant-1.0 column")
    ap.add_argument("--ols", action="store_true",
                    help="overlay global OLS line for reference")
    ap.add_argument("--save", default="", help="PNG path to save figure (else show)")
    args = ap.parse_args()

    # Load and optionally drop a constant intercept column
    X, y = load_xy(args.data)
    X = maybe_drop_bias(X, drop_bias=not args.keep_bias)

    if X.shape[1] != 1:
        # You can still run, but this plotting demo is only for 1D display.
        raise SystemExit(
            f"Expected a single feature for plotting; got X shape {X.shape}. "
            "If your file has a leading 1.0 column, omit it (default) or pass --keep-bias to keep it."
        )

    # Make figure with three bandwidths
    save = args.save.strip() or None
    plot_three_k(X, y, ks=args.k, ridge=args.ridge, overlay_ols=args.ols, save=save)


if __name__ == "__main__":
    main()

#Run
#python LinearRegressionAndClassification/regression/lwlr_ex0_old_regression.py --data ex0.txt --k 1.0 0.01 0.003 --ols --save lwlr_ex0.png
