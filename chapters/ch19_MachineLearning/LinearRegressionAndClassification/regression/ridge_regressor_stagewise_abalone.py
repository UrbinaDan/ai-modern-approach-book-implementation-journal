# ridge_stagewise_modern.py
# A modern NumPy implementation of:
# - Ridge (L2) regression coefficient paths across a lambda grid
# - Forward Stagewise linear regression (coefficient paths across iterations)
# Works great in Codespaces. Saves plots as PNGs.
from __future__ import annotations

import os
import argparse
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


# --------------------------- I/O ---------------------------

def load_xy(path: str, target_index: int = -1, delimiter: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a whitespace/CSV/TSV-like numeric file.
    By default, last column is y; all others are features X.
    """
    arr = np.loadtxt(path, delimiter=delimiter)       # float array
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    X = np.delete(arr, target_index, axis=1)
    y = arr[:, target_index]
    return X, y


# --------------------------- Scaling helpers ---------------------------

@dataclass
class ZScaler:
    mu_x: np.ndarray
    std_x: np.ndarray
    mu_y: float

    @staticmethod
    def fit(X: np.ndarray, y: np.ndarray) -> "ZScaler":
        mu_x = X.mean(axis=0)
        std_x = X.std(axis=0, ddof=0)
        std_x = np.where(std_x == 0.0, 1.0, std_x)  # avoid division by zero
        mu_y = float(y.mean())
        return ZScaler(mu_x, std_x, mu_y)

    def transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        Xs = (X - self.mu_x) / self.std_x
        ys = None if y is None else (y - self.mu_y)
        return Xs, ys

    def intercept_from_w(self, w_std: np.ndarray) -> float:
        # For standardized X and centered y, intercept in original space:
        # b = mu_y - sum_j w_std_j * (mu_x_j / std_x_j)
        return float(self.mu_y - np.sum(w_std * (self.mu_x / self.std_x)))


# --------------------------- Ridge regression ---------------------------

def ridge_solve(Xs: np.ndarray, yc: np.ndarray, lam: float) -> np.ndarray:
    """
    Solve (X^T X + λ I) w = X^T y  using a stable solver.
    Xs: standardized features, yc: centered target.
    Returns w (n_features,)
    """
    n = Xs.shape[1]
    XtX = Xs.T @ Xs
    A = XtX + lam * np.eye(n)
    b = Xs.T @ yc
    # Use solve (better than explicit inverse); fallback to pinv if singular
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(A) @ b
    return w  # shape (n,)


def ridge_path(X: np.ndarray, y: np.ndarray, lambdas: np.ndarray, scale: bool = True):
    """
    Compute ridge coefficients for each λ in `lambdas`.
    Returns (coefs, intercepts, scaler) where:
      coefs: shape (len(lambdas), n_features)
      intercepts: shape (len(lambdas),)
      scaler: ZScaler used (or identity-like if scale=False)
    """
    if scale:
        scaler = ZScaler.fit(X, y)
        Xs, yc = scaler.transform(X, y)
    else:
        scaler = ZScaler(mu_x=np.zeros(X.shape[1]), std_x=np.ones(X.shape[1]), mu_y=0.0)
        Xs, yc = X.copy(), y - y.mean()

    coefs = []
    intercepts = []
    for lam in lambdas:
        w_std = ridge_solve(Xs, yc, lam)
        b = scaler.intercept_from_w(w_std)
        coefs.append(w_std)         # note: these are on standardized scale
        intercepts.append(b)
    return np.vstack(coefs), np.asarray(intercepts), scaler


# --------------------------- Forward stagewise ---------------------------

def stagewise(
    X: np.ndarray,
    y: np.ndarray,
    eps: float = 0.01,
    iters: int = 1000,
    scale: bool = True,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, ZScaler]:
    """
    Simple forward stagewise linear regression (greedy coefficient path).
    Returns (W_hist, scaler)
      W_hist: (iters+1, n_features) coefficient history on standardized scale.
    """
    if scale:
        scaler = ZScaler.fit(X, y)
        Xs, yc = scaler.transform(X, y)
    else:
        scaler = ZScaler(mu_x=np.zeros(X.shape[1]), std_x=np.ones(X.shape[1]), mu_y=0.0)
        Xs, yc = X.copy(), y - y.mean()

    m, n = Xs.shape
    w = np.zeros(n)
    y_hat = Xs @ w  # keep running prediction to avoid recomputing full X @ w each candidate
    W_hist = np.zeros((iters + 1, n))
    W_hist[0] = w

    rng = np.random.default_rng(seed)
    feat_order = np.arange(n)

    for t in range(1, iters + 1):
        # (Optional) randomize feature order each iteration for tie-breaking variety
        rng.shuffle(feat_order)
        best_err = np.inf
        best_j = None
        best_sign = 0.0

        # Try nudging each coefficient up/down by eps and pick the best decrease in RSS
        for j in feat_order:
            # Up
            delta_up = eps
            y_try = y_hat + delta_up * Xs[:, j]
            err_up = np.sum((yc - y_try) ** 2)

            # Down
            delta_dn = -eps
            y_try2 = y_hat + delta_dn * Xs[:, j]
            err_dn = np.sum((yc - y_try2) ** 2)

            if err_up < best_err:
                best_err = err_up
                best_j = j
                best_sign = +1.0
            if err_dn < best_err:
                best_err = err_dn
                best_j = j
                best_sign = -1.0

        # Apply the best update
        if best_j is None:
            # No improvement found (should be rare); stop early
            W_hist = W_hist[:t]
            break

        w[best_j] += best_sign * eps
        y_hat += best_sign * eps * Xs[:, best_j]
        W_hist[t] = w

    return W_hist, scaler


# --------------------------- Plotting ---------------------------

def plot_ridge_paths(lambdas: np.ndarray, coefs: np.ndarray, save: Optional[str], labels: list[str] | None = None):
    """
    coefs: (n_lambdas, n_features) on standardized scale
    """
    plt.figure(figsize=(8, 5))
    for j in range(coefs.shape[1]):
        lab = labels[j] if labels is not None and j < len(labels) else f"w{j}"
        plt.plot(np.log10(lambdas), coefs[:, j], label=lab)
    plt.xlabel("log10(lambda)")
    plt.ylabel("Coefficient (standardized)")
    plt.title("Ridge coefficient paths")
    if labels is not None and len(labels) <= 12:
        plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=180)
        plt.close()
    else:
        plt.show()


def plot_stagewise_paths(W_hist: np.ndarray, save: Optional[str], labels: list[str] | None = None):
    """
    W_hist: (iters+1, n_features) on standardized scale
    """
    plt.figure(figsize=(8, 5))
    x = np.arange(W_hist.shape[0])
    for j in range(W_hist.shape[1]):
        lab = labels[j] if labels is not None and j < len(labels) else f"w{j}"
        plt.plot(x, W_hist[:, j], label=lab)
    plt.xlabel("Iteration")
    plt.ylabel("Coefficient (standardized)")
    plt.title("Forward Stagewise coefficient paths")
    if labels is not None and len(labels) <= 12:
        plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=180)
        plt.close()
    else:
        plt.show()


# --------------------------- CLI ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Ridge & Forward Stagewise Regression (NumPy only)")
    ap.add_argument("--data", required=True, help="path to numeric file; last column is y by default")
    ap.add_argument("--target-index", type=int, default=-1, help="which column is y (default: -1)")
    ap.add_argument("--delimiter", default=None, help="CSV delimiter (default: whitespace/auto)")

    # Ridge options
    ap.add_argument("--ridge", action="store_true", help="run ridge path")
    ap.add_argument("--ridge-lam-start", type=float, default=1e-4, help="start of logspace grid")
    ap.add_argument("--ridge-lam-end", type=float, default=1e+4, help="end of logspace grid")
    ap.add_argument("--ridge-lam-num", type=int, default=30, help="number of lambdas in grid")
    ap.add_argument("--ridge-out", default="", help="PNG path to save ridge plot")

    # Stagewise options
    ap.add_argument("--stagewise", action="store_true", help="run forward stagewise")
    ap.add_argument("--eps", type=float, default=0.01, help="coefficient step size")
    ap.add_argument("--iters", type=int, default=1000, help="iterations")
    ap.add_argument("--stagewise-out", default="", help="PNG path to save stagewise plot")

    # Common
    ap.add_argument("--no-scale", action="store_true", help="disable z-score scaling")
    ap.add_argument("--names", nargs="*", default=None, help="optional feature names")
    args = ap.parse_args()

    # Load
    X, y = load_xy(args.data, target_index=args.target_index, delimiter=args.delimiter)
    scale = not args.no_scale
    labels = args.names if args.names else None

    ran_any = False

    # Ridge
    if args.ridge:
        lambdas = np.logspace(np.log10(args.ridge_lam_start), np.log10(args.ridge_lam_end), num=args.ridge_lam_num)
        coefs, intercepts, scaler = ridge_path(X, y, lambdas=lambdas, scale=scale)
        print(f"[ridge] grid: {lambdas[0]:.4g} … {lambdas[-1]:.4g}  (n={len(lambdas)})")
        # Quick training RSS at a few points for sanity
        for lam in [lambdas[0], lambdas[len(lambdas)//2], lambdas[-1]]:
            w_std = ridge_solve(*ZScaler.fit(X, y).transform(X, y), lam=lam)  # on standardized
            y_pred = ((X - ZScaler.fit(X, y).mu_x) / ZScaler.fit(X, y).std_x) @ w_std + ZScaler.fit(X, y).mu_y
            rss = float(np.sum((y - y_pred) ** 2))
            print(f"  λ={lam:.4g} -> train RSS={rss:.4f}")
        plot_ridge_paths(lambdas, coefs, args.ridge_out or None, labels=labels)
        if args.ridge_out:
            print(f"[ridge] saved: {args.ridge_out}")
        ran_any = True

    # Stagewise
    if args.stagewise:
        W_hist, scaler = stagewise(X, y, eps=args.eps, iters=args.iters, scale=scale, seed=42)
        # Report final RSS
        w_final = W_hist[-1]
        y_pred = ((X - scaler.mu_x) / scaler.std_x) @ w_final + scaler.mu_y
        rss = float(np.sum((y - y_pred) ** 2))
        print(f"[stagewise] iters={W_hist.shape[0]-1}, final train RSS={rss:.4f}")
        plot_stagewise_paths(W_hist, args.stagewise_out or None, labels=labels)
        if args.stagewise_out:
            print(f"[stagewise] saved: {args.stagewise_out}")
        ran_any = True

    if not ran_any:
        print("Nothing to do. Add --ridge and/or --stagewise. See --help for options.")


if __name__ == "__main__":
    main()
