# lwlr_modern.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, os
from dataclasses import dataclass
from typing import Iterable, Tuple, List

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# ------------------------------ I/O ------------------------------

def load_xy(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a whitespace/tab-separated file where the *last* column is y and the
    remaining columns are X.
    """
    arr = np.loadtxt(path)
    if arr.ndim == 1:  # single row
        arr = arr.reshape(1, -1)
    return arr[:, :-1].astype(float), arr[:, -1].astype(float)


def train_test_split_rows(X: np.ndarray, y: np.ndarray, n_train: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Take the first n_train rows for train; the rest for test."""
    n_train = min(n_train, len(X))
    return X[:n_train], X[n_train:], y[:n_train], y[n_train:]


# ------------------------------ math ------------------------------

@dataclass
class LWLRConfig:
    k: float = 1.0           # Gaussian bandwidth
    ridge: float = 1e-8      # L2 for stability (λ)
    add_bias: bool = True    # add intercept column
    scale: bool = False      # z-score features using train stats


@dataclass
class LWLRModel:
    X: np.ndarray            # (m, d) training features (possibly scaled)
    y: np.ndarray            # (m,)   training targets
    mean: np.ndarray | None  # per-feature mean if scaled
    std: np.ndarray | None   # per-feature std if scaled
    config: LWLRConfig

    def _prep_X(self, X: np.ndarray) -> np.ndarray:
        Z = np.asarray(X, dtype=float)
        if self.config.scale:
            Z = (Z - self.mean) / self.std
        if self.config.add_bias:
            Z = np.hstack([np.ones((Z.shape[0], 1)), Z])
        return Z

    def predict_point(self, x: np.ndarray) -> float:
        """
        Predict y for a single point x by solving a weighted least-squares
        system using Gaussian weights centered at x.
        """
        Xtr = self.X
        ytr = self.y
        xi = x[None, :]  # (1, d)

        # Gaussian weights w_j = exp(-||x - x_j||^2 / (2k^2))
        diff = Xtr - xi
        # squared Euclidean distance per row
        d2 = np.sum(diff * diff, axis=1)
        k = float(self.config.k)
        w = np.exp(-d2 / (2.0 * k * k))  # (m,)

        # Weighted normal equations: (X^T W X + λI) β = X^T W y
        # Implement using broadcasting (avoid constructing W explicitly)
        XtWX = Xtr.T @ (w[:, None] * Xtr)
        XtWy = Xtr.T @ (w * ytr)

        # Ridge (do not penalize bias term if present)
        if self.config.ridge > 0.0:
            reg = np.eye(Xtr.shape[1]) * self.config.ridge
            if self.config.add_bias:
                reg[0, 0] = 0.0
            XtWX = XtWX + reg

        # Solve for β and return x * β
        try:
            beta = np.linalg.solve(XtWX, XtWy)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(XtWX, XtWy, rcond=None)[0]

        return float(x @ beta)

    def predict(self, X: np.ndarray) -> np.ndarray:
        Z = self._prep_X(X)
        preds = np.empty(Z.shape[0], dtype=float)
        for i in range(Z.shape[0]):
            preds[i] = self.predict_point(Z[i])
        return preds


def fit_lwlr(X: np.ndarray, y: np.ndarray, cfg: LWLRConfig) -> LWLRModel:
    """Prepare/train artifacts for LWLR (mainly scaling & bias)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()

    mean = std = None
    Z = X
    if cfg.scale:
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True)
        std[std == 0.0] = 1.0  # avoid divide-by-zero for constant cols
        Z = (X - mean) / std
    if cfg.add_bias:
        Z = np.hstack([np.ones((Z.shape[0], 1)), Z])

    return LWLRModel(Z, y, mean, std, cfg)


# ------------------------------ metrics & plotting ------------------------------

def rss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sum((y_true - y_pred) ** 2))


def maybe_plot_1d(Xtr, ytr, Xte, yte, yhat_grid, xgrid, title: str, save: str | None):
    """Plot only if 1D features and matplotlib is available."""
    if plt is None or Xtr.shape[1] != 1:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(Xtr[:, 0], ytr, s=16, alpha=0.7, label="train")
    if Xte is not None and len(Xte):
        ax.scatter(Xte[:, 0], yte, s=16, alpha=0.7, label="test")
    ax.plot(xgrid, yhat_grid, lw=2, label="LWLR fit")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=180, bbox_inches="tight")
    else:
        plt.show()

def _ensure_headless_if_saving(path: str):
    if path:
        import matplotlib
        matplotlib.use("Agg")  # headless backend for Codespaces/servers

def save_diagnostics(y_true, y_pred, title: str, outpath: str):
    """Parity + residuals diagnostic plot; works for any #features."""
    _ensure_headless_if_saving(outpath)
    import matplotlib.pyplot as plt
    import numpy as np

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    resid  = y_true - y_pred

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    # Parity (y_true vs y_pred)
    axs[0].scatter(y_true, y_pred, s=10, alpha=0.7)
    mn = float(min(y_true.min(), y_pred.min()))
    mx = float(max(y_true.max(), y_pred.max()))
    axs[0].plot([mn, mx], [mn, mx], lw=2)  # y=x line
    axs[0].set_xlabel("true")
    axs[0].set_ylabel("pred")
    axs[0].set_title("Parity")

    # Residuals histogram
    axs[1].hist(resid, bins=30, alpha=0.85)
    axs[1].set_title("Residuals")
    axs[1].set_xlabel("y_true - y_pred")

    fig.suptitle(title)
    fig.tight_layout()
    if outpath:
        os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
        fig.savefig(outpath, dpi=160)
        plt.close(fig)
        print(f"[saved] {outpath}")
    else:
        plt.show()


# ------------------------------ CLI ------------------------------

def main():
    ap = argparse.ArgumentParser(description="Modern LWLR (Locally Weighted Linear Regression)")
    ap.add_argument("--data", required=True, help="path to TSV file: columns [X..., y]")
    ap.add_argument("--train-rows", type=int, default=99, help="first N rows used for training")
    ap.add_argument("--k", type=float, nargs="+", default=[0.1, 1.0, 10.0], help="bandwidth(s)")
    ap.add_argument("--ridge", type=float, default=1e-8, help="ridge λ to stabilize solves")
    ap.add_argument("--no-bias", action="store_true", help="do NOT add an intercept column")
    ap.add_argument("--scale", action="store_true", help="z-score features using train stats")
    ap.add_argument("--plot", action="store_true", help="plot 1D fit (if X has one column)")
    ap.add_argument("--save-plot", default="", help="path to save plot (PNG); show if empty")
    args = ap.parse_args()

    # 1) Load and split
    X, y = load_xy(args.data)
    Xtr, Xte, ytr, yte = train_test_split_rows(X, y, args.train_rows)

    # 2) Evaluate several bandwidths
    for k in args.k:
        cfg = LWLRConfig(k=k, ridge=args.ridge, add_bias=not args.no_bias, scale=args.scale)
        model = fit_lwlr(Xtr, ytr, cfg)
        yhat_tr = model.predict(Xtr)
        tr_rss = rss(ytr, yhat_tr)
        msg = f"[k={k:g}] Train RSS = {tr_rss:.4f}"
        if len(Xte):
            yhat_te = model.predict(Xte)
            te_rss = rss(yte, yhat_te)
            msg += f" | Test RSS = {te_rss:.4f}"
        print(msg)

        # 3) Visualization
        if args.plot:
            save_base = (args.save_plot or "").strip()
            # make a safe suffix for the bandwidth value
            kstr = str(k).replace(".", "_")

            if X.shape[1] == 1:
                # dense grid over full x-range
                xmin = float(X[:, 0].min()) - 0.05
                xmax = float(X[:, 0].max()) + 0.05
                xgrid = np.linspace(xmin, xmax, 400).reshape(-1, 1)
                ygrid = model.predict(xgrid)
                title = f"LWLR fit (k={k:g}, ridge={args.ridge:g}, scale={args.scale})"

                # build output path if saving
                out = ""
                if save_base:
                    root, ext = os.path.splitext(save_base)
                    out = f"{root}_k{kstr}{ext or '.png'}"

                _ensure_headless_if_saving(out)
                maybe_plot_1d(Xtr, ytr, Xte, yte, ygrid, xgrid[:, 0], title, out)

            else:
                # Multivariate: save diagnostics (parity + residuals)
                title = f"LWLR diagnostics (k={k:g}, ridge={args.ridge:g}, scale={args.scale})"
                if len(Xte):
                    out = ""
                    if save_base:
                        root, ext = os.path.splitext(save_base)
                        out = f"{root}_k{kstr}_diag{ext or '.png'}"
                    save_diagnostics(yte, yhat_te, title, out)
                else:
                    out = ""
                    if save_base:
                        root, ext = os.path.splitext(save_base)
                        out = f"{root}_k{kstr}_diag_train{ext or '.png'}"
                    save_diagnostics(ytr, yhat_tr, title + " (train)", out)



if __name__ == "__main__":
    main()

#Run examples:
#python LinearRegressionAndClassification/regression/lwlr_regression.py --data LinearRegressionAndClassification/regression/abalone.txt --train-rows 99 --k 0.1 1 10 --plot


# # Same split style as the original (first 99 train, next 100 test),
# # and test three bandwidths:
# python lwlr_regression.py --data abalone.txt --train-rows 99 --k 0.1 1 10

# # With standardization and a bit of ridge to avoid singular matrices:
# python lwlr_regression.py --data abalone.txt --train-rows 99 --k 0.5 1 2 --scale --ridge 1e-4

# # If your X is 1-D, show/save a plot of the fitted curve:
# python lwlr_regression.py --data some_1d.tsv --plot --save-plot lwlr_fit.png
