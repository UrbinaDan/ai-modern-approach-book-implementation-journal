# run_kernel_ridge.py
# A practical runner for kernel_ridge.KernelRidge
# - Loads a CSV/TSV (or whitespace) numeric dataset
# - Lets you choose kernel/alpha/gamma/degree
# - Optional standardization of X
# - Train/test split OR K-fold CV grid search for (alpha, gamma)
# - Reports MSE; can save y_pred to disk and a parity plot

from __future__ import annotations
import argparse, os, math
from typing import List, Tuple
import numpy as np

from kernel_ridge import KernelRidge


def load_array(
    path: str,
    sep: str | None = None,
    skip_rows: int = 0,
    usecols: List[int] | None = None,
) -> np.ndarray:
    """
    Load a numeric array from file.
    - sep=None -> whitespace
    - provide sep=',' for CSV or sep='\t' for TSV
    - skip_rows to skip a header
    - usecols to select columns
    """
    return np.loadtxt(path, delimiter=sep, skiprows=skip_rows, usecols=usecols)


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simple shuffle split."""
    n = len(y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = max(1, int(round(test_size * n)))
    te = idx[:n_test]
    tr = idx[n_test:]
    return X[tr], X[te], y[tr], y[te]


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.nanmean((y_true - y_pred) ** 2))


def parse_floats_list(s: str | None) -> List[float] | None:
    """Parse '0.1,1,10' -> [0.1, 1.0, 10.0]."""
    if not s:
        return None
    return [float(v) for v in s.split(",")]


def grid_search_cv(
    X: np.ndarray,
    y: np.ndarray,
    kernel: str,
    alphas: List[float],
    gammas: List[float] | List[str] | None,
    degree: int,
    scale_X: bool,
    n_splits: int,
    seed: int,
) -> Tuple[dict, float]:
    """Very small K-fold CV grid search; returns (best_params, best_score)."""
    # build folds
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.permutation(n)
    folds = np.array_split(idx, n_splits)

    best_params = None
    best_score = math.inf

    gammas_list = gammas or [None]
    for alpha in alphas:
        for gamma in gammas_list:
            fold_scores = []
            for k in range(n_splits):
                val_idx = folds[k]
                train_idx = np.concatenate([folds[j] for j in range(n_splits) if j != k])
                Xtr, ytr = X[train_idx], y[train_idx]
                Xva, yva = X[val_idx], y[val_idx]

                model = KernelRidge(
                    kernel=kernel,
                    alpha=alpha,
                    gamma=gamma,
                    degree=degree,
                    scale_X=scale_X,
                ).fit(Xtr, ytr)
                pred = model.predict(Xva)
                fold_scores.append(mse(yva, pred))
            score = float(np.mean(fold_scores))
            if score < best_score:
                best_score = score
                best_params = {"alpha": alpha, "gamma": gamma}

    return best_params, best_score


def maybe_save_preds(path: str | None, y_true: np.ndarray, y_pred: np.ndarray):
    if not path:
        return
    out = np.vstack([y_true, y_pred]).T
    np.savetxt(path, out, delimiter=",", header="y_true,y_pred", comments="")
    print(f"[saved] predictions → {path}")


def parity_plot(path: str | None, y_true: np.ndarray, y_pred: np.ndarray):
    if not path:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[warn] matplotlib not available; cannot save plot.")
        return
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, s=14, alpha=0.7)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title("Kernel Ridge: parity plot")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[saved] parity plot → {path}")


def main():
    ap = argparse.ArgumentParser(description="Run Kernel Ridge Regression on a numeric table.")
    ap.add_argument("--data", required=True, help="Path to data file (CSV/TSV/whitespace).")
    ap.add_argument("--sep", default=None, help="Delimiter: ',', '\\t', or leave empty for whitespace.")
    ap.add_argument("--skip-rows", type=int, default=0, help="Rows to skip (e.g., header).")
    ap.add_argument("--target-col", type=int, default=-1, help="Index of target column (default last).")
    ap.add_argument("--usecols", default="", help="Optional comma-separated feature column indices, else all but target.")
    ap.add_argument("--test-size", type=float, default=0.2, help="Fraction for the test split (default 0.2).")
    ap.add_argument("--seed", type=int, default=42)

    # Model hyperparams
    ap.add_argument("--kernel", choices=["linear", "poly", "rbf"], default="rbf")
    ap.add_argument("--alpha", type=float, default=1.0, help="Ridge strength (λ).")
    ap.add_argument("--gamma", default=None, help="Gamma for rbf/poly: float or 'scale' or empty for 1/n_features.")
    ap.add_argument("--degree", type=int, default=3, help="Degree for polynomial kernel.")
    ap.add_argument("--coef0", type=float, default=1.0, help="coef0 for polynomial kernel.")
    ap.add_argument("--scale-x", action="store_true", help="Z-score features before kernel (recommended for rbf/poly).")

    # Simple CV grid
    ap.add_argument("--cv", type=int, default=0, help="If >0, do K-fold CV to pick alpha/gamma.")
    ap.add_argument("--alphas", default="", help="CSV of alpha candidates (overrides --alpha if --cv>0).")
    ap.add_argument("--gammas", default="", help="CSV of gamma candidates (numbers) or include 'scale'.")

    # Outputs
    ap.add_argument("--save-preds", default="", help="CSV path to save [y_true,y_pred].")
    ap.add_argument("--save-parity", default="", help="PNG path for parity plot.")

    args = ap.parse_args()

    # Load whole table
    sep = None if (args.sep is None or args.sep == "") else args.sep
    table = load_array(args.data, sep=sep, skip_rows=args.skip_rows)

    # Slice into X, y
    cols = table.shape[1]
    tgt = args.target_col if args.target_col >= 0 else cols + args.target_col
    if tgt < 0 or tgt >= cols:
        raise ValueError(f"target_col={args.target_col} out of range for {cols} columns.")

    if args.usecols:
        usecols = [int(i) for i in args.usecols.split(",")]
        X = table[:, usecols]
    else:
        X = np.delete(table, tgt, axis=1)
    y = table[:, tgt]

    # Train/val split (or CV)
    if args.cv and args.cv > 1:
        alphas = parse_floats_list(args.alphas) or [args.alpha]
        # gamma values: allow number(s) and/or the word "scale"
        gammas: List[float | str] | None = None
        if args.gammas:
            gammas = []
            for g in args.gammas.split(","):
                g = g.strip()
                gammas.append("scale" if g == "scale" else float(g))
        best_params, best_score = grid_search_cv(
            X, y,
            kernel=args.kernel,
            alphas=alphas,
            gammas=gammas,
            degree=args.degree,
            scale_X=args.scale_x,
            n_splits=args.cv,
            seed=args.seed,
        )
        print(f"[cv] best params: {best_params}, mean MSE={best_score:.6f}")
        alpha = best_params["alpha"]
        gamma = best_params["gamma"]
    else:
        alpha = args.alpha
        gamma = args.gamma if args.gamma not in ("", None) else None

    # Final train/test fit
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=args.test_size, seed=args.seed)
    model = KernelRidge(
        kernel=args.kernel,
        alpha=alpha,
        gamma=gamma,
        degree=args.degree,
        coef0=args.coef0,
        scale_X=args.scale_x,
    ).fit(Xtr, ytr)

    yhat_tr = model.predict(Xtr)
    yhat_te = model.predict(Xte)
    print(f"Train MSE: {mse(ytr, yhat_tr):.6f}")
    print(f"Test  MSE: {mse(yte, yhat_te):.6f}")

    maybe_save_preds(args.save_preds or None, yte, yhat_te)
    parity_plot(args.save_parity or None, yte, yhat_te)


if __name__ == "__main__":
    main()

#Run example (on iris-slwc.txt):
# python LinearRegressionAndClassification/kernel_ridge_regression/run_kernel_ridge.py \
#   --data LinearRegressionAndClassification/kernel_ridge_regression/small_data/iris-slwc.txt \
#   --sep , --target-col -1 \
#   --kernel rbf --gamma scale --alpha 1.0 --scale-x \
#   --test-size 0.25 --seed 7 \
#   --save-parity krr_parity.png
