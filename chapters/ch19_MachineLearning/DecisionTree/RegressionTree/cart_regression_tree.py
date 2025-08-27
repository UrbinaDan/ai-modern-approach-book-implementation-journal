# cart_regression_tree.py
# A minimal, modern CART-style Regression Tree you can run in Codespaces.
from __future__ import annotations

import os
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple
import matplotlib
matplotlib.use("Agg")   # safe even if a display exists
import matplotlib.pyplot as plt

import numpy as np

# Matplotlib only needed if you pass --plot (useful for 1D toy data)
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def resolve(path: str) -> str:
    """Return an absolute path. Try CWD first, then the script's folder."""
    if not path:
        return path
    if os.path.isabs(path):
        return path
    if os.path.exists(path):
        return os.path.abspath(path)
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, path)


# ----------------------------- data utils -----------------------------

def load_xy_from_tsv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load whitespace/tab file with last column = y, others = X."""
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    X, y = arr[:, :-1], arr[:, -1]
    return X, y


def make_synthetic(n: int = 200, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Simple 1D piecewise-constant function with noise."""
    rng = np.random.default_rng(seed)
    x = np.sort(6 * rng.random(n) - 3)         # [-3, 3]
    y = np.where(x < -1, -2.0,
         np.where(x < 1,  0.5, 2.5)) + 0.4 * rng.standard_normal(n)
    return x.reshape(-1, 1), y


# ----------------------------- tree model -----------------------------

@dataclass
class Node:
    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left: "Node" | None = None
    right: "Node" | None = None
    value: Optional[float] = None
    n_samples: int = 0
    impurity: float = 0.0


class CARTRegressor:
    """CART-style regression tree (binary splits, variance/MSE reduction)."""

    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        random_state: Optional[int] = None,
    ):
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.min_impurity_decrease = float(min_impurity_decrease)
        self.random_state = random_state
        self.root: Optional[Node] = None

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: np.ndarray | None = None, y_val: np.ndarray | None = None) -> "CARTRegressor":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        self.root = self._build(X, y, depth=0)
        if X_val is not None and y_val is not None and X_val.size and y_val.size:
            self._prune_reduced_error(self.root, X_val, y_val)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_row(self.root, row) for row in X], dtype=float)

    # -------- internals --------

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        node = Node()
        node.n_samples = len(y)
        node.value = float(y.mean()) if node.n_samples > 0 else 0.0
        node.impurity = float(np.var(y)) if node.n_samples > 0 else 0.0

        # stop if pure / too small / too deep
        if (
            node.n_samples < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
            or np.allclose(y, y[0])
        ):
            return node

        best_feature, best_thr, best_sse = None, None, np.inf
        current_sse = float(((y - y.mean()) ** 2).sum())
        n_features = X.shape[1]

        for j in range(n_features):
            xj = X[:, j].astype(float)
            order = np.argsort(xj)
            xj_sorted = xj[order]
            y_sorted = y[order]

            uniq = np.unique(xj_sorted)
            if uniq.size <= 1:
                continue

            csum_y = np.cumsum(y_sorted)
            csum_y2 = np.cumsum(y_sorted ** 2)
            last_pos = np.searchsorted(xj_sorted, uniq, side="right") - 1

            for pos in last_pos[:-1]:
                nL = pos + 1
                nR = node.n_samples - nL
                if nL < self.min_samples_leaf or nR < self.min_samples_leaf:
                    continue

                sumL = csum_y[pos]; sumL2 = csum_y2[pos]
                sseL = float(sumL2 - (sumL * sumL) / nL)

                sumR = csum_y[-1] - sumL; sumR2 = csum_y2[-1] - sumL2
                sseR = float(sumR2 - (sumR * sumR) / nR)

                sse_total = sseL + sseR
                if sse_total < best_sse:
                    best_sse = sse_total
                    best_feature = j
                    best_thr = float((xj_sorted[pos] + xj_sorted[pos + 1]) / 2.0)

        if best_feature is None:
            return node

        if (current_sse - best_sse) < self.min_impurity_decrease:
            return node

        mask_left = X[:, best_feature] <= best_thr
        mask_right = ~mask_left
        if mask_left.sum() < self.min_samples_leaf or mask_right.sum() < self.min_samples_leaf:
            return node

        node.feature_index = int(best_feature)
        node.threshold = float(best_thr)
        node.left = self._build(X[mask_left], y[mask_left], depth + 1)
        node.right = self._build(X[mask_right], y[mask_right], depth + 1)
        return node

    def _predict_row(self, node: Node, row: np.ndarray) -> float:
        while node.feature_index is not None:
            node = node.left if row[node.feature_index] <= node.threshold else node.right
        return node.value

    def _prune_reduced_error(self, node: Node, Xv: np.ndarray, yv: np.ndarray) -> None:
        if node is None or node.feature_index is None:
            return
        mask_left = Xv[:, node.feature_index] <= node.threshold
        XvL, yvL = Xv[mask_left], yv[mask_left]
        XvR, yvR = Xv[~mask_left], yv[~mask_left]
        self._prune_reduced_error(node.left, XvL, yvL)
        self._prune_reduced_error(node.right, XvR, yvR)
        if node.left.feature_index is None and node.right.feature_index is None:
            err_no_merge = float(((yvL - node.left.value) ** 2).sum() +
                                 ((yvR - node.right.value) ** 2).sum())
            merged = node.value
            err_merge = float(((yv - merged) ** 2).sum())
            if err_merge <= err_no_merge:
                node.feature_index = None
                node.threshold = None
                node.left = None
                node.right = None
                node.value = merged


# ----------------------------- CLI -----------------------------

def main():
    ap = argparse.ArgumentParser(description="CART-style Regression Tree (NumPy only).")
    ap.add_argument("--train", default="", help="path to whitespace/TSV file (last column is y)")
    ap.add_argument("--test",  default="", help="optional path to validation/test file for pruning")
    ap.add_argument("--max-depth", type=int, default=None)
    ap.add_argument("--min-samples-split", type=int, default=2)
    ap.add_argument("--min-samples-leaf", type=int, default=1)
    ap.add_argument("--min-impurity-decrease", type=float, default=0.0)
    ap.add_argument("--plot", action="store_true", help="plot fit if data is 1D")
    ap.add_argument("--save", action="store_true", help="save plot instead of showing it")
    ap.add_argument("--out", default="cart_fit.png", help="output image path for the plot")
    args = ap.parse_args()

    # Resolve paths so you can run from any working directory
    train_path = resolve(args.train) if args.train else ""
    test_path  = resolve(args.test)  if args.test  else ""

    # Load data (or synthetic fallback)
    if train_path and os.path.exists(train_path):
        X_train, y_train = load_xy_from_tsv(train_path)
    else:
        print(f"[info] No train file given or not found ({args.train}); using synthetic 1D data.")
        X_train, y_train = make_synthetic()

    X_val, y_val = (load_xy_from_tsv(test_path) if (test_path and os.path.exists(test_path)) else (None, None))

    # Train (+ optional pruning)
    model = CARTRegressor(
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        min_impurity_decrease=args.min_impurity_decrease,
        random_state=42,
    ).fit(X_train, y_train, X_val=X_val, y_val=y_val)

    # Metrics
    yhat_tr = model.predict(X_train)
    mse_tr = float(((y_train - yhat_tr) ** 2).mean())
    print(f"Train MSE: {mse_tr:.4f}")
    if X_val is not None:
        yhat_va = model.predict(X_val)
        mse_va = float(((y_val - yhat_va) ** 2).mean())
        print(f"Val MSE:   {mse_va:.4f}")

    # Optional 1D plot
    if args.plot and plt is not None and X_train.shape[1] == 1:
        xmin = float(X_train[:, 0].min()) - 0.25
        xmax = float(X_train[:, 0].max()) + 0.25
        xs = np.linspace(xmin, xmax, 600).reshape(-1, 1)
        ys = model.predict(xs)
        plt.figure(figsize=(7, 4))
        plt.scatter(X_train[:, 0], y_train, s=16, alpha=0.7, label="train")
        plt.plot(xs[:, 0], ys, lw=2, label="tree prediction")
        if X_val is not None:
            plt.scatter(X_val[:, 0], y_val, s=16, alpha=0.7, label="val")
        plt.title("CART Regression Tree (piecewise-constant fit)")
        plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.tight_layout(); 
        if args.save:
            plt.savefig(args.out, dpi=180, bbox_inches="tight")
            print(f"Saved plot to {args.out}")
        else:
            plt.show()



if __name__ == "__main__":
    main()
    #Run this
    # python DecisionTree/RegressionTree/cart_regression_tree.py   --train DecisionTree/RegressionTree/ex2.txt   --test  DecisionTree/RegressionTree/ex2test.txt   --max-depth 5 --min-samples-leaf 5 --plot


    # Or run this 
    #python DecisionTree/RegressionTree/cart_regression_tree.py \
#   --train DecisionTree/RegressionTree/ex2.txt \
#   --test  DecisionTree/RegressionTree/ex2test.txt \
#   --max-depth 5 --min-samples-leaf 5 \
#   --plot --save --out DecisionTree/RegressionTree/cart_fit.png
