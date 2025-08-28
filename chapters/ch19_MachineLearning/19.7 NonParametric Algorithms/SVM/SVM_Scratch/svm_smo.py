# svm_smo.py
# ------------------------------------------------------------
# Full SMO with KERNELS (linear or RBF) for 2-D toy datasets.
# - Precomputes Gram matrix, maintains error cache, heuristically picks j
# - Works with labels ±1
# - Plots decision regions + support vectors; evaluates train/test
# Example:
#   python svm_smo.py --train testSetRBF.txt --test testSetRBF2.txt \
#                     --kernel rbf --gamma 1.3 --C 200 --toler 1e-4 --max-iter 100 \
#                     --save-plot rbf_margin.png
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


def kernel_linear(X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    return X @ Z.T  # (n, m)


def kernel_rbf(X: np.ndarray, Z: np.ndarray, gamma: float) -> np.ndarray:
    # K(x,z) = exp(-||x-z||^2 / (gamma^2))  (matches your original scaling)
    X2 = (X**2).sum(axis=1, keepdims=True)
    Z2 = (Z**2).sum(axis=1, keepdims=True).T
    d2 = X2 + Z2 - 2 * (X @ Z.T)
    return np.exp(-d2 / (gamma**2 + 1e-12))


class SMO:
    def __init__(self, X, y, C=200.0, toler=1e-4, kernel="rbf", gamma=1.3, seed=0):
        self.X = X
        self.y = y
        self.C = float(C)
        self.tol = float(toler)
        self.m = X.shape[0]
        self.alphas = np.zeros(self.m, float)
        self.b = 0.0
        self.e = np.full(self.m, np.nan, float)  # error cache
        self.kernel = kernel
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)
        # Gram matrix
        if kernel == "linear":
            self.K = kernel_linear(X, X)
        elif kernel == "rbf":
            self.K = kernel_rbf(X, X, gamma)
        else:
            raise ValueError("kernel must be 'linear' or 'rbf'")

    def f_i(self, i: int) -> float:
        return (self.alphas * self.y) @ self.K[:, i] + self.b

    def E(self, i: int) -> float:
        if np.isnan(self.e[i]):
            self.e[i] = self.f_i(i) - self.y[i]
        return self.e[i]

    def update_E(self, i: int):
        self.e[i] = self.f_i(i) - self.y[i]

    def pick_j(self, i: int, Ei: float) -> Tuple[int, float]:
        # heuristic: choose j maximizing |Ei - Ej| among valid indices
        valid = np.where(~np.isnan(self.e))[0]
        if valid.size > 1:
            j = -1
            max_delta = -1.0
            Ej_best = 0.0
            for k in valid:
                if k == i:
                    continue
                Ek = self.E(k)
                delta = abs(Ei - Ek)
                if delta > max_delta:
                    max_delta = delta
                    j = k
                    Ej_best = Ek
            return j, Ej_best
        # fallback: random
        j = i
        while j == i:
            j = self.rng.integers(0, self.m)
        return j, self.E(j)

    def clip(self, a, L, H):
        return H if a > H else (L if a < L else a)

    def inner_loop(self, i: int) -> int:
        Ei = self.E(i)
        yi = self.y[i]
        ai_old = self.alphas[i]

        if (yi * Ei < -self.tol and ai_old < self.C) or (yi * Ei > self.tol and ai_old > 0):
            j, Ej = self.pick_j(i, Ei)
            yj = self.y[j]
            aj_old = self.alphas[j]

            if yi != yj:
                L = max(0.0, aj_old - ai_old)
                H = min(self.C, self.C + aj_old - ai_old)
            else:
                L = max(0.0, ai_old + aj_old - self.C)
                H = min(self.C, ai_old + aj_old)
            if L == H:
                return 0

            eta = 2.0 * self.K[i, j] - self.K[i, i] - self.K[j, j]
            if eta >= 0:
                return 0

            # update aj
            aj_new = aj_old - yj * (Ei - Ej) / eta
            aj_new = self.clip(aj_new, L, H)
            if abs(aj_new - aj_old) < 1e-5:
                return 0

            # update ai
            ai_new = ai_old + yi * yj * (aj_old - aj_new)

            # update threshold b
            b1 = self.b - Ei - yi * (ai_new - ai_old) * self.K[i, i] - yj * (aj_new - aj_old) * self.K[i, j]
            b2 = self.b - Ej - yi * (ai_new - ai_old) * self.K[i, j] - yj * (aj_new - aj_old) * self.K[j, j]
            if 0 < ai_new < self.C:
                b_new = b1
            elif 0 < aj_new < self.C:
                b_new = b2
            else:
                b_new = 0.5 * (b1 + b2)

            # commit
            self.alphas[i] = ai_new
            self.alphas[j] = aj_new
            self.b = b_new
            # update errors
            self.update_E(i)
            self.update_E(j)
            return 1
        return 0

    def fit(self, max_iter=100):
        it = 0
        examine_all = True
        changed = 0
        while it < max_iter and (changed > 0 or examine_all):
            changed = 0
            if examine_all:
                for i in range(self.m):
                    changed += self.inner_loop(i)
            else:
                # non-bound alphas
                idx = np.where((self.alphas > 0) & (self.alphas < self.C))[0]
                for i in idx:
                    changed += self.inner_loop(i)
            it += 1 if changed == 0 else 0
            examine_all = not examine_all if changed == 0 else examine_all
        return self

    def decision_function(self, X_eval: np.ndarray) -> np.ndarray:
        # use support vectors only
        sv_mask = self.alphas > 1e-8
        a = self.alphas[sv_mask]
        y = self.y[sv_mask]
        Xsv = self.X[sv_mask]
        if self.kernel == "linear":
            K = kernel_linear(X_eval, Xsv)
        else:
            K = kernel_rbf(X_eval, Xsv, self.gamma)
        return (K @ (a * y)) + self.b

    def predict(self, X_eval: np.ndarray) -> np.ndarray:
        return np.sign(self.decision_function(X_eval))


def plot_regions(model: SMO, X: np.ndarray, y: np.ndarray, out_path: str):
    xmin, xmax = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    ymin, ymax = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 300), np.linspace(ymin, ymax, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = model.predict(grid).reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, zz, levels=[-np.inf, 0, np.inf], alpha=0.2, colors=["#FFAAAA", "#AAAAFF"])
    pos = y > 0
    plt.scatter(X[pos, 0], X[pos, 1], s=25, label="+1")
    plt.scatter(X[~pos, 0], X[~pos, 1], s=25, label="-1")

    sv = model.alphas > 1e-8
    plt.scatter(X[sv, 0], X[sv, 1], s=120, facecolors="none", edgecolors="k", linewidths=1.2, label="SVs")
    plt.title(f"SVM ({model.kernel}, C={model.C}, gamma={model.gamma if model.kernel=='rbf' else 'n/a'})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    print(f"[saved] {out_path}")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="TSV train: x1 x2 y(±1)")
    ap.add_argument("--test", default="", help="TSV test: x1 x2 y(±1)")
    ap.add_argument("--kernel", choices=["linear", "rbf"], default="rbf")
    ap.add_argument("--gamma", type=float, default=1.3, help="RBF width parameter (matches your earlier scaling)")
    ap.add_argument("--C", type=float, default=200.0)
    ap.add_argument("--toler", type=float, default=1e-4)
    ap.add_argument("--max-iter", type=int, default=100)
    ap.add_argument("--save-plot", default="")
    args = ap.parse_args()

    Xtr, ytr = load_tsv(args.train)
    model = SMO(Xtr, ytr, C=args.C, toler=args.toler, kernel=args.kernel, gamma=args.gamma).fit(max_iter=args.max_iter)
    yhat_tr = model.predict(Xtr)
    print(f"Train error: {((yhat_tr != ytr).mean()*100):.2f}%  (SVs={int((model.alphas>1e-8).sum())})")

    if args.test:
        Xte, yte = load_tsv(args.test)
        yhat_te = model.predict(Xte)
        print(f"Test  error: {((yhat_te != yte).mean()*100):.2f}%")

    if args.save_plot:
        plot_regions(model, Xtr, ytr, args.save_plot)


if __name__ == "__main__":
    main()

