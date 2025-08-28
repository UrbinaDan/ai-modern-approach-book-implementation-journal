# svm_qp_from_scratch.py
# A small, modern SVM (C-SVC) solved via CVXOPT QP, with linear / poly / RBF kernels.
# No scikit-learn required.

from __future__ import annotations
import numpy as np
from numpy.linalg import norm
from dataclasses import dataclass
from typing import Literal, Tuple, Optional

import cvxopt
import cvxopt.solvers


KernelName = Literal["linear", "poly", "rbf"]


def _as_float64(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype="float64", order="C")


@dataclass
class SVMQP:
    """
    Soft-margin kernel SVM trained by solving the dual QP with CVXOPT.

    Parameters
    ----------
    C : float | None
        Soft-margin penalty. Use None for hard margin (may fail if data non-separable).
    kernel : {'linear','poly','rbf'}
    degree : int
        Degree for polynomial kernel.
    gamma : {'scale','auto'} | float
        - 'scale' -> 1 / (n_features * X.var()) on the training data
        - 'auto'  -> 1 / n_features
        - float   -> used directly
    coef0 : float
        Independent term in poly kernel.
    tol : float
        Numerical tolerance for support vector detection.
    solver_verbosity : bool
        If True, show CVXOPT solver progress.
    """
    C: Optional[float] = 1.0
    kernel: KernelName = "rbf"
    degree: int = 3
    gamma: float | str = "scale"
    coef0: float = 1.0
    tol: float = 1e-5
    solver_verbosity: bool = False

    # learned attributes
    X_: Optional[np.ndarray] = None
    y_: Optional[np.ndarray] = None  # labels in {-1,+1}
    alphas_: Optional[np.ndarray] = None
    b_: float = 0.0
    w_: Optional[np.ndarray] = None  # only for linear kernel
    support_idx_: Optional[np.ndarray] = None

    # ------------- kernel helpers -------------

    def _compute_gamma(self, X: np.ndarray) -> float:
        if isinstance(self.gamma, (float, int)):
            g = float(self.gamma)
            if g <= 0:
                raise ValueError("gamma must be > 0")
            return g
        n_features = X.shape[1]
        if self.gamma == "auto":
            return 1.0 / n_features
        if self.gamma == "scale":
            v = X.var()
            return 1.0 / (n_features * (v if v > 0 else 1.0))
        raise ValueError("gamma must be 'scale', 'auto', or a positive float")

    def _gram(self, X: np.ndarray) -> np.ndarray:
        """Compute Gram matrix K[i,j] = k(x_i, x_j)."""
        X = _as_float64(X)
        if self.kernel == "linear":
            return X @ X.T
        elif self.kernel == "poly":
            g = self._compute_gamma(X)
            return (g * (X @ X.T) + self.coef0) ** self.degree
        elif self.kernel == "rbf":
            # pairwise squared distances: ||x||^2 - 2 x·x' + ||x'||^2
            g = self._compute_gamma(X)
            sq_norms = np.sum(X * X, axis=1, keepdims=True)
            d2 = sq_norms - 2.0 * (X @ X.T) + sq_norms.T
            K = np.exp(-g * np.maximum(d2, 0.0))
            return K
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def _kernel_vec(self, X_train: np.ndarray, x: np.ndarray) -> np.ndarray:
        """k(X_train, x) for a single test sample x."""
        X = _as_float64(X_train)
        x = _as_float64(x)
        if self.kernel == "linear":
            return X @ x
        elif self.kernel == "poly":
            g = self._compute_gamma(X)
            return (g * (X @ x) + self.coef0) ** self.degree
        elif self.kernel == "rbf":
            g = self._compute_gamma(X)
            d2 = np.sum((X - x) ** 2, axis=1)
            return np.exp(-g * np.maximum(d2, 0.0))
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    # ------------- API -------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVMQP":
        """
        Train SVM on (X, y). y can be in {0,1} or {-1,+1}; it will be mapped to {-1,+1}.
        """
        X = _as_float64(X)
        y = _as_float64(y).ravel()

        # map labels to {-1, +1}
        y_unique = np.unique(y)
        if set(y_unique.tolist()) == {0.0, 1.0}:
            y = np.where(y > 0, 1.0, -1.0)
        elif set(y_unique.tolist()) == {-1.0, 1.0}:
            pass
        else:
            raise ValueError("y must contain only {0,1} or {-1,+1}")

        n = X.shape[0]
        K = self._gram(X)
        # small jitter on diagonal for numerical stability
        K.flat[:: n + 1] += 1e-12

        P = cvxopt.matrix(np.outer(y, y) * K, tc="d")
        q = cvxopt.matrix(-np.ones(n), tc="d")
        A = cvxopt.matrix(y[None, :], tc="d")
        b = cvxopt.matrix(0.0, tc="d")

        if self.C is None:  # hard margin
            G = cvxopt.matrix(-np.eye(n), tc="d")
            h = cvxopt.matrix(np.zeros(n), tc="d")
        else:
            C = float(self.C)
            G = cvxopt.matrix(np.vstack([-np.eye(n), np.eye(n)]), tc="d")
            h = cvxopt.matrix(np.hstack([np.zeros(n), np.full(n, C)]), tc="d")

        # solve QP
        cvxopt.solvers.options["show_progress"] = bool(self.solver_verbosity)
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        alphas = np.ravel(solution["x"])

        # support vectors: alpha > tol
        sv = alphas > self.tol
        self.support_idx_ = np.where(sv)[0]
        self.alphas_ = alphas[sv]
        self.X_ = X
        self.y_ = y
        Xsv = X[sv]
        ysv = y[sv]

        # bias: average over SV with 0 < alpha < C if possible
        if self.C is None:
            margin_sv = np.ones_like(self.alphas_, dtype=bool)
        else:
            margin_sv = (alphas[sv] > self.tol) & (alphas[sv] < self.C - self.tol)

        if not np.any(margin_sv):
            margin_sv = np.ones_like(self.alphas_, dtype=bool)  # fallback

        Ksv = self._gram(Xsv)
        # f(sv) = sum_j a_j y_j K(sv, sv_j) + b  →  b = y_i - f_i_no_b
        f_no_b = (self.alphas_ * ysv) @ Ksv.T
        self.b_ = float(np.mean(ysv[margin_sv] - f_no_b[margin_sv]))

        # explicit w for linear kernel
        if self.kernel == "linear":
            self.w_ = (self.alphas_ * ysv) @ Xsv
        else:
            self.w_ = None

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return raw signed scores f(x)."""
        if self.X_ is None:
            raise RuntimeError("Call fit() first.")
        X = _as_float64(X)
        if self.kernel == "linear" and self.w_ is not None:
            return X @ self.w_ + self.b_
        # kernel form
        Xsv = self.X_[self.support_idx_]
        ysv = self.y_[self.support_idx_]
        scores = np.empty(X.shape[0], dtype="float64")
        for i, x in enumerate(X):
            kx = self._kernel_vec(Xsv, x)  # shape (n_sv,)
            scores[i] = float(np.sum(self.alphas_ * ysv * kx) + self.b_)
        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return labels in {-1,+1}."""
        return np.where(self.decision_function(X) >= 0.0, 1.0, -1.0)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Accuracy for y in {0,1} or {-1,+1}."""
        y_true = np.asarray(y).ravel()
        y_pred = self.predict(X)
        if set(np.unique(y_true).tolist()) == {0, 1}:
            y_pred_out = np.where(y_pred > 0, 1, 0)
        else:
            y_pred_out = y_pred
        return float(np.mean(y_pred_out == y_true))


# ----------------------------- demo -----------------------------
if __name__ == "__main__":
    # Tiny non-linear toy dataset (two moons-ish)
    rng = np.random.default_rng(0)
    n = 120
    t = rng.random(n) * 2 * np.pi
    x1 = np.c_[np.cos(t), np.sin(t)] + 0.15 * rng.standard_normal((n, 2))
    x2 = np.c_[1 + np.cos(t), 1 - np.sin(t)] + 0.15 * rng.standard_normal((n, 2))
    X = np.vstack([x1, x2])
    y = np.hstack([-np.ones(n), +np.ones(n)])

    svm = SVMQP(C=1.0, kernel="rbf", gamma="scale", tol=1e-6, solver_verbosity=False)
    svm.fit(X, y)
    acc = svm.score(X, y)
    print(f"Train accuracy: {acc:.3f}  |  #SV = {len(svm.alphas_)}")
