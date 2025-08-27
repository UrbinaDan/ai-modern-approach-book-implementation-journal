# kernel_ridge.py
# A tiny, modern Kernel Ridge Regression (KRR) with vectorized kernels.
# - Kernels: linear, polynomial, rbf
# - Stable solve (no explicit matrix inverse)
# - Optional standardization of X for kernels that benefit from it (rbf/poly)

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal, Tuple
import numpy as np


KernelName = Literal["linear", "poly", "rbf"]


@dataclass
class _Scaler:
    """Simple feature scaler (z-score). Used when scale_X=True."""
    mean_: np.ndarray
    std_: np.ndarray

    @staticmethod
    def fit(X: np.ndarray) -> "_Scaler":
        mean = X.mean(axis=0)
        std = X.std(axis=0, ddof=0)
        std = np.where(std == 0, 1.0, std)
        return _Scaler(mean, std)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_


class KernelRidge:
    """
    Kernel Ridge Regression: solve (K + alpha I) alpha_vec = y
    prediction: y_hat(x) = sum_i alpha_vec[i] * k(x, x_i)

    Parameters
    ----------
    kernel : {"linear","poly","rbf"}
    alpha  : float, L2 regularization strength (λ). Larger -> smoother.
    gamma  : float or "scale" or None
             * For "rbf": if None -> 1 / n_features; if "scale" -> 1 / (n_features * X.var()).
             * For "poly"/"linear": if None -> 1 / n_features (like sklearn).
    degree : int, degree for the polynomial kernel
    coef0  : float, bias term for polynomial kernel
    scale_X: bool, if True z-score features before building the kernel (recommended for rbf/poly).
    """

    def __init__(
        self,
        kernel: KernelName = "rbf",
        alpha: float = 1.0,
        gamma: Optional[float | str] = None,
        degree: int = 3,
        coef0: float = 1.0,
        scale_X: bool = True,
    ):
        self.kernel = kernel
        self.alpha = float(alpha)
        self.gamma = gamma
        self.degree = int(degree)
        self.coef0 = float(coef0)
        self.scale_X = bool(scale_X)

        # learned on fit
        self.X_fit_: Optional[np.ndarray] = None
        self.alpha_vec_: Optional[np.ndarray] = None
        self._scaler: Optional[_Scaler] = None
        self.n_features_in_: Optional[int] = None

    # ---------------- Kernel builders (vectorized) ----------------

    @staticmethod
    def _linear(X1: np.ndarray, X2: np.ndarray, gamma: float, coef0: float, degree: int) -> np.ndarray:
        # Linear kernel: <x, z>
        return X1 @ X2.T

    @staticmethod
    def _poly(X1: np.ndarray, X2: np.ndarray, gamma: float, coef0: float, degree: int) -> np.ndarray:
        # Polynomial kernel: (gamma * <x, z> + coef0) ** degree
        return (gamma * (X1 @ X2.T) + coef0) ** degree

    @staticmethod
    def _rbf(X1: np.ndarray, X2: np.ndarray, gamma: float, coef0: float, degree: int) -> np.ndarray:
        # RBF (Gaussian) kernel: exp(-gamma * ||x - z||^2)
        # Vectorized squared distances
        X1_sq = np.sum(X1 * X1, axis=1, keepdims=True)           # (n1, 1)
        X2_sq = np.sum(X2 * X2, axis=1, keepdims=True).T         # (1, n2)
        d2 = X1_sq + X2_sq - 2.0 * (X1 @ X2.T)                   # (n1, n2)
        return np.exp(-gamma * d2)

    def _resolve_gamma(self, X: np.ndarray) -> float:
        if isinstance(self.gamma, str):
            if self.gamma == "scale":
                # mirror sklearn: 1 / (n_features * X.var())
                g = 1.0 / (X.shape[1] * X.var())
                return float(g) if np.isfinite(g) and g > 0 else 1.0 / X.shape[1]
            raise ValueError(f"Unknown gamma='{self.gamma}'")
        if self.gamma is None:
            return 1.0 / X.shape[1]
        return float(self.gamma)

    def _K(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        if self.kernel == "linear":
            return self._linear(X1, X2, 0.0, self.coef0, self.degree)
        if self.kernel == "poly":
            g = self._resolve_gamma(self.X_fit_ if self.X_fit_ is not None else X1)
            return self._poly(X1, X2, g, self.coef0, self.degree)
        if self.kernel == "rbf":
            g = self._resolve_gamma(self.X_fit_ if self.X_fit_ is not None else X1)
            return self._rbf(X1, X2, g, self.coef0, self.degree)
        raise ValueError(f"Unknown kernel={self.kernel}")

    # ---------------- Public API ----------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KernelRidge":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if self.scale_X:
            self._scaler = _Scaler.fit(X)
            Xs = self._scaler.transform(X)
        else:
            self._scaler = None
            Xs = X

        self.X_fit_ = Xs
        self.n_features_in_ = X.shape[1]

        K = self._K(self.X_fit_, self.X_fit_)              # (n, n)
        # Solve (K + alpha I) alpha_vec = y  (stable, no explicit inverse)
        n = K.shape[0]
        A = K + self.alpha * np.eye(n, dtype=float)
        self.alpha_vec_ = np.linalg.solve(A, y)            # (n,)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.X_fit_ is None or self.alpha_vec_ is None:
            raise RuntimeError("Call fit() before predict().")
        X = np.asarray(X, dtype=float)
        Xs = self._scaler.transform(X) if self._scaler else X
        K = self._K(Xs, self.X_fit_)                       # (n_test, n_train)
        return K @ self.alpha_vec_                         # (n_test,)
