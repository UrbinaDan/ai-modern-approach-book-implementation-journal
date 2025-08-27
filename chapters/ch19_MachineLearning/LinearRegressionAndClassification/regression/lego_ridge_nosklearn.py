# -*- coding: utf-8 -*-
"""
NumPy-only LEGO ridge regression:
- Scrapes sold listings from saved HTML pages (eBay-like).
- Builds X = [year, piece_count, is_new, msrp], y = sold_price.
- Standardizes X (z-score), centers y, fits ridge via normal equations.
- Selects lambda via k-fold CV, prints final equation in original units.
- Optional coefficient-path plot across log-spaced lambdas.

Examples
--------
# Real HTML pages
python lego_ridge_nosklearn.py \
  --html lego/lego8288.html  2006 800  49.99 \
  --html lego/lego10030.html 2002 3096 269.99 \
  --html lego/lego10179.html 2007 5195 499.99 \
  --html lego/lego10181.html 2007 3428 199.99 \
  --html lego/lego10189.html 2008 5922 299.99 \
  --html lego/lego10196.html 2009 3263 249.99 \
  --cv-folds 10 --lam-start 1e-4 --lam-end 1e4 --lam-num 30 \
  --coef-plot lego_coeff_paths.png

# Synthetic demo (no HTML needed)
python lego_ridge_nosklearn.py --demo --coef-plot demo_coeffs.png
"""
from __future__ import annotations

import os, re, argparse, math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from bs4 import BeautifulSoup  # pip install beautifulsoup4

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# ----------------------------- path resolver -----------------------------

def resolve(path: str) -> str:
    """
    Return an absolute path to `path`, trying:
      1) current working directory,
      2) this script's directory,
      3) the parent of this script's directory (common repo layout).
    Always returns an absolute path (even if missing), so error messages are clear.
    """
    path = os.path.expanduser(path)

    # already absolute?
    if os.path.isabs(path):
        return path

    # 1) relative to CWD
    cand = os.path.abspath(path)
    if os.path.exists(cand):
        return cand

    # 2) relative to this script's folder
    here = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.join(here, path)
    if os.path.exists(cand):
        return cand

    # 3) relative to parent of script folder
    cand = os.path.join(os.path.dirname(here), path)
    if os.path.exists(cand):
        return cand

    # fallback absolute (may not exist)
    return os.path.abspath(path)


# ----------------------------- scraping -----------------------------

_cmoney = re.compile(r"[\$,\s]")

def _parse_price(text: str) -> Optional[float]:
    """Extract a float price from text like '$1,234.56 Free shipping'."""
    if not text:
        return None
    txt = _cmoney.sub("", text)
    m = re.search(r"(\d+(\.\d+)?)", txt)
    return float(m.group(1)) if m else None

def _is_new(title: str) -> float:
    """Return 1.0 if title hints the item is new/NISB, else 0.0."""
    t = (title or "").lower()
    return 1.0 if (" new " in f" {t} " or "nisb" in t) else 0.0

def scrape_file(path: str, year: int, pieces: int, msrp: float,
                min_frac: float = 0.5) -> Tuple[List[List[float]], List[float]]:
    """
    Parse one saved HTML page. Try the classic table[r=i] layout; if not found,
    fall back to scanning all tables for rows that look 'sold' with a $price.
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    X_rows: List[List[float]] = []
    y_rows: List[float] = []

    def maybe_add(title: str, price_text: str):
        price = _parse_price(price_text)
        if price is None:
            return
        if price <= msrp * min_frac:  # drop incomplete/very low sales
            return
        X_rows.append([float(year), float(pieces), _is_new(title), float(msrp)])
        y_rows.append(float(price))

    # 1) Try the old "table r=i" pattern
    found_any = False
    i = 1
    while True:
        tbl = soup.find("table", attrs={"r": str(i)})
        if not tbl:
            break
        found_any = True
        try:
            links = tbl.find_all("a")
            title = links[1].get_text(strip=True) if len(links) > 1 else ""
            tds = tbl.find_all("td")
            sold_cell = tds[3] if len(tds) > 4 else None
            sold = sold_cell is not None and sold_cell.find("span") is not None
            if sold:
                price_cell = tds[4] if len(tds) > 4 else None
                price_text = price_cell.get_text(" ", strip=True) if price_cell else ""
                maybe_add(title, price_text)
        except Exception:
            pass
        i += 1

    # 2) Fallback heuristic
    if not found_any:
        for tbl in soup.find_all("table"):
            for tr in tbl.find_all("tr"):
                tds = tr.find_all("td")
                if not tds:
                    continue
                row_text = " ".join(td.get_text(" ", strip=True) for td in tds).lower()
                if "sold" not in row_text:
                    continue
                title = (tr.find("a").get_text(" ", strip=True) if tr.find("a") else "")
                price_text = tds[-1].get_text(" ", strip=True)
                if "$" in price_text:
                    maybe_add(title, price_text)

    return X_rows, y_rows


# ----------------------------- math helpers -----------------------------

@dataclass
class Standardizer:
    mu: np.ndarray
    sigma: np.ndarray
    y_mean: float

    @staticmethod
    def fit(X: np.ndarray, y: np.ndarray) -> "Standardizer":
        mu = X.mean(axis=0)
        sigma = X.std(axis=0, ddof=0)
        sigma = np.where(sigma == 0, 1.0, sigma)  # avoid divide-by-zero
        y_mean = float(y.mean())
        return Standardizer(mu=mu, sigma=sigma, y_mean=y_mean)

    def transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        Xs = (X - self.mu) / self.sigma
        ys = (y - self.y_mean) if y is not None else None
        return Xs, ys

    def to_original_coeffs(self, w: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Given weights w for standardized X (and centered y),
        returns (coef_orig, intercept_orig) for y ≈ b0 + X @ coef_orig.
        """
        coef_orig = w / self.sigma
        intercept = self.y_mean - float(np.dot(self.mu, coef_orig))
        return coef_orig, intercept


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> Tuple[np.ndarray, float, Standardizer]:
    """
    Fit ridge in standardized space, return (coef_orig, intercept, std).
    Solves (X^T X + lam I) w = X^T y in standardized coords.
    """
    std = Standardizer.fit(X, y)
    Xs, ys = std.transform(X, y)
    A = Xs.T @ Xs + lam * np.eye(Xs.shape[1], dtype=float)
    b = Xs.T @ ys
    w = np.linalg.solve(A, b)
    coef, intercept = std.to_original_coeffs(w)
    return coef, intercept, std


def ridge_path(X: np.ndarray, y: np.ndarray, lambdas: np.ndarray) -> np.ndarray:
    """
    Compute ORIGINAL-units coefficients across lambdas for a coefficient-path plot.
    Returns array shape (len(lambdas), n_features).
    """
    std = Standardizer.fit(X, y)
    Xs, ys = std.transform(X, y)
    XT = Xs.T
    n = Xs.shape[1]
    coefs_orig = []
    for lam in lambdas:
        A = XT @ Xs + lam * np.eye(n, dtype=float)
        w = np.linalg.solve(A, XT @ ys)
        coef_orig, _ = std.to_original_coeffs(w)
        coefs_orig.append(coef_orig)
    return np.asarray(coefs_orig)


def kfold_indices(n: int, k: int, seed: int = 42):
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    return np.array_split(idx, k)


def cv_select_lambda(X: np.ndarray, y: np.ndarray, lambdas: np.ndarray,
                     k: int = 10, seed: int = 42) -> Tuple[float, np.ndarray]:
    """k-fold CV over lambdas. Returns (best_lambda, mean_rss_per_lambda)."""
    folds = kfold_indices(len(y), k, seed=seed)
    mean_rss = np.zeros_like(lambdas, dtype=float)

    for i, lam in enumerate(lambdas):
        rss_sum = 0.0
        count = 0
        for f in range(k):
            val_idx = folds[f]
            tr_idx = np.concatenate([folds[j] for j in range(k) if j != f])
            coef, intercept, _ = ridge_fit(X[tr_idx], y[tr_idx], lam)
            yhat = X[val_idx] @ coef + intercept
            rss_sum += float(((y[val_idx] - yhat) ** 2).sum())
            count += len(val_idx)
        mean_rss[i] = rss_sum / count

    best_i = int(np.argmin(mean_rss))
    return float(lambdas[best_i]), mean_rss


# ----------------------------- CLI / main -----------------------------

def main():
    ap = argparse.ArgumentParser(description="NumPy-only LEGO ridge regression (scrape + CV + fit).")
    ap.add_argument("--html", nargs=4, action="append", metavar=("FILE", "YEAR", "PIECES", "MSRP"),
                    help="Add a saved HTML file with metadata. May be repeated.")
    ap.add_argument("--min-frac", type=float, default=0.5,
                    help="drop listings with price <= min_frac * MSRP (default 0.5)")
    ap.add_argument("--lam-start", type=float, default=1e-4)
    ap.add_argument("--lam-end", type=float, default=1e4)
    ap.add_argument("--lam-num", type=int, default=30)
    ap.add_argument("--cv-folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--coef-plot", default="", help="path to save coefficient-path PNG (optional)")
    ap.add_argument("--demo", action="store_true", help="run on a tiny synthetic dataset instead of scraping")
    args = ap.parse_args()

    # 1) Build dataset
    X_rows: List[List[float]] = []
    y_rows: List[float] = []

    if args.demo:
        # synthetic: price ~ 0.8*year + 0.05*pieces + 25*is_new + 0.6*msrp + noise
        rng = np.random.default_rng(args.seed)
        for _ in range(300):
            yr = rng.integers(2000, 2012)
            pcs = max(100, int(rng.normal(2500, 1200)))
            new = 1.0 if rng.random() < 0.4 else 0.0
            msrp = rng.uniform(50, 500)
            price = (0.8*(yr-2000) + 0.05*pcs + 25*new + 0.6*msrp + rng.normal(0, 50))
            X_rows.append([yr, pcs, new, msrp])
            y_rows.append(price)
    else:
        if not args.html:
            print("No --html inputs and --demo not set. Nothing to do.")
            return
        for (file, yr, pieces, msrp) in args.html:
            file_abs = resolve(file)
            yr, pieces, msrp = int(yr), int(pieces), float(msrp)
            if not os.path.exists(file_abs):
                print(f"[warn] missing file: {file_abs}")
                continue
            Xs, ys = scrape_file(file_abs, yr, pieces, msrp, min_frac=args.min_frac)
            X_rows.extend(Xs)
            y_rows.extend(ys)

    if not y_rows:
        print("No rows parsed. Exiting.")
        return

    X = np.asarray(X_rows, dtype=float)
    y = np.asarray(y_rows, dtype=float)

    # 2) Lambda grid + CV
    lambdas = np.logspace(math.log10(args.lam_start), math.log10(args.lam_end), args.lam_num)
    best_lam, mean_rss = cv_select_lambda(X, y, lambdas, k=args.cv_folds, seed=args.seed)
    print(f"[cv] best λ = {best_lam:.6g}")
    print(f"[cv] mean RSS across grid (first, mid, last): "
          f"{mean_rss[0]:.2f}, {mean_rss[len(mean_rss)//2]:.2f}, {mean_rss[-1]:.2f}")

    # 3) Final fit on all data with best lambda
    coef, intercept, _ = ridge_fit(X, y, best_lam)
    names = ["year", "pieces", "is_new", "msrp"]
    terms = " ".join([f"{coef[i]:+g}*{names[i]}" for i in range(len(names))])
    print(f"Final model:\n  price ≈ {intercept:+g} {terms}")

    # 4) Optional coefficient path plot
    if args.coef_plot:
        if plt is None:
            print("[warn] matplotlib not available; cannot save plot.")
        else:
            paths = ridge_path(X, y, lambdas)  # original-units coefficients
            plt.figure(figsize=(9, 5))
            for j in range(paths.shape[1]):
                plt.plot(np.log10(lambdas), paths[:, j])
            plt.title("Ridge coefficient paths")
            plt.xlabel("log10(lambda)")
            plt.ylabel("Coefficient (original units)")
            plt.tight_layout()
            plt.savefig(args.coef_plot, dpi=150)
            plt.close()
            print(f"[plot] saved: {args.coef_plot}")


if __name__ == "__main__":
    main()
