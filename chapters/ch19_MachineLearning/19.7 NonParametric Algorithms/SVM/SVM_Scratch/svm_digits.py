# svm_digits.py
# ------------------------------------------------------------
# Full SMO with kernels (linear/RBF) on ASCII 32x32 digit bitmaps.
# - Maps class 9 -> -1 and all others -> +1  (binary classification)
# - Trains on trainingDigits/, tests on testDigits/
# - Prints #support vectors and train/test errors
# Example:
#   python svm_digits.py --train-dir trainingDigits --test-dir testDigits \
#                        --kernel rbf --gamma 10 --C 200 --toler 1e-4 --max-iter 10
# ------------------------------------------------------------

import argparse
import os
import numpy as np
from typing import Tuple
from svm_smo import SMO, kernel_linear, kernel_rbf  # reuse implementation


def img2vec(path: str) -> np.ndarray:
    v = np.zeros(1024, float)
    with open(path, "r") as f:
        for i in range(32):
            line = f.readline()
            for j in range(32):
                v[32 * i + j] = int(line[j])
    return v


def load_digit_dir(dir_path: str, pos_digit: int = 9) -> Tuple[np.ndarray, np.ndarray]:
    files = sorted(os.listdir(dir_path))
    X, y = [], []
    for fn in files:
        if not fn.endswith(".txt"):
            continue
        label = int(fn.split("_")[0])
        yi = -1.0 if label == pos_digit else +1.0
        X.append(img2vec(os.path.join(dir_path, fn)))
        y.append(yi)
    return np.asarray(X, float), np.asarray(y, float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--kernel", choices=["linear", "rbf"], default="rbf")
    ap.add_argument("--gamma", type=float, default=10.0, help="RBF width parameter (matches previous scaling)")
    ap.add_argument("--C", type=float, default=200.0)
    ap.add_argument("--toler", type=float, default=1e-4)
    ap.add_argument("--max-iter", type=int, default=10)
    args = ap.parse_args()

    Xtr, ytr = load_digit_dir(args.train_dir)
    model = SMO(Xtr, ytr, C=args.C, toler=args.toler, kernel=args.kernel, gamma=args.gamma).fit(max_iter=args.max_iter)
    yhat_tr = model.predict(Xtr)
    tr_err = (yhat_tr != ytr).mean()
    nsv = int((model.alphas > 1e-8).sum())
    print(f"Train error: {tr_err*100:.2f}%  (SVs={nsv})")

    Xte, yte = load_digit_dir(args.test_dir)
    yhat_te = model.predict(Xte)
    te_err = (yhat_te != yte).mean()
    print(f"Test  error: {te_err*100:.2f}%")


if __name__ == "__main__":
    main()
