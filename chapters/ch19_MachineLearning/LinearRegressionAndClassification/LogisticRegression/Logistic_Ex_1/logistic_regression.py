# -*- coding: utf-8 -*-
import os
import numpy as np

def load_data(directory):
    """Load 32x32 ASCII digit images from 'directory' into (X, y).
    X: (m, 1024) float32 in {0,1}, y: (m, 1) int32 in {0,1}.
    """
    files = [f for f in os.listdir(directory) if f.endswith('.txt')]
    m = len(files)
    X = np.zeros((m, 1024), dtype=np.float32)
    y = np.zeros((m, 1), dtype=np.int32)

    for i, fname in enumerate(files):
        path = os.path.join(directory, fname)
        # Flatten 32x32 characters into 1x1024
        row = np.zeros((1024,), dtype=np.float32)
        with open(path, 'r') as f:
            for j in range(32):
                line = f.readline().strip()
                # assume each char is '0' or '1'
                row[j*32:(j+1)*32] = np.fromiter((ord(c) - 48 for c in line[:32]), dtype=np.float32, count=32)
        X[i, :] = row
        y[i, 0] = int(fname.split('.')[0].split('_')[0])  # '0_12.txt' -> 0
    return X, y

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def train_logreg(X, y, alpha=0.07, max_iters=10):
    """Gradient ascent on log-likelihood for logistic regression."""
    m, n = X.shape
    w = np.ones((n, 1), dtype=np.float64)
    for _ in range(max_iters):
        p = sigmoid(X @ w)        # (m,1)
        grad = X.T @ (y - p)      # (n,1)
        w += alpha * grad
    return w

def classify_dir(test_dir, weights, threshold=0.5):
    X, y = load_data(test_dir)
    p = sigmoid(X @ weights).reshape(-1)  # probabilities
    y_true = y.reshape(-1)
    y_pred = (p > threshold).astype(int)

    errors = (y_pred != y_true)
    for yi, pi in zip(y_true, y_pred):
        print(f"{yi} is classified as: {pi}")
    err_rate = errors.mean()
    print("error rate is: {:.4f}".format(err_rate))
    return err_rate

def digit_recognition(train_dir, test_dir, alpha=0.07, max_iters=10):
    X_train, y_train = load_data(train_dir)
    w = train_logreg(X_train, y_train, alpha=alpha, max_iters=max_iters)
    return classify_dir(test_dir, w)

# Example:
# err = digit_recognition('train', 'test', alpha=0.07, max_iters=10)
