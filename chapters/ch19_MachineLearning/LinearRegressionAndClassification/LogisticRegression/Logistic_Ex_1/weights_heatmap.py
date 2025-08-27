# weights_heatmap.py
#This code visualizes the weights learned by logistic regression on 32x32 pixel images.
# It loads training data from text files, trains a logistic regression model,
# and displays the weights as a heatmap to interpret which pixels influence classification.
# Positive weights indicate pixels pushing towards class '1', negative towards class '0'.
# Larger magnitude weights indicate more influential pixels.
# Ensure you have numpy and matplotlib installed to run this script.
import os
import numpy as np
import matplotlib.pyplot as plt

# ---------- Data loading (same logic as your script) ----------
def load_data(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.txt')]
    files.sort()  # optional: stable order
    m = len(files)
    X = np.zeros((m, 1024), dtype=np.float32)
    y = np.zeros((m, 1), dtype=np.int32)

    for i, fname in enumerate(files):
        path = os.path.join(directory, fname)
        row = np.zeros((1024,), dtype=np.float32)
        with open(path, 'r') as f:
            for j in range(32):
                line = f.readline().strip()
                row[j*32:(j+1)*32] = [ord(c) - 48 for c in line[:32]]  # '0'/'1' -> 0/1
        X[i, :] = row
        y[i, 0] = int(fname.split('.')[0].split('_')[0])  # '0_12.txt' -> 0
    return X, y

def sigmoid(x):
    # stable sigmoid to avoid overflow
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[neg])
    out[neg] = exp_x / (1.0 + exp_x)
    return out


def train_logreg(X, y, alpha=0.07, max_iters=10):
    """Gradient ascent on log-likelihood for logistic regression (binary 0/1)."""
    m, n = X.shape
    w = np.ones((n, 1), dtype=np.float64)
    for _ in range(max_iters):
        p = sigmoid(X @ w)        # (m,1)
        grad = X.T @ (y - p)      # (n,1)
        w += alpha * grad
    return w  # (1024,1)

# ---------- Train, extract weights, and plot heatmap ----------
if __name__ == "__main__":
    train_dir = "LinearRegressionAndClassification/LogisticRegression/Logistic_Ex_1/train" # change if needed
    X_train, y_train = load_data(train_dir)
    w = train_logreg(X_train, y_train, alpha=0.07, max_iters=10)  # (1024,1)

    # reshape to 32x32 for visualization
    W2D = w.reshape(32, 32)

    # Helpful normalization for display (optional):
    # W2D = W2D / np.max(np.abs(W2D))

    plt.imshow(W2D)  # do not set a style or colors per your environment rules
    plt.title("Logistic Regression Weights (32×32)")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    # Interpretation hints in console:
    print("Note:")
    print("- Positive weights push predictions toward class '1'.")
    print("- Negative weights push predictions toward class '0'.")
    print("- Larger magnitude = more influential pixel.")
    plt.savefig("weights_heatmap.png", dpi=180, bbox_inches="tight")
    print("Saved to weights_heatmap.png")

