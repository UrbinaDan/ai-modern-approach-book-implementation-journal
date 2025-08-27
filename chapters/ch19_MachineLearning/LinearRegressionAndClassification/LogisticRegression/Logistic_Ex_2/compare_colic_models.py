# compare_colic_models.py
# -*- coding: utf-8 -*-
"""
Compare three logistic-regression trainers on the Horse Colic dataset:
  1) scikit-learn LogisticRegression
  2) Custom batch gradient ascent
  3) Custom improved SGD

- Prints accuracy, confusion matrix, and (optional) classification report
- Saves plots: ROC, Precision-Recall, confusion-matrix heatmaps, accuracy bar chart
- Works headless (saves PNGs to --outdir)
"""

from __future__ import annotations
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Metrics
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Import your implementation
import colic_stochGradAscent_vs_Sklearn as clm


def ensure_outdir(path: str) -> str:
    if not os.path.isabs(path):
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, path)
    os.makedirs(path, exist_ok=True)
    return path


def plot_confusion(cm: np.ndarray, title: str, save_path: str):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    # annotate cells
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha='center', va='center')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(y_true: np.ndarray, proba_dict: dict[str, np.ndarray], save_path: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, p in proba_dict.items():
        fpr, tpr, _ = roc_curve(y_true, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0,1], [0,1], linestyle="--")
    ax.set_title("ROC Curves")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_pr_curves(y_true: np.ndarray, proba_dict: dict[str, np.ndarray], save_path: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, p in proba_dict.items():
        prec, rec, _ = precision_recall_curve(y_true, p)
        # area under PR (not average precision, but fine for visual)
        ax.plot(rec, prec, label=name)
    ax.set_title("Precision–Recall Curves")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_bars(acc_map: dict[str, float], save_path: str):
    names = list(acc_map.keys())
    vals = [acc_map[k] for k in names]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, vals)
    ax.set_ylim(0, 1)
    ax.set_title("Accuracy Comparison")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description="Compare sklearn vs custom batch vs custom SGD on Horse Colic."
    )
    ap.add_argument("--train", default="horseColicTraining.txt", help="path to training TSV")
    ap.add_argument("--test",  default="horseColicTest.txt",     help="path to test TSV")

    # Common
    ap.add_argument("--scale", action="store_true", help="standardize features")
    ap.add_argument("--outdir", default="colic_outputs", help="where to save PNGs and report.txt")
    ap.add_argument("--report", action="store_true", help="print full classification reports")

    # sklearn
    ap.add_argument("--solver", default="saga", help="sklearn solver")
    ap.add_argument("--max-iter", type=int, default=5000, help="sklearn max_iter")
    ap.add_argument("--C", type=float, default=1.0, help="sklearn inverse reg strength")

    # custom batch
    ap.add_argument("--alpha", type=float, default=1e-3, help="batch learning rate")
    ap.add_argument("--iters", type=int,   default=500,  help="batch iterations")
    ap.add_argument("--l2", type=float,    default=0.0,  help="L2 for custom trainers")

    # custom SGD
    ap.add_argument("--epochs", type=int,  default=150,  help="SGD epochs")
    ap.add_argument("--seed",   type=int,  default=42,   help="SGD shuffle seed")
    args = ap.parse_args()

    outdir = ensure_outdir(args.outdir)

    # Load data (path-safe via colic_logreg_modern)
    X_tr, y_tr = clm.load_tsv_xy(args.train)
    X_te, y_te = clm.load_tsv_xy(args.test)

    # Optional scaling (fit on train; apply to test)
    scaler = None
    if args.scale:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

    y_true = y_te.ravel()

    # -------------------- 1) sklearn --------------------
    sk_clf = LogisticRegression(
        solver=args.solver,
        max_iter=args.max_iter,
        C=args.C,
    )
    sk_clf.fit(X_tr, y_tr.ravel())
    sk_pred = sk_clf.predict(X_te)
    # predict_proba exists for most solvers; fallback if not
    if hasattr(sk_clf, "predict_proba"):
        sk_proba = sk_clf.predict_proba(X_te)[:, 1]
    else:
        sk_proba = sk_clf.decision_function(X_te)
        sk_proba = 1.0 / (1.0 + np.exp(-sk_proba))
    sk_acc = accuracy_score(y_true, sk_pred)
    sk_cm = confusion_matrix(y_true, sk_pred)

    # -------------------- 2) custom batch --------------------
    Xtr_b = np.hstack([np.ones((X_tr.shape[0], 1)), X_tr])
    Xte_b = np.hstack([np.ones((X_te.shape[0], 1)), X_te])
    w_batch = clm.train_batch(Xtr_b, y_tr, alpha=args.alpha, iters=args.iters, l2=args.l2)
    b_pred = clm.predict_label(Xte_b, w_batch)
    b_proba = clm.predict_proba(Xte_b, w_batch)
    b_acc = accuracy_score(y_true, b_pred)
    b_cm = confusion_matrix(y_true, b_pred)

    # -------------------- 3) custom SGD --------------------
    w_sgd = clm.train_sgd_improved(Xtr_b, y_tr, epochs=args.epochs, l2=args.l2, seed=args.seed)
    s_pred = clm.predict_label(Xte_b, w_sgd)
    s_proba = clm.predict_proba(Xte_b, w_sgd)
    s_acc = accuracy_score(y_true, s_pred)
    s_cm = confusion_matrix(y_true, s_pred)

    # Print summary to console
    print(f"[sklearn]      acc={sk_acc:.4f}\nConfusion:\n{sk_cm}\n")
    print(f"[custom-batch] acc={b_acc:.4f}\nConfusion:\n{b_cm}\n")
    print(f"[custom-sgd]   acc={s_acc:.4f}\nConfusion:\n{s_cm}\n")

    # Optional classification reports
    if args.report:
        print("=== Classification report (sklearn) ===")
        print(classification_report(y_true, sk_pred, digits=4))
        print("=== Classification report (custom-batch) ===")
        print(classification_report(y_true, b_pred, digits=4))
        print("=== Classification report (custom-sgd) ===")
        print(classification_report(y_true, s_pred, digits=4))

    # Save a text report
    with open(os.path.join(outdir, "report.txt"), "w", encoding="utf-8") as f:
        f.write(f"[sklearn] acc={sk_acc:.6f}\nConfusion:\n{sk_cm}\n\n")
        f.write(f"[custom-batch] acc={b_acc:.6f}\nConfusion:\n{b_cm}\n\n")
        f.write(f"[custom-sgd]   acc={s_acc:.6f}\nConfusion:\n{s_cm}\n\n")
        if args.report:
            f.write("=== Classification report (sklearn) ===\n")
            f.write(classification_report(y_true, sk_pred, digits=4))
            f.write("\n=== Classification report (custom-batch) ===\n")
            f.write(classification_report(y_true, b_pred, digits=4))
            f.write("\n=== Classification report (custom-sgd) ===\n")
            f.write(classification_report(y_true, s_pred, digits=4))

    # Plots (saved to outdir)
    plot_confusion(sk_cm, "Confusion (sklearn)", os.path.join(outdir, "cm_sklearn.png"))
    plot_confusion(b_cm,  "Confusion (batch)",   os.path.join(outdir, "cm_batch.png"))
    plot_confusion(s_cm,  "Confusion (sgd)",     os.path.join(outdir, "cm_sgd.png"))

    proba_map = {
        "sklearn": sk_proba,
        "batch":   b_proba,
        "sgd":     s_proba,
    }
    plot_roc_curves(y_true, proba_map, os.path.join(outdir, "roc.png"))
    plot_pr_curves(y_true, proba_map,  os.path.join(outdir, "pr.png"))

    plot_accuracy_bars(
        {"sklearn": sk_acc, "batch": b_acc, "sgd": s_acc},
        os.path.join(outdir, "accuracy.png")
    )

    print(f"Saved outputs to: {outdir}")
    print("Files: report.txt, cm_*.png, roc.png, pr.png, accuracy.png")


if __name__ == "__main__":
    main()
