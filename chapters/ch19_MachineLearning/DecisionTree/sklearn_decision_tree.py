# sklearn_decision_tree.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, argparse, shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib  # pip install joblib

def resolve_here(path: str) -> str:
    if os.path.isabs(path):
        return path
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, path)

def load_lenses(path: str = "lenses.txt") -> tuple[pd.DataFrame, pd.Series]:
    """
    lenses.txt expected as a tab-separated file with 5 columns:
    age, prescript, astigmatic, tearRate, target
    If your file has no header, we add one.
    """
    path = resolve_here(path)          # <— add this line
    df = pd.read_csv(path, sep="\t", header=None,
                     names=["age","prescript","astigmatic","tearRate","target"])
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y

def train_tree(X: pd.DataFrame, y: pd.Series, max_depth: int = 4, random_state: int = 42) -> Pipeline:
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), list(X.columns))
    ])
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    pipe = Pipeline([("pre", pre), ("tree", clf)])
    pipe.fit(X, y)
    return pipe

def export_graphviz_png_pdf(pipe: Pipeline, outdir: str, basename: str = "tree_graphviz"):
    """
    Export using Graphviz if `dot` is installed. Produces PNG + PDF.
    """
    if shutil.which("dot") is None:
        raise RuntimeError("Graphviz 'dot' executable not found")

    features = pipe.named_steps["pre"].get_feature_names_out()
    clf: DecisionTreeClassifier = pipe.named_steps["tree"]
    dot_str = export_graphviz(
        clf,
        out_file=None,
        feature_names=features,
        class_names=[str(c) for c in clf.classes_],
        filled=True,
        rounded=True,
        special_characters=True
    )

    # Try python-graphviz first, then fallback to pydotplus if needed
    try:
        import graphviz  # pip install graphviz
        src = graphviz.Source(dot_str)
        src.format = "png"
        src.render(os.path.join(outdir, basename), cleanup=True)
        src.format = "pdf"
        src.render(os.path.join(outdir, basename), cleanup=True)
    except Exception:
        import pydotplus  # pip install pydotplus
        graph = pydotplus.graph_from_dot_data(dot_str)
        graph.write_png(os.path.join(outdir, f"{basename}.png"))
        graph.write_pdf(os.path.join(outdir, f"{basename}.pdf"))

def export_matplotlib_png(pipe: Pipeline, outdir: str, basename: str = "tree_matplotlib"):
    """
    Fallback visualization that doesn’t need system Graphviz.
    """
    features = pipe.named_steps["pre"].get_feature_names_out()
    clf: DecisionTreeClassifier = pipe.named_steps["tree"]
    plt.figure(figsize=(12, 8))
    plot_tree(
        clf,
        feature_names=features,
        class_names=[str(c) for c in clf.classes_],
        filled=True, rounded=True, impurity=True
    )
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{basename}.png"), dpi=180)
    plt.close()

def main():
    ap = argparse.ArgumentParser(description="Decision Tree on lenses dataset with Graphviz or Matplotlib visualization.")
    ap.add_argument("--data", default="lenses.txt", help="path to lenses.txt (TSV)")
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--outdir", default="lenses_outputs")
    ap.add_argument("--viz", choices=["auto","graphviz","matplotlib"], default="auto",
                    help="prefer Graphviz if available; else matplotlib")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1) Load
    X, y = load_lenses(args.data)

    # 2) Train
    pipe = train_tree(X, y, max_depth=args.max_depth)
    preds = pipe.predict(X)
    acc = accuracy_score(y, preds)
    print(f"Training accuracy (on full tiny dataset): {acc:.3f}")

    # 3) Save model
    model_path = os.path.join(args.outdir, "lenses_tree.joblib")
    joblib.dump(pipe, model_path)
    print(f"Saved model to {model_path}")

    # 4) Visualize
    wanted_graphviz = args.viz in ("graphviz","auto")
    did_graphviz = False
    if wanted_graphviz and shutil.which("dot") is not None:
        try:
            export_graphviz_png_pdf(pipe, args.outdir, basename="tree_graphviz")
            did_graphviz = True
            print(f"Saved Graphviz visuals to {args.outdir}/tree_graphviz.png and .pdf")
        except Exception as e:
            print(f"[Graphviz route failed: {e}] Falling back to Matplotlib...")

    if not did_graphviz:
        export_matplotlib_png(pipe, args.outdir, basename="tree_matplotlib")
        print(f"Saved Matplotlib visual to {args.outdir}/tree_matplotlib.png")

    # 5) Example prediction using raw categories (no manual encoding needed)
    example = pd.DataFrame([["young", "myope", "no", "reduced"]],
                        columns=X.columns)  # must match training column names
    pred = pipe.predict(example)[0]
    print("Example prediction", example.iloc[0].to_list(), "→", pred)


if __name__ == "__main__":
    main()
    #Run this 
    #python DecisionTree/sklearn_decision_tree.py --data lenses.txt --outdir lenses_outputs --viz auto

