# decision_tree_improved.py
from __future__ import annotations
import math, pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Iterable

import numpy as np

# --------------------------- utilities ---------------------------

def _as_array(X) -> np.ndarray:
    if hasattr(X, "values"):  # pandas
        X = X.values
    return np.asarray(X, dtype=object)

def _as_1d(y) -> np.ndarray:
    if hasattr(y, "values"):
        y = y.values
    y = np.asarray(y).ravel()
    return y

def _entropy(y: np.ndarray) -> float:
    if y.size == 0: return 0.0
    _, cnt = np.unique(y, return_counts=True)
    p = cnt / cnt.sum()
    return float(-(p * np.log2(p)).sum())

def _gini(y: np.ndarray) -> float:
    if y.size == 0: return 0.0
    _, cnt = np.unique(y, return_counts=True)
    p = cnt / cnt.sum()
    return float(1.0 - (p * p).sum())

def _split_info(sizes: Iterable[int]) -> float:
    n = sum(sizes)
    return sum((s/n) * math.log2(n/s) for s in sizes if s > 0)

# --------------------------- node model ---------------------------

@dataclass
class Node:
    # for leaf
    prediction: Optional[Any] = None
    class_counts: Optional[Dict[Any, int]] = None
    # for internal node
    feature_index: Optional[int] = None
    feature_name: Optional[str] = None
    # if numeric split: x <= threshold goes left, else right
    threshold: Optional[float] = None
    left: "Node" = None
    right: "Node" = None
    # if categorical split: children by value
    children: Optional[Dict[Any, "Node"]] = None

    def is_leaf(self) -> bool:
        return self.prediction is not None

# --------------------------- main tree ---------------------------

class DecisionTreeImproved:
    """
    A small, from-scratch decision tree that supports:
      - numeric & categorical features
      - criteria: 'entropy' (information gain), 'gain_ratio', or 'gini'
      - pre-pruning: max_depth, min_samples_split, min_samples_leaf, min_gain
      - optional reduced-error post-pruning with a validation set
    """
    def __init__(
        self,
        criterion: str = "gain_ratio",  # 'entropy' | 'gain_ratio' | 'gini'
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_gain: float = 1e-7,
        feature_types: Optional[List[str]] = None,  # 'numeric' | 'categorical'
        random_state: Optional[int] = None,
    ):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.min_gain = float(min_gain)
        self.feature_types = feature_types  # optional override
        self.random_state = random_state
        self.root: Optional[Node] = None
        self.feature_names: Optional[List[str]] = None

    # -------- public API --------
    def fit(self, X, y, feature_names: Optional[List[str]] = None,
            X_val=None, y_val=None):
        X = _as_array(X)
        y = _as_1d(y)
        n, d = X.shape
        self.feature_names = feature_names or [f"x{i}" for i in range(d)]
        self._ftypes = self._infer_types(X) if self.feature_types is None else self.feature_types
        self.root = self._grow(X, y, depth=0)

        # optional reduced-error post-pruning using validation set
        if X_val is not None and y_val is not None:
            Xv = _as_array(X_val); yv = _as_1d(y_val)
            self._prune_reduced_error(self.root, Xv, yv)

        return self

    def predict(self, X) -> np.ndarray:
        X = _as_array(X)
        return np.array([self._predict_row(self.root, row) for row in X], dtype=object)

    def predict_proba(self, X) -> np.ndarray:
        X = _as_array(X)
        probs = []
        for row in X:
            node = self._traverse(self.root, row)
            counts = node.class_counts or {}
            total = sum(counts.values()) or 1
            labels = self.classes_
            probs.append([counts.get(c, 0)/total for c in labels])
        return np.asarray(probs, dtype=float)

    @property
    def classes_(self) -> List[Any]:
        # gather from root counts if present
        if self.root and self.root.class_counts:
            return sorted(self.root.class_counts.keys())
        return []

    def export_text(self) -> str:
        lines: List[str] = []
        self._dump(self.root, prefix="", lines=lines)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return self._node_to_dict(self.root)

    @staticmethod
    def from_dict(obj: Dict[str, Any]) -> "DecisionTreeImproved":
        tree = DecisionTreeImproved()
        tree.root = DecisionTreeImproved._dict_to_node(obj)
        return tree

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({
                "tree": self.to_dict(),
                "feature_names": self.feature_names,
                "criterion": self.criterion
            }, f)

    @staticmethod
    def load(path: str) -> "DecisionTreeImproved":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        t = DecisionTreeImproved(criterion=obj.get("criterion", "gain_ratio"))
        t.feature_names = obj.get("feature_names")
        t.root = DecisionTreeImproved._dict_to_node(obj["tree"])
        return t

    # -------- building / splitting --------
    def _infer_types(self, X: np.ndarray) -> List[str]:
        types = []
        for j in range(X.shape[1]):
            col = X[:, j]
            # numeric if values are ints/floats and not too few uniques
            is_num = all(isinstance(v, (int, float, np.integer, np.floating)) for v in col)
            types.append("numeric" if is_num else "categorical")
        return types

    def _leaf(self, y: np.ndarray) -> Node:
        vals, cnt = np.unique(y, return_counts=True)
        pred = vals[np.argmax(cnt)]
        counts = {v: int(c) for v, c in zip(vals, cnt)}
        return Node(prediction=pred, class_counts=counts)

    def _grow(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        # stopping conditions
        if len(np.unique(y)) == 1:
            return self._leaf(y)
        if (self.max_depth is not None and depth >= self.max_depth) \
           or X.shape[0] < self.min_samples_split \
           or X.shape[1] == 0:
            return self._leaf(y)

        # find best split
        best = self._best_split(X, y)
        if best is None or best.gain < self.min_gain:
            return self._leaf(y)

        fidx, fname = best.fidx, self.feature_names[best.fidx]
        if best.kind == "numeric":
            left_mask = X[:, fidx] <= best.threshold
            right_mask = ~left_mask
            # enforce min_samples_leaf
            if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
                return self._leaf(y)
            left = self._grow(X[left_mask], y[left_mask], depth+1)
            right = self._grow(X[right_mask], y[right_mask], depth+1)
            return Node(feature_index=fidx, feature_name=fname,
                        threshold=float(best.threshold), left=left, right=right,
                        class_counts=self._leaf(y).class_counts)
        else:  # categorical multiway split
            children = {}
            for val in best.values:
                mask = (X[:, fidx] == val)
                if mask.sum() < self.min_samples_leaf:
                    children[val] = self._leaf(y[mask])  # tiny child becomes leaf
                else:
                    # drop used column for children to avoid re-splitting on same feature (ID3-style)
                    X_child = np.delete(X[mask], fidx, axis=1)
                    child_names_bak = self.feature_names
                    self.feature_names = child_names_bak[:fidx] + child_names_bak[fidx+1:]
                    child_types_bak = self._ftypes
                    self._ftypes = child_types_bak[:fidx] + child_types_bak[fidx+1:]
                    child = self._grow(X_child, y[mask], depth+1)
                    # restore names/types for siblings
                    self.feature_names = child_names_bak
                    self._ftypes = child_types_bak
                    children[val] = child
            return Node(feature_index=fidx, feature_name=fname, children=children,
                        class_counts=self._leaf(y).class_counts)

    # structure to hold split info
    @dataclass
    class _Split:
        fidx: int
        kind: str                 # "numeric" or "categorical"
        threshold: Optional[float] = None
        values: Optional[List[Any]] = None
        gain: float = -1.0

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> Optional["_Split"]:
        parent_imp = _entropy(y) if self.criterion in ("entropy", "gain_ratio") else _gini(y)
        best: Optional[DecisionTreeImproved._Split] = None

        for j, ftype in enumerate(self._ftypes):
            col = X[:, j]

            # numeric -> binary split by threshold
            if ftype == "numeric":
                # unique sorted numeric candidates; try midpoints
                vals = np.unique(col.astype(float))
                if vals.size <= 1: 
                    continue
                thr_cands = (vals[:-1] + vals[1:]) / 2.0
                for thr in thr_cands:
                    left = (col.astype(float) <= thr)
                    right = ~left
                    if left.sum() < self.min_samples_leaf or right.sum() < self.min_samples_leaf:
                        continue
                    yL, yR = y[left], y[right]
                    if self.criterion == "gini":
                        child_imp = (yL.size/ y.size)*_gini(yL) + (yR.size/ y.size)*_gini(yR)
                        gain = parent_imp - child_imp
                    else:
                        child_imp = (yL.size/ y.size)*_entropy(yL) + (yR.size/ y.size)*_entropy(yR)
                        gain = parent_imp - child_imp
                        if self.criterion == "gain_ratio":
                            si = _split_info([yL.size, yR.size])
                            if si > 0: gain = gain / si
                            else:      gain = -1.0
                    if best is None or gain > best.gain:
                        best = self._Split(fidx=j, kind="numeric", threshold=float(thr), gain=float(gain))

            # categorical -> multiway split by value
            else:
                values = list(dict.fromkeys(col))  # preserve order
                if len(values) <= 1:
                    continue
                parts = [y[col == v] for v in values]
                if any(len(p) < self.min_samples_leaf for p in parts):
                    # still allow; tiny children will be leaves
                    pass
                if self.criterion == "gini":
                    child_imp = sum((p.size/y.size)*_gini(p) for p in parts)
                    gain = parent_imp - child_imp
                else:
                    child_imp = sum((p.size/y.size)*_entropy(p) for p in parts)
                    gain = parent_imp - child_imp
                    if self.criterion == "gain_ratio":
                        si = _split_info([p.size for p in parts])
                        if si > 0: gain = gain / si
                        else:      gain = -1.0
                if best is None or gain > best.gain:
                    best = self._Split(fidx=j, kind="categorical", values=values, gain=float(gain))

        return best

    # -------- prediction helpers --------
    def _traverse(self, node: Node, row: np.ndarray) -> Node:
        while not node.is_leaf():
            if node.threshold is not None:  # numeric binary
                val = float(row[node.feature_index])
                node = node.left if val <= node.threshold else node.right
            elif node.children is not None:  # categorical
                val = row[node.feature_index]
                node = node.children.get(val) or node.children[next(iter(node.children))]  # fallback
            else:
                break
        return node

    def _predict_row(self, node: Node, row: np.ndarray):
        leaf = self._traverse(node, row)
        return leaf.prediction

    # -------- export / pruning --------
    def _dump(self, node: Node, prefix: str, lines: List[str]):
        if node.is_leaf():
            lines.append(prefix + f"→ {node.prediction} {node.class_counts}")
            return
        fname = node.feature_name or f"f{node.feature_index}"
        if node.threshold is not None:
            lines.append(prefix + f"[{fname} ≤ {node.threshold:.6g}]")
            self._dump(node.left, prefix + "  ", lines)
            lines.append(prefix + f"[{fname} > {node.threshold:.6g}]")
            self._dump(node.right, prefix + "  ", lines)
        else:
            lines.append(prefix + f"[{fname}]")
            for v, child in (node.children or {}).items():
                lines.append(prefix + f"  = {v}")
                self._dump(child, prefix + "    ", lines)

    def _node_to_dict(self, node: Node) -> Dict[str, Any]:
        if node.is_leaf():
            return {"leaf": True, "prediction": node.prediction, "counts": node.class_counts}
        d = {"leaf": False, "feature_index": node.feature_index, "feature_name": node.feature_name}
        if node.threshold is not None:
            d.update({"threshold": node.threshold,
                      "left": self._node_to_dict(node.left),
                      "right": self._node_to_dict(node.right)})
        else:
            d["children"] = {k: self._node_to_dict(v) for k, v in (node.children or {}).items()}
        return d

    @staticmethod
    def _dict_to_node(obj: Dict[str, Any]) -> Node:
        if obj.get("leaf"):
            return Node(prediction=obj["prediction"], class_counts=obj.get("counts"))
        if "threshold" in obj:
            return Node(feature_index=obj["feature_index"], feature_name=obj.get("feature_name"),
                        threshold=obj["threshold"],
                        left=DecisionTreeImproved._dict_to_node(obj["left"]),
                        right=DecisionTreeImproved._dict_to_node(obj["right"]))
        children = {k: DecisionTreeImproved._dict_to_node(v) for k, v in obj["children"].items()}
        return Node(feature_index=obj["feature_index"], feature_name=obj.get("feature_name"), children=children)

    # simple reduced-error pruning
    def _prune_reduced_error(self, node: Node, X_val: np.ndarray, y_val: np.ndarray):
        if node.is_leaf() or X_val.size == 0:
            return
        # split validation set down current node
        if node.threshold is not None:
            mask = X_val[:, node.feature_index].astype(float) <= node.threshold
            self._prune_reduced_error(node.left,  X_val[mask],  y_val[mask])
            self._prune_reduced_error(node.right, X_val[~mask], y_val[~mask])
        else:
            for val, child in (node.children or {}).items():
                m = (X_val[:, node.feature_index] == val)
                self._prune_reduced_error(child, X_val[m], y_val[m])

        # try turning this subtree into a leaf; keep it if validation accuracy does not drop
        before = (self.predict(X_val) == y_val).mean() if X_val.size else 1.0
        backup = self._node_to_dict(node)
        leaf = self._leaf(y_val if y_val.size else np.array([list(node.class_counts.keys())[0]]))
        # replace node content with leaf
        node.__dict__.update(leaf.__dict__)
        after = (self.predict(X_val) == y_val).mean() if X_val.size else 1.0
        if after + 1e-12 < before:  # revert if worse
            restored = self._dict_to_node(backup)
            node.__dict__.update(restored.__dict__)

# --------------------------- tiny example ---------------------------

if __name__ == "__main__":
    # The original toy dataset (categoricals)
    data = [
        [0,0,0,0,'no'],
        [0,0,0,1,'no'],
        [0,1,0,1,'yes'],
        [0,1,1,0,'yes'],
        [0,0,0,0,'no'],
        [1,0,0,0,'no'],
        [1,0,0,1,'no'],
        [1,1,1,1,'yes'],
        [1,0,1,2,'yes'],
        [1,0,1,2,'yes'],
        [2,0,1,2,'yes'],
        [2,0,1,1,'yes'],
        [2,1,0,1,'yes'],
        [2,1,0,2,'yes'],
        [2,0,0,0,'no'],
    ]
    X = [r[:-1] for r in data]
    y = [r[-1]  for r in data]
    # English feature names
    labels = ["Age", "Employed", "OwnsHouse", "CreditStatus"]

    tree = DecisionTreeImproved(
        criterion="gain_ratio",      # try 'entropy' or 'gini'
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_gain=1e-6
    ).fit(X, y, feature_names=labels)

    print(tree.export_text())
    print("Predict [Age=2, Employed=1, OwnsHouse=0, CreditStatus=1]:",
          tree.predict([[2,1,0,1]])[0])
    tree.save("loan_tree.pkl")
