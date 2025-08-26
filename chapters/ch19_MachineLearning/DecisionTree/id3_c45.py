import numpy as np

class NotFittedError(Exception):
    """Raised if estimator is used before fitting."""
    pass


class DecisionTree:
    """
    Minimal decision-tree learner with ID3 (information gain) and C4.5 (gain ratio),
    plus verbose training logs and a path explainer for predictions.

    Parameters
    ----------
    criterion : {"ID3", "C4.5"}
        Split selection rule. "ID3" uses information gain, "C4.5" uses gain ratio.
    feature_names : list[str] | None
        Names for columns in X (for readable trees & explanations).
    verbose : bool
        If True, print entropy/IG/GR at each node while fitting.
    """

    def __init__(self, criterion="C4.5", feature_names=None, verbose=False):
        if criterion not in ("ID3", "C4.5"):
            raise ValueError("criterion must be 'ID3' or 'C4.5'")
        self.criterion = criterion
        self.feature_names = feature_names
        self.verbose = verbose

        self._tree = None
        self._feat_to_idx = None  # filled in fit()

    # ---------- core utilities ----------

    @staticmethod
    def _entropy(y):
        """Shannon entropy of label vector y."""
        if y.size == 0:
            return 0.0
        values, counts = np.unique(y, return_counts=True)
        probs = counts.astype(float) / y.size
        # avoid log2(0)
        return float(-(probs * np.log2(probs + 1e-15)).sum())

    def _split_dataset(self, X, y, col_idx, value):
        """Return subset (X_sub, y_sub) where X[:, col_idx] == value, and X_sub drops that column."""
        mask = (X[:, col_idx] == value)
        X_sub = X[mask][:, [j for j in range(X.shape[1]) if j != col_idx]]
        y_sub = y[mask]
        return X_sub, y_sub

    def _score_features(self, X, y, feat_names, H_parent):
        """
        Compute per-feature metrics at a node.

        Returns: list of dicts with keys:
            index, name, ig, split_info, gain_ratio, value_counts (dict[value] -> count)
        """
        n_features = X.shape[1]
        metrics = []
        N = len(y)

        for i in range(n_features):
            col = X[:, i]
            values, counts = np.unique(col, return_counts=True)
            new_entropy = 0.0
            split_info = 0.0

            for v, c in zip(values, counts):
                p = c / float(N)
                # entropy of that branch
                y_sub = y[col == v]
                new_entropy += p * self._entropy(y_sub)
                # intrinsic info (penalizes many-way splits)
                split_info -= p * np.log2(p + 1e-15)

            ig = H_parent - new_entropy
            if split_info == 0.0:
                gain_ratio = float("-inf")  # avoid choosing degenerate splits under C4.5
            else:
                gain_ratio = ig / split_info

            metrics.append({
                "index": i,
                "name": feat_names[i],
                "ig": ig,
                "split_info": split_info,
                "gain_ratio": gain_ratio,
                "value_counts": {str(v): int(c) for v, c in zip(values, counts)}
            })

        return metrics

    # ---------- training ----------

    def _create_tree(self, X, y, feat_names, depth=0):
        # If all labels equal → leaf
        unique_labels = np.unique(y)
        if unique_labels.size == 1:
            return unique_labels[0]

        # No features left → majority vote
        if X.shape[1] == 0:
            # majority label
            values, counts = np.unique(y, return_counts=True)
            return values[np.argmax(counts)]

        # Compute metrics & choose best feature according to criterion
        H_parent = self._entropy(y)
        metrics = self._score_features(X, y, feat_names, H_parent)
        if self.criterion == "ID3":
            key = "ig"
        else:
            key = "gain_ratio"

        # choose best; tie-break by larger IG to be sane
        metrics_sorted = sorted(metrics, key=lambda m: (m[key], m["ig"]), reverse=True)
        best = metrics_sorted[0]
        best_idx = best["index"]
        best_name = best["name"]

        if self.verbose:
            pos = (y == np.unique(y)[0]).sum()  # quick count print; not used for logic
            neg = len(y) - pos
            indent = "|  " * depth
            print(f"{indent}Node depth={depth}, n={len(y)}, H={H_parent:.3f}")
            for m in metrics_sorted:
                ig = m['ig']; si = m['split_info']; gr = m['gain_ratio']
                vc = ", ".join([f"{val}:{cnt}" for val, cnt in m['value_counts'].items()])
                if self.criterion == "ID3":
                    print(f"{indent}  - {m['name']:<12} IG={ig:.3f}  (values: {vc})")
                else:
                    print(f"{indent}  - {m['name']:<12} IG={ig:.3f} | SplitInfo={si:.3f} | GR={gr:.3f}  (values: {vc})")
            print(f"{indent}=> choose '{best_name}' by {self.criterion}\n")

        # Build children
        tree = {best_name: {}}
        col = X[:, best_idx]
        for v in np.unique(col):
            X_sub, y_sub = self._split_dataset(X, y, best_idx, v)
            next_feat_names = [fn for j, fn in enumerate(feat_names) if j != best_idx]
            tree[best_name][v] = self._create_tree(X_sub, y_sub, next_feat_names, depth + 1)

        return tree

    def fit(self, X, y):
        # array-ize and basic checks
        X = np.array(X, dtype=object)
        y = np.array(y, dtype=object)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError("y must be 1D and len(y) == X.shape[0]")

        # feature names
        if self.feature_names is None:
            self.feature_names = [f"x{i}" for i in range(X.shape[1])]
        if len(self.feature_names) != X.shape[1]:
            raise ValueError("feature_names length must match X.shape[1]")

        self._feat_to_idx = {name: i for i, name in enumerate(self.feature_names)}
        self._tree = self._create_tree(X, y, self.feature_names, depth=0)
        return self

    # ---------- inference ----------

    def _classify_one(self, tree, sample):
        """Recursive helper for predict/explain."""
        if not isinstance(tree, dict):
            return tree

        # get node feature and its children
        feature = next(iter(tree.keys()))
        children = tree[feature]
        col_idx = self._feat_to_idx[feature]
        val = sample[col_idx]

        # follow matching child if present; else majority fallback
        if val in children:
            return self._classify_one(children[val], sample)
        else:
            # unseen value at test-time → majority vote among available branches
            leaves = []
            def collect_labels(sub):
                if isinstance(sub, dict):
                    for kid in sub.values():
                        collect_labels(kid)
                else:
                    leaves.append(sub)
            collect_labels(children)
            if not leaves:
                # degenerate case
                return None
            values, counts = np.unique(np.array(leaves, dtype=object), return_counts=True)
            return values[np.argmax(counts)]

    def predict(self, X):
        if self._tree is None:
            raise NotFittedError("Estimator not fitted, call `fit` first.")
        X = np.array(X, dtype=object)
        if X.ndim == 1:
            return np.array([self._classify_one(self._tree, X)], dtype=object)
        return np.array([self._classify_one(self._tree, row) for row in X], dtype=object)

    # ---------- explainability helpers ----------

    def explain_one(self, sample, print_path=True):
        """
        Return the root→leaf path taken to classify `sample`.
        Each step is (feature_name, sample_value, next_is_leaf?, prediction_if_leaf).

        If print_path=True, pretty-print the path.
        """
        if self._tree is None:
            raise NotFittedError("Estimator not fitted, call `fit` first.")

        sample = np.array(sample, dtype=object)
        path = []
        node = self._tree

        while isinstance(node, dict):
            feat = next(iter(node.keys()))
            children = node[feat]
            col_idx = self._feat_to_idx[feat]
            val = sample[col_idx]
            next_node = children.get(val, None)

            if next_node is None:
                # unseen branch; fall back like predict()
                # compute majority of reachable leaves
                leaves = []
                def collect_labels(sub):
                    if isinstance(sub, dict):
                        for kid in sub.values():
                            collect_labels(kid)
                    else:
                        leaves.append(sub)
                collect_labels(children)
                maj = None
                if leaves:
                    values, counts = np.unique(np.array(leaves, dtype=object), return_counts=True)
                    maj = values[np.argmax(counts)]
                path.append((feat, val, True, maj))
                node = maj
                break
            else:
                if isinstance(next_node, dict):
                    path.append((feat, val, False, None))
                    node = next_node
                else:
                    path.append((feat, val, True, next_node))
                    node = next_node
                    break

        if print_path:
            print("Explanation (root → leaf):")
            for i, (feat, val, is_leaf, pred) in enumerate(path):
                arrow = "└─" if i == len(path) - 1 else "├─"
                if is_leaf:
                    print(f"{arrow} {feat} = {val}  →  predict {pred}")
                else:
                    print(f"{arrow} {feat} = {val}")
        return path

    # ---------- plotting adapter ----------

    def show(self):
        if self._tree is None:
            raise NotFittedError("Estimator not fitted, call `fit` first.")
        # Lazy import to keep deps optional
        import treePlotter
        treePlotter.createPlot(self._tree)
