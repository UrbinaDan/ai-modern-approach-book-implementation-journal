# search_lab/algorithms/rbfs.py
# This code implements Recursive Best-First Search (RBFS), which is a memory-efficient search algorithm that uses recursion to explore the search space.
# It is particularly useful for problems with large search spaces where traditional best-first search would consume too much memory.
# RBFS maintains a limited memory footprint by only storing the current path and its cost, making it suitable for problems with high branching factors.
# The algorithm recursively explores the most promising nodes based on their estimated cost, backtracking when necessary.
from __future__ import annotations
from typing import Tuple, Optional
from ..core.node import Node
from ..core.problem import Problem
from ..core.utils import reconstruct_path
from ..core.metrics import SearchResult, MeasuredRun

try:
    from .heuristics import h_sld as default_h
except Exception:  # pragma: no cover
    def default_h(node: Node) -> float:  # type: ignore
        return 0.0

def _t(meter: MeasuredRun) -> float | None:
    return getattr(meter, "time_s", getattr(meter, "elapsed", None))

def recursive_best_first_search(problem: Problem, h=default_h) -> SearchResult:
    """
    RBFS: Recursive Best-First Search (linear memory).
    Mimics best-first with an f-limit; backs up best-alternative f-values on unwind.
    """
    name = "RBFS"
    expanded = 0

    with MeasuredRun() as meter:
        root = Node(problem.initial_state())
        if problem.is_goal(root.state):
            a, c = reconstruct_path(root)
            return SearchResult(name, True, a, c, 0, _t(meter), meter.peak_kb)

        def rbfs(node: Node, f_limit: float) -> Tuple[Optional[Node], float]:
            nonlocal expanded
            if problem.is_goal(node.state):
                return node, node.path_cost + h(node)

            # Generate successors
            children = list(node.expand(problem))
            expanded += 1
            if not children:
                return None, float("inf")

            # Each child carries an 'f' value
            for c in children:
                c.f = max(c.path_cost + h(c), getattr(node, "f", node.path_cost + h(node)))  # type: ignore

            while True:
                # Best child and best alternative
                children.sort(key=lambda n: n.f)  # type: ignore
                best = children[0]
                if best.f > f_limit:  # type: ignore
                    return None, best.f  # type: ignore
                alternative = children[1].f if len(children) > 1 else float("inf")  # type: ignore
                result, best.f = rbfs(best, min(f_limit, alternative))  # type: ignore
                if result is not None:
                    return result, best.f

        # Seed f on root
        root.f = root.path_cost + h(root)  # type: ignore
        sol, _ = rbfs(root, float("inf"))
        if sol:
            a, c = reconstruct_path(sol)
            return SearchResult(name, True, a, c, expanded, _t(meter), meter.peak_kb)

    return SearchResult(name, False, [], float("inf"), expanded, _t(meter), meter.peak_kb)
