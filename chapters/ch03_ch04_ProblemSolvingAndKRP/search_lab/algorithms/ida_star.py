# search_lab/algorithms/ida_star.py
# This code implements the IDA* search algorithm, which is an iterative deepening version of A* that uses a depth-first search strategy.
# It combines the benefits of A*'s heuristic guidance with the space efficiency of depth-first search.
# IDA* is particularly useful for problems with large search spaces where memory is a constraint.
from __future__ import annotations
from typing import Optional, Tuple
from ..core.node import Node
from ..core.problem import Problem
from ..core.utils import reconstruct_path
from ..core.metrics import SearchResult, MeasuredRun

# Default heuristic: straight-line distance if available; else 0
try:
    from .heuristics import h_sld as default_h
except Exception:  # pragma: no cover
    def default_h(node: Node) -> float:  # type: ignore
        return 0.0

def _t(meter: MeasuredRun) -> float | None:
    return getattr(meter, "time_s", getattr(meter, "elapsed", None))

def ida_star_search(problem: Problem, h=default_h, max_iters: int = 10_000) -> SearchResult:
    """
    IDA*: Iterative Deepening A* (tree-like).
    - Bounded by f = g + h; increases the bound to the smallest f that exceeded the previous bound.
    - Uses little memory but may revisit nodes many times.
    """
    name = "IDA*"
    expanded = 0

    with MeasuredRun() as meter:
        root = Node(problem.initial_state())
        if problem.is_goal(root.state):
            a, c = reconstruct_path(root)
            return SearchResult(name, True, a, c, 0, _t(meter), meter.peak_kb)

        bound = h(root)
        iters = 0

        def search(node: Node, bound: float) -> Tuple[Optional[Node], float]:
            nonlocal expanded
            f = node.path_cost + h(node)
            if f > bound:
                return None, f
            if problem.is_goal(node.state):
                return node, f

            min_excess = float("inf")
            children = list(node.expand(problem))
            expanded += 1

            for child in children:
                solution, t = search(child, bound)
                if solution is not None:
                    return solution, t
                if t < min_excess:
                    min_excess = t
            return None, min_excess

        while iters < max_iters:
            iters += 1
            goal, next_bound = search(root, bound)
            if goal is not None:
                a, c = reconstruct_path(goal)
                return SearchResult(name, True, a, c, expanded, _t(meter), meter.peak_kb)
            if next_bound == float("inf"):
                break
            bound = next_bound

    return SearchResult(name, False, [], float("inf"), expanded, _t(meter), meter.peak_kb)
