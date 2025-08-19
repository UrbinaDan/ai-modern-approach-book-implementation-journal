# search_lab/algorithms/astar.py
from __future__ import annotations
from typing import Callable, Optional
from .best_first import best_first_search
from ..core.node import Node

def _heuristic_from_problem(problem) -> Optional[Callable[[Node], float]]:
    if hasattr(problem, "heuristic"):
        def h(n: Node) -> float:
            state = getattr(n, "state", n)
            val = problem.heuristic(state)
            return 0.0 if val is None else float(val)
        return h
    return None

def a_star_search(problem, heuristic: Optional[Callable[[Node], float]] = None, max_expansions: Optional[int] = None):
    h = heuristic or _heuristic_from_problem(problem)
    return best_first_search(problem, f=lambda n: n.path_cost, h=h, name="A*", max_expansions=max_expansions)
