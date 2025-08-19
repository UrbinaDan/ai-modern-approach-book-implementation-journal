# search_lab/algorithms/greedy.py
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

def greedy_best_first_search(problem, max_expansions: Optional[int] = None):
    h = _heuristic_from_problem(problem) or (lambda n: 0.0)
    # greedy: f = 0 + h
    return best_first_search(problem, f=lambda n: 0.0, h=h, name="Greedy", max_expansions=max_expansions)
