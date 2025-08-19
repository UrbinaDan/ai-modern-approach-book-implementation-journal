# This code implements Uniform Cost Search (UCS) by reusing the generic best-first search function.
# search_lab/algorithms/ucs.py
from __future__ import annotations
from typing import Optional
from .best_first import best_first_search

def uniform_cost_search(problem, max_expansions: Optional[int] = None):
    return best_first_search(problem, f=lambda n: n.path_cost, name="UCS", h=None, max_expansions=max_expansions)
