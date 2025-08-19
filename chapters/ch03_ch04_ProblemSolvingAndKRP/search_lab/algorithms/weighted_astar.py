# search_lab/algorithms/weighted_astar.py
# This code implements the Weighted A* search algorithm, which is a variant of A* that uses a weight factor to balance between path cost and heuristic.
from __future__ import annotations
from .best_first import best_first_search

def weighted_astar(problem, w:float=1.5):
    return best_first_search(problem, f=lambda n: n.path_cost + w*problem.heuristic(n.state), name=f"W-A*(w={w})")
