# search_lab/algorithms/beam.py
# This code implements Beam Search, which is a heuristic search algorithm that explores a graph by expanding the most promising nodes.
from __future__ import annotations
from typing import List, Dict, Set
from ..core.node import Node
from ..core.metrics import SearchResult, MeasuredRun
from ..core.problem import Problem

def beam_search(problem: Problem, k:int=30) -> SearchResult:
    """Keeps only k best nodes by f=h(n) each layer (incomplete; fast)."""
    name = f"Beam(k={k})"
    layer: List[Node] = [Node(problem.initial_state())]
    expanded = 0
    with MeasuredRun() as meter:
        while layer:
            # goal test on layer
            for n in layer:
                if problem.is_goal(n.state):
                    return SearchResult(name, True, n.solution(), n.path_cost, expanded, meter.elapsed, meter.peak_kb)
            # expand all in layer
            children = []
            for n in layer:
                expanded += 1
                children.extend(n.expand(problem))
            # rank by heuristic and keep top k
            children.sort(key=lambda c: problem.heuristic(c.state))
            layer = children[:k]
    return SearchResult(name, False, [], float("inf"), expanded, meter.elapsed, meter.peak_kb)
