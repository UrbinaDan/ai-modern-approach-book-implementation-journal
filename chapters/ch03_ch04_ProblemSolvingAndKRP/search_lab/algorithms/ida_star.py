# search_lab/algorithms/ida_star.py
# This code implements the IDA* search algorithm, which is an iterative deepening version of A* that uses a depth-first search strategy.
# It combines the benefits of A*'s heuristic guidance with the space efficiency of depth-first search.
# IDA* is particularly useful for problems with large search spaces where memory is a constraint.
from __future__ import annotations
from typing import Tuple
from ..core.node import Node
from ..core.metrics import SearchResult, MeasuredRun
from ..core.problem import Problem
from ..core.utils import reconstruct_path

def ida_star(problem: Problem) -> SearchResult:
    name = "IDA*"
    start = Node(problem.initial_state())
    bound = problem.heuristic(start.state)
    expanded = 0

    def search(node: Node, bound: float) -> Tuple[float, Node | None]:
        nonlocal expanded
        f = node.path_cost + problem.heuristic(node.state)
        if f > bound: return f, None
        if problem.is_goal(node.state): return f, node
        expanded += 1
        min_bound = float("inf")
        for c in node.expand(problem):
            t, found = search(c, bound)
            if found is not None: return t, found
            if t < min_bound: min_bound = t
        return min_bound, None

    with MeasuredRun() as meter:
        while True:
            t, found = search(start, bound)
            if found is not None:
                a,c = reconstruct_path(found)
                return SearchResult(name, True, a, c, expanded, meter.elapsed, meter.peak_kb)
            if t == float("inf"):
                return SearchResult(name, False, [], float("inf"), expanded, meter.elapsed, meter.peak_kb)
            bound = t
