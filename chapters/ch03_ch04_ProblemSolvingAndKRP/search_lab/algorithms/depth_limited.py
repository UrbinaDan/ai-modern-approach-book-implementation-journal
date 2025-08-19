# search_lab/algorithms/depth_limited.py
# This code implements Depth-Limited Search (DLS) for AI search problems, allowing a maximum depth limit.
from __future__ import annotations
from ..core.node import Node
from ..core.metrics import SearchResult, MeasuredRun
from ..core.problem import Problem
from ..core.utils import reconstruct_path

def depth_limited_search(problem: Problem, limit:int) -> SearchResult:
    name = f"DLS(l={limit})"
    expanded = 0
    cutoff = False

    def rec(node: Node, depth:int):
        nonlocal expanded, cutoff
        if problem.is_goal(node.state): return node
        if depth == limit:
            cutoff = True
            return None
        expanded += 1
        for child in node.expand(problem):
            ans = rec(child, depth+1)
            if ans is not None: return ans
        return None

    with MeasuredRun() as meter:
        result = rec(Node(problem.initial_state()), 0)
        if result:
            a,c = reconstruct_path(result)
            return SearchResult(name, True, a, c, expanded, meter.elapsed, meter.peak_kb)
        return SearchResult(name, False if not cutoff else False, [], float("inf"), expanded, meter.elapsed, meter.peak_kb)
