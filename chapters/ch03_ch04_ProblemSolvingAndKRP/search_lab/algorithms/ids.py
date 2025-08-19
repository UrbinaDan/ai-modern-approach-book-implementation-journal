# search_lab/algorithms/ids.py
# This code implements Iterative Deepening Search (IDS) by reusing the depth-limited search function.
from __future__ import annotations
from .depth_limited import depth_limited_search
from ..core.metrics import SearchResult, MeasuredRun
from ..core.utils import reconstruct_path
from ..core.node import Node
from ..core.problem import Problem

def iterative_deepening_search(problem: Problem, max_depth:int=10) -> SearchResult:
    expanded_total = 0
    with MeasuredRun() as meter:
        for l in range(max_depth+1):
            res = depth_limited_search(problem, l)
            expanded_total += res.nodes_expanded
            if res.found:
                return SearchResult("IDS", True, res.path, res.path_cost, expanded_total, meter.elapsed, res.peak_mem_kb)
    return SearchResult("IDS", False, [], float("inf"), expanded_total, meter.elapsed, None)
