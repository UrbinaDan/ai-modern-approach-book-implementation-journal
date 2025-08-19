# search_lab/algorithms/rbfs.py
# This code implements Recursive Best-First Search (RBFS), which is a memory-efficient search algorithm that uses recursion to explore the search space.
# It is particularly useful for problems with large search spaces where traditional best-first search would consume too much memory.
# RBFS maintains a limited memory footprint by only storing the current path and its cost, making it suitable for problems with high branching factors.
# The algorithm recursively explores the most promising nodes based on their estimated cost, backtracking when necessary.
from __future__ import annotations
from typing import Tuple
from ..core.node import Node
from ..core.metrics import SearchResult, MeasuredRun
from ..core.problem import Problem
from ..core.utils import reconstruct_path

def rbfs(problem: Problem) -> SearchResult:
    name = "RBFS"
    expanded = 0

    def rbfs_rec(node: Node, flimit: float) -> Tuple[Node | None, float]:
        nonlocal expanded
        f = node.path_cost + problem.heuristic(node.state)
        if problem.is_goal(node.state): return node, f
        successors = list(node.expand(problem))
        if not successors: return None, float("inf")
        for s in successors:
            s.f = s.path_cost + problem.heuristic(s.state)
        while True:
            successors.sort(key=lambda n: n.f)
            best = successors[0]
            if best.f > flimit: return None, best.f
            alternative = successors[1].f if len(successors) > 1 else float("inf")
            expanded += 1
            result, best.f = rbfs_rec(best, min(flimit, alternative))
            if result is not None: return result, best.f

    with MeasuredRun() as meter:
        n, _ = rbfs_rec(Node(problem.initial_state()), float("inf"))
        if n:
            a,c = reconstruct_path(n)
            return SearchResult(name, True, a, c, expanded, meter.elapsed, meter.peak_kb)
        return SearchResult(name, False, [], float("inf"), expanded, meter.elapsed, meter.peak_kb)
