# search_lab/algorithms/dfs.py
# This code implements Depth-First Search (DFS) using a LIFO stack to explore nodes in a search tree.
from __future__ import annotations
from ..core.frontiers import LIFOStack
from ..core.node import Node
from ..core.metrics import SearchResult, MeasuredRun
from ..core.problem import Problem
from ..core.utils import reconstruct_path

def depth_first_search(problem: Problem, cycle_check_depth:int=3) -> SearchResult:
    name = "DFS"
    frontier = LIFOStack()
    frontier.push(Node(problem.initial_state()))
    expanded = 0

    def is_cycle(n):
        # check last few ancestors
        seen = set()
        k, cur = 0, n
        while cur and k < cycle_check_depth:
            if cur.state in seen: return True
            seen.add(cur.state)
            cur = cur.parent; k += 1
        return False

    with MeasuredRun() as meter:
        while len(frontier):
            node = frontier.pop()
            if problem.is_goal(node.state):
                a,c = reconstruct_path(node)
                return SearchResult(name, True, a, c, expanded, meter.elapsed, meter.peak_kb)
            expanded += 1
            for child in node.expand(problem):
                if not is_cycle(child):
                    frontier.push(child)

    return SearchResult(name, False, [], float("inf"), expanded, meter.elapsed, meter.peak_kb)
