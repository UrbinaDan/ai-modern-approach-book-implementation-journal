from __future__ import annotations
from typing import Optional
from ..core.frontiers import FIFOQueue
from ..core.node import Node
from ..core.metrics import SearchResult, MeasuredRun
from ..core.problem import Problem
from ..core.utils import reconstruct_path

def breadth_first_search(problem: Problem, max_expansions: Optional[int] = None) -> SearchResult:
    name = "BFS"
    root = Node(problem.initial_state())
    frontier = FIFOQueue()
    frontier.push(root)
    reached = {root.state}
    expanded = 0

    with MeasuredRun() as meter:
        if problem.is_goal(root.state):
            actions, cost = reconstruct_path(root)
            return SearchResult(name, True, actions, cost, 0, meter.elapsed, meter.peak_kb)

        while frontier:
            if max_expansions is not None and expanded >= max_expansions:
                return SearchResult(name, False, [], float("inf"), expanded, meter.elapsed, meter.peak_kb)

            node = frontier.pop()
            expanded += 1
            for child in node.expand(problem):
                if child.state not in reached:
                    reached.add(child.state)
                    if problem.is_goal(child.state):
                        actions, cost = reconstruct_path(child)
                        return SearchResult(name, True, actions, cost, expanded, meter.elapsed, meter.peak_kb)
                    frontier.push(child)

    return SearchResult(name, False, [], float("inf"), expanded, meter.elapsed, meter.peak_kb)
