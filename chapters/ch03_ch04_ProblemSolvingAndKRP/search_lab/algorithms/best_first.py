from __future__ import annotations
from typing import Callable, Optional
from ..core.frontiers import PriorityQueue
from ..core.node import Node
from ..core.metrics import SearchResult, MeasuredRun
from ..core.utils import reconstruct_path
from ..core.problem import Problem

def best_first_search(
    problem: Problem,
    f: Callable[[Node], float],
    name: str = "BestFirst",
    h: Optional[Callable[[Node], float]] = None,
    max_expansions: Optional[int] = None,   # ← NEW
) -> SearchResult:
    root = Node(problem.initial_state())

    def fscore(n: Node) -> float:
        base = float(f(n))
        if h is None:
            return base
        hv = h(n)
        return base + (0.0 if hv is None else float(hv))

    frontier = PriorityQueue(key=fscore)
    frontier.push(root)

    reached = {root.state: root}
    expanded = 0

    with MeasuredRun() as meter:
        if problem.is_goal(root.state):
            actions, cost = reconstruct_path(root)
            return SearchResult(name, True, actions, cost, 0, meter.elapsed, meter.peak_kb)

        while frontier:
            # expansion cap
            if max_expansions is not None and expanded >= max_expansions:
                return SearchResult(name, False, [], float("inf"), expanded, meter.elapsed, meter.peak_kb)

            node = frontier.pop()
            if problem.is_goal(node.state):
                actions, cost = reconstruct_path(node)
                return SearchResult(name, True, actions, cost, expanded, meter.elapsed, meter.peak_kb)

            expanded += 1
            for child in node.expand(problem):
                prev = reached.get(child.state)
                if prev is None or child.path_cost < prev.path_cost:
                    reached[child.state] = child
                    frontier.push(child)

    return SearchResult(name, False, [], float("inf"), expanded, meter.elapsed, meter.peak_kb)
