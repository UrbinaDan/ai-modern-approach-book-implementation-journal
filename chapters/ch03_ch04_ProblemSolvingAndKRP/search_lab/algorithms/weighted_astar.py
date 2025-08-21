# search_lab/algorithms/weighted_astar.py
# This code implements the Weighted A* search algorithm, which is a variant of A* that uses a weight factor to balance between path cost and heuristic.
from __future__ import annotations
from ..core.node import Node
from ..core.problem import Problem
from ..core.metrics import SearchResult, MeasuredRun
from ..core.utils import reconstruct_path
from ..core.frontiers import PriorityQueue

try:
    from .heuristics import h_sld as default_h
except Exception:  # pragma: no cover
    def default_h(node: Node) -> float:  # type: ignore
        return 0.0

def _t(meter: MeasuredRun) -> float | None:
    return getattr(meter, "time_s", getattr(meter, "elapsed", None))

def weighted_a_star_search(problem: Problem, w: float = 1.5, h=default_h) -> SearchResult:
    """
    Weighted A*: f = g + w*h (w>1 focuses search; not optimal in general).
    """
    name = f"WeightedA*(w={w})"
    start = Node(problem.initial_state())

    reached = {start.state: start}
    frontier = PriorityQueue(key=lambda n: n.path_cost + w * h(n))
    frontier.push(start)
    expanded = 0

    with MeasuredRun() as meter:
        while frontier:
            node = frontier.pop()
            if problem.is_goal(node.state):
                a, c = reconstruct_path(node)
                return SearchResult(name, True, a, c, expanded, _t(meter), meter.peak_kb)

            expanded += 1
            for child in node.expand(problem):
                prev = reached.get(child.state)
                if prev is None or child.path_cost < prev.path_cost:
                    reached[child.state] = child
                    frontier.push(child)

    return SearchResult(name, False, [], float("inf"), expanded, _t(meter), meter.peak_kb)
