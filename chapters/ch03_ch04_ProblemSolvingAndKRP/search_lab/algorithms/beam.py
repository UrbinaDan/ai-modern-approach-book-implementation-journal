from __future__ import annotations
from typing import Callable, List, Set
from ..core.node import Node
from ..core.problem import Problem
from ..core.utils import reconstruct_path
from ..core.metrics import SearchResult, MeasuredRun

# Default heuristic (falls back to zero if not available)
try:
    from .heuristics import h_sld as default_h
except Exception:  # pragma: no cover
    def default_h(node: Node) -> float:
        return 0.0

def _t(meter: MeasuredRun) -> float | None:
    return getattr(meter, "time_s", getattr(meter, "elapsed", None))

def beam_search(
    problem: Problem,
    k: int = 10,
    f: Callable[[Node], float] | None = None,
) -> SearchResult:
    """
    Beam Search (informed, incomplete). Keeps only top-k nodes (by f) per layer.
    Returns SearchResult with expansion count ~ nodes expanded per layer.
    """
    name = f"Beam(k={k})"
    if f is None:
        f = default_h

    expanded = 0
    reached: Set = set()

    with MeasuredRun() as meter:
        root = Node(problem.initial_state())
        if problem.is_goal(root.state):
            actions, cost = reconstruct_path(root)
            return SearchResult(name, True, actions, cost, 0, _t(meter), meter.peak_kb)

        frontier: List[Node] = [root]
        reached.add(root.state)

        while frontier:
            expanded += len(frontier)
            children: List[Node] = []

            for node in frontier:
                for child in node.expand(problem):
                    if child.state in reached:
                        continue
                    if problem.is_goal(child.state):
                        actions, cost = reconstruct_path(child)
                        return SearchResult(name, True, actions, cost, expanded, _t(meter), meter.peak_kb)
                    reached.add(child.state)
                    children.append(child)

            if not children:
                break

            children.sort(key=f)
            frontier = children[:k]

    return SearchResult(name, False, [], float("inf"), expanded, None, None)
