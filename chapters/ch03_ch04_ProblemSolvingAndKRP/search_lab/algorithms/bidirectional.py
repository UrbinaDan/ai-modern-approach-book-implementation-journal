# search_lab/algorithms/bidirectional.py
# This code implements a bidirectional search algorithm, which is a search strategy that simultaneously explores paths from both the start and goal states.
# It is often used to find the shortest path in problems where the goal state is known.
from __future__ import annotations
from typing import Dict, Tuple, Optional
from ..core.node import Node
from ..core.problem import Problem
from ..core.metrics import SearchResult, MeasuredRun
from ..core.frontiers import PriorityQueue

try:
    from .heuristics import h_sld as default_h
except Exception:  # pragma: no cover
    def default_h(node: Node) -> float:  # type: ignore
        return 0.0

def _t(meter: MeasuredRun) -> float | None:
    return getattr(meter, "time_s", getattr(meter, "elapsed", None))

def _finalize(name: str, meet: Tuple[Node, Node] | None, expanded: int, meter: MeasuredRun) -> SearchResult:
    if meet is None:
        return SearchResult(name, False, [], float("inf"), expanded, _t(meter), meter.peak_kb)
    nf, nb = meet
    total_cost = nf.path_cost + nb.path_cost
    # Actions omitted (requires invertible actions); benchmarks use cost/expansions.
    return SearchResult(name, True, [], float(total_cost), expanded, _t(meter), meter.peak_kb)

def bibf_uniform_cost_search(problem: Problem) -> SearchResult:
    """
    Bidirectional uniform-cost search (for undirected graphs).
    Stops when min-frontier-costs sum >= best meet cost.
    """
    name = "Bidirectional UCS"
    expanded = 0

    with MeasuredRun() as meter:
        start_f = Node(problem.initial_state())
        if problem.is_goal(start_f.state):
            return SearchResult(name, True, [], 0.0, 0, _t(meter), meter.peak_kb)

        # Backwards problem: we "start" at any goal and aim for the original start.
        # For undirected graphs, we can use the same expansions.
        # Build a pseudo-goal test for backward: "state == start"
        class BackProblem(Problem):
            def initial_state(self_non) -> object:
                # We need an actual goal state to seed; try to find one by expanding until we hit a goal?
                # Simpler: assume Problem exposes one goal via attribute .goal (your Romania problem does).
                try:
                    return problem.goal  # type: ignore
                except Exception:
                    raise RuntimeError("Backward search requires a single, accessible goal state via `problem.goal`.")
            def is_goal(self_non, s) -> bool:
                return s == problem.initial_state()
            # Delegate transitions/costs
            def actions(self_non, s): return problem.actions(s)
            def result(self_non, s, a): return problem.result(s, a)
            def cost(self_non, s, a, s2): return problem.cost(s, a, s2)

        problem_b = BackProblem()

        frontier_f = PriorityQueue(key=lambda n: n.path_cost)
        frontier_b = PriorityQueue(key=lambda n: n.path_cost)

        reached_f: Dict[object, Node] = {}
        reached_b: Dict[object, Node] = {}

        frontier_f.push(start_f); reached_f[start_f.state] = start_f
        start_b = Node(problem_b.initial_state())
        frontier_b.push(start_b); reached_b[start_b.state] = start_b

        best_meet: Optional[Tuple[Node, Node]] = None
        best_cost = float("inf")

        while frontier_f and frontier_b:
            top_f = frontier_f.top().path_cost
            top_b = frontier_b.top().path_cost
            if best_meet is not None and (top_f + top_b) >= best_cost:
                break

            # Expand the frontier with smaller top path-cost
            expand_forward = top_f <= top_b
            if expand_forward:
                node = frontier_f.pop()
                expanded += 1
                for child in node.expand(problem):
                    prev = reached_f.get(child.state)
                    if prev is None or child.path_cost < prev.path_cost:
                        reached_f[child.state] = child
                        frontier_f.push(child)
                    # Meet check
                    if child.state in reached_b:
                        total = child.path_cost + reached_b[child.state].path_cost
                        if total < best_cost:
                            best_cost = total
                            best_meet = (child, reached_b[child.state])
            else:
                node = frontier_b.pop()
                expanded += 1
                for child in node.expand(problem_b):
                    prev = reached_b.get(child.state)
                    if prev is None or child.path_cost < prev.path_cost:
                        reached_b[child.state] = child
                        frontier_b.push(child)
                    # Meet check
                    if child.state in reached_f:
                        total = child.path_cost + reached_f[child.state].path_cost
                        if total < best_cost:
                            best_cost = total
                            best_meet = (reached_f[child.state], child)

        return _finalize(name, best_meet, expanded, meter)

def bidirectional_a_star(problem: Problem, h_f=default_h, h_b=default_h) -> SearchResult:
    """
    Bidirectional A* (for undirected graphs).
    Uses f = g + h for each direction; terminates when min f-sum >= best meet cost.
    Defaults to zero heuristics if domain-specific heuristics not supplied for both directions.
    """
    name = "Bidirectional A*"
    expanded = 0

    with MeasuredRun() as meter:
        start_f = Node(problem.initial_state())
        if problem.is_goal(start_f.state):
            return SearchResult(name, True, [], 0.0, 0, _t(meter), meter.peak_kb)

        class BackProblem(Problem):
            def initial_state(self_non) -> object:
                try:
                    return problem.goal  # type: ignore
                except Exception:
                    raise RuntimeError("Backward search requires a single goal via `problem.goal`.")
            def is_goal(self_non, s) -> bool:
                return s == problem.initial_state()
            def actions(self_non, s): return problem.actions(s)
            def result(self_non, s, a): return problem.result(s, a)
            def cost(self_non, s, a, s2): return problem.cost(s, a, s2)

        problem_b = BackProblem()

        frontier_f = PriorityQueue(key=lambda n: n.path_cost + h_f(n))
        frontier_b = PriorityQueue(key=lambda n: n.path_cost + h_b(n))

        reached_f: Dict[object, Node] = {}
        reached_b: Dict[object, Node] = {}

        frontier_f.push(start_f); reached_f[start_f.state] = start_f
        start_b = Node(problem_b.initial_state())
        frontier_b.push(start_b); reached_b[start_b.state] = start_b

        best_meet: Optional[Tuple[Node, Node]] = None
        best_cost = float("inf")

        while frontier_f and frontier_b:
            fF = frontier_f.top().path_cost + h_f(frontier_f.top())
            fB = frontier_b.top().path_cost + h_b(frontier_b.top())
            if best_meet is not None and (fF + fB) >= best_cost:
                break

            if fF <= fB:
                node = frontier_f.pop()
                expanded += 1
                for child in node.expand(problem):
                    prev = reached_f.get(child.state)
                    if prev is None or child.path_cost < prev.path_cost:
                        reached_f[child.state] = child
                        frontier_f.push(child)
                    if child.state in reached_b:
                        total = child.path_cost + reached_b[child.state].path_cost
                        if total < best_cost:
                            best_cost = total
                            best_meet = (child, reached_b[child.state])
            else:
                node = frontier_b.pop()
                expanded += 1
                for child in node.expand(problem_b):
                    prev = reached_b.get(child.state)
                    if prev is None or child.path_cost < prev.path_cost:
                        reached_b[child.state] = child
                        frontier_b.push(child)
                    if child.state in reached_f:
                        total = child.path_cost + reached_f[child.state].path_cost
                        if total < best_cost:
                            best_cost = total
                            best_meet = (reached_f[child.state], child)

        return _finalize(name, best_meet, expanded, meter)
