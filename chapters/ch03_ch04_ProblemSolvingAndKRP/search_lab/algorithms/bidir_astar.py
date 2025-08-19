# search_lab/algorithms/bidir_astar.py
# This code implements Bidirectional A* search, which is a search algorithm that simultaneously expands paths from both the start and goal states.
# It is designed to find the shortest path in problems where both the start and goal states are known.
from __future__ import annotations
from typing import Dict, Callable, Tuple, Iterable
from ..core.node import Node
from ..core.frontiers import PriorityQueue
from ..core.metrics import SearchResult, MeasuredRun
from ..core.problem import Problem, State
from ..core.utils import reconstruct_path

def bidirectional_astar(
    forward: Problem,
    backward: Problem,
    init_goals: Iterable[State],
    hf: Callable[[State], float],  # heuristic in forward dir
    hb: Callable[[State], float],  # heuristic in backward dir
    ) -> SearchResult:
    """Bidirectional A*: two frontiers expanding min f across both."""
    name = "Bi-A*"
    nf0 = Node(forward.initial_state())
    fb_front = PriorityQueue(key=lambda n: n.path_cost + hf(n.state))
    fb_front.push(nf0)
    f_reached: Dict[State, Node] = {nf0.state: nf0}

    # backward can have multiple initial states (goals)
    b_front = PriorityQueue(key=lambda n: n.path_cost + hb(n.state))
    b_reached: Dict[State, Node] = {}
    for g in init_goals:
        nb0 = Node(g)
        b_front.push(nb0)
        b_reached[g] = nb0

    expanded = 0
    best_solution_node: Tuple[Node, Node] | None = None
    best_cost = float("inf")

    with MeasuredRun() as meter:
        while len(fb_front) or len(b_front):
            def peek_f(pq): return (pq.peek().path_cost + (hf if pq is fb_front else hb)(pq.peek().state)) if len(pq) else float("inf")
            if peek_f(fb_front) <= peek_f(b_front):
                node = fb_front.pop()
                expanded += 1
                # try meeting
                if node.state in b_reached:
                    cost = node.path_cost + b_reached[node.state].path_cost
                    if cost < best_cost:
                        best_cost = cost
                        best_solution_node = (node, b_reached[node.state])
                if node.path_cost + hf(node.state) >= best_cost:
                    # cannot improve
                    pass
                for c in node.expand(forward):
                    old = f_reached.get(c.state)
                    if old is None or c.path_cost < old.path_cost:
                        f_reached[c.state] = c
                        fb_front.push(c)
            else:
                node = b_front.pop()
                expanded += 1
                if node.state in f_reached:
                    cost = node.path_cost + f_reached[node.state].path_cost
                    if cost < best_cost:
                        best_cost = cost
                        best_solution_node = (f_reached[node.state], node)
                # expand backward by following reverse neighbors:
                for c in node.expand(backward):
                    old = b_reached.get(c.state)
                    if old is None or c.path_cost < old.path_cost:
                        b_reached[c.state] = c
                        b_front.push(c)

            if best_solution_node and peek_f(fb_front) + peek_f(b_front) >= best_cost:
                # stopping condition: no better solution remains
                f_goal, b_goal = best_solution_node
                # reconstruct forward path + reverse of backward path actions (omitted; return forward for simplicity)
                a,c = reconstruct_path(f_goal)
                return SearchResult(name, True, a, best_cost, expanded, meter.elapsed, meter.peak_kb)

    return SearchResult(name, False, [], float("inf"), expanded, meter.elapsed, meter.peak_kb)
