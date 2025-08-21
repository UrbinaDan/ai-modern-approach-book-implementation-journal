from __future__ import annotations
from typing import Optional, Tuple
from ..core.node import Node
from ..core.problem import Problem
from ..core.utils import reconstruct_path
from ..core.metrics import SearchResult, MeasuredRun

def _t(meter: MeasuredRun) -> float | None:
    # Robust against older/newer MeasuredRun versions
    return getattr(meter, "time_s", getattr(meter, "elapsed", None))

def iterative_deepening_search(problem: Problem, max_depth: int = 64) -> SearchResult:
    """
    Iterative Deepening Search (tree-like). Repeats a depth-limited DFS with limits 0..max_depth.
    Expansion count = number of nodes we expand (i.e., call .expand on).
    """
    name = "IDS"
    expanded_total = 0

    with MeasuredRun() as meter:
        root = Node(problem.initial_state())
        if problem.is_goal(root.state):
            actions, cost = reconstruct_path(root)
            return SearchResult(name, True, actions, cost, 0, _t(meter), meter.peak_kb)

        for limit in range(max_depth + 1):
            expanded_this_iter = 0

            def on_path(n: Node, s) -> bool:
                cur = n
                while cur is not None:
                    if cur.state == s:
                        return True
                    cur = cur.parent
                return False

            def recursive_dls(node: Node, depth_left: int) -> Tuple[Optional[Node], bool]:
                nonlocal expanded_this_iter
                if problem.is_goal(node.state):
                    return node, False
                if depth_left == 0:
                    return None, True

                children = list(node.expand(problem))
                expanded_this_iter += 1

                cutoff_any = False
                for child in children:
                    if on_path(node, child.state):
                        continue
                    goal, cutoff = recursive_dls(child, depth_left - 1)
                    if goal is not None:
                        return goal, False
                    if cutoff:
                        cutoff_any = True
                return None, cutoff_any

            goal_node, cutoff_hit = recursive_dls(root, limit)
            expanded_total += expanded_this_iter

            if goal_node is not None:
                actions, cost = reconstruct_path(goal_node)
                return SearchResult(name, True, actions, cost, expanded_total, _t(meter), meter.peak_kb)

            if not cutoff_hit:
                break  # fully explored up to `limit`; nothing deeper

    return SearchResult(name, False, [], float("inf"), expanded_total, None, None)
