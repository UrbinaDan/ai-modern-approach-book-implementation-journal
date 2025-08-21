# search_lab/algorithms/sma_star.py
# This code implements Simplified Memory-Bounded A* (SMA*), which is a variant of the A* search algorithm that uses a limited amount of memory.
# SMA* is designed to handle large search spaces by dropping the worst leaf nodes when the memory limit is reached.
# It maintains a balance between exploration and exploitation by backing up f estimates shallowly, allowing it to find solutions in memory-constrained environments.
from __future__ import annotations
from typing import Dict, List, Optional
from ..core.node import Node
from ..core.problem import Problem
from ..core.metrics import SearchResult, MeasuredRun
from ..core.utils import reconstruct_path

from heapq import heappush, heappop

try:
    from .heuristics import h_sld as default_h
except Exception:  # pragma: no cover
    def default_h(node: Node) -> float:  # type: ignore
        return 0.0

def _t(meter: MeasuredRun) -> float | None:
    return getattr(meter, "time_s", getattr(meter, "elapsed", None))

class _PQ:
    def __init__(self): self.h=[]; self.c=0
    def push(self, prio: float, node: Node): heappush(self.h,(prio,self.c,node)); self.c+=1
    def pop(self)->Node: return heappop(self.h)[2]
    def top_prio(self)->float: return self.h[0][0]
    def __len__(self): return len(self.h)
    def remove_worst(self)->Node:
        # Grab worst by scanning (small fronts); OK for teaching-scale problems
        idx=max(range(len(self.h)),key=lambda i:self.h[i][0])
        _,_,n=self.h.pop(idx)
        return n

def sma_star_search(problem: Problem, h=default_h, max_nodes: int = 1000) -> SearchResult:
    """
    SMA*: Memory-bounded A*. Keeps at most `max_nodes` leaves.
    Drops the worst f-leaf when memory is full, backing up f to its parent.
    """
    name = f"SMA*(N={max_nodes})"
    expanded = 0

    with MeasuredRun() as meter:
        root = Node(problem.initial_state())
        if problem.is_goal(root.state):
            a,c=reconstruct_path(root)
            return SearchResult(name, True, a, c, 0, _t(meter), meter.peak_kb)

        frontier = _PQ()
        fval: Dict[Node, float] = {}
        children: Dict[Node, List[Node]] = {}

        def f(n: Node) -> float:
            return n.path_cost + h(n)

        fval[root] = f(root)
        frontier.push(fval[root], root)

        best_solution: Optional[Node] = None
        best_solution_cost = float("inf")

        def backup(n: Node):
            # Parent's f is min child f, recursively propagate
            p = n.parent
            while p is not None:
                if children.get(p):
                    best = min(fval[c] for c in children[p])
                    if best > fval.get(p, float("inf")):
                        fval[p] = best
                    else:
                        fval[p] = best
                p = p.parent

        while frontier:
            node = frontier.pop()

            if problem.is_goal(node.state):
                a,c = reconstruct_path(node)
                return SearchResult(name, True, a, c, expanded, _t(meter), meter.peak_kb)

            # expand
            succs = list(node.expand(problem))
            expanded += 1
            children[node] = succs

            if not succs:
                fval[node] = float("inf")
                backup(node)
                continue

            for s in succs:
                fval[s] = max(f(s), fval.get(node, f(node)))  # monotone backup
                frontier.push(fval[s], s)

            # enforce memory bound over leaves
            while len(frontier) > max_nodes:
                worst = frontier.remove_worst()
                # Forget worst leaf; back up info to parent
                fval[worst] = fval.get(worst, f(worst))
                backup(worst)

    return SearchResult(name, False, [], float("inf"), expanded, _t(meter), meter.peak_kb)
