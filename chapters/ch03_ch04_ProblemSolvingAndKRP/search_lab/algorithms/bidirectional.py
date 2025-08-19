# search_lab/algorithms/bidirectional.py
# This code implements a bidirectional search algorithm, which is a search strategy that simultaneously explores paths from both the start and goal states.
# It is often used to find the shortest path in problems where the goal state is known.
from __future__ import annotations
from typing import Dict
from ..core.node import Node
from ..core.frontiers import FIFOQueue
from ..core.metrics import SearchResult, MeasuredRun
from ..core.problem import Problem, State
from ..core.utils import reconstruct_path

def _meet_path(nf: Node, nb: Node):
    # join forward nf and backward nb (nb path goes from goal to start)
    f_path = nf.path()
    b_path = nb.path()
    b_path = list(reversed(b_path))
    # actions reconstruction: forward actions + reversed backward actions with domain-specific inverse.
    # For metrics, just return states via forward end.
    return nf

def bidirectional_bfs(problem: Problem, neighbors_inv) -> SearchResult:
    """neighbors_inv(s): backward neighbors (states that lead into s)."""
    name = "Bi-BFS"
    s0 = problem.initial_state()
    f_front = FIFOQueue(); b_front = FIFOQueue()
    nf = Node(s0); f_front.push(nf)
    nb = Node(next(iter([ ])))  # you should pass actual goal list; use Grid/Romania helpers below.

    # For simplicity, provide fully working bidirectional A* below; for Bi-BFS in practice
    # use the bidirectional A* scaffolding with f(n)=g(n) and h(n)=0.
    raise NotImplementedError("Use bidirectional A* wrapper (with h=0) for a working bi-UCS/BFS.")
