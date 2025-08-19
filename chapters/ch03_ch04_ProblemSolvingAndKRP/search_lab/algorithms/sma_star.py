# search_lab/algorithms/sma_star.py
# This code implements Simplified Memory-Bounded A* (SMA*), which is a variant of the A* search algorithm that uses a limited amount of memory.
# SMA* is designed to handle large search spaces by dropping the worst leaf nodes when the memory limit is reached.
# It maintains a balance between exploration and exploitation by backing up f estimates shallowly, allowing it to find solutions in memory-constrained environments.
from __future__ import annotations
from typing import Dict, List
import heapq
from ..core.node import Node
from ..core.metrics import SearchResult, MeasuredRun
from ..core.problem import Problem
from ..core.utils import reconstruct_path

class _SMAQueue:
    """Frontier that can drop worst leaf when size > cap."""
    def __init__(self, key, cap:int):
        self.key = key; self.cap = cap
        self.h: List[tuple] = []
        self.counter = 0
    def push(self, x):
        self.counter += 1
        heapq.heappush(self.h, (self.key(x), self.counter, x))
        if len(self.h) > self.cap:
            # drop worst leaf: this is a min-heap; 'worst' = max f. We pop all to find worst.
            worst_i = max(range(len(self.h)), key=lambda i: self.h[i][0])
            self.h.pop(worst_i)
            heapq.heapify(self.h)
    def pop(self): return heapq.heappop(self.h)[2]
    def __len__(self): return len(self.h)
    def peek(self): return self.h[0][2]

def sma_star(problem: Problem, memory_nodes:int=5000) -> SearchResult:
    """Simplified memory-bounded A* (drops worst leaves; backs up f estimates shallowly)."""
    name = f"SMA*(cap={memory_nodes})"
    start = Node(problem.initial_state())
    start.f = problem.heuristic(start.state)
    frontier = _SMAQueue(key=lambda n: n.path_cost + problem.heuristic(n.state), cap=memory_nodes)
    frontier.push(start)
    expanded = 0
    best_goal = None

    with MeasuredRun() as meter:
        while len(frontier):
            node = frontier.pop()
            if problem.is_goal(node.state):
                best_goal = node
                break
            expanded += 1
            children = list(node.expand(problem))
            if not children:
                # dead-end; mark with inf f so its parent can deprioritize it
                node.f = float("inf")
            for c in children:
                c.f = c.path_cost + problem.heuristic(c.state)
                frontier.push(c)

        if best_goal:
            a,c = reconstruct_path(best_goal)
            return SearchResult(name, True, a, c, expanded, meter.elapsed, meter.peak_kb)
        return SearchResult(name, False, [], float("inf"), expanded, meter.elapsed, meter.peak_kb)
