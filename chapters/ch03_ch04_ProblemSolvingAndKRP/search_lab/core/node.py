# search_lab/core/node.py
# This code defines a Node class used in AI search algorithms (like BFS, DFS, A*, etc.) to represent states in a search tree.
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Any, List, Iterable
from .problem import Problem, State


class Node:
    def __init__(self, state, parent=None, action=None, path_cost: float = 0.0, depth: int = 0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = float(path_cost)
        self.depth = depth

    def expand(self, problem):
        """Generate child Nodes by applying ACTIONS(s), using RESULT and step_cost."""
        s = self.state
        for a in problem.actions(s):
            s2 = problem.result(s, a)
            cost = problem.step_cost(s, a, s2)
            if cost is None:
                raise ValueError(
                    f"step_cost returned None for (s={s!r}, a={a!r}, s'={s2!r}). "
                    "Check your problem’s ACTIONS/RESULT/cost mapping."
                )
            yield Node(
                state=s2,
                parent=self,
                action=a,
                path_cost=self.path_cost + float(cost),
                depth=self.depth + 1,
            )
