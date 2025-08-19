# search_lab/problems/graph_romania.py
#This code defines a search problem for the Romania road map, a classic example from AI textbooks, for use with search algorithms like BFS, DFS, A*, etc.
from __future__ import annotations
from typing import Dict, Iterable, Tuple
from dataclasses import dataclass
from ..core.problem import Problem, State, Action

@dataclass
class Romania(Problem):
    start: str
    goal: str
    graph: Dict[str, Dict[str, int]]
    coords: Dict[str, Tuple[float,float]]  # for straight-line heuristic

    def initial_state(self) -> State: return self.start
    def is_goal(self, s: State) -> bool: return s == self.goal
    def actions(self, s: State) -> Iterable[Action]:
        return self.graph[s].keys()
    def result(self, s: State, a: Action) -> State:
        return a  # action is the neighbor city
    def step_cost(self, s: State, a: Action, s2: State) -> float:
        return float(self.graph[s][s2])
    def heuristic(self, s: State) -> float:
        import math
        (x1,y1), (x2,y2) = self.coords[s], self.coords[self.goal]
        return math.hypot(x1-x2, y1-y2)
