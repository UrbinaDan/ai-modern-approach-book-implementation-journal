from __future__ import annotations
from typing import Dict, Iterable, Any, Optional
from ..core.problem import Problem

# Undirected graph: every road both ways with the same distance
_ROADS: Dict[str, Dict[str, float]] = {
    "Arad": {"Zerind": 75, "Sibiu": 140, "Timisoara": 118},
    "Zerind": {"Arad": 75, "Oradea": 71},
    "Oradea": {"Zerind": 71, "Sibiu": 151},
    "Sibiu": {"Arad": 140, "Oradea": 151, "Fagaras": 99, "Rimnicu Vilcea": 80},
    "Timisoara": {"Arad": 118, "Lugoj": 111},
    "Lugoj": {"Timisoara": 111, "Mehadia": 70},
    "Mehadia": {"Lugoj": 70, "Drobeta": 75},
    "Drobeta": {"Mehadia": 75, "Craiova": 120},
    "Craiova": {"Drobeta": 120, "Rimnicu Vilcea": 146, "Pitesti": 138},
    "Rimnicu Vilcea": {"Sibiu": 80, "Craiova": 146, "Pitesti": 97},
    "Fagaras": {"Sibiu": 99, "Bucharest": 211},
    "Pitesti": {"Rimnicu Vilcea": 97, "Craiova": 138, "Bucharest": 101},
    "Bucharest": {"Fagaras": 211, "Pitesti": 101, "Giurgiu": 90, "Urziceni": 85},
    "Giurgiu": {"Bucharest": 90},
    "Urziceni": {"Bucharest": 85, "Hirsova": 98, "Vaslui": 142},
    "Hirsova": {"Urziceni": 98, "Eforie": 86},
    "Eforie": {"Hirsova": 86},
    "Vaslui": {"Urziceni": 142, "Iasi": 92},
    "Iasi": {"Vaslui": 92, "Neamt": 87},
    "Neamt": {"Iasi": 87},
}

# Straight-line distance (heuristic) to Bucharest (hSLD). Missing entries default to 0.0.
_SLD_TO_BUC: Dict[str, float] = {
    "Arad": 366, "Bucharest": 0, "Craiova": 160, "Drobeta": 242, "Eforie": 161,
    "Fagaras": 176, "Giurgiu": 77, "Hirsova": 151, "Iasi": 226, "Lugoj": 244,
    "Mehadia": 241, "Neamt": 234, "Oradea": 380, "Pitesti": 100, "Rimnicu Vilcea": 193,
    "Sibiu": 253, "Timisoara": 329, "Urziceni": 80, "Vaslui": 199, "Zerind": 374,
}

class RomaniaMap(Problem):
    def __init__(self, start: str = "Arad", goal: str = "Bucharest"):
        self._start = start
        self._goal = goal

    # Problem API
    def initial_state(self) -> Any:
        return self._start

    def is_goal(self, state: Any) -> bool:
        return state == self._goal

    def actions(self, state: Any) -> Iterable[Any]:
        return _ROADS.get(state, {}).keys()

    def result(self, state: Any, action: Any) -> Any:
        # For this formulation, the "action" is the target city.
        return action

    def step_cost(self, state: Any, action: Any, next_state: Any) -> float:
        # Return cost, or raise KeyError if inconsistent definition
        try:
            return float(_ROADS[state][next_state])
        except KeyError as e:
            # Helpful message if the graph/action is inconsistent
            raise KeyError(f"No road cost for ({state} -> {next_state}). "
                           f"Check _ROADS and actions(state).") from e

    # Heuristic (optional; used by Greedy/A*)
    def heuristic(self, state: Any) -> float:
        return float(_SLD_TO_BUC.get(state, 0.0))

def romania_problem(start: str = "Arad", goal: str = "Bucharest") -> RomaniaMap:
    return RomaniaMap(start=start, goal=goal)
