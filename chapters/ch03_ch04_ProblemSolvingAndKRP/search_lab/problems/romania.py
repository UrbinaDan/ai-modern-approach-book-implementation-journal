# search_lab/problems/romania.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Mapping, Iterable, List

from ..core.problem import Problem


# --- Data --------------------------------------------------------------------

# Road distances (bidirectional) from AIMA Fig. 3.1
_GRAPH: Dict[str, Dict[str, int]] = {
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
    "Urziceni": {"Bucharest": 85, "Vaslui": 142, "Hirsova": 98},
    "Hirsova": {"Urziceni": 98, "Eforie": 86},
    "Eforie": {"Hirsova": 86},
    "Vaslui": {"Urziceni": 142, "Iasi": 92},
    "Iasi": {"Vaslui": 92, "Neamt": 87},
    "Neamt": {"Iasi": 87},
}

# Straight-line distance to Bucharest (AIMA Fig. 3.16)
_SLD: Dict[str, int] = {
    "Arad": 366, "Zerind": 374, "Oradea": 380, "Sibiu": 253, "Timisoara": 329,
    "Lugoj": 244, "Mehadia": 241, "Drobeta": 242, "Craiova": 160, "Rimnicu Vilcea": 193,
    "Fagaras": 176, "Pitesti": 100, "Bucharest": 0, "Giurgiu": 77, "Urziceni": 80,
    "Hirsova": 151, "Eforie": 161, "Vaslui": 199, "Iasi": 226, "Neamt": 234,
}


# --- Public container (so imports of RomaniaMap succeed) ---------------------

@dataclass(frozen=True)
class RomaniaMap:
    graph: Mapping[str, Mapping[str, int]]
    sld_to_bucharest: Mapping[str, int]


ROMANIA = RomaniaMap(graph=_GRAPH, sld_to_bucharest=_SLD)


# --- Problem definition -------------------------------------------------------

class RomaniaProblem(Problem):
    """
    Standard AIMA Romania route-finding problem.
    States are city names (strings).
    ACTIONS(s) are neighboring cities; RESULT(s,a) = a; step_cost is road distance.
    """

    def __init__(self, start: str = "Arad", goal: str = "Bucharest", data: RomaniaMap = ROMANIA):
        self.start = start
        self.goal = goal
        self.data = data  # exposes .graph and .sld_to_bucharest

    # Problem API expected by your core/Node.expand etc.
    def initial_state(self):
        return self.start

    def is_goal(self, state) -> bool:
        return state == self.goal

    def actions(self, state) -> Iterable[str]:
        return self.data.graph[state].keys()

    def result(self, state, action) -> str:
        # Action is the next city name
        return action

    def step_cost(self, state, action, next_state) -> float:
        return float(self.data.graph[state][next_state])

    # Optional: a convenience heuristic some algorithms may use
    def heuristic(self, state) -> float:
        return float(self.data.sld_to_bucharest.get(state, 0))


def romania_problem(start: str = "Arad", goal: str = "Bucharest") -> RomaniaProblem:
    """
    Factory for a ready-to-use RomaniaProblem.
    """
    return RomaniaProblem(start=start, goal=goal, data=ROMANIA)
