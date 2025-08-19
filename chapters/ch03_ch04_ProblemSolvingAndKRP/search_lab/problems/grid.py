# search_lab/problems/grid.py
from __future__ import annotations
from typing import Iterable, Hashable, Tuple, Set
from ..core.problem import Problem

Coord = Tuple[int, int]

_MOVES = {
    "Up": (-1, 0),
    "Down": (1, 0),
    "Left": (0, -1),
    "Right": (0, 1),
}

class GridProblem(Problem):
    """
    4-neighbor grid pathfinding with unit costs.

    - State: (row, col) tuple
    - ACTIONS(s): subset of {'Up','Down','Left','Right'} that keep you in-bounds and off walls
    - RESULT(s,a): next (row, col)
    - IS-GOAL(s): s == goal
    - c(s,a,s'): 1.0
    - heuristic(s): Manhattan distance (admissible on 4-neighbor grid)
    """
    def __init__(self, rows: int, cols: int, start: Coord, goal: Coord, walls: Set[Coord] | None = None):
        self.rows = rows
        self.cols = cols
        self._start = start
        self._goal = goal
        self.walls = walls or set()

    def initial_state(self) -> Hashable:
        return self._start

    def is_goal(self, state: Hashable) -> bool:
        return state == self._goal

    def actions(self, state: Hashable) -> Iterable[str]:
        r, c = state
        for name, (dr, dc) in _MOVES.items():
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols and (nr, nc) not in self.walls:
                yield name

    def result(self, state: Hashable, action: Hashable) -> Hashable:
        r, c = state
        dr, dc = _MOVES[action]
        return (r + dr, c + dc)

    def cost(self, state: Hashable, action: Hashable, next_state: Hashable) -> float:
        return 1.0

    def heuristic(self, state: Hashable) -> float:
        r, c = state
        gr, gc = self._goal
        return float(abs(r - gr) + abs(c - gc))

def make_grid_problem() -> GridProblem:
    # Example: 5x7 grid, a few walls
    walls = {(1,3), (2,3), (3,3), (3,4)}
    return GridProblem(rows=5, cols=7, start=(0,0), goal=(4,6), walls=walls)
