# Defines the standard interface for any search problem (states, actions, goals, costs, heuristic).
# search_lab/core/problem.py
from __future__ import annotations
from typing import Any, Iterable, Tuple, Protocol, Optional, Hashable, Callable

Action = Hashable
State = Hashable

class Problem(Protocol):
    """Canonical AI search problem interface (atomic state-space view)."""
    def initial_state(self) -> State: ...
    def is_goal(self, s: State) -> bool: ...
    def actions(self, s: State) -> Iterable[Action]: ...
    def result(self, s: State, a: Action) -> State: ...
    def step_cost(self, s: State, a: Action, s2: State) -> float: ...
    # Optional heuristic for informed search; default 0
    def heuristic(self, s: State) -> float: return 0.0

# Helper for bidirectional: reverse problem wrapper
class ReverseProblem:
    """Wraps a problem to run the search backward from goal(s).
    Requires that ACTIONS/RESULT are invertible in your problem domain."""
    def __init__(self, problem: Problem, goal_states: Iterable[State]):
        self.p = problem
        self._initials = tuple(goal_states)

    def initial_states(self) -> Tuple[State, ...]:
        return self._initials

    def is_goal(self, s: State) -> bool:
        return s == self.p.initial_state()

    # You must implement inverse actions in your Problem to support this robustly.
    # For grid and Romania examples below, we include invertible actions.
    def actions(self, s: State) -> Iterable[Action]:
        # reverse = the inverse actions that lead to s
        # Provide a domain-specific implementation; for our examples we’ll override.
        raise NotImplementedError

    def result(self, s: State, a: Action) -> State:
        # Applying reverse action means going “backwards” one step
        raise NotImplementedError

    def step_cost(self, s: State, a: Action, s2: State) -> float:
        # same costs in reverse direction
        return self.p.step_cost(s2, a, s)

    def heuristic(self, s: State) -> float:
        # admissible reverse heuristic if symmetric; else 0
        return 0.0
