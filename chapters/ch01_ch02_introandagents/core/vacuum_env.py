# core/vacuum_env.py
from dataclasses import dataclass
from typing import Optional, Tuple

Action = str  # "Up","Down","Left","Right","Suck","NoOp"


# location: Either the agent’s current position or None if location is hidden.
# dirty: Whether the agent thinks the current cell is dirty (may be noisy).

@dataclass # dataclass is used mostly for storage classes 
class Percept:
    location: Optional[Tuple[int,int]]  # None when partial observability
    dirty: bool                         # possibly noisy reading



# VacuumEnv class for a grid-based vacuum cleaner environment.
# Initializes the environment with:
# Grid size (rows, cols)
# Dirt probability (p_dirty)
# Partial observability toggle
# Sensor noise level
# Random seed for reproducibility
#       Without setting a seed, each run gives different results. With a seed, you get the same sequence every time.

class VacuumEnv:
    """
    rows x cols grid; each cell dirty/clean.
    If partial=True, hide location from the percept.
    sensor_noise flips the dirt reading with given probability.

    Rewards (simple shaping):
      +10 for Suck on truly dirty cell
      -1  for move (time/energy)
      -1  for Suck on clean (waste)
      -1  for NoOp/invalid (time passes)
    """
    def __init__(self, rows=3, cols=3, p_dirty=0.5, partial=False, sensor_noise=0.0, seed=None):
        import random
        self.R, self.C = rows, cols
        self.p_dirty = p_dirty
        self.partial = partial
        self.sensor_noise = sensor_noise
        self.rng = random.Random(seed)
        self.reset()

    # Reset the environment to a new state.
    # t is the time step, reward is the accumulated reward.
    # The agent starts at a random position, and the grid is initialized with dirt based on p_dirty.
    # Returns the initial percept of the environment.
    def reset(self):
        self.agent = (self.rng.randrange(self.R), self.rng.randrange(self.C))
        self.grid = [[self.rng.random() < self.p_dirty for _ in range(self.C)] for _ in range(self.R)]
        self.t = 0
        self.reward = 0
        return self.percept()

    #Helper method to check if a position is inside the grid.
    def _in_bounds(self, r, c): return 0 <= r < self.R and 0 <= c < self.C

    # Gets the agent’s current location and dirt status.
    # Applies sensor noise if enabled.
    # Returns a Percept object containing the location and whether the cell is dirty.
    def percept(self) -> Percept:
        r, c = self.agent
        dirty_true = self.grid[r][c]
        noisy = (not dirty_true) if (self.rng.random() < self.sensor_noise) else dirty_true
        loc = None if self.partial else (r, c)
        return Percept(loc, noisy)
        
    # Executes the agent’s action:
    #       "Suck": Cleans dirt if present, rewards or penalizes accordingly.
    #       Movement: Updates position if within bounds, penalizes for energy/time.
    #       "NoOp" or invalid: Penalizes for wasting time.
    def step(self, action: Action) -> Percept:
        r, c = self.agent
        if action == "Suck":
            if self.grid[r][c]:
                self.grid[r][c] = False
                self.reward += 10
            else:
                self.reward -= 1
        elif action in ("Up","Down","Left","Right"):
            nr, nc = r, c
            if action == "Up": nr -= 1
            elif action == "Down": nr += 1
            elif action == "Left": nc -= 1
            elif action == "Right": nc += 1
            if self._in_bounds(nr, nc):
                self.agent = (nr, nc)
            self.reward -= 1
        else:
            self.reward -= 1
        self.t += 1
        return self.percept()
