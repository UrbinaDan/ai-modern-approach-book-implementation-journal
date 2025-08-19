#tests a more sophisticated agent that maintains internal state, allowing it to perform better when it can't directly observe its location.

from typing import Dict, Optional, Tuple, List
from core.vacuum_env import VacuumEnv, Percept, Action
import statistics

class ModelBasedReflex:
    """
    Internal belief of location (updated via last action) + serpentine sweep.
    """
    def __init__(self, rows, cols):
        self.R, self.C = rows, cols
        self.reset()
    def reset(self):
        # “Uses memory to plan ahead or make informed decisions”
        self.belief_loc: Tuple[int,int] = (0,0)
        self.last_action: Optional[Action] = None

    #🧠 What This Does
        # belief_loc is the agent’s internal guess of where it is on the grid.
        # It updates this guess based on the last action taken, not on any external feedback.
        # This allows the agent to track its position even when it can't directly observe it (due to partial observability or sensor noise).
    def _update_belief(self, action: Action):
        r,c = self.belief_loc
        nr, nc = r, c
        if action == "Up": nr -= 1
        elif action == "Down": nr += 1
        elif action == "Left": nc -= 1
        elif action == "Right": nc += 1
        nr = min(max(nr,0), self.R-1); nc = min(max(nc,0), self.C-1)
        self.belief_loc = (nr, nc)

    def act(self, percept: Percept) -> Action:
        if self.last_action: self._update_belief(self.last_action)
        if percept.dirty:
            self.last_action = "Suck"; return "Suck"
        # This is a form of planning ahead—it’s not reacting randomly, but following a structured path.
        r,c = self.belief_loc
        # serpentine sweep: even rows go right; odd rows go left
        if r % 2 == 0:
            a = "Right" if c < self.C-1 else ("Down" if r < self.R-1 else "Left")
        else:
            a = "Left" if c > 0 else ("Down" if r < self.R-1 else "Right")
        self.last_action = a
        return a

def run(agent, env_cfg: Dict, episodes=100, steps=40, seed=42, reset_agent=True):
    rewards: List[int] = []
    env = VacuumEnv(**env_cfg, seed=seed)
    for _ in range(episodes):
        env.reset()
        if reset_agent and hasattr(agent, "reset"): agent.reset()
        for _ in range(steps):
            env.step(agent.act(env.percept()))
        rewards.append(env.reward)
    return rewards

if __name__ == "__main__":
    env_cfg = dict(rows=3, cols=3, p_dirty=0.5, partial=True, sensor_noise=0.05)
    mbr = ModelBasedReflex(3,3)
    m_rewards = run(mbr, env_cfg)
    from simple_reflex import SimpleReflexRandomized, run as run_sr
    r_rewards = run_sr(SimpleReflexRandomized(seed=2), env_cfg, episodes=100, steps=40, seed=42)
    print("Model-Based (partial) avg:", round(statistics.mean(m_rewards),2))
    print("Random SimpleReflex (partial) avg:", round(statistics.mean(r_rewards),2))
