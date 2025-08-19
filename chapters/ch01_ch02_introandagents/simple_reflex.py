#They have no memory of previous actions or positions and do not try to infer or track their location.

from typing import Dict, List
from core.vacuum_env import VacuumEnv, Percept, Action
import statistics

class SimpleReflexDeterministic:
    """Rule: if dirty -> Suck; else -> Right."""
    def act(self, percept: Percept) -> Action:
        return "Suck" if percept.dirty else "Right"

class SimpleReflexRandomized:
    """Rule: if dirty -> Suck; else -> random move (break loops/explore)."""
    def __init__(self, seed=None):
        import random; self.rng = random.Random(seed)
    def act(self, percept: Percept) -> Action:
        if percept.dirty: return "Suck"
        return self.rng.choice(["Up","Down","Left","Right"])


# env.percept() → The environment gives the agent a Percept object (e.g., location and dirt status).
# agent.act(...) → The agent decides what to do based on that percept (e.g., "Suck" or "Right").
# env.step(...) → The environment executes that action and updates its state.
def run(agent, env_cfg: Dict, episodes=100, steps=40, seed=42):
    rewards: List[int] = []
    env = VacuumEnv(**env_cfg, seed=seed)
    for _ in range(episodes):
        env.reset()
        for _ in range(steps):
            env.step(agent.act(env.percept()))
        rewards.append(env.reward)
    return rewards

if __name__ == "__main__":
    env_cfg = dict(rows=3, cols=3, p_dirty=0.5, partial=False, sensor_noise=0.0)
    det = run(SimpleReflexDeterministic(), env_cfg)
    rnd = run(SimpleReflexRandomized(seed=1), env_cfg)
    print("Deterministic avg reward:", round(statistics.mean(det),2))
    print("Randomized   avg reward:", round(statistics.mean(rnd),2))

