#Even pre-ML, exploration helps when the world changes
# Can occasional randomness (exploration) help an agent adapt to a changing world better than a purely greedy strategy?

from typing import List
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from core.vacuum_env import VacuumEnv
from model_based import ModelBasedReflex

class EpsilonWrapper:
    def __init__(self, base_agent, eps=0.1, seed=None):
        self.base = base_agent
        self.eps = eps
        self.rng = random.Random(seed)
    def act(self, percept):
        if self.rng.random() < self.eps:
            return self.rng.choice(["Up","Down","Left","Right","Suck"])
        return self.base.act(percept)
    def reset(self):
        if hasattr(self.base, "reset"): self.base.reset()

def drift_experiment(agent_factory, episodes=200, steps=30, seed=7) -> List[int]:
    rng = random.Random(seed)
    p_cycle = [0.2, 0.5, 0.8, 0.3]  # drift every 50 eps
    rewards = []
    for ep in range(episodes):
        p_dirty = p_cycle[(ep//50) % len(p_cycle)]
        env = VacuumEnv(rows=3, cols=3, p_dirty=p_dirty, partial=True, sensor_noise=0.05, seed=rng.randrange(1<<30))
        agent = agent_factory()
        if hasattr(agent, "reset"): agent.reset()
        env.reset()
        for _ in range(steps):
            env.step(agent.act(env.percept()))
        rewards.append(env.reward)
    return rewards

def factory_base(): return ModelBasedReflex(3,3)
def factory_eps():  return EpsilonWrapper(ModelBasedReflex(3,3), eps=0.1, seed=999)

if __name__ == "__main__":
    base = drift_experiment(factory_base)
    eps  = drift_experiment(factory_eps)

    df = pd.DataFrame({"episode": np.arange(1,201), "reward_base": base, "reward_eps": eps})
    df.to_csv("files/data_epsilon_concept_drift_rewards.csv", index=False)
    print("Saved -> files/data_epsilon_concept_drift_rewards.csv")

    plt.figure()
    plt.plot(base, label="Base agent")
    plt.plot(eps,  label="ε-greedy (ε=0.1)")
    plt.xlabel("Episode"); plt.ylabel("Reward")
    plt.title("Concept Drift: Rewards per Episode (Partial Obs, Sensor Noise)")
    plt.legend(); plt.tight_layout()
    plt.savefig("files/data_epsilon_concept_drift_rewards.png")  # <-- Save the plot as PNG
    print("Saved plot -> files/data_epsilon_concept_drift_rewards.png")
    plt.show()