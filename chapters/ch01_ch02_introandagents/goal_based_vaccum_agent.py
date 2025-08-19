from typing import List, Tuple
from core.vacuum_env import VacuumEnv, Percept, Action
from collections import deque

State = Tuple[int, int]

def neighbors(s: State, R: int, C: int) -> List[Tuple[State, Action]]:
    r, c = s
    moves = [(-1,0,"Up"), (1,0,"Down"), (0,-1,"Left"), (0,1,"Right")]
    result = []
    for dr, dc, a in moves:
        nr, nc = r+dr, c+dc
        if 0 <= nr < R and 0 <= nc < C:
            result.append(((nr, nc), a))
    return result

def bfs_path(start: State, goal: State, R: int, C: int) -> List[Action]:
    q = deque([start]); parent={start:None}; action={}
    while q:
        s = q.popleft()
        if s == goal: break
        for ns, a in neighbors(s, R, C):
            if ns not in parent:
                parent[ns]=s; action[ns]=a; q.append(ns)
    if goal not in parent: return []
    path = []; s = goal
    while parent[s] is not None:
        path.append(action[s]); s = parent[s]
    path.reverse()
    return path

class PlanningVacuumAgent:
    def __init__(self, rows, cols):
        self.R, self.C = rows, cols
        self.reset()

    def reset(self):
        self.plan: List[Action] = []
        self.known_dirt: List[State] = []

    def act(self, percept: Percept, grid: List[List[bool]]) -> Action:
        if percept.location is None:
            return "NoOp"

        r, c = percept.location
        if percept.dirty:
            return "Suck"

        if not self.plan:
            self.known_dirt = [(i,j) for i in range(self.R) for j in range(self.C) if grid[i][j]]
            if not self.known_dirt:
                return "NoOp"
            goal = self.known_dirt[0]
            self.plan = bfs_path((r,c), goal, self.R, self.C)

        return self.plan.pop(0)

def run(agent, env_cfg, episodes=100, steps=40, seed=42):
    from statistics import mean
    rewards = []
    for _ in range(episodes):
        env = VacuumEnv(**env_cfg, seed=seed)
        env.reset()
        agent.reset()
        for _ in range(steps):
            percept = env.percept()
            action = agent.act(percept, env.grid)
            env.step(action)
        rewards.append(env.reward)
    print("Planning agent avg reward:", round(mean(rewards), 2))

if __name__ == "__main__":
    env_cfg = dict(rows=3, cols=3, p_dirty=0.5, partial=False, sensor_noise=0.0)
    agent = PlanningVacuumAgent(3, 3)
    run(agent, env_cfg) 
