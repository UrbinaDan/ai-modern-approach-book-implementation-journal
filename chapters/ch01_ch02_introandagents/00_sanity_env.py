# To verify that the environment initializes and responds to actions as expected.
from core.vacuum_env import VacuumEnv
env = VacuumEnv(rows=3, cols=3, p_dirty=0.5, seed=0)
for _ in range(3):
    print(env.percept())
    env.step("Right")
