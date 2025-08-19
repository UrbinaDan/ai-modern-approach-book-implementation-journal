from typing import Tuple, List, Set, Dict, Iterable

# deque: A double-ended queue.
    # Fast for appending and popping from both ends.
    # Ideal for BFS, which uses a queue to explore nodes level by level.
from collections import deque
# heapq: Implements a priority queue using a min-heap.
    # Used in A\* to always expand the node with the lowest estimated cost.
import heapq

State = Tuple[int,int]

class GridWorld:
    def __init__(self, rows=3, cols=3, walls: List[State] = None):
        self.R, self.C = rows, cols
        self.walls: Set[State] = set(walls or [])
    def neighbors(self, s: State) -> Iterable[Tuple[State,str]]:
        r,c = s
        for dr,dc,a in [(-1,0,"Up"),(1,0,"Down"),(0,-1,"Left"),(0,1,"Right")]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < self.R and 0 <= nc < self.C and (nr,nc) not in self.walls:
                yield (nr,nc), a

# Builds both the path and the action sequence.
def reconstruct(parent: Dict[State,State], action: Dict[State,str], start: State, goal: State):
    if goal not in parent: return None
    path, acts = [], []
    s = goal
    while parent[s] is not None:
        path.append(s); acts.append(action[s]); s = parent[s]
    path.append(start); path.reverse(); acts.reverse()
    return path, acts

# Finds the shortest path using BFS.
def bfs_path(world: GridWorld, start: State, goal: State):
    q = deque([start]); parent={start:None}; action={}
    while q:
        s = q.popleft()
        if s == goal: break
        for ns, a in world.neighbors(s):
            if ns not in parent:
                parent[ns]=s; action[ns]=a; q.append(ns)
    return reconstruct(parent, action, start, goal)

def manhattan(a: State, b: State): return abs(a[0]-b[0]) + abs(a[1]-b[1])

def astar_path(world: GridWorld, start: State, goal: State):
    g={start:0}; parent={start:None}; action={}
    openh=[]; heapq.heappush(openh,(0,start))
    while openh:
        _, s = heapq.heappop(openh)
        if s == goal: break
        for ns, a in world.neighbors(s):
            ng = g[s]+1
            if ns not in g or ng < g[ns]:
                g[ns]=ng; parent[ns]=s; action[ns]=a
                heapq.heappush(openh,(ng+manhattan(ns,goal), ns))
    return reconstruct(parent, action, start, goal)

if __name__ == "__main__":
    world = GridWorld(3,3,walls=[(1,1)])
    start, goal = (0,0), (2,2)
    bfs = bfs_path(world, start, goal)
    ast = astar_path(world, start, goal)
    print("BFS:", bfs[0], bfs[1])
    print("A* :", ast[0], ast[1])
    start_time = time.time()





