# chapter4_lab.py
"""
Chapter 4 Maze Lab — Interactive GUI for classic local/online/nondeterministic search
-------------------------------------------------------------------------------
No external deps; uses Tkinter only.

Algorithms:
- Hill-Climbing (steepest descent on heuristic)
- Simulated Annealing (geometric cooling)
- Local Beam Search (k-best by f = g + h)
- Stochastic Beam Search (softmax sampling to keep k)
- Genetic / Evolutionary Search (fixed-length move genomes; selection + crossover + mutation)
- Online DFS Agent (unknown-space depth-first exploration)
- LRTA* (Learning Real-Time A*)
- AND–OR Planner (simple nondeterministic model: slip-to-stay with probability p_slip)

Notes & interpretations for a grid world
----------------------------------------
- We minimize h(n) = distance-to-goal (Manhattan if no diagonals; Octile if diagonals).
- Hill-Climbing uses only the current node's neighbors; stops at plateaus/local minima.
- SA can accept "downhill" (worse) moves with probability exp(-ΔE / T), T cools over time.
- Beam variants use f = g + h and keep only k most promising frontier states each layer.
- Genetic Search evolves *move sequences* (N,S,E,W, and diagonals if enabled). Fitness favors
  short, feasible paths that get closer to the goal.
- Online DFS & LRTA* interleave acting with local updates (no global search tree).
- AND–OR planner builds a *conditional plan* when moves can "slip" (stay put) with some probability.
  We model RESULTS(s,a) = {s, move(s,a)} when p_slip > 0; cycles are detected; small depth limit.

GUI goodies
-----------
- Draw walls; set Start/Goal; randomize walls; diagonals toggle; speed slider.
- Per-algorithm params (k for Beam, τ for Stochastic Beam, GA population/mutation, SA cooling, slip prob).
- Live animation: expanded/closed/frontier/path overlays; compact metrics on finish.

Color legend
------------
- Walls: dark gray
- Visited/closed: light blue
- Frontier: orange
- Current "best path" guess: yellow
- Expanding/acting state: pink
- Start: green  |  Goal: red
"""

from __future__ import annotations
import math, random, time, heapq, tkinter as tk
from tkinter import ttk, messagebox
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Iterable, Generator, Callable

# --------------------------
# Core grid + heuristics
# --------------------------

Cell = Tuple[int, int]

@dataclass
class Grid:
    rows: int
    cols: int
    walls: Set[Cell] = field(default_factory=set)
    start: Optional[Cell] = None
    goal: Optional[Cell] = None
    allow_diagonals: bool = False

    def in_bounds(self, p: Cell) -> bool:
        r, c = p
        return 0 <= r < self.rows and 0 <= c < self.cols

    def passable(self, p: Cell) -> bool:
        return p not in self.walls

    def neighbors(self, p: Cell) -> Iterable[Tuple[Cell, float]]:
        r, c = p
        steps4 = [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]
        for q in steps4:
            if self.in_bounds(q) and self.passable(q):
                yield q, 1.0
        if self.allow_diagonals:
            stepsd = [(r-1,c-1),(r-1,c+1),(r+1,c-1),(r+1,c+1)]
            for q in stepsd:
                if self.in_bounds(q) and self.passable(q):
                    yield q, math.sqrt(2.0)

def manhattan(a: Cell, b: Cell) -> float:
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def octile(a: Cell, b: Cell) -> float:
    dx, dy = abs(a[0]-b[0]), abs(a[1]-b[1])
    return (max(dx, dy) - min(dx, dy)) + math.sqrt(2) * min(dx, dy)

def hfun(grid: Grid) -> Callable[[Cell, Cell], float]:
    return octile if grid.allow_diagonals else manhattan

def reconstruct(parent: Dict[Cell, Cell], s: Cell, t: Cell) -> List[Cell]:
    if t not in parent and t != s: return []
    path = [t]
    cur = t
    while cur != s:
        cur = parent[cur]
        path.append(cur)
    path.reverse()
    return path

# --------------------------
# Frame + Result for animation
# --------------------------

@dataclass
class SearchResult:
    name: str
    success: bool
    path: List[Cell]
    cost: float
    nodes_expanded: int
    time_s: float

@dataclass
class Frame:
    expand: Optional[Cell]
    frontier: Set[Cell]
    closed: Set[Cell]
    best_path: List[Cell]
    done: bool = False
    final: Optional[SearchResult] = None

# --------------------------
# Generic best-first (covers Beam/A*/Greedy/UCS-like)
# --------------------------

def best_first_steps(
    grid: Grid,
    name: str,
    priority: Callable[[float, float], float],  # inputs: g, h -> f
    tie_breaker: bool = True
) -> Generator[Frame, None, SearchResult]:
    start, goal = grid.start, grid.goal
    if start is None or goal is None:
        raise RuntimeError("Set Start and Goal first.")
    h = hfun(grid)
    t0 = time.perf_counter()

    g: Dict[Cell, float] = {start: 0.0}
    parent: Dict[Cell, Cell] = {}
    closed: Set[Cell] = set()
    pq: List[Tuple[float, int, Cell]] = []
    counter = 0

    heapq.heappush(pq, (priority(0.0, h(start, goal)), counter, start))
    counter += 1
    expanded = 0

    while pq:
        f, _, u = heapq.heappop(pq)
        if u in closed: continue
        closed.add(u); expanded += 1

        frontier_set = {n for _, _, n in pq}
        yield Frame(expand=u, frontier=frontier_set, closed=set(closed),
                    best_path=reconstruct(parent, start, u))

        if u == goal:
            path = reconstruct(parent, start, goal)
            res = SearchResult(name, True, path, g[u], expanded, time.perf_counter()-t0)
            yield Frame(expand=u, frontier=frontier_set, closed=set(closed),
                        best_path=path, done=True, final=res)
            return res

        for v, w in grid.neighbors(u):
            ng = g[u] + w
            if v not in g or ng < g[v]:
                g[v] = ng
                parent[v] = u
                fv = priority(ng, h(v, goal))
                heapq.heappush(pq, (fv, counter if tie_breaker else 0, v))
                counter += 1

    res = SearchResult(name, False, [], float("inf"), expanded, time.perf_counter()-t0)
    yield Frame(None, set(), closed, [], True, res)
    return res

# --------------------------
# Hill-Climbing (steepest descent)
# --------------------------

def hill_climb_steps(grid: Grid) -> Generator[Frame, None, SearchResult]:
    """Steepest *descent* on h(n): move to neighbor with lowest h; stop if no improvement."""
    start, goal = grid.start, grid.goal
    if start is None or goal is None:
        raise RuntimeError("Set Start and Goal first.")
    h = hfun(grid)
    t0 = time.perf_counter()

    current = start
    closed: Set[Cell] = set()
    expanded = 0

    while True:
        closed.add(current); expanded += 1
        cur_h = h(current, goal)

        # show current as expanded; neighbors as frontier
        neigh = list(grid.neighbors(current))
        frontier_set = {v for v,_ in neigh}
        yield Frame(expand=current, frontier=frontier_set, closed=set(closed),
                    best_path=[current])

        if current == goal:
            res = SearchResult("Hill-Climbing", True, [current], 0.0,
                               expanded, time.perf_counter()-t0)
            yield Frame(current, frontier_set, set(closed), [current], True, res)
            return res

        # pick best neighbor strictly better
        best = None
        best_h = cur_h
        for v, _ in neigh:
            hv = h(v, goal)
            if hv < best_h or (hv == best_h and best is None):
                best_h = hv; best = v
        if best is None or best_h >= cur_h:
            # stuck
            res = SearchResult("Hill-Climbing", False, [current], cur_h,
                               expanded, time.perf_counter()-t0)
            yield Frame(current, frontier_set, set(closed), [current], True, res)
            return res
        current = best

# --------------------------
# Simulated Annealing
# --------------------------

def simulated_annealing_steps(
    grid: Grid, T0=3.0, alpha=0.995, max_steps=20000
) -> Generator[Frame, None, SearchResult]:
    """SA on h(n). ΔE = h(cur) - h(next); accept if ΔE>0 else with exp(ΔE/T)."""
    start, goal = grid.start, grid.goal
    if start is None or goal is None:
        raise RuntimeError("Set Start and Goal first.")
    h = hfun(grid)
    t0 = time.perf_counter()

    cur = start
    closed: Set[Cell] = set()
    expanded = 0
    T = T0

    for t in range(max_steps):
        closed.add(cur); expanded += 1
        neigh = list(grid.neighbors(cur))
        frontier_set = {v for v,_ in neigh}
        yield Frame(expand=cur, frontier=frontier_set, closed=set(closed),
                    best_path=[cur])

        if cur == goal:
            res = SearchResult("Simulated Annealing", True, [cur], 0.0,
                               expanded, time.perf_counter()-t0)
            yield Frame(cur, frontier_set, set(closed), [cur], True, res)
            return res

        if not neigh:
            break

        nxt, _ = random.choice(neigh)
        dE = h(cur, goal) - h(nxt, goal)
        if dE > 0 or random.random() < math.exp(dE / max(T, 1e-9)):
            cur = nxt
        T *= alpha
        if T <= 1e-6:
            break

    res = SearchResult("Simulated Annealing", False, [cur], h(cur, goal),
                       expanded, time.perf_counter()-t0)
    yield Frame(cur, set(), set(closed), [cur], True, res)
    return res

# --------------------------
# Beam & Stochastic Beam
# --------------------------

def beam_steps(grid: Grid, k: int) -> Generator[Frame, None, SearchResult]:
    """Classic local beam (k-best by f=g+h)."""
    start, goal = grid.start, grid.goal
    if start is None or goal is None: raise RuntimeError("Set Start/Goal.")
    H = hfun(grid); t0 = time.perf_counter()
    expanded = 0
    frontier: List[Tuple[Cell, float, Dict[Cell, Cell]]] = [(start, 0.0, {})]
    closed: Set[Cell] = set()

    while frontier:
        # rank by f
        scored = []
        for node, g, parent in frontier:
            f = g + H(node, goal)
            scored.append((f, node, g, parent))
        scored.sort(key=lambda x: x[0])
        layer = scored[:max(1, k)]
        frontier_set = {n for _, n, _, _ in layer}
        guess = []
        if layer:
            _, u0, _, p0 = layer[0]
            guess = reconstruct(dict(p0), start, u0)
        yield Frame(None, frontier_set, set(closed), guess)

        new_frontier: List[Tuple[Cell, float, Dict[Cell, Cell]]] = []
        for f, u, g, parent in layer:
            if u in closed: continue
            closed.add(u); expanded += 1
            if u == goal:
                path = reconstruct(parent, start, goal)
                res = SearchResult(f"Beam(k={k})", True, path, g, expanded, time.perf_counter()-t0)
                yield Frame(u, frontier_set, set(closed), path, True, res)
                return res
            for v, w in grid.neighbors(u):
                if v in closed: continue
                p2 = dict(parent); p2[v] = u
                new_frontier.append((v, g+w, p2))
        frontier = new_frontier

    res = SearchResult(f"Beam(k={k})", False, [], float("inf"), expanded, time.perf_counter()-t0)
    yield Frame(None, set(), set(closed), [], True, res)
    return res

def stochastic_beam_steps(grid: Grid, k: int, tau: float = 1.0) -> Generator[Frame, None, SearchResult]:
    """Softmax sample k from all successors each layer to keep diversity."""
    start, goal = grid.start, grid.goal
    if start is None or goal is None: raise RuntimeError("Set Start/Goal.")
    H = hfun(grid); t0 = time.perf_counter()
    expanded = 0
    frontier: List[Tuple[Cell, float, Dict[Cell, Cell]]] = [(start, 0.0, {})]
    closed: Set[Cell] = set()

    while frontier:
        # collect all successors of current beam
        pool: List[Tuple[float, Cell, float, Dict[Cell, Cell]]] = []
        for u, g, parent in frontier:
            if u in closed: continue
            closed.add(u); expanded += 1
            if u == goal:
                path = reconstruct(parent, start, goal)
                res = SearchResult(f"StochasticBeam(k={k},τ={tau})", True, path, g, expanded, time.perf_counter()-t0)
                yield Frame(u, set([u]), set(closed), path, True, res)
                return res
            for v, w in grid.neighbors(u):
                p2 = dict(parent); p2[v] = u
                f = (g+w) + H(v, goal)
                pool.append((f, v, g+w, p2))

        if not pool:
            break

        # softmax over negative f (smaller f should be higher prob)
        fs = [p[0] for p in pool]
        minf = min(fs)
        logits = [-(f - minf)/max(1e-9, tau) for f in fs]
        m = max(logits)
        exps = [math.exp(z - m) for z in logits]
        Z = sum(exps)
        probs = [e / (Z if Z>0 else 1) for e in exps]

        # sample k (with replacement for simplicity)
        picks = random.choices(range(len(pool)), weights=probs, k=max(1, k))
        next_frontier = [ (pool[i][1], pool[i][2], pool[i][3]) for i in picks ]
        frontier = next_frontier

        # frame: show chosen nodes as frontier
        frontier_set = {u for u,_,_ in frontier}
        guess = []
        if frontier:
            u0, g0, p0 = frontier[0]
            guess = reconstruct(p0, start, u0)
        yield Frame(None, frontier_set, set(closed), guess)

    res = SearchResult(f"StochasticBeam(k={k},τ={tau})", False, [], float("inf"), expanded, time.perf_counter()-t0)
    yield Frame(None, set(), set(closed), [], True, res)
    return res

# --------------------------
# Genetic / Evolutionary Search
# --------------------------

MOVE_DIRS_4 = [(-1,0),(1,0),(0,-1),(0,1)]
MOVE_DIRS_8 = MOVE_DIRS_4 + [(-1,-1),(-1,1),(1,-1),(1,1)]

def genetic_steps(
    grid: Grid,
    pop_size=60,
    genome_len=None,
    mutation_rate=0.10,
    elite=2,
    max_gens=300,
) -> Generator[Frame, None, SearchResult]:
    """Evolve fixed-length move sequences. Fitness rewards reaching goal, short + feasible path."""
    start, goal = grid.start, grid.goal
    if start is None or goal is None: raise RuntimeError("Set Start/Goal.")
    H = hfun(grid); t0 = time.perf_counter()

    dirs = MOVE_DIRS_8 if grid.allow_diagonals else MOVE_DIRS_4
    # reasonable default length
    if genome_len is None:
        genome_len = int(2.5 * (H(start, goal) + 10))

    def rand_gene(): return random.randrange(len(dirs))
    def new_individual(): return [rand_gene() for _ in range(genome_len)]

    def simulate(genes: List[int]) -> Tuple[List[Cell], float]:
        pos = start
        path = [pos]
        cost = 0.0
        for g in genes:
            dr, dc = dirs[g]
            nxt = (pos[0]+dr, pos[1]+dc)
            if not grid.in_bounds(nxt) or not grid.passable(nxt):
                # bump into wall: penalize but stay in place
                cost += 0.5  # small penalty
                path.append(pos)
                continue
            w = math.sqrt(2) if grid.allow_diagonals and abs(dr)==1 and abs(dc)==1 else 1.0
            pos = nxt; cost += w; path.append(pos)
            if pos == goal:
                break
        return path, cost

    def fitness(genes: List[int]) -> Tuple[float, Dict]:
        path, cost = simulate(genes)
        end = path[-1]
        d = H(end, goal)
        reached = (end == goal)
        # lower is better; we return NEGATIVE for selection (higher is better)
        score = - (10.0 * d + 0.1 * cost + 0.01 * len(path))
        if reached:
            score += 1000.0  # big bonus
        return score, {"path": path, "cost": cost, "reached": reached}

    def crossover(a: List[int], b: List[int]) -> List[int]:
        c = random.randrange(1, genome_len)
        return a[:c] + b[c:]

    def mutate(genes: List[int]) -> None:
        for i in range(genome_len):
            if random.random() < mutation_rate:
                genes[i] = rand_gene()

    # init pop
    pop = [new_individual() for _ in range(pop_size)]
    expanded = 0
    best_path: List[Cell] = []

    for gen in range(1, max_gens+1):
        scored = []
        for ind in pop:
            s, info = fitness(ind)
            scored.append((s, ind, info))
        scored.sort(key=lambda x: -x[0])  # high score first
        best = scored[0]
        best_path = best[2]["path"]
        expanded += len(pop)

        # show generation frame
        yield Frame(None, frontier=set(), closed=set(), best_path=best_path)

        if best[2]["reached"]:
            res = SearchResult("Genetic", True, best_path, best[2]["cost"], expanded, time.perf_counter()-t0)
            yield Frame(None, set(), set(), best_path, True, res)
            return res

        # selection: roulette by shifted-positive scores
        scores = [max(1e-6, s - min(0.0, scored[-1][0]) + 1e-6) for s,_,_ in scored]
        total = sum(scores)
        def pick():
            r = random.random() * total
            acc = 0.0
            for s, ind, info in scored:
                val = max(1e-6, s - min(0.0, scored[-1][0]) + 1e-6)
                acc += val
                if acc >= r:
                    return ind
            return scored[0][1]

        # next gen with elitism
        next_pop = [scored[i][1][:] for i in range(min(elite, pop_size))]
        while len(next_pop) < pop_size:
            p1, p2 = pick(), pick()
            child = crossover(p1, p2)
            mutate(child)
            next_pop.append(child)
        pop = next_pop

    res = SearchResult("Genetic", False, best_path, float("inf"), expanded, time.perf_counter()-t0)
    yield Frame(None, set(), set(), best_path, True, res)
    return res

# --------------------------
# Online DFS Agent
# --------------------------

def online_dfs_steps(grid: Grid) -> Generator[Frame, None, SearchResult]:
    """Depth-first exploration with physical backtracking; stops when goal reached."""
    start, goal = grid.start, grid.goal
    if start is None or goal is None: raise RuntimeError("Set Start/Goal.")
    t0 = time.perf_counter()
    result: Dict[Tuple[Cell, Cell], Cell] = {}
    untried: Dict[Cell, List[Cell]] = {}
    unback: Dict[Cell, List[Cell]] = {}
    s_prev: Optional[Cell] = None
    a_prev: Optional[Cell] = None

    s = start
    visited: Set[Cell] = set()
    expanded = 0

    while True:
        visited.add(s); expanded += 1
        # frame
        yield Frame(expand=s, frontier=set(untried.get(s, [])), closed=set(visited), best_path=[])

        if s == goal:
            res = SearchResult("Online DFS", True, [s], 0.0, expanded, time.perf_counter()-t0)
            yield Frame(s, set(), set(visited), [s], True, res)
            return res

        if s not in untried:
            untried[s] = [v for v,_ in grid.neighbors(s)]
        if s_prev is not None and a_prev is not None:
            result[(s_prev, a_prev)] = s
            unback.setdefault(s, []).insert(0, s_prev)

        if not untried[s]:
            if not unback.get(s):
                res = SearchResult("Online DFS", False, [], float("inf"), expanded, time.perf_counter()-t0)
                yield Frame(s, set(), set(visited), [], True, res); return res
            a = unback[s].pop(0)
        else:
            a = untried[s].pop()

        s_prev, a_prev = s, a
        s = a

# --------------------------
# LRTA*
# --------------------------

def lrta_star_steps(grid: Grid) -> Generator[Frame, None, SearchResult]:
    """LRTA*: update H(s) with min c(s,a,s') + H(s'); act greedily."""
    start, goal = grid.start, grid.goal
    if start is None or goal is None: raise RuntimeError("Set Start/Goal.")
    Hest: Dict[Cell, float] = {}
    h = hfun(grid)
    t0 = time.perf_counter()

    def LRTA_cost(s: Cell, a: Cell) -> float:
        # c + H(s')
        w = math.sqrt(2) if (grid.allow_diagonals and abs(s[0]-a[0])==1 and abs(s[1]-a[1])==1) else 1.0
        return w + Hest.get(a, h(a, goal))

    s = start
    expanded = 0
    visited: Set[Cell] = set()

    while True:
        visited.add(s); expanded += 1
        Hest.setdefault(s, h(s, goal))
        yield Frame(expand=s, frontier=set(v for v,_ in grid.neighbors(s)),
                    closed=set(visited), best_path=[])

        if s == goal:
            res = SearchResult("LRTA*", True, [s], 0.0, expanded, time.perf_counter()-t0)
            yield Frame(s, set(), set(visited), [s], True, res); return res

        # update H(s)
        nbrs = [v for v,_ in grid.neighbors(s)]
        if not nbrs:
            res = SearchResult("LRTA*", False, [], float("inf"), expanded, time.perf_counter()-t0)
            yield Frame(s, set(), set(visited), [], True, res); return res

        Hest[s] = min(LRTA_cost(s, v) for v in nbrs)
        # pick action
        s = min(nbrs, key=lambda v: LRTA_cost(s, v))

# --------------------------
# AND–OR Planner (nondeterminism via slip-to-stay)
# --------------------------

def and_or_plan_steps(grid: Grid, p_slip: float = 0.2, depth_limit: int = 200) -> Generator[Frame, None, SearchResult]:
    """
    Build a contingent plan for a simple nondeterministic model:
    RESULTS(s,a) = {s, move(s,a)} when a is legal (slips by staying put).
    Then execute the plan on a simulated run with actual slips and animate steps.
    """
    start, goal = grid.start, grid.goal
    if start is None or goal is None: raise RuntimeError("Set Start/Goal.")
    t0 = time.perf_counter()

    # legal actions = neighbors' targets
    def actions(s: Cell) -> List[Cell]:
        return [v for v,_ in grid.neighbors(s)]

    # RESULT set for nondeterminism: either move succeeds or slip => stay put
    def RESULTS(s: Cell, a: Cell) -> Set[Cell]:
        return {s, a}

    # Depth-first AND-OR search with cycle check
    visited_nodes = 0
    def OR_SEARCH(s: Cell, path: List[Cell]) -> Optional[Dict]:
        nonlocal visited_nodes
        visited_nodes += 1
        if s == goal: return {}  # empty plan
        if s in path or len(path) > depth_limit: return None
        for a in actions(s):
            plan = AND_SEARCH(RESULTS(s, a), path+[s])
            if plan is not None:
                # return plan: at s take action a, then follow subplans for outcomes
                return {"s": s, "a": a, "out": plan}
        return None

    def AND_SEARCH(states: Set[Cell], path: List[Cell]) -> Optional[Dict]:
        branches = {}
        for sp in states:
            sub = OR_SEARCH(sp, path)
            if sub is None:
                return None
            branches[sp] = sub
        return branches

    plan = OR_SEARCH(start, [])
    # visualize planning as a frame (we use 'closed' to show explored)
    explored = set()  # (best-effort; not collecting all)
    yield Frame(expand=None, frontier=set(), closed=explored, best_path=[])

    if plan is None:
        res = SearchResult(f"AND–OR (p_slip={p_slip:.2f})", False, [], float("inf"), visited_nodes, time.perf_counter()-t0)
        yield Frame(None, set(), explored, [], True, res)
        return res

    # Execute the plan once in a simulated slippery environment
    def execute(policy, s: Cell, max_steps=200) -> List[Cell]:
        path = [s]
        for _ in range(max_steps):
            if s == goal: break
            a = policy.get("a") if policy and policy.get("s")==s else None
            if a is None:  # find an entry keyed by current s among branches (if nested)
                # descend using outcome mapping if needed
                # try to find a mapping whose key contains current s
                # (this is a minimal, functional policy representation)
                # If not found, stop.
                break
            # slip?
            if random.random() < p_slip:
                s_next = s
            else:
                s_next = a
            path.append(s_next)
            # descend subplan
            policy = policy.get("out", {}).get(s_next, {})
            s = s_next
        return path

    exec_path = execute(plan, start)
    reached = exec_path[-1] == goal
    yield Frame(None, frontier=set(), closed=set(exec_path), best_path=exec_path)

    res = SearchResult(f"AND–OR (p_slip={p_slip:.2f})", reached, exec_path,
                       float(len(exec_path)-1), visited_nodes, time.perf_counter()-t0)
    yield Frame(None, set(), set(exec_path), exec_path, True, res)
    return res

# --------------------------
# GUI
# --------------------------

class Chapter4Lab:
    def __init__(self, root: tk.Tk, rows=25, cols=25, cell=22):
        self.root = root
        self.rows, self.cols, self.cell = rows, cols, cell
        self.margin = 10
        self.grid = Grid(rows, cols)
        self.mode = tk.StringVar(value="wall")
        self.algo = tk.StringVar(value="A* (for ref)")
        self.speed_ms = tk.IntVar(value=10)
        self.diag = tk.BooleanVar(value=False)

        # Params
        self.k_beam = tk.IntVar(value=12)
        self.tau = tk.DoubleVar(value=1.0)
        self.ga_pop = tk.IntVar(value=60)
        self.ga_len = tk.IntVar(value=0)  # 0 => auto
        self.ga_mut = tk.DoubleVar(value=0.10)
        self.sa_T0 = tk.DoubleVar(value=3.0)
        self.sa_alpha = tk.DoubleVar(value=0.995)
        self.slip = tk.DoubleVar(value=0.20)

        self.running = False
        self._gen: Optional[Generator[Frame, None, SearchResult]] = None
        self._after = None

        self._build_ui()
        self._draw()

    def _build_ui(self):
        self.root.title("Chapter 4 Maze Lab")
        wrap = ttk.Frame(self.root, padding=6); wrap.pack(fill="both", expand=True)
        cw = self.cols*self.cell + 2*self.margin
        ch = self.rows*self.cell + 2*self.margin
        self.canvas = tk.Canvas(wrap, width=cw, height=ch, bg="white")
        self.canvas.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0,6))
        wrap.columnconfigure(0, weight=1); wrap.rowconfigure(0, weight=1)
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_click)

        right = ttk.Frame(wrap); right.grid(row=0, column=1, sticky="ns")

        ttk.Label(right, text="Algorithm").grid(row=0, column=0, sticky="w")
        values = [
            "Hill-Climbing",
            "Simulated Annealing",
            "Beam (k-best)",
            "Stochastic Beam",
            "Genetic",
            "Online DFS",
            "LRTA*",
            "AND–OR (slippery)",
            "A* (for ref)"  # reference baseline using best-first
        ]
        cb = ttk.Combobox(right, textvariable=self.algo, state="readonly", values=values, width=24)
        cb.grid(row=1, column=0, sticky="ew", pady=(0,6))
        cb.bind("<<ComboboxSelected>>", lambda e: self._update_params())

        ttk.Label(right, text="Paint Mode").grid(row=2, column=0, sticky="w")
        for i,(label,val) in enumerate([("Wall","wall"),("Erase","erase"),("Start","start"),("Goal","goal")], start=3):
            ttk.Radiobutton(right, text=label, variable=self.mode, value=val).grid(row=i, column=0, sticky="w")

        ttk.Separator(right).grid(row=7, column=0, sticky="ew", pady=6)
        ttk.Checkbutton(right, text="Allow diagonals", variable=self.diag, command=self._toggle_diag).grid(row=8, column=0, sticky="w")
        ttk.Label(right, text="Speed (ms/step)").grid(row=9, column=0, sticky="w")
        tk.Scale(right, from_=0, to=200, orient="horizontal", variable=self.speed_ms).grid(row=10, column=0, sticky="ew")

        # Param boxes
        self.box_beam = ttk.LabelFrame(right, text="Beam params")
        ttk.Label(self.box_beam, text="k").grid(row=0, column=0, sticky="w")
        tk.Scale(self.box_beam, from_=2, to=200, orient="horizontal", variable=self.k_beam).grid(row=1, column=0, sticky="ew")

        self.box_sbeam = ttk.LabelFrame(right, text="Stochastic Beam params")
        ttk.Label(self.box_sbeam, text="k").grid(row=0, column=0, sticky="w")
        tk.Scale(self.box_sbeam, from_=2, to=200, orient="horizontal", variable=self.k_beam).grid(row=1, column=0, sticky="ew")
        ttk.Label(self.box_sbeam, text="τ (temperature)").grid(row=2, column=0, sticky="w")
        tk.Scale(self.box_sbeam, from_=0.2, to=3.0, resolution=0.05, orient="horizontal", variable=self.tau).grid(row=3, column=0, sticky="ew")

        self.box_ga = ttk.LabelFrame(right, text="Genetic params")
        ttk.Label(self.box_ga, text="Population").grid(row=0, column=0, sticky="w")
        tk.Scale(self.box_ga, from_=10, to=200, orient="horizontal", variable=self.ga_pop).grid(row=1, column=0, sticky="ew")
        ttk.Label(self.box_ga, text="Genome length (0=auto)").grid(row=2, column=0, sticky="w")
        tk.Scale(self.box_ga, from_=0, to=300, orient="horizontal", variable=self.ga_len).grid(row=3, column=0, sticky="ew")
        ttk.Label(self.box_ga, text="Mutation rate").grid(row=4, column=0, sticky="w")
        tk.Scale(self.box_ga, from_=0.0, to=0.5, resolution=0.01, orient="horizontal", variable=self.ga_mut).grid(row=5, column=0, sticky="ew")

        self.box_sa = ttk.LabelFrame(right, text="SA params")
        ttk.Label(self.box_sa, text="T0 (start temp)").grid(row=0, column=0, sticky="w")
        tk.Scale(self.box_sa, from_=0.5, to=10.0, resolution=0.1, orient="horizontal", variable=self.sa_T0).grid(row=1, column=0, sticky="ew")
        ttk.Label(self.box_sa, text="alpha (cool rate)").grid(row=2, column=0, sticky="w")
        tk.Scale(self.box_sa, from_=0.90, to=0.999, resolution=0.001, orient="horizontal", variable=self.sa_alpha).grid(row=3, column=0, sticky="ew")

        self.box_slip = ttk.LabelFrame(right, text="Slippery moves")
        ttk.Label(self.box_slip, text="p_slip (stay put)").grid(row=0, column=0, sticky="w")
        tk.Scale(self.box_slip, from_=0.0, to=0.5, resolution=0.01, orient="horizontal", variable=self.slip).grid(row=1, column=0, sticky="ew")

        self._update_params()

        ttk.Separator(right).grid(row=30, column=0, sticky="ew", pady=6)
        ttk.Button(right, text="Run", command=self.run).grid(row=31, column=0, sticky="ew")
        ttk.Button(right, text="Stop", command=self.stop).grid(row=32, column=0, sticky="ew")
        ttk.Button(right, text="Step Once", command=self.step_once).grid(row=33, column=0, sticky="ew")

        ttk.Separator(right).grid(row=34, column=0, sticky="ew", pady=6)
        self.rand_density = tk.DoubleVar(value=0.25)
        ttk.Label(right, text="Random obstacles").grid(row=35, column=0, sticky="w")
        tk.Scale(right, from_=0.0, to=0.7, resolution=0.05, orient="horizontal", variable=self.rand_density).grid(row=36, column=0, sticky="ew")
        ttk.Button(right, text="Fill random", command=self.random_fill).grid(row=37, column=0, sticky="ew")
        ttk.Button(right, text="Clear walls", command=self.clear_walls).grid(row=38, column=0, sticky="ew")
        ttk.Button(right, text="Reset colors", command=lambda: self._draw()).grid(row=39, column=0, sticky="ew")

        stats = ttk.LabelFrame(wrap, text="Stats")
        stats.grid(row=1, column=1, sticky="ew")
        self.lbl = ttk.Label(stats, text="Ready.")
        self.lbl.pack(anchor="w")

    def _update_params(self):
        # Hide/Show param boxes based on algorithm
        for box in (self.box_beam, self.box_sbeam, self.box_ga, self.box_sa, self.box_slip):
            box.grid_forget()
        a = self.algo.get()
        row = 12
        if a == "Beam (k-best)":
            self.box_beam.grid(row=row, column=1, sticky="ew")
        elif a == "Stochastic Beam":
            self.box_sbeam.grid(row=row, column=1, sticky="ew")
        elif a == "Genetic":
            self.box_ga.grid(row=row, column=1, sticky="ew")
        elif a == "Simulated Annealing":
            self.box_sa.grid(row=row, column=1, sticky="ew")
        elif a == "AND–OR (slippery)":
            self.box_slip.grid(row=row, column=1, sticky="ew")

    def _toggle_diag(self):
        self.grid.allow_diagonals = self.diag.get()
        self._draw()

    # Drawing
    def _bbox(self, r, c):
        x0 = self.margin + c*self.cell
        y0 = self.margin + r*self.cell
        return (x0, y0, x0+self.cell, y0+self.cell)

    def _draw(self, frame: Optional[Frame]=None):
        cv = self.canvas; cv.delete("all")
        for r in range(self.rows):
            for c in range(self.cols):
                x0,y0,x1,y1 = self._bbox(r,c)
                fill = "white"
                if (r,c) in self.grid.walls:
                    fill = "#222"
                cv.create_rectangle(x0, y0, x1, y1, fill=fill, outline="#ddd")

        if frame:
            for (r,c) in frame.closed:
                if (r,c) not in self.grid.walls:
                    x0,y0,x1,y1 = self._bbox(r,c)
                    cv.create_rectangle(x0,y0,x1,y1, fill="#bfe3ff", outline="#cde8ff")
            for (r,c) in frame.frontier:
                if (r,c) not in self.grid.walls:
                    x0,y0,x1,y1 = self._bbox(r,c)
                    cv.create_rectangle(x0,y0,x1,y1, fill="#ffd4a3", outline="#ffddb7")
            for (r,c) in frame.best_path:
                if (r,c) not in self.grid.walls:
                    x0,y0,x1,y1 = self._bbox(r,c)
                    cv.create_rectangle(x0,y0,x1,y1, fill="#fff59c", outline="#fff59c")
            if frame.expand:
                r,c = frame.expand
                x0,y0,x1,y1 = self._bbox(r,c)
                cv.create_rectangle(x0,y0,x1,y1, fill="#f9a8d4", outline="#f9a8d4")

        if self.grid.start:
            x0,y0,x1,y1 = self._bbox(*self.grid.start)
            cv.create_rectangle(x0,y0,x1,y1, fill="#34d399", outline="#34d399")
        if self.grid.goal:
            x0,y0,x1,y1 = self._bbox(*self.grid.goal)
            cv.create_rectangle(x0,y0,x1,y1, fill="#ef4444", outline="#ef4444")

    # Mouse
    def on_click(self, e: tk.Event):
        r = (e.y - self.margin) // self.cell
        c = (e.x - self.margin) // self.cell
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            return
        p = (int(r), int(c))
        m = self.mode.get()
        if m == "wall":
            if p != self.grid.start and p != self.grid.goal:
                self.grid.walls.add(p)
        elif m == "erase":
            self.grid.walls.discard(p)
        elif m == "start":
            if p != self.grid.goal and p not in self.grid.walls:
                self.grid.start = p
        elif m == "goal":
            if p != self.grid.start and p not in self.grid.walls:
                self.grid.goal = p
        self._draw()

    # Randomize/Clear
    def random_fill(self):
        self.grid.walls.clear()
        dens = self.rand_density.get()
        for r in range(self.rows):
            for c in range(self.cols):
                if random.random() < dens:
                    self.grid.walls.add((r,c))
        for pin in (self.grid.start, self.grid.goal):
            if pin: self.grid.walls.discard(pin)
        self._draw()

    def clear_walls(self):
        self.grid.walls.clear(); self._draw()

    # Run / animate
    def pick_gen(self) -> Generator[Frame, None, SearchResult]:
        a = self.algo.get()
        self.grid.allow_diagonals = self.diag.get()
        if a == "Hill-Climbing":
            return hill_climb_steps(self.grid)
        if a == "Simulated Annealing":
            return simulated_annealing_steps(self.grid, self.sa_T0.get(), self.sa_alpha.get())
        if a == "Beam (k-best)":
            return beam_steps(self.grid, self.k_beam.get())
        if a == "Stochastic Beam":
            return stochastic_beam_steps(self.grid, self.k_beam.get(), self.tau.get())
        if a == "Genetic":
            L = self.ga_len.get() or None
            return genetic_steps(self.grid, pop_size=self.ga_pop.get(),
                                 genome_len=L, mutation_rate=self.ga_mut.get())
        if a == "Online DFS":
            return online_dfs_steps(self.grid)
        if a == "LRTA*":
            return lrta_star_steps(self.grid)
        if a == "AND–OR (slippery)":
            return and_or_plan_steps(self.grid, p_slip=self.slip.get())
        # Reference A* (best-first engine)
        return best_first_steps(self.grid, "A* (ref)", priority=lambda g,h: g+h)

    def run(self):
        if self.running: return
        try:
            if self.grid.start is None or self.grid.goal is None:
                messagebox.showinfo("Chapter 4 Lab", "Set both Start and Goal.")
                return
            if self.grid.start in self.grid.walls or self.grid.goal in self.grid.walls:
                messagebox.showinfo("Chapter 4 Lab", "Start/Goal cannot be walls.")
                return
            self._gen = self.pick_gen()
            self.running = True
            self.lbl.config(text=f"Running {self.algo.get()}...")
            self._advance()
        except Exception as e:
            messagebox.showerror("Chapter 4 Lab", str(e))
            self.running = False; self._gen = None

    def stop(self):
        if self._after:
            self.canvas.after_cancel(self._after)
            self._after = None
        self.running = False; self._gen = None
        self.lbl.config(text="Stopped.")

    def step_once(self):
        if self._gen is None:
            try:
                self._gen = self.pick_gen()
            except Exception as e:
                messagebox.showerror("Chapter 4 Lab", str(e)); return
        try:
            fr = next(self._gen)
            self._draw(fr)
            if fr.done and fr.final:
                self._finish(fr.final)
                self._gen = None
        except StopIteration:
            self._gen = None

    def _advance(self):
        if not self.running or self._gen is None: return
        try:
            fr = next(self._gen)
            self._draw(fr)
            if fr.done and fr.final:
                self._finish(fr.final)
                self.running = False; self._gen = None
                return
            self._after = self.canvas.after(self.speed_ms.get(), self._advance)
        except StopIteration:
            self.running = False; self._gen = None
            self.lbl.config(text="Finished.")

    def _finish(self, res: SearchResult):
        if res.success:
            self.lbl.config(text=f"{res.name}: OK | cost={res.cost:.3f} | expanded={res.nodes_expanded} | time={res.time_s*1000:.2f} ms")
        else:
            self.lbl.config(text=f"{res.name}: FAIL | expanded={res.nodes_expanded} | time={res.time_s*1000:.2f} ms")

# --------------------------
# Main
# --------------------------

if __name__ == "__main__":
    random.seed()
    root = tk.Tk()
    app = Chapter4Lab(root, rows=25, cols=25, cell=22)
    app.grid.start = (2,2)
    app.grid.goal = (22,22)
    app._draw()
    root.mainloop()
