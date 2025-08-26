# chapter4_lab_streamlit.py
# A headless-friendly maze lab with multiple algorithms:
# - BFS, Greedy Best-First, A*, Weighted A*
# - Hill Climbing (steepest descent on heuristic)
# - Simulated Annealing
# - Beam Search (k-beam)
#
# Runs in Streamlit. No Tk/X display required.

import time
import math
import random
from collections import deque
from heapq import heappush, heappop

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt


# -----------------------
# Maze generation & utils
# -----------------------

FREE, WALL = 0, 1

def neighbors4(r, c, R, C):
    for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
        nr, nc = r+dr, c+dc
        if 0 <= nr < R and 0 <= nc < C:
            yield nr, nc

def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def draw_grid(grid, start, goal, path=None, visited_counts=None, title=""):
    R, C = grid.shape
    # Base image: free=1 (white), wall=0 (black)
    img = np.ones((R, C, 3), dtype=float)
    img[grid == WALL] = 0.0

    # Optional visitation heatmap overlay (light blue tint)
    if visited_counts is not None:
        vmax = visited_counts.max() if visited_counts.size and visited_counts.max() > 0 else 1.0
        if vmax > 0:
            alpha = (visited_counts / vmax) * 0.6  # scale alpha
            blue = np.dstack([np.zeros_like(alpha), np.zeros_like(alpha), np.ones_like(alpha)])
            img = (1 - alpha[..., None]) * img + alpha[..., None] * blue

    # Start & goal highlights
    sr, sc = start
    gr, gc = goal
    img[sr, sc] = [0.2, 0.8, 0.2]   # green
    img[gr, gc] = [0.2, 0.2, 0.9]   # blue

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)

    # Path overlay
    if path:
        ys = [c for r, c in path]
        xs = [r for r, c in path]
        ax.plot(ys, xs, linewidth=2.5)

    st.pyplot(fig)

def generate_maze(R, C, density=0.25, rng=None, max_tries=50):
    """Generate a random maze and ensure it's solvable by BFS."""
    rng = rng or random.Random()
    for _ in range(max_tries):
        grid = np.full((R, C), FREE, dtype=np.int8)
        # sprinkle walls
        for r in range(R):
            for c in range(C):
                if (r, c) not in [(0, 0), (R-1, C-1)] and rng.random() < density:
                    grid[r, c] = WALL
        # ensure solvable via a quick BFS
        ok = bfs(grid, (0, 0), (R-1, C-1), measure_only=True)["success"]
        if ok:
            return grid
    return grid  # last try even if not solvable (rare with low density)


# -----------------------
# Algorithm implementations
# -----------------------

def reconstruct_path(parent, end):
    path = []
    cur = end
    while cur in parent:
        path.append(cur)
        cur = parent[cur]
    path.append(cur)
    path.reverse()
    return path

def bfs(grid, start, goal, measure_only=False):
    R, C = grid.shape
    sr, sc = start
    gr, gc = goal
    t0 = time.perf_counter()
    q = deque([(sr, sc)])
    parent = {}
    seen = {start}
    expanded = 0
    visit_counts = np.zeros_like(grid, dtype=np.int32)

    while q:
        r, c = q.popleft()
        expanded += 1
        visit_counts[r, c] += 1
        if (r, c) == (gr, gc):
            t1 = time.perf_counter()
            path = [] if measure_only else reconstruct_path(parent, (gr, gc))
            return {
                "name": "BFS",
                "success": True,
                "path": path,
                "cost": len(path)-1 if path else None,
                "expanded": expanded,
                "time": t1 - t0,
                "visited": visit_counts,
            }
        for nr, nc in neighbors4(r, c, R, C):
            if grid[nr, nc] == WALL:
                continue
            if (nr, nc) not in seen:
                seen.add((nr, nc))
                parent[(nr, nc)] = (r, c)
                q.append((nr, nc))
    t1 = time.perf_counter()
    return {
        "name": "BFS",
        "success": False,
        "path": None,
        "cost": None,
        "expanded": expanded,
        "time": t1 - t0,
        "visited": visit_counts,
    }

def greedy_best_first(grid, start, goal):
    R, C = grid.shape
    t0 = time.perf_counter()
    openh = []
    heappush(openh, (manhattan(start, goal), start))
    parent, seen = {}, {start}
    expanded = 0
    visit_counts = np.zeros_like(grid, dtype=np.int32)

    while openh:
        _, (r, c) = heappop(openh)
        expanded += 1
        visit_counts[r, c] += 1
        if (r, c) == goal:
            t1 = time.perf_counter()
            path = reconstruct_path(parent, goal)
            return {
                "name": "Greedy",
                "success": True,
                "path": path,
                "cost": len(path)-1,
                "expanded": expanded,
                "time": t1 - t0,
                "visited": visit_counts,
            }
        for nr, nc in neighbors4(r, c, R, C):
            if grid[nr, nc] == WALL: 
                continue
            if (nr, nc) not in seen:
                seen.add((nr, nc))
                parent[(nr, nc)] = (r, c)
                heappush(openh, (manhattan((nr, nc), goal), (nr, nc)))

    t1 = time.perf_counter()
    return {"name": "Greedy", "success": False, "path": None, "cost": None,
            "expanded": expanded, "time": t1 - t0, "visited": visit_counts}

def a_star(grid, start, goal, w=1.0):
    R, C = grid.shape
    t0 = time.perf_counter()
    g = {start: 0}
    parent = {}
    openh = []
    heappush(openh, (w * manhattan(start, goal), start))
    in_open = {start}
    expanded = 0
    visit_counts = np.zeros_like(grid, dtype=np.int32)

    while openh:
        f, cur = heappop(openh)
        in_open.discard(cur)
        r, c = cur
        expanded += 1
        visit_counts[r, c] += 1

        if cur == goal:
            t1 = time.perf_counter()
            path = reconstruct_path(parent, goal)
            return {
                "name": "A*" if w == 1.0 else f"WeightedA*(w={w})",
                "success": True,
                "path": path,
                "cost": g[cur],
                "expanded": expanded,
                "time": t1 - t0,
                "visited": visit_counts,
            }
        for nr, nc in neighbors4(r, c, R, C):
            if grid[nr, nc] == WALL:
                continue
            ng = g[cur] + 1
            nxt = (nr, nc)
            if ng < g.get(nxt, 1e18):
                g[nxt] = ng
                parent[nxt] = cur
                f2 = ng + w * manhattan(nxt, goal)
                if nxt not in in_open:
                    heappush(openh, (f2, nxt))
                    in_open.add(nxt)

    t1 = time.perf_counter()
    return {"name": "A*", "success": False, "path": None, "cost": None,
            "expanded": expanded, "time": t1 - t0, "visited": visit_counts}

def hill_climb(grid, start, goal, max_steps=500, tie_break="random"):
    """Steepest descent on h = Manhattan distance. May get stuck."""
    R, C = grid.shape
    t0 = time.perf_counter()
    cur = start
    path = [cur]
    expanded = 0
    visit_counts = np.zeros_like(grid, dtype=np.int32)

    for _ in range(max_steps):
        if cur == goal:
            t1 = time.perf_counter()
            return {"name": "HillClimb", "success": True, "path": path, "cost": len(path)-1,
                    "expanded": expanded, "time": t1 - t0, "visited": visit_counts}
        r, c = cur
        visit_counts[r, c] += 1
        expanded += 1
        h0 = manhattan(cur, goal)
        nbrs = [(manhattan((nr, nc), goal), (nr, nc)) for nr, nc in neighbors4(r, c, R, C)
                if grid[nr, nc] != WALL]
        if not nbrs:
            break
        best_h = min(h for h, _ in nbrs)
        if best_h >= h0:
            # no improvement → stuck
            break
        candidates = [s for h, s in nbrs if h == best_h]
        nxt = random.choice(candidates) if tie_break == "random" else candidates[0]
        cur = nxt
        path.append(cur)

    t1 = time.perf_counter()
    return {"name": "HillClimb", "success": cur == goal, "path": path if cur == goal else path,
            "cost": len(path)-1 if cur == goal else None,
            "expanded": expanded, "time": t1 - t0, "visited": visit_counts}

def simulated_annealing(grid, start, goal, T0=2.0, alpha=0.995, max_steps=5000):
    """SA on h = Manhattan; accepts worse moves with prob exp(-(ΔE)/T)."""
    R, C = grid.shape
    t0 = time.perf_counter()
    rng = random.Random(0)
    cur = start
    best = cur
    best_h = manhattan(cur, goal)
    path = [cur]
    expanded = 0
    visit_counts = np.zeros_like(grid, dtype=np.int32)

    T = T0
    for step in range(max_steps):
        if cur == goal:
            t1 = time.perf_counter()
            return {"name": "SimAnneal", "success": True, "path": path, "cost": len(path)-1,
                    "expanded": expanded, "time": t1 - t0, "visited": visit_counts}
        r, c = cur
        visit_counts[r, c] += 1
        expanded += 1
        nbrs = [(nr, nc) for nr, nc in neighbors4(r, c, R, C) if grid[nr, nc] != WALL]
        if not nbrs:
            break
        nxt = rng.choice(nbrs)
        h_cur = manhattan(cur, goal)
        h_nxt = manhattan(nxt, goal)
        dE = (h_nxt - h_cur)  # minimize h
        accept = (dE < 0) or (rng.random() < math.exp(-dE / max(T, 1e-9)))
        if accept:
            cur = nxt
            path.append(cur)
            if h_nxt < best_h:
                best, best_h = cur, h_nxt
        T *= alpha

    t1 = time.perf_counter()
    return {"name": "SimAnneal", "success": cur == goal, "path": path if cur == goal else None,
            "cost": len(path)-1 if cur == goal else None,
            "expanded": expanded, "time": t1 - t0, "visited": visit_counts}

def beam_search(grid, start, goal, k=10, max_iters=5000, stochastic=False):
    """Keep k best states by h; expand all; repeat. Track paths per candidate."""
    R, C = grid.shape
    t0 = time.perf_counter()
    rng = random.Random(0)
    beam = [(manhattan(start, goal), start, [start])]
    expanded = 0
    visit_counts = np.zeros_like(grid, dtype=np.int32)

    for it in range(max_iters):
        # goal check
        for h, s, p in beam:
            visit_counts[s[0], s[1]] += 1
            if s == goal:
                t1 = time.perf_counter()
                return {"name": f"Beam(k={k}{',stoch' if stochastic else ''})",
                        "success": True, "path": p, "cost": len(p)-1,
                        "expanded": expanded, "time": t1 - t0, "visited": visit_counts}

        # expand all
        pool = []
        seen_here = set()
        for h, s, p in beam:
            r, c = s
            for nr, nc in neighbors4(r, c, R, C):
                if grid[nr, nc] == WALL: 
                    continue
                ns = (nr, nc)
                if ns in p:  # avoid immediate cycles
                    continue
                expanded += 1
                npth = p + [ns]
                heappush(pool, (manhattan(ns, goal), ns, npth))

        if not pool:
            break

        # pick next beam
        if stochastic:
            # sample by softmax-ish weights (1/(1+h))
            items = []
            weights = []
            while pool:
                h, s, p = heappop(pool)
                items.append((h, s, p))
                weights.append(1.0 / (1.0 + h))
            if sum(weights) == 0:
                # fallback to plain best k
                items = sorted(items, key=lambda x: x[0])[:k]
            else:
                # sample w/o replacement
                chosen = []
                for _ in range(min(k, len(items))):
                    total = sum(weights)
                    r = rng.random() * total
                    cum = 0.0
                    for i, w in enumerate(weights):
                        cum += w
                        if cum >= r:
                            chosen.append(items.pop(i))
                            weights.pop(i)
                            break
                items = chosen
            beam = items
        else:
            # deterministic top-k
            next_beam = []
            while pool and len(next_beam) < k:
                h, s, p = heappop(pool)
                if (h, s) in seen_here:
                    continue
                seen_here.add((h, s))
                next_beam.append((h, s, p))
            beam = next_beam

    t1 = time.perf_counter()
    return {"name": f"Beam(k={k}{',stoch' if stochastic else ''})",
            "success": False, "path": None, "cost": None,
            "expanded": expanded, "time": t1 - t0, "visited": visit_counts}


# -----------------------
# Streamlit UI
# -----------------------

st.set_page_config(page_title="Chapter 4 – Maze Lab", layout="centered")
st.title("Chapter 4 – Local & Heuristic Search Maze Lab (Headless-Friendly)")

with st.sidebar:
    st.header("Maze")
    R = st.slider("Rows", 8, 60, 20, 1)
    C = st.slider("Cols", 8, 60, 20, 1)
    density = st.slider("Wall density", 0.0, 0.6, 0.25, 0.01)
    seed = st.number_input("Random seed", value=0, step=1, min_value=0)
    ensure_solvable = st.checkbox("Ensure solvable at generation (BFS check)", value=True)
    st.markdown("---")
    st.header("Algorithm")
    algo = st.selectbox(
        "Pick an algorithm",
        [
            "BFS",
            "Greedy Best-First",
            "A*",
            "Weighted A*",
            "Hill Climbing",
            "Simulated Annealing",
            "Beam Search",
            "Stochastic Beam Search",
        ],
    )

    if algo == "Weighted A*":
        w = st.slider("Weight (w)", 1.0, 5.0, 1.5, 0.1)
    if algo == "Hill Climbing":
        hc_steps = st.slider("Max steps", 50, 5000, 500, 50)
    if algo == "Simulated Annealing":
        T0 = st.slider("Initial T", 0.1, 10.0, 2.0, 0.1)
        alpha = st.slider("Cooling α", 0.900, 0.999, 0.995, 0.001, format="%.3f")
        sa_steps = st.slider("Max steps", 100, 20000, 5000, 100)
    if "Beam" in algo:
        k = st.slider("Beam width (k)", 2, 200, 30, 2)

    st.markdown("---")
    show_heat = st.checkbox("Show expansion heatmap", value=True)
    run_btn = st.button("Generate & Solve")

if run_btn:
    rng = random.Random(seed)
    np.random.seed(seed)

    # Generate maze
    grid = generate_maze(R, C, density=density, rng=rng) if ensure_solvable else \
           (lambda: (lambda g: g)(np.where(np.random.rand(R, C) < density, WALL, FREE).astype(np.int8)))()
    grid[0,0] = FREE
    grid[R-1,C-1] = FREE
    start, goal = (0,0), (R-1, C-1)

    st.subheader("Maze")
    draw_grid(grid, start, goal, path=None, visited_counts=None, title="Generated Maze")

    # Run algorithm
    if algo == "BFS":
        res = bfs(grid, start, goal)
    elif algo == "Greedy Best-First":
        res = greedy_best_first(grid, start, goal)
    elif algo == "A*":
        res = a_star(grid, start, goal, w=1.0)
    elif algo == "Weighted A*":
        res = a_star(grid, start, goal, w=w)
    elif algo == "Hill Climbing":
        res = hill_climb(grid, start, goal, max_steps=hc_steps)
    elif algo == "Simulated Annealing":
        res = simulated_annealing(grid, start, goal, T0=T0, alpha=alpha, max_steps=sa_steps)
    elif algo == "Beam Search":
        res = beam_search(grid, start, goal, k=k, stochastic=False)
    elif algo == "Stochastic Beam Search":
        res = beam_search(grid, start, goal, k=k, stochastic=True)
    else:
        res = {"success": False, "path": None, "visited": np.zeros_like(grid)}

    st.subheader("Result")
    st.write({
        "algorithm": res.get("name"),
        "success": res.get("success"),
        "cost": res.get("cost"),
        "nodes_expanded": res.get("expanded"),
        "time_s": round(res.get("time", 0.0), 6)
    })

    draw_grid(
        grid,
        start,
        goal,
        path=res.get("path"),
        visited_counts=res.get("visited") if show_heat else None,
        title=f"{res.get('name')} – {'SUCCESS' if res.get('success') else 'FAIL'}",
    )

else:
    st.info("Set your maze and algorithm in the sidebar, then click **Generate & Solve**.")


