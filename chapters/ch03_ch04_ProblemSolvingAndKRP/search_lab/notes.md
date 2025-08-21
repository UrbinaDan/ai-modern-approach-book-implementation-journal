# Chapter 3 — Solving Problems by Searching (Notes)
What this folder contains

- A clean, modular implementation of uninformed and informed search algorithms.
- A standard problem formulation interface (initial state, ACTIONS, RESULT, IS_GOAL, c).
- A small benchmark harness and plotting utilities.
- A canonical testbed: Romania map problem (AIMA classic).

Paths:

- Algorithms: search_lab/algorithms/
- Core (node, frontiers, metrics): search_lab/core/
- Problems: search_lab/problems/
- Benchmarks & plots: search_lab/benchmarks/

# Problem formulation (state-space view)
- Each Problem must provide:
- Initial state initial_state()
- Actions actions(s) (finite set)
- Transition model result(s, a) → s′
- Goal test is_goal(s)
- Action cost cost(s, a, s′) = c(s,a,s′)
- States are atomic (no internal structure is assumed by the algorithms)

Why: this uniform API lets every algorithm plug into any problem with minimal glue.

# Algorithms implemented (1-liners)
**Uninformed**
- BFS — explores by depth; optimal for unit costs.
- UCS (Dijkstra) — explores by path cost; optimal for positive costs.
- DFS — goes deep; low memory, not complete/optimal in general.
- DLS — DFS with a depth cutoff ℓ.
- IDS — iterative deepening over ℓ; complete (with cycle checks), optimal for unit costs.
- Bidirectional (BFS/UCS/A*) — grow two frontiers (start↔goal); can be much faster if goal is a single known state and reverse operators exist.

**Informed**
- Greedy best-first — expand lowest h(n); fast but not optimal.
- A* — expand lowest f(n)=g(n)+h(n); optimal if admissible (and consistent) h.
- IDA* — A* with iterative deepening on f; tiny memory, more re-expansions.
- RBFS — A*-like with linear space; backs up best alternative f limits.
- SMA* — A* with a hard memory cap; evicts worst leaves, backs up values.
- Beam search — keep top-k nodes by f; incomplete, often good-enough quickly.
- Weighted A* — f(n)=g(n)+w·h(n) (w>1); faster, not optimal.

# How to run the ch03.../seacrhc lab
1. Do cd.. ch03..
2. start the venv 
3. run this python -m search_lab.benchmarks.plot_results

Outputs:

- JSON: search_lab/benchmarks/results.json
- Markdown summary: search_lab/benchmarks/results.md
- Plots: nodes_expanded.png, time.png, cost.png

Heuristics: quick reminders
- Heuristic h(n) estimates remaining cost.
- Admissible: h(n) ≤ h*(n) (never overestimates) → A* is optimal.
- Consistent: h(n) ≤ c(n,a,n′) + h(n′) (triangle inequality) → no reopens needed.
- Weighted A*: trade optimality for speed (bigger w → greedier).

Mini cheat-sheet (theory-level bounds)
Algorithm	Complete	Optimal	Time (worst-case)	Space
BFS	Yes	Yes (unit costs)	O(b^d)	O(b^d)
UCS	Yes (c>0)	Yes	O(b^(1+⌊C*/ε⌋))	Same order
DFS	No	No	O(b^m)	O(b·m)
DLS(ℓ)	No (if ℓ<d)	No	O(b^ℓ)	O(b^ℓ)
IDS	Yes	Yes (unit)	O(b^d)	O(b·d)
Greedy	Yes (finite)	No	O(	V
A* (adm.)	Yes	Yes	Depends on h quality	O(#frontier)
IDA*	Yes	Yes	Often > A* (revisits)	O(d)
RBFS	Yes	Yes (adm.)	Varies (revisits)	O(d)
SMA*	If solution fits	If optimal path fits	A*-like w/ eviction	Up to cap
Beam(k)	No	No	Fast in practice	O(k)
Bi-(BFS/UCS/A*)	Yes (reqs met)	Yes (with UCS/A*)	~O(b^(d/2))	Two frontiers

Symbols:
b branching factor, d optimal depth, m max depth, C* optimal cost, ε min action cost.

Benchmarks (Romania)

- Problem: route from Arad → Bucharest
- Heuristic: straight-line distance to Bucharest (AIMA)
- Typical observations (your exact numbers in results.md):
    - A* finds the optimal path with fewer expansions than UCS.
    - Greedy expands very few nodes but often returns a suboptimal path.
    - IDA* / RBFS use tiny memory but pay with more re-expansions.
    - SMA* performs like A* until memory pressure; then it evicts worst leaves and backs up costs.
# Design choices (why this structure)
- Frontiers are abstracted (FIFOQueue, LIFOStack, PriorityQueue) so algorithms map to data structures cleanly.
- Node stores (state, parent, action, path_cost, depth), enabling path reconstruction and cheap cycle checks.
- reached (hash set / dict) avoids redundant paths; controlled per algorithm (e.g., tree-like DFS skips it).
- Metrics via MeasuredRun (wall time, nodes expanded, peak KB) standardize benchmark output.

# Common gotchas (and how we handled them)
- Early vs late goal test:
    - UCS/A*: goal should be tested on expansion, not on generation (prevents returning a higher-cost path).
- Positive costs only: UCS/A* assume c(s,a,s′) > 0. No negative cycles here.
- Bidirectional search: requires single concrete goal and reverse operators. We added PriorityQueue.peek() so Bi-A*/Bi-UCS can choose the next frontier by min-f.
- Heuristic units: match units of c (e.g., km vs minutes). Mixing units breaks behavior.
- Beam search: incomplete by design—great for fast, “good enough” routes.
- Memory-bounded A*: expect occasional thrashing when the optimal subtree doesn’t fit.

# How to extend

- New problems: create a class in search_lab/problems/ implementing the Problem API.
- New heuristics: drop into search_lab/algorithms/heuristics.py (keep functions pure).
- More metrics: add counters (reopens, max frontier size) to SearchResult.
- CI regression: pin a few problem instances + assert known costs (# nodes may vary by tie-breaking).

Quick reading list (what to Google if stuck)
1. AIMA 4e, Chapter 3: Solving Problems by Searching.
2. “admissible vs consistent heuristic” examples.
3. IDA*, RBFS, SMA*: memory-bounded optimal search.

# TL;DR takeaways

- A good heuristic shrinks search dramatically; A* with an admissible, consistent h is the default workhorse.
- Memory is the real bottleneck for A*; switch to IDA* / RBFS / SMA* when graphs get big.
- Bidirectional shines when you can run a proper reverse search to a single known goal.
- Keep algorithms modular and measured—benchmarks + plots make trade-offs obvious.