# Description
A grid maze and run these Chapter-4 algorithms side-by-side with live animations and metrics:
- Hill-Climbing (steepest descent on heuristic)
- Simulated Annealing
- Local Beam Search
- Stochastic Beam Search
- Evolutionary / Genetic Search (fixed-length path genome)
- Online DFS Agent
- LRTA*
- AND–OR Planner (with a simple “slippery moves” nondeterminism)

# What you get (mapped to your table)
- Hill-Climbing: exactly “move to best neighbor; stop when no better.” Tiny memory; watch it stall on ridges/plateaus in cluttered mazes.
- Simulated Annealing: accepts worse moves early; cools via 𝑇←𝛼𝑇. Great to see it escape local minima.
- Local Beam: keeps top-k by 𝑔+ℎ; super fast but can collapse to a single thread.
- Stochastic Beam: softmax sampling to keep diversity (τ slider).
- Evolutionary/Genetic: evolves fixed-length move strings; naturally parallel; not guaranteed optimal but fun to watch best-so-far improve.
- Online DFS Agent: explores by acting—no global plan; good for discovering unknown mazes.
- LRTA*: learns local cost estimates 𝐻(𝑠) and acts greedily; eventually “flattens” local minima and escapes.
- AND–OR (slippery): constructs a small contingent plan when actions can “slip” and then executes it once; tweak p_slip.

Why no “Gradient/Newton” knob?
On a discrete grid these reduce to steepest-descent hill-climbing on the heuristic field. If you want, rename Hill-Climbing to “Discrete Gradient Descent” and it’s the same mechanics. True Newton needs a smooth objective and Hessian—out of scope for a blocky maze.