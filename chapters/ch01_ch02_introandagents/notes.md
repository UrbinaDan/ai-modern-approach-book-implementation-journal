# Chapter 2 — Intelligent Agents: Projects (Notes & Conclusions)

## Why these projects?
To move from **abstract agent functions** to **working agent programs**, mirroring AIMA’s progression:
1) **Reflex** (rules) → 2) **Model-based** (memory + models) →  
3) **Goal-based** (search/plan) → 4) **Utility-based** (trade-offs) →  
5) **Learning wrapper** (exploration helps adaptation).

**What you just practiced (the WHY in one line each)**

 - Env & Percepts: separates agent function (math) from agent program (code).
 - Simple Reflex: condition–action rules; good only when the current percept is enough.
 - Model-Based: internal state + transition/sensor models handle partial observability/noise.
 - Goal-Based: explicit goals enable planning (BFS/A*) and easy goal swaps.
 - Utility-Based: rational choices under trade-offs/uncertainty via expected utility.
 - Exploration: ε-greedy shows why trying suboptimal actions can pay off under drift.

---

## 1) Simple Reflex Agent
**Design:** Condition–action rules. If `dirty` → `Suck`; else:  
- Deterministic: `Right`  
- Randomized: random move

**Why:** Demonstrates AIMA’s SIMPLE-REFLEX idea; randomized behavior can avoid loops.

**Takeaways:**  
- Deterministic tends to sweep predictably but may waste moves in clean runs.  
- Randomized explores more but also wastes steps.  
- In fully observable, noise-free worlds, simple reflex can be OK; scalability is limited.

**Result:** See summary table for average rewards (100 eps, 3×3, 40 steps).

---

## 2) Model-Based Reflex Agent (Partial Observability)
**Design:** Hide location; 5% sensor noise on dirt. Agent keeps an **internal location belief**, updated via a simple **transition model** (last action) and follows a **serpentine sweep**.

**Why:** Matches AIMA’s Fig 2.11–2.12. Internal state compensates for missing/perhaps noisy percepts.

**Takeaways:**  
- **Model-based** outperforms **randomized simple reflex** under partial observability/noise.  
- Even a crude model + memory yields robust behavior.

**Result:** See summary table (avg rewards, 100 eps).

---

## 3) Goal-Based Agent (BFS / A*)
**Design:** 3×3 grid, wall at (1,1), start (0,0), goal (2,2).  
- **BFS** finds shortest path by breadth.  
- **A\*** uses **Manhattan heuristic** to focus search.

**Why:** Explicit goals enable lookahead and planning (AIMA Ch.3). Heuristics make search efficient.

**Takeaways:**  
- Both find shortest paths; A* expands fewer nodes (not shown, but implied by heuristic focus).  
- Easy to retarget goals (flexibility vs reflex rules).

**Result:**  
- BFS: `Down, Down, Right, Right`  
- A*: `Right, Right, Down, Down`

---

## 4) Utility-Based Agent (Trade-offs & Uncertainty)
**Design:** Two routes: A (fast, risky), B (slow, safe).  
Utility: `U = −(expected_travel_time) − risk_weight × accident_risk`.

**Why:** Goals are binary; **utility** handles **trade-offs** and **uncertainty**.

**Takeaways:**  
- With `risk_weight = 5`, Route A beats B.  
- Increasing `risk_weight` flips the choice → policy depends on preferences.

**Result:** Route A utility/time: see summary table.

---

## 5) Learning Wrapper (ε-greedy + Concept Drift)
**Design:** Dirt probability drifts every 50 episodes: `[0.2, 0.5, 0.8, 0.3]`.  
Compare base **model-based reflex** vs **ε-greedy (ε=0.1)**.

**Why:** Even without full ML, exploration can uncover better actions under change.

**Takeaways:**  
- Rewards fluctuate with drift; ε-greedy sometimes adapts faster but can also add noise.  
- The CSV log can seed future ML (bandits/RL): `state → action → reward`.

**Artifacts:**  
- Learning curve plot (rewards vs episode).  
- CSV: `epsilon_concept_drift_rewards.csv`.

---

## Overall Conclusions
- **Memory + models** dominate under partial observability and noise.  
- **Goals** (search) add flexibility; **utility** adds rational trade-offs.  
- **Exploration** is essential when environments drift—even before using full RL algorithms.  
- These patterns exactly echo AIMA’s agent hierarchy and motivate later chapters (search, probability, planning, RL).

