# search_lab/benchmarks/run_all.py
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Callable, List, Tuple, Any

# ---- Tunables (overridable via environment variables) -----------------------
DLS_LIMIT = int(os.getenv("DLS_LIMIT", "12"))          # depth-limited search
BEAM_K    = int(os.getenv("BEAM_K", "10"))             # beam width
WA_W      = float(os.getenv("WASTAR_W", "1.5"))        # weighted A* weight
SMA_MAX   = int(os.getenv("SMA_MAX_NODES", "50000"))   # SMA* node budget (if supported)

# ---- Helpers ----------------------------------------------------------------
def _fmt_time(x):
    try:
        return f"{float(x):.4f}"
    except Exception:
        return "n/a"

def _load_problem():
    """
    Prefer Romania map; if you have another sample (e.g., GridProblem),
    swap the import here.
    """
    try:
        from ..problems.romania import romania_problem
        return romania_problem()
    except Exception as e:
        raise RuntimeError(
            "Could not import problems.romania: "
            f"{e}\nIf you have a grid problem, swap this to load it instead."
        )

def _maybe_append(algos: List[Tuple[str, Callable[[Any], Any]]], name: str, fn: Callable):
    if callable(fn):
        algos.append((name, fn))

def _load_algos():
    """
    Try to import all algorithms; skip those that aren't implemented yet.
    Each algorithm is expected to be a callable taking (problem) and returning
    a result object with attributes:
        .algo, .success, .cost, .nodes_expanded, .time_s, .peak_kb, .error
    """
    algos: List[Tuple[str, Callable[[Any], Any]]] = []

    # ---------------- Uninformed ----------------
    # BFS
    try:
        from ..algorithms.bfs import breadth_first_search
        _maybe_append(algos, "BFS", breadth_first_search)
    except Exception as e:
        print(f"  Skipping BFS import: {repr(e)}")

    # UCS / Dijkstra
    try:
        from ..algorithms.ucs import uniform_cost_search
        _maybe_append(algos, "UCS", uniform_cost_search)
    except Exception as e:
        print(f"  Skipping UCS import: {repr(e)}")

    # DFS
    try:
        from ..algorithms.dfs import depth_first_search
        _maybe_append(algos, "DFS", depth_first_search)
    except Exception as e:
        print(f"  Skipping DFS import: {repr(e)}")

    # Depth-Limited Search
    try:
        from ..algorithms.depth_limited import depth_limited_search
        _maybe_append(algos, f"DLS(ℓ={DLS_LIMIT})", lambda p: depth_limited_search(p, limit=DLS_LIMIT))
    except Exception as e:
        print(f"  Skipping DLS import: {repr(e)}")

    # Iterative Deepening Search (IDS)
    try:
        from ..algorithms.ids import iterative_deepening_search
        _maybe_append(algos, "IDS", iterative_deepening_search)
    except Exception as e:
        print(f"  Skipping IDS import: {repr(e)}")

    # Bidirectional Search (uninformed / uniform-cost)
    # Try a few common names/signatures:
    try:
        from ..algorithms.bidirectional import bidirectional_uniform_cost_search
        _maybe_append(algos, "Bi-UCS", bidirectional_uniform_cost_search)
    except Exception:
        try:
            from ..algorithms.bidirectional import bibf_uniform_cost_search
            _maybe_append(algos, "Bi-UCS", bibf_uniform_cost_search)
        except Exception as e:
            print(f"  Skipping Bidirectional UCS import: {repr(e)}")

    # ---------------- Informed ----------------
    # Greedy Best-First
    try:
        from ..algorithms.greedy import greedy_best_first_search
        _maybe_append(algos, "Greedy", greedy_best_first_search)
    except Exception as e:
        print(f"  Skipping Greedy import: {repr(e)}")

    # A*
    try:
        from ..algorithms.astar import a_star_search
        _maybe_append(algos, "A*", a_star_search)
    except Exception as e:
        print(f"  Skipping A* import: {repr(e)}")

    # Bidirectional A*
    try:
        from ..algorithms.bidirectional import bidirectional_a_star
        _maybe_append(algos, "Bi-A*", bidirectional_a_star)
    except Exception as e:
        print(f"  Skipping Bidirectional A* import: {repr(e)}")

    # IDA*
    try:
        from ..algorithms.ida_star import ida_star_search
        _maybe_append(algos, "IDA*", ida_star_search)
    except Exception as e:
        print(f"  Skipping IDA* import: {repr(e)}")

    # RBFS
    try:
        from ..algorithms.rbfs import recursive_best_first_search
        _maybe_append(algos, "RBFS", recursive_best_first_search)
    except Exception as e:
        print(f"  Skipping RBFS import: {repr(e)}")

    # SMA*
    try:
        from ..algorithms.sma_star import sma_star_search
        # If your SMA* supports max_nodes, pass it; otherwise call without kwargs.
        try:
            _maybe_append(algos, f"SMA*(N={SMA_MAX})", lambda p: sma_star_search(p, max_nodes=SMA_MAX))
        except TypeError:
            _maybe_append(algos, "SMA*", sma_star_search)
    except Exception as e:
        print(f"  Skipping SMA* import: {repr(e)}")

    # Beam Search
    try:
        from ..algorithms.beam import beam_search
        # Prefer k argument if available
        try:
            _maybe_append(algos, f"Beam(k={BEAM_K})", lambda p: beam_search(p, k=BEAM_K))
        except TypeError:
            _maybe_append(algos, "Beam", beam_search)
    except Exception as e:
        print(f"  Skipping Beam import: {repr(e)}")

    # Weighted A*
    try:
        from ..algorithms.weighted_astar import weighted_a_star_search
        # Prefer weight argument if available
        try:
            _maybe_append(algos, f"WeightedA*(w={WA_W})", lambda p: weighted_a_star_search(p, w=WA_W))
        except TypeError:
            _maybe_append(algos, "WeightedA*", weighted_a_star_search)
    except Exception as e:
        print(f"  Skipping Weighted A* import: {repr(e)}")

    # in search_lab/benchmarks/run_all.py -> _load_algos()
    try:
        from ..algorithms.bidirectional import bibf_uniform_cost_search
        algos.append(("Bidirectional UCS", bibf_uniform_cost_search))
    except Exception as e:
        print(f"  Skipping Bidirectional UCS import: {repr(e)}")

    try:
        from ..algorithms.bidirectional import bidirectional_a_star
        algos.append(("Bidirectional A*", bidirectional_a_star))
    except Exception as e:
        print(f"  Skipping Bidirectional A* import: {repr(e)}")

    try:
        from ..algorithms.ida_star import ida_star_search
        algos.append(("IDA*", ida_star_search))
    except Exception as e:
        print(f"  Skipping IDA* import: {repr(e)}")

    try:
        from ..algorithms.rbfs import recursive_best_first_search
        algos.append(("RBFS", recursive_best_first_search))
    except Exception as e:
        print(f"  Skipping RBFS import: {repr(e)}")

    try:
        from ..algorithms.sma_star import sma_star_search
        algos.append(("SMA*", sma_star_search))
    except Exception as e:
        print(f"  Skipping SMA* import: {repr(e)}")

    try:
        from ..algorithms.weighted_astar import weighted_a_star_search
        algos.append(("Weighted A*", weighted_a_star_search))
    except Exception as e:
        print(f"  Skipping Weighted A* import: {repr(e)}")


    return algos

def main():
    problem = _load_problem()
    algos = _load_algos()

    if not algos:
        raise SystemExit("No algorithms loaded. Implement algorithms or fix imports in _load_algos().")

    rows = []
    for name, fn in algos:
        print(f"→ Running {name} ...")
        try:
            r = fn(problem)
            algo_name = getattr(r, "algo", name)
            success = getattr(r, "success", False)
            cost = getattr(r, "cost", None)
            expanded = getattr(r, "nodes_expanded", None)
            time_s = getattr(r, "time_s", None)

            print(
                f"  {algo_name}: "
                f"{'OK' if success else 'FAIL'} "
                f"cost={cost} "
                f"expanded={expanded}, "
                f"time={_fmt_time(time_s)}s"
            )
            rows.append({
                "algo": algo_name,
                "success": success,
                "cost": cost,
                "nodes_expanded": expanded,
                "time_s": time_s,                 # canonical field
                "peak_kb": getattr(r, "peak_kb", None),
                "error": getattr(r, "error", None),
            })
        except Exception as e:
            print(f"  {name}: ERROR {repr(e)}")
            rows.append({
                "algo": name,
                "success": False,
                "error": repr(e),
                "nodes_expanded": None,
                "cost": None,
                "time_s": None,
                "peak_kb": None,
            })

    out = {"results": rows, "ts": time.time()}
    print(json.dumps(out, indent=2))

    # Save JSON next to this script
    out_path = Path(__file__).with_name("results.json")
    try:
        out_path.write_text(json.dumps(out, indent=2))
    except Exception:
        pass

if __name__ == "__main__":
    main()
