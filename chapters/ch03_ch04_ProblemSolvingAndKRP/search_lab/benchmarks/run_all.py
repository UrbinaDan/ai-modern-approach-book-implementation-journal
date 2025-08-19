# search_lab/benchmarks/run_all.py
from __future__ import annotations

import json
import time
from pathlib import Path

def _fmt_time(x):
    try:
        return f"{float(x):.4f}"
    except Exception:
        return "n/a"

def _load_problem():
    # Prefer Romania; fall back to a tiny grid if you have one.
    try:
        from ..problems.romania import romania_problem
        return romania_problem()
    except Exception as e:
        raise RuntimeError(
            "Could not import problems.romania: "
            f"{e}\nIf you have a grid problem, swap this to load it instead."
        )

def _load_algos():
    algos = []
    # Add/Remove as your implementations become available.
    try:
        from ..algorithms.bfs import breadth_first_search
        algos.append(("BFS", breadth_first_search))
    except Exception as e:
        print(f"  Skipping BFS import: {repr(e)}")

    try:
        from ..algorithms.ucs import uniform_cost_search
        algos.append(("UCS", uniform_cost_search))
    except Exception as e:
        print(f"  Skipping UCS import: {repr(e)}")

    try:
        from ..algorithms.greedy import greedy_best_first_search
        algos.append(("Greedy", greedy_best_first_search))
    except Exception as e:
        print(f"  Skipping Greedy import: {repr(e)}")

    try:
        from ..algorithms.astar import a_star_search
        algos.append(("A*", a_star_search))
    except Exception as e:
        print(f"  Skipping A* import: {repr(e)}")

    return algos

def main():
    problem = _load_problem()
    algos = _load_algos()

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
