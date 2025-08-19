# search_lab/benchmarks/plot_results.py
from __future__ import annotations
import json
import math
from pathlib import Path
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
RESULTS_JSON = HERE / "results.json"
OUT_DIR = HERE

def _load_rows():
    if not RESULTS_JSON.exists():
        raise SystemExit(f"Missing {RESULTS_JSON}. Run: python -m search_lab.benchmarks.run_all")
    data = json.loads(RESULTS_JSON.read_text())
    rows = data.get("results", [])
    # Keep only successful runs
    rows = [r for r in rows if r.get("success")]
    if not rows:
        raise SystemExit("No successful rows to plot.")
    return rows

def _sorted(rows, key):
    def key_fn(r):
        v = r.get(key)
        if v is None:
            return math.inf
        return v
    return sorted(rows, key=key_fn)

def _bar(ax, rows, metric, title, ylabel):
    algos = [r["algo"] for r in rows]
    vals = [r.get(metric) for r in rows]

    # Positions for bars and ticks
    x = list(range(len(algos)))
    ax.bar(x, vals)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(algos, rotation=20, ha="right")

    # Add value labels on top of bars (nice for small benchmarks)
    for xi, v in zip(x, vals):
        if v is None:
            label = "n/a"
            y = 0
        else:
            # compact formatting
            if isinstance(v, float) and v < 0.01:
                label = f"{v:.4f}"
            elif isinstance(v, float):
                label = f"{v:.3f}"
            else:
                label = f"{v}"
            y = v
        ax.text(xi, (y if y is not None else 0) + (0.01 * (max(vals) or 1)), label,
                ha="center", va="bottom", fontsize=8)

def _fmt_table(rows):
    # Markdown table
    lines = [
        "| Algorithm | Cost | Nodes Expanded | Time (s) | Peak KB |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        def fnum(x):
            if isinstance(x, (int, float)):
                return f"{x:.6f}" if isinstance(x, float) else f"{x}"
            return "n/a"
        lines.append(
            f"| {r['algo']} | {fnum(r.get('cost'))} | {fnum(r.get('nodes_expanded'))} | "
            f"{fnum(r.get('time_s'))} | {fnum(r.get('peak_kb'))} |"
        )
    return "\n".join(lines)

def main():
    rows = _load_rows()

    # Save a markdown summary
    md = _fmt_table(rows)
    md_path = OUT_DIR / "results.md"
    md_path.write_text(md)
    print(f"Wrote {md_path}")

    # Plot 3 bar charts (sorted for readability)
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    _bar(ax1, _sorted(rows, "nodes_expanded"), "nodes_expanded",
         "Nodes Expanded (lower is better)", "nodes")
    fig1.tight_layout()
    (OUT_DIR / "nodes_expanded.png").write_bytes(fig1_to_png_bytes(fig1))
    print(f"Wrote {OUT_DIR / 'nodes_expanded.png'}")

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    _bar(ax2, _sorted(rows, "time_s"), "time_s",
         "Wall Time (lower is better)", "seconds")
    fig2.tight_layout()
    (OUT_DIR / "time.png").write_bytes(fig2_to_png_bytes(fig2))
    print(f"Wrote {OUT_DIR / 'time.png'}")

    fig3, ax3 = plt.subplots(figsize=(6, 4))
    _bar(ax3, _sorted(rows, "cost"), "cost",
         "Path Cost (lower is better)", "cost")
    fig3.tight_layout()
    (OUT_DIR / "cost.png").write_bytes(fig3_to_png_bytes(fig3))
    print(f"Wrote {OUT_DIR / 'cost.png'}")

def fig1_to_png_bytes(fig):
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160)
    return buf.getvalue()

# alias the same helper for other figs
fig2_to_png_bytes = fig1_to_png_bytes
fig3_to_png_bytes = fig1_to_png_bytes

if __name__ == "__main__":
    main()
