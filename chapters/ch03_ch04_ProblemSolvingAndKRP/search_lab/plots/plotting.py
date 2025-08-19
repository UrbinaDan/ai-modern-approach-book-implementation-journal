# search_lab/plots/plotting.py
# This code provides a utility function for creating bar plots to compare search algorithm results.
# It visualizes the performance of different search algorithms based on nodes expanded, path cost, time taken, and peak memory usage.
# The function takes a list of results and generates a 2x2 grid of bar plots for easy comparison.
from __future__ import annotations
import matplotlib.pyplot as plt

def bar_compare(results, title="Search Comparison"):
    names = [r.name for r in results]
    nodes = [r.nodes_expanded for r in results]
    costs = [r.path_cost for r in results]
    times = [r.time_sec for r in results]
    mems  = [r.peak_mem_kb or 0 for r in results]

    fig, axs = plt.subplots(2, 2, figsize=(11,8))
    axs = axs.ravel()
    axs[0].bar(names, nodes); axs[0].set_title("Nodes Expanded"); axs[0].tick_params(axis='x', rotation=45)
    axs[1].bar(names, costs); axs[1].set_title("Path Cost"); axs[1].tick_params(axis='x', rotation=45)
    axs[2].bar(names, times); axs[2].set_title("Time (s)"); axs[2].tick_params(axis='x', rotation=45)
    axs[3].bar(names, mems); axs[3].set_title("Peak Memory (KB)"); axs[3].tick_params(axis='x', rotation=45)
    fig.suptitle(title)
    fig.tight_layout(rect=[0,0,1,0.95])
    return fig
