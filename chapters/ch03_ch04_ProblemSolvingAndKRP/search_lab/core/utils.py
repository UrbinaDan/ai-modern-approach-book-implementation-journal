# search_lab/core/utils.py
#This code provides a utility function for reconstructing the solution path from a goal node in a search tree.
from __future__ import annotations
from typing import List, Tuple
from .node import Node

def reconstruct_path(node: Node) -> Tuple[List, float]:
    actions = []
    cost = float(node.path_cost)
    cur = node
    while cur.parent is not None:
        actions.append(cur.action)
        cur = cur.parent
    actions.reverse()
    return actions, cost
