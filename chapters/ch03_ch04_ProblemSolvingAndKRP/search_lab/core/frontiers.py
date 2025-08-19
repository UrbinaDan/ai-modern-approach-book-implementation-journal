# search_lab/core/frontiers.py
from __future__ import annotations
import heapq
from collections import deque

class FIFOQueue:
    def __init__(self):
        self.q = deque()
    def push(self, x): self.q.append(x)
    def pop(self): return self.q.popleft()
    def __len__(self): return len(self.q)
    def peek(self): return self.q[0]

class LIFOStack:
    def __init__(self):
        self.q = []
    def push(self, x): self.q.append(x)
    def pop(self): return self.q.pop()
    def __len__(self): return len(self.q)
    def peek(self): return self.q[-1]

class PriorityQueue:
    """Min-heap by key(x)."""
    def __init__(self, key):
        self.key = key
        self.h = []
        self.counter = 0  # tie-breaker for stability
    def push(self, x):
        self.counter += 1
        heapq.heappush(self.h, (self.key(x), self.counter, x))
    def pop(self):
        return heapq.heappop(self.h)[2]
    def __len__(self): return len(self.h)
    def peek(self):
        return self.h[0][2]
