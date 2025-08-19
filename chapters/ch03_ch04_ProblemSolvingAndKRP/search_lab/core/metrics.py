# search_lab/core/metrics.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import time, tracemalloc

@dataclass
class SearchResult:
    algo: str
    success: bool
    actions: List[str]
    cost: float
    nodes_expanded: int
    time_s: float
    peak_kb: int
    error: Optional[str] = None

class MeasuredRun:
    """
    Context manager for timing and (approximate) peak memory.
    Safe to query .elapsed and .peak_kb *inside* the with-block.
    """
    def __init__(self) -> None:
        self.t0: Optional[float] = None
        self.t1: Optional[float] = None
        self._peak_kb: int = 0
        self._tracing: bool = False

    def __enter__(self) -> "MeasuredRun":
        self._tracing = True
        tracemalloc.start()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.t1 = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self._tracing = False
        self._peak_kb = max(self._peak_kb, peak // 1024)
        return False  # don't suppress exceptions

    @property
    def elapsed(self) -> float:
        """Seconds elapsed. Works before and after __exit__."""
        if self.t0 is None:
            return 0.0
        if self.t1 is None:
            return time.perf_counter() - self.t0
        return self.t1 - self.t0

    @property
    def peak_kb(self) -> int:
        """Approx peak KB. Works before and after __exit__."""
        if self._tracing:
            current, peak = tracemalloc.get_traced_memory()
            return max(self._peak_kb, peak // 1024)
        return self._peak_kb
