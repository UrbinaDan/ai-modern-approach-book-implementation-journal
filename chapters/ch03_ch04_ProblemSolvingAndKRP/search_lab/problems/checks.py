def sanity_check_problem(problem, max_states: int = 10_000):
    """Walks states breadth-first and checks step_cost never returns None."""
    from collections import deque
    seen = set()
    q = deque([problem.initial_state()])
    steps = 0
    while q and steps < max_states:
        s = q.popleft()
        if s in seen:
            continue
        seen.add(s)
        for a in problem.actions(s):
            s2 = problem.result(s, a)
            cost = problem.step_cost(s, a, s2)
            if cost is None:
                raise AssertionError(f"step_cost is None for (s={s}, a={a}, s'={s2})")
            q.append(s2)
        steps += 1
    return f"OK: visited {len(seen)} states; no None costs."
