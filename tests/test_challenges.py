"""Cross-check challenge reference solutions against brute force on small inputs."""

import itertools

import numpy as np
import pytest

from generator.core import build, templates


def brute(name, ns):
    if name == "longest_budget_window":
        x = next(ns[k] for k in ("durations", "tokens", "frames_mb") if k in ns).tolist()
        return max([j - i for i in range(len(x)) for j in range(i + 1, len(x) + 1) if sum(x[i:j]) <= ns["S"]], default=0)
    if name == "count_subarrays_sum_k":
        x = next(ns[k] for k in ("reward", "delta", "balance") if k in ns).tolist()
        return sum(sum(x[i:j]) == ns["K"] for i in range(len(x)) for j in range(i + 1, len(x) + 1))
    if name == "split_min_max_load":
        x = next(ns[k] for k in ("shards", "chapters") if k in ns).tolist()
        best = None
        for cuts in itertools.combinations(range(1, len(x)), ns["D"] - 1):
            b = [0, *cuts, len(x)]
            load = max(sum(x[b[i]:b[i + 1]]) for i in range(len(b) - 1))
            best = load if best is None else min(best, load)
        return best
    if name == "wait_for_higher":
        x = next(ns[k] for k in ("temp_c", "acc", "price") if k in ns).tolist()
        return np.array([next((j - i for j in range(i + 1, len(x)) if x[j] > x[i]), 0) for i in range(len(x))])
    if name == "merge_segments":
        cover = set()
        for a, b in ns["segs"].tolist():
            cover.update(range(a, b))
        pts = sorted(cover)
        runs = 0
        for i, p in enumerate(pts):
            if i == 0 or p != pts[i - 1] + 1:
                runs += 1
        return (runs, len(pts))
    if name == "count_pairs_under":
        x = next(ns[k] for k in ("mem_gb", "lengths") if k in ns).tolist()
        return sum(x[i] + x[j] <= ns["T"] for i in range(len(x)) for j in range(i + 1, len(x)))
    if name == "fewest_drops_sorted":
        x = next(ns[k] for k in ("ts", "version") if k in ns).tolist()
        best = [1] * len(x)
        for j in range(len(x)):
            for i in range(j):
                if x[i] <= x[j]:
                    best[j] = max(best[j], best[i] + 1)
        return len(x) - max(best)
    if name == "best_window_length_range":
        x = next(ns[k] for k in ("gain", "reward") if k in ns).tolist()
        L, R = ns["L"], ns["R"]
        return max(sum(x[i:i + w]) for w in range(L, R + 1) for i in range(len(x) - w + 1))
    raise KeyError(name)


@pytest.mark.parametrize("tpl", templates("numpy", "challenge"), ids=lambda t: t.name)
def test_matches_brute_force(tpl):
    for seed in range(60):
        p = build(tpl, seed)
        ns = {}
        exec(p.setup, ns)
        expected = brute(tpl.name, dict(ns))
        exec(p.solution, ns)
        got = ns["result"]
        if isinstance(expected, np.ndarray):
            assert np.array_equal(got, expected), (seed, p.setup)
        else:
            assert got == expected, (seed, p.setup, got, expected)
