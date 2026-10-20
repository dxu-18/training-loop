"""Cross-check every challenge reference solution against a brute force on small inputs."""

import itertools
import math

import numpy as np
import pytest

from generator.core import REGISTRY, build, templates

CHALLENGES = [t for s in REGISTRY for t in templates(s) if t.tier == "challenge"]
BRUTE = {}


def brute(fn):
    BRUTE[fn.__name__] = fn
    return fn


def first(ns, *names):
    return next(ns[k] for k in names if k in ns)


# ---------------------------------------------------------------- 1.01

@brute
def min_linewidth(ns):
    w, K = ns["widths"].tolist(), ns["K"]
    L = max(w)
    while True:
        lines, cur = 1, -1
        for x in w:
            if cur + 1 + x <= L:
                cur += 1 + x
            else:
                lines, cur = lines + 1, x
        if lines <= K:
            return L
        L += 1


@brute
def max_edgeitems(ns):
    dims = ns["dims"].tolist()
    return max(e for e in range(ns["E"] + 1) if math.prod(min(d, 2 * e) for d in dims) <= ns["B"])


# ---------------------------------------------------------------- 1.02

@brute
def count_free_squares(ns):
    g = ns["grid"]
    n, m = g.shape
    return sum(bool(g[i:i + k, j:j + k].all()) for k in range(1, min(n, m) + 1)
               for i in range(n - k + 1) for j in range(m - k + 1))


@brute
def largest_plus(ns):
    g = ns["grid"]
    n, m = g.shape
    best = 0
    for i in range(n):
        for j in range(m):
            k = 0
            while (i - k >= 0 and i + k < n and j - k >= 0 and j + k < m
                   and g[i - k, j] and g[i + k, j] and g[i, j - k] and g[i, j + k]):
                k += 1
            best = max(best, k)
    return best


# ---------------------------------------------------------------- 1.03

@brute
def inverse_cdf_counts(ns):
    w, u = ns["weights"].tolist(), ns["u"].tolist()
    counts = [0] * len(w)
    for v in u:
        acc = 0
        for i, x in enumerate(w):
            acc += x
            if acc > v * sum(w):
                counts[i] += 1
                break
    return np.array(counts)


@brute
def shuffle_period(ns):
    p = ns["p"]
    order, k = np.arange(len(p))[p], 1
    while not np.array_equal(order, np.arange(len(p))):
        order, k = order[p], k + 1
    return k


# ---------------------------------------------------------------- 1.04

@brute
def wrapped_sum_windows(ns):
    x, t, L = ns["x"].tolist(), ns["t"], ns["L"]
    return sum(sum(x[i:j]) % 256 == t for i in range(len(x)) for j in range(i + L, len(x) + 1))


@brute
def pack_bits_dp(ns):
    x, H = ns["x"].tolist(), ns["H"]
    best = None
    for mask in range(2 ** (len(x) - 1)):  # every way to cut between neighbours
        cuts = [0] + [i + 1 for i in range(len(x) - 1) if mask >> i & 1] + [len(x)]
        cost = sum(H + (b - a) * next(w for w in (8, 16, 32, 64) if max(x[a:b]) < 2 ** w)
                   for a, b in zip(cuts, cuts[1:]))
        best = cost if best is None else min(best, cost)
    return best


# ---------------------------------------------------------------- 1.05

@brute
def count_above_in_ranges(ns):
    x = ns["x"].tolist()
    return np.array([sum(v > t for v in x[a:b + 1]) for a, b, t in zip(ns["L"], ns["R"], ns["T"])])


@brute
def min_jumps(ns):
    from collections import deque
    a = ns["jump"].tolist()
    dist = [0] + [None] * (len(a) - 1)  # BFS over frames
    dq = deque([0])
    while dq:
        i = dq.popleft()
        for j in range(i + 1, min(len(a), i + a[i] + 1)):
            if dist[j] is None:
                dist[j] = dist[i] + 1
                dq.append(j)
    return -1 if dist[-1] is None else dist[-1]


# ---------------------------------------------------------------- 1.06 - 1.09

@brute
def stable_window(ns):
    x = first(ns, "temp_c", "latency_ms").tolist()
    return max(j - i for i in range(len(x)) for j in range(i + 1, len(x) + 1)
               if max(x[i:j]) - min(x[i:j]) <= ns["D"])


@brute
def sorted_row_shapes(ns):
    x = ns["x"]
    n = len(x)
    return sum(all((row[:-1] <= row[1:]).all() for row in x.reshape(n // c, c)) for c in range(1, n + 1) if n % c == 0)


@brute
def transpose_cycles(ns):
    r, c = ns["r"], ns["c"]
    src = np.arange(r * c).reshape(r, c).T.ravel()  # element now at position p came from src[p]
    perm = np.empty(r * c, dtype=int)
    perm[src] = np.arange(r * c)
    seen, cycles = set(), 0
    for k in range(r * c):
        if k not in seen:
            cycles += 1
            while k not in seen:
                seen.add(k)
                k = int(perm[k])
    return cycles


@brute
def roll_to_match(ns):
    a, b = ns["a"], ns["b"]
    return next((s for s in range(len(a)) if np.array_equal(np.roll(a, s), b)), -1)


@brute
def count_within_radius(ns):
    x, r = ns["x"].tolist(), ns["r"]
    return np.array([sum(abs(v - q) <= r for v in x) for q in ns["queries"].tolist()])


# ---------------------------------------------------------------- 1.10 - 1.12

@brute
def gcd_subarrays(ns):
    x, G = ns["x"].tolist(), ns["G"]
    count = 0
    for i in range(len(x)):
        g = 0
        for j in range(i, len(x)):
            g = math.gcd(g, x[j])
            count += g == G
    return count


@brute
def halving_ops(ns):
    x, S = ns["x"].tolist(), ns["S"]
    levels = []  # every value an element can reach, with the halvings it costs
    for v in x:
        seq = [v]
        while seq[-1]:
            seq.append(seq[-1] // 2)
        levels.append(list(enumerate(seq)))
    return min(sum(k for k, _ in combo) for combo in itertools.product(*levels)
               if sum(v for _, v in combo) <= S)


@brute
def count_submatrices_sum_k(ns):
    g, K = ns["grid"], ns["K"]
    n, m = g.shape
    return sum(int(g[a:b, c:d].sum() == K) for a in range(n) for b in range(a + 1, n + 1)
               for c in range(m) for d in range(c + 1, m + 1))


@brute
def longest_balanced(ns):
    f = [1 if v else -1 for v in first(ns, "is_speech", "is_positive", "is_pass").tolist()]
    return max([j - i for i in range(len(f)) for j in range(i + 1, len(f) + 1) if sum(f[i:j]) == 0], default=0)


# ---------------------------------------------------------------- 1.14 - 1.18

@brute
def walks_mod(ns):
    A, k, s, t = ns["A"].tolist(), ns["k"], ns["s"], ns["t"]
    vec = [int(i == s) for i in range(len(A))]
    for _ in range(k):
        vec = [sum(vec[i] * A[i][j] for i in range(len(A))) for j in range(len(A))]
    return vec[t] % (10**9 + 7)


@brute
def gf2_rank(ns):
    rows = [int("".join(map(str, r[::-1])), 2) for r in ns["bits"].tolist()]
    span = {0}
    for r in rows:
        span |= {v ^ r for v in span}
    return len(span).bit_length() - 1


@brute
def count_weight_combos(ns):
    C, W = ns["counts"].tolist(), ns["W"]
    total = 0
    for choice in itertools.product(*[range(len(row)) for row in C]):
        if sum(choice) == W:
            total += math.prod(row[w] for row, w in zip(C, choice))
    return total % (10**9 + 7)


@brute
def max_xor_pair(ns):
    x = ns["x"].tolist()
    return max(a ^ b for a in x for b in x)


@brute
def max_concurrent_sessions(ns):
    st, en = ns["start"], ns["end"]
    return max(int(((st <= t) & (t < en)).sum()) for t in st)


@brute
def inversions_on_disk(ns):
    a = np.load("ranks.npy").tolist()
    return sum(a[i] > a[j] for i in range(len(a)) for j in range(i + 1, len(a)))


@brute
def infer_row_length(ns):
    raw = np.fromfile("log.bin", dtype=np.int64).tolist()
    n = len(raw)
    for c in range(2, n + 1):
        col = raw[::c]
        if n % c == 0 and all(a < b for a, b in zip(col, col[1:])):
            return c
    return -1


@brute
def best_threshold(ns):
    s, y = ns["scores"].tolist(), ns["y"].tolist()
    return max(sum((v >= t) == bool(lab) for v, lab in zip(s, y)) for t in s + [float("inf")])


@brute
def dynamic_padding(ns):
    L, K = ns["lengths"].tolist(), ns["K"]
    best = None
    for mask in range(2 ** (len(L) - 1)):
        cuts = [0] + [i + 1 for i in range(len(L) - 1) if mask >> i & 1] + [len(L)]
        groups = [L[a:b] for a, b in zip(cuts, cuts[1:])]
        if all(len(g) <= K for g in groups):
            cost = sum(len(g) * max(g) for g in groups)
            best = cost if best is None else min(best, cost)
    return best


# ---------------------------------------------------------------- shared by later modules

@brute
def longest_budget_window(ns):
    x = first(ns, "durations", "tokens", "frames_mb").tolist()
    return max([j - i for i in range(len(x)) for j in range(i + 1, len(x) + 1) if sum(x[i:j]) <= ns["S"]], default=0)


@brute
def count_subarrays_sum_k(ns):
    x = first(ns, "reward", "delta", "balance").tolist()
    return sum(sum(x[i:j]) == ns["K"] for i in range(len(x)) for j in range(i + 1, len(x) + 1))


@brute
def split_min_max_load(ns):
    x = first(ns, "shards", "chapters").tolist()
    best = None
    for cuts in itertools.combinations(range(1, len(x)), ns["D"] - 1):
        b = [0, *cuts, len(x)]
        load = max(sum(x[b[i]:b[i + 1]]) for i in range(len(b) - 1))
        best = load if best is None else min(best, load)
    return best


@brute
def wait_for_higher(ns):
    x = first(ns, "temp_c", "acc", "price").tolist()
    return np.array([next((j - i for j in range(i + 1, len(x)) if x[j] > x[i]), 0) for i in range(len(x))])


@brute
def merge_segments(ns):
    cover = set()
    for a, b in ns["segs"].tolist():
        cover.update(range(a, b))
    pts = sorted(cover)
    runs = sum(1 for i, p in enumerate(pts) if i == 0 or p != pts[i - 1] + 1)
    return (runs, len(pts))


@brute
def count_pairs_under(ns):
    x = first(ns, "mem_gb", "lengths").tolist()
    return sum(x[i] + x[j] <= ns["T"] for i in range(len(x)) for j in range(i + 1, len(x)))


@brute
def fewest_drops_sorted(ns):
    x = first(ns, "ts", "version").tolist()
    best = [1] * len(x)
    for j in range(len(x)):
        for i in range(j):
            if x[i] <= x[j]:
                best[j] = max(best[j], best[i] + 1)
    return len(x) - max(best)


@brute
def best_window_length_range(ns):
    x = first(ns, "gain", "reward").tolist()
    L, R = ns["L"], ns["R"]
    return max(sum(x[i:i + w]) for w in range(L, R + 1) for i in range(len(x) - w + 1))


# ---------------------------------------------------------------- the test

def test_every_challenge_has_a_brute_force():
    missing = sorted({t.name for t in CHALLENGES} - set(BRUTE))
    assert not missing, missing


def _same(got, expected):
    if isinstance(expected, np.ndarray):
        return np.array_equal(np.asarray(got), expected)
    if isinstance(expected, float):
        return math.isclose(got, expected, rel_tol=1e-9, abs_tol=1e-9)
    return got == expected


@pytest.mark.parametrize("tpl", CHALLENGES, ids=lambda t: f"{t.module}-{t.name}")
def test_matches_brute_force(tpl, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # some setups write files
    if tpl.name not in BRUTE:
        pytest.skip("no brute force yet")
    for seed in range(60):
        p = build(tpl, seed)
        ns = {}
        exec(p.setup, ns)
        expected = BRUTE[tpl.name](dict(ns))
        exec(p.solution, ns)
        assert _same(ns["result"], expected), (seed, p.setup, ns["result"], expected)
