"""Cross-check every challenge reference solution against a brute force on small inputs."""

import itertools
import math

import numpy as np
import pytest

from generator.core import REGISTRY, build, templates

# Python challenges (bash ones are checked against their brute forces in test_bash_challenges.py)
CHALLENGES = [t for s in REGISTRY for t in templates(s) if t.tier == "challenge" and build(t, 0).kind == "python"]
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


# ---------------------------------------------------------------- pandas 2.01 - 2.04

@brute
def asof_lookup(ns):
    s, out = ns["s"], []
    for q in ns["queries"].tolist():
        earlier = [v for t, v in s.items() if t <= q]
        out.append(earlier[-1] if earlier else np.nan)
    return np.array(out, dtype=float)


@brute
def k_distinct_windows(ns):
    x, K = ns["labels"].tolist(), ns["K"]
    return sum(len(set(x[i:j])) <= K for i in range(len(x)) for j in range(i + 1, len(x) + 1))


@brute
def sparse_to_dense(ns):
    import pandas as pd
    cells = {}
    for r, c, v in ns["updates"].values.tolist():
        cells[(r, c)] = v
    rows, cols = sorted({r for r, _ in cells}), sorted({c for _, c in cells})
    return pd.DataFrame({c: [cells.get((r, c), np.nan) for r in rows] for c in cols}, index=rows, dtype=float)


@brute
def na_gap_budget(ns):
    x, S = ns["x"].tolist(), ns["S"]
    return max([j - i for i in range(len(x)) for j in range(i + 1, len(x) + 1)
                if not any(v != v for v in x[i:j]) and sum(x[i:j]) <= S], default=0)


@brute
def chunked_distinct(ns):
    import pandas as pd
    users = pd.read_csv("events.csv")["user"].tolist()
    k = ns["CHUNK"]
    return [len(set(users[:end])) for end in range(k, len(users) + k, k)][: -(-len(users) // k)]


@brute
def log_percentiles(ns):
    by = {}
    for line in open("requests.log").read().splitlines():
        parts = dict(p.split("=", 1) for p in line.split()[1:])
        by.setdefault(parts["endpoint"], []).append(int(parts["ms"]))
    return {ep: sorted(v)[math.ceil(0.95 * len(v)) - 1] for ep, v in by.items()}


@brute
def kth_pair_distance(ns):
    a, k = ns["df"]["age"].tolist(), ns["k"]
    return sorted(abs(a[i] - a[j]) for i in range(len(a)) for j in range(i + 1, len(a)))[k - 1]


@brute
def flip_equal_rows(ns):
    rows = ns["flags"].values.tolist()
    best = 0
    for mask in itertools.product([0, 1], repeat=len(rows[0])):
        best = max(best, sum(len({v ^ m for v, m in zip(r, mask)}) == 1 for r in rows))
    return best


# ---------------------------------------------------------------- pandas 2.05 - 2.08

@brute
def pareto_front(ns):
    pts = list(zip(ns["df"]["acc"].tolist(), ns["df"]["speed"].tolist()))
    return sum(not any(q[0] >= p[0] and q[1] >= p[1] and q != p for q in pts) for p in pts)


@brute
def range_count_queries(ns):
    df = ns["df"]
    return [int(((df["team"] == t) & df["score"].between(lo, hi)).sum())
            for t, lo, hi in ns["queries"].itertuples(index=False)]


@brute
def range_add_updates(ns):
    import pandas as pd
    x = ns["df"]["x"].tolist()
    for l_, r, d in ns["updates"].itertuples(index=False):
        for i in range(l_, r + 1):
            x[i] += d
    return pd.Series(x, name="x")


@brute
def min_swaps_sort(ns):
    v = ns["df"]["rank"].tolist()
    swaps = 0
    for i in range(len(v)):  # selection sort, counting real swaps
        j = min(range(i, len(v)), key=v.__getitem__)
        if j != i:
            v[i], v[j] = v[j], v[i]
            swaps += 1
    return swaps


@brute
def nesting_chain(ns):
    b = sorted(zip(ns["df"]["w"].tolist(), ns["df"]["h"].tolist()))
    best = [1] * len(b)
    for j in range(len(b)):
        for i in range(j):
            if b[i][0] < b[j][0] and b[i][1] < b[j][1]:
                best[j] = max(best[j], best[i] + 1)
    return max(best)


@brute
def make_unique_cost(ns):
    v = sorted(ns["df"]["slot"].tolist())
    lo, hi = v[0], v[-1] + len(v)
    best = None
    for targets in itertools.combinations(range(lo, hi), len(v)):
        if all(t >= x for t, x in zip(targets, v)):
            cost = sum(t - x for t, x in zip(targets, v))
            best = cost if best is None else min(best, cost)
    return best


@brute
def nearest_fill(ns):
    import pandas as pd
    v = ns["s"].tolist()
    out = []
    for i, x in enumerate(v):
        if x == x:
            out.append(x)
            continue
        for d in range(1, len(v)):
            if i - d >= 0 and v[i - d] == v[i - d]:
                out.append(v[i - d])
                break
            if i + d < len(v) and v[i + d] == v[i + d]:
                out.append(v[i + d])
                break
    return pd.Series(out)


@brute
def largest_complete_block(ns):
    ok = ns["df"].notna().to_numpy()
    r, c = ok.shape
    return max([(b - a) * (d - c0) for a in range(r) for b in range(a + 1, r + 1)
                for c0 in range(c) for d in range(c0 + 1, c + 1) if ok[a:b, c0:d].all()], default=0)


# ---------------------------------------------------------------- pandas 2.09 - 2.12

@brute
def alias_resolution(ns):
    import pandas as pd
    par = ns["parent"].to_dict()
    out = []
    for x in par:
        for _ in range(len(par) + 1):
            if par[x] == x:
                break
            x = par[x]
        out.append(x if par[x] == x else -1)
    return pd.Series(out, index=ns["parent"].index)


@brute
def tiered_cost(ns):
    caps, rates = ns["tiers"]["up_to"].tolist(), ns["tiers"]["rate"].tolist()
    total = 0
    for u in ns["usage"].tolist():
        for t in range(1, u + 1):  # price every token on its own
            i = next((k for k, cap in enumerate(caps) if t <= cap), None)
            total += rates[i] if i is not None else ns["last_rate"]
    return total


@brute
def shortest_unique_prefix(ns):
    import pandas as pd
    w = ns["words"].tolist()
    out = []
    for i, word in enumerate(w):
        others = w[:i] + w[i + 1:]
        out.append(next((word[:k] for k in range(1, len(word) + 1)
                         if not any(o.startswith(word[:k]) for o in others)), word))
    return pd.Series(out)


@brute
def overlapping_counts(ns):
    import pandas as pd
    p = ns["pattern"]
    return pd.Series([sum(t[i:i + len(p)] == p for i in range(len(t))) for t in ns["texts"].tolist()])


@brute
def sessionize(ns):
    import pandas as pd
    out = {}
    for user, ts in ns["events"].groupby("user")["ts"]:
        ts = sorted(ts)
        out[user] = 1 + sum(b - a > pd.Timedelta(minutes=30) for a, b in zip(ts, ts[1:]))
    return out


@brute
def longest_daily_streak(ns):
    import pandas as pd
    best = {}
    for user, days in ns["log"].groupby("user")["day"]:
        ds = set(days)
        best[user] = max(next(k for k in range(1, 400) if d + pd.Timedelta(days=k) not in ds) for d in ds)
    top = max(best.values())
    return (min(u for u, v in best.items() if v == top), top)


@brute
def optimal_binning(ns):
    v, k = sorted(ns["x"].tolist()), ns["k"]
    best = None
    for cuts in itertools.combinations(range(1, len(v)), k - 1):
        b = [0, *cuts, len(v)]
        cost = sum(sum((t - sum(v[i:j]) / (j - i)) ** 2 for t in v[i:j]) for i, j in zip(b, b[1:]))
        best = cost if best is None else min(best, cost)
    return round(best, 6)


@brute
def balanced_window(ns):
    x = ns["labels"].tolist()
    classes = set(x)
    if len(classes) < 2:
        return 0
    return max([j - i for i in range(len(x)) for j in range(i + 1, len(x) + 1)
                if len({x[i:j].count(c) for c in classes}) == 1], default=0)


# ---------------------------------------------------------------- pandas 2.13 - 2.16

@brute
def pairs_within_d(ns):
    rows = list(ns["df"][["team", "score"]].itertuples(index=False))
    return sum(a.team == b.team and abs(a.score - b.score) <= ns["D"]
               for i, a in enumerate(rows) for b in rows[i + 1:])


@brute
def entity_resolution(ns):
    df = ns["df"]
    n = len(df)
    e, p = df["email"].tolist(), df["phone"].tolist()
    seen, sizes = set(), []
    for s0 in range(n):
        if s0 in seen:
            continue
        stack, comp = [s0], 0
        seen.add(s0)
        while stack:
            i = stack.pop()
            comp += 1
            for j in range(n):
                if j not in seen and (e[i] == e[j] or p[i] == p[j]):
                    seen.add(j)
                    stack.append(j)
        sizes.append(comp)
    return (len(sizes), max(sizes))


@brute
def grid_max_path(ns):
    c = ns["cells"]
    grid = {(r, k): v for r, k, v in c[["row", "col", "value"]].itertuples(index=False)}
    R, C = c["row"].max() + 1, c["col"].max() + 1

    def walk(i, j):
        if (i, j) == (R - 1, C - 1):
            return grid[(i, j)]
        options = []
        if i + 1 < R:
            options.append(walk(i + 1, j))
        if j + 1 < C:
            options.append(walk(i, j + 1))
        return grid[(i, j)] + max(options)

    return walk(0, 0)


@brute
def cooccurring_tags(ns):
    docs = [set(t) for t in ns["docs"]["tags"]]
    vocab = sorted(set().union(*docs))
    counts = {(a, b): sum(a in d and b in d for d in docs) for a, b in itertools.combinations(vocab, 2)}
    best = max(counts.values())
    return (min(p for p, c in counts.items() if c == best), best)


@brute
def interval_lookup(ns):
    sh = list(ns["shifts"].itertuples(index=False))
    return [next((s.shift for s in sh if s.start <= t < s.end), None) for t in ns["events"]["t"].tolist()]


@brute
def join_size(ns):
    L, R = ns["L"], ns["R"]
    return (len(L.merge(R, on="key", how="inner")), len(L.merge(R, on="key", how="left")))


@brute
def count_windows_mean_ge(ns):
    x, t = ns["s"].tolist(), ns["t"]
    return sum(sum(x[i:j]) >= t * (j - i) for i in range(len(x)) for j in range(i + 1, len(x) + 1))


@brute
def longest_window_mean_ge(ns):
    x, t = ns["s"].tolist(), ns["t"]
    return max([j - i for i in range(len(x)) for j in range(i + 1, len(x) + 1) if sum(x[i:j]) >= t * (j - i)], default=0)


# ---------------------------------------------------------------- pandas 2.17 - 2.19

@brute
def grid_components(ns):
    cells = {(int(r), int(c)) for r, c in ns["active"].index}
    R, C = max(r for r, _ in cells) + 2, max(c for _, c in cells) + 2
    label, sizes = {}, []
    for r in range(R):  # dense flood fill over the bounding box
        for c in range(C):
            if (r, c) in cells and (r, c) not in label:
                label[(r, c)] = len(sizes)
                todo, size = [(r, c)], 0
                while todo:
                    i, j = todo.pop()
                    size += 1
                    for nb in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                        if nb in cells and nb not in label:
                            label[nb] = len(sizes)
                            todo.append(nb)
                sizes.append(size)
    return (len(sizes), max(sizes))


@brute
def subtree_sums(ns):
    import pandas as pd
    par, cost = ns["tree"]["parent"].tolist(), ns["tree"]["cost"].tolist()
    total = [0] * len(par)
    for v in range(len(par)):  # add every node's cost to itself and all its ancestors
        u = v
        while u != -1:
            total[u] += cost[v]
            u = par[u]
    return pd.Series(total, index=ns["tree"].index)


@brute
def nice_ticks(ns):
    out = []
    for lo, hi, m in ns["ranges"].itertuples(index=False):
        step = next(s for s in itertools.chain.from_iterable((1 * 10**e, 2 * 10**e, 5 * 10**e) for e in range(20))
                    if len(range(math.floor(lo / s) * s, math.ceil(hi / s) * s + 1, s)) <= m)
        out.append(step)
    return out


@brute
def max_visible_labels(ns):
    boxes = list(zip(ns["labels"]["left"], ns["labels"]["right"]))
    best = 0
    for mask in range(1 << len(boxes)):
        chosen = sorted(b for i, b in enumerate(boxes) if mask >> i & 1)
        if all(a[1] <= b[0] for a, b in zip(chosen, chosen[1:])):
            best = max(best, len(chosen))
    return best


@brute
def pairs_sum_divisible(ns):
    x, k = ns["s"].tolist(), ns["k"]
    return sum((x[i] + x[j]) % k == 0 for i in range(len(x)) for j in range(i + 1, len(x)))


@brute
def max_subarray_mod(ns):
    x, m = ns["s"].tolist(), ns["m"]
    return max(sum(x[i:j]) % m for i in range(len(x)) for j in range(i + 1, len(x) + 1))


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

# ---------------------------------------------------------------- 3.01

@brute
def window_peak_memory(ns):
    x, k = ns["mem"].tolist(), ns["k"]
    return [max(x[i:i + k]) for i in range(len(x) - k + 1)]


@brute
def running_kth_largest(ns):
    x, k = ns["scores"].tolist(), ns["k"]
    return [sorted(x[:i + 1], reverse=True)[k - 1] for i in range(k - 1, len(x))]


# ---------------------------------------------------------------- 3.02

@brute
def longest_broadcastable_run(ns):
    rows = ns["shapes"].tolist()

    def ok(win):
        return all(len({r[d] for r in win if r[d] != 1}) <= 1 for d in range(4))
    return max(j - i for i in range(len(rows)) for j in range(i + 1, len(rows) + 1) if ok(rows[i:j]))


@brute
def count_3d_views(ns):
    out = []
    for n in ns["sizes"].tolist():
        out.append(sum(1 for a in range(1, n + 1) if n % a == 0 for b in range(1, n // a + 1) if (n // a) % b == 0))
    return out


# ---------------------------------------------------------------- 3.03

@brute
def backprop_dag_mod(ns):
    n, edges = ns["n"], ns["edges"].tolist()

    def paths(u):
        if u == n - 1:
            return 1
        return sum(w * paths(v) for a, v, w in edges if a == u)
    return paths(0) % (10**9 + 7)


@brute
def backward_schedule(ns):
    n, edges = ns["n"], [tuple(e) for e in ns["edges"].tolist()]
    best = None
    for order in itertools.permutations(range(n)):
        pos = {op: i for i, op in enumerate(order)}
        if all(pos[b] < pos[a] for a, b in edges) and (best is None or list(order) > best):
            best = list(order)
    return best


# ---------------------------------------------------------------- 3.04

@brute
def dynamic_weighted_sampling(ns):
    w, out = ns["weights"].tolist(), []
    for t, a, b in ns["ops"].tolist():
        if t == 0:
            w[a] = b
        else:
            acc = 0
            for i, v in enumerate(w):
                acc += v
                if acc > a:
                    out.append(i)
                    break
    return out


def _set_partitions(items):
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for part in _set_partitions(rest):
        for i in range(len(part)):
            yield part[:i] + [[first] + part[i]] + part[i + 1:]
        yield [[first]] + part


@brute
def bucketed_padding_cost(ns):
    L, B, C = ns["lengths"].tolist(), ns["B"], ns["C"]
    return min(sum(len(g) * max(g) - sum(g) + C for g in part)
               for part in _set_partitions(L) if all(len(g) <= B for g in part))


# ---------------------------------------------------------------- 3.05

@brute
def longest_improving_checkpoints(ns):
    x = ns["val_acc"].tolist()
    best = [1] * len(x)
    for i in range(len(x)):
        for j in range(i):
            if x[j] < x[i]:
                best[i] = max(best[i], best[j] + 1)
    return max(best)


@brute
def tied_param_groups(ns):
    numel, n = ns["numel"].tolist(), len(ns["numel"])
    adj = {i: set() for i in range(n)}
    for a, b in ns["ties"].tolist():
        adj[a].add(b)
        adj[b].add(a)
    seen, groups, total = set(), 0, 0
    for i in range(n):
        if i in seen:
            continue
        groups += 1
        total += numel[i]
        stack = [i]
        seen.add(i)
        while stack:
            for j in adj[stack.pop()]:
                if j not in seen:
                    seen.add(j)
                    stack.append(j)
    return [groups, total]


# ---------------------------------------------------------------- 3.06

@brute
def max_grad_scale_chain(ns):
    x = ns["factors"].tolist()
    return max(math.prod(x[i:j]) for i in range(len(x)) for j in range(i + 1, len(x) + 1))


@brute
def range_grad_products(ns):
    x = ns["factors"].tolist()
    return [math.prod(x[l:r + 1]) % (10**9 + 7) for l, r in ns["queries"].tolist()]


# ---------------------------------------------------------------- 3.07

@brute
def best_fit_allocator(ns):
    N, used, live, out = ns["N"], [], {}, []   # used: list of (addr, size)
    for t, i, s in ns["ops"].tolist():
        if t == 1:
            if i in live:
                used.remove(live.pop(i))
            continue
        blocks, pos = [], 0
        for a, sz in sorted(used):
            if a > pos:
                blocks.append((a - pos, pos))
            pos = a + sz
        if pos < N:
            blocks.append((N - pos, pos))
        fit = sorted(b for b in blocks if b[0] >= s)
        if not fit:
            out.append(-1)
            continue
        a = fit[0][1]
        live[i] = (a, s)
        used.append((a, s))
        out.append(a)
    return out


@brute
def largest_rectangle_hist(ns):
    h = ns["free_mb"].tolist()
    return max(min(h[i:j]) * (j - i) for i in range(len(h)) for j in range(i + 1, len(h) + 1))


# ---------------------------------------------------------------- 3.08

@brute
def windowed_attention_peak(ns):
    q, k, w = ns["q"].tolist(), ns["k"].tolist(), ns["w"]
    return [max(q[i] * k[j] for j in range(max(0, i - w + 1), i + 1)) for i in range(len(q))]


@brute
def causal_top1_counts(ns):
    q, k = ns["q"].tolist(), ns["k"].tolist()
    counts = [0] * len(k)
    for i in range(len(q)):
        scores = [q[i] * k[j] for j in range(i + 1)]
        counts[scores.index(max(scores))] += 1
    return counts


# ---------------------------------------------------------------- 3.09

@brute
def greedy_decode_cycle(ns):
    t, table = ns["start"], ns["next_tok"].tolist()
    for _ in range(ns["N"]):
        t = table[t]
    return t


@brute
def top_p_cutoff_queries(ns):
    w = sorted(ns["weights"].tolist(), reverse=True)
    out = []
    for p in ns["thresholds"].tolist():
        acc, k = 0, 0
        while acc < p:
            acc += w[k]
            k += 1
        out.append(k)
    return out


# ---------------------------------------------------------------- 3.10

@brute
def range_max_activation(ns):
    x = ns["peak"].tolist()
    return [max(x[l:r + 1]) for l, r in ns["queries"].tolist()]


@brute
def spike_pairs(ns):
    x = ns["grad_norm"].tolist()
    return sum(x[j] >= 2 * x[i] for i in range(len(x)) for j in range(i + 1, len(x)))


# ---------------------------------------------------------------- 3.11

@brute
def optimal_kv_evictions(ns):
    seq, C = ns["blocks"].tolist(), ns["C"]
    cache, misses = set(), 0
    for i, b in enumerate(seq):
        if b in cache:
            continue
        misses += 1
        if len(cache) == C:
            def next_use(x):
                return next((j for j in range(i + 1, len(seq)) if seq[j] == x), len(seq) + 1)
            cache.remove(max(cache, key=next_use))
        cache.add(b)
    return misses


@brute
def prefix_cache_reuse(ns):
    reqs, total = ns["requests"], 0
    for i, r in enumerate(reqs):
        best = 0
        for p in reqs[:i]:
            k = 0
            while k < min(len(r), len(p)) and r[k] == p[k]:
                k += 1
            best = max(best, k)
        total += best
    return total


# ---------------------------------------------------------------- 3.12

@brute
def lora_rank_allocation(ns):
    cost, gains, budget = ns["cost"].tolist(), ns["gains"].tolist(), ns["budget"]
    ranks = (0, 4, 8, 16)
    best = 0
    for pick in itertools.product(range(4), repeat=len(cost)):
        if sum(ranks[k] * c for k, c in zip(pick, cost)) <= budget:
            best = max(best, sum(g[k - 1] for k, g in zip(pick, gains) if k))
    return best


@brute
def min_rank_for_tolerance(ns):
    sv = ns["sv"].tolist()
    return [next(r for r in range(len(sv) + 1) if sum(v * v for v in sv[r:]) <= tau) for tau in ns["tolerances"].tolist()]


# ---------------------------------------------------------------- 3.13

@brute
def bf16_collisions(ns):
    def bf16(v):  # round the float64 mantissa to 8 bits; Python's round() is ties-to-even
        m, e = math.frexp(v)
        return round(m * 256) * 2.0 ** (e - 8)
    x = ns["x"].tolist()
    return sum(x[i] != x[j] and bf16(x[i]) == bf16(x[j]) for i in range(len(x)) for j in range(i + 1, len(x)))


@brute
def grad_scaler_trace(ns):
    scale, streak, skipped = 2.0 ** ns["k0"], 0, 0
    for v in ns["grads"].tolist():
        if v * scale > 65504:
            skipped, scale, streak = skipped + 1, scale / 2, 0
        else:
            streak += 1
            if streak == ns["G"]:
                scale, streak = scale * 2, 0
    return [skipped, int(math.log2(scale))]


# ---------------------------------------------------------------- 3.14

@brute
def critical_path_ms(ns):
    d, deps = ns["dur_us"].tolist(), ns["deps"].tolist()
    memo = {}

    def finish(i):
        if i not in memo:
            memo[i] = d[i] + max([finish(a) for a, b in deps if b == i], default=0)
        return memo[i]
    return max(finish(i) for i in range(len(d)))


@brute
def top_self_time(ns):
    parent, tot = ns["parent"].tolist(), ns["total_us"].tolist()
    self_us = [tot[i] - sum(tot[j] for j in range(len(tot)) if parent[j] == i) for i in range(len(tot))]
    order = sorted(range(len(tot)), key=lambda i: (-self_us[i], i))
    return order[:ns["k"]]


# ---------------------------------------------------------------- 3.15

@brute
def swizzled_tile_loads(ns):
    pm, pn, G = ns["num_pid_m"], ns["num_pid_n"], ns["GROUP_M"]
    per = G * pn
    out = []
    for P in ns["queries"].tolist():
        rows, cols = set(), set()
        for pid in range(P):
            first = pid // per * G
            size = min(pm - first, G)
            rows.add(first + (pid % per) % size)
            cols.add((pid % per) // size)
        out.append(len(rows) + len(cols))
    return out


@brute
def pow2_block_fit(ns):
    x, S = ns["x"].tolist(), ns["S"]
    best, B = 0, 1
    while B < 2 * len(x):
        if all(sum(x[i:i + B]) <= S for i in range(0, len(x), B)):
            best = B
        B *= 2
    return best


# ---------------------------------------------------------------- 3.16

@brute
def ddp_bucket_timeline(ns):
    sz, rd, cap = ns["size"].tolist(), ns["ready"].tolist(), ns["cap"]
    order = list(range(len(sz)))[::-1]
    buckets = [[order[0]]]
    for i in order[1:]:
        if sum(sz[j] for j in buckets[-1]) + sz[i] > cap:
            buckets.append([i])
        else:
            buckets[-1].append(i)
    t = 0
    for b in buckets:
        t = max(t, *(rd[i] for i in b)) + sum(sz[i] for i in b)
    return t


@brute
def world_sizes_low_waste(ns):
    n, d = ns["n"], ns["d"]
    return sum(n % r <= d for r in range(1, n + 1))


# ---------------------------------------------------------------- 3.17

@brute
def checkpoint_segment_size(ns):
    a = ns["a"].tolist()
    n = len(a)
    best = None
    for s in range(1, n + 1):
        chunks = [a[i:i + s] for i in range(0, n, s)]
        cand = (sum(c[0] for c in chunks) + max(sum(c) for c in chunks), s)
        best = cand if best is None or cand < best else best
    return list(best)


@brute
def fsdp_unit_padding(ns):
    x, w, c = ns["numel"].tolist(), ns["world"], ns["overhead"]
    best = None
    for cuts in itertools.product([0, 1], repeat=len(x) - 1):
        units, cur = [], [x[0]]
        for v, cut in zip(x[1:], cuts):
            if cut:
                units.append(cur)
                cur = [v]
            else:
                cur.append(v)
        units.append(cur)
        cost = sum(c + (-sum(u)) % w for u in units)
        best = cost if best is None else min(best, cost)
    return best


# ---------------------------------------------------------------- 3.18

@brute
def count_recompiles(ns):
    calls, limit = [tuple(s) for s in ns["shapes"].tolist()], ns["limit"]
    graphs, misses, fallbacks = [], [], 0
    for s in calls:
        if any(all(v == q or (q is None and v >= 2) for v, q in zip(s, g)) for g in graphs):
            continue
        misses.append(s)
        dyn = {i for m in misses for i in range(len(s)) if m[i] != misses[0][i]}
        if len(graphs) < limit:
            graphs.append(tuple(None if i in dyn and v >= 2 else v for i, v in enumerate(s)))
        else:
            fallbacks += 1
    return [len(graphs), fallbacks]


@brute
def longest_graph_k_fixes(ns):
    f, k = ns["is_break"].tolist(), ns["k"]
    return max(j - i for i in range(len(f)) for j in range(i + 1, len(f) + 1) if sum(f[i:j]) <= k)


# ---------------------------------------------------------------- 3.19

@brute
def discordant_pairs(ns):
    p, r = ns["policy"].tolist(), ns["ref"].tolist()
    return sum((p[i] - p[j]) * (r[i] - r[j]) < 0 for i in range(len(p)) for j in range(i + 1, len(p)))


@brute
def max_preference_pairs(ns):
    c, r, d = ns["chosen"].tolist(), ns["rejected"].tolist(), ns["margin"]

    def best(i, used):
        if i == len(c):
            return 0
        out = best(i + 1, used)
        for j in range(len(r)):
            if not used >> j & 1 and c[i] >= r[j] + d:
                out = max(out, 1 + best(i + 1, used | 1 << j))
        return out
    return best(0, 0)


def test_every_challenge_has_a_brute_force():
    missing = sorted({t.name for t in CHALLENGES} - set(BRUTE))
    assert not missing, missing


def _same(got, expected):
    import pandas as pd
    if isinstance(expected, pd.DataFrame):
        try:
            pd.testing.assert_frame_equal(got, expected, check_dtype=False, check_names=False, check_index_type=False,
                                          check_column_type=False)
            return True
        except AssertionError:
            return False
    if isinstance(expected, pd.Series):
        try:
            pd.testing.assert_series_equal(pd.Series(got), expected, check_dtype=False, check_names=False,
                                           check_index_type=False)
            return True
        except AssertionError:
            return False
    if isinstance(expected, np.ndarray):
        if expected.dtype.kind == "f":
            return np.allclose(np.asarray(got, dtype=float), expected, equal_nan=True)
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
