"""4.02 Datasets."""

from collections import Counter, deque

import numpy as np

from . import brute


@brute
def shortest_class_window(ns):
    y, k, m = ns["y"].tolist(), ns["k"], ns["m"]
    good = [(r - l, l) for l in range(len(y)) for r in range(l + 1, len(y) + 1)
            if all(y[l:r].count(c) >= m for c in range(k))]
    if not good:
        return -1
    best = min(g[0] for g in good)
    return {"length": best, "start": min(l for g, l in good if g == best),
            "count": sum(g == best for g, _ in good)}[ns["report"]]


@brute
def label_set_pairs(ns):
    rows = [tuple(r) for r in ns["Y"].tolist()]
    if ns["report"] == "pairs":
        return sum(rows[i] == rows[j] for i in range(len(rows)) for j in range(i + 1, len(rows)))
    if ns["report"] == "distinct":
        return len(set(rows))
    c = Counter(rows)
    code = lambda r: sum(2 ** j for j, v in enumerate(r) if v)  # noqa: E731
    best = min(c, key=lambda r: (-c[r], code(r)))
    return [j for j, v in enumerate(best) if v]


@brute
def swiss_roll_unrolled(ns):
    from scipy.integrate import quad
    X = ns["X"]
    t = np.sqrt(X[:, 0] ** 2 + X[:, 2] ** 2)
    u = np.array([quad(lambda s: np.sqrt(1 + s * s), 0, ti, epsabs=1e-13, epsrel=1e-13)[0] for ti in t])
    n = len(u)
    if ns["report"] == "nearest":
        return np.array([min(abs(u[i] - u[j]) for j in range(n) if j != i) for i in range(n)])
    if ns["report"] == "gap":
        us = sorted(u)
        return max(b - a for a, b in zip(us, us[1:]))
    return sum(abs(u[i] - u[j]) <= ns["D"] for i in range(n) for j in range(i + 1, n))


@brute
def circles_radius_stump(ns):
    X, y = ns["X"], ns["y"]
    r = np.sqrt((X ** 2).sum(axis=1))
    cands = [(-np.inf, int((y == 1).sum()))]
    for th in sorted(r):
        cands.append((th, int(((r <= th) != (y == 1)).sum())))
    best = min(e for _, e in cands)
    if ns["report"] == "errors":
        return best
    if ns["report"] == "accuracy":
        return 1 - best / len(y)
    th = min(t for t, e in cands if e == best)
    return -1.0 if th == -np.inf else float(th)


def _components(mask, nbrs):
    h, w = mask.shape
    seen, comps = np.zeros_like(mask), []
    for i in range(h):
        for j in range(w):
            if mask[i, j] and not seen[i, j]:
                seen[i, j] = True
                q, cells = deque([(i, j)]), []
                while q:
                    a, b = q.popleft()
                    cells.append((a, b))
                    for da, db in nbrs:
                        x, z = a + da, b + db
                        if 0 <= x < h and 0 <= z < w and mask[x, z] and not seen[x, z]:
                            seen[x, z] = True
                            q.append((x, z))
                comps.append(cells)
    return comps


@brute
def digit_holes(ns):
    four = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    eight = four + [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    out = []
    for img in ns["imgs"]:
        ink = img > ns["thr"]
        if ns["report"] == "strokes":
            out.append(len(_components(ink, eight)))
        else:
            comps = _components(~ink, four)
            out.append(sum(all(0 < a < 7 and 0 < b < 7 for a, b in c) for c in comps))
    out = np.array(out)
    if ns["report"] == "histogram":
        return np.array([(out == 0).sum(), (out == 1).sum(), (out >= 2).sum()])
    return out


@brute
def class_slice_queries(ns):
    y, k, out = ns["y"].tolist(), ns["k"], []
    for lo, hi in zip(ns["l"], ns["r"]):
        part = y[lo:hi]
        if ns["report"] == "distinct":
            out.append(len(set(part)))
        elif ns["report"] == "missing":
            out.append(next((c for c in range(k) if c not in part), -1))
        else:
            out.append(1 + sum(a != b for a, b in zip(part, part[1:])))
    return np.array(out)
