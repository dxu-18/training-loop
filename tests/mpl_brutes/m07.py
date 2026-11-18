"""5.07 Pie, polar & 3D: naive replays (exact integer pie borders, O(n²) sectors, per-point zoom checks)."""

import math

import numpy as np

from . import brute


@brute
def pie_wedge_lookup(ns):
    sizes = ns["sizes"].tolist()
    T = sum(sizes)
    C = [0]
    for x in sizes:
        C.append(C[-1] + x)
    sign = 1 if ns["counterclock"] else -1
    out = []
    if ns["report"] == "wedge":
        for a in ns["angles"].tolist():
            d = (sign * (a - ns["start"])) % 360
            out.append(next(i for i in range(len(sizes)) if 360 * C[i] <= d * T < 360 * C[i + 1]))
    else:
        for x, y in zip(ns["xs"].tolist(), ns["ys"].tolist()):
            if math.hypot(x, y) > 1:
                out.append(-1)
                continue
            d = float(np.mod(sign * (np.degrees(np.arctan2(y, x)) - ns["start"]), 360))
            d = d - 360 if d >= 360 else d
            out.append(next(i for i in range(len(sizes)) if 360 * C[i] / T <= d < 360 * C[i + 1] / T))
    return np.array(out)


@brute
def densest_sector(ns):
    a = ns["a"].tolist()
    rep = ns["report"]
    if rep == "count":
        return max(sum((x - t) % 36000 <= ns["w"] for x in a) for t in a)
    if rep == "cover":
        return min(sorted((x - t) % 36000 for x in a)[ns["k"] - 1] for t in a)
    gaps = [min([(x - t) % 36000 for x in a if (x - t) % 36000 > 0], default=36000) for t in a]
    return max(gaps)


@brute
def polar_zoom_counts(ns):
    pts = list(zip(ns["theta"].tolist(), ns["r"].tolist()))
    out = []
    for t1, t2, r1, r2 in zip(ns["t1"].tolist(), ns["t2"].tolist(), ns["r1"].tolist(), ns["r2"].tolist()):
        vis = [r for th, r in pts if (t1 <= th <= t2 if t1 <= t2 else (th >= t1 or th <= t2)) and r1 <= r <= r2]
        out.append(len(vis) if ns["report"] == "count" else (sum(vis) / len(vis) if vis else 0.0))
    return np.array(out, dtype=int if ns["report"] == "count" else float)
