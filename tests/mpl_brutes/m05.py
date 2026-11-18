"""5.05 Areas, error bars, reference lines & spans: naive unit-cell sweeps for the Gantt lanes, and Matplotlib itself
(fill_between's polygons, axline's clipped segment) for the band areas and the infinite lines."""

import numpy as np

from . import brute


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


@brute
def gantt_lanes(ns):
    lane, start, dur, L = ns["lane"].tolist(), ns["start"].tolist(), ns["dur"].tolist(), ns["L"]
    T = max(s + d for s, d in zip(start, dur)) + 1
    if ns["report"] == "peak":      # bars are half-open [s, s + d): count them on every unit cell [t, t + 1)
        return max(sum(s <= t < s + d for s, d in zip(start, dur)) for t in range(T))
    out = []
    for k in range(L):
        cells = [any(l == k and s <= t < s + d for l, s, d in zip(lane, start, dur)) for t in range(T)]
        if ns["report"] == "busy":
            out.append(sum(cells))
        else:                        # idle gaps = busy blocks - 1
            blocks = sum(c and (t == 0 or not cells[t - 1]) for t, c in enumerate(cells))
            out.append(max(blocks - 1, 0))
    return np.array(out, dtype=np.int64)


def _shoelace(v):
    x, y = v[:, 0], v[:, 1]
    return abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))) / 2


@brute
def band_area(ns):
    plt = _plt()
    x, y1, y2 = ns["x"], ns["y1"], ns["y2"]
    fig, ax = plt.subplots()
    try:
        if ns["report"] == "regions":
            return len(ax.fill_between(x, y1, y2, where=y1 > y2).get_paths())
        wheres = [y1 > y2] if ns["report"] == "positive" else [y1 > y2, y1 < y2]
        return sum(_shoelace(p.vertices) for w in wheres
                   for p in ax.fill_between(x, y1, y2, where=w, interpolate=True).get_paths())
    finally:
        plt.close(fig)


@brute
def axline_clip(ns):
    plt = _plt()
    (x0, x1), (y0, y1) = ns["xlim"], ns["ylim"]
    fig, ax = plt.subplots()
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    segs = []
    for px, py, m in zip(ns["px"].tolist(), ns["py"].tolist(), ns["slope"].tolist()):
        line = ax.axline((px, py), slope=m)
        disp = line.get_transform().transform([[0.0, 0.0], [1.0, 1.0]])      # the segment Matplotlib draws
        (sx, sy), (ex, ey) = ax.transData.inverted().transform(disp)
        segs.append([sx, sy, ex, ey])
        line.remove()
    plt.close(fig)
    segs = np.array(segs)
    eps = 1e-6
    inside = [x0 - eps <= a <= x1 + eps and x0 - eps <= c <= x1 + eps and y0 - eps <= b <= y1 + eps
              and y0 - eps <= d <= y1 + eps for a, b, c, d in segs.tolist()]
    if ns["report"] == "segment":
        return segs
    if ns["report"] == "visible":
        return sum(inside)
    return float(sum(np.hypot(c - a, d - b) for (a, b, c, d), ok in zip(segs.tolist(), inside) if ok))
