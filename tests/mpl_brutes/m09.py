"""5.09 Legends, annotations, text & math: brute forces (naive loops over every pair / box / way to break)."""

import itertools

import numpy as np

from . import brute


@brute
def label_collisions(ns):
    x, y, score, w, h = ns["x"].tolist(), ns["y"].tolist(), ns["score"].tolist(), ns["w"], ns["h"]
    order = sorted(range(len(x)), key=lambda i: (-score[i], i))
    kept = []
    for i in order:
        bi = (x[i], x[i] + w, y[i], y[i] + h)
        hit = False
        for j in kept:
            bj = (x[j], x[j] + w, y[j], y[j] + h)
            if bi[0] < bj[1] and bj[0] < bi[1] and bi[2] < bj[3] and bj[2] < bi[3]:
                hit = True
        if not hit:
            kept.append(i)
    return {"kept": np.array(sorted(kept)), "count": len(kept), "last": kept[-1]}[ns["report"]]


@brute
def legend_best_spot(ns):
    W, H, pad, px, py = ns["W"], ns["H"], ns["pad"], ns["px"].tolist(), ns["py"].tolist()
    locs, bad = [], []
    for w, h in zip(ns["lw"].tolist(), ns["lh"].tolist()):
        mx, my = (W - w) // 2, (H - h) // 2
        spots = [(W - pad - w, H - pad - h), (pad, H - pad - h), (pad, pad), (W - pad - w, pad), (W - pad - w, my),
                 (pad, my), (W - pad - w, my), (mx, pad), (mx, H - pad - h), (mx, my)]
        counts = [sum(x0 < a < x0 + w and y0 < b < y0 + h for a, b in zip(px, py)) for x0, y0 in spots]
        m = min(counts)
        locs.append(counts.index(m) + 1)
        bad.append(m)
    return {"loc": np.array(locs), "badness": np.array(bad), "clear": sum(b == 0 for b in bad)}[ns["report"]]


def _lines(lens, L):
    rows = [[lens[0]]]
    for w in lens[1:]:
        if sum(rows[-1]) + len(rows[-1]) + w <= L:
            rows[-1].append(w)
        else:
            rows.append([w])
    return len(rows)


@brute
def caption_wrap(ns):
    lens, L, k, rep = ns["lens"].tolist(), ns["L"], ns["k"], ns["report"]
    if rep == "greedy":
        return _lines(lens, L)
    if rep == "balanced":
        width = max(lens)
        while _lines(lens, width) > k:
            width += 1
        return width
    n, best = len(lens), None
    for cuts in itertools.product((0, 1), repeat=n - 1):
        rows, cur = [], [lens[0]]
        for c, w in zip(cuts, lens[1:]):
            if c:
                rows.append(cur)
                cur = [w]
            else:
                cur.append(w)
        rows.append(cur)
        widths = [sum(r) + len(r) - 1 for r in rows]
        if max(widths) > L:
            continue
        cost = sum((L - wd) ** 2 for wd in widths[:-1])
        best = cost if best is None else min(best, cost)
    return best
