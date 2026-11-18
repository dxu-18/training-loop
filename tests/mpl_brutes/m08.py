"""5.08 Titles, labels, limits, scales & ticks: exact fractions over the step ladder, Matplotlib's own ScalarFormatter
for the offset text, and naive pair checks for the label thinning."""

import math
from fractions import Fraction

import numpy as np

from . import brute


@brute
def nice_tick_steps(ns):
    ladder = sorted({1, 2, 5} | {m * 10 ** k for k in range(12) for m in (10, 20, 25, 50)})
    out = []
    for lo, hi, nb in zip(ns["lo"].tolist(), ns["hi"].tolist(), ns["nbins"].tolist()):
        s = next(s for s in ladder if math.ceil(Fraction(hi, s)) - math.floor(Fraction(lo, s)) <= nb)
        if ns["report"] == "step":
            out.append(s)
        elif ns["report"] == "ticks":
            out.append(sum(t % s == 0 for t in range(lo, hi + 1)))
        else:
            out.append([math.floor(Fraction(lo, s)) * s, math.ceil(Fraction(hi, s)) * s])
    return np.array(out, dtype=np.int64)


@brute
def offset_text(ns):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    fig, ax = plt.subplots()
    fmt = ScalarFormatter(useOffset=True)
    ax.yaxis.set_major_formatter(fmt)
    fmt._offset_threshold = ns["thr"]          # what rcParams['axes.formatter.offset_threshold'] sets
    off = []
    for lo, hi in zip(ns["lo"].tolist(), ns["hi"].tolist()):
        ax.set_ylim(lo, hi)
        fmt.set_locs([lo, hi])
        off.append(int(fmt.offset))
    plt.close(fig)
    off = np.array(off, dtype=np.int64)
    if ns["report"] == "count":
        return int(np.count_nonzero(off))
    return ns["hi"] - off if ns["report"] == "shown" else off


@brute
def label_thinning(ns):
    x, w, pad = ns["x"].tolist(), ns["w"].tolist(), ns["pad"]
    n = len(x)

    def hit(i, j):
        return x[i] + w[i] / 2 + pad > x[j] - w[j] / 2

    def collisions(s):
        return sum(hit(i, i + s) for i in range(0, n - s, s))

    if ns["report"] == "overlaps":
        return np.array([collisions(s) for s in ns["k"].tolist()])
    if ns["report"] == "stride":
        return next(s for s in range(1, n + 1) if collisions(s) == 0)
    kept = [0]
    for j in range(1, n):
        if not hit(kept[-1], j):
            kept.append(j)
    return len(kept)
