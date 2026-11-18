"""5.06 Images, heatmaps, contours & vector fields: naive grids, and Matplotlib's own colormap lookup."""

import numpy as np

from . import brute


@brute
def contour_crossings(ns):
    Z, rep = ns["Z"].tolist(), ns["report"]
    R, C = len(Z), len(Z[0])

    def cross(a, b, t):
        return (a < t) != (b < t)

    out = []
    for t in ns["levels"].tolist():
        if rep == "edges":
            k = sum(cross(Z[i][j], Z[i][j + 1], t) for i in range(R) for j in range(C - 1))
            k += sum(cross(Z[i][j], Z[i + 1][j], t) for i in range(R - 1) for j in range(C))
        else:
            k = 0
            for i in range(R - 1):
                for j in range(C - 1):
                    ring = [Z[i][j], Z[i][j + 1], Z[i + 1][j + 1], Z[i + 1][j]]
                    e = sum(cross(ring[m], ring[(m + 1) % 4], t) for m in range(4))
                    k += (e > 0) if rep == "cells" else e // 2
        out.append(k)
    return np.array(out)


@brute
def colormap_lookup(ns):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import colors
    v, N, rep = ns["v"], ns["N"], ns["report"]
    cmap = colors.ListedColormap(np.zeros((N, 3)))
    cmap._init()
    cmap._lut[:, 0] = np.arange(N + 3)          # read the table row back as the red channel
    if rep == "log":
        norm = colors.LogNorm(ns["vmin"], ns["vmax"])
    elif rep == "fixed":
        norm = colors.Normalize(ns["vmin"], ns["vmax"])
    else:
        norm = colors.Normalize()
        norm.autoscale_None(np.ma.masked_invalid(v))
    idx = np.rint(cmap(norm(v))[..., 0]).astype(int)
    return np.bincount(idx, minlength=N + 3) if rep == "fixed" else idx


@brute
def spy_blocks(ns):
    n, B, rep = ns["n"], ns["B"], ns["report"]
    s = -(-n // B)
    nz = set(zip(ns["rows"].tolist(), ns["cols"].tolist()))
    count = [[0] * B for _ in range(B)]
    for I in range(B):
        for J in range(B):
            count[I][J] = sum(I * s <= r < (I + 1) * s and J * s <= c < (J + 1) * s for r, c in nz)
    if rep == "dark":
        return sum(c > 0 for row in count for c in row)
    if rep == "busiest":
        return max(c for row in count for c in row)
    return np.array([sum(c > 0 for c in row) for row in count])
