"""5.10 Colors, colormaps, norms & colorbars: ask Matplotlib's own norms and colormaps."""

import numpy as np

from . import brute


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.colors as mc
    return mc


@brute
def boundary_bins(ns):
    mc = _mpl()
    idx = np.asarray(mc.BoundaryNorm(ns["b"], ns["ncolors"], extend=ns["extend"])(ns["x"]).filled(-99), dtype=int)
    rep, nc = ns["report"], ns["ncolors"]
    if rep == "index":
        return idx
    if rep == "counts":
        return np.array([int((idx == k).sum()) for k in range(-1, nc + 1)])
    return len({k for k in idx.tolist() if 0 <= k < nc})


@brute
def lut_lookup(ns):
    mc = _mpl()
    N, kind = ns["N"], ns["norm"]
    cols = [((i + 1) / 64, (i * 7 % 64) / 64, 0.25) for i in range(N + 3)]
    cmap = mc.ListedColormap(cols[:N]).with_extremes(under=cols[N], over=cols[N + 1], bad=cols[N + 2])
    norm = {"linear": lambda: mc.Normalize(ns["vmin"], ns["vmax"]), "log": lambda: mc.LogNorm(ns["vmin"], ns["vmax"]),
            "twoslope": lambda: mc.TwoSlopeNorm(ns["vcenter"], ns["vmin"], ns["vmax"])}[kind]()
    rgba = cmap(norm(ns["x"]))
    where = {tuple(np.round(c, 6)): i for i, c in enumerate(cmap(np.arange(N)))}
    where[tuple(np.round(cmap.get_under(), 6))] = N
    where[tuple(np.round(cmap.get_over(), 6))] = N + 1
    where[tuple(np.round(cmap.get_bad(), 6))] = N + 2
    out = np.zeros(N + 3, dtype=int)
    for c in rgba:
        out[where[tuple(np.round(c, 6))]] += 1
    return out


@brute
def gradient_lut(ns):
    mc = _mpl()
    cmap = mc.LinearSegmentedColormap.from_list("c", list(zip(ns["pos"].tolist(), ns["rgb"].tolist())), N=ns["N"])
    c = cmap(ns["v"])[:, :3]
    rep = ns["report"]
    if rep == "rgb":
        return c
    if rep == "gray":
        return 0.2126 * c[:, 0] + 0.7152 * c[:, 1] + 0.0722 * c[:, 2]
    return len({tuple(row) for row in np.round(c, 9).tolist()})
