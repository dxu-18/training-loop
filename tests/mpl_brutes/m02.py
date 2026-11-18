"""5.02 Figures, subplots & layouts: ask Matplotlib itself (subplot_mosaic, GridSpec, shared-axes autoscaling)."""

import numpy as np

from . import brute


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


@brute
def mosaic_spans(ns):
    grid, report = ns["grid"], ns["report"]
    R, C = grid.shape
    if report == "touching":
        pairs = set()
        for r in range(R):
            for c in range(C):
                for rr, cc in ((r + 1, c), (r, c + 1)):
                    if rr < R and cc < C:
                        a, b = int(grid[r, c]), int(grid[rr, cc])
                        if a >= 0 and b >= 0 and a != b:
                            pairs.add((min(a, b), max(a, b)))
        return len(pairs)
    plt = _plt()
    if report == "spans":           # let subplot_mosaic itself place the panels
        fig, axd = plt.subplot_mosaic(grid.tolist(), empty_sentinel=-1)
        out = []
        for k in range(len(axd)):
            ss = axd[k].get_subplotspec()
            out.append([ss.rowspan.start, ss.rowspan.stop, ss.colspan.start, ss.colspan.stop])
        plt.close(fig)
        return np.array(out)
    bad = []                        # invalid: subplot_mosaic raises for exactly these labels
    for k in sorted({int(v) for v in grid.ravel()} - {-1}):
        mask = [[v == k for v in row] for row in grid.tolist()]
        try:
            fig, _ = plt.subplot_mosaic([[1 if m else 0 for m in row] for row in mask], empty_sentinel=0)
            plt.close(fig)
        except ValueError:
            bad.append(k)
    return bad


@brute
def gridspec_positions(ns):
    plt = _plt()
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=ns["figsize"])
    gs = GridSpec(ns["nrows"], ns["ncols"], figure=fig, left=ns["left"], right=ns["right"], bottom=ns["bottom"],
                  top=ns["top"], wspace=ns["wspace"], hspace=ns["hspace"], width_ratios=ns["width_ratios"],
                  height_ratios=ns["height_ratios"])
    out = []
    for r0, r1, c0, c1 in zip(ns["r0"].tolist(), ns["r1"].tolist(), ns["c0"].tolist(), ns["c1"].tolist()):
        b = gs[r0:r1, c0:c1].get_position(fig)
        out.append([b.x0, b.y0, b.width, b.height])
    plt.close(fig)
    out = np.array(out)
    if ns["report"] == "bounds":
        return out
    if ns["report"] == "aspect":
        return out[:, 2] * ns["figsize"][0] / (out[:, 3] * ns["figsize"][1])
    return np.column_stack([out[:, 0] + out[:, 2] / 2, out[:, 1] + out[:, 3] / 2])


@brute
def shared_axes_groups(ns):
    plt = _plt()
    n = ns["n"]
    fig, axes = plt.subplots(1, n)
    for ax, lo, hi in zip(axes, ns["lo"].tolist(), ns["hi"].tolist()):
        ax.plot([lo, hi], [0, 1])
    grouper = axes[0]._shared_axes["x"]
    for i, j in zip(ns["a"].tolist(), ns["b"].tolist()):
        grouper.join(axes[i], axes[j])
    for ax in axes:
        ax._request_autoscale_view("x")
    groups = {frozenset(id(s) for s in grouper.get_siblings(ax)) for ax in axes}
    lims = np.array([ax.get_xlim() for ax in axes])
    plt.close(fig)
    return {"xlim": lims, "groups": len(groups), "largest": max(len(g) for g in groups)}[ns["report"]]
