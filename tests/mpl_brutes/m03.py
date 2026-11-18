"""5.03 Line & scatter plots: naive decimation, a real-Matplotlib replay of the property cycle, and real transData."""

import numpy as np

from . import brute


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


@brute
def decimate_series(ns):
    x, y, W = ns["x"].tolist(), ns["y"].tolist(), ns["W"]
    cols: dict = {}
    for i, (a, b) in enumerate(zip(x, y)):
        c = min(W - 1, int((a - x[0]) / (x[-1] - x[0]) * W))
        cols.setdefault(c, []).append(i)
    if ns["report"] == "envelope":
        env = np.full((W, 2), np.nan)
        for c, idx in cols.items():
            env[c] = [min(y[i] for i in idx), max(y[i] for i in idx)]
        return env
    keep = set()
    for idx in cols.values():
        lo = min(idx, key=lambda i: (y[i], i))
        hi = min(idx, key=lambda i: (-y[i], i))
        keep |= {idx[0], idx[-1], lo, hi}
    return np.array(sorted(keep)) if ns["report"] == "indices" else len(keep)


@brute
def prop_cycle_replay(ns):
    plt = _plt()
    from matplotlib.colors import to_hex
    P = ns["P"]
    palette = ["#%06x" % (0x102030 + 0x0b1d2f * i) for i in range(P)]
    index = {c: i for i, c in enumerate(palette)}
    fig, ax = plt.subplots()
    ax.set_prop_cycle(color=palette)
    got, scat = [], []          # palette index of every default-colored artist / of every scatter
    for o, a in zip(ns["op"].tolist(), ns["arg"].tolist()):
        if o == 0:
            got += [index[to_hex(l.get_color())] for l in ax.plot(np.zeros((2, a)))]
        elif o == 1:
            ax.plot([0, 1], color="k")
        elif o == 2:
            i = index[to_hex(ax.scatter([0], [0]).get_facecolor()[0])]
            got.append(i)
            scat.append(i)
        elif o == 3:
            got.append(index[to_hex(ax.bar([0], [1])[0].get_facecolor())])
        elif o == 4:
            got.append(index[to_hex(ax.hist([1, 2, 3])[2][0].get_facecolor())])
        else:
            ax.set_prop_cycle(color=palette[:a])
    plt.close(fig)
    if ns["report"] == "histogram":
        return np.bincount(np.array(got, dtype=int), minlength=P)
    if ns["report"] == "scatter":
        return scat
    return got[-1] if got else -1


@brute
def hover_pick(ns):
    plt = _plt()
    fig = plt.figure(figsize=ns["figsize"], dpi=ns["dpi"])
    ax = fig.add_axes(ns["box"])
    ax.set_xscale(ns["xscale"])
    ax.set_xlim(*ns["xlim"])
    ax.set_ylim(*ns["ylim"])
    P = ax.transData.transform(np.column_stack([ns["px"], ns["py"]]))
    Q = ax.transData.transform(np.column_stack([ns["qx"], ns["qy"]]))
    plt.close(fig)
    out = []
    for qx, qy in Q.tolist():
        d = [(qx - px) ** 2 + (qy - py) ** 2 for px, py in P.tolist()]
        if ns["report"] == "within":
            out.append(sum(v <= ns["radius"] ** 2 for v in d))
        else:
            out.append(min(range(len(d)), key=lambda i: (d[i], i)))
    return np.array(out)
