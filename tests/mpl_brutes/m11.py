"""5.11 Styles & rcParams: naive replays (full snapshots, scans) and real Matplotlib property cycles."""

import numpy as np

from . import brute


@brute
def rc_context_stack(ns):
    K = ns["K"]
    cur, snaps, answers = [0] * K, [], []
    for o, k, v in zip(ns["op"].tolist(), ns["key"].tolist(), ns["val"].tolist()):
        if o == 0:
            cur[k] = v
        elif o == 1:
            snaps.append(list(cur))
        elif o == 2:
            if snaps:
                cur = snaps.pop()
        elif o == 3:
            cur = [0] * K
        else:
            answers.append(cur[k])
    return {"queries": np.array(answers, dtype=int), "sum": sum(cur), "changed": sum(x != 0 for x in cur)}[ns["report"]]


@brute
def style_cascade(ns):
    sheets = {}
    for st, k, v in zip(ns["style"].tolist(), ns["key"].tolist(), ns["value"].tolist()):
        sheets.setdefault(st, {})[k] = v
    vals, srcs = [], []
    for a, b, k in zip(ns["qa"].tolist(), ns["qb"].tolist(), ns["qk"].tolist()):
        rc, src = {}, {}
        for st in range(a, b + 1):
            for kk, v in sheets.get(st, {}).items():
                rc[kk], src[kk] = v, st
        vals.append(rc.get(k, 0))
        srcs.append(src.get(k, -1))
    rep = ns["report"]
    if rep == "value":
        return np.array(vals)
    if rep == "source":
        return np.array(srcs)
    return sum(s == -1 for s in srcs)


@brute
def cycle_advance_rule(ns):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    A, c = ns["A"], ns["c"]
    cols = [f"#{i:02x}{i:02x}{i:02x}" for i in range(c)]
    lws = [i + 1.0 for i in range(c)]
    fig, axs = plt.subplots(1, A, squeeze=False)
    axs = axs[0]
    try:
        for a in axs:
            a.set_prop_cycle(color=cols, linewidth=lws)
        used = []
        for a, k in zip(ns["ax_i"].tolist(), ns["kind"].tolist()):
            ax = axs[a]
            if k == 3:
                ax.set_prop_cycle(color=cols, linewidth=lws)
            elif k == 2:
                ax.plot([0, 1], color="k", linewidth=0.5)
                used.append(-1)
            else:
                line, = ax.plot([0, 1], color="k") if k == 1 else ax.plot([0, 1])
                used.append(int(round(line.get_linewidth())) - 1)
        nxt = [int(round(ax.plot([0, 1])[0].get_linewidth())) - 1 for ax in axs]
    finally:
        plt.close(fig)
    u = np.array(used, dtype=int)
    return {"used": u, "next": np.array(nxt), "counts": np.bincount(u[u >= 0], minlength=c)}[ns["report"]]
