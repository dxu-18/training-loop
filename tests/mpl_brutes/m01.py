"""5.01 Setup, backends & the two interfaces: replay the scripts in real pyplot."""

import warnings

import numpy as np

from . import brute


@brute
def pyplot_state_machine(ns):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.close("all")
    old = plt.rcParams["figure.max_open_warning"]
    plt.rcParams["figure.max_open_warning"] = ns["limit"]
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            for o, k in zip(ns["op"].tolist(), ns["arg"].tolist()):
                if o == 0:
                    plt.figure(k)
                elif o == 1:
                    plt.figure()
                elif o == 2:
                    plt.close(k)
                else:
                    plt.plot([0, 1])
            fired = sum("max_open_warning" in str(x.message) or "More than" in str(x.message) for x in w)
        cur = plt.gcf().number if plt.get_fignums() else 0
        lines = len(plt.gca().get_lines()) if cur else 0
        return {"current": cur, "warnings": fired, "lines": lines}[ns["report"]]
    finally:
        plt.close("all")
        plt.rcParams["figure.max_open_warning"] = old


@brute
def autoscale_ranges(ns):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = []
    for a, b in zip(ns["a"].tolist(), ns["b"].tolist()):
        fig, ax = plt.subplots()
        if ns["report"] == "log":
            ax.set_yscale("log")
        for i in range(a, b + 1):
            ax.plot([0, 1], [ns["lo"][i], ns["hi"][i]])
        out.append(ax.get_ylim())
        plt.close(fig)
    lim = np.array(out)
    return lim[:, 1] - lim[:, 0] if ns["report"] == "height" else lim
