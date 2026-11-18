"""5.04 Bars, histograms & distributions: ask Matplotlib itself (hexbin, cbook.boxplot_stats, hist)."""

import numpy as np

from . import brute


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


@brute
def hexbin_counts(ns):
    plt = _plt()
    fig, ax = plt.subplots()
    hb = ax.hexbin(ns["x"], ns["y"], gridsize=ns["gridsize"], extent=ns["extent"], mincnt=1)
    counts = np.asarray(hb.get_array()).astype(int)
    offs = np.asarray(hb.get_offsets())
    plt.close(fig)
    if ns["report"] == "nonempty":
        return len(counts)
    if ns["report"] == "histogram":
        return sorted(counts.tolist(), reverse=True)
    return offs[int(np.argmax(counts))].tolist()     # offsets keep hexbin's own order; argmax takes the first


@brute
def box_stats(ns):
    from matplotlib import cbook
    out = []
    for k in range(ns["G"]):
        st = cbook.boxplot_stats(ns["values"][ns["group"] == k])[0]
        out.append({"whiskers": [st["whislo"], st["whishi"]], "fliers": len(st["fliers"]),
                    "notch": [st["cilo"], st["cihi"]]}[ns["report"]])
    return np.array(out)


@brute
def hist_binning(ns):
    plt = _plt()
    fig, ax = plt.subplots()
    if ns["report"] == "edges":
        counts, _, _ = ax.hist(ns["data"], bins=ns["edges"])
    else:
        counts, _, _ = ax.hist(ns["data"], bins=ns["bins"], range=(ns["lo"], ns["hi"]), density=ns["report"] == "density")
    plt.close(fig)
    return counts if ns["report"] == "density" else counts.astype(int)
