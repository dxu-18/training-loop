"""4.20 Custom estimators, persistence & configuration."""

import bisect

import numpy as np

from . import brute


@brute
def clipper_quantile_sweep(ns):
    x, out = ns["x"], []
    for q in ns["qs"]:
        lo, hi = np.quantile(x, [q, 1 - q])
        c = np.clip(x, lo, hi)
        out.append({"mean": c.mean(), "var": c.var(), "changed": int((c != x).sum())}[ns["report"]])
    return np.array(out)


@brute
def prefix_balanced_weights(ns):
    from sklearn.utils.class_weight import compute_class_weight
    y, out = ns["y"], []
    for t in range(1, len(y) + 1):
        cls = np.unique(y[:t])
        w = compute_class_weight("balanced", classes=cls, y=y[:t])
        out.append({"current": w[list(cls).index(y[t - 1])], "max": w.max(), "min": w.min()}[ns["report"]])
    return np.array(out, dtype=float)


@brute
def online_ecdf(ns):
    seen, out = [], []
    for is_add, v in zip(ns["op"].tolist(), ns["val"].tolist()):
        if is_add:
            bisect.insort(seen, v)
            continue
        less = sum(s < v for s in seen)
        leq = sum(s <= v for s in seen)
        out.append(leq / len(seen) if ns["report"] == "leq" else (less + leq) / (2 * len(seen)))
    return np.array(out, dtype=float)


@brute
def smoothed_target_encoder(ns):
    cat, y, m = ns["cat"].tolist(), ns["y"].tolist(), ns["smoothing"]

    def enc(c, rows):
        prior = sum(y[i] for i in rows) / len(rows)
        sel = [y[i] for i in rows if cat[i] == c]
        return (sum(sel) + m * prior) / (len(sel) + m)

    every = range(len(y))
    if ns["report"] == "encode":
        return np.array([enc(c, every) for c in ns["query"].tolist()])
    return np.array([enc(cat[i], [j for j in every if j != i]) for i in every])


@brute
def multioutput_sample_weight(ns):
    from sklearn.utils.class_weight import compute_sample_weight
    w = compute_sample_weight("balanced", ns["Y"])
    if ns["report"] == "weights":
        return w
    if ns["report"] == "top":
        return int(np.argmax(w))
    return float(w.sum())
