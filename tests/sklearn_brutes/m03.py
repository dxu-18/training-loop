"""4.03 Splitting & cross-validation."""

from fractions import Fraction
import math

import numpy as np

from . import brute


@brute
def kfold_mean_model(ns):
    from sklearn.dummy import DummyRegressor
    from sklearn.model_selection import KFold
    y, X = ns["y"], np.zeros((len(ns["y"]), 1))
    mse, worst = [], []
    for k in ns["ks"]:
        errs = []
        for tr, te in KFold(n_splits=int(k)).split(X):
            p = DummyRegressor().fit(X[tr], y[tr]).predict(X[te])
            errs.append(float(((y[te] - p) ** 2).mean()))
        mse.append(np.mean(errs))
        worst.append(max(errs))
    if ns["report"] == "best":
        return int(ns["ks"][int(np.argmin(mse))])
    return np.array(mse if ns["report"] == "mse" else worst)


@brute
def stratified_test_counts(ns):
    counts = [int(c) for c in ns["counts"]]
    N = sum(counts)
    t = math.ceil(ns["test_size"] * N)
    share = [Fraction(c * t, N) for c in counts]
    alloc = [math.floor(s) for s in share]
    for _ in range(t - sum(alloc)):            # one row at a time to the largest remaining fraction
        best = max(range(len(counts)), key=lambda c: (share[c] - alloc[c] if share[c] - alloc[c] < 1 else -1, -c))
        alloc[best] += 1
    if ns["report"] == "counts":
        return np.array(alloc)
    if ns["report"] == "zero":
        return sum(a == 0 for a in alloc)
    return float(max(abs(a - s) for a, s in zip(alloc, share)))


@brute
def group_folds_greedy(ns):
    sizes, k = ns["sizes"].tolist(), ns["k"]
    rows, fold_of = [0] * k, [0] * len(sizes)
    for gi in sorted(range(len(sizes)), key=lambda g: (-sizes[g], g)):
        f = min(range(k), key=lambda f: (rows[f], f))
        fold_of[gi] = f
        rows[f] += sizes[gi]
    return {"sizes": np.array(rows), "fold_of": np.array(fold_of), "spread": max(rows) - min(rows)}[ns["report"]]


@brute
def time_series_split_sizes(ns):
    from sklearn.model_selection import TimeSeriesSplit
    tot, capped = [], []
    for n, k, gap, M in zip(ns["n"], ns["n_splits"], ns["gap"], ns["max_train"]):
        full = [len(tr) for tr, te in TimeSeriesSplit(n_splits=int(k), gap=int(gap)).split(np.arange(n))]
        sizes = [len(tr) for tr, te in TimeSeriesSplit(n_splits=int(k), gap=int(gap),
                                                       max_train_size=int(M) or None).split(np.arange(n))]
        tot.append(sum(sizes))
        capped.append(sum(a != b for a, b in zip(full, sizes)))
    if ns["report"] == "mean_train":
        return np.array(tot) / np.array(ns["n_splits"])
    return np.array(tot if ns["report"] == "train_total" else capped)


@brute
def loo_nearest_neighbour(ns):
    x = ns["x"].tolist()
    nbr = []
    for i, xi in enumerate(x):
        nbr.append(min((j for j in range(len(x)) if j != i), key=lambda j: (abs(x[j] - xi), x[j])))
    nbr = np.array(nbr)
    if ns["report"] == "accuracy":
        return float((ns["label"][nbr] == ns["label"]).mean())
    if ns["report"] == "errors":
        return np.flatnonzero(ns["label"][nbr] != ns["label"])
    return float(((ns["target"][nbr] - ns["target"]) ** 2).mean())


@brute
def loo_linear_shortcut(ns):
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import LeaveOneOut, cross_val_predict
    loo = ns["y"] - cross_val_predict(LinearRegression(), ns["X"], ns["y"], cv=LeaveOneOut())
    return {"press": float((loo ** 2).sum()), "residuals": loo, "worst": int(np.abs(loo).argmax())}[ns["report"]]
