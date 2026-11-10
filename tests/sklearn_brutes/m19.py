"""4.19 Outliers, multi-output, semi-supervised & other models."""

import numpy as np

from . import brute


@brute
def pava_isotonic(ns):
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(increasing=ns["increasing"], out_of_bounds="clip")
    fit = iso.fit_transform(ns["x"], ns["y"], sample_weight=ns["w"])
    if ns["report"] == "fitted":
        return fit
    if ns["report"] == "sse":
        return float((ns["w"] * (ns["y"] - fit) ** 2).sum())
    return iso.predict(ns["queries"])


@brute
def lof_1d(ns):
    from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
    x, k = ns["x"][:, None], ns["k"]
    if ns["report"] == "kdist":
        dist, _ = NearestNeighbors(n_neighbors=k).fit(x).kneighbors()
        return dist[:, -1]
    if ns["report"] == "scores":
        return LocalOutlierFactor(n_neighbors=k).fit(x).negative_outlier_factor_
    return int((LocalOutlierFactor(n_neighbors=k).fit_predict(x) == -1).sum())


@brute
def running_baseline(ns):
    from sklearn.dummy import DummyClassifier, DummyRegressor
    y, out = ns["y"], []
    for t in range(1, len(y) + 1):
        X = np.zeros((t, 1))
        if ns["report"] == "median":
            out.append(float(DummyRegressor(strategy="median").fit(X, y[:t]).constant_.ravel()[0]))
        else:
            out.append(int(DummyClassifier(strategy="most_frequent").fit(X, y[:t]).predict(X[:1])[0]))
    return np.array(out)


@brute
def loo_mahalanobis(ns):
    from sklearn.covariance import EmpiricalCovariance
    X = ns["X"]
    d2 = np.array([EmpiricalCovariance().fit(np.delete(X, i, axis=0)).mahalanobis(X[i:i + 1])[0]
                   for i in range(len(X))])
    if ns["report"] == "dist":
        return d2
    if ns["report"] == "count":
        return int((d2 > ns["threshold"]).sum())
    return np.argsort(-d2, kind="stable")[:ns["m"]]


@brute
def ovo_vote_aggregation(ns):
    conf, pairs, k = ns["conf"], ns["pairs"], ns["k"]
    preds, ties = [], 0
    for row in conf.tolist():
        votes, sc = [0] * k, [0.0] * k
        for (i, j), c in zip(pairs, row):
            votes[j if c > 0 else i] += 1
            sc[i] -= c
            sc[j] += c
        ties += votes.count(max(votes)) > 1
        total = [v + c / (3 * (abs(c) + 1)) for v, c in zip(votes, sc)]
        preds.append(total.index(max(total)))
    return np.array(preds) if ns["report"] == "pred" else ties
