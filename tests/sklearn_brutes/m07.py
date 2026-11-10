"""4.07 Feature selection."""

import numpy as np

from . import brute


@brute
def chi2_term_selection(ns):
    from sklearn.feature_selection import SelectKBest, chi2
    X = np.zeros((ns["n_docs"], ns["V"]))
    np.add.at(X, (ns["doc"], ns["term"]), ns["count"])
    y = ns["y"]
    if ns["report"] == "topk":
        return SelectKBest(chi2, k=ns["k"]).fit(X, y).get_support(indices=True)
    scores, pvals = chi2(X, y)
    return scores if ns["report"] == "scores" else pvals


@brute
def percentile_queries(ns):
    from sklearn.feature_selection import SelectPercentile
    scores = ns["scores"]
    X, y = np.zeros((2, len(scores))), np.zeros(2)
    out = []
    for p in ns["p"]:
        mask = SelectPercentile(lambda X_, y_: scores.copy(), percentile=int(p)).fit(X, y).get_support()
        idx = np.flatnonzero(mask)
        out.append(int(mask.sum()) if ns["report"] == "count" else (int(idx.max()) if len(idx) else -1))
    return np.array(out)


@brute
def rfe_fixed_scores(ns):
    from sklearn.base import BaseEstimator
    from sklearn.feature_selection import RFE

    class Fixed(BaseEstimator):
        """A 'model' whose coefficient for a feature is the first row's value of that column."""
        def fit(self, X, y):
            self.coef_ = np.asarray(X[0], dtype=float).copy()
            return self

    w = ns["w"]
    rfe = RFE(Fixed(), n_features_to_select=ns["n_select"], step=ns["step"]).fit(w.reshape(1, -1), np.zeros(1))
    return rfe.ranking_
