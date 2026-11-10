"""4.09 SVMs, nearest neighbours & naive Bayes."""

import warnings

import numpy as np

from . import brute


@brute
def knn_on_a_line(ns):
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, RadiusNeighborsRegressor
    X, y, Q = ns["x"].reshape(-1, 1), ns["y"], ns["q"].reshape(-1, 1)
    if ns["mode"] == "knn_mean":
        return KNeighborsRegressor(n_neighbors=ns["k"]).fit(X, y).predict(Q)
    if ns["mode"] == "knn_vote":
        return KNeighborsClassifier(n_neighbors=ns["k"]).fit(X, y).predict(Q)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # empty neighbourhoods predict nan with a warning
        return RadiusNeighborsRegressor(radius=ns["r"]).fit(X, y).predict(Q)


@brute
def naive_bayes_sparse(ns):
    from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
    V, y = ns["V"], ns["y"]
    Xtr = np.zeros((len(y), V))
    np.add.at(Xtr, (ns["doc"], ns["term"]), ns["count"])
    Xte = np.zeros((ns["n_test"], V))
    np.add.at(Xte, (ns["tdoc"], ns["tterm"]), ns["tcount"])
    model = {"multinomial": MultinomialNB, "complement": ComplementNB, "bernoulli": BernoulliNB}[ns["model"]]
    return model(alpha=ns["alpha"]).fit(Xtr, y).predict(Xte)


@brute
def kernel_collapse(ns):
    from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel
    sv, Q = ns["sv"], ns["Q"]
    K = linear_kernel(Q, sv) if ns["kernel"] == "linear" else \
        polynomial_kernel(Q, sv, degree=2, gamma=ns["gamma"], coef0=ns["coef0"])
    f = K @ ns["dual"] + ns["b"]
    return f if ns["report"] == "decision" else (f > 0).astype(int)
