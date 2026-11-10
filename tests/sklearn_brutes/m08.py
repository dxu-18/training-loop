"""4.08 Linear models."""

import numpy as np

from . import brute


@brute
def ridge_loo_shortcut(ns):
    from sklearn.linear_model import Ridge, RidgeCV
    X, y, alphas = ns["X"], ns["y"], ns["alphas"]
    if ns["report"] == "alpha":
        return float(RidgeCV(alphas=alphas).fit(X, y).alpha_)
    preds = []                                   # explicit leave-one-out refits
    for a in alphas:
        p = np.empty(len(y))
        for i in range(len(y)):
            keep = np.arange(len(y)) != i
            p[i] = Ridge(alpha=a).fit(X[keep], y[keep]).predict(X[i:i + 1])[0]
        preds.append(p)
    preds = np.array(preds)
    mse = ((y - preds) ** 2).mean(axis=1)
    return mse if ns["report"] == "mse" else preds[np.argmin(mse)]


@brute
def lasso_orthogonal_path(ns):
    from sklearn.linear_model import Lasso
    b = ns["b"]
    d = len(b)
    X = np.sqrt(d) * np.eye(d)                   # X^T X / n = I with n = d rows
    y = b * np.sqrt(d)                           # so that X^T y / n = b
    out = []
    for a in ns["alphas"]:
        w = Lasso(alpha=a, fit_intercept=False, tol=1e-12, max_iter=10000).fit(X, y).coef_
        out.append({"count": int((w != 0).sum()), "l1": np.abs(w).sum(), "l2sq": (w ** 2).sum()}[ns["report"]])
    return np.array(out)


@brute
def huber_epsilon_queries(ns):
    r, sigma, pen = ns["r"], ns["sigma"], ns["penalty"]
    out = []
    for e in ns["eps"]:
        outl = np.abs(r) > e * sigma
        if ns["report"] == "outliers":
            out.append(int(outl.sum()))
        else:
            out.append(len(r) * sigma + (r[~outl] ** 2).sum() / sigma
                       + (2 * e * np.abs(r[outl]) - sigma * e ** 2).sum() + pen)
    return np.array(out)
