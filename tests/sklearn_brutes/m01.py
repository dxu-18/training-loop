"""4.01 The estimator API."""

import numpy as np

from . import brute


@brute
def rolling_standardize(ns):
    from sklearn.preprocessing import StandardScaler
    x, out = ns["x"], []
    for lo, hi in zip(ns["l"], ns["r"]):
        z = StandardScaler().fit_transform(x[lo:hi].reshape(-1, 1)).ravel()
        out.append(z[-1] if ns["report"] == "last" else z[0])
    return np.array(out)


@brute
def merge_scaler_shards(ns):
    n, mu, var = 0, np.zeros(ns["means"].shape[1]), np.zeros(ns["means"].shape[1])
    for c, m, v in zip(ns["counts"], ns["means"], ns["variances"]):       # Chan et al., one shard at a time
        delta, tot = m - mu, n + c
        var = (n * var + c * v + delta ** 2 * n * c / tot) / tot
        mu, n = mu + delta * c / tot, tot
    return np.vstack([mu, np.sqrt(var)]) if ns["report"] == "mean_scale" else (var, int(n))


@brute
def prefix_r2(ns):
    from sklearn.linear_model import LinearRegression
    x, y, out = ns["x"], ns["y"], []
    for k in range(3, len(x) + 1):
        X = x[:k].reshape(-1, 1)
        reg = LinearRegression().fit(X, y[:k])
        out.append({"r2": reg.score(X, y[:k]), "slope": reg.coef_[0],
                    "mse": float(((y[:k] - reg.predict(X)) ** 2).mean())}[ns["report"]])
    return np.array(out)
