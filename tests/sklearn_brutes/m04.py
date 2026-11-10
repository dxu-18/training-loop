"""4.04 Scaling & feature transforms."""

import numpy as np

from . import brute


@brute
def sliding_minmax_scale(ns):
    from sklearn.preprocessing import MinMaxScaler
    x, w, out = ns["x"], ns["w"], []
    for i in range(len(x) - w + 1):
        win = x[i:i + w].reshape(-1, 1)
        z = MinMaxScaler().fit_transform(win).ravel()
        out.append({"last": z[-1], "first": z[0], "range": win.max() - win.min()}[ns["report"]])
    return np.array(out, dtype=float)


@brute
def rolling_robust_scale(ns):
    from sklearn.preprocessing import RobustScaler
    x, w, out = ns["x"], ns["w"], []
    for i in range(len(x) - w + 1):
        win = x[i:i + w].reshape(-1, 1)
        rs = RobustScaler().fit(win)
        out.append({"median": rs.center_[0], "iqr": np.percentile(win, 75) - np.percentile(win, 25),
                    "robust_last": rs.transform(win)[-1, 0]}[ns["report"]])
    return np.array(out, dtype=float)


@brute
def polynomial_feature_index(ns):
    from sklearn.preprocessing import PolynomialFeatures
    pf = PolynomialFeatures(degree=ns["degree"], interaction_only=ns["interaction_only"],
                            include_bias=ns["include_bias"]).fit(np.zeros((1, ns["n_features"])))
    if ns["report"] == "index":
        rows = [tuple(r) for r in pf.powers_.tolist()]
        return np.array([rows.index(tuple(q)) for q in ns["powers"].tolist()])
    names = pf.get_feature_names_out()
    return [str(names[c]) for c in ns["cols"]]


@brute
def sparse_maxabs_scale(ns):
    import scipy.sparse as sp
    from sklearn.preprocessing import MaxAbsScaler
    A = sp.coo_matrix((ns["vals"], (ns["rows"], ns["cols"])), shape=(ns["n_rows"], ns["n_cols"])).tocsr()
    sc = MaxAbsScaler().fit(A)
    Z = sc.transform(A).toarray()
    if ns["report"] == "scaled":
        return Z[ns["rows"], ns["cols"]]
    if ns["report"] == "max_abs":
        return sc.max_abs_
    return np.sqrt((Z ** 2).sum(axis=1))


@brute
def binarizer_threshold_sweep(ns):
    from sklearn.preprocessing import Binarizer
    out = []
    for t in ns["ts"]:
        B = Binarizer(threshold=t).fit_transform(ns["X"])
        out.append({"ones": int(B.sum()), "rows_any": int(B.any(axis=1).sum()),
                    "rows_all": int(B.all(axis=1).sum())}[ns["report"]])
    return np.array(out)


@brute
def quantile_uniform_transform(ns):
    xs = sorted(ns["x"].tolist())
    n, out = len(xs), []
    for v in ns["queries"].tolist():
        if v <= xs[0]:
            out.append(0.0)
        elif v >= xs[-1]:
            out.append(1.0)
        elif v in xs:
            pos = [i for i, a in enumerate(xs) if a == v]
            out.append((pos[0] + pos[-1]) / 2 / (n - 1))
        else:
            i = next(i for i, a in enumerate(xs) if a > v)
            out.append((i - 1 + (v - xs[i - 1]) / (xs[i] - xs[i - 1])) / (n - 1))
    u = np.array(out)
    if ns["report"] == "values":
        return u
    if ns["report"] == "mean":
        return float(u.mean())
    q = ns["queries"]
    return int(((q <= xs[0]) | (q >= xs[-1])).sum())
