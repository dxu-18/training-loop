"""4.06 Pipelines & ColumnTransformer."""

import numpy as np

from . import brute


@brute
def onehot_column_layout(ns):
    from sklearn.preprocessing import OneHotEncoder
    X, X_new, mode = ns["X"], ns["X_new"], ns["mode"]
    if mode == "ignore":
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    elif mode == "min_frequency":
        enc = OneHotEncoder(min_frequency=ns["m"], handle_unknown="infrequent_if_exist", sparse_output=False)
    else:
        enc = OneHotEncoder(max_categories=ns["m"], handle_unknown="infrequent_if_exist", sparse_output=False)
    out = enc.fit(X).transform(X_new)
    names = enc.get_feature_names_out()
    starts, k = [], 0
    for j in range(X.shape[1]):
        starts.append(k)
        k += sum(1 for nm in names if nm.split("_", 1)[0] == f"x{j}")
    starts.append(k)
    res = np.full(X_new.shape, -1)
    for i in range(len(X_new)):
        for j in range(X.shape[1]):
            block = out[i, starts[j]:starts[j + 1]]
            if block.any():
                res[i, j] = starts[j] + int(block.argmax())
    return res


@brute
def oof_impute_scale(ns):
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import KFold, PredefinedSplit
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    x, K = ns["x"].reshape(-1, 1), ns["K"]
    splits = PredefinedSplit(ns["fold"]).split() if "fold" in ns else KFold(n_splits=K).split(x)
    scales, z = [], np.zeros(len(x))
    tested = np.zeros(len(x), dtype=bool)
    for tr, te in splits:
        pipe = make_pipeline(SimpleImputer(strategy="mean"), StandardScaler()).fit(x[tr])
        scales.append(pipe[-1].scale_[0])
        z[te] = pipe.transform(x[te]).ravel()
        tested[te] = True
    return np.array(scales) if ns["report"] == "scale" else z[tested]


@brute
def causal_imputation(ns):
    from sklearn.impute import SimpleImputer
    x = ns["x"].copy()
    out = x.copy()
    for i in range(len(x)):
        if np.isnan(x[i]):
            past = x[:i][~np.isnan(x[:i])]
            if len(past) == 0:
                out[i] = ns["fallback"]
            else:
                out[i] = SimpleImputer(strategy=ns["strategy"]).fit(past.reshape(-1, 1)).statistics_[0]
    return out
