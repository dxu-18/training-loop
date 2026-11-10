"""4.17 Model inspection & built-in plots."""

import numpy as np

from . import brute


@brute
def tree_partial_dependence(ns):
    left, right, feature, thr = ns["left"], ns["right"], ns["feature"], ns["threshold"]
    value, weight, f = ns["value"], ns["weight"], ns["f"]

    def walk(node, w, v):                       # the recursion method, one grid value at a time
        if left[node] == -1:
            return w * value[node]
        if feature[node] == f:
            return walk(left[node] if v <= thr[node] else right[node], w, v)
        share = weight[left[node]] / weight[node]
        return walk(left[node], w * share, v) + walk(right[node], w * (1 - share), v)

    pd_ = np.array([walk(0, 1.0, v) for v in ns["grid"]])
    return {"pd": pd_, "range": float(pd_.max() - pd_.min()), "mean": float(pd_.mean())}[ns["report"]]


@brute
def knn_boundary_on_a_line(ns):
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=ns["k"]).fit(ns["x"].reshape(-1, 1), ns["y"])
    lab = knn.predict(ns["grid"].reshape(-1, 1)).astype(int)
    return {"labels": lab, "changes": int((lab[1:] != lab[:-1]).sum()), "share": float(lab.mean())}[ns["report"]]


@brute
def knn_validation_curve(ns):
    from sklearn.model_selection import LeaveOneOut, cross_val_predict
    from sklearn.neighbors import KNeighborsRegressor
    X, y = ns["x"].reshape(-1, 1), ns["y"]
    mse = np.array([((y - cross_val_predict(KNeighborsRegressor(n_neighbors=k), X, y, cv=LeaveOneOut())) ** 2).mean()
                    for k in range(1, ns["K"] + 1)])
    return {"curve": mse, "best_k": int(np.argmin(mse)) + 1, "best_mse": float(mse.min())}[ns["report"]]
