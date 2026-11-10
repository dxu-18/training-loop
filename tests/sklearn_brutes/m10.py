"""4.10 Trees & ensembles."""

import numpy as np

from . import brute


@brute
def best_stump_split(ns):
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    X = ns["x"].reshape(-1, 1)
    model = DecisionTreeClassifier if ns["report"] == "gini" else DecisionTreeRegressor
    t = model(max_depth=1, random_state=0).fit(X, ns["y"]).tree_
    if ns["report"] == "threshold":
        return float(t.threshold[0])
    l, r = t.children_left[0], t.children_right[0]
    w = t.n_node_samples
    return float((w[l] * t.impurity[l] + w[r] * t.impurity[r]) / w[0])


@brute
def tree_apply_batch(ns):
    left, right, feature, threshold = ns["left"], ns["right"], ns["feature"], ns["threshold"]
    leaf, depth = [], []
    for row in ns["X"]:
        i, dep = 0, 0
        while left[i] != -1:
            i = left[i] if row[feature[i]] <= threshold[i] else right[i]
            dep += 1
        leaf.append(i)
        depth.append(dep)
    leaf = np.array(leaf)
    return {"leaf": leaf, "value": ns["value"][leaf], "depth": np.array(depth)}[ns["report"]]


@brute
def bagging_oob_votes(ns):
    boot, pred, y, K = ns["boot"], ns["pred"], ns["y"], ns["K"]
    T, n = boot.shape
    bags = [set(b.tolist()) for b in boot]
    labels, counts = [], []
    for i in range(n):
        votes = [0] * K
        c = 0
        for t in range(T):
            if i not in bags[t]:
                votes[pred[t, i]] += 1
                c += 1
        counts.append(c)
        labels.append(votes.index(max(votes)) if c else -1)
    labels, counts = np.array(labels), np.array(counts)
    if ns["report"] == "accuracy":
        ok = counts > 0
        return float(np.mean(labels[ok] == y[ok]))
    return counts if ns["report"] == "counts" else labels


@brute
def boosting_stumps_1d(ns):
    from sklearn.ensemble import GradientBoostingRegressor
    X, y = ns["x"].reshape(-1, 1), ns["y"]
    gb = GradientBoostingRegressor(n_estimators=ns["T"], learning_rate=ns["lr"], max_depth=1, random_state=0).fit(X, y)
    if ns["report"] == "pred":
        return gb.predict(X)
    if ns["report"] == "mse":
        return np.array([((y - p) ** 2).mean() for p in gb.staged_predict(X)])
    return np.array([e[0].tree_.threshold[0] for e in gb.estimators_])


@brute
def weakest_link_alphas(ns):
    left, right, risk = ns["left"], ns["right"], ns["risk"]

    def leaves(t):
        stack, out = [t], []
        while stack:
            u = stack.pop()
            if left[u] == -1:
                out.append(u)
            else:
                stack += [left[u], right[u]]
        return out
    inner = [t for t in range(len(left)) if left[t] != -1]
    g = []
    for t in inner:
        lv = leaves(t)
        g.append((risk[t] - sum(risk[u] for u in lv)) / (len(lv) - 1))
    if ns["report"] == "all":
        return np.array(g)
    if ns["report"] == "weakest":
        j = min(range(len(g)), key=lambda k: (g[k], inner[k]))
        return (inner[j], float(g[j]))
    return sum(v <= ns["alpha"] for v in g)


@brute
def histogram_split_gain(ns):
    bins, grad, hess, lam, ml = ns["bins"], ns["grad"], ns["hess"], ns["lam"], ns["min_leaf"]
    n, F, B = ns["n"], ns["F"], ns["B"]
    G, H = grad.sum(), hess.sum()
    best = []                                  # (gain, feature, bin) per feature
    for f in range(F):
        cur = (-np.inf, f, -1)
        for b in range(B - 1):
            left = bins[:, f] <= b
            if left.sum() < ml or n - left.sum() < ml:
                continue
            gl, hl = grad[left].sum(), hess[left].sum()
            gain = gl ** 2 / (hl + lam) + (G - gl) ** 2 / (H - hl + lam) - G ** 2 / (H + lam)
            if gain > cur[0]:
                cur = (gain, f, b)
        best.append(cur)
    if ns["report"] == "per_feature":
        return np.array([c[0] for c in best])
    top = max(best, key=lambda c: (c[0], -c[1]))
    return (top[1], top[2], float(top[0]))
