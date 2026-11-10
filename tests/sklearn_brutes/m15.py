"""4.15 Hyperparameter search."""

import numpy as np

from . import brute


@brute
def parameter_grid_lookup(ns):
    from sklearn.model_selection import ParameterGrid
    grid, out = ParameterGrid(ns["grids"]), []
    sizes = [len(ParameterGrid(g)) for g in ns["grids"]]
    for i in ns["idx"]:
        cand = grid[int(i)]
        if ns["report"] == "subgrid":
            out.append(int(np.searchsorted(np.cumsum(sizes), int(i), side="right")))
        elif ns["report"] == "value":
            out.append(float(cand.get(ns["key"], np.nan)))
        else:
            out.append(float(sum(cand.values())))
    return np.array(out, dtype=int if ns["report"] == "subgrid" else float)


@brute
def ridge_leave_one_out(ns):
    from sklearn.linear_model import Ridge
    X, y, mse = ns["X"], ns["y"], []
    for a in ns["alphas"]:
        errs = []
        for i in range(len(X)):
            keep = np.arange(len(X)) != i
            reg = Ridge(alpha=a, fit_intercept=False).fit(X[keep], y[keep])
            errs.append((y[i] - reg.predict(X[i:i + 1])[0]) ** 2)
        mse.append(np.mean(errs))
    mse = np.array(mse)
    return {"loo_mse": mse, "best_alpha": float(ns["alphas"][int(np.argmin(mse))]),
            "best_rmse": float(np.sqrt(mse.min()))}[ns["report"]]


@brute
def successive_halving_rounds(ns):
    a, b, f, r = ns["a"], ns["b"], ns["factor"], ns["r0"]
    alive = list(range(len(a)))
    rounds, budget = [0] * len(a), 0
    while len(alive) > 1:
        for i in alive:
            rounds[i] += 1
        budget += len(alive) * r
        alive = sorted(alive, key=lambda i: (-(a[i] - b[i] / r), i))[:(len(alive) + f - 1) // f]
        r *= f
    return {"winner": alive[0], "budget": budget, "rounds": np.array(rounds)}[ns["report"]]
