"""4.15 Hyperparameter search: GridSearchCV, RandomizedSearchCV, successive halving, cv_results_, scoring (several
metrics, custom scorers), refit, nested CV."""

import math

from ._common import P, catalog, challenge, lit, seeded, synth_clf, synth_reg, table, template

M = "4.15"
GS = "from sklearn.model_selection import GridSearchCV\n"
RS = "from sklearn.model_selection import RandomizedSearchCV\nfrom scipy.stats import loguniform\n"
HALVING = ("from sklearn.experimental import enable_halving_search_cv  # noqa: F401\n"
           "from sklearn.model_selection import HalvingGridSearchCV\n")
PIPE_SETUP = ("from sklearn.pipeline import Pipeline\nfrom sklearn.preprocessing import StandardScaler\n"
              "from sklearn.svm import SVC\npipe = Pipeline([('scale', StandardScaler()), ('clf', SVC())])")
PIPE_DESC = "`pipe = Pipeline([('scale', StandardScaler()), ('clf', SVC())])`"
LR_PIPE = ("from sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\n"
           "from sklearn.linear_model import LogisticRegression\n"
           "lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))")


def _data(rng, n=(150, 260), cancer: float = 0.4):
    """Binary data: a seeded subset of breast cancer, or make_classification."""
    if rng.random() < cancer:
        m = rng.randint(*n)
        return (f"from sklearn.datasets import load_breast_cancer\nX, y = load_breast_cancer(return_X_y=True)\n"
                f"keep = np.random.default_rng({seeded(rng)}).permutation(len(X))[:{m}]\nX, y = X[keep], y[keep]",
                f"`X` holds {m} breast-tumour samples (30 cell measurements on very different scales) and `y` the "
                f"diagnosis (0 malignant, 1 benign)")
    d = synth_clf(rng, n=n, d=(4, 10), classes=2)
    return d.setup, d.desc


def _cs(rng, k=3):
    return sorted(rng.sample([0.01, 0.1, 1.0, 10.0, 100.0], k))


# ================================================================= lite

@template(M, "lite")
def grid_search_basics(rng):
    setup, desc = _data(rng)
    cs, cv = _cs(rng), rng.choice([3, 5])
    kernels = ["linear", "rbf"]
    g = f"GridSearchCV(pipe, param_grid={{'clf__C': {cs}}}, cv={cv})"
    sol = f"{GS}grid = {g}.fit(X, y)\n"
    items = [
        ("best_params_", f"Search `C` over `{cs}` with `{g}`, fit it on `X, y` and return `best_params_`.",
         sol + "result = grid.best_params_", {"require": ["GridSearchCV", "best_params_"]}),
        ("best_score_", f"Fit `{g}` on `X, y` and return `best_score_` (the best mean cross-validated accuracy).",
         sol + "result = grid.best_score_"),
        ("predict", f"Fit `{g}` on `X, y`. With the default `refit=True` the best model is retrained on all the data, "
                    f"so the search itself predicts: return `grid.predict(X[:5])`.", sol + "result = grid.predict(X[:5])",
         {"require": ["GridSearchCV", "predict"]}),
        ("cv_results_", f"Fit `{g}` on `X, y` and return the mean test score of every candidate "
                        f"(`cv_results_['mean_test_score']`).", sol + "result = grid.cv_results_['mean_test_score']"),
        ("cv_results_", f"Fit `{g}` on `X, y` and return the candidates' ranks (`cv_results_['rank_test_score']`; 1 = "
                        f"best).", sol + "result = grid.cv_results_['rank_test_score']"),
        ("best_index_", f"Fit `{g}` on `X, y` and return `best_index_`: the position of the winner in `cv_results_`.",
         sol + "result = int(grid.best_index_)"),
        ("best_estimator_", f"Fit `{g}` on `X, y` and return the `C` of the refitted best pipeline, read from "
                            f"`best_estimator_` (`.named_steps['clf'].C`).",
         sol + "result = grid.best_estimator_.named_steps['clf'].C"),
        ("n_splits_", f"Fit `{g}` on `X, y` and return how many folds it used (`n_splits_`).",
         sol + "result = grid.n_splits_"),
        ("param_grid", f"Search two hyperparameters at once: `param_grid={{'clf__C': {cs}, 'clf__kernel': "
                       f"{kernels}}}` with `cv={cv}`. Fit on `X, y` and return `best_params_`.",
         f"{GS}grid = GridSearchCV(pipe, param_grid={{'clf__C': {cs}, 'clf__kernel': {kernels}}}, cv={cv}).fit(X, y)\n"
         "result = grid.best_params_", {"require": ["GridSearchCV", "param_grid"]}),
        ("cv_results_", f"Search `param_grid={{'clf__C': {cs}, 'clf__kernel': {kernels}}}` with `cv={cv}` and return the "
                        f"number of candidates it tried (`len(cv_results_['params'])`).",
         f"{GS}grid = GridSearchCV(pipe, param_grid={{'clf__C': {cs}, 'clf__kernel': {kernels}}}, cv={cv}).fit(X, y)\n"
         "result = len(grid.cv_results_['params'])"),
    ]
    return catalog(rng, items, setup + "\n" + PIPE_SETUP, f"{desc}; {PIPE_DESC}. Pipeline parameters are named "
                                                          f"`<step>__<param>`.", tol=1e-9)


@template(M, "lite")
def random_search_basics(rng):
    setup, desc = _data(rng, cancer=0.3)
    n_iter, r = rng.randint(5, 10), rng.randint(0, 99)
    lo, hi = rng.choice([(1e-3, 1e2), (1e-2, 1e3), (1e-4, 1e1)])
    dist = f"{{'logisticregression__C': loguniform({lo}, {hi})}}"
    srch = f"RandomizedSearchCV(lr, {dist}, n_iter={n_iter}, cv=3, random_state={r})"
    sol = f"{RS}rand = {srch}.fit(X, y)\n"
    items = [
        ("RandomizedSearchCV", f"Draw `C` on a log scale: fit `{srch}` on `X, y` and return the best `C` "
                               f"(`best_params_['logisticregression__C']`).",
         sol + "result = rand.best_params_['logisticregression__C']", {"require": ["RandomizedSearchCV", "loguniform"]}),
        ("n_iter", f"Fit `{srch}` on `X, y` and return how many candidates it evaluated (`len(cv_results_['params'])`, "
                   f"which is `n_iter`).", sol + "result = len(rand.cv_results_['params'])",
         {"require": ["RandomizedSearchCV", "n_iter"]}),
        ("cv_results_", f"Fit `{srch}` on `X, y` and return the list of `C` values it tried, in order (from "
                        f"`cv_results_['params']`).",
         sol + "result = [p['logisticregression__C'] for p in rand.cv_results_['params']]"),
        ("best_score_", f"Fit `{srch}` on `X, y` and return `best_score_`.", sol + "result = rand.best_score_"),
        ("scoring", f"Fit `{srch[:-1]}, scoring='roc_auc')` on `X, y` and return `best_score_` (the best mean ROC AUC).",
         f"{RS}rand = {srch[:-1]}, scoring='roc_auc').fit(X, y)\nresult = rand.best_score_",
         {"require": ["RandomizedSearchCV", "scoring"]}),
        ("loguniform", f"Sample directly from the distribution: return `loguniform({lo}, {hi}).rvs(size=6, "
                       f"random_state={r})` (`scipy.stats.loguniform`). Every decade is equally likely.",
         f"from scipy.stats import loguniform\nresult = loguniform({lo}, {hi}).rvs(size=6, random_state={r})"),
    ]
    return catalog(rng, items, setup + "\n" + LR_PIPE, f"{desc}; `lr = make_pipeline(StandardScaler(), "
                                                       f"LogisticRegression(max_iter=1000))` (its step is named "
                                                       f"`logisticregression`).", tol=1e-9)


@template(M, "lite")
def scorer_lookup(rng):
    real = ["neg_root_mean_squared_error", "neg_mean_absolute_error", "roc_auc", "balanced_accuracy", "neg_log_loss",
            "f1_macro", "r2", "average_precision"]
    fake = ["root_mean_squared_error", "mean_absolute_error", "rmse", "auc", "logloss"]
    name = rng.choice(real + fake)
    word = rng.choice(["f1", "recall", "precision", "error"])
    b = rng.choice([0.5, 2, 3])
    setup, desc = _data(rng, n=(120, 200), cancer=0.0)
    items = [
        ("get_scorer_names", f"Is {name!r} a valid `scoring` string? Return `{name!r} in get_scorer_names()` "
                             f"(`sklearn.metrics`).", f"from sklearn.metrics import get_scorer_names\nresult = {name!r} in get_scorer_names()"),
        ("get_scorer_names", "Error metrics are negated so that higher is always better. Return how many scorer names "
                             "start with `'neg_'`.",
         "from sklearn.metrics import get_scorer_names\nresult = sum(n.startswith('neg_') for n in get_scorer_names())"),
        ("get_scorer_names", f"Return the sorted list of scorer names that contain {word!r}.",
         f"from sklearn.metrics import get_scorer_names\nresult = sorted(n for n in get_scorer_names() if {word!r} in n)"),
        ("make_scorer", f"Turn F-beta with `beta={b}` into a scorer (`make_scorer(fbeta_score, beta={b})`) and use it "
                        f"in `cross_val_score(LogisticRegression(max_iter=1000), X, y, cv=3, scoring=...)`. Return the "
                        f"mean.",
         "from sklearn.metrics import make_scorer, fbeta_score\nfrom sklearn.model_selection import cross_val_score\n"
         f"from sklearn.linear_model import LogisticRegression\nf = make_scorer(fbeta_score, beta={b})\n"
         "result = cross_val_score(LogisticRegression(max_iter=1000), X, y, cv=3, scoring=f).mean()",
         {"require": ["make_scorer", "fbeta_score"]}),
        ("get_scorer", "A scorer is called as `scorer(estimator, X, y)`. Fit `LogisticRegression(max_iter=1000)` on "
                       "`X, y` and return `get_scorer('balanced_accuracy')(clf, X, y)` (`sklearn.metrics.get_scorer`).",
         "from sklearn.metrics import get_scorer\nfrom sklearn.linear_model import LogisticRegression\n"
         "clf = LogisticRegression(max_iter=1000).fit(X, y)\nresult = get_scorer('balanced_accuracy')(clf, X, y)"),
    ]
    return catalog(rng, items, setup, f"{desc}.", tol=1e-9)


# ================================================================= easy

@template(M, "easy")
def grid_search_classic(rng):
    setup, desc = _data(rng, cancer=0.25)
    cv = rng.choice([3, 4, 5])
    s = seeded(rng)
    models = [
        ("KNeighborsClassifier", "sklearn.neighbors", "KNeighborsClassifier()", "n_neighbors",
         sorted(rng.sample([1, 3, 5, 9, 15, 25], 4))),
        ("DecisionTreeClassifier", "sklearn.tree", f"DecisionTreeClassifier(random_state={s})", "max_depth",
         sorted(rng.sample([1, 2, 3, 4, 6, 8], 4))),
        ("LogisticRegression", "sklearn.linear_model", "LogisticRegression(max_iter=5000)", "C", _cs(rng, 4)),
        ("GaussianNB", "sklearn.naive_bayes", "GaussianNB()", "var_smoothing", [1e-9, 1e-6, 1e-3, 1e-1]),
    ]
    name, mod, ctor, param, vals = rng.choice(models)
    if "cancer" in setup and name == "LogisticRegression":
        setup += "\nX = (X - X.mean(axis=0)) / X.std(axis=0)   # standardized"
    want = rng.choice(["params", "score", "value"])
    g = f"GridSearchCV({ctor}, {{{param!r}: {vals}}}, cv={cv})"
    sol = f"{GS}from {mod} import {name}\ngrid = {g}.fit(X, y)\n"
    tail = {"params": ("`best_params_`", "result = grid.best_params_"),
            "score": ("`best_score_`", "result = grid.best_score_"),
            "value": (f"the best `{param}` together with its mean CV accuracy, as a tuple", f"result = (grid.best_params_[{param!r}], grid.best_score_)")}[want]
    return P(f"{desc}. Tune `{param}` of `{ctor}` (from `{mod}`) over `{vals}` with `GridSearchCV(..., cv={cv})`; fit "
             f"on `X, y` and return {tail[0]}.", setup, sol + tail[1], require=["GridSearchCV", name], tol=1e-9)


@template(M, "easy")
def cv_results_table(rng):
    setup, desc = _data(rng)
    cs = _cs(rng, 4)
    setup += "\n" + PIPE_SETUP + f"\n{GS.strip()}\ngrid = GridSearchCV(pipe, {{'clf__C': {cs}, 'clf__kernel': ['linear', 'rbf']}}, cv=3).fit(X, y)"
    cols = "['params', 'mean_test_score', 'std_test_score', 'rank_test_score']"
    task = rng.choice(["top", "sorted", "ties", "std", "kernel"])
    head = (f"{desc}. `grid = GridSearchCV(pipe, {{'clf__C': {cs}, 'clf__kernel': ['linear', 'rbf']}}, cv=3)` has been "
            f"fitted on them ({PIPE_DESC}). Put the results in a table: `pd.DataFrame(grid.cv_results_)[{cols}]`, "
            f"sorted by `rank_test_score`.")
    sol = f"df = pd.DataFrame(grid.cv_results_)[{cols}].sort_values('rank_test_score', kind='stable')\n"
    if task == "top":
        return P(f"{head} Return the `params` of the top 3 rows as a list of dicts.", setup,
                 sol + "result = list(df['params'].head(3))", require=["cv_results_", "sort_values"])
    if task == "sorted":
        return P(f"{head} Return its `mean_test_score` column as a NumPy array (best first).", setup,
                 sol + "result = df['mean_test_score'].to_numpy()", require=["cv_results_", "sort_values"], tol=1e-9)
    if task == "ties":
        return P(f"{head} Several candidates can share rank 1. Return how many rows have `rank_test_score == 1`.",
                 setup, sol + "result = int((df['rank_test_score'] == 1).sum())", require=["cv_results_"])
    if task == "std":
        return P(f"{head} Return `(mean, std)` of the best row: how noisy is the winning score across folds?", setup,
                 sol + "result = (df['mean_test_score'].iloc[0], df['std_test_score'].iloc[0])",
                 require=["cv_results_"], tol=1e-9)
    return P(f"{head} Return the best mean test score per kernel as a dict `{{'linear': ..., 'rbf': ...}}` "
             f"(`cv_results_['param_clf__kernel']` holds each candidate's kernel).", setup,
             "df = pd.DataFrame(grid.cv_results_)\n"
             "result = df.groupby('param_clf__kernel')['mean_test_score'].max().to_dict()", require=["cv_results_"],
             tol=1e-9)


@template(M, "easy")
def refit_behaviour(rng):
    setup, desc = _data(rng, cancer=0.0)
    cs = _cs(rng)
    s = seeded(rng)
    task = rng.choice(["same", "norefit", "allrows"])
    g = f"GridSearchCV(LogisticRegression(max_iter=1000), {{'C': {cs}}}, cv=3"
    imp = f"{GS}from sklearn.linear_model import LogisticRegression\n"
    if task == "same":
        return P(f"{desc}. Fit `grid = {g})` on `X, y`. The search predicts with its refitted `best_estimator_`: "
                 f"return `(grid.predict(X[:6]), grid.best_estimator_.predict(X[:6]))` (identical).", setup,
                 imp + f"grid = {g}).fit(X, y)\nresult = (grid.predict(X[:6]), grid.best_estimator_.predict(X[:6]))",
                 require=["best_estimator_", "predict"])
    if task == "norefit":
        return P(f"{desc}. With `refit=False` the search only scores the candidates and keeps no model. Fit "
                 f"`{g}, refit=False)` on `X, y` and return `(hasattr(grid, 'best_estimator_'), grid.best_params_)`.",
                 setup, imp + f"grid = {g}, refit=False).fit(X, y)\nresult = (hasattr(grid, 'best_estimator_'), grid.best_params_)",
                 require=["refit"])
    return P(f"{desc}. After the search, `best_estimator_` is retrained on **all** rows passed to `fit`. Fit `grid = "
             f"{g})` on `X, y`, then fit a fresh `LogisticRegression(max_iter=1000, C=best_C)` on `X, y` yourself and "
             f"return `(grid.best_estimator_.coef_, your_model.coef_)` (equal).", setup,
             imp + f"grid = {g}).fit(X, y)\nmine = LogisticRegression(max_iter=1000, C=grid.best_params_['C']).fit(X, y)\n"
                   "result = (grid.best_estimator_.coef_, mine.coef_)", require=["best_params_"], tol=1e-6)


@template(M, "easy")
def list_of_grids(rng):
    setup, desc = _data(rng)
    c1, c2 = _cs(rng, 2), _cs(rng, 2)
    gam = sorted(rng.sample([1e-3, 1e-2, 1e-1, 1.0], 2))
    grids = f"[{{'clf__kernel': ['linear'], 'clf__C': {c1}}}, {{'clf__kernel': ['rbf'], 'clf__C': {c2}, 'clf__gamma': {gam}}}]"
    head = (f"{desc}; {PIPE_DESC}. A list of grids searches separate sub-spaces (`gamma` only matters for the RBF "
            f"kernel): `grids = {grids}`.")
    task = rng.choice(["count", "best", "kernel"])
    sol = f"{GS}grid = GridSearchCV(pipe, grids, cv=3).fit(X, y)\n"
    setup2 = setup + "\n" + PIPE_SETUP + f"\ngrids = {grids}"
    if task == "count":
        return P(f"{head} Fit `GridSearchCV(pipe, grids, cv=3)` on `X, y` and return the number of candidates "
                 f"(`len(cv_results_['params'])`): the sum of the sub-grid sizes, not their product.", setup2,
                 sol + "result = len(grid.cv_results_['params'])", require=["GridSearchCV", "cv_results_"])
    if task == "best":
        return P(f"{head} Fit `GridSearchCV(pipe, grids, cv=3)` on `X, y` and return `best_params_`.", setup2,
                 sol + "result = grid.best_params_", require=["GridSearchCV", "best_params_"])
    return P(f"{head} Fit `GridSearchCV(pipe, grids, cv=3)` on `X, y` and return the best mean test score of each "
             f"kernel as a tuple `(linear_best, rbf_best)`.", setup2,
             sol + "m = grid.cv_results_['mean_test_score']\nk = np.array([p['clf__kernel'] for p in grid.cv_results_['params']])\n"
                   "result = (m[k == 'linear'].max(), m[k == 'rbf'].max())", require=["cv_results_"], tol=1e-9)


@template(M, "easy")
def scoring_changes_winner(rng):
    setup, desc = _data(rng, cancer=0.0)
    if rng.random() < 0.6:
        setup = synth_clf(rng, n=(200, 320), d=(4, 8), classes=2, imbalance=rng.choice([0.1, 0.15, 0.2])).setup
        desc = "`X`, `y` come from `make_classification` with a rare class 1"
    metric = rng.choice(["f1", "recall", "balanced_accuracy", "roc_auc", "average_precision"])
    cw = "[None, 'balanced']"
    g = f"GridSearchCV(LogisticRegression(max_iter=1000), {{'class_weight': {cw}, 'C': [0.1, 1.0]}}, cv=3"
    imp = f"{GS}from sklearn.linear_model import LogisticRegression\n"
    if rng.random() < 0.5:
        return P(f"{desc}. The metric decides the winner. Fit `{g}, scoring={metric!r})` on `X, y` and return "
                 f"`(best_params_, best_score_)`.", setup,
                 imp + f"grid = {g}, scoring={metric!r}).fit(X, y)\nresult = (grid.best_params_, grid.best_score_)",
                 require=["scoring"], tol=1e-9)
    return P(f"{desc}. Fit `{g})` twice: with the default accuracy and with `scoring={metric!r}`. Return the two "
             f"`best_params_` as a tuple `(by_accuracy, by_{metric})`.", setup,
             imp + f"a = {g}).fit(X, y)\nb = {g}, scoring={metric!r}).fit(X, y)\nresult = (a.best_params_, b.best_params_)",
             require=["scoring"])


@template(M, "easy")
def search_space_size(rng):
    keys = rng.sample(["C", "gamma", "kernel", "degree", "tol", "class_weight"], rng.randint(2, 4))
    vals = {"C": [0.1, 1, 10], "gamma": [0.01, 0.1], "kernel": ["'rbf'", "'poly'"], "degree": [2, 3, 4],
            "tol": [1e-4, 1e-3], "class_weight": ["None", "'balanced'"]}
    grid = "{" + ", ".join(f"{k!r}: [{', '.join(str(v) for v in vals[k])}]" for k in keys) + "}"
    cv = rng.choice([3, 5, 10])
    n_iter, r = rng.randint(3, 6), rng.randint(0, 50)
    task = rng.choice(["len", "fits", "sampler"])
    setup = f"grid = {grid}"
    if task == "len":
        return P(f"`grid = {grid}`. Return the number of candidates `GridSearchCV` would try: `len(ParameterGrid(grid))` "
                 f"(`sklearn.model_selection`).", setup,
                 "from sklearn.model_selection import ParameterGrid\nresult = len(ParameterGrid(grid))",
                 require=["ParameterGrid"])
    if task == "fits":
        return P(f"`grid = {grid}`. How many times would `GridSearchCV(SVC(), grid, cv={cv})` call `fit`? Every candidate "
                 f"is fitted once per fold, plus one final refit of the winner. Compute it with `ParameterGrid` (no "
                 f"fitting needed).", setup,
                 f"from sklearn.model_selection import ParameterGrid\nresult = len(ParameterGrid(grid)) * {cv} + 1",
                 require=["ParameterGrid"])
    return P(f"`RandomizedSearchCV` draws its candidates with `ParameterSampler`. `space = {{'C': loguniform(0.01, "
             f"100)}}` (`scipy.stats`). Return the list of `C` values in `list(ParameterSampler(space, n_iter={n_iter}, "
             f"random_state={r}))`.", setup,
             f"from sklearn.model_selection import ParameterSampler\nfrom scipy.stats import loguniform\n"
             f"result = [p['C'] for p in ParameterSampler({{'C': loguniform(0.01, 100)}}, n_iter={n_iter}, random_state={r})]",
             require=["ParameterSampler", "loguniform"], tol=1e-9)


@template(M, "easy")
def regression_search_scoring(rng):
    d = synth_reg(rng, n=(120, 260)) if rng.random() < 0.6 else None
    setup = d.setup if d else "from sklearn.datasets import load_diabetes\nX, y = load_diabetes(return_X_y=True)"
    desc = d.desc if d else "`X` holds 442 diabetes patients (10 standardized measurements) and `y` disease progression"
    alphas = sorted(rng.sample([0.01, 0.1, 1.0, 10.0, 100.0], 4))
    task = rng.choice(["rmse", "mae", "r2"])
    g = f"GridSearchCV(Ridge(), {{'alpha': {alphas}}}, cv=5"
    imp = f"{GS}from sklearn.linear_model import Ridge\n"
    if task == "rmse":
        return P(f"{desc}. Tune `alpha` with `{g}, scoring='neg_root_mean_squared_error')`. Scores are negated errors "
                 f"(higher is better), so return `(best_alpha, best_rmse)` with a **positive** RMSE.", setup,
                 imp + f"grid = {g}, scoring='neg_root_mean_squared_error').fit(X, y)\n"
                       "result = (grid.best_params_['alpha'], -grid.best_score_)", require=["GridSearchCV"], tol=1e-9)
    if task == "mae":
        return P(f"{desc}. Tune `alpha` with `{g}, scoring='neg_mean_absolute_error')` and return the cross-validated "
                 f"MAE of every candidate as a positive array (from `cv_results_['mean_test_score']`).", setup,
                 imp + f"grid = {g}, scoring='neg_mean_absolute_error').fit(X, y)\nresult = -grid.cv_results_['mean_test_score']",
                 require=["cv_results_"], tol=1e-9)
    return P(f"{desc}. A regressor's default score is R². Fit `{g})` on `X, y` and return `best_score_` (a mean "
             f"cross-validated R²).", setup, imp + f"result = {g}).fit(X, y).best_score_", require=["best_score_"],
             tol=1e-9)


@template(M, "easy")
def pipeline_param_names(rng):
    setup, desc = _data(rng, cancer=0.5)
    cs = _cs(rng)
    task = rng.choice(["keys", "make", "wrong"])
    if task == "keys":
        return P(f"{desc}; {PIPE_DESC}. A search addresses pipeline parameters as `<step>__<param>`. Return the sorted "
                 f"list of `pipe.get_params()` keys that start with `'clf__'` and end in `'C'` or `'gamma'`.",
                 setup + "\n" + PIPE_SETUP,
                 "result = sorted(k for k in pipe.get_params() if k.startswith('clf__') and k.endswith(('C', 'gamma')))",
                 require=["get_params"])
    if task == "make":
        return P(f"{desc}. `make_pipeline` names steps after their lowercased class. Tune `C` over `{cs}` of "
                 f"`make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))` with `cv=3` and return "
                 f"`best_params_` (its key is `'logisticregression__C'`).", setup,
                 f"{GS}from sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\n"
                 "from sklearn.linear_model import LogisticRegression\n"
                 "lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))\n"
                 f"result = GridSearchCV(lr, {{'logisticregression__C': {cs}}}, cv=3).fit(X, y).best_params_",
                 require=["make_pipeline", "GridSearchCV"])
    return P(f"{desc}; {PIPE_DESC}. `GridSearchCV(pipe, {{'C': {cs}}}, cv=3).fit(X, y)` raises `ValueError` (the "
             f"pipeline has no parameter `C`). Fix the grid and return `best_score_`.", setup + "\n" + PIPE_SETUP,
             f"{GS}result = GridSearchCV(pipe, {{'clf__C': {cs}}}, cv=3).fit(X, y).best_score_",
             require=["GridSearchCV"], tol=1e-9)


@template(M, "easy")
def halving_basics(rng):
    setup, desc = _data(rng, n=(180, 300), cancer=0.3)
    cs = sorted(rng.sample([0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0], rng.randint(4, 6)))
    f = rng.choice([2, 3])
    h = f"HalvingGridSearchCV(pipe, {{'clf__C': {cs}}}, cv=3, factor={f}, random_state=0)"
    task = rng.choice(["best", "iters", "cands", "res"])
    tail = {"best": ("`best_params_`", "result = h.best_params_"),
            "iters": ("how many rounds it ran (`n_iterations_`)", "result = int(h.n_iterations_)"),
            "cands": ("the number of candidates in each round (`n_candidates_`, a list)", "result = list(h.n_candidates_)"),
            "res": ("the number of training samples used in each round (`n_resources_`)", "result = list(h.n_resources_)")}[task]
    return P(f"{desc}; {PIPE_DESC}. Successive halving scores all candidates on a few samples, keeps the best "
             f"1/{f}, and gives the survivors more samples. It is experimental: enable it first with `from "
             f"sklearn.experimental import enable_halving_search_cv`. Fit `{h}` on `X, y` and return {tail[0]}.",
             setup + "\n" + PIPE_SETUP, HALVING + f"h = {h}.fit(X, y)\n" + tail[1],
             require=["HalvingGridSearchCV", "enable_halving_search_cv"])


# ================================================================= medium

@template(M, "medium")
def forest_grid(rng):
    setup, desc = _data(rng, n=(120, 200), cancer=0.4)
    depths = rng.choice([[2, 4, None], [1, 3, 6], [2, 5]])
    ests = rng.choice([[10, 30], [5, 20], [10, 25]])
    r = rng.randint(0, 9)
    g = (f"GridSearchCV(RandomForestClassifier(random_state={r}), {{'max_depth': {depths}, 'n_estimators': {ests}}}, "
         f"cv=3)")
    sol = f"{GS}from sklearn.ensemble import RandomForestClassifier\ngrid = {g}.fit(X, y)\n"
    task = rng.choice(["params", "table", "gap", "depth"])
    head = f"{desc}. Grid-search a random forest: `{g}`, fitted on `X, y`."
    if task == "params":
        return P(f"{head} Return `(best_params_, best_score_)`.", setup,
                 sol + "result = (grid.best_params_, grid.best_score_)", require=["RandomForestClassifier", "GridSearchCV"],
                 tol=1e-9)
    if task == "table":
        return P(f"{head} Return the mean CV accuracy as a dict keyed by `(max_depth, n_estimators)` tuples.", setup,
                 sol + "res = grid.cv_results_\n"
                       "result = {(p['max_depth'], p['n_estimators']): s for p, s in zip(res['params'], res['mean_test_score'])}",
                 require=["cv_results_"], tol=1e-9)
    if task == "gap":
        return P(f"{head} How much does tuning matter here? Return the difference between the best and the worst mean "
                 f"test score.", setup,
                 sol + "m = grid.cv_results_['mean_test_score']\nresult = m.max() - m.min()", require=["cv_results_"],
                 tol=1e-9)
    return P(f"{head} For each `max_depth`, return its best mean test score over the `n_estimators` values, as a list "
             f"in the order of `{depths}`.", setup,
             sol + "res = grid.cv_results_\n"
                   f"result = [max(s for p, s in zip(res['params'], res['mean_test_score']) if p['max_depth'] == d) for d in {depths}]",
             require=["cv_results_"], tol=1e-9)


@template(M, "medium")
def log_scale_random_search(rng):
    setup, desc = _data(rng, n=(150, 250), cancer=0.4)
    n_iter, r = rng.choice([10, 12, 15]), rng.randint(0, 99)
    use_svc = rng.random() < 0.5
    if use_svc:
        est, key, extra = "pipe", "clf__C", PIPE_SETUP
        space = f"{{'clf__C': loguniform(1e-2, 1e3), 'clf__gamma': loguniform(1e-4, 1e0)}}"
        edesc = PIPE_DESC
    else:
        est, key, extra = "lr", "logisticregression__C", LR_PIPE
        space = "{'logisticregression__C': loguniform(1e-3, 1e3)}"
        edesc = "`lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))`"
    srch = f"RandomizedSearchCV({est}, {space}, n_iter={n_iter}, cv=3, random_state={r})"
    sol = f"{RS}rand = {srch}.fit(X, y)\n"
    task = rng.choice(["best", "below", "vs_grid", "range"])
    head = f"{desc}; {edesc}. Random-search on a log scale: `{srch}`, fitted on `X, y`."
    if task == "best":
        return P(f"{head} Return `(best_params_, best_score_)`.", setup + "\n" + extra,
                 sol + "result = (rand.best_params_, rand.best_score_)", require=["RandomizedSearchCV", "loguniform"],
                 tol=1e-9)
    if task == "below":
        return P(f"{head} A log-uniform prior spreads the draws over decades. Return how many of the sampled `C` "
                 f"values are below 1.", setup + "\n" + extra,
                 sol + f"result = sum(p[{key!r}] < 1 for p in rand.cv_results_['params'])",
                 require=["RandomizedSearchCV", "loguniform"])
    if task == "vs_grid":
        return P(f"{head} Also run a grid search over `C` in `[0.01, 1, 100]` (the same `cv=3`) and return "
                 f"`(random_best_score, grid_best_score)`.", setup + "\n" + extra,
                 sol + f"{GS}grid = GridSearchCV({est}, {{{key!r}: [0.01, 1, 100]}}, cv=3).fit(X, y)\n"
                       "result = (rand.best_score_, grid.best_score_)",
                 require=["RandomizedSearchCV", "GridSearchCV"], tol=1e-9)
    return P(f"{head} Return `(min_C, max_C)` of the sampled candidates and the `C` of the best one, as a 3-tuple.",
             setup + "\n" + extra,
             sol + f"cs = [p[{key!r}] for p in rand.cv_results_['params']]\nresult = (min(cs), max(cs), rand.best_params_[{key!r}])",
             require=["RandomizedSearchCV", "loguniform"], tol=1e-9)


@template(M, "medium")
def nested_cross_validation(rng):
    setup, desc = _data(rng, n=(150, 240), cancer=0.4)
    cs = _cs(rng)
    outer = rng.choice([3, 4, 5])
    inner = f"GridSearchCV(pipe, {{'clf__C': {cs}}}, cv=3)"
    task = rng.choice(["compare", "folds", "chosen"])
    setup2 = setup + "\n" + PIPE_SETUP
    if task == "compare":
        return P(f"{desc}; {PIPE_DESC}. `best_score_` is optimistic: it is the maximum of several noisy estimates. "
                 f"Return `(nested, plain)`: the mean of `cross_val_score({inner}, X, y, cv={outer})` (an honest "
                 f"estimate of the tuned model) and the `best_score_` of the same search fitted on all of `X, y`.",
                 setup2, f"{GS}from sklearn.model_selection import cross_val_score\n"
                         f"nested = cross_val_score({inner}, X, y, cv={outer}).mean()\n"
                         f"plain = {inner}.fit(X, y).best_score_\nresult = (nested, plain)",
                 require=["cross_val_score", "GridSearchCV"], tol=1e-9)
    if task == "folds":
        return P(f"{desc}; {PIPE_DESC}. Nested CV: return the {outer} outer-fold scores of `cross_val_score({inner}, X, "
                 f"y, cv={outer})`.", setup2,
                 f"{GS}from sklearn.model_selection import cross_val_score\nresult = cross_val_score({inner}, X, y, cv={outer})",
                 require=["cross_val_score", "GridSearchCV"], tol=1e-9)
    return P(f"{desc}; {PIPE_DESC}. Nested CV by hand shows that the tuned value can change between folds: loop over "
             f"`StratifiedKFold(n_splits={outer}).split(X, y)`, fit `{inner}` on each outer training part, and return "
             f"the list of chosen `clf__C` values.", setup2,
             f"{GS}from sklearn.model_selection import StratifiedKFold\nresult = []\n"
             f"for tr, te in StratifiedKFold(n_splits={outer}).split(X, y):\n"
             f"    result.append({inner}.fit(X[tr], y[tr]).best_params_['clf__C'])",
             require=["StratifiedKFold", "GridSearchCV"])


@template(M, "medium")
def multi_metric_search(rng):
    setup, desc = _data(rng, cancer=0.3)
    cs = _cs(rng)
    pair = rng.choice([("auc", "roc_auc", "f1", "f1"), ("acc", "accuracy", "bal", "balanced_accuracy"),
                       ("ap", "average_precision", "acc", "accuracy")])
    a, sa, b, sb = pair
    scoring = f"{{{a!r}: {sa!r}, {b!r}: {sb!r}}}"
    g = f"GridSearchCV(pipe, {{'clf__C': {cs}}}, scoring={scoring}, refit={a!r}, cv=3)"
    sol = f"{GS}grid = {g}.fit(X, y)\n"
    task = rng.choice(["best", "other", "keys", "disagree"])
    head = (f"{desc}; {PIPE_DESC}. Score several metrics at once and refit on one: `{g}`, fitted on `X, y`.")
    setup2 = setup + "\n" + PIPE_SETUP
    if task == "best":
        return P(f"{head} Return `(best_params_, best_score_)`; the best score is the {a!r} metric (the `refit` one).",
                 setup2, sol + "result = (grid.best_params_, grid.best_score_)", require=["refit", "scoring"], tol=1e-9)
    if task == "other":
        return P(f"{head} Return the mean {b!r} score of the chosen candidate (`cv_results_['mean_test_{b}']` at "
                 f"`best_index_`).", setup2, sol + f"result = grid.cv_results_['mean_test_{b}'][grid.best_index_]",
                 require=["cv_results_", "best_index_"], tol=1e-9)
    if task == "keys":
        return P(f"{head} Return the sorted list of `cv_results_` keys that start with `'mean_test_'`.", setup2,
                 sol + "result = sorted(k for k in grid.cv_results_ if k.startswith('mean_test_'))",
                 require=["cv_results_"])
    return P(f"{head} Would the other metric pick the same `C`? Return `(C_by_{a}, C_by_{b})`: the `clf__C` with the "
             f"highest `mean_test_{a}` and the one with the highest `mean_test_{b}` (first one on ties).", setup2,
             sol + "res = grid.cv_results_\n"
                   f"result = (res['params'][int(np.argmax(res['mean_test_{a}']))]['clf__C'], "
                   f"res['params'][int(np.argmax(res['mean_test_{b}']))]['clf__C'])", require=["cv_results_"])


@template(M, "medium")
def custom_scorer_search(rng):
    d = synth_clf(rng, n=(200, 320), d=(4, 8), classes=2, imbalance=rng.choice([0.15, 0.25]))
    b = rng.choice([2, 3])
    cost_fn, cost_fp = rng.choice([(5, 1), (10, 1), (3, 1)])
    cs = _cs(rng)
    imp = f"{GS}from sklearn.linear_model import LogisticRegression\n"
    est = "LogisticRegression(max_iter=1000)"
    task = rng.choice(["fbeta", "cost", "compare"])
    if task == "fbeta":
        return P(f"{d.desc} (class 1 is rare). Recall matters {b}× as much as precision: tune `C` over `{cs}` and "
                 f"`class_weight` over `[None, 'balanced']` of `{est}` with `cv=3` and `scoring=make_scorer(fbeta_score, "
                 f"beta={b})`. Return `(best_params_, best_score_)`.", d.setup,
                 imp + "from sklearn.metrics import make_scorer, fbeta_score\n"
                       f"grid = GridSearchCV({est}, {{'C': {cs}, 'class_weight': [None, 'balanced']}}, cv=3, "
                       f"scoring=make_scorer(fbeta_score, beta={b})).fit(X, y)\nresult = (grid.best_params_, grid.best_score_)",
                 require=["make_scorer", "fbeta_score"], tol=1e-9)
    if task == "cost":
        return P(f"{d.desc} (class 1 is rare). A missed positive costs {cost_fn}, a false alarm {cost_fp}. Write "
                 f"`def cost(y_true, y_pred)` returning the **average** cost per row, wrap it with "
                 f"`make_scorer(cost, greater_is_better=False)`, and tune `class_weight` over `[None, 'balanced', "
                 f"{{0: 1, 1: 5}}]` of `{est}` with `cv=3`. Return `(best_params_, best_score_)` (the score comes back "
                 f"negated).", d.setup,
                 imp + "from sklearn.metrics import make_scorer\n"
                       f"def cost(y_true, y_pred):\n    return ({cost_fn} * ((y_true == 1) & (y_pred == 0)).sum() + "
                       f"{cost_fp} * ((y_true == 0) & (y_pred == 1)).sum()) / len(y_true)\n"
                       f"grid = GridSearchCV({est}, {{'class_weight': [None, 'balanced', {{0: 1, 1: 5}}]}}, cv=3, "
                       "scoring=make_scorer(cost, greater_is_better=False)).fit(X, y)\n"
                       "result = (grid.best_params_, grid.best_score_)",
                 require=["make_scorer", "greater_is_better"], tol=1e-9)
    return P(f"{d.desc} (class 1 is rare). Tune `C` over `{cs}` of `{est}` with `cv=3` twice: with `scoring='f1'` and "
             f"with `scoring=make_scorer(fbeta_score, beta={b})`. Return both `best_score_`s as a tuple `(f1, f{b})`.",
             d.setup,
             imp + "from sklearn.metrics import make_scorer, fbeta_score\n"
                   f"a = GridSearchCV({est}, {{'C': {cs}}}, cv=3, scoring='f1').fit(X, y)\n"
                   f"b = GridSearchCV({est}, {{'C': {cs}}}, cv=3, scoring=make_scorer(fbeta_score, beta={b})).fit(X, y)\n"
                   "result = (a.best_score_, b.best_score_)", require=["make_scorer"], tol=1e-9)


@template(M, "medium")
def tuned_pipeline_on_table(rng):
    t = table(rng, kind="clf", numeric_only=True, n=(160, 280))
    s = seeded(rng)
    ks = sorted(rng.sample([1, 3, 5, 9, 15, 25], 4))
    cs = _cs(rng)
    setup = (t.setup + "\nfrom sklearn.model_selection import train_test_split\n"
             f"X_tr, X_te, y_tr, y_te = train_test_split(df, y, stratify=y, random_state={s})")
    sc = "from sklearn.pipeline import Pipeline\nfrom sklearn.preprocessing import StandardScaler\n"
    knn = f"{sc}from sklearn.neighbors import KNeighborsClassifier\npipe = Pipeline([('scale', StandardScaler()), ('knn', KNeighborsClassifier())])\n"
    items = [
        ("KNeighborsClassifier", f"Tune `n_neighbors` over `{ks}` of `Pipeline([('scale', StandardScaler()), ('knn', "
                                 f"KNeighborsClassifier())])` with `cv=5` on the training part and return `best_params_`.",
         f"{GS}{knn}result = GridSearchCV(pipe, {{'knn__n_neighbors': {ks}}}, cv=5).fit(X_tr, y_tr).best_params_"),
        ("class_weight", f"Tune `C` over `{cs}` and `class_weight` over `[None, 'balanced']` of "
                         f"`LogisticRegression(max_iter=1000)` (after a `StandardScaler`, as `make_pipeline`) with "
                         f"`scoring='balanced_accuracy'`, `cv=5`, on the training part. Return `best_params_`.",
         f"{GS}from sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\n"
         "from sklearn.linear_model import LogisticRegression\nlr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))\n"
         f"result = GridSearchCV(lr, {{'logisticregression__C': {cs}, 'logisticregression__class_weight': [None, 'balanced']}}, "
         "scoring='balanced_accuracy', cv=5).fit(X_tr, y_tr).best_params_", {"require": ["GridSearchCV", "scoring"]}),
        ("DecisionTreeClassifier", "Tune `max_depth` over `[2, 3, 5]` and `min_samples_leaf` over `[1, 5, 20]` of "
                                   "`DecisionTreeClassifier(random_state=0)` with `cv=5` on the training part. Return "
                                   "`(best_params_, test accuracy of the search)`.",
         f"{GS}from sklearn.tree import DecisionTreeClassifier\n"
         "grid = GridSearchCV(DecisionTreeClassifier(random_state=0), {'max_depth': [2, 3, 5], 'min_samples_leaf': [1, 5, 20]}, cv=5).fit(X_tr, y_tr)\n"
         "result = (grid.best_params_, grid.score(X_te, y_te))"),
        ("passthrough", f"Is scaling worth it? A grid can swap whole steps: search `{{'scale': [StandardScaler(), "
                        f"'passthrough'], 'knn__n_neighbors': {ks}}}` over `Pipeline([('scale', StandardScaler()), "
                        f"('knn', KNeighborsClassifier())])` with `cv=5` on the training part. Return `(scaled, k)`: "
                        f"`True` if the best candidate scales, and its `n_neighbors`.",
         f"{GS}{knn}grid = GridSearchCV(pipe, {{'scale': [StandardScaler(), 'passthrough'], 'knn__n_neighbors': {ks}}}, cv=5).fit(X_tr, y_tr)\n"
         "result = (grid.best_params_['scale'] != 'passthrough', grid.best_params_['knn__n_neighbors'])", {"require": ["GridSearchCV", "best_params_"]}),
        ("score", f"Does tuning beat the defaults on fresh data? Return `(default_test_acc, tuned_test_acc)`: a default "
                  f"`KNeighborsClassifier()` after scaling vs the same pipeline tuned over `n_neighbors` in `{ks}` "
                  f"(`cv=5`), both fitted on the training part and scored on the test part.",
         f"{GS}{knn}default = pipe.fit(X_tr, y_tr).score(X_te, y_te)\n"
         f"tuned = GridSearchCV(pipe, {{'knn__n_neighbors': {ks}}}, cv=5).fit(X_tr, y_tr).score(X_te, y_te)\n"
         "result = (default, tuned)"),
        ("rank_test_score", f"Tune `n_neighbors` over `{ks}` (scaled k-NN pipeline, `cv=5`, training part) and return "
                            f"the `n_neighbors` values ordered from best to worst rank (stable on ties).",
         f"{GS}{knn}res = GridSearchCV(pipe, {{'knn__n_neighbors': {ks}}}, cv=5).fit(X_tr, y_tr).cv_results_\n"
         "order = np.argsort(res['rank_test_score'], kind='stable')\nresult = [res['params'][i]['knn__n_neighbors'] for i in order]", {"require": ["cv_results_"]}),
        ("return_train_score", f"Spot overfitting: tune `n_neighbors` over `{ks}` (scaled k-NN pipeline, `cv=5`, "
                               f"training part) with `return_train_score=True` and return the train and test mean "
                               f"scores per candidate as two arrays `(train, test)`.",
         f"{GS}{knn}res = GridSearchCV(pipe, {{'knn__n_neighbors': {ks}}}, cv=5, return_train_score=True).fit(X_tr, y_tr).cv_results_\n"
         "result = (res['mean_train_score'], res['mean_test_score'])"),
        ("best_score_", f"Tune `n_neighbors` over `{ks}` (scaled k-NN pipeline, `cv=5`, training part). Return "
                        f"`(best_score_, test accuracy)`: the CV estimate next to the score on unseen data.",
         f"{GS}{knn}grid = GridSearchCV(pipe, {{'knn__n_neighbors': {ks}}}, cv=5).fit(X_tr, y_tr)\n"
         "result = (grid.best_score_, grid.score(X_te, y_te))"),
    ]
    return catalog(rng, items, setup, f"{t.desc}. It was split with `train_test_split(df, y, stratify=y, "
                                      f"random_state={s})` into `X_tr, X_te, y_tr, y_te`.", tol=1e-9)


@template(M, "medium")
def regression_search_tasks(rng):
    t = table(rng, kind="reg", numeric_only=True, n=(150, 260))
    s = seeded(rng)
    setup = (t.setup + "\nfrom sklearn.model_selection import train_test_split\n"
             f"X_tr, X_te, y_tr, y_te = train_test_split(df, y, random_state={s})")
    rmse = "from sklearn.metrics import root_mean_squared_error\n"
    items = [
        ("Ridge", "Tune `alpha` over `[0.1, 1, 10, 100]` of `make_pipeline(StandardScaler(), Ridge())` with `cv=5` "
                  "and `scoring='neg_root_mean_squared_error'` on the training part. Return `(best alpha, CV RMSE)` "
                  "with a positive RMSE.",
         f"{GS}from sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\n"
         "from sklearn.linear_model import Ridge\ngrid = GridSearchCV(make_pipeline(StandardScaler(), Ridge()), "
         "{'ridge__alpha': [0.1, 1, 10, 100]}, cv=5, scoring='neg_root_mean_squared_error').fit(X_tr, y_tr)\n"
         "result = (grid.best_params_['ridge__alpha'], -grid.best_score_)"),
        ("KNeighborsRegressor", "Tune `n_neighbors` over `[3, 7, 15]` and `weights` over `['uniform', 'distance']` of a "
                                "scaled `KNeighborsRegressor` (`make_pipeline(StandardScaler(), KNeighborsRegressor())`) "
                                "with `cv=5` on the training part. Return `best_params_`.",
         f"{GS}from sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\n"
         "from sklearn.neighbors import KNeighborsRegressor\ngrid = GridSearchCV(make_pipeline(StandardScaler(), KNeighborsRegressor()), "
         "{'kneighborsregressor__n_neighbors': [3, 7, 15], 'kneighborsregressor__weights': ['uniform', 'distance']}, cv=5).fit(X_tr, y_tr)\n"
         "result = grid.best_params_"),
        ("DecisionTreeRegressor", "Tune `max_depth` over `[2, 4, 6, 8]` of `DecisionTreeRegressor(random_state=0)` with "
                                  "`cv=5` and `scoring='neg_mean_absolute_error'` on the training part. Return `(best "
                                  "depth, test MAE of the refitted search)`.",
         f"{GS}from sklearn.tree import DecisionTreeRegressor\nfrom sklearn.metrics import mean_absolute_error\n"
         "grid = GridSearchCV(DecisionTreeRegressor(random_state=0), {'max_depth': [2, 4, 6, 8]}, cv=5, "
         "scoring='neg_mean_absolute_error').fit(X_tr, y_tr)\n"
         "result = (grid.best_params_['max_depth'], mean_absolute_error(y_te, grid.predict(X_te)))"),
        ("score", "With a custom `scoring`, `search.score` uses that metric too. Tune `alpha` over `[0.1, 10, 1000]` "
                  "of `Ridge()` with `cv=5`, `scoring='neg_mean_absolute_error'` on the training part, and return "
                  "`grid.score(X_te, y_te)` (a negated test MAE).",
         f"{GS}from sklearn.linear_model import Ridge\ngrid = GridSearchCV(Ridge(), {{'alpha': [0.1, 10, 1000]}}, cv=5, "
         "scoring='neg_mean_absolute_error').fit(X_tr, y_tr)\nresult = grid.score(X_te, y_te)"),
        ("root_mean_squared_error", "Tune `alpha` over `[0.01, 1, 100]` of `Ridge()` (`cv=5`, default R² scoring) on "
                                    "the training part. Return `(cv_r2, test_rmse)` of the winner.",
         f"{GS}{rmse}from sklearn.linear_model import Ridge\ngrid = GridSearchCV(Ridge(), {{'alpha': [0.01, 1, 100]}}, cv=5).fit(X_tr, y_tr)\n"
         "result = (grid.best_score_, root_mean_squared_error(y_te, grid.predict(X_te)))"),
    ]
    return catalog(rng, items, setup, f"{t.desc}. It was split with `train_test_split(df, y, random_state={s})` into "
                                      f"`X_tr, X_te, y_tr, y_te`.", tol=1e-9)


@template(M, "medium")
def halving_vs_grid(rng):
    setup, desc = _data(rng, n=(200, 320), cancer=0.3)
    cs = sorted(rng.sample([0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0], rng.randint(5, 7)))
    f = rng.choice([2, 3])
    h = f"HalvingGridSearchCV(pipe, {{'clf__C': {cs}}}, cv=3, factor={f}, random_state=0)"
    g = f"GridSearchCV(pipe, {{'clf__C': {cs}}}, cv=3)"
    task = rng.choice(["agree", "fits", "lastround", "minres"])
    head = (f"{desc}; {PIPE_DESC}. Compare `{g}` with successive halving `{h}` (enable it with `from "
            f"sklearn.experimental import enable_halving_search_cv`), both fitted on `X, y`.")
    sol = HALVING + f"from sklearn.model_selection import GridSearchCV\nh = {h}.fit(X, y)\n"
    setup2 = setup + "\n" + PIPE_SETUP
    if task == "agree":
        return P(f"{head} Return `(halving_best_C, grid_best_C)`.", setup2,
                 sol + f"grid = {g}.fit(X, y)\nresult = (h.best_params_['clf__C'], grid.best_params_['clf__C'])",
                 require=["HalvingGridSearchCV", "GridSearchCV"])
    if task == "fits":
        return P(f"{head} Halving is cheaper. Return `(halving_evaluations, grid_evaluations)`: the number of rows "
                 f"in each search's `cv_results_['params']` (halving evaluates a candidate again in every round it "
                 f"survives).", setup2,
                 sol + f"grid = {g}.fit(X, y)\nresult = (len(h.cv_results_['params']), len(grid.cv_results_['params']))",
                 require=["HalvingGridSearchCV", "cv_results_"])
    if task == "lastround":
        return P(f"{head} Return the `clf__C` values of the candidates that reached the **last** round "
                 f"(`cv_results_['iter']` equals `n_iterations_ - 1`), in order.", setup2,
                 sol + "res = h.cv_results_\nlast = h.n_iterations_ - 1\n"
                       "result = [p['clf__C'] for p, it in zip(res['params'], res['iter']) if it == last]",
                 require=["HalvingGridSearchCV", "cv_results_"])
    return P(f"{head} Return `(min_resources_, max_resources_, n_resources_)` of the halving search (the last as a "
             f"list): how many samples the first and the last rounds got.", setup2,
             sol + "result = (int(h.min_resources_), int(h.max_resources_), list(h.n_resources_))",
             require=["HalvingGridSearchCV"])


@template(M, "medium")
def search_with_cv_splitters(rng):
    task = rng.choice(["strat", "group", "time"])
    s = seeded(rng)
    cs = _cs(rng)
    if task == "strat":
        setup, desc = _data(rng, cancer=0.4)
        return P(f"{desc}. Pass a splitter object as `cv`: tune `C` over `{cs}` of `make_pipeline(StandardScaler(), "
                 f"LogisticRegression(max_iter=1000))` with `cv=StratifiedKFold(n_splits=4, shuffle=True, "
                 f"random_state={s})`. Return `(best_params_, best_score_)`.", setup + "\n" + LR_PIPE,
                 f"{GS}from sklearn.model_selection import StratifiedKFold\n"
                 f"grid = GridSearchCV(lr, {{'logisticregression__C': {cs}}}, cv=StratifiedKFold(n_splits=4, shuffle=True, random_state={s})).fit(X, y)\n"
                 "result = (grid.best_params_, grid.best_score_)", require=["StratifiedKFold", "GridSearchCV"],
                 tol=1e-9)
    if task == "group":
        n_g = rng.randint(6, 10)
        setup = (f"g = np.random.default_rng({s})\ngroups = np.repeat(np.arange({n_g}), 20)\n"
                 f"X = g.normal(size=({n_g * 20}, 4)) + g.normal(0, 2, ({n_g}, 4))[groups]\n"
                 f"y = (X[:, 0] + X[:, 1] + g.normal(0, 1, {n_g * 20}) > 0).astype(int)")
        return P(f"`X`, `y` hold 20 measurements from each of {n_g} patients (`groups`); rows of one patient are "
                 f"alike. Tune `n_neighbors` over `[1, 5, 15]` of `KNeighborsClassifier()` with `cv=GroupKFold("
                 f"n_splits=3)` so that no patient is in both a training and a validation fold: pass `groups=groups` "
                 f"to `fit`. Return `(best_params_, best_score_)`.", setup,
                 f"{GS}from sklearn.model_selection import GroupKFold\nfrom sklearn.neighbors import KNeighborsClassifier\n"
                 "grid = GridSearchCV(KNeighborsClassifier(), {'n_neighbors': [1, 5, 15]}, cv=GroupKFold(n_splits=3)).fit(X, y, groups=groups)\n"
                 "result = (grid.best_params_, grid.best_score_)", require=["GroupKFold", "groups"], tol=1e-9)
    n = rng.randint(150, 240)
    setup = (f"g = np.random.default_rng({s})\nt = np.arange({n})\n"
             f"X = np.column_stack([t, np.sin(t / 8), g.normal(size={n})])\n"
             f"y = 0.05 * t + 3 * np.sin(t / 8) + g.normal(0, 0.5, {n})")
    return P(f"`X` and `y` are {n} time-ordered observations (column 0 is time). Validation folds must come after "
             f"their training data: tune `alpha` over `[0.01, 1, 100]` of `Ridge()` with `cv=TimeSeriesSplit(n_splits=4)` "
             f"and `scoring='neg_mean_absolute_error'`. Return `(best alpha, CV MAE as a positive number)`.", setup,
             f"{GS}from sklearn.model_selection import TimeSeriesSplit\nfrom sklearn.linear_model import Ridge\n"
             "grid = GridSearchCV(Ridge(), {'alpha': [0.01, 1, 100]}, cv=TimeSeriesSplit(n_splits=4), "
             "scoring='neg_mean_absolute_error').fit(X, y)\nresult = (grid.best_params_['alpha'], -grid.best_score_)",
             require=["TimeSeriesSplit"], tol=1e-9)


@template(M, "medium")
def cv_results_analysis(rng):
    setup, desc = _data(rng, cancer=0.4)
    cs = sorted(rng.sample([0.001, 0.01, 0.1, 1.0, 10.0, 100.0], 5))
    setup += "\n" + LR_PIPE + f"\n{GS.strip()}\ngrid = GridSearchCV(lr, {{'logisticregression__C': {cs}}}, cv=5).fit(X, y)"
    key = "logisticregression__C"
    items = [
        ("cv_results_", "Return the mean test score as a pandas Series indexed by `C` (from `cv_results_['params']` and "
                        "`cv_results_['mean_test_score']`).",
         f"res = grid.cv_results_\nresult = pd.Series(res['mean_test_score'], index=[p[{key!r}] for p in res['params']])"),
        ("cv_results_", "Return the five per-fold scores of the best candidate (`cv_results_['split0_test_score']` ... "
                        "`cv_results_['split4_test_score']` at `grid.best_index_`).",
         "res = grid.cv_results_\nresult = np.array([res[f'split{i}_test_score'][grid.best_index_] for i in range(5)])"),
        ("cv_results_", "Return the `C` whose score varies **most** across folds (largest `std_test_score`).",
         f"res = grid.cv_results_\nresult = res['params'][int(np.argmax(res['std_test_score']))][{key!r}]"),
        ("cv_results_", "One-standard-error rule: among the candidates whose mean score is at least (best mean − its "
                        "std), return the **smallest** `C` (the most regularized, simplest model).",
         f"res = grid.cv_results_\nm, sd = res['mean_test_score'], res['std_test_score']\nb = int(np.argmax(m))\n"
         f"result = min(p[{key!r}] for p, v in zip(res['params'], m) if v >= m[b] - sd[b])"),
        ("DataFrame", "Build `pd.DataFrame(grid.cv_results_)` and return the `C` values (column `param_logisticregression__C`) of "
                      "the rows with `rank_test_score` ≤ 2, as a list of floats in rank order.",
         "df = pd.DataFrame(grid.cv_results_).sort_values('rank_test_score', kind='stable')\n"
         f"result = [float(c) for c in df.loc[df['rank_test_score'] <= 2, 'param_{key}']]"),
        ("cv_results_", "Return how many candidates score within 0.01 of `grid.best_score_` (the winner included).",
         "result = int((grid.cv_results_['mean_test_score'] >= grid.best_score_ - 0.01).sum())"),
    ]
    return catalog(rng, items, setup, f"{desc}; `lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))`"
                                      f" and `grid = GridSearchCV(lr, {{'logisticregression__C': {cs}}}, cv=5)` was "
                                      f"fitted on them.", tol=1e-9)


# ================================================================= hard

@template(M, "hard")
def leaky_tuning(rng):
    d = synth_clf(rng, n=(80, 140), d=(40, 80), informative=3)
    k = rng.choice([3, 5, 8])
    cs = _cs(rng)
    task = rng.choice(["select", "scale"])
    if task == "select":
        return P(f"{d.desc} (mostly noise features). A colleague picks the {k} best features with "
                 f"`SelectKBest(f_classif, k={k})` on **all** rows, then tunes `C` over `{cs}` of "
                 f"`LogisticRegression(max_iter=1000)` with `GridSearchCV(cv=5)` on the selected columns. The selection "
                 f"saw the validation folds, so the score is inflated. Return `(leaky, honest)`: their `best_score_` "
                 f"and the `best_score_` when `SelectKBest` sits inside a `Pipeline` that is tuned (step names "
                 f"`'select'`, `'clf'`).", d.setup,
                 f"{GS}from sklearn.pipeline import Pipeline\nfrom sklearn.feature_selection import SelectKBest, f_classif\n"
                 "from sklearn.linear_model import LogisticRegression\n"
                 f"Xs = SelectKBest(f_classif, k={k}).fit_transform(X, y)\n"
                 f"leaky = GridSearchCV(LogisticRegression(max_iter=1000), {{'C': {cs}}}, cv=5).fit(Xs, y).best_score_\n"
                 f"pipe = Pipeline([('select', SelectKBest(f_classif, k={k})), ('clf', LogisticRegression(max_iter=1000))])\n"
                 f"honest = GridSearchCV(pipe, {{'clf__C': {cs}}}, cv=5).fit(X, y).best_score_\nresult = (leaky, honest)",
                 require=["Pipeline", "SelectKBest", "GridSearchCV"], tol=1e-9)
    return P(f"{d.desc}. Tune `k` of `SelectKBest(f_classif)` over `[2, {k}, 20]` **and** `C` over `{cs}` of "
             f"`LogisticRegression(max_iter=1000)` in one `Pipeline([('select', ...), ('clf', ...)])` with "
             f"`GridSearchCV(cv=5)`, so every fold selects its own features. Return `(best_params_, best_score_)`.",
             d.setup,
             f"{GS}from sklearn.pipeline import Pipeline\nfrom sklearn.feature_selection import SelectKBest, f_classif\n"
             "from sklearn.linear_model import LogisticRegression\n"
             "pipe = Pipeline([('select', SelectKBest(f_classif)), ('clf', LogisticRegression(max_iter=1000))])\n"
             f"grid = GridSearchCV(pipe, {{'select__k': [2, {k}, 20], 'clf__C': {cs}}}, cv=5).fit(X, y)\n"
             "result = (grid.best_params_, grid.best_score_)", require=["Pipeline", "SelectKBest"], tol=1e-9)


@template(M, "hard")
def winners_curse(rng):
    setup, desc = _data(rng, n=(120, 200), cancer=0.0)
    m = rng.choice([8, 12, 16])
    s = seeded(rng)
    if rng.random() < 0.5:
        return P(f"{desc}. Tuning on noise: search `random_state` over `range({m})` of `DecisionTreeClassifier("
                 f"max_depth=3, random_state=...)`, with `max_features=1` so each seed builds a different tree, using "
                 f"`GridSearchCV(cv=5)` on `X, y`. Return `(best_score_, mean_of_all_candidates)`: the winner looks "
                 f"better than a typical seed only by luck.", setup,
                 f"{GS}from sklearn.tree import DecisionTreeClassifier\n"
                 f"grid = GridSearchCV(DecisionTreeClassifier(max_depth=3, max_features=1), {{'random_state': list(range({m}))}}, cv=5).fit(X, y)\n"
                 "result = (grid.best_score_, grid.cv_results_['mean_test_score'].mean())",
                 require=["GridSearchCV", "cv_results_"], tol=1e-9)
    return P(f"{desc}. Hold out a test set (`train_test_split(X, y, test_size=0.3, stratify=y, random_state={s})`). "
             f"Tune `random_state` over `range({m})` of `DecisionTreeClassifier(max_depth=3, max_features=1)` with "
             f"`GridSearchCV(cv=5)` on the training part. Return `(best_score_, test_score)`: the CV score of the "
             f"lucky winner vs its accuracy on the untouched test set.", setup,
             f"{GS}from sklearn.model_selection import train_test_split\nfrom sklearn.tree import DecisionTreeClassifier\n"
             f"X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state={s})\n"
             f"grid = GridSearchCV(DecisionTreeClassifier(max_depth=3, max_features=1), {{'random_state': list(range({m}))}}, cv=5).fit(X_tr, y_tr)\n"
             "result = (grid.best_score_, grid.score(X_te, y_te))", require=["GridSearchCV"], tol=1e-9)


@template(M, "hard")
def grid_search_by_hand(rng):
    setup, desc = _data(rng, cancer=0.0)
    cs, ks = _cs(rng), rng.sample([1, 3, 5, 9], 2)
    cv = rng.choice([3, 4, 5])
    task = rng.choice(["best", "means", "ranks"])
    loop = ("from sklearn.model_selection import ParameterGrid, cross_val_score\n"
            "from sklearn.linear_model import LogisticRegression\n"
            f"cands = list(ParameterGrid({{'C': {cs}, 'class_weight': [None, 'balanced']}}))\n"
            f"means = np.array([cross_val_score(LogisticRegression(max_iter=1000, **p), X, y, cv={cv}).mean() for p in cands])\n")
    head = (f"{desc}. Reimplement `GridSearchCV(LogisticRegression(max_iter=1000), {{'C': {cs}, 'class_weight': [None, "
            f"'balanced']}}, cv={cv})` by hand: iterate over `ParameterGrid` of that grid (same candidate order), score "
            f"each candidate with `cross_val_score(..., cv={cv})`.")
    if task == "best":
        return P(f"{head} Return the winning parameter dict (the first one on ties, as `np.argmax` gives); it equals "
                 f"`best_params_`.", setup, loop + "result = cands[int(np.argmax(means))]",
                 require=["ParameterGrid", "cross_val_score"])
    if task == "means":
        return P(f"{head} Return the array of mean scores; it equals `cv_results_['mean_test_score']`.", setup,
                 loop + "result = means", require=["ParameterGrid", "cross_val_score"], tol=1e-9)
    return P(f"{head} Return the ranks like `rank_test_score`: 1 for the best mean, tied means share the **smallest** "
             f"rank (`scipy.stats.rankdata(-means, method='min')` as ints).", setup,
             loop + "from scipy.stats import rankdata\nresult = rankdata(-means, method='min').astype(int)",
             require=["ParameterGrid", "cross_val_score"])


@template(M, "hard")
def one_standard_error_refit(rng):
    setup, desc = _data(rng, n=(150, 250), cancer=0.4)
    cs = sorted(rng.sample([0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 10.0], 6))
    rule = ("among the candidates whose `mean_test_score` is at least (best mean − std of the best), pick the one with "
            "the **smallest** `C` (candidates are listed with `C` increasing)")
    if rng.random() < 0.5:
        return P(f"{desc}; `lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))`. Prefer a simpler "
                 f"model when it is statistically as good: `refit` accepts a function of `cv_results_` that returns "
                 f"the index of the model to refit. Write one that implements the one-standard-error rule ({rule}), "
                 f"pass it as `refit=` to `GridSearchCV(lr, {{'logisticregression__C': {cs}}}, cv=5)`, fit on `X, y` "
                 f"and return `(best_index_, best_params_)`.", setup + "\n" + LR_PIPE,
                 f"{GS}def one_se(res):\n    m, sd = res['mean_test_score'], res['std_test_score']\n"
                 "    b = int(np.argmax(m))\n    return int(np.flatnonzero(m >= m[b] - sd[b])[0])\n"
                 f"grid = GridSearchCV(lr, {{'logisticregression__C': {cs}}}, cv=5, refit=one_se).fit(X, y)\n"
                 "result = (int(grid.best_index_), grid.best_params_)", require=["refit"])
    return P(f"{desc}; `lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))`. Fit "
             f"`GridSearchCV(lr, {{'logisticregression__C': {cs}}}, cv=5)` on `X, y` and apply the one-standard-error "
             f"rule to its `cv_results_` yourself ({rule}). Return `(plain_best_C, one_se_C)`.",
             setup + "\n" + LR_PIPE,
             f"{GS}grid = GridSearchCV(lr, {{'logisticregression__C': {cs}}}, cv=5).fit(X, y)\nres = grid.cv_results_\n"
             "m, sd = res['mean_test_score'], res['std_test_score']\nb = int(np.argmax(m))\n"
             "pick = int(np.flatnonzero(m >= m[b] - sd[b])[0])\n"
             "result = (grid.best_params_['logisticregression__C'], res['params'][pick]['logisticregression__C'])",
             require=["cv_results_"])


@template(M, "hard")
def search_debugging(rng):
    setup, desc = _data(rng, n=(150, 240), cancer=0.4)
    cs = _cs(rng)
    s = seeded(rng)
    n_iter = rng.choice([5, 8])
    items = [
        ("GridSearchCV", f"`GridSearchCV(pipe, {{'C': {cs}}}, cv=3).fit(X, y)` raises `ValueError: Invalid parameter "
                         f"'C'`. Fix the grid and return `best_params_`.",
         f"{GS}result = GridSearchCV(pipe, {{'clf__C': {cs}}}, cv=3).fit(X, y).best_params_"),
        ("scoring", f"`GridSearchCV(pipe, {{'clf__C': {cs}}}, cv=3, scoring='auc')` raises because `'auc'` is not a "
                    f"scorer name (check `get_scorer_names()`). Use the ROC AUC scorer and return `best_score_`.",
         f"{GS}result = GridSearchCV(pipe, {{'clf__C': {cs}}}, cv=3, scoring='roc_auc').fit(X, y).best_score_"),
        ("enable_halving_search_cv", f"`from sklearn.model_selection import HalvingGridSearchCV` raises `ImportError`. "
                                     f"Fix the import, then fit `HalvingGridSearchCV(pipe, {{'clf__C': {cs}}}, cv=3, "
                                     f"factor=2, random_state=0)` and return `best_params_`.",
         HALVING + f"result = HalvingGridSearchCV(pipe, {{'clf__C': {cs}}}, cv=3, factor=2, random_state=0).fit(X, y).best_params_",
         {"require": ["enable_halving_search_cv", "HalvingGridSearchCV"]}),
        ("random_state", f"`RandomizedSearchCV(pipe, {{'clf__C': loguniform(1e-2, 1e2)}}, n_iter={n_iter}, cv=3)` "
                         f"gives a different answer on every run. Make it reproducible with `random_state={s}` and "
                         f"return `best_params_`.",
         f"{RS}result = RandomizedSearchCV(pipe, {{'clf__C': loguniform(1e-2, 1e2)}}, n_iter={n_iter}, cv=3, "
         f"random_state={s}).fit(X, y).best_params_", {"require": ["RandomizedSearchCV", "random_state"]}),
        ("best_estimator_", f"`GridSearchCV(pipe, {{'clf__C': {cs}}}, cv=3, refit=False)` was fitted and then "
                            f"`grid.best_estimator_.predict(X[:5])` raised `AttributeError`. Fix it (refit the winner) "
                            f"and return those 5 predictions.",
         f"{GS}grid = GridSearchCV(pipe, {{'clf__C': {cs}}}, cv=3, refit=True).fit(X, y)\n"
         "result = grid.best_estimator_.predict(X[:5])"),
        ("param_grid", f"`GridSearchCV(pipe, {{'clf__C': {cs[1]}}}, cv=3)` raises `TypeError`: grid values must be "
                       f"lists of candidates. Search `C` over `[{cs[1]}, {cs[1] * 10}]` instead and return "
                       f"`cv_results_['mean_test_score']`.",
         f"{GS}result = GridSearchCV(pipe, {{'clf__C': [{cs[1]}, {cs[1] * 10}]}}, cv=3).fit(X, y).cv_results_['mean_test_score']",
         {"require": ["GridSearchCV"]}),
        ("best_params_", f"A colleague searched `alpha` of `Ridge()` over `{cs}` on `(X, y.astype(float))` with "
                        f"`scoring='neg_mean_squared_error'`, `cv=3`, and reported the `alpha` with the **lowest** "
                        f"`mean_test_score` as the best. Negated errors are higher-is-better. Return `(their_alpha, "
                        f"best_alpha)`.",
         f"{GS}from sklearn.linear_model import Ridge\ngrid = GridSearchCV(Ridge(), {{'alpha': {cs}}}, cv=3, "
         "scoring='neg_mean_squared_error').fit(X, y.astype(float))\nm = grid.cv_results_['mean_test_score']\n"
         f"result = ({cs}[int(np.argmin(m))], grid.best_params_['alpha'])"),
        ("cross_val_score", f"To estimate the tuned model's accuracy, a colleague reports `GridSearchCV(pipe, "
                            f"{{'clf__C': {cs}}}, cv=3).fit(X, y).best_score_`. That number is biased upward. Return "
                            f"the nested estimate instead: the mean of `cross_val_score` of that search with `cv=3`.",
         f"{GS}from sklearn.model_selection import cross_val_score\n"
         f"result = cross_val_score(GridSearchCV(pipe, {{'clf__C': {cs}}}, cv=3), X, y, cv=3).mean()",
         {"require": ["cross_val_score", "GridSearchCV"]}),
    ]
    return catalog(rng, items, setup + "\n" + PIPE_SETUP, f"{desc}; {PIPE_DESC}.", tol=1e-9)


@template(M, "hard")
def nested_search_by_hand(rng):
    setup, desc = _data(rng, n=(150, 240), cancer=0.4)
    cs = _cs(rng)
    outer, s = rng.choice([3, 4, 5]), seeded(rng)
    inner = f"GridSearchCV(pipe, {{'clf__C': {cs}}}, cv=3)"
    loop = (f"{GS}from sklearn.model_selection import StratifiedKFold\nscores, chosen = [], []\n"
            f"for tr, te in StratifiedKFold(n_splits={outer}, shuffle=True, random_state={s}).split(X, y):\n"
            f"    g = {inner}.fit(X[tr], y[tr])\n    scores.append(g.score(X[te], y[te]))\n"
            "    chosen.append(g.best_params_['clf__C'])\n")
    head = (f"{desc}; {PIPE_DESC}. Write nested CV by hand: for each split of `StratifiedKFold(n_splits={outer}, "
            f"shuffle=True, random_state={s})`, fit `{inner}` on the outer training rows and score it on the outer "
            f"test rows.")
    task = rng.choice(["mean", "pairs", "match"])
    if task == "mean":
        return P(f"{head} Return `(mean outer score, std of the outer scores)`.", setup + "\n" + PIPE_SETUP,
                 loop + "result = (float(np.mean(scores)), float(np.std(scores)))",
                 require=["StratifiedKFold", "GridSearchCV"], tol=1e-9)
    if task == "pairs":
        return P(f"{head} Return a list of `(chosen C, outer score)` tuples, one per outer fold.",
                 setup + "\n" + PIPE_SETUP, loop + "result = list(zip(chosen, scores))",
                 require=["StratifiedKFold", "GridSearchCV"], tol=1e-9)
    return P(f"{head} Return the outer scores as an array; they equal `cross_val_score({inner}, X, y, "
             f"cv=StratifiedKFold(n_splits={outer}, shuffle=True, random_state={s}))`.", setup + "\n" + PIPE_SETUP,
             loop + "result = np.array(scores)", require=["StratifiedKFold", "GridSearchCV"], tol=1e-9)


@template(M, "hard")
def halving_schedule(rng):
    setup, desc = _data(rng, n=(200, 320), cancer=0.3)
    n_c = rng.randint(5, 8)
    cs = sorted(rng.sample([0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0], n_c))
    f = rng.choice([2, 3])
    h = f"HalvingGridSearchCV(pipe, {{'clf__C': {cs}}}, cv=3, factor={f}, random_state=0)"
    if rng.random() < 0.5:
        return P(f"{desc}; {PIPE_DESC}. Fit `h = {h}` on `X, y` (enable it with `from sklearn.experimental import "
                 f"enable_halving_search_cv`). Then rebuild its schedule **by hand** from `h.min_resources_`, the "
                 f"factor and the number of candidates: round `i` uses `min_resources_ · {f}^i` samples, and each round "
                 f"keeps `ceil(candidates / {f})` of them, for `h.n_iterations_` rounds. Return `(candidates, "
                 f"resources)` as two lists; they equal `n_candidates_` and `n_resources_`.", setup + "\n" + PIPE_SETUP,
                 HALVING + f"h = {h}.fit(X, y)\ncands, res, c = [], [], {n_c}\nfor i in range(h.n_iterations_):\n"
                           f"    cands.append(c)\n    res.append(int(h.min_resources_ * {f} ** i))\n"
                           f"    c = int(np.ceil(c / {f}))\nresult = (cands, res)",
                 require=["HalvingGridSearchCV", "min_resources_"])
    return P(f"{desc}; {PIPE_DESC}. Write successive halving by hand over `C` in `{cs}`: start with all candidates and "
             f"`r = 40` rows; each round, score every remaining candidate with `cross_val_score(pipe.set_params("
             f"clf__C=C), X[:r], y[:r], cv=3).mean()`, keep the best `ceil(len / {f})` (stable order on ties: sort by "
             f"score descending with `kind='stable'`), and multiply `r` by {f} (at most `len(X)`); stop when one "
             f"candidate is left. Return the list of surviving `C` lists, one per round **after** elimination.",
             setup + "\n" + PIPE_SETUP,
             "from sklearn.model_selection import cross_val_score\n"
             f"alive, r, rounds = {cs}, 40, []\nwhile len(alive) > 1:\n"
             "    sc = np.array([cross_val_score(pipe.set_params(clf__C=C), X[:r], y[:r], cv=3).mean() for C in alive])\n"
             f"    keep = int(np.ceil(len(alive) / {f}))\n"
             "    alive = [alive[i] for i in np.argsort(-sc, kind='stable')[:keep]]\n"
             f"    rounds.append(alive)\n    r = min(r * {f}, len(X))\nresult = rounds",
             require=["cross_val_score", "set_params"])


@template(M, "hard")
def tuning_workflows(rng):
    t = table(rng, kind="clf", n=(180, 300), missing=1)
    s = seeded(rng)
    miss = t.missing[0]
    num, cat = t.num, t.cat
    setup = (t.setup + f"\nnum, cat = {lit(num)}, {lit(cat)}\nfrom sklearn.model_selection import train_test_split\n"
             f"X_tr, X_te, y_tr, y_te = train_test_split(df, y, stratify=y, random_state={s})")
    prep = ("from sklearn.compose import ColumnTransformer\nfrom sklearn.pipeline import Pipeline, make_pipeline\n"
            "from sklearn.impute import SimpleImputer\nfrom sklearn.preprocessing import OneHotEncoder, StandardScaler\n"
            "from sklearn.linear_model import LogisticRegression\n"
            "prep = ColumnTransformer([('num', make_pipeline(SimpleImputer(), StandardScaler()), num), "
            "('cat', OneHotEncoder(handle_unknown='ignore'), cat)])\n"
            "model = Pipeline([('prep', prep), ('clf', LogisticRegression(max_iter=1000))])\n")
    ptxt = ("`model = Pipeline([('prep', prep), ('clf', LogisticRegression(max_iter=1000))])` where `prep = "
            "ColumnTransformer([('num', make_pipeline(SimpleImputer(), StandardScaler()), num), ('cat', "
            "OneHotEncoder(handle_unknown='ignore'), cat)])`")
    items = [
        ("SimpleImputer", f"Build {ptxt}. Tune the imputation strategy (`prep__num__simpleimputer__strategy` over "
                          f"`['mean', 'median']`) and `clf__C` over `[0.1, 1, 10]` with `GridSearchCV(cv=5)` on the "
                          f"training part. Return `(best_params_, test accuracy)`.",
         f"{GS}{prep}grid = GridSearchCV(model, {{'prep__num__simpleimputer__strategy': ['mean', 'median'], "
         "'clf__C': [0.1, 1, 10]}, cv=5).fit(X_tr, y_tr)\nresult = (grid.best_params_, grid.score(X_te, y_te))"),
        ("OneHotEncoder", f"Build {ptxt}. Is the categorical block worth keeping? A grid can drop a transformer: search "
                          f"`prep__cat` over `[OneHotEncoder(handle_unknown='ignore'), 'drop']` with `cv=5` on the "
                          f"training part. Return `(kept, best_score_)` with `kept = best_params_['prep__cat'] != "
                          f"'drop'`.",
         f"{GS}{prep}grid = GridSearchCV(model, {{'prep__cat': [OneHotEncoder(handle_unknown='ignore'), 'drop']}}, cv=5).fit(X_tr, y_tr)\n"
         "result = (not isinstance(grid.best_params_['prep__cat'], str), grid.best_score_)"),
        ("class_weight", f"Build {ptxt}. Tune `clf__class_weight` over `[None, 'balanced']` twice, with `scoring="
                         f"'accuracy'` and with `scoring='recall'` (`cv=5`, training part). Return the two chosen "
                         f"weights as a tuple.",
         f"{GS}{prep}a = GridSearchCV(model, {{'clf__class_weight': [None, 'balanced']}}, cv=5, scoring='accuracy').fit(X_tr, y_tr)\n"
         "b = GridSearchCV(model, {'clf__class_weight': [None, 'balanced']}, cv=5, scoring='recall').fit(X_tr, y_tr)\n"
         "result = (a.best_params_['clf__class_weight'], b.best_params_['clf__class_weight'])", {"require": ["scoring"]}),
        ("randint", "Random-search a tree's integer hyperparameters: `RandomizedSearchCV(DecisionTreeClassifier("
                    "random_state=0), {'max_depth': randint(1, 8), 'min_samples_leaf': randint(1, 30)}, n_iter=8, "
                    f"cv=5, random_state={s})` on the training part's **numeric** columns with missing values filled "
                    f"by `SimpleImputer()` in a pipeline (`make_pipeline(SimpleImputer(), DecisionTreeClassifier("
                    f"random_state=0))`, parameters `decisiontreeclassifier__...`). `randint` is `scipy.stats.randint`. "
                    f"Return `best_params_`.",
         "from sklearn.model_selection import RandomizedSearchCV\nfrom scipy.stats import randint\n"
         "from sklearn.pipeline import make_pipeline\nfrom sklearn.impute import SimpleImputer\n"
         "from sklearn.tree import DecisionTreeClassifier\n"
         "tree = make_pipeline(SimpleImputer(), DecisionTreeClassifier(random_state=0))\n"
         "rand = RandomizedSearchCV(tree, {'decisiontreeclassifier__max_depth': randint(1, 8), "
         f"'decisiontreeclassifier__min_samples_leaf': randint(1, 30)}}, n_iter=8, cv=5, random_state={s}).fit(X_tr[num], y_tr)\n"
         "result = rand.best_params_", {"require": ["RandomizedSearchCV", "randint"]}),
        ("refit", f"Build {ptxt}. Search `clf__C` over `[0.01, 0.1, 1, 10]` scoring both `'roc_auc'` and `'f1'` with "
                  f"`refit='f1'` (`cv=5`, training part). Return `(best_params_, mean ROC AUC of that candidate)`.",
         f"{GS}{prep}grid = GridSearchCV(model, {{'clf__C': [0.01, 0.1, 1, 10]}}, scoring={{'roc_auc': 'roc_auc', 'f1': 'f1'}}, "
         "refit='f1', cv=5).fit(X_tr, y_tr)\nresult = (grid.best_params_, grid.cv_results_['mean_test_roc_auc'][grid.best_index_])"),
        ("GridSearchCV", f"Build {ptxt}. Tune `clf__C` over `[0.01, 0.1, 1, 10]` with `cv=5` on the training part, and "
                         f"return the test accuracy of the tuned model minus that of the untuned `model` (both fitted "
                         f"on the training part).",
         f"{GS}{prep}tuned = GridSearchCV(model, {{'clf__C': [0.01, 0.1, 1, 10]}}, cv=5).fit(X_tr, y_tr).score(X_te, y_te)\n"
         "result = tuned - model.fit(X_tr, y_tr).score(X_te, y_te)"),
    ]
    return catalog(rng, items, setup, f"{t.desc}; the numeric column `{miss}` has missing values. `num` and `cat` list "
                                      f"the numeric and categorical columns; the rows were split with "
                                      f"`train_test_split(df, y, stratify=y, random_state={s})` into `X_tr, X_te, y_tr, "
                                      f"y_te`.", tol=1e-9)


@template(M, "hard")
def scorer_internals(rng):
    setup, desc = _data(rng, n=(150, 250), cancer=0.0)
    task = rng.choice(["mae", "logloss", "sign"])
    if task == "mae":
        d = synth_reg(rng, n=(100, 200))
        return P(f"{d.desc}. A scorer string is a function of `(estimator, X, y)`. Fit `Ridge()` on `X, y` and return "
                 f"`(get_scorer('neg_mean_absolute_error')(reg, X, y), mean_absolute_error(y, reg.predict(X)))`: same "
                 f"number, opposite sign.", d.setup,
                 "from sklearn.metrics import get_scorer, mean_absolute_error\nfrom sklearn.linear_model import Ridge\n"
                 "reg = Ridge().fit(X, y)\nresult = (get_scorer('neg_mean_absolute_error')(reg, X, y), mean_absolute_error(y, reg.predict(X)))",
                 require=["get_scorer", "mean_absolute_error"], tol=1e-9)
    if task == "logloss":
        return P(f"{desc}. Build the log-loss scorer yourself: `make_scorer(log_loss, greater_is_better=False, "
                 f"response_method='predict_proba')`. Fit `LogisticRegression(max_iter=1000)` on `X, y` and return "
                 f"`(yours(clf, X, y), get_scorer('neg_log_loss')(clf, X, y))` (equal).", setup,
                 "from sklearn.metrics import make_scorer, log_loss, get_scorer\nfrom sklearn.linear_model import LogisticRegression\n"
                 "clf = LogisticRegression(max_iter=1000).fit(X, y)\n"
                 "mine = make_scorer(log_loss, greater_is_better=False, response_method='predict_proba')\n"
                 "result = (mine(clf, X, y), get_scorer('neg_log_loss')(clf, X, y))",
                 require=["make_scorer", "response_method"], tol=1e-9)
    return P(f"{desc}. Tune `C` over `[0.01, 1, 100]` of `LogisticRegression(max_iter=1000)` with `GridSearchCV(cv=3, "
             f"scoring='neg_log_loss')`. Return `(best_C, mean CV log loss of the winner as a positive number)`.",
             setup, f"{GS}from sklearn.linear_model import LogisticRegression\n"
                    "grid = GridSearchCV(LogisticRegression(max_iter=1000), {'C': [0.01, 1, 100]}, cv=3, scoring='neg_log_loss').fit(X, y)\n"
                    "result = (grid.best_params_['C'], -grid.best_score_)", require=["GridSearchCV"], tol=1e-9)


# ================================================================= challenge

_NAMES = ["alpha", "beta", "C", "depth", "eta", "gamma", "lr", "min_leaf", "n_est", "tol", "width", "zeta"]


@template(M, "challenge")
def parameter_grid_lookup(rng):
    s = seeded(rng)
    report = rng.choice(["subgrid", "value", "sum"])
    sub = rng.randint(2, 3)
    grids = []
    for _ in range(sub):
        ks = rng.sample(_NAMES, rng.randint(1, 3))
        grids.append({k: sorted(rng.sample([0.01, 0.1, 0.5, 1, 2, 3, 5, 10, 100], rng.randint(2, 3))) for k in ks})
    key = rng.choice(sorted({k for g in grids for k in g}))
    total = sum(math.prod(len(v) for v in g.values()) for g in grids)
    q = min(total, rng.randint(5, 8))
    setup = (f"grids = {grids!r}\nidx = np.random.default_rng({s}).integers(0, {total}, {q})\n"
             f"key = {key!r}\nreport = {report!r}")
    tests = [f"g = np.random.default_rng({s + i})\nnames = {_NAMES!r}\n"
             "grids = [{str(nm): [float(v) for v in np.round(g.uniform(0, 10, g.integers(2, 20)), 3)] "
             "for nm in g.choice(names, g.integers(5, 11), replace=False)} for _ in range(4)]\n"
             "total = sum(int(np.prod([len(v) for v in gr.values()])) for gr in grids)\n"
             "idx = g.integers(0, total, 200000)\nkey = 'gamma'\n"
             f"report = {report!r}" for i in (1, 2)]
    sol = ("sizes = [int(np.prod([len(v) for v in gr.values()])) for gr in grids]\nstarts = np.cumsum([0] + sizes)\n"
           "which = np.searchsorted(starts, idx, side='right') - 1\n"
           + {"subgrid": "result = which",
              "value": "result = np.full(len(idx), np.nan)\n",
              "sum": "result = np.zeros(len(idx))\n"}[report])
    if report != "subgrid":
        sol += ("for w, gr in enumerate(grids):\n    rows = np.flatnonzero(which == w)\n    rest = idx[rows] - starts[w]\n"
                "    for name in sorted(gr)[::-1]:\n        vals = np.array(gr[name], dtype=float)\n"
                "        rest, off = np.divmod(rest, len(vals))\n")
        sol += ("        if name == key:\n            result[rows] = vals[off]" if report == "value"
                else "        result[rows] += vals[off]")
    want = {"subgrid": "the index of the sub-grid (dict) that candidate `idx[i]` comes from (an int array)",
            "value": "the value of the hyperparameter named `key` in candidate `idx[i]`, or `np.nan` if that "
                     "candidate's sub-grid has no such key (a float array)",
            "sum": "the sum of all hyperparameter values of candidate `idx[i]` (a float array)"}[report]
    return challenge(
        "Indexing a huge parameter grid",
        "`ParameterGrid(grids)` (what `GridSearchCV` iterates over) enumerates the sub-grids in order; within a "
        "sub-grid the keys are taken in **sorted** order and the **last** key varies fastest, like the digits of a "
        "mixed-radix number. `ParameterGrid(grids)[i]` returns candidate `i` without listing the others. For every "
        f"query `idx[i]`, return {want}. (`report` says which: {report!r}.) Real grids are far too large to list "
        "(10^12 candidates and more).", setup, sol, tests, "≤ 4 sub-grids of ≤ 10 keys with ≤ 20 values each, ≤ 2·10^5 "
        "queries", speed="O(queries · keys), vectorized", tol=1e-9)


@template(M, "challenge")
def ridge_leave_one_out(rng):
    n, d, s = rng.randint(8, 13), rng.randint(2, 3), seeded(rng)
    q = rng.randint(3, 5)
    report = rng.choice(["loo_mse", "best_alpha", "best_rmse"])
    setup = (f"g = np.random.default_rng({s})\nX = g.normal(size=({n}, {d})).round(2)\n"
             f"y = (X @ g.normal(0, 2, {d}) + g.normal(0, 1, {n})).round(2)\n"
             f"alphas = np.round(10.0 ** g.uniform(-2, 2, {q}), 3)\nreport = {report!r}")
    tests = [f"g = np.random.default_rng({s + i})\nX = g.normal(size=(20000, 10))\n"
             "y = X @ g.normal(0, 2, 10) + g.normal(0, 3, 20000)\nalphas = 10.0 ** g.uniform(-2, 4, 40)\n"
             f"report = {report!r}" for i in (1, 2)]
    sol = ("U, sv, _ = np.linalg.svd(X, full_matrices=False)\nUy, U2 = U.T @ y, U ** 2\nmse = []\nfor a in alphas:\n"
           "    f = sv ** 2 / (sv ** 2 + a)\n    res = (y - U @ (f * Uy)) / (1 - U2 @ f)\n    mse.append((res ** 2).mean())\n"
           "mse = np.array(mse)\n"
           + {"loo_mse": "result = mse", "best_alpha": "result = float(alphas[int(np.argmin(mse))])",
              "best_rmse": "result = float(np.sqrt(mse.min()))"}[report])
    want = {"loo_mse": "the leave-one-out MSE for every alpha (a float array, in the order of `alphas`)",
            "best_alpha": "the alpha with the smallest leave-one-out MSE (a float; the first one on ties)",
            "best_rmse": "the smallest leave-one-out RMSE over the alphas (a float)"}[report]
    return challenge(
        "Leave-one-out for every alpha",
        "`RidgeCV` picks `alpha` by leave-one-out without refitting n times. For `Ridge(alpha=a, fit_intercept=False)` "
        "the predictions are ŷ = H·y with the hat matrix H = X (XᵀX + a·I)⁻¹ Xᵀ, and the leave-one-out residual of row "
        "i is (y_i − ŷ_i) / (1 − H_ii). With one SVD of `X`, H's diagonal and ŷ cost O(n·d) per alpha. Return "
        f"{want}. (`report` says which: {report!r}.)", setup, sol, tests,
        "n ≤ 2·10^4 rows, d ≤ 10 features, ≤ 40 alphas", speed="O(n·d² + n·d·(number of alphas))", tol=1e-6)


@template(M, "challenge")
def successive_halving_rounds(rng):
    n, s = rng.randint(7, 12), seeded(rng)
    f = rng.choice([2, 3])
    r0 = rng.choice([10, 20, 50])
    report = rng.choice(["winner", "budget", "rounds"])
    setup = (f"g = np.random.default_rng({s})\na = g.uniform(0.6, 0.95, {n}).round(4)\nb = g.uniform(0, 8, {n}).round(4)\n"
             f"factor, r0 = {f}, {r0}\nreport = {report!r}")
    tests = [f"g = np.random.default_rng({s + i})\na = g.uniform(0.6, 0.95, 200000)\nb = g.uniform(0, 8, 200000)\n"
             f"factor, r0 = {f}, {r0}\nreport = {report!r}" for i in (1, 2)]
    sol = ("alive = np.arange(len(a))\nrounds = np.zeros(len(a), dtype=int)\nbudget, r = 0, r0\nwhile len(alive) > 1:\n"
           "    rounds[alive] += 1\n    budget += len(alive) * r\n    sc = a[alive] - b[alive] / r\n"
           "    keep = -(-len(alive) // factor)\n    alive = alive[np.lexsort((alive, -sc))[:keep]]\n    r *= factor\n"
           + {"winner": "result = int(alive[0])", "budget": "result = int(budget)", "rounds": "result = rounds"}[report])
    want = {"winner": "the index of the winning candidate (an int)",
            "budget": "the total budget spent: the sum over rounds of (candidates evaluated × resources per candidate), "
                      "an int",
            "rounds": "for every candidate, the number of rounds in which it was evaluated (an int array)"}[report]
    return challenge(
        "Successive halving",
        "`HalvingGridSearchCV` races candidates: candidate `i` evaluated with `r` resources scores "
        "`a[i] - b[i] / r` (more resources, better score). Round 0 evaluates all candidates with `r = r0`. Each round "
        "keeps the best `ceil(m / factor)` of its `m` candidates (higher score first, smaller index on ties), and the "
        "next round multiplies `r` by `factor`. The race ends as soon as a round leaves a single candidate (that "
        f"candidate is not evaluated again). Return {want}. (`report` says which: {report!r}.)", setup, sol, tests,
        "≤ 2·10^5 candidates, factor ∈ {2, 3}", speed="O(n log n): sort each round, the rounds shrink geometrically")
