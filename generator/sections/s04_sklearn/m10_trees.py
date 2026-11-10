"""4.10 Trees & ensembles: decision trees, forests, boosting, bagging, voting and stacking."""

from ._common import P, catalog, challenge, lit, seeded, synth_clf, synth_reg, table, template

M = "4.10"
SPLIT = ("from sklearn.model_selection import train_test_split\n"
         "X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)")
SPLIT_DESC = "`X_tr, X_te, y_tr, y_te` is a 70/30 train/test split of them"

_LOAD = {"cancer": ("load_breast_cancer", "569 breast-tumour samples (30 cell measurements)",
                    "the diagnosis (0 malignant, 1 benign)", 2, 30),
         "wine": ("load_wine", "178 wines (13 chemical measurements)", "the cultivar (0, 1, 2)", 3, 13),
         "iris": ("load_iris", "150 iris flowers (4 measurements in cm)", "the species (0, 1, 2)", 3, 4)}


def _clf(rng, allow=("cancer", "wine", "iris", "synth"), n=(150, 400), d=(4, 8), classes=(2, 3)):
    """(setup, desc, n_classes, n_features): X, y and the feature `names`."""
    kind = rng.choice(allow)
    if kind == "synth":
        dd = synth_clf(rng, n=n, d=d, classes=classes)
        names = [f"x{i}" for i in range(dd.d)]
        return (f"{dd.setup}\nnames = {lit(names)}", f"{dd.desc}; `names` lists the feature names", dd.n_classes,
                dd.d)
    fn, what, target, k, d_ = _LOAD[kind]
    return (f"from sklearn.datasets import {fn}\ndata = {fn}()\n"
            f"order = np.random.default_rng({seeded(rng)}).permutation(len(data.target))\n"
            "X, y = data.data[order], data.target[order]\nnames = list(data.feature_names)",
            f"`X` holds {what} with rows shuffled, `y` {target}, and `names` the feature names", k, d_)


def _binary(rng):
    return _clf(rng, allow=("cancer", "synth"), classes=2)


def _reg(rng, allow=("diabetes", "synth", "table")):
    """(setup, desc, n_features): a regression X, y and `names`."""
    kind = rng.choice(allow)
    if kind == "diabetes":
        return ("from sklearn.datasets import load_diabetes\ndata = load_diabetes()\n"
                f"order = np.random.default_rng({seeded(rng)}).permutation(len(data.target))\n"
                "X, y = data.data[order], data.target[order]\nnames = list(data.feature_names)",
                "`X` holds 442 diabetes patients (10 standardized measurements, rows shuffled), `y` the disease "
                "progression a year later, and `names` the feature names", 10)
    if kind == "synth":
        dd = synth_reg(rng, n=(150, 300), d=(3, 7))
        return (f"{dd.setup}\nnames = {lit([f'x{i}' for i in range(dd.d)])}",
                f"{dd.desc}; `names` lists the feature names", dd.d)
    t = table(rng, kind="reg", numeric_only=True, n=(150, 300))
    return (t.setup + "\nX, names = df.to_numpy(), list(df.columns)",
            f"{t.desc}; `X = df.to_numpy()` and `names` lists the columns", len(t.num))


# ================================================================= lite

@template(M, "lite")
def tree_fitted_attributes(rng):
    setup, desc, k, d = _clf(rng)
    depth, leaf, n = rng.randint(2, 4), rng.choice([1, 3, 5, 10]), rng.randint(2, 5)
    ctor = f"DecisionTreeClassifier(max_depth={depth}, min_samples_leaf={leaf}, random_state=0)"
    fit = f"from sklearn.tree import DecisionTreeClassifier\ntree = {ctor}.fit(X, y)\n"
    head = f"Fit `tree = {ctor}` (from `sklearn.tree`) on `X, y` and return"
    items = [
        ("get_depth", f"{head} the depth of the learned tree (`get_depth()`; at most {depth}).",
         fit + "result = tree.get_depth()"),
        ("get_n_leaves", f"{head} its number of leaves (`get_n_leaves()`).", fit + "result = tree.get_n_leaves()"),
        ("feature_importances_", f"{head} the name (from `names`) of the feature with the largest "
                                 f"`feature_importances_`.", fit + "result = names[tree.feature_importances_.argmax()]"),
        ("predict_proba", f"{head} `predict_proba` for the first {n} rows (a tree's probabilities are the class "
                          f"fractions in the leaf a row lands in).", fit + f"result = tree.predict_proba(X[:{n}])"),
        ("apply", f"{head} the id of the leaf each of the first {n} rows lands in (`apply`).",
         fit + f"result = tree.apply(X[:{n}])"),
        ("export_text", f"{head} its rules as text: `export_text(tree, feature_names=names)` (also from "
                        f"`sklearn.tree`).",
         f"from sklearn.tree import DecisionTreeClassifier, export_text\ntree = {ctor}.fit(X, y)\n"
         "result = export_text(tree, feature_names=names)"),
        ("score", f"{head} its training accuracy.", fit + "result = tree.score(X, y)"),
        ("node_count", f"{head} its total number of nodes, splits and leaves together (`tree.tree_.node_count`).",
         fit + "result = tree.tree_.node_count"),
        ("criterion", f"Fit `DecisionTreeClassifier(max_depth={depth}, criterion='entropy', random_state=0)` on `X, "
                      f"y` (splits chosen by information gain instead of Gini) and return its number of leaves.",
         f"from sklearn.tree import DecisionTreeClassifier\n"
         f"result = DecisionTreeClassifier(max_depth={depth}, criterion='entropy', random_state=0).fit(X, y).get_n_leaves()"),
        ("n_node_samples", f"{head} the smallest number of training rows in any leaf (leaves have "
                           f"`tree_.children_left == -1`; `tree_.n_node_samples` counts rows per node). It is never "
                           f"below `min_samples_leaf={leaf}`.",
         fit + "t = tree.tree_\nresult = int(t.n_node_samples[t.children_left == -1].min())"),
    ]
    return catalog(rng, items, setup, f"{desc}.", tol=1e-9)


@template(M, "lite")
def forest_fitted_attributes(rng):
    setup, desc, k, d = _clf(rng)
    m, rs, n = rng.choice([10, 15, 20, 30]), rng.randint(0, 9), rng.randint(2, 4)
    md = rng.randint(2, 4)
    ctor = f"RandomForestClassifier(n_estimators={m}, random_state={rs})"
    imp = "from sklearn.ensemble import RandomForestClassifier\n"
    fit = f"{imp}rf = {ctor}.fit(X, y)\n"
    head = f"Fit `rf = {ctor}` (from `sklearn.ensemble`) on `X, y` and return"
    items = [
        ("estimators_", f"{head} how many trees it holds (`len(rf.estimators_)`).", fit + "result = len(rf.estimators_)"),
        ("oob_score_", f"Fit `RandomForestClassifier(n_estimators={max(m, 30)}, oob_score=True, random_state={rs})` "
                       f"on `X, y` and return its out-of-bag accuracy `oob_score_` (each tree is scored on the rows "
                       f"its bootstrap sample left out).",
         f"{imp}rf = RandomForestClassifier(n_estimators={max(m, 30)}, oob_score=True, random_state={rs}).fit(X, y)\n"
         "result = rf.oob_score_"),
        ("feature_importances_", f"{head} the names of its 3 most important features (by `feature_importances_`), "
                                 f"most important first.",
         fit + "result = [names[i] for i in rf.feature_importances_.argsort()[::-1][:3]]"),
        ("feature_importances_", f"{head} the sum of its `feature_importances_` (they are normalized).",
         fit + "result = rf.feature_importances_.sum()"),
        ("predict_proba", f"{head} `predict_proba` for the first {n} rows: the average of the trees' probabilities.",
         fit + f"result = rf.predict_proba(X[:{n}])"),
        ("get_depth", f"{head} the list of its trees' depths (`get_depth()` of every tree in `estimators_`).",
         fit + "result = [t.get_depth() for t in rf.estimators_]"),
        ("max_depth", f"Fit `RandomForestClassifier(n_estimators={m}, max_depth={md}, random_state={rs})` on `X, y` "
                      f"and return its training accuracy.",
         f"{imp}result = RandomForestClassifier(n_estimators={m}, max_depth={md}, random_state={rs}).fit(X, y).score(X, y)"),
        ("max_features", f"Fit `RandomForestClassifier(n_estimators={m}, max_features=1, random_state={rs})` (every "
                         f"split looks at one random feature) on `X, y` and return the number of leaves of its first "
                         f"tree (`estimators_[0]`).",
         f"{imp}rf = RandomForestClassifier(n_estimators={m}, max_features=1, random_state={rs}).fit(X, y)\n"
         "result = rf.estimators_[0].get_n_leaves()"),
    ]
    return catalog(rng, items, setup, f"{desc}.", tol=1e-9)


@template(M, "lite")
def ensemble_test_scores(rng):
    setup, desc, k, d = _clf(rng, n=(150, 300))
    m, lr, md = rng.choice([10, 20, 30]), rng.choice([0.1, 0.3, 0.5]), rng.randint(2, 5)
    E = "from sklearn.ensemble import "
    T = "from sklearn.tree import DecisionTreeClassifier\n"
    fit = ".fit(X_tr, y_tr)\nresult = model.score(X_te, y_te)"
    items = [
        ("ExtraTreesClassifier", f"`ExtraTreesClassifier(n_estimators={m}, random_state=0)` (a forest that also "
                                 f"draws the split thresholds at random)",
         f"{E}ExtraTreesClassifier\nmodel = ExtraTreesClassifier(n_estimators={m}, random_state=0)" + fit),
        ("GradientBoostingClassifier", f"`GradientBoostingClassifier(n_estimators={m}, learning_rate={lr}, "
                                       f"max_depth=2, random_state=0)`",
         f"{E}GradientBoostingClassifier\nmodel = GradientBoostingClassifier(n_estimators={m}, learning_rate={lr}, "
         f"max_depth=2, random_state=0)" + fit),
        ("AdaBoostClassifier", f"`AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), "
                               f"n_estimators={m}, random_state=0)` (boosted stumps)",
         f"{E}AdaBoostClassifier\n{T}model = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), "
         f"n_estimators={m}, random_state=0)" + fit),
        ("BaggingClassifier", f"`BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators={m}, "
                              f"max_samples=0.8, random_state=0)` (each tree sees a bootstrap sample of 80% of the "
                              f"rows)",
         f"{E}BaggingClassifier\n{T}model = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators={m}, "
         f"max_samples=0.8, random_state=0)" + fit),
        ("HistGradientBoostingClassifier", f"`HistGradientBoostingClassifier(max_iter={m}, learning_rate={lr}, "
                                           f"random_state=0)`",
         f"{E}HistGradientBoostingClassifier\nmodel = HistGradientBoostingClassifier(max_iter={m}, learning_rate={lr}, "
         f"random_state=0)" + fit),
        ("RandomForestClassifier", f"`RandomForestClassifier(n_estimators={m}, random_state=0)`",
         f"{E}RandomForestClassifier\nmodel = RandomForestClassifier(n_estimators={m}, random_state=0)" + fit),
        ("DecisionTreeClassifier", f"`DecisionTreeClassifier(max_depth={md}, random_state=0)`",
         f"{T}model = DecisionTreeClassifier(max_depth={md}, random_state=0)" + fit),
    ]
    p = catalog(rng, items, setup + "\n" + SPLIT, f"{desc}; {SPLIT_DESC}. Fit")
    p.prompt += " on the training split and return its test accuracy (`score`)."
    return p


@template(M, "lite")
def ensemble_regressors(rng):
    setup, desc, d = _reg(rng)
    m, lr, n = rng.choice([10, 20, 30]), rng.choice([0.1, 0.3]), rng.randint(2, 4)
    E = "from sklearn.ensemble import "
    items = [
        ("RandomForestRegressor", f"Fit `RandomForestRegressor(n_estimators={m}, random_state=0)` on the training "
                                  f"split and return its test R² (`score`).",
         f"{E}RandomForestRegressor\nresult = RandomForestRegressor(n_estimators={m}, random_state=0).fit(X_tr, y_tr)"
         ".score(X_te, y_te)"),
        ("HistGradientBoostingRegressor", f"Fit `HistGradientBoostingRegressor(max_iter={m}, random_state=0)` on the "
                                          f"training split and return its test R².",
         f"{E}HistGradientBoostingRegressor\nresult = HistGradientBoostingRegressor(max_iter={m}, random_state=0)"
         ".fit(X_tr, y_tr).score(X_te, y_te)"),
        ("loss", f"Fit `HistGradientBoostingRegressor(loss='absolute_error', max_iter={m}, random_state=0)` (it "
                 f"targets the median, so outliers pull less) on the training split and return its predictions for "
                 f"the first {n} test rows.",
         f"{E}HistGradientBoostingRegressor\nhgb = HistGradientBoostingRegressor(loss='absolute_error', max_iter={m}, "
         f"random_state=0).fit(X_tr, y_tr)\nresult = hgb.predict(X_te[:{n}])"),
        ("feature_importances_", f"Fit `RandomForestRegressor(n_estimators={m}, random_state=0)` on the training split "
                                 f"and return the name of its most important feature.",
         f"{E}RandomForestRegressor\nrf = RandomForestRegressor(n_estimators={m}, random_state=0).fit(X_tr, y_tr)\n"
         "result = names[rf.feature_importances_.argmax()]"),
        ("DecisionTreeRegressor", "Fit `DecisionTreeRegressor(max_depth=2, random_state=0)` (from `sklearn.tree`) on "
                                  "the training split and return how many **distinct** values it predicts on `X_te` "
                                  "(one per leaf it reaches, so at most 4).",
         "from sklearn.tree import DecisionTreeRegressor\n"
         "tree = DecisionTreeRegressor(max_depth=2, random_state=0).fit(X_tr, y_tr)\n"
         "result = len(np.unique(tree.predict(X_te)))"),
        ("GradientBoostingRegressor", f"Fit `GradientBoostingRegressor(n_estimators={m}, learning_rate={lr}, "
                                      f"max_depth=2, random_state=0)` on the training split and return its test R².",
         f"{E}GradientBoostingRegressor\nresult = GradientBoostingRegressor(n_estimators={m}, learning_rate={lr}, "
         "max_depth=2, random_state=0).fit(X_tr, y_tr).score(X_te, y_te)"),
        ("ExtraTreesRegressor", f"Fit `ExtraTreesRegressor(n_estimators={m}, random_state=0)` on the training split and "
                                f"return its test R².",
         f"{E}ExtraTreesRegressor\nresult = ExtraTreesRegressor(n_estimators={m}, random_state=0).fit(X_tr, y_tr)"
         ".score(X_te, y_te)"),
    ]
    return catalog(rng, items, setup + "\n" + SPLIT, f"{desc}; {SPLIT_DESC}.", tol=1e-6)


# ================================================================= easy

@template(M, "easy")
def stump_readout(rng):
    task = rng.choice(["split", "left_count", "reg_means"])
    if task == "reg_means":
        setup, desc, d = _reg(rng)
        return P(f"{desc}. Fit `stump = DecisionTreeRegressor(max_depth=1, random_state=0)` (one split) on `X, y`. A "
                 f"regression leaf predicts the **mean** target of its training rows. Compute by hand the mean of `y` "
                 f"on each side of the root split (rows with `X[:, f] <= thr` go left, where `f = "
                 f"stump.tree_.feature[0]` and `thr = stump.tree_.threshold[0]`) and return `(left_mean, right_mean)`; "
                 f"they equal the stump's two predictions.", setup,
                 "from sklearn.tree import DecisionTreeRegressor\nstump = DecisionTreeRegressor(max_depth=1, random_state=0)"
                 ".fit(X, y)\nf, thr = stump.tree_.feature[0], stump.tree_.threshold[0]\nleft = X[:, f] <= thr\n"
                 "result = (y[left].mean(), y[~left].mean())", require=["DecisionTreeRegressor", "threshold"], tol=1e-6)
    setup, desc, k, d = _clf(rng)
    head = (f"{desc}. Fit `stump = DecisionTreeClassifier(max_depth=1, random_state=0)` (a *stump*: one split) on "
            f"`X, y`. The root's split lives in `stump.tree_`: `feature[0]` is the column index and `threshold[0]` the "
            f"cut (rows with a value `<=` it go left).")
    sol = "from sklearn.tree import DecisionTreeClassifier\nstump = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X, y)\n"
    if task == "split":
        return P(f"{head} Return `(feature_name, threshold)` using `names`.", setup,
                 sol + "result = (names[stump.tree_.feature[0]], stump.tree_.threshold[0])",
                 require=["DecisionTreeClassifier", "threshold"])
    return P(f"{head} Count **by hand** how many training rows go left and return `(count, "
             f"stump.tree_.n_node_samples[1])` (node 1 is the root's left child, so the two agree).", setup,
             sol + "f, thr = stump.tree_.feature[0], stump.tree_.threshold[0]\n"
                   "result = (int((X[:, f] <= thr).sum()), int(stump.tree_.n_node_samples[1]))",
             require=["DecisionTreeClassifier", "n_node_samples"])


@template(M, "easy")
def impurity_by_hand(rng):
    setup, desc, k, d = _clf(rng)
    task = rng.choice(["gini", "entropy", "decrease"])
    if task == "gini":
        return P(f"{desc}. Fit `tree = DecisionTreeClassifier(max_depth=2, random_state=0)` on `X, y`. The root's "
                 f"`tree_.impurity[0]` is the Gini impurity of all labels: 1 − Σ p_k² over the class fractions p_k. "
                 f"Compute it by hand from `y` (`np.bincount`) and return `(by_hand, tree.tree_.impurity[0])`.", setup,
                 "from sklearn.tree import DecisionTreeClassifier\ntree = DecisionTreeClassifier(max_depth=2, random_state=0)"
                 ".fit(X, y)\np = np.bincount(y) / len(y)\nresult = (1 - (p ** 2).sum(), tree.tree_.impurity[0])",
                 require=["DecisionTreeClassifier", "impurity"], tol=1e-9)
    if task == "entropy":
        return P(f"{desc}. Fit `tree = DecisionTreeClassifier(max_depth=2, criterion='entropy', random_state=0)` on "
                 f"`X, y`. With this criterion the root's `tree_.impurity[0]` is the entropy of the labels in **bits**: "
                 f"−Σ p_k·log2(p_k). Compute it by hand and return `(by_hand, tree.tree_.impurity[0])`.", setup,
                 "from sklearn.tree import DecisionTreeClassifier\n"
                 "tree = DecisionTreeClassifier(max_depth=2, criterion='entropy', random_state=0).fit(X, y)\n"
                 "p = np.bincount(y) / len(y)\np = p[p > 0]\nresult = (-(p * np.log2(p)).sum(), tree.tree_.impurity[0])",
                 require=["criterion", "impurity"], tol=1e-6)
    return P(f"{desc}. Fit `tree = DecisionTreeClassifier(max_depth=2, random_state=0)` on `X, y`. How much did the "
             f"root split lower the impurity? With `t = tree.tree_`, `l, r = t.children_left[0], t.children_right[0]` "
             f"and the per-node `t.impurity` and `t.n_node_samples`, return the root impurity minus the "
             f"sample-weighted impurity of its two children.", setup,
             "from sklearn.tree import DecisionTreeClassifier\nt = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y).tree_\n"
             "l, r = t.children_left[0], t.children_right[0]\nn = t.n_node_samples\n"
             "result = t.impurity[0] - (n[l] * t.impurity[l] + n[r] * t.impurity[r]) / n[0]",
             require=["children_left", "impurity"], tol=1e-9)


@template(M, "easy")
def forest_average_by_hand(rng):
    m, n = rng.choice([5, 10, 20]), rng.randint(3, 6)
    task = rng.choice(["proba", "reg", "votes"])
    if task == "reg":
        setup, desc, d = _reg(rng)
        return P(f"{desc}. Fit `rf = RandomForestRegressor(n_estimators={m}, random_state=0)` on `X, y`. A forest "
                 f"predicts the **mean** of its trees' predictions. Rebuild `rf.predict(X[:{n}])` by averaging the "
                 f"predictions of the trees in `rf.estimators_`, and return your array.", setup,
                 f"from sklearn.ensemble import RandomForestRegressor\nrf = RandomForestRegressor(n_estimators={m}, "
                 f"random_state=0).fit(X, y)\nresult = np.mean([t.predict(X[:{n}]) for t in rf.estimators_], axis=0)",
                 require=["estimators_"], tol=1e-6)
    setup, desc, k, d = _clf(rng)
    sol = (f"from sklearn.ensemble import RandomForestClassifier\nrf = RandomForestClassifier(n_estimators={m}, "
           f"random_state=0).fit(X, y)\n")
    if task == "proba":
        return P(f"{desc}. Fit `rf = RandomForestClassifier(n_estimators={m}, random_state=0)` on `X, y`. A forest's "
                 f"`predict_proba` is the **average** of its trees' `predict_proba`. Rebuild it for the first {n} rows "
                 f"by averaging over `rf.estimators_` and return your array.", setup,
                 sol + f"result = np.mean([t.predict_proba(X[:{n}]) for t in rf.estimators_], axis=0)",
                 require=["estimators_"], tol=1e-9)
    c = rng.randrange(k)
    return P(f"{desc}. Fit `rf = RandomForestClassifier(n_estimators={m}, random_state=0)` on `X, y`. For each of the "
             f"first {n} rows, return the fraction of trees whose own `predict` says class {c} (a *hard* vote; the "
             f"forest itself averages probabilities). The labels are 0, 1, ... so you can compare the trees' "
             f"predictions with {c} directly.", setup,
             sol + f"result = np.mean([t.predict(X[:{n}]) == {c} for t in rf.estimators_], axis=0)",
             require=["estimators_"], tol=1e-9)


@template(M, "easy")
def tree_rules_text(rng):
    setup, desc, k, d = _clf(rng, allow=("iris", "wine", "synth"))
    task = rng.choice(["decimals", "count", "weights", "truncate", "used"])
    fit = ("from sklearn.tree import DecisionTreeClassifier, export_text\n"
           "tree = DecisionTreeClassifier(max_depth={}, random_state=0).fit(X, y)\n")
    head = f"{desc}. Fit `tree = DecisionTreeClassifier(max_depth={{}}, random_state=0)` on `X, y`."
    if task == "decimals":
        return P(head.format(2) + " Print-ready rules: return `export_text(tree, feature_names=names, decimals=1)` (from "
                                  "`sklearn.tree`; thresholds shown with one decimal).", setup,
                 fit.format(2) + "result = export_text(tree, feature_names=names, decimals=1)", require=["export_text"])
    if task == "count":
        return P(head.format(3) + " Every leaf prints one `class:` line in `export_text(tree, feature_names=names)`. "
                                  "Return `(number of such lines, tree.get_n_leaves())`.", setup,
                 fit.format(3) + "text = export_text(tree, feature_names=names)\n"
                                 "result = (text.count('class:'), tree.get_n_leaves())",
                 require=["export_text", "get_n_leaves"])
    if task == "weights":
        return P(head.format(2) + " Return `export_text(tree, feature_names=names, show_weights=True)`: each leaf then "
                                  "shows how many training rows of each class it holds.", setup,
                 fit.format(2) + "result = export_text(tree, feature_names=names, show_weights=True)",
                 require=["export_text"])
    if task == "truncate":
        return P(head.format(4) + " Print only the top of the tree: return `export_text(tree, feature_names=names, "
                                  "max_depth=1)` (deeper parts show as truncated branches).", setup,
                 fit.format(4) + "result = export_text(tree, feature_names=names, max_depth=1)", require=["export_text"])
    return P(head.format(3) + " Which features appear in its rules? `tree_.feature` holds each node's column index "
                              "(leaves hold -2). Return the sorted list of the names of the features it splits on.",
             setup, fit.format(3) + "result = sorted({names[f] for f in tree.tree_.feature if f >= 0})",
             require=["DecisionTreeClassifier", "feature"])


@template(M, "easy")
def depth_sweep(rng):
    setup, desc, k, d = _clf(rng)
    depths = sorted(rng.sample(range(1, 9), 4))
    task = rng.choice(["test", "best", "train", "leaves"])
    head = f"{desc}; {SPLIT_DESC}. For each `max_depth` in `{depths}`, fit `DecisionTreeClassifier(max_depth=..., random_state=0)` on the training split"
    sol = (f"from sklearn.tree import DecisionTreeClassifier\n"
           f"trees = [DecisionTreeClassifier(max_depth=k, random_state=0).fit(X_tr, y_tr) for k in {depths}]\n")
    setup = setup + "\n" + SPLIT
    if task == "test":
        return P(f"{head} and return the list of test accuracies.", setup,
                 sol + "result = [t.score(X_te, y_te) for t in trees]", require=["max_depth"])
    if task == "best":
        return P(f"{head} and return the depth with the best test accuracy (the smallest on ties).", setup,
                 sol + f"scores = [t.score(X_te, y_te) for t in trees]\nresult = {depths}[int(np.argmax(scores))]",
                 require=["max_depth"])
    if task == "train":
        return P(f"{head} and return `(train_accuracies, test_accuracies)` as two lists: deeper trees fit the "
                 f"training rows better, but not necessarily the test rows.", setup,
                 sol + "result = ([t.score(X_tr, y_tr) for t in trees], [t.score(X_te, y_te) for t in trees])",
                 require=["max_depth"])
    return P(f"{head} and return the list of their numbers of leaves.", setup,
             sol + "result = [t.get_n_leaves() for t in trees]", require=["max_depth", "get_n_leaves"])


@template(M, "easy")
def boosting_stages(rng):
    setup, desc, k, d = _clf(rng, n=(150, 300))
    m, lr = rng.choice([10, 15, 20]), rng.choice([0.1, 0.3, 0.5])
    task = rng.choice(["staged", "best", "ada"])
    setup += "\n" + SPLIT
    gb = (f"from sklearn.ensemble import GradientBoostingClassifier\ngb = GradientBoostingClassifier(n_estimators={m}, "
          f"learning_rate={lr}, max_depth=2, random_state=0).fit(X_tr, y_tr)\n")
    head = (f"{desc}; {SPLIT_DESC}. Fit `gb = GradientBoostingClassifier(n_estimators={m}, learning_rate={lr}, "
            f"max_depth=2, random_state=0)` on the training split. `gb.staged_predict(X_te)` yields the test "
            f"predictions after each boosting round.")
    if task == "staged":
        return P(f"{head} Return the list of test accuracies after rounds 1, 2, ..., {m}.", setup,
                 gb + "result = [float((p == y_te).mean()) for p in gb.staged_predict(X_te)]",
                 require=["staged_predict"])
    if task == "best":
        return P(f"{head} After which round (counting from 1) is the test accuracy highest? Return the earliest such "
                 f"round.", setup,
                 gb + "acc = [(p == y_te).mean() for p in gb.staged_predict(X_te)]\nresult = int(np.argmax(acc)) + 1",
                 require=["staged_predict"])
    return P(f"{desc}; {SPLIT_DESC}. Fit `ada = AdaBoostClassifier(n_estimators={m}, learning_rate={lr}, "
             f"random_state=0)` (boosted stumps) on the training split and return the list of test accuracies after "
             f"each round, from `ada.staged_score(X_te, y_te)`.", setup,
             f"from sklearn.ensemble import AdaBoostClassifier\nada = AdaBoostClassifier(n_estimators={m}, "
             f"learning_rate={lr}, random_state=0).fit(X_tr, y_tr)\nresult = list(ada.staged_score(X_te, y_te))",
             require=["AdaBoostClassifier", "staged_score"])


@template(M, "easy")
def bagging_vs_tree(rng):
    setup, desc, k, d = _clf(rng, n=(150, 300))
    m, ms = rng.choice([10, 20]), rng.choice([0.5, 0.7, 0.8, 1.0])
    task = rng.choice(["compare", "sizes", "distinct"])
    ctor = (f"BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators={m}, max_samples={ms}, "
            f"random_state=0)")
    sol = (f"from sklearn.ensemble import BaggingClassifier\nfrom sklearn.tree import DecisionTreeClassifier\n"
           f"bag = {ctor}.fit(X_tr, y_tr)\n")
    head = f"{desc}; {SPLIT_DESC}. Fit `bag = {ctor}` on the training split."
    setup += "\n" + SPLIT
    if task == "compare":
        return P(f"{head} Also fit a single `DecisionTreeClassifier(random_state=0)`. Return `(tree_test_accuracy, "
                 f"bagging_test_accuracy)`: averaging many deep trees usually reduces their variance.", setup,
                 sol + "tree = DecisionTreeClassifier(random_state=0).fit(X_tr, y_tr)\n"
                       "result = (tree.score(X_te, y_te), bag.score(X_te, y_te))",
                 require=["BaggingClassifier", "estimator"])
    if task == "sizes":
        return P(f"{head} `bag.estimators_samples_` lists the training-row indices each tree was fitted on. Return the "
                 f"list of their lengths (each is `max_samples` × the number of training rows).", setup,
                 sol + "result = [len(s) for s in bag.estimators_samples_]",
                 require=["estimators_samples_", "max_samples"])
    return P(f"{head} Bootstrap samples repeat rows: return the number of **distinct** training rows each of the first "
             f"3 trees saw (`np.unique` of `bag.estimators_samples_[i]`).", setup,
             sol + "result = [len(np.unique(s)) for s in bag.estimators_samples_[:3]]",
             require=["estimators_samples_"])


@template(M, "easy")
def importance_tasks(rng):
    setup, desc, k, d = _clf(rng)
    m = rng.choice([20, 30])
    rf = f"from sklearn.ensemble import RandomForestClassifier\nrf = RandomForestClassifier(n_estimators={m}, random_state=0).fit(X, y)\n"
    items = [
        ("feature_importances_", f"Fit `RandomForestClassifier(n_estimators={m}, random_state=0)` on `X, y` and return a "
                                 f"dict {{name: importance}} of the features whose importance is above average "
                                 f"(greater than 1 / number of features).",
         rf + "imp = rf.feature_importances_\nresult = {n: v for n, v in zip(names, imp) if v > 1 / len(imp)}"),
        ("ExtraTreesClassifier", f"Fit `RandomForestClassifier(n_estimators={m}, random_state=0)` and "
                                 f"`ExtraTreesClassifier(n_estimators={m}, random_state=0)` on `X, y`. Return the name "
                                 f"of each one's most important feature, as `(rf_top, et_top)`.",
         rf + f"from sklearn.ensemble import ExtraTreesClassifier\net = ExtraTreesClassifier(n_estimators={m}, "
              f"random_state=0).fit(X, y)\nresult = (names[rf.feature_importances_.argmax()], "
              f"names[et.feature_importances_.argmax()])"),
        ("max_depth", "Fit `DecisionTreeClassifier(max_depth=1, random_state=0)` on `X, y` and return its "
                      "`feature_importances_`: a stump splits on one feature, so one entry is 1 and the rest 0.",
         "from sklearn.tree import DecisionTreeClassifier\n"
         "result = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X, y).feature_importances_"),
        ("GradientBoostingClassifier", "Fit `GradientBoostingClassifier(n_estimators=10, max_depth=2, random_state=0)` "
                                       "on `X, y` and return the names of the features it **never** split on (zero "
                                       "importance), in column order.",
         "from sklearn.ensemble import GradientBoostingClassifier\ngb = GradientBoostingClassifier(n_estimators=10, "
         "max_depth=2, random_state=0).fit(X, y)\nresult = [n for n, v in zip(names, gb.feature_importances_) if v == 0]"),
        ("feature_importances_", "Fit `DecisionTreeClassifier(max_depth=3, random_state=0)` on `X, y` and return how "
                                 "many features it actually uses (non-zero importance).",
         "from sklearn.tree import DecisionTreeClassifier\n"
         "result = int((DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y).feature_importances_ > 0).sum())"),
        ("argsort", f"Fit `RandomForestClassifier(n_estimators={m}, random_state=0)` on `X, y` and return the column "
                    f"indices ordered from most to least important (`argsort()[::-1]`).",
         rf + "result = rf.feature_importances_.argsort()[::-1]"),
    ]
    return catalog(rng, items, setup, f"{desc}.", tol=1e-9)


@template(M, "easy")
def hgb_missing_values(rng):
    t = table(rng, missing=2, numeric_only=True, n=(150, 300))
    clf = t.kind == "clf"
    cls = "HistGradientBoostingClassifier" if clf else "HistGradientBoostingRegressor"
    m = rng.choice([20, 30, 40])
    task = rng.choice(["score", "nan_rows", "forest"])
    imp = f"from sklearn.ensemble import {cls}\n"
    head = (f"{t.desc}. Columns {' and '.join(f'`{c}`' for c in t.missing)} have missing values (NaN), and "
            f"`{cls}` (from `sklearn.ensemble`) handles NaN natively: no imputer needed.")
    if task == "score":
        return P(f"{head} Fit `{cls}(max_iter={m}, random_state=0)` on the first 70% of the rows (`h = int(0.7 * "
                 f"len(df))`) and return its `score` on the rest.", t.setup,
                 imp + f"h = int(0.7 * len(df))\nmodel = {cls}(max_iter={m}, random_state=0).fit(df.iloc[:h], y[:h])\n"
                       "result = model.score(df.iloc[h:], y[h:])", require=[cls], tol=1e-6)
    if task == "nan_rows":
        return P(f"{head} Fit `{cls}(max_iter={m}, random_state=0)` on `df, y` and return its predictions for the rows "
                 f"that have at least one missing value (in row order).", t.setup,
                 imp + f"model = {cls}(max_iter={m}, random_state=0).fit(df, y)\n"
                       "result = model.predict(df[df.isna().any(axis=1)])", require=[cls], tol=1e-6)
    rf = "RandomForestClassifier" if clf else "RandomForestRegressor"
    return P(f"{head} Random forests accept NaN too (scikit-learn ≥ 1.4). Fit `{cls}(max_iter={m}, random_state=0)` and "
             f"`{rf}(n_estimators=20, random_state=0)` on the first 70% of the rows and return both test scores as "
             f"`(hgb, forest)`.", t.setup,
             f"from sklearn.ensemble import {cls}, {rf}\nh = int(0.7 * len(df))\nXa, ya, Xb, yb = df.iloc[:h], y[:h], "
             f"df.iloc[h:], y[h:]\nresult = ({cls}(max_iter={m}, random_state=0).fit(Xa, ya).score(Xb, yb), "
             f"{rf}(n_estimators=20, random_state=0).fit(Xa, ya).score(Xb, yb))", require=[cls, rf], tol=1e-6)


# ================================================================= medium

@template(M, "medium")
def ccp_alpha_path(rng):
    setup, desc, k, d = _clf(rng, allow=("cancer", "wine", "synth"))
    alphas = sorted(rng.sample([0.002, 0.005, 0.01, 0.02, 0.05], 3))
    path = ("from sklearn.tree import DecisionTreeClassifier\n"
            "path = DecisionTreeClassifier(random_state=0).cost_complexity_pruning_path(X_tr, y_tr)\n")
    items = [
        ("cost_complexity_pruning_path", "Compute `path = DecisionTreeClassifier(random_state=0)"
                                         ".cost_complexity_pruning_path(X_tr, y_tr)` and return how many candidate "
                                         "alphas it lists (`len(path.ccp_alphas)`).", path + "result = len(path.ccp_alphas)"),
        ("ccp_alphas", "Compute the pruning path of `DecisionTreeClassifier(random_state=0)` on the training split and "
                       "return its first 5 `ccp_alphas` (they increase: bigger alpha, more pruning).",
         path + "result = path.ccp_alphas[:5]"),
        ("ccp_alpha", "Compute the pruning path on the training split, then fit `DecisionTreeClassifier(ccp_alpha=a, "
                      "random_state=0)` with `a = path.ccp_alphas[-2]` (the last alpha only leaves the root) and "
                      "return its number of leaves.",
         path + "tree = DecisionTreeClassifier(ccp_alpha=path.ccp_alphas[-2], random_state=0).fit(X_tr, y_tr)\n"
                "result = tree.get_n_leaves()"),
        ("ccp_alpha", "Choose the pruning strength: for every alpha in the training split's pruning path, fit "
                      "`DecisionTreeClassifier(ccp_alpha=alpha, random_state=0)` on the training split and score it on "
                      "the test split. Return the alpha with the best test accuracy (on ties, the **largest** alpha: the "
                      "simpler tree).",
         path + "acc = np.array([DecisionTreeClassifier(ccp_alpha=a, random_state=0).fit(X_tr, y_tr).score(X_te, y_te) "
                "for a in path.ccp_alphas])\nresult = float(path.ccp_alphas[np.flatnonzero(acc == acc.max())[-1]])"),
        ("impurities", "`path.impurities[i]` is the total leaf impurity of the tree pruned with `ccp_alphas[i]`. A "
                       "fully pruned tree is a single leaf, so its total impurity is the root's. Return "
                       "`(path.impurities[-1], tree.tree_.impurity[0])` for the training split's path and an unpruned "
                       "`DecisionTreeClassifier(random_state=0)` fitted on the training split.",
         path + "tree = DecisionTreeClassifier(random_state=0).fit(X_tr, y_tr)\n"
                "result = (path.impurities[-1], tree.tree_.impurity[0])"),
        ("ccp_alpha", f"For `ccp_alpha` in `{[0.0] + alphas}`, fit `DecisionTreeClassifier(ccp_alpha=..., "
                      f"random_state=0)` on the training split and return the list of numbers of leaves.",
         "from sklearn.tree import DecisionTreeClassifier\n"
         f"result = [DecisionTreeClassifier(ccp_alpha=a, random_state=0).fit(X_tr, y_tr).get_n_leaves() for a in "
         f"{[0.0] + alphas}]"),
    ]
    return catalog(rng, items, setup + "\n" + SPLIT, f"{desc}; {SPLIT_DESC}. Cost-complexity pruning cuts back a "
                                                    f"fully grown tree.", tol=1e-9)


@template(M, "medium")
def oob_estimates(rng):
    task = rng.choice(["vs_test", "decision", "regressor", "curve"])
    m = rng.choice([30, 40, 50])
    if task == "regressor":
        setup, desc, d = _reg(rng)
        return P(f"{desc}; {SPLIT_DESC}. Fit `RandomForestRegressor(n_estimators={m}, oob_score=True, random_state=0)` "
                 f"on the training split. For a regressor `oob_score_` is an out-of-bag **R²**. Return `(oob_score_, "
                 f"test R²)`: the free OOB estimate should be close to the test score.", setup + "\n" + SPLIT,
                 f"from sklearn.ensemble import RandomForestRegressor\nrf = RandomForestRegressor(n_estimators={m}, "
                 f"oob_score=True, random_state=0).fit(X_tr, y_tr)\nresult = (rf.oob_score_, rf.score(X_te, y_te))",
                 require=["oob_score_"], tol=1e-6)
    setup, desc, k, d = _clf(rng)
    ctor = f"RandomForestClassifier(n_estimators={m}, oob_score=True, random_state=0)"
    sol = f"from sklearn.ensemble import RandomForestClassifier\nrf = {ctor}.fit(X_tr, y_tr)\n"
    head = f"{desc}; {SPLIT_DESC}. Fit `rf = {ctor}` on the training split."
    setup += "\n" + SPLIT
    if task == "vs_test":
        return P(f"{head} Return `(rf.oob_score_, test accuracy)`: the out-of-bag estimate needs no held-out data.",
                 setup, sol + "result = (rf.oob_score_, rf.score(X_te, y_te))", require=["oob_score_"], tol=1e-9)
    if task == "decision":
        n = rng.randint(3, 6)
        return P(f"{head} `rf.oob_decision_function_` gives each training row's class probabilities from only the "
                 f"trees that did **not** see it. Return its first {n} rows, and the OOB accuracy you get by taking "
                 f"the argmax of every row, as `(rows, accuracy)`.", setup,
                 sol + f"oob = rf.oob_decision_function_\nresult = (oob[:{n}], float((oob.argmax(axis=1) == y_tr).mean()))",
                 require=["oob_decision_function_"], tol=1e-9)
    sizes = [30, 60, 90]
    return P(f"{desc}; {SPLIT_DESC}. How does the out-of-bag accuracy change with more trees? For `n_estimators` in "
             f"`{sizes}`, fit `RandomForestClassifier(n_estimators=..., oob_score=True, random_state=0)` on the "
             f"training split and return the list of `oob_score_` values.", setup,
             "from sklearn.ensemble import RandomForestClassifier\n"
             f"result = [RandomForestClassifier(n_estimators=k, oob_score=True, random_state=0).fit(X_tr, y_tr).oob_score_ "
             f"for k in {sizes}]", require=["oob_score", "n_estimators"], tol=1e-9)


@template(M, "medium")
def early_stopping_rounds(rng):
    task = rng.choice(["hgb", "gb", "best", "hgb_reg"])
    lr, s, pat = rng.choice([0.1, 0.2, 0.3]), rng.randint(0, 9), rng.choice([3, 5, 8])
    if task == "hgb_reg":
        setup, desc, d = _reg(rng)
        return P(f"{desc}. Fit `HistGradientBoostingRegressor(max_iter=200, learning_rate={lr}, early_stopping=True, "
                 f"validation_fraction=0.2, n_iter_no_change={pat}, random_state={s})` on `X, y`: it holds out 20% of "
                 f"the rows and stops once {pat} rounds in a row fail to improve the validation loss. Return the number "
                 f"of boosting rounds it actually ran (`n_iter_`).", setup,
                 "from sklearn.ensemble import HistGradientBoostingRegressor\n"
                 f"result = HistGradientBoostingRegressor(max_iter=200, learning_rate={lr}, early_stopping=True, "
                 f"validation_fraction=0.2, n_iter_no_change={pat}, random_state={s}).fit(X, y).n_iter_",
                 require=["early_stopping", "n_iter_"])
    setup, desc, k, d = _clf(rng, n=(150, 350))
    hgb = (f"HistGradientBoostingClassifier(max_iter=200, learning_rate={lr}, early_stopping=True, "
           f"validation_fraction=0.2, n_iter_no_change={pat}, random_state={s})")
    if task == "hgb":
        return P(f"{desc}. Fit `{hgb}` on `X, y`: it holds out 20% of the rows and stops boosting once {pat} rounds in "
                 f"a row fail to improve the validation loss. Return the number of rounds it actually used "
                 f"(`n_iter_`).", setup,
                 f"from sklearn.ensemble import HistGradientBoostingClassifier\nresult = {hgb}.fit(X, y).n_iter_",
                 require=["early_stopping", "n_iter_"])
    if task == "best":
        return P(f"{desc}. Fit `hgb = {hgb}` on `X, y`. `hgb.validation_score_` records the validation score (the "
                 f"negative loss) before boosting and after every round. Return `(len(hgb.validation_score_), "
                 f"hgb.n_iter_, best_round)` where `best_round` is the index of the highest validation score.", setup,
                 f"from sklearn.ensemble import HistGradientBoostingClassifier\nhgb = {hgb}.fit(X, y)\n"
                 "result = (len(hgb.validation_score_), hgb.n_iter_, int(np.argmax(hgb.validation_score_)))",
                 require=["validation_score_"])
    gb = (f"GradientBoostingClassifier(n_estimators=200, learning_rate={lr}, max_depth=2, validation_fraction=0.2, "
          f"n_iter_no_change={pat}, random_state={s})")
    return P(f"{desc}. `GradientBoostingClassifier` stops early when you set `n_iter_no_change`. Fit `{gb}` on `X, y` "
             f"and return the number of trees it kept (`n_estimators_`, with a trailing underscore).", setup,
             f"from sklearn.ensemble import GradientBoostingClassifier\nresult = {gb}.fit(X, y).n_estimators_",
             require=["n_iter_no_change", "n_estimators_"])


@template(M, "medium")
def compare_ensembles_cv(rng):
    dd = synth_clf(rng, n=(150, 250), d=(4, 8), classes=2)
    m = rng.choice([15, 20])
    task = rng.choice(["means", "best", "vote_std"])
    models = (f"from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier\n"
              "from sklearn.linear_model import LogisticRegression\nfrom sklearn.model_selection import cross_val_score\n"
              "from sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\n"
              f"models = {{'rf': RandomForestClassifier(n_estimators={m}, random_state=0),\n"
              f"          'hgb': HistGradientBoostingClassifier(max_iter={m}, random_state=0),\n"
              "          'vote': VotingClassifier([('lr', make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),\n"
              f"                                   ('rf', RandomForestClassifier(n_estimators={m}, random_state=0))], voting='soft')}}\n")
    desc = (f"{dd.desc}. Compare three models with 5-fold `cross_val_score` (from `sklearn.model_selection`): "
            f"`'rf'`: `RandomForestClassifier(n_estimators={m}, random_state=0)`; `'hgb'`: "
            f"`HistGradientBoostingClassifier(max_iter={m}, random_state=0)`; `'vote'`: a soft `VotingClassifier` of "
            f"`('lr', make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)))` and `('rf', "
            f"RandomForestClassifier(n_estimators={m}, random_state=0))`.")
    if task == "means":
        return P(f"{desc} Return a dict {{name: mean accuracy}}.", dd.setup,
                 models + "result = {k: cross_val_score(m, X, y, cv=5).mean() for k, m in models.items()}",
                 require=["VotingClassifier", "voting", "cross_val_score"], tol=1e-9)
    if task == "best":
        return P(f"{desc} Return the name of the model with the best mean accuracy (the first of 'rf', 'hgb', 'vote' "
                 f"on ties).", dd.setup,
                 models + "means = {k: cross_val_score(m, X, y, cv=5).mean() for k, m in models.items()}\n"
                          "result = max(means, key=means.get)", require=["VotingClassifier", "cross_val_score"])
    return P(f"{desc} Return a dict {{name: (mean, std)}} of each model's 5 fold accuracies (`np.std`, population).",
             dd.setup, models + "result = {}\nfor k, m in models.items():\n    s = cross_val_score(m, X, y, cv=5)\n"
                                "    result[k] = (s.mean(), s.std())",
             require=["VotingClassifier", "cross_val_score"], tol=1e-9)


@template(M, "medium")
def voting_by_hand(rng):
    setup, desc, k, d = _clf(rng, n=(150, 300))
    n = rng.randint(3, 6)
    w = rng.sample([1, 2, 3], 3)
    task = rng.choice(["soft", "weights", "hard", "members"])
    members = ("[('lr', make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),\n"
               "     ('tree', DecisionTreeClassifier(max_depth=3, random_state=0)), ('nb', GaussianNB())]")
    imp = ("from sklearn.ensemble import VotingClassifier\nfrom sklearn.linear_model import LogisticRegression\n"
           "from sklearn.naive_bayes import GaussianNB\nfrom sklearn.pipeline import make_pipeline\n"
           "from sklearn.preprocessing import StandardScaler\nfrom sklearn.tree import DecisionTreeClassifier\n")
    shown = ("`[('lr', make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))), ('tree', "
             "DecisionTreeClassifier(max_depth=3, random_state=0)), ('nb', GaussianNB())]`")
    head = f"{desc}. Build a `VotingClassifier` (from `sklearn.ensemble`) over the estimators {shown}"
    if task == "soft":
        return P(f"{head} with `voting='soft'` and fit it on `X, y`. Soft voting averages the members' "
                 f"probabilities: rebuild `vote.predict_proba(X[:{n}])` from the fitted members in `vote.estimators_` "
                 f"and return your array.", setup,
                 imp + f"vote = VotingClassifier({members}, voting='soft').fit(X, y)\n"
                       f"result = np.mean([e.predict_proba(X[:{n}]) for e in vote.estimators_], axis=0)",
                 require=["VotingClassifier", "estimators_"], tol=1e-9)
    if task == "weights":
        return P(f"{head} with `voting='soft', weights={w}` and fit it on `X, y`. Return `vote.predict_proba(X[:{n}])` "
                 f"rebuilt by hand: the weighted average (`np.average(..., weights=...)`) of the members' "
                 f"probabilities.", setup,
                 imp + f"vote = VotingClassifier({members}, voting='soft', weights={w}).fit(X, y)\n"
                       f"result = np.average([e.predict_proba(X[:{n}]) for e in vote.estimators_], axis=0, weights={w})",
                 require=["VotingClassifier", "weights"], tol=1e-9)
    if task == "hard":
        return P(f"{head} with `voting='hard'` and fit it on `X, y`. Hard voting takes the majority label (ties go to "
                 f"the smallest label, as `np.bincount(...).argmax()` does). Rebuild `vote.predict(X)` from the members "
                 f"in `vote.estimators_` and return the fraction of rows where your vote equals it (1.0).", setup,
                 imp + f"vote = VotingClassifier({members}, voting='hard').fit(X, y)\n"
                       "P_ = np.array([e.predict(X) for e in vote.estimators_])\n"
                       "mine = np.array([np.bincount(col).argmax() for col in P_.T])\n"
                       "result = float((mine == vote.predict(X)).mean())",
                 require=["VotingClassifier", "estimators_"])
    return P(f"{head} with `voting='soft'` and fit it on `X, y`. `vote.named_estimators_` maps each name to its fitted "
             f"member. Return a dict {{name: training accuracy}} for the three members plus `'vote'` for the "
             f"ensemble itself.", setup,
             imp + f"vote = VotingClassifier({members}, voting='soft').fit(X, y)\n"
                   "result = {k: e.score(X, y) for k, e in vote.named_estimators_.items()}\n"
                   "result['vote'] = vote.score(X, y)",
             require=["named_estimators_"], tol=1e-9)


@template(M, "medium")
def stacking_inspect(rng):
    setup, desc, k, d = _clf(rng, allow=("cancer", "wine", "synth"), n=(150, 300))
    m, n = rng.choice([10, 20]), rng.randint(2, 4)
    task = rng.choice(["score", "coef", "transform", "compare"])
    build = ("from sklearn.ensemble import StackingClassifier, RandomForestClassifier\n"
             "from sklearn.linear_model import LogisticRegression\nfrom sklearn.pipeline import make_pipeline\n"
             "from sklearn.preprocessing import StandardScaler\nfrom sklearn.svm import SVC\n"
             f"stack = StackingClassifier([('rf', RandomForestClassifier(n_estimators={m}, random_state=0)),\n"
             "                            ('svc', make_pipeline(StandardScaler(), SVC(random_state=0)))],\n"
             "                           final_estimator=LogisticRegression(), cv=5).fit(X_tr, y_tr)\n")
    head = (f"{desc}; {SPLIT_DESC}. Fit `stack = StackingClassifier([('rf', RandomForestClassifier(n_estimators={m}, "
            f"random_state=0)), ('svc', make_pipeline(StandardScaler(), SVC(random_state=0)))], "
            f"final_estimator=LogisticRegression(), cv=5)` (from `sklearn.ensemble`) on the training split: the final "
            f"logistic regression learns from the members' 5-fold out-of-fold outputs.")
    setup += "\n" + SPLIT
    if task == "score":
        return P(f"{head} Return its test accuracy.", setup, build + "result = stack.score(X_te, y_te)",
                 require=["StackingClassifier", "final_estimator"])
    if task == "coef":
        return P(f"{head} Return the final estimator's learned weights, `stack.final_estimator_.coef_` (one per "
                 f"member output: the forest's probabilities and the SVC's decision values).", setup,
                 build + "result = stack.final_estimator_.coef_", require=["final_estimator_"], tol=1e-4)
    if task == "transform":
        return P(f"{head} `stack.transform` returns the features the final estimator sees. Return them for the first "
                 f"{n} test rows.", setup, build + f"result = stack.transform(X_te[:{n}])",
                 require=["StackingClassifier", "transform"], tol=1e-6)
    return P(f"{head} Is the stack better than its members? Return `(stack, rf, svc)` test accuracies, using the "
             f"members refitted on the whole training split (`stack.named_estimators_`).", setup,
             build + "result = (stack.score(X_te, y_te), stack.named_estimators_['rf'].score(X_te, y_te), "
                     "stack.named_estimators_['svc'].score(X_te, y_te))",
             require=["StackingClassifier", "named_estimators_"])


@template(M, "medium")
def hgb_native_categories(rng):
    t = table(rng, n=(150, 300))
    clf = t.kind == "clf"
    cls = "HistGradientBoostingClassifier" if clf else "HistGradientBoostingRegressor"
    m = rng.choice([20, 30])
    task = rng.choice(["score", "flags", "codes", "new_row"])
    setup = t.setup + f"\nfor c in {lit(t.cat)}:\n    df[c] = df[c].astype('category')"
    ctor = f"{cls}(categorical_features='from_dtype', max_iter={m}, random_state=0)"
    imp = f"from sklearn.ensemble import {cls}\n"
    head = (f"{t.desc}. The text columns {', '.join(f'`{c}`' for c in t.cat)} have the pandas `category` dtype, which "
            f"`{ctor}` treats as categories natively (no one-hot encoding).")
    if task == "score":
        return P(f"{head} Fit it on the first 70% of the rows (`h = int(0.7 * len(df))`) and return its `score` on the "
                 f"rest.", setup,
                 imp + f"h = int(0.7 * len(df))\nmodel = {ctor}.fit(df.iloc[:h], y[:h])\n"
                       "result = model.score(df.iloc[h:], y[h:])", require=[cls, "categorical_features"], tol=1e-6)
    if task == "flags":
        return P(f"{head} Fit it on `df, y` and return its `is_categorical_` mask as a list of bools (one per column of "
                 f"`df`).", setup, imp + f"result = {ctor}.fit(df, y).is_categorical_.tolist()",
                 require=["categorical_features", "is_categorical_"])
    if task == "codes":
        return P(f"{head} Compare with pretending the category codes are numbers: in a copy of `df`, replace each "
                 f"category column by its integer codes (`.cat.codes`) and fit `{cls}(max_iter={m}, random_state=0)` "
                 f"on that. Train both on the first 70% of the rows and return their test scores as "
                 f"`(native, codes)`.", setup,
                 imp + f"h = int(0.7 * len(df))\ncodes = df.copy()\nfor c in {lit(t.cat)}:\n    codes[c] = df[c].cat.codes\n"
                       f"native = {ctor}.fit(df.iloc[:h], y[:h]).score(df.iloc[h:], y[h:])\n"
                       f"plain = {cls}(max_iter={m}, random_state=0).fit(codes.iloc[:h], y[:h]).score(codes.iloc[h:], y[h:])\n"
                       "result = (native, plain)", require=["categorical_features", "cat"], tol=1e-6)
    row = {c: 1 for c in t.num}
    row.update({c: t.levels[c][rng.randrange(len(t.levels[c]))] for c in t.cat})
    return P(f"{head} Fit it on `df, y`, then predict `new_row` (a dict). The one-row DataFrame must use the **same** "
             f"category dtypes as `df` (`.astype(df.dtypes.to_dict())` after putting the columns in `df`'s order). "
             f"Return the prediction as a float.", setup + f"\nnew_row = {lit(row)}",
             imp + f"model = {ctor}.fit(df, y)\nnew = pd.DataFrame([new_row])[df.columns].astype(df.dtypes.to_dict())\n"
                   "result = float(model.predict(new)[0])", require=["categorical_features", "astype"], tol=1e-6)


@template(M, "medium")
def forest_tasks_themed(rng):
    t = table(rng, numeric_only=True, n=(150, 300))
    clf = t.kind == "clf"
    cls = "RandomForestClassifier" if clf else "RandomForestRegressor"
    m = rng.choice([20, 30])
    ctor = f"{cls}(n_estimators={m}, random_state=0)"
    imp = f"from sklearn.ensemble import {cls}\n"
    sizes = [5, 10, 20, 40]
    leaves = sorted(rng.sample([1, 3, 5, 10, 20], 3))
    row = {c: 1 for c in t.num}
    items = [
        ("feature_importances_", f"Fit `{ctor}` on `df, y` and return the name of the most important column.",
         imp + f"rf = {ctor}.fit(df, y)\nresult = df.columns[rf.feature_importances_.argmax()]"),
        ("score", f"Fit `{ctor}` on the first 70% of the rows (`h = int(0.7 * len(df))`) and return its `score` on the "
                  f"rest.", imp + f"h = int(0.7 * len(df))\nresult = {ctor}.fit(df.iloc[:h], y[:h]).score(df.iloc[h:], y[h:])"),
        ("predict", f"Fit `{ctor}` on `df, y` and predict `new_row` (a dict; build a one-row DataFrame with `df`'s "
                    f"columns). Return the prediction as a float.",
         imp + f"rf = {ctor}.fit(df, y)\nresult = float(rf.predict(pd.DataFrame([new_row])[df.columns])[0])"),
        ("n_estimators", f"More trees, better forest? Train on the first 70% of the rows and, for `n_estimators` in "
                         f"`{sizes}`, return the list of test scores of `{cls}(n_estimators=..., random_state=0)`.",
         imp + f"h = int(0.7 * len(df))\nresult = [{cls}(n_estimators=k, random_state=0).fit(df.iloc[:h], y[:h])"
               f".score(df.iloc[h:], y[h:]) for k in {sizes}]"),
        ("min_samples_leaf", f"For `min_samples_leaf` in `{leaves}`, fit `{cls}(n_estimators=10, min_samples_leaf=..., "
                             f"random_state=0)` on `df, y` and return the number of leaves of the first tree "
                             f"(`estimators_[0]`) each time.",
         imp + f"result = [{cls}(n_estimators=10, min_samples_leaf=k, random_state=0).fit(df, y).estimators_[0]"
               f".get_n_leaves() for k in {leaves}]"),
        ("max_features", f"Train on the first 70% of the rows and compare `max_features=None` (every split sees every "
                         f"column) with `max_features=1`: return the two test scores as a tuple, using "
                         f"`{cls}(n_estimators={m}, max_features=..., random_state=0)`.",
         imp + f"h = int(0.7 * len(df))\nresult = tuple({cls}(n_estimators={m}, max_features=f, random_state=0)"
               f".fit(df.iloc[:h], y[:h]).score(df.iloc[h:], y[h:]) for f in (None, 1))"),
        ("permutation_importance", f"Fit `{ctor}` on the first 70% of the rows, then compute "
                                   f"`permutation_importance(rf, X_test, y_test, n_repeats=5, random_state=0)` (from "
                                   f"`sklearn.inspection`) on the rest. Return the column whose shuffling hurts the "
                                   f"test score most (largest `importances_mean`).",
         imp + "from sklearn.inspection import permutation_importance\nh = int(0.7 * len(df))\n"
               f"rf = {ctor}.fit(df.iloc[:h], y[:h])\n"
               "pi = permutation_importance(rf, df.iloc[h:], y[h:], n_repeats=5, random_state=0)\n"
               "result = df.columns[pi.importances_mean.argmax()]"),
        ("max_samples", f"Fit `{cls}(n_estimators={m}, max_samples=0.5, oob_score=True, random_state=0)` on `df, y` "
                        f"(each tree sees a bootstrap sample of half the rows) and return its `oob_score_`.",
         imp + f"result = {cls}(n_estimators={m}, max_samples=0.5, oob_score=True, random_state=0).fit(df, y).oob_score_"),
    ]
    return catalog(rng, items, t.setup + f"\nnew_row = {lit(row)}", f"{t.desc}. `new_row` is a dict of feature values.",
                   tol=1e-6)


@template(M, "medium")
def learning_rate_tradeoff(rng):
    setup, desc, k, d = _clf(rng, n=(150, 300))
    lrs = sorted(rng.sample([0.03, 0.1, 0.3, 0.6, 1.0], 3))
    subs = [0.5, 0.8, 1.0]
    m = rng.choice([15, 20])
    task = rng.choice(["gb", "hgb", "subsample", "train_score"])
    setup += "\n" + SPLIT
    head = f"{desc}; {SPLIT_DESC}."
    if task == "gb":
        return P(f"{head} A smaller learning rate shrinks each tree's contribution. For `learning_rate` in `{lrs}`, fit "
                 f"`GradientBoostingClassifier(n_estimators={m}, learning_rate=..., max_depth=2, random_state=0)` on "
                 f"the training split and return the list of test accuracies.", setup,
                 "from sklearn.ensemble import GradientBoostingClassifier\n"
                 f"result = [GradientBoostingClassifier(n_estimators={m}, learning_rate=lr, max_depth=2, random_state=0)"
                 f".fit(X_tr, y_tr).score(X_te, y_te) for lr in {lrs}]", require=["learning_rate"])
    if task == "hgb":
        return P(f"{head} For `learning_rate` in `{lrs}`, fit `HistGradientBoostingClassifier(max_iter={m}, "
                 f"learning_rate=..., max_leaf_nodes=8, random_state=0)` on the training split and return a dict "
                 f"{{learning_rate: test accuracy}}.", setup,
                 "from sklearn.ensemble import HistGradientBoostingClassifier\n"
                 f"result = {{lr: HistGradientBoostingClassifier(max_iter={m}, learning_rate=lr, max_leaf_nodes=8, "
                 f"random_state=0).fit(X_tr, y_tr).score(X_te, y_te) for lr in {lrs}}}",
                 require=["learning_rate", "max_leaf_nodes"])
    if task == "subsample":
        return P(f"{head} *Stochastic* gradient boosting fits each tree on a random fraction of the rows. For "
                 f"`subsample` in `{subs}`, fit `GradientBoostingClassifier(n_estimators={m}, subsample=..., "
                 f"max_depth=2, random_state=0)` on the training split and return the list of test accuracies.", setup,
                 "from sklearn.ensemble import GradientBoostingClassifier\n"
                 f"result = [GradientBoostingClassifier(n_estimators={m}, subsample=s, max_depth=2, random_state=0)"
                 f".fit(X_tr, y_tr).score(X_te, y_te) for s in {subs}]", require=["subsample"])
    lr = lrs[-1]
    return P(f"{head} Fit `gb = GradientBoostingClassifier(n_estimators={m}, learning_rate={lr}, max_depth=2, "
             f"random_state=0)` on the training split. `gb.train_score_[i]` is the training loss after round i + 1. "
             f"Return `(gb.train_score_[0], gb.train_score_[-1], all_decreasing)` where `all_decreasing` says whether "
             f"every round lowered the training loss.", setup,
             "from sklearn.ensemble import GradientBoostingClassifier\n"
             f"gb = GradientBoostingClassifier(n_estimators={m}, learning_rate={lr}, max_depth=2, random_state=0)"
             ".fit(X_tr, y_tr)\nresult = (gb.train_score_[0], gb.train_score_[-1], bool((np.diff(gb.train_score_) < 0).all()))",
             require=["train_score_"], tol=1e-6)


@template(M, "medium")
def tree_leaf_bookkeeping(rng):
    setup, desc, k, d = _clf(rng)
    depth = rng.randint(2, 4)
    task = rng.choice(["counts", "value", "pure", "rule"])
    fit = (f"from sklearn.tree import DecisionTreeClassifier\n"
           f"tree = DecisionTreeClassifier(max_depth={depth}, random_state=0).fit(X, y)\nt = tree.tree_\n")
    head = f"{desc}. Fit `tree = DecisionTreeClassifier(max_depth={depth}, random_state=0)` on `X, y`; `t = tree.tree_`."
    if task == "counts":
        return P(f"{head} Count the training rows that land in each leaf with `tree.apply(X)` and return a dict "
                 f"{{leaf id: count}}; it agrees with `t.n_node_samples` at those leaves.", setup,
                 fit + "ids, cnt = np.unique(tree.apply(X), return_counts=True)\n"
                       "result = {int(i): int(c) for i, c in zip(ids, cnt)}", require=["apply"])
    if task == "value":
        i = rng.randint(0, 20)
        return P(f"{head} For a classifier, `t.value[node, 0]` holds the class fractions of a node's training rows. "
                 f"Find the leaf that row {i} lands in and return `t.value[leaf, 0]`; it equals "
                 f"`tree.predict_proba(X[{i}:{i + 1}])[0]`.", setup,
                 fit + f"leaf = tree.apply(X[{i}:{i + 1}])[0]\nresult = t.value[leaf, 0]", require=["apply", "value"],
                 tol=1e-9)
    if task == "pure":
        return P(f"{head} How many leaves are **pure** (all their training rows share one class, so impurity 0)? "
                 f"Leaves have `t.children_left == -1`; return `(pure_leaves, all_leaves)`.", setup,
                 fit + "leaf = t.children_left == -1\nresult = (int((t.impurity[leaf] == 0).sum()), int(leaf.sum()))",
                 require=["children_left", "impurity"])
    return P(f"{desc}. Fit `stump = DecisionTreeClassifier(max_depth=1, random_state=0)` on `X, y`. Rewrite its "
             f"`predict` as one `np.where`: rows with `X[:, f] <= thr` get the left leaf's class, the others the right "
             f"leaf's, where `f, thr = t.feature[0], t.threshold[0]` and a leaf's class is "
             f"`stump.classes_[t.value[leaf, 0].argmax()]` (left leaf = node 1, right = node 2). Return your "
             f"predictions for all of `X`.", setup,
             "from sklearn.tree import DecisionTreeClassifier\nstump = DecisionTreeClassifier(max_depth=1, random_state=0)"
             ".fit(X, y)\nt = stump.tree_\nf, thr = t.feature[0], t.threshold[0]\n"
             "left, right = stump.classes_[t.value[1, 0].argmax()], stump.classes_[t.value[2, 0].argmax()]\n"
             "result = np.where(X[:, f] <= thr, left, right)", require=["threshold", "value"])


# ================================================================= hard

@template(M, "hard")
def root_split_by_hand(rng):
    n, d, s = rng.randint(40, 80), rng.randint(2, 4), seeded(rng)
    task = rng.choice(["regression", "gini", "entropy"])
    base = f"g = np.random.default_rng({s})\nX = g.integers(0, 80, ({n}, {d})) / 4\n"
    if task == "regression":
        setup = base + f"y = (X @ g.uniform(-2, 2, {d}) + 3 * np.sin(X[:, 0]) + g.normal(0, 1, {n})).round(3)"
        return P(f"`X` ({n} × {d}, values on a 0.25 grid) and the continuous `y` are given. Find the best first split of "
                 f"a regression tree **by hand**: for every feature `f` and every threshold halfway between two "
                 f"consecutive distinct sorted values of `X[:, f]`, compute the total squared error of the two sides "
                 f"(each side predicts its mean). Return `(f, threshold)` of the best split; it matches "
                 f"`DecisionTreeRegressor(max_depth=1, random_state=0)`'s `tree_.feature[0]` and "
                 f"`tree_.threshold[0]`.", setup,
                 "best = (np.inf, None, None)\nfor f in range(X.shape[1]):\n    v = np.unique(X[:, f])\n"
                 "    for thr in (v[:-1] + v[1:]) / 2:\n        left = X[:, f] <= thr\n"
                 "        sse = ((y[left] - y[left].mean()) ** 2).sum() + ((y[~left] - y[~left].mean()) ** 2).sum()\n"
                 "        if sse < best[0]:\n            best = (sse, f, thr)\nresult = (int(best[1]), float(best[2]))",
                 require=[], tol=1e-9)
    setup = base + (f"z = X @ g.uniform(0.5, 1.5, {d}) + g.normal(0, 3, {n})\n"
                     "y = np.digitize(z, np.quantile(z, [0.35, 0.7]))")
    if task == "gini":
        crit, formula, fn = "Gini", "1 − Σ p_k²", "    p = np.bincount(lab) / len(lab)\n    return 1 - (p ** 2).sum()\n"
    else:
        crit, formula, fn = ("entropy", "−Σ p_k·log2(p_k)",
                             "    p = np.bincount(lab) / len(lab)\n    p = p[p > 0]\n    return -(p * np.log2(p)).sum()\n")
    kw = "" if task == "gini" else ", criterion='entropy'"
    return P(f"`X` ({n} × {d}, values on a 0.25 grid) and the labels `y` (0, 1, 2) are given. Search the best first "
             f"split of a classification tree **by hand**: for every feature and every threshold halfway between two "
             f"consecutive distinct sorted values, compute the children's sample-weighted {crit} impurity "
             f"(({formula}) of each side, weighted by its share of rows) and keep the smallest. Then fit "
             f"`DecisionTreeClassifier(max_depth=1{kw}, random_state=0)` and compute the same weighted child impurity "
             f"from its `tree_` (`impurity` and `n_node_samples` of nodes 1 and 2). Return `(yours, sklearns)`: they "
             f"are equal.", setup,
             f"from sklearn.tree import DecisionTreeClassifier\ndef imp(lab):\n{fn}"
             "best = np.inf\nfor f in range(X.shape[1]):\n    v = np.unique(X[:, f])\n"
             "    for thr in (v[:-1] + v[1:]) / 2:\n        left = X[:, f] <= thr\n"
             "        w = (left.sum() * imp(y[left]) + (~left).sum() * imp(y[~left])) / len(y)\n"
             "        best = min(best, w)\n"
             f"t = DecisionTreeClassifier(max_depth=1{kw}, random_state=0).fit(X, y).tree_\n"
             "n = t.n_node_samples\nresult = (best, (n[1] * t.impurity[1] + n[2] * t.impurity[2]) / n[0])",
             require=["DecisionTreeClassifier", "impurity"], tol=1e-9)


@template(M, "hard")
def boosting_by_hand(rng):
    m, lr, n = rng.choice([5, 10, 20]), rng.choice([0.1, 0.3, 0.5]), rng.randint(3, 6)
    task = rng.choice(["clf", "reg", "staged"])
    if task == "clf":
        setup, desc, k, d = _binary(rng)
        return P(f"{desc}. Fit `gb = GradientBoostingClassifier(n_estimators={m}, learning_rate={lr}, max_depth=2, "
                 f"random_state=0)` on `X, y`. Its log-odds start at log(p / (1 − p)) with p = `y.mean()`, and every "
                 f"round adds `learning_rate` × a regression tree's output (`gb.estimators_[i, 0]`). Rebuild P(class 1) "
                 f"for the first {n} rows by hand (sum the log-odds, then apply the sigmoid) and return it; it equals "
                 f"`gb.predict_proba(X[:{n}])[:, 1]`.", setup,
                 f"from sklearn.ensemble import GradientBoostingClassifier\ngb = GradientBoostingClassifier(n_estimators={m}, "
                 f"learning_rate={lr}, max_depth=2, random_state=0).fit(X, y)\np = y.mean()\n"
                 f"z = np.log(p / (1 - p)) + {lr} * sum(t[0].predict(X[:{n}]) for t in gb.estimators_)\n"
                 "result = 1 / (1 + np.exp(-z))", require=["estimators_"], tol=1e-6)
    setup, desc, d = _reg(rng)
    sol = (f"from sklearn.ensemble import GradientBoostingRegressor\ngb = GradientBoostingRegressor(n_estimators={m}, "
           f"learning_rate={lr}, max_depth=2, random_state=0).fit(X, y)\n")
    head = (f"{desc}. Fit `gb = GradientBoostingRegressor(n_estimators={m}, learning_rate={lr}, max_depth=2, "
            f"random_state=0)` on `X, y`. Its prediction starts at the mean of `y`, and each round adds "
            f"`learning_rate` × the output of one regression tree (`gb.estimators_[i, 0]`), fitted to the residuals.")
    if task == "reg":
        return P(f"{head} Rebuild `gb.predict(X[:{n}])` by hand from `y.mean()` and the trees, and return your array.",
                 setup, sol + f"result = y.mean() + {lr} * sum(t[0].predict(X[:{n}]) for t in gb.estimators_)",
                 require=["estimators_"], tol=1e-6)
    return P(f"{head} Accumulate the predictions on all of `X` round by round and return the training mean squared "
             f"error after each of the {m} rounds, as an array (it matches using `gb.staged_predict(X)`).", setup,
             sol + f"F = np.full(len(y), y.mean())\nout = []\nfor t in gb.estimators_[:, 0]:\n"
                   f"    F = F + {lr} * t.predict(X)\n    out.append(((y - F) ** 2).mean())\nresult = np.array(out)",
             require=["estimators_"], tol=1e-6)


@template(M, "hard")
def bagging_oob_by_hand(rng):
    setup, desc, k, d = _clf(rng, n=(150, 300))
    m = rng.choice([30, 40])
    task = rng.choice(["score", "coverage", "average"])
    ctor = f"BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators={m}, oob_score=True, random_state=0)"
    sol = ("from sklearn.ensemble import BaggingClassifier\nfrom sklearn.tree import DecisionTreeClassifier\n"
           f"bag = {ctor}.fit(X, y)\nn = len(X)\n"
           "oob = np.ones((len(bag.estimators_), n), dtype=bool)\nfor i, s in enumerate(bag.estimators_samples_):\n"
           "    oob[i, s] = False\n")
    head = (f"{desc}. Fit `bag = {ctor}` on `X, y`. `bag.estimators_samples_[i]` holds the row indices tree i was "
            f"fitted on; the rows it never drew are *out of bag* for it.")
    if task == "score":
        return P(f"{head} Recompute `bag.oob_score_` by hand: for every row, add up `predict_proba` of the trees for "
                 f"which it is out of bag, predict the argmax (a row with no such tree gets all zeros, so class 0, as "
                 f"scikit-learn does), and return `(your_accuracy, bag.oob_score_)`.", setup,
                 sol + "votes = np.zeros((n, len(bag.classes_)))\nfor i, est in enumerate(bag.estimators_):\n"
                       "    votes[oob[i]] += est.predict_proba(X[oob[i]])\n"
                       "result = (float((votes.argmax(axis=1) == y).mean()), bag.oob_score_)",
                 require=["estimators_samples_"], tol=1e-9)
    if task == "coverage":
        return P(f"{head} Return the fraction of rows that are out of bag for **at least one** tree (only those get a "
                 f"real OOB prediction).", setup, sol + "result = float(oob.any(axis=0).mean())",
                 require=["estimators_samples_"], tol=1e-9)
    return P(f"{head} On average, for how many trees is a row out of bag? Return `(your_average, {m} * (1 - 1/n) ** n)`: "
             f"a bootstrap sample misses each row with probability (1 − 1/n)^n ≈ 0.368.", setup,
             sol + f"result = (float(oob.sum(axis=0).mean()), {m} * (1 - 1 / n) ** n)", require=["estimators_samples_"],
             tol=1e-9)


@template(M, "hard")
def adaboost_weighted_vote(rng):
    setup, desc, k, d = _binary(rng)
    m, lr = rng.choice([5, 10, 20]), rng.choice([0.5, 1.0])
    task = rng.choice(["vote", "first_error", "alpha"])
    ctor = f"AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators={m}, learning_rate={lr}, random_state=0)"
    sol = ("from sklearn.ensemble import AdaBoostClassifier\nfrom sklearn.tree import DecisionTreeClassifier\n"
           f"ada = {ctor}.fit(X, y)\n")
    head = f"{desc}. Fit `ada = {ctor}` on `X, y`: the stumps in `ada.estimators_` vote with weights `ada.estimator_weights_`."
    if task == "vote":
        return P(f"{head} Rebuild `ada.predict(X)` by hand: each stump adds +weight if it predicts class 1 and −weight "
                 f"otherwise; predict 1 where the total is positive. Return your predictions as an int array.", setup,
                 sol + "score = sum(w * np.where(e.predict(X) == 1, 1, -1) for e, w in zip(ada.estimators_, ada.estimator_weights_))\n"
                       "result = (score > 0).astype(int)", require=["estimator_weights_"])
    if task == "first_error":
        return P(f"{head} The first stump is fitted with equal sample weights, so its weighted error is just its "
                 f"misclassification rate on `X, y`. Return `(by_hand, ada.estimator_errors_[0])`.", setup,
                 sol + "result = (float((ada.estimators_[0].predict(X) != y).mean()), ada.estimator_errors_[0])",
                 require=["estimator_errors_"], tol=1e-9)
    return P(f"{head} For two classes, a stump with weighted error `err` gets the weight `learning_rate × log((1 − err) / "
             f"err)`. Compute the first stump's weight from `ada.estimator_errors_[0]` and return `(by_hand, "
             f"ada.estimator_weights_[0])`.", setup,
             sol + f"err = ada.estimator_errors_[0]\nresult = ({lr} * np.log((1 - err) / err), ada.estimator_weights_[0])",
             require=["estimator_errors_", "estimator_weights_"], tol=1e-9)


@template(M, "hard")
def importance_pitfalls(rng):
    dd = synth_clf(rng, n=(200, 350), d=(4, 6), classes=2)
    s, m = seeded(rng), rng.choice([20, 30])
    names = [f"x{i}" for i in range(dd.d)] + ["row_id"]
    setup = (dd.setup + f"\nX = np.column_stack([X, np.random.default_rng({s}).permutation(len(X))])\n"
             f"names = {lit(names)}\n" + SPLIT)
    task = rng.choice(["mdi_rank", "both", "perm_top"])
    head = (f"{dd.desc}, plus a last column `row_id`: a random permutation of 0..n−1, pure noise but with a distinct "
            f"value on every row. `names` lists the columns; {SPLIT_DESC}. Fit `rf = RandomForestClassifier("
            f"n_estimators={m}, random_state=0)` on the training split.")
    sol = (f"from sklearn.ensemble import RandomForestClassifier\nrf = RandomForestClassifier(n_estimators={m}, "
           "random_state=0).fit(X_tr, y_tr)\n")
    perm = ("from sklearn.inspection import permutation_importance\n"
            "pi = permutation_importance(rf, X_te, y_te, n_repeats=5, random_state=0)\n")
    if task == "mdi_rank":
        return P(f"{head} Impurity-based importances favour features with many distinct values. Return the rank of "
                 f"`row_id` among `rf.feature_importances_` (0 = most important) and its importance, as a tuple.",
                 setup, sol + "order = list(rf.feature_importances_.argsort()[::-1])\n"
                              "result = (order.index(len(names) - 1), rf.feature_importances_[-1])",
                 require=["feature_importances_"], tol=1e-9)
    if task == "both":
        return P(f"{head} Then compute `permutation_importance(rf, X_te, y_te, n_repeats=5, random_state=0)` (from "
                 f"`sklearn.inspection`) on the **test** split. Return `row_id`'s `(impurity importance, "
                 f"permutation importances_mean)`: on held-out data the noise column is worth about nothing.", setup,
                 sol + perm + "result = (rf.feature_importances_[-1], pi.importances_mean[-1])",
                 require=["feature_importances_", "permutation_importance"], tol=1e-9)
    return P(f"{head} Then compute `permutation_importance(rf, X_te, y_te, n_repeats=5, random_state=0)` (from "
             f"`sklearn.inspection`). Return the names of the 3 columns with the largest `importances_mean`, largest "
             f"first.", setup, sol + perm + "result = [names[i] for i in pi.importances_mean.argsort()[::-1][:3]]",
             require=["permutation_importance"])


@template(M, "hard")
def imbalanced_ensembles(rng):
    dd = synth_clf(rng, n=(300, 500), d=(4, 8), classes=2, imbalance=rng.choice([0.08, 0.1, 0.15]))
    m = rng.choice([20, 30])
    t = rng.choice([0.2, 0.3])
    setup = dd.setup + ("\nfrom sklearn.model_selection import train_test_split\n"
                        "X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)")
    head = (f"{dd.desc}; class 1 is rare. `X_tr, X_te, y_tr, y_te` is a stratified 70/30 split. Accuracy hides how "
            f"many rare cases a model misses, so look at the **recall** of class 1 (the share of true 1s predicted "
            f"as 1).")
    rec = "def recall(p):\n    return float((p[y_te == 1] == 1).mean())\n"
    task = rng.choice(["weights", "threshold", "hgb"])
    if task == "weights":
        return P(f"{head} Fit `RandomForestClassifier(n_estimators={m}, random_state=0)` and the same with "
                 f"`class_weight='balanced'` on the training split. Return their test recalls of class 1 as "
                 f"`(plain, balanced)`.", setup,
                 f"from sklearn.ensemble import RandomForestClassifier\n{rec}"
                 f"a = RandomForestClassifier(n_estimators={m}, random_state=0).fit(X_tr, y_tr)\n"
                 f"b = RandomForestClassifier(n_estimators={m}, class_weight='balanced', random_state=0).fit(X_tr, y_tr)\n"
                 "result = (recall(a.predict(X_te)), recall(b.predict(X_te)))", require=["class_weight"])
    if task == "threshold":
        return P(f"{head} Fit `rf = RandomForestClassifier(n_estimators={m}, random_state=0)` on the training split. "
                 f"Predict class 1 when P(class 1) ≥ 0.5 and, separately, when it is ≥ {t}. Return the two test "
                 f"recalls and the number of test rows flagged as 1 at the lower cut-off, as `(recall_05, "
                 f"recall_{str(t).replace('.', '')}, flagged)`.", setup,
                 f"from sklearn.ensemble import RandomForestClassifier\n{rec}"
                 f"p = RandomForestClassifier(n_estimators={m}, random_state=0).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]\n"
                 f"result = (recall((p >= 0.5).astype(int)), recall((p >= {t}).astype(int)), int((p >= {t}).sum()))",
                 require=["predict_proba"])
    return P(f"{head} Fit `HistGradientBoostingClassifier(max_iter={m}, random_state=0)` and the same with "
             f"`class_weight='balanced'` on the training split. Return both test `balanced_accuracy_score`s (from "
             f"`sklearn.metrics`: the mean of the per-class recalls) as `(plain, balanced)`.", setup,
             "from sklearn.ensemble import HistGradientBoostingClassifier\nfrom sklearn.metrics import balanced_accuracy_score\n"
             f"a = HistGradientBoostingClassifier(max_iter={m}, random_state=0).fit(X_tr, y_tr)\n"
             f"b = HistGradientBoostingClassifier(max_iter={m}, class_weight='balanced', random_state=0).fit(X_tr, y_tr)\n"
             "result = (balanced_accuracy_score(y_te, a.predict(X_te)), balanced_accuracy_score(y_te, b.predict(X_te)))",
             require=["class_weight", "balanced_accuracy_score"])


@template(M, "hard")
def ensemble_bug_fixes(rng):
    setup, desc, k, d = _clf(rng, allow=("cancer", "wine", "synth"), n=(150, 300))
    m = rng.choice([10, 20])
    depths = [2, 4, 6, 8]
    items = [
        ("probability", "This crashes with an AttributeError: soft voting needs `predict_proba`, which `SVC` only "
                        "offers when built with `probability=True`. Fix it (use `SVC(probability=True, "
                        "random_state=0)`) and return the test accuracy.\n```\n"
                        "vote = VotingClassifier([('svc', make_pipeline(StandardScaler(), SVC())),\n"
                        f"                         ('rf', RandomForestClassifier(n_estimators={m}, random_state=0))],\n"
                        "                        voting='soft').fit(X_tr, y_tr)\nresult = vote.score(X_te, y_te)\n```",
         "from sklearn.ensemble import VotingClassifier, RandomForestClassifier\nfrom sklearn.pipeline import make_pipeline\n"
         "from sklearn.preprocessing import StandardScaler\nfrom sklearn.svm import SVC\n"
         "vote = VotingClassifier([('svc', make_pipeline(StandardScaler(), SVC(probability=True, random_state=0))),\n"
         f"                         ('rf', RandomForestClassifier(n_estimators={m}, random_state=0))],\n"
         "                        voting='soft').fit(X_tr, y_tr)\nresult = vote.score(X_te, y_te)",
         {"require": ["probability", "VotingClassifier"]}),
        ("estimator", "This raises a TypeError in current scikit-learn (the old keyword was renamed). Fix it and "
                      "return the test accuracy.\n```\n"
                      f"ada = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2), n_estimators={m},\n"
                      "                         random_state=0).fit(X_tr, y_tr)\nresult = ada.score(X_te, y_te)\n```",
         "from sklearn.ensemble import AdaBoostClassifier\nfrom sklearn.tree import DecisionTreeClassifier\n"
         f"ada = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2), n_estimators={m},\n"
         "                         random_state=0).fit(X_tr, y_tr)\nresult = ada.score(X_te, y_te)",
         {"require": ["AdaBoostClassifier", "estimator"]}),
        ("bootstrap", "This raises a ValueError: out-of-bag rows only exist when trees are fitted on bootstrap samples. "
                      "Fix it (keep `oob_score=True`) and return `oob_score_`.\n```\n"
                      "rf = RandomForestClassifier(n_estimators=30, bootstrap=False, oob_score=True,\n"
                      "                            random_state=0).fit(X_tr, y_tr)\nresult = rf.oob_score_\n```",
         "from sklearn.ensemble import RandomForestClassifier\n"
         "rf = RandomForestClassifier(n_estimators=30, bootstrap=True, oob_score=True,\n"
         "                            random_state=0).fit(X_tr, y_tr)\nresult = rf.oob_score_",
         {"require": ["oob_score_"]}),
        ("HistGradientBoostingClassifier", "This crashes because `GradientBoostingClassifier` rejects NaN, and a few "
                                           "values of `X_tr` are missing (`X_nan`). Switch to "
                                           f"`HistGradientBoostingClassifier(max_iter={m}, random_state=0)`, which "
                                           "handles NaN natively, and return the test accuracy.\n```\n"
                                           f"gb = GradientBoostingClassifier(n_estimators={m}, random_state=0).fit(X_nan, y_tr)\n"
                                           "result = gb.score(X_te, y_te)\n```",
         "from sklearn.ensemble import HistGradientBoostingClassifier\n"
         f"gb = HistGradientBoostingClassifier(max_iter={m}, random_state=0).fit(X_nan, y_tr)\n"
         "result = gb.score(X_te, y_te)",
         {"setup": "X_nan = X_tr.copy()\nX_nan[::9, 0] = np.nan"}),
        ("feature_names", "This raises a ValueError: the tree was fitted on the first 3 columns only, but all the names "
                          "are passed. Fix the call and return the text.\n```\n"
                          "tree = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X_tr[:, :3], y_tr)\n"
                          "result = export_text(tree, feature_names=names)\n```",
         "from sklearn.tree import DecisionTreeClassifier, export_text\n"
         "tree = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X_tr[:, :3], y_tr)\n"
         "result = export_text(tree, feature_names=names[:3])"),
        ("cross_val_score", f"This code picks `max_depth` by **test** accuracy and reports that same test accuracy: an "
                            f"optimistic, leaky estimate.\n```\nscores = {{k: RandomForestClassifier(n_estimators={m}, "
                            f"max_depth=k, random_state=0)\n          .fit(X_tr, y_tr).score(X_te, y_te) for k in "
                            f"{depths}}}\nbest = max(scores, key=scores.get)\nresult = (best, scores[best])\n```\n"
                            f"Fix it: pick the depth with the best mean 5-fold `cross_val_score` on the **training** "
                            f"split (the first on ties), then refit on the training split and return `(best, "
                            f"test_accuracy)`.",
         "from sklearn.ensemble import RandomForestClassifier\nfrom sklearn.model_selection import cross_val_score\n"
         f"cv = {{k: cross_val_score(RandomForestClassifier(n_estimators={m}, max_depth=k, random_state=0), X_tr, y_tr, "
         f"cv=5).mean() for k in {depths}}}\nbest = max(cv, key=cv.get)\n"
         f"result = (best, RandomForestClassifier(n_estimators={m}, max_depth=best, random_state=0).fit(X_tr, y_tr)"
         ".score(X_te, y_te))"),
    ]
    return catalog(rng, items, setup + "\n" + SPLIT, f"{desc}; {SPLIT_DESC}.", tol=1e-9)


@template(M, "hard")
def select_on_validation(rng):
    setup, desc, k, d = _clf(rng, allow=("cancer", "wine", "synth"), n=(200, 400))
    task = rng.choice(["depth", "lr", "grid"])
    split3 = ("\nn = len(X)\na, b = int(0.6 * n), int(0.8 * n)\n"
              "X_tr, y_tr, X_val, y_val, X_te, y_te = X[:a], y[:a], X[a:b], y[a:b], X[b:], y[b:]")
    head = (f"{desc}. Rows `[:60%]`, `[60%:80%]` and `[80%:]` form the training, validation and test splits "
            f"(`X_tr, y_tr, X_val, y_val, X_te, y_te`).")
    if task == "depth":
        depths = [2, 4, 8, None]
        return P(f"{head} Choose `max_depth` among `{depths}` for `RandomForestClassifier(n_estimators=20, "
                 f"max_depth=..., random_state=0)` by **validation** accuracy (the first on ties), refit it on "
                 f"training + validation together, and return `(best_depth, test_accuracy)`.", setup + split3,
                 "from sklearn.ensemble import RandomForestClassifier\n"
                 f"val = [RandomForestClassifier(n_estimators=20, max_depth=k, random_state=0).fit(X_tr, y_tr).score(X_val, y_val) "
                 f"for k in {depths}]\nbest = {depths}[int(np.argmax(val))]\n"
                 "model = RandomForestClassifier(n_estimators=20, max_depth=best, random_state=0)\n"
                 "model.fit(np.vstack([X_tr, X_val]), np.concatenate([y_tr, y_val]))\nresult = (best, model.score(X_te, y_te))",
                 require=["max_depth"])
    if task == "lr":
        lrs = [0.03, 0.1, 0.3, 1.0]
        return P(f"{head} Choose `learning_rate` among `{lrs}` for `HistGradientBoostingClassifier(max_iter=20, "
                 f"learning_rate=..., random_state=0)` by validation accuracy (the first on ties). Return `(best_lr, "
                 f"validation_accuracies, test_accuracy_of_the_chosen_model)` (the chosen model trained on the "
                 f"training split only).", setup + split3,
                 "from sklearn.ensemble import HistGradientBoostingClassifier\n"
                 f"models = [HistGradientBoostingClassifier(max_iter=20, learning_rate=lr, random_state=0).fit(X_tr, y_tr) "
                 f"for lr in {lrs}]\nval = [m.score(X_val, y_val) for m in models]\ni = int(np.argmax(val))\n"
                 f"result = ({lrs}[i], val, models[i].score(X_te, y_te))", require=["learning_rate"])
    grid = {"max_depth": [2, 3, 5], "min_samples_leaf": [1, 5, 20]}
    return P(f"{head} Use `GridSearchCV(DecisionTreeClassifier(random_state=0), {grid}, cv=5)` (from "
             f"`sklearn.model_selection`) on training + validation together, and return `(best_params_, test accuracy "
             f"of best_estimator_)`.", setup + split3,
             "from sklearn.model_selection import GridSearchCV\nfrom sklearn.tree import DecisionTreeClassifier\n"
             f"gs = GridSearchCV(DecisionTreeClassifier(random_state=0), {grid}, cv=5)\n"
             "gs.fit(np.vstack([X_tr, X_val]), np.concatenate([y_tr, y_val]))\n"
             "result = (gs.best_params_, gs.best_estimator_.score(X_te, y_te))",
             require=["GridSearchCV"])


@template(M, "hard")
def tree_arrays_walk(rng):
    setup, desc, k, d = _clf(rng)
    depth, n = rng.randint(3, 5), rng.randint(4, 8)
    task = rng.choice(["apply", "depths", "mdi", "path"])
    fit = (f"from sklearn.tree import DecisionTreeClassifier\n"
           f"tree = DecisionTreeClassifier(max_depth={depth}, random_state=0).fit(X, y)\nt = tree.tree_\n")
    head = (f"{desc}. Fit `tree = DecisionTreeClassifier(max_depth={depth}, random_state=0)` on `X, y` and let "
            f"`t = tree.tree_`: node `i` sends a row to `t.children_left[i]` if `row[t.feature[i]] <= t.threshold[i]`, "
            f"else to `t.children_right[i]`; leaves have `children_left == -1`.")
    if task == "apply":
        return P(f"{head} Walk the tree **by hand** for each of the first {n} rows and return the leaf ids reached; "
                 f"they equal `tree.apply(X[:{n}])`.", setup,
                 fit + f"out = []\nfor row in X[:{n}]:\n    i = 0\n    while t.children_left[i] != -1:\n"
                       "        i = t.children_left[i] if row[t.feature[i]] <= t.threshold[i] else t.children_right[i]\n"
                       "    out.append(i)\nresult = np.array(out)", require=["children_left", "threshold"])
    if task == "depths":
        return P(f"{head} Children always have larger ids than their parent. Compute the depth of every node (root = 0) "
                 f"in one pass over the ids and return `(depths_of_the_leaves_in_id_order, tree.get_depth())`; the "
                 f"largest leaf depth equals `get_depth()`.", setup,
                 fit + "depth = np.zeros(t.node_count, dtype=int)\nfor i in range(t.node_count):\n"
                       "    if t.children_left[i] != -1:\n"
                       "        depth[t.children_left[i]] = depth[t.children_right[i]] = depth[i] + 1\n"
                       "result = (depth[t.children_left == -1], tree.get_depth())",
                 require=["children_left", "get_depth"])
    if task == "mdi":
        return P(f"{head} Recompute `tree.feature_importances_` **by hand**: every split node adds to its feature "
                 f"`w[i]·imp[i] − w[l]·imp[l] − w[r]·imp[r]` (with `w = t.weighted_n_node_samples`, `imp = "
                 f"t.impurity`, `l`, `r` its children); then divide by the total so the importances sum to 1. Return "
                 f"your array.", setup,
                 fit + f"out = np.zeros({d})\nw, imp = t.weighted_n_node_samples, t.impurity\n"
                       "for i in range(t.node_count):\n    l, r = t.children_left[i], t.children_right[i]\n"
                       "    if l != -1:\n        out[t.feature[i]] += w[i] * imp[i] - w[l] * imp[l] - w[r] * imp[r]\n"
                       "result = out / out.sum()", require=["weighted_n_node_samples", "impurity"], tol=1e-9)
    return P(f"{desc}. Fit `tree = DecisionTreeClassifier(max_depth={depth}, random_state=0)` on `X, y`. "
             f"`tree.decision_path(X)` is a sparse rows × nodes indicator matrix of the nodes each row passes through. "
             f"Return, for the first {n} rows, how many nodes each one visits, and the id of the deepest node it "
             f"visits (its leaf), as `(counts, leaves)`.", setup,
             "from sklearn.tree import DecisionTreeClassifier\n"
             f"tree = DecisionTreeClassifier(max_depth={depth}, random_state=0).fit(X, y)\n"
             f"M_ = tree.decision_path(X[:{n}]).toarray()\n"
             f"result = (M_.sum(axis=1), np.array([np.flatnonzero(r).max() for r in M_]))",
             require=["decision_path"])


@template(M, "hard")
def forest_uncertainty(rng):
    setup, desc, d = _reg(rng)
    m, n = rng.choice([20, 30]), rng.randint(3, 6)
    task = rng.choice(["std", "top", "interval"])
    sol = (f"from sklearn.ensemble import RandomForestRegressor\nrf = RandomForestRegressor(n_estimators={m}, "
           f"random_state=0).fit(X_tr, y_tr)\nT = np.array([t.predict(X_te) for t in rf.estimators_])\n")
    head = (f"{desc}; {SPLIT_DESC}. Fit `rf = RandomForestRegressor(n_estimators={m}, random_state=0)` on the "
            f"training split. How much its trees disagree on a row is a rough measure of uncertainty.")
    setup += "\n" + SPLIT
    if task == "std":
        return P(f"{head} Return the standard deviation (`np.std`, population) of the trees' predictions for each of "
                 f"the first {n} test rows.", setup, sol + f"result = T[:, :{n}].std(axis=0)",
                 require=["estimators_"], tol=1e-6)
    if task == "top":
        return P(f"{head} Return the indices (into `X_te`) of the 3 test rows where the trees' predictions have the "
                 f"largest standard deviation, largest first, and the forest's absolute error on each of them, as "
                 f"`(indices, errors)`.", setup,
                 sol + "idx = np.argsort(T.std(axis=0))[::-1][:3]\nresult = (idx, np.abs(rf.predict(X_te[idx]) - y_te[idx]))",
                 require=["estimators_"], tol=1e-6)
    return P(f"{head} For every test row, take the 10th and 90th percentiles (`np.percentile`) of the trees' "
             f"predictions as an interval. Return the fraction of test targets that fall inside their interval "
             f"(ends included).", setup,
             sol + "lo, hi = np.percentile(T, [10, 90], axis=0)\nresult = float(((y_te >= lo) & (y_te <= hi)).mean())",
             require=["estimators_"], tol=1e-9)


# ================================================================= challenge

@template(M, "challenge")
def best_stump_split(rng):
    n, s = rng.randint(8, 12), seeded(rng)
    report = rng.choice(["threshold", "mse", "gini"])

    def gen(seed, n_, hi):
        u = "x" if hi == 40 else f"(x / {hi // 40})"
        y = (f"np.digitize({u} + g.normal(0, 2, {n_}), [3.5, 6.5])" if report == "gini" else
             f"(3 * np.sin({u}) + 0.5 * {u} + g.normal(0, 1, {n_})).round(3)")
        return f"g = np.random.default_rng({seed})\nx = g.integers(0, {hi}, {n_}) / 4\ny = {y}\nreport = {report!r}"
    setup = gen(s, n, 40)
    sol = ("o = np.argsort(x, kind='stable')\nxs, ys = x[o], y[o]\nn = len(xs)\ni = np.arange(1, n)\n"
           "if report == 'gini':\n    K = int(ys.max()) + 1\n    cl = np.cumsum(np.eye(K)[ys], axis=0)[:-1]\n"
           "    cr = np.bincount(ys, minlength=K) - cl\n"
           "    imp = (i - (cl ** 2).sum(axis=1) / i + (n - i) - (cr ** 2).sum(axis=1) / (n - i)) / n\nelse:\n"
           "    s1, s2 = np.cumsum(ys)[:-1], np.cumsum(ys ** 2)[:-1]\n    t1, t2 = ys.sum() - s1, (ys ** 2).sum() - s2\n"
           "    imp = (s2 - s1 ** 2 / i + t2 - t1 ** 2 / (n - i)) / n\n"
           "imp = np.where(xs[1:] > xs[:-1], imp, np.inf)\nb = int(np.argmin(imp))\n"
           "result = float(xs[b] / 2 + xs[b + 1] / 2) if report == 'threshold' else float(imp[b])")
    tests = [gen(s + i, 200000, 40000) for i in (1, 2)]
    what = {"threshold": "the **threshold** of the best split of a regression stump (squared error)",
            "mse": "the smallest achievable **weighted child MSE** of a regression stump: (SSE of the left rows + SSE "
                   "of the right rows) / n, each side predicting its own mean",
            "gini": "the smallest achievable **weighted child Gini impurity** (n_L·G_L + n_R·G_R) / n of a "
                    "classification stump, where G = 1 − Σ p_k² over the class fractions of a side"}[report]
    return challenge(
        "Best stump split",
        "A decision stump (`max_depth=1`) on one feature `x` tries every split between two consecutive **distinct** "
        "sorted values (rows with `x <= threshold` go left; the threshold is the midpoint `a / 2 + b / 2` of the two "
        f"values) and keeps the one with the lowest weighted impurity of the two children. Return {what}, exactly "
        f"what `DecisionTree{'Classifier' if report == 'gini' else 'Regressor'}(max_depth=1)` finds on "
        f"`x.reshape(-1, 1), y`. If several splits tie, the smallest threshold wins. (`report` = {report!r}; labels "
        f"are 0, 1, 2 for 'gini' and floats otherwise.)",
        setup, sol, tests, "len(x) ≤ 2·10^5", speed="O(n log n): sort once, then prefix sums",
        answer_desc=" (a float)", tol=1e-6)


@template(M, "challenge")
def tree_apply_batch(rng):
    s = seeded(rng)
    leaves, q = rng.randint(5, 8), rng.randint(5, 8)
    report = rng.choice(["leaf", "value", "depth"])

    def gen(seed, nfit, leaves_, nq):
        return (f"from sklearn.tree import DecisionTreeRegressor\ng = np.random.default_rng({seed})\n"
                f"A = g.uniform(0, 10, ({nfit}, 2)).round(2)\n"
                f"t = DecisionTreeRegressor(max_leaf_nodes={leaves_}, random_state=0).fit(A, A[:, 0] * A[:, 1] + "
                f"g.normal(0, 3, {nfit})).tree_\n"
                "left, right, feature, threshold = t.children_left, t.children_right, t.feature, t.threshold\n"
                f"value = t.value[:, 0, 0].round(2)\nX = g.uniform(0, 10, ({nq}, 2)).round(2)\ndel t, A\n"
                f"report = {report!r}")
    setup = gen(s, 80, leaves, q)
    sol = ("node = np.zeros(len(X), dtype=np.intp)\ndepth = np.zeros(len(X), dtype=np.intp)\n"
           "active = np.flatnonzero(left[node] != -1)\nwhile len(active):\n    nd = node[active]\n"
           "    go = X[active, feature[nd]] <= threshold[nd]\n    node[active] = np.where(go, left[nd], right[nd])\n"
           "    depth[active] += 1\n    active = active[left[node[active]] != -1]\n"
           "result = {'leaf': node, 'value': value[node], 'depth': depth}[report]")
    tests = [gen(s + i, 6000, 3000, 100000) for i in (1, 2)]
    what = {"leaf": "the id of the leaf each row of `X` reaches (what `tree.apply(X)` returns)",
            "value": "the prediction for each row of `X`: `value` of the leaf it reaches",
            "depth": "the depth of the leaf each row of `X` reaches (the number of splits on its path)"}[report]
    return challenge(
        "Batch tree prediction",
        "A fitted regression tree is given only as its arrays: node `i` sends a row to `left[i]` if `row[feature[i]] "
        "<= threshold[i]` and to `right[i]` otherwise; leaves have `left[i] == -1`, and `value[i]` is a node's "
        f"prediction. Starting at node 0, route every row of `X` and return {what}. (`report` = {report!r}.) Walking "
        "the rows one at a time in Python is too slow: move all rows down the tree together, one level per step.",
        setup, sol, tests, "number of rows ≤ 10^5, nodes ≤ 6·10^3, depth ≤ 60",
        speed="O(depth) vectorized steps over all rows", answer_desc=" (a NumPy array, one value per row)", tol=1e-9)


@template(M, "challenge")
def bagging_oob_votes(rng):
    n, T, K, s = rng.randint(8, 12), rng.randint(5, 7), rng.randint(2, 3), seeded(rng)
    report = rng.choice(["accuracy", "counts", "labels"])

    def gen(seed, n_, T_, K_):
        return (f"g = np.random.default_rng({seed})\nn, T, K = {n_}, {T_}, {K_}\ny = g.integers(0, K, n)\n"
                "boot = g.integers(0, n, (T, n))\n"
                "pred = np.where(g.random((T, n)) < 0.7, y, g.integers(0, K, (T, n)))\n"
                f"report = {report!r}")
    sol = ("inbag = np.zeros((T, n), dtype=bool)\ninbag[np.arange(T)[:, None], boot] = True\noob = ~inbag\n"
           "votes = np.stack([((pred == k) & oob).sum(axis=0) for k in range(K)], axis=1)\ncnt = oob.sum(axis=0)\n"
           "lab = np.where(cnt > 0, votes.argmax(axis=1), -1)\n"
           "if report == 'accuracy':\n    result = float((lab[cnt > 0] == y[cnt > 0]).mean())\n"
           "elif report == 'counts':\n    result = cnt\nelse:\n    result = lab")
    tests = [gen(s + i, 5000, 40, 3) for i in (1, 2)]
    what = {"accuracy": "the **OOB accuracy**: the fraction of rows (among those with at least one out-of-bag tree) "
                        "whose OOB label equals `y`",
            "counts": "for every row, the **number of trees** for which it is out of bag (an int array)",
            "labels": "every row's **OOB label** (an int array; −1 for a row that is in every tree's sample)"}[report]
    return challenge(
        "Out-of-bag votes",
        f"A bagged ensemble of `T` trees was trained on `n` rows with labels `y` in 0..K−1. Tree `t` was fitted on "
        f"the bootstrap rows `boot[t]` (with repeats) and predicts label `pred[t, i]` for row `i`. A row is *out of "
        f"bag* for tree `t` if `i` does not appear in `boot[t]`; its OOB label is the majority vote of those trees "
        f"only (ties: the smallest label). Return {what}. (`report` = {report!r}.)",
        gen(s, n, T, K), sol, tests, "T·n ≤ 2·10^5, K ≤ 3",
        speed="O(T·n·K): build an in-bag mask instead of searching each sample", tol=1e-9)


@template(M, "challenge")
def boosting_stumps_1d(rng):
    n, s = rng.randint(8, 12), seeded(rng)
    T, lr = rng.randint(2, 4), rng.choice([0.3, 0.5, 1.0])
    report = rng.choice(["pred", "mse", "thresholds"])

    def gen(seed, n_, hi, T_):
        return (f"g = np.random.default_rng({seed})\nx = g.integers(0, {hi}, {n_}) / 8\n"
                f"y = (2 * np.sin(x * {64 / hi}) + 0.3 * x * {64 / hi} + g.normal(0, 0.5, {n_})).round(3)\n"
                f"T, lr = {T_}, {lr}\nreport = {report!r}")
    sol = ("o = np.argsort(x, kind='stable')\nxs, ys = x[o], y[o]\nn = len(xs)\ni = np.arange(1, n)\n"
           "ok = xs[1:] > xs[:-1]\nF = np.full(n, ys.mean())\nmse, thr = [], []\nfor _ in range(T):\n"
           "    r = ys - F\n    c = np.cumsum(r)[:-1]\n    tot = r.sum()\n"
           "    gain = np.where(ok, c ** 2 / i + (tot - c) ** 2 / (n - i), -np.inf)\n    b = int(np.argmax(gain))\n"
           "    F[:b + 1] += lr * c[b] / (b + 1)\n    F[b + 1:] += lr * (tot - c[b]) / (n - b - 1)\n"
           "    thr.append(xs[b] / 2 + xs[b + 1] / 2)\n    mse.append(((ys - F) ** 2).mean())\n"
           "pred = np.empty(n)\npred[o] = F\n"
           "result = {'pred': pred, 'mse': np.array(mse), 'thresholds': np.array(thr)}[report]")
    tests = [gen(s + i, 100000, 8000, 15) for i in (1, 2)]
    what = {"pred": "the final **training predictions** (in the original row order)",
            "mse": "the **training MSE after each round** (an array of length `T`)",
            "thresholds": "the **threshold** of each round's stump (an array of length `T`)"}[report]
    return challenge(
        "Gradient boosting with stumps",
        "Reimplement `GradientBoostingRegressor(n_estimators=T, learning_rate=lr, max_depth=1)` on one feature `x`. "
        "Start every prediction at `y.mean()`. Each round: compute the residuals `r = y − prediction`, fit the best "
        "stump to them (try every split between consecutive distinct sorted `x` values, rows with `x <= threshold` "
        "go left, threshold = midpoint `a / 2 + b / 2`; pick the split with the smallest total squared error, the "
        "smallest threshold on ties; each side predicts its mean residual), then add `lr` × the stump's output to "
        f"the predictions. Return {what}. (`report` = {report!r}.)",
        gen(s, n, 64, T), sol, tests, "len(x) ≤ 10^5, T ≤ 15",
        speed="O(n log n + T·n): sort once, prefix sums per round", answer_desc=" (a NumPy float array)", tol=1e-6)


@template(M, "challenge")
def weakest_link_alphas(rng):
    m, s = rng.randint(5, 8), seeded(rng)
    report = rng.choice(["all", "weakest", "count"])

    def gen(seed, m_):
        return (f"g = np.random.default_rng({seed})\nleft, right, leaves = [-1], [-1], [0]\nfor _ in range({m_}):\n"
                "    t = leaves.pop(len(leaves) - 1 - min(int(g.exponential(2)), len(leaves) - 1))\n"
                "    left[t], right[t] = len(left), len(left) + 1\n    left += [-1, -1]\n    right += [-1, -1]\n"
                "    leaves += [len(left) - 2, len(left) - 1]\nleft, right = np.array(left), np.array(right)\n"
                "risk = g.uniform(0, 1, len(left)).round(4)\nextra = (g.uniform(0, 1.5, len(left)) ** 2).round(4)\n"
                "for t in range(len(left) - 1, -1, -1):\n    if left[t] != -1:\n"
                "        risk[t] = risk[left[t]] + risk[right[t]] + extra[t]\ndel extra\n"
                f"alpha = 0.5\nreport = {report!r}")
    sol = ("N = len(left)\nlsum = risk.astype(float).copy()\nnleaf = np.ones(N)\nfor t in range(N - 1, -1, -1):\n"
           "    if left[t] != -1:\n        lsum[t] = lsum[left[t]] + lsum[right[t]]\n"
           "        nleaf[t] = nleaf[left[t]] + nleaf[right[t]]\ninner = np.flatnonzero(left != -1)\n"
           "gt = (risk[inner] - lsum[inner]) / (nleaf[inner] - 1)\nif report == 'all':\n    result = gt\n"
           "elif report == 'weakest':\n    j = int(np.argmin(gt))\n    result = (int(inner[j]), float(gt[j]))\n"
           "else:\n    result = int((gt <= alpha).sum())")
    tests = [gen(s + i, 70000) for i in (1, 2)]
    what = {"all": "`g(t)` for every internal node, in increasing node id (a float array)",
            "weakest": "the **weakest link** as `(node id, g)`: the internal node with the smallest `g(t)` (the "
                       "smallest id on ties)",
            "count": "how many internal nodes have `g(t) <= alpha`"}[report]
    return challenge(
        "Weakest-link pruning",
        "Cost-complexity pruning (`ccp_alpha`, `cost_complexity_pruning_path`) repeatedly cuts the subtree whose "
        "removal costs least per leaf saved. A tree is given by `left`/`right` child arrays (−1 at leaves; children "
        "always have larger ids than their parent) and each node's risk `risk[t]` (in scikit-learn: its impurity × "
        "its share of the training rows). For an internal node `t`, let `R(T_t)` be the total risk of the leaves "
        "below it and `L_t` their number; its effective alpha is `g(t) = (risk[t] − R(T_t)) / (L_t − 1)`. Return "
        f"{what}. (`report` = {report!r}.) The tree can be very deep: avoid recursion and per-node subtree walks.",
        gen(s, m), sol, tests, "nodes ≤ 1.4·10^5",
        speed="O(nodes): one pass from the highest id down", tol=1e-6)


@template(M, "challenge")
def histogram_split_gain(rng):
    n, F, B, s = rng.randint(10, 14), rng.randint(2, 3), rng.randint(3, 5), seeded(rng)
    report = rng.choice(["best", "per_feature"])
    lam, ml = rng.choice([0.1, 1.0]), rng.choice([1, 2])

    def gen(seed, n_, F_, B_):
        return (f"g = np.random.default_rng({seed})\nn, F, B = {n_}, {F_}, {B_}\nbins = g.integers(0, B, (n, F))\n"
                "grad = g.normal(0, 1, n).round(3)\nhess = g.uniform(0.1, 1, n).round(3)\n"
                f"lam, min_leaf = {lam}, {ml}\nreport = {report!r}")
    sol = ("G, H = grad.sum(), hess.sum()\nbest = []\nfor f in range(F):\n"
           "    gl = np.cumsum(np.bincount(bins[:, f], weights=grad, minlength=B))[:-1]\n"
           "    hl = np.cumsum(np.bincount(bins[:, f], weights=hess, minlength=B))[:-1]\n"
           "    cl = np.cumsum(np.bincount(bins[:, f], minlength=B))[:-1]\n"
           "    gain = gl ** 2 / (hl + lam) + (G - gl) ** 2 / (H - hl + lam) - G ** 2 / (H + lam)\n"
           "    best.append(np.where((cl >= min_leaf) & (n - cl >= min_leaf), gain, -np.inf))\n"
           "if report == 'per_feature':\n    result = np.array([b.max() for b in best])\nelse:\n"
           "    f = int(np.argmax([b.max() for b in best]))\n    b = int(np.argmax(best[f]))\n"
           "    result = (f, b, float(best[f][b]))")
    tests = [gen(s + i, 50000, 4, 255) for i in (1, 2)]
    what = {"best": "the best split overall as `(feature, bin, gain)` (ties: the smallest feature, then the smallest "
                    "bin)",
            "per_feature": "each feature's best gain, as a float array of length `F`"}[report]
    return challenge(
        "Histogram split finding",
        "`HistGradientBoosting*` first buckets every feature into integer bins, then finds splits from per-bin sums "
        "of the loss gradients and hessians. Here `bins[i, f]` (in 0..B−1) is row `i`'s bin for feature `f`, and "
        "`grad`, `hess` are the rows' gradients and hessians. Splitting feature `f` after bin `b` (left: `bins[:, f] "
        "<= b`, for b = 0..B−2) has gain `G_L²/(H_L+lam) + G_R²/(H_R+lam) − G²/(H+lam)` (G, H: sums of grad and hess "
        "over a side, or over all rows); a split is only allowed if both sides keep at least `min_leaf` rows. Return "
        f"{what}. (`report` = {report!r}; every feature has at least one allowed split.)",
        gen(s, n, F, B), sol, tests, "n·F ≤ 2·10^5, B ≤ 255",
        speed="O(n·F + B·F): one histogram per feature", tol=1e-6)
