"""4.03 Splitting & cross-validation: train_test_split, the CV splitters, cross_val_score / cross_validate /
cross_val_predict, and the leakage traps of grouped and temporal data."""

from ._common import P, catalog, challenge, lit, seeded, synth_clf, table, template

M = "4.03"
MS = "sklearn.model_selection"


def _binary(rng, n=(120, 300)):
    """A binary problem: make_classification or the (shuffled) breast-cancer data."""
    if rng.random() < 0.3:
        s = seeded(rng)
        return ("from sklearn.datasets import load_breast_cancer\nX, y = load_breast_cancer(return_X_y=True)\n"
                f"order = np.random.default_rng({s}).permutation(len(X))\nX, y = X[order], y[order]",
                "`X, y` are the 569 breast-cancer samples (30 features, rows shuffled; `y`: 0 malignant, 1 benign)", 569)
    d = synth_clf(rng, n=n, d=(4, 8), classes=2)
    return d.setup, d.desc, d.n


def _multi(rng, n=(120, 300)):
    """A classification problem with 2-3 classes: make_classification or shuffled iris / wine."""
    r = rng.random()
    if r < 0.3:
        key, rows, name = rng.choice([("load_iris", 150, "iris flowers"), ("load_wine", 178, "wines")])
        s = seeded(rng)
        return (f"from sklearn.datasets import {key}\nX, y = {key}(return_X_y=True)\n"
                f"order = np.random.default_rng({s}).permutation(len(X))\nX, y = X[order], y[order]",
                f"`X, y` are the {rows} {name} of `{key}` (rows shuffled; 3 classes)", rows)
    d = synth_clf(rng, n=n, d=(4, 8), classes=(2, 3))
    return d.setup, d.desc, d.n


def _model(rng, forest=True):
    """(import line, constructor) of a quick classifier that needs no scaling."""
    opts = [("from sklearn.tree import DecisionTreeClassifier", f"DecisionTreeClassifier(max_depth={rng.randint(2, 5)}, "
                                                              f"random_state=0)"),
            ("from sklearn.neighbors import KNeighborsClassifier", f"KNeighborsClassifier(n_neighbors={rng.choice([3, 5, 7])})"),
            ("from sklearn.naive_bayes import GaussianNB", "GaussianNB()")]
    if forest:
        opts.append(("from sklearn.ensemble import RandomForestClassifier",
                     f"RandomForestClassifier(n_estimators={rng.choice([10, 15, 20])}, random_state=0)"))
    return rng.choice(opts)


# ================================================================= lite

@template(M, "lite")
def train_test_split_basics(rng):
    n, d, s = rng.randint(20, 60), rng.randint(2, 4), rng.randint(0, 99)
    ts = rng.choice([0.2, 0.25, 0.3])
    m = rng.randint(4, 9)
    setup = (f"X = np.arange({n * d}).reshape({n}, {d})\ny = np.array({lit([i % 2 for i in range(n)])})\n"
             f"w = np.linspace(0.5, 1.5, {n})")
    imp = "from sklearn.model_selection import train_test_split\n"
    items = [
        ("train_test_split", f"Split `X, y` with `train_test_split(X, y, test_size={ts}, random_state={s})` and return "
                             f"`(X_train.shape, X_test.shape)`.",
         imp + f"X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={ts}, random_state={s})\n"
               "result = (X_train.shape, X_test.shape)"),
        ("train_test_split", f"An integer `test_size` is a row count. Split with `test_size={m}, random_state={s}` and "
                             f"return `y_test`.",
         imp + f"X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={m}, random_state={s})\nresult = y_test"),
        ("train_test_split", f"With `shuffle=False` the test set is simply the last rows. Split with `test_size={ts}, "
                             f"shuffle=False` and return `X_test`.",
         imp + f"X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={ts}, shuffle=False)\nresult = X_test"),
        ("train_test_split", f"Split with `test_size={ts}, random_state={s}, stratify=y` (keep the class proportions) "
                             f"and return the class counts of `y_test`.",
         imp + f"X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={ts}, random_state={s}, stratify=y)\n"
               "result = np.bincount(y_test)"),
        ("train_test_split", f"`train_test_split` splits any number of arrays the same way. Split `X, y, w` together "
                             f"(`test_size={ts}, random_state={s}`) and return the test part of `w`.",
         imp + f"X_tr, X_te, y_tr, y_te, w_tr, w_te = train_test_split(X, y, w, test_size={ts}, random_state={s})\n"
               "result = w_te"),
        ("train_test_split", f"Split the row **indices** instead: `train_test_split(np.arange(len(X)), test_size={ts}, "
                             f"random_state={s})`, and return the sorted test indices.",
         imp + f"idx_train, idx_test = train_test_split(np.arange(len(X)), test_size={ts}, random_state={s})\n"
               "result = np.sort(idx_test)"),
        ("train_size", f"Keep only {m} rows for training: `train_size={m}, random_state={s}` (the rest is test). Return "
                       f"`(len(X_train), len(X_test))`.",
         imp + f"X_train, X_test, y_train, y_test = train_test_split(X, y, train_size={m}, random_state={s})\n"
               "result = (len(X_train), len(X_test))"),
    ]
    return catalog(rng, items, setup, f"`X` has {n} rows, `y` alternates 0/1 and `w` holds a weight per row. "
                                      f"(`train_test_split` is in `{MS}`.)")


@template(M, "lite")
def splitter_objects(rng):
    n, k, s = rng.randint(9, 16), rng.randint(3, 4), rng.randint(0, 99)
    G = rng.randint(4, 6)
    groups = [rng.randrange(G) for _ in range(n)]
    for g in range(G):
        groups[g] = g          # every group appears
    ys = [rng.randrange(2) for _ in range(n)]
    ys[0], ys[1], ys[2], ys[3], ys[4], ys[5] = 0, 0, 0, 1, 1, 1
    setup = f"X = np.arange({n * 2}).reshape({n}, 2)\ny = np.array({lit(ys)})\ngroups = np.array({lit(groups)})"
    imp = f"from {MS} import "
    items = [
        ("KFold", f"Return the test indices of every fold of `KFold(n_splits={k})` (no shuffling) as a list of lists.",
         imp + f"KFold\nresult = [te.tolist() for tr, te in KFold(n_splits={k}).split(X)]"),
        ("KFold", f"Return the training-set sizes of the folds of `KFold(n_splits={k}, shuffle=True, random_state={s})`.",
         imp + f"KFold\nresult = [len(tr) for tr, te in KFold(n_splits={k}, shuffle=True, random_state={s}).split(X)]"),
        ("KFold", f"Return the test indices of the **first** fold of `KFold(n_splits={k}, shuffle=True, "
                  f"random_state={s})`.",
         imp + f"KFold\ntr, te = next(iter(KFold(n_splits={k}, shuffle=True, random_state={s}).split(X)))\nresult = te"),
        ("LeaveOneOut", "How many splits does `LeaveOneOut()` make on `X`? Return `get_n_splits(X)`.",
         imp + "LeaveOneOut\nresult = LeaveOneOut().get_n_splits(X)"),
        ("ShuffleSplit", f"Return the test indices of each split of `ShuffleSplit(n_splits=3, test_size=0.25, "
                         f"random_state={s})`, as a list of arrays.",
         imp + f"ShuffleSplit\nresult = [te for tr, te in ShuffleSplit(n_splits=3, test_size=0.25, random_state={s}).split(X)]"),
        ("TimeSeriesSplit", f"Return `(len(train), len(test))` for each split of `TimeSeriesSplit(n_splits={k})`.",
         imp + f"TimeSeriesSplit\nresult = [(len(tr), len(te)) for tr, te in TimeSeriesSplit(n_splits={k}).split(X)]"),
        ("StratifiedKFold", "Return the class counts of `y` in each test fold of `StratifiedKFold(n_splits=3)` "
                            "(a list of `np.bincount(y[te], minlength=2)` arrays).",
         imp + "StratifiedKFold\nresult = [np.bincount(y[te], minlength=2) for tr, te in StratifiedKFold(n_splits=3).split(X, y)]"),
        ("GroupKFold", f"Return the sorted groups of each test fold of `GroupKFold(n_splits=3)` (pass `groups=groups`), "
                       f"as a list of lists.",
         imp + "GroupKFold\nresult = [sorted(set(groups[te].tolist())) for tr, te in GroupKFold(n_splits=3).split(X, y, groups=groups)]"),
        ("RepeatedStratifiedKFold", f"How many splits does `RepeatedStratifiedKFold(n_splits=3, n_repeats={k}, "
                                    f"random_state={s})` make? Return `get_n_splits()`.",
         imp + f"RepeatedStratifiedKFold\nresult = RepeatedStratifiedKFold(n_splits=3, n_repeats={k}, random_state={s}).get_n_splits()"),
        ("StratifiedShuffleSplit", f"Return the class counts of the test set of `StratifiedShuffleSplit(n_splits=1, "
                                   f"test_size=0.5, random_state={s})`.",
         imp + f"StratifiedShuffleSplit\ntr, te = next(StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state={s})"
               ".split(X, y))\nresult = np.bincount(y[te])"),
    ]
    return catalog(rng, items, setup, f"`X` has {n} rows, `y` holds 0/1 labels and `groups` a group id per row. "
                                      f"Splitters live in `{MS}`; `split` yields `(train_indices, test_indices)` pairs.")


@template(M, "lite")
def cv_score_calls(rng):
    setup, desc, n = _binary(rng)
    imp, ctor = _model(rng)
    k = rng.choice([3, 4, 5])
    m = rng.randint(4, 8)
    full = setup + f"\n{imp}\nmodel = {ctor}"
    items = [
        ("cross_val_score", f"Return the {k} accuracy scores of `cross_val_score(model, X, y, cv={k})`.",
         f"from {MS} import cross_val_score\nresult = cross_val_score(model, X, y, cv={k})"),
        ("cross_val_score", f"Return the mean {k}-fold cross-validated accuracy of `model`.",
         f"from {MS} import cross_val_score\nresult = cross_val_score(model, X, y, cv={k}).mean()"),
        ("scoring", f"Return the mean {k}-fold cross-validated F1 score (`scoring='f1'`).",
         f"from {MS} import cross_val_score\nresult = cross_val_score(model, X, y, cv={k}, scoring='f1').mean()",
         {"require": ["cross_val_score", "scoring"]}),
        ("scoring", f"Return the mean {k}-fold cross-validated ROC AUC (`scoring='roc_auc'`).",
         f"from {MS} import cross_val_score\nresult = cross_val_score(model, X, y, cv={k}, scoring='roc_auc').mean()",
         {"require": ["cross_val_score", "scoring"]}),
        ("cross_validate", f"`cross_validate` returns a dict. Run it with `cv={k}` and return its sorted keys.",
         f"from {MS} import cross_validate\nresult = sorted(cross_validate(model, X, y, cv={k}))"),
        ("cross_validate", f"Run `cross_validate(model, X, y, cv={k})` and return its `'test_score'` array.",
         f"from {MS} import cross_validate\nresult = cross_validate(model, X, y, cv={k})['test_score']"),
        ("cross_val_predict", f"`cross_val_predict` returns one out-of-fold prediction per row. Return the first {m} "
                              f"of them with `cv={k}`.",
         f"from {MS} import cross_val_predict\nresult = cross_val_predict(model, X, y, cv={k})[:{m}]"),
        ("cross_val_predict", f"Return the out-of-fold accuracy: the fraction of rows where "
                              f"`cross_val_predict(model, X, y, cv={k})` equals `y`.",
         f"from {MS} import cross_val_predict\nresult = (cross_val_predict(model, X, y, cv={k}) == y).mean()"),
    ]
    return catalog(rng, items, full, f"{desc}; `model = {ctor}` (not fitted). CV helpers live in `{MS}`.", tol=1e-9)


# ================================================================= easy

@template(M, "easy")
def kfold_by_hand(rng):
    n, k, s = rng.randint(11, 40), rng.randint(3, 6), rng.randint(0, 99)
    i = rng.randrange(n)
    task = rng.choice(["sizes", "which", "shuffled", "train"])
    if task == "sizes":
        return P(f"`KFold(n_splits={k})` on {n} rows makes the first `n % k` folds one row larger than `n // k`. Compute "
                 f"the test-fold sizes **by hand** (no scikit-learn) as a list.", f"n, k = {n}, {k}",
                 "result = [n // k + (1 if f < n % k else 0) for f in range(k)]", require=[])
    if task == "which":
        return P(f"Without shuffling, `KFold(n_splits={k})` cuts the {n} rows into consecutive blocks. Which fold "
                 f"(0-based) holds row {i} in its test set? Use `KFold(...).split` and return the fold number.",
                 f"X = np.zeros(({n}, 1))",
                 f"from {MS} import KFold\nresult = next(f for f, (tr, te) in enumerate(KFold(n_splits={k}).split(X)) "
                 f"if {i} in te)", require=["KFold"])
    if task == "shuffled":
        return P(f"`KFold(n_splits={k}, shuffle=True, random_state={s})` shuffles the row indices with "
                 f"`np.random.RandomState({s}).permutation({n})` and cuts the result into consecutive blocks "
                 f"(`np.array_split`). Rebuild the test folds that way and return them **sorted** (a list of arrays; "
                 f"they equal what `KFold` yields).", f"n, k = {n}, {k}",
                 f"perm = np.random.RandomState({s}).permutation(n)\nresult = [np.sort(f) for f in np.array_split(perm, k)]",
                 require=["permutation"])
    return P(f"Return the training-set sizes of `KFold(n_splits={k})` on {n} rows, and check them against the rule "
             f"`n − test size`: return `(from_kfold, by_rule)` as two lists.", f"X = np.zeros(({n}, 1))",
             f"from {MS} import KFold\nfrom_kfold = [len(tr) for tr, te in KFold(n_splits={k}).split(X)]\n"
             f"by_rule = [{n} - ({n} // {k} + (1 if f < {n} % {k} else 0)) for f in range({k})]\nresult = (from_kfold, by_rule)",
             require=["KFold"])


@template(M, "easy")
def stratified_vs_plain(rng):
    key, rows = rng.choice([("load_iris", 150), ("load_wine", 178)])
    k = rng.choice([3, 5])
    depth = rng.randint(2, 4)
    setup = f"from sklearn.datasets import {key}\nX, y = {key}(return_X_y=True)"
    head = f"`X, y = {key}(return_X_y=True)`: {rows} rows **sorted by class** (0s, then 1s, then 2s)."
    tree = f"DecisionTreeClassifier(max_depth={depth}, random_state=0)"
    task = rng.choice(["counts", "missing", "kfold_scores", "skf_scores"])
    if task == "counts":
        return P(f"{head} Return `(plain, strat)`: the per-fold test class counts (`np.bincount(y[te], minlength=3)`, "
                 f"stacked into a {k} × 3 array) for `KFold(n_splits={k})` and for `StratifiedKFold(n_splits={k})`.",
                 setup, f"from {MS} import KFold, StratifiedKFold\n"
                        f"plain = np.array([np.bincount(y[te], minlength=3) for tr, te in KFold(n_splits={k}).split(X)])\n"
                        f"strat = np.array([np.bincount(y[te], minlength=3) for tr, te in StratifiedKFold(n_splits={k})"
                        ".split(X, y)])\nresult = (plain, strat)", require=["KFold", "StratifiedKFold"])
    if task == "missing":
        return P(f"{head} With `KFold(n_splits={k})` (no shuffle), how many test folds hold a **single** class? Return "
                 f"an int.", setup,
                 f"from {MS} import KFold\nresult = sum(len(np.unique(y[te])) == 1 for tr, te in KFold(n_splits={k}).split(X))",
                 require=["KFold"])
    if task == "kfold_scores":
        return P(f"{head} Return `cross_val_score({tree}, X, y, cv=KFold(n_splits={k}))`: with unshuffled plain folds "
                 f"the model is tested on classes it barely saw.", setup,
                 f"from {MS} import KFold, cross_val_score\nfrom sklearn.tree import DecisionTreeClassifier\n"
                 f"result = cross_val_score({tree}, X, y, cv=KFold(n_splits={k}))", require=["KFold", "cross_val_score"])
    return P(f"{head} Return the fold scores of `{tree}` with `cv=StratifiedKFold(n_splits={k})` and with "
             f"`cv=KFold(n_splits={k})`, as a tuple `(stratified, plain)`.", setup,
             f"from {MS} import KFold, StratifiedKFold, cross_val_score\nfrom sklearn.tree import DecisionTreeClassifier\n"
             f"result = (cross_val_score({tree}, X, y, cv=StratifiedKFold(n_splits={k})), "
             f"cross_val_score({tree}, X, y, cv=KFold(n_splits={k})))", require=["StratifiedKFold", "cross_val_score"])


@template(M, "easy")
def time_series_folds(rng):
    n, k = rng.randint(12, 30), rng.randint(2, 5)
    mx, gap = rng.randint(3, 6), rng.randint(1, 3)
    gap = max(1, min(gap, n - k * (n // (k + 1)) - 1))
    ts = rng.randint(2, max(2, (n - gap) // (k + 1)))
    setup = f"t = np.arange({n})"
    task = rng.choice(["sizes", "max_train", "gap", "test_size"])
    imp = f"from {MS} import TimeSeriesSplit\n"
    head = f"`t = np.arange({n})` are time steps. `TimeSeriesSplit` always trains on the past and tests on the future."
    if task == "sizes":
        return P(f"{head} Return `(len(train), len(test))` for each split of `TimeSeriesSplit(n_splits={k})`.", setup,
                 imp + f"result = [(len(tr), len(te)) for tr, te in TimeSeriesSplit(n_splits={k}).split(t)]",
                 require=["TimeSeriesSplit"])
    if task == "max_train":
        return P(f"{head} `max_train_size={mx}` keeps only the most recent rows. Return `(train[0], train[-1], "
                 f"test[0], test[-1])` for each split of `TimeSeriesSplit(n_splits={k}, max_train_size={mx})`.", setup,
                 imp + f"result = [(tr[0], tr[-1], te[0], te[-1]) for tr, te in TimeSeriesSplit(n_splits={k}, "
                       f"max_train_size={mx}).split(t)]", require=["TimeSeriesSplit"])
    if task == "gap":
        return P(f"{head} `gap={gap}` leaves {gap} step(s) out between train and test. Return `(train[-1], test[0])` "
                 f"for each split of `TimeSeriesSplit(n_splits={k}, gap={gap})`.", setup,
                 imp + f"result = [(tr[-1], te[0]) for tr, te in TimeSeriesSplit(n_splits={k}, gap={gap}).split(t)]",
                 require=["TimeSeriesSplit"])
    return P(f"{head} Return the first test index of each split of `TimeSeriesSplit(n_splits={k}, test_size={ts})`.",
             setup, imp + f"result = [te[0] for tr, te in TimeSeriesSplit(n_splits={k}, test_size={ts}).split(t)]",
             require=["TimeSeriesSplit"])


@template(M, "easy")
def three_way_split(rng):
    setup, desc, n = _multi(rng)
    s = rng.randint(0, 99)
    task = rng.choice(["sizes", "proportions", "val_rows", "drill"])
    two = (f"from {MS} import train_test_split\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, "
           f"random_state={s}, stratify=y)\nX_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.25, "
           f"random_state={s}, stratify=y_train)\n")
    how = (f"Make a 60/20/20 train/validation/test split with two `train_test_split` calls: first hold out 20% as test "
           f"(`test_size=0.2, random_state={s}, stratify=y`), then split the remaining 80% with `test_size=0.25, "
           f"random_state={s}, stratify=y_train` (25% of 80% = 20%).")
    if task == "sizes":
        return P(f"{desc}. {how} Return `(len(X_tr), len(X_val), len(X_test))`.", setup,
                 two + "result = (len(X_tr), len(X_val), len(X_test))", require=["train_test_split"])
    if task == "proportions":
        return P(f"{desc}. {how} Return the class proportions of `y_tr`, `y_val` and `y_test` (each `np.bincount(...) / "
                 f"len(...)`), as a tuple of three arrays: stratifying keeps them close.", setup,
                 two + "result = tuple(np.bincount(v) / len(v) for v in (y_tr, y_val, y_test))", require=["train_test_split"])
    if task == "val_rows":
        return P(f"{desc}. {how} Return `y_val`.", setup, two + "result = y_val", require=["train_test_split"])
    return P(f"{desc}. Make a **70/15/15** stratified split: hold out 30% (`test_size=0.3, random_state={s}, "
             f"stratify=y`), then cut that 30% in half (`test_size=0.5, random_state={s}`, stratified). Return the "
             f"sizes `(train, validation, test)`.", setup,
             f"from {MS} import train_test_split\nX_tr, X_rest, y_tr, y_rest = train_test_split(X, y, test_size=0.3, "
             f"random_state={s}, stratify=y)\nX_val, X_te, y_val, y_te = train_test_split(X_rest, y_rest, test_size=0.5, "
             f"random_state={s}, stratify=y_rest)\nresult = (len(X_tr), len(X_val), len(X_te))", require=["train_test_split"])


@template(M, "easy")
def group_split_check(rng):
    G, s = rng.randint(8, 14), rng.randint(0, 99)
    k = rng.choice([3, 4])
    sizes = [rng.randint(2, 6) for _ in range(G)]
    setup = (f"rng = np.random.default_rng({s})\ngroups = np.repeat(np.arange({G}), {sizes})\n"
             f"X = rng.normal(size=(len(groups), 3))\ny = rng.integers(0, 2, len(groups))")
    head = f"`groups` gives the patient each row comes from ({G} patients, several rows each); `X, y` are the rows."
    task = rng.choice(["groups", "leak", "rows", "scores"])
    if task == "groups":
        return P(f"{head} Return the sorted patient ids of each test fold of `GroupKFold(n_splits={k})`, as a list of "
                 f"lists. No patient appears in two folds.", setup,
                 f"from {MS} import GroupKFold\nresult = [sorted(set(groups[te].tolist())) for tr, te in "
                 f"GroupKFold(n_splits={k}).split(X, y, groups)]", require=["GroupKFold"])
    if task == "leak":
        return P(f"{head} A plain shuffled `KFold(n_splits={k}, shuffle=True, random_state={s})` ignores patients. For "
                 f"its **first** fold, return the sorted list of patients that have rows on both sides (train and test).",
                 setup, f"from {MS} import KFold\ntr, te = next(iter(KFold(n_splits={k}, shuffle=True, random_state={s})"
                        ".split(X)))\nresult = sorted(set(groups[tr].tolist()) & set(groups[te].tolist()))",
                 require=["KFold"])
    if task == "rows":
        return P(f"{head} `GroupKFold` balances the number of **rows** per fold. Return the test-fold sizes of "
                 f"`GroupKFold(n_splits={k})`.", setup,
                 f"from {MS} import GroupKFold\nresult = [len(te) for tr, te in GroupKFold(n_splits={k}).split(X, y, groups)]",
                 require=["GroupKFold"])
    return P(f"{head} Return `cross_val_score(GaussianNB(), X, y, groups=groups, cv=GroupKFold(n_splits={k}))`.", setup,
             f"from {MS} import GroupKFold, cross_val_score\nfrom sklearn.naive_bayes import GaussianNB\n"
             f"result = cross_val_score(GaussianNB(), X, y, groups=groups, cv=GroupKFold(n_splits={k}))",
             require=["GroupKFold", "cross_val_score"])


@template(M, "easy")
def cv_scoring_options(rng):
    setup, desc, n = _binary(rng)
    k = rng.choice([3, 5])
    depth = rng.choice([None, 8, 12])
    full = setup + f"\nfrom sklearn.tree import DecisionTreeClassifier\nmodel = DecisionTreeClassifier(max_depth={depth}, random_state=0)"
    head = f"{desc}; `model = DecisionTreeClassifier(max_depth={depth}, random_state=0)`."
    task = rng.choice(["two_metrics", "gap", "balanced", "dict"])
    if task == "two_metrics":
        return P(f"{head} Run `cross_validate` with `cv={k}`, `scoring=['accuracy', 'f1']` and `return_train_score=True`; "
                 f"return `(mean test F1, mean train accuracy)`.", full,
                 f"from {MS} import cross_validate\nres = cross_validate(model, X, y, cv={k}, scoring=['accuracy', 'f1'], "
                 "return_train_score=True)\nresult = (res['test_f1'].mean(), res['train_accuracy'].mean())",
                 require=["cross_validate", "return_train_score"])
    if task == "gap":
        return P(f"{head} A deep tree overfits. Return mean train accuracy minus mean test accuracy from "
                 f"`cross_validate(..., cv={k}, return_train_score=True)`.", full,
                 f"from {MS} import cross_validate\nres = cross_validate(model, X, y, cv={k}, return_train_score=True)\n"
                 "result = res['train_score'].mean() - res['test_score'].mean()", require=["cross_validate", "return_train_score"])
    if task == "balanced":
        return P(f"{head} Return the mean {k}-fold CV score under `scoring='accuracy'` and under "
                 f"`scoring='balanced_accuracy'`, as a tuple.", full,
                 f"from {MS} import cross_val_score\nresult = (cross_val_score(model, X, y, cv={k}, scoring='accuracy').mean(), "
                 f"cross_val_score(model, X, y, cv={k}, scoring='balanced_accuracy').mean())",
                 require=["cross_val_score", "scoring"])
    return P(f"{head} `scoring` may be a dict that names each metric. Run `cross_validate(model, X, y, cv={k}, "
             f"scoring={{'acc': 'accuracy', 'prec': 'precision', 'rec': 'recall'}})` and return a dict mapping each "
             f"name to its mean test score.", full,
             f"from {MS} import cross_validate\nres = cross_validate(model, X, y, cv={k}, scoring={{'acc': 'accuracy', "
             "'prec': 'precision', 'rec': 'recall'})\nresult = {m: res['test_' + m].mean() for m in ('acc', 'prec', 'rec')}",
             require=["cross_validate", "scoring"])


@template(M, "easy")
def leave_one_out_small(rng):
    n, s = rng.randint(16, 40), rng.randint(0, 99)
    k = rng.choice([1, 3, 5])
    task = rng.choice(["acc", "errors", "count", "regression"])
    if task == "regression":
        setup = (f"rng = np.random.default_rng({s})\nX = rng.uniform(0, 10, ({n}, 1))\n"
                 f"y = np.sin(X[:, 0]) + rng.normal(0, 0.2, {n})")
        return P(f"`X` ({n} × 1) and `y` (a noisy sine). Run `cross_val_score(KNeighborsRegressor(n_neighbors={k}), X, "
                 f"y, cv=LeaveOneOut(), scoring='neg_mean_squared_error')` and return the leave-one-out **MSE** (a "
                 f"positive number).", setup,
                 f"from {MS} import LeaveOneOut, cross_val_score\nfrom sklearn.neighbors import KNeighborsRegressor\n"
                 f"result = -cross_val_score(KNeighborsRegressor(n_neighbors={k}), X, y, cv=LeaveOneOut(), "
                 "scoring='neg_mean_squared_error').mean()", require=["LeaveOneOut", "cross_val_score"])
    setup = (f"rng = np.random.default_rng({s})\nX = rng.normal(size=({n}, 2))\n"
             f"y = (X[:, 0] + X[:, 1] + rng.normal(0, 0.7, {n}) > 0).astype(int)")
    head = (f"`X` ({n} × 2) and `y` (0/1). Leave-one-out CV fits {n} models, each tested on the single row it left "
            f"out. Model: `KNeighborsClassifier(n_neighbors={k})`.")
    sol = (f"from {MS} import LeaveOneOut, cross_val_score\nfrom sklearn.neighbors import KNeighborsClassifier\n"
           f"scores = cross_val_score(KNeighborsClassifier(n_neighbors={k}), X, y, cv=LeaveOneOut())\n")
    if task == "acc":
        return P(f"{head} Return the leave-one-out accuracy (`cv=LeaveOneOut()`).", setup, sol + "result = scores.mean()",
                 require=["LeaveOneOut"])
    if task == "errors":
        return P(f"{head} Each fold scores 0 or 1. Return the indices of the rows the left-out model gets wrong.", setup,
                 sol + "result = np.flatnonzero(scores == 0)", require=["LeaveOneOut"])
    return P(f"{head} Return `(LeaveOneOut().get_n_splits(X), number of scores, number of distinct score values)`.",
             setup, sol + "result = (LeaveOneOut().get_n_splits(X), len(scores), len(np.unique(scores)))",
             require=["LeaveOneOut"])


@template(M, "easy")
def split_then_score(rng):
    setup, desc, n = _multi(rng)
    imp, ctor = _model(rng, forest=False)
    s, s2 = rng.randint(0, 99), rng.randint(100, 199)
    ts = rng.choice([0.2, 0.25, 0.3])
    split = (f"from {MS} import train_test_split\nX_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size={ts}, "
             f"random_state={s}, stratify=y)\n")
    how = f"Split with `train_test_split(X, y, test_size={ts}, random_state={s}, stratify=y)`"
    items = [
        ("train_test_split", f"{how}, fit `model` on the training part and return its test accuracy.",
         split + "result = model.fit(X_tr, y_tr).score(X_te, y_te)"),
        ("train_test_split", f"{how}, fit `model` and return `(train accuracy, test accuracy)`.",
         split + "model.fit(X_tr, y_tr)\nresult = (model.score(X_tr, y_tr), model.score(X_te, y_te))"),
        ("train_test_split", f"{how}, fit `model` and return the indices (within `X_te`) of the test rows it gets "
                             f"wrong.", split + "result = np.flatnonzero(model.fit(X_tr, y_tr).predict(X_te) != y_te)"),
        ("stratify", f"Return the class proportions of the test set with and without stratifying (`train_test_split(X, "
                     f"y, test_size={ts}, random_state={s})` with and without `stratify=y`), as a tuple of two arrays.",
         f"from {MS} import train_test_split\n_, _, _, a = train_test_split(X, y, test_size={ts}, random_state={s}, stratify=y)\n"
         f"_, _, _, b = train_test_split(X, y, test_size={ts}, random_state={s})\n"
         "k = len(np.unique(y))\nresult = (np.bincount(a, minlength=k) / len(a), np.bincount(b, minlength=k) / len(b))"),
        ("random_state", f"One split is one noisy estimate. Return the test accuracies of `model` for two splits that "
                         f"differ only in `random_state` ({s} and {s2}; `test_size={ts}`, stratified), as a tuple.",
         f"from {MS} import train_test_split\nout = []\nfor rs in ({s}, {s2}):\n"
         f"    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size={ts}, random_state=rs, stratify=y)\n"
         "    out.append(model.fit(X_tr, y_tr).score(X_te, y_te))\nresult = tuple(out)"),
        ("train_test_split", f"Split a DataFrame: `df = pd.DataFrame(X)`; `train_test_split(df, y, test_size={ts}, "
                             f"random_state={s})` keeps the row labels. Return the sorted index labels of the test part.",
         f"from {MS} import train_test_split\ndf = pd.DataFrame(X)\ndf_tr, df_te, y_tr, y_te = train_test_split(df, y, "
         f"test_size={ts}, random_state={s})\nresult = np.sort(df_te.index.to_numpy())"),
        ("train_test_split", f"{how}, fit `model` and return how many test rows of each class it predicts "
                             f"(`np.bincount` with `minlength` = number of classes).",
         split + "result = np.bincount(model.fit(X_tr, y_tr).predict(X_te), minlength=len(np.unique(y)))"),
    ]
    return catalog(rng, items, setup + f"\n{imp}\nmodel = {ctor}", f"{desc}; `model = {ctor}`.", tol=1e-9)


# ================================================================= medium

@template(M, "medium")
def cv_on_tables(rng):
    t = table(rng, kind="clf", numeric_only=True, n=(150, 300))
    s = rng.randint(0, 99)
    depth = rng.randint(2, 4)
    depths = sorted(rng.sample([1, 2, 3, 4, 6, 8], 3))
    col = rng.choice(t.num)
    model = f"DecisionTreeClassifier(max_depth={depth}, random_state=0)"
    skf = f"StratifiedKFold(n_splits=5, shuffle=True, random_state={s})"
    items = [
        ("cross_val_score", "Return the mean 5-fold CV accuracy of `model` (`cv=5`).",
         f"from {MS} import cross_val_score\nresult = cross_val_score(model, df, y, cv=5).mean()"),
        ("KFold", f"Return the fold accuracies with `cv=KFold(n_splits=5, shuffle=True, random_state={s})`.",
         f"from {MS} import KFold, cross_val_score\nresult = cross_val_score(model, df, y, cv=KFold(n_splits=5, "
         f"shuffle=True, random_state={s}))"),
        ("cross_validate", "Return `(mean train accuracy, mean test accuracy)` from `cross_validate(model, df, y, cv=5, "
                           "return_train_score=True)`.",
         f"from {MS} import cross_validate\nres = cross_validate(model, df, y, cv=5, return_train_score=True)\n"
         "result = (res['train_score'].mean(), res['test_score'].mean())"),
        ("cross_val_score", f"Compare depths: for `max_depth` in `{depths}`, return a dict depth → mean 5-fold CV accuracy "
                            f"of `DecisionTreeClassifier(max_depth=depth, random_state=0)`.",
         f"from {MS} import cross_val_score\nfrom sklearn.tree import DecisionTreeClassifier\n"
         f"result = {{d: cross_val_score(DecisionTreeClassifier(max_depth=d, random_state=0), df, y, cv=5).mean() "
         f"for d in {depths}}}"),
        ("StratifiedKFold", f"With `cv={skf}`, which fold (0-based) gets the **lowest** accuracy? Return its number.",
         f"from {MS} import StratifiedKFold, cross_val_score\nresult = int(np.argmin(cross_val_score(model, df, y, "
         f"cv={skf})))"),
        ("cross_val_score", "Return the standard deviation of the 5 fold accuracies (`cv=5`): how much the estimate "
                            "wobbles.", f"from {MS} import cross_val_score\nresult = cross_val_score(model, df, y, cv=5).std()"),
        ("RepeatedStratifiedKFold", f"Repeat 5-fold CV 3 times with different shuffles: `RepeatedStratifiedKFold("
                                    f"n_splits=5, n_repeats=3, random_state={s})`. Return `(number of scores, mean)`.",
         f"from {MS} import RepeatedStratifiedKFold, cross_val_score\nsc = cross_val_score(model, df, y, "
         f"cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state={s}))\nresult = (len(sc), sc.mean())"),
        ("scoring", "Return the mean 5-fold CV F1 score (`scoring='f1'`) of `model`.",
         f"from {MS} import cross_val_score\nresult = cross_val_score(model, df, y, cv=5, scoring='f1').mean()",
         {"require": ["cross_val_score", "scoring"]}),
        ("cross_val_score", f"Does `{col}` help? Return the mean 5-fold CV accuracy with all columns minus the one "
                            f"without `{col}`.",
         f"from {MS} import cross_val_score\nresult = cross_val_score(model, df, y, cv=5).mean() - cross_val_score(model, "
         f"df.drop(columns={col!r}), y, cv=5).mean()"),
        ("RandomForestClassifier", f"Return the mean 5-fold CV accuracy of `RandomForestClassifier(n_estimators=20, "
                                   f"random_state=0)` minus that of `model`.",
         f"from {MS} import cross_val_score\nfrom sklearn.ensemble import RandomForestClassifier\n"
         "result = cross_val_score(RandomForestClassifier(n_estimators=20, random_state=0), df, y, cv=5).mean() - "
         "cross_val_score(model, df, y, cv=5).mean()"),
    ]
    return catalog(rng, items, t.setup + f"\nfrom sklearn.tree import DecisionTreeClassifier\nmodel = {model}",
                   f"{t.desc}; `model = {model}`.", tol=1e-9)


@template(M, "medium")
def manual_cv_loop(rng):
    setup, desc, n = _binary(rng, n=(150, 300))
    k, s = rng.choice([3, 4, 5]), rng.randint(0, 99)
    nb = rng.choice([3, 5, 9])
    skf = f"StratifiedKFold(n_splits={k}, shuffle=True, random_state={s})"
    head = (f"{desc}. Write the manual cross-validation loop: `for tr_idx, va_idx in skf.split(X, y)` with `skf = {skf}`, "
            f"fitting a fresh model on the training rows each time.")
    loop = f"from {MS} import StratifiedKFold\nskf = {skf}\n"
    task = rng.choice(["scores", "sizes", "scaled", "f1", "same"])
    if task == "scores":
        return P(f"{head} Model: `GaussianNB()`. Collect the validation accuracies in a list and return it.", setup,
                 loop + "from sklearn.naive_bayes import GaussianNB\nscores = []\nfor tr_idx, va_idx in skf.split(X, y):\n"
                        "    m = GaussianNB().fit(X[tr_idx], y[tr_idx])\n    scores.append(m.score(X[va_idx], y[va_idx]))\n"
                        "result = scores", require=["StratifiedKFold", "split"])
    if task == "sizes":
        return P(f"{head} Return `(len(tr_idx), len(va_idx), positives in va_idx)` for every fold, as a list of tuples.",
                 setup, loop + "result = [(len(tr), len(va), int(y[va].sum())) for tr, va in skf.split(X, y)]",
                 require=["StratifiedKFold", "split"])
    if task == "scaled":
        return P(f"{head} Model: `KNeighborsClassifier(n_neighbors={nb})` on features standardized by a "
                 f"`StandardScaler` fitted on **each fold's training rows only**. Return the list of fold accuracies.",
                 setup, loop + "from sklearn.neighbors import KNeighborsClassifier\nfrom sklearn.preprocessing import "
                               "StandardScaler\nscores = []\nfor tr, va in skf.split(X, y):\n    sc = StandardScaler().fit(X[tr])\n"
                               f"    m = KNeighborsClassifier(n_neighbors={nb}).fit(sc.transform(X[tr]), y[tr])\n"
                               "    scores.append(m.score(sc.transform(X[va]), y[va]))\nresult = scores",
                 require=["StratifiedKFold", "StandardScaler"])
    if task == "f1":
        return P(f"{head} Model: `GaussianNB()`. For each fold compute the F1 score of class 1 **by hand** (2·TP / "
                 f"(2·TP + FP + FN)) and return the mean over folds.", setup,
                 loop + "from sklearn.naive_bayes import GaussianNB\nf1s = []\nfor tr, va in skf.split(X, y):\n"
                        "    p = GaussianNB().fit(X[tr], y[tr]).predict(X[va])\n    tp = ((p == 1) & (y[va] == 1)).sum()\n"
                        "    fp = ((p == 1) & (y[va] == 0)).sum()\n    fn = ((p == 0) & (y[va] == 1)).sum()\n"
                        "    f1s.append(2 * tp / (2 * tp + fp + fn))\nresult = float(np.mean(f1s))",
                 require=["StratifiedKFold"])
    return P(f"{head} Model: `DecisionTreeClassifier(max_depth=3, random_state=0)`. Return `(manual_scores, "
             f"cross_val_score(model, X, y, cv=skf))`: passing the splitter object gives the very same folds.", setup,
             loop + f"from {MS} import cross_val_score\nfrom sklearn.tree import DecisionTreeClassifier\n"
                    "model = DecisionTreeClassifier(max_depth=3, random_state=0)\n"
                    "manual = np.array([model.fit(X[tr], y[tr]).score(X[va], y[va]) for tr, va in skf.split(X, y)])\n"
                    "result = (manual, cross_val_score(model, X, y, cv=skf))", require=["StratifiedKFold", "cross_val_score"])


@template(M, "medium")
def out_of_fold_predictions(rng):
    setup, desc, n = _binary(rng, n=(150, 300))
    k, m = rng.choice([3, 5]), rng.randint(4, 8)
    imp, ctor = _model(rng)
    full = setup + f"\n{imp}\nmodel = {ctor}"
    head = f"{desc}; `model = {ctor}`. `cross_val_predict` gives each row the prediction of the model that did **not** see it."
    oof = f"from {MS} import cross_val_predict\n"
    task = rng.choice(["proba", "auc", "acc_vs_mean", "overfit", "decision"])
    if task == "proba":
        return P(f"{head} Return the out-of-fold P(class 1) of the first {m} rows (`method='predict_proba'`, `cv={k}`).",
                 full, oof + f"result = cross_val_predict(model, X, y, cv={k}, method='predict_proba')[:{m}, 1]",
                 require=["cross_val_predict", "method"])
    if task == "auc":
        return P(f"{head} Return `(out-of-fold AUC, in-sample AUC)`: `roc_auc_score` (from `sklearn.metrics`) of "
                 f"`y` against the out-of-fold P(class 1) (`cv={k}`), and against the P(class 1) of `model` fitted on "
                 f"all rows.", full,
                 oof + "from sklearn.metrics import roc_auc_score\n"
                       f"p_oof = cross_val_predict(model, X, y, cv={k}, method='predict_proba')[:, 1]\n"
                       "p_in = model.fit(X, y).predict_proba(X)[:, 1]\nresult = (roc_auc_score(y, p_oof), roc_auc_score(y, p_in))",
                 require=["cross_val_predict", "roc_auc_score"])
    if task == "acc_vs_mean":
        return P(f"{head} Return `(accuracy of the out-of-fold predictions, mean of cross_val_score)` with `cv={k}`: "
                 f"close, but not equal when the folds have different sizes.", full,
                 f"from {MS} import cross_val_predict, cross_val_score\n"
                 f"result = ((cross_val_predict(model, X, y, cv={k}) == y).mean(), cross_val_score(model, X, y, cv={k}).mean())",
                 require=["cross_val_predict", "cross_val_score"])
    if task == "overfit":
        return P(f"{head} How many rows does `model` (fitted on all rows) classify correctly in-sample while its "
                 f"out-of-fold prediction (`cv={k}`) is wrong? Return an int.", full,
                 oof + f"oof = cross_val_predict(model, X, y, cv={k})\nins = model.fit(X, y).predict(X)\n"
                       "result = int(((ins == y) & (oof != y)).sum())", require=["cross_val_predict"])
    d = synth_clf(rng, n=(150, 300), d=(4, 8), classes=2)
    return P(f"{d.desc}. Use `LogisticRegression(max_iter=1000)` with `cross_val_predict(..., cv={k}, "
             f"method='decision_function')` and return the out-of-fold margins of the first {m} rows.", d.setup,
             oof + "from sklearn.linear_model import LogisticRegression\n"
                   f"result = cross_val_predict(LogisticRegression(max_iter=1000), X, y, cv={k}, "
                   f"method='decision_function')[:{m}]", require=["cross_val_predict", "method"], tol=1e-4)


@template(M, "medium")
def repeated_and_shuffled(rng):
    setup, desc, n = _multi(rng, n=(100, 200))
    s, ns_ = rng.randint(0, 99), rng.randint(3, 6)
    ts = rng.choice([0.2, 0.25, 0.3])
    reps = rng.randint(2, 4)
    full = setup + "\nfrom sklearn.naive_bayes import GaussianNB\nmodel = GaussianNB()"
    head = f"{desc}; `model = GaussianNB()`."
    task = rng.choice(["repeated", "never", "max_count", "sss", "ss_scores"])
    if task == "repeated":
        return P(f"{head} Score it with `RepeatedStratifiedKFold(n_splits=5, n_repeats={reps}, random_state={s})` and "
                 f"return `(number of scores, mean, standard deviation)`.", full,
                 f"from {MS} import RepeatedStratifiedKFold, cross_val_score\n"
                 f"sc = cross_val_score(model, X, y, cv=RepeatedStratifiedKFold(n_splits=5, n_repeats={reps}, random_state={s}))\n"
                 "result = (len(sc), sc.mean(), sc.std())", require=["RepeatedStratifiedKFold"])
    ss = f"ShuffleSplit(n_splits={ns_}, test_size={ts}, random_state={s})"
    if task == "never":
        return P(f"{head} Unlike K-fold, `{ss}` draws each test set independently, so some rows are never tested. "
                 f"Return how many rows appear in **no** test set.", full,
                 f"from {MS} import ShuffleSplit\ntested = np.zeros(len(X), bool)\nfor tr, te in {ss}.split(X):\n"
                 "    tested[te] = True\nresult = int((~tested).sum())", require=["ShuffleSplit"])
    if task == "max_count":
        return P(f"{head} With `{ss}`, how often is each row tested? Return `np.bincount` of the per-row counts (how "
                 f"many rows are tested 0 times, 1 time, ...).", full,
                 f"from {MS} import ShuffleSplit\ncount = np.zeros(len(X), int)\nfor tr, te in {ss}.split(X):\n"
                 "    count[te] += 1\nresult = np.bincount(count)", require=["ShuffleSplit"])
    if task == "sss":
        sss = f"StratifiedShuffleSplit(n_splits={ns_}, test_size={ts}, random_state={s})"
        return P(f"{head} Return the class counts of every test set of `{sss}` as a {ns_} × (number of classes) array.",
                 full, f"from {MS} import StratifiedShuffleSplit\nk = len(np.unique(y))\n"
                       f"result = np.array([np.bincount(y[te], minlength=k) for tr, te in {sss}.split(X, y)])",
                 require=["StratifiedShuffleSplit"])
    return P(f"{head} Return the scores of `cross_val_score(model, X, y, cv={ss})`.", full,
             f"from {MS} import ShuffleSplit, cross_val_score\nresult = cross_val_score(model, X, y, cv={ss})",
             require=["ShuffleSplit", "cross_val_score"])


@template(M, "medium")
def pick_setting_by_cv(rng):
    setup, desc, n = _multi(rng, n=(150, 300))
    ks = sorted(rng.sample([1, 3, 5, 9, 15, 25], 4))
    depths = sorted(rng.sample([1, 2, 3, 4, 5, 8], 4))
    task = rng.choice(["best_k", "models", "best_depth", "gain"])
    if task == "best_k":
        return P(f"{desc}. For `n_neighbors` in `{ks}`, compute the mean 5-fold CV accuracy of "
                 f"`KNeighborsClassifier(n_neighbors=k)` (`cv=5`) and return the best `k` (the smallest on ties).", setup,
                 f"from {MS} import cross_val_score\nfrom sklearn.neighbors import KNeighborsClassifier\n"
                 f"means = [cross_val_score(KNeighborsClassifier(n_neighbors=k), X, y, cv=5).mean() for k in {ks}]\n"
                 f"result = {ks}[int(np.argmax(means))]", require=["cross_val_score"])
    if task == "models":
        return P(f"{desc}. Return a dict mapping `'tree'`, `'nb'` and `'forest'` to the mean 5-fold CV accuracy of "
                 f"`DecisionTreeClassifier(max_depth=3, random_state=0)`, `GaussianNB()` and "
                 f"`RandomForestClassifier(n_estimators=20, random_state=0)`.", setup,
                 f"from {MS} import cross_val_score\nfrom sklearn.tree import DecisionTreeClassifier\n"
                 "from sklearn.naive_bayes import GaussianNB\nfrom sklearn.ensemble import RandomForestClassifier\n"
                 "models = {'tree': DecisionTreeClassifier(max_depth=3, random_state=0), 'nb': GaussianNB(), "
                 "'forest': RandomForestClassifier(n_estimators=20, random_state=0)}\n"
                 "result = {name: cross_val_score(m, X, y, cv=5).mean() for name, m in models.items()}",
                 require=["cross_val_score", "RandomForestClassifier"])
    if task == "best_depth":
        return P(f"{desc}. For `max_depth` in `{depths}`, compute the mean 5-fold CV accuracy of "
                 f"`DecisionTreeClassifier(max_depth=d, random_state=0)` and return `(best depth, its mean)` (smallest "
                 f"depth on ties).", setup,
                 f"from {MS} import cross_val_score\nfrom sklearn.tree import DecisionTreeClassifier\n"
                 f"means = [cross_val_score(DecisionTreeClassifier(max_depth=d, random_state=0), X, y, cv=5).mean() "
                 f"for d in {depths}]\ni = int(np.argmax(means))\nresult = ({depths}[i], means[i])", require=["cross_val_score"])
    return P(f"{desc}. How much does tuning buy? Return the best mean 5-fold CV accuracy over `n_neighbors` in `{ks}` "
             f"minus the mean CV accuracy of the default `KNeighborsClassifier()`.", setup,
             f"from {MS} import cross_val_score\nfrom sklearn.neighbors import KNeighborsClassifier\n"
             f"best = max(cross_val_score(KNeighborsClassifier(n_neighbors=k), X, y, cv=5).mean() for k in {ks})\n"
             "result = best - cross_val_score(KNeighborsClassifier(), X, y, cv=5).mean()", require=["cross_val_score"])


@template(M, "medium")
def cv_regression(rng):
    t = table(rng, kind="reg", numeric_only=True, n=(120, 250))
    s = rng.randint(0, 99)
    task = rng.choice(["r2", "mae", "gap", "residuals", "sorted"])
    head = f"{t.desc}."
    if task == "r2":
        return P(f"{head} Return the 5 fold R² scores of `LinearRegression()` (`cross_val_score(..., cv=5)`; a regressor's "
                 f"default score is R²).", t.setup,
                 f"from {MS} import cross_val_score\nfrom sklearn.linear_model import LinearRegression\n"
                 "result = cross_val_score(LinearRegression(), df, y, cv=5)", require=["cross_val_score"])
    if task == "mae":
        return P(f"{head} Error metrics are negated so that larger is better. Return the mean 5-fold CV **mean absolute "
                 f"error** of `LinearRegression()` as a positive number (`scoring='neg_mean_absolute_error'`).", t.setup,
                 f"from {MS} import cross_val_score\nfrom sklearn.linear_model import LinearRegression\n"
                 "result = -cross_val_score(LinearRegression(), df, y, cv=5, scoring='neg_mean_absolute_error').mean()",
                 require=["cross_val_score", "scoring"])
    if task == "gap":
        return P(f"{head} Return `(mean train R², mean test R²)` of an unpruned `DecisionTreeRegressor(random_state=0)` "
                 f"(from `sklearn.tree`) with `cross_validate(..., cv=5, return_train_score=True)`.", t.setup,
                 f"from {MS} import cross_validate\nfrom sklearn.tree import DecisionTreeRegressor\n"
                 "res = cross_validate(DecisionTreeRegressor(random_state=0), df, y, cv=5, return_train_score=True)\n"
                 "result = (res['train_score'].mean(), res['test_score'].mean())", require=["cross_validate"])
    if task == "residuals":
        return P(f"{head} Return the root-mean-square of the out-of-fold residuals `y − cross_val_predict("
                 f"LinearRegression(), df, y, cv=5)`.", t.setup,
                 f"from {MS} import cross_val_predict\nfrom sklearn.linear_model import LinearRegression\n"
                 "result = np.sqrt(((y - cross_val_predict(LinearRegression(), df, y, cv=5)) ** 2).mean())",
                 require=["cross_val_predict"])
    return P(f"{head} Someone sorted the rows by target: `order = np.argsort(y)`. Unshuffled `KFold(n_splits=5)` then "
             f"tests each fold on a range of targets the model never saw. Return the mean R² of `LinearRegression()` on "
             f"the sorted rows with `KFold(n_splits=5)` and with `KFold(n_splits=5, shuffle=True, random_state={s})`, "
             f"as a tuple.", t.setup,
             f"from {MS} import KFold, cross_val_score\nfrom sklearn.linear_model import LinearRegression\n"
             "order = np.argsort(y)\nXs, ys = df.iloc[order], y[order]\n"
             "result = (cross_val_score(LinearRegression(), Xs, ys, cv=KFold(n_splits=5)).mean(), "
             f"cross_val_score(LinearRegression(), Xs, ys, cv=KFold(n_splits=5, shuffle=True, random_state={s})).mean())",
             require=["KFold", "shuffle"])


@template(M, "medium")
def cv_diagnostics(rng):
    setup, desc, n = _multi(rng, n=(150, 300))
    s = rng.randint(0, 99)
    depth = rng.randint(2, 5)
    fracs = sorted(rng.sample([0.1, 0.2, 0.3, 0.5, 0.7], 3))
    model = f"DecisionTreeClassifier(max_depth={depth}, random_state=0)"
    skf = f"StratifiedKFold(n_splits=5, shuffle=True, random_state={s})"
    items = [
        ("StratifiedKFold", f"Return the class counts of every test fold of `{skf}` as a 5 × (number of classes) array.",
         f"from {MS} import StratifiedKFold\nk = len(np.unique(y))\n"
         f"result = np.array([np.bincount(y[te], minlength=k) for tr, te in {skf}.split(X, y)])"),
        ("cross_val_score", "More folds, noisier folds: return a dict mapping `cv` in (3, 5, 10) to the standard "
                            "deviation of the fold accuracies of `model`.",
         f"from {MS} import cross_val_score\nresult = {{k: cross_val_score(model, X, y, cv=k).std() for k in (3, 5, 10)}}"),
        ("ShuffleSplit", f"A learning curve with `ShuffleSplit`: for `train_size` in `{fracs}`, score `model` with "
                         f"`cv=ShuffleSplit(n_splits=5, train_size=frac, test_size=0.2, random_state={s})` and return "
                         f"the mean accuracies as a list.",
         f"from {MS} import ShuffleSplit, cross_val_score\nresult = [cross_val_score(model, X, y, cv=ShuffleSplit("
         f"n_splits=5, train_size=f, test_size=0.2, random_state={s})).mean() for f in {fracs}]"),
        ("cross_val_predict", "Build the out-of-fold confusion matrix (`cv=5`) by hand: `C[t, p]` counts rows of true "
                              "class `t` predicted as `p`.",
         f"from {MS} import cross_val_predict\npred = cross_val_predict(model, X, y, cv=5)\nk = len(np.unique(y))\n"
         "result = np.zeros((k, k), int)\nnp.add.at(result, (y, pred), 1)"),
        ("scoring", "Return the mean 5-fold CV **log loss** of `GaussianNB()` as a positive number "
                    "(`scoring='neg_log_loss'`).",
         f"from {MS} import cross_val_score\nfrom sklearn.naive_bayes import GaussianNB\n"
         "result = -cross_val_score(GaussianNB(), X, y, cv=5, scoring='neg_log_loss').mean()",
         {"require": ["cross_val_score", "scoring"]}),
        ("return_estimator", "`cross_validate(..., return_estimator=True)` also returns the fitted models. Run it on an "
                             "unpruned `DecisionTreeClassifier(random_state=0)` with `cv=5` and return each fold tree's "
                             "depth (`get_depth()`).",
         f"from {MS} import cross_validate\nfrom sklearn.tree import DecisionTreeClassifier\n"
         "res = cross_validate(DecisionTreeClassifier(random_state=0), X, y, cv=5, return_estimator=True)\n"
         "result = [est.get_depth() for est in res['estimator']]"),
        ("RepeatedStratifiedKFold", f"Score `model` with `RepeatedStratifiedKFold(n_splits=5, n_repeats=3, "
                                    f"random_state={s})`; the scores come repeat by repeat. Return the mean of each repeat "
                                    f"(reshape to 3 × 5).",
         f"from {MS} import RepeatedStratifiedKFold, cross_val_score\nsc = cross_val_score(model, X, y, "
         f"cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state={s}))\nresult = sc.reshape(3, 5).mean(axis=1)"),
        ("KFold", f"Return `(stratified_std, plain_std)`: the standard deviation of the 5 fold accuracies of `model` "
                  f"with `{skf}` and with `KFold(n_splits=5, shuffle=True, random_state={s})`.",
         f"from {MS} import KFold, StratifiedKFold, cross_val_score\n"
         f"result = (cross_val_score(model, X, y, cv={skf}).std(), cross_val_score(model, X, y, cv=KFold(n_splits=5, "
         f"shuffle=True, random_state={s})).std())"),
        ("GroupKFold", "Pretend rows come in batches of 10 (`groups = np.arange(len(X)) // 10`) and return the fold "
                       "accuracies of `model` with `GroupKFold(n_splits=5)`.",
         f"from {MS} import GroupKFold, cross_val_score\ngroups = np.arange(len(X)) // 10\n"
         "result = cross_val_score(model, X, y, groups=groups, cv=GroupKFold(n_splits=5))"),
        ("cross_val_score", "Which rows are hard? Over `cv=5`, return the out-of-fold accuracy of each class (fraction "
                            "of its rows predicted correctly by `cross_val_predict`), as a list in class order.",
         f"from {MS} import cross_val_predict\npred = cross_val_predict(model, X, y, cv=5)\n"
         "result = [float((pred[y == c] == c).mean()) for c in np.unique(y)]", {"require": ["cross_val_predict"]}),
    ]
    return catalog(rng, items, setup + f"\nfrom sklearn.tree import DecisionTreeClassifier\nmodel = {model}",
                   f"{desc}; `model = {model}`.", tol=1e-9)


# ================================================================= hard

@template(M, "hard")
def selection_leakage(rng):
    n, d, s = rng.randint(60, 100), rng.choice([200, 300, 500]), rng.randint(0, 999)
    top = rng.choice([5, 10, 20])
    setup = (f"rng = np.random.default_rng({s})\nX = rng.normal(size=({n}, {d}))\ny = rng.integers(0, 2, {n})")
    head = (f"`X` ({n} × {d}) is pure noise and `y` random 0/1 labels: no model can beat 50% honestly. Feature "
            f"selection: keep the {top} columns with the largest |correlation| with the labels (`np.corrcoef(X[:, j], "
            f"y)[0, 1]`). Model: `GaussianNB()`, `cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)`.")
    pick = (f"def top_cols(A, b):\n    r = np.array([abs(np.corrcoef(A[:, j], b)[0, 1]) for j in range(A.shape[1])])\n"
            f"    return np.argsort(-r)[:{top}]\n")
    imp = f"from {MS} import StratifiedKFold, cross_val_score\nfrom sklearn.naive_bayes import GaussianNB\n" \
          "cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)\n"
    leaky = "cols = top_cols(X, y)\nleaky = cross_val_score(GaussianNB(), X[:, cols], y, cv=cv).mean()\n"
    honest = ("hs = []\nfor tr, te in cv.split(X, y):\n    c = top_cols(X[tr], y[tr])\n"
              "    hs.append(GaussianNB().fit(X[tr][:, c], y[tr]).score(X[te][:, c], y[te]))\nhonest = float(np.mean(hs))\n")
    task = rng.choice(["leaky", "honest", "both", "overlap"])
    if task == "leaky":
        return P(f"{head} Do it the **leaky** way: select on all rows, then cross-validate on the selected columns. "
                 f"Return the mean CV accuracy (suspiciously good).", setup, imp + pick + leaky + "result = leaky",
                 require=["cross_val_score"])
    if task == "honest":
        return P(f"{head} Do it **honestly**: inside each fold, select columns from the training rows only, then fit and "
                 f"score. Return the mean fold accuracy.", setup, imp + pick + honest + "result = honest",
                 require=["StratifiedKFold"])
    if task == "both":
        return P(f"{head} Return `(leaky, honest)`: the mean CV accuracy when selecting on all rows first, and when "
                 f"selecting inside each fold from its training rows.", setup,
                 imp + pick + leaky + honest + "result = (leaky, honest)", require=["StratifiedKFold", "cross_val_score"])
    return P(f"{head} How many of the {top} columns selected on all rows are also selected from the training rows of "
             f"the **first** fold of `cv`? Return an int (selection on noise is unstable).", setup,
             imp + pick + "tr, te = next(cv.split(X, y))\nresult = len(set(top_cols(X, y).tolist()) & set(top_cols(X[tr], "
                          "y[tr]).tolist()))", require=["StratifiedKFold"])


@template(M, "hard")
def grouped_leakage(rng):
    G, m, s = rng.randint(20, 40), rng.randint(4, 8), rng.randint(0, 999)
    k = rng.choice([4, 5])
    setup = (f"rng = np.random.default_rng({s})\ngroups = np.repeat(np.arange({G}), {m})\n"
             f"profile = rng.normal(size=({G}, 4))\nX = profile[groups] + rng.normal(0, 0.1, (len(groups), 4))\n"
             f"y = rng.integers(0, 2, {G})[groups]")
    head = (f"{G} patients with {m} visits each: `groups` holds the patient id of each row, a patient's rows are "
            f"near-copies of each other, and each patient's label is random (nothing generalizes to **new** patients). "
            f"Model: `KNeighborsClassifier(n_neighbors=1)`.")
    imp = f"from sklearn.neighbors import KNeighborsClassifier\nknn = KNeighborsClassifier(n_neighbors=1)\n"
    kf = f"KFold(n_splits={k}, shuffle=True, random_state={s % 100})"
    task = rng.choice(["kfold", "group", "both", "share"])
    if task == "kfold":
        return P(f"{head} Return the mean CV accuracy with `cv={kf}`: rows of the same patient sit on both sides.",
                 setup, imp + f"from {MS} import KFold, cross_val_score\nresult = cross_val_score(knn, X, y, cv={kf}).mean()",
                 require=["KFold", "cross_val_score"])
    if task == "group":
        return P(f"{head} Return the mean CV accuracy with `cv=GroupKFold(n_splits={k})` (pass `groups=groups`): the "
                 f"honest estimate.", setup,
                 imp + f"from {MS} import GroupKFold, cross_val_score\nresult = cross_val_score(knn, X, y, groups=groups, "
                       f"cv=GroupKFold(n_splits={k})).mean()", require=["GroupKFold", "groups"])
    if task == "both":
        return P(f"{head} Return `(shuffled_kfold, group_kfold)` mean CV accuracies, with `cv={kf}` and with "
                 f"`GroupKFold(n_splits={k})`.", setup,
                 imp + f"from {MS} import GroupKFold, KFold, cross_val_score\n"
                       f"result = (cross_val_score(knn, X, y, cv={kf}).mean(), cross_val_score(knn, X, y, groups=groups, "
                       f"cv=GroupKFold(n_splits={k})).mean())", require=["GroupKFold", "KFold"])
    return P(f"{head} Over all folds of `{kf}`, what fraction of the test rows belong to a patient who also has rows in "
             f"that fold's training set? Return the fraction (over all test rows of all folds).", setup,
             f"from {MS} import KFold\nshared = 0\nfor tr, te in {kf}.split(X):\n"
             "    shared += np.isin(groups[te], groups[tr]).sum()\nresult = shared / len(X)", require=["KFold"])


@template(M, "hard")
def temporal_leakage(rng):
    n, s = rng.randint(150, 300), rng.randint(0, 999)
    lags = rng.randint(2, 4)
    k = rng.choice([3, 4, 5])
    gap = rng.randint(1, 5)
    setup = (f"rng = np.random.default_rng({s})\nseries = np.cumsum(rng.normal(0, 1, {n + lags}))\n"
             f"X = np.column_stack([series[{lags} - j - 1:{n + lags} - j - 1] for j in range({lags})])\ny = series[{lags}:]")
    head = (f"`series` is a random walk; row `t` of `X` holds the previous {lags} values and `y` the next one, in time "
            f"order. Model: `DecisionTreeRegressor(max_depth=5, random_state=0)` (from `sklearn.tree`; it can't "
            f"extrapolate beyond the targets it has seen).")
    imp = "from sklearn.tree import DecisionTreeRegressor\nmodel = DecisionTreeRegressor(max_depth=5, random_state=0)\n"
    kf = f"KFold(n_splits={k}, shuffle=True, random_state=0)"
    task = rng.choice(["shuffled", "forward", "both", "gap"])
    if task == "shuffled":
        return P(f"{head} Return the mean R² with `cv={kf}`: future rows leak into training.", setup,
                 imp + f"from {MS} import KFold, cross_val_score\nresult = cross_val_score(model, X, y, cv={kf}).mean()",
                 require=["KFold"])
    if task == "forward":
        return P(f"{head} Return the R² of each split of `TimeSeriesSplit(n_splits={k})` (train on the past only).", setup,
                 imp + f"from {MS} import TimeSeriesSplit, cross_val_score\nresult = cross_val_score(model, X, y, "
                       f"cv=TimeSeriesSplit(n_splits={k}))", require=["TimeSeriesSplit"])
    if task == "both":
        return P(f"{head} Return `(shuffled, forward)`: the mean R² with `cv={kf}` and with `TimeSeriesSplit(n_splits="
                 f"{k})`.", setup,
                 imp + f"from {MS} import KFold, TimeSeriesSplit, cross_val_score\n"
                       f"result = (cross_val_score(model, X, y, cv={kf}).mean(), cross_val_score(model, X, y, "
                       f"cv=TimeSeriesSplit(n_splits={k})).mean())", require=["TimeSeriesSplit", "KFold"])
    return P(f"{head} The lag features of the first test rows overlap the last training targets. Return the per-split "
             f"mean absolute error with `TimeSeriesSplit(n_splits={k}, gap={gap})` (fit and score in a manual loop; "
             f"MAE = mean |y − prediction|).", setup,
             imp + f"from {MS} import TimeSeriesSplit\nresult = []\nfor tr, te in TimeSeriesSplit(n_splits={k}, gap={gap})"
                   ".split(X):\n    p = model.fit(X[tr], y[tr]).predict(X[te])\n    result.append(np.abs(y[te] - p).mean())",
             require=["TimeSeriesSplit", "gap"])


@template(M, "hard")
def nested_cv_by_hand(rng):
    setup, desc, n = _multi(rng, n=(120, 220))
    ks = sorted(rng.sample([1, 3, 5, 9, 15], 3))
    k_out, s = rng.choice([3, 4]), rng.randint(0, 99)
    head = (f"{desc}. Nested CV estimates the score of the whole *tuning procedure*. Outer loop: `KFold(n_splits={k_out}, "
            f"shuffle=True, random_state={s})`. In each outer fold, pick `n_neighbors` from `{ks}` by the mean of "
            f"`cross_val_score(KNeighborsClassifier(n_neighbors=k), X_tr, y_tr, cv=3)` on the outer training rows only "
            f"(smallest `k` on ties), refit with it on those rows and score on the outer test rows.")
    sol = (f"from {MS} import KFold, cross_val_score\nfrom sklearn.neighbors import KNeighborsClassifier\nchosen, outer = [], []\n"
           f"for tr, te in KFold(n_splits={k_out}, shuffle=True, random_state={s}).split(X):\n"
           f"    means = [cross_val_score(KNeighborsClassifier(n_neighbors=k), X[tr], y[tr], cv=3).mean() for k in {ks}]\n"
           f"    best = {ks}[int(np.argmax(means))]\n    chosen.append(best)\n"
           "    outer.append(KNeighborsClassifier(n_neighbors=best).fit(X[tr], y[tr]).score(X[te], y[te]))\n")
    task = rng.choice(["chosen", "outer", "both"])
    if task == "chosen":
        return P(f"{head} Return the list of chosen `k` values, one per outer fold.", setup, sol + "result = chosen",
                 require=["KFold", "cross_val_score"])
    if task == "outer":
        return P(f"{head} Return the mean outer-fold accuracy.", setup, sol + "result = float(np.mean(outer))",
                 require=["KFold", "cross_val_score"])
    return P(f"{head} Return `(chosen, outer)`: the chosen `k` and the outer accuracy of every outer fold, as two lists.",
             setup, sol + "result = (chosen, outer)", require=["KFold", "cross_val_score"])


@template(M, "hard")
def one_standard_error_rule(rng):
    setup, desc, n = _multi(rng, n=(150, 300))
    s = rng.randint(0, 99)
    depths = list(range(1, rng.randint(6, 8)))
    head = (f"{desc}. For `max_depth` in `{depths}`, score `DecisionTreeClassifier(max_depth=d, random_state=0)` with "
            f"`cv=StratifiedKFold(n_splits=5, shuffle=True, random_state={s})`; keep each depth's mean and standard "
            f"error (`std / sqrt(5)` of the 5 fold scores).")
    sol = (f"from {MS} import StratifiedKFold, cross_val_score\nfrom sklearn.tree import DecisionTreeClassifier\n"
           f"cv = StratifiedKFold(n_splits=5, shuffle=True, random_state={s})\n"
           f"sc = [cross_val_score(DecisionTreeClassifier(max_depth=d, random_state=0), X, y, cv=cv) for d in {depths}]\n"
           "mean = np.array([v.mean() for v in sc])\nse = np.array([v.std() / np.sqrt(5) for v in sc])\n"
           f"b = int(np.argmax(mean))\nsimple = {depths}[int(np.flatnonzero(mean >= mean[b] - se[b])[0])]\n")
    task = rng.choice(["rule", "pair", "table"])
    if task == "rule":
        return P(f"{head} The **one-standard-error rule** picks the simplest (smallest) depth whose mean is within one "
                 f"standard error of the best mean (`mean ≥ best_mean − best_se`). Return that depth.", setup,
                 sol + "result = simple", require=["cross_val_score"])
    if task == "pair":
        return P(f"{head} Return `(best_depth, one_se_depth)`: the depth with the highest mean (smallest on ties) and "
                 f"the smallest depth with `mean ≥ best_mean − best_se`.", setup,
                 sol + f"result = ({depths}[b], simple)", require=["cross_val_score"])
    return P(f"{head} Return a dict depth → `(mean, standard error)`.", setup,
             sol + f"result = {{d: (m, e) for d, m, e in zip({depths}, mean, se)}}", require=["cross_val_score"])


@template(M, "hard")
def broken_cv_code(rng):
    s = rng.randint(0, 99)
    task = rng.choice(["stratify", "sorted", "groups", "swapped", "scaler"])
    if task == "stratify":
        setup, desc, n = _multi(rng)
        return P(f"{desc}. This line crashes:\n\n```\nX_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, "
                 f"random_state={s}, stratify=True)\n```\n\nFix it (stratify on the labels) and return the class counts "
                 f"of `y_te`.", setup,
                 f"from {MS} import train_test_split\nX_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, "
                 f"random_state={s}, stratify=y)\nresult = np.bincount(y_te)", require=["train_test_split", "stratify"])
    if task == "sorted":
        key = rng.choice(["load_iris", "load_wine"])
        return P(f"`X, y = {key}(return_X_y=True)` (rows sorted by class). This scores near 0, because each test fold "
                 f"holds a class the training folds lack:\n\n```\ncross_val_score(GaussianNB(), X, y, cv=KFold(n_splits=3))\n"
                 f"```\n\nFix it by shuffling inside the splitter (`shuffle=True, random_state={s}`) and return the fold "
                 f"scores.", f"from sklearn.datasets import {key}\nX, y = {key}(return_X_y=True)",
                 f"from {MS} import KFold, cross_val_score\nfrom sklearn.naive_bayes import GaussianNB\n"
                 f"result = cross_val_score(GaussianNB(), X, y, cv=KFold(n_splits=3, shuffle=True, random_state={s}))",
                 require=["KFold", "shuffle"])
    if task == "groups":
        G = rng.randint(10, 16)
        setup = (f"rng = np.random.default_rng({s})\ngroups = rng.integers(0, {G}, 120)\nX = rng.normal(size=(120, 3))\n"
                 f"y = (X[:, 0] + rng.normal(0, 1, 120) > 0).astype(int)")
        return P(f"`X, y` hold 120 rows from {G} users (`groups`). This crashes:\n\n```\ncross_val_score(GaussianNB(), X, y, "
                 f"cv=GroupKFold(n_splits=4))\n```\n\nFix it and return the fold scores.", setup,
                 f"from {MS} import GroupKFold, cross_val_score\nfrom sklearn.naive_bayes import GaussianNB\n"
                 "result = cross_val_score(GaussianNB(), X, y, groups=groups, cv=GroupKFold(n_splits=4))",
                 require=["GroupKFold", "groups"])
    if task == "swapped":
        n = rng.randint(60, 120)
        setup = (f"rng = np.random.default_rng({s})\nX = np.arange({n}, dtype=float).reshape(-1, 1)\n"
                 f"y = 0.5 * X[:, 0] + rng.normal(0, 2, {n})")
        return P(f"`X` is time (0, 1, 2, ...) and `y` a noisy trend. This loop is wrong: it trains on the future and "
                 f"tests on the past.\n\n```\nfor tr, te in TimeSeriesSplit(n_splits=3).split(X):\n"
                 f"    model = LinearRegression().fit(X[te], y[te])\n    scores.append(model.score(X[tr], y[tr]))\n```\n\n"
                 f"Fix it and return the list of R² scores.", setup,
                 f"from {MS} import TimeSeriesSplit\nfrom sklearn.linear_model import LinearRegression\nscores = []\n"
                 "for tr, te in TimeSeriesSplit(n_splits=3).split(X):\n    model = LinearRegression().fit(X[tr], y[tr])\n"
                 "    scores.append(model.score(X[te], y[te]))\nresult = scores", require=["TimeSeriesSplit"])
    setup, desc, n = _binary(rng, n=(120, 200))
    k = rng.choice([3, 5])
    return P(f"{desc}. This leaks: the scaler sees the validation rows before CV.\n\n```\nXs = StandardScaler()"
             f".fit_transform(X)\ncross_val_score(KNeighborsClassifier(n_neighbors={k}), Xs, y, cv=5)\n```\n\nFix it with "
             f"a manual loop over `StratifiedKFold(n_splits=5)` that fits the scaler on each fold's training rows only, "
             f"and return the fold accuracies as a list.", setup,
             f"from {MS} import StratifiedKFold\nfrom sklearn.neighbors import KNeighborsClassifier\n"
             "from sklearn.preprocessing import StandardScaler\nscores = []\nfor tr, te in StratifiedKFold(n_splits=5).split(X, y):\n"
             f"    sc = StandardScaler().fit(X[tr])\n    m = KNeighborsClassifier(n_neighbors={k}).fit(sc.transform(X[tr]), y[tr])\n"
             "    scores.append(m.score(sc.transform(X[te]), y[te]))\nresult = scores", require=["StratifiedKFold"])


@template(M, "hard")
def splitters_reproduced(rng):
    n, k, s = rng.randint(10, 30), rng.randint(3, 5), rng.randint(0, 99)
    ns_ = rng.randint(2, 4)
    ts = rng.choice([0.2, 0.25, 0.3])
    task = rng.choice(["kfold", "tss", "shuffle_split", "loo"])
    if task == "kfold":
        return P(f"Rebuild `KFold(n_splits={k}, shuffle=True, random_state={s})` on {n} rows without it: permute the "
                 f"indices with `np.random.RandomState({s}).permutation({n})`, cut them into {k} consecutive blocks "
                 f"(the first `n % k` blocks one longer), and return the list of `(sorted train indices, sorted test "
                 f"indices)` pairs. Check against `KFold`.", f"n, k = {n}, {k}",
                 f"perm = np.random.RandomState({s}).permutation(n)\nsizes = [n // k + (f < n % k) for f in range(k)]\n"
                 "ends = np.cumsum(sizes)\nresult = []\nfor e, size in zip(ends, sizes):\n    te = np.sort(perm[e - size:e])\n"
                 "    result.append((np.setdiff1d(np.arange(n), te), te))", require=["permutation"])
    if task == "tss":
        return P(f"Rebuild `TimeSeriesSplit(n_splits={k})` on {n} rows by hand: the test size is `n // (n_splits + 1)`, "
                 f"the test blocks are the last `n_splits` blocks of that size, and each training set is everything "
                 f"before its test block. Return the list of `(train_end, test_start, test_end)` (exclusive ends).",
                 f"n, k = {n}, {k}",
                 "t = n // (k + 1)\nresult = [(n - (k - i) * t, n - (k - i) * t, n - (k - i - 1) * t) for i in range(k)]",
                 require=[])
    if task == "shuffle_split":
        return P(f"`ShuffleSplit(n_splits={ns_}, test_size={ts}, random_state={s})` on {n} rows works like this: "
                 f"`n_test = ceil({ts} · n)`; `rs = np.random.RandomState({s})`; for each split, `perm = "
                 f"rs.permutation(n)`, the test indices are `perm[:n_test]` and the training indices `perm[n_test:]`. "
                 f"Rebuild the test index arrays (in that order, not sorted) and return them as a list.", f"n = {n}",
                 f"import math\nn_test = math.ceil({ts} * n)\nrs = np.random.RandomState({s})\n"
                 f"result = [rs.permutation(n)[:n_test] for _ in range({ns_})]", require=["permutation"])
    return P(f"`LeaveOneOut()` is `KFold(n_splits=n)`. On {n} rows, return whether both yield exactly the same "
             f"`(train, test)` index arrays (a bool), and the number of splits, as a tuple.", f"X = np.zeros(({n}, 2))",
             f"from {MS} import KFold, LeaveOneOut\na = list(LeaveOneOut().split(X))\nb = list(KFold(n_splits=len(X)).split(X))\n"
             "same = all(np.array_equal(p[0], q[0]) and np.array_equal(p[1], q[1]) for p, q in zip(a, b))\n"
             "result = (bool(same and len(a) == len(b)), len(a))", require=["LeaveOneOut", "KFold"])


@template(M, "hard")
def stacking_out_of_fold(rng):
    setup, desc, n = _binary(rng, n=(200, 400))
    h = int(n * 0.7) if n != 569 else 400
    k = rng.choice([3, 5, 9])
    setup += f"\nX_tr, X_te, y_tr, y_te = X[:{h}], X[{h}:], y[:{h}], y[{h}:]"
    head = (f"{desc}; `X_tr, y_tr` are the first {h} rows and `X_te, y_te` the rest. Stack two base models, "
            f"`GaussianNB()` and `KNeighborsClassifier(n_neighbors={k})`: a meta-model `LogisticRegression()` learns "
            f"from their P(class 1).")
    base = (f"from {MS} import cross_val_predict\nfrom sklearn.naive_bayes import GaussianNB\n"
            "from sklearn.neighbors import KNeighborsClassifier\nfrom sklearn.linear_model import LogisticRegression\n"
            f"bases = [GaussianNB(), KNeighborsClassifier(n_neighbors={k})]\n"
            "F_tr = np.column_stack([cross_val_predict(b, X_tr, y_tr, cv=5, method='predict_proba')[:, 1] for b in bases])\n"
            "F_te = np.column_stack([b.fit(X_tr, y_tr).predict_proba(X_te)[:, 1] for b in bases])\n")
    task = rng.choice(["features", "accuracy", "leaky"])
    if task == "features":
        return P(f"{head} Build the meta-features of the training rows **out of fold** (`cross_val_predict(..., cv=5, "
                 f"method='predict_proba')[:, 1]` for each base) and return the {h} × 2 matrix.", setup,
                 base + "result = F_tr", require=["cross_val_predict"])
    if task == "accuracy":
        return P(f"{head} Fit the meta-model on the out-of-fold training features (`cross_val_predict`, `cv=5`, "
                 f"`method='predict_proba'`), build the test features from the base models refitted on all of `X_tr`, "
                 f"and return the stacked model's test accuracy.", setup,
                 base + "result = LogisticRegression().fit(F_tr, y_tr).score(F_te, y_te)", require=["cross_val_predict"])
    return P(f"{head} Compare two ways to build the meta-model's training features: out of fold (`cross_val_predict`, "
             f"`cv=5`) and in-sample (bases fitted on `X_tr` predicting `X_tr`, which overrates the k-NN). Test features "
             f"come from the bases refitted on `X_tr`. Return the meta-model's coefficients for both, as `(oof_coef, "
             f"insample_coef)` (each `coef_[0]`).", setup,
             base + "F_in = np.column_stack([b.fit(X_tr, y_tr).predict_proba(X_tr)[:, 1] for b in bases])\n"
                    "result = (LogisticRegression().fit(F_tr, y_tr).coef_[0], LogisticRegression().fit(F_in, y_tr).coef_[0])",
             require=["cross_val_predict"], tol=1e-4)


# ================================================================= challenge

@template(M, "challenge")
def kfold_mean_model(rng):
    n, s = rng.randint(9, 14), seeded(rng)
    K = rng.randint(4, 6)
    report = rng.choice(["mse", "best", "worst_fold"])
    setup = f"g = np.random.default_rng({s})\ny = g.normal(10, 3, {n}).round(2)\nks = np.arange(2, {K + 1})\nreport = {report!r}"
    sol = ("n = len(y)\nc1 = np.concatenate([[0.0], np.cumsum(y)])\nc2 = np.concatenate([[0.0], np.cumsum(y * y)])\n"
           "mse, worst = [], []\nfor k in ks.tolist():\n    sizes = np.full(k, n // k)\n    sizes[: n % k] += 1\n"
           "    b = np.concatenate([[0], np.cumsum(sizes)])\n    a, e = b[:-1], b[1:]\n"
           "    S, Q, L = c1[e] - c1[a], c2[e] - c2[a], e - a\n    m = (c1[n] - S) / (n - L)\n"
           "    f = (Q - 2 * m * S + L * m * m) / L\n    mse.append(f.mean())\n    worst.append(f.max())\n"
           "mse = np.array(mse)\nresult = {'mse': mse, 'best': int(ks[np.argmin(mse)]), 'worst_fold': np.array(worst)}[report]")
    tests = [f"g = np.random.default_rng({s + i})\ny = g.normal(10, 3, 200000).round(2)\nks = np.arange(2, {K2})\n"
             f"report = {report!r}" for i, K2 in ((1, 2001), (2, 600))]
    what = {"mse": "the CV score for every `k` in `ks`, as a float array",
            "best": "the `k` with the lowest CV score (the smallest such `k` on ties)",
            "worst_fold": "for every `k` in `ks`, the largest single-fold MSE, as a float array"}[report]
    return challenge(
        "Cross-validating a baseline for many k",
        "The simplest regression baseline (`DummyRegressor()`) predicts the mean of its training targets. For each "
        "fold count `k` in `ks`, run `KFold(n_splits=k)` (no shuffle: consecutive blocks, the first `n % k` one row "
        "longer) over the targets `y`; in each fold the baseline predicts the mean of the other folds' targets, and the "
        "fold's error is the mean squared error on its own block. The CV score of `k` is the average of its fold errors. "
        f"Return {what}. (`report`: {report!r}.)",
        setup, sol, tests, "len(y) ≤ 2·10^5, sum of ks ≤ 2·10^6", speed="O(n + Σk): prefix sums, no loop over rows per fold",
        answer_desc=" (an int)" if report == "best" else "", tol=1e-6)


@template(M, "challenge")
def stratified_test_counts(rng):
    C, s = rng.randint(4, 7), seeded(rng)
    ts = rng.choice([0.1, 0.2, 0.25, 0.3])
    report = rng.choice(["counts", "zero", "deviation"])
    setup = f"g = np.random.default_rng({s})\ncounts = g.integers(1, 30, {C})\ntest_size = {ts}\nreport = {report!r}"
    sol = ("import math\nN = int(counts.sum())\nt = math.ceil(test_size * N)\nc = counts.astype(np.int64)\n"
           "base = c * t // N\nrem = c * t % N\nleft = t - int(base.sum())\n"
           "order = np.lexsort((np.arange(len(c)), -rem))\nalloc = base.copy()\nalloc[order[:left]] += 1\n"
           "if report == 'counts':\n    result = alloc\nelif report == 'zero':\n    result = int((alloc == 0).sum())\n"
           "else:\n    result = float(np.abs(alloc - c * t / N).max())")
    tests = [f"g = np.random.default_rng({s + i})\ncounts = g.integers(1, {hi}, 200000)\ntest_size = {ts}\nreport = {report!r}"
             for i, hi in ((1, 40), (2, 4))]
    what = {"counts": "the test count of every class, as an int array",
            "zero": "how many classes get no test row at all",
            "deviation": "the largest |test count − exact share| over the classes, as a float"}[report]
    return challenge(
        "Stratified test-set sizes",
        "`train_test_split(..., stratify=y)` must choose how many rows of each class go to the test set. With class "
        "sizes `counts` (total `N`), the test set gets `t = ceil(test_size · N)` rows and class `c` deserves the exact "
        "share `counts[c] · t / N`. Every class first gets the floor of its share; the `t − Σ floors` rows left over go, "
        "one each, to the classes with the largest fractional parts (ties: the smaller class index first; scikit-learn "
        f"breaks ties at random). Compare fractional parts exactly (integers: `counts[c] · t mod N`). Return {what}. "
        f"(`report`: {report!r}.)",
        setup, sol, tests, "number of classes ≤ 2·10^5, counts ≤ 40", speed="O(C log C): sort the remainders once",
        tol=1e-9)


@template(M, "challenge")
def group_folds_greedy(rng):
    G, k, s = rng.randint(7, 12), rng.randint(2, 4), seeded(rng)
    report = rng.choice(["sizes", "fold_of", "spread"])
    setup = f"g = np.random.default_rng({s})\nsizes = g.integers(1, 9, {G})\nk = {k}\nreport = {report!r}"
    sol = ("import heapq\norder = np.lexsort((np.arange(len(sizes)), -sizes))\nheap = [(0, f) for f in range(k)]\n"
           "fold_of = np.empty(len(sizes), np.int64)\nsz = sizes.tolist()\nfor gi in order.tolist():\n"
           "    load, f = heapq.heappop(heap)\n    fold_of[gi] = f\n    heapq.heappush(heap, (load + sz[gi], f))\n"
           "rows = np.bincount(fold_of, weights=sizes, minlength=k).astype(np.int64)\n"
           "result = {'sizes': rows, 'fold_of': fold_of, 'spread': int(rows.max() - rows.min())}[report]")
    tests = [f"g = np.random.default_rng({s + i})\nsizes = g.integers(1, {hi}, {GG})\nk = {kk}\nreport = {report!r}"
             for i, (GG, hi, kk) in ((1, (200000, 50, 50000)), (2, (150000, 1000, 7)))]
    what = {"sizes": "the number of rows each fold ends up with, as an int array of length `k`",
            "fold_of": "the fold of every group, as an int array of length `len(sizes)`",
            "spread": "the difference between the largest and the smallest fold, in rows"}[report]
    return challenge(
        "Balancing groups across folds",
        "`GroupKFold` keeps each group (user, patient...) inside one fold and balances the folds greedily: groups are "
        "taken from the largest to the smallest (`sizes[g]` rows; ties: smaller group id first), and each goes to the "
        f"fold that currently has the fewest rows (ties: smaller fold index). With `k` folds, return {what}. "
        f"(`report`: {report!r}.)",
        setup, sol, tests, "number of groups ≤ 2·10^5, k ≤ 5·10^4", speed="O(G log G + G log k): a heap of fold loads")


@template(M, "challenge")
def time_series_split_sizes(rng):
    q, s = rng.randint(4, 6), seeded(rng)
    report = rng.choice(["train_total", "capped", "mean_train"])
    setup = (f"g = np.random.default_rng({s})\nn_splits = g.integers(2, 6, {q})\nn = n_splits * g.integers(3, 8, {q}) + "
             f"g.integers(0, 12, {q})\ngap = g.integers(0, 2, {q})\nmax_train = np.where(g.random({q}) < 0.5, 0, "
             f"g.integers(2, 20, {q}))\nreport = {report!r}")
    sol = ("n_, k, gp, M = (a.astype(np.int64) for a in (n, n_splits, gap, max_train))\nt = n_ // (k + 1)\n"
           "e0 = n_ - k * t - gp\n"   # training end of the first split; split i ends at e0 + i*t
           "cap = np.where(M > 0, M, np.iinfo(np.int64).max)\n"
           "j = np.where(cap >= e0, np.minimum(k, (cap - e0) // t + 1), 0)\n"   # splits whose training set is not capped
           "free = j * e0 + t * j * (j - 1) // 2\ntotal = free + (k - j) * np.where(M > 0, M, 0)\n"
           "result = {'train_total': total, 'capped': k - j, 'mean_train': total / k}[report]")
    tests = [f"g = np.random.default_rng({s + i})\nq = 200000\nn_splits = g.integers(2, 100000, q)\n"
             f"n = n_splits * g.integers(3, 10000, q) + g.integers(0, 50000, q)\ngap = g.integers(0, 2, q)\n"
             f"max_train = np.where(g.random(q) < 0.3, 0, g.integers(1, 10 ** 9, q))\nreport = {report!r}" for i in (1, 2)]
    what = {"train_total": "the total number of training rows over all splits (an int array)",
            "capped": "how many splits have their training set cut by `max_train_size` (an int array)",
            "mean_train": "the average training-set size over the splits (a float array)"}[report]
    return challenge(
        "Sizing time-series splits",
        "`TimeSeriesSplit(n_splits, gap=gap, max_train_size=...)` on `n` rows uses test blocks of `t = n // (n_splits + "
        "1)` rows: split `i` (0-based) tests on rows `[n − (n_splits − i)·t, n − (n_splits − i − 1)·t)` and trains on the "
        "rows before the test block minus the last `gap` rows; with a `max_train_size`, only the most recent that many "
        "of those are kept. Query `i` gives `n[i]`, `n_splits[i]`, `gap[i]` and `max_train[i]` (0 means no limit); "
        f"every training set is non-empty. For each query return {what}. (`report`: {report!r}.)",
        setup, sol, tests, "number of queries ≤ 2·10^5, n_splits ≤ 10^5, n ≤ 10^9",
        speed="O(1) per query: arithmetic series, no loop over splits", tol=1e-9)


@template(M, "challenge")
def loo_nearest_neighbour(rng):
    n, s = rng.randint(8, 12), seeded(rng)
    report = rng.choice(["accuracy", "errors", "mse"])
    setup = (f"g = np.random.default_rng({s})\nx = g.uniform(0, 100, {n})\nlabel = (np.sin(x / 7) + g.normal(0, 0.6, {n}) "
             f"> 0).astype(int)\ntarget = np.sin(x / 7) * 10 + g.normal(0, 1, {n})\nreport = {report!r}")
    sol = ("o = np.argsort(x)\nxs = x[o]\nleft = np.concatenate([[np.inf], np.diff(xs)])\n"
           "right = np.concatenate([np.diff(xs), [np.inf]])\npos = np.arange(len(x))\nnb = np.where(left <= right, pos - 1, pos + 1)\n"
           "nbr = np.empty(len(x), np.int64)\nnbr[o] = o[nb]\n"
           "if report == 'accuracy':\n    result = float((label[nbr] == label).mean())\n"
           "elif report == 'errors':\n    result = np.flatnonzero(label[nbr] != label)\n"
           "else:\n    result = float(((target[nbr] - target) ** 2).mean())")
    tests = [f"g = np.random.default_rng({s + i})\nx = g.uniform(0, 100, 200000)\nlabel = (np.sin(x / 7) + "
             f"g.normal(0, 0.6, 200000) > 0).astype(int)\ntarget = np.sin(x / 7) * 10 + g.normal(0, 1, 200000)\n"
             f"report = {report!r}" for i in (1, 2)]
    what = {"accuracy": "the leave-one-out **accuracy** of a 1-nearest-neighbour classifier predicting `label` (a float)",
            "errors": "the indices (ascending) of the rows whose `label` the left-out 1-NN classifier gets wrong",
            "mse": "the leave-one-out **mean squared error** of a 1-nearest-neighbour regressor predicting `target` "
                   "(a float)"}[report]
    return challenge(
        "Leave-one-out for 1-NN in one dimension",
        "`cross_val_score(KNeighborsClassifier(n_neighbors=1), x.reshape(-1, 1), label, cv=LeaveOneOut())` refits once "
        "per row, but for one nearest neighbour the left-out model's prediction for row `i` is simply the value of the "
        "row closest to `x[i]` among the others (the `x` values are distinct floats; a tie in distance goes to the "
        f"smaller `x`). Return {what}. (`report`: {report!r}.)",
        setup, sol, tests, "len(x) ≤ 2·10^5", speed="O(n log n): sort once", tol=1e-9)


@template(M, "challenge")
def loo_linear_shortcut(rng):
    n, d, s = rng.randint(8, 12), rng.randint(1, 3), seeded(rng)
    report = rng.choice(["press", "residuals", "worst"])
    setup = (f"g = np.random.default_rng({s})\nX = g.normal(size=({n}, {d}))\ny = X @ g.normal(size={d}) + 3 + "
             f"g.normal(0, 1, {n})\nreport = {report!r}")
    sol = ("A = np.column_stack([np.ones(len(X)), X])\nG = np.linalg.inv(A.T @ A)\nh = np.einsum('ij,jk,ik->i', A, G, A)\n"
           "beta = G @ (A.T @ y)\nloo = (y - A @ beta) / (1 - h)\n"
           "result = {'press': float((loo ** 2).sum()), 'residuals': loo, 'worst': int(np.abs(loo).argmax())}[report]")
    tests = [f"g = np.random.default_rng({s + i})\nX = g.normal(size=(60000, {d}))\ny = X @ g.normal(size={d}) + 3 + "
             f"g.normal(0, 1, 60000)\nreport = {report!r}" for i in (1, 2)]
    what = {"press": "the PRESS statistic: the sum of squared leave-one-out residuals (a float)",
            "residuals": "the leave-one-out residuals `y[i] − (prediction of the model fitted without row i)`, as a "
                         "float array",
            "worst": "the index of the row with the largest absolute leave-one-out residual"}[report]
    return challenge(
        "Leave-one-out without refitting",
        "For ordinary least squares (`LinearRegression()`, with an intercept), leave-one-out CV needs no refits: with "
        "the design matrix `A = [1, X]` and the hat values `h_i` (the diagonal of `A (AᵀA)⁻¹ Aᵀ`), the residual of the "
        "model fitted without row `i` is `e_i / (1 − h_i)`, where `e_i` is row `i`'s ordinary residual. Return "
        f"{what}, matching `cross_val_predict(LinearRegression(), X, y, cv=LeaveOneOut())`. (`report`: {report!r}.)",
        setup, sol, tests, "rows ≤ 6·10^4, features ≤ 3", speed="O(n · d²): no refits", tol=1e-6)
