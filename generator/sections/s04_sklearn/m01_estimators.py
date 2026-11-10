"""4.01 The estimator API: construct, fit, predict / transform, learned attributes, params, clone."""

from ._common import (P, catalog, challenge, fmt, lit, mat, seeded, synth_clf, table, template)

M = "4.01"
LR = "LogisticRegression(max_iter=1000)"


def _data(rng, classes=(2, 3)):
    """iris or a small make_classification problem (both converge fast without scaling)."""
    three = classes == 3 or (isinstance(classes, tuple) and classes[1] >= 3)
    if three and rng.random() < 0.35:
        return ("from sklearn.datasets import load_iris\nX, y = load_iris(return_X_y=True)\n"
                f"order = np.random.default_rng({seeded(rng)}).permutation(len(X))\nX, y = X[order], y[order]",
                "`X` holds 150 iris flowers (4 measurements; rows shuffled) and `y` their species (0, 1, 2)", 3, 4)
    d = synth_clf(rng, n=(80, 240), d=(3, 6), classes=classes)
    return d.setup, d.desc, d.n_classes, d.d


def _matrix(rng, r=(4, 6), c=(2, 3), lo=-5, hi=20):
    n, d = rng.randint(*r), rng.randint(*c)
    rows = [[round(rng.uniform(lo, hi), 1) for _ in range(d)] for _ in range(n)]
    return rows


# ================================================================= lite

@template(M, "lite")
def fitted_classifier(rng):
    setup, desc, k, d = _data(rng)
    n = rng.randint(2, 5)
    items = [
        ("predict", f"Fit `clf = {LR}` on `X, y` and return its predicted classes for the first {n} rows of `X`.",
         f"from sklearn.linear_model import LogisticRegression\nclf = {LR}.fit(X, y)\nresult = clf.predict(X[:{n}])",
         {"require": ["LogisticRegression", "predict"]}),
        ("predict_proba", f"Fit `{LR}` on `X, y` and return the class probabilities (`predict_proba`) of the first {n} "
                          f"rows: one row per sample, one column per class, each row summing to 1.",
         f"from sklearn.linear_model import LogisticRegression\nclf = {LR}.fit(X, y)\nresult = clf.predict_proba(X[:{n}])"),
        ("score", f"Fit `{LR}` on `X, y` and return its **training accuracy** with the classifier's own `score` method.",
         f"from sklearn.linear_model import LogisticRegression\nclf = {LR}.fit(X, y)\nresult = clf.score(X, y)"),
        ("classes_", f"Fit `{LR}` on `X, y` and return the class labels it learned (`classes_`).",
         f"from sklearn.linear_model import LogisticRegression\nresult = {LR}.fit(X, y).classes_"),
        ("n_features_in_", f"Fit `{LR}` on `X, y` and return how many features it saw during `fit` "
                           f"(`n_features_in_`).",
         f"from sklearn.linear_model import LogisticRegression\nresult = {LR}.fit(X, y).n_features_in_"),
        ("coef_", f"Fit `{LR}` on `X, y` and return the **shape** of its coefficient matrix `coef_` as a tuple.",
         f"from sklearn.linear_model import LogisticRegression\nresult = {LR}.fit(X, y).coef_.shape"),
        ("intercept_", f"Fit `{LR}` on `X, y` and return its intercepts (`intercept_`).",
         f"from sklearn.linear_model import LogisticRegression\nresult = {LR}.fit(X, y).intercept_", {"tol": 1e-5}),
    ]
    return catalog(rng, items, setup, f"{desc}.", tol=1e-5)


@template(M, "lite")
def scaler_basics(rng):
    rows = _matrix(rng)
    new = [round(rng.uniform(-5, 20), 1) for _ in rows[0]]
    setup = f"X = {mat(rows)}"
    items = [
        ("fit_transform", "Standardize the columns of `X` with `StandardScaler` (each column gets mean 0 and standard "
                          "deviation 1) and return the result.",
         "from sklearn.preprocessing import StandardScaler\nresult = StandardScaler().fit_transform(X)",
         {"require": ["StandardScaler"]}),
        ("mean_", "Fit a `StandardScaler` on `X` and return the column means it learned (`mean_`).",
         "from sklearn.preprocessing import StandardScaler\nresult = StandardScaler().fit(X).mean_"),
        ("scale_", "Fit a `StandardScaler` on `X` and return the per-column scale it divides by (`scale_`).",
         "from sklearn.preprocessing import StandardScaler\nresult = StandardScaler().fit(X).scale_"),
        ("transform", f"Fit a `StandardScaler` on `X`, then use it to transform the new row `{lit(new)}` (pass it as a "
                      f"2-D array of shape (1, {len(new)})).",
         f"from sklearn.preprocessing import StandardScaler\nsc = StandardScaler().fit(X)\n"
         f"result = sc.transform(np.array([{lit(new)}]))"),
        ("inverse_transform", "Standardize `X` with a `StandardScaler`, then undo it with `inverse_transform` and return "
                              "the recovered matrix (it should equal `X`).",
         "from sklearn.preprocessing import StandardScaler\nsc = StandardScaler()\nZ = sc.fit_transform(X)\n"
         "result = sc.inverse_transform(Z)"),
        ("var_", "Fit a `StandardScaler` on `X` and return the column variances it learned (`var_`).",
         "from sklearn.preprocessing import StandardScaler\nresult = StandardScaler().fit(X).var_"),
    ]
    return catalog(rng, items, setup, f"`X` is a {len(rows)} × {len(rows[0])} matrix.")


@template(M, "lite")
def params(rng):
    c = rng.choice([0.01, 0.1, 0.5, 2.0, 10.0])
    it = rng.choice([200, 500, 2000])
    setup = f"from sklearn.linear_model import LogisticRegression\nclf = LogisticRegression(C={c}, max_iter={it})"
    new_c = rng.choice([v for v in (0.05, 0.3, 1.5, 5.0) if v != c])
    items = [
        ("get_params", "Return `clf`'s hyperparameters as a dict (`get_params`).", "result = clf.get_params()"),
        ("get_params", "Return the value of `clf`'s `C` hyperparameter, read through `get_params()`.",
         "result = clf.get_params()['C']"),
        ("set_params", f"Change `clf`'s `C` to {new_c} with `set_params`, then return `clf.get_params()['C']`.",
         f"clf.set_params(C={new_c})\nresult = clf.get_params()['C']"),
        ("set_params", f"Set `C={new_c}` and `max_iter=300` on `clf` in **one** `set_params` call, then return "
                       f"`(clf.C, clf.max_iter)`.", f"clf.set_params(C={new_c}, max_iter=300)\nresult = (clf.C, clf.max_iter)"),
        ("is_classifier", "Return whether `clf` is a classifier, using `sklearn.base.is_classifier`.",
         "from sklearn.base import is_classifier\nresult = is_classifier(clf)"),
        ("is_classifier", "Return `[is_classifier(clf), is_classifier(LinearRegression())]` "
                          "(`sklearn.base.is_classifier`; a regressor is not a classifier).",
         "from sklearn.base import is_classifier\nfrom sklearn.linear_model import LinearRegression\n"
         "result = [is_classifier(clf), is_classifier(LinearRegression())]"),
    ]
    return catalog(rng, items, setup, f"`clf = LogisticRegression(C={c}, max_iter={it})` is constructed but not "
                                      f"fitted.")


@template(M, "lite")
def linear_regression(rng):
    a, b = round(rng.uniform(-3, 3), 1), round(rng.uniform(-5, 5), 1)
    xs = sorted(rng.sample(range(0, 20), rng.randint(5, 7)))
    ys = [round(a * x + b + rng.uniform(-1, 1), 2) for x in xs]
    q = rng.randint(20, 30)
    setup = f"X = np.array({lit(xs)}).reshape(-1, 1)\ny = np.array({lit(ys)})"
    items = [
        ("predict", f"Fit `LinearRegression` on `X, y` and predict `y` for x = {q} (a 2-D input: `[[{q}]]`). Return the "
                    f"prediction array.",
         f"from sklearn.linear_model import LinearRegression\nresult = LinearRegression().fit(X, y).predict([[{q}]])",
         {"require": ["LinearRegression", "predict"]}),
        ("coef_", "Fit `LinearRegression` on `X, y` and return `(slope, intercept)`: `coef_[0]` and `intercept_`.",
         "from sklearn.linear_model import LinearRegression\nreg = LinearRegression().fit(X, y)\n"
         "result = (reg.coef_[0], reg.intercept_)"),
        ("score", "Fit `LinearRegression` on `X, y` and return its training R² (a regressor's `score` is R²).",
         "from sklearn.linear_model import LinearRegression\nresult = LinearRegression().fit(X, y).score(X, y)"),
    ]
    return catalog(rng, items, setup, "`X` (one feature, as a column) and `y` are noisy points on a line.", tol=1e-6)


@template(M, "lite")
def one_liners(rng):
    setup, desc, k, d = _data(rng, classes=2)
    rows = [[round(rng.uniform(-2, 2), 2) for _ in range(d)] for _ in range(rng.randint(2, 3))] if "iris" not in setup \
        else [[round(rng.uniform(4.5, 7.5), 1), round(rng.uniform(2.2, 4), 1), round(rng.uniform(1, 6.5), 1),
               round(rng.uniform(0.1, 2.4), 1)] for _ in range(rng.randint(2, 3))]
    items = [
        ("fit", f"`fit` returns the estimator itself, so calls chain. In **one** expression, fit `{LR}` on `X, y` and "
                f"predict the new rows `X_new`.",
         f"from sklearn.linear_model import LogisticRegression\n{LR}.fit(X, y).predict(X_new)",
         {"require": ["LogisticRegression", "fit", "predict"]}),
        ("fit", "In **one** expression, fit a `StandardScaler` on `X` and use it to transform `X_new`.",
         "from sklearn.preprocessing import StandardScaler\nStandardScaler().fit(X).transform(X_new)",
         {"require": ["StandardScaler", "transform"]}),
        ("fit", f"Fit `{LR}` on `X, y` and check that `fit` returned the very same object: return `clf.fit(X, y) is clf`.",
         f"from sklearn.linear_model import LogisticRegression\nclf = {LR}\nresult = clf.fit(X, y) is clf"),
    ]
    return catalog(rng, items, setup + f"\nX_new = np.array({lit(rows)})", f"{desc}; `X_new` holds {len(rows)} new rows.",
                   tol=1e-5)


# ================================================================= easy

@template(M, "easy")
def same_api_any_model(rng):
    setup, desc, k, d = _data(rng)
    models = [("KNeighborsClassifier", "sklearn.neighbors", f"KNeighborsClassifier(n_neighbors={rng.choice([3, 5, 7])})"),
              ("DecisionTreeClassifier", "sklearn.tree",
               f"DecisionTreeClassifier(max_depth={rng.randint(2, 4)}, random_state={seeded(rng)})"),
              ("GaussianNB", "sklearn.naive_bayes", "GaussianNB()"),
              ("LogisticRegression", "sklearn.linear_model", LR)]
    name, mod, ctor = rng.choice(models)
    if rng.random() < 0.5:
        return P(f"{desc}. Every scikit-learn model has the same API. Fit `{ctor}` (from `{mod}`) on `X, y` and return "
                 f"its training accuracy with `score`.", setup,
                 f"from {mod} import {name}\nresult = {ctor}.fit(X, y).score(X, y)", require=[name, "score"])
    return P(f"{desc}. Fit `{ctor}` (from `{mod}`) on `X, y`, predict `X`, and compute the training accuracy **by "
             f"hand** as the fraction of predictions equal to `y` (it matches `score`).", setup,
             f"from {mod} import {name}\nresult = ({ctor}.fit(X, y).predict(X) == y).mean()", require=[name, "predict"])


@template(M, "easy")
def proba_to_labels(rng):
    setup, desc, k, d = _data(rng)
    n = rng.randint(4, 8)
    if rng.random() < 0.5:
        return P(f"{desc}. Fit `clf = {LR}` on `X, y`. `predict` returns the class with the highest probability: "
                 f"rebuild the predictions of the first {n} rows yourself from `predict_proba` (argmax per row) and "
                 f"`classes_`.", setup,
                 f"from sklearn.linear_model import LogisticRegression\nclf = {LR}.fit(X, y)\n"
                 f"result = clf.classes_[clf.predict_proba(X[:{n}]).argmax(axis=1)]",
                 require=["predict_proba", "classes_"])
    return P(f"{desc}. Fit `clf = {LR}` on `X, y` and return, for the first {n} rows, the probability the model gives "
             f"to its **predicted** class (the largest probability in each row of `predict_proba`).", setup,
             f"from sklearn.linear_model import LogisticRegression\nclf = {LR}.fit(X, y)\n"
             f"result = clf.predict_proba(X[:{n}]).max(axis=1)", require=["predict_proba"], tol=1e-5)


@template(M, "easy")
def decision_function_binary(rng):
    d = synth_clf(rng, n=(80, 200), d=(3, 6), classes=2)
    n = rng.randint(3, 6)
    task = rng.choice(["manual", "sign", "log"])
    head = f"{d.desc}. Fit `clf = {LR}` on `X, y`."
    sol = f"from sklearn.linear_model import LogisticRegression\nclf = {LR}.fit(X, y)\n"
    if task == "manual":
        return P(f"{head} For a binary linear model, `decision_function(X)` is `X @ coef_.T + intercept_` (one margin per "
                 f"row). Compute the margins of the first {n} rows **by hand** from `coef_` and `intercept_`, as a 1-D "
                 f"array.", d.setup, sol + f"result = X[:{n}] @ clf.coef_[0] + clf.intercept_[0]",
                 require=["coef_", "intercept_"], tol=1e-6)
    if task == "sign":
        return P(f"{head} A positive margin means class 1. Using only `decision_function`, return the predicted labels "
                 f"of the first {n} rows as integers (they match `predict`).", d.setup,
                 sol + f"result = (clf.decision_function(X[:{n}]) > 0).astype(int)", require=["decision_function"])
    return P(f"{head} Return the log-probabilities of the first {n} rows (`predict_log_proba`).", d.setup,
             sol + f"result = clf.predict_log_proba(X[:{n}])", require=["predict_log_proba"], tol=1e-6)


@template(M, "easy")
def clone_unfitted(rng):
    setup, desc, k, d = _data(rng)
    c = rng.choice([0.05, 0.2, 3.0, 20.0])
    task = rng.choice(["fitted", "param", "coef"])
    base = (f"{desc}. `clf = {LR}` has been fitted on `X, y`. `sklearn.base.clone` makes a new estimator with the "
            f"same hyperparameters but **nothing learned**.")
    s2 = setup + f"\nfrom sklearn.linear_model import LogisticRegression\nclf = {LR}.fit(X, y)"
    if task == "fitted":
        return P(f"{base} Clone `clf` into `fresh` and return `(hasattr(clf, 'coef_'), hasattr(fresh, 'coef_'))`.", s2,
                 "from sklearn.base import clone\nfresh = clone(clf)\nresult = (hasattr(clf, 'coef_'), hasattr(fresh, 'coef_'))",
                 require=["clone"])
    if task == "param":
        return P(f"{base} Clone `clf`, set the clone's `C` to {c}, and return `(clf.C, clone_C)`: changing the clone "
                 f"leaves the original alone.", s2,
                 f"from sklearn.base import clone\nfresh = clone(clf).set_params(C={c})\nresult = (clf.C, fresh.C)",
                 require=["clone", "set_params"])
    return P(f"{base} Clone `clf` with `C={c}`, fit the clone on `X, y`, and return the clone's training accuracy.", s2,
             f"from sklearn.base import clone\nresult = clone(clf).set_params(C={c}).fit(X, y).score(X, y)",
             require=["clone"])


@template(M, "easy")
def standardize_and_invert(rng):
    rows = _matrix(rng, r=(5, 8), c=(2, 4), lo=0, hi=100)
    task = rng.choice(["moments", "roundtrip", "one_col"])
    setup = f"X = {mat(rows)}"
    if task == "moments":
        return P("Standardize `X` with `StandardScaler` and return `(Z.mean(axis=0), Z.std(axis=0))` of the result `Z`: "
                 "means of 0 and standard deviations of 1 (up to rounding).", setup,
                 "from sklearn.preprocessing import StandardScaler\nZ = StandardScaler().fit_transform(X)\n"
                 "result = (Z.mean(axis=0), Z.std(axis=0))", require=["StandardScaler"], tol=1e-9)
    if task == "roundtrip":
        return P("Standardize `X`, invert the transform, and return the largest absolute difference between the "
                 "recovered matrix and `X` (it's tiny: floating-point error only).", setup,
                 "from sklearn.preprocessing import StandardScaler\nsc = StandardScaler()\n"
                 "result = np.abs(sc.inverse_transform(sc.fit_transform(X)) - X).max()",
                 require=["inverse_transform"])
    j = rng.randrange(len(rows[0]))
    return P(f"Fit a `StandardScaler` on `X` and return the standardized values of column {j} only.", setup,
             f"from sklearn.preprocessing import StandardScaler\nresult = StandardScaler().fit_transform(X)[:, {j}]",
             require=["StandardScaler"])


@template(M, "easy")
def scaler_by_hand(rng):
    rows = _matrix(rng, r=(4, 7), c=(2, 3), lo=0, hi=50)
    setup = f"X = {mat(rows)}"
    if rng.random() < 0.5:
        return P("Reproduce `StandardScaler().fit_transform(X)` **by hand** with NumPy. Careful: scikit-learn divides by "
                 "the *population* standard deviation (`np.std`, ddof=0), not the sample one.", setup,
                 "result = (X - X.mean(axis=0)) / X.std(axis=0)", tol=1e-9)
    return P("Fit a `StandardScaler` on `X` and show that `scale_` is the population standard deviation: return "
             "`sc.scale_ - X.std(axis=0)` (all zeros, up to rounding).", setup,
             "from sklearn.preprocessing import StandardScaler\nsc = StandardScaler().fit(X)\nresult = sc.scale_ - X.std(axis=0)",
             require=["scale_"], tol=1e-9)


@template(M, "easy")
def dataframe_names(rng):
    t = table(rng, numeric_only=True, n=(40, 80))
    task = rng.choice(["out", "in", "count"])
    if task == "out":
        return P(f"{t.desc}. Fit a `StandardScaler` on `df` (a DataFrame) and return the names of the columns it "
                 f"outputs (`get_feature_names_out()`).", t.setup,
                 "from sklearn.preprocessing import StandardScaler\nresult = StandardScaler().fit(df).get_feature_names_out()",
                 require=["get_feature_names_out"])
    if task == "in":
        return P(f"{t.desc}. Fitting on a DataFrame records its column names. Fit a `StandardScaler` on `df` and return "
                 f"`feature_names_in_`.", t.setup,
                 "from sklearn.preprocessing import StandardScaler\nresult = StandardScaler().fit(df).feature_names_in_",
                 require=["feature_names_in_"])
    return P(f"{t.desc}. Fit a `StandardScaler` on `df` and return a dict mapping each column name to the mean the "
             f"scaler learned for it.", t.setup,
             "from sklearn.preprocessing import StandardScaler\nsc = StandardScaler().fit(df)\n"
             "result = dict(zip(sc.feature_names_in_, sc.mean_))", require=["mean_"])


# ================================================================= medium

@template(M, "medium")
def fit_on_train_only(rng):
    rows = _matrix(rng, r=(7, 10), c=(2, 3), lo=0, hi=100)
    k = rng.randint(4, len(rows) - 2)
    setup = f"X = {mat(rows)}\nX_train, X_test = X[:{k}], X[{k}:]"
    if rng.random() < 0.5:
        return P(f"THE rule: learn the statistics from the training rows only. Fit a `StandardScaler` on `X_train` (the "
                 f"first {k} rows) and return the transformed `X_test`.", setup,
                 "from sklearn.preprocessing import StandardScaler\nsc = StandardScaler().fit(X_train)\nresult = sc.transform(X_test)",
                 require=["StandardScaler", "transform"])
    return P(f"Fit one `StandardScaler` on `X_train` and transform **both** `X_train` and `X_test` with it. Return "
             f"`(X_train_s, X_test_s)`. (The test rows' means won't be exactly 0: that's expected.)", setup,
             "from sklearn.preprocessing import StandardScaler\nsc = StandardScaler().fit(X_train)\n"
             "X_train_s, X_test_s = sc.transform(X_train), sc.transform(X_test)\nresult = (X_train_s, X_test_s)",
             require=["StandardScaler"])


@template(M, "medium")
def regularization_strength(rng):
    d = synth_clf(rng, n=(100, 220), d=(4, 8), classes=2)
    cs = sorted(rng.sample([0.001, 0.01, 0.1, 1.0, 10.0, 100.0], 4))
    task = rng.choice(["norm", "acc"])
    if task == "norm":
        return P(f"{d.desc}. In `LogisticRegression`, `C` is the *inverse* regularization strength: smaller `C` shrinks "
                 f"the weights more. For each `C` in `{cs}`, fit `LogisticRegression(C=C, max_iter=1000)` on `X, y` and "
                 f"record the L2 norm of `coef_` (`np.linalg.norm`). Return the norms as a list in that order.", d.setup,
                 "from sklearn.linear_model import LogisticRegression\n"
                 f"result = [np.linalg.norm(LogisticRegression(C=C, max_iter=1000).fit(X, y).coef_) for C in {cs}]",
                 require=["LogisticRegression"], tol=1e-4)
    return P(f"{d.desc}. Reuse **one** estimator: create `clf = LogisticRegression(max_iter=1000)` once, then for each "
             f"`C` in `{cs}` call `set_params(C=C)`, refit on `X, y`, and record the training accuracy. Return a dict "
             f"{{C: accuracy}}.", d.setup,
             "from sklearn.linear_model import LogisticRegression\nclf = LogisticRegression(max_iter=1000)\n"
             f"result = {{C: clf.set_params(C=C).fit(X, y).score(X, y) for C in {cs}}}",
             require=["set_params"], tol=1e-9)


@template(M, "medium")
def custom_threshold(rng):
    d = synth_clf(rng, n=(150, 300), d=(4, 8), classes=2, imbalance=rng.choice([0.2, 0.3]))
    t = rng.choice([0.2, 0.3, 0.4, 0.6, 0.7])
    task = rng.choice(["labels", "flips"])
    head = (f"{d.desc} (class 1 is the rarer one). Fit `clf = {LR}` on `X, y`. `predict` uses a probability cut-off "
            f"of 0.5;")
    sol = f"from sklearn.linear_model import LogisticRegression\nclf = {LR}.fit(X, y)\np1 = clf.predict_proba(X)[:, 1]\n"
    if task == "labels":
        return P(f"{head} predict class 1 whenever P(class 1) ≥ {t} instead, and return the number of rows predicted "
                 f"as class 1.", d.setup, sol + f"result = int((p1 >= {t}).sum())", require=["predict_proba"])
    return P(f"{head} with a cut-off of {t} instead, how many rows get a **different** label than `predict` gives? "
             f"Return that count.", d.setup,
             sol + f"result = int(((p1 >= {t}).astype(int) != clf.predict(X)).sum())", require=["predict_proba"])


@template(M, "medium")
def multiclass_coefficients(rng):
    data = rng.choice(["iris", "wine"])
    if data == "iris":
        setup = ("from sklearn.datasets import load_iris\nX, y = load_iris(return_X_y=True)\n"
                 "names = ['sepal length', 'sepal width', 'petal length', 'petal width']")
        desc = "`X` holds 150 iris flowers (4 measurements, named in `names`) and `y` their species (0, 1, 2)"
    else:
        setup = ("from sklearn.datasets import load_wine\nfrom sklearn.preprocessing import StandardScaler\n"
                 "data = load_wine()\nX, y = StandardScaler().fit_transform(data.data), data.target\nnames = data.feature_names")
        desc = "`X` holds 178 wines (13 standardized measurements, named in `names`) and `y` their cultivar (0, 1, 2)"
    task = rng.choice(["argmax", "shape", "top"])
    sol = f"from sklearn.linear_model import LogisticRegression\nclf = {LR}.fit(X, y)\n"
    if task == "shape":
        return P(f"{desc}. Fit `{LR}` on `X, y`. A multiclass linear model has one row of coefficients per class. "
                 f"Return `(clf.coef_.shape, clf.intercept_.shape)`.", setup,
                 sol + "result = (clf.coef_.shape, clf.intercept_.shape)", require=["coef_", "intercept_"])
    if task == "argmax":
        return P(f"{desc}. Fit `{LR}` on `X, y`. For each class, which feature has the largest **absolute** "
                 f"coefficient? Return the list of feature names, one per class.", setup,
                 sol + "result = [names[j] for j in np.abs(clf.coef_).argmax(axis=1)]", require=["coef_"])
    c = rng.randrange(3)
    return P(f"{desc}. Fit `{LR}` on `X, y` and return the names of the 2 features that push hardest **towards** class "
             f"{c} (its 2 largest coefficients, largest first).", setup,
             sol + f"result = [names[j] for j in np.argsort(clf.coef_[{c}])[::-1][:2]]", require=["coef_"])


@template(M, "medium")
def column_order(rng):
    t = table(rng, kind="clf", numeric_only=True, n=(60, 120))
    k = rng.randint(2, 4)
    return P(f"{t.desc}. Fit `clf = {LR}` on `df, y`. New rows arrive in `new` with the **same columns in a different "
             f"order**, which scikit-learn refuses. Reorder `new`'s columns with `clf.feature_names_in_` and return "
             f"`clf.predict` on them.", t.setup + f"\nnew = df.sample({k}, random_state=1)[df.columns[::-1]]",
             f"from sklearn.linear_model import LogisticRegression\nclf = {LR}.fit(df, y)\n"
             "result = clf.predict(new[clf.feature_names_in_])", require=["feature_names_in_"])


@template(M, "medium")
def coef_back_to_units(rng):
    t = table(rng, kind="reg", numeric_only=True, n=(60, 150))
    return P(f"{t.desc}. Fit a `LinearRegression` on standardized features (`sc = StandardScaler().fit(df)`, then fit "
             f"on `sc.transform(df)`). Convert its coefficients back to the original units: divide `coef_` by "
             f"`sc.scale_`; the intercept becomes `intercept_ - (coef_ * sc.mean_ / sc.scale_).sum()`. Return "
             f"`(coef, intercept)` in original units (they equal a regression fitted on `df` directly).", t.setup,
             "from sklearn.linear_model import LinearRegression\nfrom sklearn.preprocessing import StandardScaler\n"
             "sc = StandardScaler().fit(df)\nreg = LinearRegression().fit(sc.transform(df), y)\n"
             "coef = reg.coef_ / sc.scale_\nintercept = reg.intercept_ - (reg.coef_ * sc.mean_ / sc.scale_).sum()\n"
             "result = (coef, intercept)", require=["scale_", "mean_"], tol=1e-6)


@template(M, "medium")
def score_is_a_metric(rng):
    if rng.random() < 0.5:
        t = table(rng, kind="reg", numeric_only=True, n=(60, 150))
        return P(f"{t.desc}. Fit `LinearRegression` on `df, y` and compute its training R² **by hand**: 1 − (sum of "
                 f"squared residuals) / (sum of squared deviations of `y` from its mean). It equals `score`.", t.setup,
                 "from sklearn.linear_model import LinearRegression\npred = LinearRegression().fit(df, y).predict(df)\n"
                 "result = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()", require=["predict"], tol=1e-9)
    setup, desc, k, d = _data(rng)
    m = rng.randint(20, 60)
    return P(f"{desc}. Fit `{LR}` on the first {m} rows only and return its accuracy on the **remaining** rows "
             f"(`score` on held-out data).", setup,
             f"from sklearn.linear_model import LogisticRegression\nresult = {LR}.fit(X[:{m}], y[:{m}]).score(X[{m}:], y[{m}:])",
             require=["score"])


@template(M, "medium")
def iterations(rng):
    d = synth_clf(rng, n=(100, 250), d=(4, 8), classes=2)
    tol = rng.choice(["1e-2", "1e-4", "1e-6"])
    return P(f"{d.desc}. How many iterations did the solver need? Fit `LogisticRegression(tol={tol}, max_iter=1000)` "
             f"on `X, y` and return `n_iter_[0]`.", d.setup,
             f"from sklearn.linear_model import LogisticRegression\nresult = int(LogisticRegression(tol={tol}, "
             f"max_iter=1000).fit(X, y).n_iter_[0])", require=["n_iter_"])


# ================================================================= hard

@template(M, "hard")
def softmax_by_hand(rng):
    setup, desc, k, d = _data(rng, classes=3)
    n = rng.randint(3, 6)
    return P(f"{desc}. Fit `clf = {LR}` on `X, y`. For a multiclass logistic regression, `predict_proba` is the softmax "
             f"of the decision scores `X @ coef_.T + intercept_`. Reproduce `predict_proba` for the first {n} rows "
             f"**by hand** from `coef_` and `intercept_` (subtract each row's max before `np.exp` for stability).",
             setup, f"from sklearn.linear_model import LogisticRegression\nclf = {LR}.fit(X, y)\n"
                    f"z = X[:{n}] @ clf.coef_.T + clf.intercept_\nz = z - z.max(axis=1, keepdims=True)\n"
                    "e = np.exp(z)\nresult = e / e.sum(axis=1, keepdims=True)", require=["coef_", "intercept_"], tol=1e-6)


@template(M, "hard")
def sigmoid_by_hand(rng):
    d = synth_clf(rng, n=(80, 200), d=(3, 7), classes=2)
    n = rng.randint(3, 6)
    return P(f"{d.desc}. Fit `clf = {LR}` on `X, y`. For a binary logistic regression, P(class 1) is the sigmoid of "
             f"the decision function. Rebuild the **full** `predict_proba` matrix of the first {n} rows (columns: "
             f"P(class 0), P(class 1)) from `decision_function` alone.", d.setup,
             f"from sklearn.linear_model import LogisticRegression\nclf = {LR}.fit(X, y)\n"
             f"p1 = 1 / (1 + np.exp(-clf.decision_function(X[:{n}])))\nresult = np.column_stack([1 - p1, p1])",
             require=["decision_function"], tol=1e-6)


@template(M, "hard")
def bagging_with_clone(rng):
    setup, desc, k, d = _data(rng)
    m = rng.randint(3, 6)
    depth = rng.randint(1, 3)
    s = seeded(rng)
    q = rng.choice([15, 20, 25])
    return P(f"{desc}. Build a tiny ensemble with `clone`: `base = DecisionTreeClassifier(max_depth={depth}, "
             f"random_state=0)`. With `rs = np.random.default_rng({s})`, repeat {m} times: draw `idx = "
             f"rs.integers(0, len(X), len(X))` (a bootstrap sample), fit `clone(base)` on `X[idx], y[idx]` and keep "
             f"it. Return the **majority vote** of the {m} trees on every {q}th row of `X` (`X[::{q}]`; ties: the "
             f"smallest label, as `np.bincount(...).argmax()` gives).", setup,
             f"from sklearn.base import clone\nfrom sklearn.tree import DecisionTreeClassifier\n"
             f"base = DecisionTreeClassifier(max_depth={depth}, random_state=0)\nrs = np.random.default_rng({s})\n"
             f"trees = []\nfor _ in range({m}):\n    idx = rs.integers(0, len(X), len(X))\n"
             f"    trees.append(clone(base).fit(X[idx], y[idx]))\nvotes = np.array([t.predict(X[::{q}]) for t in trees])\n"
             f"result = np.array([np.bincount(col).argmax() for col in votes.T])", require=["clone"])


@template(M, "hard")
def changed_params(rng):
    choices = {"C": [0.1, 0.5, 2.0], "max_iter": [200, 500], "fit_intercept": [False], "tol": [1e-3, 1e-5],
               "class_weight": ["'balanced'"], "solver": ["'liblinear'", "'newton-cg'"]}
    keys = rng.sample(sorted(choices), rng.randint(2, 4))
    args = ", ".join(f"{k}={rng.choice(choices[k])}" for k in sorted(keys))
    return P(f"`clf = LogisticRegression({args})`. Which hyperparameters differ from the defaults? Compare "
             f"`clf.get_params()` with `LogisticRegression().get_params()` and return the sorted list of the names whose "
             f"values differ.", f"from sklearn.linear_model import LogisticRegression\nclf = LogisticRegression({args})",
             "default = LogisticRegression().get_params()\n"
             "result = sorted(k for k, v in clf.get_params().items() if default[k] != v)", require=["get_params"])


@template(M, "hard")
def scaled_target(rng):
    t = table(rng, kind="reg", numeric_only=True, n=(60, 150))
    k = rng.randint(3, 5)
    return P(f"{t.desc}. Some models like a standardized **target**. Scale `y` with its own `StandardScaler` (it "
             f"needs a 2-D column: `y.reshape(-1, 1)`), fit `LinearRegression` on `df` and the scaled target, predict "
             f"the first {k} rows of `df`, and map the predictions back to the original units with "
             f"`inverse_transform`. Return them as a 1-D array.", t.setup,
             "from sklearn.linear_model import LinearRegression\nfrom sklearn.preprocessing import StandardScaler\n"
             "ys = StandardScaler().fit(y.reshape(-1, 1))\nreg = LinearRegression().fit(df, ys.transform(y.reshape(-1, 1)))\n"
             f"result = ys.inverse_transform(reg.predict(df.iloc[:{k}]).reshape(-1, 1)).ravel()",
             require=["inverse_transform"], tol=1e-6)


@template(M, "hard")
def model_card(rng):
    setup, desc, k, d = _data(rng)
    return P(f"{desc}. Fit `clf = {LR}` on `X, y` and return a dict summarizing it: `'classes'` (list of ints), "
             f"`'n_features'`, `'accuracy'` (training), `'C'` (from `get_params`) and `'n_weights'` (the number of "
             f"learned numbers: coefficients plus intercepts).", setup,
             f"from sklearn.linear_model import LogisticRegression\nclf = {LR}.fit(X, y)\n"
             "result = {'classes': clf.classes_.tolist(), 'n_features': clf.n_features_in_, 'accuracy': clf.score(X, y), "
             "'C': clf.get_params()['C'], 'n_weights': clf.coef_.size + clf.intercept_.size}",
             require=["classes_", "get_params"], tol=1e-9)


# ================================================================= challenge

@template(M, "challenge")
def rolling_standardize(rng):
    n, q, s = rng.randint(8, 12), rng.randint(4, 6), seeded(rng)
    setup = (f"g = np.random.default_rng({s})\nx = g.normal(50, 15, {n}).round(1)\n"
             f"l = g.integers(0, {n} - 2, {q})\nr = l + 2 + (g.integers(0, {n}, {q}) % ({n} - l - 1))")
    report = rng.choice(["last", "first"])
    setup += f"\nreport = {report!r}"
    pick_ = "x[r - 1]" if report == "last" else "x[l]"
    sol = ("c1 = np.concatenate([[0.0], np.cumsum(x)])\nc2 = np.concatenate([[0.0], np.cumsum(x * x)])\n"
           "cnt = r - l\nmean = (c1[r] - c1[l]) / cnt\nvar = np.maximum((c2[r] - c2[l]) / cnt - mean ** 2, 0)\n"
           f"std = np.sqrt(var)\nstd[std == 0] = 1.0\nresult = ({pick_} - mean) / std")
    tests = [f"g = np.random.default_rng({s + i})\nx = g.normal(50, 15, 200000).round(1)\n"
             f"l = g.integers(0, 199990, 200000)\nr = l + 2 + (g.integers(0, 200000, 200000) % (200000 - l - 1))\n"
             f"report = {report!r}" for i in (1, 2)]
    which = ("the **standardized value of the window's last reading** `x[r[i] - 1]`, exactly as "
             "`StandardScaler().fit_transform(...)[-1]` would" if report == "last" else
             "the **standardized value of the window's first reading** `x[l[i]]`, exactly as "
             "`StandardScaler().fit_transform(...)[0]` would")
    return challenge(
        "Streaming standardization",
        "A monitoring job standardizes a sensor's readings `x` on sliding windows. Query `i` fits a `StandardScaler` on "
        f"the readings `x[l[i]:r[i]]` (at least 2 of them) and reports {which} (population standard deviation; a "
        f"window whose values are all equal has scale 1). Return the answers as a float array. (`report` says "
        f"which reading: {report!r}.)",
        setup, sol, tests, "len(x), number of queries ≤ 2·10^5", speed="O(n + q): prefix sums, no loop over windows",
        answer_desc=" (a NumPy float array, one value per query)", tol=1e-6)


@template(M, "challenge")
def merge_scaler_shards(rng):
    k, s = rng.randint(3, 6), seeded(rng)
    setup = (f"g = np.random.default_rng({s})\nsizes = g.integers(1, 9, {k})\n"
             "shards = [g.normal(g.uniform(-5, 5, 3), g.uniform(0.5, 3, 3), (m, 3)) for m in sizes]\n"
             "counts = np.array([len(a) for a in shards])\nmeans = np.array([a.mean(axis=0) for a in shards])\n"
             "variances = np.array([a.var(axis=0) for a in shards])\ndel shards")
    report = rng.choice(["mean_scale", "var_count"])
    setup += f"\nreport = {report!r}"
    sol = ("N = counts.sum()\nmu = (counts[:, None] * means).sum(axis=0) / N\n"
           "ex2 = (counts[:, None] * (variances + means ** 2)).sum(axis=0) / N\n"
           + ("result = np.vstack([mu, np.sqrt(np.maximum(ex2 - mu ** 2, 0))])" if report == "mean_scale" else
              "result = (np.maximum(ex2 - mu ** 2, 0), int(N))"))
    want = ("return a 2 × d array whose first row is `mean_` and second row is `scale_` (the population standard "
            "deviation)" if report == "mean_scale" else
            "return the tuple `(var_, n_samples_seen_)`: the population variance of every column and the total number "
            "of rows")
    tests = [f"g = np.random.default_rng({s + i})\ncounts = g.integers(1, 50, 200000)\n"
             f"means = g.normal(0, 3, (200000, 3))\nvariances = g.uniform(0.1, 4, (200000, 3))\nreport = {report!r}"
             for i in (1, 2)]
    return challenge(
        "Merging scaler shards",
        "A dataset too big for one machine was split into shards. Each worker fitted its own `StandardScaler` and sent "
        "back only its row count (`counts[i]`), column means (`means[i]`) and population variances (`variances[i]`), "
        f"never the rows. Combine them into the statistics a single `StandardScaler` fitted on **all** rows would "
        f"learn: {want}.", setup, sol, tests, "number of shards ≤ 2·10^5, d = 3", speed="O(shards · d), vectorized",
        tol=1e-6)


@template(M, "challenge")
def prefix_r2(rng):
    n, s = rng.randint(7, 11), seeded(rng)
    setup = (f"g = np.random.default_rng({s})\nx = g.uniform(0, 10, {n}).round(3)\n"
             f"y = (g.uniform(-2, 2) * x + g.normal(0, 3, {n})).round(3)")
    report = rng.choice(["r2", "slope", "mse"])
    setup += f"\nreport = {report!r}"
    last = {"r2": "result = (cxy ** 2 / (vx * vy))[2:]", "slope": "result = (cxy / vx)[2:]",
            "mse": "result = ((vy - cxy ** 2 / vx) / k)[2:]"}[report]
    sol = ("k = np.arange(1, len(x) + 1)\nsx, sy = np.cumsum(x), np.cumsum(y)\n"
           "sxx, syy, sxy = np.cumsum(x * x), np.cumsum(y * y), np.cumsum(x * y)\n"
           "vx = sxx - sx * sx / k\nvy = syy - sy * sy / k\ncxy = sxy - sx * sy / k\n" + last)
    what = {"r2": "its **training R²** (`score`)", "slope": "its **slope** (`coef_[0]`)",
            "mse": "its **training mean squared error** (mean of squared residuals)"}[report]
    tests = [f"g = np.random.default_rng({s + i})\nx = g.uniform(0, 10, 200000).round(3)\n"
             f"y = (g.uniform(-2, 2) * x + g.normal(0, 3, 200000)).round(3)\nreport = {report!r}" for i in (1, 2)]
    return challenge(
        "Learning curve in one pass",
        "Points `(x[i], y[i])` arrive one by one. After each arrival from the 3rd on, an analyst refits "
        f"`LinearRegression` on all points so far (one feature) and records {what}. Return the recorded values "
        f"(length `len(x) - 2`) without refitting from scratch each time. (The x values are distinct and the targets "
        f"noisy, so no prefix is degenerate.)", setup, sol, tests,
        "len(x) ≤ 2·10^5", speed="O(n): running sums", answer_desc=" (a float array)", tol=1e-6)


# ================================================================= catalogs: many tasks on one dataset

def _fitted(rng, classes=(2, 3)):
    setup, desc, k, d = _data(rng, classes)
    return setup + f"\nfrom sklearn.linear_model import LogisticRegression\nclf = {LR}.fit(X, y)", desc, k, d


@template(M, "easy")
def api_tour(rng):
    setup, desc, k, d = _fitted(rng)
    n, i, c = rng.randint(3, 6), rng.randint(0, 79), rng.randrange(2)
    a = rng.randint(10, 40)
    items = [
        ("predict", f"Return `clf`'s predictions for rows {a} to {a + n - 1} of `X` (inclusive).",
         f"result = clf.predict(X[{a}:{a + n}])"),
        ("predict", f"Predict the single row `X[{i}]` and return the label as a plain int. (`predict` wants a 2-D input: "
                    f"`X[{i}:{i + 1}]` or `X[[{i}]]`.)", f"result = int(clf.predict(X[[{i}]])[0])"),
        ("predict", "How many rows does `clf` assign to each class? Return `np.bincount` of its predictions on `X`.",
         "result = np.bincount(clf.predict(X))"),
        ("predict", "Return the **indices** of the rows of `X` that `clf` misclassifies (`np.flatnonzero`).",
         "result = np.flatnonzero(clf.predict(X) != y)"),
        ("predict", "Return the number of training rows `clf` gets wrong.", "result = int((clf.predict(X) != y).sum())"),
        ("predict_proba", f"Return P(class {c}) for the first {n} rows: one column of `predict_proba`.",
         f"result = clf.predict_proba(X[:{n}])[:, {c}]"),
        ("predict_proba", "Return the model's average confidence on `X`: the mean over rows of the largest class "
                          "probability.", "result = clf.predict_proba(X).max(axis=1).mean()"),
        ("predict_proba", "Check that probabilities sum to 1: return the largest |row sum − 1| of `predict_proba(X)`.",
         "result = np.abs(clf.predict_proba(X).sum(axis=1) - 1).max()"),
        ("decision_function", "Return the shape of `clf.decision_function(X)` as a tuple.",
         "result = clf.decision_function(X).shape"),
        ("classes_", "Return `clf.classes_` as a Python list of ints.", "result = clf.classes_.tolist()"),
        ("coef_", "How many coefficients did `clf` learn in total (all classes)? Return `coef_.size`.",
         "result = clf.coef_.size"),
        ("score", "Return `clf.score(X, y) == (clf.predict(X) == y).mean()`: accuracy is the classifier's default score.",
         "result = bool(clf.score(X, y) == (clf.predict(X) == y).mean())"),
    ]
    return catalog(rng, items, setup, f"{desc}; `clf = {LR}` has been fitted on them.", tol=1e-6)


@template(M, "medium")
def api_tasks(rng):
    setup, desc, k, d = _fitted(rng)
    n, m = rng.randint(3, 6), rng.randint(25, 60)
    c_small = rng.choice([0.001, 0.01, 0.05])
    items = [
        ("score", f"Fit a fresh `{LR}` on the first {m} rows only and return its accuracy on the rest.",
         f"result = {LR}.fit(X[:{m}], y[:{m}]).score(X[{m}:], y[{m}:])", {"require": ["fit", "score"]}),
        ("predict", "Return the accuracy of `clf` **per class** (the fraction of each class's rows it predicts "
                    "correctly), as a list in class order.",
         "pred = clf.predict(X)\nresult = [float((pred[y == c] == c).mean()) for c in clf.classes_]"),
        ("predict_proba", f"Return the indices of the {n} rows `clf` is **least** sure about (smallest top-class "
                          f"probability), least sure first.",
         f"result = np.argsort(clf.predict_proba(X).max(axis=1))[:{n}]"),
        ("predict", "Build the confusion counts by hand: a k × k int array `C` where `C[t, p]` counts rows of true class "
                    "`t` predicted as `p` (`np.add.at` or a loop).",
         "pred = clf.predict(X)\nresult = np.zeros((len(clf.classes_),) * 2, dtype=int)\nnp.add.at(result, (y, pred), 1)"),
        ("fit_intercept", f"Fit `LogisticRegression(fit_intercept=False, max_iter=1000)` on `X, y` and return "
                          f"`(clf.score(X, y), no_intercept_accuracy)`.",
         "nb = LogisticRegression(fit_intercept=False, max_iter=1000).fit(X, y)\nresult = (clf.score(X, y), nb.score(X, y))"),
        ("set_params", f"Refit a clone of `clf` with `C={c_small}` (strong regularization) and return how many training "
                       f"predictions change compared with `clf`.",
         f"from sklearn.base import clone\nstrong = clone(clf).set_params(C={c_small}).fit(X, y)\n"
         "result = int((strong.predict(X) != clf.predict(X)).sum())", {"require": ["clone|set_params"]}),
        ("predict_proba", "Return the mean **entropy** (in nats) of `clf`'s predicted distributions on `X`: "
                          "−Σ p·log p per row, averaged.",
         "p = clf.predict_proba(X)\nresult = float(-(p * np.log(p)).sum(axis=1).mean())"),
        ("predict_proba", f"For the first {n} rows, return the gap between the two largest class probabilities "
                          f"(`np.sort` each row).",
         f"p = np.sort(clf.predict_proba(X[:{n}]), axis=1)\nresult = p[:, -1] - p[:, -2]"),
        ("StandardScaler", f"Standardize `X` with a `StandardScaler`, fit a new `{LR}` on the scaled data and return "
                           f"its training accuracy.",
         f"from sklearn.preprocessing import StandardScaler\nZ = StandardScaler().fit_transform(X)\n"
         f"result = {LR}.fit(Z, y).score(Z, y)"),
        ("StandardScaler", "Standardize `X` and count the values more than 2 standard deviations from their column's "
                           "mean (|z| > 2).",
         "from sklearn.preprocessing import StandardScaler\nresult = int((np.abs(StandardScaler().fit_transform(X)) > 2).sum())"),
        ("with_std", "Only center `X` (no scaling): `StandardScaler(with_std=False)`. Return the first transformed row.",
         "from sklearn.preprocessing import StandardScaler\nresult = StandardScaler(with_std=False).fit_transform(X)[0]"),
        ("with_mean", "Only scale `X` (no centering): `StandardScaler(with_mean=False)`. Return the column means of the "
                      "result.",
         "from sklearn.preprocessing import StandardScaler\nresult = StandardScaler(with_mean=False).fit_transform(X).mean(axis=0)"),
        ("decision_function", "The predicted class is the column with the largest decision score. Return "
                              "`clf.classes_[...]` of the argmax of `decision_function(X)` for multiclass data, or "
                              "`(scores > 0)` for binary, and check it equals `predict`: return the agreement rate.",
         "s = clf.decision_function(X)\nlab = (s > 0).astype(int) if s.ndim == 1 else clf.classes_[s.argmax(axis=1)]\n"
         "result = float((lab == clf.predict(X)).mean())"),
        ("predict_log_proba", f"Return the largest absolute difference between `predict_log_proba(X[:{n}])` and "
                              f"`np.log(predict_proba(X[:{n}]))` (tiny).",
         f"result = float(np.abs(clf.predict_log_proba(X[:{n}]) - np.log(clf.predict_proba(X[:{n}]))).max())"),
    ]
    return catalog(rng, items, setup, f"{desc}; `clf = {LR}` has been fitted on them (`LogisticRegression` is "
                                      f"imported).", tol=1e-6)


@template(M, "medium")
def regression_tasks(rng):
    t = table(rng, kind="reg", numeric_only=True, n=(60, 160))
    n = rng.randint(3, 5)
    col = rng.choice(t.num)
    items = [
        ("predict", f"Return the residuals `y − prediction` of the first {n} rows.",
         f"result = y[:{n}] - reg.predict(df.iloc[:{n}])"),
        ("predict", "Return the mean of the training residuals (ordinary least squares with an intercept makes it ~0).",
         "result = float((y - reg.predict(df)).mean())", {"tol": 1e-6}),
        ("predict", "Predict a new row given as a dict: build a one-row DataFrame from `new_row` (keep the column order "
                    "of `df`) and return the prediction as a float.",
         "result = float(reg.predict(pd.DataFrame([new_row])[df.columns])[0])"),
        ("coef_", "Return a dict mapping each column name to its coefficient.",
         "result = dict(zip(df.columns, reg.coef_))"),
        ("coef_", f"By how much does the prediction change when `{col}` grows by 10 and nothing else changes? Return it "
                  f"(10 × its coefficient).", f"result = 10 * reg.coef_[list(df.columns).index({col!r})]"),
        ("score", "Return the training R² of `reg`.", "result = reg.score(df, y)"),
        ("intercept_", "Return the prediction for an all-zeros row: it's the intercept.", "result = reg.intercept_"),
        ("score", f"Refit on the first half of the rows and return R² on the second half.",
         "h = len(df) // 2\nresult = LinearRegression().fit(df.iloc[:h], y[:h]).score(df.iloc[h:], y[h:])",
         {"require": ["score"]}),
    ]
    first = {c: 1 for c in t.num}
    return catalog(rng, items, t.setup + f"\nfrom sklearn.linear_model import LinearRegression\nreg = "
                                         f"LinearRegression().fit(df, y)\nnew_row = {lit(first)}",
                   f"{t.desc}; `reg = LinearRegression()` has been fitted on them. `new_row` is a dict of feature "
                   f"values.", tol=1e-6)


@template(M, "hard")
def from_scratch(rng):
    setup, desc, k, d = _data(rng)
    m = rng.randint(4, 7)
    s = seeded(rng)
    sizes = sorted(rng.sample([20, 30, 40, 60, 80], 3))
    cs = sorted(rng.sample([0.01, 0.1, 1.0, 10.0, 100.0], 3))
    items = [
        ("LogisticRegression", "Do one-vs-rest by hand: for each class `c` fit a binary `LogisticRegression(max_iter="
                               "1000)` on `(X, y == c)`, then predict, for every row, the class whose model gives the "
                               "largest `decision_function`. Return the training accuracy of this combined model.",
         "from sklearn.linear_model import LogisticRegression\ncs = np.unique(y)\n"
         "S = np.column_stack([LogisticRegression(max_iter=1000).fit(X, y == c).decision_function(X) for c in cs])\n"
         "result = float((cs[S.argmax(axis=1)] == y).mean())"),
        ("lstsq", "`LinearRegression` solves ordinary least squares. Reproduce what it would learn on `X` and the "
                  "target `y.astype(float)` with `np.linalg.lstsq` on `X` with a column of ones appended, and return "
                  "the solution `[*coef, intercept]`.",
         "A = np.column_stack([X, np.ones(len(X))])\nresult = np.linalg.lstsq(A, y.astype(float), rcond=None)[0]"),
        ("score", f"A learning curve by hand: hold out the last 40 rows; for each training size in `{sizes}` fit "
                  f"`LogisticRegression(max_iter=1000)` on the first that many rows and record the held-out accuracy. "
                  f"Return the list.",
         "from sklearn.linear_model import LogisticRegression\nXt, yt = X[-40:], y[-40:]\n"
         f"result = [LogisticRegression(max_iter=1000).fit(X[:n], y[:n]).score(Xt, yt) for n in {sizes}]",
         {"setup": f"order = np.random.default_rng({s}).permutation(len(X))\nX, y = X[order], y[order]"}),
        ("coef_", f"How stable is the first coefficient? With `rs = np.random.default_rng({s})`, fit "
                  f"`LogisticRegression(max_iter=1000)` on {m} bootstrap samples (`idx = rs.integers(0, len(X), len(X))` "
                  f"each time) and return the standard deviation of `coef_[0, 0]` across them.",
         f"from sklearn.linear_model import LogisticRegression\nrs = np.random.default_rng({s})\nvals = []\n"
         f"for _ in range({m}):\n    idx = rs.integers(0, len(X), len(X))\n"
         f"    vals.append(LogisticRegression(max_iter=1000).fit(X[idx], y[idx]).coef_[0, 0])\nresult = float(np.std(vals))",
         {"tol": 1e-5}),
        ("set_params", f"Pick `C` on a validation split: train on the first 60% of rows, score on the rest, for `C` in "
                       f"`{cs}` (one `LogisticRegression(max_iter=1000)` object, changed with `set_params`). Return the "
                       f"best `C` (the first one on ties).",
         "from sklearn.linear_model import LogisticRegression\nh = int(0.6 * len(X))\nclf = LogisticRegression(max_iter=1000)\n"
         f"scores = [clf.set_params(C=C).fit(X[:h], y[:h]).score(X[h:], y[h:]) for C in {cs}]\n"
         f"result = {cs}[int(np.argmax(scores))]",
         {"setup": f"order = np.random.default_rng({s}).permutation(len(X))\nX, y = X[order], y[order]"}),
        ("clone", "Compare depths with one template: `base = DecisionTreeClassifier(random_state=0)`; for `max_depth` in "
                  "1, 2, 3, 4 fit `clone(base).set_params(max_depth=...)` on `X, y` and return the training accuracies.",
         "from sklearn.base import clone\nfrom sklearn.tree import DecisionTreeClassifier\nbase = DecisionTreeClassifier(random_state=0)\n"
         "result = [clone(base).set_params(max_depth=k).fit(X, y).score(X, y) for k in (1, 2, 3, 4)]"),
        ("log", "Compute the training **log loss** by hand: the mean over rows of −log(probability given to the true "
                "class), using a `LogisticRegression(max_iter=1000)` fitted on `X, y`.",
         "from sklearn.linear_model import LogisticRegression\np = LogisticRegression(max_iter=1000).fit(X, y).predict_proba(X)\n"
         "result = float(-np.log(p[np.arange(len(y)), y]).mean())", {"require": ["predict_proba"]}),
        ("StandardScaler", "Standardize **only the columns with a mean above the overall median of the column means**: fit "
                           "a `StandardScaler` on those columns, put the scaled values back into a float copy of `X`, "
                           "and return the copy.",
         "from sklearn.preprocessing import StandardScaler\ncols = X.mean(axis=0) > np.median(X.mean(axis=0))\n"
         "Z = X.astype(float).copy()\nZ[:, cols] = StandardScaler().fit_transform(X[:, cols])\nresult = Z"),
    ]
    return catalog(rng, items, setup, f"{desc}.", tol=1e-6)


@template(M, "medium")
def scaler_tasks(rng):
    t = table(rng, numeric_only=True, n=(50, 120))
    new = {c: 1 for c in t.num}
    items = [
        ("transform", "Fit a `StandardScaler` on `df` and return the z-scores of the **first row** as a dict "
                      "{column: z}.",
         "from sklearn.preprocessing import StandardScaler\nsc = StandardScaler().fit(df)\n"
         "result = dict(zip(df.columns, sc.transform(df.iloc[:1])[0]))"),
        ("scale_", "Which column has the largest spread? Fit a `StandardScaler` on `df` and return the name of the column "
                   "with the largest `scale_`.",
         "from sklearn.preprocessing import StandardScaler\nsc = StandardScaler().fit(df)\nresult = df.columns[sc.scale_.argmax()]"),
        ("transform", "Fit a `StandardScaler` on `df` and standardize the new row `new_row` (a dict; make a one-row "
                      "DataFrame with `df`'s columns). Return the transformed row as a 1-D array.",
         "from sklearn.preprocessing import StandardScaler\nsc = StandardScaler().fit(df)\n"
         "result = sc.transform(pd.DataFrame([new_row])[df.columns])[0]"),
        ("fit_transform", "Return the names of the columns that have at least one value more than 2.5 standard "
                          "deviations from their mean (standardize with `StandardScaler`).",
         "from sklearn.preprocessing import StandardScaler\nZ = StandardScaler().fit_transform(df)\n"
         "result = [c for c, bad in zip(df.columns, (np.abs(Z) > 2.5).any(axis=0)) if bad]"),
        ("inverse_transform", "What original row does an all-zeros standardized row stand for? Return "
                              "`inverse_transform` of `np.zeros((1, n_columns))` as a 1-D array (it's the column means).",
         "from sklearn.preprocessing import StandardScaler\nsc = StandardScaler().fit(df)\n"
         "result = sc.inverse_transform(np.zeros((1, df.shape[1])))[0]"),
        ("set_output", "Make the scaler return a DataFrame: `StandardScaler().set_output(transform='pandas')`, fit-transform "
                       "`df`, and return the result's first 3 rows.",
         "from sklearn.preprocessing import StandardScaler\nresult = StandardScaler().set_output(transform='pandas').fit_transform(df).head(3)"),
    ]
    return catalog(rng, items, t.setup + f"\nnew_row = {lit(new)}", f"{t.desc}. `new_row` is a dict of feature values.",
                   tol=1e-6)


@template(M, "hard")
def regression_diagnostics(rng):
    t = table(rng, kind="reg", numeric_only=True, n=(80, 180))
    a, b = rng.sample(t.num, 2)
    pct = rng.choice([5, 10])
    items = [
        ("StandardScaler", "Which feature matters most? Standardize `df` (so coefficients are comparable), fit "
                           "`LinearRegression`, and return the name of the column with the largest |coefficient|.",
         "from sklearn.linear_model import LinearRegression\nfrom sklearn.preprocessing import StandardScaler\n"
         "reg = LinearRegression().fit(StandardScaler().fit_transform(df), y)\nresult = df.columns[np.abs(reg.coef_).argmax()]"),
        ("predict", "Fit `LinearRegression` on `df, y` and return the **index label** of the row with the largest "
                    "absolute residual.",
         "from sklearn.linear_model import LinearRegression\nres = y - LinearRegression().fit(df, y).predict(df)\n"
         "result = df.index[np.abs(res).argmax()]"),
        ("score", f"Robustness check: fit `LinearRegression`, drop the {pct}% of rows with the largest absolute "
                  f"residuals (keep those at or below `np.percentile(|res|, {100 - pct})`), refit on the rest and "
                  f"return its R² on the rows kept.",
         "from sklearn.linear_model import LinearRegression\nres = np.abs(y - LinearRegression().fit(df, y).predict(df))\n"
         f"keep = res <= np.percentile(res, {100 - pct})\nresult = LinearRegression().fit(df[keep], y[keep]).score(df[keep], y[keep])"),
        ("score", "Which feature can the model least afford to lose? For each column, fit `LinearRegression` without it "
                  "and record the training R²; return the column whose removal gives the **lowest** R².",
         "from sklearn.linear_model import LinearRegression\n"
         "r2 = {c: LinearRegression().fit(df.drop(columns=c), y).score(df.drop(columns=c), y) for c in df.columns}\n"
         "result = min(r2, key=r2.get)"),
        ("transform", "Fit a `StandardScaler` on `df` and a `LinearRegression` on the scaled data, then predict "
                      "`new_row` (a dict): scale it with the **same** scaler first. Return the prediction as a float.",
         "from sklearn.linear_model import LinearRegression\nfrom sklearn.preprocessing import StandardScaler\n"
         "sc = StandardScaler().fit(df)\nreg = LinearRegression().fit(sc.transform(df), y)\n"
         "result = float(reg.predict(sc.transform(pd.DataFrame([new_row])[df.columns]))[0])"),
        ("PolynomialFeatures", f"Does an interaction help? Add a column `{a} × {b}` to a copy of `df`, refit "
                               f"`LinearRegression`, and return the gain in training R² over the plain model.",
         "from sklearn.linear_model import LinearRegression\nd2 = df.copy()\n"
         f"d2['inter'] = df[{a!r}] * df[{b!r}]\n"
         "result = LinearRegression().fit(d2, y).score(d2, y) - LinearRegression().fit(df, y).score(df, y)",
         {"require": ["LinearRegression"]}),
        ("predict", "How many training predictions fall **outside** the range of observed targets "
                    "`[y.min(), y.max()]`? Return the count for a `LinearRegression` fitted on `df, y`.",
         "from sklearn.linear_model import LinearRegression\np = LinearRegression().fit(df, y).predict(df)\n"
         "result = int(((p < y.min()) | (p > y.max())).sum())"),
        ("sample_weight", "Recent rows matter more: fit `LinearRegression` with `sample_weight=np.linspace(0.5, 1.5, "
                          "len(df))` (rows are in time order) and return its coefficients.",
         "from sklearn.linear_model import LinearRegression\n"
         "result = LinearRegression().fit(df, y, sample_weight=np.linspace(0.5, 1.5, len(df))).coef_"),
        ("StandardScaler", "Standardize the **target** too: fit `LinearRegression` on standardized `df` and standardized "
                           "`y` (each with its own `StandardScaler`) and return the coefficients (standardized betas).",
         "from sklearn.linear_model import LinearRegression\nfrom sklearn.preprocessing import StandardScaler\n"
         "Xs = StandardScaler().fit_transform(df)\nys = StandardScaler().fit_transform(y.reshape(-1, 1)).ravel()\n"
         "result = LinearRegression().fit(Xs, ys).coef_"),
        ("score", "Out-of-sample R² by halves: fit `LinearRegression` on the even-numbered rows (`df.iloc[::2]`) and "
                  "return R² on the odd ones, and the other way round, as a tuple.",
         "from sklearn.linear_model import LinearRegression\nA, B = df.iloc[::2], df.iloc[1::2]\nya, yb = y[::2], y[1::2]\n"
         "result = (LinearRegression().fit(A, ya).score(B, yb), LinearRegression().fit(B, yb).score(A, ya))"),
    ]
    new = {c: 1.0 for c in t.num}
    return catalog(rng, items, t.setup + f"\nnew_row = {lit(new)}", f"{t.desc}. `new_row` is a dict of feature values.",
                   tol=1e-6)


@template(M, "hard")
def composed_tasks(rng):
    d = synth_clf(rng, n=(150, 300), d=(4, 7), classes=2)
    j = rng.randrange(d.d)
    s = seeded(rng)
    recall = rng.choice([0.8, 0.9, 0.95])
    k = rng.choice([3, 5, 9])
    setup = d.setup + f"\nfrom sklearn.linear_model import LogisticRegression\nclf = {LR}.fit(X, y)"
    items = [
        ("predict_proba", f"Partial dependence by hand: for each value `v` in `np.linspace(X[:, {j}].min(), X[:, {j}].max(), "
                          f"5)`, set column {j} of a copy of `X` to `v` and record the **average** P(class 1) that "
                          f"`clf` predicts. Return the 5 averages.",
         f"grid = np.linspace(X[:, {j}].min(), X[:, {j}].max(), 5)\nout = []\nfor v in grid:\n    Xc = X.copy()\n"
         f"    Xc[:, {j}] = v\n    out.append(clf.predict_proba(Xc)[:, 1].mean())\nresult = np.array(out)"),
        ("KNeighborsClassifier", f"Fit `KNeighborsClassifier(n_neighbors={k})` on `X, y` too. Return `(agreement, "
                                 f"acc_where_agree)`: the fraction of rows where it and `clf` predict the same label, "
                                 f"and the accuracy on those rows.",
         f"from sklearn.neighbors import KNeighborsClassifier\na = clf.predict(X)\nb = KNeighborsClassifier(n_neighbors={k}).fit(X, y).predict(X)\n"
         "same = a == b\nresult = (float(same.mean()), float((a[same] == y[same]).mean()))"),
        ("score", f"Permutation importance by hand: for each column `c`, shuffle it in a copy of `X` with "
                  f"`np.random.default_rng({s}).permutation` (a **new** generator with that seed for every column) "
                  f"and record `clf.score(X, y) − clf.score(X_shuffled, y)`. Return the drops as an array.",
         f"base = clf.score(X, y)\nout = []\nfor c in range(X.shape[1]):\n    Xc = X.copy()\n"
         f"    Xc[:, c] = np.random.default_rng({s}).permutation(Xc[:, c])\n    out.append(base - clf.score(Xc, y))\n"
         "result = np.array(out)"),
        ("predict_proba", f"Pick a decision threshold for recall ≥ {recall}: among the distinct values of P(class 1) on "
                          f"`X`, return the **largest** threshold `t` such that predicting class 1 when P ≥ t catches at "
                          f"least {recall:.0%} of the true class-1 rows.",
         f"p = clf.predict_proba(X)[:, 1]\nts = np.unique(p)\nrec = np.array([(p[y == 1] >= t).mean() for t in ts])\n"
         f"result = float(ts[rec >= {recall}].max())"),
        ("GaussianNB", "Soft voting by hand: average the `predict_proba` of `clf` and of a `GaussianNB` fitted on `X, y`, "
                       "predict the class with the larger average probability, and return the training accuracy.",
         "from sklearn.naive_bayes import GaussianNB\np = (clf.predict_proba(X) + GaussianNB().fit(X, y).predict_proba(X)) / 2\n"
         "result = float((p.argmax(axis=1) == y).mean())"),
        ("predict_proba", "Which rows does `clf` get wrong **with confidence**? Return the indices of misclassified rows "
                          "whose top probability is above 0.8.",
         "p = clf.predict_proba(X)\nresult = np.flatnonzero((clf.predict(X) != y) & (p.max(axis=1) > 0.8))"),
        ("coef_", "Rank the features by how much they move the log-odds per standard deviation: return the column indices "
                  "sorted by |coef_[0] × X.std(axis=0)|, largest first.",
         "result = np.argsort(-np.abs(clf.coef_[0] * X.std(axis=0)))"),
        ("decision_function", "Return the 3 rows closest to the decision boundary (smallest |decision_function|), "
                              "closest first.", "result = np.argsort(np.abs(clf.decision_function(X)))[:3]"),
    ]
    return catalog(rng, items, setup, f"{d.desc}; `clf = {LR}` has been fitted on them.", tol=1e-6)
