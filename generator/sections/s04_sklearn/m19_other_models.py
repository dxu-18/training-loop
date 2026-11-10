"""4.19 Outliers, multi-output, semi-supervised learning and other models (MLP, Gaussian processes, QDA, isotonic
regression, kernel approximation, dummy baselines)."""

from ._common import P, catalog, challenge, lit, seeded, synth_clf, table, template

M = "4.19"
LR = "LogisticRegression(max_iter=1000)"
LRI = "from sklearn.linear_model import LogisticRegression\n"


# ================================================================= data

def _outliers(rng, n=(150, 260), d=(2, 4), m=(4, 10)):
    """Gaussian inliers plus a few uniform outliers, shuffled; `is_outlier` marks the injected ones."""
    n_, d_, m_ = rng.randint(*n), rng.randint(*d), rng.randint(*m)
    s = seeded(rng)
    setup = (f"g = np.random.default_rng({s})\ninliers = g.normal(0, 1, ({n_}, {d_}))\n"
             f"outliers = g.uniform(-6, 6, ({m_}, {d_}))\nX = np.vstack([inliers, outliers])\n"
             f"is_outlier = np.r_[np.zeros({n_}, int), np.ones({m_}, int)]\n"
             "order = g.permutation(len(X))\nX, is_outlier = X[order], is_outlier[order]")
    desc = (f"`X` holds {n_ + m_} points in {d_}-D: {n_} from a standard normal cloud and {m_} injected outliers "
            f"(`is_outlier` marks them with 1)")
    return setup, desc, n_ + m_, d_, m_


def _cancer(rng, n=(200, 380)):
    n_, s = rng.randint(*n), seeded(rng)
    return (f"from sklearn.datasets import load_breast_cancer\nfrom sklearn.preprocessing import StandardScaler\n"
            f"X, y = load_breast_cancer(return_X_y=True)\nrows = np.random.default_rng({s}).permutation(len(X))[:{n_}]\n"
            f"X, y = StandardScaler().fit_transform(X[rows]), y[rows]",
            f"`X` holds {n_} breast-tumour samples (30 standardized measurements; rows shuffled) and `y` the "
            f"diagnosis (0 malignant, 1 benign)", n_)


def _clf(rng, classes=2, n=(160, 320)):
    """Scaled classification data: breast cancer (binary) or make_classification."""
    if classes == 2 and rng.random() < 0.4:
        return _cancer(rng, n)
    d = synth_clf(rng, n=n, d=(4, 8), classes=classes)
    return d.setup, d.desc, d.n


def _multiclass(rng):
    r = rng.random()
    if r < 0.3:
        return ("from sklearn.datasets import load_iris\nX, y = load_iris(return_X_y=True)\n"
                f"order = np.random.default_rng({seeded(rng)}).permutation(len(X))\nX, y = X[order], y[order]",
                "`X` holds 150 iris flowers (4 measurements; rows shuffled) and `y` their species (0, 1, 2)", 3)
    if r < 0.5:
        return ("from sklearn.datasets import load_wine\nfrom sklearn.preprocessing import StandardScaler\n"
                "X, y = load_wine(return_X_y=True)\nX = StandardScaler().fit_transform(X)",
                "`X` holds 178 wines (13 standardized measurements) and `y` their cultivar (0, 1, 2)", 3)
    n, d, k, s = rng.randint(150, 300), rng.randint(4, 7), rng.randint(3, 5), seeded(rng)
    return (f"from sklearn.datasets import make_classification\nX, y = make_classification(n_samples={n}, "
            f"n_features={d}, n_informative={d}, n_redundant=0, n_classes={k}, random_state={s})",
            f"`X` ({n} × {d}) and `y` ({k} classes) come from `make_classification`", k)


def _hidden(rng, frac=None):
    f = frac or rng.choice([0.8, 0.9, 0.95])
    s = seeded(rng)
    return f, (f"\nmask = np.random.default_rng({s}).random(len(y)) < {f}   # rows whose label is hidden\n"
               "for c in np.unique(y):   # but every class keeps at least one label\n"
               "    mask[np.flatnonzero(y == c)[0]] = False\n"
               "y_partial = y.copy()\ny_partial[mask] = -1")


def _sine(rng, n=(15, 30)):
    n_, s = rng.randint(*n), seeded(rng)
    noise = rng.choice([0.05, 0.1, 0.2])
    return (f"g = np.random.default_rng({s})\nx = np.sort(g.uniform(0, 10, {n_}))[:, None]   # one feature, as a column\n"
            f"y = np.sin(x).ravel() + g.normal(0, {noise}, {n_})",
            f"`x` holds {n_} sorted inputs in [0, 10] (a column) and `y` noisy values of sin(x) (noise sd {noise})", n_)


def _monotone(rng, n=(10, 15)):
    n_ = rng.randint(*n)
    a, s = rng.uniform(1, 3), rng.choice([0.4, 0.8, 1.2])
    vals = [round(a * (i ** 0.5) + rng.gauss(0, s), 2) for i in range(n_)]
    return vals, n_


def _multilabel(rng, n=(150, 280)):
    n_, d, k, s = rng.randint(*n), rng.randint(5, 8), rng.randint(2, 4), seeded(rng)
    return (f"from sklearn.datasets import make_multilabel_classification\nfrom sklearn.preprocessing import StandardScaler\n"
            f"X, Y = make_multilabel_classification(n_samples={n_}, n_features={d}, n_classes={k}, random_state={s})\n"
            f"X = StandardScaler().fit_transform(X)",
            f"`X` ({n_} × {d}, standardized) and the 0/1 label matrix `Y` ({n_} × {k}: each column is one yes/no "
            f"target, and a row may have several 1s) come from `make_multilabel_classification`", k, n_)


def _multireg(rng):
    n_, d, k, s = rng.randint(80, 200), rng.randint(3, 6), rng.randint(2, 3), seeded(rng)
    return (f"from sklearn.datasets import make_regression\nX, Y = make_regression(n_samples={n_}, n_features={d}, "
            f"n_targets={k}, noise={rng.choice([5, 10, 20])}, random_state={s})",
            f"`X` ({n_} × {d}) and the target matrix `Y` ({n_} × {k}: {k} continuous targets per row) come from "
            f"`make_regression(..., n_targets={k})`", k)


# ================================================================= lite

@template(M, "lite")
def outlier_detector_basics(rng):
    setup, desc, n, d, m = _outliers(rng)
    c = rng.choice([0.02, 0.03, 0.05])
    k = rng.choice([10, 15, 20])
    new = [[round(rng.uniform(-1, 1), 2) for _ in range(d)], [round(rng.choice([-1, 1]) * rng.uniform(4, 6), 2)
                                                               for _ in range(d)]]
    rng.shuffle(new)
    items = [
        ("IsolationForest", f"Fit `IsolationForest(contamination={c}, random_state=0)` (from `sklearn.ensemble`) on `X` "
                            f"and return its labels for the first 8 rows (+1 inlier, -1 outlier).",
         f"from sklearn.ensemble import IsolationForest\n"
         f"result = IsolationForest(contamination={c}, random_state=0).fit(X).predict(X[:8])"),
        ("IsolationForest", f"How many rows of `X` does `IsolationForest(contamination={c}, random_state=0)` flag as "
                            f"outliers (label -1)?",
         f"from sklearn.ensemble import IsolationForest\n"
         f"result = int((IsolationForest(contamination={c}, random_state=0).fit(X).predict(X) == -1).sum())"),
        ("score_samples", "Fit `IsolationForest(random_state=0)` on `X` and return `score_samples(X_new)` (the lower, "
                          "the more abnormal).",
         "from sklearn.ensemble import IsolationForest\nresult = IsolationForest(random_state=0).fit(X).score_samples(X_new)"),
        ("decision_function", "Fit `IsolationForest(random_state=0)` on `X` and return `decision_function(X_new)` "
                              "(negative means outlier).",
         "from sklearn.ensemble import IsolationForest\n"
         "result = IsolationForest(random_state=0).fit(X).decision_function(X_new)"),
        ("LocalOutlierFactor", f"`LocalOutlierFactor` (from `sklearn.neighbors`) compares each point's density with "
                               f"its neighbours'. Return the indices of the rows `LocalOutlierFactor(n_neighbors={k})"
                               f".fit_predict(X)` labels -1.",
         f"from sklearn.neighbors import LocalOutlierFactor\n"
         f"result = np.flatnonzero(LocalOutlierFactor(n_neighbors={k}).fit_predict(X) == -1)",
         {"require": ["LocalOutlierFactor", "fit_predict"]}),
        ("negative_outlier_factor_", f"Fit `LocalOutlierFactor(n_neighbors={k})` on `X` and return the index of the most "
                                     f"outlying row (the smallest `negative_outlier_factor_`).",
         f"from sklearn.neighbors import LocalOutlierFactor\n"
         f"result = int(LocalOutlierFactor(n_neighbors={k}).fit(X).negative_outlier_factor_.argmin())"),
        ("novelty", "To score **new** points, `LocalOutlierFactor` needs `novelty=True`. Fit `LocalOutlierFactor("
                    "novelty=True)` on `X` and return its labels for `X_new`.",
         "from sklearn.neighbors import LocalOutlierFactor\nresult = LocalOutlierFactor(novelty=True).fit(X).predict(X_new)"),
        ("EllipticEnvelope", "`EllipticEnvelope` (from `sklearn.covariance`) fits a robust Gaussian and flags far-away "
                             "points. Fit `EllipticEnvelope(random_state=0)` on `X` and return its labels for `X_new`.",
         "from sklearn.covariance import EllipticEnvelope\nresult = EllipticEnvelope(random_state=0).fit(X).predict(X_new)"),
        ("EllipticEnvelope", f"How many rows of `X` does `EllipticEnvelope(contamination={c}, random_state=0)` flag?",
         f"from sklearn.covariance import EllipticEnvelope\n"
         f"result = int((EllipticEnvelope(contamination={c}, random_state=0).fit(X).predict(X) == -1).sum())"),
    ]
    return catalog(rng, items, setup + f"\nX_new = np.array({lit(new)})", f"{desc}; `X_new` holds 2 new points.")


@template(M, "lite")
def dummy_baselines(rng):
    d = synth_clf(rng, n=(100, 250), d=(3, 6), classes=2, imbalance=rng.choice([0.2, 0.3]))
    q = rng.choice([0.1, 0.25, 0.75, 0.9])
    k = rng.randint(4, 7)
    cst = round(rng.uniform(-50, 50), 1)
    s = seeded(rng)
    setup = (d.setup + f"\nfrom sklearn.datasets import make_regression\n"
             f"X_r, y_r = make_regression(n_samples=80, n_features=3, noise=20, random_state={s})")
    dc, dr = "from sklearn.dummy import DummyClassifier\n", "from sklearn.dummy import DummyRegressor\n"
    items = [
        ("DummyClassifier", "Every model should beat a baseline. Return the training accuracy of "
                            "`DummyClassifier(strategy='most_frequent')` (from `sklearn.dummy`) on `X, y`.",
         dc + "result = DummyClassifier(strategy='most_frequent').fit(X, y).score(X, y)"),
        ("DummyClassifier", f"Return the predictions of `DummyClassifier(strategy='most_frequent')` fitted on `X, y` for "
                            f"the first {k} rows.",
         dc + f"result = DummyClassifier(strategy='most_frequent').fit(X, y).predict(X[:{k}])"),
        ("predict_proba", "Return `predict_proba(X[:3])` of `DummyClassifier(strategy='prior')` fitted on `X, y` (every "
                          "row is the class frequencies).",
         dc + "result = DummyClassifier(strategy='prior').fit(X, y).predict_proba(X[:3])"),
        ("strategy", f"Return the first {k} predictions of `DummyClassifier(strategy='stratified', random_state=0)` "
                     f"fitted on `X, y` (random labels drawn with the class frequencies).",
         dc + f"result = DummyClassifier(strategy='stratified', random_state=0).fit(X, y).predict(X[:{k}])"),
        ("constant", "Return the training accuracy of `DummyClassifier(strategy='constant', constant=1)` on `X, y` "
                     "(it always predicts 1).",
         dc + "result = DummyClassifier(strategy='constant', constant=1).fit(X, y).score(X, y)"),
        ("DummyRegressor", "Fit `DummyRegressor(strategy='mean')` on `X_r, y_r` and return its predictions for the "
                           "first 3 rows.", dr + "result = DummyRegressor(strategy='mean').fit(X_r, y_r).predict(X_r[:3])"),
        ("constant_", "Fit `DummyRegressor(strategy='median')` on `X_r, y_r` and return the value it predicts "
                      "(`constant_`).", dr + "result = DummyRegressor(strategy='median').fit(X_r, y_r).constant_"),
        ("quantile", f"Fit `DummyRegressor(strategy='quantile', quantile={q})` on `X_r, y_r` and return its "
                     f"`constant_`.", dr + f"result = DummyRegressor(strategy='quantile', quantile={q}).fit(X_r, y_r).constant_"),
        ("score", "The median baseline doesn't even reach R² = 0 on its training data. Return the training R² of "
                  "`DummyRegressor(strategy='median')` on `X_r, y_r`.",
         dr + "result = DummyRegressor(strategy='median').fit(X_r, y_r).score(X_r, y_r)"),
        ("constant", f"Return the training R² of `DummyRegressor(strategy='constant', constant={cst})` on `X_r, y_r`.",
         dr + f"result = DummyRegressor(strategy='constant', constant={cst}).fit(X_r, y_r).score(X_r, y_r)"),
    ]
    return catalog(rng, items, setup, f"{d.desc} (class 1 is rarer); `X_r, y_r` is a small regression problem.",
                   tol=1e-9)


@template(M, "lite")
def assorted_models(rng):
    setup, desc, k = _multiclass(rng)
    vals, n = _monotone(rng)
    g_ = rng.choice([0.1, 0.5, 1.0])
    m = rng.choice([20, 50, 100])
    h = rng.choice([8, 16, 32])
    setup += f"\nv = np.array({lit(vals)})   # noisy, but trending upwards"
    items = [
        ("QuadraticDiscriminantAnalysis", "`QuadraticDiscriminantAnalysis` (from `sklearn.discriminant_analysis`) fits "
                                          "one Gaussian per class, each with its own covariance. Return its training "
                                          "accuracy on `X, y`.",
         "from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis\n"
         "result = QuadraticDiscriminantAnalysis().fit(X, y).score(X, y)"),
        ("predict_proba", "Fit `QuadraticDiscriminantAnalysis()` on `X, y` and return `predict_proba(X[:3])`.",
         "from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis\n"
         "result = QuadraticDiscriminantAnalysis().fit(X, y).predict_proba(X[:3])"),
        ("IsotonicRegression", f"`IsotonicRegression` (from `sklearn.isotonic`) fits the best non-decreasing step "
                               f"function. Fit it on `np.arange({n})` (x) and `v` (y) and return the fitted values "
                               f"(`fit_transform`).",
         f"from sklearn.isotonic import IsotonicRegression\nresult = IsotonicRegression().fit_transform(np.arange({n}), v)"),
        ("out_of_bounds", f"Fit `IsotonicRegression(out_of_bounds='clip')` on `np.arange({n})`, `v` and predict x = -5 "
                          f"and x = {n + 5} (outside the training range: clipped to the end values).",
         f"from sklearn.isotonic import IsotonicRegression\n"
         f"result = IsotonicRegression(out_of_bounds='clip').fit(np.arange({n}), v).predict([-5, {n + 5}])"),
        ("RBFSampler", f"`RBFSampler` (from `sklearn.kernel_approximation`) maps data to random Fourier features that "
                       f"approximate an RBF kernel. Return the shape of `RBFSampler(gamma={g_}, n_components={m}, "
                       f"random_state=0).fit_transform(X)`.",
         f"from sklearn.kernel_approximation import RBFSampler\n"
         f"result = RBFSampler(gamma={g_}, n_components={m}, random_state=0).fit_transform(X).shape"),
        ("Nystroem", f"Return the first row of `Nystroem(kernel='rbf', gamma={g_}, n_components={m}, "
                     f"random_state=0).fit_transform(X)` (from `sklearn.kernel_approximation`).",
         f"from sklearn.kernel_approximation import Nystroem\n"
         f"result = Nystroem(kernel='rbf', gamma={g_}, n_components={m}, random_state=0).fit_transform(X)[0]",
         {"tol": 1e-6}),
        ("OneVsRestClassifier", "`OneVsRestClassifier` (from `sklearn.multiclass`) trains one binary model per class. "
                                "Fit `OneVsRestClassifier(SVC())` on `X, y` and return how many estimators it trained "
                                "(`len(estimators_)`).",
         "from sklearn.multiclass import OneVsRestClassifier\nfrom sklearn.svm import SVC\n"
         "result = len(OneVsRestClassifier(SVC()).fit(X, y).estimators_)"),
        ("OneVsOneClassifier", "`OneVsOneClassifier` trains one model per **pair** of classes. Fit "
                               "`OneVsOneClassifier(LinearSVC())` on `X, y` and return `len(estimators_)`.",
         "from sklearn.multiclass import OneVsOneClassifier\nfrom sklearn.svm import LinearSVC\n"
         "result = len(OneVsOneClassifier(LinearSVC()).fit(X, y).estimators_)"),
        ("MLPClassifier", f"Fit a small neural network, `MLPClassifier(hidden_layer_sizes=({h},), early_stopping=True, "
                          f"random_state=0)` (from `sklearn.neural_network`), on standardized `X` (`StandardScaler`) and "
                          f"`y`, and return its training accuracy.",
         "from sklearn.neural_network import MLPClassifier\nfrom sklearn.preprocessing import StandardScaler\n"
         f"Xs = StandardScaler().fit_transform(X)\n"
         f"result = MLPClassifier(hidden_layer_sizes=({h},), early_stopping=True, random_state=0).fit(Xs, y).score(Xs, y)"),
    ]
    return catalog(rng, items, setup, f"{desc}; `v` holds {n} noisy values that trend upwards (indexed 0..{n - 1}).",
                   tol=1e-9)


@template(M, "lite")
def multioutput_basics(rng):
    if rng.random() < 0.5:
        setup, desc, k, n = _multilabel(rng)
        kind = "clf"
    else:
        setup, desc, k = _multireg(rng)
        kind = "reg"
    r = rng.randint(2, 4)
    order = list(range(k))[::-1]
    if kind == "clf":
        moc = f"MultiOutputClassifier({LR})"
        imp = "from sklearn.multioutput import MultiOutputClassifier\n" + LRI
        items = [
            ("MultiOutputClassifier", f"`MultiOutputClassifier` (from `sklearn.multioutput`) fits one classifier per "
                                      f"target column. Fit `{moc}` on `X, Y` and return its predictions for the first "
                                      f"{r} rows (one column per target).",
             imp + f"result = {moc}.fit(X, Y).predict(X[:{r}])"),
            ("estimators_", f"Fit `{moc}` on `X, Y` and return how many classifiers it holds (`len(estimators_)`).",
             imp + f"result = len({moc}.fit(X, Y).estimators_)"),
            ("predict_proba", f"For several targets, `predict_proba` returns a **list** with one (rows × 2) array per "
                              f"target. Fit `{moc}` on `X, Y` and return P(label = 1) of row 0 for every target, as a "
                              f"list.",
             imp + f"probs = {moc}.fit(X, Y).predict_proba(X[:1])\nresult = [float(p[0, 1]) for p in probs]",
             {"tol": 1e-6}),
            ("ClassifierChain", f"`ClassifierChain` (from `sklearn.multioutput`) feeds each target's prediction to the "
                                f"next model. Fit `ClassifierChain({LR}, order={order})` on `X, Y` and return its "
                                f"predictions for the first {r} rows.",
             "from sklearn.multioutput import ClassifierChain\n" + LRI +
             f"result = ClassifierChain({LR}, order={order}).fit(X, Y).predict(X[:{r}])"),
            ("score", f"Fit `{moc}` on `X, Y` and return its `score(X, Y)`: the **subset accuracy** (a row counts as "
                      f"right only if every target is right).", imp + f"result = {moc}.fit(X, Y).score(X, Y)"),
            ("shape", "Return `Y.shape` and the number of rows with at least two 1s, as a tuple.",
             "result = (Y.shape, int((Y.sum(axis=1) >= 2).sum()))", {"require": []}),
        ]
    else:
        mor = "MultiOutputRegressor(Ridge())"
        imp = "from sklearn.multioutput import MultiOutputRegressor\nfrom sklearn.linear_model import Ridge\n"
        items = [
            ("MultiOutputRegressor", f"`MultiOutputRegressor` (from `sklearn.multioutput`) fits one regressor per "
                                     f"target. Fit `{mor}` on `X, Y` and return its predictions for the first {r} rows.",
             imp + f"result = {mor}.fit(X, Y).predict(X[:{r}])"),
            ("estimators_", f"Fit `{mor}` on `X, Y` and return the coefficients of the model for target 0 "
                            f"(`estimators_[0].coef_`).", imp + f"result = {mor}.fit(X, Y).estimators_[0].coef_"),
            ("Ridge", "Some models handle several targets natively. Fit `Ridge()` directly on `X, Y` and return the "
                      "shape of its `coef_` (one row per target).",
             "from sklearn.linear_model import Ridge\nresult = Ridge().fit(X, Y).coef_.shape"),
            ("score", f"Fit `{mor}` on `X, Y` and return its training `score`: the R² averaged over the targets.",
             imp + f"result = {mor}.fit(X, Y).score(X, Y)"),
            ("make_regression", f"Generate a fresh problem with `make_regression(n_samples=50, n_features=3, "
                                f"n_targets={k + 1}, random_state=0)` (from `sklearn.datasets`) and return the shapes "
                                f"of its X and Y as a tuple.",
             f"from sklearn.datasets import make_regression\nX2, Y2 = make_regression(n_samples=50, n_features=3, "
             f"n_targets={k + 1}, random_state=0)\nresult = (X2.shape, Y2.shape)"),
            ("MLPRegressor", "Neural networks predict several targets at once too. Fit `MLPRegressor(hidden_layer_sizes="
                             "(32,), max_iter=1000, random_state=0)` (from `sklearn.neural_network`) on standardized `X` "
                             "and standardized `Y` (two `StandardScaler`s) and return the shape of its predictions.",
             "from sklearn.neural_network import MLPRegressor\nfrom sklearn.preprocessing import StandardScaler\n"
             "Xs, Ys = StandardScaler().fit_transform(X), StandardScaler().fit_transform(Y)\n"
             "result = MLPRegressor(hidden_layer_sizes=(32,), max_iter=1000, random_state=0).fit(Xs, Ys).predict(Xs).shape"),
        ]
    return catalog(rng, items, setup, f"{desc}.", tol=1e-6)


# ================================================================= easy

@template(M, "easy")
def flag_top_anomalies(rng):
    setup, desc, n, d, m = _outliers(rng, n=(180, 300))
    pct = rng.choice([1, 2, 3, 5])
    task = rng.choice(["top", "contamination", "recall", "offset", "levels"])
    iso = "from sklearn.ensemble import IsolationForest\n"
    if task == "top":
        return P(f"{desc}. Flag the top {pct}% anomalies: fit `IsolationForest(random_state=0)` on `X` and return the "
                 f"indices of the `int(np.ceil({pct / 100} * len(X)))` rows with the **lowest** `score_samples`, most "
                 f"abnormal first.", setup,
                 iso + f"s = IsolationForest(random_state=0).fit(X).score_samples(X)\n"
                       f"result = np.argsort(s)[:int(np.ceil({pct / 100} * len(X)))]",
                 require=["IsolationForest", "score_samples"])
    if task == "contamination":
        return P(f"{desc}. `contamination` sets the share of points flagged. Fit `IsolationForest(contamination="
                 f"{pct / 100}, random_state=0)` on `X` and return the sorted indices of the rows it labels -1.", setup,
                 iso + f"pred = IsolationForest(contamination={pct / 100}, random_state=0).fit(X).predict(X)\n"
                       "result = np.flatnonzero(pred == -1)", require=["IsolationForest", "contamination"])
    if task == "recall":
        return P(f"{desc}. Fit `IsolationForest(contamination={m / n:.3f}, random_state=0)` on `X` (the true outlier "
                 f"share). What fraction of the injected outliers does it catch? Return the recall.", setup,
                 iso + f"pred = IsolationForest(contamination={m / n:.3f}, random_state=0).fit(X).predict(X)\n"
                       "result = float((pred[is_outlier == 1] == -1).mean())", require=["IsolationForest"])
    if task == "offset":
        return P(f"{desc}. An `IsolationForest`'s `decision_function` is `score_samples` minus a threshold `offset_`. "
                 f"Fit `IsolationForest(contamination={pct / 100}, random_state=0)` on `X` and return `offset_`; check "
                 f"(not graded) that `decision_function(X) == score_samples(X) - offset_`.", setup,
                 iso + f"result = IsolationForest(contamination={pct / 100}, random_state=0).fit(X).offset_",
                 require=["offset_"])
    cs = sorted(rng.sample([0.01, 0.02, 0.05, 0.1, 0.2], 3))
    return P(f"{desc}. For `contamination` in `{cs}`, fit `IsolationForest(contamination=c, random_state=0)` on `X` and "
             f"record how many rows it flags. Return the list of counts.", setup,
             iso + f"result = [int((IsolationForest(contamination=c, random_state=0).fit(X).predict(X) == -1).sum()) "
                   f"for c in {cs}]", require=["IsolationForest"])


@template(M, "easy")
def lof_neighbourhoods(rng):
    setup, desc, n, d, m = _outliers(rng)
    ks = sorted(rng.sample([5, 10, 20, 35, 50], 3))
    k = rng.choice(ks)
    c = rng.choice([0.02, 0.04, 0.05])
    new = [[round(rng.uniform(-5, 5), 2) for _ in range(d)] for _ in range(3)]
    lof = "from sklearn.neighbors import LocalOutlierFactor\n"
    items = [
        ("n_neighbors", f"The neighbourhood size matters. For `n_neighbors` in `{ks}`, return the number of rows "
                        f"`LocalOutlierFactor(n_neighbors=k).fit_predict(X)` flags (-1).",
         lof + f"result = [int((LocalOutlierFactor(n_neighbors=k).fit_predict(X) == -1).sum()) for k in {ks}]"),
        ("negative_outlier_factor_", f"Fit `LocalOutlierFactor(n_neighbors={k})` on `X` and return the indices of the "
                                     f"5 most outlying rows (smallest `negative_outlier_factor_`), most outlying first.",
         lof + f"result = np.argsort(LocalOutlierFactor(n_neighbors={k}).fit(X).negative_outlier_factor_)[:5]"),
        ("contamination", f"Return the sorted indices of the rows `LocalOutlierFactor(n_neighbors={k}, "
                          f"contamination={c}).fit_predict(X)` flags.",
         lof + f"result = np.flatnonzero(LocalOutlierFactor(n_neighbors={k}, contamination={c}).fit_predict(X) == -1)"),
        ("IsolationForest", f"Do the detectors agree? Return the fraction of rows on which "
                            f"`LocalOutlierFactor(n_neighbors={k}, contamination={c}).fit_predict(X)` and "
                            f"`IsolationForest(contamination={c}, random_state=0).fit(X).predict(X)` give the same "
                            f"label.",
         lof + "from sklearn.ensemble import IsolationForest\n"
               f"a = LocalOutlierFactor(n_neighbors={k}, contamination={c}).fit_predict(X)\n"
               f"b = IsolationForest(contamination={c}, random_state=0).fit(X).predict(X)\nresult = float((a == b).mean())"),
        ("novelty", f"Use LOF as a novelty detector: fit `LocalOutlierFactor(n_neighbors={k}, novelty=True)` on `X` and "
                    f"return `score_samples(X_new)` (larger = more normal).",
         lof + f"result = LocalOutlierFactor(n_neighbors={k}, novelty=True).fit(X).score_samples(X_new)"),
        ("negative_outlier_factor_", f"Fit `LocalOutlierFactor(n_neighbors={k})` on `X`. Return the mean "
                                     f"`negative_outlier_factor_` of the injected outliers and of the normal points, as "
                                     f"a tuple (outliers first).",
         lof + f"s = LocalOutlierFactor(n_neighbors={k}).fit(X).negative_outlier_factor_\n"
               "result = (float(s[is_outlier == 1].mean()), float(s[is_outlier == 0].mean()))"),
    ]
    return catalog(rng, items, setup + f"\nX_new = np.array({lit(new)})", f"{desc}; `X_new` holds 3 new points.",
                   tol=1e-9)


@template(M, "easy")
def beat_the_baseline(rng):
    task = rng.choice(["holdout", "cv", "imbalanced", "regression", "gap"])
    dc = "from sklearn.dummy import DummyClassifier\n"
    if task == "regression":
        t = table(rng, kind="reg", numeric_only=True, n=(100, 220))
        h = t.n * 2 // 3
        return P(f"{t.desc}. Train `Ridge()` and `DummyRegressor(strategy='mean')` on the first {h} rows and return "
                 f"their R² on the remaining rows, as a tuple (model first). A useful model must beat the baseline "
                 f"(whose test R² is usually slightly below 0).", t.setup,
                 "from sklearn.dummy import DummyRegressor\nfrom sklearn.linear_model import Ridge\n"
                 f"a = Ridge().fit(df[:{h}], y[:{h}]).score(df[{h}:], y[{h}:])\n"
                 f"b = DummyRegressor(strategy='mean').fit(df[:{h}], y[:{h}]).score(df[{h}:], y[{h}:])\nresult = (a, b)",
                 require=["DummyRegressor"], tol=1e-6)
    if task == "imbalanced":
        d = synth_clf(rng, n=(200, 350), d=(4, 8), classes=2, imbalance=rng.choice([0.05, 0.1]))
        return P(f"{d.desc}; class 1 is rare. On imbalanced data accuracy flatters: return the training accuracy of "
                 f"`DummyClassifier(strategy='most_frequent')` and the share of class 0 in `y`, as a tuple (they are "
                 f"equal).", d.setup,
                 dc + "result = (DummyClassifier(strategy='most_frequent').fit(X, y).score(X, y), float((y == 0).mean()))",
                 require=["DummyClassifier"], tol=1e-9)
    setup, desc, n = _clf(rng)
    h = n * 2 // 3
    if task == "holdout":
        return P(f"{desc}. Train `{LR}` and `DummyClassifier(strategy='most_frequent')` (from `sklearn.dummy`) on the "
                 f"first {h} rows and return both test accuracies on the rest, as a tuple (model first).", setup,
                 dc + LRI + f"a = {LR}.fit(X[:{h}], y[:{h}]).score(X[{h}:], y[{h}:])\n"
                            f"b = DummyClassifier(strategy='most_frequent').fit(X[:{h}], y[:{h}]).score(X[{h}:], y[{h}:])\n"
                            "result = (a, b)", require=["DummyClassifier"], tol=1e-6)
    if task == "cv":
        return P(f"{desc}. Compare with 5-fold `cross_val_score`: return the mean scores of `{LR}` and of "
                 f"`DummyClassifier(strategy='stratified', random_state=0)` as a tuple (model first).", setup,
                 dc + LRI + "from sklearn.model_selection import cross_val_score\n"
                            f"result = (cross_val_score({LR}, X, y, cv=5).mean(), "
                            "cross_val_score(DummyClassifier(strategy='stratified', random_state=0), X, y, cv=5).mean())",
                 require=["DummyClassifier", "cross_val_score"], tol=1e-6)
    return P(f"{desc}. By how many accuracy points does `{LR}` beat the `DummyClassifier(strategy='most_frequent')` "
             f"baseline on the rows after the first {h} (both trained on the first {h})? Return the difference "
             f"(model − baseline).", setup,
             dc + LRI + f"a = {LR}.fit(X[:{h}], y[:{h}]).score(X[{h}:], y[{h}:])\n"
                        f"b = DummyClassifier(strategy='most_frequent').fit(X[:{h}], y[:{h}]).score(X[{h}:], y[{h}:])\n"
                        "result = a - b", require=["DummyClassifier"], tol=1e-6)


@template(M, "easy")
def one_vs_rest_one_vs_one(rng):
    setup, desc, k = _multiclass(rng)
    mc = "from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier\n"
    sv = "from sklearn.svm import LinearSVC\n"
    items = [
        ("OneVsOneClassifier", "Return `(len(ovr.estimators_), len(ovo.estimators_))` for `ovr = OneVsRestClassifier("
                               "LinearSVC())` and `ovo = OneVsOneClassifier(LinearSVC())` fitted on `X, y` "
                               "(k versus k(k − 1)/2 models).",
         mc + sv + "ovr = OneVsRestClassifier(LinearSVC()).fit(X, y)\novo = OneVsOneClassifier(LinearSVC()).fit(X, y)\n"
                   "result = (len(ovr.estimators_), len(ovo.estimators_))", {"require": ["OneVsOneClassifier",
                                                                                         "OneVsRestClassifier"]}),
        ("decision_function", "Fit `OneVsRestClassifier(LinearSVC())` on `X, y` and return the shape of its "
                              "`decision_function(X)` (one score per class).",
         mc + sv + "result = OneVsRestClassifier(LinearSVC()).fit(X, y).decision_function(X).shape"),
        ("OneVsOneClassifier", "Fit `OneVsRestClassifier(LinearSVC())` and `OneVsOneClassifier(LinearSVC())` on `X, y` "
                               "and return the fraction of rows on which their predictions agree.",
         mc + sv + "a = OneVsRestClassifier(LinearSVC()).fit(X, y).predict(X)\n"
                   "b = OneVsOneClassifier(LinearSVC()).fit(X, y).predict(X)\nresult = float((a == b).mean())"),
        ("OneVsRestClassifier", "Return the training accuracies of `OneVsRestClassifier(SVC())` and "
                                "`OneVsOneClassifier(SVC())` on `X, y` as a tuple.",
         mc + "from sklearn.svm import SVC\nresult = (OneVsRestClassifier(SVC()).fit(X, y).score(X, y), "
              "OneVsOneClassifier(SVC()).fit(X, y).score(X, y))"),
        ("estimators_", f"Fit `OneVsRestClassifier({LR})` on `X, y`. Its estimator number c separates class c from "
                        f"the rest: return the training accuracy of `estimators_[0]` on the binary task `y == 0`.",
         mc + LRI + f"ovr = OneVsRestClassifier({LR}).fit(X, y)\nresult = ovr.estimators_[0].score(X, y == 0)"),
        ("predict_proba", f"Fit `OneVsRestClassifier({LR})` on `X, y` and return `predict_proba(X[:3])` (the per-class "
                          f"probabilities, renormalized to sum to 1).",
         mc + LRI + f"result = OneVsRestClassifier({LR}).fit(X, y).predict_proba(X[:3])", {"tol": 1e-5}),
    ]
    return catalog(rng, items, setup, f"{desc}.", tol=1e-9)


@template(M, "easy")
def isotonic_fits(rng):
    vals, n = _monotone(rng, n=(10, 16))
    task = rng.choice(["levels", "decreasing", "interp", "clip", "monotone", "auto"])
    iso = "from sklearn.isotonic import IsotonicRegression\n"
    setup = f"x = np.arange({n})\nv = np.array({lit(vals)})"
    head = f"`x` is `0..{n - 1}` and `v` holds {n} noisy values that trend upwards."
    if task == "levels":
        return P(f"{head} Isotonic regression produces flat steps. Fit `IsotonicRegression()` on `x, v` and return the "
                 f"number of distinct fitted values (steps).", setup,
                 iso + "result = len(np.unique(IsotonicRegression().fit_transform(x, v)))", require=["IsotonicRegression"])
    if task == "decreasing":
        return P(f"{head} Flip the trend: fit `IsotonicRegression(increasing=False)` on `x` and `v[::-1]` and return "
                 f"the fitted values (a non-increasing sequence).", setup,
                 iso + "result = IsotonicRegression(increasing=False).fit_transform(x, v[::-1])",
                 require=["IsotonicRegression", "increasing"])
    if task == "interp":
        q = [round(rng.uniform(0.5, n - 1.5), 1) for _ in range(3)]
        return P(f"{head} Fit `IsotonicRegression()` on `x, v` and predict at `{q}`: between training points it "
                 f"interpolates linearly.", setup, iso + f"result = IsotonicRegression().fit(x, v).predict({q})",
                 require=["IsotonicRegression"])
    if task == "clip":
        return P(f"{head} Outside the training range `IsotonicRegression` returns NaN by default. Fit it with "
                 f"`out_of_bounds='clip'` and with the default (`'nan'`), predict x = {n + 3} with each, and return "
                 f"`(clipped prediction, np.isnan(default prediction))`.", setup,
                 iso + f"a = IsotonicRegression(out_of_bounds='clip').fit(x, v).predict([{n + 3}])[0]\n"
                       f"b = IsotonicRegression().fit(x, v).predict([{n + 3}])[0]\nresult = (float(a), bool(np.isnan(b)))",
                 require=["out_of_bounds"])
    if task == "monotone":
        return P(f"{head} How much error does forcing monotonicity cost? Return `(sum of squared errors of "
                 f"IsotonicRegression().fit_transform(x, v), sum of squared errors of a LinearRegression on x)` (x as "
                 f"a column for the linear model).", setup,
                 iso + "from sklearn.linear_model import LinearRegression\nf = IsotonicRegression().fit_transform(x, v)\n"
                       "lin = LinearRegression().fit(x[:, None], v).predict(x[:, None])\n"
                       "result = (float(((v - f) ** 2).sum()), float(((v - lin) ** 2).sum()))",
                 require=["IsotonicRegression"])
    return P(f"{head} With `increasing='auto'`, the direction is chosen from the data (Spearman correlation). Fit "
             f"`IsotonicRegression(increasing='auto')` on `x, v` and on `x, -v` and return the two learned directions "
             f"(`increasing_`) as a tuple.", setup,
             iso + "a = IsotonicRegression(increasing='auto').fit(x, v).increasing_\n"
                   "b = IsotonicRegression(increasing='auto').fit(x, -v).increasing_\nresult = (bool(a), bool(b))",
             require=["increasing_"])


@template(M, "easy")
def kernel_approximation(rng):
    n, s = rng.randint(150, 260), seeded(rng)
    shape = rng.choice(["moons", "circles"])
    gen = (f"make_moons(n_samples={n}, noise=0.25, random_state={s})" if shape == "moons" else
           f"make_circles(n_samples={n}, noise=0.1, factor=0.5, random_state={s})")
    setup = f"from sklearn.datasets import make_{shape}\nX, y = {gen}"
    gm = rng.choice([0.5, 1.0, 2.0])
    m = rng.choice([30, 50, 100])
    ka = "from sklearn.kernel_approximation import Nystroem, RBFSampler\n"
    pipe = "from sklearn.pipeline import make_pipeline\n"
    items = [
        ("RBFSampler", f"A linear model can't separate {shape}. Fit `make_pipeline(RBFSampler(gamma={gm}, "
                       f"n_components={m}, random_state=0), {LR})` on `X, y` and return its training accuracy, "
                       f"alongside that of a plain `{LR}`, as a tuple (approximate kernel first).",
         ka + LRI + pipe + f"a = make_pipeline(RBFSampler(gamma={gm}, n_components={m}, random_state=0), {LR})"
                           f".fit(X, y).score(X, y)\nb = {LR}.fit(X, y).score(X, y)\nresult = (a, b)",
         {"require": ["RBFSampler"]}),
        ("Nystroem", f"Return the training accuracy of `make_pipeline(Nystroem(kernel='rbf', gamma={gm}, "
                     f"n_components={m}, random_state=0), {LR})` on `X, y`.",
         ka + LRI + pipe + f"result = make_pipeline(Nystroem(kernel='rbf', gamma={gm}, n_components={m}, "
                           f"random_state=0), {LR}).fit(X, y).score(X, y)"),
        ("SVC", f"Compare with the exact kernel: return the training accuracies of `SVC(kernel='rbf', gamma={gm})` "
                f"and of the `Nystroem(kernel='rbf', gamma={gm}, n_components={m}, random_state=0)` + `LinearSVC()` "
                f"pipeline, as a tuple (exact first).",
         ka + pipe + "from sklearn.svm import SVC, LinearSVC\n"
                     f"a = SVC(kernel='rbf', gamma={gm}).fit(X, y).score(X, y)\n"
                     f"b = make_pipeline(Nystroem(kernel='rbf', gamma={gm}, n_components={m}, random_state=0), "
                     "LinearSVC()).fit(X, y).score(X, y)\nresult = (a, b)", {"require": ["Nystroem", "SVC"]}),
        ("rbf_kernel", f"How good is the approximation? With `Z = RBFSampler(gamma={gm}, n_components={m}, "
                       f"random_state=0).fit_transform(X)`, `Z @ Z.T` approximates `rbf_kernel(X, gamma={gm})` (from "
                       f"`sklearn.metrics.pairwise`). Return the mean absolute difference.",
         ka + "from sklearn.metrics.pairwise import rbf_kernel\n"
              f"Z = RBFSampler(gamma={gm}, n_components={m}, random_state=0).fit_transform(X)\n"
              f"result = float(np.abs(Z @ Z.T - rbf_kernel(X, gamma={gm})).mean())", {"require": ["RBFSampler"]}),
        ("n_components", f"More components, better approximation. For `n_components` in `[10, {m}, 300]`, fit "
                         f"`RBFSampler(gamma={gm}, n_components=c, random_state=0)` + `{LR}` and return the training "
                         f"accuracies as a list.",
         ka + LRI + pipe + f"result = [make_pipeline(RBFSampler(gamma={gm}, n_components=c, random_state=0), {LR})"
                           f".fit(X, y).score(X, y) for c in [10, {m}, 300]]"),
    ]
    return catalog(rng, items, setup, f"`X, y` come from `{gen}` (two classes that a straight line can't split).",
                   tol=1e-6)


@template(M, "easy")
def multi_target_checks(rng):
    task = rng.choice(["per_target", "subset", "raw_r2", "chain_order", "hamming"])
    mo = "from sklearn.multioutput import MultiOutputClassifier\n"
    if task == "raw_r2":
        setup, desc, k = _multireg(rng)
        return P(f"{desc}. Fit `MultiOutputRegressor(Ridge())` on the first 60 rows and return the test R² of each "
                 f"target separately on the rest (`r2_score(..., multioutput='raw_values')`).", setup,
                 "from sklearn.multioutput import MultiOutputRegressor\nfrom sklearn.linear_model import Ridge\n"
                 "from sklearn.metrics import r2_score\nm = MultiOutputRegressor(Ridge()).fit(X[:60], Y[:60])\n"
                 "result = r2_score(Y[60:], m.predict(X[60:]), multioutput='raw_values')",
                 require=["MultiOutputRegressor"], tol=1e-6)
    setup, desc, k, n = _multilabel(rng)
    h = n * 2 // 3
    fit = mo + LRI + f"m = MultiOutputClassifier({LR}).fit(X[:{h}], Y[:{h}])\nP = m.predict(X[{h}:])\n"
    if task == "per_target":
        return P(f"{desc}. Fit `MultiOutputClassifier({LR})` on the first {h} rows and return the test accuracy of "
                 f"every target column on the rest, as an array.", setup, fit + f"result = (P == Y[{h}:]).mean(axis=0)",
                 require=["MultiOutputClassifier"])
    if task == "subset":
        return P(f"{desc}. Fit `MultiOutputClassifier({LR})` on the first {h} rows. On the rest, return `(mean "
                 f"per-label accuracy, subset accuracy)`: the subset accuracy only counts rows whose labels are "
                 f"**all** right (it equals `m.score`).", setup,
                 fit + f"result = (float((P == Y[{h}:]).mean()), float((P == Y[{h}:]).all(axis=1).mean()))",
                 require=["MultiOutputClassifier"])
    if task == "hamming":
        return P(f"{desc}. Fit `MultiOutputClassifier({LR})` on the first {h} rows and return the test "
                 f"`hamming_loss` (from `sklearn.metrics`: the fraction of wrong labels).", setup,
                 fit + "from sklearn.metrics import hamming_loss\n" + f"result = hamming_loss(Y[{h}:], P)",
                 require=["MultiOutputClassifier", "hamming_loss"])
    order = list(range(k))
    rng.shuffle(order)
    return P(f"{desc}. Fit `ClassifierChain({LR}, order={order})` on the first {h} rows and return its subset "
             f"accuracy on the rest (`score`).", setup,
             "from sklearn.multioutput import ClassifierChain\n" + LRI +
             f"result = ClassifierChain({LR}, order={order}).fit(X[:{h}], Y[:{h}]).score(X[{h}:], Y[{h}:])",
             require=["ClassifierChain", "order"])


# ================================================================= medium

def _anomaly_table(rng):
    t = table(rng, numeric_only=True, n=(120, 240))
    m, s = rng.randint(4, 8), seeded(rng)
    extra = (f"\nh = np.random.default_rng({s})\nbad = h.choice(len(df), {m}, replace=False)   # corrupted rows\n"
             f"df.iloc[bad] = df.iloc[bad] * h.choice([-3, 4, 6], ({m}, df.shape[1]))\n"
             f"is_anomaly = np.isin(np.arange(len(df)), bad).astype(int)")
    return t, t.setup + extra, m


@template(M, "medium")
def anomalies_in_table(rng):
    t, setup, m = _anomaly_table(rng)
    c = rng.choice([0.02, 0.03, 0.05])
    k = rng.choice([10, 20])
    sc = "from sklearn.preprocessing import StandardScaler\n"
    iso = "from sklearn.ensemble import IsolationForest\n"
    items = [
        ("IsolationForest", f"Return the index labels of the rows `IsolationForest(contamination={c}, random_state=0)` "
                            f"flags on `df`, sorted.",
         iso + f"pred = IsolationForest(contamination={c}, random_state=0).fit(df).predict(df)\n"
               "result = sorted(df.index[pred == -1])"),
        ("score_samples", "Rank the rows by `IsolationForest(random_state=0)` `score_samples` on `df` and return the "
                          f"fraction of the {m} corrupted rows among the {m} lowest-scoring ones.",
         iso + f"s = IsolationForest(random_state=0).fit(df).score_samples(df)\n"
               f"result = float(is_anomaly[np.argsort(s)[:{m}]].mean())"),
        ("roc_auc_score", f"Which detector ranks the corrupted rows better? Using `roc_auc_score(is_anomaly, ...)` (from "
                          f"`sklearn.metrics`) with **minus** the scores (higher = more abnormal), return the AUCs of "
                          f"`IsolationForest(random_state=0).score_samples` and of `LocalOutlierFactor(n_neighbors={k})`'s "
                          f"`negative_outlier_factor_`, both on standardized `df`, as a tuple.",
         iso + sc + "from sklearn.metrics import roc_auc_score\nfrom sklearn.neighbors import LocalOutlierFactor\n"
               "Z = StandardScaler().fit_transform(df)\na = -IsolationForest(random_state=0).fit(Z).score_samples(Z)\n"
               f"b = -LocalOutlierFactor(n_neighbors={k}).fit(Z).negative_outlier_factor_\n"
               "result = (roc_auc_score(is_anomaly, a), roc_auc_score(is_anomaly, b))"),
        ("EllipticEnvelope", f"Fit `EllipticEnvelope(contamination={c}, random_state=0)` on standardized `df` "
                             f"(`StandardScaler`) and return how many of the flagged rows are truly corrupted.",
         sc + "from sklearn.covariance import EllipticEnvelope\nZ = StandardScaler().fit_transform(df)\n"
              f"pred = EllipticEnvelope(contamination={c}, random_state=0).fit(Z).predict(Z)\n"
              "result = int(is_anomaly[pred == -1].sum())"),
        ("LocalOutlierFactor", f"On standardized `df`, return the index labels flagged by **both** "
                               f"`LocalOutlierFactor(n_neighbors={k}, contamination={c})` and "
                               f"`IsolationForest(contamination={c}, random_state=0)`, sorted.",
         iso + sc + "from sklearn.neighbors import LocalOutlierFactor\nZ = StandardScaler().fit_transform(df)\n"
               f"a = LocalOutlierFactor(n_neighbors={k}, contamination={c}).fit_predict(Z) == -1\n"
               f"b = IsolationForest(contamination={c}, random_state=0).fit(Z).predict(Z) == -1\n"
               "result = sorted(df.index[a & b])"),
        ("score_samples", "Explain the most abnormal row: find the row with the lowest `IsolationForest("
                          "random_state=0)` `score_samples` on `df`, and return the name of its column with the "
                          "largest absolute z-score (standardize `df` with `StandardScaler`).",
         iso + sc + "i = IsolationForest(random_state=0).fit(df).score_samples(df).argmin()\n"
               "z = StandardScaler().fit_transform(df)[i]\nresult = df.columns[np.abs(z).argmax()]"),
    ]
    return catalog(rng, items, setup, f"{t.desc}. {m} rows were corrupted (some of their values multiplied by -3, 4 "
                                      f"or 6); `is_anomaly` marks them.", tol=1e-6)


@template(M, "medium")
def self_training_study(rng):
    setup, desc, n = _clf(rng, n=(200, 360))
    f, hide = _hidden(rng)
    th = rng.choice([0.7, 0.8, 0.9, 0.95])
    st = "from sklearn.semi_supervised import SelfTrainingClassifier\n"
    fit = st + LRI + f"st = SelfTrainingClassifier({LR}, threshold={th}).fit(X, y_partial)\n"
    items = [
        ("SelfTrainingClassifier", f"Self-training labels the unlabelled rows it is confident about and refits. Fit "
                                   f"`SelfTrainingClassifier({LR}, threshold={th})` (from `sklearn.semi_supervised`) "
                                   f"on `X, y_partial` and return its accuracy on the hidden rows (`X[mask]` against "
                                   f"`y[mask]`).", fit + "result = st.score(X[mask], y[mask])"),
        ("SelfTrainingClassifier", f"Does self-training help? Return `(accuracy on the hidden rows of {LR} trained on the "
                                   f"labelled rows only, accuracy of SelfTrainingClassifier({LR}, threshold={th}) "
                                   f"fitted on X, y_partial)`.",
         fit + f"base = {LR}.fit(X[~mask], y[~mask])\nresult = (base.score(X[mask], y[mask]), st.score(X[mask], y[mask]))"),
        ("transduction_", f"Fit `SelfTrainingClassifier({LR}, threshold={th})` on `X, y_partial`. How many hidden rows "
                          f"received a pseudo-label (`transduction_` is not -1 there)?",
         fit + "result = int((st.transduction_[mask] != -1).sum())"),
        ("labeled_iter_", f"Fit `SelfTrainingClassifier({LR}, threshold={th})` on `X, y_partial` and return a dict "
                          f"{{iteration: number of rows labelled in it}} from `labeled_iter_` (0 = labelled from the "
                          f"start; -1 = never labelled).",
         fit + "it, cnt = np.unique(st.labeled_iter_, return_counts=True)\n"
               "result = {int(a): int(b) for a, b in zip(it, cnt)}"),
        ("termination_condition_", f"Fit `SelfTrainingClassifier({LR}, threshold={th})` on `X, y_partial` and return "
                                   f"`(n_iter_, termination_condition_)`.",
         fit + "result = (st.n_iter_, st.termination_condition_)"),
        ("transduction_", f"How good are the pseudo-labels? Fit `SelfTrainingClassifier({LR}, threshold={th})` on "
                          f"`X, y_partial`; among hidden rows that got a pseudo-label, return the fraction equal to the "
                          f"true `y`.",
         fit + "got = mask & (st.transduction_ != -1)\nresult = float((st.transduction_[got] == y[got]).mean())"),
        ("SVC", "Self-training needs probabilities. Fit `SelfTrainingClassifier(SVC(probability=True, random_state=0))` "
                "on `X, y_partial` and return its accuracy on the hidden rows.",
         st + "from sklearn.svm import SVC\n"
              "st = SelfTrainingClassifier(SVC(probability=True, random_state=0)).fit(X, y_partial)\n"
              "result = st.score(X[mask], y[mask])", {"require": ["SelfTrainingClassifier", "SVC"]}),
    ]
    return catalog(rng, items, setup + hide, f"{desc}. About {int(f * 100)}% of the labels are hidden: `y_partial` is "
                                             f"`y` with -1 (unlabelled) where `mask` is True.", tol=1e-6)


@template(M, "medium")
def label_spreading_study(rng):
    setup, desc, n = _clf(rng, classes=rng.choice([2, 3]), n=(150, 300))
    f, hide = _hidden(rng)
    k = rng.choice([5, 7, 10])
    a = rng.choice([0.1, 0.2, 0.5, 0.8])
    ls = "from sklearn.semi_supervised import LabelSpreading\n"
    sc = "Xs = StandardScaler().fit_transform(X)\n"
    sci = "from sklearn.preprocessing import StandardScaler\n"
    items = [
        ("LabelSpreading", f"`LabelSpreading` (from `sklearn.semi_supervised`) spreads labels along a neighbour graph. "
                           f"Fit `LabelSpreading(kernel='knn', n_neighbors={k})` on standardized `X` (`StandardScaler`) "
                           f"and `y_partial`, and return the accuracy of its `transduction_` on the hidden rows.",
         ls + sci + sc + f"m = LabelSpreading(kernel='knn', n_neighbors={k}).fit(Xs, y_partial)\n"
                         "result = float((m.transduction_[mask] == y[mask]).mean())"),
        ("label_distributions_", f"Fit `LabelSpreading(kernel='knn', n_neighbors={k})` on standardized `X` and "
                                 f"`y_partial`. Return the mean, over hidden rows, of the largest value in each row of "
                                 f"`label_distributions_` (how sure it is).",
         ls + sci + sc + f"m = LabelSpreading(kernel='knn', n_neighbors={k}).fit(Xs, y_partial)\n"
                         "result = float(m.label_distributions_[mask].max(axis=1).mean())"),
        ("alpha", f"`alpha` controls how much the given labels may change (0 = never). Fit `LabelSpreading(kernel='knn', "
                  f"n_neighbors={k}, alpha={a}, max_iter=300)` on standardized `X` and `y_partial` and return how many **labelled** "
                  f"rows end up with a different `transduction_` than their given label.",
         ls + sci + sc + f"m = LabelSpreading(kernel='knn', n_neighbors={k}, alpha={a}, max_iter=300).fit(Xs, y_partial)\n"
                         "result = int((m.transduction_[~mask] != y[~mask]).sum())"),
        ("LabelPropagation", f"Compare with `LabelPropagation(kernel='knn', n_neighbors={k})` (its labelled rows stay "
                             f"fixed). Fit both on standardized `X`, `y_partial` and return their accuracies on the "
                             f"hidden rows as a tuple (spreading first).",
         ls + "from sklearn.semi_supervised import LabelPropagation\n" + sci + sc +
         f"a = LabelSpreading(kernel='knn', n_neighbors={k}).fit(Xs, y_partial).transduction_\n"
         f"b = LabelPropagation(kernel='knn', n_neighbors={k}).fit(Xs, y_partial).transduction_\n"
         "result = (float((a[mask] == y[mask]).mean()), float((b[mask] == y[mask]).mean()))"),
        ("score", f"A fitted `LabelSpreading` also predicts new points. Fit `LabelSpreading(kernel='knn', "
                    f"n_neighbors={k})` on the standardized first 80% of the rows (with `y_partial`) and return its "
                    f"accuracy on the last 20% (true `y`).",
         ls + sci + "h = int(0.8 * len(X))\nsc = StandardScaler().fit(X[:h])\n"
                    f"m = LabelSpreading(kernel='knn', n_neighbors={k}).fit(sc.transform(X[:h]), y_partial[:h])\n"
                    "result = m.score(sc.transform(X[h:]), y[h:])"),
    ]
    return catalog(rng, items, setup + hide, f"{desc}. About {int(f * 100)}% of the labels are hidden: `y_partial` is "
                                             f"`y` with -1 where `mask` is True.", tol=1e-6)


@template(M, "medium")
def mlp_training(rng):
    setup, desc, n = _clf(rng, n=(200, 380))
    hs = rng.choice([(16,), (32,), (32, 16), (64, 32)])
    al = rng.choice([1e-4, 1e-3, 1e-2])
    lr_ = rng.choice([1e-3, 3e-3, 1e-2])
    h = n * 3 // 4
    mlp = (f"MLPClassifier(hidden_layer_sizes={hs}, alpha={al}, learning_rate_init={lr_}, max_iter=300, "
           f"early_stopping=True, random_state=0)")
    imp = "from sklearn.neural_network import MLPClassifier\n"
    fit = imp + f"mlp = {mlp}.fit(X[:{h}], y[:{h}])\n"
    items = [
        ("MLPClassifier", f"Train `{mlp}` on the first {h} rows and return its test accuracy on the rest.",
         fit + f"result = mlp.score(X[{h}:], y[{h}:])"),
        ("loss_curve_", f"Train `{mlp}` on the first {h} rows and return `(number of epochs run (n_iter_), final "
                        f"training loss (loss_curve_[-1]))`.", fit + "result = (mlp.n_iter_, mlp.loss_curve_[-1])"),
        ("best_validation_score_", f"With `early_stopping=True`, 10% of the training rows are held out to decide when "
                                   f"to stop. Train `{mlp}` on the first {h} rows and return `best_validation_score_`.",
         fit + "result = mlp.best_validation_score_"),
        ("coefs_", f"Train `{mlp}` on the first {h} rows and return the shapes of its weight matrices (`coefs_`) as a "
                   f"list of tuples.", fit + "result = [w.shape for w in mlp.coefs_]"),
        ("intercepts_", f"How many parameters does `{mlp}` learn? Train it on the first {h} rows and return the total "
                        f"size of `coefs_` and `intercepts_`.",
         fit + "result = sum(w.size for w in mlp.coefs_) + sum(b.size for b in mlp.intercepts_)"),
        ("alpha", f"`alpha` is the L2 penalty. Train `{mlp}` and the same network with `alpha=10.0` on the first {h} "
                  f"rows; return the sums of squared weights (over all `coefs_`) as a tuple (original first).",
         imp + f"a = {mlp}.fit(X[:{h}], y[:{h}])\nb = {mlp.replace(f'alpha={al}', 'alpha=10.0')}.fit(X[:{h}], y[:{h}])\n"
               "result = (sum((w ** 2).sum() for w in a.coefs_), sum((w ** 2).sum() for w in b.coefs_))"),
        ("validation_scores_", f"Train `{mlp}` on the first {h} rows and return the epoch (0-based) with the best "
                               f"validation score (`np.argmax(validation_scores_)`, first on ties).",
         fit + "result = int(np.argmax(mlp.validation_scores_))"),
    ]
    return catalog(rng, items, setup, f"{desc}.", tol=1e-5)


@template(M, "medium")
def mlp_regression(rng):
    t = table(rng, kind="reg", numeric_only=True, n=(120, 240))
    hs = rng.choice([(16,), (32,), (32, 16)])
    h = t.n * 3 // 4
    task = rng.choice(["r2", "vs_linear", "scaled_target", "iters"])
    imp = ("from sklearn.neural_network import MLPRegressor\nfrom sklearn.pipeline import make_pipeline\n"
           "from sklearn.preprocessing import StandardScaler\n")
    mlp = f"MLPRegressor(hidden_layer_sizes={hs}, max_iter=1000, random_state=0)"
    ttr = (f"TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(), {mlp}), "
           f"transformer=StandardScaler())")
    tti = "from sklearn.compose import TransformedTargetRegressor\n"
    head = (f"{t.desc}. Neural networks need scaled inputs **and** a scaled target. The model is `{ttr}` (`Transformed"
            f"TargetRegressor` from `sklearn.compose` scales `y` before fitting and unscales predictions).")
    if task == "r2":
        return P(f"{head} Train it on the first {h} rows and return its R² on the rest.", t.setup,
                 imp + tti + f"m = {ttr}.fit(df[:{h}], y[:{h}])\nresult = m.score(df[{h}:], y[{h}:])",
                 require=["MLPRegressor"], tol=1e-5)
    if task == "vs_linear":
        return P(f"{head} Train it and a `LinearRegression` on the first {h} rows and return both test R² values as a "
                 f"tuple (network first).", t.setup,
                 imp + tti + "from sklearn.linear_model import LinearRegression\n"
                 f"a = {ttr}.fit(df[:{h}], y[:{h}]).score(df[{h}:], y[{h}:])\n"
                 f"b = LinearRegression().fit(df[:{h}], y[:{h}]).score(df[{h}:], y[{h}:])\nresult = (a, b)",
                 require=["MLPRegressor"], tol=1e-5)
    if task == "scaled_target":
        return P(f"{head} Train it on the first {h} rows and return its predictions for the next 3 rows (in the "
                 f"target's original units).", t.setup,
                 imp + tti + f"result = {ttr}.fit(df[:{h}], y[:{h}]).predict(df[{h}:{h + 3}])",
                 require=["MLPRegressor"], tol=1e-5)
    return P(f"{head} Train it on the first {h} rows and return how many epochs the network ran (`n_iter_` of "
             f"`regressor_[-1]`) and its final training loss (`loss_`), as a tuple.", t.setup,
             imp + tti + f"m = {ttr}.fit(df[:{h}], y[:{h}])\nnet = m.regressor_[-1]\nresult = (net.n_iter_, net.loss_)",
             require=["MLPRegressor"], tol=1e-5)


@template(M, "medium")
def gaussian_process_fit(rng):
    setup, desc, n = _sine(rng)
    q = sorted(round(rng.uniform(0, 12), 1) for _ in range(3))
    far = round(rng.uniform(13, 16), 1)
    kern = "ConstantKernel() * RBF(length_scale=1.0) + WhiteKernel()"
    imp = ("from sklearn.gaussian_process import GaussianProcessRegressor\n"
           "from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel\n")
    fit = imp + f"gp = GaussianProcessRegressor(kernel={kern}, normalize_y=True).fit(x, y)\n"
    items = [
        ("GaussianProcessRegressor", f"Fit `GaussianProcessRegressor(kernel={kern}, normalize_y=True)` (from "
                                     f"`sklearn.gaussian_process`; kernels from `sklearn.gaussian_process.kernels`) on "
                                     f"`x, y` and return the predicted means at x = {q} (pass a column).",
         fit + f"result = gp.predict(np.array({q})[:, None])"),
        ("return_std", f"Fit the GP (`kernel={kern}`, `normalize_y=True`) on `x, y` and return `(mean, std)` at "
                       f"x = {q} with `predict(..., return_std=True)`.",
         fit + f"mean, std = gp.predict(np.array({q})[:, None], return_std=True)\nresult = (mean, std)"),
        ("return_std", f"Uncertainty grows away from the data. Fit the GP (`kernel={kern}`, `normalize_y=True`) and "
                       f"return the predicted std at x = 5.0 and at x = {far} as a tuple.",
         fit + f"_, std = gp.predict(np.array([[5.0], [{far}]]), return_std=True)\nresult = (float(std[0]), float(std[1]))"),
        ("kernel_", f"The kernel's hyperparameters are tuned during `fit`. Fit the GP (`kernel={kern}`, "
                    f"`normalize_y=True`) and return the learned RBF length scale and white-noise level as a tuple "
                    f"(`gp.kernel_.k1.k2.length_scale`, `gp.kernel_.k2.noise_level`).",
         fit + "result = (gp.kernel_.k1.k2.length_scale, gp.kernel_.k2.noise_level)", {"tol": 1e-3}),
        ("log_marginal_likelihood_value_", f"Fit the GP (`kernel={kern}`, `normalize_y=True`) and return "
                                           f"`log_marginal_likelihood_value_`.",
         fit + "result = gp.log_marginal_likelihood_value_", {"tol": 1e-4}),
        ("return_std", f"Calibration check: fit the GP (`kernel={kern}`, `normalize_y=True`), predict `x` with "
                       f"`return_std=True`, and return the fraction of training points whose `y` lies within mean ± 2 "
                       f"std.",
         fit + "mean, std = gp.predict(x, return_std=True)\nresult = float((np.abs(y - mean) <= 2 * std).mean())"),
        ("linspace", f"Where is the GP least sure inside the data range? Fit it (`kernel={kern}`, `normalize_y=True`), "
                     f"predict on the grid `np.linspace(0, 10, 101)` (as a column) with `return_std=True`, and return "
                     f"the grid point with the largest std.",
         fit + "grid = np.linspace(0, 10, 101)[:, None]\n_, std = gp.predict(grid, return_std=True)\n"
               "result = float(grid[std.argmax(), 0])", {"require": ["linspace", "return_std"]}),
        ("optimizer", "Freeze the kernel: fit `GaussianProcessRegressor(kernel=RBF(length_scale=1.0), alpha=0.01, "
                      f"optimizer=None)` on `x, y` (no tuning) and return the predicted means at x = {q}.",
         "from sklearn.gaussian_process import GaussianProcessRegressor\nfrom sklearn.gaussian_process.kernels import RBF\n"
         "gp = GaussianProcessRegressor(kernel=RBF(length_scale=1.0), alpha=0.01, optimizer=None).fit(x, y)\n"
         f"result = gp.predict(np.array({q})[:, None])"),
    ]
    return catalog(rng, items, setup, f"{desc}.", tol=1e-4)


@template(M, "medium")
def multi_label_models(rng):
    setup, desc, k, n = _multilabel(rng)
    h = n * 2 // 3
    order = list(range(k))[::-1]
    mo = "from sklearn.multioutput import ClassifierChain, MultiOutputClassifier\n"
    split = f"\nX_train, X_test, Y_train, Y_test = X[:{h}], X[{h}:], Y[:{h}], Y[{h}:]"
    items = [
        ("ClassifierChain", f"Independent models or a chain? Train `MultiOutputClassifier({LR})` and "
                            f"`ClassifierChain({LR}, order={order})` on the training rows and return their test subset "
                            f"accuracies (`score`) as a tuple.",
         mo + LRI + f"a = MultiOutputClassifier({LR}).fit(X_train, Y_train).score(X_test, Y_test)\n"
                    f"b = ClassifierChain({LR}, order={order}).fit(X_train, Y_train).score(X_test, Y_test)\nresult = (a, b)",
         {"require": ["ClassifierChain", "MultiOutputClassifier"]}),
        ("MultiOutputClassifier", f"Train `MultiOutputClassifier(KNeighborsClassifier(n_neighbors=5))` on the training "
                                  f"rows and return the test accuracy of each label column.",
         mo + "from sklearn.neighbors import KNeighborsClassifier\n"
              "m = MultiOutputClassifier(KNeighborsClassifier(n_neighbors=5)).fit(X_train, Y_train)\n"
              "result = (m.predict(X_test) == Y_test).mean(axis=0)"),
        ("KNeighborsClassifier", "Some classifiers accept a label matrix directly. Train `KNeighborsClassifier("
                                 "n_neighbors=5)` on `X_train, Y_train` (no wrapper) and return its test predictions' "
                                 "shape and its test subset accuracy, as a tuple.",
         "from sklearn.neighbors import KNeighborsClassifier\nm = KNeighborsClassifier(n_neighbors=5).fit(X_train, Y_train)\n"
         "result = (m.predict(X_test).shape, m.score(X_test, Y_test))"),
        ("hamming_loss", f"Return the test `hamming_loss` (from `sklearn.metrics`) of `MultiOutputClassifier({LR})` "
                         f"and of `ClassifierChain({LR}, order={order})`, as a tuple.",
         mo + LRI + "from sklearn.metrics import hamming_loss\n"
                    f"a = MultiOutputClassifier({LR}).fit(X_train, Y_train).predict(X_test)\n"
                    f"b = ClassifierChain({LR}, order={order}).fit(X_train, Y_train).predict(X_test)\n"
                    "result = (hamming_loss(Y_test, a), hamming_loss(Y_test, b))"),
        ("predict_proba", f"Train `MultiOutputClassifier({LR})` on the training rows and return a matrix of P(label = 1) "
                          f"for the first 3 test rows (rows × labels), built from the list `predict_proba` returns.",
         mo + LRI + f"m = MultiOutputClassifier({LR}).fit(X_train, Y_train)\n"
                    "result = np.column_stack([p[:, 1] for p in m.predict_proba(X_test[:3])])"),
        ("f1_score", f"Train `MultiOutputClassifier({LR})` on the training rows and return the test "
                     f"`f1_score(..., average='macro')` (from `sklearn.metrics`) over the label columns.",
         mo + LRI + "from sklearn.metrics import f1_score\n"
                    f"P = MultiOutputClassifier({LR}).fit(X_train, Y_train).predict(X_test)\n"
                    "result = f1_score(Y_test, P, average='macro')"),
    ]
    return catalog(rng, items, setup + split, f"{desc}. The first {h} rows are the training split "
                                              f"(`X_train, Y_train`), the rest the test split.", tol=1e-6)


@template(M, "medium")
def qda_and_friends(rng):
    n, s = rng.randint(150, 300), seeded(rng)
    setup = (f"g = np.random.default_rng({s})\nn = {n}\ny = g.integers(0, 2, n)\n"
             "# class 0: a tight round cloud; class 1: a wide, tilted one around the same centre\n"
             "X0 = g.normal(0, 0.6, (n, 2))\nX1 = g.normal(0, 1, (n, 2)) @ np.array([[2.5, 1.2], [0.0, 1.0]])\n"
             "X = np.where(y[:, None] == 0, X0, X1)")
    h = n * 2 // 3
    da = "from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis\n"
    items = [
        ("QuadraticDiscriminantAnalysis", f"The classes share a centre but not a shape, so a linear boundary fails. "
                                          f"Train `QuadraticDiscriminantAnalysis()` and `LinearDiscriminantAnalysis()` "
                                          f"on the first {h} rows and return their test accuracies as a tuple "
                                          f"(QDA first).",
         da + f"a = QuadraticDiscriminantAnalysis().fit(X[:{h}], y[:{h}]).score(X[{h}:], y[{h}:])\n"
              f"b = LinearDiscriminantAnalysis().fit(X[:{h}], y[:{h}]).score(X[{h}:], y[{h}:])\nresult = (a, b)"),
        ("means_", "Fit `QuadraticDiscriminantAnalysis()` on `X, y` and return `(priors_, means_)`.",
         da + "q = QuadraticDiscriminantAnalysis().fit(X, y)\nresult = (q.priors_, q.means_)"),
        ("covariance", "Fit `QuadraticDiscriminantAnalysis(store_covariance=True)` on `X, y` and return the determinant "
                       "of each class's covariance matrix (`covariance_`), as a list.",
         da + "q = QuadraticDiscriminantAnalysis(store_covariance=True).fit(X, y)\n"
              "result = [float(np.linalg.det(c)) for c in q.covariance_]", {"require": ["store_covariance"]}),
        ("reg_param", "`reg_param` shrinks each covariance towards the identity. For `reg_param` in [0.0, 0.5, 0.9], "
                      f"train QDA on the first {h} rows and return the test accuracies as a list.",
         da + f"result = [QuadraticDiscriminantAnalysis(reg_param=r).fit(X[:{h}], y[:{h}]).score(X[{h}:], y[{h}:]) "
              "for r in [0.0, 0.5, 0.9]]"),
        ("GaussianNB", f"`GaussianNB` also fits a Gaussian per class, but with a **diagonal** covariance (no tilt). "
                       f"Train it and QDA on the first {h} rows and return both test accuracies as a tuple "
                       f"(QDA first).",
         da + "from sklearn.naive_bayes import GaussianNB\n"
              f"a = QuadraticDiscriminantAnalysis().fit(X[:{h}], y[:{h}]).score(X[{h}:], y[{h}:])\n"
              f"b = GaussianNB().fit(X[:{h}], y[:{h}]).score(X[{h}:], y[{h}:])\nresult = (a, b)"),
        ("predict_proba", "Fit `QuadraticDiscriminantAnalysis()` on `X, y` and return P(class 1) at the points (0, 0) "
                          "and (3, 1).",
         da + "q = QuadraticDiscriminantAnalysis().fit(X, y)\nresult = q.predict_proba(np.array([[0.0, 0.0], [3.0, 1.0]]))[:, 1]"),
    ]
    return catalog(rng, items, setup, f"`X` holds {n} 2-D points and `y` their class (0 or 1).", tol=1e-6)


@template(M, "medium")
def dummy_strategy_table(rng):
    d = synth_clf(rng, n=(150, 300), d=(3, 6), classes=(2, 3), imbalance=None)
    h = d.n * 2 // 3
    task = rng.choice(["strategies", "regression", "balanced", "proba"])
    dc = "from sklearn.dummy import DummyClassifier\n"
    if task == "strategies":
        return P(f"{d.desc}. Train a `DummyClassifier(strategy=s, random_state=0)` for each `s` in `['most_frequent', "
                 f"'prior', 'stratified', 'uniform']` on the first {h} rows and return a dict {{strategy: test "
                 f"accuracy}} on the rest.", d.setup,
                 dc + "result = {s: DummyClassifier(strategy=s, random_state=0).fit(X[:" + str(h) + "], y[:" + str(h) +
                 "]).score(X[" + str(h) + ":], y[" + str(h) + ":]) for s in ['most_frequent', 'prior', 'stratified', "
                 "'uniform']}", require=["DummyClassifier", "strategy"], tol=1e-9)
    if task == "balanced":
        return P(f"{d.desc}. Accuracy hides a useless baseline; balanced accuracy doesn't. Train `DummyClassifier("
                 f"strategy='most_frequent')` on the first {h} rows and return its test `accuracy_score` and "
                 f"`balanced_accuracy_score` (from `sklearn.metrics`) as a tuple.", d.setup,
                 dc + "from sklearn.metrics import accuracy_score, balanced_accuracy_score\n"
                      f"p = DummyClassifier(strategy='most_frequent').fit(X[:{h}], y[:{h}]).predict(X[{h}:])\n"
                      f"result = (accuracy_score(y[{h}:], p), balanced_accuracy_score(y[{h}:], p))",
                 require=["DummyClassifier"], tol=1e-9)
    if task == "proba":
        return P(f"{d.desc}. Return the log loss (`log_loss` from `sklearn.metrics`) on the rows after the first {h} of "
                 f"`DummyClassifier(strategy='prior')` and of `{LR}`, both trained on the first {h}, as a tuple "
                 f"(baseline first).", d.setup,
                 dc + LRI + "from sklearn.metrics import log_loss\n"
                 f"a = log_loss(y[{h}:], DummyClassifier(strategy='prior').fit(X[:{h}], y[:{h}]).predict_proba(X[{h}:]))\n"
                 f"b = log_loss(y[{h}:], {LR}.fit(X[:{h}], y[:{h}]).predict_proba(X[{h}:]))\nresult = (a, b)",
                 require=["DummyClassifier"], tol=1e-6)
    t = table(rng, kind="reg", numeric_only=True, n=(100, 220))
    h = t.n * 2 // 3
    return P(f"{t.desc}. The mean minimizes squared error, the median absolute error. Train `DummyRegressor(strategy="
             f"'mean')` and `DummyRegressor(strategy='median')` on the first {h} rows and return their test "
             f"`mean_absolute_error` (from `sklearn.metrics`) as a tuple (mean first).", t.setup,
             "from sklearn.dummy import DummyRegressor\nfrom sklearn.metrics import mean_absolute_error\n"
             f"a = mean_absolute_error(y[{h}:], DummyRegressor(strategy='mean').fit(df[:{h}], y[:{h}]).predict(df[{h}:]))\n"
             f"b = mean_absolute_error(y[{h}:], DummyRegressor(strategy='median').fit(df[:{h}], y[:{h}]).predict(df[{h}:]))\n"
             "result = (a, b)", require=["DummyRegressor"], tol=1e-9)


# ================================================================= hard

@template(M, "hard")
def outlier_workflow(rng):
    t = table(rng, kind="reg", numeric_only=True, n=(150, 260))
    m, s = rng.randint(5, 10), seeded(rng)
    h = t.n * 2 // 3
    c = rng.choice([0.03, 0.05, 0.08])
    setup = (t.setup + f"\nh = np.random.default_rng({s})\nbad = h.choice({h}, {m}, replace=False)   # corrupted "
             f"training rows\ny = y.astype(float)\ny[bad] = y[bad] * 5 + 100\n"
             f"X_train, X_test, y_train, y_test = df[:{h}], df[{h}:], y[:{h}], y[{h}:]")
    iso = "from sklearn.ensemble import IsolationForest\n"
    items = [
        ("IsolationForest", f"Some training targets are corrupted. Detect them in feature + target space: fit "
                            f"`IsolationForest(contamination={c}, random_state=0)` on `np.column_stack([X_train, "
                            f"y_train])`, drop the flagged training rows, fit `LinearRegression` on the rest, and return "
                            f"`(test R² without cleaning, test R² with cleaning)`.",
         iso + "from sklearn.linear_model import LinearRegression\nZ = np.column_stack([X_train, y_train])\n"
               f"keep = IsolationForest(contamination={c}, random_state=0).fit(Z).predict(Z) == 1\n"
               "a = LinearRegression().fit(X_train, y_train).score(X_test, y_test)\n"
               "b = LinearRegression().fit(X_train[keep], y_train[keep]).score(X_test, y_test)\nresult = (a, b)"),
        ("LocalOutlierFactor", "This snippet fails: `LocalOutlierFactor(n_neighbors=20).fit(X_train).predict(X_test)` "
                               "(plain LOF only scores its training data). Fix it so LOF can label **new** rows and "
                               "return its labels for `X_test` (standardize with a `StandardScaler` fitted on "
                               "`X_train` first).",
         "from sklearn.neighbors import LocalOutlierFactor\nfrom sklearn.preprocessing import StandardScaler\n"
         "sc = StandardScaler().fit(X_train)\n"
         "result = LocalOutlierFactor(n_neighbors=20, novelty=True).fit(sc.transform(X_train)).predict(sc.transform(X_test))",
         {"require": ["novelty"]}),
        ("HuberRegressor", "Another fix is a robust loss. Return the test R² of `make_pipeline(StandardScaler(), "
                           "HuberRegressor(max_iter=1000))` and of `LinearRegression`, both trained on all training "
                           "rows, as a tuple (Huber first).",
         "from sklearn.linear_model import HuberRegressor, LinearRegression\nfrom sklearn.pipeline import make_pipeline\n"
         "from sklearn.preprocessing import StandardScaler\n"
         "a = make_pipeline(StandardScaler(), HuberRegressor(max_iter=1000)).fit(X_train, y_train).score(X_test, y_test)\n"
         "b = LinearRegression().fit(X_train, y_train).score(X_test, y_test)\nresult = (a, b)"),
        ("contamination", f"Choose `contamination` with the known corrupted rows: for c in `[0.01, 0.03, 0.05, 0.1]` "
                          f"fit `IsolationForest(contamination=c, random_state=0)` on `np.column_stack([X_train, "
                          f"y_train])` and compute the F1 score of its flags (-1 = corrupted) against the true ones "
                          f"(`bad`). Return the best c (first on ties).",
         iso + "from sklearn.metrics import f1_score\nZ = np.column_stack([X_train, y_train])\n"
               "truth = np.isin(np.arange(len(Z)), bad).astype(int)\nscores = []\nfor c in [0.01, 0.03, 0.05, 0.1]:\n"
               "    flag = (IsolationForest(contamination=c, random_state=0).fit(Z).predict(Z) == -1).astype(int)\n"
               "    scores.append(f1_score(truth, flag))\nresult = [0.01, 0.03, 0.05, 0.1][int(np.argmax(scores))]"),
        ("EllipticEnvelope", f"Flag rows by their residuals: fit `LinearRegression` on all training rows, then fit "
                             f"`EllipticEnvelope(contamination={c}, random_state=0)` on the residuals (as a column). "
                             f"Return the sorted positions of the flagged training rows.",
         "from sklearn.covariance import EllipticEnvelope\nfrom sklearn.linear_model import LinearRegression\n"
         "r = (y_train - LinearRegression().fit(X_train, y_train).predict(X_train)).reshape(-1, 1)\n"
         f"result = np.flatnonzero(EllipticEnvelope(contamination={c}, random_state=0).fit(r).predict(r) == -1)"),
    ]
    return catalog(rng, items, setup, f"{t.desc}. The first {h} rows are the training split (`X_train, y_train`), "
                                      f"the rest the test split; the targets of {m} training rows (positions `bad`) "
                                      f"were corrupted.", tol=1e-6)


@template(M, "hard")
def semi_supervised_curves(rng):
    setup, desc, n = _clf(rng, n=(240, 380))
    s = seeded(rng)
    fracs = sorted(rng.sample([0.03, 0.05, 0.1, 0.2, 0.4], 3))
    st = "from sklearn.semi_supervised import SelfTrainingClassifier\n"
    h = int(n * 0.75)
    setup += (f"\n\ndef hide_labels(f):   # each training label survives with probability f (always one per class)\n"
              f"    known = np.random.default_rng({s}).random({h}) < f\n"
              f"    for c in np.unique(y):\n        known[np.flatnonzero(y[:{h}] == c)[0]] = True\n"
              f"    return np.where(known, y[:{h}], -1)")
    head = (f"{desc}. The first {h} rows are for training, the rest for testing. Labels are scarce: "
            f"`hide_labels(f)` returns the training labels with most of them replaced by -1 (each one kept with "
            f"probability `f`).")
    task = rng.choice(["curve", "gain", "spreading"])
    if task == "curve":
        return P(f"{head} For `f` in `{fracs}`, fit `SelfTrainingClassifier({LR})` on `X[:{h}]` and `hide_labels(f)` "
                 f"and record its test accuracy. Return the list.", setup,
                 st + LRI + f"out = []\nfor f in {fracs}:\n    yp = hide_labels(f)\n"
                 f"    out.append(SelfTrainingClassifier({LR}).fit(X[:{h}], yp).score(X[{h}:], y[{h}:]))\nresult = out",
                 require=["SelfTrainingClassifier"], tol=1e-6)
    if task == "gain":
        return P(f"{head} For `f` in `{fracs}`, compute (test accuracy of `SelfTrainingClassifier({LR})` fitted on "
                 f"`X[:{h}]`, `hide_labels(f)`) − (test accuracy of `{LR}` fitted on the labelled rows only). Return "
                 f"the list of gains.", setup,
                 st + LRI + f"out = []\nfor f in {fracs}:\n    yp = hide_labels(f)\n    known = yp != -1\n"
                 f"    a = SelfTrainingClassifier({LR}).fit(X[:{h}], yp).score(X[{h}:], y[{h}:])\n"
                 f"    b = {LR}.fit(X[:{h}][known], yp[known]).score(X[{h}:], y[{h}:])\n"
                 "    out.append(a - b)\nresult = out", require=["SelfTrainingClassifier"], tol=1e-6)
    f = fracs[1]
    return P(f"{head} Use `yp = hide_labels({f})`. Fit `SelfTrainingClassifier({LR})` and `LabelSpreading(kernel='knn', "
             f"n_neighbors=7)` on `X[:{h}], yp` and return `(fraction of the unlabelled training rows on which their "
             f"labels agree, self-training test accuracy, label-spreading test accuracy)` (compare `st.transduction_` "
             f"with `ls.transduction_`; `score` for the test rows).", setup,
             st + LRI + "from sklearn.semi_supervised import LabelSpreading\n"
             f"yp = hide_labels({f})\nst = SelfTrainingClassifier({LR}).fit(X[:{h}], yp)\n"
             f"ls = LabelSpreading(kernel='knn', n_neighbors=7).fit(X[:{h}], yp)\n"
             "agree = float((st.transduction_[yp == -1] == ls.transduction_[yp == -1]).mean())\n"
             f"result = (agree, st.score(X[{h}:], y[{h}:]), ls.score(X[{h}:], y[{h}:]))",
             require=["SelfTrainingClassifier", "LabelSpreading"], tol=1e-6)


@template(M, "hard")
def mlp_forward_by_hand(rng):
    task = rng.choice(["binary", "multi", "regress", "tanh"])
    k = rng.randint(3, 6)
    if task in ("binary", "tanh"):
        setup, desc, n = _clf(rng, n=(150, 300))
        act = "tanh" if task == "tanh" else "relu"
        hs = rng.choice([(8,), (16,), (8, 4)])
        f = "np.tanh(a @ W + b)" if act == "tanh" else "np.maximum(a @ W + b, 0)"
        return P(f"{desc}. Fit `mlp = MLPClassifier(hidden_layer_sizes={hs}, activation='{act}', early_stopping=True, "
                 f"random_state=0)` on `X, y`. Reproduce `mlp.predict_proba(X[:{k}])` **by hand** from `coefs_` and "
                 f"`intercepts_`: each hidden layer is `{act}(a @ W + b)`, and the binary output layer is the sigmoid "
                 f"of `a @ W + b`, giving P(class 1). Return the full (rows × 2) matrix.", setup,
                 "from sklearn.neural_network import MLPClassifier\n"
                 f"mlp = MLPClassifier(hidden_layer_sizes={hs}, activation='{act}', early_stopping=True, "
                 f"random_state=0).fit(X, y)\na = X[:{k}]\n"
                 f"for W, b in zip(mlp.coefs_[:-1], mlp.intercepts_[:-1]):\n    a = {f}\n"
                 "p1 = 1 / (1 + np.exp(-(a @ mlp.coefs_[-1] + mlp.intercepts_[-1]).ravel()))\n"
                 "result = np.column_stack([1 - p1, p1])", require=["coefs_", "intercepts_"], tol=1e-6)
    if task == "multi":
        setup, desc, kk = _multiclass(rng)
        return P(f"{desc}. Fit `mlp = MLPClassifier(hidden_layer_sizes=(10,), early_stopping=True, random_state=0)` on "
                 f"standardized `X` (`Xs = StandardScaler().fit_transform(X)`) and `y`. Reproduce "
                 f"`mlp.predict_proba(Xs[:{k}])` **by hand**: hidden layer `relu(Xs @ W1 + b1)`, output the softmax of "
                 f"`h @ W2 + b2`.", setup,
                 "from sklearn.neural_network import MLPClassifier\nfrom sklearn.preprocessing import StandardScaler\n"
                 "Xs = StandardScaler().fit_transform(X)\n"
                 "mlp = MLPClassifier(hidden_layer_sizes=(10,), early_stopping=True, random_state=0).fit(Xs, y)\n"
                 f"h = np.maximum(Xs[:{k}] @ mlp.coefs_[0] + mlp.intercepts_[0], 0)\n"
                 "z = h @ mlp.coefs_[1] + mlp.intercepts_[1]\ne = np.exp(z - z.max(axis=1, keepdims=True))\n"
                 "result = e / e.sum(axis=1, keepdims=True)", require=["coefs_", "intercepts_"], tol=1e-6)
    setup, desc, kt = _multireg(rng)
    return P(f"{desc}. Fit `mlp = MLPRegressor(hidden_layer_sizes=(16,), max_iter=1000, random_state=0)` on "
             f"standardized `X` and standardized `Y` (`Xs`, `Ys`, each from its own `StandardScaler`). Reproduce "
             f"`mlp.predict(Xs[:{k}])` **by hand**: `relu(Xs @ W1 + b1) @ W2 + b2` (the output layer of a regressor "
             f"is the identity).", setup,
             "from sklearn.neural_network import MLPRegressor\nfrom sklearn.preprocessing import StandardScaler\n"
             "Xs, Ys = StandardScaler().fit_transform(X), StandardScaler().fit_transform(Y)\n"
             "mlp = MLPRegressor(hidden_layer_sizes=(16,), max_iter=1000, random_state=0).fit(Xs, Ys)\n"
             f"h = np.maximum(Xs[:{k}] @ mlp.coefs_[0] + mlp.intercepts_[0], 0)\n"
             "result = h @ mlp.coefs_[1] + mlp.intercepts_[1]", require=["coefs_", "intercepts_"], tol=1e-6)


@template(M, "hard")
def gp_posterior_by_hand(rng):
    setup, desc, n = _sine(rng)
    ls_ = rng.choice([0.5, 1.0, 1.5, 2.0])
    al = rng.choice([0.01, 0.04, 0.1])
    q = sorted(round(rng.uniform(0, 11), 1) for _ in range(3))
    task = rng.choice(["mean", "std", "both"])
    gp = f"GaussianProcessRegressor(kernel=RBF(length_scale={ls_}), alpha={al}, optimizer=None)"
    base = ("k = lambda A, B: np.exp(-((A[:, None, 0] - B[None, :, 0]) ** 2) / (2 * " + str(ls_) + " ** 2))\n"
            f"K = k(x, x) + {al} * np.eye(len(x))\nq = np.array({q})[:, None]\nKs = k(q, x)\n")
    head = (f"{desc}. `{gp}` (no hyperparameter tuning, no target normalization) has a closed-form posterior. With "
            f"the RBF kernel k(a, b) = exp(−(a − b)² / (2·{ls_}²)) and K = k(x, x) + {al}·I, the mean at new points q "
            f"is k(q, x) K⁻¹ y and the variance is 1 − diag(k(q, x) K⁻¹ k(x, q)).")
    if task == "mean":
        return P(f"{head} Compute the posterior **mean** at q = {q} by hand (use `np.linalg.solve`); it matches "
                 f"`{gp}.fit(x, y).predict(...)`.", setup, base + "result = Ks @ np.linalg.solve(K, y)", tol=1e-6)
    if task == "std":
        return P(f"{head} Compute the posterior **standard deviation** at q = {q} by hand; it matches "
                 f"`predict(..., return_std=True)[1]`.", setup,
                 base + "var = 1 - np.einsum('ij,ji->i', Ks, np.linalg.solve(K, Ks.T))\nresult = np.sqrt(np.maximum(var, 0))",
                 tol=1e-6)
    return P(f"{head} Compute the mean at q = {q} by hand, fit `{gp}` on `x, y`, and return `(whether the two "
             f"agree within 1e-8, your mean)`.", setup,
             "from sklearn.gaussian_process import GaussianProcessRegressor\nfrom sklearn.gaussian_process.kernels import RBF\n"
             + base + f"mine = Ks @ np.linalg.solve(K, y)\ntheirs = {gp}.fit(x, y).predict(q)\n"
                      "result = (float(np.abs(mine - theirs).max()) < 1e-8, mine)",
             require=["GaussianProcessRegressor"], tol=1e-6)


@template(M, "hard")
def isotonic_calibration(rng):
    d = synth_clf(rng, n=(300, 500), d=(4, 8), classes=2)
    task = rng.choice(["calibrate", "brier", "pava", "monotone_feature"])
    a, b = int(d.n * 0.5), int(d.n * 0.75)
    head = (f"{d.desc}. Rows `[:{a}]` train a model, rows `[{a}:{b}]` calibrate it, rows `[{b}:]` test it. The model "
            f"is `GaussianNB()` (its probabilities are often over-confident).")
    fit = (f"from sklearn.isotonic import IsotonicRegression\nfrom sklearn.naive_bayes import GaussianNB\n"
           f"nb = GaussianNB().fit(X[:{a}], y[:{a}])\n"
           f"iso = IsotonicRegression(out_of_bounds='clip').fit(nb.predict_proba(X[{a}:{b}])[:, 1], y[{a}:{b}])\n"
           f"p_raw = nb.predict_proba(X[{b}:])[:, 1]\np_cal = iso.predict(p_raw)\n")
    if task == "calibrate":
        return P(f"{head} Calibrate with isotonic regression: fit `IsotonicRegression(out_of_bounds='clip')` mapping the "
                 f"model's P(class 1) on the calibration rows to their labels, then apply it to the test rows' "
                 f"probabilities. Return the first 5 calibrated test probabilities.", d.setup, fit + "result = p_cal[:5]",
                 require=["IsotonicRegression", "out_of_bounds"], tol=1e-9)
    if task == "brier":
        return P(f"{head} Calibrate with `IsotonicRegression(out_of_bounds='clip')` fitted on the calibration rows "
                 f"(P(class 1) → label) and return the test `brier_score_loss` (from `sklearn.metrics`) before and "
                 f"after, as a tuple.", d.setup,
                 fit + f"from sklearn.metrics import brier_score_loss\n"
                       f"result = (brier_score_loss(y[{b}:], p_raw), brier_score_loss(y[{b}:], p_cal))",
                 require=["IsotonicRegression"], tol=1e-9)
    if task == "pava":
        return P(f"{head} Isotonic regression is the pool-adjacent-violators algorithm. Sort the calibration rows by "
                 f"the model's P(class 1) (stable sort) and run PAVA **by hand** on their labels: keep a stack of "
                 f"blocks (mean, size); push each label as a new block, then, while the block below has a mean ≥ the "
                 f"top block's, pool the two (size-weighted mean). Return the fitted values in sorted order (the same "
                 f"values `IsotonicRegression().fit_transform` gives on the sorted data).", d.setup,
                 f"from sklearn.naive_bayes import GaussianNB\nnb = GaussianNB().fit(X[:{a}], y[:{a}])\n"
                 f"s = nb.predict_proba(X[{a}:{b}])[:, 1]\nt = y[{a}:{b}][np.argsort(s, kind='stable')].astype(float)\n"
                 "blocks = []   # [mean, size]\nfor v in t:\n    blocks.append([v, 1])\n"
                 "    while len(blocks) > 1 and blocks[-2][0] >= blocks[-1][0]:\n        m2, n2 = blocks.pop()\n"
                 "        m1, n1 = blocks.pop()\n        blocks.append([(m1 * n1 + m2 * n2) / (n1 + n2), n1 + n2])\n"
                 "result = np.concatenate([[m] * k for m, k in blocks])", require=["GaussianNB"], tol=1e-9)
    j = rng.randrange(4)
    return P(f"{d.desc}. Isotonic regression also makes a monotone 1-feature model. Fit `IsotonicRegression("
             f"out_of_bounds='clip', increasing='auto')` on feature {j} of the first {a} rows (x) and their labels, and "
             f"return the test accuracy on the rest of thresholding its output at 0.5 (predict 1 when ≥ 0.5).", d.setup,
             f"from sklearn.isotonic import IsotonicRegression\n"
             f"iso = IsotonicRegression(out_of_bounds='clip', increasing='auto').fit(X[:{a}, {j}], y[:{a}])\n"
             f"result = float(((iso.predict(X[{a}:, {j}]) >= 0.5).astype(int) == y[{a}:]).mean())",
             require=["IsotonicRegression"], tol=1e-9)


@template(M, "hard")
def ovr_ovo_by_hand(rng):
    setup, desc, k = _multiclass(rng)
    task = rng.choice(["ovr", "ovo_votes", "ovr_proba"])
    mc = "from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier\n"
    if task == "ovr":
        return P(f"{desc}. Fit `ovr = OneVsRestClassifier({LR})` on `X, y`. Rebuild its predictions **by hand**: stack "
                 f"`decision_function(X)` of each of `ovr.estimators_` as columns and take the argmax (mapped through "
                 f"`ovr.classes_`). Return the fraction of rows where your labels equal `ovr.predict(X)`, together with "
                 f"your labels for the first 10 rows: `(agreement, labels)`.", setup,
                 mc + LRI + f"ovr = OneVsRestClassifier({LR}).fit(X, y)\n"
                 "S = np.column_stack([e.decision_function(X) for e in ovr.estimators_])\nmine = ovr.classes_[S.argmax(axis=1)]\n"
                 "result = (float((mine == ovr.predict(X)).mean()), mine[:10])",
                 require=["OneVsRestClassifier", "estimators_"], tol=1e-9)
    if task == "ovo_votes":
        return P(f"{desc}. Fit `ovo = OneVsOneClassifier(LinearSVC())` on `X, y`. Its estimators come in pair order "
                 f"(0, 1), (0, 2), …, (1, 2), …; the estimator for pair (i, j) predicts 1 for class j and 0 for class i. "
                 f"Count the votes **by hand**: return the (rows × classes) vote matrix for the first 6 rows "
                 f"(`predict` of each estimator, as ints).", setup,
                 mc + "from sklearn.svm import LinearSVC\novo = OneVsOneClassifier(LinearSVC()).fit(X, y)\n"
                 "k = len(ovo.classes_)\nvotes = np.zeros((6, k), dtype=int)\npairs = [(i, j) for i in range(k) for j "
                 "in range(i + 1, k)]\nfor (i, j), est in zip(pairs, ovo.estimators_):\n    p = est.predict(X[:6])\n"
                 "    votes[p == 0, i] += 1\n    votes[p == 1, j] += 1\nresult = votes",
                 require=["OneVsOneClassifier", "estimators_"])
    return P(f"{desc}. `OneVsRestClassifier({LR}).predict_proba` takes each binary model's P(positive) and divides "
             f"every row by its sum. Fit it on `X, y` and rebuild `predict_proba(X[:5])` **by hand** from "
             f"`estimators_`.", setup,
             mc + LRI + f"ovr = OneVsRestClassifier({LR}).fit(X, y)\n"
             "P = np.column_stack([e.predict_proba(X[:5])[:, 1] for e in ovr.estimators_])\n"
             "result = P / P.sum(axis=1, keepdims=True)", require=["estimators_"], tol=1e-9)


@template(M, "hard")
def baseline_audit(rng):
    d = synth_clf(rng, n=(250, 400), d=(4, 8), classes=2, imbalance=rng.choice([0.05, 0.08, 0.12]), sep=0.6)
    h = d.n * 2 // 3
    dc = "from sklearn.dummy import DummyClassifier\n"
    items = [
        ("DummyClassifier", f"A colleague reports 'accuracy above 90%!'. Train `{LR}` and `DummyClassifier(strategy="
                            f"'most_frequent')` on the first {h} rows and return the test accuracy and test "
                            f"`balanced_accuracy_score` of each, as `((model acc, model bal), (dummy acc, dummy bal))`.",
         dc + LRI + "from sklearn.metrics import accuracy_score, balanced_accuracy_score\n"
                    f"pm = {LR}.fit(X[:{h}], y[:{h}]).predict(X[{h}:])\n"
                    f"pd_ = DummyClassifier(strategy='most_frequent').fit(X[:{h}], y[:{h}]).predict(X[{h}:])\n"
                    f"result = ((accuracy_score(y[{h}:], pm), balanced_accuracy_score(y[{h}:], pm)), "
                    f"(accuracy_score(y[{h}:], pd_), balanced_accuracy_score(y[{h}:], pd_)))"),
        ("class_weight", f"Fix the rare class being ignored: train `LogisticRegression(class_weight='balanced', "
                         f"max_iter=1000)` and `{LR}` on the first {h} rows; return the test recall of class 1 "
                         f"(`recall_score`) of each as a tuple (balanced first).",
         LRI + "from sklearn.metrics import recall_score\n"
               f"a = LogisticRegression(class_weight='balanced', max_iter=1000).fit(X[:{h}], y[:{h}]).predict(X[{h}:])\n"
               f"b = {LR}.fit(X[:{h}], y[:{h}]).predict(X[{h}:])\n"
               f"result = (recall_score(y[{h}:], a), recall_score(y[{h}:], b))"),
        ("roc_auc_score", f"Ranking metrics expose baselines too. Return the test `roc_auc_score` of `{LR}` and of "
                          f"`DummyClassifier(strategy='prior')` (both trained on the first {h} rows, scored with "
                          f"`predict_proba[:, 1]`), as a tuple.",
         dc + LRI + "from sklearn.metrics import roc_auc_score\n"
                    f"a = roc_auc_score(y[{h}:], {LR}.fit(X[:{h}], y[:{h}]).predict_proba(X[{h}:])[:, 1])\n"
                    f"b = roc_auc_score(y[{h}:], DummyClassifier(strategy='prior').fit(X[:{h}], y[:{h}])"
                    f".predict_proba(X[{h}:])[:, 1])\nresult = (a, b)"),
        ("cross_val_score", f"Use stratified 5-fold CV (`StratifiedKFold(n_splits=5, shuffle=True, random_state=0)`) "
                            f"with `scoring='f1'` and return the mean F1 of `{LR}` and of `DummyClassifier("
                            f"strategy='stratified', random_state=0)` as a tuple.",
         dc + LRI + "from sklearn.model_selection import StratifiedKFold, cross_val_score\n"
                    "cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)\n"
                    f"result = (cross_val_score({LR}, X, y, cv=cv, scoring='f1').mean(), cross_val_score(DummyClassifier("
                    "strategy='stratified', random_state=0), X, y, cv=cv, scoring='f1').mean())"),
    ]
    return catalog(rng, items, d.setup, f"{d.desc}; class 1 is rare.", tol=1e-6)


@template(M, "hard")
def multilabel_and_tuning(rng):
    setup, desc, k, n = _multilabel(rng, n=(180, 300))
    h = n * 2 // 3
    s = seeded(rng)
    kb = rng.choice([5, 10, 20])
    gammas = sorted(rng.sample([0.01, 0.05, 0.1, 0.5, 1.0], 3))
    setup += (f"\ny = Y[:, 0]   # the first label alone, as a binary target\n"
              f"X_train, X_test, Y_train, Y_test = X[:{h}], X[{h}:], Y[:{h}], Y[{h}:]")
    items = [
        ("ClassifierChain", f"A chain's order matters, so average several. Fit `ClassifierChain({LR}, order='random', "
                            f"random_state=r)` for r = 0, 1, 2 on the training rows, average their `predict_proba` on "
                            f"the test rows, threshold at 0.5, and return the test subset accuracy of the ensemble.",
         "from sklearn.multioutput import ClassifierChain\n" + LRI +
         f"P = np.mean([ClassifierChain({LR}, order='random', random_state=r).fit(X_train, Y_train)"
         ".predict_proba(X_test) for r in range(3)], axis=0)\n"
         "result = float(((P >= 0.5).astype(int) == Y_test).all(axis=1).mean())"),
        ("OneVsRestClassifier", f"`OneVsRestClassifier` also accepts a 0/1 label matrix (one model per label). Train "
                                f"`OneVsRestClassifier({LR})` and `MultiOutputClassifier({LR})` on the training rows and "
                                f"return the fraction of test label entries on which they agree.",
         "from sklearn.multiclass import OneVsRestClassifier\nfrom sklearn.multioutput import MultiOutputClassifier\n" + LRI +
         f"a = OneVsRestClassifier({LR}).fit(X_train, Y_train).predict(X_test)\n"
         f"b = MultiOutputClassifier({LR}).fit(X_train, Y_train).predict(X_test)\nresult = float((a == b).mean())"),
        ("MLPClassifier", "Neural networks handle several labels natively (one sigmoid output per label). Train "
                          "`MLPClassifier(hidden_layer_sizes=(32,), early_stopping=True, random_state=0)` on `X_train, "
                          "Y_train` and return its test subset accuracy (`score`).",
         "from sklearn.neural_network import MLPClassifier\n"
         "result = MLPClassifier(hidden_layer_sizes=(32,), early_stopping=True, random_state=0)"
         ".fit(X_train, Y_train).score(X_test, Y_test)"),
        ("GridSearchCV", f"Tune an approximate kernel: grid-search `make_pipeline(RBFSampler(n_components=100, "
                         f"random_state=0), {LR})` over `{{'rbfsampler__gamma': {gammas}}}` with `cv=3` on `X_train` and "
                         f"`y[:{h}]`, and return `(best gamma, test accuracy of the refitted best pipeline on y[{h}:])`.",
         "from sklearn.kernel_approximation import RBFSampler\nfrom sklearn.model_selection import GridSearchCV\n"
         "from sklearn.pipeline import make_pipeline\n" + LRI +
         f"gs = GridSearchCV(make_pipeline(RBFSampler(n_components=100, random_state=0), {LR}), "
         f"{{'rbfsampler__gamma': {gammas}}}, cv=3).fit(X_train, y[:{h}])\n"
         f"result = (gs.best_params_['rbfsampler__gamma'], gs.score(X_test, y[{h}:]))"),
        ("k_best", f"Self-training can add a fixed number of pseudo-labels per round instead of using a threshold. Hide "
                   f"80% of the training labels of `y` (`hide = np.random.default_rng({s}).random({h}) < 0.8`, except the "
                   f"first training row of each class; -1 there), fit `SelfTrainingClassifier({LR}, criterion='k_best', k_best={kb}, max_iter=5)` and return "
                   f"`(n_iter_, number of training rows still unlabelled in transduction_, test accuracy on y[{h}:])`.",
         "from sklearn.semi_supervised import SelfTrainingClassifier\n" + LRI +
         f"hide = np.random.default_rng({s}).random({h}) < 0.8\nfor c in (0, 1):\n"
         f"    hide[np.flatnonzero(y[:{h}] == c)[0]] = False\nyp = np.where(hide, -1, y[:{h}])\n"
         f"st = SelfTrainingClassifier({LR}, criterion='k_best', k_best={kb}, max_iter=5).fit(X_train, yp)\n"
         f"result = (st.n_iter_, int((st.transduction_ == -1).sum()), st.score(X_test, y[{h}:]))",
         {"require": ["SelfTrainingClassifier", "k_best"]}),
        ("QuadraticDiscriminantAnalysis", f"Pick QDA's shrinkage by cross-validation: `GridSearchCV(QuadraticDiscriminantAnalysis(), "
                      f"{{'reg_param': [0.0, 0.1, 0.5, 0.9]}}, cv=5)` on `X_train, y[:{h}]`. Return `best_params_`.",
         "from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis\n"
         "from sklearn.model_selection import GridSearchCV\n"
         "gs = GridSearchCV(QuadraticDiscriminantAnalysis(), {'reg_param': [0.0, 0.1, 0.5, 0.9]}, cv=5)"
         f".fit(X_train, y[:{h}])\nresult = gs.best_params_"),
        ("DummyClassifier", "Multi-label baselines: `DummyClassifier(strategy='most_frequent')` also takes a label "
                            "matrix (it predicts each label's most frequent value). Return its test subset accuracy and "
                            f"that of `MultiOutputClassifier({LR})`, both trained on the training rows, as a tuple "
                            "(baseline first).",
         "from sklearn.dummy import DummyClassifier\nfrom sklearn.multioutput import MultiOutputClassifier\n" + LRI +
         "a = DummyClassifier(strategy='most_frequent').fit(X_train, Y_train).score(X_test, Y_test)\n"
         f"b = MultiOutputClassifier({LR}).fit(X_train, Y_train).score(X_test, Y_test)\nresult = (a, b)"),
    ]
    return catalog(rng, items, setup, f"{desc}. The first {h} rows are the training split (`X_train, Y_train`), the "
                                      f"rest the test split; `y = Y[:, 0]` is the first label alone.", tol=1e-6)


# ================================================================= challenge

def _iso_setup(s, n, inc, report, weighted, q, prefix="g"):
    w = f"{prefix}.integers(1, 4, {n}).astype(float)" if weighted else f"np.ones({n})"
    sign = "" if inc else "-"
    return (f"{prefix} = np.random.default_rng({s})\nx = np.sort({prefix}.choice(10 * {n}, {n}, replace=False)).astype(float)\n"
            f"y = {sign}np.sqrt(x) + {prefix}.normal(0, 3, {n})\nw = {w}\n"
            f"queries = {prefix}.uniform(-5, 10 * {n} + 5, {q})\nincreasing = {inc}\nreport = {report!r}")


@template(M, "challenge")
def pava_isotonic(rng):
    s = seeded(rng)
    report = rng.choice(["fitted", "sse", "queries"])
    inc = rng.choice([True, False])
    weighted = rng.random() < 0.5
    setup = _iso_setup(s, rng.randint(7, 10), inc, report, weighted, 4)
    tests = [_iso_setup(s + i, 200000, inc, report, weighted, 50000) for i in (1, 2)]
    sol = ("v = y if increasing else -y\nmeans, wts, sizes = [], [], []\n"
           "for vi, wi in zip(v.tolist(), w.tolist()):\n    m, ww, k = vi, wi, 1\n"
           "    while means and means[-1] >= m:\n        m = (means[-1] * wts[-1] + m * ww) / (wts[-1] + ww)\n"
           "        ww += wts.pop()\n        k += sizes.pop()\n        means.pop()\n"
           "    means.append(m)\n    wts.append(ww)\n    sizes.append(k)\n"
           "fit = np.repeat(means, sizes)\nfit = fit if increasing else -fit\n")
    sol += {"fitted": "result = fit", "sse": "result = float((w * (y - fit) ** 2).sum())",
            "queries": "result = np.interp(queries, x, fit)"}[report]
    want = {"fitted": "the fitted values at `x` (what `fit_transform` returns), as a float array",
            "sse": "the weighted sum of squared errors Σ w·(y − fit)²",
            "queries": "the model's predictions at `queries` with `out_of_bounds='clip'` (linear interpolation between "
                       "fitted points, end values outside the range), as a float array"}[report]
    return challenge(
        "Isotonic regression in linear time",
        "Fit `IsotonicRegression(increasing=increasing)` to the points `(x[i], y[i])` with sample weights `w[i]` (`x` "
        "is sorted and distinct): the best monotone (non-decreasing if `increasing` is True, else non-increasing) fit "
        f"in weighted least squares. Return {want}. (`increasing` = {inc}, `report` = {report!r}.)",
        setup, sol, tests, "n ≤ 2·10^5 points, ≤ 5·10^4 queries",
        speed="O(n) with the pool-adjacent-violators algorithm (a stack of blocks)", tol=1e-6)


def _lof_setup(s, n, k, report, prefix="g"):
    return (f"{prefix} = np.random.default_rng({s})\n"
            f"x = np.concatenate([{prefix}.normal(0, 1, {n}), {prefix}.normal(8, 0.5, {n} // 3), "
            f"{prefix}.uniform(-30, 30, {n} // 20 + 1)])\n{prefix}.shuffle(x)\nk = {k}\nreport = {report!r}")


@template(M, "challenge")
def lof_1d(rng):
    s = seeded(rng)
    report = rng.choice(["scores", "count", "kdist"])
    k = rng.randint(2, 4)
    setup = _lof_setup(s, rng.randint(9, 14), k, report)
    tests = [_lof_setup(s + i, 150000, 20, report) for i in (1, 2)]
    sol = ("n = len(x)\norder = np.argsort(x)\nxs = x[order]\npad = np.concatenate([np.full(k, np.inf), xs, "
           "np.full(k, np.inf)])\nidx = np.arange(n)[:, None] + np.r_[np.arange(k), np.arange(k + 1, 2 * k + 1)][None, :]\n"
           "cand = np.abs(pad[idx] - xs[:, None])\nsel = np.argsort(cand, axis=1, kind='stable')[:, :k]\n"
           "dist = np.take_along_axis(cand, sel, axis=1)\nnb = np.take_along_axis(idx, sel, axis=1) - k   # sorted positions\n"
           "kd = dist[:, -1]\nreach = np.maximum(dist, kd[nb])\nlrd = 1 / (reach.mean(axis=1) + 1e-10)\n"
           "lof = (lrd[nb] / lrd[:, None]).mean(axis=1)\n"
           "back = np.empty(n, dtype=int)\nback[order] = np.arange(n)\n")
    sol += {"scores": "result = -lof[back]", "kdist": "result = kd[back]",
            "count": "result = int((lof > 1.5).sum())"}[report]
    want = {"scores": "`negative_outlier_factor_` (minus the LOF of every point, in the order of `x`)",
            "kdist": "every point's k-distance (the distance to its k-th nearest other point), in the order of `x`",
            "count": "how many points have LOF > 1.5 (the points `fit_predict` labels -1 with the default "
                     "`contamination='auto'`, whose threshold on `negative_outlier_factor_` is -1.5)"}[report]
    return challenge(
        "Local outlier factor on a line",
        "`x` holds distinct 1-D readings. `LocalOutlierFactor(n_neighbors=k).fit(x[:, None])` computes, for every "
        "point p: its k nearest other points N(p) and the k-distance kd(p) (distance to the k-th of them); the "
        "reachability distance reach(p, o) = max(kd(o), |p − o|); the local reachability density lrd(p) = 1 / (mean "
        "of reach(p, o) over o in N(p) + 1e-10); and LOF(p) = mean of lrd(o) / lrd(p) over o in N(p). Return "
        f"{want}. (`k` = {k}, `report` = {report!r}.)",
        setup, sol, tests, "len(x) ≤ 2·10^5, k ≤ 20",
        speed="O(n log n + n·k): sort once; the k nearest neighbours of a point lie within k positions of it",
        tol=1e-6)


def _run_setup(s, n, report, prefix="g"):
    if report == "median":
        data = f"y = {prefix}.normal(50, 15, {n}).round(2)"
    else:
        data = f"y = {prefix}.integers(0, 6, {n})"
    return f"{prefix} = np.random.default_rng({s})\n{data}\nreport = {report!r}"


@template(M, "challenge")
def running_baseline(rng):
    s = seeded(rng)
    report = rng.choice(["median", "most_frequent"])
    setup = _run_setup(s, rng.randint(7, 11), report)
    tests = [_run_setup(s + i, 200000, report) for i in (1, 2)]
    if report == "median":
        sol = ("import heapq\nlo, hi, out = [], [], []   # lo: max-heap (negated) of the smaller half, hi: min-heap\n"
               "for v in y.tolist():\n    if lo and v > -lo[0]:\n        heapq.heappush(hi, v)\n    else:\n"
               "        heapq.heappush(lo, -v)\n    if len(lo) > len(hi) + 1:\n        heapq.heappush(hi, -heapq.heappop(lo))\n"
               "    elif len(hi) > len(lo):\n        heapq.heappush(lo, -heapq.heappop(hi))\n"
               "    out.append(-lo[0] if len(lo) > len(hi) else (-lo[0] + hi[0]) / 2)\nresult = np.array(out)")
        what = ("`DummyRegressor(strategy='median')` on `y[:t]` and records its `constant_` (the median; the mean of "
                "the two middle values for even t)")
        ans = " (a float array of length `len(y)`)"
    else:
        sol = ("cnt = np.zeros(int(y.max()) + 1, dtype=int)\nbest, out = -1, []\nfor v in y.tolist():\n    cnt[v] += 1\n"
               "    if best < 0 or cnt[v] > cnt[best] or (cnt[v] == cnt[best] and v < best):\n        best = v\n"
               "    out.append(best)\nresult = np.array(out)")
        what = ("`DummyClassifier(strategy='most_frequent')` on `y[:t]` and records its prediction (the most frequent "
                "label so far; the smallest such label on ties)")
        ans = " (an int array of length `len(y)`)"
    return challenge(
        "A baseline that keeps up",
        f"A dashboard shows a baseline next to every model. After each new observation (t = 1, 2, …, len(y)) it refits "
        f"{what}. Return the recorded values without refitting from scratch. (`report` = {report!r}.)",
        setup, sol, tests, "len(y) ≤ 2·10^5", speed="O(n log n): two heaps for a running median, counts for a mode",
        answer_desc=ans)


def _maha_setup(s, n, d, report, thr, m, prefix="g"):
    return (f"{prefix} = np.random.default_rng({s})\nA = {prefix}.normal(0, 1, ({d}, {d}))\n"
            f"X = {prefix}.normal(0, 1, ({n}, {d})) @ A + {prefix}.normal(0, 5, {d})\n"
            f"X[:{n} // 25 + 1] += {prefix}.normal(0, 6, ({n} // 25 + 1, {d}))   # a few shifted rows\n"
            f"threshold = {thr}\nm = {m}\nreport = {report!r}")


@template(M, "challenge")
def loo_mahalanobis(rng):
    s = seeded(rng)
    report = rng.choice(["dist", "count", "top"])
    d = rng.randint(2, 3)
    thr = rng.choice([9.0, 12.0, 16.0])
    setup = _maha_setup(s, rng.randint(9, 14), d, report, thr, 3)
    tests = [_maha_setup(s + i, 200000, 3, report, thr, 20) for i in (1, 2)]
    sol = ("n = len(X)\nmu = X.mean(axis=0)\nZ = X - mu\nS = Z.T @ Z   # scatter matrix\n"
           "a = np.einsum('ij,ij->i', Z @ np.linalg.inv(S), Z)   # z_i^T S^-1 z_i\n"
           "# leaving row i out: mean shifts by z_i / (n - 1), scatter loses n/(n-1) z_i z_i^T (Sherman-Morrison)\n"
           "d2 = n * n / (n - 1) * a / (1 - n / (n - 1) * a)\n")
    sol += {"dist": "result = d2", "count": "result = int((d2 > threshold).sum())",
            "top": "result = np.argsort(-d2, kind='stable')[:m]"}[report]
    want = {"dist": "the leave-one-out squared distances, as a float array",
            "count": "how many rows have a leave-one-out squared distance above `threshold`",
            "top": "the indices of the `m` rows with the largest leave-one-out squared distance, largest first"}[report]
    return challenge(
        "Leave-one-out Mahalanobis distances",
        "An outlier should not hide itself by pulling the mean and covariance towards it. For every row `X[i]`, fit "
        "`EmpiricalCovariance()` (from `sklearn.covariance`: the mean and the maximum-likelihood covariance, divided by "
        "the number of rows) on **all other rows** and compute the squared Mahalanobis distance of `X[i]` to that fit "
        f"(what its `.mahalanobis` returns). Return {want}. (`report` = {report!r}.)",
        setup, sol, tests, "n ≤ 2·10^5 rows, d ≤ 3 columns",
        speed="O(n·d²): one fit for all rows and a rank-one update per row (Sherman–Morrison)", tol=1e-6)


def _ovo_setup(s, n, k, report, prefix="g"):
    return (f"{prefix} = np.random.default_rng({s})\nk = {k}\npairs = [(i, j) for i in range(k) for j in range(i + 1, k)]\n"
            f"conf = {prefix}.integers(-12, 13, ({n}, len(pairs))) / 4   # decision values, one column per pair\n"
            f"report = {report!r}")


@template(M, "challenge")
def ovo_vote_aggregation(rng):
    s = seeded(rng)
    report = rng.choice(["pred", "ties"])
    k = rng.randint(3, 5)
    setup = _ovo_setup(s, rng.randint(5, 8), k, report)
    tests = [_ovo_setup(s + i, 4000, 25, report) for i in (1, 2)]
    sol = ("P = np.array(pairs)\nn = len(conf)\nwin = np.where(conf > 0, P[:, 1], P[:, 0])\n"
           "votes = np.zeros((n, k))\nnp.add.at(votes, (np.repeat(np.arange(n), len(P)), win.ravel()), 1)\n"
           "sc = np.zeros((n, k))\nsc += conf @ (np.arange(k) == P[:, 1:2]).astype(float)\n"
           "sc -= conf @ (np.arange(k) == P[:, 0:1]).astype(float)\n")
    if report == "pred":
        sol += "result = (votes + sc / (3 * (np.abs(sc) + 1))).argmax(axis=1)"
        want = "the predicted class of every sample (an int array)"
    else:
        sol += "top = votes.max(axis=1, keepdims=True)\nresult = int(((votes == top).sum(axis=1) > 1).sum())"
        want = "how many samples have a **tie** in the vote count (two or more classes with the most votes)"
    return challenge(
        "One-vs-one voting",
        "A `OneVsOneClassifier` over `k` classes holds one binary model per pair `(i, j)`, i < j, in the order of "
        "`pairs`. For each sample, `conf[s, p]` is the decision value of pair p's model: it votes for class j when "
        "`conf > 0`, else for class i. scikit-learn breaks vote ties with the confidences: each class sums +conf over "
        "pairs where it is j and −conf over pairs where it is i; that sum c is squashed to c / (3(|c| + 1)) and added "
        f"to the votes; the prediction is the argmax (smallest class on exact ties). Return {want}. "
        f"(`report` = {report!r}.)",
        setup, sol, tests, "samples ≤ 4000, k ≤ 25 (300 pairs)",
        speed="vectorized over samples and pairs (no Python loop over samples)")
