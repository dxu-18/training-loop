"""4.20 Custom estimators, persistence (joblib) and configuration (config_context), plus sklearn.utils helpers
(shuffle, resample, class weights)."""

from ._common import P, catalog, challenge, lit, seeded, synth_clf, table, template

M = "4.20"
LR = "LogisticRegression(max_iter=1000)"
LRI = "from sklearn.linear_model import LogisticRegression\n"
BASE = "from sklearn.base import BaseEstimator, TransformerMixin\n"
CHECKS = "from sklearn.utils.validation import check_is_fitted, validate_data\n"
RESET = "from sklearn import set_config as _set_config\n_set_config(transform_output='default', assume_finite=False)\n"

CLIPPER = (BASE + CHECKS +
           "\n\nclass Clipper(TransformerMixin, BaseEstimator):\n"
           "    def __init__(self, quantile=0.01):\n        self.quantile = quantile\n\n"
           "    def fit(self, X, y=None):\n        X = validate_data(self, X)\n"
           "        self.low_, self.high_ = np.quantile(X, [self.quantile, 1 - self.quantile], axis=0)\n"
           "        return self\n\n"
           "    def transform(self, X):\n        check_is_fitted(self)\n        X = validate_data(self, X, reset=False)\n"
           "        return np.clip(X, self.low_, self.high_)\n")

MAJORITY = ("from sklearn.base import BaseEstimator, ClassifierMixin\n" + CHECKS +
            "\n\nclass MajorityClassifier(ClassifierMixin, BaseEstimator):\n"
            "    def fit(self, X, y):\n        X, y = validate_data(self, X, y)\n"
            "        self.classes_, counts = np.unique(y, return_counts=True)\n"
            "        self.majority_ = self.classes_[counts.argmax()]\n        return self\n\n"
            "    def predict(self, X):\n        check_is_fitted(self)\n        X = validate_data(self, X, reset=False)\n"
            "        return np.full(len(X), self.majority_)\n")


def _err(expr: str, var: str = "_r") -> str:
    """Check code: the name of the exception `expr` raises ('no error' if none)."""
    return (f"try:\n    {expr}\n    {var} = 'no error'\nexcept Exception as _e:\n    {var} = type(_e).__name__\n")


def _fresh(s, n=40, d=3, kind="t"):
    gen = f"_g.standard_t(3, size=({n}, {d}))" if kind == "t" else f"_g.normal(0, 1, ({n}, {d}))"
    return f"_g = np.random.default_rng({s})\n_A = {gen}\n_B = {gen.replace(str(n), '8', 1)} * 2\n"


def _imbalanced(rng, n=(120, 300)):
    w = rng.choice([0.1, 0.2, 0.3])
    d = synth_clf(rng, n=n, d=(3, 6), classes=rng.choice([2, 2, 3]), imbalance=w)
    return d


# ================================================================= lite

@template(M, "lite")
def sample_utils(rng):
    d = _imbalanced(rng, n=(40, 90))
    k = rng.randint(6, 10)
    r = rng.randint(0, 9)
    m = rng.randint(12, 20)
    w = rng.choice([2.0, 3.0, 5.0])
    items = [
        ("shuffle", f"Shuffle `X` and `y` **together** with `shuffle(X, y, random_state={r})` (from `sklearn.utils`) "
                    f"and return the first {k} shuffled labels.",
         f"from sklearn.utils import shuffle\nXs, ys = shuffle(X, y, random_state={r})\nresult = ys[:{k}]"),
        ("shuffle", f"Shuffle `X` and `y` with `shuffle(X, y, random_state={r})` and return `(Xs.shape, Xs[:3, 0])`: the "
                    f"shape is unchanged, only the order moves.",
         f"from sklearn.utils import shuffle\nXs, ys = shuffle(X, y, random_state={r})\nresult = (Xs.shape, Xs[:3, 0])"),
        ("resample", f"Draw a bootstrap sample of the row numbers with `resample(np.arange(len(y)), random_state={r})` "
                     f"(from `sklearn.utils`; with replacement, same size) and return how many **distinct** rows it "
                     f"contains.",
         f"from sklearn.utils import resample\nresult = len(np.unique(resample(np.arange(len(y)), random_state={r})))"),
        ("resample", f"Return the labels of `resample(X, y, n_samples={m}, replace=False, random_state={r})` (a "
                     f"subsample without repeats).",
         f"from sklearn.utils import resample\nresult = resample(X, y, n_samples={m}, replace=False, random_state={r})[1]"),
        ("stratify", f"Draw `resample(X, y, n_samples={m}, stratify=y, random_state={r})` (class proportions kept) and "
                     f"return the class counts of the sampled labels (`np.bincount`).",
         f"from sklearn.utils import resample\nXr, yr = resample(X, y, n_samples={m}, stratify=y, random_state={r})\n"
         "result = np.bincount(yr)"),
        ("compute_class_weight", "Return `compute_class_weight('balanced', classes=np.unique(y), y=y)` (from "
                                 "`sklearn.utils.class_weight`): rare classes get large weights.",
         "from sklearn.utils.class_weight import compute_class_weight\n"
         "result = compute_class_weight('balanced', classes=np.unique(y), y=y)"),
        ("compute_sample_weight", "Return the first 6 entries of `compute_sample_weight('balanced', y)` (from "
                                  "`sklearn.utils.class_weight`): each row gets its class's weight.",
         "from sklearn.utils.class_weight import compute_sample_weight\nresult = compute_sample_weight('balanced', y)[:6]"),
        ("compute_sample_weight", f"Weights can come from a dict. Return the sum of `compute_sample_weight({{0: 1.0, 1: "
                                  f"{w}}}, y)` over all rows (classes missing from the dict get weight 1).",
         f"from sklearn.utils.class_weight import compute_sample_weight\nresult = compute_sample_weight({{0: 1.0, 1: {w}}}, "
         f"y).sum()"),
    ]
    return catalog(rng, items, d.setup, f"{d.desc}; the classes are imbalanced.", tol=1e-9)


@template(M, "lite")
def config_settings(rng):
    t = table(rng, numeric_only=True, n=(30, 60))
    items = [
        ("config_context", "`config_context` (from `sklearn`) changes scikit-learn settings inside a `with` block only. "
                           "Return `(inside, outside)`: `get_config()['assume_finite']` inside `with "
                           "config_context(assume_finite=True):` and after it.",
         "from sklearn import config_context, get_config\nwith config_context(assume_finite=True):\n"
         "    inside = get_config()['assume_finite']\nresult = (inside, get_config()['assume_finite'])"),
        ("transform_output", "Inside `with config_context(transform_output='pandas'):`, transformers return DataFrames. "
                             "Fit-transform `df` with `StandardScaler()` there and return the result.",
         "from sklearn import config_context\nfrom sklearn.preprocessing import StandardScaler\n"
         "with config_context(transform_output='pandas'):\n    result = StandardScaler().fit_transform(df)"),
        ("get_config", "Return the current value of the `transform_output` setting (`get_config()` returns a dict).",
         "from sklearn import get_config\nresult = get_config()['transform_output']"),
        ("set_config", "`set_config` changes a setting **globally** (it leaks into later code). Call "
                       "`set_config(transform_output='pandas')`, fit-transform `df.to_numpy()` with `StandardScaler()`, "
                       "then restore the default with `set_config(transform_output='default')` and return the output's "
                       "column names as a list.",
         "from sklearn import set_config\nfrom sklearn.preprocessing import StandardScaler\n"
         "set_config(transform_output='pandas')\nZ = StandardScaler().fit_transform(df.to_numpy())\n"
         "set_config(transform_output='default')\nresult = list(Z.columns)"),
        ("config_context", "Contexts nest: inside `config_context(assume_finite=True)` open `config_context("
                           "assume_finite=False)` and return the `assume_finite` value read in the inner block and "
                           "back in the outer block, as a tuple.",
         "from sklearn import config_context, get_config\nwith config_context(assume_finite=True):\n"
         "    with config_context(assume_finite=False):\n        inner = get_config()['assume_finite']\n"
         "    outer = get_config()['assume_finite']\nresult = (inner, outer)"),
        ("set_output", "The per-estimator version of the pandas output: `StandardScaler().set_output(transform='pandas')`. "
                       "Fit-transform `df` with it and return the column means of the result (a Series).",
         "from sklearn.preprocessing import StandardScaler\n"
         "result = StandardScaler().set_output(transform='pandas').fit_transform(df).mean()", {"tol": 1e-9}),
        ("assume_finite", "With `assume_finite=True` scikit-learn skips its NaN/inf checks (a little faster, but bad "
                          "data passes silently). Inside `with config_context(assume_finite=True):`, fit `StandardScaler()` "
                          "on `df` and return its `mean_`.",
         "from sklearn import config_context\nfrom sklearn.preprocessing import StandardScaler\n"
         "with config_context(assume_finite=True):\n    result = StandardScaler().fit(df).mean_"),
    ]
    return catalog(rng, items, t.setup, f"{t.desc}.", post=RESET + "_", tol=1e-9)


@template(M, "lite")
def joblib_basics(rng):
    d = synth_clf(rng, n=(80, 200), d=(3, 6), classes=2)
    k = rng.randint(3, 6)
    name = rng.choice(["model", "clf", "pipeline", "churn_model", "baseline"])
    setup = (d.setup + f"\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.pipeline import make_pipeline\n"
             f"from sklearn.preprocessing import StandardScaler\n"
             f"model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X, y)")
    items = [
        ("dump", f"Save `model` with `joblib.dump(model, '{name}.joblib')`, load it back with `joblib.load` and return "
                 f"the loaded model's predictions for the first {k} rows.",
         f"import joblib\njoblib.dump(model, '{name}.joblib')\nloaded = joblib.load('{name}.joblib')\n"
         f"result = loaded.predict(X[:{k}])", {"require": ["dump", "load"]}),
        ("dump", f"`joblib.dump` returns the list of files it wrote. Return it for `joblib.dump(model, '{name}.joblib')`.",
         f"import joblib\nresult = joblib.dump(model, '{name}.joblib')"),
        ("load", f"Save `model` to `'{name}.joblib'`, load it, and return the loaded model's accuracy on `X, y`.",
         f"import joblib\njoblib.dump(model, '{name}.joblib')\nresult = joblib.load('{name}.joblib').score(X, y)"),
        ("pickle", "joblib builds on Python's `pickle`. Round-trip `model` through `pickle.dumps` / `pickle.loads` (in "
                   "memory, no file) and return the copy's `predict_proba(X[:3])`.",
         "import pickle\ncopy = pickle.loads(pickle.dumps(model))\nresult = copy.predict_proba(X[:3])",
         {"require": ["dumps", "loads"]}),
        ("compress", f"Save `model` compressed (`joblib.dump(model, '{name}.joblib', compress=3)`), load it and return the "
                     f"coefficients of its logistic-regression step (`[-1].coef_`).",
         f"import joblib\njoblib.dump(model, '{name}.joblib', compress=3)\n"
         f"result = joblib.load('{name}.joblib')[-1].coef_"),
        ("dump", f"Save a bundle: `joblib.dump({{'model': model, 'n_features': X.shape[1]}}, '{name}.joblib')`. Load it "
                 f"and return `(bundle['n_features'], bundle['model'].predict(X[:{k}]))`.",
         f"import joblib\njoblib.dump({{'model': model, 'n_features': X.shape[1]}}, '{name}.joblib')\n"
         f"bundle = joblib.load('{name}.joblib')\nresult = (bundle['n_features'], bundle['model'].predict(X[:{k}]))",
         {"require": ["dump", "load"]}),
    ]
    return catalog(rng, items, setup, f"{d.desc}; `model` is a fitted `StandardScaler` + `LogisticRegression` pipeline. "
                                      f"Your code runs in an empty temporary folder, so you can write files.", tol=1e-9)


@template(M, "lite")
def tiny_estimators(rng):
    s = seeded(rng)
    c = round(rng.uniform(-3, 3), 1)
    f = rng.choice([0.5, 2.0, 10.0])
    lab = rng.choice([0, 1])
    data = _fresh(s, n=30, d=3, kind="normal") + "_y = (_A[:, 0] > 0).astype(int)\n"
    base = "from sklearn.base import BaseEstimator, TransformerMixin\n"
    items = [
        ("TransformerMixin", f"Write a transformer `AddConstant(TransformerMixin, BaseEstimator)` (from `sklearn.base`) "
                             f"whose `__init__(self, c=1.0)` only stores `c`, whose `fit(X, y=None)` returns `self`, and "
                             f"whose `transform(X)` returns `X + self.c`. It is checked with `AddConstant(c={c})."
                             f"fit_transform(A)` on fresh data (`fit_transform` comes free from `TransformerMixin`) and "
                             f"`get_params()`.",
         base + "\n\nclass AddConstant(TransformerMixin, BaseEstimator):\n    def __init__(self, c=1.0):\n        self.c = c\n\n"
                "    def fit(self, X, y=None):\n        return self\n\n    def transform(self, X):\n        return X + self.c\n",
         {"post": data + f"_t = AddConstant(c={c})\n(_t.fit_transform(_A), _t.get_params())",
          "require": ["TransformerMixin", "BaseEstimator"]}),
        ("BaseEstimator", f"Write `Scale(TransformerMixin, BaseEstimator)` with `__init__(self, factor=1.0)` (store it, "
                          f"nothing else), a `fit` that returns `self`, and `transform(X)` returning `X * self.factor`. "
                          f"`BaseEstimator` makes `set_params` work: the check runs `Scale().set_params(factor={f})"
                          f".fit_transform(A)` on fresh data and reads `repr` of the estimator.",
         base + "\n\nclass Scale(TransformerMixin, BaseEstimator):\n    def __init__(self, factor=1.0):\n"
                "        self.factor = factor\n\n    def fit(self, X, y=None):\n        return self\n\n"
                "    def transform(self, X):\n        return X * self.factor\n",
         {"post": data + f"_t = Scale().set_params(factor={f})\n(_t.fit_transform(_A), repr(_t))",
          "require": ["TransformerMixin", "BaseEstimator"]}),
        ("validate_data", "Write `Center(TransformerMixin, BaseEstimator)`: `fit` validates `X` with `validate_data(self, "
                          "X)` (from `sklearn.utils.validation`; it also records `n_features_in_`), stores the column "
                          "means in `mean_` and returns `self`; `transform` returns `X - self.mean_`. Checked with "
                          "`Center().fit(A)`: its `transform(B)`, `mean_` and `n_features_in_` on fresh data.",
         base + "from sklearn.utils.validation import validate_data\n\n\nclass Center(TransformerMixin, BaseEstimator):\n"
                "    def fit(self, X, y=None):\n        X = validate_data(self, X)\n        self.mean_ = X.mean(axis=0)\n"
                "        return self\n\n    def transform(self, X):\n        return X - self.mean_\n",
         {"post": data + "_t = Center().fit(_A)\n(_t.transform(_B), _t.mean_, _t.n_features_in_)",
          "require": ["validate_data", "TransformerMixin"]}),
        ("ClassifierMixin", f"Write `ConstantClassifier(ClassifierMixin, BaseEstimator)` with `__init__(self, label=0)`: "
                            f"`fit(X, y)` stores `classes_ = np.unique(y)` and returns `self`; `predict(X)` returns "
                            f"`np.full(len(X), self.label)`. `ClassifierMixin` adds `score` (accuracy). Checked with "
                            f"`ConstantClassifier(label={lab}).fit(A, y)`: its `predict(B)`, `classes_` and `score(A, y)` "
                            f"on fresh data.",
         "from sklearn.base import BaseEstimator, ClassifierMixin\n\n\n"
         "class ConstantClassifier(ClassifierMixin, BaseEstimator):\n    def __init__(self, label=0):\n"
         "        self.label = label\n\n    def fit(self, X, y):\n        self.classes_ = np.unique(y)\n        return self\n\n"
         "    def predict(self, X):\n        return np.full(len(X), self.label)\n",
         {"post": data + f"_m = ConstantClassifier(label={lab}).fit(_A, _y)\n(_m.predict(_B), _m.classes_, _m.score(_A, _y))",
          "require": ["ClassifierMixin", "BaseEstimator"]}),
        ("RegressorMixin", "Write `MeanRegressor(RegressorMixin, BaseEstimator)` (from `sklearn.base`): `fit(X, y)` "
                           "stores `mean_ = np.mean(y)` and returns `self`; `predict(X)` returns `np.full(len(X), "
                           "self.mean_)`. `RegressorMixin` adds `score` (R²). Checked on fresh data: `predict(B)` and "
                           "`score` on a target that is not centred.",
         "from sklearn.base import BaseEstimator, RegressorMixin\n\n\nclass MeanRegressor(RegressorMixin, BaseEstimator):\n"
         "    def fit(self, X, y):\n        self.mean_ = np.mean(y)\n        return self\n\n"
         "    def predict(self, X):\n        return np.full(len(X), self.mean_)\n",
         {"post": data + "_t = _A[:, 1] * 3 + 5\n_m = MeanRegressor().fit(_A[:20], _t[:20])\n"
                         "(_m.predict(_B), _m.score(_A[20:], _t[20:]))",
          "require": ["RegressorMixin", "BaseEstimator"]}),
        ("get_params", "`BaseEstimator` builds `get_params` from the `__init__` signature. Write `Window(BaseEstimator)` "
                       "with `__init__(self, size=3, step=1, mode='mean')` storing the three arguments unchanged. "
                       "Checked with `Window(size=5).get_params()` and `clone(Window(mode='max')).mode` "
                       "(`sklearn.base.clone`).",
         "from sklearn.base import BaseEstimator\n\n\nclass Window(BaseEstimator):\n"
         "    def __init__(self, size=3, step=1, mode='mean'):\n        self.size = size\n        self.step = step\n"
         "        self.mode = mode\n",
         {"post": "from sklearn.base import clone as _clone\n(Window(size=5).get_params(), _clone(Window(mode='max')).mode)",
          "require": ["BaseEstimator"]}),
    ]
    return catalog(rng, items, "", "Custom estimators plug into scikit-learn when they follow its conventions.",
                   tol=1e-9)


# ================================================================= easy

@template(M, "easy")
def clipper_transformer(rng):
    s = seeded(rng)
    q = rng.choice([0.05, 0.1, 0.2])
    task = rng.choice(["values", "unfitted", "features", "pipeline", "names"])
    head = ("Write the guide's `Clipper(TransformerMixin, BaseEstimator)`: `__init__(self, quantile=0.01)` only stores "
            "`quantile`; `fit(X, y=None)` validates `X` with `validate_data(self, X)`, stores the per-column "
            "`quantile` and `1 - quantile` quantiles (`np.quantile(..., axis=0)`) as `low_` and `high_`, and returns "
            "`self`; `transform(X)` calls `check_is_fitted(self)`, validates with `validate_data(self, X, reset=False)` "
            "and returns `np.clip(X, self.low_, self.high_)`.")
    data = _fresh(s)
    req = ["TransformerMixin", "BaseEstimator", "check_is_fitted", "validate_data"]
    if task == "values":
        return P(f"{head} It is checked on fresh heavy-tailed data: `Clipper(quantile={q}).fit(A)`, then its "
                 f"`transform(B)`, `low_` and `high_`.", "", CLIPPER,
                 post=data + f"_c = Clipper(quantile={q}).fit(_A)\n(_c.transform(_B), _c.low_, _c.high_)", require=req)
    if task == "unfitted":
        return P(f"{head} The check calls `transform` on an **unfitted** `Clipper()` (it must raise `NotFittedError`), "
                 f"then `Clipper(quantile={q}).fit(A).transform(B)` on fresh data.", "", CLIPPER,
                 post=data + _err("Clipper().transform(_B)") + f"(_r, Clipper(quantile={q}).fit(_A).transform(_B))",
                 require=req)
    if task == "features":
        return P(f"{head} `validate_data` records `n_features_in_` and rejects inputs with a different number of "
                 f"columns. The check fits `Clipper(quantile={q})` on fresh 3-column data and returns `n_features_in_` "
                 f"and the exception name raised by `transform` on 2-column data.", "", CLIPPER,
                 post=data + f"_c = Clipper(quantile={q}).fit(_A)\n" + _err("_c.transform(_B[:, :2])") +
                 "(_c.n_features_in_, _r)", require=req)
    if task == "pipeline":
        return P(f"{head} Then use it: `make_pipeline(Clipper({q}), StandardScaler(), {LR})`, fitted on the breast "
                 f"cancer data `X, y`, and return its training accuracy.",
                 "from sklearn.datasets import load_breast_cancer\nX, y = load_breast_cancer(return_X_y=True)",
                 CLIPPER + "from sklearn.linear_model import LogisticRegression\nfrom sklearn.pipeline import make_pipeline\n"
                 "from sklearn.preprocessing import StandardScaler\n\n"
                 f"result = make_pipeline(Clipper({q}), StandardScaler(), {LR}).fit(X, y).score(X, y)",
                 require=req + ["make_pipeline"], tol=1e-6)
    return P(f"{head} Fitting on a DataFrame records its column names: the check fits `Clipper(quantile={q})` on a "
             f"fresh DataFrame and returns `feature_names_in_` and the clipped values of a new frame.", "", CLIPPER,
             post=data + "_D = pd.DataFrame(_A, columns=['cpu', 'mem', 'disk'])\n"
                         f"_c = Clipper(quantile={q}).fit(_D)\n"
                         "(_c.feature_names_in_, _c.transform(pd.DataFrame(_B, columns=['cpu', 'mem', 'disk'])))",
             require=req)


@template(M, "easy")
def majority_classifier(rng):
    s = seeded(rng)
    task = rng.choice(["predict", "score", "cv", "ties", "unfitted"])
    head = ("Write the guide's `MajorityClassifier(ClassifierMixin, BaseEstimator)` (no hyperparameters): `fit(X, y)` "
            "validates with `X, y = validate_data(self, X, y)`, stores `classes_` and the most frequent label "
            "`majority_` (`np.unique(..., return_counts=True)` then `argmax`: the smallest label on ties) and returns "
            "`self`; `predict(X)` checks it is fitted, validates `X` with `reset=False`, and returns "
            "`np.full(len(X), self.majority_)`.")
    data = (f"_g = np.random.default_rng({s})\n_A = _g.normal(0, 1, (30, 2))\n_y = _g.choice([3, 7, 9], 30, "
            f"p=[0.2, 0.5, 0.3])\n")
    req = ["ClassifierMixin", "BaseEstimator", "check_is_fitted", "validate_data"]
    if task == "predict":
        return P(f"{head} Checked on fresh data with labels 3, 7, 9: `classes_`, `majority_` and `predict` of 4 rows.",
                 "", MAJORITY, post=data + "_m = MajorityClassifier().fit(_A, _y)\n(_m.classes_, _m.majority_, "
                                           "_m.predict(_A[:4]))", require=req)
    if task == "score":
        return P(f"{head} `ClassifierMixin` provides `score` (accuracy). Return the training accuracy of "
                 f"`MajorityClassifier()` on the breast cancer data `X, y` (the share of the majority class).",
                 "from sklearn.datasets import load_breast_cancer\nX, y = load_breast_cancer(return_X_y=True)",
                 MAJORITY + "\nresult = MajorityClassifier().fit(X, y).score(X, y)", require=req + ["score"])
    if task == "cv":
        return P(f"{head} A conforming estimator works with every scikit-learn tool. Return the mean 5-fold "
                 f"`cross_val_score` of `MajorityClassifier()` on the breast cancer data `X, y`.",
                 "from sklearn.datasets import load_breast_cancer\nX, y = load_breast_cancer(return_X_y=True)",
                 MAJORITY + "from sklearn.model_selection import cross_val_score\n\n"
                 "result = cross_val_score(MajorityClassifier(), X, y, cv=5).mean()",
                 require=req + ["cross_val_score"], tol=1e-9)
    if task == "ties":
        return P(f"{head} Checked on labels with a tie (`['b', 'a', 'b', 'a', 'c']`: the smallest of the tied labels "
                 f"wins) and on fresh data.", "", MAJORITY,
                 post=data + "_t = MajorityClassifier().fit(np.zeros((5, 1)), np.array(['b', 'a', 'b', 'a', 'c']))\n"
                             "(_t.majority_, MajorityClassifier().fit(_A, _y).predict(_A[:3]))", require=req)
    return P(f"{head} The check calls `predict` on an unfitted `MajorityClassifier()` (expecting `NotFittedError`) and "
             f"on a fitted one with the wrong number of columns (expecting `ValueError`).", "", MAJORITY,
             post=data + _err("MajorityClassifier().predict(_A)", "_r1") +
             _err("MajorityClassifier().fit(_A, _y).predict(np.zeros((2, 5)))", "_r2") + "(_r1, _r2)", require=req)


@template(M, "easy")
def class_weight_tasks(rng):
    d = _imbalanced(rng, n=(150, 300))
    cu = "from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight\n"
    items = [
        ("compute_class_weight", "The 'balanced' weight of class c is `n_samples / (n_classes * count_c)`. Compute it "
                                 "**by hand** with `np.bincount` and return it (it equals `compute_class_weight("
                                 "'balanced', classes=np.unique(y), y=y)`).",
         "result = len(y) / (len(np.unique(y)) * np.bincount(y))", {"require": ["bincount"]}),
        ("compute_class_weight", "Turn the balanced class weights into the dict `{class: weight}` that "
                                 "`LogisticRegression(class_weight=...)` accepts (classes as ints) and return it.",
         cu + "cls = np.unique(y)\nresult = dict(zip(cls.tolist(), compute_class_weight('balanced', classes=cls, y=y)))"),
        ("compute_sample_weight", "Balanced sample weights make every class carry the same total weight. Return the "
                                  "total weight of each class (`np.bincount(y, weights=...)`) for "
                                  "`compute_sample_weight('balanced', y)`.",
         cu + "w = compute_sample_weight('balanced', y)\nresult = np.bincount(y, weights=w)"),
        ("compute_sample_weight", "Return the sum of `compute_sample_weight('balanced', y)`: it always equals the number "
                                  "of rows.",
         cu + "result = compute_sample_weight('balanced', y).sum()"),
        ("class_weight", f"`class_weight='balanced'` is the same as passing balanced sample weights. Fit `{LR[:-1]}, "
                         f"class_weight='balanced')` on `X, y` and return its predicted class counts (`np.bincount`), "
                         f"alongside those of a plain `{LR}`, as a tuple (balanced first).",
         LRI + f"a = LogisticRegression(max_iter=1000, class_weight='balanced').fit(X, y).predict(X)\n"
               f"b = {LR}.fit(X, y).predict(X)\nresult = (np.bincount(a, minlength=len(np.unique(y))), "
               "np.bincount(b, minlength=len(np.unique(y))))"),
        ("compute_sample_weight", "Return the ratio between the largest and the smallest balanced sample weight: it "
                                  "equals the ratio between the largest and smallest class counts.",
         cu + "w = compute_sample_weight('balanced', y)\nresult = w.max() / w.min()"),
    ]
    return catalog(rng, items, d.setup, f"{d.desc}; the classes are imbalanced.", tol=1e-6)


@template(M, "easy")
def resampling_tasks(rng):
    d = synth_clf(rng, n=(150, 300), d=(3, 6), classes=2, imbalance=rng.choice([0.1, 0.2]))
    r = rng.randint(0, 9)
    b = rng.choice([20, 30, 50])
    task = rng.choice(["upsample", "downsample", "bootstrap_std", "stratified", "shuffle_pairs"])
    ru = "from sklearn.utils import resample\n"
    if task == "upsample":
        return P(f"{d.desc}; class 1 is rare. Upsample it: draw `resample(X[y == 1], y[y == 1], replace=True, "
                 f"n_samples=(y == 0).sum(), random_state={r})`, stack the draws under the class-0 rows, and return the "
                 f"class counts of the new label vector (`np.bincount`).", d.setup,
                 ru + f"Xu, yu = resample(X[y == 1], y[y == 1], replace=True, n_samples=(y == 0).sum(), random_state={r})\n"
                 "y_new = np.concatenate([y[y == 0], yu])\nresult = np.bincount(y_new)", require=["resample"])
    if task == "downsample":
        return P(f"{d.desc}; class 1 is rare. Downsample class 0 to the size of class 1 with `resample(X[y == 0], "
                 f"replace=False, n_samples=(y == 1).sum(), random_state={r})` and return the column means of the "
                 f"sampled class-0 rows.", d.setup,
                 ru + f"X0 = resample(X[y == 0], replace=False, n_samples=(y == 1).sum(), random_state={r})\n"
                 "result = X0.mean(axis=0)", require=["resample", "replace"], tol=1e-9)
    if task == "bootstrap_std":
        return P(f"{d.desc}. How uncertain is the mean of feature 0? Draw {b} bootstrap samples of `X[:, 0]` with "
                 f"`resample(X[:, 0], random_state=i)` for i = 0..{b - 1} and return the standard deviation of their "
                 f"means (the bootstrap standard error).", d.setup,
                 ru + f"means = [resample(X[:, 0], random_state=i).mean() for i in range({b})]\nresult = float(np.std(means))",
                 require=["resample"], tol=1e-9)
    if task == "stratified":
        m = rng.choice([40, 50, 60])
        return P(f"{d.desc}. Draw {m} rows with and without stratification (`resample(X, y, n_samples={m}, "
                 f"random_state={r})` and the same with `stratify=y`) and return the share of class 1 in `y`, in the "
                 f"plain sample and in the stratified sample, as a tuple.", d.setup,
                 ru + f"_, a = resample(X, y, n_samples={m}, random_state={r})\n"
                 f"_, s = resample(X, y, n_samples={m}, stratify=y, random_state={r})\n"
                 "result = (float(y.mean()), float(a.mean()), float(s.mean()))", require=["resample", "stratify"],
                 tol=1e-9)
    return P(f"{d.desc}. `shuffle` can shuffle several arrays consistently. Shuffle `X`, `y` and the row numbers "
             f"`np.arange(len(y))` together with `shuffle(..., random_state={r})` (from `sklearn.utils`) and return "
             f"`(first 5 row numbers, whether y_shuffled equals y[row numbers])`.", d.setup,
             f"from sklearn.utils import shuffle\nXs, ys, idx = shuffle(X, y, np.arange(len(y)), random_state={r})\n"
             "result = (idx[:5], bool((ys == y[idx]).all()))", require=["shuffle"])


@template(M, "easy")
def save_and_reload(rng):
    t = table(rng, kind="clf", numeric_only=True, n=(100, 220))
    h = t.n * 3 // 4
    name = rng.choice(["model.joblib", "pipe.joblib", "v1.joblib", "risk_model.joblib"])
    task = rng.choice(["verify", "score", "folder", "search", "bundle"])
    fit = (LRI + "from sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\n"
           f"import joblib\nmodel = make_pipeline(StandardScaler(), {LR}).fit(df[:{h}], y[:{h}])\n")
    head = (f"{t.desc}. Your code runs in an empty temporary folder. Fit `make_pipeline(StandardScaler(), {LR})` on "
            f"the first {h} rows")
    if task == "verify":
        return P(f"{head}, save it with `joblib.dump` to `'{name}'`, load it back and verify it: return "
                 f"`(loaded test predictions, whether they equal the original model's)` on the remaining rows.",
                 t.setup, fit + f"joblib.dump(model, '{name}')\nloaded = joblib.load('{name}')\n"
                 f"p = loaded.predict(df[{h}:])\nresult = (p, bool((p == model.predict(df[{h}:])).all()))",
                 require=["dump", "load"])
    if task == "score":
        return P(f"{head}, save it to `'{name}'`, load it in a fresh variable and return the loaded model's "
                 f"`predict_proba` for the next 3 rows (the loaded model still knows the column names: pass a "
                 f"DataFrame).", t.setup,
                 fit + f"joblib.dump(model, '{name}')\nresult = joblib.load('{name}').predict_proba(df[{h}:{h + 3}])",
                 require=["dump", "load"], tol=1e-9)
    if task == "folder":
        return P(f"{head}, create a folder `models` (`os.makedirs`) and save the model as `models/{name}`. Return "
                 f"`(sorted(os.listdir('models')), loaded model's accuracy on the remaining rows)`.", t.setup,
                 fit + f"import os\nos.makedirs('models', exist_ok=True)\njoblib.dump(model, 'models/{name}')\n"
                 f"result = (sorted(os.listdir('models')), joblib.load('models/{name}').score(df[{h}:], y[{h}:]))",
                 require=["dump", "load", "makedirs"])
    if task == "search":
        return P(f"{t.desc}. Tune `C` of `make_pipeline(StandardScaler(), {LR})` with `GridSearchCV(..., "
                 f"{{'logisticregression__C': [0.1, 1.0, 10.0]}}, cv=3)` on the first {h} rows, save the "
                 f"`best_estimator_` to `'{name}'`, reload it and return `(best_params_, reloaded model's accuracy on "
                 f"the remaining rows)`.", t.setup,
                 LRI + "from sklearn.model_selection import GridSearchCV\nfrom sklearn.pipeline import make_pipeline\n"
                 "from sklearn.preprocessing import StandardScaler\nimport joblib\n"
                 f"gs = GridSearchCV(make_pipeline(StandardScaler(), {LR}), {{'logisticregression__C': [0.1, 1.0, 10.0]}}, "
                 f"cv=3).fit(df[:{h}], y[:{h}])\njoblib.dump(gs.best_estimator_, '{name}')\n"
                 f"result = (gs.best_params_, joblib.load('{name}').score(df[{h}:], y[{h}:]))",
                 require=["GridSearchCV", "best_params_", "dump"], tol=1e-9)
    return P(f"{head}. Models should travel with their metadata: save `{{'model': model, 'columns': list(df.columns), "
             f"'train_rows': {h}}}` to `'{name}'`, load the bundle and return `(bundle['columns'], "
             f"bundle['train_rows'], bundle['model'].score(df[{h}:], y[{h}:]))`.", t.setup,
             fit + f"joblib.dump({{'model': model, 'columns': list(df.columns), 'train_rows': {h}}}, '{name}')\n"
             f"bundle = joblib.load('{name}')\n"
             f"result = (bundle['columns'], bundle['train_rows'], bundle['model'].score(df[{h}:], y[{h}:]))",
             require=["dump", "load"], tol=1e-9)


@template(M, "easy")
def pandas_output_config(rng):
    t = table(rng, n=(40, 80))
    num, cat = t.num, t.cat
    ct = (f"make_column_transformer((StandardScaler(), {num!r}), (OneHotEncoder(sparse_output=False), {cat!r}))")
    imp = ("from sklearn import config_context\nfrom sklearn.compose import make_column_transformer\n"
           "from sklearn.preprocessing import OneHotEncoder, StandardScaler\n")
    items = [
        ("config_context", f"Inside `with config_context(transform_output='pandas'):`, fit-transform `df` with `ct = "
                           f"{ct}` and return the column names of the resulting DataFrame as a list.",
         imp + f"with config_context(transform_output='pandas'):\n    out = {ct}.fit_transform(df)\nresult = list(out.columns)"),
        ("config_context", f"Inside `with config_context(transform_output='pandas'):`, fit-transform `df` with `{ct}` and "
                           f"return the first 3 rows of the result (a DataFrame).",
         imp + f"with config_context(transform_output='pandas'):\n    result = {ct}.fit_transform(df).head(3)"),
        ("set_output", f"Without any global setting, ask a single transformer for pandas output: `{ct}.set_output("
                       f"transform='pandas')`. Return the result's shape and the dtype name of its first column as "
                       f"a tuple.",
         imp + f"out = {ct}.set_output(transform='pandas').fit_transform(df)\nresult = (out.shape, str(out.dtypes.iloc[0]))"),
        ("config_context", f"Settings revert when the block ends. Return `(type name of {ct}.fit_transform(df) inside "
                       f"config_context(transform_output='pandas'), type name of the same call after the block)`.",
         imp + f"with config_context(transform_output='pandas'):\n    a = type({ct}.fit_transform(df)).__name__\n"
               f"b = type({ct}.fit_transform(df)).__name__\nresult = (a, b)"),
    ]
    return catalog(rng, items, t.setup, f"{t.desc}.", post=RESET + "_", tol=1e-9)


@template(M, "easy")
def custom_estimator_api(rng):
    s = seeded(rng)
    q, q2 = rng.sample([0.01, 0.05, 0.1, 0.2], 2)
    n = rng.randint(30, 60)
    setup = (CLIPPER.replace("\n\n\n", "\n\n") +
             f"\n\nX = np.random.default_rng({s}).standard_t(3, size=({n}, 3)).round(3)   # heavy-tailed data")
    sp = "from sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\n"
    items = [
        ("get_params", f"`BaseEstimator` reads the hyperparameters from `__init__`. Return `Clipper(quantile={q})"
                       f".get_params()`.", f"result = Clipper(quantile={q}).get_params()"),
        ("set_params", f"Create `c = Clipper(quantile={q})`, change it with `set_params(quantile={q2})`, fit it on `X` "
                       f"and return `c.low_`.", f"c = Clipper(quantile={q}).set_params(quantile={q2}).fit(X)\nresult = c.low_"),
        ("clone", f"Fit `c = Clipper(quantile={q})` on `X`, then `clone` it (`sklearn.base.clone`). Return "
                  f"`(clone's quantile, hasattr(clone, 'low_'))`: a clone copies hyperparameters, not what was learned.",
         f"from sklearn.base import clone\nc = Clipper(quantile={q}).fit(X)\nk = clone(c)\n"
         "result = (k.quantile, hasattr(k, 'low_'))"),
        ("repr", f"Return `repr(Clipper(quantile={q}))`: `BaseEstimator` prints only the changed parameters.",
         f"result = repr(Clipper(quantile={q}))", {"require": []}),
        ("make_pipeline", "Return the step names `make_pipeline(Clipper(), StandardScaler())` gives its steps "
                          "(the lowercased class names), as a list.",
         sp + "result = [name for name, _ in make_pipeline(Clipper(), StandardScaler()).steps]"),
        ("get_params", "Nested parameters are reachable as `step__param`. Return the sorted list of the keys of "
                       "`make_pipeline(Clipper(), StandardScaler()).get_params()` that start with `'clipper__'`.",
         sp + "result = sorted(k for k in make_pipeline(Clipper(), StandardScaler()).get_params() "
              "if k.startswith('clipper__'))"),
        ("fit_transform", f"`TransformerMixin` gives `fit_transform` for free. Return the first 3 rows of "
                          f"`Clipper(quantile={q}).fit_transform(X)`.", f"result = Clipper(quantile={q}).fit_transform(X)[:3]"),
        ("check_is_fitted", "Return the name of the exception `check_is_fitted(Clipper())` raises (from "
                            "`sklearn.utils.validation`), and `check_is_fitted` of a fitted clipper's return value, as a "
                            "tuple.",
         "try:\n    check_is_fitted(Clipper())\n    name = 'no error'\nexcept Exception as e:\n    name = type(e).__name__\n"
         "result = (name, check_is_fitted(Clipper().fit(X)))"),
    ]
    return catalog(rng, items, setup, "The guide's `Clipper` transformer is defined above; `X` holds heavy-tailed "
                                      "data.", tol=1e-9)


# ================================================================= medium

LOG1P = (BASE + CHECKS + "\n\nclass Log1pColumns(TransformerMixin, BaseEstimator):\n"
         "    def __init__(self, columns=(0,)):\n        self.columns = columns\n\n"
         "    def fit(self, X, y=None):\n        validate_data(self, X)\n        return self\n\n"
         "    def transform(self, X):\n        check_is_fitted(self)\n        X = validate_data(self, X, reset=False)\n"
         "        return np.hstack([X, np.log1p(X[:, list(self.columns)])])\n")


@template(M, "medium")
def log1p_columns(rng):
    t = table(rng, numeric_only=True, n=(80, 160))
    s = seeded(rng)
    cols = sorted(rng.sample(range(len(t.num)), rng.randint(1, 2)))
    task = rng.choice(["values", "pipeline", "names", "replace"])
    head = ("Write `Log1pColumns(TransformerMixin, BaseEstimator)` with `__init__(self, columns=(0,))` (positions of the "
            "columns to transform; store it unchanged). `fit` validates `X` with `validate_data` and returns `self`; "
            "`transform` checks it is fitted, validates `X` (`reset=False`) and returns `X` with `np.log1p` copies of "
            "the chosen columns **appended** on the right.")
    data = f"_g = np.random.default_rng({s})\n_A = _g.gamma(2, 50, (12, 4))\n"
    req = ["TransformerMixin", "BaseEstimator", "validate_data"]
    if task == "values":
        return P(f"{head} Checked on fresh positive data: `Log1pColumns(columns={cols}).fit_transform(A)` and its "
                 f"`n_features_in_`.", "", LOG1P,
                 post=data + f"_t = Log1pColumns(columns={cols})\n(_t.fit_transform(_A), _t.n_features_in_)", require=req)
    if task == "pipeline":
        model = f"{LR}" if t.kind == "clf" else "Ridge()"
        mi = LRI if t.kind == "clf" else "from sklearn.linear_model import Ridge\n"
        return P(f"{head} Then fit `make_pipeline(Log1pColumns(columns={cols}), StandardScaler(), {model})` on `df` "
                 f"(as a NumPy array: `df.to_numpy()`) and `y`, and return its 5-fold `cross_val_score` mean.",
                 t.setup, LOG1P + mi + "from sklearn.model_selection import cross_val_score\n"
                 "from sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\n\n"
                 f"pipe = make_pipeline(Log1pColumns(columns={cols}), StandardScaler(), {model})\n"
                 "result = cross_val_score(pipe, df.to_numpy(), y, cv=5).mean()", require=req, tol=1e-6)
    if task == "names":
        sol = LOG1P.replace("        return np.hstack", "        return np.hstack") + (
            "\n    def get_feature_names_out(self, input_features=None):\n"
            "        names = list(getattr(self, 'feature_names_in_', [f'x{i}' for i in range(self.n_features_in_)]))\n"
            "        return np.array(names + ['log1p_' + names[j] for j in self.columns], dtype=object)\n")
        return P(f"{head} Also give it `get_feature_names_out(self, input_features=None)` returning the input names "
                 f"(`feature_names_in_` if it was fitted on a DataFrame, else `x0, x1, ...`) followed by `'log1p_' + "
                 f"name` for each chosen column, as an object array. Checked on a fresh DataFrame and a fresh array "
                 f"with `columns={cols}`.", "", sol,
                 post=data + "_D = pd.DataFrame(_A, columns=['a', 'b', 'c', 'd'])\n"
                             f"(Log1pColumns(columns={cols}).fit(_D).get_feature_names_out(), "
                             f"Log1pColumns(columns={cols}).fit(_A).get_feature_names_out())", require=req)
    sol = LOG1P.replace("        return np.hstack([X, np.log1p(X[:, list(self.columns)])])\n",
                        "        X = X.astype(float).copy()\n        cols = list(self.columns)\n"
                        "        X[:, cols] = np.log1p(X[:, cols]) if self.replace else X[:, cols]\n"
                        "        return X if self.replace else np.hstack([X, np.log1p(X[:, cols])])\n")
    sol = sol.replace("    def __init__(self, columns=(0,)):\n        self.columns = columns\n",
                      "    def __init__(self, columns=(0,), replace=False):\n        self.columns = columns\n"
                      "        self.replace = replace\n")
    return P(f"{head} Add a second parameter `replace=False`: when True, the chosen columns are replaced in place "
             f"(same shape as `X`) instead of appended. Checked on fresh data with `columns={cols}` and both values of "
             f"`replace`.", "", sol,
             post=data + f"(Log1pColumns(columns={cols}).fit_transform(_A), "
                         f"Log1pColumns(columns={cols}, replace=True).fit_transform(_A))", require=req)


@template(M, "medium")
def tune_custom_transformer(rng):
    task = rng.choice(["clip_params", "clip_score", "log_params", "results"])
    qs = sorted(rng.sample([0.0, 0.01, 0.05, 0.1, 0.2], 3))
    s = seeded(rng)
    head = ("Write the guide's `Clipper` transformer: `class Clipper(TransformerMixin, BaseEstimator)` with "
            "`__init__(self, quantile=0.01)`; `fit` stores the per-column quantiles `low_` (at `quantile`) and "
            "`high_` (at `1 - quantile`) and returns `self`; `transform` clips `X` to them.")
    setup = (f"from sklearn.datasets import load_breast_cancer\nX, y = load_breast_cancer(return_X_y=True)\n"
             f"rows = np.random.default_rng({s}).permutation(len(X))[:300]\nX, y = X[rows], y[rows]")
    imp = (LRI + "from sklearn.model_selection import GridSearchCV\nfrom sklearn.pipeline import make_pipeline\n"
           "from sklearn.preprocessing import StandardScaler\n\n")
    pipe = f"make_pipeline(Clipper(), StandardScaler(), {LR})"
    gs = f"gs = GridSearchCV({pipe}, {{'clipper__quantile': {qs}}}, cv=3).fit(X, y)\n"
    if task == "clip_params":
        return P(f"{head} Because `BaseEstimator` exposes `quantile` through `get_params`, a grid search can tune it: "
                 f"`GridSearchCV({pipe}, {{'clipper__quantile': {qs}}}, cv=3)` on `X, y`. Return `best_params_`.",
                 setup, CLIPPER + imp + gs + "result = gs.best_params_", require=["GridSearchCV", "best_params_",
                                                                               "TransformerMixin"])
    if task == "clip_score":
        return P(f"{head} Grid-search `{pipe}` over `{{'clipper__quantile': {qs}}}` with `cv=3` on `X, y` and return "
                 f"`(best_score_, best_estimator_[0].low_[:3])` (the refitted clipper's lower bounds of the first 3 "
                 f"columns).", setup, CLIPPER + imp + gs + "result = (gs.best_score_, gs.best_estimator_[0].low_[:3])",
                 require=["GridSearchCV", "best_estimator_"], tol=1e-6)
    if task == "results":
        return P(f"{head} Grid-search `{pipe}` over `{{'clipper__quantile': {qs}}}` with `cv=3` on `X, y` and return "
                 f"`cv_results_['mean_test_score']`.", setup,
                 CLIPPER + imp + gs + "result = gs.cv_results_['mean_test_score']", require=["GridSearchCV"], tol=1e-6)
    opts = [[0], [0, 1], [0, 1, 2, 3]]
    return P(f"Write the `Log1pColumns(columns=...)` transformer (it appends `np.log1p` copies of the chosen column "
             f"positions; `fit` returns `self`). Tune which columns get the copies: `GridSearchCV(make_pipeline("
             f"Log1pColumns(), StandardScaler(), {LR}), {{'log1pcolumns__columns': {opts}}}, cv=3)` on `X, y`. Return "
             f"`best_params_`.", setup,
             LOG1P + imp + f"gs = GridSearchCV(make_pipeline(Log1pColumns(), StandardScaler(), {LR}), "
                           f"{{'log1pcolumns__columns': {opts}}}, cv=3).fit(X, y)\nresult = gs.best_params_",
             require=["GridSearchCV", "best_params_"])


CENTROID = ("from sklearn.base import BaseEstimator, ClassifierMixin\n" + CHECKS +
            "\n\nclass CentroidClassifier(ClassifierMixin, BaseEstimator):\n"
            "    def fit(self, X, y):\n        X, y = validate_data(self, X, y)\n        self.classes_ = np.unique(y)\n"
            "        self.centroids_ = np.array([X[y == c].mean(axis=0) for c in self.classes_])\n        return self\n\n"
            "    def predict(self, X):\n        check_is_fitted(self)\n        X = validate_data(self, X, reset=False)\n"
            "        d = ((X[:, None, :] - self.centroids_[None]) ** 2).sum(axis=2)\n"
            "        return self.classes_[d.argmin(axis=1)]\n")


@template(M, "medium")
def centroid_classifier(rng):
    s = seeded(rng)
    task = rng.choice(["vs_sklearn", "proba", "shrink", "score"])
    head = ("Write `CentroidClassifier(ClassifierMixin, BaseEstimator)`: `fit(X, y)` validates the data, stores "
            "`classes_` and one mean vector per class in `centroids_` (rows in `classes_` order), and returns `self`; "
            "`predict(X)` checks it is fitted, validates `X` and returns the class of the nearest centroid (Euclidean; "
            "the first class on ties).")
    data = (f"_g = np.random.default_rng({s})\n_y = _g.integers(0, 3, 60)\n"
            "_A = _g.normal(0, 1, (60, 2)) + np.array([[0, 0], [3, 0], [0, 3]])[_y]\n_B = _g.normal(1, 2, (10, 2))\n")
    req = ["ClassifierMixin", "BaseEstimator"]
    if task == "vs_sklearn":
        return P(f"{head} It should agree with scikit-learn's `NearestCentroid`: the check returns your predictions on "
                 f"fresh data and their agreement with `NearestCentroid().fit(A, y).predict(B)`.", "", CENTROID,
                 post=data + "from sklearn.neighbors import NearestCentroid as _NC\n_p = CentroidClassifier().fit(_A, _y)"
                             ".predict(_B)\n(_p, float((_p == _NC().fit(_A, _y).predict(_B)).mean()))", require=req)
    if task == "score":
        d = synth_clf(rng, n=(150, 300), d=(3, 6), classes=(2, 4))
        return P(f"{head} Then return the mean 5-fold `cross_val_score` of `CentroidClassifier()` on `X, y` "
                 f"({d.desc}).", d.setup,
                 CENTROID + "from sklearn.model_selection import cross_val_score\n\n"
                 "result = cross_val_score(CentroidClassifier(), X, y, cv=5).mean()", require=req + ["cross_val_score"],
                 tol=1e-9)
    if task == "proba":
        sol = CENTROID + ("\n    def predict_proba(self, X):\n        check_is_fitted(self)\n"
                          "        X = validate_data(self, X, reset=False)\n"
                          "        d = ((X[:, None, :] - self.centroids_[None]) ** 2).sum(axis=2)\n"
                          "        e = np.exp(-(d - d.min(axis=1, keepdims=True)))\n"
                          "        return e / e.sum(axis=1, keepdims=True)\n")
        return P(f"{head} Add `predict_proba(X)`: the softmax of minus the **squared** distances to the centroids "
                 f"(subtract each row's smallest distance before `np.exp`). Checked on fresh data: `predict_proba(B)` "
                 f"and `predict(B)`.", "", sol,
                 post=data + "_m = CentroidClassifier().fit(_A, _y)\n(_m.predict_proba(_B), _m.predict(_B))", require=req)
    sol = CENTROID.replace("    def fit(self, X, y):", "    def __init__(self, shrink=0.0):\n        self.shrink = shrink\n\n"
                                                       "    def fit(self, X, y):")
    sol = sol.replace("        self.centroids_ = np.array([X[y == c].mean(axis=0) for c in self.classes_])\n",
                      "        mu = X.mean(axis=0)\n        self.centroids_ = np.array([(1 - self.shrink) * X[y == c].mean(axis=0) "
                      "+ self.shrink * mu for c in self.classes_])\n")
    sh = rng.choice([0.2, 0.5, 0.8])
    return P(f"{head} Add a hyperparameter `shrink=0.0` (stored unchanged in `__init__`): each centroid becomes `(1 - "
             f"shrink) * class mean + shrink * overall mean`. Checked on fresh data with `shrink={sh}`: `centroids_` and "
             f"`predict(B)`.", "", sol,
             post=data + f"_m = CentroidClassifier(shrink={sh}).fit(_A, _y)\n(_m.centroids_, _m.predict(_B))",
             require=req)


@template(M, "medium")
def custom_regressors(rng):
    s = seeded(rng)
    task = rng.choice(["ridge", "binned", "knn"])
    data = (f"_g = np.random.default_rng({s})\n_A = _g.uniform(0, 10, (50, 2))\n"
            "_t = 3 * _A[:, 0] - 2 * _A[:, 1] + _g.normal(0, 1, 50)\n_B = _g.uniform(0, 10, (6, 2))\n")
    base = "from sklearn.base import BaseEstimator, RegressorMixin\nfrom sklearn.utils.validation import check_is_fitted, validate_data\n\n\n"
    if task == "ridge":
        a = rng.choice([0.1, 1.0, 10.0])
        sol = (base + "class ClosedFormRidge(RegressorMixin, BaseEstimator):\n    def __init__(self, alpha=1.0):\n"
               "        self.alpha = alpha\n\n    def fit(self, X, y):\n        X, y = validate_data(self, X, y)\n"
               "        xm, ym = X.mean(axis=0), y.mean()\n        Xc = X - xm\n"
               "        self.coef_ = np.linalg.solve(Xc.T @ Xc + self.alpha * np.eye(X.shape[1]), Xc.T @ (y - ym))\n"
               "        self.intercept_ = ym - xm @ self.coef_\n        return self\n\n"
               "    def predict(self, X):\n        check_is_fitted(self)\n        X = validate_data(self, X, reset=False)\n"
               "        return X @ self.coef_ + self.intercept_\n")
        return P(f"Write `ClosedFormRidge(RegressorMixin, BaseEstimator)` with `alpha=1.0`: `fit` centres `X` and `y`, "
                 f"solves `(Xcᵀ Xc + alpha·I) w = Xcᵀ yc` (`np.linalg.solve`), stores `coef_` and `intercept_ = mean(y) "
                 f"- mean(X) @ coef_`, and returns `self`; `predict` returns `X @ coef_ + intercept_`. It should match "
                 f"`sklearn.linear_model.Ridge`: the check returns your `coef_`, `intercept_`, `score` on fresh data "
                 f"with `alpha={a}`, and the largest gap to `Ridge(alpha={a})`'s coefficients.", "", sol,
                 post=data + f"from sklearn.linear_model import Ridge as _R\n_m = ClosedFormRidge(alpha={a}).fit(_A, _t)\n"
                             f"(_m.coef_, _m.intercept_, _m.score(_A, _t), "
                             f"bool(np.abs(_m.coef_ - _R(alpha={a}).fit(_A, _t).coef_).max() < 1e-8))",
                 require=["RegressorMixin", "BaseEstimator"], tol=1e-6)
    if task == "binned":
        nb = rng.choice([3, 4, 5])
        sol = (base + "class BinnedMeanRegressor(RegressorMixin, BaseEstimator):\n    def __init__(self, n_bins=4):\n"
               "        self.n_bins = n_bins\n\n    def fit(self, X, y):\n        X, y = validate_data(self, X, y)\n"
               "        self.edges_ = np.quantile(X[:, 0], np.linspace(0, 1, self.n_bins + 1))\n"
               "        b = np.clip(np.searchsorted(self.edges_, X[:, 0], side='right') - 1, 0, self.n_bins - 1)\n"
               "        self.means_ = np.array([y[b == k].mean() if (b == k).any() else y.mean() for k in range(self.n_bins)])\n"
               "        return self\n\n    def predict(self, X):\n        check_is_fitted(self)\n"
               "        X = validate_data(self, X, reset=False)\n"
               "        b = np.clip(np.searchsorted(self.edges_, X[:, 0], side='right') - 1, 0, self.n_bins - 1)\n"
               "        return self.means_[b]\n")
        return P(f"Write `BinnedMeanRegressor(RegressorMixin, BaseEstimator)` with `n_bins=4`. It only looks at the "
                 f"first column: `fit` stores the bin edges `edges_ = np.quantile(X[:, 0], np.linspace(0, 1, n_bins + "
                 f"1))`, assigns each row to bin `clip(searchsorted(edges_, x, side='right') - 1, 0, n_bins - 1)`, and "
                 f"stores the mean target of each bin in `means_` (the overall mean for an empty bin); `predict` "
                 f"returns the mean of each row's bin. Checked on fresh data with `n_bins={nb}`: `means_`, "
                 f"`predict(B)` and `score`.", "", sol,
                 post=data + f"_m = BinnedMeanRegressor(n_bins={nb}).fit(_A, _t)\n(_m.means_, _m.predict(_B), "
                             "_m.score(_A, _t))", require=["RegressorMixin", "BaseEstimator"], tol=1e-9)
    k = rng.choice([3, 5, 7])
    sol = (base + "class KNNMeanRegressor(RegressorMixin, BaseEstimator):\n    def __init__(self, k=5):\n"
           "        self.k = k\n\n    def fit(self, X, y):\n        X, y = validate_data(self, X, y)\n"
           "        self.X_, self.y_ = X, y\n        return self\n\n    def predict(self, X):\n"
           "        check_is_fitted(self)\n        X = validate_data(self, X, reset=False)\n"
           "        d = ((X[:, None, :] - self.X_[None]) ** 2).sum(axis=2)\n"
           "        idx = np.argsort(d, axis=1, kind='stable')[:, :self.k]\n        return self.y_[idx].mean(axis=1)\n")
    return P(f"Write `KNNMeanRegressor(RegressorMixin, BaseEstimator)` with `k=5`: `fit` stores the training data as "
             f"`X_` and `y_`; `predict` returns, for each row, the mean target of its `k` nearest training rows "
             f"(Euclidean; ties by training order: a stable argsort). It should agree with "
             f"`KNeighborsRegressor(n_neighbors=k)`: the check returns your `predict(B)` with `k={k}` on fresh data and "
             f"the largest gap to scikit-learn's.", "", sol,
             post=data + f"from sklearn.neighbors import KNeighborsRegressor as _K\n_p = KNNMeanRegressor(k={k}).fit(_A, _t)"
                         f".predict(_B)\n(_p, bool(np.abs(_p - _K(n_neighbors={k}).fit(_A, _t).predict(_B)).max() < 1e-9))",
             require=["RegressorMixin", "BaseEstimator"], tol=1e-9)


@template(M, "medium")
def persistence_workflows(rng):
    t = table(rng, n=(100, 200))
    h = t.n * 3 // 4
    model = LR if t.kind == "clf" else "Ridge()"
    mi = LRI if t.kind == "clf" else "from sklearn.linear_model import Ridge\n"
    pipe = (f"make_pipeline(make_column_transformer((StandardScaler(), {t.num!r}), (OneHotEncoder(handle_unknown="
            f"'ignore'), {t.cat!r})), {model})")
    imp = (mi + "from sklearn.compose import make_column_transformer\nfrom sklearn.pipeline import make_pipeline\n"
           "from sklearn.preprocessing import OneHotEncoder, StandardScaler\nimport joblib\n"
           f"model = {pipe}.fit(df[:{h}], y[:{h}])\n")
    items = [
        ("dump", "Save `model` to `'model.joblib'`, reload it and return the reloaded model's score on the remaining "
                 "rows (DataFrame input: the column names travel with the model).",
         imp + f"joblib.dump(model, 'model.joblib')\nresult = joblib.load('model.joblib').score(df[{h}:], y[{h}:])",
         {"require": ["dump", "load"]}),
        ("load", "A saved model must reproduce its predictions exactly. Save and reload `model`, then return the largest "
                 "absolute difference between the original and reloaded predictions on the remaining rows, and the "
                 "first 3 reloaded predictions, as a tuple.",
         imp + f"joblib.dump(model, 'model.joblib')\nloaded = joblib.load('model.joblib')\n"
               f"a, b = model.predict(df[{h}:]), loaded.predict(df[{h}:])\n"
               "result = (float(np.abs(a.astype(float) - b.astype(float)).max()), b[:3])"),
        ("__version__", "Unpickling needs the same scikit-learn version. Save a bundle `{'model': model, 'sklearn': "
                        "sklearn.__version__, 'columns': list(df.columns)}`, reload it, and return `(bundle['sklearn'] "
                        "== sklearn.__version__, bundle['columns'])`.",
         "import sklearn\n" + imp + "joblib.dump({'model': model, 'sklearn': sklearn.__version__, 'columns': "
                                    "list(df.columns)}, 'bundle.joblib')\nbundle = joblib.load('bundle.joblib')\n"
                                    "result = (bundle['sklearn'] == sklearn.__version__, bundle['columns'])"),
        ("named_steps", "Reload `model` from a file and inspect it: return the categories the reloaded one-hot encoder "
                        "learned (`named_steps['columntransformer'].named_transformers_['onehotencoder'].categories_`, as "
                        "a list of lists).",
         imp + "joblib.dump(model, 'model.joblib')\nloaded = joblib.load('model.joblib')\n"
               "enc = loaded.named_steps['columntransformer'].named_transformers_['onehotencoder']\n"
               "result = [list(c) for c in enc.categories_]"),
        ("dump", "Save several models in one file: fit two more copies of the pipeline on the first and second halves of "
                 f"the training rows, dump the list `[model, half1, half2]`, reload it, and return the three scores on "
                 f"the remaining rows.",
         imp + f"half1 = {pipe}.fit(df[:{h // 2}], y[:{h // 2}])\nhalf2 = {pipe}.fit(df[{h // 2}:{h}], y[{h // 2}:{h}])\n"
               "joblib.dump([model, half1, half2], 'models.joblib')\n"
               f"result = [m.score(df[{h}:], y[{h}:]) for m in joblib.load('models.joblib')]"),
    ]
    return catalog(rng, items, t.setup, f"{t.desc}. `model` will be `{pipe}` fitted on the first {h} rows. Your code "
                                        f"runs in an empty temporary folder.", tol=1e-6)


@template(M, "medium")
def weights_in_models(rng):
    d = synth_clf(rng, n=(200, 400), d=(4, 8), classes=2, imbalance=rng.choice([0.08, 0.12, 0.2]))
    h = d.n * 2 // 3
    cu = "from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight\n"
    items = [
        ("compute_sample_weight", "Not every model has `class_weight`. Train `GradientBoostingClassifier(n_estimators=50, "
                                  "random_state=0)` on the first rows with and without `sample_weight=compute_sample_"
                                  f"weight('balanced', y[:{h}])` and return the test recall of class 1 of each, as a "
                                  f"tuple (weighted first).",
         cu + "from sklearn.ensemble import GradientBoostingClassifier\nfrom sklearn.metrics import recall_score\n"
              f"w = compute_sample_weight('balanced', y[:{h}])\n"
              f"a = GradientBoostingClassifier(n_estimators=50, random_state=0).fit(X[:{h}], y[:{h}], sample_weight=w)\n"
              f"b = GradientBoostingClassifier(n_estimators=50, random_state=0).fit(X[:{h}], y[:{h}])\n"
              f"result = (recall_score(y[{h}:], a.predict(X[{h}:])), recall_score(y[{h}:], b.predict(X[{h}:])))"),
        ("class_weight", f"For a decision tree, `class_weight='balanced'` and balanced sample weights are the same thing. "
                         f"Fit `DecisionTreeClassifier(max_depth=4, random_state=0)` both ways on the first {h} rows and "
                         f"return whether their test predictions agree everywhere, and the test balanced accuracy.",
         cu + "from sklearn.metrics import balanced_accuracy_score\nfrom sklearn.tree import DecisionTreeClassifier\n"
              f"a = DecisionTreeClassifier(max_depth=4, random_state=0, class_weight='balanced').fit(X[:{h}], y[:{h}])\n"
              f"b = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X[:{h}], y[:{h}], "
              f"sample_weight=compute_sample_weight('balanced', y[:{h}]))\n"
              f"result = (bool((a.predict(X[{h}:]) == b.predict(X[{h}:])).all()), "
              f"balanced_accuracy_score(y[{h}:], a.predict(X[{h}:])))"),
        ("compute_class_weight", f"Soften the balancing: use the **square root** of the balanced class weights as a "
                                 f"`class_weight` dict for `{LR}` on the first {h} rows and return the test recall of "
                                 f"class 1.",
         cu + LRI + "from sklearn.metrics import recall_score\ncls = np.unique(y)\n"
              f"w = np.sqrt(compute_class_weight('balanced', classes=cls, y=y[:{h}]))\n"
              f"m = LogisticRegression(max_iter=1000, class_weight=dict(zip(cls.tolist(), w))).fit(X[:{h}], y[:{h}])\n"
              f"result = recall_score(y[{h}:], m.predict(X[{h}:]))"),
        ("sample_weight", f"`LogisticRegression(class_weight='balanced')` equals fitting with balanced `sample_weight`. "
                          f"Fit `{LR}` with `sample_weight=compute_sample_weight('balanced', y)` on all rows and return "
                          f"its `coef_` (it matches the `class_weight='balanced'` model up to solver tolerance).",
         cu + LRI + f"result = {LR}.fit(X, y, sample_weight=compute_sample_weight('balanced', y)).coef_", {"tol": 1e-4}),
        ("compute_sample_weight", "Weight a subset only: `compute_sample_weight('balanced', y, indices=...)` balances "
                                  "using just those rows. Return the weights for the first 5 rows with `indices="
                                  "np.arange(100)`.",
         cu + "result = compute_sample_weight('balanced', y, indices=np.arange(100))[:5]"),
    ]
    return catalog(rng, items, d.setup, f"{d.desc}; class 1 is rare. The first {h} rows are for training, the rest "
                                        f"for testing.", tol=1e-6)


@template(M, "medium")
def feature_names_custom(rng):
    s = seeded(rng)
    task = rng.choice(["one_to_one", "prefix", "set_output"])
    cols = rng.sample(["alpha", "beta", "gamma", "delta", "cpu", "mem", "disk", "temp", "wind", "rain"], 3)
    n = rng.randint(5, 8)
    setup = f"D = pd.DataFrame(np.random.default_rng({s}).normal(0, 3, ({n}, 3)).round(2), columns={cols})"
    data = ""
    if task == "one_to_one":
        sol = ("from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin\n" + CHECKS +
               "\n\nclass Abs(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):\n"
               "    def fit(self, X, y=None):\n        validate_data(self, X)\n        return self\n\n"
               "    def transform(self, X):\n        check_is_fitted(self)\n        X = validate_data(self, X, reset=False)\n"
               "        return np.abs(X)\n")
        return P("Write `Abs(OneToOneFeatureMixin, TransformerMixin, BaseEstimator)` (all from `sklearn.base`): `fit` "
                 "validates `X` and returns `self`; `transform` returns `np.abs(X)`. `OneToOneFeatureMixin` adds a "
                 "`get_feature_names_out` that keeps the input names. Checked on the DataFrame `D`: "
                 "`get_feature_names_out()` and the transformed values.", setup, sol,
                 post=data + "_t = Abs().fit(D)\n(_t.get_feature_names_out(), _t.transform(D))",
                 require=["OneToOneFeatureMixin", "validate_data"])
    if task == "prefix":
        sol = ("from sklearn.base import BaseEstimator, ClassNamePrefixFeaturesOutMixin, TransformerMixin\n" + CHECKS +
               "\n\nclass RowStats(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):\n"
               "    def fit(self, X, y=None):\n        validate_data(self, X)\n        self._n_features_out = 3\n"
               "        return self\n\n    def transform(self, X):\n        check_is_fitted(self)\n"
               "        X = validate_data(self, X, reset=False)\n"
               "        return np.column_stack([X.min(axis=1), X.mean(axis=1), X.max(axis=1)])\n")
        return P("Write `RowStats(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator)`: `transform` "
                 "returns 3 columns per row (min, mean, max). `ClassNamePrefixFeaturesOutMixin` names the outputs "
                 "`rowstats0, rowstats1, ...` if `fit` sets `self._n_features_out = 3`. Checked on the DataFrame `D`: "
                 "`get_feature_names_out()` and the transformed values.", setup, sol,
                 post=data + "_t = RowStats().fit(D)\n(_t.get_feature_names_out(), _t.transform(D))",
                 require=["ClassNamePrefixFeaturesOutMixin", "TransformerMixin"])
    sol = ("from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin\n" + CHECKS +
           "\n\nclass Demean(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):\n"
           "    def fit(self, X, y=None):\n        X = validate_data(self, X)\n        self.mean_ = X.mean(axis=0)\n"
           "        return self\n\n    def transform(self, X):\n        check_is_fitted(self)\n"
           "        X = validate_data(self, X, reset=False)\n        return X - self.mean_\n")
    return P("Write `Demean(OneToOneFeatureMixin, TransformerMixin, BaseEstimator)`: `fit` stores the column means in "
             "`mean_`, `transform` subtracts them. Because it has `get_feature_names_out`, `TransformerMixin` lets it "
             "return DataFrames: the check runs `Demean().set_output(transform='pandas').fit_transform(D)` on the "
             "DataFrame `D`, and the same inside `config_context(transform_output='pandas')`.", setup, sol,
             post=data + "from sklearn import config_context as _cc\n_a = Demean().set_output(transform='pandas')"
                         ".fit_transform(D)\nwith _cc(transform_output='pandas'):\n    _b = Demean().fit_transform(D)\n"
                         "(_a, list(_b.columns))", require=["OneToOneFeatureMixin", "TransformerMixin"])


@template(M, "medium")
def custom_transformer_usage(rng):
    t = table(rng, numeric_only=True, n=(80, 160))
    q, q2 = rng.sample([0.01, 0.05, 0.1], 2)
    j = rng.randrange(len(t.num))
    cols = sorted(rng.sample(range(len(t.num)), 2))
    model = LR if t.kind == "clf" else "Ridge()"
    mi = LRI if t.kind == "clf" else "from sklearn.linear_model import Ridge\n"
    setup = (t.setup + "\nX = df.to_numpy(dtype=float)\n" + CLIPPER.replace("\n\n\n", "\n\n") +
             LOG1P.split("\n\n\n", 1)[1])
    sp = "from sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\n"
    pipe = f"make_pipeline(Clipper({q}), StandardScaler(), {model})"
    items = [
        ("make_column_transformer", f"Clip only columns {cols}: `make_column_transformer((Clipper({q}), {cols}), "
                                    f"remainder='passthrough')`. Fit-transform `X` and return the first row (the clipped "
                                    f"columns come first).",
         f"from sklearn.compose import make_column_transformer\n"
         f"ct = make_column_transformer((Clipper({q}), {cols}), remainder='passthrough')\nresult = ct.fit_transform(X)[0]"),
        ("FunctionTransformer", f"A stateless function needs no class: `FunctionTransformer(np.log1p)` (from "
                                f"`sklearn.preprocessing`). Return `(whether Log1pColumns(columns=[{j}]).fit_transform(X)"
                                f"[:, -1] equals FunctionTransformer(np.log1p).fit_transform(X[:, [{j}]]).ravel(), the "
                                f"first 3 values)`.",
         f"from sklearn.preprocessing import FunctionTransformer\na = Log1pColumns(columns=[{j}]).fit_transform(X)[:, -1]\n"
         f"b = FunctionTransformer(np.log1p).fit_transform(X[:, [{j}]]).ravel()\nresult = (bool(np.allclose(a, b)), b[:3])"),
        ("cross_validate", f"Each fold fits its own clipper. Run `cross_validate({pipe}, X, y, cv=3, "
                           f"return_estimator=True)` and return the clipper's `high_[0]` from each fold's estimator, as a "
                           f"list.",
         mi + sp + "from sklearn.model_selection import cross_validate\n"
         f"cv = cross_validate({pipe}, X, y, cv=3, return_estimator=True)\n"
         "result = [float(e[0].high_[0]) for e in cv['estimator']]", {"require": ["cross_validate", "return_estimator"]}),
        ("cross_val_score", f"Do the custom steps help? Return the mean 5-fold `cross_val_score` of `{pipe}` and of "
                            f"`make_pipeline(Log1pColumns(columns={cols}), StandardScaler(), {model})`, as a tuple.",
         mi + sp + "from sklearn.model_selection import cross_val_score\n"
         f"a = cross_val_score({pipe}, X, y, cv=5).mean()\n"
         f"b = cross_val_score(make_pipeline(Log1pColumns(columns={cols}), StandardScaler(), {model}), X, y, cv=5).mean()\n"
         "result = (a, b)"),
        ("set_params", f"Fit `pipe = {pipe}` on `X, y`, then change the clipper with `pipe.set_params(clipper__quantile="
                       f"{q2})`, refit, and return the clipper's `low_` before and after, as a tuple.",
         mi + sp + f"pipe = {pipe}.fit(X, y)\nbefore = pipe[0].low_.copy()\n"
         f"after = pipe.set_params(clipper__quantile={q2}).fit(X, y)[0].low_\nresult = (before, after)"),
        ("clone", f"Fit `pipe = {pipe}` on `X, y`; `clone(pipe)` gives an unfitted copy with the same settings. Return "
                  f"`(hasattr(clone(pipe)[0], 'low_'), clone(pipe)[0].quantile, pipe.score(X, y))`.",
         mi + sp + f"from sklearn.base import clone\npipe = {pipe}.fit(X, y)\nk = clone(pipe)\n"
         "result = (hasattr(k[0], 'low_'), k[0].quantile, pipe.score(X, y))"),
    ]
    return catalog(rng, items, setup, f"{t.desc}; `X` is `df` as a float array. The guide's `Clipper` and the "
                                      f"`Log1pColumns(columns=...)` transformer (appends `np.log1p` copies of the chosen "
                                      f"columns) are defined above.", tol=1e-6)


# ================================================================= hard

@template(M, "hard")
def fix_broken_estimator(rng):
    s = seeded(rng)
    data = _fresh(s, n=30, d=2, kind="normal")
    task = rng.choice(["init_logic", "no_return", "no_underscore", "mixin_order", "refit_in_transform"])
    if task == "init_logic":
        return P("This transformer validates its parameter in `__init__`:\n```\nclass Shrink(TransformerMixin, "
                 "BaseEstimator):\n    def __init__(self, factor=0.5):\n        self.factor = min(max(factor, 0.0), "
                 "1.0)\n    def fit(self, X, y=None):\n        return self\n    def transform(self, X):\n        return X "
                 "* self.factor\n```\n`set_params` bypasses `__init__`, so `Shrink().set_params(factor=2.0)` keeps 2.0 "
                 "while `Shrink(factor=2.0)` gets 1.0, and grid searches and clones disagree. Fix it: store `factor` "
                 "unchanged and do the clipping in `fit` (store it as `factor_`), using it in `transform`. Checked with "
                 "both spellings on fresh data and with `get_params()`.", "",
                 BASE + "\n\nclass Shrink(TransformerMixin, BaseEstimator):\n    def __init__(self, factor=0.5):\n"
                        "        self.factor = factor\n\n    def fit(self, X, y=None):\n"
                        "        self.factor_ = min(max(self.factor, 0.0), 1.0)\n        return self\n\n"
                        "    def transform(self, X):\n        return X * self.factor_\n",
                 post=data + "(Shrink(factor=2.0).fit_transform(_A), Shrink().set_params(factor=2.0).fit_transform(_A), "
                             "Shrink(factor=2.0).get_params())", require=["TransformerMixin", "BaseEstimator"])
    if task == "no_return":
        return P("This scaler forgets something:\n```\nclass MaxScaler(TransformerMixin, BaseEstimator):\n    def fit(self, "
                 "X, y=None):\n        self.max_ = np.abs(X).max(axis=0)\n    def transform(self, X):\n        return X "
                 "/ self.max_\n```\n`MaxScaler().fit(X).transform(X)` crashes and it breaks inside pipelines. Fix it "
                 "(`fit` must return `self`) and add `check_is_fitted` to `transform`. Checked with "
                 "`MaxScaler().fit(A).transform(B)` on fresh data, inside `make_pipeline(MaxScaler())`, and on an "
                 "unfitted instance.", "",
                 BASE + "from sklearn.utils.validation import check_is_fitted\n\n\n"
                        "class MaxScaler(TransformerMixin, BaseEstimator):\n    def fit(self, X, y=None):\n"
                        "        self.max_ = np.abs(X).max(axis=0)\n        return self\n\n    def transform(self, X):\n"
                        "        check_is_fitted(self)\n        return X / self.max_\n",
                 post=data + "from sklearn.pipeline import make_pipeline as _mp\n" + _err("MaxScaler().transform(_B)") +
                 "(MaxScaler().fit(_A).transform(_B), _mp(MaxScaler()).fit(_A).transform(_B), _r)",
                 require=["check_is_fitted"])
    if task == "no_underscore":
        return P("`check_is_fitted` decides that an estimator is fitted by looking for attributes ending in `_`. This "
                 "one stores `self.median` in `fit`, so `check_is_fitted` always says 'not fitted':\n```\nclass "
                 "MedianImputer(TransformerMixin, BaseEstimator):\n    def fit(self, X, y=None):\n        self.median = "
                 "np.nanmedian(X, axis=0)\n        return self\n    def transform(self, X):\n        check_is_fitted("
                 "self)\n        return np.where(np.isnan(X), self.median, X)\n```\nFix it (rename to `median_`). "
                 "Checked on fresh data with missing values and on an unfitted instance.", "",
                 BASE + "from sklearn.utils.validation import check_is_fitted\n\n\n"
                        "class MedianImputer(TransformerMixin, BaseEstimator):\n    def fit(self, X, y=None):\n"
                        "        self.median_ = np.nanmedian(X, axis=0)\n        return self\n\n    def transform(self, X):\n"
                        "        check_is_fitted(self)\n        return np.where(np.isnan(X), self.median_, X)\n",
                 post=data + "_A[::4, 0] = np.nan\n_B[::3, 1] = np.nan\n" + _err("MedianImputer().transform(_B)") +
                 "_m = MedianImputer().fit(_A)\n(_m.transform(_B), _m.median_, _r)", require=["check_is_fitted"])
    if task == "mixin_order":
        return P("Mixins must come **before** `BaseEstimator`: written as `class Doubler(BaseEstimator, "
                 "TransformerMixin)`, scikit-learn's tags no longer say it is a transformer (`get_tags(Doubler())."
                 "transformer_tags` is `None`; `get_tags` is in `sklearn.utils`). Write `Doubler` correctly (`fit` "
                 "returns `self`, `transform` returns `2 * X`). Checked with `get_tags(Doubler()).transformer_tags is "
                 "not None` and `fit_transform` on fresh data.", "",
                 BASE + "\n\nclass Doubler(TransformerMixin, BaseEstimator):\n    def fit(self, X, y=None):\n"
                        "        return self\n\n    def transform(self, X):\n        return 2 * X\n",
                 post=data + "from sklearn.utils import get_tags as _gt\n"
                             "(_gt(Doubler()).transformer_tags is not None, Doubler().fit_transform(_A))",
                 require=["TransformerMixin", "BaseEstimator"])
    return P("This standardizer leaks: it recomputes the statistics on whatever it transforms, so test data is "
             "scaled with its own mean and std:\n```\nclass Standardize(TransformerMixin, BaseEstimator):\n    def "
             "fit(self, X, y=None):\n        return self\n    def transform(self, X):\n        return (X - X.mean(axis=0)) "
             "/ X.std(axis=0)\n```\nFix it: learn `mean_` and `std_` in `fit` and only apply them in `transform`. "
             "Checked with `Standardize().fit(A).transform(B)` on fresh data (it must match `StandardScaler`).", "",
             BASE + "\n\nclass Standardize(TransformerMixin, BaseEstimator):\n    def fit(self, X, y=None):\n"
                    "        self.mean_, self.std_ = X.mean(axis=0), X.std(axis=0)\n        return self\n\n"
                    "    def transform(self, X):\n        return (X - self.mean_) / self.std_\n",
             post=data + "from sklearn.preprocessing import StandardScaler as _SS\n_s = Standardize().fit(_A)\n"
                         "(_s.transform(_B), bool(np.allclose(_s.transform(_B), _SS().fit(_A).transform(_B))))",
             require=["TransformerMixin"])


ENCODER = (BASE + CHECKS + "\n\nclass TargetMeanEncoder(TransformerMixin, BaseEstimator):\n"
           "    def __init__(self, smoothing=10.0):\n        self.smoothing = smoothing\n\n"
           "    def fit(self, X, y):\n        X = np.asarray(X, dtype=object)\n        y = np.asarray(y, dtype=float)\n"
           "        self.n_features_in_ = X.shape[1]\n        self.prior_ = y.mean()\n        self.maps_ = []\n"
           "        for j in range(X.shape[1]):\n            m = {}\n            for c in np.unique(X[:, j]):\n"
           "                sel = X[:, j] == c\n"
           "                m[c] = (y[sel].sum() + self.smoothing * self.prior_) / (sel.sum() + self.smoothing)\n"
           "            self.maps_.append(m)\n        return self\n\n"
           "    def transform(self, X):\n        check_is_fitted(self)\n        X = np.asarray(X, dtype=object)\n"
           "        return np.column_stack([[self.maps_[j].get(v, self.prior_) for v in X[:, j]] "
           "for j in range(X.shape[1])])\n")


@template(M, "hard")
def target_mean_encoder(rng):
    t = table(rng, kind="clf", n=(120, 240))
    m = rng.choice([1.0, 5.0, 20.0])
    s = seeded(rng)
    task = rng.choice(["values", "unseen", "pipeline"])
    head = (f"Write `TargetMeanEncoder(TransformerMixin, BaseEstimator)` with `smoothing=10.0`: `fit(X, y)` stores "
            f"the overall target mean `prior_` and, for every column and category c, the smoothed mean `(sum of y in c "
            f"+ smoothing·prior_) / (count of c + smoothing)` (store them in `maps_`, one dict per column); "
            f"`transform(X)` checks it is fitted and replaces each category by its encoding (`prior_` for a category "
            f"never seen in `fit`). Work on `X` as a 2-D array of strings (`np.asarray(X, dtype=object)`).")
    cols = t.cat
    data = (f"_g = np.random.default_rng({s})\n_C = _g.choice(['red', 'green', 'blue'], (40, 2))\n"
            "_t = ((_C[:, 0] == 'red') ^ (_g.random(40) < 0.3)).astype(int)\n")
    req = ["TransformerMixin", "check_is_fitted"]
    if task == "values":
        return P(f"{head} Checked on fresh data with `smoothing={m}`: `prior_` and the transformed training rows.", "",
                 ENCODER, post=data + f"_e = TargetMeanEncoder(smoothing={m}).fit(_C, _t)\n(_e.prior_, "
                                      "_e.transform(_C[:6]))", require=req)
    if task == "unseen":
        return P(f"{head} Checked with `smoothing={m}` on fresh data whose new rows contain a colour never seen in "
                 f"`fit` (it must map to `prior_`), and on an unfitted encoder (`NotFittedError`).", "", ENCODER,
                 post=data + _err("TargetMeanEncoder().transform(_C)") +
                 f"_e = TargetMeanEncoder(smoothing={m}).fit(_C, _t)\n"
                 "(_e.transform(np.array([['red', 'pink'], ['violet', 'blue']], dtype=object)), _r)", require=req)
    return P(f"{head} Then encode the categorical columns `{cols}` of `df` inside a pipeline so the encoding is learned "
             f"within each fold: `make_pipeline(make_column_transformer((TargetMeanEncoder(smoothing={m}), {cols!r}), "
             f"remainder=StandardScaler()), {LR})`. Return its mean 5-fold `cross_val_score` on `df, y`.", t.setup,
             ENCODER + LRI + "from sklearn.compose import make_column_transformer\n"
             "from sklearn.model_selection import cross_val_score\nfrom sklearn.pipeline import make_pipeline\n"
             "from sklearn.preprocessing import StandardScaler\n\n"
             f"pipe = make_pipeline(make_column_transformer((TargetMeanEncoder(smoothing={m}), {cols!r}), "
             f"remainder=StandardScaler()), {LR})\nresult = cross_val_score(pipe, df, y, cv=5).mean()",
             require=req + ["cross_val_score"], tol=1e-6)


@template(M, "hard")
def custom_estimator_in_search(rng):
    d = synth_clf(rng, n=(150, 300), d=(4, 8), classes=(2, 3))
    shrinks = sorted(rng.sample([0.0, 0.2, 0.4, 0.6, 0.8], 3))
    task = rng.choice(["best", "clone", "scores"])
    sol = CENTROID.replace("    def fit(self, X, y):", "    def __init__(self, shrink=0.0):\n        self.shrink = shrink\n\n"
                                                       "    def fit(self, X, y):")
    sol = sol.replace("        self.centroids_ = np.array([X[y == c].mean(axis=0) for c in self.classes_])\n",
                      "        mu = X.mean(axis=0)\n        self.centroids_ = np.array([(1 - self.shrink) * X[y == c].mean("
                      "axis=0) + self.shrink * mu for c in self.classes_])\n")
    head = (f"{d.desc}. Write `CentroidClassifier(ClassifierMixin, BaseEstimator)` with a `shrink=0.0` hyperparameter: "
            f"`fit` stores `classes_` and per-class centroids `(1 - shrink)·class mean + shrink·overall mean` in "
            f"`centroids_`; `predict` returns the class of the nearest centroid (Euclidean).")
    gs = (f"from sklearn.model_selection import GridSearchCV\nfrom sklearn.pipeline import make_pipeline\n"
          f"from sklearn.preprocessing import StandardScaler\n\n"
          f"gs = GridSearchCV(make_pipeline(StandardScaler(), CentroidClassifier()), "
          f"{{'centroidclassifier__shrink': {shrinks}}}, cv=5).fit(X, y)\n")
    if task == "best":
        return P(f"{head} Tune `shrink` in `{shrinks}` with `GridSearchCV(make_pipeline(StandardScaler(), "
                 f"CentroidClassifier()), ..., cv=5)` on `X, y` and return `(best_params_, best_score_)`.", d.setup,
                 sol + gs + "result = (gs.best_params_, gs.best_score_)", require=["GridSearchCV", "best_params_"],
                 tol=1e-9)
    if task == "scores":
        return P(f"{head} Tune `shrink` in `{shrinks}` with `GridSearchCV(make_pipeline(StandardScaler(), "
                 f"CentroidClassifier()), ..., cv=5)` on `X, y` and return `cv_results_['mean_test_score']`.", d.setup,
                 sol + gs + "result = gs.cv_results_['mean_test_score']", require=["GridSearchCV"], tol=1e-9)
    return P(f"{head} Grid search relies on `clone`, which rebuilds an estimator from `get_params()`. Return "
             f"`(clone(CentroidClassifier(shrink={shrinks[1]})).shrink, hasattr(clone(fitted), 'centroids_'))` where "
             f"`fitted = CentroidClassifier().fit(X, y)` (a clone is never fitted), plus the fitted model's training "
             f"accuracy.", d.setup,
             sol + f"from sklearn.base import clone\n\nfitted = CentroidClassifier().fit(X, y)\n"
                   f"result = (clone(CentroidClassifier(shrink={shrinks[1]})).shrink, hasattr(clone(fitted), "
                   "'centroids_'), fitted.score(X, y))", require=["clone"], tol=1e-9)


@template(M, "hard")
def bootstrap_evaluation(rng):
    d = synth_clf(rng, n=(150, 300), d=(4, 8), classes=2)
    b = rng.choice([20, 30, 40])
    h = d.n * 2 // 3
    task = rng.choice(["ci", "oob", "coef"])
    ru = "from sklearn.utils import resample\n"
    if task == "ci":
        return P(f"{d.desc}. Train `{LR}` on the first {h} rows. Bootstrap its **test** accuracy: for i = 0..{b - 1}, "
                 f"resample the test rows with `resample(X_te, y_te, random_state=i)` and score the (fixed) model on "
                 f"each resample. Return the 2.5th and 97.5th percentiles (`np.percentile`) of the scores as a tuple.",
                 d.setup + f"\nX_te, y_te = X[{h}:], y[{h}:]",
                 ru + LRI + f"clf = {LR}.fit(X[:{h}], y[:{h}])\n"
                 f"scores = [clf.score(*resample(X_te, y_te, random_state=i)) for i in range({b})]\n"
                 "result = (float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5)))",
                 require=["resample"], tol=1e-9)
    if task == "oob":
        return P(f"{d.desc}. Out-of-bag evaluation by hand: for i = 0..{b - 1}, draw bootstrap row numbers `idx = "
                 f"resample(np.arange(len(y)), random_state=i)`, fit `{LR}` on those rows and predict the rows **not** "
                 f"drawn. Each row's OOB prediction is the majority vote of the models that did not see it (ties: "
                 f"class 1). Return the OOB accuracy over rows that were left out at least once.", d.setup,
                 ru + LRI + f"votes = np.zeros((len(y), 2))\nfor i in range({b}):\n"
                 "    idx = resample(np.arange(len(y)), random_state=i)\n    out = np.setdiff1d(np.arange(len(y)), idx)\n"
                 f"    p = {LR}.fit(X[idx], y[idx]).predict(X[out])\n    votes[out, p] += 1\n"
                 "seen = votes.sum(axis=1) > 0\npred = (votes[:, 1] >= votes[:, 0]).astype(int)\n"
                 "result = float((pred[seen] == y[seen]).mean())", require=["resample"], tol=1e-9)
    return P(f"{d.desc}. How stable are the coefficients? For i = 0..{b - 1}, fit `{LR}` on `resample(X, y, "
             f"stratify=y, random_state=i)` and return the standard deviation of every coefficient across the fits "
             f"(an array, `coef_[0]` per fit).", d.setup,
             ru + LRI + f"C = np.array([{LR}.fit(*resample(X, y, stratify=y, random_state=i)).coef_[0] for i in "
                        f"range({b})])\nresult = C.std(axis=0)", require=["resample", "stratify"], tol=1e-5)


@template(M, "hard")
def config_and_validation(rng):
    t = table(rng, numeric_only=True, n=(60, 120))
    col = rng.choice(t.num)
    items = [
        ("assume_finite", "Data with an infinity sneaks past validation when `assume_finite=True`. Put `np.inf` in row 0 "
                          f"of `{col}` (in a copy of `df`). Return `(name of the exception StandardScaler().fit raises "
                          "normally, whether the mean_ learned inside config_context(assume_finite=True) is finite for "
                          f"`{col}`)`.",
         "from sklearn import config_context\nfrom sklearn.preprocessing import StandardScaler\nbad = df.astype(float)\n"
         f"bad.loc[bad.index[0], {col!r}] = np.inf\ntry:\n    StandardScaler().fit(bad)\n    err = 'no error'\n"
         "except Exception as e:\n    err = type(e).__name__\nwith config_context(assume_finite=True), np.errstate(invalid='ignore'):\n"
         f"    m = StandardScaler().fit(bad).mean_[list(df.columns).index({col!r})]\nresult = (err, bool(np.isfinite(m)))"),
        ("skip_parameter_validation", "scikit-learn validates hyperparameters at `fit`: `StandardScaler(with_mean="
                                      "'yes')` raises an `InvalidParameterError`. Return `(name of the exception it raises "
                                      "on fit, name of the exception (or 'no error') inside config_context("
                                      "skip_parameter_validation=True))`.",
         "from sklearn import config_context\nfrom sklearn.preprocessing import StandardScaler\nnames = []\n"
         "for skip in (False, True):\n    with config_context(skip_parameter_validation=skip):\n        try:\n"
         "            StandardScaler(with_mean='yes').fit(df)\n            names.append('no error')\n"
         "        except Exception as e:\n            names.append(type(e).__name__)\nresult = tuple(names)"),
        ("check_is_fitted", "`check_is_fitted` works on any estimator. Return the list of exception names (or 'ok') it "
                            "gives for: an unfitted `StandardScaler()`, a fitted one, an unfitted `make_pipeline("
                            "StandardScaler())`, and that pipeline after fitting on `df`.",
         "from sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\n"
         "from sklearn.utils.validation import check_is_fitted\npipe = make_pipeline(StandardScaler())\n"
         "objs = [StandardScaler(), StandardScaler().fit(df), pipe]\nout = []\n"
         "for o in objs + ['refit']:\n    o = pipe.fit(df) if isinstance(o, str) else o\n    try:\n"
         "        check_is_fitted(o)\n        out.append('ok')\n    except Exception as e:\n"
         "        out.append(type(e).__name__)\nresult = out"),
        ("transform_output", "Mixing settings: inside `config_context(transform_output='pandas')`, fit-transform `df` "
                             "with `make_pipeline(StandardScaler(), PCA(n_components=2))` and return the output "
                             "DataFrame's columns and the first row, as a tuple.",
         "from sklearn import config_context\nfrom sklearn.decomposition import PCA\nfrom sklearn.pipeline import make_pipeline\n"
         "from sklearn.preprocessing import StandardScaler\nwith config_context(transform_output='pandas'):\n"
         "    out = make_pipeline(StandardScaler(), PCA(n_components=2)).fit_transform(df)\n"
         "result = (list(out.columns), out.iloc[0].to_numpy())"),
    ]
    return catalog(rng, items, t.setup, f"{t.desc}.", post=RESET + "_", tol=1e-6)


@template(M, "hard")
def pipeline_with_custom_steps(rng):
    s = seeded(rng)
    qs = sorted(rng.sample([0.0, 0.01, 0.05, 0.1], 3))
    cs = sorted(rng.sample([0.1, 1.0, 10.0], 2))
    q = rng.choice(qs[1:])
    k = rng.randint(8, 15)
    setup = (f"from sklearn.datasets import load_breast_cancer\nX, y = load_breast_cancer(return_X_y=True)\n"
             f"rows = np.random.default_rng({s}).permutation(len(X))[:300]\nX, y = X[rows], y[rows]\n" +
             CLIPPER.replace("\n\n\n", "\n\n"))
    sp = (LRI + "from sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\n")
    pipe = f"make_pipeline(Clipper({q}), StandardScaler(), {LR})"
    items = [
        ("GridSearchCV", f"Tune the custom step and the model together: `GridSearchCV(make_pipeline(Clipper(), "
                         f"StandardScaler(), {LR}), {{'clipper__quantile': {qs}, 'logisticregression__C': {cs}}}, "
                         f"cv=3)` on `X, y`. Return `best_params_`.",
         sp + "from sklearn.model_selection import GridSearchCV\n"
         f"gs = GridSearchCV(make_pipeline(Clipper(), StandardScaler(), {LR}), {{'clipper__quantile': {qs}, "
         f"'logisticregression__C': {cs}}}, cv=3).fit(X, y)\nresult = gs.best_params_", {"require": ["GridSearchCV"]}),
        ("compute_sample_weight", f"Fit parameters are routed by step name: fit `{pipe}` with "
                                  f"`logisticregression__sample_weight=compute_sample_weight('balanced', y)` and return "
                                  f"its training `balanced_accuracy_score`.",
         sp + "from sklearn.metrics import balanced_accuracy_score\n"
         "from sklearn.utils.class_weight import compute_sample_weight\n"
         f"pipe = {pipe}.fit(X, y, logisticregression__sample_weight=compute_sample_weight('balanced', y))\n"
         "result = balanced_accuracy_score(y, pipe.predict(X))"),
        ("FunctionTransformer", f"A `FunctionTransformer(np.clip, kw_args=...)` with fixed bounds can't learn them. Fit "
                                f"`Clipper({q})` on the first 200 rows, then build `FunctionTransformer(np.clip, kw_args="
                                f"{{'a_min': c.low_, 'a_max': c.high_}})` and return `(whether both transform the last "
                                f"100 rows identically, the clipped value of row 200, column 0)`.",
         f"from sklearn.preprocessing import FunctionTransformer\nc = Clipper({q}).fit(X[:200])\n"
         "f = FunctionTransformer(np.clip, kw_args={'a_min': c.low_, 'a_max': c.high_})\n"
         "a, b = c.transform(X[200:]), f.fit_transform(X[200:])\nresult = (bool(np.array_equal(a, b)), float(a[0, 0]))"),
        ("cross_validate", f"Inspect the fold models: `cross_validate({pipe}, X, y, cv=4, return_estimator=True)`. "
                           f"Return each fold's clipper `low_[0]` as a list, and the mean test score.",
         sp + "from sklearn.model_selection import cross_validate\n"
         f"cv = cross_validate({pipe}, X, y, cv=4, return_estimator=True)\n"
         "result = ([float(e[0].low_[0]) for e in cv['estimator']], cv['test_score'].mean())"),
        ("make_column_transformer", f"Clip only the first {k} columns and pass the rest through: "
                                    f"`make_column_transformer((Clipper({q}), list(range({k}))), remainder='passthrough')`"
                                    f" followed by `StandardScaler()` and `{LR}`. Return its mean 5-fold "
                                    f"`cross_val_score`.",
         sp + "from sklearn.compose import make_column_transformer\nfrom sklearn.model_selection import cross_val_score\n"
         f"ct = make_column_transformer((Clipper({q}), list(range({k}))), remainder='passthrough')\n"
         f"result = cross_val_score(make_pipeline(ct, StandardScaler(), {LR}), X, y, cv=5).mean()"),
        ("pipeline slicing", f"Slicing a fitted pipeline gives its preprocessing alone: fit `pipe = {pipe}` on `X, y` "
                             f"and return `pipe[:-1].transform(X[:5])` (clipped then scaled features) restricted to the "
                             f"first 3 columns.",
         sp + f"pipe = {pipe}.fit(X, y)\nresult = pipe[:-1].transform(X[:5])[:, :3]", {"require": ["transform"]}),
        ("scoring", f"Tune for the metric you care about: `GridSearchCV(make_pipeline(Clipper(), StandardScaler(), "
                    f"{LR}), {{'clipper__quantile': {qs}}}, cv=3, scoring='balanced_accuracy')` on `X, y`. Return "
                    f"`(best_params_, best_score_)`.",
         sp + "from sklearn.model_selection import GridSearchCV\n"
         f"gs = GridSearchCV(make_pipeline(Clipper(), StandardScaler(), {LR}), {{'clipper__quantile': {qs}}}, cv=3, "
         "scoring='balanced_accuracy').fit(X, y)\nresult = (gs.best_params_, gs.best_score_)",
         {"require": ["GridSearchCV", "scoring"]}),
        ("transform", "The clipper checks its input (`validate_data`): fit `Clipper()` on `X` and return the names of the exceptions "
                          "raised by `transform` on `X[:, :5]` (wrong width) and by `fit` on a copy of `X` with a NaN, "
                          "as a tuple.",
         "c = Clipper().fit(X)\nnames = []\nbad = X.copy()\nbad[0, 0] = np.nan\n"
         "for call in (lambda: c.transform(X[:, :5]), lambda: Clipper().fit(bad)):\n    try:\n        call()\n"
         "        names.append('no error')\n    except Exception as e:\n        names.append(type(e).__name__)\n"
         "result = tuple(names)"),
    ]
    return catalog(rng, items, setup, "`X, y` hold 300 shuffled breast-cancer rows (30 raw measurements; 0 malignant, "
                                      "1 benign). The guide's `Clipper` transformer is defined above.", tol=1e-6)


@template(M, "hard")
def utils_in_practice(rng):
    d = synth_clf(rng, n=(200, 360), d=(4, 8), classes=2, imbalance=rng.choice([0.1, 0.15, 0.2]))
    r = rng.randint(0, 9)
    ru = "from sklearn.utils import resample\n"
    items = [
        ("resample", f"Upsample correctly, **inside** cross-validation: with `StratifiedKFold(n_splits=5, shuffle=True, "
                     f"random_state=0)`, in each fold upsample the training rows of class 1 to the size of the training "
                     f"class 0 (`resample(..., replace=True, n_samples=..., random_state={r})`), fit `{LR}` on the "
                     f"balanced training set and record the `balanced_accuracy_score` on the untouched test fold. "
                     f"Return the mean.",
         ru + LRI + "from sklearn.metrics import balanced_accuracy_score\nfrom sklearn.model_selection import StratifiedKFold\n"
         "scores = []\nfor tr, te in StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(X, y):\n"
         "    Xt, yt = X[tr], y[tr]\n"
         f"    Xu, yu = resample(Xt[yt == 1], yt[yt == 1], replace=True, n_samples=int((yt == 0).sum()), random_state={r})\n"
         "    Xb, yb = np.vstack([Xt[yt == 0], Xu]), np.concatenate([yt[yt == 0], yu])\n"
         f"    scores.append(balanced_accuracy_score(y[te], {LR}.fit(Xb, yb).predict(X[te])))\nresult = float(np.mean(scores))"),
        ("resample", f"The classic leak: upsample class 1 **before** splitting. Upsample it to the size of class 0 with "
                     f"`resample(X[y == 1], replace=True, n_samples=(y == 0).sum(), random_state={r})`, stack with the "
                     f"class-0 rows, split with `train_test_split(..., test_size=0.3, random_state=0)`, and return how "
                     f"many test rows are exact copies of some training row.",
         ru + "from sklearn.model_selection import train_test_split\n"
         f"Xu = resample(X[y == 1], replace=True, n_samples=int((y == 0).sum()), random_state={r})\n"
         "Xb = np.vstack([X[y == 0], Xu])\nyb = np.r_[np.zeros((y == 0).sum(), int), np.ones(len(Xu), int)]\n"
         "Xtr, Xte, ytr, yte = train_test_split(Xb, yb, test_size=0.3, random_state=0)\n"
         "seen = {tuple(row) for row in Xtr}\nresult = sum(tuple(row) in seen for row in Xte)"),
        ("shuffle", f"Sorted data breaks unshuffled K-fold. Sort the rows by label, then return the mean 5-fold score "
                    f"(`KFold(5)`, no shuffling) of `DecisionTreeClassifier(max_depth=3, random_state=0)` on the sorted "
                    f"data and on the same data after `shuffle(..., random_state={r})`, as a tuple.",
         "from sklearn.model_selection import KFold, cross_val_score\nfrom sklearn.tree import DecisionTreeClassifier\n"
         "from sklearn.utils import shuffle\norder = np.argsort(y, kind='stable')\nXs, ys = X[order], y[order]\n"
         "tree = DecisionTreeClassifier(max_depth=3, random_state=0)\n"
         f"Xh, yh = shuffle(Xs, ys, random_state={r})\n"
         "result = (cross_val_score(tree, Xs, ys, cv=KFold(5)).mean(), cross_val_score(tree, Xh, yh, cv=KFold(5)).mean())"),
        ("compute_sample_weight", "Two targets at once: `Y = np.column_stack([y, (X[:, 0] > 0).astype(int)])`. Return "
                                  "the first 5 values of `compute_sample_weight('balanced', Y)` (the product of the "
                                  "per-column balanced weights) and check (not graded) it equals that product.",
         "from sklearn.utils.class_weight import compute_sample_weight\nY = np.column_stack([y, (X[:, 0] > 0).astype(int)])\n"
         "result = compute_sample_weight('balanced', Y)[:5]"),
        ("compute_class_weight", f"How far are 'balanced' weights from inverse frequencies? Return `(balanced weights, "
                                 f"1 / class frequency, their ratio)` for `y` (the ratio is the constant 1 / n_classes).",
         "from sklearn.utils.class_weight import compute_class_weight\ncls = np.unique(y)\n"
         "w = compute_class_weight('balanced', classes=cls, y=y)\ninv = 1 / (np.bincount(y) / len(y))\nresult = (w, inv, w / inv)"),
        ("stratify", f"Make a small stratified training set: `resample(X, y, n_samples=60, replace=False, stratify=y, "
                     f"random_state={r})`; train `{LR}` on it and return its accuracy on all of `X, y` and the class "
                     f"counts of the sample, as a tuple.",
         ru + LRI + f"Xs, ys = resample(X, y, n_samples=60, replace=False, stratify=y, random_state={r})\n"
         f"result = ({LR}.fit(Xs, ys).score(X, y), np.bincount(ys))"),
    ]
    return catalog(rng, items, d.setup, f"{d.desc}; class 1 is rare.", tol=1e-6)


# ================================================================= challenge

def _clip_setup(s, n, q, report, prefix="g"):
    return (f"{prefix} = np.random.default_rng({s})\nx = {prefix}.standard_t(2, {n}).round(3)\n"
            f"qs = np.sort({prefix}.uniform(0, 0.3, {q}).round(4))\nreport = {report!r}")


@template(M, "challenge")
def clipper_quantile_sweep(rng):
    s = seeded(rng)
    report = rng.choice(["mean", "changed", "var"])
    setup = _clip_setup(s, rng.randint(7, 11), 4, report)
    tests = [_clip_setup(s + i, 200000, 50000, report) for i in (1, 2)]
    sol = ("v = np.sort(x)\nn = len(v)\n\ndef quant(q):   # np.quantile's default (linear) interpolation\n"
           "    pos = q * (n - 1)\n    lo = np.floor(pos).astype(int)\n    hi = np.minimum(lo + 1, n - 1)\n"
           "    return v[lo] + (pos - lo) * (v[hi] - v[lo])\n\nlow, high = quant(qs), quant(1 - qs)\n"
           "a = np.searchsorted(v, low, side='left')     # values below low: v[:a]\n"
           "b = np.searchsorted(v, high, side='right')   # values above high: v[b:]\n")
    if report == "changed":
        sol += "result = a + (n - b)"
        want = "how many values clipping actually changes (strictly below `low` or strictly above `high`), as an int array"
    else:
        sol += ("c1 = np.concatenate([[0.0], np.cumsum(v)])\nmid = c1[b] - c1[a]\n"
                "mean = (a * low + mid + (n - b) * high) / n\n")
        if report == "mean":
            sol += "result = mean"
            want = "the mean of the clipped data, as a float array"
        else:
            sol += ("c2 = np.concatenate([[0.0], np.cumsum(v * v)])\nsq = (a * low ** 2 + c2[b] - c2[a] + (n - b) * high ** 2) / n\n"
                    "result = sq - mean ** 2")
            want = "the (population) variance of the clipped data, as a float array"
    return challenge(
        "Sweeping the clipping quantile",
        "The guide's `Clipper(quantile=q)` learns `low = np.quantile(x, q)` and `high = np.quantile(x, 1 - q)` (default "
        "linear interpolation) and clips to `[low, high]`. A grid search wants a statistic of `Clipper(q)."
        f"fit_transform(x)` for every `q` in `qs` (one column of data `x`). Return {want}, one value per `q`. "
        f"(`report` = {report!r}.)",
        setup, sol, tests, "len(x) ≤ 2·10^5, len(qs) ≤ 5·10^4",
        speed="O((n + Q) log n): sort once, prefix sums and binary search per q", tol=1e-6)


def _weights_setup(s, n, k, report, prefix="g"):
    return (f"{prefix} = np.random.default_rng({s})\ny = {prefix}.choice({k}, {n}, p={prefix}.dirichlet(np.ones({k})))\n"
            f"report = {report!r}")


@template(M, "challenge")
def prefix_balanced_weights(rng):
    s = seeded(rng)
    report = rng.choice(["current", "max", "min"])
    setup = _weights_setup(s, rng.randint(8, 12), rng.randint(3, 4), report)
    tests = [_weights_setup(s + i, 200000, 300, report) for i in (1, 2)]
    if report == "current":
        sol = ("cnt = np.zeros(int(y.max()) + 1, dtype=np.int64)\nseen = np.zeros(len(cnt), bool)\nk, out = 0, []\n"
               "for t, c in enumerate(y.tolist(), 1):\n    cnt[c] += 1\n    if not seen[c]:\n        seen[c] = True\n"
               "        k += 1\n    out.append(t / (k * cnt[c]))\nresult = np.array(out)")
        want = "the weight of the class of the newest label `y[t - 1]`"
    elif report == "min":
        sol = ("cnt = np.zeros(int(y.max()) + 1, dtype=np.int64)\nk, top, out = 0, 0, []\n"
               "for t, c in enumerate(y.tolist(), 1):\n    k += cnt[c] == 0\n    cnt[c] += 1\n    top = max(top, cnt[c])\n"
               "    out.append(t / (k * top))\nresult = np.array(out)")
        want = "the **smallest** class weight (that of the most frequent class so far)"
    else:
        sol = ("cnt = np.zeros(int(y.max()) + 1, dtype=np.int64)\nfreq = np.zeros(len(y) + 2, dtype=np.int64)  "
               "# freq[c] = number of seen classes with count c\nk, low, out = 0, 0, []\n"
               "for t, c in enumerate(y.tolist(), 1):\n    old = cnt[c]\n    cnt[c] += 1\n    if old == 0:\n"
               "        k += 1\n        low = 1\n    else:\n        freq[old] -= 1\n"
               "        if old == low and freq[old] == 0:\n            low += 1\n    freq[old + 1] += 1\n"
               "    out.append(t / (k * low))\nresult = np.array(out)")
        want = "the **largest** class weight (that of the rarest class seen so far)"
    return challenge(
        "Balanced weights on a stream",
        "Labels arrive one at a time. After each arrival t (t = 1, 2, …, len(y)) a trainer recomputes "
        "`compute_class_weight('balanced', classes=np.unique(y[:t]), y=y[:t])`: class c gets t / (k · count_c), "
        f"where k is the number of distinct classes seen so far. Record {want} after every arrival and return the "
        f"records as a float array. (`report` = {report!r}.)",
        setup, sol, tests, "len(y) ≤ 2·10^5, up to 300 classes",
        speed="O(n): update counts incrementally (no recount per prefix)", tol=1e-9)


def _ecdf_setup(s, n, report, prefix="g"):
    return (f"{prefix} = np.random.default_rng({s})\nop = {prefix}.random({n}) < 0.5   # True: add value, False: query\n"
            f"op[0] = True\nval = {prefix}.integers(0, {max(10, n // 2)}, {n})\nreport = {report!r}")


@template(M, "challenge")
def online_ecdf(rng):
    s = seeded(rng)
    report = rng.choice(["leq", "mid"])
    setup = _ecdf_setup(s, rng.randint(9, 14), report)
    tests = [_ecdf_setup(s + i, 200000, report) for i in (1, 2)]
    sol = ("u = np.unique(val)\nr = np.searchsorted(u, val)          # compressed coordinates\n"
           "tree = [0] * (len(u) + 1)\n\ndef add(i):\n    i += 1\n    while i <= len(u):\n        tree[i] += 1\n"
           "        i += i & -i\n\ndef count(i):   # how many added values have rank < i\n    tot = 0\n    while i > 0:\n"
           "        tot += tree[i]\n        i -= i & -i\n    return tot\n\nout, n_added = [], 0\n"
           "for is_add, ri in zip(op.tolist(), r.tolist()):\n    if is_add:\n        add(ri)\n        n_added += 1\n"
           "    else:\n")
    if report == "leq":
        sol += "        out.append(count(ri + 1) / n_added)\nresult = np.array(out)"
        want = "the fraction of fitted values **≤** the queried value"
    else:
        sol += ("        less, leq = count(ri), count(ri + 1)\n        out.append((less + leq) / (2 * n_added))\n"
                "result = np.array(out)")
        want = "the mid-rank fraction ((number of fitted values < v) + (number ≤ v)) / (2 · number of fitted values)"
    return challenge(
        "An empirical-CDF transformer that keeps learning",
        "A custom transformer maps a value v to its position in the training data. It is updated online: operation "
        "i either adds `val[i]` to the fitted data (`op[i]` True, like a `partial_fit`) or transforms the query "
        f"`val[i]` (`op[i]` False), returning {want} among the values added so far (the first operation is an add). "
        f"Return the answers to the queries, in order, as a float array. (`report` = {report!r}.)",
        setup, sol, tests, "number of operations ≤ 2·10^5", speed="O(n log n): a Fenwick tree over compressed values",
        tol=1e-9)


def _enc_setup(s, n, k, m, report, prefix="g"):
    return (f"{prefix} = np.random.default_rng({s})\ncat = {prefix}.integers(0, {k}, {n})\n"
            f"y = ({prefix}.random({n}) < 0.2 + 0.6 * (cat % 3 == 0)).astype(float)\n"
            f"query = {prefix}.integers(0, {k} + 2, {max(3, n // 2)})   # may contain unseen categories\n"
            f"smoothing = {m}\nreport = {report!r}")


@template(M, "challenge")
def smoothed_target_encoder(rng):
    s = seeded(rng)
    report = rng.choice(["encode", "loo"])
    m = rng.choice([1.0, 5.0, 10.0])
    setup = _enc_setup(s, rng.randint(8, 12), rng.randint(3, 4), m, report)
    tests = [_enc_setup(s + i, 200000, 50000, m, report) for i in (1, 2)]
    sol = ("K = int(max(cat.max(), query.max())) + 1\ncnt = np.bincount(cat, minlength=K).astype(float)\n"
           "tot = np.bincount(cat, weights=y, minlength=K)\nprior = y.mean()\n")
    if report == "encode":
        sol += "result = ((tot + smoothing * prior) / (cnt + smoothing))[query]"
        want = ("the encodings of `query` (a category never seen in `cat` gets (0 + smoothing·prior)/(0 + smoothing) = "
                "prior)")
    else:
        sol += ("n = len(y)\nprior_loo = (y.sum() - y) / (n - 1)\n"
                "result = (tot[cat] - y + smoothing * prior_loo) / (cnt[cat] - 1 + smoothing)")
        want = ("the **leave-one-out** encoding of every training row: computed as if row i were not in the data "
                "(its own y removed from the category sum, the count and the prior, which becomes the mean of the other "
                "n − 1 targets)")
    return challenge(
        "Smoothed target encoding",
        "A custom `TargetMeanEncoder(smoothing)` encodes a category c by `(sum of y over rows with category c + "
        "smoothing · prior) / (count of c + smoothing)`, where `prior` is the mean of all of `y`. It is fitted on the "
        f"category ids `cat` (ints) with targets `y`. Return {want}, as a float array. (`report` = {report!r}.)",
        setup, sol, tests, "len(cat) ≤ 2·10^5, category ids < 5·10^4 + 2", speed="O(n + K) with bincount", tol=1e-9)


def _msw_setup(s, n, k, report, prefix="g"):
    return (f"{prefix} = np.random.default_rng({s})\nY = np.column_stack([{prefix}.choice(c, {n}, "
            f"p={prefix}.dirichlet(np.ones(c))) for c in {prefix}.integers(2, 6, {k})])\nreport = {report!r}")


@template(M, "challenge")
def multioutput_sample_weight(rng):
    s = seeded(rng)
    report = rng.choice(["weights", "top", "total"])
    setup = _msw_setup(s, rng.randint(8, 12), rng.randint(2, 3), report)
    tests = [_msw_setup(s + i, 200000, 4, report) for i in (1, 2)]
    sol = ("w = np.ones(len(Y))\nfor j in range(Y.shape[1]):\n    col = Y[:, j]\n"
           "    cls, inv, cnt = np.unique(col, return_inverse=True, return_counts=True)\n"
           "    w *= (len(col) / (len(cls) * cnt))[inv]\n")
    sol += {"weights": "result = w", "top": "result = int(np.argmax(w))", "total": "result = float(w.sum())"}[report]
    want = {"weights": "the weights, as a float array",
            "top": "the index of the row with the largest weight (the first one on ties)",
            "total": "the sum of all weights"}[report]
    return challenge(
        "Balanced weights for several targets",
        "`compute_sample_weight('balanced', Y)` with a 2-D `Y` (one column per target) gives every row the **product** "
        "over columns of its class's balanced weight in that column, n / (number of classes in the column · count of "
        f"the row's class). Compute these sample weights and return {want}. (`report` = {report!r}.)",
        setup, sol, tests, "rows ≤ 2·10^5, ≤ 4 targets", speed="O(n log n) per column (np.unique), no loop over rows",
        tol=1e-9)
