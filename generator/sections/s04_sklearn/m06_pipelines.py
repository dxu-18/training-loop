"""4.06 Pipelines & ColumnTransformer: chain steps, route columns, index/slice/set_params, feature names, pandas
output, FeatureUnion, TransformedTargetRegressor, and leakage-free preprocessing."""

from ._common import (P, catalog, challenge, lit, seeded, synth_clf, table, template)

M = "4.06"
LR = "LogisticRegression(max_iter=1000)"
IMPORTS = ("from sklearn.compose import ColumnTransformer\nfrom sklearn.pipeline import Pipeline\n"
           "from sklearn.impute import SimpleImputer\nfrom sklearn.preprocessing import OneHotEncoder, StandardScaler\n")
POSITIVE = ("bikes", "phones", "experiments")    # regression scenarios whose target is never below 0 (log1p-safe)


def _num_data(rng, classes=(2, 3)):
    """Numeric classification data: wine (scales differ a lot, so a scaler matters) or make_classification."""
    if rng.random() < 0.4:
        return ("from sklearn.datasets import load_wine\nX, y = load_wine(return_X_y=True)\n"
                f"order = np.random.default_rng({seeded(rng)}).permutation(len(X))\nX, y = X[order], y[order]",
                "`X` holds 178 wines (13 chemical measurements on very different scales; rows shuffled) and `y` their "
                "cultivar (0, 1, 2)", 13)
    d = synth_clf(rng, n=(100, 250), d=(4, 8), classes=classes)
    return d.setup, d.desc, d.d


def _small_numeric(rng):
    """iris or wine (rows shuffled) for FeatureUnion problems."""
    name = rng.choice(["iris", "wine"])
    d = 4 if name == "iris" else 13
    what = ("150 iris flowers (4 measurements) and `y` their species (0, 1, 2)" if name == "iris" else
            "178 wines (13 chemical measurements) and `y` their cultivar (0, 1, 2)")
    return (f"from sklearn.datasets import load_{name}\nX, y = load_{name}(return_X_y=True)\n"
            f"order = np.random.default_rng({seeded(rng)}).permutation(len(X))\nX, y = X[order], y[order]",
            f"`X` holds {what} (rows shuffled)", d)


def _cols(t):
    return f"num_cols, cat_cols = {lit(t.num)}, {lit(t.cat)}"


def _prep(strategy="median", ohe="handle_unknown='ignore', sparse_output=False"):
    return ("preprocess = ColumnTransformer([\n"
            f"    ('num', Pipeline([('impute', SimpleImputer(strategy={strategy!r})), ('scale', StandardScaler())]), "
            "num_cols),\n"
            f"    ('cat', OneHotEncoder({ohe}), cat_cols),\n])")


def _model(t, final=None, fitted=True, strategy="median"):
    """Setup lines: the guide's preprocess + final step, built (and fitted) on df, y."""
    if final is None:
        final = ("from sklearn.linear_model import LogisticRegression", "clf", LR) if t.kind == "clf" else \
            ("from sklearn.linear_model import Ridge", "reg", "Ridge(alpha=1.0)")
    imp, name, ctor = final
    fit = ".fit(df, y)" if fitted else ""
    return (f"{IMPORTS}{imp}\n{_cols(t)}\n{_prep(strategy)}\n"
            f"model = Pipeline([('prep', preprocess), ('{name}', {ctor})]){fit}")


def _new_rows(rng, t, k=None):
    """`new`: k rows sampled from df, the first one with a never-seen level of a categorical column."""
    k = k or rng.randint(3, 5)
    col = rng.choice(t.cat)
    return (f"new = df.sample({k}, random_state={seeded(rng)}).reset_index(drop=True)\n"
            f"new.loc[0, {col!r}] = 'unseen'   # a level the training data never had"), col, k


# ================================================================= lite

@template(M, "lite")
def pipeline_step_access(rng):
    setup, desc, d = _num_data(rng)
    n, c = rng.randint(2, 5), rng.choice([0.01, 0.1, 0.5, 5.0])
    setup += ("\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.preprocessing import StandardScaler\n"
              f"from sklearn.linear_model import LogisticRegression\npipe = Pipeline([('scale', StandardScaler()), "
              f"('clf', {LR})]).fit(X, y)")
    items = [
        ("named_steps", "Return the coefficients learned by the classifier inside `pipe`, reached by name through "
                        "`named_steps`.", "result = pipe.named_steps['clf'].coef_"),
        ("mean_", "Index the pipeline by step name (`pipe['scale']`) and return the column means the scaler learned.",
         "result = pipe['scale'].mean_"),
        ("classes_", "Index the pipeline by position (`pipe[-1]` is the last step) and return the classes the "
                     "classifier learned.", "result = pipe[-1].classes_"),
        ("transform", f"Slice the classifier off (`pipe[:-1]` is a pipeline of the steps before it) and return the "
                      f"scaled version of the first {n} rows of `X`.", f"result = pipe[:-1].transform(X[:{n}])"),
        ("predict", f"A pipeline predicts like a model: it scales, then classifies. Return the predictions for the "
                    f"first {n} rows of `X`.", f"result = pipe.predict(X[:{n}])"),
        ("predict_proba", f"Return the class probabilities for the first {n} rows of `X`.",
         f"result = pipe.predict_proba(X[:{n}])"),
        ("score", "Return the pipeline's training accuracy on `X, y`.", "result = pipe.score(X, y)"),
        ("steps", "Return the step names in order, as a list (`pipe.steps` is a list of `(name, estimator)` pairs).",
         "result = [name for name, _ in pipe.steps]"),
        ("set_params", f"Change the classifier's `C` to {c} through the pipeline (`<step>__<param>` syntax: "
                       f"`set_params(clf__C=...)`) and return `pipe.get_params()['clf__C']`.",
         f"pipe.set_params(clf__C={c})\nresult = pipe.get_params()['clf__C']"),
        ("n_features_in_", "Return how many input features the pipeline expects (`n_features_in_`).",
         "result = pipe.n_features_in_"),
    ]
    return catalog(rng, items, setup, f"{desc}. `pipe` = standard scaling followed by `{LR}`, fitted on `X, y` (step "
                                      f"names `'scale'` and `'clf'`).", tol=1e-6)


@template(M, "lite")
def pipeline_construction(rng):
    setup, desc, d = _num_data(rng)
    n, p = rng.randint(2, 4), rng.randint(2, 3)
    items = [
        ("Pipeline", f"Build `Pipeline([('scale', StandardScaler()), ('clf', {LR})])` (from `sklearn.pipeline`), fit "
                     f"it on `X, y` and return its training accuracy.",
         "from sklearn.pipeline import Pipeline\nfrom sklearn.preprocessing import StandardScaler\n"
         f"from sklearn.linear_model import LogisticRegression\npipe = Pipeline([('scale', StandardScaler()), ('clf', "
         f"{LR})])\nresult = pipe.fit(X, y).score(X, y)"),
        ("make_pipeline", f"`make_pipeline` names the steps for you (lower-cased class names). Build `make_pipeline("
                          f"StandardScaler(), {LR})` and return its step names as a list (`list(pipe.named_steps)`).",
         "from sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\n"
         f"from sklearn.linear_model import LogisticRegression\npipe = make_pipeline(StandardScaler(), {LR})\n"
         "result = list(pipe.named_steps)"),
        ("make_pipeline", f"Fit `make_pipeline(StandardScaler(), PCA(n_components={p}), {LR})` on `X, y` and return the "
                          f"shape of what reaches the classifier: `pipe[:-1].transform(X).shape`.",
         "from sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\n"
         f"from sklearn.decomposition import PCA\nfrom sklearn.linear_model import LogisticRegression\n"
         f"pipe = make_pipeline(StandardScaler(), PCA(n_components={p}), {LR}).fit(X, y)\n"
         "result = pipe[:-1].transform(X).shape", {"require": ["make_pipeline", "PCA"]}),
        ("SimpleImputer", f"`X_missing` is `X` with some values knocked out. Build `make_pipeline(SimpleImputer("
                          f"strategy='mean'), StandardScaler(), {LR})`, fit it on `X_missing, y` and return its "
                          f"training accuracy.",
         "from sklearn.pipeline import make_pipeline\nfrom sklearn.impute import SimpleImputer\n"
         f"from sklearn.preprocessing import StandardScaler\nfrom sklearn.linear_model import LogisticRegression\n"
         f"pipe = make_pipeline(SimpleImputer(strategy='mean'), StandardScaler(), {LR})\n"
         "result = pipe.fit(X_missing, y).score(X_missing, y)",
         {"setup": f"X_missing = X.copy()\nX_missing[np.random.default_rng({seeded(rng)}).random(X.shape) < 0.1] = np.nan",
          "require": ["SimpleImputer", "make_pipeline|Pipeline"]}),
        ("fit", f"`fit` returns the pipeline, so calls chain. In **one** expression, fit `make_pipeline(StandardScaler(), "
                f"{LR})` on `X, y` and predict the first {n} rows.",
         "from sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\n"
         f"from sklearn.linear_model import LogisticRegression\nmake_pipeline(StandardScaler(), {LR}).fit(X, y).predict(X[:{n}])",
         {"require": ["make_pipeline", "fit", "predict"]}),
    ]
    return catalog(rng, items, setup, f"{desc}.", tol=1e-6)


@template(M, "lite")
def column_transformer_parts(rng):
    t = table(rng, n=(60, 150))
    col = rng.choice(t.cat)
    pat = rng.choice(["_", "^[a-m]", "e"])
    ct = ("ct = ColumnTransformer([('num', StandardScaler(), num_cols), "
          "('cat', OneHotEncoder(sparse_output=False), cat_cols)])")
    imp = "from sklearn.compose import ColumnTransformer\nfrom sklearn.preprocessing import OneHotEncoder, StandardScaler\n"
    items = [
        ("ColumnTransformer", "Build `ColumnTransformer([('num', StandardScaler(), num_cols), ('cat', OneHotEncoder("
                              "sparse_output=False), cat_cols)])` (from `sklearn.compose`), fit-transform `df` and return "
                              "the shape of the result.", f"{imp}{ct}\nresult = ct.fit_transform(df).shape"),
        ("get_feature_names_out", "Fit that `ColumnTransformer` (standard-scale `num_cols`, one-hot `cat_cols` with "
                                  "`sparse_output=False`; names `'num'` and `'cat'`) on `df` and return its output "
                                  "feature names as a list.", f"{imp}{ct}\nresult = list(ct.fit(df).get_feature_names_out())"),
        ("remainder", "One-hot encode **only** `cat_cols` (`OneHotEncoder(sparse_output=False)`, name `'cat'`) and keep "
                      "every other column unchanged with `remainder='passthrough'`. Fit on `df` and return the output "
                      "feature names as a list.",
         "from sklearn.compose import ColumnTransformer\nfrom sklearn.preprocessing import OneHotEncoder\n"
         "ct = ColumnTransformer([('cat', OneHotEncoder(sparse_output=False), cat_cols)], remainder='passthrough')\n"
         "result = list(ct.fit(df).get_feature_names_out())"),
        ("ColumnTransformer", "Columns you don't list are **dropped** by default (`remainder='drop'`). Standardize only "
                              "`num_cols` with a `ColumnTransformer` (one transformer, `'num'`), fit-transform `df` and "
                              "return the shape.",
         "from sklearn.compose import ColumnTransformer\nfrom sklearn.preprocessing import StandardScaler\n"
         "result = ColumnTransformer([('num', StandardScaler(), num_cols)]).fit_transform(df).shape"),
        ("named_transformers_", "Fit the `ColumnTransformer` (scale `num_cols` as `'num'`, one-hot `cat_cols` as `'cat'`) "
                                "on `df` and return the means its scaler learned: `named_transformers_['num'].mean_`.",
         f"{imp}{ct}\nresult = ct.fit(df).named_transformers_['num'].mean_"),
        ("categories_", f"Fit the `ColumnTransformer` (scale `num_cols` as `'num'`, one-hot `cat_cols` as `'cat'`) on "
                        f"`df` and return the levels the encoder learned for `{col}`, as a list.",
         f"{imp}{ct}\nenc = ct.fit(df).named_transformers_['cat']\nresult = list(enc.categories_[{t.cat.index(col)}])"),
        ("make_column_transformer", "`make_column_transformer` names the transformers for you. Build "
                                    "`make_column_transformer((StandardScaler(), num_cols), (OneHotEncoder(), cat_cols))`, "
                                    "fit it on `df` and return the names in `transformers_` as a list.",
         "from sklearn.compose import make_column_transformer\nfrom sklearn.preprocessing import OneHotEncoder, StandardScaler\n"
         "ct = make_column_transformer((StandardScaler(), num_cols), (OneHotEncoder(), cat_cols)).fit(df)\n"
         "result = [name for name, *_ in ct.transformers_]"),
        ("make_column_selector", "A `make_column_selector` picks columns by dtype; called on a DataFrame it returns their "
                                 "names. Return the columns `make_column_selector(dtype_include='number')` selects from `df`.",
         "from sklearn.compose import make_column_selector\nresult = make_column_selector(dtype_include='number')(df)"),
        ("make_column_selector", "Return the columns `make_column_selector(dtype_exclude='number')` selects from `df` "
                                 "(the text columns).",
         "from sklearn.compose import make_column_selector\nresult = make_column_selector(dtype_exclude='number')(df)"),
        ("make_column_selector", f"Selectors can also match names with a regex. Return the columns "
                                 f"`make_column_selector(pattern={pat!r})` selects from `df`.",
         f"from sklearn.compose import make_column_selector\nresult = make_column_selector(pattern={pat!r})(df)"),
        ("set_output", "Build the `ColumnTransformer` (scale `num_cols` as `'num'`, one-hot `cat_cols` with "
                       "`sparse_output=False` as `'cat'`), call `.set_output(transform='pandas')` on it, fit-transform "
                       "`df`, and return the columns of the resulting DataFrame as a list.",
         f"{imp}{ct}\nct.set_output(transform='pandas')\nresult = list(ct.fit_transform(df).columns)"),
        ("output_indices_", "Fit the `ColumnTransformer` (scale `num_cols` as `'num'`, one-hot `cat_cols` as `'cat'`) on "
                            "`df`. Which output columns came from `'cat'`? Return `(start, stop)` of the slice "
                            "`output_indices_['cat']`.",
         f"{imp}{ct}\ns = ct.fit(df).output_indices_['cat']\nresult = (s.start, s.stop)"),
    ]
    return catalog(rng, items, t.setup + "\n" + _cols(t), f"{t.desc}. `num_cols` and `cat_cols` list its numeric and "
                                                          f"text columns.", tol=1e-6)


@template(M, "lite")
def target_transform_basics(rng):
    t = table(rng, kind="reg", numeric_only=True, only=POSITIVE, n=(60, 160))
    n, a = rng.randint(2, 4), rng.choice([0.1, 1.0, 10.0])
    ttr = f"TransformedTargetRegressor(regressor=Ridge(alpha={a}), func=np.log1p, inverse_func=np.expm1)"
    imp = "from sklearn.compose import TransformedTargetRegressor\nfrom sklearn.linear_model import Ridge\n"
    items = [
        ("TransformedTargetRegressor", f"Model the target on a log scale without doing it by hand: fit `{ttr}` (from "
                                       f"`sklearn.compose`) on `df, y` and return its predictions for the first {n} "
                                       f"rows (they come back in the original units).",
         f"{imp}ttr = {ttr}.fit(df, y)\nresult = ttr.predict(df.iloc[:{n}])"),
        ("regressor_", f"Fit `{ttr}` on `df, y` and return the coefficients of the fitted inner model (`regressor_`), "
                       f"which was trained on `log1p(y)`.", f"{imp}result = {ttr}.fit(df, y).regressor_.coef_"),
        ("regressor_", f"Fit `{ttr}` on `df, y`. Return the **raw** predictions of the inner `regressor_` for the first "
                       f"{n} rows: these are still on the log1p scale.",
         f"{imp}ttr = {ttr}.fit(df, y)\nresult = ttr.regressor_.predict(df.iloc[:{n}])"),
        ("score", f"Fit `{ttr}` on `df, y` and return its training R² (`score` compares predictions with `y` in the "
                  f"original units).", f"{imp}result = {ttr}.fit(df, y).score(df, y)"),
        ("transformer_", f"A target transformer can also be an estimator: fit `TransformedTargetRegressor(regressor="
                         f"Ridge(alpha={a}), transformer=StandardScaler())` on `df, y` and return the target mean its "
                         f"fitted `transformer_` learned (`mean_`, an array of length 1).",
         f"{imp}from sklearn.preprocessing import StandardScaler\n"
         f"ttr = TransformedTargetRegressor(regressor=Ridge(alpha={a}), transformer=StandardScaler()).fit(df, y)\n"
         "result = ttr.transformer_.mean_"),
    ]
    return catalog(rng, items, t.setup, f"{t.desc} (never negative).", tol=1e-6)


@template(M, "lite")
def feature_union_basics(rng):
    setup, desc, d = _small_numeric(rng)
    p, k = rng.randint(1, 3), rng.randint(1, 3)
    u = f"FeatureUnion([('pca', PCA(n_components={p})), ('best', SelectKBest(k={k}))])"
    imp = ("from sklearn.pipeline import FeatureUnion\nfrom sklearn.decomposition import PCA\n"
           "from sklearn.feature_selection import SelectKBest\n")
    items = [
        ("FeatureUnion", f"A `FeatureUnion` runs transformers side by side and stacks their outputs. Fit-transform "
                         f"`{u}` on `X, y` and return the shape of the result.",
         f"{imp}result = {u}.fit_transform(X, y).shape"),
        ("get_feature_names_out", f"Fit `{u}` on `X, y` and return its output feature names as a list.",
         f"{imp}result = list({u}.fit(X, y).get_feature_names_out())"),
        ("named_transformers", f"Fit `{u}` on `X, y` and return the indices of the columns its `'best'` part kept "
                               f"(`named_transformers['best'].get_support(indices=True)`).",
         f"{imp}result = {u}.fit(X, y).named_transformers['best'].get_support(indices=True)"),
        ("fit_transform", f"Fit-transform `{u}` on `X, y` and return the first 2 rows of the output.",
         f"{imp}result = {u}.fit_transform(X, y)[:2]"),
    ]
    return catalog(rng, items, setup, f"{desc}.", tol=1e-6)


# ================================================================= easy

@template(M, "easy")
def pipeline_equals_manual(rng):
    setup, desc, d = _num_data(rng)
    n = rng.randint(3, 6)
    task = rng.choice(["chain", "manual", "slice"])
    s2 = (setup + "\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.preprocessing import StandardScaler\n"
          f"from sklearn.linear_model import LogisticRegression\npipe = Pipeline([('scale', StandardScaler()), "
          f"('clf', {LR})]).fit(X, y)")
    head = f"{desc}. `pipe` = `StandardScaler` then `{LR}` (steps `'scale'`, `'clf'`), fitted on `X, y`."
    if task == "chain":
        return P(f"{head} A pipeline's `predict_proba` just passes the data through each step. Rebuild "
                 f"`pipe.predict_proba(X[:{n}])` yourself: transform with `pipe['scale']`, then call the classifier "
                 f"`pipe['clf']`.", s2,
                 f"result = pipe['clf'].predict_proba(pipe['scale'].transform(X[:{n}]))",
                 require=["transform", "predict_proba"], tol=1e-9)
    if task == "manual":
        return P(f"{desc}. Without a pipeline: fit a `StandardScaler` on `X`, fit `{LR}` on the scaled `X`, and return "
                 f"the class probabilities of the first {n} rows (scale them with the same scaler). It matches what a "
                 f"pipeline of the two would give.", setup,
                 "from sklearn.preprocessing import StandardScaler\nfrom sklearn.linear_model import LogisticRegression\n"
                 f"sc = StandardScaler().fit(X)\nclf = {LR}.fit(sc.transform(X), y)\n"
                 f"result = clf.predict_proba(sc.transform(X[:{n}]))", require=["StandardScaler", "transform"], tol=1e-6)
    return P(f"{head} Show that the pipeline's scaler is an ordinary scaler: return the largest absolute difference "
             f"between `pipe[:-1].transform(X)` and `StandardScaler().fit_transform(X)` (zero up to rounding).", s2,
             "result = float(np.abs(pipe[:-1].transform(X) - StandardScaler().fit_transform(X)).max())",
             require=["transform"], tol=1e-9)


@template(M, "easy")
def impute_then_scale(rng):
    t = table(rng, numeric_only=True, missing=2, n=(50, 120))
    s = rng.choice(["mean", "median"])
    n = rng.randint(2, 4)
    pipe = f"Pipeline([('impute', SimpleImputer(strategy={s!r})), ('scale', StandardScaler())])"
    imp = "from sklearn.pipeline import Pipeline\nfrom sklearn.impute import SimpleImputer\nfrom sklearn.preprocessing import StandardScaler\n"
    items = [
        ("SimpleImputer", f"Build `{pipe}`, fit-transform `df` and return the first {n} rows of the result.",
         f"{imp}result = {pipe}.fit_transform(df)[:{n}]"),
        ("statistics_", f"Fit `{pipe}` on `df` and return the fill values the imputer learned (`statistics_`, one per "
                        f"column), reached through `named_steps`.",
         f"{imp}result = {pipe}.fit(df).named_steps['impute'].statistics_"),
        ("add_indicator", f"Impute with `SimpleImputer(strategy={s!r}, add_indicator=True)` (it appends a 0/1 "
                          f"\"was missing\" column for every column that had gaps), then scale. Return the output shape.",
         f"{imp}pipe = Pipeline([('impute', SimpleImputer(strategy={s!r}, add_indicator=True)), ('scale', StandardScaler())])\n"
         "result = pipe.fit_transform(df).shape"),
        ("isnan", f"Return `(missing_before, missing_after)`: the number of NaNs in `df` and in the output of `{pipe}` "
                  f"fitted on it.",
         f"{imp}Z = {pipe}.fit_transform(df)\nresult = (int(df.isna().sum().sum()), int(np.isnan(Z).sum()))",
         {"require": ["SimpleImputer"]}),
        ("mean_", "Mean imputation doesn't move a column's mean. Fit `Pipeline([('impute', SimpleImputer(strategy='mean'))"
                  ", ('scale', StandardScaler())])` on `df` and return `pipe['scale'].mean_ - pipe['impute'].statistics_` "
                  "(zeros up to rounding).",
         f"{imp}pipe = Pipeline([('impute', SimpleImputer(strategy='mean')), ('scale', StandardScaler())]).fit(df)\n"
         "result = pipe['scale'].mean_ - pipe['impute'].statistics_", {"tol": 1e-9}),
    ]
    return catalog(rng, items, t.setup, f"{t.desc}; some of its numeric values are missing (NaN).", tol=1e-6)


@template(M, "easy")
def sweep_with_set_params(rng):
    setup, desc, d = _num_data(rng)
    task = rng.choice(["C", "passthrough", "pca"])
    imp = ("from sklearn.pipeline import Pipeline\nfrom sklearn.preprocessing import StandardScaler\n"
           "from sklearn.linear_model import LogisticRegression\n")
    if task == "C":
        cs = sorted(rng.sample([0.001, 0.01, 0.1, 1.0, 10.0], 3))
        return P(f"{desc}. Build `pipe = Pipeline([('scale', StandardScaler()), ('clf', {LR})])` once. For each `C` in "
                 f"`{cs}`, change it with `pipe.set_params(clf__C=C)`, refit on `X, y` and record the training "
                 f"accuracy. Return the list.", setup,
                 f"{imp}pipe = Pipeline([('scale', StandardScaler()), ('clf', {LR})])\n"
                 f"result = [pipe.set_params(clf__C=C).fit(X, y).score(X, y) for C in {cs}]",
                 require=["set_params"], tol=1e-9)
    if task == "passthrough":
        return P(f"{desc}. A step can be switched off by setting it to `'passthrough'`. Build `pipe = Pipeline([('scale', "
                 f"StandardScaler()), ('clf', KNeighborsClassifier(n_neighbors=5))])`; return the training accuracy "
                 f"`(with_scaling, without_scaling)`, using `pipe.set_params(scale='passthrough')` for the second.",
                 setup,
                 "from sklearn.pipeline import Pipeline\nfrom sklearn.preprocessing import StandardScaler\n"
                 "from sklearn.neighbors import KNeighborsClassifier\n"
                 "pipe = Pipeline([('scale', StandardScaler()), ('clf', KNeighborsClassifier(n_neighbors=5))])\n"
                 "a = pipe.fit(X, y).score(X, y)\nb = pipe.set_params(scale='passthrough').fit(X, y).score(X, y)\n"
                 "result = (a, b)", require=["set_params"], tol=1e-9)
    ks = list(range(1, min(d, 4) + 1))
    return P(f"{desc}. Build `pipe = Pipeline([('scale', StandardScaler()), ('pca', PCA()), ('clf', {LR})])`. For "
             f"`n_components` in `{ks}`, set it with `set_params(pca__n_components=...)`, refit on `X, y` and record the "
             f"training accuracy. Return the list.", setup,
             f"{imp}from sklearn.decomposition import PCA\n"
             f"pipe = Pipeline([('scale', StandardScaler()), ('pca', PCA()), ('clf', {LR})])\n"
             f"result = [pipe.set_params(pca__n_components=k).fit(X, y).score(X, y) for k in {ks}]",
             require=["set_params", "PCA"], tol=1e-9)


@template(M, "easy")
def ct_output_width(rng):
    t = table(rng, n=(60, 150))
    col = rng.choice(t.cat)
    imp = "from sklearn.compose import ColumnTransformer\nfrom sklearn.preprocessing import OneHotEncoder, StandardScaler\n"
    ct = ("ColumnTransformer([('num', StandardScaler(), num_cols), ('cat', OneHotEncoder(sparse_output=False), "
          "cat_cols)])")
    items = [
        ("OneHotEncoder", f"Predict the width of `{ct}`'s output **before** fitting: the numeric columns plus one column "
                          f"per distinct level of every text column (`nunique`). Return `(by_hand, actual)`, the actual "
                          f"being `fit_transform(df).shape[1]`.",
         f"{imp}by_hand = len(num_cols) + int(df[cat_cols].nunique().sum())\n"
         f"result = (by_hand, {ct}.fit_transform(df).shape[1])", {"require": ["ColumnTransformer", "nunique"]}),
        ("drop", "For linear models one level per text column is redundant. Build the same `ColumnTransformer` (scale "
                 "`num_cols`, one-hot `cat_cols`) but with `OneHotEncoder(drop='first', sparse_output=False)` and return "
                 "the number of output columns after fitting on `df`.",
         f"{imp}ct = ColumnTransformer([('num', StandardScaler(), num_cols), "
         "('cat', OneHotEncoder(drop='first', sparse_output=False), cat_cols)])\nresult = ct.fit_transform(df).shape[1]"),
        ("verbose_feature_names_out", f"By default, output names get a `<transformer>__` prefix. Build `{ct[:-1]}, "
                                      f"verbose_feature_names_out=False)`, fit it on `df` and return the output names as "
                                      f"a list.",
         f"{imp}ct = {ct[:-1]}, verbose_feature_names_out=False)\nresult = list(ct.fit(df).get_feature_names_out())"),
        ("get_feature_names_out", f"Fit `{ct}` on `df` and return how many of its output names start with `'cat__'`.",
         f"{imp}names = {ct}.fit(df).get_feature_names_out()\nresult = sum(n.startswith('cat__') for n in names)"),
        ("get_feature_names_out", f"Fit `{ct}` on `df` and return, as a list, the output names that come from the "
                                  f"column `{col}` (they start with `'cat__{col}_'`).",
         f"{imp}names = {ct}.fit(df).get_feature_names_out()\n"
         f"result = [n for n in names if n.startswith('cat__{col}_')]"),
    ]
    return catalog(rng, items, t.setup + "\n" + _cols(t), f"{t.desc}. `num_cols` and `cat_cols` list its numeric and "
                                                          f"text columns.")


@template(M, "easy")
def auto_step_names(rng):
    t = table(rng, kind="reg", numeric_only=True, missing=1, n=(60, 150))
    a = rng.choice([0.1, 1.0, 10.0])
    task = rng.choice(["names", "dupes", "coef"])
    imp = ("from sklearn.pipeline import make_pipeline\nfrom sklearn.impute import SimpleImputer\n"
           "from sklearn.preprocessing import StandardScaler\nfrom sklearn.linear_model import Ridge\n")
    if task == "names":
        return P(f"{t.desc}; a few values are missing. Build `make_pipeline(SimpleImputer(), StandardScaler(), "
                 f"Ridge(alpha={a}))`, fit it on `df, y` and return its step names as a list.", t.setup,
                 f"{imp}pipe = make_pipeline(SimpleImputer(), StandardScaler(), Ridge(alpha={a})).fit(df, y)\n"
                 "result = list(pipe.named_steps)", require=["make_pipeline", "named_steps"])
    if task == "dupes":
        return P(f"{t.desc}; a few values are missing. What does `make_pipeline` call two steps of the same class? "
                 f"Build `make_pipeline(SimpleImputer(), StandardScaler(), StandardScaler(), Ridge(alpha={a}))` and "
                 f"return its step names as a list.", t.setup,
                 f"{imp}pipe = make_pipeline(SimpleImputer(), StandardScaler(), StandardScaler(), Ridge(alpha={a}))\n"
                 "result = list(pipe.named_steps)", require=["make_pipeline"])
    return P(f"{t.desc}; a few values are missing. Fit `make_pipeline(SimpleImputer(strategy='median'), "
             f"StandardScaler(), Ridge(alpha={a}))` on `df, y`. `named_steps` also allows attribute access: return "
             f"`pipe.named_steps.ridge.coef_` as a dict {{column: coefficient}}.", t.setup,
             f"{imp}pipe = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha={a})).fit(df, y)\n"
             "result = dict(zip(df.columns, pipe.named_steps.ridge.coef_))", require=["named_steps"], tol=1e-6)


@template(M, "easy")
def holdout_pipeline(rng):
    t = table(rng, kind="clf", missing=rng.randint(1, 2), n=(150, 300))
    m = rng.randint(t.n // 2, 3 * t.n // 4)
    if rng.random() < 0.5:
        return P(f"{t.desc}. `model` is the guide's pipeline (median-impute + scale the numbers, one-hot the text, then "
                 f"`{LR}`), not fitted yet. Fit it on the first {m} rows of `df` and `y` only and return its accuracy "
                 f"on the remaining rows. Every preprocessing step learns from the training rows alone.",
                 t.setup + "\n" + _model(t, fitted=False),
                 f"result = model.fit(df.iloc[:{m}], y[:{m}]).score(df.iloc[{m}:], y[{m}:])", require=["fit", "score"])
    new, col, k = _new_rows(rng, t)
    return P(f"{t.desc}. `model` is the guide's pipeline (impute + scale the numbers, one-hot the text with "
             f"`handle_unknown='ignore'`, then `{LR}`), fitted on `df, y`. The first of the {k} rows in `new` has a "
             f"`{col}` the model never saw. Return the predicted classes for `new`: an unseen level just gets all-zero "
             f"one-hot columns.", t.setup + "\n" + _model(t) + "\n" + new, "result = model.predict(new)",
             require=["predict"])


@template(M, "easy")
def log_target_by_hand(rng):
    t = table(rng, kind="reg", numeric_only=True, only=POSITIVE, n=(60, 160))
    a, n = rng.choice([0.1, 1.0, 10.0]), rng.randint(3, 5)
    task = rng.choice(["by_hand", "sqrt", "compare"])
    imp = "from sklearn.compose import TransformedTargetRegressor\nfrom sklearn.linear_model import Ridge\n"
    if task == "by_hand":
        return P(f"{t.desc} (never negative). Do by hand what `TransformedTargetRegressor(regressor=Ridge(alpha={a}), "
                 f"func=np.log1p, inverse_func=np.expm1)` does: fit `Ridge(alpha={a})` on `df` and `np.log1p(y)`, "
                 f"predict the first {n} rows and map back with `np.expm1`. Return those predictions.", t.setup,
                 f"from sklearn.linear_model import Ridge\nreg = Ridge(alpha={a}).fit(df, np.log1p(y))\n"
                 f"result = np.expm1(reg.predict(df.iloc[:{n}]))", require=["Ridge", "log1p", "expm1"], tol=1e-6)
    if task == "sqrt":
        return P(f"{t.desc} (never negative). Fit `TransformedTargetRegressor(regressor=Ridge(alpha={a}), func=np.sqrt, "
                 f"inverse_func=np.square)` on `df, y` (the model learns `sqrt(y)`) and return its predictions for the "
                 f"first {n} rows.", t.setup,
                 f"{imp}ttr = TransformedTargetRegressor(regressor=Ridge(alpha={a}), func=np.sqrt, inverse_func=np.square)\n"
                 f"result = ttr.fit(df, y).predict(df.iloc[:{n}])", require=["TransformedTargetRegressor"], tol=1e-6)
    return P(f"{t.desc} (never negative). Does a log target help? Return the training R² `(plain, logged)` of "
             f"`Ridge(alpha={a})` fitted on `df, y` and of `TransformedTargetRegressor(regressor=Ridge(alpha={a}), "
             f"func=np.log1p, inverse_func=np.expm1)` fitted on `df, y`.", t.setup,
             f"{imp}plain = Ridge(alpha={a}).fit(df, y).score(df, y)\n"
             f"ttr = TransformedTargetRegressor(regressor=Ridge(alpha={a}), func=np.log1p, inverse_func=np.expm1)\n"
             "result = (plain, ttr.fit(df, y).score(df, y))", require=["TransformedTargetRegressor"], tol=1e-6)


@template(M, "easy")
def union_by_hand(rng):
    setup, desc, d = _small_numeric(rng)
    p, k = rng.randint(1, 3), rng.randint(1, 3)
    u = f"FeatureUnion([('pca', PCA(n_components={p})), ('best', SelectKBest(k={k}))])"
    imp = ("from sklearn.pipeline import FeatureUnion\nfrom sklearn.decomposition import PCA\n"
           "from sklearn.feature_selection import SelectKBest\n")
    if rng.random() < 0.5:
        return P(f"{desc}. A `FeatureUnion` is just side-by-side transformers. Fit-transform `{u}` on `X, y`, rebuild "
                 f"the same matrix with `np.hstack` of `PCA(n_components={p}).fit_transform(X)` and "
                 f"`SelectKBest(k={k}).fit_transform(X, y)`, and return the largest absolute difference (≈ 0).", setup,
                 f"{imp}U = {u}.fit_transform(X, y)\n"
                 f"H = np.hstack([PCA(n_components={p}).fit_transform(X), SelectKBest(k={k}).fit_transform(X, y)])\n"
                 "result = float(np.abs(U - H).max())", require=["FeatureUnion", "hstack"], tol=1e-9)
    w = rng.choice([0.5, 2.0, 10.0])
    return P(f"{desc}. `transformer_weights` multiplies each part's output. Fit-transform "
             f"`{u[:-1]}, transformer_weights={{'pca': {w}}})` on `X, y` and return the first 3 rows.", setup,
             f"{imp}result = {u[:-1]}, transformer_weights={{'pca': {w}}}).fit_transform(X, y)[:3]",
             require=["FeatureUnion", "transformer_weights"], tol=1e-6)


@template(M, "easy")
def model_pipeline_tour(rng):
    t = table(rng, kind="clf", missing=1, n=(100, 220))
    n = rng.randint(2, 5)
    items = [
        ("predict", f"Return `model`'s predicted classes for the first {n} rows of `df`.",
         f"result = model.predict(df.iloc[:{n}])"),
        ("predict_proba", f"Return P(class 1) for the first {n} rows of `df`.",
         f"result = model.predict_proba(df.iloc[:{n}])[:, 1]"),
        ("score", "Return `model`'s training accuracy.", "result = model.score(df, y)"),
        ("intercept_", "Return the classifier's intercept (`model['clf'].intercept_`).", "result = model['clf'].intercept_"),
        ("coef_", "How many weights does the classifier have? Return `model[-1].coef_.shape`.",
         "result = model[-1].coef_.shape"),
        ("transformers_", "Return the names of the fitted transformers of the preprocessing step, as a list "
                          "(`model['prep'].transformers_` holds `(name, transformer, columns)` triples).",
         "result = [name for name, _, _ in model['prep'].transformers_]"),
        ("classes_", "Return the classes the pipeline knows (`model.classes_`, forwarded from its last step) as a list.",
         "result = model.classes_.tolist()"),
        ("transform", "Return the shape of the preprocessed first 2 rows: `model[:-1].transform(df.head(2)).shape`.",
         "result = model[:-1].transform(df.head(2)).shape", {"require": ["transform", "head"]}),
    ]
    return catalog(rng, items, t.setup + "\n" + _model(t),
                   f"{t.desc}. `model` is the guide's pipeline: `preprocess` (median-impute + scale `num_cols`, one-hot "
                   f"`cat_cols`) then `{LR}` as `'clf'`, fitted on `df, y`.", tol=1e-6)


# ================================================================= medium

@template(M, "medium")
def full_pipeline_tasks(rng):
    t = table(rng, kind="clf", missing=rng.randint(1, 2), n=(150, 300))
    new, col, k = _new_rows(rng, t)
    c = rng.choice([0.01, 0.05, 0.2, 5.0])
    m = rng.randint(t.n // 2, 3 * t.n // 4)
    cat = rng.choice(t.cat)
    items = [
        ("get_feature_names_out", "Return a dict mapping each feature that reaches the classifier "
                                  "(`model[:-1].get_feature_names_out()`) to its coefficient.",
         "names = model[:-1].get_feature_names_out()\nresult = dict(zip(names, model.named_steps['clf'].coef_[0]))"),
        ("get_feature_names_out", "Return the names of the 3 features with the largest **absolute** coefficient, "
                                  "largest first.",
         "names = model[:-1].get_feature_names_out()\n"
         "result = list(names[np.argsort(-np.abs(model.named_steps['clf'].coef_[0]))[:3]])"),
        ("predict_proba", f"Return P(class 1) for the {k} rows of `new` (the first has a `{col}` never seen in training: "
                          f"the encoder ignores it).", "result = model.predict_proba(new)[:, 1]",
         {"setup": new}),
        ("named_transformers_", "Return the medians the numeric imputer learned: go through `model['prep']`, its "
                                "`named_transformers_['num']` (a pipeline) and that pipeline's `'impute'` step.",
         "result = model['prep'].named_transformers_['num']['impute'].statistics_"),
        ("categories_", "Return the levels the one-hot encoder learned, one list per text column (a list of lists).",
         "result = [list(c) for c in model['prep'].named_transformers_['cat'].categories_]"),
        ("n_features_in_", "How many features does the classifier see? Return `n_features_in_` of the last step, and of "
                           "the whole pipeline, as a tuple `(classifier, pipeline)`.",
         "result = (model[-1].n_features_in_, model.n_features_in_)"),
        ("set_params", f"Weaken the regularization: `set_params(clf__C={c})`, refit on `df, y` and return the training "
                       f"accuracy.", f"result = model.set_params(clf__C={c}).fit(df, y).score(df, y)"),
        ("score", f"Refit `model` on the first {m} rows only and return its accuracy on the remaining rows.",
         f"result = model.fit(df.iloc[:{m}], y[:{m}]).score(df.iloc[{m}:], y[{m}:])"),
        ("predict", f"For each level of `{cat}`, how many rows does `model` predict as class 1? Return a dict "
                    f"{{level: count}}.",
         f"pred = model.predict(df)\nresult = {{lv: int(pred[(df[{cat!r}] == lv).to_numpy()].sum()) "
         f"for lv in sorted(df[{cat!r}].unique())}}"),
        ("transform", "Missing values never reach the classifier. Return `(missing_in_df, missing_after_prep)`: the NaN "
                      "count of `df` and of `model[:-1].transform(df)`.",
         "result = (int(df.isna().sum().sum()), int(np.isnan(model[:-1].transform(df)).sum()))"),
        ("output_indices_", "Which output columns of the preprocessing come from the one-hot encoder? Return "
                            "`(start, stop)` of `model['prep'].output_indices_['cat']`.",
         "s = model['prep'].output_indices_['cat']\nresult = (s.start, s.stop)"),
        ("predict_proba", "Return the indices of the 5 rows the model is most sure are class 1, most sure first.",
         "result = np.argsort(-model.predict_proba(df)[:, 1])[:5]"),
    ]
    return catalog(rng, items, t.setup + "\n" + _model(t),
                   f"{t.desc}. `model` is the guide's pipeline: `preprocess` (median-impute + scale `num_cols`, one-hot "
                   f"`cat_cols` with `handle_unknown='ignore'`) then `{LR}` as `'clf'`, fitted on `df, y`.", tol=1e-6)


@template(M, "medium")
def regression_pipeline_tasks(rng):
    t = table(rng, kind="reg", only=POSITIVE, missing=1, n=(120, 260))
    new, col, k = _new_rows(rng, t)
    alphas = sorted(rng.sample([0.01, 0.1, 1.0, 10.0, 100.0], 3))
    m = rng.randint(t.n // 2, 3 * t.n // 4)
    items = [
        ("score", "Return the training R² of `model`.", "result = model.score(df, y)"),
        ("predict", f"Predict the {k} rows of `new` (the first one has a `{col}` never seen in training).",
         "result = model.predict(new)", {"setup": new}),
        ("set_params", f"For each `alpha` in `{alphas}`, set `reg__alpha` with `set_params`, refit on `df, y` and record "
                       f"the training R². Return the list.",
         f"result = [model.set_params(reg__alpha=a).fit(df, y).score(df, y) for a in {alphas}]"),
        ("set_params", "Steps can be swapped too: replace the regressor with `LinearRegression()` "
                       "(`set_params(reg=LinearRegression())`), refit on `df, y` and return the training R².",
         "from sklearn.linear_model import LinearRegression\n"
         "result = model.set_params(reg=LinearRegression()).fit(df, y).score(df, y)"),
        ("score", f"Refit `model` on the first {m} rows and return R² on the rest.",
         f"result = model.fit(df.iloc[:{m}], y[:{m}]).score(df.iloc[{m}:], y[{m}:])"),
        ("TransformedTargetRegressor", "Wrap the whole pipeline: `TransformedTargetRegressor(regressor=model, "
                                       "func=np.log1p, inverse_func=np.expm1)`. Fit it on `df, y` and return its training "
                                       "R².",
         "from sklearn.compose import TransformedTargetRegressor\n"
         "ttr = TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)\n"
         "result = ttr.fit(df, y).score(df, y)"),
        ("get_feature_names_out", "Return the name of the feature with the largest absolute Ridge coefficient.",
         "names = model[:-1].get_feature_names_out()\nresult = names[np.abs(model['reg'].coef_).argmax()]"),
        ("predict", "Return the mean absolute training error of `model`.",
         "result = float(np.abs(y - model.predict(df)).mean())"),
    ]
    return catalog(rng, items, t.setup + "\n" + _model(t),
                   f"{t.desc}. `model` is `preprocess` (median-impute + scale `num_cols`, one-hot `cat_cols` with "
                   f"`handle_unknown='ignore'`) then `Ridge(alpha=1.0)` as `'reg'`, fitted on `df, y`.", tol=1e-6)


@template(M, "medium")
def column_selector_tasks(rng):
    t = table(rng, n=(80, 200))
    sel = ("from sklearn.compose import make_column_transformer, make_column_selector\n"
           "from sklearn.preprocessing import OneHotEncoder, StandardScaler\n")
    ct = ("make_column_transformer((StandardScaler(), make_column_selector(dtype_include='number')),\n"
          "                        (OneHotEncoder(handle_unknown='ignore'), make_column_selector(dtype_exclude='number')))")
    drop = rng.choice(t.num)
    items = [
        ("make_column_transformer", "Selectors make a preprocessor that works on any table without listing columns. "
                                    f"Fit-transform `{ct}` on `df` and return the shape of the output.",
         f"{sel}ct = {ct}\nresult = ct.fit_transform(df).shape"),
        ("make_column_transformer", f"Fit `{ct}` on `df` and return its output feature names as a list (the "
                                    f"automatic transformer names become the prefixes).",
         f"{sel}ct = {ct}\nresult = list(ct.fit(df).get_feature_names_out())"),
        ("make_column_selector", f"The selectors are resolved at fit time. Fit `{ct}` on `df.drop(columns={drop!r})` and "
                                 f"return the columns its first transformer was applied to (`transformers_[0][2]`).",
         f"{sel}ct = {ct}\nresult = list(ct.fit(df.drop(columns={drop!r})).transformers_[0][2])"),
        ("make_column_selector", "Scale the numeric columns of `df` and one-hot the rest with `make_column_transformer` "
                                 "and selectors, then feed that into a model: return the shape of "
                                 "`ct.fit_transform(df)` when `OneHotEncoder(drop='if_binary', sparse_output=False)` "
                                 "is used (2-level columns become a single 0/1 column).",
         f"{sel}ct = make_column_transformer((StandardScaler(), make_column_selector(dtype_include='number')),\n"
         "    (OneHotEncoder(drop='if_binary', sparse_output=False), make_column_selector(dtype_exclude='number')))\n"
         "result = ct.fit_transform(df).shape"),
        ("remainder", "Scale only the numeric columns (`make_column_transformer` with one selector) and pass the text "
                      "columns through untouched with `remainder='passthrough'`. Return the output feature names.",
         f"{sel}ct = make_column_transformer((StandardScaler(), make_column_selector(dtype_include='number')), "
         "remainder='passthrough')\nresult = list(ct.fit(df).get_feature_names_out())"),
        ("sparse_threshold", "A `ColumnTransformer` returns a sparse matrix when the output is sparse enough. With "
                             f"`{ct}`, return `(issparse(default), issparse(dense))` where `dense` uses "
                             f"`sparse_threshold=0` (`scipy.sparse.issparse`).",
         f"{sel}from scipy.sparse import issparse\n"
         "a = make_column_transformer((StandardScaler(), make_column_selector(dtype_include='number')),\n"
         "    (OneHotEncoder(handle_unknown='ignore'), make_column_selector(dtype_exclude='number'))).fit_transform(df)\n"
         "b = make_column_transformer((StandardScaler(), make_column_selector(dtype_include='number')),\n"
         "    (OneHotEncoder(handle_unknown='ignore'), make_column_selector(dtype_exclude='number')), "
         "sparse_threshold=0).fit_transform(df)\nresult = (issparse(a), issparse(b))"),
        ("make_column_selector", "Return the output shape when the numeric columns are **min-max** scaled "
                                 "(`MinMaxScaler`) and the text columns one-hot encoded, both chosen with selectors.",
         "from sklearn.compose import make_column_transformer, make_column_selector\n"
         "from sklearn.preprocessing import OneHotEncoder, MinMaxScaler\n"
         "ct = make_column_transformer((MinMaxScaler(), make_column_selector(dtype_include='number')),\n"
         "    (OneHotEncoder(), make_column_selector(dtype_exclude='number')))\nresult = ct.fit_transform(df).shape"),
    ]
    return catalog(rng, items, t.setup, f"{t.desc}.")


@template(M, "medium")
def pandas_output(rng):
    t = table(rng, missing=1, n=(60, 150))
    n = rng.randint(2, 4)
    items = [
        ("set_output", f"Call `preprocess.set_output(transform='pandas')`, fit-transform `df` and return the first {n} "
                       f"rows of the resulting DataFrame (`head`).",
         f"preprocess.set_output(transform='pandas')\nresult = preprocess.fit_transform(df).head({n})",
         {"require": ["set_output", "head"]}),
        ("set_output", "With pandas output on (`preprocess.set_output(transform='pandas')`), fit-transform `df` and return "
                       "the name of the output column with the largest standard deviation.",
         "preprocess.set_output(transform='pandas')\nout = preprocess.fit_transform(df)\nresult = out.std().idxmax()"),
        ("set_config", "Switch pandas output on **globally** with `set_config(transform_output='pandas')` (from "
                       "`sklearn`), fit-transform `df` with `preprocess`, then switch it back with "
                       "`set_config(transform_output='default')`. Return `(type(out).__name__, list(out.columns))`.",
         "from sklearn import set_config\nset_config(transform_output='pandas')\nout = preprocess.fit_transform(df)\n"
         "set_config(transform_output='default')\nresult = (type(out).__name__, list(out.columns))",
         {"post": "from sklearn import get_config, set_config\n_ok = get_config()['transform_output'] == 'default'\n"
                  "set_config(transform_output='default')\n(_, _ok)"}),
        ("set_output", "Pandas output keeps the row index. Turn it on, transform `df.iloc[10:15]` (after fitting on all "
                       "of `df`) and return the index of the result as a list.",
         "preprocess.set_output(transform='pandas')\npreprocess.fit(df)\n"
         "result = list(preprocess.transform(df.iloc[10:15]).index)"),
    ]
    return catalog(rng, items, t.setup + "\nfrom sklearn import set_config\nset_config(transform_output='default')\n"
                   + IMPORTS + _cols(t) + "\n" + _prep(), f"{t.desc}. `preprocess` is the guide's `ColumnTransformer` "
                                                         f"(median-impute + scale `num_cols`, one-hot `cat_cols`), not "
                                                         f"fitted yet.", tol=1e-6)


@template(M, "medium")
def union_in_pipeline(rng):
    setup, desc, d = _small_numeric(rng)
    p, k = rng.randint(1, 3), rng.randint(1, 3)
    imp = ("from sklearn.pipeline import FeatureUnion, Pipeline\nfrom sklearn.decomposition import PCA\n"
           "from sklearn.feature_selection import SelectKBest\nfrom sklearn.preprocessing import StandardScaler\n"
           "from sklearn.linear_model import LogisticRegression\n")
    pipe = (f"pipe = Pipeline([('scale', StandardScaler()),\n                 ('feats', FeatureUnion([('pca', "
            f"PCA(n_components={p})), ('best', SelectKBest(k={k}))])),\n                 ('clf', {LR})])")
    m = rng.randint(80, 120)
    items = [
        ("FeatureUnion", f"Fit `pipe` on `X, y` and return its training accuracy.", "result = pipe.fit(X, y).score(X, y)",
         {"require": ["fit", "score"]}),
        ("get_feature_names_out", "Fit `pipe` on `X, y` and return the names of the features reaching the classifier "
                                  "(`pipe[:-1].get_feature_names_out()`) as a list.",
         "result = list(pipe.fit(X, y)[:-1].get_feature_names_out())"),
        ("named_transformers", "Fit `pipe` on `X, y` and return the indices of the columns the `'best'` part of the union "
                               "kept.",
         "result = pipe.fit(X, y)['feats'].named_transformers['best'].get_support(indices=True)"),
        ("set_params", f"Nested names reach inside the union: set `feats__pca__n_components=1` and `feats__best__k=1`, "
                       f"fit on `X, y` and return the shape of `pipe[:-1].transform(X)`.",
         "pipe.set_params(feats__pca__n_components=1, feats__best__k=1).fit(X, y)\nresult = pipe[:-1].transform(X).shape"),
        ("explained_variance_ratio_", "Fit `pipe` on `X, y` and return the explained variance ratio of the union's PCA.",
         "result = pipe.fit(X, y)['feats'].named_transformers['pca'].explained_variance_ratio_"),
        ("score", f"Fit `pipe` on the first {m} rows and return its accuracy on the rest.",
         f"result = pipe.fit(X[:{m}], y[:{m}]).score(X[{m}:], y[{m}:])"),
    ]
    return catalog(rng, items, setup + "\n" + imp + pipe, f"{desc}. `pipe` scales, then stacks {p} PCA component(s) "
                                                          f"with the {k} best original feature(s), then classifies.",
                   tol=1e-6)


@template(M, "medium")
def nested_param_names(rng):
    t = table(rng, kind="clf", missing=1, n=(100, 200))
    s = rng.choice(["mean", "most_frequent", "constant"])
    items = [
        ("get_params", "Every nested setting has a `__` path. Return the sorted list of `model.get_params()` keys that "
                       "contain `'impute__'`.",
         "result = sorted(k for k in model.get_params() if 'impute__' in k)"),
        ("set_params", f"Change the numeric imputation strategy through the whole pipeline: "
                       f"`set_params(prep__num__impute__strategy={s!r})`, refit on `df, y` and return the imputer's "
                       f"`statistics_`.",
         f"model.set_params(prep__num__impute__strategy={s!r}).fit(df, y)\n"
         "result = model['prep'].named_transformers_['num']['impute'].statistics_"),
        ("set_params", "Turn off the text columns: `set_params(prep__cat='drop')`, refit on `df, y` and return how many "
                       "features the classifier now sees.",
         "result = model.set_params(prep__cat='drop').fit(df, y)[-1].n_features_in_"),
        ("set_params", "Swap the one-hot encoder for `OneHotEncoder(drop='first', sparse_output=False)` with "
                       "`set_params(prep__cat=...)`, refit and return the number of classifier coefficients.",
         "model.set_params(prep__cat=OneHotEncoder(drop='first', sparse_output=False)).fit(df, y)\n"
         "result = model[-1].coef_.size"),
        ("get_params", "Return `model.get_params()['prep__num__scale__with_mean']` and "
                       "`model.get_params()['clf__C']` as a tuple.",
         "p = model.get_params()\nresult = (p['prep__num__scale__with_mean'], p['clf__C'])"),
    ]
    return catalog(rng, items, t.setup + "\n" + _model(t, fitted=False),
                   f"{t.desc}. `model` is the guide's pipeline: `'prep'` (a `ColumnTransformer` whose `'num'` part is a "
                   f"pipeline of `'impute'` and `'scale'`, and whose `'cat'` part one-hot encodes) then `'clf'`, "
                   f"not fitted.", tol=1e-6)


# ================================================================= hard

@template(M, "hard")
def leaky_preprocessing(rng):
    t = table(rng, kind="clf", numeric_only=True, missing=rng.randint(1, 2), n=(150, 300))
    m = rng.randint(t.n // 2, 3 * t.n // 4)
    task = rng.choice(["holdout", "median", "cv"])
    leak = ("```\nimp = SimpleImputer(strategy='median').fit(df)        # sees every row\n"
            "sc = StandardScaler().fit(imp.transform(df))           # sees every row\n"
            f"Z = sc.transform(imp.transform(df))\nclf = {LR}.fit(Z[:{m}], y[:{m}])\n"
            f"print(clf.score(Z[{m}:], y[{m}:]))\n```")
    imp = ("from sklearn.pipeline import make_pipeline\nfrom sklearn.impute import SimpleImputer\n"
           "from sklearn.preprocessing import StandardScaler\nfrom sklearn.linear_model import LogisticRegression\n")
    if task == "holdout":
        return P(f"{t.desc}; some values are missing. This evaluation leaks: the imputer and scaler learn from the test "
                 f"rows too.\n{leak}\nFix it with `make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), "
                 f"{LR})` fitted on the first {m} rows only, and return its accuracy on the remaining rows.", t.setup,
                 f"{imp}pipe = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), {LR})\n"
                 f"result = pipe.fit(df.iloc[:{m}], y[:{m}]).score(df.iloc[{m}:], y[{m}:])",
                 require=["make_pipeline|Pipeline"])
    if task == "median":
        return P(f"{t.desc}; some values are missing. Training rows are the first {m}. How far off are the leaky "
                 f"statistics? Return `(leaky, honest)`: the medians a `SimpleImputer(strategy='median')` learns from "
                 f"**all** of `df`, and from the training rows only.", t.setup,
                 "from sklearn.impute import SimpleImputer\nleaky = SimpleImputer(strategy='median').fit(df).statistics_\n"
                 f"honest = SimpleImputer(strategy='median').fit(df.iloc[:{m}]).statistics_\nresult = (leaky, honest)",
                 require=["SimpleImputer", "statistics_"])
    return P(f"{t.desc}; some values are missing. Cross-validation must refit the preprocessing on each training fold, "
             f"which a pipeline does automatically. Return `(leaky, honest)` mean 5-fold `cross_val_score` accuracies "
             f"(from `sklearn.model_selection`, default folds): `leaky` scores `{LR}` on `df` already imputed (median) and "
             f"scaled using all rows; `honest` scores `make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), "
             f"{LR})` on the raw `df`.", t.setup,
             f"{imp}from sklearn.model_selection import cross_val_score\n"
             "Z = StandardScaler().fit_transform(SimpleImputer(strategy='median').fit_transform(df))\n"
             f"leaky = cross_val_score({LR}, Z, y, cv=5).mean()\n"
             f"honest = cross_val_score(make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), {LR}), df, y, "
             "cv=5).mean()\nresult = (leaky, honest)", require=["cross_val_score", "make_pipeline|Pipeline"], tol=1e-6)


@template(M, "hard")
def pipeline_selection_leakage(rng):
    n, d, k, s = rng.choice([50, 60, 80]), rng.choice([300, 500, 800]), rng.choice([5, 10, 20]), seeded(rng)
    setup = (f"g = np.random.default_rng({s})\nX = g.normal(size=({n}, {d}))   # pure noise\n"
             f"y = np.array([0, 1] * {n // 2})\ng.shuffle(y)")
    return P(f"`X` is {n} rows of {d} **pure-noise** features and `y` random balanced labels: no model can beat 50% "
             f"honestly. Return `(leaky, honest)` mean 5-fold `cross_val_score` accuracies: `leaky` picks the {k} best "
             f"features with `SelectKBest(f_classif, k={k})` on **all** rows first and cross-validates `{LR}` on them; "
             f"`honest` cross-validates `make_pipeline(SelectKBest(f_classif, k={k}), {LR})` on `X`, so the selection "
             f"is redone inside each training fold. (The leaky score is wildly optimistic.)", setup,
             "from sklearn.feature_selection import SelectKBest, f_classif\nfrom sklearn.pipeline import make_pipeline\n"
             "from sklearn.linear_model import LogisticRegression\nfrom sklearn.model_selection import cross_val_score\n"
             f"Xk = SelectKBest(f_classif, k={k}).fit_transform(X, y)\nleaky = cross_val_score({LR}, Xk, y, cv=5).mean()\n"
             f"honest = cross_val_score(make_pipeline(SelectKBest(f_classif, k={k}), {LR}), X, y, cv=5).mean()\n"
             "result = (leaky, honest)", require=["SelectKBest", "make_pipeline|Pipeline", "cross_val_score"], tol=1e-6)


@template(M, "hard")
def grid_over_pipeline(rng):
    t = table(rng, kind="clf", missing=1, n=(120, 220))
    cs = sorted(rng.sample([0.01, 0.1, 1.0, 10.0], 2))
    task = rng.choice(["params", "score", "table"])
    grid = f"{{'clf__C': {cs}, 'prep__num__impute__strategy': ['mean', 'median']}}"
    head = (f"{t.desc}. `model` is the guide's pipeline (not fitted). Tune it with `GridSearchCV(model, {grid}, cv=3)` "
            f"(from `sklearn.model_selection`): pipeline parameters use the same `<step>__<param>` names as "
            f"`set_params`, so preprocessing settings are tuned together with the model, leakage-free.")
    sol = f"from sklearn.model_selection import GridSearchCV\ngs = GridSearchCV(model, {grid}, cv=3).fit(df, y)\n"
    if task == "params":
        return P(f"{head} Fit it on `df, y` and return `best_params_`.", t.setup + "\n" + _model(t, fitted=False),
                 sol + "result = gs.best_params_", require=["GridSearchCV"])
    if task == "score":
        return P(f"{head} Fit it on `df, y` and return `(best_score_, best_estimator_ training accuracy on df, y)`.",
                 t.setup + "\n" + _model(t, fitted=False),
                 sol + "result = (gs.best_score_, gs.best_estimator_.score(df, y))", require=["GridSearchCV"], tol=1e-6)
    return P(f"{head} Fit it on `df, y` and return the mean test score of every candidate as a dict "
             f"{{(C, strategy): score}} built from `cv_results_['params']` and `cv_results_['mean_test_score']`.",
             t.setup + "\n" + _model(t, fitted=False),
             sol + "res = gs.cv_results_\nresult = {(p['clf__C'], p['prep__num__impute__strategy']): s "
                   "for p, s in zip(res['params'], res['mean_test_score'])}", require=["GridSearchCV"], tol=1e-6)


@template(M, "hard")
def ct_by_hand(rng):
    t = table(rng, missing=1, n=(40, 100))
    task = rng.choice(["num", "cat", "all"])
    head = (f"{t.desc}; some numeric values are missing. `out = preprocess.fit_transform(df)` where `preprocess` is the "
            f"guide's `ColumnTransformer` (median-impute + standardize `num_cols`, then one-hot `cat_cols`; categories in "
            f"sorted order). Rebuild by hand with pandas/NumPy (no scikit-learn):")
    num = ("A = df[num_cols].fillna(df[num_cols].median())\nA = ((A - A.mean()) / A.std(ddof=0)).to_numpy()\n")
    cat = "B = pd.get_dummies(df[cat_cols], dtype=float).to_numpy()\n"
    if task == "num":
        return P(f"{head} the numeric block: fill each column's NaNs with its median (`fillna`), then standardize "
                 f"with the population standard deviation. Return it as an array.", t.setup + "\n" + _cols(t),
                 num + "result = A", require=["fillna", "median"], tol=1e-9)
    if task == "cat":
        return P(f"{head} the one-hot block (`pd.get_dummies(..., dtype=float)` sorts the levels like the encoder). "
                 f"Return it as an array.", t.setup + "\n" + _cols(t), cat + "result = B",
                 require=["get_dummies"], tol=1e-9)
    return P(f"{head} the **whole** output: the numeric block (median `fillna`, standardize with the population std) "
             f"followed by the one-hot block (`pd.get_dummies(..., dtype=float)`). Return the matrix (`np.hstack`).",
             t.setup + "\n" + _cols(t), num + cat + "result = np.hstack([A, B])",
             require=["fillna", "get_dummies"], tol=1e-9)


@template(M, "hard")
def coef_per_original_column(rng):
    t = table(rng, kind="clf", missing=1, n=(150, 280))
    task = rng.choice(["dict", "top"])
    sol = ("names = model[:-1].get_feature_names_out()\ncoef = np.abs(model['clf'].coef_[0])\n"
           "orig = [n.split('__', 1)[1] for n in names]\n"
           "orig = [o if o in num_cols else next(c for c in cat_cols if o.startswith(c + '_')) for o in orig]\n"
           "total = {}\nfor o, w in zip(orig, coef):\n    total[o] = total.get(o, 0.0) + w\n")
    head = (f"{t.desc}. `model` (the guide's pipeline, `{LR}` as `'clf'`) is fitted on `df, y`. One text column becomes "
            f"several one-hot features. Map each output feature (`model[:-1].get_feature_names_out()`, like "
            f"`'num__age'` or `'cat__plan_basic'`) back to its **original** column and add up the absolute coefficients "
            f"per original column.")
    if task == "dict":
        return P(f"{head} Return the dict {{original column: total |coef|}}.", t.setup + "\n" + _model(t),
                 sol + "result = total", require=["get_feature_names_out"], tol=1e-6)
    return P(f"{head} Return the original columns sorted by that total, largest first.", t.setup + "\n" + _model(t),
             sol + "result = sorted(total, key=total.get, reverse=True)", require=["get_feature_names_out"])


@template(M, "hard")
def broken_pipelines(rng):
    t = table(rng, kind="clf", missing=1, n=(100, 220))
    new, col, k = _new_rows(rng, t)
    c = rng.choice([0.05, 0.3, 3.0])
    items = [
        ("ColumnTransformer", f"This crashes (`StandardScaler` can't scale text, and `LogisticRegression` can't take "
                              f"NaN):\n```\nmodel = make_pipeline(StandardScaler(), {LR}).fit(df, y)\n```\nFix it with "
                              f"the guide's preprocessing (median-impute + scale `num_cols`, one-hot `cat_cols` with "
                              f"`handle_unknown='ignore'`) in a `ColumnTransformer` before the classifier, and return the "
                              f"training accuracy.",
         f"{IMPORTS}from sklearn.linear_model import LogisticRegression\n{_prep()}\n"
         f"model = Pipeline([('prep', preprocess), ('clf', {LR})]).fit(df, y)\nresult = model.score(df, y)"),
        ("handle_unknown", f"This model crashes on `new` (its first row has a `{col}` level never seen in training):\n"
                           f"```\nprep = ColumnTransformer([('num', SimpleImputer(), num_cols), "
                           f"('cat', OneHotEncoder(), cat_cols)])\nmodel = make_pipeline(prep, {LR}).fit(df, y)\n"
                           f"model.predict(new)\n```\nFix the encoder so unseen levels are ignored and return the "
                           f"predictions for `new`.",
         "from sklearn.compose import ColumnTransformer\nfrom sklearn.impute import SimpleImputer\n"
         "from sklearn.preprocessing import OneHotEncoder\nfrom sklearn.pipeline import make_pipeline\n"
         "from sklearn.linear_model import LogisticRegression\n"
         "prep = ColumnTransformer([('num', SimpleImputer(), num_cols), ('cat', OneHotEncoder(handle_unknown='ignore'), "
         f"cat_cols)])\nmodel = make_pipeline(prep, {LR}).fit(df, y)\nresult = model.predict(new)",
         {"setup": new}),
        ("remainder", "This preprocessing silently loses the text columns:\n```\nprep = ColumnTransformer([('num', "
                      "SimpleImputer(strategy='median'), num_cols)])\n```\nKeep them unchanged instead "
                      "(`remainder='passthrough'`), fit on `df`, and return the output feature names.",
         "from sklearn.compose import ColumnTransformer\nfrom sklearn.impute import SimpleImputer\n"
         "prep = ColumnTransformer([('num', SimpleImputer(strategy='median'), num_cols)], remainder='passthrough')\n"
         "result = list(prep.fit(df).get_feature_names_out())"),
        ("set_params", f"`model` is the guide's (unfitted) pipeline. This raises `ValueError: Invalid parameter 'C'`:\n"
                       f"```\nmodel.set_params(C={c})\n```\nUse the right `<step>__<param>` name, then fit on `df, y` "
                       f"and return `(model['clf'].C, training accuracy)`.",
         f"model.set_params(clf__C={c}).fit(df, y)\nresult = (model['clf'].C, model.score(df, y))",
         {"setup": _model(t, fitted=False)}),
        ("transform", f"`model` is the guide's pipeline fitted on `df, y`. This code wants the preprocessed version of "
                      f"`new` but refits the preprocessing on it, so `new` is scaled with its own statistics:\n```\n"
                      f"Z = model['prep'].fit_transform(new)\n```\nReturn the correctly preprocessed `new` (with the "
                      f"statistics learned from `df`).", "result = model['prep'].transform(new)",
         {"setup": _model(t) + "\n" + new}),
        ("DataFrame", f"`model` is the guide's pipeline fitted on the DataFrame `df`. Columns are picked **by name**, so "
                      f"this crashes:\n```\nrows = df.to_numpy()[:{k}]\nmodel.predict(rows)\n```\nRebuild a DataFrame "
                      f"with `df`'s columns and return the predictions.",
         f"rows = df.to_numpy()[:{k}]\nresult = model.predict(pd.DataFrame(rows, columns=df.columns))",
         {"setup": _model(t)}),
        ("make_pipeline", f"This raises a `TypeError` (every step but the last must be a transformer):\n```\n"
                          f"pipe = make_pipeline(SimpleImputer(strategy='median'), {LR}, StandardScaler())\n"
                          f"pipe.fit(df[num_cols], y)\n```\nPut the steps in a working order (impute, scale, classify), "
                          f"fit on `df[num_cols], y` and return the training accuracy.",
         "from sklearn.pipeline import make_pipeline\nfrom sklearn.impute import SimpleImputer\n"
         "from sklearn.preprocessing import StandardScaler\nfrom sklearn.linear_model import LogisticRegression\n"
         f"pipe = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), {LR})\n"
         "result = pipe.fit(df[num_cols], y).score(df[num_cols], y)"),
        ("sparse_output", "This fails because pandas output can't hold a sparse matrix (and the one-hot encoder is "
                          "sparse by default):\n```\nprep = ColumnTransformer([('cat', OneHotEncoder(), cat_cols)])\n"
                          "prep.set_output(transform='pandas').fit_transform(df)\n```\nMake the encoder dense and return "
                          "the column sums of the resulting DataFrame (a Series: how many rows have each level).",
         "from sklearn.compose import ColumnTransformer\nfrom sklearn.preprocessing import OneHotEncoder\n"
         "prep = ColumnTransformer([('cat', OneHotEncoder(sparse_output=False), cat_cols)])\n"
         "result = prep.set_output(transform='pandas').fit_transform(df).sum()"),
    ]
    return catalog(rng, items, t.setup + "\n" + _cols(t), f"{t.desc}; a few numeric values are missing.", tol=1e-6)


@template(M, "hard")
def function_transformer_column(rng):
    t = table(rng, kind="reg", numeric_only=True, only=("bikes", "servers", "phones", "houses"), n=(100, 250))
    skew = rng.choice(t.num)
    rest = [c for c in t.num if c != skew]
    task = rng.choice(["compare", "names"])
    imp = ("from sklearn.compose import ColumnTransformer\nfrom sklearn.preprocessing import FunctionTransformer, "
           "StandardScaler\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.linear_model import Ridge\n")
    ct = (f"ct = ColumnTransformer([('log', make_pipeline(FunctionTransformer(np.log1p, feature_names_out='one-to-one'), "
          f"StandardScaler()), [{skew!r}]), ('num', StandardScaler(), {lit(rest)})])")
    setup = t.setup + f"\ndf[{skew!r}] = np.expm1(df[{skew!r}] / df[{skew!r}].max() * 6)   # make it very skewed"
    if task == "compare":
        return P(f"{t.desc}. Column `{skew}` was made very skewed. Compare two pipelines ending in `Ridge(alpha=1.0)`: "
                 f"plain standard scaling of every column, versus a `ColumnTransformer` that applies "
                 f"`make_pipeline(FunctionTransformer(np.log1p), StandardScaler())` to `{skew}` (as `'log'`) and "
                 f"`StandardScaler()` to the others (as `'num'`). Return their training R² `(plain, logged)`.", setup,
                 f"{imp}{ct}\nplain = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(df, y).score(df, y)\n"
                 "logged = make_pipeline(ct, Ridge(alpha=1.0)).fit(df, y).score(df, y)\nresult = (plain, logged)",
                 require=["FunctionTransformer", "ColumnTransformer"], tol=1e-6)
    return P(f"{t.desc}. Column `{skew}` was made very skewed. Build a `ColumnTransformer` that applies "
             f"`make_pipeline(FunctionTransformer(np.log1p, feature_names_out='one-to-one'), StandardScaler())` to `{skew}` "
             f"(named `'log'`; `feature_names_out` lets the names flow through) and `StandardScaler()` to the other "
             f"numeric columns (named `'num'`), fit `make_pipeline(ct, Ridge(alpha=1.0))` "
             f"on `df, y` and return a dict {{output feature name: coefficient}}.", setup,
             f"{imp}{ct}\npipe = make_pipeline(ct, Ridge(alpha=1.0)).fit(df, y)\n"
             "result = dict(zip(pipe[:-1].get_feature_names_out(), pipe[-1].coef_))",
             require=["FunctionTransformer", "get_feature_names_out"], tol=1e-6)


@template(M, "hard")
def log_target_workflows(rng):
    t = table(rng, kind="reg", only=POSITIVE, missing=1, n=(120, 260))
    m = rng.randint(t.n // 2, 3 * t.n // 4)
    nq = rng.choice([20, 50])
    ttr = "TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)"
    imp = "from sklearn.compose import TransformedTargetRegressor\n"
    items = [
        ("inverse_func", f"This raises a `ValueError`:\n```\nttr = TransformedTargetRegressor(regressor=model, "
                         f"func=np.log1p).fit(df, y)\n```\nscikit-learn needs the way back too. Fix it and return the "
                         f"training predictions for the first 3 rows.",
         f"{imp}ttr = {ttr}.fit(df, y)\nresult = ttr.predict(df.iloc[:3])"),
        ("TransformedTargetRegressor", f"Held-out comparison: fit both `model` and `{ttr}` on the first {m} rows and "
                                       f"return their R² on the remaining rows as `(plain, logged)` (use `clone(model)` "
                                       f"for one of them so they don't share a fitted object).",
         f"{imp}from sklearn.base import clone\na = clone(model).fit(df.iloc[:{m}], y[:{m}]).score(df.iloc[{m}:], y[{m}:])\n"
         f"b = {ttr}.fit(df.iloc[:{m}], y[:{m}]).score(df.iloc[{m}:], y[{m}:])\nresult = (a, b)"),
        ("QuantileTransformer", f"Any transformer can reshape the target: fit `TransformedTargetRegressor(regressor="
                                f"model, transformer=QuantileTransformer(n_quantiles={nq}, output_distribution='normal'))` "
                                f"on `df, y` and return its training R².",
         f"{imp}from sklearn.preprocessing import QuantileTransformer\n"
         f"ttr = TransformedTargetRegressor(regressor=model, transformer=QuantileTransformer(n_quantiles={nq}, "
         "output_distribution='normal'))\nresult = ttr.fit(df, y).score(df, y)"),
        ("regressor_", f"Fit `{ttr}` on `df, y`. Its inner pipeline was trained on `log1p(y)`: return the training R² "
                       f"of `regressor_` **on the log scale** (`regressor_.score(df, np.log1p(y))`) and of the wrapper on "
                       f"the original scale, as a tuple.",
         f"{imp}ttr = {ttr}.fit(df, y)\nresult = (ttr.regressor_.score(df, np.log1p(y)), ttr.score(df, y))"),
        ("mean_absolute_error", f"Return the training mean absolute error `(plain, logged)` of `model` and of `{ttr}`, "
                                f"both fitted on `df, y` (`mean_absolute_error` from `sklearn.metrics`).",
         f"{imp}from sklearn.metrics import mean_absolute_error\nfrom sklearn.base import clone\n"
         f"a = mean_absolute_error(y, clone(model).fit(df, y).predict(df))\n"
         f"b = mean_absolute_error(y, {ttr}.fit(df, y).predict(df))\nresult = (a, b)"),
    ]
    return catalog(rng, items, t.setup + "\n" + _model(t, fitted=False),
                   f"{t.desc} (never negative). `model` is `preprocess` (median-impute + scale `num_cols`, one-hot "
                   f"`cat_cols`) then `Ridge(alpha=1.0)` as `'reg'`, not fitted.", tol=1e-6)


@template(M, "hard")
def pipeline_workflows(rng):
    t = table(rng, kind="clf", missing=1, n=(150, 260))
    s = seeded(rng)
    col = rng.choice(t.num + t.cat)
    items = [
        ("cross_val_score", "Return the 5 fold accuracies of `cross_val_score(model, df, y, cv=5)` (from "
                            "`sklearn.model_selection`): every fold refits the imputer, scaler and encoder on its "
                            "training part.",
         "from sklearn.model_selection import cross_val_score\nresult = cross_val_score(model, df, y, cv=5)"),
        ("cross_val_predict", "Return the out-of-fold predicted probabilities of class 1: `cross_val_predict(model, df, "
                              "y, cv=5, method='predict_proba')[:, 1]`.",
         "from sklearn.model_selection import cross_val_predict\n"
         "result = cross_val_predict(model, df, y, cv=5, method='predict_proba')[:, 1]"),
        ("cross_val_score", "Do the text columns help? Return the mean 5-fold CV accuracy `(with_text, without_text)`, "
                            "where the second drops them via `set_params(prep__cat='drop')` (use `clone` so the "
                            "first model is unchanged).",
         "from sklearn.model_selection import cross_val_score\nfrom sklearn.base import clone\n"
         "a = cross_val_score(model, df, y, cv=5).mean()\n"
         "b = cross_val_score(clone(model).set_params(prep__cat='drop'), df, y, cv=5).mean()\nresult = (a, b)"),
        ("score", f"Permutation importance on **raw** columns (possible because the pipeline takes the raw table): fit "
                  f"`model` on `df, y`, shuffle column `{col}` of a copy of `df` with `np.random.default_rng({s})."
                  f"permutation`, and return the drop in training accuracy.",
         f"model.fit(df, y)\nd2 = df.copy()\nd2[{col!r}] = np.random.default_rng({s}).permutation(d2[{col!r}].to_numpy())\n"
         "result = model.score(df, y) - model.score(d2, y)"),
        ("permutation_importance", f"Use `permutation_importance(model, df, y, n_repeats=3, random_state={s})` (from "
                                  f"`sklearn.inspection`) on `model` fitted on `df, y`, and return a dict {{raw column: "
                                  f"importances_mean}}.",
         f"from sklearn.inspection import permutation_importance\nmodel.fit(df, y)\n"
         f"r = permutation_importance(model, df, y, n_repeats=3, random_state={s})\n"
         "result = dict(zip(df.columns, r.importances_mean))"),
        ("predict_proba", "Choose a threshold on held-out data: fit on the first 70% of rows (`int(0.7 * len(df))`), "
                          "then among the thresholds 0.3, 0.4, 0.5, 0.6, 0.7 return the one with the best accuracy on "
                          "the rest (predict 1 when P(class 1) ≥ threshold; the first on ties).",
         "h = int(0.7 * len(df))\np = model.fit(df.iloc[:h], y[:h]).predict_proba(df.iloc[h:])[:, 1]\n"
         "ts = [0.3, 0.4, 0.5, 0.6, 0.7]\naccs = [((p >= t).astype(int) == y[h:]).mean() for t in ts]\n"
         "result = ts[int(np.argmax(accs))]"),
        ("set_params", "Is scaling the numbers worth it here? Return the mean 5-fold CV accuracy with scaling and with "
                       "`set_params(prep__num__scale='passthrough')` (on a `clone`), as a tuple.",
         "from sklearn.model_selection import cross_val_score\nfrom sklearn.base import clone\n"
         "a = cross_val_score(model, df, y, cv=5).mean()\n"
         "b = cross_val_score(clone(model).set_params(prep__num__scale='passthrough'), df, y, cv=5).mean()\n"
         "result = (a, b)"),
        ("predict", "Fit `model` on `df, y`. Return the training accuracy separately for rows that had at least one "
                    "missing value and rows that had none, as a tuple `(with_missing, complete)`.",
         "pred = model.fit(df, y).predict(df)\nmiss = df.isna().any(axis=1).to_numpy()\n"
         "result = (float((pred[miss] == y[miss]).mean()), float((pred[~miss] == y[~miss]).mean()))"),
    ]
    return catalog(rng, items, t.setup + "\n" + _model(t, fitted=False),
                   f"{t.desc}. `model` is the guide's pipeline (`'prep'`: median-impute + scale `num_cols` as `'num'`, "
                   f"one-hot `cat_cols` as `'cat'`; then `{LR}` as `'clf'`), not fitted.", tol=1e-6)


# ================================================================= challenge

@template(M, "challenge")
def onehot_column_layout(rng):
    k, n, q, s = rng.randint(2, 3), rng.randint(8, 12), rng.randint(4, 6), seeded(rng)
    mode = rng.choice(["ignore", "min_frequency", "max_categories"])
    setup = (f"g = np.random.default_rng({s})\nX = g.integers(0, 6, ({n}, {k})) * 10\n"
             f"X_new = g.integers(0, 7, ({q}, {k})) * 10\nmode = {mode!r}")
    if mode == "min_frequency":
        setup += f"\nm = {rng.randint(2, 3)}"
        tests_extra = "\nm = 40"
    elif mode == "max_categories":
        setup += f"\nm = {rng.randint(2, 4)}"
        tests_extra = "\nm = 2000"
    else:
        tests_extra = ""
    core = ("res = np.full(X_new.shape, -1)\noffset = 0\nfor j in range(X.shape[1]):\n"
            "    vals, counts = np.unique(X[:, j], return_counts=True)\n")
    if mode == "ignore":
        core += ("    pos = np.searchsorted(vals, X_new[:, j])\n    pos = np.minimum(pos, len(vals) - 1)\n"
                 "    hit = vals[pos] == X_new[:, j]\n    res[hit, j] = offset + pos[hit]\n    offset += len(vals)\n")
    else:
        if mode == "min_frequency":
            core += "    infreq = counts < m\n"
        else:
            core += ("    infreq = np.zeros(len(vals), dtype=bool)\n    if m < len(vals) + 1:\n"
                     "        infreq[np.argsort(counts, kind='mergesort')[:len(vals) - (m - 1)]] = True\n")
        core += ("    freq = vals[~infreq]\n    has_inf = bool(infreq.any())\n"
                 "    pos = np.searchsorted(freq, X_new[:, j])\n    pos_c = np.minimum(pos, max(len(freq) - 1, 0))\n"
                 "    hit = (pos < len(freq)) & (freq[pos_c] == X_new[:, j]) if len(freq) else np.zeros(len(X_new), bool)\n"
                 "    res[hit, j] = offset + pos_c[hit]\n"
                 "    if has_inf:\n        res[~hit, j] = offset + len(freq)\n"
                 "    offset += len(freq) + has_inf\n")
    sol = core + "result = res"
    rule = {
        "ignore": "`OneHotEncoder(handle_unknown='ignore')`: each feature's levels are sorted; a level never seen in "
                  "training produces no hot column (answer -1 for that feature)",
        "min_frequency": "`OneHotEncoder(min_frequency=m, handle_unknown='infrequent_if_exist')`: levels seen fewer "
                         "than `m` times in training are merged into one *infrequent* column placed after the feature's "
                         "frequent levels (which stay sorted); unseen levels also go to the infrequent column if the "
                         "feature has one, else they produce no hot column (-1)",
        "max_categories": "`OneHotEncoder(max_categories=m, handle_unknown='infrequent_if_exist')`: if a feature has "
                          "more than `m - 1` levels, only its `m - 1` most frequent levels keep a column (ties in count: "
                          "the **larger** level wins, as scikit-learn's stable sort does) and all others share one "
                          "*infrequent* column placed after the frequent ones (sorted); unseen levels go to the "
                          "infrequent column if there is one, else they produce no hot column (-1)",
    }[mode]
    tests = [f"g = np.random.default_rng({s + i})\nX = g.integers(0, 3000, (100000, 2)) ** 2 // 7\n"
             f"X_new = g.integers(0, 3100, (100000, 2)) ** 2 // 7\nmode = {mode!r}{tests_extra}" for i in (1, 2)]
    return challenge(
        "Finding the hot column",
        f"An encoder is fitted on the integer-coded categorical matrix `X` and then transforms `X_new`. The output "
        f"concatenates the features' blocks in order (feature 0's columns first). The encoder is {rule}. For every "
        f"row of `X_new` and every feature, return the index of the output column that holds the 1 (or -1), as an "
        f"int array of shape `X_new.shape`. (`mode` = {mode!r}.)",
        setup, sol, tests, "rows of X and X_new ≤ 10^5, 2-3 features, values < 2·10^6",
        speed="O(n log n): sort/unique each column once, then binary search", answer_desc=" (an int array)")


@template(M, "challenge")
def oof_impute_scale(rng):
    n, K, s = rng.randint(9, 14), rng.randint(3, 4), seeded(rng)
    split = rng.choice(["kfold", "predefined"])
    report = rng.choice(["z", "scale"])
    setup = (f"g = np.random.default_rng({s})\nx = g.normal(20, 5, {n})\nx[g.random({n}) < 0.25] = np.nan\n"
             f"K = {K}\nreport = {report!r}")
    if split == "predefined":
        setup += f"\nfold = g.integers(-1, K, {n})\nfold[:K] = np.arange(K)   # every fold non-empty"
        fold_code = "f = fold\n"
        how = ("`fold[i]` gives row `i`'s test fold (`PredefinedSplit(fold)`: fold `-1` means the row is never a test "
               "row, it is always used for training); fold `f` trains on every row with `fold != f`")
    else:
        fold_code = ("n = len(x)\nsizes = np.full(K, n // K)\nsizes[:n % K] += 1\n"
                     "f = np.repeat(np.arange(K), sizes)\n")
        how = ("folds are `KFold(n_splits=K)` without shuffling: contiguous blocks, the first `n % K` folds one row "
               "longer; fold `f` trains on all rows outside block `f`")
    sol = (fold_code +
           "obs = ~np.isnan(x)\nv = np.where(obs, x, 0.0)\nfm = f >= 0\n"
           "c_all, s_all, q_all, n_all = obs.sum(), v.sum(), (v * v).sum(), len(x)\n"
           "cf = np.bincount(f[fm], weights=obs[fm].astype(float), minlength=K)\n"
           "sf = np.bincount(f[fm], weights=v[fm], minlength=K)\nqf = np.bincount(f[fm], weights=(v * v)[fm], minlength=K)\n"
           "nf = np.bincount(f[fm], minlength=K)\n"
           "c, sm, sq, nt = c_all - cf, s_all - sf, q_all - qf, n_all - nf\nmu = sm / c\n"
           "var = (sq - sm * mu) / nt\nsd = np.sqrt(np.maximum(var, 0))\nsd[sd == 0] = 1.0\n")
    if report == "scale":
        sol += "result = sd"
        what = ("return the `scale_` each fold's scaler learns, as an array of length `K`")
    else:
        sol += ("z = np.zeros(len(x))\nt = fm & obs\nz[t] = (x[t] - mu[f[t]]) / sd[f[t]]\n"
                "result = z[fm]")
        what = ("return the transformed value of every row in its own test fold, in row order, skipping rows that are "
                "never a test row (a missing reading becomes the fold's mean, i.e. 0 after scaling)")
    tests = []
    for i in (1, 2):
        tt = (f"g = np.random.default_rng({s + i})\nx = g.normal(20, 5, 200000)\n"
              f"x[g.random(200000) < 0.25] = np.nan\nK = 2000\nreport = {report!r}")
        if split == "predefined":
            tt += "\nfold = g.integers(-1, K, 200000)\nfold[:K] = np.arange(K)"
        tests.append(tt)
    return challenge(
        "Leak-free preprocessing in every fold",
        f"Cross-validating `make_pipeline(SimpleImputer(strategy='mean'), StandardScaler(), model)` on a single feature "
        f"`x` (with NaNs) refits the imputer and the scaler on each fold's training rows. There are `K` folds; {how}. "
        f"Each fold's imputer learns the mean of the training rows' observed values and fills their NaNs with it; "
        f"the scaler then learns the mean and population standard deviation of the **imputed** training column (a "
        f"zero spread becomes scale 1). Without running the folds one by one, {what}. (`report` = {report!r}.)",
        setup, sol, tests, "len(x) ≤ 2·10^5, K ≤ 2000, every training part has at least two observed values",
        speed="O(n + K): per-fold sums, subtracted from the totals", answer_desc=" (a float array)", tol=1e-6)


@template(M, "challenge")
def causal_imputation(rng):
    n, s = rng.randint(10, 15), seeded(rng)
    strategy = rng.choice(["median", "most_frequent"])
    setup = (f"g = np.random.default_rng({s})\nx = g.integers(0, 6, {n}).astype(float)\nx[g.random({n}) < 0.3] = np.nan\n"
             f"fallback = -1.0\nstrategy = {strategy!r}")
    if strategy == "median":
        sol = ("import heapq\nlo, hi = [], []   # max-heap (negated) of the lower half, min-heap of the upper half\n"
               "out = []\nfor v in x.tolist():\n    if v != v:   # NaN\n"
               "        out.append(fallback if not lo else -lo[0] if len(lo) > len(hi) else (-lo[0] + hi[0]) / 2)\n"
               "        continue\n    out.append(v)\n"
               "    if lo and v > -lo[0]:\n        heapq.heappush(hi, v)\n    else:\n        heapq.heappush(lo, -v)\n"
               "    if len(lo) > len(hi) + 1:\n        heapq.heappush(hi, -heapq.heappop(lo))\n"
               "    elif len(hi) > len(lo):\n        heapq.heappush(lo, -heapq.heappop(hi))\nresult = np.array(out)")
        rule = ("`SimpleImputer(strategy='median')` (the median of the observed readings; with an even count, the "
                "mean of the two middle ones)")
    else:
        sol = ("cnt = {}\nbest, bc = None, 0\nout = []\nfor v in x.tolist():\n    if v != v:   # NaN\n"
               "        out.append(fallback if best is None else best)\n        continue\n    out.append(v)\n"
               "    c = cnt.get(v, 0) + 1\n    cnt[v] = c\n"
               "    if c > bc or (c == bc and v < best):   # counts only grow: the mode changes only here\n"
               "        best, bc = v, c\nresult = np.array(out)")
        rule = ("`SimpleImputer(strategy='most_frequent')` (the most common observed reading; ties go to the "
                "**smallest** value)")
    tests = [f"g = np.random.default_rng({s + i})\nx = g.integers(0, 50000, 200000).astype(float)\n"
             f"x[g.random(200000) < 0.3] = np.nan\nfallback = -1.0\nstrategy = {strategy!r}" for i in (1, 2)]
    return challenge(
        "Imputation without peeking",
        f"Readings `x` arrive over time; some are missing (NaN). A production pipeline must not use the future, so "
        f"each missing reading `x[i]` is filled by an imputer fitted on the **observed readings before it** "
        f"(`x[:i]`, NaNs ignored): {rule}. If no reading has been observed yet, use `fallback`. Return `x` with every "
        f"NaN filled (observed readings unchanged). (`strategy` = {strategy!r}.)",
        setup, sol, tests, "len(x) ≤ 2·10^5", speed="O(n log n): two heaps for the median, running counts for the mode",
        answer_desc=" (a float array)")
