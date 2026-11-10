"""4.07 Feature selection: univariate filters (SelectKBest, SelectPercentile, f_classif, chi2, mutual information),
VarianceThreshold, model-based selection (RFE, RFECV, SelectFromModel) and SequentialFeatureSelector."""

from ._common import (P, catalog, challenge, lit, seeded, table, template)

M = "4.07"
LR = "LogisticRegression(max_iter=5000)"
FS = "from sklearn.feature_selection import "


def _fs_data(rng, nonneg=False, synth=0.4):
    """X, y, names (rows shuffled). nonneg: only datasets whose features are >= 0 (chi2)."""
    s = seeded(rng)
    if not nonneg and rng.random() < synth:
        n, d = rng.randint(150, 300), rng.randint(8, 14)
        i = rng.randint(2, 4)
        setup = (f"from sklearn.datasets import make_classification\nX, y = make_classification(n_samples={n}, "
                 f"n_features={d}, n_informative={i}, n_redundant=0, shuffle=False, random_state={s})\n"
                 f"names = [f'f{{j}}' for j in range({d})]")
        desc = (f"`X` ({n} × {d}) and `y` (2 classes) come from `make_classification(..., shuffle=False)`: only the "
                f"first {i} columns carry signal, the rest are noise. `names` holds `'f0'`, `'f1'`, ...")
        return setup, desc, d, i
    name = rng.choice(["breast_cancer", "wine"])
    what = ("569 breast-tumour samples (30 cell measurements) and `y` the diagnosis (0 malignant, 1 benign)"
            if name == "breast_cancer" else "178 wines (13 chemical measurements) and `y` the cultivar (0, 1, 2)")
    setup = (f"from sklearn.datasets import load_{name}\ndata = load_{name}()\n"
             f"order = np.random.default_rng({s}).permutation(len(data.target))\n"
             "X, y, names = data.data[order], data.target[order], list(data.feature_names)")
    return setup, f"`X` holds {what} (rows shuffled); `names` lists the feature names", \
        30 if name == "breast_cancer" else 13, 0


def _scaled(setup):
    return setup + "\nfrom sklearn.preprocessing import StandardScaler\nXs = StandardScaler().fit_transform(X)"


# ================================================================= lite

@template(M, "lite")
def kbest_basics(rng):
    setup, desc, d, inf = _fs_data(rng, nonneg=True)
    k, p, j = rng.randint(2, 6), rng.choice([10, 20, 25, 40]), rng.randrange(d)
    skb = f"SelectKBest(score_func=f_classif, k={k})"
    imp = f"{FS}SelectKBest, f_classif\n"
    items = [
        ("get_support", f"Fit `{skb}` on `X, y` and return the indices of the kept columns "
                        f"(`get_support(indices=True)`).", f"{imp}result = {skb}.fit(X, y).get_support(indices=True)"),
        ("scores_", f"Fit `{skb}` on `X, y` and return the ANOVA F-scores of the first 3 features (`scores_`).",
         f"{imp}result = {skb}.fit(X, y).scores_[:3]"),
        ("transform", f"Fit `{skb}` on `X, y` and return the shape of `transform(X)`.",
         f"{imp}result = {skb}.fit(X, y).transform(X).shape"),
        ("chi2", f"`chi2` scores need non-negative features (counts, measurements). Return the first 2 rows of "
                 f"`SelectKBest(chi2, k={k}).fit_transform(X, y)`.",
         f"{FS}SelectKBest, chi2\nresult = SelectKBest(chi2, k={k}).fit_transform(X, y)[:2]"),
        ("pvalues_", f"Fit `{skb}` on `X, y` and return how many features have a p-value below 0.001 (`pvalues_`).",
         f"{imp}result = int(({skb}.fit(X, y).pvalues_ < 0.001).sum())"),
        ("get_support", f"Fit `{skb}` on `X, y` and return the **names** of the kept features, as a list (use the boolean "
                        f"mask `get_support()` on `np.array(names)`).",
         f"{imp}result = list(np.array(names)[{skb}.fit(X, y).get_support()])"),
        ("f_classif", f"The score functions can be called directly: `F, p = f_classif(X, y)`. Return `F[{j}]`.",
         f"{FS}f_classif\nF, p = f_classif(X, y)\nresult = F[{j}]"),
        ("chi2", f"Call `chi2(X, y)` directly and return the index of the feature with the largest chi² score.",
         f"{FS}chi2\nscores, p = chi2(X, y)\nresult = int(np.argmax(scores))"),
        ("SelectPercentile", f"Keep the top {p}% of features by F-score: return the shape of "
                             f"`SelectPercentile(f_classif, percentile={p}).fit_transform(X, y)`.",
         f"{FS}SelectPercentile, f_classif\nresult = SelectPercentile(f_classif, percentile={p}).fit_transform(X, y).shape"),
        ("mutual_info_classif", f"Mutual information catches non-linear links too. Return the shape of "
                                f"`SelectPercentile(mutual_info_classif, percentile={p}).fit_transform(X, y)`.",
         f"{FS}SelectPercentile, mutual_info_classif\n"
         f"result = SelectPercentile(mutual_info_classif, percentile={p}).fit_transform(X, y).shape",
         {"require": ["mutual_info_classif", "SelectPercentile"]}),
        ("inverse_transform", f"Fit `{skb}` on `X, y`; `inverse_transform` puts the kept columns back in place and zeros "
                              f"elsewhere. Return `inverse_transform(transform(X[:1]))`.",
         f"{imp}skb = {skb}.fit(X, y)\nresult = skb.inverse_transform(skb.transform(X[:1]))"),
        ("get_feature_names_out", f"Fit `{skb}` on `pd.DataFrame(X, columns=names)` and `y`, and return "
                                  f"`get_feature_names_out()` as a list.",
         f"{imp}result = list({skb}.fit(pd.DataFrame(X, columns=names), y).get_feature_names_out())"),
    ]
    return catalog(rng, items, setup, f"{desc}.", tol=1e-6)


@template(M, "lite")
def variance_threshold_basics(rng):
    s = seeded(rng)
    kind = rng.choice(["digits", "flags"])
    if kind == "digits":
        setup = "from sklearn.datasets import load_digits\nX, y = load_digits(return_X_y=True)"
        desc = "`X` holds 1797 handwritten digits as 64 pixel intensities (0-16); border pixels are almost always 0"
        t = rng.choice([0.5, 1.0, 5.0, 10.0])
    else:
        n, d = rng.randint(80, 200), rng.randint(6, 10)
        ps = [round(rng.choice([0.01, 0.03, 0.1, 0.3, 0.5, 0.7, 0.95, 1.0]), 2) for _ in range(d)]
        setup = (f"g = np.random.default_rng({s})\nX = (g.random(({n}, {d})) < {lit(ps)}).astype(float)   "
                 f"# column j is 1 with probability p[j]")
        desc = f"`X` is a {n} × {d} matrix of 0/1 flags; column `j` is 1 with probability `{ps}[j]`"
        t = rng.choice([0.05, 0.1, 0.16])
    items = [
        ("VarianceThreshold", f"Drop the low-variance columns: return the shape of "
                              f"`VarianceThreshold(threshold={t}).fit_transform(X)`.",
         f"{FS}VarianceThreshold\nresult = VarianceThreshold(threshold={t}).fit_transform(X).shape"),
        ("variances_", "Fit `VarianceThreshold()` on `X` and return the variances it computed (`variances_`).",
         f"{FS}VarianceThreshold\nresult = VarianceThreshold().fit(X).variances_"),
        ("get_support", "The default threshold (0) removes only constant columns. Return the indices of the columns "
                        "`VarianceThreshold()` keeps.",
         f"{FS}VarianceThreshold\nresult = VarianceThreshold().fit(X).get_support(indices=True)"),
        ("get_support", "How many columns are constant? Return `X.shape[1]` minus the number `VarianceThreshold()` keeps.",
         f"{FS}VarianceThreshold\nresult = X.shape[1] - int(VarianceThreshold().fit(X).get_support().sum())"),
        ("VarianceThreshold", f"Return the indices of the columns **removed** by `VarianceThreshold(threshold={t})`.",
         f"{FS}VarianceThreshold\nresult = np.flatnonzero(~VarianceThreshold(threshold={t}).fit(X).get_support())"),
    ]
    return catalog(rng, items, setup, f"{desc}.", tol=1e-9)


@template(M, "lite")
def model_selectors_basics(rng):
    setup, desc, d, inf = _fs_data(rng)
    setup = _scaled(setup)
    k, st = rng.randint(2, 5), rng.choice([1, 2, 3])
    m = min(d, 8)
    lr = "from sklearn.linear_model import LogisticRegression\n"
    rf = "RandomForestClassifier(n_estimators=50, random_state=0)"
    items = [
        ("RFE", f"Recursive feature elimination: fit `RFE({LR}, n_features_to_select={k}, step={st})` on `Xs, y` and "
                f"return `ranking_` (1 = kept; larger = eliminated earlier).",
         f"{FS}RFE\n{lr}result = RFE({LR}, n_features_to_select={k}, step={st}).fit(Xs, y).ranking_"),
        ("RFE", f"Fit `RFE({LR}, n_features_to_select={k})` on `Xs, y` and return the names of the kept features.",
         f"{FS}RFE\n{lr}rfe = RFE({LR}, n_features_to_select={k}).fit(Xs, y)\nresult = list(np.array(names)[rfe.support_])",
         {"require": ["RFE", "support_|get_support"]}),
        ("RFECV", f"Let cross-validation choose how many to keep: fit `RFECV({LR}, cv=3, scoring='accuracy', "
                  f"min_features_to_select=2)` on the first {m} columns of `Xs` and `y`, and return `n_features_`.",
         f"{FS}RFECV\n{lr}result = int(RFECV({LR}, cv=3, scoring='accuracy', min_features_to_select=2)"
         f".fit(Xs[:, :{m}], y).n_features_)"),
        ("SelectFromModel", f"Keep the features whose random-forest importance is at least the median: fit "
                            f"`SelectFromModel({rf}, threshold='median')` on `X, y` and return how many it keeps.",
         f"{FS}SelectFromModel\nfrom sklearn.ensemble import RandomForestClassifier\n"
         f"result = int(SelectFromModel({rf}, threshold='median').fit(X, y).get_support().sum())",
         {"require": ["SelectFromModel", "RandomForestClassifier"]}),
        ("threshold_", f"Fit `SelectFromModel({rf}, threshold='mean')` on `X, y` and return the importance cut-off it "
                       f"used (`threshold_`).",
         f"{FS}SelectFromModel\nfrom sklearn.ensemble import RandomForestClassifier\n"
         f"result = SelectFromModel({rf}, threshold='mean').fit(X, y).threshold_"),
        ("SequentialFeatureSelector", f"Greedy forward selection: fit `SequentialFeatureSelector(KNeighborsClassifier(3), "
                                      f"n_features_to_select={min(k, 3)}, direction='forward', cv=3)` on the first {m} "
                                      f"columns of `Xs` and `y`, and return `get_support()`.",
         f"{FS}SequentialFeatureSelector\nfrom sklearn.neighbors import KNeighborsClassifier\n"
         f"sfs = SequentialFeatureSelector(KNeighborsClassifier(3), n_features_to_select={min(k, 3)}, direction='forward', "
         f"cv=3)\nresult = sfs.fit(Xs[:, :{m}], y).get_support()",
         {"require": ["SequentialFeatureSelector", "KNeighborsClassifier"]}),
        ("SelectFromModel", "L1 regularization zeroes coefficients, so it selects. Fit `SelectFromModel(LinearSVC("
                            "C=0.05, penalty='l1', dual=False))` (`LinearSVC` from `sklearn.svm`) on `Xs, y` and return "
                            "the indices it keeps.",
         f"{FS}SelectFromModel\nfrom sklearn.svm import LinearSVC\n"
         "sfm = SelectFromModel(LinearSVC(C=0.05, penalty='l1', dual=False))\n"
         "result = sfm.fit(Xs, y).get_support(indices=True)", {"require": ["SelectFromModel", "LinearSVC"]}),
        ("estimator_", f"Fit `RFE({LR}, n_features_to_select={k})` on `Xs, y` and return the coefficients of the model it "
                       f"refitted on the kept features (`estimator_.coef_`).",
         f"{FS}RFE\n{lr}result = RFE({LR}, n_features_to_select={k}).fit(Xs, y).estimator_.coef_", {"tol": 1e-4}),
    ]
    return catalog(rng, items, setup, f"{desc}. `Xs` is `X` standardized.", tol=1e-6)


# ================================================================= easy

@template(M, "easy")
def f_classif_by_hand(rng):
    setup, desc, d, inf = _fs_data(rng)
    j = rng.randrange(d)
    task = rng.choice(["one", "pvalue", "between"])
    if task == "one":
        return P(f"{desc}. `f_classif` is a one-way ANOVA per feature. For feature {j}, compute **by hand** F = "
                 f"(between-class sum of squares / (k − 1)) / (within-class sum of squares / (n − k)), where k is the "
                 f"number of classes. Return `(by_hand, f_classif(X, y)[0][{j}])`.", setup,
                 f"{FS}f_classif\nx = X[:, {j}]\ncls = np.unique(y)\nk, n = len(cls), len(x)\n"
                 "ssb = sum((y == c).sum() * (x[y == c].mean() - x.mean()) ** 2 for c in cls)\n"
                 "ssw = sum(((x[y == c] - x[y == c].mean()) ** 2).sum() for c in cls)\n"
                 f"result = ((ssb / (k - 1)) / (ssw / (n - k)), f_classif(X, y)[0][{j}])", require=["f_classif"],
                 tol=1e-6)
    if task == "pvalue":
        return P(f"{desc}. The p-value of an F-score comes from the F distribution with (k − 1, n − k) degrees of "
                 f"freedom (k classes, n rows). Fit `SelectKBest(f_classif, k='all')` on `X, y` and recompute "
                 f"`pvalues_[{j}]` from `scores_[{j}]` with `scipy.stats.f.sf`. Return `(yours, pvalues_[{j}])`.", setup,
                 f"{FS}SelectKBest, f_classif\nfrom scipy.stats import f\nskb = SelectKBest(f_classif, k='all').fit(X, y)\n"
                 f"k, n = len(np.unique(y)), len(y)\nresult = (f.sf(skb.scores_[{j}], k - 1, n - k), skb.pvalues_[{j}])",
                 require=["pvalues_"], tol=1e-6)
    return P(f"{desc}. A high F-score means the class means of a feature are far apart relative to the spread inside "
             f"each class. Return the per-class means of the feature with the **highest** `f_classif` score, as an "
             f"array in class order.", setup,
             f"{FS}f_classif\nj = int(np.argmax(f_classif(X, y)[0]))\n"
             "result = np.array([X[y == c, j].mean() for c in np.unique(y)])", require=["f_classif"], tol=1e-6)


@template(M, "easy")
def chi2_by_hand(rng):
    setup, desc, d, inf = _fs_data(rng, nonneg=True)
    j = rng.randrange(d)
    task = rng.choice(["one", "all"])
    head = (f"{desc}. `chi2` treats each feature like a count: the *observed* total of feature j in class c is "
            f"`X[y == c, j].sum()`; the *expected* total is (fraction of rows in class c) × `X[:, j].sum()`; the score "
            f"is Σ over classes of (observed − expected)² / expected.")
    if task == "one":
        return P(f"{head} Compute it by hand for feature {j} and return `(by_hand, chi2(X, y)[0][{j}])`.", setup,
                 f"{FS}chi2\ncls = np.unique(y)\nobs = np.array([X[y == c, {j}].sum() for c in cls])\n"
                 f"exp = np.array([(y == c).mean() for c in cls]) * X[:, {j}].sum()\n"
                 f"result = (((obs - exp) ** 2 / exp).sum(), chi2(X, y)[0][{j}])", require=["chi2"], tol=1e-6)
    return P(f"{head} Compute the scores of **all** features at once with matrix operations (a one-hot matrix of `y` "
             f"times `X` gives the observed totals) and return them.", setup,
             "Y = (y[:, None] == np.unique(y)[None, :]).astype(float)\nobs = Y.T @ X\n"
             "exp = Y.mean(axis=0)[:, None] * X.sum(axis=0)[None, :]\nresult = ((obs - exp) ** 2 / exp).sum(axis=0)",
             tol=1e-6)


@template(M, "easy")
def variance_by_hand(rng):
    t = table(rng, numeric_only=True, n=(60, 150))
    task = rng.choice(["mask", "scaled"])
    thr = rng.choice([1.0, 10.0, 100.0])
    if task == "mask":
        return P(f"{t.desc}. `VarianceThreshold(threshold={thr})` keeps the columns whose (population) variance is "
                 f"strictly above the threshold. Return `(by_hand, actual)`: the list of kept column names computed with "
                 f"`df.var(ddof=0)`, and `get_feature_names_out()` of the fitted selector as a list.", t.setup,
                 f"{FS}VarianceThreshold\nv = df.var(ddof=0)\nby_hand = list(v.index[v > {thr}])\n"
                 f"result = (by_hand, list(VarianceThreshold(threshold={thr}).fit(df).get_feature_names_out()))",
                 require=["VarianceThreshold"])
    return P(f"{t.desc}. Variance depends on units, so `VarianceThreshold` on raw columns mostly measures scale. "
             f"Return `(raw, scaled)`: how many columns `VarianceThreshold(threshold=0.05)` keeps on `df`, and on "
             f"`MinMaxScaler().fit_transform(df)` (every column squeezed into [0, 1]).", t.setup,
             f"{FS}VarianceThreshold\nfrom sklearn.preprocessing import MinMaxScaler\n"
             "raw = VarianceThreshold(threshold=0.05).fit_transform(df).shape[1]\n"
             "scaled = VarianceThreshold(threshold=0.05).fit_transform(MinMaxScaler().fit_transform(df)).shape[1]\n"
             "result = (raw, scaled)", require=["VarianceThreshold", "MinMaxScaler"])


@template(M, "easy")
def selected_column_names(rng):
    t = table(rng, kind="clf", numeric_only=True, n=(100, 250))
    k = rng.randint(1, len(t.num) - 1)
    task = rng.choice(["names", "mask", "scores"])
    if task == "names":
        return P(f"{t.desc}. Fit `SelectKBest(f_classif, k={k})` on the DataFrame `df` and `y` and return the names of "
                 f"the kept columns (`get_feature_names_out()`) as a list.", t.setup,
                 f"{FS}SelectKBest, f_classif\nresult = list(SelectKBest(f_classif, k={k}).fit(df, y).get_feature_names_out())",
                 require=["SelectKBest", "get_feature_names_out"])
    if task == "mask":
        return P(f"{t.desc}. Fit `SelectKBest(f_classif, k={k})` on `df, y` and use it to return the **dropped** column "
                 f"names (`df.columns` with the inverted `get_support()` mask) as a list.", t.setup,
                 f"{FS}SelectKBest, f_classif\nmask = SelectKBest(f_classif, k={k}).fit(df, y).get_support()\n"
                 "result = list(df.columns[~mask])", require=["get_support"])
    return P(f"{t.desc}. Fit `SelectKBest(f_classif, k='all')` on `df, y` and return a dict {{column: F-score}}.",
             t.setup, f"{FS}SelectKBest, f_classif\nskb = SelectKBest(f_classif, k='all').fit(df, y)\n"
                      "result = dict(zip(df.columns, skb.scores_))", require=["scores_"], tol=1e-6)


@template(M, "easy")
def percentile_vs_kbest(rng):
    setup, desc, d, inf = _fs_data(rng)
    p = rng.choice([10, 20, 30, 50])
    task = rng.choice(["count", "same"])
    if task == "count":
        return P(f"{desc}. How many features does `SelectPercentile(f_classif, percentile={p})` keep out of {d}? "
                 f"Return `(kept, int({d} * {p} / 100))` (they agree when the scores have no ties).", setup,
                 f"{FS}SelectPercentile, f_classif\nkept = int(SelectPercentile(f_classif, percentile={p}).fit(X, y)"
                 f".get_support().sum())\nresult = (kept, int({d} * {p} / 100))", require=["SelectPercentile"])
    return P(f"{desc}. Fit `SelectPercentile(f_classif, percentile={p})` on `X, y`, then `SelectKBest(f_classif, k=...)` "
             f"with `k` equal to the number of features the first one kept. Return `(k, same)` where `same` says "
             f"whether both kept exactly the same columns.", setup,
             f"{FS}SelectPercentile, SelectKBest, f_classif\nsp = SelectPercentile(f_classif, percentile={p}).fit(X, y)\n"
             "k = int(sp.get_support().sum())\nsk = SelectKBest(f_classif, k=k).fit(X, y)\n"
             "result = (k, bool((sp.get_support() == sk.get_support()).all()))",
             require=["SelectPercentile", "SelectKBest"])


@template(M, "easy")
def rfe_elimination_order(rng):
    setup, desc, d, inf = _fs_data(rng, synth=0.6)
    setup = _scaled(setup)
    st = rng.choice([2, 3, 4])
    task = rng.choice(["order", "steps", "last"])
    lr = f"{FS}RFE\nfrom sklearn.linear_model import LogisticRegression\n"
    if task == "order":
        return P(f"{desc}; `Xs` is `X` standardized. With `n_features_to_select=1` and `step=1`, `RFE` ranks every "
                 f"feature. Fit `RFE({LR}, n_features_to_select=1)` on `Xs, y` and return the feature **names** from "
                 f"most to least important (sort by `ranking_`).", setup,
                 f"{lr}rfe = RFE({LR}, n_features_to_select=1).fit(Xs, y)\n"
                 "result = [names[j] for j in np.argsort(rfe.ranking_)]", require=["RFE", "ranking_"])
    if task == "steps":
        return P(f"{desc}; `Xs` is `X` standardized. Fit `RFE({LR}, n_features_to_select=2, step={st})` on `Xs, y`. "
                 f"Return `(rounds, max_rank)`: the number of elimination rounds, `ceil(({d} - 2) / {st})`, and the "
                 f"largest value in `ranking_` (it is rounds + 1).", setup,
                 f"import math\n{lr}rfe = RFE({LR}, n_features_to_select=2, step={st}).fit(Xs, y)\n"
                 f"result = (math.ceil(({d} - 2) / {st}), int(rfe.ranking_.max()))", require=["ranking_"])
    return P(f"{desc}; `Xs` is `X` standardized. Fit `RFE({LR}, n_features_to_select=2, step={st})` on `Xs, y` and "
             f"return the names of the features eliminated in the **first** round (the largest `ranking_`).", setup,
             f"{lr}rfe = RFE({LR}, n_features_to_select=2, step={st}).fit(Xs, y)\n"
             "result = [names[j] for j in np.flatnonzero(rfe.ranking_ == rfe.ranking_.max())]", require=["ranking_"])


@template(M, "easy")
def select_from_model_thresholds(rng):
    setup, desc, d, inf = _fs_data(rng)
    rf = "RandomForestClassifier(n_estimators=40, random_state=0)"
    thr = rng.choice(["'mean'", "'median'", "'1.5*mean'", "'0.5*mean'"])
    task = rng.choice(["count", "by_hand", "max_features"])
    imp = f"{FS}SelectFromModel\nfrom sklearn.ensemble import RandomForestClassifier\n"
    if task == "count":
        return P(f"{desc}. Fit `SelectFromModel({rf}, threshold={thr})` on `X, y` and return `(threshold_, kept)`: the "
                 f"cut-off it used and how many features it keeps.", setup,
                 f"{imp}sfm = SelectFromModel({rf}, threshold={thr}).fit(X, y)\n"
                 "result = (sfm.threshold_, int(sfm.get_support().sum()))", require=["SelectFromModel"], tol=1e-9)
    if task == "by_hand":
        return P(f"{desc}. `SelectFromModel` keeps the features whose importance is **at least** the threshold. Fit "
                 f"`SelectFromModel({rf}, threshold={thr})` on `X, y`, rebuild its mask from "
                 f"`estimator_.feature_importances_` and `threshold_`, and return `(by_hand == get_support()).all()`.",
                 setup, f"{imp}sfm = SelectFromModel({rf}, threshold={thr}).fit(X, y)\n"
                        "mask = sfm.estimator_.feature_importances_ >= sfm.threshold_\n"
                        "result = bool((mask == sfm.get_support()).all())", require=["feature_importances_"])
    k = rng.randint(2, 4)
    return P(f"{desc}. `max_features` caps the count: fit `SelectFromModel({rf}, threshold=-np.inf, max_features={k})` "
             f"(no threshold, just the {k} most important) on `X, y` and return the kept names.", setup,
             f"{imp}sfm = SelectFromModel({rf}, threshold=-np.inf, max_features={k}).fit(X, y)\n"
             "result = list(np.array(names)[sfm.get_support()])", require=["max_features"])


@template(M, "easy")
def selection_then_model(rng):
    setup, desc, d, inf = _fs_data(rng)
    k = rng.randint(2, 5)
    m = rng.randint(80, 120)
    task = rng.choice(["pipe", "manual"])
    if task == "pipe":
        return P(f"{desc}. Put the selection in a pipeline: fit `make_pipeline(StandardScaler(), SelectKBest(f_classif, "
                 f"k={k}), {LR})` on the first {m} rows and return its accuracy on the rest.", setup,
                 f"{FS}SelectKBest, f_classif\nfrom sklearn.pipeline import make_pipeline\n"
                 "from sklearn.preprocessing import StandardScaler\nfrom sklearn.linear_model import LogisticRegression\n"
                 f"pipe = make_pipeline(StandardScaler(), SelectKBest(f_classif, k={k}), {LR})\n"
                 f"result = pipe.fit(X[:{m}], y[:{m}]).score(X[{m}:], y[{m}:])", require=["SelectKBest"])
    return P(f"{desc}. Without a pipeline: fit `SelectKBest(f_classif, k={k})` on the first {m} rows only, transform "
             f"both parts with it, fit `{LR}` on the selected training columns and return the accuracy on the rest.",
             setup, f"{FS}SelectKBest, f_classif\nfrom sklearn.linear_model import LogisticRegression\n"
                    f"skb = SelectKBest(f_classif, k={k}).fit(X[:{m}], y[:{m}])\n"
                    f"clf = {LR}.fit(skb.transform(X[:{m}]), y[:{m}])\nresult = clf.score(skb.transform(X[{m}:]), y[{m}:])",
             require=["SelectKBest", "transform"])


@template(M, "easy")
def chi2_needs_nonnegative(rng):
    setup, desc, d, inf = _fs_data(rng)
    k = rng.randint(2, 5)
    return P(f"{desc}. `SelectKBest(chi2, k={k}).fit(StandardScaler().fit_transform(X), y)` raises `ValueError: Input "
             f"X must be non-negative`. Scale with `MinMaxScaler()` instead (every column into [0, 1]) and return the "
             f"indices `SelectKBest(chi2, k={k})` keeps.", setup,
             f"{FS}SelectKBest, chi2\nfrom sklearn.preprocessing import MinMaxScaler\n"
             f"result = SelectKBest(chi2, k={k}).fit(MinMaxScaler().fit_transform(X), y).get_support(indices=True)",
             require=["chi2", "MinMaxScaler"])


@template(M, "easy")
def selector_api_tour(rng):
    setup, desc, d, inf = _fs_data(rng)
    k, k2 = rng.randint(2, 4), rng.randint(5, 7)
    setup += f"\n{FS}SelectKBest, f_classif\nskb = SelectKBest(f_classif, k={k}).fit(X, y)"
    n = rng.randint(2, 4)
    items = [
        ("get_support", "Return `skb`'s boolean mask of kept columns.", "result = skb.get_support()"),
        ("get_support", "Return the indices of the kept columns, best score first (sort them by `scores_`).",
         "idx = skb.get_support(indices=True)\nresult = idx[np.argsort(-skb.scores_[idx])]"),
        ("transform", f"Return `skb.transform` of the first {n} rows.", f"result = skb.transform(X[:{n}])"),
        ("scores_", "Return the indices of all features sorted by F-score, best first.",
         "result = np.argsort(-skb.scores_)"),
        ("pvalues_", "Return the largest p-value among the kept features.",
         "result = float(skb.pvalues_[skb.get_support()].max())"),
        ("get_feature_names_out", "`skb` was fitted on a plain array, so its output names are generic. Return "
                                  "`get_feature_names_out()` as a list.", "result = list(skb.get_feature_names_out())"),
        ("set_params", f"Keep more features: `skb.set_params(k={k2})`, refit on `X, y` and return the kept indices.",
         f"result = skb.set_params(k={k2}).fit(X, y).get_support(indices=True)"),
        ("n_features_in_", "Return `(n_features_in_, number kept)` for `skb`.",
         "result = (skb.n_features_in_, int(skb.get_support().sum()))"),
        ("scores_", "What fraction of the total F-score do the kept features hold? Return it as a float.",
         "result = float(skb.scores_[skb.get_support()].sum() / skb.scores_.sum())"),        ("inverse_transform", "Return the shape of `skb.inverse_transform(skb.transform(X))` and how many of its columns "
                              "are entirely zero, as a tuple.",
         "Z = skb.inverse_transform(skb.transform(X))\nresult = (Z.shape, int((~Z.any(axis=0)).sum()))"),
        ("get_support", "Return the **names** of the features `skb` dropped, as a list.",
         "result = [nm for nm, keep in zip(names, skb.get_support()) if not keep]"),
    ]
    return catalog(rng, items, setup, f"{desc}. `skb = SelectKBest(f_classif, k={k})` is fitted on `X, y`.", tol=1e-6)


# ================================================================= medium

@template(M, "medium")
def table_selection_tasks(rng):
    t = table(rng, kind="clf", numeric_only=True, n=(150, 300))
    s = seeded(rng)
    k = rng.randint(1, len(t.num) - 1)
    setup = (t.setup + f"\nnoise = np.random.default_rng({s}).normal(size=(len(df), 3))\n"
             "for j in range(3):\n    df[f'noise{j}'] = noise[:, j]   # useless columns")
    lr = "from sklearn.linear_model import LogisticRegression\nfrom sklearn.preprocessing import StandardScaler\n"
    items = [
        ("SelectKBest", f"Return the {k + 1} column names `SelectKBest(f_classif, k={k + 1})` keeps (as a list).",
         f"{FS}SelectKBest, f_classif\nresult = list(SelectKBest(f_classif, k={k + 1}).fit(df, y).get_feature_names_out())"),
        ("f_classif", "Return the columns sorted by F-score, best first.",
         f"{FS}f_classif\nF, p = f_classif(df, y)\nresult = list(df.columns[np.argsort(-F)])"),
        ("mutual_info_classif", f"Return a dict {{column: mutual information}} from `mutual_info_classif(df, y, "
                                f"random_state={s})` (the estimator adds a little noise, hence the seed).",
         f"{FS}mutual_info_classif\nresult = dict(zip(df.columns, mutual_info_classif(df, y, random_state={s})))"),
        ("chi2", f"`chi2` needs non-negative inputs: min-max scale `df` first (`MinMaxScaler`), then return the {k + 1} "
                 f"column names `SelectKBest(chi2, k={k + 1})` keeps.",
         f"{FS}SelectKBest, chi2\nfrom sklearn.preprocessing import MinMaxScaler\n"
         f"skb = SelectKBest(chi2, k={k + 1}).fit(MinMaxScaler().fit_transform(df), y)\n"
         "result = list(df.columns[skb.get_support()])"),
        ("RFE", f"Standardize `df` (`StandardScaler`), then fit `RFE({LR}, n_features_to_select=1)` and return a dict "
                f"{{column: rank}}.",
         f"{FS}RFE\n{lr}rfe = RFE({LR}, n_features_to_select=1).fit(StandardScaler().fit_transform(df), y)\n"
         "result = dict(zip(df.columns, rfe.ranking_.tolist()))"),
        ("SelectFromModel", "Fit `SelectFromModel(RandomForestClassifier(n_estimators=50, random_state=0), "
                            "threshold='mean')` on `df, y` and return the kept column names.",
         f"{FS}SelectFromModel\nfrom sklearn.ensemble import RandomForestClassifier\n"
         "sfm = SelectFromModel(RandomForestClassifier(n_estimators=50, random_state=0), threshold='mean').fit(df, y)\n"
         "result = list(sfm.get_feature_names_out())"),
        ("SequentialFeatureSelector", f"Standardize `df`, then run forward selection with "
                                      f"`SequentialFeatureSelector({LR}, n_features_to_select=2, direction='forward', "
                                      f"cv=3)` and return the 2 chosen column names.",
         f"{FS}SequentialFeatureSelector\n{lr}"
         f"sfs = SequentialFeatureSelector({LR}, n_features_to_select=2, direction='forward', cv=3)\n"
         "sfs.fit(StandardScaler().fit_transform(df), y)\nresult = list(df.columns[sfs.get_support()])"),
        ("SequentialFeatureSelector", "Standardize `df`, then run **backward** elimination with "
                                      f"`SequentialFeatureSelector({LR}, n_features_to_select=2, direction='backward', "
                                      f"cv=3)` and return the 2 chosen column names.",
         f"{FS}SequentialFeatureSelector\n{lr}"
         f"sfs = SequentialFeatureSelector({LR}, n_features_to_select=2, direction='backward', cv=3)\n"
         "sfs.fit(StandardScaler().fit_transform(df), y)\nresult = list(df.columns[sfs.get_support()])"),
        ("SelectKBest", "How many of the 3 noise columns survive `SelectKBest(f_classif, k=3)`? Return the count.",
         f"{FS}SelectKBest, f_classif\nkept = SelectKBest(f_classif, k=3).fit(df, y).get_feature_names_out()\n"
         "result = sum(c.startswith('noise') for c in kept)"),
        ("pvalues_", "Return the names of the columns whose `f_classif` p-value is below 0.01.",
         f"{FS}SelectKBest, f_classif\nskb = SelectKBest(f_classif, k='all').fit(df, y)\n"
         "result = list(df.columns[skb.pvalues_ < 0.01])"),
        ("VarianceThreshold", "Standardize `df` and apply `VarianceThreshold(threshold=0.5)`: return how many columns "
                              "it keeps (after standardizing, every variance is 1).",
         f"{FS}VarianceThreshold\nfrom sklearn.preprocessing import StandardScaler\n"
         "result = VarianceThreshold(threshold=0.5).fit_transform(StandardScaler().fit_transform(df)).shape[1]"),
        ("score", f"Does selection help? Return the 3-fold cross-validated accuracy (`cross_val_score`, mean) of "
                  f"`make_pipeline(StandardScaler(), {LR})` on all columns and of the same with "
                  f"`SelectKBest(f_classif, k={k})` after the scaler, as a tuple.",
         f"{FS}SelectKBest, f_classif\n{lr}from sklearn.pipeline import make_pipeline\n"
         "from sklearn.model_selection import cross_val_score\n"
         f"a = cross_val_score(make_pipeline(StandardScaler(), {LR}), df, y, cv=3).mean()\n"
         f"b = cross_val_score(make_pipeline(StandardScaler(), SelectKBest(f_classif, k={k}), {LR}), df, y, cv=3).mean()\n"
         "result = (a, b)", {"require": ["cross_val_score", "SelectKBest"]}),
    ]
    return catalog(rng, setup=setup, items=items,
                   intro=f"{t.desc}. Three pure-noise columns `noise0`-`noise2` were appended to `df`.", tol=1e-6)


def _signal_noise(rng):
    n, d, i = rng.randint(150, 300), rng.randint(10, 20), rng.randint(2, 4)
    s = seeded(rng)
    setup = (f"from sklearn.datasets import make_classification\nX, y = make_classification(n_samples={n}, "
             f"n_features={d}, n_informative={i}, n_redundant=0, shuffle=False, random_state={s})\n"
             f"from sklearn.preprocessing import StandardScaler\nXs = StandardScaler().fit_transform(X)")
    desc = (f"`X` ({n} × {d}) and `y` come from `make_classification(..., shuffle=False)`: columns 0-{i - 1} are "
            f"informative, the other {d - i} pure noise; `Xs` is `X` standardized")
    return setup, desc, d, i


@template(M, "medium")
def find_the_signal(rng):
    setup, desc, d, i = _signal_noise(rng)
    rf = "RandomForestClassifier(n_estimators=50, random_state=0)"
    items = [
        ("SelectKBest", f"Fit `SelectKBest(f_classif, k={i})` on `X, y`. How many of the {i} kept columns are truly "
                        f"informative (index < {i})? Return the count.",
         f"{FS}SelectKBest, f_classif\nidx = SelectKBest(f_classif, k={i}).fit(X, y).get_support(indices=True)\n"
         f"result = int((idx < {i}).sum())"),
        ("mutual_info_classif", f"Same question for mutual information: keep {i} columns with `SelectKBest(partial("
                                f"mutual_info_classif, random_state=0), k={i})` (`functools.partial` fixes the seed) and "
                                f"return how many are informative.",
         f"from functools import partial\n{FS}SelectKBest, mutual_info_classif\n"
         f"skb = SelectKBest(partial(mutual_info_classif, random_state=0), k={i}).fit(X, y)\n"
         f"result = int((skb.get_support(indices=True) < {i}).sum())"),
        ("RFE", f"Fit `RFE({LR}, n_features_to_select={i})` on `Xs, y` and return the kept indices.",
         f"{FS}RFE\nfrom sklearn.linear_model import LogisticRegression\n"
         f"result = RFE({LR}, n_features_to_select={i}).fit(Xs, y).get_support(indices=True)"),
        ("SelectFromModel", f"Fit `SelectFromModel({rf})` (default threshold: the mean importance) on `X, y` and return "
                            f"`(kept_informative, kept_noise)` counts.",
         f"{FS}SelectFromModel\nfrom sklearn.ensemble import RandomForestClassifier\n"
         f"idx = SelectFromModel({rf}).fit(X, y).get_support(indices=True)\n"
         f"result = (int((idx < {i}).sum()), int((idx >= {i}).sum()))"),
        ("SelectFromModel", "Fit `SelectFromModel(LogisticRegression(C=0.05, l1_ratio=1.0, solver='liblinear', random_state=0))` on "
                            "`Xs, y` (L1 zeroes weak coefficients) and return the kept indices.",
         f"{FS}SelectFromModel\nfrom sklearn.linear_model import LogisticRegression\n"
         "sfm = SelectFromModel(LogisticRegression(C=0.05, l1_ratio=1.0, solver='liblinear', random_state=0)).fit(Xs, y)\n"
         "result = sfm.get_support(indices=True)"),
        ("f_classif", f"Return the mean F-score of the informative columns and of the noise columns, as a tuple.",
         f"{FS}f_classif\nF, p = f_classif(X, y)\nresult = (F[:{i}].mean(), F[{i}:].mean())"),
        ("RFECV", f"Fit `RFECV({LR}, cv=3, min_features_to_select=1)` on `Xs, y` and return `(n_features_, kept "
                  f"indices)`.",
         f"{FS}RFECV\nfrom sklearn.linear_model import LogisticRegression\n"
         f"r = RFECV({LR}, cv=3, min_features_to_select=1).fit(Xs, y)\n"
         "result = (int(r.n_features_), r.get_support(indices=True))"),
    ]
    return catalog(rng, items, setup, f"{desc}.", tol=1e-6)


@template(M, "medium")
def rfecv_curve(rng):
    setup, desc, d, i = _signal_noise(rng)
    mn = rng.randint(1, 3)
    task = rng.choice(["curve", "best", "std", "shape"])
    sol = (f"{FS}RFECV\nfrom sklearn.linear_model import LogisticRegression\n"
           f"r = RFECV({LR}, cv=3, scoring='accuracy', min_features_to_select={mn}).fit(Xs, y)\n")
    head = (f"{desc}. Fit `RFECV({LR}, cv=3, scoring='accuracy', min_features_to_select={mn})` on `Xs, y`. Its "
            f"`cv_results_` dict holds the cross-validated scores for each number of features tried "
            f"(`cv_results_['n_features']`).")
    if task == "curve":
        return P(f"{head} Return the score-vs-number-of-features curve: `cv_results_['mean_test_score']`.", setup,
                 sol + "result = r.cv_results_['mean_test_score']", require=["RFECV", "cv_results_"], tol=1e-6)
    if task == "best":
        return P(f"{head} Check the choice: return `(n_features_, best_by_hand)` where `best_by_hand` is the number of "
                 f"features with the highest mean score (the first on ties).", setup,
                 sol + "res = r.cv_results_\n"
                       "result = (int(r.n_features_), int(res['n_features'][np.argmax(res['mean_test_score'])]))",
                 require=["n_features_"])
    if task == "std":
        return P(f"{head} Return a dict {{n_features: std_test_score}}.", setup,
                 sol + "res = r.cv_results_\nresult = dict(zip(res['n_features'].tolist(), res['std_test_score']))",
                 require=["cv_results_"], tol=1e-6)
    return P(f"{head} Return `r.transform(Xs).shape` and `n_features_` as a tuple.", setup,
             sol + "result = (r.transform(Xs).shape, int(r.n_features_))", require=["transform", "n_features_"])


@template(M, "medium")
def tune_k_in_pipeline(rng):
    setup, desc, d, i = _signal_noise(rng)
    ks = sorted(rng.sample(range(1, min(d, 8) + 1), 4))
    task = rng.choice(["grid", "scores", "manual"])
    pipe = f"Pipeline([('select', SelectKBest(f_classif)), ('clf', {LR})])"
    imp = (f"{FS}SelectKBest, f_classif\nfrom sklearn.pipeline import Pipeline\n"
           "from sklearn.linear_model import LogisticRegression\n")
    if task == "grid":
        return P(f"{desc}. Tune `k` without leakage: `GridSearchCV({pipe}, {{'select__k': {ks}}}, cv=3)` on `Xs, y`. "
                 f"Return `(best_params_['select__k'], best_score_)`.", setup,
                 f"{imp}from sklearn.model_selection import GridSearchCV\n"
                 f"gs = GridSearchCV({pipe}, {{'select__k': {ks}}}, cv=3).fit(Xs, y)\n"
                 "result = (gs.best_params_['select__k'], gs.best_score_)", require=["GridSearchCV", "SelectKBest"],
                 tol=1e-6)
    if task == "scores":
        return P(f"{desc}. For each `k` in `{ks}`, return the mean 3-fold `cross_val_score` of `{pipe}` with "
                 f"`select__k` set to `k` (`set_params`), as a dict {{k: score}}.", setup,
                 f"{imp}from sklearn.model_selection import cross_val_score\npipe = {pipe}\n"
                 f"result = {{k: cross_val_score(pipe.set_params(select__k=k), Xs, y, cv=3).mean() for k in {ks}}}",
                 require=["set_params", "cross_val_score"], tol=1e-6)
    return P(f"{desc}. Pick `k` from `{ks}` on a validation split: fit `{pipe}` (with `select__k=k`) on the first 60% "
             f"of rows (`int(0.6 * len(Xs))`) and score on the rest. Return the best `k` (the smallest on ties).", setup,
             f"{imp}h = int(0.6 * len(Xs))\npipe = {pipe}\n"
             f"scores = [pipe.set_params(select__k=k).fit(Xs[:h], y[:h]).score(Xs[h:], y[h:]) for k in {ks}]\n"
             f"result = {ks}[int(np.argmax(scores))]", require=["set_params"])


@template(M, "medium")
def l1_path_selection(rng):
    setup, desc, d, i = _signal_noise(rng)
    cs = sorted(rng.sample([0.01, 0.03, 0.1, 0.3, 1.0], 3))
    task = rng.choice(["counts", "first", "lasso", "names", "threshold"])
    if task == "counts":
        return P(f"{desc}. For each `C` in `{cs}`, fit `SelectFromModel(LogisticRegression(C=C, l1_ratio=1.0, "
                 f"solver='liblinear', random_state=0))` on `Xs, y` and record how many features it keeps. Return the list (weaker "
                 f"regularization keeps more).", setup,
                 f"{FS}SelectFromModel\nfrom sklearn.linear_model import LogisticRegression\n"
                 "result = [int(SelectFromModel(LogisticRegression(C=C, l1_ratio=1.0, solver='liblinear', random_state=0)).fit(Xs, y)"
                 f".get_support().sum()) for C in {cs}]", require=["SelectFromModel", "l1_ratio"])
    if task == "first":
        return P(f"{desc}. With very strong L1 regularization only the strongest features survive. Fit "
                 f"`LogisticRegression(C={cs[0]}, l1_ratio=1.0, solver='liblinear', random_state=0)` on `Xs, y` and return the indices "
                 f"of its non-zero coefficients.", setup,
                 f"from sklearn.linear_model import LogisticRegression\n"
                 f"clf = LogisticRegression(C={cs[0]}, l1_ratio=1.0, solver='liblinear', random_state=0).fit(Xs, y)\n"
                 "result = np.flatnonzero(clf.coef_[0])", require=["l1_ratio", "coef_"])
    if task == "lasso":
        a = rng.choice([0.02, 0.05, 0.1])
        return P(f"{desc}. Treat `y` as a 0/1 number and select with a regression model: fit "
                 f"`SelectFromModel(Lasso(alpha={a}))` on `Xs, y` and return the kept indices.", setup,
                 f"{FS}SelectFromModel\nfrom sklearn.linear_model import Lasso\n"
                 f"result = SelectFromModel(Lasso(alpha={a})).fit(Xs, y).get_support(indices=True)",
                 require=["SelectFromModel", "Lasso"])
    if task == "names":
        return P(f"{desc}. Put `Xs` in a DataFrame `D` with columns named `x0`, `x1`, ..., fit "
                 f"`SelectFromModel(LogisticRegression(C={cs[1]}, l1_ratio=1.0, solver='liblinear', random_state=0))` on `D, y`, and "
                 f"return the kept names (`get_feature_names_out`).",
                 setup,
                 f"{FS}SelectFromModel\nfrom sklearn.linear_model import LogisticRegression\n"
                 f"D = pd.DataFrame(Xs, columns=[f'x{{j}}' for j in range({d})])\n"
                 f"sfm = SelectFromModel(LogisticRegression(C={cs[1]}, l1_ratio=1.0, solver='liblinear', random_state=0)).fit(D, y)\n"
                 "result = list(sfm.get_feature_names_out())", require=["get_feature_names_out"])
    return P(f"{desc}. For an L1 model, `SelectFromModel`'s default threshold is tiny (1e-5): it keeps exactly the "
             f"non-zero coefficients. Fit it with `LogisticRegression(C={cs[1]}, l1_ratio=1.0, solver='liblinear', random_state=0)` on "
             f"`Xs, y` and return `(threshold_, kept, nonzero)`: the threshold, the number kept and the number of "
             f"non-zero coefficients of `estimator_`.", setup,
             f"{FS}SelectFromModel\nfrom sklearn.linear_model import LogisticRegression\n"
             f"sfm = SelectFromModel(LogisticRegression(C={cs[1]}, l1_ratio=1.0, solver='liblinear', random_state=0)).fit(Xs, y)\n"
             "result = (sfm.threshold_, int(sfm.get_support().sum()), int((sfm.estimator_.coef_ != 0).sum()))",
             require=["threshold_", "estimator_"], tol=1e-9)


@template(M, "medium")
def regression_filters(rng):
    t = table(rng, kind="reg", numeric_only=True, n=(100, 250))
    s = seeded(rng)
    k = rng.randint(1, len(t.num) - 1)
    imp = "from sklearn.feature_selection import "
    items = [
        ("f_regression", f"For a continuous target use `f_regression`. Return the {k} column names "
                         f"`SelectKBest(f_regression, k={k})` keeps.",
         f"{imp}SelectKBest, f_regression\nresult = list(SelectKBest(f_regression, k={k}).fit(df, y).get_feature_names_out())"),
        ("r_regression", "`r_regression` returns each column's Pearson correlation with `y` (with its sign). Return a "
                         "dict {column: r}.",
         f"{imp}r_regression\nresult = dict(zip(df.columns, r_regression(df, y)))"),
        ("f_regression", "`f_regression`'s F is r² / (1 − r²) × (n − 2), r being the correlation with `y`. Compute it by "
                         "hand for every column from `np.corrcoef` and return `(by_hand, f_regression(df, y)[0])`.",
         f"{imp}f_regression\nn = len(df)\nr = np.array([np.corrcoef(df[c], y)[0, 1] for c in df.columns])\n"
         "result = (r ** 2 / (1 - r ** 2) * (n - 2), f_regression(df, y)[0])"),
        ("mutual_info_regression", f"Return the column with the highest `mutual_info_regression(df, y, "
                                   f"random_state={s})`.",
         f"{imp}mutual_info_regression\nresult = df.columns[np.argmax(mutual_info_regression(df, y, random_state={s}))]"),
        ("SelectPercentile", "Keep the top 50% of columns by `f_regression` with `SelectPercentile` and return the "
                             "training R² of `LinearRegression` on just those columns.",
         f"{imp}SelectPercentile, f_regression\nfrom sklearn.linear_model import LinearRegression\n"
         "Z = SelectPercentile(f_regression, percentile=50).fit_transform(df, y)\n"
         "result = LinearRegression().fit(Z, y).score(Z, y)"),
        ("f_regression", f"`f_regression` ranks columns by their squared correlation with `y`. Return `(kept, by_corr)`: "
                         f"the sorted names `SelectKBest(f_regression, k={k})` keeps, and the sorted names of the {k} "
                         f"columns with the largest |correlation| (`df.corrwith(pd.Series(y))`).",
         f"{imp}SelectKBest, f_regression\nkept = sorted(SelectKBest(f_regression, k={k}).fit(df, y).get_feature_names_out())\n"
         f"r = df.corrwith(pd.Series(y)).abs()\nresult = (kept, sorted(r.sort_values(ascending=False).index[:{k}]))"),
        ("pvalues_", "Return the names of the columns whose `f_regression` p-value is below 0.05 (fit "
                     "`SelectKBest(f_regression, k='all')` and use `pvalues_`).",
         f"{imp}SelectKBest, f_regression\nskb = SelectKBest(f_regression, k='all').fit(df, y)\n"
         "result = list(df.columns[skb.pvalues_ < 0.05])"),
        ("RFE", "Standardize `df`, then fit `RFE(LinearRegression(), n_features_to_select=1)` and return the column "
                "names from most to least important.",
         f"{imp}RFE\nfrom sklearn.linear_model import LinearRegression\nfrom sklearn.preprocessing import StandardScaler\n"
         "rfe = RFE(LinearRegression(), n_features_to_select=1).fit(StandardScaler().fit_transform(df), y)\n"
         "result = list(df.columns[np.argsort(rfe.ranking_)])"),
    ]
    return catalog(rng, items, t.setup, f"{t.desc}.", tol=1e-6)


@template(M, "medium")
def redundant_copies(rng):
    setup, desc, d, i = _signal_noise(rng)
    task = rng.choice(["kbest", "rfe", "sfs"])
    setup += "\nX2 = np.column_stack([X, X[:, 0] + 0.01 * np.random.default_rng(0).normal(size=len(X))])   # a near-copy of column 0"
    setup += "\nXs2 = StandardScaler().fit_transform(X2)"
    head = (f"{desc}. `X2` appends a near-copy of column 0 as the last column (index {d}); `Xs2` is it standardized. "
            f"Univariate filters score features one at a time, so they can't see redundancy.")
    if task == "kbest":
        return P(f"{head} Fit `SelectKBest(f_classif, k={i + 1})` on `X2, y` and return `(0 in kept, {d} in kept)` for its "
                 f"kept indices.", setup,
                 f"{FS}SelectKBest, f_classif\nkept = set(SelectKBest(f_classif, k={i + 1}).fit(X2, y).get_support(indices=True).tolist())\n"
                 f"result = (0 in kept, {d} in kept)", require=["SelectKBest"])
    if task == "rfe":
        return P(f"{head} Fit `RFE({LR}, n_features_to_select={i})` on `Xs2, y` and return the kept indices.", setup,
                 f"{FS}RFE\nfrom sklearn.linear_model import LogisticRegression\n"
                 f"result = RFE({LR}, n_features_to_select={i}).fit(Xs2, y).get_support(indices=True)", require=["RFE"])
    return P(f"{head} Fit `SequentialFeatureSelector({LR}, n_features_to_select={i}, cv=3)` (forward) on `Xs2, y` and "
             f"return the kept indices: adding the copy after column 0 brings nothing.", setup,
             f"{FS}SequentialFeatureSelector\nfrom sklearn.linear_model import LogisticRegression\n"
             f"sfs = SequentialFeatureSelector({LR}, n_features_to_select={i}, cv=3).fit(Xs2, y)\n"
             "result = sfs.get_support(indices=True)", require=["SequentialFeatureSelector"])


# ================================================================= hard

@template(M, "hard")
def selection_leak_holdout(rng):
    n, d, k, s = rng.choice([60, 80, 100]), rng.choice([400, 600, 1000]), rng.choice([5, 10, 20]), seeded(rng)
    m = n // 2
    setup = (f"g = np.random.default_rng({s})\nX = g.normal(size=({n}, {d}))   # pure noise\n"
             f"y = np.array([0, 1] * {n // 2})\ng.shuffle(y)")
    task = rng.choice(["leaky", "honest", "both"])
    leaky = (f"skb = SelectKBest(f_classif, k={k}).fit(X, y)\nZ = skb.transform(X)\n"
             f"leaky = {LR}.fit(Z[:{m}], y[:{m}]).score(Z[{m}:], y[{m}:])\n")
    honest = (f"skb2 = SelectKBest(f_classif, k={k}).fit(X[:{m}], y[:{m}])\n"
              f"honest = {LR}.fit(skb2.transform(X[:{m}]), y[:{m}]).score(skb2.transform(X[{m}:]), y[{m}:])\n")
    imp = f"{FS}SelectKBest, f_classif\nfrom sklearn.linear_model import LogisticRegression\n"
    code = (f"```\nskb = SelectKBest(f_classif, k={k}).fit(X, y)       # uses the test labels!\nZ = skb.transform(X)\n"
            f"clf = {LR}.fit(Z[:{m}], y[:{m}])\nprint(clf.score(Z[{m}:], y[{m}:]))\n```")
    head = (f"`X` is {n} rows of {d} pure-noise features, `y` random balanced labels; the first {m} rows are for "
            f"training, the rest for testing. This evaluation leaks:\n{code}")
    if task == "leaky":
        return P(f"{head}\nRun it as written and return the (misleadingly high) test accuracy.", setup,
                 imp + leaky + "result = leaky", require=["SelectKBest"])
    if task == "honest":
        return P(f"{head}\nFix it: select on the training rows only, then return the honest test accuracy.", setup,
                 imp + honest + "result = honest", require=["SelectKBest"])
    return P(f"{head}\nReturn `(leaky, honest)` test accuracies: as written, and with the selector fitted on the "
             f"training rows only.", setup, imp + leaky + honest + "result = (leaky, honest)", require=["SelectKBest"])


@template(M, "hard")
def rfe_by_hand(rng):
    setup, desc, d, inf = _fs_data(rng, synth=0.6)
    setup = _scaled(setup)
    k = rng.randint(1, 3)
    task = rng.choice(["ranking", "support"])
    loop = ("from sklearn.linear_model import LogisticRegression\nremaining = list(range(Xs.shape[1]))\n"
            "rank = np.ones(Xs.shape[1], dtype=int)\n"
            f"while len(remaining) > {k}:\n    coef = {LR}.fit(Xs[:, remaining], y).coef_\n"
            "    imp = (coef ** 2).sum(axis=0)\n    worst = remaining[int(np.argmin(imp))]\n    remaining.remove(worst)\n"
            "    rank[[j for j in range(Xs.shape[1]) if j not in remaining]] += 1\n")
    head = (f"{desc}; `Xs` is `X` standardized. Reimplement `RFE({LR}, n_features_to_select={k}, step=1)`: while more "
            f"than {k} features remain, fit the model on the remaining columns, compute each one's importance as the sum "
            f"over classes of its squared coefficients (`(coef_ ** 2).sum(axis=0)`), and drop the least important.")
    if task == "ranking":
        return P(f"{head} `ranking_` starts at 1 for everyone and, after each drop, grows by 1 for every feature "
                 f"dropped so far. Return your ranking array (it equals `RFE(...).ranking_`).", setup,
                 loop + "result = rank", require=["LogisticRegression", "coef_"])
    return P(f"{head} Return the sorted indices of the {k} surviving features.", setup,
             loop + "result = sorted(remaining)", require=["LogisticRegression", "coef_"])


@template(M, "hard")
def forward_selection_by_hand(rng):
    setup, desc, d, inf = _fs_data(rng, synth=0.6)
    setup = _scaled(setup)
    m, k = min(d, 7), rng.randint(2, 3)
    nn = rng.choice([3, 5])
    direction = rng.choice(["forward", "backward"])
    if direction == "forward":
        loop = (f"chosen = []\nfor _ in range({k}):\n"
                f"    cand = [j for j in range({m}) if j not in chosen]\n"
                f"    scores = [cross_val_score(KNeighborsClassifier({nn}), Xs[:, sorted(chosen + [j])], y, cv=3).mean() for j in cand]\n"
                "    chosen.append(cand[int(np.argmax(scores))])\nresult = sorted(chosen)")
        how = (f"start from no features; {k} times, add the candidate whose addition gives the best mean 3-fold "
               f"`cross_val_score` of `KNeighborsClassifier({nn})` (ties: the smallest index)")
    else:
        loop = (f"chosen = list(range({m}))\nwhile len(chosen) > {k}:\n"
                f"    scores = [cross_val_score(KNeighborsClassifier({nn}), Xs[:, [c for c in chosen if c != j]], y, "
                f"cv=3).mean() for j in chosen]\n"
                "    chosen.remove(chosen[int(np.argmax(scores))])\nresult = sorted(chosen)")
        how = (f"start from all {m} features; while more than {k} remain, remove the one whose removal leaves the best "
               f"mean 3-fold `cross_val_score` of `KNeighborsClassifier({nn})` (ties: the smallest index)")
    return P(f"{desc}; `Xs` is `X` standardized. Reimplement `SequentialFeatureSelector(KNeighborsClassifier({nn}), "
             f"n_features_to_select={k}, direction={direction!r}, cv=3)` on the first {m} columns of `Xs`: {how}. Return "
             f"the sorted indices of the selected features (they match `get_support(indices=True)`).", setup,
             "from sklearn.model_selection import cross_val_score\nfrom sklearn.neighbors import KNeighborsClassifier\n"
             + loop, require=["cross_val_score", "KNeighborsClassifier"])


@template(M, "hard")
def sfm_threshold_by_hand(rng):
    setup, desc, d, inf = _fs_data(rng)
    rf = "RandomForestClassifier(n_estimators=40, random_state=0)"
    task = rng.choice(["median", "scaled_mean", "max_features"])
    imp = "from sklearn.ensemble import RandomForestClassifier\n"
    if task == "median":
        return P(f"{desc}. Reproduce `SelectFromModel({rf}, threshold='median')` without `SelectFromModel`: fit the "
                 f"forest on `X, y`, take the median of `feature_importances_` as the cut-off, keep importances ≥ "
                 f"cut-off. Return the kept indices.", setup,
                 f"{imp}fi = {rf}.fit(X, y).feature_importances_\nresult = np.flatnonzero(fi >= np.median(fi))",
                 require=["feature_importances_"])
    if task == "scaled_mean":
        f = rng.choice([0.5, 0.75, 1.25, 2.0])
        return P(f"{desc}. Reproduce `SelectFromModel({rf}, threshold='{f}*mean')` without `SelectFromModel` (cut-off: "
                 f"{f} × the mean importance; keep importances ≥ cut-off). Return the kept names.", setup,
                 f"{imp}fi = {rf}.fit(X, y).feature_importances_\nresult = [names[j] for j in np.flatnonzero(fi >= {f} * fi.mean())]",
                 require=["feature_importances_"])
    k = rng.randint(2, 5)
    return P(f"{desc}. Reproduce `SelectFromModel({rf}, threshold='mean', max_features={k})` without "
             f"`SelectFromModel`: keep the features with importance ≥ the mean, but at most the {k} most important "
             f"of them. Return the kept indices in increasing order.", setup,
             f"{imp}fi = {rf}.fit(X, y).feature_importances_\nok = np.flatnonzero(fi >= fi.mean())\n"
             f"result = np.sort(ok[np.argsort(-fi[ok], kind='stable')][:{k}])", require=["feature_importances_"])


@template(M, "hard")
def anova_vectorized(rng):
    setup, desc, d, inf = _fs_data(rng)
    task = rng.choice(["F", "top"])
    sol = ("cls = np.unique(y)\nk, n = len(cls), len(y)\nmu = X.mean(axis=0)\n"
           "means = np.array([X[y == c].mean(axis=0) for c in cls])\ncounts = np.array([(y == c).sum() for c in cls])\n"
           "ssb = (counts[:, None] * (means - mu) ** 2).sum(axis=0)\n"
           "ssw = sum(((X[y == c] - means[i]) ** 2).sum(axis=0) for i, c in enumerate(cls))\n"
           "F = (ssb / (k - 1)) / (ssw / (n - k))\n")
    head = (f"{desc}. Compute the ANOVA F-score of **every** feature at once with NumPy (between-class sum of squares "
            f"/ (k − 1) over within-class sum of squares / (n − k)), looping only over the classes, never over "
            f"features.")
    if task == "F":
        return P(f"{head} Return the F array (it equals `f_classif(X, y)[0]`).", setup, sol + "result = F", tol=1e-6)
    kk = rng.randint(2, 5)
    return P(f"{head} Then return the indices `SelectKBest(f_classif, k={kk})` would keep, in increasing order.", setup,
             sol + f"result = np.sort(np.argsort(F)[-{kk}:])", tol=1e-6)


@template(M, "hard")
def stability_selection(rng):
    setup, desc, d, inf = _fs_data(rng, synth=0.7)
    b, k, s = rng.randint(5, 10), rng.randint(2, 4), seeded(rng)
    task = rng.choice(["freq", "stable"])
    loop = (f"{FS}SelectKBest, f_classif\nrs = np.random.default_rng({s})\ncount = np.zeros(X.shape[1])\n"
            f"for _ in range({b}):\n    idx = rs.integers(0, len(X), len(X))\n"
            f"    count += SelectKBest(f_classif, k={k}).fit(X[idx], y[idx]).get_support()\n")
    head = (f"{desc}. How stable is the selection? With `rs = np.random.default_rng({s})`, repeat {b} times: draw a "
            f"bootstrap sample `idx = rs.integers(0, len(X), len(X))` and fit `SelectKBest(f_classif, k={k})` on "
            f"`X[idx], y[idx]`.")
    if task == "freq":
        return P(f"{head} Return, for every feature, the fraction of runs that kept it.", setup,
                 loop + f"result = count / {b}", require=["SelectKBest"])
    return P(f"{head} Return the names of the features kept in **every** run.", setup,
             loop + f"result = [names[j] for j in np.flatnonzero(count == {b})]", require=["SelectKBest"])


@template(M, "hard")
def discrete_mutual_info(rng):
    n, d, s = rng.randint(80, 200), rng.randint(4, 7), seeded(rng)
    setup = (f"g = np.random.default_rng({s})\ny = g.integers(0, 3, {n})\nX = g.integers(0, 4, ({n}, {d}))\n"
             "X[:, 0] = (y + g.integers(0, 2, len(y))) % 4   # linked to y\n"
             "X[:, 1] = np.where(g.random(len(y)) < 0.6, y, X[:, 1])   # linked to y")
    j = rng.randrange(d)
    task = rng.choice(["one", "all"])
    mi = ("def mi(x, t):\n    n = len(x)\n    out = 0.0\n    for a in np.unique(x):\n        for b in np.unique(t):\n"
          "            pab = ((x == a) & (t == b)).mean()\n"
          "            if pab > 0:\n                out += pab * np.log(pab / ((x == a).mean() * (t == b).mean()))\n"
          "    return out\n")
    head = (f"`X` holds {d} categorical features coded 0-3 for {n} rows and `y` a class 0-2. For discrete features "
            f"`mutual_info_classif(X, y, discrete_features=True)` is exact: MI = Σ p(a, b) · log(p(a, b) / (p(a) p(b))) "
            f"over the value pairs (natural log).")
    if task == "one":
        return P(f"{head} Compute it by hand for feature {j} and return `(by_hand, sklearn)`.", setup,
                 f"{FS}mutual_info_classif\n" + mi +
                 f"result = (mi(X[:, {j}], y), mutual_info_classif(X, y, discrete_features=True)[{j}])",
                 require=["mutual_info_classif"], tol=1e-6)
    return P(f"{head} Compute it by hand for every feature and return the array.", setup,
             mi + "result = np.array([mi(X[:, j], y) for j in range(X.shape[1])])", tol=1e-6)


@template(M, "hard")
def broken_selection(rng):
    setup, desc, d, inf = _fs_data(rng)
    k = rng.randint(2, 4)
    items = [
        ("MinMaxScaler", f"This crashes (`chi2` needs non-negative input):\n```\nZ = StandardScaler().fit_transform(X)\n"
                         f"SelectKBest(chi2, k={k}).fit(Z, y)\n```\nFix it by min-max scaling instead and return the kept "
                         f"indices.",
         f"{FS}SelectKBest, chi2\nfrom sklearn.preprocessing import MinMaxScaler\n"
         f"result = SelectKBest(chi2, k={k}).fit(MinMaxScaler().fit_transform(X), y).get_support(indices=True)"),
        ("get_support", f"This code wants the kept column **indices** but prints a boolean mask:\n```\n"
                        f"skb = SelectKBest(f_classif, k={k}).fit(X, y)\nprint(skb.get_support())\n```\nReturn the "
                        f"indices instead.",
         f"{FS}SelectKBest, f_classif\nresult = SelectKBest(f_classif, k={k}).fit(X, y).get_support(indices=True)"),
        ("k", f"This raises a `ValueError` because `k` is larger than the number of features:\n```\n"
              f"SelectKBest(f_classif, k={d + 5}).fit(X, y)\n```\nUse `k='all'` and return the shape of the "
              f"transformed `X`.",
         f"{FS}SelectKBest, f_classif\nresult = SelectKBest(f_classif, k='all').fit(X, y).transform(X).shape",
         {"require": ["SelectKBest"]}),
        ("importance_getter", f"This raises a `ValueError`: `KNeighborsClassifier` has no `coef_` or "
                              f"`feature_importances_` for `RFE` to rank.\n```\nRFE(KNeighborsClassifier(), "
                              f"n_features_to_select={k}).fit(X, y)\n```\nUse a model that has weights, "
                              f"`{LR}`, on standardized `X`, and return the kept indices.",
         f"{FS}RFE\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.preprocessing import StandardScaler\n"
         f"result = RFE({LR}, n_features_to_select={k}).fit(StandardScaler().fit_transform(X), y).get_support(indices=True)",
         {"require": ["RFE", "LogisticRegression"]}),
        ("transform", f"This code selects on the training rows but forgets to apply the **same** selection to the test "
                      f"rows (it refits on them, using the test labels):\n```\nskb = SelectKBest(f_classif, k={k})\n"
                      f"Ztr = skb.fit_transform(X[:100], y[:100])\nZte = skb.fit_transform(X[100:], y[100:])\n```\n"
                      f"Return the correct `Zte`.",
         f"{FS}SelectKBest, f_classif\nskb = SelectKBest(f_classif, k={k}).fit(X[:100], y[:100])\n"
         "result = skb.transform(X[100:])"),
        ("feature_names_in_", "This loses the column names (the selector was fitted on an array):\n```\n"
                              f"skb = SelectKBest(f_classif, k={k}).fit(X, y)\nskb.get_feature_names_out()   # x0, x3 ...\n"
                              "```\nFit on `pd.DataFrame(X, columns=names)` instead and return the real names.",
         f"{FS}SelectKBest, f_classif\n"
         f"result = list(SelectKBest(f_classif, k={k}).fit(pd.DataFrame(X, columns=names), y).get_feature_names_out())",
         {"require": ["DataFrame"]}),
    ]
    return catalog(rng, items, setup, f"{desc}.", tol=1e-6)


@template(M, "hard")
def correlation_filter(rng):
    t = table(rng, numeric_only=True, n=(80, 200))
    s = seeded(rng)
    thr = rng.choice([0.8, 0.9, 0.95])
    setup = (t.setup + f"\ng = np.random.default_rng({s})\nfor c in list(df.columns)[:2]:\n"
             "    df[c + '_copy'] = df[c] * g.uniform(0.5, 2) + g.normal(0, df[c].std() * 0.1, len(df))")
    task = rng.choice(["drop", "keep"])
    sol = (f"corr = df.corr().abs().to_numpy()\nkeep = []\nfor j in range(df.shape[1]):\n"
           f"    if all(corr[j, i] <= {thr} for i in keep):\n        keep.append(j)\n")
    head = (f"{t.desc}. Noisy rescaled copies of two columns were appended (`*_copy`). Univariate selectors keep both "
            f"copies; a simple fix is a greedy correlation filter: go through the columns left to right and keep a "
            f"column only if its absolute Pearson correlation with every **already kept** column is ≤ {thr}.")
    if task == "drop":
        return P(f"{head} Return the names of the dropped columns.", setup,
                 sol + "result = [c for j, c in enumerate(df.columns) if j not in keep]")
    return P(f"{head} Return the names of the kept columns.", setup, sol + "result = list(df.columns[keep])")


@template(M, "hard")
def percentile_rule_by_hand(rng):
    d, s = rng.randint(8, 15), seeded(rng)
    p = rng.choice([10, 20, 25, 30, 40, 50])
    setup = (f"g = np.random.default_rng({s})\nscores = g.integers(0, 6, {d}).astype(float)   # many ties\n"
             f"percentile = {p}")
    return P(f"`scores` are {d} feature scores with many ties. Reproduce `SelectPercentile`'s mask without fitting it: "
             f"`threshold = np.percentile(scores, 100 - percentile)`; keep every score strictly above it; then, among "
             f"the features whose score **equals** the threshold, keep the lowest-index ones until `int(len(scores) * "
             f"percentile / 100)` features are kept in total (none if the strict ones already reach that). Return the "
             f"boolean mask.", setup,
             "thr = np.percentile(scores, 100 - percentile)\nmask = scores > thr\nties = np.flatnonzero(scores == thr)\n"
             "room = int(len(scores) * percentile / 100) - int(mask.sum())\nif room > 0:\n    mask[ties[:room]] = True\n"
             "result = mask", require=["percentile"])


@template(M, "hard")
def selection_workflows(rng):
    setup, desc, d, i = _signal_noise(rng)
    s = seeded(rng)
    items = [
        ("cross_val_score", "Which selector generalizes best? Return a dict of mean 5-fold `cross_val_score` accuracies "
                            f"of `make_pipeline(selector, {LR})` on `Xs, y` for the selectors `'kbest'`: "
                            f"`SelectKBest(f_classif, k={i})`, `'rfe'`: `RFE({LR}, n_features_to_select={i})` and "
                            f"`'none'`: `'passthrough'`.",
         f"{FS}SelectKBest, f_classif, RFE\nfrom sklearn.pipeline import make_pipeline\n"
         "from sklearn.linear_model import LogisticRegression\nfrom sklearn.model_selection import cross_val_score\n"
         f"sel = {{'kbest': SelectKBest(f_classif, k={i}), 'rfe': RFE({LR}, n_features_to_select={i}), 'none': 'passthrough'}}\n"
         f"result = {{name: cross_val_score(make_pipeline(s, {LR}), Xs, y, cv=5).mean() for name, s in sel.items()}}"),
        ("permutation_importance", f"Model-agnostic importance: fit `{LR}` on `Xs, y`, run `permutation_importance(clf, "
                                   f"Xs, y, n_repeats=3, random_state={s})` (from `sklearn.inspection`) and return the "
                                   f"indices of the {i} largest `importances_mean`, largest first.",
         "from sklearn.linear_model import LogisticRegression\nfrom sklearn.inspection import permutation_importance\n"
         f"clf = {LR}.fit(Xs, y)\nr = permutation_importance(clf, Xs, y, n_repeats=3, random_state={s})\n"
         f"result = np.argsort(-r.importances_mean)[:{i}]"),
        ("SelectKBest", f"Feature selection as a learning curve: for k = 1 … {min(d, 8)}, return the training accuracy "
                        f"of `{LR}` on the k best `f_classif` columns of `Xs` (list).",
         f"{FS}SelectKBest, f_classif\nfrom sklearn.linear_model import LogisticRegression\n"
         f"result = []\nfor k in range(1, {min(d, 8) + 1}):\n    Z = SelectKBest(f_classif, k=k).fit_transform(Xs, y)\n"
         f"    result.append({LR}.fit(Z, y).score(Z, y))"),
        ("RFECV", f"Return `(rfecv_kept, true_informative_kept)`: how many features `RFECV({LR}, cv=5)` keeps on `Xs, y` "
                  f"and how many of them are among the first {i} columns.",
         f"{FS}RFECV\nfrom sklearn.linear_model import LogisticRegression\nr = RFECV({LR}, cv=5).fit(Xs, y)\n"
         f"result = (int(r.n_features_), int((r.get_support(indices=True) < {i}).sum()))"),
        ("SelectFromModel", "Prefit trick: `SelectFromModel(model, prefit=True)` reuses an already fitted model. Fit "
                            "`RandomForestClassifier(n_estimators=50, random_state=0)` on `X, y`, wrap it with "
                            "`prefit=True, threshold='median'` and return `transform(X).shape`.",
         f"{FS}SelectFromModel\nfrom sklearn.ensemble import RandomForestClassifier\n"
         "rf = RandomForestClassifier(n_estimators=50, random_state=0).fit(X, y)\n"
         "result = SelectFromModel(rf, prefit=True, threshold='median').transform(X).shape"),
        ("SequentialFeatureSelector", f"Let `SequentialFeatureSelector` decide when to stop: "
                                      f"`SequentialFeatureSelector({LR}, n_features_to_select='auto', tol=0.005, cv=3)` "
                                      f"adds features while the CV score improves by more than `tol`. Fit it on the "
                                      f"first 8 columns of `Xs` and return the kept indices.",
         f"{FS}SequentialFeatureSelector\nfrom sklearn.linear_model import LogisticRegression\n"
         f"sfs = SequentialFeatureSelector({LR}, n_features_to_select='auto', tol=0.005, cv=3).fit(Xs[:, :8], y)\n"
         "result = sfs.get_support(indices=True)"),
    ]
    return catalog(rng, items, setup, f"{desc}.", tol=1e-6)


# ================================================================= challenge

@template(M, "challenge")
def chi2_term_selection(rng):
    nd, V, nnz, C, s = rng.randint(5, 7), rng.randint(5, 7), rng.randint(12, 18), rng.randint(2, 3), seeded(rng)
    report = rng.choice(["scores", "topk", "pvalues"])
    gen = ("g = np.random.default_rng({s})\nn_docs, V = {nd}, {V}\ny = g.integers(0, {C}, n_docs)\ny[:{C}] = np.arange({C})\n"
           "doc = g.integers(0, n_docs, {nnz})\nterm = np.concatenate([np.arange(V), g.integers(0, V, {nnz} - V)])\n"
           "count = g.integers(1, 4, {nnz})\nk = {k}\nreport = {report!r}")
    setup = gen.format(s=s, nd=nd, V=V, C=C, nnz=nnz, k=rng.randint(2, 3), report=report)
    tests = [gen.format(s=s + i, nd=20000, V=30000, C=C, nnz=200000, k=500, report=report) for i in (1, 2)]
    sol = ("obs = np.bincount(y[doc] * V + term, weights=count, minlength=y.max() * V + V).reshape(-1, V)\n"
           "prior = np.bincount(y) / len(y)\nexp = prior[:, None] * obs.sum(axis=0)[None, :]\n"
           "chi = ((obs - exp) ** 2 / exp).sum(axis=0)\n")
    if report == "scores":
        sol += "result = chi"
        what = "the chi² score of every term (a float array of length `V`)"
    elif report == "pvalues":
        sol += "from scipy.stats import chi2 as chi2_dist\nresult = chi2_dist.sf(chi, len(prior) - 1)"
        what = ("the p-value of every term: the upper tail of the chi² distribution with (number of classes − 1) degrees "
                "of freedom at the score (`scipy.stats.chi2.sf`)")
    else:
        sol += "result = np.sort(np.argsort(chi, kind='mergesort')[-k:])"
        what = ("the ids of the `k` terms `SelectKBest(chi2, k=k)` keeps, in increasing order (it stable-sorts the "
                "scores and takes the last `k`, so among equal scores the **larger** term id wins)")
    return challenge(
        "Chi-squared keywords from a sparse log",
        "A spam filter keeps only the most class-revealing words. The training corpus is given as a sparse log: entry "
        "`i` says document `doc[i]` contains term `term[i]` `count[i]` times (a (doc, term) pair may repeat; counts add "
        "up). Document `j` has class `y[j]` (every class occurs; some documents have no terms at all, and every term "
        "occurs). `chi2` scores term `t` by Σ over classes c of (O − E)² / E, where O is the total count of `t` in class "
        "`c` documents and E = (fraction of documents in class `c`) × (total count of `t`). Return " + what +
        f". (`report` = {report!r}.)",
        setup, sol, tests, "n_docs ≤ 2·10^4, V ≤ 3·10^4, log entries ≤ 2·10^5, at most 3 classes",
        speed="O(entries + V · classes): never build the dense document × term matrix",
        answer_desc=" (a NumPy array)", tol=1e-6)


@template(M, "challenge")
def percentile_queries(rng):
    d, q, s = rng.randint(8, 12), rng.randint(4, 6), seeded(rng)
    report = rng.choice(["count", "last"])
    gen = ("g = np.random.default_rng({s})\nscores = g.integers(0, {hi}, {d}).astype(float)\n"
           "p = g.integers(1, 100, {q})\nreport = {report!r}")
    setup = gen.format(s=s, hi=6, d=d, q=q, report=report)
    tests = [gen.format(s=s + i, hi=1000, d=100000, q=100000, report=report) for i in (1, 2)]
    sol = ("n = len(scores)\nthr = np.percentile(scores, 100 - p)   # all thresholds in one call\n"
           "order = np.lexsort((np.arange(n), scores))   # by score, then index\nsv = scores[order]\n"
           "lo = np.searchsorted(sv, thr, 'left')\nhi = np.searchsorted(sv, thr, 'right')\n"
           "greater, ties = n - hi, hi - lo\nroom = (n * p) // 100 - greater\n"
           "kept_ties = np.where(room >= 0, np.minimum(room, ties), np.maximum(ties + room, 0))   # Python slice ties[:room]\n")
    if report == "count":
        sol += "result = greater + kept_ties"
        what = "the number of features kept"
    else:
        sol += ("suffix = np.maximum.accumulate(order[::-1])[::-1]   # largest index among sorted positions >= i\n"
                "suffix = np.append(suffix, -1)\nbest = suffix[hi]\n"
                "tie_last = np.where(kept_ties > 0, order[np.minimum(lo + kept_ties - 1, n - 1)], -1)\n"
                "result = np.maximum(best, tie_last)")
        what = "the **largest index** of a kept feature (-1 if none is kept)"
    return challenge(
        "SelectPercentile under many settings",
        "A dashboard lets analysts drag a percentile slider over fixed feature `scores` (integers stored as floats: "
        "lots of ties). For each query percentile `p[i]` (1-99), `SelectPercentile` builds its mask like this: "
        "`threshold = np.percentile(scores, 100 - p[i])` (NumPy's default linear interpolation); keep every feature "
        "with a score strictly above it; then `ties = np.flatnonzero(scores == threshold)` and keep "
        "`ties[:int(len(scores) * p[i] / 100) - n_strict]` too, with Python slice semantics (a negative stop drops "
        f"ties from the end!). For every query return {what}. (`report` = {report!r}.) Tip: `np.percentile` accepts "
        "an array of percentiles.", setup, sol, tests, "len(scores), number of queries ≤ 10^5",
        speed="O((n + q) log n): sort once, binary-search each threshold", answer_desc=" (an int array, one value per query)")


@template(M, "challenge")
def rfe_fixed_scores(rng):
    d, s = rng.randint(7, 11), seeded(rng)
    mode = rng.choice(["int", "fraction"])
    if mode == "int":
        params = f"n_select = {rng.randint(1, 3)}\nstep = {rng.randint(1, 3)}"
        tparams = "n_select = 1000\nstep = 7"
        rule = ("`RFE(model, n_features_to_select=n_select, step=step)`")
        prep = "n_keep, st = n_select, step\n"
    else:
        params = f"n_select = {rng.choice([0.2, 0.3, 0.5])}\nstep = {rng.choice([0.1, 0.25, 0.4])}"
        tparams = "n_select = 0.05\nstep = 0.001"
        rule = ("`RFE(model, n_features_to_select=n_select, step=step)` with **fractions**: it keeps "
                "`int(n_features * n_select)` features and removes `int(max(1, step * n_features))` per round (computed "
                "once, from the initial number of features)")
        prep = "n_keep, st = int(len(w) * n_select), int(max(1, step * len(w)))\n"
    setup = f"g = np.random.default_rng({s})\nw = g.normal(0, 1, {d})\n{params}"
    tests = [f"g = np.random.default_rng({s + i})\nw = g.normal(0, 1, 200000)\n{tparams}" for i in (1, 2)]
    sol = (prep + "d = len(w)\norder = np.argsort(np.abs(w), kind='stable')   # least important first\n"
           "n_drop = d - n_keep\nrounds = -(-n_drop // st)\nrank = np.ones(d, dtype=int)\n"
           "pos = np.arange(n_drop)\nrank[order[:n_drop]] = rounds - pos // st + 1\nresult = rank")
    return challenge(
        "Recursive elimination without refitting",
        "A linear model on **orthogonal** features learns the same coefficient for a feature whatever other features "
        "are present, so its importances never change: feature `j` always has `coef_[j] = w[j]` (no ties in `|w|`). "
        f"Feature selection uses {rule}. Each round, RFE refits the model on the remaining features and removes the "
        "`step` least important ones (by `coef_ ** 2`), or just enough to reach the target; after every round, every "
        "removed feature's rank grows by 1. Kept features have rank 1, so the last round's victims end at rank 2. "
        f"Return `ranking_` (an int array) without simulating the refits. (mode: {mode!r}.)",
        setup, sol, tests, "len(w) ≤ 2·10^5", speed="O(n log n): one sort", answer_desc=" (an int array)")
