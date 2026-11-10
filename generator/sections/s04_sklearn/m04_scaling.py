"""4.04 Scaling & feature transforms: the scalers, Normalizer, quantile / power transforms, polynomial and spline
features, FunctionTransformer, binning and Binarizer, and the fit-on-train rule."""

from ._common import P, catalog, challenge, lit, seeded, synth_clf, table, template

M = "4.04"
PRE = "sklearn.preprocessing"


def _outlier_matrix(rng, rows=(5, 8)):
    """A small matrix: column 0 small positive values, column 1 in the hundreds with one huge outlier, column 2 signed."""
    n = rng.randint(*rows)
    c0 = [round(rng.uniform(0.5, 9.5), 1) for _ in range(n)]
    c1 = [float(rng.randint(100, 900)) for _ in range(n)]
    c1[rng.randrange(n)] = float(rng.choice([8000, 10000, 25000]))
    c2 = [round(rng.uniform(-6, 6), 1) for _ in range(n)]
    return [[a, b, c] for a, b, c in zip(c0, c1, c2)]


def _skewed(rng, n=(80, 200)):
    """Setup code for a positive right-skewed 2-column matrix `X` (log-normal and exponential)."""
    s, m = seeded(rng), rng.randint(*n)
    return (f"rng = np.random.default_rng({s})\nX = np.column_stack([rng.lognormal(3, 0.8, {m}), rng.exponential(5, {m}) "
            f"+ 0.1])"), m


# ================================================================= lite

@template(M, "lite")
def scaler_zoo(rng):
    rows = _outlier_matrix(rng)
    n = len(rows)
    lo, hi = rng.choice([(0, 1), (-1, 1), (0, 10)])
    t = round(rng.uniform(1, 5), 1)
    norm = rng.choice(["l1", "l2", "max"])
    b = rng.choice([2, 3, 4])
    imp = f"from {PRE} import "
    items = [
        ("StandardScaler", "Standardize every column (`StandardScaler`).",
         imp + "StandardScaler\nresult = StandardScaler().fit_transform(X)"),
        ("MinMaxScaler", f"Rescale every column to the range [{lo}, {hi}] with `MinMaxScaler(feature_range=({lo}, {hi}))`.",
         imp + f"MinMaxScaler\nresult = MinMaxScaler(feature_range=({lo}, {hi})).fit_transform(X)"),
        ("RobustScaler", "Scale with `RobustScaler` (subtract the median, divide by the interquartile range).",
         imp + "RobustScaler\nresult = RobustScaler().fit_transform(X)"),
        ("MaxAbsScaler", "Divide each column by its largest absolute value (`MaxAbsScaler`; no centring).",
         imp + "MaxAbsScaler\nresult = MaxAbsScaler().fit_transform(X)"),
        ("Normalizer", f"Scale each **row** to unit {norm} norm with `Normalizer(norm='{norm}')`.",
         imp + f"Normalizer\nresult = Normalizer(norm='{norm}').fit_transform(X)"),
        ("Binarizer", f"Turn values above {t} into 1 and the rest into 0 (`Binarizer(threshold={t})`).",
         imp + f"Binarizer\nresult = Binarizer(threshold={t}).fit_transform(X)"),
        ("QuantileTransformer", f"Map each column to its empirical quantiles in [0, 1] with "
                                f"`QuantileTransformer(n_quantiles={n})` (as many quantiles as rows).",
         imp + f"QuantileTransformer\nresult = QuantileTransformer(n_quantiles={n}).fit_transform(X)"),
        ("PowerTransformer", "Make the columns more Gaussian with `PowerTransformer()` (Yeo-Johnson, then standardized).",
         imp + "PowerTransformer\nresult = PowerTransformer().fit_transform(X)", {"tol": 1e-5}),
        ("FunctionTransformer", "Apply `np.log1p` to the first two columns (`X[:, :2]`, all positive) with "
                                "`FunctionTransformer(np.log1p)`.",
         imp + "FunctionTransformer\nresult = FunctionTransformer(np.log1p).fit_transform(X[:, :2])"),
        ("KBinsDiscretizer", f"Cut each column into {b} equal-width bins and return the bin numbers: "
                             f"`KBinsDiscretizer(n_bins={b}, encode='ordinal', strategy='uniform')`.",
         imp + f"KBinsDiscretizer\nresult = KBinsDiscretizer(n_bins={b}, encode='ordinal', strategy='uniform').fit_transform(X)"),
    ]
    return catalog(rng, items, f"X = np.array({lit(rows)})",
                   f"`X` is a {n} × 3 matrix (column 1 has a huge outlier). Transformers live in `{PRE}`.", tol=1e-6)


@template(M, "lite")
def learned_statistics(rng):
    rows = _outlier_matrix(rng)
    n = len(rows)
    d = rng.choice([2, 3])
    b = rng.choice([2, 3, 4])
    imp = f"from {PRE} import "
    pos = "np.abs(X) + 1"
    items = [
        ("data_min_", "Fit a `MinMaxScaler` and return `(data_min_, data_max_)`.",
         imp + "MinMaxScaler\nmm = MinMaxScaler().fit(X)\nresult = (mm.data_min_, mm.data_max_)"),
        ("data_range_", "Fit a `MinMaxScaler` and return the per-column ranges it learned (`data_range_`).",
         imp + "MinMaxScaler\nresult = MinMaxScaler().fit(X).data_range_"),
        ("center_", "Fit a `RobustScaler` and return `(center_, scale_)`: the medians and interquartile ranges.",
         imp + "RobustScaler\nrs = RobustScaler().fit(X)\nresult = (rs.center_, rs.scale_)"),
        ("max_abs_", "Fit a `MaxAbsScaler` and return `max_abs_`.", imp + "MaxAbsScaler\nresult = MaxAbsScaler().fit(X).max_abs_"),
        ("lambdas_", f"Fit `PowerTransformer(method='box-cox')` on the positive matrix `{pos}` (Box-Cox needs positive "
                     f"data) and return the fitted exponents `lambdas_`.",
         imp + f"PowerTransformer\nresult = PowerTransformer(method='box-cox').fit({pos}).lambdas_", {"tol": 1e-4}),
        ("bin_edges_", f"Fit `KBinsDiscretizer(n_bins={b}, encode='ordinal', strategy='uniform')` and return the bin "
                       f"edges of column 1 (`bin_edges_[1]`).",
         imp + f"KBinsDiscretizer\nresult = KBinsDiscretizer(n_bins={b}, encode='ordinal', strategy='uniform').fit(X).bin_edges_[1]"),
        ("quantiles_", f"Fit `QuantileTransformer(n_quantiles={n})` and return the quantiles it stored for column 0 "
                       f"(`quantiles_[:, 0]`; the sorted column).",
         imp + f"QuantileTransformer\nresult = QuantileTransformer(n_quantiles={n}).fit(X).quantiles_[:, 0]"),
        ("n_output_features_", f"How many columns does `PolynomialFeatures(degree={d})` produce from the 3 columns of "
                               f"`X` (bias included)? Fit it and return `n_output_features_`.",
         imp + f"PolynomialFeatures\nresult = PolynomialFeatures(degree={d}).fit(X).n_output_features_"),
        ("powers_", "Fit `PolynomialFeatures(degree=2, include_bias=False)` and return `powers_` (one row per output "
                    "column: the exponent of each input).",
         imp + "PolynomialFeatures\nresult = PolynomialFeatures(degree=2, include_bias=False).fit(X).powers_"),
        ("n_features_out_", f"Fit `SplineTransformer(n_knots={b + 2}, degree={d})` and return `n_features_out_` (it "
                            f"makes `n_knots + degree − 1` columns per input column).",
         imp + f"SplineTransformer\nresult = SplineTransformer(n_knots={b + 2}, degree={d}).fit(X).n_features_out_"),
    ]
    return catalog(rng, items, f"X = np.array({lit(rows)})", f"`X` is a {n} × 3 matrix.", tol=1e-6)


@template(M, "lite")
def output_feature_names(rng):
    d = rng.choice([2, 3])
    names = rng.choice([["a", "b"], ["x", "y"], ["age", "income"], ["temp", "wind"]])
    k = rng.randint(3, 6)
    b = rng.choice([2, 3])
    setup = f"X = np.array({lit([[round(rng.uniform(0, 9), 1) for _ in range(2)] for _ in range(6)])})"
    imp = f"from {PRE} import "
    items = [
        ("get_feature_names_out", f"Fit `PolynomialFeatures(degree={d}, include_bias=False)` on `X` and return its output "
                                  f"names for the inputs `{names}`.",
         imp + f"PolynomialFeatures\nresult = PolynomialFeatures(degree={d}, include_bias=False).fit(X)"
               f".get_feature_names_out({names})"),
        ("interaction_only", f"Fit `PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)` and return "
                             f"its output names for `{names}` (no squares, only products of different columns).",
         imp + "PolynomialFeatures\nresult = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)"
               f".fit(X).get_feature_names_out({names})"),
        ("PolynomialFeatures", f"Return the output names of `PolynomialFeatures(degree={d})` (bias included) for "
                               f"`{names}`.",
         imp + f"PolynomialFeatures\nresult = PolynomialFeatures(degree={d}).fit(X).get_feature_names_out({names})"),
        ("SplineTransformer", f"Fit `SplineTransformer(n_knots={k}, degree={d})` on `X` and return its output names.",
         imp + f"SplineTransformer\nresult = SplineTransformer(n_knots={k}, degree={d}).fit(X).get_feature_names_out()"),
        ("KBinsDiscretizer", f"Fit `KBinsDiscretizer(n_bins={b}, encode='onehot-dense', strategy='uniform')` and "
                             f"return its output names.",
         imp + f"KBinsDiscretizer\nresult = KBinsDiscretizer(n_bins={b}, encode='onehot-dense', strategy='uniform')"
               ".fit(X).get_feature_names_out()"),
        ("FunctionTransformer", f"`FunctionTransformer` only knows its output names when told how they relate to the "
                                f"inputs. Fit `FunctionTransformer(np.log1p, feature_names_out='one-to-one')` and return "
                                f"the names for `{names}`.",
         imp + "FunctionTransformer\nresult = FunctionTransformer(np.log1p, feature_names_out='one-to-one').fit(X)"
               f".get_feature_names_out({names})"),
        ("PolynomialFeatures", f"Fit `PolynomialFeatures(degree={d}, include_bias=False)` on a DataFrame with columns "
                               f"`{names}` (`pd.DataFrame(X, columns={names})`) and return the shape of the output.",
         imp + f"PolynomialFeatures\ndf = pd.DataFrame(X, columns={names})\n"
               f"result = PolynomialFeatures(degree={d}, include_bias=False).fit_transform(df).shape"),
    ]
    return catalog(rng, items, setup, "`X` is a 6 × 2 matrix.")


# ================================================================= easy

@template(M, "easy")
def transforms_by_hand(rng):
    rows = _outlier_matrix(rng)
    n = len(rows)
    t = round(rng.uniform(1, 5), 1)
    b = rng.choice([3, 4, 5])
    lo, hi = rng.choice([(0, 1), (-1, 1), (2, 5)])
    imp = f"from {PRE} import "
    items = [
        ("MinMaxScaler", f"`MinMaxScaler(feature_range=({lo}, {hi}))` computes `(X − min) / (max − min)` per column, "
                         f"then stretches it to [{lo}, {hi}]. Do it with NumPy and return `(by_hand, from_sklearn)`.",
         imp + f"MinMaxScaler\nZ = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))\n"
               f"result = (Z * ({hi} - {lo}) + {lo}, MinMaxScaler(feature_range=({lo}, {hi})).fit_transform(X))"),
        ("MaxAbsScaler", "`MaxAbsScaler` divides each column by its largest absolute value. Do it with NumPy and return "
                         "`(by_hand, from_sklearn)`.",
         imp + "MaxAbsScaler\nresult = (X / np.abs(X).max(axis=0), MaxAbsScaler().fit_transform(X))"),
        ("RobustScaler", "`RobustScaler` subtracts each column's median and divides by `np.percentile(X, 75, axis=0) − "
                         "np.percentile(X, 25, axis=0)`. Do it with NumPy and return `(by_hand, from_sklearn)`.",
         imp + "RobustScaler\niqr = np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)\n"
               "result = ((X - np.median(X, axis=0)) / iqr, RobustScaler().fit_transform(X))"),
        ("Normalizer", "`Normalizer()` (L2) divides each **row** by its Euclidean length. Do it with NumPy and return "
                       "`(by_hand, from_sklearn)`.",
         imp + "Normalizer\nresult = (X / np.linalg.norm(X, axis=1, keepdims=True), Normalizer().fit_transform(X))"),
        ("Normalizer", "`Normalizer(norm='l1')` divides each row by the sum of its absolute values. Do it with NumPy and "
                       "return `(by_hand, from_sklearn)`.",
         imp + "Normalizer\nresult = (X / np.abs(X).sum(axis=1, keepdims=True), Normalizer(norm='l1').fit_transform(X))"),
        ("Binarizer", f"`Binarizer(threshold={t})` maps values strictly above {t} to 1.0, the rest to 0.0. Do it with "
                      f"NumPy and return `(by_hand, from_sklearn)`.",
         imp + f"Binarizer\nresult = ((X > {t}).astype(float), Binarizer(threshold={t}).fit_transform(X))"),
        ("KBinsDiscretizer", f"`KBinsDiscretizer(n_bins={b}, strategy='uniform')` uses equal-width edges from each "
                             f"column's min to its max. Compute the {b + 1} edges of column 0 with `np.linspace` and "
                             f"return `(by_hand, bin_edges_[0])`.",
         imp + f"KBinsDiscretizer\nkb = KBinsDiscretizer(n_bins={b}, encode='ordinal', strategy='uniform').fit(X)\n"
               f"result = (np.linspace(X[:, 0].min(), X[:, 0].max(), {b + 1}), kb.bin_edges_[0])"),
        ("PolynomialFeatures", "`PolynomialFeatures(degree=2, include_bias=False)` on two columns `a, b` outputs `a, b, "
                               "a², a·b, b²`. Build that for `X[:, :2]` with NumPy and return `(by_hand, from_sklearn)`.",
         imp + "PolynomialFeatures\na, b = X[:, 0], X[:, 1]\nresult = (np.column_stack([a, b, a * a, a * b, b * b]), "
               "PolynomialFeatures(degree=2, include_bias=False).fit_transform(X[:, :2]))"),
        ("QuantileTransformer", f"For a column without ties, `QuantileTransformer(n_quantiles={n})` maps the training "
                                f"values to rank / (n − 1). Compute that for column 2 with `argsort` (ranks from 0) and "
                                f"return `(by_hand, from_sklearn)` (column 2 of the transformed `X`).",
         imp + f"QuantileTransformer\nr = np.empty({n})\nr[np.argsort(X[:, 2])] = np.arange({n})\n"
               f"result = (r / ({n} - 1), QuantileTransformer(n_quantiles={n}).fit_transform(X)[:, 2])"),
        ("StandardScaler", "Reproduce `StandardScaler` with NumPy (population standard deviation) and return "
                           "`(by_hand, from_sklearn)`.",
         imp + "StandardScaler\nresult = ((X - X.mean(axis=0)) / X.std(axis=0), StandardScaler().fit_transform(X))"),
    ]
    return catalog(rng, items, f"X = np.array({lit(rows)})",
                   f"`X` is a {n} × 3 matrix. Rebuild a transform with NumPy and check it against scikit-learn.", tol=1e-9)


@template(M, "easy")
def outlier_column(rng):
    n = rng.randint(8, 14)
    vals = [float(rng.randint(10, 60)) for _ in range(n - 1)] + [float(rng.choice([900, 2000, 5000]))]
    rng.shuffle(vals)
    setup = f"x = np.array({lit(vals)}).reshape(-1, 1)"
    head = f"`x` is one feature ({n} values as a column), with one huge outlier."
    task = rng.choice(["spread", "outlier", "squash", "both"])
    if task == "spread":
        return P(f"{head} Ignoring the outlier, how spread out are the other values after scaling? Return `(standard, "
                 f"robust)`: the range (max − min) of the non-outlier values after `StandardScaler` and after "
                 f"`RobustScaler`.", setup,
                 f"from {PRE} import RobustScaler, StandardScaler\nkeep = x[:, 0] < x.max()\n"
                 "s = StandardScaler().fit_transform(x)[keep, 0]\nr = RobustScaler().fit_transform(x)[keep, 0]\n"
                 "result = (s.max() - s.min(), r.max() - r.min())", require=["StandardScaler", "RobustScaler"])
    if task == "outlier":
        return P(f"{head} Return the scaled value of the outlier under `StandardScaler` and under `RobustScaler`, as a "
                 f"tuple. The robust one stays huge: the median and IQR ignore it.", setup,
                 f"from {PRE} import RobustScaler, StandardScaler\ni = x[:, 0].argmax()\n"
                 "result = (StandardScaler().fit_transform(x)[i, 0], RobustScaler().fit_transform(x)[i, 0])",
                 require=["StandardScaler", "RobustScaler"])
    if task == "squash":
        return P(f"{head} After `MinMaxScaler`, how many values land below 0.1? Return an int (the outlier squashes "
                 f"everything else near 0).", setup,
                 f"from {PRE} import MinMaxScaler\nresult = int((MinMaxScaler().fit_transform(x) < 0.1).sum())",
                 require=["MinMaxScaler"])
    return P(f"{head} Return the medians of the scaled column under `StandardScaler`, `RobustScaler` and "
             f"`MinMaxScaler`, as a list.", setup,
             f"from {PRE} import MinMaxScaler, RobustScaler, StandardScaler\n"
             "result = [float(np.median(s().fit_transform(x))) for s in (StandardScaler, RobustScaler, MinMaxScaler)]",
             require=["StandardScaler", "RobustScaler", "MinMaxScaler"])


@template(M, "easy")
def train_statistics_on_test(rng):
    n, s = rng.randint(30, 80), rng.randint(0, 99)
    k = rng.randint(4, 8)
    setup = (f"rng = np.random.default_rng({s})\nX_train = rng.normal(50, 10, ({n}, 2))\n"
             f"X_test = rng.normal(55, 15, ({k}, 2))")
    head = f"`X_train` ({n} × 2) and `X_test` ({k} × 2, a bit shifted). THE rule: fit on train, transform both."
    task = rng.choice(["minmax", "clip", "quantile", "bins", "both"])
    if task == "minmax":
        return P(f"{head} Fit a `MinMaxScaler` on `X_train` and return the transformed `X_test` (values outside [0, 1] "
                 f"are expected).", setup,
                 f"from {PRE} import MinMaxScaler\nresult = MinMaxScaler().fit(X_train).transform(X_test)",
                 require=["MinMaxScaler", "transform"])
    if task == "clip":
        return P(f"{head} `MinMaxScaler(clip=True)` clips test values into the feature range. Fit it on `X_train` and "
                 f"return the number of test values that ended exactly at 0 or 1.", setup,
                 f"from {PRE} import MinMaxScaler\nZ = MinMaxScaler(clip=True).fit(X_train).transform(X_test)\n"
                 "result = int(((Z == 0) | (Z == 1)).sum())", require=["MinMaxScaler", "clip"])
    if task == "quantile":
        return P(f"{head} Fit `QuantileTransformer(n_quantiles={n})` on `X_train` and return the transformed "
                 f"`X_test`: each value becomes its quantile in the training distribution.", setup,
                 f"from {PRE} import QuantileTransformer\nresult = QuantileTransformer(n_quantiles={n}).fit(X_train)"
                 ".transform(X_test)", require=["QuantileTransformer"])
    if task == "bins":
        return P(f"{head} Fit `KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform')` on `X_train` and "
                 f"return the bins of `X_test` (out-of-range values go to the first or last bin).", setup,
                 f"from {PRE} import KBinsDiscretizer\nresult = KBinsDiscretizer(n_bins=4, encode='ordinal', "
                 "strategy='uniform').fit(X_train).transform(X_test)", require=["KBinsDiscretizer"])
    return P(f"{head} Fit one `StandardScaler` on `X_train`, transform both sets and return `(mean of the scaled "
             f"train columns, mean of the scaled test columns)`.", setup,
             f"from {PRE} import StandardScaler\nsc = StandardScaler().fit(X_train)\n"
             "result = (sc.transform(X_train).mean(axis=0), sc.transform(X_test).mean(axis=0))",
             require=["StandardScaler"], tol=1e-9)


@template(M, "easy")
def polynomial_counts(rng):
    n, d = rng.randint(2, 6), rng.randint(2, 4)
    task = rng.choice(["formula", "names3", "interaction", "powers"])
    names = [f"x{i}" for i in range(n)]
    setup = f"X = np.ones((3, {n}))"
    if task == "formula":
        return P(f"`PolynomialFeatures(degree={d})` on {n} features makes one column per monomial of degree ≤ {d}: "
                 f"`comb({n} + {d}, {d})` of them (bias included). Return `(by_formula, from_sklearn)` using "
                 f"`math.comb` and `n_output_features_`.", setup,
                 f"import math\nfrom {PRE} import PolynomialFeatures\n"
                 f"result = (math.comb({n} + {d}, {d}), PolynomialFeatures(degree={d}).fit(X).n_output_features_)",
                 require=["PolynomialFeatures"])
    if task == "names3":
        a, b = rng.choice([("a", "b"), ("x", "y"), ("u", "v")])
        return P(f"Create degree-3 polynomial features of a 2-column DataFrame `df` (columns `{a}`, `{b}`) with "
                 f"`PolynomialFeatures(degree=3, include_bias=False)` and return them as a DataFrame whose columns are "
                 f"the readable output names.",
                 f"df = pd.DataFrame({{'{a}': [1.0, 2.0, 3.0], '{b}': [0.5, -1.0, 2.0]}})",
                 f"from {PRE} import PolynomialFeatures\npf = PolynomialFeatures(degree=3, include_bias=False)\n"
                 "result = pd.DataFrame(pf.fit_transform(df), columns=pf.get_feature_names_out())",
                 require=["PolynomialFeatures", "get_feature_names_out"])
    if task == "interaction":
        return P(f"With `interaction_only=True`, `PolynomialFeatures(degree={d}, include_bias=False)` keeps only products "
                 f"of **distinct** features: `Σ comb({n}, k)` for k = 1..{d}. Return `(by_formula, from_sklearn)`.",
                 setup, f"import math\nfrom {PRE} import PolynomialFeatures\n"
                        f"result = (sum(math.comb({n}, k) for k in range(1, {d} + 1)), PolynomialFeatures(degree={d}, "
                        "interaction_only=True, include_bias=False).fit(X).n_output_features_)",
                 require=["PolynomialFeatures", "interaction_only"])
    j = rng.randrange(n)
    return P(f"Fit `PolynomialFeatures(degree={d}, include_bias=False)` on `X` ({n} features) and return the output "
             f"names of the columns in which feature `x{j}` appears (its exponent in `powers_` is > 0).", setup,
             f"from {PRE} import PolynomialFeatures\npf = PolynomialFeatures(degree={d}, include_bias=False).fit(X)\n"
             f"result = pf.get_feature_names_out()[pf.powers_[:, {j}] > 0]", require=["powers_"])


@template(M, "easy")
def log_transform(rng):
    setup, m = _skewed(rng)
    task = rng.choice(["skew", "inverse", "custom", "names"])
    if task == "skew":
        return P(f"`X` ({m} × 2) holds positive, right-skewed values. Apply `FunctionTransformer(np.log1p)` and return "
                 f"the skewness of each column before and after (`pd.DataFrame(...).skew().to_numpy()`), as a tuple.",
                 setup, f"from {PRE} import FunctionTransformer\nZ = FunctionTransformer(np.log1p).fit_transform(X)\n"
                        "result = (pd.DataFrame(X).skew().to_numpy(), pd.DataFrame(Z).skew().to_numpy())",
                 require=["FunctionTransformer"])
    if task == "inverse":
        return P(f"`X` ({m} × 2) holds positive skewed values. Build `ft = FunctionTransformer(np.log1p, "
                 f"inverse_func=np.expm1)`, transform `X`, and return `ft.inverse_transform` of the first 3 transformed "
                 f"rows (it gives back `X[:3]`).", setup,
                 f"from {PRE} import FunctionTransformer\nft = FunctionTransformer(np.log1p, inverse_func=np.expm1)\n"
                 "Z = ft.fit_transform(X)\nresult = ft.inverse_transform(Z[:3])", require=["FunctionTransformer", "inverse_func"])
    if task == "custom":
        p = rng.choice([0, 1, 2])
        return P(f"`X` ({m} × 2) holds positive skewed values. `FunctionTransformer` wraps any function; `kw_args` "
                 f"passes extra keyword arguments to it. Apply `np.round` with `kw_args={{'decimals': {p}}}` and return "
                 f"the first 4 rows.",
                 setup, f"from {PRE} import FunctionTransformer\n"
                        f"result = FunctionTransformer(np.round, kw_args={{'decimals': {p}}}).fit_transform(X)[:4]",
                 require=["FunctionTransformer", "kw_args"])
    return P(f"`X` ({m} × 2) holds positive skewed values. Put it in a DataFrame with columns `price` and `wait`, "
             f"apply `FunctionTransformer(np.log1p, feature_names_out='one-to-one').set_output(transform='pandas')`, "
             f"and return the first 3 rows of the resulting DataFrame.", setup,
             f"from {PRE} import FunctionTransformer\ndf = pd.DataFrame(X, columns=['price', 'wait'])\n"
             "ft = FunctionTransformer(np.log1p, feature_names_out='one-to-one').set_output(transform='pandas')\n"
             "result = ft.fit_transform(df).head(3)", require=["FunctionTransformer", "set_output"])


@template(M, "easy")
def row_normalization(rng):
    n, d, s = rng.randint(5, 9), rng.randint(3, 5), rng.randint(0, 99)
    setup = f"rng = np.random.default_rng({s})\nX = rng.integers(0, 10, ({n}, {d})).astype(float) + 1"
    task = rng.choice(["norms", "cosine", "max", "nearest"])
    head = f"`X` ({n} × {d}) holds positive counts, one document per row. `Normalizer` scales each **row**."
    if task == "norms":
        return P(f"{head} Return the row lengths (`np.linalg.norm(..., axis=1)`) after `Normalizer()`: all 1.", setup,
                 f"from {PRE} import Normalizer\nresult = np.linalg.norm(Normalizer().fit_transform(X), axis=1)",
                 require=["Normalizer"])
    if task == "cosine":
        return P(f"{head} After L2-normalizing the rows, dot products are cosine similarities. Return the {n} × {n} "
                 f"cosine-similarity matrix `Z @ Z.T`.", setup,
                 f"from {PRE} import Normalizer\nZ = Normalizer().fit_transform(X)\nresult = Z @ Z.T",
                 require=["Normalizer"])
    if task == "max":
        return P(f"{head} Apply `Normalizer(norm='max')` and return the column index of the 1.0 in every row (the "
                 f"first one if several).", setup,
                 f"from {PRE} import Normalizer\nresult = Normalizer(norm='max').fit_transform(X).argmax(axis=1)",
                 require=["Normalizer"])
    return P(f"{head} Using cosine similarity (rows L2-normalized with `Normalizer`), which other row is most similar "
             f"to row 0? Return its index.", setup,
             f"from {PRE} import Normalizer\nZ = Normalizer().fit_transform(X)\nsim = Z @ Z[0]\nsim[0] = -np.inf\n"
             "result = int(sim.argmax())", require=["Normalizer"])


@template(M, "easy")
def binning_basics(rng):
    n, s = rng.randint(30, 80), rng.randint(0, 99)
    b = rng.choice([3, 4, 5])
    t = rng.choice([40, 50, 60])
    setup = f"rng = np.random.default_rng({s})\nx = rng.uniform(0, 100, ({n}, 1)).round(1)"
    task = rng.choice(["uniform", "quantile", "onehot", "binarize"])
    if task == "uniform":
        return P(f"`x` holds {n} values. Bin them into {b} equal-width bins (`KBinsDiscretizer(n_bins={b}, "
                 f"encode='ordinal', strategy='uniform')`) and return how many values fall in each bin.", setup,
                 f"from {PRE} import KBinsDiscretizer\nbins = KBinsDiscretizer(n_bins={b}, encode='ordinal', "
                 f"strategy='uniform').fit_transform(x)\nresult = np.bincount(bins[:, 0].astype(int), minlength={b})",
                 require=["KBinsDiscretizer"])
    if task == "quantile":
        return P(f"`x` holds {n} values. `strategy='quantile'` puts roughly the same number of values in each bin. "
                 f"Bin them with `KBinsDiscretizer(n_bins={b}, encode='ordinal', strategy='quantile', "
                 f"quantile_method='averaged_inverted_cdf')` and return the bin counts.", setup,
                 f"from {PRE} import KBinsDiscretizer\nbins = KBinsDiscretizer(n_bins={b}, encode='ordinal', "
                 f"strategy='quantile', quantile_method='averaged_inverted_cdf').fit_transform(x)\n"
                 f"result = np.bincount(bins[:, 0].astype(int), minlength={b})", require=["KBinsDiscretizer", "strategy"])
    if task == "onehot":
        return P(f"`x` holds {n} values. `encode='onehot-dense'` returns one 0/1 column per bin. Bin `x` with "
                 f"`KBinsDiscretizer(n_bins={b}, encode='onehot-dense', strategy='uniform')` and return the column sums.",
                 setup, f"from {PRE} import KBinsDiscretizer\nresult = KBinsDiscretizer(n_bins={b}, encode='onehot-dense', "
                        "strategy='uniform').fit_transform(x).sum(axis=0)", require=["KBinsDiscretizer", "encode"])
    return P(f"`x` holds {n} values. Return how many are strictly above {t} using `Binarizer(threshold={t})`.", setup,
             f"from {PRE} import Binarizer\nresult = int(Binarizer(threshold={t}).fit_transform(x).sum())",
             require=["Binarizer"])


# ================================================================= medium

@template(M, "medium")
def table_transform_tasks(rng):
    t = table(rng, numeric_only=True, n=(80, 200))
    col = rng.choice(t.num)
    b = rng.choice([3, 4, 5])
    new = {c: 1 for c in t.num}
    items = [
        ("RobustScaler", "Fit a `RobustScaler` on `df` and return the name of the column with the largest interquartile "
                         "range (`scale_`).",
         f"from {PRE} import RobustScaler\nresult = df.columns[RobustScaler().fit(df).scale_.argmax()]"),
        ("MinMaxScaler", "Fit a `MinMaxScaler` on `df` and transform `new_row` (a dict; make a one-row DataFrame with "
                         "`df`'s columns). Return the transformed row as a 1-D array.",
         f"from {PRE} import MinMaxScaler\nmm = MinMaxScaler().fit(df)\nresult = mm.transform(pd.DataFrame([new_row])[df.columns])[0]"),
        ("QuantileTransformer", f"Transform `{col}` to a standard normal shape with `QuantileTransformer(n_quantiles=50, "
                                f"output_distribution='normal')` and return `(mean, std)` of the result.",
         f"from {PRE} import QuantileTransformer\nz = QuantileTransformer(n_quantiles=50, output_distribution='normal')"
         f".fit_transform(df[[{col!r}]])\nresult = (z.mean(), z.std())", {"require": ["QuantileTransformer", "output_distribution"]}),
        ("PowerTransformer", "Fit a `PowerTransformer()` (Yeo-Johnson) on `df` and return a dict column → fitted lambda.",
         f"from {PRE} import PowerTransformer\npt = PowerTransformer().fit(df)\nresult = dict(zip(df.columns, pt.lambdas_))",
         {"tol": 1e-4}),
        ("KBinsDiscretizer", f"Bin every column into {b} equal-width bins (`KBinsDiscretizer(n_bins={b}, encode='ordinal', "
                             f"strategy='uniform')`) and return, per column, how many rows fall in the **last** bin, as "
                             f"a dict.",
         f"from {PRE} import KBinsDiscretizer\nB = KBinsDiscretizer(n_bins={b}, encode='ordinal', strategy='uniform')"
         f".fit_transform(df)\nresult = {{c: int((B[:, j] == {b - 1}).sum()) for j, c in enumerate(df.columns)}}"),
        ("set_output", "Standardize `df` with `StandardScaler().set_output(transform='pandas')` and return the rows "
                       "where any value is beyond ±2.5 (a DataFrame).",
         f"from {PRE} import StandardScaler\nZ = StandardScaler().set_output(transform='pandas').fit_transform(df)\n"
         "result = Z[(Z.abs() > 2.5).any(axis=1)]", {"require": ["set_output"]}),
        ("MaxAbsScaler", "Fit a `MaxAbsScaler` on `df` and return each column's largest scaled value (they are all 1 "
                         "when the columns are positive), as a Series indexed by column (`set_output` or wrap it).",
         f"from {PRE} import MaxAbsScaler\nZ = MaxAbsScaler().set_output(transform='pandas').fit_transform(df)\nresult = Z.max()"),
        ("FunctionTransformer", f"Log-transform `{col}` with `FunctionTransformer(np.log1p)` (values are ≥ 0) and "
                                f"return the ratio of its standard deviation after / before.",
         f"from {PRE} import FunctionTransformer\nz = FunctionTransformer(np.log1p).fit_transform(df[[{col!r}]].clip(lower=0))\n"
         f"result = float(z.to_numpy().std() / df[{col!r}].clip(lower=0).std(ddof=0))"),
        ("RobustScaler", "Return a dict column → (median, IQR) learned by `RobustScaler(quantile_range=(10, 90))` "
                         "(the 'IQR' is then the 10–90 range).",
         f"from {PRE} import RobustScaler\nrs = RobustScaler(quantile_range=(10, 90)).fit(df)\n"
         "result = {c: (m, s) for c, m, s in zip(df.columns, rs.center_, rs.scale_)}"),
    ]
    return catalog(rng, items, t.setup + f"\nnew_row = {lit(new)}", f"{t.desc}. `new_row` is a dict of feature values.",
                   tol=1e-6)


@template(M, "medium")
def scaling_for_distances(rng):
    n, s = rng.randint(150, 300), rng.randint(0, 99)
    big = rng.choice([1000, 5000, 20000])
    k = rng.choice([3, 5, 9])
    setup = (f"rng = np.random.default_rng({s})\nX = rng.normal(size=({n}, 3))\ny = (X[:, 0] + X[:, 1] > 0).astype(int)\n"
             f"X[:, 2] = rng.normal(0, {big}, {n})   # an irrelevant feature on a huge scale\n"
             f"X_tr, X_te, y_tr, y_te = X[:{n // 2}], X[{n // 2}:], y[:{n // 2}], y[{n // 2}:]")
    head = (f"`X` has two useful features and an irrelevant third one with values in the thousands; `X_tr, y_tr` / "
            f"`X_te, y_te` are halves. Model: `KNeighborsClassifier(n_neighbors={k})` (distances!).")
    knn = f"from sklearn.neighbors import KNeighborsClassifier\nknn = KNeighborsClassifier(n_neighbors={k})\n"
    task = rng.choice(["raw", "standard", "minmax", "all3", "drop"])
    if task == "raw":
        return P(f"{head} Return its test accuracy on the raw features.", setup,
                 knn + "result = knn.fit(X_tr, y_tr).score(X_te, y_te)", require=["KNeighborsClassifier"])
    if task == "standard":
        return P(f"{head} Standardize with a `StandardScaler` fitted on `X_tr`, then return the test accuracy.", setup,
                 knn + f"from {PRE} import StandardScaler\nsc = StandardScaler().fit(X_tr)\n"
                       "result = knn.fit(sc.transform(X_tr), y_tr).score(sc.transform(X_te), y_te)",
                 require=["StandardScaler"])
    if task == "minmax":
        return P(f"{head} Rescale with a `MinMaxScaler` fitted on `X_tr`, then return the test accuracy.", setup,
                 knn + f"from {PRE} import MinMaxScaler\nsc = MinMaxScaler().fit(X_tr)\n"
                       "result = knn.fit(sc.transform(X_tr), y_tr).score(sc.transform(X_te), y_te)",
                 require=["MinMaxScaler"])
    if task == "all3":
        return P(f"{head} Return a dict mapping `'none'`, `'standard'` and `'robust'` to the test accuracy with no "
                 f"scaling, `StandardScaler` and `RobustScaler` (scalers fitted on `X_tr`).", setup,
                 knn + f"from {PRE} import RobustScaler, StandardScaler\nresult = {{'none': knn.fit(X_tr, y_tr).score(X_te, y_te)}}\n"
                       "for name, S in (('standard', StandardScaler), ('robust', RobustScaler)):\n    sc = S().fit(X_tr)\n"
                       "    result[name] = knn.fit(sc.transform(X_tr), y_tr).score(sc.transform(X_te), y_te)",
                 require=["StandardScaler", "RobustScaler"])
    return P(f"{head} Standardize (fitted on `X_tr`) and compare with dropping the irrelevant column (also "
             f"standardized). Return `(standardized_all, standardized_without_col2)` test accuracies.", setup,
             knn + f"from {PRE} import StandardScaler\nsc = StandardScaler().fit(X_tr)\n"
                   "a = knn.fit(sc.transform(X_tr), y_tr).score(sc.transform(X_te), y_te)\n"
                   "sc2 = StandardScaler().fit(X_tr[:, :2])\nb = knn.fit(sc2.transform(X_tr[:, :2]), y_tr).score(sc2.transform(X_te[:, :2]), y_te)\n"
                   "result = (a, b)", require=["StandardScaler"])


@template(M, "medium")
def polynomial_regression(rng):
    n, s = rng.randint(40, 100), rng.randint(0, 99)
    degs = sorted(rng.sample([1, 2, 3, 4, 5, 6, 8], 4))
    d = rng.choice([2, 3, 4])
    setup = (f"rng = np.random.default_rng({s})\nx = rng.uniform(-3, 3, ({n}, 1))\n"
             f"y = 0.5 * x[:, 0] ** 3 - x[:, 0] ** 2 + rng.normal(0, 2, {n})\n"
             f"x_tr, x_te, y_tr, y_te = x[::2], x[1::2], y[::2], y[1::2]")
    head = f"`x` ({n} × 1) and `y` (a noisy cubic); even rows are training (`x_tr, y_tr`), odd rows test (`x_te, y_te`)."
    fit = (f"from {PRE} import PolynomialFeatures\nfrom sklearn.linear_model import LinearRegression\n")
    task = rng.choice(["curve", "best", "coef", "gap"])
    if task == "curve":
        return P(f"{head} For each degree in `{degs}`, expand `x` with `PolynomialFeatures(degree=d, include_bias=False)` "
                 f"(fitted on `x_tr`), fit `LinearRegression` and record the test R². Return the list.", setup,
                 fit + "result = []\n"
                       f"for d in {degs}:\n    pf = PolynomialFeatures(degree=d, include_bias=False).fit(x_tr)\n"
                       "    result.append(LinearRegression().fit(pf.transform(x_tr), y_tr).score(pf.transform(x_te), y_te))",
                 require=["PolynomialFeatures"], tol=1e-6)
    if task == "best":
        return P(f"{head} Which degree in `{degs}` gives the best test R² (polynomial features without bias + "
                 f"`LinearRegression`)? Return it.", setup,
                 fit + f"r2 = []\nfor d in {degs}:\n    pf = PolynomialFeatures(degree=d, include_bias=False).fit(x_tr)\n"
                       "    r2.append(LinearRegression().fit(pf.transform(x_tr), y_tr).score(pf.transform(x_te), y_te))\n"
                       f"result = {degs}[int(np.argmax(r2))]", require=["PolynomialFeatures"])
    if task == "coef":
        return P(f"{head} Fit `LinearRegression` on `PolynomialFeatures(degree={d}, include_bias=False)` of `x_tr` and "
                 f"return a dict output name → coefficient.", setup,
                 fit + f"pf = PolynomialFeatures(degree={d}, include_bias=False).fit(x_tr)\n"
                       "reg = LinearRegression().fit(pf.transform(x_tr), y_tr)\nresult = dict(zip(pf.get_feature_names_out(), reg.coef_))",
                 require=["PolynomialFeatures", "get_feature_names_out"], tol=1e-6)
    return P(f"{head} Return `(train R², test R²)` with degree-{max(degs) + 4} polynomial features (no bias) and "
             f"`LinearRegression`: a high degree overfits.", setup,
             fit + f"pf = PolynomialFeatures(degree={max(degs) + 4}, include_bias=False).fit(x_tr)\n"
                   "reg = LinearRegression().fit(pf.transform(x_tr), y_tr)\n"
                   "result = (reg.score(pf.transform(x_tr), y_tr), reg.score(pf.transform(x_te), y_te))",
             require=["PolynomialFeatures"], tol=1e-4)


@template(M, "medium")
def spline_regression(rng):
    n, s = rng.randint(60, 150), rng.randint(0, 99)
    k, d = rng.randint(4, 9), rng.choice([2, 3])
    setup = (f"rng = np.random.default_rng({s})\nx = np.sort(rng.uniform(0, 10, {n})).reshape(-1, 1)\n"
             f"y = np.sin(x[:, 0]) + 0.1 * x[:, 0] + rng.normal(0, 0.2, {n})")
    head = f"`x` ({n} sorted values in [0, 10], as a column) and `y` (a noisy wave)."
    fit = f"from {PRE} import SplineTransformer\nfrom sklearn.linear_model import LinearRegression\n"
    task = rng.choice(["r2", "partition", "vs_poly", "extrapolate", "nonzero"])
    if task == "r2":
        return P(f"{head} Expand `x` with `SplineTransformer(n_knots={k}, degree={d})`, fit `LinearRegression` and "
                 f"return the training R².", setup,
                 fit + f"S = SplineTransformer(n_knots={k}, degree={d}).fit_transform(x)\n"
                       "result = LinearRegression().fit(S, y).score(S, y)", require=["SplineTransformer"], tol=1e-6)
    if task == "partition":
        return P(f"{head} B-spline features form a partition of unity: return the row sums of "
                 f"`SplineTransformer(n_knots={k}, degree={d}).fit_transform(x)` for the first 5 rows (all 1).", setup,
                 f"from {PRE} import SplineTransformer\nresult = SplineTransformer(n_knots={k}, degree={d})"
                 ".fit_transform(x)[:5].sum(axis=1)", require=["SplineTransformer"])
    if task == "vs_poly":
        return P(f"{head} Return `(spline R², polynomial R²)` on the training data: `SplineTransformer(n_knots={k}, "
                 f"degree=3)` vs `PolynomialFeatures(degree=3)`, each followed by `LinearRegression`.", setup,
                 fit + f"from {PRE} import PolynomialFeatures\nS = SplineTransformer(n_knots={k}, degree=3).fit_transform(x)\n"
                       "Q = PolynomialFeatures(degree=3).fit_transform(x)\n"
                       "result = (LinearRegression().fit(S, y).score(S, y), LinearRegression().fit(Q, y).score(Q, y))",
                 require=["SplineTransformer", "PolynomialFeatures"], tol=1e-6)
    if task == "extrapolate":
        return P(f"{head} Outside the training range, `extrapolation='constant'` (the default) freezes the features "
                 f"while `'linear'` continues them. Fit `SplineTransformer(n_knots={k}, degree={d}, extrapolation=...)` "
                 f"+ `LinearRegression` for both settings and return their predictions at x = 12, as a tuple.", setup,
                 fit + "out = []\nfor ex in ('constant', 'linear'):\n"
                       f"    st = SplineTransformer(n_knots={k}, degree={d}, extrapolation=ex).fit(x)\n"
                       "    reg = LinearRegression().fit(st.transform(x), y)\n"
                       "    out.append(float(reg.predict(st.transform([[12.0]]))[0]))\nresult = tuple(out)",
                 require=["SplineTransformer", "extrapolation"], tol=1e-5)
    return P(f"{head} Each row of `SplineTransformer(n_knots={k}, degree={d})` has at most `degree + 1` non-zero "
             f"features. Return the number of non-zero entries in each of the first 6 rows.", setup,
             f"from {PRE} import SplineTransformer\nS = SplineTransformer(n_knots={k}, degree={d}).fit_transform(x)\n"
             "result = (np.abs(S[:6]) > 1e-12).sum(axis=1)", require=["SplineTransformer"])


@template(M, "medium")
def bins_as_features(rng):
    n, s = rng.randint(100, 250), rng.randint(0, 99)
    bins = sorted(rng.sample([2, 3, 4, 6, 8, 12], 3))
    b = rng.choice([4, 6, 8])
    setup = (f"rng = np.random.default_rng({s})\nx = rng.uniform(-3, 3, ({n}, 1))\n"
             f"y = np.where(x[:, 0] > 0, 2.0, -1.0) + np.cos(2 * x[:, 0]) + rng.normal(0, 0.3, {n})")
    head = f"`x` ({n} × 1) and `y` (a step plus a wiggle): a straight line fits it badly."
    fit = f"from {PRE} import KBinsDiscretizer\nfrom sklearn.linear_model import LinearRegression\n"
    task = rng.choice(["one", "sweep", "vs_linear", "means"])
    if task == "one":
        return P(f"{head} One-hot encode `x` into {b} equal-width bins (`KBinsDiscretizer(n_bins={b}, "
                 f"encode='onehot-dense', strategy='uniform')`), fit `LinearRegression` and return the training R².",
                 setup, fit + f"B = KBinsDiscretizer(n_bins={b}, encode='onehot-dense', strategy='uniform').fit_transform(x)\n"
                              "result = LinearRegression().fit(B, y).score(B, y)", require=["KBinsDiscretizer"])
    if task == "sweep":
        return P(f"{head} For `n_bins` in `{bins}`, one-hot bin `x` (uniform strategy), fit `LinearRegression` and "
                 f"record the training R². Return the list.", setup,
                 fit + f"result = []\nfor b in {bins}:\n    B = KBinsDiscretizer(n_bins=b, encode='onehot-dense', "
                       "strategy='uniform').fit_transform(x)\n    result.append(LinearRegression().fit(B, y).score(B, y))",
                 require=["KBinsDiscretizer"])
    if task == "vs_linear":
        return P(f"{head} Return `(linear R², binned R²)`: `LinearRegression` on raw `x` vs on {b} one-hot uniform bins.",
                 setup, fit + f"B = KBinsDiscretizer(n_bins={b}, encode='onehot-dense', strategy='uniform').fit_transform(x)\n"
                              "result = (LinearRegression().fit(x, y).score(x, y), LinearRegression().fit(B, y).score(B, y))",
                 require=["KBinsDiscretizer"])
    return P(f"{head} With one-hot bins, linear regression predicts each bin's average. Bin `x` into {b} ordinal "
             f"uniform bins and return the mean `y` of each bin, as an array in bin order.", setup,
             f"from {PRE} import KBinsDiscretizer\nB = KBinsDiscretizer(n_bins={b}, encode='ordinal', strategy='uniform')"
             f".fit_transform(x)[:, 0]\nresult = np.array([y[B == k].mean() for k in range({b})])",
             require=["KBinsDiscretizer"])


@template(M, "medium")
def gaussianize_skewed(rng):
    setup, m = _skewed(rng)
    task = rng.choice(["yj", "boxcox", "quantile", "standardize", "lambdas"])
    skew = "pd.DataFrame(Z).skew().to_numpy()"
    if task == "yj":
        return P(f"`X` ({m} × 2) holds right-skewed positive values. Apply `PowerTransformer()` (Yeo-Johnson) and "
                 f"return the skewness of the result per column (pandas `skew`).", setup,
                 f"from {PRE} import PowerTransformer\nZ = PowerTransformer().fit_transform(X)\nresult = {skew}",
                 require=["PowerTransformer"], tol=1e-4)
    if task == "boxcox":
        return P(f"`X` ({m} × 2) holds right-skewed positive values. Apply `PowerTransformer(method='box-cox')` and "
                 f"return the skewness per column, before and after, as a tuple.", setup,
                 f"from {PRE} import PowerTransformer\nZ = PowerTransformer(method='box-cox').fit_transform(X)\n"
                 f"result = (pd.DataFrame(X).skew().to_numpy(), {skew})", require=["PowerTransformer", "method"], tol=1e-4)
    if task == "quantile":
        return P(f"`X` ({m} × 2) holds right-skewed values. Force a normal shape with `QuantileTransformer("
                 f"n_quantiles={min(m, 100)}, output_distribution='normal')` and return the column medians and "
                 f"standard deviations as a tuple.", setup,
                 f"from {PRE} import QuantileTransformer\nZ = QuantileTransformer(n_quantiles={min(m, 100)}, "
                 "output_distribution='normal').fit_transform(X)\nresult = (np.median(Z, axis=0), Z.std(axis=0))",
                 require=["QuantileTransformer", "output_distribution"], tol=1e-6)
    if task == "standardize":
        return P(f"`X` ({m} × 2) is skewed. `PowerTransformer(standardize=False)` skips the final standardization. "
                 f"Return the column means of `PowerTransformer().fit_transform(X)` and of the `standardize=False` "
                 f"version, as a tuple.", setup,
                 f"from {PRE} import PowerTransformer\nresult = (PowerTransformer().fit_transform(X).mean(axis=0), "
                 "PowerTransformer(standardize=False).fit_transform(X).mean(axis=0))", require=["PowerTransformer"], tol=1e-4)
    return P(f"`X` ({m} × 2) is skewed. A lambda near 0 means 'take the log'. Fit `PowerTransformer(method='box-cox')` "
             f"and return the index of the column whose lambda is closest to 0.", setup,
             f"from {PRE} import PowerTransformer\nresult = int(np.abs(PowerTransformer(method='box-cox').fit(X).lambdas_).argmin())",
             require=["lambdas_"])


@template(M, "medium")
def interaction_terms(rng):
    t = table(rng, kind="reg", numeric_only=True, n=(100, 250))
    task = rng.choice(["top", "gain", "count", "scaled_first"])
    pf = f"from {PRE} import PolynomialFeatures\nfrom sklearn.linear_model import LinearRegression\n"
    if task == "top":
        return P(f"{t.desc}. Standardize `df` (`StandardScaler`), add the pairwise products with "
                 f"`PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)`, fit `LinearRegression` "
                 f"and return the output name of the **interaction** term with the largest |coefficient|.", t.setup,
                 pf + f"from {PRE} import StandardScaler\nZ = StandardScaler().fit_transform(df)\n"
                      "p = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False).fit(Z)\n"
                      "coef = LinearRegression().fit(p.transform(Z), y).coef_\nnames = p.get_feature_names_out(df.columns)\n"
                      "inter = p.powers_.sum(axis=1) == 2\nresult = names[inter][np.abs(coef[inter]).argmax()]",
                 require=["PolynomialFeatures", "interaction_only"])
    if task == "gain":
        return P(f"{t.desc}. Return the gain in training R² from adding all degree-2 terms "
                 f"(`PolynomialFeatures(degree=2, include_bias=False)`) to a plain `LinearRegression` on `df`.", t.setup,
                 pf + "Q = PolynomialFeatures(degree=2, include_bias=False).fit_transform(df)\n"
                      "result = LinearRegression().fit(Q, y).score(Q, y) - LinearRegression().fit(df, y).score(df, y)",
                 require=["PolynomialFeatures"], tol=1e-6)
    if task == "count":
        return P(f"{t.desc}. Return `(columns with interaction_only=True, columns with all squares too)` for degree-2 "
                 f"`PolynomialFeatures(include_bias=False)` on `df`.", t.setup,
                 pf + "a = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False).fit_transform(df).shape[1]\n"
                      "b = PolynomialFeatures(degree=2, include_bias=False).fit_transform(df).shape[1]\nresult = (a, b)",
                 require=["PolynomialFeatures"])
    return P(f"{t.desc}. Products of raw columns have wildly different scales. Return the standard deviation of every "
             f"output column of `PolynomialFeatures(degree=2, include_bias=False)` on raw `df`, as a Series indexed by "
             f"output name (`set_output(transform='pandas')` helps).", t.setup,
             f"from {PRE} import PolynomialFeatures\nQ = PolynomialFeatures(degree=2, include_bias=False)"
             ".set_output(transform='pandas').fit_transform(df)\nresult = Q.std()", require=["PolynomialFeatures"], tol=1e-6)


@template(M, "medium")
def transformer_comparisons(rng):
    setup, m = _skewed(rng, n=(60, 150))
    b = rng.choice([3, 4, 5])
    k = rng.choice([4, 5, 6])
    parts = rng.randint(3, 5)
    imp = f"from {PRE} import "
    items = [
        ("MinMaxScaler", "Return a dict mapping `'standard'`, `'minmax'`, `'robust'`, `'maxabs'` to `(min, max)` of "
                         "column 0 after `StandardScaler`, `MinMaxScaler`, `RobustScaler`, `MaxAbsScaler`.",
         imp + "MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler\n"
               "out = {'standard': StandardScaler(), 'minmax': MinMaxScaler(), 'robust': RobustScaler(), 'maxabs': MaxAbsScaler()}\n"
               "result = {}\nfor name, t in out.items():\n    z = t.fit_transform(X)[:, 0]\n    result[name] = (z.min(), z.max())"),
        ("KBinsDiscretizer", f"Bin column 0 into {b} bins with `strategy='uniform'`, `'quantile'` (with "
                             f"`quantile_method='averaged_inverted_cdf'`) and `'kmeans'` (`KBinsDiscretizer(n_bins={b}, "
                             f"encode='ordinal', ...)`) and return a dict strategy → bin counts.",
         imp + "KBinsDiscretizer\nresult = {}\nfor st in ('uniform', 'quantile', 'kmeans'):\n"
               "    kw = {'quantile_method': 'averaged_inverted_cdf'} if st == 'quantile' else {}\n"
               f"    kb = KBinsDiscretizer(n_bins={b}, encode='ordinal', strategy=st, **kw)\n"
               f"    result[st] = np.bincount(kb.fit_transform(X[:, :1])[:, 0].astype(int), minlength={b})"),
        ("QuantileTransformer", f"`QuantileTransformer` only reorders by rank: return whether the row order of column 0 "
                                f"is the same before and after `QuantileTransformer(n_quantiles={m}, "
                                f"output_distribution='normal')` (compare `np.argsort`), and the transformed column's "
                                f"max, as a tuple.",
         imp + f"QuantileTransformer\nz = QuantileTransformer(n_quantiles={m}, output_distribution='normal').fit_transform(X)[:, 0]\n"
               "result = (bool(np.array_equal(np.argsort(X[:, 0]), np.argsort(z))), z.max())"),
        ("PowerTransformer", "Return the correlation (`np.corrcoef`) between the Yeo-Johnson transform of column 0 "
                             "(`PowerTransformer()`) and `np.log1p` of it.",
         imp + "PowerTransformer\nz = PowerTransformer().fit_transform(X)[:, 0]\nresult = np.corrcoef(z, np.log1p(X[:, 0]))[0, 1]",
         {"tol": 1e-5}),
        ("SplineTransformer", f"With `knots='quantile'` the spline knots follow the data instead of being evenly spaced. "
                              f"Fit `SplineTransformer(n_knots={k}, degree=3, knots=...)` on column 0 for `'uniform'` and "
                              f"`'quantile'` and return the {k} base knots of each (`bsplines_[0].t[3:-3]`), as a tuple.",
         imp + f"SplineTransformer\nresult = tuple(SplineTransformer(n_knots={k}, degree=3, knots=kn).fit(X[:, :1])"
               ".bsplines_[0].t[3:-3] for kn in ('uniform', 'quantile'))"),
        ("partial_fit", f"Data often arrives in batches. Feed `X` to a `MinMaxScaler` in {parts} chunks "
                        f"(`np.array_split(X, {parts})`) with `partial_fit` and return `(data_min_, data_max_)`: the same "
                        f"as one `fit` on all rows.",
         imp + f"MinMaxScaler\nmm = MinMaxScaler()\nfor chunk in np.array_split(X, {parts}):\n    mm.partial_fit(chunk)\n"
               "result = (mm.data_min_, mm.data_max_)"),
        ("partial_fit", f"Feed `X` to a `StandardScaler` in {parts} chunks with `partial_fit` and return `(mean_, "
                        f"scale_, n_samples_seen_)`.",
         imp + f"StandardScaler\nsc = StandardScaler()\nfor chunk in np.array_split(X, {parts}):\n    sc.partial_fit(chunk)\n"
               "result = (sc.mean_, sc.scale_, int(sc.n_samples_seen_))"),
        ("RobustScaler", "Compare centring choices: return `(StandardScaler().fit(X).mean_, RobustScaler().fit(X).center_)`:"
                         " for skewed columns the mean sits well above the median.",
         imp + "RobustScaler, StandardScaler\nresult = (StandardScaler().fit(X).mean_, RobustScaler().fit(X).center_)"),
    ]
    return catalog(rng, items, setup, f"`X` ({m} × 2) holds positive, right-skewed values.", tol=1e-6)


# ================================================================= hard

@template(M, "hard")
def leaky_scaling(rng):
    d = synth_clf(rng, n=(150, 300), d=(4, 8), classes=2)
    s, k = rng.randint(0, 99), rng.choice([3, 5, 9])
    split = (f"from sklearn.model_selection import train_test_split\nX_tr, X_te, y_tr, y_te = train_test_split(X, y, "
             f"test_size=0.3, random_state={s})\n")
    task = rng.choice(["fixed", "difference", "same_test", "minmax_range"])
    leak = f"```\nXs = StandardScaler().fit_transform(X)\nX_tr, X_te, y_tr, y_te = train_test_split(Xs, y, test_size=0.3, random_state={s})\n```"
    if task == "fixed":
        return P(f"{d.desc}. This preprocessing leaks test statistics into training:\n\n{leak}\n\nFix it (split first, "
                 f"fit the scaler on the training part only), fit `KNeighborsClassifier(n_neighbors={k})` and return the "
                 f"test accuracy.", d.setup,
                 split + f"from {PRE} import StandardScaler\nfrom sklearn.neighbors import KNeighborsClassifier\n"
                         f"sc = StandardScaler().fit(X_tr)\nresult = KNeighborsClassifier(n_neighbors={k}).fit(sc.transform(X_tr), "
                         "y_tr).score(sc.transform(X_te), y_te)", require=["StandardScaler", "train_test_split"])
    if task == "difference":
        return P(f"{d.desc}. Compare the leaky version\n\n{leak}\n\nwith the correct one (scaler fitted on the training "
                 f"rows). Return the largest absolute difference between the two scaled test sets.", d.setup,
                 split + f"from {PRE} import StandardScaler\nleaky = StandardScaler().fit_transform(X)\n"
                         f"_, leaky_te = train_test_split(leaky, test_size=0.3, random_state={s})\n"
                         "good_te = StandardScaler().fit(X_tr).transform(X_te)\nresult = np.abs(leaky_te - good_te).max()",
                 require=["StandardScaler"], tol=1e-9)
    if task == "same_test":
        return P(f"{d.desc}. Split with `train_test_split(X, y, test_size=0.3, random_state={s})`, fit a "
                 f"`StandardScaler` on `X_tr` and return `(train column means, test column means)` after scaling: only "
                 f"the training means are exactly 0.", d.setup,
                 split + f"from {PRE} import StandardScaler\nsc = StandardScaler().fit(X_tr)\n"
                         "result = (sc.transform(X_tr).mean(axis=0), sc.transform(X_te).mean(axis=0))",
                 require=["StandardScaler"], tol=1e-9)
    return P(f"{d.desc}. Split with `train_test_split(X, y, test_size=0.3, random_state={s})` and fit a `MinMaxScaler` "
             f"on `X_tr`. Return `(min, max)` of the scaled **test** set (not exactly 0 and 1).", d.setup,
             split + f"from {PRE} import MinMaxScaler\nZ = MinMaxScaler().fit(X_tr).transform(X_te)\nresult = (Z.min(), Z.max())",
             require=["MinMaxScaler"])


@template(M, "hard")
def power_transform_by_hand(rng):
    setup, m = _skewed(rng, n=(40, 100))
    task = rng.choice(["boxcox", "yeojohnson", "standardized"])
    if task == "boxcox":
        return P(f"`X` ({m} × 2) holds positive skewed values. Fit `pt = PowerTransformer(method='box-cox', "
                 f"standardize=False)` and reproduce `pt.transform(X)` **by hand** from `pt.lambdas_`: Box-Cox is "
                 f"`(x^λ − 1) / λ` (or `log x` when λ = 0).", setup,
                 f"from {PRE} import PowerTransformer\npt = PowerTransformer(method='box-cox', standardize=False).fit(X)\n"
                 "lam = pt.lambdas_\nresult = np.where(lam == 0, np.log(X), (X ** lam - 1) / np.where(lam == 0, 1, lam))",
                 require=["lambdas_"], tol=1e-6)
    if task == "yeojohnson":
        return P(f"`X` ({m} × 2) holds skewed values; shift it by −10 first (`X − 10`, so some values are negative). "
                 f"Fit `pt = PowerTransformer(standardize=False)` (Yeo-Johnson) on the shifted data and reproduce its "
                 f"transform by hand from `lambdas_`: for x ≥ 0, `((x + 1)^λ − 1) / λ`; for x < 0, `−((1 − x)^(2 − λ) − "
                 f"1) / (2 − λ)` (λ ≠ 0, 2 here).", setup,
                 f"from {PRE} import PowerTransformer\nZ = X - 10\npt = PowerTransformer(standardize=False).fit(Z)\n"
                 "lam = pt.lambdas_\npos = ((np.maximum(Z, 0) + 1) ** lam - 1) / lam\n"
                 "neg = -((1 - np.minimum(Z, 0)) ** (2 - lam) - 1) / (2 - lam)\n"
                 "result = np.where(Z >= 0, pos, neg)", require=["lambdas_"], tol=1e-6)
    return P(f"`X` ({m} × 2) holds positive skewed values. `PowerTransformer(method='box-cox')` (standardized) is the "
             f"Box-Cox transform followed by standardization. Reproduce it: take `lambdas_` from the fitted transformer, "
             f"apply `(x^λ − 1) / λ` per column, then standardize with NumPy (population std). Return the result.",
             setup,
             f"from {PRE} import PowerTransformer\nlam = PowerTransformer(method='box-cox').fit(X).lambdas_\n"
             "B = (X ** lam - 1) / lam\nresult = (B - B.mean(axis=0)) / B.std(axis=0)", require=["lambdas_"], tol=1e-6)


@template(M, "hard")
def quantile_binning_by_hand(rng):
    n, s = rng.randint(20, 60), rng.randint(0, 99)
    b = rng.choice([3, 4, 5])
    nq = rng.choice([5, 9, 11])
    setup = f"rng = np.random.default_rng({s})\nx = rng.gamma(2.0, 3.0, ({n}, 1))\nnew = rng.uniform(0, 15, (6, 1))"
    task = rng.choice(["edges", "bins", "quantile_interp", "uniform_bins"])
    if task == "edges":
        return P(f"`x` ({n} × 1) is skewed. `KBinsDiscretizer(n_bins={b}, strategy='quantile', quantile_method='linear')` "
                 f"puts its edges at the 0, 100/{b}, ..., 100 percentiles. Return the edges computed with "
                 f"`np.percentile` and check them against `bin_edges_[0]`.", setup,
                 f"result = np.percentile(x[:, 0], np.linspace(0, 100, {b + 1}))", require=["percentile"], tol=1e-9)
    if task == "bins":
        return P(f"`x` ({n} × 1) is skewed and `new` holds 6 new values. Fit `KBinsDiscretizer(n_bins={b}, "
                 f"encode='ordinal', strategy='uniform')` on `x`, then reproduce its bins for `new` by hand: "
                 f"`np.searchsorted(inner_edges, value, side='right')` with the inner edges `bin_edges_[0][1:-1]` "
                 f"(values outside the range land in the first / last bin). Return the bins as a float array.", setup,
                 f"from {PRE} import KBinsDiscretizer\nedges = KBinsDiscretizer(n_bins={b}, encode='ordinal', "
                 "strategy='uniform').fit(x).bin_edges_[0]\nresult = np.searchsorted(edges[1:-1], new[:, 0], side='right')"
                 ".astype(float)", require=["bin_edges_"], tol=1e-9)
    if task == "quantile_interp":
        return P(f"`x` ({n} × 1) is skewed, `new` holds 6 values. `QuantileTransformer(n_quantiles={nq})` stores the "
                 f"{nq} percentiles of `x` (`quantiles_`) at the references `np.linspace(0, 1, {nq})` and maps a new "
                 f"value by linear interpolation between them (`np.interp`, clipped to [0, 1]). Reproduce its "
                 f"transform of `new` (there are no ties, so one `np.interp` is enough).", setup,
                 f"from {PRE} import QuantileTransformer\nq = QuantileTransformer(n_quantiles={nq}).fit(x).quantiles_[:, 0]\n"
                 f"result = np.interp(new[:, 0], q, np.linspace(0, 1, {nq}))", require=["quantiles_|QuantileTransformer"],
                 tol=1e-9)
    return P(f"`x` ({n} × 1) is skewed. Bin it into {b} **equal-width** bins by hand: edges `np.linspace(min, max, "
             f"{b + 1})` and `np.searchsorted(edges[1:-1], x, side='right')`. Return `(your bins, sklearn's bins)` with "
             f"`KBinsDiscretizer(n_bins={b}, encode='ordinal', strategy='uniform')`, both as 1-D float arrays.", setup,
             f"from {PRE} import KBinsDiscretizer\nedges = np.linspace(x.min(), x.max(), {b + 1})\n"
             "mine = np.searchsorted(edges[1:-1], x[:, 0], side='right').astype(float)\n"
             f"theirs = KBinsDiscretizer(n_bins={b}, encode='ordinal', strategy='uniform').fit_transform(x)[:, 0]\n"
             "result = (mine, theirs)", require=["KBinsDiscretizer"], tol=1e-9)


@template(M, "hard")
def inverse_round_trips(rng):
    rows = _outlier_matrix(rng)
    n = len(rows)
    task = rng.choice(["clip", "quantile_new", "power", "function_identity", "normalizer"])
    setup = f"X = np.array({lit(rows)})\nX_new = X * 1.5 + 1"
    head = f"`X` is a {n} × 3 matrix and `X_new = X * 1.5 + 1` new data on a wider range."
    if task == "clip":
        return P(f"{head} Fit `MinMaxScaler(clip=True)` on `X`, transform `X_new`, and map it back with "
                 f"`inverse_transform`. Return the recovered matrix: clipping loses everything beyond the training "
                 f"range.", setup,
                 f"from {PRE} import MinMaxScaler\nmm = MinMaxScaler(clip=True).fit(X)\nresult = mm.inverse_transform(mm.transform(X_new))",
                 require=["inverse_transform", "clip"], tol=1e-9)
    if task == "quantile_new":
        return P(f"{head} Fit `QuantileTransformer(n_quantiles={n})` on `X`, transform `X_new`, invert it and return how "
                 f"many recovered values differ from `X_new` by more than 1e-6 (out-of-range values come back clipped "
                 f"to the training range).", setup,
                 f"from {PRE} import QuantileTransformer\nqt = QuantileTransformer(n_quantiles={n}).fit(X)\n"
                 "back = qt.inverse_transform(qt.transform(X_new))\nresult = int((np.abs(back - X_new) > 1e-6).sum())",
                 require=["QuantileTransformer", "inverse_transform"])
    if task == "power":
        return P(f"{head} Fit `PowerTransformer()` on `X` and return `inverse_transform(transform(X_new))` (it round-trips "
                 f"exactly, even outside the training range).", setup,
                 f"from {PRE} import PowerTransformer\npt = PowerTransformer().fit(X)\nresult = pt.inverse_transform(pt.transform(X_new))",
                 require=["PowerTransformer", "inverse_transform"], tol=1e-6)
    if task == "function_identity":
        return P(f"{head} A `FunctionTransformer(np.log1p)` without `inverse_func` inverts with the identity. Apply it to "
                 f"`np.abs(X)`, then call `inverse_transform` and return the result: it is still the log, not `|X|`. "
                 f"(Then imagine adding `inverse_func=np.expm1`.)", setup,
                 f"from {PRE} import FunctionTransformer\nft = FunctionTransformer(np.log1p)\n"
                 "result = ft.inverse_transform(ft.fit_transform(np.abs(X)))", require=["FunctionTransformer"], tol=1e-9)
    return P(f"{head} `Normalizer` has no `inverse_transform`: the row lengths are lost. Return `(lengths, "
             f"rebuilt)`: the L2 row lengths of `X`, and `X` rebuilt from `Normalizer().fit_transform(X)` times those "
             f"lengths.", setup,
             f"from {PRE} import Normalizer\nL = np.linalg.norm(X, axis=1)\nresult = (L, Normalizer().fit_transform(X) * L[:, None])",
             require=["Normalizer"], tol=1e-9)


@template(M, "hard")
def sparse_friendly_scaling(rng):
    n, d, s = rng.randint(40, 120), rng.randint(20, 60), rng.randint(0, 99)
    dens = rng.choice([0.03, 0.05, 0.1])
    setup = (f"import scipy.sparse as sp\nX = sp.random({n}, {d}, density={dens}, format='csr', random_state={s}, "
             f"data_rvs=lambda k: np.random.default_rng({s}).normal(0, 5, k))")
    head = f"`X` is a {n} × {d} SciPy sparse matrix (CSR, ~{dens:.0%} non-zero)."
    task = rng.choice(["nnz", "maxabs", "with_mean", "center_breaks"])
    if task == "nnz":
        return P(f"{head} `MaxAbsScaler` doesn't centre, so zeros stay zeros. Return `(X.nnz, scaled.nnz, "
                 f"type(scaled).__name__)` for `scaled = MaxAbsScaler().fit_transform(X)`.", setup,
                 f"from {PRE} import MaxAbsScaler\nZ = MaxAbsScaler().fit_transform(X)\nresult = (X.nnz, Z.nnz, type(Z).__name__)",
                 require=["MaxAbsScaler"])
    if task == "maxabs":
        return P(f"{head} Fit `MaxAbsScaler` and return `max_abs_` (all-zero columns get 0).", setup,
                 f"from {PRE} import MaxAbsScaler\nresult = MaxAbsScaler().fit(X).max_abs_", require=["MaxAbsScaler"])
    if task == "with_mean":
        return P(f"{head} `StandardScaler(with_mean=False)` divides by the standard deviation without centring, which "
                 f"works on sparse input. Return the dense version of the first 3 rows of the result "
                 f"(`.toarray()`).", setup,
                 f"from {PRE} import StandardScaler\nresult = StandardScaler(with_mean=False).fit_transform(X)[:3].toarray()",
                 require=["StandardScaler", "with_mean"])
    return P(f"{head} Centring would fill in every zero. `StandardScaler()` (with centring) refuses sparse input: catch "
             f"the error and return its type name, together with the number of stored values centring a dense copy "
             f"would need (`X.shape[0] * X.shape[1]`), as a tuple.", setup,
             f"from {PRE} import StandardScaler\ntry:\n    StandardScaler().fit(X)\n    name = None\n"
             "except Exception as e:\n    name = type(e).__name__\nresult = (name, X.shape[0] * X.shape[1])",
             require=["StandardScaler"])


@template(M, "hard")
def choose_scaler_by_cv(rng):
    d = synth_clf(rng, n=(150, 250), d=(4, 6), classes=2)
    s, k = rng.randint(0, 99), rng.choice([5, 7, 9])
    setup = d.setup + f"\nX = X * np.array([1, 10, 100, 1000, 0.1, 5][:X.shape[1]])"
    head = (f"{d.desc}, with the columns multiplied by very different factors. Choose a scaler for "
            f"`KNeighborsClassifier(n_neighbors={k})` by 5-fold CV (`StratifiedKFold(n_splits=5, shuffle=True, "
            f"random_state={s})`), fitting the scaler inside each fold on the training rows only.")
    sol = (f"from sklearn.model_selection import StratifiedKFold\nfrom sklearn.neighbors import KNeighborsClassifier\n"
           f"from {PRE} import MinMaxScaler, RobustScaler, StandardScaler\n"
           f"cv = StratifiedKFold(n_splits=5, shuffle=True, random_state={s})\n"
           "def cv_score(make):\n    sc = []\n    for tr, te in cv.split(X, y):\n"
           "        A, B = X[tr], X[te]\n        if make is not None:\n            t = make().fit(A)\n            A, B = t.transform(A), t.transform(B)\n"
           f"        sc.append(KNeighborsClassifier(n_neighbors={k}).fit(A, y[tr]).score(B, y[te]))\n    return float(np.mean(sc))\n"
           "scores = {'none': cv_score(None), 'standard': cv_score(StandardScaler), 'minmax': cv_score(MinMaxScaler), "
           "'robust': cv_score(RobustScaler)}\n")
    task = rng.choice(["dict", "best", "gain"])
    if task == "dict":
        return P(f"{head} Return a dict mapping `'none'`, `'standard'`, `'minmax'` and `'robust'` to the mean CV accuracy.",
                 setup, sol + "result = scores", require=["StandardScaler", "MinMaxScaler", "RobustScaler"])
    if task == "best":
        return P(f"{head} Candidates: `'none'`, `'standard'`, `'minmax'`, `'robust'` (in that order; first one wins "
                 f"ties). Return the name of the best.", setup, sol + "result = max(scores, key=scores.get)",
                 require=["StandardScaler", "MinMaxScaler", "RobustScaler"])
    return P(f"{head} Return the mean CV accuracy of the best of `StandardScaler`, `MinMaxScaler` and `RobustScaler` "
             f"minus that of no scaling.", setup,
             sol + "result = max(scores['standard'], scores['minmax'], scores['robust']) - scores['none']",
             require=["StandardScaler", "MinMaxScaler", "RobustScaler"])


@template(M, "hard")
def feature_engineering_tasks(rng):
    t = table(rng, kind="reg", numeric_only=True, n=(120, 250))
    a, b = rng.sample(t.num, 2)
    k = rng.choice([4, 5, 6])
    items = [
        ("FunctionTransformer", f"Add `log1p({a})` (clipped at 0 first) as an extra column with a `FunctionTransformer` "
                                f"and return the training R² of `LinearRegression` with and without it, as a tuple.",
         f"from {PRE} import FunctionTransformer\nfrom sklearn.linear_model import LinearRegression\n"
         f"extra = FunctionTransformer(np.log1p).fit_transform(df[[{a!r}]].clip(lower=0))\n"
         "X2 = np.column_stack([df.to_numpy(), extra])\n"
         "result = (LinearRegression().fit(X2, y).score(X2, y), LinearRegression().fit(df, y).score(df, y))"),
        ("SplineTransformer", f"Replace `{a}` by `SplineTransformer(n_knots={k}, degree=3)` features (keep the other "
                              f"columns) and return the training R² of `LinearRegression`.",
         f"from {PRE} import SplineTransformer\nfrom sklearn.linear_model import LinearRegression\n"
         f"S = SplineTransformer(n_knots={k}, degree=3).fit_transform(df[[{a!r}]])\n"
         f"X2 = np.column_stack([df.drop(columns={a!r}).to_numpy(), S])\nresult = LinearRegression().fit(X2, y).score(X2, y)"),
        ("KBinsDiscretizer", f"Replace `{b}` by {k} one-hot equal-width bins (`KBinsDiscretizer(n_bins={k}, "
                             f"encode='onehot-dense', strategy='uniform')`) and return the training R² of "
                             f"`LinearRegression`.",
         f"from {PRE} import KBinsDiscretizer\nfrom sklearn.linear_model import LinearRegression\n"
         f"B = KBinsDiscretizer(n_bins={k}, encode='onehot-dense', strategy='uniform').fit_transform(df[[{b!r}]])\n"
         f"X2 = np.column_stack([df.drop(columns={b!r}).to_numpy(), B])\nresult = LinearRegression().fit(X2, y).score(X2, y)"),
        ("PolynomialFeatures", f"Add only the product `{a} · {b}` using `PolynomialFeatures(degree=2, "
                               f"interaction_only=True, include_bias=False)` on those two columns (its last output), "
                               f"and return the coefficient `LinearRegression` gives it.",
         f"from {PRE} import PolynomialFeatures\nfrom sklearn.linear_model import LinearRegression\n"
         f"inter = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False).fit_transform(df[[{a!r}, {b!r}]])[:, -1]\n"
         "X2 = np.column_stack([df.to_numpy(), inter])\nresult = LinearRegression().fit(X2, y).coef_[-1]"),
        ("Binarizer", f"Add a flag column `{a} > its median` built with `Binarizer(threshold=median)` and return the "
                      f"coefficient of the flag in `LinearRegression`.",
         f"from {PRE} import Binarizer\nfrom sklearn.linear_model import LinearRegression\n"
         f"flag = Binarizer(threshold=df[{a!r}].median()).fit_transform(df[[{a!r}]])\n"
         "X2 = np.column_stack([df.to_numpy(), flag])\nresult = LinearRegression().fit(X2, y).coef_[-1]"),
        ("QuantileTransformer", "Rank-transform every column with `QuantileTransformer(n_quantiles=100)` and return the "
                                "training R² of `LinearRegression` on the result.",
         f"from {PRE} import QuantileTransformer\nfrom sklearn.linear_model import LinearRegression\n"
         "Q = QuantileTransformer(n_quantiles=100).fit_transform(df)\nresult = LinearRegression().fit(Q, y).score(Q, y)"),
        ("StandardScaler", "Standardize `df` and the target (`y` scaled with its own `StandardScaler`), fit "
                           "`LinearRegression` and return the column name with the largest |coefficient|.",
         f"from {PRE} import StandardScaler\nfrom sklearn.linear_model import LinearRegression\n"
         "Z = StandardScaler().fit_transform(df)\nyz = StandardScaler().fit_transform(y.reshape(-1, 1)).ravel()\n"
         "result = df.columns[np.abs(LinearRegression().fit(Z, yz).coef_).argmax()]"),
    ]
    return catalog(rng, items, t.setup, f"{t.desc}.", tol=1e-6)


@template(M, "hard")
def transform_pitfalls(rng):
    n, s = rng.randint(20, 60), rng.randint(0, 99)
    t = rng.randint(2, 7)
    setup = (f"rng = np.random.default_rng({s})\nX = rng.integers(0, 10, ({n}, 2)).astype(float)\n"
             f"X_test = rng.integers(0, 12, (5, 2)).astype(float)")
    imp = f"from {PRE} import "
    items = [
        ("PowerTransformer", "Box-Cox needs strictly positive data and `X` contains zeros. Try "
                             "`PowerTransformer(method='box-cox').fit(X)`, catch the error, and fall back to the default "
                             "Yeo-Johnson. Return `(error type name, Yeo-Johnson lambdas_)`.",
         imp + "PowerTransformer\ntry:\n    PowerTransformer(method='box-cox').fit(X)\n    name = None\n"
               "except ValueError as e:\n    name = type(e).__name__\nresult = (name, PowerTransformer().fit(X).lambdas_)",
         {"tol": 1e-4}),
        ("QuantileTransformer", f"`QuantileTransformer()` defaults to 1000 quantiles, more than the {n} rows: it warns "
                                f"and uses fewer. Fit it (silence the warning with `warnings.catch_warnings()`) and return "
                                f"`n_quantiles_`.",
         "import warnings\n" + imp + "QuantileTransformer\nwith warnings.catch_warnings():\n"
         "    warnings.simplefilter('ignore')\n    result = QuantileTransformer().fit(X).n_quantiles_"),
        ("PolynomialFeatures", "Order matters: return the column means of `PolynomialFeatures(degree=2, "
                               "include_bias=False)` applied **after** standardizing `X`, then of the expansion "
                               "standardized **afterwards**, as a tuple. Squares of standardized data have mean 1, not 0.",
         imp + "PolynomialFeatures, StandardScaler\npf = PolynomialFeatures(degree=2, include_bias=False)\n"
               "a = pf.fit_transform(StandardScaler().fit_transform(X)).mean(axis=0)\n"
               "b = StandardScaler().fit_transform(pf.fit_transform(X)).mean(axis=0)\nresult = (a, b)"),
        ("Binarizer", f"`Binarizer(threshold={t})` is a strict `>`. Return `(ones with Binarizer, ones with value >= "
                      f"{t})` counted over `X`: the difference is the values equal to {t}.",
         imp + f"Binarizer\nresult = (int(Binarizer(threshold={t}).fit_transform(X).sum()), int((X >= {t}).sum()))"),
        ("transform", "This is a bug: it re-learns the statistics from the test rows.\n\n```\nsc = StandardScaler()"
                      ".fit(X)\nX_test_s = sc.fit_transform(X_test)\n```\n\nFix it and return `(buggy, fixed)` "
                      "scaled test sets.",
         imp + "StandardScaler\nsc = StandardScaler().fit(X)\nfixed = sc.transform(X_test)\n"
               "buggy = StandardScaler().fit_transform(X_test)\nresult = (buggy, fixed)", {"require": ["transform"]}),
    ]
    return catalog(rng, items, setup, f"`X` ({n} × 2) holds integer counts 0-9 and `X_test` 5 new rows.", tol=1e-6)


# ================================================================= challenge

@template(M, "challenge")
def sliding_minmax_scale(rng):
    n, w, s = rng.randint(9, 14), rng.randint(3, 5), seeded(rng)
    report = rng.choice(["last", "first", "range"])
    setup = f"g = np.random.default_rng({s})\nx = g.integers(0, 20, {n}).astype(float)\nw = {w}\nreport = {report!r}"
    sol = ("from collections import deque\nxs = x.tolist()\nlo, hi = deque(), deque()\nmn, mx = [], []\n"
           "for i, v in enumerate(xs):\n    while lo and xs[lo[-1]] >= v:\n        lo.pop()\n    lo.append(i)\n"
           "    while hi and xs[hi[-1]] <= v:\n        hi.pop()\n    hi.append(i)\n    if lo[0] <= i - w:\n        lo.popleft()\n"
           "    if hi[0] <= i - w:\n        hi.popleft()\n    if i >= w - 1:\n        mn.append(xs[lo[0]])\n        mx.append(xs[hi[0]])\n"
           "mn, mx = np.array(mn), np.array(mx)\nrng_ = mx - mn\nscale = np.where(rng_ == 0, 1.0, rng_)\n"
           "if report == 'last':\n    result = (x[w - 1:] - mn) / scale\nelif report == 'first':\n"
           "    result = (x[:len(x) - w + 1] - mn) / scale\nelse:\n    result = rng_")
    tests = [f"g = np.random.default_rng({s + i})\nx = g.normal(0, 10, 200000).round(1)\nw = {ww}\nreport = {report!r}"
             for i, ww in ((1, 50000), (2, 3000))]
    what = {"last": "the scaled value of the window's **last** reading", "first": "the scaled value of the window's "
            "**first** reading", "range": "the window's range `max − min`"}[report]
    return challenge(
        "Sliding min-max scaling",
        "A dashboard rescales a sensor stream `x` to [0, 1] over a sliding window: for every window `x[i:i+w]` (all "
        f"`len(x) − w + 1` of them, in order) it fits a `MinMaxScaler` and reports {what}. A window whose values are all "
        f"equal scales everything to 0 (scikit-learn treats a zero range as 1). Return a float array. "
        f"(`report`: {report!r}.)",
        setup, sol, tests, "len(x) ≤ 2·10^5, w ≤ len(x)", speed="O(n): monotonic deques for the window min and max",
        tol=1e-9)


@template(M, "challenge")
def rolling_robust_scale(rng):
    n, w, s = rng.randint(9, 14), rng.randint(3, 6), seeded(rng)
    report = rng.choice(["median", "iqr", "robust_last"])
    setup = f"g = np.random.default_rng({s})\nx = g.integers(0, 30, {n}).astype(float)\nw = {w}\nreport = {report!r}"
    sol = ("from bisect import insort, bisect_left\nxs = x.tolist()\nwin = sorted(xs[:w])\n"
           "def pct(a, q):\n    pos = q * (len(a) - 1)\n    i = int(pos)\n    f = pos - i\n"
           "    return a[i] + (a[i + 1] - a[i]) * f if f else a[i]\n"
           "out = []\nfor i in range(w - 1, len(xs)):\n    if i >= w:\n        del win[bisect_left(win, xs[i - w])]\n"
           "        insort(win, xs[i])\n    med = pct(win, 0.5)\n    iqr = pct(win, 0.75) - pct(win, 0.25)\n"
           "    if report == 'median':\n        out.append(med)\n    elif report == 'iqr':\n        out.append(iqr)\n"
           "    else:\n        out.append((xs[i] - med) / (iqr if iqr else 1.0))\nresult = np.array(out)")
    tests = [f"g = np.random.default_rng({s + i})\nx = g.normal(0, 10, 200000).round(2)\nw = {ww}\nreport = {report!r}"
             for i, ww in ((1, 5000), (2, 801))]
    what = {"median": "the window **median** (`center_`)", "iqr": "the window **interquartile range** (`scale_`: "
            "75th minus 25th percentile)", "robust_last": "the robust-scaled value of the window's **last** reading, "
            "`(x − median) / IQR` (an IQR of 0 counts as 1, as in scikit-learn)"}[report]
    return challenge(
        "Rolling robust scaling",
        "A robust anomaly detector refits a `RobustScaler` on every sliding window `x[i:i+w]` (all `len(x) − w + 1` of "
        f"them, in order) and reports {what}. Percentiles use linear interpolation (NumPy's default): the q-th "
        f"percentile of the sorted window `a` sits at position `q/100 · (w − 1)`. Return a float array. "
        f"(`report`: {report!r}.)",
        setup, sol, tests, "len(x) ≤ 2·10^5, w ≤ 5000", speed="about O(n log w + n·w/word): keep the window sorted "
        "(`bisect`), don't re-sort it", tol=1e-9)


@template(M, "challenge")
def polynomial_feature_index(rng):
    nf, deg, s = rng.randint(2, 4), rng.randint(2, 3), seeded(rng)
    inter, bias = rng.random() < 0.4, rng.random() < 0.5
    report = rng.choice(["index", "name"])
    q = rng.randint(4, 6)
    if inter:
        nf = max(nf, deg)
    setup = (f"g = np.random.default_rng({s})\nn_features, degree = {nf}, {deg}\ninteraction_only, include_bias = "
             f"{inter}, {bias}\nreport = {report!r}\n")
    gen = ("def _draw(g, q, nf, deg, inter):\n    out = []\n    while len(out) < q:\n"
           "        d = int(g.integers(1, deg + 1))\n"
           "        v = g.choice(nf, d, replace=False) if inter else g.integers(0, nf, d)\n"
           "        out.append(np.bincount(v, minlength=nf))\n    return np.array(out)\n")
    total = ("import math\ncnt = (lambda e: math.comb(n_features, e)) if interaction_only else "
             "(lambda e: math.comb(n_features + e - 1, e))\n")
    if report == "index":
        setup += gen + f"powers = _draw(g, {q}, n_features, degree, interaction_only)\ndel _draw"
    else:
        setup += ("import math\n_tot = sum((math.comb(n_features, e) if interaction_only else math.comb(n_features + e - 1, e))"
                  f" for e in range(1, degree + 1)) + include_bias\ncols = g.integers(0, _tot, {q})\ndel _tot")
    sol = (total +
           "def comp(n, r, v):   # number of ways to complete r more slots with indices > v (inter) or >= v\n"
           "    return math.comb(n - v - 1, r) if interaction_only else math.comb(n - v + r - 1, r)\n"
           "def block(lo, hi, n, r):   # sum of comp(n, r, v) for v in [lo, hi)\n"
           "    if interaction_only:\n        return math.comb(n - lo, r + 1) - math.comb(n - hi, r + 1)\n"
           "    return math.comb(n - lo + r, r + 1) - math.comb(n - hi + r, r + 1)\n"
           "n = n_features\nstarts = [int(include_bias)]\nfor e in range(1, degree + 1):\n    starts.append(starts[-1] + cnt(e))\n"
           "if report == 'index':\n    result = []\n    for row in powers.tolist():\n"
           "        idx = [j for j, p in enumerate(row) for _ in range(p)]\n        d = len(idx)\n        r, prev = starts[d - 1], 0\n"
           "        for pos, v in enumerate(idx):\n            r += block(prev, v, n, d - pos - 1)\n"
           "            prev = v + 1 if interaction_only else v\n        result.append(r)\n    result = np.array(result)\n"
           "else:\n    result = []\n    for c in cols.tolist():\n        if include_bias and c == 0:\n"
           "            result.append('1')\n            continue\n        d = next(e for e in range(1, degree + 1) if c < starts[e])\n"
           "        r, prev, idx = c - starts[d - 1], 0, []\n        for pos in range(d):\n            v = prev\n"
           "            while comp(n, d - pos - 1, v) <= r:\n                r -= comp(n, d - pos - 1, v)\n                v += 1\n"
           "            idx.append(v)\n            prev = v + 1 if interaction_only else v\n"
           "        cnts = np.bincount(idx, minlength=n)\n"
           "        result.append(' '.join(f'x{j}' if p == 1 else f'x{j}^{p}' for j, p in enumerate(cnts.tolist()) if p))")
    if report == "index":
        tests = [f"g = np.random.default_rng({s + i})\nn_features, degree = {a}, {b}\ninteraction_only, include_bias = "
                 f"{inter}, {bias}\nreport = 'index'\n" + gen + f"powers = _draw(g, 5000, n_features, degree, interaction_only)\n"
                 "del _draw" for i, (a, b) in ((1, (40, 4)), (2, (30, 5)))]
    else:
        tests = [f"g = np.random.default_rng({s + i})\nn_features, degree = {a}, {b}\ninteraction_only, include_bias = "
                 f"{inter}, {bias}\nreport = 'name'\nimport math\n_tot = sum((math.comb(n_features, e) if interaction_only "
                 f"else math.comb(n_features + e - 1, e)) for e in range(1, degree + 1)) + include_bias\n"
                 f"cols = g.integers(0, _tot, 5000)\ndel _tot" for i, (a, b) in ((1, (40, 4)), (2, (30, 5)))]
    what = ("`powers` has one row per query: the exponent of each input feature in a monomial (as in `powers_`). "
            "Return the output column index of every queried monomial, as an int array" if report == "index" else
            "`cols` holds output column indices. Return the output **name** of each, as `get_feature_names_out()` "
            "writes it (`'1'` for the bias, then e.g. `'x0 x2^2'`: features in index order, exponent shown when above "
            "1), as a list of str")
    return challenge(
        "Where is that polynomial feature?",
        "`PolynomialFeatures(degree, interaction_only=..., include_bias=...)` orders its outputs by total degree (the "
        "bias `1` first when included); within one degree it lists the monomials as sorted tuples of feature indices "
        "in lexicographic order (`x0², x0x1, x0x2, x1², ...` — with `interaction_only` no index repeats). Building the "
        f"whole list is too slow for many features and a high degree. {what}. (`report`: {report!r}.)",
        setup, sol, tests, "n_features ≤ 40, degree ≤ 5, queries ≤ 5000 (the full output can have millions of columns)",
        speed="O(degree · n_features) per query with combinatorial counting (`math.comb`)")


@template(M, "challenge")
def sparse_maxabs_scale(rng):
    R, C, s = rng.randint(4, 7), rng.randint(3, 6), seeded(rng)
    nnz = rng.randint(6, 10)
    report = rng.choice(["scaled", "max_abs", "row_norms"])
    setup = (f"g = np.random.default_rng({s})\nn_rows, n_cols = {R}, {C}\nlin = g.choice(n_rows * n_cols, {nnz}, replace=False)\n"
             f"g.shuffle(lin)\nrows, cols = lin // n_cols, lin % n_cols\nvals = g.integers(-9, 10, {nnz}).astype(float)\n"
             f"vals[vals == 0] = 1.0\ndel lin\nreport = {report!r}")
    sol = ("m = np.zeros(n_cols)\nnp.maximum.at(m, cols, np.abs(vals))\nscale = np.where(m == 0, 1.0, m)\nz = vals / scale[cols]\n"
           "if report == 'scaled':\n    result = z\nelif report == 'max_abs':\n    result = m\nelse:\n"
           "    result = np.sqrt(np.bincount(rows, weights=z * z, minlength=n_rows))")
    tests = [f"g = np.random.default_rng({s + i})\nn_rows, n_cols = 100000, {cc}\nlin = np.unique(g.integers(0, n_rows * n_cols, "
             f"200000))\ng.shuffle(lin)\nrows, cols = lin // n_cols, lin % n_cols\nvals = g.normal(0, 3, len(lin))\ndel lin\n"
             f"report = {report!r}" for i, cc in ((1, 100000), (2, 3000))]
    what = {"scaled": "the scaled values, in the same order as `vals`",
            "max_abs": "`max_abs_`: the largest absolute value of every column (0 for an empty column), a float array "
                       "of length `n_cols`",
            "row_norms": "the Euclidean length of every row of the **scaled** matrix (0 for an empty row), a float "
                         "array of length `n_rows`"}[report]
    return challenge(
        "Max-abs scaling a sparse matrix",
        "A bag-of-words matrix is far too big to store densely, so it is kept as coordinate triplets: entry `k` is "
        "`vals[k]` at row `rows[k]`, column `cols[k]` (no duplicate positions; every other entry is 0). `MaxAbsScaler` "
        "suits it because it never centres: each column is divided by its largest absolute value (a column with no "
        f"entries is left alone). Without densifying, return {what}. (`report`: {report!r}.)",
        setup, sol, tests, "n_rows, n_cols ≤ 10^5, non-zeros ≤ 2·10^5", speed="O(nnz): scatter operations "
        "(`np.maximum.at`, `np.bincount`), never a dense n_rows × n_cols array", tol=1e-9)


@template(M, "challenge")
def binarizer_threshold_sweep(rng):
    n, d, q, s = rng.randint(4, 7), rng.randint(2, 4), rng.randint(4, 6), seeded(rng)
    report = rng.choice(["ones", "rows_any", "rows_all"])
    setup = (f"g = np.random.default_rng({s})\nX = g.integers(0, 10, ({n}, {d})).astype(float)\n"
             f"ts = g.integers(-1, 11, {q}) + 0.5 * g.integers(0, 2, {q})\nreport = {report!r}")
    sol = ("if report == 'ones':\n    v = np.sort(X.ravel())\nelif report == 'rows_any':\n    v = np.sort(X.max(axis=1))\n"
           "else:\n    v = np.sort(X.min(axis=1))\nresult = len(v) - np.searchsorted(v, ts, side='right')")
    tests = [f"g = np.random.default_rng({s + i})\nX = g.normal(0, 1, ({r}, {c})).round(3)\nts = g.normal(0, 1.5, 200000).round(3)\n"
             f"report = {report!r}" for i, (r, c) in ((1, (100000, 2)), (2, (20000, 10)))]
    what = {"ones": "the number of ones in the whole output", "rows_any": "the number of rows with at least one 1",
            "rows_all": "the number of rows that are all ones"}[report]
    return challenge(
        "Sweeping a Binarizer threshold",
        "To pick a cut-off, an analyst applies `Binarizer(threshold=t)` (a value becomes 1 when it is **strictly "
        f"greater** than `t`) to the matrix `X` for every `t` in `ts` and records {what}. Return the counts as an int "
        f"array, one per threshold. (`report`: {report!r}.)",
        setup, sol, tests, "X has ≤ 2·10^5 values, len(ts) ≤ 2·10^5", speed="O((n + q) log n): sort once, binary search")


@template(M, "challenge")
def quantile_uniform_transform(rng):
    n, q, s = rng.randint(6, 10), rng.randint(5, 8), seeded(rng)
    report = rng.choice(["values", "mean", "clipped"])
    setup = (f"g = np.random.default_rng({s})\nx = g.integers(0, 8, {n}).astype(float)\nqueries = (g.integers(-2, 20, {q}) "
             f"/ 2)\nreport = {report!r}")
    sol = ("xs = np.sort(x)\nn = len(xs)\nlo = np.searchsorted(xs, queries, side='left')\nhi = np.searchsorted(xs, queries, side='right')\n"
           "tie = 0.5 * (lo + hi - 1)\nj = np.clip(lo, 1, n - 1)\nleft, right = xs[j - 1], xs[j]\n"
           "between = (j - 1) + (queries - left) / np.where(right > left, right - left, 1)\n"
           "u = np.where(hi > lo, tie, between) / (n - 1)\nu = np.where(queries <= xs[0], 0.0, np.where(queries >= xs[-1], 1.0, u))\n"
           "result = {'values': u, 'mean': float(u.mean()), 'clipped': int(((queries <= xs[0]) | (queries >= xs[-1])).sum())}[report]")
    tests = [f"g = np.random.default_rng({s + i})\nx = g.integers(0, {hi}, 200000).astype(float)\n"
             f"queries = g.integers(-5, 2 * {hi} + 5, 200000) / 2\nreport = {report!r}" for i, hi in ((1, 1000), (2, 50000))]
    what = {"values": "the transformed value of every query (a float array)",
            "mean": "the mean transformed value over the queries (a float)",
            "clipped": "how many queries fall at or outside the training range (mapped to exactly 0 or 1)"}[report]
    return challenge(
        "Quantile transform with ties",
        "`QuantileTransformer(n_quantiles=len(x))` fitted on `x` maps a value `v` to its position in the sorted "
        "training data `xs`, scaled to [0, 1]: `v ≤ xs[0]` → 0 and `v ≥ xs[-1]` → 1; if `v` equals training values at "
        "sorted positions `a … b`, it maps to the middle position `(a + b) / 2 / (n − 1)` (the transformer averages an "
        "upward and a downward interpolation for exactly this); otherwise `v` lies strictly between `xs[i − 1]` and "
        "`xs[i]` and maps to `(i − 1 + (v − xs[i − 1]) / (xs[i] − xs[i − 1])) / (n − 1)`. For the values in `queries`, "
        f"return {what}. (`report`: {report!r}.)",
        setup, sol, tests, "len(x), len(queries) ≤ 2·10^5", speed="O((n + q) log n): sort once, binary search",
        tol=1e-9)
