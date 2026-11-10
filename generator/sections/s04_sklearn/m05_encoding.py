"""4.05 Encoding categories & imputing missing values: OneHotEncoder, OrdinalEncoder, the label encoders,
TargetEncoder, SimpleImputer, KNNImputer, IterativeImputer and MissingIndicator."""

from ._common import P, catalog, challenge, lit, seeded, synth_reg, table, template

M = "4.05"
PRE = "sklearn.preprocessing"
IMP = "sklearn.impute"
ITER = "from sklearn.experimental import enable_iterative_imputer  # noqa: F401\nfrom sklearn.impute import IterativeImputer\n"

# column -> levels (ordered ones are listed in their natural order)
VOCAB = {
    "color": ["red", "green", "blue", "black"],
    "size": ["S", "M", "L", "XL"],
    "plan": ["basic", "standard", "premium"],
    "city": ["paris", "lima", "oslo", "tokyo", "cairo"],
    "fuel": ["petrol", "diesel", "electric", "hybrid"],
    "rating": ["low", "medium", "high"],
    "browser": ["firefox", "chrome", "safari", "edge"],
}
ORDERED = ["size", "plan", "rating"]


def _cat_frame(rng, n=(6, 10), cols=None):
    """A small categorical DataFrame `cat` (2 columns, every column with at least 2 levels)."""
    cols = cols or rng.sample(sorted(VOCAB), 2)
    n_ = rng.randint(*n)
    data = {}
    for c in cols:
        levels = VOCAB[c][:rng.randint(2, len(VOCAB[c]))] if len(VOCAB[c]) > 2 else VOCAB[c]
        vals = [rng.choice(levels) for _ in range(n_)]
        vals[0], vals[1] = levels[0], levels[1]
        data[c] = vals
    return "cat = pd.DataFrame({" + ", ".join(f"{c!r}: {lit(v)}" for c, v in data.items()) + "})", cols, data, n_


def _nan_matrix(rng, r=(6, 9), c=(2, 4), floats=False):
    """A small float matrix with NaNs: every column keeps at least 2 observed values, at least one value is missing.
    floats: continuous values (no ties in distances)."""
    n, d = rng.randint(*r), rng.randint(*c)
    draw = (lambda: round(rng.uniform(0, 20), 2)) if floats else (lambda: float(rng.randint(0, 20)))
    rows = [[draw() for _ in range(d)] for _ in range(n)]
    holes = 0
    for j in range(d):
        for i in rng.sample(range(n), rng.randint(0 if j else 1, max(1, n // 3))):
            rows[i][j] = None
            holes += 1
    text = "[" + ", ".join("[" + ", ".join("np.nan" if v is None else repr(v) for v in row) + "]" for row in rows) + "]"
    return f"X = np.array({text})", n, d


# ================================================================= lite

@template(M, "lite")
def encoder_calls(rng):
    setup, cols, data, n = _cat_frame(rng)
    a, b = cols
    imp = f"from {PRE} import "
    ys = [rng.choice(["cat", "dog", "bird"]) for _ in range(rng.randint(4, 7))]
    items = [
        ("OneHotEncoder", "One-hot encode both columns with `OneHotEncoder(sparse_output=False)` and return the dense "
                          "array.", imp + "OneHotEncoder\nresult = OneHotEncoder(sparse_output=False).fit_transform(cat)"),
        ("categories_", "Fit a `OneHotEncoder` on `cat` and return its learned levels (`categories_`) as a list of "
                        "lists (they come sorted).",
         imp + "OneHotEncoder\nresult = [list(c) for c in OneHotEncoder().fit(cat).categories_]"),
        ("get_feature_names_out", "Fit a `OneHotEncoder` on `cat` and return its output column names.",
         imp + "OneHotEncoder\nresult = OneHotEncoder().fit(cat).get_feature_names_out()"),
        ("drop", "Fit `OneHotEncoder(drop='first', sparse_output=False)` (one level dropped per column, as linear "
                 "models like) and return the shape of the encoded array.",
         imp + "OneHotEncoder\nresult = OneHotEncoder(drop='first', sparse_output=False).fit_transform(cat).shape"),
        ("OrdinalEncoder", "Encode each level as an integer with `OrdinalEncoder()` and return the result (levels are "
                           "numbered in sorted order).", imp + "OrdinalEncoder\nresult = OrdinalEncoder().fit_transform(cat)"),
        ("handle_unknown", f"Fit `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` on `cat` and transform a "
                           f"row with an unseen `{a}`: `pd.DataFrame({{{a!r}: ['???'], {b!r}: [{data[b][0]!r}]}})`. "
                           f"Return the encoded row (all zeros for `{a}`; scikit-learn warns about the unknown level).",
         imp + f"OneHotEncoder\nohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(cat)\n"
               f"result = ohe.transform(pd.DataFrame({{{a!r}: ['???'], {b!r}: [{data[b][0]!r}]}}))"),
        ("LabelEncoder", f"`LabelEncoder` is for **targets**. Encode `labels = {ys}` and return the integer codes.",
         imp + f"LabelEncoder\nresult = LabelEncoder().fit_transform({ys})"),
        ("LabelBinarizer", f"One-hot the target `labels = {ys}` with `LabelBinarizer` and return the matrix.",
         imp + f"LabelBinarizer\nresult = LabelBinarizer().fit_transform({ys})"),
        ("MultiLabelBinarizer", "Each sample may carry several tags: encode `tags = [['a', 'b'], ['c'], ['b', 'c'], "
                                "['a']]` with `MultiLabelBinarizer` and return the 0/1 matrix.",
         imp + "MultiLabelBinarizer\nresult = MultiLabelBinarizer().fit_transform([['a', 'b'], ['c'], ['b', 'c'], ['a']])"),
        ("nunique", f"How many output columns will one-hot encoding create? Return the number of distinct levels "
                         f"per column as a dict (`nunique`), then check it against `OneHotEncoder`.",
         "result = cat.nunique().to_dict()", {"require": []}),
    ]
    return catalog(rng, items, setup, f"`cat` is a DataFrame of {n} rows with the categorical columns `{a}` and `{b}`. "
                                      f"Encoders live in `{PRE}`.")


@template(M, "lite")
def imputer_calls(rng):
    setup, n, d = _nan_matrix(rng)
    v = rng.choice([-1, 0, 99])
    k = rng.choice([1, 2, 3])
    imp = f"from {IMP} import "
    items = [
        ("SimpleImputer", "Fill every NaN with its column mean (`SimpleImputer()`, the default strategy) and return the "
                          "result.", imp + "SimpleImputer\nresult = SimpleImputer().fit_transform(X)"),
        ("strategy", "Fill every NaN with its column **median** (`SimpleImputer(strategy='median')`).",
         imp + "SimpleImputer\nresult = SimpleImputer(strategy='median').fit_transform(X)", {"require": ["SimpleImputer", "strategy"]}),
        ("strategy", "Fill every NaN with its column's most frequent value (`strategy='most_frequent'`; ties go to the "
                     "smallest value).", imp + "SimpleImputer\nresult = SimpleImputer(strategy='most_frequent').fit_transform(X)",
         {"require": ["SimpleImputer", "strategy"]}),
        ("fill_value", f"Fill every NaN with the constant {v} (`strategy='constant', fill_value={v}`).",
         imp + f"SimpleImputer\nresult = SimpleImputer(strategy='constant', fill_value={v}).fit_transform(X)"),
        ("statistics_", "Fit a mean `SimpleImputer` and return the values it will fill in (`statistics_`).",
         imp + "SimpleImputer\nresult = SimpleImputer().fit(X).statistics_"),
        ("add_indicator", "Mean-impute with `add_indicator=True`: missing-flag columns are appended. Return the shape of "
                          "the result.", imp + "SimpleImputer\nresult = SimpleImputer(add_indicator=True).fit_transform(X).shape"),
        ("MissingIndicator", "Return the boolean missing-value mask that `MissingIndicator()` produces (one column per "
                             "feature that had missing values during `fit`).",
         imp + "MissingIndicator\nresult = MissingIndicator().fit_transform(X)"),
        ("KNNImputer", f"Fill each NaN with the mean of that column over the row's {k} nearest neighbours "
                       f"(`KNNImputer(n_neighbors={k})`).", imp + f"KNNImputer\nresult = KNNImputer(n_neighbors={k}).fit_transform(X)"),
        ("IterativeImputer", "Model each column from the others with `IterativeImputer(random_state=0, max_iter=50, "
                             "tol=0.1)` (import `enable_iterative_imputer` from `sklearn.experimental` first) and return "
                             "the imputed matrix.",
         ITER + "result = IterativeImputer(random_state=0, max_iter=50, tol=0.1).fit_transform(X)",
         {"tol": 1e-4}),
        ("isnan", "Before imputing, look: return the number of NaNs per column.", "result = np.isnan(X).sum(axis=0)",
         {"require": []}),
    ]
    return catalog(rng, items, setup, f"`X` ({n} × {d}) has missing values (`np.nan`). Imputers live in `{IMP}`.",
                   tol=1e-9)


@template(M, "lite")
def label_tools(rng):
    pool = rng.choice([["cat", "dog", "bird", "fish"], ["spam", "ham"], ["low", "mid", "high"], ["setosa", "versicolor",
                                                                                                   "virginica"]])
    labels = [rng.choice(pool) for _ in range(rng.randint(5, 8))]
    labels[:2] = pool[:2]
    tags = [sorted(rng.sample(["ml", "python", "sql", "stats", "viz"], rng.randint(1, 3))) for _ in range(rng.randint(3, 5))]
    setup = f"labels = np.array({labels})\ntags = {tags}"
    imp = f"from {PRE} import "
    items = [
        ("classes_", "Fit a `LabelEncoder` on `labels` and return `classes_` (the sorted distinct labels).",
         imp + "LabelEncoder\nresult = LabelEncoder().fit(labels).classes_"),
        ("inverse_transform", "Encode `labels` with a `LabelEncoder`, then decode the codes `[1, 0]` back with "
                              "`inverse_transform`.",
         imp + "LabelEncoder\nle = LabelEncoder().fit(labels)\nresult = le.inverse_transform([1, 0])"),
        ("LabelBinarizer", "Return the shape of `LabelBinarizer().fit_transform(labels)` (a binary problem gets a "
                           "single column).", imp + "LabelBinarizer\nresult = LabelBinarizer().fit_transform(labels).shape"),
        ("inverse_transform", "Binarize `labels` with a `LabelBinarizer` and decode the result back with "
                              "`inverse_transform` (it returns the labels).",
         imp + "LabelBinarizer\nlb = LabelBinarizer().fit(labels)\nresult = lb.inverse_transform(lb.transform(labels))"),
        ("MultiLabelBinarizer", "Fit a `MultiLabelBinarizer` on `tags` and return `classes_`.",
         imp + "MultiLabelBinarizer\nresult = MultiLabelBinarizer().fit(tags).classes_"),
        ("MultiLabelBinarizer", "Encode `tags` with a `MultiLabelBinarizer` and return how many samples carry each tag "
                                "(column sums).", imp + "MultiLabelBinarizer\nresult = MultiLabelBinarizer().fit_transform(tags).sum(axis=0)"),
        ("inverse_transform", "Encode `tags` with a `MultiLabelBinarizer` and decode the matrix back with "
                              "`inverse_transform` (a list of tuples).",
         imp + "MultiLabelBinarizer\nmlb = MultiLabelBinarizer()\nresult = mlb.inverse_transform(mlb.fit_transform(tags))"),
    ]
    return catalog(rng, items, setup, "`labels` is a target array of strings and `tags` a list of tag lists.")


# ================================================================= easy

@template(M, "easy")
def onehot_by_hand(rng):
    setup, cols, data, n = _cat_frame(rng)
    a, b = cols
    task = rng.choice(["column", "dummies", "names", "sums"])
    if task == "column":
        return P(f"`cat` has columns `{a}` and `{b}`. Build the one-hot matrix of `{a}` by hand: one 0/1 column per "
                 f"sorted distinct level (`np.unique`), then return `(by_hand, from_sklearn)` where the second is "
                 f"`OneHotEncoder(sparse_output=False)` fitted on `cat[[{a!r}]]`.", setup,
                 f"from {PRE} import OneHotEncoder\nv = cat[{a!r}].to_numpy()\nlevels = np.unique(v)\n"
                 "mine = (v[:, None] == levels[None, :]).astype(float)\n"
                 f"result = (mine, OneHotEncoder(sparse_output=False).fit_transform(cat[[{a!r}]]))", require=["OneHotEncoder"])
    if task == "dummies":
        return P(f"`pd.get_dummies(cat, dtype=float)` and `OneHotEncoder(sparse_output=False)` give the same numbers. "
                 f"Return `(get_dummies values, encoder output)`.", setup,
                 f"from {PRE} import OneHotEncoder\nresult = (pd.get_dummies(cat, dtype=float).to_numpy(), "
                 "OneHotEncoder(sparse_output=False).fit_transform(cat))", require=["OneHotEncoder"])
    if task == "names":
        return P(f"Encode `cat` with `OneHotEncoder(sparse_output=False)` and return a DataFrame with the encoder's "
                 f"output names as columns.", setup,
                 f"from {PRE} import OneHotEncoder\nohe = OneHotEncoder(sparse_output=False)\n"
                 "result = pd.DataFrame(ohe.fit_transform(cat), columns=ohe.get_feature_names_out())",
                 require=["OneHotEncoder", "get_feature_names_out"])
    return P(f"Encode `cat` with `OneHotEncoder(sparse_output=False)` and return a dict output name → how many rows "
             f"have that level (the column sums).", setup,
             f"from {PRE} import OneHotEncoder\nohe = OneHotEncoder(sparse_output=False)\nZ = ohe.fit_transform(cat)\n"
             "result = dict(zip(ohe.get_feature_names_out(), Z.sum(axis=0)))", require=["OneHotEncoder"])


@template(M, "easy")
def ordered_levels(rng):
    col = rng.choice(ORDERED)
    other = rng.choice([c for c in VOCAB if c not in ORDERED])
    setup, cols, data, n = _cat_frame(rng, cols=[col, other])
    order = VOCAB[col]
    task = rng.choice(["explicit", "alphabetical", "inverse", "both"])
    if task == "explicit":
        return P(f"`{col}` has a natural order `{order}`. Encode it with `OrdinalEncoder(categories=[{order}])` "
                 f"(fit on `cat[[{col!r}]]`) and return the codes as a 1-D array.", setup,
                 f"from {PRE} import OrdinalEncoder\nresult = OrdinalEncoder(categories=[{order}]).fit_transform(cat[[{col!r}]])[:, 0]",
                 require=["OrdinalEncoder", "categories"])
    if task == "alphabetical":
        return P(f"Without `categories`, `OrdinalEncoder` numbers the levels alphabetically, which can scramble a "
                 f"natural order like `{order}`. Return `(default_codes, ordered_codes)` for `cat[[{col!r}]]` (1-D arrays).",
                 setup, f"from {PRE} import OrdinalEncoder\nX = cat[[{col!r}]]\n"
                        f"result = (OrdinalEncoder().fit_transform(X)[:, 0], OrdinalEncoder(categories=[{order}]).fit_transform(X)[:, 0])",
                 require=["OrdinalEncoder", "categories"])
    if task == "inverse":
        codes = [float(rng.randrange(len(order))) for _ in range(3)]
        return P(f"Fit `OrdinalEncoder(categories=[{order}])` on `cat[[{col!r}]]` and decode the codes `{codes}` back "
                 f"to levels with `inverse_transform` (it needs a column: `np.array({codes}).reshape(-1, 1)`). Return "
                 f"them as a list.", setup,
                 f"from {PRE} import OrdinalEncoder\noe = OrdinalEncoder(categories=[{order}]).fit(cat[[{col!r}]])\n"
                 f"result = oe.inverse_transform(np.array({codes}).reshape(-1, 1))[:, 0].tolist()",
                 require=["OrdinalEncoder", "inverse_transform"])
    return P(f"Encode both columns at once: `{col}` in its natural order `{order}` and `{other}` alphabetically "
             f"(`categories=[{order}, sorted levels of {other}]`). Return the encoded array.", setup,
             f"from {PRE} import OrdinalEncoder\n"
             f"result = OrdinalEncoder(categories=[{order}, sorted(cat[{other!r}].unique())]).fit_transform(cat)",
             require=["OrdinalEncoder", "categories"])


@template(M, "easy")
def unseen_levels(rng):
    setup, cols, data, n = _cat_frame(rng)
    a, b = cols
    new = f"new = pd.DataFrame({{{a!r}: ['unknown', {data[a][0]!r}], {b!r}: [{data[b][1]!r}, 'mystery']}})"
    setup += "\n" + new
    head = (f"`cat` ({n} rows, columns `{a}`, `{b}`) is the training data; `new` holds 2 rows with a level never seen "
            f"in training (one per column).")
    task = rng.choice(["ignore", "crash", "ordinal", "zeros"])
    if task == "ignore":
        return P(f"{head} Fit `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` on `cat` and return the "
                 f"encoded `new` (unknown levels become all zeros; scikit-learn warns about them).", setup,
                 f"from {PRE} import OneHotEncoder\nresult = OneHotEncoder(handle_unknown='ignore', sparse_output=False)"
                 ".fit(cat).transform(new)", require=["OneHotEncoder", "handle_unknown"])
    if task == "crash":
        return P(f"{head} With the default `handle_unknown='error'`, transforming `new` raises. Catch it and return the "
                 f"exception's type name.", setup,
                 f"from {PRE} import OneHotEncoder\nohe = OneHotEncoder().fit(cat)\ntry:\n    ohe.transform(new)\n"
                 "    result = None\nexcept Exception as e:\n    result = type(e).__name__", require=["OneHotEncoder"])
    if task == "ordinal":
        return P(f"{head} Fit `OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)` on `cat` and "
                 f"return the encoded `new`.", setup,
                 f"from {PRE} import OrdinalEncoder\nresult = OrdinalEncoder(handle_unknown='use_encoded_value', "
                 "unknown_value=-1).fit(cat).transform(new)", require=["OrdinalEncoder", "unknown_value"])
    return P(f"{head} Fit `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` on `cat`; for each row of `new`, "
             f"return how many of its output values are 1 (the unknown level contributes none).", setup,
             f"from {PRE} import OneHotEncoder\nZ = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(cat)"
             ".transform(new)\nresult = Z.sum(axis=1)", require=["OneHotEncoder", "handle_unknown"])


@template(M, "easy")
def impute_with_flags(rng):
    setup, n, d = _nan_matrix(rng)
    task = rng.choice(["median_flags", "indicator_features", "stats", "all_features"])
    if task == "median_flags":
        return P(f"`X` ({n} × {d}) has NaNs. Median-impute it **and** append missing-indicator columns "
                 f"(`SimpleImputer(strategy='median', add_indicator=True)`). Return the result.", setup,
                 f"from {IMP} import SimpleImputer\nresult = SimpleImputer(strategy='median', add_indicator=True).fit_transform(X)",
                 require=["SimpleImputer", "add_indicator"])
    if task == "indicator_features":
        return P(f"`X` ({n} × {d}) has NaNs. Fit `MissingIndicator()` and return `features_`: the columns that had "
                 f"missing values (only those get a flag).", setup,
                 f"from {IMP} import MissingIndicator\nresult = MissingIndicator().fit(X).features_", require=["MissingIndicator"])
    if task == "stats":
        return P(f"`X` ({n} × {d}) has NaNs. Return `(mean statistics_, median statistics_)` of two fitted "
                 f"`SimpleImputer`s.", setup,
                 f"from {IMP} import SimpleImputer\nresult = (SimpleImputer().fit(X).statistics_, "
                 "SimpleImputer(strategy='median').fit(X).statistics_)", require=["SimpleImputer", "statistics_"])
    return P(f"`X` ({n} × {d}) has NaNs. `MissingIndicator(features='all')` flags every column, even complete ones. "
             f"Return the mask as integers (0/1).", setup,
             f"from {IMP} import MissingIndicator\nresult = MissingIndicator(features='all').fit_transform(X).astype(int)",
             require=["MissingIndicator"])


@template(M, "easy")
def imputers_by_hand(rng):
    setup, n, d = _nan_matrix(rng)
    v = rng.choice([-1.0, 0.0])
    imp = f"from {IMP} import SimpleImputer\n"
    items = [
        ("nanmean", "Mean imputation by hand: replace each NaN by its column's `np.nanmean`. Return `(by_hand, "
                    "from_sklearn)` (the latter from `SimpleImputer()`).",
         imp + "mine = np.where(np.isnan(X), np.nanmean(X, axis=0), X)\nresult = (mine, SimpleImputer().fit_transform(X))",
         {"require": ["SimpleImputer"]}),
        ("nanmedian", "Median imputation by hand with `np.nanmedian`. Return `(by_hand, from_sklearn)`.",
         imp + "mine = np.where(np.isnan(X), np.nanmedian(X, axis=0), X)\n"
               "result = (mine, SimpleImputer(strategy='median').fit_transform(X))", {"require": ["SimpleImputer"]}),
        ("most_frequent", "Most-frequent imputation by hand: for each column take the most common observed value "
                          "(smallest on ties: `np.unique` with counts, then `argmax`). Return `(by_hand, from_sklearn)`.",
         imp + "fill = []\nfor j in range(X.shape[1]):\n    vals, cnt = np.unique(X[~np.isnan(X[:, j]), j], return_counts=True)\n"
               "    fill.append(vals[cnt.argmax()])\nmine = np.where(np.isnan(X), np.array(fill), X)\n"
               "result = (mine, SimpleImputer(strategy='most_frequent').fit_transform(X))", {"require": ["SimpleImputer"]}),
        ("fill_value", f"Constant imputation by hand (`np.nan_to_num` or `np.where`) with {v}. Return `(by_hand, "
                       f"from_sklearn)` (`strategy='constant', fill_value={v}`).",
         imp + f"result = (np.where(np.isnan(X), {v}, X), SimpleImputer(strategy='constant', fill_value={v}).fit_transform(X))",
         {"require": ["SimpleImputer"]}),
        ("add_indicator", "Rebuild `SimpleImputer(add_indicator=True).fit_transform(X)` by hand: mean-impute, then append "
                          "one 0/1 column for each column that had a NaN. Return `(by_hand, from_sklearn)`.",
         imp + "miss = np.isnan(X)\ncols = miss.any(axis=0)\n"
               "mine = np.column_stack([np.where(miss, np.nanmean(X, axis=0), X), miss[:, cols].astype(float)])\n"
               "result = (mine, SimpleImputer(add_indicator=True).fit_transform(X))", {"require": ["SimpleImputer"]}),
    ]
    return catalog(rng, items, setup, f"`X` ({n} × {d}) has NaNs.", tol=1e-9)


@template(M, "easy")
def target_label_encoding(rng):
    pool = rng.choice([["no", "yes"], ["cat", "dog", "bird"], ["bronze", "silver", "gold", "platinum"]])
    y = [rng.choice(pool) for _ in range(rng.randint(6, 10))]
    y[:len(pool)] = pool
    rng.shuffle(y)
    setup = f"y = np.array({y})"
    task = rng.choice(["codes", "roundtrip", "binarizer", "counts"])
    if task == "codes":
        return P(f"`y` holds string class labels. Encode them with `LabelEncoder` and return `(codes, classes_)`.", setup,
                 f"from {PRE} import LabelEncoder\nle = LabelEncoder()\nresult = (le.fit_transform(y), le.classes_)",
                 require=["LabelEncoder"])
    if task == "roundtrip":
        return P(f"`y` holds string labels. A model predicts codes `[0, 1, 0]`; turn them back into labels with the "
                 f"`LabelEncoder` fitted on `y` and return them as a list.", setup,
                 f"from {PRE} import LabelEncoder\nle = LabelEncoder().fit(y)\nresult = le.inverse_transform([0, 1, 0]).tolist()",
                 require=["LabelEncoder", "inverse_transform"])
    if task == "binarizer":
        return P(f"`y` holds string labels. Return `LabelBinarizer().fit_transform(y)` and its `classes_`, as a tuple. "
                 f"(Two classes give one column; more give one per class.)", setup,
                 f"from {PRE} import LabelBinarizer\nlb = LabelBinarizer()\nresult = (lb.fit_transform(y), lb.classes_)",
                 require=["LabelBinarizer"])
    return P(f"`y` holds string labels. Return the class counts in `LabelEncoder` order: `np.bincount` of the codes.",
             setup, f"from {PRE} import LabelEncoder\nresult = np.bincount(LabelEncoder().fit_transform(y))",
             require=["LabelEncoder"])


@template(M, "easy")
def drop_options(rng):
    setup, cols, data, n = _cat_frame(rng)
    a, b = cols
    setup += f"\ncat['flag'] = np.where(np.arange({n}) % 3 == 0, 'yes', 'no')"
    task = rng.choice(["first", "if_binary", "names", "explicit"])
    if task == "first":
        return P(f"`cat` has columns `{a}`, `{b}` and a yes/no `flag`. Return the output names of "
                 f"`OneHotEncoder(drop='first')`: the first (alphabetical) level of every column is dropped.", setup,
                 f"from {PRE} import OneHotEncoder\nresult = OneHotEncoder(drop='first').fit(cat).get_feature_names_out()",
                 require=["OneHotEncoder", "drop"])
    if task == "if_binary":
        return P(f"`cat` has columns `{a}`, `{b}` and a yes/no `flag`. `drop='if_binary'` drops a level only for "
                 f"two-level columns. Return `(number of columns with drop=None, with 'if_binary', with 'first')`.",
                 setup, f"from {PRE} import OneHotEncoder\n"
                        "result = tuple(OneHotEncoder(drop=d, sparse_output=False).fit_transform(cat).shape[1] "
                        "for d in (None, 'if_binary', 'first'))", require=["OneHotEncoder", "drop"])
    if task == "names":
        return P(f"`cat` has columns `{a}`, `{b}` and a yes/no `flag`. Return the output names of "
                 f"`OneHotEncoder(drop='if_binary')`.", setup,
                 f"from {PRE} import OneHotEncoder\nresult = OneHotEncoder(drop='if_binary').fit(cat).get_feature_names_out()",
                 require=["OneHotEncoder", "drop"])
    return P(f"`cat` has columns `{a}`, `{b}` and a yes/no `flag`. `drop` may also name the level to drop per column. "
             f"Drop `{data[a][1]!r}` from `{a}`, `{data[b][0]!r}` from `{b}` and `'no'` from `flag`, and return the output "
             f"names.", setup,
             f"from {PRE} import OneHotEncoder\nresult = OneHotEncoder(drop=[{data[a][1]!r}, {data[b][0]!r}, 'no'])"
             ".fit(cat).get_feature_names_out()", require=["OneHotEncoder", "drop"])


# ================================================================= medium

def _table_with_missing_cat(rng, **kw):
    t = table(rng, **kw)
    c = rng.choice(t.cat)
    return t, c, t.setup + f"\ndf.loc[rng.random(n) < 0.08, {c!r}] = np.nan"


@template(M, "medium")
def encode_table_tasks(rng):
    t = table(rng, n=(80, 200))
    a, b = t.cat
    k = rng.choice([5, 10, 20])
    model = "LogisticRegression(max_iter=1000)" if t.kind == "clf" else "LinearRegression()"
    mimp = ("from sklearn.linear_model import LogisticRegression" if t.kind == "clf" else
            "from sklearn.linear_model import LinearRegression")
    items = [
        ("OneHotEncoder", f"One-hot encode the categorical columns `{a}` and `{b}` (`OneHotEncoder(sparse_output="
                          f"False)`) and return the shape of the result.",
         f"from {PRE} import OneHotEncoder\nresult = OneHotEncoder(sparse_output=False).fit_transform(df[[{a!r}, {b!r}]]).shape"),
        ("get_feature_names_out", f"One-hot encode `{a}` and `{b}` and return a dict output name → number of rows "
                                  f"with that level.",
         f"from {PRE} import OneHotEncoder\nohe = OneHotEncoder(sparse_output=False)\nZ = ohe.fit_transform(df[[{a!r}, {b!r}]])\n"
         "result = dict(zip(ohe.get_feature_names_out(), Z.sum(axis=0)))"),
        ("OneHotEncoder", f"Build a model matrix: the numeric columns `{t.num}` standardized (`StandardScaler`) "
                          f"followed by the one-hot columns of `{a}` and `{b}` (`drop='first'`). Fit `{model}` and "
                          f"return its training score.",
         f"from {PRE} import OneHotEncoder, StandardScaler\n{mimp}\nZ = OneHotEncoder(drop='first', sparse_output=False).fit_transform(df[[{a!r}, {b!r}]])\n"
         f"M = np.column_stack([StandardScaler().fit_transform(df[{t.num}]), Z])\nresult = {model}.fit(M, y).score(M, y)",
         {"require": ["OneHotEncoder"]}),
        ("OrdinalEncoder", f"Encode `{a}` and `{b}` with `OrdinalEncoder` and return the encoded first 5 rows.",
         f"from {PRE} import OrdinalEncoder\nresult = OrdinalEncoder().fit_transform(df[[{a!r}, {b!r}]])[:5]"),
        ("categories_", f"Which level of `{a}` is most common? Return it as a str, using the encoder's `categories_` "
                        f"and the column sums of the one-hot matrix.",
         f"from {PRE} import OneHotEncoder\nohe = OneHotEncoder(sparse_output=False)\nZ = ohe.fit_transform(df[[{a!r}]])\n"
         "result = str(ohe.categories_[0][Z.sum(axis=0).argmax()])"),
        ("groupby", f"Return the mean target per level of `{b}` (a dict level → mean of `y`): what a target encoder "
                    f"would start from.", f"result = pd.Series(y).groupby(df[{b!r}].to_numpy()).mean().to_dict()",
         {"require": []}),
        ("min_frequency", f"Group levels seen fewer than {k} times into one 'infrequent' column: "
                          f"`OneHotEncoder(min_frequency={k}, sparse_output=False)` on `{a}` and `{b}`. Return the "
                          f"output names.",
         f"from {PRE} import OneHotEncoder\nresult = OneHotEncoder(min_frequency={k}, sparse_output=False).fit(df[[{a!r}, "
         f"{b!r}]]).get_feature_names_out()"),
        ("set_output", f"One-hot encode `{a}` with `OneHotEncoder(sparse_output=False).set_output(transform='pandas')` "
                       f"and return the first 4 rows (a DataFrame).",
         f"from {PRE} import OneHotEncoder\nresult = OneHotEncoder(sparse_output=False).set_output(transform='pandas')"
         f".fit_transform(df[[{a!r}]]).head(4)", {"require": ["OneHotEncoder", "set_output"]}),
        ("OneHotEncoder", f"How does the target differ across `{a}`? Fit `{model}` on the one-hot columns of `{a}` alone "
                          f"(no drop) and return its training score.",
         f"from {PRE} import OneHotEncoder\n{mimp}\nZ = OneHotEncoder(sparse_output=False).fit_transform(df[[{a!r}]])\n"
         f"result = {model}.fit(Z, y).score(Z, y)"),
    ]
    return catalog(rng, items, t.setup, f"{t.desc}. Numeric columns: `{t.num}`; categorical: `{a}`, `{b}`.", tol=1e-6)


@template(M, "medium")
def impute_table_tasks(rng):
    t = table(rng, numeric_only=True, missing=2, n=(80, 200))
    c = t.missing[0]
    k = rng.choice([3, 5, 10])
    items = [
        ("statistics_", "Fit a median `SimpleImputer` on `df` and return a dict column → fill value.",
         f"from {IMP} import SimpleImputer\nimp = SimpleImputer(strategy='median').fit(df)\nresult = dict(zip(df.columns, imp.statistics_))"),
        ("SimpleImputer", "Mean-impute `df` with `SimpleImputer().set_output(transform='pandas')` and return the column "
                          "means of the result (unchanged by mean imputation).",
         f"from {IMP} import SimpleImputer\nresult = SimpleImputer().set_output(transform='pandas').fit_transform(df).mean()"),
        ("SimpleImputer", f"Imputation shrinks the spread. Return `(std before, std after)` of `{c}` for mean imputation "
                          f"(pandas `std`, which skips NaNs).",
         f"from {IMP} import SimpleImputer\nZ = SimpleImputer().set_output(transform='pandas').fit_transform(df)\n"
         f"result = (df[{c!r}].std(), Z[{c!r}].std())"),
        ("KNNImputer", f"Impute `df` with `KNNImputer(n_neighbors={k})` and return the values filled into `{c}` (in row "
                       f"order).",
         f"from {IMP} import KNNImputer\nZ = KNNImputer(n_neighbors={k}).fit_transform(df)\n"
         f"result = Z[df[{c!r}].isna().to_numpy(), list(df.columns).index({c!r})]"),
        ("IterativeImputer", f"Impute `df` with `IterativeImputer(random_state=0, max_iter=25)` and return the mean of "
                             f"the values it filled into `{c}`.",
         ITER + f"Z = IterativeImputer(random_state=0, max_iter=25).fit_transform(df)\n"
                f"result = Z[df[{c!r}].isna().to_numpy(), list(df.columns).index({c!r})].mean()", {"tol": 1e-4}),
        ("MissingIndicator", "Return a dict column → number of missing values, read from `MissingIndicator()` (its "
                             "`features_` say which columns the mask covers).",
         f"from {IMP} import MissingIndicator\nmi = MissingIndicator().fit(df)\nmask = mi.transform(df)\n"
         "result = {df.columns[j]: int(mask[:, i].sum()) for i, j in enumerate(mi.features_)}"),
        ("add_indicator", "Mean-impute with `add_indicator=True` and return the output names "
                          "(`get_feature_names_out`).",
         f"from {IMP} import SimpleImputer\nresult = SimpleImputer(add_indicator=True).fit(df).get_feature_names_out()"),
        ("SimpleImputer", "Which rows were incomplete? Return the number of rows with at least one NaN and the mean "
                          "target over those rows, as a tuple.",
         "miss = df.isna().any(axis=1).to_numpy()\nresult = (int(miss.sum()), y[miss].mean())", {"require": []}),
        ("KNNImputer", f"Compare imputers on `{c}`: return a dict `'mean'`, `'median'`, `'knn'` → the average value "
                       f"filled in (`SimpleImputer` strategies and `KNNImputer(n_neighbors={k})`).",
         f"from {IMP} import KNNImputer, SimpleImputer\nmiss = df[{c!r}].isna().to_numpy()\nj = list(df.columns).index({c!r})\n"
         f"imps = {{'mean': SimpleImputer(), 'median': SimpleImputer(strategy='median'), 'knn': KNNImputer(n_neighbors={k})}}\n"
         "result = {name: imp.fit_transform(df)[miss, j].mean() for name, imp in imps.items()}"),
    ]
    return catalog(rng, items, t.setup, f"{t.desc}; the columns `{t.missing}` have missing values.", tol=1e-6)


@template(M, "medium")
def rare_levels(rng):
    n, s = rng.randint(60, 150), rng.randint(0, 99)
    levels = rng.sample(["mx", "us", "fr", "de", "jp", "br", "in", "ng", "kr", "it"], rng.randint(6, 9))
    k = rng.choice([3, 5, 8])
    mc = rng.randint(3, 5)
    setup = (f"rng = np.random.default_rng({s})\np = rng.dirichlet(np.ones({len(levels)}) * 0.6)\n"
             f"cat = pd.DataFrame({{'country': rng.choice({levels}, {n}, p=p)}})\n"
             f"new = pd.DataFrame({{'country': [{levels[0]!r}, 'zz', {levels[-1]!r}]}})")
    head = f"`cat` has {n} rows of a `country` column with a few common and several rare levels; `new` has 3 rows."
    task = rng.choice(["min_freq", "max_cat", "infrequent", "unknown", "counts"])
    if task == "min_freq":
        return P(f"{head} Return the output names of `OneHotEncoder(min_frequency={k})` fitted on `cat`: levels seen "
                 f"fewer than {k} times share one `infrequent_sklearn` column.", setup,
                 f"from {PRE} import OneHotEncoder\nresult = OneHotEncoder(min_frequency={k}).fit(cat).get_feature_names_out()",
                 require=["OneHotEncoder", "min_frequency"])
    if task == "max_cat":
        return P(f"{head} Keep at most {mc} output columns: `OneHotEncoder(max_categories={mc})` keeps the {mc - 1} most "
                 f"frequent levels and groups the rest. Return its output names.", setup,
                 f"from {PRE} import OneHotEncoder\nresult = OneHotEncoder(max_categories={mc}).fit(cat).get_feature_names_out()",
                 require=["OneHotEncoder", "max_categories"])
    if task == "infrequent":
        return P(f"{head} Fit `OneHotEncoder(min_frequency={k})` and return the levels it treated as infrequent "
                 f"(`infrequent_categories_[0]`, or an empty list if none) as a list.", setup,
                 f"from {PRE} import OneHotEncoder\ninf = OneHotEncoder(min_frequency={k}).fit(cat).infrequent_categories_[0]\n"
                 "result = [] if inf is None else list(inf)", require=["infrequent_categories_"])
    if task == "unknown":
        return P(f"{head} With `handle_unknown='infrequent_if_exist'`, unseen levels land in the infrequent column. Fit "
                 f"`OneHotEncoder(min_frequency={k}, handle_unknown='infrequent_if_exist', sparse_output=False)` on "
                 f"`cat` and return the encoded `new`.", setup,
                 f"from {PRE} import OneHotEncoder\nresult = OneHotEncoder(min_frequency={k}, handle_unknown='infrequent_if_exist', "
                 "sparse_output=False).fit(cat).transform(new)", require=["OneHotEncoder", "handle_unknown"])
    return P(f"{head} Return the column sums of `OneHotEncoder(min_frequency={k}, sparse_output=False).fit_transform(cat)` "
             f"together with the output names, as a dict name → count.", setup,
             f"from {PRE} import OneHotEncoder\nohe = OneHotEncoder(min_frequency={k}, sparse_output=False)\n"
             "Z = ohe.fit_transform(cat)\nresult = dict(zip(ohe.get_feature_names_out(), Z.sum(axis=0)))",
             require=["OneHotEncoder", "min_frequency"])


@template(M, "medium")
def target_encoder_basics(rng):
    t = table(rng, n=(100, 250))
    a = rng.choice(t.cat)
    m = rng.choice([1.0, 5.0, 20.0])
    cv = rng.choice([2, 3, 5])
    kind = "continuous" if t.kind == "reg" else "binary"
    head = (f"{t.desc}. `TargetEncoder` (from `{PRE}`) replaces each level of `{a}` with a (shrunk) mean of the target "
            f"for that level.")
    imp = f"from {PRE} import TargetEncoder\n"
    task = rng.choice(["encodings", "smooth0", "crossfit", "smooth"])
    if task == "encodings":
        return P(f"{head} Fit `TargetEncoder(target_type='{kind}', smooth={m})` on `df[[{a!r}]]` and `y`; return a dict "
                 f"level → encoding (`categories_[0]` and `encodings_[0]`) and the global `target_mean_`, as a tuple.",
                 t.setup, imp + f"te = TargetEncoder(target_type='{kind}', smooth={m}).fit(df[[{a!r}]], y)\n"
                                "result = (dict(zip(te.categories_[0], te.encodings_[0])), te.target_mean_)",
                 require=["TargetEncoder"], tol=1e-9)
    if task == "smooth0":
        return P(f"{head} With `smooth=0.0` there is no shrinkage: the encodings are the plain per-level means. Return "
                 f"`(encodings_, groupby means)` for `TargetEncoder(target_type='{kind}', smooth=0.0)` fitted on "
                 f"`df[[{a!r}]]`, `y` (the second from pandas, in sorted level order).", t.setup,
                 imp + f"te = TargetEncoder(target_type='{kind}', smooth=0.0).fit(df[[{a!r}]], y)\n"
                       f"result = (te.encodings_[0], pd.Series(y).groupby(df[{a!r}].to_numpy()).mean().sort_index().to_numpy())",
                 require=["TargetEncoder"], tol=1e-9)
    if task == "crossfit":
        return P(f"{head} `fit_transform` is **cross-fitted** (each row encoded by a model that didn't see it), while "
                 f"`fit(...).transform(...)` uses all rows. With `TargetEncoder(target_type='{kind}', cv={cv}, "
                 f"shuffle=False)`, return `(fit_transform values, fit-then-transform values)` for `df[[{a!r}]]` "
                 f"(1-D arrays).", t.setup,
                 imp + f"te = TargetEncoder(target_type='{kind}', cv={cv}, shuffle=False)\n"
                       f"A = te.fit_transform(df[[{a!r}]], y)[:, 0]\nB = te.transform(df[[{a!r}]])[:, 0]\nresult = (A, B)",
                 require=["TargetEncoder", "cv"], tol=1e-9)
    return P(f"{head} Larger `smooth` pulls rare levels toward the global mean. For `smooth` in (0.0, {m}, 1000.0) fit "
             f"`TargetEncoder(target_type='{kind}', smooth=...)` on `df[[{a!r}]]`, `y` and return the spread (max − min) "
             f"of `encodings_[0]`, as a list.", t.setup,
             imp + f"result = [float(np.ptp(TargetEncoder(target_type='{kind}', smooth=s).fit(df[[{a!r}]], y).encodings_[0])) "
                   f"for s in (0.0, {m}, 1000.0)]", require=["TargetEncoder", "smooth"], tol=1e-9)


@template(M, "medium")
def imputation_accuracy(rng):
    d = synth_reg(rng, n=(80, 200), d=(3, 5))
    s = rng.randint(0, 99)
    frac = rng.choice([0.1, 0.2, 0.3])
    k = rng.choice([3, 5])
    setup = (d.setup + f"\nfull = np.column_stack([X, X[:, 0] * 2 + X[:, 1] + np.random.default_rng({s}).normal(0, 0.3, "
                       f"len(X))])\nmask = np.random.default_rng({s + 1}).random(full.shape) < {frac}\n"
                       "mask[:, :-1] = False\nholed = np.where(mask, np.nan, full)")
    head = (f"`full` is a complete matrix whose last column depends strongly on the first two; `holed` is `full` with "
            f"{frac:.0%} of the last column removed (`mask` marks them). Score an imputer by the RMSE between its "
            f"filled values and the true ones.")
    rmse = "lambda Z: float(np.sqrt(((Z[mask] - full[mask]) ** 2).mean()))"
    task = rng.choice(["dict", "best", "iter_gain", "knn_k"])
    if task == "dict":
        return P(f"{head} Return a dict `'mean'`, `'median'`, `'knn'`, `'iterative'` → RMSE for `SimpleImputer()`, "
                 f"`SimpleImputer(strategy='median')`, `KNNImputer(n_neighbors={k})` and `IterativeImputer(random_state=0, "
                 f"max_iter=25)`.", setup,
                 ITER + f"from {IMP} import KNNImputer, SimpleImputer\nrmse = {rmse}\n"
                        f"imps = {{'mean': SimpleImputer(), 'median': SimpleImputer(strategy='median'), 'knn': "
                        f"KNNImputer(n_neighbors={k}), 'iterative': IterativeImputer(random_state=0, max_iter=25)}}\n"
                        "result = {name: rmse(imp.fit_transform(holed)) for name, imp in imps.items()}",
                 require=["KNNImputer", "IterativeImputer"], tol=1e-4)
    if task == "best":
        return P(f"{head} Among `SimpleImputer()`, `KNNImputer(n_neighbors={k})` and `IterativeImputer(random_state=0, "
                 f"max_iter=25)`, return the name (`'mean'`, `'knn'` or `'iterative'`) with the lowest RMSE.", setup,
                 ITER + f"from {IMP} import KNNImputer, SimpleImputer\nrmse = {rmse}\n"
                        f"imps = {{'mean': SimpleImputer(), 'knn': KNNImputer(n_neighbors={k}), 'iterative': "
                        "IterativeImputer(random_state=0, max_iter=25)}\n"
                        "scores = {name: rmse(imp.fit_transform(holed)) for name, imp in imps.items()}\n"
                        "result = min(scores, key=scores.get)", require=["IterativeImputer"])
    if task == "iter_gain":
        return P(f"{head} Return `(mean_rmse, iterative_rmse)` for `SimpleImputer()` and `IterativeImputer(random_state=0, "
                 f"max_iter=25)`: modelling the column from the others pays off here.", setup,
                 ITER + f"from {IMP} import SimpleImputer\nrmse = {rmse}\n"
                        "result = (rmse(SimpleImputer().fit_transform(holed)), rmse(IterativeImputer(random_state=0, "
                        "max_iter=25).fit_transform(holed)))", require=["IterativeImputer", "SimpleImputer"], tol=1e-4)
    return P(f"{head} Return the RMSE of `KNNImputer(n_neighbors=k)` for k in (1, 3, 10) as a list.", setup,
             f"from {IMP} import KNNImputer\nrmse = {rmse}\nresult = [rmse(KNNImputer(n_neighbors=k).fit_transform(holed)) "
             "for k in (1, 3, 10)]", require=["KNNImputer"])


@template(M, "medium")
def tag_matrices(rng):
    s, n = rng.randint(0, 99), rng.randint(8, 20)
    pool = rng.sample(["python", "sql", "ml", "stats", "viz", "cloud", "rust", "web"], rng.randint(4, 6))
    setup = (f"rng = np.random.default_rng({s})\npool = {pool}\n"
             f"tags = [sorted(rng.choice(pool, rng.integers(1, 4), replace=False).tolist()) for _ in range({n})]")
    head = f"`tags` lists the tags of {n} job posts (1 to 3 each, from `pool`)."
    imp = f"from {PRE} import MultiLabelBinarizer\n"
    items = [
        ("MultiLabelBinarizer", "Return the tag that appears in the most posts (from `classes_` and the column sums; "
                                "first in alphabetical order on ties).",
         imp + "mlb = MultiLabelBinarizer()\nY = mlb.fit_transform(tags)\nresult = str(mlb.classes_[Y.sum(axis=0).argmax()])"),
        ("MultiLabelBinarizer", "Return the tag co-occurrence matrix `Y.T @ Y` of the binarized tags.",
         imp + "Y = MultiLabelBinarizer().fit_transform(tags)\nresult = Y.T @ Y"),
        ("classes", "Fix the column order with `MultiLabelBinarizer(classes=pool)` and return the first 3 rows.",
         imp + "result = MultiLabelBinarizer(classes=pool).fit_transform(tags)[:3]"),
        ("MultiLabelBinarizer", "Which two different tags appear together most often? Return them as a sorted list (ties: "
                                "the first pair in row-major order of the co-occurrence matrix).",
         imp + "mlb = MultiLabelBinarizer()\nY = mlb.fit_transform(tags)\nC = Y.T @ Y\nnp.fill_diagonal(C, -1)\n"
               "i, j = np.unravel_index(C.argmax(), C.shape)\nresult = sorted([str(mlb.classes_[i]), str(mlb.classes_[j])])"),
        ("sparse_output", "`MultiLabelBinarizer(sparse_output=True)` returns a SciPy sparse matrix. Return its `nnz` "
                          "(the total number of tags) and shape, as a tuple.",
         imp + "Y = MultiLabelBinarizer(sparse_output=True).fit_transform(tags)\nresult = (Y.nnz, Y.shape)"),
        ("MultiLabelBinarizer", "Return the number of tags per post (row sums of the binarized matrix).",
         imp + "result = MultiLabelBinarizer().fit_transform(tags).sum(axis=1)"),
    ]
    return catalog(rng, items, setup, head)


@template(M, "medium")
def encoder_attributes(rng):
    t = table(rng, n=(60, 150))
    a, b = t.cat
    lv = t.levels[a]
    extra = rng.choice(["zzz", "other", "unknown"])
    items = [
        ("drop_idx_", f"Fit `OneHotEncoder(drop='first')` on `df[[{a!r}, {b!r}]]` and return the dropped level of each "
                      f"column (`categories_[i][drop_idx_[i]]`), as a list.",
         f"from {PRE} import OneHotEncoder\nohe = OneHotEncoder(drop='first').fit(df[[{a!r}, {b!r}]])\n"
         "result = [str(c[i]) for c, i in zip(ohe.categories_, ohe.drop_idx_)]"),
        ("inverse_transform", f"Fit `OneHotEncoder(sparse_output=False)` on `df[[{a!r}, {b!r}]]`, encode the first 3 rows "
                              f"and decode them back with `inverse_transform`; return the decoded array as a list of "
                              f"lists.",
         f"from {PRE} import OneHotEncoder\nohe = OneHotEncoder(sparse_output=False).fit(df[[{a!r}, {b!r}]])\n"
         f"result = ohe.inverse_transform(ohe.transform(df[[{a!r}, {b!r}]].iloc[:3])).tolist()"),
        ("categories_", f"Fit an `OrdinalEncoder` on `df[[{a!r}, {b!r}]]` and return a dict column → list of levels "
                        f"(`feature_names_in_` with `categories_`).",
         f"from {PRE} import OrdinalEncoder\noe = OrdinalEncoder().fit(df[[{a!r}, {b!r}]])\n"
         "result = {str(n): [str(v) for v in c] for n, c in zip(oe.feature_names_in_, oe.categories_)}"),
        ("dtype", f"`OneHotEncoder(dtype=int, sparse_output=False)` outputs integers. Encode `df[[{a!r}]]` and return "
                  f"`(dtype name, column sums)`.",
         f"from {PRE} import OneHotEncoder\nZ = OneHotEncoder(dtype=int, sparse_output=False).fit_transform(df[[{a!r}]])\n"
         "result = (Z.dtype.name, Z.sum(axis=0))"),
        ("categories", f"Declare the levels up front, including one that never occurs: `OneHotEncoder(categories="
                       f"[{lv + [extra]}], sparse_output=False)` on `df[[{a!r}]]`. Return the column sums (the unused "
                       f"level's column is all zeros).",
         f"from {PRE} import OneHotEncoder\nresult = OneHotEncoder(categories=[{lv + [extra]}], sparse_output=False)"
         f".fit_transform(df[[{a!r}]]).sum(axis=0)"),
        ("OneHotEncoder", f"By default `OneHotEncoder` returns a SciPy sparse matrix. Encode `df[[{a!r}, {b!r}]]` and "
                          f"return `(type name, nnz, shape)`: exactly one 1 per row and column.",
         f"from {PRE} import OneHotEncoder\nZ = OneHotEncoder().fit_transform(df[[{a!r}, {b!r}]])\n"
         "result = (type(Z).__name__, Z.nnz, Z.shape)"),
        ("TargetEncoder", f"Fit `TargetEncoder(target_type='{'continuous' if t.kind == 'reg' else 'binary'}', smooth=5.0)` "
                          f"on both columns `df[[{a!r}, {b!r}]]` and return the number of encodings per column (a list) "
                          f"and the shape of `transform`, as a tuple.",
         f"from {PRE} import TargetEncoder\nte = TargetEncoder(target_type='{'continuous' if t.kind == 'reg' else 'binary'}', "
         f"smooth=5.0).fit(df[[{a!r}, {b!r}]], y)\nresult = ([len(e) for e in te.encodings_], te.transform(df[[{a!r}, {b!r}]]).shape)"),
        ("n_features_in_", f"Fit a `OneHotEncoder` on `df[[{a!r}, {b!r}]]` and return `(n_features_in_, number of output "
                           f"columns)`.",
         f"from {PRE} import OneHotEncoder\nohe = OneHotEncoder().fit(df[[{a!r}, {b!r}]])\n"
         "result = (ohe.n_features_in_, len(ohe.get_feature_names_out()))"),
    ]
    return catalog(rng, items, t.setup, f"{t.desc}; `{a}` and `{b}` are categorical.")


# ================================================================= hard

@template(M, "hard")
def target_encoding_by_hand(rng):
    t = table(rng, kind="reg", n=(60, 150))
    a = rng.choice(t.cat)
    m = rng.choice([1.0, 4.0, 10.0])
    cv = rng.choice([2, 3, 4])
    task = rng.choice(["shrunk", "crossfit", "unknown"])
    imp = f"from {PRE} import TargetEncoder\n"
    if task == "shrunk":
        return P(f"{t.desc}. With a numeric `smooth = m`, `TargetEncoder` encodes level `c` as `(sum_c + m · mean) / "
                 f"(n_c + m)` (sum and count of the targets of level `c`, global target mean). Compute the encodings of "
                 f"`{a}` by hand (levels in sorted order) with m = {m} and return `(by_hand, encodings_[0])` from "
                 f"`TargetEncoder(target_type='continuous', smooth={m})` fitted on `df[[{a!r}]]`, `y`.", t.setup,
                 imp + f"g = pd.Series(y).groupby(df[{a!r}].to_numpy())\nmine = ((g.sum() + {m} * y.mean()) / (g.count() + {m}))"
                       f".sort_index().to_numpy()\nte = TargetEncoder(target_type='continuous', smooth={m}).fit(df[[{a!r}]], y)\n"
                       "result = (mine, te.encodings_[0])", require=["TargetEncoder"], tol=1e-9)
    if task == "crossfit":
        return P(f"{t.desc}. Reproduce the cross-fitting of `TargetEncoder(target_type='continuous', smooth={m}, cv={cv}, "
                 f"shuffle=False).fit_transform(df[[{a!r}]], y)`: for each fold of `KFold(n_splits={cv})`, fit the "
                 f"encoder on the other folds and transform the fold. Return `(by_hand, fit_transform)` as 1-D arrays.",
                 t.setup,
                 imp + f"from sklearn.model_selection import KFold\nX = df[[{a!r}]]\nmine = np.empty(len(X))\n"
                       f"for tr, te in KFold(n_splits={cv}).split(X):\n"
                       f"    enc = TargetEncoder(target_type='continuous', smooth={m}).fit(X.iloc[tr], y[tr])\n"
                       "    mine[te] = enc.transform(X.iloc[te])[:, 0]\n"
                       f"full = TargetEncoder(target_type='continuous', smooth={m}, cv={cv}, shuffle=False).fit_transform(X, y)[:, 0]\n"
                       "result = (mine, full)", require=["TargetEncoder", "KFold"], tol=1e-9)
    return P(f"{t.desc}. A level never seen in training is encoded with the global target mean. Fit "
             f"`TargetEncoder(target_type='continuous', smooth={m})` on `df[[{a!r}]]`, `y` and transform "
             f"`pd.DataFrame({{{a!r}: ['never-seen', df[{a!r}].iloc[0]]}})`; return `(encoded values, target_mean_)`.",
             t.setup, imp + f"te = TargetEncoder(target_type='continuous', smooth={m}).fit(df[[{a!r}]], y)\n"
                            f"new = pd.DataFrame({{{a!r}: ['never-seen', df[{a!r}].iloc[0]]}})\n"
                            "result = (te.transform(new)[:, 0], te.target_mean_)", require=["TargetEncoder"], tol=1e-9)


@template(M, "hard")
def leaky_imputation(rng):
    t = table(rng, numeric_only=True, missing=2, n=(100, 220))
    s = rng.randint(0, 99)
    c = t.missing[0]
    split = (f"from sklearn.model_selection import train_test_split\ndf_tr, df_te, y_tr, y_te = train_test_split(df, y, "
             f"test_size=0.3, random_state={s})\n")
    task = rng.choice(["fixed", "fills", "model"])
    if task == "fixed":
        return P(f"{t.desc}; `{t.missing}` have NaNs. This leaks: the imputer learns from the test rows.\n\n```\n"
                 f"filled = SimpleImputer(strategy='median').fit_transform(df)\n```\n\nSplit first with "
                 f"`train_test_split(df, y, test_size=0.3, random_state={s})`, fit the imputer on the training part and "
                 f"return the imputed **test** matrix.", t.setup,
                 split + f"from {IMP} import SimpleImputer\nresult = SimpleImputer(strategy='median').fit(df_tr).transform(df_te)",
                 require=["SimpleImputer", "train_test_split"])
    if task == "fills":
        return P(f"{t.desc}; `{t.missing}` have NaNs. Split with `train_test_split(df, y, test_size=0.3, "
                 f"random_state={s})`. Return `(leaky, honest)`: the median fill values (`statistics_`) learned from all "
                 f"of `df` and from the training part only.", t.setup,
                 split + f"from {IMP} import SimpleImputer\nresult = (SimpleImputer(strategy='median').fit(df).statistics_, "
                         "SimpleImputer(strategy='median').fit(df_tr).statistics_)", require=["SimpleImputer"])
    model = "LogisticRegression(max_iter=1000)" if t.kind == "clf" else "LinearRegression()"
    mimp = "LogisticRegression" if t.kind == "clf" else "LinearRegression"
    return P(f"{t.desc}; `{t.missing}` have NaNs. Split with `train_test_split(df, y, test_size=0.3, random_state={s})`, "
             f"standardize after mean-imputing (`SimpleImputer` then `StandardScaler`, both fitted on the training part), "
             f"fit `{model}` and return its test score.", t.setup,
             split + f"from {IMP} import SimpleImputer\nfrom {PRE} import StandardScaler\n"
                     f"from sklearn.linear_model import {mimp}\nimp = SimpleImputer().fit(df_tr)\n"
                     "sc = StandardScaler().fit(imp.transform(df_tr))\nA, B = sc.transform(imp.transform(df_tr)), sc.transform(imp.transform(df_te))\n"
                     f"result = {model}.fit(A, y_tr).score(B, y_te)", require=["SimpleImputer", "StandardScaler"])


@template(M, "hard")
def knn_imputer_by_hand(rng):
    setup, n, d = _nan_matrix(rng, r=(7, 10), c=(3, 4), floats=True)
    k = rng.choice([1, 2, 3])
    task = rng.choice(["distances", "one_cell", "uniform_vs_distance"])
    if task == "distances":
        return P(f"`X` ({n} × {d}) has NaNs. `KNNImputer` measures distance with `nan_euclidean_distances` (from "
                 f"`sklearn.metrics.pairwise`): it skips coordinates missing in either row and scales up by "
                 f"`features / present`. For the first row with a NaN, compute its distances to every row by hand and "
                 f"return `(by_hand, nan_euclidean_distances(X[[i]], X)[0])` (both 1-D; NaN where no coordinate is "
                 f"shared).", setup,
                 "from sklearn.metrics.pairwise import nan_euclidean_distances\ni = int(np.flatnonzero(np.isnan(X).any(axis=1))[0])\n"
                 "both = ~np.isnan(X[i]) & ~np.isnan(X)\ndiff = np.where(both, X - X[i], 0.0)\ncnt = both.sum(axis=1)\n"
                 "mine = np.sqrt(np.where(cnt > 0, X.shape[1] / np.maximum(cnt, 1) * (diff ** 2).sum(axis=1), np.nan))\n"
                 "result = (mine, nan_euclidean_distances(X[[i]], X)[0])", require=["nan_euclidean_distances"], tol=1e-9)
    if task == "one_cell":
        return P(f"`X` ({n} × {d}) has NaNs. Reproduce `KNNImputer(n_neighbors={k})` for the **first** missing cell (in "
                 f"row-major order) at row `i`, column `j`: among rows where column `j` is observed (excluding row `i`), "
                 f"take the {k} with the smallest `nan_euclidean_distances` to row `i` (stable order on ties) and average "
                 f"their column `j`. Return `(by_hand, KNNImputer value)`.", setup,
                 f"from sklearn.metrics.pairwise import nan_euclidean_distances\nfrom {IMP} import KNNImputer\n"
                 "i, j = map(int, np.argwhere(np.isnan(X))[0])\ndist = nan_euclidean_distances(X[[i]], X)[0]\n"
                 "donors = np.flatnonzero(~np.isnan(X[:, j]) & (np.arange(len(X)) != i))\n"
                 f"near = donors[np.argsort(dist[donors], kind='stable')[:{k}]]\n"
                 f"result = (X[near, j].mean(), KNNImputer(n_neighbors={k}).fit_transform(X)[i, j])",
                 require=["KNNImputer", "nan_euclidean_distances"], tol=1e-9)
    return P(f"`X` ({n} × {d}) has NaNs. `weights='distance'` weights each neighbour by 1 / distance. Return the imputed "
             f"matrices of `KNNImputer(n_neighbors={k + 1})` and `KNNImputer(n_neighbors={k + 1}, weights='distance')`, as a "
             f"tuple.", setup,
             f"from {IMP} import KNNImputer\nresult = (KNNImputer(n_neighbors={k + 1}).fit_transform(X), "
             f"KNNImputer(n_neighbors={k + 1}, weights='distance').fit_transform(X))", require=["KNNImputer", "weights"],
             tol=1e-9)


@template(M, "hard")
def informative_missingness(rng):
    n, s = rng.randint(200, 400), rng.randint(0, 999)
    setup = (f"rng = np.random.default_rng({s})\nX = rng.normal(size=({n}, 3))\ny = (X[:, 0] + rng.normal(0, 1, {n}) > 0)"
             f".astype(int)\n# income is more often missing for class 1\nmiss = rng.random({n}) < np.where(y == 1, 0.5, 0.1)\n"
             f"X[miss, 2] = np.nan\nX_tr, X_te, y_tr, y_te = X[::2], X[1::2], y[::2], y[1::2]")
    head = (f"`X` ({n} × 3): column 2 is missing far more often for class 1, so *being missing* is a signal. "
            f"`X_tr, y_tr` are the even rows, `X_te, y_te` the odd rows. Model: `LogisticRegression()`.")
    fit = (f"from {IMP} import SimpleImputer\nfrom sklearn.linear_model import LogisticRegression\n")
    task = rng.choice(["both", "with_flag", "coef"])
    if task == "both":
        return P(f"{head} Mean-impute (fitted on `X_tr`) with and without `add_indicator=True`, fit the model and return "
                 f"`(without_flag, with_flag)` test accuracies.", setup,
                 fit + "out = []\nfor flag in (False, True):\n    imp = SimpleImputer(add_indicator=flag).fit(X_tr)\n"
                       "    out.append(LogisticRegression().fit(imp.transform(X_tr), y_tr).score(imp.transform(X_te), y_te))\n"
                       "result = tuple(out)", require=["SimpleImputer", "add_indicator"])
    if task == "with_flag":
        return P(f"{head} Mean-impute with `add_indicator=True` (fitted on `X_tr`), fit the model and return its test "
                 f"accuracy.", setup,
                 fit + "imp = SimpleImputer(add_indicator=True).fit(X_tr)\n"
                       "result = LogisticRegression().fit(imp.transform(X_tr), y_tr).score(imp.transform(X_te), y_te)",
                 require=["SimpleImputer", "add_indicator"])
    return P(f"{head} Mean-impute with `add_indicator=True` (fitted on `X_tr`), fit the model on `X_tr` and return the "
             f"coefficient of the indicator column (the last one): positive, since missing points to class 1.", setup,
             fit + "imp = SimpleImputer(add_indicator=True).fit(X_tr)\nresult = LogisticRegression().fit(imp.transform(X_tr), "
                   "y_tr).coef_[0, -1]", require=["SimpleImputer", "add_indicator"], tol=1e-5)


@template(M, "hard")
def train_test_encoding(rng):
    t = table(rng, kind="clf", n=(120, 250))
    a, b = t.cat
    s = rng.randint(0, 99)
    split = (f"from sklearn.model_selection import train_test_split\ndf_tr, df_te, y_tr, y_te = train_test_split(df, y, "
             f"test_size=0.3, random_state={s})\n")
    task = rng.choice(["dummies_bug", "separate_fit_bug", "model", "rare_test"])
    if task == "dummies_bug":
        return P(f"{t.desc}. Split with `train_test_split(df, y, test_size=0.3, random_state={s})`. `pd.get_dummies` "
                 f"on train and test separately can give different columns. Instead fit one `OneHotEncoder("
                 f"handle_unknown='ignore')` on the training part's `{a}` and `{b}` and return `(train columns, test "
                 f"columns)` of the encoded matrices (always equal).", t.setup,
                 split + f"from {PRE} import OneHotEncoder\nohe = OneHotEncoder(handle_unknown='ignore').fit(df_tr[[{a!r}, {b!r}]])\n"
                         f"result = (ohe.transform(df_tr[[{a!r}, {b!r}]]).shape[1], ohe.transform(df_te[[{a!r}, {b!r}]]).shape[1])",
                 require=["OneHotEncoder"])
    if task == "separate_fit_bug":
        return P(f"{t.desc}. Split with `train_test_split(df, y, test_size=0.3, random_state={s})`. This is wrong: "
                 f"the test encoder may number the levels differently.\n\n```\nA = OrdinalEncoder().fit_transform(df_tr[[{a!r}]])\n"
                 f"B = OrdinalEncoder().fit_transform(df_te[[{a!r}]])\n```\n\nFix it (one encoder fitted on the training "
                 f"part, with `handle_unknown='use_encoded_value', unknown_value=-1`) and return the encoded test column "
                 f"as a 1-D array.", t.setup,
                 split + f"from {PRE} import OrdinalEncoder\noe = OrdinalEncoder(handle_unknown='use_encoded_value', "
                         f"unknown_value=-1).fit(df_tr[[{a!r}]])\nresult = oe.transform(df_te[[{a!r}]])[:, 0]",
                 require=["OrdinalEncoder"])
    if task == "model":
        return P(f"{t.desc}. Split with `train_test_split(df, y, test_size=0.3, random_state={s})`. Build features: "
                 f"the numeric columns `{t.num}` standardized (`StandardScaler`) plus `{a}` and `{b}` one-hot encoded "
                 f"(`OneHotEncoder(handle_unknown='ignore', sparse_output=False)`), all fitted on the training part; fit "
                 f"`LogisticRegression(max_iter=1000)` and return the test accuracy.", t.setup,
                 split + f"from {PRE} import OneHotEncoder, StandardScaler\nfrom sklearn.linear_model import LogisticRegression\n"
                         f"sc = StandardScaler().fit(df_tr[{t.num}])\nohe = OneHotEncoder(handle_unknown='ignore', "
                         f"sparse_output=False).fit(df_tr[[{a!r}, {b!r}]])\n"
                         f"feats = lambda d: np.column_stack([sc.transform(d[{t.num}]), ohe.transform(d[[{a!r}, {b!r}]])])\n"
                         "result = LogisticRegression(max_iter=1000).fit(feats(df_tr), y_tr).score(feats(df_te), y_te)",
                 require=["OneHotEncoder", "StandardScaler"])
    level = t.levels[a][0]
    return P(f"{t.desc}. Pretend level `{level!r}` of `{a}` never reached training: drop those rows from the training "
             f"part of `train_test_split(df, y, test_size=0.3, random_state={s})`, fit `OneHotEncoder(handle_unknown="
             f"'ignore', sparse_output=False)` on the remaining training `{a}`, and return how many **test** rows encode "
             f"to all zeros (scikit-learn warns about the unknown level).", t.setup,
             split + f"from {PRE} import OneHotEncoder\nkeep = (df_tr[{a!r}] != {level!r}).to_numpy()\n"
                     f"ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(df_tr.loc[keep, [{a!r}]])\n"
                     f"result = int((ohe.transform(df_te[[{a!r}]]).sum(axis=1) == 0).sum())", require=["OneHotEncoder"])


@template(M, "hard")
def mixed_preprocessing_tasks(rng):
    t, c, setup = _table_with_missing_cat(rng, missing=1, n=(100, 220))
    num_miss = t.missing[0]
    others = [x for x in t.cat if x != c]
    o = others[0]
    items = [
        ("SimpleImputer", f"`{c}` (text) has NaNs too. Fill them with the most frequent level using "
                          f"`SimpleImputer(strategy='most_frequent')` on `df[[{c!r}]]` and return the fill value.",
         f"from {IMP} import SimpleImputer\nresult = str(SimpleImputer(strategy='most_frequent').fit(df[[{c!r}]]).statistics_[0])"),
        ("fill_value", f"Make missingness its own level: `SimpleImputer(strategy='constant', fill_value='missing')` on "
                       f"`df[[{c!r}]]`, then one-hot encode with `OneHotEncoder(sparse_output=False)`. Return the output "
                       f"names.",
         f"from {IMP} import SimpleImputer\nfrom {PRE} import OneHotEncoder\n"
         f"filled = SimpleImputer(strategy='constant', fill_value='missing').fit_transform(df[[{c!r}]])\n"
         f"result = OneHotEncoder().fit(pd.DataFrame(filled, columns=[{c!r}])).get_feature_names_out()"),
        ("OneHotEncoder", f"`OneHotEncoder` treats NaN as a level of its own. Encode `df[[{c!r}]]` with "
                          f"`OneHotEncoder(sparse_output=False)` and return the output names.",
         f"from {PRE} import OneHotEncoder\nresult = OneHotEncoder(sparse_output=False).fit(df[[{c!r}]]).get_feature_names_out()"),
        ("encoded_missing_value", f"`OrdinalEncoder(encoded_missing_value=-1)` gives NaNs their own code. Encode "
                                  f"`df[[{c!r}]]` and return how many rows got -1, and the number of distinct codes, as "
                                  f"a tuple.",
         f"from {PRE} import OrdinalEncoder\nz = OrdinalEncoder(encoded_missing_value=-1).fit_transform(df[[{c!r}]])[:, 0]\n"
         "result = (int((z == -1).sum()), len(np.unique(z)))"),
        ("KNNImputer", f"Impute the numeric column `{num_miss}` with `KNNImputer(n_neighbors=5)` applied to the numeric "
                       f"columns `{t.num}` only, and return the mean of the filled values.",
         f"from {IMP} import KNNImputer\nZ = KNNImputer(n_neighbors=5).fit_transform(df[{t.num}])\n"
         f"miss = df[{num_miss!r}].isna().to_numpy()\nresult = Z[miss, {t.num.index(num_miss)}].mean()"),
        ("SimpleImputer", f"Impute numeric and text columns separately (median for `{t.num}`, most frequent for "
                          f"`{c}`), then return the number of NaNs left in the combined result (0) and the number of "
                          f"rows changed in total, as a tuple.",
         f"from {IMP} import SimpleImputer\nN = SimpleImputer(strategy='median').fit_transform(df[{t.num}])\n"
         f"C = SimpleImputer(strategy='most_frequent').fit_transform(df[[{c!r}]])\n"
         f"left = int(np.isnan(N).sum()) + int(pd.isna(C).sum())\nresult = (left, int(df[{t.num} + [{c!r}]].isna().any(axis=1).sum()))"),
        ("OneHotEncoder", f"Cross the two categorical columns: one-hot encode `{c}` (NaN as its own level) and `{o}` and "
                          f"return the number of output columns, and the number of distinct (`{c}`, `{o}`) pairs that "
                          f"occur, as a tuple.",
         f"from {PRE} import OneHotEncoder\nw = OneHotEncoder(sparse_output=False).fit_transform(df[[{c!r}, {o!r}]]).shape[1]\n"
         f"result = (w, len(df[[{c!r}, {o!r}]].astype(str).drop_duplicates()))"),
    ]
    return catalog(rng, items, setup, f"{t.desc}; the numeric column `{num_miss}` and the text column `{c}` have missing "
                                      f"values.", tol=1e-6)


@template(M, "hard")
def encoding_workflows(rng):
    t = table(rng, kind="reg", missing=1, n=(120, 250), only=["houses", "servers", "experiments", "bikes", "phones"])
    a, b = t.cat
    num = t.num
    nm = t.missing[0]
    s = rng.randint(0, 99)
    lv = t.levels[a]
    split = (f"from sklearn.model_selection import train_test_split\ndf_tr, df_te, y_tr, y_te = train_test_split(df, y, "
             f"test_size=0.3, random_state={s})\n")
    items = [
        ("OneHotEncoder", f"A category's effect need not be monotone in its code. Fit `LinearRegression` on `{a}` alone, "
                          f"encoded with `OrdinalEncoder()` and with `OneHotEncoder(sparse_output=False)`, and return "
                          f"the two training R² values as `(ordinal, onehot)`.",
         f"from {PRE} import OneHotEncoder, OrdinalEncoder\nfrom sklearn.linear_model import LinearRegression\n"
         f"A = OrdinalEncoder().fit_transform(df[[{a!r}]])\nB = OneHotEncoder(sparse_output=False).fit_transform(df[[{a!r}]])\n"
         "result = (LinearRegression().fit(A, y).score(A, y), LinearRegression().fit(B, y).score(B, y))"),
        ("drop", f"With `drop='first'`, the coefficient of a level is its mean difference from the dropped level. Fit "
                 f"`LinearRegression` on `OneHotEncoder(drop='first', sparse_output=False)` of `{a}` alone and return "
                 f"`(coef_, by_hand)` where `by_hand` is each kept level's mean `y` minus the first level's mean `y`.",
         f"from {PRE} import OneHotEncoder\nfrom sklearn.linear_model import LinearRegression\n"
         f"Z = OneHotEncoder(drop='first', sparse_output=False).fit_transform(df[[{a!r}]])\n"
         f"means = pd.Series(y).groupby(df[{a!r}].to_numpy()).mean().sort_index().to_numpy()\n"
         "result = (LinearRegression().fit(Z, y).coef_, means[1:] - means[0])"),
        ("handle_unknown", f"Fit `LinearRegression` on `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` of "
                           f"`{a}` and predict a never-seen level: it gets all zeros, so the prediction is the intercept. "
                           f"Return `(prediction, intercept_)` (scikit-learn warns about the unknown level).",
         f"from {PRE} import OneHotEncoder\nfrom sklearn.linear_model import LinearRegression\n"
         f"ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(df[[{a!r}]])\n"
         f"reg = LinearRegression().fit(ohe.transform(df[[{a!r}]]), y)\n"
         f"p = reg.predict(ohe.transform(pd.DataFrame({{{a!r}: ['never-seen']}})))[0]\nresult = (p, reg.intercept_)"),
        ("dropna", f"Two ways to handle the NaNs in `{nm}`: drop incomplete rows, or mean-impute with indicators. Split "
                   f"with `train_test_split(df[{num}], y, test_size=0.3, random_state={s})`; return `(rows kept for "
                   f"training when dropping, test R² of LinearRegression with SimpleImputer(add_indicator=True))` "
                   f"(imputer fitted on the training part).",
         f"from sklearn.model_selection import train_test_split\nfrom {IMP} import SimpleImputer\n"
         "from sklearn.linear_model import LinearRegression\n"
         f"X_tr, X_te, y_tr, y_te = train_test_split(df[{num}], y, test_size=0.3, random_state={s})\n"
         "imp = SimpleImputer(add_indicator=True).fit(X_tr)\n"
         "result = (int(X_tr.notna().all(axis=1).sum()), LinearRegression().fit(imp.transform(X_tr), y_tr).score(imp.transform(X_te), y_te))",
         {"require": ["SimpleImputer", "add_indicator"]}),
        ("n_iter_", f"Fit `IterativeImputer(random_state=0, max_iter=30, tol=1e-3)` on the numeric columns `{num}` and "
                    f"return `n_iter_`: how many rounds it needed.",
         ITER + f"result = IterativeImputer(random_state=0, max_iter=30, tol=1e-3).fit(df[{num}]).n_iter_"),
        ("KNNImputer", f"Distances need comparable scales. Return the values `KNNImputer(n_neighbors=5)` fills into `{nm}` "
                       f"on the raw numeric columns `{num}` and on the standardized ones (`StandardScaler`, which ignores "
                       f"NaNs; map the filled values back with `inverse_transform`), as a tuple of two arrays.",
         f"from {IMP} import KNNImputer\nfrom {PRE} import StandardScaler\nX = df[{num}]\nj = {num.index(nm)}\n"
         "miss = X.iloc[:, j].isna().to_numpy()\nraw = KNNImputer(n_neighbors=5).fit_transform(X)[miss, j]\n"
         "sc = StandardScaler().fit(X)\nZ = sc.inverse_transform(KNNImputer(n_neighbors=5).fit_transform(sc.transform(X)))\n"
         "result = (raw, Z[miss, j])"),
        ("TargetEncoder", f"Compare encodings of `{a}` and `{b}` on held-out data: split with `train_test_split(df, y, "
                          f"test_size=0.3, random_state={s})`, encode both columns with `OneHotEncoder(handle_unknown="
                          f"'ignore', sparse_output=False)` or with `TargetEncoder(target_type='continuous', cv=3, "
                          f"shuffle=False)` (fitted on the training part: `fit_transform` there, `transform` on test), "
                          f"fit `LinearRegression` and return the test R² values as `(onehot, target)`.",
         split + f"from {PRE} import OneHotEncoder, TargetEncoder\nfrom sklearn.linear_model import LinearRegression\n"
                 f"cols = [{a!r}, {b!r}]\nohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(df_tr[cols])\n"
                 "r1 = LinearRegression().fit(ohe.transform(df_tr[cols]), y_tr).score(ohe.transform(df_te[cols]), y_te)\n"
                 "te = TargetEncoder(target_type='continuous', cv=3, shuffle=False)\nA = te.fit_transform(df_tr[cols], y_tr)\n"
                 "r2 = LinearRegression().fit(A, y_tr).score(te.transform(df_te[cols]), y_te)\nresult = (r1, r2)"),
        ("MissingIndicator", f"Rebuild `SimpleImputer(add_indicator=True)` from parts: mean-impute the numeric columns "
                             f"`{num}` with `SimpleImputer()` and append `MissingIndicator().fit_transform(...)` as floats. "
                             f"Return `(by_parts, from_add_indicator)`.",
         f"from {IMP} import MissingIndicator, SimpleImputer\nX = df[{num}]\n"
         "parts = np.column_stack([SimpleImputer().fit_transform(X), MissingIndicator().fit_transform(X).astype(float)])\n"
         "result = (parts, SimpleImputer(add_indicator=True).fit_transform(X))"),
    ]
    return catalog(rng, items, t.setup, f"{t.desc}; `{nm}` has missing values; `{a}` and `{b}` are categorical.",
                   tol=1e-6)


# ================================================================= challenge

@template(M, "challenge")
def target_encoding_oof(rng):
    n, C, K, s = rng.randint(10, 16), rng.randint(3, 5), rng.randint(2, 3), seeded(rng)
    m = rng.choice([1.0, 2.0, 5.0])
    report = rng.choice(["encoded", "mse"])
    setup = (f"g = np.random.default_rng({s})\ncats = g.integers(0, {C}, {n})\ny = (cats * 1.5 + g.normal(0, 1, {n})).round(2)\n"
             f"fold = g.integers(0, {K}, {n})\nfold[:2] = [0, 1]\nm = {m}\nreport = {report!r}")
    sol = ("K = int(fold.max()) + 1\nC = int(cats.max()) + 1\nkey = fold * C + cats\n"
           "S = np.bincount(key, weights=y, minlength=K * C).reshape(K, C)\nN = np.bincount(key, minlength=K * C).reshape(K, C)\n"
           "S_out = S.sum(axis=0) - S\nN_out = N.sum(axis=0) - N\n"
           "tot_s = np.bincount(fold, weights=y, minlength=K)\ntot_n = np.bincount(fold, minlength=K)\n"
           "mean_out = (y.sum() - tot_s) / (len(y) - tot_n)\n"
           "enc = (S_out[fold, cats] + m * mean_out[fold]) / (N_out[fold, cats] + m)\n"
           "result = enc if report == 'encoded' else float(((enc - y) ** 2).mean())")
    tests = [f"g = np.random.default_rng({s + i})\ncats = g.integers(0, {cc}, 200000)\ny = (cats % 97 * 0.3 + "
             f"g.normal(0, 1, 200000)).round(2)\nfold = g.integers(0, {kk}, 200000)\nm = {m}\nreport = {report!r}"
             for i, (cc, kk) in ((1, (50000, 5)), (2, (300, 10)))]
    what = ("the encoded value of every row (a float array)" if report == "encoded" else
            "the mean squared error between the encoded values and `y` (a float)")
    return challenge(
        "Out-of-fold target encoding",
        "`TargetEncoder(smooth=m)` cross-fits: each row is encoded by statistics of the **other** folds only. Row `i` "
        "has level `cats[i]`, target `y[i]` and fold `fold[i]`. For a row in fold `f` with level `c`, let `S`, `N` be "
        "the sum and count of the targets of level `c` outside fold `f`, and `μ` the mean target outside fold `f`; its "
        "encoding is `(S + m·μ) / (N + m)` (a level absent outside the fold gets `μ`). Every fold leaves some rows "
        f"outside it. Return {what}. (`report`: {report!r}.)",
        setup, sol, tests, "rows ≤ 2·10^5, levels ≤ 5·10^4, folds ≤ 10", speed="O(n + K·C): counts per (fold, level), "
        "then subtract", tol=1e-9)


@template(M, "challenge")
def knn_impute_line(rng):
    n, k, s = rng.randint(9, 14), rng.randint(1, 3), seeded(rng)
    report = rng.choice(["imputed", "filled", "mean"])
    setup = (f"g = np.random.default_rng({s})\nx = g.uniform(0, 100, {n})\nv = np.sin(x / 9) * 10 + g.normal(0, 1, {n})\n"
             f"v[g.random({n}) < 0.35] = np.nan\nv[:{k + 1}] = np.where(np.isnan(v[:{k + 1}]), 1.0, v[:{k + 1}])\n"
             f"k = {k}\nreport = {report!r}")
    sol = ("have = ~np.isnan(v)\nxd, vd = x[have], v[have]\no = np.argsort(xd)\nxd, vd = xd[o], vd[o]\nq = x[~have]\n"
           "kk = min(k, len(xd))\npos = np.searchsorted(xd, q)\noff = np.arange(-kk, kk)\n"
           "idx = pos[:, None] + off[None, :]\nok = (idx >= 0) & (idx < len(xd))\nidc = np.clip(idx, 0, len(xd) - 1)\n"
           "dist = np.where(ok, np.abs(xd[idc] - q[:, None]), np.inf)\nnear = np.argsort(dist, axis=1, kind='stable')[:, :kk]\n"
           "imputed = vd[np.take_along_axis(idc, near, axis=1)].mean(axis=1)\nfilled = v.copy()\nfilled[~have] = imputed\n"
           "result = {'imputed': imputed, 'filled': filled, 'mean': float(imputed.mean()) if len(imputed) else 0.0}[report]")
    tests = [f"g = np.random.default_rng({s + i})\nx = g.uniform(0, 1000, 200000)\nv = np.sin(x / 9) * 10 + g.normal(0, 1, 200000)\n"
             f"v[g.random(200000) < {fr}] = np.nan\nk = {kk}\nreport = {report!r}" for i, (fr, kk) in ((1, (0.5, 5)), (2, (0.1, 10)))]
    what = {"imputed": "the imputed values, in the order of the missing rows",
            "filled": "the whole column `v` with its NaNs filled", "mean": "the mean of the imputed values"}[report]
    return challenge(
        "KNN imputation along one feature",
        "A two-column table has a complete feature `x` (distinct floats) and a column `v` with NaNs. "
        "`KNNImputer(n_neighbors=k).fit_transform(np.column_stack([x, v]))` fills each missing `v[i]` with the mean `v` "
        "of the `k` rows (among those with `v` present) nearest to row `i` — and with only `x` shared, nearest means "
        f"smallest `|x[j] − x[i]|`. Return {what}. (`report`: {report!r}.)",
        setup, sol, tests, "len(x) ≤ 2·10^5, k ≤ 10", speed="O(n log n + n·k): sort the donors, look k to each side",
        tol=1e-9)


@template(M, "challenge")
def onehot_active_columns(rng):
    n_tr, n_te, s = rng.randint(6, 10), rng.randint(4, 6), seeded(rng)
    drop = rng.choice([None, "first", "if_binary"])
    report = rng.choice(["active", "zeros"])
    setup = (f"g = np.random.default_rng({s})\ntrain = np.column_stack([g.integers(0, 4, {n_tr}) * 7, g.integers(0, 2, {n_tr}) + 100, "
             f"g.integers(0, 3, {n_tr}) * 1000])\ntest = np.column_stack([g.integers(0, 5, {n_te}) * 7, g.integers(0, 2, {n_te}) + 100, "
             f"g.integers(-1, 3, {n_te}) * 1000])\ndrop = {drop!r}\nreport = {report!r}")
    sol = ("F = train.shape[1]\nout = np.full(test.shape, -1, dtype=np.int64)\noffset = 0\nfor f in range(F):\n"
           "    cats = np.unique(train[:, f])\n    d = drop == 'first' or (drop == 'if_binary' and len(cats) == 2)\n"
           "    p = np.searchsorted(cats, test[:, f])\n    pc = np.clip(p, 0, len(cats) - 1)\n    known = cats[pc] == test[:, f]\n"
           "    col = pc - (1 if d else 0)\n    ok = known & (col >= 0)\n    out[ok, f] = offset + col[ok]\n"
           "    offset += len(cats) - (1 if d else 0)\n"
           "result = out if report == 'active' else int((out < 0).all(axis=1).sum())")
    tests = [f"g = np.random.default_rng({s + i})\ntrain = g.integers(0, {hi}, (100000, 3)) * 3\ntest = g.integers(0, {hi}, "
             f"(100000, 3)) * 3 + (g.random((100000, 3)) < 0.05)\ndrop = {drop!r}\nreport = {report!r}"
             for i, hi in ((1, 60000), (2, 2))]
    what = ("an int array of shape `test.shape`: entry `[i, f]` is the output column that is 1 for feature `f` of "
            "test row `i`, or -1 if none is (unknown or dropped level)" if report == "active" else
            "how many test rows encode to all zeros")
    return challenge(
        "Where does each one-hot 1 land?",
        "`OneHotEncoder(drop=drop, handle_unknown='ignore')` is fitted on the integer matrix `train` and applied to "
        "`test`. Each feature gets one output column per sorted distinct training level, features one after another; "
        "`drop='first'` removes each feature's first (smallest) level, `drop='if_binary'` does so only for features with "
        "exactly 2 levels, and `drop=None` keeps all. A level not seen in training (or a dropped one) sets no column. "
        f"Without building the matrix, return {what}. (`drop`: {drop!r}; `report`: {report!r}.)",
        setup, sol, tests, "rows ≤ 10^5, 3 features, levels per feature ≤ 6·10^4",
        speed="O(n log n): sorted levels and binary search per feature")


@template(M, "challenge")
def group_mode_impute(rng):
    n, G, s = rng.randint(12, 18), rng.randint(3, 5), seeded(rng)
    report = rng.choice(["filled", "fallback", "fills"])
    setup = (f"g = np.random.default_rng({s})\ngroup = g.integers(0, {G}, {n})\nvalue = g.integers(0, 4, {n})\n"
             f"value[g.random({n}) < 0.4] = -1\nvalue[0] = 0\nreport = {report!r}")
    sol = ("obs = value >= 0\nG = int(group.max()) + 1\nV = int(value.max()) + 1\n"
           "vals, cnt = np.unique(value[obs], return_counts=True)\nglobal_mode = int(vals[np.argmax(cnt)])\n"
           "key = group[obs].astype(np.int64) * V + value[obs]\nk, c = np.unique(key, return_counts=True)\n"
           "gk, vk = k // V, k % V\norder = np.lexsort((vk, -c, gk))\ngk, vk = gk[order], vk[order]\n"
           "first = np.concatenate([[True], gk[1:] != gk[:-1]])\nfill = np.full(G, global_mode, np.int64)\n"
           "fill[gk[first]] = vk[first]\nhas = np.zeros(G, bool)\nhas[gk] = True\nfilled = np.where(obs, value, fill[group])\n"
           "if report == 'filled':\n    result = filled\nelif report == 'fallback':\n    result = int((~obs & ~has[group]).sum())\n"
           "else:\n    result = np.bincount(fill[group][~obs], minlength=V)")
    tests = [f"g = np.random.default_rng({s + i})\ngroup = g.integers(0, {gg}, 200000)\nvalue = g.integers(0, {vv}, 200000)\n"
             f"value[g.random(200000) < 0.3] = -1\nvalue[0] = 0\nreport = {report!r}" for i, (gg, vv) in ((1, (80000, 6)), (2, (500, 40)))]
    what = {"filled": "the completed `value` array",
            "fallback": "how many missing entries had to use the global fallback (their group has no observed value)",
            "fills": "how many missing entries were filled with each value `0 … max(value)`, as an int array"}[report]
    return challenge(
        "Most-frequent imputation per group",
        "`SimpleImputer(strategy='most_frequent')` fills a missing entry with the column's most common value (ties: the "
        "smallest). A better version imputes **within each group** (store, user, ...): `value[i]` is a category code "
        "≥ 0, or -1 when missing; fill it with the most frequent observed value of its own `group[i]` (ties: smallest), "
        "or, if that group has no observed value, with the most frequent observed value overall (ties: smallest; "
        f"`value` always has an observed entry). Return {what}. (`report`: {report!r}.)",
        setup, sol, tests, "len(value) ≤ 2·10^5", speed="O(n log n): count (group, value) pairs once, no loop over groups")


@template(M, "challenge")
def unseen_category_prefix(rng):
    n, q, s = rng.randint(10, 15), rng.randint(4, 6), seeded(rng)
    report = rng.choice(["rows", "levels", "first_ok"])
    setup = (f"g = np.random.default_rng({s})\nc = (g.zipf(1.6, {n}) % 50)\nts = np.sort(g.choice(np.arange(1, {n}), {q}, "
             f"replace=False))\nbudget = {rng.randint(0, 3)}\nreport = {report!r}")
    sol = ("vals, first = np.unique(c, return_index=True)\nf_row = first[np.searchsorted(vals, c)]\n"
           "rows_sorted = np.sort(f_row)\nlev_sorted = np.sort(first)\n"
           "rows = len(c) - np.searchsorted(rows_sorted, ts, side='left')\nlevels = len(first) - np.searchsorted(lev_sorted, ts, side='left')\n"
           "if report == 'rows':\n    result = rows\nelif report == 'levels':\n    result = levels\nelse:\n"
           "    ok = np.flatnonzero(rows <= budget)\n    result = int(ts[ok[0]]) if len(ok) else -1")
    tests = [f"g = np.random.default_rng({s + i})\nc = g.zipf({a}, 200000) % {mod}\nts = np.sort(g.choice(np.arange(1, 200000), "
             f"100000, replace=False))\nbudget = {b}\nreport = {report!r}" for i, (a, mod, b) in ((1, (1.3, 10 ** 6, 5000)),
                                                                                                   (2, (2.0, 500, 3)))]
    what = {"rows": "the number of rows of `c[t:]` that encode to all zeros (their level never occurs in `c[:t]`)",
            "levels": "the number of distinct levels in `c[t:]` that never occur in `c[:t]`",
            "first_ok": "the smallest `t` in `ts` for which at most `budget` rows of `c[t:]` encode to all zeros (a "
                        "single int; -1 if none)"}[report]
    return challenge(
        "How much history covers the new data?",
        "A stream of category ids `c` arrives in time order. Fitting `OneHotEncoder(handle_unknown='ignore')` on the "
        "first `t` rows and transforming the rest turns every row whose level the encoder never saw into all zeros. For "
        f"every `t` in `ts` (sorted, `1 ≤ t < len(c)`), compute {what}. (`report`: {report!r}.)",
        setup, sol, tests, "len(c) ≤ 2·10^5, len(ts) ≤ 10^5", speed="O((n + q) log n): first occurrences, then counting",
        answer_desc="" if report == "first_ok" else " (an int array, one value per t)")


@template(M, "challenge")
def tag_cooccurrence(rng):
    docs, T, s = rng.randint(6, 10), rng.randint(4, 6), seeded(rng)
    report = rng.choice(["top_pair", "pairs_at_least", "partners"])
    setup = (f"g = np.random.default_rng({s})\nsizes = g.integers(1, 4, {docs})\ndoc = np.repeat(np.arange({docs}), sizes)\n"
             f"tag = np.concatenate([g.choice({T}, k, replace=False) for k in sizes])\nt = 2\nreport = {report!r}")
    sol = ("o = np.lexsort((tag, doc))\nd, tg = doc[o], tag[o]\nT = int(tag.max()) + 1\nkeys = []\n"
           "for off in range(1, len(d)):\n    same = d[off:] == d[:-off]\n    if not same.any():\n        break\n"
           "    keys.append(tg[:-off][same].astype(np.int64) * T + tg[off:][same])\n"
           "keys = np.concatenate(keys) if keys else np.zeros(0, np.int64)\npk, pc = np.unique(keys, return_counts=True)\n"
           "if report == 'top_pair':\n    b = int(pk[np.argmax(pc)]) if len(pk) else -1\n    result = (b // T, b % T) if b >= 0 else (-1, -1)\n"
           "elif report == 'pairs_at_least':\n    result = int((pc >= t).sum())\nelse:\n"
           "    deg = np.bincount(np.concatenate([pk // T, pk % T]), minlength=T)\n    result = int(np.argmax(deg))")
    tests = [f"g = np.random.default_rng({s + i})\nsizes = g.integers(1, 7, {nd})\ndoc = np.repeat(np.arange({nd}), sizes)\n"
             f"tag = np.concatenate([g.choice({tt}, k, replace=False) for k in sizes])\nt = {th}\nreport = {report!r}"
             for i, (nd, tt, th) in ((1, (50000, 3000, 2)), (2, (30000, 60, 40)))]
    what = {"top_pair": "the pair of tags `(a, b)`, `a < b`, that appears together in the most posts (ties: the "
                        "smallest `(a, b)`), as a tuple; `(-1, -1)` if no post has two tags",
            "pairs_at_least": "the number of distinct tag pairs that appear together in at least `t` posts",
            "partners": "the tag that co-occurs with the largest number of **different** other tags (ties: the "
                        "smallest tag id)"}[report]
    return challenge(
        "Tag co-occurrence without the dense matrix",
        "`MultiLabelBinarizer` turns posts' tag lists into a 0/1 matrix `Y`, and `Y.T @ Y` counts how often two tags "
        "appear together — impossible to store densely with thousands of tags. Here entry `k` says post `doc[k]` "
        "carries tag `tag[k]` (no post repeats a tag; tags are ids `0 … max(tag)`; `doc` is sorted). Return "
        f"{what}. (`report`: {report!r}.)",
        setup, sol, tests, "entries ≤ 2·10^5, at most 6 tags per post, tag ids < 3000",
        speed="O(entries · tags per post): enumerate the pairs inside each post",
        answer_desc=" (a tuple of two ints)" if report == "top_pair" else " (an int)")
