"""4.02 Datasets: the bundled loaders (Bunch, return_X_y, as_frame) and the make_* generators."""

from ._common import P, catalog, challenge, lit, seeded, template

M = "4.02"

# key: (loader, rows, features, classes (0 = regression), what the rows are)
_LOAD = {
    "iris": ("load_iris", 150, 4, 3, "iris flowers"),
    "wine": ("load_wine", 178, 13, 3, "wines"),
    "cancer": ("load_breast_cancer", 569, 30, 2, "breast-tumour samples"),
    "digits": ("load_digits", 1797, 64, 10, "handwritten digits"),
    "diabetes": ("load_diabetes", 442, 10, 0, "diabetes patients"),
}
_FEATURES = {
    "iris": ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"],
    "wine": ["alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium", "total_phenols", "flavanoids",
             "nonflavanoid_phenols", "proanthocyanins", "color_intensity", "hue", "od280/od315_of_diluted_wines",
             "proline"],
    "cancer": ["mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness", "mean compactness",
               "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension", "radius error",
               "texture error", "worst radius", "worst texture", "worst perimeter", "worst area", "worst smoothness",
               "worst concavity"],
    "diabetes": ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"],
}
LR = "LogisticRegression(max_iter=1000)"


def _shuffled(key, s):
    """Code loading a bundled classification dataset with its rows shuffled (iris and wine are sorted by class)."""
    L = _LOAD[key][0]
    return (f"from sklearn.datasets import {L}\nX, y = {L}(return_X_y=True)\n"
            f"order = np.random.default_rng({s}).permutation(len(X))\nX, y = X[order], y[order]")


# ================================================================= lite

@template(M, "lite")
def bundled_loader_facts(rng):
    key = rng.choice(list(_LOAD))
    L, n, d, k, what = _LOAD[key]
    m, i = rng.randint(3, 6), rng.randrange(n - 6)
    imp = f"from sklearn.datasets import {L}\n"
    clf = k > 0
    items = [
        (L, f"Load it with `{L}(return_X_y=True)` and return the shape of `X` as a tuple.",
         imp + f"X, y = {L}(return_X_y=True)\nresult = X.shape"),
        (L, f"Load it with `{L}(return_X_y=True)` and return `(number of samples, number of distinct values in y)`.",
         imp + f"X, y = {L}(return_X_y=True)\nresult = (len(y), len(np.unique(y)))"),
        ("feature_names", f"Load the Bunch (`{L}()`, no `return_X_y`) and return its first {m} feature names as a list.",
         imp + f"result = list({L}().feature_names[:{m}])"),
        ("target_names", "Load the Bunch and return its class names (`target_names`) as a list.",
         imp + f"result = list({L}().target_names)") if clf else
        ("target", "Load the Bunch and return `(smallest, largest)` value of its `target` (a continuous score).",
         imp + f"data = {L}()\nresult = (data.target.min(), data.target.max())"),
        ("frame", f"Load it with `as_frame=True` and return the first {m} rows of the Bunch's `frame` DataFrame "
                  f"(`.head({m})`).", imp + f"result = {L}(as_frame=True).frame.head({m})"),
        ("frame", "Load it with `as_frame=True`. Its `frame` holds every feature **plus** a `target` column: return the "
                  "frame's shape.", imp + f"result = {L}(as_frame=True).frame.shape"),
        ("as_frame", f"Call `{L}(return_X_y=True, as_frame=True)` (pandas objects instead of arrays) and return "
                     f"`(type(X).__name__, type(y).__name__, y.name)`.",
         imp + f"X, y = {L}(return_X_y=True, as_frame=True)\nresult = (type(X).__name__, type(y).__name__, y.name)"),
        ("data", f"Load the Bunch and return row {i} of its `data` array.", imp + f"result = {L}().data[{i}]"),
        ("target", f"Load the Bunch and return the targets of rows {i} to {i + m - 1} (inclusive).",
         imp + f"result = {L}().target[{i}:{i + m}]"),
        ("bincount", "Load it with `return_X_y=True` and return how many samples each class has (`np.bincount(y)`).",
         imp + f"X, y = {L}(return_X_y=True)\nresult = np.bincount(y)") if clf else
        ("mean", "Load it with `return_X_y=True` and return the mean target value.",
         imp + f"X, y = {L}(return_X_y=True)\nresult = y.mean()", {"require": [L]}),
    ]
    return catalog(rng, items, "", f"`{L}` (from `sklearn.datasets`) loads {n} {what}.", tol=1e-9)


@template(M, "lite")
def generator_calls(rng):
    n, s = rng.choice([120, 150, 200, 240, 300, 360]), rng.randint(0, 99)
    d, inf = rng.randint(5, 10), rng.randint(3, 4)
    red, k = rng.randint(0, d - inf), rng.randint(2, 4)
    p = rng.choice([0.05, 0.1, 0.2])
    c, std = rng.randint(2, 5), rng.choice([0.5, 0.8, 1.0, 1.5])
    noise, factor, L = rng.choice([0.05, 0.1, 0.2]), rng.choice([0.3, 0.5, 0.7]), rng.randint(3, 6)
    imp = "from sklearn.datasets import "
    items = [
        ("make_classification", f"Generate `X, y = make_classification(n_samples={n}, n_features={d}, n_informative="
                                f"{inf}, n_redundant={red}, n_classes={k}, random_state={s})` and return `(X.shape, "
                                f"number of distinct classes in y)`.",
         imp + f"make_classification\nX, y = make_classification(n_samples={n}, n_features={d}, n_informative={inf}, "
               f"n_redundant={red}, n_classes={k}, random_state={s})\nresult = (X.shape, len(np.unique(y)))"),
        ("make_classification", f"Generate an imbalanced binary problem: `make_classification(n_samples={n}, "
                                f"weights=[{1 - p:.2f}, {p}], flip_y=0, random_state={s})` and return the class counts "
                                f"(`np.bincount(y)`).",
         imp + f"make_classification\nX, y = make_classification(n_samples={n}, weights=[{1 - p:.2f}, {p}], flip_y=0, "
               f"random_state={s})\nresult = np.bincount(y)"),
        ("make_regression", f"`make_regression(n_samples={n}, n_features={d}, n_informative={inf}, noise=5.0, coef=True, "
                            f"random_state={s})` also returns the true coefficients. Return how many of them are "
                            f"non-zero.",
         imp + f"make_regression\nX, y, coef = make_regression(n_samples={n}, n_features={d}, n_informative={inf}, "
               f"noise=5.0, coef=True, random_state={s})\nresult = int(np.count_nonzero(coef))"),
        ("make_blobs", f"Generate `make_blobs(n_samples={n}, centers={c}, cluster_std={std}, random_state={s})` and "
                       f"return the number of points per blob (`np.bincount(y)`).",
         imp + f"make_blobs\nX, y = make_blobs(n_samples={n}, centers={c}, cluster_std={std}, random_state={s})\n"
               "result = np.bincount(y)"),
        ("make_moons", f"Generate two interleaving half-circles with `make_moons(n_samples={n}, noise={noise}, "
                       f"random_state={s})` and return the first 3 rows of `X`.",
         imp + f"make_moons\nX, y = make_moons(n_samples={n}, noise={noise}, random_state={s})\nresult = X[:3]"),
        ("make_circles", f"Generate `make_circles(n_samples={n}, noise={noise}, factor={factor}, random_state={s})` "
                         f"(class 1 is the inner circle, `factor` times smaller) and return the mean distance to the "
                         f"origin of the class-1 points.",
         imp + f"make_circles\nX, y = make_circles(n_samples={n}, noise={noise}, factor={factor}, random_state={s})\n"
               "result = np.linalg.norm(X[y == 1], axis=1).mean()"),
        ("make_swiss_roll", f"`make_swiss_roll(n_samples={n}, noise={noise}, random_state={s})` returns 3-D points `X` "
                            f"and each point's position `t` along the roll. Return `(X.shape, t.shape)`.",
         imp + f"make_swiss_roll\nX, t = make_swiss_roll(n_samples={n}, noise={noise}, random_state={s})\n"
               "result = (X.shape, t.shape)"),
        ("make_multilabel_classification",
         f"`make_multilabel_classification(n_samples={n}, n_classes={L}, random_state={s})` returns features `X` and a "
         f"0/1 label matrix `Y` (a sample may have several labels). Return how many samples carry each label "
         f"(`Y.sum(axis=0)`).",
         imp + f"make_multilabel_classification\nX, Y = make_multilabel_classification(n_samples={n}, n_classes={L}, "
               f"random_state={s})\nresult = Y.sum(axis=0)"),
    ]
    return catalog(rng, items, "", "The `make_*` functions of `sklearn.datasets` generate synthetic data; a fixed "
                                   "`random_state` makes them reproducible.", tol=1e-9)


@template(M, "lite")
def digit_images(rng):
    i, k = rng.randrange(1797), rng.randint(2, 9)
    items = [
        ("images", "Return the shape of `digits.images` (one 8×8 picture per sample).", "result = digits.images.shape"),
        ("images", f"Return image {i} as an 8×8 array (`digits.images`).", f"result = digits.images[{i}]"),
        ("data", f"Each row of `digits.data` is a picture flattened to 64 values. Rebuild image {i} from "
                 f"`digits.data` with `reshape`.", f"result = digits.data[{i}].reshape(8, 8)"),
        ("target", f"Return the digit shown by image {i} as an int.", f"result = int(digits.target[{i}])"),
        ("target_names", "Return `digits.target_names`.", "result = digits.target_names"),
        ("data", "Return the largest pixel value anywhere in `digits.data`.", "result = digits.data.max()"),
        ("images", f"How many pixels of image {i} are non-zero (ink)? Return an int.",
         f"result = int(np.count_nonzero(digits.images[{i}]))"),
        ("images", f"Return the 8 row sums of image {i} (how much ink each row of the picture has).",
         f"result = digits.images[{i}].sum(axis=1)"),
        ("load_digits", f"`load_digits(n_class={k})` keeps only the digits 0 to {k - 1}. Load them with "
                        f"`return_X_y=True` and return the shape of `X`.",
         f"from sklearn.datasets import load_digits\nX, y = load_digits(n_class={k}, return_X_y=True)\nresult = X.shape"),
    ]
    return catalog(rng, items, "from sklearn.datasets import load_digits\ndigits = load_digits()",
                   "`digits = load_digits()` is the handwritten-digits Bunch (1797 samples).", tol=1e-9)


# ================================================================= easy

@template(M, "easy")
def labeled_frame_queries(rng):
    key = rng.choice(["iris", "wine", "cancer"])
    L, n, d, k, what = _LOAD[key]
    col = rng.choice(_FEATURES[key][:12])
    a, m, c = rng.randint(0, n - 6), rng.randint(3, 5), rng.randrange(k)
    items = [
        ("groupby", f"Return the mean `{col}` of each class: group `data.frame` by its `target` column (a Series "
                    f"indexed by class number).", f"result = data.frame.groupby('target')[{col!r}].mean()",
         {"require": ["frame"]}),
        ("target_names", "Return a dict mapping each class **name** (`target_names`) to its number of samples.",
         "counts = data.frame['target'].value_counts()\n"
         "result = {str(data.target_names[c]): int(v) for c, v in counts.items()}"),
        ("target_names", f"Return the class names of rows {a} to {a + m - 1} (inclusive) as a list of str: index "
                         f"`target_names` with the targets.",
         f"result = [str(s) for s in data.target_names[data.target.to_numpy()[{a}:{a + m}]]]"),
        ("feature_names", "Return the first 5 feature names as a list.", "result = list(data.feature_names[:5])"),
        ("frame", f"How many samples have a `{col}` above that column's mean? Return an int.",
         f"col = data.frame[{col!r}]\nresult = int((col > col.mean()).sum())"),
        ("target_names", f"Which class has the largest average `{col}`? Return its name (a str from `target_names`).",
         f"means = data.frame.groupby('target')[{col!r}].mean()\nresult = str(data.target_names[means.idxmax()])"),
        ("corr", f"Return the correlation between `{col}` and the `target` column of `data.frame`.",
         f"result = data.frame[{col!r}].corr(data.frame['target'])", {"require": ["frame"]}),
        ("frame", "Return the name of the feature with the largest mean (drop the `target` column first).",
         "result = data.frame.drop(columns='target').mean().idxmax()"),
        ("frame", f"Return the standard deviation of every feature over the samples of class {c} only, as a Series.",
         f"df = data.frame\nresult = df[df['target'] == {c}].drop(columns='target').std()"),
        ("target_names", f"Make a copy of `data.frame` with an extra column `label` holding each row's class name, and "
                         f"return its rows {a} to {a + m - 1}.",
         f"df = data.frame.copy()\ndf['label'] = data.target_names[df['target'].to_numpy()]\nresult = df.iloc[{a}:{a + m}]"),
    ]
    return catalog(rng, items, f"from sklearn.datasets import {L}\ndata = {L}(as_frame=True)",
                   f"`data = {L}(as_frame=True)`: {n} {what}; `data.frame` is a DataFrame of the features plus a "
                   f"`target` column.", tol=1e-9)


@template(M, "easy")
def imbalanced_generation(rng):
    n = rng.choice([400, 500, 800, 1000])
    p = rng.choice([0.01, 0.02, 0.05, 0.1])
    s = rng.randint(0, 99)
    call = f"make_classification(n_samples={n}, weights=[{1 - p:.2f}, {p}], flip_y=0, random_state={s})"
    imp = "from sklearn.datasets import make_classification\n"
    task = rng.choice(["counts", "rate", "flip", "ratio"])
    head = f"`weights` sets the share of each class in `make_classification`."
    if task == "counts":
        return P(f"{head} Generate `X, y = {call}` ({p:.0%} positives) and return the class counts.", "",
                 imp + f"X, y = {call}\nresult = np.bincount(y)", require=["make_classification"])
    if task == "rate":
        return P(f"{head} Generate `X, y = {call}` and return the fraction of positive (class 1) samples.", "",
                 imp + f"X, y = {call}\nresult = y.mean()", require=["make_classification"])
    if task == "flip":
        return P(f"{head} By default it also flips 1% of the labels at random (`flip_y=0.01`), so the counts drift. "
                 f"Return `(np.bincount(y_exact), np.bincount(y_default))` for `{call}` and the same call **without** "
                 f"`flip_y=0`.", "",
                 imp + f"X, y_exact = {call}\nX, y_default = make_classification(n_samples={n}, weights=[{1 - p:.2f}, "
                       f"{p}], random_state={s})\nresult = (np.bincount(y_exact), np.bincount(y_default))",
                 require=["make_classification"])
    return P(f"{head} Generate `X, y = {call}` and return the imbalance ratio: majority-class count divided by "
             f"minority-class count, as a float.", "",
             imp + f"X, y = {call}\ncounts = np.bincount(y)\nresult = counts.max() / counts.min()",
             require=["make_classification"])


@template(M, "easy")
def regression_ground_truth(rng):
    n, d, s = rng.randint(40, 120), rng.randint(4, 8), rng.randint(0, 99)
    inf = rng.randint(2, d - 1)
    bias, noise = rng.choice([0.0, 2.5, 10.0, -4.0]), rng.choice([1.0, 3.0, 8.0])
    call = (f"make_regression(n_samples={n}, n_features={d}, n_informative={inf}, noise={noise}, bias={bias}, "
            f"coef=True, random_state={s})")
    imp = f"from sklearn.datasets import make_regression\nX, y, coef = {call}\n"
    task = rng.choice(["exact", "support", "noise", "recover"])
    head = f"`X, y, coef = {call}` returns the data and the true coefficients."
    if task == "exact":
        return P(f"{head} The target is `X @ coef + bias + noise`. Return the largest absolute value of "
                 f"`y - (X @ coef + {bias})` (with `noise={noise}`, it measures the noise).", "",
                 imp + f"result = np.abs(y - (X @ coef + {bias})).max()", require=["make_regression"])
    if task == "support":
        return P(f"{head} Only `n_informative` features influence `y`. Return the indices of the features with a "
                 f"non-zero true coefficient.", "", imp + "result = np.flatnonzero(coef)", require=["make_regression"])
    if task == "noise":
        return P(f"{head} Return the standard deviation of the residual `y - X @ coef - {bias}` (close to `noise`).", "",
                 imp + f"result = (y - X @ coef - {bias}).std()", require=["make_regression"])
    return P(f"{head} Fit `LinearRegression` on `X, y` and return the largest absolute difference between its "
             f"`coef_` and the true `coef`.", "",
             imp + "from sklearn.linear_model import LinearRegression\n"
                   "result = np.abs(LinearRegression().fit(X, y).coef_ - coef).max()",
             require=["make_regression", "LinearRegression"], tol=1e-6)


@template(M, "easy")
def blob_geometry(rng):
    k, s = rng.randint(2, 4), rng.randint(0, 99)
    centers = [[rng.randint(-8, 8), rng.randint(-8, 8)] for _ in range(k)]
    sizes = [rng.randint(15, 60) for _ in range(k)]
    std = rng.choice([0.3, 0.5, 1.0])
    task = rng.choice(["means", "sizes", "spread", "returned"])
    imp = "from sklearn.datasets import make_blobs\n"
    if task == "means":
        call = f"make_blobs(n_samples={sum(sizes)}, centers={centers}, cluster_std={std}, random_state={s})"
        return P(f"`centers` can list the blob centres. Generate `X, y = {call}` and return the mean point of each "
                 f"blob as a {k} × 2 array (row `c` = mean of the points with `y == c`; close to the centres).", "",
                 imp + f"X, y = {call}\nresult = np.array([X[y == c].mean(axis=0) for c in range({k})])",
                 require=["make_blobs"])
    if task == "sizes":
        call = f"make_blobs(n_samples={sizes}, centers={centers}, cluster_std={std}, random_state={s})"
        return P(f"`n_samples` may be a list: one size per blob. Generate `X, y = {call}` and return "
                 f"`(np.bincount(y), X.shape)`.", "", imp + f"X, y = {call}\nresult = (np.bincount(y), X.shape)",
                 require=["make_blobs"])
    if task == "spread":
        call = f"make_blobs(n_samples={sum(sizes)}, centers={centers}, cluster_std={std}, random_state={s})"
        return P(f"Generate `X, y = {call}` and return each blob's standard deviation along x and y as a {k} × 2 "
                 f"array (both close to `cluster_std`).", "",
                 imp + f"X, y = {call}\nresult = np.array([X[y == c].std(axis=0) for c in range({k})])",
                 require=["make_blobs"])
    call = f"make_blobs(n_samples={sum(sizes)}, centers={k}, cluster_std={std}, return_centers=True, random_state={s})"
    return P(f"With a number of `centers`, the centres are drawn at random; `return_centers=True` returns them too. "
             f"Generate `X, y, centers = {call}` and return the mean distance of the points to their own blob's "
             f"centre.", "",
             imp + f"X, y, centers = {call}\nresult = np.linalg.norm(X - centers[y], axis=1).mean()",
             require=["make_blobs"])


@template(M, "easy")
def noise_free_shapes(rng):
    n, factor = rng.randint(30, 90), rng.choice([0.2, 0.4, 0.5, 0.8])
    noise, s = rng.choice([0.05, 0.1, 0.2]), rng.randint(0, 99)
    task = rng.choice(["circle_radii", "moon_centres", "noisy_radii", "moon_box"])
    if task == "circle_radii":
        return P(f"Without noise, `make_circles` puts points exactly on two circles. Generate `X, y = make_circles("
                 f"n_samples={n}, factor={factor}, noise=None, random_state={s})` and return the sorted distinct "
                 f"distances to the origin, rounded to 6 decimals (`np.unique(np.round(..., 6))`).", "",
                 f"from sklearn.datasets import make_circles\nX, y = make_circles(n_samples={n}, factor={factor}, "
                 f"noise=None, random_state={s})\nresult = np.unique(np.round(np.linalg.norm(X, axis=1), 6))",
                 require=["make_circles"])
    if task == "moon_centres":
        return P(f"Without noise, class 0 of `make_moons` lies on the unit circle around (0, 0) and class 1 on the unit "
                 f"circle around (1, 0.5). Generate `X, y = make_moons(n_samples={n}, noise=None, random_state={s})` "
                 f"and return the distinct distances of the points from their own class's centre, rounded to 6 decimals "
                 f"(`np.unique(np.round(..., 6))`; a single value).", "",
                 f"from sklearn.datasets import make_moons\nX, y = make_moons(n_samples={n}, noise=None, "
                 f"random_state={s})\nc = np.array([[0.0, 0.0], [1.0, 0.5]])[y]\n"
                 "result = np.unique(np.round(np.linalg.norm(X - c, axis=1), 6))", require=["make_moons"])
    if task == "noisy_radii":
        return P(f"Generate `X, y = make_circles(n_samples={n * 4}, noise={noise}, factor={factor}, random_state={s})` "
                 f"and return the mean distance to the origin of each class, `[class 0, class 1]` (about 1 and "
                 f"{factor}).", "",
                 f"from sklearn.datasets import make_circles\nX, y = make_circles(n_samples={n * 4}, noise={noise}, "
                 f"factor={factor}, random_state={s})\nr = np.linalg.norm(X, axis=1)\n"
                 "result = [r[y == 0].mean(), r[y == 1].mean()]", require=["make_circles"])
    return P(f"Generate `X, y = make_moons(n_samples={n * 4}, noise={noise}, random_state={s})` and return the "
             f"bounding box of class 1 as `(x_min, x_max, y_min, y_max)`.", "",
             f"from sklearn.datasets import make_moons\nX, y = make_moons(n_samples={n * 4}, noise={noise}, "
             f"random_state={s})\nB = X[y == 1]\nresult = (B[:, 0].min(), B[:, 0].max(), B[:, 1].min(), B[:, 1].max())",
             require=["make_moons"])


@template(M, "easy")
def diabetes_scaling(rng):
    j = rng.randrange(10)
    col = _FEATURES["diabetes"][j]
    task = rng.choice(["norms", "means", "raw", "rebuild"])
    imp = "from sklearn.datasets import load_diabetes\n"
    if task == "norms":
        return P("The diabetes features come pre-scaled: every column is centred and divided so its **sum of squares** "
                 "is 1. Load `X, y = load_diabetes(return_X_y=True)` and return each column's sum of squares.", "",
                 imp + "X, y = load_diabetes(return_X_y=True)\nresult = (X ** 2).sum(axis=0)",
                 require=["load_diabetes"])
    if task == "means":
        return P("Load `X, y = load_diabetes(return_X_y=True)` and return `(column means, column standard deviations)`: "
                 "the means are ~0 and the standard deviations all equal `1 / sqrt(442)`.", "",
                 imp + "X, y = load_diabetes(return_X_y=True)\nresult = (X.mean(axis=0), X.std(axis=0))",
                 require=["load_diabetes"], tol=1e-6)
    if task == "raw":
        return P(f"`load_diabetes(scaled=False)` gives the original units. Load the raw data as a DataFrame "
                 f"(`as_frame=True`) and return the mean and the max of the `{col}` column as a tuple.", "",
                 imp + f"df = load_diabetes(scaled=False, as_frame=True).frame\nresult = (df[{col!r}].mean(), "
                       f"df[{col!r}].max())", require=["load_diabetes"])
    return P(f"Rebuild the scaled column `{col}` from the raw one: centre the raw column (`scaled=False`) and divide "
             f"it by its standard deviation times `sqrt(n)`. Return the first 5 rebuilt values (they match "
             f"`load_diabetes().data[:5, {j}]`).", "",
             imp + f"raw = load_diabetes(scaled=False).data[:, {j}]\n"
                   "mine = (raw - raw.mean()) / (raw.std() * np.sqrt(len(raw)))\nresult = mine[:5]",
             require=["load_diabetes"], tol=1e-6)


@template(M, "easy")
def multilabel_counts(rng):
    n, L, s = rng.randint(80, 250), rng.randint(3, 6), rng.randint(0, 99)
    nl = rng.choice([1, 2, 3])
    j = rng.randrange(L)
    call = f"make_multilabel_classification(n_samples={n}, n_classes={L}, n_labels={nl}, random_state={s})"
    items = [
        ("sum", "Return the label cardinality: the average number of labels per sample.", "result = Y.sum(axis=1).mean()"),
        ("sum", "Some samples have no label at all. Return how many.", "result = int((Y.sum(axis=1) == 0).sum())"),
        ("argmax", "Return the index of the most frequent label.", "result = int(Y.sum(axis=0).argmax())"),
        ("mean", "Return the label density: the fraction of ones in `Y`.", "result = Y.mean()"),
        ("sum", "`X` holds word counts. Return the number of words of the first 5 documents (row sums of `X`).",
         "result = X[:5].sum(axis=1)"),
        ("bincount", "Return how many samples have 0, 1, 2, ... labels (`np.bincount` of the row sums).",
         "result = np.bincount(Y.sum(axis=1))"),
        ("sum", f"Return how many samples carry label {j} **and** at least one other label.",
         f"result = int(((Y[:, {j}] == 1) & (Y.sum(axis=1) > 1)).sum())"),
    ]
    return catalog(rng, items, f"from sklearn.datasets import make_multilabel_classification\nX, Y = {call}",
                   f"`X, Y = {call}`: each row of the 0/1 matrix `Y` says which of {L} labels a sample has.",
                   tol=1e-9)


# ================================================================= medium

@template(M, "medium")
def dataset_eda(rng):
    key = rng.choice(["cancer", "wine"])
    L, n, d, k, what = _LOAD[key]
    col = rng.choice(_FEATURES[key][:12])
    top = rng.randint(2, 4)
    items = [
        ("corr", f"Return the names of the {top} features most correlated with the target (largest absolute "
                 f"correlation), most correlated first.",
         f"r = df.drop(columns='target').corrwith(df['target']).abs()\nresult = list(r.sort_values(ascending=False)"
         f".index[:{top}])", {"require": []}),
        ("groupby", "Return the feature whose class means differ most **relative to its spread**: (max class mean − "
                    "min class mean) / overall standard deviation, largest.",
         "g = df.groupby('target').mean()\nresult = ((g.max() - g.min()) / df.drop(columns='target').std()).idxmax()",
         {"require": []}),
        ("corr", "Which two different features are most correlated with each other? Return their names as a sorted "
                 "list.",
         "C = df.drop(columns='target').corr().abs().to_numpy().copy()\nnp.fill_diagonal(C, 0)\ni, j = np.unravel_index(C.argmax(), C.shape)\n"
         "names = df.columns[:-1]\nresult = sorted([names[i], names[j]])", {"require": []}),
        ("target_names", f"Return a dict mapping each class name to the median `{col}` of that class.",
         f"med = df.groupby('target')[{col!r}].median()\nresult = {{str(data.target_names[c]): v for c, v in med.items()}}"),
        ("skew", "How many features are strongly right-skewed (pandas `skew()` above 1)? Return an int.",
         "result = int((df.drop(columns='target').skew() > 1).sum())", {"require": []}),
        ("describe", f"Return the 25%, 50% and 75% quantiles of `{col}` as a list (from `describe()` or `quantile`).",
         f"result = df[{col!r}].quantile([0.25, 0.5, 0.75]).tolist()", {"require": []}),
        ("frame", f"Fit `{LR}` on the {top} features most correlated with the target (absolute correlation) after "
                  f"standardizing them with `StandardScaler`, and return the training accuracy.",
         f"from sklearn.linear_model import LogisticRegression\nfrom sklearn.preprocessing import StandardScaler\n"
         f"r = df.drop(columns='target').corrwith(df['target']).abs()\ncols = r.sort_values(ascending=False).index[:{top}]\n"
         f"Z = StandardScaler().fit_transform(df[cols])\nresult = {LR}.fit(Z, df['target']).score(Z, df['target'])",
         {"require": ["LogisticRegression"]}),
        ("frame", f"Return the fraction of samples of each class (sorted by class number) whose `{col}` is above the "
                  f"overall median of `{col}`, as a list.",
         f"above = df[{col!r}] > df[{col!r}].median()\nresult = above.groupby(df['target']).mean().tolist()",
         {"require": []}),
    ]
    return catalog(rng, items, f"from sklearn.datasets import {L}\ndata = {L}(as_frame=True)\ndf = data.frame",
                   f"`data = {L}(as_frame=True)` holds {n} {what}; `df = data.frame` has the {d} features and a "
                   f"`target` column.", tol=1e-6)


@template(M, "medium")
def classification_generator_knobs(rng):
    n, s = rng.choice([200, 300, 400]), rng.randint(0, 99)
    inf, red, rep = rng.randint(2, 4), rng.randint(1, 2), rng.randint(1, 2)
    useless = rng.randint(0, 2)
    d = inf + red + rep + useless
    f = rng.choice([0.1, 0.2, 0.3])
    seps = sorted(rng.sample([0.3, 0.5, 1.0, 1.5, 2.0], 3))
    base = (f"n_samples={n}, n_features={d}, n_informative={inf}, n_redundant={red}, n_repeated={rep}")
    imp = "from sklearn.datasets import make_classification\n"
    items = [
        ("make_classification", f"Redundant features are linear combinations of the informative ones and repeated "
                                f"features are copies, so they add no rank. Generate `X, y = make_classification({base}, "
                                f"random_state={s})` and return `np.linalg.matrix_rank(X)`.",
         imp + f"X, y = make_classification({base}, random_state={s})\nresult = int(np.linalg.matrix_rank(X))"),
        ("make_classification", f"Generate `X, y = make_classification({base}, random_state={s})` and find the repeated "
                                f"features: return the sorted list of pairs `(i, j)`, `i < j`, of identical columns.",
         imp + f"X, y = make_classification({base}, random_state={s})\n"
               f"result = [(i, j) for i in range({d}) for j in range(i + 1, {d}) if np.array_equal(X[:, i], X[:, j])]"),
        ("make_classification", f"`flip_y` assigns a random class to a fraction of the samples. With `shuffle=False` "
                                f"the features don't depend on it: compare `make_classification({base}, shuffle=False, "
                                f"flip_y=0, random_state={s})` and the same with `flip_y={f}`; return the fraction of "
                                f"labels that differ.",
         imp + f"X0, y0 = make_classification({base}, shuffle=False, flip_y=0, random_state={s})\n"
               f"X1, y1 = make_classification({base}, shuffle=False, flip_y={f}, random_state={s})\n"
               "result = (y0 != y1).mean()"),
        ("class_sep", f"`class_sep` spreads the classes apart. For `class_sep` in `{seps}`, generate "
                      f"`make_classification({base}, class_sep=..., random_state={s})`, fit `{LR}` and record the "
                      f"training accuracy. Return the list.",
         imp + "from sklearn.linear_model import LogisticRegression\nresult = []\n"
               f"for sep in {seps}:\n    X, y = make_classification({base}, class_sep=sep, random_state={s})\n"
               f"    result.append({LR}.fit(X, y).score(X, y))", {"require": ["make_classification"]}),
        ("n_classes", f"Generate a 3-class problem with 2 informative features: `make_classification(n_samples={n}, "
                      f"n_features={d}, n_informative=2, n_redundant=0, n_classes=3, n_clusters_per_class=1, "
                      f"random_state={s})` and return the class counts.",
         imp + f"X, y = make_classification(n_samples={n}, n_features={d}, n_informative=2, n_redundant=0, n_classes=3, "
               f"n_clusters_per_class=1, random_state={s})\nresult = np.bincount(y)", {"require": ["make_classification"]}),
        ("shuffle", f"With `shuffle=False`, the columns come in order: informative, redundant, repeated, then useless. "
                    f"Generate `make_classification({base}, shuffle=False, random_state={s})` and return the absolute "
                    f"correlation of every column with `y` (`np.corrcoef`), as an array.",
         imp + f"X, y = make_classification({base}, shuffle=False, random_state={s})\n"
               f"result = np.array([abs(np.corrcoef(X[:, j], y)[0, 1]) for j in range({d})])",
         {"require": ["make_classification"]}),
        ("weights", f"Return the class counts of `make_classification(n_samples={n}, n_classes=3, n_informative=3, "
                    f"weights=[0.6, 0.3, 0.1], flip_y=0, random_state={s})`.",
         imp + f"X, y = make_classification(n_samples={n}, n_classes=3, n_informative=3, weights=[0.6, 0.3, 0.1], "
               f"flip_y=0, random_state={s})\nresult = np.bincount(y)", {"require": ["make_classification"]}),
    ]
    return catalog(rng, items, "", "`make_classification` (from `sklearn.datasets`) has knobs that shape the problem.",
                   tol=1e-6)


@template(M, "medium")
def shapes_need_nonlinear_models(rng):
    n, s = rng.choice([200, 240, 300]), rng.randint(0, 99)
    noise = rng.choice([0.05, 0.1, 0.15])
    factor = rng.choice([0.3, 0.5])
    k = rng.choice([3, 5, 7])
    gen = rng.choice(["moons", "circles"])
    call = (f"make_moons(n_samples={n}, noise={noise}, random_state={s})" if gen == "moons" else
            f"make_circles(n_samples={n}, noise={noise}, factor={factor}, random_state={s})")
    fn = "make_moons" if gen == "moons" else "make_circles"
    setup = f"from sklearn.datasets import {fn}\nX, y = {call}\nX_tr, X_te, y_tr, y_te = X[::2], X[1::2], y[::2], y[1::2]"
    head = f"`X, y = {call}`; the even rows are for training (`X_tr, y_tr`), the odd rows for testing (`X_te, y_te`)."
    task = rng.choice(["linear", "knn", "both", "radius", "tree"])
    if task == "linear":
        return P(f"{head} Fit `LogisticRegression()` (a straight-line boundary) and return its test accuracy.", setup,
                 "from sklearn.linear_model import LogisticRegression\nresult = LogisticRegression().fit(X_tr, y_tr)"
                 ".score(X_te, y_te)", require=["LogisticRegression"])
    if task == "knn":
        return P(f"{head} Fit `KNeighborsClassifier(n_neighbors={k})` (from `sklearn.neighbors`) and return its test "
                 f"accuracy.", setup,
                 f"from sklearn.neighbors import KNeighborsClassifier\nresult = KNeighborsClassifier(n_neighbors={k})"
                 ".fit(X_tr, y_tr).score(X_te, y_te)", require=["KNeighborsClassifier"])
    if task == "both":
        return P(f"{head} Return `(linear, knn)`: the test accuracies of `LogisticRegression()` and "
                 f"`KNeighborsClassifier(n_neighbors={k})`. The curved shape favours the neighbours.", setup,
                 "from sklearn.linear_model import LogisticRegression\nfrom sklearn.neighbors import KNeighborsClassifier\n"
                 f"result = (LogisticRegression().fit(X_tr, y_tr).score(X_te, y_te), KNeighborsClassifier(n_neighbors={k})"
                 ".fit(X_tr, y_tr).score(X_te, y_te))", require=["LogisticRegression", "KNeighborsClassifier"])
    if task == "radius":
        return P(f"{head} A good feature makes a linear model work: add the squared distance to the centre of the data "
                 f"(`((X - X_tr.mean(axis=0)) ** 2).sum(axis=1)`, using the **training** mean) as a third column to both "
                 f"sets, fit `LogisticRegression()` and return its test accuracy.", setup,
                 "from sklearn.linear_model import LogisticRegression\nc = X_tr.mean(axis=0)\n"
                 "aug = lambda A: np.column_stack([A, ((A - c) ** 2).sum(axis=1)])\n"
                 "result = LogisticRegression().fit(aug(X_tr), y_tr).score(aug(X_te), y_te)",
                 require=["LogisticRegression"])
    depth = rng.randint(2, 6)
    return P(f"{head} Fit `DecisionTreeClassifier(max_depth={depth}, random_state=0)` (from `sklearn.tree`) and return "
             f"`(train accuracy, test accuracy)`.", setup,
             f"from sklearn.tree import DecisionTreeClassifier\ntree = DecisionTreeClassifier(max_depth={depth}, "
             "random_state=0).fit(X_tr, y_tr)\nresult = (tree.score(X_tr, y_tr), tree.score(X_te, y_te))",
             require=["DecisionTreeClassifier"])


@template(M, "medium")
def multilabel_analysis(rng):
    n, L, s = rng.randint(120, 300), rng.randint(4, 6), rng.randint(0, 99)
    j = rng.randrange(L)
    call = f"make_multilabel_classification(n_samples={n}, n_classes={L}, n_labels=2, random_state={s})"
    items = [
        ("T", "Return the label co-occurrence matrix: entry `[a, b]` counts samples with both labels `a` and `b` "
              "(the diagonal counts each label). `Y.T @ Y` does it.", "result = Y.T @ Y", {"require": []}),
        ("unique", "Return the most frequent label **set** as a sorted list of label indices (ties: the set that comes "
                   "first in `np.unique(Y, axis=0)` order).",
         "sets, counts = np.unique(Y, axis=0, return_counts=True)\nresult = np.flatnonzero(sets[counts.argmax()]).tolist()"),
        ("unique", "How many different label sets occur? Return an int.", "result = len(np.unique(Y, axis=0))"),
        ("LogisticRegression", f"Predict label {j} alone: fit `{LR}` on `X` and column {j} of `Y`, and return its "
                               f"training accuracy.",
         f"from sklearn.linear_model import LogisticRegression\nresult = {LR}.fit(X, Y[:, {j}]).score(X, Y[:, {j}])"),
        ("corrcoef", "Which two labels are most positively correlated? Return them as a sorted list `[a, b]` "
                     "(`np.corrcoef` of the columns, ignoring the diagonal).",
         "C = np.corrcoef(Y.T)\nnp.fill_diagonal(C, -2)\na, b = np.unravel_index(C.argmax(), C.shape)\n"
         "result = sorted([int(a), int(b)])", {"require": []}),
        ("mean", f"Do samples with label {j} have longer documents? Return `(mean row sum of X with label {j}, mean "
                 f"row sum without)`.",
         f"lengths = X.sum(axis=1)\nresult = (lengths[Y[:, {j}] == 1].mean(), lengths[Y[:, {j}] == 0].mean())",
         {"require": []}),
        ("allow_unlabeled", f"Regenerate with `allow_unlabeled=False` (every sample gets a label) and return the "
                            f"smallest number of labels any sample has, and the label cardinality, as a tuple.",
         f"from sklearn.datasets import make_multilabel_classification\nX2, Y2 = make_multilabel_classification("
         f"n_samples={n}, n_classes={L}, n_labels=2, allow_unlabeled=False, random_state={s})\n"
         "result = (int(Y2.sum(axis=1).min()), Y2.sum(axis=1).mean())", {"require": ["make_multilabel_classification"]}),
    ]
    return catalog(rng, items, f"from sklearn.datasets import make_multilabel_classification\nX, Y = {call}",
                   f"`X, Y = {call}`: `X` holds word counts and the 0/1 matrix `Y` the {L} labels of each sample.",
                   tol=1e-6)


@template(M, "medium")
def swiss_roll_geometry(rng):
    n, s = rng.randint(100, 300), rng.randint(0, 99)
    noise = rng.choice([0.2, 0.5, 1.0])
    k = rng.randint(3, 6)
    call = f"make_swiss_roll(n_samples={n}, noise=0.0, random_state={s})"
    items = [
        ("hypot", f"Without noise, a point's distance from the roll's axis in the (x, z) plane equals its position `t`: "
                  f"recover `t` from `X` alone (`np.hypot`) for the first {k} points (it matches `t[:{k}]`).",
         f"result = np.hypot(X[:{k}, 0], X[:{k}, 2])"),
        ("bincount", "The roll is `x = t·cos t, z = t·sin t`, so every 2π of `t` is one turn. Return how many points "
                     "lie in each turn: `np.bincount` of `(t // (2π))` as ints.",
         "result = np.bincount((t // (2 * np.pi)).astype(int))"),
        ("min", "Return the range of `t` and of the height `X[:, 1]` as `(t.min(), t.max(), h.min(), h.max())`.",
         "h = X[:, 1]\nresult = (t.min(), t.max(), h.min(), h.max())", {"require": []}),
        ("argsort", f"Unroll it: the 2-D coordinates `(t, X[:, 1])` lay the sheet flat. Return the indices of the {k} "
                    f"points with the smallest `t`, in increasing order of `t`.",
         f"result = np.argsort(t)[:{k}]", {"require": []}),
        ("make_swiss_roll", f"Regenerate with noise: `make_swiss_roll(n_samples={n}, noise={noise}, random_state={s})`, "
                            f"and return the standard deviation of `√(x² + z²) − t` over all points.",
         f"from sklearn.datasets import make_swiss_roll\nXn, tn = make_swiss_roll(n_samples={n}, noise={noise}, "
         f"random_state={s})\nresult = (np.hypot(Xn[:, 0], Xn[:, 2]) - tn).std()"),
        ("norm", "Points close in 3-D can be far apart on the roll. For point 0, find its nearest other point in 3-D "
                 "(Euclidean) and return `(3-D distance, |t difference|)`.",
         "d = np.linalg.norm(X - X[0], axis=1)\nd[0] = np.inf\nj = d.argmin()\nresult = (d[j], abs(t[j] - t[0]))",
         {"require": []}),
    ]
    return catalog(rng, items, f"from sklearn.datasets import make_swiss_roll\nX, t = {call}",
                   f"`X, t = {call}`: 3-D points on a rolled-up sheet and each point's position `t` along the roll.",
                   tol=1e-6)


@template(M, "medium")
def digits_analysis(rng):
    c, k = rng.randrange(10), rng.randint(2, 4)
    items = [
        ("images", f"Return the average image of digit {c} (8×8).", f"result = digits.images[digits.target == {c}].mean(axis=0)"),
        ("data", "Which digit uses the most ink on average? Return it as an int (mean of the row sums per digit).",
         "ink = digits.data.sum(axis=1)\nresult = int(np.argmax([ink[digits.target == d].mean() for d in range(10)]))"),
        ("data", "How many of the 64 pixel positions are blank (0) in **every** image? Return an int.",
         "result = int((digits.data.max(axis=0) == 0).sum())"),
        ("data", "Nearest-mean classifier: compute the mean image of each digit, assign every image to the digit whose "
                 "mean is closest (Euclidean) and return the accuracy.",
         "means = np.array([digits.data[digits.target == d].mean(axis=0) for d in range(10)])\n"
         "dist = ((digits.data[:, None, :] - means[None]) ** 2).sum(axis=2)\nresult = (dist.argmin(axis=1) == digits.target).mean()"),
        ("images", f"Return the average ink per **row** of the picture (8 values) over all images of digit {c}.",
         f"result = digits.images[digits.target == {c}].sum(axis=2).mean(axis=0)"),
        ("frame", "Load the digits with `as_frame=True` and return the names of the first 3 pixel columns of the "
                  "frame.",
         "from sklearn.datasets import load_digits\nresult = list(load_digits(as_frame=True).frame.columns[:3])",
         {"require": ["frame"]}),
        ("images", f"Mirror-symmetry check: for digit {c}, return the mean absolute difference between each image and "
                   f"its left-right mirror (`[:, :, ::-1]`).",
         f"im = digits.images[digits.target == {c}]\nresult = np.abs(im - im[:, :, ::-1]).mean()"),
        ("data", f"Downsample to 4×4: sum each 2×2 block of every image (shape (1797, 4, 4)) and return the result for "
                 f"the first {k} images.",
         f"result = digits.images[:{k}].reshape({k}, 4, 2, 4, 2).sum(axis=(2, 4))", {"require": ["images"]}),
    ]
    return catalog(rng, items, "from sklearn.datasets import load_digits\ndigits = load_digits()",
                   "`digits = load_digits()`: 1797 8×8 images (`images`, flattened in `data`) and their digit (`target`).",
                   tol=1e-6)


@template(M, "medium")
def regression_recovery(rng):
    n, d, s = rng.randint(80, 200), rng.randint(5, 9), rng.randint(0, 99)
    inf = rng.randint(2, d - 2)
    noise = rng.choice([1.0, 5.0, 20.0])
    call = f"make_regression(n_samples={n}, n_features={d}, n_informative={inf}, noise={noise}, coef=True, random_state={s})"
    setup = f"from sklearn.datasets import make_regression\nX, y, coef = {call}"
    head = f"`X, y, coef = {call}`."
    task = rng.choice(["rank", "r2", "support", "error"])
    if task == "rank":
        return P(f"{head} Rank the features by the size of their **true** coefficient and by the size of the "
                 f"coefficient a `LinearRegression` learns; return whether the {inf} largest are the same set (a bool).",
                 setup, "from sklearn.linear_model import LinearRegression\nest = LinearRegression().fit(X, y).coef_\n"
                        f"result = set(np.argsort(-np.abs(coef))[:{inf}]) == set(np.argsort(-np.abs(est))[:{inf}])",
                 require=["LinearRegression"])
    if task == "r2":
        return P(f"{head} Fit `LinearRegression` on the first half of the rows and return its R² on the second half.",
                 setup, "from sklearn.linear_model import LinearRegression\nh = len(X) // 2\n"
                        "result = LinearRegression().fit(X[:h], y[:h]).score(X[h:], y[h:])",
                 require=["LinearRegression"])
    if task == "support":
        return P(f"{head} Fit `LinearRegression` and guess the informative features as those whose learned "
                 f"|coefficient| exceeds 1. Return `(guess, truth)`: the indices of the guessed features and of the "
                 f"truly informative ones (`coef != 0`), both from `np.flatnonzero`.", setup,
                 "from sklearn.linear_model import LinearRegression\nest = LinearRegression().fit(X, y).coef_\n"
                 "result = (np.flatnonzero(np.abs(est) > 1), np.flatnonzero(coef))", require=["LinearRegression"])
    return P(f"{head} Fit `LinearRegression` and return the root-mean-square error between its training predictions "
             f"and the **noise-free** target `X @ coef` (the model also learns an intercept).", setup,
             "from sklearn.linear_model import LinearRegression\npred = LinearRegression().fit(X, y).predict(X)\n"
             "result = np.sqrt(((pred - X @ coef) ** 2).mean())", require=["LinearRegression"], tol=1e-6)


# ================================================================= hard

@template(M, "hard")
def toy_shapes_by_hand(rng):
    n = rng.randint(7, 25)
    factor = rng.choice([0.3, 0.5, 0.8])
    task = rng.choice(["moons", "circles", "labels"])
    if task == "moons":
        return P(f"Rebuild `make_moons(n_samples={n}, noise=None, shuffle=False)` with NumPy alone. It puts `n_out = n "
                 f"// 2` points on the upper half of the unit circle, `x = cos θ, y = sin θ` for θ = `np.linspace(0, π, "
                 f"n_out)`, then `n_in = n − n_out` points at `x = 1 − cos θ, y = 1 − sin θ − 0.5` for θ = "
                 f"`np.linspace(0, π, n_in)`. Return `X` (the outer moon first) and check it against scikit-learn.",
                 f"n = {n}",
                 "n_out = n // 2\nn_in = n - n_out\na, b = np.linspace(0, np.pi, n_out), np.linspace(0, np.pi, n_in)\n"
                 "result = np.vstack([np.column_stack([np.cos(a), np.sin(a)]), "
                 "np.column_stack([1 - np.cos(b), 1 - np.sin(b) - 0.5])])", tol=1e-9)
    if task == "circles":
        return P(f"Rebuild `make_circles(n_samples={n}, factor={factor}, noise=None, shuffle=False)` with NumPy alone: "
                 f"`n_out = n // 2` points evenly spaced on the unit circle starting at angle 0 (`np.linspace(0, 2π, "
                 f"n_out, endpoint=False)`), then `n_in = n − n_out` points spaced the same way on the circle of radius "
                 f"`factor`. Return `X`.", f"n = {n}",
                 "n_out = n // 2\nn_in = n - n_out\na = np.linspace(0, 2 * np.pi, n_out, endpoint=False)\n"
                 "b = np.linspace(0, 2 * np.pi, n_in, endpoint=False)\n"
                 f"result = np.vstack([np.column_stack([np.cos(a), np.sin(a)]), {factor} * np.column_stack([np.cos(b), "
                 "np.sin(b)])])", tol=1e-9)
    gen = rng.choice(["make_moons", "make_circles"])
    s = rng.randint(0, 99)
    return P(f"`{gen}(n_samples={n}, shuffle=False)` returns the labels in blocks (`n // 2` zeros, then ones). Return "
             f"`(y_blocks, y_shuffled_counts)`: the label array of `{gen}(n_samples={n}, shuffle=False)` and "
             f"`np.bincount` of the labels of `{gen}(n_samples={n}, random_state={s})` (shuffled: same counts).", "",
             f"from sklearn.datasets import {gen}\nX, y_blocks = {gen}(n_samples={n}, shuffle=False)\n"
             f"X2, y2 = {gen}(n_samples={n}, random_state={s})\nresult = (y_blocks, np.bincount(y2))", require=[gen])


@template(M, "hard")
def sorted_rows_pitfall(rng):
    key = rng.choice(["iris", "wine"])
    L, n, d, k, what = _LOAD[key]
    cut = rng.choice([100, 110, 120]) if key == "iris" else rng.choice([110, 120, 130])
    s = rng.randint(0, 999)
    depth = rng.randint(2, 4)
    tree = f"DecisionTreeClassifier(max_depth={depth}, random_state=0)"
    setup = f"from sklearn.datasets import {L}\nX, y = {L}(return_X_y=True)"
    head = (f"`X, y = {L}(return_X_y=True)` ({n} {what}). A colleague trains on the first {cut} rows and tests on the "
            f"rest, not knowing the rows are **sorted by class**.")
    task = rng.choice(["unseen", "broken", "fixed", "counts"])
    if task == "unseen":
        return P(f"{head} Return the sorted list of classes that appear in their test rows but never in their training "
                 f"rows.", setup, f"result = sorted(set(y[{cut}:].tolist()) - set(y[:{cut}].tolist()))",
                 require=[])
    if task == "broken":
        return P(f"{head} Fit `{tree}` (from `sklearn.tree`) their way and return its test accuracy.", setup,
                 f"from sklearn.tree import DecisionTreeClassifier\nresult = {tree}.fit(X[:{cut}], y[:{cut}])"
                 f".score(X[{cut}:], y[{cut}:])", require=["DecisionTreeClassifier"])
    if task == "fixed":
        return P(f"{head} Fix it: shuffle the rows with `order = np.random.default_rng({s}).permutation(len(X))` first, "
                 f"then do the same split and fit `{tree}`. Return `(broken_accuracy, fixed_accuracy)`.", setup,
                 f"from sklearn.tree import DecisionTreeClassifier\nbroken = {tree}.fit(X[:{cut}], y[:{cut}])"
                 f".score(X[{cut}:], y[{cut}:])\norder = np.random.default_rng({s}).permutation(len(X))\n"
                 f"Xs, ys = X[order], y[order]\nfixed = {tree}.fit(Xs[:{cut}], ys[:{cut}]).score(Xs[{cut}:], ys[{cut}:])\n"
                 "result = (broken, fixed)", require=["DecisionTreeClassifier"])
    return P(f"{head} Return the per-class counts (`np.bincount(..., minlength=3)`) of their training labels and of the "
             f"training labels after shuffling the rows with `np.random.default_rng({s}).permutation(len(X))`, as a "
             f"tuple of two arrays.", setup,
             f"order = np.random.default_rng({s}).permutation(len(X))\n"
             f"result = (np.bincount(y[:{cut}], minlength=3), np.bincount(y[order][:{cut}], minlength=3))", require=[])


@template(M, "hard")
def accuracy_paradox(rng):
    n, s = rng.choice([400, 600, 800]), rng.randint(0, 99)
    p = rng.choice([0.05, 0.08, 0.1])
    call = f"make_classification(n_samples={n}, n_features=6, weights=[{1 - p:.2f}, {p}], flip_y=0, class_sep=0.6, random_state={s})"
    h = int(n * 0.7)
    setup = (f"from sklearn.datasets import make_classification\nX, y = {call}\n"
             f"X_tr, X_te, y_tr, y_te = X[:{h}], X[{h}:], y[:{h}], y[{h}:]")
    head = (f"`X, y = {call}` has {p:.0%} positives; `X_tr, y_tr` are the first {h} rows and `X_te, y_te` the rest. "
            f"Model: `{LR}`.")
    lr = f"from sklearn.linear_model import LogisticRegression\npred = {LR}.fit(X_tr, y_tr).predict(X_te)\n"
    task = rng.choice(["baseline", "recall", "balanced", "missed"])
    if task == "baseline":
        return P(f"{head} Return `(baseline, model)`: the test accuracy of always predicting the majority class of "
                 f"`y_tr`, and the test accuracy of the model. High accuracy alone says little here.", setup,
                 lr + "majority = np.bincount(y_tr).argmax()\nresult = ((y_te == majority).mean(), (pred == y_te).mean())",
                 require=["LogisticRegression"])
    if task == "recall":
        return P(f"{head} Return the model's **recall** on the positives of the test set: the fraction of test rows "
                 f"with `y_te == 1` that it predicts as 1.", setup, lr + "result = (pred[y_te == 1] == 1).mean()",
                 require=["LogisticRegression"])
    if task == "balanced":
        return P(f"{head} Compare test recall on the positives for the plain model and for `LogisticRegression("
                 f"max_iter=1000, class_weight='balanced')`, which up-weights the rare class. Return `(plain, "
                 f"balanced)`.", setup,
                 lr + "bal = LogisticRegression(max_iter=1000, class_weight='balanced').fit(X_tr, y_tr).predict(X_te)\n"
                      "result = ((pred[y_te == 1] == 1).mean(), (bal[y_te == 1] == 1).mean())",
                 require=["LogisticRegression", "class_weight"])
    return P(f"{head} How many positive test rows does the model miss (predict as 0), and how many negative test rows "
             f"does it flag as 1? Return `(missed, false_alarms)` as ints.", setup,
             lr + "result = (int(((y_te == 1) & (pred == 0)).sum()), int(((y_te == 0) & (pred == 1)).sum()))",
             require=["LogisticRegression"])


@template(M, "hard")
def classification_column_layout(rng):
    n, s = rng.choice([150, 200, 300]), rng.randint(0, 99)
    inf, red, rep, useless = rng.randint(2, 4), rng.randint(1, 3), rng.randint(0, 2), rng.randint(1, 3)
    d = inf + red + rep + useless
    call = (f"make_classification(n_samples={n}, n_features={d}, n_informative={inf}, n_redundant={red}, "
            f"n_repeated={rep}, shuffle=False, random_state={s})")
    setup = f"from sklearn.datasets import make_classification\nX, y = {call}"
    head = (f"`X, y = {call}`. With `shuffle=False` the columns are, in order: {inf} informative, {red} redundant "
            f"(linear combinations of the informative ones), {rep} repeated (copies of earlier columns) and {useless} "
            f"pure-noise columns.")
    task = rng.choice(["residuals", "repeats", "subset", "rank"])
    if task == "residuals":
        return P(f"{head} Recover how the redundant columns were mixed: solve `X[:, :{inf}] @ W = X[:, {inf}:{inf + red}]` "
                 f"with `np.linalg.lstsq` (the fit is exact) and return `W` ({inf} × {red}).", setup,
                 f"result = np.linalg.lstsq(X[:, :{inf}], X[:, {inf}:{inf + red}], rcond=None)[0]", require=["lstsq"],
                 tol=1e-6)
    if task == "repeats":
        return P(f"{head} For each repeated column, return the index of the **first** earlier column it copies, as a "
                 f"list (empty if there are none).", setup,
                 f"result = [next(i for i in range(j) if np.array_equal(X[:, i], X[:, j])) for j in range({inf + red}, "
                 f"{inf + red + rep})]", require=[])
    if task == "subset":
        return P(f"{head} Fit `{LR}` on the informative columns only and on the noise columns only; return the two "
                 f"training accuracies as a tuple.", setup,
                 f"from sklearn.linear_model import LogisticRegression\ninf = X[:, :{inf}]\nnoise = X[:, {d - useless}:]\n"
                 f"result = ({LR}.fit(inf, y).score(inf, y), {LR}.fit(noise, y).score(noise, y))",
                 require=["LogisticRegression"])
    return P(f"{head} Only informative and noise columns add rank. Return `(rank of X, rank of X without its noise "
             f"columns)` with `np.linalg.matrix_rank`.", setup,
             f"result = (int(np.linalg.matrix_rank(X)), int(np.linalg.matrix_rank(X[:, :{d - useless}])))",
             require=["matrix_rank"])


@template(M, "hard")
def nearest_centre_rule(rng):
    n, k, s = rng.choice([150, 200, 300]), rng.randint(3, 5), rng.randint(0, 99)
    std = rng.choice([1.5, 2.0, 2.5, 3.0])
    call = f"make_blobs(n_samples={n}, centers={k}, cluster_std={std}, return_centers=True, random_state={s})"
    setup = f"from sklearn.datasets import make_blobs\nX, y, centers = {call}"
    head = f"`X, y, centers = {call}`: overlapping blobs and their true centres."
    task = rng.choice(["true_rule", "confused", "centroid", "gap"])
    dist = "D = np.linalg.norm(X[:, None, :] - centers[None], axis=2)\n"
    if task == "true_rule":
        return P(f"{head} Classify every point by its nearest **true** centre and return the accuracy (even the true "
                 f"centres can't be perfect when blobs overlap).", setup,
                 dist + "result = (D.argmin(axis=1) == y).mean()", require=[])
    if task == "confused":
        return P(f"{head} Return the indices of the points that are closer to another blob's centre than to their own.",
                 setup, dist + "result = np.flatnonzero(D.argmin(axis=1) != y)", require=[])
    if task == "centroid":
        return P(f"{head} `NearestCentroid` (from `sklearn.neighbors`) learns one mean per class. Fit it on `X, y` and "
                 f"return the largest distance between a learned centroid (`centroids_`) and the true centre of its "
                 f"class.", setup,
                 "from sklearn.neighbors import NearestCentroid\nnc = NearestCentroid().fit(X, y)\n"
                 "result = np.linalg.norm(nc.centroids_ - centers, axis=1).max()", require=["NearestCentroid"])
    return P(f"{head} Return `(true_centre_accuracy, nearest_centroid_accuracy)`: classifying by the nearest true "
             f"centre vs. by a `NearestCentroid` fitted on `X, y` (training accuracy).", setup,
             "from sklearn.neighbors import NearestCentroid\n" + dist +
             "result = ((D.argmin(axis=1) == y).mean(), NearestCentroid().fit(X, y).score(X, y))",
             require=["NearestCentroid"])


@template(M, "hard")
def noise_sweep(rng):
    s = rng.randint(0, 99)
    task = rng.choice(["regression", "flip", "moons", "circles"])
    if task == "regression":
        noises = sorted(rng.sample([0.0, 5.0, 10.0, 25.0, 50.0, 100.0], 3))
        return P(f"For each `noise` in `{noises}`, generate `make_regression(n_samples=200, n_features=5, noise=noise, "
                 f"random_state={s})`, fit `LinearRegression` on the first 100 rows and record R² on the last 100. "
                 f"Return the list (R² falls as noise grows).", "",
                 "from sklearn.datasets import make_regression\nfrom sklearn.linear_model import LinearRegression\n"
                 f"result = []\nfor noise in {noises}:\n    X, y = make_regression(n_samples=200, n_features=5, "
                 f"noise=noise, random_state={s})\n    result.append(LinearRegression().fit(X[:100], y[:100])"
                 ".score(X[100:], y[100:]))", require=["make_regression", "LinearRegression"])
    if task == "flip":
        flips = sorted(rng.sample([0.0, 0.05, 0.1, 0.2, 0.3, 0.4], 3))
        return P(f"For each `flip_y` in `{flips}`, generate `make_classification(n_samples=300, n_features=5, "
                 f"flip_y=flip_y, random_state={s})`, fit `{LR}` on the first 200 rows and record the accuracy on the "
                 f"last 100. Return the list.", "",
                 "from sklearn.datasets import make_classification\nfrom sklearn.linear_model import LogisticRegression\n"
                 f"result = []\nfor f in {flips}:\n    X, y = make_classification(n_samples=300, n_features=5, flip_y=f, "
                 f"random_state={s})\n    result.append({LR}.fit(X[:200], y[:200]).score(X[200:], y[200:]))",
                 require=["make_classification", "flip_y"])
    noises = sorted(rng.sample([0.05, 0.1, 0.2, 0.3, 0.5], 3))
    gen = "make_moons" if task == "moons" else "make_circles"
    extra = "" if task == "moons" else ", factor=0.5"
    k = rng.choice([5, 9])
    return P(f"For each `noise` in `{noises}`, generate `{gen}(n_samples=300, noise=noise{extra}, random_state={s})`, "
             f"fit `KNeighborsClassifier(n_neighbors={k})` on the even rows and record the accuracy on the odd rows. "
             f"Return the list.", "",
             f"from sklearn.datasets import {gen}\nfrom sklearn.neighbors import KNeighborsClassifier\nresult = []\n"
             f"for noise in {noises}:\n    X, y = {gen}(n_samples=300, noise=noise{extra}, random_state={s})\n"
             f"    result.append(KNeighborsClassifier(n_neighbors={k}).fit(X[::2], y[::2]).score(X[1::2], y[1::2]))",
             require=[gen, "KNeighborsClassifier"])


@template(M, "hard")
def balanced_subsample(rng):
    n, s, s2 = rng.choice([500, 600, 800]), rng.randint(0, 99), rng.randint(0, 999)
    p = rng.choice([0.05, 0.1, 0.15])
    call = f"make_classification(n_samples={n}, n_features=5, weights=[{1 - p:.2f}, {p}], flip_y=0, random_state={s})"
    setup = f"from sklearn.datasets import make_classification\nX, y = {call}"
    head = (f"`X, y = {call}`. Undersample the majority class: keep every positive row and draw as many negative rows "
            f"with `np.random.default_rng({s2}).choice(neg, size=len(pos), replace=False)`, where `pos` and `neg` are "
            f"the indices of the positive and negative rows (`np.flatnonzero`). Keep the rows in the order `[*pos, "
            f"*chosen]`.")
    sub = (f"pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)\n"
           f"chosen = np.random.default_rng({s2}).choice(neg, size=len(pos), replace=False)\n"
           "idx = np.concatenate([pos, chosen])\nXb, yb = X[idx], y[idx]\n")
    task = rng.choice(["counts", "recall", "indices"])
    if task == "counts":
        return P(f"{head} Return `(np.bincount(y), np.bincount(yb))`: the class counts before and after.", setup,
                 sub + "result = (np.bincount(y), np.bincount(yb))", require=["make_classification|flatnonzero"])
    if task == "recall":
        return P(f"{head} Fit `{LR}` on the full data and on the balanced subsample, and return the recall of each on "
                 f"the positives of the **full** data, as `(full_model, balanced_model)`.", setup,
                 sub + f"from sklearn.linear_model import LogisticRegression\nfull = {LR}.fit(X, y).predict(X)\n"
                       f"bal = {LR}.fit(Xb, yb).predict(X)\nresult = ((full[y == 1] == 1).mean(), (bal[y == 1] == 1).mean())",
                 require=["LogisticRegression"])
    return P(f"{head} Return the first 5 of the chosen negative indices, sorted ascending.", setup,
             sub + "result = np.sort(chosen)[:5]", require=["choice"])


@template(M, "hard")
def dataset_workflows(rng):
    s, n = rng.randint(0, 99), rng.choice([200, 240, 300])
    d = rng.randint(3, 5)
    k = rng.choice([5, 7, 9])
    pct = rng.choice([10, 20, 25])
    noise = rng.choice([0.1, 0.15, 0.2])
    items = [
        ("target_names", "Load iris with `as_frame=True` and return a 3 × 4 DataFrame of the mean of every feature per "
                         "species, indexed by species **name** (from `target_names`).",
         "from sklearn.datasets import load_iris\ndata = load_iris(as_frame=True)\ndf = data.frame\n"
         "names = data.target_names[df['target'].to_numpy()]\nresult = df.drop(columns='target').groupby(names).mean()"),
        ("make_classification", f"Generate `X, y = make_classification(n_samples={n}, n_features={d}, n_informative=2, "
                                f"n_redundant=0, random_state={s})`, put it in a DataFrame with columns `f0` … "
                                f"`f{d - 1}` plus a `label` column, and return the mean of every feature per label "
                                f"(`groupby('label').mean()`).",
         f"from sklearn.datasets import make_classification\nX, y = make_classification(n_samples={n}, n_features={d}, "
         f"n_informative=2, n_redundant=0, random_state={s})\ndf = pd.DataFrame(X, columns=[f'f{{i}}' for i in range({d})])\n"
         "df['label'] = y\nresult = df.groupby('label').mean()"),
        ("make_moons", f"Does a model trained on one sample of a generator work on a fresh one? Fit "
                       f"`KNeighborsClassifier(n_neighbors={k})` on `make_moons(n_samples={n}, noise={noise}, "
                       f"random_state={s})` and return its accuracy on `make_moons(n_samples={n}, noise={noise}, "
                       f"random_state={s + 1})` and on `make_moons(n_samples={n}, noise={noise * 3:.2f}, "
                       f"random_state={s + 1})` (noisier), as a tuple.",
         f"from sklearn.datasets import make_moons\nfrom sklearn.neighbors import KNeighborsClassifier\n"
         f"X, y = make_moons(n_samples={n}, noise={noise}, random_state={s})\n"
         f"knn = KNeighborsClassifier(n_neighbors={k}).fit(X, y)\n"
         f"A, a = make_moons(n_samples={n}, noise={noise}, random_state={s + 1})\n"
         f"B, b = make_moons(n_samples={n}, noise={noise * 3:.2f}, random_state={s + 1})\nresult = (knn.score(A, a), knn.score(B, b))"),
        ("load_breast_cancer", f"Load the breast-cancer data as a DataFrame (`as_frame=True`). Among the {pct}% of "
                               f"samples with the largest `mean radius` (those at or above its {100 - pct}th "
                               f"percentile, `quantile`), what fraction is malignant (`target == 0`)?",
         "from sklearn.datasets import load_breast_cancer\ndf = load_breast_cancer(as_frame=True).frame\n"
         f"big = df['mean radius'] >= df['mean radius'].quantile({(100 - pct) / 100})\n"
         "result = (df.loc[big, 'target'] == 0).mean()"),
        ("return_indicator", f"`make_multilabel_classification(n_samples={n}, n_classes={d + 2}, "
                             f"return_indicator='sparse', random_state={s})` returns `Y` as a SciPy sparse matrix. "
                             f"Return `(Y.nnz, Y.shape, label counts)` where the counts come from the dense array "
                             f"(`Y.toarray().sum(axis=0)`).",
         f"from sklearn.datasets import make_multilabel_classification\nX, Y = make_multilabel_classification("
         f"n_samples={n}, n_classes={d + 2}, return_indicator='sparse', random_state={s})\n"
         "result = (Y.nnz, Y.shape, Y.toarray().sum(axis=0))"),
        ("make_regression", f"Outliers pull least squares. Generate `X, y, coef = make_regression(n_samples={n}, "
                            f"n_features={d}, noise=5.0, coef=True, random_state={s})`, then add 500 to every 10th "
                            f"target (`y[::10]`) in a copy. Fit `LinearRegression` on the clean and on the corrupted "
                            f"target and return each model's largest |coefficient error| against `coef`, as a tuple.",
         f"from sklearn.datasets import make_regression\nfrom sklearn.linear_model import LinearRegression\n"
         f"X, y, coef = make_regression(n_samples={n}, n_features={d}, noise=5.0, coef=True, random_state={s})\n"
         "bad = y.copy()\nbad[::10] += 500\nerr = lambda t: np.abs(LinearRegression().fit(X, t).coef_ - coef).max()\n"
         "result = (err(y), err(bad))"),
        ("make_blobs", f"Generate `X, y, centers = make_blobs(n_samples={n}, n_features={d}, centers=3, "
                       f"return_centers=True, random_state={s})` and return the 3 × 3 matrix of distances between the "
                       f"true centres.",
         f"from sklearn.datasets import make_blobs\nX, y, centers = make_blobs(n_samples={n}, n_features={d}, centers=3, "
         f"return_centers=True, random_state={s})\nresult = np.linalg.norm(centers[:, None] - centers[None], axis=2)"),
        ("load_wine", "Which single wine feature best separates class 0 from the others with one threshold at its "
                      "median? For every feature compute the accuracy of predicting class 0 when the value is above "
                      "the median, or below-or-equal (whichever is better), and return the best feature's name.",
         "from sklearn.datasets import load_wine\ndata = load_wine()\nX, is0 = data.data, data.target == 0\n"
         "above = X > np.median(X, axis=0)\nacc = (above == is0[:, None]).mean(axis=0)\nacc = np.maximum(acc, 1 - acc)\n"
         "result = data.feature_names[int(acc.argmax())]"),
    ]
    return catalog(rng, items, "", "Small workflows around the dataset functions of `sklearn.datasets`.", tol=1e-6)


# ================================================================= challenge

@template(M, "challenge")
def shortest_class_window(rng):
    n, k, m, s = rng.randint(14, 24), rng.randint(2, 4), rng.randint(1, 2), seeded(rng)
    report = rng.choice(["length", "start", "count"])
    setup = (f"g = np.random.default_rng({s})\nk, m = {k}, {m}\ny = np.repeat(np.tile(g.permutation(k), 12), "
             f"g.integers(1, 4, 12 * k))[:{n}]\nreport = {report!r}")
    sol = ("cnt = [0] * k\nneed = k\nl = 0\nbest, start, count = None, -1, 0\nyl = y.tolist()\n"
           "for r, c in enumerate(yl):\n    cnt[c] += 1\n    if cnt[c] == m:\n        need -= 1\n"
           "    if need:\n        continue\n    while cnt[yl[l]] > m:\n        cnt[yl[l]] -= 1\n        l += 1\n"
           "    length = r - l + 1\n    if best is None or length < best:\n        best, start, count = length, l, 1\n"
           "    elif length == best:\n        count += 1\n"
           "result = -1 if best is None else {'length': best, 'start': start, 'count': count}[report]")
    tests = [f"g = np.random.default_rng({s + i})\nk, m = {kk}, {mm}\ny = np.repeat(g.integers(0, k, 120000), "
             f"g.integers(1, {run}, 120000))[:200000]\nreport = {report!r}"
             for i, (kk, mm, run) in ((1, (5, 3, 40)), (2, (8, 2, 300)))]
    what = {"length": "the length of the shortest such slice",
            "start": "the start index `l` of the shortest such slice (the leftmost one if several are shortest)",
            "count": "how many slices have that shortest length"}[report]
    return challenge(
        "Shortest slice with every class",
        "Some datasets (iris, wine) store their rows sorted by class, so a contiguous slice of rows can miss classes "
        f"entirely. The labels `y` hold classes `0 … k-1` in long runs. Find the contiguous slices `y[l:r]` that contain "
        f"**at least `m` samples of every class** and return {what}; return -1 if no slice qualifies. (`report` says "
        f"which: {report!r}.)",
        setup, sol, tests, "len(y) ≤ 2·10^5, k ≤ 10", speed="O(n): two pointers", answer_desc=" (an int)")


@template(M, "challenge")
def label_set_pairs(rng):
    n, L, s = rng.randint(10, 18), rng.randint(3, 4), seeded(rng)
    report = rng.choice(["pairs", "distinct", "top"])
    setup = (f"from sklearn.datasets import make_multilabel_classification\nX, Y = make_multilabel_classification("
             f"n_samples={n}, n_classes={L}, n_labels=1, random_state={s})\ndel X\nreport = {report!r}")
    sol = ("codes = Y.astype(np.int64) @ (1 << np.arange(Y.shape[1], dtype=np.int64))\n"
           "vals, counts = np.unique(codes, return_counts=True)\n"
           "if report == 'pairs':\n    result = int((counts * (counts - 1) // 2).sum())\n"
           "elif report == 'distinct':\n    result = len(vals)\nelse:\n    best = int(vals[counts.argmax()])\n"
           "    result = [j for j in range(Y.shape[1]) if best >> j & 1]")
    tests = [f"g = np.random.default_rng({s + i})\nY = (g.random(({rows}, {cols})) < {pr}).astype(int)\nreport = {report!r}"
             for i, (rows, cols, pr) in ((1, (20000, 10, 0.2)), (2, (12000, 16, 0.08)))]
    what = {"pairs": "the number of unordered pairs of samples `(i, j)`, `i < j`, with exactly the same label set",
            "distinct": "the number of distinct label sets (rows of `Y`)",
            "top": "the most frequent label set, as a sorted list of label indices. Encode a set as the integer "
                   "`Σ 2^j` over its labels `j`; on ties pick the set with the smallest code (the empty set is `[]`)"}[report]
    return challenge(
        "Duplicate label sets",
        "`Y` is a multilabel indicator matrix like the one `make_multilabel_classification` returns: row `i` has a 1 in "
        f"column `j` when sample `i` carries label `j`. Before training a label-powerset model you need statistics of "
        f"the label sets: return {what}. (`report`: {report!r}.)",
        setup, sol, tests, "rows ≤ 2·10^4, labels ≤ 16", speed="O(rows · labels): encode each row as one integer",
        answer_desc=" (an int)" if report != "top" else " (a list of ints)")


@template(M, "challenge")
def swiss_roll_unrolled(rng):
    n, s = rng.randint(8, 14), seeded(rng)
    report = rng.choice(["nearest", "gap", "pairs"])
    D = rng.choice([1.0, 2.0, 4.0])
    setup = (f"from sklearn.datasets import make_swiss_roll\nX, t = make_swiss_roll(n_samples={n}, noise=0.0, "
             f"random_state={s})\ndel t\nD = {D}\nreport = {report!r}")
    sol = ("t = np.hypot(X[:, 0], X[:, 2])\nu = (t * np.sqrt(1 + t * t) + np.arcsinh(t)) / 2\n"
           "if report == 'nearest':\n    o = np.argsort(u)\n    us = u[o]\n    gaps = np.diff(us)\n"
           "    best = np.minimum(np.concatenate([[np.inf], gaps]), np.concatenate([gaps, [np.inf]]))\n"
           "    result = np.empty(len(u))\n    result[o] = best\n"
           "elif report == 'gap':\n    result = float(np.diff(np.sort(u)).max())\nelse:\n    us = np.sort(u)\n"
           "    result = int((np.searchsorted(us, us + D, side='right') - np.arange(len(us)) - 1).sum())")
    tests = [f"from sklearn.datasets import make_swiss_roll\nX, t = make_swiss_roll(n_samples=200000, noise=0.0, "
             f"random_state={s + i})\ndel t\nD = {D / 1000}\nreport = {report!r}" for i in (1, 2)]
    what = {"nearest": "for every point, the unrolled distance to its nearest other point, as a float array in the "
                       "order of `X`",
            "gap": "the largest gap between consecutive points along the unrolled sheet (a float)",
            "pairs": "the number of unordered pairs of points whose unrolled distance is at most `D` (an int)"}[report]
    return challenge(
        "Unrolling the swiss roll",
        "`X` comes from `make_swiss_roll(noise=0.0)`, with its `t` thrown away: each point is `(t·cos t, h, t·sin t)` "
        "for an unknown `t` > 0 and height `h`. Ignoring the height, the distance between two points **along the "
        "sheet** is the arc length of the spiral `(t cos t, t sin t)` between their `t` values, and `∫ √(1 + t²) dt = "
        f"(t√(1 + t²) + asinh t) / 2`. Return {what}. (`report`: {report!r}.)",
        setup, sol, tests, "len(X) ≤ 2·10^5", speed="O(n log n): recover t, unroll, sort", tol=1e-6)


@template(M, "challenge")
def circles_radius_stump(rng):
    n, s = rng.randint(10, 18), seeded(rng)
    noise, factor = rng.choice([0.1, 0.15, 0.2]), rng.choice([0.5, 0.6, 0.7])
    report = rng.choice(["errors", "accuracy", "threshold"])
    setup = (f"from sklearn.datasets import make_circles\nX, y = make_circles(n_samples={n}, noise={noise}, "
             f"factor={factor}, random_state={s})\nreport = {report!r}")
    sol = ("r = np.hypot(X[:, 0], X[:, 1])\no = np.argsort(r, kind='stable')\nrs, ys = r[o], y[o]\n"
           "outer_in = np.cumsum(ys == 0)\ninner_out = (ys == 1).sum() - np.cumsum(ys == 1)\n"
           "err = outer_in + inner_out\nbest = int(err.min())\nnone = int((y == 1).sum())\n"
           "if report == 'errors':\n    result = min(best, none)\nelif report == 'accuracy':\n"
           "    result = 1 - min(best, none) / len(y)\nelse:\n    result = -1.0 if none <= best else float(rs[err.argmin()])")
    tests = [f"from sklearn.datasets import make_circles\nX, y = make_circles(n_samples=200000, noise={noise}, "
             f"factor={factor}, random_state={s + i})\nreport = {report!r}" for i in (1, 2)]
    what = {"errors": "the smallest number of misclassified points any `θ` achieves (an int)",
            "accuracy": "the best training accuracy any `θ` achieves (a float)",
            "threshold": "the smallest radius `θ` among the points' own radii that achieves the fewest errors (a "
                         "float), or -1.0 if predicting no inner point at all is at least as good"}[report]
    return challenge(
        "One-split circle classifier",
        "`make_circles` puts class 1 on an inner circle and class 0 on an outer one. A decision stump on the radius "
        "`r = √(x² + y²)` predicts class 1 when `r ≤ θ` and class 0 otherwise. Over every threshold `θ` (try the "
        f"points' own radii, and also a θ below all of them), return {what}. (`report`: {report!r}.)",
        setup, sol, tests, "len(X) ≤ 2·10^5", speed="O(n log n): sort once, prefix counts", tol=1e-9)


@template(M, "challenge")
def digit_holes(rng):
    m, thr, s = rng.randint(4, 7), rng.choice([4, 6, 8]), seeded(rng)
    report = rng.choice(["holes", "strokes", "histogram"])
    setup = (f"from sklearn.datasets import load_digits\nimgs = load_digits().images[np.random.default_rng({s})"
             f".choice(1797, {m}, replace=False)].astype(int)\nthr = {thr}\nreport = {report!r}")
    sol = ("from scipy import ndimage\nink = imgs > thr\nnb = ink.shape[0]\n"
           "if report == 'strokes':\n    st = np.zeros((3, 3, 3), bool)\n    st[1] = True\n"
           "    lab, _ = ndimage.label(ink, structure=st)\n    out = np.zeros(nb, int)\n"
           "    for i in range(nb):\n        out[i] = len(np.unique(lab[i])) - 1\n    result = out\nelse:\n"
           "    bg = np.pad(~ink, ((0, 0), (1, 1), (1, 1)), constant_values=True)\n"
           "    st = np.zeros((3, 3, 3), bool)\n    st[1] = ndimage.generate_binary_structure(2, 1)\n"
           "    lab, _ = ndimage.label(bg, structure=st)\n"
           "    holes = np.array([len(np.unique(lab[i][lab[i] > 0])) - 1 for i in range(nb)])\n"
           "    result = holes if report == 'holes' else np.bincount(np.minimum(holes, 2), minlength=3)")
    tests = [f"g = np.random.default_rng({s + i})\nimgs = g.integers(0, 17, ({cnt}, 8, 8))\nthr = {thr}\nreport = {report!r}"
             for i, cnt in ((1, 3000), (2, 2500))]
    what = {"holes": "the number of **holes** of every image: connected regions of non-ink pixels (4-neighbour "
                     "connectivity: up, down, left, right) that don't touch the image border. A 0 has 1 hole, an 8 has 2",
            "strokes": "the number of **strokes** of every image: connected regions of ink pixels, where pixels touching "
                       "by a side or a corner (8 neighbours) are connected",
            "histogram": "how many images have 0, 1, and 2-or-more **holes**, as an array of 3 counts. A hole is a "
                         "connected region of non-ink pixels (4-neighbour connectivity: up, down, left, right) that "
                         "doesn't touch the image border"}[report]
    return challenge(
        "Holes in handwritten digits",
        f"`imgs` holds 8×8 digit images from `load_digits` (pixel values 0-16). A pixel is **ink** when its value is "
        f"greater than `thr` = {thr}. Topological features like holes help tell digits apart: return {what}. "
        f"(`report`: {report!r}.)",
        setup, sol, tests, "number of images ≤ 3000 (8×8 each)", speed="linear in the number of pixels (flood fill or "
        "union-find)", answer_desc=" (an int array)")


@template(M, "challenge")
def class_slice_queries(rng):
    n, k, q, s = rng.randint(12, 20), rng.randint(3, 4), rng.randint(4, 6), seeded(rng)
    report = rng.choice(["distinct", "missing", "runs"])
    setup = (f"g = np.random.default_rng({s})\nk = {k}\ny = np.repeat(g.integers(0, k, 30), g.integers(1, 5, 30))[:{n}]\n"
             f"l = g.integers(0, {n} - 1, {q})\nr = l + 1 + g.integers(0, {n}, {q}) % ({n} - l)\nreport = {report!r}")
    sol = ("if report == 'runs':\n    change = np.concatenate([[0], np.cumsum(y[1:] != y[:-1])])\n"
           "    result = change[r - 1] - change[l] + 1\nelse:\n"
           "    P = np.zeros((len(y) + 1, k), np.int64)\n    P[1:] = np.cumsum(np.eye(k, dtype=np.int64)[y], axis=0)\n"
           "    present = (P[r] - P[l]) > 0\n    if report == 'distinct':\n        result = present.sum(axis=1)\n"
           "    else:\n        absent = ~present\n        result = np.where(absent.any(axis=1), absent.argmax(axis=1), -1)")
    tests = [f"g = np.random.default_rng({s + i})\nk = {kk}\ny = np.repeat(g.integers(0, k, 100000), "
             f"g.integers(1, {run}, 100000))[:200000]\nl = g.integers(0, 199999, 200000)\n"
             f"r = l + 1 + g.integers(0, 200000, 200000) % (200000 - l)\nreport = {report!r}"
             for i, (kk, run) in ((1, (10, 30)), (2, (4, 400)))]
    what = {"distinct": "how many different classes the slice contains",
            "missing": "the smallest class in `0 … k-1` that does **not** appear in the slice (-1 if all appear)",
            "runs": "how many maximal runs of equal labels the slice consists of (`[0, 0, 2, 2, 0]` has 3)"}[report]
    return challenge(
        "Class checks on row slices",
        "Before training on a block of rows of a class-sorted dataset, a script checks which classes the block holds. "
        f"`y` holds labels `0 … k-1`; query `i` asks about the slice `y[l[i]:r[i]]` (never empty). For each query "
        f"return {what}. (`report`: {report!r}.)",
        setup, sol, tests, "len(y), number of queries ≤ 2·10^5, k ≤ 10",
        speed="O((n + q) · k): prefix counts, no loop over slices", answer_desc=" (an int array, one value per query)")
