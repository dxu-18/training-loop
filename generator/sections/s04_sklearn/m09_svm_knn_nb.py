"""4.09 SVMs, nearest neighbours & naive Bayes: SVC/LinearSVC/SVR/OneClassSVM, k-NN classification and regression,
NearestNeighbors retrieval, NearestCentroid, and the naive Bayes family (Gaussian, multinomial, complement,
Bernoulli, categorical)."""

from ._common import (P, catalog, challenge, seeded, synth_reg, table, template)

M = "4.09"
SVM = "from sklearn.svm import "
NB = "from sklearn.naive_bayes import "
NN = "from sklearn.neighbors import "


def _scaled_clf(rng, binary=False):
    """Standardized classification data (rows shuffled): X, y."""
    s = seeded(rng)
    r = rng.random()
    if r < 0.35:
        return ("from sklearn.datasets import load_breast_cancer\nfrom sklearn.preprocessing import StandardScaler\n"
                f"X, y = load_breast_cancer(return_X_y=True)\norder = np.random.default_rng({s}).permutation(len(y))\n"
                "X, y = StandardScaler().fit_transform(X)[order], y[order]",
                "`X` holds 569 breast-tumour samples (30 standardized measurements; rows shuffled) and `y` the diagnosis "
                "(0 malignant, 1 benign)", 2, 30)
    if r < 0.6 and not binary:
        return ("from sklearn.datasets import load_wine\nfrom sklearn.preprocessing import StandardScaler\n"
                f"X, y = load_wine(return_X_y=True)\norder = np.random.default_rng({s}).permutation(len(y))\n"
                "X, y = StandardScaler().fit_transform(X)[order], y[order]",
                "`X` holds 178 wines (13 standardized chemical measurements; rows shuffled) and `y` the cultivar (0, 1, 2)",
                3, 13)
    n, d = rng.randint(150, 350), rng.randint(4, 8)
    k = 2 if binary else rng.choice([2, 3])
    return (f"from sklearn.datasets import make_classification\nfrom sklearn.preprocessing import StandardScaler\n"
            f"X, y = make_classification(n_samples={n}, n_features={d}, n_informative={min(d, 3)}, n_redundant=0, "
            f"n_classes={k}, flip_y=0.05, random_state={s})\nX = StandardScaler().fit_transform(X)",
            f"`X` ({n} × {d}, standardized) and `y` ({k} classes) come from `make_classification`", k, d)


def _docs(rng, n=(80, 200), V=(12, 25), classes=(2, 3)):
    """Word-count documents: X (counts), y (class), each class with its own word rates."""
    n_, V_, k, s = rng.randint(*n), rng.randint(*V), rng.randint(*classes), seeded(rng)
    setup = (f"g = np.random.default_rng({s})\ny = g.integers(0, {k}, {n_})\n"
             f"rates = g.gamma(0.6, 1.5, ({k}, {V_}))   # each class has its own word frequencies\n"
             f"X = g.poisson(rates[y])   # word counts: one row per document, one column per word")
    return setup, f"`X` holds word counts for {n_} documents over a {V_}-word vocabulary and `y` their topic (0-{k - 1})", V_


# ================================================================= lite

@template(M, "lite")
def svm_basics(rng):
    setup, desc, k, d = _scaled_clf(rng)
    c, n = rng.choice([0.1, 1.0, 10.0]), rng.randint(2, 4)
    items = [
        ("SVC", f"Fit `SVC(kernel='rbf', C={c}, gamma='scale')` on `X, y` and return its training accuracy.",
         f"{SVM}SVC\nresult = SVC(kernel='rbf', C={c}, gamma='scale').fit(X, y).score(X, y)"),
        ("support_vectors_", f"Fit `SVC(C={c})` on `X, y` and return the shape of `support_vectors_` (the training rows "
                             f"that define the boundary).", f"{SVM}SVC\nresult = SVC(C={c}).fit(X, y).support_vectors_.shape"),
        ("n_support_", f"Fit `SVC(C={c})` on `X, y` and return `n_support_` (support vectors per class).",
         f"{SVM}SVC\nresult = SVC(C={c}).fit(X, y).n_support_"),
        ("decision_function", f"Fit `SVC(C={c})` on `X, y` and return `decision_function` of the first {n} rows.",
         f"{SVM}SVC\nresult = SVC(C={c}).fit(X, y).decision_function(X[:{n}])"),
        ("predict_proba", f"SVMs give probabilities only with `probability=True` (an internal cross-validated "
                          f"calibration, hence the seed). Fit `SVC(C={c}, probability=True, random_state=0)` on `X, y` "
                          f"and return `predict_proba` of the first {n} rows.",
         f"{SVM}SVC\nresult = SVC(C={c}, probability=True, random_state=0).fit(X, y).predict_proba(X[:{n}])"),
        ("kernel", "Fit `SVC(kernel='linear')` on `X, y` and return the shape of its `coef_` (only linear kernels have "
                   "weights).", f"{SVM}SVC\nresult = SVC(kernel='linear').fit(X, y).coef_.shape"),
        ("degree", "Fit `SVC(kernel='poly', degree=3)` on `X, y` and return its training accuracy.",
         f"{SVM}SVC\nresult = SVC(kernel='poly', degree=3).fit(X, y).score(X, y)"),
        ("LinearSVC", f"`LinearSVC` scales to large data. Fit `LinearSVC(C={c})` on `X, y` and return its training "
                      f"accuracy.", f"{SVM}LinearSVC\nresult = LinearSVC(C={c}).fit(X, y).score(X, y)"),
        ("support_", f"Fit `SVC(C={c})` on `X, y` and return how many training rows are support vectors (`support_` "
                     f"holds their indices).", f"{SVM}SVC\nresult = len(SVC(C={c}).fit(X, y).support_)"),
    ]
    return catalog(rng, items, setup, f"{desc}.", tol=1e-6)


@template(M, "lite")
def neighbors_basics(rng):
    setup, desc, k, d = _scaled_clf(rng)
    nn, n, i = rng.choice([3, 5, 7]), rng.randint(2, 4), rng.randint(0, 99)
    items = [
        ("KNeighborsClassifier", f"Fit `KNeighborsClassifier(n_neighbors={nn})` on `X, y` and predict the first {n} rows.",
         f"{NN}KNeighborsClassifier\nresult = KNeighborsClassifier(n_neighbors={nn}).fit(X, y).predict(X[:{n}])"),
        ("predict_proba", f"Fit `KNeighborsClassifier(n_neighbors={nn})` on `X, y` and return `predict_proba` of the "
                          f"first {n} rows (the share of each class among the neighbours).",
         f"{NN}KNeighborsClassifier\nresult = KNeighborsClassifier(n_neighbors={nn}).fit(X, y).predict_proba(X[:{n}])"),
        ("weights", f"Closer neighbours can count more: fit `KNeighborsClassifier(n_neighbors={nn}, weights='distance')` on "
                    f"`X, y` and return its training accuracy (1.0: each row is its own nearest neighbour at distance 0).",
         f"{NN}KNeighborsClassifier\nresult = KNeighborsClassifier(n_neighbors={nn}, weights='distance').fit(X, y).score(X, y)"),
        ("kneighbors", f"Fit `KNeighborsClassifier(n_neighbors={nn})` on `X, y` and return `kneighbors(X[{i}:{i + 1}])`: "
                       f"the distances and indices of row {i}'s neighbours (the first is the row itself).",
         f"{NN}KNeighborsClassifier\nresult = KNeighborsClassifier(n_neighbors={nn}).fit(X, y).kneighbors(X[{i}:{i + 1}])"),
        ("NearestNeighbors", f"Similarity search: fit `NearestNeighbors(n_neighbors=3, metric='cosine')` on `X` and return "
                             f"the **indices** of the 3 most similar rows to each of the first {n} rows.",
         f"{NN}NearestNeighbors\nresult = NearestNeighbors(n_neighbors=3, metric='cosine').fit(X).kneighbors(X[:{n}])[1]"),
        ("NearestCentroid", "Fit `NearestCentroid()` on `X, y` and return the class centroids (`centroids_`).",
         f"{NN}NearestCentroid\nresult = NearestCentroid().fit(X, y).centroids_"),
        ("metric", f"Fit `KNeighborsClassifier(n_neighbors={nn}, metric='minkowski', p=1)` (Manhattan distance) on "
                   f"`X, y` and predict the first {n} rows.",
         f"{NN}KNeighborsClassifier\nresult = KNeighborsClassifier(n_neighbors={nn}, metric='minkowski', p=1).fit(X, y).predict(X[:{n}])"),
    ]
    return catalog(rng, items, setup, f"{desc}.", tol=1e-6)


@template(M, "lite")
def naive_bayes_basics(rng):
    setup, desc, V = _docs(rng)
    a, n = rng.choice([0.1, 0.5, 1.0, 2.0]), rng.randint(2, 4)
    items = [
        ("MultinomialNB", f"Fit `MultinomialNB(alpha={a})` on the counts and predict the first {n} documents.",
         f"{NB}MultinomialNB\nresult = MultinomialNB(alpha={a}).fit(X, y).predict(X[:{n}])"),
        ("feature_log_prob_", f"Fit `MultinomialNB(alpha={a})` on `X, y` and return the shape of `feature_log_prob_` "
                              f"(log P(word | class)).", f"{NB}MultinomialNB\nresult = MultinomialNB(alpha={a}).fit(X, y).feature_log_prob_.shape"),
        ("class_log_prior_", "Fit `MultinomialNB()` on `X, y` and return `class_log_prior_`.",
         f"{NB}MultinomialNB\nresult = MultinomialNB().fit(X, y).class_log_prior_"),
        ("ComplementNB", f"`ComplementNB` (good with uneven classes) has the same API. Fit `ComplementNB(alpha={a})` and "
                         f"predict the first {n} documents.",
         f"{NB}ComplementNB\nresult = ComplementNB(alpha={a}).fit(X, y).predict(X[:{n}])"),
        ("BernoulliNB", f"`BernoulliNB` only looks at whether a word occurs. Fit it on `X > 0` and `y` and return "
                        f"`predict_proba` of the first {n} documents.",
         f"{NB}BernoulliNB\nresult = BernoulliNB().fit(X > 0, y).predict_proba(X[:{n}] > 0)"),
        ("CategoricalNB", f"`CategoricalNB` treats each column as a category code. Fit it on `X, y` and predict the "
                          f"first {n} rows.", f"{NB}CategoricalNB\nresult = CategoricalNB().fit(X, y).predict(X[:{n}])"),
        ("class_count_", "Fit `MultinomialNB()` on `X, y` and return `class_count_` (documents per class).",
         f"{NB}MultinomialNB\nresult = MultinomialNB().fit(X, y).class_count_"),
        ("feature_count_", "Fit `MultinomialNB()` on `X, y` and return the total count of word 0 in each class "
                           "(`feature_count_[:, 0]`).", f"{NB}MultinomialNB\nresult = MultinomialNB().fit(X, y).feature_count_[:, 0]"),
    ]
    return catalog(rng, items, setup, f"{desc}.", tol=1e-6)


@template(M, "lite")
def continuous_models_basics(rng):
    d = synth_reg(rng, n=(80, 200), d=(2, 5))
    setup, desc, k, dd = _scaled_clf(rng)
    reg = d.setup + "\nfrom sklearn.preprocessing import StandardScaler\nX = StandardScaler().fit_transform(X)"
    rdesc = d.desc + "; `X` is standardized"
    n, nn = rng.randint(2, 4), rng.choice([3, 5])
    nu = rng.choice([0.02, 0.05, 0.1])
    items = [
        ("KNeighborsRegressor", f"{rdesc}. Fit `KNeighborsRegressor(n_neighbors={nn})` on `X, y` and predict the first "
                                f"{n} rows (the mean target of the neighbours).",
         f"{NN}KNeighborsRegressor\nresult = KNeighborsRegressor(n_neighbors={nn}).fit(X, y).predict(X[:{n}])",
         {"setup": reg}),
        ("SVR", f"{rdesc}. Fit `SVR(kernel='rbf', C=10, epsilon=0.1)` on `X, y` and predict the first {n} rows.",
         f"{SVM}SVR\nresult = SVR(kernel='rbf', C=10, epsilon=0.1).fit(X, y).predict(X[:{n}])", {"setup": reg}),
        ("SVR", f"{rdesc}. Rows inside the ε-tube don't become support vectors. Fit `SVR(kernel='rbf', C=10, "
                f"epsilon=e)` for e = 1 and e = 20 on `X, y` and return the numbers of support vectors as a tuple.",
         f"{SVM}SVR\nresult = tuple(len(SVR(kernel='rbf', C=10, epsilon=e).fit(X, y).support_) for e in (1, 20))",
         {"setup": reg}),
        ("OneClassSVM", f"{desc}. Novelty detection: fit `OneClassSVM(nu={nu})` on `X` (no labels) and return `predict` "
                        f"of the first 5 rows (+1 inlier, −1 outlier).",
         f"{SVM}OneClassSVM\nresult = OneClassSVM(nu={nu}).fit(X).predict(X[:5])", {"setup": setup}),
        ("OneClassSVM", f"{desc}. Fit `OneClassSVM(nu={nu})` on `X` and return the fraction of training rows it calls "
                        f"outliers (about `nu`).",
         f"{SVM}OneClassSVM\nresult = float((OneClassSVM(nu={nu}).fit(X).predict(X) == -1).mean())", {"setup": setup}),
    ]
    return catalog(rng, items, "", tol=1e-6)


# ================================================================= easy

@template(M, "easy")
def knn_by_hand(rng):
    setup, desc, k, d = _scaled_clf(rng)
    nn, i = rng.choice([3, 5, 7]), rng.randint(0, 99)
    task = rng.choice(["vote", "regress", "proba"])
    setup += f"\nq = X[{i}] + 0.05   # a new point near row {i}"
    dist = f"dist = np.sqrt(((X - q) ** 2).sum(axis=1))\nnb = np.argsort(dist)[:{nn}]\n"
    if task == "vote":
        return P(f"{desc}. `q` is a new point. Classify it **by hand** like `KNeighborsClassifier(n_neighbors={nn})`: "
                 f"Euclidean distances to every row, the {nn} nearest, majority vote (ties: the smallest label). Return "
                 f"`(by_hand, sklearn)` labels as ints.", setup,
                 f"{NN}KNeighborsClassifier\n{dist}mine = int(np.bincount(y[nb]).argmax())\n"
                 f"result = (mine, int(KNeighborsClassifier(n_neighbors={nn}).fit(X, y).predict(q[None])[0]))",
                 require=["KNeighborsClassifier", "argsort"])
    if task == "proba":
        return P(f"{desc}. `q` is a new point. Compute `KNeighborsClassifier(n_neighbors={nn}).predict_proba` for it "
                 f"**by hand**: the share of each class among its {nn} nearest rows (Euclidean). Return the array "
                 f"(one entry per class).", setup,
                 f"{dist}result = np.bincount(y[nb], minlength=len(np.unique(y))) / {nn}", require=["argsort"], tol=1e-9)
    return P(f"{desc}. `q` is a new point. Treat the label as a number: `KNeighborsRegressor(n_neighbors={nn})` would "
             f"predict the mean `y` of the {nn} nearest rows. Return `(by_hand, sklearn)`.", setup,
             f"{NN}KNeighborsRegressor\n{dist}"
             f"result = (float(y[nb].mean()), float(KNeighborsRegressor(n_neighbors={nn}).fit(X, y).predict(q[None])[0]))",
             require=["KNeighborsRegressor", "argsort"], tol=1e-9)


@template(M, "easy")
def centroid_by_hand(rng):
    setup, desc, k, d = _scaled_clf(rng)
    n = rng.randint(3, 6)
    if rng.random() < 0.5:
        return P(f"{desc}. `NearestCentroid` summarizes each class by its mean. Compute the class means by hand and "
                 f"return `np.abs(by_hand - NearestCentroid().fit(X, y).centroids_).max()` (≈ 0).", setup,
                 f"{NN}NearestCentroid\nmeans = np.array([X[y == c].mean(axis=0) for c in np.unique(y)])\n"
                 "result = float(np.abs(means - NearestCentroid().fit(X, y).centroids_).max())",
                 require=["NearestCentroid"], tol=1e-9)
    return P(f"{desc}. Predict the first {n} rows **by hand** like `NearestCentroid`: each row goes to the class whose "
             f"mean is closest (Euclidean). Return the labels as an array.", setup,
             f"cls = np.unique(y)\nmeans = np.array([X[y == c].mean(axis=0) for c in cls])\n"
             f"result = cls[((X[:{n}, None, :] - means[None]) ** 2).sum(axis=2).argmin(axis=1)]")


@template(M, "easy")
def gaussian_nb_by_hand(rng):
    setup, desc, k, d = _scaled_clf(rng)
    task = rng.choice(["theta", "var", "prior"])
    if task == "theta":
        return P(f"{desc}. `GaussianNB` models each feature in each class as a normal distribution. Return the largest "
                 f"absolute difference between its `theta_` and the per-class means computed by hand.", setup,
                 f"{NB}GaussianNB\nm = GaussianNB().fit(X, y)\n"
                 "means = np.array([X[y == c].mean(axis=0) for c in m.classes_])\nresult = float(np.abs(m.theta_ - means).max())",
                 require=["theta_"], tol=1e-9)
    if task == "var":
        return P(f"{desc}. `GaussianNB`'s `var_` is each class's (population) variance plus a tiny `epsilon_` = "
                 f"`var_smoothing` × the largest variance of any feature. Rebuild `var_` by hand (`epsilon_` = 1e-9 × "
                 f"`X.var(axis=0).max()`) and return it.", setup,
                 "eps = 1e-9 * X.var(axis=0).max()\nresult = np.array([X[y == c].var(axis=0) for c in np.unique(y)]) + eps",
                 tol=1e-9)
    return P(f"{desc}. Fit `GaussianNB()` on `X, y` and return `(class_prior_, by_hand)`: the priors are just the class "
             f"frequencies.", setup,
             f"{NB}GaussianNB\nresult = (GaussianNB().fit(X, y).class_prior_, np.bincount(y) / len(y))",
             require=["class_prior_"], tol=1e-9)


@template(M, "easy")
def multinomial_nb_by_hand(rng):
    setup, desc, V = _docs(rng)
    a = rng.choice([0.5, 1.0, 2.0])
    task = rng.choice(["flp", "prior"])
    if task == "flp":
        return P(f"{desc}. `MultinomialNB(alpha={a})` estimates P(word | class) = (count of the word in the class + α) / "
                 f"(total words in the class + α · vocabulary size). Compute `feature_log_prob_` by hand (a classes × "
                 f"words array of logs).", setup,
                 "N = np.array([X[y == c].sum(axis=0) for c in np.unique(y)])\n"
                 f"result = np.log((N + {a}) / (N.sum(axis=1, keepdims=True) + {a} * X.shape[1]))", tol=1e-9)
    return P(f"{desc}. Return `(class_log_prior_, by_hand)` for `MultinomialNB()`: the log of each class's share of the "
             f"documents.", setup, f"{NB}MultinomialNB\nresult = (MultinomialNB().fit(X, y).class_log_prior_, "
                                   "np.log(np.bincount(y) / len(y)))", require=["class_log_prior_"], tol=1e-9)


@template(M, "easy")
def bernoulli_binarize(rng):
    setup, desc, V = _docs(rng)
    t = rng.choice([0, 1, 2])
    if rng.random() < 0.5:
        return P(f"{desc}. `BernoulliNB(binarize={t})` turns counts into 0/1 with `X > {t}` before fitting. Show it: "
                 f"return whether its predictions on `X` equal those of `BernoulliNB(binarize=None)` fitted on and "
                 f"applied to `(X > {t})`.", setup,
                 f"{NB}BernoulliNB\na = BernoulliNB(binarize={t}).fit(X, y).predict(X)\n"
                 f"b = BernoulliNB(binarize=None).fit(X > {t}, y).predict(X > {t})\nresult = bool((a == b).all())",
                 require=["BernoulliNB", "binarize"])
    return P(f"{desc}. `BernoulliNB()` estimates P(word present | class) = (documents of the class containing the word "
             f"+ 1) / (documents of the class + 2). Compute it by hand from `X > 0` and return the log of that matrix "
             f"(it equals `feature_log_prob_`).", setup,
             "B = X > 0\ncls = np.unique(y)\n"
             "result = np.log(np.array([(B[y == c].sum(axis=0) + 1) / ((y == c).sum() + 2) for c in cls]))", tol=1e-9)


@template(M, "easy")
def scale_matters(rng):
    s = seeded(rng)
    setup = ("from sklearn.datasets import load_wine\nX, y = load_wine(return_X_y=True)\n"
             f"order = np.random.default_rng({s}).permutation(len(y))\nX, y = X[order], y[order]")
    m = rng.randint(100, 130)
    model, imp = rng.choice([("KNeighborsClassifier(n_neighbors=5)", NN + "KNeighborsClassifier"),
                             ("SVC()", SVM + "SVC")])
    return P(f"`X` holds 178 wines (13 raw measurements on very different scales: proline is in the hundreds, hue "
             f"around 1; rows shuffled) and `y` the cultivar. Distance-based models need scaling. Train `{model}` on the "
             f"first {m} rows and return its accuracy on the rest, without and with `StandardScaler` in a pipeline "
             f"(`make_pipeline(StandardScaler(), {model})`), as a tuple.", setup,
             f"{imp}\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\n"
             f"a = {model}.fit(X[:{m}], y[:{m}]).score(X[{m}:], y[{m}:])\n"
             f"b = make_pipeline(StandardScaler(), {model}).fit(X[:{m}], y[:{m}]).score(X[{m}:], y[{m}:])\nresult = (a, b)",
             require=["StandardScaler"])


@template(M, "easy")
def linear_svm_flavours(rng):
    setup, desc, k, d = _scaled_clf(rng, binary=True)
    c = rng.choice([0.1, 1.0])
    task = rng.choice(["agree", "coef", "count"])
    if task == "agree":
        return P(f"{desc}. `LinearSVC` and `SVC(kernel='linear')` solve slightly different problems (LinearSVC also "
                 f"penalizes the intercept and uses a squared hinge). Fit both with `C={c}` on `X, y` and return the "
                 f"fraction of rows where they predict the same label.", setup,
                 f"{SVM}LinearSVC, SVC\na = LinearSVC(C={c}).fit(X, y).predict(X)\nb = SVC(kernel='linear', C={c}).fit(X, y).predict(X)\n"
                 "result = float((a == b).mean())", require=["LinearSVC", "SVC"])
    if task == "coef":
        return P(f"{desc}. For a linear kernel the weights are a combination of the support vectors: `coef_` = "
                 f"`dual_coef_ @ support_vectors_`. Fit `SVC(kernel='linear', C={c})` on `X, y` and return the largest "
                 f"absolute difference between the two (≈ 0).", setup,
                 f"{SVM}SVC\nm = SVC(kernel='linear', C={c}).fit(X, y)\n"
                 "result = float(np.abs(m.dual_coef_ @ m.support_vectors_ - m.coef_).max())",
                 require=["dual_coef_", "support_vectors_"], tol=1e-9)
    cs = [0.01, 0.1, 1.0, 10.0]
    return P(f"{desc}. A smaller `C` means a softer margin with more support vectors. Return the number of support "
             f"vectors of `SVC(kernel='linear', C=C)` fitted on `X, y` for `C` in `{cs}`.", setup,
             f"{SVM}SVC\nresult = [int(SVC(kernel='linear', C=C).fit(X, y).support_vectors_.shape[0]) for C in {cs}]",
             require=["support_vectors_"])


@template(M, "easy")
def outlier_fraction(rng):
    t = table(rng, kind="reg", numeric_only=True, only=("servers", "bikes", "houses"), n=(150, 300))
    setup = t.setup + "\nfrom sklearn.preprocessing import StandardScaler\nX = StandardScaler().fit_transform(df)"
    nus = sorted(rng.sample([0.01, 0.05, 0.1, 0.2, 0.3], 3))
    if rng.random() < 0.5:
        return P(f"{t.desc}; `X` is `df` standardized. `nu` bounds the fraction of training outliers. For `nu` in "
                 f"`{nus}`, fit `OneClassSVM(nu=nu)` on `X` and return the fraction predicted −1.", setup,
                 f"{SVM}OneClassSVM\nresult = [float((OneClassSVM(nu=v).fit(X).predict(X) == -1).mean()) for v in {nus}]",
                 require=["OneClassSVM"])
    return P(f"{t.desc}; `X` is `df` standardized. Fit `OneClassSVM(nu={nus[1]})` on `X` and return the row labels "
             f"(`df.index`) of the 5 most anomalous rows (smallest `decision_function`), most anomalous first.", setup,
             f"{SVM}OneClassSVM\nscore = OneClassSVM(nu={nus[1]}).fit(X).decision_function(X)\n"
             "result = list(df.index[np.argsort(score)[:5]])", require=["OneClassSVM", "decision_function"])


@template(M, "easy")
def cosine_retrieval(rng):
    setup, desc, V = _docs(rng, n=(60, 150))
    i, k = rng.randint(0, 49), rng.choice([3, 5])
    task = rng.choice(["idx", "dist", "by_hand"])
    if task == "idx":
        return P(f"{desc}. Find the {k} documents most similar to document {i} (cosine distance), **excluding** itself: "
                 f"fit `NearestNeighbors(n_neighbors={k + 1}, metric='cosine')` on `X`, query `X[{i}:{i + 1}]` and drop "
                 f"the first hit. Return the indices.", setup,
                 f"{NN}NearestNeighbors\nnn = NearestNeighbors(n_neighbors={k + 1}, metric='cosine').fit(X)\n"
                 f"dist, idx = nn.kneighbors(X[{i}:{i + 1}])\nresult = idx[0][1:]", require=["NearestNeighbors", "kneighbors"])
    if task == "dist":
        return P(f"{desc}. Fit `NearestNeighbors(n_neighbors={k}, metric='cosine')` on `X` and return the cosine "
                 f"distances from document {i} to its {k} nearest documents (the first is 0: itself).", setup,
                 f"{NN}NearestNeighbors\nresult = NearestNeighbors(n_neighbors={k}, metric='cosine').fit(X).kneighbors(X[{i}:{i + 1}])[0][0]",
                 require=["kneighbors"], tol=1e-9)
    return P(f"{desc}. Cosine distance is 1 − cos(angle). Compute it by hand from document {i} to every document "
             f"(normalize the rows to unit length, take dot products) and return the indices of the {k} closest "
             f"**other** documents, closest first.", setup,
             "U = X / np.linalg.norm(X, axis=1, keepdims=True).clip(1e-12)\n"
             f"d = 1 - U @ U[{i}]\nd[{i}] = np.inf\nresult = np.argsort(d, kind='stable')[:{k}]", tol=1e-9)


@template(M, "easy")
def model_family_tour(rng):
    setup, desc, k, d = _scaled_clf(rng)
    n = rng.randint(2, 4)
    setup += (f"\n{SVM}SVC\n{NN}KNeighborsClassifier\n{NB}GaussianNB\n"
              "svc = SVC().fit(X, y)\nknn = KNeighborsClassifier(n_neighbors=5).fit(X, y)\ngnb = GaussianNB().fit(X, y)")
    items = [
        ("score", "Return the training accuracies of `svc`, `knn` and `gnb` as a dict with those keys.",
         "result = {'svc': svc.score(X, y), 'knn': knn.score(X, y), 'gnb': gnb.score(X, y)}"),
        ("predict", "On how many rows do all three models agree?",
         "a, b, c = svc.predict(X), knn.predict(X), gnb.predict(X)\nresult = int(((a == b) & (b == c)).sum())"),
        ("predict_proba", f"Return `gnb.predict_proba` of the first {n} rows.", f"result = gnb.predict_proba(X[:{n}])"),
        ("decision_function", f"Return the shape of `svc.decision_function(X[:{n}])`.",
         f"result = svc.decision_function(X[:{n}]).shape"),
        ("kneighbors", "Return the mean distance from each of the first 3 rows to its 5 neighbours (`knn.kneighbors`), "
                       "as an array.", "result = knn.kneighbors(X[:3])[0].mean(axis=1)"),
        ("support_vectors_", "What fraction of the training rows does `svc` keep as support vectors?",
         "result = len(svc.support_vectors_) / len(X)"),
        ("theta_", "Return the shape of `gnb.theta_` (one mean per class and feature).", "result = gnb.theta_.shape"),
        ("classes_", "Return `(svc.classes_, knn.classes_, gnb.classes_)` as lists.",
         "result = (svc.classes_.tolist(), knn.classes_.tolist(), gnb.classes_.tolist())"),
        ("predict_proba", "Return `knn.predict_proba(X[:3])` multiplied by 5: the neighbour **counts** per class.",
         "result = knn.predict_proba(X[:3]) * 5"),
        ("n_support_", "Return the number of support vectors per class (`svc.n_support_`) as a list.",
         "result = svc.n_support_.tolist()"),
    ]
    return catalog(rng, items, setup, f"{desc}. `svc = SVC()`, `knn = KNeighborsClassifier(n_neighbors=5)` and `gnb = "
                                      f"GaussianNB()` are fitted on `X, y`.", tol=1e-6)


# ================================================================= medium

@template(M, "medium")
def table_knn_tasks(rng):
    t = table(rng, kind="clf", numeric_only=True, n=(150, 300))
    m = rng.randint(t.n // 2, 3 * t.n // 4)
    s = seeded(rng)
    setup = (t.setup + f"\norder = np.random.default_rng({s}).permutation(len(df))\ndf, y = df.iloc[order], y[order]\n"
             f"X_train, X_test, y_train, y_test = df.iloc[:{m}], df.iloc[{m}:], y[:{m}], y[{m}:]")
    pipe = f"from sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\n{NN}KNeighborsClassifier\n"
    ks = [1, 3, 5, 9, 15]
    items = [
        ("n_neighbors", f"Return the test accuracy of `make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))` "
                        f"for k in `{ks}` as a dict {{k: accuracy}}.",
         f"{pipe}result = {{k: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k)).fit(X_train, y_train)"
         f".score(X_test, y_test) for k in {ks}}}"),
        ("n_neighbors", f"Return `(train, test)` accuracy for k = 1 and for k = 15 (scaled pipeline) as a dict "
                        f"{{k: (train, test)}}: k = 1 memorizes the training set.",
         f"{pipe}result = {{}}\nfor k in (1, 15):\n"
         "    p = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k)).fit(X_train, y_train)\n"
         "    result[k] = (p.score(X_train, y_train), p.score(X_test, y_test))"),
        ("weights", "Return the test accuracy of the scaled 7-NN pipeline with `weights='uniform'` and with "
                    "`weights='distance'`, as a tuple.",
         f"{pipe}result = tuple(make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7, weights=w))"
         ".fit(X_train, y_train).score(X_test, y_test) for w in ('uniform', 'distance'))"),
        ("p", "Return the test accuracy of the scaled 5-NN pipeline with Manhattan (`p=1`) and Euclidean (`p=2`) "
              "distance, as a tuple.",
         f"{pipe}result = tuple(make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, p=q))"
         ".fit(X_train, y_train).score(X_test, y_test) for q in (1, 2))"),
        ("kneighbors", "Explain a prediction: fit the scaled 5-NN pipeline on the training rows and return the **index "
                       "labels** (`X_train.index`) of the 5 training rows closest to the first test row.",
         f"{pipe}p = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)).fit(X_train, y_train)\n"
         "dist, idx = p[-1].kneighbors(p[0].transform(X_test.iloc[:1]))\nresult = list(X_train.index[idx[0]])"),
        ("RadiusNeighborsClassifier", "Return how many training rows lie within distance 1.0 of each of the first 5 test "
                                      "rows, after standardizing with a scaler fitted on the training rows "
                                      "(`NearestNeighbors(radius=1.0).radius_neighbors`).",
         f"{NN}NearestNeighbors\nfrom sklearn.preprocessing import StandardScaler\nsc = StandardScaler().fit(X_train)\n"
         "nn = NearestNeighbors(radius=1.0).fit(sc.transform(X_train))\n"
         "dist, idx = nn.radius_neighbors(sc.transform(X_test.iloc[:5]))\nresult = [len(i) for i in idx]",
         {"require": ["radius_neighbors"]}),
        ("NearestCentroid", "Return the test accuracy of `make_pipeline(StandardScaler(), NearestCentroid())` and of the "
                            "scaled 5-NN pipeline, as a tuple.",
         f"{pipe}from sklearn.neighbors import NearestCentroid\n"
         "a = make_pipeline(StandardScaler(), NearestCentroid()).fit(X_train, y_train).score(X_test, y_test)\n"
         "b = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)).fit(X_train, y_train).score(X_test, y_test)\n"
         "result = (a, b)"),
    ]
    return catalog(rng, items, setup, f"{t.desc}. Rows were shuffled and split into `X_train, y_train` (first {m}) and "
                                      f"`X_test, y_test`.", tol=1e-6)


@template(M, "medium")
def svm_tuning(rng):
    setup, desc, k, d = _scaled_clf(rng)
    s = seeded(rng)
    setup += f"\norder = np.random.default_rng({s}).permutation(len(y))\nX, y = X[order], y[order]\nh = int(0.7 * len(y))"
    items = [
        ("C", "Return the number of support vectors of `SVC(C=C)` fitted on all of `X, y`, for `C` in 0.01, 0.1, 1, "
              "10, 100.", f"{SVM}SVC\nresult = [len(SVC(C=C).fit(X, y).support_) for C in (0.01, 0.1, 1, 10, 100)]"),
        ("gamma", "Return `(train, test)` accuracy of `SVC(gamma=g)` trained on the first `h` rows and tested on the "
                  "rest, for `g` in 0.001, 0.1, 10, as a dict {g: (train, test)}: a huge gamma memorizes.",
         f"{SVM}SVC\nresult = {{g: (SVC(gamma=g).fit(X[:h], y[:h]).score(X[:h], y[:h]), "
         "SVC(gamma=g).fit(X[:h], y[:h]).score(X[h:], y[h:])) for g in (0.001, 0.1, 10)}"),
        ("cross_val_score", "Compare kernels with 5-fold cross-validation: return a dict {kernel: mean "
                            "`cross_val_score`} of `SVC(kernel=kernel)` for 'linear', 'rbf' and 'poly'.",
         f"{SVM}SVC\nfrom sklearn.model_selection import cross_val_score\n"
         "result = {kn: cross_val_score(SVC(kernel=kn), X, y, cv=5).mean() for kn in ('linear', 'rbf', 'poly')}"),
        ("GridSearchCV", "Tune with `GridSearchCV(SVC(), {'C': [0.1, 1, 10], 'gamma': ['scale', 0.01]}, cv=3)` on the "
                         "first `h` rows and return `(best_params_, test accuracy of best_estimator_)`.",
         f"{SVM}SVC\nfrom sklearn.model_selection import GridSearchCV\n"
         "gs = GridSearchCV(SVC(), {'C': [0.1, 1, 10], 'gamma': ['scale', 0.01]}, cv=3).fit(X[:h], y[:h])\n"
         "result = (gs.best_params_, gs.best_estimator_.score(X[h:], y[h:]))"),
        ("probability", "`decision_function` and calibrated probabilities usually rank rows alike. Fit "
                        "`SVC(probability=True, random_state=0)` on the first `h` rows and return the fraction of test "
                        "rows where `argmax(predict_proba)` gives the same class as `predict`.",
         f"{SVM}SVC\nm = SVC(probability=True, random_state=0).fit(X[:h], y[:h])\n"
         "result = float((m.classes_[m.predict_proba(X[h:]).argmax(axis=1)] == m.predict(X[h:])).mean())"),
        ("class_weight", "Return the test accuracy of `SVC()` and `SVC(class_weight='balanced')` trained on the first "
                         "`h` rows, as a tuple.",
         f"{SVM}SVC\nresult = (SVC().fit(X[:h], y[:h]).score(X[h:], y[h:]), "
         "SVC(class_weight='balanced').fit(X[:h], y[:h]).score(X[h:], y[h:]))"),
        ("LinearSVC", "Return the test accuracy of `LinearSVC(C=C)` trained on the first `h` rows for `C` in 0.001, "
                      "0.01, 0.1, 1.",
         f"{SVM}LinearSVC\nresult = [LinearSVC(C=C).fit(X[:h], y[:h]).score(X[h:], y[h:]) for C in (0.001, 0.01, 0.1, 1)]"),
        ("decision_function_shape", "Multiclass SVC trains one model per pair of classes. Return the shapes of "
                                    "`decision_function(X[:4])` for `SVC(decision_function_shape='ovo')` and for the "
                                    "default `'ovr'`, both fitted on `X, y`, as a tuple.",
         f"{SVM}SVC\nresult = (SVC(decision_function_shape='ovo').fit(X, y).decision_function(X[:4]).shape, "
         "SVC().fit(X, y).decision_function(X[:4]).shape)"),
        ("CalibratedClassifierCV", "`LinearSVC` has no `predict_proba`; wrap it: fit `CalibratedClassifierCV(LinearSVC(), "
                                   "cv=3)` (from `sklearn.calibration`) on the first `h` rows and return `predict_proba` of "
                                   "the first 3 test rows.",
         f"{SVM}LinearSVC\nfrom sklearn.calibration import CalibratedClassifierCV\n"
         "result = CalibratedClassifierCV(LinearSVC(), cv=3).fit(X[:h], y[:h]).predict_proba(X[h:h + 3])"),
    ]
    return catalog(rng, items, setup, f"{desc}. Rows were reshuffled; `h` = 70% of the rows (training part).", tol=1e-6)


@template(M, "medium")
def text_nb_tasks(rng):
    setup, desc, V = _docs(rng, n=(150, 300))
    setup += "\nh = int(0.7 * len(y))"
    items = [
        ("MultinomialNB", "Classify word counts: return the test accuracy (train on the first `h` documents) of "
                          "`MultinomialNB()`, `ComplementNB()` and `BernoulliNB()` as a dict with keys 'multinomial', "
                          "'complement', 'bernoulli'.",
         f"{NB}MultinomialNB, ComplementNB, BernoulliNB\n"
         "models = {'multinomial': MultinomialNB(), 'complement': ComplementNB(), 'bernoulli': BernoulliNB()}\n"
         "result = {k: m.fit(X[:h], y[:h]).score(X[h:], y[h:]) for k, m in models.items()}"),
        ("alpha", "Smoothing matters for rare words. Return the test accuracy of `MultinomialNB(alpha=a)` (trained on "
                  "the first `h` documents) for `a` in 0.001, 0.1, 1, 10.",
         f"{NB}MultinomialNB\nresult = [MultinomialNB(alpha=a).fit(X[:h], y[:h]).score(X[h:], y[h:]) for a in (0.001, 0.1, 1, 10)]"),
        ("feature_log_prob_", "Which words are most typical of each topic? Fit `MultinomialNB()` on all documents and "
                              "return, for each class, the index of the word with the highest `feature_log_prob_`.",
         f"{NB}MultinomialNB\nresult = MultinomialNB().fit(X, y).feature_log_prob_.argmax(axis=1)"),
        ("feature_log_prob_", "Which word best separates topic 0 from topic 1? Fit `MultinomialNB()` on all documents and "
                              "return the index of the word with the largest `feature_log_prob_[0] − feature_log_prob_[1]`.",
         f"{NB}MultinomialNB\nf = MultinomialNB().fit(X, y).feature_log_prob_\nresult = int(np.argmax(f[0] - f[1]))"),
        ("predict_proba", "A new document contains each word exactly once (`np.ones((1, X.shape[1]))`). Return the "
                          "class probabilities `MultinomialNB()` (fitted on all documents) gives it.",
         f"{NB}MultinomialNB\nresult = MultinomialNB().fit(X, y).predict_proba(np.ones((1, X.shape[1])))"),
        ("partial_fit", "Naive Bayes can learn incrementally. Feed `MultinomialNB()` the documents in chunks of 30 with "
                        "`partial_fit(..., classes=np.unique(y))` and return whether its `feature_count_` equals that of "
                        "a `MultinomialNB()` fitted on all documents at once.",
         f"{NB}MultinomialNB\nm = MultinomialNB()\nfor i in range(0, len(y), 30):\n"
         "    m.partial_fit(X[i:i + 30], y[i:i + 30], classes=np.unique(y))\n"
         "result = bool(np.allclose(m.feature_count_, MultinomialNB().fit(X, y).feature_count_))"),
        ("TfidfTransformer", "Return the test accuracy of `MultinomialNB()` trained on the first `h` documents after "
                             "`TfidfTransformer()` (from `sklearn.feature_extraction.text`) re-weights the counts (inside "
                             "`make_pipeline`).",
         f"{NB}MultinomialNB\nfrom sklearn.feature_extraction.text import TfidfTransformer\nfrom sklearn.pipeline import make_pipeline\n"
         "result = make_pipeline(TfidfTransformer(), MultinomialNB()).fit(X[:h], y[:h]).score(X[h:], y[h:])"),
    ]
    return catalog(rng, items, setup, f"{desc}. `h` = 70% of the documents (the training part).", tol=1e-6)


@template(M, "medium")
def gaussian_nb_tasks(rng):
    t = table(rng, kind="clf", numeric_only=True, n=(150, 300))
    new = {c: 1 for c in t.num}
    setup = t.setup + f"\nnew_row = pd.DataFrame([{new}])"
    items = [
        ("GaussianNB", "Fit `GaussianNB()` on `df, y` and return P(class 1) for every row of `new_row` (a one-row "
                       "DataFrame).", f"{NB}GaussianNB\nresult = GaussianNB().fit(df, y).predict_proba(new_row)[:, 1]"),
        ("priors", "Fixed priors override the class frequencies: return the training accuracy of `GaussianNB()` and "
                   "`GaussianNB(priors=[0.5, 0.5])`, as a tuple.",
         f"{NB}GaussianNB\nresult = (GaussianNB().fit(df, y).score(df, y), GaussianNB(priors=[0.5, 0.5]).fit(df, y).score(df, y))"),
        ("theta_", "Return a dict {column: (mean in class 0, mean in class 1)} from `GaussianNB().fit(df, y).theta_`.",
         f"{NB}GaussianNB\nth = GaussianNB().fit(df, y).theta_\nresult = {{c: (th[0, j], th[1, j]) for j, c in enumerate(df.columns)}}"),
        ("var_smoothing", "Return the training accuracy of `GaussianNB(var_smoothing=v)` for `v` in 1e-9, 1e-3, 1, 100: "
                          "a big smoothing flattens every distribution.",
         f"{NB}GaussianNB\nresult = [GaussianNB(var_smoothing=v).fit(df, y).score(df, y) for v in (1e-9, 1e-3, 1, 100)]"),
        ("partial_fit", "Stream the rows in chunks of 50 into `GaussianNB().partial_fit(..., classes=[0, 1])` and return "
                        "the largest absolute difference between its `theta_` and that of a one-shot fit (≈ 0).",
         f"{NB}GaussianNB\nm = GaussianNB()\nfor i in range(0, len(df), 50):\n"
         "    m.partial_fit(df.iloc[i:i + 50], y[i:i + 50], classes=[0, 1])\n"
         "result = float(np.abs(m.theta_ - GaussianNB().fit(df, y).theta_).max())", {"tol": 1e-6}),
        ("predict_log_proba", "Return the training log loss of `GaussianNB()` computed from `predict_log_proba`: minus "
                              "the mean log-probability of the true class.",
         f"{NB}GaussianNB\nlp = GaussianNB().fit(df, y).predict_log_proba(df)\nresult = float(-lp[np.arange(len(y)), y].mean())"),
    ]
    return catalog(rng, items, setup, f"{t.desc}. `new_row` is a one-row DataFrame with every feature equal to 1.",
                   tol=1e-6)


@template(M, "medium")
def categorical_nb_tasks(rng):
    t = table(rng, kind="clf", n=(150, 300))
    col = rng.choice(t.cat)
    setup = (t.setup + f"\nfrom sklearn.preprocessing import OrdinalEncoder\ncat_cols = {t.cat}\n"
             "enc = OrdinalEncoder().fit(df[cat_cols])\nC = enc.transform(df[cat_cols]).astype(int)   # category codes")
    j = t.cat.index(col)
    items = [
        ("CategoricalNB", "Fit `CategoricalNB()` on the codes `C` and `y` and return its training accuracy.",
         f"{NB}CategoricalNB\nresult = CategoricalNB().fit(C, y).score(C, y)"),
        ("category_count_", f"Fit `CategoricalNB()` on `C, y` and return `category_count_[{j}]` (rows of each class per "
                            f"level of `{col}`).", f"{NB}CategoricalNB\nresult = CategoricalNB().fit(C, y).category_count_[{j}]"),
        ("feature_log_prob_", f"Fit `CategoricalNB(alpha=1)` on `C, y` and rebuild `feature_log_prob_[{j}]` by hand: "
                              f"log((count of the level in the class + 1) / (rows of the class + number of levels)).",
         f"counts = np.array([np.bincount(C[y == c, {j}], minlength=len(enc.categories_[{j}])) for c in np.unique(y)])\n"
         "result = np.log((counts + 1) / (counts.sum(axis=1, keepdims=True) + counts.shape[1]))",
         {"require": ["bincount"]}),
        ("min_categories", "New data may contain a level code the training rows never had. Fit `CategoricalNB("
                           "min_categories=6)` on `C, y` (it reserves 6 codes per column) and predict a row of all 5s.",
         f"{NB}CategoricalNB\nresult = CategoricalNB(min_categories=6).fit(C, y).predict(np.full((1, C.shape[1]), 5))"),
        ("predict_proba", "Fit `CategoricalNB()` on `C, y` and return P(class 1) for the first 3 rows.",
         f"{NB}CategoricalNB\nresult = CategoricalNB().fit(C, y).predict_proba(C[:3])[:, 1]"),
    ]
    return catalog(rng, items, setup, f"{t.desc}. The text columns `cat_cols` were turned into integer codes `C` with "
                                      f"`OrdinalEncoder` (`enc`).", tol=1e-6)


@template(M, "medium")
def anomaly_tasks(rng):
    t = table(rng, kind="reg", numeric_only=True, only=("servers", "bikes", "phones"), n=(150, 300))
    s = seeded(rng)
    setup = (t.setup + f"\nweird = np.random.default_rng({s}).choice(len(df), 5, replace=False)\n"
             "df.iloc[weird, 0] = df.iloc[:, 0].max() * 3   # 5 broken readings\n"
             "from sklearn.preprocessing import StandardScaler\nX = StandardScaler().fit_transform(df)")
    items = [
        ("OneClassSVM", "Fit `OneClassSVM(nu=0.05)` on `X` and return how many of the 5 broken rows (`weird`) it flags.",
         f"{SVM}OneClassSVM\nresult = int((OneClassSVM(nu=0.05).fit(X).predict(X[weird]) == -1).sum())"),
        ("gamma", "Return the number of training rows flagged by `OneClassSVM(nu=0.05, gamma=g)` for `g` in 0.01, 0.1, 1.",
         f"{SVM}OneClassSVM\nresult = [int((OneClassSVM(nu=0.05, gamma=g).fit(X).predict(X) == -1).sum()) for g in (0.01, 0.1, 1)]"),
        ("decision_function", "Fit `OneClassSVM(nu=0.05)` on `X` and return the 5 row positions with the lowest "
                              "`decision_function`, sorted ascending by position.",
         f"{SVM}OneClassSVM\nresult = np.sort(np.argsort(OneClassSVM(nu=0.05).fit(X).decision_function(X))[:5])"),
        ("kneighbors", "A k-NN anomaly score: the mean distance to the 5 nearest **other** rows. Return the 5 row "
                       "positions with the largest score, sorted ascending (`NearestNeighbors(n_neighbors=6)`, drop the "
                       "first column: the row itself).",
         f"{NN}NearestNeighbors\ndist, idx = NearestNeighbors(n_neighbors=6).fit(X).kneighbors(X)\n"
         "result = np.sort(np.argsort(-dist[:, 1:].mean(axis=1))[:5])"),
        ("score_samples", "Return the fraction of rows where `OneClassSVM(nu=0.05)`'s `score_samples` minus `offset_` "
                          "equals `decision_function` (all of them: that's its definition).",
         f"{SVM}OneClassSVM\nm = OneClassSVM(nu=0.05).fit(X)\n"
         "result = float(np.isclose(m.score_samples(X) - m.offset_, m.decision_function(X)).mean())"),
    ]
    return catalog(rng, items, setup, f"{t.desc}. Five rows (`weird`) got an impossible value in the first column; `X` "
                                      f"is `df` standardized.", tol=1e-6)


@template(M, "medium")
def knn_regression_tasks(rng):
    t = table(rng, kind="reg", numeric_only=True, n=(150, 300))
    m = rng.randint(t.n // 2, 3 * t.n // 4)
    s = seeded(rng)
    setup = (t.setup + f"\norder = np.random.default_rng({s}).permutation(len(df))\ndf, y = df.iloc[order], y[order]\n"
             f"X_train, X_test, y_train, y_test = df.iloc[:{m}], df.iloc[{m}:], y[:{m}], y[{m}:]")
    pipe = f"from sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\n{NN}KNeighborsRegressor\n"
    task = rng.choice(["k", "weights", "svr"])
    if task == "k":
        return P(f"{t.desc}. Rows were shuffled and split into training (first {m}) and test parts. Return the test R² "
                 f"of `make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=k))` for k in 1, 5, 20 as a list.",
                 setup, f"{pipe}result = [make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=k))"
                        ".fit(X_train, y_train).score(X_test, y_test) for k in (1, 5, 20)]",
                 require=["KNeighborsRegressor"], tol=1e-6)
    if task == "weights":
        return P(f"{t.desc}. Rows were shuffled and split into training (first {m}) and test parts. Return the test R² of "
                 f"the scaled `KNeighborsRegressor(n_neighbors=10)` pipeline with `weights='uniform'` and "
                 f"`weights='distance'`, as a tuple.", setup,
                 f"{pipe}result = tuple(make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=10, weights=w))"
                 ".fit(X_train, y_train).score(X_test, y_test) for w in ('uniform', 'distance'))",
                 require=["KNeighborsRegressor", "weights"], tol=1e-6)
    return P(f"{t.desc}. Rows were shuffled and split into training (first {m}) and test parts. SVR's `epsilon` is in "
             f"target units, so scale the target too: wrap `make_pipeline(StandardScaler(), SVR(C=10))` in "
             f"`TransformedTargetRegressor(regressor=..., transformer=StandardScaler())` and return `(test R², number of "
             f"support vectors)`.", setup,
             f"{SVM}SVR\nfrom sklearn.compose import TransformedTargetRegressor\nfrom sklearn.pipeline import make_pipeline\n"
             "from sklearn.preprocessing import StandardScaler\n"
             "m = TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(), SVR(C=10)), transformer=StandardScaler())\n"
             "m.fit(X_train, y_train)\nresult = (m.score(X_test, y_test), len(m.regressor_[-1].support_))",
             require=["SVR"], tol=1e-6)


# ================================================================= hard

@template(M, "hard")
def kernel_decision_by_hand(rng):
    setup, desc, k, d = _scaled_clf(rng, binary=True)
    n = rng.randint(3, 6)
    kern = rng.choice(["rbf", "poly", "linear"])
    sol = f"{SVM}SVC\n"
    if kern == "rbf":
        sol += ("m = SVC(kernel='rbf').fit(X, y)\ngamma = 1 / (X.shape[1] * X.var())\n"
                f"K = np.exp(-gamma * ((X[:{n}, None, :] - m.support_vectors_[None]) ** 2).sum(axis=2))\n")
        how = ("the RBF kernel K(s, x) = exp(−γ‖s − x‖²) with `gamma='scale'`, i.e. γ = 1 / (n_features · X.var())")
        ctor = "SVC(kernel='rbf')"
    elif kern == "poly":
        sol += (f"m = SVC(kernel='poly', degree=3, coef0=1).fit(X, y)\ngamma = 1 / (X.shape[1] * X.var())\n"
                f"K = (gamma * X[:{n}] @ m.support_vectors_.T + 1) ** 3\n")
        how = "the polynomial kernel K(s, x) = (γ s·x + 1)³ with γ = 1 / (n_features · X.var())"
        ctor = "SVC(kernel='poly', degree=3, coef0=1)"
    else:
        sol += f"m = SVC(kernel='linear').fit(X, y)\nK = X[:{n}] @ m.support_vectors_.T\n"
        how = "the linear kernel K(s, x) = s·x"
        ctor = "SVC(kernel='linear')"
    sol += "result = K @ m.dual_coef_[0] + m.intercept_[0]"
    return P(f"{desc}. Fit `{ctor}` on `X, y`. Its decision function is Σᵢ αᵢyᵢ K(sᵢ, x) + b, summed over the support "
             f"vectors sᵢ (`support_vectors_`), with αᵢyᵢ in `dual_coef_[0]` and b = `intercept_[0]`, using {how}. "
             f"Recompute `decision_function` for the first {n} rows by hand.", setup, sol,
             require=["dual_coef_", "support_vectors_"], tol=1e-6)


@template(M, "hard")
def gaussian_nb_proba_by_hand(rng):
    setup, desc, k, d = _scaled_clf(rng)
    n = rng.randint(3, 6)
    task = rng.choice(["proba", "jll"])
    sol = (f"{NB}GaussianNB\nm = GaussianNB().fit(X, y)\nx = X[:{n}, None, :]\n"
           "jll = np.log(m.class_prior_) - 0.5 * (np.log(2 * np.pi * m.var_)[None] + (x - m.theta_[None]) ** 2 / m.var_[None]).sum(axis=2)\n")
    head = (f"{desc}. Fit `GaussianNB()` on `X, y`. Its joint log-likelihood for class c is log P(c) + Σⱼ log N(xⱼ; "
            f"θ_cj, σ²_cj): use `class_prior_`, `theta_` and `var_` (`N` is the normal density).")
    if task == "jll":
        return P(f"{head} Compute it by hand for the first {n} rows (a rows × classes array).", setup, sol + "result = jll",
                 require=["theta_", "var_"], tol=1e-6)
    return P(f"{head} Compute it by hand for the first {n} rows and normalize with a softmax (subtract the row max "
             f"before `np.exp`) to rebuild `predict_proba`.", setup,
             sol + "e = np.exp(jll - jll.max(axis=1, keepdims=True))\nresult = e / e.sum(axis=1, keepdims=True)",
             require=["theta_", "var_"], tol=1e-6)


@template(M, "hard")
def multinomial_nb_proba_by_hand(rng):
    setup, desc, V = _docs(rng)
    a = rng.choice([0.5, 1.0])
    n = rng.randint(3, 6)
    model = rng.choice(["multinomial", "complement"])
    if model == "multinomial":
        return P(f"{desc}. Fit `MultinomialNB(alpha={a})` on `X, y`. A document's class score is `class_log_prior_[c]` + "
                 f"Σ_w count_w · `feature_log_prob_[c, w]`. Rebuild `predict_proba` of the first {n} documents from "
                 f"these scores (softmax).", setup,
                 f"{NB}MultinomialNB\nm = MultinomialNB(alpha={a}).fit(X, y)\n"
                 f"s = X[:{n}] @ m.feature_log_prob_.T + m.class_log_prior_\ne = np.exp(s - s.max(axis=1, keepdims=True))\n"
                 "result = e / e.sum(axis=1, keepdims=True)", require=["feature_log_prob_"], tol=1e-6)
    return P(f"{desc}. `ComplementNB(alpha={a})` learns from the **other** classes: for class c, count each word over "
             f"all documents not in c, add α, normalize per class, take logs and negate: `feature_log_prob_[c] = "
             f"−log((comp_c + α) / Σ_w (comp_c + α))`. Compute that matrix by hand and return it.", setup,
             "N = np.array([X[y == c].sum(axis=0) for c in np.unique(y)])\n"
             f"comp = N.sum(axis=0) - N + {a}\nresult = -np.log(comp / comp.sum(axis=1, keepdims=True))", tol=1e-9)


@template(M, "hard")
def weighted_knn_by_hand(rng):
    d = synth_reg(rng, n=(80, 200), d=(2, 5))
    setup = d.setup + "\nfrom sklearn.preprocessing import StandardScaler\nX = StandardScaler().fit_transform(X)\nQ = X[:4] + 0.1"
    k = rng.choice([3, 5, 8])
    return P(f"{d.desc}; `X` is standardized and `Q` holds 4 query points. Reproduce "
             f"`KNeighborsRegressor(n_neighbors={k}, weights='distance').predict(Q)` by hand: for each query, take the "
             f"{k} nearest rows (Euclidean) and average their targets with weights 1 / distance.", setup,
             f"D = np.sqrt(((Q[:, None, :] - X[None]) ** 2).sum(axis=2))\nnb = np.argsort(D, axis=1)[:, :{k}]\n"
             "w = 1 / np.take_along_axis(D, nb, axis=1)\nresult = (w * y[nb]).sum(axis=1) / w.sum(axis=1)",
             require=["argsort"], tol=1e-6)


@template(M, "hard")
def cv_kernel_choice(rng):
    t = table(rng, kind="clf", numeric_only=True, n=(150, 260))
    task = rng.choice(["best", "leak", "scores"])
    pipe = f"from sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\n{SVM}SVC\n"
    if task == "best":
        return P(f"{t.desc}. Compare `make_pipeline(StandardScaler(), SVC(kernel=k))` for k = 'linear' and 'rbf' with "
                 f"5-fold `cross_val_score` and return the better kernel's name and its mean score as a tuple (the first "
                 f"on ties).", t.setup,
                 f"{pipe}from sklearn.model_selection import cross_val_score\n"
                 "s = {k: cross_val_score(make_pipeline(StandardScaler(), SVC(kernel=k)), df, y, cv=5).mean() for k in ('linear', 'rbf')}\n"
                 "best = max(s, key=s.get)\nresult = (best, s[best])", require=["cross_val_score", "SVC"], tol=1e-6)
    if task == "leak":
        return P(f"{t.desc}. Return `(leaky, honest)` mean 5-fold `cross_val_score` of a k-NN classifier "
                 f"(`KNeighborsClassifier(n_neighbors=5)`): `leaky` standardizes all of `df` first, `honest` puts the "
                 f"scaler in a pipeline.", t.setup,
                 f"{pipe}{NN}KNeighborsClassifier\nfrom sklearn.model_selection import cross_val_score\n"
                 "Z = StandardScaler().fit_transform(df)\nleaky = cross_val_score(KNeighborsClassifier(n_neighbors=5), Z, y, cv=5).mean()\n"
                 "honest = cross_val_score(make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)), df, y, cv=5).mean()\n"
                 "result = (leaky, honest)", require=["cross_val_score", "KNeighborsClassifier"], tol=1e-6)
    return P(f"{t.desc}. Return the mean 5-fold `cross_val_score` of `SVC()` on the raw `df` and of "
             f"`make_pipeline(StandardScaler(), SVC())`, as a tuple: RBF kernels are distance-based.", t.setup,
             f"{pipe}from sklearn.model_selection import cross_val_score\n"
             "result = (cross_val_score(SVC(), df, y, cv=5).mean(), cross_val_score(make_pipeline(StandardScaler(), SVC()), df, y, cv=5).mean())",
             require=["cross_val_score"], tol=1e-6)


@template(M, "hard")
def curse_of_dimensionality(rng):
    setup, desc, k, d = _scaled_clf(rng)
    s = seeded(rng)
    extra = [0, 10, 50, 200]
    task = rng.choice(["knn", "ratio"])
    setup += f"\nnoise = np.random.default_rng({s}).normal(size=(len(X), 200))\nh = int(0.7 * len(X))"
    if task == "knn":
        return P(f"{desc}. `noise` holds 200 useless standard-normal columns. For m in `{extra}`, append the first m noise "
                 f"columns to `X`, train `KNeighborsClassifier(n_neighbors=5)` on the first `h` rows and score on the "
                 f"rest. Return the accuracies: k-NN drowns in irrelevant dimensions.", setup,
                 f"{NN}KNeighborsClassifier\nresult = []\nfor m in {extra}:\n    Z = np.hstack([X, noise[:, :m]])\n"
                 "    result.append(KNeighborsClassifier(n_neighbors=5).fit(Z[:h], y[:h]).score(Z[h:], y[h:]))",
                 require=["KNeighborsClassifier"])
    return P(f"{desc}. `noise` holds 200 useless standard-normal columns. In high dimensions all points are almost "
             f"equally far apart. For m in `{extra}`, append the first m noise columns to `X` and compute, for the first "
             f"20 rows, the ratio (distance to the nearest other row) / (distance to the farthest row) with "
             f"`NearestNeighbors(n_neighbors=len(X))`. Return the mean ratio for each m.", setup,
             f"{NN}NearestNeighbors\nresult = []\nfor m in {extra}:\n    Z = np.hstack([X, noise[:, :m]])\n"
             "    dist, idx = NearestNeighbors(n_neighbors=len(Z)).fit(Z).kneighbors(Z[:20])\n"
             "    result.append(float((dist[:, 1] / dist[:, -1]).mean()))", require=["NearestNeighbors"], tol=1e-6)


@template(M, "hard")
def broken_svm_knn_nb(rng):
    setup, desc, k, d = _scaled_clf(rng)
    n = rng.randint(2, 4)
    s = seeded(rng)
    items = [
        ("probability", f"This raises an `AttributeError` (`predict_proba` needs `probability=True` at construction):\n"
                        f"```\nSVC().fit(X, y).predict_proba(X[:{n}])\n```\nFix it with `random_state=0` and return the "
                        f"probabilities.",
         f"{SVM}SVC\nresult = SVC(probability=True, random_state=0).fit(X, y).predict_proba(X[:{n}])"),
        ("kneighbors", "To find each row's 3 nearest **other** rows, this code returns the row itself first:\n```\n"
                       "NearestNeighbors(n_neighbors=3).fit(X).kneighbors(X)[1]\n```\nAsk for 4 neighbours and drop "
                       "the first column; return the first 5 rows of the result.",
         f"{NN}NearestNeighbors\nresult = NearestNeighbors(n_neighbors=4).fit(X).kneighbors(X)[1][:5, 1:]"),
        ("MinMaxScaler", "`MultinomialNB` needs non-negative inputs, so this raises a `ValueError` on standardized "
                         "data:\n```\nMultinomialNB().fit(X, y)\n```\nRescale `X` into [0, 1] with `MinMaxScaler` first "
                         "and return the training accuracy.",
         f"{NB}MultinomialNB\nfrom sklearn.preprocessing import MinMaxScaler\nZ = MinMaxScaler().fit_transform(X)\n"
         "result = MultinomialNB().fit(Z, y).score(Z, y)"),
        ("GaussianNB", "Continuous features don't suit `MultinomialNB` even after shifting. Use the right model for "
                       "real-valued measurements, `GaussianNB()`, and return its training accuracy.",
         f"{NB}GaussianNB\nresult = GaussianNB().fit(X, y).score(X, y)"),
        ("n_neighbors", "This raises a `ValueError` (more neighbours than training rows):\n```\n"
                        "KNeighborsClassifier(n_neighbors=50).fit(X[:20], y[:20])\n            .predict(X[20:25])\n```\n"
                        "Use at most as many neighbours as rows (`n_neighbors=20`) and return the predictions.",
         f"{NN}KNeighborsClassifier\nresult = KNeighborsClassifier(n_neighbors=20).fit(X[:20], y[:20]).predict(X[20:25])"),
        ("transform", "This code standardizes the query with its **own** statistics, so a single row becomes all zeros:\n"
                      "```\nq = StandardScaler().fit_transform(raw[:1])\n```\n`raw` is `X` before scaling (here "
                      f"`X * 3 + 7`). Fit the scaler on `raw` instead, transform `raw[:1]` with it and return the "
                      f"indices of its 3 nearest rows in the scaled data.",
         f"{NN}NearestNeighbors\nfrom sklearn.preprocessing import StandardScaler\nsc = StandardScaler().fit(raw)\n"
         "Z = sc.transform(raw)\nresult = NearestNeighbors(n_neighbors=3).fit(Z).kneighbors(sc.transform(raw[:1]))[1]",
         {"setup": "raw = X * 3 + 7"}),
        ("random_state", f"`SVC(probability=True)` calibrates with an internal random split, so two runs can differ. "
                         f"Make it reproducible with `random_state={s}` and return `predict_proba(X[:{n}])[:, 0]` (the "
                         f"probability of the first class for the first {n} rows).",
         f"{SVM}SVC\nresult = SVC(probability=True, random_state={s}).fit(X, y).predict_proba(X[:{n}])[:, 0]"),
    ]
    return catalog(rng, items, setup, f"{desc}.", tol=1e-6)


@template(M, "hard")
def nb_workflows(rng):
    setup, desc, V = _docs(rng, n=(150, 300))
    s = seeded(rng)
    setup += "\nh = int(0.7 * len(y))"
    items = [
        ("cross_val_score", "Pick the smoothing by 5-fold CV: return the `alpha` among 0.01, 0.1, 1, 10 with the best mean "
                            "`cross_val_score` of `MultinomialNB(alpha=alpha)` on all documents (the smallest on ties).",
         f"{NB}MultinomialNB\nfrom sklearn.model_selection import cross_val_score\n"
         "s = {a: cross_val_score(MultinomialNB(alpha=a), X, y, cv=5).mean() for a in (0.01, 0.1, 1, 10)}\n"
         "result = max(s, key=s.get)"),
        ("predict_log_proba", "Rank the test documents (after the first `h`) by how sure `MultinomialNB()` (trained on the "
                              "first `h`) is of its prediction: return the positions (within the test part) of the 3 "
                              "least sure, least sure first (smallest top-class probability).",
         f"{NB}MultinomialNB\np = MultinomialNB().fit(X[:h], y[:h]).predict_proba(X[h:])\nresult = np.argsort(p.max(axis=1))[:3]", {"require": ["predict_proba"]}),
        ("class_prior", "Priors can be set by hand: return the number of test documents predicted as class 0 by "
                        "`MultinomialNB()` and by `MultinomialNB(fit_prior=False)` (uniform priors), both trained on the "
                        "first `h` documents, as a tuple.",
         f"{NB}MultinomialNB\na = int((MultinomialNB().fit(X[:h], y[:h]).predict(X[h:]) == 0).sum())\n"
         "b = int((MultinomialNB(fit_prior=False).fit(X[:h], y[:h]).predict(X[h:]) == 0).sum())\nresult = (a, b)",
         {"require": ["fit_prior"]}),
        ("ComplementNB", f"Make class 0 rare: keep only 10% of its training documents (rows among the first `h` with "
                         f"`y == 0`, kept where `np.random.default_rng({s}).random(h) < 0.1`). Return the test recall of "
                         f"class 0 for `MultinomialNB()` and `ComplementNB()` trained on the reduced set, as a tuple.",
         f"{NB}MultinomialNB, ComplementNB\nkeep = (y[:h] != 0) | (np.random.default_rng({s}).random(h) < 0.1)\n"
         "Xt, yt = X[:h][keep], y[:h][keep]\nyte = y[h:]\n"
         "r = [float((m.fit(Xt, yt).predict(X[h:])[yte == 0] == 0).mean()) for m in (MultinomialNB(), ComplementNB())]\n"
         "result = tuple(r)"),
        ("BernoulliNB", "Presence vs counts: double every count (`2 * X`). Return whether the predictions of "
                        "`BernoulliNB()` and of `MultinomialNB()` (both trained on the first `h` doubled documents, "
                        "evaluated on the doubled test part) change compared with the undoubled data, as a tuple of "
                        "bools `(bernoulli_changed, multinomial_changed)`.",
         f"{NB}BernoulliNB, MultinomialNB\nout = []\nfor M_ in (BernoulliNB, MultinomialNB):\n"
         "    a = M_().fit(X[:h], y[:h]).predict(X[h:])\n    b = M_().fit(2 * X[:h], y[:h]).predict(2 * X[h:])\n"
         "    out.append(bool((a != b).any()))\nresult = tuple(out)"),
        ("NearestNeighbors", "k-NN with cosine distance vs naive Bayes on documents: return the test accuracy of "
                             "`KNeighborsClassifier(n_neighbors=5, metric='cosine')` and of `MultinomialNB()`, both "
                             "trained on the first `h` documents, as a tuple.",
         f"{NB}MultinomialNB\n{NN}KNeighborsClassifier\n"
         "a = KNeighborsClassifier(n_neighbors=5, metric='cosine').fit(X[:h], y[:h]).score(X[h:], y[h:])\n"
         "result = (a, MultinomialNB().fit(X[:h], y[:h]).score(X[h:], y[h:]))", {"require": ["KNeighborsClassifier"]}),
    ]
    return catalog(rng, items, setup, f"{desc}. `h` = 70% of the documents (the training part).", tol=1e-6)


@template(M, "hard")
def neighbours_and_margins_workflows(rng):
    setup, desc, k, d = _scaled_clf(rng)
    s = seeded(rng)
    t = rng.choice([0.5, 1.0, 1.5])
    setup += f"\norder = np.random.default_rng({s}).permutation(len(y))\nX, y = X[order], y[order]\nh = int(0.7 * len(y))"
    items = [
        ("radius_neighbors", f"Near-duplicate detection: how many **pairs** of distinct rows are within distance {t} of "
                             f"each other? Use `NearestNeighbors(radius={t}).fit(X).radius_neighbors(X)` (each row "
                             f"finds itself too; each pair is found twice).",
         f"{NN}NearestNeighbors\ndist, idx = NearestNeighbors(radius={t}).fit(X).radius_neighbors(X)\n"
         "result = int((sum(len(i) for i in idx) - len(X)) // 2)"),
        ("cross_val_predict", "Return the 5-fold out-of-fold confusion counts of `KNeighborsClassifier(n_neighbors=5)`: "
                              "a classes × classes int array `C[t, p]` built from `cross_val_predict`.",
         f"{NN}KNeighborsClassifier\nfrom sklearn.model_selection import cross_val_predict\n"
         "pred = cross_val_predict(KNeighborsClassifier(n_neighbors=5), X, y, cv=5)\nk = len(np.unique(y))\n"
         "result = np.zeros((k, k), dtype=int)\nnp.add.at(result, (y, pred), 1)"),
        ("score", "A learning curve: train `SVC()` on the first m rows for m in 20, 40, 80 and `h`, and return the "
                  "accuracies on the rows after `h`.",
         f"{SVM}SVC\nresult = [SVC().fit(X[:m], y[:m]).score(X[h:], y[h:]) for m in (20, 40, 80, h)]"),
        ("GridSearchCV", "Tune k-NN: `GridSearchCV(KNeighborsClassifier(), {'n_neighbors': [1, 3, 5, 9, 15], 'weights': "
                         "['uniform', 'distance']}, cv=5)` on the first `h` rows. Return `(best_params_, test accuracy)`.",
         f"{NN}KNeighborsClassifier\nfrom sklearn.model_selection import GridSearchCV\n"
         "gs = GridSearchCV(KNeighborsClassifier(), {'n_neighbors': [1, 3, 5, 9, 15], 'weights': ['uniform', 'distance']}, "
         "cv=5).fit(X[:h], y[:h])\nresult = (gs.best_params_, gs.score(X[h:], y[h:]))"),
        ("support_", "Support vectors are the hard cases: fit `SVC()` on `X, y` and return `(mean score over support "
                     "vectors, mean score over the other rows)`, where a row's score is `abs(decision_function)` for "
                     "binary data and the row's largest `decision_function` value for multiclass data.",
         f"{SVM}SVC\nm = SVC().fit(X, y)\ndf_ = m.decision_function(X)\n"
         "score = np.abs(df_) if df_.ndim == 1 else df_.max(axis=1)\nis_sv = np.zeros(len(X), bool)\nis_sv[m.support_] = True\n"
         "result = (float(score[is_sv].mean()), float(score[~is_sv].mean()))"),
        ("NearestCentroid", "Shrinking centroids toward the overall mean drops noisy features: return the test accuracy of "
                            "`NearestCentroid(shrink_threshold=t)` trained on the first `h` rows for t in None, 0.2, 0.5, 1.0.",
         f"{NN}NearestCentroid\n"
         "result = [NearestCentroid(shrink_threshold=t).fit(X[:h], y[:h]).score(X[h:], y[h:]) for t in (None, 0.2, 0.5, 1.0)]"),
    ]
    return catalog(rng, items, setup, f"{desc}. Rows were reshuffled; `h` = 70% of the rows (training part).", tol=1e-6)


# ================================================================= challenge

@template(M, "challenge")
def knn_on_a_line(rng):
    n, q, s = rng.randint(8, 12), rng.randint(4, 6), seeded(rng)
    mode = rng.choice(["knn_mean", "knn_vote", "radius_mean"])
    if mode == "radius_mean":
        par, tpar = f"r = {round(rng.uniform(0.5, 1.5), 2)}", "r = 0.0004"
    else:
        par, tpar = f"k = {rng.randint(2, 4)}", "k = 40"
    ytxt = "g.integers(0, 2, {n})" if mode == "knn_vote" else "g.normal(0, 10, {n}).round(2)"
    gen = ("g = np.random.default_rng({s})\nx = g.uniform(0, 10, {n})\ny = " + ytxt +
           "\nq = g.uniform(-1, 11, {q})\n{par}\nmode = {mode!r}")
    setup = gen.format(s=s, n=n, q=q, par=par, mode=mode)
    tests = [gen.format(s=s + i, n=100000, q=100000, par=tpar, mode=mode) for i in (1, 2)]
    sol = "order = np.argsort(x)\na, v = x[order], y[order].astype(float)\nn = len(a)\npre = np.concatenate([[0.0], np.cumsum(v)])\n"
    if mode == "radius_mean":
        sol += ("lo = np.searchsorted(a, q - r, 'left')\nhi = np.searchsorted(a, q + r, 'right')\ncnt = hi - lo\n"
                "with np.errstate(invalid='ignore', divide='ignore'):\n    result = (pre[hi] - pre[lo]) / cnt\n"
                "result[cnt == 0] = np.nan")
        what = ("`RadiusNeighborsRegressor(radius=r)`: the mean `y` of all training points within distance `r` "
                "(inclusive) of the query, or nan when there are none")
    else:
        sol += ("lo = np.zeros(len(q), dtype=int)\nhi = np.full(len(q), n - k)\nwhile (lo < hi).any():   # leftmost best window start\n"
                "    act = lo < hi\n    mid = (lo + hi) // 2\n    right = (q - a[mid]) > (a[np.minimum(mid + k, n - 1)] - q)\n"
                "    lo = np.where(act & right, mid + 1, lo)\n    hi = np.where(act & ~right, mid, hi)\n"
                "tot = pre[lo + k] - pre[lo]\n")
        if mode == "knn_mean":
            sol += "result = tot / k"
            what = "`KNeighborsRegressor(n_neighbors=k)`: the mean `y` of the `k` nearest training points"
        else:
            sol += "result = (2 * tot > k).astype(int)"
            what = ("`KNeighborsClassifier(n_neighbors=k)` with 0/1 labels: the majority label of the `k` nearest "
                    "training points (a tie goes to 0, the smaller label)")
    return challenge(
        "Nearest neighbours on a line",
        f"Training points `x` (distinct floats) with targets `y` live on a line; for every query `q[i]` predict like "
        f"{what}. Return the predictions as an array. (`mode` = {mode!r}; no distance ties occur.) Hint: the k nearest "
        f"points of a sorted line form a contiguous window.",
        setup, sol, tests, "len(x), number of queries ≤ 10^5, k ≤ 50",
        speed="O((n + q) log n): sort once, binary-search each query, prefix sums",
        answer_desc=" (a NumPy array, one value per query)", tol=1e-6)


@template(M, "challenge")
def naive_bayes_sparse(rng):
    model = rng.choice(["multinomial", "complement", "bernoulli"])
    s = seeded(rng)
    nd, nt, V, C = rng.randint(6, 8), rng.randint(3, 4), rng.randint(5, 7), rng.randint(2, 3)
    gen = ("g = np.random.default_rng({s})\nV = {V}\ny = g.integers(0, {C}, {nd})\ny[:{C}] = np.arange({C})\n"
           "doc = g.integers(0, {nd}, {nnz})\nterm = (g.integers(0, V, {nnz}) + 3 * y[doc]) % V   # topic-dependent words\n"
           "count = g.integers(1, 4, {nnz})\nn_test = {nt}\ntdoc = g.integers(0, n_test, {tnnz})\n"
           "tterm = g.integers(0, V, {tnnz})\ntcount = g.integers(1, 4, {tnnz})\nalpha = {alpha}\nmodel = {model!r}")
    a = rng.choice([0.5, 1.0])
    setup = gen.format(s=s, V=V, C=C, nd=nd, nnz=rng.randint(15, 25), nt=nt, tnnz=rng.randint(6, 10), alpha=a, model=model)
    tests = [gen.format(s=s + i, V=30000, C=C, nd=20000, nnz=200000, nt=20000, tnnz=200000, alpha=a, model=model)
             for i in (1, 2)]
    sol = "C = y.max() + 1\nprior = np.log(np.bincount(y, minlength=C) / len(y))\n"
    if model == "multinomial":
        sol += ("N = np.bincount(y[doc] * V + term, weights=count, minlength=C * V).reshape(C, V)\n"
                "flp = np.log((N + alpha) / (N.sum(axis=1, keepdims=True) + alpha * V))\n"
                "S = np.array([np.bincount(tdoc, weights=tcount * flp[c, tterm], minlength=n_test) for c in range(C)])\n"
                "S += prior[:, None]\n")
        rule = ("`MultinomialNB(alpha=alpha)`: P(word | class) = (count of the word in the class + α) / (all words of "
                "the class + α·V); a test document's score for class c is log P(c) + Σ count · log P(word | c)")
    elif model == "complement":
        sol += ("N = np.bincount(y[doc] * V + term, weights=count, minlength=C * V).reshape(C, V)\n"
                "comp = N.sum(axis=0) - N + alpha\nflp = -np.log(comp / comp.sum(axis=1, keepdims=True))\n"
                "S = np.array([np.bincount(tdoc, weights=tcount * flp[c, tterm], minlength=n_test) for c in range(C)])\n")
        rule = ("`ComplementNB(alpha=alpha)`: for class c, count each word over the documents **not** in c and add α "
                "(comp); its weight is w_cw = −log(comp_cw / Σ_w comp_cw); a test document's score for class c is "
                "Σ count · w_cw (no prior when there are several classes)")
    else:
        sol += ("pairs = np.unique(doc * V + term)   # (document, word) presence, repeats removed\n"
                "pd_, pt = pairs // V, pairs % V\n"
                "Np = np.bincount(y[pd_] * V + pt, minlength=C * V).reshape(C, V)\nnc = np.bincount(y, minlength=C)\n"
                "p = (Np + alpha) / (nc[:, None] + 2 * alpha)\nlp, lq = np.log(p), np.log1p(-p)\n"
                "tp = np.unique(tdoc * V + tterm)\ntd, tw = tp // V, tp % V\n"
                "S = np.array([np.bincount(td, weights=lp[c, tw] - lq[c, tw], minlength=n_test) for c in range(C)])\n"
                "S += (lq.sum(axis=1) + prior)[:, None]\n")
        rule = ("`BernoulliNB(alpha=alpha)` on word **presence** (count > 0): P(present | class) = (documents of the "
                "class containing the word + α) / (documents of the class + 2α); a test document's score for class c "
                "is log P(c) + Σ over all V words of log P(present | c) if present else log(1 − P(present | c))")
    sol += "result = S.argmax(axis=0)"
    return challenge(
        "Naive Bayes on a sparse corpus",
        f"Training documents come as a sparse log: entry `i` says document `doc[i]` contains word `term[i]` `count[i]` "
        f"times ((document, word) pairs may repeat: counts add up); document `j` has class `y[j]` (every class "
        f"occurs; some documents are empty). Test documents `0 … n_test − 1` are given the same way by `tdoc, tterm, "
        f"tcount`. Classify the test documents like {rule}. Predict the class with the highest score (ties: the "
        f"smallest class) and return the labels as an int array of length `n_test`. (`model` = {model!r}.)",
        setup, sol, tests, "≤ 2·10^4 documents per set, V ≤ 3·10^4 words, ≤ 2·10^5 log entries per set, ≤ 3 classes",
        speed="O(entries + V · classes): never build a dense document × word matrix",
        answer_desc=" (an int array)")


@template(M, "challenge")
def kernel_collapse(rng):
    m, q, d, s = rng.randint(5, 8), rng.randint(3, 5), rng.randint(2, 3), seeded(rng)
    kernel = rng.choice(["linear", "poly2"])
    report = rng.choice(["decision", "predict"])
    gen = ("g = np.random.default_rng({s})\nsv = g.normal(0, 1, ({m}, {d}))   # support vectors\n"
           "dual = g.normal(0, 1, {m})   # alpha_i * y_i (dual_coef_)\nb = {b}\ngamma = {gamma}\ncoef0 = {coef0}\n"
           "Q = g.normal(0, 1, ({q}, {d}))   # query points\nkernel = {kernel!r}\nreport = {report!r}")
    b, gamma, coef0 = round(rng.uniform(-1, 1), 2), rng.choice([0.1, 0.25, 0.5]), rng.choice([0.0, 1.0, 2.0])
    setup = gen.format(s=s, m=m, d=d, q=q, b=b, gamma=gamma, coef0=coef0, kernel=kernel, report=report)
    tests = [gen.format(s=s + i, m=50000, d=6, q=50000, b=b, gamma=gamma, coef0=coef0, kernel=kernel, report=report)
             for i in (1, 2)]
    if kernel == "linear":
        sol = "w = dual @ sv\nf = Q @ w + b\n"
        kdesc = "the linear kernel K(s, x) = s·x"
    else:
        sol = ("A = sv.T @ (dual[:, None] * sv)   # Σ dual_i s_i s_iᵀ\nw = dual @ sv\n"
               "f = gamma ** 2 * ((Q @ A) * Q).sum(axis=1) + 2 * gamma * coef0 * (Q @ w) + coef0 ** 2 * dual.sum() + b\n")
        kdesc = "the polynomial kernel K(s, x) = (γ·s·x + coef0)² (`SVC(kernel='poly', degree=2)`)"
    sol += "result = f" if report == "decision" else "result = (f > 0).astype(int)"
    what = ("the decision values f(x) as a float array" if report == "decision" else
            "the predicted labels (1 where f(x) > 0, else 0) as an int array")
    return challenge(
        "Collapsing the kernel expansion",
        f"A trained SVM decides with f(x) = Σᵢ `dual[i]` · K(`sv[i]`, x) + `b`, a sum over its support vectors, using "
        f"{kdesc} (`gamma`, `coef0` given). Evaluating that sum for every query costs (support vectors × queries). "
        f"Return {what} for every row of `Q`, much faster: for these kernels the sum can be folded into a few "
        f"precomputed arrays. (`kernel` = {kernel!r}, `report` = {report!r}.)",
        setup, sol, tests, "≤ 5·10^4 support vectors and queries, d ≤ 6",
        speed="O((m + q) · d²): never form the m × q kernel matrix", answer_desc=" (a NumPy array)", tol=1e-6)
