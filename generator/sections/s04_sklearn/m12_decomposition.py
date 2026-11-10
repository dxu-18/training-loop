"""4.12 Dimensionality reduction & embeddings: PCA and friends, matrix factorizations, manifold learning, LDA."""

from ._common import P, catalog, challenge, lit, seeded, synth_clf, table, template

M = "4.12"
WORDS = ["goal", "match", "team", "coach", "vote", "party", "law", "senate", "gene", "cell", "virus", "protein"]


def _digits(rng, n=(300, 450)):
    n_ = rng.randint(*n)
    a = rng.randint(0, 1797 - n_)
    return (f"from sklearn.datasets import load_digits\nX, y = load_digits(return_X_y=True)\n"
            f"X, y = X[{a}:{a + n_}], y[{a}:{a + n_}]",
            f"`X` holds {n_} handwritten digits (rows {a} to {a + n_ - 1} of `load_digits`: 8×8 images flattened to 64 "
            f"pixel values from 0 to 16) and `y` the digit shown", n_, 64)


def _lowrank(rng):
    n, d, r, s = rng.randint(150, 300), rng.randint(6, 10), rng.randint(2, 3), seeded(rng)
    nz = rng.choice([0.1, 0.2, 0.3])
    return (f"g = np.random.default_rng({s})\nX = g.normal(0, 1, ({n}, {r})) @ g.normal(0, 1, ({r}, {d})) + "
            f"g.normal(0, {nz}, ({n}, {d}))",
            f"`X` ({n} × {d}) is a rank-{r} signal plus a little noise (σ = {nz})", n, d)


def _scaled(rng):
    name = rng.choice(["wine", "cancer"])
    fn, n, d, what = {"wine": ("load_wine", 178, 13, "178 wines (13 chemical measurements)"),
                      "cancer": ("load_breast_cancer", 569, 30, "569 breast-tumour samples (30 cell measurements)")}[name]
    return (f"from sklearn.datasets import {fn}\nfrom sklearn.preprocessing import StandardScaler\n"
            f"X, y = {fn}(return_X_y=True)\nX = StandardScaler().fit_transform(X)",
            f"`X` holds {what}, standardized, and `y` their class", n, d)


def _dense(rng):
    return rng.choice([_digits, _lowrank, _scaled, _scaled])(rng)


def _counts(rng, n=(50, 80)):
    n_, s = rng.randint(*n), seeded(rng)
    setup = (f"g = np.random.default_rng({s})\nwords = {lit(WORDS)}\ntopic = g.integers(0, 3, {n_})\n"
             "P_ = np.full((3, 12), 0.03) + np.kron(np.eye(3), np.ones(4)) * 0.22   # each topic favours its 4 words\n"
             "counts = np.array([g.multinomial(25, P_[t] / P_[t].sum()) for t in topic])")
    desc = (f"`counts` is a {n_} × 12 word-count matrix: {n_} short documents over the vocabulary `words` (sports, "
            f"politics and biology terms, 4 each); each document was written about one hidden topic `topic` (0, 1, 2)")
    return setup, desc, n_


def _scurve(rng, n=(150, 250)):
    n_, s = rng.randint(*n), seeded(rng)
    return (f"from sklearn.datasets import make_s_curve\nX, t = make_s_curve(n_samples={n_}, noise=0.05, random_state={s})",
            f"`X` holds {n_} points on an S-shaped 2-D sheet bent through 3-D space (`make_s_curve`), and `t` is each "
            f"point's position along the S", n_)


def _ordinal(k):
    return {1: "1st", 2: "2nd", 3: "3rd"}.get(k, f"{k}th")


# ================================================================= lite

@template(M, "lite")
def pca_fitted_attributes(rng):
    setup, desc, n, d = _dense(rng)
    k = rng.randint(2, 4)
    t = rng.choice([0.8, 0.9, 0.95])
    fit = f"from sklearn.decomposition import PCA\npca = PCA(n_components={k}).fit(X)\n"
    head = f"Fit `pca = PCA(n_components={k})` (from `sklearn.decomposition`) on `X` and return"
    items = [
        ("explained_variance_ratio_", f"{head} the share of the variance each component explains "
                                      f"(`explained_variance_ratio_`).", fit + "result = pca.explained_variance_ratio_"),
        ("components_", f"{head} the shape of `components_` as a tuple (one row of loadings per component).",
         fit + "result = pca.components_.shape"),
        ("transform", f"{head} the coordinates of the first 3 rows in the new space (`transform`).",
         fit + "result = pca.transform(X[:3])"),
        ("inverse_transform", f"{head} the shape of `pca.inverse_transform(pca.transform(X))`: mapping back gives "
                              f"the original number of columns.", fit + "result = pca.inverse_transform(pca.transform(X)).shape"),
        ("n_components_", f"Fit `PCA(n_components={t})` (a float: keep enough components for {t:.0%} of the variance) "
                          f"on `X` and return how many it kept (`n_components_`).",
         f"from sklearn.decomposition import PCA\nresult = PCA(n_components={t}).fit(X).n_components_"),
        ("cumsum", "Fit `PCA()` (all components) on `X` and return the cumulative explained variance ratio of the first "
                   "5 components (`np.cumsum`).",
         "from sklearn.decomposition import PCA\nresult = np.cumsum(PCA().fit(X).explained_variance_ratio_)[:5]"),
        ("mean_", f"{head} the first 5 entries of `mean_` (PCA centres the data before projecting).",
         fit + "result = pca.mean_[:5]"),
        ("explained_variance_", f"{head} the variance along each component (`explained_variance_`).",
         fit + "result = pca.explained_variance_"),
        ("singular_values_", f"{head} its `singular_values_`.", fit + "result = pca.singular_values_"),
        ("fit_transform", f"Return the shape of `PCA(n_components={k}).fit_transform(X)` as a tuple.",
         f"from sklearn.decomposition import PCA\nresult = PCA(n_components={k}).fit_transform(X).shape"),
    ]
    return catalog(rng, items, setup, f"{desc}.", tol=1e-6)


@template(M, "lite")
def more_decompositions(rng):
    setup, desc, n, d = _digits(rng, n=(200, 400))
    k = rng.randint(3, 8)
    b = rng.choice([50, 100])
    g = rng.choice([1e-4, 5e-4, 1e-3])
    D = "from sklearn.decomposition import "
    items = [
        ("TruncatedSVD", f"Fit `TruncatedSVD(n_components={k}, random_state=0)` (an SVD without centring, so it also "
                         f"works on sparse matrices) on `X` and return its `explained_variance_ratio_.sum()`.",
         f"{D}TruncatedSVD\nresult = TruncatedSVD(n_components={k}, random_state=0).fit(X).explained_variance_ratio_.sum()"),
        ("NMF", f"Fit `NMF(n_components={k}, init='nndsvda', max_iter=500, random_state=0)` (non-negative factors W and "
                f"H with X ≈ W @ H) on `X` and return its `reconstruction_err_` (the Frobenius norm of X − WH).",
         f"{D}NMF\nresult = NMF(n_components={k}, init='nndsvda', max_iter=500, random_state=0).fit(X).reconstruction_err_"),
        ("FastICA", f"Fit `FastICA(n_components={k}, whiten='unit-variance', max_iter=1000, random_state=0)` on `X` and "
                    f"return the shape of its `mixing_` matrix as a tuple.",
         f"{D}FastICA\nresult = FastICA(n_components={k}, whiten='unit-variance', max_iter=1000, random_state=0).fit(X).mixing_.shape"),
        ("FactorAnalysis", f"Fit `FactorAnalysis(n_components={k}, random_state=0)` on `X` and return the indices of the "
                           f"3 pixels with the largest estimated noise variance (`noise_variance_`), largest first.",
         f"{D}FactorAnalysis\nfa = FactorAnalysis(n_components={k}, random_state=0).fit(X)\n"
         "result = np.argsort(fa.noise_variance_)[::-1][:3]"),
        ("IncrementalPCA", f"Fit `IncrementalPCA(n_components={k}, batch_size={b})` (PCA computed batch by batch, for "
                           f"data that doesn't fit in memory) on `X` and return its `explained_variance_ratio_`.",
         f"{D}IncrementalPCA\nresult = IncrementalPCA(n_components={k}, batch_size={b}).fit(X).explained_variance_ratio_"),
        ("KernelPCA", f"Fit `KernelPCA(n_components=2, kernel='rbf', gamma={g})` on the first 150 rows of `X` and "
                      f"return their first 3 projected rows (`fit_transform(X[:150])[:3]`).",
         f"{D}KernelPCA\nresult = KernelPCA(n_components=2, kernel='rbf', gamma={g}).fit_transform(X[:150])[:3]"),
        ("LinearDiscriminantAnalysis", "Fit `LinearDiscriminantAnalysis(n_components=2)` (from "
                                       "`sklearn.discriminant_analysis`; supervised: it uses `y` to separate the classes) "
                                       "on `X, y` and return its `explained_variance_ratio_`.",
         "from sklearn.discriminant_analysis import LinearDiscriminantAnalysis\n"
         "result = LinearDiscriminantAnalysis(n_components=2).fit(X, y).explained_variance_ratio_"),
        ("TruncatedSVD", f"Return the shape of `TruncatedSVD(n_components={k}, random_state=0).fit_transform(X)`.",
         f"{D}TruncatedSVD\nresult = TruncatedSVD(n_components={k}, random_state=0).fit_transform(X).shape"),
    ]
    return catalog(rng, items, setup, f"{desc}.", tol=1e-5)


@template(M, "lite")
def manifold_learners(rng):
    setup, desc, n = _scurve(rng)
    nn, p = rng.choice([8, 10, 12]), rng.choice([10, 20, 30])
    MF = "from sklearn.manifold import "
    items = [
        ("TSNE", f"Embed `X` in 2-D with `TSNE(n_components=2, perplexity={p}, init='pca', learning_rate='auto', "
                 f"max_iter=300, random_state=0)` (from `sklearn.manifold`) and return its final `kl_divergence_` (how "
                 f"well the 2-D neighbourhoods match the original ones; lower is better).",
         f"{MF}TSNE\nresult = TSNE(n_components=2, perplexity={p}, init='pca', learning_rate='auto', max_iter=300, "
         f"random_state=0).fit(X).kl_divergence_", {"tol": 1e-4}),
        ("Isomap", f"Embed `X` with `Isomap(n_neighbors={nn}, n_components=2)` (from `sklearn.manifold`) and return the "
                   f"embedding of the first 3 points.",
         f"{MF}Isomap\nresult = Isomap(n_neighbors={nn}, n_components=2).fit_transform(X)[:3]"),
        ("SpectralEmbedding", "Embed `X` with `SpectralEmbedding(n_components=2, random_state=0)` and return the "
                              "embedding of the first 3 points.",
         f"{MF}SpectralEmbedding\nresult = SpectralEmbedding(n_components=2, random_state=0).fit_transform(X)[:3]"),
        ("LocallyLinearEmbedding", f"Fit `LocallyLinearEmbedding(n_neighbors={nn}, n_components=2, random_state=0)` on "
                                   f"`X` and return its `reconstruction_error_`.",
         f"{MF}LocallyLinearEmbedding\nresult = LocallyLinearEmbedding(n_neighbors={nn}, n_components=2, "
         f"random_state=0).fit(X).reconstruction_error_", {"tol": 1e-4}),
        ("Isomap", f"Fit `Isomap(n_neighbors={nn}, n_components=2)` on `X` and return its `reconstruction_error()`.",
         f"{MF}Isomap\nresult = Isomap(n_neighbors={nn}, n_components=2).fit(X).reconstruction_error()"),
        ("TSNE", "t-SNE only embeds the points it was fitted on: return `(hasattr(TSNE(), 'transform'), "
                 "hasattr(Isomap(), 'transform'))` (both from `sklearn.manifold`).",
         f"{MF}TSNE, Isomap\nresult = (hasattr(TSNE(), 'transform'), hasattr(Isomap(), 'transform'))"),
    ]
    return catalog(rng, items, setup, f"{desc}.", tol=1e-6)


# ================================================================= easy

@template(M, "easy")
def components_for_variance(rng):
    setup, desc, n, d = _dense(rng)
    t = rng.choice([0.8, 0.9, 0.95, 0.99])
    task = rng.choice(["cumsum", "float", "both"])
    if task == "cumsum":
        return P(f"{desc}. How many principal components keep {t:.0%} of the variance? Fit `PCA()` on `X`, take "
                 f"`np.cumsum` of `explained_variance_ratio_` and return the smallest number of components whose "
                 f"cumulative ratio reaches {t}.", setup,
                 f"from sklearn.decomposition import PCA\nc = np.cumsum(PCA().fit(X).explained_variance_ratio_)\n"
                 f"result = int(np.argmax(c >= {t})) + 1", require=["explained_variance_ratio_", "cumsum"])
    if task == "float":
        return P(f"{desc}. Fit `PCA(n_components={t})` on `X` and return `(pca.n_components_, kept)`, where `kept` is "
                 f"the variance ratio the kept components explain together.", setup,
                 f"from sklearn.decomposition import PCA\npca = PCA(n_components={t}).fit(X)\n"
                 "result = (pca.n_components_, pca.explained_variance_ratio_.sum())",
                 require=["n_components_"], tol=1e-6)
    return P(f"{desc}. Find the number of components for {t:.0%} of the variance in two ways: from the `np.cumsum` of "
             f"`PCA().fit(X).explained_variance_ratio_`, and as `PCA(n_components={t}).fit(X).n_components_`. Return "
             f"both as a tuple (they agree).", setup,
             f"from sklearn.decomposition import PCA\nc = np.cumsum(PCA().fit(X).explained_variance_ratio_)\n"
             f"result = (int(np.argmax(c >= {t})) + 1, PCA(n_components={t}).fit(X).n_components_)",
             require=["cumsum", "n_components_"])


@template(M, "easy")
def pca_by_hand(rng):
    setup, desc, n, d = rng.choice([_lowrank, _scaled])(rng)
    k = rng.randint(2, 3)
    task = rng.choice(["eigvals", "ratio", "total", "project"])
    if task == "eigvals":
        return P(f"{desc}. PCA's `explained_variance_` are the eigenvalues of the covariance matrix of `X` (`np.cov(X, "
                 f"rowvar=False)`, which divides by n − 1). Compute them with `np.linalg.eigvalsh`, sort them in "
                 f"decreasing order and return the top {k}; they equal `PCA(n_components={k}).fit(X)"
                 f".explained_variance_`.", setup,
                 f"result = np.sort(np.linalg.eigvalsh(np.cov(X, rowvar=False)))[::-1][:{k}]", require=["eigvalsh"],
                 tol=1e-6)
    if task == "ratio":
        return P(f"{desc}. Rebuild `PCA().fit(X).explained_variance_ratio_` with NumPy: the eigenvalues of the "
                 f"covariance matrix, sorted in decreasing order, divided by their sum. Return the array.", setup,
                 "ev = np.sort(np.linalg.eigvalsh(np.cov(X, rowvar=False)))[::-1]\nresult = ev / ev.sum()",
                 require=[], tol=1e-6)
    if task == "total":
        return P(f"{desc}. PCA only rotates the data, so it keeps the total variance. Fit `PCA()` on `X` and return "
                 f"`(pca.explained_variance_.sum(), X.var(axis=0, ddof=1).sum())`.", setup,
                 "from sklearn.decomposition import PCA\nresult = (PCA().fit(X).explained_variance_.sum(), "
                 "X.var(axis=0, ddof=1).sum())", require=["explained_variance_"], tol=1e-6)
    return P(f"{desc}. Fit `pca = PCA(n_components={k})` on `X`. `transform` centres the data and projects it on the "
             f"components: reproduce `pca.transform(X[:5])` by hand from `pca.mean_` and `pca.components_` and return "
             f"your array.", setup,
             f"from sklearn.decomposition import PCA\npca = PCA(n_components={k}).fit(X)\n"
             "result = (X[:5] - pca.mean_) @ pca.components_.T", require=["mean_", "components_"], tol=1e-6)


@template(M, "easy")
def reconstruction_error(rng):
    setup, desc, n, d = _digits(rng)
    k, i = rng.choice([5, 10, 20]), rng.randint(0, 50)
    task = rng.choice(["one", "all", "sweep"])
    if task == "one":
        return P(f"{desc}. Reconstruct digit {i} from {k} components: fit `pca = PCA(n_components={k})` on `X`, map row "
                 f"{i} into PCA space and back (`inverse_transform`), and return `(reconstructed row rounded to 1 "
                 f"decimal, mean squared error against the original row)`.", setup,
                 f"from sklearn.decomposition import PCA\npca = PCA(n_components={k}).fit(X)\n"
                 f"r = pca.inverse_transform(pca.transform(X[{i}:{i + 1}]))[0]\n"
                 f"result = (r.round(1), float(((r - X[{i}]) ** 2).mean()))", require=["inverse_transform"], tol=1e-6)
    if task == "all":
        return P(f"{desc}. Fit `pca = PCA(n_components={k})` on `X`, reconstruct every row (`inverse_transform` of "
                 f"`transform`) and return the mean squared error per pixel over the whole dataset.", setup,
                 f"from sklearn.decomposition import PCA\npca = PCA(n_components={k}).fit(X)\n"
                 "result = float(((pca.inverse_transform(pca.transform(X)) - X) ** 2).mean())",
                 require=["inverse_transform"], tol=1e-6)
    ks = [1, 5, 10, 20, 40]
    return P(f"{desc}. For `n_components` in `{ks}`, fit a `PCA` on `X`, reconstruct `X` and record the mean squared "
             f"error per pixel. Return the list of errors (it falls as components are added).", setup,
             "from sklearn.decomposition import PCA\nresult = []\n"
             f"for k in {ks}:\n    pca = PCA(n_components=k).fit(X)\n"
             "    result.append(float(((pca.inverse_transform(pca.transform(X)) - X) ** 2).mean()))",
             require=["inverse_transform"], tol=1e-6)


@template(M, "easy")
def whitening(rng):
    setup, desc, n, d = rng.choice([_lowrank, _scaled, _digits])(rng)
    k = rng.randint(2, 4)
    task = rng.choice(["std", "compare", "scale"])
    if task == "std":
        return P(f"{desc}. `PCA(n_components={k}, whiten=True)` rescales every component to unit variance. Return the "
                 f"standard deviations (`ddof=1`) of the columns of `PCA(n_components={k}, whiten=True).fit_transform(X)` "
                 f"(all 1).", setup,
                 f"from sklearn.decomposition import PCA\nresult = PCA(n_components={k}, whiten=True).fit_transform(X).std(axis=0, ddof=1)",
                 require=["whiten"], tol=1e-6)
    if task == "compare":
        return P(f"{desc}. Return the standard deviations (`ddof=1`) of the columns of `PCA(n_components={k})"
                 f".fit_transform(X)` and of the same with `whiten=True`, as a tuple of arrays: without whitening they "
                 f"are the square roots of `explained_variance_`.", setup,
                 f"from sklearn.decomposition import PCA\na = PCA(n_components={k}).fit_transform(X).std(axis=0, ddof=1)\n"
                 f"b = PCA(n_components={k}, whiten=True).fit_transform(X).std(axis=0, ddof=1)\nresult = (a, b)",
                 require=["whiten"], tol=1e-6)
    return P(f"{desc}. Whitening divides each component's scores by the square root of its variance. Fit `pca = "
             f"PCA(n_components={k}, whiten=True)` on `X` and rebuild `pca.transform(X[:4])` by hand from `mean_`, "
             f"`components_` and `explained_variance_`.", setup,
             f"from sklearn.decomposition import PCA\npca = PCA(n_components={k}, whiten=True).fit(X)\n"
             "result = (X[:4] - pca.mean_) @ pca.components_.T / np.sqrt(pca.explained_variance_)",
             require=["whiten", "explained_variance_"], tol=1e-6)


@template(M, "easy")
def scaling_matters_pca(rng):
    t = table(rng, numeric_only=True, n=(100, 250))
    task = rng.choice(["top_raw", "ratio", "both"])
    head = f"{t.desc}. Its columns have very different units, and PCA chases variance."
    sc = "from sklearn.decomposition import PCA\nfrom sklearn.preprocessing import StandardScaler\n"
    if task == "top_raw":
        return P(f"{head} Fit `PCA(n_components=1)` on the raw `df` and return the name of the column with the largest "
                 f"absolute loading in `components_[0]` and `df.var().idxmax()` as a tuple (the same column).", t.setup,
                 "from sklearn.decomposition import PCA\nc = PCA(n_components=1).fit(df).components_[0]\n"
                 "result = (df.columns[np.abs(c).argmax()], df.var().idxmax())", require=["components_"])
    if task == "ratio":
        return P(f"{head} Return the first component's explained variance ratio on the raw `df` and on `df` "
                 f"standardized with `StandardScaler`, as a tuple.", t.setup,
                 sc + "a = PCA().fit(df).explained_variance_ratio_[0]\n"
                      "b = PCA().fit(StandardScaler().fit_transform(df)).explained_variance_ratio_[0]\nresult = (a, b)",
                 require=["StandardScaler", "explained_variance_ratio_"], tol=1e-6)
    return P(f"{head} Fit `PCA(n_components=1)` inside `make_pipeline(StandardScaler(), PCA(n_components=1))` on `df` "
             f"and return a dict {{column: loading}} of the first component (`pipe[-1].components_[0]`).", t.setup,
             sc + "from sklearn.pipeline import make_pipeline\npipe = make_pipeline(StandardScaler(), PCA(n_components=1)).fit(df)\n"
                  "result = dict(zip(df.columns, pipe[-1].components_[0]))", require=["make_pipeline", "components_"],
             tol=1e-6)


@template(M, "easy")
def lda_vs_pca(rng):
    name = rng.choice(["iris", "wine", "digits"])
    if name == "digits":
        setup, desc, n, d = _digits(rng)
        k = 10
    else:
        fn = {"iris": "load_iris", "wine": "load_wine"}[name]
        setup = (f"from sklearn.datasets import {fn}\nfrom sklearn.preprocessing import StandardScaler\n"
                 f"X, y = {fn}(return_X_y=True)\nX = StandardScaler().fit_transform(X)")
        desc = f"`X` holds the {name} measurements, standardized, and `y` the class (0, 1, 2)"
        k = 3
    task = rng.choice(["knn", "shape", "ratio"])
    if task == "knn":
        return P(f"{desc}. Project `X` to 2-D with `PCA(n_components=2)` and with `LinearDiscriminantAnalysis("
                 f"n_components=2)` (from `sklearn.discriminant_analysis`; it uses `y`), then return the 5-fold "
                 f"`cross_val_score` mean of `KNeighborsClassifier(n_neighbors=5)` on each projection, as `(pca, "
                 f"lda)`. (Projecting before cross-validation leaks label information into LDA; here it just "
                 f"illustrates the separation.)", setup,
                 "from sklearn.decomposition import PCA\nfrom sklearn.discriminant_analysis import LinearDiscriminantAnalysis\n"
                 "from sklearn.model_selection import cross_val_score\nfrom sklearn.neighbors import KNeighborsClassifier\n"
                 "Zp = PCA(n_components=2).fit_transform(X)\nZl = LinearDiscriminantAnalysis(n_components=2).fit_transform(X, y)\n"
                 "knn = KNeighborsClassifier(n_neighbors=5)\n"
                 "result = (cross_val_score(knn, Zp, y, cv=5).mean(), cross_val_score(knn, Zl, y, cv=5).mean())",
                 require=["LinearDiscriminantAnalysis", "PCA"], tol=1e-9)
    if task == "shape":
        return P(f"{desc}. LDA can produce at most (number of classes − 1) components. Fit "
                 f"`LinearDiscriminantAnalysis()` (no `n_components`) on `X, y` and return `(transform(X).shape, "
                 f"number of classes − 1)`.", setup,
                 "from sklearn.discriminant_analysis import LinearDiscriminantAnalysis\n"
                 "Z = LinearDiscriminantAnalysis().fit(X, y).transform(X)\nresult = (Z.shape, len(np.unique(y)) - 1)",
                 require=["LinearDiscriminantAnalysis"])
    return P(f"{desc}. Return the `explained_variance_ratio_` of `LinearDiscriminantAnalysis()` fitted on `X, y` (the "
             f"share of between-class separation each discriminant carries; {k - 1} of them).", setup,
             "from sklearn.discriminant_analysis import LinearDiscriminantAnalysis\n"
             "result = LinearDiscriminantAnalysis().fit(X, y).explained_variance_ratio_",
             require=["explained_variance_ratio_"], tol=1e-6)


@template(M, "easy")
def truncated_svd_text(rng):
    setup, desc, n = _counts(rng)
    k = rng.randint(2, 4)
    task = rng.choice(["ratio", "sparse", "no_center"])
    setup += "\ndocs = [' '.join(np.repeat(words, c)) for c in counts]"
    desc += "; `docs` holds the same documents as strings"
    if task == "ratio":
        return P(f"{desc}. Latent semantic analysis: turn `docs` into a sparse TF-IDF matrix with `TfidfVectorizer()` "
                 f"(from `sklearn.feature_extraction.text`), fit `TruncatedSVD(n_components={k}, random_state=0)` on it "
                 f"and return its `explained_variance_ratio_`.", setup,
                 "from sklearn.feature_extraction.text import TfidfVectorizer\nfrom sklearn.decomposition import TruncatedSVD\n"
                 f"T = TfidfVectorizer().fit_transform(docs)\nresult = TruncatedSVD(n_components={k}, random_state=0).fit(T)"
                 ".explained_variance_ratio_", require=["TruncatedSVD", "TfidfVectorizer"], tol=1e-5)
    if task == "sparse":
        return P(f"{desc}. Build the sparse TF-IDF matrix `T = TfidfVectorizer().fit_transform(docs)`, then return "
                 f"`(type(T).__name__, TruncatedSVD(n_components={k}, random_state=0).fit_transform(T).shape)`: "
                 f"`TruncatedSVD` takes the sparse matrix as it is.", setup,
                 "from sklearn.feature_extraction.text import TfidfVectorizer\nfrom sklearn.decomposition import TruncatedSVD\n"
                 f"T = TfidfVectorizer().fit_transform(docs)\n"
                 f"result = (type(T).__name__, TruncatedSVD(n_components={k}, random_state=0).fit_transform(T).shape)",
                 require=["TruncatedSVD"])
    return P(f"{desc}. `TruncatedSVD` does not centre the data, PCA does. On the dense `counts.astype(float)`, return the "
             f"first explained variance ratio of `TruncatedSVD(n_components={k}, random_state=0)` and of "
             f"`PCA(n_components={k})`, and the same `TruncatedSVD` ratio after subtracting the column means first, as a "
             f"3-tuple (the last one matches PCA's).", setup,
             "from sklearn.decomposition import TruncatedSVD, PCA\nC = counts.astype(float)\n"
             f"a = TruncatedSVD(n_components={k}, random_state=0).fit(C).explained_variance_ratio_[0]\n"
             f"b = PCA(n_components={k}).fit(C).explained_variance_ratio_[0]\n"
             f"c = TruncatedSVD(n_components={k}, random_state=0).fit(C - C.mean(axis=0)).explained_variance_ratio_[0]\n"
             "result = (a, b, c)", require=["TruncatedSVD", "PCA"], tol=1e-4)


@template(M, "easy")
def incremental_pca_batches(rng):
    setup, desc, n, d = rng.choice([_digits, _lowrank])(rng)
    k, b = rng.randint(2, 5), rng.choice([50, 75, 100])
    task = rng.choice(["compare", "partial_fit", "seen"])
    if task == "compare":
        return P(f"{desc}. Fit `PCA(n_components={k})` and `IncrementalPCA(n_components={k}, batch_size={b})` on `X` and "
                 f"return both `explained_variance_ratio_` arrays as a tuple: the incremental one is very close.", setup,
                 "from sklearn.decomposition import PCA, IncrementalPCA\n"
                 f"result = (PCA(n_components={k}).fit(X).explained_variance_ratio_, "
                 f"IncrementalPCA(n_components={k}, batch_size={b}).fit(X).explained_variance_ratio_)",
                 require=["IncrementalPCA"], tol=1e-6)
    if task == "partial_fit":
        return P(f"{desc}. Pretend `X` arrives in chunks of {b} rows: create `ipca = IncrementalPCA(n_components={k})`, "
                 f"call `ipca.partial_fit(X[i:i + {b}])` for every chunk, then return `ipca.explained_variance_`.", setup,
                 f"from sklearn.decomposition import IncrementalPCA\nipca = IncrementalPCA(n_components={k})\n"
                 f"for i in range(0, len(X), {b}):\n    ipca.partial_fit(X[i:i + {b}])\nresult = ipca.explained_variance_",
                 require=["partial_fit"], tol=1e-6)
    return P(f"{desc}. Fit `ipca = IncrementalPCA(n_components={k}, batch_size={b})` on `X` and return "
             f"`(ipca.n_samples_seen_, ipca.mean_[:3])`.", setup,
             f"from sklearn.decomposition import IncrementalPCA\nipca = IncrementalPCA(n_components={k}, batch_size={b}).fit(X)\n"
             "result = (int(ipca.n_samples_seen_), ipca.mean_[:3])", require=["n_samples_seen_"], tol=1e-9)


@template(M, "easy")
def nmf_parts(rng):
    setup, desc, n, d = _digits(rng, n=(200, 350))
    k = rng.randint(4, 10)
    task = rng.choice(["nonneg", "error", "top_pixels"])
    fit = (f"from sklearn.decomposition import NMF\nnmf = NMF(n_components={k}, init='nndsvda', max_iter=500, random_state=0)\n"
           "W = nmf.fit_transform(X)\nH = nmf.components_\n")
    head = (f"{desc}. Factor the images with `nmf = NMF(n_components={k}, init='nndsvda', max_iter=500, random_state=0)`: "
            f"`W = nmf.fit_transform(X)` (one weight per part and image) and `H = nmf.components_` (the parts).")
    if task == "nonneg":
        return P(f"{head} Return `(W.shape, H.shape, W.min() >= 0 and H.min() >= 0)`: every factor is non-negative.",
                 setup, fit + "result = (W.shape, H.shape, bool(W.min() >= 0 and H.min() >= 0))",
                 require=["NMF", "components_"])
    if task == "error":
        return P(f"{head} Return `(np.linalg.norm(X - W @ H), nmf.reconstruction_err_)`: the Frobenius norm of the "
                 f"residual, computed two ways.", setup, fit + "result = (np.linalg.norm(X - W @ H), nmf.reconstruction_err_)",
                 require=["reconstruction_err_"], tol=1e-6)
    return P(f"{head} Each part lights up a few pixels. For every part (row of `H`), return the index of its brightest "
             f"pixel, as an array.", setup, fit + "result = H.argmax(axis=1)", require=["components_"])


@template(M, "easy")
def pca_tour(rng):
    setup, desc, n, d = rng.choice([_lowrank, _scaled])(rng)
    k = rng.randint(2, 3)
    s = seeded(rng)
    items = [
        ("components_", "Return the index of the feature with the largest absolute loading on the first component.",
         "result = int(np.abs(pca.components_[0]).argmax())"),
        ("components_", "Principal components are orthonormal: return `pca.components_ @ pca.components_.T` (the "
                        f"identity, {k} × {k}).", "result = pca.components_ @ pca.components_.T"),
        ("transform", "The scores on different components are uncorrelated and their variances are "
                      "`explained_variance_`: return the variances (`ddof=1`) of the columns of `pca.transform(X)`.",
         "result = pca.transform(X).var(axis=0, ddof=1)"),
        ("transform", f"Project a new point `x_new` (given) and return its {k} coordinates.",
         "result = pca.transform(x_new.reshape(1, -1))[0]"),
        ("inverse_transform", "Map the point with coordinates (1, 0, ...) in PCA space back to the original space and "
                              "return it (it equals `mean_ + components_[0]`).",
         f"z = np.zeros((1, {k}))\nz[0, 0] = 1\nresult = pca.inverse_transform(z)[0]"),
        ("explained_variance_ratio_", "What share of the variance is **lost** by keeping only these components? Return "
                                      "1 minus the sum of `explained_variance_ratio_`.",
         "result = 1 - pca.explained_variance_ratio_.sum()"),
    ]
    return catalog(rng, items, setup + f"\nfrom sklearn.decomposition import PCA\npca = PCA(n_components={k}).fit(X)\n"
                                       f"x_new = np.random.default_rng({s}).normal(0, 1, X.shape[1])",
                   f"{desc}; `pca = PCA(n_components={k})` has been fitted on `X`, and `x_new` is a new point.", tol=1e-6)


# ================================================================= medium

@template(M, "medium")
def pca_then_classifier(rng):
    setup, desc, n, d = rng.choice([_digits, _scaled])(rng)
    ks = sorted(rng.sample([2, 5, 10, 20], 3)) if d > 20 else sorted(rng.sample([2, 4, 6, 8], 3))
    t = rng.choice([0.8, 0.9, 0.95])
    task = rng.choice(["sweep", "best", "float", "vs_raw"])
    imp = ("from sklearn.decomposition import PCA\nfrom sklearn.linear_model import LogisticRegression\n"
           "from sklearn.model_selection import cross_val_score\nfrom sklearn.pipeline import make_pipeline\n"
           "from sklearn.preprocessing import StandardScaler\n")
    pipe = "make_pipeline(StandardScaler(), PCA(n_components={}), LogisticRegression(max_iter=2000))"
    if task in ("sweep", "best"):
        what = ("the list of mean accuracies" if task == "sweep" else
                "the `n_components` with the best mean accuracy (the smallest on ties)")
        last = "result = scores" if task == "sweep" else f"result = {ks}[int(np.argmax(scores))]"
        return P(f"{desc}. For `n_components` in `{ks}`, score `{pipe.format('k')}` with 5-fold `cross_val_score` on "
                 f"`X, y` and return {what}.", setup,
                 imp + f"scores = [cross_val_score({pipe.format('k')}, X, y, cv=5).mean() for k in {ks}]\n" + last,
                 require=["make_pipeline", "PCA"], tol=1e-9)
    if task == "float":
        return P(f"{desc}. Fit `pipe = {pipe.format(t)}` on `X, y` and return `(pipe[1].n_components_, training "
                 f"accuracy)`: the PCA step keeps enough components for {t:.0%} of the (standardized) variance.", setup,
                 imp + f"pipe = {pipe.format(t)}.fit(X, y)\nresult = (pipe[1].n_components_, pipe.score(X, y))",
                 require=["n_components_"], tol=1e-9)
    k = ks[0]
    return P(f"{desc}. Does compressing hurt? Return the 5-fold `cross_val_score` mean of `{pipe.format(k)}` and of "
             f"`make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))`, as `(with_pca, without)`.", setup,
             imp + f"a = cross_val_score({pipe.format(k)}, X, y, cv=5).mean()\n"
                   "b = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), X, y, cv=5).mean()\n"
                   "result = (a, b)", require=["PCA", "cross_val_score"], tol=1e-9)


@template(M, "medium")
def kernel_pca_circles(rng):
    n, s, nz = rng.randint(120, 200), seeded(rng), rng.choice([0.03, 0.05])
    setup = (f"from sklearn.datasets import make_circles\nX, y = make_circles(n_samples={n}, factor=0.3, noise={nz}, "
             f"random_state={s})")
    desc = f"`X` holds {n} 2-D points on two concentric circles (`make_circles`, `factor=0.3`), and `y` says which circle"
    gamma = rng.choice([2, 5, 10])
    task = rng.choice(["first_component", "gammas", "inverse", "kernels"])
    acc = ("def acc1(Z):\n    from sklearn.linear_model import LogisticRegression\n"
           "    return LogisticRegression().fit(Z[:, :1], y).score(Z[:, :1], y)\n")
    if task == "first_component":
        return P(f"{desc}. No straight line separates the circles, but an RBF kernel can unfold them. Project `X` with "
                 f"`PCA(n_components=2)` and with `KernelPCA(n_components=2, kernel='rbf', gamma={gamma})`, then fit "
                 f"`LogisticRegression()` on the **first** projected coordinate alone and return its training accuracy "
                 f"for each, as `(pca, kernel_pca)`.", setup,
                 "from sklearn.decomposition import PCA, KernelPCA\n" + acc +
                 f"result = (acc1(PCA(n_components=2).fit_transform(X)), "
                 f"acc1(KernelPCA(n_components=2, kernel='rbf', gamma={gamma}).fit_transform(X)))",
                 require=["KernelPCA", "kernel"], tol=1e-9)
    if task == "gammas":
        gs = [0.1, 1, 10]
        return P(f"{desc}. For `gamma` in `{gs}`, project `X` with `KernelPCA(n_components=2, kernel='rbf', gamma=g)`, "
                 f"fit `LogisticRegression()` on the first projected coordinate alone and return the list of training "
                 f"accuracies.", setup,
                 "from sklearn.decomposition import KernelPCA\n" + acc +
                 f"result = [acc1(KernelPCA(n_components=2, kernel='rbf', gamma=g).fit_transform(X)) for g in {gs}]",
                 require=["KernelPCA", "gamma"], tol=1e-9)
    if task == "inverse":
        return P(f"{desc}. Kernel PCA can only map back if it learns an approximate inverse: fit `kp = KernelPCA("
                 f"n_components=2, kernel='rbf', gamma={gamma}, fit_inverse_transform=True)` on `X` and return the mean "
                 f"squared error between `X` and `kp.inverse_transform(kp.transform(X))`.", setup,
                 f"from sklearn.decomposition import KernelPCA\nkp = KernelPCA(n_components=2, kernel='rbf', gamma={gamma}, "
                 "fit_inverse_transform=True).fit(X)\nresult = float(((kp.inverse_transform(kp.transform(X)) - X) ** 2).mean())",
                 require=["fit_inverse_transform", "inverse_transform"], tol=1e-6)
    return P(f"{desc}. Compare kernels: for `kernel` in `['linear', 'poly', 'rbf']`, project `X` with `KernelPCA("
             f"n_components=2, kernel=k, gamma={gamma}, degree=2)`, fit `LogisticRegression()` on the first projected "
             f"coordinate and return a dict {{kernel: training accuracy}}.", setup,
             "from sklearn.decomposition import KernelPCA\n" + acc +
             f"result = {{k: acc1(KernelPCA(n_components=2, kernel=k, gamma={gamma}, degree=2).fit_transform(X)) "
             "for k in ['linear', 'poly', 'rbf']}", require=["KernelPCA", "kernel"], tol=1e-9)


@template(M, "medium")
def tsne_vs_pca_digits(rng):
    setup, desc, n, d = _digits(rng, n=(120, 160))
    p = rng.choice([10, 20, 30])
    ts = f"TSNE(n_components=2, perplexity={p}, init='pca', learning_rate='auto', max_iter=300, random_state=0)"
    task = rng.choice(["knn", "trust", "kl", "spread"])
    imp = "from sklearn.decomposition import PCA\nfrom sklearn.manifold import TSNE\n"
    emb = f"Zp = PCA(n_components=2).fit_transform(X)\nZt = {ts}.fit_transform(X)\n"
    head = f"{desc}. Embed `X` in 2-D with `PCA(n_components=2)` and with `{ts}` (from `sklearn.manifold`)."
    if task == "knn":
        return P(f"{head} How well do the 2-D maps keep digits of a kind together? Return the 5-fold `cross_val_score` "
                 f"mean of `KNeighborsClassifier(n_neighbors=5)` on each embedding, as `(pca, tsne)`.", setup,
                 imp + "from sklearn.model_selection import cross_val_score\nfrom sklearn.neighbors import KNeighborsClassifier\n"
                 + emb + "knn = KNeighborsClassifier(n_neighbors=5)\n"
                         "result = (cross_val_score(knn, Zp, y, cv=5).mean(), cross_val_score(knn, Zt, y, cv=5).mean())",
                 require=["TSNE", "PCA"], tol=1e-9)
    if task == "trust":
        return P(f"{head} Return `trustworthiness(X, Z, n_neighbors=5)` (from `sklearn.manifold`; 1 means the "
                 f"embedding invents no false neighbours) for each embedding, as `(pca, tsne)`.", setup,
                 imp + "from sklearn.manifold import trustworthiness\n" + emb +
                 "result = (trustworthiness(X, Zp, n_neighbors=5), trustworthiness(X, Zt, n_neighbors=5))",
                 require=["TSNE", "trustworthiness"], tol=1e-9)
    if task == "kl":
        ps = [5, 15, 40]
        return P(f"{desc}. For `perplexity` in `{ps}`, fit `TSNE(n_components=2, perplexity=p, init='pca', "
                 f"learning_rate='auto', max_iter=300, random_state=0)` on `X` and return the list of final "
                 f"`kl_divergence_` values.", setup,
                 "from sklearn.manifold import TSNE\nresult = [TSNE(n_components=2, perplexity=p, init='pca', "
                 f"learning_rate='auto', max_iter=300, random_state=0).fit(X).kl_divergence_ for p in {ps}]",
                 require=["perplexity", "kl_divergence_"], tol=1e-4)
    return P(f"{head} For each embedding, compute the ratio of the mean distance between the class centroids (mean "
             f"position of each digit's points) to the mean distance of points to their own class centroid. Return "
             f"`(pca_ratio, tsne_ratio)`: bigger means better separated clusters.", setup,
             imp + emb + "def ratio(Z):\n    C = np.array([Z[y == c].mean(axis=0) for c in np.unique(y)])\n"
                          "    between = np.linalg.norm(C[:, None] - C[None], axis=2)[np.triu_indices(len(C), 1)].mean()\n"
                          "    within = np.mean([np.linalg.norm(Z[y == c] - Z[y == c].mean(axis=0), axis=1).mean() for c in np.unique(y)])\n"
                          "    return between / within\nresult = (ratio(Zp), ratio(Zt))",
             require=["TSNE", "PCA"], tol=1e-6)


@template(M, "medium")
def manifold_unrolling(rng):
    setup, desc, n = _scurve(rng)
    nn = rng.choice([8, 10, 12])
    task = rng.choice(["isomap_vs_pca", "all", "lle", "dist"])
    corr = ("def unrolled(Z):\n    return max(abs(np.corrcoef(Z[:, j], t)[0, 1]) for j in range(Z.shape[1]))\n")
    head = (f"{desc}. A good 2-D embedding *unrolls* the sheet, so one of its coordinates tracks `t`. Score an embedding "
            f"`Z` by the largest absolute correlation (`np.corrcoef`) between a column of `Z` and `t`.")
    if task == "isomap_vs_pca":
        return P(f"{head} Return the score of `PCA(n_components=2)` and of `Isomap(n_neighbors={nn}, n_components=2)`, as "
                 f"a tuple.", setup,
                 "from sklearn.decomposition import PCA\nfrom sklearn.manifold import Isomap\n" + corr +
                 f"result = (unrolled(PCA(n_components=2).fit_transform(X)), unrolled(Isomap(n_neighbors={nn}, "
                 f"n_components=2).fit_transform(X)))", require=["Isomap", "PCA"], tol=1e-6)
    if task == "all":
        return P(f"{head} Return a dict of scores for `'pca'`: `PCA(n_components=2)`, `'isomap'`: `Isomap(n_neighbors={nn}, "
                 f"n_components=2)`, `'lle'`: `LocallyLinearEmbedding(n_neighbors={nn}, n_components=2, random_state=0)` "
                 f"and `'spectral'`: `SpectralEmbedding(n_components=2, random_state=0)`.", setup,
                 "from sklearn.decomposition import PCA\n"
                 "from sklearn.manifold import Isomap, LocallyLinearEmbedding, SpectralEmbedding\n" + corr +
                 f"models = {{'pca': PCA(n_components=2), 'isomap': Isomap(n_neighbors={nn}, n_components=2),\n"
                 f"          'lle': LocallyLinearEmbedding(n_neighbors={nn}, n_components=2, random_state=0),\n"
                 "          'spectral': SpectralEmbedding(n_components=2, random_state=0)}\n"
                 "result = {k: unrolled(m.fit_transform(X)) for k, m in models.items()}",
                 require=["Isomap", "LocallyLinearEmbedding", "SpectralEmbedding"], tol=1e-6)
    if task == "lle":
        nns = [5, 10, 20]
        return P(f"{head} For `n_neighbors` in `{nns}`, embed `X` with `LocallyLinearEmbedding(n_neighbors=k, "
                 f"n_components=2, random_state=0)` and return the list of scores.", setup,
                 "from sklearn.manifold import LocallyLinearEmbedding\n" + corr +
                 f"result = [unrolled(LocallyLinearEmbedding(n_neighbors=k, n_components=2, random_state=0).fit_transform(X)) "
                 f"for k in {nns}]", require=["LocallyLinearEmbedding", "n_neighbors"], tol=1e-5)
    return P(f"{desc}. Isomap replaces straight-line distances by distances along the sheet (shortest paths in a "
             f"{nn}-nearest-neighbour graph). Fit `iso = Isomap(n_neighbors={nn}, n_components=2)` on `X` and, for the "
             f"two points with the smallest and the largest `t`, return `(straight-line distance, "
             f"iso.dist_matrix_[i, j])`.", setup,
             f"from sklearn.manifold import Isomap\niso = Isomap(n_neighbors={nn}, n_components=2).fit(X)\n"
             "i, j = int(np.argmin(t)), int(np.argmax(t))\nresult = (float(np.linalg.norm(X[i] - X[j])), iso.dist_matrix_[i, j])",
             require=["dist_matrix_"], tol=1e-6)


@template(M, "medium")
def topic_model_counts(rng):
    setup, desc, n = _counts(rng)
    k = 3
    task = rng.choice(["top_words", "doc_topics", "dominant", "nmf"])
    lda = (f"from sklearn.decomposition import LatentDirichletAllocation\n"
           f"lda = LatentDirichletAllocation(n_components={k}, max_iter=10, random_state=0).fit(counts)\n")
    head = (f"{desc}. Fit `lda = LatentDirichletAllocation(n_components={k}, max_iter=10, random_state=0)` (from "
            f"`sklearn.decomposition`; a topic model for count data) on `counts`.")
    if task == "top_words":
        return P(f"{head} `lda.components_` scores every word for every topic. Return, for each topic, its 4 "
                 f"highest-scoring words (highest first), as a list of lists.", setup,
                 lda + "result = [[words[i] for i in np.argsort(row)[::-1][:4]] for row in lda.components_]",
                 require=["LatentDirichletAllocation", "components_"])
    if task == "doc_topics":
        return P(f"{head} `lda.transform(counts)` gives each document's topic mixture. Return the mixtures of the first "
                 f"3 documents and the largest |row sum − 1| over all documents, as a tuple.", setup,
                 lda + "D = lda.transform(counts)\nresult = (D[:3], float(np.abs(D.sum(axis=1) - 1).max()))",
                 require=["LatentDirichletAllocation", "transform"], tol=1e-5)
    if task == "dominant":
        return P(f"{head} Give every document its dominant topic (argmax of `lda.transform(counts)`) and return "
                 f"`adjusted_rand_score(topic, dominant)` (from `sklearn.metrics`): how well the model recovered the "
                 f"hidden topics, whatever their numbering.", setup,
                 lda + "from sklearn.metrics import adjusted_rand_score\n"
                       "result = adjusted_rand_score(topic, lda.transform(counts).argmax(axis=1))",
                 require=["LatentDirichletAllocation", "adjusted_rand_score"], tol=1e-9)
    return P(f"{desc}. NMF finds topics too. Fit `nmf = NMF(n_components=3, init='nndsvda', max_iter=500, "
             f"random_state=0)` on `counts.astype(float)`, give every document its dominant part (argmax of "
             f"`fit_transform`) and return `(adjusted_rand_score(topic, dominant), top word of each part)` (the top "
             f"word is the argmax of each row of `components_`).", setup,
             "from sklearn.decomposition import NMF\nfrom sklearn.metrics import adjusted_rand_score\n"
             "nmf = NMF(n_components=3, init='nndsvda', max_iter=500, random_state=0)\nW = nmf.fit_transform(counts.astype(float))\n"
             "result = (adjusted_rand_score(topic, W.argmax(axis=1)), [words[i] for i in nmf.components_.argmax(axis=1)])",
             require=["NMF", "components_"], tol=1e-9)


@template(M, "medium")
def ica_unmixing(rng):
    n, s = rng.randint(300, 500), seeded(rng)
    setup = (f"g = np.random.default_rng({s})\nt = np.linspace(0, 8, {n})\n"
             "S = np.column_stack([np.sin(2 * t), np.sign(np.sin(3 * t)), (t % 1.5) - 0.75])   # sine, square, sawtooth\n"
             "A = g.uniform(0.5, 2, (3, 3))   # unknown mixing matrix\nX = S @ A.T + g.normal(0, 0.02, (len(t), 3))")
    desc = (f"Three sources (a sine, a square wave and a sawtooth, columns of `S`) were recorded by 3 microphones as "
            f"mixtures: `X = S @ A.T` plus a little noise ({n} time steps)")
    task = rng.choice(["recover", "pca", "mixing"])
    corr = ("def match(Z):\n    C = np.abs(np.corrcoef(S.T, Z.T)[:3, 3:])\n    return C.max(axis=1)\n")
    ica = "FastICA(n_components=3, whiten='unit-variance', max_iter=1000, random_state=0)"
    if task == "recover":
        return P(f"{desc}. Unmix them with `{ica}` (`fit_transform(X)`). ICA recovers sources only up to order, sign "
                 f"and scale, so score it by, for each true source, the largest absolute correlation with any recovered "
                 f"component. Return those 3 scores (close to 1).", setup,
                 "from sklearn.decomposition import FastICA\n" + corr + f"result = match({ica}.fit_transform(X))",
                 require=["FastICA"], tol=1e-6)
    if task == "pca":
        return P(f"{desc}. Score an unmixing `Z` by, for each true source, the largest absolute correlation with any "
                 f"column of `Z`. Return the scores of `{ica}.fit_transform(X)` and of `PCA(n_components=3)"
                 f".fit_transform(X)`, as a tuple of arrays: PCA only decorrelates, it doesn't unmix.", setup,
                 "from sklearn.decomposition import FastICA, PCA\n" + corr +
                 f"result = (match({ica}.fit_transform(X)), match(PCA(n_components=3).fit_transform(X)))",
                 require=["FastICA", "PCA"], tol=1e-6)
    return P(f"{desc}. Fit `ica = {ica}` on `X`. Its `mixing_` matrix maps sources back to microphones: return the "
             f"largest absolute difference between `X` and `ica.transform(X) @ ica.mixing_.T + ica.mean_` (tiny: the "
             f"model reproduces the recordings) and `ica.mixing_.shape`, as a tuple.", setup,
             f"from sklearn.decomposition import FastICA\nica = {ica}.fit(X)\n"
             "result = (float(np.abs(ica.transform(X) @ ica.mixing_.T + ica.mean_ - X).max()) < 1e-6, ica.mixing_.shape)",
             require=["mixing_"])


@template(M, "medium")
def factor_analysis_noise(rng):
    n, d, s = rng.randint(200, 400), rng.randint(5, 8), seeded(rng)
    setup = (f"g = np.random.default_rng({s})\nnoise_sd = g.uniform(0.1, 1.5, {d})\n"
             f"X = g.normal(0, 1, ({n}, 2)) @ g.normal(0, 1, (2, {d})) + g.normal(0, 1, ({n}, {d})) * noise_sd")
    desc = (f"`X` ({n} × {d}) comes from 2 hidden factors plus noise whose standard deviation differs per column "
            f"(`noise_sd`)")
    task = rng.choice(["noisiest", "estimates", "choose"])
    fa = "from sklearn.decomposition import FactorAnalysis\nfa = FactorAnalysis(n_components=2, random_state=0).fit(X)\n"
    if task == "noisiest":
        return P(f"{desc}. Unlike PCA, factor analysis models a separate noise variance per feature. Fit "
                 f"`FactorAnalysis(n_components=2, random_state=0)` on `X` and return `(column with the largest "
                 f"noise_variance_, column with the largest noise_sd)`.", setup,
                 fa + "result = (int(fa.noise_variance_.argmax()), int(noise_sd.argmax()))", require=["noise_variance_"])
    if task == "estimates":
        return P(f"{desc}. Fit `FactorAnalysis(n_components=2, random_state=0)` on `X` and return the estimated noise "
                 f"standard deviations (`np.sqrt(noise_variance_)`) next to `noise_sd`, as a tuple.", setup,
                 fa + "result = (np.sqrt(fa.noise_variance_), noise_sd)", require=["noise_variance_"], tol=1e-5)
    return P(f"{desc}. Choose the number of factors by cross-validated log-likelihood: for `n_components` in 1 .. 4, "
             f"return the mean 5-fold `cross_val_score(FactorAnalysis(n_components=k, random_state=0), X, cv=5)` (its "
             f"`score` is the average log-likelihood) as a list.", setup,
             "from sklearn.decomposition import FactorAnalysis\nfrom sklearn.model_selection import cross_val_score\n"
             "result = [cross_val_score(FactorAnalysis(n_components=k, random_state=0), X, cv=5).mean() for k in range(1, 5)]",
             require=["FactorAnalysis", "cross_val_score"], tol=1e-5)


@template(M, "medium")
def denoise_with_pca(rng):
    setup, desc, n, d = _digits(rng)
    s, sd = seeded(rng), rng.choice([2, 3, 4])
    setup += f"\nnoisy = X + np.random.default_rng({s}).normal(0, {sd}, X.shape)"
    desc += f"; `noisy` is `X` plus Gaussian noise (σ = {sd})"
    task = rng.choice(["error", "sweep", "best"])
    ks = [5, 10, 20, 40, 64]
    head = (f"{desc}. Denoise by projecting onto a few principal components: fit `PCA(n_components=k)` on `noisy` and "
            f"map `noisy` into PCA space and back. Measure the mean squared error against the clean `X`.")
    if task == "error":
        k = rng.choice([10, 15, 20])
        return P(f"{head} Return `(error of noisy itself, error after denoising with k = {k})`.", setup,
                 f"from sklearn.decomposition import PCA\npca = PCA(n_components={k}).fit(noisy)\n"
                 "den = pca.inverse_transform(pca.transform(noisy))\n"
                 "result = (float(((noisy - X) ** 2).mean()), float(((den - X) ** 2).mean()))",
                 require=["inverse_transform"], tol=1e-6)
    sol = ("from sklearn.decomposition import PCA\nerrs = []\n"
           f"for k in {ks}:\n    pca = PCA(n_components=k).fit(noisy)\n"
           "    errs.append(float(((pca.inverse_transform(pca.transform(noisy)) - X) ** 2).mean()))\n")
    if task == "sweep":
        return P(f"{head} Return the list of errors for k in `{ks}` (k = 64 keeps everything, noise included).", setup,
                 sol + "result = errs", require=["inverse_transform"], tol=1e-6)
    return P(f"{head} Return the k in `{ks}` with the smallest error.", setup, sol + f"result = {ks}[int(np.argmin(errs))]",
             require=["inverse_transform"])


@template(M, "medium")
def pca_tasks_themed(rng):
    t = table(rng, numeric_only=True, n=(100, 250))
    pipe = ("from sklearn.decomposition import PCA\nfrom sklearn.pipeline import make_pipeline\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "pipe = make_pipeline(StandardScaler(), PCA(n_components=2)).fit(df)\npca = pipe[-1]\n")
    items = [
        ("components_", "Return the loadings as a DataFrame: one row per component (index `['PC1', 'PC2']`), one column "
                        "per feature of `df`.",
         pipe + "result = pd.DataFrame(pca.components_, index=['PC1', 'PC2'], columns=df.columns)"),
        ("transform", "Return the index labels of the 3 rows with the highest PC1 score, highest first.",
         pipe + "result = list(df.index[np.argsort(pipe.transform(df)[:, 0])[::-1][:3]])"),
        ("explained_variance_ratio_", "What share of the standardized variance do the two components keep together?",
         pipe + "result = pca.explained_variance_ratio_.sum()"),
        ("transform", f"Return the correlation (`np.corrcoef`) between the PC1 scores and the target `y` "
                      f"({t.target}).", pipe + "result = float(np.corrcoef(pipe.transform(df)[:, 0], y)[0, 1])"),
        ("components_", "Which feature contributes most to PC2? Return its name (largest absolute loading).",
         pipe + "result = df.columns[np.abs(pca.components_[1]).argmax()]"),
        ("inverse_transform", "Reconstruct `df` from its 2 PCA scores, back in original units (`pipe.inverse_transform` "
                              "undoes both steps), and return the mean absolute error per column as a Series.",
         pipe + "rec = pipe.inverse_transform(pipe.transform(df))\nresult = (df - rec).abs().mean()"),
        ("transform", "Return the 2-D scores of `new_row` (a dict; make a one-row DataFrame with `df`'s columns).",
         pipe + "result = pipe.transform(pd.DataFrame([new_row])[df.columns])[0]"),
    ]
    row = {c: 1 for c in t.num}
    return catalog(rng, items, t.setup + f"\nnew_row = {lit(row)}",
                   f"{t.desc}. Summarize the features with `pipe = make_pipeline(StandardScaler(), PCA(n_components=2))` "
                   f"fitted on `df`; `pca = pipe[-1]`. `new_row` is a dict of feature values.", tol=1e-6)


@template(M, "medium")
def svd_relation(rng):
    setup, desc, n, d = rng.choice([_lowrank, _scaled])(rng)
    k = rng.randint(2, 3)
    task = rng.choice(["singular", "components", "scores"])
    svd = "Xc = X - X.mean(axis=0)\nU, S_, Vt = np.linalg.svd(Xc, full_matrices=False)\n"
    head = (f"{desc}. PCA is an SVD of the centred data: with `Xc = X - X.mean(axis=0)` and `U, S, Vt = "
            f"np.linalg.svd(Xc, full_matrices=False)`, ")
    fit = f"from sklearn.decomposition import PCA\npca = PCA(n_components={k}).fit(X)\n"
    if task == "singular":
        return P(f"{head}`explained_variance_` = S² / (n − 1). Return `(S[:{k}] ** 2 / (n − 1), "
                 f"PCA(n_components={k}).fit(X).explained_variance_)`.", setup,
                 svd + fit + f"result = (S_[:{k}] ** 2 / (len(X) - 1), pca.explained_variance_)",
                 require=["svd", "explained_variance_"], tol=1e-6)
    if task == "components":
        return P(f"{head}the rows of `Vt` are the components, up to sign. Return the absolute values of the cosine "
                 f"similarity between each of the first {k} rows of `Vt` and the matching row of "
                 f"`PCA(n_components={k}).fit(X).components_` (all 1).", setup,
                 svd + fit + f"result = np.abs((Vt[:{k}] * pca.components_).sum(axis=1))",
                 require=["svd", "components_"], tol=1e-6)
    return P(f"{head}the scores `transform(X)` are `U * S` (up to the sign of each column). Return the largest absolute "
             f"difference between `np.abs(U[:, :{k}] * S[:{k}])` and `np.abs(PCA(n_components={k}).fit_transform(X))`, "
             f"compared with the largest score, as `(difference, largest |score|)`.", setup,
             svd + f"from sklearn.decomposition import PCA\nZ = PCA(n_components={k}).fit_transform(X)\n"
                   f"result = (float(np.abs(np.abs(U[:, :{k}] * S_[:{k}]) - np.abs(Z)).max()) < 1e-6, float(np.abs(Z).max()))",
             require=["svd", "fit_transform"], tol=1e-6)


@template(M, "medium")
def lsa_document_search(rng):
    setup, desc, n = _counts(rng)
    k = rng.randint(2, 4)
    i = rng.randint(0, 9)
    setup += "\ndocs = [' '.join(np.repeat(words, c)) for c in counts]"
    task = rng.choice(["nearest", "same_topic", "pipeline"])
    lsa = ("from sklearn.decomposition import TruncatedSVD\nfrom sklearn.feature_extraction.text import TfidfVectorizer\n"
           "from sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import Normalizer\n"
           f"lsa = make_pipeline(TfidfVectorizer(), TruncatedSVD(n_components={k}, random_state=0), Normalizer())\n"
           "Z = lsa.fit_transform(docs)\nS = Z @ Z.T\nnp.fill_diagonal(S, -np.inf)\n")
    head = (f"{desc}; `docs` holds the same documents as strings. Build a small semantic search: `lsa = "
            f"make_pipeline(TfidfVectorizer(), TruncatedSVD(n_components={k}, random_state=0), Normalizer())` (the "
            f"`Normalizer` from `sklearn.preprocessing` scales every row to length 1, so dot products of the rows of "
            f"`Z = lsa.fit_transform(docs)` are cosine similarities).")
    if task == "nearest":
        return P(f"{head} Return the indices of the 3 documents most similar to document {i} (excluding itself), most "
                 f"similar first.", setup, lsa + f"result = np.argsort(S[{i}])[::-1][:3]",
                 require=["TruncatedSVD", "Normalizer"])
    if task == "same_topic":
        return P(f"{head} For every document, find its most similar other document. Return the fraction of documents "
                 f"whose match has the same hidden `topic`.", setup,
                 lsa + "result = float((topic[S.argmax(axis=1)] == topic).mean())", require=["TruncatedSVD"], tol=1e-9)
    q = rng.choice(["goal team", "vote senate", "gene virus"])
    return P(f"{head} Fit it on `docs`, then embed the query `{q!r}` with `lsa.transform` and return the indices of the "
             f"3 documents with the highest cosine similarity to it, highest first.", setup,
             lsa + f"qv = lsa.transform([{q!r}])[0]\nresult = np.argsort(Z @ qv)[::-1][:3]",
             require=["TruncatedSVD", "transform"])


# ================================================================= hard

@template(M, "hard")
def pca_from_scratch(rng):
    setup, desc, n, d = rng.choice([_lowrank, _scaled])(rng)
    k = rng.randint(2, 3)
    task = rng.choice(["components", "scores", "ratio"])
    base = ("mu = X.mean(axis=0)\nw, V = np.linalg.eigh(np.cov(X - mu, rowvar=False))\norder = np.argsort(w)[::-1]\n"
            f"w, V = w[order][:{k}], V[:, order][:, :{k}].T\n"
            "V = V * np.sign(V[np.arange(len(V)), np.abs(V).argmax(axis=1)])[:, None]\n")
    head = (f"{desc}. Implement `PCA(n_components={k})` **from scratch** with NumPy: centre `X`, take the eigenvectors of "
            f"its covariance matrix (`np.linalg.eigh` on `np.cov(..., rowvar=False)`) for the {k} largest eigenvalues, "
            f"and fix each eigenvector's sign the way scikit-learn does: the entry with the largest absolute value must "
            f"be positive.")
    if task == "components":
        return P(f"{head} Return your {k} × d components; they equal `PCA(n_components={k}).fit(X).components_` exactly "
                 f"(not just up to sign).", setup, base + "result = V", require=["eigh"], tol=1e-6)
    if task == "scores":
        return P(f"{head} Return the scores of the first 5 rows (`(X[:5] − mean) @ components.T`); they equal "
                 f"`PCA(n_components={k}).fit_transform(X)[:5]`.", setup, base + "result = (X[:5] - mu) @ V.T",
                 require=["eigh"], tol=1e-6)
    return P(f"{head} Return `(your explained_variance_ratio_, PCA(n_components={k}).fit(X).explained_variance_ratio_)`: "
             f"the ratios divide by the **total** variance, not by the kept eigenvalues.", setup,
             base + f"from sklearn.decomposition import PCA\nratio = w / np.trace(np.cov(X, rowvar=False))\n"
                    f"result = (ratio, PCA(n_components={k}).fit(X).explained_variance_ratio_)",
             require=["eigh", "explained_variance_ratio_"], tol=1e-6)


@template(M, "hard")
def pca_leakage(rng):
    setup, desc, n, d = rng.choice([_digits, _scaled])(rng)
    k = rng.choice([5, 10]) if d > 20 else rng.choice([3, 5])
    task = rng.choice(["both", "fix", "test_transform"])
    leaky = (f"Z = PCA(n_components={k}).fit_transform(StandardScaler().fit_transform(X))\n"
             "score = cross_val_score(LogisticRegression(max_iter=2000), Z, y, cv=5).mean()")
    imp = ("from sklearn.decomposition import PCA\nfrom sklearn.linear_model import LogisticRegression\n"
           "from sklearn.model_selection import cross_val_score\nfrom sklearn.pipeline import make_pipeline\n"
           "from sklearn.preprocessing import StandardScaler\n")
    good = (f"pipe = make_pipeline(StandardScaler(), PCA(n_components={k}), LogisticRegression(max_iter=2000))\n"
            "fixed = cross_val_score(pipe, X, y, cv=5).mean()\n")
    if task == "both":
        return P(f"{desc}. This code fits the scaler and PCA on **all** rows before cross-validating, so every test fold "
                 f"helped shape the features:\n```\n{leaky}\n```\nPut the scaler and PCA in a pipeline with the "
                 f"classifier so they are refitted inside each fold, and return `(leaky_score, pipeline_score)`.", setup,
                 imp + leaky + "\n" + good + "result = (score, fixed)", require=["make_pipeline", "cross_val_score"],
                 tol=1e-9)
    if task == "fix":
        return P(f"{desc}. Fix the leakage in this code by moving the scaler and PCA into a pipeline with the classifier, "
                 f"and return the 5-fold score of the pipeline:\n```\n{leaky}\n```", setup, imp + good + "result = fixed",
                 require=["make_pipeline", "cross_val_score"], tol=1e-9)
    return P(f"{desc}. Rows `[:70%]` are the training split and the rest the test split. This code fits a **second** PCA "
             f"on the test rows, so the test coordinates live in a different space:\n```\n"
             f"Z_tr = PCA(n_components={k}).fit_transform(X_tr)\nZ_te = PCA(n_components={k}).fit_transform(X_te)\n```\n"
             f"Fit one PCA on the training rows, transform both splits with it, train `LogisticRegression(max_iter=2000)` "
             f"on `Z_tr` and return its test accuracy.", setup + "\nh = int(0.7 * len(X))\nX_tr, X_te, y_tr, y_te = X[:h], X[h:], y[:h], y[h:]",
             f"from sklearn.decomposition import PCA\nfrom sklearn.linear_model import LogisticRegression\n"
             f"pca = PCA(n_components={k}).fit(X_tr)\nZ_tr, Z_te = pca.transform(X_tr), pca.transform(X_te)\n"
             "result = LogisticRegression(max_iter=2000).fit(Z_tr, y_tr).score(Z_te, y_te)", require=["transform"],
             tol=1e-9)


@template(M, "hard")
def choose_components_cv(rng):
    setup, desc, n, d = rng.choice([_digits, _scaled])(rng)
    ks = [2, 5, 10, 20] if d > 20 else [2, 4, 6, 8]
    task = rng.choice(["grid", "table", "two_params"])
    imp = ("from sklearn.decomposition import PCA\nfrom sklearn.linear_model import LogisticRegression\n"
           "from sklearn.model_selection import GridSearchCV\nfrom sklearn.pipeline import Pipeline\n"
           "from sklearn.preprocessing import StandardScaler\n"
           "pipe = Pipeline([('scale', StandardScaler()), ('pca', PCA()), ('clf', LogisticRegression(max_iter=2000))])\n")
    head = (f"{desc}. Build `pipe = Pipeline([('scale', StandardScaler()), ('pca', PCA()), ('clf', "
            f"LogisticRegression(max_iter=2000))])` and tune it with `GridSearchCV(pipe, grid, cv=5)` (from "
            f"`sklearn.model_selection`).")
    if task == "grid":
        return P(f"{head} With `grid = {{'pca__n_components': {ks}}}`, return `(best_params_, best_score_)`.", setup,
                 imp + f"gs = GridSearchCV(pipe, {{'pca__n_components': {ks}}}, cv=5).fit(X, y)\n"
                       "result = (gs.best_params_, gs.best_score_)", require=["GridSearchCV", "PCA"], tol=1e-9)
    if task == "table":
        return P(f"{head} With `grid = {{'pca__n_components': {ks}}}`, return the list of mean test scores from "
                 f"`cv_results_['mean_test_score']`, in grid order.", setup,
                 imp + f"gs = GridSearchCV(pipe, {{'pca__n_components': {ks}}}, cv=5).fit(X, y)\n"
                       "result = list(gs.cv_results_['mean_test_score'])", require=["GridSearchCV", "cv_results_"],
                 tol=1e-9)
    cs = [0.1, 1.0]
    return P(f"{head} Tune two steps at once with `grid = {{'pca__n_components': {ks[:3]}, 'clf__C': {cs}}}` and return "
             f"`best_params_`.", setup,
             imp + f"gs = GridSearchCV(pipe, {{'pca__n_components': {ks[:3]}, 'clf__C': {cs}}}, cv=5).fit(X, y)\n"
                   "result = gs.best_params_", require=["GridSearchCV"])


@template(M, "hard")
def pca_anomaly_scores(rng):
    setup, desc, n, d = _lowrank(rng)
    m, s = rng.randint(3, 6), seeded(rng)
    k = int(desc.split("rank-")[1][0])
    setup += (f"\nbad = np.random.default_rng({s}).normal(0, 3, ({m}, X.shape[1]))\nX = np.vstack([X, bad])   # the last "
              f"{m} rows don't follow the pattern")
    desc += f"; {m} anomalous rows (pure noise) were appended at the end"
    task = rng.choice(["top", "recall", "threshold"])
    sol = (f"from sklearn.decomposition import PCA\npca = PCA(n_components={k}).fit(X)\n"
           "err = ((X - pca.inverse_transform(pca.transform(X))) ** 2).sum(axis=1)\n")
    head = (f"{desc}. Score every row by its PCA reconstruction error: fit `PCA(n_components={k})` on `X` and take the "
            f"sum of squared differences between each row and its reconstruction (`inverse_transform` of `transform`).")
    if task == "top":
        return P(f"{head} Return the indices of the {m} rows with the largest error, largest first.", setup,
                 sol + f"result = np.argsort(err)[::-1][:{m}]", require=["inverse_transform"])
    if task == "recall":
        return P(f"{head} Flag the {m} rows with the largest error and return how many of them are among the {m} "
                 f"appended rows.", setup,
                 sol + f"top = np.argsort(err)[::-1][:{m}]\nresult = int((top >= len(X) - {m}).sum())",
                 require=["inverse_transform"])
    return P(f"{head} Flag rows whose error exceeds 3 × the median error and return the sorted flagged indices.", setup,
             sol + "result = np.flatnonzero(err > 3 * np.median(err))", require=["inverse_transform"])


@template(M, "hard")
def fisher_lda_by_hand(rng):
    task = rng.choice(["ratio", "direction", "scatter"])
    if task == "direction":
        dd = synth_clf(rng, n=(150, 300), d=(3, 6), classes=2)
        return P(f"{dd.desc}. For two classes, Fisher's discriminant direction is `w = Sw⁻¹ (μ1 − μ0)`, where `Sw` is the "
                 f"pooled within-class covariance (`Σ_c (n_c / n) · cov_c`, with `np.cov(..., bias=True)`). Compute `w` "
                 f"by hand and return the absolute cosine similarity between `w` and "
                 f"`LinearDiscriminantAnalysis().fit(X, y).coef_[0]` (1: the same direction).", dd.setup,
                 "from sklearn.discriminant_analysis import LinearDiscriminantAnalysis\n"
                 "Sw = sum((y == c).mean() * np.cov(X[y == c], rowvar=False, bias=True) for c in (0, 1))\n"
                 "w = np.linalg.solve(Sw, X[y == 1].mean(axis=0) - X[y == 0].mean(axis=0))\n"
                 "c = LinearDiscriminantAnalysis().fit(X, y).coef_[0]\n"
                 "result = abs(w @ c) / (np.linalg.norm(w) * np.linalg.norm(c))", require=["coef_"], tol=1e-6)
    name = rng.choice(["iris", "wine"])
    fn = {"iris": "load_iris", "wine": "load_wine"}[name]
    setup = f"from sklearn.datasets import {fn}\nX, y = {fn}(return_X_y=True)"
    desc = f"`X` holds the {name} measurements and `y` the class (0, 1, 2)"
    base = ("Sw = sum((y == c).mean() * np.cov(X[y == c], rowvar=False, bias=True) for c in np.unique(y))\n"
            "St = np.cov(X, rowvar=False, bias=True)\nSb = St - Sw\n")
    if task == "scatter":
        return P(f"{desc}. LDA splits the total covariance `St` (`np.cov(X, rowvar=False, bias=True)`) into a within-class "
                 f"part `Sw = Σ_c (n_c / n) · cov_c` and a between-class part `Sb = St − Sw`. Compute `Sb` and check it "
                 f"equals `Σ_c (n_c / n) (μ_c − μ)(μ_c − μ)ᵀ`: return the largest absolute difference and `np.trace(Sb)`, "
                 f"as a tuple.", setup,
                 base + "mu = X.mean(axis=0)\nB = sum((y == c).mean() * np.outer(X[y == c].mean(axis=0) - mu, "
                        "X[y == c].mean(axis=0) - mu) for c in np.unique(y))\n"
                        "result = (float(np.abs(B - Sb).max()) < 1e-8, float(np.trace(Sb)))", require=["cov"], tol=1e-6)
    return P(f"{desc}. Reproduce `LinearDiscriminantAnalysis(solver='eigen').fit(X, y).explained_variance_ratio_` by "
             f"hand: with `Sw = Σ_c (n_c / n) · cov_c` (`np.cov(..., rowvar=False, bias=True)` per class), `St` the "
             f"biased covariance of all of `X` and `Sb = St − Sw`, take the eigenvalues of `Sw⁻¹ Sb` "
             f"(`np.linalg.eigvals`, real part), divide by their sum and return the 2 largest, largest first.", setup,
             base + "ev = np.real(np.linalg.eigvals(np.linalg.solve(Sw, Sb)))\nresult = np.sort(ev / ev.sum())[::-1][:2]",
             require=["eigvals"], tol=1e-6)


@template(M, "hard")
def kernel_pca_by_hand(rng):
    n, s = rng.randint(40, 80), seeded(rng)
    gamma = rng.choice([0.5, 1.0, 2.0])
    setup = (f"from sklearn.datasets import make_circles\nX, y = make_circles(n_samples={n}, factor=0.4, noise=0.05, "
             f"random_state={s})")
    desc = f"`X` holds {n} 2-D points on two concentric circles"
    task = rng.choice(["eigenvalues", "projection", "centering"])
    base = (f"K = np.exp(-{gamma} * ((X[:, None] - X[None]) ** 2).sum(axis=2))\nn = len(X)\nJ = np.ones((n, n)) / n\n"
            "Kc = K - J @ K - K @ J + J @ K @ J\n")
    head = (f"{desc}. Do kernel PCA **by hand** with the RBF kernel `K[i, j] = exp(−{gamma}·‖x_i − x_j‖²)`: centre it "
            f"in feature space (`Kc = K − 1K − K1 + 1K1`, where `1` is the n × n matrix filled with 1/n), then "
            f"eigendecompose `Kc` (`np.linalg.eigh`).")
    if task == "eigenvalues":
        return P(f"{head} Return the 2 largest eigenvalues; they equal `KernelPCA(n_components=2, kernel='rbf', "
                 f"gamma={gamma}).fit(X).eigenvalues_`.", setup, base + "result = np.sort(np.linalg.eigvalsh(Kc))[::-1][:2]",
                 require=["eigvalsh|eigh"], tol=1e-6)
    if task == "projection":
        return P(f"{head} The projection on component j is `v_j · sqrt(λ_j)` (eigenvector times the square root of its "
                 f"eigenvalue). Return the absolute values of the projections on the top 2 components (signs are "
                 f"arbitrary) for the first 5 points; they equal `np.abs(KernelPCA(n_components=2, kernel='rbf', "
                 f"gamma={gamma}).fit_transform(X)[:5])`.", setup,
                 base + "w, V = np.linalg.eigh(Kc)\nidx = np.argsort(w)[::-1][:2]\nresult = np.abs(V[:5, idx] * np.sqrt(w[idx]))",
                 require=["eigh"], tol=1e-6)
    return P(f"{desc}. Kernel centring must give a matrix whose rows and columns all sum to 0. Build `K[i, j] = "
             f"exp(−{gamma}·‖x_i − x_j‖²)` and centre it as `Kc = K − 1K − K1 + 1K1` (`1` = the n × n matrix of 1/n), "
             f"but compute it in O(n²) from the row means, column means and grand mean instead of matrix products. "
             f"Return `(largest |row sum| of Kc, Kc[0, :3])`.", setup,
             f"K = np.exp(-{gamma} * ((X[:, None] - X[None]) ** 2).sum(axis=2))\n"
             "Kc = K - K.mean(axis=0)[None, :] - K.mean(axis=1)[:, None] + K.mean()\n"
             "result = (float(np.abs(Kc.sum(axis=1)).max()) < 1e-9, Kc[0, :3])", require=["mean"], tol=1e-9)


@template(M, "hard")
def decomposition_bug_fixes(rng):
    setup, desc, n, d = _digits(rng, n=(150, 250))
    k = rng.randint(2, 5)
    items = [
        ("TSNE", "This crashes with an AttributeError: t-SNE has no `transform`, it can only embed the points it is "
                 "fitted on.\n```\ntsne = TSNE(n_components=2, init='pca', learning_rate='auto', max_iter=300, "
                 "random_state=0).fit(X[:100])\nZ_new = tsne.transform(X[100:])\n```\nEmbed all rows together with "
                 "`fit_transform(X)` (same settings) and return the embedding of rows 100 to 104.",
         "from sklearn.manifold import TSNE\nZ = TSNE(n_components=2, init='pca', learning_rate='auto', max_iter=300, "
         "random_state=0).fit_transform(X)\nresult = Z[100:105]", {"tol": 1e-4}),
        ("NMF", "This raises a ValueError because NMF needs non-negative input, and centring made some pixels "
                f"negative.\n```\nW = NMF(n_components={k}, init='nndsvda', max_iter=500, random_state=0).fit_transform(X "
                "- X.mean(axis=0))\n```\nFactor the raw (non-negative) `X` instead and return `reconstruction_err_`.",
         f"from sklearn.decomposition import NMF\nresult = NMF(n_components={k}, init='nndsvda', max_iter=500, "
         "random_state=0).fit(X).reconstruction_err_"),
        ("n_components", "This raises a ValueError: LDA can give at most (number of classes − 1) components.\n```\n"
                         "Z = LinearDiscriminantAnalysis(n_components=12).fit_transform(X, y)\n```\nAsk for the maximum "
                         "allowed instead and return `Z.shape`.",
         "from sklearn.discriminant_analysis import LinearDiscriminantAnalysis\n"
         "Z = LinearDiscriminantAnalysis(n_components=len(np.unique(y)) - 1).fit_transform(X, y)\nresult = Z.shape"),
        ("fit_inverse_transform", "This raises an error: `KernelPCA` only learns `inverse_transform` when asked.\n```\n"
                                  "kp = KernelPCA(n_components=5, kernel='rbf', gamma=1e-3).fit(X)\n"
                                  "X_back = kp.inverse_transform(kp.transform(X))\n```\nFix it and return the mean "
                                  "squared error between `X_back` and `X`.",
         "from sklearn.decomposition import KernelPCA\nkp = KernelPCA(n_components=5, kernel='rbf', gamma=1e-3, "
         "fit_inverse_transform=True).fit(X)\nX_back = kp.inverse_transform(kp.transform(X))\n"
         "result = float(((X_back - X) ** 2).mean())", {"tol": 1e-5}),
        ("transform", f"This projects new images with a PCA refitted on them, so their coordinates are not comparable "
                      f"with the old ones:\n```\npca = PCA(n_components={k}).fit(X[:100])\n"
                      f"Z_new = PCA(n_components={k}).fit_transform(X[100:])\n```\nUse the fitted `pca` to transform "
                      f"`X[100:]` and return the first 3 rows of `Z_new`.",
         f"from sklearn.decomposition import PCA\npca = PCA(n_components={k}).fit(X[:100])\nZ_new = pca.transform(X[100:])\n"
         "result = Z_new[:3]"),
        ("explained_variance_ratio_", "This reports the variance kept by 2 components as a share of the **kept** "
                                      "variance, which is always 1 when all you keep is divided by itself:\n```\n"
                                      "pca = PCA(n_components=2).fit(X)\n"
                                      "kept = pca.explained_variance_.sum() / pca.explained_variance_.sum()\n```\n"
                                      "Return the share of the **total** variance instead.",
         "from sklearn.decomposition import PCA\nresult = PCA(n_components=2).fit(X).explained_variance_ratio_.sum()"),
    ]
    return catalog(rng, items, setup, f"{desc}.", tol=1e-6)


@template(M, "hard")
def variance_puzzles(rng):
    setup, desc, n, d = _lowrank(rng)
    j = rng.randint(0, 2)
    f = rng.choice([10, 100])
    task = rng.choice(["scale", "duplicate", "constant"])
    if task == "scale":
        return P(f"{desc}. Multiply column {j} of a copy of `X` by {f} and return the first explained variance ratio of "
                 f"`PCA()` before and after, and the absolute loading of column {j} on the first component after, as a "
                 f"3-tuple: one loud column takes over PCA.", setup,
                 f"from sklearn.decomposition import PCA\nX2 = X.copy()\nX2[:, {j}] *= {f}\n"
                 f"a = PCA().fit(X).explained_variance_ratio_[0]\np2 = PCA().fit(X2)\n"
                 f"result = (a, p2.explained_variance_ratio_[0], abs(p2.components_[0, {j}]))",
                 require=["explained_variance_ratio_"], tol=1e-6)
    if task == "duplicate":
        return P(f"{desc}. Append a copy of column {j} to `X` (`np.column_stack`). Fit `PCA()` on both and return "
                 f"`(number of features, number of components with explained variance ratio > 1e-10)` for the "
                 f"original and for the augmented data: the duplicate adds a feature but no new direction of variance.",
                 setup,
                 f"from sklearn.decomposition import PCA\nX2 = np.column_stack([X, X[:, {j}]])\n"
                 "count = lambda A: int((PCA().fit(A).explained_variance_ratio_ > 1e-10).sum())\n"
                 "result = ((X.shape[1], count(X)), (X2.shape[1], count(X2)))", require=["explained_variance_ratio_"])
    return P(f"{desc}. Append a constant column of 7s to `X`. Return the explained variance ratios of `PCA()` on the "
             f"augmented data, rounded to 6 decimals: the constant column adds a direction with zero variance.", setup,
             "from sklearn.decomposition import PCA\nX2 = np.column_stack([X, np.full(len(X), 7.0)])\n"
             "result = PCA().fit(X2).explained_variance_ratio_.round(6)", require=["explained_variance_ratio_"],
             tol=1e-6)


@template(M, "hard")
def neighbourhood_preservation(rng):
    setup, desc, n, d = _digits(rng, n=(120, 160))
    kk = rng.choice([5, 10])
    task = rng.choice(["overlap", "trust_curve", "perplexity"])
    nb = ("from sklearn.neighbors import NearestNeighbors\n"
          f"def overlap(A, B, k={kk}):\n"
          "    a = NearestNeighbors(n_neighbors=k + 1).fit(A).kneighbors(A, return_distance=False)[:, 1:]\n"
          "    b = NearestNeighbors(n_neighbors=k + 1).fit(B).kneighbors(B, return_distance=False)[:, 1:]\n"
          "    return float(np.mean([len(set(r) & set(s)) / k for r, s in zip(a, b)]))\n")
    ts = "TSNE(n_components=2, perplexity={}, init='pca', learning_rate='auto', max_iter=300, random_state=0)"
    head = (f"{desc}. Measure how well an embedding `Z` keeps neighbourhoods: for every point, the share of its {kk} "
            f"nearest neighbours in `X` that are also among its {kk} nearest neighbours in `Z` (use "
            f"`NearestNeighbors(n_neighbors={kk + 1})` and drop the first column, the point itself), averaged over "
            f"points.")
    if task == "overlap":
        return P(f"{head} Return the score of `PCA(n_components=2)` and of `{ts.format(20)}`, as a tuple.", setup,
                 "from sklearn.decomposition import PCA\nfrom sklearn.manifold import TSNE\n" + nb +
                 f"result = (overlap(X, PCA(n_components=2).fit_transform(X)), overlap(X, {ts.format(20)}.fit_transform(X)))",
                 require=["TSNE", "kneighbors"], tol=1e-6)
    if task == "perplexity":
        ps = [5, 20, 50]
        return P(f"{head} Return the scores of t-SNE embeddings `{ts.format('p')}` for p in `{ps}`, as a list.", setup,
                 "from sklearn.manifold import TSNE\n" + nb + f"result = [overlap(X, {ts.format('p')}.fit_transform(X)) for p in {ps}]",
                 require=["perplexity", "kneighbors"], tol=1e-6)
    ks = [2, 5, 10, 20]
    return P(f"{desc}. Return the `trustworthiness(X, Z, n_neighbors=k)` (from `sklearn.manifold`) of `Z = "
             f"PCA(n_components=k).fit_transform(X)` for k in `{ks}`: more components, fewer false neighbours.", setup,
             "from sklearn.decomposition import PCA\nfrom sklearn.manifold import trustworthiness\n"
             f"result = [trustworthiness(X, PCA(n_components=k).fit_transform(X), n_neighbors=5) for k in {ks}]",
             require=["trustworthiness"], tol=1e-9)


# ================================================================= challenge

@template(M, "challenge")
def pca_threshold_queries(rng):
    d, q, s = rng.randint(6, 9), rng.randint(4, 6), seeded(rng)
    report = rng.choice(["n", "kept", "dropped"])

    def gen(seed, d_, q_):
        return (f"g = np.random.default_rng({seed})\nvariances = (g.pareto(1.5, {d_}) + 0.01).round(4)\n"
                f"thresholds = g.uniform(0, 1, {q_}).round(6)\nreport = {report!r}")
    sol = ("v = np.sort(variances)[::-1]\nratio = v / v.sum()\nc = np.cumsum(ratio)\n"
           "k = np.minimum(np.searchsorted(c, thresholds, side='right') + 1, len(v))\n"
           "if report == 'n':\n    result = k\nelif report == 'kept':\n    result = c[k - 1]\nelse:\n"
           "    tail = np.concatenate([np.cumsum(v[::-1])[::-1], [0.0]])\n    result = tail[k]")
    tests = [gen(s + i, 100000, 100000) for i in (1, 2)]
    what = {"n": "the number of components `n_components_` it keeps (an int array)",
            "kept": "the variance ratio actually kept, `Σ` of the kept ratios (a float array)",
            "dropped": "the total variance of the dropped components (a float array, 0.0 if nothing is dropped)"}[report]
    return challenge(
        "Variance thresholds",
        "`variances` are the eigenvalues of a covariance matrix (in **no** particular order). `PCA(n_components=t)` "
        "with a float `t` sorts them in decreasing order, takes the cumulative explained variance ratio `c` and keeps "
        "the smallest number of components k such that `c[k − 1] > t` (strictly: scikit-learn uses "
        "`np.searchsorted(c, t, side='right') + 1`), never more than all of them. For every threshold in "
        f"`thresholds`, return {what}. (`report` = {report!r}.)",
        gen(s, d, q), sol, tests, "len(variances), len(thresholds) ≤ 10^5", speed="O((d + q) log d)", tol=1e-6)


@template(M, "challenge")
def nmf_mu_updates(rng):
    n, d, k, s, T = rng.randint(5, 7), rng.randint(4, 6), 2, seeded(rng), rng.randint(2, 4)
    report = rng.choice(["error", "W", "H"])

    def gen(seed, n_, d_, k_, T_):
        return (f"g = np.random.default_rng({seed})\nX = g.uniform(0, 5, ({n_}, {d_})).round(2)\n"
                f"W0 = g.uniform(0.1, 1, ({n_}, {k_})).round(3)\nH0 = g.uniform(0.1, 1, ({k_}, {d_})).round(3)\n"
                f"T = {T_}\nreport = {report!r}")
    sol = ("W, H = W0.astype(float).copy(), H0.astype(float).copy()\nfor _ in range(T):\n"
           "    W *= (X @ H.T) / (W @ (H @ H.T))\n    H *= (W.T @ X) / ((W.T @ W) @ H)\n"
           "result = {'error': float(np.linalg.norm(X - W @ H)), 'W': W, 'H': H}[report]")
    tests = [gen(s + i, 2000, 100, 5, 40) for i in (1, 2)]
    what = {"error": "the Frobenius norm `‖X − W H‖` after the last iteration (what `reconstruction_err_` reports)",
            "W": "the final `W`", "H": "the final `H`"}[report]
    return challenge(
        "NMF multiplicative updates",
        "`NMF(solver='mu', beta_loss='frobenius', init='custom')` factors a non-negative `X` (n × d) as `W @ H` with "
        "the Lee–Seung multiplicative updates. Starting from `W0`, `H0`, run `T` iterations; each one first updates "
        "`W ← W ∘ (X Hᵀ) / (W H Hᵀ)`, then, with the new `W`, `H ← H ∘ (Wᵀ X) / (Wᵀ W H)` (∘ and / element-wise). "
        f"Return {what}. (`report` = {report!r}.) Order the matrix products so you never build an n × d × k array "
        "and never form `W H` inside the loop.",
        gen(s, n, d, k, T), sol, tests, "n·d ≤ 2·10^5, k ≤ 5, T ≤ 40", speed="O(T·k·n·d)", tol=1e-6)


@template(M, "challenge")
def tsne_perplexity_search(rng):
    n, s = rng.randint(6, 8), seeded(rng)
    perp = rng.choice([2.0, 3.0])
    report = rng.choice(["sigma", "pmax", "joint"])

    def gen(seed, n_, p):
        return (f"g = np.random.default_rng({seed})\nX = g.normal(0, 1, ({n_}, 3)).round(3)\nperplexity = {p}\n"
                f"report = {report!r}")
    sol = ("D = ((X[:, None] - X[None]) ** 2).sum(axis=2)\nn = len(X)\nnp.fill_diagonal(D, np.inf)\n"
           "target = np.log(perplexity)\nlo, hi = np.full(n, -30.0), np.full(n, 30.0)   # bisection on log(beta)\n"
           "def probs(lb):\n    L = -np.exp(lb)[:, None] * D\n    L = L - L.max(axis=1, keepdims=True)\n"
           "    Pc = np.exp(L)\n    return Pc / Pc.sum(axis=1, keepdims=True)\n"
           "for _ in range(100):\n    mid = (lo + hi) / 2\n    Pc = probs(mid)\n"
           "    H = -(Pc * np.log(np.where(Pc > 0, Pc, 1))).sum(axis=1)\n"
           "    hi = np.where(H < target, mid, hi)\n    lo = np.where(H < target, lo, mid)\n"
           "lb = (lo + hi) / 2\nPc = probs(lb)\nif report == 'sigma':\n    result = np.sqrt(1 / (2 * np.exp(lb)))\n"
           "elif report == 'pmax':\n    result = Pc.max(axis=1)\nelse:\n    result = ((Pc + Pc.T) / (2 * n)).max(axis=1)")
    tests = [gen(s + i, 450, 30.0) for i in (1, 2)]
    what = {"sigma": "each point's bandwidth `σ_i = sqrt(1 / (2 β_i))` (an array)",
            "pmax": "each point's largest conditional probability `max_j p_{j|i}` (an array)",
            "joint": "for each i, the largest entry of row i of the symmetric joint matrix `P = (P_c + P_cᵀ) / (2n)`, "
                     "where `P_c[i, j] = p_{j|i}`"}[report]
    return challenge(
        "t-SNE perplexity calibration",
        "Before optimizing, t-SNE turns distances into neighbour probabilities `p_{j|i} ∝ exp(−β_i·‖x_i − x_j‖²)` "
        "(j ≠ i, each row normalized to 1) and tunes every `β_i` by binary search so that the row's entropy `H_i = "
        "−Σ_j p_{j|i} ln p_{j|i}` equals `ln(perplexity)` (the entropy falls as β grows). Calibrate every row to "
        f"precision far below 1e-6 and return {what}. (`report` = {report!r}.) Search all rows at once: a separate "
        "Python search per row is too slow.",
        gen(s, n, perp), sol, tests, "n ≤ 450 points", speed="O(iterations · n²) vectorized", tol=1e-5)


@template(M, "challenge")
def lda_fisher_ratio(rng):
    n, d, K, s = rng.randint(12, 16), rng.randint(2, 3), 3, seeded(rng)
    report = rng.choice(["ratio", "top"])

    def gen(seed, n_, d_, K_):
        return (f"g = np.random.default_rng({seed})\ny = g.permutation(np.arange({n_}) % {K_})\n"
                f"X = (g.normal(0, 1, ({n_}, {d_})) + g.normal(0, 2, ({K_}, {d_}))[y]).round(3)\nreport = {report!r}")
    sol = ("n, d = X.shape\nK = int(y.max()) + 1\ncnt = np.bincount(y, minlength=K).astype(float)\n"
           "S1 = np.zeros((K, d))\nnp.add.at(S1, y, X)\nmeans = S1 / cnt[:, None]\nmu = X.mean(axis=0)\n"
           "Xc = X - means[y]\nSw = Xc.T @ Xc / n\nB = means - mu\nSb = (B * (cnt / n)[:, None]).T @ B\n"
           "ev = np.sort(np.real(np.linalg.eigvals(np.linalg.solve(Sw, Sb))))[::-1]\n"
           "m = min(K - 1, d)\nresult = (ev / ev.sum())[:m] if report == 'ratio' else float(ev[0])")
    tests = [gen(s + i, 50000, 4, 2000) for i in (1, 2)]
    what = {"ratio": "`explained_variance_ratio_`: the generalized eigenvalues of `(Sb, Sw)` (the eigenvalues of "
                     "`Sw⁻¹ Sb`) divided by their sum, sorted in decreasing order, the first `min(K − 1, d)` of them",
            "top": "the largest eigenvalue of `Sw⁻¹ Sb` (Fisher's best separation ratio)"}[report]
    return challenge(
        "Fisher discriminant at scale",
        "`LinearDiscriminantAnalysis(solver='eigen')` works from two d × d matrices: the within-class covariance "
        "`Sw = Σ_c (n_c / n)·cov_c` (each `cov_c` divides by `n_c`) and the between-class covariance `Sb = Σ_c (n_c / n)"
        "(μ_c − μ)(μ_c − μ)ᵀ`. Rows of `X` have labels `y` in 0..K−1 (every class present). Return "
        f"{what}. (`report` = {report!r}.) With thousands of classes, a Python loop over classes (or a boolean mask "
        "per class) is too slow: accumulate per-class sums in one pass.",
        gen(s, n, d, K), sol, tests, "n ≤ 5·10^4, d ≤ 4, K ≤ 2000", speed="O(n·d² + K·d²)", tol=1e-6)


@template(M, "challenge")
def reconstruction_error_curve(rng):
    n, d, s = rng.randint(8, 12), rng.randint(3, 4), seeded(rng)
    report = rng.choice(["mse", "k_for", "r2"])
    frac = rng.choice([0.1, 0.25])

    def gen(seed, n_, d_):
        return (f"g = np.random.default_rng({seed})\nX = (g.normal(0, 1, ({n_}, {d_})) * np.linspace(3, 0.3, {d_})"
                f" @ np.linalg.qr(g.normal(0, 1, ({d_}, {d_})))[0]).round(3)\nfrac = {frac}\nreport = {report!r}")
    sol = ("n, d = X.shape\nXc = X - X.mean(axis=0)\nev = np.sort(np.linalg.eigvalsh(Xc.T @ Xc / n))[::-1]\n"
           "tail = np.concatenate([np.cumsum(ev[::-1])[::-1], [0.0]]) / d\n"
           "if report == 'mse':\n    result = tail\nelif report == 'k_for':\n"
           "    result = int(np.argmax(tail <= frac * tail[0] + 1e-12))\nelse:\n    result = 1 - tail / tail[0]")
    tests = [gen(s + i, 2000, 100) for i in (1, 2)]
    what = {"mse": "the reconstruction MSE for every k = 0, 1, ..., d (an array of length d + 1; k = 0 predicts the "
                   "column means)",
            "k_for": "the smallest k whose reconstruction MSE is at most `frac` × the MSE at k = 0",
            "r2": "`1 − MSE_k / MSE_0` for every k = 0..d (an array)"}[report]
    return challenge(
        "Reconstruction error for every k",
        "For each k, `PCA(n_components=k)` reconstructs `X` (n × d) as `inverse_transform(transform(X))`; its mean "
        "squared error over all n·d entries tells how much is lost. Refitting a PCA for every k is far too slow for "
        "wide data. Use one eigendecomposition: the error of keeping k components is the sum of the dropped "
        f"eigenvalues of the (biased) covariance matrix, divided by d. Return {what}. (`report` = {report!r}.)",
        gen(s, n, d), sol, tests, "n·d ≤ 2·10^5, d ≤ 100", speed="O(n·d² + d³): one eigendecomposition", tol=1e-6)


@template(M, "challenge")
def trustworthiness_score(rng):
    n, s, k = rng.randint(10, 14), seeded(rng), rng.randint(2, 3)
    report = rng.choice(["trust", "continuity", "intruders"])

    def gen(seed, n_, k_):
        return (f"g = np.random.default_rng({seed})\nX = g.normal(0, 1, ({n_}, 5))\n"
                f"Z = X[:, :2] + g.normal(0, 0.5, ({n_}, 2))   # a rough 2-D embedding\nk = {k_}\nreport = {report!r}")
    sol = ("def ranks_and_knn(A, B, k):\n    n = len(A)\n"
           "    DA = ((A[:, None] - A[None]) ** 2).sum(axis=2)\n    np.fill_diagonal(DA, np.inf)\n"
           "    rank = np.empty((n, n), dtype=np.int64)\n"
           "    rank[np.arange(n)[:, None], np.argsort(DA, axis=1)] = np.arange(1, n + 1)[None, :]\n"
           "    DB = ((B[:, None] - B[None]) ** 2).sum(axis=2)\n    np.fill_diagonal(DB, np.inf)\n"
           "    knn = np.argpartition(DB, k, axis=1)[:, :k]\n    return rank[np.arange(n)[:, None], knn] - k\n"
           "def trust(A, B, k):\n    n = len(A)\n    r = ranks_and_knn(A, B, k)\n"
           "    return 1 - 2 / (n * k * (2 * n - 3 * k - 1)) * np.maximum(r, 0).sum()\n"
           "if report == 'trust':\n    result = float(trust(X, Z, k))\nelif report == 'continuity':\n"
           "    result = float(trust(Z, X, k))\nelse:\n    result = int((ranks_and_knn(X, Z, k) > 0).sum())")
    tests = [gen(s + i, 450, 10) for i in (1, 2)]
    what = {"trust": "the trustworthiness of `Z`: `1 − 2 / (n·k·(2n − 3k − 1)) · Σ_i Σ_{j ∈ U_i} (r(i, j) − k)`, where "
                     "`U_i` are the points among i's k nearest neighbours in `Z` that are **not** among its k nearest "
                     "in `X`, and `r(i, j)` is j's rank among i's neighbours in `X` (1 = nearest)",
            "continuity": "the *continuity*: the same formula with the roles of `X` and `Z` swapped (`trustworthiness(Z, "
                          "X, n_neighbors=k)`)",
            "intruders": "the total number of *intruders*: pairs (i, j) where j is among i's k nearest neighbours in "
                         "`Z` but not in `X`"}[report]
    return challenge(
        "Trustworthiness of an embedding",
        "`sklearn.manifold.trustworthiness` checks whether an embedding `Z` of the points `X` invents false "
        "neighbours. Distances are Euclidean, a point is never its own neighbour, and there are no ties. Return "
        f"{what}. (`report` = {report!r}, `k` is given.) Compute all ranks at once (one `argsort` of the distance "
        "matrix) instead of searching for every pair.",
        gen(s, n, k), sol, tests, "n ≤ 450 points, k ≤ 10", speed="O(n² log n)", tol=1e-9)
