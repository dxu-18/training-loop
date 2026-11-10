"""4.14 Regression, clustering & distance metrics: error metrics for regressors, cluster quality with and without
ground truth, pairwise distances and kernels, ranking quality (NDCG)."""

from ._common import P, catalog, challenge, lit, mat, reg_data, seeded, table, template

M = "4.14"
MET = "from sklearn.metrics import "
KM = "from sklearn.cluster import KMeans\n"
REG_METRICS = {   # name -> (what it is, NumPy formula on y_true / y_hat)
    "mean_squared_error": ("the mean squared error", "((y_true - y_hat) ** 2).mean()"),
    "root_mean_squared_error": ("the root mean squared error", "np.sqrt(((y_true - y_hat) ** 2).mean())"),
    "mean_absolute_error": ("the mean absolute error", "np.abs(y_true - y_hat).mean()"),
    "median_absolute_error": ("the median absolute error", "np.median(np.abs(y_true - y_hat))"),
    "max_error": ("the largest absolute error", "np.abs(y_true - y_hat).max()"),
    "mean_absolute_percentage_error": ("the mean absolute percentage error (as a fraction, not ×100)",
                                       "(np.abs(y_true - y_hat) / np.abs(y_true)).mean()"),
    "mean_squared_log_error": ("the mean squared log error",
                               "((np.log1p(y_true) - np.log1p(y_hat)) ** 2).mean()"),
}


def _pair(rng, n=(5, 8)):
    """y_true / y_hat setup lines (positive values: every metric works)."""
    k = rng.randint(*n)
    lo, hi, noise = rng.choice([(1, 10, 1.0), (10, 80, 6.0), (100, 400, 30.0)])
    yt = [round(rng.uniform(lo, hi), 1) for _ in range(k)]
    yp = [round(max(0.1, v + rng.gauss(0, noise)), 1) for v in yt]
    return f"y_true = np.array({lit(yt)})\ny_hat = np.array({lit(yp)})", yt, yp


def _blobs(rng, k=None, n=(60, 180), std=None, labels: int | None = None):
    """make_blobs data X, y_true (+ KMeans `labels` with `labels` clusters when given)."""
    k = k or rng.randint(2, 5)
    n_ = rng.randint(*n)
    std = std or rng.choice([0.6, 0.9, 1.2, 1.6])
    d = rng.choice([2, 2, 3])
    setup = (f"from sklearn.datasets import make_blobs\nX, y_true = make_blobs(n_samples={n_}, centers={k}, "
             f"n_features={d}, cluster_std={std}, random_state={seeded(rng)})")
    desc = f"`X` holds {n_} points in {d}-D drawn around {k} centres (`make_blobs`) and `y_true` the centre each came from"
    if labels:
        setup += f"\nfrom sklearn.cluster import KMeans\nlabels = KMeans(n_clusters={labels}, n_init='auto', random_state=0).fit_predict(X)"
        desc += f"; `labels` is a {labels}-cluster `KMeans` result on `X`"
    return setup, desc, k


def _ab(rng, na=(3, 5), nb=(2, 4), d=(2, 4)):
    s, d_ = seeded(rng), rng.randint(*d)
    a, b = rng.randint(*na), rng.randint(*nb)
    setup = (f"A = np.random.default_rng({s}).normal(size=({a}, {d_})).round(2)\n"
             f"B = np.random.default_rng({s + 1}).normal(size=({b}, {d_})).round(2)")
    return setup, f"`A` ({a} × {d_}) and `B` ({b} × {d_}) hold random rows", a, b, d_


# ================================================================= lite

@template(M, "lite")
def regression_error_metrics(rng):
    setup, _, _ = _pair(rng)
    items = [(name, f"Return {what} of `y_hat` against `y_true` (`{name}` from `sklearn.metrics`).",
              f"{MET}{name}\nresult = {name}(y_true, y_hat)") for name, (what, _) in REG_METRICS.items()]
    items += [
        ("r2_score", "Return the R² of `y_hat` against `y_true` (`r2_score`; the true values come first).",
         f"{MET}r2_score\nresult = r2_score(y_true, y_hat)"),
        ("explained_variance_score", "Return the explained variance score of `y_hat` (`explained_variance_score`).",
         f"{MET}explained_variance_score\nresult = explained_variance_score(y_true, y_hat)"),
    ]
    return catalog(rng, items, setup, "`y_true` holds true targets and `y_hat` a model's predictions.", tol=1e-9)


@template(M, "lite")
def cluster_quality_scores(rng):
    k = rng.randint(2, 4)
    fit_k = rng.choice([k, k, k + 1])
    setup, desc, _ = _blobs(rng, k=k, labels=fit_k)
    m = rng.randint(3, 6)
    r = rng.randint(1, 9)
    items = [
        ("silhouette_score", "Return the mean silhouette coefficient of the clustering `labels` (`silhouette_score(X, "
                             "labels)`; near 1 = tight, well-separated clusters).",
         f"{MET}silhouette_score\nresult = silhouette_score(X, labels)"),
        ("silhouette_samples", f"Return the silhouette coefficient of each of the first {m} points "
                               f"(`silhouette_samples`, one value per point).",
         f"{MET}silhouette_samples\nresult = silhouette_samples(X, labels)[:{m}]"),
        ("davies_bouldin_score", "Return the Davies-Bouldin index of `labels` (`davies_bouldin_score`; lower is "
                                 "better).", f"{MET}davies_bouldin_score\nresult = davies_bouldin_score(X, labels)"),
        ("calinski_harabasz_score", "Return the Calinski-Harabasz index of `labels` (`calinski_harabasz_score`; "
                                    "higher is better).",
         f"{MET}calinski_harabasz_score\nresult = calinski_harabasz_score(X, labels)"),
        ("adjusted_rand_score", "Compare the clustering with the truth: return `adjusted_rand_score(y_true, labels)` "
                                "(1 = same partition, about 0 = chance).",
         f"{MET}adjusted_rand_score\nresult = adjusted_rand_score(y_true, labels)"),
        ("normalized_mutual_info_score", "Return the normalized mutual information between `y_true` and `labels` "
                                         "(`normalized_mutual_info_score`).",
         f"{MET}normalized_mutual_info_score\nresult = normalized_mutual_info_score(y_true, labels)"),
        ("homogeneity_score", "Return the homogeneity of `labels` with respect to `y_true` (`homogeneity_score`: 1 "
                              "when every cluster holds a single true group).",
         f"{MET}homogeneity_score\nresult = homogeneity_score(y_true, labels)"),
        ("fit_predict", f"Cluster `X` yourself with `KMeans(n_clusters={k}, n_init='auto', random_state={r})` using "
                        f"`fit_predict`, and return the cluster sizes (`np.bincount` of the labels).",
         f"{KM}result = np.bincount(KMeans(n_clusters={k}, n_init='auto', random_state={r}).fit_predict(X))",
         {"require": ["KMeans", "fit_predict"]}),
        ("silhouette_score", "Return the silhouette score of `labels` measured with Manhattan distances "
                             "(`metric='manhattan'`).",
         f"{MET}silhouette_score\nresult = silhouette_score(X, labels, metric='manhattan')"),
    ]
    return catalog(rng, items, setup, f"{desc}.", tol=1e-9)


@template(M, "lite")
def pairwise_and_kernels(rng):
    setup, desc, a, b, d = _ab(rng)
    g = rng.choice([0.1, 0.5, 1.0, 2.0])
    pw = "from sklearn.metrics.pairwise import "
    items = [
        ("cosine_similarity", "Return the cosine similarity of every row of `A` with every row of `B` "
                              "(`cosine_similarity` from `sklearn.metrics.pairwise`).",
         f"{pw}cosine_similarity\nresult = cosine_similarity(A, B)"),
        ("cosine_similarity", "What shape does `cosine_similarity(A, B)` have? Return it as a tuple.",
         f"{pw}cosine_similarity\nresult = cosine_similarity(A, B).shape"),
        ("euclidean_distances", "Return the Euclidean distance between every row of `A` and every row of `B` "
                                "(`euclidean_distances`).", f"{pw}euclidean_distances\nresult = euclidean_distances(A, B)"),
        ("manhattan_distances", "Return the Manhattan (L1) distances between the rows of `A` and the rows of `B` "
                                "(`manhattan_distances`).", f"{pw}manhattan_distances\nresult = manhattan_distances(A, B)"),
        ("rbf_kernel", f"Return the RBF (Gaussian) kernel between the rows of `A` and `B` with `gamma={g}` "
                       f"(`rbf_kernel`).", f"{pw}rbf_kernel\nresult = rbf_kernel(A, B, gamma={g})"),
        ("linear_kernel", "Return the linear kernel (all dot products) between the rows of `A` and `B` "
                          "(`linear_kernel`).", f"{pw}linear_kernel\nresult = linear_kernel(A, B)"),
        ("pairwise_distances", "Return the cosine **distances** among the rows of `A` (`pairwise_distances` from "
                               "`sklearn.metrics` with `metric='cosine'`; a square matrix).",
         f"{MET}pairwise_distances\nresult = pairwise_distances(A, metric='cosine')"),
        ("pairwise_distances", "Return the Euclidean distances from the rows of `A` to the rows of `B` with the "
                               "general `pairwise_distances(A, B, metric='euclidean')`.",
         f"{MET}pairwise_distances\nresult = pairwise_distances(A, B, metric='euclidean')"),
        ("euclidean_distances", "With a single argument, `euclidean_distances(A)` compares `A` with itself. Return "
                                "that square matrix (zeros on the diagonal).",
         f"{pw}euclidean_distances\nresult = euclidean_distances(A)"),
    ]
    return catalog(rng, items, setup, f"{desc}.", tol=1e-9)


@template(M, "lite")
def ndcg_basics(rng):
    q, m = rng.randint(1, 3), rng.randint(4, 7)
    rel = [[rng.randint(0, 3) for _ in range(m)] for _ in range(q)]
    for r in rel:
        r[rng.randrange(m)] = 3
    sc = [[round(rng.random(), 2) for _ in range(m)] for _ in range(q)]
    k = rng.randint(2, m - 1)
    setup = f"true_relevance = np.array({lit(rel)})\nscores = np.array({lit(sc)})"
    items = [
        ("ndcg_score", "Return the NDCG of the ranking `scores` (`ndcg_score(true_relevance, scores)`, averaged over "
                       "the queries; 1 = ideal order).", f"{MET}ndcg_score\nresult = ndcg_score(true_relevance, scores)"),
        ("ndcg_score", f"Only the top {k} results are shown to users: return the NDCG@{k} (`k={k}`).",
         f"{MET}ndcg_score\nresult = ndcg_score(true_relevance, scores, k={k})"),
        ("ndcg_score", "Reverse the ranking (use `-scores` as the scores) and return its NDCG: a bad order scores lower.",
         f"{MET}ndcg_score\nresult = ndcg_score(true_relevance, -scores)"),
        ("ndcg_score", "Return the NDCG of the **first** query only (keep it 2-D: `true_relevance[:1]`, `scores[:1]`).",
         f"{MET}ndcg_score\nresult = ndcg_score(true_relevance[:1], scores[:1])"),
        ("ndcg_score", "Return `(ndcg of scores, ndcg of the ideal ranking)`: use the relevances themselves as the "
                       "scores for the second one.",
         f"{MET}ndcg_score\nresult = (ndcg_score(true_relevance, scores), ndcg_score(true_relevance, true_relevance))"),
    ]
    return catalog(rng, items, setup, f"A search engine answered {q} quer{'y' if q == 1 else 'ies'} with {m} results "
                                      f"each: `true_relevance` grades each result (0-3) and `scores` is the engine's "
                                      f"score (higher = ranked earlier); one row per query.", tol=1e-9)


# ================================================================= easy

@template(M, "easy")
def regression_errors_by_hand(rng):
    setup, _, _ = _pair(rng)
    name = rng.choice(sorted(REG_METRICS))
    what, formula = REG_METRICS[name]
    return P(f"`y_true` holds true targets and `y_hat` predictions. Compute {what} **by hand** with NumPy, then with "
             f"`{name}` from `sklearn.metrics`, and return both as a tuple `(by_hand, sklearn_value)` (they agree).",
             setup, f"{MET}{name}\nresult = ({formula}, {name}(y_true, y_hat))", require=[name], tol=1e-9)


@template(M, "easy")
def r2_score_facts(rng):
    setup, yt, yp = _pair(rng)
    c = rng.choice([2, 5, -3, 10])
    task = rng.choice(["hand", "baseline", "offset", "order", "negative"])
    head = "`y_true` holds true targets and `y_hat` predictions."
    if task == "hand":
        return P(f"{head} R² = 1 − (sum of squared residuals) / (sum of squared deviations of `y_true` from its mean). "
                 f"Return `(by_hand, r2_score(y_true, y_hat))`.", setup,
                 f"{MET}r2_score\nresult = (1 - ((y_true - y_hat) ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum(), "
                 "r2_score(y_true, y_hat))", require=["r2_score"], tol=1e-9)
    if task == "baseline":
        return P(f"{head} Baselines: return the R² of always predicting the **mean** of `y_true` and of always "
                 f"predicting its **median**, as a tuple. (The mean scores exactly 0; anything worse goes negative.)",
                 setup, f"{MET}r2_score\nn = len(y_true)\nresult = (r2_score(y_true, np.full(n, y_true.mean())), "
                        "r2_score(y_true, np.full(n, np.median(y_true))))", require=["r2_score"], tol=1e-9)
    if task == "offset":
        return P(f"{head} Shift every prediction by {c} (`y_hat + {c}`) and return `(r2_score, "
                 f"explained_variance_score)` of the shifted predictions: the explained variance ignores a constant "
                 f"bias, R² does not.", setup,
                 f"{MET}r2_score, explained_variance_score\nshifted = y_hat + {c}\n"
                 "result = (r2_score(y_true, shifted), explained_variance_score(y_true, shifted))",
                 require=["r2_score", "explained_variance_score"], tol=1e-9)
    if task == "order":
        return P(f"{head} Argument order matters: return `(r2_score(y_true, y_hat), r2_score(y_hat, y_true))` (only "
                 f"the first is the model's R²).", setup,
                 f"{MET}r2_score\nresult = (r2_score(y_true, y_hat), r2_score(y_hat, y_true))", require=["r2_score"],
                 tol=1e-9)
    return P(f"{head} R² has no lower bound. Return the R² of the predictions in **reversed** order (`y_hat[::-1]`).",
             setup, f"{MET}r2_score\nresult = r2_score(y_true, y_hat[::-1])", require=["r2_score"], tol=1e-9)


@template(M, "easy")
def holdout_regression_metric(rng):
    d = reg_data(rng)
    s = seeded(rng)
    name = rng.choice(["mean_absolute_error", "root_mean_squared_error", "r2_score", "median_absolute_error",
                       "explained_variance_score", "max_error"])
    what = {"r2_score": "R²", "explained_variance_score": "explained variance score"}.get(name, REG_METRICS.get(name, ("",))[0])
    if rng.random() < 0.5:
        model, imp = "LinearRegression()", "from sklearn.linear_model import LinearRegression"
    else:
        a = rng.choice([0.1, 1.0, 10.0])
        model, imp = f"Ridge(alpha={a})", "from sklearn.linear_model import Ridge"
    return P(f"{d.desc}. Split with `train_test_split(X, y, test_size=0.25, random_state={s})`, fit `{model}` on the "
             f"training part and return its {what} on the **test** part (`{name}`).", d.setup,
             f"from sklearn.model_selection import train_test_split\n{imp}\n{MET}{name}\n"
             f"X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state={s})\n"
             f"result = {name}(y_te, {model}.fit(X_tr, y_tr).predict(X_te))", require=[name], tol=1e-6)


@template(M, "easy")
def distances_by_hand(rng):
    setup, desc, a, b, d = _ab(rng)
    g = rng.choice([0.2, 0.5, 1.0])
    pw = "from sklearn.metrics.pairwise import "
    task = rng.choice(["cos", "euc", "man", "rbf", "cosdist"])
    if task == "cos":
        return P(f"{desc}. Compute their cosine similarities **by hand**: dot products divided by the product of the "
                 f"row norms (`np.linalg.norm(..., axis=1)`, `np.outer`). It matches `cosine_similarity(A, B)`.", setup,
                 "result = A @ B.T / np.outer(np.linalg.norm(A, axis=1), np.linalg.norm(B, axis=1))", tol=1e-9)
    if task == "euc":
        return P(f"{desc}. Compute every Euclidean distance between a row of `A` and a row of `B` **by hand** with "
                 f"broadcasting (`A[:, None, :] - B[None, :, :]`); it matches `euclidean_distances(A, B)`.", setup,
                 "result = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2))", tol=1e-9)
    if task == "man":
        return P(f"{desc}. Compute the Manhattan distances between the rows of `A` and of `B` **by hand** with "
                 f"broadcasting; they match `manhattan_distances(A, B)`.", setup,
                 "result = np.abs(A[:, None, :] - B[None, :, :]).sum(axis=2)", tol=1e-9)
    if task == "rbf":
        return P(f"{desc}. The RBF kernel is exp(−γ·‖a − b‖²). Rebuild `rbf_kernel(A, B, gamma={g})` from "
                 f"`euclidean_distances(A, B)` and return it.", setup,
                 f"{pw}euclidean_distances\nresult = np.exp(-{g} * euclidean_distances(A, B) ** 2)",
                 require=["euclidean_distances"], tol=1e-9)
    return P(f"{desc}. Cosine *distance* is 1 − cosine similarity. Return the cosine distances between the rows of "
             f"`A` and `B` computed from `cosine_similarity` (they equal `pairwise_distances(A, B, metric='cosine')`).",
             setup, f"{pw}cosine_similarity\nresult = 1 - cosine_similarity(A, B)", require=["cosine_similarity"],
             tol=1e-9)


@template(M, "easy")
def kmeans_then_score(rng):
    k = rng.randint(2, 4)
    setup, desc, _ = _blobs(rng, k=k)
    fit_k = rng.choice([k, k + 1, max(2, k - 1)])
    r = rng.randint(0, 9)
    name = rng.choice(["silhouette_score", "adjusted_rand_score", "normalized_mutual_info_score", "homogeneity_score",
                       "davies_bouldin_score", "calinski_harabasz_score"])
    truth = name in ("adjusted_rand_score", "normalized_mutual_info_score", "homogeneity_score")
    call = f"{name}(y_true, labels)" if truth else f"{name}(X, labels)"
    note = "it compares with the ground truth `y_true`" if truth else "it needs only `X` and the labels, no ground truth"
    return P(f"{desc}. Cluster `X` with `KMeans(n_clusters={fit_k}, n_init='auto', random_state={r})` (`fit_predict`) "
             f"and return the `{name}` of the result ({note}).", setup,
             f"{KM}{MET}{name}\nlabels = KMeans(n_clusters={fit_k}, n_init='auto', random_state={r}).fit_predict(X)\n"
             f"result = {call}", require=["KMeans", name], tol=1e-9)


@template(M, "easy")
def silhouette_sample_stats(rng):
    k = rng.randint(2, 4)
    setup, desc, _ = _blobs(rng, k=k, std=rng.choice([1.2, 1.6, 2.2]), labels=k)
    task = rng.choice(["mean", "neg", "per", "worst", "frac"])
    t = rng.choice([0.5, 0.6, 0.7])
    sol = f"{MET}silhouette_samples\ns = silhouette_samples(X, labels)\n"
    if task == "mean":
        return P(f"{desc}. The silhouette score is the mean of the per-point coefficients. Return "
                 f"`(silhouette_samples(X, labels).mean(), silhouette_score(X, labels))`.", setup,
                 f"{MET}silhouette_samples, silhouette_score\n"
                 "result = (silhouette_samples(X, labels).mean(), silhouette_score(X, labels))",
                 require=["silhouette_samples", "silhouette_score"], tol=1e-9)
    if task == "neg":
        return P(f"{desc}. A negative silhouette means a point sits closer to another cluster than to its own. Return "
                 f"how many points have one (`silhouette_samples`).", setup, sol + "result = int((s < 0).sum())",
                 require=["silhouette_samples"])
    if task == "per":
        return P(f"{desc}. Return the mean silhouette coefficient of each cluster as a list, cluster 0 first "
                 f"(`silhouette_samples`, then average per label).", setup,
                 sol + f"result = [float(s[labels == c].mean()) for c in range({k})]", require=["silhouette_samples"],
                 tol=1e-9)
    if task == "worst":
        return P(f"{desc}. Return the index of the point with the **lowest** silhouette coefficient.", setup,
                 sol + "result = int(s.argmin())", require=["silhouette_samples"])
    return P(f"{desc}. Return the fraction of points whose silhouette coefficient is above {t}.", setup,
             sol + f"result = float((s > {t}).mean())", require=["silhouette_samples"], tol=1e-9)


@template(M, "easy")
def cluster_labels_are_names(rng):
    k, n = rng.randint(2, 4), rng.randint(12, 24)
    yt = [i % k for i in range(n)]
    rng.shuffle(yt)
    perm = list(range(k))
    while perm == list(range(k)):
        rng.shuffle(perm)
    task = rng.choice(["acc", "nmi", "split", "merge"])
    setup = f"y_true = np.array({lit(yt)})"
    if task in ("acc", "nmi"):
        setup += f"\nlabels = np.array({lit(perm)})[y_true]   # the same groups, other names"
        if task == "acc":
            return P("`labels` puts exactly the same points together as `y_true`, but with different cluster numbers. "
                     "Return `(accuracy, ari)`: the fraction of positions where `labels == y_true`, and "
                     "`adjusted_rand_score(y_true, labels)`. Cluster numbers are arbitrary; ARI sees that.", setup,
                     f"{MET}adjusted_rand_score\nresult = (float((labels == y_true).mean()), adjusted_rand_score(y_true, labels))",
                     require=["adjusted_rand_score"], tol=1e-9)
        return P("`labels` puts exactly the same points together as `y_true`, with different cluster numbers. Return "
                 "`(normalized_mutual_info_score(y_true, labels), homogeneity_score(y_true, labels))` (both 1: "
                 "renaming clusters changes nothing).", setup,
                 f"{MET}normalized_mutual_info_score, homogeneity_score\n"
                 "result = (normalized_mutual_info_score(y_true, labels), homogeneity_score(y_true, labels))",
                 require=["normalized_mutual_info_score", "homogeneity_score"], tol=1e-9)
    if task == "split":
        setup += "\nlabels = 2 * y_true + np.arange(len(y_true)) % 2   # each true group split in two"
        return P("`labels` splits every true group of `y_true` into two clusters. Return `(homogeneity_score(y_true, "
                 "labels), adjusted_rand_score(y_true, labels))`: every cluster is still pure (homogeneity 1), but "
                 "the partitions differ.", setup,
                 f"{MET}homogeneity_score, adjusted_rand_score\n"
                 "result = (homogeneity_score(y_true, labels), adjusted_rand_score(y_true, labels))",
                 require=["homogeneity_score", "adjusted_rand_score"], tol=1e-9)
    setup += "\nlabels = y_true // 2   # true groups 0 and 1 merged, 2 and 3 merged"
    return P("`labels` merges pairs of true groups of `y_true` into single clusters. Return `(homogeneity_score(y_true, "
             "labels), completeness_score(y_true, labels))` (both in `sklearn.metrics`): merged clusters are no longer "
             "pure, but each true group still sits in one cluster.", setup,
             f"{MET}homogeneity_score, completeness_score\n"
             "result = (homogeneity_score(y_true, labels), completeness_score(y_true, labels))",
             require=["homogeneity_score"], tol=1e-9)


@template(M, "easy")
def metric_identities(rng):
    setup, _, _ = _pair(rng)
    task = rng.choice(["msle", "rmse", "maxerr"])
    if task == "msle":
        return P("`y_true` and `y_hat` are positive. The mean squared log error is the MSE of `log1p` values. Return "
                 "`(mean_squared_log_error(y_true, y_hat), mean_squared_error(np.log1p(y_true), np.log1p(y_hat)))`.",
                 setup, f"{MET}mean_squared_log_error, mean_squared_error\n"
                        "result = (mean_squared_log_error(y_true, y_hat), mean_squared_error(np.log1p(y_true), np.log1p(y_hat)))",
                 require=["mean_squared_log_error", "mean_squared_error"], tol=1e-9)
    if task == "rmse":
        return P("`y_true` holds targets and `y_hat` predictions. Return `(root_mean_squared_error(y_true, y_hat), "
                 "np.sqrt(mean_squared_error(y_true, y_hat)))`: RMSE is the square root of the MSE, in the target's "
                 "units.", setup,
                 f"{MET}root_mean_squared_error, mean_squared_error\n"
                 "result = (root_mean_squared_error(y_true, y_hat), np.sqrt(mean_squared_error(y_true, y_hat)))",
                 require=["root_mean_squared_error", "mean_squared_error"], tol=1e-9)
    return P("`y_true` holds targets and `y_hat` predictions. Return `(max_error(y_true, y_hat), i)` where `i` is the "
             "index of the prediction that makes that worst error.", setup,
             f"{MET}max_error\nresult = (max_error(y_true, y_hat), int(np.abs(y_true - y_hat).argmax()))",
             require=["max_error"], tol=1e-9)


@template(M, "easy")
def nearest_rows_by_metric(rng):
    n, d, s = rng.randint(6, 10), rng.randint(2, 4), seeded(rng)
    metric = rng.choice(["euclidean", "manhattan", "cosine", "chebyshev"])
    setup = f"X = np.random.default_rng({s}).normal(size=({n}, {d})).round(2)"
    task = rng.choice(["nn", "pair", "far"])
    sol = f"{MET}pairwise_distances\nD = pairwise_distances(X, metric={metric!r})\n"
    if task == "nn":
        return P(f"`X` holds {n} rows. With `pairwise_distances(X, metric={metric!r})`, return for every row the index "
                 f"of its nearest **other** row (set the diagonal to `np.inf` first).", setup,
                 sol + "np.fill_diagonal(D, np.inf)\nresult = D.argmin(axis=1)", require=["pairwise_distances"])
    if task == "pair":
        return P(f"`X` holds {n} rows. Using `pairwise_distances` with `metric={metric!r}`, return the closest pair of "
                 f"different rows as a tuple `(i, j)` with `i < j`.", setup,
                 sol + "D[np.tril_indices(len(X))] = np.inf\ni, j = np.unravel_index(D.argmin(), D.shape)\n"
                       "result = (int(i), int(j))", require=["pairwise_distances"])
    r = rng.randrange(n)
    return P(f"`X` holds {n} rows. Using `pairwise_distances` with `metric={metric!r}`, return `(index, distance)` of "
             f"the row **farthest** from row {r}.", setup,
             sol + f"j = int(D[{r}].argmax())\nresult = (j, float(D[{r}, j]))", require=["pairwise_distances"], tol=1e-9)


# ================================================================= medium

@template(M, "medium")
def regression_holdout_report(rng):
    t = table(rng, kind="reg", numeric_only=True, n=(120, 260), only=["houses", "servers", "phones", "experiments"])
    s = seeded(rng)
    a = rng.choice([1.0, 10.0, 100.0])
    setup = (t.setup + "\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.linear_model import "
             f"LinearRegression\nX_tr, X_te, y_tr, y_te = train_test_split(df, y, test_size=0.3, random_state={s})\n"
             "reg = LinearRegression().fit(X_tr, y_tr)\npred = reg.predict(X_te)")
    items = [
        ("root_mean_squared_error", "Return a dict with the test `'mae'`, `'rmse'` and `'r2'` of `pred`.",
         f"{MET}mean_absolute_error, root_mean_squared_error, r2_score\n"
         "result = {'mae': mean_absolute_error(y_te, pred), 'rmse': root_mean_squared_error(y_te, pred), "
         "'r2': r2_score(y_te, pred)}", {"require": ["mean_absolute_error", "root_mean_squared_error", "r2_score"]}),
        ("mean_absolute_error", "Is the model better than a constant? Return `(model_mae, baseline_mae)` on the test "
                                "set, where the baseline always predicts the **training** mean `y_tr.mean()`.",
         f"{MET}mean_absolute_error\nresult = (mean_absolute_error(y_te, pred), "
         "mean_absolute_error(y_te, np.full(len(y_te), y_tr.mean())))"),
        ("max_error", "Return `(max_error(y_te, pred), label)` where `label` is the index label (in `X_te.index`) of "
                      "the test row with that worst error.",
         f"{MET}max_error\nresult = (max_error(y_te, pred), int(X_te.index[np.abs(y_te - pred).argmax()]))"),
        ("mean_absolute_percentage_error", "Report the test error as a percentage: return 100 × "
                                           "`mean_absolute_percentage_error(y_te, pred)`.",
         f"{MET}mean_absolute_percentage_error\nresult = 100 * mean_absolute_percentage_error(y_te, pred)"),
        ("Ridge", f"Fit `Ridge(alpha={a})` on the same training data and return the test RMSE of both models as "
                  f"`(linear_rmse, ridge_rmse)`.",
         f"from sklearn.linear_model import Ridge\n{MET}root_mean_squared_error\n"
         f"ridge = Ridge(alpha={a}).fit(X_tr, y_tr)\n"
         "result = (root_mean_squared_error(y_te, pred), root_mean_squared_error(y_te, ridge.predict(X_te)))",
         {"require": ["Ridge", "root_mean_squared_error"]}),
        ("r2_score", "Return `(train_r2, test_r2)` with `r2_score` (the model's predictions on `X_tr` and on `X_te`).",
         f"{MET}r2_score\nresult = (r2_score(y_tr, reg.predict(X_tr)), r2_score(y_te, pred))"),
        ("median_absolute_error", "Return `(median_absolute_error, mean_absolute_error)` on the test set; the median "
                                  "ignores the few large misses.",
         f"{MET}median_absolute_error, mean_absolute_error\n"
         "result = (median_absolute_error(y_te, pred), mean_absolute_error(y_te, pred))",
         {"require": ["median_absolute_error", "mean_absolute_error"]}),
        ("explained_variance_score", "Return `(explained_variance_score, r2_score)` on the test set (they differ by "
                                     "the squared mean residual over the variance).",
         f"{MET}explained_variance_score, r2_score\n"
         "result = (explained_variance_score(y_te, pred), r2_score(y_te, pred))",
         {"require": ["explained_variance_score", "r2_score"]}),
        ("root_mean_squared_error", "Return the test RMSE divided by the standard deviation of `y_te` (a unit-free "
                                    "'normalized RMSE'; below 1 beats predicting the mean).",
         f"{MET}root_mean_squared_error\nresult = root_mean_squared_error(y_te, pred) / y_te.std()"),
        ("mean_absolute_percentage_error", "Return the MAPE (`mean_absolute_percentage_error`, as a fraction) on the "
                                           "test rows whose target is above the median of `y_te` only.",
         f"{MET}mean_absolute_percentage_error\nbig = y_te > np.median(y_te)\n"
         "result = mean_absolute_percentage_error(y_te[big], pred[big])"),
        ("mean_squared_error", "The MSE splits into bias² + variance of the residuals. Return "
                               "`(mean_squared_error(y_te, pred), res.mean() ** 2 + res.var())` with `res = y_te - pred`.",
         f"{MET}mean_squared_error\nres = y_te - pred\nresult = (mean_squared_error(y_te, pred), res.mean() ** 2 + res.var())"),
    ]
    return catalog(rng, items, setup, f"{t.desc}. The rows were split with `train_test_split(df, y, test_size=0.3, "
                                      f"random_state={s})` into `X_tr, X_te, y_tr, y_te`; `reg = LinearRegression()` "
                                      f"was fitted on the training part and `pred` holds its test predictions.",
                   tol=1e-6)


@template(M, "medium")
def choose_k_by_silhouette(rng):
    setup, desc, k = _blobs(rng, k=rng.randint(2, 5), std=rng.choice([0.7, 1.0, 1.3]))
    hi = rng.choice([6, 7])
    ks = f"range(2, {hi})"
    km = "KMeans(n_clusters=k, n_init='auto', random_state=0).fit_predict(X)"
    task = rng.choice(["best", "scores", "db", "ch", "ari"])
    head = (f"{desc} (pretend you don't know how many). For each `k` in `{ks}`, cluster with `KMeans(n_clusters=k, "
            f"n_init='auto', random_state=0)`")
    if task == "best":
        return P(f"{head} and return the `k` with the highest `silhouette_score`.", setup,
                 f"{KM}{MET}silhouette_score\nscores = {{k: silhouette_score(X, {km}) for k in {ks}}}\n"
                 "result = max(scores, key=scores.get)", require=["KMeans", "silhouette_score"])
    if task == "scores":
        return P(f"{head} and return the list of silhouette scores in order of `k`.", setup,
                 f"{KM}{MET}silhouette_score\nresult = [silhouette_score(X, {km}) for k in {ks}]",
                 require=["KMeans", "silhouette_score"], tol=1e-9)
    if task == "db":
        return P(f"{head} and return the `k` with the **lowest** `davies_bouldin_score` (lower is better for this "
                 f"index).", setup,
                 f"{KM}{MET}davies_bouldin_score\nscores = {{k: davies_bouldin_score(X, {km}) for k in {ks}}}\n"
                 "result = min(scores, key=scores.get)", require=["KMeans", "davies_bouldin_score"])
    if task == "ch":
        return P(f"{head} and return the `k` with the highest `calinski_harabasz_score`.", setup,
                 f"{KM}{MET}calinski_harabasz_score\nscores = {{k: calinski_harabasz_score(X, {km}) for k in {ks}}}\n"
                 "result = max(scores, key=scores.get)", require=["KMeans", "calinski_harabasz_score"])
    return P(f"{head}, pick the `k` with the best `silhouette_score`, and return `(k, ari)`: that `k` and the "
             f"`adjusted_rand_score` of its clustering against `y_true`.", setup,
             f"{KM}{MET}silhouette_score, adjusted_rand_score\nlab = {{k: {km} for k in {ks}}}\n"
             "best = max(lab, key=lambda k: silhouette_score(X, lab[k]))\n"
             "result = (best, adjusted_rand_score(y_true, lab[best]))",
             require=["KMeans", "silhouette_score", "adjusted_rand_score"], tol=1e-9)


_ITEMS = {"films": ["alien", "brazil", "casablanca", "dune", "fargo", "gravity", "heat", "inception", "jaws", "kes",
                    "lucy", "memento"],
          "products": ["kettle", "toaster", "blender", "mixer", "grill", "fryer", "juicer", "scale", "oven", "wok",
                       "teapot", "sieve"],
          "songs": ["aria", "blues", "canon", "dirge", "etude", "fugue", "gigue", "hymn", "jig", "lullaby", "march",
                    "nocturne"]}


@template(M, "medium")
def most_similar_items(rng):
    kind = rng.choice(sorted(_ITEMS))
    n, d, s = rng.randint(6, 10), rng.randint(3, 6), seeded(rng)
    names = _ITEMS[kind][:n]
    setup = (f"names = {lit(names)}\ng = np.random.default_rng({s})\n"
             f"E = (g.normal(size=({n}, {d})) * g.uniform(0.5, 3, ({n}, 1))).round(2)")
    desc = f"`E` holds a {d}-D embedding for each of {n} {kind} (row `i` is `names[i]`)"
    q = rng.randrange(n)
    pw = "from sklearn.metrics.pairwise import "
    task = rng.choice(["pair", "far", "top", "both", "central"])
    if task == "pair":
        return P(f"{desc}. Which two {kind} are most alike? Return the pair with the highest `cosine_similarity` as a "
                 f"tuple of names `(names[i], names[j])` with `i < j`.", setup,
                 f"{pw}cosine_similarity\nS = cosine_similarity(E)\nS[np.tril_indices(len(E))] = -np.inf\n"
                 "i, j = np.unravel_index(S.argmax(), S.shape)\nresult = (names[i], names[j])",
                 require=["cosine_similarity"])
    if task == "far":
        return P(f"{desc}. Return the two {kind} that point in the most **opposite** directions (lowest cosine "
                 f"similarity) as a tuple of names `(names[i], names[j])` with `i < j`.", setup,
                 f"{pw}cosine_similarity\nS = cosine_similarity(E)\nS[np.tril_indices(len(E))] = np.inf\n"
                 "i, j = np.unravel_index(S.argmin(), S.shape)\nresult = (names[i], names[j])",
                 require=["cosine_similarity"])
    if task == "top":
        return P(f"{desc}. Recommend: return the names of the 3 {kind} most similar to {names[q]!r} by cosine "
                 f"similarity, most similar first (not {names[q]!r} itself).", setup,
                 f"{pw}cosine_similarity\nsim = cosine_similarity(E[[{q}]], E)[0]\nsim[{q}] = -np.inf\n"
                 "result = [names[i] for i in np.argsort(-sim)[:3]]", require=["cosine_similarity"])
    if task == "both":
        return P(f"{desc}. Rows have very different lengths, so the metric matters. Return `(by_cosine, "
                 f"by_euclidean)`: the name of the nearest other item to {names[q]!r} by highest `cosine_similarity` "
                 f"and by smallest `euclidean_distances`.", setup,
                 f"{pw}cosine_similarity, euclidean_distances\ncs = cosine_similarity(E[[{q}]], E)[0]\n"
                 f"ed = euclidean_distances(E[[{q}]], E)[0]\ncs[{q}], ed[{q}] = -np.inf, np.inf\n"
                 "result = (names[int(cs.argmax())], names[int(ed.argmin())])",
                 require=["cosine_similarity", "euclidean_distances"])
    return P(f"{desc}. Which item is most 'central'? Return the name with the highest **average** cosine similarity "
             f"to the **other** items (leave out the 1 on the diagonal).", setup,
             f"{pw}cosine_similarity\nS = cosine_similarity(E)\nnp.fill_diagonal(S, 0)\n"
             "result = names[int((S.sum(axis=1) / (len(E) - 1)).argmax())]", require=["cosine_similarity"])


@template(M, "medium")
def clusters_vs_classes(rng):
    data = rng.choice(["iris", "wine"])
    loader = "load_iris" if data == "iris" else "load_wine"
    setup = f"from sklearn.datasets import {loader}\nX, y = {loader}(return_X_y=True)"
    desc = (f"`X` holds {'150 iris flowers (4 measurements)' if data == 'iris' else '178 wines (13 chemical measurements)'}"
            f" and `y` their {'species' if data == 'iris' else 'cultivar'} (0, 1, 2)")
    r = rng.randint(0, 9)
    km = f"KMeans(n_clusters=3, n_init='auto', random_state={r})"
    sc = "from sklearn.preprocessing import StandardScaler\n"
    task = rng.choice(["scale", "nmi", "homog", "purity", "disagree"])
    if task == "scale":
        return P(f"{desc}. Does scaling help KMeans find the classes? Cluster with `{km}` on raw `X` and on "
                 f"`StandardScaler().fit_transform(X)`; return both `adjusted_rand_score`s against `y` as `(raw, "
                 f"scaled)`.", setup,
                 f"{KM}{sc}{MET}adjusted_rand_score\nraw = {km}.fit_predict(X)\n"
                 f"scaled = {km}.fit_predict(StandardScaler().fit_transform(X))\n"
                 "result = (adjusted_rand_score(y, raw), adjusted_rand_score(y, scaled))",
                 require=["KMeans", "adjusted_rand_score"], tol=1e-9)
    if task == "nmi":
        return P(f"{desc}. Standardize `X`, cluster it with `{km}` and return the `normalized_mutual_info_score` "
                 f"between `y` and the clusters.", setup,
                 f"{KM}{sc}{MET}normalized_mutual_info_score\n"
                 f"lab = {km}.fit_predict(StandardScaler().fit_transform(X))\nresult = normalized_mutual_info_score(y, lab)",
                 require=["KMeans", "normalized_mutual_info_score"], tol=1e-9)
    if task == "homog":
        return P(f"{desc}. More clusters make each one purer. For `k` in 2..6 cluster the standardized `X` with "
                 f"`KMeans(n_clusters=k, n_init='auto', random_state={r})` and return the list of `homogeneity_score`s "
                 f"against `y`.", setup,
                 f"{KM}{sc}{MET}homogeneity_score\nZ = StandardScaler().fit_transform(X)\n"
                 f"result = [homogeneity_score(y, KMeans(n_clusters=k, n_init='auto', random_state={r}).fit_predict(Z)) "
                 "for k in range(2, 7)]", require=["KMeans", "homogeneity_score"], tol=1e-9)
    if task == "purity":
        return P(f"{desc}. Cluster the standardized `X` with `{km}`. Label each cluster with its most common class "
                 f"and return the fraction of samples whose class matches their cluster's label (the clustering's "
                 f"'purity').", setup,
                 f"{KM}{sc}lab = {km}.fit_predict(StandardScaler().fit_transform(X))\n"
                 "result = sum(np.bincount(y[lab == c]).max() for c in np.unique(lab)) / len(y)",
                 require=["KMeans"], tol=1e-9)
    return P(f"{desc}. Do the unsupervised and supervised scores agree? Cluster the standardized `X` with "
             f"`KMeans(n_clusters=k, n_init='auto', random_state={r})` for `k` in 2, 3, 4, 5 and return `(k_best_"
             f"silhouette, k_best_ari)`.", setup,
             f"{KM}{sc}{MET}silhouette_score, adjusted_rand_score\nZ = StandardScaler().fit_transform(X)\n"
             f"lab = {{k: KMeans(n_clusters=k, n_init='auto', random_state={r}).fit_predict(Z) for k in range(2, 6)}}\n"
             "result = (max(lab, key=lambda k: silhouette_score(Z, lab[k])), max(lab, key=lambda k: adjusted_rand_score(y, lab[k])))",
             require=["silhouette_score", "adjusted_rand_score"])


@template(M, "medium")
def ranking_quality_ndcg(rng):
    q, m, s = rng.randint(3, 6), rng.randint(5, 8), seeded(rng)
    setup = (f"g = np.random.default_rng({s})\nrelevance = g.integers(0, 4, ({q}, {m}))\nrelevance[:, 0] = 3\n"
             f"ranker_a = (relevance + g.normal(0, 1.0, ({q}, {m}))).round(3)\n"
             f"ranker_b = (relevance + g.normal(0, 1.6, ({q}, {m}))).round(3)")
    desc = (f"Two search rankers scored the same {q} queries × {m} documents: `relevance` holds the true grades "
            f"(0-3) and `ranker_a`, `ranker_b` the rankers' scores (one row per query)")
    k = rng.randint(2, 4)
    task = rng.choice(["both", "atk", "per", "worst", "winner"])
    if task == "both":
        return P(f"{desc}. Return `(ndcg_a, ndcg_b)` with `ndcg_score`.", setup,
                 f"{MET}ndcg_score\nresult = (ndcg_score(relevance, ranker_a), ndcg_score(relevance, ranker_b))",
                 require=["ndcg_score"], tol=1e-9)
    if task == "atk":
        return P(f"{desc}. Users only look at the top {k}. Return `(ndcg_a, ndcg_b)` at `k={k}`.", setup,
                 f"{MET}ndcg_score\nresult = (ndcg_score(relevance, ranker_a, k={k}), ndcg_score(relevance, ranker_b, k={k}))",
                 require=["ndcg_score"], tol=1e-9)
    if task == "per":
        return P(f"{desc}. `ndcg_score` averages over queries. Return the NDCG of ranker A **per query** as a list "
                 f"(pass one row at a time, kept 2-D: `relevance[[i]]`).", setup,
                 f"{MET}ndcg_score\nresult = [ndcg_score(relevance[[i]], ranker_a[[i]]) for i in range(len(relevance))]",
                 require=["ndcg_score"], tol=1e-9)
    if task == "worst":
        return P(f"{desc}. On which query does ranker B do worst? Return the index of the query with the lowest "
                 f"per-query NDCG@{k}.", setup,
                 f"{MET}ndcg_score\nper = [ndcg_score(relevance[[i]], ranker_b[[i]], k={k}) for i in range(len(relevance))]\n"
                 "result = int(np.argmin(per))", require=["ndcg_score"])
    return P(f"{desc}. Return `(full_a - full_b, top1_a - top1_b)`: the NDCG difference between the rankers over the "
             f"whole list and at `k=1`.", setup,
             f"{MET}ndcg_score\nresult = (ndcg_score(relevance, ranker_a) - ndcg_score(relevance, ranker_b), "
             "ndcg_score(relevance, ranker_a, k=1) - ndcg_score(relevance, ranker_b, k=1))",
             require=["ndcg_score"], tol=1e-9)


@template(M, "medium")
def kernel_matrix_tasks(rng):
    n, d, s = rng.randint(5, 9), rng.randint(2, 5), seeded(rng)
    setup = f"X = np.random.default_rng({s}).normal(size=({n}, {d})).round(2)"
    pw = "from sklearn.metrics.pairwise import "
    gammas = sorted(rng.sample([0.01, 0.1, 0.5, 1.0, 5.0], 3))
    items = [
        ("rbf_kernel", f"`gamma` sets how fast similarity decays. For each `gamma` in `{gammas}` compute "
                       f"`rbf_kernel(X, gamma=gamma)` and record the mean of its **off-diagonal** entries; return the "
                       f"list.",
         f"{pw}rbf_kernel\nmask = ~np.eye(len(X), dtype=bool)\nresult = [rbf_kernel(X, gamma=g)[mask].mean() for g in {gammas}]"),
        ("rbf_kernel", "By default `rbf_kernel` uses `gamma = 1 / n_features`. Rebuild `rbf_kernel(X)` from "
                       "`euclidean_distances(X)` with that gamma and return it.",
         f"{pw}euclidean_distances\nresult = np.exp(-(1 / X.shape[1]) * euclidean_distances(X) ** 2)",
         {"require": ["euclidean_distances"]}),
        ("linear_kernel", "The linear kernel of `X` with itself is the Gram matrix `X @ X.T`; its trace is the sum of "
                          "squared row norms. Return the trace of `linear_kernel(X)`.",
         f"{pw}linear_kernel\nresult = np.trace(linear_kernel(X))"),
        ("linear_kernel", "Distances from a kernel: ‖a − b‖² = k(a, a) + k(b, b) − 2·k(a, b). Turn `K = "
                          "linear_kernel(X)` into the Euclidean distance matrix (clip tiny negatives at 0 before the "
                          "square root) and return it.",
         f"{pw}linear_kernel\nK = linear_kernel(X)\nq = np.diag(K)\n"
         "result = np.sqrt(np.maximum(q[:, None] + q[None, :] - 2 * K, 0))"),
        ("rbf_kernel", "Return the pair of different rows `(i, j)`, `i < j`, with the largest `rbf_kernel(X, "
                       "gamma=0.5)` value (it is also the closest pair in Euclidean distance).",
         f"{pw}rbf_kernel\nK = rbf_kernel(X, gamma=0.5)\nK[np.tril_indices(len(X))] = -1\n"
         "i, j = np.unravel_index(K.argmax(), K.shape)\nresult = (int(i), int(j))"),
        ("cosine_similarity", "Cosine similarity of **centred** rows is the Pearson correlation between rows. Subtract "
                              "each row's own mean from it and return `cosine_similarity` of the result (it equals "
                              "`np.corrcoef(X)`).",
         f"{pw}cosine_similarity\nresult = cosine_similarity(X - X.mean(axis=1, keepdims=True))"),
        ("manhattan_distances", "Which row is the most 'typical' under the L1 metric? Return the index of the row with "
                                "the smallest total Manhattan distance to all rows (`manhattan_distances`).",
         f"{pw}manhattan_distances\nresult = int(manhattan_distances(X).sum(axis=1).argmin())"),
    ]
    return catalog(rng, items, setup, f"`X` holds {n} random rows with {d} features.", tol=1e-9)


@template(M, "medium")
def cluster_quality_tasks(rng):
    k = rng.randint(2, 4)
    setup, desc, _ = _blobs(rng, k=k, std=rng.choice([1.0, 1.5, 2.0]), labels=k)
    m = rng.choice([40, 50])
    items = [
        ("silhouette_samples", "Return the mean silhouette coefficient of each cluster as a dict `{cluster: mean}`.",
         f"{MET}silhouette_samples\ns = silhouette_samples(X, labels)\n"
         "result = {int(c): float(s[labels == c].mean()) for c in np.unique(labels)}"),
        ("silhouette_samples", "Return how many points of **each** cluster have a negative silhouette coefficient, as "
                               "a list in cluster order.",
         f"{MET}silhouette_samples\ns = silhouette_samples(X, labels)\n"
         "result = [int((s[labels == c] < 0).sum()) for c in np.unique(labels)]"),
        ("davies_bouldin_score", f"Would one more cluster help? Fit `KMeans(n_clusters={k + 1}, n_init='auto', "
                                 f"random_state=0)` too and return `(db_labels, db_more)` (lower is better).",
         f"{KM}{MET}davies_bouldin_score\nmore = KMeans(n_clusters={k + 1}, n_init='auto', random_state=0).fit_predict(X)\n"
         "result = (davies_bouldin_score(X, labels), davies_bouldin_score(X, more))", {"require": ["KMeans", "davies_bouldin_score"]}),
        ("calinski_harabasz_score", "Internal indices can rate the truth too: return `(ch_labels, ch_truth)`, the "
                                    "`calinski_harabasz_score` of `labels` and of `y_true`.",
         f"{MET}calinski_harabasz_score\nresult = (calinski_harabasz_score(X, labels), calinski_harabasz_score(X, y_true))"),
        ("adjusted_rand_score", f"Is KMeans stable? Refit `KMeans(n_clusters={k}, n_init=1, random_state=s)` for `s` "
                                f"in 1, 2, 3 and return the list of `adjusted_rand_score`s between each run and "
                                f"`labels`.",
         f"{KM}{MET}adjusted_rand_score\n"
         f"result = [adjusted_rand_score(labels, KMeans(n_clusters={k}, n_init=1, random_state=s).fit_predict(X)) for s in (1, 2, 3)]",
         {"require": ["KMeans", "adjusted_rand_score"]}),
        ("silhouette_score", "Return the silhouette score of `labels` under three metrics, as a dict keyed by "
                             "`'euclidean'`, `'manhattan'`, `'cosine'`.",
         f"{MET}silhouette_score\nresult = {{m: silhouette_score(X, labels, metric=m) for m in ('euclidean', 'manhattan', 'cosine')}}"),
        ("normalized_mutual_info_score", "Return `(nmi, homogeneity)` of `labels` against `y_true`.",
         f"{MET}normalized_mutual_info_score, homogeneity_score\n"
         "result = (normalized_mutual_info_score(y_true, labels), homogeneity_score(y_true, labels))"),
        ("silhouette_score", f"On big data the silhouette is estimated on a sample. Return `(full, sampled)`: the exact "
                             f"score and `silhouette_score(X, labels, sample_size={m}, random_state=0)`.",
         f"{MET}silhouette_score\nresult = (silhouette_score(X, labels), silhouette_score(X, labels, sample_size={m}, random_state=0))"),
    ]
    return catalog(rng, items, setup, f"{desc}.", tol=1e-9)


@template(M, "medium")
def metric_behaviour(rng):
    setup, yt, yp = _pair(rng, n=(7, 10))
    big = round(max(yt) * rng.choice([2, 3, 5]), 1)
    f = rng.choice([10, 100, 1000])
    w = [round(rng.uniform(0.5, 3), 1) for _ in yt]
    items = [
        ("root_mean_squared_error", f"One prediction goes badly wrong: `y_bad = y_hat.copy(); y_bad[0] += {big}`. "
                                    f"Return how much each metric grows, as `(mae_ratio, rmse_ratio, medae_ratio)` "
                                    f"(metric on `y_bad` / metric on `y_hat`). RMSE punishes the outlier most; the "
                                    f"median barely moves.",
         f"{MET}mean_absolute_error, root_mean_squared_error, median_absolute_error\ny_bad = y_hat.copy()\n"
         f"y_bad[0] += {big}\nresult = tuple(fn(y_true, y_bad) / fn(y_true, y_hat) for fn in "
         "(mean_absolute_error, root_mean_squared_error, median_absolute_error))",
         {"require": ["mean_absolute_error", "root_mean_squared_error", "median_absolute_error"]}),
        ("r2_score", f"Change the units: multiply both arrays by {f}. Return `(rmse, r2)` of the rescaled arrays: RMSE "
                     f"scales with the units, R² does not.",
         f"{MET}root_mean_squared_error, r2_score\n"
         f"result = (root_mean_squared_error({f} * y_true, {f} * y_hat), r2_score({f} * y_true, {f} * y_hat))",
         {"require": ["root_mean_squared_error", "r2_score"]}),
        ("mean_absolute_percentage_error", f"MAPE is relative: return `(mape, mape_shifted)` where the shifted version "
                                           f"adds {f} to **both** arrays (same absolute errors, bigger targets).",
         f"{MET}mean_absolute_percentage_error\nresult = (mean_absolute_percentage_error(y_true, y_hat), "
         f"mean_absolute_percentage_error(y_true + {f}, y_hat + {f}))"),
        ("mean_absolute_error", "Two targets at once: `Y = np.column_stack([y_true, 2 * y_true])` and `Y_hat = "
                                "np.column_stack([y_hat, 2 * y_true + 1])`. Return the MAE **per target** "
                                "(`multioutput='raw_values'`).",
         f"{MET}mean_absolute_error\nY = np.column_stack([y_true, 2 * y_true])\nY_hat = np.column_stack([y_hat, 2 * y_true + 1])\n"
         "result = mean_absolute_error(Y, Y_hat, multioutput='raw_values')"),
        ("mean_squared_error", "Some rows matter more: return the MSE weighted by `w` (`sample_weight=w`).",
         f"{MET}mean_squared_error\nresult = mean_squared_error(y_true, y_hat, sample_weight=w)",
         {"setup": f"w = np.array({lit(w)})"}),
        ("r2_score", "Two targets: `Y = np.column_stack([y_true, y_true ** 2])`, `Y_hat = np.column_stack([y_hat, "
                     "y_hat ** 2])`. Return `(uniform, weighted)`: R² with `multioutput='uniform_average'` and with "
                     "`'variance_weighted'` (the high-variance target dominates).",
         f"{MET}r2_score\nY = np.column_stack([y_true, y_true ** 2])\nY_hat = np.column_stack([y_hat, y_hat ** 2])\n"
         "result = (r2_score(Y, Y_hat, multioutput='uniform_average'), r2_score(Y, Y_hat, multioutput='variance_weighted'))"),
    ]
    return catalog(rng, items, setup, "`y_true` holds true targets and `y_hat` a model's predictions.", tol=1e-9)


# ================================================================= hard

@template(M, "hard")
def silhouette_from_scratch(rng):
    k = rng.randint(2, 4)
    setup, desc, _ = _blobs(rng, k=k, n=(40, 90), std=rng.choice([1.0, 1.5, 2.0]), labels=k)
    i = rng.randint(0, 39)
    task = rng.choice(["all", "one", "manhattan"])
    rule = ("For point i: a = mean distance to the **other** points of its own cluster, b = the smallest mean distance "
            "to the points of another cluster, s = (b − a) / max(a, b).")
    body = ("n = len(X)\ncnt = np.bincount(labels)\n"
            "M = np.column_stack([D[:, labels == c].sum(axis=1) for c in range(len(cnt))])\n"
            "a = M[np.arange(n), labels] / (cnt[labels] - 1)\nM = M / cnt\nM[np.arange(n), labels] = np.inf\n"
            "b = M.min(axis=1)\ns = (b - a) / np.maximum(a, b)\n")
    if task == "all":
        return P(f"{desc} (every cluster has several points). {rule} Compute every point's silhouette **by hand** from "
                 f"`pairwise_distances(X)` and return the array (it matches `silhouette_samples`).", setup,
                 f"{MET}pairwise_distances\nD = pairwise_distances(X)\n{body}result = s",
                 require=["pairwise_distances"], tol=1e-9)
    if task == "one":
        return P(f"{desc}. {rule} Return `(a, b, s)` for point {i} only, computed from its distances to all points "
                 f"(`euclidean_distances(X[[{i}]], X)`).", setup,
                 f"from sklearn.metrics.pairwise import euclidean_distances\nd = euclidean_distances(X[[{i}]], X)[0]\n"
                 f"own = labels == labels[{i}]\na = d[own].sum() / (own.sum() - 1)\n"
                 f"b = min(d[labels == c].mean() for c in np.unique(labels) if c != labels[{i}])\n"
                 "result = (a, b, (b - a) / max(a, b))", require=["euclidean_distances"], tol=1e-9)
    return P(f"{desc}. {rule} With **Manhattan** distances (`pairwise_distances(X, metric='manhattan')`), compute the "
             f"mean silhouette by hand; it matches `silhouette_score(X, labels, metric='manhattan')`.", setup,
             f"{MET}pairwise_distances\nD = pairwise_distances(X, metric='manhattan')\n{body}result = s.mean()",
             require=["pairwise_distances"], tol=1e-9)


def _two_labelings(rng):
    n, k1, k2 = rng.randint(15, 40), rng.randint(2, 4), rng.randint(2, 5)
    a = [rng.randrange(k1) for _ in range(n)]
    b = [x if rng.random() < 0.7 else rng.randrange(k2) for x in a]
    return f"y_true = np.array({lit(a)})\nlabels = np.array({lit(b)})"


@template(M, "hard")
def rand_index_by_hand(rng):
    setup = _two_labelings(rng)
    head = ("`y_true` and `labels` are two partitions of the same points. Build the contingency table `C` (`C[i, j]` = "
            "points with true group `i` and cluster `j`). With pairs(v) = v·(v − 1)/2: `same_both` = Σ pairs(C), "
            "`same_true` = Σ pairs(row sums), `same_pred` = Σ pairs(column sums), `total` = pairs(n).")
    base = ("C = np.zeros((y_true.max() + 1, labels.max() + 1))\nnp.add.at(C, (y_true, labels), 1)\n"
            "pairs = lambda v: v * (v - 1) / 2\nboth, rt, rp = pairs(C).sum(), pairs(C.sum(axis=1)).sum(), pairs(C.sum(axis=0)).sum()\n"
            "total = pairs(len(y_true))\n")
    if rng.random() < 0.5:
        return P(f"{head} The adjusted Rand index is (same_both − E) / ((same_true + same_pred)/2 − E) with E = "
                 f"same_true·same_pred/total. Compute it by hand; it equals `adjusted_rand_score(y_true, labels)`.",
                 setup, base + "E = rt * rp / total\nresult = (both - E) / ((rt + rp) / 2 - E)", tol=1e-9)
    return P(f"{head} The plain Rand index is the fraction of point pairs on which both partitions agree (together in "
             f"both, or apart in both): (total + 2·same_both − same_true − same_pred) / total. Compute it by hand; it "
             f"equals `rand_score(y_true, labels)` from `sklearn.metrics`.", setup,
             base + "result = (total + 2 * both - rt - rp) / total", tol=1e-9)


@template(M, "hard")
def entropy_scores_by_hand(rng):
    setup = _two_labelings(rng)
    base = ("C = np.zeros((y_true.max() + 1, labels.max() + 1))\nnp.add.at(C, (y_true, labels), 1)\n"
            "C = C[C.sum(axis=1) > 0][:, C.sum(axis=0) > 0] / len(y_true)\npt, pc = C.sum(axis=1), C.sum(axis=0)\n"
            "H = lambda p: -(p[p > 0] * np.log(p[p > 0])).sum()\n")
    task = rng.choice(["nmi", "homog", "vmeasure"])
    head = ("`y_true` and `labels` are two partitions of the same points. Work with the joint distribution `C / n` of "
            "(true group, cluster) and entropies H(p) = −Σ p·log p.")
    if task == "nmi":
        return P(f"{head} The mutual information is I = Σ p_ij·log(p_ij / (p_i·p_j)). Return I / ((H(true) + "
                 f"H(clusters)) / 2) computed by hand; it equals `normalized_mutual_info_score(y_true, labels)`.",
                 setup, base + "nz = C > 0\nI = (C[nz] * np.log(C[nz] / np.outer(pt, pc)[nz])).sum()\n"
                               "result = I / ((H(pt) + H(pc)) / 2)", tol=1e-9)
    if task == "homog":
        return P(f"{head} Homogeneity is 1 − H(true | cluster) / H(true), with H(true | cluster) = H(joint) − "
                 f"H(clusters). Compute it by hand; it equals `homogeneity_score(y_true, labels)`.", setup,
                 base + "result = 1 - (H(C.ravel()) - H(pc)) / H(pt)", tol=1e-9)
    return P(f"{head} Homogeneity = 1 − H(true | cluster)/H(true), completeness = 1 − H(cluster | true)/H(cluster) "
             f"(conditional entropy = joint entropy − entropy of the condition). Return their harmonic mean (the "
             f"V-measure, `v_measure_score`), computed by hand.", setup,
             base + "h = 1 - (H(C.ravel()) - H(pc)) / H(pt)\nc = 1 - (H(C.ravel()) - H(pt)) / H(pc)\n"
                    "result = 2 * h * c / (h + c)", tol=1e-9)


@template(M, "hard")
def internal_indices_by_hand(rng):
    k = rng.randint(2, 4)
    setup, desc, _ = _blobs(rng, k=k, n=(40, 120), labels=k)
    base = "ks = np.unique(labels)\ncent = np.array([X[labels == c].mean(axis=0) for c in ks])\n"
    if rng.random() < 0.5:
        return P(f"{desc}. Compute the Davies-Bouldin index **by hand**: for each cluster, s_i = mean distance of its "
                 f"points to its centroid; R_ij = (s_i + s_j) / ‖c_i − c_j‖; the index is the mean over clusters of "
                 f"max_(j≠i) R_ij. It equals `davies_bouldin_score(X, labels)`.", setup,
                 base + "s = np.array([np.linalg.norm(X[labels == c] - cent[i], axis=1).mean() for i, c in enumerate(ks)])\n"
                        "dist = np.linalg.norm(cent[:, None] - cent[None], axis=2)\nnp.fill_diagonal(dist, np.inf)\n"
                        "result = ((s[:, None] + s[None]) / dist).max(axis=1).mean()", tol=1e-9)
    return P(f"{desc}. Compute the Calinski-Harabasz index **by hand**: B = Σ n_c·‖c_c − mean‖² (between clusters), W = "
             f"Σ ‖x − c_label(x)‖² (within), index = (B / (k − 1)) / (W / (n − k)). It equals "
             f"`calinski_harabasz_score(X, labels)`.", setup,
             base + "n, kk = len(X), len(ks)\nB = sum((labels == c).sum() * ((cent[i] - X.mean(axis=0)) ** 2).sum() for i, c in enumerate(ks))\n"
                    "W = sum(((X[labels == c] - cent[i]) ** 2).sum() for i, c in enumerate(ks))\n"
                    "result = (B / (kk - 1)) / (W / (n - kk))", tol=1e-9)


@template(M, "hard")
def ndcg_by_hand(rng):
    m, s = rng.randint(5, 9), seeded(rng)
    k = rng.randint(2, m - 2)
    setup = (f"g = np.random.default_rng({s})\nrel = g.integers(0, 4, {m})\nrel[g.integers({m})] = 3\n"
             f"score = g.permutation({m}) / {m}   # distinct scores: no ties")
    head = (f"One query: `rel` holds the relevance of {m} documents and `score` the model's scores (higher = shown "
            f"earlier). DCG = Σ rel[doc at rank r] / log2(r + 1) for ranks r = 1, 2, ...")
    task = rng.choice(["ndcg", "atk", "dcg"])
    if task == "ndcg":
        return P(f"{head} NDCG = DCG / DCG of the ideal order (relevances sorted descending). Compute it **by hand**; "
                 f"it equals `ndcg_score([rel], [score])`.", setup,
                 f"disc = 1 / np.log2(np.arange(2, {m} + 2))\ndcg = (rel[np.argsort(-score)] * disc).sum()\n"
                 "ideal = (np.sort(rel)[::-1] * disc).sum()\nresult = dcg / ideal", tol=1e-9)
    if task == "atk":
        return P(f"{head} NDCG@{k} keeps only the first {k} ranks in both the DCG and the ideal DCG. Compute it **by "
                 f"hand**; it equals `ndcg_score([rel], [score], k={k})`.", setup,
                 f"disc = 1 / np.log2(np.arange(2, {k} + 2))\ndcg = (rel[np.argsort(-score)][:{k}] * disc).sum()\n"
                 f"ideal = (np.sort(rel)[::-1][:{k}] * disc).sum()\nresult = dcg / ideal", tol=1e-9)
    return P(f"{head} Compute the DCG **by hand** and return `(by_hand, dcg_score([rel], [score]))` (`dcg_score` is in "
             f"`sklearn.metrics`).", setup,
             f"{MET}dcg_score\ndisc = 1 / np.log2(np.arange(2, {m} + 2))\n"
             "result = ((rel[np.argsort(-score)] * disc).sum(), dcg_score([rel], [score]))", require=["dcg_score"],
             tol=1e-9)


@template(M, "hard")
def metric_pitfalls(rng):
    setup, yt, yp = _pair(rng, n=(7, 10))
    k = rng.randint(2, 3)
    s = seeded(rng)
    n_neg = rng.randint(1, 2)
    neg = ", ".join(str(round(-rng.uniform(1.5, 4), 1)) for _ in range(n_neg))
    items = [
        ("root_mean_squared_error", "This line worked in old scikit-learn but raises `TypeError` in 1.8: "
                                    "`mean_squared_error(y_true, y_hat, squared=False)`. Fix it (RMSE has its own "
                                    "function now) and return the RMSE.",
         f"{MET}root_mean_squared_error\nresult = root_mean_squared_error(y_true, y_hat)"),
        ("mean_squared_log_error", f"A count model predicted negative values, and `mean_squared_log_error(y_true, "
                                   f"y_neg)` raises `ValueError` (logs of values ≤ −1). `y_neg` is `y_hat` with its last "
                                   f"{n_neg} value(s) replaced by {neg}. Counts can't be negative: clip the predictions "
                                   f"at 0 (`np.clip`) and return the MSLE.",
         f"{MET}mean_squared_log_error\nresult = mean_squared_log_error(y_true, np.clip(y_neg, 0, None))",
         {"setup": f"y_neg = y_hat.copy()\ny_neg[-{n_neg}:] = [{neg}]"}),
        ("mean_absolute_percentage_error", "`y_zero` is `y_true` with its first value set to 0, and MAPE explodes "
                                           "(it divides by |y_true|). Return `(exploded, fixed)`: the MAPE of `y_hat` "
                                           "against `y_zero`, and the MAPE computed only on rows where `y_zero != 0`.",
         f"{MET}mean_absolute_percentage_error\nok = y_zero != 0\nresult = (mean_absolute_percentage_error(y_zero, y_hat), "
         "mean_absolute_percentage_error(y_zero[ok], y_hat[ok]))", {"setup": "y_zero = y_true.copy()\ny_zero[0] = 0"}),
        ("r2_score", "A colleague reports `r2_score(y_hat, y_true)`. Return `(reported, correct)`: their number and the "
                     "model's real R² (the true values go **first**).",
         f"{MET}r2_score\nresult = (r2_score(y_hat, y_true), r2_score(y_true, y_hat))"),
        ("adjusted_rand_score", f"`clusters` is a perfect clustering of `groups` with shuffled cluster numbers, yet "
                                f"`(clusters == groups).mean()` looks terrible. Return `(accuracy, ari)` to show "
                                f"the right comparison.",
         f"{MET}adjusted_rand_score\nresult = (float((clusters == groups).mean()), adjusted_rand_score(groups, clusters))",
         {"setup": f"groups = np.random.default_rng({s}).integers(0, {k + 1}, 20)\n"
                   f"clusters = (groups + 1) % {k + 1}"}),
        ("euclidean_distances", "`euclidean_distances(u, v)` raises `ValueError` because `u` and `v` are 1-D vectors "
                                "(it expects 2-D arrays of rows). Fix the call and return the distance as a float.",
         "from sklearn.metrics.pairwise import euclidean_distances\nresult = float(euclidean_distances([u], [v])[0, 0])",
         {"setup": f"u = y_true[:3]\nv = y_hat[:3]"}),
        ("silhouette_score", f"This loop crashes on its first step: `[silhouette_score(Xb, KMeans(n_clusters=k, "
                             f"n_init='auto', random_state=0).fit_predict(Xb)) for k in range(1, 6)]` (a silhouette "
                             f"needs at least 2 clusters). Fix the range so it starts at 2 and return the scores.",
         f"{KM}{MET}silhouette_score\nresult = [silhouette_score(Xb, KMeans(n_clusters=k, n_init='auto', random_state=0)"
         ".fit_predict(Xb)) for k in range(2, 6)]",
         {"setup": f"from sklearn.datasets import make_blobs\nXb, _ = make_blobs(n_samples=80, centers={k + 1}, "
                   f"random_state={s})"}),
        ("StandardScaler", f"`Xs` has one feature in metres and one in millimetres, so KMeans only sees the second. "
                           f"Return `(ari_raw, ari_scaled)` of `KMeans(n_clusters={k}, n_init='auto', random_state=0)` "
                           f"against `ys` on raw `Xs` and on `StandardScaler().fit_transform(Xs)`.",
         f"{KM}from sklearn.preprocessing import StandardScaler\n{MET}adjusted_rand_score\n"
         f"km = KMeans(n_clusters={k}, n_init='auto', random_state=0)\n"
         "result = (adjusted_rand_score(ys, km.fit_predict(Xs)), adjusted_rand_score(ys, km.fit_predict(StandardScaler().fit_transform(Xs))))",
         {"setup": f"from sklearn.datasets import make_blobs\nXs, ys = make_blobs(n_samples=90, centers={k}, n_features=2, "
                   f"cluster_std=0.5, random_state={s})\nXs = Xs * np.array([1.0, 0.0]) + "
                   f"np.random.default_rng({s}).normal(0, 1, (90, 2)) * np.array([0.0, 1000.0])"}),
    ]
    return catalog(rng, items, setup, "`y_true` holds true targets and `y_hat` a model's predictions.", tol=1e-9)


@template(M, "hard")
def kmeans_internals(rng):
    k = rng.randint(2, 4)
    setup, desc, _ = _blobs(rng, k=k, n=(60, 150))
    setup += (f"\nfrom sklearn.cluster import KMeans\nkm = KMeans(n_clusters={k}, n_init='auto', random_state=0).fit(X)\n"
              f"X_new = np.random.default_rng({seeded(rng)}).uniform(X.min(axis=0), X.max(axis=0), (5, X.shape[1])).round(2)")
    desc += f"; `km = KMeans(n_clusters={k}, n_init='auto', random_state=0)` was fitted on it and `X_new` holds 5 new points"
    pw = "from sklearn.metrics.pairwise import euclidean_distances\n"
    task = rng.choice(["inertia", "predict", "nearest", "lloyd"])
    if task == "inertia":
        return P(f"{desc}. `km.inertia_` is the sum of squared distances from each point to its closest centre. "
                 f"Recompute it from `euclidean_distances(X, km.cluster_centers_)`.", setup,
                 pw + "result = (euclidean_distances(X, km.cluster_centers_).min(axis=1) ** 2).sum()",
                 require=["euclidean_distances", "cluster_centers_"], tol=1e-6)
    if task == "predict":
        return P(f"{desc}. Reproduce `km.predict(X_new)` without calling it: each point goes to the centre with the "
                 f"smallest Euclidean distance.", setup,
                 pw + "result = euclidean_distances(X_new, km.cluster_centers_).argmin(axis=1)",
                 require=["euclidean_distances", "cluster_centers_"])
    if task == "nearest":
        return P(f"{desc}. `km.transform` returns the distance to every centre. Return, for each row of `X_new`, the "
                 f"distance to its **nearest** centre, and check it matches `euclidean_distances`: return "
                 f"`(from_transform, from_euclidean)`.", setup,
                 pw + "result = (km.transform(X_new).min(axis=1), euclidean_distances(X_new, km.cluster_centers_).min(axis=1))",
                 require=["transform", "euclidean_distances"], tol=1e-6)
    return P(f"{desc}. Do **one** Lloyd iteration by hand from the starting centres `C0 = X[:{k}]`: assign every point "
             f"to its nearest centre (`euclidean_distances`), then move each centre to the mean of its points. Return "
             f"the new {k} × d centre array (it is what `KMeans(n_clusters={k}, init=C0, n_init=1, max_iter=1)` would "
             f"learn).", setup,
             pw + f"C0 = X[:{k}]\nassign = euclidean_distances(X, C0).argmin(axis=1)\n"
                  f"result = np.array([X[assign == c].mean(axis=0) for c in range({k})])",
             require=["euclidean_distances"], tol=1e-6)


@template(M, "hard")
def knn_from_distances(rng):
    d = reg_data(rng, synth=1.0, n=(60, 120), d=(2, 4))
    k = rng.choice([3, 5, 7])
    s = seeded(rng)
    setup = (d.setup + f"\nX_new = np.random.default_rng({s}).normal(size=(4, X.shape[1])).round(2)")
    task = rng.choice(["mean", "manhattan", "weighted"])
    if task == "mean":
        return P(f"{d.desc}; `X_new` holds 4 new rows. Predict them **by hand** with {k}-nearest-neighbour regression: "
                 f"`euclidean_distances(X_new, X)`, the {k} closest training rows per new row (`np.argsort`), and the "
                 f"mean of their targets. It equals `KNeighborsRegressor(n_neighbors={k})`.", setup,
                 f"from sklearn.metrics.pairwise import euclidean_distances\nidx = np.argsort(euclidean_distances(X_new, X), axis=1)[:, :{k}]\n"
                 "result = y[idx].mean(axis=1)", require=["euclidean_distances"], tol=1e-9)
    if task == "manhattan":
        return P(f"{d.desc}; `X_new` holds 4 new rows. Predict them by hand with {k}-nearest-neighbour regression under "
                 f"the **Manhattan** metric (`manhattan_distances`), averaging the {k} nearest targets. It equals "
                 f"`KNeighborsRegressor(n_neighbors={k}, metric='manhattan')`.", setup,
                 f"from sklearn.metrics.pairwise import manhattan_distances\nidx = np.argsort(manhattan_distances(X_new, X), axis=1)[:, :{k}]\n"
                 "result = y[idx].mean(axis=1)", require=["manhattan_distances"], tol=1e-9)
    return P(f"{d.desc}; `X_new` holds 4 new rows. Predict them by hand with **distance-weighted** {k}-NN regression: "
             f"take the {k} nearest training rows (`euclidean_distances`) and average their targets with weights 1 / "
             f"distance. It equals `KNeighborsRegressor(n_neighbors={k}, weights='distance')`.", setup,
             f"from sklearn.metrics.pairwise import euclidean_distances\nD = euclidean_distances(X_new, X)\n"
             f"idx = np.argsort(D, axis=1)[:, :{k}]\nw = 1 / np.take_along_axis(D, idx, axis=1)\n"
             "result = (w * y[idx]).sum(axis=1) / w.sum(axis=1)", require=["euclidean_distances"], tol=1e-9)


@template(M, "hard")
def regression_error_analysis(rng):
    t = table(rng, kind="reg", n=(160, 300))
    cat = rng.choice(t.cat)
    s = seeded(rng)
    a = rng.choice([10.0, 100.0, 1000.0])
    setup = (t.setup + f"\nnum = {lit(t.num)}\nfrom sklearn.model_selection import train_test_split\n"
             "from sklearn.linear_model import LinearRegression\n"
             f"X_tr, X_te, y_tr, y_te = train_test_split(df, y, test_size=0.3, random_state={s})\n"
             "reg = LinearRegression().fit(X_tr[num], y_tr)\npred = reg.predict(X_te[num])")
    items = [
        ("mean_absolute_error", f"Where does the model struggle? Return the test MAE per `{cat}` level as a dict "
                                f"`{{level: mae}}` (levels present in `X_te`).",
         f"{MET}mean_absolute_error\ngrp = X_te[{cat!r}].to_numpy()\n"
         "result = {g: mean_absolute_error(y_te[grp == g], pred[grp == g]) for g in np.unique(grp)}"),
        ("root_mean_squared_error", f"How sure is the test RMSE? Bootstrap it: with `rs = np.random.default_rng({s})`, "
                                    f"200 times draw `idx = rs.integers(0, n, n)` (n = test size) and record the RMSE "
                                    f"on those rows. Return the 5th and 95th percentiles as a tuple.",
         f"{MET}root_mean_squared_error\nrs = np.random.default_rng({s})\nn = len(y_te)\nvals = []\n"
         "for _ in range(200):\n    idx = rs.integers(0, n, n)\n    vals.append(root_mean_squared_error(y_te[idx], pred[idx]))\n"
         "result = tuple(np.percentile(vals, [5, 95]))"),
        ("mean_squared_error", "Return the skill score 1 − MSE(model) / MSE(baseline) on the test set, where the "
                               "baseline predicts the training mean `y_tr.mean()` everywhere.",
         f"{MET}mean_squared_error\nbase = np.full(len(y_te), y_tr.mean())\n"
         "result = 1 - mean_squared_error(y_te, pred) / mean_squared_error(y_te, base)"),
        ("r2_score", "R² compares with the **test** mean: return `(r2_score(y_te, pred), 1 - mse / np.var(y_te))` (equal).",
         f"{MET}r2_score, mean_squared_error\nresult = (r2_score(y_te, pred), 1 - mean_squared_error(y_te, pred) / np.var(y_te))",
         {"require": ["r2_score", "mean_squared_error"]}),
        ("root_mean_squared_error", f"Does `{cat}` help? One-hot encode it with `pd.get_dummies(df[num + [{cat!r}]], "
                                    f"dtype=float)`, take the same train/test rows (`.loc[X_tr.index]`, "
                                    f"`.loc[X_te.index]`), refit `LinearRegression` and return `(rmse_numeric_only, "
                                    f"rmse_with_dummies)` on the test set.",
         f"{MET}root_mean_squared_error\nD = pd.get_dummies(df[num + [{cat!r}]], dtype=float)\n"
         "reg2 = LinearRegression().fit(D.loc[X_tr.index], y_tr)\n"
         "result = (root_mean_squared_error(y_te, pred), root_mean_squared_error(y_te, reg2.predict(D.loc[X_te.index])))"),
        ("Ridge", f"Fit `Ridge(alpha={a})` on the same numeric training columns. On which metrics is it **strictly** "
                  f"better than `reg` on the test set? Check `mean_absolute_error`, `root_mean_squared_error`, "
                  f"`median_absolute_error`, `max_error` and return the sorted list of the winning metric names.",
         f"from sklearn.linear_model import Ridge\n{MET}mean_absolute_error, root_mean_squared_error, median_absolute_error, max_error\n"
         f"p2 = Ridge(alpha={a}).fit(X_tr[num], y_tr).predict(X_te[num])\n"
         "fns = [mean_absolute_error, root_mean_squared_error, median_absolute_error, max_error]\n"
         "result = sorted(f.__name__ for f in fns if f(y_te, p2) < f(y_te, pred))",
         {"require": ["Ridge", "median_absolute_error"]}),
        ("mean_absolute_error", "Recent rows count more: the test rows' weights are their index labels (the original "
                                "row order, `X_te.index + 1`). Return `(plain_mae, weighted_mae)` (`sample_weight=`).",
         f"{MET}mean_absolute_error\nresult = (mean_absolute_error(y_te, pred), "
         "mean_absolute_error(y_te, pred, sample_weight=X_te.index.to_numpy() + 1))"),
        ("mean_squared_error", "How concentrated are the errors? Return the share of the total squared error (`n · "
                               "mean_squared_error`) that comes from the 10% of test rows with the largest squared "
                               "residuals (`int(0.1 * n)` rows).",
         f"{MET}mean_squared_error\nn = len(y_te)\nsq = np.sort((y_te - pred) ** 2)[::-1]\n"
         "result = sq[:int(0.1 * n)].sum() / (n * mean_squared_error(y_te, pred))"),
    ]
    return catalog(rng, items, setup, f"{t.desc}. `num` lists the numeric columns. The rows were split with "
                                      f"`train_test_split(df, y, test_size=0.3, random_state={s})` into `X_tr, X_te, "
                                      f"y_tr, y_te`; `reg = LinearRegression()` was fitted on `X_tr[num]` and `pred` "
                                      f"holds its predictions for `X_te[num]`.", tol=1e-6)


@template(M, "hard")
def clustering_model_selection(rng):
    data = rng.choice(["wine", "iris"])
    loader = "load_wine" if data == "wine" else "load_iris"
    setup = (f"from sklearn.datasets import {loader}\nfrom sklearn.preprocessing import StandardScaler\n"
             f"X, y = {loader}(return_X_y=True)\nZ = StandardScaler().fit_transform(X)")
    desc = (f"`Z` holds the standardized {'wine' if data == 'wine' else 'iris'} measurements and `y` the true classes "
            f"(0, 1, 2)")
    task = rng.choice(["pipeline", "stability", "agglo"])
    r = rng.randint(0, 5)
    if task == "pipeline":
        return P(f"{desc}. Choose `k` without the labels, then check with them: for `k` in 2..6 fit `KMeans(n_clusters=k, "
                 f"n_init='auto', random_state={r})` on `Z`, keep the `k` with the best `silhouette_score`, and return "
                 f"`(k, ari)` where `ari` compares that clustering with `y`.", setup,
                 f"{KM}{MET}silhouette_score, adjusted_rand_score\n"
                 f"lab = {{k: KMeans(n_clusters=k, n_init='auto', random_state={r}).fit_predict(Z) for k in range(2, 7)}}\n"
                 "best = max(lab, key=lambda k: silhouette_score(Z, lab[k]))\nresult = (best, adjusted_rand_score(y, lab[best]))",
                 require=["KMeans", "silhouette_score", "adjusted_rand_score"], tol=1e-9)
    if task == "stability":
        return P(f"{desc}. Measure stability without labels: fit `KMeans(n_clusters=3, n_init=1, random_state=s)` on `Z` "
                 f"for `s` in 0..5 and return the mean `adjusted_rand_score` between run 0 and each of runs 1..5.", setup,
                 f"{KM}{MET}adjusted_rand_score\n"
                 "runs = [KMeans(n_clusters=3, n_init=1, random_state=s).fit_predict(Z) for s in range(6)]\n"
                 "result = float(np.mean([adjusted_rand_score(runs[0], r) for r in runs[1:]]))",
                 require=["KMeans", "adjusted_rand_score"], tol=1e-9)
    link = rng.choice(["ward", "average", "complete"])
    return P(f"{desc}. Compare two algorithms with 3 clusters on `Z`: `KMeans(n_clusters=3, n_init='auto', "
             f"random_state={r})` and `AgglomerativeClustering(n_clusters=3, linkage={link!r})` (from "
             f"`sklearn.cluster`). Return `{{'kmeans': (silhouette, ari), 'agglo': (silhouette, ari)}}` with the ARI "
             f"against `y`.", setup,
             f"from sklearn.cluster import KMeans, AgglomerativeClustering\n{MET}silhouette_score, adjusted_rand_score\n"
             f"labs = {{'kmeans': KMeans(n_clusters=3, n_init='auto', random_state={r}).fit_predict(Z), "
             f"'agglo': AgglomerativeClustering(n_clusters=3, linkage={link!r}).fit_predict(Z)}}\n"
             "result = {name: (silhouette_score(Z, l), adjusted_rand_score(y, l)) for name, l in labs.items()}",
             require=["AgglomerativeClustering", "silhouette_score", "adjusted_rand_score"], tol=1e-9)


# ================================================================= challenge

@template(M, "challenge")
def silhouette_on_a_line(rng):
    n, k, s = rng.randint(9, 14), rng.randint(2, 3), seeded(rng)
    report = rng.choice(["mean", "negatives", "per_cluster"])
    setup = (f"g = np.random.default_rng({s})\nlabels = g.permutation(np.arange({n}) % {k})\n"
             f"x = (labels * 3 + g.normal(0, 1.8, {n})).round(3)\nreport = {report!r}")
    tests = [f"g = np.random.default_rng({s + i})\nlabels = g.permutation(np.arange(200000) % {k + i})\n"
             f"x = labels * 2 + g.normal(0, 1.5, 200000)\nreport = {report!r}" for i in (1, 2)]
    tail = {"mean": "result = float(sil.mean())", "negatives": "result = int((sil < 0).sum())",
            "per_cluster": "result = np.array([sil[labels == c].mean() for c in range(K)])"}[report]
    sol = ("n, K = len(x), labels.max() + 1\nS = np.empty((n, K))\nfor c in range(K):\n    v = np.sort(x[labels == c])\n"
           "    pre = np.concatenate([[0.0], np.cumsum(v)])\n    j = np.searchsorted(v, x)\n"
           "    S[:, c] = (x * j - pre[j]) + (pre[-1] - pre[j] - x * (len(v) - j))\n"
           "cnt = np.bincount(labels, minlength=K)\nrows = np.arange(n)\na = S[rows, labels] / (cnt[labels] - 1)\n"
           "S = S / cnt\nS[rows, labels] = np.inf\nb = S.min(axis=1)\nsil = (b - a) / np.maximum(a, b)\n" + tail)
    want = {"mean": "the **mean** silhouette coefficient (what `silhouette_score` returns), as a float",
            "negatives": "the **number of points** with a negative silhouette coefficient, as an int",
            "per_cluster": "the **mean silhouette of each cluster** (cluster 0 first), as a float array"}[report]
    return challenge(
        "Silhouettes on a line",
        "A 1-D clustering: point `i` sits at `x[i]` and belongs to cluster `labels[i]` (clusters are numbered 0..K−1 and "
        "each has at least 2 points). With distances |x[i] − x[j]|, a point's silhouette is (b − a) / max(a, b), where "
        "a = its mean distance to the **other** points of its cluster and b = the smallest mean distance to the points "
        f"of another cluster. Return {want}. (`report` says which: {report!r}.) `silhouette_samples` needs all n² "
        "distances; sort each cluster once and use prefix sums instead.",
        setup, sol, tests, "n ≤ 2·10^5, K ≤ 4", speed="O(n·K·log n)", tol=1e-6)


@template(M, "challenge")
def prefix_adjusted_rand(rng):
    n, s = rng.randint(8, 13), seeded(rng)
    report = rng.choice(["ari", "rand"])
    setup = (f"g = np.random.default_rng({s})\na = g.integers(0, 3, {n})\n"
             f"b = np.where(g.random({n}) < 0.7, a, g.integers(0, 3, {n}))\nreport = {report!r}")
    tests = [f"g = np.random.default_rng({s + i})\na = g.integers(0, {40 * i}, 200000)\n"
             f"b = np.where(g.random(200000) < 0.6, a, g.integers(0, {50 * i}, 200000))\nreport = {report!r}"
             for i in (1, 2)]
    sol = ("def seen_before(v):\n    order = np.argsort(v, kind='stable')\n    sv = v[order]\n"
           "    start = np.r_[0, np.flatnonzero(sv[1:] != sv[:-1]) + 1]\n"
           "    first = np.repeat(start, np.diff(np.r_[start, len(v)]))\n    out = np.empty(len(v))\n"
           "    out[order] = np.arange(len(v)) - first\n    return out\n"
           "pair = a.astype(np.int64) * (int(b.max()) + 1) + b\n"
           "both, ra, rb = np.cumsum(seen_before(pair)), np.cumsum(seen_before(a)), np.cumsum(seen_before(b))\n"
           "t = np.arange(1, len(a) + 1)\ntotal = t * (t - 1) / 2\nboth, ra, rb, total = both[1:], ra[1:], rb[1:], total[1:]\n")
    if report == "ari":
        sol += ("same = (ra == both) & (rb == both)\nexp = ra * rb / total\nden = (ra + rb) / 2 - exp\n"
                "result = np.where(same, 1.0, (both - exp) / np.where(same, 1.0, den))")
        want = ("the **adjusted Rand index** `adjusted_rand_score(a[:t], b[:t])` (it is 1.0 when the two prefixes "
                "group every pair the same way)")
    else:
        sol += "result = (total + 2 * both - ra - rb) / total"
        want = "the **Rand index** `rand_score(a[:t], b[:t])`: the fraction of pairs on which both labelings agree"
    return challenge(
        "Agreement as labels stream in",
        "Points arrive one at a time with a true group `a[i]` and a cluster `b[i]`. After each arrival from the 2nd "
        f"on, a dashboard shows {want}. Return the values for t = 2, 3, ..., n as a float array. Recomputing the "
        "score for every prefix is quadratic; note that when a point arrives, the number of new pairs that share its "
        "group, its cluster, or both is the number of earlier points with the same value.",
        setup, sol, tests, "n ≤ 2·10^5", speed="O(n log n)", answer_desc=" (a float array of length n − 1)", tol=1e-6)


@template(M, "challenge")
def ndcg_every_cutoff(rng):
    n, q, s = rng.randint(8, 12), rng.randint(4, 6), seeded(rng)
    report = rng.choice(["ndcg", "dcg"])
    setup = (f"g = np.random.default_rng({s})\nrel = g.integers(0, 4, {n})\nrel[0] = 3\n"
             f"score = g.integers(0, 6, {n}) / 5   # coarse scores: ties!\nks = g.integers(1, {n} + 1, {q})\n"
             f"report = {report!r}")
    tests = [f"g = np.random.default_rng({s + i})\nrel = g.integers(0, 4, 100000)\nrel[0] = 3\n"
             f"score = g.integers(0, {100 * i}, 100000) / 7\nks = g.integers(1, 100001, 100000)\nreport = {report!r}"
             for i in (1, 2)]
    sol = ("n = len(rel)\ndisc = 1 / np.log2(np.arange(n) + 2)\nD = np.r_[0.0, np.cumsum(disc)]\n"
           "vals, inv, cnt = np.unique(-score, return_inverse=True, return_counts=True)\n"
           "mean_gain = np.bincount(inv, weights=rel) / cnt\nend = np.cumsum(cnt)\nstart = end - cnt\n"
           "G = np.r_[0.0, np.cumsum(mean_gain * (D[end] - D[start]))]\nj = np.searchsorted(end, ks, side='right')\n"
           "part = np.where(j < len(cnt), mean_gain[np.minimum(j, len(cnt) - 1)] * (D[ks] - D[start[np.minimum(j, len(cnt) - 1)]]), 0.0)\n"
           "dcg = G[j] + part\n")
    if report == "ndcg":
        sol += "ideal = np.r_[0.0, np.cumsum(np.sort(rel)[::-1] * disc)]\nresult = dcg / ideal[ks]"
        want = "`ndcg_score([rel], [score], k=k)`"
    else:
        sol += "result = dcg"
        want = "`dcg_score([rel], [score], k=k)`"
    return challenge(
        "NDCG at every cut-off",
        "One query, n documents: `rel[i]` is the relevance of document `i` and `score[i]` the model's score (higher is "
        "ranked earlier). Scores tie a lot, and scikit-learn handles a tie by giving every position of a tied group "
        "the group's **average** relevance: DCG@k = Σ over tied groups (in score order) of mean_rel(group) × Σ "
        "1/log2(p + 2) over the group's positions p < k (positions counted from 0). The ideal DCG@k sums the k "
        f"largest relevances with the same discounts. For every `k` in `ks`, compute {want}. A fresh computation per "
        "`k` is too slow: precompute prefix sums of discounts and of group contributions.",
        setup, sol, tests, "n, len(ks) ≤ 10^5, 1 ≤ k ≤ n", speed="O((n + len(ks)) log n)",
        answer_desc=" (a float array, one value per k)", tol=1e-6)


@template(M, "challenge")
def manhattan_row_sums(rng):
    n, d, s = rng.randint(7, 11), rng.randint(2, 3), seeded(rng)
    report = rng.choice(["total", "row_sums", "medoid"])
    setup = f"X = np.random.default_rng({s}).uniform(0, 10, ({n}, {d})).round(3)\nreport = {report!r}"
    tests = [f"X = np.random.default_rng({s + i}).uniform(0, 1000, (100000, 2))\nreport = {report!r}" for i in (1, 2)]
    tail = {"total": "result = float(rows.sum() / 2)", "row_sums": "result = rows",
            "medoid": "result = int(rows.argmin())"}[report]
    sol = ("n = len(X)\nrows = np.zeros(n)\ni = np.arange(n)\nfor col in X.T:\n    o = np.argsort(col)\n    v = col[o]\n"
           "    pre = np.cumsum(v)\n    below = v * i - (pre - v)\n    above = (pre[-1] - pre) - v * (n - 1 - i)\n"
           "    rows[o] += below + above\n" + tail)
    want = {"total": "the **sum of the L1 distances over all unordered pairs** of rows (a float)",
            "row_sums": "for every row, the **sum of its L1 distances to all rows** (`manhattan_distances(X).sum(axis=1)`, "
                        "a float array)",
            "medoid": "the index of the **medoid**: the row whose summed L1 distance to all rows is smallest (an int)"}[report]
    return challenge(
        "Manhattan medoid",
        "k-medoids and the L1 silhouette both need, for every row of `X`, the total Manhattan distance to all other "
        f"rows. Return {want}. (`report` says which: {report!r}.) The full `manhattan_distances(X)` matrix has n² "
        "entries; the L1 distance splits by column, and in one column the distances to all values follow from sorting "
        "and prefix sums.",
        setup, sol, tests, "n ≤ 10^5 rows, d ≤ 3 columns", speed="O(d·n log n)", tol=1e-6)
