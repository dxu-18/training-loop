"""4.11 Clustering & mixture models: k-means, density-based and hierarchical clustering, Gaussian mixtures."""

from ._common import P, catalog, challenge, lit, seeded, synth_clf, table, template

M = "4.11"
HDB = "copy=True"     # silences a FutureWarning in scikit-learn 1.8


def _blobs(rng, n=(150, 300), k=(3, 5), std=(0.5, 1.2), d=2):
    n_, k_ = rng.randint(*n), rng.randint(*k)
    sd = round(rng.uniform(*std), 1)
    extra = f", n_features={d}" if d != 2 else ""
    setup = (f"from sklearn.datasets import make_blobs\nX, y_true = make_blobs(n_samples={n_}, centers={k_}, "
             f"cluster_std={sd}{extra}, random_state={seeded(rng)})")
    desc = (f"`X` holds {n_} points in {d}-D drawn around {k_} centers (`make_blobs`, `cluster_std={sd}`), and "
            f"`y_true` says which center each point came from")
    return setup, desc, k_, n_


def _moons(rng, n=(150, 300)):
    n_, noise = rng.randint(*n), rng.choice([0.04, 0.05, 0.06, 0.08])
    setup = (f"from sklearn.datasets import make_moons\nX, y_true = make_moons(n_samples={n_}, noise={noise}, "
             f"random_state={seeded(rng)})")
    desc = (f"`X` holds {n_} 2-D points on two interleaving half-circles (`make_moons`, `noise={noise}`), and "
            f"`y_true` says which moon each point belongs to")
    return setup, desc, n_


def _km(k, rs=0):
    return f"KMeans(n_clusters={k}, n_init='auto', random_state={rs})"


KM = "from sklearn.cluster import KMeans\n"


# ================================================================= lite

@template(M, "lite")
def kmeans_fitted_attributes(rng):
    setup, desc, k, n = _blobs(rng)
    kk = max(2, k + rng.choice([-1, 0, 0, 1]))
    new = [[round(rng.uniform(-10, 10), 1), round(rng.uniform(-10, 10), 1)] for _ in range(3)]
    fit = f"{KM}km = {_km(kk)}.fit(X)\n"
    head = f"Fit `km = {_km(kk)}` (from `sklearn.cluster`) on `X` and return"
    items = [
        ("labels_", f"{head} the cluster labels of the first 10 points (`labels_`).", fit + "result = km.labels_[:10]"),
        ("cluster_centers_", f"{head} its cluster centers sorted by their first coordinate (sort the rows of "
                             f"`cluster_centers_` with `np.argsort` on column 0).",
         fit + "C = km.cluster_centers_\nresult = C[np.argsort(C[:, 0])]"),
        ("inertia_", f"{head} its `inertia_`: the sum of squared distances from every point to its closest center.",
         fit + "result = km.inertia_"),
        ("predict", f"{head} the clusters it assigns to the new points `X_new` (`predict`).",
         fit + "result = km.predict(X_new)"),
        ("transform", f"{head} `km.transform(X[:3])`: each point's distance to every center.",
         fit + "result = km.transform(X[:3])"),
        ("n_iter_", f"{head} the number of iterations its best run needed (`n_iter_`).", fit + "result = km.n_iter_"),
        ("labels_", f"{head} the cluster sizes sorted from largest to smallest (count the `labels_`).",
         fit + "result = sorted(np.bincount(km.labels_).tolist(), reverse=True)"),
        ("fit_predict", f"Use `{_km(kk)}.fit_predict(X)` (fit and return the labels in one call) and return how many "
                        f"points land in the same cluster as the first point.",
         f"{KM}labels = {_km(kk)}.fit_predict(X)\nresult = int((labels == labels[0]).sum())"),
    ]
    return catalog(rng, items, setup + f"\nX_new = np.array({lit(new)})", f"{desc}; `X_new` holds 3 new points.",
                   tol=1e-6)


@template(M, "lite")
def density_clusterers(rng):
    setup, desc, n = _moons(rng)
    eps, ms = rng.choice([0.15, 0.2, 0.25]), rng.choice([4, 5, 8])
    mcs, ms2 = rng.choice([5, 10, 15]), rng.choice([5, 10, 15])
    db = f"from sklearn.cluster import DBSCAN\ndb = DBSCAN(eps={eps}, min_samples={ms}).fit(X)\n"
    head = f"Fit `db = DBSCAN(eps={eps}, min_samples={ms})` (from `sklearn.cluster`) on `X` and return"
    items = [
        ("DBSCAN", f"{head} the sorted list of distinct labels it assigns (-1 marks noise; no `n_clusters` needed).",
         db + "result = sorted(set(db.labels_.tolist()))"),
        ("labels_", f"{head} the number of noise points (label -1).", db + "result = int((db.labels_ == -1).sum())"),
        ("core_sample_indices_", f"{head} the number of core points (`core_sample_indices_`: points with at least "
                                 f"{ms} points, themselves included, within `eps`).",
         db + "result = len(db.core_sample_indices_)"),
        ("DBSCAN", f"Use `DBSCAN(eps={eps}, min_samples={ms}).fit_predict(X)` and return the number of clusters it "
                   f"finds (labels ≥ 0).",
         f"from sklearn.cluster import DBSCAN\nlabels = DBSCAN(eps={eps}, min_samples={ms}).fit_predict(X)\n"
         "result = len(set(labels.tolist()) - {-1})"),
        ("HDBSCAN", f"Fit `HDBSCAN(min_cluster_size={mcs}, {HDB})` (from `sklearn.cluster`; no `eps` to tune) on `X` "
                    f"and return the number of clusters it finds (labels ≥ 0).",
         f"from sklearn.cluster import HDBSCAN\nlabels = HDBSCAN(min_cluster_size={mcs}, {HDB}).fit(X).labels_\n"
         "result = int(labels.max()) + 1"),
        ("OPTICS", f"Fit `OPTICS(min_samples={ms2})` (from `sklearn.cluster`) on `X` and return its number of noise "
                   f"points (label -1).",
         f"from sklearn.cluster import OPTICS\nresult = int((OPTICS(min_samples={ms2}).fit(X).labels_ == -1).sum())"),
    ]
    return catalog(rng, items, setup, f"{desc}.")


@template(M, "lite")
def more_clusterers(rng):
    setup, desc, k, n = _blobs(rng)
    link, q = rng.choice(["ward", "complete", "average", "single"]), rng.choice([0.1, 0.2, 0.3])
    b = rng.choice([32, 64, 128])
    sizes = "result = sorted(np.bincount(labels).tolist(), reverse=True)"
    items = [
        ("AgglomerativeClustering", f"Cluster `X` with `AgglomerativeClustering(n_clusters={k}, linkage={link!r})` "
                                    f"(`fit_predict`) and return the cluster sizes, largest first.",
         f"from sklearn.cluster import AgglomerativeClustering\n"
         f"labels = AgglomerativeClustering(n_clusters={k}, linkage={link!r}).fit_predict(X)\n" + sizes),
        ("SpectralClustering", f"Cluster `X` with `SpectralClustering(n_clusters={k}, random_state=0)` (clusters the "
                               f"eigenvectors of a similarity graph; `fit_predict`) and return the cluster sizes, "
                               f"largest first.",
         f"from sklearn.cluster import SpectralClustering\nlabels = SpectralClustering(n_clusters={k}, "
         f"random_state=0).fit_predict(X)\n" + sizes),
        ("MeanShift", f"Fit `MeanShift(bandwidth=estimate_bandwidth(X, quantile={q}))` (both from `sklearn.cluster`) on "
                      f"`X` and return how many clusters it found (`cluster_centers_.shape[0]`): mean shift picks the "
                      f"number itself.",
         f"from sklearn.cluster import MeanShift, estimate_bandwidth\n"
         f"result = MeanShift(bandwidth=estimate_bandwidth(X, quantile={q})).fit(X).cluster_centers_.shape[0]"),
        ("estimate_bandwidth", f"Return `estimate_bandwidth(X, quantile={q})` (from `sklearn.cluster`): a kernel width "
                               f"for mean shift, from the distances to each point's nearest {int(q * 100)}% of points.",
         f"from sklearn.cluster import estimate_bandwidth\nresult = estimate_bandwidth(X, quantile={q})"),
        ("Birch", f"Cluster `X` with `Birch(n_clusters={k})` (`fit_predict`) and return the cluster sizes, largest "
                  f"first.", f"from sklearn.cluster import Birch\nlabels = Birch(n_clusters={k}).fit_predict(X)\n" + sizes),
        ("MiniBatchKMeans", f"Fit `MiniBatchKMeans(n_clusters={k}, batch_size={b}, n_init=3, random_state=0)` (k-means "
                            f"on small random batches) on `X` and return its `inertia_`.",
         f"from sklearn.cluster import MiniBatchKMeans\n"
         f"result = MiniBatchKMeans(n_clusters={k}, batch_size={b}, n_init=3, random_state=0).fit(X).inertia_"),
    ]
    return catalog(rng, items, setup, f"{desc}.", tol=1e-6)


@template(M, "lite")
def gmm_fitted_attributes(rng):
    setup, desc, k, n = _blobs(rng)
    ct = rng.choice(["full", "diag", "spherical", "tied"])
    ctor = f"GaussianMixture(n_components={k}, covariance_type={ct!r}, random_state=0)"
    fit = f"from sklearn.mixture import GaussianMixture\ngmm = {ctor}.fit(X)\n"
    head = f"Fit `gmm = {ctor}` (from `sklearn.mixture`) on `X` and return"
    items = [
        ("means_", f"{head} its component means sorted by their first coordinate.",
         fit + "m = gmm.means_\nresult = m[np.argsort(m[:, 0])]"),
        ("weights_", f"{head} its mixing weights (`weights_`) sorted in increasing order.",
         fit + "result = np.sort(gmm.weights_)"),
        ("predict_proba", f"{head} the soft assignments of the first 2 points, rounded: "
                          f"`gmm.predict_proba(X[:2]).round(3)`.", fit + "result = gmm.predict_proba(X[:2]).round(3)"),
        ("bic", f"{head} its Bayesian information criterion on `X` (`bic`; lower is better).", fit + "result = gmm.bic(X)"),
        ("aic", f"{head} its Akaike information criterion on `X` (`aic`).", fit + "result = gmm.aic(X)"),
        ("sample", f"{head} 5 new points drawn from it: `gmm.sample(5)` returns `(points, component labels)`; return "
                   f"the points.", fit + "result = gmm.sample(5)[0]"),
        ("covariances_", f"{head} the shape of `covariances_` as a tuple (it depends on `covariance_type`).",
         fit + "result = gmm.covariances_.shape"),
        ("score", f"{head} the average log-likelihood of `X` under it (`score`).", fit + "result = gmm.score(X)"),
        ("BayesianGaussianMixture", "Fit `BayesianGaussianMixture(n_components=10, max_iter=500, random_state=0)` (from "
                                    "`sklearn.mixture`) on `X`: with more components than needed, it shrinks the "
                                    "unneeded ones' weights towards 0. Return `weights_.round(2)`.",
         "from sklearn.mixture import BayesianGaussianMixture\n"
         "result = BayesianGaussianMixture(n_components=10, max_iter=500, random_state=0).fit(X).weights_.round(2)"),
    ]
    return catalog(rng, items, setup, f"{desc}.", tol=1e-5)


# ================================================================= easy

@template(M, "easy")
def kmeans_by_hand(rng):
    setup, desc, k, n = _blobs(rng)
    new = [[round(rng.uniform(-10, 10), 1), round(rng.uniform(-10, 10), 1)] for _ in range(rng.randint(3, 5))]
    task = rng.choice(["inertia", "predict", "transform", "means"])
    fit = f"{KM}km = {_km(k)}.fit(X)\nC = km.cluster_centers_\n"
    head = f"{desc}. Fit `km = {_km(k)}` on `X`; `C = km.cluster_centers_`."
    if task == "inertia":
        return P(f"{head} Compute the inertia **by hand**: the sum over all points of the squared distance to their "
                 f"own center `C[km.labels_]`. Return `(by_hand, km.inertia_)`.", setup,
                 fit + "result = (((X - C[km.labels_]) ** 2).sum(), km.inertia_)", require=["cluster_centers_", "labels_"],
                 tol=1e-6)
    if task == "predict":
        return P(f"{head} `predict` assigns each point to its nearest center. Rebuild `km.predict(X_new)` by hand from "
                 f"`C` (squared distances with broadcasting, then `argmin`) and return your labels.",
                 setup + f"\nX_new = np.array({lit(new)})",
                 fit + "result = ((X_new[:, None, :] - C[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)",
                 require=["cluster_centers_"])
    if task == "transform":
        return P(f"{head} `km.transform(X)` gives each point's Euclidean distance to every center. Rebuild it by hand "
                 f"for the first 5 points and return your 5 × {k} array.", setup,
                 fit + "result = np.sqrt(((X[:5, None, :] - C[None, :, :]) ** 2).sum(axis=2))",
                 require=["cluster_centers_"], tol=1e-9)
    return P(f"{head} After convergence each center is the mean of the points assigned to it. Compute the mean of "
             f"the points of each cluster (grouped by `km.labels_`) and return them as a {k} × 2 array in label order; "
             f"they match `C`.", setup,
             fit + f"result = np.array([X[km.labels_ == j].mean(axis=0) for j in range({k})])",
             require=["labels_"], tol=1e-6)


@template(M, "easy")
def elbow_inertias(rng):
    setup, desc, k, n = _blobs(rng)
    K = rng.randint(6, 8)
    task = rng.choice(["list", "drops", "elbow"])
    sol = f"{KM}inertias = [KMeans(n_clusters=k, n_init='auto', random_state=0).fit(X).inertia_ for k in range(1, {K + 1})]\n"
    head = (f"{desc}. The *elbow method*: fit `KMeans(n_clusters=k, n_init='auto', random_state=0)` for k = 1 .. {K} "
            f"and record each `inertia_`.")
    if task == "list":
        return P(f"{head} Return the list of inertias (it always decreases as k grows).", setup,
                 sol + "result = inertias", require=["inertia_"], tol=1e-6)
    if task == "drops":
        return P(f"{head} Return the relative drop at each step, `1 − inertia[k] / inertia[k − 1]` for k = 2 .. {K}, as "
                 f"a list: big drops until the true number of clusters, small ones after.", setup,
                 sol + "result = [1 - b / a for a, b in zip(inertias, inertias[1:])]", require=["inertia_"], tol=1e-6)
    return P(f"{head} Return the elbow: the smallest k for which going to k + 1 lowers the inertia by less than 25% "
             f"(`inertia[k + 1] > 0.75 · inertia[k]`), or {K} if there is none.", setup,
             sol + f"result = next((k for k in range(1, {K}) if inertias[k] > 0.75 * inertias[k - 1]), {K})",
             require=["inertia_"])


@template(M, "easy")
def silhouette_choose_k(rng):
    setup, desc, k, n = _blobs(rng)
    K = rng.randint(6, 7)
    task = rng.choice(["scores", "best", "agglo"])
    head = f"{desc}. The silhouette score (from `sklearn.metrics`) rates a clustering between −1 and 1, higher is better."
    if task == "agglo":
        return P(f"{head} For k = 2 .. {K}, cluster `X` with `AgglomerativeClustering(n_clusters=k)` (Ward linkage) and "
                 f"compute `silhouette_score(X, labels)`. Return the k with the best score.", setup,
                 "from sklearn.cluster import AgglomerativeClustering\nfrom sklearn.metrics import silhouette_score\n"
                 f"scores = {{k: silhouette_score(X, AgglomerativeClustering(n_clusters=k).fit_predict(X)) for k in "
                 f"range(2, {K + 1})}}\nresult = max(scores, key=scores.get)",
                 require=["AgglomerativeClustering", "silhouette_score"])
    sol = ("from sklearn.metrics import silhouette_score\n" + KM +
           f"scores = [silhouette_score(X, KMeans(n_clusters=k, n_init='auto', random_state=0).fit_predict(X)) for k in "
           f"range(2, {K + 1})]\n")
    if task == "scores":
        return P(f"{head} For k = 2 .. {K}, fit `KMeans(n_clusters=k, n_init='auto', random_state=0)` and return the "
                 f"list of `silhouette_score(X, labels)` values.", setup, sol + "result = scores",
                 require=["silhouette_score"], tol=1e-6)
    return P(f"{head} For k = 2 .. {K}, fit `KMeans(n_clusters=k, n_init='auto', random_state=0)` and return the k with "
             f"the best `silhouette_score(X, labels)` (unlike inertia it does not simply favour more clusters).", setup,
             sol + "result = int(np.argmax(scores)) + 2", require=["silhouette_score"])


@template(M, "easy")
def dbscan_noise(rng):
    setup, desc, n = _moons(rng)
    eps, ms = rng.choice([0.15, 0.2, 0.25]), rng.choice([4, 5, 8])
    task = rng.choice(["noise", "eps_sweep", "ms_sweep", "ari"])
    head = f"{desc}."
    if task == "noise":
        return P(f"{head} Cluster it with `DBSCAN(eps={eps}, min_samples={ms})` and return `(number of clusters, "
                 f"number of noise points)`.", setup,
                 f"from sklearn.cluster import DBSCAN\nlabels = DBSCAN(eps={eps}, min_samples={ms}).fit(X).labels_\n"
                 "result = (len(set(labels.tolist()) - {-1}), int((labels == -1).sum()))", require=["DBSCAN", "eps"])
    if task == "eps_sweep":
        epss = [0.05, 0.1, 0.2, 0.3]
        return P(f"{head} For `eps` in `{epss}`, fit `DBSCAN(eps=eps, min_samples={ms})` and return the list of noise "
                 f"counts: a small radius labels almost everything noise.", setup,
                 f"from sklearn.cluster import DBSCAN\nresult = [int((DBSCAN(eps=e, min_samples={ms}).fit(X).labels_ == -1)"
                 f".sum()) for e in {epss}]", require=["DBSCAN", "eps"])
    if task == "ms_sweep":
        mss = [3, 5, 10, 20]
        return P(f"{head} For `min_samples` in `{mss}`, fit `DBSCAN(eps={eps}, min_samples=m)` and return the list of "
                 f"numbers of clusters found (labels ≥ 0).", setup,
                 f"from sklearn.cluster import DBSCAN\nresult = [len(set(DBSCAN(eps={eps}, min_samples=m).fit(X).labels_"
                 f".tolist()) - {{-1}}) for m in {mss}]", require=["DBSCAN", "min_samples"])
    return P(f"{head} Compare with the truth: return `adjusted_rand_score(y_true, labels)` (from `sklearn.metrics`; 1 = "
             f"perfect agreement, whatever the label names) for `DBSCAN(eps={eps}, min_samples={ms})` and for "
             f"`KMeans(n_clusters=2, n_init='auto', random_state=0)`, as `(dbscan, kmeans)`.", setup,
             f"from sklearn.cluster import DBSCAN, KMeans\nfrom sklearn.metrics import adjusted_rand_score\n"
             f"a = DBSCAN(eps={eps}, min_samples={ms}).fit_predict(X)\nb = {_km(2)}.fit_predict(X)\n"
             "result = (adjusted_rand_score(y_true, a), adjusted_rand_score(y_true, b))",
             require=["DBSCAN", "adjusted_rand_score"], tol=1e-9)


@template(M, "easy")
def gmm_choose_components(rng):
    setup, desc, k, n = _blobs(rng)
    K = rng.randint(6, 7)
    task = rng.choice(["bic", "aic", "types"])
    if task == "types":
        return P(f"{desc}. For each `covariance_type` in `['full', 'tied', 'diag', 'spherical']`, fit "
                 f"`GaussianMixture(n_components={k}, covariance_type=..., random_state=0)` on `X` and return a dict "
                 f"{{type: BIC}}.", setup,
                 "from sklearn.mixture import GaussianMixture\n"
                 f"result = {{t: GaussianMixture(n_components={k}, covariance_type=t, random_state=0).fit(X).bic(X) "
                 "for t in ['full', 'tied', 'diag', 'spherical']}", require=["covariance_type", "bic"], tol=1e-6)
    crit = task
    return P(f"{desc}. Choose the number of mixture components by {crit.upper()}: for n = 1 .. {K}, fit "
             f"`GaussianMixture(n_components=n, random_state=0)` on `X` and compute `{crit}(X)`. Return `(best_n, "
             f"list_of_{crit}s)` (lower is better).", setup,
             f"from sklearn.mixture import GaussianMixture\nvals = [GaussianMixture(n_components=m, random_state=0)"
             f".fit(X).{crit}(X) for m in range(1, {K + 1})]\nresult = (int(np.argmin(vals)) + 1, vals)",
             require=["GaussianMixture", crit], tol=1e-6)


@template(M, "easy")
def compare_with_truth(rng):
    setup, desc, k, n = _blobs(rng, std=(0.8, 1.6))
    task = rng.choice(["dict", "best", "crosstab"])
    sol = ("from sklearn.cluster import KMeans, AgglomerativeClustering\nfrom sklearn.mixture import GaussianMixture\n"
           "from sklearn.metrics import adjusted_rand_score\n"
           f"labels = {{'kmeans': {_km(k)}.fit_predict(X),\n"
           f"          'agglo': AgglomerativeClustering(n_clusters={k}).fit_predict(X),\n"
           f"          'gmm': GaussianMixture(n_components={k}, random_state=0).fit_predict(X)}}\n"
           "ari = {m: adjusted_rand_score(y_true, l) for m, l in labels.items()}\n")
    head = (f"{desc}. Cluster `X` into {k} groups with `'kmeans'`: `{_km(k)}`, `'agglo'`: "
            f"`AgglomerativeClustering(n_clusters={k})` and `'gmm'`: `GaussianMixture(n_components={k}, "
            f"random_state=0)` (each with `fit_predict`), and score each against `y_true` with `adjusted_rand_score` "
            f"(from `sklearn.metrics`; label names don't matter).")
    if task == "dict":
        return P(f"{head} Return a dict {{name: ARI}}.", setup, sol + "result = ari",
                 require=["adjusted_rand_score"], tol=1e-6)
    if task == "best":
        return P(f"{head} Return the name of the best method (the first of kmeans, agglo, gmm on ties).", setup,
                 sol + "result = max(ari, key=ari.get)", require=["adjusted_rand_score"])
    return P(f"{desc}. Fit `{_km(k)}` on `X` and return `pd.crosstab(y_true, labels)`: rows are the true groups, "
             f"columns the clusters (a good clustering has one big number per row).", setup,
             f"{KM}labels = {_km(k)}.fit_predict(X)\nresult = pd.crosstab(y_true, labels)", require=["KMeans", "crosstab"])


@template(M, "easy")
def scale_before_clustering(rng):
    t = table(rng, numeric_only=True, n=(120, 250))
    k = rng.randint(3, 4)
    task = rng.choice(["ari", "dominant", "pipeline"])
    head = f"{t.desc}. Its columns use very different units, and k-means only sees distances."
    if task == "ari":
        return P(f"{head} Cluster `df` with `{_km(k)}` once on the raw values and once after `StandardScaler`, and "
                 f"return `adjusted_rand_score(raw_labels, scaled_labels)` (from `sklearn.metrics`): how much the two "
                 f"clusterings agree.", t.setup,
                 f"{KM}from sklearn.preprocessing import StandardScaler\nfrom sklearn.metrics import adjusted_rand_score\n"
                 f"a = {_km(k)}.fit_predict(df)\nb = {_km(k)}.fit_predict(StandardScaler().fit_transform(df))\n"
                 "result = adjusted_rand_score(a, b)", require=["StandardScaler", "adjusted_rand_score"], tol=1e-9)
    if task == "dominant":
        return P(f"{head} Fit `{_km(k)}` on the raw `df`. Which column drives the clusters? Return the name of the "
                 f"column whose cluster centers are spread the widest (largest max − min across the "
                 f"`cluster_centers_`).", t.setup,
                 f"{KM}C = {_km(k)}.fit(df).cluster_centers_\nresult = df.columns[np.ptp(C, axis=0).argmax()]",
                 require=["cluster_centers_"])
    return P(f"{head} Build `make_pipeline(StandardScaler(), {_km(k)})`, fit it on `df` and return the cluster sizes, "
             f"largest first.", t.setup,
             f"{KM}from sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\n"
             f"labels = make_pipeline(StandardScaler(), {_km(k)}).fit_predict(df)\n"
             "result = sorted(np.bincount(labels).tolist(), reverse=True)", require=["make_pipeline", "StandardScaler"])


@template(M, "easy")
def linkage_comparison(rng):
    setup, desc, n = _moons(rng)
    task = rng.choice(["dict", "best", "single_sizes"])
    sol = ("from sklearn.cluster import AgglomerativeClustering\nfrom sklearn.metrics import adjusted_rand_score\n"
           "ari = {l: adjusted_rand_score(y_true, AgglomerativeClustering(n_clusters=2, linkage=l).fit_predict(X))\n"
           "       for l in ['ward', 'complete', 'average', 'single']}\n")
    head = (f"{desc}. Cluster `X` with `AgglomerativeClustering(n_clusters=2, linkage=...)` for each linkage in "
            f"`['ward', 'complete', 'average', 'single']` and score each with `adjusted_rand_score(y_true, labels)`.")
    if task == "dict":
        return P(f"{head} Return a dict {{linkage: ARI}}.", setup, sol + "result = ari", require=["linkage"], tol=1e-9)
    if task == "best":
        return P(f"{head} Return the best linkage (the first in that order on ties). Single linkage follows chains of "
                 f"close points, which suits thin curved shapes when the noise is low.", setup,
                 sol + "result = max(ari, key=ari.get)", require=["linkage"])
    return P(f"{desc}. Cluster `X` with `AgglomerativeClustering(n_clusters=2, linkage='single')` and with "
             f"`linkage='ward'`. Return both pairs of cluster sizes, each sorted largest first, as `(single, ward)`.",
             setup, "from sklearn.cluster import AgglomerativeClustering\n"
                    "s = AgglomerativeClustering(n_clusters=2, linkage='single').fit_predict(X)\n"
                    "w = AgglomerativeClustering(n_clusters=2, linkage='ward').fit_predict(X)\n"
                    "result = (sorted(np.bincount(s).tolist(), reverse=True), sorted(np.bincount(w).tolist(), reverse=True))",
             require=["linkage"])


@template(M, "easy")
def soft_assignments(rng):
    setup, desc, k, n = _blobs(rng, std=(1.2, 2.0))
    thr = rng.choice([0.8, 0.9, 0.95])
    task = rng.choice(["unsure", "argmax", "entropy"])
    fit = f"from sklearn.mixture import GaussianMixture\ngmm = GaussianMixture(n_components={k}, random_state=0).fit(X)\nR = gmm.predict_proba(X)\n"
    head = (f"{desc}. Fit `gmm = GaussianMixture(n_components={k}, random_state=0)` on `X`; `R = gmm.predict_proba(X)` "
            f"holds every point's probability of belonging to each component (rows sum to 1).")
    if task == "unsure":
        return P(f"{head} How many points does the mixture assign with less than {thr:.0%} confidence (largest "
                 f"probability below {thr})? Return the count.", setup, fit + f"result = int((R.max(axis=1) < {thr}).sum())",
                 require=["predict_proba"])
    if task == "argmax":
        return P(f"{head} `predict` returns the most probable component. Return the fraction of points where "
                 f"`R.argmax(axis=1)` equals `gmm.predict(X)` (1.0) and the mean of `R.max(axis=1)`, as a tuple.",
                 setup, fit + "result = (float((R.argmax(axis=1) == gmm.predict(X)).mean()), float(R.max(axis=1).mean()))",
                 require=["predict_proba", "predict"], tol=1e-6)
    return P(f"{head} Return the mean entropy of the rows of `R` in nats (−Σ r·log r, treating 0·log 0 as 0): near 0 "
             f"when the components are well separated.", setup,
             fit + "Rc = np.clip(R, 1e-300, 1)\nresult = float(-(R * np.log(Rc)).sum(axis=1).mean())",
             require=["predict_proba"], tol=1e-5)


@template(M, "easy")
def kmeans_tour(rng):
    setup, desc, k, n = _blobs(rng)
    i, r = rng.randint(0, 40), rng.choice([0.5, 1.0, 1.5])
    items = [
        ("transform", "Return the mean distance from each point to its own (closest) center: the row minimum of "
                      "`km.transform(X)`, averaged.", "result = km.transform(X).min(axis=1).mean()"),
        ("transform", "Which point is farthest from its own center (a candidate outlier)? Return its index.",
         "result = int(km.transform(X).min(axis=1).argmax())"),
        ("cluster_centers_", "Return the index of the center closest to the origin `(0, 0)`.",
         "result = int(np.linalg.norm(km.cluster_centers_, axis=1).argmin())"),
        ("transform", f"How many points lie within distance {r} of their own center? Return the count.",
         f"result = int((km.transform(X).min(axis=1) <= {r}).sum())"),
        ("cluster_centers_", "Return the smallest distance between two different centers.",
         "C = km.cluster_centers_\nD = np.linalg.norm(C[:, None] - C[None], axis=2)\nresult = D[np.triu_indices(len(C), 1)].min()"),
        ("predict", f"Return `(cluster, distance)` for point {i}: its predicted cluster and its distance to that center.",
         f"c = int(km.predict(X[{i}:{i + 1}])[0])\nresult = (c, float(km.transform(X[{i}:{i + 1}])[0, c]))"),
    ]
    return catalog(rng, items, setup + f"\n{KM}km = {_km(k)}.fit(X)", f"{desc}; `km = {_km(k)}` has been fitted on `X`.",
                   tol=1e-6)


# ================================================================= medium

@template(M, "medium")
def customer_segments(rng):
    t = table(rng, numeric_only=True, n=(120, 250))
    k = rng.randint(3, 4)
    row = {c: 1 for c in t.num}
    pipe = (f"{KM}from sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\n"
            f"pipe = make_pipeline(StandardScaler(), {_km(k)}).fit(df)\nlabels = pipe[-1].labels_\n")
    items = [
        ("labels_", "Return the number of rows in each segment (`np.bincount` of the labels).",
         pipe + "result = np.bincount(labels)"),
        ("groupby", "Profile the segments: return `df.groupby(labels).mean()` (one row per segment, the column means "
                    "in original units).", pipe + "result = df.groupby(labels).mean()"),
        ("labels_", f"Which segment has the highest average target `y` ({t.target})? Return its label.",
         pipe + "result = int(pd.Series(y).groupby(labels).mean().idxmax())"),
        ("predict", "Return the segment of `new_row` (a dict; make a one-row DataFrame with `df`'s columns).",
         pipe + "result = int(pipe.predict(pd.DataFrame([new_row])[df.columns])[0])"),
        ("transform", "Return the indices of the 3 rows farthest from their own center in the scaled space "
                      "(`pipe.transform(df)` gives the distances to every center), farthest first.",
         pipe + "result = np.argsort(pipe.transform(df).min(axis=1))[::-1][:3]"),
        ("inverse_transform", "Return the segment centers in **original units**: map `cluster_centers_` back through the "
                              "scaler's `inverse_transform`.",
         pipe + "result = pipe[0].inverse_transform(pipe[-1].cluster_centers_)"),
        ("silhouette_score", "Return the `silhouette_score` (from `sklearn.metrics`) of the segments, measured on the "
                             "scaled data.",
         pipe + "from sklearn.metrics import silhouette_score\nresult = silhouette_score(pipe[0].transform(df), labels)"),
        ("cluster_centers_", "Which column separates the segments most? Return the column whose scaled centers have the "
                             "largest standard deviation across segments.",
         pipe + "result = df.columns[pipe[-1].cluster_centers_.std(axis=0).argmax()]"),
    ]
    return catalog(rng, items, t.setup + f"\nnew_row = {lit(row)}",
                   f"{t.desc}. Segment them: `pipe = make_pipeline(StandardScaler(), {_km(k)})`, fitted on `df` (not on "
                   f"`y`); `labels = pipe[-1].labels_`. `new_row` is a dict of feature values.", tol=1e-6)


@template(M, "medium")
def minibatch_vs_full(rng):
    setup, desc, k, n = _blobs(rng, n=(300, 500))
    b = rng.choice([32, 64, 100])
    task = rng.choice(["inertia", "ari", "partial_fit", "batch_sizes"])
    mbk = f"MiniBatchKMeans(n_clusters={k}, batch_size={b}, n_init=3, random_state=0)"
    imp = "from sklearn.cluster import KMeans, MiniBatchKMeans\n"
    if task == "inertia":
        return P(f"{desc}. Fit `{_km(k)}` and `{mbk}` on `X` and return their inertias as `(full, minibatch)`: the "
                 f"mini-batch version is cheaper and usually only slightly worse.", setup,
                 imp + f"result = ({_km(k)}.fit(X).inertia_, {mbk}.fit(X).inertia_)", require=["MiniBatchKMeans"],
                 tol=1e-6)
    if task == "ari":
        return P(f"{desc}. Cluster `X` with `{_km(k)}` and with `{mbk}` and return `adjusted_rand_score` (from "
                 f"`sklearn.metrics`) between the two labelings.", setup,
                 imp + "from sklearn.metrics import adjusted_rand_score\n"
                       f"result = adjusted_rand_score({_km(k)}.fit_predict(X), {mbk}.fit_predict(X))",
                 require=["MiniBatchKMeans", "adjusted_rand_score"], tol=1e-9)
    if task == "partial_fit":
        return P(f"{desc}. Pretend `X` arrives as a stream: create `mbk = MiniBatchKMeans(n_clusters={k}, "
                 f"random_state=0)` and call `mbk.partial_fit` on consecutive chunks of 50 rows (`X[i:i + 50]`). "
                 f"Return the final centers sorted by their first coordinate.", setup,
                 f"from sklearn.cluster import MiniBatchKMeans\nmbk = MiniBatchKMeans(n_clusters={k}, random_state=0)\n"
                 "for i in range(0, len(X), 50):\n    mbk.partial_fit(X[i:i + 50])\n"
                 "C = mbk.cluster_centers_\nresult = C[np.argsort(C[:, 0])]", require=["partial_fit"], tol=1e-6)
    sizes = [16, 64, 256]
    return P(f"{desc}. For `batch_size` in `{sizes}`, fit `MiniBatchKMeans(n_clusters={k}, batch_size=..., n_init=3, "
             f"random_state=0)` on `X` and return the list of inertias.", setup,
             f"from sklearn.cluster import MiniBatchKMeans\nresult = [MiniBatchKMeans(n_clusters={k}, batch_size=s, "
             f"n_init=3, random_state=0).fit(X).inertia_ for s in {sizes}]", require=["batch_size"], tol=1e-6)


@template(M, "medium")
def gmm_covariance_types(rng):
    setup, desc, k, n = _blobs(rng, k=(2, 3))
    s = seeded(rng)
    setup += f"\nX = X @ np.random.default_rng({s}).normal(0, 1, (2, 2))   # stretch and rotate the blobs"
    desc += ", then stretched and rotated by a random 2 × 2 matrix (so the clusters are elongated ellipses)"
    task = rng.choice(["bic", "best", "params", "shapes"])
    types = ["full", "tied", "diag", "spherical"]
    fit = (f"from sklearn.mixture import GaussianMixture\n"
           f"models = {{t: GaussianMixture(n_components={k}, covariance_type=t, random_state=0).fit(X) for t in {types}}}\n")
    head = f"{desc}. Fit `GaussianMixture(n_components={k}, covariance_type=t, random_state=0)` for each t in `{types}`."
    if task == "bic":
        return P(f"{head} Return a dict {{t: BIC on X}}.", setup, fit + "result = {t: m.bic(X) for t, m in models.items()}",
                 require=["covariance_type", "bic"], tol=1e-6)
    if task == "best":
        return P(f"{head} Return the covariance type with the lowest BIC (elongated clusters need off-diagonal "
                 f"covariance terms).", setup,
                 fit + "bic = {t: m.bic(X) for t, m in models.items()}\nresult = min(bic, key=bic.get)",
                 require=["covariance_type", "bic"])
    if task == "params":
        return P(f"{head} BIC and AIC differ only in how they charge for the p free parameters: `bic − aic = p·(ln n − "
                 f"2)`. Recover p for each type from `bic(X)` and `aic(X)` and return a dict {{t: p}} with p rounded to "
                 f"an int.", setup,
                 fit + "n = len(X)\nresult = {t: int(round((m.bic(X) - m.aic(X)) / (np.log(n) - 2))) for t, m in models.items()}",
                 require=["bic", "aic"])
    return P(f"{head} Return a dict {{t: covariances_.shape}} (shapes as tuples).", setup,
             fit + "result = {t: m.covariances_.shape for t, m in models.items()}", require=["covariances_"])


@template(M, "medium")
def bayesian_gmm_pruning(rng):
    setup, desc, k, n = _blobs(rng, k=(2, 4))
    cut = rng.choice([0.01, 0.02, 0.05])
    task = rng.choice(["active", "priors", "means"])
    ctor = "BayesianGaussianMixture(n_components=8, max_iter=500, random_state=0)"
    if task == "active":
        return P(f"{desc}. Fit `{ctor}` (from `sklearn.mixture`) on `X`. It starts with 8 components and shrinks the "
                 f"weights of the ones it doesn't need. Return `(number of components with weight > {cut}, {k})`: the "
                 f"first number should be close to the true number of blobs.", setup,
                 f"from sklearn.mixture import BayesianGaussianMixture\nbgm = {ctor}.fit(X)\n"
                 f"result = (int((bgm.weights_ > {cut}).sum()), {k})", require=["BayesianGaussianMixture", "weights_"])
    if task == "priors":
        priors = [0.001, 1.0, 1000.0]
        return P(f"{desc}. `weight_concentration_prior` controls how eagerly a `BayesianGaussianMixture` switches "
                 f"components off (small = fewer components). For each value in `{priors}`, fit "
                 f"`BayesianGaussianMixture(n_components=8, weight_concentration_prior=..., max_iter=500, "
                 f"random_state=0)` on `X` and return the list of numbers of components with weight > {cut}.", setup,
                 "from sklearn.mixture import BayesianGaussianMixture\n"
                 f"result = [int((BayesianGaussianMixture(n_components=8, weight_concentration_prior=p, max_iter=500, "
                 f"random_state=0).fit(X).weights_ > {cut}).sum()) for p in {priors}]",
                 require=["weight_concentration_prior"])
    return P(f"{desc}. Fit `{ctor}` on `X` and return the means of the components with weight > {cut}, sorted by their "
             f"first coordinate.", setup,
             f"from sklearn.mixture import BayesianGaussianMixture\nbgm = {ctor}.fit(X)\n"
             f"m = bgm.means_[bgm.weights_ > {cut}]\nresult = m[np.argsort(m[:, 0])]", require=["means_", "weights_"],
             tol=1e-5)


@template(M, "medium")
def density_params_sweep(rng):
    setup, desc, n = _moons(rng)
    task = rng.choice(["eps", "hdbscan", "probabilities", "optics_dbscan"])
    s = seeded(rng)
    setup += f"\nX = np.vstack([X, np.random.default_rng({s}).uniform(-1.5, 2.5, (15, 2))])   # plus 15 scattered points"
    desc += "; 15 uniformly scattered noise points were appended to `X` (they have no `y_true`)"
    if task == "eps":
        epss = [0.1, 0.15, 0.2, 0.3, 0.5]
        return P(f"{desc}. For `eps` in `{epss}`, fit `DBSCAN(eps=eps, min_samples=5)` and return the list of "
                 f"`(clusters, noise points)` pairs.", setup,
                 "from sklearn.cluster import DBSCAN\nresult = []\n"
                 f"for e in {epss}:\n    lab = DBSCAN(eps=e, min_samples=5).fit(X).labels_\n"
                 "    result.append((len(set(lab.tolist()) - {-1}), int((lab == -1).sum())))", require=["DBSCAN", "eps"])
    if task == "hdbscan":
        sizes = [5, 10, 20, 40]
        return P(f"{desc}. For `min_cluster_size` in `{sizes}`, fit `HDBSCAN(min_cluster_size=m, {HDB})` and return "
                 f"the list of numbers of clusters found.", setup,
                 f"from sklearn.cluster import HDBSCAN\nresult = [int(HDBSCAN(min_cluster_size=m, {HDB}).fit(X).labels_"
                 f".max()) + 1 for m in {sizes}]", require=["HDBSCAN", "min_cluster_size"])
    if task == "probabilities":
        return P(f"{desc}. Fit `hdb = HDBSCAN(min_cluster_size=10, {HDB})` on `X`. `hdb.probabilities_` says how "
                 f"strongly each point belongs to its cluster (0 for noise). Return `(number of noise points, number "
                 f"of clustered points with probability < 0.5)`.", setup,
                 f"from sklearn.cluster import HDBSCAN\nhdb = HDBSCAN(min_cluster_size=10, {HDB}).fit(X)\n"
                 "p, lab = hdb.probabilities_, hdb.labels_\nresult = (int((lab == -1).sum()), int(((lab >= 0) & (p < 0.5)).sum()))",
                 require=["probabilities_"])
    eps = rng.choice([0.15, 0.2])
    return P(f"{desc}. `OPTICS` can extract DBSCAN-like clusters at any radius after one fit. Fit `OPTICS(min_samples=5, "
             f"cluster_method='dbscan', eps={eps})` and `DBSCAN(eps={eps}, min_samples=5)` on `X` and return "
             f"`adjusted_rand_score` (from `sklearn.metrics`) between their labelings, and the two noise counts, as "
             f"`(ari, optics_noise, dbscan_noise)`.", setup,
             "from sklearn.cluster import DBSCAN, OPTICS\nfrom sklearn.metrics import adjusted_rand_score\n"
             f"a = OPTICS(min_samples=5, cluster_method='dbscan', eps={eps}).fit(X).labels_\n"
             f"b = DBSCAN(eps={eps}, min_samples=5).fit(X).labels_\n"
             "result = (adjusted_rand_score(a, b), int((a == -1).sum()), int((b == -1).sum()))",
             require=["OPTICS", "cluster_method"], tol=1e-9)


@template(M, "medium")
def k_distance_eps(rng):
    setup, desc, n = _moons(rng)
    ms, pct = rng.choice([4, 5, 8]), rng.choice([90, 95])
    task = rng.choice(["eps", "result"])
    sol = (f"from sklearn.neighbors import NearestNeighbors\nd, _ = NearestNeighbors(n_neighbors={ms}).fit(X).kneighbors(X)\n"
           f"eps = np.percentile(d[:, -1], {pct})\n")
    head = (f"{desc}. A common recipe for DBSCAN's `eps`: for every point, find the distance to its {ms}-th nearest "
            f"neighbour, counting the point itself (`NearestNeighbors(n_neighbors={ms}).fit(X).kneighbors(X)`, last "
            f"column), and take the {pct}th percentile of these distances.")
    if task == "eps":
        return P(f"{head} Return that `eps`.", setup, sol + "result = float(eps)", require=["kneighbors"], tol=1e-9)
    return P(f"{head} Then fit `DBSCAN(eps=eps, min_samples={ms})` and return `(eps, number of clusters, number of noise "
             f"points)`.", setup,
             sol + f"from sklearn.cluster import DBSCAN\nlab = DBSCAN(eps=eps, min_samples={ms}).fit(X).labels_\n"
                   "result = (float(eps), len(set(lab.tolist()) - {-1}), int((lab == -1).sum()))",
             require=["kneighbors", "DBSCAN"], tol=1e-9)


@template(M, "medium")
def spectral_vs_kmeans(rng):
    shape = rng.choice(["moons", "circles"])
    if shape == "moons":
        setup, desc, n = _moons(rng)
    else:
        n, s, nz = rng.randint(150, 300), seeded(rng), rng.choice([0.03, 0.05])
        setup = (f"from sklearn.datasets import make_circles\nX, y_true = make_circles(n_samples={n}, factor=0.5, "
                 f"noise={nz}, random_state={s})")
        desc = (f"`X` holds {n} 2-D points on two concentric circles (`make_circles`, `factor=0.5`, `noise={nz}`), and "
                f"`y_true` says which circle each point is on")
    task = rng.choice(["compare", "gammas", "affinity"])
    sc = "SpectralClustering(n_clusters=2, affinity='nearest_neighbors', n_neighbors=20, random_state=0)"
    if task == "compare":
        return P(f"{desc}. Cluster `X` into 2 groups with `{_km(2)}` and with `{sc}` (from `sklearn.cluster`). Return "
                 f"both `adjusted_rand_score(y_true, labels)` values as `(kmeans, spectral)`: spectral clustering "
                 f"follows the neighbourhood graph, so it can separate non-convex shapes.", setup,
                 "from sklearn.cluster import KMeans, SpectralClustering\nfrom sklearn.metrics import adjusted_rand_score\n"
                 f"result = (adjusted_rand_score(y_true, {_km(2)}.fit_predict(X)), "
                 f"adjusted_rand_score(y_true, {sc}.fit_predict(X)))", require=["SpectralClustering", "affinity"],
                 tol=1e-9)
    if task == "gammas":
        gammas = [1.0, 10.0, 50.0]
        return P(f"{desc}. With the default `affinity='rbf'`, `gamma` sets how fast similarity decays with distance. For "
                 f"`gamma` in `{gammas}`, cluster `X` with `SpectralClustering(n_clusters=2, gamma=..., random_state=0)` "
                 f"and return the list of `adjusted_rand_score(y_true, labels)` values.", setup,
                 "from sklearn.cluster import SpectralClustering\nfrom sklearn.metrics import adjusted_rand_score\n"
                 "result = [adjusted_rand_score(y_true, SpectralClustering(n_clusters=2, gamma=g, random_state=0)"
                 f".fit_predict(X)) for g in {gammas}]", require=["SpectralClustering", "gamma"], tol=1e-9)
    nn = rng.choice([20, 25, 30])
    return P(f"{desc}. Fit `sc = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', n_neighbors={nn}, "
             f"random_state=0)` on `X`. Its `affinity_matrix_` is a sparse neighbourhood graph. Return `(shape, number "
             f"of stored non-zeros (.nnz), adjusted_rand_score(y_true, sc.labels_))`.", setup,
             "from sklearn.cluster import SpectralClustering\nfrom sklearn.metrics import adjusted_rand_score\n"
             f"sc = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', n_neighbors={nn}, random_state=0).fit(X)\n"
             "A = sc.affinity_matrix_\nresult = (A.shape, int(A.nnz), adjusted_rand_score(y_true, sc.labels_))",
             require=["affinity_matrix_"], tol=1e-9)


@template(M, "medium")
def meanshift_bandwidth(rng):
    setup, desc, k, n = _blobs(rng, n=(120, 220))
    task = rng.choice(["clusters", "bandwidths", "predict"])
    qs = [0.05, 0.1, 0.2, 0.4]
    if task == "clusters":
        return P(f"{desc}. Mean shift's only knob is the kernel width. For `quantile` in `{qs}`, set `bandwidth = "
                 f"estimate_bandwidth(X, quantile=q)`, fit `MeanShift(bandwidth=bandwidth, bin_seeding=True)` and "
                 f"return the list of numbers of clusters found.", setup,
                 "from sklearn.cluster import MeanShift, estimate_bandwidth\n"
                 "result = [len(MeanShift(bandwidth=estimate_bandwidth(X, quantile=q), bin_seeding=True).fit(X)"
                 f".cluster_centers_) for q in {qs}]", require=["MeanShift", "estimate_bandwidth"])
    if task == "bandwidths":
        return P(f"{desc}. Return the list of `estimate_bandwidth(X, quantile=q)` values for `q` in `{qs}` (from "
                 f"`sklearn.cluster`): a larger quantile averages over more neighbours, so the bandwidth grows.", setup,
                 f"from sklearn.cluster import estimate_bandwidth\nresult = [estimate_bandwidth(X, quantile=q) for q in {qs}]",
                 require=["estimate_bandwidth", "quantile"], tol=1e-9)
    q = rng.choice([0.1, 0.2])
    return P(f"{desc}. Fit `ms = MeanShift(bandwidth=estimate_bandwidth(X, quantile={q}))` on `X`. Return the cluster "
             f"of the points `(0, 0)` and `(5, 5)` (`predict`) and the location of the center the first one is assigned "
             f"to, as `(labels, center)`.", setup,
             f"from sklearn.cluster import MeanShift, estimate_bandwidth\nms = MeanShift(bandwidth=estimate_bandwidth(X, "
             f"quantile={q})).fit(X)\nlab = ms.predict(np.array([[0.0, 0.0], [5.0, 5.0]]))\n"
             "result = (lab, ms.cluster_centers_[lab[0]])", require=["MeanShift", "predict"], tol=1e-6)


@template(M, "medium")
def cluster_features(rng):
    dd = synth_clf(rng, n=(200, 400), d=(4, 6), classes=2)
    ks = sorted(rng.sample([2, 4, 8, 16], 3))
    task = rng.choice(["distances", "pipeline", "label_onehot"])
    setup = dd.setup + ("\nfrom sklearn.model_selection import train_test_split\n"
                        "X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)")
    head = f"{dd.desc}; `X_tr, X_te, y_tr, y_te` is a 70/30 split."
    if task == "distances":
        k = ks[-1]
        return P(f"{head} Use clusters as features: fit `km = {_km(k)}` on `X_tr`, then train "
                 f"`LogisticRegression(max_iter=1000)` on `km.transform(X_tr)` (distances to the {k} centers). Return "
                 f"its test accuracy (transform `X_te` with the same `km`), next to a `LogisticRegression(max_iter=1000)` "
                 f"on the raw features, as `(distances, raw)`.", setup,
                 f"{KM}from sklearn.linear_model import LogisticRegression\nkm = {_km(k)}.fit(X_tr)\n"
                 "a = LogisticRegression(max_iter=1000).fit(km.transform(X_tr), y_tr).score(km.transform(X_te), y_te)\n"
                 "b = LogisticRegression(max_iter=1000).fit(X_tr, y_tr).score(X_te, y_te)\nresult = (a, b)",
                 require=["transform"], tol=1e-9)
    if task == "pipeline":
        return P(f"{head} `KMeans` is also a transformer, so it fits in a pipeline. For `n_clusters` in `{ks}`, fit "
                 f"`make_pipeline(KMeans(n_clusters=k, n_init='auto', random_state=0), LogisticRegression(max_iter=1000))` "
                 f"on the training split and return the list of test accuracies.", setup,
                 f"{KM}from sklearn.linear_model import LogisticRegression\nfrom sklearn.pipeline import make_pipeline\n"
                 f"result = [make_pipeline({_km('k')}, LogisticRegression(max_iter=1000)).fit(X_tr, y_tr).score(X_te, y_te) "
                 f"for k in {ks}]", require=["make_pipeline", "KMeans"], tol=1e-9)
    k = ks[1]
    return P(f"{head} Fit `km = {_km(k)}` on `X_tr` and append the one-hot encoded cluster label (`np.eye({k})[labels]`) "
             f"to the features of both splits. Return the test accuracy of `LogisticRegression(max_iter=1000)` trained on "
             f"the augmented training split.", setup,
             f"{KM}from sklearn.linear_model import LogisticRegression\nkm = {_km(k)}.fit(X_tr)\n"
             f"A = np.hstack([X_tr, np.eye({k})[km.labels_]])\nB = np.hstack([X_te, np.eye({k})[km.predict(X_te)]])\n"
             "result = LogisticRegression(max_iter=1000).fit(A, y_tr).score(B, y_te)", require=["predict"], tol=1e-9)


@template(M, "medium")
def birch_subclusters(rng):
    setup, desc, k, n = _blobs(rng, n=(200, 400))
    task = rng.choice(["count", "thresholds", "agree"])
    t = rng.choice([0.5, 0.8, 1.0])
    if task == "count":
        return P(f"{desc}. `Birch` first compresses the data into small subclusters (a CF-tree) whose radius stays "
                 f"under `threshold`. Fit `Birch(threshold={t}, n_clusters=None)` on `X` (no final clustering step) and "
                 f"return the number of subclusters (`subcluster_centers_`).", setup,
                 f"from sklearn.cluster import Birch\nresult = len(Birch(threshold={t}, n_clusters=None).fit(X).subcluster_centers_)",
                 require=["Birch", "subcluster_centers_"])
    if task == "thresholds":
        ts = [0.3, 0.6, 1.2, 2.4]
        return P(f"{desc}. For `threshold` in `{ts}`, fit `Birch(threshold=t, n_clusters=None)` on `X` and return the "
                 f"list of subcluster counts (`len(subcluster_centers_)`).", setup,
                 f"from sklearn.cluster import Birch\nresult = [len(Birch(threshold=t, n_clusters=None).fit(X)"
                 f".subcluster_centers_) for t in {ts}]", require=["Birch", "threshold"])
    return P(f"{desc}. Cluster `X` with `Birch(n_clusters={k})` and with `{_km(k)}` and return `adjusted_rand_score` "
             f"(from `sklearn.metrics`) between the two labelings and between Birch and `y_true`, as a tuple.", setup,
             "from sklearn.cluster import Birch, KMeans\nfrom sklearn.metrics import adjusted_rand_score\n"
             f"b = Birch(n_clusters={k}).fit_predict(X)\nresult = (adjusted_rand_score(b, {_km(k)}.fit_predict(X)), "
             "adjusted_rand_score(y_true, b))", require=["Birch", "n_clusters"], tol=1e-9)


@template(M, "medium")
def clusters_vs_species(rng):
    name = rng.choice(["iris", "wine"])
    fn = {"iris": "load_iris", "wine": "load_wine"}[name]
    n_ = {"iris": 150, "wine": 178}[name]
    setup = (f"from sklearn.datasets import {fn}\nfrom sklearn.preprocessing import StandardScaler\n"
             f"X, y = {fn}(return_X_y=True)\nX = StandardScaler().fit_transform(X)")
    desc = (f"`X` holds the {n_} {'iris flowers' if name == 'iris' else 'wines'}, standardized, and `y` their true "
            f"class (0, 1, 2), which the clustering never sees")
    ari = "from sklearn.metrics import adjusted_rand_score\n"
    items = [
        ("KMeans", f"Cluster `X` with `{_km(3)}` and return `adjusted_rand_score(y, labels)` (from `sklearn.metrics`).",
         f"{KM}{ari}result = adjusted_rand_score(y, {_km(3)}.fit_predict(X))"),
        ("crosstab", f"Cluster `X` with `{_km(3)}` and return `pd.crosstab(y, labels)` (true classes × clusters).",
         f"{KM}result = pd.crosstab(y, {_km(3)}.fit_predict(X))"),
        ("KMeans", f"Cluster `X` with `{_km(3)}`, name each cluster after the most common true class inside it, and "
                   f"return the accuracy of that naming.",
         f"{KM}lab = {_km(3)}.fit_predict(X)\nmapped = np.array([np.bincount(y[lab == c]).argmax() for c in lab])\n"
         "result = float((mapped == y).mean())"),
        ("GaussianMixture", f"Cluster `X` with `GaussianMixture(n_components=3, random_state=0)` and return "
                            f"`adjusted_rand_score(y, labels)`.",
         f"from sklearn.mixture import GaussianMixture\n{ari}"
         "result = adjusted_rand_score(y, GaussianMixture(n_components=3, random_state=0).fit_predict(X))"),
        ("AgglomerativeClustering", "Return a dict {linkage: ARI vs `y`} for `AgglomerativeClustering(n_clusters=3, "
                                    "linkage=...)` with linkage in `['ward', 'average', 'complete']`.",
         f"from sklearn.cluster import AgglomerativeClustering\n{ari}"
         "result = {l: adjusted_rand_score(y, AgglomerativeClustering(n_clusters=3, linkage=l).fit_predict(X)) "
         "for l in ['ward', 'average', 'complete']}"),
        ("homogeneity_score", f"Cluster `X` with `{_km(3)}` and return `(homogeneity_score(y, labels), "
                              f"completeness_score(y, labels))` (from `sklearn.metrics`): does each cluster hold one "
                              f"class, and does each class stay in one cluster?",
         f"{KM}from sklearn.metrics import homogeneity_score, completeness_score\nlab = {_km(3)}.fit_predict(X)\n"
         "result = (homogeneity_score(y, lab), completeness_score(y, lab))"),
    ]
    return catalog(rng, items, setup, f"{desc}.", tol=1e-6)


# ================================================================= hard

@template(M, "hard")
def lloyd_by_hand(rng):
    setup, desc, k, n = _blobs(rng)
    task = rng.choice(["converge", "one_step", "inertia"])
    head = (f"{desc}. Implement k-means (Lloyd's algorithm) **by hand**, starting from the first {k} points as centers "
            f"(`C = X[:{k}].copy()`): assign every point to its nearest center (squared Euclidean distance, `argmin`), "
            f"then move each center to the mean of its points.")
    if task == "one_step":
        return P(f"{head} Do **one** assign + update step and return the new centers; they equal "
                 f"`KMeans(n_clusters={k}, init=X[:{k}], n_init=1, max_iter=1).fit(X).cluster_centers_`.", setup,
                 f"C = X[:{k}].copy()\nlab = ((X[:, None] - C[None]) ** 2).sum(axis=2).argmin(axis=1)\n"
                 f"result = np.array([X[lab == j].mean(axis=0) for j in range({k})])", require=[], tol=1e-9)
    loop = (f"C = X[:{k}].copy()\nfor _ in range(300):\n    lab = ((X[:, None] - C[None]) ** 2).sum(axis=2).argmin(axis=1)\n"
            f"    new = np.array([X[lab == j].mean(axis=0) for j in range({k})])\n"
            "    if np.array_equal(new, C):\n        break\n    C = new\n")
    if task == "converge":
        return P(f"{head} Repeat until the centers stop moving and return the final centers; they equal "
                 f"`KMeans(n_clusters={k}, init=X[:{k}], n_init=1, tol=0).fit(X).cluster_centers_`.", setup,
                 loop + "result = C", require=[], tol=1e-9)
    return P(f"{head} Repeat until the centers stop moving, then return `(your_inertia, sklearn_inertia)` where the "
             f"second comes from `KMeans(n_clusters={k}, init=X[:{k}], n_init=1, tol=0).fit(X).inertia_`.", setup,
             loop + f"{KM}mine = ((X - C[lab]) ** 2).sum()\n"
                    f"result = (mine, KMeans(n_clusters={k}, init=X[:{k}], n_init=1, tol=0).fit(X).inertia_)",
             require=["KMeans", "inertia_"], tol=1e-9)


@template(M, "hard")
def kmeans_init_matters(rng):
    setup, desc, k, n = _blobs(rng, k=(5, 7), std=(0.6, 1.0))
    seeds = rng.randint(6, 10)
    task = rng.choice(["means", "bad_runs", "n_init"])
    rnd = f"[KMeans(n_clusters={k}, init='random', n_init=1, random_state=s).fit(X).inertia_ for s in range({seeds})]"
    pp = f"[KMeans(n_clusters={k}, init='k-means++', n_init=1, random_state=s).fit(X).inertia_ for s in range({seeds})]"
    if task == "means":
        return P(f"{desc}. k-means only finds a local optimum, so the starting centers matter. For random_state 0 .. "
                 f"{seeds - 1}, fit `KMeans(n_clusters={k}, init=..., n_init=1, random_state=s)` with `init='random'` "
                 f"and with `init='k-means++'`. Return the mean inertia of each, as `(random, kmeanspp)`.", setup,
                 f"{KM}a = {rnd}\nb = {pp}\nresult = (float(np.mean(a)), float(np.mean(b)))", require=["init"], tol=1e-6)
    if task == "bad_runs":
        return P(f"{desc}. For random_state 0 .. {seeds - 1}, fit `KMeans(n_clusters={k}, init='random', n_init=1, "
                 f"random_state=s)`. A run is *stuck* if its inertia is more than 1% above the best inertia of all "
                 f"runs. Return `(number of stuck runs, best inertia)`.", setup,
                 f"{KM}a = np.array({rnd})\nresult = (int((a > 1.01 * a.min()).sum()), float(a.min()))",
                 require=["init"], tol=1e-6)
    return P(f"{desc}. `n_init` reruns k-means from several starts and keeps the best. Return the inertias of "
             f"`KMeans(n_clusters={k}, init='random', n_init=m, random_state=0)` for m in 1, 3, 10, as a list (never "
             f"increasing).", setup,
             f"{KM}result = [KMeans(n_clusters={k}, init='random', n_init=m, random_state=0).fit(X).inertia_ for m in (1, 3, 10)]",
             require=["n_init"], tol=1e-6)


@template(M, "hard")
def gmm_e_step_by_hand(rng):
    setup, desc, k, n = _blobs(rng, std=(1.0, 2.0))
    nr = rng.randint(3, 6)
    task = rng.choice(["full", "score_samples", "diag"])
    ct = "diag" if task == "diag" else "full"
    fit = (f"from sklearn.mixture import GaussianMixture\ngmm = GaussianMixture(n_components={k}, covariance_type={ct!r}, "
           f"random_state=0).fit(X)\nXs = X[:{nr}]\n")
    if ct == "full":
        dens = ("def logpdf(x, m, S):\n    d = x - m\n    q = np.einsum('ij,jk,ik->i', d, np.linalg.inv(S), d)\n"
                "    return -0.5 * (q + np.log(np.linalg.det(S)) + x.shape[1] * np.log(2 * np.pi))\n"
                "L = np.column_stack([np.log(w) + logpdf(Xs, m, S) for w, m, S in zip(gmm.weights_, gmm.means_, gmm.covariances_)])\n")
        how = ("a full covariance matrix Σ_k per component: log N(x | μ, Σ) = −½((x−μ)ᵀΣ⁻¹(x−μ) + ln det Σ + d·ln 2π)")
    else:
        dens = ("L = np.column_stack([np.log(w) - 0.5 * (((Xs - m) ** 2 / v).sum(axis=1) + np.log(v).sum() + "
                "Xs.shape[1] * np.log(2 * np.pi)) for w, m, v in zip(gmm.weights_, gmm.means_, gmm.covariances_)])\n")
        how = ("one variance per feature and component (`covariances_` is k × d): log N(x | μ, v) = −½(Σ_j (x_j−μ_j)²/v_j "
               "+ Σ_j ln v_j + d·ln 2π)")
    head = (f"{desc}. Fit `gmm = GaussianMixture(n_components={k}, covariance_type={ct!r}, random_state=0)` on `X`; it "
            f"has {how}. Let `L[i, k] = ln weights_[k] + log N(x_i | means_[k], covariances_[k])` for the first {nr} "
            f"points `Xs = X[:{nr}]`.")
    if task == "score_samples":
        return P(f"{head} Return `log Σ_k exp(L[i, k])` for each point (use a stable log-sum-exp); it equals "
                 f"`gmm.score_samples(Xs)`.", setup,
                 fit + dens + "mx = L.max(axis=1)\nresult = mx + np.log(np.exp(L - mx[:, None]).sum(axis=1))",
                 require=["weights_", "means_", "covariances_"], tol=1e-6)
    return P(f"{head} The E-step turns `L` into responsibilities: `R = exp(L)` normalized per row (subtract each row's max "
             f"first). Return `R`; it equals `gmm.predict_proba(Xs)`.", setup,
             fit + dens + "E = np.exp(L - L.max(axis=1, keepdims=True))\nresult = E / E.sum(axis=1, keepdims=True)",
             require=["weights_", "means_", "covariances_"], tol=1e-6)


@template(M, "hard")
def dbscan_by_hand(rng):
    setup, desc, n = _moons(rng, n=(100, 200))
    eps, ms = rng.choice([0.15, 0.2, 0.25]), rng.choice([4, 5, 8])
    task = rng.choice(["core", "noise", "components"])
    base = ("D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(axis=2))\n"
            f"near = D <= {eps}\ncore = near.sum(axis=1) >= {ms}\n")
    head = (f"{desc}. Reimplement DBSCAN's bookkeeping with `eps={eps}`, `min_samples={ms}` **by hand** from the pairwise "
            f"distance matrix: a point is a *core* point if at least {ms} points (itself included) lie within distance "
            f"{eps} of it.")
    if task == "core":
        return P(f"{head} Return the sorted indices of the core points; they equal `DBSCAN(eps={eps}, "
                 f"min_samples={ms}).fit(X).core_sample_indices_`.", setup, base + "result = np.flatnonzero(core)",
                 require=[])
    if task == "noise":
        return P(f"{head} A non-core point within `eps` of some core point is a *border* point; every other non-core "
                 f"point is noise. Return the sorted indices of the noise points (DBSCAN labels them -1).", setup,
                 base + "result = np.flatnonzero(~core & ~(near[:, core].any(axis=1)))", require=[])
    return P(f"{head} Clusters are the connected groups of core points, linking two core points when they are within "
             f"`eps`. Count them with `connected_components` (from `scipy.sparse.csgraph`) on the core-to-core part of "
             f"the neighbour matrix and return `(your_count, number of clusters DBSCAN finds)`.", setup,
             base + f"from scipy.sparse.csgraph import connected_components\nfrom sklearn.cluster import DBSCAN\n"
                    "count = connected_components(near[np.ix_(core, core)], directed=False)[0]\n"
                    f"lab = DBSCAN(eps={eps}, min_samples={ms}).fit(X).labels_\n"
                    "result = (int(count), len(set(lab.tolist()) - {-1}))", require=["connected_components", "DBSCAN"])


@template(M, "hard")
def silhouette_by_hand(rng):
    setup, desc, k, n = _blobs(rng, n=(60, 120), std=(0.8, 1.5))
    task = rng.choice(["samples", "per_cluster", "worst"])
    sol = (f"{KM}lab = {_km(k)}.fit_predict(X)\nD = np.sqrt(((X[:, None] - X[None]) ** 2).sum(axis=2))\n"
           "s = np.zeros(len(X))\nfor i in range(len(X)):\n    own = lab == lab[i]\n"
           "    a = D[i, own].sum() / (own.sum() - 1)\n"
           "    b = min(D[i, lab == c].mean() for c in set(lab.tolist()) if c != lab[i])\n"
           "    s[i] = (b - a) / max(a, b)\n")
    head = (f"{desc}. Cluster it with `{_km(k)}` and compute the silhouette of every point **by hand**: `a` = mean "
            f"distance to the *other* points of its own cluster, `b` = the smallest mean distance to the points of "
            f"another cluster, `s = (b − a) / max(a, b)`.")
    if task == "samples":
        return P(f"{head} Return the array of `s`; it equals `silhouette_samples(X, labels)` (from `sklearn.metrics`) "
                 f"and its mean is `silhouette_score`.", setup, sol + "result = s", require=[], tol=1e-9)
    if task == "per_cluster":
        return P(f"{head} Return the mean silhouette of each cluster, as an array in label order.", setup,
                 sol + f"result = np.array([s[lab == c].mean() for c in range({k})])", require=[], tol=1e-9)
    return P(f"{head} Return the indices of the 3 points with the lowest silhouette (the worst fitted), lowest first.",
             setup, sol + "result = np.argsort(s)[:3]", require=[])


@template(M, "hard")
def cluster_stability(rng):
    setup, desc, k, n = _blobs(rng, std=(1.0, 2.0))
    s = seeded(rng)
    task = rng.choice(["matrix", "bootstrap", "kchoice"])
    if task == "matrix":
        return P(f"{desc}. How stable is k-means with a single random start? Cluster `X` with `KMeans(n_clusters={k}, "
                 f"init='random', n_init=1, random_state=r)` for r = 0 .. 4 and return the 5 × 5 matrix of pairwise "
                 f"`adjusted_rand_score`s between the labelings (1 on the diagonal).", setup,
                 f"{KM}from sklearn.metrics import adjusted_rand_score\n"
                 f"L = [KMeans(n_clusters={k}, init='random', n_init=1, random_state=r).fit_predict(X) for r in range(5)]\n"
                 "result = np.array([[adjusted_rand_score(a, b) for b in L] for a in L])",
                 require=["adjusted_rand_score"], tol=1e-9)
    if task == "bootstrap":
        return P(f"{desc}. Bootstrap stability: fit `ref = {_km(k)}` on `X`. Then, with `rs = "
                 f"np.random.default_rng({s})`, 5 times draw `idx = rs.integers(0, len(X), len(X))`, fit `{_km(k)}` on "
                 f"`X[idx]`, predict all of `X` and compute `adjusted_rand_score` with `ref.labels_`. Return the 5 "
                 f"scores as an array.", setup,
                 f"{KM}from sklearn.metrics import adjusted_rand_score\nref = {_km(k)}.fit(X)\n"
                 f"rs = np.random.default_rng({s})\nout = []\nfor _ in range(5):\n"
                 "    idx = rs.integers(0, len(X), len(X))\n"
                 f"    out.append(adjusted_rand_score(ref.labels_, {_km(k)}.fit(X[idx]).predict(X)))\nresult = np.array(out)",
                 require=["adjusted_rand_score", "predict"], tol=1e-9)
    ks = list(range(2, 7))
    return P(f"{desc}. Pick k by stability: for k in `{ks}`, split `X` into halves `A = X[::2]`, `B = X[1::2]`, fit "
             f"`KMeans(n_clusters=k, n_init='auto', random_state=0)` on each half, let both models predict all of `X`, "
             f"and compute `adjusted_rand_score` between the two predictions. Return the dict {{k: score}}.", setup,
             f"{KM}from sklearn.metrics import adjusted_rand_score\nresult = {{}}\nfor k in {ks}:\n"
             "    a = KMeans(n_clusters=k, n_init='auto', random_state=0).fit(X[::2]).predict(X)\n"
             "    b = KMeans(n_clusters=k, n_init='auto', random_state=0).fit(X[1::2]).predict(X)\n"
             "    result[k] = adjusted_rand_score(a, b)", require=["adjusted_rand_score"], tol=1e-9)


@template(M, "hard")
def clustering_pitfalls(rng):
    t = table(rng, numeric_only=True, n=(120, 220))
    k = rng.randint(3, 4)
    items = [
        ("StandardScaler", f"This clusters customers on raw columns, so the column with the biggest numbers decides "
                           f"everything:\n```\nlabels = {_km(k)}.fit_predict(df)\n```\nFix it by standardizing first "
                           f"(`make_pipeline(StandardScaler(), {_km(k)})`) and return the cluster sizes, largest first.",
         f"{KM}from sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\n"
         f"labels = make_pipeline(StandardScaler(), {_km(k)}).fit_predict(df)\n"
         "result = sorted(np.bincount(labels).tolist(), reverse=True)"),
        ("adjusted_rand_score", f"Two analysts clustered the same scaled data with `{_km(k)}` and "
                                f"`AgglomerativeClustering(n_clusters={k})` and compared them like this:\n```\n"
                                "agreement = (labels_a == labels_b).mean()\n```\nCluster ids are arbitrary names, so "
                                "this can be near 0 for identical groupings. Return the proper "
                                "`adjusted_rand_score(labels_a, labels_b)` (from `sklearn.metrics`) instead.",
         f"{KM}from sklearn.cluster import AgglomerativeClustering\nfrom sklearn.metrics import adjusted_rand_score\n"
         f"from sklearn.preprocessing import StandardScaler\nZ = StandardScaler().fit_transform(df)\n"
         f"labels_a = {_km(k)}.fit_predict(Z)\nlabels_b = AgglomerativeClustering(n_clusters={k}).fit_predict(Z)\n"
         "result = adjusted_rand_score(labels_a, labels_b)"),
        ("linkage", f"This raises a ValueError: Ward linkage only works with Euclidean distances.\n```\n"
                    f"labels = AgglomerativeClustering(n_clusters={k}, linkage='ward', metric='cosine').fit_predict(Z)\n```\n"
                    f"Keep the cosine metric but switch to `linkage='average'`, and return the cluster sizes, largest "
                    f"first (`Z` is `df` standardized).",
         f"from sklearn.cluster import AgglomerativeClustering\nfrom sklearn.preprocessing import StandardScaler\n"
         f"Z = StandardScaler().fit_transform(df)\n"
         f"labels = AgglomerativeClustering(n_clusters={k}, linkage='average', metric='cosine').fit_predict(Z)\n"
         "result = sorted(np.bincount(labels).tolist(), reverse=True)"),
        ("silhouette_score", "This picks the number of clusters with the lowest inertia, which always means the "
                             "largest k tried:\n```\nbest = min(range(2, 8), key=lambda k: KMeans(n_clusters=k, "
                             "n_init='auto', random_state=0).fit(Z).inertia_)\n```\nUse the highest `silhouette_score` "
                             "instead (`Z` is `df` standardized) and return `(best_k, its score)`.",
         f"{KM}from sklearn.metrics import silhouette_score\nfrom sklearn.preprocessing import StandardScaler\n"
         "Z = StandardScaler().fit_transform(df)\n"
         "scores = {k: silhouette_score(Z, KMeans(n_clusters=k, n_init='auto', random_state=0).fit_predict(Z)) for k in range(2, 8)}\n"
         "best = max(scores, key=scores.get)\nresult = (best, scores[best])"),
        ("components_", "`DBSCAN` has no `predict`, so this crashes:\n```\ndb = DBSCAN(eps=1.0, min_samples=5).fit(Z)\n"
                        "new_labels = db.predict(Z_new)\n```\nAssign each new row the label of its nearest core point "
                        "(`db.components_` are the core points, `db.labels_[db.core_sample_indices_]` their labels) if "
                        "that point is within `eps`, else -1. Return the labels of `Z_new` (`Z` is the first 80% of "
                        "the rows of `df` standardized, `Z_new` the rest, scaled with the same scaler).",
         "from sklearn.cluster import DBSCAN\nfrom sklearn.preprocessing import StandardScaler\n"
         "h = int(0.8 * len(df))\nsc = StandardScaler().fit(df.iloc[:h])\nZ, Z_new = sc.transform(df.iloc[:h]), sc.transform(df.iloc[h:])\n"
         "db = DBSCAN(eps=1.0, min_samples=5).fit(Z)\ncore_lab = db.labels_[db.core_sample_indices_]\n"
         "d = np.sqrt(((Z_new[:, None] - db.components_[None]) ** 2).sum(axis=2))\n"
         "j = d.argmin(axis=1)\nresult = np.where(d[np.arange(len(Z_new)), j] <= 1.0, core_lab[j], -1)"),
        ("predict", f"This fits k-means on the test rows separately, so cluster 0 in one split need not be cluster 0 in "
                    f"the other:\n```\ntrain_labels = {_km(k)}.fit_predict(Z_train)\n"
                    f"test_labels = {_km(k)}.fit_predict(Z_test)\n```\nFit once on the training rows (the first 70% "
                    f"of `df`, standardized with a scaler fitted on them) and `predict` the test rows. Return the "
                    f"test cluster sizes in label order (`np.bincount`, length {k}).",
         f"{KM}from sklearn.preprocessing import StandardScaler\nh = int(0.7 * len(df))\nsc = StandardScaler().fit(df.iloc[:h])\n"
         f"km = {_km(k)}.fit(sc.transform(df.iloc[:h]))\n"
         f"result = np.bincount(km.predict(sc.transform(df.iloc[h:])), minlength={k})"),
    ]
    return catalog(rng, items, t.setup, f"{t.desc}.", tol=1e-9)


@template(M, "hard")
def gmm_anomaly_detection(rng):
    setup, desc, k, n = _blobs(rng, n=(200, 300))
    m, s, pct = rng.randint(4, 8), seeded(rng), rng.choice([1, 2, 3])
    setup += (f"\nout = np.random.default_rng({s}).uniform(-15, 15, ({m}, 2))\nX = np.vstack([X, out])   # the last {m} "
              f"rows are injected outliers")
    task = rng.choice(["flagged", "recall", "new"])
    fit = (f"from sklearn.mixture import GaussianMixture\ngmm = GaussianMixture(n_components={k}, random_state=0).fit(X)\n"
           f"dens = gmm.score_samples(X)\ncut = np.percentile(dens, {pct})\n")
    head = (f"{desc}; then {m} uniform outliers were appended as the last rows of `X`. Fit `gmm = "
            f"GaussianMixture(n_components={k}, random_state=0)` on `X` and use `gmm.score_samples` (the log-density of "
            f"each point) to flag anomalies: a point is flagged if its log-density is below the "
            f"{ {1: '1st', 2: '2nd', 3: '3rd'}[pct]} percentile (`np.percentile`) of all the training log-densities.")
    if task == "flagged":
        return P(f"{head} Return the sorted indices of the flagged rows.", setup, fit + "result = np.flatnonzero(dens < cut)",
                 require=["score_samples"])
    if task == "recall":
        return P(f"{head} Return `(how many of the {m} injected outliers are flagged, how many flagged rows are not "
                 f"injected)`.", setup,
                 fit + f"flag = dens < cut\nresult = (int(flag[-{m}:].sum()), int(flag[:-{m}].sum()))",
                 require=["score_samples"])
    new = [[0.0, 0.0], [20.0, -20.0], [3.0, 3.0]]
    return P(f"{head} Then check the new points `{new}`: return a list of booleans, True when a point's log-density is "
             f"below the cut-off.", setup, fit + f"result = (gmm.score_samples(np.array({new})) < cut).tolist()",
             require=["score_samples"])


@template(M, "hard")
def hierarchical_tree(rng):
    setup, desc, k, n = _blobs(rng, n=(40, 90))
    task = rng.choice(["threshold", "distances", "last_merge"])
    if task == "threshold":
        ts = sorted(rng.sample([2.0, 5.0, 10.0, 20.0, 40.0], 3))
        return P(f"{desc}. Instead of fixing the number of clusters, cut the Ward tree at a height: for each "
                 f"`distance_threshold` in `{ts}`, fit `AgglomerativeClustering(n_clusters=None, "
                 f"distance_threshold=t)` and return the list of numbers of clusters it finds (`n_clusters_`).", setup,
                 "from sklearn.cluster import AgglomerativeClustering\n"
                 f"result = [AgglomerativeClustering(n_clusters=None, distance_threshold=t).fit(X).n_clusters_ for t in {ts}]",
                 require=["distance_threshold", "n_clusters_"])
    if task == "distances":
        return P(f"{desc}. Fit `AgglomerativeClustering(n_clusters=1, compute_distances=True)` (Ward) on `X`. "
                 f"`distances_` holds the height of each of the n − 1 merges in order. Return the 4 largest merge "
                 f"heights, largest first, and the number of clusters you get by cutting at the biggest jump between "
                 f"consecutive heights (1 + the number of merges above the jump), as `(heights, k)`.", setup,
                 "from sklearn.cluster import AgglomerativeClustering\n"
                 "d = np.sort(AgglomerativeClustering(n_clusters=1, compute_distances=True).fit(X).distances_)\n"
                 "jump = int(np.argmax(np.diff(d)))\nresult = (d[::-1][:4], len(d) - 1 - jump + 1)",
                 require=["compute_distances", "distances_"], tol=1e-9)
    return P(f"{desc}. Fit `agg = AgglomerativeClustering(n_clusters=1)` on `X`. `agg.children_[i]` names the two nodes "
             f"merged at step i (ids below n = {n} are points; id n + j is the node formed at step j). Return the sizes "
             f"of the two groups joined by the **last** merge, smaller first.", setup,
             "from sklearn.cluster import AgglomerativeClustering\nagg = AgglomerativeClustering(n_clusters=1).fit(X)\n"
             "n = len(X)\nsize = {}\nfor j, (a, b) in enumerate(agg.children_):\n"
             "    size[n + j] = size.get(a, 1) + size.get(b, 1)\n"
             "a, b = agg.children_[-1]\nresult = sorted([size.get(a, 1), size.get(b, 1)])",
             require=["children_"])


@template(M, "hard")
def color_quantization(rng):
    s, npx = seeded(rng), rng.randint(200, 400)
    setup = (f"g = np.random.default_rng({s})\npalette = g.uniform(0, 255, (6, 3))\n"
             f"pixels = (palette[g.integers(0, 6, {npx})] + g.normal(0, 12, ({npx}, 3))).clip(0, 255)")
    desc = (f"`pixels` holds the RGB colours (0-255) of a {npx}-pixel image made of about 6 colours plus "
            f"noise, one row per pixel")
    task = rng.choice(["mse", "sweep"])
    if task == "mse":
        k = rng.choice([3, 4, 6, 8])
        return P(f"{desc}. Colour quantization: fit `{_km(k)}` on `pixels`, replace every pixel by its cluster's center "
                 f"(`cluster_centers_[labels_]`), and return `(number of distinct colours left, mean squared error "
                 f"per value against the original)`.", setup,
                 f"{KM}km = {_km(k)}.fit(pixels)\nq = km.cluster_centers_[km.labels_]\n"
                 "result = (len(np.unique(q, axis=0)), float(((q - pixels) ** 2).mean()))",
                 require=["cluster_centers_", "labels_"], tol=1e-6)
    ks = [2, 4, 6, 10]
    return P(f"{desc}. For palette sizes k in `{ks}`, quantize the image with `KMeans(n_clusters=k, n_init='auto', "
             f"random_state=0)` (each pixel becomes its cluster's center) and return the list of mean squared errors "
             f"per value. Where does adding colours stop paying off?", setup,
             f"{KM}result = []\nfor k in {ks}:\n    km = KMeans(n_clusters=k, n_init='auto', random_state=0).fit(pixels)\n"
             "    result.append(float(((km.cluster_centers_[km.labels_] - pixels) ** 2).mean()))",
             require=["cluster_centers_"], tol=1e-6)


# ================================================================= challenge

@template(M, "challenge")
def kmeanspp_seeding(rng):
    n, k, s = rng.randint(8, 12), rng.randint(3, 4), seeded(rng)
    report = rng.choice(["indices", "potential", "potentials"])

    def gen(seed, n_, k_):
        return (f"g = np.random.default_rng({seed})\nX = g.normal(0, 1, ({n_}, 2)) + g.integers(0, 4, ({n_}, 1)) * 4\n"
                f"k = {k_}\nfirst = int(g.integers(0, {n_}))\nu = g.random(k - 1)\nreport = {report!r}")
    sol = ("d2 = ((X - X[first]) ** 2).sum(axis=1)\nchosen, pots = [first], [float(d2.sum())]\nfor r in u:\n"
           "    c = np.cumsum(d2)\n    j = int(np.searchsorted(c, r * c[-1], side='right'))\n    chosen.append(j)\n"
           "    d2 = np.minimum(d2, ((X - X[j]) ** 2).sum(axis=1))\n    pots.append(float(d2.sum()))\n"
           "result = {'indices': chosen, 'potential': pots[-1], 'potentials': np.array(pots)}[report]")
    tests = [gen(s + i, 50000, 40) for i in (1, 2)]
    what = {"indices": "the list of the k chosen row indices, in order",
            "potential": "the final potential: Σ over all points of the squared distance to the nearest chosen center",
            "potentials": "the potential after each center is added (an array of length k)"}[report]
    return challenge(
        "k-means++ seeding",
        "k-means++ picks initial centers far apart. The first center is row `first` of `X`. Then, k − 1 times: let "
        "`D2[i]` be point i's squared distance to its nearest chosen center and `c = np.cumsum(D2)`; with the next "
        "uniform draw `r` from `u`, the next center is the smallest index `j` with `c[j] > r · c[-1]` (so a point is "
        f"picked with probability proportional to `D2`). Return {what}. (`report` = {report!r}.) scikit-learn's "
        "version also tries a few candidates per step; here every step uses exactly one draw.",
        gen(s, n, k), sol, tests, "n ≤ 5·10^4 points in 2-D, k ≤ 40",
        speed="O(n·k): update the nearest distances incrementally", tol=1e-6)


@template(M, "challenge")
def lloyd_iterations(rng):
    n, k, s, it = rng.randint(10, 14), rng.randint(2, 3), seeded(rng), rng.randint(1, 3)
    report = rng.choice(["inertia", "centers", "sizes"])

    def gen(seed, n_, k_, it_):
        return (f"g = np.random.default_rng({seed})\nX = (g.normal(0, 1, ({n_}, 2)) + g.integers(0, 3, ({n_}, 1)) * 3).round(3)\n"
                f"C0 = X[g.choice({n_}, {k_}, replace=False)].copy()\niters = {it_}\nreport = {report!r}")
    sol = ("C = C0.astype(float).copy()\nk = len(C)\n"
           "def assign(C):\n    return ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)\nfor _ in range(iters):\n    lab = assign(C)\n"
           "    cnt = np.bincount(lab, minlength=k)\n    S = np.zeros_like(C)\n    np.add.at(S, lab, X)\n"
           "    C = np.where(cnt[:, None] > 0, S / np.maximum(cnt, 1)[:, None], C)\nlab = assign(C)\n"
           "if report == 'inertia':\n    result = float(((X - C[lab]) ** 2).sum())\nelif report == 'centers':\n"
           "    result = C\nelse:\n    result = np.bincount(lab, minlength=k)")
    tests = [gen(s + i, 100000, 8, 10) for i in (1, 2)]
    what = {"inertia": "the final inertia: Σ of squared distances from each point to its nearest final center",
            "centers": "the final centers (a k × 2 array, in the order of `C0`)",
            "sizes": "the number of points nearest to each final center (an int array of length k)"}[report]
    return challenge(
        "Lloyd's iterations",
        "Run exactly `iters` iterations of k-means (Lloyd's algorithm) from the centers `C0`: assign every point of "
        "`X` to its nearest center (squared Euclidean distance; ties go to the lower index), then move every center to "
        "the mean of its points (a center with no points stays where it is). After the last iteration, assign the "
        f"points once more to the final centers. Return {what}. (`report` = {report!r}.)",
        gen(s, n, k, it), sol, tests, "n ≤ 10^5 points in 2-D, k ≤ 8, iters ≤ 10",
        speed="O(iters·n·k) vectorized: no Python loop over points", tol=1e-6)


@template(M, "challenge")
def silhouette_1d(rng):
    n, k, s = rng.randint(9, 12), rng.randint(2, 3), seeded(rng)
    report = rng.choice(["mean", "samples", "per_cluster"])

    def gen(seed, n_, k_):
        return (f"g = np.random.default_rng({seed})\nlabels = g.permutation(np.arange({n_}) % {k_})\n"
                f"x = (labels * 5 + g.normal(0, 2, {n_})).round(2)\nreport = {report!r}")
    sol = ("k = int(labels.max()) + 1\nn = len(x)\nmeans = np.zeros((n, k))\nsizes = np.bincount(labels, minlength=k)\n"
           "for c in range(k):\n    v = np.sort(x[labels == c])\n    pre = np.concatenate([[0.0], np.cumsum(v)])\n"
           "    j = np.searchsorted(v, x, side='right')\n"
           "    means[:, c] = x * j - pre[j] + (pre[-1] - pre[j]) - x * (len(v) - j)\n"
           "own = means[np.arange(n), labels]\na = own / np.maximum(sizes[labels] - 1, 1)\n"
           "other = means / np.maximum(sizes, 1)\nother[np.arange(n), labels] = np.inf\nb = other.min(axis=1)\n"
           "sil = np.where(sizes[labels] > 1, (b - a) / np.maximum(a, b), 0.0)\n"
           "if report == 'mean':\n    result = float(sil.mean())\nelif report == 'samples':\n    result = sil\n"
           "else:\n    result = np.array([sil[labels == c].mean() for c in range(k)])")
    tests = [gen(s + i, 200000, 5) for i in (1, 2)]
    what = {"mean": "the silhouette score (the mean over all points)",
            "samples": "every point's silhouette (an array)",
            "per_cluster": "the mean silhouette of each cluster, in label order"}[report]
    return challenge(
        "Silhouette on a line",
        "Points on a line `x` carry cluster `labels` (0..k−1). A point's silhouette is `(b − a) / max(a, b)`, where `a` "
        "is its mean distance `|x_i − x_j|` to the other points of its own cluster and `b` the smallest mean distance "
        "to the points of another cluster (a point alone in its cluster gets 0), exactly as `silhouette_samples` "
        f"computes it. Return {what}. (`report` = {report!r}.) The pairwise distance matrix is far too big: in 1-D, "
        "the sum of distances from a value to a sorted set comes from prefix sums and a binary search.",
        gen(s, n, k), sol, tests, "n ≤ 2·10^5, k ≤ 5", speed="O(k·n log n)", tol=1e-6)


@template(M, "challenge")
def dbscan_1d(rng):
    n, s = rng.randint(12, 16), seeded(rng)
    report = rng.choice(["n_clusters", "n_noise", "n_core"])
    eps, ms = rng.choice([1.5, 2.5]), rng.randint(2, 4)

    def gen(seed, n_, hi):
        return (f"g = np.random.default_rng({seed})\nx = g.integers(0, {hi}, {n_}).astype(float)\n"
                f"eps, min_samples = {eps}, {ms}\nreport = {report!r}")
    sol = ("xs = np.sort(x)\ncnt = np.searchsorted(xs, xs + eps, side='right') - np.searchsorted(xs, xs - eps, side='left')\n"
           "core = cnt >= min_samples\ncv = xs[core]\nif report == 'n_core':\n    result = int(core.sum())\n"
           "elif report == 'n_clusters':\n    result = int(len(cv) > 0) + int((np.diff(cv) > eps).sum())\nelse:\n"
           "    j = np.searchsorted(cv, xs)\n    lo = np.where(j > 0, xs - cv[np.maximum(j - 1, 0)], np.inf) if len(cv) else np.full(len(xs), np.inf)\n"
           "    hi = np.where(j < len(cv), cv[np.minimum(j, len(cv) - 1)] - xs, np.inf) if len(cv) else np.full(len(xs), np.inf)\n"
           "    result = int((~core & (np.minimum(lo, hi) > eps)).sum())")
    tests = [gen(s + i, 200000, 600000) for i in (1, 2)]
    what = {"n_clusters": "the number of clusters", "n_noise": "the number of noise points",
            "n_core": "the number of core points"}[report]
    return challenge(
        "DBSCAN on a line",
        "Run DBSCAN on the 1-D points `x` (duplicates allowed) with radius `eps` and `min_samples`: a point is *core* "
        "if at least `min_samples` points (itself included) lie within distance `eps`; core points within `eps` of "
        "each other are in the same cluster (chains link them); a non-core point within `eps` of a core point joins a "
        "cluster, any other point is noise. Return "
        f"{what}, as `DBSCAN(eps=eps, min_samples=min_samples).fit(x.reshape(-1, 1))` would find. (`report` = "
        f"{report!r}.)", gen(s, n, 40), sol, tests, "len(x) ≤ 2·10^5", speed="O(n log n): sort and binary search")


@template(M, "challenge")
def single_linkage_1d(rng):
    n, s = rng.randint(8, 12), seeded(rng)
    report = rng.choice(["sizes", "threshold", "heights"])
    k, m = rng.randint(2, 4), rng.randint(2, 4)

    def gen(seed, n_, k_, t):
        return (f"g = np.random.default_rng({seed})\nx = g.normal(0, 1, {n_}) + g.integers(0, 4, {n_}) * 4\n"
                f"k, t, m = {k_}, {t}, {m}\nreport = {report!r}")
    t = rng.choice([0.6, 0.9, 1.3])
    sol = ("gaps = np.diff(np.sort(x))\nif report == 'sizes':\n    xs = np.sort(x)\n"
           "    cut = np.sort(np.argsort(gaps)[::-1][:k - 1])\n    bounds = np.concatenate([[0], cut + 1, [len(xs)]])\n"
           "    result = sorted(np.diff(bounds).tolist(), reverse=True)\nelif report == 'threshold':\n"
           "    result = 1 + int((gaps >= t).sum())\nelse:\n    result = np.sort(gaps)[::-1][:m]")
    tests = [gen(s + i, 200000, 50, 0.002) for i in (1, 2)]
    what = {"sizes": "the cluster sizes when the hierarchy is cut into `k` clusters (`n_clusters=k`), largest first",
            "threshold": "the number of clusters left when merging stops at `distance_threshold=t` (clusters merge "
                         "only while their distance is below `t`)",
            "heights": "the heights of the last `m` merges, largest first (an array)"}[report]
    return challenge(
        "Single linkage on a line",
        "Single-linkage agglomerative clustering repeatedly merges the two clusters whose closest points are nearest; "
        "the merge happens at that distance (its height). Do it for the 1-D points `x` (all distinct) and return "
        f"{what}, as `AgglomerativeClustering(linkage='single')` would. (`report` = {report!r}.) Hint: in one "
        "dimension the merges follow the gaps between neighbouring sorted values.",
        gen(s, n, k, t), sol, tests, "len(x) ≤ 2·10^5", speed="O(n log n)", tol=1e-9)


@template(M, "challenge")
def gmm_responsibilities(rng):
    n, K, s = rng.randint(8, 12), rng.randint(2, 3), seeded(rng)
    report = rng.choice(["loglik", "counts", "nk", "means"])

    def gen(seed, n_, K_):
        return (f"g = np.random.default_rng({seed})\nK = {K_}\nw = g.dirichlet(np.ones(K) * 3)\n"
                "mu = np.sort(g.uniform(-10, 10, K))\nsd = g.uniform(0.3, 1.0, K)\n"
                f"x = np.concatenate([g.normal(mu[g.choice(K, {n_} - 2, p=w)], 1.0), g.uniform(-400, 400, 2)])\n"
                f"report = {report!r}")
    sol = ("L = np.log(w) - np.log(sd) - 0.5 * np.log(2 * np.pi) - 0.5 * ((x[:, None] - mu) / sd) ** 2\n"
           "mx = L.max(axis=1, keepdims=True)\nlse = mx[:, 0] + np.log(np.exp(L - mx).sum(axis=1))\n"
           "R = np.exp(L - lse[:, None])\nif report == 'loglik':\n    result = float(lse.mean())\n"
           "elif report == 'counts':\n    result = np.bincount(L.argmax(axis=1), minlength=K)\n"
           "elif report == 'nk':\n    result = R.sum(axis=0)\nelse:\n    result = (R * x[:, None]).sum(axis=0) / R.sum(axis=0)")
    tests = [gen(s + i, 200000, 4) for i in (1, 2)]
    what = {"loglik": "the average log-likelihood of `x` (what `gmm.score` returns)",
            "counts": "how many points get each component as their most probable one (`predict`), as an int array",
            "nk": "the soft counts `N_k = Σ_i r_ik` (an array of length K)",
            "means": "the M-step's updated means `Σ_i r_ik·x_i / N_k` (an array of length K)"}[report]
    return challenge(
        "Stable E-step",
        "A 1-D Gaussian mixture has weights `w`, means `mu` and standard deviations `sd` (K components). For a point "
        "x, `log p(x, k) = log w_k + log N(x | mu_k, sd_k²)` and the responsibility `r_ik = p(x_i, k) / Σ_j p(x_i, j)`. "
        "Some points lie hundreds of standard deviations from every component, so computing `exp` first underflows "
        f"to 0: work in log space (log-sum-exp). Return {what}. (`report` = {report!r}.)",
        gen(s, n, K), sol, tests, "len(x) ≤ 2·10^5, K ≤ 4", speed="O(n·K) vectorized", tol=1e-6)
