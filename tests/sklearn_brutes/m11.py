"""4.11 Clustering & mixture models."""

import numpy as np

from . import brute


@brute
def kmeanspp_seeding(ns):
    X, u = ns["X"], ns["u"]
    chosen = [ns["first"]]
    pots = []

    def d2(i):
        return min(float(((X[i] - X[c]) ** 2).sum()) for c in chosen)
    for r in list(u) + [None]:
        D = [d2(i) for i in range(len(X))]
        pots.append(sum(D))
        if r is None:
            break
        total, acc = sum(D), 0.0
        for j, v in enumerate(D):
            acc += v
            if acc > r * total:
                chosen.append(j)
                break
    return {"indices": chosen, "potential": pots[-1], "potentials": np.array(pots)}[ns["report"]]


@brute
def lloyd_iterations(ns):
    X, C = ns["X"], [c.astype(float) for c in ns["C0"]]
    k = len(C)

    def assign():
        return [min(range(k), key=lambda j: (((x - C[j]) ** 2).sum(), j)) for x in X]
    for _ in range(ns["iters"]):
        lab = assign()
        for j in range(k):
            pts = [x for x, l in zip(X, lab) if l == j]
            if pts:
                C[j] = np.mean(pts, axis=0)
    lab = assign()
    if ns["report"] == "inertia":
        return float(sum(((x - C[l]) ** 2).sum() for x, l in zip(X, lab)))
    if ns["report"] == "centers":
        return np.array(C)
    return np.bincount(lab, minlength=k)


@brute
def silhouette_1d(ns):
    from sklearn.metrics import silhouette_samples
    x, lab = ns["x"], ns["labels"]
    s = silhouette_samples(x.reshape(-1, 1), lab)
    if ns["report"] == "mean":
        return float(s.mean())
    if ns["report"] == "samples":
        return s
    return np.array([s[lab == c].mean() for c in range(lab.max() + 1)])


@brute
def dbscan_1d(ns):
    from sklearn.cluster import DBSCAN
    db = DBSCAN(eps=ns["eps"], min_samples=ns["min_samples"]).fit(ns["x"].reshape(-1, 1))
    lab = db.labels_
    return {"n_clusters": len(set(lab.tolist()) - {-1}), "n_noise": int((lab == -1).sum()),
            "n_core": len(db.core_sample_indices_)}[ns["report"]]


@brute
def single_linkage_1d(ns):
    from sklearn.cluster import AgglomerativeClustering
    X = ns["x"].reshape(-1, 1)
    if ns["report"] == "sizes":
        lab = AgglomerativeClustering(n_clusters=ns["k"], linkage="single").fit_predict(X)
        return sorted(np.bincount(lab).tolist(), reverse=True)
    if ns["report"] == "threshold":
        return AgglomerativeClustering(n_clusters=None, distance_threshold=ns["t"], linkage="single").fit(X).n_clusters_
    d = AgglomerativeClustering(n_clusters=1, linkage="single", compute_distances=True).fit(X).distances_
    return np.sort(d)[::-1][:ns["m"]]


@brute
def gmm_responsibilities(ns):
    from sklearn.mixture import GaussianMixture
    K, x = ns["K"], ns["x"]
    gm = GaussianMixture(n_components=K, covariance_type="diag")
    gm.weights_ = ns["w"]
    gm.means_ = ns["mu"].reshape(-1, 1)
    gm.covariances_ = (ns["sd"] ** 2).reshape(-1, 1)
    gm.precisions_cholesky_ = 1 / ns["sd"].reshape(-1, 1)
    X = x.reshape(-1, 1)
    if ns["report"] == "loglik":
        return float(gm.score(X))
    if ns["report"] == "counts":
        return np.bincount(gm.predict(X), minlength=K)
    R = gm.predict_proba(X)
    return R.sum(axis=0) if ns["report"] == "nk" else (R * x[:, None]).sum(axis=0) / R.sum(axis=0)
