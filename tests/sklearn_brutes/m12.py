"""4.12 Dimensionality reduction & embeddings."""

import numpy as np

from . import brute


@brute
def pca_threshold_queries(ns):
    v = sorted(ns["variances"].tolist(), reverse=True)
    total = sum(v)
    c = np.cumsum(np.array(v) / total)
    out = []
    for t in ns["thresholds"]:
        k = next((i + 1 for i in range(len(v)) if c[i] > t), len(v))
        out.append({"n": k, "kept": c[k - 1], "dropped": float(sum(v[k:]))}[ns["report"]])
    return np.array(out) if ns["report"] != "n" else np.array(out, dtype=int)


@brute
def nmf_mu_updates(ns):
    from sklearn.decomposition import NMF
    k, X = ns["W0"].shape[1], ns["X"]
    m = NMF(n_components=k, init="custom", solver="mu", beta_loss="frobenius", max_iter=ns["T"], tol=0)
    W = m.fit_transform(X, W=ns["W0"].astype(float).copy(), H=ns["H0"].astype(float).copy())
    return {"error": float(m.reconstruction_err_), "W": W, "H": m.components_}[ns["report"]]


@brute
def tsne_perplexity_search(ns):
    X, perp = ns["X"], ns["perplexity"]
    n = len(X)
    P = np.zeros((n, n))
    sig, pmax = [], []
    for i in range(n):
        d = np.array([((X[i] - X[j]) ** 2).sum() for j in range(n) if j != i])
        lo, hi = -30.0, 30.0

        def row(lb):
            e = np.exp(-np.exp(lb) * (d - d.min()))
            return e / e.sum()
        for _ in range(100):
            mid = (lo + hi) / 2
            p = row(mid)
            H = -sum(x * np.log(x) for x in p if x > 0)
            if H < np.log(perp):
                hi = mid
            else:
                lo = mid
        lb = (lo + hi) / 2
        p = row(lb)
        sig.append(np.sqrt(1 / (2 * np.exp(lb))))
        pmax.append(p.max())
        P[i, [j for j in range(n) if j != i]] = p
    if ns["report"] == "sigma":
        return np.array(sig)
    if ns["report"] == "pmax":
        return np.array(pmax)
    return ((P + P.T) / (2 * n)).max(axis=1)


@brute
def lda_fisher_ratio(ns):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from scipy import linalg
    X, y = ns["X"], ns["y"]
    if ns["report"] == "ratio":
        return LinearDiscriminantAnalysis(solver="eigen").fit(X, y).explained_variance_ratio_
    Sw = sum((y == c).mean() * np.cov(X[y == c], rowvar=False, bias=True) for c in np.unique(y))
    Sb = np.cov(X, rowvar=False, bias=True) - Sw
    return float(linalg.eigh(Sb, Sw, eigvals_only=True).max())


@brute
def reconstruction_error_curve(ns):
    from sklearn.decomposition import PCA
    X = ns["X"]
    n, d = X.shape
    mse = [float(((X - X.mean(axis=0)) ** 2).mean())]
    for k in range(1, d + 1):
        pca = PCA(n_components=k).fit(X)
        mse.append(float(((X - pca.inverse_transform(pca.transform(X))) ** 2).mean()))
    mse = np.array(mse)
    if ns["report"] == "mse":
        return mse
    if ns["report"] == "k_for":
        return next(k for k in range(d + 1) if mse[k] <= ns["frac"] * mse[0] + 1e-12)
    return 1 - mse / mse[0]


@brute
def trustworthiness_score(ns):
    from sklearn.manifold import trustworthiness
    X, Z, k = ns["X"], ns["Z"], ns["k"]
    if ns["report"] == "trust":
        return float(trustworthiness(X, Z, n_neighbors=k))
    if ns["report"] == "continuity":
        return float(trustworthiness(Z, X, n_neighbors=k))
    n = len(X)

    def knn(A, i):
        d = [(((A[i] - A[j]) ** 2).sum(), j) for j in range(n) if j != i]
        return {j for _, j in sorted(d)[:k]}
    return sum(len(knn(Z, i) - knn(X, i)) for i in range(n))
