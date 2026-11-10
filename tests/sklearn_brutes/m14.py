"""4.14 Regression, clustering & distance metrics."""

import numpy as np

from . import brute


@brute
def silhouette_on_a_line(ns):
    from sklearn.metrics import silhouette_samples
    x, labels = ns["x"], ns["labels"]
    s = silhouette_samples(x.reshape(-1, 1), labels)
    if ns["report"] == "mean":
        return float(s.mean())
    if ns["report"] == "negatives":
        return int((s < 0).sum())
    return np.array([s[labels == c].mean() for c in range(labels.max() + 1)])


@brute
def prefix_adjusted_rand(ns):
    from sklearn.metrics import adjusted_rand_score, rand_score
    a, b = ns["a"], ns["b"]
    fn = adjusted_rand_score if ns["report"] == "ari" else rand_score
    return np.array([fn(a[:t], b[:t]) for t in range(2, len(a) + 1)], dtype=float)


@brute
def ndcg_every_cutoff(ns):
    from sklearn.metrics import dcg_score, ndcg_score
    fn = ndcg_score if ns["report"] == "ndcg" else dcg_score
    return np.array([fn([ns["rel"]], [ns["score"]], k=int(k)) for k in ns["ks"]], dtype=float)


@brute
def manhattan_row_sums(ns):
    from sklearn.metrics.pairwise import manhattan_distances
    rows = manhattan_distances(ns["X"]).sum(axis=1)
    return {"total": float(rows.sum() / 2), "row_sums": rows, "medoid": int(rows.argmin())}[ns["report"]]
