"""5.15 ML visualization recipes: thresholds tried one by one, scikit-learn's own AUC, and NumPy's np.histogram (what
plt.hist uses) per layer."""

from fractions import Fraction

import numpy as np

from . import brute


def _counts(scores, labels, t):
    tp = sum(1 for v, l in zip(scores, labels) if v >= t and l == 1)
    fp = sum(1 for v, l in zip(scores, labels) if v >= t and l == 0)
    return tp, fp


@brute
def roc_from_scores(ns):
    s, y = ns["scores"].tolist(), ns["labels"].tolist()
    rep = ns["report"]
    if rep == "auc":
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y, s))
    P = sum(y)
    N = len(y) - P
    pts, best = [(0.0, 0.0)], None
    for t in sorted(set(s), reverse=True):
        tp, fp = _counts(s, y, t)
        pts.append((fp / N, tp / P))
        j = Fraction(tp, P) - Fraction(fp, N)
        if best is None or j > best[0]:          # strictly better: ties keep the higher threshold
            best = (j, t)
    return np.array(pts) if rep == "points" else float(best[1])


@brute
def pr_operating_point(ns):
    s, y = ns["scores"].tolist(), ns["labels"].tolist()
    P = sum(y)
    rep = ns["report"]
    if rep == "at":
        out = []
        for t in ns["thr"].tolist():
            tp, fp = _counts(s, y, t)
            out.append([tp / (tp + fp) if tp + fp else 1.0, tp / P])
        return np.array(out)
    best, rec = None, 0.0
    for t in sorted(set(s), reverse=True):
        tp, fp = _counts(s, y, t)
        f1 = Fraction(2 * tp, 2 * tp + fp + (P - tp))
        if best is None or f1 > best[0]:
            best = (f1, t)
        if tp / (tp + fp) >= ns["p_min"]:
            rec = max(rec, tp / P)
    return [float(best[1]), float(best[0])] if rep == "best_f1" else rec


@brute
def layer_histograms(ns):
    w, layer, L, B = ns["w"], ns["layer"], ns["L"], ns["bins"]
    rng = (float(w.min()), float(w.max()))
    counts = np.array([np.histogram(w[layer == l], bins=B, range=rng)[0] for l in range(L)])
    rep = ns["report"]
    if rep == "counts":
        return counts
    if rep == "mode":
        return np.array([max(range(B), key=lambda b: (c[b], -b)) for c in counts.tolist()])
    return np.array([sum(min(a, b) for a, b in zip(counts[l], counts[l + 1])) for l in range(L - 1)])
