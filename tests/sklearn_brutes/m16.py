"""4.16 Probability calibration & decision thresholds."""

import numpy as np

from . import brute


@brute
def isotonic_calibration_pava(ns):
    from sklearn.isotonic import IsotonicRegression
    score, y = ns["score"], ns["y"]
    iso = IsotonicRegression(out_of_bounds="clip").fit(score, y)
    fitted = iso.predict(score)
    return {"fitted": fitted, "predict": iso.predict(ns["q"]),
            "brier": float(((fitted - y) ** 2).mean())}[ns["report"]]


@brute
def best_threshold_scan(ns):
    p, y, report = ns["p"], ns["y"], ns["report"]
    best = None
    for t in sorted(set(p.tolist())):              # increasing: keep the first (smallest) on ties
        pred = p >= t
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        if report == "f1":
            val = 2 * tp / (2 * tp + fp + fn)
            if best is None or val > best[1]:
                best = (float(t), val)
        elif report == "cost":
            val = ns["c_fn"] * fn + ns["c_fp"] * fp
            if best is None or val < best[1]:
                best = (float(t), val)
        elif 100 * tp >= ns["R"] * (y == 1).sum():  # recall: keep the largest such t
            best = (float(t), tp / (tp + fp))
    return best


@brute
def calibration_error_bins(ns):
    from sklearn.calibration import calibration_curve
    p, y, nb = ns["p"], ns["y"], ns["n_bins"]
    prob_true, prob_pred = calibration_curve(y, p, n_bins=nb)
    if ns["report"] == "curve":
        return prob_true
    edges = np.linspace(0, 1, nb + 1)[1:-1]
    bins = [sum(v > e for e in edges) for v in p]
    counts = np.array([bins.count(b) for b in sorted(set(bins))])
    gaps = np.abs(prob_true - prob_pred)
    return float((counts * gaps).sum() / len(p)) if ns["report"] == "ece" else float(gaps.max())


@brute
def metrics_at_thresholds(ns):
    from sklearn.metrics import f1_score, precision_score, recall_score
    fn = {"precision": precision_score, "recall": recall_score, "f1": f1_score}[ns["report"]]
    return np.array([fn(ns["y"], (ns["p"] >= t).astype(int), zero_division=0) for t in ns["ts"]], dtype=float)
