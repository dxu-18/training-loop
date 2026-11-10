"""4.13 Classification metrics."""

import numpy as np

from . import brute


def _counts(y, s, t):
    tp = sum(1 for a, b in zip(y, s) if b >= t and a == 1)
    fp = sum(1 for a, b in zip(y, s) if b >= t and a == 0)
    return tp, fp, sum(y) - tp


@brute
def roc_auc_ties(ns):
    from sklearn.metrics import roc_auc_score
    y, s = ns["y"], ns["s"]
    if ns["report"] == "discordant":
        pos, neg = s[y == 1].tolist(), s[y == 0].tolist()
        return sum(1 for a in pos for b in neg if b > a)
    auc = roc_auc_score(y, s)
    return float(auc) if ns["report"] == "auc" else float(2 * auc - 1)


@brute
def best_f1_threshold_scan(ns):
    y, s = ns["y"].tolist(), ns["s"].tolist()
    rows = []
    for t in sorted(set(s)):
        tp, fp, fn = _counts(y, s, t)
        rows.append((t, 2 * tp / (2 * tp + fp + fn), tp, fp))
    if ns["report"] == "f1":
        return max(r[1] for r in rows)
    if ns["report"] == "threshold":
        best = max(r[1] for r in rows)
        return max(r[0] for r in rows if r[1] == best)
    return max(tp / (tp + fp) for t, f, tp, fp in rows if tp >= 0.8 * sum(y))


@brute
def average_precision_ties(ns):
    from sklearn.metrics import average_precision_score, precision_recall_curve, auc
    y, s = ns["y"], ns["s"]
    if ns["report"] == "ap":
        return float(average_precision_score(y, s))
    prec, rec, _ = precision_recall_curve(y, s)
    return float(auc(rec, prec))


@brute
def confusion_macro_metrics(ns):
    from sklearn.metrics import f1_score, matthews_corrcoef, cohen_kappa_score
    a, b = ns["y_true"], ns["y_pred"]
    if ns["report"] == "macro_f1":
        return float(f1_score(a, b, average="macro", zero_division=0))
    return float(matthews_corrcoef(a, b) if ns["report"] == "mcc" else cohen_kappa_score(a, b))


@brute
def min_cost_threshold(ns):
    y, s = ns["y"].tolist(), ns["s"].tolist()
    best = None
    for t in [np.inf] + sorted(set(s), reverse=True):
        tp, fp, fn = _counts(y, s, t)
        cost = ns["c_fp"] * fp + ns["c_fn"] * fn
        if best is None or cost < best[0]:
            best = (cost, t, tp + fp)
    return {"cost": int(best[0]), "threshold": float(best[1]), "flagged": int(best[2])}[ns["report"]]


@brute
def multilabel_scores(ns):
    from sklearn.metrics import hamming_loss, jaccard_score, f1_score, accuracy_score
    T, Pr = ns["Y_true"], ns["Y_pred"]
    return float({"hamming": lambda: hamming_loss(T, Pr),
                  "jaccard_samples": lambda: jaccard_score(T, Pr, average="samples"),
                  "micro_f1": lambda: f1_score(T, Pr, average="micro"),
                  "subset": lambda: accuracy_score(T, Pr)}[ns["report"]]())
