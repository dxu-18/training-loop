"""4.05 Encoding categories & imputing missing values."""

from collections import Counter
from itertools import combinations
import warnings

import numpy as np

from . import brute


@brute
def target_encoding_oof(ns):
    from sklearn.preprocessing import TargetEncoder
    cats, y, fold = ns["cats"], ns["y"], ns["fold"]
    enc = np.empty(len(y))
    for f in np.unique(fold):
        inside = fold == f
        te = TargetEncoder(target_type="continuous", smooth=ns["m"]).fit(cats[~inside].reshape(-1, 1), y[~inside])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            enc[inside] = te.transform(cats[inside].reshape(-1, 1))[:, 0]
    return enc if ns["report"] == "encoded" else float(((enc - y) ** 2).mean())


@brute
def knn_impute_line(ns):
    from sklearn.impute import KNNImputer
    x, v = ns["x"], ns["v"]
    filled = KNNImputer(n_neighbors=ns["k"]).fit_transform(np.column_stack([x, v]))[:, 1]
    imputed = filled[np.isnan(v)]
    return {"imputed": imputed, "filled": filled,
            "mean": float(imputed.mean()) if len(imputed) else 0.0}[ns["report"]]


@brute
def onehot_active_columns(ns):
    from sklearn.preprocessing import OneHotEncoder
    train, test = ns["train"], ns["test"]
    ohe = OneHotEncoder(drop=ns["drop"], handle_unknown="ignore", sparse_output=False).fit(train)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Z = ohe.transform(test)
    widths = [len(c) - (0 if ohe.drop_idx_ is None or ohe.drop_idx_[f] is None else 1)
              for f, c in enumerate(ohe.categories_)]
    out, start = np.full(test.shape, -1), 0
    for f, w in enumerate(widths):
        block = Z[:, start:start + w]
        hit = block.sum(axis=1) > 0
        out[hit, f] = start + block[hit].argmax(axis=1)
        start += w
    return out if ns["report"] == "active" else int((Z.sum(axis=1) == 0).sum())


@brute
def group_mode_impute(ns):
    group, value = ns["group"].tolist(), ns["value"].tolist()
    mode = lambda vals: min(Counter(vals).items(), key=lambda kv: (-kv[1], kv[0]))[0]  # noqa: E731
    overall = mode([v for v in value if v >= 0])
    per = {}
    for g in set(group):
        obs = [v for gg, v in zip(group, value) if gg == g and v >= 0]
        per[g] = mode(obs) if obs else None
    filled, fallback = [], 0
    for g, v in zip(group, value):
        if v >= 0:
            filled.append(v)
        elif per[g] is None:
            filled.append(overall)
            fallback += 1
        else:
            filled.append(per[g])
    if ns["report"] == "filled":
        return np.array(filled)
    if ns["report"] == "fallback":
        return fallback
    fills = [f for f, v in zip(filled, value) if v < 0]
    return np.bincount(np.array(fills, dtype=int), minlength=max(value) + 1)


@brute
def unseen_category_prefix(ns):
    from sklearn.preprocessing import OneHotEncoder
    c, rows, levels = ns["c"].reshape(-1, 1), [], []
    for t in ns["ts"]:
        ohe = OneHotEncoder(handle_unknown="ignore").fit(c[:t])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Z = ohe.transform(c[t:]).toarray()
        rows.append(int((Z.sum(axis=1) == 0).sum()))
        levels.append(len(set(c[t:, 0].tolist()) - set(c[:t, 0].tolist())))
    if ns["report"] == "rows":
        return np.array(rows)
    if ns["report"] == "levels":
        return np.array(levels)
    ok = [t for t, r in zip(ns["ts"], rows) if r <= ns["budget"]]
    return int(ok[0]) if ok else -1


@brute
def tag_cooccurrence(ns):
    posts = {}
    for d, t in zip(ns["doc"].tolist(), ns["tag"].tolist()):
        posts.setdefault(d, []).append(t)
    pairs = Counter(p for tags in posts.values() for p in combinations(sorted(tags), 2))
    if ns["report"] == "top_pair":
        return min(pairs, key=lambda p: (-pairs[p], p)) if pairs else (-1, -1)
    if ns["report"] == "pairs_at_least":
        return sum(c >= ns["t"] for c in pairs.values())
    partners = Counter()
    for a, b in pairs:
        partners[a] += 1
        partners[b] += 1
    T = int(ns["tag"].max()) + 1
    deg = [partners.get(t, 0) for t in range(T)]
    return int(np.argmax(deg))
