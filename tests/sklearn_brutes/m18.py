"""4.18 Text features."""

from collections import Counter

import numpy as np

from . import brute


def _docs(ns):
    t, st = ns["tokens"].tolist(), ns["starts"].tolist()
    return [t[a:b] for a, b in zip(st, st[1:])]


@brute
def tfidf_token_stream(ns):
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    V = ns["V"]
    text = [" ".join(f"t{w}" for w in d) for d in _docs(ns)]
    C = CountVectorizer(vocabulary=[f"t{i}" for i in range(V)], token_pattern=r"\S+", lowercase=False).transform(text)
    tt = TfidfTransformer(norm=None).fit(C)
    W = tt.transform(C).toarray()
    if ns["report"] == "idf":
        return tt.idf_
    if ns["report"] == "norms":
        return np.sqrt((W ** 2).sum(axis=1))
    return W.argmax(axis=1)


@brute
def hashed_bucket_collisions(ns):
    from sklearn.utils import murmurhash3_32
    nf, words = ns["n_features"], ns["words"]
    col = [abs(murmurhash3_32(w, seed=0, positive=False)) % nf for w in words]
    if ns["report"] == "pairs":
        return sum(col[i] == col[j] for i in range(len(col)) for j in range(i + 1, len(col)))
    if ns["report"] == "buckets":
        return len(set(col))
    return max(Counter(col).values())


@brute
def df_pruned_vocabulary(ns):
    docs = _docs(ns)
    D, V = len(docs), ns["V"]
    df = [sum(t in set(d) for d in docs) for t in range(V)]
    keep = [t for t in range(V) if df[t] >= ns["min_df"] and df[t] <= ns["max_df"] * D]
    if ns["report"] == "kept":
        return np.array(keep, dtype=int)
    if ns["report"] == "count":
        return len(keep)
    return sum(t not in keep for d in docs for t in d)


@brute
def bigram_vocabulary(ns):
    docs = _docs(ns)
    per_doc = [[(d[i], d[i + 1]) for i in range(len(d) - 1)] for d in docs]
    if ns["report"] == "distinct":
        return len({b for bs in per_doc for b in bs})
    if ns["report"] == "in_two_docs":
        df = Counter(b for bs in per_doc for b in set(bs))
        return sum(c >= 2 for c in df.values())
    cnt = Counter(b for bs in per_doc for b in bs)
    best = max(cnt.values())
    return min(b for b, c in cnt.items() if c == best)


@brute
def multinomial_nb_tokens(ns):
    from sklearn.naive_bayes import MultinomialNB
    V = ns["V"]

    def matrix(tokens, lengths):
        X, pos = np.zeros((len(lengths), V)), 0
        for i, k in enumerate(lengths):
            for w in tokens[pos:pos + k]:
                X[i, w] += 1
            pos += k
        return X

    Xtr = matrix(ns["train_tokens"].tolist(), ns["len_train"].tolist())
    Xte = matrix(ns["test_tokens"].tolist(), ns["len_test"].tolist())
    nb = MultinomialNB(alpha=ns["alpha"]).fit(Xtr, ns["y_train"])
    if ns["report"] == "labels":
        return nb.predict(Xte)
    jll = nb.predict_joint_log_proba(Xte)
    return jll[:, 1] - jll[:, 0]
