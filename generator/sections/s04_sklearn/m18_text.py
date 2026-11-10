"""4.18 Text features: bag of words, TF-IDF, n-grams, hashing, dict features and text pipelines."""

import re
from dataclasses import dataclass

from ._common import P, catalog, challenge, lit, seeded, table, template

M = "4.18"
LR = "LogisticRegression(max_iter=1000)"
TXT = "from sklearn.feature_extraction.text import "
CVI = TXT + "CountVectorizer\n"
TFI = TXT + "TfidfVectorizer\n"


# ================================================================= themed corpora

@dataclass
class _Theme:
    plural: str       # "emails"
    target: str       # what the labels mean
    labels: list
    phrases: dict     # label -> phrases typical of that label
    filler: list      # phrases any document may contain
    fn: str           # name of the document maker in the setup


THEMES = {
    "emails": _Theme(
        "emails", "1 = spam, 0 = a normal email", [0, 1],
        {1: ["win a free cruise", "claim your prize now", "limited time offer", "you have been selected",
             "cheap meds online", "earn money from home", "exclusive deal just for you", "act now before it expires",
             "click here to claim", "free trial with no strings", "your account won a gift card",
             "lowest prices guaranteed"],
         0: ["meeting moved to thursday", "please review the attached report", "lunch tomorrow at noon",
             "notes from the standup", "can you send the slides", "draft of the quarterly budget",
             "the deploy finished without errors", "reminder about the team offsite", "thanks for your help yesterday",
             "agenda for the workshop", "the client approved the design", "updated the shared calendar"]},
        ["hi", "hello there", "best regards", "cheers", "dear friend", "fyi", "quick question", "see below"],
        "message"),
    "reviews": _Theme(
        "product reviews", "1 = positive, 0 = negative", [0, 1],
        {1: ["absolutely love it", "works perfectly", "great value for money", "excellent build quality",
             "would buy again", "highly recommend it", "battery lasts all day", "setup was quick and easy",
             "exceeded my expectations", "five stars from me", "sound quality is superb",
             "arrived early and well packed"],
         0: ["stopped working after a week", "total waste of money", "very poor quality", "arrived broken",
             "would not recommend", "battery dies quickly", "support never replied", "feels cheap and flimsy",
             "returned it the next day", "one star at best", "instructions were useless",
             "much smaller than advertised"]},
        ["the headphones", "this blender", "my new kettle", "the desk lamp", "these running shoes",
         "the coffee maker", "bought it last month", "got it as a gift"],
        "review"),
    "tickets": _Theme(
        "support tickets", "the team that should handle the ticket", ["billing", "bug", "login", "shipping"],
        {"billing": ["charged twice this month", "need a refund for my last invoice", "update the credit card on file",
                     "wrong amount on my bill", "cancel my subscription", "invoice shows the old price",
                     "promo code was not applied", "payment failed but money left my account"],
         "bug": ["app crashes on startup", "error 500 when saving a draft", "page freezes after the upload",
                 "export button does nothing", "csv export produces an empty file", "dark mode makes text unreadable",
                 "search returns duplicate results", "notifications show the wrong time"],
         "login": ["cannot log in to my account", "password reset email never arrives", "two factor code is rejected",
                   "account is locked after failed attempts", "forgot my username", "single sign on loops forever",
                   "session expires every minute", "login page says invalid token"],
         "shipping": ["package has not arrived", "tracking number does not work", "parcel went to the wrong address",
                      "need to change the delivery date", "order stuck in transit", "courier left no delivery note",
                      "box arrived damaged", "shipping label was never created"]},
        ["hello support", "please help", "since yesterday", "this is urgent", "thanks in advance",
         "on the mobile app", "from my laptop", "second time asking"],
        "ticket"),
    "papers": _Theme(
        "paper titles", "the research area", ["nlp", "rl", "vision"],
        {"vision": ["convolutional networks for image segmentation", "object detection with transformers",
                    "self supervised visual representations", "image classification at scale",
                    "depth estimation from a single image", "video action recognition",
                    "generative models for image synthesis", "pixel level semantic labeling"],
         "nlp": ["language models as few shot learners", "neural machine translation with attention",
                 "sentence embeddings for semantic search", "question answering over long documents",
                 "tokenization for multilingual text", "summarization of news articles",
                 "named entity recognition in clinical notes", "instruction tuning of language models"],
         "rl": ["policy gradients with variance reduction", "reward shaping for robot control",
                "offline reinforcement learning", "exploration bonuses for sparse rewards",
                "model based planning in atari games", "multi agent cooperation in games",
                "value functions for continuous control", "learning to walk with deep reinforcement learning"]},
        ["efficient", "towards robust", "a simple baseline for", "revisiting", "scaling", "on the benefits of",
         "rethinking", "a survey of"],
        "title"),
    "commits": _Theme(
        "commit messages", "the kind of change", ["docs", "feat", "fix"],
        {"fix": ["fix null pointer in the parser", "fix off by one error in pagination",
                 "resolve crash when config is missing", "patch memory leak in the cache", "correct timezone handling",
                 "handle empty input gracefully", "fix race condition in the scheduler",
                 "restore broken login redirect"],
         "feat": ["add dark mode toggle", "implement csv export", "add retry logic to the http client",
                  "support oauth login", "introduce batch upload endpoint", "add search filters to the dashboard",
                  "allow custom themes", "add webhook notifications"],
         "docs": ["update readme with install steps", "document the api endpoints", "fix typo in the contributing guide",
                  "add docstrings to utils", "improve changelog wording", "add usage examples to the docs",
                  "explain config options in the wiki", "clarify license section"]},
        ["minor", "wip", "refactor", "small change", "cleanup", "per review", "follow up", "quick"],
        "commit"),
}
BINARY = ["emails", "reviews"]
MULTI = ["tickets", "papers", "commits"]
UNSEEN = ["zeppelin", "quokka", "marmalade", "trombone", "glacier", "origami"]


def _theme(rng, binary=None, only=None):
    keys = [k for k in THEMES if (binary is None or (k in BINARY) == binary) and (only is None or k in only)]
    return THEMES[rng.choice(keys)]


def _make_doc(rng, th, c):
    parts = [rng.choice(th.phrases[c])]
    if rng.random() < 0.6:
        parts.append(rng.choice(th.filler))
    if rng.random() < 0.3:
        parts.append(rng.choice(th.phrases[c]))
    rng.shuffle(parts)
    return " ".join(parts)


def _tokens(docs):
    return [w for d in docs for w in re.findall(r"\b\w\w+\b", d.lower())]


@dataclass
class _Small:
    setup: str
    desc: str
    docs: list
    labels: list
    th: _Theme


def _small(rng, n=(5, 7), labelled=False, binary=None, only=None):
    """A handful of short documents written out literally in the setup (`docs`, maybe `y`)."""
    th = _theme(rng, binary, only)
    k = rng.randint(*n)
    labels = [th.labels[i % len(th.labels)] for i in range(k)]
    rng.shuffle(labels)
    while True:   # distinct documents, and some word shared by two of them (so min_df=2 keeps something)
        docs = []
        for c in labels:
            d = _make_doc(rng, th, c)
            while d in docs:
                d = _make_doc(rng, th, c)
            docs.append(d)
        df = {}
        for d in docs:
            for w in set(_tokens([d])):
                df[w] = df.get(w, 0) + 1
        if max(df.values()) >= 2:
            break
    setup = "docs = [\n" + "".join(f"    {d!r},\n" for d in docs) + "]"
    if labelled:
        setup += f"\ny = np.array({lit(labels)})"
    desc = f"`docs` holds {k} short {th.plural}" + (f" and `y` their labels ({th.target})" if labelled else "")
    return _Small(setup, desc, docs, labels, th)


def _wrap(items, indent=8, width=112):
    lines, cur = [], ""
    for it in items:
        piece = repr(it) + ", "
        if cur and len(cur) + len(piece) + indent > width:
            lines.append(" " * indent + cur.rstrip())
            cur = ""
        cur += piece
    if cur:
        lines.append(" " * indent + cur.rstrip())
    return "\n".join(lines)


@dataclass
class _Corpus:
    setup: str
    desc: str
    th: _Theme
    n: int
    binary: bool


def _corpus(rng, binary=None, n=(150, 260), only=None, noise=0.2):
    """A generated corpus `docs` (list of str) with labels `y`, built in the setup from a seed."""
    th = _theme(rng, binary, only)
    n_, s = rng.randint(*n), seeded(rng)
    body = "".join(f"    {lab!r}: [\n{_wrap(ph)}\n    ],\n" for lab, ph in th.phrases.items())
    setup = (f"g = np.random.default_rng({s})\nphrases = {{\n{body}}}\nfiller = [\n{_wrap(th.filler, 4)}\n]\n"
             f"labels = {lit(th.labels)}\ny = g.choice(labels, {n_})\n\n"
             f"def {th.fn}(c):   # two phrases of class c (sometimes one of another class) and a filler, shuffled\n"
             f"    other = phrases[c] if g.random() > {noise} else phrases[g.choice(labels)]\n"
             f"    parts = [g.choice(phrases[c]), g.choice(other), g.choice(filler)]\n"
             f"    g.shuffle(parts)\n    return ' '.join(parts)\n\n"
             f"docs = [{th.fn}(c) for c in y]")
    desc = f"`docs` holds {n_} {th.plural} (strings) and `y` their labels ({th.target})"
    return _Corpus(setup, desc, th, n_, th.plural in ("emails", "product reviews"))


def _new_docs(rng, th, k=3):
    return [_make_doc(rng, th, rng.choice(th.labels)) for _ in range(k)]


# ---------------------------------------------------------------- records (for DictVectorizer / FeatureHasher)

RECORDS = {
    "rides": ("taxi rides", {"city": ["Lyon", "Oslo", "Porto", "Quito"], "vehicle": ["sedan", "van", "bike"]},
              {"distance_km": (1, 30, 1), "passengers": (1, 4, 0)}),
    "listings": ("apartment listings", {"district": ["center", "docks", "uptown"], "heating": ["gas", "electric"]},
                 {"area_m2": (25, 140, 0), "floor": (0, 12, 0)}),
    "weather": ("weather readings", {"station": ["airport", "harbor", "hilltop"], "sky": ["clear", "cloudy", "rain"]},
                {"temp_c": (-5, 32, 1), "wind_kmh": (0, 60, 0)}),
    "orders": ("online orders", {"country": ["BR", "DE", "FR", "JP"], "payment": ["card", "paypal", "voucher"]},
               {"items": (1, 6, 0), "total_eur": (5, 250, 2)}),
}
TAGS = ["urgent", "gift", "fragile", "express", "returning", "vip", "night", "weekend"]


def _value(rng, lo, hi, nd):
    return rng.randint(lo, hi) if nd == 0 else round(rng.uniform(lo, hi), nd)


def _records(rng, k=(4, 6), drop=0.25):
    """-> (setup, desc, records, categorical keys, numeric keys, cat levels)"""
    name = rng.choice(sorted(RECORDS))
    plural, cats, nums = RECORDS[name]
    recs = []
    for _ in range(rng.randint(*k)):
        r = {c: rng.choice(v) for c, v in cats.items()}
        r.update({c: _value(rng, *v) for c, v in nums.items()})
        if rng.random() < drop:
            del r[rng.choice(sorted(r))]
        recs.append(r)
    setup = "records = [\n" + "".join(f"    {r!r},\n" for r in recs) + "]"
    desc = f"`records` is a list of {len(recs)} {plural} (dicts; some keys may be missing)"
    return setup, desc, recs, list(cats), list(nums), cats


# ================================================================= lite

@template(M, "lite")
def bag_of_words_basics(rng):
    s = _small(rng)
    w = rng.choice(sorted(set(_tokens(s.docs))))
    i = rng.randrange(len(s.docs))
    wi = rng.choice(_tokens([s.docs[i]]))
    new = _make_doc(rng, s.th, rng.choice(s.th.labels)) + " " + rng.choice(UNSEEN)
    items = [
        ("CountVectorizer", "Fit a `CountVectorizer` (from `sklearn.feature_extraction.text`) on `docs` and return the "
                            "shape of the document-term matrix it builds, as a tuple (documents, vocabulary words).",
         CVI + "result = CountVectorizer().fit_transform(docs).shape"),
        ("get_feature_names_out", "Fit `CountVectorizer()` on `docs` and return its vocabulary in column order "
                                  "(`get_feature_names_out()`: alphabetical).",
         CVI + "result = CountVectorizer().fit(docs).get_feature_names_out()"),
        ("vocabulary_", f"Fit `CountVectorizer()` on `docs` and return the column index of the word {w!r} "
                        f"(`vocabulary_` is a dict word → column).",
         CVI + f"result = CountVectorizer().fit(docs).vocabulary_[{w!r}]"),
        ("toarray", "Fit-transform `docs` with `CountVectorizer()` and return the counts as a dense NumPy array (the "
                    "result is a SciPy sparse matrix: call `.toarray()`).",
         CVI + "result = CountVectorizer().fit_transform(docs).toarray()"),
        ("fit_transform", f"How many times does {wi!r} occur in `docs[{i}]`? Fit-transform `docs` with `cv = "
                          f"CountVectorizer()` and read row {i}, column `cv.vocabulary_[{wi!r}]` of the matrix.",
         CVI + f"cv = CountVectorizer()\nX = cv.fit_transform(docs)\nresult = X[{i}, cv.vocabulary_[{wi!r}]]"),
        ("transform", "Fit `CountVectorizer()` on `docs`, then transform `new_doc` (wrap it in a list: `transform` "
                      "wants a list of documents) and return the dense row. Words never seen in `docs` are ignored.",
         CVI + "cv = CountVectorizer().fit(docs)\nresult = cv.transform([new_doc]).toarray()"),
        ("binary", "With `CountVectorizer(binary=True)` every entry is 1 if the word occurs (however often) and 0 "
                   "otherwise. Fit-transform `docs` that way and return the dense matrix.",
         CVI + "result = CountVectorizer(binary=True).fit_transform(docs).toarray()"),
        ("nnz", "The count matrix is sparse: it stores only non-zero entries. Fit-transform `docs` with "
                "`CountVectorizer()` and return how many entries it stores (`.nnz`).",
         CVI + "result = CountVectorizer().fit_transform(docs).nnz"),
        ("sum", "Return the total count of every vocabulary word over all of `docs`, in column order, as a 1-D array "
                "(`X.sum(axis=0)` gives a 1 × V matrix: flatten it with `np.asarray(...).ravel()`).",
         CVI + "X = CountVectorizer().fit_transform(docs)\nresult = np.asarray(X.sum(axis=0)).ravel()",
         {"require": ["CountVectorizer"]}),
        ("stop_words", "Fit `CountVectorizer(stop_words='english')` on `docs` and return its vocabulary "
                       "(`get_feature_names_out()`): common words like 'the' and 'for' are gone.",
         CVI + "result = CountVectorizer(stop_words='english').fit(docs).get_feature_names_out()"),
        ("lowercase", "`CountVectorizer` lowercases by default. Fit it on `shouty = docs + [d.upper() for d in docs]` "
                      "twice, with the default and with `lowercase=False`, and return the two vocabulary sizes as a "
                      "tuple.",
         CVI + "shouty = docs + [d.upper() for d in docs]\n"
               "result = (len(CountVectorizer().fit(shouty).vocabulary_), "
               "len(CountVectorizer(lowercase=False).fit(shouty).vocabulary_))"),
    ]
    return catalog(rng, items, s.setup + f"\nnew_doc = {new!r}", f"{s.desc}; `new_doc` is one more document.")


@template(M, "lite")
def tfidf_basics(rng):
    s = _small(rng)
    i = rng.randrange(len(s.docs))
    new = _make_doc(rng, s.th, rng.choice(s.th.labels))
    items = [
        ("TfidfVectorizer", "Fit-transform `docs` with `TfidfVectorizer()` (from `sklearn.feature_extraction.text`) "
                            "and return the dense tf-idf matrix (`.toarray()`).",
         TFI + "result = TfidfVectorizer().fit_transform(docs).toarray()"),
        ("idf_", "Fit `TfidfVectorizer()` on `docs` and return its learned inverse document frequencies `idf_` (one "
                 "per vocabulary word, in column order: rarer words weigh more).",
         TFI + "result = TfidfVectorizer().fit(docs).idf_"),
        ("idf_", "Fit `TfidfVectorizer()` on `docs` and return the vocabulary word with the **smallest** `idf_` (it "
                 "appears in the most documents; the first in column order on ties).",
         TFI + "tf = TfidfVectorizer().fit(docs)\nresult = tf.get_feature_names_out()[tf.idf_.argmin()]"),
        ("norm", "By default each tf-idf row is scaled to unit length. Fit-transform `docs` with `TfidfVectorizer()` "
                 "and return the Euclidean norm of every row (`np.linalg.norm(..., axis=1)` of the dense matrix).",
         TFI + "result = np.linalg.norm(TfidfVectorizer().fit_transform(docs).toarray(), axis=1)",
         {"require": ["TfidfVectorizer"]}),
        ("norm", "Fit-transform `docs` with `TfidfVectorizer(norm=None)` (raw count × idf, no row scaling) and return "
                 "the dense matrix.", TFI + "result = TfidfVectorizer(norm=None).fit_transform(docs).toarray()"),
        ("use_idf", "Fit-transform `docs` with `TfidfVectorizer(use_idf=False)`: just counts scaled to unit-length "
                    "rows. Return the dense matrix.",
         TFI + "result = TfidfVectorizer(use_idf=False).fit_transform(docs).toarray()"),
        ("sublinear_tf", "Fit-transform `docs` with `TfidfVectorizer(sublinear_tf=True)` (a count c becomes 1 + log c) "
                         "and return the dense matrix.",
         TFI + "result = TfidfVectorizer(sublinear_tf=True).fit_transform(docs).toarray()"),
        ("smooth_idf", "Fit `TfidfVectorizer(smooth_idf=False)` on `docs` and return its `idf_`.",
         TFI + "result = TfidfVectorizer(smooth_idf=False).fit(docs).idf_"),
        ("TfidfTransformer", "Do tf-idf in two steps: count words with `CountVectorizer()`, then reweight the count "
                             "matrix with `TfidfTransformer()`. Return the dense result (it equals `TfidfVectorizer`'s).",
         TXT + "CountVectorizer, TfidfTransformer\ncounts = CountVectorizer().fit_transform(docs)\n"
               "result = TfidfTransformer().fit_transform(counts).toarray()"),
        ("transform", "Fit `TfidfVectorizer()` on `docs` and return the dense tf-idf row of `new_doc` (transform a "
                      "one-element list).",
         TFI + "tf = TfidfVectorizer().fit(docs)\nresult = tf.transform([new_doc]).toarray()"),
        ("get_feature_names_out", f"Which word of `docs[{i}]` gets the largest tf-idf weight? Fit-transform `docs` "
                                  f"with `TfidfVectorizer()` and return that word (first in column order on ties).",
         TFI + f"tf = TfidfVectorizer()\nX = tf.fit_transform(docs)\n"
               f"result = tf.get_feature_names_out()[X[{i}].toarray().argmax()]"),
        ("norm", "With `TfidfVectorizer(norm='l1')` every row's absolute values sum to 1. Fit-transform `docs` that way "
                 "and return the dense matrix.", TFI + "result = TfidfVectorizer(norm='l1').fit_transform(docs).toarray()"),
    ]
    return catalog(rng, items, s.setup + f"\nnew_doc = {new!r}", f"{s.desc}; `new_doc` is one more document.")


@template(M, "lite")
def ngram_options(rng):
    s = _small(rng, n=(5, 8))
    k = rng.randint(4, 8)
    word = rng.choice([w for w in _tokens(s.docs) if 4 <= len(w) <= 8] or ["model"])
    items = [
        ("ngram_range", "Fit `CountVectorizer(ngram_range=(1, 2))` on `docs` (single words **and** pairs of adjacent "
                        "words) and return the vocabulary size.",
         CVI + "result = len(CountVectorizer(ngram_range=(1, 2)).fit(docs).vocabulary_)"),
        ("ngram_range", "Fit `CountVectorizer(ngram_range=(2, 2))` on `docs` and return its features: the word pairs "
                        "(bigrams) it found.",
         CVI + "result = CountVectorizer(ngram_range=(2, 2)).fit(docs).get_feature_names_out()"),
        ("analyzer", "Character n-grams cope with typos. Fit-transform `docs` with `CountVectorizer(analyzer='char_wb', "
                     "ngram_range=(2, 3))` and return the shape of the matrix.",
         CVI + "result = CountVectorizer(analyzer='char_wb', ngram_range=(2, 3)).fit_transform(docs).shape"),
        ("analyzer", f"Fit `CountVectorizer(analyzer='char_wb', ngram_range=(3, 3))` on the one-word corpus "
                     f"`[{word!r}]` and return its features. (`char_wb` pads each word with a space on both sides.)",
         CVI + f"result = CountVectorizer(analyzer='char_wb', ngram_range=(3, 3)).fit([{word!r}]).get_feature_names_out()"),
        ("analyzer", "Fit `CountVectorizer(analyzer='char', ngram_range=(3, 3))` on `docs` (character trigrams, "
                     "spaces included) and return the vocabulary size.",
         CVI + "result = len(CountVectorizer(analyzer='char', ngram_range=(3, 3)).fit(docs).vocabulary_)"),
        ("max_features", f"Keep only the {k} most frequent words: fit `CountVectorizer(max_features={k})` on `docs` and "
                         f"return its features.",
         CVI + f"result = CountVectorizer(max_features={k}).fit(docs).get_feature_names_out()"),
        ("min_df", "Fit `CountVectorizer(min_df=2)` on `docs` (keep words that appear in at least 2 documents) and "
                   "return its features.", CVI + "result = CountVectorizer(min_df=2).fit(docs).get_feature_names_out()"),
        ("max_df", "`max_df=0.5` drops words that appear in more than half of the documents. Return the sorted list of "
                   "words the default `CountVectorizer()` keeps but `CountVectorizer(max_df=0.5)` drops.",
         CVI + "full = set(CountVectorizer().fit(docs).vocabulary_)\n"
               "kept = set(CountVectorizer(max_df=0.5).fit(docs).vocabulary_)\nresult = sorted(full - kept)"),
        ("token_pattern", "The default token pattern `r\"(?u)\\b\\w\\w+\\b\"` skips 1-letter words. Fit "
                          "`CountVectorizer(token_pattern=r\"(?u)\\b\\w+\\b\")` on `docs` and return `(default vocabulary "
                          "size, new vocabulary size)`.",
         CVI + "result = (len(CountVectorizer().fit(docs).vocabulary_), "
               "len(CountVectorizer(token_pattern=r\"(?u)\\b\\w+\\b\").fit(docs).vocabulary_))"),
    ]
    return catalog(rng, items, s.setup, f"{s.desc}.")


@template(M, "lite")
def dict_vectorizer_lite(rng):
    setup, desc, recs, cats, nums, levels = _records(rng)
    c = rng.choice(cats)
    v = rng.choice(levels[c])
    new = {c: "unknown", nums[0]: 1}
    imp = "from sklearn.feature_extraction import DictVectorizer\n"
    items = [
        ("DictVectorizer", "Turn `records` into a feature matrix with `DictVectorizer(sparse=False)` (from "
                           "`sklearn.feature_extraction`) and return it. String values become one-hot columns, "
                           "numbers pass through, missing keys become 0.",
         imp + "result = DictVectorizer(sparse=False).fit_transform(records)"),
        ("get_feature_names_out", "Fit `DictVectorizer()` on `records` and return its feature names (string features "
                                  "are named `key=value`; the names are sorted).",
         imp + "result = DictVectorizer().fit(records).get_feature_names_out()"),
        ("vocabulary_", f"Fit `DictVectorizer()` on `records` and return the column index of the feature "
                        f"`'{c}={v}'` (`vocabulary_`; return -1 if no record has that value).",
         imp + f"result = DictVectorizer().fit(records).vocabulary_.get('{c}={v}', -1)"),
        ("inverse_transform", "Vectorize `records` with `DictVectorizer(sparse=False)`, then map the matrix back with "
                              "`inverse_transform` and return the list of dicts you get (one-hot columns come back as "
                              "`'key=value': 1.0`).",
         imp + "dv = DictVectorizer(sparse=False)\nresult = dv.inverse_transform(dv.fit_transform(records))"),
        ("transform", "Fit `DictVectorizer(sparse=False)` on `records`, then transform `[new_record]`: its "
                      f"`{c}` value was never seen, so no one-hot column lights up. Return the transformed row.",
         imp + "dv = DictVectorizer(sparse=False).fit(records)\nresult = dv.transform([new_record])"),
        ("FeatureHasher", "`FeatureHasher` (from `sklearn.feature_extraction`) hashes `key=value` features into a "
                          "fixed number of columns, without a vocabulary. Return "
                          "`FeatureHasher(n_features=16).transform(records).toarray()`.",
         "from sklearn.feature_extraction import FeatureHasher\n"
         "result = FeatureHasher(n_features=16).transform(records).toarray()"),
    ]
    return catalog(rng, items, setup + f"\nnew_record = {new!r}", f"{desc}; `new_record` is one more record.")


@template(M, "lite")
def hashing_lite(rng):
    s = _small(rng, n=(4, 6))
    tags = [rng.sample(TAGS, rng.randint(1, 3)) for _ in range(rng.randint(3, 5))]
    k = rng.choice([6, 8, 10, 12])
    m = rng.choice([4, 8, 16])
    imp = TXT + "HashingVectorizer\n"
    items = [
        ("HashingVectorizer", f"`HashingVectorizer` maps each word to a column with a hash function: no vocabulary, "
                              f"no fitting. Return the shape of `HashingVectorizer(n_features=2**{k}).transform(docs)`.",
         imp + f"result = HashingVectorizer(n_features=2**{k}).transform(docs).shape"),
        ("HashingVectorizer", "With `norm=None, alternate_sign=False`, a `HashingVectorizer` row holds plain word "
                              "counts. Transform `docs` with `HashingVectorizer(n_features=2**10, norm=None, "
                              "alternate_sign=False)` and return the row sums as a 1-D array (words per document).",
         imp + "H = HashingVectorizer(n_features=2**10, norm=None, alternate_sign=False).transform(docs)\n"
               "result = np.asarray(H.sum(axis=1)).ravel()"),
        ("nnz", f"Transform `docs` with `HashingVectorizer(n_features=2**{k})` and return the number of stored "
                f"non-zeros (`.nnz`).", imp + f"result = HashingVectorizer(n_features=2**{k}).transform(docs).nnz"),
        ("FeatureHasher", f"`tags` holds a list of strings per item. Hash them with `FeatureHasher(n_features={m}, "
                          f"input_type='string')` (from `sklearn.feature_extraction`) and return the dense matrix.",
         f"from sklearn.feature_extraction import FeatureHasher\n"
         f"result = FeatureHasher(n_features={m}, input_type='string').transform(tags).toarray()"),
        ("alternate_sign", f"By default a hasher gives every feature a sign (+1 or -1) so collisions tend to cancel "
                           f"out. Hash `tags` with `FeatureHasher(n_features={m}, input_type='string', "
                           f"alternate_sign=False)` and return the dense matrix: all entries are now counts.",
         f"from sklearn.feature_extraction import FeatureHasher\n"
         f"result = FeatureHasher(n_features={m}, input_type='string', alternate_sign=False).transform(tags).toarray()"),
        ("HashingVectorizer", f"Hashing is stateless, so there is nothing to fit. Return the dense rows of "
                              f"`HashingVectorizer(n_features=2**{k}, norm=None)` for the first two documents.",
         imp + f"result = HashingVectorizer(n_features=2**{k}, norm=None).transform(docs[:2]).toarray()"),
    ]
    return catalog(rng, items, s.setup + f"\ntags = {tags!r}", f"{s.desc}; `tags` lists the tags of a few items.")


# ================================================================= easy

@template(M, "easy")
def count_matrix_questions(rng):
    s = _small(rng, n=(5, 8))
    i = rng.randrange(len(s.docs))
    w = rng.choice(sorted(set(_tokens(s.docs))))
    setup = s.setup + f"\n{CVI}cv = CountVectorizer()\nX = cv.fit_transform(docs)\nnames = cv.get_feature_names_out()"
    items = [
        ("sum", "Return the number of words (tokens the vectorizer counted) in each document, as a 1-D array.",
         "result = np.asarray(X.sum(axis=1)).ravel()"),
        ("vocabulary_", f"In how many documents does {w!r} appear?",
         f"result = int((X[:, cv.vocabulary_[{w!r}]].toarray() > 0).sum())"),
        ("toarray", "Return the document frequency of every word (how many documents contain it), in column order.",
         "result = (X.toarray() > 0).sum(axis=0)"),
        ("sum", "Which word occurs most often over all documents? Return it (first in column order on ties).", "result = names[np.asarray(X.sum(axis=0)).ravel().argmax()]"),
        ("toarray", f"Return the most frequent word of `docs[{i}]` (first in column order on ties).",
         f"result = names[X[{i}].toarray().argmax()]"),
        ("toarray", f"Return the vocabulary words that occur in `docs[{i}]`, in column order.",
         f"result = names[X[{i}].toarray().ravel() > 0]"),
        ("toarray", "Return the sorted list of words that appear in exactly one document.",
         "result = sorted(names[(X.toarray() > 0).sum(axis=0) == 1])"),
        ("nnz", "What fraction of the matrix's entries are non-zero (its density)? Return `nnz / (rows × columns)`.",
         "result = X.nnz / (X.shape[0] * X.shape[1])"),
        ("inverse_transform", f"`cv.inverse_transform` turns rows back into the words they contain. Return the words "
                              f"of `docs[{i}]` that way (an array, alphabetical).",
         f"result = cv.inverse_transform(X[{i}])[0]"),
        ("vocabulary_", f"Return a dict {{word: count}} for the words of `docs[{i}]` (only words that occur).",
         f"row = X[{i}].toarray().ravel()\nresult = {{w: int(row[j]) for w, j in cv.vocabulary_.items() if row[j]}}"),
    ]
    return catalog(rng, items, setup, f"{s.desc}. `cv = CountVectorizer()` was fitted on them: `X` is the sparse count "
                                      f"matrix and `names` its vocabulary in column order.")


@template(M, "easy")
def idf_by_hand(rng):
    s = _small(rng, n=(5, 8))
    task = rng.choice(["smooth", "raw", "matrix", "sublinear", "l1"])
    sol = CVI + "C = CountVectorizer().fit_transform(docs).toarray()\nn = len(docs)\ndf = (C > 0).sum(axis=0)\n"
    if task == "smooth":
        return P(f"{s.desc}. `TfidfVectorizer` uses idf = ln((1 + n) / (1 + df)) + 1, where n is the number of "
                 f"documents and df a word's document frequency. Compute `idf_` **by hand** from a `CountVectorizer` "
                 f"count matrix (column order) and return it (it matches `TfidfVectorizer().fit(docs).idf_`).", s.setup,
                 sol + "result = np.log((1 + n) / (1 + df)) + 1", require=["CountVectorizer"], tol=1e-9)
    if task == "raw":
        return P(f"{s.desc}. With `smooth_idf=False` the formula is idf = ln(n / df) + 1. Compute it **by hand** from a "
                 f"`CountVectorizer` count matrix and return the array (it matches "
                 f"`TfidfVectorizer(smooth_idf=False).fit(docs).idf_`).", s.setup,
                 sol + "result = np.log(n / df) + 1", require=["CountVectorizer"], tol=1e-9)
    if task == "matrix":
        return P(f"{s.desc}. Reproduce `TfidfVectorizer().fit_transform(docs).toarray()` **by hand**: counts × idf "
                 f"(idf = ln((1 + n) / (1 + df)) + 1), then divide every row by its Euclidean norm. Start from a "
                 f"`CountVectorizer` count matrix.", s.setup,
                 sol + "W = C * (np.log((1 + n) / (1 + df)) + 1)\nresult = W / np.linalg.norm(W, axis=1, keepdims=True)",
                 require=["CountVectorizer"], tol=1e-9)
    if task == "sublinear":
        return P(f"{s.desc}. Reproduce `TfidfVectorizer(sublinear_tf=True, norm=None).fit_transform(docs).toarray()` "
                 f"**by hand**: a count c > 0 becomes 1 + ln c (zeros stay 0), times the smoothed idf "
                 f"ln((1 + n) / (1 + df)) + 1. Start from a `CountVectorizer` count matrix.", s.setup,
                 sol + "tf = np.where(C > 0, 1 + np.log(np.maximum(C, 1)), 0)\n"
                       "result = tf * (np.log((1 + n) / (1 + df)) + 1)", require=["CountVectorizer"], tol=1e-9)
    return P(f"{s.desc}. Reproduce `TfidfVectorizer(norm='l1').fit_transform(docs).toarray()` **by hand**: counts × "
             f"smoothed idf (ln((1 + n) / (1 + df)) + 1), then divide every row by its sum. Start from a "
             f"`CountVectorizer` count matrix.", s.setup,
             sol + "W = C * (np.log((1 + n) / (1 + df)) + 1)\nresult = W / W.sum(axis=1, keepdims=True)",
             require=["CountVectorizer"], tol=1e-9)


_SETTINGS = ["", "binary=True", "ngram_range=(1, 2)", "ngram_range=(2, 2)", "min_df=2", "max_df=0.4",
             "stop_words='english'", "max_features=10", "analyzer='char_wb', ngram_range=(2, 2)",
             "token_pattern=r'(?u)\\b\\w+\\b'", "ngram_range=(1, 3), min_df=2", "analyzer='char', ngram_range=(1, 1)"]


@template(M, "easy")
def vocabulary_size_by_setting(rng):
    s = _small(rng, n=(6, 9))
    picks = rng.sample(_SETTINGS, 3)
    calls = [f"CountVectorizer({a})" for a in picks]
    shown = ", ".join(f"`{c}`" for c in calls)
    if rng.random() < 0.5:
        return P(f"{s.desc}. How big is the vocabulary under different settings? Fit each of {shown} on `docs` and "
                 f"return the list of vocabulary sizes, in that order.", s.setup,
                 CVI + "result = [len(cv.fit(docs).vocabulary_) for cv in [" + ", ".join(calls) + "]]",
                 require=["CountVectorizer", "vocabulary_"])
    return P(f"{s.desc}. Fit each of {shown} on `docs` and return a list with the **number of stored non-zero "
             f"entries** (`nnz`) of each document-term matrix, in that order.", s.setup,
             CVI + "result = [cv.fit_transform(docs).nnz for cv in [" + ", ".join(calls) + "]]",
             require=["CountVectorizer", "nnz"])


@template(M, "easy")
def dict_vectorizer_records(rng):
    setup, desc, recs, cats, nums, levels = _records(rng, k=(5, 7), drop=0.35)
    c = rng.choice(cats)
    num = rng.choice(nums)
    i = rng.randrange(len(recs))
    new = {c: rng.choice(levels[c] + ["unknown"]), num: 1}
    imp = "from sklearn.feature_extraction import DictVectorizer\n"
    items = [
        ("get_feature_names_out", "How many columns does `DictVectorizer` produce for `records`? Return the count: one "
                                  "per numeric key plus one per distinct string value.",
         imp + "result = len(DictVectorizer().fit(records).get_feature_names_out())"),
        ("transform", "Fit `DictVectorizer(sparse=False)` on `records` and transform `[new_record]`. Return the row as "
                      "a dict {feature name: value}, keeping only the non-zero values.",
         imp + "dv = DictVectorizer(sparse=False).fit(records)\nrow = dv.transform([new_record])[0]\n"
               "result = {f: v for f, v in zip(dv.get_feature_names_out(), row) if v != 0}"),
        ("sort", "`DictVectorizer(sort=False)` keeps features in the order they are first seen instead of sorting "
                 "them. Fit it on `records` and return the feature names.",
         imp + "result = DictVectorizer(sort=False).fit(records).get_feature_names_out()"),
        ("separator", "Fit `DictVectorizer(separator='_')` on `records` and return the feature names (one-hot names "
                      "become `key_value`).", imp + "result = DictVectorizer(separator='_').fit(records).get_feature_names_out()"),
        ("restrict", "`restrict(mask)` keeps only some features of a fitted `DictVectorizer`. Fit `dv = "
                     "DictVectorizer()` on `records`, keep only the numeric features (names without `=`), and return "
                     "`dv.get_feature_names_out()` afterwards.",
         imp + "dv = DictVectorizer().fit(records)\ndv.restrict([('=' not in f) for f in dv.get_feature_names_out()])\n"
               "result = dv.get_feature_names_out()"),
        ("vocabulary_", f"Count the records per `{c}` value with the one-hot columns: vectorize with "
                        f"`DictVectorizer(sparse=False)` and return a dict {{value: count}} over the `{c}=...` "
                        f"columns (values as strings, counts as ints).",
         imp + "dv = DictVectorizer(sparse=False)\nX = dv.fit_transform(records)\n"
               f"result = {{f.split('=')[1]: int(X[:, j].sum()) for f, j in dv.vocabulary_.items() "
               f"if f.startswith('{c}=')}}"),
        ("fit_transform", f"Records missing `{num}` get 0 there. Vectorize `records` with `DictVectorizer(sparse=False)` "
                          f"and return the `{num}` column.",
         imp + f"dv = DictVectorizer(sparse=False)\nX = dv.fit_transform(records)\nresult = X[:, dv.vocabulary_['{num}']]",
         {"setup": f"records[{i}].pop('{num}', None)"}),
    ]
    return catalog(rng, items, setup + f"\nnew_record = {new!r}", f"{desc}; `new_record` is one more record.")


@template(M, "easy")
def hashing_properties(rng):
    s = _small(rng, n=(5, 7))
    more = _new_docs(rng, s.th, 2)
    k = rng.choice([8, 9, 10])
    m = rng.choice([4, 8, 16])
    task = rng.choice(["stateless", "collide", "pairs", "negative", "vs_count"])
    imp = TXT + "HashingVectorizer\n"
    setup = s.setup + f"\nmore = {more!r}"
    head = f"{s.desc}; `more` holds 2 newer documents."
    if task == "stateless":
        return P(f"{head} A `HashingVectorizer` needs no fitting, so documents arriving later get the same columns. "
                 f"Transform `docs + more` in one go with `HashingVectorizer(n_features=2**{k})` and return the dense "
                 f"rows of the last two documents.", setup,
                 imp + f"result = HashingVectorizer(n_features=2**{k}).transform(docs + more)[-2:].toarray()",
                 require=["HashingVectorizer"])
    if task == "collide":
        return P(f"{head} With few columns, different words collide. Fit a `CountVectorizer` on `docs` to get its "
                 f"vocabulary, hash all those words as one document with `HashingVectorizer(n_features={m}, "
                 f"alternate_sign=False)`, and return `(vocabulary size, number of columns used)` (columns used = "
                 f"`nnz` of that one row).", setup,
                 TXT + f"CountVectorizer, HashingVectorizer\nvocab = CountVectorizer().fit(docs).get_feature_names_out()\n"
                       f"row = HashingVectorizer(n_features={m}, alternate_sign=False).transform([' '.join(vocab)])\n"
                       "result = (len(vocab), row.nnz)", require=["HashingVectorizer", "alternate_sign"])
    if task == "pairs":
        recs = [[(t, round(rng.uniform(0.5, 3), 1)) for t in rng.sample(TAGS, rng.randint(1, 3))] for _ in range(3)]
        return P(f"`weighted` holds (feature, value) pairs for 3 items. Hash them with `FeatureHasher(n_features={m}, "
                 f"input_type='pair', alternate_sign=False)` and return the dense matrix (each value lands in its "
                 f"feature's column).", f"weighted = {recs!r}",
                 f"from sklearn.feature_extraction import FeatureHasher\n"
                 f"result = FeatureHasher(n_features={m}, input_type='pair', alternate_sign=False).transform(weighted)"
                 f".toarray()", require=["FeatureHasher", "input_type"])
    if task == "negative":
        return P(f"{head} Transform `docs` with `HashingVectorizer(n_features=2**{k}, norm=None)` (signs alternate by "
                 f"default) and return `(number of positive entries, number of negative entries)` of the matrix.",
                 setup, imp + f"H = HashingVectorizer(n_features=2**{k}, norm=None).transform(docs).toarray()\n"
                              "result = (int((H > 0).sum()), int((H < 0).sum()))", require=["HashingVectorizer"])
    return P(f"{head} Check that hashing preserves counts: transform `docs` with `HashingVectorizer(n_features=2**20, "
             f"norm=None, alternate_sign=False)` and with `CountVectorizer()`, and return the two arrays of per-row "
             f"sums as a tuple (they agree unless two words of a document collide).", setup,
             TXT + "CountVectorizer, HashingVectorizer\n"
                   "H = HashingVectorizer(n_features=2**20, norm=None, alternate_sign=False).transform(docs)\n"
                   "C = CountVectorizer().fit_transform(docs)\n"
                   "result = (np.asarray(H.sum(axis=1)).ravel(), np.asarray(C.sum(axis=1)).ravel())",
             require=["HashingVectorizer", "CountVectorizer"])


@template(M, "easy")
def first_text_classifier(rng):
    s = _small(rng, n=(7, 10), labelled=True)
    new = _new_docs(rng, s.th, rng.randint(2, 3))
    imp = (TFI + "from sklearn.linear_model import LogisticRegression\nfrom sklearn.pipeline import make_pipeline\n")
    fit = imp + "clf = make_pipeline(TfidfVectorizer(), LogisticRegression()).fit(docs, y)\n"
    items = [
        ("make_pipeline", "Chain `TfidfVectorizer()` and `LogisticRegression()` with `make_pipeline`, fit it on the raw "
                          "strings `docs, y`, and return its predictions for `new_docs`.",
         fit + "result = clf.predict(new_docs)", {"require": ["make_pipeline", "TfidfVectorizer"]}),
        ("predict_proba", "Fit `make_pipeline(TfidfVectorizer(), LogisticRegression())` on `docs, y` and return the "
                          "class probabilities of `new_docs` (columns in `classes_` order).",
         fit + "result = clf.predict_proba(new_docs)", {"tol": 1e-5}),
        ("score", "Fit `make_pipeline(TfidfVectorizer(), LogisticRegression())` on `docs, y` and return its training "
                  "accuracy.", fit + "result = clf.score(docs, y)"),
        ("named_steps", "Fit `make_pipeline(TfidfVectorizer(), LogisticRegression())` on `docs, y`, reach the fitted "
                        "vectorizer through `named_steps['tfidfvectorizer']`, and return its vocabulary size.",
         fit + "result = len(clf.named_steps['tfidfvectorizer'].vocabulary_)"),
        ("MultinomialNB", "Naive Bayes on word counts is a classic text baseline. Fit `make_pipeline(CountVectorizer(), "
                          "MultinomialNB())` (from `sklearn.naive_bayes`) on `docs, y` and return its predictions for "
                          "`new_docs`.",
         CVI + "from sklearn.naive_bayes import MultinomialNB\nfrom sklearn.pipeline import make_pipeline\n"
               "clf = make_pipeline(CountVectorizer(), MultinomialNB()).fit(docs, y)\nresult = clf.predict(new_docs)"),
        ("coef_", "Fit `make_pipeline(TfidfVectorizer(), LogisticRegression())` on `docs, y` and return the shape of the "
                  "classifier's `coef_` (the last step: `clf[-1]`).", fit + "result = clf[-1].coef_.shape"),
        ("classes_", "Fit `make_pipeline(TfidfVectorizer(), LogisticRegression())` on `docs, y` and return `classes_` "
                     "as a list.", fit + "result = clf.classes_.tolist()"),
    ]
    return catalog(rng, items, s.setup + f"\nnew_docs = {new!r}", f"{s.desc}; `new_docs` holds {len(new)} unlabelled "
                                                                   f"{s.th.plural}.")


@template(M, "easy")
def analyzer_tokens(rng):
    s = _small(rng, n=(3, 5))
    i = rng.randrange(len(s.docs))
    word = rng.choice([w for w in _tokens(s.docs) if 4 <= len(w) <= 7] or ["text"])
    items = [
        ("build_analyzer", f"`build_analyzer()` returns the function a vectorizer uses to split a document into "
                           f"features. Return `CountVectorizer().build_analyzer()(docs[{i}])` (a list of tokens).",
         CVI + f"result = CountVectorizer().build_analyzer()(docs[{i}])"),
        ("build_analyzer", f"Return the features `CountVectorizer(stop_words='english')` extracts from `docs[{i}]` "
                           f"(use `build_analyzer()`).",
         CVI + f"result = CountVectorizer(stop_words='english').build_analyzer()(docs[{i}])"),
        ("build_analyzer", f"Return the features `CountVectorizer(ngram_range=(1, 2))` extracts from `docs[{i}]`: "
                           f"the words first, then the bigrams (use `build_analyzer()`).",
         CVI + f"result = CountVectorizer(ngram_range=(1, 2)).build_analyzer()(docs[{i}])"),
        ("build_analyzer", f"Return the features `CountVectorizer(analyzer='char_wb', ngram_range=(2, 2))` extracts "
                           f"from {word!r} (use `build_analyzer()`).",
         CVI + f"result = CountVectorizer(analyzer='char_wb', ngram_range=(2, 2)).build_analyzer()({word!r})"),
        ("build_tokenizer", f"`build_tokenizer()` only splits (no lowercasing). Return "
                            f"`CountVectorizer().build_tokenizer()(docs[{i}].title())`.",
         CVI + f"result = CountVectorizer().build_tokenizer()(docs[{i}].title())"),
        ("build_preprocessor", f"`build_preprocessor()` is the step before tokenizing (lowercasing). Return what it "
                               f"makes of `docs[{i}].upper()`.",
         CVI + f"result = CountVectorizer().build_preprocessor()(docs[{i}].upper())"),
        ("build_analyzer", "Return the number of tokens the default `CountVectorizer` analyzer finds in each document, "
                           "as a list.",
         CVI + "analyze = CountVectorizer().build_analyzer()\nresult = [len(analyze(d)) for d in docs]"),
    ]
    return catalog(rng, items, s.setup, f"{s.desc}.")


# ================================================================= medium

def _split(c: _Corpus, frac=0.75):
    k = int(c.n * frac)
    return k, f"\ndocs_train, docs_test = docs[:{k}], docs[{k}:]\ny_train, y_test = y[:{k}], y[{k}:]"


@template(M, "medium")
def spam_filter_pipeline(rng):
    c = _corpus(rng, binary=True)
    k, split = _split(c)
    new = _new_docs(rng, c.th, 4)
    imp = (TFI + "from sklearn.linear_model import LogisticRegression\nfrom sklearn.pipeline import make_pipeline\n")
    pipe = "make_pipeline(TfidfVectorizer(), LogisticRegression())"
    items = [
        ("make_pipeline", f"Train `{pipe}` on the training split and return its accuracy on the test split.",
         imp + f"clf = {pipe}.fit(docs_train, y_train)\nresult = clf.score(docs_test, y_test)",
         {"require": ["make_pipeline", "TfidfVectorizer"]}),
        ("cross_val_score", f"Return the mean 5-fold `cross_val_score` of `{pipe}` on all of `docs, y` (the vectorizer "
                            f"is refitted inside every fold).",
         imp + f"from sklearn.model_selection import cross_val_score\nresult = cross_val_score({pipe}, docs, y, cv=5).mean()"),
        ("MultinomialNB", f"Compare two classic text classifiers on the split: `{pipe}` and `make_pipeline("
                          f"CountVectorizer(), MultinomialNB())`. Return their test accuracies as a tuple (in that "
                          f"order).",
         imp + "from sklearn.feature_extraction.text import CountVectorizer\nfrom sklearn.naive_bayes import MultinomialNB\n"
               f"a = {pipe}.fit(docs_train, y_train).score(docs_test, y_test)\n"
               "b = make_pipeline(CountVectorizer(), MultinomialNB()).fit(docs_train, y_train).score(docs_test, y_test)\n"
               "result = (a, b)"),
        ("predict", f"Train `{pipe}` on all of `docs, y` and return its predictions for `new_docs`.",
         imp + f"result = {pipe}.fit(docs, y).predict(new_docs)"),
        ("predict_proba", f"Train `{pipe}` on all of `docs, y` and return P(class 1) for each of `new_docs`.",
         imp + f"result = {pipe}.fit(docs, y).predict_proba(new_docs)[:, 1]", {"tol": 1e-5}),
        ("confusion_matrix", f"Train `{pipe}` on the training split and return the `confusion_matrix` (from "
                             f"`sklearn.metrics`) of its test predictions.",
         imp + f"from sklearn.metrics import confusion_matrix\nclf = {pipe}.fit(docs_train, y_train)\n"
               "result = confusion_matrix(y_test, clf.predict(docs_test))"),
        ("BernoulliNB", "Presence/absence features suit `BernoulliNB`. Train `make_pipeline(CountVectorizer(binary=True), "
                        "BernoulliNB())` (from `sklearn.naive_bayes`) on the training split and return its test "
                        "accuracy.",
         CVI + "from sklearn.naive_bayes import BernoulliNB\nfrom sklearn.pipeline import make_pipeline\n"
               "clf = make_pipeline(CountVectorizer(binary=True), BernoulliNB()).fit(docs_train, y_train)\n"
               "result = clf.score(docs_test, y_test)"),
        ("predict", f"Train `{pipe}` on the training split. How many test documents does it get wrong? Return the "
                    f"count.", imp + f"clf = {pipe}.fit(docs_train, y_train)\n"
                                     "result = int((clf.predict(docs_test) != y_test).sum())"),
    ]
    return catalog(rng, items, c.setup + split + f"\nnew_docs = {new!r}",
                   f"{c.desc}. The first {k} documents form the training split (`docs_train, y_train`), the rest the "
                   f"test split; `new_docs` holds 4 unlabelled ones.", tol=1e-6)


@template(M, "medium")
def binary_keywords(rng):
    c = _corpus(rng, binary=True)
    k = rng.randint(3, 6)
    w = rng.choice(_tokens([rng.choice(c.th.phrases[1]), rng.choice(c.th.phrases[0])]))
    setup = (c.setup + f"\n{TFI}from sklearn.linear_model import LogisticRegression\n"
             "tf = TfidfVectorizer()\nX = tf.fit_transform(docs)\nclf = LogisticRegression().fit(X, y)\n"
             "names = tf.get_feature_names_out()")
    items = [
        ("coef_", f"Return the {k} words with the **largest** coefficients (the strongest evidence for class 1), "
                  f"largest first.", f"result = names[np.argsort(clf.coef_[0])[::-1][:{k}]]"),
        ("coef_", f"Return the {k} words with the **most negative** coefficients (the strongest evidence for class 0), "
                  f"most negative first.", f"result = names[np.argsort(clf.coef_[0])[:{k}]]"),
        ("vocabulary_", f"Return the coefficient the model learned for the word {w!r}.",
         f"result = clf.coef_[0, tf.vocabulary_[{w!r}]]"),
        ("coef_", "How many words push towards class 1 (positive coefficient)? Return the count.",
         "result = int((clf.coef_[0] > 0).sum())"),
        ("coef_", f"Return a dict {{word: coefficient}} for the {k} words with the largest **absolute** coefficient.",
         f"top = np.argsort(-np.abs(clf.coef_[0]))[:{k}]\nresult = {{names[j]: clf.coef_[0, j] for j in top}}"),
        ("feature_log_prob_", "Fit `MultinomialNB()` (from `sklearn.naive_bayes`) on `X, y` too. Its `feature_log_prob_` "
                              "holds log P(word | class). Return the word with the largest log-ratio "
                              "`feature_log_prob_[1] - feature_log_prob_[0]`.",
         "from sklearn.naive_bayes import MultinomialNB\nnb = MultinomialNB().fit(X, y)\n"
         "result = names[np.argmax(nb.feature_log_prob_[1] - nb.feature_log_prob_[0])]"),
    ]
    return catalog(rng, items, setup, f"{c.desc}. `tf = TfidfVectorizer()` turned them into `X`, `names` is the "
                                      f"vocabulary in column order, and `clf = LogisticRegression()` was fitted on "
                                      f"`X, y`.", tol=1e-5)


@template(M, "medium")
def class_keywords(rng):
    c = _corpus(rng, binary=False)
    k = rng.randint(2, 4)
    lab = rng.choice(c.th.labels)
    w = rng.choice(_tokens([rng.choice(c.th.phrases[lab])]))
    setup = (c.setup + f"\n{TFI}from sklearn.linear_model import LogisticRegression\n"
             f"tf = TfidfVectorizer()\nX = tf.fit_transform(docs)\nclf = {LR}.fit(X, y)\n"
             "names = tf.get_feature_names_out()")
    items = [
        ("coef_", "A multiclass linear model has one row of coefficients per class (`classes_` order). Return a dict "
                  "{class: its highest-weighted word}.",
         "result = {cl: names[row.argmax()] for cl, row in zip(clf.classes_, clf.coef_)}"),
        ("coef_", f"Return the {k} highest-weighted words for class {lab!r}, highest first.",
         f"row = clf.coef_[list(clf.classes_).index({lab!r})]\nresult = names[np.argsort(row)[::-1][:{k}]]"),
        ("classes_", f"Which class does the word {w!r} push hardest towards? Return the class whose coefficient for "
                     f"that word is largest.", f"result = clf.classes_[clf.coef_[:, tf.vocabulary_[{w!r}]].argmax()]"),
        ("feature_log_prob_", "Fit `MultinomialNB()` (from `sklearn.naive_bayes`) on `X, y` and return a dict {class: "
                              "the word with the largest `feature_log_prob_`} (the most probable word of each class).",
         "from sklearn.naive_bayes import MultinomialNB\nnb = MultinomialNB().fit(X, y)\n"
         "result = {cl: names[row.argmax()] for cl, row in zip(nb.classes_, nb.feature_log_prob_)}"),
        ("predict", "Return the training accuracy of `clf` per class as a dict {class: accuracy}.",
         "pred = clf.predict(X)\nresult = {cl: float((pred[y == cl] == cl).mean()) for cl in clf.classes_}"),
        ("transform", f"Classify each phrase of class {lab!r} on its own: transform `phrases[{lab!r}]` with `tf` and "
                      f"return the fraction `clf` predicts as {lab!r}.",
         f"result = float((clf.predict(tf.transform(phrases[{lab!r}])) == {lab!r}).mean())"),
    ]
    return catalog(rng, items, setup, f"{c.desc}. `tf = TfidfVectorizer()` turned them into `X`, `names` is the "
                                      f"vocabulary in column order, and `clf = {LR}` was fitted on `X, y`.", tol=1e-5)


@template(M, "medium")
def word_vs_char_features(rng):
    c = _corpus(rng, n=(140, 220))
    k, split = _split(c)
    lo, hi = rng.choice([(2, 4), (3, 5), (2, 3)])
    s = seeded(rng)
    typos = (f"\nt = np.random.default_rng({s})\n\ndef typo(doc):   # swap two neighbouring letters, twice\n"
             "    ch = list(doc)\n    for _ in range(2):\n        i = t.integers(len(ch) - 1)\n"
             "        ch[i], ch[i + 1] = ch[i + 1], ch[i]\n    return ''.join(ch)\n\n"
             "docs_typo = [typo(d) for d in docs_test]")
    imp = (TFI + "from sklearn.linear_model import LogisticRegression\nfrom sklearn.pipeline import make_pipeline\n")
    word = f"make_pipeline(TfidfVectorizer(), {LR})"
    char = f"make_pipeline(TfidfVectorizer(analyzer='char_wb', ngram_range=({lo}, {hi})), {LR})"
    items = [
        ("cross_val_score", f"Compare word and character features with 5-fold cross-validation on all of `docs, y`: "
                            f"`{word}` versus `{char}`. Return the two mean scores as a tuple (word first).",
         imp + "from sklearn.model_selection import cross_val_score\n"
               f"result = (cross_val_score({word}, docs, y, cv=5).mean(), cross_val_score({char}, docs, y, cv=5).mean())",
         {"require": ["cross_val_score", "analyzer"]}),
        ("analyzer", f"How many features does each representation create? Fit `TfidfVectorizer()` and "
                     f"`TfidfVectorizer(analyzer='char_wb', ngram_range=({lo}, {hi}))` on `docs_train` and return the "
                     f"two vocabulary sizes as a tuple (word first).",
         TFI + f"result = (len(TfidfVectorizer().fit(docs_train).vocabulary_), len(TfidfVectorizer(analyzer='char_wb', "
               f"ngram_range=({lo}, {hi})).fit(docs_train).vocabulary_))"),
        ("analyzer", f"Typos break word features but barely touch character n-grams. Train `{word}` and `{char}` on "
                     f"the training split and return their accuracies on `docs_typo` (the test documents with typos, "
                     f"labels `y_test`) as a tuple (word first).",
         imp + f"a = {word}.fit(docs_train, y_train).score(docs_typo, y_test)\n"
               f"b = {char}.fit(docs_train, y_train).score(docs_typo, y_test)\nresult = (a, b)", {"setup": typos}),
        ("transform", "How many words of the typo'd test documents does a word vectorizer still recognise? Fit "
                      "`TfidfVectorizer()` on `docs_train` and return the fraction of `docs_typo` rows that are "
                      "**entirely zero** (no known word left) — and the same fraction for the clean `docs_test`, as a "
                      "tuple (typos first).",
         TFI + "tf = TfidfVectorizer().fit(docs_train)\nzero = lambda D: float((tf.transform(D).getnnz(axis=1) == 0).mean())\n"
               "result = (zero(docs_typo), zero(docs_test))", {"setup": typos}),
        ("ngram_range", f"Do word pairs help? Return the mean 5-fold `cross_val_score` on `docs, y` of `{word}` and of "
                        f"the same pipeline with `TfidfVectorizer(ngram_range=(1, 2))`, as a tuple (unigrams first).",
         imp + "from sklearn.model_selection import cross_val_score\n"
               f"result = (cross_val_score({word}, docs, y, cv=5).mean(), cross_val_score(make_pipeline("
               f"TfidfVectorizer(ngram_range=(1, 2)), {LR}), docs, y, cv=5).mean())"),
        ("analyzer", f"Train `{char}` on the training split and return its test accuracy (`docs_test, y_test`).",
         imp + f"result = {char}.fit(docs_train, y_train).score(docs_test, y_test)"),
    ]
    return catalog(rng, items, c.setup + split, f"{c.desc}. The first {k} documents are the training split "
                                                f"(`docs_train, y_train`), the rest the test split.", tol=1e-6)


@template(M, "medium")
def similar_documents(rng):
    c = _corpus(rng, n=(80, 160))
    q = _make_doc(rng, c.th, rng.choice(c.th.labels))
    a, b = rng.sample(range(20), 2)
    k = rng.randint(3, 5)
    thr = rng.choice([0.6, 0.7, 0.8])
    setup = c.setup + f"\nquery = {q!r}\n{TFI}tf = TfidfVectorizer()\nX = tf.fit_transform(docs)"
    items = [
        ("cosine_similarity", "Tf-idf rows have unit length, so cosine similarity is a dot product. Return the index of "
                              "the document most similar to `query` (use `cosine_similarity` from "
                              "`sklearn.metrics.pairwise`, transforming `query` with `tf`).",
         "from sklearn.metrics.pairwise import cosine_similarity\n"
         "result = int(cosine_similarity(tf.transform([query]), X).argmax())"),
        ("linear_kernel", f"Return the indices of the {k} documents most similar to `query`, most similar first "
                          f"(`linear_kernel` from `sklearn.metrics.pairwise` equals cosine similarity here; on ties the "
                          f"smaller index first: sort with `np.argsort(-sims, kind='stable')`).",
         "from sklearn.metrics.pairwise import linear_kernel\nsims = linear_kernel(tf.transform([query]), X).ravel()\n"
         f"result = np.argsort(-sims, kind='stable')[:{k}]"),
        ("cosine_similarity", f"Return the cosine similarity between `docs[{a}]` and `docs[{b}]`.",
         f"from sklearn.metrics.pairwise import cosine_similarity\nresult = float(cosine_similarity(X[{a}], X[{b}])[0, 0])"),
        ("cosine_similarity", f"Near-duplicates: how many pairs of documents (i < j) have cosine similarity above {thr}? "
                              f"Return the count.",
         "from sklearn.metrics.pairwise import cosine_similarity\nS = cosine_similarity(X)\n"
         f"result = int((np.triu(S, k=1) > {thr}).sum())"),
        ("cosine_similarity", "Are documents closer to their own class? Return `(mean similarity of pairs with the same "
                              "label, mean similarity of pairs with different labels)` over all pairs i < j.",
         "from sklearn.metrics.pairwise import cosine_similarity\nS = cosine_similarity(X)\n"
         "iu = np.triu_indices(len(docs), k=1)\nsame = (y[:, None] == y[None, :])[iu]\n"
         "result = (float(S[iu][same].mean()), float(S[iu][~same].mean()))"),
        ("cosine_similarity", "Return a dict {label: mean cosine similarity between `query` and the documents with "
                              "that label} (labels as in `y`).",
         "from sklearn.metrics.pairwise import cosine_similarity\nsims = cosine_similarity(tf.transform([query]), X).ravel()\n"
         "result = {lab: float(sims[y == lab].mean()) for lab in np.unique(y)}"),
    ]
    return catalog(rng, items, setup, f"{c.desc}. `X` holds their tf-idf vectors (`tf = TfidfVectorizer()` fitted "
                                      f"on `docs`); `query` is a new document.", tol=1e-6)


@template(M, "medium")
def vocabulary_pruning(rng):
    c = _corpus(rng, n=(120, 240))
    mn = rng.choice([2, 3, 5, 10])
    frac = rng.choice([0.02, 0.05, 0.1])
    mx = rng.choice([0.1, 0.2, 0.3])
    k = rng.choice([10, 15, 20])
    items = [
        ("min_df", f"Rare words are mostly noise. Return the vocabulary sizes of `CountVectorizer()` and "
                   f"`CountVectorizer(min_df={mn})` on `docs`, as a tuple.",
         CVI + f"result = (len(CountVectorizer().fit(docs).vocabulary_), "
               f"len(CountVectorizer(min_df={mn}).fit(docs).vocabulary_))"),
        ("min_df", f"A float `min_df` is a fraction of the documents. Return the words `CountVectorizer(min_df={frac})` "
                   f"keeps (alphabetical).", CVI + f"result = CountVectorizer(min_df={frac}).fit(docs).get_feature_names_out()"),
        ("max_df", f"Very common words carry little signal. Return the sorted list of words that "
                   f"`CountVectorizer(max_df={mx})` removes (in the default vocabulary but not in the pruned one).",
         CVI + f"full = set(CountVectorizer().fit(docs).vocabulary_)\n"
               f"result = sorted(full - set(CountVectorizer(max_df={mx}).fit(docs).vocabulary_))"),
        ("max_features", f"Fit `CountVectorizer(max_features={k})` on `docs` and return the total number of times its "
                         f"{k} words occur in the corpus (the sum of the whole count matrix).",
         CVI + f"result = int(CountVectorizer(max_features={k}).fit_transform(docs).sum())"),
        ("min_df", f"Combine the filters: `CountVectorizer(min_df={mn}, max_df={mx}, stop_words='english')`. Return the "
                   f"resulting vocabulary size.",
         CVI + f"result = len(CountVectorizer(min_df={mn}, max_df={mx}, stop_words='english').fit(docs).vocabulary_)"),
        ("toarray", f"Return the document frequency of each word that appears in more than {int(mx * 100)}% of the "
                    f"documents, as a dict {{word: number of documents}}.",
         CVI + "cv = CountVectorizer()\nX = cv.fit_transform(docs)\ndf = (X.toarray() > 0).sum(axis=0)\n"
               f"result = {{w: int(df[j]) for w, j in cv.vocabulary_.items() if df[j] > {mx} * len(docs)}}"),
        ("stop_words", "What fraction of all word occurrences in the corpus are English stop words? Compare the totals "
                       "of `CountVectorizer()` and `CountVectorizer(stop_words='english')` count matrices: return "
                       "`1 - pruned_total / full_total`.",
         CVI + "full = CountVectorizer().fit_transform(docs).sum()\n"
               "pruned = CountVectorizer(stop_words='english').fit_transform(docs).sum()\nresult = float(1 - pruned / full)"),
    ]
    return catalog(rng, items, c.setup, f"{c.desc}.", tol=1e-9)


@template(M, "medium")
def records_to_model(rng):
    t = table(rng, n=(80, 200))
    col = rng.choice(t.cat)
    num = rng.choice(t.num)
    model = "LogisticRegression(max_iter=1000)" if t.kind == "clf" else "Ridge()"
    mmod = "from sklearn.linear_model import " + ("LogisticRegression" if t.kind == "clf" else "Ridge") + "\n"
    imp = "from sklearn.feature_extraction import DictVectorizer\n"
    pipe = f"make_pipeline(DictVectorizer(sparse=False), StandardScaler(), {model})"
    pimp = imp + mmod + "from sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\n"
    setup = t.setup + "\nrecords = df.to_dict('records')"
    items = [
        ("DictVectorizer", "Vectorize `records` with `DictVectorizer(sparse=False)` and return the shape of the result.",
         imp + "result = DictVectorizer(sparse=False).fit_transform(records).shape"),
        ("get_feature_names_out", f"Return the names of the one-hot columns `DictVectorizer` creates for `{col}`.",
         imp + f"names = DictVectorizer().fit(records).get_feature_names_out()\n"
               f"result = [f for f in names if f.startswith('{col}=')]"),
        ("make_pipeline", f"`DictVectorizer` works inside a pipeline. Fit `{pipe}` on `records, y` and return its "
                          f"training score.", pimp + f"result = {pipe}.fit(records, y).score(records, y)",
         {"require": ["make_pipeline", "DictVectorizer"]}),
        ("cross_val_score", f"Return the mean 5-fold `cross_val_score` of `{pipe}` on `records, y`.",
         pimp + f"from sklearn.model_selection import cross_val_score\nresult = cross_val_score({pipe}, records, y, cv=5).mean()"),
        ("transform", f"Fit `DictVectorizer(sparse=False)` on `records` and transform a record that only has "
                      f"`{num}` = 1 (all other features missing). Return the transformed row.",
         imp + f"dv = DictVectorizer(sparse=False).fit(records)\nresult = dv.transform([{{'{num}': 1}}])[0]"),
        ("vocabulary_", f"`DictVectorizer` builds the same columns as one-hot encoding `{col}` plus the numeric columns. "
                        f"Return the column of `DictVectorizer(sparse=False).fit_transform(records)` that belongs to "
                        f"`{num}` (it equals `df[{num!r}]`).",
         imp + f"dv = DictVectorizer(sparse=False)\nX = dv.fit_transform(records)\nresult = X[:, dv.vocabulary_['{num}']]"),
    ]
    return catalog(rng, items, setup, f"{t.desc}; `records = df.to_dict('records')` holds the same rows as dicts.",
                   tol=1e-6)


@template(M, "medium")
def hashing_vs_vocabulary(rng):
    c = _corpus(rng, n=(150, 240))
    k, split = _split(c)
    e = rng.choice([8, 10, 12])
    sizes = sorted(rng.sample([2, 4, 8, 16, 64, 256, 1024], 3))
    imp = (TXT + "HashingVectorizer\nfrom sklearn.linear_model import LogisticRegression\n"
           "from sklearn.pipeline import make_pipeline\n")
    items = [
        ("HashingVectorizer", f"Train `make_pipeline(HashingVectorizer(n_features=2**{e}), {LR})` on the training split "
                              f"and return its test accuracy.",
         imp + f"clf = make_pipeline(HashingVectorizer(n_features=2**{e}), {LR}).fit(docs_train, y_train)\n"
               "result = clf.score(docs_test, y_test)"),
        ("TfidfTransformer", f"Hashing gives no idf weighting by itself: add it with `TfidfTransformer`. Train "
                             f"`make_pipeline(HashingVectorizer(n_features=2**{e}, alternate_sign=False, norm=None), "
                             f"TfidfTransformer(), {LR})` on the training split and return its test accuracy.",
         imp + "from sklearn.feature_extraction.text import TfidfTransformer\n"
               f"clf = make_pipeline(HashingVectorizer(n_features=2**{e}, alternate_sign=False, norm=None), "
               f"TfidfTransformer(), {LR}).fit(docs_train, y_train)\nresult = clf.score(docs_test, y_test)"),
        ("n_features", f"Too few hash columns hurt. For `n_features` in `{sizes}`, train `make_pipeline("
                       f"HashingVectorizer(n_features=n), {LR})` on the training split and return the list of test "
                       f"accuracies.",
         imp + f"result = [make_pipeline(HashingVectorizer(n_features=n), {LR}).fit(docs_train, y_train)"
               f".score(docs_test, y_test) for n in {sizes}]"),
        ("HashingVectorizer", f"How many distinct vocabulary words share a column? Fit a `CountVectorizer` on `docs` "
                              f"to get the V vocabulary words; hash each word on its own with `HashingVectorizer("
                              f"n_features={sizes[1]})` and return V minus the number of distinct columns used.",
         TXT + f"CountVectorizer, HashingVectorizer\nvocab = CountVectorizer().fit(docs).get_feature_names_out()\n"
               f"H = HashingVectorizer(n_features={sizes[1]}).transform(vocab)\n"
               "result = len(vocab) - len(np.unique(H.indices))"),
        ("transform", f"A vocabulary-based vectorizer ignores words it never saw; a hasher gives them a column anyway. "
                      f"Fit `CountVectorizer()` on `docs_train`; for the test documents, return `(number of word "
                      f"occurrences CountVectorizer still counts, number HashingVectorizer(n_features=2**20, "
                      f"norm=None, alternate_sign=False) counts)`.",
         TXT + "CountVectorizer, HashingVectorizer\ncv = CountVectorizer().fit(docs_train)\n"
               "hv = HashingVectorizer(n_features=2**20, norm=None, alternate_sign=False)\n"
               "result = (int(cv.transform(docs_test).sum()), int(hv.transform(docs_test).sum()))"),
    ]
    return catalog(rng, items, c.setup + split, f"{c.desc}. The first {k} documents are the training split "
                                                f"(`docs_train, y_train`), the rest the test split.", tol=1e-6)


@template(M, "medium")
def tfidf_weights_tour(rng):
    c = _corpus(rng, n=(80, 160))
    i = rng.randrange(20)
    k = rng.randint(3, 5)
    setup = c.setup + f"\n{TFI}tf = TfidfVectorizer()\nX = tf.fit_transform(docs)\nnames = tf.get_feature_names_out()"
    items = [
        ("idf_", f"Return the {k} words with the **largest** `idf_` (the rarest), largest first; ties in column "
                 f"order (sort `-idf_` with `kind='stable'`).",
         f"result = names[np.argsort(-tf.idf_, kind='stable')[:{k}]]"),
        ("idf_", f"Return the {k} words with the **smallest** `idf_` (the most widespread), smallest first; ties in "
                 f"column order (sort with `kind='stable'`).", f"result = names[np.argsort(tf.idf_, kind='stable')[:{k}]]"),
        ("idf_", "Recover every word's document frequency from `idf_` alone: invert idf = ln((1 + n) / (1 + df)) + 1 and "
                 "return the rounded integer array.",
         "n = len(docs)\nresult = np.rint((1 + n) / np.exp(tf.idf_ - 1) - 1).astype(int)"),
        ("toarray", f"Return the {k} words with the largest tf-idf weight in `docs[{i}]`, largest first (only words "
                    f"of that document; stable sort on ties).",
         f"row = X[{i}].toarray().ravel()\norder = np.argsort(-row, kind='stable')[:{k}]\nresult = names[order]"),
        ("max", "Which word has the largest tf-idf weight anywhere in the corpus? Return it.",
         "result = names[X.max(axis=0).toarray().ravel().argmax()]"),
        ("mean", "Average the tf-idf matrix over documents and return the word with the largest mean weight.",
         "result = names[np.asarray(X.mean(axis=0)).ravel().argmax()]"),
    ]
    return catalog(rng, items, setup, f"{c.desc}. `X` holds their tf-idf matrix (`tf = TfidfVectorizer()` fitted on "
                                      f"`docs`); `names` is the vocabulary in column order.", tol=1e-9)


# ================================================================= hard

@template(M, "hard")
def text_leakage(rng):
    c = _corpus(rng, n=(150, 240))
    k, split = _split(c)
    imp = (TFI + "from sklearn.linear_model import LogisticRegression\n")
    items = [
        ("vocabulary_", "Fitting the vectorizer on all documents leaks test-only words into the vocabulary. Return "
                        "`(vocabulary size fitted on docs, vocabulary size fitted on docs_train)`.",
         TFI + "result = (len(TfidfVectorizer().fit(docs).vocabulary_), len(TfidfVectorizer().fit(docs_train).vocabulary_))"),
        ("idf_", "The idf weights leak too. Fit `TfidfVectorizer()` on `docs_train` and on `docs`; for the words both "
                 "vocabularies share, return the largest absolute difference between the two `idf_` values.",
         TFI + "a = TfidfVectorizer().fit(docs_train)\nb = TfidfVectorizer().fit(docs)\n"
               "shared = sorted(set(a.vocabulary_) & set(b.vocabulary_))\n"
               "ia = a.idf_[[a.vocabulary_[w] for w in shared]]\nib = b.idf_[[b.vocabulary_[w] for w in shared]]\n"
               "result = float(np.abs(ia - ib).max())"),
        ("cross_val_score", f"Compare a leaky and a clean cross-validation: (a) `X_all = TfidfVectorizer()."
                            f"fit_transform(docs)` then `cross_val_score({LR}, X_all, y, cv=5)`; (b) "
                            f"`cross_val_score(make_pipeline(TfidfVectorizer(), {LR}), docs, y, cv=5)`. Return the two "
                            f"mean scores as a tuple (leaky first).",
         imp + "from sklearn.model_selection import cross_val_score\nfrom sklearn.pipeline import make_pipeline\n"
               f"leaky = cross_val_score({LR}, TfidfVectorizer().fit_transform(docs), y, cv=5).mean()\n"
               f"clean = cross_val_score(make_pipeline(TfidfVectorizer(), {LR}), docs, y, cv=5).mean()\n"
               "result = (leaky, clean)", {"require": ["cross_val_score", "make_pipeline"]}),
        ("transform", "The correct way: fit the vectorizer on the training split only and `transform` the test split. "
                      f"Do that with `TfidfVectorizer()`, train `{LR}` on the training matrix, and return the test "
                      f"accuracy.",
         imp + f"tf = TfidfVectorizer().fit(docs_train)\nclf = {LR}.fit(tf.transform(docs_train), y_train)\n"
               "result = clf.score(tf.transform(docs_test), y_test)"),
        ("build_analyzer", "Fit `TfidfVectorizer()` on `docs_train`. How many test-split word occurrences are dropped "
                      "because the word is not in the training vocabulary? Count them with the default "
                      "`CountVectorizer` tokenization (`build_analyzer()`).",
         TFI + "tf = TfidfVectorizer().fit(docs_train)\nanalyze = tf.build_analyzer()\n"
               "result = sum(w not in tf.vocabulary_ for d in docs_test for w in analyze(d))"),
    ]
    return catalog(rng, items, c.setup + split, f"{c.desc}. The first {k} documents are the training split "
                                                f"(`docs_train, y_train`), the rest the test split.", tol=1e-6)


_GRIDS = [
    ("tfidfvectorizer__ngram_range", "[(1, 1), (1, 2)]"),
    ("tfidfvectorizer__min_df", "[1, 2, 3]"),
    ("tfidfvectorizer__sublinear_tf", "[False, True]"),
    ("tfidfvectorizer__stop_words", "[None, 'english']"),
    ("logisticregression__C", "[0.1, 1.0, 10.0]"),
    ("tfidfvectorizer__use_idf", "[False, True]"),
    ("tfidfvectorizer__binary", "[False, True]"),
]


@template(M, "hard")
def text_grid_search(rng):
    c = _corpus(rng, n=(120, 200))
    g1, g2 = rng.sample(_GRIDS, 2)
    grid = f"{{{g1[0]!r}: {g1[1]}, {g2[0]!r}: {g2[1]}}}"
    what = rng.choice(["params", "score", "table"])
    imp = (TFI + "from sklearn.linear_model import LogisticRegression\nfrom sklearn.model_selection import GridSearchCV\n"
           "from sklearn.pipeline import make_pipeline\n")
    gs = f"gs = GridSearchCV(make_pipeline(TfidfVectorizer(), {LR}), {grid}, cv=3).fit(docs, y)\n"
    head = (f"{c.desc}. Tune the vectorizer and the classifier together: grid-search `make_pipeline(TfidfVectorizer(), "
            f"{LR})` over `{grid}` with `GridSearchCV(..., cv=3)` on `docs, y` (step names are the lowercased class "
            f"names, hence `tfidfvectorizer__...`).")
    if what == "params":
        return P(f"{head} Return `best_params_`.", c.setup, imp + gs + "result = gs.best_params_",
                 require=["GridSearchCV", "best_params_"])
    if what == "score":
        return P(f"{head} Return `(best_score_, the best pipeline's vocabulary size)` (the refitted vectorizer is "
                 f"`gs.best_estimator_.named_steps['tfidfvectorizer']`).", c.setup,
                 imp + gs + "result = (gs.best_score_, len(gs.best_estimator_.named_steps['tfidfvectorizer'].vocabulary_))",
                 require=["GridSearchCV", "best_estimator_"], tol=1e-6)
    return P(f"{head} Return the mean test score of every candidate as an array, in `cv_results_` order.", c.setup,
             imp + gs + "result = gs.cv_results_['mean_test_score']", require=["GridSearchCV", "cv_results_"], tol=1e-6)


@template(M, "hard")
def text_and_tabular(rng):
    c = _corpus(rng, n=(120, 200))
    s = seeded(rng)
    extra = (f"\nh = np.random.default_rng({s})\ndf = pd.DataFrame({{'text': docs, 'channel': h.choice(['web', 'app', "
             f"'mail'], len(docs)), 'length': [len(d) for d in docs]}})")
    ct = ("make_column_transformer((TfidfVectorizer(), 'text'), (OneHotEncoder(), ['channel']), "
          "(StandardScaler(), ['length']))")
    imp = (TFI + "from sklearn.compose import make_column_transformer\n"
           "from sklearn.preprocessing import OneHotEncoder, StandardScaler\n")
    items = [
        ("make_column_transformer", f"Text and ordinary columns can share one model. Build `ct = {ct}` and return the "
                                    f"shape of `ct.fit_transform(df)`. (Note the text column is passed as the string "
                                    f"`'text'`, not `['text']`: a vectorizer wants a 1-D sequence of documents.)",
         imp + f"ct = {ct}\nresult = ct.fit_transform(df).shape"),
        ("get_feature_names_out", f"Build `ct = {ct}`, fit it on `df`, and return how many output features come from "
                                  f"each part as a dict {{prefix: count}} (output names look like `tfidfvectorizer__"
                                  f"word`; the prefix is the part before `__`).",
         imp + f"ct = {ct}.fit(df)\nprefixes = [n.split('__')[0] for n in ct.get_feature_names_out()]\n"
               "result = {p: prefixes.count(p) for p in sorted(set(prefixes))}"),
        ("make_pipeline", f"Return the mean 5-fold `cross_val_score` of `make_pipeline({ct}, {LR})` on `df, y`.",
         imp + "from sklearn.linear_model import LogisticRegression\nfrom sklearn.model_selection import cross_val_score\n"
               f"from sklearn.pipeline import make_pipeline\nresult = cross_val_score(make_pipeline({ct}, {LR}), df, y, "
               "cv=5).mean()", {"require": ["make_column_transformer", "cross_val_score"]}),
        ("make_column_transformer", "This snippet crashes:\n```\nct = make_column_transformer((TfidfVectorizer(), "
                                    "['text']), remainder='drop')\nct.fit_transform(df)\n```\nFix it (the vectorizer "
                                    "needs the column as a 1-D selection) and return the shape of the result.",
         TFI + "from sklearn.compose import make_column_transformer\n"
               "ct = make_column_transformer((TfidfVectorizer(), 'text'), remainder='drop')\n"
               "result = ct.fit_transform(df).shape"),
        ("TfidfVectorizer", "How much does the text add? Return the mean 5-fold `cross_val_score` of a "
                            f"`{LR}` on the `channel` and `length` columns only (`make_column_transformer"
                            f"((OneHotEncoder(), ['channel']), (StandardScaler(), ['length']))`) and of the full "
                            f"`{ct}` pipeline, as a tuple (without text first).",
         imp + "from sklearn.linear_model import LogisticRegression\nfrom sklearn.model_selection import cross_val_score\n"
               "from sklearn.pipeline import make_pipeline\n"
               "base = make_column_transformer((OneHotEncoder(), ['channel']), (StandardScaler(), ['length']))\n"
               f"result = (cross_val_score(make_pipeline(base, {LR}), df, y, cv=5).mean(), "
               f"cross_val_score(make_pipeline({ct}, {LR}), df, y, cv=5).mean())"),
    ]
    return catalog(rng, items, c.setup + extra, f"{c.desc}. `df` puts the documents in a `text` column next to the "
                                                f"`channel` they came through and their `length` in characters.",
                   tol=1e-6)


@template(M, "hard")
def tfidf_from_scratch(rng):
    s = _small(rng, n=(5, 8))
    sub = rng.choice([False, True])
    smooth = rng.choice([True, False])
    norm = rng.choice(["'l2'", "'l1'", "None"])
    opts = f"sublinear_tf={sub}, smooth_idf={smooth}, norm={norm}"
    tf_line = "tf = np.where(C > 0, 1 + np.log(np.maximum(C, 1)), 0.0)" if sub else "tf = C.astype(float)"
    idf_line = ("idf = np.log((1 + n) / (1 + df)) + 1" if smooth else "idf = np.log(n / df) + 1")
    norm_line = {"'l2'": "W = W / np.linalg.norm(W, axis=1, keepdims=True)",
                 "'l1'": "W = W / np.abs(W).sum(axis=1, keepdims=True)", "None": ""}[norm]
    sol = ("import re\ntoks = [re.findall(r'(?u)\\b\\w\\w+\\b', d.lower()) for d in docs]\n"
           "vocab = sorted({w for t in toks for w in t})\ncol = {w: j for j, w in enumerate(vocab)}\n"
           "C = np.zeros((len(docs), len(vocab)))\nfor i, t in enumerate(toks):\n    for w in t:\n"
           "        C[i, col[w]] += 1\nn = len(docs)\ndf = (C > 0).sum(axis=0)\n"
           f"{tf_line}\n{idf_line}\nW = tf * idf\n{norm_line}\nresult = W").replace("\n\n", "\n")
    return P(f"{s.desc}. Reimplement `TfidfVectorizer({opts}).fit_transform(docs).toarray()` **without "
             f"scikit-learn**: lowercase, tokenize with the regex `r\"(?u)\\b\\w\\w+\\b\"`, sort the vocabulary "
             f"alphabetically, count, then apply the term-frequency transform (sublinear: c → 1 + ln c), the idf "
             f"(smooth: ln((1 + n)/(1 + df)) + 1, else ln(n/df) + 1) and the row normalization. Return the dense "
             f"matrix (your result should match scikit-learn's).", s.setup, sol, tol=1e-9)


@template(M, "hard")
def fix_text_bugs(rng):
    c = _corpus(rng, n=(100, 180))
    k, split = _split(c)
    one = _make_doc(rng, c.th, rng.choice(c.th.labels))
    imp = TFI + "from sklearn.linear_model import LogisticRegression\n"
    items = [
        ("transform", f"This code trains fine but crashes at prediction time:\n```\ntf = TfidfVectorizer()\n"
                      f"clf = {LR}.fit(tf.fit_transform(docs_train), y_train)\n"
                      f"pred = clf.predict(TfidfVectorizer().fit_transform(docs_test))\n```\nThe test split got its own "
                      f"vocabulary (different columns). Fix it and return the test predictions.",
         imp + f"tf = TfidfVectorizer()\nclf = {LR}.fit(tf.fit_transform(docs_train), y_train)\n"
               "result = clf.predict(tf.transform(docs_test))"),
        ("predict", f"This crashes with 'Iterable over raw text documents expected, string object received':\n```\n"
                    f"clf = make_pipeline(TfidfVectorizer(), {LR}).fit(docs, y)\nclf.predict(one_doc)\n```\nFix it "
                    f"and return the prediction for `one_doc` (an array with one label).",
         imp + "from sklearn.pipeline import make_pipeline\n"
               f"clf = make_pipeline(TfidfVectorizer(), {LR}).fit(docs, y)\nresult = clf.predict([one_doc])"),
        ("get_feature_names_out", f"This 'top words' code is wrong:\n```\ntf = TfidfVectorizer()\n"
                                  f"clf = {LR}.fit(tf.fit_transform(docs), y)\n"
                                  f"top = np.argsort(clf.coef_[0])[:5]\nwords = sorted(tf.vocabulary_)[::-1]\n"
                                  f"result = [words[j] for j in top]\n```\nIt takes the 5 *smallest* coefficients "
                                  f"and indexes a reversed word list. Return the 5 words with the **largest** "
                                  f"coefficients in `coef_[0]`, largest first.",
         imp + f"tf = TfidfVectorizer()\nclf = {LR}.fit(tf.fit_transform(docs), y)\n"
               "result = list(tf.get_feature_names_out()[np.argsort(clf.coef_[0])[::-1][:5]])"),
        ("fit_transform", "This code builds a vocabulary per document by mistake:\n```\n"
                          "rows = [CountVectorizer().fit_transform([d]).toarray() for d in docs]\n```\nEvery row has its "
                          "own columns. Build **one** `CountVectorizer` on all of `docs` and return the dense count "
                          "matrix's shape and its total sum as a tuple.",
         CVI + "X = CountVectorizer().fit_transform(docs)\nresult = (X.shape, int(X.sum()))"),
        ("TfidfVectorizer", f"This pipeline uses tf-idf twice: `make_pipeline(TfidfVectorizer(), TfidfTransformer(), "
                            f"{LR})`. It runs, but the second reweighting distorts the features. Remove the extra step, "
                            f"train on the training split and return the test accuracy.",
         imp + "from sklearn.pipeline import make_pipeline\n"
               f"result = make_pipeline(TfidfVectorizer(), {LR}).fit(docs_train, y_train).score(docs_test, y_test)"),
    ]
    return catalog(rng, items, c.setup + split + f"\none_doc = {one!r}",
                   f"{c.desc}. The first {k} documents are the training split (`docs_train, y_train`), the rest the "
                   f"test split; `one_doc` is a single new document.", tol=1e-6)


@template(M, "hard")
def keyword_selection(rng):
    c = _corpus(rng, n=(120, 220))
    k, split = _split(c)
    m = rng.choice([5, 10, 20])
    imp = CVI + "from sklearn.feature_selection import SelectKBest, chi2\n"
    items = [
        ("chi2", f"`chi2` scores how strongly each word count depends on the class. Fit `SelectKBest(chi2, k={m})` "
                 f"(from `sklearn.feature_selection`) on the `CountVectorizer()` counts of `docs`, `y` and return the "
                 f"selected words, alphabetical.",
         imp + f"cv = CountVectorizer()\nX = cv.fit_transform(docs)\nsel = SelectKBest(chi2, k={m}).fit(X, y)\n"
               "result = sorted(cv.get_feature_names_out()[sel.get_support()])", {"require": ["SelectKBest", "chi2"]}),
        ("chi2", "Return the word with the highest chi² score on the `CountVectorizer()` counts of `docs`, `y` "
                 "(`chi2(X, y)` returns `(scores, p_values)`).",
         imp + "cv = CountVectorizer()\nX = cv.fit_transform(docs)\nscores, _ = chi2(X, y)\n"
               "result = cv.get_feature_names_out()[np.argmax(scores)]"),
        ("SelectKBest", f"Does keeping only {m} words hurt? Train `make_pipeline(CountVectorizer(), "
                        f"SelectKBest(chi2, k={m}), {LR})` and the same pipeline without the selection on the training "
                        f"split; return the two test accuracies as a tuple (with selection first).",
         imp + "from sklearn.linear_model import LogisticRegression\nfrom sklearn.pipeline import make_pipeline\n"
               f"a = make_pipeline(CountVectorizer(), SelectKBest(chi2, k={m}), {LR}).fit(docs_train, y_train)\n"
               f"b = make_pipeline(CountVectorizer(), {LR}).fit(docs_train, y_train)\n"
               "result = (a.score(docs_test, y_test), b.score(docs_test, y_test))"),
        ("SelectKBest", f"Selecting words on all documents before cross-validating leaks the labels. Compute the mean "
                        f"5-fold `cross_val_score` of `make_pipeline(CountVectorizer(), SelectKBest(chi2, k={m}), "
                        f"{LR})` on `docs, y` (selection inside each fold) and return it.",
         imp + "from sklearn.linear_model import LogisticRegression\nfrom sklearn.model_selection import cross_val_score\n"
               "from sklearn.pipeline import make_pipeline\n"
               f"result = cross_val_score(make_pipeline(CountVectorizer(), SelectKBest(chi2, k={m}), {LR}), docs, y, "
               "cv=5).mean()"),
    ]
    return catalog(rng, items, c.setup + split, f"{c.desc}. The first {k} documents are the training split "
                                                f"(`docs_train, y_train`), the rest the test split.", tol=1e-6)


# ================================================================= challenge

def _token_stream(var_s, n_docs, vocab, lo, hi, prefix="g"):
    return (f"{prefix} = np.random.default_rng({var_s})\nlengths = {prefix}.integers({lo}, {hi}, {n_docs})\n"
            f"starts = np.concatenate([[0], np.cumsum(lengths)])\n"
            f"tokens = np.minimum({prefix}.zipf(1.4, starts[-1]) - 1, {vocab} - 1)\nV = {vocab}")


@template(M, "challenge")
def tfidf_token_stream(rng):
    s = seeded(rng)
    nd, V = rng.randint(4, 6), rng.randint(6, 9)
    report = rng.choice(["idf", "norms", "top"])
    setup = _token_stream(s, nd, V, 2, 7) + f"\nreport = {report!r}"
    tests = [_token_stream(s + i, 40000, 5000, 1, 9) + f"\nreport = {report!r}" for i in (1, 2)]
    base = ("D = len(starts) - 1\ndoc = np.repeat(np.arange(D), np.diff(starts))\n"
            "key, cnt = np.unique(doc.astype(np.int64) * V + tokens, return_counts=True)\nd_of, w_of = key // V, key % V\n"
            "df = np.bincount(w_of, minlength=V)\nidf = np.log((1 + D) / (1 + df)) + 1\n")
    if report == "idf":
        sol = base + "result = idf"
        want = "the `idf_` array of length `V` (term `t` is column `t`; a term no document contains has df = 0)"
    elif report == "norms":
        sol = base + "w = cnt * idf[w_of]\nresult = np.sqrt(np.bincount(d_of, weights=w * w, minlength=D))"
        want = "for every document the Euclidean norm of its **unnormalized** tf-idf row (`norm=None`), as a float array"
    else:
        sol = (base + "w = cnt * idf[w_of]\norder = np.lexsort((w_of, -w, d_of))\nfirst = np.ones(len(order), bool)\n"
               "first[1:] = d_of[order][1:] != d_of[order][:-1]\nresult = w_of[order][first]")
        want = ("for every document the term with the **largest** tf-idf weight (smallest term id on ties), as an int "
                "array")
    return challenge(
        "TF-IDF on a token stream",
        "A corpus arrives already tokenized: `tokens` holds term ids (0 ≤ id < `V`) of all documents concatenated, and "
        "document `d` is `tokens[starts[d]:starts[d + 1]]` (every document has at least one token). Compute what "
        "`TfidfVectorizer` (default smoothing: idf = ln((1 + D) / (1 + df)) + 1, with D documents and df the number of "
        f"documents containing the term; tf = raw count) would give, and return {want}. (`report` = {report!r}.)",
        setup, sol, tests, "total tokens ≤ 2·10^5, V ≤ 10^4, documents ≤ 5·10^4",
        speed="O(T log T) with sorting or bincount, no Python loop over tokens", tol=1e-6)


def _words_setup(s, n, length, alphabet, nf, report, prefix="g"):
    return (f"{prefix} = np.random.default_rng({s})\nletters = np.array(list({alphabet!r}))\n"
            f"grid = letters[{prefix}.integers(0, len(letters), ({n}, {length}))]   # one random word per row\n"
            f"words = np.unique(grid.view('<U{length}').ravel()).tolist()   # distinct, sorted\n"
            f"n_features = {nf}\nreport = {report!r}")


@template(M, "challenge")
def hashed_bucket_collisions(rng):
    s = seeded(rng)
    report = rng.choice(["pairs", "buckets", "largest"])
    nf = rng.choice([4, 8, 16])
    setup = _words_setup(s, rng.randint(8, 12), rng.randint(2, 4), "abcdefgh", nf, report)
    tests = [_words_setup(s + i, 150000, 6, "abcdefghijklmnopqrstuvwxyz", 2 ** 14, report) for i in (1, 2)]
    sol = ("from sklearn.feature_extraction import FeatureHasher\n"
           "H = FeatureHasher(n_features=n_features, input_type='string', alternate_sign=False)\n"
           "col = H.transform([[w] for w in words]).indices\nsizes = np.bincount(col, minlength=n_features)\n")
    sol += {"pairs": "result = int((sizes * (sizes - 1) // 2).sum())",
            "buckets": "result = int((sizes > 0).sum())", "largest": "result = int(sizes.max())"}[report]
    want = {"pairs": "the number of **pairs of distinct words** that land in the same column",
            "buckets": "the number of columns that receive at least one word",
            "largest": "the largest number of words sharing one column"}[report]
    return challenge(
        "A crowded hash table",
        "A team wants to replace its vocabulary with the hashing trick. `words` holds the distinct vocabulary words. "
        "Map every word to its column exactly as `FeatureHasher(n_features=n_features, input_type='string')` (or a "
        "`HashingVectorizer`) would (scikit-learn hashes with MurmurHash3, `sklearn.utils.murmurhash3_32`, seed 0, "
        f"and takes |hash| mod n_features), and return {want}. (`report` = {report!r}.)",
        setup, sol, tests, "number of words ≤ 1.5·10^5, n_features ≤ 2^14",
        speed="O(W) hashing (vectorized: let a hasher map all words at once) and a bincount, no pairwise comparisons")


def _df_setup(s, nd, V, lo, hi, a, b, report, prefix="g"):
    return _token_stream(s, nd, V, lo, hi, prefix) + f"\nmin_df = {a}\nmax_df = {b}\nreport = {report!r}"


@template(M, "challenge")
def df_pruned_vocabulary(rng):
    s = seeded(rng)
    report = rng.choice(["kept", "count", "dropped_tokens"])
    nd, V = rng.randint(6, 9), rng.randint(6, 9)
    a, b = rng.choice([2, 3]), rng.choice([0.5, 0.6, 0.75])
    setup = _df_setup(s, nd, V, 2, 6, a, b, report)
    tests = [_df_setup(s + i, 30000, 20000, 1, 12, 5, 0.3, report) for i in (1, 2)]
    sol = ("D = len(starts) - 1\ndoc = np.repeat(np.arange(D), np.diff(starts)).astype(np.int64)\n"
           "pairs = np.unique(doc * V + tokens)\ndf = np.bincount(pairs % V, minlength=V)\n"
           "keep = (df >= min_df) & (df <= max_df * D)\n")
    sol += {"kept": "result = np.flatnonzero(keep)", "count": "result = int(keep.sum())",
            "dropped_tokens": "result = int((~keep[tokens]).sum())"}[report]
    want = {"kept": "the sorted term ids that survive, as an int array",
            "count": "how many term ids survive",
            "dropped_tokens": "how many tokens of the corpus (occurrences, not distinct terms) are dropped because "
                              "their term is pruned"}[report]
    return challenge(
        "Pruning by document frequency",
        "Before vectorizing, rare and ubiquitous terms are pruned like `CountVectorizer(min_df=min_df, "
        "max_df=max_df)` does: a term is kept when it occurs in at least `min_df` documents (an int) and in at most "
        "`max_df × D` documents (a fraction of the D documents). The corpus is tokenized: `tokens` holds term ids "
        "(0 ≤ id < `V`) of all documents concatenated and document `d` is `tokens[starts[d]:starts[d + 1]]`. A term "
        f"id that never occurs has document frequency 0. Return {want}. (`report` = {report!r}.)",
        setup, sol, tests, "total tokens ≤ 2·10^5, V ≤ 2·10^4",
        speed="O(T log T): count each (document, term) pair once")


def _bigram_setup(s, nd, V, lo, hi, report, prefix="g"):
    return _token_stream(s, nd, V, lo, hi, prefix) + f"\nreport = {report!r}"


@template(M, "challenge")
def bigram_vocabulary(rng):
    s = seeded(rng)
    report = rng.choice(["distinct", "in_two_docs", "top"])
    setup = _bigram_setup(s, rng.randint(4, 6), rng.randint(4, 6), 2, 7, report)
    tests = [_bigram_setup(s + i, 30000, 3000, 2, 12, report) for i in (1, 2)]
    sol = ("D = len(starts) - 1\ndoc = np.repeat(np.arange(D), np.diff(starts)).astype(np.int64)\n"
           "same = doc[:-1] == doc[1:]\na, b = tokens[:-1][same].astype(np.int64), tokens[1:][same].astype(np.int64)\n"
           "code = a * V + b\n")
    sol += {"distinct": "result = int(len(np.unique(code)))",
            "in_two_docs": "pairs = np.unique(doc[:-1][same] * V * V + code)\n"
                           "df = np.unique(pairs % (V * V), return_counts=True)[1]\nresult = int((df >= 2).sum())",
            "top": "u, cnt = np.unique(code, return_counts=True)\nbest = u[cnt.argmax()]\n"
                   "result = (int(best // V), int(best % V))"}[report]
    want = {"distinct": "how many **distinct** bigrams the corpus has (the size of `CountVectorizer(ngram_range=(2, 2))`'s "
                        "vocabulary)",
            "in_two_docs": "how many distinct bigrams occur in at least 2 different documents (they would survive "
                           "`min_df=2`)",
            "top": "the most frequent bigram over the whole corpus as a tuple `(first id, second id)` (on ties, the "
                   "smallest pair in lexicographic order)"}[report]
    return challenge(
        "Counting bigrams",
        "A bigram is a pair of **adjacent** tokens inside one document (pairs never span two documents). The corpus is "
        "tokenized: `tokens` holds term ids (0 ≤ id < `V`) of all documents concatenated and document `d` is "
        f"`tokens[starts[d]:starts[d + 1]]`. Return {want}. (`report` = {report!r}.)",
        setup, sol, tests, "total tokens ≤ 2·10^5, V ≤ 3000",
        speed="O(T log T): encode each bigram as one integer, then sort / unique")


def _nb_setup(s, n_train, n_test, V, lo, hi, alpha, report, prefix="g"):
    return (f"{prefix} = np.random.default_rng({s})\nV = {V}\nalpha = {alpha}\n"
            f"bias = {prefix}.dirichlet(np.ones(V) * 0.5, 2)       # word distributions of class 0 and 1\n"
            f"y_train = {prefix}.permutation(np.arange({n_train}) % 2)   # both classes occur\n"
            f"len_train = {prefix}.integers({lo}, {hi}, {n_train})\nlen_test = {prefix}.integers({lo}, {hi}, {n_test})\n"
            f"train_tokens = np.concatenate([{prefix}.choice(V, k, p=bias[c]) for c, k in zip(y_train, len_train)])\n"
            f"test_tokens = np.concatenate([{prefix}.choice(V, k, p=bias[{prefix}.integers(2)]) for k in len_test])\n"
            f"report = {report!r}")


@template(M, "challenge")
def multinomial_nb_tokens(rng):
    s = seeded(rng)
    report = rng.choice(["labels", "log_odds"])
    alpha = rng.choice([1.0, 0.5, 0.1])
    setup = _nb_setup(s, rng.randint(5, 8), rng.randint(3, 5), rng.randint(5, 8), 2, 6, alpha, report)
    tests = [_nb_setup(s + i, 6000, 6000, 2000, 5, 15, alpha, report) for i in (1, 2)]
    sol = ("tr_doc = np.repeat(np.arange(len(len_train)), len_train)\nte_doc = np.repeat(np.arange(len(len_test)), len_test)\n"
           "counts = np.zeros((2, V))\nnp.add.at(counts, (y_train[tr_doc], train_tokens), 1)\n"
           "logp = np.log(counts + alpha) - np.log((counts + alpha).sum(axis=1, keepdims=True))\n"
           "prior = np.log(np.bincount(y_train, minlength=2) / len(y_train))\n"
           "score = np.zeros((len(len_test), 2))\nnp.add.at(score, te_doc, logp[:, test_tokens].T)\nscore += prior\n")
    sol += ("result = score.argmax(axis=1)" if report == "labels" else "result = score[:, 1] - score[:, 0]")
    want = ("the predicted label of every test document (an int array; ties go to class 0)" if report == "labels" else
            "for every test document the log-odds log P(1 | doc) − log P(0 | doc) the model computes (a float array)")
    return challenge(
        "Naive Bayes from token streams",
        "Train `MultinomialNB(alpha=alpha)` on documents given as token streams, without building a count matrix row "
        "by row. Training document `i` has `len_train[i]` tokens (ids 0 ≤ id < `V`), stored one after another in "
        "`train_tokens`, with label `y_train[i]` (0 or 1; both classes occur). Test documents are stored the same way "
        "in `test_tokens` / `len_test`. The model: class priors are the label frequencies; log P(word w | class c) = "
        "log((count of w in class c + alpha) / (total tokens of class c + alpha·V)); a document's class score is the "
        "log prior plus the sum of its tokens' log probabilities (repeats count each time). Return "
        f"{want}. (`report` = {report!r}.)",
        setup, sol, tests, "training and test tokens ≤ 10^5 each, V ≤ 2000",
        speed="O(T + V) with np.add.at / bincount, no Python loop over tokens", tol=1e-6)
