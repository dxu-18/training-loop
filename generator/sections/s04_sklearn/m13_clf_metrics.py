"""4.13 Classification metrics: confusion matrix, precision/recall/F, ROC and PR curves, probabilistic scores,
multiclass averaging and multilabel metrics."""

from ._common import P, catalog, challenge, lit, mat, seeded, synth_clf, table, template

M = "4.13"
MET = "from sklearn.metrics import "
PIPE = "make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))"


def _labels(rng, n=(14, 20), pos=(0.3, 0.5)):
    """Literal binary labels, scores and the 0.5-threshold predictions."""
    n_ = rng.randint(*n)
    p1 = rng.uniform(*pos)
    while True:
        y = [int(rng.random() < p1) for _ in range(n_)]
        prob = [round(rng.uniform(0.25, 0.98) if t else rng.uniform(0.02, 0.75), 2) for t in y]
        pred = [int(p >= 0.5) for p in prob]
        if 0 < sum(y) < n_ and 0 < sum(pred) < n_ and len(set(prob)) == n_:
            break
    setup = f"y_true = np.array({lit(y)})\ny_pred = np.array({lit(pred)})\ny_prob = np.array({lit(prob)})"
    desc = (f"`y_true` holds {n_} true labels (1 = positive), `y_pred` a classifier's predictions and `y_prob` its "
            f"predicted probabilities of class 1 (`y_pred` is `y_prob >= 0.5`)")
    return setup, desc, n_


def _multi(rng, n=(15, 24), k=3):
    """Literal 3-class labels, a probability matrix and its argmax predictions."""
    n_ = rng.randint(*n)
    while True:
        y = [rng.randrange(k) for _ in range(n_)]
        P_ = []
        for t in y:
            w = [rng.uniform(0.1, 1.0) for _ in range(k)]
            w[t] += rng.uniform(0.0, 1.6)
            s = sum(w)
            row = [round(v / s, 3) for v in w[:-1]]
            row.append(round(1 - sum(row), 3))
            P_.append(row)
        pred = [max(range(k), key=lambda j: r[j]) for r in P_]
        if len(set(y)) == k and len(set(pred)) == k and all(len(set(r)) == k for r in P_):
            break
    setup = f"y_true = np.array({lit(y)})\ny_pred = np.array({lit(pred)})\nproba = np.array({mat(P_)})"
    desc = (f"`y_true` holds {n_} true labels from 3 classes (0, 1, 2), `proba` a model's predicted probabilities (one "
            f"row per sample, one column per class) and `y_pred` its predictions (the argmax of each row)")
    return setup, desc, n_


def _multilabel(rng, n=(6, 10), L=(3, 5)):
    n_, L_ = rng.randint(*n), rng.randint(*L)
    while True:
        Yt = [[int(rng.random() < 0.45) for _ in range(L_)] for _ in range(n_)]
        Yp = [[v if rng.random() < 0.8 else 1 - v for v in row] for row in Yt]
        if all(any(r) for r in Yt) and all(any(r) for r in Yp) and all(any(c) for c in zip(*Yt)) \
                and all(any(c) for c in zip(*Yp)) and Yt != Yp:
            break
    setup = f"Y_true = np.array({mat(Yt)})\nY_pred = np.array({mat(Yp)})"
    desc = (f"`Y_true` and `Y_pred` are {n_} × {L_} multilabel indicator matrices (row = sample, column = label, 1 = "
            f"the label applies): the truth and a model's predictions")
    return setup, desc, n_, L_


def _fitted_binary(rng, imbalance=None):
    """A logistic-regression pipeline fitted in the setup; test labels, predictions and probabilities."""
    kind = rng.choice(["cancer", "synth", "table"]) if imbalance is None else "synth"
    s = seeded(rng)
    if kind == "cancer":
        data = "from sklearn.datasets import load_breast_cancer\nX, y = load_breast_cancer(return_X_y=True)"
        what = "the breast-cancer data (1 = benign)"
    elif kind == "synth":
        dd = synth_clf(rng, n=(300, 600), d=(4, 10), classes=2, imbalance=imbalance)
        data = dd.setup
        what = f"a `make_classification` problem ({dd.n} rows{', class 1 rare' if imbalance else ''})"
    else:
        t = table(rng, kind="clf", numeric_only=True, n=(250, 450))
        data = t.setup + "\nX = df.to_numpy()"
        what = f"{t.n} {t.noun}s (y = {t.target})"
    setup = (data + "\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.pipeline import make_pipeline\n"
             "from sklearn.preprocessing import StandardScaler\nfrom sklearn.linear_model import LogisticRegression\n"
             f"X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, random_state={s})\n"
             f"clf = {PIPE}.fit(X_tr, y_tr)\ny_pred = clf.predict(X_te)\ny_prob = clf.predict_proba(X_te)[:, 1]")
    desc = (f"`clf = {PIPE}` was trained on a stratified split of {what}; `y_te` holds the test labels, `y_pred = "
            f"clf.predict(X_te)` and `y_prob = clf.predict_proba(X_te)[:, 1]` (P(class 1))")
    return setup, desc


def _fitted_multi(rng):
    name = rng.choice(["iris", "wine", "digits"])
    s = seeded(rng)
    if name == "digits":
        data = "from sklearn.datasets import load_digits\nX, y = load_digits(return_X_y=True)\nX, y = X[:600], y[:600]"
        what, k = "600 handwritten digits (10 classes)", 10
    else:
        fn = {"iris": "load_iris", "wine": "load_wine"}[name]
        data = f"from sklearn.datasets import {fn}\nX, y = {fn}(return_X_y=True)"
        what, k = f"the {name} data (3 classes)", 3
    setup = (data + "\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.pipeline import make_pipeline\n"
             "from sklearn.preprocessing import StandardScaler\nfrom sklearn.linear_model import LogisticRegression\n"
             f"X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4, stratify=y, random_state={s})\n"
             f"clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)).fit(X_tr, y_tr)\n"
             "y_pred = clf.predict(X_te)\nproba = clf.predict_proba(X_te)")
    desc = (f"`clf` (StandardScaler + LogisticRegression(C=0.1)) was trained on a stratified split of {what}; `y_te` "
            f"holds the test labels, `y_pred` the predictions and `proba = clf.predict_proba(X_te)`")
    return setup, desc, k


# ================================================================= lite

@template(M, "lite")
def binary_metric_calls(rng):
    setup, desc, n = _labels(rng)
    b = rng.choice([0.5, 2])
    items = [
        ("accuracy_score", "Return the accuracy of `y_pred` (`accuracy_score`, from `sklearn.metrics`).",
         f"{MET}accuracy_score\nresult = accuracy_score(y_true, y_pred)"),
        ("precision_score", "Return the precision of `y_pred`: of the samples predicted positive, the share that are "
                            "positive.", f"{MET}precision_score\nresult = precision_score(y_true, y_pred)"),
        ("recall_score", "Return the recall of `y_pred`: of the positive samples, the share predicted positive.",
         f"{MET}recall_score\nresult = recall_score(y_true, y_pred)"),
        ("f1_score", "Return the F1 score of `y_pred` (the harmonic mean of precision and recall).",
         f"{MET}f1_score\nresult = f1_score(y_true, y_pred)"),
        ("fbeta_score", f"Return the F-beta score of `y_pred` with `beta={b}` (`fbeta_score`; beta > 1 weighs recall "
                        f"more, beta < 1 precision).", f"{MET}fbeta_score\nresult = fbeta_score(y_true, y_pred, beta={b})"),
        ("balanced_accuracy_score", "Return the balanced accuracy of `y_pred` (the mean of the recalls of both classes).",
         f"{MET}balanced_accuracy_score\nresult = balanced_accuracy_score(y_true, y_pred)"),
        ("confusion_matrix", "Return the confusion matrix of `y_pred` (rows = true class, columns = predicted class).",
         f"{MET}confusion_matrix\nresult = confusion_matrix(y_true, y_pred)"),
        ("matthews_corrcoef", "Return the Matthews correlation coefficient of `y_pred` (−1 to 1; 0 = no better than "
                              "chance).", f"{MET}matthews_corrcoef\nresult = matthews_corrcoef(y_true, y_pred)"),
        ("cohen_kappa_score", "Return Cohen's kappa between `y_true` and `y_pred` (agreement corrected for chance).",
         f"{MET}cohen_kappa_score\nresult = cohen_kappa_score(y_true, y_pred)"),
        ("roc_auc_score", "Return the ROC AUC of the scores `y_prob` (AUC needs scores, not hard labels).",
         f"{MET}roc_auc_score\nresult = roc_auc_score(y_true, y_prob)"),
        ("average_precision_score", "Return the average precision (the area under the precision-recall curve, as "
                                    "`average_precision_score` computes it) of `y_prob`.",
         f"{MET}average_precision_score\nresult = average_precision_score(y_true, y_prob)"),
        ("log_loss", "Return the log loss (cross-entropy) of the probabilities `y_prob`.",
         f"{MET}log_loss\nresult = log_loss(y_true, y_prob)"),
        ("brier_score_loss", "Return the Brier score of `y_prob`: the mean squared difference between the "
                             "probabilities and the 0/1 labels.", f"{MET}brier_score_loss\nresult = brier_score_loss(y_true, y_prob)"),
    ]
    return catalog(rng, items, setup, f"{desc}.", tol=1e-9)


@template(M, "lite")
def multiclass_metric_calls(rng):
    setup, desc, n = _multi(rng)
    avg = rng.choice(["macro", "weighted", "micro"])
    items = [
        ("f1_score", f"Return the `{avg}`-averaged F1 score of `y_pred` (`f1_score(..., average={avg!r})`).",
         f"{MET}f1_score\nresult = f1_score(y_true, y_pred, average={avg!r})"),
        ("recall_score", "Return the recall of every class (`average=None`), as an array in class order.",
         f"{MET}recall_score\nresult = recall_score(y_true, y_pred, average=None)"),
        ("precision_score", f"Return the `{avg}`-averaged precision of `y_pred`.",
         f"{MET}precision_score\nresult = precision_score(y_true, y_pred, average={avg!r})"),
        ("confusion_matrix", "Return the confusion matrix normalized over each **true** class (`normalize='true'`: rows "
                             "sum to 1).", f"{MET}confusion_matrix\nresult = confusion_matrix(y_true, y_pred, normalize='true')"),
        ("classification_report", "Return the macro-averaged F1 read from `classification_report(y_true, y_pred, "
                                  "output_dict=True)` (the `'macro avg'` entry, key `'f1-score'`).",
         f"{MET}classification_report\nresult = classification_report(y_true, y_pred, output_dict=True)['macro avg']['f1-score']"),
        ("precision_recall_fscore_support", "Return `precision_recall_fscore_support(y_true, y_pred, average=None)`: "
                                            "four arrays (precision, recall, F1, support per class).",
         f"{MET}precision_recall_fscore_support\nresult = precision_recall_fscore_support(y_true, y_pred, average=None)"),
        ("top_k_accuracy_score", "Return the top-2 accuracy of `proba`: the share of samples whose true class is among "
                                 "the 2 most probable ones (`top_k_accuracy_score(..., k=2)`).",
         f"{MET}top_k_accuracy_score\nresult = top_k_accuracy_score(y_true, proba, k=2)"),
        ("roc_auc_score", "Return the one-vs-rest multiclass ROC AUC of `proba` (`multi_class='ovr'`).",
         f"{MET}roc_auc_score\nresult = roc_auc_score(y_true, proba, multi_class='ovr')"),
        ("jaccard_score", "Return the macro-averaged Jaccard score of `y_pred` (per class: |true ∩ predicted| / |true ∪ "
                          "predicted|).", f"{MET}jaccard_score\nresult = jaccard_score(y_true, y_pred, average='macro')"),
        ("log_loss", "Return the multiclass log loss of `proba`.", f"{MET}log_loss\nresult = log_loss(y_true, proba)"),
        ("accuracy_score", "Return how many samples `y_pred` gets right (`accuracy_score(..., normalize=False)` counts "
                           "instead of averaging).", f"{MET}accuracy_score\nresult = accuracy_score(y_true, y_pred, normalize=False)"),
    ]
    return catalog(rng, items, setup, f"{desc}.", tol=1e-9)


@template(M, "lite")
def multilabel_metric_calls(rng):
    setup, desc, n, L = _multilabel(rng)
    items = [
        ("hamming_loss", "Return the Hamming loss: the share of individual label entries that are wrong.",
         f"{MET}hamming_loss\nresult = hamming_loss(Y_true, Y_pred)"),
        ("jaccard_score", "Return the sample-averaged Jaccard score (`average='samples'`: per row |true ∩ pred| / "
                          "|true ∪ pred|, averaged).", f"{MET}jaccard_score\nresult = jaccard_score(Y_true, Y_pred, average='samples')"),
        ("f1_score", "Return the micro-averaged F1 (`average='micro'`: pool every label decision).",
         f"{MET}f1_score\nresult = f1_score(Y_true, Y_pred, average='micro')"),
        ("accuracy_score", "Return `accuracy_score(Y_true, Y_pred)`: for multilabel input it is the *subset accuracy*, "
                           "the share of rows predicted exactly right.",
         f"{MET}accuracy_score\nresult = accuracy_score(Y_true, Y_pred)"),
        ("precision_score", "Return the precision of every label (`average=None`), one value per column.",
         f"{MET}precision_score\nresult = precision_score(Y_true, Y_pred, average=None)"),
        ("multilabel_confusion_matrix", "Return `multilabel_confusion_matrix(Y_true, Y_pred)`: one 2 × 2 matrix "
                                        "`[[tn, fp], [fn, tp]]` per label.",
         f"{MET}multilabel_confusion_matrix\nresult = multilabel_confusion_matrix(Y_true, Y_pred)"),
    ]
    return catalog(rng, items, setup, f"{desc}.", tol=1e-9)


@template(M, "lite")
def report_and_matrix(rng):
    setup, desc, n = _labels(rng)
    names = rng.choice([["negative", "positive"], ["healthy", "sick"], ["ham", "spam"], ["stayed", "churned"]])
    dg = rng.choice([2, 3, 4])
    items = [
        ("ravel", "Unpack the binary confusion matrix: return `(tn, fp, fn, tp)` from `confusion_matrix(y_true, "
                  "y_pred).ravel()` as plain ints.",
         f"{MET}confusion_matrix\ntn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()\n"
         "result = (int(tn), int(fp), int(fn), int(tp))"),
        ("classification_report", f"Return the text of `classification_report(y_true, y_pred, target_names={names}, "
                                  f"digits={dg})`.",
         f"{MET}classification_report\nresult = classification_report(y_true, y_pred, target_names={names}, digits={dg})"),
        ("output_dict", f"Return the recall of class `{names[1]!r}` read from `classification_report(y_true, y_pred, "
                        f"target_names={names}, output_dict=True)`.",
         f"{MET}classification_report\n"
         f"result = classification_report(y_true, y_pred, target_names={names}, output_dict=True)[{names[1]!r}]['recall']"),
        ("confusion_matrix", "Return the confusion matrix normalized over each **predicted** class (`normalize='pred'`: "
                             "columns sum to 1, so the diagonal holds each class's precision).",
         f"{MET}confusion_matrix\nresult = confusion_matrix(y_true, y_pred, normalize='pred')"),
    ]
    return catalog(rng, items, setup, f"{desc}.", tol=1e-9)


# ================================================================= easy

@template(M, "easy")
def metrics_by_hand(rng):
    setup, desc, n = _labels(rng, n=(16, 26))
    task = rng.choice(["prf", "specificity", "fbeta", "mcc", "balanced"])
    cm = f"{MET}confusion_matrix\ntn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()\n"
    head = f"{desc}. Get `tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()` and compute"
    if task == "prf":
        return P(f"{head} precision, recall and F1 **by hand** from them; return `((p, r, f1), (precision_score, "
                 f"recall_score, f1_score))` with the second triple from `sklearn.metrics`.", setup,
                 cm + f"{MET}precision_score, recall_score, f1_score\np = tp / (tp + fp)\nr = tp / (tp + fn)\n"
                      "result = ((p, r, 2 * p * r / (p + r)), (precision_score(y_true, y_pred), recall_score(y_true, y_pred), "
                      "f1_score(y_true, y_pred)))", require=["confusion_matrix", "f1_score"], tol=1e-9)
    if task == "specificity":
        return P(f"{head} the **specificity** (true-negative rate, tn / (tn + fp)) by hand. It is the recall of class 0: "
                 f"return `(by_hand, recall_score(y_true, y_pred, pos_label=0))`.", setup,
                 cm + f"{MET}recall_score\nresult = (tn / (tn + fp), recall_score(y_true, y_pred, pos_label=0))",
                 require=["confusion_matrix", "pos_label"], tol=1e-9)
    if task == "fbeta":
        b = rng.choice([0.5, 2, 3])
        return P(f"{head} F-beta with beta = {b} by hand: `(1 + β²)·p·r / (β²·p + r)`. Return `(by_hand, "
                 f"fbeta_score(y_true, y_pred, beta={b}))`.", setup,
                 cm + f"{MET}fbeta_score\np, r = tp / (tp + fp), tp / (tp + fn)\n"
                      f"result = ((1 + {b} ** 2) * p * r / ({b} ** 2 * p + r), fbeta_score(y_true, y_pred, beta={b}))",
                 require=["confusion_matrix", "fbeta_score"], tol=1e-9)
    if task == "mcc":
        return P(f"{head} the Matthews correlation by hand: `(tp·tn − fp·fn) / sqrt((tp+fp)(tp+fn)(tn+fp)(tn+fn))`. "
                 f"Return `(by_hand, matthews_corrcoef(y_true, y_pred))`.", setup,
                 cm + f"{MET}matthews_corrcoef\n"
                      "result = ((tp * tn - fp * fn) / np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))), "
                      "matthews_corrcoef(y_true, y_pred))", require=["confusion_matrix", "matthews_corrcoef"], tol=1e-9)
    return P(f"{head} the balanced accuracy by hand (the mean of the true-positive and true-negative rates). Return "
             f"`(by_hand, balanced_accuracy_score(y_true, y_pred), accuracy_score(y_true, y_pred))`.", setup,
             cm + f"{MET}balanced_accuracy_score, accuracy_score\n"
                  "result = ((tp / (tp + fn) + tn / (tn + fp)) / 2, balanced_accuracy_score(y_true, y_pred), "
                  "accuracy_score(y_true, y_pred))", require=["confusion_matrix", "balanced_accuracy_score"], tol=1e-9)


@template(M, "easy")
def threshold_effects(rng):
    setup, desc = _fitted_binary(rng)
    t = rng.choice([0.2, 0.3, 0.4, 0.6, 0.7, 0.8])
    task = rng.choice(["pr", "sweep", "count"])
    if task == "pr":
        return P(f"{desc}. Use a custom decision threshold: predict 1 when `y_prob >= {t}` (`.astype(int)`). Return "
                 f"`(precision, recall)` at that threshold and at the default 0.5 (`y_pred`), as two tuples.", setup,
                 f"{MET}precision_score, recall_score\ncustom = (y_prob >= {t}).astype(int)\n"
                 "result = ((precision_score(y_te, custom), recall_score(y_te, custom)), "
                 "(precision_score(y_te, y_pred), recall_score(y_te, y_pred)))", require=["astype", "precision_score"],
                 tol=1e-9)
    ts = [0.1, 0.3, 0.5, 0.7, 0.9]
    if task == "sweep":
        return P(f"{desc}. For thresholds `{ts}` (predict 1 when `y_prob >= t`), return the list of recalls and the list "
                 f"of precisions, as a tuple of two lists: lowering the threshold trades precision for recall.", setup,
                 f"{MET}precision_score, recall_score\n"
                 f"result = ([recall_score(y_te, (y_prob >= t).astype(int)) for t in {ts}], "
                 f"[precision_score(y_te, (y_prob >= t).astype(int), zero_division=0) for t in {ts}])",
                 require=["recall_score", "precision_score"], tol=1e-9)
    return P(f"{desc}. How many test samples are flagged positive at thresholds `{ts}` (`y_prob >= t`)? Return the "
             f"counts as a list, and the F1 at each threshold as a second list.", setup,
             f"{MET}f1_score\nresult = ([int((y_prob >= t).sum()) for t in {ts}], "
             f"[f1_score(y_te, (y_prob >= t).astype(int)) for t in {ts}])", require=["f1_score"], tol=1e-9)


@template(M, "easy")
def accuracy_trap(rng):
    dd = synth_clf(rng, n=(300, 600), d=(4, 8), classes=2, imbalance=rng.choice([0.03, 0.05, 0.08]))
    task = rng.choice(["dummy", "zeros", "report"])
    setup = dd.setup + ("\nfrom sklearn.model_selection import train_test_split\n"
                        "X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, random_state=0)")
    head = f"{dd.desc}; class 1 is rare. `X_tr, X_te, y_tr, y_te` is a stratified split."
    if task == "dummy":
        return P(f"{head} A model that always predicts the majority class looks great on accuracy. Fit "
                 f"`DummyClassifier(strategy='most_frequent')` (from `sklearn.dummy`) on the training split and return "
                 f"its test `(accuracy, recall, f1, balanced_accuracy)`.", setup,
                 f"from sklearn.dummy import DummyClassifier\n{MET}accuracy_score, recall_score, f1_score, balanced_accuracy_score\n"
                 "p = DummyClassifier(strategy='most_frequent').fit(X_tr, y_tr).predict(X_te)\n"
                 "result = (accuracy_score(y_te, p), recall_score(y_te, p), f1_score(y_te, p, zero_division=0), "
                 "balanced_accuracy_score(y_te, p))", require=["DummyClassifier", "balanced_accuracy_score"], tol=1e-9)
    if task == "zeros":
        return P(f"{head} Compare predicting all zeros with `{PIPE}` trained on the training split: return a dict "
                 f"{{'zeros': (accuracy, balanced_accuracy), 'model': (accuracy, balanced_accuracy)}} on the test split.",
                 setup,
                 "from sklearn.linear_model import LogisticRegression\nfrom sklearn.pipeline import make_pipeline\n"
                 f"from sklearn.preprocessing import StandardScaler\n{MET}accuracy_score, balanced_accuracy_score\n"
                 f"z = np.zeros_like(y_te)\nm = {PIPE}.fit(X_tr, y_tr).predict(X_te)\n"
                 "result = {'zeros': (accuracy_score(y_te, z), balanced_accuracy_score(y_te, z)),\n"
                 "          'model': (accuracy_score(y_te, m), balanced_accuracy_score(y_te, m))}",
                 require=["balanced_accuracy_score"], tol=1e-9)
    return P(f"{head} Train `{PIPE}` on the training split and return the test F1 of each class from "
             f"`classification_report(y_te, pred, output_dict=True)` as a dict {{'0': f1, '1': f1}}, plus the overall "
             f"accuracy, as `(dict, accuracy)`.", setup,
             "from sklearn.linear_model import LogisticRegression\nfrom sklearn.pipeline import make_pipeline\n"
             f"from sklearn.preprocessing import StandardScaler\n{MET}classification_report\n"
             f"pred = {PIPE}.fit(X_tr, y_tr).predict(X_te)\nrep = classification_report(y_te, pred, output_dict=True, zero_division=0)\n"
             "result = ({c: rep[c]['f1-score'] for c in ('0', '1')}, rep['accuracy'])",
             require=["classification_report", "output_dict"], tol=1e-9)


@template(M, "easy")
def averaging_modes(rng):
    setup, desc, k = _fitted_multi(rng)
    task = rng.choice(["macro_weighted", "micro_is_acc", "macro_by_hand", "per_class"])
    if task == "macro_weighted":
        return P(f"{desc}. Return the macro- and the weighted-averaged F1 of `y_pred` as a tuple (macro treats every "
                 f"class equally, weighted weighs them by support).", setup,
                 f"{MET}f1_score\nresult = (f1_score(y_te, y_pred, average='macro'), f1_score(y_te, y_pred, average='weighted'))",
                 require=["f1_score", "average"], tol=1e-9)
    if task == "micro_is_acc":
        return P(f"{desc}. For single-label multiclass data, micro-averaged precision, recall and F1 all equal the "
                 f"accuracy. Return `(micro_f1, micro_precision, accuracy)`.", setup,
                 f"{MET}f1_score, precision_score, accuracy_score\n"
                 "result = (f1_score(y_te, y_pred, average='micro'), precision_score(y_te, y_pred, average='micro'), "
                 "accuracy_score(y_te, y_pred))", require=["average", "accuracy_score"], tol=1e-9)
    if task == "macro_by_hand":
        return P(f"{desc}. Macro F1 is the plain mean of the per-class F1 scores, weighted F1 their support-weighted "
                 f"mean. Compute both from `f1_score(..., average=None)` and `np.bincount(y_te)` by hand and return "
                 f"them as a tuple.", setup,
                 f"{MET}f1_score\nf = f1_score(y_te, y_pred, average=None)\nw = np.bincount(y_te)\n"
                 "result = (f.mean(), (f * w).sum() / w.sum())", require=["f1_score"], tol=1e-9)
    return P(f"{desc}. Return the class with the lowest recall and its recall, as `(class, recall)`, using "
             f"`recall_score(..., average=None)`.", setup,
             f"{MET}recall_score\nr = recall_score(y_te, y_pred, average=None)\nresult = (int(r.argmin()), float(r.min()))",
             require=["recall_score"], tol=1e-9)


@template(M, "easy")
def roc_curve_points(rng):
    setup, desc = _fitted_binary(rng)
    f = rng.choice([0.05, 0.1, 0.2])
    task = rng.choice(["trapz", "tpr_at", "len", "shape"])
    roc = f"{MET}roc_curve\nfpr, tpr, thr = roc_curve(y_te, y_prob)\n"
    if task == "trapz":
        return P(f"{desc}. Compute `fpr, tpr, thr = roc_curve(y_te, y_prob)` and the area under it with the "
                 f"trapezoid rule (`np.trapezoid(tpr, fpr)`). Return `(your_area, roc_auc_score(y_te, y_prob))`.", setup,
                 roc + f"{MET}roc_auc_score\nresult = (float(np.trapezoid(tpr, fpr)), roc_auc_score(y_te, y_prob))",
                 require=["roc_curve", "roc_auc_score"], tol=1e-9)
    if task == "tpr_at":
        return P(f"{desc}. From `roc_curve(y_te, y_prob)`, return the best true-positive rate reachable with a "
                 f"false-positive rate of at most {f}, and the largest threshold that reaches it, as a tuple.", setup,
                 roc + f"ok = fpr <= {f}\nbest = tpr[ok].max()\nresult = (float(best), float(thr[ok & (tpr == best)].max()))",
                 require=["roc_curve"], tol=1e-9)
    if task == "len":
        return P(f"{desc}. `roc_curve` drops thresholds that don't change the curve's shape unless you pass "
                 f"`drop_intermediate=False`. Return the number of thresholds with and without dropping, as "
                 f"`(default, all)`.", setup,
                 f"{MET}roc_curve\nresult = (len(roc_curve(y_te, y_prob)[2]), len(roc_curve(y_te, y_prob, drop_intermediate=False)[2]))",
                 require=["roc_curve", "drop_intermediate"])
    return P(f"{desc}. `fpr, tpr, thr = roc_curve(y_te, y_prob)` starts at the point (0, 0) with threshold `inf`. "
             f"Return `(fpr[:3], tpr[:3], thr[1])`.", setup, roc + "result = (fpr[:3], tpr[:3], float(thr[1]))",
             require=["roc_curve"], tol=1e-9)


@template(M, "easy")
def confusion_reading(rng):
    setup, desc, k = _fitted_multi(rng)
    task = rng.choice(["recall_diag", "precision_cols", "confused_pair"])
    cm = f"{MET}confusion_matrix\nC = confusion_matrix(y_te, y_pred)\n"
    if task == "recall_diag":
        return P(f"{desc}. With `C = confusion_matrix(y_te, y_pred)` (rows = true class), each class's recall is its "
                 f"diagonal entry divided by its row sum. Return `(by_hand, recall_score(y_te, y_pred, average=None))`.",
                 setup, cm + f"{MET}recall_score\nresult = (np.diag(C) / C.sum(axis=1), recall_score(y_te, y_pred, average=None))",
                 require=["confusion_matrix"], tol=1e-9)
    if task == "precision_cols":
        return P(f"{desc}. Return the diagonal of `confusion_matrix(y_te, y_pred, normalize='pred')`: the precision of "
                 f"each class (columns are normalized).", setup,
                 f"{MET}confusion_matrix\nresult = np.diag(confusion_matrix(y_te, y_pred, normalize='pred'))",
                 require=["confusion_matrix", "normalize"], tol=1e-9)
    return P(f"{desc}. Which mistake is most common? From `C = confusion_matrix(y_te, y_pred)`, return `(true class, "
             f"predicted class, count)` of the largest off-diagonal entry (the first in row-major order on ties).", setup,
             cm + "D = C.copy()\nnp.fill_diagonal(D, -1)\ni, j = np.unravel_index(D.argmax(), D.shape)\n"
                  "result = (int(i), int(j), int(C[i, j]))", require=["confusion_matrix"])


@template(M, "easy")
def logloss_brier_by_hand(rng):
    setup, desc, n = _labels(rng, n=(16, 26))
    task = rng.choice(["logloss", "brier", "prior"])
    if task == "logloss":
        return P(f"{desc}. Compute the log loss **by hand**: the mean over samples of −ln(probability given to the "
                 f"true class), i.e. `y_prob` for positives and `1 − y_prob` for negatives. Return `(by_hand, "
                 f"log_loss(y_true, y_prob))`.", setup,
                 f"{MET}log_loss\np = np.where(y_true == 1, y_prob, 1 - y_prob)\nresult = (float(-np.log(p).mean()), log_loss(y_true, y_prob))",
                 require=["log_loss"], tol=1e-9)
    if task == "brier":
        return P(f"{desc}. Compute the Brier score by hand (the mean of `(y_prob − y_true)²`) and return `(by_hand, "
                 f"brier_score_loss(y_true, y_prob))`.", setup,
                 f"{MET}brier_score_loss\nresult = (float(((y_prob - y_true) ** 2).mean()), brier_score_loss(y_true, y_prob))",
                 require=["brier_score_loss"], tol=1e-9)
    return P(f"{desc}. A model with no information predicts the base rate `y_true.mean()` for everyone. Return the log "
             f"loss and Brier score of that constant prediction, and of `y_prob`, as `((const_ll, const_brier), "
             f"(model_ll, model_brier))`.", setup,
             f"{MET}log_loss, brier_score_loss\nc = np.full(len(y_true), y_true.mean())\n"
             "result = ((log_loss(y_true, c), brier_score_loss(y_true, c)), (log_loss(y_true, y_prob), brier_score_loss(y_true, y_prob)))",
             require=["log_loss", "brier_score_loss"], tol=1e-9)


@template(M, "easy")
def agreement_scores(rng):
    setup, desc, n = _multi(rng)
    task = rng.choice(["kappa", "mcc", "weighted_kappa"])
    if task == "kappa":
        return P(f"{desc}. Cohen's kappa compares the observed agreement `p_o` (accuracy) with the agreement expected by "
                 f"chance `p_e = Σ_k (row_k / n)·(col_k / n)` from the confusion matrix's marginals: `κ = (p_o − p_e) / "
                 f"(1 − p_e)`. Compute it by hand and return `(by_hand, cohen_kappa_score(y_true, y_pred))`.", setup,
                 f"{MET}confusion_matrix, cohen_kappa_score\nC = confusion_matrix(y_true, y_pred)\nn = C.sum()\n"
                 "po = np.trace(C) / n\npe = (C.sum(axis=1) * C.sum(axis=0)).sum() / n ** 2\n"
                 "result = ((po - pe) / (1 - pe), cohen_kappa_score(y_true, y_pred))", require=["cohen_kappa_score"],
                 tol=1e-9)
    if task == "mcc":
        return P(f"{desc}. Return `(matthews_corrcoef(y_true, y_pred), cohen_kappa_score(y_true, y_pred), "
                 f"accuracy_score(y_true, y_pred))`: three views of the same predictions.", setup,
                 f"{MET}matthews_corrcoef, cohen_kappa_score, accuracy_score\n"
                 "result = (matthews_corrcoef(y_true, y_pred), cohen_kappa_score(y_true, y_pred), accuracy_score(y_true, y_pred))",
                 require=["matthews_corrcoef", "cohen_kappa_score"], tol=1e-9)
    return P(f"{desc}. If the classes are ordered (say ratings 0 < 1 < 2), confusing 0 with 2 is worse than 0 with 1. "
             f"Return Cohen's kappa unweighted and with `weights='linear'` and `weights='quadratic'`, as a 3-tuple.",
             setup, f"{MET}cohen_kappa_score\nresult = tuple(cohen_kappa_score(y_true, y_pred, weights=w) for w in (None, 'linear', 'quadratic'))",
             require=["cohen_kappa_score", "weights"], tol=1e-9)


@template(M, "easy")
def report_dict_tasks(rng):
    setup, desc, k = _fitted_multi(rng)
    rep = f"{MET}classification_report\nrep = classification_report(y_te, y_pred, output_dict=True)\n"
    items = [
        ("classification_report", "Return the macro-averaged F1 (`rep['macro avg']['f1-score']`).",
         rep + "result = rep['macro avg']['f1-score']"),
        ("classification_report", "Return the weighted-average precision.", rep + "result = rep['weighted avg']['precision']"),
        ("classification_report", "Return a dict {class label (as a string): support} for every class.",
         rep + "result = {c: v['support'] for c, v in rep.items() if c not in ('accuracy', 'macro avg', 'weighted avg')}"),
        ("classification_report", "Return the label (a string key) of the class with the lowest recall.",
         rep + "cls = {c: v['recall'] for c, v in rep.items() if c not in ('accuracy', 'macro avg', 'weighted avg')}\n"
               "result = min(cls, key=cls.get)"),
        ("classification_report", "Return `rep['accuracy']`: note it's a bare number, not a dict like the other entries.",
         rep + "result = rep['accuracy']"),
        ("DataFrame", "Turn the report into a table: `pd.DataFrame(rep).T` (one row per class plus the averages). Return "
                      "it without the `'accuracy'` row, rounded to 3 decimals.",
         rep + "result = pd.DataFrame(rep).T.drop(index='accuracy').round(3)"),
    ]
    return catalog(rng, items, setup, f"{desc}. Let `rep = classification_report(y_te, y_pred, output_dict=True)`.",
                   tol=1e-9)


# ================================================================= medium

@template(M, "medium")
def best_f1_threshold(rng):
    setup, desc = _fitted_binary(rng)
    r = rng.choice([0.8, 0.9, 0.95])
    task = rng.choice(["threshold", "f1", "precision_at", "recall_at"])
    pr = (f"{MET}precision_recall_curve\nprec, rec, thr = precision_recall_curve(y_te, y_prob)\n"
          "f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)\n")
    head = (f"{desc}. `prec, rec, thr = precision_recall_curve(y_te, y_prob)` lists precision and recall for every "
            f"threshold (`prec` and `rec` have one extra final point (1, 0) with no threshold).")
    if task == "threshold":
        return P(f"{head} Compute F1 = 2·p·r / (p + r) at every threshold and return the threshold with the highest F1 "
                 f"(the first on ties).", setup, pr + "result = float(thr[np.argmax(f1)])",
                 require=["precision_recall_curve"], tol=1e-9)
    if task == "f1":
        return P(f"{head} Return `(best F1 over the thresholds, F1 at the default threshold 0.5)`; the second from "
                 f"`f1_score(y_te, y_pred)`.", setup,
                 pr + f"{MET}f1_score\nresult = (float(f1.max()), f1_score(y_te, y_pred))",
                 require=["precision_recall_curve", "f1_score"], tol=1e-9)
    if task == "precision_at":
        return P(f"{head} What's the best precision you can get while keeping recall ≥ {r}? Return it with the threshold "
                 f"that gives it (the largest such threshold on ties).", setup,
                 f"{MET}precision_recall_curve\nprec, rec, thr = precision_recall_curve(y_te, y_prob)\n"
                 f"ok = rec[:-1] >= {r}\nbest = prec[:-1][ok].max()\n"
                 "result = (float(best), float(thr[ok & (prec[:-1] == best)].max()))",
                 require=["precision_recall_curve"], tol=1e-9)
    return P(f"{head} Return the recall you get at the **smallest** threshold whose precision is at least {r} (or 0.0 "
             f"if no threshold reaches it), and that threshold (`None` if none).", setup,
             f"{MET}precision_recall_curve\nprec, rec, thr = precision_recall_curve(y_te, y_prob)\n"
             f"ok = np.flatnonzero(prec[:-1] >= {r})\n"
             "result = (float(rec[ok[0]]), float(thr[ok[0]])) if len(ok) else (0.0, None)",
             require=["precision_recall_curve"], tol=1e-9)


@template(M, "medium")
def roc_operating_points(rng):
    setup, desc = _fitted_binary(rng)
    f = rng.choice([0.05, 0.1])
    task = rng.choice(["youden", "budget", "counts"])
    roc = f"{MET}roc_curve\nfpr, tpr, thr = roc_curve(y_te, y_prob)\n"
    if task == "youden":
        return P(f"{desc}. Pick an operating point on the ROC curve with Youden's J = TPR − FPR: return the threshold "
                 f"maximizing J (the first on ties) and the J value, from `roc_curve(y_te, y_prob)`.", setup,
                 roc + "j = tpr - fpr\ni = int(np.argmax(j))\nresult = (float(thr[i]), float(j[i]))",
                 require=["roc_curve"], tol=1e-9)
    if task == "budget":
        return P(f"{desc}. The team can afford a false-positive rate of {f}. Using `roc_curve(y_te, y_prob)`, apply the "
                 f"smallest threshold whose FPR is at most {f} and return the resulting test `(precision, recall)` "
                 f"(predict 1 when `y_prob >= threshold`).", setup,
                 roc + f"{MET}precision_score, recall_score\nt = thr[fpr <= {f}].min()\npred = (y_prob >= t).astype(int)\n"
                       "result = (precision_score(y_te, pred), recall_score(y_te, pred))", require=["roc_curve"], tol=1e-9)
    return P(f"{desc}. Check the ROC curve against raw counts: for the threshold `thr[i]` with `i = len(thr) // 2` from "
             f"`roc_curve(y_te, y_prob)`, count by hand the true- and false-positive rates of `y_prob >= thr[i]` and "
             f"return `((tpr_by_hand, fpr_by_hand), (tpr[i], fpr[i]))`.", setup,
             roc + "i = len(thr) // 2\np = y_prob >= thr[i]\n"
                   "result = ((float(p[y_te == 1].mean()), float(p[y_te == 0].mean())), (float(tpr[i]), float(fpr[i])))",
             require=["roc_curve"], tol=1e-9)


@template(M, "medium")
def compare_models_metrics(rng):
    dd = synth_clf(rng, n=(300, 500), d=(5, 10), classes=2, imbalance=rng.choice([None, 0.2]))
    task = rng.choice(["auc", "ap_rank", "logloss", "table"])
    setup = dd.setup + ("\nfrom sklearn.model_selection import train_test_split\n"
                        "X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, random_state=0)")
    models = ("from sklearn.ensemble import RandomForestClassifier\nfrom sklearn.linear_model import LogisticRegression\n"
              "from sklearn.naive_bayes import GaussianNB\n"
              "models = {'lr': LogisticRegression(max_iter=1000), 'rf': RandomForestClassifier(n_estimators=50, random_state=0),\n"
              "          'nb': GaussianNB()}\nprob = {k: m.fit(X_tr, y_tr).predict_proba(X_te)[:, 1] for k, m in models.items()}\n")
    head = (f"{dd.desc}; `X_tr, X_te, y_tr, y_te` is a stratified split. Train `'lr'`: `LogisticRegression(max_iter=1000)`, "
            f"`'rf'`: `RandomForestClassifier(n_estimators=50, random_state=0)` and `'nb'`: `GaussianNB()` on the training "
            f"split and get each one's test P(class 1).")
    if task == "auc":
        return P(f"{head} Return a dict {{name: ROC AUC}}.", setup,
                 models + f"{MET}roc_auc_score\nresult = {{k: roc_auc_score(y_te, p) for k, p in prob.items()}}",
                 require=["roc_auc_score", "predict_proba"], tol=1e-9)
    if task == "ap_rank":
        return P(f"{head} Rank the models by average precision, best first, and return the list of names.", setup,
                 models + f"{MET}average_precision_score\nap = {{k: average_precision_score(y_te, p) for k, p in prob.items()}}\n"
                          "result = sorted(ap, key=ap.get, reverse=True)", require=["average_precision_score"])
    if task == "logloss":
        return P(f"{head} Which model has the best (lowest) log loss, and which the best Brier score? Return the two "
                 f"names as a tuple: these reward well-calibrated probabilities, not just good ranking.", setup,
                 models + f"{MET}log_loss, brier_score_loss\nll = {{k: log_loss(y_te, p) for k, p in prob.items()}}\n"
                          "bs = {k: brier_score_loss(y_te, p) for k, p in prob.items()}\nresult = (min(ll, key=ll.get), min(bs, key=bs.get))",
                 require=["log_loss", "brier_score_loss"])
    return P(f"{head} Return a DataFrame indexed by model name with columns `'auc'`, `'ap'` and `'f1'` (F1 at the 0.5 "
             f"threshold, `p >= 0.5`), rounded to 4 decimals.", setup,
             models + f"{MET}roc_auc_score, average_precision_score, f1_score\n"
                      "result = pd.DataFrame({k: {'auc': roc_auc_score(y_te, p), 'ap': average_precision_score(y_te, p),\n"
                      "                           'f1': f1_score(y_te, (p >= 0.5).astype(int))} for k, p in prob.items()}).T.round(4)",
             require=["roc_auc_score", "average_precision_score"], tol=1e-9)


@template(M, "medium")
def cost_sensitive_threshold(rng):
    setup, desc = _fitted_binary(rng)
    cfp, cfn = rng.choice([(1, 5), (1, 10), (2, 3), (5, 1)])
    task = rng.choice(["best", "compare", "curve"])
    sol = (f"{MET}confusion_matrix\ndef cost(t):\n    tn, fp, fn, tp = confusion_matrix(y_te, (y_prob >= t).astype(int), labels=[0, 1]).ravel()\n"
           f"    return {cfp} * fp + {cfn} * fn\nts = np.round(np.arange(0.05, 1.0, 0.05), 2)\ncosts = [cost(t) for t in ts]\n")
    head = (f"{desc}. A false positive costs {cfp} and a false negative costs {cfn}. For thresholds `t` in "
            f"`np.round(np.arange(0.05, 1.0, 0.05), 2)`, predict 1 when `y_prob >= t` and compute the total cost from "
            f"`confusion_matrix(..., labels=[0, 1]).ravel()`.")
    if task == "best":
        return P(f"{head} Return `(best threshold, its cost)` (the smallest threshold on ties).", setup,
                 sol + "i = int(np.argmin(costs))\nresult = (float(ts[i]), int(costs[i]))", require=["confusion_matrix"],
                 tol=1e-9)
    if task == "compare":
        return P(f"{head} Return `(cost at 0.5, lowest cost)`.", setup,
                 sol + "result = (int(cost(0.5)), int(min(costs)))", require=["confusion_matrix"])
    return P(f"{head} Return the list of costs.", setup, sol + "result = [int(c) for c in costs]",
             require=["confusion_matrix"])


@template(M, "medium")
def multiclass_auc_variants(rng):
    setup, desc, k = _fitted_multi(rng)
    task = rng.choice(["averages", "ovo", "per_class", "topk"])
    if task == "averages":
        return P(f"{desc}. Return the one-vs-rest ROC AUC of `proba` with macro and with weighted averaging "
                 f"(`multi_class='ovr', average=...`), as a tuple.", setup,
                 f"{MET}roc_auc_score\nresult = (roc_auc_score(y_te, proba, multi_class='ovr', average='macro'), "
                 "roc_auc_score(y_te, proba, multi_class='ovr', average='weighted'))", require=["multi_class"], tol=1e-9)
    if task == "ovo":
        return P(f"{desc}. Return the multiclass ROC AUC of `proba` computed one-vs-rest (`'ovr'`) and one-vs-one "
                 f"(`'ovo'`: averaged over all pairs of classes), as a tuple.", setup,
                 f"{MET}roc_auc_score\nresult = (roc_auc_score(y_te, proba, multi_class='ovr'), "
                 "roc_auc_score(y_te, proba, multi_class='ovo'))", require=["multi_class"], tol=1e-9)
    if task == "per_class":
        return P(f"{desc}. Compute each class's one-vs-rest AUC by hand: `roc_auc_score(y_te == c, proba[:, c])` for "
                 f"every class c. Return the array of per-class AUCs and their mean, which equals the macro `ovr` AUC, "
                 f"as a tuple.", setup,
                 f"{MET}roc_auc_score\na = np.array([roc_auc_score(y_te == c, proba[:, c]) for c in range(proba.shape[1])])\n"
                 "result = (a, a.mean())", require=["roc_auc_score"], tol=1e-9)
    ks = [1, 2, 3]
    return P(f"{desc}. Return the top-k accuracy of `proba` for k = 1, 2, 3 (`top_k_accuracy_score`), as a list; top-1 "
             f"equals the plain accuracy.", setup,
             f"{MET}top_k_accuracy_score\nresult = [top_k_accuracy_score(y_te, proba, k=k, labels=np.arange(proba.shape[1])) for k in {ks}]",
             require=["top_k_accuracy_score"], tol=1e-9)


@template(M, "medium")
def reliability_bins(rng):
    setup, desc = _fitted_binary(rng)
    nb = rng.choice([4, 5])
    task = rng.choice(["table", "ece", "models"])
    sol = (f"edges = np.linspace(0, 1, {nb + 1})\nb = np.clip(np.digitize(y_prob, edges[1:-1]), 0, {nb - 1})\n"
           f"rows = [(int(k), float(y_prob[b == k].mean()), float(y_te[b == k].mean()), int((b == k).sum())) "
           f"for k in range({nb}) if (b == k).any()]\n")
    head = (f"{desc}. Check calibration with a reliability table: split `[0, 1]` into {nb} equal-width bins "
            f"(`np.digitize(y_prob, np.linspace(0, 1, {nb + 1})[1:-1])`), and for each non-empty bin compare the mean "
            f"predicted probability with the observed share of positives.")
    if task == "table":
        return P(f"{head} Return a list of `(bin, mean_predicted, observed_rate, count)` tuples for the non-empty bins.",
                 setup, sol + "result = rows", require=["digitize"], tol=1e-9)
    if task == "ece":
        return P(f"{head} Return the *expected calibration error*: Σ over bins of (count / n)·|mean_predicted − "
                 f"observed_rate|, and the Brier score of `y_prob` (`brier_score_loss`), as a tuple.", setup,
                 sol + f"{MET}brier_score_loss\nn = len(y_prob)\n"
                       "result = (sum(c / n * abs(m - o) for _, m, o, c in rows), brier_score_loss(y_te, y_prob))",
                 require=["brier_score_loss"], tol=1e-9)
    return P(f"{desc}. Compare probability quality on the test split: return the Brier score and log loss of `y_prob` "
             f"and of `GaussianNB()` trained on `X_tr, y_tr`, as `((lr_brier, lr_logloss), (nb_brier, nb_logloss))`: "
             f"naive Bayes tends to be over-confident.", setup,
             f"from sklearn.naive_bayes import GaussianNB\n{MET}brier_score_loss, log_loss\n"
             "q = GaussianNB().fit(X_tr, y_tr).predict_proba(X_te)[:, 1]\n"
             "result = ((brier_score_loss(y_te, y_prob), log_loss(y_te, y_prob)), (brier_score_loss(y_te, q), log_loss(y_te, q)))",
             require=["brier_score_loss", "log_loss"], tol=1e-6)


@template(M, "medium")
def cv_scoring_strings(rng):
    dd = synth_clf(rng, n=(200, 400), d=(4, 8), classes=2, imbalance=rng.choice([None, 0.25]))
    task = rng.choice(["strings", "cross_validate", "make_scorer", "proba_scorer"])
    imp = ("from sklearn.linear_model import LogisticRegression\nfrom sklearn.pipeline import make_pipeline\n"
           f"from sklearn.preprocessing import StandardScaler\nclf = {PIPE}\n")
    head = f"{dd.desc}. Evaluate `clf = {PIPE}` with 5-fold cross-validation (from `sklearn.model_selection`)."
    if task == "strings":
        sc = ["accuracy", "f1", "roc_auc", "balanced_accuracy"]
        return P(f"{head} Return a dict {{scoring: mean score}} for `scoring` in `{sc}` (`cross_val_score(clf, X, y, "
                 f"cv=5, scoring=...)`).", dd.setup,
                 imp + "from sklearn.model_selection import cross_val_score\n"
                       f"result = {{s: cross_val_score(clf, X, y, cv=5, scoring=s).mean() for s in {sc}}}",
                 require=["cross_val_score", "scoring"], tol=1e-9)
    if task == "cross_validate":
        return P(f"{head} Use `cross_validate(clf, X, y, cv=5, scoring=['precision', 'recall', 'neg_log_loss'])` and "
                 f"return a dict of the mean test scores `{{'precision': ..., 'recall': ..., 'log_loss': ...}}` with the "
                 f"log loss made positive again.", dd.setup,
                 imp + "from sklearn.model_selection import cross_validate\n"
                       "cv = cross_validate(clf, X, y, cv=5, scoring=['precision', 'recall', 'neg_log_loss'])\n"
                       "result = {'precision': cv['test_precision'].mean(), 'recall': cv['test_recall'].mean(), "
                       "'log_loss': -cv['test_neg_log_loss'].mean()}", require=["cross_validate"], tol=1e-9)
    if task == "make_scorer":
        return P(f"{head} There is no built-in F2 scorer: build one with `make_scorer(fbeta_score, beta=2)` (both from "
                 f"`sklearn.metrics`) and return the 5 fold scores from `cross_val_score(clf, X, y, cv=5, "
                 f"scoring=f2)`.", dd.setup,
                 imp + f"from sklearn.model_selection import cross_val_score\n{MET}make_scorer, fbeta_score\n"
                       "f2 = make_scorer(fbeta_score, beta=2)\nresult = cross_val_score(clf, X, y, cv=5, scoring=f2)",
                 require=["make_scorer", "fbeta_score"], tol=1e-9)
    return P(f"{head} Collect out-of-fold probabilities with `cross_val_predict(clf, X, y, cv=5, method='predict_proba')` "
             f"and return `(roc_auc_score on them, the mean of cross_val_score(..., scoring='roc_auc'))`: pooling "
             f"predictions and averaging per-fold scores give slightly different numbers.", dd.setup,
             imp + f"from sklearn.model_selection import cross_val_predict, cross_val_score\n{MET}roc_auc_score\n"
                   "p = cross_val_predict(clf, X, y, cv=5, method='predict_proba')[:, 1]\n"
                   "result = (roc_auc_score(y, p), cross_val_score(clf, X, y, cv=5, scoring='roc_auc').mean())",
             require=["cross_val_predict", "roc_auc_score"], tol=1e-9)


@template(M, "medium")
def metric_tasks_themed(rng):
    t = table(rng, kind="clf", numeric_only=True, n=(250, 450))
    s = seeded(rng)
    pct = rng.choice([10, 20])
    setup = (t.setup + "\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.pipeline import make_pipeline\n"
             "from sklearn.preprocessing import StandardScaler\nfrom sklearn.linear_model import LogisticRegression\n"
             f"X_tr, X_te, y_tr, y_te = train_test_split(df, y, stratify=y, random_state={s})\n"
             f"clf = {PIPE}.fit(X_tr, y_tr)\ny_pred = clf.predict(X_te)\ny_prob = clf.predict_proba(X_te)[:, 1]")
    items = [
        ("confusion_matrix", "Return the confusion matrix as a DataFrame with index `['actual 0', 'actual 1']` and "
                             "columns `['pred 0', 'pred 1']`.",
         f"{MET}confusion_matrix\nresult = pd.DataFrame(confusion_matrix(y_te, y_pred), index=['actual 0', 'actual 1'], "
         "columns=['pred 0', 'pred 1'])"),
        ("recall_score", "Of the actual positives in the test split, what share does the model catch? Return the recall.",
         f"{MET}recall_score\nresult = recall_score(y_te, y_pred)"),
        ("precision_score", f"The team can only review the top {pct}% highest-scored test cases. Return the precision "
                            f"among them (the share of true positives in the top `ceil({pct / 100} · n)` by `y_prob`).",
         f"k = int(np.ceil({pct / 100} * len(y_prob)))\ntop = np.argsort(y_prob)[::-1][:k]\nresult = float(y_te[top].mean())",
         {"require": ["argsort"]}),
        ("argsort", f"Return the *lift* of the top {pct}%: the precision among the top `ceil({pct / 100} · n)` "
                    f"highest-scored cases divided by the overall positive rate of the test split.",
         f"k = int(np.ceil({pct / 100} * len(y_prob)))\ntop = np.argsort(y_prob)[::-1][:k]\nresult = float(y_te[top].mean() / y_te.mean())"),
        ("f1_score", "Return the test F1 and balanced accuracy as a tuple.",
         f"{MET}f1_score, balanced_accuracy_score\nresult = (f1_score(y_te, y_pred), balanced_accuracy_score(y_te, y_pred))"),
        ("average_precision_score", "Return the test ROC AUC and average precision as a tuple.",
         f"{MET}roc_auc_score, average_precision_score\nresult = (roc_auc_score(y_te, y_prob), average_precision_score(y_te, y_prob))"),
        ("matthews_corrcoef", "Return the test Matthews correlation coefficient.",
         f"{MET}matthews_corrcoef\nresult = matthews_corrcoef(y_te, y_pred)"),
    ]
    return catalog(rng, items, setup, f"{t.desc}. `clf = {PIPE}` was trained on a stratified split (`X_tr, X_te, y_tr, "
                                      f"y_te`); `y_pred` and `y_prob` (P(class 1)) are its test predictions.", tol=1e-9)


@template(M, "medium")
def per_class_report(rng):
    setup, desc, k = _fitted_multi(rng)
    task = rng.choice(["worst_f1", "support", "frame"])
    if task == "worst_f1":
        return P(f"{desc}. Use `precision_recall_fscore_support(y_te, y_pred, average=None)` and return `(class with the "
                 f"lowest F1, its precision, its recall)`.", setup,
                 f"{MET}precision_recall_fscore_support\np, r, f, s = precision_recall_fscore_support(y_te, y_pred, average=None)\n"
                 "c = int(f.argmin())\nresult = (c, float(p[c]), float(r[c]))", require=["precision_recall_fscore_support"],
                 tol=1e-9)
    if task == "support":
        return P(f"{desc}. Return the support array from `precision_recall_fscore_support(y_te, y_pred, average=None)` "
                 f"and check it equals `np.bincount(y_te)`: return both as a tuple.", setup,
                 f"{MET}precision_recall_fscore_support\n"
                 "result = (precision_recall_fscore_support(y_te, y_pred, average=None)[3], np.bincount(y_te))",
                 require=["precision_recall_fscore_support"])
    return P(f"{desc}. Return a DataFrame with one row per class and columns `'precision'`, `'recall'`, `'f1'`, "
             f"`'support'` built from `precision_recall_fscore_support(y_te, y_pred, average=None)`, sorted by `'f1'` "
             f"ascending.", setup,
             f"{MET}precision_recall_fscore_support\np, r, f, s = precision_recall_fscore_support(y_te, y_pred, average=None)\n"
             "result = pd.DataFrame({'precision': p, 'recall': r, 'f1': f, 'support': s}).sort_values('f1')",
             require=["precision_recall_fscore_support"], tol=1e-9)


@template(M, "medium")
def multilabel_tasks(rng):
    n, L, s = rng.randint(200, 350), rng.randint(3, 5), seeded(rng)
    setup = (f"from sklearn.datasets import make_multilabel_classification\nfrom sklearn.model_selection import train_test_split\n"
             f"X, Y = make_multilabel_classification(n_samples={n}, n_features=8, n_classes={L}, random_state={s})\n"
             "X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, random_state=0)\n"
             "from sklearn.multioutput import MultiOutputClassifier\nfrom sklearn.linear_model import LogisticRegression\n"
             "model = MultiOutputClassifier(LogisticRegression(max_iter=1000)).fit(X_tr, Y_tr)\nY_pred = model.predict(X_te)")
    desc = (f"`X, Y` is a multilabel problem ({n} samples, {L} labels; `Y` is a 0/1 indicator matrix). `model = "
            f"MultiOutputClassifier(LogisticRegression(max_iter=1000))` was trained on a split; `Y_pred` holds its test "
            f"predictions for `Y_te`")
    task = rng.choice(["summary", "per_label", "subset", "jaccard_by_hand"])
    if task == "summary":
        return P(f"{desc}. Return `(hamming_loss, jaccard_score(..., average='samples'), f1_score(..., average='micro'))` "
                 f"on the test split.", setup,
                 f"{MET}hamming_loss, jaccard_score, f1_score\n"
                 "result = (hamming_loss(Y_te, Y_pred), jaccard_score(Y_te, Y_pred, average='samples', zero_division=1), "
                 "f1_score(Y_te, Y_pred, average='micro'))", require=["hamming_loss", "jaccard_score"], tol=1e-9)
    if task == "per_label":
        return P(f"{desc}. Return the F1 of every label (`average=None`) and the index of the worst label, as a tuple.",
                 setup, f"{MET}f1_score\nf = f1_score(Y_te, Y_pred, average=None)\nresult = (f, int(f.argmin()))",
                 require=["f1_score"], tol=1e-9)
    if task == "subset":
        return P(f"{desc}. Return `(subset accuracy, 1 − hamming_loss)`: exact-match is far stricter than counting "
                 f"individual label decisions.", setup,
                 f"{MET}accuracy_score, hamming_loss\nresult = (accuracy_score(Y_te, Y_pred), 1 - hamming_loss(Y_te, Y_pred))",
                 require=["accuracy_score", "hamming_loss"], tol=1e-9)
    return P(f"{desc}. Compute the sample-averaged Jaccard score by hand: for each row |true ∩ predicted| / |true ∪ "
             f"predicted|, counting a row where both are empty as 1, then average. Return `(by_hand, "
             f"jaccard_score(Y_te, Y_pred, average='samples', zero_division=1))`.", setup,
             f"{MET}jaccard_score\ninter = (Y_te & Y_pred).sum(axis=1)\nunion = (Y_te | Y_pred).sum(axis=1)\n"
             "mine = np.where(union == 0, 1.0, inter / np.maximum(union, 1)).mean()\n"
             "result = (float(mine), jaccard_score(Y_te, Y_pred, average='samples', zero_division=1))",
             require=["jaccard_score"], tol=1e-9)


@template(M, "medium")
def pr_vs_roc_imbalance(rng):
    rate = rng.choice([0.03, 0.05, 0.1])
    setup, desc = _fitted_binary(rng, imbalance=rate)
    task = rng.choice(["both", "baselines", "random_scores"])
    if task == "both":
        return P(f"{desc}. On rare positives, ROC AUC can look comfortable while average precision is sobering. Return "
                 f"`(roc_auc_score, average_precision_score, positive rate of y_te)` for `y_prob`.", setup,
                 f"{MET}roc_auc_score, average_precision_score\n"
                 "result = (roc_auc_score(y_te, y_prob), average_precision_score(y_te, y_prob), float(y_te.mean()))",
                 require=["roc_auc_score", "average_precision_score"], tol=1e-9)
    if task == "baselines":
        return P(f"{desc}. A random scorer gets ROC AUC 0.5 but average precision ≈ the positive rate. Return each "
                 f"metric's lift over its random baseline: `(roc_auc − 0.5, average_precision − y_te.mean())`.", setup,
                 f"{MET}roc_auc_score, average_precision_score\n"
                 "result = (roc_auc_score(y_te, y_prob) - 0.5, average_precision_score(y_te, y_prob) - y_te.mean())",
                 require=["roc_auc_score", "average_precision_score"], tol=1e-9)
    s = seeded(rng)
    return P(f"{desc}. Score the test split with random numbers `r = np.random.default_rng({s}).random(len(y_te))` and "
             f"return `(roc_auc_score(y_te, r), average_precision_score(y_te, r))`.", setup,
             f"{MET}roc_auc_score, average_precision_score\nr = np.random.default_rng({s}).random(len(y_te))\n"
             "result = (roc_auc_score(y_te, r), average_precision_score(y_te, r))",
             require=["roc_auc_score", "average_precision_score"], tol=1e-9)


# ================================================================= hard

@template(M, "hard")
def auc_by_hand(rng):
    setup, desc, n = _labels(rng, n=(18, 30))
    task = rng.choice(["pairs", "ranks", "ties"])
    if task == "pairs":
        return P(f"{desc}. ROC AUC is the probability that a random positive gets a higher score than a random "
                 f"negative. Compute it **by hand** over all positive-negative pairs (a tie counts ½) and return "
                 f"`(by_hand, roc_auc_score(y_true, y_prob))`.", setup,
                 f"{MET}roc_auc_score\npos, neg = y_prob[y_true == 1], y_prob[y_true == 0]\n"
                 "d = pos[:, None] - neg[None, :]\nresult = (float(((d > 0) + 0.5 * (d == 0)).mean()), roc_auc_score(y_true, y_prob))",
                 require=["roc_auc_score"], tol=1e-9)
    if task == "ranks":
        return P(f"{desc}. The rank formula avoids building all pairs: with `r` = the ranks of `y_prob` (1 = lowest; "
                 f"`scipy.stats.rankdata` averages ties), AUC = (Σ r over positives − P(P+1)/2) / (P·N). Return "
                 f"`(by_hand, roc_auc_score(y_true, y_prob))`.", setup,
                 f"from scipy.stats import rankdata\n{MET}roc_auc_score\nr = rankdata(y_prob)\n"
                 "P_, N_ = int((y_true == 1).sum()), int((y_true == 0).sum())\n"
                 "result = (float((r[y_true == 1].sum() - P_ * (P_ + 1) / 2) / (P_ * N_)), roc_auc_score(y_true, y_prob))",
                 require=["rankdata"], tol=1e-9)
    return P(f"{desc}. Round the scores to one decimal (`np.round(y_prob, 1)`) so that many tie. Return the AUC of the "
             f"rounded scores by hand over all pairs (a tie counts ½), next to `roc_auc_score` of the rounded scores and "
             f"of the original ones, as a 3-tuple.", setup,
             f"{MET}roc_auc_score\ns = np.round(y_prob, 1)\nd = s[y_true == 1][:, None] - s[y_true == 0][None, :]\n"
             "result = (float(((d > 0) + 0.5 * (d == 0)).mean()), roc_auc_score(y_true, s), roc_auc_score(y_true, y_prob))",
             require=["roc_auc_score"], tol=1e-9)


@template(M, "hard")
def average_precision_by_hand(rng):
    setup, desc, n = _labels(rng, n=(18, 30))
    task = rng.choice(["from_curve", "from_sort", "not_trapz"])
    if task == "from_curve":
        return P(f"{desc}. Average precision is the step-wise area `Σ_n (R_n − R_{{n−1}})·P_n` over the thresholds, from "
                 f"high to low. Compute it from `prec, rec, _ = precision_recall_curve(y_true, y_prob)` (recall is "
                 f"decreasing along the arrays) and return `(by_hand, average_precision_score(y_true, y_prob))`.", setup,
                 f"{MET}precision_recall_curve, average_precision_score\nprec, rec, _ = precision_recall_curve(y_true, y_prob)\n"
                 "result = (float(-np.sum(np.diff(rec) * prec[:-1])), average_precision_score(y_true, y_prob))",
                 require=["precision_recall_curve"], tol=1e-9)
    if task == "from_sort":
        return P(f"{desc}. With distinct scores, average precision is the mean, over the positives, of the precision "
                 f"among the samples scored at least as high as that positive. Sort by `y_prob` descending, compute it "
                 f"by hand and return `(by_hand, average_precision_score(y_true, y_prob))`.", setup,
                 f"{MET}average_precision_score\no = np.argsort(-y_prob)\nhit = y_true[o]\n"
                 "prec_at = np.cumsum(hit) / np.arange(1, len(hit) + 1)\n"
                 "result = (float(prec_at[hit == 1].mean()), average_precision_score(y_true, y_prob))",
                 require=["average_precision_score"], tol=1e-9)
    return P(f"{desc}. `average_precision_score` is **not** the trapezoidal area under the PR curve. Return "
             f"`(average_precision_score(y_true, y_prob), auc(rec, prec))` where `prec, rec, _ = "
             f"precision_recall_curve(y_true, y_prob)` and `auc` (from `sklearn.metrics`) applies the trapezoid rule.",
             setup, f"{MET}precision_recall_curve, average_precision_score, auc\n"
                    "prec, rec, _ = precision_recall_curve(y_true, y_prob)\nresult = (average_precision_score(y_true, y_prob), auc(rec, prec))",
             require=["auc", "precision_recall_curve"], tol=1e-9)


@template(M, "hard")
def threshold_on_validation(rng):
    dd = synth_clf(rng, n=(400, 700), d=(5, 10), classes=2, imbalance=rng.choice([0.15, 0.25]))
    s = seeded(rng)
    setup = dd.setup + ("\nfrom sklearn.model_selection import train_test_split\n"
                        f"X_tmp, X_te, y_tmp, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state={s})\n"
                        f"X_tr, X_val, y_tr, y_val = train_test_split(X_tmp, y_tmp, test_size=1/3, stratify=y_tmp, random_state={s})")
    head = (f"{dd.desc}; it was split (stratified) into training, validation and test parts (`X_tr`, `X_val`, `X_te` "
            f"and their `y`). Train `clf = {PIPE}` on the training part. Choosing a threshold on the test set would "
            f"leak, so tune it on validation.")
    sol = ("from sklearn.linear_model import LogisticRegression\nfrom sklearn.pipeline import make_pipeline\n"
           f"from sklearn.preprocessing import StandardScaler\n{MET}precision_recall_curve, f1_score\n"
           f"clf = {PIPE}.fit(X_tr, y_tr)\npv = clf.predict_proba(X_val)[:, 1]\n"
           "prec, rec, thr = precision_recall_curve(y_val, pv)\nf1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)\n"
           "t = thr[np.argmax(f1)]\npt = clf.predict_proba(X_te)[:, 1]\n")
    task = rng.choice(["chosen", "compare", "leak_gap"])
    if task == "chosen":
        return P(f"{head} Pick the threshold with the best validation F1 from `precision_recall_curve` (the first on "
                 f"ties) and return `(threshold, test F1 at that threshold)` (predict 1 when P ≥ threshold).", setup,
                 sol + "result = (float(t), f1_score(y_te, (pt >= t).astype(int)))", require=["precision_recall_curve"],
                 tol=1e-9)
    if task == "compare":
        return P(f"{head} Pick the threshold with the best validation F1 from `precision_recall_curve` (the first on "
                 f"ties) and return the test F1 at the default 0.5 and at the tuned threshold, as a tuple.", setup,
                 sol + "result = (f1_score(y_te, (pt >= 0.5).astype(int)), f1_score(y_te, (pt >= t).astype(int)))",
                 require=["precision_recall_curve", "f1_score"], tol=1e-9)
    return P(f"{head} Pick the threshold with the best validation F1 (the first on ties). Return `(best validation F1, "
             f"test F1 at that threshold)`: the validation number is optimistic because the threshold was chosen on it.",
             setup, sol + "result = (float(f1.max()), f1_score(y_te, (pt >= t).astype(int)))",
             require=["precision_recall_curve"], tol=1e-9)


@template(M, "hard")
def bootstrap_metric_ci(rng):
    setup, desc = _fitted_binary(rng)
    s, B = seeded(rng), rng.choice([200, 300])
    task = rng.choice(["f1", "auc", "diff"])
    head = (f"{desc}. Bootstrap the test set for an uncertainty estimate: with `rs = np.random.default_rng({s})`, repeat "
            f"{B} times: draw `idx = rs.integers(0, len(y_te), len(y_te))` and score the resampled rows.")
    if task == "f1":
        return P(f"{head} Return the 2.5th and 97.5th percentiles (`np.percentile`) of the {B} F1 scores of `y_pred`, as "
                 f"a tuple.", setup,
                 f"{MET}f1_score\nrs = np.random.default_rng({s})\nvals = []\nfor _ in range({B}):\n"
                 "    idx = rs.integers(0, len(y_te), len(y_te))\n    vals.append(f1_score(y_te[idx], y_pred[idx]))\n"
                 "result = tuple(np.percentile(vals, [2.5, 97.5]))", require=["f1_score"], tol=1e-9)
    if task == "auc":
        return P(f"{head} Return the mean and standard deviation (`np.std`) of the {B} ROC AUCs of `y_prob`. (Skip a "
                 f"resample that contains only one class; it can't be scored.)", setup,
                 f"{MET}roc_auc_score\nrs = np.random.default_rng({s})\nvals = []\nfor _ in range({B}):\n"
                 "    idx = rs.integers(0, len(y_te), len(y_te))\n    if len(np.unique(y_te[idx])) == 2:\n"
                 "        vals.append(roc_auc_score(y_te[idx], y_prob[idx]))\nresult = (float(np.mean(vals)), float(np.std(vals)))",
                 require=["roc_auc_score"], tol=1e-9)
    return P(f"{head} Is the 0.3 threshold better than 0.5 for F1? For every resample compute F1 at 0.3 minus F1 at 0.5 "
             f"(predict 1 when `y_prob >= t`); return the share of resamples where the difference is positive and the "
             f"mean difference.", setup,
             f"{MET}f1_score\nrs = np.random.default_rng({s})\nd = []\nfor _ in range({B}):\n"
             "    idx = rs.integers(0, len(y_te), len(y_te))\n    yt, p = y_te[idx], y_prob[idx]\n"
             "    d.append(f1_score(yt, (p >= 0.3).astype(int)) - f1_score(yt, (p >= 0.5).astype(int)))\n"
             "d = np.array(d)\nresult = (float((d > 0).mean()), float(d.mean()))", require=["f1_score"], tol=1e-9)


@template(M, "hard")
def clf_metric_pitfalls(rng):
    setup, desc = _fitted_binary(rng)
    items = [
        ("roc_auc_score", "This passes hard labels to `roc_auc_score`, which then only sees one threshold:\n```\n"
                          "auc = roc_auc_score(y_te, y_pred)\n```\nUse the probabilities and return `(wrong, right)`.",
         f"{MET}roc_auc_score\nresult = (roc_auc_score(y_te, y_pred), roc_auc_score(y_te, y_prob))"),
        ("precision_score", "The arguments are swapped, which turns precision into recall:\n```\n"
                            "p = precision_score(y_pred, y_te)\n```\nReturn `(swapped, correct precision, recall)`.",
         f"{MET}precision_score, recall_score\n"
         "result = (precision_score(y_pred, y_te), precision_score(y_te, y_pred), recall_score(y_te, y_pred))"),
        ("ravel", "This unpacks the confusion matrix in the wrong order:\n```\ntp, fn, fp, tn = confusion_matrix(y_te, "
                  "y_pred).ravel()\nrecall = tp / (tp + fn)\n```\nscikit-learn's binary order is `tn, fp, fn, tp`. Fix it "
                  "and return the correct recall.",
         f"{MET}confusion_matrix\ntn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()\nresult = tp / (tp + fn)"),
        ("log_loss", "This computes the log loss of hard labels, which punishes every mistake as if made with total "
                     "confidence:\n```\nll = log_loss(y_te, y_pred)\n```\nReturn `(hard-label value, log loss of y_prob)`.",
         f"{MET}log_loss\nresult = (log_loss(y_te, y_pred), log_loss(y_te, y_prob))"),
        ("roc_auc_score", "This uses the wrong column of `predict_proba`, P(class 0):\n```\nauc = roc_auc_score(y_te, "
                          "clf.predict_proba(X_te)[:, 0])\n```\nReturn `(wrong, right)`: the wrong one is exactly "
                          "1 − right.",
         f"{MET}roc_auc_score\nresult = (roc_auc_score(y_te, clf.predict_proba(X_te)[:, 0]), roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1]))"),
        ("zero_division", "At a very strict threshold no sample is predicted positive, so precision is 0/0 and "
                          "scikit-learn warns:\n```\np = precision_score(y_te, (y_prob >= 0.9999).astype(int))\n```\nSay "
                          "explicitly what you want with `zero_division=0` and return the precision and the recall at "
                          "that threshold.",
         f"{MET}precision_score, recall_score\nstrict = (y_prob >= 0.9999).astype(int)\n"
         "result = (precision_score(y_te, strict, zero_division=0), recall_score(y_te, strict))"),
        ("average", "This crashes on 3 classes: the default `average='binary'` only works for two.\n```\n"
                    "f = f1_score(y3_true, y3_pred)\n```\nReturn the macro F1 instead (`y3_true`, `y3_pred` are given).",
         f"{MET}f1_score\nresult = f1_score(y3_true, y3_pred, average='macro')",
         {"setup": "y3_true = np.array([0, 1, 2, 2, 1, 0, 2, 1, 0, 2])\ny3_pred = np.array([0, 2, 2, 2, 1, 0, 1, 1, 0, 0])"}),
    ]
    return catalog(rng, items, setup, f"{desc}.", tol=1e-9)


@template(M, "hard")
def out_of_fold_metrics(rng):
    dd = synth_clf(rng, n=(300, 500), d=(4, 8), classes=2, imbalance=rng.choice([0.1, 0.2]))
    task = rng.choice(["pooled", "threshold", "per_fold"])
    imp = ("from sklearn.linear_model import LogisticRegression\nfrom sklearn.pipeline import make_pipeline\n"
           "from sklearn.preprocessing import StandardScaler\nfrom sklearn.model_selection import StratifiedKFold, cross_val_predict\n"
           f"clf = {PIPE}\ncv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)\n")
    head = (f"{dd.desc}. Use `cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)` (stratified, so every fold "
            f"keeps the rare class) and `clf = {PIPE}`.")
    if task == "pooled":
        return P(f"{head} Get out-of-fold probabilities with `cross_val_predict(clf, X, y, cv=cv, "
                 f"method='predict_proba')[:, 1]` and return `(roc_auc, average_precision, f1 at 0.5)` on them.",
                 dd.setup, imp + f"{MET}roc_auc_score, average_precision_score, f1_score\n"
                                 "p = cross_val_predict(clf, X, y, cv=cv, method='predict_proba')[:, 1]\n"
                                 "result = (roc_auc_score(y, p), average_precision_score(y, p), f1_score(y, (p >= 0.5).astype(int)))",
                 require=["cross_val_predict", "StratifiedKFold"], tol=1e-9)
    if task == "threshold":
        return P(f"{head} Get out-of-fold probabilities with `cross_val_predict` and choose the threshold maximizing F1 "
                 f"on them via `precision_recall_curve` (the first on ties). Return `(threshold, F1 there, F1 at 0.5)`.",
                 dd.setup, imp + f"{MET}precision_recall_curve, f1_score\n"
                                 "p = cross_val_predict(clf, X, y, cv=cv, method='predict_proba')[:, 1]\n"
                                 "prec, rec, thr = precision_recall_curve(y, p)\n"
                                 "f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)\ni = int(np.argmax(f1))\n"
                                 "result = (float(thr[i]), float(f1[i]), f1_score(y, (p >= 0.5).astype(int)))",
                 require=["cross_val_predict", "precision_recall_curve"], tol=1e-9)
    return P(f"{head} Loop over `cv.split(X, y)`, fit `clf` on each training fold and compute the average precision on "
             f"the held-out fold. Return the 5 per-fold values as an array.", dd.setup,
             imp + f"{MET}average_precision_score\nout = []\nfor tr, te in cv.split(X, y):\n"
                   "    clf.fit(X[tr], y[tr])\n    out.append(average_precision_score(y[te], clf.predict_proba(X[te])[:, 1]))\n"
                   "result = np.array(out)", require=["StratifiedKFold", "average_precision_score"], tol=1e-9)


@template(M, "hard")
def multiclass_from_confusion(rng):
    setup, desc, k = _fitted_multi(rng)
    task = rng.choice(["precisions", "mcc", "weighted_recall"])
    cm = f"{MET}confusion_matrix\nC = confusion_matrix(y_te, y_pred).astype(float)\n"
    if task == "precisions":
        return P(f"{desc}. From `C = confusion_matrix(y_te, y_pred)` alone, compute the macro-, micro- and "
                 f"weighted-averaged precision **by hand** (per-class precision = diagonal / column sum; weighted uses "
                 f"each class's support, its row sum). Return them as a tuple, in that order.", setup,
                 cm + "p = np.diag(C) / np.maximum(C.sum(axis=0), 1)\nsup = C.sum(axis=1)\n"
                      "result = (p.mean(), np.trace(C) / C.sum(), (p * sup).sum() / sup.sum())", require=["confusion_matrix"],
                 tol=1e-9)
    if task == "mcc":
        return P(f"{desc}. The multiclass Matthews correlation from the confusion matrix `C` is `(c·s − Σ_k p_k·t_k) / "
                 f"sqrt((s² − Σ p_k²)(s² − Σ t_k²))` with `c` = the trace, `s` = the total, `p_k` = column sums and `t_k` "
                 f"= row sums. Compute it and return `(by_hand, matthews_corrcoef(y_te, y_pred))`.", setup,
                 cm + f"{MET}matthews_corrcoef\nc, s = np.trace(C), C.sum()\npk, tk = C.sum(axis=0), C.sum(axis=1)\n"
                      "result = ((c * s - pk @ tk) / np.sqrt((s ** 2 - pk @ pk) * (s ** 2 - tk @ tk)), matthews_corrcoef(y_te, y_pred))",
                 require=["confusion_matrix", "matthews_corrcoef"], tol=1e-9)
    return P(f"{desc}. Show that the support-weighted recall equals the accuracy: from `C = confusion_matrix(y_te, "
             f"y_pred)` compute per-class recall, weight it by support and return `(weighted recall, "
             f"recall_score(..., average='weighted'), accuracy_score(y_te, y_pred))`.", setup,
             cm + f"{MET}recall_score, accuracy_score\nr = np.diag(C) / C.sum(axis=1)\nsup = C.sum(axis=1)\n"
                  "result = ((r * sup).sum() / sup.sum(), recall_score(y_te, y_pred, average='weighted'), accuracy_score(y_te, y_pred))",
             require=["confusion_matrix", "recall_score"], tol=1e-9)


@template(M, "hard")
def top_k_by_hand(rng):
    setup, desc, k = _fitted_multi(rng)
    kk = 2
    task = rng.choice(["overall", "per_class"])
    if task == "overall":
        return P(f"{desc}. Compute the top-{kk} accuracy **by hand**: a sample counts if its true class is among the "
                 f"{kk} largest entries of its row of `proba` (`np.argsort`). Return `(by_hand, "
                 f"top_k_accuracy_score(y_te, proba, k={kk}, labels=np.arange(proba.shape[1])))`.", setup,
                 f"{MET}top_k_accuracy_score\ntop = np.argsort(proba, axis=1)[:, -{kk}:]\n"
                 f"result = (float((top == y_te[:, None]).any(axis=1).mean()), top_k_accuracy_score(y_te, proba, k={kk}, labels=np.arange(proba.shape[1])))",
                 require=["argsort", "top_k_accuracy_score"], tol=1e-9)
    return P(f"{desc}. For each class, return the share of its test samples whose true class is among the {kk} most "
             f"probable ones (a per-class top-{kk} recall), as an array in class order.", setup,
             f"top = np.argsort(proba, axis=1)[:, -{kk}:]\nhit = (top == y_te[:, None]).any(axis=1)\n"
             "result = np.array([hit[y_te == c].mean() for c in range(proba.shape[1])])", require=["argsort"], tol=1e-9)


@template(M, "hard")
def gains_and_lift(rng):
    setup, desc = _fitted_binary(rng)
    fr = rng.choice([0.1, 0.2, 0.3])
    task = rng.choice(["gains", "lift_table", "budget"])
    head = f"{desc}. Marketing-style evaluation: rank the test samples by `y_prob`, highest first."
    if task == "gains":
        return P(f"{head} The *cumulative gains* at a fraction f is the share of all positives found in the top f·n "
                 f"samples (`ceil`). Return the gains at f = 0.1, 0.2, ..., 1.0 as an array.", setup,
                 "o = np.argsort(-y_prob, kind='stable')\nhits = np.cumsum(y_te[o])\nn = len(o)\n"
                 "ks = np.ceil(np.round(np.arange(1, 11) / 10 * n, 9)).astype(int)\nresult = hits[ks - 1] / y_te.sum()",
                 require=["argsort"], tol=1e-9)
    if task == "lift_table":
        return P(f"{head} Split the ranking into 5 equal groups (`np.array_split` of the sorted order) and return each "
                 f"group's positive rate divided by the overall positive rate (the lift per quintile), as an array.",
                 setup, "o = np.argsort(-y_prob, kind='stable')\n"
                        "result = np.array([y_te[g].mean() / y_te.mean() for g in np.array_split(o, 5)])",
                 require=["array_split"], tol=1e-9)
    return P(f"{head} With budget to contact the top {fr:.0%} (`ceil({fr}·n)` samples), return `(precision among them, "
             f"recall: the share of all positives they include, the score of the last one contacted)`.", setup,
             f"o = np.argsort(-y_prob, kind='stable')\nk = int(np.ceil({fr} * len(o)))\ntop = o[:k]\n"
             "result = (float(y_te[top].mean()), float(y_te[top].sum() / y_te.sum()), float(y_prob[top[-1]]))",
             require=["argsort"], tol=1e-9)


@template(M, "hard")
def log_loss_extremes(rng):
    setup, desc, n = _labels(rng, n=(16, 24))
    task = rng.choice(["confident_mistake", "clip", "temperature"])
    if task == "confident_mistake":
        i = rng.randint(0, 5)
        return P(f"{desc}. Log loss punishes confident mistakes without limit. Make a copy of `y_prob` where sample {i} "
                 f"gets probability 0.001 for its **true** class (0.001 if it is positive, 0.999 if negative) and return "
                 f"the log loss before and after, as a tuple.", setup,
                 f"{MET}log_loss\nq = y_prob.copy()\nq[{i}] = 0.001 if y_true[{i}] == 1 else 0.999\n"
                 "result = (log_loss(y_true, y_prob), log_loss(y_true, q))", require=["log_loss"], tol=1e-9)
    if task == "clip":
        return P(f"{desc}. Probabilities of exactly 0 or 1 make the log loss infinite, so it is common to clip them. Turn "
                 f"`y_prob` into hard 0/1 values (`(y_prob >= 0.5).astype(float)`), clip them to `[eps, 1 − eps]` for "
                 f"eps in 1e-2, 1e-4, 1e-8 and return the three log losses as a list: the smaller eps, the harsher the "
                 f"penalty for each mistake.", setup,
                 f"{MET}log_loss\nh = (y_prob >= 0.5).astype(float)\n"
                 "result = [log_loss(y_true, np.clip(h, e, 1 - e)) for e in (1e-2, 1e-4, 1e-8)]",
                 require=["log_loss", "clip"], tol=1e-9)
    return P(f"{desc}. Soften or sharpen the probabilities with a temperature T: `p_T = sigmoid(logit(y_prob) / T)`. For "
             f"T in 0.5, 1, 2, 4, return the log loss and the Brier score of `p_T` as a list of tuples; the accuracy "
             f"never changes, only the confidence.", setup,
             f"{MET}log_loss, brier_score_loss\nz = np.log(y_prob / (1 - y_prob))\nresult = []\nfor T in (0.5, 1, 2, 4):\n"
             "    p = 1 / (1 + np.exp(-z / T))\n    result.append((log_loss(y_true, p), brier_score_loss(y_true, p)))",
             require=["log_loss", "brier_score_loss"], tol=1e-9)


# ================================================================= challenge

def _scores_gen(report, ties):
    def gen(seed, n_):
        s = "g.integers(0, 50, n).astype(float) / 50" if ties else "g.random(n).round(6)"
        return (f"g = np.random.default_rng({seed})\nn = {n_}\ny = (g.random(n) < 0.3).astype(int)\ny[:2] = [1, 0]\n"
                f"s = np.clip({s} + 0.25 * y, 0, 1)\nreport = {report!r}")
    return gen


@template(M, "challenge")
def roc_auc_ties(rng):
    n, s = rng.randint(10, 14), seeded(rng)
    report = rng.choice(["auc", "gini", "discordant"])
    gen = _scores_gen(report, True)
    sol = ("o = np.argsort(s, kind='stable')\nss = s[o]\nstarts = np.r_[0, np.flatnonzero(np.diff(ss)) + 1]\n"
           "ends = np.r_[starts[1:], len(ss)]\nr = np.empty(len(s))\nr[o] = np.repeat((starts + ends + 1) / 2, ends - starts)\n"
           "P_, N_ = int(y.sum()), int(len(y) - y.sum())\nauc = (r[y == 1].sum() - P_ * (P_ + 1) / 2) / (P_ * N_)\n"
           "if report == 'auc':\n    result = float(auc)\nelif report == 'gini':\n    result = float(2 * auc - 1)\nelse:\n"
           "    neg = np.sort(s[y == 0])\n    result = int((len(neg) - np.searchsorted(neg, s[y == 1], side='right')).sum())")
    tests = [gen(s + i, 200000) for i in (1, 2)]
    what = {"auc": "the ROC AUC: the probability that a random positive outscores a random negative, a tie counting ½ "
                   "(what `roc_auc_score(y, s)` returns)",
            "gini": "the Gini coefficient `2·AUC − 1`",
            "discordant": "the number of positive-negative pairs ranked the **wrong** way (negative strictly higher), an "
                          "int"}[report]
    return challenge(
        "ROC AUC with ties",
        f"Binary labels `y` (both classes present) and scores `s`, with many ties, are given. Return {what}. "
        f"(`report` = {report!r}.) Comparing every positive with every negative is far too slow for large n: sort "
        "once and use (average) ranks.",
        gen(s, n), sol, tests, "n ≤ 2·10^5", speed="O(n log n)", tol=1e-9)


@template(M, "challenge")
def best_f1_threshold_scan(rng):
    n, s = rng.randint(10, 14), seeded(rng)
    report = rng.choice(["f1", "threshold", "precision_at_recall"])
    gen = _scores_gen(report, True)
    sol = ("o = np.argsort(-s, kind='stable')\nss, yy = s[o], y[o]\nlast = np.r_[np.flatnonzero(np.diff(ss)), len(ss) - 1]\n"
           "tp = np.cumsum(yy)[last]\nfp = (last + 1) - tp\nP_ = yy.sum()\nfn = P_ - tp\nthr = ss[last]\n"
           "f1 = 2 * tp / (2 * tp + fp + fn)\nif report == 'f1':\n    result = float(f1.max())\n"
           "elif report == 'threshold':\n    result = float(thr[np.flatnonzero(f1 == f1.max())].max())\nelse:\n"
           "    ok = tp >= 0.8 * P_\n    result = float((tp / (tp + fp))[ok].max())")
    tests = [gen(s + i, 200000) for i in (1, 2)]
    what = {"f1": "the best F1 over all thresholds",
            "threshold": "the **largest** threshold achieving the best F1",
            "precision_at_recall": "the best precision `TP / (TP + FP)` among thresholds whose recall `TP / P` is at "
                                   "least 0.8"}[report]
    return challenge(
        "Threshold scan",
        "Binary labels `y` (both classes present) and scores `s` (with ties) are given. A threshold `t` (any value "
        "appearing in `s`) predicts 1 for `s >= t`; compute F1 as `2·TP / (2·TP + FP + FN)` (exactly this formula, so "
        f"equal F1 values compare equal). Return {what}. (`report` = {report!r}.) Recounting the confusion matrix for "
        "every threshold is quadratic: sort once and sweep.",
        gen(s, n), sol, tests, "n ≤ 2·10^5", speed="O(n log n)", tol=1e-9)


@template(M, "challenge")
def average_precision_ties(rng):
    n, s = rng.randint(10, 14), seeded(rng)
    report = rng.choice(["ap", "trapz"])
    gen = _scores_gen(report, True)
    sol = ("o = np.argsort(-s, kind='stable')\nss, yy = s[o], y[o]\nlast = np.r_[np.flatnonzero(np.diff(ss)), len(ss) - 1]\n"
           "tp = np.cumsum(yy)[last]\nfp = (last + 1) - tp\nprec = tp / (tp + fp)\nrec = tp / yy.sum()\n"
           "if report == 'ap':\n    result = float(np.sum(np.diff(np.r_[0.0, rec]) * prec))\nelse:\n"
           "    R, Pp = np.r_[0.0, rec], np.r_[1.0, prec]\n    result = float(np.sum(np.diff(R) * (Pp[1:] + Pp[:-1]) / 2))")
    tests = [gen(s + i, 200000) for i in (1, 2)]
    what = {"ap": "the average precision `Σ_k (R_k − R_{k−1})·P_k` over the distinct thresholds from high to low "
                  "(`R_0 = 0`), what `average_precision_score(y, s)` returns",
            "trapz": "the trapezoidal area under the precision-recall curve made of the point (recall 0, precision 1) "
                     "followed by one point per distinct threshold, from high to low (what `auc(rec, prec)` gives for "
                     "`precision_recall_curve`'s output)"}[report]
    return challenge(
        "Average precision with ties",
        "Binary labels `y` (both classes present) and scores `s` (with ties) are given. Each **distinct** score `t`, "
        "from high to low, is a threshold (predict 1 for `s >= t`) with precision `P = TP / (TP + FP)` and recall `R = "
        f"TP / (number of positives)`. Return {what}. (`report` = {report!r}.)",
        gen(s, n), sol, tests, "n ≤ 2·10^5", speed="O(n log n)", tol=1e-9)


@template(M, "challenge")
def confusion_macro_metrics(rng):
    n, K, s = rng.randint(12, 16), rng.randint(3, 4), seeded(rng)
    report = rng.choice(["macro_f1", "mcc", "kappa"])

    def gen(seed, n_, K_):
        return (f"g = np.random.default_rng({seed})\nK = {K_}\ny_true = g.permutation(np.arange({n_}) % K)\n"
                f"y_pred = np.where(g.random({n_}) < 0.6, y_true, g.integers(0, K, {n_}))\nreport = {report!r}")
    sol = ("C = np.bincount(y_true * K + y_pred, minlength=K * K).reshape(K, K).astype(float)\n"
           "t, p = C.sum(axis=1), C.sum(axis=0)\ntp = np.diag(C)\nn = C.sum()\nif report == 'macro_f1':\n"
           "    present = (t + p) > 0\n    result = float((2 * tp / np.maximum(t + p, 1))[present].mean())\n"
           "elif report == 'mcc':\n    c = tp.sum()\n    result = float((c * n - p @ t) / np.sqrt((n * n - p @ p) * (n * n - t @ t)))\n"
           "else:\n    po, pe = tp.sum() / n, (t @ p) / n ** 2\n    result = float((po - pe) / (1 - pe))")
    tests = [gen(s + i, 200000, 50) for i in (1, 2)]
    what = {"macro_f1": "the macro-averaged F1 (`f1_score(y_true, y_pred, average='macro')`: per class `2·TP / (2·TP + "
                        "FP + FN)`, averaged over the classes that appear in `y_true` or `y_pred`)",
            "mcc": "the multiclass Matthews correlation (`matthews_corrcoef`)",
            "kappa": "Cohen's kappa (`cohen_kappa_score`)"}[report]
    return challenge(
        "Metrics from a big confusion matrix",
        f"Labels `y_true` and predictions `y_pred` take values 0..K−1 (`K` given). Return {what}. (`report` = "
        f"{report!r}.) Build the K × K confusion matrix in one vectorized pass (no Python loop over samples) and derive "
        "the metric from its diagonal and margins.",
        gen(s, n, K), sol, tests, "n ≤ 2·10^5, K ≤ 50", speed="O(n + K²)", tol=1e-9)


@template(M, "challenge")
def min_cost_threshold(rng):
    n, s = rng.randint(10, 14), seeded(rng)
    report = rng.choice(["cost", "threshold", "flagged"])
    cfp, cfn = rng.choice([(1, 4), (1, 9), (3, 2)])

    def gen(seed, n_):
        return (f"g = np.random.default_rng({seed})\nn = {n_}\ny = (g.random(n) < 0.3).astype(int)\n"
                f"s = np.clip(g.integers(0, 50, n) / 50 + 0.25 * y, 0, 1)\nc_fp, c_fn = {cfp}, {cfn}\nreport = {report!r}")
    sol = ("o = np.argsort(-s, kind='stable')\nss, yy = s[o], y[o]\nlast = np.r_[np.flatnonzero(np.diff(ss)), len(ss) - 1]\n"
           "tp = np.r_[0, np.cumsum(yy)[last]]\nflag = np.r_[0, last + 1]\nfp = flag - tp\nfn = yy.sum() - tp\n"
           "thr = np.r_[np.inf, ss[last]]\ncost = c_fp * fp + c_fn * fn\nbest = np.flatnonzero(cost == cost.min())[0]\n"
           "result = {'cost': int(cost[best]), 'threshold': float(thr[best]), 'flagged': int(flag[best])}[report]")
    tests = [gen(s + i, 200000) for i in (1, 2)]
    what = {"cost": "the minimum total cost (an int)",
            "threshold": "the **largest** threshold achieving the minimum cost (`np.inf` if flagging nobody is best)",
            "flagged": "how many samples are flagged at the largest cost-minimizing threshold"}[report]
    return challenge(
        "Cheapest threshold",
        "Binary labels `y` and scores `s` (with ties) are given; a false positive costs `c_fp` and a false negative "
        "`c_fn`. A threshold `t` flags (predicts 1 for) every sample with `s >= t`; the candidates are the distinct "
        "values of `s` plus `t = np.inf` (flag nobody). The total cost is `c_fp·FP + c_fn·FN`. Return "
        f"{what}. (`report` = {report!r}.)",
        gen(s, n), sol, tests, "n ≤ 2·10^5", speed="O(n log n)", tol=1e-9)


@template(M, "challenge")
def multilabel_scores(rng):
    n, L, s = rng.randint(6, 9), rng.randint(3, 4), seeded(rng)
    report = rng.choice(["hamming", "jaccard_samples", "micro_f1", "subset"])

    def gen(seed, n_, L_):
        return (f"g = np.random.default_rng({seed})\nY_true = (g.random(({n_}, {L_})) < 0.4).astype(int)\n"
                f"Y_true[np.arange({n_}), g.integers(0, {L_}, {n_})] = 1   # every sample has a label\n"
                f"Y_pred = np.where(g.random(({n_}, {L_})) < 0.8, Y_true, 1 - Y_true)\nreport = {report!r}")
    sol = ("T, Pr = Y_true.astype(bool), Y_pred.astype(bool)\nif report == 'hamming':\n    result = float((T != Pr).mean())\n"
           "elif report == 'jaccard_samples':\n    result = float(((T & Pr).sum(axis=1) / (T | Pr).sum(axis=1)).mean())\n"
           "elif report == 'micro_f1':\n    tp = (T & Pr).sum()\n    result = float(2 * tp / (T.sum() + Pr.sum()))\nelse:\n"
           "    result = float((T == Pr).all(axis=1).mean())")
    tests = [gen(s + i, 40000, 5) for i in (1, 2)]
    what = {"hamming": "the Hamming loss (`hamming_loss`)",
            "jaccard_samples": "the sample-averaged Jaccard score (`jaccard_score(..., average='samples')`)",
            "micro_f1": "the micro-averaged F1 (`f1_score(..., average='micro')`)",
            "subset": "the subset accuracy (`accuracy_score` on the matrices: rows predicted exactly right)"}[report]
    return challenge(
        "Multilabel scores",
        "`Y_true` and `Y_pred` are n × L 0/1 indicator matrices (every row of `Y_true` has at least one label). Return "
        f"{what}, without calling scikit-learn and without a Python loop over samples. (`report` = {report!r}.)",
        gen(s, n, L), sol, tests, "n·L ≤ 2·10^5", speed="O(n·L) vectorized", tol=1e-9)
