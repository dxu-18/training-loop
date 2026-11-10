"""4.16 Probability calibration & decision thresholds: CalibratedClassifierCV (sigmoid / isotonic), reliability
curves, FrozenEstimator, TunedThresholdClassifierCV, FixedThresholdClassifier, cost-sensitive cut-offs."""

from ._common import P, catalog, challenge, lit, seeded, table, template

M = "4.16"
CAL = "from sklearn.calibration import CalibratedClassifierCV\n"
CURVE = "from sklearn.calibration import calibration_curve\n"
NB = "from sklearn.naive_bayes import GaussianNB\n"
LRI = "from sklearn.linear_model import LogisticRegression\n"
LR = "LogisticRegression(max_iter=1000)"
TUNED = "from sklearn.model_selection import TunedThresholdClassifierCV\n"
FIXED = "from sklearn.model_selection import FixedThresholdClassifier\n"
FROZEN = "from sklearn.frozen import FrozenEstimator\n"
BRIER = "from sklearn.metrics import brier_score_loss\n"


def _imb(rng, n=(300, 600), split: bool = True):
    """Imbalanced make_classification data, split stratified into X_tr / X_te."""
    n_ = rng.randint(*n)
    w = rng.choice([0.1, 0.15, 0.2, 0.3])
    d = rng.randint(5, 12)
    s = seeded(rng)
    setup = (f"from sklearn.datasets import make_classification\nX, y = make_classification(n_samples={n_}, "
             f"n_features={d}, weights=[{round(1 - w, 2)}, {w}], random_state={s})")
    desc = (f"`X`, `y` come from `make_classification(n_samples={n_}, n_features={d}, weights=[{round(1 - w, 2)}, {w}], "
            f"random_state={s})` (class 1 is rare: about {int(w * 100)}%)")
    if split:
        setup += ("\nfrom sklearn.model_selection import train_test_split\n"
                  "X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, random_state=0)")
        desc += (", split with `train_test_split(X, y, stratify=y, random_state=0)` into `X_tr, X_te, y_tr, y_te`")
    return setup, desc


def _three_way(rng, n=(450, 650)):
    """train / calibration / test split of imbalanced data."""
    setup, desc = _imb(rng, n=n, split=False)
    setup += ("\nfrom sklearn.model_selection import train_test_split\n"
              "X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, random_state=0)\n"
              "X_fit, X_cal, y_fit, y_cal = train_test_split(X_tr, y_tr, stratify=y_tr, test_size=0.4, random_state=1)")
    desc += (". It was split into a test set (`X_te, y_te`), and the rest into a fitting set (`X_fit, y_fit`) and a "
             "calibration set (`X_cal, y_cal`)")
    return setup, desc


# ================================================================= lite

@template(M, "lite")
def calibration_api(rng):
    setup, desc = _imb(rng)
    cv, nb = rng.choice([3, 5]), rng.choice([5, 8, 10])
    m = rng.randint(3, 6)
    items = [
        ("CalibratedClassifierCV", f"Calibrate `GaussianNB()` with `CalibratedClassifierCV(GaussianNB(), "
                                   f"method='isotonic', cv={cv})` fitted on the training part, and return the calibrated "
                                   f"P(class 1) of the first {m} test rows.",
         f"{CAL}{NB}cal = CalibratedClassifierCV(GaussianNB(), method='isotonic', cv={cv}).fit(X_tr, y_tr)\n"
         f"result = cal.predict_proba(X_te[:{m}])[:, 1]", {"require": ["CalibratedClassifierCV", "predict_proba"]}),
        ("method", f"Platt scaling: calibrate `GaussianNB()` with `CalibratedClassifierCV(..., method='sigmoid', "
                   f"cv={cv})` on the training part and return P(class 1) for the first {m} test rows.",
         f"{CAL}{NB}cal = CalibratedClassifierCV(GaussianNB(), method='sigmoid', cv={cv}).fit(X_tr, y_tr)\n"
         f"result = cal.predict_proba(X_te[:{m}])[:, 1]", {"require": ["CalibratedClassifierCV", "method"]}),
        ("calibration_curve", f"Fit `GaussianNB()` on the training part and return the reliability curve of its test "
                              f"probabilities: `calibration_curve(y_te, p, n_bins={nb})` with `p` = P(class 1), as the "
                              f"tuple `(prob_true, prob_pred)`.",
         f"{CURVE}{NB}p = GaussianNB().fit(X_tr, y_tr).predict_proba(X_te)[:, 1]\nresult = calibration_curve(y_te, p, n_bins={nb})"),
        ("calibrated_classifiers_", f"Fit `CalibratedClassifierCV(GaussianNB(), cv={cv})` on the training part and "
                                    f"return how many (model, calibrator) pairs it keeps (`len(calibrated_classifiers_)`: "
                                    f"one per fold).",
         f"{CAL}{NB}result = len(CalibratedClassifierCV(GaussianNB(), cv={cv}).fit(X_tr, y_tr).calibrated_classifiers_)"),
        ("ensemble", f"With `ensemble=False`, `CalibratedClassifierCV(GaussianNB(), cv={cv}, ensemble=False)` fits one "
                     f"calibrator on out-of-fold predictions and one final model. Fit it on the training part and "
                     f"return `len(calibrated_classifiers_)`.",
         f"{CAL}{NB}result = len(CalibratedClassifierCV(GaussianNB(), cv={cv}, ensemble=False).fit(X_tr, y_tr).calibrated_classifiers_)"),
        ("predict", f"Calibrate `GaussianNB()` (`CalibratedClassifierCV`, `method='isotonic'`, `cv={cv}`) on the "
                    f"training part and return its predicted labels for the first 8 test rows.",
         f"{CAL}{NB}result = CalibratedClassifierCV(GaussianNB(), method='isotonic', cv={cv}).fit(X_tr, y_tr).predict(X_te[:8])",
         {"require": ["CalibratedClassifierCV", "predict"]}),
        ("strategy", f"Bins with equal counts: return `calibration_curve(y_te, p, n_bins=4, strategy='quantile')` for the "
                     f"test P(class 1) `p` of a `GaussianNB()` fitted on the training part.",
         f"{CURVE}{NB}p = GaussianNB().fit(X_tr, y_tr).predict_proba(X_te)[:, 1]\n"
         "result = calibration_curve(y_te, p, n_bins=4, strategy='quantile')", {"require": ["calibration_curve", "strategy"]}),
    ]
    return catalog(rng, items, setup, f"{desc}.", tol=1e-9)


@template(M, "lite")
def threshold_api(rng):
    setup, desc = _imb(rng)
    t = rng.choice([0.2, 0.25, 0.3, 0.4, 0.6])
    sc = rng.choice(["f1", "balanced_accuracy"])
    items = [
        ("FixedThresholdClassifier", f"Predict class 1 whenever P(class 1) ≥ {t}: fit `FixedThresholdClassifier({LR}, "
                                     f"threshold={t})` on the training part and return its predictions for the first 10 "
                                     f"test rows.",
         f"{FIXED}{LRI}result = FixedThresholdClassifier({LR}, threshold={t}).fit(X_tr, y_tr).predict(X_te[:10])"),
        ("FixedThresholdClassifier", f"Fit `{LR}` and `FixedThresholdClassifier({LR}, threshold={t})` on the training "
                                     f"part. Return how many test rows each predicts as class 1, as `(default, fixed)`.",
         f"{FIXED}{LRI}a = {LR}.fit(X_tr, y_tr).predict(X_te).sum()\n"
         f"b = FixedThresholdClassifier({LR}, threshold={t}).fit(X_tr, y_tr).predict(X_te).sum()\nresult = (int(a), int(b))"),
        ("TunedThresholdClassifierCV", f"Learn the cut-off: fit `TunedThresholdClassifierCV({LR}, scoring={sc!r}, cv=5)` "
                                       f"on the training part and return `best_threshold_`.",
         f"{TUNED}{LRI}result = TunedThresholdClassifierCV({LR}, scoring={sc!r}, cv=5).fit(X_tr, y_tr).best_threshold_",
         {"require": ["TunedThresholdClassifierCV", "best_threshold_"]}),
        ("best_score_", f"Fit `TunedThresholdClassifierCV({LR}, scoring={sc!r}, cv=5)` on the training part and return "
                        f"`best_score_` (the cross-validated {sc} at the best threshold).",
         f"{TUNED}{LRI}result = TunedThresholdClassifierCV({LR}, scoring={sc!r}, cv=5).fit(X_tr, y_tr).best_score_",
         {"require": ["TunedThresholdClassifierCV", "best_score_"]}),
        ("TunedThresholdClassifierCV", f"Fit `TunedThresholdClassifierCV({LR}, scoring={sc!r}, cv=5)` on the training "
                                       f"part and return its predictions for the first 10 test rows.",
         f"{TUNED}{LRI}result = TunedThresholdClassifierCV({LR}, scoring={sc!r}, cv=5).fit(X_tr, y_tr).predict(X_te[:10])",
         {"require": ["TunedThresholdClassifierCV", "predict"]}),
        ("FixedThresholdClassifier", f"A threshold changes labels, not probabilities. Fit `FixedThresholdClassifier("
                                     f"{LR}, threshold={t})` on the training part and return `predict_proba` of the first "
                                     f"3 test rows (the same as the plain model's).",
         f"{FIXED}{LRI}result = FixedThresholdClassifier({LR}, threshold={t}).fit(X_tr, y_tr).predict_proba(X_te[:3])",
         {"require": ["FixedThresholdClassifier", "predict_proba"]}),
        ("FrozenEstimator", "`fitted = GaussianNB().fit(X_tr, y_tr)`. Wrapping it in `FrozenEstimator` (from "
                            "`sklearn.frozen`) makes `fit` a no-op. Call `FrozenEstimator(fitted).fit(X_te, y_te)` and "
                            "return its `predict_proba(X_te[:3])`: still the original model's.",
         f"{FROZEN}{NB}fitted = GaussianNB().fit(X_tr, y_tr)\nresult = FrozenEstimator(fitted).fit(X_te, y_te).predict_proba(X_te[:3])"),
        ("FrozenEstimator", "Calibrate an already-trained model: `fitted = GaussianNB().fit(X_tr, y_tr)`; fit "
                            "`CalibratedClassifierCV(FrozenEstimator(fitted), method='sigmoid')` on the first half of the "
                            "test rows and return P(class 1) for the last 5 test rows.",
         f"{CAL}{FROZEN}{NB}fitted = GaussianNB().fit(X_tr, y_tr)\nh = len(X_te) // 2\n"
         "cal = CalibratedClassifierCV(FrozenEstimator(fitted), method='sigmoid').fit(X_te[:h], y_te[:h])\n"
         "result = cal.predict_proba(X_te[-5:])[:, 1]", {"require": ["FrozenEstimator", "CalibratedClassifierCV"]}),
    ]
    return catalog(rng, items, setup, f"{desc}.", tol=1e-9)


@template(M, "lite")
def probability_scores(rng):
    setup, desc = _imb(rng)
    items = [
        ("brier_score_loss", "The Brier score is the mean squared error of the predicted probabilities (lower is "
                             "better). Fit `GaussianNB()` on the training part and return `brier_score_loss(y_te, p)` "
                             "for its test P(class 1).",
         f"{BRIER}{NB}p = GaussianNB().fit(X_tr, y_tr).predict_proba(X_te)[:, 1]\nresult = brier_score_loss(y_te, p)"),
        ("log_loss", f"Fit `{LR}` on the training part and return the test `log_loss` of its probabilities (pass the "
                     f"full `predict_proba` matrix).",
         f"from sklearn.metrics import log_loss\n{LRI}result = log_loss(y_te, {LR}.fit(X_tr, y_tr).predict_proba(X_te))"),
        ("brier_score_loss", "Return the Brier score of the 'no-skill' forecast that always predicts the training rate "
                             "of class 1 (`np.full(len(y_te), y_tr.mean())`).",
         f"{BRIER}result = brier_score_loss(y_te, np.full(len(y_te), y_tr.mean()))"),
        ("predict_proba", f"Calibration-in-the-large: fit `{LR}` on the training part and return `(mean predicted "
                          f"P(class 1) on the test set, actual rate of class 1 in y_te)`.",
         f"{LRI}p = {LR}.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]\nresult = (p.mean(), y_te.mean())"),
    ]
    return catalog(rng, items, setup, f"{desc}.", tol=1e-9)


# ================================================================= easy

@template(M, "easy")
def reliability_summary(rng):
    setup, desc = _imb(rng)
    nb = rng.choice([5, 8, 10])
    model, imp = rng.choice([("GaussianNB()", NB), (LR, LRI)])
    task = rng.choice(["bins", "gap", "mean_gap", "monotone"])
    head = (f"{desc}. Fit `{model}` on the training part, take its test P(class 1) `p`, and compute "
            f"`prob_true, prob_pred = calibration_curve(y_te, p, n_bins={nb})`.")
    sol = f"{CURVE}{imp}p = {model}.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]\nprob_true, prob_pred = calibration_curve(y_te, p, n_bins={nb})\n"
    if task == "bins":
        return P(f"{head} Empty bins are dropped: return how many bins are left.", setup, sol + "result = len(prob_true)",
                 require=["calibration_curve"])
    if task == "gap":
        return P(f"{head} Return the largest |prob_true − prob_pred| over the bins (the worst-calibrated bin).", setup,
                 sol + "result = float(np.abs(prob_true - prob_pred).max())", require=["calibration_curve"], tol=1e-9)
    if task == "mean_gap":
        return P(f"{head} Return the mean of `prob_true − prob_pred` (negative = the model is over-confident about "
                 f"class 1 on average over bins).", setup, sol + "result = float((prob_true - prob_pred).mean())",
                 require=["calibration_curve"], tol=1e-9)
    return P(f"{head} Return `(bool(np.all(np.diff(prob_pred) > 0)), bool(np.all(np.diff(prob_true) >= 0)))`: bins "
             f"are ordered by predicted probability, but the observed frequencies need not be.", setup,
             sol + "result = (bool(np.all(np.diff(prob_pred) > 0)), bool(np.all(np.diff(prob_true) >= 0)))",
             require=["calibration_curve"])


@template(M, "easy")
def calibration_curve_by_hand(rng):
    setup, desc = _imb(rng)
    nb = rng.choice([4, 5, 8])
    what = rng.choice(["true", "pred"])
    col = "the observed rate of class 1 (`prob_true`)" if what == "true" else "the mean predicted probability (`prob_pred`)"
    val = "y_te" if what == "true" else "p"
    return P(f"{desc}. Fit `GaussianNB()` on the training part; `p` = its test P(class 1). Rebuild "
             f"`calibration_curve(y_te, p, n_bins={nb})` **by hand**: the bin edges are `np.linspace(0, 1, {nb + 1})`, "
             f"a probability goes to bin `np.searchsorted(edges[1:-1], p)`, and each non-empty bin reports {col}. "
             f"Return that array.", setup,
             f"{NB}p = GaussianNB().fit(X_tr, y_tr).predict_proba(X_te)[:, 1]\nedges = np.linspace(0, 1, {nb + 1})\n"
             f"b = np.searchsorted(edges[1:-1], p)\ncnt = np.bincount(b, minlength={nb})\n"
             f"tot = np.bincount(b, weights={val}, minlength={nb})\nresult = tot[cnt > 0] / cnt[cnt > 0]",
             require=["searchsorted"], tol=1e-9)


@template(M, "easy")
def fixed_threshold_by_hand(rng):
    setup, desc = _imb(rng)
    t = rng.choice([0.15, 0.25, 0.35, 0.45, 0.65])
    task = rng.choice(["same", "changed", "proba"])
    if task == "same":
        return P(f"{desc}. Return `(by_hand, wrapped)`: the test predictions of `{LR}` fitted on the training part with "
                 f"the rule P(class 1) ≥ {t} applied by hand (as ints), and those of `FixedThresholdClassifier({LR}, "
                 f"threshold={t})`.", setup,
                 f"{FIXED}{LRI}p = {LR}.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]\n"
                 f"result = ((p >= {t}).astype(int), FixedThresholdClassifier({LR}, threshold={t}).fit(X_tr, y_tr).predict(X_te))",
                 require=["FixedThresholdClassifier", "predict_proba"])
    if task == "changed":
        return P(f"{desc}. How many test predictions change when the cut-off moves from 0.5 to {t}? Compare `{LR}` with "
                 f"`FixedThresholdClassifier({LR}, threshold={t})`, both fitted on the training part.", setup,
                 f"{FIXED}{LRI}a = {LR}.fit(X_tr, y_tr).predict(X_te)\n"
                 f"b = FixedThresholdClassifier({LR}, threshold={t}).fit(X_tr, y_tr).predict(X_te)\nresult = int((a != b).sum())",
                 require=["FixedThresholdClassifier"])
    return P(f"{desc}. The wrapped model is in `estimator_`. Fit `clf = FixedThresholdClassifier({LR}, threshold={t})` "
             f"on the training part and return `(clf.estimator_.coef_, the fraction of test rows predicted as 1)`.",
             setup, f"{FIXED}{LRI}clf = FixedThresholdClassifier({LR}, threshold={t}).fit(X_tr, y_tr)\n"
                    "result = (clf.estimator_.coef_, float(clf.predict(X_te).mean()))",
             require=["FixedThresholdClassifier", "estimator_"], tol=1e-6)


@template(M, "easy")
def sigmoid_vs_isotonic(rng):
    setup, desc = _imb(rng)
    cv = rng.choice([3, 5])
    metric = rng.choice(["brier", "logloss", "unique"])
    sol = (f"{CAL}{NB}raw = GaussianNB().fit(X_tr, y_tr).predict_proba(X_te)[:, 1]\n"
           f"sig = CalibratedClassifierCV(GaussianNB(), method='sigmoid', cv={cv}).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]\n"
           f"iso = CalibratedClassifierCV(GaussianNB(), method='isotonic', cv={cv}).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]\n")
    head = (f"{desc}. Compare the test P(class 1) of a raw `GaussianNB()` with two calibrated versions, "
            f"`CalibratedClassifierCV(GaussianNB(), method=..., cv={cv})` with `'sigmoid'` and `'isotonic'` (all fitted "
            f"on the training part).")
    if metric == "brier":
        return P(f"{head} Return the three Brier scores `(raw, sigmoid, isotonic)` (`brier_score_loss`).", setup,
                 BRIER + sol + "result = tuple(brier_score_loss(y_te, q) for q in (raw, sig, iso))",
                 require=["CalibratedClassifierCV", "brier_score_loss"], tol=1e-9)
    if metric == "logloss":
        return P(f"{head} Return the three test log losses `(raw, sigmoid, isotonic)` (`log_loss(y_te, q)` with the "
                 f"P(class 1) vectors).", setup,
                 "from sklearn.metrics import log_loss\n" + sol + "result = tuple(log_loss(y_te, q) for q in (raw, sig, iso))",
                 require=["CalibratedClassifierCV", "log_loss"], tol=1e-9)
    return P(f"{head} Isotonic calibration is a step function. Return how many **distinct** probability values each "
             f"produces on the test set, as `(raw, sigmoid, isotonic)`.", setup,
             sol + "result = tuple(len(np.unique(np.round(q, 12))) for q in (raw, sig, iso))",
             require=["CalibratedClassifierCV"])


@template(M, "easy")
def tuned_threshold_scoring(rng):
    setup, desc = _imb(rng)
    task = rng.choice(["pair", "fbeta", "vs_default", "thresholds"])
    sc = rng.choice(["f1", "balanced_accuracy", "accuracy"])
    head = f"{desc}."
    if task == "pair":
        return P(f"{head} Fit `TunedThresholdClassifierCV({LR}, scoring={sc!r}, cv=5)` on the training part and return "
                 f"`(best_threshold_, best_score_)`.", setup,
                 f"{TUNED}{LRI}t = TunedThresholdClassifierCV({LR}, scoring={sc!r}, cv=5).fit(X_tr, y_tr)\n"
                 "result = (t.best_threshold_, t.best_score_)", require=["TunedThresholdClassifierCV"], tol=1e-9)
    if task == "fbeta":
        b = rng.choice([2, 3])
        return P(f"{head} Missing a positive is worse than a false alarm: tune the threshold of `{LR}` for F{b} with "
                 f"`TunedThresholdClassifierCV(..., scoring=make_scorer(fbeta_score, beta={b}), cv=5)` on the training "
                 f"part. Return `best_threshold_`.", setup,
                 f"{TUNED}{LRI}from sklearn.metrics import make_scorer, fbeta_score\n"
                 f"t = TunedThresholdClassifierCV({LR}, scoring=make_scorer(fbeta_score, beta={b}), cv=5).fit(X_tr, y_tr)\n"
                 "result = t.best_threshold_", require=["TunedThresholdClassifierCV", "make_scorer"], tol=1e-9)
    if task == "vs_default":
        return P(f"{head} Return the test F1 of `{LR}` (cut-off 0.5) and of `TunedThresholdClassifierCV({LR}, "
                 f"scoring='f1', cv=5)`, both fitted on the training part, as `(default_f1, tuned_f1)`.", setup,
                 f"{TUNED}{LRI}from sklearn.metrics import f1_score\n"
                 f"a = f1_score(y_te, {LR}.fit(X_tr, y_tr).predict(X_te))\n"
                 f"b = f1_score(y_te, TunedThresholdClassifierCV({LR}, scoring='f1', cv=5).fit(X_tr, y_tr).predict(X_te))\n"
                 "result = (a, b)", require=["TunedThresholdClassifierCV", "f1_score"], tol=1e-9)
    k = rng.choice([10, 20, 50])
    return P(f"{head} `thresholds={k}` makes `TunedThresholdClassifierCV` try only {k} cut-offs. Fit it on `{LR}` with "
             f"`scoring={sc!r}`, `cv=5`, `thresholds={k}`, `store_cv_results=True` on the training part and return "
             f"`(len(cv_results_['thresholds']), best_threshold_)`.", setup,
             f"{TUNED}{LRI}t = TunedThresholdClassifierCV({LR}, scoring={sc!r}, cv=5, thresholds={k}, store_cv_results=True).fit(X_tr, y_tr)\n"
             "result = (len(t.cv_results_['thresholds']), t.best_threshold_)",
             require=["TunedThresholdClassifierCV", "store_cv_results"], tol=1e-9)


@template(M, "easy")
def frozen_model_calibration(rng):
    setup, desc = _three_way(rng)
    m = rng.choice(["sigmoid", "isotonic"])
    task = rng.choice(["probs", "untouched", "brier"])
    head = f"{desc}. `model = GaussianNB().fit(X_fit, y_fit)` is already trained and must not be retrained."
    setup += "\nfrom sklearn.naive_bayes import GaussianNB\nmodel = GaussianNB().fit(X_fit, y_fit)"
    sol = f"{CAL}{FROZEN}cal = CalibratedClassifierCV(FrozenEstimator(model), method={m!r}).fit(X_cal, y_cal)\n"
    if task == "probs":
        return P(f"{head} Calibrate it on the calibration set with `CalibratedClassifierCV(FrozenEstimator(model), "
                 f"method={m!r})` and return P(class 1) for the first 6 test rows.", setup,
                 sol + "result = cal.predict_proba(X_te[:6])[:, 1]", require=["FrozenEstimator", "CalibratedClassifierCV"],
                 tol=1e-9)
    if task == "untouched":
        return P(f"{head} Calibrate it on the calibration set with `CalibratedClassifierCV(FrozenEstimator(model), "
                 f"method={m!r})`, then check that `model` itself did not change: return `(model.theta_, "
                 f"cal.predict(X_te[:6]))` (`theta_` = the class means it learned from `X_fit`).", setup,
                 f"before = model.theta_.copy()\n{sol}result = (model.theta_, cal.predict(X_te[:6]))",
                 require=["FrozenEstimator"], tol=1e-9)
    return P(f"{head} Calibrate it on the calibration set (`CalibratedClassifierCV(FrozenEstimator(model), "
             f"method={m!r})`) and return the test Brier scores `(before, after)`.", setup,
             BRIER + sol + "result = (brier_score_loss(y_te, model.predict_proba(X_te)[:, 1]), "
                           "brier_score_loss(y_te, cal.predict_proba(X_te)[:, 1]))",
             require=["FrozenEstimator", "brier_score_loss"], tol=1e-9)


@template(M, "easy")
def threshold_sweep_counts(rng):
    setup, desc = _imb(rng)
    ts = sorted(rng.sample([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], 4))
    task = rng.choice(["positives", "precision", "recall"])
    sol = f"{LRI}p = {LR}.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]\n"
    head = f"{desc}. Fit `{LR}` on the training part; `p` = its test P(class 1). For each threshold `t` in `{ts}` predict 1 when `p >= t`"
    if task == "positives":
        return P(f"{head} and return the number of predicted positives per threshold (a list of ints; it can only "
                 f"shrink).", setup, sol + f"result = [int((p >= t).sum()) for t in {ts}]", require=["predict_proba"])
    if task == "precision":
        return P(f"{head} and return the test precision per threshold (`precision_score(..., zero_division=0)`).", setup,
                 sol + f"from sklearn.metrics import precision_score\n"
                       f"result = [precision_score(y_te, (p >= t).astype(int), zero_division=0) for t in {ts}]",
                 require=["precision_score"], tol=1e-9)
    return P(f"{head} and return the test recall per threshold (`recall_score`).", setup,
             sol + f"from sklearn.metrics import recall_score\nresult = [recall_score(y_te, (p >= t).astype(int)) for t in {ts}]",
             require=["recall_score"], tol=1e-9)


@template(M, "easy")
def calibration_in_the_large(rng):
    setup, desc = _imb(rng)
    if rng.random() < 0.5:
        return P(f"{desc}. Naive Bayes is often over-confident. Return `(raw_mean, calibrated_mean, actual_rate)`: the "
                 f"mean test P(class 1) of `GaussianNB()` and of `CalibratedClassifierCV(GaussianNB(), "
                 f"method='isotonic', cv=5)` (both fitted on the training part), and `y_te.mean()`.", setup,
                 f"{CAL}{NB}raw = GaussianNB().fit(X_tr, y_tr).predict_proba(X_te)[:, 1]\n"
                 "cal = CalibratedClassifierCV(GaussianNB(), method='isotonic', cv=5).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]\n"
                 "result = (raw.mean(), cal.mean(), y_te.mean())", require=["CalibratedClassifierCV"], tol=1e-9)
    return P(f"{desc}. Return the fraction of test rows to which `GaussianNB()` (fitted on the training part) gives "
             f"P(class 1) above 0.99 or below 0.01, and the same fraction for `CalibratedClassifierCV(GaussianNB(), "
             f"method='sigmoid', cv=5)`, as a tuple: calibration tames extreme probabilities.", setup,
             f"{CAL}{NB}raw = GaussianNB().fit(X_tr, y_tr).predict_proba(X_te)[:, 1]\n"
             "cal = CalibratedClassifierCV(GaussianNB(), method='sigmoid', cv=5).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]\n"
             "ext = lambda q: float(((q > 0.99) | (q < 0.01)).mean())\nresult = (ext(raw), ext(cal))",
             require=["CalibratedClassifierCV"], tol=1e-9)


@template(M, "easy")
def decision_function_threshold(rng):
    setup, desc = _imb(rng)
    m = rng.choice([-1.0, -0.5, 0.5, 1.0])
    if rng.random() < 0.5:
        return P(f"{desc}. A threshold can apply to the decision function instead of probabilities. Fit "
                 f"`FixedThresholdClassifier({LR}, threshold=0.0, response_method='decision_function')` on the training "
                 f"part and return `(its test predictions, the plain model's test predictions)`: a margin of 0 is the "
                 f"default rule.", setup,
                 f"{FIXED}{LRI}a = FixedThresholdClassifier({LR}, threshold=0.0, response_method='decision_function').fit(X_tr, y_tr).predict(X_te)\n"
                 f"result = (a, {LR}.fit(X_tr, y_tr).predict(X_te))", require=["FixedThresholdClassifier", "response_method"])
    return P(f"{desc}. Fit `FixedThresholdClassifier({LR}, threshold={m}, response_method='decision_function')` on the "
             f"training part. Which probability cut-off is that? For logistic regression P = 1 / (1 + e^(−margin)). "
             f"Return `(number of test rows predicted 1, the equivalent probability threshold)`.", setup,
             f"{FIXED}{LRI}clf = FixedThresholdClassifier({LR}, threshold={m}, response_method='decision_function').fit(X_tr, y_tr)\n"
             f"result = (int(clf.predict(X_te).sum()), 1 / (1 + np.exp(-{m})))",
             require=["FixedThresholdClassifier", "response_method"], tol=1e-9)


@template(M, "easy")
def cost_based_cutoff(rng):
    setup, desc = _imb(rng)
    c_fn, c_fp = rng.choice([(5, 1), (10, 1), (4, 1), (1, 2)])
    t = round(c_fp / (c_fp + c_fn), 6)
    task = rng.choice(["count", "cost"])
    head = (f"{desc}. A missed positive costs {c_fn}, a false alarm costs {c_fp}. With calibrated probabilities the "
            f"cheapest rule predicts 1 when P(class 1) ≥ c_fp / (c_fp + c_fn).")
    sol = (f"{FIXED}{CAL}{NB}cal = CalibratedClassifierCV(GaussianNB(), method='isotonic', cv=5)\n"
           f"clf = FixedThresholdClassifier(cal, threshold={c_fp} / ({c_fp} + {c_fn})).fit(X_tr, y_tr)\npred = clf.predict(X_te)\n")
    if task == "count":
        return P(f"{head} Wrap `CalibratedClassifierCV(GaussianNB(), method='isotonic', cv=5)` in a "
                 f"`FixedThresholdClassifier` with that threshold, fit it on the training part, and return the number of "
                 f"test rows predicted as 1.", setup, sol + "result = int(pred.sum())",
                 require=["FixedThresholdClassifier", "CalibratedClassifierCV"])
    return P(f"{head} Wrap `CalibratedClassifierCV(GaussianNB(), method='isotonic', cv=5)` in a "
             f"`FixedThresholdClassifier` with that threshold (fit on the training part) and return the total test cost "
             f"`(cost at that threshold, cost at 0.5)`, using the same calibrated model for both.", setup,
             sol + f"p = clf.estimator_.predict_proba(X_te)[:, 1]\n"
                   f"cost = lambda yp: {c_fn} * int(((y_te == 1) & (yp == 0)).sum()) + {c_fp} * int(((y_te == 0) & (yp == 1)).sum())\n"
                   "result = (cost(pred), cost((p >= 0.5).astype(int)))", require=["FixedThresholdClassifier"])


# ================================================================= medium

@template(M, "medium")
def reliability_before_after(rng):
    setup, desc = _imb(rng, n=(400, 650))
    nb = rng.choice([5, 8])
    base, imp, name = rng.choice([("GaussianNB()", NB, "naive Bayes"),
                                  (f"RandomForestClassifier(n_estimators=30, random_state=0)",
                                   "from sklearn.ensemble import RandomForestClassifier\n", "random forest")])
    task = rng.choice(["gap", "brier", "curves", "bins"])
    sol = (f"{CAL}{CURVE}{imp}raw = {base}.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]\n"
           f"cal = CalibratedClassifierCV({base}, method='isotonic', cv=3).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]\n")
    head = (f"{desc}. Compare the {name} `{base}` with its isotonic calibration `CalibratedClassifierCV({base}, "
            f"method='isotonic', cv=3)`, both fitted on the training part, through their test P(class 1).")
    if task == "gap":
        return P(f"{head} For each, compute `calibration_curve(y_te, q, n_bins={nb})` and the mean |prob_true − "
                 f"prob_pred| over its bins. Return `(before, after)`.", setup,
                 sol + f"gap = lambda q: float(np.abs(np.subtract(*calibration_curve(y_te, q, n_bins={nb}))).mean())\n"
                       "result = (gap(raw), gap(cal))", require=["calibration_curve", "CalibratedClassifierCV"], tol=1e-9)
    if task == "brier":
        return P(f"{head} Return the Brier scores `(before, after)`.", setup,
                 BRIER + sol + "result = (brier_score_loss(y_te, raw), brier_score_loss(y_te, cal))",
                 require=["CalibratedClassifierCV", "brier_score_loss"], tol=1e-9)
    if task == "curves":
        return P(f"{head} Return the observed rates `prob_true` of `calibration_curve(..., n_bins={nb})` for both, as "
                 f"`(before, after)`.", setup,
                 sol + f"result = (calibration_curve(y_te, raw, n_bins={nb})[0], calibration_curve(y_te, cal, n_bins={nb})[0])",
                 require=["calibration_curve", "CalibratedClassifierCV"], tol=1e-9)
    return P(f"{head} Return, for both, the number of test rows whose probability falls in the most extreme bins "
             f"(below 0.1 or at least 0.9), as `(before, after)`.", setup,
             sol + "ext = lambda q: int(((q < 0.1) | (q >= 0.9)).sum())\nresult = (ext(raw), ext(cal))",
             require=["CalibratedClassifierCV"])


@template(M, "medium")
def recall_target_threshold(rng):
    setup, desc = _three_way(rng)
    r = rng.choice([0.8, 0.85, 0.9, 0.95])
    task = rng.choice(["val", "test", "curve"])
    sol = (f"{LRI}model = {LR}.fit(X_fit, y_fit)\np = model.predict_proba(X_cal)[:, 1]\nts = np.unique(p)\n"
           f"rec = np.array([(p[y_cal == 1] >= t).mean() for t in ts])\nt = ts[rec >= {r}].max()\n"
           "prec = y_cal[p >= t].mean()\n")
    head = (f"{desc}. Fit `{LR}` on the fitting set. On the calibration set, find the **largest** threshold `t` among "
            f"the distinct values of P(class 1) for which predicting 1 when P ≥ t keeps recall ≥ {r}.")
    if task == "val":
        return P(f"{head} Return `(t, precision at t on the calibration set)`.", setup, sol + "result = (float(t), float(prec))",
                 require=["predict_proba"], tol=1e-9)
    if task == "test":
        return P(f"{head} Then apply it on the test set with `FixedThresholdClassifier(FrozenEstimator(model), "
                 f"threshold=t)` and return the test `(precision, recall)`: does the target still hold?", setup,
                 sol + f"{FIXED}{FROZEN}from sklearn.metrics import precision_score, recall_score\n"
                       "pred = FixedThresholdClassifier(FrozenEstimator(model), threshold=t).fit(X_cal, y_cal).predict(X_te)\n"
                       "result = (precision_score(y_te, pred, zero_division=0), recall_score(y_te, pred))",
                 require=["FixedThresholdClassifier", "FrozenEstimator"], tol=1e-9)
    return P(f"{desc}. Fit `{LR}` on the fitting set and get `precision, recall, thresholds = "
             f"precision_recall_curve(y_cal, p)` for its calibration-set P(class 1) `p`. Among thresholds with recall "
             f"≥ {r}, return `(threshold, precision)` of the one with the highest precision (the first on ties). (Drop "
             f"the curve's last point, which has no threshold.)", setup,
             f"{LRI}from sklearn.metrics import precision_recall_curve\np = {LR}.fit(X_fit, y_fit).predict_proba(X_cal)[:, 1]\n"
             "prec, rec, ts = precision_recall_curve(y_cal, p)\nprec, rec = prec[:-1], rec[:-1]\n"
             f"ok = np.flatnonzero(rec >= {r})\ni = ok[np.argmax(prec[ok])]\nresult = (float(ts[i]), float(prec[i]))",
             require=["precision_recall_curve"], tol=1e-9)


@template(M, "medium")
def wrap_trained_model(rng):
    setup, desc = _three_way(rng)
    t = rng.choice([0.1, 0.2, 0.3])
    task = rng.choice(["predict", "recall", "confusion"])
    setup += f"\n{LRI.strip()}\nmodel = {LR}.fit(X_fit, y_fit)"
    head = (f"{desc}. `model = {LR}` was trained on the fitting set. Wrap it, without retraining, with a fixed {t} "
            f"cut-off: `FixedThresholdClassifier(FrozenEstimator(model), threshold={t})` (calling `fit` on the wrapper "
            f"leaves `model` as it is; fit it on the calibration set).")
    sol = f"{FIXED}{FROZEN}clf = FixedThresholdClassifier(FrozenEstimator(model), threshold={t}).fit(X_cal, y_cal)\n"
    if task == "predict":
        return P(f"{head} Return the wrapper's predictions for the first 12 test rows.", setup,
                 sol + "result = clf.predict(X_te[:12])", require=["FixedThresholdClassifier", "FrozenEstimator"])
    if task == "recall":
        return P(f"{head} Return the test recall `(at 0.5, at {t})`.", setup,
                 sol + "from sklearn.metrics import recall_score\nresult = (recall_score(y_te, model.predict(X_te)), "
                       "recall_score(y_te, clf.predict(X_te)))", require=["FixedThresholdClassifier", "recall_score"],
                 tol=1e-9)
    return P(f"{head} Return the test confusion matrix of the wrapper (`confusion_matrix(y_te, ...)`).", setup,
             sol + "from sklearn.metrics import confusion_matrix\nresult = confusion_matrix(y_te, clf.predict(X_te))",
             require=["FixedThresholdClassifier", "confusion_matrix"])


@template(M, "medium")
def calibration_on_table(rng):
    t = table(rng, kind="clf", numeric_only=True, n=(300, 500))
    s = seeded(rng)
    setup = (t.setup + "\nfrom sklearn.model_selection import train_test_split\n"
             f"X_tr, X_te, y_tr, y_te = train_test_split(df, y, stratify=y, random_state={s})")
    rf = "RandomForestClassifier(n_estimators=30, random_state=0)"
    rfi = "from sklearn.ensemble import RandomForestClassifier\n"
    items = [
        ("CalibratedClassifierCV", f"Does calibration help a forest? Return the test Brier scores `(raw, calibrated)` "
                                   f"of `{rf}` and of `CalibratedClassifierCV({rf}, method='isotonic', cv=3)`, both "
                                   f"fitted on the training part.",
         f"{CAL}{BRIER}{rfi}raw = {rf}.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]\n"
         f"cal = CalibratedClassifierCV({rf}, method='isotonic', cv=3).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]\n"
         "result = (brier_score_loss(y_te, raw), brier_score_loss(y_te, cal))"),
        ("LinearSVC", "`LinearSVC` has no `predict_proba`; calibration gives it one. Fit "
                      "`CalibratedClassifierCV(make_pipeline(StandardScaler(), LinearSVC()), cv=3)` (sigmoid by default) "
                      "on the training part and return P(class 1) of the first 5 test rows.",
         f"{CAL}from sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\n"
         "from sklearn.svm import LinearSVC\ncal = CalibratedClassifierCV(make_pipeline(StandardScaler(), LinearSVC()), cv=3).fit(X_tr, y_tr)\n"
         "result = cal.predict_proba(X_te.iloc[:5])[:, 1]", {"require": ["CalibratedClassifierCV", "LinearSVC"]}),
        ("calibration_curve", f"Fit `make_pipeline(StandardScaler(), {LR})` on the training part and return "
                              f"`calibration_curve(y_te, p, n_bins=5, strategy='quantile')` of its test P(class 1).",
         f"{CURVE}from sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\n{LRI}"
         f"p = make_pipeline(StandardScaler(), {LR}).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]\n"
         "result = calibration_curve(y_te, p, n_bins=5, strategy='quantile')"),
        ("log_loss", f"Return the test log loss `(raw, sigmoid)` of `GaussianNB()` and of `CalibratedClassifierCV("
                     f"GaussianNB(), method='sigmoid', cv=5)`, both fitted on the training part.",
         f"{CAL}{NB}from sklearn.metrics import log_loss\nraw = GaussianNB().fit(X_tr, y_tr).predict_proba(X_te)\n"
         "cal = CalibratedClassifierCV(GaussianNB(), method='sigmoid', cv=5).fit(X_tr, y_tr).predict_proba(X_te)\n"
         "result = (log_loss(y_te, raw), log_loss(y_te, cal))"),
        ("ensemble", "How different are the two ways of building a calibrated model? Fit `CalibratedClassifierCV("
                     "GaussianNB(), method='sigmoid', cv=5)` with `ensemble=True` and with `ensemble=False` on the "
                     "training part and return the largest absolute difference between their test P(class 1).",
         f"{CAL}{NB}a = CalibratedClassifierCV(GaussianNB(), method='sigmoid', cv=5, ensemble=True).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]\n"
         "b = CalibratedClassifierCV(GaussianNB(), method='sigmoid', cv=5, ensemble=False).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]\n"
         "result = float(np.abs(a - b).max())"),
        ("predict_proba", f"Return the mean test P(class 1) of `GaussianNB()` and of `CalibratedClassifierCV("
                          f"GaussianNB(), method='isotonic', cv=5)` (both fitted on the training part) next to the "
                          f"actual test rate, as a 3-tuple.",
         f"{CAL}{NB}raw = GaussianNB().fit(X_tr, y_tr).predict_proba(X_te)[:, 1]\n"
         "cal = CalibratedClassifierCV(GaussianNB(), method='isotonic', cv=5).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]\n"
         "result = (raw.mean(), cal.mean(), y_te.mean())", {"require": ["CalibratedClassifierCV", "predict_proba"]}),
        ("calibration_curve", "Fit `GaussianNB()` on the training part and return its expected calibration error on "
                              "the test set with 5 uniform bins: Σ over non-empty bins of (bin count / n) × "
                              "|prob_true − prob_pred|, using `calibration_curve` for the per-bin rates and "
                              "`np.searchsorted(np.linspace(0, 1, 6)[1:-1], p)` for the bin counts.",
         f"{CURVE}{NB}p = GaussianNB().fit(X_tr, y_tr).predict_proba(X_te)[:, 1]\n"
         "pt, pp = calibration_curve(y_te, p, n_bins=5)\ncnt = np.bincount(np.searchsorted(np.linspace(0, 1, 6)[1:-1], p), minlength=5)\n"
         "result = float((cnt[cnt > 0] / len(p) * np.abs(pt - pp)).sum())"),
    ]
    return catalog(rng, items, setup, f"{t.desc}. It was split with `train_test_split(df, y, stratify=y, "
                                      f"random_state={s})` into `X_tr, X_te, y_tr, y_te`.", tol=1e-9)


@template(M, "medium")
def threshold_on_table(rng):
    t = table(rng, kind="clf", numeric_only=True, n=(300, 500), only=["churn", "loans", "patients", "flights"])
    s = seeded(rng)
    setup = (t.setup + "\nfrom sklearn.model_selection import train_test_split\n"
             f"X_tr, X_te, y_tr, y_te = train_test_split(df, y, stratify=y, random_state={s})\n"
             "from sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\n"
             "from sklearn.linear_model import LogisticRegression\n"
             "model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))")
    gain_tp, cost_fp = rng.choice([(10, 1), (5, 2), (20, 3)])
    items = [
        ("TunedThresholdClassifierCV", "Tune the cut-off of `model` for balanced accuracy "
                                       "(`TunedThresholdClassifierCV(model, scoring='balanced_accuracy', cv=5)`) on the "
                                       "training part. Return `(best_threshold_, test balanced accuracy of the tuned "
                                       "model, test balanced accuracy of model at 0.5)`.",
         f"{TUNED}from sklearn.metrics import balanced_accuracy_score\n"
         "tuned = TunedThresholdClassifierCV(model, scoring='balanced_accuracy', cv=5).fit(X_tr, y_tr)\n"
         "plain = model.fit(X_tr, y_tr)\n"
         "result = (tuned.best_threshold_, balanced_accuracy_score(y_te, tuned.predict(X_te)), balanced_accuracy_score(y_te, plain.predict(X_te)))"),
        ("TunedThresholdClassifierCV", "Tune the cut-off of `model` for F1 (`cv=5`, training part) and return the test "
                                       "F1 `(at 0.5, tuned)`.",
         f"{TUNED}from sklearn.metrics import f1_score\ntuned = TunedThresholdClassifierCV(model, scoring='f1', cv=5).fit(X_tr, y_tr)\n"
         "result = (f1_score(y_te, model.fit(X_tr, y_tr).predict(X_te)), f1_score(y_te, tuned.predict(X_te)))"),
        ("store_cv_results", "Fit `TunedThresholdClassifierCV(model, scoring='f1', cv=5, store_cv_results=True)` on the "
                             "training part. `cv_results_` holds the tried `'thresholds'` and their mean `'scores'`. "
                             "Return `(len(thresholds), scores.max() == best_score_)`.",
         f"{TUNED}tuned = TunedThresholdClassifierCV(model, scoring='f1', cv=5, store_cv_results=True).fit(X_tr, y_tr)\n"
         "result = (len(tuned.cv_results_['thresholds']), bool(tuned.cv_results_['scores'].max() == tuned.best_score_))"),
        ("make_scorer", f"Business value: each caught positive earns {gain_tp}, each false alarm costs {cost_fp}. Write "
                        f"`def gain(y_true, y_pred)` returning the total, tune the cut-off with "
                        f"`TunedThresholdClassifierCV(model, scoring=make_scorer(gain), cv=5)` on the training part, and "
                        f"return `(best_threshold_, test gain of the tuned model)`.",
         f"{TUNED}from sklearn.metrics import make_scorer\n"
         f"def gain(y_true, y_pred):\n    return {gain_tp} * int(((y_true == 1) & (y_pred == 1)).sum()) - {cost_fp} * int(((y_true == 0) & (y_pred == 1)).sum())\n"
         "tuned = TunedThresholdClassifierCV(model, scoring=make_scorer(gain), cv=5).fit(X_tr, y_tr)\n"
         "result = (tuned.best_threshold_, gain(y_te, tuned.predict(X_te)))"),
        ("FixedThresholdClassifier", "Return the test confusion matrix of `FixedThresholdClassifier(model, "
                                     "threshold=0.3)` fitted on the training part.",
         f"{FIXED}from sklearn.metrics import confusion_matrix\n"
         "result = confusion_matrix(y_te, FixedThresholdClassifier(model, threshold=0.3).fit(X_tr, y_tr).predict(X_te))"),
        ("FixedThresholdClassifier", "How does the share of flagged rows react to the cut-off? For `t` in `[0.2, 0.35, "
                                     "0.5, 0.65]` fit `FixedThresholdClassifier(model, threshold=t)` on the training "
                                     "part and return the fraction of test rows predicted 1, as a list.",
         f"{FIXED}result = [float(FixedThresholdClassifier(model, threshold=t).fit(X_tr, y_tr).predict(X_te).mean()) for t in [0.2, 0.35, 0.5, 0.65]]"),
        ("thresholds", "Fewer candidates: fit `TunedThresholdClassifierCV(model, scoring='f1', cv=5, thresholds=np.linspace("
                       "0.1, 0.9, 9))` on the training part and return `best_threshold_`.",
         f"{TUNED}result = TunedThresholdClassifierCV(model, scoring='f1', cv=5, thresholds=np.linspace(0.1, 0.9, 9)).fit(X_tr, y_tr).best_threshold_",
         {"require": ["TunedThresholdClassifierCV", "thresholds"]}),
    ]
    return catalog(rng, items, setup, f"{t.desc}. It was split with `train_test_split(df, y, stratify=y, random_state={s})` "
                                      f"into `X_tr, X_te, y_tr, y_te`; `model = make_pipeline(StandardScaler(), "
                                      f"LogisticRegression(max_iter=1000))` (not fitted).", tol=1e-9)


@template(M, "medium")
def calibrator_internals(rng):
    setup, desc = _three_way(rng)
    setup += f"\n{NB.strip()}\nmodel = GaussianNB().fit(X_fit, y_fit)"
    m = rng.randint(3, 6)
    if rng.random() < 0.5:
        return P(f"{desc}. `model = GaussianNB()` was trained on the fitting set. Calibrate it on the calibration set "
                 f"with `cal = CalibratedClassifierCV(FrozenEstimator(model), method='sigmoid')`. The fitted sigmoid is "
                 f"`c = cal.calibrated_classifiers_[0].calibrators[0]` with parameters `c.a_`, `c.b_`, and P(class 1) = "
                 f"1 / (1 + exp(a·s + b)) where s = the model's P(class 1). Recompute it by hand for the first {m} test "
                 f"rows and return the array (it equals `cal.predict_proba`).", setup,
                 f"{CAL}{FROZEN}cal = CalibratedClassifierCV(FrozenEstimator(model), method='sigmoid').fit(X_cal, y_cal)\n"
                 "c = cal.calibrated_classifiers_[0].calibrators[0]\n"
                 f"s = model.predict_proba(X_te[:{m}])[:, 1]\nresult = 1 / (1 + np.exp(c.a_ * s + c.b_))",
                 require=["calibrated_classifiers_", "FrozenEstimator"], tol=1e-9)
    return P(f"{desc}. `model = GaussianNB()` was trained on the fitting set. Isotonic calibration is an "
             f"`IsotonicRegression(out_of_bounds='clip')` (`sklearn.isotonic`) fitted on (the model's calibration-set "
             f"P(class 1), `y_cal`). Fit one yourself and return its output for the test probabilities of the first "
             f"{m} test rows (it equals `CalibratedClassifierCV(FrozenEstimator(model), method='isotonic')` fitted on the "
             f"calibration set).", setup,
             "from sklearn.isotonic import IsotonicRegression\n"
             "iso = IsotonicRegression(out_of_bounds='clip').fit(model.predict_proba(X_cal)[:, 1], y_cal)\n"
             f"result = iso.predict(model.predict_proba(X_te[:{m}])[:, 1])", require=["IsotonicRegression"], tol=1e-9)


@template(M, "medium")
def isotonic_regression_basics(rng):
    n, s = rng.randint(12, 20), seeded(rng)
    setup = (f"g = np.random.default_rng({s})\nscore = np.sort(g.uniform(0, 1, {n})).round(2)\n"
             f"y = (g.uniform(0, 1, {n}) < score).astype(int)")
    head = (f"`score` holds {n} sorted model scores and `y` the true labels (0/1). `IsotonicRegression` (from "
            f"`sklearn.isotonic`) fits the best non-decreasing step function of `score` to `y`.")
    task = rng.choice(["fit", "levels", "predict", "clip"])
    sol = "from sklearn.isotonic import IsotonicRegression\niso = IsotonicRegression(out_of_bounds='clip').fit(score, y)\n"
    if task == "fit":
        return P(f"{head} Return its fitted values on the training scores (`IsotonicRegression(out_of_bounds='clip')`, "
                 f"then `predict(score)`).", setup, sol + "result = iso.predict(score)", require=["IsotonicRegression"],
                 tol=1e-9)
    if task == "levels":
        return P(f"{head} Return the sorted distinct fitted values (the steps) as an array.", setup,
                 sol + "result = np.unique(np.round(iso.predict(score), 12))", require=["IsotonicRegression"], tol=1e-9)
    if task == "predict":
        return P(f"{head} Fit `IsotonicRegression(out_of_bounds='clip')` and predict the new scores `[0.05, 0.3, 0.55, "
                 f"0.8]` (between known scores it interpolates linearly).", setup,
                 sol + "result = iso.predict([0.05, 0.3, 0.55, 0.8])", require=["IsotonicRegression"], tol=1e-9)
    return P(f"{head} Scores outside the training range: return `(clip, nan)` where `clip` is the prediction at `[-1.0, "
             f"2.0]` with `out_of_bounds='clip'` (as a list) and `nan` counts the NaNs that `out_of_bounds='nan'` gives "
             f"for the same two scores.", setup,
             sol + "other = IsotonicRegression(out_of_bounds='nan').fit(score, y)\n"
                   "result = (iso.predict([-1.0, 2.0]).tolist(), int(np.isnan(other.predict([-1.0, 2.0])).sum()))",
             require=["IsotonicRegression", "out_of_bounds"], tol=1e-9)


@template(M, "medium")
def brier_decomposition(rng):
    setup, desc = _imb(rng)
    task = rng.choice(["hand", "skill", "baseline"])
    sol = f"{NB}p = GaussianNB().fit(X_tr, y_tr).predict_proba(X_te)[:, 1]\n"
    if task == "hand":
        return P(f"{desc}. Fit `GaussianNB()` on the training part; `p` = its test P(class 1). Return `(by_hand, "
                 f"brier_score_loss(y_te, p))` with the by-hand Brier score = mean((p − y)²).", setup,
                 sol + BRIER + "result = (float(((p - y_te) ** 2).mean()), brier_score_loss(y_te, p))",
                 require=["brier_score_loss"], tol=1e-9)
    if task == "skill":
        return P(f"{desc}. Fit `GaussianNB()` on the training part. Return its Brier **skill** score on the test set: 1 − "
                 f"Brier(model) / Brier(always predicting the training rate `y_tr.mean()`). Positive = better than the "
                 f"base rate.", setup,
                 sol + BRIER + "result = 1 - brier_score_loss(y_te, p) / brier_score_loss(y_te, np.full(len(y_te), y_tr.mean()))",
                 require=["brier_score_loss"], tol=1e-9)
    return P(f"{desc}. A constant forecast `c` has Brier score r·(1 − c)² + (1 − r)·c² where r is the test rate of class "
             f"1, minimized at c = r. Return `(brier_score_loss(y_te, np.full(n, r)), r * (1 - r))` (equal).", setup,
             BRIER + "r = y_te.mean()\nresult = (brier_score_loss(y_te, np.full(len(y_te), r)), r * (1 - r))",
             require=["brier_score_loss"], tol=1e-9)


@template(M, "medium")
def multiclass_calibration(rng):
    data = rng.choice(["wine", "iris"])
    loader = "load_wine" if data == "wine" else "load_iris"
    setup = (f"from sklearn.datasets import {loader}\nfrom sklearn.model_selection import train_test_split\n"
             f"X, y = {loader}(return_X_y=True)\n"
             f"X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, random_state={seeded(rng)})")
    desc = f"`X`, `y` hold the {data} data (3 classes), split stratified into `X_tr, X_te, y_tr, y_te`"
    m = rng.choice(["sigmoid", "isotonic"])
    if rng.random() < 0.5:
        return P(f"{desc}. With several classes, `CalibratedClassifierCV` calibrates each class one-vs-rest and then "
                 f"renormalizes. Fit `CalibratedClassifierCV(GaussianNB(), method={m!r}, cv=3)` on the training part "
                 f"and return `(predict_proba(X_te[:4]), its row sums)`.", setup,
                 f"{CAL}{NB}P_ = CalibratedClassifierCV(GaussianNB(), method={m!r}, cv=3).fit(X_tr, y_tr).predict_proba(X_te[:4])\n"
                 "result = (P_, P_.sum(axis=1))", require=["CalibratedClassifierCV"], tol=1e-9)
    return P(f"{desc}. Return the test log loss `(raw, calibrated)` of `GaussianNB()` and of "
             f"`CalibratedClassifierCV(GaussianNB(), method={m!r}, cv=3)`, both fitted on the training part.", setup,
             f"{CAL}{NB}from sklearn.metrics import log_loss\nraw = GaussianNB().fit(X_tr, y_tr).predict_proba(X_te)\n"
             f"cal = CalibratedClassifierCV(GaussianNB(), method={m!r}, cv=3).fit(X_tr, y_tr).predict_proba(X_te)\n"
             "result = (log_loss(y_te, raw, labels=[0, 1, 2]), log_loss(y_te, cal, labels=[0, 1, 2]))",
             require=["CalibratedClassifierCV", "log_loss"], tol=1e-9)


@template(M, "medium")
def f1_by_threshold(rng):
    setup, desc = _three_way(rng)
    ts = sorted(rng.sample([0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], 5))
    setup += f"\n{LRI.strip()}\nmodel = {LR}.fit(X_fit, y_fit)"
    task = rng.choice(["list", "best", "test"])
    loop = (f"{FIXED}{FROZEN}from sklearn.metrics import f1_score\n"
            f"f1 = [f1_score(y_cal, FixedThresholdClassifier(FrozenEstimator(model), threshold=t).fit(X_cal, y_cal).predict(X_cal)) for t in {ts}]\n")
    head = (f"{desc}. `model = {LR}` was trained on the fitting set. For each cut-off in `{ts}`, wrap it as "
            f"`FixedThresholdClassifier(FrozenEstimator(model), threshold=t)` and score F1 on the calibration set.")
    if task == "list":
        return P(f"{head} Return the list of F1 scores.", setup, loop + "result = f1",
                 require=["FixedThresholdClassifier", "FrozenEstimator"], tol=1e-9)
    if task == "best":
        return P(f"{head} Return the cut-off with the best F1 (the first on ties).", setup,
                 loop + f"result = {ts}[int(np.argmax(f1))]", require=["FixedThresholdClassifier", "FrozenEstimator"])
    return P(f"{head} Pick the best cut-off (first on ties) and return the test F1 `(at 0.5, at the chosen cut-off)`.",
             setup, loop + f"best = {ts}[int(np.argmax(f1))]\n"
                           "chosen = FixedThresholdClassifier(FrozenEstimator(model), threshold=best).fit(X_cal, y_cal)\n"
                           "result = (f1_score(y_te, model.predict(X_te)), f1_score(y_te, chosen.predict(X_te)))",
             require=["FixedThresholdClassifier", "FrozenEstimator"], tol=1e-9)


@template(M, "medium")
def calibration_data_hygiene(rng):
    setup, desc = _three_way(rng)
    m = rng.choice(["sigmoid", "isotonic"])
    if rng.random() < 0.5:
        return P(f"{desc}. Calibrating on the rows the model was trained on teaches the calibrator the model's "
                 f"training-set confidence. `model = GaussianNB().fit(X_fit, y_fit)`. Return the test Brier scores of "
                 f"`CalibratedClassifierCV(FrozenEstimator(model), method={m!r})` calibrated on `X_fit, y_fit` and on "
                 f"`X_cal, y_cal`, as `(on_fit, on_cal)`.", setup,
                 f"{CAL}{FROZEN}{NB}{BRIER}model = GaussianNB().fit(X_fit, y_fit)\n"
                 f"a = CalibratedClassifierCV(FrozenEstimator(model), method={m!r}).fit(X_fit, y_fit)\n"
                 f"b = CalibratedClassifierCV(FrozenEstimator(model), method={m!r}).fit(X_cal, y_cal)\n"
                 "result = (brier_score_loss(y_te, a.predict_proba(X_te)[:, 1]), brier_score_loss(y_te, b.predict_proba(X_te)[:, 1]))",
                 require=["FrozenEstimator", "CalibratedClassifierCV"], tol=1e-9)
    return P(f"{desc}. Compare two ways to use the non-test data: (a) `CalibratedClassifierCV(GaussianNB(), "
             f"method={m!r}, cv=5)` fitted on `X_tr, y_tr` (cross-fitting uses every row for both jobs); (b) "
             f"`GaussianNB()` fitted on `X_fit` and calibrated on `X_cal` with `FrozenEstimator`. Return the test Brier "
             f"scores `(a, b)`.", setup,
             f"{CAL}{FROZEN}{NB}{BRIER}a = CalibratedClassifierCV(GaussianNB(), method={m!r}, cv=5).fit(X_tr, y_tr)\n"
             f"b = CalibratedClassifierCV(FrozenEstimator(GaussianNB().fit(X_fit, y_fit)), method={m!r}).fit(X_cal, y_cal)\n"
             "result = (brier_score_loss(y_te, a.predict_proba(X_te)[:, 1]), brier_score_loss(y_te, b.predict_proba(X_te)[:, 1]))",
             require=["FrozenEstimator", "CalibratedClassifierCV"], tol=1e-9)


# ================================================================= hard

@template(M, "hard")
def expected_calibration_error(rng):
    setup, desc = _imb(rng, n=(400, 650))
    nb = rng.choice([5, 10])
    task = rng.choice(["ece", "mce", "compare"])
    ece = (f"def ece(y, p, n_bins={nb}):\n    b = np.searchsorted(np.linspace(0, 1, n_bins + 1)[1:-1], p)\n"
           "    cnt = np.bincount(b, minlength=n_bins)\n    acc = np.bincount(b, weights=y, minlength=n_bins)\n"
           "    conf = np.bincount(b, weights=p, minlength=n_bins)\n    nz = cnt > 0\n"
           "    gaps = np.abs(acc[nz] - conf[nz]) / cnt[nz]\n")
    rule = (f"Bin the probabilities like `calibration_curve(..., n_bins={nb})` (edges `np.linspace(0, 1, {nb + 1})`, bin "
            f"`np.searchsorted(edges[1:-1], p)`); per non-empty bin, gap = |rate of class 1 − mean probability|.")
    sol = f"{NB}p = GaussianNB().fit(X_tr, y_tr).predict_proba(X_te)[:, 1]\n"
    if task == "ece":
        return P(f"{desc}. Fit `GaussianNB()` on the training part. {rule} Return the expected calibration error "
                 f"Σ (bin count / n) · gap on the test set.", setup,
                 sol + ece + "    return float((cnt[nz] / len(p) * gaps).sum())\nresult = ece(y_te, p)",
                 require=["searchsorted"], tol=1e-9)
    if task == "mce":
        return P(f"{desc}. Fit `GaussianNB()` on the training part. {rule} Return the maximum calibration error (the "
                 f"largest gap) on the test set.", setup,
                 sol + ece + "    return float(gaps.max())\nresult = ece(y_te, p)", require=["searchsorted"], tol=1e-9)
    return P(f"{desc}. {rule} The expected calibration error is Σ (bin count / n) · gap. Return the test ECE of "
             f"`GaussianNB()` and of `CalibratedClassifierCV(GaussianNB(), method='isotonic', cv=5)` (both fitted on the "
             f"training part), as `(raw, calibrated)`.", setup,
             f"{CAL}" + sol + ece + "    return float((cnt[nz] / len(p) * gaps).sum())\n"
             "q = CalibratedClassifierCV(GaussianNB(), method='isotonic', cv=5).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]\n"
             "result = (ece(y_te, p), ece(y_te, q))", require=["CalibratedClassifierCV"], tol=1e-9)


@template(M, "hard")
def platt_scaling_by_hand(rng):
    setup, desc = _three_way(rng)
    if rng.random() < 0.5:
        return P(f"{desc}. `LinearSVC` has only a `decision_function`. Fit `svm = make_pipeline(StandardScaler(), "
                 f"LinearSVC())` on the fitting set. Platt scaling by hand: fit `LogisticRegression(C=1e6)` on the "
                 f"calibration set's margins (`svm.decision_function(X_cal)` as a single column) and `y_cal`, then "
                 f"return P(class 1) for the first 6 test rows.", setup,
                 "from sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\n"
                 f"from sklearn.svm import LinearSVC\n{LRI}svm = make_pipeline(StandardScaler(), LinearSVC()).fit(X_fit, y_fit)\n"
                 "platt = LogisticRegression(C=1e6).fit(svm.decision_function(X_cal).reshape(-1, 1), y_cal)\n"
                 "result = platt.predict_proba(svm.decision_function(X_te[:6]).reshape(-1, 1))[:, 1]",
                 require=["decision_function", "LogisticRegression"], tol=1e-5)
    return P(f"{desc}. Fit `svm = make_pipeline(StandardScaler(), LinearSVC())` on the fitting set and calibrate it on "
             f"the calibration set with `CalibratedClassifierCV(FrozenEstimator(svm), method='sigmoid')`. The calibrator "
             f"(`calibrated_classifiers_[0].calibrators[0]`, parameters `a_`, `b_`) maps a margin d to 1 / (1 + exp(a·d "
             f"+ b)). Return `(a_, b_, P(class 1) of the first test row computed by hand from its margin)`.", setup,
             f"{CAL}{FROZEN}from sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\n"
             "from sklearn.svm import LinearSVC\nsvm = make_pipeline(StandardScaler(), LinearSVC()).fit(X_fit, y_fit)\n"
             "cal = CalibratedClassifierCV(FrozenEstimator(svm), method='sigmoid').fit(X_cal, y_cal)\n"
             "c = cal.calibrated_classifiers_[0].calibrators[0]\nd = svm.decision_function(X_te[:1])[0]\n"
             "result = (c.a_, c.b_, 1 / (1 + np.exp(c.a_ * d + c.b_)))", require=["FrozenEstimator", "a_"], tol=1e-6)


@template(M, "hard")
def best_threshold_by_hand(rng):
    setup, desc = _three_way(rng)
    setup += f"\n{LRI.strip()}\nmodel = {LR}.fit(X_fit, y_fit)\np_cal = model.predict_proba(X_cal)[:, 1]"
    task = rng.choice(["f1", "youden", "apply"])
    scan = ("ts = np.unique(p_cal)\npos = y_cal == 1\n"
            "tp = np.array([(p_cal[pos] >= t).sum() for t in ts])\nfp = np.array([(p_cal[~pos] >= t).sum() for t in ts])\n"
            "fn = pos.sum() - tp\n")
    head = (f"{desc}. `model = {LR}` was trained on the fitting set and `p_cal` holds its P(class 1) on the calibration "
            f"set. Scan every distinct value `t` of `p_cal` as a cut-off (predict 1 when P ≥ t)")
    if task == "f1":
        return P(f"{head} and return `(t, F1)` for the cut-off with the highest F1 = 2·TP / (2·TP + FP + FN) on the "
                 f"calibration set (the smallest `t` on ties).", setup,
                 scan + "f1 = 2 * tp / (2 * tp + fp + fn)\ni = int(np.argmax(f1))\nresult = (float(ts[i]), float(f1[i]))",
                 require=["unique"], tol=1e-9)
    if task == "youden":
        return P(f"{head} and return the cut-off that maximizes Youden's J = TPR − FPR on the calibration set (the "
                 f"smallest on ties), with its J, as a tuple.", setup,
                 scan + "j = tp / pos.sum() - fp / (~pos).sum()\ni = int(np.argmax(j))\nresult = (float(ts[i]), float(j[i]))",
                 require=["unique"], tol=1e-9)
    return P(f"{head}, pick the one with the highest F1 = 2·TP / (2·TP + FP + FN) (smallest on ties), and apply it to the "
             f"test set with `FixedThresholdClassifier(FrozenEstimator(model), threshold=t)`. Return the test F1 `(at "
             f"0.5, at t)`.", setup,
             scan + f"{FIXED}{FROZEN}from sklearn.metrics import f1_score\n"
                    "t = ts[int(np.argmax(2 * tp / (2 * tp + fp + fn)))]\n"
                    "clf = FixedThresholdClassifier(FrozenEstimator(model), threshold=t).fit(X_cal, y_cal)\n"
                    "result = (f1_score(y_te, model.predict(X_te)), f1_score(y_te, clf.predict(X_te)))",
             require=["FixedThresholdClassifier", "FrozenEstimator"], tol=1e-9)


@template(M, "hard")
def cost_sensitive_decisions(rng):
    setup, desc = _imb(rng, n=(400, 650))
    c_fn, c_fp = rng.choice([(10, 1), (5, 1), (20, 2)])
    cost = (f"def cost(y_true, y_pred):\n    return {c_fn} * int(((y_true == 1) & (y_pred == 0)).sum()) + "
            f"{c_fp} * int(((y_true == 0) & (y_pred == 1)).sum())\n")
    task = rng.choice(["bayes", "tuned", "compare"])
    head = f"{desc}. A missed positive costs {c_fn} and a false alarm {c_fp}."
    if task == "bayes":
        return P(f"{head} With calibrated probabilities, predicting 1 when P ≥ c_fp / (c_fp + c_fn) minimizes the "
                 f"expected cost. Return the total test cost of `GaussianNB()` with that rule applied to its **raw** "
                 f"probabilities and to those of `CalibratedClassifierCV(GaussianNB(), method='isotonic', cv=5)` (both "
                 f"fitted on the training part), as `(raw, calibrated)`.", setup,
                 f"{CAL}{NB}{cost}t = {c_fp} / ({c_fp} + {c_fn})\nraw = GaussianNB().fit(X_tr, y_tr).predict_proba(X_te)[:, 1]\n"
                 "cal = CalibratedClassifierCV(GaussianNB(), method='isotonic', cv=5).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]\n"
                 "result = (cost(y_te, (raw >= t).astype(int)), cost(y_te, (cal >= t).astype(int)))",
                 require=["CalibratedClassifierCV"])
    if task == "tuned":
        return P(f"{head} Write `cost(y_true, y_pred)` (total cost) and let `TunedThresholdClassifierCV({LR}, "
                 f"scoring=make_scorer(cost, greater_is_better=False), cv=5)` pick the cut-off on the training part. "
                 f"Return `(best_threshold_, test cost of the tuned model, test cost at 0.5)`.", setup,
                 f"{TUNED}{LRI}from sklearn.metrics import make_scorer\n{cost}"
                 f"tuned = TunedThresholdClassifierCV({LR}, scoring=make_scorer(cost, greater_is_better=False), cv=5).fit(X_tr, y_tr)\n"
                 f"result = (tuned.best_threshold_, cost(y_te, tuned.predict(X_te)), cost(y_te, {LR}.fit(X_tr, y_tr).predict(X_te)))",
                 require=["TunedThresholdClassifierCV", "make_scorer"], tol=1e-9)
    return P(f"{head} Fit `{LR}` on the training part. Return the total test cost at the cut-offs `[0.05, 0.1, 0.2, 0.3, "
             f"0.5]` (predict 1 when P ≥ t) as a list, and the Bayes cut-off c_fp / (c_fp + c_fn), as a tuple `(costs, "
             f"bayes_t)`.", setup,
             f"{LRI}{cost}p = {LR}.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]\n"
             f"result = ([cost(y_te, (p >= t).astype(int)) for t in [0.05, 0.1, 0.2, 0.3, 0.5]], {c_fp} / ({c_fp} + {c_fn}))",
             require=["predict_proba"], tol=1e-9)


@template(M, "hard")
def calibration_pitfalls(rng):
    setup, desc = _three_way(rng)
    t = rng.choice([0.2, 0.3])
    items = [
        ("FrozenEstimator", "This raises in scikit-learn 1.8: `CalibratedClassifierCV(model, cv='prefit').fit(X_cal, "
                            "y_cal)` (the `'prefit'` option is gone). `model = GaussianNB().fit(X_fit, y_fit)`. Do the "
                            "same with `FrozenEstimator` and `method='sigmoid'`, and return P(class 1) for the first 5 "
                            "test rows.",
         f"{CAL}{FROZEN}{NB}model = GaussianNB().fit(X_fit, y_fit)\n"
         "cal = CalibratedClassifierCV(FrozenEstimator(model), method='sigmoid').fit(X_cal, y_cal)\nresult = cal.predict_proba(X_te[:5])[:, 1]",
         {"require": ["FrozenEstimator", "CalibratedClassifierCV"]}),
        ("calibration_curve", "A colleague passed **labels** to the reliability curve: `calibration_curve(y_te, "
                              "model.predict(X_te), n_bins=5)` (only bins 0 and 1 get anything). `model = GaussianNB()"
                              ".fit(X_fit, y_fit)`. Use P(class 1) instead and return `(prob_true, prob_pred)`.",
         f"{CURVE}{NB}model = GaussianNB().fit(X_fit, y_fit)\n"
         "result = calibration_curve(y_te, model.predict_proba(X_te)[:, 1], n_bins=5)"),
        ("calibration_curve", "`calibration_curve(p, y_te, n_bins=5)` raises: the arguments are swapped (true labels "
                              "come first). With `p` = the test P(class 1) of `GaussianNB().fit(X_fit, y_fit)`, fix the "
                              "call and return `prob_true`.",
         f"{CURVE}{NB}p = GaussianNB().fit(X_fit, y_fit).predict_proba(X_te)[:, 1]\nresult = calibration_curve(y_te, p, n_bins=5)[0]"),
        ("TunedThresholdClassifierCV", f"A colleague tuned the cut-off of `{LR}` for F1 on the **test** set, so the "
                                       f"reported F1 is optimistic. Tune it on the non-test rows instead with "
                                       f"`TunedThresholdClassifierCV({LR}, scoring='f1', cv=5)` fitted on `X_tr, y_tr`, "
                                       f"and return `(best_threshold_, test F1)`.",
         f"{TUNED}{LRI}from sklearn.metrics import f1_score\ntuned = TunedThresholdClassifierCV({LR}, scoring='f1', cv=5).fit(X_tr, y_tr)\n"
         "result = (tuned.best_threshold_, f1_score(y_te, tuned.predict(X_te)))"),
        ("response_method", f"`FixedThresholdClassifier({LR}, threshold={t}, response_method='decision_function')` "
                            f"was meant to flag rows with P(class 1) ≥ {t}, but it thresholds the **margin** instead. "
                            f"Fix it (`response_method='predict_proba'`), fit on `X_tr, y_tr`, and return `(flagged "
                            f"with the bug, flagged after the fix)`: the counts of test rows predicted 1.",
         f"{FIXED}{LRI}bad = FixedThresholdClassifier({LR}, threshold={t}, response_method='decision_function').fit(X_tr, y_tr)\n"
         f"good = FixedThresholdClassifier({LR}, threshold={t}, response_method='predict_proba').fit(X_tr, y_tr)\n"
         "result = (int(bad.predict(X_te).sum()), int(good.predict(X_te).sum()))"),
        ("method", "With only 40 calibration rows, isotonic calibration overfits. `model = GaussianNB().fit(X_fit, "
                   "y_fit)`. Calibrate it on `X_cal[:40], y_cal[:40]` with `FrozenEstimator` and each method; return the "
                   "test Brier scores `(sigmoid, isotonic)`.",
         f"{CAL}{FROZEN}{NB}{BRIER}model = GaussianNB().fit(X_fit, y_fit)\n"
         "b = {m: brier_score_loss(y_te, CalibratedClassifierCV(FrozenEstimator(model), method=m).fit(X_cal[:40], y_cal[:40]).predict_proba(X_te)[:, 1]) for m in ('sigmoid', 'isotonic')}\n"
         "result = (b['sigmoid'], b['isotonic'])", {"require": ["FrozenEstimator", "method"]}),
        ("FixedThresholdClassifier", f"`FixedThresholdClassifier(model, threshold={t})` was fitted on `X_cal, y_cal`, "
                                     f"which **retrained** the already-trained `model = {LR}.fit(X_fit, y_fit)` on the "
                                     f"calibration rows. Wrap it in `FrozenEstimator` so it keeps its weights and return "
                                     f"`(coef_ of the wrapped estimator, test predictions of the first 8 rows)`.",
         f"{FIXED}{FROZEN}{LRI}model = {LR}.fit(X_fit, y_fit)\n"
         f"clf = FixedThresholdClassifier(FrozenEstimator(model), threshold={t}).fit(X_cal, y_cal)\n"
         "result = (clf.estimator_.estimator.coef_, clf.predict(X_te[:8]))", {"require": ["FrozenEstimator"]}),
    ]
    return catalog(rng, items, setup, f"{desc}; `X_tr, y_tr` (all non-test rows) are kept too.", tol=1e-9)


@template(M, "hard")
def recall_constrained_tuning(rng):
    setup, desc = _imb(rng, n=(400, 650))
    r = rng.choice([0.8, 0.85, 0.9])
    score = (f"def precision_at_recall(y_true, y_pred):\n    rec = ((y_true == 1) & (y_pred == 1)).sum() / (y_true == 1).sum()\n"
             f"    if rec < {r}:\n        return 0.0\n    return ((y_true == 1) & (y_pred == 1)).sum() / max((y_pred == 1).sum(), 1)\n")
    head = (f"{desc}. Tune a cut-off for recall ≥ {r} and the best precision you can get: write "
            f"`precision_at_recall(y_true, y_pred)` that returns the precision when recall ≥ {r} and 0 otherwise, and "
            f"pass `make_scorer(precision_at_recall)` as the `scoring` of `TunedThresholdClassifierCV({LR}, cv=5)`, "
            f"fitted on the training part.")
    sol = (f"{TUNED}{LRI}from sklearn.metrics import make_scorer, precision_score, recall_score\n{score}"
           f"tuned = TunedThresholdClassifierCV({LR}, scoring=make_scorer(precision_at_recall), cv=5).fit(X_tr, y_tr)\n")
    if rng.random() < 0.5:
        return P(f"{head} Return `(best_threshold_, best_score_)`.", setup,
                 sol + "result = (tuned.best_threshold_, tuned.best_score_)", require=["TunedThresholdClassifierCV", "make_scorer"],
                 tol=1e-9)
    return P(f"{head} Return the test `(precision, recall)` of the tuned model.", setup,
             sol + "pred = tuned.predict(X_te)\nresult = (precision_score(y_te, pred, zero_division=0), recall_score(y_te, pred))",
             require=["TunedThresholdClassifierCV", "make_scorer"], tol=1e-9)


@template(M, "hard")
def quantile_reliability_by_hand(rng):
    setup, desc = _imb(rng, n=(400, 650))
    nb = rng.choice([4, 5, 8])
    if rng.random() < 0.4:
        return P(f"{desc}. Fit `GaussianNB()` on the training part; `p` = its test P(class 1). Compare the two binning "
                 f"strategies of `calibration_curve` with {nb} bins: count the test rows per bin for uniform edges "
                 f"(`np.linspace(0, 1, {nb + 1})`) and for quantile edges (`np.percentile(p, np.linspace(0, 100, "
                 f"{nb + 1}))`), assigning bins with `np.searchsorted(edges[1:-1], p)`. Return the two count arrays as "
                 f"`(uniform, quantile)` (length {nb} each, empty bins included).", setup,
                 f"{NB}p = GaussianNB().fit(X_tr, y_tr).predict_proba(X_te)[:, 1]\n"
                 f"count = lambda e: np.bincount(np.searchsorted(e[1:-1], p), minlength={nb})\n"
                 f"result = (count(np.linspace(0, 1, {nb + 1})), count(np.percentile(p, np.linspace(0, 100, {nb + 1}))))",
                 require=["percentile", "searchsorted"])
    return P(f"{desc}. Fit `GaussianNB()` on the training part; `p` = its test P(class 1). Rebuild "
             f"`calibration_curve(y_te, p, n_bins={nb}, strategy='quantile')` **by hand**: the edges are "
             f"`np.percentile(p, np.linspace(0, 100, {nb + 1}))`, a probability goes to bin `np.searchsorted(edges[1:-1], "
             f"p)`, and non-empty bins report (rate of class 1, mean probability). Return the tuple of the two arrays.",
             setup, f"{NB}p = GaussianNB().fit(X_tr, y_tr).predict_proba(X_te)[:, 1]\n"
                    f"edges = np.percentile(p, np.linspace(0, 100, {nb + 1}))\nb = np.searchsorted(edges[1:-1], p)\n"
                    f"cnt = np.bincount(b, minlength={nb})\nnz = cnt > 0\n"
                    f"result = (np.bincount(b, weights=y_te, minlength={nb})[nz] / cnt[nz], np.bincount(b, weights=p, minlength={nb})[nz] / cnt[nz])",
             require=["percentile", "searchsorted"], tol=1e-9)


@template(M, "hard")
def deployment_workflow(rng):
    t = table(rng, kind="clf", n=(400, 600), only=["churn", "loans", "patients", "flights", "students"])
    s = seeded(rng)
    num, cat = t.num, t.cat
    setup = (t.setup + f"\nnum, cat = {lit(num)}, {lit(cat)}\nfrom sklearn.model_selection import train_test_split\n"
             f"X_tr, X_te, y_tr, y_te = train_test_split(df, y, stratify=y, random_state={s})\n"
             "X_fit, X_cal, y_fit, y_cal = train_test_split(X_tr, y_tr, stratify=y_tr, test_size=0.4, random_state=1)\n"
             "from sklearn.compose import ColumnTransformer\nfrom sklearn.pipeline import Pipeline\n"
             "from sklearn.preprocessing import OneHotEncoder, StandardScaler\nfrom sklearn.ensemble import RandomForestClassifier\n"
             "prep = ColumnTransformer([('num', StandardScaler(), num), ('cat', OneHotEncoder(handle_unknown='ignore'), cat)])\n"
             "model = Pipeline([('prep', prep), ('rf', RandomForestClassifier(n_estimators=40, min_samples_leaf=3, random_state=0))]).fit(X_fit, y_fit)")
    c_fn, c_fp = rng.choice([(5, 1), (8, 1), (3, 1)])
    items = [
        ("CalibratedClassifierCV", "Calibrate the trained `model` on the calibration set (`CalibratedClassifierCV("
                                   "FrozenEstimator(model), method='isotonic')`) and return the test Brier scores "
                                   "`(before, after)`.",
         f"{CAL}{FROZEN}{BRIER}cal = CalibratedClassifierCV(FrozenEstimator(model), method='isotonic').fit(X_cal, y_cal)\n"
         "result = (brier_score_loss(y_te, model.predict_proba(X_te)[:, 1]), brier_score_loss(y_te, cal.predict_proba(X_te)[:, 1]))",
         {"require": ["FrozenEstimator", "CalibratedClassifierCV"]}),
        ("TunedThresholdClassifierCV", "Tune the cut-off of the trained `model` for F1 on the calibration set without "
                                       "retraining it: `TunedThresholdClassifierCV(FrozenEstimator(model), scoring='f1', "
                                       "cv='prefit', refit=False)` fitted on `X_cal, y_cal`. Return `(best_threshold_, "
                                       "test F1)`.",
         f"{TUNED}{FROZEN}from sklearn.metrics import f1_score\n"
         "tuned = TunedThresholdClassifierCV(FrozenEstimator(model), scoring='f1', cv='prefit', refit=False).fit(X_cal, y_cal)\n"
         "result = (tuned.best_threshold_, f1_score(y_te, tuned.predict(X_te)))",
         {"require": ["TunedThresholdClassifierCV", "FrozenEstimator"]}),
        ("FixedThresholdClassifier", f"Costs: a miss {c_fn}, a false alarm {c_fp}. Calibrate `model` on the "
                                     f"calibration set (`CalibratedClassifierCV(FrozenEstimator(model), "
                                     f"method='sigmoid')`), wrap the calibrated model in `FixedThresholdClassifier(..., "
                                     f"threshold={c_fp} / ({c_fp} + {c_fn}))` (frozen too) and return the total test "
                                     f"cost `(at 0.5 with the raw model, with the wrapper)`.",
         f"{CAL}{FROZEN}{FIXED}cal = CalibratedClassifierCV(FrozenEstimator(model), method='sigmoid').fit(X_cal, y_cal)\n"
         f"clf = FixedThresholdClassifier(FrozenEstimator(cal), threshold={c_fp} / ({c_fp} + {c_fn})).fit(X_cal, y_cal)\n"
         f"cost = lambda yp: {c_fn} * int(((y_te == 1) & (yp == 0)).sum()) + {c_fp} * int(((y_te == 0) & (yp == 1)).sum())\n"
         "result = (cost(model.predict(X_te)), cost(clf.predict(X_te)))", {"require": ["FixedThresholdClassifier", "FrozenEstimator"]}),
        ("calibration_curve", "Return the test reliability curves `calibration_curve(y_te, q, n_bins=5)[0]` (observed "
                              "rates) of the raw `model` and of its sigmoid calibration on the calibration set "
                              "(`FrozenEstimator`), as `(raw, calibrated)`.",
         f"{CAL}{FROZEN}{CURVE}cal = CalibratedClassifierCV(FrozenEstimator(model), method='sigmoid').fit(X_cal, y_cal)\n"
         "result = (calibration_curve(y_te, model.predict_proba(X_te)[:, 1], n_bins=5)[0], "
         "calibration_curve(y_te, cal.predict_proba(X_te)[:, 1], n_bins=5)[0])"),
        ("FrozenEstimator", "Check that freezing really protects the model: calibrate `FrozenEstimator(model)` on the "
                            "calibration set (isotonic), then return `(model.predict_proba(X_te[:3])[:, 1], "
                            "cal.predict_proba(X_te[:3])[:, 1])`: the original model's probabilities are unchanged.",
         f"{CAL}{FROZEN}before = model.predict_proba(X_te[:3])[:, 1]\n"
         "cal = CalibratedClassifierCV(FrozenEstimator(model), method='isotonic').fit(X_cal, y_cal)\n"
         "result = (model.predict_proba(X_te[:3])[:, 1], cal.predict_proba(X_te[:3])[:, 1])"),
        ("TunedThresholdClassifierCV", "Tune a balanced-accuracy cut-off for the trained `model` on the calibration set "
                                       "(`TunedThresholdClassifierCV(FrozenEstimator(model), cv='prefit', refit=False)`, "
                                       "default scoring) and return the test balanced accuracy `(at 0.5, tuned)`.",
         f"{TUNED}{FROZEN}from sklearn.metrics import balanced_accuracy_score\n"
         "tuned = TunedThresholdClassifierCV(FrozenEstimator(model), cv='prefit', refit=False).fit(X_cal, y_cal)\n"
         "result = (balanced_accuracy_score(y_te, model.predict(X_te)), balanced_accuracy_score(y_te, tuned.predict(X_te)))"),
    ]
    return catalog(rng, items, setup, f"{t.desc}. `num`, `cat` list the numeric and categorical columns. It was split "
                                      f"into a test set (`X_te, y_te`) and, from the rest, a fitting set (`X_fit, y_fit`) "
                                      f"and a calibration set (`X_cal, y_cal`). `model` (a `ColumnTransformer` + "
                                      f"`RandomForestClassifier(n_estimators=40, min_samples_leaf=3, random_state=0)` "
                                      f"pipeline) is already trained on the fitting set.", tol=1e-9)


@template(M, "hard")
def prior_shift_adjustment(rng):
    setup, desc = _imb(rng, n=(400, 650))
    new = rng.choice([0.05, 0.4, 0.5])
    task = rng.choice(["adjust", "brier", "cutoff"])
    rule = (f"The model was trained where class 1 has rate π = `y_tr.mean()`; in deployment the rate is {new}. Bayes' "
            f"rule rescales the odds: odds' = odds × ({new} / (1 − {new})) / (π / (1 − π)), and P' = odds' / (1 + odds').")
    sol = (f"{LRI}p = {LR}.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]\npi = y_tr.mean()\n"
           f"odds = p / (1 - p) * ({new} / (1 - {new})) / (pi / (1 - pi))\nq = odds / (1 + odds)\n")
    if task == "adjust":
        return P(f"{desc}. Fit `{LR}` on the training part. {rule} Return the adjusted P(class 1) of the first 6 test "
                 f"rows.", setup, sol + "result = q[:6]", require=["predict_proba"], tol=1e-9)
    if task == "cutoff":
        return P(f"{desc}. Fit `{LR}` on the training part. {rule} Instead of adjusting every probability you can move "
                 f"the cut-off: P' ≥ 0.5 exactly when the original P ≥ k / (1 + k) with k = (π / (1 − π)) / ({new} / "
                 f"(1 − {new})). Return `(that cut-off, number of test rows with P' ≥ 0.5)`.", setup,
                 sol + f"k = (pi / (1 - pi)) / ({new} / (1 - {new}))\nresult = (float(k / (1 + k)), int((q >= 0.5).sum()))",
                 require=["predict_proba"], tol=1e-9)
    return P(f"{desc}. Fit `{LR}` on the training part. {rule} Return `(mean adjusted probability on the test set, "
             f"number of test rows whose label at 0.5 flips after the adjustment)`.", setup,
             sol + "result = (float(q.mean()), int(((p >= 0.5) != (q >= 0.5)).sum()))", require=["predict_proba"], tol=1e-9)


# ================================================================= challenge

@template(M, "challenge")
def isotonic_calibration_pava(rng):
    n, s = rng.randint(10, 15), seeded(rng)
    report = rng.choice(["fitted", "predict", "brier"])
    setup = (f"g = np.random.default_rng({s})\nscore = g.uniform(0, 1, {n}).round(1)   # coarse: repeated scores\n"
             f"y = (g.uniform(0, 1, {n}) < score).astype(int)\nq = g.uniform(-0.1, 1.1, 5).round(2)\nreport = {report!r}")
    tests = [f"g = np.random.default_rng({s + i})\nscore = g.uniform(0, 1, 200000).round(5)\n"
             "y = (g.uniform(0, 1, 200000) < score ** 2).astype(int)\nq = g.uniform(-0.1, 1.1, 200000)\n"
             f"report = {report!r}" for i in (1, 2)]
    sol = ("xs, inv, cnt = np.unique(score, return_inverse=True, return_counts=True)\nsy = np.bincount(inv, weights=y)\n"
           "S, C, L = [], [], []   # blocks: sum of labels, count, number of distinct scores\n"
           "for a, c in zip(sy.astype(np.int64).tolist(), cnt.tolist()):\n    S.append(a); C.append(c); L.append(1)\n"
           "    while len(S) > 1 and S[-2] * C[-1] >= S[-1] * C[-2]:\n"
           "        a2, c2, l2 = S.pop(), C.pop(), L.pop()\n        S[-1] += a2; C[-1] += c2; L[-1] += l2\n"
           "level = np.repeat(np.array(S) / np.array(C), L)\n"
           + {"fitted": "result = level[inv]",
              "predict": "result = np.interp(np.clip(q, xs[0], xs[-1]), xs, level)",
              "brier": "result = float(((level[inv] - y) ** 2).mean())"}[report])
    want = {"fitted": "the calibrated value of every training score, in the original order (a float array; what "
                      "`fit_transform` returns)",
            "predict": "the calibrated value of every query score `q[j]` (a float array): between two distinct training "
                       "scores interpolate linearly, outside their range clip to the end values",
            "brier": "the Brier score of the calibrated training values, mean((fitted − y)²) (a float)"}[report]
    return challenge(
        "Isotonic calibration from scratch",
        "`CalibratedClassifierCV(method='isotonic')` fits `IsotonicRegression(out_of_bounds='clip')` to (score, label) "
        "pairs: the non-decreasing step function closest to the labels in squared error. Equal scores are first merged "
        "into one point (their mean label, weighted by their count); then the pool-adjacent-violators algorithm merges "
        "neighbouring blocks while a block's mean is not below the previous block's mean (equal means merge too). "
        f"Return {want}. (`report` says which: {report!r}.) Restarting the scan after every merge is quadratic; a "
        "stack of blocks makes it linear after sorting.", setup, sol, tests,
        "n, len(q) ≤ 2·10^5", speed="O(n log n)", tol=1e-6)


@template(M, "challenge")
def best_threshold_scan(rng):
    n, s = rng.randint(10, 16), seeded(rng)
    report = rng.choice(["f1", "cost", "recall"])
    c_fn, c_fp = rng.choice([(5, 1), (3, 2), (10, 1)])
    r = rng.choice([60, 75, 90])
    setup = (f"g = np.random.default_rng({s})\ny = (g.uniform(0, 1, {n}) < 0.35).astype(int)\ny[:2] = [0, 1]\n"
             f"p = np.clip(0.3 * y + g.uniform(0, 0.7, {n}), 0, 1).round(1)   # coarse: tied scores\n"
             f"c_fn, c_fp, R = {c_fn}, {c_fp}, {r}\nreport = {report!r}")
    tests = [f"g = np.random.default_rng({s + i})\ny = (g.uniform(0, 1, 200000) < 0.2).astype(int)\n"
             "p = np.clip(0.25 * y + g.uniform(0, 0.75, 200000), 0, 1).round(5)\n"
             f"c_fn, c_fp, R = {c_fn}, {c_fp}, {r}\nreport = {report!r}" for i in (1, 2)]
    sol = ("ts, inv = np.unique(p, return_inverse=True)\npos_at = np.bincount(inv, weights=y, minlength=len(ts))\n"
           "neg_at = np.bincount(inv, minlength=len(ts)) - pos_at\n"
           "tp = np.cumsum(pos_at[::-1])[::-1]\nfp = np.cumsum(neg_at[::-1])[::-1]\nP_ = y.sum()\nfn = P_ - tp\n")
    sol += {"f1": "f1 = 2 * tp / (2 * tp + fp + fn)\ni = int(np.argmax(f1))\nresult = (float(ts[i]), float(f1[i]))",
            "cost": "cost = c_fn * fn + c_fp * fp\ni = int(np.argmin(cost))\nresult = (float(ts[i]), int(cost[i]))",
            "recall": "ok = np.flatnonzero(100 * tp >= R * P_)\ni = int(ok[-1])\nresult = (float(ts[i]), float(tp[i] / (tp[i] + fp[i])))"}[report]
    want = {"f1": "the cut-off with the highest F1 = 2·TP / (2·TP + FP + FN), as the tuple `(t, f1)` (the smallest `t` "
                  "on ties)",
            "cost": "the cut-off with the lowest total cost `c_fn·FN + c_fp·FP`, as the tuple `(t, cost)` with an int "
                    "cost (the smallest `t` on ties)",
            "recall": "the **largest** cut-off whose recall is at least `R`% (compare exactly: 100·TP ≥ R·(number of "
                      "positives)), as the tuple `(t, precision at t)`"}[report]
    return challenge(
        "Sweeping every cut-off",
        "`TunedThresholdClassifierCV` searches for the best decision threshold. Here you search them all: `p[i]` is a "
        "model's P(class 1) for row `i` and `y[i]` its label; a cut-off `t` predicts 1 when `p >= t`, and the candidate "
        f"cut-offs are the distinct values of `p`. Return {want}. (`report` says which: {report!r}.) Counting "
        "TP and FP from scratch for every candidate is quadratic; sort once and use cumulative counts.",
        setup, sol, tests, "n ≤ 2·10^5", speed="O(n log n)", tol=1e-9)


@template(M, "challenge")
def calibration_error_bins(rng):
    n, s = rng.randint(12, 18), seeded(rng)
    nb = rng.randint(3, 5)
    report = rng.choice(["ece", "mce", "curve"])
    setup = (f"g = np.random.default_rng({s})\np = g.uniform(0, 1, {n}).round(3)\n"
             f"y = (g.uniform(0, 1, {n}) < p ** 1.5).astype(int)\nn_bins = {nb}\nreport = {report!r}")
    tests = [f"g = np.random.default_rng({s + i})\np = g.uniform(0, 1, 200000)\n"
             f"y = (g.uniform(0, 1, 200000) < p ** 1.5).astype(int)\nn_bins = {50000 * i}\nreport = {report!r}"
             for i in (1, 2)]
    sol = ("b = np.searchsorted(np.linspace(0, 1, n_bins + 1)[1:-1], p)\ncnt = np.bincount(b, minlength=n_bins)\n"
           "rate = np.bincount(b, weights=y, minlength=n_bins)\nconf = np.bincount(b, weights=p, minlength=n_bins)\n"
           "nz = cnt > 0\ngap = np.abs(rate[nz] - conf[nz]) / cnt[nz]\n"
           + {"ece": "result = float((gap * cnt[nz]).sum() / len(p))", "mce": "result = float(gap.max())",
              "curve": "result = rate[nz] / cnt[nz]"}[report])
    want = {"ece": "the **expected calibration error** Σ (bin count / n) · gap (a float)",
            "mce": "the **maximum calibration error**, the largest gap (a float)",
            "curve": "the observed rate of class 1 of every non-empty bin, in bin order (a float array: the "
                     "`prob_true` of `calibration_curve(y, p, n_bins=n_bins)`)"}[report]
    return challenge(
        "Calibration error with many bins",
        "`calibration_curve(y, p, n_bins)` cuts [0, 1] into `n_bins` equal bins with edges `np.linspace(0, 1, n_bins + "
        "1)`; probability `p[i]` goes to bin `b` when edge_b < p ≤ edge_(b+1) (0 goes to the first bin). For every "
        "non-empty bin, gap = |rate of class 1 in the bin − mean probability in the bin|. Return "
        f"{want}. (`report` says which: {report!r}.) A loop over bins that scans all points each time is too slow "
        "when there are many bins.", setup, sol, tests, "n ≤ 2·10^5 probabilities, n_bins ≤ 10^5",
        speed="O(n log n_bins)", tol=1e-6)


@template(M, "challenge")
def metrics_at_thresholds(rng):
    n, s, q = rng.randint(10, 16), seeded(rng), rng.randint(4, 6)
    report = rng.choice(["precision", "recall", "f1"])
    setup = (f"g = np.random.default_rng({s})\ny = (g.uniform(0, 1, {n}) < 0.4).astype(int)\ny[:2] = [0, 1]\n"
             f"p = np.clip(0.3 * y + g.uniform(0, 0.7, {n}), 0, 1).round(2)\n"
             f"ts = np.sort(g.uniform(0, 1, {q})).round(2)\nreport = {report!r}")
    tests = [f"g = np.random.default_rng({s + i})\ny = (g.uniform(0, 1, 200000) < 0.3).astype(int)\n"
             "p = np.clip(0.3 * y + g.uniform(0, 0.7, 200000), 0, 1)\nts = g.uniform(0, 1, 200000)\n"
             f"report = {report!r}" for i in (1, 2)]
    sol = ("pp, pn = np.sort(p[y == 1]), np.sort(p[y == 0])\ntp = len(pp) - np.searchsorted(pp, ts, side='left')\n"
           "fp = len(pn) - np.searchsorted(pn, ts, side='left')\n"
           + {"precision": "result = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)",
              "recall": "result = tp / len(pp)",
              "f1": "result = np.where(tp > 0, 2 * tp / np.maximum(2 * tp + fp + (len(pp) - tp), 1), 0.0)"}[report])
    want = {"precision": "the **precision** TP / (TP + FP), or 0 when nothing is predicted positive",
            "recall": "the **recall** TP / (number of positives)",
            "f1": "the **F1** 2·TP / (2·TP + FP + FN), or 0 when TP = 0"}[report]
    return challenge(
        "What-if thresholds",
        "A dashboard lets analysts try cut-offs for `FixedThresholdClassifier`: `p[i]` is the model's P(class 1) for row "
        "`i`, `y[i]` its label, and query `j` predicts 1 when `p >= ts[j]`. For every query return "
        f"{want}, as a float array. (`report` says which: {report!r}.) Re-scanning all rows per query is too slow.",
        setup, sol, tests, "n, number of queries ≤ 2·10^5", speed="O((n + queries) log n)", tol=1e-9)
