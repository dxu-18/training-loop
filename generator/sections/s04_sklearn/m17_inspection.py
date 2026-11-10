"""4.17 Model inspection & built-in plots: permutation and impurity importances, partial dependence / ICE, and the
Display classes (confusion matrix, ROC, PR, DET, calibration, decision boundary, learning / validation curves,
prediction error).

The page loads matplotlib and the grader forces the Agg backend. Plots are graded through the values a Display
stores (`confusion_matrix`, `roc_auc`, `pd_results`, `response`, ...) or what landed on its Axes (number of lines,
labels), never through images."""

from ._common import P, catalog, challenge, lit, seeded, synth_clf, table, template

M = "4.17"
PLT = "import matplotlib.pyplot as plt\n"
LR_PIPE = "make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))"
LR_IMP = ("from sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\n"
          "from sklearn.linear_model import LogisticRegression\n")
RF = "RandomForestClassifier(n_estimators=30, random_state=0)"
RF_IMP = "from sklearn.ensemble import RandomForestClassifier\n"
MET = "from sklearn.metrics import "
INS = "from sklearn.inspection import "
MS = "from sklearn.model_selection import "


def _binary(rng, n=(250, 420)):
    """A binary problem as DataFrames: breast cancer ('mean ...' columns), a themed table, or make_classification."""
    s = seeded(rng)
    kind = rng.choice(["cancer", "table", "synth"])
    split = f"\nfrom sklearn.model_selection import train_test_split\nX_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, random_state={s})"
    if kind == "cancer":
        return (PLT + "from sklearn.datasets import load_breast_cancer\ndata = load_breast_cancer(as_frame=True)\n"
                f"order = np.random.default_rng({s}).permutation(569)\n"
                "X, y = data.data.iloc[order, :10].reset_index(drop=True), data.target.iloc[order].reset_index(drop=True)" + split,
                "`X` holds the 10 'mean ...' measurements of 569 breast tumours (a DataFrame, rows shuffled) and `y` the "
                "diagnosis (0 malignant, 1 benign)", ["malignant", "benign"])
    if kind == "table":
        t = table(rng, kind="clf", numeric_only=True, n=n)
        return (PLT + t.setup + "\nX = df.astype(float)" + split, f"{t.desc}; `X = df.astype(float)`", ["no", "yes"])
    d = synth_clf(rng, n=n, d=(4, 7), classes=2)
    return (PLT + d.setup + "\nX = pd.DataFrame(X, columns=[f'x{i}' for i in range(X.shape[1])])" + split,
            d.desc + " (as a DataFrame with columns `x0`, `x1`, ...)", ["negative", "positive"])


def _split_desc(desc):
    return f"{desc}, split with `train_test_split(X, y, stratify=y, random_state=...)` into `X_tr, X_te, y_tr, y_te`"


def _moons(rng):
    n, s = rng.randint(120, 220), seeded(rng)
    noise = rng.choice([0.15, 0.2, 0.3])
    if rng.random() < 0.6:
        return (PLT + f"from sklearn.datasets import make_moons\nX2, y2 = make_moons(n_samples={n}, noise={noise}, random_state={s})",
                f"`X2` holds {n} 2-D points in two interleaving half-moons (`make_moons(noise={noise})`) and `y2` their class")
    return (PLT + f"from sklearn.datasets import make_blobs\nX2, y2 = make_blobs(n_samples={n}, centers=3, cluster_std=2.0, random_state={s})",
            f"`X2` holds {n} 2-D points around 3 centres (`make_blobs`) and `y2` their cluster (0, 1, 2)")


# ================================================================= lite

@template(M, "lite")
def classifier_displays(rng):
    setup, desc, names = _binary(rng)
    setup += f"\n{LR_IMP}clf = {LR_PIPE}.fit(X_tr, y_tr)"
    nb = rng.choice([4, 5, 8])
    name = rng.choice(["LR", "logreg", "baseline"])
    items = [
        ("ConfusionMatrixDisplay", "Draw `ConfusionMatrixDisplay.from_estimator(clf, X_te, y_te)` (`sklearn.metrics`) and "
                                   "return the matrix it shows (`disp.confusion_matrix`).",
         f"{MET}ConfusionMatrixDisplay\ndisp = ConfusionMatrixDisplay.from_estimator(clf, X_te, y_te)\nresult = disp.confusion_matrix",
         {"require": ["ConfusionMatrixDisplay", "from_estimator"]}),
        ("from_predictions", "Build the same plot from labels you already have: `ConfusionMatrixDisplay.from_predictions("
                             "y_te, clf.predict(X_te))`. Return its `confusion_matrix`.",
         f"{MET}ConfusionMatrixDisplay\ndisp = ConfusionMatrixDisplay.from_predictions(y_te, clf.predict(X_te))\n"
         "result = disp.confusion_matrix", {"require": ["ConfusionMatrixDisplay", "from_predictions"]}),
        ("normalize", "Show rates instead of counts: `ConfusionMatrixDisplay.from_estimator(clf, X_te, y_te, "
                      "normalize='true')` (each row sums to 1). Return the displayed matrix.",
         f"{MET}ConfusionMatrixDisplay\nresult = ConfusionMatrixDisplay.from_estimator(clf, X_te, y_te, normalize='true').confusion_matrix",
         {"require": ["ConfusionMatrixDisplay", "normalize"]}),
        ("display_labels", f"Name the classes: draw the confusion matrix with `display_labels={names}` and return the "
                           f"x tick labels of its Axes (`[t.get_text() for t in disp.ax_.get_xticklabels()]`).",
         f"{MET}ConfusionMatrixDisplay\ndisp = ConfusionMatrixDisplay.from_estimator(clf, X_te, y_te, display_labels={names})\n"
         "result = [t.get_text() for t in disp.ax_.get_xticklabels()]", {"require": ["display_labels", "ax_"]}),
        ("cmap", "Draw the confusion matrix with `cmap='Blues'` and return the colormap's name (`disp.im_.get_cmap()"
                 ".name`) and the matrix, as a tuple.",
         f"{MET}ConfusionMatrixDisplay\ndisp = ConfusionMatrixDisplay.from_estimator(clf, X_te, y_te, cmap='Blues')\n"
         "result = (disp.im_.get_cmap().name, disp.confusion_matrix)", {"require": ["cmap"]}),
        ("RocCurveDisplay", "Draw the ROC curve with `RocCurveDisplay.from_estimator(clf, X_te, y_te)` and return the "
                            "area under it (`roc_auc`).",
         f"{MET}RocCurveDisplay\nresult = RocCurveDisplay.from_estimator(clf, X_te, y_te).roc_auc"),
        ("name", f"Draw `RocCurveDisplay.from_estimator(clf, X_te, y_te, name={name!r})` and return the label of its line "
                 f"(`disp.line_.get_label()`): the name with the AUC.",
         f"{MET}RocCurveDisplay\nresult = RocCurveDisplay.from_estimator(clf, X_te, y_te, name={name!r}).line_.get_label()",
         {"require": ["RocCurveDisplay", "name"]}),
        ("PrecisionRecallDisplay", "Draw the precision-recall curve (`PrecisionRecallDisplay.from_estimator`) and return "
                                   "its `average_precision`.",
         f"{MET}PrecisionRecallDisplay\nresult = PrecisionRecallDisplay.from_estimator(clf, X_te, y_te).average_precision"),
        ("DetCurveDisplay", "Draw the detection-error-tradeoff curve (`DetCurveDisplay.from_estimator`) and return "
                            "`(disp.fpr, disp.fnr)`.",
         f"{MET}DetCurveDisplay\ndisp = DetCurveDisplay.from_estimator(clf, X_te, y_te)\nresult = (disp.fpr, disp.fnr)"),
        ("CalibrationDisplay", f"Draw the reliability diagram `CalibrationDisplay.from_estimator(clf, X_te, y_te, "
                               f"n_bins={nb})` (`sklearn.calibration`) and return its `prob_true`.",
         f"from sklearn.calibration import CalibrationDisplay\n"
         f"result = CalibrationDisplay.from_estimator(clf, X_te, y_te, n_bins={nb}).prob_true"),
    ]
    return catalog(rng, items, setup, f"{_split_desc(desc)}; `clf = {LR_PIPE}` was fitted on the training part and "
                                      f"`matplotlib.pyplot` is imported as `plt`.", tol=1e-9)


@template(M, "lite")
def importance_basics(rng):
    setup, desc, _ = _binary(rng)
    setup += f"\n{RF_IMP}clf = {RF}.fit(X_tr, y_tr)"
    k, rep, r = rng.randint(2, 4), rng.choice([3, 5]), rng.randint(0, 9)
    pi = f"permutation_importance(clf, X_te, y_te, n_repeats={rep}, random_state={r})"
    items = [
        ("feature_importances_", f"Return the {k} features with the largest impurity-based importance: "
                                 f"`pd.Series(clf.feature_importances_, index=X.columns).nlargest({k})`.",
         f"result = pd.Series(clf.feature_importances_, index=X.columns).nlargest({k})",
         {"require": ["feature_importances_", "nlargest"]}),
        ("permutation_importance", f"Measure importance on held-out data: run `{pi}` (`sklearn.inspection`) and return "
                                   f"`importances_mean`.",
         f"{INS}permutation_importance\nresult = {pi}.importances_mean"),
        ("permutation_importance", f"Run `{pi}` and return the shape of `importances` (one row per feature, one column "
                                   f"per repeat) as a tuple.",
         f"{INS}permutation_importance\nresult = {pi}.importances.shape"),
        ("importances_std", f"Run `{pi}` and return `importances_std`: how much the importance varies between "
                            f"shuffles.", f"{INS}permutation_importance\nresult = {pi}.importances_std"),
        ("importances_mean", f"Run `{pi}` and return the top {k} features as a Series: `pd.Series(r.importances_mean, "
                             f"index=X.columns).sort_values(ascending=False).head({k})`.",
         f"{INS}permutation_importance\nr = {pi}\n"
         f"result = pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False).head({k})",
         {"require": ["permutation_importance", "sort_values"]}),
        ("feature_importances_", "Return the name of the single most important feature by `feature_importances_`.",
         "result = X.columns[clf.feature_importances_.argmax()]"),
    ]
    return catalog(rng, items, setup, f"{_split_desc(desc)}; `clf = {RF}` was fitted on the training part.", tol=1e-9)


@template(M, "lite")
def curve_display_basics(rng):
    s = seeded(rng)
    setup = (PLT + "from sklearn.datasets import load_breast_cancer\nfrom sklearn.preprocessing import StandardScaler\n"
             f"data = load_breast_cancer(as_frame=True)\norder = np.random.default_rng({s}).permutation(569)\n"
             "X, y = data.data.iloc[order, :10], data.target.iloc[order]\nXs = StandardScaler().fit_transform(X)")
    sizes = rng.choice(["np.linspace(0.2, 1.0, 4)", "np.linspace(0.1, 1.0, 5)", "np.linspace(0.3, 1.0, 3)"])
    cv = rng.choice([3, 4])
    lc = f"LearningCurveDisplay.from_estimator(LogisticRegression(max_iter=1000), Xs, y, cv={cv}, train_sizes={sizes})"
    vc = f"ValidationCurveDisplay.from_estimator(SVC(), Xs, y, param_name='C', param_range=np.logspace(-2, 2, 5), cv={cv})"
    lci = f"{MS}LearningCurveDisplay\nfrom sklearn.linear_model import LogisticRegression\n"
    vci = f"{MS}ValidationCurveDisplay\nfrom sklearn.svm import SVC\n"
    items = [
        ("LearningCurveDisplay", f"Draw a learning curve: `{lc}` (`sklearn.model_selection`). Return the absolute "
                                 f"training-set sizes it used (`train_sizes`).", f"{lci}result = {lc}.train_sizes"),
        ("LearningCurveDisplay", f"Draw `{lc}` and return the mean validation score per training size "
                                 f"(`test_scores.mean(axis=1)`).", f"{lci}result = {lc}.test_scores.mean(axis=1)"),
        ("score_type", f"Draw `{lc[:-1]}, score_type='both')` (training and validation curves) and return how many "
                       f"lines its Axes holds (`len(disp.ax_.get_lines())`).",
         f"{lci}disp = {lc[:-1]}, score_type='both')\nresult = len(disp.ax_.get_lines())",
         {"require": ["LearningCurveDisplay", "score_type"]}),
        ("ValidationCurveDisplay", f"Draw a validation curve over `C`: `{vc}`. Return the mean validation score per `C`.",
         f"{vci}result = {vc}.test_scores.mean(axis=1)"),
        ("param_range", f"Draw `{vc}` and return the `C` with the best mean validation score.",
         f"{vci}disp = {vc}\nresult = float(disp.param_range[disp.test_scores.mean(axis=1).argmax()])",
         {"require": ["ValidationCurveDisplay", "param_range"]}),
        ("close", f"Draw `{lc}`, then free the figure with `plt.close('all')` and return how many figures are still "
                  f"open (`len(plt.get_fignums())`).", f"{lci}disp = {lc}\nplt.close('all')\nresult = len(plt.get_fignums())",
         {"require": ["close"]}),
    ]
    return catalog(rng, items, setup, "`X` holds the 10 'mean ...' measurements of 569 breast tumours (rows shuffled), `y` = `data.target`"
                                      " (0 malignant, 1 benign) and `Xs = StandardScaler().fit_transform(X)`; "
                                      "`matplotlib.pyplot` is imported as `plt`.", tol=1e-9)


@template(M, "lite")
def regression_and_boundary_displays(rng):
    s = seeded(rng)
    a = rng.choice([0.1, 1.0, 10.0])
    setup = (PLT + "from sklearn.datasets import load_diabetes, make_moons\nfrom sklearn.linear_model import Ridge\n"
             f"Xr, yr = load_diabetes(return_X_y=True)\npred = Ridge(alpha={a}).fit(Xr, yr).predict(Xr)\n"
             f"X2, y2 = make_moons(n_samples=150, noise=0.25, random_state={s})")
    items = [
        ("PredictionErrorDisplay", "Plot actual against predicted values: `PredictionErrorDisplay.from_predictions(yr, "
                                   "pred, kind='actual_vs_predicted')` (`sklearn.metrics`). Return the Axes labels "
                                   "`(xlabel, ylabel)`.",
         f"{MET}PredictionErrorDisplay\ndisp = PredictionErrorDisplay.from_predictions(yr, pred, kind='actual_vs_predicted')\n"
         "result = (disp.ax_.get_xlabel(), disp.ax_.get_ylabel())", {"require": ["PredictionErrorDisplay", "kind"]}),
        ("PredictionErrorDisplay", "The default plot shows residuals against predictions. Draw "
                                   "`PredictionErrorDisplay.from_predictions(yr, pred)` and return its y label.",
         f"{MET}PredictionErrorDisplay\nresult = PredictionErrorDisplay.from_predictions(yr, pred).ax_.get_ylabel()"),
        ("DecisionBoundaryDisplay", "Fit `SVC(gamma=2)` on `X2, y2` and draw its regions with "
                                    "`DecisionBoundaryDisplay.from_estimator(svm, X2, response_method='predict', "
                                    "grid_resolution=50)` (`sklearn.inspection`). Return the fraction of the grid "
                                    "predicted as class 1 (`disp.response.mean()`).",
         f"from sklearn.svm import SVC\n{INS}DecisionBoundaryDisplay\nsvm = SVC(gamma=2).fit(X2, y2)\n"
         "disp = DecisionBoundaryDisplay.from_estimator(svm, X2, response_method='predict', grid_resolution=50)\n"
         "result = float(disp.response.mean())", {"require": ["DecisionBoundaryDisplay", "response_method"]}),
        ("DecisionBoundaryDisplay", "Fit `SVC(gamma=2)` on `X2, y2`, draw `DecisionBoundaryDisplay.from_estimator(svm, X2, "
                                    "response_method='predict', grid_resolution=40, alpha=0.4)` and return the shape of "
                                    "its `response` grid.",
         f"from sklearn.svm import SVC\n{INS}DecisionBoundaryDisplay\nsvm = SVC(gamma=2).fit(X2, y2)\n"
         "result = DecisionBoundaryDisplay.from_estimator(svm, X2, response_method='predict', grid_resolution=40, alpha=0.4).response.shape",
         {"require": ["DecisionBoundaryDisplay", "alpha"]}),
        ("scatter", "Fit `SVC(gamma=2)` on `X2, y2`, draw its `DecisionBoundaryDisplay` (`response_method='predict'`, "
                    "`grid_resolution=40`), then add the points on the same Axes with `disp.ax_.scatter(X2[:, 0], X2[:, "
                    "1], c=y2, edgecolor='k')`. Return the number of points in that scatter "
                    "(`len(sc.get_offsets())`).",
         f"from sklearn.svm import SVC\n{INS}DecisionBoundaryDisplay\nsvm = SVC(gamma=2).fit(X2, y2)\n"
         "disp = DecisionBoundaryDisplay.from_estimator(svm, X2, response_method='predict', grid_resolution=40)\n"
         "sc = disp.ax_.scatter(X2[:, 0], X2[:, 1], c=y2, edgecolor='k')\nresult = len(sc.get_offsets())",
         {"require": ["scatter", "ax_"]}),
    ]
    return catalog(rng, items, setup, f"`Xr`, `yr` hold the diabetes data and `pred` the in-sample predictions of "
                                      f"`Ridge(alpha={a})`; `X2`, `y2` hold 150 points of `make_moons`. "
                                      f"`matplotlib.pyplot` is imported as `plt`.", tol=1e-9)


# ================================================================= easy

@template(M, "easy")
def confusion_matrix_display_tasks(rng):
    setup, desc, names = _binary(rng)
    setup += f"\n{LR_IMP}clf = {LR_PIPE}.fit(X_tr, y_tr)"
    task = rng.choice(["true", "pred", "all", "text", "ticks"])
    head = f"{_split_desc(desc)}; `clf = {LR_PIPE}` was fitted on the training part."
    cm = f"{MET}ConfusionMatrixDisplay\n"
    if task in ("true", "pred", "all"):
        what = {"true": "each row (true class) sums to 1: the diagonal holds the recall of each class",
                "pred": "each column (predicted class) sums to 1: the diagonal holds the precision of each class",
                "all": "all cells sum to 1"}[task]
        return P(f"{head} Draw `ConfusionMatrixDisplay.from_estimator(clf, X_te, y_te, normalize={task!r})` ({what}) and "
                 f"return `(diagonal, row sums)` of the displayed matrix.", setup,
                 cm + f"m = ConfusionMatrixDisplay.from_estimator(clf, X_te, y_te, normalize={task!r}).confusion_matrix\n"
                      "result = (np.diag(m), m.sum(axis=1))", require=["ConfusionMatrixDisplay", "normalize"], tol=1e-9)
    if task == "text":
        return P(f"{head} Draw `ConfusionMatrixDisplay.from_estimator(clf, X_te, y_te)` and return the numbers written in "
                 f"its cells, as strings in row order (`[t.get_text() for t in disp.text_.ravel()]`).", setup,
                 cm + "disp = ConfusionMatrixDisplay.from_estimator(clf, X_te, y_te)\nresult = [t.get_text() for t in disp.text_.ravel()]",
                 require=["ConfusionMatrixDisplay", "text_"])
    return P(f"{head} Draw `ConfusionMatrixDisplay.from_predictions(y_te, clf.predict(X_te), display_labels={names})` "
             f"and return `(x tick labels, y tick labels, x label)` of its Axes.", setup,
             cm + f"disp = ConfusionMatrixDisplay.from_predictions(y_te, clf.predict(X_te), display_labels={names})\nax = disp.ax_\n"
                  "result = ([t.get_text() for t in ax.get_xticklabels()], [t.get_text() for t in ax.get_yticklabels()], ax.get_xlabel())",
             require=["from_predictions", "display_labels"])


@template(M, "easy")
def roc_curves_one_axes(rng):
    setup, desc, _ = _binary(rng)
    k = rng.choice([5, 15])
    models = (f"models = {{'LR': {LR_PIPE}, 'RF': {RF}, 'kNN': make_pipeline(StandardScaler(), "
              f"KNeighborsClassifier(n_neighbors={k}))}}")
    imp = f"{LR_IMP}{RF_IMP}from sklearn.neighbors import KNeighborsClassifier\n{MET}RocCurveDisplay\n"
    loop = ("fig, ax = plt.subplots()\naucs = {}\nfor name, m in models.items():\n"
            "    aucs[name] = RocCurveDisplay.from_estimator(m.fit(X_tr, y_tr), X_te, y_te, ax=ax, name=name).roc_auc\n")
    head = (f"{_split_desc(desc)}. Overlay the test ROC curves of three models on **one** Axes (`fig, ax = "
            f"plt.subplots()`, then `RocCurveDisplay.from_estimator(..., ax=ax, name=name)` for each, after fitting it "
            f"on the training part): `{models[9:]}`.")
    task = rng.choice(["legend", "aucs", "lines"])
    if task == "legend":
        return P(f"{head} Return the legend texts (`[t.get_text() for t in ax.get_legend().get_texts()]`).", setup,
                 imp + models + "\n" + loop + "result = [t.get_text() for t in ax.get_legend().get_texts()]",
                 require=["RocCurveDisplay", "subplots"])
    if task == "aucs":
        return P(f"{head} Return a dict `{{name: roc_auc}}`.", setup, imp + models + "\n" + loop + "result = aucs",
                 require=["RocCurveDisplay", "subplots"], tol=1e-9)
    return P(f"{head} Pass `plot_chance_level=True` for the last model only, then return `(number of lines on ax, name "
             f"of the best model by AUC)`.", setup,
             imp + models + "\nfig, ax = plt.subplots()\naucs = {}\nfor i, (name, m) in enumerate(models.items()):\n"
                            "    aucs[name] = RocCurveDisplay.from_estimator(m.fit(X_tr, y_tr), X_te, y_te, ax=ax, name=name, "
                            "plot_chance_level=(i == len(models) - 1)).roc_auc\n"
                            "result = (len(ax.get_lines()), max(aucs, key=aucs.get))",
             require=["RocCurveDisplay", "plot_chance_level"])


@template(M, "easy")
def permutation_ranking(rng):
    setup, desc, _ = _binary(rng)
    setup += f"\n{RF_IMP}clf = {RF}.fit(X_tr, y_tr)"
    k, r = rng.randint(2, 4), rng.randint(0, 9)
    task = rng.choice(["top", "auc", "train_vs_test"])
    head = f"{_split_desc(desc)}; `clf = {RF}` was fitted on the training part."
    if task == "top":
        return P(f"{head} Rank the features by permutation importance on the **test** set (`permutation_importance(clf, "
                 f"X_te, y_te, n_repeats=5, random_state={r})`) and return the names of the top {k} as a list.", setup,
                 f"{INS}permutation_importance\nr = permutation_importance(clf, X_te, y_te, n_repeats=5, random_state={r})\n"
                 f"result = list(pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False).index[:{k}])",
                 require=["permutation_importance", "importances_mean"])
    if task == "auc":
        return P(f"{head} Permutation importance can use any scorer. Return `importances_mean` of "
                 f"`permutation_importance(clf, X_te, y_te, scoring='roc_auc', n_repeats=5, random_state={r})` as a "
                 f"Series indexed by the column names.", setup,
                 f"{INS}permutation_importance\n"
                 f"r = permutation_importance(clf, X_te, y_te, scoring='roc_auc', n_repeats=5, random_state={r})\n"
                 "result = pd.Series(r.importances_mean, index=X.columns)", require=["permutation_importance", "scoring"],
                 tol=1e-9)
    return P(f"{head} A forest memorizes its training rows, so importances computed there look inflated. Return the "
             f"**sum** of `importances_mean` on the training set and on the test set (`n_repeats=5, random_state={r}`), "
             f"as `(train, test)`.", setup,
             f"{INS}permutation_importance\n"
             f"tr = permutation_importance(clf, X_tr, y_tr, n_repeats=5, random_state={r}).importances_mean.sum()\n"
             f"te = permutation_importance(clf, X_te, y_te, n_repeats=5, random_state={r}).importances_mean.sum()\n"
             "result = (tr, te)", require=["permutation_importance"], tol=1e-9)


@template(M, "easy")
def knn_decision_boundaries(rng):
    setup, desc = _moons(rng)
    k1, k2 = 1, rng.choice([15, 25, 30])
    g = rng.choice([30, 50])
    head = (f"{desc}. Draw the decision regions of `KNeighborsClassifier(n_neighbors=k)` (fitted on `X2, y2`) for k = "
            f"{k1} and k = {k2} with `DecisionBoundaryDisplay.from_estimator(knn, X2, response_method='predict', "
            f"grid_resolution={g})`.")
    loop = (f"from sklearn.neighbors import KNeighborsClassifier\n{INS}DecisionBoundaryDisplay\nresp = {{}}\n"
            f"for k in ({k1}, {k2}):\n    knn = KNeighborsClassifier(n_neighbors=k).fit(X2, y2)\n"
            f"    resp[k] = DecisionBoundaryDisplay.from_estimator(knn, X2, response_method='predict', grid_resolution={g}).response\n")
    task = rng.choice(["changes", "share", "agree"])
    if task == "changes":
        return P(f"{head} A wiggly boundary changes label often: for each k, count the horizontally adjacent grid cells "
                 f"with different labels (`(r[:, 1:] != r[:, :-1]).sum()`) and return `(count_k{k1}, count_k{k2})`.",
                 setup, loop + f"result = tuple(int((resp[k][:, 1:] != resp[k][:, :-1]).sum()) for k in ({k1}, {k2}))",
                 require=["DecisionBoundaryDisplay", "KNeighborsClassifier"])
    if task == "share":
        return P(f"{head} Return the fraction of the grid assigned to class 0 for each k, as a tuple.", setup,
                 loop + f"result = tuple(float((resp[k] == 0).mean()) for k in ({k1}, {k2}))",
                 require=["DecisionBoundaryDisplay", "KNeighborsClassifier"], tol=1e-9)
    return P(f"{head} Return the fraction of grid cells where the two models agree.", setup,
             loop + f"result = float((resp[{k1}] == resp[{k2}]).mean())",
             require=["DecisionBoundaryDisplay", "KNeighborsClassifier"], tol=1e-9)


@template(M, "easy")
def curve_values_match_metrics(rng):
    setup, desc, _ = _binary(rng)
    setup += f"\n{LR_IMP}clf = {LR_PIPE}.fit(X_tr, y_tr)"
    task = rng.choice(["ap", "det", "chance"])
    head = f"{_split_desc(desc)}; `clf = {LR_PIPE}` was fitted on the training part."
    if task == "ap":
        return P(f"{head} A Display stores the metric it shows. Return `(PrecisionRecallDisplay.from_estimator(clf, X_te, "
                 f"y_te).average_precision, average_precision_score(y_te, p))` with `p` = the test P(class 1).", setup,
                 f"{MET}PrecisionRecallDisplay, average_precision_score\np = clf.predict_proba(X_te)[:, 1]\n"
                 "result = (PrecisionRecallDisplay.from_estimator(clf, X_te, y_te).average_precision, average_precision_score(y_te, p))",
                 require=["PrecisionRecallDisplay", "average_precision_score"], tol=1e-9)
    if task == "det":
        return P(f"{head} Draw `DetCurveDisplay.from_estimator(clf, X_te, y_te)` and return the point of its curve where "
                 f"the false positive and false negative rates are closest: `(fpr, fnr)` at the index minimizing "
                 f"|fpr − fnr|.", setup,
                 f"{MET}DetCurveDisplay\nd = DetCurveDisplay.from_estimator(clf, X_te, y_te)\ni = int(np.abs(d.fpr - d.fnr).argmin())\n"
                 "result = (float(d.fpr[i]), float(d.fnr[i]))", require=["DetCurveDisplay"], tol=1e-9)
    return P(f"{head} Draw `PrecisionRecallDisplay.from_estimator(clf, X_te, y_te, plot_chance_level=True)` and return "
             f"`(number of lines on its Axes, the chance line's height)`: the chance level of a PR curve is the share of "
             f"positives (`disp.chance_level_.get_ydata()[0]`).", setup,
             f"{MET}PrecisionRecallDisplay\nd = PrecisionRecallDisplay.from_estimator(clf, X_te, y_te, plot_chance_level=True)\n"
             "result = (len(d.ax_.get_lines()), float(d.chance_level_.get_ydata()[0]))",
             require=["PrecisionRecallDisplay", "plot_chance_level"], tol=1e-9)


@template(M, "easy")
def calibration_display_tasks(rng):
    setup, desc, _ = _binary(rng)
    nb = rng.choice([4, 5, 8])
    task = rng.choice(["match", "pred", "gap"])
    imp = f"from sklearn.calibration import CalibrationDisplay, calibration_curve\n{RF_IMP}"
    head = f"{_split_desc(desc)}. Fit `{RF}` on the training part; `p` = its test P(class 1)."
    sol = imp + f"clf = {RF}.fit(X_tr, y_tr)\np = clf.predict_proba(X_te)[:, 1]\n"
    if task == "match":
        return P(f"{head} Return `(CalibrationDisplay.from_estimator(clf, X_te, y_te, n_bins={nb}).prob_true, "
                 f"calibration_curve(y_te, p, n_bins={nb})[0])` (the same).", setup,
                 sol + f"result = (CalibrationDisplay.from_estimator(clf, X_te, y_te, n_bins={nb}).prob_true, calibration_curve(y_te, p, n_bins={nb})[0])",
                 require=["CalibrationDisplay"], tol=1e-9)
    if task == "pred":
        return P(f"{head} Draw it from the probabilities: `CalibrationDisplay.from_predictions(y_te, p, n_bins={nb})`, "
                 f"and return `(prob_pred, number of points on its curve)`.", setup,
                 sol + f"d = CalibrationDisplay.from_predictions(y_te, p, n_bins={nb})\nresult = (d.prob_pred, len(d.line_.get_xdata()))",
                 require=["CalibrationDisplay", "from_predictions"], tol=1e-9)
    return P(f"{head} Draw `CalibrationDisplay.from_predictions(y_te, p, n_bins={nb})` and return how many of its points "
             f"lie **below** the diagonal (predicted probability above the observed rate: over-confident).", setup,
             sol + f"d = CalibrationDisplay.from_predictions(y_te, p, n_bins={nb})\nresult = int((d.prob_pred > d.prob_true).sum())",
             require=["CalibrationDisplay"])


@template(M, "easy")
def impurity_vs_permutation(rng):
    setup, desc, _ = _binary(rng)
    setup += f"\n{RF_IMP}clf = {RF}.fit(X_tr, y_tr)"
    r = rng.randint(0, 9)
    if rng.random() < 0.5:
        return P(f"{_split_desc(desc)}; `clf = {RF}` was fitted on the training part. Do the two importance measures "
                 f"agree? Return `(top by feature_importances_, top by permutation importance on the test set)` "
                 f"(feature names; `permutation_importance(clf, X_te, y_te, n_repeats=5, random_state={r})`).", setup,
                 f"{INS}permutation_importance\nr = permutation_importance(clf, X_te, y_te, n_repeats=5, random_state={r})\n"
                 "result = (X.columns[clf.feature_importances_.argmax()], X.columns[r.importances_mean.argmax()])",
                 require=["feature_importances_", "permutation_importance"])
    return P(f"{_split_desc(desc)}; `clf = {RF}` was fitted on the training part. Return a DataFrame indexed by feature "
             f"with columns `'impurity'` (`feature_importances_`) and `'permutation'` (test `importances_mean`, "
             f"`n_repeats=5, random_state={r}`), sorted by `'permutation'` descending.", setup,
             f"{INS}permutation_importance\nr = permutation_importance(clf, X_te, y_te, n_repeats=5, random_state={r})\n"
             "result = pd.DataFrame({'impurity': clf.feature_importances_, 'permutation': r.importances_mean}, "
             "index=X.columns).sort_values('permutation', ascending=False)",
             require=["feature_importances_", "permutation_importance"], tol=1e-9)


@template(M, "easy")
def learning_curve_tasks(rng):
    setup, desc, _ = _binary(rng)
    model, imp = rng.choice([(LR_PIPE, LR_IMP), ("DecisionTreeClassifier(max_depth=4, random_state=0)",
                                                 "from sklearn.tree import DecisionTreeClassifier\n")])
    cv = rng.choice([3, 5])
    lc = (f"LearningCurveDisplay.from_estimator({model}, X, y, cv={cv}, train_sizes=np.linspace(0.2, 1.0, 4), "
          f"score_type='both')")
    sol = f"{MS}LearningCurveDisplay\n{imp}d = {lc}\n"
    task = rng.choice(["gap", "gain", "sizes"])
    head = f"{desc}. Draw `{lc}` (training and validation curves)."
    if task == "gap":
        return P(f"{head} Return the gap between the mean training and mean validation score at each size (an array).",
                 setup, sol + "result = d.train_scores.mean(axis=1) - d.test_scores.mean(axis=1)",
                 require=["LearningCurveDisplay"], tol=1e-9)
    if task == "gain":
        return P(f"{head} Does more data help? Return the mean validation score at the largest size minus the one at "
                 f"the smallest.", setup, sol + "m = d.test_scores.mean(axis=1)\nresult = float(m[-1] - m[0])",
                 require=["LearningCurveDisplay"], tol=1e-9)
    return P(f"{head} Return `(train_sizes, test_scores.shape)`: one row per size, one column per fold.", setup,
             sol + "result = (d.train_sizes, d.test_scores.shape)", require=["LearningCurveDisplay", "train_sizes"])


@template(M, "easy")
def validation_curve_tasks(rng):
    setup, desc, _ = _binary(rng)
    task = rng.choice(["depth", "c", "overfit"])
    if task in ("depth", "overfit"):
        vc = ("ValidationCurveDisplay.from_estimator(DecisionTreeClassifier(random_state=0), X, y, param_name='max_depth', "
              "param_range=[1, 2, 3, 5, 8, 12], cv=3)")
        sol = f"{MS}ValidationCurveDisplay\nfrom sklearn.tree import DecisionTreeClassifier\nd = {vc}\n"
        if task == "depth":
            return P(f"{desc}. Draw `{vc}` and return the depth with the best mean validation score (the first on ties).",
                     setup, sol + "result = int(d.param_range[int(d.test_scores.mean(axis=1).argmax())])",
                     require=["ValidationCurveDisplay"])
        return P(f"{desc}. Draw `{vc}` and return `(mean train scores, mean validation scores)`: the training score "
                 f"keeps rising while the validation score stalls.", setup,
                 sol + "result = (d.train_scores.mean(axis=1), d.test_scores.mean(axis=1))",
                 require=["ValidationCurveDisplay"], tol=1e-9)
    vc = (f"ValidationCurveDisplay.from_estimator({LR_PIPE}, X, y, param_name='logisticregression__C', "
          f"param_range=np.logspace(-3, 2, 6), cv=3)")
    return P(f"{desc}. A validation curve works on a pipeline parameter too: draw `{vc}` and return `(best C, its mean "
             f"validation score)`.", setup,
             f"{MS}ValidationCurveDisplay\n{LR_IMP}d = {vc}\nm = d.test_scores.mean(axis=1)\n"
             "result = (float(d.param_range[int(m.argmax())]), float(m.max()))", require=["ValidationCurveDisplay", "logspace"],
             tol=1e-9)


@template(M, "easy")
def prediction_error_tasks(rng):
    s = seeded(rng)
    t = table(rng, kind="reg", numeric_only=True, n=(150, 260))
    setup = (PLT + t.setup + "\nfrom sklearn.model_selection import train_test_split\n"
             f"X_tr, X_te, y_tr, y_te = train_test_split(df, y, random_state={s})")
    task = rng.choice(["resid", "kind", "subsample"])
    head = f"{t.desc}, split with `train_test_split(df, y, random_state={s})`. Fit `Ridge()` on the training part."
    sol = (f"from sklearn.linear_model import Ridge\n{MET}PredictionErrorDisplay\npred = Ridge().fit(X_tr, y_tr).predict(X_te)\n")
    if task == "resid":
        return P(f"{head} Draw `PredictionErrorDisplay.from_predictions(y_te, pred)` (residuals against predictions) and "
                 f"return the y values of its points (`disp.scatter_.get_offsets()[:, 1]`): the residuals y − ŷ.", setup,
                 sol + "d = PredictionErrorDisplay.from_predictions(y_te, pred)\nresult = np.asarray(d.scatter_.get_offsets())[:, 1]",
                 require=["PredictionErrorDisplay"], tol=1e-6)
    if task == "kind":
        return P(f"{head} Draw the test predictions with `kind='actual_vs_predicted'` and return `(xlabel, ylabel, number "
                 f"of points)`.", setup,
                 sol + "d = PredictionErrorDisplay.from_predictions(y_te, pred, kind='actual_vs_predicted')\n"
                       "result = (d.ax_.get_xlabel(), d.ax_.get_ylabel(), len(d.scatter_.get_offsets()))",
                 require=["PredictionErrorDisplay", "kind"])
    m = rng.choice([20, 30])
    return P(f"{head} Big test sets get thinned out: draw `PredictionErrorDisplay.from_predictions(y_te, pred, "
             f"subsample={m}, random_state=0)` and return `(points drawn, len(disp.y_true))`.", setup,
             sol + f"d = PredictionErrorDisplay.from_predictions(y_te, pred, subsample={m}, random_state=0)\n"
                   "result = (len(d.scatter_.get_offsets()), len(d.y_true))", require=["PredictionErrorDisplay", "subsample"])


@template(M, "easy")
def partial_dependence_basics(rng):
    setup, desc, _ = _binary(rng)
    setup += f"\n{LR_IMP}clf = {LR_PIPE}.fit(X_tr, y_tr)"
    g = rng.choice([5, 8, 10])
    task = rng.choice(["avg", "grid", "ice_lines"])
    head = f"{_split_desc(desc)}; `clf = {LR_PIPE}` was fitted on the training part."
    feat = "X.columns[0]"
    if task == "avg":
        return P(f"{head} Draw `PartialDependenceDisplay.from_estimator(clf, X_te, features=[X.columns[0]], "
                 f"grid_resolution={g})` and return the average curve (`disp.pd_results[0]['average'][0]`).", setup,
                 f"{INS}PartialDependenceDisplay\nd = PartialDependenceDisplay.from_estimator(clf, X_te, features=[{feat}], grid_resolution={g})\n"
                 "result = d.pd_results[0]['average'][0]", require=["PartialDependenceDisplay", "pd_results"], tol=1e-9)
    if task == "grid":
        return P(f"{head} Draw `PartialDependenceDisplay.from_estimator(clf, X_te, features=[X.columns[0]], "
                 f"grid_resolution={g})` and return the grid of feature values it used "
                 f"(`disp.pd_results[0]['grid_values'][0]`): evenly spaced between the 5th and 95th percentiles.", setup,
                 f"{INS}PartialDependenceDisplay\nd = PartialDependenceDisplay.from_estimator(clf, X_te, features=[{feat}], grid_resolution={g})\n"
                 "result = d.pd_results[0]['grid_values'][0]", require=["PartialDependenceDisplay"], tol=1e-9)
    m = rng.choice([10, 20, 30])
    return P(f"{head} Draw partial dependence **and** individual (ICE) curves: `PartialDependenceDisplay.from_estimator("
             f"clf, X_te, features=[X.columns[0]], kind='both', subsample={m}, random_state=0, grid_resolution={g})`. "
             f"Return how many lines its Axes holds (`len(disp.axes_[0, 0].get_lines())`).", setup,
             f"{INS}PartialDependenceDisplay\nd = PartialDependenceDisplay.from_estimator(clf, X_te, features=[{feat}], "
             f"kind='both', subsample={m}, random_state=0, grid_resolution={g})\nresult = len(d.axes_[0, 0].get_lines())",
             require=["PartialDependenceDisplay", "kind"])


# ================================================================= medium

@template(M, "medium")
def model_comparison_plots(rng):
    setup, desc, names = _binary(rng)
    models = (f"{LR_IMP}{RF_IMP}from sklearn.naive_bayes import GaussianNB\n"
              f"models = {{'LR': {LR_PIPE}, 'RF': {RF}, 'NB': GaussianNB()}}\n"
              "for m in models.values():\n    m.fit(X_tr, y_tr)\n")
    items = [
        ("RocCurveDisplay", "Overlay their test ROC curves on one Axes (`fig, ax = plt.subplots()`, `ax=ax`, "
                            "`name=...`) and return the legend texts.",
         f"{MET}RocCurveDisplay\nfig, ax = plt.subplots()\n"
         "for name, m in models.items():\n    RocCurveDisplay.from_estimator(m, X_te, y_te, ax=ax, name=name)\n"
         "result = [t.get_text() for t in ax.get_legend().get_texts()]", {"require": ["RocCurveDisplay", "ax"]}),
        ("PrecisionRecallDisplay", "Overlay their precision-recall curves on one Axes and return `{name: "
                                   "average_precision}`.",
         f"{MET}PrecisionRecallDisplay\nfig, ax = plt.subplots()\n"
         "result = {name: PrecisionRecallDisplay.from_estimator(m, X_te, y_te, ax=ax, name=name).average_precision for name, m in models.items()}"),
        ("ConfusionMatrixDisplay", f"Pick the model with the best test ROC AUC (`RocCurveDisplay.from_estimator(...)"
                                   f".roc_auc`) and return `(its name, its confusion matrix normalized over true "
                                   f"classes)` from `ConfusionMatrixDisplay.from_estimator(..., display_labels={names}, "
                                   f"normalize='true')`.",
         f"{MET}RocCurveDisplay, ConfusionMatrixDisplay\n"
         "auc = {name: RocCurveDisplay.from_estimator(m, X_te, y_te).roc_auc for name, m in models.items()}\nbest = max(auc, key=auc.get)\n"
         f"result = (best, ConfusionMatrixDisplay.from_estimator(models[best], X_te, y_te, display_labels={names}, normalize='true').confusion_matrix)",
         {"require": ["ConfusionMatrixDisplay", "RocCurveDisplay"]}),
        ("DetCurveDisplay", "Overlay their DET curves on one Axes and return `{name: smallest false negative rate "
                            "reached while the false positive rate is at most 0.1}` (from `disp.fpr`, `disp.fnr`).",
         f"{MET}DetCurveDisplay\nfig, ax = plt.subplots()\nresult = {{}}\nfor name, m in models.items():\n"
         "    d = DetCurveDisplay.from_estimator(m, X_te, y_te, ax=ax, name=name)\n"
         "    result[name] = float(d.fnr[d.fpr <= 0.1].min()) if (d.fpr <= 0.1).any() else 1.0"),
        ("CalibrationDisplay", "Overlay their reliability curves (`CalibrationDisplay.from_estimator(..., n_bins=5, "
                               "ax=ax, name=...)`) and return the name of the model whose points are closest to the "
                               "diagonal (smallest mean |prob_true − prob_pred|).",
         "from sklearn.calibration import CalibrationDisplay\nfig, ax = plt.subplots()\ngap = {}\nfor name, m in models.items():\n"
         "    d = CalibrationDisplay.from_estimator(m, X_te, y_te, n_bins=5, ax=ax, name=name)\n"
         "    gap[name] = float(np.abs(d.prob_true - d.prob_pred).mean())\nresult = min(gap, key=gap.get)"),
        ("subplots", "Draw one confusion matrix per model side by side (`fig, axes = plt.subplots(1, 3)`, "
                     "`ConfusionMatrixDisplay.from_estimator(m, X_te, y_te, ax=axes[i])`), title each Axes with "
                     "`axes[i].set_title(name)`, and return `(number of titled Axes in fig.axes, their titles)` (the "
                     "colorbars are untitled Axes).",
         f"{MET}ConfusionMatrixDisplay\nfig, axes = plt.subplots(1, 3)\n"
         "for i, (name, m) in enumerate(models.items()):\n    ConfusionMatrixDisplay.from_estimator(m, X_te, y_te, ax=axes[i])\n"
         "    axes[i].set_title(name)\ntitles = [a.get_title() for a in fig.axes if a.get_title()]\nresult = (len(titles), titles)",
         {"require": ["subplots", "ConfusionMatrixDisplay"]}),
    ]
    return catalog(rng, items, setup + "\n" + models, f"{_split_desc(desc)}; `models` maps `'LR'`, `'RF'`, `'NB'` to "
                                                      f"`{LR_PIPE}`, `{RF}` and `GaussianNB()`, all fitted on the "
                                                      f"training part.", tol=1e-9)


@template(M, "medium")
def permutation_importance_tasks(rng):
    t = table(rng, kind="clf", numeric_only=True, n=(250, 400))
    s, r = seeded(rng), rng.randint(0, 9)
    setup = (PLT + t.setup + f"\nfrom sklearn.model_selection import train_test_split\n"
             f"X_tr, X_te, y_tr, y_te = train_test_split(df, y, stratify=y, random_state={s})")
    rf = f"{RF_IMP}clf = {RF}.fit(X_tr, y_tr)\n"
    pi = f"{INS}permutation_importance\n"
    items = [
        ("permutation_importance", f"Fit `{RF}` on the training part and return the test permutation importances "
                                   f"(`n_repeats=5, random_state={r}`) as a Series indexed by column, largest first.",
         rf + pi + f"r = permutation_importance(clf, X_te, y_te, n_repeats=5, random_state={r})\n"
                   "result = pd.Series(r.importances_mean, index=df.columns).sort_values(ascending=False)"),
        ("scoring", f"Fit `{RF}` on the training part. The classes are unbalanced, so measure the drop in balanced "
                    f"accuracy: `permutation_importance(clf, X_te, y_te, scoring='balanced_accuracy', n_repeats=5, "
                    f"random_state={r})`. Return the name of the most important column.",
         rf + pi + f"r = permutation_importance(clf, X_te, y_te, scoring='balanced_accuracy', n_repeats=5, random_state={r})\n"
                   "result = df.columns[r.importances_mean.argmax()]", {"require": ["permutation_importance", "scoring"]}),
        ("feature_importances_", f"Impurity importance is biased toward columns with many distinct values. Add a pure "
                                 f"noise column `noise = np.random.default_rng({s}).normal(size=len(df))` to copies of "
                                 f"both splits (same values as in `df`, via the index), refit `{RF}`, and return "
                                 f"`(impurity importance of noise, test permutation importance of noise)` "
                                 f"(`n_repeats=5, random_state={r}`).",
         f"{RF_IMP}{pi}noise = pd.Series(np.random.default_rng({s}).normal(size=len(df)), index=df.index)\n"
         "A, B = X_tr.assign(noise=noise), X_te.assign(noise=noise)\n"
         f"clf = {RF}.fit(A, y_tr)\nr = permutation_importance(clf, B, y_te, n_repeats=5, random_state={r})\n"
         "result = (float(clf.feature_importances_[-1]), float(r.importances_mean[-1]))"),
        ("importances", f"Fit `{RF}` on the training part, run `permutation_importance(clf, X_te, y_te, n_repeats=8, "
                        f"random_state={r})` and return, for the most important column, the 8 individual drops "
                        f"(`importances[j]`).",
         rf + pi + f"r = permutation_importance(clf, X_te, y_te, n_repeats=8, random_state={r})\n"
                   "result = r.importances[r.importances_mean.argmax()]"),
        ("importances_mean", f"Fit `{RF}` on the training part and return the columns whose test permutation importance "
                             f"(`n_repeats=5, random_state={r}`) is at most 0: shuffling them does not hurt.",
         rf + pi + f"r = permutation_importance(clf, X_te, y_te, n_repeats=5, random_state={r})\n"
                   "result = [c for c, v in zip(df.columns, r.importances_mean) if v <= 0]"),
        ("permutation_importance", f"Permutation importance works for any model, even a pipeline: fit `{LR_PIPE}` on "
                                   f"the training part and return its test importances (`n_repeats=5, "
                                   f"random_state={r}`) as a dict `{{column: importance}}`.",
         LR_IMP + pi + f"clf = {LR_PIPE}.fit(X_tr, y_tr)\nr = permutation_importance(clf, X_te, y_te, n_repeats=5, random_state={r})\n"
                       "result = dict(zip(df.columns, r.importances_mean))"),
        ("max_samples", f"On big test sets you can subsample: fit `{RF}` on the training part and return "
                        f"`importances_mean` of `permutation_importance(clf, X_te, y_te, n_repeats=5, max_samples=0.5, "
                        f"random_state={r})`.",
         rf + pi + f"result = permutation_importance(clf, X_te, y_te, n_repeats=5, max_samples=0.5, random_state={r}).importances_mean",
         {"require": ["permutation_importance", "max_samples"]}),
    ]
    return catalog(rng, items, setup, f"{t.desc}. It was split with `train_test_split(df, y, stratify=y, "
                                      f"random_state={s})` into `X_tr, X_te, y_tr, y_te`.", tol=1e-9)


@template(M, "medium")
def partial_dependence_tasks(rng):
    t = table(rng, kind=rng.choice(["clf", "reg"]), numeric_only=True, n=(200, 320))
    s = seeded(rng)
    a, b = rng.sample(t.num, 2)
    g = rng.choice([6, 8, 10])
    if t.kind == "clf":
        model = "HistGradientBoostingClassifier(max_iter=50, random_state=0)"
        imp = "from sklearn.ensemble import HistGradientBoostingClassifier\n"
    else:
        model = "HistGradientBoostingRegressor(max_iter=50, random_state=0)"
        imp = "from sklearn.ensemble import HistGradientBoostingRegressor\n"
    setup = PLT + t.setup + f"\ndf = df.astype(float)\n{imp}model = {model}.fit(df, y)"
    pdd = f"{INS}PartialDependenceDisplay\n"
    items = [
        ("PartialDependenceDisplay", f"Draw the partial dependence on `{a}` (`features=[{a!r}]`, "
                                     f"`grid_resolution={g}`) and return the average curve.",
         pdd + f"d = PartialDependenceDisplay.from_estimator(model, df, features=[{a!r}], grid_resolution={g})\n"
               "result = d.pd_results[0]['average'][0]"),
        ("PartialDependenceDisplay", f"Draw the partial dependence on `{a}` and on `{b}` in one call (`features=[{a!r}, "
                                     f"{b!r}]`, `grid_resolution={g}`). Which one moves the prediction more? Return a "
                                     f"dict `{{feature: max − min of its average curve}}`.",
         pdd + f"d = PartialDependenceDisplay.from_estimator(model, df, features=[{a!r}, {b!r}], grid_resolution={g})\n"
               f"result = {{f: float(np.ptp(r['average'])) for f, r in zip([{a!r}, {b!r}], d.pd_results)}}"),
        ("kind", f"Draw only the ICE curves for `{a}` (`kind='individual'`, `grid_resolution={g}`) and return the shape "
                 f"of `pd_results[0]['individual']`: one curve per row of `df`.",
         pdd + f"d = PartialDependenceDisplay.from_estimator(model, df, features=[{a!r}], kind='individual', grid_resolution={g})\n"
               "result = d.pd_results[0]['individual'].shape", {"require": ["PartialDependenceDisplay", "kind"]}),
        ("subsample", f"Draw PDP + ICE for `{a}` with `kind='both'`, `subsample=25`, `random_state=0`, "
                      f"`grid_resolution={g}` and return the number of lines on its Axes (`disp.axes_[0, 0]`).",
         pdd + f"d = PartialDependenceDisplay.from_estimator(model, df, features=[{a!r}], kind='both', subsample=25, random_state=0, grid_resolution={g})\n"
               "result = len(d.axes_[0, 0].get_lines())", {"require": ["PartialDependenceDisplay", "subsample"]}),
        ("PartialDependenceDisplay", f"Draw a two-way partial dependence (`features=[({a!r}, {b!r})]`, "
                                     f"`grid_resolution={g}`) and return the shape of its average surface.",
         pdd + f"d = PartialDependenceDisplay.from_estimator(model, df, features=[({a!r}, {b!r})], grid_resolution={g})\n"
               "result = d.pd_results[0]['average'].shape"),
        ("partial_dependence", f"The numbers without the plot: return `partial_dependence(model, df, [{a!r}], "
                               f"grid_resolution={g})['average'][0]` (`sklearn.inspection`); it equals the display's "
                               f"curve.",
         f"{INS}partial_dependence\nresult = partial_dependence(model, df, [{a!r}], grid_resolution={g})['average'][0]"),
        ("centered", f"Centred ICE curves start at 0, which makes differences in shape visible. Draw `kind='individual', "
                     f"centered=True, random_state=0` for `{a}` (`grid_resolution={g}`; the seed fixes the order the curves "
                     f"are drawn in) and return the first column of "
                     f"return the y values of the first plotted curve (`disp.axes_[0, 0].get_lines()[0].get_ydata()`).",
         pdd + f"d = PartialDependenceDisplay.from_estimator(model, df, features=[{a!r}], kind='individual', centered=True, random_state=0, grid_resolution={g})\n"
               "result = np.asarray(d.axes_[0, 0].get_lines()[0].get_ydata())", {"require": ["PartialDependenceDisplay", "centered"]}),
    ]
    return catalog(rng, items, setup, f"{t.desc}; `model = {model}` was fitted on `df, y`.", tol=1e-6)


@template(M, "medium")
def decision_boundary_tasks(rng):
    setup, desc = _moons(rng)
    g = rng.choice([30, 40, 50])
    dbd = f"{INS}DecisionBoundaryDisplay\n"
    items = [
        ("response_method", f"Fit `{LR_PIPE}` on `X2, y2` and draw `DecisionBoundaryDisplay.from_estimator(clf, X2, "
                            f"response_method='predict_proba', grid_resolution={g})`. For two classes it shades P(class "
                            f"1); return `(response.min(), response.max())`.",
         LR_IMP + dbd + f"clf = {LR_PIPE}.fit(X2, y2)\nd = DecisionBoundaryDisplay.from_estimator(clf, X2, response_method='predict_proba', grid_resolution={g})\n"
                        "result = (float(d.response.min()), float(d.response.max()))",
         {"require": ["DecisionBoundaryDisplay", "response_method"]}),
        ("SVC", f"Compare two RBF SVMs: for `gamma` in 0.5 and 5, fit `SVC(gamma=gamma)` on `X2, y2`, draw its "
                f"`DecisionBoundaryDisplay` (`response_method='predict'`, `grid_resolution={g}`) and record the number "
                f"of horizontally adjacent cells with different labels. Return the two counts as a tuple.",
         dbd + "from sklearn.svm import SVC\nout = []\nfor gamma in (0.5, 5):\n"
               f"    r = DecisionBoundaryDisplay.from_estimator(SVC(gamma=gamma).fit(X2, y2), X2, response_method='predict', grid_resolution={g}).response\n"
               "    out.append(int((r[:, 1:] != r[:, :-1]).sum()))\nresult = tuple(out)"),
        ("eps", f"The grid extends `eps` beyond the data on every side (default 1.0). Fit `{LR_PIPE}` on `X2, y2`, draw "
                f"`DecisionBoundaryDisplay.from_estimator(clf, X2, response_method='predict', grid_resolution={g}, "
                f"eps=0.5)` and return `(xx0.min(), xx0.max())` of the grid.",
         LR_IMP + dbd + f"clf = {LR_PIPE}.fit(X2, y2)\nd = DecisionBoundaryDisplay.from_estimator(clf, X2, response_method='predict', grid_resolution={g}, eps=0.5)\n"
                        "result = (float(d.xx0.min()), float(d.xx0.max()))", {"require": ["DecisionBoundaryDisplay", "eps"]}),
        ("plot_method", f"Fit `DecisionTreeClassifier(max_depth=4, random_state=0)` on `X2, y2` and draw its regions as "
                        f"lines only: `plot_method='contour'`, `response_method='predict'`, `grid_resolution={g}`. "
                        f"Return the counts of each predicted label on the grid (`np.bincount` of the response).",
         dbd + f"from sklearn.tree import DecisionTreeClassifier\nclf = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X2, y2)\n"
               f"d = DecisionBoundaryDisplay.from_estimator(clf, X2, plot_method='contour', response_method='predict', grid_resolution={g})\n"
               "result = np.bincount(d.response.ravel().astype(int))", {"require": ["DecisionBoundaryDisplay", "plot_method"]}),
        ("xlabel", f"Feature names become axis labels. Put `X2` in a DataFrame with columns `['width', 'height']`, fit "
                   f"`KNeighborsClassifier(n_neighbors=7)` on it, draw `DecisionBoundaryDisplay.from_estimator(knn, df2, "
                   f"response_method='predict', grid_resolution={g})` and return `(xlabel, ylabel)` of its Axes.",
         dbd + "from sklearn.neighbors import KNeighborsClassifier\ndf2 = pd.DataFrame(X2, columns=['width', 'height'])\n"
               f"knn = KNeighborsClassifier(n_neighbors=7).fit(df2, y2)\n"
               f"d = DecisionBoundaryDisplay.from_estimator(knn, df2, response_method='predict', grid_resolution={g})\n"
               "result = (d.ax_.get_xlabel(), d.ax_.get_ylabel())", {"require": ["DecisionBoundaryDisplay", "ax_"]}),
        ("DecisionBoundaryDisplay", f"How well does the grid picture match the data? Fit `KNeighborsClassifier("
                                    f"n_neighbors=15)` on `X2, y2`, draw its `DecisionBoundaryDisplay` "
                                    f"(`response_method='predict'`, `grid_resolution={g}`), and return `(share of the "
                                    f"grid labelled 0, share of the points labelled 0)`.",
         dbd + "from sklearn.neighbors import KNeighborsClassifier\nknn = KNeighborsClassifier(n_neighbors=15).fit(X2, y2)\n"
               f"d = DecisionBoundaryDisplay.from_estimator(knn, X2, response_method='predict', grid_resolution={g})\n"
               "result = (float((d.response == 0).mean()), float((y2 == 0).mean()))"),
    ]
    return catalog(rng, items, setup, f"{desc}; `matplotlib.pyplot` is imported as `plt`.", tol=1e-9)


@template(M, "medium")
def learning_and_validation_tasks(rng):
    setup, desc, _ = _binary(rng)
    cv = rng.choice([3, 4])
    lc = f"{MS}LearningCurveDisplay\n"
    vc = f"{MS}ValidationCurveDisplay\n"
    tree = "from sklearn.tree import DecisionTreeClassifier\n"
    items = [
        ("LearningCurveDisplay", f"Is the tree data-hungry? Draw `LearningCurveDisplay.from_estimator(DecisionTreeClassifier("
                                 f"random_state=0), X, y, cv={cv}, train_sizes=np.linspace(0.1, 1.0, 5), score_type="
                                 f"'both')` and return `(mean train score at every size, mean validation score at every "
                                 f"size)`.",
         lc + tree + f"d = LearningCurveDisplay.from_estimator(DecisionTreeClassifier(random_state=0), X, y, cv={cv}, "
                     "train_sizes=np.linspace(0.1, 1.0, 5), score_type='both')\n"
                     "result = (d.train_scores.mean(axis=1), d.test_scores.mean(axis=1))"),
        ("score_name", f"Draw `LearningCurveDisplay.from_estimator({LR_PIPE}, X, y, cv={cv}, scoring='roc_auc')` and "
                       f"return `(disp.score_name, the y label of its Axes)`.",
         lc + LR_IMP + f"d = LearningCurveDisplay.from_estimator({LR_PIPE}, X, y, cv={cv}, scoring='roc_auc')\n"
                       "result = (d.score_name, d.ax_.get_ylabel())", {"require": ["LearningCurveDisplay", "score_name"]}),
        ("ValidationCurveDisplay", f"Tune by eye: draw `ValidationCurveDisplay.from_estimator(DecisionTreeClassifier("
                                   f"random_state=0), X, y, param_name='min_samples_leaf', param_range=[1, 5, 20, 50], "
                                   f"cv={cv})` and return the `min_samples_leaf` with the best mean validation score.",
         vc + tree + f"d = ValidationCurveDisplay.from_estimator(DecisionTreeClassifier(random_state=0), X, y, "
                     f"param_name='min_samples_leaf', param_range=[1, 5, 20, 50], cv={cv})\n"
                     "result = int(d.param_range[int(d.test_scores.mean(axis=1).argmax())])"),
        ("ValidationCurveDisplay", f"Draw `ValidationCurveDisplay.from_estimator(DecisionTreeClassifier(random_state=0), "
                                   f"X, y, param_name='max_depth', param_range=[1, 2, 4, 8, 16], cv={cv})` and return the "
                                   f"smallest depth whose mean **training** score reaches 1.0 (`None` if none does).",
         vc + tree + f"d = ValidationCurveDisplay.from_estimator(DecisionTreeClassifier(random_state=0), X, y, "
                     f"param_name='max_depth', param_range=[1, 2, 4, 8, 16], cv={cv})\n"
                     "full = [int(p) for p, s in zip(d.param_range, d.train_scores.mean(axis=1)) if s >= 1.0]\n"
                     "result = full[0] if full else None"),
        ("std", f"How noisy is the estimate? Draw `LearningCurveDisplay.from_estimator({LR_PIPE}, X, y, cv={cv}, "
                f"train_sizes=np.linspace(0.2, 1.0, 4))` and return the standard deviation of the validation scores "
                f"across folds at each size.",
         lc + LR_IMP + f"d = LearningCurveDisplay.from_estimator({LR_PIPE}, X, y, cv={cv}, train_sizes=np.linspace(0.2, 1.0, 4))\n"
                       "result = d.test_scores.std(axis=1)", {"require": ["LearningCurveDisplay"]}),
        ("negate_score", f"For an error metric, `negate_score=True` plots it as a positive error. Draw "
                         f"`LearningCurveDisplay.from_estimator({LR_PIPE}, X, y, cv={cv}, train_sizes=np.linspace(0.3, "
                         f"1.0, 3), scoring='neg_log_loss', negate_score=True)` and return the y data of its validation "
                         f"line (the last line on the Axes).",
         lc + LR_IMP + f"d = LearningCurveDisplay.from_estimator({LR_PIPE}, X, y, cv={cv}, train_sizes=np.linspace(0.3, 1.0, 3), "
                       "scoring='neg_log_loss', negate_score=True)\nresult = np.asarray(d.ax_.get_lines()[-1].get_ydata())",
         {"require": ["LearningCurveDisplay", "negate_score"]}),
    ]
    return catalog(rng, items, setup, f"{desc} (`X`, `y` are the full data); `matplotlib.pyplot` is imported as `plt`.",
                   tol=1e-9)


@template(M, "medium")
def regression_inspection(rng):
    s, r = seeded(rng), rng.randint(0, 9)
    setup = (PLT + "from sklearn.datasets import load_diabetes\nfrom sklearn.model_selection import train_test_split\n"
             "data = load_diabetes(as_frame=True)\nX, y = data.data, data.target\n"
             f"X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state={s})")
    a = rng.choice([0.1, 1.0])
    items = [
        ("PredictionErrorDisplay", f"Fit `Ridge(alpha={a})` on the training part and draw "
                                   f"`PredictionErrorDisplay.from_estimator(reg, X_te, y_te)`. Return the largest "
                                   f"absolute residual shown.",
         f"from sklearn.linear_model import Ridge\n{MET}PredictionErrorDisplay\nreg = Ridge(alpha={a}).fit(X_tr, y_tr)\n"
         "d = PredictionErrorDisplay.from_estimator(reg, X_te, y_te)\nresult = float(np.abs(np.asarray(d.scatter_.get_offsets())[:, 1]).max())"),
        ("permutation_importance", f"Fit `Ridge(alpha={a})` on the training part and return the top 3 features by test "
                                   f"permutation importance (default R² scoring, `n_repeats=5, random_state={r}`) as a "
                                   f"Series.",
         f"from sklearn.linear_model import Ridge\n{INS}permutation_importance\nreg = Ridge(alpha={a}).fit(X_tr, y_tr)\n"
         f"imp = permutation_importance(reg, X_te, y_te, n_repeats=5, random_state={r}).importances_mean\n"
         "result = pd.Series(imp, index=X.columns).nlargest(3)"),
        ("coef_", f"For a linear model on standardized features, |coefficient| is a quick importance. Fit `Ridge(alpha={a})` "
                  f"on the training part (the diabetes features are already scaled) and return the top 3 features by "
                  f"|coef_| next to the top 3 by test permutation importance (`n_repeats=5, random_state={r}`), as two "
                  f"lists.",
         f"from sklearn.linear_model import Ridge\n{INS}permutation_importance\nreg = Ridge(alpha={a}).fit(X_tr, y_tr)\n"
         f"imp = permutation_importance(reg, X_te, y_te, n_repeats=5, random_state={r}).importances_mean\n"
         "result = (list(pd.Series(np.abs(reg.coef_), index=X.columns).nlargest(3).index), list(pd.Series(imp, index=X.columns).nlargest(3).index))"),
        ("PartialDependenceDisplay", "A linear model's partial dependence is a straight line. Fit `Ridge()` on the "
                                     "training part, draw the PDP of `'bmi'` (`grid_resolution=5`) and return the "
                                     "differences between consecutive points of the curve divided by the grid steps "
                                     "(all equal to the `bmi` coefficient).",
         f"from sklearn.linear_model import Ridge\n{INS}PartialDependenceDisplay\nreg = Ridge().fit(X_tr, y_tr)\n"
         "d = PartialDependenceDisplay.from_estimator(reg, X_te, features=['bmi'], grid_resolution=5)\n"
         "res = d.pd_results[0]\nresult = np.diff(res['average'][0]) / np.diff(res['grid_values'][0])"),
        ("scoring", f"Fit `RandomForestRegressor(n_estimators=30, random_state=0)` on the training part and return the "
                    f"test permutation importances measured in MAE (`scoring='neg_mean_absolute_error'`, `n_repeats=5, "
                    f"random_state={r}`) of `'bmi'` and `'s5'`, as a tuple.",
         f"from sklearn.ensemble import RandomForestRegressor\n{INS}permutation_importance\n"
         "reg = RandomForestRegressor(n_estimators=30, random_state=0).fit(X_tr, y_tr)\n"
         f"r = permutation_importance(reg, X_te, y_te, scoring='neg_mean_absolute_error', n_repeats=5, random_state={r})\n"
         "imp = pd.Series(r.importances_mean, index=X.columns)\nresult = (float(imp['bmi']), float(imp['s5']))",
         {"require": ["permutation_importance", "scoring"]}),
    ]
    return catalog(rng, items, setup, f"`X` (a DataFrame of 10 standardized measurements) and `y` = `data.target` hold "
                                      f"the diabetes data, split with `train_test_split(X, y, random_state={s})`.",
                   tol=1e-6)


@template(M, "medium")
def multiclass_confusion(rng):
    data = rng.choice(["wine", "iris", "digits"])
    loader = {"wine": "load_wine", "iris": "load_iris", "digits": "load_digits"}[data]
    s = seeded(rng)
    setup = (PLT + f"from sklearn.datasets import {loader}\nfrom sklearn.model_selection import train_test_split\n"
             f"d = {loader}()\nX, y, names = d.data, d.target, list(map(str, d.target_names))\n"
             f"X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, random_state={s})\n"
             f"{LR_IMP}clf = {LR_PIPE}.fit(X_tr, y_tr)")
    cm = f"{MET}ConfusionMatrixDisplay\n"
    task = rng.choice(["recall", "pair", "precision", "labels", "format"])
    head = (f"`X`, `y` hold the {data} data and `names` the class names; `clf = {LR_PIPE}` was fitted on the training "
            f"part of a stratified split (`X_tr, X_te, y_tr, y_te`).")
    if task == "recall":
        return P(f"{head} Draw `ConfusionMatrixDisplay.from_estimator(clf, X_te, y_te, normalize='true')`; its diagonal is "
                 f"each class's recall. Return `{{name: recall}}`.", setup,
                 cm + "m = ConfusionMatrixDisplay.from_estimator(clf, X_te, y_te, normalize='true').confusion_matrix\n"
                      "result = dict(zip(names, np.diag(m)))", require=["ConfusionMatrixDisplay", "normalize"], tol=1e-9)
    if task == "pair":
        return P(f"{head} Draw the test confusion matrix (counts) and return the most frequent mistake as `(true name, "
                 f"predicted name, count)` (largest off-diagonal cell; the first in row order on ties).", setup,
                 cm + "m = ConfusionMatrixDisplay.from_estimator(clf, X_te, y_te).confusion_matrix.copy()\n"
                      "np.fill_diagonal(m, -1)\ni, j = np.unravel_index(m.argmax(), m.shape)\n"
                      "result = (names[i], names[j], int(m[i, j]))", require=["ConfusionMatrixDisplay"])
    if task == "precision":
        return P(f"{head} Draw the confusion matrix with `normalize='pred'` (columns sum to 1, the diagonal is each "
                 f"class's precision) and return the diagonal; it equals `precision_score(y_te, pred, average=None)`.",
                 setup, cm + "result = np.diag(ConfusionMatrixDisplay.from_estimator(clf, X_te, y_te, normalize='pred').confusion_matrix)",
                 require=["ConfusionMatrixDisplay", "normalize"], tol=1e-9)
    if task == "format":
        return P(f"{head} Show percentages in the cells: draw `ConfusionMatrixDisplay.from_estimator(clf, X_te, y_te, "
                 f"normalize='true', values_format='.0%')` and return the texts on the diagonal cells, in class order.",
                 setup, cm + "d = ConfusionMatrixDisplay.from_estimator(clf, X_te, y_te, normalize='true', values_format='.0%')\n"
                             "result = [d.text_[i, i].get_text() for i in range(len(names))]",
                 require=["ConfusionMatrixDisplay", "values_format"])
    return P(f"{head} Draw `ConfusionMatrixDisplay.from_predictions(y_te, clf.predict(X_te), display_labels=names, "
             f"xticks_rotation='vertical')` and return `(the y tick labels, the matrix's trace)`.", setup,
             cm + "d = ConfusionMatrixDisplay.from_predictions(y_te, clf.predict(X_te), display_labels=names, xticks_rotation='vertical')\n"
                  "result = ([t.get_text() for t in d.ax_.get_yticklabels()], int(np.trace(d.confusion_matrix)))",
             require=["from_predictions", "display_labels"])


# ================================================================= hard

@template(M, "hard")
def importance_by_hand(rng):
    t = table(rng, kind="clf", numeric_only=True, n=(200, 320))
    s, reps = seeded(rng), rng.choice([3, 4])
    setup = (PLT + t.setup + "\nfrom sklearn.model_selection import train_test_split\n"
             f"X_tr, X_te, y_tr, y_te = train_test_split(df, y, stratify=y, random_state={s})\n"
             f"{LR_IMP}clf = {LR_PIPE}.fit(X_tr, y_tr)")
    head = f"{t.desc}, split into `X_tr, X_te, y_tr, y_te`; `clf = {LR_PIPE}` was fitted on the training part."
    task = rng.choice(["perm", "drop", "group"])
    if task == "perm":
        return P(f"{head} Permutation importance by hand: `rs = np.random.default_rng({s})`; for each column (in order), "
                 f"{reps} times: copy `X_te`, replace the column with `rs.permutation` of its values, and record the drop "
                 f"`clf.score(X_te, y_te) − clf.score(shuffled, y_te)`. Return the mean drop per column as a Series.",
                 setup, f"rs = np.random.default_rng({s})\nbase = clf.score(X_te, y_te)\nout = {{}}\nfor c in X_te.columns:\n"
                        f"    drops = []\n    for _ in range({reps}):\n        Xs = X_te.copy()\n"
                        "        Xs[c] = rs.permutation(Xs[c].to_numpy())\n        drops.append(base - clf.score(Xs, y_te))\n"
                        "    out[c] = np.mean(drops)\nresult = pd.Series(out)", require=["score"], tol=1e-9)
    if task == "drop":
        return P(f"{head} Drop-column importance retrains instead of shuffling: for each column, refit `{LR_PIPE}` on the "
                 f"training part **without** it and record the test accuracy lost compared with `clf`. Return the "
                 f"Series of losses, largest first.", setup,
                 f"from sklearn.base import clone\nbase = clf.score(X_te, y_te)\n"
                 "loss = {c: base - clone(clf).fit(X_tr.drop(columns=c), y_tr).score(X_te.drop(columns=c), y_te) for c in X_te.columns}\n"
                 "result = pd.Series(loss).sort_values(ascending=False)", require=["clone|make_pipeline"], tol=1e-9)
    a, b = rng.sample(t.num, 2)
    return P(f"{head} Correlated features share credit, so shuffle them **together**: with `rs = np.random.default_rng("
             f"{s})`, {reps} times draw one permutation `idx = rs.permutation(len(X_te))` and apply it to both `{a}` and "
             f"`{b}` at once; return the mean accuracy drop.", setup,
             f"rs = np.random.default_rng({s})\nbase = clf.score(X_te, y_te)\ndrops = []\nfor _ in range({reps}):\n"
             "    idx = rs.permutation(len(X_te))\n    Xs = X_te.copy()\n"
             f"    Xs[[{a!r}, {b!r}]] = X_te[[{a!r}, {b!r}]].to_numpy()[idx]\n    drops.append(base - clf.score(Xs, y_te))\n"
             "result = float(np.mean(drops))", require=["score"], tol=1e-9)


@template(M, "hard")
def partial_dependence_by_hand(rng):
    t = table(rng, kind="reg", numeric_only=True, n=(150, 260))
    a = rng.choice(t.num)
    g = rng.choice([4, 5, 6])
    setup = (PLT + t.setup + "\ndf = df.astype(float)\nfrom sklearn.ensemble import RandomForestRegressor\n"
             "model = RandomForestRegressor(n_estimators=20, max_depth=5, random_state=0).fit(df, y)")
    head = f"{t.desc}; `model = RandomForestRegressor(n_estimators=20, max_depth=5, random_state=0)` was fitted on it."
    if rng.random() < 0.5:
        return P(f"{head} Partial dependence by hand: for each of {g} values `v` in `np.linspace(df[{a!r}].min(), "
                 f"df[{a!r}].max(), {g})`, set `{a}` to `v` in a copy of `df` and average the model's predictions. "
                 f"Return the {g} averages (they equal `partial_dependence(model, df, [{a!r}], method='brute', "
                 f"custom_values={{{a!r}: grid}})['average'][0]`).", setup,
                 f"grid = np.linspace(df[{a!r}].min(), df[{a!r}].max(), {g})\nout = []\nfor v in grid:\n"
                 f"    d2 = df.copy()\n    d2[{a!r}] = v\n    out.append(model.predict(d2).mean())\nresult = np.array(out)",
                 require=["predict"], tol=1e-9)
    i = rng.randint(0, 99)
    return P(f"{head} An ICE curve follows one row: set `{a}` of row {i} (`df.iloc[[{i}]]`) to each of {g} values in "
             f"`np.linspace(df[{a!r}].min(), df[{a!r}].max(), {g})` and return the predictions. Then return `(curve, "
             f"curve − curve[0])` (the centred ICE curve too).", setup,
             f"grid = np.linspace(df[{a!r}].min(), df[{a!r}].max(), {g})\nrows = pd.concat([df.iloc[[{i}]]] * {g}, ignore_index=True)\n"
             f"rows[{a!r}] = grid\ncurve = model.predict(rows)\nresult = (curve, curve - curve[0])", require=["predict"],
             tol=1e-9)


@template(M, "hard")
def curve_areas_by_hand(rng):
    setup, desc, _ = _binary(rng)
    setup += f"\n{LR_IMP}clf = {LR_PIPE}.fit(X_tr, y_tr)"
    if rng.random() < 0.5:
        return P(f"{_split_desc(desc)}; `clf = {LR_PIPE}` was fitted on the training part. Draw "
                 f"`RocCurveDisplay.from_estimator(clf, X_te, y_te)` and recompute its AUC **by hand** from `disp.fpr` "
                 f"and `disp.tpr` with the trapezoidal rule (`np.trapezoid(tpr, fpr)`). Return `(by_hand, "
                 f"disp.roc_auc)`.", setup,
                 f"{MET}RocCurveDisplay\nd = RocCurveDisplay.from_estimator(clf, X_te, y_te)\nresult = (float(np.trapezoid(d.tpr, d.fpr)), d.roc_auc)",
                 require=["RocCurveDisplay"], tol=1e-9)
    return P(f"{_split_desc(desc)}; `clf = {LR_PIPE}` was fitted on the training part. Draw "
             f"`PrecisionRecallDisplay.from_estimator(clf, X_te, y_te)`; its `recall` decreases along the arrays. Average "
             f"precision is the step sum Σ (R_(k) − R_(k+1)) · P_(k) over consecutive points. Return `(by_hand, "
             f"disp.average_precision)`.", setup,
             f"{MET}PrecisionRecallDisplay\nd = PrecisionRecallDisplay.from_estimator(clf, X_te, y_te)\n"
             "result = (float(-np.sum(np.diff(d.recall) * d.precision[:-1])), d.average_precision)",
             require=["PrecisionRecallDisplay"], tol=1e-9)


@template(M, "hard")
def inspection_pitfalls(rng):
    setup, desc, names = _binary(rng)
    s, r = seeded(rng), rng.randint(0, 9)
    items = [
        ("permutation_importance", f"A colleague ranks features with `permutation_importance(clf, X_tr, y_tr, ...)` "
                                   f"for `clf = DecisionTreeClassifier(random_state=0)` (fully grown) fitted on the "
                                   f"training part, so every feature it memorized looks useful. Return `(number of "
                                   f"features with positive importance on the training set, the same on the test set)` "
                                   f"with `n_repeats=5, random_state={r}`.",
         f"from sklearn.tree import DecisionTreeClassifier\n{INS}permutation_importance\n"
         "clf = DecisionTreeClassifier(random_state=0).fit(X_tr, y_tr)\n"
         f"tr = permutation_importance(clf, X_tr, y_tr, n_repeats=5, random_state={r}).importances_mean\n"
         f"te = permutation_importance(clf, X_te, y_te, n_repeats=5, random_state={r}).importances_mean\n"
         "result = (int((tr > 0).sum()), int((te > 0).sum()))"),
        ("feature_importances_", f"Add an ID-like column `row_id` (`np.random.default_rng({s}).permutation(len(X))`, "
                                 f"aligned through the index) to both splits. A forest gives it a sizeable impurity "
                                 f"importance although it is meaningless. Fit `{RF}` and return `(impurity importance "
                                 f"of row_id, its test permutation importance)` (`n_repeats=5, random_state={r}`).",
         f"{RF_IMP}{INS}permutation_importance\nrid = pd.Series(np.random.default_rng({s}).permutation(len(X)), index=X.index)\n"
         f"A, B = X_tr.assign(row_id=rid), X_te.assign(row_id=rid)\nclf = {RF}.fit(A, y_tr)\n"
         f"result = (float(clf.feature_importances_[-1]), float(permutation_importance(clf, B, y_te, n_repeats=5, random_state={r}).importances_mean[-1]))"),
        ("permutation_importance", f"Duplicated features hide each other: with a copy of the first column added "
                                   f"(`dup`), shuffling one copy barely hurts because the other remains. Fit `{RF}` on "
                                   f"the training part with and without `dup` and return the test permutation importance "
                                   f"of the first column in both cases (`n_repeats=5, random_state={r}`), as `(alone, "
                                   f"with_dup)`.",
         f"{RF_IMP}{INS}permutation_importance\nc0 = X.columns[0]\n"
         f"alone = permutation_importance({RF}.fit(X_tr, y_tr), X_te, y_te, n_repeats=5, random_state={r}).importances_mean[0]\n"
         "A, B = X_tr.assign(dup=X_tr[c0]), X_te.assign(dup=X_te[c0])\n"
         f"dup = permutation_importance({RF}.fit(A, y_tr), B, y_te, n_repeats=5, random_state={r}).importances_mean[0]\n"
         "result = (float(alone), float(dup))"),
        ("labels", f"`ConfusionMatrixDisplay.from_predictions(y_te, pred, display_labels={names[::-1]})` puts the names "
                   f"on the wrong rows (the classes are 0, 1 in that order). Fix the labels and also force the class "
                   f"order explicitly with `labels=[0, 1]`; return `(tick labels, matrix)` for `pred` from `{LR_PIPE}` "
                   f"fitted on the training part.",
         f"{MET}ConfusionMatrixDisplay\n{LR_IMP}pred = {LR_PIPE}.fit(X_tr, y_tr).predict(X_te)\n"
         f"d = ConfusionMatrixDisplay.from_predictions(y_te, pred, labels=[0, 1], display_labels={names})\n"
         "result = ([t.get_text() for t in d.ax_.get_xticklabels()], d.confusion_matrix)", {"require": ["labels", "display_labels"]}),
        ("from_predictions", f"`RocCurveDisplay.from_predictions(y_te, clf.predict(X_te))` draws a curve with a single "
                             f"corner: hard labels throw away the ranking. Return `(auc_from_labels, auc_from_scores)` "
                             f"using `clf.predict_proba(X_te)[:, 1]` for the second, `clf = {LR_PIPE}` fitted on the "
                             f"training part.",
         f"{MET}RocCurveDisplay\n{LR_IMP}clf = {LR_PIPE}.fit(X_tr, y_tr)\n"
         "result = (RocCurveDisplay.from_predictions(y_te, clf.predict(X_te)).roc_auc, "
         "RocCurveDisplay.from_predictions(y_te, clf.predict_proba(X_te)[:, 1]).roc_auc)"),
        ("ax", f"This loop draws each model on a **new** figure: `for m in models: RocCurveDisplay.from_estimator(m, "
               f"X_te, y_te)`. Overlay them on one Axes instead (`fig, ax = plt.subplots()`, `ax=ax`) for `models = "
               f"[{LR_PIPE}, {RF}]` fitted on the training part, and return `(len(plt.get_fignums()), "
               f"len(ax.get_lines()))`.",
         f"{MET}RocCurveDisplay\n{LR_IMP}{RF_IMP}models = [{LR_PIPE}, {RF}]\nfig, ax = plt.subplots()\n"
         "for m in models:\n    RocCurveDisplay.from_estimator(m.fit(X_tr, y_tr), X_te, y_te, ax=ax)\n"
         "result = (len(plt.get_fignums()), len(ax.get_lines()))", {"require": ["ax", "subplots"]}),
        ("DecisionBoundaryDisplay", f"`DecisionBoundaryDisplay.from_estimator(clf, X_te, ...)` raises: it can only draw "
                                    f"2 features. Fit `{LR_PIPE}` on the **first two columns** of the training part and "
                                    f"draw it on `X_te.iloc[:, :2]` (`response_method='predict'`, `grid_resolution=30`). "
                                    f"Return the share of the grid predicted as 1.",
         f"{INS}DecisionBoundaryDisplay\n{LR_IMP}clf = {LR_PIPE}.fit(X_tr.iloc[:, :2], y_tr)\n"
         "d = DecisionBoundaryDisplay.from_estimator(clf, X_te.iloc[:, :2], response_method='predict', grid_resolution=30)\n"
         "result = float(d.response.mean())"),
    ]
    return catalog(rng, items, setup, f"{_split_desc(desc)}; `matplotlib.pyplot` is imported as `plt`.", tol=1e-9)


@template(M, "hard")
def decision_grid_by_hand(rng):
    setup, desc = _moons(rng)
    g, eps = rng.choice([20, 25, 30]), rng.choice([0.5, 1.0])
    k = rng.choice([5, 9, 15])
    grid = (f"x0 = np.linspace(X2[:, 0].min() - {eps}, X2[:, 0].max() + {eps}, {g})\n"
            f"x1 = np.linspace(X2[:, 1].min() - {eps}, X2[:, 1].max() + {eps}, {g})\nxx0, xx1 = np.meshgrid(x0, x1)\n"
            f"from sklearn.neighbors import KNeighborsClassifier\nknn = KNeighborsClassifier(n_neighbors={k}).fit(X2, y2)\n"
            "resp = knn.predict(np.c_[xx0.ravel(), xx1.ravel()]).reshape(xx0.shape)\n")
    head = (f"{desc}. `DecisionBoundaryDisplay` evaluates the model on a grid: {g} evenly spaced values per feature from "
            f"its min − eps to its max + eps (eps = {eps}), combined with `np.meshgrid`, predicted, and reshaped to the "
            f"grid. Do it **by hand** for `KNeighborsClassifier(n_neighbors={k})` fitted on `X2, y2`")
    if rng.random() < 0.5:
        return P(f"{head} and return the label grid (it equals `DecisionBoundaryDisplay.from_estimator(knn, X2, "
                 f"response_method='predict', grid_resolution={g}, eps={eps}).response`).", setup, grid + "result = resp",
                 require=["meshgrid", "predict"])
    return P(f"{head}, and return for every grid row (fixed second feature) the number of label changes along it, "
             f"as an int array.", setup, grid + "result = (resp[:, 1:] != resp[:, :-1]).sum(axis=1)",
             require=["meshgrid", "predict"])


@template(M, "hard")
def learning_curve_by_hand(rng):
    setup, desc, _ = _binary(rng)
    cv = rng.choice([3, 4])
    sizes = rng.choice(["[0.25, 0.5, 1.0]", "[0.2, 0.6, 1.0]", "[0.3, 0.65, 1.0]"])
    if rng.random() < 0.5:
        return P(f"{desc} (`X`, `y` are the full data). Reproduce `LearningCurveDisplay.from_estimator({LR_PIPE}, X, y, "
                 f"cv={cv}, train_sizes={sizes}).test_scores` by hand: for each split of `StratifiedKFold(n_splits={cv})`"
                 f" and each fraction `f`, fit on the **first** `int(f · len(train))` training indices and score on "
                 f"the validation indices. Return a (sizes × folds) array.", setup,
                 f"{MS}StratifiedKFold\n{LR_IMP}Xa, ya = np.asarray(X), np.asarray(y)\nsplits = list(StratifiedKFold(n_splits={cv}).split(Xa, ya))\n"
                 f"n_max = len(splits[0][0])\nout = np.zeros((len({sizes}), {cv}))\n"
                 f"for i, f in enumerate({sizes}):\n    n = int(f * n_max)\n    for j, (tr, te) in enumerate(splits):\n"
                 f"        out[i, j] = {LR_PIPE}.fit(Xa[tr[:n]], ya[tr[:n]]).score(Xa[te], ya[te])\nresult = out",
                 require=["StratifiedKFold"], tol=1e-9)
    vals = rng.choice(["[0.001, 0.01, 0.1, 1.0]", "[0.01, 0.1, 1.0, 10.0]"])
    return P(f"{desc} (`X`, `y` are the full data). A validation curve is cross-validation in a loop: for each `C` in "
             f"`{vals}` compute `cross_val_score({LR_PIPE}.set_params(logisticregression__C=C), X, y, cv={cv}).mean()`. "
             f"Return `(mean scores, best C)`; they match `ValidationCurveDisplay` with `param_name="
             f"'logisticregression__C'`.", setup,
             f"{MS}cross_val_score\n{LR_IMP}m = np.array([cross_val_score({LR_PIPE}.set_params(logisticregression__C=C), X, y, cv={cv}).mean() for C in {vals}])\n"
             f"result = (m, {vals}[int(m.argmax())])", require=["cross_val_score"], tol=1e-9)


@template(M, "hard")
def importance_with_preprocessing(rng):
    t = table(rng, kind="clf", n=(250, 400))
    s, r = seeded(rng), rng.randint(0, 9)
    setup = (PLT + t.setup + f"\nnum, cat = {lit(t.num)}, {lit(t.cat)}\nfrom sklearn.model_selection import train_test_split\n"
             f"X_tr, X_te, y_tr, y_te = train_test_split(df, y, stratify=y, random_state={s})\n"
             "from sklearn.compose import ColumnTransformer\nfrom sklearn.pipeline import Pipeline\n"
             "from sklearn.preprocessing import OneHotEncoder, StandardScaler\nfrom sklearn.ensemble import RandomForestClassifier\n"
             "prep = ColumnTransformer([('num', StandardScaler(), num), ('cat', OneHotEncoder(handle_unknown='ignore'), cat)])\n"
             "model = Pipeline([('prep', prep), ('rf', RandomForestClassifier(n_estimators=30, random_state=0))]).fit(X_tr, y_tr)")
    items = [
        ("permutation_importance", f"Permutation importance on the whole pipeline works on the **original** columns "
                                   f"(categoricals included). Return the test importances (`n_repeats=5, "
                                   f"random_state={r}`) as a Series indexed by `df.columns`, largest first.",
         f"{INS}permutation_importance\nr = permutation_importance(model, X_te, y_te, n_repeats=5, random_state={r})\n"
         "result = pd.Series(r.importances_mean, index=X_te.columns).sort_values(ascending=False)"),
        ("get_feature_names_out", "The forest's `feature_importances_` refer to the **encoded** columns. Return them as a "
                                  "Series indexed by `model[:-1].get_feature_names_out()`, largest 4 first.",
         "imp = model.named_steps['rf'].feature_importances_\n"
         "result = pd.Series(imp, index=model[:-1].get_feature_names_out()).nlargest(4)"),
        ("feature_importances_", "Add up the impurity importances of each original column's encoded columns (names "
                                 "look like `num__age` and `cat__plan_basic`; the original name for a categorical is "
                                 "the part before the level). Return a Series per original column, sorted by name.",
         "names = model[:-1].get_feature_names_out()\nimp = model.named_steps['rf'].feature_importances_\n"
         "orig = [n.split('__', 1)[1] if n.startswith('num__') else next(c for c in cat if n.startswith('cat__' + c + '_')) for n in names]\n"
         "result = pd.Series(imp, index=orig).groupby(level=0).sum().sort_index()"),
        ("permutation_importance", f"Which matters more, the numeric or the categorical block? Shuffle each block's "
                                   f"columns separately with `permutation_importance` (`n_repeats=5, random_state={r}`) "
                                   f"and return `(sum over numeric columns, sum over categorical columns)`.",
         f"{INS}permutation_importance\nimp = pd.Series(permutation_importance(model, X_te, y_te, n_repeats=5, random_state={r}).importances_mean, index=X_te.columns)\n"
         "result = (float(imp[num].sum()), float(imp[cat].sum()))"),
        ("PartialDependenceDisplay", f"Partial dependence also takes the raw DataFrame through the pipeline. Draw the "
                                     f"PDP of the categorical column `{t.cat[0]}` (`features=[{t.cat[0]!r}]`, "
                                     f"`categorical_features=[{t.cat[0]!r}]`) on `X_te` and return `{{level: average}}`.",
         f"{INS}PartialDependenceDisplay\nd = PartialDependenceDisplay.from_estimator(model, X_te, features=[{t.cat[0]!r}], categorical_features=[{t.cat[0]!r}])\n"
         "res = d.pd_results[0]\nresult = {str(k): float(v) for k, v in zip(res['grid_values'][0], res['average'][0])}",
         {"require": ["PartialDependenceDisplay", "categorical_features"]}),
    ]
    return catalog(rng, items, setup, f"{t.desc}. `num`, `cat` list the numeric and categorical columns; the rows were "
                                      f"split with `train_test_split(df, y, stratify=y, random_state={s})`, and `model` "
                                      f"(a `ColumnTransformer` with `StandardScaler` / `OneHotEncoder`, step `'prep'`, "
                                      f"then `RandomForestClassifier(n_estimators=30, random_state=0)`, step `'rf'`) was "
                                      f"fitted on `X_tr, y_tr`.", tol=1e-9)


@template(M, "hard")
def operating_points_from_displays(rng):
    setup, desc, _ = _binary(rng)
    setup += f"\n{LR_IMP}clf = {LR_PIPE}.fit(X_tr, y_tr)"
    a = rng.choice([0.05, 0.1, 0.2])
    items = [
        ("RocCurveDisplay", f"From `RocCurveDisplay.from_estimator(clf, X_te, y_te)`, return the highest true positive "
                            f"rate reachable with a false positive rate of at most {a}.",
         f"{MET}RocCurveDisplay\nd = RocCurveDisplay.from_estimator(clf, X_te, y_te)\nresult = float(d.tpr[d.fpr <= {a}].max())"),
        ("PrecisionRecallDisplay", "From `PrecisionRecallDisplay.from_estimator(clf, X_te, y_te)`, return the best F1 "
                                   "along the curve, 2·P·R / (P + R) (skip points where P + R = 0).",
         f"{MET}PrecisionRecallDisplay\nd = PrecisionRecallDisplay.from_estimator(clf, X_te, y_te)\nok = d.precision + d.recall > 0\n"
         "result = float((2 * d.precision[ok] * d.recall[ok] / (d.precision[ok] + d.recall[ok])).max())"),
        ("DetCurveDisplay", "From `DetCurveDisplay.from_estimator(clf, X_te, y_te)`, return the equal error rate "
                            "estimate min over the points of max(fpr, fnr).",
         f"{MET}DetCurveDisplay\nd = DetCurveDisplay.from_estimator(clf, X_te, y_te)\nresult = float(np.maximum(d.fpr, d.fnr).min())"),
        ("CalibrationDisplay", "From `CalibrationDisplay.from_estimator(clf, X_te, y_te, n_bins=5)`, return the "
                               "largest |prob_true − prob_pred| and whether the model over-estimates class 1 in that bin "
                               "(`bool(prob_pred > prob_true)` there), as a tuple.",
         "from sklearn.calibration import CalibrationDisplay\nd = CalibrationDisplay.from_estimator(clf, X_te, y_te, n_bins=5)\n"
         "i = int(np.abs(d.prob_true - d.prob_pred).argmax())\n"
         "result = (float(abs(d.prob_true[i] - d.prob_pred[i])), bool(d.prob_pred[i] > d.prob_true[i]))"),
        ("ConfusionMatrixDisplay", "From `ConfusionMatrixDisplay.from_estimator(clf, X_te, y_te)` (counts), compute "
                                   "`(sensitivity, specificity, balanced accuracy)` for class 1.",
         f"{MET}ConfusionMatrixDisplay\nm = ConfusionMatrixDisplay.from_estimator(clf, X_te, y_te).confusion_matrix\n"
         "sens, spec = m[1, 1] / m[1].sum(), m[0, 0] / m[0].sum()\nresult = (float(sens), float(spec), float((sens + spec) / 2))"),
    ]
    return catalog(rng, items, setup, f"{_split_desc(desc)}; `clf = {LR_PIPE}` was fitted on the training part.",
                   tol=1e-9)


@template(M, "hard")
def ice_heterogeneity(rng):
    t = table(rng, kind="reg", numeric_only=True, n=(150, 240))
    a = rng.choice(t.num)
    g = rng.choice([5, 8])
    setup = (PLT + t.setup + "\ndf = df.astype(float)\nfrom sklearn.ensemble import HistGradientBoostingRegressor\n"
             "model = HistGradientBoostingRegressor(max_iter=60, random_state=0).fit(df, y)")
    disp = (f"from sklearn.inspection import PartialDependenceDisplay\nd = PartialDependenceDisplay.from_estimator(model, df, "
            f"features=[{a!r}], kind='individual', grid_resolution={g})\nice = d.pd_results[0]['individual'][0]\n")
    head = (f"{t.desc}; `model = HistGradientBoostingRegressor(max_iter=60, random_state=0)` was fitted on it. Draw the "
            f"ICE curves of `{a}` (`kind='individual'`, `grid_resolution={g}`); `pd_results[0]['individual'][0]` holds "
            f"one curve per row.")
    if rng.random() < 0.5:
        return P(f"{head} Which row reacts most to `{a}`? Return `(row index, max − min of its curve)` for the curve "
                 f"with the largest range.", setup,
                 disp + "rng_ = np.ptp(ice, axis=1)\ni = int(rng_.argmax())\nresult = (i, float(rng_[i]))",
                 require=["PartialDependenceDisplay"], tol=1e-6)
    return P(f"{head} Do all rows move the same way? Return `(rows whose curve ends above its start, rows whose curve "
             f"ends below)` as counts, and check the PDP is their mean: add `np.abs(ice.mean(axis=0) - "
             f"pd_average).max() < 1e-9` as a third element (compute `pd_average` with `kind='average'`).", setup,
             disp + f"avg = PartialDependenceDisplay.from_estimator(model, df, features=[{a!r}], grid_resolution={g}).pd_results[0]['average'][0]\n"
                    "result = (int((ice[:, -1] > ice[:, 0]).sum()), int((ice[:, -1] < ice[:, 0]).sum()), bool(np.abs(ice.mean(axis=0) - avg).max() < 1e-9))",
             require=["PartialDependenceDisplay"])


@template(M, "hard")
def linear_effects_vs_permutation(rng):
    t = table(rng, kind="reg", numeric_only=True, n=(200, 320))
    s, r = seeded(rng), rng.randint(0, 9)
    setup = (PLT + t.setup + "\ndf = df.astype(float)\nfrom sklearn.model_selection import train_test_split\n"
             f"X_tr, X_te, y_tr, y_te = train_test_split(df, y, random_state={s})\n"
             "from sklearn.linear_model import LinearRegression\nreg = LinearRegression().fit(X_tr, y_tr)")
    head = (f"{t.desc}, split with `train_test_split(df, y, random_state={s})`; `reg = LinearRegression()` was fitted on "
            f"the raw training columns.")
    if rng.random() < 0.5:
        return P(f"{head} Raw coefficients depend on the units, so rank by the effect of one standard deviation, "
                 f"|coef × std of the column in `X_tr`|. Return `(top 2 by that effect, top 2 by test permutation "
                 f"importance)` as two lists of names (`n_repeats=5, random_state={r}`).", setup,
                 f"{INS}permutation_importance\neffect = pd.Series(np.abs(reg.coef_ * X_tr.std(ddof=0).to_numpy()), index=df.columns)\n"
                 f"imp = pd.Series(permutation_importance(reg, X_te, y_te, n_repeats=5, random_state={r}).importances_mean, index=df.columns)\n"
                 "result = (list(effect.nlargest(2).index), list(imp.nlargest(2).index))",
                 require=["permutation_importance", "coef_"])
    return P(f"{head} For a linear model, shuffling column j adds about 2·coef_j²·Var(x_j) to the squared error, so the "
             f"expected drop in R² is 2·coef_j²·Var(x_j) / Var(y). For the column with the largest test permutation "
             f"importance (`n_repeats=20, random_state={r}`), return `(name, measured drop, predicted drop)` using the "
             f"test set's variances (`ddof=0`).", setup,
             f"{INS}permutation_importance\nimp = permutation_importance(reg, X_te, y_te, n_repeats=20, random_state={r}).importances_mean\n"
             "j = int(imp.argmax())\npred = 2 * reg.coef_[j] ** 2 * X_te.iloc[:, j].var(ddof=0) / np.var(y_te)\n"
             "result = (df.columns[j], float(imp[j]), float(pred))", require=["permutation_importance", "coef_"], tol=1e-6)


# ================================================================= challenge

@template(M, "challenge")
def tree_partial_dependence(rng):
    n, s = rng.randint(30, 50), seeded(rng)
    report = rng.choice(["pd", "range", "mean"])
    f = rng.randrange(3)
    build = ("from sklearn.tree import DecisionTreeRegressor\n"
             "t = DecisionTreeRegressor({args}random_state=0).fit(X_, y_).tree_\n"
             "left, right, feature, threshold = t.children_left, t.children_right, t.feature, t.threshold\n"
             "value, weight = t.value[:, 0, 0], t.weighted_n_node_samples\ndel t, X_, y_, DecisionTreeRegressor")
    setup = (f"g = np.random.default_rng({s})\nX_ = g.uniform(0, 10, ({n}, 3)).round(2)\n"
             f"y_ = (X_[:, 0] * X_[:, 1] / 10 + np.sin(X_[:, 2]) + g.normal(0, 0.3, {n})).round(2)\n"
             + build.format(args="max_depth=3, ")
             + f"\nf = {f}\ngrid = np.sort(g.uniform(-1, 11, 6)).round(2)\nreport = {report!r}")
    tests = [f"g = np.random.default_rng({s + i})\nX_ = g.uniform(0, 10, (20000, 4))\n"
             "y_ = X_[:, 0] * X_[:, 1] / 10 + np.sin(X_[:, 2]) - X_[:, 3] + g.normal(0, 0.3, 20000)\n"
             + build.format(args="max_leaf_nodes=3000, ")
             + f"\nf = {i}\ngrid = g.uniform(-1, 11, 200000)\nreport = {report!r}" for i in (1, 2)]
    sol = ("lo_l, hi_l, c_l = [], [], []\nstack = [(0, 1.0, -np.inf, np.inf)]\nwhile stack:\n"
           "    node, w, lo, hi = stack.pop()\n    if left[node] == -1:\n"
           "        lo_l.append(lo); hi_l.append(hi); c_l.append(w * value[node])\n    elif feature[node] == f:\n"
           "        thr = threshold[node]\n        stack.append((left[node], w, lo, min(hi, thr)))\n"
           "        stack.append((right[node], w, max(lo, thr), hi))\n    else:\n"
           "        wl = weight[left[node]] / weight[node]\n        stack.append((left[node], w * wl, lo, hi))\n"
           "        stack.append((right[node], w * (1 - wl), lo, hi))\n"
           "lo_a, hi_a, c_a = np.array(lo_l), np.array(hi_l), np.array(c_l)\n"
           "ol, oh = np.argsort(lo_a), np.argsort(hi_a)\nA = np.r_[0.0, np.cumsum(c_a[ol])]\nB = np.r_[0.0, np.cumsum(c_a[oh])]\n"
           "pd_ = A[np.searchsorted(lo_a[ol], grid, side='left')] - B[np.searchsorted(hi_a[oh], grid, side='left')]\n"
           + {"pd": "result = pd_", "range": "result = float(pd_.max() - pd_.min())",
              "mean": "result = float(pd_.mean())"}[report])
    want = {"pd": "the partial dependence at every value of `grid` (a float array)",
            "range": "max − min of the partial dependence over `grid` (a float)",
            "mean": "the mean of the partial dependence over `grid` (a float)"}[report]
    return challenge(
        "Partial dependence of a tree, fast",
        "For trees, `partial_dependence(..., method='recursion')` never looks at the data. The fitted regression tree is "
        "given as arrays (node 0 is the root; `left[i] == -1` marks a leaf; an inner node sends x to `left` when "
        "`x[feature[i]] <= threshold[i]`, else to `right`; `value[i]` is a leaf's prediction and `weight[i]` the number "
        "of training rows that reached node `i`). The partial dependence on feature `f` at value v walks down from the "
        "root with weight 1: at a split on `f` it follows the branch that v takes; at any other split it goes both "
        "ways, multiplying the weight by the share of the node's training rows that went to each child; the answer is "
        f"Σ weight × value over the leaves reached. Return {want}. (`report` says which: {report!r}.) Walking the "
        "tree once per grid value is too slow for big trees and grids: each leaf contributes a constant over an "
        "interval of v.", setup, sol, tests, "≤ 6·10^3 tree nodes, len(grid) ≤ 2·10^5",
        speed="O((nodes + grid) log)", tol=1e-6)


@template(M, "challenge")
def knn_boundary_on_a_line(rng):
    n, s = rng.randint(9, 14), seeded(rng)
    k = rng.choice([1, 3, 5])
    report = rng.choice(["labels", "changes", "share"])
    setup = (f"g = np.random.default_rng({s})\nx = np.sort(g.uniform(0, 10, {n}))\n"
             f"y = (np.sin(x) + g.normal(0, 0.6, {n}) > 0).astype(int)\nk = {k}\n"
             f"grid = np.sort(g.uniform(-1, 11, 8))\nreport = {report!r}")
    tests = [f"g = np.random.default_rng({s + i})\nx = np.sort(g.uniform(0, 1000, 100000))\n"
             f"y = (np.sin(x / 7) + g.normal(0, 0.8, 100000) > 0).astype(int)\nk = {[15, 31][i - 1]}\n"
             f"grid = np.sort(g.uniform(-10, 1010, 100000))\nreport = {report!r}" for i in (1, 2)]
    sol = ("n = len(x)\npre = np.r_[0, np.cumsum(y)]\nmid = (x[:n - k] + x[k:]) / 2\n"
           "start = np.searchsorted(mid, grid, side='left')\nones = pre[start + k] - pre[start]\n"
           "lab = (2 * ones > k).astype(int)\n"
           + {"labels": "result = lab", "changes": "result = int((lab[1:] != lab[:-1]).sum())",
              "share": "result = float(lab.mean())"}[report])
    want = {"labels": "the predicted label at every grid point (an int array)",
            "changes": "how many times the predicted label changes along the sorted grid (an int: the number of "
                       "boundary crossings a `DecisionBoundaryDisplay` would show)",
            "share": "the fraction of grid points predicted as 1 (a float)"}[report]
    return challenge(
        "k-NN decision boundary on a line",
        "`DecisionBoundaryDisplay` evaluates a model on a dense grid. Here the model is `KNeighborsClassifier("
        "n_neighbors=k)` on one feature: training points `x` (sorted, distinct) with labels `y` (0/1), `k` odd, so the "
        "majority vote never ties (and no grid point is exactly as far from two training points). The `k` nearest "
        f"neighbours of a point always form a contiguous window of the sorted `x`. For the sorted `grid`, return {want}. "
        f"(`report` says which: {report!r}.) Comparing every grid point with every training point is too slow; find each "
        "window with a binary search and count its 1s with prefix sums.",
        setup, sol, tests, "len(x), len(grid) ≤ 10^5, k ≤ 31", speed="O((n + grid) log n)", tol=1e-9)


@template(M, "challenge")
def knn_validation_curve(rng):
    n, s = rng.randint(9, 14), seeded(rng)
    K = rng.randint(3, 5)
    report = rng.choice(["curve", "best_k", "best_mse"])
    setup = (f"g = np.random.default_rng({s})\nx = np.sort(g.uniform(0, 10, {n}))\n"
             f"y = (np.sin(x) + g.normal(0, 0.4, {n})).round(3)\nK = {K}\nreport = {report!r}")
    tests = [f"g = np.random.default_rng({s + i})\nx = np.sort(g.uniform(0, 1000, 100000))\n"
             f"y = np.sin(x / 20) + g.normal(0, 0.5, 100000)\nK = {10 * i}\nreport = {report!r}" for i in (1, 2)]
    sol = ("n = len(x)\nidx = np.arange(n)\nl, r = idx - 1, idx + 1\ntot = np.zeros(n)\nmse = []\n"
           "for k in range(1, K + 1):\n    dl = np.where(l >= 0, x - x[np.maximum(l, 0)], np.inf)\n"
           "    dr = np.where(r < n, x[np.minimum(r, n - 1)] - x, np.inf)\n    take_left = dl < dr\n"
           "    tot += np.where(take_left, y[np.maximum(l, 0)], y[np.minimum(r, n - 1)])\n"
           "    l, r = np.where(take_left, l - 1, l), np.where(take_left, r, r + 1)\n"
           "    mse.append(((y - tot / k) ** 2).mean())\nmse = np.array(mse)\n"
           + {"curve": "result = mse", "best_k": "result = int(np.argmin(mse)) + 1",
              "best_mse": "result = float(mse.min())"}[report])
    want = {"curve": "the leave-one-out MSE for k = 1, ..., K (a float array of length K)",
            "best_k": "the k in 1..K with the smallest leave-one-out MSE (an int; the smallest k on ties)",
            "best_mse": "the smallest leave-one-out MSE over k = 1..K (a float)"}[report]
    return challenge(
        "Validation curve for k, in one sweep",
        "A `ValidationCurveDisplay` over `n_neighbors` for `KNeighborsRegressor` on one feature: the training points `x` "
        "are sorted and distinct, with targets `y`, and no two distances tie. With leave-one-out, point `i` is predicted "
        "by the mean target of its k nearest **other** points. Return "
        f"{want}. (`report` says which: {report!r}.) Refitting per k and per left-out point is far too slow; the "
        "neighbours of a point on a line come from growing a window outwards, one step per k.",
        setup, sol, tests, "n ≤ 10^5, K ≤ 20", speed="O(n·K)", tol=1e-6)
