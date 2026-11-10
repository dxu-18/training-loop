"""4.08 Linear models: least squares and its regularized cousins (Ridge, Lasso, ElasticNet, the CV variants), robust and
quantile regression, GLMs (Poisson), Bayesian ridge, SGD, and linear classifiers (logistic regression, ridge
classifier, perceptron)."""

from ._common import (P, catalog, challenge, lit, seeded, table, template)

M = "4.08"
LR = "LogisticRegression(max_iter=1000)"
LM = "from sklearn.linear_model import "
LRCV = "LogisticRegressionCV(Cs={cs}, cv={cv}, l1_ratios=(0.0,), use_legacy_attributes=False, max_iter=2000)"


def _reg(rng, n=(100, 250), d=(3, 7), noise=(5, 20)):
    """make_regression with coef=True -> X, y, true_coef (standard-normal features: no scaling needed)."""
    n_, d_ = rng.randint(*n), rng.randint(*d)
    inf, nz, s = rng.randint(2, d_), rng.randint(*noise), seeded(rng)
    setup = (f"from sklearn.datasets import make_regression\nX, y, true_coef = make_regression(n_samples={n_}, "
             f"n_features={d_}, n_informative={inf}, noise={nz}, coef=True, random_state={s})")
    desc = (f"`X` ({n_} × {d_}, standard-normal features) and `y` come from `make_regression` with {inf} informative "
            f"features and noise {nz}; `true_coef` holds the coefficients that generated `y`")
    return setup, desc, d_, n_


def _bin(rng):
    """Binary classification data, standardized: X, y (and names)."""
    s = seeded(rng)
    if rng.random() < 0.4:
        return ("from sklearn.datasets import load_breast_cancer\nfrom sklearn.preprocessing import StandardScaler\n"
                f"X, y = load_breast_cancer(return_X_y=True)\nnames = list(load_breast_cancer().feature_names)\n"
                f"order = np.random.default_rng({s}).permutation(len(y))\n"
                "X, y = StandardScaler().fit_transform(X)[order], y[order]",
                "`X` holds 569 breast-tumour samples (30 standardized cell measurements, named in `names`; rows shuffled) "
                "and `y` the diagnosis (0 malignant, 1 benign)", 30)
    n, d = rng.randint(150, 400), rng.randint(5, 10)
    i = rng.randint(2, 4)
    return (f"from sklearn.datasets import make_classification\nfrom sklearn.preprocessing import StandardScaler\n"
            f"X, y = make_classification(n_samples={n}, n_features={d}, n_informative={i}, n_redundant=0, flip_y=0.05, "
            f"random_state={s})\nX = StandardScaler().fit_transform(X)\nnames = [f'f{{j}}' for j in range({d})]",
            f"`X` ({n} × {d}, standardized) and `y` (0/1) come from `make_classification`; `names` holds `'f0'`, "
            f"`'f1'`, ...", d)


# ================================================================= lite

@template(M, "lite")
def linear_regressor_zoo(rng):
    setup, desc, d, n = _reg(rng)
    a, la, r, q = rng.choice([0.1, 1.0, 10.0]), rng.choice([0.5, 2.0, 5.0]), rng.choice([0.2, 0.5, 0.8]), \
        rng.choice([0.1, 0.5, 0.9])
    items = [
        ("LinearRegression", "Fit `LinearRegression` on `X, y` and return its coefficients.",
         f"{LM}LinearRegression\nresult = LinearRegression().fit(X, y).coef_"),
        ("intercept_", "Fit `LinearRegression` on `X, y` and return its intercept.",
         f"{LM}LinearRegression\nresult = LinearRegression().fit(X, y).intercept_"),
        ("score", "Fit `LinearRegression` on `X, y` and return its training R² (`score`).",
         f"{LM}LinearRegression\nresult = LinearRegression().fit(X, y).score(X, y)"),
        ("Ridge", f"Ridge adds an L2 penalty that shrinks the weights. Return the coefficients of `Ridge(alpha={a})` "
                  f"fitted on `X, y`.", f"{LM}Ridge\nresult = Ridge(alpha={a}).fit(X, y).coef_"),
        ("Lasso", f"Lasso's L1 penalty sets some weights exactly to 0. Fit `Lasso(alpha={la})` on `X, y` and return how "
                  f"many coefficients are non-zero.", f"{LM}Lasso\nresult = int((Lasso(alpha={la}).fit(X, y).coef_ != 0).sum())"),
        ("ElasticNet", f"Return the coefficients of `ElasticNet(alpha={la}, l1_ratio={r})` (a mix of L1 and L2) fitted "
                       f"on `X, y`.", f"{LM}ElasticNet\nresult = ElasticNet(alpha={la}, l1_ratio={r}).fit(X, y).coef_"),
        ("RidgeCV", "Let efficient leave-one-out CV pick the penalty: fit `RidgeCV(alphas=np.logspace(-3, 3, 13))` on "
                    "`X, y` and return `alpha_`.",
         f"{LM}RidgeCV\nresult = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(X, y).alpha_"),
        ("LassoCV", "Fit `LassoCV(cv=5, random_state=0)` on `X, y` and return the chosen `alpha_`.",
         f"{LM}LassoCV\nresult = LassoCV(cv=5, random_state=0).fit(X, y).alpha_"),
        ("HuberRegressor", "Fit `HuberRegressor()` (robust to outliers) on `X, y` and return its coefficients.",
         f"{LM}HuberRegressor\nresult = HuberRegressor().fit(X, y).coef_", {"tol": 1e-4}),
        ("QuantileRegressor", f"Fit `QuantileRegressor(quantile={q}, alpha=0.0)` (it predicts the {q} quantile of `y` "
                              f"instead of its mean) on `X, y` and return its intercept.",
         f"{LM}QuantileRegressor\nresult = QuantileRegressor(quantile={q}, alpha=0.0).fit(X, y).intercept_",
         {"tol": 1e-4}),
        ("BayesianRidge", "Fit `BayesianRidge()` on `X, y` and return its coefficients.",
         f"{LM}BayesianRidge\nresult = BayesianRidge().fit(X, y).coef_"),
        ("PoissonRegressor", "Counts need a Poisson model. Fit `PoissonRegressor()` on `X` and the count target "
                             "`np.abs(y).round()` and return its coefficients.",
         f"{LM}PoissonRegressor\nresult = PoissonRegressor().fit(X, np.abs(y).round()).coef_",
         {"require": ["PoissonRegressor", "abs", "round"], "tol": 1e-4}),
        ("SGDRegressor", "Fit `SGDRegressor(max_iter=1000, tol=1e-3, alpha=1e-4, random_state=0)` on "
                         "`StandardScaler().fit_transform(X)` and `y`, and return its coefficients.",
         f"{LM}SGDRegressor\nfrom sklearn.preprocessing import StandardScaler\n"
         "result = SGDRegressor(max_iter=1000, tol=1e-3, alpha=1e-4, random_state=0).fit(StandardScaler().fit_transform(X), y).coef_",
         {"require": ["SGDRegressor", "StandardScaler"]}),
        ("coef_", "How close did least squares get to the truth? Return `LinearRegression().fit(X, y).coef_ - true_coef`.",
         f"{LM}LinearRegression\nresult = LinearRegression().fit(X, y).coef_ - true_coef"),
    ]
    return catalog(rng, items, setup, f"{desc}.", tol=1e-6)


@template(M, "lite")
def linear_classifier_zoo(rng):
    setup, desc, d = _bin(rng)
    c, n = rng.choice([0.05, 0.3, 1.0, 5.0]), rng.randint(2, 4)
    items = [
        ("LogisticRegression", f"`C` is the *inverse* regularization strength. Fit `LogisticRegression(C={c}, "
                               f"max_iter=1000)` on `X, y` and return its training accuracy.",
         f"{LM}LogisticRegression\nresult = LogisticRegression(C={c}, max_iter=1000).fit(X, y).score(X, y)"),
        ("l1_ratio", f"L1-regularized logistic regression (scikit-learn 1.8 spells it `l1_ratio=1.0`): fit "
                     f"`LogisticRegression(C={c}, l1_ratio=1.0, solver='liblinear', random_state=0)` on `X, y` and return how many "
                     f"coefficients are non-zero.",
         f"{LM}LogisticRegression\nclf = LogisticRegression(C={c}, l1_ratio=1.0, solver='liblinear', random_state=0).fit(X, y)\n"
         "result = int((clf.coef_ != 0).sum())"),
        ("l1_ratio", f"Elastic net: fit `LogisticRegression(C={c}, l1_ratio=0.5, solver='saga', max_iter=5000, "
                     f"random_state=0)` on `X, y` and return how many coefficients are non-zero.",
         f"{LM}LogisticRegression\nclf = LogisticRegression(C={c}, l1_ratio=0.5, solver='saga', max_iter=5000, "
         "random_state=0).fit(X, y)\nresult = int((clf.coef_ != 0).sum())"),
        ("class_weight", "`class_weight='balanced'` reweights the classes inversely to their frequency. Fit "
                         "`LogisticRegression(class_weight='balanced', max_iter=1000)` on `X, y` and return how many "
                         "rows it predicts as class 1.",
         f"{LM}LogisticRegression\n"
         "result = int(LogisticRegression(class_weight='balanced', max_iter=1000).fit(X, y).predict(X).sum())"),
        ("LogisticRegressionCV", f"Let cross-validation pick `C`: fit `{LRCV.format(cs=5, cv=3)}` on `X, y` and return "
                                 f"`C_` (the `l1_ratios`/`use_legacy_attributes` arguments just silence 1.8 "
                                 f"transition warnings).",
         f"{LM}LogisticRegressionCV\nresult = float({LRCV.format(cs=5, cv=3)}.fit(X, y).C_)"),
        ("SGDClassifier", "Fit `SGDClassifier(loss='log_loss', alpha=1e-4, random_state=0)` (logistic regression trained "
                          "by stochastic gradient descent) on `X, y` and return its training accuracy.",
         f"{LM}SGDClassifier\nresult = SGDClassifier(loss='log_loss', alpha=1e-4, random_state=0).fit(X, y).score(X, y)"),
        ("decision_function", f"`loss='hinge'` makes `SGDClassifier` a linear SVM. Fit `SGDClassifier(loss='hinge', "
                              f"random_state=0)` on `X, y` and return `decision_function` of the first {n} rows.",
         f"{LM}SGDClassifier\nresult = SGDClassifier(loss='hinge', random_state=0).fit(X, y).decision_function(X[:{n}])"),
        ("partial_fit", "Train in mini-batches: create `sgd = SGDClassifier(loss='log_loss', random_state=0)` and call "
                        "`sgd.partial_fit(X[i:i + 50], y[i:i + 50], classes=np.array([0, 1]))` for i = 0, 50, 100, ... "
                        "Return the final training accuracy on `X, y`.",
         f"{LM}SGDClassifier\nsgd = SGDClassifier(loss='log_loss', random_state=0)\nfor i in range(0, len(X), 50):\n"
         "    sgd.partial_fit(X[i:i + 50], y[i:i + 50], classes=np.array([0, 1]))\nresult = sgd.score(X, y)",
         {"require": ["partial_fit", "array"]}),
        ("RidgeClassifier", "Fit `RidgeClassifier()` (ridge regression on ±1 targets) on `X, y` and return its training "
                            "accuracy.", f"{LM}RidgeClassifier\nresult = RidgeClassifier().fit(X, y).score(X, y)"),
        ("Perceptron", "Fit `Perceptron(random_state=0)` on `X, y` and return its training accuracy.",
         f"{LM}Perceptron\nresult = Perceptron(random_state=0).fit(X, y).score(X, y)"),
        ("predict_proba", f"`loss='modified_huber'` gives `SGDClassifier` probabilities. Fit `SGDClassifier("
                          f"loss='modified_huber', random_state=0)` on `X, y` and return `predict_proba` of the first "
                          f"{n} rows.",
         f"{LM}SGDClassifier\nresult = SGDClassifier(loss='modified_huber', random_state=0).fit(X, y).predict_proba(X[:{n}])"),
        ("inf", "`C=np.inf` turns the penalty off. Fit `LogisticRegression(C=np.inf, max_iter=1000)` and the default "
                "`LogisticRegression(max_iter=1000)` on `X, y` and return the L2 norms of their `coef_` as a tuple "
                "`(unpenalized, default)`. (scikit-learn 1.8 emits a spurious UserWarning for `C=np.inf`, and the "
                "unpenalized fit may not converge on separable data: run it inside `warnings.catch_warnings()` with "
                "`warnings.simplefilter('ignore')`.)",
         f"import warnings\n{LM}LogisticRegression\nwith warnings.catch_warnings():\n    warnings.simplefilter('ignore')\n"
         "    free = LogisticRegression(C=np.inf, max_iter=1000).fit(X, y)\n"
         "result = (np.linalg.norm(free.coef_), np.linalg.norm(LogisticRegression(max_iter=1000).fit(X, y).coef_))",
         {"tol": 1e-4}),
    ]
    return catalog(rng, items, setup, f"{desc}.", tol=1e-6)


# ================================================================= easy

@template(M, "easy")
def lasso_sparsity(rng):
    setup, desc, d, n = _reg(rng, d=(5, 9))
    task = rng.choice(["lasso", "enet", "compare"])
    if task == "lasso":
        alphas = sorted(rng.sample([0.1, 1.0, 5.0, 10.0, 30.0, 60.0], 3))
        return P(f"{desc}. Fit `Lasso(alpha=a)` on `X, y` for each `a` in `{alphas}` and return the number of non-zero "
                 f"coefficients for each, as a list.", setup,
                 f"{LM}Lasso\nresult = [int((Lasso(alpha=a).fit(X, y).coef_ != 0).sum()) for a in {alphas}]",
                 require=["Lasso"])
    if task == "enet":
        rs = [0.1, 0.5, 1.0]
        a = rng.choice([5.0, 10.0, 20.0])
        return P(f"{desc}. Fit `ElasticNet(alpha={a}, l1_ratio=r)` on `X, y` for each `r` in `{rs}` (more L1 → sparser) "
                 f"and return the number of non-zero coefficients for each, as a list.", setup,
                 f"{LM}ElasticNet\nresult = [int((ElasticNet(alpha={a}, l1_ratio=r).fit(X, y).coef_ != 0).sum()) for r in {rs}]",
                 require=["ElasticNet"])
    a = rng.choice([5.0, 10.0, 20.0])
    return P(f"{desc}. L2 shrinks, L1 zeroes. Return `(ridge_zeros, lasso_zeros)`: how many coefficients are exactly 0 "
             f"for `Ridge(alpha={a})` and for `Lasso(alpha={a})`, both fitted on `X, y`.", setup,
             f"{LM}Ridge, Lasso\nresult = (int((Ridge(alpha={a}).fit(X, y).coef_ == 0).sum()), "
             f"int((Lasso(alpha={a}).fit(X, y).coef_ == 0).sum()))", require=["Ridge", "Lasso"])


@template(M, "easy")
def ridge_closed_form(rng):
    setup, desc, d, n = _reg(rng)
    a = rng.choice([0.5, 1.0, 10.0, 50.0])
    task = rng.choice(["coef", "intercept", "no_intercept"])
    cen = ("Xc, yc = X - X.mean(axis=0), y - y.mean()\n"
           f"w = np.linalg.solve(Xc.T @ Xc + {a} * np.eye(X.shape[1]), Xc.T @ yc)\n")
    if task == "coef":
        return P(f"{desc}. Ridge has a closed form. With an intercept, center the data first (`Xc = X - mean`, `yc = y - "
                 f"mean`), then w = (Xcᵀ Xc + α I)⁻¹ Xcᵀ yc. Compute it by hand for α = {a} and return w (it equals "
                 f"`Ridge(alpha={a}).fit(X, y).coef_`).", setup, cen + "result = w", require=["solve"], tol=1e-6)
    if task == "intercept":
        return P(f"{desc}. After centering (w = (Xcᵀ Xc + α I)⁻¹ Xcᵀ yc with α = {a}), the intercept puts the line "
                 f"through the means: b = mean(y) − mean(X) · w. Return `(by_hand_b, Ridge(alpha={a}).fit(X, y)"
                 f".intercept_)`.", setup,
                 f"{LM}Ridge\n" + cen + f"result = (y.mean() - X.mean(axis=0) @ w, Ridge(alpha={a}).fit(X, y).intercept_)",
                 require=["Ridge", "intercept_"], tol=1e-6)
    return P(f"{desc}. Without an intercept no centering is needed: w = (Xᵀ X + α I)⁻¹ Xᵀ y. Return `(by_hand, "
             f"Ridge(alpha={a}, fit_intercept=False).fit(X, y).coef_)`.", setup,
             f"{LM}Ridge\nw = np.linalg.solve(X.T @ X + {a} * np.eye(X.shape[1]), X.T @ y)\n"
             f"result = (w, Ridge(alpha={a}, fit_intercept=False).fit(X, y).coef_)", require=["Ridge", "fit_intercept"],
             tol=1e-6)


@template(M, "easy")
def ols_normal_equations(rng):
    setup, desc, d, n = _reg(rng)
    task = rng.choice(["solve", "orthogonal", "r2"])
    if task == "solve":
        return P(f"{desc}. Least squares solves the normal equations AᵀA β = Aᵀy, where A is `X` with a column of ones "
                 f"appended. Solve them with `np.linalg.solve` and return β = `[*coef, intercept]`.", setup,
                 "A = np.column_stack([X, np.ones(len(X))])\nresult = np.linalg.solve(A.T @ A, A.T @ y)",
                 require=["solve"], tol=1e-6)
    if task == "orthogonal":
        return P(f"{desc}. Least-squares residuals are orthogonal to every feature and sum to zero. Fit "
                 f"`LinearRegression` on `X, y` and return `(X.T @ r, r.sum())` for the residuals `r = y − predict(X)` "
                 f"(all ≈ 0).", setup,
                 f"{LM}LinearRegression\nr = y - LinearRegression().fit(X, y).predict(X)\nresult = (X.T @ r, r.sum())",
                 require=["LinearRegression"], tol=1e-6)
    return P(f"{desc}. Fit `LinearRegression` on `X, y` and return `(by_hand, score)`: the training R² computed by hand "
             f"as 1 − SS_res / SS_tot, and `score(X, y)`.", setup,
             f"{LM}LinearRegression\nreg = LinearRegression().fit(X, y)\nr = y - reg.predict(X)\n"
             "result = (1 - (r ** 2).sum() / ((y - y.mean()) ** 2).sum(), reg.score(X, y))", require=["score"], tol=1e-9)


@template(M, "easy")
def l1_logistic_survivors(rng):
    setup, desc, d = _bin(rng)
    c = rng.choice([0.02, 0.05, 0.1, 0.3])
    task = rng.choice(["names", "counts"])
    if task == "names":
        return P(f"{desc}. Fit L1-regularized `LogisticRegression(C={c}, l1_ratio=1.0, solver='liblinear', random_state=0)` on `X, y` "
                 f"and return the names of the features with a non-zero coefficient.", setup,
                 f"{LM}LogisticRegression\nclf = LogisticRegression(C={c}, l1_ratio=1.0, solver='liblinear', random_state=0).fit(X, y)\n"
                 "result = [names[j] for j in np.flatnonzero(clf.coef_[0])]", require=["l1_ratio", "coef_"])
    cs = sorted(rng.sample([0.01, 0.03, 0.1, 0.3, 1.0], 3))
    return P(f"{desc}. For each `C` in `{cs}`, fit `LogisticRegression(C=C, l1_ratio=1.0, solver='liblinear', random_state=0)` on "
             f"`X, y` and count the non-zero coefficients. Return the counts as a list.", setup,
             f"{LM}LogisticRegression\n"
             f"result = [int((LogisticRegression(C=C, l1_ratio=1.0, solver='liblinear', random_state=0).fit(X, y).coef_ != 0).sum()) for C in {cs}]",
             require=["l1_ratio"])


@template(M, "easy")
def minibatch_sgd(rng):
    setup, desc, d = _bin(rng)
    b = rng.choice([25, 50, 100])
    task = rng.choice(["score", "epochs", "coef"])
    loop = (f"{LM}SGDClassifier\nsgd = SGDClassifier(loss='log_loss', random_state=0)\n"
            f"for i in range(0, len(X), {b}):\n    sgd.partial_fit(X[i:i + {b}], y[i:i + {b}], classes=np.array([0, 1]))\n")
    head = (f"{desc}. Stream the data: create `sgd = SGDClassifier(loss='log_loss', random_state=0)` and feed it "
            f"mini-batches of {b} rows in order with `partial_fit` (pass `classes=np.array([0, 1])`: the first batch "
            f"might not contain every class).")
    if task == "score":
        return P(f"{head} After one pass, return the training accuracy on `X, y`.", setup, loop + "result = sgd.score(X, y)",
                 require=["partial_fit"])
    if task == "epochs":
        return P(f"{head} Make **3** passes over the data and return the training accuracy after each pass, as a list.",
                 setup,
                 f"{LM}SGDClassifier\nsgd = SGDClassifier(loss='log_loss', random_state=0)\nresult = []\nfor epoch in range(3):\n"
                 f"    for i in range(0, len(X), {b}):\n        sgd.partial_fit(X[i:i + {b}], y[i:i + {b}], classes=np.array([0, 1]))\n"
                 "    result.append(sgd.score(X, y))", require=["partial_fit"])
    return P(f"{head} After one pass, return `(np.linalg.norm(sgd.coef_), sgd.t_)`: the size of the weights and "
             f"`t_`, the update counter (samples seen + 1).", setup, loop + "result = (np.linalg.norm(sgd.coef_), sgd.t_)",
             require=["partial_fit"], tol=1e-6)


@template(M, "easy")
def shrinkage_paths(rng):
    setup, desc, d, n = _reg(rng)
    task = rng.choice(["ridge", "alpha_max"])
    if task == "ridge":
        alphas = sorted(rng.sample([0.1, 1.0, 10.0, 100.0, 1000.0], 4))
        return P(f"{desc}. Watch Ridge shrink the weights: for each `alpha` in `{alphas}` fit `Ridge(alpha=alpha)` on "
                 f"`X, y` and record the L2 norm of `coef_`. Return the list.", setup,
                 f"{LM}Ridge\nresult = [np.linalg.norm(Ridge(alpha=a).fit(X, y).coef_) for a in {alphas}]",
                 require=["Ridge"], tol=1e-6)
    return P(f"{desc}. Above `alpha_max = max |Xcᵀ (y − mean(y))| / n` (Xc = centered `X`), Lasso sets **every** "
             f"coefficient to 0. Return `(alpha_max, nonzero just above, nonzero at 0.9 × alpha_max)`, fitting "
             f"`Lasso(alpha=1.01 * alpha_max)` and `Lasso(alpha=0.9 * alpha_max)` on `X, y`.", setup,
             f"{LM}Lasso\nXc = X - X.mean(axis=0)\namax = np.abs(Xc.T @ (y - y.mean())).max() / len(y)\n"
             "result = (amax, int((Lasso(alpha=1.01 * amax).fit(X, y).coef_ != 0).sum()), "
             "int((Lasso(alpha=0.9 * amax).fit(X, y).coef_ != 0).sum()))", require=["Lasso"], tol=1e-9)


@template(M, "easy")
def huber_vs_ols(rng):
    setup, desc, d, n = _reg(rng, noise=(2, 6))
    k, s = rng.randint(5, 12), seeded(rng)
    setup += f"\nbad = np.random.default_rng({s}).choice(len(y), {k}, replace=False)\ny[bad] += 400   # gross outliers"
    task = rng.choice(["error", "outliers", "intercept"])
    if task == "error":
        return P(f"{desc}; then {k} targets were corrupted by +400. Return `(ols_error, huber_error)`: the distance "
                 f"(`np.linalg.norm`) between `true_coef` and the coefficients of `LinearRegression` and of "
                 f"`HuberRegressor()`, each fitted on `X, y`.", setup,
                 f"{LM}LinearRegression, HuberRegressor\na = np.linalg.norm(LinearRegression().fit(X, y).coef_ - true_coef)\n"
                 "b = np.linalg.norm(HuberRegressor().fit(X, y).coef_ - true_coef)\nresult = (a, b)",
                 require=["HuberRegressor"], tol=1e-4)
    if task == "outliers":
        return P(f"{desc}; then {k} targets (indices `bad`) were corrupted by +400. Fit `HuberRegressor()` on `X, y`; "
                 f"its `outliers_` mask flags rows with large residuals. Return `(flagged, caught)`: how many rows it "
                 f"flags and how many of the `bad` rows are among them.", setup,
                 f"{LM}HuberRegressor\nm = HuberRegressor().fit(X, y).outliers_\n"
                 "result = (int(m.sum()), int(m[bad].sum()))", require=["outliers_"])
    return P(f"{desc}; then {k} targets were corrupted by +400. Outliers drag the intercept. Return the intercepts "
             f"`(ols, huber)` of `LinearRegression` and `HuberRegressor()` fitted on `X, y`.", setup,
             f"{LM}LinearRegression, HuberRegressor\n"
             "result = (LinearRegression().fit(X, y).intercept_, HuberRegressor().fit(X, y).intercept_)",
             require=["HuberRegressor"], tol=1e-4)


@template(M, "easy")
def quantile_coverage(rng):
    setup, desc, d, n = _reg(rng, noise=(10, 30))
    q = rng.choice([0.1, 0.25, 0.75, 0.9])
    if rng.random() < 0.5:
        return P(f"{desc}. Fit `QuantileRegressor(quantile={q}, alpha=0.0)` on `X, y` and return the fraction of rows "
                 f"whose target is **at or below** the prediction (close to {q}: that's what a quantile means).", setup,
                 f"{LM}QuantileRegressor\npred = QuantileRegressor(quantile={q}, alpha=0.0).fit(X, y).predict(X)\n"
                 "result = float((y <= pred).mean())", require=["QuantileRegressor"])
    return P(f"{desc}. A prediction band: fit `QuantileRegressor(quantile=0.1, alpha=0.0)` and `QuantileRegressor("
             f"quantile=0.9, alpha=0.0)` on `X, y` and return `(coverage, mean_width)`: the fraction of rows inside "
             f"[low, high] and the mean of high − low.", setup,
             f"{LM}QuantileRegressor\nlo = QuantileRegressor(quantile=0.1, alpha=0.0).fit(X, y).predict(X)\n"
             "hi = QuantileRegressor(quantile=0.9, alpha=0.0).fit(X, y).predict(X)\n"
             "result = (float(((y >= lo) & (y <= hi)).mean()), float((hi - lo).mean()))",
             require=["QuantileRegressor"], tol=1e-4)


@template(M, "easy")
def poisson_link(rng):
    t = table(rng, kind="reg", numeric_only=True, only=("bikes",), n=(100, 250))
    n = rng.randint(3, 5)
    task = rng.choice(["by_hand", "negative"])
    setup = t.setup + "\nfrom sklearn.preprocessing import StandardScaler\nX = StandardScaler().fit_transform(df)"
    if task == "by_hand":
        return P(f"{t.desc} (a count); `X` is `df` standardized. A Poisson GLM predicts exp(X·w + b). Fit "
                 f"`PoissonRegressor()` on `X, y` and recompute its predictions for the first {n} rows by hand from "
                 f"`coef_` and `intercept_`.", setup,
                 f"{LM}PoissonRegressor\nm = PoissonRegressor().fit(X, y)\nresult = np.exp(X[:{n}] @ m.coef_ + m.intercept_)",
                 require=["PoissonRegressor", "exp"], tol=1e-6)
    return P(f"{t.desc} (a count); `X` is `df` standardized. Return `(linear_negative, poisson_min)`: how many training "
             f"predictions of `LinearRegression` are below 0, and the smallest prediction of `PoissonRegressor()` (never "
             f"negative), both fitted on `X, y`.", setup,
             f"{LM}LinearRegression, PoissonRegressor\nneg = int((LinearRegression().fit(X, y).predict(X) < 0).sum())\n"
             "result = (neg, float(PoissonRegressor().fit(X, y).predict(X).min()))", require=["PoissonRegressor"],
             tol=1e-6)


@template(M, "easy")
def margin_classifiers(rng):
    setup, desc, d = _bin(rng)
    n = rng.randint(3, 6)
    task = rng.choice(["ridge", "perceptron", "sign"])
    if task == "ridge":
        return P(f"{desc}. Fit `RidgeClassifier()` on `X, y` and recompute `decision_function` of the first {n} rows by "
                 f"hand from `coef_` and `intercept_` (a 1-D array; flatten `coef_` with `ravel()`).", setup,
                 f"{LM}RidgeClassifier\nm = RidgeClassifier().fit(X, y)\nresult = X[:{n}] @ m.coef_.ravel() + m.intercept_",
                 require=["coef_", "intercept_"], tol=1e-6)
    if task == "perceptron":
        return P(f"{desc}. Fit `Perceptron(random_state=0)` on `X, y`. Its prediction is class 1 when X·w + b > 0. "
                 f"Rebuild the predictions for all rows from `coef_` and `intercept_` and return how many match "
                 f"`predict(X)`.", setup,
                 f"{LM}Perceptron\nm = Perceptron(random_state=0).fit(X, y)\n"
                 "mine = (X @ m.coef_.ravel() + m.intercept_ > 0).astype(int)\nresult = int((mine == m.predict(X)).sum())",
                 require=["Perceptron", "coef_"])
    return P(f"{desc}. `RidgeClassifier` regresses on targets −1/+1. Fit `Ridge(alpha=1.0)` on `X` and `2 * y - 1` and "
             f"return `(same_coef, same_predictions)`: whether its `coef_` matches `RidgeClassifier().fit(X, y).coef_.ravel()` "
             f"(`np.allclose`) and whether `(ridge.predict(X) > 0)` matches `RidgeClassifier`'s predictions.", setup,
             f"{LM}Ridge, RidgeClassifier\nr = Ridge(alpha=1.0).fit(X, 2 * y - 1)\nrc = RidgeClassifier().fit(X, y)\n"
             "result = (bool(np.allclose(r.coef_, rc.coef_.ravel())), bool(((r.predict(X) > 0).astype(int) == rc.predict(X)).all()))",
             require=["Ridge", "RidgeClassifier"])


@template(M, "easy")
def regression_api_tour(rng):
    t = table(rng, kind="reg", numeric_only=True, n=(80, 200))
    a = rng.choice([0.1, 1.0, 5.0])
    col = rng.choice(t.num)
    setup = (t.setup + "\nfrom sklearn.preprocessing import StandardScaler\nX = StandardScaler().fit_transform(df)\n"
             f"{LM}Ridge, Lasso\nridge = Ridge(alpha={a}).fit(X, y)\nlasso = Lasso(alpha={a}).fit(X, y)")
    j = t.num.index(col)
    items = [
        ("coef_", "Return a dict {column: ridge coefficient}.", "result = dict(zip(df.columns, ridge.coef_))"),
        ("coef_", f"Return `(ridge, lasso)` coefficients of `{col}`.", f"result = (ridge.coef_[{j}], lasso.coef_[{j}])"),
        ("intercept_", "Both intercepts equal the mean of `y` (the features are centered). Return `(ridge.intercept_, "
                       "lasso.intercept_, y.mean())`.", "result = (ridge.intercept_, lasso.intercept_, y.mean())"),
        ("predict", "Return the ridge predictions for the first 3 rows.", "result = ridge.predict(X[:3])"),
        ("score", "Return the training R² of both models as a tuple `(ridge, lasso)`.",
         "result = (ridge.score(X, y), lasso.score(X, y))"),
        ("n_iter_", "Lasso is solved by coordinate descent: return `lasso.n_iter_` (passes over the features).",
         "result = int(lasso.n_iter_)"),
        ("predict", "Return the mean absolute difference between the ridge and the lasso predictions on `X`.",
         "result = float(np.abs(ridge.predict(X) - lasso.predict(X)).mean())"),
        ("coef_", "Which column has the largest absolute ridge coefficient? Return its name.",
         "result = df.columns[np.abs(ridge.coef_).argmax()]"),
        ("n_features_in_", "Return `(ridge.n_features_in_, lasso.coef_.shape)`.",
         "result = (ridge.n_features_in_, lasso.coef_.shape)"),
    ]
    return catalog(rng, items, setup, f"{t.desc}. `X` is `df` standardized; `ridge = Ridge(alpha={a})` and `lasso = "
                                      f"Lasso(alpha={a})` are fitted on `X, y`.", tol=1e-6)


# ================================================================= medium

@template(M, "medium")
def regressor_comparison(rng):
    t = table(rng, kind="reg", numeric_only=True, n=(150, 300))
    m = rng.randint(t.n // 2, 3 * t.n // 4)
    s = seeded(rng)
    setup = t.setup + (f"\norder = np.random.default_rng({s}).permutation(len(df))\ndf, y = df.iloc[order], y[order]\n"
                       f"X_train, X_test, y_train, y_test = df.iloc[:{m}], df.iloc[{m}:], y[:{m}], y[{m}:]")
    pipe = "from sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\n"
    a = rng.choice([0.5, 1.0, 5.0])
    items = [
        ("Ridge", "Return the test R² of `make_pipeline(StandardScaler(), model)` for `model` = `LinearRegression()`, "
                  f"`Ridge(alpha={a * 10})` and `Lasso(alpha={a})`, as a dict with keys `'ols'`, `'ridge'`, `'lasso'`.",
         f"{pipe}{LM}LinearRegression, Ridge, Lasso\n"
         f"models = {{'ols': LinearRegression(), 'ridge': Ridge(alpha={a * 10}), 'lasso': Lasso(alpha={a})}}\n"
         "result = {k: make_pipeline(StandardScaler(), m).fit(X_train, y_train).score(X_test, y_test) for k, m in models.items()}"),
        ("RidgeCV", "Fit `make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))` on the training rows "
                    "and return `(chosen alpha_, test R²)`.",
         f"{pipe}{LM}RidgeCV\np = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13))).fit(X_train, y_train)\n"
         "result = (p[-1].alpha_, p.score(X_test, y_test))", {"require": ["RidgeCV", "logspace"]}),
        ("LassoCV", "Fit `make_pipeline(StandardScaler(), LassoCV(cv=5, random_state=0))` on the training rows and "
                    "return a dict {column: coefficient} for the columns whose coefficient is non-zero.",
         f"{pipe}{LM}LassoCV\np = make_pipeline(StandardScaler(), LassoCV(cv=5, random_state=0)).fit(X_train, y_train)\n"
         "result = {c: w for c, w in zip(df.columns, p[-1].coef_) if w != 0}"),
        ("ElasticNet", f"Return the test R² of `make_pipeline(StandardScaler(), ElasticNet(alpha={a}, l1_ratio=r))` for "
                       f"`r` in 0.1, 0.5, 0.9 as a list.",
         f"{pipe}{LM}ElasticNet\n"
         f"result = [make_pipeline(StandardScaler(), ElasticNet(alpha={a}, l1_ratio=r)).fit(X_train, y_train).score(X_test, y_test) "
         "for r in (0.1, 0.5, 0.9)]"),
        ("HuberRegressor", "Return the test mean absolute error of `make_pipeline(StandardScaler(), HuberRegressor())` "
                           "and of the same pipeline with `LinearRegression()`, as a tuple `(huber, ols)`.",
         f"{pipe}{LM}HuberRegressor, LinearRegression\n"
         "h = make_pipeline(StandardScaler(), HuberRegressor()).fit(X_train, y_train).predict(X_test)\n"
         "o = make_pipeline(StandardScaler(), LinearRegression()).fit(X_train, y_train).predict(X_test)\n"
         "result = (float(np.abs(y_test - h).mean()), float(np.abs(y_test - o).mean()))"),
        ("BayesianRidge", "`BayesianRidge` also gives an uncertainty: fit `make_pipeline(StandardScaler(), "
                          "BayesianRidge())` on the training rows and return `predict(X_test.iloc[:3], return_std=True)` "
                          "(a tuple of means and standard deviations).",
         f"{pipe}{LM}BayesianRidge\np = make_pipeline(StandardScaler(), BayesianRidge()).fit(X_train, y_train)\n"
         "result = p.predict(X_test.iloc[:3], return_std=True)"),
        ("SGDRegressor", "Return the test R² of `make_pipeline(StandardScaler(), SGDRegressor(max_iter=1000, tol=1e-3, "
                         "random_state=0))` trained on the training rows.",
         f"{pipe}{LM}SGDRegressor\n"
         "result = make_pipeline(StandardScaler(), SGDRegressor(max_iter=1000, tol=1e-3, random_state=0)).fit(X_train, y_train).score(X_test, y_test)"),
        ("Lasso", "Return the column whose Lasso coefficient survives the strongest penalty: standardize the training "
                  "rows, and among `alpha` in `np.logspace(-1, 3, 30)` find, for each column, the largest `alpha` at "
                  "which its coefficient is still non-zero; return the column with the largest such alpha.",
         f"{LM}Lasso\nfrom sklearn.preprocessing import StandardScaler\nZ = StandardScaler().fit_transform(X_train)\n"
         "alive = {}\nfor a in np.logspace(-1, 3, 30):\n    for c, w in zip(df.columns, Lasso(alpha=a).fit(Z, y_train).coef_):\n"
         "        if w != 0:\n            alive[c] = a\nresult = max(alive, key=alive.get)"),
        ("score", "Does the penalty matter on the test rows? Return the test R² of `make_pipeline(StandardScaler(), "
                  "Ridge(alpha=a))` for `a` in 0.01, 1, 100, 10000.",
         f"{pipe}{LM}Ridge\n"
         "result = [make_pipeline(StandardScaler(), Ridge(alpha=a)).fit(X_train, y_train).score(X_test, y_test) for a in (0.01, 1, 100, 10000)]"),
    ]
    return catalog(rng, items, setup, f"{t.desc}. Rows were shuffled and split into `X_train, y_train` (first {m}) and "
                                      f"`X_test, y_test`.", tol=1e-6)


@template(M, "medium")
def logistic_tasks(rng):
    t = table(rng, kind="clf", numeric_only=True, n=(150, 300))
    setup = t.setup + "\nfrom sklearn.preprocessing import StandardScaler\nX = StandardScaler().fit_transform(df)"
    cs = sorted(rng.sample([0.001, 0.01, 0.1, 1.0, 10.0], 3))
    c = rng.choice([0.05, 0.2])
    items = [
        ("LogisticRegression", f"For `C` in `{cs}`, fit `LogisticRegression(C=C, max_iter=1000)` on `X, y` and record "
                               f"the L2 norm of `coef_`. Return the list (smaller C, smaller weights).",
         f"{LM}LogisticRegression\nresult = [np.linalg.norm(LogisticRegression(C=C, max_iter=1000).fit(X, y).coef_) for C in {cs}]"),
        ("coef_", f"Fit `{LR}` on `X, y` and return a dict {{column: coefficient}}.",
         f"{LM}LogisticRegression\nresult = dict(zip(df.columns, {LR}.fit(X, y).coef_[0]))"),
        ("l1_ratio", f"Fit `LogisticRegression(C={c}, l1_ratio=1.0, solver='liblinear', random_state=0)` on `X, y` and return the names "
                     f"of the columns it keeps (non-zero weight).",
         f"{LM}LogisticRegression\nclf = LogisticRegression(C={c}, l1_ratio=1.0, solver='liblinear', random_state=0).fit(X, y)\n"
         "result = list(df.columns[clf.coef_[0] != 0])"),
        ("LogisticRegressionCV", f"Fit `{LRCV.format(cs=6, cv=4)}` on `X, y` and return `(C_, training accuracy)`.",
         f"{LM}LogisticRegressionCV\nm = {LRCV.format(cs=6, cv=4)}.fit(X, y)\nresult = (float(m.C_), m.score(X, y))"),
        ("Cs_", f"Fit `{LRCV.format(cs=6, cv=4)}` on `X, y`. `Cs_` is the grid it tried and `scores_` has shape "
                f"(folds, 1, len(Cs_)). Return the mean CV accuracy for each C, as an array.",
         f"{LM}LogisticRegressionCV\nm = {LRCV.format(cs=6, cv=4)}.fit(X, y)\nresult = m.scores_.mean(axis=0)[0]",
         {"require": ["scores_"]}),
        ("SGDClassifier", "Compare losses: return the training accuracy of `SGDClassifier(loss=L, random_state=0)` for "
                          "L in 'hinge', 'log_loss', 'modified_huber', as a dict.",
         f"{LM}SGDClassifier\n"
         "result = {L: SGDClassifier(loss=L, random_state=0).fit(X, y).score(X, y) for L in ('hinge', 'log_loss', 'modified_huber')}"),
        ("predict_proba", f"Fit `{LR}` on `X, y`. How does the decision threshold trade errors? For thresholds 0.3, 0.5 "
                          f"and 0.7, return the number of rows predicted positive (P(class 1) ≥ threshold).",
         f"{LM}LogisticRegression\np = {LR}.fit(X, y).predict_proba(X)[:, 1]\nresult = [int((p >= t).sum()) for t in (0.3, 0.5, 0.7)]"),
        ("RidgeClassifier", f"Return the training accuracies `(logistic, ridge, perceptron)` of `{LR}`, "
                            f"`RidgeClassifier()` and `Perceptron(random_state=0)` on `X, y`.",
         f"{LM}LogisticRegression, RidgeClassifier, Perceptron\n"
         f"result = ({LR}.fit(X, y).score(X, y), RidgeClassifier().fit(X, y).score(X, y), "
         "Perceptron(random_state=0).fit(X, y).score(X, y))"),
        ("n_iter_", "Return the number of iterations `LogisticRegression(max_iter=1000)` needed on the standardized `X` "
                    "and on the raw `df` (`n_iter_[0]`), as a tuple: scaling helps the solver.",
         f"import warnings\n{LM}LogisticRegression\na = int({LR}.fit(X, y).n_iter_[0])\n"
         f"with warnings.catch_warnings():\n    warnings.simplefilter('ignore')\n    b = int({LR}.fit(df, y).n_iter_[0])\n"
         "result = (a, b)"),
    ]
    return catalog(rng, items, setup, f"{t.desc}. `X` is `df` standardized.", tol=1e-6)


@template(M, "medium")
def imbalanced_weights(rng):
    n, d, s = rng.randint(200, 400), rng.randint(4, 8), seeded(rng)
    w = rng.choice([0.05, 0.1, 0.15])
    setup = (f"from sklearn.datasets import make_classification\nX, y = make_classification(n_samples={n}, n_features={d}, "
             f"n_informative=3, n_redundant=0, weights=[{1 - w}, {w}], flip_y=0, random_state={s})")
    task = rng.choice(["recall", "weights", "custom"])
    if task == "recall":
        return P(f"`X` ({n} × {d}) and `y` come from `make_classification` with only ~{w:.0%} of rows in class 1. "
                 f"Return the training **recall** of class 1 (the fraction of true 1s predicted as 1) for `{LR}` and "
                 f"for `LogisticRegression(class_weight='balanced', max_iter=1000)`, as a tuple.", setup,
                 f"{LM}LogisticRegression\na = {LR}.fit(X, y).predict(X)\n"
                 "b = LogisticRegression(class_weight='balanced', max_iter=1000).fit(X, y).predict(X)\n"
                 "result = (float(a[y == 1].mean()), float(b[y == 1].mean()))", require=["class_weight"])
    if task == "weights":
        return P(f"`X` ({n} × {d}) and `y` come from `make_classification` with only ~{w:.0%} of rows in class 1. "
                 f"`class_weight='balanced'` weighs class c by n_samples / (n_classes × count_c). Compute these weights "
                 f"by hand and return `(by_hand, compute_class_weight('balanced', classes=np.array([0, 1]), y=y))` "
                 f"(`sklearn.utils.class_weight`).", setup,
                 "from sklearn.utils.class_weight import compute_class_weight\n"
                 "by_hand = len(y) / (2 * np.bincount(y))\n"
                 "result = (by_hand, compute_class_weight('balanced', classes=np.array([0, 1]), y=y))",
                 require=["compute_class_weight"], tol=1e-9)
    k = rng.choice([3, 5, 10])
    return P(f"`X` ({n} × {d}) and `y` come from `make_classification` with only ~{w:.0%} of rows in class 1. Weights "
             f"can be any dict: fit `LogisticRegression(class_weight={{0: 1, 1: {k}}}, max_iter=1000)` on `X, y` and "
             f"return `(predicted positives, true positives)` on the training rows.", setup,
             f"{LM}LogisticRegression\np = LogisticRegression(class_weight={{0: 1, 1: {k}}}, max_iter=1000).fit(X, y).predict(X)\n"
             "result = (int(p.sum()), int((p[y == 1] == 1).sum()))", require=["class_weight"])


@template(M, "medium")
def regularization_path(rng):
    setup, desc, d, n = _reg(rng, d=(5, 8))
    task = rng.choice(["lasso_path", "first_zero", "ridge_trace", "order"])
    alphas = "np.logspace(-1, 2.5, 25)"
    if task == "lasso_path":
        return P(f"{desc}. `lasso_path` (from `sklearn.linear_model`) computes the whole regularization path at once. "
                 f"Call `alphas_, coefs, _ = lasso_path(X - X.mean(axis=0), y - y.mean(), alphas={alphas})` and "
                 f"return the number of non-zero coefficients at each alpha (in the order of `alphas_`).", setup,
                 f"{LM}lasso_path\nalphas_, coefs, _ = lasso_path(X - X.mean(axis=0), y - y.mean(), alphas={alphas})\n"
                 "result = (coefs != 0).sum(axis=0)", require=["lasso_path"])
    if task == "first_zero":
        return P(f"{desc}. For each feature, find the **smallest** alpha in `{alphas}` at which `Lasso(alpha=alpha)` "
                 f"fitted on `X, y` sets its coefficient to 0. Return the array (np.nan if it never reaches 0).", setup,
                 f"{LM}Lasso\nA = {alphas}\nC = np.array([Lasso(alpha=a).fit(X, y).coef_ for a in A])\n"
                 "result = np.array([A[np.flatnonzero(C[:, j] == 0)[0]] if (C[:, j] == 0).any() else np.nan "
                 "for j in range(C.shape[1])])", require=["Lasso"], tol=1e-9)
    if task == "ridge_trace":
        return P(f"{desc}. Ridge never zeroes a weight, it shrinks all of them. For `alpha` in `[0.1, 10, 1000, "
                 f"100000]`, fit `Ridge(alpha=alpha)` on `X, y` and return the 4 × {d} array of coefficients.", setup,
                 f"{LM}Ridge\nresult = np.array([Ridge(alpha=a).fit(X, y).coef_ for a in [0.1, 10, 1000, 100000]])",
                 require=["Ridge"], tol=1e-6)
    return P(f"{desc}. In which order do features leave the Lasso model as the penalty grows? Using `Lasso(alpha=a)` "
             f"for `a` in `{alphas}` (fitted on `X, y`), return the feature indices sorted by the first alpha at which "
             f"they become 0 (features that never reach 0 last; ties by index).", setup,
             f"{LM}Lasso\nA = {alphas}\nC = np.array([Lasso(alpha=a).fit(X, y).coef_ for a in A])\n"
             "first = [int(np.flatnonzero(C[:, j] == 0)[0]) if (C[:, j] == 0).any() else len(A) for j in range(C.shape[1])]\n"
             "result = sorted(range(C.shape[1]), key=lambda j: (first[j], j))", require=["Lasso"])


@template(M, "medium")
def count_regression(rng):
    t = table(rng, kind="reg", numeric_only=True, only=("bikes",), n=(150, 300))
    m = rng.randint(t.n // 2, 3 * t.n // 4)
    setup = (t.setup + "\nfrom sklearn.preprocessing import StandardScaler\n"
             f"sc = StandardScaler().fit(df.iloc[:{m}])\nX_train, X_test = sc.transform(df.iloc[:{m}]), sc.transform(df.iloc[{m}:])\n"
             f"y_train, y_test = y[:{m}], y[{m}:]")
    task = rng.choice(["deviance", "alpha", "mae"])
    head = f"{t.desc} (a count). The data was split into standardized `X_train, X_test` and `y_train, y_test`."
    if task == "deviance":
        return P(f"{head} Return the test `mean_poisson_deviance` (from `sklearn.metrics`) of `PoissonRegressor()` and "
                 f"of `LinearRegression()` (clip its predictions to at least 0.1 first: the deviance needs positive "
                 f"predictions), as a tuple `(poisson, linear)`.", setup,
                 f"{LM}PoissonRegressor, LinearRegression\nfrom sklearn.metrics import mean_poisson_deviance\n"
                 "p = PoissonRegressor().fit(X_train, y_train).predict(X_test)\n"
                 "l = np.clip(LinearRegression().fit(X_train, y_train).predict(X_test), 0.1, None)\n"
                 "result = (mean_poisson_deviance(y_test, p), mean_poisson_deviance(y_test, l))",
                 require=["PoissonRegressor", "mean_poisson_deviance"], tol=1e-6)
    if task == "alpha":
        return P(f"{head} `PoissonRegressor`'s `alpha` is an L2 penalty (default 1). For `alpha` in 0, 0.1, 1, 10, "
                 f"return the test `mean_poisson_deviance` as a list.", setup,
                 f"{LM}PoissonRegressor\nfrom sklearn.metrics import mean_poisson_deviance\n"
                 "result = [mean_poisson_deviance(y_test, PoissonRegressor(alpha=a).fit(X_train, y_train).predict(X_test)) "
                 "for a in (0, 0.1, 1, 10)]", require=["PoissonRegressor"], tol=1e-5)
    return P(f"{head} Return the test mean absolute errors `(poisson, linear)` of `PoissonRegressor(alpha=0.1)` and "
             f"`LinearRegression()`.", setup,
             f"{LM}PoissonRegressor, LinearRegression\n"
             "p = PoissonRegressor(alpha=0.1).fit(X_train, y_train).predict(X_test)\n"
             "l = LinearRegression().fit(X_train, y_train).predict(X_test)\n"
             "result = (float(np.abs(y_test - p).mean()), float(np.abs(y_test - l).mean()))",
             require=["PoissonRegressor"], tol=1e-5)


@template(M, "medium")
def sgd_options(rng):
    setup, desc, d = _bin(rng)
    task = rng.choice(["early", "l1", "constant", "averaged"])
    if task == "early":
        return P(f"{desc}. Fit `SGDClassifier(loss='log_loss', early_stopping=True, validation_fraction=0.2, "
                 f"random_state=0)` on `X, y` (it holds out 20% of the training rows and stops when their score stops "
                 f"improving) and return `(n_iter_, training accuracy)`.", setup,
                 f"{LM}SGDClassifier\nm = SGDClassifier(loss='log_loss', early_stopping=True, validation_fraction=0.2, "
                 "random_state=0).fit(X, y)\nresult = (m.n_iter_, m.score(X, y))", require=["early_stopping"], tol=1e-9)
    if task == "l1":
        a = rng.choice([0.01, 0.05, 0.1])
        return P(f"{desc}. Return the number of non-zero coefficients of `SGDClassifier(loss='log_loss', "
                 f"penalty='l1', alpha={a}, random_state=0)` and of the same with `penalty='l2'`, as a tuple.", setup,
                 f"{LM}SGDClassifier\n"
                 f"a = int((SGDClassifier(loss='log_loss', penalty='l1', alpha={a}, random_state=0).fit(X, y).coef_ != 0).sum())\n"
                 f"b = int((SGDClassifier(loss='log_loss', penalty='l2', alpha={a}, random_state=0).fit(X, y).coef_ != 0).sum())\n"
                 "result = (a, b)", require=["SGDClassifier", "penalty"])
    if task == "constant":
        etas = [0.0001, 0.01, 1.0]
        return P(f"{desc}. With `learning_rate='constant'` the step size is `eta0`. For `eta0` in `{etas}`, fit "
                 f"`SGDClassifier(loss='log_loss', learning_rate='constant', eta0=eta0, max_iter=20, tol=None, "
                 f"random_state=0)` on `X, y` and return the training accuracies (a list).", setup,
                 f"{LM}SGDClassifier\nresult = [SGDClassifier(loss='log_loss', learning_rate='constant', eta0=e, "
                 f"max_iter=20, tol=None, random_state=0).fit(X, y).score(X, y) for e in {etas}]",
                 require=["learning_rate", "eta0"])
    return P(f"{desc}. Averaging the weights over the updates smooths SGD. Return the training accuracy of "
             f"`SGDClassifier(loss='hinge', random_state=0)` and of the same with `average=True`, as a tuple.", setup,
             f"{LM}SGDClassifier\nresult = (SGDClassifier(loss='hinge', random_state=0).fit(X, y).score(X, y), "
             "SGDClassifier(loss='hinge', average=True, random_state=0).fit(X, y).score(X, y))", require=["average"])


@template(M, "medium")
def cv_estimators(rng):
    setup, desc, d, n = _reg(rng)
    task = rng.choice(["ridgecv_scores", "lassocv_path", "bayes", "best_score"])
    if task == "ridgecv_scores":
        return P(f"{desc}. Fit `RidgeCV(alphas=[0.1, 1, 10, 100], store_cv_results=True)` on `X, y`. `cv_results_` holds "
                 f"each row's leave-one-out squared error for every alpha. Return its column means (one per alpha) and "
                 f"`alpha_` as a tuple: `alpha_` has the smallest mean.", setup,
                 f"{LM}RidgeCV\nr = RidgeCV(alphas=[0.1, 1, 10, 100], store_cv_results=True).fit(X, y)\n"
                 "result = (r.cv_results_.mean(axis=0), r.alpha_)", require=["RidgeCV", "cv_results_"], tol=1e-6)
    if task == "lassocv_path":
        return P(f"{desc}. Fit `LassoCV(cv=5, random_state=0)` on `X, y`. `mse_path_` has shape (n_alphas, n_folds). "
                 f"Return `(mse_path_.shape, alpha at the smallest mean MSE, alpha_)`: the last two agree.", setup,
                 f"{LM}LassoCV\nm = LassoCV(cv=5, random_state=0).fit(X, y)\n"
                 "result = (m.mse_path_.shape, m.alphas_[m.mse_path_.mean(axis=1).argmin()], m.alpha_)",
                 require=["LassoCV", "mse_path_"], tol=1e-9)
    if task == "bayes":
        return P(f"{desc}. `BayesianRidge` learns the noise precision `alpha_` and the weight precision `lambda_` from "
                 f"the data. Fit it on `X, y` and return `(1 / alpha_, the variance of the training residuals)`: the "
                 f"learned noise variance is close to the residual variance.", setup,
                 f"{LM}BayesianRidge\nm = BayesianRidge().fit(X, y)\nresult = (1 / m.alpha_, float(np.var(y - m.predict(X))))",
                 require=["BayesianRidge", "alpha_"], tol=1e-6)
    return P(f"{desc}. Fit `RidgeCV(alphas=np.logspace(-3, 3, 13))` on `X, y` and return `(alpha_, best_score_)` "
             f"(`best_score_` is the negative mean squared leave-one-out error of the best alpha).", setup,
             f"{LM}RidgeCV\nr = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(X, y)\nresult = (r.alpha_, r.best_score_)",
             require=["RidgeCV", "best_score_"], tol=1e-6)


@template(M, "medium")
def robust_tasks(rng):
    setup, desc, d, n = _reg(rng, noise=(2, 8))
    k, s = rng.randint(5, 15), seeded(rng)
    setup += f"\nbad = np.random.default_rng({s}).choice(len(y), {k}, replace=False)\ny[bad] += 300   # outliers"
    eps = sorted(rng.sample([1.1, 1.35, 2.0, 3.0], 3))
    items = [
        ("epsilon", f"`epsilon` sets where Huber switches from squared to absolute loss (smaller = more robust). For "
                    f"`epsilon` in `{eps}`, return the number of rows flagged in `outliers_` by `HuberRegressor("
                    f"epsilon=e)` fitted on `X, y`.",
         f"{LM}HuberRegressor\nresult = [int(HuberRegressor(epsilon=e).fit(X, y).outliers_.sum()) for e in {eps}]"),
        ("QuantileRegressor", "The median is robust too: return the distance to `true_coef` of `QuantileRegressor("
                              "quantile=0.5, alpha=0.0)` and of `LinearRegression`, both fitted on `X, y`, as a tuple.",
         f"{LM}QuantileRegressor, LinearRegression\n"
         "a = np.linalg.norm(QuantileRegressor(quantile=0.5, alpha=0.0).fit(X, y).coef_ - true_coef)\n"
         "b = np.linalg.norm(LinearRegression().fit(X, y).coef_ - true_coef)\nresult = (a, b)"),
        ("scale_", "`HuberRegressor` also estimates the noise scale `scale_`. Fit it on `X, y` and return `(scale_, "
                   "median absolute residual)`.",
         f"{LM}HuberRegressor\nm = HuberRegressor().fit(X, y)\nresult = (m.scale_, float(np.median(np.abs(y - m.predict(X)))))"),
        ("RANSACRegressor", "`RANSACRegressor(random_state=0)` (from `sklearn.linear_model`) fits on random subsets and "
                            "keeps the consensus inliers. Return `(number of outliers it rejects, how many of them are in "
                            "bad)` (`inlier_mask_`).",
         f"{LM}RANSACRegressor\nm = RANSACRegressor(random_state=0).fit(X, y)\nout = ~m.inlier_mask_\n"
         "result = (int(out.sum()), int(out[bad].sum()))"),
        ("HuberRegressor", "Refit trick: fit `HuberRegressor()` on `X, y`, drop the rows it flags in `outliers_`, refit "
                           "`LinearRegression` on the rest and return its coefficients.",
         f"{LM}HuberRegressor, LinearRegression\nkeep = ~HuberRegressor().fit(X, y).outliers_\n"
         "result = LinearRegression().fit(X[keep], y[keep]).coef_"),
    ]
    return catalog(rng, items, setup, f"{desc}; then {k} targets (indices `bad`) were pushed up by 300.", tol=1e-4)


# ================================================================= hard

@template(M, "hard")
def lasso_coordinate_descent(rng):
    setup, desc, d, n = _reg(rng, n=(60, 150), d=(3, 6))
    a = rng.choice([0.5, 2.0, 5.0, 10.0])
    sweeps = rng.choice([200, 300])
    task = rng.choice(["coef", "intercept"])
    sol = ("Xc, yc = X - X.mean(axis=0), y - y.mean()\nn, p = Xc.shape\nw = np.zeros(p)\nsq = (Xc ** 2).sum(axis=0) / n\n"
           f"for _ in range({sweeps}):\n    for j in range(p):\n        r = yc - Xc @ w + Xc[:, j] * w[j]\n"
           f"        rho = Xc[:, j] @ r / n\n        w[j] = np.sign(rho) * max(abs(rho) - {a}, 0) / sq[j]\n")
    head = (f"{desc}. Implement Lasso yourself with cyclic **coordinate descent**: center `X` and `y` (for the "
            f"intercept), start from w = 0, and for {sweeps} sweeps update each coordinate j in turn with "
            f"ρ = Xcⱼ · (residual without feature j) / n and w_j = sign(ρ) · max(|ρ| − α, 0) / (Xcⱼ · Xcⱼ / n), with "
            f"α = {a} (the objective is ‖y − Xw‖² / (2n) + α‖w‖₁).")
    if task == "coef":
        return P(f"{head} Return w (it matches `Lasso(alpha={a}).fit(X, y).coef_`).", setup, sol + "result = w",
                 require=["sign"], tol=1e-4)
    return P(f"{head} Return `(w, intercept)` with intercept = mean(y) − mean(X) · w.", setup,
             sol + "result = (w, y.mean() - X.mean(axis=0) @ w)", require=["sign"], tol=1e-4)


@template(M, "hard")
def logistic_optimality(rng):
    setup, desc, d = _bin(rng)
    c = rng.choice([0.1, 0.5, 1.0, 2.0])
    task = rng.choice(["coef", "intercept"])
    head = (f"{desc}. `LogisticRegression(C={c}, tol=1e-10, max_iter=10000)` minimizes ½‖w‖² + C · Σᵢ log-loss(i) "
            f"(the intercept is not penalized). At the optimum the gradient vanishes: w = C · Xᵀ(y − p) and Σ(y − p) = "
            f"0, where p are the predicted probabilities of class 1.")
    sol = (f"{LM}LogisticRegression\nclf = LogisticRegression(C={c}, tol=1e-10, max_iter=10000).fit(X, y)\n"
           "p = clf.predict_proba(X)[:, 1]\n")
    if task == "coef":
        return P(f"{head} Fit it on `X, y` and rebuild `coef_[0]` from `predict_proba` alone with the first condition.",
                 setup, sol + f"result = {c} * X.T @ (y - p)", require=["predict_proba"], tol=1e-4)
    return P(f"{head} Fit it on `X, y` and return `(sum of y − p, mean of p, mean of y)`: the model's average "
             f"predicted probability matches the class-1 rate.", setup,
             sol + "result = (float((y - p).sum()), float(p.mean()), float(y.mean()))", require=["predict_proba"],
             tol=1e-4)


@template(M, "hard")
def lasso_kkt(rng):
    setup, desc, d, n = _reg(rng, d=(5, 9))
    a = rng.choice([1.0, 5.0, 10.0, 20.0])
    task = rng.choice(["c", "check"])
    sol = (f"{LM}Lasso\nm = Lasso(alpha={a}, tol=1e-10, max_iter=100000).fit(X, y)\nr = y - m.predict(X)\n"
           f"c = X.T @ r / (len(y) * {a})\n")
    head = (f"{desc}. Fit `Lasso(alpha={a}, tol=1e-10, max_iter=100000)` on `X, y`. Its optimality (KKT) conditions "
            f"say: with residuals r = y − predict(X) and c_j = X_jᵀ r / (n · α), c_j = sign(w_j) for every non-zero "
            f"weight, and |c_j| ≤ 1 for every zero weight.")
    if task == "c":
        return P(f"{head} Return the array c.", setup, sol + "result = c", require=["Lasso"], tol=1e-6)
    return P(f"{head} Check them: return `(active_ok, inactive_ok)`: whether `np.allclose(c[w != 0], np.sign(w[w != 0]), "
             f"atol=1e-4)` and whether `(np.abs(c[w == 0]) <= 1 + 1e-4).all()`.", setup,
             sol + "w = m.coef_\nresult = (bool(np.allclose(c[w != 0], np.sign(w[w != 0]), atol=1e-4)), "
                   "bool((np.abs(c[w == 0]) <= 1 + 1e-4).all()))", require=["Lasso"])


@template(M, "hard")
def collinear_stability(rng):
    n, s = rng.randint(40, 100), seeded(rng)
    a = rng.choice([1.0, 10.0])
    setup = (f"g = np.random.default_rng({s})\nx = g.normal(size={n})\n"
             f"X = np.column_stack([x, x + 1e-3 * g.normal(size={n}), g.normal(size={n})])   # columns 0 and 1 nearly equal\n"
             f"y = 3 * x + 2 * X[:, 2] + g.normal(0, 0.5, {n})")
    task = rng.choice(["coef", "bootstrap"])
    if task == "coef":
        return P(f"Columns 0 and 1 of `X` are almost identical and `y` = 3·x + 2·X[:, 2] + noise. Return the coefficients "
                 f"of `LinearRegression` and of `Ridge(alpha={a})`, both fitted on `X, y`, as a tuple: least squares "
                 f"splits the weight 3 between the twins wildly; ridge shares it evenly.", setup,
                 f"{LM}LinearRegression, Ridge\nresult = (LinearRegression().fit(X, y).coef_, Ridge(alpha={a}).fit(X, y).coef_)",
                 require=["Ridge"], tol=1e-4)
    b = rng.randint(5, 10)
    return P(f"Columns 0 and 1 of `X` are almost identical and `y` = 3·x + 2·X[:, 2] + noise. Measure stability: with "
             f"`rs = np.random.default_rng(0)`, draw {b} bootstrap samples (`idx = rs.integers(0, len(X), len(X))`) and "
             f"fit `LinearRegression` and `Ridge(alpha={a})` on each. Return the standard deviation of `coef_[0]` across "
             f"the samples for both, as a tuple `(ols, ridge)`.", setup,
             f"{LM}LinearRegression, Ridge\nrs = np.random.default_rng(0)\na, b = [], []\nfor _ in range({b}):\n"
             "    idx = rs.integers(0, len(X), len(X))\n    a.append(LinearRegression().fit(X[idx], y[idx]).coef_[0])\n"
             f"    b.append(Ridge(alpha={a}).fit(X[idx], y[idx]).coef_[0])\nresult = (float(np.std(a)), float(np.std(b)))",
             require=["Ridge"], tol=1e-4)


@template(M, "hard")
def progressive_validation(rng):
    setup, desc, d = _bin(rng)
    b = rng.choice([25, 50])
    task = rng.choice(["list", "mean", "compare"])
    loop = (f"{LM}SGDClassifier\nsgd = SGDClassifier(loss='log_loss', random_state=0)\naccs = []\n"
            f"for i in range(0, len(X), {b}):\n    Xb, yb = X[i:i + {b}], y[i:i + {b}]\n"
            "    if i > 0:\n        accs.append(sgd.score(Xb, yb))\n"
            "    sgd.partial_fit(Xb, yb, classes=np.array([0, 1]))\n")
    head = (f"{desc}. Evaluate a streaming model honestly with *progressive validation*: process batches of {b} rows "
            f"in order; score `sgd = SGDClassifier(loss='log_loss', random_state=0)` on each batch **before** training "
            f"on it with `partial_fit(..., classes=np.array([0, 1]))` (skip the score for the first batch).")
    if task == "list":
        return P(f"{head} Return the list of batch accuracies.", setup, loop + "result = accs", require=["partial_fit"])
    if task == "mean":
        return P(f"{head} Return `(mean progressive accuracy, final training accuracy on X, y)`.", setup,
                 loop + "result = (float(np.mean(accs)), sgd.score(X, y))", require=["partial_fit"], tol=1e-9)
    return P(f"{head} Then fit a fresh `SGDClassifier(loss='log_loss', random_state=0)` on all of `X, y` with `fit`. "
             f"Return `(mean progressive accuracy, fit's training accuracy)`: the second is optimistic.", setup,
             loop + "full = SGDClassifier(loss='log_loss', random_state=0).fit(X, y).score(X, y)\n"
                    "result = (float(np.mean(accs)), full)", require=["partial_fit"], tol=1e-9)


@template(M, "hard")
def losses_by_hand(rng):
    task = rng.choice(["poisson", "pinball", "log_loss"])
    if task == "poisson":
        t = table(rng, kind="reg", numeric_only=True, only=("bikes",), n=(100, 250))
        setup = t.setup + "\nfrom sklearn.preprocessing import StandardScaler\nX = StandardScaler().fit_transform(df)"
        return P(f"{t.desc} (a count); `X` is `df` standardized. Fit `PoissonRegressor()` on `X, y` and compute the mean "
                 f"Poisson deviance of its training predictions μ **by hand**: 2 · mean(y · log(y / μ) − (y − μ)), "
                 f"where y · log(y / μ) is 0 when y = 0. Return `(by_hand, mean_poisson_deviance(y, μ))`.", setup,
                 f"{LM}PoissonRegressor\nfrom sklearn.metrics import mean_poisson_deviance\n"
                 "mu = PoissonRegressor().fit(X, y).predict(X)\nterm = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0.0)\n"
                 "result = (2 * np.mean(term - (y - mu)), mean_poisson_deviance(y, mu))",
                 require=["PoissonRegressor", "mean_poisson_deviance"], tol=1e-6)
    setup, desc, d, n = _reg(rng, noise=(10, 30))
    if task == "pinball":
        q = rng.choice([0.1, 0.25, 0.75, 0.9])
        return P(f"{desc}. `QuantileRegressor` minimizes the pinball loss: mean of q·(y − ŷ) when y ≥ ŷ and (1 − q)·(ŷ − "
                 f"y) otherwise. Fit `QuantileRegressor(quantile={q}, alpha=0.0)` on `X, y` and return `(by_hand, "
                 f"mean_pinball_loss(y, pred, alpha={q}))`.", setup,
                 f"{LM}QuantileRegressor\nfrom sklearn.metrics import mean_pinball_loss\n"
                 f"pred = QuantileRegressor(quantile={q}, alpha=0.0).fit(X, y).predict(X)\ndiff = y - pred\n"
                 f"result = (float(np.mean(np.where(diff >= 0, {q} * diff, ({q} - 1) * diff))), mean_pinball_loss(y, pred, alpha={q}))",
                 require=["QuantileRegressor", "mean_pinball_loss"], tol=1e-6)
    setup, desc, d = _bin(rng)
    return P(f"{desc}. Fit `{LR}` on `X, y` and compute its training log loss by hand: −mean(y·log p + (1 − y)·log(1 − p)) "
             f"with p = P(class 1). Return `(by_hand, log_loss(y, predict_proba(X)))` (`log_loss` from `sklearn.metrics`).",
             setup, f"{LM}LogisticRegression\nfrom sklearn.metrics import log_loss\nP_ = {LR}.fit(X, y).predict_proba(X)\n"
                    "p = P_[:, 1]\nresult = (float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), log_loss(y, P_))",
             require=["log_loss", "predict_proba"], tol=1e-6)


@template(M, "hard")
def penalty_needs_scaling(rng):
    setup, desc, d, n = _reg(rng, d=(4, 7), noise=(5, 15))
    j, f = rng.randrange(d), rng.choice([0.001, 0.01])
    a = rng.choice([1.0, 3.0, 10.0])
    setup += f"\nX[:, {j}] *= {f}   # this feature is now measured in tiny units"
    task = rng.choice(["sets", "coef"])
    if task == "sets":
        return P(f"{desc}; then column {j} was multiplied by {f} (a unit change). Penalties compare raw weight sizes, so "
                 f"a feature in tiny units needs a huge weight and gets punished. Return the indices of the non-zero "
                 f"coefficients of `Lasso(alpha={a})` fitted on raw `X` and of `make_pipeline(StandardScaler(), "
                 f"Lasso(alpha={a}))`, as a tuple of two arrays.", setup,
                 f"{LM}Lasso\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\n"
                 f"raw = np.flatnonzero(Lasso(alpha={a}).fit(X, y).coef_)\n"
                 f"sc = np.flatnonzero(make_pipeline(StandardScaler(), Lasso(alpha={a})).fit(X, y)[-1].coef_)\nresult = (raw, sc)",
                 require=["Lasso", "StandardScaler"])
    return P(f"{desc}; then column {j} was multiplied by {f} (a unit change). Return the coefficient of column {j} "
             f"(in **original units**) for `LinearRegression` and `Ridge(alpha={a})` fitted on raw `X`, and for "
             f"`Ridge(alpha={a})` fitted on standardized `X` (divide its coefficient by the scaler's `scale_[{j}]` to get "
             f"back to original units), as a tuple. Plain least squares doesn't care about units; the raw ridge does.",
             setup, f"{LM}LinearRegression, Ridge\nfrom sklearn.preprocessing import StandardScaler\n"
                    f"sc = StandardScaler().fit(X)\nr = Ridge(alpha={a}).fit(sc.transform(X), y)\n"
                    f"result = (LinearRegression().fit(X, y).coef_[{j}], Ridge(alpha={a}).fit(X, y).coef_[{j}], "
                    f"r.coef_[{j}] / sc.scale_[{j}])", require=["Ridge", "scale_"], tol=1e-5)


@template(M, "hard")
def broken_linear(rng):
    t = table(rng, kind="clf", numeric_only=True, n=(120, 250))
    a = rng.choice([0.05, 0.2, 1.0])
    items = [
        ("StandardScaler", f"This fit stops early with a `ConvergenceWarning` (the raw columns have very different "
                           f"scales):\n```\nLogisticRegression(max_iter=100).fit(df, y)\n```\nFix it by standardizing "
                           f"inside a pipeline, `make_pipeline(StandardScaler(), LogisticRegression(max_iter=100))`, and "
                           f"return its training accuracy.",
         f"{LM}LogisticRegression\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\n"
         "result = make_pipeline(StandardScaler(), LogisticRegression(max_iter=100)).fit(df, y).score(df, y)"),
        ("l1_ratio", f"In scikit-learn 1.8 this raises a `FutureWarning` (`penalty` is deprecated):\n```\n"
                     f"LogisticRegression(penalty='l1', C={a}, solver='liblinear', random_state=0)\n```\nWrite it the 1.8 way, fit it "
                     f"on standardized `df` and return the number of non-zero coefficients.",
         f"{LM}LogisticRegression\nfrom sklearn.preprocessing import StandardScaler\n"
         f"clf = LogisticRegression(l1_ratio=1.0, C={a}, solver='liblinear', random_state=0).fit(StandardScaler().fit_transform(df), y)\n"
         "result = int((clf.coef_ != 0).sum())"),
        ("classes", "The first `partial_fit` call must list every class (the first batch may miss one), so this "
                    "raises:\n```\nsgd = SGDClassifier(random_state=0)\nsgd.partial_fit(X[:40], y[:40])\n```\nFix the "
                    "loop over batches of 40 standardized rows (`X = StandardScaler().fit_transform(df)`) and return "
                    "the final training accuracy.",
         f"{LM}SGDClassifier\nfrom sklearn.preprocessing import StandardScaler\nX = StandardScaler().fit_transform(df)\n"
         "sgd = SGDClassifier(random_state=0)\nfor i in range(0, len(X), 40):\n"
         "    sgd.partial_fit(X[i:i + 40], y[i:i + 40], classes=np.unique(y))\nresult = sgd.score(X, y)",
         {"require": ["partial_fit", "classes"]}),
        ("reshape", f"This raises `ValueError: Expected 2D array` (one feature must still be a column):\n```\n"
                    f"LinearRegression().fit(df[{t.num[0]!r}], y)\n```\nFix it and return the slope and intercept as a "
                    f"tuple.",
         f"{LM}LinearRegression\nm = LinearRegression().fit(df[[{t.num[0]!r}]], y)\nresult = (m.coef_[0], m.intercept_)",
         {"require": ["LinearRegression"]}),
        ("LinearRegression", "This target is 0/1, and least squares predicts numbers outside [0, 1]:\n```\n"
                             "LinearRegression().fit(df, y).predict(df)\n```\nUse `LogisticRegression` (standardize "
                             "`df` in a pipeline) and return `(how many linear predictions fall outside [0, 1], min and "
                             "max logistic P(class 1))`.",
         f"{LM}LinearRegression, LogisticRegression\nfrom sklearn.pipeline import make_pipeline\n"
         "from sklearn.preprocessing import StandardScaler\nlin = LinearRegression().fit(df, y).predict(df)\n"
         "p = make_pipeline(StandardScaler(), LogisticRegression()).fit(df, y).predict_proba(df)[:, 1]\n"
         "result = (int(((lin < 0) | (lin > 1)).sum()), float(p.min()), float(p.max()))"),
        ("SGDRegressor", "`SGDRegressor` on raw columns of very different scales blows up (huge or nan weights). Use "
                         "`make_pipeline(StandardScaler(), SGDRegressor(random_state=0))` to predict the first numeric "
                         f"column `{t.num[0]}` from the other columns and return the training R².",
         f"{LM}SGDRegressor\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\n"
         f"Xo, yo = df.drop(columns={t.num[0]!r}), df[{t.num[0]!r}]\n"
         "result = make_pipeline(StandardScaler(), SGDRegressor(random_state=0)).fit(Xo, yo).score(Xo, yo)"),
    ]
    return catalog(rng, items, t.setup, f"{t.desc}.", tol=1e-6)


@template(M, "hard")
def interpret_logistic(rng):
    t = table(rng, kind="clf", numeric_only=True, n=(150, 300))
    col = rng.choice(t.num)
    j = t.num.index(col)
    task = rng.choice(["odds", "delta", "units"])
    setup = t.setup + "\nfrom sklearn.preprocessing import StandardScaler\nsc = StandardScaler().fit(df)\nX = sc.transform(df)"
    fit = f"{LM}LogisticRegression\nclf = {LR}.fit(X, y)\n"
    if task == "odds":
        return P(f"{t.desc}. `X` is `df` standardized (`sc`). Fit `{LR}` on `X, y`. A coefficient is a change in "
                 f"log-odds per standard deviation, so `exp(coef)` is an odds ratio. Return a dict {{column: odds "
                 f"ratio}}.", setup, fit + "result = dict(zip(df.columns, np.exp(clf.coef_[0])))", require=["coef_", "exp"],
                 tol=1e-6)
    if task == "delta":
        return P(f"{t.desc}. `X` is `df` standardized. Fit `{LR}` on `X, y`. Take the average row (all zeros after "
                 f"standardizing) and return P(class 1) there and after raising `{col}` by one standard deviation, as a "
                 f"tuple.", setup,
                 fit + f"z = np.zeros((1, X.shape[1]))\nz2 = z.copy()\nz2[0, {j}] = 1\n"
                       "result = (clf.predict_proba(z)[0, 1], clf.predict_proba(z2)[0, 1])", require=["predict_proba"],
                 tol=1e-6)
    return P(f"{t.desc}. `X` is `df` standardized with `sc`. Fit `{LR}` on `X, y` and express the effect of `{col}` in "
             f"its **original units**: return the odds ratio for a +1 unit increase, `exp(coef_[0][{j}] / "
             f"sc.scale_[{j}])`, and for a +10 unit increase.", setup,
             fit + f"w = clf.coef_[0][{j}] / sc.scale_[{j}]\nresult = (float(np.exp(w)), float(np.exp(10 * w)))",
             require=["scale_"], tol=1e-6)


@template(M, "hard")
def polynomial_degree_choice(rng):
    n, s = rng.randint(80, 160), seeded(rng)
    setup = (f"g = np.random.default_rng({s})\nx = np.sort(g.uniform(-3, 3, {n}))\n"
             f"y = np.sin(x) * 3 + 0.5 * x + g.normal(0, 0.4, {n})\nX = x.reshape(-1, 1)\nval = np.arange({n}) % 4 == 0")
    a = rng.choice([1e-3, 0.1, 1.0])
    task = rng.choice(["best", "scores"])
    sol = ("from sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import PolynomialFeatures, StandardScaler\n"
           f"{LM}Ridge\nscores = {{}}\nfor deg in range(1, 10):\n"
           f"    m = make_pipeline(PolynomialFeatures(deg), StandardScaler(), Ridge(alpha={a})).fit(X[~val], y[~val])\n"
           "    scores[deg] = m.score(X[val], y[val])\n")
    head = (f"`y` is a noisy curve of `x` (`X` is it as a column). Linear models fit curves through feature expansion. "
            f"For degree 1 … 9, fit `make_pipeline(PolynomialFeatures(degree), StandardScaler(), Ridge(alpha={a}))` on the "
            f"rows where `val` is False and score R² on the rows where it is True.")
    if task == "best":
        return P(f"{head} Return the best degree (the smallest on ties).", setup,
                 sol + "result = max(scores, key=scores.get)", require=["PolynomialFeatures", "Ridge"])
    return P(f"{head} Return the dict {{degree: validation R²}}.", setup, sol + "result = scores",
             require=["PolynomialFeatures", "Ridge"], tol=1e-6)


@template(M, "hard")
def linear_model_workflows(rng):
    setup, desc, d, n = _reg(rng, n=(150, 300))
    s = seeded(rng)
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
    items = [
        ("RidgeCV", f"Pick Ridge's alpha two ways: on a validation split (fit `Ridge(alpha=a)` on the first 70% of rows, "
                    f"`int(0.7 * len(X))`, score R² on the rest; best of `{alphas}`, first on ties) and with "
                    f"`RidgeCV(alphas={alphas})` on all rows. Return both as a tuple.",
         f"{LM}Ridge, RidgeCV\nh = int(0.7 * len(X))\n"
         f"sc = [Ridge(alpha=a).fit(X[:h], y[:h]).score(X[h:], y[h:]) for a in {alphas}]\n"
         f"result = ({alphas}[int(np.argmax(sc))], RidgeCV(alphas={alphas}).fit(X, y).alpha_)"),
        ("ElasticNetCV", "Let CV choose both knobs: fit `ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=5, random_state=0)` "
                         "on `X, y` and return `(l1_ratio_, alpha_)`.",
         f"{LM}ElasticNetCV\nm = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=5, random_state=0).fit(X, y)\n"
         "result = (m.l1_ratio_, m.alpha_)"),
        ("warm_start", "Walk down a Lasso path with `warm_start=True`: one `Lasso(warm_start=True)` object, `set_params("
                       "alpha=a)` then `fit` for a in `np.logspace(2, -1, 10)` (each fit starts from the previous "
                       "solution). Return the total `n_iter_` with warm starts and with a fresh `Lasso(alpha=a)` every "
                       "time, as a tuple.",
         f"{LM}Lasso\nA = np.logspace(2, -1, 10)\nm = Lasso(warm_start=True)\nwarm = 0\nfor a in A:\n"
         "    warm += m.set_params(alpha=a).fit(X, y).n_iter_\ncold = sum(Lasso(alpha=a).fit(X, y).n_iter_ for a in A)\n"
         "result = (int(warm), int(cold))"),
        ("sample_weight", "Rows are in time order and recent rows should count more: fit `LinearRegression` with "
                          "`sample_weight=np.linspace(0.2, 1.8, len(X))` and return the coefficients minus those of an "
                          "unweighted fit.",
         f"{LM}LinearRegression\nw = np.linspace(0.2, 1.8, len(X))\n"
         "result = LinearRegression().fit(X, y, sample_weight=w).coef_ - LinearRegression().fit(X, y).coef_"),
        ("cross_val_score", f"Compare models honestly: return a dict of mean 5-fold `cross_val_score` R² (shuffled "
                            f"`KFold(5, shuffle=True, random_state={s})`) for `'ols'`: `LinearRegression()`, "
                            f"`'huber'`: `HuberRegressor()` and `'bayes'`: `BayesianRidge()`.",
         f"{LM}LinearRegression, HuberRegressor, BayesianRidge\nfrom sklearn.model_selection import KFold, cross_val_score\n"
         f"cv = KFold(5, shuffle=True, random_state={s})\n"
         "models = {'ols': LinearRegression(), 'huber': HuberRegressor(), 'bayes': BayesianRidge()}\n"
         "result = {k: cross_val_score(m, X, y, cv=cv).mean() for k, m in models.items()}"),
        ("SGDRegressor", "Streaming regression: feed `SGDRegressor(random_state=0)` batches of 50 rows with "
                         "`partial_fit`, 5 passes over the data, and return the training R² after each pass.",
         f"{LM}SGDRegressor\nm = SGDRegressor(random_state=0)\nresult = []\nfor _ in range(5):\n"
         "    for i in range(0, len(X), 50):\n        m.partial_fit(X[i:i + 50], y[i:i + 50])\n"
         "    result.append(m.score(X, y))", {"require": ["partial_fit"]}),
    ]
    return catalog(rng, items, setup, f"{desc}.", tol=1e-6)


# ================================================================= challenge

@template(M, "challenge")
def ridge_loo_shortcut(rng):
    n, d, s = rng.randint(8, 12), rng.randint(2, 3), seeded(rng)
    report = rng.choice(["mse", "alpha", "loo"])
    gen = ("g = np.random.default_rng({s})\nX = g.normal(size=({n}, {d}))\ny = X @ g.normal(0, 2, {d}) + g.normal(0, 1, {n})\n"
           "alphas = np.array({alphas})\nreport = {report!r}")
    alphas = sorted(rng.sample([0.01, 0.1, 0.5, 1.0, 3.0, 10.0, 30.0], 4))
    setup = gen.format(s=s, n=n, d=d, alphas=lit(alphas), report=report)
    tests = [gen.format(s=s + i, n=100000, d=4, alphas="np.logspace(-2, 4, 25).tolist()", report=report) for i in (1, 2)]
    sol = ("n = len(y)\nXc, yc = X - X.mean(axis=0), y - y.mean()\nU, sv, _ = np.linalg.svd(Xc, full_matrices=False)\n"
           "Uy = U.T @ yc\nU2 = U ** 2\nerrs = []\nfor a in alphas:\n    f = sv ** 2 / (sv ** 2 + a)\n"
           "    h = 1 / n + U2 @ f\n    fit = y.mean() + U @ (f * Uy)\n    errs.append((y - fit) / (1 - h))\n"
           "errs = np.array(errs)\nmse = (errs ** 2).mean(axis=1)\n")
    if report == "mse":
        sol += "result = mse"
        what = "the mean squared leave-one-out error for every alpha (a float array, like `cv_results_.mean(axis=0)`)"
    elif report == "alpha":
        sol += "result = float(alphas[np.argmin(mse)])"
        what = "the alpha `RidgeCV` picks: the one with the smallest mean squared leave-one-out error (first on ties)"
    else:
        sol += "result = y - errs[np.argmin(mse)]"
        what = ("the leave-one-out **predictions** ŷ₋ᵢ (row i predicted by the model fitted without row i) for the "
                "best alpha, as a float array")
    return challenge(
        "Leave-one-out for free",
        "`RidgeCV` evaluates every alpha by leave-one-out cross-validation without refitting n models. For ridge with "
        "an intercept, the fitted values are ŷ = H y with the hat matrix H = 11ᵀ/n + Xc (Xcᵀ Xc + αI)⁻¹ Xcᵀ "
        "(Xc = centered `X`), and the leave-one-out residual of row i is (yᵢ − ŷᵢ) / (1 − Hᵢᵢ). With the SVD Xc = U S Vᵀ, "
        "Hᵢᵢ = 1/n + Σⱼ Uᵢⱼ² sⱼ² / (sⱼ² + α). Given `X`, `y` and the candidate `alphas`, return " + what +
        f". (`report` = {report!r}.)", setup, sol, tests,
        "n ≤ 10^5 rows, d ≤ 4 features, ≤ 25 alphas", speed="O(n·d² + n·d·len(alphas)): one SVD, no refits",
        answer_desc=" (see above)", tol=1e-6)


@template(M, "challenge")
def lasso_orthogonal_path(rng):
    d, q, s = rng.randint(6, 10), rng.randint(4, 6), seeded(rng)
    report = rng.choice(["count", "l1", "l2sq"])
    gen = ("g = np.random.default_rng({s})\nb = g.normal(0, 2, {d})   # b = Xᵀy / n for an orthonormal design\n"
           "alphas = g.uniform(0, 4, {q})\nreport = {report!r}")
    setup = gen.format(s=s, d=d, q=q, report=report)
    tests = [gen.format(s=s + i, d=200000, q=200000, report=report) for i in (1, 2)]
    sol = ("a = np.sort(np.abs(b))\nk = np.searchsorted(a, alphas, side='right')   # a[k:] > alpha\n"
           "m = len(a) - k\n")
    if report == "count":
        sol += "result = m"
        what = "the number of non-zero coefficients"
    elif report == "l1":
        sol += "suf = np.concatenate([np.cumsum(a[::-1])[::-1], [0.0]])\nresult = suf[k] - m * alphas"
        what = "the L1 norm ‖w‖₁ of the solution"
    else:
        sol += ("suf = np.concatenate([np.cumsum(a[::-1])[::-1], [0.0]])\n"
                "suf2 = np.concatenate([np.cumsum((a * a)[::-1])[::-1], [0.0]])\n"
                "result = suf2[k] - 2 * alphas * suf[k] + m * alphas ** 2")
        what = "the squared L2 norm ‖w‖₂² of the solution"
    return challenge(
        "A Lasso path in one sort",
        "When the design is orthonormal (Xᵀ X / n = I), `Lasso(alpha, fit_intercept=False)` has a closed form: each "
        "weight is the soft-thresholded correlation, wⱼ = sign(bⱼ) · max(|bⱼ| − α, 0), with b = Xᵀ y / n. You are given "
        "`b` and many query values `alphas`. For every alpha return " + what +
        f" (an array, one value per query). (`report` = {report!r}.)",
        setup, sol, tests, "len(b), number of queries ≤ 2·10^5", speed="O((d + q) log d): sort |b|, prefix sums, binary search",
        answer_desc=" (a NumPy array)", tol=1e-6)


@template(M, "challenge")
def huber_epsilon_queries(rng):
    n, q, s = rng.randint(8, 12), rng.randint(4, 6), seeded(rng)
    report = rng.choice(["outliers", "loss"])
    gen = ("g = np.random.default_rng({s})\nr = g.standard_t(2, {n}) * 3   # residuals y - Xw (heavy tails)\n"
           "sigma = {sigma}\npenalty = {pen}   # alpha * ||w||^2\neps = g.uniform(1.0, 4.0, {q})\nreport = {report!r}")
    sigma, pen = round(rng.uniform(1, 4), 2), round(rng.uniform(0, 2), 2)
    setup = gen.format(s=s, n=n, q=q, sigma=sigma, pen=pen, report=report)
    tests = [gen.format(s=s + i, n=200000, q=200000, sigma=sigma, pen=pen, report=report) for i in (1, 2)]
    sol = ("a = np.sort(np.abs(r))\nk = np.searchsorted(a, eps * sigma, side='right')   # a[k:] are outliers\n"
           "n_out = len(a) - k\n")
    if report == "outliers":
        sol += "result = n_out"
        what = "the number of outliers (|rᵢ| > ε·σ: the rows `outliers_` would flag)"
    else:
        sol += ("pre2 = np.concatenate([[0.0], np.cumsum(a * a)])\npre1 = np.concatenate([[0.0], np.cumsum(a)])\n"
                "result = len(r) * sigma + pre2[k] / sigma + 2 * eps * (pre1[-1] - pre1[k]) - sigma * n_out * eps ** 2 + penalty")
        what = ("the objective value n·σ + Σ_inliers rᵢ²/σ + Σ_outliers (2ε|rᵢ| − σε²) + `penalty` (inliers: |rᵢ| ≤ ε·σ)")
    return challenge(
        "Huber's switch point",
        "`HuberRegressor` treats a residual quadratically while |r| ≤ ε·σ and linearly beyond (σ is its learned scale, "
        "ε its `epsilon`). For fixed residuals `r`, scale `sigma` and penalty term `penalty`, an analyst wants to see "
        "what each candidate `eps[i]` would do. For every query return " + what +
        f". (`report` = {report!r}.)", setup, sol, tests, "len(r), number of queries ≤ 2·10^5",
        speed="O((n + q) log n): sort, prefix sums, binary search", answer_desc=" (a NumPy array, one value per query)",
        tol=1e-6)
