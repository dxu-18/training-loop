"""1.12 Comparisons, logic & missing values."""

from ... import themes as T
from ._common import P, about, arr, challenge, floats, lit, mat, rand_ints, seeded, series, template

M = "1.12"


# ================================================================= lite

@template(M, "lite")
def compare_mask(rng):
    th, m, _, vals = series(rng)
    v = m.var
    t = T.threshold(rng, vals, m)
    op, word = rng.choice([(">", "greater than"), ("<", "less than"), (">=", "at least"), ("<=", "at most")])
    return P(f"{about(th, m, len(vals), v)}. Create a **boolean array** that is `True` where the value "
             f"is {word} {t}.", f"{v} = {arr(vals)}", f"{v} {op} {t}")


@template(M, "lite")
def float_equality(rng):
    n = rng.randint(3, 5)
    a = floats(rng, n, 0, 5, 3)
    eps = rng.choice([1e-9, 1e-12, 1e-4])
    b = [round(x + eps, 12) for x in a]
    desc, setup, sol, req = rng.choice([
        ("whether `a` and `b` are equal **within floating-point tolerance**, as one bool (`np.allclose`; never "
         "use `==` on floats)", f"a = {arr(a)}\nb = {arr(b)}", "np.allclose(a, b)", "allclose"),
        ("the element-wise closeness of `a` and `b` as a boolean array (`np.isclose`)",
         f"a = {arr(a)}\nb = {arr(b)}", "np.isclose(a, b)", "isclose"),
        ("whether `0.1 + 0.2` equals `0.3` up to tolerance (`np.isclose`; `0.1 + 0.2 == 0.3` is False!)",
         "", "np.isclose(0.1 + 0.2, 0.3)", "isclose"),
        ("whether `a` and `c` are exactly equal **treating NaNs as equal** (`np.array_equal(..., equal_nan=True)`)",
         f"a = {arr(a + [float('nan')])}\nc = {arr(a + [float('nan')])}", "np.array_equal(a, c, equal_nan=True)",
         "array_equal"),
    ])
    return P(f"Compare two runs' outputs: compute {desc}.", setup, sol, require=[req])


# ================================================================= easy

@template(M, "easy")
def count_condition(rng):
    th, m, _, vals = series(rng, 7, 12)
    v = m.var
    t = T.threshold(rng, vals, m)
    op, word = rng.choice([(">", "greater than"), ("<", "less than")])
    if rng.random() < 0.5:
        return P(f"{about(th, m, len(vals), v)}. **How many** of them have a value {word} {t}? Count "
                 f"with a boolean mask (each `True` counts as 1).",
                 f"{v} = {arr(vals)}", f"np.sum({v} {op} {t})", no_loops=True)
    return P(f"{about(th, m, len(vals), v)}. What **fraction** of them have a value {word} {t}? "
             f"(Hint: the mean of a boolean array.)", f"{v} = {arr(vals)}", f"np.mean({v} {op} {t})",
             no_loops=True)


@template(M, "easy")
def missing_values(rng):
    th, m, _, vals = series(rng, 7, 11)
    vals = [float(x) for x in vals]
    for i in rng.sample(range(len(vals)), rng.randint(1, 3)):
        vals[i] = float("nan")
    v = m.var
    desc, sol, req = rng.choice([
        ("the mean, ignoring missing values (use a NaN-aware function)", f"np.nanmean({v})", ["nanmean"]),
        ("the maximum, ignoring missing values (use a NaN-aware function)", f"np.nanmax({v})", ["nanmax"]),
        ("the number of missing values (use `np.isnan`)", f"np.isnan({v}).sum()", ["isnan"]),
        ("the array with missing values removed (use `np.isnan` and `~`)", f"{v}[~np.isnan({v})]", ["isnan"]),
        ("the array with every missing value replaced by 0", f"np.nan_to_num({v}, nan=0.0)", []),
        ("the fraction of values that are missing", f"np.isnan({v}).mean()", ["isnan"]),
    ])
    return P(f"{about(th, m, len(vals), v)}; some readings are missing (`np.nan`). Compute {desc}. "
             f"Remember that `x == np.nan` is always False.", f"{v} = {arr(vals)}", sol, require=req)


@template(M, "easy")
def any_all(rng):
    r, c = rng.randint(3, 5), rng.randint(3, 5)
    s = seeded(rng)
    setup = (f"M = np.random.default_rng({s}).integers(0, 3, size=({r}, {c}))\n"
             f"M[{rng.randrange(r)}] = 0")
    desc, sol, req = rng.choice([
        ("whether **any row is all zeros** (a dead feature row): `.all(axis=1)` then `.any()`",
         "np.any(np.all(M == 0, axis=1))", "all"),
        ("a boolean per **column**: is every value in that column positive?", "np.all(M > 0, axis=0)", "all"),
        ("a boolean per **row**: does it contain at least one 2?", "np.any(M == 2, axis=1)", "any"),
        ("the indices of the rows that are entirely zero (`np.flatnonzero` of a per-row `all`)",
         "np.flatnonzero(np.all(M == 0, axis=1))", "all"),
    ])
    return P(f"`M` is a `({r}, {c})` activation matrix (rows = neurons, columns = inputs). Compute {desc}. "
             f"(`if M > 0:` would raise \"truth value is ambiguous\".)", setup, sol, require=[req])


@template(M, "easy")
def logical_ops(rng):
    n = rng.randint(6, 10)
    s = seeded(rng)
    setup = (f"g = np.random.default_rng({s})\nconf = g.random({n}).round(2)\n"
             f"is_speech = g.random({n}) < 0.5")
    t = rng.choice([0.5, 0.6, 0.7, 0.8])
    desc, sol = rng.choice([
        (f"the frames where the detector is confident (`conf > {t}`) **and** the frame is speech", f"(conf > {t}) & is_speech"),
        (f"the frames that are confident (`conf > {t}`) **or** speech", f"(conf > {t}) | is_speech"),
        (f"the frames that are speech but **not** confident (`~`)", f"is_speech & ~(conf > {t})"),
        (f"the frames where exactly one of 'confident (`conf > {t}`)' and 'speech' holds (`^` or "
         f"`np.logical_xor`)", f"(conf > {t}) ^ is_speech"),
    ])
    return P(f"`conf` holds a voice detector's confidence for {n} audio frames and `is_speech` the ground truth. "
             f"Build a boolean mask of {desc}. Use `&`, `|`, `~`, `^` with parentheses, never `and`/`or`.",
             setup, sol, no_loops=True)


# ================================================================= medium

@template(M, "medium")
def clean_non_finite(rng):
    vals = floats(rng, rng.randint(6, 9), -5, 5, 2)
    for i, bad in zip(rng.sample(range(len(vals)), 3), [float("nan"), float("inf"), float("-inf")]):
        vals[i] = bad
    big = rng.choice([1e6, 1e3, 99.0])
    desc, sol, req = rng.choice([
        (f"replace NaN with 0, +inf with {big} and −inf with −{big} (`np.nan_to_num` with `nan=`, `posinf=`, "
         f"`neginf=`)", f"np.nan_to_num(x, nan=0.0, posinf={big}, neginf=-{big})", "nan_to_num"),
        ("keep only the **finite** values (`np.isfinite`)", "x[np.isfinite(x)]", "isfinite"),
        ("count the infinite values (`np.isinf`)", "np.isinf(x).sum()", "isinf"),
        ("the mean of the finite values only", "x[np.isfinite(x)].mean()", "isfinite"),
    ])
    return P(f"A gradient blew up: `x` contains NaN, +inf and −inf entries. Compute: {desc}.", f"x = {arr(vals)}",
             sol, require=[req])


@template(M, "medium")
def masked_arrays(rng):
    vals = floats(rng, rng.randint(6, 9), 0, 10, 1)
    for i in rng.sample(range(len(vals)), 2):
        vals[i] = float("nan")
    t = rng.choice([7.0, 8.0, 5.0])
    desc, sol, req = rng.choice([
        ("the mean, skipping invalid entries, via `np.ma.masked_invalid(x).mean()`",
         "float(np.ma.masked_invalid(x).mean())", "masked_invalid"),
        ("the array with invalid entries filled with −1 via `np.ma.masked_invalid(x).filled(-1)`",
         "np.ma.masked_invalid(x).filled(-1)", "masked_invalid"),
        ("the valid entries only, as a plain array, via `.compressed()` of `np.ma.masked_invalid(x)`",
         "np.ma.masked_invalid(x).compressed()", "masked_invalid"),
        (f"the sum of the valid entries that are at most {t}: mask the invalid ones, then also mask values "
         f"above {t} with `np.ma.masked_where`", f"float(np.ma.masked_where(np.ma.masked_invalid(x) > {t}, "
         f"np.ma.masked_invalid(x)).sum())", "masked_where"),
    ])
    return P(f"`x` holds sensor readings with NaN gaps. Use a **masked array** (`np.ma`) to get {desc}.",
             f"x = {arr(vals)}", sol, require=[req])


@template(M, "medium")
def silent_division(rng):
    n = rng.randint(5, 8)
    a = [float(rng.randint(0, 9)) for _ in range(n)]
    b = [float(rng.choice([0, rng.randint(1, 5)])) for _ in range(n)]
    b[0], a[0] = 0.0, 0.0  # 0/0 -> NaN
    b[1], a[1] = 0.0, 3.0  # x/0 -> inf
    return P(f"Compute the ratio `a / b` where some `b` are 0, **without warnings**: do the division inside "
             f"`with np.errstate(divide='ignore', invalid='ignore'):`, then replace every non-finite result "
             f"(inf or NaN) with 0.", f"a = {arr(a)}\nb = {arr(b)}",
             "with np.errstate(divide='ignore', invalid='ignore'):\n    r = a / b\nresult = np.where(np.isfinite(r), r, 0.0)",
             require=["errstate"])


# ================================================================= hard

@template(M, "hard")
def precision_recall_f1(rng):
    pos = rng.choice(["spam", "toxic", "defect", "fraud", "speech", "cat"])
    n = rng.randint(12, 20)
    while True:
        yt = [int(rng.random() < 0.45) for _ in range(n)]
        yp = [t if rng.random() < 0.7 else 1 - t for t in yt]
        tp = sum(a and b for a, b in zip(yt, yp))
        if tp >= 1 and sum(yt) >= 1 and sum(yp) >= 1:
            break
    return P(f"A binary {pos} detector made predictions `y_pred` for {n} samples with true labels "
             f"`y_true` (1 = {pos}). Compute **precision, recall and F1** for the positive class and "
             f"return them as a tuple `(precision, recall, f1)`. Use boolean masks, no loops.",
             f"y_true = {arr(yt)}\ny_pred = {arr(yp)}",
             "tp = np.sum((y_pred == 1) & (y_true == 1))\nprecision = tp / np.sum(y_pred == 1)\n"
             "recall = tp / np.sum(y_true == 1)\nf1 = 2 * precision * recall / (precision + recall)\n"
             "result = (precision, recall, f1)", no_loops=True)


@template(M, "hard")
def impute_column_means(rng):
    r, c = rng.randint(4, 6), rng.randint(2, 4)
    X = [[round(rng.uniform(0, 10), 1) for _ in range(c)] for _ in range(r)]
    for _ in range(rng.randint(2, 4)):
        X[rng.randrange(r)][rng.randrange(c)] = float("nan")
    for j in range(c):  # keep at least one value per column
        X[j % r][j] = round(rng.uniform(0, 10), 1)
    if rng.random() < 0.5:
        return P(f"`X` is a `({r}, {c})` feature table with missing values (NaN). **Impute** every NaN with its "
                 f"column's mean, computed while ignoring NaNs (`np.nanmean(..., axis=0)` + `np.where`). No loops.",
                 f"X = {mat(X)}", "result = np.where(np.isnan(X), np.nanmean(X, axis=0), X)", no_loops=True)
    return P(f"`X` is a `({r}, {c})` feature table with missing values (NaN). Return a tuple: the boolean "
             f"**missing-indicator** matrix (`np.isnan(X)`, often fed to the model as extra features) and `X` "
             f"with every NaN replaced by its column's **median** (ignoring NaNs, `np.nanmedian`). No loops.",
             f"X = {mat(X)}", "miss = np.isnan(X)\nresult = (miss, np.where(miss, np.nanmedian(X, axis=0), X))",
             no_loops=True)


@template(M, "hard")
def early_stopping(rng):
    n = rng.randint(10, 15)
    loss, cur = [], rng.uniform(2, 3)
    for _ in range(n):
        cur += rng.uniform(-0.3, 0.12)
        loss.append(round(cur, 3))
    patience, delta = rng.randint(2, 3), rng.choice([0.0, 0.01])
    return P(f"`val_loss` holds the validation loss after each of {n} epochs (0-based). Epoch e **improves** if "
             f"`val_loss[e] < best - {delta}`, where `best` is the lowest loss of all epochs before e (epoch 0 "
             f"always improves). Early stopping with patience {patience} stops at the first epoch that completes "
             f"{patience} epochs in a row without improvement. Return that epoch's index, or −1 if training "
             f"never stops. (`np.minimum.accumulate` gives the running best.)",
             f"val_loss = {arr(loss)}\npatience, min_delta = {patience}, {delta}",
             "best_before = np.r_[np.inf, np.minimum.accumulate(val_loss)[:-1]]\n"
             "improved = val_loss < best_before - min_delta\nrun = 0\nresult = -1\n"
             "for e, ok in enumerate(improved):\n    run = 0 if ok else run + 1\n    if run == patience:\n"
             "        result = e\n        break")


# ================================================================= challenge

@template(M, "challenge")
def wait_for_higher(rng):
    n = rng.randint(8, 12)
    vals = [rng.randint(30, 60) for _ in range(n)]
    var, story = rng.choice([
        ("temp_c", "`temp_c[i]` is a GPU's temperature at minute i."),
        ("acc", "`acc[i]` is a model's validation accuracy (%) after epoch i."),
        ("price", "`price[i]` is the spot price of a cloud GPU on day i."),
    ])
    sol = (f"vals = {var}.tolist()\nwait = [0] * len(vals)\nstack = []\nfor i, x in enumerate(vals):\n"
           "    while stack and vals[stack[-1]] < x:\n        j = stack.pop()\n        wait[j] = i - j\n"
           "    stack.append(i)\nresult = np.array(wait)")
    tests = [rand_ints(var, seeded(rng), 0, 100, 20000),
             f"{var} = np.arange(20000, 0, -1)",
             f"{var} = np.full(20000, 7)",
             f"{var} = np.array([1, 2, 3])"]
    return challenge("How long until it's higher?",
                     story + " For every i, find how many steps you must wait after i to see a **strictly "
                     "higher** value (0 if it never happens). Return a NumPy int array of length n.",
                     f"{var} = {arr(vals)}", sol, tests, "1 ≤ n ≤ 2·10⁴, 0 ≤ values ≤ 10⁵",
                     " (a NumPy array)", speed="O(n) with a monotonic stack")


@template(M, "challenge")
def longest_balanced(rng):
    n = rng.randint(9, 13)
    flags = [rng.random() < 0.5 for _ in range(n)]
    var, story = rng.choice([
        ("is_speech", "`is_speech[i]` says whether a voice-activity detector marked frame i as speech."),
        ("is_positive", "`is_positive[i]` says whether the i-th review in a stream is positive."),
        ("is_pass", "`is_pass[i]` says whether the i-th nightly test run passed."),
    ])
    sol = (f"first = {{0: -1}}\nbal = best = 0\nfor i, f in enumerate({var}.tolist()):\n"
           "    bal += 1 if f else -1\n    if bal in first:\n        best = max(best, i - first[bal])\n"
           "    else:\n        first[bal] = i\nresult = best")
    tests = [f"{var} = np.random.default_rng({seeded(rng)}).random(20000) < 0.5",
             f"{var} = np.random.default_rng({seeded(rng)}).random(20000) < 0.9",
             f"{var} = np.ones(20000, dtype=bool)", f"{var} = np.array([True, False])"]
    return challenge("Longest balanced stretch",
                     story + " Find the length of the **longest contiguous stretch** with as many `True` as "
                     "`False` values (0 if there is none).", f"{var} = np.array({lit(flags)})", sol, tests,
                     "1 ≤ n ≤ 2·10⁴", speed="O(n): prefix balance + first occurrence")
