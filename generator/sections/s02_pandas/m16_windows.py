"""2.16 Window functions (rolling, expanding, ewm, leakage-safe lag features)."""

from ._common import P, challenge, lit, seeded, template

M = "2.16"


def _walk(rng, n=None, var="price"):
    n = n or rng.randint(20, 40)
    start = f"2024-0{rng.randint(1, 9)}-01"
    return (f"{var} = pd.Series(np.random.default_rng({seeded(rng)}).normal(size={n}).cumsum().round(3) + 100, "
            f"index=pd.date_range({start!r}, periods={n}, freq='D'))"), n


# ================================================================= lite

@template(M, "lite")
def rolling_mean(rng):
    setup, n = _walk(rng)
    k = rng.choice([3, 5, 7])
    desc, sol = rng.choice([
        (f"the {k}-day rolling **mean** (the first {k - 1} values are NaN)", f"price.rolling({k}).mean()"),
        (f"the {k}-day rolling mean, allowing partial windows at the start (`min_periods=1`)", f"price.rolling({k}, min_periods=1).mean()"),
        (f"the {k}-day **centered** rolling median (`center=True`)", f"price.rolling({k}, center=True).median()"),
        (f"the {k}-day rolling **sum**", f"price.rolling({k}).sum()"),
    ])
    return P(f"`price` is a daily price series ({n} days). Compute {desc}.", setup, sol, require=["rolling"])


@template(M, "lite")
def expanding_stats(rng):
    setup, n = _walk(rng)
    desc, sol, req = rng.choice([
        ("the **running maximum** so far (`expanding().max()` or `cummax()`)", "price.expanding().max()", "expanding|cummax"),
        ("the running mean of everything up to each day (`expanding().mean()`)", "price.expanding().mean()", "expanding"),
        ("the running mean, but only once at least 5 values exist (`expanding(min_periods=5)`)",
         "price.expanding(min_periods=5).mean()", "expanding"),
        ("the running minimum so far (`cummin()`)", "price.cummin()", "cummin|expanding"),
    ])
    return P(f"`price` is a daily price series. Compute {desc}.", setup, sol, require=[req])


# ================================================================= easy

@template(M, "easy")
def rolling_aggregates(rng):
    setup, n = _walk(rng)
    k = rng.choice([5, 7, 10])
    desc, sol = rng.choice([
        (f"the {k}-day rolling mean **and** std as two columns (`rolling({k}).agg(['mean', 'std'])`)",
         f"price.rolling({k}).agg(['mean', 'std'])"),
        (f"the {k}-day rolling 90th percentile (`rolling({k}).quantile(0.9)`)", f"price.rolling({k}).quantile(0.9)"),
        (f"the {k}-day rolling max minus rolling min (the window's range)",
         f"price.rolling({k}).max() - price.rolling({k}).min()"),
    ])
    return P(f"`price` is a daily price series. Compute {desc}.", setup, sol, require=["rolling"])


@template(M, "easy")
def time_based_window(rng):
    days = sorted(rng.sample(range(1, 40), rng.randint(8, 12)))
    vals = [rng.randint(1, 20) for _ in days]
    w = rng.choice([3, 7])
    return P(f"`visits` is indexed by **irregular** dates (some days are missing). Compute the total over the last "
             f"**{w} calendar days** for each date with a time-based window (`rolling('{w}D').sum()`; a fixed "
             f"`rolling({w})` would count rows, not days).",
             f"visits = pd.Series({vals!r}, index=pd.Timestamp('2024-01-01') + pd.to_timedelta({days!r}, unit='D'))",
             f"visits.rolling('{w}D').sum()", require=["rolling"])


@template(M, "easy")
def exponential_weighting(rng):
    setup, n = _walk(rng)
    span, alpha = rng.choice([5, 10, 12, 26]), rng.choice([0.1, 0.3, 0.5])
    desc, sol = rng.choice([
        (f"the exponentially weighted mean with `span={span}`", f"price.ewm(span={span}).mean()"),
        (f"the exponentially weighted mean with `alpha={alpha}` and `adjust=False` (the classic recursive EMA)",
         f"price.ewm(alpha={alpha}, adjust=False).mean()"),
    ])
    return P(f"`price` is a daily price series. Compute {desc}.", setup, sol, require=["ewm"])


# ================================================================= medium

@template(M, "medium")
def macd(rng):
    setup, n = _walk(rng, rng.randint(40, 60))
    fast, slow = rng.choice([(12, 26), (5, 20), (8, 21)])
    return P(f"`price` is a daily price series. Compute the **MACD line**: the EMA with `span={fast}` minus the EMA with "
             f"`span={slow}` (both `ewm(...).mean()` with the default `adjust=True`).", setup,
             f"price.ewm(span={fast}).mean() - price.ewm(span={slow}).mean()", require=["ewm"])


@template(M, "medium")
def rolling_custom(rng):
    setup, n = _walk(rng)
    k = rng.choice([3, 5, 7])
    return P(f"`price` is a daily price series. Compute, for each {k}-day window, the **change from the first to the "
             f"last day** of the window with `rolling({k}).apply(lambda w: w[-1] - w[0], raw=True)` (`raw=True` passes "
             f"NumPy arrays, which is faster).", setup, f"price.rolling({k}).apply(lambda w: w[-1] - w[0], raw=True)",
             require=["rolling", "apply"])


@template(M, "medium")
def per_entity_lags(rng):
    users = ["u1", "u2", "u3"][:rng.randint(2, 3)]
    rows = [(u, rng.randint(0, 50)) for u in users for _ in range(rng.randint(3, 4))]
    setup = f"feat = pd.DataFrame({rows!r}, columns=['user', 'value'])"
    desc, sol = rng.choice([
        ("each user's **previous** value (lag 1; NaN for their first row)", "feat.groupby('user')['value'].shift(1)"),
        ("the change from each user's previous value", "feat.groupby('user')['value'].diff()"),
        ("each user's value two rows earlier (lag 2)", "feat.groupby('user')['value'].shift(2)"),
    ])
    return P(f"`feat` holds time-ordered values for several users. Compute {desc}, within each user only "
             f"(`groupby(...).shift` / `diff`).", setup, sol, require=["groupby"])


@template(M, "medium")
def strided_rolling(rng):
    n = rng.randint(10, 16)
    s = seeded(rng)
    k, step = rng.choice([(3, 2), (4, 2), (4, 4), (5, 3)])
    return P(f"`df` has {n} rows of two signals. Compute a {k}-row rolling mean but only **every {step}th** window "
             f"(`rolling({k}, step={step}).mean()`), e.g. to cut the cost of a long series.",
             f"df = pd.DataFrame(np.random.default_rng({s}).normal(size=({n}, 2)).round(2), columns=['a', 'b'])",
             f"df.rolling({k}, step={step}).mean()", require=["rolling"])


# ================================================================= hard

@template(M, "hard")
def leakage_safe_features(rng):
    users = ["u1", "u2", "u3"][:rng.randint(2, 3)]
    rows = [(u, rng.randint(0, 50)) for u in users for _ in range(rng.randint(3, 5))]
    k = rng.randint(2, 3)
    return P(f"`feat` holds time-ordered values per user. Build **leakage-safe** features that only use each user's "
             f"**past** rows, and return `feat` with them: `lag1` (previous value) and `roll{k}_mean` (mean of the "
             f"previous {k} values, `min_periods=1`): within each user, `shift(1)` first, then `rolling`.",
             f"feat = pd.DataFrame({rows!r}, columns=['user', 'value'])",
             "g = feat.groupby('user')['value']\nfeat['lag1'] = g.shift(1)\n"
             f"feat['roll{k}_mean'] = g.transform(lambda x: x.shift(1).rolling({k}, min_periods=1).mean())\nresult = feat",
             require=["shift", "rolling"])


@template(M, "hard")
def bollinger_bands(rng):
    setup, n = _walk(rng, rng.randint(30, 45))
    k = rng.choice([5, 10, 20])
    return P(f"`price` is a daily price series. Build **Bollinger bands** as a DataFrame with columns `mid` ({k}-day "
             f"rolling mean), `upper` (mid + 2·rolling std) and `lower` (mid − 2·rolling std), plus a boolean column "
             f"`breakout` that is True where `price` is above `upper`.", setup,
             f"r = price.rolling({k})\nmid, sd = r.mean(), r.std()\n"
             "result = pd.DataFrame({'mid': mid, 'upper': mid + 2 * sd, 'lower': mid - 2 * sd, 'breakout': price > mid + 2 * sd})",
             require=["rolling"])


@template(M, "hard")
def rolling_correlation(rng):
    setup, n = _walk(rng, rng.randint(30, 45))
    k = rng.choice([7, 10, 14])
    return P(f"`price` is a daily price series. Measure short-term momentum: the {k}-day **rolling correlation** between "
             f"each day's price **change** (`diff()`) and the previous day's change (`diff().shift(1)`).", setup,
             f"chg = price.diff()\nresult = chg.rolling({k}).corr(chg.shift(1))", require=["rolling", "corr"])


# ================================================================= challenge

@template(M, "challenge")
def count_windows_mean_ge(rng):
    n = rng.randint(6, 9)
    x = [rng.randint(0, 10) for _ in range(n)]
    t = rng.randint(3, 7)
    sol = ("from bisect import bisect_right, insort\nprefix = 0\nseen = [0]\ncount = 0\nfor v in s.tolist():\n"
           "    prefix += v - t\n    count += bisect_right(seen, prefix)\n    insort(seen, prefix)\nresult = count")

    def big(seed, n_, hi, t_):
        return f"s = pd.Series(np.random.default_rng({seed}).integers(0, {hi}, {n_}))\nt = {t_}"

    tests = [big(seeded(rng), 20000, 100, 50), big(seeded(rng), 20000, 100, 99), big(seeded(rng), 20000, 10, 0),
             "s = pd.Series([3])\nt = 4"]
    return challenge("Windows that beat the target",
                     "`s` holds a model's daily accuracy (an integer percentage). Count the **contiguous windows** "
                     "(any length ≥ 1) whose **average** is at least the target `t`. (Hint: subtract `t` from every "
                     "value; then you need windows with sum ≥ 0, i.e. pairs of prefix sums in order.)",
                     f"s = pd.Series({lit(x)})\nt = {t}", sol, tests, "1 ≤ n ≤ 2·10⁴",
                     speed="O(n log n), e.g. a sorted list of prefix sums or a Fenwick tree")


@template(M, "challenge")
def longest_window_mean_ge(rng):
    n = rng.randint(6, 9)
    x = [rng.randint(0, 10) for _ in range(n)]
    t = rng.randint(3, 7)
    sol = ("p = [0]\nfor v in s.tolist():\n    p.append(p[-1] + v - t)  # window (i, j] is on target iff p[i] <= p[j]\n"
           "n = len(p)\nmin_left, max_right = p[:], p[:]\nfor i in range(1, n):\n    min_left[i] = min(min_left[i - 1], p[i])\n"
           "for j in range(n - 2, -1, -1):\n    max_right[j] = max(max_right[j + 1], p[j])\n"
           "best = i = j = 0\nwhile i < n and j < n:\n    if min_left[i] <= max_right[j]:\n        best = max(best, j - i)\n"
           "        j += 1\n    else:\n        i += 1\nresult = best")

    def big(seed, n_, hi, t_):
        return f"s = pd.Series(np.random.default_rng({seed}).integers(0, {hi}, {n_}))\nt = {t_}"

    tests = [big(seeded(rng), 20000, 100, 50), big(seeded(rng), 20000, 100, 52), big(seeded(rng), 20000, 10, 11),
             "s = pd.Series([5])\nt = 5"]
    return challenge("Longest window on target",
                     "`s` holds a model's daily accuracy (integer percentages) and `t` is the target. Return the length "
                     "of the **longest contiguous window** whose **average** is at least `t` (0 if none).",
                     f"s = pd.Series({lit(x)})\nt = {t}", sol, tests, "1 ≤ n ≤ 2·10⁴",
                     speed="O(n): prefix minima and suffix maxima of the shifted prefix sums, then two pointers")
