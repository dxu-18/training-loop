"""pandas catalog templates: the rest of the guide's API, one method per problem (2.01-2.19)."""

from ._common import STDOUT, catalog, lit, seeded, template


def _df(rng, n=None):
    """A small literal people/score frame with a few duplicates and a missing score."""
    n = n or rng.randint(5, 7)
    names = rng.sample(["ana", "ben", "cy", "dee", "eli", "fay", "gus", "hal"], n)
    teams = [rng.choice(["red", "blue", "green"]) for _ in range(n)]
    teams[0], teams[1] = "red", "blue"
    ages = [rng.randint(20, 45) for _ in range(n)]
    scores = [round(rng.uniform(50, 99), 1) for _ in range(n)]
    scores[rng.randrange(1, n)] = None
    ages[-1] = ages[0]
    code = (f"df = pd.DataFrame({{\n    'name': {lit(names)},\n    'team': {lit(teams)},\n    'age': {lit(ages)},\n"
            f"    'score': {lit([float('nan') if v is None else v for v in scores])},\n}})")
    return code, n, ages


# ================================================================= 2.01 core objects & options

@template("2.01", "lite")
def options_index_catalog(rng):
    keys = rng.sample(list("abcdefgh"), rng.randint(3, 5))
    setup = f"idx = pd.Index({lit(keys)}, name='key')"
    items = [
        ("reset_option", "Change `display.max_rows` to 5 with `pd.set_option`, then restore the default with `pd.reset_option` and "
                         "return `pd.get_option('display.max_rows')`.",
         "pd.set_option('display.max_rows', 5)\npd.reset_option('display.max_rows')\nresult = pd.get_option('display.max_rows')"),
        ("is_unique", "Return `idx.is_unique`.", "idx.is_unique"),
        ("values", "Return the underlying NumPy array of `idx` (`idx.values`).", "idx.values"),
        ("dtype", "Return the dtype of `pd.Series(np.arange(3))` (`.dtype`).", "pd.Series(np.arange(3)).dtype", {"require": ["dtype", "arange"]}),
    ]
    return catalog(rng, items, setup)


# ================================================================= 2.02 creating

@template("2.02", "lite")
def construct_catalog(rng):
    s = seeded(rng)
    vals = [rng.randint(1, 9), None, rng.randint(1, 9)]
    setup = f"rng = np.random.default_rng({s})"
    items = [
        ("array", f"Make a **nullable** integer array of `{vals}` (`pd.array(..., dtype='Int64')`): the None becomes `pd.NA`, "
                  f"not a float NaN. Return it as a list.", f"result = pd.array({vals}, dtype='Int64').tolist()"),
        ("infer_objects", "`raw = pd.DataFrame({'n': pd.Series([1, 2, 3], dtype=object)})` stores numbers as objects. Return the "
                          "dtype of `n` after `raw.infer_objects()` as a string.",
         "raw = pd.DataFrame({'n': pd.Series([1, 2, 3], dtype=object)})\nresult = str(raw.infer_objects()['n'].dtype)"),
        ("NaT", "Return `[pd.isna(pd.NaT), pd.isna(pd.NA), pd.isna(np.nan)]`: the missing markers for datetimes, nullable dtypes "
                "and floats.", "[pd.isna(pd.NaT), pd.isna(pd.NA), pd.isna(np.nan)]", {"require": ["NaT", "NA"]}),
        ("normal", "Make a Series of 4 standard-normal draws from `rng.normal(size=4)` with index `list('abcd')`.",
         "pd.Series(rng.normal(size=4), index=list('abcd'))"),
    ]
    return catalog(rng, items, setup)


# ================================================================= 2.03 I/O

@template("2.03", "easy")
def io_more_catalog(rng):
    setup, n, _ = _df(rng)
    items = [
        ("read_table", "Write `df` as a tab-separated file `df.tsv` (`to_csv(..., sep='\\t', index=False)`) and read it back with "
                       "`pd.read_table` (TSV is its default). Return the first 3 rows (`head(3)`).",
         "df.to_csv('df.tsv', sep='\\t', index=False)\nresult = pd.read_table('df.tsv').head(3)", {"require": ["read_table", "head"]}),
        ("keys", "Round-trip `df` through `df.csv` and return the loaded frame's `keys()` as a list (the column names).",
         "df.to_csv('df.csv', index=False)\nresult = list(pd.read_csv('df.csv').keys())"),
        ("shape", "Round-trip `df` through a JSON file (`to_json('df.json')` / `pd.read_json`) and return the loaded `shape`.",
         "df.to_json('df.json')\nresult = pd.read_json('df.json').shape", {"require": ["read_json", "shape"]}),
        ("to_latex", "Render `df[['name', 'age']]` as a LaTeX table for a paper (`to_latex(index=False)`) and return whether it "
                     "contains `\\\\begin{tabular}`.", "result = '\\\\begin{tabular}' in df[['name', 'age']].to_latex(index=False)"),
    ]
    return catalog(rng, items, setup)


# ================================================================= 2.04 inspecting

@template("2.04", "lite")
def inspect_more_catalog(rng):
    setup, n, _ = _df(rng)
    items = [
        ("T", "Return the shape of the transposed frame `df.T`.", "df.T.shape"),
        ("index", "Return the row labels of `df` as a list (`df.index`).", "list(df.index)"),
        ("ndim", "Return `[df.ndim, df['age'].ndim]`.", "[df.ndim, df['age'].ndim]"),
        ("is_unique", "Are the values of `df['age']` unique? (`.is_unique`)", "df['age'].is_unique"),
        ("is_monotonic_increasing", "Is `df['age']` sorted ascending? Return `[df['age'].is_monotonic_increasing, "
                                    "df['age'].sort_values().is_monotonic_increasing]`.",
         "[df['age'].is_monotonic_increasing, df['age'].sort_values().is_monotonic_increasing]"),
        ("describe", "Summarise only the numeric columns: return `df.describe(include='number')`.", "df.describe(include='number')",
         {"require": ["describe", "include"]}),
        ("select_dtypes", "Return the columns that are **not** numeric (`df.select_dtypes(exclude='number').columns`) as a list.",
         "list(df.select_dtypes(exclude='number').columns)", {"require": ["select_dtypes", "exclude"]}),
        ("info", "Print `df.info(memory_usage='deep')` (it counts string contents). Graded on the output.",
         "df.info(memory_usage='deep')", {"answer": [STDOUT], "require": ["info", "memory_usage"]}),
        ("unique", "Return the distinct teams in order of appearance (`df['team'].unique()`).", "df['team'].unique()"),
        ("memory_usage", "Return the total bytes `df` uses, counting string contents (`df.memory_usage(deep=True).sum()`).",
         "df.memory_usage(deep=True).sum()", {"require": ["memory_usage", "deep"]}),
        ("info", "Print a summary of `df` with `df.info()` (columns, non-null counts, dtypes, memory). Graded on the output.",
         "df.info()", {"answer": [STDOUT]}),
    ]
    return catalog(rng, items, setup)


# ================================================================= 2.05-2.09 selecting, columns, sorting, missing, applying

@template("2.05", "easy")
def select_more_catalog(rng):
    setup, n, _ = _df(rng)
    k = rng.randint(1, 3)
    items = [
        ("nsmallest", f"Return the {k} youngest rows with `df.nsmallest({k}, 'age')`.", f"df.nsmallest({k}, 'age')"),
        ("name", "Selecting one column gives a Series named after it. Return `df['score'].name`.", "df['score'].name"),
        ("get", "Look up a column that might not exist without a KeyError: return `df.get('missing_col', default=None) is None`.",
         "df.get('missing_col', default=None) is None", {"require": ["get", "default"]}),
    ]
    return catalog(rng, items, setup)


@template("2.06", "easy")
def copy_catalog(rng):
    setup, n, _ = _df(rng)
    items = [
        ("copy", "Make an independent copy with `df.copy()`, add 1 to the copy's `age` column, and return the **original** "
                 "`df['age']` (unchanged).", "d2 = df.copy()\nd2['age'] = d2['age'] + 1\nresult = df['age']"),
        ("copy", "Return `df.copy(deep=True)['name'].tolist() == df['name'].tolist()`.", "df.copy(deep=True)['name'].tolist() == df['name'].tolist()"),
    ]
    return catalog(rng, items, setup)


@template("2.07", "easy")
def duplicated_catalog(rng):
    setup, n, _ = _df(rng)
    items = [
        ("duplicated", "Count rows whose `age` already appeared earlier (`df.duplicated(subset=['age']).sum()`).",
         "df.duplicated(subset=['age']).sum()"),
        ("duplicated", "Show **all** rows involved in an age duplicate (`df[df.duplicated(subset=['age'], keep=False)]`).",
         "df[df.duplicated(subset=['age'], keep=False)]", {"require": ["duplicated", "keep"]}),
    ]
    return catalog(rng, items, setup)


@template("2.08", "easy")
def missing_more_catalog(rng):
    setup, n, _ = _df(rng)
    items = [
        ("set_index", "Move `name` into the index (`set_index`), sort by it (`sort_index`), and return the `score` column.",
         "df.set_index('name').sort_index()['score']", {"require": ["set_index", "sort_index"]}),
        ("any", "Return, per column, whether it has any missing value (`df.isna().any()`).", "df.isna().any()"),
        ("dropna", "Drop only the rows where **every** value is missing (`dropna(how='all')`) from `df` with an all-NaN row "
                   "appended; return the number of rows left.",
         "d = pd.concat([df, pd.DataFrame([[np.nan] * 4], columns=df.columns)], ignore_index=True)\nresult = len(d.dropna(how='all'))",
         {"require": ["dropna", "how"]}),
        ("NA", "Return `[pd.isna(pd.NA), pd.NA is pd.NA]`: the nullable-dtype missing marker.", "[pd.isna(pd.NA), pd.NA is pd.NA]"),
        ("NA", "Return the nullable integer Series `pd.Series([1, None, 3], dtype='Int64')` with its missing value filled by 0.",
         "pd.Series([1, None, 3], dtype='Int64').fillna(0)", {"require": ["fillna"]}),
        ("NaT", "Parse `['2024-01-01', 'oops']` with `pd.to_datetime(..., errors='coerce')` and return whether the bad value became NaT.",
         "result = pd.to_datetime(['2024-01-01', 'oops'], errors='coerce')[1] is pd.NaT", {"require": ["to_datetime", "NaT"]}),
    ]
    return catalog(rng, items, setup)


@template("2.09", "easy")
def query_chain_catalog(rng):
    setup, n, ages = _df(rng)
    a = sorted(ages)[len(ages) // 2]
    items = [
        ("query", f"Return the rows with `age > {a}` and team 'red' using `df.query(\"age > {a} and team == 'red'\")`.",
         f"df.query(\"age > {a} and team == 'red'\")"),
        ("apply", "Return a per-row label `name + '-' + team` computed with `df.apply(lambda r: r['name'] + '-' + r['team'], axis=1)`.",
         "df.apply(lambda r: r['name'] + '-' + r['team'], axis=1)", {"require": ["apply", "axis"]}),
        ("query", f"Reference a Python variable inside `query` with `@`: `min_age = {a}`; return `df.query('age > @min_age')`.",
         f"min_age = {a}\nresult = df.query('age > @min_age')"),
        ("sort_values", f"Chain: `df.query('age > {a - 5}').assign(score=lambda d: d['score'].fillna(0)).sort_values('score', "
                        f"ascending=False).head(3)`.", f"df.query('age > {a - 5}').assign(score=lambda d: d['score'].fillna(0)).sort_values('score', ascending=False).head(3)",
         {"require": ["query", "sort_values", "head"]}),
    ]
    return catalog(rng, items, setup)


# ================================================================= 2.10 strings

@template("2.10", "easy")
def str_more_catalog(rng):
    words = rng.sample(["hello world", "  Data Science", "ml ops ", "deep  learning", "Spam", "abc", "x42"], 4)
    setup = f"s = pd.Series({lit(words)})"
    fns = [("capitalize", "s.str.capitalize()"), ("swapcase", "s.str.swapcase()"), ("isalpha", "s.str.isalpha()"),
           ("islower", "s.str.islower()"), ("isspace", "s.str.isspace()"), ("isdigit", "pd.Series(['7', 'x', '42']).str.isdigit()"),
           ("isnumeric", "pd.Series(['7', 'x', '42']).str.isnumeric()"), ("removeprefix", "s.str.removeprefix('  ')"),
           ("removesuffix", "s.str.removesuffix(' ')"), ("repeat", "s.str.repeat(2)"), ("wrap", "s.str.wrap(5)"),
           ("center", "s.str.center(15, '*')"), ("rsplit", "s.str.rsplit(' ', n=1)"), ("partition", "s.str.partition(' ')"),
           ("match", "s.str.match(r'\\s*[A-Z]')"), ("fullmatch", "s.str.fullmatch(r'[a-z ]+')"),
           ("cat", "s.str.cat(sep=' | ', na_rep='?')"),
           ("contains", "pd.Series(['data', None, 'ml']).str.contains('a', na=False)"),
           ("findall", "s.str.findall(r'[a-z]+')"), ("extractall", "pd.Series(['a1b2', 'c3']).str.extractall(r'(\\d)')"),
           ("join", "pd.Series([['a', 'b'], ['c']]).str.join('-')")]
    items = [(name, f"Return `{code}`.", code) for name, code in fns]
    return catalog(rng, items, setup, "`s` is a Series of strings.")


# ================================================================= 2.11 dates & times

@template("2.11", "medium")
def datetime_more_catalog(rng):
    y, m = rng.randint(2021, 2025), rng.randint(1, 12)
    setup = (f"s = pd.Series(pd.to_datetime(['{y}-{m:02d}-15 09:45', '{y}-{m:02d}-28 17:05', '{y + 1}-01-31 23:59']))\n"
             f"daily = pd.DataFrame({{'sales': np.arange(14.0)}}, index=pd.date_range('{y}-{m:02d}-01', periods=14, freq='D'))")
    items = [
        ("MonthEnd", f"Return `pd.Timestamp('{y}-{m:02d}-15') + pd.offsets.MonthEnd(0)`: roll forward to the month's end.",
         f"pd.Timestamp('{y}-{m:02d}-15') + pd.offsets.MonthEnd(0)", {"require": ["MonthEnd", "offsets"]}),
        ("Period", f"Return the end time of the monthly period `pd.Period('{y}-{m:02d}', freq='M')` (`.end_time`).",
         f"pd.Period('{y}-{m:02d}', freq='M').end_time"),
        ("period_range", f"Return `pd.period_range('{y}-01', periods=3, freq='M')` as a list of strings.",
         f"[str(p) for p in pd.period_range('{y}-01', periods=3, freq='M')]"),
        ("asfreq", "Downsample `daily` to Mondays with `asfreq('W-MON')` (it picks values, it doesn't aggregate).", "daily.asfreq('W-MON')"),
        ("agg", "Resample `daily` to weeks and compute the sum and max of `sales` with `.agg(['sum', 'max'])`.",
         "daily.resample('W')['sales'].agg(['sum', 'max'])", {"require": ["resample", "agg"]}),
        ("index", "Return the mean sales per weekday number (`daily.groupby(daily.index.dayofweek)['sales'].mean()`).",
         "daily.groupby(daily.index.dayofweek)['sales'].mean()", {"require": ["index", "dayofweek"]}),
        ("tz_localize", "Attach the UTC time zone to `daily`'s index with `daily.tz_localize('UTC')` and return the index's tz name.",
         "str(daily.tz_localize('UTC').index.tz)"),
        ("ceil", "Round `s` up to the next day (`s.dt.ceil('D')`).", "s.dt.ceil('D')"),
        ("tz_convert", "Interpret `s` as UTC and convert it to New York time (`s.dt.tz_localize('UTC').dt.tz_convert("
                       "'America/New_York')`); return the hours.", "s.dt.tz_localize('UTC').dt.tz_convert('America/New_York').dt.hour"),
        ("date", "Return the calendar dates of `s` (`s.dt.date`) as a list.", "list(s.dt.date)"),
        ("time", "Return the times of day of `s` (`s.dt.time`) as a list.", "list(s.dt.time)"),
        ("day", "Return `[s.dt.day.tolist(), s.dt.minute.tolist()]`.", "[s.dt.day.tolist(), s.dt.minute.tolist()]", {"require": ["day", "minute"]}),
        ("dayofyear", "Return the day of the year of each timestamp (`s.dt.dayofyear`).", "s.dt.dayofyear"),
        ("days_in_month", "Return how many days each timestamp's month has (`s.dt.days_in_month`).", "s.dt.days_in_month"),
        ("is_month_end", "Return which timestamps fall on the last day of their month (`s.dt.is_month_end`).", "s.dt.is_month_end"),
        ("isocalendar", "Return the ISO week number of each timestamp (`s.dt.isocalendar().week`).", "s.dt.isocalendar().week"),
        ("now", "`pd.Timestamp.now()` and `pd.Timestamp.today()` read the clock, so only their type is graded: return "
                "`[type(pd.Timestamp.now()).__name__, pd.Timestamp.today().normalize() <= pd.Timestamp.now()]`.",
         "[type(pd.Timestamp.now()).__name__, pd.Timestamp.today().normalize() <= pd.Timestamp.now()]", {"require": ["now", "today"]}),
        ("head", "Return the first 3 rows of `daily` (`head(3)`).", "daily.head(3)"),
    ]
    return catalog(rng, items, setup)


# ================================================================= 2.12 categoricals

@template("2.12", "easy")
def category_more_catalog(rng):
    lv = rng.choice([["low", "mid", "high"], ["S", "M", "L"], ["bronze", "silver", "gold"]])
    vals = [rng.choice(lv) for _ in range(6)]
    setup = f"s = pd.Series({lit(vals)})"
    items = [
        ("CategoricalDtype", f"Make an **ordered** category type `pd.CategoricalDtype({lv}, ordered=True)`, convert `s` to it, and "
                             f"return its minimum.", f"t = pd.CategoricalDtype({lv}, ordered=True)\nresult = s.astype(t).min()",
         {"require": ["CategoricalDtype", "min"]}),
        ("reorder_categories", f"Reorder the categories of `s.astype('category')` to `{lv[::-1]}` (`.cat.reorder_categories`) "
                               f"and return the new category list.", f"list(s.astype('category').cat.reorder_categories({lv[::-1]}).cat.categories)"),
        ("memory_usage", "Do categories save memory? Return `s.astype('category').memory_usage(deep=True) < s.memory_usage(deep=True)` "
                         "for `s` repeated 100 times.", "big = pd.concat([s] * 100, ignore_index=True)\n"
                                                         "result = bool(big.astype('category').memory_usage(deep=True) < big.memory_usage(deep=True))",
         {"require": ["memory_usage", "deep"]}),
    ]
    return catalog(rng, items, setup)


# ================================================================= 2.13 groupby

@template("2.13", "medium")
def groupby_more_catalog(rng):
    setup, n, ages = _df(rng, rng.randint(6, 7))
    setup += "\ng = df.groupby('team')"
    edges = sorted({20, 30, 40, 50})
    items = [
        ("get_group", "Return the rows of the red team (`g.get_group('red')`).", "g.get_group('red')"),
        ("groups", "Return `{team: list(labels)}` from `g.groups` (which row labels belong to each team).",
         "{k: list(v) for k, v in g.groups.items()}"),
        ("ngroups", "Return `[g.ngroups, g.ngroup().tolist()]`: the number of groups and each row's group number.",
         "[g.ngroups, g.ngroup().tolist()]", {"require": ["ngroups", "ngroup"]}),
        ("head", "Return each team's first row (`g.head(1)`).", "g.head(1)"),
        ("nth", "Return each team's first row with `g.nth(0)`.", "g.nth(0)"),
        ("first", "Return the first **non-null** score of each team (`g['score'].first()`).", "g['score'].first()"),
        ("last", "Return the last non-null score of each team (`g['score'].last()`).", "g['score'].last()"),
        ("rank", "Rank scores **within** each team, highest first (`g['score'].rank(ascending=False, method='first')`).",
         "g['score'].rank(ascending=False, method='first')", {"require": ["rank", "method"]}),
        ("diff", "Return the age difference to the previous member of the same team (`g['age'].diff()`).", "g['age'].diff()"),
        ("nlargest", "Return the 2 highest ages overall with `df['age'].nlargest(2)`.", "df['age'].nlargest(2)"),
        ("apply", "Return each team's age range with `g['age'].apply(lambda a: a.max() - a.min())`.", "g['age'].apply(lambda a: a.max() - a.min())"),
        ("describe", "Return `g['age'].describe()` (count, mean, std, quartiles per team).", "g['age'].describe()"),
        ("cut", f"Bin ages into decades with `pd.cut(df['age'], bins={edges})` and count rows per bin (`observed=True`).",
         f"df.groupby(pd.cut(df['age'], bins={edges}), observed=True).size()", {"require": ["cut", "observed"]}),
        ("rolling", "Return a 2-row rolling mean of age within each team (`g['age'].rolling(2, min_periods=1).mean()`).",
         "g['age'].rolling(2, min_periods=1).mean()", {"require": ["rolling", "min_periods"]}),
        ("sample", "Draw one row per team (a stratified sample) with `g.sample(n=1, random_state=0)` and return the teams of the "
                   "sampled rows, sorted.", "sorted(g.sample(n=1, random_state=0)['team'])", {"require": ["sample", "random_state"]}),
        ("shape", "Return the shape of `g.size().reset_index()`.", "g.size().reset_index().shape"),
        ("dt", "Group by month: add `joined = pd.date_range('2024-01-25', periods=len(df), freq='3D')` and return the row count "
               "per month (`df.groupby(joined.month).size()`, using the `.dt` accessor on a Series of dates).",
         "joined = pd.Series(pd.date_range('2024-01-25', periods=len(df), freq='3D'))\nresult = df.groupby(joined.dt.month).size()",
         {"require": ["dt", "month"]}),
        ("year", "Group by calendar year: with `joined = pd.Series(pd.date_range('2023-12-20', periods=len(df), freq='5D'))`, "
                 "return `df.groupby(joined.dt.year)['age'].mean()`.",
         "joined = pd.Series(pd.date_range('2023-12-20', periods=len(df), freq='5D'))\nresult = df.groupby(joined.dt.year)['age'].mean()"),
        ("groupby", "Keep missing keys and the original order: on a copy whose first team is None, return "
                    "`d.groupby('team', dropna=False, sort=False).size()`.",
         "d = df.copy()\nd.loc[0, 'team'] = None\nresult = d.groupby('team', dropna=False, sort=False).size()", {"require": ["dropna", "sort"]}),
    ]
    return catalog(rng, items, setup)


# ================================================================= 2.15 combining

@template("2.15", "easy")
def update_catalog(rng):
    setup, n, _ = _df(rng)
    items = [
        ("update", "`patch = pd.DataFrame({'score': [100.0]}, index=[1])`. Overwrite `df` **in place** with patch's non-NA values "
                   "(`df.update(patch)`, on a `df.copy()`) and return the score column.",
         "d = df.copy()\nd.update(pd.DataFrame({'score': [100.0]}, index=[1]))\nresult = d['score']", {"require": ["update", "copy"]}),
        ("shape", "Stack `df` with itself (`pd.concat([df, df])`) and return the shape.", "pd.concat([df, df]).shape"),
        ("concat", "Keep only the columns two frames share: return the columns of `pd.concat([df, df[['name', 'age']]], join='inner')`.",
         "list(pd.concat([df, df[['name', 'age']]], join='inner').columns)", {"require": ["concat", "join"]}),
        ("loc", "Add a `joined` column of dates `pd.to_datetime(['2024-01-0' + str(i + 1) for i in range(len(df))])` and return "
                "`df.loc[0, 'joined']`.", "d = df.assign(joined=pd.to_datetime(['2024-01-0' + str(i + 1) for i in range(len(df))]))\nresult = d.loc[0, 'joined']",
         {"require": ["loc", "to_datetime"]}),
    ]
    return catalog(rng, items, setup)


# ================================================================= 2.16 windows

@template("2.16", "easy")
def window_more_catalog(rng):
    vals = [rng.randint(1, 20) for _ in range(rng.randint(6, 9))]
    setup = f"s = pd.Series({lit(vals)}, index=pd.date_range('2024-01-01', periods={len(vals)}, freq='D'))"
    items = [
        ("cummax", "Return the running maximum of `s` (`s.cummax()`).", "s.cummax()"),
        ("index", "Return the dates where `s` hits a new running maximum (`s.index[s == s.cummax()]`) as strings.",
         "[str(d.date()) for d in s.index[s == s.cummax()]]", {"require": ["index", "cummax"]}),
        ("ewm", "Return a time-aware exponential average with a 2-day half-life (`s.ewm(halflife='2 days', times=s.index).mean()`).",
         "s.ewm(halflife='2 days', times=s.index).mean()", {"require": ["ewm", "halflife", "times"]}),
        ("expanding", "Return the expanding mean with at least 3 values (`s.expanding(min_periods=3).mean()`).",
         "s.expanding(min_periods=3).mean()"),
    ]
    return catalog(rng, items, setup)


# ================================================================= 2.18 plotting & styling

@template("2.18", "medium")
def plot_more_catalog(rng):
    setup, n, _ = _df(rng)
    setup = "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n" + setup
    items = [
        ("area", "Draw an unstacked area chart of `df[['age', 'score']].fillna(0)` (`.plot.area(stacked=False, alpha=0.4)`) and "
                 "return how many filled areas the axes hold (`len(ax.collections)`).",
         "ax = df[['age', 'score']].fillna(0).plot.area(stacked=False, alpha=0.4)\nresult = len(ax.collections)"),
        ("boxplot", "Draw `df.boxplot(column='age', by='team')` and return how many boxes it drew (the number of teams).",
         "ax = df.boxplot(column='age', by='team')\nresult = len(ax.get_xticklabels())", {"require": ["boxplot", "by"]}),
        ("scatter_matrix", "Return the shape of the axes grid from `pd.plotting.scatter_matrix(df[['age', 'score']], diagonal='hist')` "
                           "(`diagonal='kde'` would need SciPy).",
         "pd.plotting.scatter_matrix(df[['age', 'score']], diagonal='hist').shape", {"require": ["scatter_matrix", "diagonal"]}),
        ("plot", "Draw `df.plot(x='name', y=['age', 'score'], style=['o-', '--'], grid=True, secondary_y='score')` and return "
                 "how many lines the main axes has (`len(ax.get_lines())`).",
         "ax = df.fillna(0).plot(x='name', y=['age', 'score'], style=['o-', '--'], grid=True, secondary_y='score')\nresult = len(ax.get_lines())",
         {"require": ["plot", "secondary_y", "style"]}),
        ("scatter", "Draw `df.plot.scatter(x='age', y='score', c='age', colormap='viridis')` on the rows without NaN and return "
                    "how many points it drew.",
         "ax = df.dropna().plot.scatter(x='age', y='score', c='age', colormap='viridis')\nresult = len(ax.collections[0].get_offsets())",
         {"require": ["scatter", "colormap"]}),
        ("box", "Draw box plots of `df[['age', 'score']]` with `.plot.box()` and return the tick labels' texts.",
         "ax = df[['age', 'score']].plot.box()\nresult = [t.get_text() for t in ax.get_xticklabels()]"),
        ("close", "Draw `df['age'].plot()`, then free it with `plt.close('all')` and return how many figures are open "
                  "(`len(plt.get_fignums())`).", "df['age'].plot()\nplt.close('all')\nresult = len(plt.get_fignums())"),
        ("background_gradient", "Style the frame with `df.style.background_gradient(subset=['age'], cmap='Greens')` and return "
                                "whether its HTML contains `background-color`.",
         "result = 'background-color' in df.style.background_gradient(subset=['age'], cmap='Greens').to_html()",
         {"require": ["background_gradient", "cmap"]}),
        ("highlight_null", "Return whether `df.style.highlight_null().to_html()` contains `background-color` (the missing score is "
                           "highlighted).", "'background-color' in df.style.highlight_null().to_html()"),
        ("bar", "Draw in-cell bars with `df.style.bar(subset=['age'], color='#9cf')` and return whether the HTML contains "
                "`linear-gradient`.", "'linear-gradient' in df.style.bar(subset=['age'], color='#9cf').to_html()", {"require": ["bar", "color"]}),
    ]
    return catalog(rng, items, setup)


# ================================================================= 2.19 performance & handing data to models

@template("2.19", "easy")
def model_handoff_catalog(rng):
    setup, n, _ = _df(rng)
    items = [
        ("from_numpy", "Hand features to PyTorch: return `torch.from_numpy(df[['age']].to_numpy(dtype='float32'))` (it shares "
                       "memory with the array).", "import torch\nresult = torch.from_numpy(df[['age']].to_numpy(dtype='float32')).tolist()",
         {"require": ["from_numpy"]}),
        ("tensor", "Return `torch.tensor(df['age'].to_numpy())` as a list (a copy).", "import torch\nresult = torch.tensor(df['age'].to_numpy()).tolist()",
         {"require": ["tensor"]}),
        ("iterrows", "`iterrows` is the slowest way to loop (it builds a Series per row). Return the names from "
                     "`[row['name'] for _, row in df.head(3).iterrows()]`.", "[row['name'] for _, row in df.head(3).iterrows()]",
         {"require": ["iterrows", "head"]}),
        ("memory_usage", "Return `df.memory_usage(deep=True).sum()`: total bytes, counting string contents.",
         "df.memory_usage(deep=True).sum()", {"require": ["memory_usage", "sum"]}),
        ("assert_series_equal", "Call `pd.testing.assert_series_equal(df['age'], df['age'].copy())` in try/except AssertionError "
                                "and return True if it passed.", "try:\n    pd.testing.assert_series_equal(df['age'], df['age'].copy())\n"
                                                                 "    result = True\nexcept AssertionError:\n    result = False"),
        ("arange", "Build a 10,000-row frame `big = pd.DataFrame({'a': np.arange(10_000)})` and return its column sum.",
         "big = pd.DataFrame({'a': np.arange(10_000)})\nresult = big['a'].sum()", {"require": ["arange", "sum"]}),
    ]
    return catalog(rng, items, setup)
