"""2.03 Reading & writing data.

Runs happen in a fresh temporary directory, so setups can write input files and `post`
checks can read back what the user's code wrote.
"""

from ... import themes as T
from ._common import P, challenge, frame, lit, seeded, template

M = "2.03"


def _to_csv(f, path="data.csv"):
    return f.code() + f"\ndf.to_csv({path!r}, index=False)\ndel df"


# ================================================================= lite

@template(M, "lite")
def csv_roundtrip(rng):
    f = frame(rng, (4, 6), nums=2)
    path = rng.choice(["people.csv", "metrics.csv", "models.csv", "export.csv"])
    if rng.random() < 0.5:
        return P(f"{f.describe()}. Write it to `'{path}'` with `to_csv`, **without the index** (`index=False`).",
                 f.code(), f"df.to_csv({path!r}, index=False)", post=f"open({path!r}).read()", require=["to_csv"])
    return P(f"A colleague exported a table of {f.th.plural} to `'{path}'`. Load it into a DataFrame with "
             f"`pd.read_csv`.", _to_csv(f, path), f"pd.read_csv({path!r})", require=["read_csv"])


@template(M, "lite")
def to_python(rng):
    f = frame(rng, (3, 4), nums=1)
    desc, sol, req = rng.choice([
        ("a list of row dicts (`orient='records'`), e.g. to send as JSON", "df.to_dict(orient='records')", "to_dict"),
        ("a dict mapping each column name to a list of its values (`orient='list'`)", "df.to_dict(orient='list')", "to_dict"),
        ("a NumPy array (`to_numpy`)", "df.to_numpy()", "to_numpy"),
        ("a list of the column names", "df.columns.tolist()", "tolist|list"),
    ])
    return P(f"{f.describe()}. Convert it to {desc}.", f.code(), sol, require=[req])


# ================================================================= easy

@template(M, "easy")
def read_csv_options(rng):
    f = frame(rng, (4, 6), nums=2, date=True)
    a, b = f.num
    k = rng.randint(2, 3)
    desc, sol = rng.choice([
        (f"only the columns `{f.key}` and `{a}` (`usecols=`)", f"pd.read_csv('data.csv', usecols=[{f.key!r}, {a!r}])"),
        (f"only the first {k} rows (`nrows=`)", f"pd.read_csv('data.csv', nrows={k})"),
        (f"with `{f.key}` as the index (`index_col=`)", f"pd.read_csv('data.csv', index_col={f.key!r})"),
        (f"with `{f.date}` parsed as datetimes (`parse_dates=`)", f"pd.read_csv('data.csv', parse_dates=[{f.date!r}])"),
        (f"with the columns renamed on the way in to `{['c' + str(i) for i in range(len(f.cols))]}` "
         f"(`names=` plus `header=0` to replace the file's header)",
         f"pd.read_csv('data.csv', names={['c' + str(i) for i in range(len(f.cols))]!r}, header=0)"),
    ])
    return P(f"`data.csv` holds a table of {f.th.plural} (columns {', '.join(f'`{c}`' for c in f.cols)}). Read it "
             f"{desc}.", _to_csv(f), sol, require=["read_csv"])


@template(M, "easy")
def json_lines(rng):
    f = frame(rng, (3, 5), nums=2)
    if rng.random() < 0.5:
        return P(f"{f.describe()}. Write it as **JSON Lines** to `'rows.jsonl'` (one JSON object per line: "
                 f"`orient='records', lines=True`).", f.code(), "df.to_json('rows.jsonl', orient='records', lines=True)",
                 post="open('rows.jsonl').read()", require=["to_json"])
    return P(f"`rows.jsonl` holds one JSON object per line (JSON Lines) describing {f.th.plural}. Load it with "
             f"`pd.read_json(..., lines=True)`.",
             f.code() + "\ndf.to_json('rows.jsonl', orient='records', lines=True)\ndel df",
             "pd.read_json('rows.jsonl', lines=True)", require=["read_json"])


@template(M, "easy")
def text_tables(rng):
    f = frame(rng, (3, 4), nums=2)
    desc, sol = rng.choice([
        ("a plain-text table **without the index** (`to_string(index=False)`)", "df.to_string(index=False)"),
        ("CSV text without the index, returned as a string (call `to_csv` with no path)", "df.to_csv(index=False)"),
        ("an HTML table without the index (`to_html(index=False)`)", "df.to_html(index=False)"),
    ])
    return P(f"{f.describe()}. For a report, render it as {desc}.", f.code(), sol)


# ================================================================= medium

@template(M, "medium")
def messy_csv(rng):
    th = T.theme(rng)
    names = T.items(rng, th, 4)
    vals = [rng.choice(["?", str(rng.randint(1, 9) * 1000 + rng.randint(0, 999))]) for _ in names]
    vals[0] = str(rng.randint(1, 9) * 1000 + 123)
    variant = rng.choice(["na", "thousands", "comment"])
    if variant == "na":
        text = "name,value\n" + "".join(f"{n},{v.replace('?', '?')}\n" for n, v in zip(names, vals))
        return P("`raw.csv` uses `?` for missing values. Read it so `?` becomes NaN (`na_values=`).",
                 f"with open('raw.csv', 'w') as fh:\n    fh.write({text!r})",
                 "pd.read_csv('raw.csv', na_values=['?'])", require=["read_csv"])
    if variant == "thousands":
        nums = [f"{rng.randint(1, 99)},{rng.randint(100, 999)}" for _ in names]
        text = "name,amount\n" + "".join(f'{n},"{v}"\n' for n, v in zip(names, nums))
        return P("`raw.csv` writes numbers with thousands separators inside quotes (e.g. `\"12,345\"`). Read it so "
                 "`amount` is numeric (`thousands=','`).", f"with open('raw.csv', 'w') as fh:\n    fh.write({text!r})",
                 "pd.read_csv('raw.csv', thousands=',')", require=["read_csv"])
    text = "# exported by tool v2\nname,value\n" + "".join(f"{n},{rng.randint(1, 99)}\n" for n in names) + "# end\n"
    return P("`raw.csv` has comment lines starting with `#`. Read it, skipping them (`comment='#'`).",
             f"with open('raw.csv', 'w') as fh:\n    fh.write({text!r})", "pd.read_csv('raw.csv', comment='#')",
             require=["read_csv"])


@template(M, "medium")
def chunked_read(rng):
    n, k = rng.randint(20, 60), rng.choice([5, 7, 10])
    s = seeded(rng)
    setup = (f"g = np.random.default_rng({s})\npd.DataFrame({{'user': g.integers(0, 8, {n}), "
             f"'tokens': g.integers(1, 500, {n})}}).to_csv('big.csv', index=False)")
    if rng.random() < 0.5:
        return P(f"`big.csv` is too large for memory (pretend!). Stream it with `pd.read_csv(..., chunksize={k})` "
                 f"and return the **total** of the `tokens` column, adding up chunk by chunk.", setup,
                 f"total = 0\nfor chunk in pd.read_csv('big.csv', chunksize={k}):\n    total += chunk['tokens'].sum()\nresult = total",
                 require=["chunksize"])
    return P(f"Stream `big.csv` with `pd.read_csv(..., chunksize={k})` and return a list with the **number of rows** "
             f"in each chunk.", setup, f"result = [len(c) for c in pd.read_csv('big.csv', chunksize={k})]",
             require=["chunksize"])


@template(M, "medium")
def nested_json(rng):
    n = rng.randint(2, 4)
    names = rng.sample(["Ana", "Ben", "Cal", "Dee", "Eve"], n)
    recs = [{"id": i + 1, "user": {"name": nm, "plan": rng.choice(["free", "pro"])},
             "usage": {"tokens": rng.randint(10, 999)}} for i, nm in enumerate(names)]
    if rng.random() < 0.5:
        return P("`records` is a list of **nested** JSON objects from an API. Flatten it into a DataFrame with "
                 "`pd.json_normalize` (nested keys become dotted column names like `user.name`).",
                 f"records = {recs!r}", "pd.json_normalize(records)", require=["json_normalize"])
    return P("`records` is a list of nested JSON objects. Flatten it with `pd.json_normalize` and keep only the "
             "columns `id`, `user.name` and `usage.tokens`.", f"records = {recs!r}",
             "pd.json_normalize(records)[['id', 'user.name', 'usage.tokens']]", require=["json_normalize"])


@template(M, "medium")
def sqlite_query(rng):
    f = frame(rng, (5, 8), nums=1)
    a = f.num[0]
    setup = f.code() + "\nimport sqlite3\ncon = sqlite3.connect(':memory:')"
    if rng.random() < 0.5:
        return P(f"{f.describe()}. Load it into the SQLite connection `con` as a table named `t` (`to_sql`, no "
                 f"index), then use `pd.read_sql` to get the average `{a}` per `{f.group}` as columns `{f.group}`, "
                 f"`avg_{a}`, ordered by `{f.group}`.", setup,
                 f"df.to_sql('t', con, index=False)\nresult = pd.read_sql('SELECT {f.group}, AVG({a}) AS avg_{a} "
                 f"FROM t GROUP BY {f.group} ORDER BY {f.group}', con)", require=["to_sql", "read_sql"])
    t = sorted(v for v in f.cols[a])[len(f.cols[a]) // 2]
    return P(f"{f.describe()}. Load it into SQLite (`con`, table `t`, no index) and query the rows with `{a}` "
             f"**greater than {t}**, ordered by `{a}`, passing the threshold as a **parameter** "
             f"(`params=({t},)` with a `?` placeholder), never by formatting it into the SQL string.", setup,
             f"df.to_sql('t', con, index=False)\nresult = pd.read_sql('SELECT * FROM t WHERE {a} > ? ORDER BY {a}', "
             f"con, params=({t},))", require=["read_sql", "params"])


# ================================================================= hard

@template(M, "hard")
def csv_pipeline(rng):
    f = frame(rng, (6, 9), nums=2, date=True)
    a, b = f.num
    lvl = f.levels[0]
    return P(f"`data.csv` holds a table of {f.th.plural}. Read it with `{f.date}` parsed as dates and `{f.key}` as the "
             f"index, keep the rows whose `{f.group}` is `{lvl!r}`, and write their `{a}` and `{b}` columns to "
             f"`'subset.csv'` with 2 decimals (`float_format='%.2f'`), keeping the index.", _to_csv(f),
             f"df = pd.read_csv('data.csv', parse_dates=[{f.date!r}], index_col={f.key!r})\n"
             f"df[df[{f.group!r}] == {lvl!r}][[{a!r}, {b!r}]].to_csv('subset.csv', float_format='%.2f')",
             post="open('subset.csv').read()", require=["read_csv", "to_csv"])


@template(M, "hard")
def pickle_vs_csv(rng):
    f = frame(rng, (4, 6), nums=1, date=True)
    return P(f"{f.describe()}. First make `{f.group}` a `category`. Then round-trip the frame through **CSV** "
             f"(`'t.csv'`, no index) and through **pickle** (`'t.pkl'`), and return the tuple "
             f"`(dtypes_after_csv, dtypes_after_pickle)`: CSV loses the category and datetime types, pickle keeps "
             f"them.", f.code(),
             f"df[{f.group!r}] = df[{f.group!r}].astype('category')\ndf.to_csv('t.csv', index=False)\n"
             "df.to_pickle('t.pkl')\nresult = (pd.read_csv('t.csv').dtypes, pd.read_pickle('t.pkl').dtypes)",
             require=["to_pickle", "read_pickle"])


@template(M, "hard")
def sql_join_report(rng):
    f = frame(rng, (6, 8), nums=1)
    a = f.num[0]
    leads = {lvl: rng.choice(["Kim", "Lee", "Max", "Ola", "Raj"]) for lvl in f.levels}
    return P(f"{f.describe()}. `leads` maps each `{f.group}` to its lead. Load `df` as table `t` and `leads` as table "
             f"`l` (columns `{f.group}`, `lead`) into SQLite (`con`, no index), then with **one** `pd.read_sql` query "
             f"return, per lead, the number of rows (`n`) and the max `{a}` (`best`), ordered by lead.",
             f.code() + f"\nleads = {leads!r}\nimport sqlite3\ncon = sqlite3.connect(':memory:')",
             f"df.to_sql('t', con, index=False)\npd.DataFrame(list(leads.items()), columns=[{f.group!r}, 'lead']).to_sql('l', con, index=False)\n"
             f"result = pd.read_sql('SELECT l.lead, COUNT(*) AS n, MAX(t.{a}) AS best FROM t JOIN l ON t.{f.group} = l.{f.group} "
             f"GROUP BY l.lead ORDER BY l.lead', con)", require=["read_sql"])


# ================================================================= challenge

@template(M, "challenge")
def chunked_distinct(rng):
    n, k = rng.randint(9, 13), rng.randint(3, 4)
    users = [rng.randint(1, 6) for _ in range(n)]
    sol = ("seen = set()\nresult = []\nfor chunk in pd.read_csv('events.csv', chunksize=CHUNK):\n"
           "    seen.update(chunk['user'].tolist())\n    result.append(len(seen))")

    def big(seed, n_, users_, chunk):
        return (f"g = np.random.default_rng({seed})\npd.DataFrame({{'user': g.integers(0, {users_}, {n_})}})"
                f".to_csv('events.csv', index=False)\nCHUNK = {chunk}")

    tests = [big(seeded(rng), 20000, 5000, 997), big(seeded(rng), 20000, 20000, 20000), big(seeded(rng), 2000, 3, 1),
             "pd.DataFrame({'user': [7]}).to_csv('events.csv', index=False)\nCHUNK = 5"]
    return challenge("Distinct users so far",
                     "`events.csv` logs one `user` id per event, in time order. A dashboard reads it in chunks of "
                     "`CHUNK` rows (`pd.read_csv(..., chunksize=CHUNK)`). After each chunk it shows how many "
                     "**distinct users have appeared so far**. Return that list.",
                     f"pd.DataFrame({{'user': {lit(users)}}}).to_csv('events.csv', index=False)\nCHUNK = {k}",
                     sol, tests, "≤ 2·10⁴ rows", speed="O(n) overall")


@template(M, "challenge")
def log_percentiles(rng):
    eps = rng.sample(["/predict", "/embed", "/chat", "/search"], 3)
    lines = [f"2024-05-0{rng.randint(1, 9)}T10:{rng.randint(10, 59)}:00 user=u{rng.randint(1, 9)} "
             f"endpoint={rng.choice(eps)} ms={rng.randint(5, 400)}" for _ in range(rng.randint(8, 11))]
    sol = ("import math\nlog = pd.read_csv('requests.log', sep=' ', header=None, names=['ts', 'user', 'endpoint', 'ms'])\n"
           "log['endpoint'] = log['endpoint'].str.removeprefix('endpoint=')\n"
           "log['ms'] = log['ms'].str.removeprefix('ms=').astype(int)\nresult = {}\n"
           "for ep, ms in log.groupby('endpoint')['ms']:\n    v = sorted(ms)\n"
           "    result[ep] = v[math.ceil(0.95 * len(v)) - 1]")

    def big(seed, n_):
        return (f"g = np.random.default_rng({seed})\neps = np.array(['/predict', '/embed', '/chat', '/search', '/tokenize'])\n"
                f"with open('requests.log', 'w') as fh:\n"
                f"    fh.write(''.join(f'2024-05-01T10:00:00 user=u{{u}} endpoint={{e}} ms={{m}}\\n' for u, e, m in "
                f"zip(g.integers(0, 99, {n_}), eps[g.integers(0, 5, {n_})], g.integers(1, 5000, {n_}))))")

    tests = [big(seeded(rng), 20000), big(seeded(rng), 7),
             "with open('requests.log', 'w') as fh:\n    fh.write('2024-05-01T10:00:00 user=u1 endpoint=/x ms=42\\n')"]
    return challenge("Tail latency per endpoint",
                     "`requests.log` has one request per line: `<timestamp> user=<id> endpoint=<path> ms=<latency>`. "
                     "Parse it and return a dict mapping every endpoint to its **p95 latency** using the "
                     "**nearest-rank** method: sort the endpoint's latencies and take the `ceil(0.95·n)`-th smallest "
                     "(1-based).",
                     f"with open('requests.log', 'w') as fh:\n    fh.write({chr(10).join(lines) + chr(10)!r})", sol, tests,
                     "≤ 2·10⁴ lines", speed="O(n log n)")
