"""1.16 Strings, dates, structured arrays & bits."""

from ... import themes as T
from ._common import P, arr, challenge, lit, rand_ints, seeded, template

M = "1.16"


def _messy(rng, words):
    out = []
    for w in words:
        w = rng.choice([w, w.upper(), w.capitalize(), w.lower()])
        out.append(rng.choice(["", " ", "  "]) + w + rng.choice(["", " ", "\t"]))
    return out


def _date(rng, year=None):
    y = year or rng.choice([2023, 2024, 2025])
    return f"{y}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"


# ================================================================= lite

@template(M, "lite")
def string_cleanup(rng):
    th = T.theme(rng)
    words = [w for w in T.items(rng, th, rng.randint(4, 6))]
    raw = _messy(rng, words)
    desc, sol, req = rng.choice([
        ("**upper-case** every entry (`np.strings.upper`)", "np.strings.upper(raw)", "upper"),
        ("**lower-case** every entry (`np.strings.lower`)", "np.strings.lower(raw)", "lower"),
        ("**strip** surrounding whitespace from every entry (`np.strings.strip`)", "np.strings.strip(raw)", "strip"),
        ("get the **length** of every entry (`np.strings.str_len`)", "np.strings.str_len(raw)", "str_len"),
    ])
    return P(f"`raw` holds {th.noun} names scraped from logs, with inconsistent case and whitespace. "
             f"Vectorized string ops: {desc}.", f"raw = np.array({lit(raw)})", sol, require=[req])


@template(M, "lite")
def date_arithmetic(rng):
    d = _date(rng)
    k = rng.randint(3, 120)
    d2 = _date(rng)
    desc, sol = rng.choice([
        (f"the date {k} days after `start` (add `np.timedelta64({k}, 'D')`)", f"start + np.timedelta64({k}, 'D')"),
        (f"the date {k} days before `start`", f"start - np.timedelta64({k}, 'D')"),
        ("the number of days from `start` to `end` as an int (subtract, then divide by "
         "`np.timedelta64(1, 'D')`)", "int((end - start) / np.timedelta64(1, 'D'))"),
    ])
    return P(f"`start` and `end` are `np.datetime64` dates of a training run. Compute {desc}.",
             f"start = np.datetime64('{d}')\nend = np.datetime64('{d2}')", sol, require=["timedelta64"])


# ================================================================= easy

@template(M, "easy")
def string_search_replace(rng):
    files = rng.sample(["train_001.wav", "train_002.wav", "test_001.wav", "val_003.flac", "train_004.flac",
                        "test_002.wav", "val_001.wav", "train_010.mp3", "test_007.mp3"], rng.randint(4, 6))
    desc, sol, req = rng.choice([
        ("a boolean mask of the files that **start with** `'train'` (`np.strings.startswith`)",
         "np.strings.startswith(files, 'train')", "startswith"),
        ("a boolean mask of the files that **end with** `'.wav'` (`np.strings.endswith`)",
         "np.strings.endswith(files, '.wav')", "endswith"),
        ("the file names with the extension `.wav` replaced by `.flac` (`np.strings.replace`)",
         "np.strings.replace(files, '.wav', '.flac')", "replace"),
        ("the position of the first `'_'` in every name (`np.strings.find`)", "np.strings.find(files, '_')", "find"),
        ("every name prefixed with `'data/'` (`np.strings.add`)", "np.strings.add('data/', files)", "add"),
    ])
    return P(f"`files` lists audio files in a speech dataset. Build {desc}.", f"files = np.array({lit(files)})",
             sol, require=[req])


@template(M, "easy")
def month_starts(rng):
    y = rng.choice([2023, 2024, 2025])
    a = rng.randint(1, 6)
    b = rng.randint(a + 2, 12)
    if rng.random() < 0.5:
        return P(f"Make an array of the **first day of every month** from {y}-{a:02d} up to (but not including) "
                 f"{y}-{b:02d}, as day-resolution dates: build months with `np.arange(..., dtype='datetime64[M]')` "
                 f"and convert with `.astype('datetime64[D]')`.", "",
                 f"np.arange('{y}-{a:02d}', '{y}-{b:02d}', dtype='datetime64[M]').astype('datetime64[D]')",
                 require=["arange"])
    k = rng.randint(5, 10)
    step = rng.choice([1, 7])
    return P(f"Make an array of {k} evaluation dates starting at {y}-{a:02d}-01, one every {step} day"
             f"{'s' if step > 1 else ''}: `np.arange` with a `datetime64` start and a `timedelta64` step.", "",
             f"np.arange(np.datetime64('{y}-{a:02d}-01'), np.datetime64('{y}-{a:02d}-01') + {k * step}, {step})",
             require=["arange", "datetime64"])


@template(M, "easy")
def business_days(rng):
    y = rng.choice([2024, 2025])
    m1 = rng.randint(1, 11)
    d1 = f"{y}-{m1:02d}-01"
    d2 = f"{y}-{m1 + 1:02d}-01"
    d3 = _date(rng, y)
    k = rng.randint(1, 10)
    desc, sol, req = rng.choice([
        (f"the number of **business days** (Mon-Fri) from {d1} up to (not including) {d2} (`np.busday_count`)",
         f"np.busday_count('{d1}', '{d2}')", "busday_count"),
        (f"whether {d3} is a business day (`np.is_busday`)", f"np.is_busday('{d3}')", "is_busday"),
        (f"the date {k} business days after {d3} (`np.busday_offset`, rolling forward if {d3} is a weekend)",
         f"np.busday_offset('{d3}', {k}, roll='forward')", "busday_offset"),
    ])
    return P(f"A labeling team only works on weekdays. Compute {desc}.", "", sol, require=[req])


@template(M, "easy")
def structured_fields(rng):
    th = T.theme(rng)
    ms = rng.sample(th.measures, 2)
    n = rng.randint(4, 6)
    names = T.items(rng, th, n)
    recs = [(nm, T.value(rng, ms[0]), T.value(rng, ms[1])) for nm in names]
    f1, f2 = T.ident(ms[0].name), T.ident(ms[1].name)
    dt = f"[('name', 'U20'), ('{f1}', 'f8'), ('{f2}', 'f8')]"
    setup = f"recs = np.array({recs!r}, dtype={dt})"  # structured arrays need tuples
    t = sorted(r[1] for r in recs)[n // 2]
    desc, sol = rng.choice([
        (f"the `{f1}` field of every record", f"recs['{f1}']"),
        (f"the **names** of the records whose `{f1}` is at least {t}", f"recs[recs['{f1}'] >= {t}]['name']"),
        (f"the mean of the `{f2}` field", f"recs['{f2}'].mean()"),
    ])
    return P(f"`recs` is a **structured array** (a lightweight table) of {th.plural} with fields `name`, `{f1}` "
             f"and `{f2}`. Get {desc}.", setup, sol)


# ================================================================= medium

@template(M, "medium")
def structured_sort(rng):
    th = T.theme(rng)
    n = rng.randint(4, 6)
    names = T.items(rng, th, n)
    scores = rng.sample(range(40, 100), n)
    desc = rng.random() < 0.5
    recs = list(zip(names, scores))
    if rng.random() < 0.5:
        return P(f"Build a **structured array** `board` with dtype `[('name', 'U20'), ('score', 'i4')]` from the "
                 f"list of `(name, score)` tuples `rows`, then return the names sorted by score "
                 f"{'highest first' if desc else 'lowest first'} (`np.sort(board, order='score')`).",
                 f"rows = {recs!r}",
                 "board = np.array(rows, dtype=[('name', 'U20'), ('score', 'i4')])\n"
                 f"result = np.sort(board, order='score')['name']{'[::-1]' if desc else ''}", require=["order"])
    return P(f"Build a **structured array** with dtype `[('name', 'U20'), ('score', 'i4')]` from `rows`, then use "
             f"`np.rec.array` (record access by attribute) to return the **mean score** as `rec.score.mean()`.",
             f"rows = {recs!r}",
             "rec = np.rec.array(np.array(rows, dtype=[('name', 'U20'), ('score', 'i4')]))\nresult = rec.score.mean()",
             require=["rec"])


@template(M, "medium")
def time_deltas(rng):
    n = rng.randint(4, 6)
    base = f"2024-{rng.randint(1, 12):02d}-{rng.randint(1, 27):02d}T{rng.randint(0, 20):02d}:00"
    offsets = sorted(rng.sample(range(5, 3000), n))
    unit, word = rng.choice([("h", "hours"), ("m", "minutes"), ("s", "seconds")])
    setup = (f"t0 = np.datetime64('{base}')\nevents = t0 + np.array({lit(offsets)}, dtype='timedelta64[m]')")
    if rng.random() < 0.5:
        return P(f"`events` holds {n} job-completion timestamps and `t0` the submit time. Compute each job's "
                 f"duration in **{word}** as floats: subtract and divide by `np.timedelta64(1, '{unit}')`.",
                 setup, f"(events - t0) / np.timedelta64(1, '{unit}')", require=["timedelta64"])
    return P(f"`events` holds {n} timestamps with minute resolution. **Truncate** each to the hour "
             f"(`.astype('datetime64[h]')`) and return the result.", setup, "events.astype('datetime64[h]')",
             require=["astype"])


@template(M, "medium")
def bit_packing(rng):
    n = rng.choice([8, 16, 24])
    s = seeded(rng)
    setup = f"flags = np.random.default_rng({s}).integers(0, 2, size={n}).astype(np.uint8)"
    if rng.random() < 0.5:
        return P(f"`flags` holds {n} binary feature flags (0/1). **Pack** them into bytes with `np.packbits` "
                 f"({n // 8} byte{'s' if n > 8 else ''}).", setup, "np.packbits(flags)", require=["packbits"])
    v = [rng.randint(0, 255) for _ in range(rng.randint(2, 4))]
    return P(f"`b` holds bytes from a compact binary label file. Count the **set bits** (1s) in each byte: unpack "
             f"to bits with `np.unpackbits` on a column (`b[:, None]`), then sum along the bit axis.",
             f"b = np.array({lit(v)}, dtype=np.uint8)", "np.unpackbits(b[:, None], axis=1).sum(axis=1)",
             require=["unpackbits|bitwise_count"])


@template(M, "medium")
def bitwise_ops(rng):
    n = rng.randint(4, 6)
    x = [rng.randint(0, 255) for _ in range(n)]
    k, mask = rng.randint(1, 3), rng.choice([0b1, 0b100, 0b1111, 0b10000000])
    desc, sol = rng.choice([
        (f"shift every value right by {k} bits (divides by 2**{k})", f"x >> {k}"),
        (f"shift every value left by {k} bits, keeping uint8 (overflowing bits drop off)", f"x << {k}"),
        (f"keep only the bits in the mask {bin(mask)} (`&`)", f"x & {mask}"),
        (f"flip the bits in the mask {bin(mask)} (`^`)", f"x ^ {mask}"),
        (f"build a boolean array saying whether bit {k} is set (`(x >> {k}) & 1`, then compare)",
         f"((x >> {k}) & 1) == 1"),
        ("invert every bit (`~` or `np.invert`)", "~x"),
    ])
    return P(f"`x` is a `uint8` array of packed status flags from a robot. Using bitwise operators, {desc}.",
             f"x = np.array({lit(x)}, dtype=np.uint8)", sol)


# ================================================================= hard

@template(M, "hard")
def normalize_labels(rng):
    what, classes = T.labels(rng, min_classes=3)
    used = rng.sample(classes, 3)
    raw = _messy(rng, [rng.choice(used) for _ in range(rng.randint(7, 11))] + used)
    return P(f"Crowd-sourced {what} labels in `raw` have inconsistent case and stray whitespace. **Normalize** them "
             f"(lower-case and strip with `np.strings`) and return a dict mapping each clean label to its count "
             f"(`np.unique(..., return_counts=True)`).", f"raw = np.array({lit(raw)})",
             "clean = np.strings.strip(np.strings.lower(raw))\nlabels, counts = np.unique(clean, return_counts=True)\n"
             "result = dict(zip(labels.tolist(), counts.tolist()))", require=["unique"])


@template(M, "hard")
def requests_per_hour(rng):
    n = rng.randint(8, 14)
    base = f"2024-{rng.randint(1, 12):02d}-{rng.randint(1, 27):02d}T{rng.randint(0, 18):02d}:00"
    offs = sorted(rng.randint(0, 240) for _ in range(n))
    setup = f"ts = np.datetime64('{base}') + np.array({lit(offs)}, dtype='timedelta64[m]')"
    if rng.random() < 0.5:
        return P(f"`ts` holds {n} API request timestamps. Count the requests **per hour**: truncate to the hour "
                 f"with `.astype('datetime64[h]')`, then `np.unique(..., return_counts=True)`. Return the tuple "
                 f"`(hours, counts)`.", setup,
                 "result = np.unique(ts.astype('datetime64[h]'), return_counts=True)", require=["unique"])
    return P(f"`ts` holds {n} sorted API request timestamps. Return the **gaps** between consecutive requests in "
             f"minutes as floats, and the index of the largest gap, as a tuple `(gaps, argmax)`.", setup,
             "gaps = np.diff(ts) / np.timedelta64(1, 'm')\nresult = (gaps, int(gaps.argmax()))", require=["diff"])


@template(M, "hard")
def label_bitmasks(rng):
    what, classes = T.labels(rng, min_classes=4)
    classes = list(classes[:rng.randint(4, 5)])
    n = rng.randint(4, 6)
    sets = [sorted(rng.sample(range(len(classes)), rng.randint(1, 2))) for _ in range(n)]
    return P(f"Each of {n} samples has a **set** of {what} labels (indices into `classes`), given in `label_sets`. "
             f"Encode every set as a bitmask (bit i set if label i is present), then count the **pairs** of samples "
             f"(i < j) that share at least one label, using `&` on the masks with broadcasting and the upper "
             f"triangle (`np.triu(..., k=1)`).", f"classes = {lit(classes)}\nlabel_sets = {lit(sets)}",
             "masks = np.array([sum(1 << c for c in s) for s in label_sets])\n"
             "share = (masks[:, None] & masks[None, :]) != 0\nresult = int(np.triu(share, k=1).sum())")


# ================================================================= challenge

@template(M, "challenge")
def max_xor_pair(rng):
    n = rng.randint(6, 9)
    x = [rng.randint(0, 63) for _ in range(n)]
    sol = ("vals = x.tolist()\nbest = 0\nfor bit in range(29, -1, -1):\n    cand = best | (1 << bit)\n"
           "    prefixes = {v >> bit for v in vals}\n    if any((p ^ (cand >> bit)) in prefixes for p in prefixes):\n"
           "        best = cand\nresult = best")
    tests = [rand_ints("x", seeded(rng), 0, 2**30, 20000), rand_ints("x", seeded(rng), 0, 8, 20000),
             "x = np.full(20000, 12345)", "x = np.array([5])"]
    return challenge("Most different hashes",
                     "`x` holds 30-bit feature hashes. Find the **maximum of `x[i] ^ x[j]`** over all pairs "
                     "(i, j), i.e. the two hashes that differ the most bit-wise (0 if all are equal or n = 1).",
                     f"x = {arr(x)}", sol, tests, "1 ≤ n ≤ 2·10⁴, 0 ≤ x[i] < 2³⁰",
                     speed="O(30·n): greedy bit-by-bit with prefix sets (or a binary trie)")


@template(M, "challenge")
def max_concurrent_sessions(rng):
    n = rng.randint(5, 8)
    day = f"2024-{rng.randint(1, 12):02d}-{rng.randint(1, 27):02d}"
    starts = [rng.randint(0, 120) for _ in range(n)]
    lens = [rng.randint(5, 60) for _ in range(n)]
    sol = ("events = []\nfor s, e in zip(start.tolist(), end.tolist()):\n    events.append((s, 1))\n"
           "    events.append((e, -1))\nevents.sort(key=lambda p: (p[0], p[1]))\ncur = best = 0\n"
           "for _, d in events:\n    cur += d\n    best = max(best, cur)\nresult = best")

    def big(s, n_, span):
        return (f"g = np.random.default_rng({s})\nstart = np.datetime64('2024-01-01T00:00') + "
                f"g.integers(0, {span}, size={n_}).astype('timedelta64[m]')\n"
                f"end = start + g.integers(1, 600, size={n_}).astype('timedelta64[m]')")

    tests = [big(seeded(rng), 20000, 10**6), big(seeded(rng), 20000, 1000),
             "start = np.array(['2024-01-01T00:00', '2024-01-01T01:00'], dtype='datetime64[m]')\n"
             "end = np.array(['2024-01-01T01:00', '2024-01-01T02:00'], dtype='datetime64[m]')"]
    return challenge("Peak concurrent sessions",
                     "`start[i]` and `end[i]` are `datetime64` timestamps of user sessions on a chatbot (a "
                     "session covers `[start, end)`, so one ending exactly when another starts doesn't overlap "
                     "it). Find the **maximum number of sessions active at the same moment**.",
                     f"start = np.datetime64('{day}T09:00') + np.array({lit(starts)}, dtype='timedelta64[m]')\n"
                     f"end = start + np.array({lit(lens)}, dtype='timedelta64[m]')", sol, tests,
                     "1 ≤ n ≤ 2·10⁴", speed="O(n log n) sweep line")
