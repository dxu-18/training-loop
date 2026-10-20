"""1.01 Setup, printing & help."""

from ... import themes as T
from ._common import P, STDOUT, arr, challenge, lit, mat, rand_ints, seeded, series, template

M = "1.01"

OPTION_KEYS = (
    ("precision", "number of decimals NumPy prints for floats"),
    ("threshold", "array size above which NumPy summarizes with '...'"),
    ("edgeitems", "number of items shown at each end of a summarized array"),
    ("linewidth", "maximum number of characters per printed line"),
)
DOC_FUNCS = ("clip", "where", "argsort", "linspace", "concatenate", "unique", "searchsorted",
             "percentile", "einsum", "pad", "histogram", "meshgrid")


def float_series(rng, decimals=5):
    th, m, _, _ = series(rng)
    n = rng.randint(4, 7)
    vals = [round(rng.uniform(m.lo, m.hi), decimals) for _ in range(n)]
    return th, m, vals


# ================================================================= lite

@template(M, "lite")
def version_and_options(rng):
    ctx = rng.choice(["Before filing a bug report", "At the top of a new notebook",
                      "While debugging a teammate's script", "Before comparing two machines"])
    if rng.random() < 0.2:
        return P(f"{ctx}, check which NumPy you are running: get the installed **version string**.",
                 "", "np.__version__", require=["__version__"])
    key, what = rng.choice(OPTION_KEYS)
    return P(f"{ctx}, inspect the print settings. Use `np.get_printoptions()` (it returns a dict) to "
             f"read the current **{key}**: the {what}.", "", f'np.get_printoptions()["{key}"]',
             require=["get_printoptions"])


@template(M, "lite")
def array_to_string(rng):
    th, m, vals = float_series(rng)
    v, p = m.var, rng.randint(1, 3)
    desc, sol = rng.choice([
        (f"with {p} decimal{'s' if p > 1 else ''} (`precision={p}`)", f"np.array2string({v}, precision={p})"),
        ("with `', '` between the values (`separator=`)", f"np.array2string({v}, separator=', ')"),
        (f"with {p} decimal{'s' if p > 1 else ''} and `', '` between the values",
         f"np.array2string({v}, precision={p}, separator=', ')"),
    ])
    return P(f"`{v}` holds the {m.label} of {len(vals)} {th.plural}. Use `np.array2string` to turn it into "
             f"a string {desc}. Return the string.", f"{v} = {arr(vals)}", sol, require=["array2string"])


@template(M, "lite")
def format_scalar(rng):
    v, what = rng.choice([
        (round(rng.uniform(1e-5, 9e-4), 8), "learning rate"), (round(rng.uniform(0.5, 7), 7), "validation loss"),
        (round(rng.uniform(1e3, 9e5), 4), "tokens per second"), (round(rng.uniform(0.9, 0.99999), 7), "accuracy"),
    ])
    p = rng.randint(2, 4)
    if rng.random() < 0.5:
        return P(f"`v` holds the {what}. Format it as a plain decimal string (no exponent) with at most {p} "
                 f"digits after the point using `np.format_float_positional`.", f"v = {v!r}",
                 f"np.format_float_positional(v, precision={p})", require=["format_float_positional"])
    return P(f"`v` holds the {what}. Format it in scientific notation with {p} digits after the point "
             f"using `np.format_float_scientific`.", f"v = {v!r}",
             f"np.format_float_scientific(v, precision={p})", require=["format_float_scientific"])


# ================================================================= easy

@template(M, "easy")
def print_with_options(rng):
    th, m, vals = float_series(rng)
    v, p = m.var, rng.randint(1, 3)
    variant = rng.choice(["precision", "suppress", "fixed"])
    if variant == "suppress":
        vals = [round(x * 1e-5, 9) if i % 2 else x for i, x in enumerate(vals)]
        return P(f"`{v}` mixes normal values with tiny ones, so NumPy prints it in scientific notation. "
                 f"**Print** it with {p} decimals and no scientific notation (`suppress=True`), using the "
                 f"temporary context manager `np.printoptions` so the global settings stay unchanged.",
                 f"{v} = {arr(vals)}", f"with np.printoptions(precision={p}, suppress=True):\n    print({v})",
                 answer=[STDOUT], require=["printoptions"])
    if variant == "fixed":
        return P(f"`{v}` holds the {m.label} of {len(vals)} {th.plural}. **Print** it so every value shows "
                 f"exactly {p} decimals, padding with zeros (`precision={p}, floatmode='fixed'`), inside a "
                 f"`with np.printoptions(...)` block.", f"{v} = {arr(vals)}",
                 f"with np.printoptions(precision={p}, floatmode='fixed'):\n    print({v})",
                 answer=[STDOUT], require=["printoptions"])
    return P(f"`{v}` holds the {m.label} of {len(vals)} {th.plural}. **Print** it with {p} decimals "
             f"without changing the global options: use the context manager `with np.printoptions(...):`.",
             f"{v} = {arr(vals)}", f"with np.printoptions(precision={p}):\n    print({v})",
             answer=[STDOUT], require=["printoptions"])


@template(M, "easy")
def show_docs(rng):
    fn = rng.choice(DOC_FUNCS)
    ctx = rng.choice(["You forgot the argument order of", "A teammate asks what this does:",
                      "You need the docstring of", "Before using it in a pipeline, read up on"])
    return P(f"{ctx} `np.{fn}`. **Print** its documentation from code with `np.info` "
             f"(in Jupyter you could also write `np.{fn}?`).", "", f"np.info(np.{fn})",
             answer=[STDOUT], require=["info"])


@template(M, "easy")
def set_global_options(rng):
    variant = rng.choice(["precision", "summarize", "linewidth"])
    if variant == "precision":
        th, m, vals = float_series(rng)
        v, p = m.var, rng.randint(1, 3)
        return P(f"Call `np.set_printoptions` to make NumPy print floats with {p} decimals and without "
                 f"scientific notation (`suppress=True`), then return `repr({v})` (a string).",
                 f"{v} = {arr(vals)}", f"np.set_printoptions(precision={p}, suppress=True)\nrepr({v})",
                 require=["set_printoptions"])
    if variant == "summarize":
        n, k = rng.randint(30, 200), rng.randint(2, 4)
        return P(f"`ids` holds {n} token ids. Call `np.set_printoptions` so that any array with more than "
                 f"20 elements is summarized showing {k} items at each end (`threshold`, `edgeitems`), "
                 f"then return `repr(ids)`.", f"ids = np.arange(100, {100 + n})",
                 f"np.set_printoptions(threshold=20, edgeitems={k})\nrepr(ids)", require=["set_printoptions"])
    w = rng.choice([30, 40, 50])
    n = rng.randint(15, 30)
    return P(f"`steps` holds {n} checkpoint step numbers. Call `np.set_printoptions` so no printed line is "
             f"longer than {w} characters (`linewidth`), then return `repr(steps)`.",
             f"steps = np.arange(0, {n * 500}, 500)", f"np.set_printoptions(linewidth={w})\nrepr(steps)",
             require=["set_printoptions"])


# ================================================================= medium

@template(M, "medium")
def print_summarized(rng):
    k = rng.randint(1, 4)
    if rng.random() < 0.5:
        n = rng.randint(200, 2000)
        return P(f"`x` holds {n} audio samples. **Print** it so only {k} item{'s' if k > 1 else ''} appear "
                 f"at each end (with `...` in between), using `np.printoptions` with `threshold` and "
                 f"`edgeitems` inside a `with` block.",
                 f"x = np.round(np.sin(np.arange({n}) / 7), 3)",
                 f"with np.printoptions(threshold=10, edgeitems={k}):\n    print(x)",
                 answer=[STDOUT], require=["printoptions", "edgeitems"])
    r, c = rng.randint(10, 40), rng.randint(10, 40)
    return P(f"`emb` is a {r}×{c} embedding matrix. **Print** it so only {k} row{'s' if k > 1 else ''} and "
             f"column{'s' if k > 1 else ''} appear at each edge, with 2 decimals, using a "
             f"`with np.printoptions(...)` block (`threshold`, `edgeitems`, `precision`).",
             f"emb = np.random.default_rng({seeded(rng)}).normal(size=({r}, {c}))",
             f"with np.printoptions(threshold=10, edgeitems={k}, precision=2):\n    print(emb)",
             answer=[STDOUT], require=["printoptions", "edgeitems"])


@template(M, "medium")
def print_with_formatter(rng):
    n = rng.randint(4, 7)
    desc, fmt, vals = rng.choice([
        ("a percentage with one decimal (e.g. `12.3%`)", "f'{v:.1%}'", [round(rng.random(), 4) for _ in range(n)]),
        ("exactly two decimals", "f'{v:.2f}'", [round(rng.uniform(-50, 50), 4) for _ in range(n)]),
        ("an explicit sign and three decimals (e.g. `+0.125`)", "f'{v:+.3f}'", [round(rng.uniform(-2, 2), 5) for _ in range(n)]),
        ("scientific notation with two decimals (e.g. `3.14e-04`)", "f'{v:.2e}'", [round(rng.uniform(1e-5, 1e-2), 8) for _ in range(n)]),
    ])
    var = rng.choice(["probs", "scores", "deltas", "grads"])
    return P(f"**Print** `{var}` with every float shown as {desc}. Pass a `formatter` dict "
             f"(key `'float_kind'`) to `np.printoptions` in a `with` block.",
             f"{var} = {arr(vals)}",
             f"with np.printoptions(formatter={{'float_kind': lambda v: {fmt}}}):\n    print({var})",
             answer=[STDOUT], require=["printoptions", "formatter"])


@template(M, "medium")
def array2string_options(rng):
    variant = rng.choice(["matrix", "width", "summary"])
    s = seeded(rng)
    if variant == "matrix":
        r, c, p = rng.randint(2, 4), rng.randint(2, 4), rng.randint(1, 3)
        return P(f"`W` is a {r}×{c} weight matrix with some tiny entries. Turn it into a string with "
                 f"`np.array2string` using `', '` as the separator, {p} decimals and `suppress_small=True` "
                 f"(tiny values print as 0).",
                 f"W = np.random.default_rng({s}).normal(size=({r}, {c})) * np.array({lit([1, 1e-6] * 2)}[:{c}])",
                 f"np.array2string(W, separator=', ', precision={p}, suppress_small=True)",
                 require=["array2string"])
    if variant == "width":
        w, n = rng.choice([25, 35, 45]), rng.randint(12, 25)
        return P(f"`ids` holds {n} token ids. Turn it into a string with `np.array2string` so that no line "
                 f"is longer than {w} characters (`max_line_width`).",
                 f"ids = np.random.default_rng({s}).integers(0, 50000, size={n})",
                 f"np.array2string(ids, max_line_width={w})", require=["array2string"])
    n, k = rng.randint(50, 300), rng.randint(1, 3)
    return P(f"`loss` holds {n} logged loss values. Turn it into a short string with `np.array2string`: "
             f"summarize any array longer than 10 elements, showing {k} at each end (`threshold`, "
             f"`edgeitems`), with 3 decimals.",
             f"loss = np.round(np.exp(-np.arange({n}) / 50) + 0.1, 5)",
             f"np.array2string(loss, threshold=10, edgeitems={k}, precision=3)", require=["array2string"])


# ================================================================= hard

@template(M, "hard")
def print_report(rng):
    th, m, ax, rows, cols, data = _report_table(rng)
    v = m.var
    sep, p = rng.choice([": ", " -> ", " | "]), rng.randint(1, 2)
    return P(f"`{v}` holds {m.label} with one row per {th.noun} (names in `{th.var}`) and one column per "
             f"{ax.noun}. **Print one line per {th.noun}**: its name, then `{sep!r}`, then its row formatted "
             f"with `np.array2string(row, precision={p}, separator=', ')`. Example line: "
             f"`{rows[0]}{sep}[...]`.",
             f"{th.var} = {lit(rows)}\n{v} = {mat(data)}",
             f"for name, row in zip({th.var}, {v}):\n"
             f"    print(name + {sep!r} + np.array2string(row, precision={p}, separator=', '))",
             answer=[STDOUT])


def _report_table(rng):
    th = T.theme(rng)
    ax = T.axis(rng, min_labels=3, max_labels=12)
    m = rng.choice(th.measures)
    r, c = rng.randint(2, 4), rng.randint(3, 5)
    data = [[round(rng.uniform(m.lo, m.hi), 4) for _ in range(c)] for _ in range(r)]
    return th, m, ax, T.items(rng, th, r), list(ax.labels[:c]), data


@template(M, "hard")
def print_summary(rng):
    r, c = rng.randint(3, 6), rng.randint(2, 4)
    name = rng.choice(["activations", "features", "embeddings", "spectrogram", "logits"])
    s = seeded(rng)
    return P(f"Write a quick sanity-check printout for `{name}`. **Print exactly three lines**:\n\n"
             f"1. `shape=<shape> dtype=<dtype>` (e.g. `shape=(4, 3) dtype=float64`)\n"
             f"2. `min=<min> mean=<mean> max=<max>`, each with 3 decimals\n"
             f"3. `col means: ` followed by `np.array2string` of the column means with precision 2",
             f"{name} = np.random.default_rng({s}).normal(size=({r}, {c}))",
             f"x = {name}\nprint(f'shape={{x.shape}} dtype={{x.dtype}}')\n"
             f"print(f'min={{x.min():.3f}} mean={{x.mean():.3f}} max={{x.max():.3f}}')\n"
             f"print('col means: ' + np.array2string(x.mean(axis=0), precision=2))",
             answer=[STDOUT])


# ================================================================= challenge

@template(M, "challenge")
def min_linewidth(rng):
    n = rng.randint(8, 12)
    widths = [rng.randint(1, 9) for _ in range(n)]
    K = rng.randint(2, 4)
    sol = ("w = widths.tolist()\n\ndef lines_needed(L):\n    lines, cur = 1, -1\n    for x in w:\n"
           "        if cur + 1 + x <= L:\n            cur += 1 + x\n        else:\n            lines += 1\n"
           "            cur = x\n    return lines\n\nlo, hi = max(w), sum(w) + len(w) - 1\nwhile lo < hi:\n"
           "    mid = (lo + hi) // 2\n    if lines_needed(mid) <= K:\n        hi = mid\n    else:\n"
           "        lo = mid + 1\nresult = lo")
    tests = [rand_ints("widths", seeded(rng), 1, 13, 20000) + "\nK = 37",
             rand_ints("widths", seeded(rng), 1, 13, 20000) + "\nK = 20000",
             rand_ints("widths", seeded(rng), 5, 6, 20000) + "\nK = 1",
             "widths = np.array([7])\nK = 1"]
    return challenge("Smallest linewidth",
                     "NumPy wraps long arrays to fit `linewidth`. In a simplified model, the printed tokens "
                     "have lengths `widths[i]`, in order. A line holds consecutive tokens separated by single "
                     "spaces, and its length (tokens plus spaces) must not exceed the linewidth. Tokens are "
                     "placed greedily: a token joins the current line if it fits, otherwise it starts a new "
                     "line. Find the **smallest linewidth** that prints everything in at most `K` lines.",
                     f"widths = {arr(widths)}\nK = {K}", sol, tests,
                     "1 ≤ K ≤ n ≤ 2·10⁴, 1 ≤ widths[i] ≤ 12")


@template(M, "challenge")
def max_edgeitems(rng):
    k = rng.randint(2, 3)
    dims = [rng.randint(5, 40) for _ in range(k)]
    E = rng.randint(5, 25)
    size = 1
    for d in dims:
        size *= d
    B = rng.randint(8, max(9, size // 2))
    sol = ("dims_ = [int(d) for d in dims]\n\ndef shown(e):\n    total = 1\n    for d in dims_:\n"
           "        total *= min(d, 2 * e)\n    return total\n\nlo, hi = 0, E\nwhile lo < hi:\n"
           "    mid = (lo + hi + 1) // 2\n    if shown(mid) <= B:\n        lo = mid\n    else:\n"
           "        hi = mid - 1\nresult = lo")
    tests = ["dims = np.array([10**9, 10**9, 3])\nE = 10**9\nB = 10**17",
             "dims = np.array([999999937] * 6)\nE = 10**9\nB = 10**40",
             f"dims = np.random.default_rng({seeded(rng)}).integers(1, 10**9, size=20)\nE = 10**9\nB = 10**60",
             "dims = np.array([1, 1, 1])\nE = 10**9\nB = 0"]
    return challenge("Largest edgeitems that fits",
                     "A summarized array of shape `dims` prints at most `2·e` entries along each axis, "
                     "where `e` is `edgeitems`: along an axis of length `d` it prints `min(d, 2·e)` entries, "
                     "so `prod(min(d, 2·e))` values appear on screen. Find the **largest integer e** with "
                     "`0 ≤ e ≤ E` such that at most `B` values appear (`e = 0` shows nothing).",
                     f"dims = {arr(dims)}\nE = {E}\nB = {B}", sol, tests,
                     "1 ≤ len(dims) ≤ 20, 1 ≤ dims[i] ≤ 10⁹, 1 ≤ E ≤ 10⁹, 0 ≤ B ≤ 10⁶⁰ (use Python ints)",
                     speed="O(len(dims) · log E)")
