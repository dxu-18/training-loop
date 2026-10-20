"""2.18 Plotting & styling.

Plots are graded by inspecting the Matplotlib objects after the user's code (`post`
checks on `plt.gca()` / `plt.gcf()`); styled tables by their rendered HTML with a fixed
uuid. Needs matplotlib + jinja2 in Pyodide; the grader forces the Agg backend.
"""

from ._common import P, challenge, frame, seeded, template

M = "2.18"
PLT = "import matplotlib.pyplot as plt\n"

BARS = "[round(float(p.get_height()), 6) for p in plt.gca().patches]"
WIDTHS = "[round(float(p.get_width()), 6) for p in plt.gca().patches]"
XLABELS = "[t.get_text() for t in plt.gca().get_xticklabels()]"
YLABELS = "[t.get_text() for t in plt.gca().get_yticklabels()]"
LINES = "[[round(float(v), 6) for v in l.get_ydata()] for l in plt.gca().get_lines()]"
HTML = "_.set_uuid('chk').to_html()"


# ================================================================= lite

@template(M, "lite")
def bar_chart(rng):
    f = frame(rng, (4, 6), nums=1)
    a = f.num[0]
    title = rng.choice([f"{f.nums[0].name.capitalize()} by {f.th.noun}", "Leaderboard", "Quarterly report"])
    return P(f"{f.describe()}. Draw a **bar chart** of `{a}` per `{f.key}` with `df.plot(x={f.key!r}, y={a!r}, "
             f"kind='bar', title={title!r}, legend=False)`.", PLT + f.code(),
             f"df.plot(x={f.key!r}, y={a!r}, kind='bar', title={title!r}, legend=False)",
             post=f"{{'title': plt.gca().get_title(), 'heights': {BARS}, 'labels': {XLABELS}}}", require=["plot"])


@template(M, "lite")
def line_chart(rng):
    n = rng.randint(5, 9)
    s = seeded(rng)
    if rng.random() < 0.5:
        return P(f"`loss` holds a training loss for {n} epochs. Plot it as a **line** with `loss.plot()`.",
                 PLT + f"loss = pd.Series(np.round(np.exp(-np.arange({n}) / 3) + np.random.default_rng({s}).random({n}) / 10, 4))",
                 "loss.plot()", post=f"{{'lines': {LINES}}}", require=["plot"])
    return P(f"`curves` holds train and validation loss for {n} epochs as two columns. Plot **both** as lines in one "
             f"call (`curves.plot()`).",
             PLT + f"g = np.random.default_rng({s})\ncurves = pd.DataFrame({{'train': np.round(np.exp(-np.arange({n}) / 3), 4), "
             f"'val': np.round(np.exp(-np.arange({n}) / 4) + g.random({n}) / 10, 4)}})",
             "curves.plot()", post=f"{{'lines': {LINES}, 'legend': [t.get_text() for t in plt.gca().get_legend().get_texts()]}}",
             require=["plot"])


# ================================================================= easy

@template(M, "easy")
def histogram(rng):
    k = rng.choice([4, 5, 8, 10])
    s = seeded(rng)
    return P(f"`latency_ms` holds 200 request latencies. Draw a **histogram** with {k} bins (`.plot.hist(bins={k})`).",
             PLT + f"latency_ms = pd.Series(np.random.default_rng({s}).gamma(2, 50, 200).round(1))",
             f"latency_ms.plot.hist(bins={k})", post=f"{{'heights': {BARS}}}", require=["hist"])


@template(M, "easy")
def scatter_chart(rng):
    f = frame(rng, (5, 8), nums=2, group=False)
    a, b = f.num
    return P(f"{f.describe()}. Draw a **scatter plot** of `{b}` (y) against `{a}` (x) with `df.plot.scatter`.",
             PLT + f.code(), f"df.plot.scatter(x={a!r}, y={b!r})",
             post="{'points': [[round(float(v), 6) for v in p] for p in plt.gca().collections[0].get_offsets()], "
                  "'xlabel': plt.gca().get_xlabel(), 'ylabel': plt.gca().get_ylabel()}", require=["scatter"])


@template(M, "easy")
def sorted_barh(rng):
    f = frame(rng, (7, 10), nums=1)
    a = f.num[0]
    return P(f"{f.describe()}. Plot the mean `{a}` per `{f.group}` as a **horizontal** bar chart, sorted ascending "
             f"(`groupby(...).mean().sort_values().plot.barh()`).", PLT + f.code(),
             f"df.groupby({f.group!r})[{a!r}].mean().sort_values().plot.barh()",
             post=f"{{'widths': {WIDTHS}, 'labels': {YLABELS}}}", require=["barh"])


# ================================================================= medium

@template(M, "medium")
def labelled_axes(rng):
    f = frame(rng, (4, 6), nums=2, group=False)
    a, b = f.num
    title = rng.choice(["Speed vs. quality", "Trade-off", "Model comparison"])
    return P(f"{f.describe()}. Make a scatter plot of `{b}` against `{a}`, then set its title to `{title!r}`, x-label to "
             f"`{f.nums[0].label!r}` and y-label to `{f.nums[1].label!r}` through the returned Axes (`ax.set_title`, "
             f"`ax.set_xlabel`, `ax.set_ylabel`).", PLT + f.code(),
             f"ax = df.plot.scatter(x={a!r}, y={b!r})\nax.set_title({title!r})\nax.set_xlabel({f.nums[0].label!r})\n"
             f"ax.set_ylabel({f.nums[1].label!r})",
             post="{'title': plt.gca().get_title(), 'x': plt.gca().get_xlabel(), 'y': plt.gca().get_ylabel()}",
             require=["set_title"])


@template(M, "medium")
def style_format(rng):
    f = frame(rng, (3, 5), nums=2)
    a, b = f.num
    p = rng.randint(0, 2)
    if rng.random() < 0.5:
        return P(f"{f.describe()}. For a notebook report, return a **Styler** that shows `{a}` with {p} decimals "
                 f"(`df.style.format({{{a!r}: '{{:.{p}f}}'}})`).", f.code(), f"df.style.format({{{a!r}: '{{:.{p}f}}'}})",
                 post=HTML, require=["style"])
    return P(f"{f.describe()}. Return a **Styler** that hides the index (`hide(axis='index')`) and formats `{b}` with "
             f"thousands separators and 1 decimal (`'{{:,.1f}}'`).", f.code(),
             f"df.style.format({{{b!r}: '{{:,.1f}}'}}).hide(axis='index')", post=HTML, require=["style"])


@template(M, "medium")
def subplots_chart(rng):
    n = rng.randint(5, 8)
    s = seeded(rng)
    return P(f"`ts` holds two daily series `a` and `b` ({n} days). Plot them on **separate** stacked subplots in one call "
             f"(`ts.plot(subplots=True, layout=(2, 1), sharex=True)`).",
             PLT + f"ts = pd.DataFrame(np.random.default_rng({s}).normal(size=({n}, 2)).cumsum(axis=0).round(3), columns=['a', 'b'], "
             f"index=pd.date_range('2024-01-01', periods={n}))",
             "ts.plot(subplots=True, layout=(2, 1), sharex=True)",
             post="{'axes': len(plt.gcf().axes), 'lines': [[[round(float(v), 6) for v in l.get_ydata()] for l in ax.get_lines()] "
                  "for ax in plt.gcf().axes]}", require=["subplots"])


@template(M, "medium")
def pie_chart(rng):
    f = frame(rng, (6, 10), nums=1)
    return P(f"{f.describe()}. Draw a **pie chart** of how many rows each `{f.group}` has, with percentage labels "
             f"(`value_counts().plot.pie(autopct='%1.0f%%')`).", PLT + f.code(),
             f"df[{f.group!r}].value_counts().plot.pie(autopct='%1.0f%%')",
             post="{'wedges': len(plt.gca().patches), 'texts': sorted(t.get_text() for t in plt.gca().texts)}",
             require=["pie"])


# ================================================================= hard

@template(M, "hard")
def styled_report(rng):
    f = frame(rng, (4, 6), nums=2)
    a, b = f.num
    return P(f"{f.describe()}. Build a report **Styler** in one chain: format `{a}` and `{b}` with 1 decimal, highlight "
             f"the maximum of each of those two columns (`highlight_max(subset=[...])`), and hide the index.", f.code(),
             f"df.style.format({{{a!r}: '{{:.1f}}', {b!r}: '{{:.1f}}'}}).highlight_max(subset=[{a!r}, {b!r}]).hide(axis='index')",
             post=HTML, require=["highlight_max"])


@template(M, "hard")
def bar_value_labels(rng):
    f = frame(rng, (4, 6), nums=1)
    a = f.num[0]
    return P(f"{f.describe()}. Plot the total `{a}` per `{f.group}` as a bar chart, then write each bar's value on top "
             f"with `ax.bar_label(ax.containers[0], fmt='%.0f')`.", PLT + f.code(),
             f"ax = df.groupby({f.group!r})[{a!r}].sum().plot.bar()\nax.bar_label(ax.containers[0], fmt='%.0f')",
             post=f"{{'heights': {BARS}, 'texts': [t.get_text() for t in plt.gca().texts]}}", require=["bar_label"])


@template(M, "hard")
def save_chart(rng):
    f = frame(rng, (4, 6), nums=2, group=False)
    a, b = f.num
    w, h = rng.choice([(4, 3), (6, 4), (5, 5)])
    dpi = rng.choice([50, 80, 100])
    return P(f"{f.describe()}. Make a scatter plot of `{b}` against `{a}` on a figure of size {w}×{h} inches "
             f"(`figsize=({w}, {h})`), then save it to `'chart.png'` at **{dpi} dpi** (`ax.figure.savefig(..., dpi={dpi})`).",
             PLT + f.code(), f"ax = df.plot.scatter(x={a!r}, y={b!r}, figsize=({w}, {h}))\nax.figure.savefig('chart.png', dpi={dpi})",
             post="import matplotlib.image as mpimg\nmpimg.imread('chart.png').shape[:2]", require=["savefig"])


# ================================================================= challenge

@template(M, "challenge")
def nice_ticks(rng):
    q = rng.randint(3, 5)
    rows = []
    for _ in range(q):
        lo = rng.randint(0, 200)
        rows.append((lo, lo + rng.randint(1, 900), rng.randint(3, 8)))
    sol = ("out = []\nfor lo, hi, m in ranges.itertuples(index=False):\n    exp = 1\n    found = None\n"
           "    while found is None:\n        for mult in (1, 2, 5):\n            step = mult * exp\n"
           "            if -(-hi // step) - lo // step + 1 <= m:\n                found = step\n                break\n"
           "        exp *= 10\n    out.append(found)\nresult = out")

    def big(seed, n_):
        return (f"g = np.random.default_rng({seed})\nlo = g.integers(0, 10**9, {n_})\n"
                f"ranges = pd.DataFrame({{'lo': lo, 'hi': lo + g.integers(1, 10**9, {n_}), 'm': g.integers(2, 12, {n_})}})")

    tests = [big(seeded(rng), 20000), "ranges = pd.DataFrame({'lo': [0, 7], 'hi': [1, 7 + 10**12], 'm': [2, 2]})"]
    return challenge("Nice axis ticks",
                     "A plotting library picks tick steps from the \"nice\" sequence 1, 2, 5, 10, 20, 50, 100, … For each "
                     "row `(lo, hi, m)` of `ranges` (integers, lo < hi), ticks run from `floor(lo / step)·step` to "
                     "`ceil(hi / step)·step` in steps of `step`. Return, for every row, the **smallest nice step** whose "
                     "number of ticks is at most `m`.", f"ranges = pd.DataFrame({rows!r}, columns=['lo', 'hi', 'm'])",
                     sol, tests, "≤ 2·10⁴ ranges, values ≤ 10¹²", speed="O(log(range)) per row")


@template(M, "challenge")
def max_visible_labels(rng):
    n = rng.randint(5, 8)
    boxes = []
    for _ in range(n):
        left = rng.randint(0, 40)
        boxes.append((left, left + rng.randint(2, 10)))
    sol = ("count, end = 0, None\nfor left, right in sorted(zip(labels['left'], labels['right']), key=lambda b: b[1]):\n"
           "    if end is None or left >= end:\n        count += 1\n        end = right\nresult = count")

    def big(seed, n_, span):
        return (f"g = np.random.default_rng({seed})\nleft = g.integers(0, {span}, {n_})\n"
                f"labels = pd.DataFrame({{'left': left, 'right': left + g.integers(1, 1000, {n_})}})")

    tests = [big(seeded(rng), 20000, 10**6), big(seeded(rng), 20000, 1000), "labels = pd.DataFrame({'left': [0, 5], 'right': [5, 9]})"]
    return challenge("Labels that don't collide",
                     "A scatter plot wants to label its points, but labels can't overlap. Each row of `labels` is a "
                     "label's horizontal extent `[left, right)` in pixels; two labels collide if their extents overlap "
                     "(touching is fine). Return the **maximum number of labels** that can be shown at once.",
                     f"labels = pd.DataFrame({boxes!r}, columns=['left', 'right'])", sol, tests, "≤ 2·10⁴ labels",
                     speed="O(n log n): greedily keep the label that ends first")
