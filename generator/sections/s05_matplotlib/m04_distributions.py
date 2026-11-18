"""5.04 Bars, histograms & distributions: bar / barh with labels and error bars, grouped and stacked bars,
histograms (density, step, stacked, cumulative), box and violin plots, ECDFs, hexbin and 2-D histograms."""

from ._common import P, catalog, categories, challenge, classes, curves, lit, seeded, template

M = "5.04"
BARS = "fc.bars(fc.ax())"
HEIGHTS = "fc.heights(fc.ax())"
LINESXY = "[fc.r(l.get_xydata(), 4) for l in fc.ax().get_lines()]"
TICKS = "fc.tick_labels(fc.ax(), 'x')"
ERR = ("[fc.r(c.errorbar.lines[2][0].get_segments(), 4) for c in fc.ax().containers "
       "if getattr(c, 'errorbar', None) is not None]")
SEGS = "[fc.r(c.get_segments(), 4) for c in fc.ax().collections if type(c).__name__ == 'LineCollection']"


def _groups04(rng, k=(3, 4), n=(30, 60)):
    """`groups`: a list of k 1-D samples (different centres / spreads) and `names`."""
    k_ = rng.randint(*k)
    n_ = rng.randint(*n)
    s = seeded(rng)
    names = rng.choice([["SGD", "Adam", "AdamW", "Lion"], ["small", "base", "large", "xl"],
                        ["fold 1", "fold 2", "fold 3", "fold 4"], ["ResNet", "ViT", "MLP", "CNN"]])[:k_]
    code = (f"g = np.random.default_rng({s})\ngroups = [np.round(g.normal(i, 1 + 0.3 * i, {n_}), 3) for i in range({k_})]\n"
            f"names = {lit(names)}")
    return code, f"`groups` holds {k_} samples of {n_} values each and `names` their names", k_, names


def _data04(rng, n=(150, 400)):
    n_ = rng.randint(*n)
    s = seeded(rng)
    mu, sd = rng.choice([(0, 1), (5, 2), (-1, 0.5), (10, 3)])
    return (f"data = np.round(np.random.default_rng({s}).normal({mu}, {sd}, {n_}), 3)",
            f"`data` holds {n_} samples from a normal distribution (mean {mu}, std {sd})", n_, mu, sd)


# ================================================================= lite

@template(M, "lite")
def bar_basics(rng):
    code, desc, names, vals, th, m = categories(rng)
    k = len(names)
    col = rng.choice(["C2", "tab:purple", "0.5", "#8c564b"])
    w = rng.choice([0.4, 0.6, 0.9])
    pad = rng.choice([2, 3, 5])
    items = [
        ("bar", "Draw a vertical bar chart of `values` with one bar per name (`ax.bar(names, values)`).",
         "fig, ax = plt.subplots()\nax.bar(names, values)", {"post": f"{BARS}, {TICKS}"}),
        ("barh", "Draw a **horizontal** bar chart of `values`, one bar per name (`ax.barh`).",
         "fig, ax = plt.subplots()\nax.barh(names, values)", {"post": f"{BARS}, fc.tick_labels(fc.ax(), 'y')"}),
        ("color", f"Draw `ax.bar(names, values)` with every bar in the color {col!r}.",
         f"fig, ax = plt.subplots()\nax.bar(names, values, color={col!r})", {"post": "fc.bars(fc.ax(), 6, 'color')"}),
        ("color", f"Draw `ax.bar(names, values)` with one color per bar: `['C0', 'C1', …, 'C{k - 1}']` (pass a list).",
         f"fig, ax = plt.subplots()\nax.bar(names, values, color=[f'C{{i}}' for i in range({k})])",
         {"post": "fc.bars(fc.ax(), 6, 'color')"}),
        ("edgecolor", "Draw `ax.bar(names, values)` with black bar outlines (`edgecolor='black'`).",
         "fig, ax = plt.subplots()\nax.bar(names, values, edgecolor='black')", {"post": "fc.bars(fc.ax(), 6, 'edgecolor')"}),
        ("width", f"Draw `ax.bar(names, values)` with bars {w} wide (the default is 0.8 of the slot).",
         f"fig, ax = plt.subplots()\nax.bar(names, values, width={w})", {"post": BARS}),
        ("bar_label", "Draw the bars and write each value on top of its bar with `ax.bar_label(bars)` (keep the "
                      "container `bars = ax.bar(...)`).",
         "fig, ax = plt.subplots()\nbars = ax.bar(names, values)\nax.bar_label(bars)", {"post": f"{BARS}, fc.texts()"}),
        ("padding", f"Label the bars with their values as whole numbers (`fmt='%d'`), {pad} points above the bars "
                    f"(`padding={pad}`).", f"fig, ax = plt.subplots()\nbars = ax.bar(names, values)\n"
                                         f"ax.bar_label(bars, fmt='%d', padding={pad})",
         {"post": "fc.texts(), [t.xyann for t in fc.ax().texts]", "require": ["bar_label", "fmt", "padding"]}),
        ("set_xticks", "Draw the bars at numeric positions `xpos = np.arange(len(names))` and label the ticks with the "
                       "names afterwards (`ax.set_xticks(xpos, names)`).",
         "fig, ax = plt.subplots()\nxpos = np.arange(len(names))\nax.bar(xpos, values)\nax.set_xticks(xpos, names)",
         {"post": f"{BARS}, fc.ticks(fc.ax(), 'x')", "require": ["arange", "set_xticks"]}),
        ("xerr", f"Draw horizontal bars with error bars of ±`errs` (`xerr=errs`) and caps of size 3 (`capsize=3`).",
         "fig, ax = plt.subplots()\nax.barh(names, values, xerr=errs, capsize=3)",
         {"post": f"{BARS}, {ERR}, fc.lines(fc.ax(), 'ms', data=False)", "require": ["barh", "xerr", "capsize"],
          "setup": f"errs = np.round(np.array(values) * 0.1, 2)"}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "lite")
def hist_basics(rng):
    code, desc, n, mu, sd = _data04(rng)
    b = rng.choice([10, 20, 30])
    col = rng.choice(["C1", "tab:green", "gray"])
    lo, hi = mu - sd, mu + 2 * sd
    items = [
        ("hist", f"Draw a histogram of `data` with {b} bins.", f"fig, ax = plt.subplots()\nax.hist(data, bins={b})",
         {"post": BARS, "require": ["hist", "bins"]}),
        ("density", f"Draw a histogram of `data` with {b} bins **normalized** to a probability density (`density=True`: the "
                    f"bar areas sum to 1).", f"fig, ax = plt.subplots()\nax.hist(data, bins={b}, density=True)",
         {"post": BARS}),
        ("histtype", f"Draw `data` as an outline-only histogram (`histtype='step'`) with {b} bins.",
         f"fig, ax = plt.subplots()\nax.hist(data, bins={b}, histtype='step')", {"post": "fc.patches(fc.ax())"}),
        ("edgecolor", f"Histogram of `data` ({b} bins) in {col!r}, half transparent (`alpha=0.5`), with white bar edges.",
         f"fig, ax = plt.subplots()\nax.hist(data, bins={b}, color={col!r}, alpha=0.5, edgecolor='white')",
         {"post": "fc.bars(fc.ax(), 6, 'color', 'edgecolor', 'alpha')", "require": ["color", "alpha", "edgecolor"]}),
        ("cumulative", f"Draw a **cumulative** histogram of `data` ({b} bins, `cumulative=True`): each bar counts the "
                       f"values up to its right edge.", f"fig, ax = plt.subplots()\nax.hist(data, bins={b}, cumulative=True)",
         {"post": HEIGHTS}),
        ("orientation", f"Draw the histogram of `data` ({b} bins) **sideways**, bars growing to the right "
                        f"(`orientation='horizontal'`).",
         f"fig, ax = plt.subplots()\nax.hist(data, bins={b}, orientation='horizontal')", {"post": BARS}),
        ("range", f"Histogram only the values between {lo} and {hi}: {b} bins over `range=({lo}, {hi})` (values outside "
                  f"are ignored).", f"fig, ax = plt.subplots()\nax.hist(data, bins={b}, range=({lo}, {hi}))", {"post": BARS}),
        ("bins", f"Pass the bin **edges** yourself: `bins=np.linspace({lo}, {hi}, 9)` (8 bins).",
         f"fig, ax = plt.subplots()\nax.hist(data, bins=np.linspace({lo}, {hi}, 9))", {"post": BARS, "require": ["hist", "linspace"]}),
        ("hist", f"`ax.hist` returns `(counts, edges, patches)`. Draw {b} bins and return `[counts.sum(), len(edges), "
                 f"len(patches)]`.",
         f"fig, ax = plt.subplots()\ncounts, edges, patches = ax.hist(data, bins={b})\nresult = [counts.sum(), len(edges), len(patches)]"),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "lite")
def distribution_basics(rng):
    code, desc, k, names = _groups04(rng)
    gs = rng.choice([10, 15, 20])
    s = seeded(rng)
    items = [
        ("boxplot", "Draw one box plot per sample in `groups` (`ax.boxplot(groups)`).",
         "fig, ax = plt.subplots()\nax.boxplot(groups)", {"post": LINESXY}),
        ("tick_labels", "Draw box plots of `groups` labelled with `names` under the boxes (`tick_labels=names`).",
         "fig, ax = plt.subplots()\nax.boxplot(groups, tick_labels=names)", {"post": f"{LINESXY}, {TICKS}"}),
        ("showmeans", "Draw box plots of `groups` that also mark each mean (`showmeans=True`).",
         "fig, ax = plt.subplots()\nax.boxplot(groups, showmeans=True)", {"post": LINESXY}),
        ("patch_artist", "Draw **filled** boxes (`patch_artist=True`: the boxes become patches you can color).",
         "fig, ax = plt.subplots()\nax.boxplot(groups, patch_artist=True)",
         {"post": f"[type(p).__name__ for p in fc.ax().patches], {LINESXY}"}),
        ("notch", "Draw **notched** box plots of `groups` (`notch=True`): the notch shows a confidence interval for the "
                  "median.", "fig, ax = plt.subplots()\nax.boxplot(groups, notch=True)", {"post": LINESXY}),
        ("orientation", "Draw the box plots **horizontally** (`orientation='horizontal'`).",
         "fig, ax = plt.subplots()\nax.boxplot(groups, orientation='horizontal')", {"post": LINESXY}),
        ("violinplot", "Draw a violin plot of `groups` (`ax.violinplot`).", "fig, ax = plt.subplots()\nax.violinplot(groups)",
         {"post": f"len(fc.ax().collections), {SEGS}"}),
        ("showmedians", "Draw violins of `groups` that mark both the mean and the median (`showmeans=True, "
                        "showmedians=True`).", "fig, ax = plt.subplots()\nax.violinplot(groups, showmeans=True, showmedians=True)",
         {"post": f"len(fc.ax().collections), {SEGS}", "require": ["violinplot", "showmeans", "showmedians"]}),
        ("ecdf", "Draw the empirical CDF of the first sample, `groups[0]` (`ax.ecdf`).",
         "fig, ax = plt.subplots()\nax.ecdf(groups[0])", {"post": "fc.lines(fc.ax(), 'drawstyle')"}),
        ("hexbin", f"Show the density of the 2-D points `xy` with hexagonal bins: `ax.hexbin(xy[:, 0], xy[:, 1], "
                   f"gridsize={gs})`.", f"fig, ax = plt.subplots()\nax.hexbin(xy[:, 0], xy[:, 1], gridsize={gs})",
         {"post": "fc.r(np.ma.filled(np.ma.asarray(fc.ax().collections[0].get_array(), dtype=float), np.nan), 4)", "require": ["hexbin", "gridsize"]}),
        ("hist2d", f"Show the density of `xy` as a 2-D histogram with {gs} × {gs} bins (`ax.hist2d`).",
         f"fig, ax = plt.subplots()\nax.hist2d(xy[:, 0], xy[:, 1], bins={gs})",
         {"post": "fc.r(np.ma.filled(np.ma.asarray(fc.ax().collections[0].get_array(), dtype=float), np.nan), 4)", "require": ["hist2d", "bins"]}),
    ]
    xy = f"xy = np.round(np.random.default_rng({s}).multivariate_normal([0, 0], [[1, 0.6], [0.6, 1]], 300), 3)"
    return catalog(rng, items, f"{code}\n{xy}", f"{desc}; `xy` holds 300 correlated 2-D points.")


# ================================================================= easy

@template(M, "easy")
def bar_readback(rng):
    code, desc, names, vals, th, m = categories(rng, k=(4, 6))
    k = len(names)
    i = rng.randrange(k)
    w = rng.choice([0.5, 0.6, 0.7])
    items = [
        ("get_height", "Draw `bars = ax.bar(names, values)` and read the heights back from the bars "
                       "(`[b.get_height() for b in bars]`).",
         "fig, ax = plt.subplots()\nbars = ax.bar(names, values)\nresult = [b.get_height() for b in bars]"),
        ("get_x", f"Bars are centred on their positions: draw `ax.bar(np.arange({k}), values, width={w})` and return "
                  f"the **left edge** of bar {i} (`get_x()`) rounded to 4 decimals.",
         f"fig, ax = plt.subplots()\nbars = ax.bar(np.arange({k}), values, width={w})\nresult = round(bars[{i}].get_x(), 4)"),
        ("bar_label", "`ax.bar_label` returns the text objects it adds. Label the bars with `fmt='%.1f'` and return their "
                      "texts.", "fig, ax = plt.subplots()\nbars = ax.bar(names, values)\nlabels = ax.bar_label(bars, fmt='%.1f')\n"
                                "result = [t.get_text() for t in labels]"),
        ("containers", "Each `bar` call adds a container to `ax.containers`. Draw `values` and then `values` again "
                       "stacked on top (`bottom=values`), and return `[len(ax.containers), len(ax.patches)]`.",
         "fig, ax = plt.subplots()\nax.bar(names, values)\nax.bar(names, values, bottom=values)\nresult = [len(ax.containers), len(ax.patches)]"),
        ("set_color", f"Highlight the largest value: draw the bars in light grey (`color='0.7'`), then recolor the "
                      f"tallest bar 'C3' through its handle (`bars[j].set_color('C3')`).",
         "fig, ax = plt.subplots()\nbars = ax.bar(names, values, color='0.7')\nbars[int(np.argmax(values))].set_color('C3')",
         {"post": "fc.bars(fc.ax(), 6, 'color')", "require": ["set_color", "argmax"]}),
        ("barh", "Draw horizontal bars sorted from smallest (bottom) to largest (top): sort `values` and reorder "
                 "`names` the same way, then `ax.barh(...)`. Return the y tick labels from bottom to top.",
         "order = np.argsort(values)\nfig, ax = plt.subplots()\nax.barh([names[j] for j in order], np.array(values)[order])\n"
         "result = [t.get_text() for t in ax.get_yticklabels()]", {"post": f"_, {BARS}"}),
        ("bar_label", f"Label every bar with its **share of the total** in percent, one decimal and a % sign: pass "
                      f"the strings yourself with `ax.bar_label(bars, labels=[...])`.",
         "fig, ax = plt.subplots()\nbars = ax.bar(names, values)\ntotal = sum(values)\n"
         "ax.bar_label(bars, labels=[f'{100 * v / total:.1f}%' for v in values])", {"post": "fc.texts()"}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "easy")
def grouped_and_stacked(rng):
    s = seeded(rng)
    k = rng.randint(3, 5)
    cats = rng.choice([["A", "B", "C", "D", "E"], ["mon", "tue", "wed", "thu", "fri"], ["cat", "dog", "bird", "fish", "frog"]])[:k]
    y1, y2 = rng.choice([("2023", "2024"), ("train", "val"), ("before", "after")])
    w = rng.choice([0.35, 0.38, 0.4])
    setup = (f"g = np.random.default_rng({s})\ncats = {lit(cats)}\na = g.integers(5, 50, {k})\nb = g.integers(5, 50, {k})")
    items = [
        ("bar", f"Grouped bars: series `a` ({y1!r}) and `b` ({y2!r}) side by side for each category. Use "
                f"`xpos = np.arange(len(cats))`, width {w}, the first series at `xpos - {w} / 2`, the second at "
                f"`xpos + {w} / 2`; label the ticks with `cats` and add a legend.",
         f"fig, ax = plt.subplots()\nxpos, w = np.arange(len(cats)), {w}\nax.bar(xpos - w / 2, a, w, label={y1!r})\n"
         f"ax.bar(xpos + w / 2, b, w, label={y2!r})\nax.set_xticks(xpos, cats)\nax.legend()",
         {"post": f"{BARS}, fc.ticks(fc.ax(), 'x'), fc.legend()", "require": ["bar", "set_xticks", "legend"]}),
        ("bottom", f"Stacked bars: draw `a` ({y1!r}) and put `b` ({y2!r}) on top of it with `bottom=a`; legend.",
         f"fig, ax = plt.subplots()\nax.bar(cats, a, label={y1!r})\nax.bar(cats, b, bottom=a, label={y2!r})\nax.legend()",
         {"post": f"{BARS}, fc.legend()"}),
        ("bar_label", "Stacked bars `a` + `b` (by category), with the **total** written above each stack. By default "
                      "`bar_label` writes each segment's own value, so label only the top container and pass the "
                      "totals yourself: `ax.bar_label(top, labels=[str(v) for v in a + b])`.",
         "fig, ax = plt.subplots()\nax.bar(cats, a)\ntop = ax.bar(cats, b, bottom=a)\nax.bar_label(top, labels=[str(v) for v in a + b])",
         {"post": f"{BARS}, fc.texts()", "require": ["bottom", "bar_label"]}),
        ("barh", f"Horizontal stacked bars: `ax.barh(cats, a)` then `b` stacked to the right of it (`left=a`).",
         "fig, ax = plt.subplots()\nax.barh(cats, a)\nax.barh(cats, b, left=a)", {"post": BARS, "require": ["barh", "left"]}),
        ("bar_label", f"Grouped bars as in the guide (width {w}), each container labelled with its values "
                      f"(`ax.bar_label(c, padding=2)` for both containers).",
         f"fig, ax = plt.subplots()\nxpos, w = np.arange(len(cats)), {w}\nc1 = ax.bar(xpos - w / 2, a, w)\nc2 = ax.bar(xpos + w / 2, b, w)\n"
         f"ax.bar_label(c1, padding=2)\nax.bar_label(c2, padding=2)\nax.set_xticks(xpos, cats)",
         {"post": f"{BARS}, fc.texts()", "require": ["bar_label", "padding"]}),
        ("bar", "Show the **difference** `b - a` per category as bars colored 'C2' where it's positive and 'C3' where "
                "it's negative (a list of colors).",
         "fig, ax = plt.subplots()\nd = b - a\nax.bar(cats, d, color=['C2' if v > 0 else 'C3' for v in d])",
         {"post": "fc.bars(fc.ax(), 6, 'color')"}),
    ]
    return catalog(rng, items, setup, f"`cats` holds {k} categories; `a` and `b` are two series of counts, one per category.")


@template(M, "easy")
def hist_readback(rng):
    code, desc, n, mu, sd = _data04(rng)
    b = rng.choice([8, 12, 20])
    items = [
        ("hist", f"Draw {b} bins with `density=True` and check the normalization: return the sum of count × bin width "
                 f"(`(counts * np.diff(edges)).sum()`), which is 1.",
         f"fig, ax = plt.subplots()\ncounts, edges, _ = ax.hist(data, bins={b}, density=True)\nresult = (counts * np.diff(edges)).sum()",
         {"tol": 1e-9}),
        ("hist", f"Draw {b} bins and return the **centre** of the tallest bin (the mode of the histogram): `(edges[i] + "
                 f"edges[i + 1]) / 2` with i the index of the largest count.",
         f"fig, ax = plt.subplots()\ncounts, edges, _ = ax.hist(data, bins={b})\ni = counts.argmax()\nresult = (edges[i] + edges[i + 1]) / 2"),
        ("histtype", f"Compare two samples on the **same bins**: draw `data` ({b} bins) and keep its edges, then draw "
                     f"`data * 0.5 + 1` with `bins=edges` as a step outline (`histtype='step'`, linewidth 2).",
         f"fig, ax = plt.subplots()\ncounts, edges, patches = ax.hist(data, bins={b})\n"
         f"ax.hist(data * 0.5 + 1, bins=edges, histtype='step', linewidth=2)",
         {"post": "fc.patches(fc.ax())", "require": ["histtype", "linewidth"]}),
        ("get_height", f"Draw {b} bins and return the bar heights read back from the **patches** "
                       f"(`[p.get_height() for p in patches]`), which equal `counts`.",
         f"fig, ax = plt.subplots()\ncounts, edges, patches = ax.hist(data, bins={b})\nresult = [p.get_height() for p in patches]"),
        ("set_facecolor", f"Draw {b} bins and color the bars whose **left edge** is at or above the mean of `data` "
                          f"'C3' (the others keep the default): loop over `zip(edges, patches)` and use "
                          f"`p.set_facecolor`.",
         f"fig, ax = plt.subplots()\ncounts, edges, patches = ax.hist(data, bins={b})\nfor left, p in zip(edges, patches):\n"
         f"    if left >= data.mean():\n        p.set_facecolor('C3')", {"post": "fc.bars(fc.ax(), 6, 'color')"}),
        ("cumulative", f"Draw a cumulative, normalized histogram (`cumulative=True, density=True`, {b} bins) and return "
                       f"the height of its last bar (1.0: all the data).",
         f"fig, ax = plt.subplots()\ncounts, edges, _ = ax.hist(data, bins={b}, cumulative=True, density=True)\nresult = counts[-1]",
         {"tol": 1e-9}),
        ("hist", "Integer data needs bins centred on the integers: draw `np.round(data)` with edges at every half "
                 "integer (`np.arange(lo - 0.5, hi + 1.5)` with `lo, hi` the min and max of the rounded data) and "
                 "return the number of bins.",
         "r = np.round(data)\nlo, hi = r.min(), r.max()\nfig, ax = plt.subplots()\ncounts, edges, _ = ax.hist(r, bins=np.arange(lo - 0.5, hi + 1.5))\n"
         "result = len(counts)", {"post": f"_, {BARS}"}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "easy")
def box_violin_readback(rng):
    code, desc, k, names = _groups04(rng)
    i = rng.randrange(k)
    items = [
        ("boxplot", f"`ax.boxplot` returns a dict of artist lists. Draw `groups` and return the median line's y value "
                    f"for sample {i} (`out['medians'][{i}].get_ydata()[0]`); compare it with `np.median(groups[{i}])`.",
         f"fig, ax = plt.subplots()\nout = ax.boxplot(groups)\nresult = out['medians'][{i}].get_ydata()[0]"),
        ("boxplot", f"Draw the box plots and return the whisker ends of sample {i}: the lower whisker is "
                    f"`out['whiskers'][{2 * i}]`, the upper one `out['whiskers'][{2 * i + 1}]`; each line runs from the "
                    f"box to the whisker end (`get_ydata()[1]`). Return `[low end, high end]`.",
         f"fig, ax = plt.subplots()\nout = ax.boxplot(groups)\nresult = [out['whiskers'][{2 * i}].get_ydata()[1], out['whiskers'][{2 * i + 1}].get_ydata()[1]]"),
        ("boxplot", "Return how many outliers (fliers) each box plot shows: `len(f.get_ydata())` for each line in "
                    "`out['fliers']`.", "fig, ax = plt.subplots()\nout = ax.boxplot(groups)\n"
                                          "result = [len(f.get_ydata()) for f in out['fliers']]"),
        ("violinplot", "`ax.violinplot` returns a dict too. Draw `groups` with `showmedians=True` and return the "
                       "number of violin bodies and the median y values (`parts['cmedians'].get_segments()`: one "
                       "horizontal segment per violin).",
         "fig, ax = plt.subplots()\nparts = ax.violinplot(groups, showmedians=True)\n"
         "result = [len(parts['bodies']), [s[0][1] for s in parts['cmedians'].get_segments()]]", {"tol": 1e-9}),
        ("ecdf", f"Draw `line = ax.ecdf(groups[{i}])` and return the y value of its last point (1.0) and the number of "
                 f"points the line has (one more than the samples: it starts at 0).",
         f"fig, ax = plt.subplots()\nline = ax.ecdf(groups[{i}])\nresult = [line.get_ydata()[-1], len(line.get_ydata())]"),
        ("set_facecolor", "Fill each box with its own color: draw with `patch_artist=True` and give box i the color "
                          "`f'C{i}'` through `out['boxes'][i].set_facecolor(...)`.",
         "fig, ax = plt.subplots()\nout = ax.boxplot(groups, patch_artist=True)\nfor i, box in enumerate(out['boxes']):\n"
         "    box.set_facecolor(f'C{i}')", {"post": "[fc.hexcolor(p.get_facecolor()) for p in fc.ax().patches]",
                                             "require": ["patch_artist", "set_facecolor"]}),
        ("set_facecolor", "Color the violins: draw `parts = ax.violinplot(groups)` and give body i the face color "
                          "`f'C{i}'` and alpha 0.6.",
         "fig, ax = plt.subplots()\nparts = ax.violinplot(groups)\nfor i, body in enumerate(parts['bodies']):\n"
         "    body.set_facecolor(f'C{i}')\n    body.set_alpha(0.6)",
         {"post": "[(fc.hexcolor(c.get_facecolor()), c.get_alpha()) for c in fc.ax().collections[:len(groups)]]",
          "require": ["violinplot", "set_facecolor", "set_alpha"]}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "easy")
def pyplot_to_explicit_dists(rng):
    code, desc, n, mu, sd = _data04(rng)
    b = rng.choice([10, 15, 25])
    items = [
        ("hist", f"Rewrite with the explicit API (`fig, ax = plt.subplots()`, `ax.` methods):\n```python\n"
                 f"plt.hist(data, bins={b}, color='C1')\nplt.xlabel('value')\nplt.title('distribution')\n```",
         f"fig, ax = plt.subplots()\nax.hist(data, bins={b}, color='C1')\nax.set_xlabel('value')\nax.set_title('distribution')",
         {"post": "fc.bars(fc.ax(), 6, 'color'), fc.labels()", "require": ["hist", "set_xlabel", "set_title"]}),
        ("plt.bar", "Rewrite with **pyplot** only:\n```python\nfig, ax = plt.subplots()\nax.bar(['lo', 'hi'], "
                    "[(data < data.mean()).sum(), (data >= data.mean()).sum()])\nax.set_ylabel('count')\n```",
         "plt.bar(['lo', 'hi'], [(data < data.mean()).sum(), (data >= data.mean()).sum()])\nplt.ylabel('count')",
         {"post": f"{BARS}, fc.labels()", "require": ["plt.bar|bar", "ylabel"]}),
        ("boxplot", "Rewrite with the explicit API:\n```python\nplt.boxplot([data, data * 2], tick_labels=['x', '2x'])\n"
                    "plt.ylabel('value')\n```",
         "fig, ax = plt.subplots()\nax.boxplot([data, data * 2], tick_labels=['x', '2x'])\nax.set_ylabel('value')",
         {"post": f"{LINESXY}, {TICKS}, fc.labels()", "require": ["boxplot", "set_ylabel"]}),
        ("barh", f"Rewrite with the explicit API:\n```python\nc, e, _ = plt.hist(data, bins={b})\nplt.figure()\n"
                 f"plt.barh(range(len(c)), c)\n```\nThat's two figures: create both with `plt.subplots()`.",
         f"fig1, ax1 = plt.subplots()\nc, e, _ = ax1.hist(data, bins={b})\nfig2, ax2 = plt.subplots()\nax2.barh(range(len(c)), c)",
         {"post": "[fc.bars(f.axes[0]) for f in fc.figs()]", "require": ["hist", "barh"]}),
        ("plt.hist2d", "Rewrite with the explicit API, keeping the colorbar (`fig.colorbar(h[3], ax=ax)`):\n```python\n"
                       "h = plt.hist2d(data, np.roll(data, 1), bins=10)\nplt.colorbar()\n```",
         "fig, ax = plt.subplots()\nh = ax.hist2d(data, np.roll(data, 1), bins=10)\nfig.colorbar(h[3], ax=ax)",
         {"post": "fc.r(np.ma.filled(np.ma.asarray(fc.ax(0).collections[0].get_array(), dtype=float), np.nan), 4), len(fc.colorbars())", "require": ["hist2d", "colorbar"]}),
        ("plt.ecdf", "Rewrite with **pyplot** only:\n```python\nfig, ax = plt.subplots()\nax.ecdf(data, label='data')\n"
                     "ax.legend()\n```", "plt.ecdf(data, label='data')\nplt.legend()",
         {"post": "fc.lines(fc.ax(), 'label'), fc.legend()", "require": ["ecdf", "legend"]}),
    ]
    return catalog(rng, items, code, f"{desc}.")


# ================================================================= medium

@template(M, "medium")
def metric_bars(rng):
    s = seeded(rng)
    models = rng.choice([["ResNet", "ViT", "MLP"], ["small", "base", "large"], ["LR", "SVM", "GBM"], ["v1", "v2", "v3"]])
    w = rng.choice([0.25, 0.27])
    fmt = rng.choice(["%.2f", "%.3f"])
    setup = (f"g = np.random.default_rng({s})\nmodels = {lit(models)}\nsplits = ['train', 'val', 'test']\n"
             f"acc = np.round(np.clip(g.uniform(0.7, 0.95, (3, 1)) - [[0.0, 0.05, 0.07]] + 0.02 * g.random((3, 3)), 0, 1), 3)\n"
             f"seeds = np.round(acc[:, 2:3] + 0.01 * g.normal(size=(3, 5)), 3)")
    items = [
        ("bar_label", f"Drill: grouped bars of train / val / test accuracy (columns of `acc`) for the 3 models (rows). "
                      f"Put the models at `xpos = np.arange(3)`, the splits at offsets `-{w}`, `0`, `+{w}` with width "
                      f"{w}, label each split's container with the split name for the legend, write the values on the "
                      f"bars with `fmt={fmt!r}`, put the model names on the ticks, and add a legend.",
         f"fig, ax = plt.subplots()\nxpos, w = np.arange(3), {w}\nfor j, split in enumerate(splits):\n"
         f"    bars = ax.bar(xpos + (j - 1) * w, acc[:, j], w, label=split)\n    ax.bar_label(bars, fmt={fmt!r})\n"
         f"ax.set_xticks(xpos, models)\nax.legend()",
         {"post": f"{BARS}, fc.texts(), fc.ticks(fc.ax(), 'x'), fc.legend()", "require": ["bar", "bar_label", "set_xticks"]}),
        ("barh", "Rank the models by **test** accuracy: horizontal bars sorted with the best model on top, the value "
                 "written at the end of each bar (`bar_label` with `fmt='%.3f'`), and x limits 0 … 1.",
         "order = np.argsort(acc[:, 2])\nfig, ax = plt.subplots()\nbars = ax.barh([models[i] for i in order], acc[order, 2])\n"
         "ax.bar_label(bars, fmt='%.3f')\nax.set_xlim(0, 1)",
         {"post": f"{BARS}, fc.tick_labels(fc.ax(), 'y'), fc.texts(), fc.limits()['xlim']"}),
        ("capsize", "Seeds give uncertainty: `seeds` holds 5 test accuracies per model. Draw the mean per model as a bar "
                    "with an error bar of one standard deviation (`yerr=seeds.std(axis=1)`, `capsize=4`), and the "
                    "model names as tick labels.",
         "fig, ax = plt.subplots()\nax.bar(models, seeds.mean(axis=1), yerr=seeds.std(axis=1), capsize=4)",
         {"post": f"{BARS}, {ERR}, fc.lines(fc.ax(), 'ms', data=False)", "require": ["yerr", "capsize"], "tol": 1e-5}),
        ("set_color", "Highlight the winner: bars of test accuracy per model, all in light grey (`'0.75'`) except the "
                      "best one, colored 'C2', with its value written above it (label only that bar: pass "
                      "`labels=` with empty strings for the others) and the y-axis starting at 0.5.",
         "fig, ax = plt.subplots()\nbest = int(acc[:, 2].argmax())\nbars = ax.bar(models, acc[:, 2], color='0.75')\nbars[best].set_color('C2')\n"
         "ax.bar_label(bars, labels=[f'{v:.3f}' if i == best else '' for i, v in enumerate(acc[:, 2])])\nax.set_ylim(bottom=0.5)",
         {"post": "fc.bars(fc.ax(), 6, 'color'), fc.texts(), fc.limits()['ylim'][0]", "require": ["bar_label"]}),
        ("bar", "The generalization gap: for each model draw `acc[:, 0] - acc[:, 2]` (train minus test) as bars in "
                "'C3', with a black horizontal zero line (`ax.axhline(0, color='k', linewidth=0.8)`) and the y label "
                "'train − test'.",
         "fig, ax = plt.subplots()\nax.bar(models, acc[:, 0] - acc[:, 2], color='C3')\nax.axhline(0, color='k', linewidth=0.8)\n"
         "ax.set_ylabel('train − test')", {"post": "fc.bars(fc.ax(), 6, 'color'), fc.lines(fc.ax(), 'color', 'lw'), fc.labels()['ylabel']",
                                           "require": ["bar", "axhline"]}),
        ("bar_label", "Percent labels **inside** the bars: draw the test accuracy as bars and label them with "
                      "`label_type='center'` and the strings `f'{100 * v:.1f}%'`, in white (`color='white'`).",
         "fig, ax = plt.subplots()\nbars = ax.bar(models, acc[:, 2])\nax.bar_label(bars, labels=[f'{100 * v:.1f}%' for v in acc[:, 2]], "
         "label_type='center', color='white')", {"post": "fc.text_items()", "require": ["label_type"]}),
    ]
    return catalog(rng, items, setup, "`acc[i, j]` is model i's accuracy on split j (`models`, `splits` = train / val / "
                                      "test); `seeds` holds 5 test accuracies per model from different seeds.")


@template(M, "medium")
def hist_comparisons(rng):
    code, desc, k, names = classes(rng, k=(2, 3))
    s = seeded(rng)
    b = rng.choice([15, 20, 25])
    a = rng.choice([0.4, 0.5, 0.6])
    items = [
        ("density", f"Drill: overlay the normalized histograms of the first feature for classes 0 and 1 on **shared** "
                    f"bins: compute `edges = np.histogram_bin_edges(pts[:, 0], bins={b})` once, then draw both with "
                    f"`bins=edges, density=True, alpha={a}`, labelled with the class names; legend.",
         f"edges = np.histogram_bin_edges(pts[:, 0], bins={b})\nfig, ax = plt.subplots()\nfor c in (0, 1):\n"
         f"    ax.hist(pts[labels == c, 0], bins=edges, density=True, alpha={a}, label=class_names[c])\nax.legend()",
         {"post": "fc.bars(fc.ax(), 6, 'alpha'), fc.legend()", "require": ["density", "alpha"]}),
        ("stacked", f"Stacked histogram: pass the per-class samples of the first feature as a **list** (one array per "
                    f"class, in class order) to one `ax.hist` call with `bins={b}`, `stacked=True` and "
                    f"`label=class_names`; legend.",
         f"fig, ax = plt.subplots()\nax.hist([pts[labels == c, 0] for c in range({k})], bins={b}, stacked=True, label=class_names)\nax.legend()",
         {"post": f"{BARS}, fc.legend()", "require": ["stacked", "label"]}),
        ("histtype", f"Outlines are easier to compare than overlapping bars: draw each class' first feature as a step "
                     f"histogram (`histtype='step'`, linewidth 2, the same `bins=np.linspace(-6, 6, {b + 1})`), labelled "
                     f"by class; legend.",
         f"fig, ax = plt.subplots()\nfor c in range({k}):\n    ax.hist(pts[labels == c, 0], bins=np.linspace(-6, 6, {b + 1}), "
         f"histtype='step', linewidth=2, label=class_names[c])\nax.legend()",
         {"post": "fc.patches(fc.ax()), fc.legend()", "require": ["histtype", "linewidth"]}),
        ("hist", "Scores per class: `scores` are a model's predicted probabilities for class 1 and `y` the 0/1 truth. "
                 "Draw two histograms on bins `np.linspace(0, 1, 21)`, negatives (`y == 0`) labelled 'negative' in "
                 "'C0' and positives labelled 'positive' in 'C1', both `alpha=0.6`, plus a dashed black threshold "
                 "line at 0.5 (`ax.axvline(0.5, color='k', linestyle='--')`); legend.",
         "fig, ax = plt.subplots()\nbins = np.linspace(0, 1, 21)\nax.hist(scores[y == 0], bins=bins, color='C0', alpha=0.6, label='negative')\n"
         "ax.hist(scores[y == 1], bins=bins, color='C1', alpha=0.6, label='positive')\nax.axvline(0.5, color='k', linestyle='--')\nax.legend()",
         {"post": "fc.bars(fc.ax(), 6, 'color'), fc.lines(fc.ax(), 'ls'), fc.legend()",
          "setup": f"g2 = np.random.default_rng({s})\ny = g2.integers(0, 2, 200)\nscores = np.round(np.clip(0.3 + 0.4 * y + 0.2 * g2.normal(size=200), 0, 1), 3)"}),
        ("orientation", f"Side-by-side **mirror** histogram (population-pyramid style): on `barh`, draw class 0's counts "
                        f"of the first feature to the left (negative widths) and class 1's to the right, on the shared "
                        f"edges `np.linspace(-6, 6, {b + 1})` (bars at the bin centres, height = bin width).",
         f"edges = np.linspace(-6, 6, {b + 1})\nc0, _ = np.histogram(pts[labels == 0, 0], bins=edges)\n"
         f"c1, _ = np.histogram(pts[labels == 1, 0], bins=edges)\nmid, h = (edges[:-1] + edges[1:]) / 2, np.diff(edges)\n"
         f"fig, ax = plt.subplots()\nax.barh(mid, -c0, height=h)\nax.barh(mid, c1, height=h)",
         {"post": BARS, "require": ["barh", "histogram"]}),
        ("cumulative", f"Compare the classes' first-feature distributions as cumulative, normalized step histograms "
                       f"(`cumulative=True, density=True, histtype='step'`, `bins=np.linspace(-6, 6, {b + 1})`), labelled "
                       f"by class; legend in the upper left.",
         f"fig, ax = plt.subplots()\nfor c in range({k}):\n    ax.hist(pts[labels == c, 0], bins=np.linspace(-6, 6, {b + 1}), cumulative=True, "
         f"density=True, histtype='step', label=class_names[c])\nax.legend(loc='upper left')",
         {"post": "fc.patches(fc.ax()), fc.legend()", "require": ["cumulative", "density", "histtype"]}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "medium")
def distribution_compare(rng):
    code, desc, k, names = _groups04(rng, k=(3, 4))
    items = [
        ("violinplot", f"Drill: compare the {k} samples with box plots (left) and violins (right) side by side on "
                       f"`plt.subplots(1, 2, sharey=True, figsize=(10, 4))`; label both x-axes with `names` (for the "
                       f"violins: `ax.set_xticks(range(1, {k + 1}), names)`), and title them 'box' and 'violin'.",
         f"fig, (a, b) = plt.subplots(1, 2, sharey=True, figsize=(10, 4))\na.boxplot(groups, tick_labels=names)\na.set_title('box')\n"
         f"b.violinplot(groups)\nb.set_xticks(range(1, {k + 1}), names)\nb.set_title('violin')",
         {"post": "[(fc.tick_labels(a, 'x'), a.get_title()) for a in fc.axes()], [fc.r(l.get_xydata(), 4) for l in fc.ax(0).get_lines()], "
                  "len(fc.ax(1).collections), fc.ax(0).get_shared_y_axes().joined(fc.ax(0), fc.ax(1))",
          "require": ["boxplot", "violinplot", "set_xticks"]}),
        ("notch", "Notched, filled box plots with means: `notch=True`, `patch_artist=True`, `showmeans=True`, labels "
                  "`names`; then color box i `f'C{i}'`. Notches that don't overlap suggest different medians.",
         "fig, ax = plt.subplots()\nout = ax.boxplot(groups, notch=True, patch_artist=True, showmeans=True, tick_labels=names)\n"
         "for i, box in enumerate(out['boxes']):\n    box.set_facecolor(f'C{i}')",
         {"post": f"[fc.hexcolor(p.get_facecolor()) for p in fc.ax().patches], [fc.r(p.get_path().vertices, 4) for p in fc.ax().patches], {TICKS}",
          "require": ["notch", "patch_artist", "showmeans"]}),
        ("orientation", "Horizontal box plots (`orientation='horizontal'`) with the sample names on the y-axis "
                        "(`tick_labels=names`) and the x label 'value'.",
         "fig, ax = plt.subplots()\nax.boxplot(groups, orientation='horizontal', tick_labels=names)\nax.set_xlabel('value')",
         {"post": f"{LINESXY}, fc.tick_labels(fc.ax(), 'y'), fc.labels()['xlabel']"}),
        ("violinplot", "Half the story is the points: violins of `groups` (`showmedians=True`) with every sample's "
                       "points scattered on top at its position plus a small horizontal jitter "
                       "(`i + 1 + 0.04 * jit[i]`, `s=5`, `color='k'`, one scatter per sample).",
         "fig, ax = plt.subplots()\nax.violinplot(groups, showmedians=True)\nfor i, gr in enumerate(groups):\n"
         "    ax.scatter(i + 1 + 0.04 * jit[i], gr, s=5, color='k')",
         {"post": f"{SEGS}, [fc.r(np.asarray(c.get_offsets()), 4) for c in fc.ax().collections if type(c).__name__ == 'PathCollection']",
          "setup": "jit = [np.random.default_rng(i).normal(size=len(gr)) for i, gr in enumerate(groups)]"}),
        ("ecdf", "Compare the samples as **ECDFs** on one Axes, each labelled with its name, with a legend in the lower "
                 "right; ECDFs never hide differences behind a bin width.",
         "fig, ax = plt.subplots()\nfor gr, name in zip(groups, names):\n    ax.ecdf(gr, label=name)\nax.legend(loc='lower right')",
         {"post": "fc.lines(fc.ax(), 'label', 'drawstyle'), fc.legend()"}),
        ("boxplot", "Box plots without the outlier markers (`showfliers=False`) and with whiskers reaching the 5th and "
                    "95th percentiles (`whis=(5, 95)`), labelled with `names`.",
         "fig, ax = plt.subplots()\nax.boxplot(groups, showfliers=False, whis=(5, 95), tick_labels=names)",
         {"post": f"{LINESXY}, {TICKS}", "require": ["showfliers", "whis"]}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "medium")
def density_2d(rng):
    s = seeded(rng)
    n = rng.choice([1000, 2000, 3000])
    rho = rng.choice([0.8, 0.5, -0.6])
    gs = rng.choice([15, 20, 30])
    cm = rng.choice(["Blues", "magma", "viridis"])
    setup = f"xy = np.round(np.random.default_rng({s}).multivariate_normal([0, 0], [[1, {rho}], [{rho}, 1]], {n}), 3)"
    hb = "fc.r(np.ma.filled(np.ma.asarray(fc.ax(0).collections[0].get_array(), dtype=float), np.nan), 4)"
    items = [
        ("hexbin", f"Many points overplot each other; bin them instead: `hb = ax.hexbin(xy[:, 0], xy[:, 1], gridsize={gs}, "
                   f"cmap={cm!r}, mincnt=1)` (cells with no points stay empty) and a colorbar labelled 'count'.",
         f"fig, ax = plt.subplots()\nhb = ax.hexbin(xy[:, 0], xy[:, 1], gridsize={gs}, cmap={cm!r}, mincnt=1)\n"
         f"fig.colorbar(hb, ax=ax, label='count')",
         {"post": f"{hb}, fc.ax(0).collections[0].get_cmap().name, fc.colorbars()", "require": ["hexbin", "mincnt", "colorbar"]}),
        ("hist2d", f"A 2-D histogram with {gs} × {gs} bins in {cm!r}, keeping the return value `h` (counts, x edges, y "
                   f"edges, image) and adding a colorbar for `h[3]`. Return the total count `h[0].sum()`.",
         f"fig, ax = plt.subplots()\nh = ax.hist2d(xy[:, 0], xy[:, 1], bins={gs}, cmap={cm!r})\nfig.colorbar(h[3], ax=ax)\nresult = h[0].sum()",
         {"post": f"_, {hb}, len(fc.colorbars())", "require": ["hist2d", "colorbar"]}),
        ("hexbin", f"Counts vary a lot between the centre and the tails, so color them on a log scale: "
                   f"`ax.hexbin(..., gridsize={gs}, bins='log')`. Return the number of hexagons drawn.",
         f"fig, ax = plt.subplots()\nhb = ax.hexbin(xy[:, 0], xy[:, 1], gridsize={gs}, bins='log')\nresult = len(hb.get_array())",
         {"post": f"_, type(fc.ax().collections[0].norm).__name__", "require": ["hexbin", "bins"]}),
        ("hist2d", f"A 2-D histogram over a fixed square: `range=[[-3, 3], [-3, 3]]`, {gs} bins, and `cmin=1` so empty "
                   f"bins are left blank. Return how many bins are drawn non-blank (`np.isfinite(h[0]).sum()`).",
         f"fig, ax = plt.subplots()\nh = ax.hist2d(xy[:, 0], xy[:, 1], bins={gs}, range=[[-3, 3], [-3, 3]], cmin=1)\n"
         f"result = int(np.isfinite(h[0]).sum())", {"require": ["hist2d", "cmin"]}),
        ("hexbin", f"Scatter vs. density side by side on `plt.subplots(1, 2, sharex=True, sharey=True, figsize=(9, 4))`: "
                   f"left a scatter (`s=2`, `alpha=0.3`), right `hexbin` with `gridsize={gs}` and `mincnt=1`; titles "
                   f"'scatter' and 'hexbin'.",
         f"fig, (a, b) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(9, 4))\na.scatter(xy[:, 0], xy[:, 1], s=2, alpha=0.3)\n"
         f"b.hexbin(xy[:, 0], xy[:, 1], gridsize={gs}, mincnt=1)\na.set_title('scatter')\nb.set_title('hexbin')",
         {"post": "[a.get_title() for a in fc.axes()], fc.r(np.ma.filled(np.ma.asarray(fc.ax(1).collections[0].get_array(), dtype=float), np.nan), 4), fc.size()"}),
        ("hexbin", f"hexbin can **aggregate a value** instead of counting: color each hexagon by the mean of "
                   f"`z = xy[:, 0] * xy[:, 1]` over its points (`C=z`, `reduce_C_function=np.mean`, `gridsize={gs}`).",
         f"fig, ax = plt.subplots()\nz = xy[:, 0] * xy[:, 1]\nax.hexbin(xy[:, 0], xy[:, 1], C=z, reduce_C_function=np.mean, gridsize={gs})",
         {"post": "fc.r(np.ma.filled(np.ma.asarray(fc.ax().collections[0].get_array(), dtype=float), np.nan), 4)", "require": ["hexbin", "reduce_C_function"], "tol": 1e-4}),
        ("hexbin", f"hexbin can draw the **marginal** densities too: `ax.hexbin(xy[:, 0], xy[:, 1], gridsize={gs}, "
                   f"marginals=True)` adds bars along the bottom and the left edge. Return how many collections the "
                   f"Axes holds.",
         f"fig, ax = plt.subplots()\nax.hexbin(xy[:, 0], xy[:, 1], gridsize={gs}, marginals=True)\nresult = len(ax.collections)",
         {"post": "_, fc.r(np.ma.filled(np.ma.asarray(fc.ax().collections[0].get_array(), dtype=float), np.nan), 4)", "require": ["hexbin", "marginals"]}),
    ]
    return catalog(rng, items, setup, f"`xy` holds {n} correlated 2-D points (correlation {rho}).")


@template(M, "medium")
def stacked_compositions(rng):
    s = seeded(rng)
    k = rng.randint(3, 5)
    m = rng.randint(4, 6)
    cats = rng.choice([["low", "mid", "high", "max", "extra"], ["cat", "dog", "bird", "fish", "frog"],
                       ["neg", "neu", "pos", "mixed", "other"]])[:k]
    setup = (f"g = np.random.default_rng({s})\nlabels = {lit(cats)}\nsplits = {lit([f'shard {i}' for i in range(m)])}\n"
             f"counts = g.integers(5, 60, ({k}, {m}))")
    items = [
        ("bottom", f"Stack {k} series (rows of `counts`, labelled `labels`) over the {m} splits (columns): loop over "
                   f"the rows, keeping a running `bottom` (start at zeros); legend.",
         f"fig, ax = plt.subplots()\nbottom = np.zeros({m})\nfor row, lab in zip(counts, labels):\n"
         f"    ax.bar(splits, row, bottom=bottom, label=lab)\n    bottom += row\nax.legend()",
         {"post": f"{BARS}, fc.legend()", "require": ["bottom", "legend"]}),
        ("bottom", "Same stack but **normalized** to 100 % per split (divide each column by its total), so the "
                   "composition is comparable; y label '%'.",
         f"fig, ax = plt.subplots()\npct = 100 * counts / counts.sum(axis=0)\nbottom = np.zeros({m})\nfor row, lab in zip(pct, labels):\n"
         f"    ax.bar(splits, row, bottom=bottom, label=lab)\n    bottom += row\nax.set_ylabel('%')",
         {"post": f"{BARS}, fc.labels()['ylabel']", "tol": 1e-6}),
        ("bar_label", "Stacked bars with each segment's count written **inside** it: after each `bar` call, "
                      "`ax.bar_label(c, label_type='center')`.",
         f"fig, ax = plt.subplots()\nbottom = np.zeros({m})\nfor row in counts:\n    c = ax.bar(splits, row, bottom=bottom)\n"
         f"    ax.bar_label(c, label_type='center')\n    bottom += row",
         {"post": f"{BARS}, fc.texts()", "require": ["bar_label", "label_type"]}),
        ("barh", "Horizontal stacked bars (one bar per split, segments growing to the right with `left=`), labelled "
                 "by `labels`, with the legend above the plot in one row (`loc='lower center', "
                 "bbox_to_anchor=(0.5, 1), ncols=len(labels)`).",
         f"fig, ax = plt.subplots()\nleft = np.zeros({m})\nfor row, lab in zip(counts, labels):\n"
         f"    ax.barh(splits, row, left=left, label=lab)\n    left += row\nax.legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncols=len(labels))",
         {"post": f"{BARS}, fc.legend(), fc.ax().get_legend()._ncols", "require": ["barh", "left", "ncols"]}),
        ("bar", f"Grouped instead of stacked: {k} bars per split, width `0.8 / {k}`, bar i of each group offset by "
                f"`(i - ({k} - 1) / 2) * width` from the split's position `np.arange({m})`; split names on the ticks; "
                f"legend with `labels`.",
         f"fig, ax = plt.subplots()\nxpos, w = np.arange({m}), 0.8 / {k}\nfor i, (row, lab) in enumerate(zip(counts, labels)):\n"
         f"    ax.bar(xpos + (i - ({k} - 1) / 2) * w, row, w, label=lab)\nax.set_xticks(xpos, splits)\nax.legend()",
         {"post": f"{BARS}, fc.ticks(fc.ax(), 'x'), fc.legend()", "tol": 1e-6}),
    ]
    return catalog(rng, items, setup, f"`counts[i, j]` is how many examples of label `labels[i]` the j-th data shard "
                                      f"(`splits[j]`) holds.")


@template(M, "medium")
def ecdf_and_quantiles(rng):
    code, desc, n, mu, sd = _data04(rng)
    q = rng.choice([0.5, 0.9, 0.95])
    b = rng.choice([20, 30, 50])
    items = [
        ("ecdf", f"The guide's 'empirical CDF two ways': draw `ax.ecdf(data)` (label 'ecdf') and a cumulative, "
                 f"normalized step histogram of the same data (`bins={b}`, `cumulative=True, density=True, "
                 f"histtype='step'`, label 'hist'); legend.",
         f"fig, ax = plt.subplots()\nax.ecdf(data, label='ecdf')\nax.hist(data, bins={b}, cumulative=True, density=True, histtype='step', label='hist')\n"
         f"ax.legend()", {"post": "fc.lines(fc.ax(), 'label'), fc.patches(fc.ax()), fc.legend()",
                          "require": ["ecdf", "cumulative", "histtype"]}),
        ("ecdf", "The **complementary** ECDF (fraction of values above x) is what you read survival / tail curves "
                 "from: draw `ax.ecdf(data, complementary=True)`.", "fig, ax = plt.subplots()\nax.ecdf(data, complementary=True)",
         {"post": "fc.lines(fc.ax(), 'drawstyle')", "require": ["complementary"]}),
        ("ecdf", f"Read a quantile off the ECDF: draw it, then mark the {int(q * 100)}th percentile with a dashed vertical "
                 f"line at `np.quantile(data, {q})` (`ax.axvline(..., linestyle='--')`) and a dotted horizontal one at "
                 f"{q} (`ax.axhline({q}, linestyle=':')`). Return the quantile.",
         f"fig, ax = plt.subplots()\nax.ecdf(data)\nqv = np.quantile(data, {q})\nax.axvline(qv, linestyle='--')\nax.axhline({q}, linestyle=':')\nresult = qv",
         {"post": "_, fc.lines(fc.ax(), 'ls')", "require": ["ecdf", "quantile"]}),
        ("plot", "Build the ECDF **by hand**: sort the data and plot `np.arange(1, n + 1) / n` against it with "
                 "`drawstyle='steps-post'`; then compare with `ax.ecdf` on a second Axes (`plt.subplots(1, 2, "
                 "sharey=True)`).",
         "fig, (a, b) = plt.subplots(1, 2, sharey=True)\nxs = np.sort(data)\na.plot(xs, np.arange(1, len(xs) + 1) / len(xs), drawstyle='steps-post')\n"
         "b.ecdf(data)", {"post": "[fc.lines(a, 'drawstyle') for a in fc.axes()]", "require": ["sort", "ecdf"]}),
        ("ecdf", "Two ECDFs and their largest vertical gap (the Kolmogorov–Smirnov statistic): draw `ax.ecdf(data)` "
                 "and `ax.ecdf(other)`, then return the KS statistic computed on the sorted union of both samples "
                 "(`F(t) = np.searchsorted(np.sort(s), t, side='right') / len(s)`).",
         "fig, ax = plt.subplots()\nax.ecdf(data)\nax.ecdf(other)\nt = np.sort(np.concatenate([data, other]))\n"
         "F = lambda s: np.searchsorted(np.sort(s), t, side='right') / len(s)\nresult = np.abs(F(data) - F(other)).max()",
         {"post": "_, len(fc.ax().get_lines())", "setup": f"other = np.round(data[::2] * 1.2 + 0.3, 3)",
          "require": ["ecdf", "searchsorted"]}),
        ("ecdf", "A **weighted** ECDF: the second half of `data` comes from a more trusted source and counts double "
                 "(`w = np.r_[np.ones(n // 2), 2 * np.ones(n - n // 2)]` with `n = len(data)`). Draw "
                 "`ax.ecdf(data, weights=w)` and return the line's y value just before the last step "
                 "(`line.get_ydata()[-2]`).",
         "n = len(data)\nw = np.r_[np.ones(n // 2), 2 * np.ones(n - n // 2)]\nfig, ax = plt.subplots()\nline = ax.ecdf(data, weights=w)\n"
         "result = line.get_ydata()[-2]", {"post": "_, fc.lines(fc.ax())", "require": ["ecdf", "weights"], "tol": 1e-9}),
    ]
    return catalog(rng, items, code, f"{desc}.")



@template(M, "medium")
def class_balance_bars(rng):
    code, desc, k, names = classes(rng, k=(3, 4), n=(60, 120))
    s = seeded(rng)
    setup = (f"{code}\ng2 = np.random.default_rng({s})\npred = np.where(g2.random(len(labels)) < 0.75, labels, g2.integers(0, {k}, len(labels)))\n"
             f"folds = np.round(np.clip(0.8 + 0.08 * g2.normal(size=({k}, 5)), 0, 1), 3)")
    items = [
        ("bar_label", "Class balance: bars of how many points each class has (`np.bincount(labels)`), sorted from the "
                      "biggest class to the smallest, with each bar labelled by its share in percent "
                      "(`f'{100 * c / total:.0f}%'`).",
         "cnt = np.bincount(labels)\norder = np.argsort(cnt)[::-1]\nfig, ax = plt.subplots()\n"
         "bars = ax.bar([class_names[i] for i in order], cnt[order])\n"
         "ax.bar_label(bars, labels=[f'{100 * c / cnt.sum():.0f}%' for c in cnt[order]])",
         {"post": f"{BARS}, {TICKS}, fc.texts()", "require": ["bincount", "bar_label"]}),
        ("bar", "Predicted vs. true class frequencies: grouped bars (width 0.4, at `xpos ∓ 0.2`) of "
                "`np.bincount(labels, minlength=k)` labelled 'true' and `np.bincount(pred, minlength=k)` labelled "
                "'pred' (k = number of classes), class names on the ticks, legend.",
         f"k = {k}\nxpos = np.arange(k)\nfig, ax = plt.subplots()\nax.bar(xpos - 0.2, np.bincount(labels, minlength=k), 0.4, label='true')\n"
         f"ax.bar(xpos + 0.2, np.bincount(pred, minlength=k), 0.4, label='pred')\nax.set_xticks(xpos, class_names)\nax.legend()",
         {"post": f"{BARS}, fc.ticks(fc.ax(), 'x'), fc.legend()", "require": ["bincount", "set_xticks"]}),
        ("bar", "Per-class precision and recall from `labels` and `pred`: precision_c = (pred == c and label == c) / "
                "(pred == c), recall_c = … / (label == c). Draw them as grouped bars (width 0.35, at `xpos ∓ 0.175`, "
                "labels 'precision' and 'recall'), y limits 0 … 1, class names on the ticks, legend in the lower right.",
         f"k = {k}\nhit = np.array([np.sum((pred == c) & (labels == c)) for c in range(k)])\n"
         f"prec = hit / np.maximum(np.bincount(pred, minlength=k), 1)\nrec = hit / np.bincount(labels, minlength=k)\n"
         f"xpos = np.arange(k)\nfig, ax = plt.subplots()\nax.bar(xpos - 0.175, prec, 0.35, label='precision')\n"
         f"ax.bar(xpos + 0.175, rec, 0.35, label='recall')\nax.set_ylim(0, 1)\nax.set_xticks(xpos, class_names)\nax.legend(loc='lower right')",
         {"post": f"{BARS}, fc.limits()['ylim'], fc.legend()", "tol": 1e-6}),
        ("xerr", "Cross-validated accuracy per class: `folds[c]` holds 5 fold scores of class c. Horizontal bars of the "
                 "mean with error bars of one standard deviation (`xerr=`, `capsize=3`), class names on the y-axis, x "
                 "limits 0 … 1.",
         "fig, ax = plt.subplots()\nax.barh(class_names, folds.mean(axis=1), xerr=folds.std(axis=1), capsize=3)\nax.set_xlim(0, 1)",
         {"post": f"{BARS}, {ERR}, fc.lines(fc.ax(), 'ms', data=False), fc.limits()['xlim']", "require": ["barh", "xerr", "capsize"],
          "tol": 1e-6}),
        ("set_yscale", "Imbalanced counts read better on a log scale: bars of `np.bincount(labels) ** 3` (pretend the "
                       "imbalance is extreme) with a logarithmic y-axis and black bar edges.",
         "fig, ax = plt.subplots()\nax.bar(class_names, np.bincount(labels) ** 3, edgecolor='black')\nax.set_yscale('log')",
         {"post": "fc.bars(fc.ax(), 6, 'edgecolor'), fc.scales()", "require": ["bar", "set_yscale"]}),
        ("hatch", "Mark the classes the model **over**-predicts (more predictions than true examples) with a hatch "
                  "pattern: bars of the true counts per class, and `hatch='//'` on the bars where "
                  "`np.bincount(pred, minlength=k) > np.bincount(labels, minlength=k)` (`bar.set_hatch`).",
         f"k = {k}\nt, q = np.bincount(labels, minlength=k), np.bincount(pred, minlength=k)\nfig, ax = plt.subplots()\n"
         f"bars = ax.bar(class_names, t)\nfor b, over in zip(bars, q > t):\n    if over:\n        b.set_hatch('//')",
         {"post": f"{BARS}, [p.get_hatch() for p in fc.ax().patches]", "require": ["set_hatch"]}),
    ]
    return catalog(rng, items, setup, f"{desc}; `pred` holds a model's predicted class per point and `folds[c]` 5 "
                                      f"cross-validation scores of class c.")

# ================================================================= hard

@template(M, "hard")
def distribution_functions(rng):
    s = seeded(rng)
    w = rng.choice([0.8, 0.9])
    items = [
        ("bar", f"Write `grouped_bars(ax, data, labels, width={w})`: `data` maps a series name to one value per label; "
                f"draw one bar per series in each group (total group width `width`, bars side by side in the dict's "
                f"order, centred on `np.arange(len(labels))`), label each series' container with its name, put "
                f"`labels` on the x ticks, add a legend and return the list of containers.",
         f"def grouped_bars(ax, data, labels, width={w}):\n    xpos = np.arange(len(labels))\n    w = width / len(data)\n    out = []\n"
         f"    for i, (name, vals) in enumerate(data.items()):\n        out.append(ax.bar(xpos + (i - (len(data) - 1) / 2) * w, vals, w, label=name))\n"
         f"    ax.set_xticks(xpos, labels)\n    ax.legend()\n    return out",
         {"post": "_f, _a = plt.subplots()\n_c = grouped_bars(_a, {'a': [1, 2, 3], 'b': [3, 2, 1], 'c': [2, 2, 2]}, ['x', 'y', 'z'])\n"
                  "_g, _b = plt.subplots()\ngrouped_bars(_b, {'only': [5, 4]}, ['p', 'q'], width=0.5)\n"
                  "(fc.bars(_a), fc.ticks(_a, 'x'), fc.legend(_a), len(_c), fc.bars(_b))", "tol": 1e-6}),
        ("hist", "Write `labelled_hist(ax, x, bins=10)`: draw the histogram, write each **non-zero** count above its bar "
                 "(`ax.bar_label` with `labels=` strings, empty for zero counts), and return the counts as a list of ints.",
         "def labelled_hist(ax, x, bins=10):\n    counts, edges, patches = ax.hist(x, bins=bins)\n"
         "    ax.bar_label(patches, labels=[str(int(c)) if c else '' for c in counts])\n    return [int(c) for c in counts]",
         {"post": f"_x = np.random.default_rng({s}).normal(size=60)\n_f, _a = plt.subplots()\n_r = labelled_hist(_a, _x, bins=12)\n"
                  "(_r, fc.texts(_a), fc.heights(_a))", "require": ["hist", "bar_label"]}),
        ("bottom", "Write `stacked_bars(ax, matrix, categories, series)`: `matrix[i, j]` is series i's value in category "
                   "j; stack the series bottom-up in order (running `bottom`), label each with its series name, and "
                   "return the totals per category (a NumPy array).",
         "def stacked_bars(ax, matrix, categories, series):\n    bottom = np.zeros(len(categories))\n"
         "    for row, name in zip(np.asarray(matrix), series):\n        ax.bar(categories, row, bottom=bottom, label=name)\n"
         "        bottom = bottom + row\n    return bottom",
         {"post": "_f, _a = plt.subplots()\n_t = stacked_bars(_a, [[1, 2], [3, 4], [5, 0]], ['u', 'v'], ['p', 'q', 'r'])\n"
                  "(fc.r(_t), fc.bars(_a), [c.get_label() for c in _a.containers])", "require": ["bottom"]}),
        ("boxplot", "Write `compare_distributions(groups, names, kind='box')`: a new figure with one Axes; `kind='box'` "
                    "draws `boxplot(groups, tick_labels=names)`, `kind='violin'` draws violins with the names on ticks "
                    "1 … n; any other kind raises `ValueError`. Return the figure.",
         "def compare_distributions(groups, names, kind='box'):\n    fig, ax = plt.subplots()\n    if kind == 'box':\n"
         "        ax.boxplot(groups, tick_labels=names)\n    elif kind == 'violin':\n        ax.violinplot(groups)\n"
         "        ax.set_xticks(range(1, len(groups) + 1), names)\n    else:\n        raise ValueError(kind)\n    return fig",
         {"post": f"_g = [np.random.default_rng({s} + i).normal(i, 1, 40) for i in range(3)]\n_f = compare_distributions(_g, ['a', 'b', 'c'])\n"
                  "_v = compare_distributions(_g, ['a', 'b', 'c'], kind='violin')\ntry:\n    compare_distributions(_g, ['a', 'b', 'c'], kind='pie')\n"
                  "    _e = False\nexcept ValueError:\n    _e = True\n"
                  "(fc.tick_labels(_f.axes[0], 'x'), len(_f.axes[0].get_lines()), fc.tick_labels(_v.axes[0], 'x'), "
                  "len(_v.axes[0].collections), _e)", "require": ["boxplot", "violinplot", "ValueError"]}),
        ("hist", "Write `shared_bins_hist(ax, samples, labels, bins=20)`: compute **common** edges for all samples "
                 "(`np.histogram_bin_edges(np.concatenate(samples), bins)`), draw each sample normalized "
                 "(`density=True`) as a step histogram with its label, add a legend, and return the edges.",
         "def shared_bins_hist(ax, samples, labels, bins=20):\n    edges = np.histogram_bin_edges(np.concatenate(samples), bins)\n"
         "    for s, lab in zip(samples, labels):\n        ax.hist(s, bins=edges, density=True, histtype='step', label=lab)\n"
         "    ax.legend()\n    return edges",
         {"post": f"_r = np.random.default_rng({s})\n_f, _a = plt.subplots()\n_e = shared_bins_hist(_a, [_r.normal(size=80), _r.normal(1, 2, 50)], ['p', 'q'], bins=12)\n"
                  "(fc.r(_e, 6), fc.patches(_a), fc.legend(_a))", "require": ["histogram_bin_edges", "density"]}),
    ]
    return catalog(rng, items, "", "")


@template(M, "hard")
def debug_distributions(rng):
    code, desc, names, vals, th, m = categories(rng, k=(3, 5))
    s = seeded(rng)
    extra = (f"g = np.random.default_rng({s})\nsamples = [np.round(g.normal(i, 1, 50), 3) for i in range(3)]\n"
             f"a, b = np.array(values), np.array(values[::-1])")
    items = [
        ("bar", "Grouped bars that **overlap** completely: both series are drawn at the same positions.\n```python\n"
                "fig, ax = plt.subplots()\nax.bar(names, a, label='a')\nax.bar(names, b, label='b')\nax.legend()\n```\n"
                "Put them side by side instead: positions `np.arange(len(names)) ∓ 0.2`, width 0.4, names on the ticks.",
         "fig, ax = plt.subplots()\nxpos = np.arange(len(names))\nax.bar(xpos - 0.2, a, 0.4, label='a')\nax.bar(xpos + 0.2, b, 0.4, label='b')\n"
         "ax.set_xticks(xpos, names)\nax.legend()", {"post": f"{BARS}, fc.ticks(fc.ax(), 'x'), fc.legend()"}),
        ("bottom", "The 'stacked' bars hide each other: the second call doesn't start where the first ends.\n```python\n"
                   "fig, ax = plt.subplots()\nax.bar(names, a)\nax.bar(names, b)\n```\nStack `b` on top of `a`.",
         "fig, ax = plt.subplots()\nax.bar(names, a)\nax.bar(names, b, bottom=a)", {"post": BARS}),
        ("bar_label", "This crashes with *'Rectangle' object has no attribute 'patches'*: `bar_label` wants the whole "
                      "**container** `ax.bar` returns, not one bar.\n```python\nfig, ax = plt.subplots()\n"
                      "bars = ax.bar(names, a)\nax.bar_label(bars[0])\n```\nLabel all bars.",
         "fig, ax = plt.subplots()\nbars = ax.bar(names, a)\nax.bar_label(bars)", {"post": f"{BARS}, fc.texts()"}),
        ("hist", "The two histograms can't be compared: each call picks its **own** bins.\n```python\n"
                 "fig, ax = plt.subplots()\nax.hist(samples[0], bins=15, alpha=0.5)\nax.hist(samples[2], bins=15, alpha=0.5)\n```\n"
                 "Reuse the first call's edges for the second histogram.",
         "fig, ax = plt.subplots()\n_, edges, _ = ax.hist(samples[0], bins=15, alpha=0.5)\nax.hist(samples[2], bins=edges, alpha=0.5)",
         {"post": BARS}),
        ("density", "Different sample sizes make raw counts misleading: the big sample dwarfs the small one.\n```python\n"
                    "fig, ax = plt.subplots()\nedges = np.linspace(-3, 6, 19)\nax.hist(np.concatenate(samples), bins=edges, alpha=0.5)\n"
                    "ax.hist(samples[1], bins=edges, alpha=0.5)\n```\nNormalize both to densities.",
         "fig, ax = plt.subplots()\nedges = np.linspace(-3, 6, 19)\nax.hist(np.concatenate(samples), bins=edges, alpha=0.5, density=True)\n"
         "ax.hist(samples[1], bins=edges, alpha=0.5, density=True)", {"post": BARS, "require": ["density"]}),
        ("tick_labels", "`labels=` was renamed `tick_labels=` in Matplotlib 3.9 (the old name warns and will go away), "
                        "and here the names are also in the wrong order for the data:\n```python\nfig, ax = plt.subplots()\n"
                        "ax.boxplot(samples, labels=['c', 'b', 'a'])\n```\nsample i is called `'abc'[i]`. Fix both.",
         "fig, ax = plt.subplots()\nax.boxplot(samples, tick_labels=['a', 'b', 'c'])", {"post": f"{LINESXY}, {TICKS}"}),
        ("set_xticks", "The violins have no names, and `ax.set_xticks(range(3), ...)` puts them one slot to the left: "
                       "violins sit at positions 1 … n.\n```python\nfig, ax = plt.subplots()\nax.violinplot(samples)\n"
                       "ax.set_xticks(range(3), ['a', 'b', 'c'])\n```\nFix the positions.",
         "fig, ax = plt.subplots()\nax.violinplot(samples)\nax.set_xticks(range(1, 4), ['a', 'b', 'c'])", {"post": "fc.ticks(fc.ax(), 'x')"}),
        ("colorbar", "This crashes with *'tuple' object has no attribute 'cmap'*: "
                     "`hist2d` returns a tuple, and the colorbar needs its **image** (the 4th element).\n```python\n"
                     "fig, ax = plt.subplots()\nh = ax.hist2d(samples[0], samples[1], bins=8)\nfig.colorbar(h, ax=ax)\n```\nFix it.",
         "fig, ax = plt.subplots()\nh = ax.hist2d(samples[0], samples[1], bins=8)\nfig.colorbar(h[3], ax=ax)",
         {"post": "fc.r(np.ma.filled(np.ma.asarray(fc.ax(0).collections[0].get_array(), dtype=float), np.nan), 4), len(fc.colorbars())"}),
        ("barh", "A ranking chart should show the **largest value at the top**, but `barh` draws the first name at the "
                 "bottom:\n```python\nfig, ax = plt.subplots()\norder = np.argsort(a)[::-1]\nax.barh([names[i] for i in order], a[order])\n```\n"
                 "Keep the code but flip the y-axis (`ax.invert_yaxis()`).",
         "fig, ax = plt.subplots()\norder = np.argsort(a)[::-1]\nax.barh([names[i] for i in order], a[order])\nax.invert_yaxis()",
         {"post": f"{BARS}, fc.ax().yaxis_inverted()", "require": ["invert_yaxis"]}),
    ]
    return catalog(rng, items, f"{code}\n{extra}", f"{desc}; `a` is `values` as an array and `b` the same values "
                                                   f"reversed; `samples` holds 3 samples of 50 values.")


@template(M, "hard")
def distribution_drills(rng):
    s = seeded(rng)
    k = rng.choice([4, 4, 5])
    fmt = rng.choice(["%.2f", "%.1f"])
    setup = (f"g = np.random.default_rng({s})\nmodels = ['A', 'B', 'C']\nsplits = ['train', 'val', 'test']\n"
             f"acc = np.round(g.uniform(0.6, 0.99, (3, 3)), 3)\ndists = [np.round(g.normal(i, 1 + 0.25 * i, 80), 3) for i in range({k})]\n"
             f"data = np.round(g.normal(size=500), 3)")
    items = [
        ("bar_label", f"Drill 1: grouped bars for train / val / test accuracy of the 3 models (`acc[model, split]`), "
                      f"width 0.25 with the splits at offsets -0.25, 0, 0.25, value labels (`fmt={fmt!r}`, padding 2), "
                      f"model names on the ticks, a legend with the split names and y limits 0 … 1.1.",
         f"fig, ax = plt.subplots()\nxpos = np.arange(3)\nfor j, sp in enumerate(splits):\n"
         f"    bars = ax.bar(xpos + (j - 1) * 0.25, acc[:, j], 0.25, label=sp)\n    ax.bar_label(bars, fmt={fmt!r}, padding=2)\n"
         f"ax.set_xticks(xpos, models)\nax.legend()\nax.set_ylim(0, 1.1)",
         {"post": f"{BARS}, fc.texts(), fc.ticks(fc.ax(), 'x'), fc.legend(), fc.limits()['ylim']"}),
        ("density", "Drill 2: overlay two normalized histograms with shared bins: `data` and `data * 0.5 + 1`, "
                    "edges from the first call (`bins=30`), the second as a step histogram (linewidth 2); labels "
                    "'a' and 'b', legend.",
         "fig, ax = plt.subplots()\ncounts, edges, _ = ax.hist(data, bins=30, density=True, alpha=0.6, label='a')\n"
         "ax.hist(data * 0.5 + 1, bins=edges, density=True, histtype='step', linewidth=2, label='b')\nax.legend()",
         {"post": f"{BARS}, fc.patches(fc.ax()), fc.legend()"}),
        ("violinplot", f"Drill 3: compare the {k} distributions in `dists` with box plots and violins side by side "
                       f"(`plt.subplots(1, 2, sharey=True)`): boxes with `showmeans=True` and labels 'd0', 'd1', …; "
                       f"violins with `showmedians=True`.",
         f"labels = [f'd{{i}}' for i in range({k})]\nfig, (a, b) = plt.subplots(1, 2, sharey=True)\n"
         f"a.boxplot(dists, showmeans=True, tick_labels=labels)\nb.violinplot(dists, showmedians=True)",
         {"post": "[fc.r(l.get_xydata(), 4) for l in fc.ax(0).get_lines()], fc.tick_labels(fc.ax(0), 'x'), "
                  "[fc.r(c.get_segments(), 4) for c in fc.ax(1).collections if type(c).__name__ == 'LineCollection']"}),
        ("hexbin", "The guide's density row: on `plt.subplots(1, 3, figsize=(12, 3), layout='constrained')` draw "
                   "`ax.ecdf(data)` on the first, a cumulative normalized step histogram of `data` (50 bins) on the "
                   "first too, the hexbin of `xy = g.multivariate_normal([0, 0], [[1, 0.8], [0.8, 1]], 2000)` (`gridsize=30`, "
                   "`cmap='Blues'`, `mincnt=1`) on the second, and its "
                   "`hist2d` (`bins=40`, `cmap='magma'`) with a colorbar on the third.",
         "xy = g.multivariate_normal([0, 0], [[1, 0.8], [0.8, 1]], 2000)\nfig, axs = plt.subplots(1, 3, figsize=(12, 3), layout='constrained')\n"
         "axs[0].ecdf(data)\naxs[0].hist(data, bins=50, cumulative=True, density=True, histtype='step')\n"
         "axs[1].hexbin(xy[:, 0], xy[:, 1], gridsize=30, cmap='Blues', mincnt=1)\nh = axs[2].hist2d(xy[:, 0], xy[:, 1], bins=40, cmap='magma')\n"
         "fig.colorbar(h[3], ax=axs[2])",
         {"post": "fc.lines(fc.ax(0), 'drawstyle'), fc.r(np.ma.filled(np.ma.asarray(fc.ax(1).collections[0].get_array(), dtype=float), np.nan), 4), fc.ax(1).collections[0].get_cmap().name, "
                  "fc.r(np.ma.filled(np.ma.asarray(fc.ax(2).collections[0].get_array(), dtype=float), np.nan), 4), len(fc.colorbars())", "require": ["multivariate_normal", "hexbin", "hist2d"]}),
        ("hist", "The guide's stacked histogram: `ax.hist([data, data + 2], bins=20, stacked=True, label=['a', 'b'])` "
                 "with a legend, then save the figure as `dist.png` at 30 dpi, close it, and return the image shape.",
         "fig, ax = plt.subplots()\nax.hist([data, data + 2], bins=20, stacked=True, label=['a', 'b'])\nax.legend()\n"
         "fig.savefig('dist.png', dpi=30)\nplt.close(fig)\nresult = plt.imread('dist.png').shape",
         {"require": ["stacked", "savefig", "close"]}),
    ]
    return catalog(rng, items, setup, f"`acc[i, j]` is model i's accuracy on split j; `dists` holds {k} samples; `data` "
                                      f"500 normal samples; `g` is a seeded generator.")


@template(M, "hard")
def distribution_specs(rng):
    code, desc, k, names = classes(rng, k=(3, 4))
    s = seeded(rng)
    b = rng.choice([12, 16, 20])
    items = [
        ("subplots", f"A class report on `plt.subplots(2, 2, figsize=(9, 7), layout='constrained')`: top-left bars of "
                     f"the class sizes (`np.bincount(labels)`) with the names on the ticks and counts on the bars; "
                     f"top-right a stacked histogram of feature 0 by class ({b} bins, labels = names, legend); "
                     f"bottom-left box plots of feature 1 per class (tick labels = names); bottom-right a hexbin of all "
                     f"points (`gridsize=12`, `mincnt=1`).",
         f"fig, axs = plt.subplots(2, 2, figsize=(9, 7), layout='constrained')\nbars = axs[0, 0].bar(class_names, np.bincount(labels))\n"
         f"axs[0, 0].bar_label(bars)\naxs[0, 1].hist([pts[labels == c, 0] for c in range({k})], bins={b}, stacked=True, label=class_names)\n"
         f"axs[0, 1].legend()\naxs[1, 0].boxplot([pts[labels == c, 1] for c in range({k})], tick_labels=class_names)\n"
         f"axs[1, 1].hexbin(pts[:, 0], pts[:, 1], gridsize=12, mincnt=1)",
         {"post": "(fc.bars(fc.ax(0)), fc.texts(fc.ax(0)), fc.tick_labels(fc.ax(0), 'x'), fc.bars(fc.ax(1)), fc.legend(fc.ax(1)), "
                  "[fc.r(l.get_xydata(), 4) for l in fc.ax(2).get_lines()], fc.tick_labels(fc.ax(2), 'x'), "
                  "fc.r(np.ma.filled(np.ma.asarray(fc.ax(3).collections[0].get_array(), dtype=float), np.nan), 4))", "require": ["bar_label", "stacked", "boxplot", "hexbin"]}),
        ("barh", f"Per-class accuracy, worst first: `correct` marks which predictions were right. Compute each class' "
                 f"accuracy, draw horizontal bars sorted **ascending** from the top (worst at the top: sort ascending "
                 f"and `ax.invert_yaxis()`), x limits 0 … 1, with `bar_label(fmt='%.2f')`.",
         f"acc = np.array([correct[labels == c].mean() for c in range({k})])\norder = np.argsort(acc)\nfig, ax = plt.subplots()\n"
         f"bars = ax.barh([class_names[i] for i in order], acc[order])\nax.invert_yaxis()\nax.set_xlim(0, 1)\nax.bar_label(bars, fmt='%.2f')",
         {"post": f"{BARS}, fc.tick_labels(fc.ax(), 'y'), fc.texts(), fc.ax().yaxis_inverted()",
          "setup": f"correct = np.random.default_rng({s}).random(len(labels)) < 0.8", "require": ["barh", "invert_yaxis"]}),
        ("violinplot", "Violins of feature 0 per class at x positions 0, 2, 4, … (`positions=`), width 1.5 "
                       "(`widths=1.5`), with class names on those ticks.",
         f"fig, ax = plt.subplots()\npos = np.arange({k}) * 2\nax.violinplot([pts[labels == c, 0] for c in range({k})], positions=pos, widths=1.5)\n"
         f"ax.set_xticks(pos, class_names)",
         {"post": f"{SEGS}, fc.ticks(fc.ax(), 'x')", "require": ["positions", "widths"]}),
        ("hist", "Feature 0's histogram per class as **small multiples**: one row per class "
                 f"(`plt.subplots({k}, 1, sharex=True)`), shared edges `np.linspace(-6, 6, {b + 1})`, each panel "
                 f"in its class color `f'C{{c}}'` with the class name as y label.",
         f"fig, axes = plt.subplots({k}, 1, sharex=True)\nedges = np.linspace(-6, 6, {b + 1})\nfor c, ax in enumerate(axes):\n"
         f"    ax.hist(pts[labels == c, 0], bins=edges, color=f'C{{c}}')\n    ax.set_ylabel(class_names[c])",
         {"post": "[(fc.bars(a, 6, 'color'), a.get_ylabel()) for a in fc.axes()]"}),
        ("violinplot", f"Violin **and** box in one: violins of feature 1 per class (`showextrema=False`), then narrow "
                       f"box plots on top at the same positions 1 … {k} (`widths=0.15`, `showfliers=False`), class names "
                       f"on the ticks.",
         f"fig, ax = plt.subplots()\nd = [pts[labels == c, 1] for c in range({k})]\nax.violinplot(d, showextrema=False)\n"
         f"ax.boxplot(d, widths=0.15, showfliers=False, tick_labels=class_names)",
         {"post": f"len(fc.ax().collections), {LINESXY}, {TICKS}", "require": ["violinplot", "boxplot", "showextrema", "widths"]}),
        ("hist", "Residuals with a reference: `r = pts[:, 0] - pts[:, 0].mean()`; draw a density histogram of `r` "
                 "(20 bins, `alpha=0.5`) and the normal density with the same mean and std on "
                 "`t = np.linspace(r.min(), r.max(), 100)` as a red line, labelled 'normal'; legend.",
         "r = pts[:, 0] - pts[:, 0].mean()\nt = np.linspace(r.min(), r.max(), 100)\nm, sd = r.mean(), r.std()\n"
         "fig, ax = plt.subplots()\nax.hist(r, bins=20, density=True, alpha=0.5)\n"
         "ax.plot(t, np.exp(-((t - m) / sd) ** 2 / 2) / (sd * np.sqrt(2 * np.pi)), color='red', label='normal')\nax.legend()",
         {"post": f"{BARS}, fc.lines(fc.ax(), 'color', 'label', nd=5), fc.legend()", "tol": 1e-5}),
    ]
    return catalog(rng, items, code, f"{desc}.")



@template(M, "hard")
def histogram_math(rng):
    code, desc, n, mu, sd = _data04(rng, n=(200, 500))
    s = seeded(rng)
    items = [
        ("histogram_bin_edges", "Let NumPy choose the bins: draw `ax.hist(data, bins='auto')` and return the number of "
                                "bins it used, checked against `np.histogram_bin_edges(data, bins='auto')`.",
         "fig, ax = plt.subplots()\ncounts, edges, _ = ax.hist(data, bins='auto')\nresult = [len(counts), len(np.histogram_bin_edges(data, bins='auto')) - 1]",
         {"post": f"_, {BARS}", "require": ["histogram_bin_edges"]}),
        ("hist", "The Freedman–Diaconis rule picks the bin **width** `2 · IQR · n^(-1/3)`. Compute it, build edges "
                 "`np.arange(data.min(), data.max() + width, width)` and draw the histogram with them. Return the number "
                 "of bins.",
         "q1, q3 = np.percentile(data, [25, 75])\nwidth = 2 * (q3 - q1) * len(data) ** (-1 / 3)\n"
         "edges = np.arange(data.min(), data.max() + width, width)\nfig, ax = plt.subplots()\ncounts, _, _ = ax.hist(data, bins=edges)\n"
         "result = len(counts)", {"post": f"_, {BARS}", "require": ["percentile", "arange"]}),
        ("weights", "`density=True` makes the **areas** sum to 1; for bar **heights** that sum to 1 (fractions of the "
                    "data per bin), weight every value by 1/n instead: `ax.hist(data, bins=20, weights=np.ones(len(data)) / "
                    "len(data))`. Return the sum of the heights.",
         "fig, ax = plt.subplots()\ncounts, _, _ = ax.hist(data, bins=20, weights=np.ones(len(data)) / len(data))\nresult = counts.sum()",
         {"post": f"_, {BARS}", "require": ["weights"], "tol": 1e-9}),
        ("weights", "Plot a histogram you already have as counts (e.g. from a log): `cnt, edges = np.histogram(data, "
                    "bins=15)`, then draw it with `ax.hist` **without** the raw data, using the trick `ax.hist(edges[:-1], "
                    "bins=edges, weights=cnt)`.",
         "cnt, edges = np.histogram(data, bins=15)\nfig, ax = plt.subplots()\nax.hist(edges[:-1], bins=edges, weights=cnt)",
         {"post": BARS, "require": ["histogram", "weights"]}),
        ("bxp", "Draw a box plot from **precomputed** statistics with `ax.bxp`: build one dict with the keys 'med', "
                "'q1', 'q3', 'whislo', 'whishi' (the 5th and 95th percentiles), 'fliers' (values outside them) and "
                "'label' ('data'), and pass `[stats]`.",
         "q1, med, q3, lo, hi = np.percentile(data, [25, 50, 75, 5, 95])\n"
         "stats = {'med': med, 'q1': q1, 'q3': q3, 'whislo': lo, 'whishi': hi, 'fliers': data[(data < lo) | (data > hi)], 'label': 'data'}\n"
         "fig, ax = plt.subplots()\nax.bxp([stats])", {"post": f"{LINESXY}, {TICKS}", "require": ["bxp", "percentile"]}),
        ("plot", "A kernel density estimate by hand over a density histogram: bandwidth `h = 1.06 · std · n^(-1/5)` "
                 "(Silverman), evaluated on `t = np.linspace(data.min(), data.max(), 200)` as the mean of Gaussian bumps "
                 "`exp(-((t - x_i) / h)² / 2) / (h √(2π))`. Draw `ax.hist(data, bins=30, density=True, alpha=0.4)` and "
                 "the KDE as a black line.",
         "h = 1.06 * data.std() * len(data) ** (-1 / 5)\nt = np.linspace(data.min(), data.max(), 200)\n"
         "kde = np.exp(-((t[:, None] - data[None, :]) / h) ** 2 / 2).mean(axis=1) / (h * np.sqrt(2 * np.pi))\n"
         "fig, ax = plt.subplots()\nax.hist(data, bins=30, density=True, alpha=0.4)\nax.plot(t, kde, color='k')",
         {"post": "fc.lines(fc.ax(), 'color', nd=5), " + BARS, "require": ["density", "plot"], "tol": 1e-5}),
        ("density", "What does `density=True` compute? Draw `ax.hist(data, bins=12, density=True)` and return the "
                    "heights **computed by hand** from `cnt, edges = np.histogram(data, bins=12)`: "
                    "`cnt / (cnt.sum() * np.diff(edges))` (they must match the bars).",
         "fig, ax = plt.subplots()\nax.hist(data, bins=12, density=True)\ncnt, edges = np.histogram(data, bins=12)\n"
         "result = cnt / (cnt.sum() * np.diff(edges))", {"post": f"_, {HEIGHTS}", "require": ["histogram", "diff"]}),
        ("cumulative", "A **reverse** cumulative histogram counts the values at or above each bin's left edge "
                       "(`cumulative=-1`), i.e. 'how many runs reached at least this score'. Draw it with 15 bins and "
                       "return the first and last bar heights.",
         "fig, ax = plt.subplots()\ncounts, _, _ = ax.hist(data, bins=15, cumulative=-1)\nresult = [counts[0], counts[-1]]",
         {"post": f"_, {HEIGHTS}", "require": ["cumulative"]}),
    ]
    return catalog(rng, items, code, f"{desc}.")

# ================================================================= challenge

def _hb_setup(seed: int, n: int, nx: int, ny: int, report: str) -> str:
    return (f"g = np.random.default_rng({seed})\nxy = g.normal(0, 1, ({n}, 2)) * [1.5, 1.0]\nx, y = xy[:, 0], xy[:, 1]\n"
            f"extent = (-3.0, 3.0, -2.0, 2.0)\ngridsize = ({nx}, {ny})\nreport = {report!r}")


@template(M, "challenge")
def hexbin_counts(rng):
    s = seeded(rng)
    report = rng.choice(["nonempty", "histogram", "busiest"])
    n, nx, ny = rng.randint(15, 30), rng.randint(2, 4), rng.randint(2, 3)
    setup = _hb_setup(s, n, nx, ny, report)
    sol = ("nx, ny = gridsize\nxmin, xmax, ymin, ymax = extent\npad = 1e-9 * (xmax - xmin)\nxmin, xmax = xmin - pad, xmax + pad\n"
           "sx, sy = (xmax - xmin) / nx, (ymax - ymin) / ny\nix, iy = (x - xmin) / sx, (y - ymin) / sy\n"
           "ix1, iy1 = np.round(ix).astype(int), np.round(iy).astype(int)\nix2, iy2 = np.floor(ix).astype(int), np.floor(iy).astype(int)\n"
           "ok1 = (0 <= ix1) & (ix1 <= nx) & (0 <= iy1) & (iy1 <= ny)\nok2 = (0 <= ix2) & (ix2 < nx) & (0 <= iy2) & (iy2 < ny)\n"
           "near1 = (ix - ix1) ** 2 + 3 * (iy - iy1) ** 2 < (ix - ix2 - 0.5) ** 2 + 3 * (iy - iy2 - 0.5) ** 2\n"
           "n1 = (nx + 1) * (ny + 1)\ncells = np.concatenate([(ix1 * (ny + 1) + iy1)[near1 & ok1], n1 + (ix2 * ny + iy2)[~near1 & ok2]])\n"
           "counts = np.bincount(cells, minlength=n1 + nx * ny)\n"
           + {"nonempty": "result = int((counts > 0).sum())",
              "histogram": "result = sorted(counts[counts > 0].tolist(), reverse=True)",
              "busiest": ("i = int(counts.argmax())\nif i < n1:\n    cx, cy = i // (ny + 1), i % (ny + 1)\nelse:\n"
                          "    cx, cy = (i - n1) // ny + 0.5, (i - n1) % ny + 0.5\nresult = [xmin + cx * sx, ymin + cy * sy]")}[report])
    tests = [_hb_setup(s + i, nn, a, b, report) for i, (nn, a, b) in ((1, (1000000, 300, 170)), (2, (500000, 2000, 1)))]
    want = {"nonempty": "the number of hexagons that receive at least one point",
            "histogram": "the counts of all non-empty hexagons, sorted from largest to smallest (a list)",
            "busiest": "the centre `[cx, cy]` of the hexagon with the most points (on ties, the one that comes first in "
                       "the numbering above)"}[report]
    return challenge(
        "Hexagonal binning",
        "`ax.hexbin(x, y, gridsize=(nx, ny), extent=extent)` sorts points into hexagons laid out on **two** "
        "rectangular lattices. Pad the x-range: `xmin -= 1e-9·(xmax - xmin)`, `xmax += 1e-9·(xmax - xmin)` (y is not "
        "padded); `sx = (xmax - xmin)/nx`, `sy = (ymax - ymin)/ny`; a point's lattice coordinates are `ix = (x - "
        "xmin)/sx`, `iy = (y - ymin)/sy`. Lattice A has centres at integer `(i, j)`, 0 ≤ i ≤ nx, 0 ≤ j ≤ ny (the "
        "point's candidate is `(round(ix), round(iy))`, rounding halves to even); lattice B has centres at `(i + "
        "0.5, j + 0.5)`, 0 ≤ i < nx, 0 ≤ j < ny (candidate `(floor(ix), floor(iy))`). The point goes to A's candidate "
        "if `(ix - iA)² + 3·(iy - jA)² < (ix - iB - 0.5)² + 3·(iy - jB - 0.5)²`, else to B's — and it is **dropped** "
        "if the chosen candidate lies outside its lattice (even if the other one is inside). Hexagons are numbered "
        "A first (`i·(ny + 1) + j`), then B (`(nx + 1)(ny + 1) + i·ny + j`); a centre `(u, v)` in lattice units is "
        f"at `(xmin + u·sx, ymin + v·sy)` in data units (with the padded xmin). Return {want}. (`report` = {report!r}.)",
        setup, sol, tests, "n ≤ 10^6 points, nx·ny ≤ 10^6", speed="O(n + nx·ny), vectorized")


def _box_setup(seed: int, n: int, G: int, report: str) -> str:
    return (f"g = np.random.default_rng({seed})\nG = {G}\ngroup = np.concatenate([np.arange(G), g.integers(0, G, {n} - G)])\n"
            f"g.shuffle(group)\nvalues = np.round(g.standard_t(3, {n}) * (1 + group % 3) + group, 2)\nreport = {report!r}")


@template(M, "challenge")
def box_stats(rng):
    s = seeded(rng)
    report = rng.choice(["whiskers", "fliers", "notch"])
    n, G = rng.randint(20, 40), rng.randint(2, 4)
    setup = _box_setup(s, n, G, report)
    sol = ("order = np.lexsort((values, group))\nv, gr = values[order], group[order]\nstarts = np.searchsorted(gr, np.arange(G))\n"
           "sizes = np.diff(np.r_[starts, len(v)])\n\ndef pct(p):\n    pos = p / 100 * (sizes - 1)\n    lo = np.floor(pos).astype(int)\n"
           "    hi = np.minimum(lo + 1, sizes - 1)\n    t = pos - lo\n    a, b = v[starts + lo], v[starts + hi]\n"
           "    return a + (b - a) * t\n\nq1, med, q3 = pct(25), pct(50), pct(75)\niqr = q3 - q1\n")
    sol += {"whiskers": ("hi_ok = v <= (q3 + 1.5 * iqr)[gr]\nlo_ok = v >= (q1 - 1.5 * iqr)[gr]\n"
                         "n_hi = np.add.reduceat(hi_ok.astype(int), starts)\nn_lo = np.add.reduceat(lo_ok.astype(int), starts)\n"
                         "top = np.where(n_hi > 0, v[starts + np.maximum(n_hi, 1) - 1], q3)\n"
                         "bot = np.where(n_lo > 0, v[starts + sizes - np.maximum(n_lo, 1)], q1)\n"
                         "result = np.column_stack([np.minimum(bot, q1), np.maximum(top, q3)])"),
            "fliers": ("hi_ok = v <= (q3 + 1.5 * iqr)[gr]\nlo_ok = v >= (q1 - 1.5 * iqr)[gr]\n"
                       "n_hi = np.add.reduceat(hi_ok.astype(int), starts)\nn_lo = np.add.reduceat(lo_ok.astype(int), starts)\n"
                       "top = np.maximum(np.where(n_hi > 0, v[starts + np.maximum(n_hi, 1) - 1], q3), q3)\n"
                       "bot = np.minimum(np.where(n_lo > 0, v[starts + sizes - np.maximum(n_lo, 1)], q1), q1)\n"
                       "out = (v < bot[gr]) | (v > top[gr])\nresult = np.add.reduceat(out.astype(int), starts)"),
            "notch": ("half = 1.57 * iqr / np.sqrt(sizes)\nresult = np.column_stack([med - half, med + half])")}[report]
    tests = [_box_setup(s + i, nn, gg, report) for i, (nn, gg) in ((1, (400000, 1000)), (2, (300000, 150000)))]
    want = {"whiskers": "a G × 2 array of the whisker ends `(low, high)` per group",
            "fliers": "the number of fliers (outliers drawn as points) per group, as an integer array",
            "notch": "a G × 2 array of the notch limits `(median - 1.57·IQR/√n, median + 1.57·IQR/√n)` per group "
                     "(n = group size)"}[report]
    return challenge(
        "Box-plot statistics",
        "`ax.boxplot` computes, for each group: the quartiles Q1, median, Q3 (linear-interpolation percentiles, like "
        "`np.percentile`), IQR = Q3 − Q1; the **upper whisker** ends at the largest value ≤ Q3 + 1.5·IQR (but never "
        "below Q3: if there's no such value or it's < Q3, the whisker ends at Q3), the **lower whisker** at the "
        "smallest value ≥ Q1 − 1.5·IQR (never above Q1); values beyond the whisker ends are **fliers**. `values[i]` "
        "belongs to group `group[i]` (0 … G-1, every group non-empty). "
        f"Return {want}. (`report` = {report!r}.)",
        setup, sol, tests, "n ≤ 4·10^5 values, G ≤ 1.5·10^5 groups", speed="O(n log n): one sort, then per-group "
                                                                          "work by offsets (no Python loop over groups)")


def _hist_setup(seed: int, n: int, B: int, report: str) -> str:
    return (f"g = np.random.default_rng({seed})\ndata = np.round(g.normal(0.5, 0.3, {n}), 1)\nbins, lo, hi = {B}, 0.0, 1.0\n"
            f"edges = np.unique(np.round(np.r_[0.0, np.sort(g.uniform(0, 1, {B} - 1)), 1.0], {1 if B <= 10 else 4}))\n"
            f"report = {report!r}")


@template(M, "challenge")
def hist_binning(rng):
    s = seeded(rng)
    report = rng.choice(["counts", "density", "edges"])
    n, B = rng.randint(15, 30), rng.choice([5, 10])
    setup = _hist_setup(s, n, B, report)
    uniform = ("e = np.linspace(lo, hi, bins + 1)\nd = data[(data >= lo) & (data <= hi)]\n"
               "idx = np.searchsorted(e, d, side='right') - 1\nidx[idx == bins] = bins - 1     # the last bin includes its right edge\n"
               "counts = np.bincount(idx, minlength=bins)\n")
    sol = {"counts": uniform + "result = counts",
           "density": uniform + "result = counts / (len(d) * np.diff(e))",
           "edges": ("d = data[(data >= edges[0]) & (data <= edges[-1])]\nidx = np.searchsorted(edges, d, side='right') - 1\n"
                     "idx[idx == len(edges) - 1] = len(edges) - 2\nresult = np.bincount(idx, minlength=len(edges) - 1)")}[report]
    tests = [_hist_setup(s + i, nn, bb, report) for i, (nn, bb) in ((1, (1000000, 10)), (2, (500000, 100000)))]
    want = {"counts": "the counts of `ax.hist(data, bins=bins, range=(lo, hi))`: the edges are "
                      "`np.linspace(lo, hi, bins + 1)`",
            "density": "the bar heights of `ax.hist(data, bins=bins, range=(lo, hi), density=True)`: counts divided by "
                       "(number of values inside the range × bin width); the edges are `np.linspace(lo, hi, bins + 1)`",
            "edges": "the counts of `ax.hist(data, bins=edges)` with the given (uneven, increasing) edges"}[report]
    return challenge(
        "Which bin?",
        "A histogram bin `[e_i, e_(i+1))` includes its left edge and excludes its right edge — except the **last** bin, "
        "which includes both; values outside `[e_0, e_last]` are not counted. The data here is rounded to one decimal, "
        "so many values sit exactly on edges, where floating point is merciless: `np.linspace(0, 1, 11)[3]` is "
        "`0.30000000000000004`, so 0.3 belongs to the bin *before* it. Compare with the actual edge values, not with "
        f"`floor((x - lo) / width)`. Return {want}. (`report` = {report!r}.)",
        setup, sol, tests, "len(data) ≤ 10^6, bins ≤ 10^5", speed="O(n log bins)", tol=1e-9)
