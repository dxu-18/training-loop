"""5.05 Areas, error bars, reference lines & spans: fill_between bands, axhline / axvline / axline, hlines / vlines,
axhspan / axvspan, errorbar, stackplot, step / stairs / stem, broken_barh and eventplot."""

from ._common import P, catalog, challenge, curves, lit, seeded, signal, template

M = "5.05"
LINES = "fc.lines(fc.ax())"
POLYS = "[[fc.r(p.vertices, 4) for p in c.get_paths()] for c in fc.ax().collections]"
SEGS = "[fc.r(c.get_segments(), 4) for c in fc.ax().collections if hasattr(c, 'get_segments')]"
SPANS = "fc.patches(fc.ax())"
EB = ("[(fc.r(c.lines[0].get_xydata(), 4), c.lines[0].get_marker(), c.lines[0].get_linestyle(), "
      "[fc.r(b.get_segments(), 4) for b in c.lines[2]], [l.get_markersize() for l in c.lines[1]]) "
      "for c in fc.ax().containers if type(c).__name__ == 'ErrorbarContainer']")


def _poly(c: str) -> str:
    return f"[fc.r(p.vertices, 4) for p in {c}.get_paths()]"


def _wave05(rng):
    n = rng.choice([40, 60, 100])
    return (f"x = np.linspace(0, 10, {n})\ny = np.sin(x)\nerr = 0.2 + 0.1 * np.abs(np.cos(x))",
            f"`x` holds {n} points from 0 to 10, `y = np.sin(x)` and `err` an uncertainty per point", n)


def _runs05(rng, k=(4, 6)):
    """`runs`: k seeds × n epochs of noisy loss curves, `epochs` 1 … n."""
    k_ = rng.randint(*k)
    n_ = rng.randint(20, 40)
    s = seeded(rng)
    code = (f"g = np.random.default_rng({s})\nepochs = np.arange(1, {n_ + 1})\n"
            f"runs = np.round(1.5 * np.exp(-epochs / 8) + 0.2 + 0.08 * g.normal(size=({k_}, {n_})), 4)")
    return code, f"`runs` holds {k_} training runs (one per row, different seeds) of {n_} epochs, `epochs` = 1 … {n_}", k_, n_


# ================================================================= lite

@template(M, "lite")
def reference_lines(rng):
    code, desc, n = _wave05(rng)
    v = rng.choice([2, 3, 6, 8])
    sl = rng.choice([0.1, -0.2, 0.05])
    a, b = rng.choice([(0.1, 0.9), (0.25, 0.75), (0, 0.5)])
    items = [
        ("axhline", "Plot `y` against `x` and add a thin black horizontal line at y = 0 across the whole Axes "
                    "(`ax.axhline(0, color='k', linewidth=0.8)`).",
         "fig, ax = plt.subplots()\nax.plot(x, y)\nax.axhline(0, color='k', linewidth=0.8)",
         {"post": "fc.lines(fc.ax(), 'color', 'lw')"}),
        ("axvline", f"Plot `y` and mark x = π with a red dashed vertical line labelled 'π' (`ax.axvline`); add a legend.",
         "fig, ax = plt.subplots()\nax.plot(x, y)\nax.axvline(np.pi, color='r', linestyle='--', label='π')\nax.legend()",
         {"post": "fc.lines(fc.ax(), 'color', 'ls', 'label'), fc.legend()", "require": ["axvline", "pi"]}),
        ("axline", f"Plot `y` and draw an **infinite** straight line through (0, 0) with slope {sl} "
                   f"(`ax.axline((0, 0), slope={sl})`), dotted in purple.",
         f"fig, ax = plt.subplots()\nax.plot(x, y)\nax.axline((0, 0), slope={sl}, color='purple', linestyle=':')",
         {"post": "[(fc.r(l.get_xy1(), 4), l.get_slope(), fc.hexcolor(l.get_color()), l.get_linestyle()) for l in fc.ax().get_lines()[1:]]",
          "require": ["axline", "slope"]}),
        ("hlines", "Plot `y` and draw short horizontal segments at y = 0.5 and y = -0.5 from x = 0 to x = 2 in 'C2' "
                   "(`ax.hlines([0.5, -0.5], xmin=0, xmax=2, colors='C2')`).",
         "fig, ax = plt.subplots()\nax.plot(x, y)\nax.hlines([0.5, -0.5], xmin=0, xmax=2, colors='C2')",
         {"post": f"{SEGS}, [fc.hexcolor(c.get_color()) for c in fc.ax().collections]", "require": ["hlines", "xmin", "xmax"]}),
        ("vlines", f"Plot `y` and draw vertical segments at x = {v} and x = {v + 1} from y = -1 to y = 1 in 'C3' "
                   f"(`ax.vlines`).", f"fig, ax = plt.subplots()\nax.plot(x, y)\nax.vlines([{v}, {v + 1}], ymin=-1, ymax=1, colors='C3')",
         {"post": f"{SEGS}, [fc.hexcolor(c.get_color()) for c in fc.ax().collections]", "require": ["vlines", "ymin", "ymax"]}),
        ("axhline", f"`axhline`'s `xmin` / `xmax` are **fractions of the Axes width** (0 … 1), not data values. Plot `y` "
                    f"and draw a horizontal line at y = 0.5 covering only the part from {a} to {b} of the width.",
         f"fig, ax = plt.subplots()\nax.plot(x, y)\nax.axhline(0.5, xmin={a}, xmax={b})", {"post": LINES}),
        ("axline", "Plot `y` and draw an infinite line through the two points (0, -1) and (10, 1) "
                   "(`ax.axline(xy1, xy2)`).", "fig, ax = plt.subplots()\nax.plot(x, y)\nax.axline((0, -1), (10, 1))",
         {"post": "[(fc.r(l.get_xy1(), 4), fc.r(l.get_xy2(), 4)) for l in fc.ax().get_lines()[1:]]"}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "lite")
def spans_and_fills(rng):
    code, desc, n = _wave05(rng)
    lo, hi = rng.choice([(6, 7), (2, 3), (4.5, 5.5)])
    t = rng.choice([0.25, 0.3, 0.5])
    al = rng.choice([0.1, 0.2, 0.3])
    items = [
        ("axhspan", f"Plot `y` and shade the horizontal band -{t} ≤ y ≤ {t} in grey with `alpha={al}` (`ax.axhspan`).",
         f"fig, ax = plt.subplots()\nax.plot(x, y)\nax.axhspan(-{t}, {t}, color='gray', alpha={al})",
         {"post": f"{SPANS}, [fc.hexcolor(p.get_facecolor()) for p in fc.ax().patches]"}),
        ("axvspan", f"Plot `y` and shade the vertical strip {lo} ≤ x ≤ {hi} in orange with `alpha=0.15` (`ax.axvspan`).",
         f"fig, ax = plt.subplots()\nax.plot(x, y)\nax.axvspan({lo}, {hi}, color='orange', alpha=0.15)",
         {"post": f"{SPANS}, [fc.alpha(p.get_facecolor()) for p in fc.ax().patches]"}),
        ("fill_between", f"Plot `y` and draw its uncertainty band `y - err` … `y + err` with `ax.fill_between(..., "
                         f"alpha=0.25)`, labelled '± err'; legend.",
         "fig, ax = plt.subplots()\nax.plot(x, y)\nax.fill_between(x, y - err, y + err, alpha=0.25, label='± err')\nax.legend()",
         {"post": f"{POLYS}, fc.legend()", "require": ["fill_between", "alpha"]}),
        ("where", "Fill only the **positive** humps of `y` down to 0 in green: `ax.fill_between(x, 0, y, where=y > 0, "
                  "color='green', alpha=0.2, interpolate=True)` (`interpolate` closes each hump exactly where y "
                  "crosses 0).", "fig, ax = plt.subplots()\nax.fill_between(x, 0, y, where=y > 0, color='green', alpha=0.2, interpolate=True)",
         {"post": POLYS, "require": ["where", "interpolate"]}),
        ("fill_betweenx", "Fill **horizontally** between x = 9 and x = 10 for y from -1 to 1 "
                          "(`ax.fill_betweenx(np.linspace(-1, 1, 10), 9, 10, alpha=0.1)`).",
         "fig, ax = plt.subplots()\nax.fill_betweenx(np.linspace(-1, 1, 10), 9, 10, alpha=0.1)", {"post": POLYS}),
        ("broken_barh", "Draw two Gantt-style segments on one row: from x = 1 lasting 2 and from x = 4 lasting 1, "
                        "spanning y = -1.2 … -1.0 (`ax.broken_barh([(1, 2), (4, 1)], (-1.2, 0.2), "
                        "facecolors='tab:blue')`).",
         "fig, ax = plt.subplots()\nax.broken_barh([(1, 2), (4, 1)], (-1.2, 0.2), facecolors='tab:blue')",
         {"post": f"{POLYS}, fc.hexcolor(fc.ax().collections[0].get_facecolor())", "require": ["broken_barh", "facecolors"]}),
        ("eventplot", "Mark three events at x = 0.5, 2.5 and 7.1 as short ticks centred on y = -1.4 with length 0.2 "
                      "(`ax.eventplot(..., lineoffsets=-1.4, linelengths=0.2)`).",
         "fig, ax = plt.subplots()\nax.eventplot([0.5, 2.5, 7.1], lineoffsets=-1.4, linelengths=0.2)",
         {"post": "[(fc.r(c.get_positions(), 4), c.get_lineoffset(), c.get_linelength()) for c in fc.ax().collections]",
          "require": ["eventplot", "lineoffsets", "linelengths"]}),
        ("stackplot", "Stack `np.abs(y)` and `np.abs(np.cos(x))` as areas (`ax.stackplot`) labelled '|sin|' and '|cos|', "
                      "`alpha=0.8`, with a legend in the upper left.",
         "fig, ax = plt.subplots()\nax.stackplot(x, np.abs(y), np.abs(np.cos(x)), labels=['|sin|', '|cos|'], alpha=0.8)\n"
         "ax.legend(loc='upper left')", {"post": f"{POLYS}, fc.legend()", "require": ["stackplot", "labels"]}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "lite")
def errors_and_steps(rng):
    code, desc, n = _wave05(rng)
    k = rng.choice([5, 10])
    cs = rng.choice([3, 4, 5])
    where = rng.choice(["pre", "mid", "post"])
    items = [
        ("errorbar", f"Plot every {k}-th point of `y` (`x[::{k}]`, `y[::{k}]`) as dots with error bars `err[::{k}]` "
                     f"(`fmt='o'`) and caps of size {cs}.",
         f"fig, ax = plt.subplots()\nax.errorbar(x[::{k}], y[::{k}], yerr=err[::{k}], fmt='o', capsize={cs})",
         {"post": EB, "require": ["errorbar", "yerr", "capsize"]}),
        ("ecolor", f"Error bars on every {k}-th point (`fmt='o'`), with grey error lines (`ecolor='gray'`) of width 1 "
                   f"(`elinewidth=1`).",
         f"fig, ax = plt.subplots()\nax.errorbar(x[::{k}], y[::{k}], yerr=err[::{k}], fmt='o', ecolor='gray', elinewidth=1)",
         {"post": "[(fc.hexcolor(c.get_color()), fc.r(c.get_linewidth(), 3)) for c in fc.ax().collections]",
          "require": ["ecolor", "elinewidth"]}),
        ("yerr", f"**Asymmetric** errors on every {k}-th point: half of `err` below and the full `err` above "
                 f"(`yerr=[err[::{k}] * 0.5, err[::{k}]]`), squares joined by a line (`fmt='s-'`).",
         f"fig, ax = plt.subplots()\nax.errorbar(x[::{k}], y[::{k}], yerr=[err[::{k}] * 0.5, err[::{k}]], fmt='s-')",
         {"post": EB}),
        ("xerr", f"Horizontal error bars of ±0.2 on every {k}-th point (`xerr=0.2`, `fmt='o'`).",
         f"fig, ax = plt.subplots()\nax.errorbar(x[::{k}], y[::{k}], xerr=0.2, fmt='o')", {"post": EB}),
        ("step", f"Draw every {k}-th point of `y` as a **step** line with the steps centred on the points "
                 f"(`ax.step(..., where='{where}')`).", f"fig, ax = plt.subplots()\nax.step(x[::{k}], y[::{k}], where='{where}')",
         {"post": "fc.lines(fc.ax(), 'drawstyle')", "require": ["step", "where"]}),
        ("stairs", "Draw the histogram-like steps of the values `[3, 1, 4, 1]` between the edges `[0, 1, 2, 3, 4]`, "
                   "filled and semi-transparent (`ax.stairs(values, edges=..., fill=True, alpha=0.3)`).",
         "fig, ax = plt.subplots()\nax.stairs([3, 1, 4, 1], edges=[0, 1, 2, 3, 4], fill=True, alpha=0.3)",
         {"post": "[(fc.r(p.get_data().values, 4), fc.r(p.get_data().edges, 4), p.get_fill(), p.get_alpha()) for p in fc.ax().patches]",
          "require": ["stairs", "edges", "fill"]}),
        ("stem", f"Draw every {k}-th point of `y` as a **stem** plot (a marker on a vertical line from the baseline).",
         f"fig, ax = plt.subplots()\nax.stem(x[::{k}], y[::{k}])",
         {"post": "[(fc.r(c.markerline.get_xydata(), 4), fc.r(c.stemlines.get_segments(), 4)) for c in fc.ax().containers]"}),
    ]
    return catalog(rng, items, code, f"{desc}.")


# ================================================================= easy

@template(M, "easy")
def area_readback(rng):
    code, desc, n = _wave05(rng)
    lo, hi = rng.choice([(1, 2), (3, 4.5), (6, 7)])
    t = rng.choice([0.3, 0.5, 0.7])
    items = [
        ("fill_between", f"`fill_between` with `where=` draws one polygon per stretch where the condition holds. Fill "
                         f"between `y` and 0 where `y > {t}` and return how many polygons it made "
                         f"(`len(coll.get_paths())`).",
         f"fig, ax = plt.subplots()\ncoll = ax.fill_between(x, 0, y, where=y > {t})\nresult = len(coll.get_paths())",
         {"post": f"_, {POLYS}"}),
        ("axvspan", f"`axvspan` returns a Rectangle. Shade x = {lo} … {hi} and return `[r.get_x(), r.get_width()]`.",
         f"fig, ax = plt.subplots()\nax.plot(x, y)\nr = ax.axvspan({lo}, {hi})\nresult = [r.get_x(), r.get_width()]", {"tol": 1e-9}),
        ("axhline", f"Draw `line = ax.axhline({t})` on an Axes with `y` plotted and return the line's x and y data: x runs "
                    f"over axes fractions [0, 1], y is the data value twice.",
         f"fig, ax = plt.subplots()\nax.plot(x, y)\nline = ax.axhline({t})\nresult = [list(line.get_xdata()), list(line.get_ydata())]"),
        ("errorbar", "`errorbar` returns a container `(data_line, caplines, barlinecols)`. Draw `ax.errorbar(x[::10], "
                     "y[::10], yerr=err[::10], capsize=3)` and return `[len(caplines), len(barlinecols)]` (one cap above "
                     "and one below the points; one collection of bars).",
         "fig, ax = plt.subplots()\ndata_line, caplines, barlinecols = ax.errorbar(x[::10], y[::10], yerr=err[::10], capsize=3)\n"
         "result = [len(caplines), len(barlinecols)]"),
        ("stem", "`stem` returns `(markerline, stemlines, baseline)`. Draw `ax.stem(x[::10], y[::10])` and return the "
                 "number of stem segments and the baseline's y value.",
         "fig, ax = plt.subplots()\nmarkerline, stemlines, baseline = ax.stem(x[::10], y[::10])\n"
         "result = [len(stemlines.get_segments()), baseline.get_ydata()[0]]"),
        ("hlines", "Draw `ax.hlines(y[::10], xmin=0, xmax=x[::10])` (a 'lollipop' bar per point) and return the length "
                   "of the longest segment.",
         "fig, ax = plt.subplots()\nlc = ax.hlines(y[::10], xmin=0, xmax=x[::10])\nresult = max(abs(s[1, 0] - s[0, 0]) for s in lc.get_segments())",
         {"tol": 1e-9}),
        ("stackplot", "`stackplot` returns one area per series. Stack `np.abs(y)`, `err` and `np.abs(np.cos(x))` and "
                      "return how many areas it made and the height of the top boundary at the last x "
                      "(the sum of the three series there).",
         "fig, ax = plt.subplots()\nareas = ax.stackplot(x, np.abs(y), err, np.abs(np.cos(x)))\n"
         "result = [len(areas), np.abs(y)[-1] + err[-1] + np.abs(np.cos(x))[-1]]", {"post": f"_, {POLYS}", "tol": 1e-9}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "easy")
def uncertainty_bands(rng):
    code, desc, k, n = _runs05(rng)
    col = rng.choice(["C0", "C2", "tab:purple"])
    q = rng.choice([(10, 90), (25, 75), (5, 95)])
    items = [
        ("fill_between", f"Drill: plot the mean over seeds (`runs.mean(axis=0)`) against `epochs` in {col!r} and the band "
                         f"mean ± std (`runs.std(axis=0)`) in the same color with `alpha=0.25`.",
         f"m, s = runs.mean(axis=0), runs.std(axis=0)\nfig, ax = plt.subplots()\nax.plot(epochs, m, color={col!r})\n"
         f"ax.fill_between(epochs, m - s, m + s, color={col!r}, alpha=0.25)",
         {"post": f"fc.lines(fc.ax(), 'color'), {POLYS}, [fc.hexcolor(c.get_facecolor()) for c in fc.ax().collections]",
          "require": ["fill_between", "std"], "tol": 1e-6}),
        ("fill_between", "Show the full spread instead: the band from the minimum to the maximum over seeds, with the "
                         "median as the line.",
         "fig, ax = plt.subplots()\nax.plot(epochs, np.median(runs, axis=0))\nax.fill_between(epochs, runs.min(axis=0), runs.max(axis=0), alpha=0.3)",
         {"post": f"{LINES}, {POLYS}", "require": ["fill_between", "median"]}),
        ("fill_between", f"A percentile band: the {q[0]}th to {q[1]}th percentile over seeds (`np.percentile(runs, "
                         f"[{q[0]}, {q[1]}], axis=0)`) with `alpha=0.3`, and the mean line on top.",
         f"lo, hi = np.percentile(runs, [{q[0]}, {q[1]}], axis=0)\nfig, ax = plt.subplots()\nax.fill_between(epochs, lo, hi, alpha=0.3)\n"
         f"ax.plot(epochs, runs.mean(axis=0))", {"post": f"{LINES}, {POLYS}", "require": ["percentile"], "tol": 1e-6}),
        ("plot", "Plot every run as a thin, faint line (`linewidth=0.5`, `alpha=0.4`, all 'gray') and the mean as a thick "
                 "black line (`linewidth=2`).",
         "fig, ax = plt.subplots()\nfor r in runs:\n    ax.plot(epochs, r, color='gray', linewidth=0.5, alpha=0.4)\n"
         "ax.plot(epochs, runs.mean(axis=0), color='k', linewidth=2)", {"post": "fc.lines(fc.ax(), 'color', 'lw', 'alpha')", "tol": 1e-6}),
        ("errorbar", "The same information as error bars instead of a band: `ax.errorbar(epochs, runs.mean(axis=0), "
                     "yerr=runs.std(axis=0), fmt='-', capsize=2)`.",
         "fig, ax = plt.subplots()\nax.errorbar(epochs, runs.mean(axis=0), yerr=runs.std(axis=0), fmt='-', capsize=2)",
         {"post": EB, "tol": 1e-6}),
        ("fill_between", "A band that ends where the data ends is fine, but a band with **two** layers reads even "
                         "better: ±1 std with `alpha=0.3` and ±2 std with `alpha=0.15`, both in 'C0', mean line on top.",
         "m, s = runs.mean(axis=0), runs.std(axis=0)\nfig, ax = plt.subplots()\nax.fill_between(epochs, m - 2 * s, m + 2 * s, color='C0', alpha=0.15)\n"
         "ax.fill_between(epochs, m - s, m + s, color='C0', alpha=0.3)\nax.plot(epochs, m, color='C0')",
         {"post": f"{POLYS}, [c.get_alpha() for c in fc.ax().collections], {LINES}", "tol": 1e-6}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "easy")
def reference_marks(rng):
    code, desc, n = curves(rng, acc=True)
    w = rng.randint(2, 5)
    t = rng.choice([0.8, 0.85, 0.9])
    items = [
        ("axvline", "Drill: plot `val_loss` against `epochs` and mark the best epoch (lowest `val_loss`) with a dashed "
                    "vertical line labelled `f'best: {e}'` (e the epoch); legend.",
         "fig, ax = plt.subplots()\nax.plot(epochs, val_loss)\ne = epochs[val_loss.argmin()]\nax.axvline(e, linestyle='--', label=f'best: {e}')\n"
         "ax.legend()", {"post": "fc.lines(fc.ax(), 'ls', 'label'), fc.legend()", "require": ["axvline", "argmin"]}),
        ("axvspan", f"Drill: the learning rate warmed up during the first {w} epochs. Plot `train_loss` against `epochs` "
                    f"and shade epochs 1 … {w} with `ax.axvspan(1, {w}, alpha=0.2, color='orange')`, labelled 'warmup'; "
                    f"legend.",
         f"fig, ax = plt.subplots()\nax.plot(epochs, train_loss)\nax.axvspan(1, {w}, alpha=0.2, color='orange', label='warmup')\nax.legend()",
         {"post": f"{SPANS}, fc.legend()"}),
        ("axhline", f"Plot `val_acc` against `epochs` with a dotted red target line at {t} (`ax.axhline`) labelled "
                    f"'target', and return the first epoch whose accuracy reaches it (or 0 if none does).",
         f"fig, ax = plt.subplots()\nax.plot(epochs, val_acc)\nax.axhline({t}, color='red', linestyle=':', label='target')\n"
         f"hit = np.flatnonzero(val_acc >= {t})\nresult = int(epochs[hit[0]]) if len(hit) else 0",
         {"post": "_, fc.lines(fc.ax(), 'color', 'ls', 'label')"}),
        ("axhspan", "Shade the 'good enough' region of the loss: plot `val_loss` and shade y from its minimum to "
                    "minimum + 0.05 in green (`ax.axhspan`, `alpha=0.2`).",
         "fig, ax = plt.subplots()\nax.plot(epochs, val_loss)\nm = val_loss.min()\nax.axhspan(m, m + 0.05, color='green', alpha=0.2)",
         {"post": f"{SPANS}, [fc.hexcolor(p.get_facecolor()) for p in fc.ax().patches]"}),
        ("vlines", "Mark every epoch where `val_loss` hit a **new** minimum with a short grey vertical line from 0 up to "
                   "the curve (`ax.vlines(e, 0, v, colors='gray')`, one call with arrays).",
         "fig, ax = plt.subplots()\nax.plot(epochs, val_loss)\nnew = val_loss < np.minimum.accumulate(np.r_[np.inf, val_loss[:-1]])\n"
         "ax.vlines(epochs[new], 0, val_loss[new], colors='gray')", {"post": SEGS, "require": ["vlines", "colors"]}),
        ("axline", "Extrapolate the trend of the last 5 epochs of `train_loss`: fit a straight line (`np.polyfit(epochs[-5:], "
                   "train_loss[-5:], 1)`) and draw it with `ax.axline((epochs[-1], fitted value there), slope=...)` "
                   "dashed over the plotted curve.",
         "fig, ax = plt.subplots()\nax.plot(epochs, train_loss)\nslope, icpt = np.polyfit(epochs[-5:], train_loss[-5:], 1)\n"
         "ax.axline((epochs[-1], slope * epochs[-1] + icpt), slope=slope, linestyle='--')",
         {"post": "[(fc.r(l.get_xy1(), 5), round(l.get_slope(), 6)) for l in fc.ax().get_lines()[1:]]", "require": ["axline", "polyfit"]}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "easy")
def step_variants(rng):
    s = seeded(rng)
    n = rng.randint(6, 10)
    setup = (f"g = np.random.default_rng({s})\nt = np.arange({n})\nv = np.round(g.uniform(0, 5, {n}), 2)\n"
             f"edges = np.r_[0, np.cumsum(np.round(g.uniform(0.5, 2, {n}), 2))]")
    where = rng.choice(["pre", "post", "mid"])
    items = [
        ("step", f"`ax.step(t, v, where='{where}')` is `ax.plot` with `drawstyle='steps-{where}'`. Draw it **both** ways, "
                 f"the second shifted up by 6 (`v + 6`), and return both lines' drawstyles.",
         f"fig, ax = plt.subplots()\na, = ax.step(t, v, where='{where}')\nb, = ax.plot(t, v + 6, drawstyle='steps-{where}')\n"
         f"result = [a.get_drawstyle(), b.get_drawstyle()]", {"post": f"_, {LINES}", "require": ["step", "drawstyle"]}),
        ("stairs", "`stairs` takes the **edges** explicitly (one more than the values), so steps can have different "
                   "widths. Draw `ax.stairs(v, edges)` and return the patch's edges back (`p.get_data().edges`) as a list.",
         "fig, ax = plt.subplots()\np = ax.stairs(v, edges)\nresult = p.get_data().edges.tolist()", {"post": "_, fc.ax().patches[0].get_fill()"}),
        ("stairs", "Draw `v` as a filled staircase over `edges` starting from a baseline of 1 instead of 0 "
                   "(`baseline=1`, `fill=True`).",
         "fig, ax = plt.subplots()\nax.stairs(v, edges, baseline=1, fill=True)",
         {"post": "[(fc.r(p.get_data().values, 4), fc.r(p.get_data().baseline, 4), p.get_fill()) for p in fc.ax().patches]"}),
        ("stem", "A horizontal stem plot (stems growing to the right from x = 0): `ax.stem(t, v, orientation='horizontal')`. "
                 "Return the markers' x values.", "fig, ax = plt.subplots()\nc = ax.stem(t, v, orientation='horizontal')\n"
                                                  "result = c.markerline.get_xdata().tolist()", {"post": f"_, {SEGS}"}),
        ("stem", "Stems from a baseline at the **mean** of `v` (`bottom=v.mean()`), with red markers (`markerfmt='ro'`) "
                 "and grey stems (`linefmt='gray'`).",
         "fig, ax = plt.subplots()\nax.stem(t, v, bottom=v.mean(), markerfmt='ro', linefmt='gray')",
         {"post": "[(fc.r(c.markerline.get_xydata(), 4), fc.hexcolor(c.markerline.get_color()), fc.r(c.stemlines.get_segments(), 4), "
                  "fc.hexcolor(c.stemlines.get_color())) for c in fc.ax().containers]", "require": ["stem", "bottom", "markerfmt", "linefmt"]}),
        ("stairs", "Plot a histogram you only have as counts: `counts, bins = np.histogram(v, bins=4)` then "
                   "`ax.stairs(counts, bins)`.",
         "fig, ax = plt.subplots()\ncounts, bins = np.histogram(v, bins=4)\nax.stairs(counts, bins)",
         {"post": "[(fc.r(p.get_data().values, 4), fc.r(p.get_data().edges, 4)) for p in fc.ax().patches]", "require": ["histogram", "stairs"]}),
    ]
    return catalog(rng, items, setup, f"`t` = 0 … {n - 1}, `v` holds {n} values and `edges` {n + 1} increasing edges.")


@template(M, "easy")
def pyplot_to_explicit_areas(rng):
    code, desc, n = _wave05(rng)
    items = [
        ("fill_between", "Rewrite with the explicit API (`fig, ax = plt.subplots()` and `ax.` methods):\n```python\n"
                         "plt.plot(x, y)\nplt.fill_between(x, y - err, y + err, alpha=0.3)\nplt.axhline(0, color='k')\n```",
         "fig, ax = plt.subplots()\nax.plot(x, y)\nax.fill_between(x, y - err, y + err, alpha=0.3)\nax.axhline(0, color='k')",
         {"post": f"fc.lines(fc.ax(), 'color'), {POLYS}"}),
        ("plt.errorbar", "Rewrite with **pyplot** only:\n```python\nfig, ax = plt.subplots()\n"
                         "ax.errorbar(x[::10], y[::10], yerr=err[::10], fmt='o', capsize=3)\nax.axvspan(2, 4, alpha=0.1)\n```",
         "plt.errorbar(x[::10], y[::10], yerr=err[::10], fmt='o', capsize=3)\nplt.axvspan(2, 4, alpha=0.1)",
         {"post": f"{EB}, {SPANS}", "require": ["errorbar", "axvspan"]}),
        ("stackplot", "Rewrite with the explicit API:\n```python\nplt.stackplot(x, np.abs(y), err, labels=['|y|', 'err'])\n"
                      "plt.legend(loc='upper left')\nplt.title('stacked')\n```",
         "fig, ax = plt.subplots()\nax.stackplot(x, np.abs(y), err, labels=['|y|', 'err'])\nax.legend(loc='upper left')\nax.set_title('stacked')",
         {"post": f"{POLYS}, fc.legend(), fc.labels()['title']", "require": ["stackplot", "set_title"]}),
        ("plt.hlines", "Rewrite with **pyplot** only:\n```python\nfig, ax = plt.subplots()\nax.plot(x, y)\n"
                       "ax.hlines([0.5, -0.5], 0, 10, linestyles='dashed')\nax.axline((0, 0), slope=0.1)\n```",
         "plt.plot(x, y)\nplt.hlines([0.5, -0.5], 0, 10, linestyles='dashed')\nplt.axline((0, 0), slope=0.1)",
         {"post": f"{SEGS}, [(fc.r(l.get_xy1(), 4), l.get_slope()) for l in fc.ax().get_lines()[1:]]", "require": ["hlines", "axline"]}),
        ("step", "Rewrite with the explicit API:\n```python\nplt.step(x[::10], y[::10], where='mid', label='step')\n"
                 "plt.stem(x[::10], y[::10])\nplt.legend()\n```",
         "fig, ax = plt.subplots()\nax.step(x[::10], y[::10], where='mid', label='step')\nax.stem(x[::10], y[::10])\nax.legend()",
         {"post": "fc.lines(fc.ax(), 'drawstyle', 'label'), fc.legend()", "require": ["step", "stem"]}),
    ]
    return catalog(rng, items, code, f"{desc}.")


# ================================================================= medium

@template(M, "medium")
def annotated_training(rng):
    code, desc, n = curves(rng, acc=True)
    w = rng.randint(2, 5)
    pat = rng.randint(3, 6)
    drops = sorted(rng.sample(range(3, n - 1), 2))
    items = [
        ("axvspan", f"A training plot with context: `train_loss` and `val_loss` (labels 'train', 'val'), the warmup "
                    f"epochs 1 … {w} shaded orange (`alpha=0.15`, label 'warmup'), the best epoch (lowest `val_loss`) "
                    f"marked with a red dashed vertical line (label 'best'), and a legend.",
         f"fig, ax = plt.subplots()\nax.plot(epochs, train_loss, label='train')\nax.plot(epochs, val_loss, label='val')\n"
         f"ax.axvspan(1, {w}, color='orange', alpha=0.15, label='warmup')\n"
         f"ax.axvline(epochs[val_loss.argmin()], color='red', linestyle='--', label='best')\nax.legend()",
         {"post": f"fc.lines(fc.ax(), 'color', 'ls', 'label'), {SPANS}, fc.legend()", "require": ["axvspan", "axvline"]}),
        ("axvspan", f"Early stopping with patience {pat}: shade the epochs **after** the best one up to best + {pat} "
                    f"(where training would have kept waiting) in grey (`alpha=0.2`), clipped to the last epoch, and "
                    f"draw `val_loss`. Return the epoch at which training stops (best + {pat}, at most the last epoch).",
         f"fig, ax = plt.subplots()\nax.plot(epochs, val_loss)\nbest = epochs[val_loss.argmin()]\nstop = min(best + {pat}, epochs[-1])\n"
         f"ax.axvspan(best, stop, color='gray', alpha=0.2)\nresult = int(stop)", {"post": f"_, {SPANS}"}),
        ("vlines", f"The learning rate was divided by 10 at epochs {drops[0]} and {drops[1]}. Plot `train_loss` and mark "
                   f"both drops with dotted vertical lines over the full height of the Axes (`ax.axvline` in a loop, "
                   f"`linestyle=':'`, `color='k'`).",
         f"fig, ax = plt.subplots()\nax.plot(epochs, train_loss)\nfor e in {drops}:\n    ax.axvline(e, linestyle=':', color='k')",
         {"post": "fc.lines(fc.ax(), 'ls', 'color')", "require": ["axvline"]}),
        ("fill_between", "Show the generalization gap: plot both losses and fill **between** them in red "
                         "(`alpha=0.2`) only where `val_loss > train_loss + 0.15` (`where=`, `interpolate=True`), "
                         "labelled 'overfitting'; legend.",
         "fig, ax = plt.subplots()\nax.plot(epochs, train_loss)\nax.plot(epochs, val_loss)\n"
         "ax.fill_between(epochs, train_loss, val_loss, where=val_loss > train_loss + 0.15, interpolate=True, color='red', alpha=0.2, label='overfitting')\n"
         "ax.legend()", {"post": f"{POLYS}, fc.legend()", "require": ["fill_between", "where", "interpolate"]}),
        ("axhline", "Accuracy with reference levels: plot `val_acc`, a dashed grey line at chance level 0.5 labelled "
                    "'chance', a dotted green line at the best accuracy reached labelled `f'best {a:.3f}'`, y limits "
                    "0 … 1, legend in the lower right.",
         "fig, ax = plt.subplots()\nax.plot(epochs, val_acc)\nax.axhline(0.5, color='gray', linestyle='--', label='chance')\n"
         "a = val_acc.max()\nax.axhline(a, color='green', linestyle=':', label=f'best {a:.3f}')\nax.set_ylim(0, 1)\nax.legend(loc='lower right')",
         {"post": "fc.lines(fc.ax(), 'color', 'ls', 'label'), fc.legend(), fc.limits()['ylim']"}),
        ("axhspan", "Highlight the epochs that were within 0.02 of the best `val_loss`: plot it, shade the horizontal "
                    "band best … best + 0.02 (`axhspan`, `alpha=0.2`), and draw a dot ('o') on every epoch inside the "
                    "band. Return those epochs as a list.",
         "fig, ax = plt.subplots()\nax.plot(epochs, val_loss)\nb = val_loss.min()\nax.axhspan(b, b + 0.02, alpha=0.2)\n"
         "m = val_loss <= b + 0.02\nax.plot(epochs[m], val_loss[m], 'o')\nresult = epochs[m].tolist()",
         {"post": f"_, {SPANS}, fc.lines(fc.ax(), 'marker')"}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "medium")
def seed_bands(rng):
    code, desc, k, n = _runs05(rng)
    s = seeded(rng)
    extra = f"runs_b = np.round(runs * 0.9 + 0.03 * np.random.default_rng({s}).normal(size=runs.shape), 4)"
    items = [
        ("fill_between", "Compare two configurations, A (`runs`) and B (`runs_b`): for each, the mean line and a "
                         "mean ± std band in the same color (A 'C0', B 'C1', band `alpha=0.2`); lines labelled 'A' and "
                         "'B'; legend.",
         "fig, ax = plt.subplots()\nfor r, col, lab in ((runs, 'C0', 'A'), (runs_b, 'C1', 'B')):\n"
         "    m, s = r.mean(axis=0), r.std(axis=0)\n    ax.plot(epochs, m, color=col, label=lab)\n"
         "    ax.fill_between(epochs, m - s, m + s, color=col, alpha=0.2)\nax.legend()",
         {"post": f"fc.lines(fc.ax(), 'color', 'label'), {POLYS}, fc.legend()", "tol": 1e-6}),
        ("fill_between", "Standard error instead of standard deviation: band = mean ± std / √k (k runs), on a **log** "
                         "y-axis.",
         "m = runs.mean(axis=0)\nse = runs.std(axis=0) / np.sqrt(len(runs))\nfig, ax = plt.subplots()\nax.plot(epochs, m)\n"
         "ax.fill_between(epochs, m - se, m + se, alpha=0.3)\nax.set_yscale('log')",
         {"post": f"{LINES}, {POLYS}, fc.scales()", "require": ["fill_between", "sqrt", "set_yscale"], "tol": 1e-6}),
        ("fill_between", "Where is B **reliably** better? Plot both means and fill between them in green only where "
                         "B's mean + std is below A's mean − std (`where=`), `alpha=0.3`.",
         "ma, sa = runs.mean(axis=0), runs.std(axis=0)\nmb, sb = runs_b.mean(axis=0), runs_b.std(axis=0)\nfig, ax = plt.subplots()\n"
         "ax.plot(epochs, ma)\nax.plot(epochs, mb)\nax.fill_between(epochs, mb, ma, where=mb + sb < ma - sa, color='green', alpha=0.3)",
         {"post": f"{LINES}, {POLYS}", "tol": 1e-6}),
        ("errorbar", "Final-epoch results as error bars: for A and B, the mean of the last epoch (`r[:, -1]`) at x = 0 and "
                     "1 with ±std error bars (`fmt='o'`, `capsize=5`), x ticks labelled 'A' and 'B', x limits -0.5 … 1.5.",
         "fig, ax = plt.subplots()\nfin = [runs[:, -1], runs_b[:, -1]]\n"
         "ax.errorbar([0, 1], [f.mean() for f in fin], yerr=[f.std() for f in fin], fmt='o', capsize=5)\n"
         "ax.set_xticks([0, 1], ['A', 'B'])\nax.set_xlim(-0.5, 1.5)",
         {"post": f"{EB}, fc.ticks(fc.ax(), 'x'), fc.limits()['xlim']", "tol": 1e-6}),
        ("fill_between", "A band per **quantile pair**: shade 10–90 % (`alpha=0.15`) and 25–75 % (`alpha=0.3`) of A's runs "
                         "in 'C0', then the median as a line; same for B in 'C1'.",
         "fig, ax = plt.subplots()\nfor r, col in ((runs, 'C0'), (runs_b, 'C1')):\n    p10, p25, p50, p75, p90 = np.percentile(r, [10, 25, 50, 75, 90], axis=0)\n"
         "    ax.fill_between(epochs, p10, p90, color=col, alpha=0.15)\n    ax.fill_between(epochs, p25, p75, color=col, alpha=0.3)\n"
         "    ax.plot(epochs, p50, color=col)", {"post": f"{POLYS}, fc.lines(fc.ax(), 'color')", "require": ["percentile"], "tol": 1e-6}),
        ("fill_between", "Smoothed band: apply a 3-epoch moving average (`np.convolve(v, np.ones(3) / 3, mode='valid')`) "
                         "to A's mean and std, and draw the mean ± std band against `epochs[1:-1]` (the centres of the "
                         "windows) with the mean line.",
         "sm = lambda v: np.convolve(v, np.ones(3) / 3, mode='valid')\nm, s = sm(runs.mean(axis=0)), sm(runs.std(axis=0))\n"
         "fig, ax = plt.subplots()\nax.plot(epochs[1:-1], m)\nax.fill_between(epochs[1:-1], m - s, m + s, alpha=0.3)",
         {"post": f"{LINES}, {POLYS}", "require": ["convolve"], "tol": 1e-6}),
    ]
    return catalog(rng, items, f"{code}\n{extra}", f"{desc}; `runs_b` holds the same number of runs of a second "
                                                   f"configuration.")


@template(M, "medium")
def error_bar_charts(rng):
    s = seeded(rng)
    names = rng.choice([["LR", "SVM", "RF", "GBM", "MLP"], ["small", "base", "large", "xl", "xxl"],
                        ["fold 1", "fold 2", "fold 3", "fold 4", "fold 5"]])
    k = rng.randint(3, 5)
    setup = (f"g = np.random.default_rng({s})\nnames = {lit(names[:k])}\nscores = np.round(g.uniform(0.6, 0.9, ({k}, 8)), 3)\n"
             f"mean, std = scores.mean(axis=1), scores.std(axis=1)\nlo, hi = np.percentile(scores, [5, 95], axis=1)")
    items = [
        ("errorbar", "Model comparison: the mean score of each model at x = 0 … k-1 with ±std error bars, dots only "
                     "(`fmt='o'`), caps of size 4, grey error lines (`ecolor='gray'`), names on the x ticks.",
         f"fig, ax = plt.subplots()\nax.errorbar(np.arange({k}), mean, yerr=std, fmt='o', capsize=4, ecolor='gray')\n"
         f"ax.set_xticks(np.arange({k}), names)", {"post": f"{EB}, fc.ticks(fc.ax(), 'x')", "tol": 1e-6}),
        ("yerr", "Asymmetric intervals: error bars from the 5th to the 95th percentile around the mean "
                 "(`yerr=[mean - lo, hi - mean]`), squares (`fmt='s'`), caps of size 3.",
         f"fig, ax = plt.subplots()\nax.errorbar(np.arange({k}), mean, yerr=[mean - lo, hi - mean], fmt='s', capsize=3)",
         {"post": EB, "tol": 1e-6}),
        ("xerr", "A forest plot: horizontal error bars (`xerr=std`) at y = 0 … k-1 with names on the y ticks, dots, and "
                 "a dashed vertical reference line at the overall mean score (`ax.axvline`).",
         f"fig, ax = plt.subplots()\nax.errorbar(mean, np.arange({k}), xerr=std, fmt='o')\nax.set_yticks(np.arange({k}), names)\n"
         f"ax.axvline(scores.mean(), linestyle='--')",
         {"post": f"{EB}, fc.ticks(fc.ax(), 'y'), fc.lines(fc.ax(), 'ls')", "require": ["errorbar", "xerr", "axvline"], "tol": 1e-6}),
        ("errorbar", "All 8 fold scores behind the summary: scatter each model's scores at its x position (small grey "
                     "dots: `ax.plot(np.full(8, i), scores[i], '.', color='0.6')`), then the mean ± std error bar on "
                     "top in black (`fmt='o'`, `color='k'`, `capsize=4`).",
         f"fig, ax = plt.subplots()\nfor i in range({k}):\n    ax.plot(np.full(8, i), scores[i], '.', color='0.6')\n"
         f"ax.errorbar(np.arange({k}), mean, yerr=std, fmt='o', color='k', capsize=4)",
         {"post": f"{EB}, fc.lines(fc.ax(), 'color', 'marker')", "tol": 1e-6}),
        ("elinewidth", "Thick, light error bars behind dark markers: `fmt='D'`, `color='C3'`, `ecolor='C3'` with "
                       "`alpha` 0.4 for the error lines only — draw them in **two** calls: first `ax.errorbar(..., "
                       "fmt='none', ecolor='C3', elinewidth=4, alpha=0.4)`, then the markers with `ax.plot(..., 'D', "
                       "color='C3')`.",
         f"fig, ax = plt.subplots()\nax.errorbar(np.arange({k}), mean, yerr=std, fmt='none', ecolor='C3', elinewidth=4, alpha=0.4)\n"
         f"ax.plot(np.arange({k}), mean, 'D', color='C3')",
         {"post": "[(fc.r(c.get_segments(), 4), fc.r(c.get_linewidth(), 3), c.get_alpha()) for c in fc.ax().collections], "
                  "fc.lines(fc.ax(), 'marker', 'color')", "require": ["errorbar", "elinewidth"], "tol": 1e-6}),
        ("errorevery", "Error bars on a long curve get crowded: plot the cumulative mean score of model 0 over folds "
                       "(`np.cumsum(scores[0]) / np.arange(1, 9)`) with ±std error bars (`std[0]`) drawn only on every "
                       "2nd point (`errorevery=2`), as a line with dots (`fmt='o-'`).",
         "c = np.cumsum(scores[0]) / np.arange(1, 9)\nfig, ax = plt.subplots()\nax.errorbar(np.arange(1, 9), c, yerr=std[0], errorevery=2, fmt='o-')",
         {"post": EB, "require": ["errorbar", "errorevery", "cumsum"], "tol": 1e-6}),
    ]
    return catalog(rng, items, setup, f"`scores[i]` holds 8 cross-validation scores of model `names[i]` ({k} models); "
                                      f"`mean`, `std`, `lo` (5th) and `hi` (95th percentile) summarize them per model.")


@template(M, "medium")
def stacked_areas(rng):
    s = seeded(rng)
    n = rng.randint(15, 30)
    k = rng.randint(3, 4)
    names = rng.choice([["cat", "dog", "bird", "fish"], ["easy", "medium", "hard", "unsolved"], ["en", "de", "fr", "es"]])[:k]
    setup = (f"g = np.random.default_rng({s})\nsteps = np.arange({n})\nnames = {lit(names)}\n"
             f"share = g.random(({k}, {n})) + np.linspace(0, 1, {n}) * np.arange({k})[:, None]\nshare = np.round(share / share.sum(axis=0), 4)")
    items = [
        ("stackplot", "How the data mix changes over training: stack the rows of `share` (fractions summing to 1 at "
                      "each step) over `steps` with `labels=names`, y limits 0 … 1, legend in the upper left.",
         "fig, ax = plt.subplots()\nax.stackplot(steps, share, labels=names)\nax.set_ylim(0, 1)\nax.legend(loc='upper left')",
         {"post": f"{POLYS}, fc.legend(), fc.limits()['ylim']", "require": ["stackplot", "labels"]}),
        ("stackplot", "Same stack in **percent** (`100 * share`) with custom colors `['C0', 'C1', 'C2', 'C3'][:k]` "
                      "(`colors=`) and `alpha=0.8`; y label '% of batch'.",
         f"fig, ax = plt.subplots()\nax.stackplot(steps, 100 * share, colors=['C0', 'C1', 'C2', 'C3'][:{k}], alpha=0.8)\nax.set_ylabel('% of batch')",
         {"post": f"{POLYS}, [fc.hexcolor(c.get_facecolor()) for c in fc.ax().collections], fc.labels()['ylabel']",
          "require": ["stackplot", "colors"], "tol": 1e-6}),
        ("fill_between", "Build the stack **by hand** with `fill_between`: layer i spans from the cumulative sum of the "
                         "rows below it to that plus row i (`bottom = np.vstack([np.zeros(n), np.cumsum(share, axis=0)])`), "
                         "each labelled with its name; legend.",
         f"fig, ax = plt.subplots()\ncum = np.vstack([np.zeros({n}), np.cumsum(share, axis=0)])\nfor i, name in enumerate(names):\n"
         f"    ax.fill_between(steps, cum[i], cum[i + 1], label=name)\nax.legend()",
         {"post": f"{POLYS}, fc.legend()", "require": ["fill_between", "cumsum"], "tol": 1e-6}),
        ("stackplot", "A streamgraph: the same layers centred around zero (`baseline='sym'`). Return the y value of the "
                      "lowest boundary at step 0 (`-share[:, 0].sum() / 2`, i.e. -0.5).",
         "fig, ax = plt.subplots()\nareas = ax.stackplot(steps, share, baseline='sym')\nresult = -share[:, 0].sum() / 2",
         {"post": f"_, {POLYS}", "require": ["stackplot", "baseline"], "tol": 1e-6}),
        ("stackplot", "Stack the **counts** (`share * batch`, with `batch = 256`) and draw the total as a black dashed "
                      "line on top (it's 256 everywhere).",
         "fig, ax = plt.subplots()\nax.stackplot(steps, share * 256)\nax.plot(steps, (share * 256).sum(axis=0), 'k--')",
         {"post": f"{POLYS}, fc.lines(fc.ax(), 'color', 'ls')", "tol": 1e-6}),
    ]
    return catalog(rng, items, setup, f"`share[i, t]` is the fraction of class `names[i]` in the batch at step `steps[t]` "
                                      f"({k} classes, {n} steps; each column sums to 1).")


@template(M, "medium")
def timeline_plots(rng):
    s = seeded(rng)
    G = rng.randint(2, 4)
    setup = (f"g = np.random.default_rng({s})\njobs = [[(float(a), float(d)) for a, d in zip(np.sort(g.choice(40, 4, replace=False)) * 2, "
             f"g.integers(1, 4, 4))] for _ in range({G})]\nsaves = np.round(np.sort(g.uniform(0, 80, 6)), 1)\n"
             f"spikes = [np.round(np.sort(g.uniform(0, 80, g.integers(5, 12))), 1) for _ in range(3)]")
    items = [
        ("broken_barh", f"A Gantt chart of the jobs on {G} GPUs: GPU i's jobs (`jobs[i]`, a list of (start, duration)) "
                        f"go on row i, spanning y = i - 0.4 … i + 0.4 (`ax.broken_barh(jobs[i], (i - 0.4, 0.8))`); y "
                        f"ticks at 0 … {G - 1} labelled 'gpu 0', 'gpu 1', …",
         f"fig, ax = plt.subplots()\nfor i, row in enumerate(jobs):\n    ax.broken_barh(row, (i - 0.4, 0.8))\n"
         f"ax.set_yticks(range({G}), [f'gpu {{i}}' for i in range({G})])",
         {"post": f"{POLYS}, fc.ticks(fc.ax(), 'y')", "require": ["broken_barh", "set_yticks"]}),
        ("broken_barh", "Color each GPU's row differently (`facecolors=f'C{i}'`) and give the bars black edges "
                        "(`edgecolor='k'`). Return the total busy time per GPU (sum of durations).",
         "fig, ax = plt.subplots()\nfor i, row in enumerate(jobs):\n    ax.broken_barh(row, (i - 0.4, 0.8), facecolors=f'C{i}', edgecolor='k')\n"
         "result = [sum(d for _, d in row) for row in jobs]",
         {"post": "_, [(fc.hexcolor(c.get_facecolor()), fc.hexcolor(c.get_edgecolor())) for c in fc.ax().collections]",
          "require": ["broken_barh", "facecolors"]}),
        ("eventplot", "Checkpoint saves happened at the times in `saves`. Draw them as an event plot: one row of ticks "
                      "at y = 0 (`lineoffsets=0`, `linelengths=0.5`) in 'C3' (`colors='C3'`).",
         "fig, ax = plt.subplots()\nax.eventplot(saves, lineoffsets=0, linelengths=0.5, colors='C3')",
         {"post": "[(fc.r(c.get_positions(), 4), c.get_lineoffset(), c.get_linelength(), fc.hexcolor(c.get_color())) for c in fc.ax().collections]",
          "require": ["eventplot", "colors"]}),
        ("eventplot", "A raster of three spike trains (`spikes`, a list of arrays) in one call: rows at y = 0, 1, 2 "
                      "(`lineoffsets=[0, 1, 2]`), colors 'C0', 'C1', 'C2', line length 0.8.",
         "fig, ax = plt.subplots()\nax.eventplot(spikes, lineoffsets=[0, 1, 2], colors=['C0', 'C1', 'C2'], linelengths=0.8)",
         {"post": "[(fc.r(c.get_positions(), 4), c.get_lineoffset(), fc.hexcolor(c.get_color())) for c in fc.ax().collections]"}),
        ("axvspan", "Maintenance windows: the cluster was down from 20 to 25 and from 50 to 52. Draw the Gantt chart "
                    "(rows as above, `(i - 0.4, 0.8)`), then shade both windows red (`axvspan`, `alpha=0.2`) and draw "
                    "the checkpoint saves as dotted vertical lines (`ax.vlines(saves, -0.5, len(jobs) - 0.5, "
                    "linestyles=':')`).",
         "fig, ax = plt.subplots()\nfor i, row in enumerate(jobs):\n    ax.broken_barh(row, (i - 0.4, 0.8))\n"
         "for a, b in ((20, 25), (50, 52)):\n    ax.axvspan(a, b, color='red', alpha=0.2)\nax.vlines(saves, -0.5, len(jobs) - 0.5, linestyles=':')",
         {"post": f"{POLYS}, {SPANS}", "require": ["broken_barh", "axvspan", "vlines"]}),
    ]
    return catalog(rng, items, setup, f"`jobs[i]` lists GPU i's jobs as (start, duration) pairs ({G} GPUs); `saves` "
                                      f"holds 6 checkpoint times; `spikes` 3 arrays of event times.")


@template(M, "medium")
def threshold_shading(rng):
    code, desc, n = signal(rng)
    t = rng.choice([0.3, 0.5, 0.7])
    items = [
        ("fill_between", f"Shade where the signal is above the threshold {t}: plot `y`, a dashed line at {t} "
                         f"(`axhline`), and fill between {t} and `y` where `y > {t}` (interpolated), in red with "
                         f"`alpha=0.3`.",
         f"fig, ax = plt.subplots()\nax.plot(x, y)\nax.axhline({t}, linestyle='--')\n"
         f"ax.fill_between(x, {t}, y, where=y > {t}, interpolate=True, color='red', alpha=0.3)",
         {"post": f"{POLYS}, fc.lines(fc.ax(), 'ls')", "require": ["fill_between", "interpolate", "axhline"]}),
        ("fill_between", "Color the area under the curve by sign: green where `y > 0`, red where `y < 0` (two "
                         "interpolated `fill_between` calls against 0, `alpha=0.4`).",
         "fig, ax = plt.subplots()\nax.plot(x, y, color='k')\nax.fill_between(x, 0, y, where=y > 0, interpolate=True, color='green', alpha=0.4)\n"
         "ax.fill_between(x, 0, y, where=y < 0, interpolate=True, color='red', alpha=0.4)",
         {"post": f"{POLYS}, [fc.hexcolor(c.get_facecolor()) for c in fc.ax().collections]"}),
        ("fill_between", "Compare the signal with a clean reference `ref = np.sin(x)`: plot both and fill between them "
                         "in orange only where they differ by more than 0.1 (`where=np.abs(y - ref) > 0.1`, no "
                         "interpolation). Return how many polygons it made.",
         "ref = np.sin(x)\nfig, ax = plt.subplots()\nax.plot(x, y)\nax.plot(x, ref)\n"
         "coll = ax.fill_between(x, y, ref, where=np.abs(y - ref) > 0.1, color='orange')\nresult = len(coll.get_paths())",
         {"post": f"_, {POLYS}"}),
        ("fill_between", "Mark the stretches where the signal is above 0 with **full-height** shading: "
                         "`ax.fill_between(x, 0, 1, where=y > 0, transform=ax.get_xaxis_transform(), alpha=0.2)` "
                         "(y in axes fractions, x in data). Also plot `y`.",
         "fig, ax = plt.subplots()\nax.plot(x, y)\nax.fill_between(x, 0, 1, where=y > 0, transform=ax.get_xaxis_transform(), alpha=0.2)",
         {"post": f"{POLYS}, {LINES}", "require": ["fill_between", "get_xaxis_transform"]}),
        ("fill_betweenx", "Fill **horizontally** between the curve and the y-axis for the part where x < 5: "
                          "`ax.fill_betweenx(y[x < 5], 0, x[x < 5], alpha=0.3)` after plotting `y` against `x`.",
         "fig, ax = plt.subplots()\nax.plot(x, y)\nax.fill_betweenx(y[x < 5], 0, x[x < 5], alpha=0.3)", {"post": POLYS}),
        ("axhspan", f"Shade the 'noise band' |y| ≤ {t} grey (`axhspan`, `alpha=0.15`), plot `y`, and draw a dot on "
                    f"every point **outside** it; return how many there are.",
         f"fig, ax = plt.subplots()\nax.axhspan(-{t}, {t}, color='gray', alpha=0.15)\nax.plot(x, y)\nm = np.abs(y) > {t}\n"
         f"ax.plot(x[m], y[m], 'o')\nresult = int(m.sum())", {"post": f"_, {SPANS}, fc.lines(fc.ax(), 'marker')"}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "medium")
def step_signals(rng):
    code, desc, n = curves(rng)
    s = seeded(rng)
    every = rng.choice([3, 4, 5])
    items = [
        ("stairs", "Plot the histogram of `val_loss - train_loss` from precomputed counts: `counts, edges = "
                   "np.histogram(gap, bins=8)` then `ax.stairs(counts, edges, fill=True, alpha=0.5)` with a black "
                   "outline drawn by a second `ax.stairs(counts, edges, color='k')`.",
         "gap = val_loss - train_loss\ncounts, edges = np.histogram(gap, bins=8)\nfig, ax = plt.subplots()\n"
         "ax.stairs(counts, edges, fill=True, alpha=0.5)\nax.stairs(counts, edges, color='k')",
         {"post": "[(fc.r(p.get_data().values, 4), fc.r(p.get_data().edges, 4), p.get_fill()) for p in fc.ax().patches]"}),
        ("step", f"The learning rate halves every {every} epochs: compute `lr = 0.1 * 0.5 ** ((epochs - 1) // {every})` "
                 f"and plot it with `ax.step(epochs, lr, where='post')` on a log y-axis.",
         f"lr = 0.1 * 0.5 ** ((epochs - 1) // {every})\nfig, ax = plt.subplots()\nax.step(epochs, lr, where='post')\nax.set_yscale('log')",
         {"post": "fc.lines(fc.ax(), 'drawstyle'), fc.scales()", "require": ["step", "set_yscale"]}),
        ("stem", "Show the epoch-to-epoch **change** of `val_loss` (`np.diff(val_loss)`) as a stem plot against "
                 "`epochs[1:]`, with the baseline at 0.",
         "fig, ax = plt.subplots()\nax.stem(epochs[1:], np.diff(val_loss))",
         {"post": "[(fc.r(c.markerline.get_xydata(), 4), fc.r(c.stemlines.get_segments(), 4)) for c in fc.ax().containers]",
          "require": ["stem", "diff"]}),
        ("stem", "Sparse weights: `w` has mostly zeros. Draw only the non-zero weights as a stem plot at their indices, "
                 "with markers 'D' (`markerfmt='D'`). Return how many stems there are.",
         "nz = np.flatnonzero(w)\nfig, ax = plt.subplots()\nc = ax.stem(nz, w[nz], markerfmt='D')\nresult = len(c.stemlines.get_segments())",
         {"post": "_, [fc.r(c.markerline.get_xydata(), 4) for c in fc.ax().containers]",
          "setup": f"g = np.random.default_rng({s})\nw = np.round(np.where(g.random(40) < 0.2, g.normal(size=40), 0.0), 3)",
          "require": ["stem", "flatnonzero|nonzero"]}),
        ("step", "Best-so-far curve: plot `val_loss` faintly (`alpha=0.4`) and its running minimum "
                 "(`np.minimum.accumulate(val_loss)`) as a step line (`where='post'`) labelled 'best so far'; legend.",
         "fig, ax = plt.subplots()\nax.plot(epochs, val_loss, alpha=0.4)\nax.step(epochs, np.minimum.accumulate(val_loss), where='post', label='best so far')\n"
         "ax.legend()", {"post": "fc.lines(fc.ax(), 'drawstyle', 'alpha', 'label'), fc.legend()", "require": ["step", "accumulate"]}),
    ]
    return catalog(rng, items, code, f"{desc}.")


AXLINES = "[(fc.r(l.get_xy1(), 4), l.get_slope()) for l in fc.ax().get_lines() if hasattr(l, 'get_slope')]"
POINTS = "[(fc.r(l.get_xydata(), 4), l.get_marker()) for l in fc.ax().get_lines() if not hasattr(l, 'get_slope')]"


@template(M, "medium")
def context_regions(rng):
    s = seeded(rng)
    n = rng.choice([42, 56, 70])
    start = rng.choice(["2024-01-01", "2024-03-04", "2025-02-10", "2025-06-02"])
    k = rng.choice([1.2, 1.5, 1.8])
    tol = rng.choice([0.1, 0.15, 0.2])
    setup = (f"g = np.random.default_rng({s})\ndays = np.datetime64('{start}') + np.arange({n})\n"
             f"traffic = np.round(100 + 15 * np.sin(np.arange({n}) / 3) + g.normal(0, 6, {n}), 1)\n"
             f"true = np.round(g.uniform(1, 10, 40), 3)\npred = np.round(true * (1 + g.normal(0, 0.1, 40)), 3)")
    items = [
        ("axvspan", "Shade every weekend: plot `traffic` against `days` and, for each Saturday in `days`, call "
                    "`ax.axvspan(sat, sat + np.timedelta64(2, 'D'), color='gray', alpha=0.15)`. "
                    "(`(days.astype('int64') + 3) % 7` is the weekday, Monday = 0: 1970-01-01 was a Thursday.) Return "
                    "the number of weekends shaded.",
         "fig, ax = plt.subplots()\nax.plot(days, traffic)\nsat = days[(days.astype('int64') + 3) % 7 == 5]\nfor d in sat:\n"
         "    ax.axvspan(d, d + np.timedelta64(2, 'D'), color='gray', alpha=0.15)\nresult = len(sat)",
         {"post": f"_, {SPANS}", "require": ["axvspan", "timedelta64"]}),
        ("axline", f"A parity plot: draw `pred` against `true` as dots (`ax.plot(true, pred, 'o')`), the ideal line y = x "
                   f"(`ax.axline((0, 0), slope=1, color='k', linestyle='--')`), the ±{tol:.0%} band around it with "
                   f"`fill_between` over the sorted targets (`t = np.sort(true)`, from `(1 - {tol}) * t` to "
                   f"`(1 + {tol}) * t`, `alpha=0.2`), and equal scales (`set_aspect('equal')`). Return the fraction of "
                   f"predictions inside the band (`|pred - true| <= {tol} * true`).",
         f"fig, ax = plt.subplots()\nax.plot(true, pred, 'o')\nax.axline((0, 0), slope=1, color='k', linestyle='--')\n"
         f"t = np.sort(true)\nax.fill_between(t, (1 - {tol}) * t, (1 + {tol}) * t, alpha=0.2)\nax.set_aspect('equal')\n"
         f"result = float(np.mean(np.abs(pred - true) <= {tol} * true))",
         {"post": f"_, {POINTS}, {AXLINES}, {POLYS}, fc.ax().get_aspect()", "require": ["axline", "fill_between", "sort"]}),
        ("axhspan", f"A control chart: plot `traffic` against `days`, draw its mean as a solid black line (`axhline`), "
                    f"shade mean ± {k}·std (`axhspan`, `alpha=0.2`), and mark every day outside that band with a red dot "
                    f"(`'ro'`). Return how many days are outside.",
         f"m, sd = traffic.mean(), traffic.std()\nfig, ax = plt.subplots()\nax.plot(days, traffic)\nax.axhline(m, color='k')\n"
         f"ax.axhspan(m - {k} * sd, m + {k} * sd, alpha=0.2)\nout = np.abs(traffic - m) > {k} * sd\n"
         f"ax.plot(days[out], traffic[out], 'ro')\nresult = int(out.sum())",
         {"post": f"_, fc.lines(fc.ax(), 'color', 'marker'), {SPANS}", "require": ["axhline", "axhspan"], "tol": 1e-6}),
        ("axvspan", "Show the train / validation / test split of the series: with `a = int(0.7 * n)` and `b = int(0.85 * "
                    "n)` (n days), shade `days[0]` … `days[a]` in 'C0' labelled 'train', `days[a]` … `days[b]` in 'C1' "
                    "labelled 'val' and `days[b]` … `days[-1]` in 'C2' labelled 'test' (`axvspan`, `alpha=0.15`), plot "
                    "`traffic` on top in black, and add a legend. Return the three part sizes `[a, b - a, n - b]`.",
         "n = len(days)\na, b = int(0.7 * n), int(0.85 * n)\nfig, ax = plt.subplots()\n"
         "for (i, j), col, lab in zip([(0, a), (a, b), (b, n - 1)], ['C0', 'C1', 'C2'], ['train', 'val', 'test']):\n"
         "    ax.axvspan(days[i], days[j], color=col, alpha=0.15, label=lab)\nax.plot(days, traffic, color='k')\nax.legend()\n"
         "result = [a, b - a, n - b]",
         {"post": f"_, {SPANS}, [fc.hexcolor(p.get_facecolor()) for p in fc.ax().patches], fc.legend()"}),
        ("fill_between", "Daily traffic as steps: `ax.step(days, traffic, where='post')`, and the area under it filled "
                         "with the **same** step shape (`ax.fill_between(days, traffic, step='post', alpha=0.3)`; "
                         "without `step=` the fill would cut the corners).",
         "fig, ax = plt.subplots()\nax.step(days, traffic, where='post')\nax.fill_between(days, traffic, step='post', alpha=0.3)",
         {"post": f"fc.lines(fc.ax(), 'drawstyle'), {POLYS}", "require": ["step", "fill_between"]}),
        ("sliding_window_view", "A 7-day range band: with `w = np.lib.stride_tricks.sliding_window_view(traffic, 7)` "
                                "(one row per window), shade between each window's min and max against `days[6:]` (the "
                                "window's last day, `alpha=0.3`) and draw the windows' mean as a line on top.",
         "w = np.lib.stride_tricks.sliding_window_view(traffic, 7)\nfig, ax = plt.subplots()\n"
         "ax.fill_between(days[6:], w.min(axis=1), w.max(axis=1), alpha=0.3)\nax.plot(days[6:], w.mean(axis=1))",
         {"post": f"{POLYS}, {LINES}", "require": ["fill_between"], "tol": 1e-6}),
    ]
    return catalog(rng, items, setup, f"`days` holds {n} consecutive days from {start} (a NumPy `datetime64[D]` array) "
                                      f"and `traffic` a daily metric; `true` and `pred` are 40 regression targets and a "
                                      f"model's predictions.")


# ================================================================= hard

@template(M, "hard")
def area_functions(rng):
    code, desc, k, n = _runs05(rng)
    col = rng.choice(["C0", "C1", "tab:green"])
    items = [
        ("fill_between", f"Write `plot_band(ax, x, runs, color={col!r}, label=None, alpha=0.25)`: draw the mean over rows "
                         f"as a line (with `label`) and mean ± std as a band in the same color; return `(line, band)`.",
         f"def plot_band(ax, x, runs, color={col!r}, label=None, alpha=0.25):\n    m, s = runs.mean(axis=0), runs.std(axis=0)\n"
         f"    line, = ax.plot(x, m, color=color, label=label)\n    band = ax.fill_between(x, m - s, m + s, color=color, alpha=alpha)\n"
         f"    return line, band",
         {"post": "_f, _a = plt.subplots()\n_l, _b = plot_band(_a, epochs, runs, label='A')\n_m, _c = plot_band(_a, epochs, runs * 2, color='k', alpha=0.1)\n"
                  "(fc.lines(_a, 'color', 'label'), [[fc.r(p.vertices, 4) for p in c.get_paths()] for c in _a.collections], "
                  "[(fc.hexcolor(c.get_facecolor()), c.get_alpha()) for c in _a.collections], _l is _a.get_lines()[0], _c is _a.collections[1])",
          "tol": 1e-6}),
        ("axvspan", "Write `shade_regions(ax, x, mask, **kwargs)`: shade every **maximal run** of True in `mask` with one "
                    "`ax.axvspan(x[start], x[end], **kwargs)` (end = the run's last index); return the number of spans.",
         "def shade_regions(ax, x, mask, **kwargs):\n    count, i, n = 0, 0, len(mask)\n    while i < n:\n        if mask[i]:\n"
         "            j = i\n            while j + 1 < n and mask[j + 1]:\n                j += 1\n            ax.axvspan(x[i], x[j], **kwargs)\n"
         "            count += 1\n            i = j + 1\n        else:\n            i += 1\n    return count",
         {"post": "_f, _a = plt.subplots()\n_m = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 1], bool)\n_r = shade_regions(_a, np.arange(10) * 2.0, _m, color='red')\n"
                  "(_r, fc.patches(_a), [fc.hexcolor(p.get_facecolor()) for p in _a.patches], shade_regions(_a, np.arange(3.0), np.zeros(3, bool)))",
          "require": ["axvspan"]}),
        ("axvline", "Write `mark_best(ax, x, y, mode='min', **kwargs)`: find the best point (lowest y for 'min', highest "
                    "for 'max'), draw a vertical line there with `ax.axvline(x_best, **kwargs)` and a dot "
                    "(`ax.plot(x_best, y_best, 'o')`), and return `(x_best, y_best)`.",
         "def mark_best(ax, x, y, mode='min', **kwargs):\n    i = int(np.argmin(y) if mode == 'min' else np.argmax(y))\n"
         "    ax.axvline(x[i], **kwargs)\n    ax.plot(x[i], y[i], 'o')\n    return x[i], y[i]",
         {"post": "_f, _a = plt.subplots()\n_m = runs.mean(axis=0)\n_a.plot(epochs, _m)\n_r = mark_best(_a, epochs, _m, linestyle='--')\n"
                  "_s = mark_best(_a, epochs, -_m, mode='max', color='k')\n(fc.r(_r, 6), fc.r(_s, 6), fc.lines(_a, 'ls', 'color', 'marker'))",
          "tol": 1e-6}),
        ("broken_barh", "Write `gantt(ax, schedule)`: `schedule` maps a lane name to a list of (start, duration) pairs; "
                        "lane i (in the dict's order) is drawn with `ax.broken_barh(pairs, (i - 0.4, 0.8))`, the y ticks "
                        "are the lane names, and the function returns the time the last job ends.",
         "def gantt(ax, schedule):\n    end = 0\n    for i, (name, pairs) in enumerate(schedule.items()):\n"
         "        ax.broken_barh(pairs, (i - 0.4, 0.8))\n        end = max([end] + [s + d for s, d in pairs])\n"
         "    ax.set_yticks(range(len(schedule)), list(schedule))\n    return end",
         {"post": "_f, _a = plt.subplots()\n_e = gantt(_a, {'a': [(0, 2), (5, 1)], 'b': [(1, 7)], 'c': []})\n"
                  "(_e, [[fc.r(p.vertices, 4) for p in c.get_paths()] for c in _a.collections], fc.ticks(_a, 'y'))",
          "require": ["broken_barh", "set_yticks"]}),
        ("errorbar", "Write `summary_errorbars(ax, results, names=None)`: `results` is a list of 1-D arrays (one per "
                     "method); draw the means at x = 0, 1, … with ±std error bars (`fmt='o'`, `capsize=4`), put `names` on "
                     "the x ticks when given, and return the means as a NumPy array.",
         "def summary_errorbars(ax, results, names=None):\n    m = np.array([np.mean(r) for r in results])\n"
         "    s = np.array([np.std(r) for r in results])\n    ax.errorbar(np.arange(len(results)), m, yerr=s, fmt='o', capsize=4)\n"
         "    if names is not None:\n        ax.set_xticks(np.arange(len(results)), names)\n    return m",
         {"post": "_f, _a = plt.subplots()\n_r = summary_errorbars(_a, list(runs[:, -3:]), names=[f'r{i}' for i in range(len(runs))])\n"
                  "(fc.r(_r, 6), [fc.r(c.lines[2][0].get_segments(), 5) for c in _a.containers], fc.ticks(_a, 'x'))", "tol": 1e-6}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "hard")
def debug_areas(rng):
    code, desc, n = _wave05(rng)
    s = seeded(rng)
    items = [
        ("yerr", "This crashes: asymmetric errors must be shaped **(2, n)** (lower row, upper row), not (n, 2).\n"
                 "```python\nfig, ax = plt.subplots()\nbounds = np.column_stack([err[::10] * 0.5, err[::10]])\n"
                 "ax.errorbar(x[::10], y[::10], yerr=bounds, fmt='o')\n```\nFix it.",
         "fig, ax = plt.subplots()\nbounds = np.column_stack([err[::10] * 0.5, err[::10]])\nax.errorbar(x[::10], y[::10], yerr=bounds.T, fmt='o')",
         {"post": EB}),
        ("axhline", "The line was meant to run from x = 2 to x = 4 **in data units**, but `axhline`'s `xmin` / `xmax` are "
                    "axes fractions (0 … 1), so it spans almost the whole plot:\n```python\nfig, ax = plt.subplots()\n"
                    "ax.plot(x, y)\nax.axhline(0.5, xmin=2, xmax=4)\n```\nUse the function that takes data coordinates "
                    "(`ax.hlines`).",
         "fig, ax = plt.subplots()\nax.plot(x, y)\nax.hlines(0.5, xmin=2, xmax=4)", {"post": f"{SEGS}, len(fc.ax().get_lines())", "require": ["hlines"]}),
        ("interpolate", "The green fill leaves **gaps** next to every zero crossing, because the polygons stop at the "
                        "last sample above 0:\n```python\nfig, ax = plt.subplots()\n"
                        "ax.fill_between(x, 0, y, where=y > 0, color='green')\n```\nClose the gaps.",
         "fig, ax = plt.subplots()\nax.fill_between(x, 0, y, where=y > 0, color='green', interpolate=True)", {"post": POLYS}),
        ("hlines", "This crashes with *missing 2 required positional arguments: 'xmin' and 'xmax'*:\n```python\n"
                   "fig, ax = plt.subplots()\nax.plot(x, y)\nax.hlines([0.5, -0.5])\n```\nThe lines should span the "
                   "data's x-range (`x.min()` … `x.max()`).",
         "fig, ax = plt.subplots()\nax.plot(x, y)\nax.hlines([0.5, -0.5], x.min(), x.max())", {"post": SEGS}),
        ("legend", "The stack plot's legend is empty (*No artists with labels found*):\n```python\nfig, ax = plt.subplots()\n"
                   "ax.stackplot(x, np.abs(y), err)\nax.legend()\n```\nName the layers '|y|' and 'err'.",
         "fig, ax = plt.subplots()\nax.stackplot(x, np.abs(y), err, labels=['|y|', 'err'])\nax.legend()",
         {"post": "fc.legend()", "require": ["labels"]}),
        ("stem", "This crashes on Matplotlib ≥ 3.8 (*unexpected keyword argument 'use_line_collection'*: the option was "
                 "removed, stems are always one collection now):\n```python\nfig, ax = plt.subplots()\n"
                 "ax.stem(x[::10], y[::10], use_line_collection=True)\n```\nFix it.",
         "fig, ax = plt.subplots()\nax.stem(x[::10], y[::10])",
         {"post": "[fc.r(c.stemlines.get_segments(), 4) for c in fc.ax().containers]"}),
        ("fill_between", "The band is a zig-zag mess: the points were collected in **random order**, and fill_between "
                         "connects them in the order given.\n```python\nfig, ax = plt.subplots()\n"
                         "ax.fill_between(xs, ys - err, ys + err)\n```\nSort by x first (`np.argsort(xs)`) and reorder "
                         "both arrays.",
         "o = np.argsort(xs)\nfig, ax = plt.subplots()\nax.fill_between(xs[o], ys[o] - err, ys[o] + err)",
         {"post": POLYS, "setup": f"p = np.random.default_rng({s}).permutation({n})\nxs, ys = x[p], y[p]", "require": ["argsort"]}),
        ("axvspan", "`axvspan(xmin, xmax)` takes the two x values; this code passes a start and a **width** (like "
                    "`broken_barh` does) and so shades 6 … 1 — almost nothing visible where intended:\n```python\n"
                    "fig, ax = plt.subplots()\nax.plot(x, y)\nax.axvspan(6, 1)\n```\nShade 6 … 7.",
         "fig, ax = plt.subplots()\nax.plot(x, y)\nax.axvspan(6, 7)", {"post": SPANS}),
        ("errorbar", "The error bars come out as a **line** through the points plus bars, but the author wanted only "
                     "markers with error bars (no connecting line):\n```python\nfig, ax = plt.subplots()\n"
                     "ax.errorbar(x[::10], y[::10], yerr=err[::10])\n```\nFix it with a fmt that has a marker and no "
                     "line style.",
         "fig, ax = plt.subplots()\nax.errorbar(x[::10], y[::10], yerr=err[::10], fmt='o')", {"post": EB, "require": ["fmt"]}),
        ("errorbar", "This crashes with *'yerr' must not contain negative values*: the error was computed as a signed "
                     "difference from a reference curve.\n```python\nfig, ax = plt.subplots()\n"
                     "ax.errorbar(x[::10], y[::10], yerr=y[::10] - np.sin(x[::10] + 0.3), fmt='o')\n```\nError bars are "
                     "sizes: use the absolute difference.",
         "fig, ax = plt.subplots()\nax.errorbar(x[::10], y[::10], yerr=np.abs(y[::10] - np.sin(x[::10] + 0.3)), fmt='o')",
         {"post": EB, "require": ["errorbar", "abs"]}),
        ("stairs", "This crashes with *Size mismatch between \"values\" and \"edges\"*:\n```python\nfig, ax = plt.subplots()\n"
                   "counts, edges = np.histogram(y, bins=8)\nax.stairs(counts, edges[:-1])\n```\n`stairs` needs **one more** "
                   "edge than values (each step has a left and a right edge). Fix it.",
         "fig, ax = plt.subplots()\ncounts, edges = np.histogram(y, bins=8)\nax.stairs(counts, edges)",
         {"post": "[(fc.r(p.get_data().values, 4), fc.r(p.get_data().edges, 4)) for p in fc.ax().patches]"}),
        ("broken_barh", "Two jobs ran from 1 to 3 and from 4 to 7, but the bars come out too long: the pairs were "
                        "written as (start, **end**) while `broken_barh` wants (start, **duration**).\n```python\n"
                        "fig, ax = plt.subplots()\nruns = [(1, 3), (4, 7)]\nax.broken_barh(runs, (0, 1))\n```\n"
                        "Keep `runs` as it is and convert the pairs in the call.",
         "fig, ax = plt.subplots()\nruns = [(1, 3), (4, 7)]\nax.broken_barh([(s, e - s) for s, e in runs], (0, 1))",
         {"post": POLYS, "require": ["broken_barh"]}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "hard")
def area_drills(rng):
    s = seeded(rng)
    k = rng.choice([5, 5, 6])
    w = rng.randint(2, 4)
    setup = (f"g = np.random.default_rng({s})\nepochs = np.arange(1, 31)\n"
             f"runs = np.round(1.5 * np.exp(-epochs / 8) + 0.2 + 0.08 * g.normal(size=({k}, 30)), 4)\n"
             f"lr = np.round(np.minimum(epochs / {w}, 1) * 0.1, 4)\nx = np.linspace(0, 10, 100)\ny = np.sin(x)\n"
             f"err = 0.2 + 0.1 * np.abs(np.cos(x))")
    items = [
        ("fill_between", f"Drill 1: plot the mean of the {k} noisy runs as a line labelled 'mean' with a ±std band "
                         f"(`alpha=0.25`, label '± std'), x label 'epoch', y label 'loss', legend.",
         "m, sd = runs.mean(axis=0), runs.std(axis=0)\nfig, ax = plt.subplots()\nax.plot(epochs, m, label='mean')\n"
         "ax.fill_between(epochs, m - sd, m + sd, alpha=0.25, label='± std')\nax.set(xlabel='epoch', ylabel='loss')\nax.legend()",
         {"post": f"{LINES}, {POLYS}, fc.legend(), fc.labels()", "tol": 1e-6}),
        ("axvspan", f"Drill 2: the learning rate `lr` warms up linearly for {w} epochs. Plot the mean run, shade the "
                    f"warmup epochs (from 1 to the first epoch where `lr` reaches its maximum) with `axvspan` "
                    f"(`alpha=0.2`, label 'warmup'), legend.",
         "fig, ax = plt.subplots()\nax.plot(epochs, runs.mean(axis=0))\nend = epochs[np.argmax(lr)]\nax.axvspan(1, end, alpha=0.2, label='warmup')\nax.legend()",
         {"post": f"{SPANS}, fc.legend()", "tol": 1e-6}),
        ("axvline", "Drill 3: mark the best epoch (lowest mean loss) with a vertical line and a text label: "
                    "`ax.axvline(best, color='r', linestyle='--')` and `ax.text(best, y_top, f'best epoch {best}', "
                    "ha='left')` with `y_top = ax.get_ylim()[1]` read **after** plotting the mean.",
         "m = runs.mean(axis=0)\nfig, ax = plt.subplots()\nax.plot(epochs, m)\nbest = int(epochs[m.argmin()])\n"
         "ax.axvline(best, color='r', linestyle='--')\nax.text(best, ax.get_ylim()[1], f'best epoch {best}', ha='left')",
         {"post": "fc.lines(fc.ax(), 'color', 'ls'), fc.text_items()", "require": ["axvline", "text"], "tol": 1e-6}),
        ("broken_barh", "The guide's first panel in one Axes: `y` with a ±`err` band (label '± err'), green fill of the "
                        "positive part (interpolated), black zero line, red dashed vertical line at π (label 'π'), "
                        "grey band |y| ≤ 0.25, orange strip 6 … 7, a purple dotted `axline` through (0, 0) with slope "
                        "0.1, segments `hlines([0.5, -0.5], 0, 2, colors='C2')` and `vlines([8, 9], -1, 1, colors='C3')`, "
                        "`fill_betweenx(np.linspace(-1, 1, 10), 9, 10, alpha=0.1)`, `broken_barh([(1, 2), (4, 1)], (-1.2, "
                        "0.2))`, `eventplot([0.5, 2.5, 7.1], lineoffsets=-1.4, linelengths=0.2)`, and a legend.",
         "fig, ax = plt.subplots()\nax.plot(x, y)\nax.fill_between(x, y - err, y + err, alpha=0.25, label='± err')\n"
         "ax.fill_between(x, 0, y, where=y > 0, color='green', alpha=0.2, interpolate=True)\nax.axhline(0, color='k', linewidth=0.8)\n"
         "ax.axvline(np.pi, color='r', linestyle='--', label='π')\nax.axhspan(-0.25, 0.25, color='gray', alpha=0.1)\n"
         "ax.axvspan(6, 7, color='orange', alpha=0.15)\nax.axline((0, 0), slope=0.1, color='purple', linestyle=':')\n"
         "ax.hlines([0.5, -0.5], xmin=0, xmax=2, colors='C2')\nax.vlines([8, 9], ymin=-1, ymax=1, colors='C3')\n"
         "ax.fill_betweenx(np.linspace(-1, 1, 10), 9, 10, alpha=0.1)\nax.broken_barh([(1, 2), (4, 1)], (-1.2, 0.2), facecolors='tab:blue')\n"
         "ax.eventplot([0.5, 2.5, 7.1], lineoffsets=-1.4, linelengths=0.2)\nax.legend()",
         {"post": f"len(fc.ax().get_lines()), len(fc.ax().collections), {SPANS}, fc.legend(), {POLYS}",
          "require": ["fill_between", "axline", "hlines", "vlines", "fill_betweenx", "broken_barh", "eventplot"]}),
        ("errorbar", "The guide's error-bar and step panels on `plt.subplots(1, 2, figsize=(10, 3), layout='constrained')`: "
                     "left, `errorbar(x[::10], y[::10], yerr=err[::10], fmt='o', capsize=4, ecolor='gray', elinewidth=1)` "
                     "and the asymmetric `errorbar(x[::10], y[::10] + 1, xerr=0.2, yerr=[err[::10] * 0.5, err[::10]], "
                     "fmt='s-')`; right, `step(x[::10], y[::10], where='mid')`, `stairs([3, 1, 4, 1], edges=[0, 1, 2, 3, 4], "
                     "fill=True, alpha=0.3)` and `stem(x[::10], y[::10])`. Save it as `areas.png` at 30 dpi and close it; "
                     "return the image shape.",
         "fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3), layout='constrained')\n"
         "a.errorbar(x[::10], y[::10], yerr=err[::10], fmt='o', capsize=4, ecolor='gray', elinewidth=1)\n"
         "a.errorbar(x[::10], y[::10] + 1, xerr=0.2, yerr=[err[::10] * 0.5, err[::10]], fmt='s-')\n"
         "b.step(x[::10], y[::10], where='mid')\nb.stairs([3, 1, 4, 1], edges=[0, 1, 2, 3, 4], fill=True, alpha=0.3)\n"
         "b.stem(x[::10], y[::10])\nfig.savefig('areas.png', dpi=30)\nplt.close(fig)\nresult = plt.imread('areas.png').shape",
         {"require": ["errorbar", "step", "stairs", "stem", "savefig", "close"]}),
    ]
    return catalog(rng, items, setup, f"`runs` holds {k} noisy loss curves over `epochs` 1 … 30 and `lr` the learning "
                                      f"rate per epoch; `x`, `y = sin(x)` and `err` are the guide's wave and its error.")


@template(M, "hard")
def area_specs(rng):
    code, desc, n = curves(rng, acc=True)
    s = seeded(rng)
    w = rng.randint(2, 4)
    items = [
        ("subplots", f"A two-panel report on `plt.subplots(2, 1, sharex=True, figsize=(7, 6))`. Top: mean loss of the "
                     f"seeds in `seeds` with a ±std band, warmup epochs 1 … {w} shaded (`axvspan`, `alpha=0.15`), and the "
                     f"best epoch of the mean marked with a dashed `axvline` on **both** panels. Bottom: `val_acc` with "
                     f"a dotted target line at 0.9 (`axhline`) and the region above it shaded green (`axhspan(0.9, 1, "
                     f"alpha=0.1)`), y limits 0 … 1.",
         f"m, sd = seeds.mean(axis=0), seeds.std(axis=0)\nbest = epochs[m.argmin()]\nfig, (a, b) = plt.subplots(2, 1, sharex=True, figsize=(7, 6))\n"
         f"a.plot(epochs, m)\na.fill_between(epochs, m - sd, m + sd, alpha=0.3)\na.axvspan(1, {w}, alpha=0.15)\n"
         f"b.plot(epochs, val_acc)\nb.axhline(0.9, linestyle=':')\nb.axhspan(0.9, 1, color='green', alpha=0.1)\nb.set_ylim(0, 1)\n"
         f"for ax in (a, b):\n    ax.axvline(best, linestyle='--')",
         {"post": "[(fc.lines(a, 'ls'), [[fc.r(p.vertices, 4) for p in c.get_paths()] for c in a.collections], fc.patches(a)) for a in fc.axes()], "
                  "fc.limits(fc.ax(1))['ylim']",
          "setup": f"seeds = np.round(train_loss + 0.05 * np.random.default_rng({s}).normal(size=(5, len(epochs))), 4)", "tol": 1e-6}),
        ("errorbar", "Overfitting at a glance: plot `train_loss` and `val_loss`, fill between them in red where "
                     "`val_loss - train_loss` exceeds its median (interpolated, `alpha=0.25`), and at every 5th epoch "
                     "draw `val_loss` as dots with error bars of ±0.05 (`fmt='o'`, `capsize=3`).",
         "fig, ax = plt.subplots()\nax.plot(epochs, train_loss)\nax.plot(epochs, val_loss)\ngap = val_loss - train_loss\n"
         "ax.fill_between(epochs, train_loss, val_loss, where=gap > np.median(gap), interpolate=True, color='red', alpha=0.25)\n"
         "ax.errorbar(epochs[::5], val_loss[::5], yerr=0.05, fmt='o', capsize=3)",
         {"post": f"{POLYS}, {EB}"}),
        ("stackplot", "Where the loss comes from: the total `val_loss` is split into `val_loss - train_loss` (gap) and "
                      "`train_loss`. Draw them as a stack (train at the bottom, labels 'train', 'gap'), then the total as "
                      "a black line on top and a dashed vertical line at the epoch where the gap first exceeds 0.2 (if it "
                      "ever does); legend.",
         "fig, ax = plt.subplots()\ngap = val_loss - train_loss\nax.stackplot(epochs, train_loss, gap, labels=['train', 'gap'])\n"
         "ax.plot(epochs, val_loss, color='k')\nhit = np.flatnonzero(gap > 0.2)\nif len(hit):\n    ax.axvline(epochs[hit[0]], linestyle='--')\nax.legend()",
         {"post": f"{POLYS}, fc.lines(fc.ax(), 'color', 'ls'), fc.legend()"}),
        ("step", "A schedule figure: the learning rate halves every 5 epochs (`lr = 0.1 * 0.5 ** ((epochs - 1) // 5)`) "
                 "as a step line (`where='post'`) on a log axis, the decay epochs (6, 11, …) marked with `vlines` from "
                 "the lowest to the highest lr in grey, and `stem` markers of `val_acc` on a twin axis (`ax.twinx()`).",
         "lr = 0.1 * 0.5 ** ((epochs - 1) // 5)\nfig, ax = plt.subplots()\nax.step(epochs, lr, where='post')\nax.set_yscale('log')\n"
         "d = epochs[(epochs - 1) % 5 == 0][1:]\nax.vlines(d, lr.min(), lr.max(), colors='gray')\nax2 = ax.twinx()\nax2.stem(epochs, val_acc)",
         {"post": "fc.lines(fc.ax(0), 'drawstyle'), fc.scales(fc.ax(0)), [fc.r(c.get_segments(), 4) for c in fc.ax(0).collections], "
                  "[fc.r(c.markerline.get_xydata(), 4) for c in fc.ax(1).containers]", "require": ["step", "vlines", "stem", "twinx"]}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "hard")
def area_math(rng):
    s = seeded(rng)
    n = rng.choice([41, 61, 81])
    T = rng.choice([50, 60, 80])
    setup = (f"g = np.random.default_rng({s})\nx = np.linspace(0, 10, {n})\n"
             f"y = np.round(np.sin(x) + 0.3 * g.normal(size={n}), 3) + 1e-4\n"
             f"lower = np.round(0.5 * np.cos(x / 2) - 1, 3)\nupper = np.round(lower + 0.5 + g.random({n}), 3)")
    items = [
        ("fill_between", "`fill_between(x, lower, upper)` draws one polygon. Return its area two ways, as a list: with "
                         "the shoelace formula over the polygon's vertices (`coll.get_paths()[0].vertices`; area = "
                         "½·|Σ (xᵢ·yᵢ₊₁ − xᵢ₊₁·yᵢ)|, wrapping around), and as `np.trapezoid(upper - lower, x)`.",
         "fig, ax = plt.subplots()\ncoll = ax.fill_between(x, lower, upper)\nv = coll.get_paths()[0].vertices\n"
         "a = abs(np.sum(v[:, 0] * np.roll(v[:, 1], -1) - np.roll(v[:, 0], -1) * v[:, 1])) / 2\n"
         "result = [a, np.trapezoid(upper - lower, x)]",
         {"post": f"_, {POLYS}", "require": ["fill_between", "get_paths", "trapezoid"], "tol": 1e-9}),
        ("interpolate", "`fill_between(x, 0, y, where=y > 0, interpolate=True)` starts each polygon exactly where `y` "
                        "crosses 0 upwards. Compute those crossings yourself — for every i with `y[i] < 0 < y[i + 1]`, "
                        "the x where the straight segment between the two samples meets 0 — then draw the fill and "
                        "return the crossings as a list.",
         "i = np.flatnonzero((y[:-1] < 0) & (y[1:] > 0))\ncross = x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i])\n"
         "fig, ax = plt.subplots()\nax.fill_between(x, 0, y, where=y > 0, interpolate=True)\nresult = cross.tolist()",
         {"post": f"_, {POLYS}", "require": ["fill_between", "interpolate"], "tol": 1e-9}),
        ("get_paths", "How much of the x-axis does `fill_between(x, lower, y, where=y > lower, interpolate=True)` "
                      "cover? Draw it and read the answer off the polygons: each one spans from its smallest to its "
                      "largest vertex x. Return the total.",
         "fig, ax = plt.subplots()\ncoll = ax.fill_between(x, lower, y, where=y > lower, interpolate=True)\n"
         "result = sum(p.vertices[:, 0].max() - p.vertices[:, 0].min() for p in coll.get_paths())",
         {"post": f"_, {POLYS}", "require": ["fill_between", "get_paths"], "tol": 1e-9}),
        ("stackplot", "`stackplot` puts layer k on top of layers 0 … k−1. Stack the rows of `parts` over `x` and return, "
                      "as a list, the top of the whole stack read back from the last area's polygon (its largest vertex "
                      "y) and the same number computed with `np.cumsum(parts, axis=0)`.",
         "fig, ax = plt.subplots()\nareas = ax.stackplot(x, parts)\ntop = areas[-1].get_paths()[0].vertices[:, 1].max()\n"
         "result = [top, np.cumsum(parts, axis=0)[-1].max()]",
         {"post": f"_, {POLYS}", "require": ["stackplot", "cumsum"], "tol": 1e-9,
          "setup": "parts = np.round(np.abs(np.vstack([np.sin(x), np.cos(x), np.sin(x / 2)])), 3)"}),
        ("errorbar", "Draw asymmetric error bars on every 5th point: `ax.errorbar(x[::5], y[::5], yerr=[lo, hi], "
                     "fmt='o')` with `lo = 0.2 + 0.1 * np.abs(y[::5])` below and `hi = 2 * lo` above. Then set the y "
                     "limits yourself so every bar fits with a padding of 10% of the covered range on each side, and "
                     "return the limits as a list.",
         "lo = 0.2 + 0.1 * np.abs(y[::5])\nhi = 2 * lo\nfig, ax = plt.subplots()\nax.errorbar(x[::5], y[::5], yerr=[lo, hi], fmt='o')\n"
         "bottom, top = (y[::5] - lo).min(), (y[::5] + hi).max()\npad = 0.1 * (top - bottom)\n"
         "ax.set_ylim(bottom - pad, top + pad)\nresult = [bottom - pad, top + pad]",
         {"post": f"_, {EB}, fc.limits()['ylim']", "require": ["errorbar", "set_ylim"], "tol": 1e-9}),
        ("broken_barh", f"Each lane of `jobs` is a list of (start, duration) bars that **may overlap**; lane i is drawn "
                        f"with `ax.broken_barh(jobs[i], (i - 0.4, 0.8))`. Draw the chart and return each lane's "
                        f"utilization of the window 0 … {T}: the length of the union of its bars, clipped to the window, "
                        f"divided by {T}.",
         f"fig, ax = plt.subplots()\nutil = []\nfor i, row in enumerate(jobs):\n    ax.broken_barh(row, (i - 0.4, 0.8))\n"
         f"    busy, end = 0.0, 0.0\n    for s, d in sorted(row):\n        a, b = max(s, end), min(s + d, {T})\n"
         f"        if b > a:\n            busy += b - a\n        end = max(end, s + d)\n    util.append(busy / {T})\nresult = util",
         {"post": f"_, {POLYS}", "require": ["broken_barh"], "tol": 1e-9,
          "setup": f"jobs = [[(float(a), float(d)) for a, d in zip(np.sort(g.integers(0, {T}, 5)), g.integers(3, 15, 5))] "
                   f"for _ in range(3)]"}),
        ("eventplot", "For each spike train in `spikes`, find its longest silent gap between two consecutive events "
                      "(the first one if several are equally long). Draw the trains with `ax.eventplot(spikes, "
                      "lineoffsets=[0, 1, 2], linelengths=0.6)`, mark each train's longest gap as a red bar on its row "
                      "(`ax.broken_barh([(start, length)], (i - 0.3, 0.6), facecolors='red', alpha=0.3)`), and return "
                      "the gap lengths.",
         "fig, ax = plt.subplots()\nax.eventplot(spikes, lineoffsets=[0, 1, 2], linelengths=0.6)\ngaps = []\n"
         "for i, t in enumerate(spikes):\n    d = np.diff(t)\n    j = int(d.argmax())\n"
         "    ax.broken_barh([(t[j], d[j])], (i - 0.3, 0.6), facecolors='red', alpha=0.3)\n    gaps.append(float(d[j]))\n"
         "result = gaps",
         {"post": f"_, {POLYS}", "require": ["eventplot", "broken_barh", "diff"], "tol": 1e-9,
          "setup": "spikes = [np.round(np.sort(g.uniform(0, 80, g.integers(6, 12))), 1) for _ in range(3)]"}),
        ("stairs", "The learning rate `lr` is constant within each epoch (epoch e spans e … e + 1): draw it with "
                   "`ax.stairs(lr, np.arange(len(lr) + 1), fill=True, alpha=0.4)`. The area under the staircase is the "
                   "total learning-rate 'budget'. Find the first epoch boundary by which at least half of it has been "
                   "spent, mark it with a dashed `axvline`, and return it.",
         "fig, ax = plt.subplots()\nax.stairs(lr, np.arange(len(lr) + 1), fill=True, alpha=0.4)\nspent = np.cumsum(lr)\n"
         "e = int(np.argmax(spent >= spent[-1] / 2)) + 1\nax.axvline(e, linestyle='--')\nresult = e",
         {"post": "_, fc.lines(fc.ax(), 'ls'), [(fc.r(p.get_data().values, 5), p.get_fill()) for p in fc.ax().patches]",
          "require": ["stairs", "cumsum", "axvline"],
          "setup": f"lr = np.round(0.1 * np.minimum(np.arange(1, {n // 2}) / 5, 1) * 0.93 ** np.arange({n // 2 - 1}), 5)"}),
    ]
    return catalog(rng, items, setup, f"`x` holds {n} points from 0 to 10, `y` a noisy wave, and `lower` < `upper` the "
                                      f"edges of a band.")


# ================================================================= challenge

def _gantt_setup(seed: int, n: int, L: int, T: int, D: int, report: str) -> str:
    return (f"g = np.random.default_rng({seed})\nlane = g.integers(0, {L}, {n})\nstart = g.integers(0, {T}, {n})\n"
            f"dur = g.integers(1, {D} + 1, {n})\nL = {L}\nreport = {report!r}")


@template(M, "challenge")
def gantt_lanes(rng):
    s = seeded(rng)
    report = rng.choice(["busy", "peak", "gaps"])
    n, L = rng.randint(6, 12), rng.randint(2, 3)
    setup = _gantt_setup(s, n, L, 30, 8, report)
    head = ("order = np.lexsort((start, lane))\nln, st = lane[order], start[order]\nen = st + dur[order]\n"
            "big = (en.max() + 1) * ln             # lift each lane above the previous one\n"
            "reach = np.maximum.accumulate(en + big)\nprev = np.r_[-1, reach[:-1]]\n"
            "new = (st + big > prev) | np.r_[True, ln[1:] != ln[:-1]]   # a bar that starts a new busy block\n"
            "blk = np.cumsum(new) - 1\nb_start = st[new]\nb_end = np.zeros(new.sum(), dtype=np.int64)\n"
            "np.maximum.at(b_end, blk, en)\nb_lane = ln[new]\n")
    sol = {"busy": head + "result = np.bincount(b_lane, weights=b_end - b_start, minlength=L).astype(np.int64)",
           "gaps": head + "result = np.maximum(np.bincount(b_lane, minlength=L) - 1, 0)",
           "peak": ("t = np.concatenate([start, start + dur])\ndelta = np.concatenate([np.ones(len(start), int), -np.ones(len(start), int)])\n"
                    "o = np.lexsort((delta, t))            # at equal times, ends (-1) come before starts (+1)\n"
                    "result = int(np.cumsum(delta[o]).max())")}[report]
    tests = [_gantt_setup(s + i, nn, ll, tt, dd, report) for i, (nn, ll, tt, dd) in
             ((1, (300000, 1000, 10 ** 9, 10 ** 6)), (2, (300000, 3, 10 ** 6, 50)))]
    want = {"busy": "an array of length L: each lane's **busy time**, the total length of the union of its bars",
            "peak": "the largest number of bars (over **all** lanes) that overlap at one instant — e.g. the most jobs "
                    "running at once",
            "gaps": "an array of length L: for each lane, the number of idle gaps between its first bar's start and its "
                    "last bar's end (0 for a lane with no bars)"}[report]
    return challenge(
        "Gantt chart bookkeeping",
        "A `broken_barh` Gantt chart has one row (lane) per worker; bar i lies on lane `lane[i]` and covers the "
        "half-open time interval `[start[i], start[i] + dur[i])` (integers). Bars on one lane may overlap; bars that "
        "merely touch (one ends where the next starts) leave no gap and don't overlap. "
        f"Return {want}. (`report` = {report!r}.)",
        setup, sol, tests, "n ≤ 3·10^5 bars, L ≤ 1000 lanes, times ≤ 10^9", speed="O(n log n)")


def _band_setup(seed: int, n: int, report: str) -> str:
    return (f"g = np.random.default_rng({seed})\nx = np.cumsum(g.uniform(0.1, 1.0, {n}))\n"
            f"y1 = np.round(np.sin(x / 3) + g.normal(0, 0.3, {n}), 3)\ny2 = np.round(np.cos(x / 5) + g.normal(0, 0.3, {n}), 3) + 1e-4\n"
            f"report = {report!r}")


@template(M, "challenge")
def band_area(rng):
    s = seeded(rng)
    report = rng.choice(["positive", "between", "regions"])
    n = rng.randint(8, 16)
    setup = _band_setup(s, n, report)
    common = ("d = y1 - y2\nh = np.diff(x)\na, b = d[:-1], d[1:]\n")
    sol = {"positive": common + ("both = (a >= 0) & (b >= 0)\ncross = (a > 0) != (b > 0)\npos = np.maximum(a, b)\n"
                                 "tri = np.where(cross, pos ** 2 / (np.abs(a) + np.abs(b) + (~cross)) * h / 2, 0.0)\n"
                                 "result = float(np.sum(np.where(both, (a + b) / 2 * h, tri)))"),
           "between": common + ("cross = (a > 0) != (b > 0)\nsame = np.abs(a + b) / 2 * h\n"
                                "split = (a ** 2 + b ** 2) / (np.abs(a) + np.abs(b) + (~cross)) * h / 2\n"
                                "result = float(np.sum(np.where(cross, split, same)))"),
           "regions": "w = y1 > y2\nresult = int(np.sum(w[1:] & ~w[:-1]) + w[0])"}[report]
    tests = [_band_setup(s + i, nn, report) for i, nn in ((1, 1000000), (2, 300000))]
    want = {"positive": "the total area that `ax.fill_between(x, y1, y2, where=y1 > y2, interpolate=True)` fills: the "
                        "integral of max(y1 − y2, 0), with both curves linear between samples (so a stretch ends exactly "
                        "where the curves cross)",
            "between": "the total area between the two curves, the integral of |y1 − y2| (curves linear between samples)",
            "regions": "how many separate polygons `ax.fill_between(x, y1, y2, where=y1 > y2)` draws: one per maximal "
                       "run of consecutive samples where `y1 > y2`"}[report]
    return challenge(
        "Area between curves",
        "`fill_between` shades the area between two sampled curves, `y1` and `y2`, over the sorted positions `x`; the "
        "curves are straight between neighbouring samples. Where the difference changes sign inside an interval, the "
        f"crossing point is found by linear interpolation. Return {want}. (`report` = {report!r}.)",
        setup, sol, tests, "n ≤ 10^6 samples", speed="O(n), vectorized", tol=1e-6)


def _clip_setup(seed: int, q: int, report: str) -> str:
    return (f"g = np.random.default_rng({seed})\nxlim, ylim = (0.0, 10.0), (-2.0, 3.0)\npx = np.round(g.uniform(-5, 15, {q}), 3)\n"
            f"py = np.round(g.uniform(-6, 7, {q}), 3)\nslope = np.round(np.tan(g.uniform(-1.5, 1.5, {q})), 4)\n"
            f"slope[slope == 0] = 0.5\nreport = {report!r}")


@template(M, "challenge")
def axline_clip(rng):
    s = seeded(rng)
    report = rng.choice(["segment", "visible", "length"])
    q = rng.randint(4, 8)
    setup = _clip_setup(s, q, report)
    sol = ("(x0, x1), (y0, y1) = xlim, ylim\nX = np.column_stack([np.full_like(px, x0), np.full_like(px, x1), px + (y0 - py) / slope, px + (y1 - py) / slope])\n"
           "Y = np.column_stack([py + (x0 - px) * slope, py + (x1 - px) * slope, np.full_like(px, y0), np.full_like(px, y1)])\n"
           "o = np.lexsort((Y, X), axis=1)[:, 1:3] if False else np.argsort(X + 0 * Y, axis=1, kind='stable')[:, 1:3]\n"
           "r = np.arange(len(px))[:, None]\nsx, sy = X[r, o], Y[r, o]          # the middle two of the four crossings\n"
           "inside = np.all((sx >= x0 - 1e-9) & (sx <= x1 + 1e-9) & (sy >= y0 - 1e-9) & (sy <= y1 + 1e-9), axis=1)\n"
           + {"segment": "result = np.column_stack([sx[:, 0], sy[:, 0], sx[:, 1], sy[:, 1]])",
              "visible": "result = int(inside.sum())",
              "length": "result = float(np.sum(np.where(inside, np.hypot(sx[:, 1] - sx[:, 0], sy[:, 1] - sy[:, 0]), 0.0)))"}[report])
    tests = [_clip_setup(s + i, qq, report) for i, qq in ((1, 300000), (2, 100000))]
    want = {"segment": "a q × 4 array `[sx, sy, ex, ey]`: the start and end point Matplotlib computes for each line",
            "visible": "how many of the lines actually cross the view (their segment lies on the view's border)",
            "length": "the total length, in data units, of the visible parts of all lines (lines that miss the view "
                      "contribute 0)"}[report]
    return challenge(
        "Drawing an infinite line",
        "`ax.axline((px, py), slope=m)` has no endpoints; at draw time Matplotlib computes the four points where the "
        "line meets the view's border lines x = xlim[0], x = xlim[1], y = ylim[0], y = ylim[1], sorts them by x (then "
        "y), and draws the segment between the **middle two** (the rest is clipped away). If the line misses the view, "
        "those two points lie outside it and nothing shows. Line i passes through `(px[i], py[i])` with slope "
        f"`slope[i]` (never 0 or infinite here). Return {want}. (`report` = {report!r}.)",
        setup, sol, tests, "q ≤ 3·10^5 lines", speed="O(q), vectorized", tol=1e-6)
