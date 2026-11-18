"""5.08 Titles, labels, limits, scales & ticks (set_title / labels, limits and margins, linear / log / symlog / logit
scales, tick locators and formatters, tick_params, spines and grids, dates on an axis)."""

from ._common import catalog, challenge, curves, lit, seeded, signal, template

M = "5.08"
LINES = "fc.lines(fc.ax())"
LAB = "fc.labels(fc.ax())"
XT = "fc.ticks(fc.ax(), 'x')"
YT = "fc.ticks(fc.ax(), 'y')"
PAD = "fc.r(fc.ax().titleOffsetTrans.get_matrix()[1, 2] * 72 / fc.fig().dpi, 3)"
MT = "fc.ax().{}axis.get_major_ticks()[0]"


def _tick(axis: str, *props: str) -> str:
    """Post code: properties of the first major tick of an axis (rotation, size, dir, length, color, tickcolor)."""
    get = {"rotation": "fc.r(t.label1.get_rotation(), 2)", "size": "fc.r(t.label1.get_fontsize(), 2)",
           "dir": "t.get_tickdir()", "length": "fc.r(t.tick1line.get_markersize(), 2)",
           "color": "fc.hexcolor(t.label1.get_color())", "tickcolor": "fc.hexcolor(t.tick1line.get_color())",
           "ha": "t.label1.get_ha()", "top": "(t.tick2line.get_visible(), t.label2.get_visible())"}
    return f"(lambda t: [{', '.join(get[p] for p in props)}])({MT.format(axis)})"


def _rot(axis: str = "x") -> str:
    return f"sorted({{fc.r(t.get_rotation(), 2) for t in fc.ax().get_{axis}ticklabels()}})"


def _fmt(axis: str, values) -> str:
    return f"fc.formatted(fc.ax(), {axis!r}, {list(values)!r})"


def _days(rng):
    """`dates` (daily datetime64) and a cumulative metric `value`."""
    y0 = rng.choice([2023, 2024, 2025])
    m0 = rng.choice([1, 2, 3, 5])
    months = rng.choice([3, 4, 6, 8])
    m1y, m1 = y0 + (m0 + months - 1) // 12, (m0 + months - 1) % 12 + 1
    s = seeded(rng)
    code = (f"dates = np.arange('{y0}-{m0:02d}-01', '{m1y}-{m1:02d}-01', dtype='datetime64[D]')\n"
            f"value = np.round(np.random.default_rng({s}).normal(size=len(dates)).cumsum(), 3)")
    return code, (f"`dates` holds the days from {y0}-{m0:02d}-01 for {months} months (a NumPy `datetime64[D]` array) "
                  f"and `value` a daily metric"), y0, m0, months


# ================================================================= lite

@template(M, "lite")
def titles_labels(rng):
    code, desc, n = curves(rng)
    title = rng.choice(["Training loss", "Run 12", "Loss curves", "Validation"])
    xl, yl = rng.choice([("epoch", "loss"), ("step", "cross-entropy"), ("epoch", "NLL")])
    loc = rng.choice(["left", "right"])
    fs = rng.choice([12, 14, 16])
    lp = rng.choice([4, 8, 12])
    pad = rng.choice([6, 10, 20])
    sup = rng.choice(["Experiment A", "Ablation", "Baseline"])
    plot = "fig, ax = plt.subplots()\nax.plot(epochs, train_loss)\n"
    items = [
        (".set_title", f"Plot `train_loss` against `epochs` on `fig, ax` and title the Axes {title!r}.",
         f"{plot}ax.set_title({title!r})", {"post": f"{LINES}, fc.titles()"}),
        ("loc", f"Plot `train_loss` against `epochs` and put the title {title!r} on the **{loc}** instead of the "
                f"center (`loc=`).", f"{plot}ax.set_title({title!r}, loc={loc!r})",
         {"post": f"{LINES}, fc.titles()", "require": ["set_title|title", "loc"]}),
        ("fontweight", f"Plot `train_loss` against `epochs` and title it {title!r} in bold at font size {fs} "
                       f"(`fontsize`, `fontweight`).", f"{plot}ax.set_title({title!r}, fontsize={fs}, fontweight='bold')",
         {"post": f"{LINES}, fc.titles(), [fc.ax().title.get_fontsize(), fc.ax().title.get_fontweight()]",
          "require": ["fontsize|size", "fontweight|weight"]}),
        (".set_xlabel", f"Plot `train_loss` against `epochs` and label the x-axis {xl!r} and the y-axis {yl!r}.",
         f"{plot}ax.set_xlabel({xl!r})\nax.set_ylabel({yl!r})", {"post": f"{LINES}, {LAB}",
                                                                  "require": ["set_xlabel|xlabel", "set_ylabel|ylabel"]}),
        ("labelpad", f"Plot `train_loss` against `epochs`, label the y-axis {yl!r} and move that label {lp} points "
                     f"away from the ticks (`labelpad`).", f"{plot}ax.set_ylabel({yl!r}, labelpad={lp})",
         {"post": f"{LINES}, {LAB}, fc.ax().yaxis.labelpad", "require": ["labelpad"]}),
        ("fontsize", f"Plot `train_loss` against `epochs` and label the x-axis {xl!r} at font size {fs}.",
         f"{plot}ax.set_xlabel({xl!r}, fontsize={fs})",
         {"post": f"{LINES}, {LAB}, fc.ax().xaxis.label.get_fontsize()", "require": ["fontsize|size"]}),
        ("pad", f"Plot `train_loss` against `epochs` and title it {title!r} with {pad} points of space between the "
                f"title and the Axes (`pad=`).", f"{plot}ax.set_title({title!r}, pad={pad})",
         {"post": f"{LINES}, fc.titles(), {PAD}", "require": ["pad"]}),
        ("plt.title", f"With pyplot only, plot `train_loss` against `epochs`, title it {title!r} and label the axes "
                      f"{xl!r} and {yl!r}.", f"plt.plot(epochs, train_loss)\nplt.title({title!r})\nplt.xlabel({xl!r})\n"
                                              f"plt.ylabel({yl!r})",
         {"post": f"{LINES}, {LAB}", "require": ["title", "xlabel", "ylabel"]}),
        ("suptitle", f"Plot `train_loss` and `val_loss` on two Axes side by side (`plt.subplots(1, 2)`), titled "
                     f"'train' and 'val', and give the whole figure the title {sup!r}.",
         f"fig, (a1, a2) = plt.subplots(1, 2)\na1.plot(epochs, train_loss)\na1.set_title('train')\n"
         f"a2.plot(epochs, val_loss)\na2.set_title('val')\nfig.suptitle({sup!r})",
         {"post": "[(fc.lines(a), a.get_title()) for a in fc.axes()], fc.suptitles()['suptitle']",
          "require": ["suptitle"]}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "lite")
def limits_scales(rng):
    code, desc, n = curves(rng, acc=True)
    lo, hi = rng.choice([(1, 10), (5, n), (0, n + 5)])
    b = rng.choice([0, 0.1, -0.5])
    m = rng.choice([0, 0.1, 0.2])
    lt = rng.choice([0.01, 0.1, 1])
    plot = "fig, ax = plt.subplots()\nax.plot(epochs, train_loss)\n"
    items = [
        (".set_xlim", f"Plot `train_loss` against `epochs` and limit the x-axis to {lo} … {hi}.",
         f"{plot}ax.set_xlim({lo}, {hi})", {"post": f"{LINES}, fc.limits()['xlim']", "require": ["set_xlim|xlim"]}),
        ("bottom", f"Plot `train_loss` against `epochs` and fix only the **bottom** of the y-axis at {b} "
                   f"(`set_ylim(bottom=...)`; the top stays automatic).", f"{plot}ax.set_ylim(bottom={b})",
         {"post": f"{LINES}, fc.limits()['ylim']", "require": ["set_ylim|ylim"]}),
        ("margins", f"Plot `train_loss` against `epochs` and set the x margin to {m} (`ax.margins(x=...)`: the "
                    f"padding around the data, as a fraction of its range).", f"{plot}ax.margins(x={m})",
         {"post": f"{LINES}, fc.r(fc.ax().margins(), 4), fc.limits()['xlim']", "require": ["margins"]}),
        ("autoscale", "Plot `train_loss` against `epochs` and make the y-axis fit the data tightly, without margins "
                      "(`ax.autoscale(axis='y', tight=True)`).", f"{plot}ax.autoscale(axis='y', tight=True)",
         {"post": f"{LINES}, fc.limits()['ylim']", "require": ["autoscale"]}),
        ("set_yscale", "Plot `train_loss` against `epochs` on a **logarithmic** y-axis.", f"{plot}ax.set_yscale('log')",
         {"post": f"{LINES}, fc.scales()", "require": ["set_yscale|yscale|semilogy"]}),
        ("symlog", f"Plot `train_loss - val_loss` (it crosses zero) against `epochs` on a **symmetric log** y-axis "
                   f"that is linear between -{lt} and {lt} (`linthresh={lt}`).",
         f"fig, ax = plt.subplots()\nax.plot(epochs, train_loss - val_loss)\nax.set_yscale('symlog', linthresh={lt})",
         {"post": f"{LINES}, fc.scales(), fc.ax().yaxis.get_transform().linthresh",
          "require": ["set_yscale|yscale", "linthresh"]}),
        ("logit", "Plot `val_acc` against `epochs` on a **logit** y-axis (stretches values near 0 and 1).",
         "fig, ax = plt.subplots()\nax.plot(epochs, val_acc)\nax.set_yscale('logit')",
         {"post": f"{LINES}, fc.scales()", "require": ["set_yscale|yscale"]}),
        ("invert_yaxis", "Plot `train_loss` against `epochs` with the y-axis flipped (large values at the bottom).",
         f"{plot}ax.invert_yaxis()", {"post": f"{LINES}, fc.ax().yaxis_inverted()", "require": ["invert_yaxis|set_ylim"]}),
        ("set_aspect", "Plot `train_loss` against `val_loss` (one point per epoch) with equal scales on both axes "
                       "(`ax.set_aspect('equal')`: one unit is as long in x as in y).",
         "fig, ax = plt.subplots()\nax.plot(train_loss, val_loss, 'o')\nax.set_aspect('equal')",
         {"post": f"{LINES}, fc.ax().get_aspect()", "require": ["set_aspect|axis"]}),
        ("set_box_aspect", "Plot `train_loss` against `epochs` in an Axes box whose height is 0.6 × its width "
                           "(`ax.set_box_aspect(0.6)`).", f"{plot}ax.set_box_aspect(0.6)",
         {"post": f"{LINES}, fc.ax().get_box_aspect()", "require": ["set_box_aspect"]}),
        ("plt.xlim", f"With pyplot only, plot `train_loss` against `epochs` on a log y-axis with the x-axis limited "
                     f"to {lo} … {hi}.", f"plt.plot(epochs, train_loss)\nplt.yscale('log')\nplt.xlim({lo}, {hi})",
         {"post": f"{LINES}, fc.scales(), fc.limits()['xlim']", "require": ["yscale|semilogy", "xlim"]}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "lite")
def tick_basics(rng):
    code, desc, n = signal(rng)
    pos = rng.choice([[0, 5, 10], [0, 2.5, 5, 7.5, 10], [1, 4, 7]])
    names = rng.choice([["start", "mid", "end"], ["a", "b", "c", "d", "e"], ["lo", "mid", "hi"]])[:len(pos)]
    names = names + [f"t{i}" for i in range(len(names), len(pos))]
    rot = rng.choice([30, 45, 90])
    base = rng.choice([1, 2, 2.5])
    ln = rng.choice([4, 6, 8])
    col = rng.choice(["gray", "tab:blue", "red"])
    plot = "fig, ax = plt.subplots()\nax.plot(x, y)\n"
    items = [
        (".set_xticks", f"Plot `y` against `x` and put the x ticks exactly at {pos}.", f"{plot}ax.set_xticks({pos})",
         {"post": f"{LINES}, {XT}", "require": ["set_xticks|xticks"]}),
        ("labels", f"Plot `y` against `x` and put x ticks at {pos} labelled {names} (`set_xticks(..., labels=...)`).",
         f"{plot}ax.set_xticks({pos}, labels={names})", {"post": f"{LINES}, {XT}", "require": ["set_xticks|xticks"]}),
        ("minor", "Plot `y` against `x` and add **minor** x ticks at every integer 0 … 10 "
                  "(`ax.set_xticks(np.arange(11), minor=True)`).", f"{plot}ax.set_xticks(np.arange(11), minor=True)",
         {"post": f"{LINES}, fc.ticks(fc.ax(), 'x', minor=True)['locs']", "require": ["minor"]}),
        ("labelrotation", f"Plot `y` against `x` and rotate the x tick labels by {rot}° with `ax.tick_params`.",
         f"{plot}ax.tick_params(axis='x', labelrotation={rot})",
         {"post": f"{LINES}, {_rot('x')}, {_rot('y')}", "require": ["tick_params|xticks"]}),
        ("direction", f"Plot `y` against `x` and make the ticks of both axes point **into** the Axes, {ln} points "
                      f"long (`ax.tick_params(direction='in', length={ln})`).",
         f"{plot}ax.tick_params(direction='in', length={ln})",
         {"post": f"{LINES}, {_tick('x', 'dir', 'length')}, {_tick('y', 'dir', 'length')}", "require": ["tick_params"]}),
        ("colors", f"Plot `y` against `x` and color the **y** ticks and their labels {col!r} "
                   f"(`ax.tick_params(axis='y', colors=...)`).", f"{plot}ax.tick_params(axis='y', colors={col!r})",
         {"post": f"{LINES}, {_tick('y', 'color', 'tickcolor')}, {_tick('x', 'color')}", "require": ["tick_params"]}),
        ("set_yticks", "Plot `y` against `x` and remove all ticks (but keep the frame): `ax.set_xticks([])` and "
                       "`ax.set_yticks([])`.", f"{plot}ax.set_xticks([])\nax.set_yticks([])",
         {"post": f"{LINES}, {XT}, {YT}, fc.ax().axison", "require": ["set_xticks|xticks", "set_yticks|yticks"]}),
        ("minorticks_on", "Plot `y` against `x` and turn on the automatic minor ticks (`ax.minorticks_on()`).",
         f"{plot}ax.minorticks_on()",
         {"post": f"{LINES}, type(fc.ax().xaxis.get_minor_locator()).__name__, type(fc.ax().yaxis.get_minor_locator()).__name__"}),
        ("MultipleLocator", f"Plot `y` against `x` and put an x tick at every multiple of {base} "
                            f"(`ax.xaxis.set_major_locator(mticker.MultipleLocator({base}))`, with `import "
                            f"matplotlib.ticker as mticker`).",
         f"import matplotlib.ticker as mticker\n{plot}ax.xaxis.set_major_locator(mticker.MultipleLocator({base}))",
         {"post": f"{LINES}, {XT}['locs']", "require": ["MultipleLocator"]}),
        ("plt.xticks", f"With pyplot only, plot `y` against `x`, put x ticks at {pos} and rotate their labels by "
                       f"{rot}° (`plt.xticks(..., rotation=...)`).", f"plt.plot(x, y)\nplt.xticks({pos}, rotation={rot})",
         {"post": f"{LINES}, {XT}['locs'], {_rot('x')}", "require": ["xticks"]}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "lite")
def spines_and_grid(rng):
    code, desc, n = signal(rng)
    out = rng.choice([5, 8, 10])
    ls = rng.choice(["--", ":", "-."])
    al = rng.choice([0.3, 0.5, 0.7])
    fc_ = rng.choice(["#fafafa", "#f0f0f0", "lavender", "ivory"])
    lw = rng.choice([1.5, 2, 3])
    plot = "fig, ax = plt.subplots()\nax.plot(x, y)\n"
    gl = "fc.ax().{}axis.get_{}_ticks()[0].gridline"
    items = [
        ("spines", "Plot `y` against `x` and hide the top and right spines (`ax.spines[['top', 'right']]"
                   ".set_visible(False)`).", f"{plot}ax.spines[['top', 'right']].set_visible(False)",
         {"post": f"{LINES}, fc.spines()", "require": ["spines", "set_visible"]}),
        ("set_position", f"Plot `y` against `x` and move the left spine {out} points outward "
                         f"(`ax.spines['left'].set_position(('outward', {out}))`).",
         f"{plot}ax.spines['left'].set_position(('outward', {out}))",
         {"post": f"{LINES}, fc.ax().spines['left'].get_position()", "require": ["set_position"]}),
        (".grid", f"Plot `y` against `x` and turn on the grid with the line style {ls!r} and opacity {al}.",
         f"{plot}ax.grid(True, linestyle={ls!r}, alpha={al})",
         {"post": f"{LINES}, fc.grid_on(), (lambda g: [g.get_linestyle(), g.get_alpha()])({gl.format('x', 'major')})",
          "require": ["grid"]}),
        ("axis", "Plot `y` against `x` with grid lines for the **y** values only (horizontal lines: "
                 "`ax.grid(axis='y')`).", f"{plot}ax.grid(axis='y')", {"post": f"{LINES}, fc.grid_on()", "require": ["grid"]}),
        ("which", f"Plot `y` against `x`, turn on the minor ticks and draw grid lines at them with opacity {al} "
                  f"(`ax.grid(True, which='minor', alpha=...)`).", f"{plot}ax.minorticks_on()\nax.grid(True, which='minor', alpha={al})",
         {"post": f"{LINES}, (lambda g: [g.get_visible(), g.get_alpha()])({gl.format('x', 'minor')})",
          "require": ["minorticks_on", "which"]}),
        ("set_facecolor", f"Plot `y` against `x` on an Axes with the background color {fc_!r}.",
         f"{plot}ax.set_facecolor({fc_!r})", {"post": f"{LINES}, fc.hexcolor(fc.ax().get_facecolor())"}),
        ("axis", "Plot `y` against `x` and hide the axes completely (no spines, ticks or labels): `ax.axis('off')`.",
         f"{plot}ax.axis('off')", {"post": f"{LINES}, fc.ax().axison", "require": ["axis|set_axis_off"]}),
        ("set_color", "Plot `y` against `x` and color the bottom and left spines 'gray' (`ax.spines['bottom']"
                      ".set_color('gray')`, same for 'left').",
         f"{plot}ax.spines['bottom'].set_color('gray')\nax.spines['left'].set_color('gray')",
         {"post": f"{LINES}, [fc.hexcolor(fc.ax().spines[k].get_edgecolor()) for k in ('bottom', 'left', 'top')]",
          "require": ["spines"]}),
        ("set_linewidth", f"Plot `y` against `x` and make all four spines {lw} points thick.",
         f"{plot}for s in ax.spines.values():\n    s.set_linewidth({lw})",
         {"post": f"{LINES}, sorted({{k: fc.r(s.get_linewidth(), 3) for k, s in fc.ax().spines.items()}}.items())",
          "require": ["spines"]}),
    ]
    return catalog(rng, items, code, f"{desc}.")


# ================================================================= easy

@template(M, "easy")
def formatters(rng):
    n = rng.choice([10, 15, 20])
    s = seeded(rng)
    code = (f"g = np.random.default_rng({s})\nepochs = np.arange(1, {n + 1})\n"
            f"acc = np.round(np.clip(0.6 + 0.35 * (1 - np.exp(-epochs / 5)) + 0.01 * g.normal(size={n}), 0, 1), 4)\n"
            f"mem = np.round(2.0 ** g.uniform(20, 34, {n}))\nparams = np.round(np.linspace(1e5, 3e6, {n}))")
    desc = (f"`acc` holds {n} accuracies (fractions in 0 … 1) per epoch in `epochs`, `mem` memory use in bytes and "
            f"`params` parameter counts")
    d = rng.choice([0, 1])
    imp = "import matplotlib.ticker as mticker\n"
    acc = f"{imp}fig, ax = plt.subplots()\nax.plot(epochs, acc)\n"
    items = [
        ("PercentFormatter", "Plot `acc` against `epochs` and write the y tick labels as percentages: "
                             "`ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))` (1.0 means 100%).",
         f"{acc}ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))",
         {"post": f"{LINES}, {_fmt('y', (0.5, 0.75, 1.0))}"}),
        ("decimals", f"Plot `acc` against `epochs` with y tick labels as percentages with {d} decimal(s) "
                     f"(`PercentFormatter(xmax=1, decimals={d})`).",
         f"{acc}ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals={d}))",
         {"post": f"{LINES}, {_fmt('y', (0.5, 0.875, 1.0))}", "require": ["PercentFormatter", "decimals"]}),
        ("PercentFormatter", "Plot `acc * 100` (already in percent) against `epochs` and add a % sign to the y tick "
                             "labels (`PercentFormatter(xmax=100)`).",
         f"{imp}fig, ax = plt.subplots()\nax.plot(epochs, acc * 100)\nax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))",
         {"post": f"{LINES}, {_fmt('y', (50, 75, 100))}"}),
        ("FormatStrFormatter", "Plot `acc` against `epochs` and format the y tick labels with exactly 1 decimal "
                               "(`mticker.FormatStrFormatter('%.1f')`).",
         f"{acc}ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))",
         {"post": f"{LINES}, {_fmt('y', (0.6, 0.75, 1.0))}"}),
        ("StrMethodFormatter", "Plot `params` against `epochs` and write the y tick labels with thousands separators "
                               "and no decimals, like '1,500,000' (`mticker.StrMethodFormatter('{x:,.0f}')`).",
         f"{imp}fig, ax = plt.subplots()\nax.plot(epochs, params)\nax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{{x:,.0f}}'))",
         {"post": f"{LINES}, {_fmt('y', (0, 1500000, 3000000))}"}),
        ("FuncFormatter", "Plot `params` against `epochs` with y tick labels in thousands, like '1500k': "
                          "`mticker.FuncFormatter(lambda v, pos: f'{v / 1000:.0f}k')`.",
         f"{imp}fig, ax = plt.subplots()\nax.plot(epochs, params)\n"
         f"ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, pos: f'{{v / 1000:.0f}}k'))",
         {"post": f"{LINES}, {_fmt('y', (0, 1500000, 3000000))}"}),
        ("EngFormatter", "Plot `mem` against `epochs` with engineering-notation byte labels like '4 GB' "
                         "(`mticker.EngFormatter(unit='B')`).",
         f"{imp}fig, ax = plt.subplots()\nax.plot(epochs, mem)\nax.yaxis.set_major_formatter(mticker.EngFormatter(unit='B'))",
         {"post": f"{LINES}, {_fmt('y', (0, 2.0 ** 30, 1.5e10))}"}),
        ("FuncFormatter", "Plot `acc` against `epochs` and label the **x** ticks like 'ep 5' "
                          "(`FuncFormatter(lambda v, pos: f'ep {v:.0f}')` on `ax.xaxis`).",
         f"{acc}ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, pos: f'ep {{v:.0f}}'))",
         {"post": f"{LINES}, {_fmt('x', (0, 5, 10))}"}),
        ("ticklabel_format", "Plot `params` against `epochs` with the y labels in scientific notation, the common "
                             "power of ten moved to the top of the axis (`ax.ticklabel_format(axis='y', style='sci', "
                             "scilimits=(0, 0))`). Matplotlib shows it as '1e6'.",
         "fig, ax = plt.subplots()\nax.plot(epochs, params)\nax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))",
         {"post": f"{LINES}, {YT}['labels'], fc.ax().yaxis.get_major_formatter().get_offset()"}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "easy")
def locators(rng):
    code, desc, n = curves(rng, n=(20, 50))
    base = rng.choice([5, 10])
    nb = rng.choice([3, 4, 5, 6])
    mn = rng.choice([2, 4, 5])
    imp = "import matplotlib.ticker as mticker\n"
    plot = f"{imp}fig, ax = plt.subplots()\nax.plot(epochs, train_loss)\n"
    items = [
        ("MultipleLocator", f"Plot `train_loss` against `epochs` with an x tick every {base} epochs "
                            f"(`mticker.MultipleLocator({base})`).",
         f"{plot}ax.xaxis.set_major_locator(mticker.MultipleLocator({base}))", {"post": f"{LINES}, {XT}['locs']"}),
        ("MaxNLocator", f"Plot `train_loss` against `epochs` with at most {nb} intervals between y ticks "
                        f"(`mticker.MaxNLocator(nbins={nb})`).",
         f"{plot}ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins={nb}))", {"post": f"{LINES}, {YT}['locs']"}),
        ("integer", "Epochs are whole numbers: plot `val_loss` against the **first 6** epochs (`epochs[:6]`, "
                    "`val_loss[:6]`) and allow only integer x ticks (`mticker.MaxNLocator(integer=True)`).",
         f"{imp}fig, ax = plt.subplots()\nax.plot(epochs[:6], val_loss[:6])\n"
         f"ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))",
         {"post": f"{LINES}, {XT}['locs']", "require": ["MaxNLocator", "integer"]}),
        ("AutoMinorLocator", f"Plot `train_loss` against `epochs` and put {mn - 1} minor ticks between consecutive "
                             f"major x ticks (`ax.xaxis.set_minor_locator(mticker.AutoMinorLocator({mn}))`).",
         f"{plot}ax.xaxis.set_minor_locator(mticker.AutoMinorLocator({mn}))",
         {"post": f"{LINES}, fc.ticks(fc.ax(), 'x', minor=True)['locs']"}),
        ("NullLocator", "Plot `train_loss` against `epochs` with no y ticks at all "
                        "(`ax.yaxis.set_major_locator(mticker.NullLocator())`).",
         f"{plot}ax.yaxis.set_major_locator(mticker.NullLocator())", {"post": f"{LINES}, {YT}"}),
        ("set_minor_locator", f"Plot `train_loss` against `epochs` with major x ticks every {base} epochs and minor "
                              f"ones every epoch (two MultipleLocators), and grid lines at the minor ticks only "
                              f"(`ax.grid(True, which='minor', axis='x')`).",
         f"{plot}ax.xaxis.set_major_locator(mticker.MultipleLocator({base}))\n"
         f"ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))\nax.grid(True, which='minor', axis='x')",
         {"post": f"{LINES}, {XT}['locs'], fc.ticks(fc.ax(), 'x', minor=True)['locs'], "
                  f"fc.ax().xaxis.get_minor_ticks()[0].gridline.get_visible(), fc.grid_on()",
          "require": ["MultipleLocator", "set_minor_locator"]}),
        ("FixedLocator", "Plot `train_loss` against `epochs` and put y ticks only at the first, the best (lowest) and "
                         "the last loss: `mticker.FixedLocator([...])` with those three values.",
         f"{plot}ax.yaxis.set_major_locator(mticker.FixedLocator([train_loss[0], train_loss.min(), train_loss[-1]]))",
         {"post": f"{LINES}, sorted({YT}['locs'])"}),
        ("LogLocator", "Plot `train_loss` against `epochs` on a log y-axis with a major tick at every power of 10 "
                       "**and** at 2× and 5× each power (`mticker.LogLocator(base=10, subs=(1, 2, 5))`).",
         f"{plot}ax.set_yscale('log')\nax.yaxis.set_major_locator(mticker.LogLocator(base=10, subs=(1, 2, 5)))",
         {"post": f"{LINES}, {YT}['locs']", "require": ["LogLocator", "subs"]}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "easy")
def axis_readback(rng):
    code, desc, n = signal(rng)
    m = rng.choice([0.1, 0.2, 0.25])
    b = rng.choice([-2, -1.5, 0])
    title = rng.choice(["left side", "Phase", "Signal"])
    plot = "fig, ax = plt.subplots()\nax.plot(x, y)\n"
    items = [
        ("get_xlim", "Plot `y` against `x` and return the automatic x limits as a list: the data range 0 … 10 plus "
                     "a 5% margin on each side.", f"{plot}result = list(ax.get_xlim())", {"tol": 1e-9}),
        ("margins", f"Plot `y` against `x` with `ax.margins({m})` and return the x limits (the margin is {m} of "
                    f"the data range on each side).", f"{plot}ax.margins({m})\nresult = list(ax.get_xlim())",
         {"tol": 1e-9}),
        ("get_ylim", f"Plot `y` against `x`, call `ax.set_ylim(bottom={b})` and return the y limits: the bottom is "
                     f"fixed, the top still comes from the data (5% margin).",
         f"{plot}ax.set_ylim(bottom={b})\nresult = list(ax.get_ylim())", {"tol": 1e-9}),
        ("invert_yaxis", "Plot `y` against `x`, flip the y-axis with `ax.invert_yaxis()` and return `ax.get_ylim()` "
                         "as a list: the limits come back reversed (bottom > top).",
         f"{plot}ax.invert_yaxis()\nresult = list(ax.get_ylim())", {"tol": 1e-9}),
        ("get_xticks", "Plot `y` against `x` and return the x tick positions Matplotlib chose (`ax.get_xticks()` as "
                       "a list). (Some may lie outside the visible range.)", f"{plot}result = ax.get_xticks().tolist()"),
        ("get_xticklabels", "Plot `y` against `x`, set x ticks at 0, 5, 10 labelled 'start', 'mid', 'end', and return "
                            "the label texts read back with `ax.get_xticklabels()`.",
         f"{plot}ax.set_xticks([0, 5, 10], labels=['start', 'mid', 'end'])\nresult = [t.get_text() for t in ax.get_xticklabels()]"),
        ("get_title", f"Plot `y` against `x`, set a left title {title!r} (`loc='left'`) and return "
                      f"`[ax.get_title(), ax.get_title(loc='left')]`: the default `get_title()` reads the **center** "
                      f"one.", f"{plot}ax.set_title({title!r}, loc='left')\nresult = [ax.get_title(), ax.get_title(loc='left')]"),
        ("get_yscale", "Plot `y + 2` against `x`, make the y-axis logarithmic and return `[ax.get_xscale(), "
                       "ax.get_yscale()]`.", "fig, ax = plt.subplots()\nax.plot(x, y + 2)\nax.set_yscale('log')\n"
                                            "result = [ax.get_xscale(), ax.get_yscale()]"),
        ("bar", "Bars stick to zero: draw `ax.bar(range(len(y)), np.abs(y))` and return the y limits. The bottom is "
                "exactly 0 (no margin below the bars' base), the top has the usual 5% margin.",
         "fig, ax = plt.subplots()\nax.bar(range(len(y)), np.abs(y))\nresult = list(ax.get_ylim())", {"tol": 1e-9}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "easy")
def log_scales(rng):
    n = rng.choice([30, 50, 80])
    s = seeded(rng)
    code = (f"g = np.random.default_rng({s})\nsteps = np.arange(1, {n + 1}) * 100\n"
            f"loss = np.round(3.0 * steps ** -0.5 * np.exp(0.05 * g.normal(size={n})), 5)\n"
            f"grad = np.round(g.normal(0, 1, {n}) * np.logspace(-3, 1, {n}), 6)\n"
            f"acc = np.round(1 - 0.5 * np.exp(-np.arange({n}) / 8), 5)")
    desc = (f"`loss` was measured at `steps` (100, 200, …), `grad` holds {n} gradient values of both signs spanning "
            f"many orders of magnitude, and `acc` an accuracy that creeps towards 1")
    al = rng.choice([0.2, 0.3])
    lt = rng.choice([1e-3, 1e-2])
    b2 = rng.choice([2, 10])
    items = [
        ("set_yscale", "Plot `loss` against `steps` on a log y-axis.",
         "fig, ax = plt.subplots()\nax.plot(steps, loss)\nax.set_yscale('log')",
         {"post": f"{LINES}, fc.scales()", "require": ["set_yscale|semilogy|yscale"]}),
        ("which", f"Plot `loss` against `steps` on a log y-axis with grid lines at both the major and the minor "
                  f"ticks (`ax.grid(True, which='both', alpha={al})`).",
         f"fig, ax = plt.subplots()\nax.plot(steps, loss)\nax.set_yscale('log')\nax.grid(True, which='both', alpha={al})",
         {"post": f"{LINES}, fc.scales(), [t.gridline.get_visible() for t in fc.ax().yaxis.get_minor_ticks()[:2]], fc.grid_on()",
          "require": ["grid", "which"]}),
        ("set_xscale", "A scaling-law plot: `loss` against `steps` with **both** axes logarithmic (a power law is a "
                       "straight line there).", "fig, ax = plt.subplots()\nax.plot(steps, loss)\nax.set_xscale('log')\n"
                                                "ax.set_yscale('log')",
         {"post": f"{LINES}, fc.scales()", "require": ["set_xscale|loglog|xscale"]}),
        ("symlog", f"Plot `grad` (it has both signs) on a symmetric-log y-axis that is linear within ±{lt}.",
         f"fig, ax = plt.subplots()\nax.plot(grad)\nax.set_yscale('symlog', linthresh={lt})",
         {"post": f"{LINES}, fc.scales(), fc.ax().yaxis.get_transform().linthresh", "require": ["linthresh"]}),
        ("logit", "Plot `acc` on a logit y-axis, so the last steps towards 1 stay visible.",
         "fig, ax = plt.subplots()\nax.plot(acc)\nax.set_yscale('logit')",
         {"post": f"{LINES}, fc.scales()", "require": ["set_yscale|yscale"]}),
        ("base", f"Plot `loss` against `steps` with a log **x**-axis in base {b2} (`ax.set_xscale('log', "
                 f"base={b2})`).", f"fig, ax = plt.subplots()\nax.plot(steps, loss)\nax.set_xscale('log', base={b2})",
         {"post": f"{LINES}, fc.scales(), fc.ax().xaxis.get_transform().base", "require": ["base"]}),
        ("semilogy", "Pyplot has shortcuts: draw `loss` against `steps` with `plt.semilogy` (log y-axis) and title "
                     "it 'loss'.", "plt.semilogy(steps, loss)\nplt.title('loss')",
         {"post": f"{LINES}, fc.scales(), fc.labels(fc.ax())['title']"}),
        ("loglog", "Draw `loss` against `steps` with `ax.loglog` (both axes logarithmic in one call).",
         "fig, ax = plt.subplots()\nax.loglog(steps, loss)", {"post": f"{LINES}, fc.scales()"}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "easy")
def date_axes(rng):
    code, desc, y0, m0, months = _days(rng)
    fmt = rng.choice(["%b %Y", "%Y-%m", "%b"])
    q = rng.choice([[1, 4, 7, 10], [1, 7]])
    imp = "import matplotlib.dates as mdates\n"
    plot = f"{imp}fig, ax = plt.subplots()\nax.plot(dates, value)\n"
    items = [
        ("MonthLocator", f"Drill: plot `value` against `dates` with a tick on the first of every month "
                         f"(`mdates.MonthLocator()`) labelled like {fmt!r} (`mdates.DateFormatter`).",
         f"{plot}ax.xaxis.set_major_locator(mdates.MonthLocator())\nax.xaxis.set_major_formatter(mdates.DateFormatter({fmt!r}))",
         {"post": f"{LINES}, {XT}", "require": ["MonthLocator", "DateFormatter"]}),
        ("autofmt_xdate", "Plot `value` against `dates` with monthly ticks labelled like 'Jan 2024', then let the "
                          "figure tilt the date labels so they don't overlap (`fig.autofmt_xdate()`).",
         f"{plot}ax.xaxis.set_major_locator(mdates.MonthLocator())\nax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))\n"
         f"fig.autofmt_xdate()", {"post": f"{LINES}, {XT}['labels'], {_rot('x')}, {_tick('x', 'ha')}",
                                  "require": ["autofmt_xdate"]}),
        ("ConciseDateFormatter", "Plot `value` against `dates` and let Matplotlib write compact date labels: keep the "
                                 "automatic locator and set `mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())` "
                                 "as the major formatter.",
         f"{plot}ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))",
         {"post": f"{LINES}, {XT}", "require": ["ConciseDateFormatter", "get_major_locator"]}),
        ("bymonth", f"Plot `value` against `dates` with ticks only on the first day of the months {q} "
                    f"(`mdates.MonthLocator(bymonth=...)`) labelled like 'Jan 2024'.",
         f"{plot}ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth={q}))\n"
         f"ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))",
         {"post": f"{LINES}, {XT}", "require": ["MonthLocator", "bymonth"]}),
        ("DayLocator", "Plot `value` against `dates` with ticks on the 1st and 15th of each month "
                       "(`mdates.DayLocator(bymonthday=[1, 15])`) labelled like '03-15' (`'%m-%d'`).",
         f"{plot}ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1, 15]))\n"
         f"ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))",
         {"post": f"{LINES}, {XT}", "require": ["DayLocator"]}),
        ("set_xlim", f"Plot `value` against `dates` and zoom the x-axis onto the second month only: from "
                     f"`dates[0] + np.timedelta64(31, 'D')` to `dates[0] + np.timedelta64(59, 'D')` (date limits "
                     f"work with `datetime64` values).",
         f"fig, ax = plt.subplots()\nax.plot(dates, value)\nax.set_xlim(dates[0] + np.timedelta64(31, 'D'), "
         f"dates[0] + np.timedelta64(59, 'D'))", {"post": f"{LINES}, fc.limits()['xlim']", "require": ["set_xlim|xlim"]}),
        ("WeekdayLocator", "Plot `value` against `dates` with major ticks at month starts (`MonthLocator`, labels "
                           "'%b') and minor ticks every Monday (`mdates.WeekdayLocator(byweekday=mdates.MO)`).",
         f"{plot}ax.xaxis.set_major_locator(mdates.MonthLocator())\nax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))\n"
         f"ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=mdates.MO))",
         {"post": f"{LINES}, {XT}, fc.ticks(fc.ax(), 'x', minor=True)['locs']", "require": ["WeekdayLocator"]}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "easy")
def translate_axis_calls(rng):
    code, desc, n = curves(rng)
    rot = rng.choice([30, 45, 60])
    lo, hi = rng.choice([(0.01, 3), (0.05, 2.5), (0.02, 5)])
    title = rng.choice(["Loss", "Run 3", "Curves"])
    items = [
        ("tick_params", f"Rewrite in the explicit style (`fig, ax = plt.subplots()`, Axes methods only):\n```python\n"
                        f"plt.plot(epochs, train_loss)\nplt.xticks(rotation={rot})\nplt.yscale('log')\n```",
         f"fig, ax = plt.subplots()\nax.plot(epochs, train_loss)\nax.tick_params(axis='x', labelrotation={rot})\n"
         f"ax.set_yscale('log')", {"post": f"{LINES}, {_rot('x')}, fc.scales()", "require": ["subplots", "set_yscale"]}),
        ("plt.ylim", f"Rewrite with pyplot functions only:\n```python\nfig, ax = plt.subplots()\nax.plot(epochs, "
                     f"val_loss)\nax.set_yscale('log')\nax.set_ylim({lo}, {hi})\nax.set_title({title!r}, loc='left')\n```",
         f"plt.plot(epochs, val_loss)\nplt.yscale('log')\nplt.ylim({lo}, {hi})\nplt.title({title!r}, loc='left')",
         {"post": f"{LINES}, fc.scales(), fc.limits()['ylim'], fc.titles()", "require": ["yscale", "ylim", "title"]}),
        ("plt.gca", "Pyplot has no function for tick formatters: get the Axes with `plt.gca()`. Rewrite so it uses "
                    "**only** pyplot plus `plt.gca()`:\n```python\nimport matplotlib.ticker as mticker\nfig, ax = "
                    "plt.subplots()\nax.plot(epochs, train_loss)\nax.yaxis.set_major_formatter(mticker."
                    "FormatStrFormatter('%.2f'))\n```",
         "import matplotlib.ticker as mticker\nplt.plot(epochs, train_loss)\n"
         "plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))",
         {"post": f"{LINES}, {_fmt('y', (0.5, 1.25))}", "require": ["gca"]}),
        ("plt.axis", "Rewrite with pyplot only:\n```python\nfig, ax = plt.subplots()\nax.plot(train_loss, val_loss)\n"
                     "ax.set_aspect('equal')\n```\n(`plt.axis('equal')` is close but changes the limits instead: use "
                     "`plt.gca().set_aspect('equal')`.)",
         "plt.plot(train_loss, val_loss)\nplt.gca().set_aspect('equal')", {"post": f"{LINES}, fc.ax().get_aspect()",
                                                                           "require": ["gca", "set_aspect"]}),
        ("plt.minorticks_on", "Rewrite with pyplot only:\n```python\nfig, ax = plt.subplots()\nax.plot(epochs, "
                              "train_loss)\nax.minorticks_on()\nax.grid(True, which='minor', alpha=0.2)\n```",
         "plt.plot(epochs, train_loss)\nplt.minorticks_on()\nplt.grid(True, which='minor', alpha=0.2)",
         {"post": f"{LINES}, (lambda g: [g.get_visible(), g.get_alpha()])(fc.ax().xaxis.get_minor_ticks()[0].gridline)",
          "require": ["minorticks_on", "grid"]}),
        ("plt.tick_params", "Rewrite with pyplot only:\n```python\nfig, ax = plt.subplots()\nax.plot(epochs, "
                            "train_loss)\nax.tick_params(direction='in', length=6, labelsize=8)\n```",
         "plt.plot(epochs, train_loss)\nplt.tick_params(direction='in', length=6, labelsize=8)",
         {"post": f"{LINES}, {_tick('x', 'dir', 'length', 'size')}, {_tick('y', 'dir', 'length', 'size')}",
          "require": ["tick_params"]}),
    ]
    return catalog(rng, items, code, f"{desc}.")


# ================================================================= medium

IMP = "import matplotlib.ticker as mticker\n"
DIMP = "import matplotlib.dates as mdates\n"


@template(M, "medium")
def accuracy_percent(rng):
    code, desc, n = curves(rng, acc=True)
    code += "\ntrain_acc = np.round(np.clip(val_acc + 0.03, 0, 1), 4)"
    desc += " and `train_acc`"
    lo = rng.choice([0.5, 0.6, 0.0])
    title = rng.choice(["Validation accuracy", "Accuracy", "val acc"])
    items = [
        ("PercentFormatter", f"Drill: plot `val_acc` against `epochs` with the y-axis written as percentages "
                             f"(`PercentFormatter(xmax=1)`), y limits {lo} … 1, x label 'epoch' and the title {title!r}.",
         f"{IMP}fig, ax = plt.subplots()\nax.plot(epochs, val_acc)\nax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))\n"
         f"ax.set_ylim({lo}, 1)\nax.set_xlabel('epoch')\nax.set_title({title!r})",
         {"post": f"{LINES}, {_fmt('y', (0.5, 0.95, 1))}, fc.limits()['ylim'], {LAB}"}),
        ("MultipleLocator", "Plot `val_acc` against `epochs` with a y tick every 10% (`MultipleLocator(0.1)`), labels "
                            "like '90%' (`PercentFormatter(xmax=1, decimals=0)`) and y limits 0.5 … 1.",
         f"{IMP}fig, ax = plt.subplots()\nax.plot(epochs, val_acc)\nax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))\n"
         f"ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))\nax.set_ylim(0.5, 1)",
         {"post": f"{LINES}, {YT}", "require": ["MultipleLocator", "PercentFormatter"]}),
        ("MaxNLocator", "Plot `train_acc` and `val_acc` against `epochs`, labelled 'train' and 'val', with a legend, "
                        "percent y labels (`PercentFormatter(xmax=1)`) and integer-only x ticks "
                        "(`MaxNLocator(integer=True)`).",
         f"{IMP}fig, ax = plt.subplots()\nax.plot(epochs, train_acc, label='train')\nax.plot(epochs, val_acc, label='val')\n"
         f"ax.legend()\nax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))\n"
         f"ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))",
         {"post": f"fc.lines(fc.ax(), 'label'), fc.legend(), {_fmt('y', (0.5, 1))}, {XT}['locs']",
          "require": ["PercentFormatter", "MaxNLocator"]}),
        ("set_yscale", "Errors shrink by orders of magnitude: plot the error rate `1 - val_acc` against `epochs` on a "
                       "log y-axis with percent labels (`PercentFormatter(xmax=1)`).",
         f"{IMP}fig, ax = plt.subplots()\nax.plot(epochs, 1 - val_acc)\nax.set_yscale('log')\n"
         f"ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))",
         {"post": f"{LINES}, fc.scales(), {_fmt('y', (0.01, 0.1))}", "require": ["set_yscale", "PercentFormatter"]}),
        ("set_title", "Plot `val_acc` against `epochs` and put the best result in a **left** title, like "
                      "'best 91.2% at epoch 17' (`f'best {acc:.1%} at epoch {e}'`, first epoch if tied).",
         "b = val_acc.argmax()\nfig, ax = plt.subplots()\nax.plot(epochs, val_acc)\n"
         "ax.set_title(f'best {val_acc[b]:.1%} at epoch {epochs[b]}', loc='left')",
         {"post": f"{LINES}, fc.titles()", "require": ["set_title", "loc"]}),
        ("tick_right", "Plot `val_acc` against `epochs` with the y ticks and their percent labels on the **right** "
                       "side (`ax.yaxis.tick_right()`), and the y label 'accuracy' on the right too "
                       "(`ax.yaxis.set_label_position('right')`).",
         f"{IMP}fig, ax = plt.subplots()\nax.plot(epochs, val_acc)\nax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))\n"
         f"ax.yaxis.tick_right()\nax.set_ylabel('accuracy')\nax.yaxis.set_label_position('right')",
         {"post": f"{LINES}, {_fmt('y', (0.5, 1))}, fc.ax().yaxis.get_ticks_position(), fc.ax().yaxis.get_label_position(), {LAB}",
          "require": ["tick_right", "set_label_position"]}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "medium")
def loss_log_minor(rng):
    code, desc, n = curves(rng)
    al = rng.choice([0.15, 0.2, 0.3])
    ls = rng.choice(["--", ":"])
    decay = rng.choice([0.9, 0.85, 0.95])
    items = [
        ("minorticks_on", f"Drill: plot `train_loss` and `val_loss` against `epochs` on a log y-axis, with dashed "
                          f"major grid lines (`linestyle={ls!r}`) and faint minor grid lines (`which='minor'`, "
                          f"`alpha={al}`).",
         f"fig, ax = plt.subplots()\nax.plot(epochs, train_loss)\nax.plot(epochs, val_loss)\nax.set_yscale('log')\n"
         f"ax.grid(True, which='major', linestyle={ls!r})\nax.grid(True, which='minor', alpha={al})",
         {"post": f"{LINES}, fc.scales(), fc.ax().yaxis.get_major_ticks()[0].gridline.get_linestyle(), "
                  f"(lambda g: [g.get_visible(), g.get_alpha()])(fc.ax().yaxis.get_minor_ticks()[0].gridline)",
          "require": ["set_yscale", "grid"]}),
        ("set_ylim", "Plot `train_loss` against `epochs` on a log y-axis whose limits leave a factor 2 of room: "
                     "bottom `train_loss.min() / 2`, top `train_loss.max() * 2`.",
         "fig, ax = plt.subplots()\nax.plot(epochs, train_loss)\nax.set_yscale('log')\n"
         "ax.set_ylim(train_loss.min() / 2, train_loss.max() * 2)",
         {"post": f"{LINES}, fc.scales(), fc.limits()['ylim']", "require": ["set_ylim"]}),
        ("EngFormatter", "Plot `val_loss` against the number of training tokens `epochs * 2_000_000` on log x and "
                         "log y axes, with the x labels in engineering notation (`EngFormatter()`, e.g. '10 M').",
         f"{IMP}fig, ax = plt.subplots()\nax.plot(epochs * 2_000_000, val_loss)\nax.set_xscale('log')\nax.set_yscale('log')\n"
         f"ax.xaxis.set_major_formatter(mticker.EngFormatter())",
         {"post": f"{LINES}, fc.scales(), {_fmt('x', (1e6, 1e7, 1e8))}", "require": ["EngFormatter"]}),
        ("NullFormatter", "Plot `train_loss` against `epochs` on a log y-axis with major ticks only at powers of 10 "
                          "(`LogLocator(base=10)`), minor ticks at 2 … 9 times them (`LogLocator(base=10, "
                          "subs=np.arange(2, 10))`) and **no** labels on the minor ticks (`NullFormatter()`).",
         f"{IMP}fig, ax = plt.subplots()\nax.plot(epochs, train_loss)\nax.set_yscale('log')\n"
         f"ax.yaxis.set_major_locator(mticker.LogLocator(base=10))\n"
         f"ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10)))\n"
         f"ax.yaxis.set_minor_formatter(mticker.NullFormatter())",
         {"post": f"{LINES}, {YT}['locs'], fc.ticks(fc.ax(), 'y', minor=True)",
          "require": ["LogLocator", "NullFormatter"]}),
        ("twinx", f"Plot `train_loss` against `epochs` on a log y-axis (left), and on a twin y-axis "
                  f"(`ax2 = ax.twinx()`) the learning rate `1e-3 * {decay} ** epochs`, also on a log scale, in 'C1'; "
                  f"label the y axes 'loss' and 'lr'.",
         f"fig, ax = plt.subplots()\nax.plot(epochs, train_loss)\nax.set_yscale('log')\nax.set_ylabel('loss')\n"
         f"ax2 = ax.twinx()\nax2.plot(epochs, 1e-3 * {decay} ** epochs, color='C1')\nax2.set_yscale('log')\nax2.set_ylabel('lr')",
         {"post": "[(fc.lines(a), a.get_yscale(), a.get_ylabel()) for a in fc.axes()], fc.lines(fc.ax(1), 'color', data=False)",
          "require": ["twinx", "set_yscale"]}),
        ("set_yticks", "On a log y-axis the default labels are powers of 10. Plot `val_loss` against `epochs` on a "
                       "log y-axis with ticks exactly at 0.1, 0.2, 0.5, 1 and 2 written as plain numbers "
                       "(`ax.set_yticks([...])` then `FormatStrFormatter('%g')`), and no minor ticks "
                       "(`ax.yaxis.set_minor_locator(mticker.NullLocator())`).",
         f"{IMP}fig, ax = plt.subplots()\nax.plot(epochs, val_loss)\nax.set_yscale('log')\nax.set_yticks([0.1, 0.2, 0.5, 1, 2])\n"
         f"ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%g'))\nax.yaxis.set_minor_locator(mticker.NullLocator())",
         {"post": f"{LINES}, {YT}, fc.ticks(fc.ax(), 'y', minor=True)['locs']", "require": ["set_yticks", "FormatStrFormatter"]}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "medium")
def daily_metric(rng):
    code, desc, y0, m0, months = _days(rng)
    title = rng.choice(["Daily active users", "Requests per day", "Eval score"])
    w = rng.choice([5, 7, 14])
    items = [
        ("DateFormatter", f"Drill: plot `value` against `dates` with a tick on the first of every month labelled like "
                          f"'Jan 2024', tilt the labels with `fig.autofmt_xdate()`, and title the plot {title!r}.",
         f"{DIMP}fig, ax = plt.subplots()\nax.plot(dates, value)\nax.xaxis.set_major_locator(mdates.MonthLocator())\n"
         f"ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))\nfig.autofmt_xdate()\nax.set_title({title!r})",
         {"post": f"{LINES}, {XT}, {_rot('x')}, {LAB}['title']", "require": ["MonthLocator", "DateFormatter", "autofmt_xdate"]}),
        ("AutoDateLocator", "Plot `value` against `dates` with an explicit `loc = mdates.AutoDateLocator()` as the "
                            "major locator and `mdates.ConciseDateFormatter(loc)` as the formatter; also return the "
                            "formatter's offset text (`ax.xaxis.get_major_formatter().get_offset()`, read after the "
                            "labels are made, e.g. after `ax.get_xticklabels()`) as `result`.",
         f"{DIMP}fig, ax = plt.subplots()\nax.plot(dates, value)\nloc = mdates.AutoDateLocator()\n"
         f"ax.xaxis.set_major_locator(loc)\nax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))\n"
         f"ax.get_xticklabels()\nresult = ax.xaxis.get_major_formatter().get_offset()",
         {"post": f"{LINES}, {XT}, _", "require": ["AutoDateLocator", "ConciseDateFormatter"]}),
        ("convolve", f"Plot the raw `value` (thin, alpha 0.4, label 'daily') and its {w}-day moving average "
                     f"(`np.convolve(value, np.ones({w}) / {w}, mode='valid')`, plotted against `dates[{w - 1}:]`, "
                     f"label '{w}-day mean'), with a legend and monthly ticks labelled by month name ('%b').",
         f"{DIMP}avg = np.convolve(value, np.ones({w}) / {w}, mode='valid')\nfig, ax = plt.subplots()\n"
         f"ax.plot(dates, value, alpha=0.4, label='daily')\nax.plot(dates[{w - 1}:], avg, label='{w}-day mean')\nax.legend()\n"
         f"ax.xaxis.set_major_locator(mdates.MonthLocator())\nax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))",
         {"post": f"fc.lines(fc.ax(), 'label', 'alpha'), fc.legend(), {XT}", "require": ["convolve", "MonthLocator"]}),
        ("DateFormatter", "Weekly report: plot every 7th day (`dates[::7]`, `value[::7]`) with circle markers "
                          "(`'o-'`), a tick at each of those dates (`ax.set_xticks(dates[::7])`) labelled like "
                          "'15 Mar' (`'%d %b'`), rotated 90°.",
         f"{DIMP}fig, ax = plt.subplots()\nax.plot(dates[::7], value[::7], 'o-')\nax.set_xticks(dates[::7])\n"
         f"ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))\nax.tick_params(axis='x', labelrotation=90)",
         {"post": f"{LINES}, {XT}, {_rot('x')}", "require": ["set_xticks", "DateFormatter"]}),
        ("set_xlim", "Zoom on the last 30 days: plot `value` against `dates`, limit the x-axis to `dates[-30]` … "
                     "`dates[-1]`, with a tick every 7 days (`mdates.DayLocator(interval=7)`) labelled '%m-%d'.",
         f"{DIMP}fig, ax = plt.subplots()\nax.plot(dates, value)\nax.set_xlim(dates[-30], dates[-1])\n"
         f"ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))\nax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))",
         {"post": f"{LINES}, fc.limits()['xlim'], {XT}", "require": ["set_xlim", "DayLocator"]}),
        ("sharex", "Two panels sharing the date axis (`plt.subplots(2, 1, sharex=True)`): `value` on top and its "
                   "day-to-day change `np.diff(value)` (against `dates[1:]`) below; on the bottom Axes set a "
                   "`ConciseDateFormatter` for its major locator.",
         f"{DIMP}fig, (a1, a2) = plt.subplots(2, 1, sharex=True)\na1.plot(dates, value)\na2.plot(dates[1:], np.diff(value))\n"
         f"a2.xaxis.set_major_formatter(mdates.ConciseDateFormatter(a2.xaxis.get_major_locator()))",
         {"post": "[fc.lines(a) for a in fc.axes()], fc.ticks(fc.ax(1), 'x'), "
                  "fc.ax(0).get_shared_x_axes().joined(fc.ax(0), fc.ax(1))", "require": ["sharex", "ConciseDateFormatter"]}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "medium")
def publication_axes(rng):
    code, desc, n = signal(rng)
    out = rng.choice([4, 5, 8])
    ln = rng.choice([3, 4, 6])
    title = rng.choice(["Response", "Signal A", "Fig. 2"])
    plot = "fig, ax = plt.subplots()\nax.plot(x, y)\n"
    sp = "fc.spines()"
    items = [
        ("set_position", f"'Clean' style: plot `y` against `x`, hide the top and right spines, move the left and "
                         f"bottom spines {out} points outward, and make the ticks point outward, {ln} points long.",
         f"{plot}ax.spines[['top', 'right']].set_visible(False)\nax.spines['left'].set_position(('outward', {out}))\n"
         f"ax.spines['bottom'].set_position(('outward', {out}))\nax.tick_params(direction='out', length={ln})",
         {"post": f"{LINES}, {sp}, [fc.ax().spines[k].get_position() for k in ('left', 'bottom')], "
                  f"{_tick('x', 'dir', 'length')}, {_tick('y', 'dir', 'length')}", "require": ["set_position", "tick_params"]}),
        ("tick_params", "'Boxed' style: plot `y` against `x` with ticks on **all four** sides pointing inward "
                        "(`ax.tick_params(direction='in', top=True, right=True)`).",
         f"{plot}ax.tick_params(direction='in', top=True, right=True)",
         {"post": f"{LINES}, {_tick('x', 'dir', 'top')}, {_tick('y', 'dir', 'top')}", "require": ["tick_params"]}),
        ("set_yticks", f"'Minimal' style: plot `y` against `x`, keep only the bottom spine, remove the y ticks "
                       f"(`ax.set_yticks([])`) and put the bold title {title!r} on the left.",
         f"{plot}ax.spines[['top', 'right', 'left']].set_visible(False)\nax.set_yticks([])\n"
         f"ax.set_title({title!r}, loc='left', fontweight='bold')",
         {"post": f"{LINES}, {sp}, {YT}['locs'], fc.titles(), fc.ax()._left_title.get_fontweight()",
          "require": ["set_visible", "set_yticks"]}),
        ("set_facecolor", "'Dark panel' style: plot `y` against `x` in 'orange' on an Axes with the background "
                          "'#222222', white tick labels and ticks (`tick_params(colors='white')`) and white spines.",
         f"fig, ax = plt.subplots()\nax.plot(x, y, color='orange')\nax.set_facecolor('#222222')\n"
         f"ax.tick_params(colors='white')\nfor s in ax.spines.values():\n    s.set_color('white')",
         {"post": f"fc.lines(fc.ax(), 'color'), fc.hexcolor(fc.ax().get_facecolor()), {_tick('x', 'color', 'tickcolor')}, "
                  f"{_tick('y', 'color')}, sorted({{fc.hexcolor(s.get_edgecolor()) for s in fc.ax().spines.values()}})",
          "require": ["set_facecolor", "tick_params"]}),
        ("set_axisbelow", "'Grid first' style: plot `y` against `x` with minor ticks on, dashed major grid lines and "
                          "dotted minor ones at opacity 0.3, drawn **below** the data (`ax.set_axisbelow(True)`).",
         f"{plot}ax.minorticks_on()\nax.grid(True, which='major', linestyle='--')\nax.grid(True, which='minor', linestyle=':', alpha=0.3)\n"
         f"ax.set_axisbelow(True)",
         {"post": f"{LINES}, fc.ax().xaxis.get_major_ticks()[0].gridline.get_linestyle(), "
                  f"(lambda g: [g.get_visible(), g.get_linestyle(), g.get_alpha()])(fc.ax().xaxis.get_minor_ticks()[0].gridline), "
                  f"fc.ax().get_axisbelow()", "require": ["set_axisbelow", "minorticks_on"]}),
        ("set_box_aspect", "'Wide strip' style: plot `y` against `x` in a box 4 times wider than tall "
                           "(`set_box_aspect(0.25)`), with no x margin (`ax.margins(x=0)`) and the background '#fafafa'.",
         f"{plot}ax.set_box_aspect(0.25)\nax.margins(x=0)\nax.set_facecolor('#fafafa')",
         {"post": f"{LINES}, fc.ax().get_box_aspect(), fc.limits()['xlim'], fc.hexcolor(fc.ax().get_facecolor())",
          "require": ["set_box_aspect", "margins"]}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "medium")
def scales_compared(rng):
    n = rng.choice([30, 40, 60])
    s = seeded(rng)
    code = (f"g = np.random.default_rng({s})\nx = np.linspace(0.1, 10, {n})\npos = np.round(np.exp(x) * (1 + 0.1 * g.random({n})), 4)\n"
            f"signed = np.round(np.sinh(x - 5) * 10, 4)\nprob = np.round(1 / (1 + np.exp(-(x - 5))), 6)\n"
            f"lrs = np.logspace(-5, -1, {n})\nlr_loss = np.round((np.log10(lrs) + 3) ** 2 + 0.3 + 0.05 * g.random({n}), 4)")
    desc = (f"over `x` ({n} points, 0.1 … 10): `pos` grows exponentially, `signed` crosses zero with large values on "
            f"both sides, `prob` is a sigmoid in (0, 1); `lr_loss` is the loss reached at each learning rate in `lrs`")
    lt1, lt2 = rng.choice([(0.1, 10), (1, 100), (0.5, 50)])
    items = [
        ("set_yscale", "Plot `pos` against `x` three times side by side (`plt.subplots(1, 3)`) with the y scales "
                       "'linear', 'log' and 'symlog', each Axes titled with its scale's name.",
         "fig, axs = plt.subplots(1, 3)\nfor a, sc in zip(axs, ['linear', 'log', 'symlog']):\n    a.plot(x, pos)\n"
         "    a.set_yscale(sc)\n    a.set_title(sc)", {"post": "[(fc.lines(a), a.get_yscale(), a.get_title()) for a in fc.axes()]"}),
        ("logit", "Plot `prob` against `x` on a 2 × 2 grid with the y scales 'linear', 'log', 'symlog', 'logit' (in "
                  "reading order), each titled with its scale.",
         "fig, axs = plt.subplots(2, 2)\nfor a, sc in zip(axs.flat, ['linear', 'log', 'symlog', 'logit']):\n"
         "    a.plot(x, prob)\n    a.set_yscale(sc)\n    a.set_title(sc)",
         {"post": "[(fc.lines(a), a.get_yscale(), a.get_title()) for a in fc.axes()]", "require": ["set_yscale"]}),
        ("loglog", "A power law: plot `x ** 2.5` against `x` on two Axes side by side: `semilogy` on the left "
                   "(titled 'semilogy') and `loglog` on the right (titled 'loglog'), where it is a straight line.",
         "fig, (a1, a2) = plt.subplots(1, 2)\na1.semilogy(x, x ** 2.5)\na1.set_title('semilogy')\na2.loglog(x, x ** 2.5)\n"
         "a2.set_title('loglog')", {"post": "[(fc.lines(a), fc.scales(a), a.get_title()) for a in fc.axes()]",
                                    "require": ["semilogy", "loglog"]}),
        ("sharey", "Shared axes share their scale too: make `fig, (a1, a2) = plt.subplots(1, 2, sharey=True)`, plot "
                   "`pos` on the left and `pos / 10` on the right, and call `set_yscale('log')` **once**, on `a1`. "
                   "Return both Axes' y scales as `[a1.get_yscale(), a2.get_yscale()]`.",
         "fig, (a1, a2) = plt.subplots(1, 2, sharey=True)\na1.plot(x, pos)\na2.plot(x, pos / 10)\na1.set_yscale('log')\n"
         "result = [a1.get_yscale(), a2.get_yscale()]", {"require": ["sharey", "set_yscale"]}),
        ("linthresh", f"Compare two symlog settings for `signed`: two Axes side by side with `linthresh={lt1}` (left) "
                      f"and `linthresh={lt2}` (right), titled `f'linthresh={{t}}'`.",
         f"fig, axs = plt.subplots(1, 2)\nfor a, t in zip(axs, [{lt1}, {lt2}]):\n    a.plot(x, signed)\n"
         f"    a.set_yscale('symlog', linthresh=t)\n    a.set_title(f'linthresh={{t}}')",
         {"post": "[(fc.lines(a), a.get_yscale(), a.yaxis.get_transform().linthresh, a.get_title()) for a in fc.axes()]",
          "require": ["linthresh"]}),
        ("set_xscale", "Learning-rate sweep: plot `lr_loss` against `lrs` on a log x-axis, label the axes "
                       "'learning rate' and 'loss', and mark the best learning rate with a red circle (`'ro'`).",
         "b = lr_loss.argmin()\nfig, ax = plt.subplots()\nax.plot(lrs, lr_loss)\nax.set_xscale('log')\n"
         "ax.plot(lrs[b], lr_loss[b], 'ro')\nax.set(xlabel='learning rate', ylabel='loss')",
         {"post": f"fc.lines(fc.ax(), 'color', 'marker'), fc.scales(), {LAB}", "require": ["set_xscale|semilogx"]}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "medium")
def tick_label_specs(rng):
    k = rng.randint(5, 8)
    s = seeded(rng)
    names = rng.choice([["resnet18", "resnet50", "vit-tiny", "vit-base", "convnext", "mobilenet", "efficientnet",
                         "regnet"], ["gpt-small", "gpt-medium", "llama-1b", "llama-3b", "mistral-7b", "phi-2",
                                     "qwen-1.8b", "gemma-2b"]])[:k]
    code = (f"names = {lit(names)}\nscores = np.round(np.sort(np.random.default_rng({s}).uniform(0.55, 0.95, {k}))[::-1], 3)")
    desc = f"`names` lists {k} models and `scores` their accuracies, best first"
    rot = rng.choice([30, 45, 60])
    fs = rng.choice([7, 8, 9])
    items = [
        ("invert_yaxis", "Leaderboard: draw `ax.barh(names, scores)`, flip the y-axis so the best model (first) is "
                         "at the **top**, and write the x labels as percentages (`PercentFormatter(xmax=1)`).",
         f"{IMP}fig, ax = plt.subplots()\nax.barh(names, scores)\nax.invert_yaxis()\n"
         f"ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))",
         {"post": f"fc.bars(fc.ax(), 4), fc.ax().yaxis_inverted(), {YT}['labels'], {_fmt('x', (0.5, 0.9))}",
          "require": ["barh", "invert_yaxis"]}),
        ("setp", f"Draw `ax.bar(names, scores)` and rotate the long x labels by {rot}° with their right end at the "
                 f"tick (`plt.setp(ax.get_xticklabels(), rotation={rot}, ha='right')`).",
         f"fig, ax = plt.subplots()\nax.bar(names, scores)\nplt.setp(ax.get_xticklabels(), rotation={rot}, ha='right')",
         {"post": f"fc.bars(fc.ax(), 4), {_rot('x')}, sorted({{t.get_ha() for t in fc.ax().get_xticklabels()}})",
          "require": ["setp|tick_params|set_xticks|set_xticklabels"]}),
        ("set_visible", "Draw `ax.bar(names, scores)` and hide every **other** x label (the 2nd, 4th, …: "
                        "`for t in ax.get_xticklabels()[1::2]: t.set_visible(False)`).",
         "fig, ax = plt.subplots()\nax.bar(names, scores)\nfor t in ax.get_xticklabels()[1::2]:\n    t.set_visible(False)",
         {"post": "fc.bars(fc.ax(), 4), [t.get_visible() for t in fc.ax().get_xticklabels()]", "require": ["set_visible"]}),
        ("labeltop", "Draw `ax.bar(names, scores)` with the x ticks and labels on **top** instead of the bottom "
                     "(`ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)`).",
         "fig, ax = plt.subplots()\nax.bar(names, scores)\nax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)",
         {"post": "fc.bars(fc.ax(), 4), (lambda t: [t.tick1line.get_visible(), t.label1.get_visible(), t.tick2line.get_visible(), "
                  "t.label2.get_visible()])(fc.ax().xaxis.get_major_ticks()[0])", "require": ["tick_params"]}),
        ("fontweight", f"Draw `ax.bar(range({k}), scores)` and label the bars with `ax.set_xticks(range({k}), "
                       f"labels=names, fontsize={fs}, rotation=90)`, with the best model's label in **bold** "
                       f"(`ax.get_xticklabels()[0].set_fontweight('bold')`).",
         f"fig, ax = plt.subplots()\nax.bar(range({k}), scores)\nax.set_xticks(range({k}), labels=names, fontsize={fs}, rotation=90)\n"
         f"ax.get_xticklabels()[0].set_fontweight('bold')",
         {"post": f"fc.bars(fc.ax(), 4), {XT}, [(t.get_fontsize(), t.get_fontweight()) for t in fc.ax().get_xticklabels()], {_rot('x')}",
          "require": ["set_xticks", "set_fontweight"]}),
        ("set_yticks", f"Rank plot: plot the ranks 1 … {k} (y) against `scores` (x) as circles, label the y ticks "
                       f"1 … {k} with the model names (`ax.set_yticks(range(1, {k + 1}), labels=names)`), and set the y "
                       f"limits to ({k} + 0.5, 0.5) so rank 1 is at the top.",
         f"fig, ax = plt.subplots()\nax.plot(scores, np.arange(1, {k + 1}), 'o')\nax.set_yticks(range(1, {k + 1}), labels=names)\n"
         f"ax.set_ylim({k} + 0.5, 0.5)", {"post": f"{LINES}, {YT}, fc.limits()['ylim']", "require": ["set_yticks", "set_ylim"]}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "medium")
def big_numbers(rng):
    s = seeded(rng)
    n = rng.choice([8, 10, 12])
    code = (f"g = np.random.default_rng({s})\nsizes = np.round(np.logspace(6, 10, {n}))\n"
            f"loss = np.round(2.5 * (sizes / 1e6) ** -0.08 + 0.01 * g.random({n}), 4)\n"
            f"tokens = np.round(np.linspace(1e9, 3e10, {n}))\nstep_time = np.round(1_000_000 + g.normal(0, 3, {n}), 2)")
    desc = (f"`sizes` holds {n} model sizes (parameters, 10^6 … 10^10), `loss` their loss, `tokens` training-token "
            f"counts and `step_time` step times in microseconds (around 1,000,000, varying by a few units)")
    items = [
        ("ticklabel_format", "Plot `loss` against `tokens` with the x labels in scientific notation, the common "
                             "factor shown once at the end of the axis (`ax.ticklabel_format(axis='x', style='sci', "
                             "scilimits=(0, 0))`). Also return that offset text "
                             "(`ax.xaxis.get_major_formatter().get_offset()`, read after `ax.get_xticklabels()`).",
         "fig, ax = plt.subplots()\nax.plot(tokens, loss)\nax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))\n"
         "ax.get_xticklabels()\nresult = ax.xaxis.get_major_formatter().get_offset()",
         {"post": f"{LINES}, {XT}['labels'], _", "require": ["ticklabel_format"]}),
        ("useOffset", "`step_time` varies by a few units around 1,000,000, so Matplotlib writes the y labels as small "
                      "numbers plus an offset like '+1e6'. Plot it and turn the offset off "
                      "(`ax.ticklabel_format(axis='y', useOffset=False)`) so the labels show the full values.",
         "fig, ax = plt.subplots()\nax.plot(step_time)\nax.ticklabel_format(axis='y', useOffset=False)",
         {"post": f"{LINES}, {YT}['labels'], fc.ax().yaxis.get_major_formatter().get_offset()", "require": ["useOffset"]}),
        ("EngFormatter", "Plot `loss` against `tokens` with x labels in engineering notation (`EngFormatter()`: "
                         "'10 G' for 1e10).",
         f"{IMP}fig, ax = plt.subplots()\nax.plot(tokens, loss)\nax.xaxis.set_major_formatter(mticker.EngFormatter())",
         {"post": f"{LINES}, {_fmt('x', (1e9, 1.5e10, 3e10))}"}),
        ("FuncFormatter", "Plot `loss` against `sizes` on a log x-axis with human-readable labels: 1e6 → '1M', 1e9 "
                          "→ '1B' (a FuncFormatter: `f'{v / 1e9:g}B'` from 1e9 on, `f'{v / 1e6:g}M'` from 1e6, else "
                          "`f'{v:g}'`).",
         f"{IMP}def human(v, pos):\n    if v >= 1e9:\n        return f'{{v / 1e9:g}}B'\n    if v >= 1e6:\n        return f'{{v / 1e6:g}}M'\n"
         f"    return f'{{v:g}}'\n\nfig, ax = plt.subplots()\nax.plot(sizes, loss)\nax.set_xscale('log')\n"
         f"ax.xaxis.set_major_formatter(mticker.FuncFormatter(human))",
         {"post": f"{LINES}, fc.scales(), {_fmt('x', (1e5, 1e6, 3e7, 1e9, 2.5e10))}", "require": ["FuncFormatter"]}),
        ("StrMethodFormatter", "Plot `loss` against `sizes` on a log x-axis with every label in compact exponent "
                               "form like '1e+08' (`StrMethodFormatter('{x:.0e}')`).",
         f"{IMP}fig, ax = plt.subplots()\nax.plot(sizes, loss)\nax.set_xscale('log')\n"
         f"ax.xaxis.set_major_formatter(mticker.StrMethodFormatter('{{x:.0e}}'))",
         {"post": f"{LINES}, {_fmt('x', (1e6, 1e8, 1e10))}"}),
        ("EngFormatter", "Plot `loss` against `sizes` on a log x-axis with x labels like '100 M params' "
                         "(`EngFormatter(unit='params')`) and the x label 'model size'.",
         f"{IMP}fig, ax = plt.subplots()\nax.plot(sizes, loss)\nax.set_xscale('log')\n"
         f"ax.xaxis.set_major_formatter(mticker.EngFormatter(unit='params'))\nax.set_xlabel('model size')",
         {"post": f"{LINES}, {_fmt('x', (1e6, 1e8, 1e10))}, {LAB}"}),
    ]
    return catalog(rng, items, code, f"{desc}.")


# ================================================================= hard

@template(M, "hard")
def axis_helpers(rng):
    code, desc, n = curves(rng, acc=True)
    ln = rng.choice([2, 3, 4])
    out = rng.choice([4, 6, 8])
    step = rng.choice([0.1, 0.25, 0.5])
    every = rng.choice([2, 3])
    al = rng.choice([0.15, 0.2, 0.3])
    dec = rng.choice([0, 1, 2])
    items = [
        ("PercentFormatter", "Write `percent_axis(ax, axis='y', decimals=None)`: put a `PercentFormatter(xmax=1, "
                             "decimals=decimals)` on the Axes' x or y axis (as `axis` says) and return the formatter. "
                             "(Checked by calling it on fresh Axes.)",
         f"{IMP}def percent_axis(ax, axis='y', decimals=None):\n    fmt = mticker.PercentFormatter(xmax=1, decimals=decimals)\n"
         f"    (ax.xaxis if axis == 'x' else ax.yaxis).set_major_formatter(fmt)\n    return fmt",
         {"post": f"_f, (_a, _b) = plt.subplots(1, 2)\n_a.plot(epochs, val_acc)\n_r = percent_axis(_a)\n_b.plot(val_acc, epochs)\n"
                  f"percent_axis(_b, 'x', decimals={dec})\n(fc.formatted(_a, 'y', [0.5, 0.875, 1]), fc.formatted(_b, 'x', [0.5, 0.875]), "
                  f"_r is _a.yaxis.get_major_formatter(), type(_b.yaxis.get_major_formatter()).__name__)",
          "require": ["PercentFormatter", "set_major_formatter"]}),
        ("set_yscale", "Write `log_axis(ax, axis='y', minor_alpha=0.2)`: make that axis logarithmic, turn on its major "
                       "grid lines and its minor grid lines (at opacity `minor_alpha`) for **that axis only**, and return "
                       "how many decades the axis then spans: `log10(top / bottom)` of its limits.",
         "def log_axis(ax, axis='y', minor_alpha=0.2):\n    if axis == 'y':\n        ax.set_yscale('log')\n    else:\n"
         "        ax.set_xscale('log')\n    ax.grid(True, which='major', axis=axis)\n    ax.grid(True, which='minor', axis=axis, alpha=minor_alpha)\n"
         "    lo, hi = ax.get_ylim() if axis == 'y' else ax.get_xlim()\n    return float(np.log10(hi / lo))",
         {"post": f"_f, _a = plt.subplots()\n_a.plot(epochs, train_loss)\n_r = log_axis(_a, minor_alpha={al})\n"
                  "(fc.scales(_a), (lambda g: [g.get_visible(), g.get_alpha()])(_a.yaxis.get_minor_ticks()[0].gridline), "
                  "fc.grid_on(_a), round(_r, 6))", "require": ["set_yscale|set_xscale|semilogy", "grid"]}),
        ("set_position", "Write `clean_axes(ax, outward=5, length=3)`: hide the top and right spines, move the left and "
                         "bottom spines `outward` points out, shorten the ticks to `length` points (`tick_params`), and "
                         "return the sorted names of the hidden spines.",
         "def clean_axes(ax, outward=5, length=3):\n    ax.spines[['top', 'right']].set_visible(False)\n"
         "    for k in ('left', 'bottom'):\n        ax.spines[k].set_position(('outward', outward))\n"
         "    ax.tick_params(length=length)\n    return sorted(k for k, s in ax.spines.items() if not s.get_visible())",
         {"post": f"_f, _a = plt.subplots()\n_a.plot(epochs, train_loss)\n_r = clean_axes(_a, {out}, {ln})\n"
                  "(_r, fc.spines(_a), [_a.spines[k].get_position() for k in ('left', 'bottom')], "
                  "[fc.r(a.get_major_ticks()[0].tick1line.get_markersize(), 2) for a in (_a.xaxis, _a.yaxis)])",
          "require": ["spines", "set_position", "tick_params"]}),
        ("MonthLocator", "Write `month_axis(ax, fmt='%b %Y', interval=1)`: tick the x-axis on the first of every "
                         "`interval`-th month (`mdates.MonthLocator(interval=interval)`), label the ticks with "
                         "`mdates.DateFormatter(fmt)`, and return the tick label texts.",
         f"{DIMP}def month_axis(ax, fmt='%b %Y', interval=1):\n    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))\n"
         f"    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))\n    return [t.get_text() for t in ax.get_xticklabels()]",
         {"post": "_d = np.arange('2024-01-01', '2024-07-01', dtype='datetime64[D]')\n_f, (_a, _b) = plt.subplots(2)\n"
                  "_a.plot(_d, np.arange(len(_d)))\n_b.plot(_d, np.arange(len(_d)))\n"
                  "(month_axis(_a), month_axis(_b, '%b', 2), fc.ticks(_a, 'x'), fc.ticks(_b, 'x'))",
          "require": ["MonthLocator", "DateFormatter"]}),
        ("FuncFormatter", "Write `human_format(v, pos=None)`, a tick-label function for big counts: |v| ≥ 1e9 → "
                          "`f'{v / 1e9:g}B'`, ≥ 1e6 → `f'{v / 1e6:g}M'`, ≥ 1e3 → `f'{v / 1e3:g}k'`, else `f'{v:g}'`. "
                          "(`ax.yaxis.set_major_formatter(human_format)` wraps a plain function in a FuncFormatter.)",
         "def human_format(v, pos=None):\n    for div, suffix in ((1e9, 'B'), (1e6, 'M'), (1e3, 'k')):\n"
         "        if abs(v) >= div:\n            return f'{v / div:g}{suffix}'\n    return f'{v:g}'",
         {"post": "_f, _a = plt.subplots()\n_a.plot(epochs, epochs * 250_000)\n_a.yaxis.set_major_formatter(human_format)\n"
                  "([human_format(v) for v in (0, 950, 1500, -2e6, 2.5e7, 3.2e9)], fc.formatted(_a, 'y', [0, 2.5e6, 1e7]), "
                  "type(_a.yaxis.get_major_formatter()).__name__)", "require": []}),
        ("set_yticks", "Write `rank_axis(ax, names)` for rank plots: y ticks at 1 … n labelled with `names` (rank 1 "
                       "first) and y limits (n + 0.5, 0.5), so rank 1 is at the **top**; return `ax`.",
         "def rank_axis(ax, names):\n    n = len(names)\n    ax.set_yticks(range(1, n + 1), labels=names)\n"
         "    ax.set_ylim(n + 0.5, 0.5)\n    return ax",
         {"post": "_f, _a = plt.subplots()\n_a.plot([0.9, 0.8, 0.7, 0.6], [1, 2, 3, 4], 'o')\n_r = rank_axis(_a, ['a', 'b', 'c', 'd'])\n"
                  "(_r is _a, fc.ticks(_a, 'y'), fc.limits(_a)['ylim'], _a.yaxis_inverted())",
          "require": ["set_yticks|set_yticklabels", "set_ylim|invert_yaxis"]}),
        ("set_ylim", "Write `nice_limits(ax, step, axis='y')`: round the axis' current limits **outward** to multiples "
                     "of `step` (floor the bottom, ceil the top), set them, and return them as a list.",
         "def nice_limits(ax, step, axis='y'):\n    lo, hi = ax.get_ylim() if axis == 'y' else ax.get_xlim()\n"
         "    new = [np.floor(lo / step) * step, np.ceil(hi / step) * step]\n    if axis == 'y':\n        ax.set_ylim(*new)\n"
         "    else:\n        ax.set_xlim(*new)\n    return new",
         {"post": f"_f, _a = plt.subplots()\n_a.plot(epochs, val_loss)\n_r = nice_limits(_a, {step})\n_s = nice_limits(_a, 5, axis='x')\n"
                  "(fc.r(_r, 9), fc.r(_s, 9), fc.limits(_a))", "require": ["floor", "ceil"]}),
        ("set_visible", "Write `thin_labels(ax, every=2, axis='x')`: keep only every `every`-th tick label of that axis "
                        "visible (the 1st, the (every + 1)-th, …), hide the others, and return the visible label texts.",
         "def thin_labels(ax, every=2, axis='x'):\n    labels = ax.get_xticklabels() if axis == 'x' else ax.get_yticklabels()\n"
         "    for i, t in enumerate(labels):\n        t.set_visible(i % every == 0)\n"
         "    return [t.get_text() for t in labels if t.get_visible()]",
         {"post": f"_f, _a = plt.subplots()\n_a.bar([f'm{{i}}' for i in range(7)], np.arange(7))\n_r = thin_labels(_a, {every})\n"
                  "(_r, [t.get_visible() for t in _a.get_xticklabels()])", "require": ["set_visible"]}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "hard")
def debug_axes_ticks(rng):
    code, desc, n = curves(rng, acc=True)
    dcode, ddesc, *_ = _days(rng)
    mid = n // 2
    k = rng.randint(4, 6)
    s = seeded(rng)
    names = rng.choice([["LR", "SVM", "RF", "GBM", "MLP", "kNN"], ["tiny", "small", "base", "large", "xl", "xxl"]])[:k]
    bars = f"names = {lit(names)}\nscores = np.round(np.random.default_rng({s}).uniform(0.6, 0.95, {k}), 3)"
    plot = "fig, ax = plt.subplots()\nax.plot(epochs, train_loss)\n"
    items = [
        ("set_xticks", f"This crashes with *The number of FixedLocator locations (3), usually from a call to set_ticks, "
                       f"does not match the number of labels (2)*:\n```python\n{plot}ax.set_xticks([1, {mid}, {n}], "
                       f"labels=['start', 'end'])\n```\nLabel the middle tick 'mid'.",
         f"{plot}ax.set_xticks([1, {mid}, {n}], labels=['start', 'mid', 'end'])", {"post": f"{LINES}, {XT}"}),
        ("set_xticks", f"`names` lists {k} models and `scores` their scores. The names land under the wrong bars, and "
                       f"Matplotlib warns *set_ticklabels() should only be used with a fixed number of ticks*: "
                       f"`set_xticklabels` labels whatever ticks the automatic locator picked.\n```python\n"
                       f"fig, ax = plt.subplots()\nax.bar(range(len(names)), scores)\nax.set_xticklabels(names)\n```\n"
                       f"Put each name under its bar.",
         "fig, ax = plt.subplots()\nax.bar(range(len(names)), scores)\nax.set_xticks(range(len(names)), labels=names)",
         {"post": f"fc.bars(fc.ax(), 4), {XT}", "setup": bars, "require": ["bar"]}),
        ("PercentFormatter", f"The accuracy labels come out as tiny percentages like '0.9%': `PercentFormatter()` "
                             f"assumes the data are already in percent (`xmax=100`).\n```python\n{IMP}fig, ax = "
                             f"plt.subplots()\nax.plot(epochs, val_acc)\nax.yaxis.set_major_formatter(mticker."
                             f"PercentFormatter())\n```\nFix it for accuracies stored as fractions.",
         f"{IMP}fig, ax = plt.subplots()\nax.plot(epochs, val_acc)\nax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))",
         {"post": f"{LINES}, {_fmt('y', (0.5, 0.9, 1))}", "require": ["PercentFormatter"]}),
        ("linthresh", "The epoch-to-epoch change of the loss has both signs, and on a log axis everything ≤ 0 "
                      "vanishes:\n```python\nfig, ax = plt.subplots()\nax.plot(epochs[1:], np.diff(val_loss))\n"
                      "ax.set_yscale('log')\n```\nKeep a log-like axis for the large changes but show the negative ones "
                      "too: a symmetric log scale, linear within ±0.01.",
         "fig, ax = plt.subplots()\nax.plot(epochs[1:], np.diff(val_loss))\nax.set_yscale('symlog', linthresh=0.01)",
         {"post": f"{LINES}, fc.scales(), fc.ax().yaxis.get_transform().linthresh", "require": ["set_yscale|yscale", "linthresh"]}),
        ("set_ylim", f"Matplotlib warns *Attempt to set non-positive ylim on a log-scaled axis will be ignored*, and the "
                     f"bottom limit stays automatic:\n```python\n{plot}ax.set_yscale('log')\nax.set_ylim(0, 3)\n```\nOn "
                     f"a log axis the bottom must be positive: start it at 0.01.",
         f"{plot}ax.set_yscale('log')\nax.set_ylim(0.01, 3)", {"post": f"{LINES}, fc.scales(), fc.limits()['ylim']"}),
        ("tick_params", "Only the right panel's labels get rotated: `plt.xticks` acts on the **current** Axes, the last "
                        "one created.\n```python\nfig, (a1, a2) = plt.subplots(1, 2)\na1.plot(epochs, train_loss)\n"
                        "a2.plot(epochs, val_loss)\nplt.xticks(rotation=45)\n```\nRotate both panels' x labels by 45°.",
         "fig, (a1, a2) = plt.subplots(1, 2)\na1.plot(epochs, train_loss)\na2.plot(epochs, val_loss)\nfor a in (a1, a2):\n"
         "    a.tick_params(axis='x', labelrotation=45)",
         {"post": "[(fc.lines(a), sorted({fc.r(t.get_rotation(), 2) for t in a.get_xticklabels()})) for a in fc.axes()]",
          "require": []}),
        ("minorticks_on", f"The faint minor grid never shows up: a linear axis has **no** minor ticks until you ask for "
                          f"them.\n```python\n{plot}ax.grid(True, which='minor', alpha=0.2)\n```\nFix it (keep the grid "
                          f"call).",
         f"{plot}ax.minorticks_on()\nax.grid(True, which='minor', alpha=0.2)",
         {"post": f"{LINES}, fc.ticks(fc.ax(), 'x', minor=True)['locs'], "
                  f"(lambda g: [g.get_visible(), g.get_alpha()])(fc.ax().xaxis.get_minor_ticks()[0].gridline)",
          "require": ["minorticks_on|set_minor_locator", "grid"]}),
        ("set_title", f"This crashes with *'top' is not a valid value for loc*:\n```python\n{plot}ax.set_title('Loss', "
                      f"loc='top')\n```\nThe title always sits above the Axes; `loc` only aligns it ('left', 'center', "
                      f"'right'). The author wanted it on the left.",
         f"{plot}ax.set_title('Loss', loc='left')", {"post": f"{LINES}, fc.titles()"}),
        ("FuncFormatter", f"This crashes as soon as the labels are made, with *<lambda>() takes 1 positional argument but "
                          f"2 were given*: a FuncFormatter calls its function with the value **and** the tick position."
                          f"\n```python\n{IMP}fig, ax = plt.subplots()\nax.plot(epochs, val_acc)\n"
                          f"ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v: f'{{v:.0%}}'))\n```\nFix it.",
         f"{IMP}fig, ax = plt.subplots()\nax.plot(epochs, val_acc)\n"
         f"ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, pos: f'{{v:.0%}}'))",
         {"post": f"{LINES}, {_fmt('y', (0.5, 0.9))}", "require": ["FuncFormatter"]}),
        ("DateFormatter", f"{ddesc[0].upper()}{ddesc[1:]}. The labels read like '01 2024' but should read like 'Jan "
                          f"2024': `%m` is the month **number**, `%b` its short name.\n```python\n{DIMP}fig, ax = "
                          f"plt.subplots()\nax.plot(dates, value)\nax.xaxis.set_major_locator(mdates.MonthLocator())\n"
                          f"ax.xaxis.set_major_formatter(mdates.DateFormatter('%m %Y'))\n```\nFix it.",
         f"{DIMP}fig, ax = plt.subplots()\nax.plot(dates, value)\nax.xaxis.set_major_locator(mdates.MonthLocator())\n"
         f"ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))",
         {"post": f"{LINES}, {XT}", "setup": dcode, "require": ["DateFormatter"]}),
        ("MultipleLocator", f"Matplotlib warns *Locator attempting to generate … ticks, which exceeds Locator.MAXTICKS "
                            f"(1000)* and the x labels turn into a black smear: the x values are optimizer **steps**, "
                            f"500 per epoch.\n```python\n{IMP}fig, ax = plt.subplots()\nax.plot(epochs * 500, train_loss)\n"
                            f"ax.xaxis.set_major_locator(mticker.MultipleLocator(1))\n```\nPut a tick every 2500 steps "
                            f"instead.",
         f"{IMP}fig, ax = plt.subplots()\nax.plot(epochs * 500, train_loss)\nax.xaxis.set_major_locator(mticker.MultipleLocator(2500))",
         {"post": f"{LINES}, {XT}['locs']"}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "hard")
def axes_drills(rng):
    code, desc, n = curves(rng, acc=True)
    dcode, ddesc, *_ = _days(rng)
    code += "\ntrain_acc = np.round(np.clip(val_acc + 0.03, 0, 1), 4)\n" + dcode
    title = rng.choice(["Title", "log(x)", "Natural log"])
    rot = rng.choice([30, 45, 60])
    w, h = rng.choice([(7, 4), (8, 4), (6, 3.5)])
    dpi = rng.choice([30, 40, 50])
    items = [
        ("sharey", "Drill 1, full version: `fig, (a1, a2) = plt.subplots(1, 2, sharey=True, figsize=(8, 3))`. Left: "
                   "`train_acc` and `val_acc` against `epochs` (labels 'train', 'val', legend), titled 'all epochs'. "
                   "Right: the same two curves zoomed on the last 10 epochs (`a2.set_xlim(epochs[-10], epochs[-1])`), "
                   "titled 'last 10'. Percent y labels with one decimal (`PercentFormatter(xmax=1, decimals=1)`), set "
                   "**once**, on `a1.yaxis`: shared axes share their tick formatter.",
         f"{IMP}fig, (a1, a2) = plt.subplots(1, 2, sharey=True, figsize=(8, 3))\nfor a in (a1, a2):\n"
         f"    a.plot(epochs, train_acc, label='train')\n    a.plot(epochs, val_acc, label='val')\na1.legend()\n"
         f"a1.set_title('all epochs')\na2.set_title('last 10')\na2.set_xlim(epochs[-10], epochs[-1])\n"
         f"a1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))",
         {"post": "[(fc.lines(a, 'label'), a.get_title(), fc.formatted(a, 'y', [0.5, 0.925])) for a in fc.axes()], "
                  "fc.limits(fc.ax(1))['xlim'], fc.legend(fc.ax(0)), fc.size()", "require": ["sharey", "PercentFormatter"]}),
        ("LogLocator", "Drill 2, full version: `train_loss` and `val_loss` against `epochs` (labels 'train', 'val', "
                       "legend) on a log y-axis; minor y ticks at 2 … 9 times each power of ten (`LogLocator(base=10, "
                       "subs=np.arange(2, 10))`) with **no** labels (`NullFormatter()`); major grid lines at opacity 0.5 "
                       "and minor ones at opacity 0.15; integer-only x ticks (`MaxNLocator(integer=True)`); axis labels "
                       "'epoch' and 'loss (log)'.",
         f"{IMP}fig, ax = plt.subplots()\nax.plot(epochs, train_loss, label='train')\nax.plot(epochs, val_loss, label='val')\n"
         f"ax.legend()\nax.set_yscale('log')\nax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10)))\n"
         f"ax.yaxis.set_minor_formatter(mticker.NullFormatter())\nax.grid(True, which='major', alpha=0.5)\n"
         f"ax.grid(True, which='minor', alpha=0.15)\nax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))\n"
         f"ax.set(xlabel='epoch', ylabel='loss (log)')",
         {"post": f"fc.lines(fc.ax(), 'label'), fc.legend(), fc.scales(), fc.ticks(fc.ax(), 'y', minor=True), "
                  f"[t()[0].gridline.get_alpha() for t in (fc.ax().yaxis.get_major_ticks, fc.ax().yaxis.get_minor_ticks)], "
                  f"{XT}['locs'], {LAB}",
          "require": ["LogLocator", "NullFormatter", "MaxNLocator"]}),
        ("WeekdayLocator", f"Drill 3, full version: `value` against `dates` on a {w} × {h} inch figure (`figsize`), major "
                           f"x ticks on the first of every month labelled like 'Jan 2024', minor ticks on every Monday "
                           f"(`mdates.WeekdayLocator(byweekday=mdates.MO)`), the x-axis limited to the data "
                           f"(`ax.set_xlim(dates[0], dates[-1])`), the y label 'value', dashed grid lines on the major "
                           f"**x** ticks only (`ax.grid(True, axis='x', linestyle='--')`), and the labels tilted by "
                           f"`fig.autofmt_xdate()`.",
         f"{DIMP}fig, ax = plt.subplots(figsize=({w}, {h}))\nax.plot(dates, value)\nax.xaxis.set_major_locator(mdates.MonthLocator())\n"
         f"ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))\nax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=mdates.MO))\n"
         f"ax.set_xlim(dates[0], dates[-1])\nax.set_ylabel('value')\nax.grid(True, axis='x', linestyle='--')\nfig.autofmt_xdate()",
         {"post": f"{LINES}, {XT}, fc.ticks(fc.ax(), 'x', minor=True)['locs'], fc.limits()['xlim'], fc.grid_on(), "
                  f"{_rot('x')}, fc.size(), {LAB}['ylabel']",
          "require": ["MonthLocator", "WeekdayLocator", "autofmt_xdate", "figsize"]}),
        ("tick_params", f"The guide's main panel. On `fig, ax = plt.subplots(figsize=({w}, {h}))` plot `np.log(xs)` against "
                        f"`xs` (given: 400 points from 0.01 to 100). Title {title!r} in bold at size 14 on the left, 10 "
                        f"points above the Axes (`pad`); x label 'x (units)' at size 11; y label 'log(x)' with "
                        f"`labelpad=8`; x limits 0 … 100 and the bottom y limit at -5; major x ticks at 0, 50, 100 "
                        f"labelled 'start', 'mid', 'end' and minor x ticks every 10; x tick labels rotated {rot}° at size "
                        f"8 with the x ticks pointing in, 6 points long; grey y ticks and labels (`colors`); at most 5 "
                        f"integer y intervals (`MaxNLocator(nbins=5, integer=True)`); a dashed major grid at opacity 0.5; "
                        f"no top / right spines; the left spine 5 points outward.",
         f"{IMP}fig, ax = plt.subplots(figsize=({w}, {h}))\nax.plot(xs, np.log(xs))\n"
         f"ax.set_title({title!r}, fontsize=14, fontweight='bold', loc='left', pad=10)\nax.set_xlabel('x (units)', fontsize=11)\n"
         f"ax.set_ylabel('log(x)', labelpad=8)\nax.set_xlim(0, 100)\nax.set_ylim(bottom=-5)\n"
         f"ax.set_xticks([0, 50, 100], labels=['start', 'mid', 'end'])\nax.set_xticks(np.arange(0, 101, 10), minor=True)\n"
         f"ax.tick_params(axis='x', labelrotation={rot}, labelsize=8, direction='in', length=6)\nax.tick_params(axis='y', colors='gray')\n"
         f"ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, integer=True))\nax.grid(True, which='major', linestyle='--', alpha=0.5)\n"
         f"ax.spines[['top', 'right']].set_visible(False)\nax.spines['left'].set_position(('outward', 5))",
         {"post": f"{LINES}, fc.titles(), [fc.ax()._left_title.get_fontsize(), fc.ax()._left_title.get_fontweight()], {PAD}, "
                  f"{LAB}, fc.ax().xaxis.label.get_fontsize(), fc.ax().yaxis.labelpad, fc.limits(), {XT}, "
                  f"fc.ticks(fc.ax(), 'x', minor=True)['locs'], {_tick('x', 'rotation', 'size', 'dir', 'length')}, "
                  f"{_tick('y', 'color', 'tickcolor')}, {YT}['locs'], "
                  f"(lambda g: [g.get_visible(), g.get_linestyle(), g.get_alpha()])(fc.ax().yaxis.get_major_ticks()[0].gridline), "
                  f"fc.spines(), fc.ax().spines['left'].get_position(), fc.size()",
          "setup": "xs = np.linspace(0.01, 100, 400)", "require": ["log", "figsize", "tick_params", "MaxNLocator"]}),
        ("savefig", f"The guide's small panels, then export. On `fig, axs = plt.subplots(1, 3, figsize=(9, 3))` draw (1) "
                    f"`[1e5, 2e5, 3e5]` with its y labels in scientific notation (`ticklabel_format(axis='y', "
                    f"style='sci', scilimits=(0, 0))`), (2) `train_loss` against `val_loss` as dots with equal scales "
                    f"(`set_aspect('equal')`), (3) the same dots with every axis decoration hidden (`axis('off')`). "
                    f"Read the first panel's offset text (`axs[0].yaxis.get_major_formatter().get_offset()`, after "
                    f"`axs[0].get_yticklabels()`), save the figure as `ticks.png` at {dpi} dpi, close everything with "
                    f"`plt.close('all')`, and return `(offset text, image shape, number of open figures)`.",
         f"fig, axs = plt.subplots(1, 3, figsize=(9, 3))\naxs[0].plot([1e5, 2e5, 3e5])\n"
         f"axs[0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))\naxs[1].plot(train_loss, val_loss, 'o')\n"
         f"axs[1].set_aspect('equal')\naxs[2].plot(train_loss, val_loss, 'o')\naxs[2].axis('off')\naxs[0].get_yticklabels()\n"
         f"off = axs[0].yaxis.get_major_formatter().get_offset()\nfig.savefig('ticks.png', dpi={dpi})\nplt.close('all')\n"
         f"result = (off, plt.imread('ticks.png').shape, len(plt.get_fignums()))",
         {"require": ["ticklabel_format", "set_aspect", "savefig", "close"]}),
        ("ConciseDateFormatter", "The guide's date panels on `fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 5))` "
                                 "(**not** sharing x: shared axes would share one formatter): `value` against `dates` "
                                 "on both; the top with a tick on the first of every month labelled like 'Jan 2024', the "
                                 "bottom with the automatic locator and `mdates.ConciseDateFormatter(a2.xaxis."
                                 "get_major_locator())`. Return the bottom formatter's offset text (`a2.xaxis."
                                 "get_major_formatter().get_offset()`, read after `a2.get_xticklabels()`).",
         f"{DIMP}fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 5))\na1.plot(dates, value)\na2.plot(dates, value)\n"
         f"a1.xaxis.set_major_locator(mdates.MonthLocator())\na1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))\n"
         f"a2.xaxis.set_major_formatter(mdates.ConciseDateFormatter(a2.xaxis.get_major_locator()))\na2.get_xticklabels()\n"
         f"result = a2.xaxis.get_major_formatter().get_offset()",
         {"post": "_, [(fc.lines(a), fc.ticks(a, 'x')) for a in fc.axes()], fc.size()",
          "require": ["MonthLocator", "DateFormatter", "ConciseDateFormatter"]}),
        ("StrMethodFormatter", "A formatter tour on a 2 × 2 grid (`plt.subplots(2, 2)`), `val_acc` against `epochs` on "
                               "every panel, with the y labels written by (in reading order) `FormatStrFormatter('%.1f')`, "
                               "`StrMethodFormatter('{x:,.2f}')`, `PercentFormatter(xmax=1.0)` and `FuncFormatter(lambda "
                               "v, pos: f'{v * 100:.0f} pts')`; title each panel with its formatter's class name "
                               "(`type(f).__name__`).",
         f"{IMP}fmts = [mticker.FormatStrFormatter('%.1f'), mticker.StrMethodFormatter('{{x:,.2f}}'),\n"
         f"        mticker.PercentFormatter(xmax=1.0), mticker.FuncFormatter(lambda v, pos: f'{{v * 100:.0f}} pts')]\n"
         f"fig, axs = plt.subplots(2, 2)\nfor a, f in zip(axs.flat, fmts):\n    a.plot(epochs, val_acc)\n"
         f"    a.yaxis.set_major_formatter(f)\n    a.set_title(type(f).__name__)",
         {"post": "[(fc.lines(a), a.get_title(), fc.formatted(a, 'y', [0.5, 0.875])) for a in fc.axes()]",
          "require": ["FormatStrFormatter", "StrMethodFormatter", "PercentFormatter", "FuncFormatter"]}),
    ]
    return catalog(rng, items, code, f"{desc}, `train_acc` the training accuracy; {ddesc}.")


@template(M, "hard")
def axes_specs(rng):
    code, desc, n = curves(rng, acc=True)
    s = seeded(rng)
    m = rng.choice([8, 10, 12])
    k = rng.randint(5, 7)
    spe = rng.choice([50, 100, 250])
    c = rng.choice([2.5, 3, 3.5])
    names = rng.choice([["resnet18", "resnet50", "vit-tiny", "vit-base", "convnext", "mobilenet", "regnet"],
                        ["gpt-small", "gpt-medium", "llama-1b", "llama-3b", "mistral-7b", "phi-2", "gemma-2b"]])[:k]
    items = [
        ("polyfit", "A scaling-law figure (`sizes`: model sizes, `size_loss`: their loss): the points as dots on log–log "
                    "axes (`ax.loglog(sizes, size_loss, 'o')`); a power law fitted with `b, a = np.polyfit(np.log10(sizes), "
                    "np.log10(size_loss), 1)` and drawn as a dashed line `10 ** a * sizes ** b`; x labels in engineering "
                    "notation (`EngFormatter()`); grid lines at the major and minor ticks (`which='both'`, `alpha=0.3`); "
                    "the title `f'loss ∝ N^{b:.3f}'`. Return `b`.",
         f"{IMP}b, a = np.polyfit(np.log10(sizes), np.log10(size_loss), 1)\nfig, ax = plt.subplots()\nax.loglog(sizes, size_loss, 'o')\n"
         f"ax.plot(sizes, 10 ** a * sizes ** b, '--')\nax.xaxis.set_major_formatter(mticker.EngFormatter())\n"
         f"ax.grid(True, which='both', alpha=0.3)\nax.set_title(f'loss ∝ N^{{b:.3f}}')\nresult = b",
         {"post": f"_, fc.lines(fc.ax(), 'marker', 'ls', nd=5), fc.scales(), {_fmt('x', (1e6, 1e8, 1e10))}, {LAB}['title'], fc.grid_on()",
          "setup": f"g2 = np.random.default_rng({s})\nsizes = np.round(np.logspace(6, 10, {m}))\n"
                   f"size_loss = np.round(2.5 * (sizes / 1e6) ** -0.08 * np.exp(0.01 * g2.normal(size={m})), 4)",
          "require": ["polyfit", "EngFormatter"], "tol": 1e-6}),
        ("set_xticks", "A learning-rate finder plot (`lr_loss`: the loss reached at each learning rate in `lrs`): "
                       "`lr_loss` against `lrs` on a log x-axis; a red dashed vertical line at the best learning rate "
                       "(lowest loss); major x ticks exactly at 1e-5, 1e-4, …, 1e-1 labelled like '1e-05' "
                       "(`StrMethodFormatter('{x:.0e}')`); no minor x ticks (`NullLocator()`); y limits 0 … twice the "
                       "median loss. Return the best learning rate.",
         f"{IMP}best = lrs[lr_loss.argmin()]\nfig, ax = plt.subplots()\nax.plot(lrs, lr_loss)\nax.set_xscale('log')\n"
         f"ax.axvline(best, color='red', linestyle='--')\nax.set_xticks(10.0 ** np.arange(-5, 0))\n"
         f"ax.xaxis.set_major_formatter(mticker.StrMethodFormatter('{{x:.0e}}'))\nax.xaxis.set_minor_locator(mticker.NullLocator())\n"
         f"ax.set_ylim(0, 2 * np.median(lr_loss))\nresult = best",
         {"post": f"_, fc.lines(fc.ax(), 'color', 'ls'), fc.scales(), {XT}, fc.ticks(fc.ax(), 'x', minor=True)['locs'], fc.limits()['ylim']",
          "setup": f"g2 = np.random.default_rng({s})\nlrs = np.logspace(-5, -1, 25)\n"
                   f"lr_loss = np.round((np.log10(lrs) + {c}) ** 2 / 2 + 0.3 + 0.05 * g2.random(25), 4)",
          "require": ["set_xticks", "StrMethodFormatter", "NullLocator"], "tol": 1e-9}),
        ("invert_yaxis", f"A leaderboard (`names`: {k} models, `scores`: their accuracies, best first): `ax.barh(names, "
                         f"scores)` with the best model at the **top**; the x-axis from 0.5 to 1 with a tick every 0.1 "
                         f"(`MultipleLocator(0.1)`) labelled as whole percentages (`PercentFormatter(xmax=1, "
                         f"decimals=0)`); vertical grid lines only (`ax.grid(True, axis='x', alpha=0.4)`) drawn below the "
                         f"bars (`set_axisbelow(True)`); no top / right spines.",
         f"{IMP}fig, ax = plt.subplots()\nax.barh(names, scores)\nax.invert_yaxis()\nax.set_xlim(0.5, 1)\n"
         f"ax.xaxis.set_major_locator(mticker.MultipleLocator(0.1))\nax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))\n"
         f"ax.grid(True, axis='x', alpha=0.4)\nax.set_axisbelow(True)\nax.spines[['top', 'right']].set_visible(False)",
         {"post": f"fc.bars(fc.ax(), 4), fc.ax().yaxis_inverted(), {YT}['labels'], {XT}, fc.limits()['xlim'], fc.grid_on(), "
                  f"fc.ax().get_axisbelow(), fc.spines()",
          "setup": f"names = {lit(names)}\nscores = np.round(np.sort(np.random.default_rng({s}).uniform(0.55, 0.95, {k}))[::-1], 3)",
          "require": ["barh", "MultipleLocator", "PercentFormatter"]}),
        ("LinearLocator", "Align the grids of a twin plot: `train_loss` on `ax` (left) and `val_acc` on `ax2 = "
                          "ax.twinx()` (right, color 'C1'), both y axes with exactly 6 evenly spaced ticks "
                          "(`LinearLocator(numticks=6)`) so their tick lines coincide; grid lines from `ax` only "
                          "(`ax.grid(True)`); y labels 'loss' and 'accuracy'. Return both lists of y tick positions.",
         f"{IMP}fig, ax = plt.subplots()\nax.plot(epochs, train_loss)\nax2 = ax.twinx()\nax2.plot(epochs, val_acc, color='C1')\n"
         f"for a in (ax, ax2):\n    a.yaxis.set_major_locator(mticker.LinearLocator(numticks=6))\nax.grid(True)\n"
         f"ax.set_ylabel('loss')\nax2.set_ylabel('accuracy')\nresult = [ax.get_yticks().tolist(), ax2.get_yticks().tolist()]",
         {"post": "_, [(fc.lines(a), a.get_ylabel(), fc.grid_on(a)) for a in fc.axes()], fc.lines(fc.ax(1), 'color', data=False)",
          "require": ["twinx", "LinearLocator"], "tol": 1e-9}),
        ("set_aspect", "A reliability diagram (`conf`: the mean confidence of 10 bins, `acc_bin`: the accuracy in each): "
                       "`acc_bin` against `conf` as dots joined by a line (`'o-'`), the ideal diagonal "
                       "(`ax.axline((0, 0), slope=1, color='gray', linestyle='--')`), both axes from 0 to 1 with a tick "
                       "every 0.2 (`MultipleLocator(0.2)`) labelled as percentages (`PercentFormatter(xmax=1)`), equal "
                       "scales (`set_aspect('equal')`), axis labels 'confidence' and 'accuracy'.",
         f"{IMP}fig, ax = plt.subplots()\nax.plot(conf, acc_bin, 'o-')\nax.axline((0, 0), slope=1, color='gray', linestyle='--')\n"
         f"ax.set_xlim(0, 1)\nax.set_ylim(0, 1)\nfor axis in (ax.xaxis, ax.yaxis):\n    axis.set_major_locator(mticker.MultipleLocator(0.2))\n"
         f"    axis.set_major_formatter(mticker.PercentFormatter(xmax=1))\nax.set_aspect('equal')\n"
         f"ax.set(xlabel='confidence', ylabel='accuracy')",
         {"post": f"fc.lines(fc.ax(), 'marker', 'color', 'ls'), {XT}, {YT}, fc.limits(), fc.ax().get_aspect(), {LAB}",
          "setup": f"conf = np.round(np.linspace(0.05, 0.95, 10), 2)\n"
                   f"acc_bin = np.round(np.clip(conf - 0.05 + 0.04 * np.random.default_rng({s}).normal(size=10), 0, 1), 3)",
          "require": ["axline", "MultipleLocator", "PercentFormatter", "set_aspect"]}),
        ("secondary_xaxis", f"Epochs below, optimizer steps above ({spe} steps per epoch): plot `train_loss` against "
                            f"`epochs` with the x label 'epoch', and add a secondary x-axis on top that shows steps "
                            f"(`sec = ax.secondary_xaxis('top', functions=(lambda e: e * {spe}, lambda s: s / {spe}))`) "
                            f"labelled 'step'. Return the secondary axis' x limits after a draw (`fig.canvas.draw()`, "
                            f"then `sec.get_xlim()`) as a list.",
         f"fig, ax = plt.subplots()\nax.plot(epochs, train_loss)\nax.set_xlabel('epoch')\n"
         f"sec = ax.secondary_xaxis('top', functions=(lambda e: e * {spe}, lambda s: s / {spe}))\nsec.set_xlabel('step')\n"
         f"fig.canvas.draw()\nresult = list(sec.get_xlim())",
         {"post": "_, fc.lines(fc.ax()), fc.labels()['xlabel'], [a.get_xlabel() for a in fc.ax().child_axes]",
          "require": ["secondary_xaxis"], "tol": 1e-6}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "hard")
def tick_math(rng):
    code, desc, n = curves(rng)
    dcode, ddesc, *_ = _days(rng)
    base = rng.choice([0.25, 0.5, 2, 5])
    lo = round(rng.uniform(-10, 5), 2)
    hi = round(lo + rng.uniform(4 * base, 12 * base), 2)
    nb = rng.choice([3, 4, 5, 6])
    t = rng.choice([0.1, 1, 10])
    y0, y1 = rng.choice([(0.01, 10), (0.1, 100), (0.001, 1)])
    a, c = rng.randint(3, 20), rng.randint(3, 20)
    items = [
        ("MultipleLocator", f"Which ticks does `MultipleLocator({base})` show in the view {lo} … {hi}? Compute them "
                            f"without Matplotlib — the multiples of {base} from the first one ≥ {lo} up to {hi} — then "
                            f"plot `train_loss` with `ax.set_xlim({lo}, {hi})` and that locator on the x-axis, and return "
                            f"your list. (Check: the ticks of `ax.get_xticks()` inside the view are the same.)",
         f"{IMP}ticks = (np.arange(np.ceil({lo} / {base}), np.floor({hi} / {base}) + 1) * {base}).tolist()\n"
         f"fig, ax = plt.subplots()\nax.plot(epochs, train_loss)\nax.set_xlim({lo}, {hi})\n"
         f"ax.xaxis.set_major_locator(mticker.MultipleLocator({base}))\nresult = ticks",
         {"post": f"_, {XT}['locs'], fc.limits()['xlim']", "require": ["ceil|floor|arange", "MultipleLocator"], "tol": 1e-9}),
        ("tick_values", f"Locators work without an Axes: `locator.tick_values(vmin, vmax)` returns the ticks it would "
                        f"place. Return `mticker.MaxNLocator(nbins={nb}).tick_values(train_loss.min(), train_loss.max())` "
                        f"as a list (they may reach a step beyond the data), and plot `train_loss` with that locator on "
                        f"the y-axis.",
         f"{IMP}loc = mticker.MaxNLocator(nbins={nb})\nresult = loc.tick_values(train_loss.min(), train_loss.max()).tolist()\n"
         f"fig, ax = plt.subplots()\nax.plot(epochs, train_loss)\nax.yaxis.set_major_locator(loc)",
         {"post": f"_, {YT}['locs']", "require": ["MaxNLocator", "tick_values"], "tol": 1e-9}),
        ("set_yscale", "Predict the y limits Matplotlib picks for `train_loss` on a log y-axis: the data range plus a 5% "
                       "margin on each side **in log10 space** (with `d = log10(max) − log10(min)`, from `10 ** "
                       "(log10(min) − 0.05·d)` to `10 ** (log10(max) + 0.05·d)`). Plot it on a log y-axis and return "
                       "your two numbers as a list (they match `ax.get_ylim()`).",
         "l0, l1 = np.log10(train_loss.min()), np.log10(train_loss.max())\nd = l1 - l0\nfig, ax = plt.subplots()\n"
         "ax.plot(epochs, train_loss)\nax.set_yscale('log')\nresult = [10 ** (l0 - 0.05 * d), 10 ** (l1 + 0.05 * d)]",
         {"post": "_, fc.scales(), fc.r(fc.ax().get_ylim(), 9)", "require": ["log10", "set_yscale|semilogy"], "tol": 1e-9}),
        ("transAxes", f"On a log y-axis with limits {y0} … {y1}, where does `val_loss[-1]` land, as a fraction of the Axes "
                      f"height (0 = bottom, 1 = top)? Compute it with the log formula, set the Axes up (`train_loss` "
                      f"plotted, `set_yscale('log')`, `set_ylim({y0}, {y1})`) and return it. (Check: "
                      f"`ax.transAxes.inverted().transform(ax.transData.transform((1, v)))[1]` gives the same.)",
         f"v = val_loss[-1]\nfig, ax = plt.subplots()\nax.plot(epochs, train_loss)\nax.set_yscale('log')\nax.set_ylim({y0}, {y1})\n"
         f"result = (np.log10(v) - np.log10({y0})) / (np.log10({y1}) - np.log10({y0}))",
         {"post": "_, fc.scales(), fc.limits()['ylim']", "require": ["log10|transAxes|transData"], "tol": 1e-9}),
        ("symlog", f"Matplotlib's symlog transform (base 10, `linscale=1`, `linthresh=t`) maps |v| ≤ t linearly to "
                   f"`v · c` with `c = 1 / (1 − 1/10)`, and anything beyond to `sign(v) · t · (c + log10(|v| / t))`. "
                   f"Implement it for `g = 100 * np.diff(val_loss)` with t = {t}, plot `g` with "
                   f"`ax.set_yscale('symlog', linthresh={t})`, and return your transformed values as a list (they match "
                   f"`ax.yaxis.get_transform().transform(g)`).",
         f"g = 100 * np.diff(val_loss)\nt, c = {t}, 1 / (1 - 1 / 10)\n"
         f"out = np.where(np.abs(g) <= t, g * c, np.sign(g) * t * (c + np.log10(np.maximum(np.abs(g), t) / t)))\n"
         f"fig, ax = plt.subplots()\nax.plot(epochs[1:], g)\nax.set_yscale('symlog', linthresh=t)\nresult = out.tolist()",
         {"post": "_, fc.scales(), fc.ax().yaxis.get_transform().linthresh", "require": ["linthresh"], "tol": 1e-9}),
        ("num2date", f"{ddesc[0].upper()}{ddesc[1:]}. Matplotlib stores dates as float **days since 1970-01-01** "
                     f"(`mdates.date2num`). Plot `value` against `dates`, read the automatic x limits (with the usual 5% "
                     f"margin, so they fall at odd times of day) and return them converted back to strings like "
                     f"'2024-01-04 07:12' (`mdates.num2date(v).strftime('%Y-%m-%d %H:%M')`).",
         f"{DIMP}fig, ax = plt.subplots()\nax.plot(dates, value)\n"
         f"result = [mdates.num2date(v).strftime('%Y-%m-%d %H:%M') for v in ax.get_xlim()]",
         {"post": f"_, {LINES}", "setup": dcode, "require": ["num2date"]}),
        ("MonthLocator", f"{ddesc[0].upper()}{ddesc[1:]}. How many ticks does `mdates.MonthLocator()` place when the "
                         f"x-axis shows `dates[{a}]` … `dates[-{c}]`? Count the first-of-month days in that stretch with "
                         f"NumPy (`d.astype('datetime64[M]')` truncates to the month), then plot `value` against `dates` "
                         f"with those limits and that locator, and return your count (it matches "
                         f"`len(ax.get_xticks())`).",
         f"{DIMP}d = dates[(dates >= dates[{a}]) & (dates <= dates[-{c}])]\n"
         f"count = int(np.sum(d == d.astype('datetime64[M]').astype('datetime64[D]')))\nfig, ax = plt.subplots()\n"
         f"ax.plot(dates, value)\nax.set_xlim(dates[{a}], dates[-{c}])\nax.xaxis.set_major_locator(mdates.MonthLocator())\n"
         f"result = count", {"post": f"_, {XT}['locs']", "setup": dcode, "require": ["MonthLocator", "astype"]}),
    ]
    return catalog(rng, items, code, f"{desc}.")


# ================================================================= challenge

def _nice_setup(seed: int, q: int, R: int, report: str) -> str:
    return (f"g = np.random.default_rng({seed})\nlo = g.integers(-{R}, {R}, {q})\nhi = lo + g.integers(1, {R}, {q})\n"
            f"nbins = g.integers(2, 10, {q})\nreport = {report!r}")


@template(M, "challenge")
def nice_tick_steps(rng):
    s = seeded(rng)
    report = rng.choice(["step", "ticks", "limits"])
    q, R = rng.randint(5, 8), rng.choice([100, 1000, 5000])
    setup = _nice_setup(s, q, R, report)
    sol = ("ladder = [1, 2, 5] + [m * 10 ** k for k in range(12) for m in (10, 20, 25, 50)]\n"
           "step = np.zeros(len(lo), dtype=np.int64)\ntodo = np.ones(len(lo), dtype=bool)\n"
           "for s in ladder:                  # the count isn't monotone in s: scan the ladder in order\n"
           "    ok = todo & (-(-hi // s) - lo // s <= nbins)\n    step[ok] = s\n    todo &= ~ok\n"
           "    if not todo.any():\n        break\n"
           + {"step": "result = step",
              "ticks": "result = hi // step + (-lo) // step + 1          # floor(hi/s) - ceil(lo/s) + 1",
              "limits": "result = np.column_stack([lo // step * step, -(-hi // step) * step])"}[report])
    tests = [_nice_setup(s + i, 200000, RR, report) for i, RR in ((1, 10 ** 9), (2, 60))]
    want = {"step": "the chosen step for each view (an integer array)",
            "ticks": "how many ticks of the chosen step land **inside** each view lo … hi (ends included)",
            "limits": "the view limits after `axes.autolimit_mode = 'round_numbers'` widens each view to whole steps: a "
                      "q × 2 integer array `[⌊lo/s⌋·s, ⌈hi/s⌉·s]`"}[report]
    return challenge(
        "Nice tick steps",
        "`MaxNLocator` doesn't space ticks arbitrarily: it picks a 'nice' step. On integers, the candidate steps are 1, "
        "2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, … (1, 2 and 5, then 10, 20, 25 and 50 times every power of "
        "ten). With step s the ticks sit at the multiples of s, and covering the view `lo[j] … hi[j]` takes "
        "⌈hi/s⌉ − ⌊lo/s⌋ intervals (from the last multiple ≤ lo to the first ≥ hi). For each view the locator picks "
        f"the **smallest** candidate step that needs at most `nbins[j]` intervals. Return {want}. "
        f"(`report` = {report!r}.)",
        setup, sol, tests, "q ≤ 2·10^5 views, |lo|, |hi| ≤ 2·10^9, 2 ≤ nbins ≤ 9",
        speed="O(q · L) for the L ≈ 50 candidate steps, vectorized over the views", answer_desc="")


def _offset_setup(seed: int, q: int, thr: int, report: str) -> str:
    return (f"g = np.random.default_rng({seed})\ne = g.integers(2, 12, {q})\n"
            f"lo = g.integers(1, 10, {q}) * 10 ** e + g.integers(-10 ** 6, 10 ** 6, {q})\n"
            f"hi = lo + 1 + g.integers(0, 10 ** g.integers(0, 7, {q}))\nflip = g.random({q}) < 0.3\n"
            f"lo, hi = np.where(flip, -hi, lo), np.where(flip, -lo, hi)\nthr = {thr}\nreport = {report!r}")


@template(M, "challenge")
def offset_text(rng):
    s = seeded(rng)
    report = rng.choice(["offset", "shown", "count"])
    q, thr = rng.randint(6, 10), rng.choice([2, 3, 4])
    setup = _offset_setup(s, q, thr, report)
    sol = ("A = np.maximum(np.abs(lo), np.abs(hi))\nB = np.minimum(np.abs(lo), np.abs(hi))\n"
           "p = np.full(len(lo), -1, dtype=np.int64)\np2 = np.full(len(lo), -1, dtype=np.int64)\nfor k in range(14):\n"
           "    a, b = A // 10 ** k, B // 10 ** k\n    p[(p < 0) & (a == b)] = k          # smallest power where they agree\n"
           "    p2[(p2 < 0) & (a - b <= 1)] = k    # smallest power where they differ by at most 1\n"
           "p = np.where(100 * (A - B) <= 10 ** p, p2, p)\nlead = A // 10 ** p\n"
           "offset = np.where(((lo > 0) | (hi < 0)) & (lead >= 10 ** (thr - 1)), np.sign(lo) * lead * 10 ** p, 0)\n"
           + {"offset": "result = offset", "shown": "result = hi - offset",
              "count": "result = int(np.count_nonzero(offset))"}[report])
    tests = [_offset_setup(s + i, 200000, tt, report) for i, tt in ((1, 4), (2, 3))]
    want = {"offset": "the offset of each view (an integer array, 0 for none)",
            "shown": "the value written at each view's top tick: `hi − offset` (before any ×10^k scaling)",
            "count": "how many views get an offset"}[report]
    return challenge(
        "The offset text",
        "When tick values share their leading digits (1000003 … 1000007), `ScalarFormatter` subtracts a common "
        "**offset** and writes it once at the end of the axis ('+1e6'). For a view whose smallest and largest visible "
        "ticks are the integers `lo[j] < hi[j]`, the rule is: no offset (0) if the two have different signs or one of "
        "them is 0. Otherwise let A ≥ B be their absolute values and p the smallest power (p ≥ 0) with "
        "A // 10^p == B // 10^p. If the span is tiny next to 10^p (100·(A − B) ≤ 10^p: the ticks straddle a multiple "
        "of a big power of ten, like 999998 … 1000003), use instead the smallest p with A // 10^p − B // 10^p ≤ 1. The "
        "offset is sign · (A // 10^p) · 10^p, but only if A // 10^p has at least `thr` digits (≥ 10^(thr − 1); "
        f"`axes.formatter.offset_threshold`), else 0. Return {want}. (`report` = {report!r}.)",
        setup, sol, tests, "q ≤ 2·10^5 views, |values| < 10^12",
        speed="O(q · D) for D ≈ 13 powers of ten, vectorized over the views")


def _thin_setup(seed: int, n: int, q: int, gap: int, wid: int, pad: int, report: str) -> str:
    return (f"g = np.random.default_rng({seed})\nx = np.cumsum(g.integers(1, {gap}, {n}))\nw = 2 * g.integers(1, {wid}, {n})\n"
            f"pad = {pad}\nk = (1 + ({n} - 1) * g.random({q}) ** 6).astype(np.int64)\nreport = {report!r}")


@template(M, "challenge")
def label_thinning(rng):
    s = seeded(rng)
    report = rng.choice(["overlaps", "stride", "greedy"])
    n, q, pad = rng.randint(8, 14), rng.randint(4, 6), rng.randint(2, 6)
    setup = _thin_setup(s, n, q, 30, 30, pad, report)
    common = "left, right = x - w // 2, x + w // 2\n"
    allk = ("n = len(x)\nst = np.arange(1, n, dtype=np.int64)\ncnt = (n - 1) // st          # stride s has (n - 1) // s "
            "neighbouring pairs\nS = np.repeat(st, cnt)                     # n/1 + n/2 + ... = O(n log n) pairs in all\n"
            "I = (np.arange(len(S)) - np.repeat(np.cumsum(cnt) - cnt, cnt)) * S\n"
            "bad = np.bincount(S[right[I] + pad > left[I + S]], minlength=n + 1)\n")
    sol = common + {"overlaps": allk + "result = bad[k]",
                    "stride": allk + "result = int(np.flatnonzero(bad[1:] == 0)[0]) + 1",
                    "greedy": ("L, R = left.tolist(), right.tolist()\nkept, last = 1, 0\nfor i in range(1, len(L)):\n"
                               "    if R[last] + pad <= L[i]:\n        kept, last = kept + 1, i\nresult = kept")}[report]
    tests = [_thin_setup(s + i, 100000, 200000, gg, ww, pp, report) for i, (gg, ww, pp) in
             ((1, (30, 300, 4)), (2, (6, 2000, 10)))]
    want = {"overlaps": "how many neighbouring pairs of shown labels (0 and s, s and 2s, …) collide with each query "
                        "stride s = `k[j]` (an integer array)",
            "stride": "the smallest stride s for which no neighbouring pair of shown labels collides",
            "greedy": "how many labels a greedy pass keeps instead: walk left to right, keep label 0, and keep each "
                      "next label that doesn't collide with the **last kept** one"}[report]
    return challenge(
        "Thinning crowded tick labels",
        "Tick j sits at pixel `x[j]` (increasing) and its centred label is `w[j]` pixels wide (even), covering "
        "`x[j] − w[j]/2 … x[j] + w[j]/2`. Two labels i < j **collide** when fewer than `pad` pixels separate them "
        "(`x[i] + w[i]/2 + pad > x[j] − w[j]/2`). A common fix shows only every s-th label (labels 0, s, 2s, …; e.g. "
        "`t.set_visible(i % s == 0)` over `ax.get_xticklabels()`); here only **neighbouring** shown labels are checked. "
        f"Return {want}. (`report` = {report!r}.)",
        setup, sol, tests, "n ≤ 10^5 labels, q ≤ 2·10^5 queries",
        speed="O(n log n + q): stride s has (n − 1) // s neighbouring pairs, and n/1 + n/2 + … + n/n = O(n log n)")
