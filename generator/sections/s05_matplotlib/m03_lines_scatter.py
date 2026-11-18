"""5.03 Line & scatter plots: fmt strings and line properties, Line2D handles, the color cycle, 2-D y, step lines,
log axes, scatter sizes / colors / colormaps, colorbars, and legends for categories (legend_elements)."""

from ._common import P, catalog, challenge, classes, curves, lit, seeded, signal, template

M = "5.03"
LINES = "fc.lines(fc.ax())"
YDATA = "[fc.ydata(a) for a in fc.axes()]"
# fc.scatter's offsets come back as None for 2-D masked arrays, so read them here
SC = ("from matplotlib.collections import PathCollection as _PC\n"
      "def _sc(a=None, i=0, *p):\n    a = a or fc.ax()\n    c = [x for x in a.collections if isinstance(x, _PC)][i]\n"
      "    d = fc.scatter(a, i, 6, *p)\n    d['offsets'] = fc.r(np.asarray(c.get_offsets()), 6)\n"
      "    if 'marker' in p:\n        d['marker'] = fc.r(c.get_paths()[0].vertices, 3)\n"
      "    if 'lw' in p:\n        d['lw'] = fc.r(c.get_linewidths(), 3)\n    return d\n"
      "def _all(a=None):\n    a = a or fc.ax()\n    return [fc.r(np.asarray(c.get_offsets()), 6) for c in a.collections if isinstance(c, _PC)]\n")


def _pts03(rng, n=(30, 80)):
    """`px`, `py` (n random points) and `v` (a continuous value per point)."""
    n_ = rng.randint(*n)
    s = seeded(rng)
    kind = rng.choice(["px + py", "np.hypot(px, py)", "px * py", "np.sin(px) + py"])
    code = (f"g = np.random.default_rng({s})\npx = np.round(g.normal(size={n_}), 3)\npy = np.round(g.normal(size={n_}), 3)\n"
            f"v = np.round({kind}, 3)")
    return code, f"`px`, `py` hold {n_} random 2-D points and `v = {kind}` a value per point", n_


def _xs03(rng):
    n = rng.choice([30, 50, 80])
    return f"x = np.linspace(0, 10, {n})", n


# ================================================================= lite

@template(M, "lite")
def line_styles(rng):
    code, desc, n = signal(rng)
    fmt = rng.choice(["r--", "g:", "k-.", "b-"])
    fmt2 = rng.choice(["go-", "bs--", "r^:", "kD-"])
    col = rng.choice(["tab:green", "tab:purple", "#ff7f0e", "black"])
    ls = rng.choice(["-.", ":", "--"])
    lw, ms, k = rng.choice([1.5, 2, 3]), rng.choice([3, 5, 8]), rng.choice([3, 5, 10])
    al, z = rng.choice([0.3, 0.5, 0.8]), rng.choice([3, 5, 10])
    items = [
        ("plot", f"Plot `y` against `x` with the fmt string `{fmt!r}` (color + line style in one short string).",
         f"fig, ax = plt.subplots()\nax.plot(x, y, {fmt!r})", {"post": "fc.lines(fc.ax(), 'color', 'ls')"}),
        ("plot", f"Plot `y` against `x` with the fmt string `{fmt2!r}` (color, marker and line style).",
         f"fig, ax = plt.subplots()\nax.plot(x, y, {fmt2!r})", {"post": "fc.lines(fc.ax(), 'color', 'ls', 'marker')"}),
        ("color", f"Plot `y` against `x` in the color {col!r} with the line style {ls!r}, using the keyword arguments "
                  f"`color=` and `linestyle=`.", f"fig, ax = plt.subplots()\nax.plot(x, y, color={col!r}, linestyle={ls!r})",
         {"post": "fc.lines(fc.ax(), 'color', 'ls')", "require": ["color", "linestyle|ls"]}),
        ("linewidth", f"Plot `y` against `x` with a line width of {lw} points (`linewidth=`).",
         f"fig, ax = plt.subplots()\nax.plot(x, y, linewidth={lw})", {"post": "fc.lines(fc.ax(), 'lw')", "require": ["linewidth|lw"]}),
        ("marker", f"Plot `y` against `x` as a line with circle markers of size {ms} (`marker='o'`, `markersize=`).",
         f"fig, ax = plt.subplots()\nax.plot(x, y, marker='o', markersize={ms})",
         {"post": "fc.lines(fc.ax(), 'marker', 'ms', 'ls')", "require": ["marker", "markersize|ms"]}),
        ("markerfacecolor", "Plot `y` against `x` with **hollow-looking** circle markers: `marker='o'` with a white face "
                            "(`markerfacecolor='white'`).",
         "fig, ax = plt.subplots()\nax.plot(x, y, marker='o', markerfacecolor='white')",
         {"post": "fc.lines(fc.ax(), 'marker', 'mfc')", "require": ["markerfacecolor|mfc"]}),
        ("markevery", f"Plot `y` against `x` with square markers (`'s-'`) on every {k}-th point only (`markevery={k}`).",
         f"fig, ax = plt.subplots()\nax.plot(x, y, 's-', markevery={k})", {"post": "fc.lines(fc.ax(), 'marker', 'markevery')"}),
        ("alpha", f"Plot `y` against `x` half-transparent: `alpha={al}`.", f"fig, ax = plt.subplots()\nax.plot(x, y, alpha={al})",
         {"post": "fc.lines(fc.ax(), 'alpha')"}),
        ("label", "Plot `y` against `x` labelled 'signal' and `-y` labelled 'mirror', then show a legend.",
         "fig, ax = plt.subplots()\nax.plot(x, y, label='signal')\nax.plot(x, -y, label='mirror')\nax.legend()",
         {"post": "fc.lines(fc.ax(), 'label'), fc.legend()", "require": ["label", "legend"]}),
        ("zorder", f"Draw `-y` first, then `y` with `zorder={z}` so it sits on top of anything with the default zorder.",
         f"fig, ax = plt.subplots()\nax.plot(x, -y)\nax.plot(x, y, zorder={z})", {"post": "fc.lines(fc.ax(), 'zorder')"}),
        ("drawstyle", "Plot `y` against `x` as a **staircase** that holds each value until the next point: "
                      "`drawstyle='steps-post'`.", "fig, ax = plt.subplots()\nax.plot(x, y, drawstyle='steps-post')",
         {"post": "fc.lines(fc.ax(), 'drawstyle')"}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "lite")
def scatter_basics(rng):
    code, desc, n = _pts03(rng)
    s = rng.choice([5, 20, 50])
    cm = rng.choice(["viridis", "plasma", "coolwarm", "magma"])
    col = rng.choice(["red", "tab:orange", "k", "#2ca02c"])
    mk = rng.choice(["^", "s", "x", "D"])
    al = rng.choice([0.3, 0.5, 0.7])
    lab = rng.choice(["x + y", "value", "score"])
    items = [
        ("scatter", "Scatter `py` against `px` on `fig, ax` (`ax.scatter`).",
         "fig, ax = plt.subplots()\nax.scatter(px, py)", {"post": SC + "_sc()"}),
        ("s", f"Scatter `py` against `px` with every marker of area {s} (`s=`, in points²).",
         f"fig, ax = plt.subplots()\nax.scatter(px, py, s={s})", {"post": SC + "_sc(None, 0, 'sizes')", "require": ["s"]}),
        ("cmap", f"Scatter `py` against `px` colored by `v` (`c=v`) with the colormap {cm!r}.",
         f"fig, ax = plt.subplots()\nax.scatter(px, py, c=v, cmap={cm!r})",
         {"post": SC + "_sc(None, 0, 'array', 'cmap')", "require": ["c", "cmap"]}),
        ("color", f"Scatter `py` against `px` with all points in the color {col!r} (`color=`).",
         f"fig, ax = plt.subplots()\nax.scatter(px, py, color={col!r})", {"post": SC + "_sc(None, 0, 'color')"}),
        ("marker", f"Scatter `py` against `px` with the marker {mk!r}.",
         f"fig, ax = plt.subplots()\nax.scatter(px, py, marker={mk!r})", {"post": SC + "_sc(None, 0, 'marker')"}),
        ("alpha", f"Scatter `py` against `px` with transparency `alpha={al}`, so overlapping points look darker.",
         f"fig, ax = plt.subplots()\nax.scatter(px, py, alpha={al})", {"post": SC + "_sc(None, 0, 'alpha')"}),
        ("edgecolors", "Scatter `py` against `px` with black marker outlines (`edgecolors='k'`) of width 0.5 "
                       "(`linewidths=0.5`).", "fig, ax = plt.subplots()\nax.scatter(px, py, edgecolors='k', linewidths=0.5)",
         {"post": SC + "_sc(None, 0, 'edgecolor', 'lw')", "require": ["edgecolors", "linewidths"]}),
        ("colorbar", f"Scatter `py` against `px` colored by `v`, keep the handle `sc`, and add a colorbar labelled "
                     f"{lab!r} with `fig.colorbar(sc, ax=ax, label=...)`.",
         f"fig, ax = plt.subplots()\nsc = ax.scatter(px, py, c=v)\nfig.colorbar(sc, ax=ax, label={lab!r})",
         {"post": SC + "_sc(fc.ax(0), 0, 'array'), fc.colorbars()", "require": ["colorbar", "ax"]}),
        ("label", "Scatter the points with `px > 0` labelled 'right' and the others labelled 'left' (two scatter calls), "
                  "and add a legend.",
         "fig, ax = plt.subplots()\nax.scatter(px[px > 0], py[px > 0], label='right')\nax.scatter(px[px <= 0], py[px <= 0], label='left')\n"
         "ax.legend()", {"post": SC + "_all(), fc.legend()", "require": ["label", "legend"]}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "lite")
def log_axes(rng):
    code, n = _xs03(rng)
    b = rng.choice([2, 10])
    k = rng.choice([2, 3])
    items = [
        ("semilogy", "Plot `np.exp(x)` against `x` with a logarithmic **y**-axis in one call (`ax.semilogy`).",
         "fig, ax = plt.subplots()\nax.semilogy(x, np.exp(x))", {"post": f"{LINES}, fc.scales()"}),
        ("semilogx", "Plot `x` against `x + 1` with a logarithmic **x**-axis in one call (`ax.semilogx(x + 1, x)`).",
         "fig, ax = plt.subplots()\nax.semilogx(x + 1, x)", {"post": f"{LINES}, fc.scales()"}),
        ("loglog", f"Plot `(x + 1) ** {k}` against `x + 1` with **both** axes logarithmic (`ax.loglog`): a power law "
                   f"becomes a straight line.", f"fig, ax = plt.subplots()\nax.loglog(x + 1, (x + 1) ** {k})",
         {"post": f"{LINES}, fc.scales()"}),
        ("set_yscale", "Plot `np.exp(x / 2)` against `x` with `ax.plot`, then switch the y-axis to a log scale with "
                       "`ax.set_yscale('log')`.", "fig, ax = plt.subplots()\nax.plot(x, np.exp(x / 2))\nax.set_yscale('log')",
         {"post": f"{LINES}, fc.scales()"}),
        ("plt.semilogy", "With **pyplot** only, plot `np.exp(-x)` against `x` on a log y-axis (`plt.semilogy`).",
         "plt.semilogy(x, np.exp(-x))", {"post": f"{LINES}, fc.scales()"}),
        ("set_xscale", f"Plot `x` against `2 ** x` on a log x-axis with **base {b}** (`ax.set_xscale('log', base={b})`).",
         f"fig, ax = plt.subplots()\nax.plot(2 ** x, x)\nax.set_xscale('log', base={b})",
         {"post": f"{LINES}, fc.scales(), fc.ax().xaxis.get_transform().base"}),
    ]
    return catalog(rng, items, code, f"`x` holds {n} points from 0 to 10.")


# ================================================================= easy

@template(M, "easy")
def line_handles(rng):
    code, n = _xs03(rng)
    col = rng.choice(["purple", "tab:red", "k", "#17becf"])
    lab = rng.choice(["updated", "scaled", "new"])
    d = rng.choice([2, 4, 5])
    k = rng.randint(2, 5)
    items = [
        ("set_ydata", f"Keep the handle of a line: `line, = ax.plot(x, x / 10)`. Then, **through the handle**, change "
                      f"its color to {col!r}, its y data to `x / {10 * d}` and its label to {lab!r}, and add a legend.",
         f"fig, ax = plt.subplots()\nline, = ax.plot(x, x / 10)\nline.set_color({col!r})\nline.set_ydata(x / {10 * d})\n"
         f"line.set_label({lab!r})\nax.legend()",
         {"post": "fc.lines(fc.ax(), 'color', 'label'), fc.legend()", "require": ["set_color", "set_ydata", "set_label"]}),
        ("get_color", f"Plot {k} lines without choosing colors (`ax.plot(x, x * i)` for i = 1 … {k}) and return their "
                      f"colors read back from the lines (`get_color()`): the default cycle's 'C0', 'C1', …",
         f"fig, ax = plt.subplots()\nresult = [ax.plot(x, x * i)[0].get_color() for i in range(1, {k + 1})]"),
        ("column_stack", f"`ax.plot` with a 2-D `y` draws one line per **column**. Plot `np.column_stack([np.sin(x + p) "
                         f"for p in range({k})])` against `x` in a single call and return how many Line2D objects it "
                         f"returned.",
         f"fig, ax = plt.subplots()\nlines = ax.plot(x, np.column_stack([np.sin(x + p) for p in range({k})]))\nresult = len(lines)",
         {"post": f"_, {LINES}"}),
        ("set_linestyle", "Draw `line, = ax.plot(x, np.sin(x))`, then make it dashed with circle markers through the "
                          "handle (`line.set_linestyle('--')`, `line.set_marker('o')`).",
         "fig, ax = plt.subplots()\nline, = ax.plot(x, np.sin(x))\nline.set_linestyle('--')\nline.set_marker('o')",
         {"post": "fc.lines(fc.ax(), 'ls', 'marker')", "require": ["set_linestyle|set_ls", "set_marker"]}),
        ("get_xydata", "Draw `line, = ax.plot(x, np.cos(x))` and return the shape of `line.get_xydata()` (one (x, y) "
                       "row per point).", "fig, ax = plt.subplots()\nline, = ax.plot(x, np.cos(x))\nresult = line.get_xydata().shape"),
        ("plt.setp", f"Draw two lines (`np.sin(x)`, `np.cos(x)`) and give **both** the linewidth {d} and the color "
                     f"{col!r} with one `plt.setp(lines, ...)` call on the list `ax.plot` returns (plot both in one "
                     f"call: `ax.plot(x, np.sin(x), x, np.cos(x))`).",
         f"fig, ax = plt.subplots()\nlines = ax.plot(x, np.sin(x), x, np.cos(x))\nplt.setp(lines, linewidth={d}, color={col!r})",
         {"post": "fc.lines(fc.ax(), 'lw', 'color')"}),
        ("set_data", f"Replace a line's data entirely: draw `line, = ax.plot(x, x)`, then `line.set_data(x * {d}, x ** 2)`; "
                     f"the view doesn't follow by itself, so call `ax.relim()` and `ax.autoscale_view()`. Return the new "
                     f"x limits.",
         f"fig, ax = plt.subplots()\nline, = ax.plot(x, x)\nline.set_data(x * {d}, x ** 2)\nax.relim()\nax.autoscale_view()\n"
         f"result = list(ax.get_xlim())", {"post": f"_, {LINES}", "require": ["set_data", "relim", "autoscale_view"]}),
        ("remove", f"Plot {k} lines `np.sin(x + i)` (i = 0 … {k - 1}), then remove the **first** one again with "
                   f"`.remove()` on its handle.",
         f"fig, ax = plt.subplots()\nlines = [ax.plot(x, np.sin(x + i))[0] for i in range({k})]\nlines[0].remove()",
         {"post": LINES}),
    ]
    return catalog(rng, items, code, f"`x` holds {n} points from 0 to 10.")


@template(M, "easy")
def fmt_strings(rng):
    code, desc, n = signal(rng)
    fmt = rng.choice(["g^:", "rs--", "ko-.", "bD-"])
    names = {"g": "green", "r": "red", "k": "black", "b": "blue"}
    mk = {"^": "^", "s": "s", "o": "o", "D": "D"}
    ls = {"g^:": ":", "rs--": "--", "ko-.": "-.", "bD-": "-"}[fmt]
    c3 = rng.choice(["C1", "C2", "C3"])
    items = [
        ("plot", f"Rewrite `ax.plot(x, y, {fmt!r})` with **keyword arguments** instead of the fmt string (`color=`, "
                 f"`marker=`, `linestyle=`).",
         f"fig, ax = plt.subplots()\nax.plot(x, y, color={names[fmt[0]]!r}, marker={mk[fmt[1]]!r}, linestyle={ls!r})",
         {"post": "fc.lines(fc.ax(), 'color', 'marker', 'ls')", "require": ["color", "marker", "linestyle|ls"]}),
        ("plot", "Rewrite `ax.plot(x, y, color='black', linestyle='--')` as a fmt string.",
         "fig, ax = plt.subplots()\nax.plot(x, y, 'k--')", {"post": "fc.lines(fc.ax(), 'color', 'ls')"}),
        ("plot", "Plot `y`, `y + 1` and `y + 2` against `x` with the fmt strings 'b-', 'r--' and 'k:'.",
         "fig, ax = plt.subplots()\nax.plot(x, y, 'b-')\nax.plot(x, y + 1, 'r--')\nax.plot(x, y + 2, 'k:')",
         {"post": "fc.lines(fc.ax(), 'color', 'ls')"}),
        ("get_linestyle", "A fmt string with a marker but no line style draws **markers only**. Plot `ax.plot(x, y, 'bs')` "
                          "and return `[line.get_linestyle(), line.get_marker()]`.",
         "fig, ax = plt.subplots()\nline, = ax.plot(x, y, 'bs')\nresult = [line.get_linestyle(), line.get_marker()]"),
        ("plot", f"fmt strings accept cycle colors too: plot `y` with {c3 + 'o--'!r} and return the line's color as hex "
                 f"(`mpl.colors.to_hex(line.get_color())`).",
         f"fig, ax = plt.subplots()\nline, = ax.plot(x, y, {c3 + 'o--'!r})\nresult = mpl.colors.to_hex(line.get_color())",
         {"post": "_, fc.lines(fc.ax(), 'ls', 'marker')"}),
        ("plot", "One call, three lines: `ax.plot(x, y, 'r-', x, -y, 'b--', x, 2 * y, 'g:')` takes (x, y, fmt) groups. "
                 "Draw it and return the number of lines on the Axes.",
         "fig, ax = plt.subplots()\nax.plot(x, y, 'r-', x, -y, 'b--', x, 2 * y, 'g:')\nresult = len(ax.get_lines())",
         {"post": "_, fc.lines(fc.ax(), 'color', 'ls')"}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "easy")
def color_cycle(rng):
    code, n = _xs03(rng)
    k = rng.randint(3, 6)
    cols = rng.choice([["k", "r"], ["#1b9e77", "#d95f02", "#7570b3"], ["tab:blue", "tab:gray"]])
    cm = rng.choice(["viridis", "plasma", "cividis"])
    items = [
        ("set_prop_cycle", f"Make this Axes cycle through the colors {cols} (`ax.set_prop_cycle(color=...)`), then plot "
                           f"{k} lines `np.sin(x + i)`, i = 0 … {k - 1}.",
         f"fig, ax = plt.subplots()\nax.set_prop_cycle(color={cols})\nfor i in range({k}):\n    ax.plot(x, np.sin(x + i))",
         {"post": "fc.lines(fc.ax(), 'color')"}),
        ("set_prop_cycle", f"Cycle color **and** line style together: `ax.set_prop_cycle(color=['k', 'r'], "
                           f"linestyle=['-', '--'])`, then plot {k} lines `x * i`, i = 1 … {k}.",
         f"fig, ax = plt.subplots()\nax.set_prop_cycle(color=['k', 'r'], linestyle=['-', '--'])\nfor i in range(1, {k + 1}):\n"
         f"    ax.plot(x, x * i)", {"post": "fc.lines(fc.ax(), 'color', 'ls')"}),
        ("colormaps", f"Color {k} lines with evenly spaced colors from the {cm!r} colormap: "
                      f"`colors = mpl.colormaps[{cm!r}](np.linspace(0, 1, {k}))`, line i (`np.sin(x + i)`) gets "
                      f"`colors[i]`.",
         f"fig, ax = plt.subplots()\ncolors = mpl.colormaps[{cm!r}](np.linspace(0, 1, {k}))\nfor i in range({k}):\n"
         f"    ax.plot(x, np.sin(x + i), color=colors[i])", {"post": "fc.lines(fc.ax(), 'color')", "require": ["colormaps|cm"]}),
        ("color", f"An explicit color does **not** use up a cycle color. Plot `x` (default), `2 * x` with `color='k'`, "
                  f"then `3 * x` (default) and return the three colors as hex strings.",
         "fig, ax = plt.subplots()\na, = ax.plot(x, x)\nb, = ax.plot(x, 2 * x, color='k')\nc, = ax.plot(x, 3 * x)\n"
         "result = [mpl.colors.to_hex(l.get_color()) for l in (a, b, c)]"),
        ("rcParams", "The default cycle lives in `plt.rcParams['axes.prop_cycle']`. Return how many colors it has "
                     "(`len(plt.rcParams['axes.prop_cycle'].by_key()['color'])`) and plot that many + 1 lines `x + i`; "
                     "return `[n_colors, color of the first line == color of the last line]`.",
         "n_colors = len(plt.rcParams['axes.prop_cycle'].by_key()['color'])\nfig, ax = plt.subplots()\n"
         "lines = [ax.plot(x, x + i)[0] for i in range(n_colors + 1)]\nresult = [n_colors, lines[0].get_color() == lines[-1].get_color()]"),
        ("color", f"Use the cycle colors by name: plot `np.sin(x)` in 'C{k}' and `np.cos(x)` in 'C0', and return their "
                  f"hex colors.",
         f"fig, ax = plt.subplots()\na, = ax.plot(x, np.sin(x), color='C{k}')\nb, = ax.plot(x, np.cos(x), color='C0')\n"
         f"result = [mpl.colors.to_hex(a.get_color()), mpl.colors.to_hex(b.get_color())]"),
    ]
    return catalog(rng, items, code, f"`x` holds {n} points from 0 to 10.")


@template(M, "easy")
def scatter_readback(rng):
    code, desc, n = _pts03(rng)
    s = rng.choice([10, 25, 40])
    lo, hi = rng.choice([(-1, 1), (-2, 2), (0, 3)])
    items = [
        ("get_offsets", "Scatter `py` against `px` and return the shape of `sc.get_offsets()` (one (x, y) row per point).",
         "fig, ax = plt.subplots()\nsc = ax.scatter(px, py)\nresult = sc.get_offsets().shape"),
        ("get_sizes", f"Scatter with `s={s}` and with `s=10 + 20 * np.abs(v)`; return `[len(a.get_sizes()), "
                      f"len(b.get_sizes())]` (a scalar size is stored once).",
         f"fig, ax = plt.subplots()\na = ax.scatter(px, py, s={s})\nb = ax.scatter(px, py, s=10 + 20 * np.abs(v))\n"
         f"result = [len(a.get_sizes()), len(b.get_sizes())]", {"post": SC + "_, _sc(None, 1, 'sizes')['sizes']"}),
        ("get_array", "Scatter colored by `c=v` and return the color values the scatter stores (`sc.get_array()` as a "
                      "list) — they're mapped through the colormap at draw time.",
         "fig, ax = plt.subplots()\nsc = ax.scatter(px, py, c=v)\nresult = sc.get_array().tolist()"),
        ("set_offsets", "Move existing points instead of redrawing: scatter `py` against `px`, then shift every point "
                        "right by 1 with `sc.set_offsets(np.column_stack([px + 1, py]))`.",
         "fig, ax = plt.subplots()\nsc = ax.scatter(px, py)\nsc.set_offsets(np.column_stack([px + 1, py]))",
         {"post": SC + "_sc()", "require": ["set_offsets", "column_stack"]}),
        ("set_clim", f"Scatter colored by `v` and fix the color range to {lo} … {hi} (`sc.set_clim({lo}, {hi})`), so "
                     f"values outside it saturate. Add a colorbar.",
         f"fig, ax = plt.subplots()\nsc = ax.scatter(px, py, c=v)\nsc.set_clim({lo}, {hi})\nfig.colorbar(sc, ax=ax)",
         {"post": SC + "_sc(fc.ax(0), 0, 'clim', 'array'), len(fc.colorbars())", "require": ["set_clim|vmin"]}),
        ("get_facecolors", "With `c=` given a **list of colors** (one per point: 'red' where `v > 0`, else 'blue'), the "
                           "scatter stores an RGBA row per point. Draw it and return `sc.get_facecolors().shape`.",
         "fig, ax = plt.subplots()\nsc = ax.scatter(px, py, c=['red' if t > 0 else 'blue' for t in v])\n"
         "result = sc.get_facecolors().shape",
         {"post": "_, [fc.hexcolor(c) for c in fc.ax().collections[0].get_facecolors()]"}),
        ("set_array", "Recolor an existing scatter: draw it with `c=v`, then switch the colors to `-v` with "
                      "`sc.set_array(-v)`.", "fig, ax = plt.subplots()\nsc = ax.scatter(px, py, c=v)\nsc.set_array(-v)",
         {"post": SC + "_sc(None, 0, 'array')"}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "easy")
def class_legends(rng):
    code, desc, k, names = classes(rng)
    mk = rng.choice(["o", "^", "s"])
    ms = rng.choice([2, 3])
    loc = rng.choice(["upper right", "lower left", "best", "upper left"])
    items = [
        ("legend", "One scatter call per class gives automatic legend entries: for each class c, scatter its points "
                   "with `label=class_names[c]`, then add a legend.",
         f"fig, ax = plt.subplots()\nfor c in range({k}):\n    xk, yk = pts[labels == c].T\n    ax.scatter(xk, yk, label=class_names[c])\n"
         f"ax.legend()", {"post": SC + "_all(), fc.legend()", "require": ["scatter", "legend"]}),
        ("legend_elements", f"Scatter **all** points in one call colored by class (`c=labels, cmap='tab10'`), then build "
                            f"the legend from the scatter's own entries: `ax.legend(*sc.legend_elements(), "
                            f"title='class')`.",
         "fig, ax = plt.subplots()\nsc = ax.scatter(pts[:, 0], pts[:, 1], c=labels, cmap='tab10')\n"
         "ax.legend(*sc.legend_elements(), title='class')",
         {"post": SC + "_sc(None, 0, 'array', 'cmap'), fc.legend(), fc.legend_title()", "require": ["legend_elements", "title"]}),
        ("legend_elements", "Same single colored scatter (`c=labels`), but put the **class names** in the legend: take "
                            "the handles from `sc.legend_elements()` and pass `class_names` as the labels.",
         "fig, ax = plt.subplots()\nsc = ax.scatter(pts[:, 0], pts[:, 1], c=labels)\nhandles, _ = sc.legend_elements()\n"
         "ax.legend(handles, class_names)", {"post": SC + "_sc(None, 0, 'array'), fc.legend()"}),
        ("markerscale", f"Small markers make tiny legend symbols: scatter each class with `s=5` and its name as label, "
                        f"and enlarge the legend markers {ms}× (`ax.legend(markerscale={ms})`).",
         f"fig, ax = plt.subplots()\nfor c in range({k}):\n    ax.scatter(*pts[labels == c].T, s=5, label=class_names[c])\n"
         f"ax.legend(markerscale={ms})", {"post": SC + "_all(), fc.legend(), fc.ax().get_legend().markerscale"}),
        ("loc", f"Scatter each class with marker {mk!r} and its name as label; place the legend at {loc!r} "
                f"(`loc=`) with the title 'class'.",
         f"fig, ax = plt.subplots()\nfor c in range({k}):\n    ax.scatter(*pts[labels == c].T, marker={mk!r}, label=class_names[c])\n"
         f"ax.legend(loc={loc!r}, title='class')",
         {"post": SC + "_all(), fc.legend(), fc.legend_title(), fc.ax().get_legend()._loc", "require": ["loc", "legend"]}),
        (".T", "Unpack each class' points into x and y with `.T`: `xk, yk = pts[labels == c].T`, and plot each class as "
               "**markers only** with `ax.plot(xk, yk, 'o', label=class_names[c])` (a line plot can draw scatters too). "
               "Add a legend.",
         f"fig, ax = plt.subplots()\nfor c in range({k}):\n    xk, yk = pts[labels == c].T\n    ax.plot(xk, yk, 'o', label=class_names[c])\n"
         f"ax.legend()", {"post": "fc.lines(fc.ax(), 'ls', 'marker', 'label'), fc.legend()", "require": ["T"]}),
    ]
    return catalog(rng, items, code, f"{desc}.")


# ================================================================= medium

@template(M, "medium")
def styled_training_curves(rng):
    code, desc, n = curves(rng, acc=True)
    k = rng.choice([3, 4, 5])
    a = rng.choice([0.6, 0.8, 0.9])
    s = seeded(rng)
    items = [
        ("markevery", f"Plot `train_loss` as a solid 'C0' line labelled 'train' and `val_loss` as a dashed 'C1' line "
                      f"labelled 'val' with circle markers on every {k}-th epoch (`markevery={k}`); then mark the best "
                      f"epoch (lowest `val_loss`) with a red star (`'r*'`, `markersize=12`, `zorder=3`) and add a legend.",
         f"fig, ax = plt.subplots()\nax.plot(epochs, train_loss, color='C0', label='train')\n"
         f"ax.plot(epochs, val_loss, '--', color='C1', marker='o', markevery={k}, label='val')\nb = val_loss.argmin()\n"
         f"ax.plot(epochs[b], val_loss[b], 'r*', markersize=12, zorder=3)\nax.legend()",
         {"post": "fc.lines(fc.ax(), 'color', 'ls', 'marker', 'markevery', 'ms', 'zorder', 'label'), fc.legend()",
          "require": ["markevery", "zorder", "argmin"]}),
        ("alpha", f"Raw and smoothed: plot `val_loss` faintly (`alpha=0.3`, 'C0') and its exponential moving average "
                  f"(`s[0] = val_loss[0]`, `s[i] = {a} * s[i - 1] + {round(1 - a, 2)} * val_loss[i]`) in full 'C0' on top, "
                  f"labelled 'raw' and 'smoothed', with a legend.",
         f"fig, ax = plt.subplots()\nsm = [val_loss[0]]\nfor v in val_loss[1:]:\n    sm.append({a} * sm[-1] + {round(1 - a, 2)} * v)\n"
         f"ax.plot(epochs, val_loss, color='C0', alpha=0.3, label='raw')\nax.plot(epochs, sm, color='C0', label='smoothed')\nax.legend()",
         {"post": "fc.lines(fc.ax(), 'color', 'alpha', 'label'), fc.legend()", "tol": 1e-5}),
        ("drawstyle", "A learning rate that changes once per epoch is a staircase: plot `lr` against `epochs` with "
                      "`drawstyle='steps-post'` and a linewidth of 2, on a log y-axis, with the y label 'learning rate'.",
         "fig, ax = plt.subplots()\nax.plot(epochs, lr, drawstyle='steps-post', linewidth=2)\nax.set_yscale('log')\n"
         "ax.set_ylabel('learning rate')", {"post": "fc.lines(fc.ax(), 'drawstyle', 'lw'), fc.scales(), fc.labels()['ylabel']",
                                             "setup": "lr = np.round(0.1 * 0.5 ** ((epochs - 1) // 5), 6)",
                                             "require": ["drawstyle", "set_yscale"]}),
        ("plot", f"Five seeds of the same run are the columns of `runs` ({n} × 5). Plot all five in **one** call "
                         f"as thin grey lines (`color='0.7'`, `linewidth=0.8`), then their mean across seeds in black "
                         f"with linewidth 2, labelled 'mean' (the only legend entry).",
         "fig, ax = plt.subplots()\nax.plot(epochs, runs, color='0.7', linewidth=0.8)\n"
         "ax.plot(epochs, runs.mean(axis=1), color='k', linewidth=2, label='mean')\nax.legend()",
         {"post": "fc.lines(fc.ax(), 'color', 'lw'), fc.legend()",
          "setup": f"runs = np.round(train_loss[:, None] + 0.1 * np.random.default_rng({s}).normal(size=({n}, 5)), 4)"}),
        ("semilogy", "Plot `train_loss` and `val_loss` on a log y-axis with `ax.semilogy`, labelled 'train' and 'val', "
                     "with a grid and a legend in the upper right.",
         "fig, ax = plt.subplots()\nax.semilogy(epochs, train_loss, label='train')\nax.semilogy(epochs, val_loss, label='val')\n"
         "ax.grid(True)\nax.legend(loc='upper right')",
         {"post": "fc.lines(fc.ax(), 'label'), fc.scales(), fc.grid_on(), fc.legend(), fc.ax().get_legend()._loc",
          "require": ["semilogy", "loc"]}),
        ("markerfacecolor", "Accuracy with open markers: plot `val_acc` against `epochs` as `'o-'` in 'C2' with white "
                            "marker faces (`markerfacecolor='white'`), marker size 5, y limits 0 … 1 and the title "
                            "'validation accuracy'.",
         "fig, ax = plt.subplots()\nax.plot(epochs, val_acc, 'o-', color='C2', markerfacecolor='white', markersize=5)\n"
         "ax.set_ylim(0, 1)\nax.set_title('validation accuracy')",
         {"post": "fc.lines(fc.ax(), 'color', 'mfc', 'ms', 'marker', 'ls'), fc.limits()['ylim'], fc.labels()['title']"}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "medium")
def embedding_scatter(rng):
    code, desc, k, names = classes(rng)
    s = seeded(rng)
    markers = ["o", "s", "^", "D"][:k]
    items = [
        ("legend_elements", "Scatter the embedding in one call colored by class (`c=labels, cmap='tab10'`, `s=15`), "
                            "and add a legend with the **class names** (handles from `sc.legend_elements()`) titled "
                            "'class' in the upper right.",
         "fig, ax = plt.subplots()\nsc = ax.scatter(pts[:, 0], pts[:, 1], c=labels, cmap='tab10', s=15)\n"
         "handles, _ = sc.legend_elements()\nax.legend(handles, class_names, title='class', loc='upper right')",
         {"post": SC + "_sc(None, 0, 'array', 'cmap', 'sizes'), fc.legend(), fc.legend_title()",
          "require": ["legend_elements", "cmap"]}),
        ("marker", f"Distinguish classes by **shape** too (for grayscale printing): class c uses marker "
                   f"`{markers}[c]` and its name as label; add a legend.",
         f"fig, ax = plt.subplots()\nfor c, m in enumerate({markers}):\n    ax.scatter(*pts[labels == c].T, marker=m, label=class_names[c])\n"
         f"ax.legend()", {"post": SC + f"[_sc(None, i, 'marker') for i in range({k})], fc.legend()"}),
        ("edgecolors", "Highlight the points the model got wrong (`pred != labels`): scatter everything colored by "
                       "`labels` with `s=15`, then scatter only the misclassified points on top with `s=60`, no face "
                       "(`facecolors='none'`), red outlines (`edgecolors='red'`) and `linewidths=1.5`.",
         "fig, ax = plt.subplots()\nax.scatter(pts[:, 0], pts[:, 1], c=labels, s=15)\nwrong = pred != labels\n"
         "ax.scatter(pts[wrong, 0], pts[wrong, 1], s=60, facecolors='none', edgecolors='red', linewidths=1.5)",
         {"post": SC + "_sc(None, 0, 'array', 'sizes'), _sc(None, 1, 'sizes', 'edgecolor', 'color', 'lw')",
          "setup": f"pred = np.where(np.random.default_rng({s}).random(len(labels)) < 0.15, (labels + 1) % {k}, labels)",
          "require": ["edgecolors", "linewidths"]}),
        ("zorder", "Draw the class centroids on top of the points: scatter all points small and faint (`s=10`, "
                   "`alpha=0.4`), then each class mean as one big black 'X' (`marker='X'`, `s=200`, `color='k'`, "
                   "`zorder=3`), all centroids in a single scatter call (in class order).",
         f"fig, ax = plt.subplots()\nax.scatter(pts[:, 0], pts[:, 1], s=10, alpha=0.4)\n"
         f"cent = np.array([pts[labels == c].mean(axis=0) for c in range({k})])\n"
         f"ax.scatter(cent[:, 0], cent[:, 1], marker='X', s=200, color='k', zorder=3)",
         {"post": SC + "_sc(None, 0, 'sizes', 'alpha'), _sc(None, 1, 'sizes', 'color', 'marker'), "
                       "[c.get_zorder() for c in fc.ax().collections]", "require": ["zorder", "marker"]}),
        ("s", "Size points by their distance from the overall mean (`d = np.hypot(*(pts - pts.mean(axis=0)).T)`): "
              "`s = 10 + 20 * d`, colored by class with `cmap='tab10'`.",
         "fig, ax = plt.subplots()\nd = np.hypot(*(pts - pts.mean(axis=0)).T)\nax.scatter(pts[:, 0], pts[:, 1], s=10 + 20 * d, c=labels, cmap='tab10')",
         {"post": SC + "_sc(None, 0, 'sizes', 'array', 'cmap')", "tol": 1e-4}),
        ("legend", "Per-class scatters (label = class name, `s=12`) and **one more** entry for the centroids: scatter "
                   "all class means in one call with `marker='*'`, `s=150`, `color='k'`, `label='centroid'`; the "
                   "legend lists the classes then 'centroid'.",
         f"fig, ax = plt.subplots()\nfor c in range({k}):\n    ax.scatter(*pts[labels == c].T, s=12, label=class_names[c])\n"
         f"cent = np.array([pts[labels == c].mean(axis=0) for c in range({k})])\n"
         f"ax.scatter(cent[:, 0], cent[:, 1], marker='*', s=150, color='k', label='centroid')\nax.legend()",
         {"post": SC + "_all(), fc.legend()"}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "medium")
def continuous_color(rng):
    code, desc, n = _pts03(rng)
    cm = rng.choice(["viridis", "plasma", "magma"])
    lab = rng.choice(["value", "score", "x + y"])
    items = [
        ("colorbar", f"Drill: scatter `py` against `px` colored by `v` with the colormap {cm!r}, `s=20`, and a colorbar "
                     f"labelled {lab!r}.",
         f"fig, ax = plt.subplots()\nsc = ax.scatter(px, py, c=v, cmap={cm!r}, s=20)\nfig.colorbar(sc, ax=ax, label={lab!r})",
         {"post": SC + "_sc(fc.ax(0), 0, 'array', 'cmap', 'sizes'), fc.colorbars()"}),
        ("vmin", "`v` is signed, so use a **diverging** colormap centred on 0: `cmap='coolwarm'` with `vmin=-m`, "
                 "`vmax=m` where `m = np.abs(v).max()`; add a colorbar.",
         "fig, ax = plt.subplots()\nm = np.abs(v).max()\nsc = ax.scatter(px, py, c=v, cmap='coolwarm', vmin=-m, vmax=m)\n"
         "fig.colorbar(sc, ax=ax)", {"post": SC + "_sc(fc.ax(0), 0, 'cmap', 'clim'), len(fc.colorbars())",
                                     "require": ["vmin", "vmax", "abs"]}),
        ("c", f"Color by **order** to show a trajectory: treat the points as a walk (`wx = px.cumsum()`, `wy = "
              f"py.cumsum()`), draw it as a thin grey line, and scatter the points on top colored by their index "
              f"(`c=np.arange(len(px))`, {cm!r}) with a colorbar labelled 'step'.",
         f"fig, ax = plt.subplots()\nwx, wy = px.cumsum(), py.cumsum()\nax.plot(wx, wy, color='0.7', linewidth=0.8)\n"
         f"sc = ax.scatter(wx, wy, c=np.arange(len(px)), cmap={cm!r})\nfig.colorbar(sc, ax=ax, label='step')",
         {"post": SC + "fc.lines(fc.ax(0), 'color'), _sc(fc.ax(0), 0, 'array', 'cmap'), fc.colorbars()", "require": ["cumsum", "colorbar"]}),
        ("s", "Encode `v` twice, as size **and** color: `s=20 + 30 * np.abs(v)`, `c=v`, with black outlines "
              "(`edgecolors='k'`, `linewidths=0.3`) and `alpha=0.7`, like the guide's example.",
         "fig, ax = plt.subplots()\nax.scatter(px, py, c=v, s=20 + 30 * np.abs(v), alpha=0.7, edgecolors='k', linewidths=0.3)",
         {"post": SC + "_sc(None, 0, 'array', 'sizes', 'alpha', 'edgecolor', 'lw')", "tol": 1e-4}),
        ("colorbar", "Two panels, one color scale: on `fig, (a, b) = plt.subplots(1, 2)` scatter the points with `px < 0` "
                     "on `a` and the others on `b`, both colored by `v` with the same `vmin=v.min()` and `vmax=v.max()`, "
                     "and add **one** colorbar for both panels (`fig.colorbar(sc, ax=[a, b])`).",
         "fig, (a, b) = plt.subplots(1, 2)\nm = px < 0\nkw = dict(vmin=v.min(), vmax=v.max())\n"
         "a.scatter(px[m], py[m], c=v[m], **kw)\nsc = b.scatter(px[~m], py[~m], c=v[~m], **kw)\nfig.colorbar(sc, ax=[a, b])",
         {"post": SC + "[_sc(a, 0, 'array', 'clim') for a in fc.axes()], len(fc.colorbars())", "require": ["vmin", "vmax", "colorbar"]}),
        ("cmap", "Show which points are above the median of `v`: scatter them colored by the boolean `v > np.median(v)` "
                 "converted to int with `cmap='bwr'`, and give the figure a legend from `sc.legend_elements()` with "
                 "the labels 'low' and 'high'.",
         "fig, ax = plt.subplots()\nsc = ax.scatter(px, py, c=(v > np.median(v)).astype(int), cmap='bwr')\n"
         "handles, _ = sc.legend_elements()\nax.legend(handles, ['low', 'high'])",
         {"post": SC + "_sc(None, 0, 'array', 'cmap'), fc.legend()", "require": ["cmap", "legend_elements"]}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "medium")
def trig_drill(rng):
    n = rng.choice([100, 200, 300])
    lim = rng.choice([2, 3, 4])
    styles = rng.choice([("-", "--", ":"), ("-", "-.", "--"), (":", "-", "--")])
    k = rng.randint(3, 6)
    setup = f"x = np.linspace(-2 * np.pi, 2 * np.pi, {n})"
    s1, s2, s3 = styles
    items = [
        ("legend", f"Drill: plot `sin`, `cos` and `tan` of `x` (tan clipped to ±{lim} with `np.clip`) with the line "
                   f"styles {s1!r}, {s2!r}, {s3!r}, labelled 'sin', 'cos', 'tan', and add a legend.",
         f"fig, ax = plt.subplots()\nax.plot(x, np.sin(x), linestyle={s1!r}, label='sin')\n"
         f"ax.plot(x, np.cos(x), linestyle={s2!r}, label='cos')\nax.plot(x, np.clip(np.tan(x), -{lim}, {lim}), linestyle={s3!r}, label='tan')\n"
         f"ax.legend()", {"post": "fc.lines(fc.ax(), 'ls', 'label'), fc.legend()", "require": ["clip", "legend"]}),
        ("nan", f"Clipping draws ugly vertical jumps at tan's poles. Instead **break** the line there: copy `np.tan(x)` "
                f"and set every value with `|tan| > {lim}` to `np.nan` (NaNs leave gaps). Plot it with sin and cos "
                f"(labels 'sin', 'cos', 'tan'), y limits ±{lim}, and a legend.",
         f"fig, ax = plt.subplots()\nt = np.tan(x)\nt[np.abs(t) > {lim}] = np.nan\nax.plot(x, np.sin(x), label='sin')\n"
         f"ax.plot(x, np.cos(x), label='cos')\nax.plot(x, t, label='tan')\nax.set_ylim(-{lim}, {lim})\nax.legend()",
         {"post": "fc.lines(fc.ax(), 'label'), fc.limits()['ylim'], fc.legend()", "require": ["nan"]}),
        ("plot", f"A family of {k} phase-shifted sines `np.sin(x + p)` for `p` in `np.linspace(0, np.pi, {k})`, all in "
                 f"'C0' with alpha growing linearly from 0.2 (first) to 1.0 (last).",
         f"fig, ax = plt.subplots()\nfor p, a in zip(np.linspace(0, np.pi, {k}), np.linspace(0.2, 1.0, {k})):\n"
         f"    ax.plot(x, np.sin(x + p), color='C0', alpha=a)",
         {"post": "fc.lines(fc.ax(), 'color', 'alpha')", "tol": 1e-5}),
        ("legend", "Plot sin, cos and `sin * cos` in **one** call (`ax.plot(x, np.column_stack([...]))`) and give the "
                   "legend the labels afterwards: `ax.legend(['sin', 'cos', 'sin·cos'])`.",
         "fig, ax = plt.subplots()\nax.plot(x, np.column_stack([np.sin(x), np.cos(x), np.sin(x) * np.cos(x)]))\n"
         "ax.legend(['sin', 'cos', 'sin·cos'])", {"post": "fc.lines(fc.ax()), fc.legend()", "require": ["column_stack", "legend"]}),
        ("linewidth", f"Style by importance: `np.sin(x)` thick (linewidth 3) in 'k', `np.sin(2 * x)` medium (1.5) in "
                      f"'C1' dashed, `np.sin(3 * x)` thin (0.8) in 'C2' dotted; labels 'f1', 'f2', 'f3' and a legend in "
                      f"the lower left.",
         "fig, ax = plt.subplots()\nax.plot(x, np.sin(x), color='k', linewidth=3, label='f1')\n"
         "ax.plot(x, np.sin(2 * x), '--', color='C1', linewidth=1.5, label='f2')\n"
         "ax.plot(x, np.sin(3 * x), ':', color='C2', linewidth=0.8, label='f3')\nax.legend(loc='lower left')",
         {"post": "fc.lines(fc.ax(), 'color', 'ls', 'lw', 'label'), fc.legend(), fc.ax().get_legend()._loc"}),
    ]
    return catalog(rng, items, setup, f"`x` holds {n} angles from -2π to 2π.")


@template(M, "medium")
def log_scale_plots(rng):
    s = seeded(rng)
    a, b = rng.choice([(5.0, 0.3), (8.0, 0.25), (3.0, 0.4), (10.0, 0.35)])
    k = rng.randint(8, 14)
    items = [
        ("semilogx", "A learning-rate range test: `lrs` (log-spaced) and the loss after a few steps at each, `lr_loss`. "
                     "Plot them with a log x-axis (`ax.semilogx`), mark the lowest loss with a red circle (`'ro'`) and "
                     "return the learning rate at that minimum.",
         "fig, ax = plt.subplots()\nax.semilogx(lrs, lr_loss)\ni = lr_loss.argmin()\nax.plot(lrs[i], lr_loss[i], 'ro')\n"
         "result = lrs[i]", {"post": "_, fc.lines(fc.ax(), 'color', 'marker'), fc.scales()", "require": ["semilogx", "argmin"]}),
        ("loglog", f"Scaling law: `n_params` and the test `loss` of {k} models follow `loss ≈ a · N^(-b)`. Plot them as "
                   f"markers (`'o'`) on log–log axes, fit a straight line in log space "
                   f"(`np.polyfit(np.log10(n_params), np.log10(loss), 1)`), draw the fitted power law as a dashed "
                   f"line over the same `n_params`, and return the fitted exponent b (minus the slope).",
         "fig, ax = plt.subplots()\nax.loglog(n_params, loss, 'o')\nslope, icpt = np.polyfit(np.log10(n_params), np.log10(loss), 1)\n"
         "ax.loglog(n_params, 10 ** icpt * n_params ** slope, '--')\nresult = -slope",
         {"post": "_, fc.lines(fc.ax(), 'ls', 'marker', nd=4), fc.scales()", "require": ["loglog", "polyfit"], "tol": 1e-6}),
        ("semilogy", "Plot the **gap** to the final loss, `lr_loss - lr_loss.min() + 1e-3`, against `np.arange(len(lr_loss))` "
                     "on a log y-axis with `ax.semilogy`, as a line with square markers (`'s-'`).",
         "fig, ax = plt.subplots()\nax.semilogy(np.arange(len(lr_loss)), lr_loss - lr_loss.min() + 1e-3, 's-')",
         {"post": "fc.lines(fc.ax(), 'marker', nd=6), fc.scales()"}),
        ("loglog", "The guide's three log flavours side by side on `plt.subplots(1, 3, figsize=(10, 3))` with `t = "
                   "np.linspace(0, 10, 50)`: `semilogy(t, np.exp(t))`, `semilogx(t + 1, t)` and `loglog(t + 1, (t + 1) ** 2)`.",
         "t = np.linspace(0, 10, 50)\nfig, axs = plt.subplots(1, 3, figsize=(10, 3))\naxs[0].semilogy(t, np.exp(t))\n"
         "axs[1].semilogx(t + 1, t)\naxs[2].loglog(t + 1, (t + 1) ** 2)",
         {"post": "[fc.scales(a) for a in fc.axes()], " + YDATA + ", fc.size()", "require": ["semilogy", "semilogx", "loglog", "exp"]}),
        ("set_yscale", "Gradient norms span many orders of magnitude **and** cross zero (they're signed here: `gn`). A "
                       "log scale can't show negatives, but `ax.set_yscale('symlog', linthresh=1e-3)` can. Plot `gn` and "
                       "set that scale.",
         "fig, ax = plt.subplots()\nax.plot(gn)\nax.set_yscale('symlog', linthresh=1e-3)",
         {"post": "fc.lines(fc.ax()), fc.scales(), fc.ax().yaxis.get_transform().linthresh", "require": ["set_yscale", "linthresh"]}),
    ]
    setup = (f"g = np.random.default_rng({s})\nlrs = np.logspace(-5, 0, 30)\n"
             f"lr_loss = np.round(2.3 - 1.5 * np.exp(-(np.log10(lrs) + 2.5) ** 2) + np.where(lrs > 0.1, 3 * (lrs - 0.1), 0) "
             f"+ 0.02 * g.random(30), 4)\nn_params = np.round(np.logspace(5, 9, {k}))\n"
             f"loss = np.round({a} * n_params ** -{b} * np.exp(0.05 * g.normal(size={k})), 5)\n"
             f"gn = np.round(g.normal(size=60) * 10.0 ** g.uniform(-4, 1, 60), 6)")
    return catalog(rng, items, setup, f"`lrs` and `lr_loss` come from a learning-rate range test; `n_params` / `loss` "
                                      f"describe {k} models of growing size; `gn` holds 60 signed gradient norms.")


@template(M, "medium")
def prediction_plots(rng):
    s = seeded(rng)
    n = rng.randint(40, 90)
    setup = (f"g = np.random.default_rng({s})\ny_true = np.round(g.uniform(0, 10, {n}), 3)\n"
             f"y_pred = np.round(y_true + g.normal(0, 0.8, {n}), 3)\nprob = np.round(np.sort(g.random(10)), 3)\n"
             f"frac = np.round(np.clip(prob + g.normal(0, 0.08, 10), 0, 1), 3)\nscores = np.round(g.random({n}), 3)\n"
             f"y_bin = (g.random({n}) < scores).astype(int)")
    items = [
        ("scatter", "Parity plot: scatter `y_pred` against `y_true` (`s=15`), add the identity line from the smallest "
                    "to the largest value of both arrays as a black dashed line (`'k--'`), and label the axes 'true' "
                    "and 'predicted'.",
         "fig, ax = plt.subplots()\nax.scatter(y_true, y_pred, s=15)\nlo = min(y_true.min(), y_pred.min())\n"
         "hi = max(y_true.max(), y_pred.max())\nax.plot([lo, hi], [lo, hi], 'k--')\nax.set(xlabel='true', ylabel='predicted')",
         {"post": SC + "_sc(None, 0, 'sizes'), fc.lines(fc.ax(), 'color', 'ls'), fc.labels()"}),
        ("scatter", "Residual plot: scatter the residuals `y_pred - y_true` against `y_pred`, and draw a horizontal zero "
                    "line across the data's x-range with `ax.plot([y_pred.min(), y_pred.max()], [0, 0], color='k')`.",
         "fig, ax = plt.subplots()\nax.scatter(y_pred, y_pred - y_true)\nax.plot([y_pred.min(), y_pred.max()], [0, 0], color='k')",
         {"post": SC + "_sc(), fc.lines(fc.ax(), 'color')"}),
        ("colorbar", "Parity plot colored by the **absolute error** (`c=np.abs(y_pred - y_true)`, `cmap='Reds'`) with a "
                     "colorbar labelled 'abs error'.",
         "fig, ax = plt.subplots()\nsc = ax.scatter(y_true, y_pred, c=np.abs(y_pred - y_true), cmap='Reds')\n"
         "fig.colorbar(sc, ax=ax, label='abs error')", {"post": SC + "_sc(fc.ax(0), 0, 'array', 'cmap'), fc.colorbars()"}),
        ("plot", "Calibration (reliability) curve: plot `frac` (observed frequency) against `prob` (mean predicted "
                 "probability) as `'o-'` labelled 'model', and the perfectly calibrated diagonal from (0, 0) to (1, 1) "
                 "as a grey dotted line (`':'`, `color='gray'`) labelled 'ideal'; legend in the upper left.",
         "fig, ax = plt.subplots()\nax.plot(prob, frac, 'o-', label='model')\nax.plot([0, 1], [0, 1], ':', color='gray', label='ideal')\n"
         "ax.legend(loc='upper left')", {"post": "fc.lines(fc.ax(), 'ls', 'marker', 'color', 'label'), fc.legend()"}),
        ("drawstyle", "Plot the sorted absolute errors as an empirical CDF by hand: `e = np.sort(np.abs(y_pred - y_true))` "
                      "against `np.arange(1, len(e) + 1) / len(e)` with `drawstyle='steps-post'`, x label 'abs error'.",
         "fig, ax = plt.subplots()\ne = np.sort(np.abs(y_pred - y_true))\nax.plot(e, np.arange(1, len(e) + 1) / len(e), drawstyle='steps-post')\n"
         "ax.set_xlabel('abs error')", {"post": "fc.lines(fc.ax(), 'drawstyle'), fc.labels()['xlabel']"}),
        ("label", "Plot `y_true` and `y_pred` against the sample index sorted by `y_true` (`order = np.argsort(y_true)`): "
                  "the truth as a black line labelled 'true', predictions as small orange dots (`'.'`, `color='C1'`) "
                  "labelled 'pred'; legend.",
         "fig, ax = plt.subplots()\norder = np.argsort(y_true)\nax.plot(y_true[order], color='k', label='true')\n"
         "ax.plot(y_pred[order], '.', color='C1', label='pred')\nax.legend()",
         {"post": "fc.lines(fc.ax(), 'color', 'marker', 'ls', 'label'), fc.legend()"}),
        ("drawstyle", "Precision–recall staircase for a binary classifier: sort the samples by `scores`, highest first; "
                      "after the top k samples, precision = (positives among them) / k and recall = (positives among "
                      "them) / (all positives) (`y_bin` holds the 0/1 truth). Plot precision against recall with "
                      "`drawstyle='steps-post'` and label the axes 'recall' and 'precision'.",
         "order = np.argsort(-scores, kind='stable')\ntp = np.cumsum(y_bin[order])\nprecision = tp / np.arange(1, len(tp) + 1)\n"
         "recall = tp / y_bin.sum()\nfig, ax = plt.subplots()\nax.plot(recall, precision, drawstyle='steps-post')\n"
         "ax.set(xlabel='recall', ylabel='precision')",
         {"post": "fc.lines(fc.ax(), 'drawstyle'), fc.labels()", "require": ["drawstyle", "cumsum"]}),
    ]
    return catalog(rng, items, setup, f"`y_true` / `y_pred` are {n} regression targets and predictions; `prob` / `frac` "
                                      f"are 10 calibration bins (mean predicted probability, observed frequency); `scores` / "
                                      f"`y_bin` are a classifier's scores and the 0/1 truth for {n} samples.")


@template(M, "medium")
def spec_lines(rng):
    code, desc, n = curves(rng)
    k = rng.randint(4, 7)
    hi = rng.randrange(k)
    s = seeded(rng)
    spec = rng.sample([("SGD", "C0", "-", "o"), ("Adam", "C1", "--", "s"), ("AdamW", "C2", "-.", "^"),
                       ("RMSprop", "C3", ":", "D"), ("Lion", "k", "-", "x")], 3)
    rows = "\n".join(f"| {a} | {c} | `{l}` | `{m}` |" for a, c, l, m in spec)
    items = [
        ("plot", f"Plot the three optimizers' curves (`curves[name]`, against `epochs`) exactly as this style table "
                 f"says, each labelled with its name, markers every 5 epochs, and add a legend:\n\n| optimizer | color | "
                 f"linestyle | marker |\n|---|---|---|---|\n{rows}",
         "fig, ax = plt.subplots()\n" + "".join(f"ax.plot(epochs, curves[{a!r}], color={c!r}, linestyle={l!r}, marker={m!r}, "
                                                f"markevery=5, label={a!r})\n" for a, c, l, m in spec) + "ax.legend()",
         {"post": "fc.lines(fc.ax(), 'color', 'ls', 'marker', 'markevery', 'label'), fc.legend()",
          "setup": "curves = {" + ", ".join(f"{a!r}: np.round(train_loss * {1 + 0.1 * i:.1f}, 4)" for i, (a, *_) in enumerate(spec)) + "}"}),
        ("zorder", f"Highlight one run among {k}: plot every row of `runs` against `epochs` in light grey "
                   f"(`color='0.75'`, linewidth 1), except run {hi}, which is drawn in 'C3' with linewidth 2.5 and "
                   f"`zorder=3` so it stays on top.",
         f"fig, ax = plt.subplots()\nfor i, r in enumerate(runs):\n    if i == {hi}:\n        ax.plot(epochs, r, color='C3', linewidth=2.5, zorder=3)\n"
         f"    else:\n        ax.plot(epochs, r, color='0.75', linewidth=1)",
         {"post": "fc.lines(fc.ax(), 'color', 'lw', 'zorder')",
          "setup": f"runs = np.round(train_loss + 0.1 * np.random.default_rng({s}).normal(size=({k}, {n})), 4)"}),
        ("markevery", "Mark only specific epochs: plot `val_loss` with circle markers at the epochs where it reaches a "
                      "new minimum (`markevery=` a list of indices: i where `val_loss[i] < val_loss[:i].min()`, and i = 0).",
         "fig, ax = plt.subplots()\nidx = [0] + [i for i in range(1, len(val_loss)) if val_loss[i] < val_loss[:i].min()]\n"
         "ax.plot(epochs, val_loss, marker='o', markevery=idx)", {"post": "fc.lines(fc.ax(), 'marker', 'markevery')"}),
        ("linestyle", "A custom dash pattern: plot `val_loss` with `linestyle=(0, (5, 1, 1, 1))` (5-point dash, 1 gap, "
                      "1 dot, 1 gap) and `train_loss` with the named style 'dashdot'.",
         "fig, ax = plt.subplots()\nax.plot(epochs, val_loss, linestyle=(0, (5, 1, 1, 1)))\nax.plot(epochs, train_loss, linestyle='dashdot')",
         {"post": "fc.lines(fc.ax()), [l._unscaled_dash_pattern for l in fc.ax().get_lines()]", "require": ["linestyle|ls"]}),
        ("label", "Labels that start with an underscore are left out of the legend. Plot `train_loss` labelled 'train', "
                  "`val_loss` labelled 'val', and a horizontal line at the minimum of `val_loss` (two points spanning "
                  "the epochs) labelled `'_best'`; add a legend (it must show only 'train' and 'val').",
         "fig, ax = plt.subplots()\nax.plot(epochs, train_loss, label='train')\nax.plot(epochs, val_loss, label='val')\n"
         "ax.plot([epochs[0], epochs[-1]], [val_loss.min()] * 2, label='_best')\nax.legend()",
         {"post": "fc.lines(fc.ax(), 'label'), fc.legend()"}),
    ]
    return catalog(rng, items, code, f"{desc}.")


# ================================================================= hard

@template(M, "hard")
def plot_helpers(rng):
    code, desc, n = curves(rng)
    col = rng.choice(["C0", "C2", "tab:purple"])
    a = rng.choice([0.6, 0.8, 0.9])
    items = [
        ("plot", f"Write `plot_runs(ax, runs, label=None, color={col!r})`: `runs` is a 2-D array (one run per **row**); "
                 f"draw every run as a thin line (linewidth 0.8, `alpha=0.3`) and the mean over runs as a thick line "
                 f"(linewidth 2) labelled `label`, all in `color`; x is `np.arange(runs.shape[1])`. Return the mean's "
                 f"Line2D.",
         f"def plot_runs(ax, runs, label=None, color={col!r}):\n    t = np.arange(runs.shape[1])\n    for r in runs:\n"
         f"        ax.plot(t, r, color=color, linewidth=0.8, alpha=0.3)\n    line, = ax.plot(t, runs.mean(axis=0), color=color, linewidth=2, label=label)\n"
         f"    return line",
         {"post": "_f, _a = plt.subplots()\n_l = plot_runs(_a, np.stack([train_loss, val_loss, train_loss * 1.1]), label='mean')\n"
                  "_m = plot_runs(_a, np.stack([val_loss, val_loss + 0.1]), color='k')\n"
                  "(fc.lines(_a, 'color', 'lw', 'alpha', 'label'), _l is _a.get_lines()[3], _m.get_label())"}),
        ("gca", "Write `class_scatter(pts, labels, names=None, ax=None)`: if `ax` is None use the current Axes "
                "(`plt.gca()`); scatter each class (in label order 0, 1, …) in its own call, labelled `names[c]` if "
                "given, else `f'class {c}'`; add a legend; return the Axes.",
         "def class_scatter(pts, labels, names=None, ax=None):\n    if ax is None:\n        ax = plt.gca()\n"
         "    for c in np.unique(labels):\n        ax.scatter(*pts[labels == c].T, label=names[c] if names else f'class {c}')\n"
         "    ax.legend()\n    return ax",
         {"post": SC + "_g = np.random.default_rng(0)\n_p = np.round(_g.normal(size=(12, 2)), 3)\n_y = np.arange(12) % 3\n"
                       "_f, (_a, _b) = plt.subplots(1, 2)\nplt.sca(_a)\n_r = class_scatter(_p, _y)\n"
                       "_s = class_scatter(_p, _y, names=['x', 'y', 'z'], ax=_b)\n(_r is _a, _s is _b, _all(_a), fc.legend(_a), fc.legend(_b))",
          "require": ["gca", "scatter", "legend"]}),
        ("set_alpha", "Write `highlight(ax, i)`: make line `i` of `ax` stand out (linewidth 3, `zorder=3`, alpha 1) and "
                      "fade every other line (alpha 0.3, linewidth 1, the default `zorder=2`). Calling it again with another `i` must move the "
                      "highlight (so set **every** line's properties each time).",
         "def highlight(ax, i):\n    for j, line in enumerate(ax.get_lines()):\n        if j == i:\n"
         "            line.set(linewidth=3, zorder=3, alpha=1)\n        else:\n            line.set(linewidth=1, zorder=2, alpha=0.3)",
         {"post": "_f, _a = plt.subplots()\nfor _y in (train_loss, val_loss, train_loss / 2):\n    _a.plot(epochs, _y)\n"
                  "highlight(_a, 0)\nhighlight(_a, 2)\nfc.lines(_a, 'lw', 'alpha', data=False), "
                  "[l.get_zorder() > m.get_zorder() for l, m in zip(_a.get_lines()[2:], _a.get_lines()[:2])]", "require": []}),
        ("set_ydata", "Write `update_line(line, y)` for live plots: replace the line's y data (same length), make its "
                      "Axes recompute the data limits and rescale (`relim`, `autoscale_view`), and return the new y "
                      "limits as a tuple.",
         "def update_line(line, y):\n    line.set_ydata(y)\n    ax = line.axes\n    ax.relim()\n    ax.autoscale_view()\n"
         "    return ax.get_ylim()",
         {"post": "_f, _a = plt.subplots()\n_l, = _a.plot(epochs, train_loss)\n_r = update_line(_l, val_loss * 10)\n"
                  "(fc.r(_r, 6), fc.lines(_a))", "require": ["set_ydata", "relim", "autoscale_view"]}),
        ("colorbar", "Write `scatter_colored(x, y, v, ax=None, label='value', cmap='viridis')`: scatter colored by `v` "
                     "(current Axes when `ax` is None), add a colorbar with that label next to that Axes "
                     "(`ax.figure.colorbar(sc, ax=ax, label=label)`) and return `(sc, cbar)`.",
         "def scatter_colored(x, y, v, ax=None, label='value', cmap='viridis'):\n    if ax is None:\n        ax = plt.gca()\n"
         "    sc = ax.scatter(x, y, c=v, cmap=cmap)\n    cbar = ax.figure.colorbar(sc, ax=ax, label=label)\n    return sc, cbar",
         {"post": SC + "_f, _a = plt.subplots()\n_s, _c = scatter_colored(epochs, train_loss, val_loss, ax=_a, label='val', cmap='magma')\n"
                       "(_sc(_a, 0, 'array', 'cmap'), fc.colorbars(_f), _c.ax.get_ylabel())", "require": ["colorbar", "scatter"]}),
        ("plot", f"Write `plot_smoothed(ax, y, alpha={a})`: draw `y` faintly (alpha 0.3) and its exponential moving "
                 f"average (`s[0] = y[0]`, `s[i] = alpha · s[i-1] + (1 - alpha) · y[i]`) on top **in the same color** "
                 f"(the faint line's color, `raw.get_color()`), and return `(raw, smooth)` lines.",
         f"def plot_smoothed(ax, y, alpha={a}):\n    s = [y[0]]\n    for v in y[1:]:\n        s.append(alpha * s[-1] + (1 - alpha) * v)\n"
         f"    raw, = ax.plot(y, alpha=0.3)\n    smooth, = ax.plot(s, color=raw.get_color())\n    return raw, smooth",
         {"post": "_f, _a = plt.subplots()\n_r1 = plot_smoothed(_a, val_loss)\n_r2 = plot_smoothed(_a, train_loss, alpha=0.5)\n"
                  "fc.lines(_a, 'color', 'alpha', nd=5)", "require": ["get_color"]}),
        ("get_xlim", "Write `add_identity(ax, **kwargs)` for parity plots: draw the line y = x across the part of the view "
                     "where both axes overlap (from the larger of the two lower limits to the smaller of the two upper "
                     "limits, read from `ax.get_xlim()` / `ax.get_ylim()`), pass `kwargs` on to `ax.plot`, keep the "
                     "limits as they were (`ax.set_xlim` / `ax.set_ylim` afterwards), and return the line.",
         "def add_identity(ax, **kwargs):\n    x0, x1 = ax.get_xlim()\n    y0, y1 = ax.get_ylim()\n    lo, hi = max(x0, y0), min(x1, y1)\n"
         "    line, = ax.plot([lo, hi], [lo, hi], **kwargs)\n    ax.set_xlim(x0, x1)\n    ax.set_ylim(y0, y1)\n    return line",
         {"post": "_f, _a = plt.subplots()\n_a.scatter(train_loss, val_loss)\n_l = add_identity(_a, color='k', ls='--')\n"
                  "(fc.lines(_a, 'color', 'ls', nd=6), fc.limits(_a), _l in _a.get_lines())", "require": ["get_xlim", "get_ylim"]}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "hard")
def debug_lines(rng):
    code, desc, k, names = classes(rng, k=(3, 4))
    s = seeded(rng)
    extra = f"x = np.linspace(0, 10, 40)\ny = np.round(np.sin(x) + 0.1 * np.random.default_rng({s}).normal(size=40), 4)"
    items = [
        (".plot", "This crashes with *'list' object has no attribute 'set_color'*:\n```python\nfig, ax = plt.subplots()\n"
                  "line = ax.plot(x, y)\nline.set_color('red')\n```\n`ax.plot` returns a **list** of lines. Fix it.",
         "fig, ax = plt.subplots()\nline, = ax.plot(x, y)\nline.set_color('red')", {"post": "fc.lines(fc.ax(), 'color')"}),
        ("c", "This raises *Invalid RGBA argument* (or colors everything wrongly): the class labels were passed as "
              "**colors**:\n```python\nfig, ax = plt.subplots()\nax.scatter(pts[:, 0], pts[:, 1], color=labels, cmap='tab10')\n```\n"
              "Numbers to be mapped through a colormap go in `c=`. Fix it.",
         "fig, ax = plt.subplots()\nax.scatter(pts[:, 0], pts[:, 1], c=labels, cmap='tab10')",
         {"post": SC + "_sc(None, 0, 'array', 'cmap')", "require": ["c"]}),
        ("plot", "The line comes out **blue**, but the author wanted red dashes (and gets a warning that the color is "
                 "redundantly defined):\n```python\nfig, ax = plt.subplots()\nax.plot(x, y, 'r--', color='blue')\n```\n"
                 "Fix it so the line is red and dashed.",
         "fig, ax = plt.subplots()\nax.plot(x, y, 'r--')", {"post": "fc.lines(fc.ax(), 'color', 'ls')"}),
        ("label", "The legend shows 'runs' three times; each column should have its own entry 'seed 0', 'seed 1', "
                  "'seed 2':\n```python\nfig, ax = plt.subplots()\nax.plot(x, np.column_stack([y, y + 1, y + 2]), label='runs')\n"
                  "ax.legend()\n```\nFix it (still one `plot` call: `label=` also accepts one label per column).",
         "fig, ax = plt.subplots()\nax.plot(x, np.column_stack([y, y + 1, y + 2]), label=['seed 0', 'seed 1', 'seed 2'])\nax.legend()",
         {"post": "fc.lines(fc.ax(), 'label'), fc.legend()", "require": ["column_stack", "label"]}),
        ("loglog", "On log axes, the first point vanishes and Matplotlib complains about non-positive values, because "
                   "`x` starts at 0:\n```python\nfig, ax = plt.subplots()\nax.loglog(x, np.exp(y) * x)\n```\nPlot only "
                   "the points with `x > 0` instead.",
         "fig, ax = plt.subplots()\nm = x > 0\nax.loglog(x[m], np.exp(y[m]) * x[m])", {"post": "fc.lines(fc.ax()), fc.scales()"}),
        ("s", "The author wanted markers of **radius** 2, 4, 6 … points (proportional to the class), but `s` is an "
              "**area** in points²:\n```python\nfig, ax = plt.subplots()\nax.scatter(pts[:, 0], pts[:, 1], s=2 * (labels + 1))\n```\n"
              "Fix the sizes (square the radius).",
         "fig, ax = plt.subplots()\nax.scatter(pts[:, 0], pts[:, 1], s=(2 * (labels + 1)) ** 2)", {"post": SC + "_sc(None, 0, 'sizes')"}),
        ("set_data", "Updating a line with data of a **different length** fails at draw time (*x and y must have same "
                     "first dimension*):\n```python\nfig, ax = plt.subplots()\nline, = ax.plot(x, y)\nline.set_ydata(y[:20])\n"
                     "fig.canvas.draw()\n```\nShow only the first 20 points by updating x **and** y together.",
         "fig, ax = plt.subplots()\nline, = ax.plot(x, y)\nline.set_data(x[:20], y[:20])\nfig.canvas.draw()",
         {"post": "fc.lines(fc.ax())", "require": ["set_data"]}),
        ("colorbar", "This crashes: a colorbar needs a **mappable** (something colored through a colormap), and a line "
                     "isn't one:\n```python\nfig, ax = plt.subplots()\nline, = ax.plot(x, y)\nfig.colorbar(line, ax=ax)\n```\n"
                     "Show the points colored by `y` instead: draw the line in light grey, scatter the points with "
                     "`c=y`, and give the scatter the colorbar.",
         "fig, ax = plt.subplots()\nax.plot(x, y, color='lightgray')\nsc = ax.scatter(x, y, c=y)\nfig.colorbar(sc, ax=ax)",
         {"post": SC + "fc.lines(fc.ax(0), 'color'), _sc(fc.ax(0), 0, 'array'), len(fc.colorbars())", "require": ["colorbar", "scatter"]}),
        ("legend_elements", "The legend shows numbers ('0', '1', …) instead of class names:\n```python\nfig, ax = plt.subplots()\n"
                            "sc = ax.scatter(pts[:, 0], pts[:, 1], c=labels)\nax.legend(*sc.legend_elements())\n```\nKeep "
                            "the handles but use `class_names` as the labels.",
         "fig, ax = plt.subplots()\nsc = ax.scatter(pts[:, 0], pts[:, 1], c=labels)\nhandles, _ = sc.legend_elements()\n"
         "ax.legend(handles, class_names)", {"post": "fc.legend()", "require": ["legend_elements"]}),
        ("relim", "After updating the data, the curve runs off the top of the plot: the view limits were computed for "
                  "the **old** data and don't follow changes made through the handle.\n```python\nfig, ax = plt.subplots()\n"
                  "line, = ax.plot(x, y)\nline.set_ydata(y * 100)\n```\nFix it so the y-axis fits the new data again, and "
                  "return the new y limits.",
         "fig, ax = plt.subplots()\nline, = ax.plot(x, y)\nline.set_ydata(y * 100)\nax.relim()\nax.autoscale_view()\n"
         "result = list(ax.get_ylim())", {"post": "_, fc.lines(fc.ax())", "require": ["relim", "autoscale_view"]}),
    ]
    return catalog(rng, items, f"{code}\n{extra}", f"{desc}; `x` and `y` are a noisy sine wave (40 points).")


@template(M, "hard")
def style_drills(rng):
    s = seeded(rng)
    n = rng.randint(100, 200)
    k = rng.choice([3, 3, 4])
    lim = rng.choice([2, 3])
    setup = (f"g = np.random.default_rng({s})\nx = np.linspace(0, 10, 50)\npts = g.normal(size=({n}, 2))\n"
             f"vals = pts[:, 0] + pts[:, 1]\nclasses = g.integers(0, {k}, {n})")
    items = [
        ("plot", f"Drill 1: sin, cos and tan of `x` (tan clipped to ±{lim}) with three different line styles ('-', "
                 f"'--', ':') and colors 'C0', 'C1', 'C2', labelled 'sin', 'cos', 'tan', legend in the lower left.",
         f"fig, ax = plt.subplots()\nax.plot(x, np.sin(x), '-', color='C0', label='sin')\nax.plot(x, np.cos(x), '--', color='C1', label='cos')\n"
         f"ax.plot(x, np.clip(np.tan(x), -{lim}, {lim}), ':', color='C2', label='tan')\nax.legend(loc='lower left')",
         {"post": "fc.lines(fc.ax(), 'ls', 'color', 'label'), fc.legend()"}),
        ("colorbar", "Drill 2: scatter `pts` colored by the continuous `vals` (`cmap='viridis'`), sizes `20 + 30 * "
                     "np.abs(vals)`, `alpha=0.7`, black edges of width 0.3, and a colorbar labelled 'x + y'.",
         "fig, ax = plt.subplots()\nsc = ax.scatter(pts[:, 0], pts[:, 1], c=vals, cmap='viridis', s=20 + 30 * np.abs(vals), alpha=0.7, "
         "edgecolors='k', linewidths=0.3)\nfig.colorbar(sc, ax=ax, label='x + y')",
         {"post": SC + "_sc(fc.ax(0), 0, 'array', 'cmap', 'sizes', 'alpha', 'edgecolor', 'lw'), fc.colorbars()", "tol": 1e-5}),
        ("legend_elements", f"Drill 3: scatter the {k} classes in **one** call (shifted right by 6: `pts[:, 0] + 6`), "
                            f"colored by `classes` with `cmap='tab10'`, `s=15`, triangle markers, and a legend from "
                            f"`legend_elements()` titled 'class' in the upper right.",
         "fig, ax = plt.subplots()\nsc2 = ax.scatter(pts[:, 0] + 6, pts[:, 1], c=classes, cmap='tab10', s=15, marker='^')\n"
         "ax.legend(*sc2.legend_elements(), title='class', loc='upper right')",
         {"post": SC + "_sc(None, 0, 'array', 'cmap', 'sizes', 'marker'), fc.legend(), fc.legend_title()"}),
        ("Line2D", "The guide's line block in one figure: `ax.plot(x, np.sin(x))`; `ax.plot(x, np.cos(x), 'r--')`; the "
                   "half sine `0.5 * np.sin(x)` in 'tab:green', `'-.'`, linewidth 2, circle markers of size 4 with white "
                   "faces every 5 points, alpha 0.8, label 'half', zorder 3; then `line, = ax.plot(x, x / 10)` updated "
                   "through the handle to purple, y data `x / 20` and label 'updated'. Return `isinstance(line, Line2D)` "
                   "(`from matplotlib.lines import Line2D`).",
         "from matplotlib.lines import Line2D\nfig, ax = plt.subplots()\nax.plot(x, np.sin(x))\nax.plot(x, np.cos(x), 'r--')\n"
         "ax.plot(x, 0.5 * np.sin(x), color='tab:green', linestyle='-.', linewidth=2, marker='o', markersize=4, "
         "markerfacecolor='white', markevery=5, alpha=0.8, label='half', zorder=3)\nline, = ax.plot(x, x / 10)\n"
         "line.set_color('purple')\nline.set_ydata(x / 20)\nline.set_label('updated')\nresult = isinstance(line, Line2D)",
         {"post": "_, fc.lines(fc.ax(), 'color', 'ls', 'lw', 'marker', 'ms', 'mfc', 'markevery', 'alpha', 'label', 'zorder')",
          "require": ["Line2D", "set_ydata"]}),
        ("legend", "The per-class alternative: for each class k, `xk, yk = pts[classes == k].T` and scatter them with "
                   "`s=10` and label `f'class {k}'`; legend with `markerscale=2`. Then save the figure as "
                   "`classes.png` at 30 dpi and close it; return the saved image's shape.",
         f"fig, ax = plt.subplots()\nfor k in range({k}):\n    xk, yk = pts[classes == k].T\n    ax.scatter(xk, yk, s=10, label=f'class {{k}}')\n"
         f"ax.legend(markerscale=2)\nleg = [t.get_text() for t in ax.get_legend().get_texts()]\n"
         f"fig.savefig('classes.png', dpi=30)\nplt.close(fig)\nresult = (leg, plt.imread('classes.png').shape)",
         {"require": ["markerscale", "savefig", "close"]}),
    ]
    return catalog(rng, items, setup, f"`x` holds 50 points from 0 to 10; `pts` holds {n} 2-D points, `vals` their "
                                      f"coordinate sum and `classes` a class 0 … {k - 1} for each.")


@template(M, "hard")
def custom_legends(rng):
    code, desc, k, names = classes(rng, k=(3, 4))
    s = seeded(rng)
    colors = ["C0", "C1", "C2", "C3"][:k]
    items = [
        ("Line2D", "Build legend entries **by hand** with `Line2D` (`from matplotlib.lines import Line2D`): scatter all "
                   "points in one call with `c=[colors[l] for l in labels]`, then add a legend whose handles are "
                   "`Line2D([0], [0], marker='o', linestyle='', color=colors[c], label=class_names[c])` for each class.",
         f"from matplotlib.lines import Line2D\nfig, ax = plt.subplots()\nax.scatter(pts[:, 0], pts[:, 1], c=[colors[l] for l in labels])\n"
         f"handles = [Line2D([0], [0], marker='o', linestyle='', color=colors[c], label=class_names[c]) for c in range({k})]\n"
         f"ax.legend(handles=handles)",
         {"post": "fc.legend(), [(fc.hexcolor(h.get_color()), h.get_marker(), h.get_linestyle()) for h in fc.ax().get_legend().legend_handles]",
          "setup": f"colors = {colors}", "require": ["Line2D", "handles"]}),
        ("add_artist", "Two legends on one Axes: the line **style** means the split (solid = train, dashed = val) and "
                       "the **color** means the model. Plot `curves[m][0]` (train, solid) and `curves[m][1]` (val, "
                       "dashed) for models 'A' ('C0') and 'B' ('C1'). Legend 1 (upper right): two black Line2D handles, "
                       "'-' 'train' and '--' 'val'; add it with `ax.add_artist(...)`; legend 2 (lower left): two solid "
                       "handles in 'C0' / 'C1' labelled 'A' / 'B'.",
         "from matplotlib.lines import Line2D\nfig, ax = plt.subplots()\nfor m, col in (('A', 'C0'), ('B', 'C1')):\n"
         "    ax.plot(curves[m][0], color=col)\n    ax.plot(curves[m][1], '--', color=col)\n"
         "first = ax.legend(handles=[Line2D([0], [0], color='k', ls='-', label='train'), Line2D([0], [0], color='k', ls='--', label='val')], "
         "loc='upper right')\nax.add_artist(first)\n"
         "ax.legend(handles=[Line2D([0], [0], color='C0', label='A'), Line2D([0], [0], color='C1', label='B')], loc='lower left')",
         {"post": "fc.lines(fc.ax(), 'color', 'ls'), fc.legend(), [[t.get_text() for t in a.get_texts()] for a in fc.ax().artists "
                  "if type(a).__name__ == 'Legend']",
          "setup": f"g = np.random.default_rng({s})\ncurves = {{m: np.round(np.exp(-np.linspace(0, 3, 20))[None] * [[1], [1.2]] + 0.03 * g.random((2, 20)) + o, 4) "
                   f"for m, o in (('A', 0.0), ('B', 0.2))}}", "require": ["Line2D", "add_artist"]}),
        ("legend_elements", "A legend for **sizes**: scatter all points with `s=sizes` (one per point), then make a legend "
                            "from `sc.legend_elements(prop='sizes', num=4)` titled 'size'.",
         "fig, ax = plt.subplots()\nsc = ax.scatter(pts[:, 0], pts[:, 1], s=sizes)\nax.legend(*sc.legend_elements(prop='sizes', num=4), title='size')",
         {"post": "fc.legend(), fc.legend_title()", "setup": f"sizes = np.round(np.random.default_rng({s}).uniform(10, 200, len(pts)))",
          "require": ["legend_elements", "prop"]}),
        ("legend_elements", "Color **and** size legends together: scatter with `c=labels` and `s=sizes`; the first legend "
                            "(class colors, title 'class', upper left) must survive the second one (sizes from "
                            "`legend_elements(prop='sizes', num=3)`, title 'size', lower right) — add the first with "
                            "`ax.add_artist`.",
         "fig, ax = plt.subplots()\nsc = ax.scatter(pts[:, 0], pts[:, 1], c=labels, s=sizes)\n"
         "first = ax.legend(*sc.legend_elements(), title='class', loc='upper left')\nax.add_artist(first)\n"
         "ax.legend(*sc.legend_elements(prop='sizes', num=3), title='size', loc='lower right')",
         {"post": "fc.legend(), fc.legend_title(), [(a.get_title().get_text(), [t.get_text() for t in a.get_texts()]) "
                  "for a in fc.ax().artists if type(a).__name__ == 'Legend']",
          "setup": f"sizes = np.round(np.random.default_rng({s}).uniform(10, 200, len(pts)))", "require": ["legend_elements", "add_artist"]}),
        ("handles", "Only some lines in the legend, in a chosen order: plot `pts[:, 0]` (label 'x'), `pts[:, 1]` (label "
                    "'y') and `pts[:, 0] + pts[:, 1]` (label 'sum'), keep the handles, and make a legend listing 'sum' "
                    "first and then 'x' (no 'y'), via `ax.legend(handles=[...])`.",
         "fig, ax = plt.subplots()\nlx, = ax.plot(pts[:, 0], label='x')\nly, = ax.plot(pts[:, 1], label='y')\n"
         "ls, = ax.plot(pts[:, 0] + pts[:, 1], label='sum')\nax.legend(handles=[ls, lx])",
         {"post": "fc.lines(fc.ax(), 'label'), fc.legend()", "require": ["handles"]}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "hard")
def drawstyle_emulation(rng):
    s = seeded(rng)
    n = rng.randint(6, 12)
    k = rng.choice([2, 3, 4])
    setup = (f"g = np.random.default_rng({s})\nx = np.cumsum(np.round(g.uniform(0.5, 2, {n}), 2))\n"
             f"y = np.round(g.normal(size={n}), 3)")
    items = [
        ("drawstyle", "`drawstyle='steps-post'` draws a horizontal segment from each point to the next x, then a "
                      "vertical jump. Draw the **same picture** with a plain line instead: build `sx`, `sy` (each x "
                      "after the first appears twice: `sx = [x0, x1, x1, x2, x2, …]`, `sy = [y0, y0, y1, y1, y2, …]`) "
                      "and `ax.plot(sx, sy)`. (Checked: your vertices equal the step line's path.)",
         "sx = np.repeat(x, 2)[1:]\nsy = np.repeat(y, 2)[:-1]\nfig, ax = plt.subplots()\nax.plot(sx, sy)",
         {"post": "_f, _a = plt.subplots()\n_l, = _a.plot(x, y, drawstyle='steps-post')\n_l.recache()\n"
                  "fc.r(_l._path.vertices, 6) == fc.r(np.column_stack(fc.ax(0, fc.figs()[0]).get_lines()[0].get_data()), 6)",
          "require": ["plot"]}),
        ("drawstyle", "Same idea for `drawstyle='steps-pre'`: the vertical jump comes **first**, then the horizontal "
                      "segment (`sx = [x0, x0, x1, x1, x2, …]`, `sy = [y0, y1, y1, y2, y2, …]`). Build the arrays and "
                      "plot them with the default drawstyle.",
         "sx = np.repeat(x, 2)[:-1]\nsy = np.repeat(y, 2)[1:]\nfig, ax = plt.subplots()\nax.plot(sx, sy)",
         {"post": "_f, _a = plt.subplots()\n_l, = _a.plot(x, y, drawstyle='steps-pre')\n_l.recache()\n"
                  "fc.r(_l._path.vertices, 6) == fc.r(np.column_stack(fc.ax(0, fc.figs()[0]).get_lines()[0].get_data()), 6)",
          "require": ["plot"]}),
        ("markevery", f"`markevery={k}` puts markers on points 0, {k}, {2 * k}, …. Emulate it with **two** artists: a plain "
                      f"line through all points and a markers-only plot (`'o'`) through every {k}-th point. Return the "
                      f"marker indices that `markevery={k}` would pick.",
         f"fig, ax = plt.subplots()\nax.plot(x, y)\nax.plot(x[::{k}], y[::{k}], 'o')\nresult = list(range(0, len(x), {k}))",
         {"post": "_, fc.lines(fc.ax(), 'ls', 'marker')", "require": ["plot"]}),
        ("drawstyle", "`drawstyle='steps-mid'` jumps halfway between neighbouring points. Plot `y` against `x` with it and "
                      "return the step line's path vertices (call `line.recache()` first, then `line._path.vertices`) "
                      "as a list of [x, y] pairs: the horizontal levels change at the midpoints.",
         "fig, ax = plt.subplots()\nline, = ax.plot(x, y, drawstyle='steps-mid')\nline.recache()\nresult = line._path.vertices.tolist()",
         {"post": "_, fc.lines(fc.ax(), 'drawstyle')", "require": ["drawstyle"]}),
        ("plot", "NaNs split a line into pieces. Insert a gap after every 3rd point: build `xs`, `ys` with an `np.nan` "
                 "after points 2, 5, 8, … (in both arrays) and plot them as one line. Return how many separate "
                 "pieces are drawn (runs of non-NaN values).",
         "xs, ys = [], []\nfor i, (a, b) in enumerate(zip(x, y)):\n    xs.append(a)\n    ys.append(b)\n    if i % 3 == 2:\n"
         "        xs.append(np.nan)\n        ys.append(np.nan)\nfig, ax = plt.subplots()\nax.plot(xs, ys)\n"
         "ok = ~np.isnan(ys)\nresult = int(ok[0]) + int(np.sum(ok[1:] & ~ok[:-1]))",
         {"post": "_, fc.lines(fc.ax())", "require": ["nan"]}),
    ]
    return catalog(rng, items, setup, f"`x` holds {n} increasing positions and `y` a value at each.")


# ================================================================= challenge

def _dec_setup(seed: int, n: int, W: int, report: str) -> str:
    return (f"g = np.random.default_rng({seed})\nx = np.cumsum(g.exponential(1.0, {n}))\n"
            f"y = np.round(np.cumsum(g.normal(size={n})), 1)\nW = {W}\nreport = {report!r}")


@template(M, "challenge")
def decimate_series(rng):
    s = seeded(rng)
    report = rng.choice(["indices", "count", "envelope"])
    n, W = rng.randint(14, 24), rng.randint(3, 6)
    setup = _dec_setup(s, n, W, report)
    sol = ("n = len(x)\nb = np.minimum(((x - x[0]) / (x[-1] - x[0]) * W).astype(np.int64), W - 1)\n"
           "starts = np.flatnonzero(np.r_[True, b[1:] != b[:-1]])\nends = np.r_[starts[1:], n] - 1\n")
    sol += {"indices": ("o_min = np.lexsort((np.arange(n), y, b))       # per bucket: smallest y, then smallest index\n"
                        "o_max = np.lexsort((np.arange(n), -y, b))\nfirst_of = np.r_[0, np.cumsum(ends - starts + 1)[:-1]]\n"
                        "keep = np.concatenate([starts, ends, o_min[first_of], o_max[first_of]])\nresult = np.unique(keep)"),
            "count": ("o_min = np.lexsort((np.arange(n), y, b))\no_max = np.lexsort((np.arange(n), -y, b))\n"
                      "first_of = np.r_[0, np.cumsum(ends - starts + 1)[:-1]]\n"
                      "result = len(np.unique(np.concatenate([starts, ends, o_min[first_of], o_max[first_of]])))"),
            "envelope": ("env = np.full((W, 2), np.nan)\nenv[b[starts], 0] = np.minimum.reduceat(y, starts)\n"
                         "env[b[starts], 1] = np.maximum.reduceat(y, starts)\nresult = env")}[report]
    tests = [_dec_setup(s + i, nn, ww, report) for i, (nn, ww) in ((1, (500000, 1500)), (2, (300000, 299000)))]
    want = {"indices": "the sorted array of kept indices", "count": "how many points are kept",
            "envelope": "a W × 2 array with the (min, max) of `y` in each pixel column (NaN, NaN for empty columns) — "
                        "the vertical bar a renderer draws per column"}[report]
    return challenge(
        "Decimating a huge line",
        "A line with millions of points is wider than the screen: many points share one pixel column. A renderer "
        "can draw exactly the same picture from a few points per column. Split the x-range `x[0] … x[-1]` (x is "
        "sorted) into `W` equal pixel columns: point i goes to column `min(W - 1, floor((x[i] - x[0]) / (x[-1] - "
        "x[0]) · W))`. In every non-empty column keep the **first** and **last** point, the point with the smallest "
        "`y` and the one with the largest `y` (on ties, the smallest index). "
        f"Return {want}. (`report` = {report!r}.)",
        setup, sol, tests, "n ≤ 5·10^5, W ≤ 3·10^5", speed="O(n log n), vectorized")


def _cyc_setup(seed: int, n: int, P: int, K: int, report: str) -> str:
    return (f"g = np.random.default_rng({seed})\nP = {P}\nop = g.choice(6, {n}, p=[0.3, 0.15, 0.2, 0.15, 0.1, 0.1])\n"
            f"arg = g.integers(1, {K + 1}, {n})\narg[op == 5] = g.integers(1, P + 1, int((op == 5).sum()))\nreport = {report!r}")


@template(M, "challenge")
def prop_cycle_replay(rng):
    s = seeded(rng)
    report = rng.choice(["histogram", "scatter", "last"])
    n, P, K = rng.randint(10, 16), rng.randint(3, 6), 3
    setup = _cyc_setup(s, n, P, K, report)
    head = ("m, line_pos, patch_pos = P, 0, 0\n")
    body = {"histogram": ("diff = np.zeros(P + 1, dtype=np.int64)\n\ndef take(pos, k):\n"
                          "    q, r = divmod(k, m)\n    diff[0] += q\n    diff[m] -= q\n    end = pos + r\n"
                          "    if end <= m:\n        diff[pos] += 1\n        diff[end] -= 1\n    else:\n"
                          "        diff[pos] += 1\n        diff[m] -= 1\n        diff[0] += 1\n        diff[end - m] -= 1\n"
                          "    return (pos + k) % m\n\n"
                          "for o, a in zip(op.tolist(), arg.tolist()):\n    if o == 0:\n        line_pos = take(line_pos, a)\n"
                          "    elif o == 4:\n        line_pos = take(line_pos, 1)\n    elif o in (2, 3):\n"
                          "        patch_pos = take(patch_pos, 1)\n    elif o == 5:\n        m, line_pos, patch_pos = a, 0, 0\n"
                          "result = np.cumsum(diff[:P])"),
            "scatter": ("out = []\nfor o, a in zip(op.tolist(), arg.tolist()):\n    if o == 0:\n"
                        "        line_pos = (line_pos + a) % m\n    elif o == 4:\n        line_pos = (line_pos + 1) % m\n"
                        "    elif o in (2, 3):\n        if o == 2:\n            out.append(patch_pos)\n"
                        "        patch_pos = (patch_pos + 1) % m\n    elif o == 5:\n        m, line_pos, patch_pos = a, 0, 0\n"
                        "result = out"),
            "last": ("last = -1\nfor o, a in zip(op.tolist(), arg.tolist()):\n    if o == 0:\n"
                     "        line_pos = (line_pos + a) % m\n        last = (line_pos - 1) % m\n    elif o == 4:\n"
                     "        last = line_pos\n        line_pos = (line_pos + 1) % m\n    elif o in (2, 3):\n"
                     "        last = patch_pos\n        patch_pos = (patch_pos + 1) % m\n    elif o == 5:\n"
                     "        m, line_pos, patch_pos = a, 0, 0\nresult = last")}[report]
    tests = [_cyc_setup(s + i, nn, pp, kk, report) for i, (nn, pp, kk) in ((1, (200000, 100000, 1000000)),
                                                                          (2, (200000, 7, 50)))]
    want = {"histogram": "a length-P integer array: how many artists got each palette color (index 0 … P-1), counting "
                         "every line drawn by op 0 and one artist per op 2, 3 and 4",
            "scatter": "the list of palette indices that the scatter calls (op 2) got, in order",
            "last": "the palette index of the color the **last** default-colored artist got (op 0, 2, 3 or 4), or -1 "
                    "if there was none"}[report]
    return challenge(
        "Who gets which color?",
        "An Axes hands out default colors from its property cycle, but it keeps **two** independent positions in it: "
        "one for lines and one for patches / collections. Replay a script on a fresh Axes whose cycle is a palette "
        "of `P` colors (indices 0 … P-1). `op[i]` with argument `arg[i]`: **0** = `ax.plot(Y)` with a 2-D `Y` of "
        "`arg` columns (one line per column, each taking the next **line** color); **1** = `ax.plot(y, color='k')` "
        "(an explicit color does not advance anything); **2** = `ax.scatter(...)` (next **patch** color); **3** = "
        "`ax.bar(...)` (next patch color, one for the whole call); **4** = `ax.hist(...)` — careful: hist takes the "
        "next **line** color; **5** = `ax.set_prop_cycle(color=palette[:arg])`, which restarts **both** positions "
        "at index 0 of the shortened cycle (the length then stays `arg` until the next op 5). "
        f"Return {want}. (`report` = {report!r}.)",
        setup, head + body, tests, "len(op) ≤ 2·10^5, P ≤ 10^5, arg ≤ 10^6",
        speed="O(len(op) + P): the line count of a call can be huge, so count whole cycles arithmetically")


def _pick_setup(seed: int, n: int, q: int, report: str) -> str:
    log = report == "log"
    return (f"g = np.random.default_rng({seed})\nfigsize, dpi = (6.0, 4.0), 100\nbox = (0.125, 0.11, 0.775, 0.77)\n"
            + (f"xlim = (1.0, 1000.0)\npx = np.round(10 ** g.uniform(0, 3, {n}), 3)\nqx = np.round(10 ** g.uniform(0, 3, {q}), 3)\n"
               if log else
               f"xlim = (0.0, 100.0)\npx = np.round(g.uniform(0, 100, {n}), 3)\nqx = np.round(g.uniform(0, 100, {q}), 3)\n")
            + f"ylim = (-1.0, 1.0)\npy = np.round(g.uniform(-1, 1, {n}), 4)\nqy = np.round(g.uniform(-1, 1, {q}), 4)\n"
              f"radius = 15.0\nxscale = {'log' if log else 'linear'!r}\nreport = {report!r}")


@template(M, "challenge")
def hover_pick(rng):
    s = seeded(rng)
    report = rng.choice(["nearest", "within", "log"])
    n, q = rng.randint(8, 15), rng.randint(3, 6)
    setup = _pick_setup(s, n, q, report)
    sol = ("W, H = figsize[0] * dpi, figsize[1] * dpi\n"
           + ("f = np.log10                       # the x-scale's transform\n\n" if report == "log" else "f = lambda t: t\n\n")
           + "def to_px(x, y):\n    tx = (f(x) - f(xlim[0])) / (f(xlim[1]) - f(xlim[0]))\n    ty = (y - ylim[0]) / (ylim[1] - ylim[0])\n"
           "    return (box[0] + box[2] * tx) * W, (box[1] + box[3] * ty) * H\n\n"
           "PX, PY = to_px(px, py)\nQX, QY = to_px(qx, qy)\nout = []\nfor i in range(0, len(QX), 256):\n"
           "    d2 = (QX[i:i + 256, None] - PX) ** 2 + (QY[i:i + 256, None] - PY) ** 2\n"
           + {"nearest": "    out.append(d2.argmin(axis=1))\nresult = np.concatenate(out)",
              "log": "    out.append(d2.argmin(axis=1))\nresult = np.concatenate(out)",
              "within": "    out.append((d2 <= radius ** 2).sum(axis=1))\nresult = np.concatenate(out)"}[report])
    tests = [_pick_setup(s + i, nn, qq, report) for i, (nn, qq) in ((1, (3000, 3000)), (2, (200, 20000)))]
    want = {"nearest": "for each query, the index of the point closest to it **on screen** (in pixels; smallest index "
                       "on ties)",
            "log": "for each query, the index of the point closest to it on screen, where the x-axis is "
                   "**logarithmic** (`xscale = 'log'`: positions are spaced by log10 x)",
            "within": "for each query, how many points lie within `radius` pixels of it (the pick radius)"}[report]
    return challenge(
        "What did the mouse hover over?",
        "A hover tooltip must find the point under the cursor, and *closest* means closest **on screen**, not in data "
        "units: x and y have very different scales. The figure is `figsize` inches at `dpi`; the Axes occupies "
        "`box = (left, bottom, width, height)` in figure fractions and shows `xlim` × `ylim` with the x-scale `xscale` "
        "(y is linear). A data point maps to pixels like `ax.transData` does: `X = (left + width · t_x) · W` with "
        "`t_x` the point's fraction of the x-range (after the scale) and `W = figsize[0] · dpi`, the same for Y. "
        f"Points are `(px, py)`, cursor positions `(qx, qy)` (data coordinates). Return {want}, as an array. "
        f"(`report` = {report!r}.)",
        setup, sol, tests, "n, q ≤ 3000 (n·q ≤ 10^7)", speed="O(n·q) vectorized in chunks (or a spatial grid)")
