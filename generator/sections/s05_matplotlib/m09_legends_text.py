"""5.09 Legends, annotations, text & math (mathtext)."""

from ._common import P, catalog, categories, challenge, classes, curves, lit, seeded, signal, template

M = "5.09"

# ---------------------------------------------------------------- check code (post): legends, texts, mathtext as data
_PRE = '''from matplotlib.lines import Line2D as _L2D
_FR = {'axes fraction': 'axes', 'figure fraction': 'figure', 'data': 'data', 'offset points': 'offset points'}

def _lg(lg, *keys):
    if lg is None:
        return None
    out = {}
    for k in keys:
        if k == 'texts':
            out[k] = [t.get_text() for t in lg.get_texts()]
        elif k == 'loc':
            out[k] = lg._loc
        elif k == 'ncols':
            out[k] = lg._ncols
        elif k == 'frame':
            out[k] = lg.get_frame_on()
        elif k == 'title':
            out[k] = lg.get_title().get_text()
        elif k == 'size':
            out[k] = round(lg.get_texts()[0].get_fontsize(), 2)
        elif k == 'anchor':
            p = lg.parent
            tr = p.transAxes if hasattr(p, 'transAxes') else p.transFigure
            out[k] = [round(float(v), 3) for v in lg.get_bbox_to_anchor().transformed(tr.inverted()).bounds[:2]]
        elif k == 'handles':
            out[k] = [type(h).__name__ for h in lg.legend_handles]
        elif k == 'outside':
            out[k] = getattr(lg, '_outside_loc', None)
        elif k == 'colors':
            out[k] = [fc.hexcolor(h.get_color() if isinstance(h, _L2D) else h.get_facecolor()) for h in lg.legend_handles]
        elif k == 'ls':
            out[k] = [h.get_linestyle() if isinstance(h, _L2D) else None for h in lg.legend_handles]
        elif k == 'markerscale':
            out[k] = lg.markerscale
    return out

def _legends(a):
    return [c for c in a.get_children() if isinstance(c, mpl.legend.Legend)]

def _frame(t):
    if hasattr(t, 'anncoords'):
        return _FR.get(t.anncoords, str(t.anncoords))
    tr, a = t.get_transform(), t.axes
    if a is not None and tr == a.transData:
        return 'data'
    if a is not None and tr == a.transAxes:
        return 'axes'
    if tr == t.figure.transFigure:
        return 'figure'
    return 'other'

def _tx(t, *keys):
    ann = hasattr(t, 'xy')
    out = {}
    for k in keys:
        if k == 'text':
            out[k] = t.get_text()
        elif k == 'pos':
            out[k] = [round(float(v), 3) for v in t.get_position()]
        elif k == 'ha':
            out[k] = t.get_horizontalalignment()
        elif k == 'va':
            out[k] = t.get_verticalalignment()
        elif k == 'color':
            out[k] = fc.hexcolor(t.get_color())
        elif k == 'size':
            out[k] = round(t.get_fontsize(), 2)
        elif k == 'frame':
            out[k] = _frame(t)
        elif k == 'target':
            out[k] = [round(float(v), 3) for v in t.xy] if ann else None
        elif k == 'xycoords':
            out[k] = _FR.get(t.xycoords, str(t.xycoords)) if ann else None
        elif k == 'arrow':
            out[k] = ann and t.arrow_patch is not None
        elif k == 'style':
            out[k] = type(t.arrow_patch.get_arrowstyle()).__name__ if ann and t.arrow_patch is not None else None
        elif k == 'arrowcolor':
            out[k] = fc.hexcolor(t.arrow_patch.get_edgecolor()) if ann and t.arrow_patch is not None else None
        elif k == 'box':
            b = t.get_bbox_patch()
            out[k] = None if b is None else type(b.get_boxstyle()).__name__
        elif k == 'rotation':
            out[k] = round(t.get_rotation(), 3)
        elif k == 'bold':
            out[k] = str(t.get_fontweight()) in ('bold', '700', 'heavy', 'extra bold', 'black')
        elif k == 'alpha':
            out[k] = t.get_alpha()
        elif k == 'family':
            out[k] = t.get_family()[0]
        elif k == 'conn':
            cs = t.arrow_patch.get_connectionstyle() if ann and t.arrow_patch is not None else None
            out[k] = None if cs is None else [type(cs).__name__, getattr(cs, 'rad', None)]
    return out

def _txs(a, *keys):
    return [_tx(t, *keys) for t in a.texts]

def _math(s, *need):
    from matplotlib.mathtext import MathTextParser
    d = {'mathtext': s.replace(chr(92) + '$', '').count('$') >= 2}
    try:
        for line in s.split(chr(10)):
            MathTextParser('path').parse(line)
        d['renders'] = True
    except Exception:
        d['renders'] = False
    for n in need:
        d['uses ' + n] = n in s
    return d
'''


def post(expr: str) -> str:
    return _PRE + expr


def _lgp(keys: str, where: str = "fc.ax().get_legend()") -> str:
    return post(f"_lg({where}, {keys})")


def _runs(rng, k=None):
    """A few learning-rate runs: `lrs` and `runs` (one loss curve per row)."""
    k = k or rng.randint(3, 5)
    lrs = sorted(rng.sample([0.3, 0.1, 0.03, 0.01, 0.003, 0.001], k), reverse=True)
    s = seeded(rng)
    code = (f"lrs = {lrs}\ng = np.random.default_rng({s})\nsteps = np.arange(30)\n"
            f"runs = np.round(np.exp(-np.outer(np.sqrt(lrs), steps) / 2) + 0.02 * g.random(({k}, 30)), 4)")
    return code, f"`runs[i]` is the loss curve (30 `steps`) of a run with learning rate `lrs[i]`", k, lrs


# ================================================================= lite

@template(M, "lite")
def legend_basics(rng):
    code, desc, n = curves(rng)
    a, b = rng.choice([("train", "val"), ("training", "validation"), ("fit", "holdout"), ("train set", "dev set")])
    loc = rng.choice(["upper right", "lower left", "center right", "upper center", "lower right"])
    title = rng.choice(["split", "data", "curve", "subset"])
    fs = rng.choice(["small", "x-small", "large", "medium"])
    two = (f"fig, ax = plt.subplots()\nax.plot(epochs, train_loss, label={a!r})\n"
           f"ax.plot(epochs, val_loss, label={b!r})\n")
    both = f"plot `train_loss` and `val_loss` against `epochs` labelled {a!r} and {b!r}"
    items = [
        (".legend", f"Create `fig, ax`, {both}, and add a legend.", two + "ax.legend()", {"post": _lgp("'texts'")}),
        ("loc", f"Create `fig, ax`, {both}, and put the legend at the {loc!r} position (`loc=`).",
         two + f"ax.legend(loc={loc!r})", {"post": _lgp("'texts', 'loc'")}),
        ("ncols", f"Create `fig, ax`, {both}, and lay the legend out as one row: 2 columns (`ncols=`).",
         two + "ax.legend(ncols=2)", {"post": _lgp("'texts', 'ncols'")}),
        ("frameon", f"Create `fig, ax`, {both}, and add a legend **without** its frame (box).",
         two + "ax.legend(frameon=False)", {"post": _lgp("'texts', 'frame'")}),
        ("title", f"Create `fig, ax`, {both}, and give the legend the title {title!r}.",
         two + f"ax.legend(title={title!r})", {"post": _lgp("'texts', 'title'")}),
        ("fontsize", f"Create `fig, ax`, {both}, and add a legend whose text size is {fs!r} (`fontsize=`).",
         two + f"ax.legend(fontsize={fs!r})", {"post": _lgp("'texts', 'size'")}),
        ("plt.legend", f"With **pyplot only** (`plt.plot`, `plt.legend`), {both} and add a legend at {loc!r}.",
         f"plt.plot(epochs, train_loss, label={a!r})\nplt.plot(epochs, val_loss, label={b!r})\nplt.legend(loc={loc!r})",
         {"post": _lgp("'texts', 'loc'")}),
        (".set_label", f"Create `fig, ax` and plot `val_loss` against `epochs` **without** a label, keeping the line "
                       f"(`line, = ax.plot(...)`). Then label it {b!r} with `line.set_label` and add a legend.",
         f"fig, ax = plt.subplots()\nline, = ax.plot(epochs, val_loss)\nline.set_label({b!r})\nax.legend()",
         {"post": _lgp("'texts'")}),
        (".legend", f"Create `fig, ax` and plot `train_loss` then `val_loss` against `epochs` **without** `label=`. Pass "
                    f"the labels straight to the legend instead: `ax.legend([{a!r}, {b!r}])`.",
         f"fig, ax = plt.subplots()\nax.plot(epochs, train_loss)\nax.plot(epochs, val_loss)\nax.legend([{a!r}, {b!r}])",
         {"post": _lgp("'texts'")}),
        ("get_legend_handles_labels", f"Create `fig, ax`, {both}, and return the labels that "
                                      f"`ax.get_legend_handles_labels()` collects (its second element).",
         two + "handles, labels = ax.get_legend_handles_labels()\nresult = labels"),
        ("label", f"Only labelled artists get a legend entry. Create `fig, ax`; plot `train_loss` labelled {a!r}, "
                  f"`val_loss` with **no** label, and a horizontal line at `val_loss.min()` (`ax.axhline`) labelled "
                  f"'best'. Then add a legend.",
         f"fig, ax = plt.subplots()\nax.plot(epochs, train_loss, label={a!r})\nax.plot(epochs, val_loss)\n"
         f"ax.axhline(val_loss.min(), label='best')\nax.legend()", {"post": _lgp("'texts'")}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "lite")
def text_basics(rng):
    code, desc, n = signal(rng)
    word = rng.choice(["peak", "here", "spike", "note", "event"])
    x0, y0 = rng.choice([2, 3, 5, 7]), rng.choice([0.5, 1.0, -0.5, 1.5])
    c, fs = rng.choice(["red", "gray", "tab:green", "navy", "C3"]), rng.choice([8, 9, 12, 14])
    base = "fig, ax = plt.subplots()\nax.plot(x, y)\n"
    start = "Create `fig, ax`, plot `y` against `x`, and"
    T0 = "fc.ax().texts[0]"
    items = [
        (".text", f"{start} write {word!r} at the data position ({x0}, {y0}) with `ax.text`.",
         base + f"ax.text({x0}, {y0}, {word!r})", {"post": post(f"_tx({T0}, 'text', 'pos', 'frame')")}),
        ("ha", f"{start} write {word!r} centred horizontally on x = {x0}, with its **bottom** at y = {y0} (`ha=`, `va=`).",
         base + f"ax.text({x0}, {y0}, {word!r}, ha='center', va='bottom')",
         {"post": post(f"_tx({T0}, 'text', 'pos', 'ha', 'va')"), "require": ["ha", "va"]}),
        ("transAxes", f"{start} write 'corner' in the top-right corner using **axes coordinates**: position (0.95, 0.95) "
                      f"with `transform=ax.transAxes`, right-aligned and top-aligned.",
         base + "ax.text(0.95, 0.95, 'corner', transform=ax.transAxes, ha='right', va='top')",
         {"post": post(f"_tx({T0}, 'text', 'pos', 'frame', 'ha', 'va')")}),
        (".annotate", f"{start} annotate the highest point of `y` with the text 'max' placed 0.5 above it (same x), with "
                      f"an arrow (`arrowprops=dict(arrowstyle='->')`).",
         base + "i = y.argmax()\nax.annotate('max', xy=(x[i], y[i]), xytext=(x[i], y[i] + 0.5), "
                "arrowprops=dict(arrowstyle='->'))",
         {"post": post(f"_tx({T0}, 'text', 'target', 'pos', 'arrow')")}),
        ("xycoords", f"{start} put the annotation {word!r} at (0.02, 0.95) in **axes fraction** coordinates "
                     f"(`xycoords='axes fraction'`), top-aligned (`va='top'`).",
         base + f"ax.annotate({word!r}, xy=(0.02, 0.95), xycoords='axes fraction', va='top')",
         {"post": post(f"_tx({T0}, 'text', 'pos', 'frame', 'va')")}),
        ("textcoords", f"{start} label the last point with its value to 2 decimals (`f'{{y[-1]:.2f}}'`), the text "
                       f"offset by (10, -15) **points** from the point (`textcoords='offset points'`).",
         base + "ax.annotate(f'{y[-1]:.2f}', xy=(x[-1], y[-1]), xytext=(10, -15), textcoords='offset points')",
         {"post": post(f"_tx({T0}, 'text', 'target', 'pos', 'frame')")}),
        ("color", f"{start} write {word!r} at ({x0}, {y0}) in the color {c!r} with font size {fs}.",
         base + f"ax.text({x0}, {y0}, {word!r}, color={c!r}, fontsize={fs})",
         {"post": post(f"_tx({T0}, 'text', 'color', 'size')"), "require": ["color", "fontsize|size"]}),
        ("bbox", f"{start} write {word!r} at ({x0}, {y0}) inside a rounded box: "
                 f"`bbox=dict(boxstyle='round', fc='w')`.",
         base + f"ax.text({x0}, {y0}, {word!r}, bbox=dict(boxstyle='round', fc='w'))",
         {"post": post(f"_tx({T0}, 'text', 'pos', 'box')")}),
        ("rotation", f"{start} write 'vertical' at ({x0}, {y0}) rotated by 90 degrees.",
         base + f"ax.text({x0}, {y0}, 'vertical', rotation=90)", {"post": post(f"_tx({T0}, 'text', 'pos', 'rotation')")}),
        ("plt.text", f"With **pyplot only**, plot `y` against `x` and write {word!r} at ({x0}, {y0}) with `plt.text`.",
         f"plt.plot(x, y)\nplt.text({x0}, {y0}, {word!r})", {"post": post(f"_tx({T0}, 'text', 'pos')")}),
        ("plt.annotate", "With **pyplot only**, plot `y` against `x` and annotate its lowest point with 'min', the text "
                         "0.5 below the point, with an arrow (`arrowprops=dict(arrowstyle='->')`).",
         "plt.plot(x, y)\ni = y.argmin()\nplt.annotate('min', xy=(x[i], y[i]), xytext=(x[i], y[i] - 0.5), "
         "arrowprops=dict(arrowstyle='->'))", {"post": post(f"_tx({T0}, 'text', 'target', 'pos', 'arrow')")}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "lite")
def mathtext_basics(rng):
    n = rng.choice([50, 100, 200])
    greek = rng.choice(["alpha", "beta", "eta", "lambda", "theta", "sigma", "omega"])
    word = rng.choice(["angle", "phase", "input"])
    p = rng.choice([2, 3])
    k = rng.choice([1, 2, 3])
    base = f"fig, ax = plt.subplots()\nax.plot(x, np.sin({k} * x))\n"
    start = f"Create `fig, ax` and plot `np.sin({k} * x)` against `x`."
    g = "\\" + greek
    items = [
        (".set_xlabel", f"{start} Mathtext writes math between dollar signs (use a raw string `r'...'`). Label the x "
                        f"axis {word!r} followed by the Greek letter {greek} in mathtext, like `{word} ${g}$`.",
         base + f"ax.set_xlabel(r'{word} ${g}$')", {"post": post(f"_math(fc.ax().get_xlabel(), {word!r}, {g!r})")}),
        (".set_ylabel", f"{start} Label the y axis with the mathtext power `$y^{p}$`.",
         base + f"ax.set_ylabel(r'$y^{p}$')", {"post": post(f"_math(fc.ax().get_ylabel(), 'y^', '{p}')")}),
        (".set_title", f"{start} Title it with the fraction 1/N in mathtext, using `\\frac{{1}}{{N}}`.",
         base + "ax.set_title(r'$\\frac{1}{N}$')", {"post": post("_math(fc.ax().get_title(), '\\\\frac')")}),
        ("label", f"Create `fig, ax` and plot `np.sin(x)` and `np.cos(x)` against `x` with the mathtext labels "
                  f"`$\\sin(x)$` and `$\\cos(x)$`, then add a legend.",
         "fig, ax = plt.subplots()\nax.plot(x, np.sin(x), label=r'$\\sin(x)$')\nax.plot(x, np.cos(x), label=r'$\\cos(x)$')\n"
         "ax.legend()", {"post": post("[_math(t, '\\\\') for t in _lg(fc.ax().get_legend(), 'texts')['texts']], "
                                      "_lg(fc.ax().get_legend(), 'texts')")}),
        (".text", f"{start} Write the formula `$y = \\sin({k}x)$` with `ax.text` at the data position (π, 0.5).",
         base + f"ax.text(np.pi, 0.5, r'$y = \\sin({k}x)$')",
         {"post": post("_math(fc.ax().texts[0].get_text(), '\\\\sin'), _tx(fc.ax().texts[0], 'pos')")}),
        ("rcParams", "Mathtext is Matplotlib's own LaTeX subset (no TeX install needed). Rendering **all** text with a "
                     "real LaTeX is the rcParam `'text.usetex'`: return its current value from `plt.rcParams`.",
         "result = plt.rcParams['text.usetex']"),
        ("plt.title", f"With **pyplot only**, plot `np.sin({k} * x)` against `x` and title it with the subscripted "
                      f"mathtext `$\\{greek}_0$`.",
         f"plt.plot(x, np.sin({k} * x))\nplt.title(r'$\\{greek}_0$')",
         {"post": post(f"_math(fc.ax().get_title(), {g!r}, '_')")}),
        (".set_xlabel", f"{start} Label the x axis `learning rate $\\eta$ ($\\times 10^{{-3}}$)` — text with two "
                        f"mathtext parts.",
         base + "ax.set_xlabel(r'learning rate $\\eta$ ($\\times 10^{-3}$)')",
         {"post": post("_math(fc.ax().get_xlabel(), 'learning rate', '\\\\eta', '\\\\times', '10^{-3}')")}),
    ]
    return catalog(rng, items, f"x = np.linspace(0, 2 * np.pi, {n})", f"`x` holds {n} angles from 0 to 2π.")


# ================================================================= easy

@template(M, "easy")
def legend_layouts(rng):
    code, desc, n = curves(rng, acc=True)
    a, b = rng.choice([("train", "val"), ("training", "validation"), ("fit", "holdout")])
    two = (f"fig, ax = plt.subplots()\nax.plot(epochs, train_loss, label={a!r})\n"
           f"ax.plot(epochs, val_loss, label={b!r})\n")
    both = f"plot `train_loss` and `val_loss` against `epochs` labelled {a!r} and {b!r}"
    dx = rng.choice([1.02, 1.03, 1.05])
    base = rng.choice([0.2, 0.3, 0.25, 0.4])
    color = rng.choice(["C3", "C2", "gray", "tab:purple"])
    items = [
        ("bbox_to_anchor", f"Create `fig, ax`, {both}, and put the legend **outside** the Axes on the right: its "
                           f"center-left point at the axes position ({dx}, 0.5) (`loc='center left'` plus "
                           f"`bbox_to_anchor`).",
         two + f"ax.legend(loc='center left', bbox_to_anchor=({dx}, 0.5))", {"post": _lgp("'texts', 'loc', 'anchor'")}),
        ("bbox_to_anchor", f"Create `fig, ax`, {both}, and put the legend **above** the Axes as a single row: its "
                           f"lower-center point at (0.5, 1.02) in axes coordinates, 2 columns.",
         two + "ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncols=2)",
         {"post": _lgp("'texts', 'loc', 'anchor', 'ncols'")}),
        ("get_legend_handles_labels", f"Create `fig, ax`, {both}, and show the legend with its entries in **reverse** "
                                      f"order ({b!r} first): get them with `ax.get_legend_handles_labels()` and pass "
                                      f"both lists reversed to `ax.legend`.",
         two + "handles, labels = ax.get_legend_handles_labels()\nax.legend(handles[::-1], labels[::-1])",
         {"post": _lgp("'texts'")}),
        ("handles", f"Create `fig, ax`, {both}, and draw a horizontal baseline at {base} labelled 'baseline' "
                    f"(`ax.axhline`). Show a legend with **only** the two loss curves: keep their line handles and "
                    f"pass them with `handles=`.",
         f"fig, ax = plt.subplots()\nl1, = ax.plot(epochs, train_loss, label={a!r})\nl2, = ax.plot(epochs, val_loss, "
         f"label={b!r})\nax.axhline({base}, label='baseline')\nax.legend(handles=[l1, l2])",
         {"post": _lgp("'texts', 'handles'")}),
        ("Line2D", f"Legend entries can describe things that aren't drawn. Create `fig, ax`, {both}, then make a legend "
                   f"with the two curves **plus** a dashed black proxy line `Line2D([0], [0], color='k', ls='--', "
                   f"label='target')` (from `matplotlib.lines`), in that order.",
         "from matplotlib.lines import Line2D\n" + two + "handles, labels = ax.get_legend_handles_labels()\n"
         "handles.append(Line2D([0], [0], color='k', ls='--', label='target'))\nax.legend(handles=handles)",
         {"post": _lgp("'texts', 'handles', 'ls'")}),
        ("Patch", f"Create `fig, ax` and plot `val_loss` against `epochs` labelled {b!r}. Shade the epochs from the best "
                  f"one (lowest `val_loss`) to the last with `ax.axvspan(..., color={color!r}, alpha=0.2)`, and make a "
                  f"legend from the line plus a proxy `Patch(facecolor={color!r}, alpha=0.2, label='overfitting')` "
                  f"(from `matplotlib.patches`).",
         f"from matplotlib.patches import Patch\nfig, ax = plt.subplots()\nline, = ax.plot(epochs, val_loss, label={b!r})\n"
         f"best = epochs[val_loss.argmin()]\nax.axvspan(best, epochs[-1], color={color!r}, alpha=0.2)\n"
         f"ax.legend(handles=[line, Patch(facecolor={color!r}, alpha=0.2, label='overfitting')])",
         {"post": _lgp("'texts', 'handles', 'colors'")}),
        ("fig.legend", f"Create `fig, (ax1, ax2) = plt.subplots(1, 2, layout='constrained')`. On `ax1` plot `train_loss` "
                       f"and `val_loss` (labels {a!r}, {b!r}), on `ax2` plot `val_acc` labelled 'accuracy' (all against "
                       f"`epochs`). Add **one figure legend** outside the panels, at the top: `fig.legend(loc='outside "
                       f"upper center', ncols=3)` (no per-Axes legends).",
         f"fig, (ax1, ax2) = plt.subplots(1, 2, layout='constrained')\nax1.plot(epochs, train_loss, label={a!r})\n"
         f"ax1.plot(epochs, val_loss, label={b!r})\nax2.plot(epochs, val_acc, label='accuracy')\n"
         f"fig.legend(loc='outside upper center', ncols=3)",
         {"post": post("[_lg(l, 'texts', 'outside', 'ncols') for l in fc.fig().legends], "
                       "[a.get_legend() is None for a in fc.axes()]"), "require": ["legend"]}),
        ("loc", f"Create `fig, ax`, {both}, and in **one** `ax.legend` call put the legend at the lower left, in 2 "
                f"columns, without a frame, with small text (`fontsize='small'`) and the title 'split'.",
         two + "ax.legend(loc='lower left', ncols=2, frameon=False, fontsize='small', title='split')",
         {"post": _lgp("'texts', 'loc', 'ncols', 'frame', 'size', 'title'"), "require": ["loc", "ncols", "frameon"]}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "easy")
def annotate_extremes(rng):
    code, desc, n = curves(rng, acc=True)
    dx, dy = rng.choice([(2, 0.3), (3, 0.4), (2, 0.5), (4, 0.2)])
    px, py = rng.choice([(0, 15), (0, 10), (10, 10), (-20, 15)])
    nd = rng.choice([2, 3])
    c = rng.choice(["red", "black", "C2", "gray"])
    T0 = "fc.ax().texts[0]"
    base = "fig, ax = plt.subplots()\nax.plot(epochs, val_loss)\n"
    items = [
        (".annotate", f"Create `fig, ax` and plot `val_loss` against `epochs`. Annotate its **minimum** with the value "
                      f"to {nd} decimals, the text {dx} epochs to the right of and {dy} above the point, with an arrow "
                      f"(`arrowprops=dict(arrowstyle='->')`).",
         base + f"i = val_loss.argmin()\nax.annotate(f'{{val_loss[i]:.{nd}f}}', xy=(epochs[i], val_loss[i]), "
                f"xytext=(epochs[i] + {dx}, val_loss[i] + {dy}), arrowprops=dict(arrowstyle='->'))",
         {"post": post(f"_tx({T0}, 'text', 'target', 'pos', 'arrow')")}),
        ("textcoords", f"Create `fig, ax` and plot `val_acc` against `epochs`. Annotate its maximum with 'best acc', the "
                       f"text offset by ({px}, {py}) **points** from the point and centred horizontally (`ha='center'`).",
         f"fig, ax = plt.subplots()\nax.plot(epochs, val_acc)\ni = val_acc.argmax()\nax.annotate('best acc', "
         f"xy=(epochs[i], val_acc[i]), xytext=({px}, {py}), textcoords='offset points', ha='center')",
         {"post": post(f"_tx({T0}, 'text', 'target', 'pos', 'frame', 'ha')")}),
        (".annotate", "Create `fig, ax` and plot `train_loss` against `epochs`. Annotate its first point with 'start' and "
                      "its last point with 'end' — text right at the points, no arrows — in that order.",
         "fig, ax = plt.subplots()\nax.plot(epochs, train_loss)\nax.annotate('start', xy=(epochs[0], train_loss[0]))\n"
         "ax.annotate('end', xy=(epochs[-1], train_loss[-1]))", {"post": post("_txs(fc.ax(), 'text', 'target', 'arrow')")}),
        ("argsort", "Create `fig, ax` and plot `val_loss` against `epochs`. Mark the 3 epochs with the lowest `val_loss` "
                    "with their rank '1', '2', '3' (1 = lowest, annotated in that order), each text offset (0, 8) "
                    "points above its point and centred.",
         base + "for r, i in enumerate(np.argsort(val_loss)[:3], 1):\n    ax.annotate(str(r), xy=(epochs[i], val_loss[i]), "
                "xytext=(0, 8), textcoords='offset points', ha='center')",
         {"post": post("_txs(fc.ax(), 'text', 'target', 'pos', 'frame', 'ha')")}),
        ("arrowprops", f"Create `fig, ax` and plot `train_loss` against `epochs`. Annotate its minimum with 'min': the "
                       f"point is in data coordinates, but the text sits at (0.5, 0.8) in **axes fraction** "
                       f"(`textcoords='axes fraction'`), connected by a {c!r} arrow of style '-|>'.",
         f"fig, ax = plt.subplots()\nax.plot(epochs, train_loss)\ni = train_loss.argmin()\nax.annotate('min', "
         f"xy=(epochs[i], train_loss[i]), xytext=(0.5, 0.8), textcoords='axes fraction', "
         f"arrowprops=dict(arrowstyle='-|>', color={c!r}))",
         {"post": post(f"_tx({T0}, 'text', 'target', 'xycoords', 'pos', 'frame', 'style', 'arrowcolor')")}),
        ("bbox", "Create `fig, ax` and plot `val_loss` against `epochs`. Annotate the best epoch (lowest `val_loss`) "
                 "with 'best' in a rounded white box (`bbox=dict(boxstyle='round', fc='w')`), the text offset (20, 20) "
                 "points, with an arrow `'->'`.",
         base + "i = val_loss.argmin()\nax.annotate('best', xy=(epochs[i], val_loss[i]), xytext=(20, 20), "
                "textcoords='offset points', bbox=dict(boxstyle='round', fc='w'), arrowprops=dict(arrowstyle='->'))",
         {"post": post(f"_tx({T0}, 'text', 'target', 'pos', 'frame', 'box', 'arrow')")}),
        ("set_position", f"Create `fig, ax`, plot `val_loss` against `epochs`, and annotate its minimum with 'min' "
                         f"(text 1 epoch right and 0.5 above, arrow `'->'`), keeping the annotation: `ann = "
                         f"ax.annotate(...)`. Then **move** only the text to {dx} epochs right and {dy} above the point "
                         f"with `ann.set_position(...)`.",
         base + f"i = val_loss.argmin()\nann = ax.annotate('min', xy=(epochs[i], val_loss[i]), xytext=(epochs[i] + 1, "
                f"val_loss[i] + 0.5), arrowprops=dict(arrowstyle='->'))\n"
                f"ann.set_position((epochs[i] + {dx}, val_loss[i] + {dy}))",
         {"post": post(f"_tx({T0}, 'text', 'target', 'pos', 'arrow')")}),
        ("plt.annotate", "With **pyplot only**, plot `val_acc` against `epochs` and annotate its maximum with the value "
                         "as a percentage with one decimal (`f'{v:.1%}'`); the text 5 epochs to the left at the same "
                         "height, with an arrow `'->'`.",
         "plt.plot(epochs, val_acc)\ni = val_acc.argmax()\nplt.annotate(f'{val_acc[i]:.1%}', xy=(epochs[i], val_acc[i]), "
         "xytext=(epochs[i] - 5, val_acc[i]), arrowprops=dict(arrowstyle='->'))",
         {"post": post(f"_tx({T0}, 'text', 'target', 'pos', 'arrow')")}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "easy")
def bar_text_labels(rng):
    code, desc, names, vals, th, m = categories(rng, k=(3, 6))
    k = len(names)
    src = rng.choice(["internal survey", "2024 audit", "vendor data", "team logs"])
    items = [
        (".text", "Draw `ax.bar(names, values)` on `fig, ax` and write each value above its bar: the text `str(v)` at "
                  "(i, v), centred horizontally and bottom-aligned.",
         "fig, ax = plt.subplots()\nax.bar(names, values)\nfor i, v in enumerate(values):\n"
         "    ax.text(i, v, str(v), ha='center', va='bottom')", {"post": post("_txs(fc.ax(), 'text', 'pos', 'ha', 'va')")}),
        ("color", "Draw `ax.bar(names, values)` on `fig, ax` and write each bar's share of the total as a whole "
                  "percentage (`f'{v / total:.0%}'`) in white, centred both ways at half the bar's height.",
         "fig, ax = plt.subplots()\nax.bar(names, values)\ntotal = sum(values)\nfor i, v in enumerate(values):\n"
         "    ax.text(i, v / 2, f'{v / total:.0%}', color='white', ha='center', va='center')",
         {"post": post("_txs(fc.ax(), 'text', 'pos', 'color', 'ha', 'va')"), "require": ["color", "ha"]}),
        ("barh", "Draw horizontal bars `ax.barh(names, values)` on `fig, ax` and write each value just after the end of "
                 "its bar: the text `str(v)` at (v, i), left-aligned and vertically centred.",
         "fig, ax = plt.subplots()\nax.barh(names, values)\nfor i, v in enumerate(values):\n"
         "    ax.text(v, i, str(v), ha='left', va='center')", {"post": post("_txs(fc.ax(), 'text', 'pos', 'ha', 'va')")}),
        ("transAxes", "Draw `ax.bar(names, values)` on `fig, ax` and add a summary box in the lower-right corner: the text "
                      "`f'max: {name} ({value})'` for the largest value, at axes position (0.98, 0.02), right- and "
                      "bottom-aligned, with `bbox=dict(boxstyle='round', fc='w')`.",
         "fig, ax = plt.subplots()\nax.bar(names, values)\ni = int(np.argmax(values))\n"
         "ax.text(0.98, 0.02, f'max: {names[i]} ({values[i]})', transform=ax.transAxes, ha='right', va='bottom', "
         "bbox=dict(boxstyle='round', fc='w'))", {"post": post("_txs(fc.ax(), 'text', 'pos', 'frame', 'ha', 'va', 'box')")}),
        ("fig.text", f"Draw `ax.bar(names, values)` on `fig, ax` and add a figure-level footnote 'source: {src}' with "
                     f"`fig.text`, at (0.5, 0.01) in figure coordinates, centred horizontally.",
         f"fig, ax = plt.subplots()\nax.bar(names, values)\nfig.text(0.5, 0.01, 'source: {src}', ha='center')",
         {"post": post("[_tx(t, 'text', 'pos', 'frame', 'ha') for t in fc.fig().texts]"), "require": ["text"]}),
        (".annotate", "Draw `ax.bar(names, values)` on `fig, ax` and annotate the tallest bar with 'top': the arrow "
                      "points at the middle of the bar's top (i, v), the text 20 points above it (`xytext=(0, 20)`, "
                      "offset points), centred, with an arrow `'->'`.",
         "fig, ax = plt.subplots()\nax.bar(names, values)\ni = int(np.argmax(values))\nax.annotate('top', xy=(i, values[i]), "
         "xytext=(0, 20), textcoords='offset points', ha='center', arrowprops=dict(arrowstyle='->'))",
         {"post": post("_txs(fc.ax(), 'text', 'target', 'pos', 'frame', 'arrow')")}),
        ("va", f"Draw `ax.bar(names, values)` on `fig, ax` and write a two-line summary in the upper-left corner: "
               f"`'n = {k}'` and `f'mean = {{np.mean(values):.2f}}'` on separate lines of **one** text, at axes "
               f"position (0.02, 0.98), top-aligned.",
         "fig, ax = plt.subplots()\nax.bar(names, values)\n"
         "ax.text(0.02, 0.98, f'n = {len(values)}\\nmean = {np.mean(values):.2f}', transform=ax.transAxes, va='top')",
         {"post": post("_txs(fc.ax(), 'text', 'pos', 'frame', 'va')")}),
        ("rotation", "Draw `ax.bar(names, values)` on `fig, ax` and write each name **inside** its bar instead: rotated "
                     "90°, at half the bar's height, centred both ways.",
         "fig, ax = plt.subplots()\nax.bar(names, values)\nfor i, (name, v) in enumerate(zip(names, values)):\n"
         "    ax.text(i, v / 2, name, rotation=90, ha='center', va='center')",
         {"post": post("_txs(fc.ax(), 'text', 'pos', 'rotation', 'ha', 'va')")}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "easy")
def math_formulas(rng):
    code, desc, n = curves(rng)
    lr = rng.choice([0.1, 0.01, 0.003, 0.0003])
    m = rng.choice([50, 100, 120])
    base = "fig, ax = plt.subplots()\nax.plot(epochs, train_loss)\n"
    start = "Create `fig, ax` and plot `train_loss` against `epochs`."
    chk = "(checked: it is mathtext, it renders, and it uses {})."
    items = [
        (".set_title", f"{start} Title it with the cross-entropy loss `$\\mathcal{{L}}(\\theta) = -\\frac{{1}}{{N}}"
                       f"\\sum_i y_i \\log \\hat{{y}}_i$` in mathtext " + chk.format("`\\mathcal{L}`, `\\frac`, `\\sum`, "
                       "`\\log` and `\\hat`"),
         base + "ax.set_title(r'$\\mathcal{L}(\\theta) = -\\frac{1}{N}\\sum_i y_i \\log \\hat{y}_i$')",
         {"post": post("_math(fc.ax().get_title(), '\\\\mathcal{L}', '\\\\frac', '\\\\sum', '\\\\log', '\\\\hat')")}),
        (".set_title", f"{start} Title it with the softmax formula: `e^{{z_i}}` divided by `\\sum_j e^{{z_j}}`, written as "
                       f"one `\\frac` in mathtext " + chk.format("`\\frac`, `\\sum`, `e^{z_i}` and `e^{z_j}`"),
         base + "ax.set_title(r'$\\mathrm{softmax}(z)_i = \\frac{e^{z_i}}{\\sum_j e^{z_j}}$')",
         {"post": post("_math(fc.ax().get_title(), '\\\\frac', '\\\\sum', 'e^{z_i}', 'e^{z_j}')")}),
        (".set_ylabel", f"{start} Label the y axis with the mean squared error `$\\frac{{1}}{{N}}\\sum_i (y_i - "
                        f"\\hat{{y}}_i)^2$` " + chk.format("`\\frac`, `\\sum`, `\\hat` and `^2`"),
         base + "ax.set_ylabel(r'$\\frac{1}{N}\\sum_i (y_i - \\hat{y}_i)^2$')",
         {"post": post("_math(fc.ax().get_ylabel(), '\\\\frac', '\\\\sum', '\\\\hat', '^2')")}),
        ("label", f"Plot the sigmoid of `z` on `fig, ax` labelled with its formula `$\\sigma(z) = \\frac{{1}}{{1 + "
                  f"e^{{-z}}}}$`, and add a legend " + chk.format("`\\sigma`, `\\frac` and `e^{-z}`"),
         "fig, ax = plt.subplots()\nax.plot(z, 1 / (1 + np.exp(-z)), label=r'$\\sigma(z) = \\frac{1}{1 + e^{-z}}$')\n"
         "ax.legend()", {"post": post("fc.lines(fc.ax(), nd=4), _math(_lg(fc.ax().get_legend(), 'texts')['texts'][0], "
                                      "'\\\\sigma', '\\\\frac', 'e^{-z}')"),
                         "setup": f"z = np.linspace(-6, 6, {m})"}),
        (".set_ylabel", f"{start} Label the y axis with the gradient norm `$\\Vert\\nabla_\\theta\\Vert_2$` "
                        + chk.format("`\\Vert`, `\\nabla` and `\\theta`"),
         base + "ax.set_ylabel(r'$\\Vert\\nabla_\\theta\\Vert_2$')",
         {"post": post("_math(fc.ax().get_ylabel(), '\\\\Vert', '\\\\nabla', '\\\\theta')")}),
        ("set_xlabel", f"{start} Label the x axis `epoch (learning rate $\\eta = ...$)` with the value of the variable "
                       f"`lr` filled in by an f-string (combine `r` and `f`: `rf'...'`; it must contain `\\eta` and "
                       f"the value as `str(lr)` writes it).",
         base + "ax.set_xlabel(rf'epoch (learning rate $\\eta = {lr}$)')",
         {"post": post("_math(fc.ax().get_xlabel(), 'epoch (learning rate', '\\\\eta', str(lr))"),
          "setup": f"lr = {lr}"}),
        (".text", f"{start} Compute the correlation of `train_loss` and `val_loss` (`np.corrcoef(...)[0, 1]`) and write "
                  f"it as `$\\rho = 0.987$` (3 decimals) at (0.05, 0.9) in axes coordinates "
                  + chk.format("`\\rho` and the value"),
         base + "r = np.corrcoef(train_loss, val_loss)[0, 1]\nax.text(0.05, 0.9, rf'$\\rho = {r:.3f}$', transform=ax.transAxes)",
         {"post": post("_math(fc.ax().texts[0].get_text(), '\\\\rho', f'{np.corrcoef(train_loss, val_loss)[0, 1]:.3f}'), "
                       "_tx(fc.ax().texts[0], 'pos', 'frame')")}),
        ("label", "Create `fig, ax` and plot `train_loss ** p` against `epochs` for p = 1, 2, 3, labelled "
                  "`$\\mathcal{L}^1$`, `$\\mathcal{L}^2$`, `$\\mathcal{L}^3$` from a loop with an `rf`-string (literal "
                  "braces are doubled in f-strings: `{{L}}`), and add a legend.",
         "fig, ax = plt.subplots()\nfor p in (1, 2, 3):\n    ax.plot(epochs, train_loss ** p, label=rf'$\\mathcal{{L}}^{p}$')\n"
         "ax.legend()", {"post": post("_lg(fc.ax().get_legend(), 'texts'), [_math(t)['renders'] for t in "
                                      "_lg(fc.ax().get_legend(), 'texts')['texts']]")}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "easy")
def legend_readback(rng):
    code, desc, k, lrs = _runs(rng)
    t = rng.choice(["lr", "learning rate", "runs", "step size"])
    items = [
        ("get_legend_handles_labels", "Plot each run on `fig, ax` labelled `f'lr={lr}'`, plus `runs.mean(axis=0)` "
                                      "labelled '_mean' (labels starting with an underscore are skipped). Return the "
                                      "labels `ax.get_legend_handles_labels()` collects.",
         "fig, ax = plt.subplots()\nfor lr, r in zip(lrs, runs):\n    ax.plot(steps, r, label=f'lr={lr}')\n"
         "ax.plot(steps, runs.mean(axis=0), label='_mean')\nhandles, labels = ax.get_legend_handles_labels()\nresult = labels"),
        ("get_fontsize", "Plot the first run labelled 'run 0', add a legend with `fontsize='small'`, and return the "
                         "size in points of its text (`lg.get_texts()[0].get_fontsize()`; 'small' is a fraction of "
                         "the default 10 pt).",
         "fig, ax = plt.subplots()\nax.plot(steps, runs[0], label='run 0')\nlg = ax.legend(fontsize='small')\n"
         "result = lg.get_texts()[0].get_fontsize()"),
        ("legend_handles", "Draw a line (`runs[0]`, label 'line'), bars (`ax.bar(range(3), runs[0][:3], label='bars')`) "
                           "and a scatter (`runs[1]`, label 'points') on one Axes, add a legend, and return the type "
                           "names of its handles (`lg.legend_handles`) in order.",
         "fig, ax = plt.subplots()\nax.plot(steps, runs[0], label='line')\nax.bar(range(3), runs[0][:3], label='bars')\n"
         "ax.scatter(steps, runs[1], label='points')\nlg = ax.legend()\nresult = [type(h).__name__ for h in lg.legend_handles]"),
        ("get_legend", "Plot the first run labelled 'run 0' and return `[ax.get_legend() is None]` measured **before** "
                       "and **after** calling `ax.legend()` (a list of two bools).",
         "fig, ax = plt.subplots()\nax.plot(steps, runs[0], label='run 0')\nbefore = ax.get_legend() is None\nax.legend()\n"
         "result = [before, ax.get_legend() is None]"),
        ("get_title", f"Plot each run labelled `str(lr)`, add a legend titled {t!r}, and return `[the legend's title "
                      f"text, the number of entries]` (`lg.get_title().get_text()`, `len(lg.get_texts())`).",
         f"fig, ax = plt.subplots()\nfor lr, r in zip(lrs, runs):\n    ax.plot(steps, r, label=str(lr))\n"
         f"lg = ax.legend(title={t!r})\nresult = [lg.get_title().get_text(), len(lg.get_texts())]"),
        (".texts", "Plot the first run, annotate its first and last points with 'first' and 'last', write 'note' at "
                   "(0.5, 0.5) in axes coordinates, and return the texts of everything in `ax.texts` (annotations are "
                   "texts too).",
         "fig, ax = plt.subplots()\nax.plot(steps, runs[0])\nax.annotate('first', xy=(steps[0], runs[0][0]))\n"
         "ax.annotate('last', xy=(steps[-1], runs[0][-1]))\nax.text(0.5, 0.5, 'note', transform=ax.transAxes)\n"
         "result = [t.get_text() for t in ax.texts]"),
        ("to_hex", "Plot every run with the default colors and labels `f'lr={lr}'`, add a legend, and return each legend "
                   "handle's color as a hex string (`mpl.colors.to_hex(h.get_color())`).",
         "fig, ax = plt.subplots()\nfor lr, r in zip(lrs, runs):\n    ax.plot(steps, r, label=f'lr={lr}')\nlg = ax.legend()\n"
         "result = [mpl.colors.to_hex(h.get_color()) for h in lg.legend_handles]"),
        ("get_frame_on", "Plot each run labelled `f'lr={lr}'`, add a legend with `frameon=False`, and return "
                         "`[lg.get_frame_on(), the legend's texts]`.",
         "fig, ax = plt.subplots()\nfor lr, r in zip(lrs, runs):\n    ax.plot(steps, r, label=f'lr={lr}')\n"
         "lg = ax.legend(frameon=False)\nresult = [lg.get_frame_on(), [t.get_text() for t in lg.get_texts()]]"),
    ]
    return catalog(rng, items, code, f"{desc}.")


# ================================================================= medium

@template(M, "medium")
def annotated_training_curves(rng):
    code, desc, n = curves(rng, acc=True)
    a, b = rng.choice([("train", "val"), ("training", "validation"), ("fit", "holdout")])
    two = (f"fig, ax = plt.subplots()\nax.plot(epochs, train_loss, label={a!r})\n"
           f"ax.plot(epochs, val_loss, label={b!r})\n")
    both = f"plot `train_loss` and `val_loss` against `epochs` labelled {a!r} and {b!r}"
    c = rng.choice(["C3", "C1", "gray", "tab:red"])
    margin = rng.choice([0.03, 0.05, 0.1])
    items = [
        (".annotate", f"Create `fig, ax`, {both}, add a legend and the y label 'loss'. Annotate the best epoch (lowest "
                      f"`val_loss`) with `f'best: {{v:.3f}}'`, the text offset (30, 30) points, with an arrow `'->'`.",
         two + "ax.legend()\nax.set_ylabel('loss')\ni = val_loss.argmin()\nax.annotate(f'best: {val_loss[i]:.3f}', "
               "xy=(epochs[i], val_loss[i]), xytext=(30, 30), textcoords='offset points', arrowprops=dict(arrowstyle='->'))",
         {"post": post("fc.lines(fc.ax(), 'label'), _lg(fc.ax().get_legend(), 'texts'), fc.labels()['ylabel'], "
                       "_txs(fc.ax(), 'text', 'target', 'pos', 'frame', 'arrow')")}),
        ("axvline", f"Create `fig, ax`, {both}, and draw a dashed vertical line at the best epoch (lowest `val_loss`) "
                    f"labelled `f'best epoch ({{e}})'`; then add a legend (three entries).",
         two + "e = epochs[val_loss.argmin()]\nax.axvline(e, ls='--', label=f'best epoch ({e})')\nax.legend()",
         {"post": post("fc.lines(fc.ax(), 'ls', 'label'), _lg(fc.ax().get_legend(), 'texts')")}),
        ("bbox", f"Create `fig, ax` and {both}. Add a summary box in the upper-right corner: the text `f'best val "
                 f"{{v:.3f}} @ epoch {{e}}'` at axes position (0.97, 0.97), right- and top-aligned, in a rounded box "
                 f"(`bbox=dict(boxstyle='round', fc='w', ec='0.7')`).",
         two + "i = val_loss.argmin()\nax.text(0.97, 0.97, f'best val {val_loss[i]:.3f} @ epoch {epochs[i]}', "
               "transform=ax.transAxes, ha='right', va='top', bbox=dict(boxstyle='round', fc='w', ec='0.7'))",
         {"post": post("_txs(fc.ax(), 'text', 'pos', 'frame', 'ha', 'va', 'box')")}),
        ("arrowstyle", f"Create `fig, ax` and {both}. At the last epoch, show the generalization gap: a double-headed "
                       f"arrow (`arrowstyle='<->'`) from the train point to the val point (an annotation with empty "
                       f"text), then the text `f'gap {{g:.3f}}'` (g = val − train) at x = last epoch + 0.5, halfway "
                       f"between the two values, vertically centred.",
         two + "e, t, v = epochs[-1], train_loss[-1], val_loss[-1]\nax.annotate('', xy=(e, v), xytext=(e, t), "
               "arrowprops=dict(arrowstyle='<->'))\nax.text(e + 0.5, (t + v) / 2, f'gap {v - t:.3f}', va='center')",
         {"post": post("_txs(fc.ax(), 'text', 'pos', 'va', 'style')")}),
        ("Patch", f"Create `fig, ax`, {both}, and shade from the best epoch (lowest `val_loss`) to the last with "
                  f"`ax.axvspan(..., color={c!r}, alpha=0.15)`. Make a legend with the two curves plus a proxy "
                  f"`Patch(facecolor={c!r}, alpha=0.15, label='overfitting')`, in that order.",
         "from matplotlib.patches import Patch\n" + two + f"best = epochs[val_loss.argmin()]\nax.axvspan(best, epochs[-1], "
         f"color={c!r}, alpha=0.15)\nhandles, labels = ax.get_legend_handles_labels()\n"
         f"ax.legend(handles=handles + [Patch(facecolor={c!r}, alpha=0.15, label='overfitting')])",
         {"post": post("_lg(fc.ax().get_legend(), 'texts', 'handles', 'colors')")}),
        ("fig.legend", f"Create `fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3), layout='constrained')`: the "
                       f"losses on `ax1` (labels {a!r}, {b!r}), `val_acc` on `ax2` labelled 'accuracy', titles 'loss' "
                       f"and 'accuracy'. One figure legend below the panels: `loc='outside lower center'`, 3 columns.",
         f"fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3), layout='constrained')\n"
         f"ax1.plot(epochs, train_loss, label={a!r})\nax1.plot(epochs, val_loss, label={b!r})\nax1.set_title('loss')\n"
         f"ax2.plot(epochs, val_acc, label='accuracy')\nax2.set_title('accuracy')\n"
         f"fig.legend(loc='outside lower center', ncols=3)",
         {"post": post("[_lg(l, 'texts', 'outside', 'loc', 'ncols') for l in fc.fig().legends], "
                       "[fc.labels(a)['title'] for a in fc.axes()]"), "require": ["legend"]}),
        ("set_ylabel", "Create `fig, ax` and plot the two losses against `epochs` with the mathtext labels "
                       "`$\\mathcal{L}_{train}$` and `$\\mathcal{L}_{val}$`; add a legend, the x label `epoch $t$` and "
                       "the y label `$\\mathcal{L}$` (exactly these strings).",
         "fig, ax = plt.subplots()\nax.plot(epochs, train_loss, label=r'$\\mathcal{L}_{train}$')\n"
         "ax.plot(epochs, val_loss, label=r'$\\mathcal{L}_{val}$')\nax.legend()\nax.set_xlabel('epoch $t$')\n"
         "ax.set_ylabel(r'$\\mathcal{L}$')",
         {"post": post("_lg(fc.ax().get_legend(), 'texts'), fc.labels(), [_math(t)['renders'] for t in "
                       "(fc.ax().get_xlabel(), fc.ax().get_ylabel())]"), "require": ["set_xlabel", "set_ylabel"]}),
        ("twinx", f"Create `fig, ax`, {both}, then plot `val_acc` labelled 'accuracy' on a twin y-axis (`ax2 = "
                  f"ax.twinx()`, color 'C2'). Show **one** legend on `ax`, at 'center right', listing all three lines: "
                  f"combine both Axes' `get_legend_handles_labels()`.",
         two + "ax2 = ax.twinx()\nax2.plot(epochs, val_acc, color='C2', label='accuracy')\n"
               "h1, l1 = ax.get_legend_handles_labels()\nh2, l2 = ax2.get_legend_handles_labels()\n"
               "ax.legend(h1 + h2, l1 + l2, loc='center right')",
         {"post": post("_lg(fc.ax(0).get_legend(), 'texts', 'loc'), fc.ax(1).get_legend()"),
          "require": ["get_legend_handles_labels"]}),
        (".annotate", f"Create `fig, ax` and plot `val_acc` against `epochs`. With `thr = round(val_acc.max() - "
                      f"{margin}, 2)`, annotate the **first** epoch where `val_acc >= thr` with `f'≥ {{thr}}'`, the "
                      f"text offset (-40, -30) points, with an arrow `'->'`.",
         f"fig, ax = plt.subplots()\nax.plot(epochs, val_acc)\nthr = round(val_acc.max() - {margin}, 2)\n"
         f"i = np.argmax(val_acc >= thr)\nax.annotate(f'≥ {{thr}}', xy=(epochs[i], val_acc[i]), xytext=(-40, -30), "
         f"textcoords='offset points', arrowprops=dict(arrowstyle='->'))",
         {"post": post("_txs(fc.ax(), 'text', 'target', 'pos', 'frame', 'arrow')")}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "medium")
def class_scatter_legends(rng):
    code, desc, k, names = classes(rng)
    title = rng.choice(["class", "label", "category"])
    dy = rng.choice([-0.12, -0.15, -0.1])
    loop = f"fig, ax = plt.subplots()\nfor c, name in enumerate(class_names):\n    m = labels == c\n"
    items = [
        ("scatter", f"Draw one `ax.scatter` per class on `fig, ax`, labelled with its class name, and add a legend "
                    f"titled {title!r}.",
         loop + f"    ax.scatter(pts[m, 0], pts[m, 1], label=name)\nax.legend(title={title!r})",
         {"post": post(f"[fc.scatter(fc.ax(), i)['offsets'] for i in range({k})], "
                       f"_lg(fc.ax().get_legend(), 'texts', 'title')")}),
        ("legend_elements", "Draw **one** scatter of all points colored by class (`sc = ax.scatter(..., c=labels)`). "
                            "Build the legend from `sc.legend_elements()`: keep its handles but use `class_names` as "
                            "the labels.",
         "fig, ax = plt.subplots()\nsc = ax.scatter(pts[:, 0], pts[:, 1], c=labels)\nhandles, _ = sc.legend_elements()\n"
         "ax.legend(handles, class_names)",
         {"post": post("fc.scatter(fc.ax(), 0, 6, 'array')['array'], _lg(fc.ax().get_legend(), 'texts')")}),
        ("markerscale", "Draw one scatter per class with small markers (`s=10`), labelled by class name. Put the legend "
                        "outside on the right (center-left point at (1.01, 0.5) in axes coordinates) with markers twice "
                        "as large as in the plot (`markerscale=2`).",
         loop + "    ax.scatter(pts[m, 0], pts[m, 1], s=10, label=name)\n"
                "ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), markerscale=2)",
         {"post": post("_lg(fc.ax().get_legend(), 'texts', 'loc', 'anchor', 'markerscale')")}),
        (".text", "Scatter all points (`ax.scatter(pts[:, 0], pts[:, 1], c=labels)`) and write each class name at its "
                  "class centroid (the mean of its points), in bold, centred both ways; classes in order 0, 1, ….",
         "fig, ax = plt.subplots()\nax.scatter(pts[:, 0], pts[:, 1], c=labels)\nfor c, name in enumerate(class_names):\n"
         "    cx, cy = pts[labels == c].mean(axis=0)\n    ax.text(cx, cy, name, fontweight='bold', ha='center', va='center')",
         {"post": post("_txs(fc.ax(), 'text', 'pos', 'bold', 'ha', 'va')")}),
        (".annotate", "Scatter all points (`c=labels`) and annotate the point **farthest from its own class centroid** "
                      "with 'outlier': text offset (15, 15) points, arrow `'->'`.",
         "fig, ax = plt.subplots()\nax.scatter(pts[:, 0], pts[:, 1], c=labels)\n"
         "cent = np.array([pts[labels == c].mean(axis=0) for c in range(len(class_names))])\n"
         "i = np.argmax(np.linalg.norm(pts - cent[labels], axis=1))\nax.annotate('outlier', xy=pts[i], xytext=(15, 15), "
         "textcoords='offset points', arrowprops=dict(arrowstyle='->'))",
         {"post": post("_txs(fc.ax(), 'text', 'target', 'pos', 'frame', 'arrow')")}),
        ("Line2D", "Scatter all points in **one** call colored by class with `c=[f'C{l}' for l in labels]` (no labels). "
                   "Then build the legend from proxies, one per class: `Line2D([0], [0], marker='o', ls='', "
                   "color=f'C{i}', label=name)`.",
         "from matplotlib.lines import Line2D\nfig, ax = plt.subplots()\n"
         "ax.scatter(pts[:, 0], pts[:, 1], c=[f'C{l}' for l in labels])\n"
         "handles = [Line2D([0], [0], marker='o', ls='', color=f'C{i}', label=name) for i, name in enumerate(class_names)]\n"
         "ax.legend(handles=handles)", {"post": post("_lg(fc.ax().get_legend(), 'texts', 'handles', 'colors')")}),
        ("label", "Draw one scatter per class with legend entries that include the class size: `f'{name} (n={count})'`.",
         loop + "    ax.scatter(pts[m, 0], pts[m, 1], label=f'{name} (n={m.sum()})')\nax.legend()",
         {"post": post("_lg(fc.ax().get_legend(), 'texts')")}),
        ("bbox_to_anchor", f"Draw one scatter per class labelled by name, and put the legend **below** the Axes in one "
                           f"row: its upper-center point at (0.5, {dy}) in axes coordinates, {k} columns, no frame.",
         loop + f"    ax.scatter(pts[m, 0], pts[m, 1], label=name)\n"
                f"ax.legend(loc='upper center', bbox_to_anchor=(0.5, {dy}), ncols={k}, frameon=False)",
         {"post": post("_lg(fc.ax().get_legend(), 'texts', 'loc', 'anchor', 'ncols', 'frame')")}),
        ("marker", "Draw one scatter per class labelled by name, then the class centroids as black crosses in one more "
                   "scatter (`marker='x', c='k'`) labelled 'centroid', and add a legend.",
         loop + "    ax.scatter(pts[m, 0], pts[m, 1], label=name)\n"
                "cent = np.array([pts[labels == c].mean(axis=0) for c in range(len(class_names))])\n"
                "ax.scatter(cent[:, 0], cent[:, 1], marker='x', c='k', label='centroid')\nax.legend()",
         {"post": post(f"fc.scatter(fc.ax(), {k}, 4)['offsets'], _lg(fc.ax().get_legend(), 'texts')")}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "medium")
def bar_value_labels(rng):
    code, desc, names, vals, th, m = categories(rng, k=(4, 6))
    thr = round((min(vals) + max(vals)) / 2, 2)
    hi, lo = rng.choice([("C2", "C7"), ("tab:green", "tab:gray"), ("C0", "0.7")])
    B = "fig, ax = plt.subplots()\nbars = ax.bar(names, values)\n"
    items = [
        (".annotate", "Draw `ax.bar(names, values)` and label each bar with its value using `ax.annotate`: the point is "
                      "the top-center of the bar (i, v), the text 3 points above it (offset points), centred and "
                      "bottom-aligned.",
         B + "for i, v in enumerate(values):\n    ax.annotate(str(v), xy=(i, v), xytext=(0, 3), textcoords='offset points', "
             "ha='center', va='bottom')", {"post": post("_txs(fc.ax(), 'text', 'target', 'pos', 'frame', 'ha', 'va')")}),
        ("Patch", f"Draw `ax.bar(names, values, color=...)` with bars colored {hi!r} when the value is ≥ {thr} and {lo!r} "
                  f"otherwise. Add a legend with two proxy patches: `Patch(facecolor={hi!r}, label='above')` and "
                  f"`Patch(facecolor={lo!r}, label='below')`.",
         f"from matplotlib.patches import Patch\nfig, ax = plt.subplots()\n"
         f"ax.bar(names, values, color=[{hi!r} if v >= {thr} else {lo!r} for v in values])\n"
         f"ax.legend(handles=[Patch(facecolor={hi!r}, label='above'), Patch(facecolor={lo!r}, label='below')])",
         {"post": post("[b['color'] for b in fc.bars(fc.ax(), 6, 'color')], _lg(fc.ax().get_legend(), 'texts', 'colors')")}),
        ("barh", "Draw horizontal bars sorted ascending (so the largest ends up at the top): `ax.barh` with the names "
                 "and values in ascending order of value. Write each value at the end of its bar: at (v, row), "
                 "left-aligned, vertically centred.",
         "fig, ax = plt.subplots()\norder = np.argsort(values)\nax.barh([names[i] for i in order], [values[i] for i in order])\n"
         "for row, i in enumerate(order):\n    ax.text(values[i], row, str(values[i]), ha='left', va='center')",
         {"post": post("fc.widths(), fc.tick_labels(fc.ax(), 'y'), _txs(fc.ax(), 'text', 'pos', 'ha', 'va')")}),
        ("arrowprops", "Draw `ax.bar(names, values)` and annotate the **shortest** bar with 'lowest': the arrow points at "
                       "the top-center of the bar, the text 30 points above (offset points), centred, with a red arrow "
                       "(`arrowprops=dict(arrowstyle='->', color='red')`).",
         B + "i = int(np.argmin(values))\nax.annotate('lowest', xy=(i, values[i]), xytext=(0, 30), textcoords='offset points', "
             "ha='center', arrowprops=dict(arrowstyle='->', color='red'))",
         {"post": post("_txs(fc.ax(), 'text', 'target', 'pos', 'frame', 'arrowcolor')")}),
        ("axhline", f"Draw `ax.bar(names, values)` (no label) and a dashed target line `ax.axhline({thr}, ls='--', "
                    f"label='target ({thr})')`; add a legend (only the line is labelled), and write `f'{{k}} of {{n}} "
                    f"above target'` (values strictly above {thr}) at axes position (0.02, 0.95), top-aligned.",
         B + f"ax.axhline({thr}, ls='--', label='target ({thr})')\nax.legend()\nk = sum(v > {thr} for v in values)\n"
             f"ax.text(0.02, 0.95, f'{{k}} of {{len(values)}} above target', transform=ax.transAxes, va='top')",
         {"post": post("_lg(fc.ax().get_legend(), 'texts'), _txs(fc.ax(), 'text', 'pos', 'frame', 'va')")}),
        (".text", "Plot each value's difference from the mean: `d = np.array(values) - np.mean(values)`, `ax.bar(names, "
                  "d)`. Write each difference with a sign and one decimal (`f'{x:+.1f}'`), centred on its bar: at the "
                  "bar's end, **above** positive bars (va='bottom') and **below** negative ones (va='top').",
         "fig, ax = plt.subplots()\nd = np.array(values) - np.mean(values)\nax.bar(names, d)\nfor i, x in enumerate(d):\n"
         "    ax.text(i, x, f'{x:+.1f}', ha='center', va='bottom' if x >= 0 else 'top')",
         {"post": post("fc.heights(nd=4), _txs(fc.ax(), 'text', 'pos', 'ha', 'va')")}),
        (".set_title", "Draw `ax.bar(names, values)` and title it with the mean and standard deviation in mathtext: "
                       "`rf'$\\mu = {mean:.1f},\\ \\sigma = {std:.1f}$'` (`np.mean`, `np.std`).",
         B + "ax.set_title(rf'$\\mu = {np.mean(values):.1f},\\ \\sigma = {np.std(values):.1f}$')",
         {"post": post("_math(fc.ax().get_title(), '\\\\mu', '\\\\sigma', f'{np.mean(values):.1f}', f'{np.std(values):.1f}')")}),
        ("transAxes", "Create `fig, (ax1, ax2) = plt.subplots(1, 2)`: bars of `values` on the left, the values sorted "
                      "descending as a line on the right. Label the panels '(a)' and '(b)' in bold at axes position "
                      "(-0.1, 1.05) of each.",
         "fig, (ax1, ax2) = plt.subplots(1, 2)\nax1.bar(names, values)\nax2.plot(sorted(values, reverse=True))\n"
         "for a, tag in zip((ax1, ax2), ('(a)', '(b)')):\n    a.text(-0.1, 1.05, tag, transform=a.transAxes, fontweight='bold')",
         {"post": post("[_txs(a, 'text', 'pos', 'frame', 'bold') for a in fc.axes()], fc.lines(fc.ax(1))")}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "medium")
def multi_legend_layouts(rng):
    code, desc, k, lrs = _runs(rng)
    cut = sorted(lrs)[len(lrs) // 2]
    loc1, loc2 = rng.choice([("upper right", "lower left"), ("upper center", "lower right"), ("center right", "upper left")])
    loop = "fig, ax = plt.subplots()\nfor lr, r in zip(lrs, runs):\n"
    LEGS = "[_lg(l, 'texts', 'loc') for l in _legends(fc.ax())]"
    items = [
        ("add_artist", "Plot every run labelled `f'lr={lr}'`. Show two legends: the runs at 'upper left' (keep it with "
                       "`first = ax.legend(...)` and `ax.add_artist(first)`), and a second one at 'upper right' made of "
                       "proxies: `Line2D([0], [0], color='k', ls='--', label='baseline')` and `Patch(facecolor='C2', "
                       "alpha=0.3, label='region')`.",
         "from matplotlib.lines import Line2D\nfrom matplotlib.patches import Patch\n" + loop +
         "    ax.plot(steps, r, label=f'lr={lr}')\nfirst = ax.legend(loc='upper left')\nax.add_artist(first)\n"
         "custom = [Line2D([0], [0], color='k', ls='--', label='baseline'), Patch(facecolor='C2', alpha=0.3, label='region')]\n"
         "ax.legend(handles=custom, loc='upper right')", {"post": post(LEGS)}),
        ("fig.legend", "Create `fig, (ax1, ax2) = plt.subplots(1, 2, layout='constrained')`: the first two runs on `ax1`, "
                       "the others on `ax2`, each labelled `f'lr={lr}'`. Add one figure legend with **all** runs: "
                       "`loc='outside lower center'`, one column per run.",
         "fig, (ax1, ax2) = plt.subplots(1, 2, layout='constrained')\nfor i, (lr, r) in enumerate(zip(lrs, runs)):\n"
         "    (ax1 if i < 2 else ax2).plot(steps, r, label=f'lr={lr}')\nfig.legend(loc='outside lower center', ncols=len(lrs))",
         {"post": post("[_lg(l, 'texts', 'outside', 'ncols') for l in fc.fig().legends], [len(a.get_lines()) for a in fc.axes()]"),
          "require": ["legend"]}),
        ("add_artist", "Two models, two splits: plot `runs[0]` (color 'C0', solid), `runs[0] + 0.1` (C0, dashed), "
                       "`runs[1]` (C1, solid), `runs[1] + 0.1` (C1, dashed), no labels. Explain them with two proxy "
                       "legends: line styles at 'upper right' (black solid 'train', black dashed 'val') and colors at "
                       "'upper center' (C0 'model A', C1 'model B'); keep the first with `ax.add_artist`.",
         "from matplotlib.lines import Line2D\nfig, ax = plt.subplots()\nfor c, r in (('C0', runs[0]), ('C1', runs[1])):\n"
         "    ax.plot(steps, r, color=c)\n    ax.plot(steps, r + 0.1, color=c, ls='--')\n"
         "styles = ax.legend(handles=[Line2D([0], [0], color='k', label='train'), Line2D([0], [0], color='k', ls='--', "
         "label='val')], loc='upper right')\nax.add_artist(styles)\nax.legend(handles=[Line2D([0], [0], color='C0', "
         "label='model A'), Line2D([0], [0], color='C1', label='model B')], loc='upper center')",
         {"post": post("[_lg(l, 'texts', 'loc', 'ls', 'colors') for l in _legends(fc.ax())], fc.lines(fc.ax(), 'color', 'ls')")}),
        ("dict", f"Plot every run in a loop, colored and labelled by group: 'C0' / 'small lr' when `lr < {cut}`, else "
                 f"'C1' / 'large lr'. The legend must list each group **once**, in order of first appearance: "
                 f"`dict(zip(labels, handles))` drops duplicates.",
         loop + f"    small = lr < {cut}\n    ax.plot(steps, r, color='C0' if small else 'C1', "
                f"label='small lr' if small else 'large lr')\nhandles, labels = ax.get_legend_handles_labels()\n"
                f"by_label = dict(zip(labels, handles))\nax.legend(by_label.values(), by_label.keys())",
         {"post": post("_lg(fc.ax().get_legend(), 'texts', 'colors'), fc.lines(fc.ax(), 'color', data=False)")}),
        ("argsort", "Plot every run labelled `f'lr={lr}'`, and order the legend entries by **final** loss, lowest first "
                    "(reorder the handles and labels from `ax.get_legend_handles_labels()`).",
         loop + "    ax.plot(steps, r, label=f'lr={lr}')\nhandles, labels = ax.get_legend_handles_labels()\n"
                "order = np.argsort(runs[:, -1])\nax.legend([handles[i] for i in order], [labels[i] for i in order])",
         {"post": post("_lg(fc.ax().get_legend(), 'texts')")}),
        ("loc", f"Create `fig, (top, bottom) = plt.subplots(2, 1, sharex=True)`: `runs[0]` and `runs[1]` on `top` "
                f"(labels 'a', 'b'), `runs[-1]` on `bottom` (label 'c'). Legend at {loc1!r} on top, {loc2!r} at the "
                f"bottom.",
         f"fig, (top, bottom) = plt.subplots(2, 1, sharex=True)\ntop.plot(steps, runs[0], label='a')\n"
         f"top.plot(steps, runs[1], label='b')\nbottom.plot(steps, runs[-1], label='c')\ntop.legend(loc={loc1!r})\n"
         f"bottom.legend(loc={loc2!r})",
         {"post": post("[_lg(a.get_legend(), 'texts', 'loc') for a in fc.axes()]")}),
        ("outside", "Create `fig, ax = plt.subplots(layout='constrained')`, plot every run labelled `f'lr={lr}'`, and "
                    "place a **figure** legend outside to the right, aligned with the top: `fig.legend(loc='outside "
                    "right upper')`.",
         "fig, ax = plt.subplots(layout='constrained')\nfor lr, r in zip(lrs, runs):\n    ax.plot(steps, r, label=f'lr={lr}')\n"
         "fig.legend(loc='outside right upper')",
         {"post": post("[_lg(l, 'texts', 'outside', 'loc') for l in fc.fig().legends]"), "require": ["legend"]}),
        ("ncols", "Plot every run labelled `str(lr)` and put the legend above the plot as one row: lower-center point at "
                  "(0.5, 1.0) in axes coordinates, one column per run, no frame, title 'learning rate'.",
         loop + "    ax.plot(steps, r, label=str(lr))\nax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), "
                "ncols=len(lrs), frameon=False, title='learning rate')",
         {"post": post("_lg(fc.ax().get_legend(), 'texts', 'loc', 'anchor', 'ncols', 'frame', 'title')")}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "medium")
def formula_figures(rng):
    m = rng.choice([60, 80, 100])
    w0, c0 = rng.choice([(1.0, 0.5), (-0.5, 1.0), (2.0, 0.2), (0.5, 0.0)])
    mu, sd = rng.choice([(0, 1), (1, 0.5), (-1, 2), (2, 1.5)])
    s = seeded(rng)
    betas = rng.choice([[0.5, 0.9, 0.99], [0.8, 0.9], [0.9, 0.95, 0.99]])
    items = [
        ("label", "On `fig, ax` plot three activations of `z` labelled exactly `$\\sigma(z)$` (sigmoid), "
                  "`$\\tanh(z)$` and `$\\max(0, z)$` (ReLU), add a legend, and title it `activations $f(z)$`.",
         "fig, ax = plt.subplots()\nax.plot(z, 1 / (1 + np.exp(-z)), label=r'$\\sigma(z)$')\n"
         "ax.plot(z, np.tanh(z), label=r'$\\tanh(z)$')\nax.plot(z, np.maximum(0, z), label=r'$\\max(0, z)$')\nax.legend()\n"
         "ax.set_title('activations $f(z)$')",
         {"post": post("fc.lines(fc.ax(), nd=4), _lg(fc.ax().get_legend(), 'texts'), fc.labels()['title']")}),
        (".annotate", f"Plot the loss `(z - {w0}) ** 2 + {c0}` against `z` on `fig, ax` and annotate its minimum (at z = "
                      f"{w0}) with the mathtext `$\\nabla \\mathcal{{L}} = 0$`, the text 2 above the point, with an "
                      f"arrow `'->'`.",
         f"fig, ax = plt.subplots()\nax.plot(z, (z - {w0}) ** 2 + {c0})\nax.annotate(r'$\\nabla \\mathcal{{L}} = 0$', "
         f"xy=({w0}, {c0}), xytext=({w0}, {c0} + 2), arrowprops=dict(arrowstyle='->'))",
         {"post": post("_math(fc.ax().texts[0].get_text(), '\\\\nabla', '\\\\mathcal{L}'), "
                       "_tx(fc.ax().texts[0], 'target', 'pos', 'arrow')")}),
        ("polyfit", "Fit a line to the points `(px, py)` with `a, b = np.polyfit(px, py, 1)`. On `fig, ax` draw the "
                    "points as dots (`'o'`) and the fitted line `a * px + b`, and title it exactly "
                    "`f'$y = {a:.2f}x + {b:.2f}$'`.",
         "fig, ax = plt.subplots()\na, b = np.polyfit(px, py, 1)\nax.plot(px, py, 'o')\nax.plot(px, a * px + b)\n"
         "ax.set_title(f'$y = {a:.2f}x + {b:.2f}$')",
         {"post": post("fc.lines(fc.ax(), 'marker', nd=4), fc.labels()['title'], _math(fc.ax().get_title())['renders']"),
          "setup": f"g = np.random.default_rng({s})\npx = np.round(g.uniform(0, 5, 12), 2)\n"
                   f"py = np.round(1.5 * px - 2 + g.normal(0, 0.5, 12), 2)"}),
        (".text", f"Plot the normal density with μ = {mu}, σ = {sd} (`mu`, `sd` are given) against `z` on `fig, ax`, and "
                  f"write its formula `$\\frac{{1}}{{\\sigma\\sqrt{{2\\pi}}}} e^{{-\\frac{{(x-\\mu)^2}}{{2\\sigma^2}}}}$` "
                  f"at axes position (0.02, 0.95), top-aligned (checked: mathtext that renders and uses `\\frac`, "
                  f"`\\sqrt`, `\\pi`, `\\mu` and `\\sigma`).",
         "fig, ax = plt.subplots()\nax.plot(z, np.exp(-(z - mu) ** 2 / (2 * sd ** 2)) / (sd * np.sqrt(2 * np.pi)))\n"
         "ax.text(0.02, 0.95, r'$\\frac{1}{\\sigma\\sqrt{2\\pi}} e^{-\\frac{(x-\\mu)^2}{2\\sigma^2}}$', "
         "transform=ax.transAxes, va='top')",
         {"post": post("fc.lines(fc.ax(), nd=4), _math(fc.ax().texts[0].get_text(), '\\\\frac', '\\\\sqrt', '\\\\pi', "
                       "'\\\\mu', '\\\\sigma'), _tx(fc.ax().texts[0], 'pos', 'frame', 'va')"),
          "setup": f"mu, sd = {mu}, {sd}"}),
        (".text", "On `fig, ax` (nothing plotted), write the momentum update as **one** text of two mathtext lines "
                  "joined by a newline: `$m_t = \\beta m_{t-1} + g_t$` and `$\\theta_t = \\theta_{t-1} - \\eta m_t$`, "
                  "at axes position (0.5, 0.5), centred both ways.",
         "fig, ax = plt.subplots()\nax.text(0.5, 0.5, '$m_t = \\\\beta m_{t-1} + g_t$\\n$\\\\theta_t = \\\\theta_{t-1} "
         "- \\\\eta m_t$', transform=ax.transAxes, ha='center', va='center')",
         {"post": post("_math(fc.ax().texts[0].get_text(), '\\\\beta', '\\\\theta_t', '\\\\eta', chr(10)), "
                       "_tx(fc.ax().texts[0], 'pos', 'frame', 'ha', 'va')")}),
        ("label", f"Adam's bias correction divides by `1 - beta ** t`. For each `beta` in `betas` plot `1 - beta ** t` "
                  f"against `t = np.arange(1, 51)` labelled `rf'$\\beta = {{beta}}$'`, and give the legend the "
                  f"mathtext title `$1 - \\beta^t$`.",
         "fig, ax = plt.subplots()\nt = np.arange(1, 51)\nfor beta in betas:\n    ax.plot(t, 1 - beta ** t, label=rf'$\\beta = {beta}$')\n"
         "ax.legend(title=r'$1 - \\beta^t$')",
         {"post": post("fc.lines(fc.ax(), nd=5), _lg(fc.ax().get_legend(), 'texts', 'title')"),
          "setup": f"betas = {betas}"}),
        (".set_title", "Plot `z ** 2` on `fig, ax` and give it a two-line title: 'Ridge regression' on the first line and "
                       "the objective `$\\min_w \\Vert Xw - y\\Vert^2 + \\lambda \\Vert w\\Vert^2$` on the second "
                       "(checked: renders, has a newline, uses `\\min_w`, `\\Vert` and `\\lambda`).",
         "fig, ax = plt.subplots()\nax.plot(z, z ** 2)\n"
         "ax.set_title('Ridge regression\\n$\\\\min_w \\\\Vert Xw - y\\\\Vert^2 + \\\\lambda \\\\Vert w\\\\Vert^2$')",
         {"post": post("_math(fc.ax().get_title(), 'Ridge regression' + chr(10), '\\\\min_w', '\\\\Vert', '\\\\lambda')")}),
        ("fontsize", "Plot `np.sin(z)` on `fig, ax`; title `$y = \\sin(x)$` at font size 14 and x label `$x_i$` at font "
                     "size 12.",
         "fig, ax = plt.subplots()\nax.plot(z, np.sin(z))\nax.set_title(r'$y = \\sin(x)$', fontsize=14)\n"
         "ax.set_xlabel('$x_i$', fontsize=12)",
         {"post": post("fc.labels(), fc.ax().title.get_fontsize(), fc.ax().xaxis.label.get_fontsize()")}),
    ]
    return catalog(rng, items, f"z = np.linspace(-4, 4, {m})", f"`z` holds {m} points from -4 to 4.")


# ================================================================= hard

@template(M, "hard")
def annotation_helpers(rng):
    code, desc, n = curves(rng, acc=True)
    dx = rng.choice([0.5, 1, 2])
    pad = rng.choice([0.02, 0.03, 0.05])
    TX = "_txs(_a, 'text', 'target', 'pos', 'frame', 'arrow')"
    items = [
        (".annotate", "Write `annotate_min(ax, x, y, fmt='{:.3f}')`: annotate the minimum of `y` on `ax` with "
                      "`fmt.format(value)`, the text offset (10, 20) points from the point, with an arrow `'->'`, and "
                      "return the annotation. (Checked by calling it on fresh Axes with and without `fmt`.)",
         "def annotate_min(ax, x, y, fmt='{:.3f}'):\n    i = int(np.argmin(y))\n    return ax.annotate(fmt.format(y[i]), "
         "xy=(x[i], y[i]), xytext=(10, 20), textcoords='offset points', arrowprops=dict(arrowstyle='->'))",
         {"post": post("_f, (_a, _b) = plt.subplots(1, 2)\n_a.plot(epochs, val_loss)\n_r = annotate_min(_a, epochs, val_loss)\n"
                       "_b.plot(epochs, train_loss)\nannotate_min(_b, epochs, train_loss, fmt='min {:.2f}')\n"
                       f"({TX}, _txs(_b, 'text', 'target'), type(_r).__name__)")}),
        (".text", "Write `add_value_labels(ax, bars, fmt='{:g}')`: for each bar of the container `bars` (what `ax.bar` "
                  "returns), write `fmt.format(height)` at the top-center of the bar (x = `b.get_x() + b.get_width() / "
                  "2`, y = height), centred and bottom-aligned. Return the list of texts.",
         "def add_value_labels(ax, bars, fmt='{:g}'):\n    out = []\n    for b in bars:\n        h = b.get_height()\n"
         "        out.append(ax.text(b.get_x() + b.get_width() / 2, h, fmt.format(h), ha='center', va='bottom'))\n"
         "    return out",
         {"post": post("_f, (_a, _b) = plt.subplots(1, 2)\n_r = add_value_labels(_a, _a.bar(['a', 'b', 'c'], [3, 1.5, 2]), "
                       "fmt='{:.1f}')\nadd_value_labels(_b, _b.bar([0, 1, 2], val_loss[:3], width=0.5))\n"
                       "(_txs(_a, 'text', 'pos', 'ha', 'va'), _txs(_b, 'text', 'pos'), len(_r))")}),
        ("bbox_to_anchor", "Write `legend_outside(ax, side='right')` returning the legend it makes: for 'right', centred "
                           "vertically just outside the right edge (`loc='center left'`, anchor (1.02, 0.5)); for "
                           "'top', just above the Axes (`loc='lower center'`, anchor (0.5, 1.02)) with one column per "
                           "entry.",
         "def legend_outside(ax, side='right'):\n    if side == 'right':\n"
         "        return ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))\n"
         "    n = len(ax.get_legend_handles_labels()[1])\n    return ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncols=n)",
         {"post": post("_f, (_a, _b) = plt.subplots(1, 2)\nfor _x in (_a, _b):\n    _x.plot(epochs, train_loss, label='train')\n"
                       "    _x.plot(epochs, val_loss, label='val')\n_b.plot(epochs, val_acc, label='acc')\n"
                       "_r = legend_outside(_a)\nlegend_outside(_b, 'top')\n"
                       "[_lg(_x.get_legend(), 'texts', 'loc', 'anchor', 'ncols') for _x in (_a, _b)], _r is _a.get_legend()")}),
        (".get_lines", f"Inline labels instead of a legend: write `label_lines(ax, dx={dx})` that, for every line on "
                       f"`ax` whose label doesn't start with '_', writes the label at (last x + dx, last y) in the "
                       f"line's color, vertically centred.",
         f"def label_lines(ax, dx={dx}):\n    for line in ax.get_lines():\n        label = line.get_label()\n"
         f"        if label.startswith('_'):\n            continue\n        x, y = line.get_xdata()[-1], line.get_ydata()[-1]\n"
         f"        ax.text(x + dx, y, label, color=line.get_color(), va='center')",
         {"post": post("_f, _a = plt.subplots()\n_a.plot(epochs, train_loss, label='train')\n_a.plot(epochs, val_loss, "
                       "label='val')\n_a.plot(epochs, val_acc)\n_a.plot(epochs, val_acc / 2, label='half', color='k')\n"
                       "label_lines(_a)\nlabel_lines(_a, dx=3)\n_txs(_a, 'text', 'pos', 'color', 'va'), _a.get_legend()"),
          "require": ["get_lines", "get_label"]}),
        ("axvline", "Write `mark_best(ax, x, y, mode='min')`: find the best index (lowest `y` for 'min', highest for "
                    "'max'), draw a dashed vertical line there (`ax.axvline(x[i], ls='--')`), annotate the point with "
                    "'best' offset (10, 10) points, and return the index as an int.",
         "def mark_best(ax, x, y, mode='min'):\n    i = int(np.argmin(y) if mode == 'min' else np.argmax(y))\n"
         "    ax.axvline(x[i], ls='--')\n    ax.annotate('best', xy=(x[i], y[i]), xytext=(10, 10), textcoords='offset points')\n"
         "    return i",
         {"post": post("_f, (_a, _b) = plt.subplots(1, 2)\n_i = mark_best(_a, epochs, val_loss)\n"
                       "_j = mark_best(_b, epochs, val_acc, mode='max')\n"
                       "(_i, _j, [fc.lines(_x, 'ls') for _x in (_a, _b)], [_txs(_x, 'text', 'target', 'pos', 'frame') "
                       "for _x in (_a, _b)])")}),
        ("transAxes", f"Write `add_note(ax, text, corner='upper left', pad={pad})`: corners are 'upper left', 'upper "
                      f"right', 'lower left', 'lower right'. Put the text `pad` in from that corner in axes "
                      f"coordinates, aligned towards the corner (e.g. 'upper right' → ha='right', va='top'), in a "
                      f"rounded box (`bbox=dict(boxstyle='round', fc='w')`). Return the Text.",
         f"def add_note(ax, text, corner='upper left', pad={pad}):\n    v, h = corner.split()\n"
         f"    x = pad if h == 'left' else 1 - pad\n    y = pad if v == 'lower' else 1 - pad\n"
         f"    return ax.text(x, y, text, transform=ax.transAxes, ha=h, va='bottom' if v == 'lower' else 'top',\n"
         f"                   bbox=dict(boxstyle='round', fc='w'))",
         {"post": post("_f, _a = plt.subplots()\nfor _c in ('upper left', 'upper right', 'lower left', 'lower right'):\n"
                       "    add_note(_a, _c, _c)\nadd_note(_a, 'default')\n_txs(_a, 'text', 'pos', 'frame', 'ha', 'va', 'box')")}),
        ("fig.legend", "Write `combined_legend(fig, **kw)`: one figure legend for all of the figure's Axes, each label "
                       "once (first handle wins), in order of first appearance; pass `**kw` on to `fig.legend` and "
                       "return the legend.",
         "def combined_legend(fig, **kw):\n    seen = {}\n    for ax in fig.axes:\n"
         "        for h, l in zip(*ax.get_legend_handles_labels()):\n            seen.setdefault(l, h)\n"
         "    return fig.legend(list(seen.values()), list(seen.keys()), **kw)",
         {"post": post("_f, (_a, _b) = plt.subplots(1, 2)\n_a.plot(epochs, train_loss, label='train')\n"
                       "_a.plot(epochs, val_loss, label='val')\n_b.plot(epochs, train_loss / 2, label='train', color='k')\n"
                       "_b.plot(epochs, val_acc, label='acc')\n_r = combined_legend(_f, loc='upper center', ncols=3)\n"
                       "[_lg(l, 'texts', 'loc', 'ncols', 'colors') for l in _f.legends], _r in _f.legends"),
          "require": ["legend", "get_legend_handles_labels"]}),
        ("dict", "Write `dedupe_legend(ax, **kw)`: a legend for `ax` where every label appears once, in order of first "
                 "appearance (`dict(zip(labels, handles))`); pass `**kw` to `ax.legend` and return the legend.",
         "def dedupe_legend(ax, **kw):\n    handles, labels = ax.get_legend_handles_labels()\n"
         "    by = dict(zip(labels, handles))\n    return ax.legend(by.values(), by.keys(), **kw)",
         {"post": post("_f, _a = plt.subplots()\nfor _i, _l in enumerate(['a', 'b', 'a', 'c', 'b']):\n"
                       "    _a.plot(epochs, train_loss + _i, label=_l)\n_r = dedupe_legend(_a, loc='lower left')\n"
                       "_lg(_a.get_legend(), 'texts', 'loc'), _r is _a.get_legend()")}),
        (".annotate", "Write `annotate_points(ax, x, y, labels, offset=(5, 5))`: annotate point i with `labels[i]` "
                      "offset by `offset` points, **skipping** points whose label is None; return how many were "
                      "annotated.",
         "def annotate_points(ax, x, y, labels, offset=(5, 5)):\n    k = 0\n    for xi, yi, lab in zip(x, y, labels):\n"
         "        if lab is None:\n            continue\n        ax.annotate(lab, xy=(xi, yi), xytext=offset, "
         "textcoords='offset points')\n        k += 1\n    return k",
         {"post": post("_f, _a = plt.subplots()\n_k = annotate_points(_a, epochs[:4], val_loss[:4], ['a', None, 'c', 'd'])\n"
                       "_m = annotate_points(_a, epochs[-2:], val_acc[-2:], ['p', 'q'], offset=(0, -12))\n"
                       f"(_k, _m, _txs(_a, 'text', 'target', 'pos', 'frame'))")}),
    ]
    return catalog(rng, items, code, f"{desc}. Write the function only (no need to draw).")


@template(M, "hard")
def fix_legend_text_bugs(rng):
    code, desc, n = curves(rng)
    a, b = rng.choice([("train", "val"), ("training", "validation"), ("fit", "holdout")])
    dx, dy = rng.choice([(3, 0.5), (2, 0.4), (4, 0.3)])
    word = rng.choice(["final", "last", "end"])
    fig = "fig, ax = plt.subplots()\n"
    plot2 = f"ax.plot(epochs, train_loss, label={a!r})\nax.plot(epochs, val_loss, label={b!r})\n"
    items = [
        (".legend", f"This draws an **empty** legend and warns *No artists with labels found*:\n```python\n{fig}"
                    f"ax.legend()\n{plot2}```\nFix it so the legend lists {a!r} and {b!r}.",
         fig + plot2 + "ax.legend()", {"post": _lgp("'texts'")}),
        ("label", f"This crashes with *Line2D.set() got an unexpected keyword argument 'labels'*:\n```python\n{fig}"
                  f"ax.plot(epochs, train_loss, labels={a!r})\nax.plot(epochs, val_loss, labels={b!r})\n"
                  f"ax.legend(loc='upper right')\n```\nFix it (keep the legend's location).",
         fig + plot2 + "ax.legend(loc='upper right')", {"post": post("fc.lines(fc.ax(), 'label'), _lg(fc.ax().get_legend(), 'texts')")}),
        (".legend", f"This draws an empty legend (and warns that *Legend does not support handles for str "
                    f"instances*):\n```python\n{fig}ax.plot(epochs, train_loss)\nax.plot(epochs, val_loss)\n"
                    f"ax.legend({a!r}, {b!r})\n```\nTwo strings are taken as `handles` and `labels`. Fix the "
                    f"`ax.legend` call so the entries read {a!r} and {b!r}.",
         f"{fig}ax.plot(epochs, train_loss)\nax.plot(epochs, val_loss)\nax.legend([{a!r}, {b!r}])",
         {"post": _lgp("'texts'")}),
        ("xytext", f"The arrow should point **at** the minimum of `val_loss`, with the text {dx} epochs to the right and "
                   f"{dy} above. Here it points away from the curve into empty space:\n```python\n{fig}"
                   f"ax.plot(epochs, val_loss)\ni = val_loss.argmin()\nax.annotate('min', xy=(epochs[i] + {dx}, "
                   f"val_loss[i] + {dy}), xytext=(epochs[i], val_loss[i]),\n            arrowprops=dict(arrowstyle='->'))"
                   f"\n```\nFix it.",
         f"{fig}ax.plot(epochs, val_loss)\ni = val_loss.argmin()\nax.annotate('min', xy=(epochs[i], val_loss[i]), "
         f"xytext=(epochs[i] + {dx}, val_loss[i] + {dy}), arrowprops=dict(arrowstyle='->'))",
         {"post": post("_tx(fc.ax().texts[0], 'text', 'target', 'pos', 'arrow')")}),
        ("transAxes", f"This text is meant for the top-right corner of the Axes, but lands at the **data** position "
                      f"(0.95, 0.95), left of the first epoch:\n```python\n{fig}ax.plot(epochs, val_loss)\n"
                      f"ax.text(0.95, 0.95, {word!r}, ha='right', va='top')\n```\nFix it.",
         f"{fig}ax.plot(epochs, val_loss)\nax.text(0.95, 0.95, {word!r}, ha='right', va='top', transform=ax.transAxes)",
         {"post": post("_tx(fc.ax().texts[0], 'text', 'pos', 'frame', 'ha', 'va')")}),
        ("set_ylabel", "This y label comes out garbled: in a normal string `\\n` is a newline and `\\t` a tab, so "
                       "mathtext never sees `\\nabla` or `\\theta`:\n```python\nfig, ax = plt.subplots()\n"
                       "ax.plot(epochs, train_loss)\nax.set_ylabel('$\\nabla_\\theta \\mathcal{L}$')\n```\nFix it.",
         f"{fig}ax.plot(epochs, train_loss)\nax.set_ylabel(r'$\\nabla_\\theta \\mathcal{{L}}$')",
         {"post": post("_math(fc.ax().get_ylabel(), '\\\\nabla', '\\\\theta', '\\\\mathcal{L}')")}),
        ("add_artist", f"Only the **second** legend shows, because every `ax.legend` call replaces the Axes' legend:\n"
                       f"```python\nfrom matplotlib.lines import Line2D\n{fig}l1, = ax.plot(epochs, train_loss, "
                       f"label={a!r})\nl2, = ax.plot(epochs, val_loss, label={b!r})\nbase = Line2D([0], [0], color='k', "
                       f"ls='--', label='baseline')\nax.legend(handles=[l1, l2], loc='upper left')\n"
                       f"ax.legend(handles=[base], loc='upper right')\n```\nKeep both legends.",
         f"from matplotlib.lines import Line2D\n{fig}l1, = ax.plot(epochs, train_loss, label={a!r})\n"
         f"l2, = ax.plot(epochs, val_loss, label={b!r})\nbase = Line2D([0], [0], color='k', ls='--', label='baseline')\n"
         f"first = ax.legend(handles=[l1, l2], loc='upper left')\nax.add_artist(first)\n"
         f"ax.legend(handles=[base], loc='upper right')",
         {"post": post("[_lg(l, 'texts', 'loc') for l in _legends(fc.ax())]")}),
        ("bbox_inches", f"The saved `loss.png` cuts the legend off: it sits outside the Axes, beyond the 4 × 3 inch "
                        f"canvas.\n```python\nfig, ax = plt.subplots(figsize=(4, 3))\n{plot2}"
                        f"ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))\nfig.savefig('loss.png', dpi=40)\n"
                        f"plt.close(fig)\n```\nKeep the legend outside but make the saved image include it "
                        f"(`bbox_inches='tight'`). (Checked: `loss.png` is wider than 4 × 40 px and no figure is left "
                        f"open.)",
         f"fig, ax = plt.subplots(figsize=(4, 3))\n{plot2}ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))\n"
         f"fig.savefig('loss.png', dpi=40, bbox_inches='tight')\nplt.close(fig)",
         {"post": "plt.imread('loss.png').shape[1] > 160, len(plt.get_fignums())", "require": ["bbox_inches"]}),
        ("set_title", "Saving crashes with a *ParseSyntaxException* (`Expected \\frac{num}{den}`): a brace is missing.\n"
                      "```python\nfig, ax = plt.subplots()\nax.plot(epochs, train_loss)\n"
                      "ax.set_title(r'$\\frac{1}{N\\sum_i \\ell_i$')\nfig.savefig('loss.png', dpi=30)\n```\nFix the title: "
                      "it should read 1/N times the sum of ℓ_i (`\\frac{1}{N}\\sum_i \\ell_i`).",
         f"{fig}ax.plot(epochs, train_loss)\nax.set_title(r'$\\frac{{1}}{{N}}\\sum_i \\ell_i$')\nfig.savefig('loss.png', dpi=30)",
         {"post": post("_math(fc.ax().get_title(), '\\\\frac{1}{N}', '\\\\sum_i', '\\\\ell_i'), plt.imread('loss.png').shape")}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "hard")
def legend_text_drills(rng):
    code, desc, n = curves(rng, acc=True)
    rcode, rdesc, k, lrs = _runs(rng)
    a, b = rng.choice([("train", "val"), ("training", "validation"), ("fit", "holdout")])
    two = (f"ax.plot(epochs, train_loss, label={a!r})\nax.plot(epochs, val_loss, label={b!r})\n")
    temps = rng.choice([[0.5, 1, 2], [0.25, 1, 4], [1, 2, 5]])
    rad = rng.choice([0.2, 0.3, -0.3])
    ang = rng.choice([20, 30, 45])
    s = seeded(rng)
    items = [
        (".annotate", f"{rdesc}. Plot every run on `fig, ax` (`ax.plot(steps, r)`) and annotate each run's minimum with "
                      f"its value to 3 decimals, the text offset (5, 10) points, in the run's own line color, with an "
                      f"arrow `'-'` of that color too.",
         "fig, ax = plt.subplots()\nfor r in runs:\n    line, = ax.plot(steps, r)\n    i = r.argmin()\n"
         "    ax.annotate(f'{r[i]:.3f}', xy=(steps[i], r[i]), xytext=(5, 10), textcoords='offset points',\n"
         "                color=line.get_color(), arrowprops=dict(arrowstyle='-', color=line.get_color()))",
         {"post": post("_txs(fc.ax(), 'text', 'target', 'pos', 'color', 'arrowcolor')"), "setup": rcode}),
        ("layout", f"Put the legend outside the plot on the right **without clipping it**: create the figure with "
                   f"`layout='constrained'` (it shrinks the Axes to make room), plot the two losses labelled {a!r} and "
                   f"{b!r}, and anchor the legend's center-left at (1.02, 0.5). (Checked after drawing: the legend's "
                   f"box ends inside the figure.)",
         "fig, ax = plt.subplots(layout='constrained')\n" + two + "ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))",
         {"post": post("_f = fc.fig()\n_f.canvas.draw()\n_e = fc.ax().get_legend().get_window_extent()\n"
                       "_lg(fc.ax().get_legend(), 'texts', 'loc', 'anchor'), bool(_e.x1 <= _f.bbox.x1 + 1)")}),
        ("savefig", f"Same drill for a saved file: a 4 × 3 inch figure with the two losses (labelled {a!r}, {b!r}) and "
                    f"the legend outside on the right (center-left at (1.02, 0.5)); save it as `curves.png` at 40 dpi "
                    f"so that the legend is **in** the image (`bbox_inches='tight'`), then close the figure. Return "
                    f"`plt.imread('curves.png').shape[1] > 160` (wider than the bare 4 × 40 px canvas).",
         "fig, ax = plt.subplots(figsize=(4, 3))\n" + two + "ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))\n"
         "fig.savefig('curves.png', dpi=40, bbox_inches='tight')\nplt.close(fig)\nresult = plt.imread('curves.png').shape[1] > 160",
         {"post": "_, len(plt.get_fignums())", "require": ["bbox_inches", "close"]}),
        (".set_title", f"Softmax with temperature (`logits` holds 6 scores, `temps` a few temperatures): for each `T` in `temps` plot `softmax(logits / T)` with circle "
                       f"markers (`'o-'`) labelled `rf'$T = {{T}}$'`, add a legend, and title the plot `$p_i = "
                       f"\\frac{{e^{{z_i/T}}}}{{\\sum_j e^{{z_j/T}}}}$` (checked: renders, uses `\\frac`, `\\sum` and "
                       f"`z_i/T`).",
         "fig, ax = plt.subplots()\nfor T in temps:\n    e = np.exp(logits / T - (logits / T).max())\n"
         "    ax.plot(e / e.sum(), 'o-', label=rf'$T = {T}$')\nax.legend()\n"
         "ax.set_title(r'$p_i = \\frac{e^{z_i/T}}{\\sum_j e^{z_j/T}}$')",
         {"post": post("fc.lines(fc.ax(), nd=5), _lg(fc.ax().get_legend(), 'texts'), _math(fc.ax().get_title(), "
                       "'\\\\frac', '\\\\sum', 'z_i/T')"),
          "setup": f"temps = {temps}\nlogits = np.round(np.random.default_rng({s}).normal(0, 2, 6), 2)"}),
        ("axhline", "Mark the minimum of `val_loss` with dotted crosshairs: `ax.axvline` at its epoch and `ax.axhline` "
                    "at its value (both `ls=':'`, color 'gray'), and annotate it with `f'({e}, {v:.3f})'` offset "
                    "(10, 10) points. Plot `val_loss` against `epochs` first.",
         "fig, ax = plt.subplots()\nax.plot(epochs, val_loss)\ni = val_loss.argmin()\ne, v = epochs[i], val_loss[i]\n"
         "ax.axvline(e, ls=':', color='gray')\nax.axhline(v, ls=':', color='gray')\n"
         "ax.annotate(f'({e}, {v:.3f})', xy=(e, v), xytext=(10, 10), textcoords='offset points')",
         {"post": post("fc.lines(fc.ax(), 'ls', 'color'), _txs(fc.ax(), 'text', 'target', 'pos', 'frame')")}),
        ("argsort", "`x`, `y` hold a noisy wave (60 points). Plot `y` against `x` and label its 3 highest **local maxima** (points higher than both "
                    "neighbours), highest first: each with its value to 2 decimals in a rounded box "
                    "(`bbox=dict(boxstyle='round', fc='w')`), the text offset (0, 10) points, centred.",
         "fig, ax = plt.subplots()\nax.plot(x, y)\npeaks = np.where((y[1:-1] > y[:-2]) & (y[1:-1] > y[2:]))[0] + 1\n"
         "for i in peaks[np.argsort(y[peaks])[::-1][:3]]:\n    ax.annotate(f'{y[i]:.2f}', xy=(x[i], y[i]), xytext=(0, 10), "
         "textcoords='offset points', ha='center', bbox=dict(boxstyle='round', fc='w'))",
         {"post": post("_txs(fc.ax(), 'text', 'target', 'pos', 'frame', 'ha', 'box')"),
          "setup": f"x = np.linspace(0, 10, 60)\ny = np.round(np.sin(2 * x) + 0.3 * np.random.default_rng({s}).normal(size=60), 3)"}),
        ("fig.text", f"Stamp a watermark on the figure: plot the two losses on `fig, ax`, then write 'DRAFT' at the "
                     f"figure center (0.5, 0.5) in **figure** coordinates, centred both ways, rotated {ang}°, font size "
                     f"40, alpha 0.3.",
         "fig, ax = plt.subplots()\n" + two + f"fig.text(0.5, 0.5, 'DRAFT', ha='center', va='center', rotation={ang}, "
                                              f"fontsize=40, alpha=0.3)",
         {"post": post("[_tx(t, 'text', 'pos', 'frame', 'ha', 'va', 'rotation', 'size', 'alpha') for t in fc.fig().texts]"),
          "require": ["text"]}),
        ("connectionstyle", f"Plot `val_loss` against `epochs` and annotate its minimum with 'best' using a **curved** "
                            f"arrow: the text at (0.6, 0.7) in axes fraction, `arrowprops=dict(arrowstyle='->', "
                            f"connectionstyle='arc3,rad={rad}')`.",
         f"fig, ax = plt.subplots()\nax.plot(epochs, val_loss)\ni = val_loss.argmin()\nax.annotate('best', "
         f"xy=(epochs[i], val_loss[i]), xytext=(0.6, 0.7), textcoords='axes fraction', "
         f"arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad={rad}'))",
         {"post": post("_tx(fc.ax().texts[0], 'text', 'target', 'pos', 'frame', 'conn')")}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "hard")
def panel_annotations(rng):
    code, desc, k, lrs = _runs(rng, k=4)
    thr = rng.choice([0.1, 0.2, 0.3])
    loc = rng.choice(["upper right", "center right", "lower left"])
    items = [
        ("subplots", "Draw a 2 × 2 grid (`fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)`), one run per panel "
                     "in row-major order. Title each panel `rf'$\\eta = {lr}$'` and put a bold panel letter '(a)' … "
                     "'(d)' at (0.02, 0.95) in axes coordinates, top-aligned.",
         "fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)\nfor a, lr, r, tag in zip(axs.flat, lrs, runs, "
         "['(a)', '(b)', '(c)', '(d)']):\n    a.plot(steps, r)\n    a.set_title(rf'$\\eta = {lr}$')\n"
         "    a.text(0.02, 0.95, tag, transform=a.transAxes, va='top', fontweight='bold')",
         {"post": post("[(fc.labels(a)['title'], _txs(a, 'text', 'pos', 'frame', 'va', 'bold'), fc.lines(a, nd=4)) "
                       "for a in fc.axes()]")}),
        ("bar", f"Create `fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))`. Left: every run labelled `str(lr)` "
                f"with a legend titled 'lr' at {loc!r}. Right: the final loss of each run as bars at x = 0 … 3, each "
                f"labelled with its value to 3 decimals (top-center of the bar, centred, bottom-aligned).",
         f"fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))\nfor lr, r in zip(lrs, runs):\n    ax1.plot(steps, r, "
         f"label=str(lr))\nax1.legend(title='lr', loc={loc!r})\nfinal = runs[:, -1]\nax2.bar(range(len(final)), final)\n"
         f"for i, v in enumerate(final):\n    ax2.text(i, v, f'{{v:.3f}}', ha='center', va='bottom')",
         {"post": post("_lg(fc.ax(0).get_legend(), 'texts', 'title', 'loc'), fc.heights(fc.ax(1), nd=4), "
                       "_txs(fc.ax(1), 'text', 'pos', 'ha', 'va')")}),
        ("fig.legend", "Create `fig, (top, bottom) = plt.subplots(2, 1, sharex=True, layout='constrained')`. Top: every "
                       "run labelled `f'lr={lr}'`. Bottom: each run minus the best value at each step (`runs - "
                       "runs.min(axis=0)`), no labels, y label `$\\Delta\\mathcal{L}$`. One figure legend from the top "
                       "panel's handles, outside on the right: `loc='outside right upper'`.",
         "fig, (top, bottom) = plt.subplots(2, 1, sharex=True, layout='constrained')\nfor lr, r in zip(lrs, runs):\n"
         "    top.plot(steps, r, label=f'lr={lr}')\nfor r in runs - runs.min(axis=0):\n    bottom.plot(steps, r)\n"
         "bottom.set_ylabel(r'$\\Delta\\mathcal{L}$')\nhandles, labels = top.get_legend_handles_labels()\n"
         "fig.legend(handles, labels, loc='outside right upper')",
         {"post": post("[_lg(l, 'texts', 'outside', 'loc') for l in fc.fig().legends], fc.lines(fc.ax(1), nd=4), "
                       "_math(fc.ax(1).get_ylabel(), '\\\\Delta', '\\\\mathcal{L}')"), "require": ["legend"]}),
        ("bbox", "Create `fig, (ax1, ax2) = plt.subplots(1, 2)`. Left: the run with the lowest final loss, its minimum "
                 "annotated with 'min' (offset (10, 10) points, arrow `'->'`). Right: the last 10 steps of every run, "
                 "and a summary text in the top-right corner (0.98, 0.98 in axes coordinates, right/top-aligned, "
                 "rounded box): one line per run, `f'{lr}: {final:.3f}'`, joined with newlines.",
         "fig, (ax1, ax2) = plt.subplots(1, 2)\nbest = runs[runs[:, -1].argmin()]\nax1.plot(steps, best)\ni = best.argmin()\n"
         "ax1.annotate('min', xy=(steps[i], best[i]), xytext=(10, 10), textcoords='offset points', "
         "arrowprops=dict(arrowstyle='->'))\nfor r in runs:\n    ax2.plot(steps[-10:], r[-10:])\n"
         "summary = '\\n'.join(f'{lr}: {r[-1]:.3f}' for lr, r in zip(lrs, runs))\n"
         "ax2.text(0.98, 0.98, summary, transform=ax2.transAxes, ha='right', va='top', bbox=dict(boxstyle='round', fc='w'))",
         {"post": post("fc.lines(fc.ax(0), nd=4), _txs(fc.ax(0), 'text', 'target', 'arrow'), len(fc.ax(1).get_lines()), "
                       "_txs(fc.ax(1), 'text', 'pos', 'frame', 'ha', 'va', 'box')")}),
        ("suptitle", "Create `fig, axs = plt.subplots(1, 4, sharey=True, figsize=(10, 2.5))`, one run per panel, each "
                     "titled `f'lr = {lr}'`; the first panel gets the y label `$\\mathcal{L}$`, and the figure the "
                     "title `rf'Loss $\\mathcal{{L}}$ for {len(lrs)} learning rates'`.",
         "fig, axs = plt.subplots(1, 4, sharey=True, figsize=(10, 2.5))\nfor a, lr, r in zip(axs, lrs, runs):\n"
         "    a.plot(steps, r)\n    a.set_title(f'lr = {lr}')\naxs[0].set_ylabel(r'$\\mathcal{L}$')\n"
         "fig.suptitle(rf'Loss $\\mathcal{{L}}$ for {len(lrs)} learning rates')",
         {"post": post("[fc.labels(a) for a in fc.axes()], fc.suptitles()['suptitle'], "
                       "_math(fc.suptitles()['suptitle'])['renders']")}),
        ("color", f"Create `fig, axs = plt.subplots(1, 4, sharey=True)`, one run per panel. In each panel write a "
                  f"verdict at (0.5, 0.9) in axes coordinates, centred: 'converged' in green if the run's final loss "
                  f"is below {thr}, else 'stalled' in red.",
         f"fig, axs = plt.subplots(1, 4, sharey=True)\nfor a, r in zip(axs, runs):\n    a.plot(steps, r)\n"
         f"    ok = r[-1] < {thr}\n    a.text(0.5, 0.9, 'converged' if ok else 'stalled', color='green' if ok else 'red', "
         f"transform=a.transAxes, ha='center')",
         {"post": post("[_txs(a, 'text', 'pos', 'frame', 'ha', 'color') for a in fc.axes()]")}),
        ("family", "Create `fig, (ax1, ax2) = plt.subplots(1, 2)`: every run on `ax1`, and a text table on `ax2` with "
                   "its axes hidden (`ax2.axis('off')`): row i is `f'{lr:<8}{r[-1]:.3f}'` in a monospace font "
                   "(`family='monospace'`), at (0, 0.9 - 0.15 * i) in axes coordinates.",
         "fig, (ax1, ax2) = plt.subplots(1, 2)\nfor r in runs:\n    ax1.plot(steps, r)\nax2.axis('off')\n"
         "for i, (lr, r) in enumerate(zip(lrs, runs)):\n    ax2.text(0, 0.9 - 0.15 * i, f'{lr:<8}{r[-1]:.3f}', "
         "family='monospace', transform=ax2.transAxes)",
         {"post": post("len(fc.ax(0).get_lines()), fc.ax(1).axison, _txs(fc.ax(1), 'text', 'pos', 'frame', 'family')")}),
    ]
    return catalog(rng, items, code, f"{desc}.")


# ================================================================= challenge

@template(M, "challenge")
def label_collisions(rng):
    n, s = rng.randint(10, 14), seeded(rng)
    w, h = rng.choice([(40, 12), (50, 15), (60, 20), (36, 14)])
    report = rng.choice(["kept", "count", "last"])
    setup = (f"g = np.random.default_rng({s})\nx = g.integers(0, 200, {n})\ny = g.integers(0, 120, {n})\n"
             f"score = g.integers(0, 10, {n})\nw, h = {w}, {h}\nreport = {report!r}")
    sol = ("X, Y = x.tolist(), y.tolist()\norder = np.lexsort((np.arange(len(X)), -score)).tolist()   # by score, then index\n"
           "cell = {}      # grid cell (w x h) -> the kept label anchored in it (at most one fits per cell)\nkept = []\n"
           "for i in order:\n    cx, cy = X[i] // w, Y[i] // h\n    free = True\n    for dx in (-1, 0, 1):\n"
           "        for dy in (-1, 0, 1):\n            j = cell.get((cx + dx, cy + dy))\n"
           "            if j is not None and abs(X[j] - X[i]) < w and abs(Y[j] - Y[i]) < h:\n                free = False\n"
           "    if free:\n        cell[(cx, cy)] = i\n        kept.append(i)\n"
           + {"kept": "result = np.array(sorted(kept))", "count": "result = len(kept)", "last": "result = kept[-1]"}[report])
    tests = [f"g = np.random.default_rng({s + i})\nx = g.integers(0, {W}, 40000)\ny = g.integers(0, {H}, 40000)\n"
             f"score = g.integers(0, 1000, 40000)\nw, h = {bw}, {bh}\nreport = {report!r}"
             for i, (W, H, bw, bh) in ((1, (12000, 8000, 60, 14)), (2, (3000, 2000, 25, 9)))]
    want = {"kept": "the indices of the labels that are kept, ascending, as an array",
            "count": "how many labels are kept",
            "last": "the index of the **last** label kept in processing order"}[report]
    return challenge(
        "Greedy label decluttering",
        "Matplotlib draws every text you ask for, even on top of each other. A plotting library that wants clean "
        "labels declutters them greedily: label `i` is a `w × h` pixel box with its lower-left corner at `(x[i], "
        "y[i])` (it covers `[x, x + w) × [y, y + h)`, so boxes that only touch don't overlap). Go through the labels "
        "by decreasing `score` (ties: smaller index first) and **keep** a label if its box overlaps none of the "
        f"labels kept so far; otherwise drop it. Return {want}. (`report` = {report!r}.)",
        setup, sol, tests, "n ≤ 4·10^4 labels, coordinates ≤ 2·10^4, 1 ≤ w, h ≤ 100",
        speed="about O(n log n): a grid of w × h cells holds at most one kept label each, so check the 3 × 3 cells around",
        answer_desc={"kept": " (an int array)", "count": " (an int)", "last": " (an int)"}[report])


@template(M, "challenge")
def legend_best_spot(rng):
    n, q, s = rng.randint(25, 40), rng.randint(5, 8), seeded(rng)
    W, H, pad = rng.choice([(200, 150, 5), (240, 160, 8), (180, 120, 4)])
    report = rng.choice(["loc", "badness", "clear"])
    setup = (f"g = np.random.default_rng({s})\nW, H, pad = {W}, {H}, {pad}\npx = g.integers(0, W + 1, {n})\n"
             f"py = np.clip((H * np.exp(-px / (W / 3)) + g.normal(0, H / 8, {n})).astype(int), 0, H)\n"
             f"lw = g.integers(20, W // 2, {q})\nlh = g.integers(15, H // 2, {q})\nreport = {report!r}")
    sol = ("# C[a, b] = points at (a - 1, b - 1); S[a, b] = points with px < a and py < b\n"
           "C = np.bincount((px + 1) * (H + 2) + py + 1, minlength=(W + 2) * (H + 2)).reshape(W + 2, H + 2)\n"
           "S = C.cumsum(axis=0).cumsum(axis=1)\nright, top = W - pad - lw, H - pad - lh\n"
           "midx, midy = (W - lw) // 2, (H - lh) // 2\nleft, bottom = np.full_like(lw, pad), np.full_like(lh, pad)\n"
           "X0 = np.stack([right, left, left, right, right, left, right, midx, midx, midx], axis=1)\n"
           "Y0 = np.stack([top, top, bottom, bottom, midy, midy, midy, bottom, top, midy], axis=1)\n"
           "X1, Y1 = X0 + lw[:, None], Y0 + lh[:, None]     # strictly inside: x0 < px < x1, y0 < py < y1\n"
           "inside = S[X1, Y1] - S[X0 + 1, Y1] - S[X1, Y0 + 1] + S[X0 + 1, Y0 + 1]\n"
           + {"loc": "result = inside.argmin(axis=1) + 1",
              "badness": "result = inside.min(axis=1)",
              "clear": "result = int((inside.min(axis=1) == 0).sum())"}[report])
    tests = [f"g = np.random.default_rng({s + i})\nW, H, pad = {WW}, {HH}, {pp}\npx = g.integers(0, W + 1, 200000)\n"
             f"py = np.clip((H * np.exp(-px / (W / 3)) + g.normal(0, H / 8, 200000)).astype(int), 0, H)\n"
             f"lw = g.integers(20, W // 2, 200000)\nlh = g.integers(15, H // 2, 200000)\nreport = {report!r}"
             for i, (WW, HH, pp) in ((1, (1000, 800, 10)), (2, (640, 480, 6)))]
    want = {"loc": "the chosen location code for each legend, as an array",
            "badness": "the smallest number of points covered for each legend, as an array",
            "clear": "how many legends have a spot that covers no point at all"}[report]
    return challenge(
        "Where does `loc='best'` go?",
        "`ax.legend(loc='best')` tries the nine other locations in a fixed order and keeps the first one that covers "
        "the fewest data points. Model it: the Axes is the pixel box `[0, W] × [0, H]` holding the points `(px, py)`. "
        "A legend of `w × h` pixels is placed with its lower-left corner at, in order, code 1 'upper right' "
        "`(W - pad - w, H - pad - h)`, 2 'upper left' `(pad, H - pad - h)`, 3 'lower left' `(pad, pad)`, 4 'lower "
        "right' `(W - pad - w, pad)`, 5 'right' and 7 'center right' `(W - pad - w, my)`, 6 'center left' `(pad, my)`, "
        "8 'lower center' `(mx, pad)`, 9 'upper center' `(mx, H - pad - h)`, 10 'center' `(mx, my)`, with `mx = (W - "
        "w) // 2`, `my = (H - h) // 2`. A point covers the legend if it is **strictly** inside the box. Legend `j` "
        f"has size `lw[j] × lh[j]`; the smallest code wins ties. Return {want}. (`report` = {report!r}.)",
        setup, sol, tests, "points, legends ≤ 2·10^5, W, H ≤ 1000",
        speed="O(W·H + n + q): a 2-D prefix-sum table answers each box in O(1)",
        answer_desc=" (an int array)" if report != "clear" else " (an int)")


@template(M, "challenge")
def caption_wrap(rng):
    n, s = rng.randint(9, 12), seeded(rng)
    L = rng.randint(14, 20)
    report = rng.choice(["greedy", "balanced", "ragged"])
    k = rng.randint(2, 4)
    setup = (f"g = np.random.default_rng({s})\nlens = g.integers(1, 9, {n})\nL = {L}\nk = {k}\nreport = {report!r}")
    if report == "greedy":
        body = ("lines, width = 1, -1\nfor w in lens.tolist():\n    if width + 1 + w <= L:\n        width += 1 + w\n"
                "    else:\n        lines, width = lines + 1, w\nresult = lines")
        want = ("the number of lines when each line greedily takes as many words as fit in width `L` (a word that "
                "doesn't fit starts a new line)")
        lim, speed = "n ≤ 2·10^5 words, word lengths ≤ 8 ≤ L ≤ 100", "O(n)"
    elif report == "balanced":
        body = ("Q = np.concatenate([[0], np.cumsum(lens + 1)])   # width of words a..b-1 = Q[b] - Q[a] - 1\n"
                "def lines_needed(width):\n    i, used = 0, 0\n    while i < len(lens) and used <= k:\n"
                "        i = int(np.searchsorted(Q, Q[i] + width + 1, side='right')) - 1\n        used += 1\n    return used\n\n"
                "lo, hi = int(lens.max()), int(Q[-1] - 1)\nwhile lo < hi:\n    mid = (lo + hi) // 2\n"
                "    if lines_needed(mid) <= k:\n        hi = mid\n    else:\n        lo = mid + 1\nresult = lo")
        want = ("the **smallest** width `L` for which the caption fits in at most `k` lines (greedy filling uses "
                "the fewest lines for a given width, so test widths with it)")
        lim, speed = "n ≤ 2·10^5 words, word lengths ≤ 8, k ≤ 2000", "O(k log n log(total width)) with prefix sums"
    else:
        body = ("n = len(lens)\nP = np.concatenate([[0], np.cumsum(lens)]).tolist()\nINF = float('inf')\n"
                "best = [INF] * n + [0]     # best[i]: cost of wrapping words i..n-1\nfor i in range(n - 1, -1, -1):\n"
                "    j = i + 1\n    while j <= n and P[j] - P[i] + (j - i - 1) <= L:\n"
                "        width = P[j] - P[i] + (j - i - 1)\n        cost = 0 if j == n else (L - width) ** 2\n"
                "        best[i] = min(best[i], cost + best[j])\n        j += 1\nresult = int(best[0])")
        want = ("the minimum total **raggedness**: the sum of `(L - width) ** 2` over every line except the last, "
                "over all ways to break the caption into lines of width ≤ `L`")
        lim, speed = "n ≤ 2·10^4 words, word lengths ≤ 8 ≤ L ≤ 60", "O(n · L) dynamic programming"
    tests = [f"g = np.random.default_rng({s + i})\nlens = g.integers(1, 9, {nn})\nL = {LL}\nk = {kk}\nreport = {report!r}"
             for i, (nn, LL, kk) in ((1, ({"greedy": 200000, "balanced": 200000, "ragged": 20000}[report], 57, 1500)),
                                     (2, ({"greedy": 150000, "balanced": 150000, "ragged": 15000}[report], 23, 40)))]
    return challenge(
        "Wrapping a caption",
        "Long titles and captions must be wrapped (`plt.text(..., wrap=True)` does it at the figure edge). Model a "
        "caption in a monospace font: `lens[i]` is the length of word `i`, words keep their order and are separated "
        "by single spaces, so a line holding words `a … b` is `lens[a] + … + lens[b] + (b - a)` characters wide; "
        f"every line must be at most `L` wide. Return {want}. (`report` = {report!r}.)",
        setup, body, tests, lim, speed=speed, answer_desc=" (an int)")
