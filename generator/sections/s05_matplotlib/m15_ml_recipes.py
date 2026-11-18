"""5.15 ML visualization recipes: the 14 figures every ML project needs (training curves, confusion matrices,
decision boundaries, image grids, attention / correlation / sweep heatmaps, embeddings, weight histograms, LR
schedules, ROC and precision-recall curves, feature importances, regression diagnostics). This page also loads
scikit-learn, so some problems fit real models."""

from ._common import P, catalog, challenge, curves, lit, seeded, template
from ... import themes as T

M = "5.15"

# post helpers: _img(im) reads an image's array (masked cells -> None; figcheck.r flattens 2-D masks wrongly),
# _cells(ax) the texts written in heatmap cells (sorted by position), _ticks(ax) tick labels on both axes.
PRE = '''def _img(im, nd=4):
    import numpy as _np
    a = im.get_array()
    m = _np.ma.getmaskarray(a).tolist()
    return [[None if mm else round(float(v), nd) for v, mm in zip(rv, rm)] for rv, rm in zip(_np.ma.getdata(a).tolist(), m)]

def _cells(a=None):
    a = a or fc.ax()
    return sorted((fc.r(t.get_position(), 3), t.get_text(), t.get_ha(), t.get_va()) for t in a.texts)

def _cellcolors(a=None):
    a = a or fc.ax()
    return sorted((fc.r(t.get_position(), 3), fc.hexcolor(t.get_color())) for t in a.texts)

def _ticks(a=None):
    a = a or fc.ax()
    return fc.ticks(a, 'x'), fc.ticks(a, 'y')
'''


def _post(*exprs):
    return PRE + ", ".join(exprs)


def _warm_sklearn():
    """Import scikit-learn while a problem that uses it is generated. Its first import costs about a second here (a
    few in Pyodide); without this, the first graded run would pay it, inside its time limit (the checker flagged that
    run as slow). Only this page loads scikit-learn, and only its templates call this."""
    try:
        import sklearn.datasets, sklearn.decomposition, sklearn.ensemble, sklearn.linear_model  # noqa: E401,F401
        import sklearn.metrics, sklearn.neighbors  # noqa: E401,F401
    except ImportError:   # not installed: the problems still build; grading imports it when needed
        pass


def _cm(rng, k=(3, 4)):
    _, labs = T.labels(rng, min_classes=k[1])
    n = rng.randint(*k)
    names = list(labs[:n])
    cm = [[rng.randint(30, 60) if i == j else rng.randint(0, 8) for j in range(n)] for i in range(n)]
    return f"cm = np.array({cm})   # rows: true class, columns: predicted\nnames = {lit(names)}", n, names


def _scores(rng, n=(40, 80)):
    n_ = rng.randint(*n)
    s = seeded(rng)
    return (f"g = np.random.default_rng({s})\ny_true = g.integers(0, 2, {n_})\n"
            f"scores = np.round(np.clip(0.35 * y_true + g.normal(0.35, 0.2, {n_}), 0, 1), 2)   # model scores"), n_


def _regression(rng):
    s = seeded(rng)
    n = rng.randint(40, 120)
    noise = rng.choice([0.2, 0.3, 0.5])
    return (f"g = np.random.default_rng({s})\ny_act = np.round(g.normal(size={n}), 3)\n"
            f"y_hat = np.round(y_act + g.normal(0, {noise}, {n}), 3)   # predictions"), n


def _features(rng):
    s = seeded(rng)
    k = rng.randint(6, 10)
    pool = ["age", "income", "clicks", "dwell", "price", "rating", "tenure", "visits", "region", "device", "hour", "score"]
    names = rng.sample(pool, k)
    return f"feats = np.array({lit(names)})\nimp = np.round(np.random.default_rng({s}).random({k}), 3)   # feature importances", k


# ================================================================= lite

@template(M, "lite")
def curve_recipe_calls(rng):
    code, desc, n = curves(rng)
    s = seeded(rng)
    setup = code + (f"\ng2 = np.random.default_rng({s})\nw = np.round(g2.normal(0, 0.5, 1000), 4)   # a layer's weights\n"
                    f"steps = np.arange(200)\nlr = 3e-4 * np.minimum(1, steps / 20)\n"
                    f"y_act = np.round(g2.normal(size=50), 3)\ny_hat = np.round(y_act + g2.normal(0, 0.3, 50), 3)")
    plot = "fig, ax = plt.subplots()\nax.plot(epochs, train_loss, label='train')\nax.plot(epochs, val_loss, label='val')\n"
    bins = rng.choice([30, 40, 60])
    items = [
        ("set_yscale", "Plot `train_loss` and `val_loss` against `epochs` (labels 'train', 'val') on a **log** y-axis.",
         plot + "ax.set_yscale('log')", {"post": "fc.lines(fc.ax(), 'label'), fc.scales()"}),
        ("axvline", "Plot both curves and mark the best epoch (lowest `val_loss`) with a dashed gray vertical line: "
                    "`ax.axvline(epochs[best], ls='--', c='gray')`.",
         plot + "best = val_loss.argmin()\nax.axvline(epochs[best], ls='--', c='gray')",
         {"post": "fc.lines(fc.ax(), 'ls', 'color')", "require": ["axvline", "argmin"]}),
        ("scatter", "Plot both curves and put a red dot on the best validation point with `ax.scatter(..., c='red', "
                    "zorder=3, label=f'best @ {epochs[best]}')`, then a legend.",
         plot + "best = val_loss.argmin()\nax.scatter(epochs[best], val_loss[best], c='red', zorder=3, label=f'best @ {epochs[best]}')\nax.legend()",
         {"post": "fc.scatter(fc.ax(), 0, 6, 'color', 'label'), fc.legend()"}),
        ("hist", f"Histogram the weights `w` with {bins} bins (`ax.hist`).", f"fig, ax = plt.subplots()\nax.hist(w, bins={bins})",
         {"post": "fc.bars(fc.ax(), 4)"}),
        ("ticklabel_format", "Plot the learning-rate schedule `lr` against `steps` and write the y tick labels in "
                             "scientific notation with a common exponent: `ax.ticklabel_format(axis='y', style='sci', "
                             "scilimits=(0, 0))`.",
         "fig, ax = plt.subplots()\nax.plot(steps, lr)\nax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))",
         {"post": "f = fc.ax().yaxis.get_major_formatter()\nfc.lines(fc.ax()), f._scientific, list(f._powerlimits)"}),
        ("axhline", "Residual plot: scatter `y_hat - y_act` against `y_hat` and draw the zero line with `ax.axhline(0, "
                    "color='k')`.", "fig, ax = plt.subplots()\nax.scatter(y_hat, y_hat - y_act)\nax.axhline(0, color='k')",
         {"post": "fc.scatter(), fc.lines(fc.ax(), 'color')"}),
        ("plot", "Predicted vs actual: scatter `y_hat` against `y_act` and draw the identity line as a black dashed line "
                 "from `y_act.min()` to `y_act.max()` (`'k--'`).",
         "fig, ax = plt.subplots()\nax.scatter(y_act, y_hat)\nlims = [y_act.min(), y_act.max()]\nax.plot(lims, lims, 'k--')",
         {"post": "fc.scatter(), fc.lines(fc.ax(), 'color', 'ls')"}),
        ("set", "Plot `val_loss` against `epochs` and in **one** `ax.set(...)` call set the x label 'epoch', the y label "
                "'loss' and the title 'Validation'.",
         "fig, ax = plt.subplots()\nax.plot(epochs, val_loss)\nax.set(xlabel='epoch', ylabel='loss', title='Validation')",
         {"post": "fc.lines(fc.ax()), fc.labels()"}),
        ("legend", "Plot both curves (labels 'train', 'val') and put the legend in the lower right corner "
                   "(`loc='lower right'`).", plot + "ax.legend(loc='lower right')",
         {"post": "fc.legend(), fc.ax().get_legend()._loc"}),
        ("set_title", f"Histogram `w` ({bins} bins) and title it with its spread, `f'σ={{w.std():.2f}}'`, font size 9.",
         f"fig, ax = plt.subplots()\nax.hist(w, bins={bins})\nax.set_title(f'σ={{w.std():.2f}}', fontsize=9)",
         {"post": "fc.labels()['title'], fc.ax().title.get_fontsize(), len(fc.bars())"}),
    ]
    return catalog(rng, items, setup, f"{desc}; `w` holds a layer's weights, `lr` a warmup schedule over `steps`, and "
                                      f"`y_act` / `y_hat` regression targets and predictions. Create `fig, ax`.")


@template(M, "lite")
def matrix_recipe_calls(rng):
    code, k, names = _cm(rng)
    s = seeded(rng)
    setup = code + (f"\ng = np.random.default_rng({s})\ncorr = np.round(np.corrcoef(g.normal(size=(5, 60))), 3)\n"
                    f"emb = np.round(g.normal(size=(60, 2)), 3)\nlab = g.integers(0, 3, 60)\n"
                    f"xx, yy = np.meshgrid(np.linspace(-2, 2, 30), np.linspace(-2, 2, 30))\nproba = 1 / (1 + np.exp(-(xx + yy ** 2 - 1)))")
    cmap = rng.choice(["Blues", "Greens", "Purples", "viridis"])
    items = [
        ("imshow", f"Show the confusion matrix `cm` with `ax.imshow` and the {cmap!r} colormap, plus a colorbar "
                   f"(`fig.colorbar(im, ax=ax)`).",
         f"fig, ax = plt.subplots()\nim = ax.imshow(cm, cmap={cmap!r})\nfig.colorbar(im, ax=ax)",
         {"post": "fc.images(fc.ax(0), data=True), len(fc.colorbars())", "require": ["imshow", "colorbar"]}),
        ("set_xticks", "Show `cm` with `imshow` and label the ticks with the class names on both axes: "
                       f"`ax.set_xticks(range({k}), names)` and the same for y.",
         f"fig, ax = plt.subplots()\nax.imshow(cm)\nax.set_xticks(range({k}), names)\nax.set_yticks(range({k}), names)",
         {"post": _post("_ticks()"), "require": ["set_xticks", "set_yticks"]}),
        ("set_xticks", f"Show `cm` with `imshow` and put the class names on the x ticks rotated 45° and right-aligned "
                       f"(`rotation=45, ha='right'`).",
         f"fig, ax = plt.subplots()\nax.imshow(cm)\nax.set_xticks(range({k}), names, rotation=45, ha='right')",
         {"post": "fc.tick_labels(fc.ax(), 'x'), [(t.get_rotation(), t.get_ha()) for t in fc.ax().get_xticklabels()]"}),
        ("text", "Show `cm` with `imshow` and write each count in its cell: `ax.text(j, i, cm[i, j], ha='center', "
                 "va='center')` (column j is x, row i is y).",
         f"fig, ax = plt.subplots()\nax.imshow(cm)\nfor i in range({k}):\n    for j in range({k}):\n"
         f"        ax.text(j, i, cm[i, j], ha='center', va='center')", {"post": _post("_cells()")}),
        ("imshow", "Show `corr` with a diverging colormap centered on zero: `cmap='RdBu_r', vmin=-1, vmax=1`.",
         "fig, ax = plt.subplots()\nax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)", {"post": "fc.images(fc.ax(), data=True)"}),
        ("masked_where", "Hide the redundant upper triangle of `corr` (above the diagonal): `np.ma.masked_where(np.triu("
                         "np.ones_like(corr, dtype=bool), k=1), corr)`, and show the masked array with `imshow`.",
         "fig, ax = plt.subplots()\nax.imshow(np.ma.masked_where(np.triu(np.ones_like(corr, dtype=bool), k=1), corr))",
         {"post": _post("[_img(i) for i in fc.ax().get_images()]"), "require": ["masked_where", "triu"]}),
        ("imshow", "Show `cm` with `origin='lower'` (row 0 at the bottom, like a sweep heatmap with growing values).",
         "fig, ax = plt.subplots()\nax.imshow(cm, origin='lower')", {"post": "fc.images(fc.ax(), data=True)"}),
        ("legend_elements", "Scatter the embedding `emb` colored by `lab` (`c=lab, cmap='tab10'`) and build the legend "
                            "from the scatter itself: `ax.legend(*sc.legend_elements(), title='class')`.",
         "fig, ax = plt.subplots()\nsc = ax.scatter(emb[:, 0], emb[:, 1], c=lab, cmap='tab10')\nax.legend(*sc.legend_elements(), title='class')",
         {"post": "fc.scatter(fc.ax(), 0, 6, 'array', 'cmap'), fc.legend(), fc.legend_title()"}),
        ("contour", "Draw only the decision boundary of the probability map `proba`: `ax.contour(xx, yy, proba, "
                    "levels=[0.5], colors='k')`.",
         "fig, ax = plt.subplots()\nax.contour(xx, yy, proba, levels=[0.5], colors='k')",
         {"post": "[(type(c).__name__, fc.r(c.levels, 3)) for c in fc.ax().collections]"}),
        ("set_xticks", "An embedding's axes carry no meaning: scatter `emb` and remove all ticks with `ax.set_xticks([])` "
                       "and `ax.set_yticks([])`.",
         "fig, ax = plt.subplots()\nax.scatter(emb[:, 0], emb[:, 1])\nax.set_xticks([])\nax.set_yticks([])",
         {"post": _post("fc.scatter()", "_ticks()"), "require": ["set_xticks", "set_yticks"]}),
        ("axis", "Show `cm` as an image without any axis decoration (`ax.axis('off')`).",
         "fig, ax = plt.subplots()\nax.imshow(cm)\nax.axis('off')", {"post": "fc.images(fc.ax()), fc.ax().axison"}),
    ]
    return catalog(rng, items, setup, f"`cm` is a {k} × {k} confusion matrix over `names`, `corr` a 5 × 5 correlation "
                                      f"matrix, `emb` / `lab` a 2-D embedding with classes, and `proba` a probability map "
                                      f"over the grid `xx`, `yy`. Create `fig, ax`.")


# ================================================================= easy

@template(M, "easy")
def curve_math(rng):
    code, desc, n = curves(rng)
    beta = rng.choice([0.6, 0.8, 0.9])
    pat = rng.randint(2, 5)
    w = rng.choice([3, 5])
    warm = rng.choice([10, 20, 50])
    total = rng.choice([200, 500, 1000])
    k = rng.choice([50, 100])
    items = [
        ("zeros_like", f"Smooth `val_loss` like TensorBoard's slider: an exponential moving average with weight {beta}, "
                       f"`ema[0] = val_loss[0]`, `ema[i] = {beta} * ema[i - 1] + {1 - beta:.1f} * val_loss[i]`. Return "
                       f"`ema` (start from `np.zeros_like(val_loss)`).",
         f"ema = np.zeros_like(val_loss)\nema[0] = val_loss[0]\nfor i in range(1, len(val_loss)):\n"
         f"    ema[i] = {beta} * ema[i - 1] + {round(1 - beta, 1)} * val_loss[i]\nresult = ema"),
        ("exp", f"Debiased EMA (as in Adam): start `m = 0`, update `m = {beta} * m + {round(1 - beta, 1)} * x` for each "
                f"`x` in `train_loss`, and record `m / (1 - {beta} ** t)` after the t-th value (t = 1, 2, …). Return the "
                f"array.",
         f"m, out = 0.0, []\nfor t, x in enumerate(train_loss, start=1):\n    m = {beta} * m + {round(1 - beta, 1)} * x\n"
         f"    out.append(m / (1 - {beta} ** t))\nresult = np.array(out)", {"require": []}),
        ("argmin", "Return the best epoch number (the entry of `epochs` where `val_loss` is lowest).",
         "result = int(epochs[val_loss.argmin()])"),
        ("argmin", f"Early stopping with patience {pat}: training stops at the first epoch that is {pat} epochs past the "
                   f"best `val_loss` so far without improving on it. Return that epoch number (or `epochs[-1]` if it "
                   f"never stops).",
         f"best, stop = np.inf, int(epochs[-1])\nsince = 0\nfor e, v in zip(epochs, val_loss):\n    if v < best:\n"
         f"        best, since = v, 0\n    else:\n        since += 1\n        if since == {pat}:\n            stop = int(e)\n"
         f"            break\nresult = stop", {"require": []}),
        ("where", f"Build the guide's schedule for {total} steps: linear warmup over {warm} steps, then cosine decay to 0, "
                  f"peak 3e-4: `np.where(steps < warm, steps / warm, 0.5 * (1 + np.cos(np.pi * (steps - warm) / (total - "
                  f"warm)))) * 3e-4` with `steps = np.arange({total})`. Return it.",
         f"steps = np.arange({total})\nwarm, total = {warm}, {total}\n"
         f"result = np.where(steps < warm, steps / warm, 0.5 * (1 + np.cos(np.pi * (steps - warm) / (total - warm)))) * 3e-4",
         {"require": ["where", "cos"]}),
        ("arange", f"Step decay: the learning rate starts at 0.1 and is multiplied by 0.1 every {k} steps. Return it for "
                   f"`np.arange({3 * k})` (vectorized: `0.1 * 0.1 ** (steps // {k})`).",
         f"steps = np.arange({3 * k})\nresult = 0.1 * 0.1 ** (steps // {k})"),
        ("argmin", "Return the generalization gap at the best epoch: `val_loss - train_loss` where `val_loss` is lowest.",
         "b = val_loss.argmin()\nresult = float(val_loss[b] - train_loss[b])"),
        ("convolve", f"Smooth `train_loss` with a {w}-epoch moving average (`np.convolve(train_loss, np.ones({w}) / {w}, "
                     f"mode='valid')`) and return it.", f"result = np.convolve(train_loss, np.ones({w}) / {w}, mode='valid')"),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "easy")
def matrix_math(rng):
    code, k, names = _cm(rng)
    s = seeded(rng)
    setup = code + (f"\ng = np.random.default_rng({s})\natt = np.round(g.random((5, 5)), 3)\n"
                    f"corr = np.round(np.corrcoef(g.normal(size=(5, 50))), 3)\nlrs, wds = np.logspace(-4, -2, 5), np.logspace(-5, -3, 4)\n"
                    f"acc = np.round(g.uniform(0.7, 0.95, (4, 5)), 3)   # rows: weight decay, columns: learning rate")
    items = [
        ("sum", "Normalize each row of `cm` to sum to 1 (the diagonal becomes per-class recall) and return it.",
         "result = cm / cm.sum(axis=1, keepdims=True)"),
        ("max", "Heatmap cells need readable text: return a nested list with 'white' for every cell of `cm` above half "
                "the largest count and 'black' otherwise.",
         "result = [['white' if v > cm.max() / 2 else 'black' for v in row] for row in cm]"),
        ("trace", "Return the overall accuracy from `cm`: correct predictions (the diagonal) over all samples.",
         "result = np.trace(cm) / cm.sum()"),
        ("sum", "Return the per-class **precision** from `cm` (diagonal over column sums).",
         "result = np.diag(cm) / cm.sum(axis=0)"),
        ("argmax", "Return the most frequent mistake as `[true name, predicted name]`: the largest off-diagonal cell of "
                   "`cm`.", "off = cm - np.diag(np.diag(cm))\ni, j = np.unravel_index(off.argmax(), off.shape)\n"
                            "result = [names[i], names[j]]"),
        ("tril", "Make `att` a causal attention matrix: zero everything above the diagonal (`np.tril`) and normalize "
                 "each row to sum to 1. Return it.",
         "a = np.tril(att)\nresult = a / a.sum(axis=1, keepdims=True)", {"require": ["tril"]}),
        ("triu", "Mask the upper triangle of `corr` including the diagonal (`np.triu(np.ones_like(corr, dtype=bool))`, "
                 "k=0) with `np.ma.masked_where` and return how many cells are masked.",
         "masked = np.ma.masked_where(np.triu(np.ones_like(corr, dtype=bool)), corr)\nresult = int(masked.mask.sum())",
         {"require": ["triu", "masked_where"]}),
        ("argmax", "`acc[i, j]` is the accuracy for weight decay `wds[i]` and learning rate `lrs[j]`. Return the best "
                   "pair as `[lr, wd]`.", "i, j = np.unravel_index(acc.argmax(), acc.shape)\nresult = [lrs[j], wds[i]]"),
        ("logspace", "Rebuild the sweep's weight decays with `np.logspace(-5, -3, 4)` (4 values evenly spaced in log from "
                     "1e-5 to 1e-3) and return their tick labels, each formatted as `f'{v:.0e}'`.",
         "result = [f'{v:.0e}' for v in np.logspace(-5, -3, 4)]"),
        ("ndenumerate", "Return the annotations a sweep heatmap needs: `(i, j, f'{v:.2f}')` for every cell of `acc`, in "
                        "row order (`np.ndenumerate`).",
         "result = [(i, j, f'{v:.2f}') for (i, j), v in np.ndenumerate(acc)]"),
    ]
    return catalog(rng, items, setup, f"`cm` is a {k} × {k} confusion matrix over `names` (rows: true); `att` raw "
                                      f"attention scores; `corr` a correlation matrix; `acc` a hyperparameter sweep over "
                                      f"`wds` × `lrs`.")


@template(M, "easy")
def classifier_math(rng):
    _warm_sklearn()
    code, n = _scores(rng)
    t = rng.choice([0.4, 0.5, 0.6])
    sharp = rng.choice([1.5, 2, 3, 4])
    setup = code + "\nthr = np.linspace(0, 1, 101)\nprecision = 0.5 + 0.5 * thr ** 0.5\nrecall = 1 - thr ** 2"
    items = [
        ("trapezoid", f"The guide fakes a ROC curve as `tpr = fpr ** (1 / {sharp})` on `fpr = np.linspace(0, 1, 100)`. "
                      f"Return `[its area by np.trapezoid, the exact area {sharp}/({sharp}+1)]`.",
         f"fpr = np.linspace(0, 1, 100)\nresult = [np.trapezoid(fpr ** (1 / {sharp}), fpr), {sharp} / ({sharp} + 1)]",
         {"tol": 1e-9}),
        ("sum", f"Predict positive when `scores >= {t}`. Return `[precision, recall]` against `y_true`.",
         f"pred = scores >= {t}\ntp = (pred & (y_true == 1)).sum()\nresult = [tp / pred.sum(), tp / (y_true == 1).sum()]"),
        ("argmax", "On the guide's curves (`precision`, `recall` over `thr`), return the threshold with the best F1 = "
                   "2PR / (P + R).", "f1 = 2 * precision * recall / (precision + recall)\nresult = thr[f1.argmax()]"),
        ("argmin", "Return the threshold where the `precision` and `recall` curves cross (smallest |P − R|).",
         "result = thr[np.abs(precision - recall).argmin()]"),
        ("predict_proba", "Fit `KNeighborsClassifier(15)` on `make_moons(150, noise=0.25, random_state=0)`, evaluate "
                          "`predict_proba` on a 30 × 30 grid (`np.meshgrid` over -2 … 3 and -1.5 … 2, stacked with "
                          "`np.c_[xx.ravel(), yy.ravel()]`), and return the class-1 probabilities reshaped to the grid's "
                          "shape — just their `shape` and their mean, as `[shape, mean]`.",
         "from sklearn.datasets import make_moons\nfrom sklearn.neighbors import KNeighborsClassifier\n"
         "X, y = make_moons(150, noise=0.25, random_state=0)\nclf = KNeighborsClassifier(15).fit(X, y)\n"
         "xx, yy = np.meshgrid(np.linspace(-2, 3, 30), np.linspace(-1.5, 2, 30))\n"
         "p = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)\nresult = [p.shape, p.mean()]",
         {"require": ["predict_proba", "c_", "reshape"]}),
        ("roc_auc_score", "Return the ROC AUC of `scores` against `y_true` with `sklearn.metrics.roc_auc_score`, and "
                          "check it by its meaning: the fraction of (positive, negative) pairs ranked correctly, ties "
                          "counting half. Return `[sklearn, by_pairs]`.",
         "from sklearn.metrics import roc_auc_score\npos, neg = scores[y_true == 1], scores[y_true == 0]\n"
         "d = pos[:, None] - neg[None, :]\nresult = [roc_auc_score(y_true, scores), ((d > 0) + 0.5 * (d == 0)).mean()]",
         {"tol": 1e-9}),
        ("format", "Build ROC legend labels for the models `[('model A', 3), ('model B', 1.5)]` (name, sharpness s): "
                   "`f'{name} (AUC={s / (s + 1):.2f})'`. Return the list.",
         "result = [f'{name} (AUC={s / (s + 1):.2f})' for name, s in [('model A', 3), ('model B', 1.5)]]", {"require": []}),
        ("roc_curve", "Return the number of points sklearn's `roc_curve(y_true, scores)` keeps by default and with "
                      "`drop_intermediate=False`, as a list.",
         "from sklearn.metrics import roc_curve\nresult = [len(roc_curve(y_true, scores)[0]), "
         "len(roc_curve(y_true, scores, drop_intermediate=False)[0])]"),
    ]
    return catalog(rng, items, setup, f"`scores` are a binary classifier's scores for {n} samples with labels `y_true`; "
                                      f"`precision` / `recall` are the guide's synthetic curves over `thr`.")


@template(M, "easy")
def recipe_small_plots(rng):
    code, n = _regression(rng)
    s = seeded(rng)
    beta = rng.choice([0.8, 0.9, 0.95])
    setup = code + (f"\ng2 = np.random.default_rng({s})\nraw = np.round(np.exp(-np.arange(200) / 60) + 0.1 * g2.normal(size=200), 4)\n"
                    f"img = np.round(g2.random((28, 28)), 3)\ny_t, y_p = {rng.randint(0, 9)}, {rng.randint(0, 9)}\n"
                    f"w = np.round(g2.normal(0, 0.3, 800), 4)")
    ema = (f"ema = np.zeros_like(raw)\nema[0] = raw[0]\nfor i in range(1, len(raw)):\n"
           f"    ema[i] = {beta} * ema[i - 1] + {round(1 - beta, 2)} * raw[i]\n")
    items = [
        ("plot", f"The TensorBoard look: plot `raw` faintly (`alpha=0.3, color='C0'`) and its EMA (weight {beta}) on top "
                 f"in 'C0' labelled 'EMA {beta}', with a legend.",
         ema + f"fig, ax = plt.subplots()\nax.plot(raw, alpha=0.3, color='C0')\nax.plot(ema, color='C0', label='EMA {beta}')\nax.legend()",
         {"post": "fc.lines(fc.ax(), 'alpha', 'color', 'label', nd=5), fc.legend()"}),
        ("scatter", "Predicted vs actual: scatter `y_hat` against `y_act` (`s=8, alpha=0.6`), draw the identity line "
                    "from `y_act.min()` to `y_act.max()` as `'k--'`, and label x 'actual', y 'predicted'.",
         "fig, ax = plt.subplots()\nax.scatter(y_act, y_hat, s=8, alpha=0.6)\nlims = [y_act.min(), y_act.max()]\n"
         "ax.plot(lims, lims, 'k--')\nax.set(xlabel='actual', ylabel='predicted')",
         {"post": "fc.scatter(fc.ax(), 0, 6, 'sizes', 'alpha'), fc.lines(fc.ax(), 'color', 'ls'), fc.labels()"}),
        ("axhline", "Residuals: scatter `y_hat - y_act` against `y_hat`, add a black zero line (`axhline`), label x "
                    "'predicted', y 'residual'.",
         "fig, ax = plt.subplots()\nax.scatter(y_hat, y_hat - y_act)\nax.axhline(0, color='k')\nax.set(xlabel='predicted', ylabel='residual')",
         {"post": "fc.scatter(), fc.lines(fc.ax(), 'color'), fc.labels()"}),
        ("hist", "Histogram the residuals `y_hat - y_act` with 20 bins and mark their mean with a red vertical line.",
         "fig, ax = plt.subplots()\nr = y_hat - y_act\nax.hist(r, bins=20)\nax.axvline(r.mean(), color='red')",
         {"post": "fc.bars(fc.ax(), 4), fc.lines(fc.ax(), 'color')"}),
        ("set_title", "Show the digit image `img` in gray without axes, titled `f'{y_p} ({y_t})'` (prediction, then truth "
                      "in parentheses) in 'green' if they match, else 'red'.",
         "fig, ax = plt.subplots()\nax.imshow(img, cmap='gray')\nax.set_title(f'{y_p} ({y_t})', color='green' if y_p == y_t else 'red')\nax.axis('off')",
         {"post": "fc.images(fc.ax()), fc.ax().get_title(), fc.hexcolor(fc.ax().title.get_color()), fc.ax().axison"}),
        ("hist", "Histogram the weights `w` (40 bins) and title it with the layer's spread `f'layer0 σ={w.std():.2f}'`.",
         "fig, ax = plt.subplots()\nax.hist(w, bins=40)\nax.set_title(f'layer0 σ={w.std():.2f}')",
         {"post": "fc.bars(fc.ax(), 4), fc.labels()['title']"}),
        ("set_xlabel", "Plot the EMA-smoothed `raw` (weight 0.9; `ema[0] = raw[0]`) only, with x label 'step' "
                       "(`set_xlabel`) and y label 'loss' (`set_ylabel`), y on a log scale.",
         "ema = np.zeros_like(raw)\nema[0] = raw[0]\nfor i in range(1, len(raw)):\n    ema[i] = 0.9 * ema[i - 1] + 0.1 * raw[i]\n"
         "fig, ax = plt.subplots()\nax.plot(ema)\nax.set_xlabel('step')\nax.set_ylabel('loss')\nax.set_yscale('log')",
         {"post": "fc.lines(fc.ax(), nd=5), fc.labels(), fc.scales()", "require": ["set_xlabel", "set_yscale"]}),
    ]
    return catalog(rng, items, setup, f"`y_act` / `y_hat` hold {n} regression targets and predictions, `raw` a noisy "
                                      f"per-step loss, `img` a 28 × 28 image with true / predicted digits `y_t` / `y_p`, "
                                      f"`w` a layer's weights. Create `fig, ax`.")


# ================================================================= medium

@template(M, "medium")
def training_curve_recipes(rng):
    code, desc, n = curves(rng, acc=True)
    warm = rng.choice([50, 100, 200])
    total = rng.choice([1000, 2000])
    items = [
        ("set_yscale", "Recipe 1: on `fig, ax = plt.subplots(figsize=(7, 4), layout='constrained')`, plot `train_loss` "
                       "('train') and `val_loss` ('val') on a log y-axis, mark the best epoch with a dashed gray "
                       "`axvline` and a red dot (`scatter`, `zorder=3`) labelled `f'best @ {epochs[best]}'`, label the "
                       "axes 'epoch' / 'loss' and add a legend.",
         "fig, ax = plt.subplots(figsize=(7, 4), layout='constrained')\nax.plot(epochs, train_loss, label='train')\n"
         "ax.plot(epochs, val_loss, label='val')\nax.set_yscale('log')\nbest = val_loss.argmin()\n"
         "ax.axvline(epochs[best], ls='--', c='gray')\nax.scatter(epochs[best], val_loss[best], c='red', zorder=3, "
         "label=f'best @ {epochs[best]}')\nax.set(xlabel='epoch', ylabel='loss')\nax.legend()",
         {"post": "fc.lines(fc.ax(), 'label', 'ls', 'color'), fc.scatter(fc.ax(), 0, 6, 'color', 'label'), fc.scales(), "
                  "fc.labels(), fc.legend(), fc.size()"}),
        ("annotate", "Plot both losses and point at the best validation loss with `ax.annotate(f'{val_loss[best]:.3f}', "
                     "xy=(epochs[best], val_loss[best]), xytext=(epochs[best], val_loss[best] + 0.3), arrowprops=dict("
                     "arrowstyle='->'))`.",
         "fig, ax = plt.subplots()\nax.plot(epochs, train_loss)\nax.plot(epochs, val_loss)\nbest = val_loss.argmin()\n"
         "ax.annotate(f'{val_loss[best]:.3f}', xy=(epochs[best], val_loss[best]), xytext=(epochs[best], val_loss[best] + 0.3), "
         "arrowprops=dict(arrowstyle='->'))", {"post": "fc.lines(fc.ax()), fc.text_items()"}),
        ("axvspan", "Plot both losses and shade where the model overfits — from the best epoch to the last one — with "
                    "`ax.axvspan(..., color='red', alpha=0.1)`, plus the legend entries 'train', 'val'.",
         "fig, ax = plt.subplots()\nax.plot(epochs, train_loss, label='train')\nax.plot(epochs, val_loss, label='val')\n"
         "ax.axvspan(epochs[val_loss.argmin()], epochs[-1], color='red', alpha=0.1)\nax.legend()",
         {"post": "fc.lines(fc.ax(), 'label'), [fc.r((p.get_x(), p.get_width()), 4) for p in fc.ax().patches], fc.legend()"}),
        ("ticklabel_format", f"Recipe 8: build the warmup + cosine schedule (`steps = np.arange({total})`, warmup {warm}, "
                             f"peak 3e-4 — the guide's formula), plot it on `figsize=(6, 2.5)`, label 'step' / 'lr', and "
                             f"use scientific y tick labels (`ticklabel_format(axis='y', style='sci', scilimits=(0, 0))`).",
         f"steps = np.arange({total})\nwarm = {warm}\nlr = np.where(steps < warm, steps / warm, 0.5 * (1 + np.cos(np.pi * "
         f"(steps - warm) / ({total} - warm)))) * 3e-4\nfig, ax = plt.subplots(figsize=(6, 2.5))\nax.plot(steps, lr)\n"
         f"ax.set(xlabel='step', ylabel='lr')\nax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))",
         {"post": "f = fc.ax().yaxis.get_major_formatter()\nfc.lines(fc.ax(), nd=9), fc.labels(), f._scientific, list(f._powerlimits)",
          "require": ["where", "cos", "ticklabel_format"]}),
        ("axvline", "Recipe 13: plot the guide's synthetic `precision = 0.5 + 0.5 * thr ** 0.5` and `recall = 1 - thr ** "
                    "2` over `thr = np.linspace(0, 1, 101)` (labels 'precision', 'recall'), mark the threshold with the "
                    "best F1 by a dotted gray vertical line, and add a legend.",
         "thr = np.linspace(0, 1, 101)\nprecision = 0.5 + 0.5 * thr ** 0.5\nrecall = 1 - thr ** 2\n"
         "fig, ax = plt.subplots()\nax.plot(thr, precision, label='precision')\nax.plot(thr, recall, label='recall')\n"
         "f1 = 2 * precision * recall / (precision + recall)\nax.axvline(thr[f1.argmax()], ls=':', c='gray')\nax.legend()",
         {"post": "fc.lines(fc.ax(), 'label', 'ls', 'color'), fc.legend()"}),
        ("twinx", "Loss and accuracy on one plot: `val_loss` on `ax` ('C0', y label 'loss'), `val_acc` on `ax.twinx()` "
                  "('C1', y label 'accuracy', limits 0 … 1), x label 'epoch'.",
         "fig, ax = plt.subplots()\nax.plot(epochs, val_loss, color='C0')\nax.set(xlabel='epoch', ylabel='loss')\nax2 = ax.twinx()\n"
         "ax2.plot(epochs, val_acc, color='C1')\nax2.set_ylabel('accuracy')\nax2.set_ylim(0, 1)",
         {"post": "[(fc.lines(a, 'color'), fc.labels(a)['ylabel'], fc.limits(a)['ylim'] if i else None) for i, a in enumerate(fc.axes())], "
                  "fc.labels(fc.ax(0))['xlabel']"}),
        ("plt.subplots", "Two panels (`plt.subplots(1, 2, figsize=(8, 3))`): left both losses on a log scale with a legend "
                         "('train', 'val'), right `val_acc` with a horizontal dashed line at its maximum; titles 'loss' "
                         "and 'accuracy'.",
         "fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3))\na1.plot(epochs, train_loss, label='train')\n"
         "a1.plot(epochs, val_loss, label='val')\na1.set_yscale('log')\na1.legend()\na1.set_title('loss')\n"
         "a2.plot(epochs, val_acc)\na2.axhline(val_acc.max(), ls='--')\na2.set_title('accuracy')",
         {"post": "[(fc.lines(a, 'label', 'ls'), fc.scales(a), fc.legend(a), fc.labels(a)['title']) for a in fc.axes()]"}),
    ]
    return catalog(rng, items, code, f"{desc}. Build each figure to the spec.")


@template(M, "medium")
def confusion_recipes(rng):
    code, k, names = _cm(rng)
    cmap = rng.choice(["Blues", "Greens", "Oranges"])
    head = (f"fig, ax = plt.subplots(figsize=(4.5, 4), layout='constrained')\nim = ax.imshow(cm, cmap={cmap!r})\n"
            f"fig.colorbar(im, ax=ax)\nax.set_xticks(range({k}), names)\nax.set_yticks(range({k}), names)\n"
            f"ax.set(xlabel='predicted', ylabel='true')\n")
    full = "fc.images(fc.ax(0), data=True), _ticks(fc.ax(0)), fc.labels(fc.ax(0)), len(fc.colorbars())"
    items = [
        ("text", f"Recipe 2: show `cm` (`cmap={cmap!r}`) with a colorbar, class names on both axes, x label 'predicted', "
                 f"y label 'true', and each count written in its cell (centered) in 'white' when it is above half the "
                 f"largest count, else 'black'.",
         head + f"for i in range({k}):\n    for j in range({k}):\n        ax.text(j, i, cm[i, j], ha='center', va='center', "
                f"color='white' if cm[i, j] > cm.max() / 2 else 'black')",
         {"post": _post(full, "_cells(fc.ax(0))", "_cellcolors(fc.ax(0))")}),
        ("text", f"Normalized confusion matrix: divide each row by its sum, show it with `cmap={cmap!r}, vmin=0, vmax=1` "
                 f"and a colorbar, class names on both axes, and write each value as `f'{{v:.2f}}'` centered in its cell.",
         f"cmn = cm / cm.sum(axis=1, keepdims=True)\nfig, ax = plt.subplots()\nim = ax.imshow(cmn, cmap={cmap!r}, vmin=0, vmax=1)\n"
         f"fig.colorbar(im, ax=ax)\nax.set_xticks(range({k}), names)\nax.set_yticks(range({k}), names)\n"
         f"for (i, j), v in np.ndenumerate(cmn):\n    ax.text(j, i, f'{{v:.2f}}', ha='center', va='center')",
         {"post": _post("fc.images(fc.ax(0), data=True)", "_ticks(fc.ax(0))", "_cells(fc.ax(0))")}),
        ("set_yticks", f"Show `cm` (`cmap={cmap!r}`) with the predicted names on x and, on y, each true class with its "
                       f"recall: `f'{{name}} ({{r:.0%}})'` where r = diagonal / row sum.",
         f"fig, ax = plt.subplots()\nax.imshow(cm, cmap={cmap!r})\nrec = np.diag(cm) / cm.sum(axis=1)\n"
         f"ax.set_xticks(range({k}), names)\nax.set_yticks(range({k}), [f'{{n}} ({{r:.0%}})' for n, r in zip(names, rec)])",
         {"post": _post("_ticks()")}),
        ("masked_where", "Show only the **errors**: mask the diagonal of `cm` (`np.ma.masked_where(np.eye(len(cm), "
                         "dtype=bool), cm)`), show it with `cmap='Reds'`, class names on both axes, and write the "
                         "off-diagonal counts in their cells.",
         f"err = np.ma.masked_where(np.eye(len(cm), dtype=bool), cm)\nfig, ax = plt.subplots()\nax.imshow(err, cmap='Reds')\n"
         f"ax.set_xticks(range({k}), names)\nax.set_yticks(range({k}), names)\nfor i in range({k}):\n    for j in range({k}):\n"
         f"        if i != j:\n            ax.text(j, i, cm[i, j], ha='center', va='center')",
         {"post": _post("[_img(i) for i in fc.ax().get_images()]", "_ticks()", "_cells()"), "require": ["masked_where"]}),
        ("Rectangle", f"Recipe 2 without the text, but outline the correct predictions: show `cm` (`cmap={cmap!r}`), class "
                      f"names on both axes, and draw an unfilled black `Rectangle((i - 0.5, i - 0.5), 1, 1)` around each "
                      f"diagonal cell.",
         f"from matplotlib.patches import Rectangle\nfig, ax = plt.subplots()\nax.imshow(cm, cmap={cmap!r})\n"
         f"ax.set_xticks(range({k}), names)\nax.set_yticks(range({k}), names)\nfor i in range({k}):\n"
         f"    ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor='k'))",
         {"post": _post("fc.images(fc.ax(), data=True)", "_ticks()", "fc.patches(fc.ax())", "[fc.hexcolor(p.get_edgecolor()) for p in fc.ax().patches]")}),
        ("colorbar", f"Show `cm` (`cmap={cmap!r}`) with a **horizontal** colorbar below it labelled 'count' "
                     f"(`fig.colorbar(im, ax=ax, orientation='horizontal', label='count')`), class names on both axes.",
         f"fig, ax = plt.subplots()\nim = ax.imshow(cm, cmap={cmap!r})\nfig.colorbar(im, ax=ax, orientation='horizontal', label='count')\n"
         f"ax.set_xticks(range({k}), names)\nax.set_yticks(range({k}), names)",
         {"post": _post("fc.images(fc.ax(0), data=True)", "fc.colorbars()", "_ticks(fc.ax(0))")}),
        ("plt.subplots", f"Counts and normalized side by side (`plt.subplots(1, 2, figsize=(8, 3.5))`): left `cm`, right "
                         f"the row-normalized matrix with `vmin=0, vmax=1`, both `cmap={cmap!r}` with class names on both "
                         f"axes, titled 'counts' and 'normalized'.",
         f"fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.5))\nfor a, m, t in ((a1, cm, 'counts'), (a2, cm / cm.sum(axis=1, keepdims=True), 'normalized')):\n"
         f"    a.imshow(m, cmap={cmap!r}, vmin=0 if t == 'normalized' else None, vmax=1 if t == 'normalized' else None)\n"
         f"    a.set_xticks(range({k}), names)\n    a.set_yticks(range({k}), names)\n    a.set_title(t)",
         {"post": _post("[(fc.images(a, 4, data=True), _ticks(a), fc.labels(a)['title']) for a in fc.axes()]")}),
    ]
    return catalog(rng, items, code, f"`cm` is a {k} × {k} confusion matrix (rows: true class, columns: predicted) over "
                                     f"`names`. Create the figure yourself.")


@template(M, "medium")
def heatmap_recipes(rng):
    s = seeded(rng)
    toks = rng.choice([["The", "cat", "sat", "on", "the", "mat"], ["I", "love", "deep", "learning", "!"],
                       ["Paris", "is", "the", "capital", "of", "France"], ["a", "dog", "ran", "home"]])
    nt = len(toks)
    setup = (f"g = np.random.default_rng({s})\ntokens = {lit(toks)}\natt = np.tril(g.random(({nt}, {nt})))\n"
             f"att = np.round(att / att.sum(1, keepdims=True), 3)   # causal attention, rows sum to 1\n"
             f"corr = np.round(np.corrcoef(g.normal(size=(6, 100)) + g.normal(size=100) * 0.5), 3)\n"
             f"lrs, wds = np.logspace(-4, -2, 5), np.logspace(-5, -3, 4)\nacc = np.round(g.uniform(0.7, 0.95, (4, 5)), 3)")
    sweep = ("fig, ax = plt.subplots()\nim = ax.imshow(acc, cmap='viridis', origin='lower')\n"
             "ax.set_xticks(range(5), [f'{v:.0e}' for v in lrs])\nax.set_yticks(range(4), [f'{v:.0e}' for v in wds])\n"
             "ax.set(xlabel='lr', ylabel='weight decay')\n")
    masked = "masked = np.ma.masked_where(np.triu(np.ones_like(corr, dtype=bool), k=1), corr)\n"
    items = [
        ("set_xticks", f"Recipe 5: on `plt.subplots(figsize=(5, 4.5), layout='constrained')` show `att` (viridis) with a "
                       f"colorbar, the tokens on both axes (x rotated 45°, `ha='right'`), x label 'key', y label 'query'.",
         f"fig, ax = plt.subplots(figsize=(5, 4.5), layout='constrained')\nim = ax.imshow(att, cmap='viridis')\nfig.colorbar(im, ax=ax)\n"
         f"ax.set_xticks(range({nt}), tokens, rotation=45, ha='right')\nax.set_yticks(range({nt}), tokens)\nax.set(xlabel='key', ylabel='query')",
         {"post": _post("fc.images(fc.ax(0), data=True)", "_ticks(fc.ax(0))", "[(t.get_rotation(), t.get_ha()) for t in fc.ax(0).get_xticklabels()]",
                        "fc.labels(fc.ax(0))", "len(fc.colorbars())")}),
        ("masked_where", "Causal attention without the empty half: mask the zeros above the diagonal (`np.ma.masked_where("
                         "np.triu(np.ones_like(att, dtype=bool), k=1), att)`), show it with `cmap='magma'`, tokens on both "
                         "axes, and write each visible weight as `f'{v:.2f}'` (centered, font size 7).",
         f"m = np.ma.masked_where(np.triu(np.ones_like(att, dtype=bool), k=1), att)\nfig, ax = plt.subplots()\n"
         f"ax.imshow(m, cmap='magma')\nax.set_xticks(range({nt}), tokens)\nax.set_yticks(range({nt}), tokens)\n"
         f"for i in range({nt}):\n    for j in range(i + 1):\n        ax.text(j, i, f'{{att[i, j]:.2f}}', ha='center', va='center', fontsize=7)",
         {"post": _post("[_img(i) for i in fc.ax().get_images()]", "_ticks()", "_cells()")}),
        ("masked_where", "Recipe 12: mask the upper triangle of `corr` (above the diagonal), show it with "
                         "`cmap='RdBu_r', vmin=-1, vmax=1` and a colorbar.",
         masked + "fig, ax = plt.subplots()\nim = ax.imshow(masked, cmap='RdBu_r', vmin=-1, vmax=1)\nfig.colorbar(im, ax=ax)",
         {"post": _post("[_img(i) for i in fc.ax(0).get_images()]", "fc.images(fc.ax(0))", "len(fc.colorbars())")}),
        ("ndenumerate", "Recipe 12 with numbers: the masked lower-triangle `corr` heatmap (`cmap='RdBu_r', vmin=-1, "
                        "vmax=1`), labelled `f0` … `f5` on both axes, with each value **on or below** the diagonal written "
                        "as `f'{v:.2f}'` (centered, font size 7).",
         masked + "fig, ax = plt.subplots()\nax.imshow(masked, cmap='RdBu_r', vmin=-1, vmax=1)\n"
                  "ax.set_xticks(range(6), [f'f{i}' for i in range(6)])\nax.set_yticks(range(6), [f'f{i}' for i in range(6)])\n"
                  "for (i, j), v in np.ndenumerate(corr):\n    if j <= i:\n        ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7)",
         {"post": _post("[_img(i) for i in fc.ax().get_images()]", "_ticks()", "_cells()")}),
        ("ndenumerate", "Recipe 14: show the sweep `acc` (viridis, `origin='lower'`), tick labels `f'{v:.0e}'` for `lrs` "
                        "(x) and `wds` (y), labels 'lr' / 'weight decay', and every cell annotated `f'{v:.2f}'` in white, "
                        "font size 8, centered.",
         sweep + "for (i, j), val in np.ndenumerate(acc):\n    ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8, color='w')",
         {"post": _post("fc.images(fc.ax(), data=True)", "_ticks()", "fc.labels()", "_cells()", "_cellcolors()")}),
        ("Rectangle", "Recipe 14 with the winner marked: the sweep heatmap (viridis, `origin='lower'`, `f'{v:.0e}'` tick "
                      "labels) and an unfilled red `Rectangle` (line width 2) around the best cell.",
         "from matplotlib.patches import Rectangle\n" + sweep + "i, j = np.unravel_index(acc.argmax(), acc.shape)\n"
         "ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor='red', linewidth=2))",
         {"post": _post("fc.images(fc.ax(), data=True)", "_ticks()", "fc.patches(fc.ax())")}),
        ("text", "Recipe 14 with readable text everywhere: the sweep heatmap (viridis, `origin='lower'`, `f'{v:.0e}'` "
                 "ticks), cells annotated `f'{v:.2f}'`, colored 'black' when the value is above the midpoint of "
                 "`acc.min()` and `acc.max()` (bright cells) and 'white' otherwise.",
         sweep + "mid = (acc.min() + acc.max()) / 2\nfor (i, j), val in np.ndenumerate(acc):\n"
                 "    ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='black' if val > mid else 'white')",
         {"post": _post("fc.images(fc.ax(), data=True)", "_cells()", "_cellcolors()")}),
    ]
    return catalog(rng, items, setup, f"`att` is a causal attention matrix over `tokens`, `corr` a 6 × 6 feature "
                                      f"correlation matrix, and `acc[i, j]` a sweep's accuracy for weight decay `wds[i]` "
                                      f"and learning rate `lrs[j]`.")


@template(M, "medium")
def classifier_recipes(rng):
    _warm_sklearn()
    s = seeded(rng)
    k = rng.choice([5, 10, 15])
    noise = rng.choice([0.2, 0.25, 0.3])
    setup = (f"from sklearn.datasets import make_moons\nXm, ym = make_moons(150, noise={noise}, random_state={s % 100})\n"
             f"xx, yy = np.meshgrid(np.linspace(Xm[:, 0].min() - 0.5, Xm[:, 0].max() + 0.5, 40), "
             f"np.linspace(Xm[:, 1].min() - 0.5, Xm[:, 1].max() + 0.5, 40))\n"
             f"g = np.random.default_rng({s})\nemb = np.round(np.concatenate([g.normal(c, 0.6, size=(50, 2)) for c in "
             f"[(0, 0), (3, 1), (1, 4)]]), 3)\nlab = np.repeat([0, 1, 2], 50)")
    knn = (f"from sklearn.neighbors import KNeighborsClassifier\nclf = KNeighborsClassifier({k}).fit(Xm, ym)\n"
           f"proba = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)\n")
    cont = ("[(type(c).__name__, len(getattr(c, 'levels', [])), c.get_cmap().name if c.get_array() is not None else None) "
            "for c in fc.ax().collections]")
    items = [
        ("contourf", f"Recipe 3: fit `KNeighborsClassifier({k})` on `Xm, ym`, compute the class-1 probability on the grid "
                     f"(`predict_proba` on `np.c_[xx.ravel(), yy.ravel()]`, reshaped), and draw `contourf(xx, yy, proba, "
                     f"levels=20, cmap='RdBu_r', alpha=0.6)`, the 0.5 contour in black, and the points colored by `ym` "
                     f"(`cmap='RdBu_r', edgecolors='k', s=15`).",
         knn + "fig, ax = plt.subplots()\nax.contourf(xx, yy, proba, levels=20, cmap='RdBu_r', alpha=0.6)\n"
               "ax.contour(xx, yy, proba, levels=[0.5], colors='k')\nax.scatter(Xm[:, 0], Xm[:, 1], c=ym, cmap='RdBu_r', edgecolors='k', s=15)",
         {"post": f"{cont}, fc.r(fc.ax().collections[1].levels, 3), fc.scatter(fc.ax(), 0, 6, 'array', 'cmap', 'edgecolor', 'sizes')",
          "require": ["fit", "predict_proba", "contourf", "contour"]}),
        ("LogisticRegression", "Same recipe with a linear model: `LogisticRegression().fit(Xm, ym)`, its class-1 "
                               "probability on the grid as `contourf(..., levels=10, cmap='coolwarm', alpha=0.5)`, and "
                               "the points colored by `ym` (`cmap='coolwarm'`). The boundary is a straight line.",
         "from sklearn.linear_model import LogisticRegression\nclf = LogisticRegression().fit(Xm, ym)\n"
         "proba = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)\nfig, ax = plt.subplots()\n"
         "ax.contourf(xx, yy, proba, levels=10, cmap='coolwarm', alpha=0.5)\nax.scatter(Xm[:, 0], Xm[:, 1], c=ym, cmap='coolwarm')",
         {"post": f"{cont}, fc.scatter(fc.ax(), 0, 6, 'array')", "require": ["LogisticRegression", "predict_proba", "contourf"]}),
        ("contour", f"Compare two neighborhoods: fit `KNeighborsClassifier(1)` and `KNeighborsClassifier({k})` and draw "
                    f"both decision boundaries (the 0.5 contour of each class-1 probability) on one Axes in 'C0' and "
                    f"'C3', over the points (`c=ym`, `cmap='coolwarm'`, `s=10`).",
         f"from sklearn.neighbors import KNeighborsClassifier\nfig, ax = plt.subplots()\nax.scatter(Xm[:, 0], Xm[:, 1], c=ym, cmap='coolwarm', s=10)\n"
         f"for kk, col in ((1, 'C0'), ({k}, 'C3')):\n    clf = KNeighborsClassifier(kk).fit(Xm, ym)\n"
         f"    p = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)\n"
         f"    ax.contour(xx, yy, p, levels=[0.5], colors=col)",
         {"post": "[(type(c).__name__, fc.r(getattr(c, 'levels', []), 3)) for c in fc.ax().collections], "
                  "[fc.hexcolor(c.get_edgecolor()) for c in fc.ax().collections[1:]]"}),
        ("legend_elements", "Recipe 6: scatter the embedding `emb` colored by `lab` (`cmap='tab10', s=8, alpha=0.8`), a "
                            "legend built with `sc.legend_elements()` titled 'class', and no ticks on either axis.",
         "fig, ax = plt.subplots()\nsc = ax.scatter(emb[:, 0], emb[:, 1], c=lab, cmap='tab10', s=8, alpha=0.8)\n"
         "ax.legend(*sc.legend_elements(), title='class')\nax.set_xticks([])\nax.set_yticks([])",
         {"post": _post("fc.scatter(fc.ax(), 0, 6, 'array', 'cmap', 'alpha')", "fc.legend()", "fc.legend_title()", "_ticks()")}),
        ("scatter", "Recipe 6 with class names: scatter each class of `emb` separately (`lab == c`) labelled "
                    "`['cat', 'dog', 'bird'][c]` in color `f'C{c}'`, `s=8`, with a legend and no ticks.",
         "fig, ax = plt.subplots()\nfor c, name in enumerate(['cat', 'dog', 'bird']):\n"
         "    ax.scatter(emb[lab == c, 0], emb[lab == c, 1], s=8, color=f'C{c}', label=name)\nax.legend()\nax.set_xticks([])\nax.set_yticks([])",
         {"post": _post("[fc.scatter(fc.ax(), i, 6, 'color', 'label') for i in range(fc.n_scatters())]", "fc.legend()", "_ticks()")}),
        ("roc_curve", "A real ROC curve: fit `LogisticRegression()` on `Xm, ym`, get `fpr, tpr, _ = roc_curve(ym, "
                      "clf.predict_proba(Xm)[:, 1])`, plot it labelled `f'logreg (AUC={auc:.2f})'` with `auc = "
                      "roc_auc_score(...)`, add the chance diagonal `'k--'` (lw 1), equal aspect, labels 'FPR' / 'TPR', "
                      "legend in the lower right.",
         "from sklearn.linear_model import LogisticRegression\nfrom sklearn.metrics import roc_curve, roc_auc_score\n"
         "clf = LogisticRegression().fit(Xm, ym)\np = clf.predict_proba(Xm)[:, 1]\nfpr, tpr, _ = roc_curve(ym, p)\n"
         "auc = roc_auc_score(ym, p)\nfig, ax = plt.subplots(figsize=(4.5, 4.5))\nax.plot(fpr, tpr, label=f'logreg (AUC={auc:.2f})')\n"
         "ax.plot([0, 1], [0, 1], 'k--', lw=1)\nax.set(xlabel='FPR', ylabel='TPR', aspect='equal')\nax.legend(loc='lower right')",
         {"post": "fc.lines(fc.ax(), 'label', 'ls', nd=4), fc.labels(), fc.ax().get_aspect(), fc.legend(), fc.ax().get_legend()._loc",
          "require": ["roc_curve", "roc_auc_score"]}),
        ("plot", "Recipe 9: on `plt.subplots(figsize=(4.5, 4.5))` plot the guide's model curves `fpr ** (1 / sharp)` "
                 "(`fpr = np.linspace(0, 1, 100)`) for `[('model A', 3), ('model B', 1.5)]` labelled `f'{name} "
                 "(AUC={sharp / (sharp + 1):.2f})'`, the chance diagonal `'k--'` (lw 1), equal aspect, labels "
                 "'FPR' / 'TPR', legend lower right.",
         "fig, ax = plt.subplots(figsize=(4.5, 4.5))\nfpr = np.linspace(0, 1, 100)\nfor name, sharp in [('model A', 3), ('model B', 1.5)]:\n"
         "    ax.plot(fpr, fpr ** (1 / sharp), label=f'{name} (AUC={sharp / (sharp + 1):.2f})')\n"
         "ax.plot([0, 1], [0, 1], 'k--', lw=1)\nax.set(xlabel='FPR', ylabel='TPR', aspect='equal')\nax.legend(loc='lower right')",
         {"post": "fc.lines(fc.ax(), 'label', 'ls', 'lw', nd=4), fc.labels(), fc.ax().get_aspect(), fc.legend(), fc.ax().get_legend()._loc"}),
    ]
    return catalog(rng, items, setup, "`Xm`, `ym` are 150 two-moons points with labels, `xx`, `yy` a 40 × 40 grid "
                                      "around them, and `emb`, `lab` a 3-class 2-D embedding (150 points). Import "
                                      "scikit-learn pieces as needed.")


@template(M, "medium")
def diagnostic_recipes(rng):
    s = seeded(rng)
    code, k = _features(rng)
    L = rng.randint(3, 4)
    nimg = rng.choice([8, 12, 16])
    cols = nimg // 2
    setup = (f"g = np.random.default_rng({s})\nimgs = np.round(g.random(({nimg}, 8, 8)), 3)\ny_true = g.integers(0, 10, {nimg})\n"
             f"y_pred = np.where(g.random({nimg}) < 0.75, y_true, (y_true + 1) % 10)\n"
             f"layer_w = {{f'layer{{i}}': np.round(g.normal(0, 1 / (i + 1), 1000), 4) for i in range({L})}}\n"
             f"{code}\nerr = np.round(g.uniform(0.01, 0.08, len(imp)), 3)   # std of each importance\n"
             f"y_act = np.round(g.normal(size=80), 3)\ny_hat = np.round(y_act + g.normal(0, 0.3, 80), 3)")
    items = [
        ("set_title", f"Recipe 4: a 2 × {cols} grid (`plt.subplots(2, {cols}, figsize=(12, 3.5))`) of the images (gray, "
                      f"axes off), each titled `f'{{p}} ({{t}})'` (predicted, true) in 'green' if correct else 'red', "
                      f"font size 9.",
         f"fig, axes = plt.subplots(2, {cols}, figsize=(12, 3.5))\nfor ax, im_, t, p in zip(axes.flat, imgs, y_true, y_pred):\n"
         f"    ax.imshow(im_, cmap='gray')\n    ax.set_title(f'{{p}} ({{t}})', color='green' if p == t else 'red', fontsize=9)\n    ax.axis('off')",
         {"post": "[(a.get_title(), fc.hexcolor(a.title.get_color()), fc.images(a)[0]['cmap'], a.axison) for a in fc.axes()], fc.grid_shape()",
          "require": ["flat", "set_title"]}),
        ("set_title", "Only the mistakes: show the misclassified images in a 1 × m row (`plt.subplots(1, m, figsize=(2 * m, "
                      "2), squeeze=False)`, m = number of mistakes), gray, axes off, titled `f'pred {p} / true {t}'`.",
         "wrong = np.flatnonzero(y_pred != y_true)\nm = len(wrong)\nfig, axes = plt.subplots(1, m, figsize=(2 * m, 2), squeeze=False)\n"
         "for ax, i in zip(axes.flat, wrong):\n    ax.imshow(imgs[i], cmap='gray')\n    ax.set_title(f'pred {y_pred[i]} / true {y_true[i]}')\n"
         "    ax.axis('off')", {"post": "[(a.get_title(), fc.images(a, 3, data=True), a.axison) for a in fc.axes()]"}),
        ("hist", f"Recipe 7: one histogram per layer side by side (`plt.subplots(1, {L}, figsize=(12, 2.5), sharey=True, "
                 f"layout='constrained')`), 60 bins each, titled `f'{{name}} σ={{wv.std():.2f}}'` (font size 9), looping "
                 f"over `layer_w.items()`.",
         f"fig, axes = plt.subplots(1, {L}, figsize=(12, 2.5), sharey=True, layout='constrained')\n"
         f"for ax, (name, wv) in zip(axes, layer_w.items()):\n    ax.hist(wv, bins=60)\n    ax.set_title(f'{{name}} σ={{wv.std():.2f}}', fontsize=9)",
         {"post": "[(len(fc.bars(a)), fc.r(sum(fc.heights(a)), 3), a.get_title()) for a in fc.axes()], "
                  "fc.ax(0).get_shared_y_axes().joined(fc.ax(0), fc.ax(1))", "require": ["hist", "items"]}),
        ("hist", "All layers on **one** Axes with shared bins (`np.linspace(-3, 3, 61)`), as step outlines "
                 "(`histtype='step'`) labelled by layer name, plus a legend: dead (narrow) layers stand out.",
         "fig, ax = plt.subplots()\nbins = np.linspace(-3, 3, 61)\nfor name, wv in layer_w.items():\n"
         "    ax.hist(wv, bins=bins, histtype='step', label=name)\nax.legend()",
         {"post": "fc.legend(), [type(p).__name__ for p in fc.ax().patches], [fc.r(p.get_xy()[:, 1].sum(), 3) for p in fc.ax().patches]"}),
        ("barh", "Recipe 10: sorted feature importances — `order = np.argsort(imp)`, `ax.barh(feats[order], imp[order], "
                 "color='C2')` on `figsize=(5, 4)`, x label 'importance' (most important at the top).",
         "order = np.argsort(imp)\nfig, ax = plt.subplots(figsize=(5, 4))\nax.barh(feats[order], imp[order], color='C2')\nax.set_xlabel('importance')",
         {"post": "fc.bars(fc.ax(), 4, 'color'), fc.tick_labels(fc.ax(), 'y'), fc.labels()['xlabel']", "require": ["barh", "argsort"]}),
        ("barh", "Top 5 with error bars: the five most important features, sorted so the largest is on top, as "
                 "`ax.barh(names, values, xerr=errs, color='C0', capsize=3)` (errors from `err`).",
         "order = np.argsort(imp)[-5:]\nfig, ax = plt.subplots()\nax.barh(feats[order], imp[order], xerr=err[order], color='C0', capsize=3)",
         {"post": "fc.bars(fc.ax(), 4), fc.tick_labels(fc.ax(), 'y'), "
                  "[fc.r(np.ravel(s)[:2], 3) for c in fc.ax().collections for s in [c.get_segments()]][:1]"}),
        ("plt.subplots", "Recipe 11: regression diagnostics on `plt.subplots(1, 2, figsize=(9, 4), layout='constrained')`: "
                         "left `y_hat` vs `y_act` (`s=8, alpha=0.6`) with the dashed black identity line from "
                         "`y_act.min()` to `y_act.max()`, labels 'actual' / 'predicted'; right the residuals `y_hat - "
                         "y_act` vs `y_hat` with a black zero line, labels 'predicted' / 'residual'.",
         "fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4), layout='constrained')\na1.scatter(y_act, y_hat, s=8, alpha=0.6)\n"
         "lims = [y_act.min(), y_act.max()]\na1.plot(lims, lims, 'k--')\na1.set(xlabel='actual', ylabel='predicted')\n"
         "a2.scatter(y_hat, y_hat - y_act, s=8, alpha=0.6)\na2.axhline(0, color='k')\na2.set(xlabel='predicted', ylabel='residual')",
         {"post": "[(fc.scatter(a, 0, 6, 'sizes', 'alpha'), fc.lines(a, 'color', 'ls'), fc.labels(a)) for a in fc.axes()]"}),
    ]
    return catalog(rng, items, setup, f"`imgs` are {nimg} small images with labels `y_true` and predictions `y_pred`; "
                                      f"`layer_w` maps {L} layer names to weights; `feats` / `imp` / `err` are feature "
                                      f"importances with their spread; `y_act` / `y_hat` regression results.")


@template(M, "medium")
def lr_schedule_recipes(rng):
    s = seeded(rng)
    total = rng.choice([200, 300, 400])
    warm = rng.choice([20, 30, 40])
    peak = rng.choice(["3e-4", "1e-3", "5e-4"])
    k = rng.choice([50, 100])
    tau = rng.choice([60, 80, 120])
    betas = list(rng.choice([(0.6, 0.9, 0.99), (0.5, 0.8, 0.95), (0.7, 0.9, 0.98)]))
    setup = (f"steps = np.arange({total})\ng = np.random.default_rng({s})\n"
             f"raw = np.round(np.exp(-steps / {tau}) + 0.1 * g.normal(size={total}), 4)   # noisy per-step loss\n"
             f"lrs = np.logspace(-6, 0, 60)\nx = np.log10(lrs)\nrt_loss = np.round(2.3 - 1.8 / (1 + np.exp(-2.5 * (x + 3.5))) "
             f"+ 0.5 * np.exp(3 * (x + 1)) + 0.02 * g.normal(size=60), 4)   # an LR range test")
    wc = (f"np.where(steps < {warm}, steps / {warm}, 0.5 * (1 + np.cos(np.pi * (steps - {warm}) / ({total} - {warm})))) "
          f"* {peak}")
    items = [
        ("axvline", f"Warmup then linear decay: `lr = np.where(steps < {warm}, steps / {warm}, ({total} - steps) / ({total} "
                    f"- {warm})) * {peak}`. Plot it against `steps`, mark the end of the warmup with a dotted gray vertical "
                    f"line at step {warm}, and label the axes 'step' / 'lr'.",
         f"lr = np.where(steps < {warm}, steps / {warm}, ({total} - steps) / ({total} - {warm})) * {peak}\n"
         f"fig, ax = plt.subplots()\nax.plot(steps, lr)\nax.axvline({warm}, ls=':', c='gray')\nax.set(xlabel='step', ylabel='lr')",
         {"post": "fc.lines(fc.ax(), 'ls', 'color', nd=9), fc.labels()", "require": ["where", "axvline"]}),
        ("drawstyle", f"Step decay: `lr = 0.1 * 0.1 ** (steps // {k})`. Draw it as a staircase against `steps` "
                      f"(`drawstyle='steps-post'`) on a log y-axis, with the axis labels 'step' / 'lr'.",
         f"lr = 0.1 * 0.1 ** (steps // {k})\nfig, ax = plt.subplots()\nax.plot(steps, lr, drawstyle='steps-post')\n"
         f"ax.set_yscale('log')\nax.set(xlabel='step', ylabel='lr')",
         {"post": "fc.lines(fc.ax(), 'drawstyle', nd=9), fc.scales(), fc.labels()", "require": ["drawstyle", "set_yscale"]}),
        ("legend", f"Compare three schedules (peak {peak}) against `steps` on one Axes, with a legend: `np.full({total}, "
                   f"{peak})` labelled 'constant', `0.5 * (1 + np.cos(np.pi * steps / {total})) * {peak}` labelled 'cosine', "
                   f"and the guide's warmup ({warm} steps) + cosine labelled 'warmup + cosine'.",
         f"fig, ax = plt.subplots()\nax.plot(steps, np.full({total}, {peak}), label='constant')\n"
         f"ax.plot(steps, 0.5 * (1 + np.cos(np.pi * steps / {total})) * {peak}, label='cosine')\n"
         f"ax.plot(steps, {wc}, label='warmup + cosine')\nax.legend()",
         {"post": "fc.lines(fc.ax(), 'label', nd=9), fc.legend()", "require": ["cos", "legend"]}),
        ("interp", f"One-cycle: the lr climbs linearly from {peak} / 25 to {peak} at step {total // 3}, then falls back to "
                   f"{peak} / 25 at the last step — `np.interp(steps, [0, {total // 3}, {total - 1}], [lo, hi, lo])`. Plot it "
                   f"against `steps` and mark the peak with a red dot (`ax.scatter`).",
         f"lo, hi = {peak} / 25, {peak}\nlr = np.interp(steps, [0, {total // 3}, {total - 1}], [lo, hi, lo])\n"
         f"fig, ax = plt.subplots()\nax.plot(steps, lr)\nax.scatter({total // 3}, hi, c='red')",
         {"post": "fc.lines(fc.ax(), nd=9), fc.scatter(fc.ax(), 0, 9, 'color')", "require": ["interp", "scatter"]}),
        ("set_xscale", "LR range test: plot `rt_loss` against `lrs` on a log x-axis with the labels 'learning rate' / "
                       "'loss', and mark the learning rate where the loss falls fastest (the smallest "
                       "`np.gradient(rt_loss)`) with a dashed red vertical line.",
         "fig, ax = plt.subplots()\nax.plot(lrs, rt_loss)\nax.set_xscale('log')\nax.set(xlabel='learning rate', ylabel='loss')\n"
         "ax.axvline(lrs[np.gradient(rt_loss).argmin()], ls='--', c='red')",
         {"post": "fc.lines(fc.ax(), 'ls', 'color', nd=9), fc.scales(), fc.labels()", "require": ["set_xscale", "gradient"]}),
        ("plot", f"The smoothing slider, three ways: plot `raw` faintly (`alpha=0.25, color='gray'`), then its EMA for each "
                 f"weight b in {betas} (`ema[0] = raw[0]`, `ema[i] = b * ema[i - 1] + (1 - b) * raw[i]`) labelled "
                 f"`f'EMA {{b}}'`, and a legend.",
         f"fig, ax = plt.subplots()\nax.plot(raw, alpha=0.25, color='gray')\nfor b in {betas}:\n    ema = np.zeros_like(raw)\n"
         f"    ema[0] = raw[0]\n    for i in range(1, len(raw)):\n        ema[i] = b * ema[i - 1] + (1 - b) * raw[i]\n"
         f"    ax.plot(ema, label=f'EMA {{b}}')\nax.legend()",
         {"post": "fc.lines(fc.ax(), 'alpha', 'label', nd=5), fc.legend()", "require": ["zeros_like", "legend"]}),
        ("legend", "Why EMAs are debiased: with weight 0.98, plot `raw` in 'lightgray', the plain EMA started at 0 (`m = "
                   "0.98 * m + 0.02 * v` over the values v of `raw`) labelled 'EMA from 0', and the debiased EMA (`m / (1 - "
                   "0.98 ** t)` after the t-th value, t = 1, 2, …) labelled 'debiased'; add a legend.",
         "plain, deb, m = [], [], 0.0\nfor t, v in enumerate(raw, start=1):\n    m = 0.98 * m + 0.02 * v\n    plain.append(m)\n"
         "    deb.append(m / (1 - 0.98 ** t))\nfig, ax = plt.subplots()\nax.plot(raw, color='lightgray')\n"
         "ax.plot(plain, label='EMA from 0')\nax.plot(deb, label='debiased')\nax.legend()",
         {"post": "fc.lines(fc.ax(), 'label', nd=5), fc.legend(), "
                  "[fc.hexcolor(l.get_color()) for l in fc.ax().get_lines() if l.get_label().startswith('_')]"}),
        ("twinx", f"Loss and learning rate on one figure: plot `raw` against `steps` on `ax` (y label 'loss', x label "
                  f"'step'), and the guide's warmup ({warm} steps) + cosine schedule with peak {peak} on `ax.twinx()` in "
                  f"'C1' (y label 'lr').",
         f"lr = {wc}\nfig, ax = plt.subplots()\nax.plot(steps, raw)\nax.set(xlabel='step', ylabel='loss')\nax2 = ax.twinx()\n"
         f"ax2.plot(steps, lr, color='C1')\nax2.set_ylabel('lr')",
         {"post": "[(fc.lines(a, nd=9), fc.labels(a)['ylabel']) for a in fc.axes()], fc.labels(fc.ax(0))['xlabel'], "
                  "fc.lines(fc.ax(1), 'color', data=False)", "require": ["twinx", "cos"]}),
    ]
    return catalog(rng, items, setup, f"`steps` counts {total} training steps and `raw` is a noisy per-step loss; `lrs` / "
                                      f"`rt_loss` are an LR range test (60 learning rates from 1e-6 to 1). Recipe 8 and "
                                      f"recipe 1's smoothing, to the spec.")


# ================================================================= hard

@template(M, "hard")
def recipe_functions(rng):
    _warm_sklearn()
    code, k, names = _cm(rng)
    cd, desc, n = curves(rng)
    s = seeded(rng)
    setup = (code + "\n" + cd + f"\ng3 = np.random.default_rng({s})\nimgs = np.round(g3.random((6, 5, 5)), 3)\n"
             f"t6, p6 = g3.integers(0, 3, 6), g3.integers(0, 3, 6)")
    items = [
        ("imshow", "Write `plot_confusion(cm, names, ax=None, normalize=False, cmap='Blues')`: on `ax` (default "
                   "`plt.gca()`) show the matrix (row-normalized first when `normalize`), class names on both axes, "
                   "x label 'predicted', y label 'true', and each cell's value centered — `f'{v:.2f}'` when normalized, "
                   "else `str(v)` — in white above half the maximum, else black. Return `ax`.",
         "def plot_confusion(cm, names, ax=None, normalize=False, cmap='Blues'):\n    ax = ax or plt.gca()\n"
         "    m = cm / cm.sum(axis=1, keepdims=True) if normalize else cm\n    ax.imshow(m, cmap=cmap)\n"
         "    ax.set_xticks(range(len(names)), names)\n    ax.set_yticks(range(len(names)), names)\n"
         "    ax.set(xlabel='predicted', ylabel='true')\n    for (i, j), v in np.ndenumerate(m):\n"
         "        ax.text(j, i, f'{v:.2f}' if normalize else str(v), ha='center', va='center', "
         "color='white' if v > m.max() / 2 else 'black')\n    return ax",
         {"post": PRE + "_f, (_a, _b) = plt.subplots(1, 2)\nplot_confusion(cm, names, ax=_a)\n_r = plot_confusion(cm, names, ax=_b, normalize=True, cmap='Greens')\n"
                        "[(fc.images(a, 4, data=True), _ticks(a), fc.labels(a), _cells(a), _cellcolors(a)) for a in (_a, _b)], _r is _b"}),
        ("argmin", "Write `plot_history(train, val, ax=None, log=True)`: plots both curves against epochs 1 … n "
                   "(labels 'train', 'val'), log y when `log`, a dashed gray vertical line at the best (lowest val) "
                   "epoch, a legend, and returns the best epoch number.",
         "def plot_history(train, val, ax=None, log=True):\n    ax = ax or plt.gca()\n    ep = np.arange(1, len(train) + 1)\n"
         "    ax.plot(ep, train, label='train')\n    ax.plot(ep, val, label='val')\n    if log:\n        ax.set_yscale('log')\n"
         "    best = int(ep[np.argmin(val)])\n    ax.axvline(best, ls='--', color='gray')\n    ax.legend()\n    return best",
         {"post": "_f, (_a, _b) = plt.subplots(1, 2)\n_r1 = plot_history(train_loss, val_loss, ax=_a)\nplt.sca(_b)\n"
                  "_r2 = plot_history(val_loss[::-1], train_loss, log=False)\n"
                  "[(fc.lines(a, 'label', 'ls', nd=4), fc.scales(a), fc.legend(a)) for a in (_a, _b)], _r1, _r2"}),
        ("predict_proba", "Write `plot_decision_boundary(clf, X, y, ax=None, n=40)`: a grid of n × n points spanning X's "
                          "range ± 0.5, `contourf` of `clf.predict_proba(...)[:, 1]` (`levels=10, cmap='RdBu_r', "
                          "alpha=0.5`), the 0.5 contour in black, the points colored by y (`cmap='RdBu_r', "
                          "edgecolors='k'`). Return `ax`.",
         "def plot_decision_boundary(clf, X, y, ax=None, n=40):\n    ax = ax or plt.gca()\n"
         "    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, n), "
         "np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, n))\n"
         "    p = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)\n"
         "    ax.contourf(xx, yy, p, levels=10, cmap='RdBu_r', alpha=0.5)\n    ax.contour(xx, yy, p, levels=[0.5], colors='k')\n"
         "    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu_r', edgecolors='k')\n    return ax",
         {"post": "from sklearn.datasets import make_moons\nfrom sklearn.neighbors import KNeighborsClassifier\n"
                  "_X, _y = make_moons(80, noise=0.2, random_state=1)\n_c = KNeighborsClassifier(7).fit(_X, _y)\n"
                  "_f, _a = plt.subplots()\n_r = plot_decision_boundary(_c, _X, _y, ax=_a, n=25)\n"
                  "[(type(c).__name__, fc.r(getattr(c, 'levels', []), 3)) for c in _a.collections], fc.scatter(_a, 0, 6, 'array'), "
                  "fc.r(_a.collections[0].allsegs[5][0][:3], 4) if _a.collections[0].allsegs[5] else None, _r is _a",
          "require": ["predict_proba", "contourf"]}),
        ("set_title", "Write `plot_image_grid(images, y_true, y_pred, ncols=4)` that makes its own figure with "
                      "`ceil(len / ncols)` rows (`squeeze=False`), shows each image in gray titled `f'{p} ({t})'` "
                      "green/red by correctness, hides **every** Axes (also unused ones), and returns the figure.",
         "def plot_image_grid(images, y_true, y_pred, ncols=4):\n    nrows = -(-len(images) // ncols)\n"
         "    fig, axes = plt.subplots(nrows, ncols, squeeze=False)\n    for ax in axes.flat:\n        ax.axis('off')\n"
         "    for ax, im, t, p in zip(axes.flat, images, y_true, y_pred):\n        ax.imshow(im, cmap='gray')\n"
         "        ax.set_title(f'{p} ({t})', color='green' if p == t else 'red')\n    return fig",
         {"post": "_g = plot_image_grid(imgs, t6, p6, ncols=4)\n[(a.get_title(), fc.hexcolor(a.title.get_color()) if a.get_title() else None, "
                  "len(a.get_images()), a.axison) for a in _g.axes], fc.grid_shape(_g)"}),
        ("barh", "Write `plot_importances(names, imp, ax=None, top=None, color='C2')`: sort by importance, keep the "
                 "`top` largest when given, draw them with `barh` (largest on top), x label 'importance'; return the "
                 "names in plotted order (bottom to top).",
         "def plot_importances(names, imp, ax=None, top=None, color='C2'):\n    ax = ax or plt.gca()\n"
         "    order = np.argsort(imp)\n    if top:\n        order = order[-top:]\n    names = np.asarray(names)[order]\n"
         "    ax.barh(names, np.asarray(imp)[order], color=color)\n    ax.set_xlabel('importance')\n    return list(names)",
         {"post": "_f, (_a, _b) = plt.subplots(1, 2)\n_n = [f'f{i}' for i in range(7)]\n_v = [0.3, 0.1, 0.7, 0.2, 0.9, 0.05, 0.4]\n"
                  "_r1 = plot_importances(_n, _v, ax=_a)\n_r2 = plot_importances(_n, _v, ax=_b, top=3, color='C1')\n"
                  "_r1, _r2, [(fc.bars(a, 4, 'color'), fc.labels(a)['xlabel']) for a in (_a, _b)]"}),
        ("masked_where", "Write `plot_corr(X, labels, ax=None)` for a data matrix X (samples × features): compute "
                         "`np.corrcoef(X.T)`, mask the upper triangle (above the diagonal), show it with `cmap='RdBu_r', "
                         "vmin=-1, vmax=1`, tick labels `labels` on both axes, and return the correlation matrix.",
         "def plot_corr(X, labels, ax=None):\n    ax = ax or plt.gca()\n    C = np.corrcoef(X.T)\n"
         "    ax.imshow(np.ma.masked_where(np.triu(np.ones_like(C, dtype=bool), k=1), C), cmap='RdBu_r', vmin=-1, vmax=1)\n"
         "    ax.set_xticks(range(len(labels)), labels)\n    ax.set_yticks(range(len(labels)), labels)\n    return C",
         {"post": PRE + "_f, _a = plt.subplots()\n_X = np.column_stack([train_loss, val_loss, train_loss ** 2])\n"
                        "_C = plot_corr(_X, ['a', 'b', 'c'], ax=_a)\nfc.r(_C, 6), [_img(i) for i in _a.get_images()], fc.images(_a), _ticks(_a)",
          "require": ["masked_where", "corrcoef"]}),
    ]
    return catalog(rng, items, setup, f"`cm` / `names` are a confusion matrix, {desc}; `imgs`, `t6`, `p6` six images "
                                      f"with true / predicted classes. Each recipe becomes a reusable function, checked "
                                      f"by calling it.")


@template(M, "hard")
def recipe_bugs(rng):
    code, k, names = _cm(rng)
    cd, desc, n = curves(rng)
    s = seeded(rng)
    setup = (code + "\n" + cd + f"\ng = np.random.default_rng({s})\nacc = np.round(g.uniform(0.7, 0.95, (3, 4)), 3)   # rows: wd, cols: lr\n"
             f"feats = np.array(['a', 'b', 'c', 'd', 'e'])\nimp = np.round(g.random(5), 3)\n"
             f"raw = np.round(np.abs(g.normal(size=60)) * np.exp(-np.arange(60) / 20), 4)")
    items = [
        ("text", f"The counts land in the wrong cells — mirrored across the diagonal:\n```python\nfig, ax = plt.subplots()\n"
                 f"ax.imshow(cm)\nfor i in range({k}):\n    for j in range({k}):\n        ax.text(i, j, cm[i, j], ha='center', "
                 f"va='center')\n```\n`ax.text` takes (x, y) = (column, row). Fix it.",
         f"fig, ax = plt.subplots()\nax.imshow(cm)\nfor i in range({k}):\n    for j in range({k}):\n"
         f"        ax.text(j, i, cm[i, j], ha='center', va='center')", {"post": _post("_cells()")}),
        ("set_xticks", "The class names don't match the cells (Matplotlib even warns about a FixedFormatter): the "
                       "labels were set without fixing the tick positions.\n```python\nfig, ax = plt.subplots()\n"
                       "ax.imshow(cm)\nax.set_xticklabels(names); ax.set_yticklabels(names)\n```\nPut one tick per cell "
                       "with its name.",
         f"fig, ax = plt.subplots()\nax.imshow(cm)\nax.set_xticks(range({k}), names)\nax.set_yticks(range({k}), names)",
         {"post": _post("_ticks()"), "require": ["set_xticks", "set_yticks"]}),
        ("clip", "On a log axis the zero losses vanish and Matplotlib warns:\n```python\nfig, ax = plt.subplots()\n"
                 "ax.plot(raw - raw.min())\nax.set_yscale('log')\n```\nKeep the log scale but plot the shifted loss "
                 "clipped from below at 1e-4 (`np.clip`).",
         "fig, ax = plt.subplots()\nax.plot(np.clip(raw - raw.min(), 1e-4, None))\nax.set_yscale('log')",
         {"post": "fc.lines(fc.ax(), nd=6), fc.scales()", "require": ["clip"]}),
        ("imshow", "The sweep heatmap is upside down: the smallest weight decay (row 0) should be at the **bottom**, like "
                   "a plot's y-axis.\n```python\nfig, ax = plt.subplots()\nax.imshow(acc, cmap='viridis')\n"
                   "ax.set_yticks(range(3), ['1e-05', '1e-04', '1e-03'])\n```\nFix it without flipping the data.",
         "fig, ax = plt.subplots()\nax.imshow(acc, cmap='viridis', origin='lower')\nax.set_yticks(range(3), ['1e-05', '1e-04', '1e-03'])",
         {"post": _post("fc.images(fc.ax(), data=True)", "_ticks()[1]")}),
        ("barh", "The importance chart is unsorted and the biggest bar ends up anywhere:\n```python\nfig, ax = "
                 "plt.subplots()\nax.barh(feats, imp)\n```\nSort ascending (`np.argsort`) so the most important "
                 "feature is at the top.",
         "fig, ax = plt.subplots()\norder = np.argsort(imp)\nax.barh(feats[order], imp[order])",
         {"post": "fc.bars(fc.ax(), 4), fc.tick_labels(fc.ax(), 'y')", "require": ["argsort", "barh"]}),
        ("triu", "The masked correlation heatmap also hides the diagonal (all ones), which should stay visible:\n"
                 "```python\nC = np.corrcoef(np.vstack([train_loss, val_loss, train_loss ** 2]))\n"
                 "ax.imshow(np.ma.masked_where(np.triu(np.ones_like(C, dtype=bool)), C), vmin=-1, vmax=1)\n```\nMask "
                 "only **above** the diagonal (on `fig, ax = plt.subplots()`).",
         "C = np.corrcoef(np.vstack([train_loss, val_loss, train_loss ** 2]))\nfig, ax = plt.subplots()\n"
         "ax.imshow(np.ma.masked_where(np.triu(np.ones_like(C, dtype=bool), k=1), C), vmin=-1, vmax=1)",
         {"post": _post("[_img(i) for i in fc.ax().get_images()]"), "require": ["triu", "masked_where"]}),
        ("zeros_like", "This EMA lags wrongly — it mixes in the **raw** previous value instead of the previous EMA:\n"
                       "```python\nema = np.zeros_like(raw); ema[0] = raw[0]\nfor i in range(1, len(raw)):\n    ema[i] = 0.9 "
                       "* raw[i - 1] + 0.1 * raw[i]\n```\nFix the recursion, plot `raw` (alpha 0.3) and `ema` labelled "
                       "'EMA 0.9' with a legend.",
         "ema = np.zeros_like(raw)\nema[0] = raw[0]\nfor i in range(1, len(raw)):\n    ema[i] = 0.9 * ema[i - 1] + 0.1 * raw[i]\n"
         "fig, ax = plt.subplots()\nax.plot(raw, alpha=0.3)\nax.plot(ema, label='EMA 0.9')\nax.legend()",
         {"post": "fc.lines(fc.ax(), 'alpha', 'label', nd=5), fc.legend()"}),
        ("legend_elements", "The legend comes out empty (*No artists with labels found*): one scatter colored by class has "
                            "no labels.\n```python\nfig, ax = plt.subplots()\nsc = ax.scatter(train_loss, val_loss, "
                            "c=epochs % 3, cmap='tab10')\nax.legend()\n```\nBuild the legend from the scatter's own "
                            "colors with `legend_elements`, titled 'class'.",
         "fig, ax = plt.subplots()\nsc = ax.scatter(train_loss, val_loss, c=epochs % 3, cmap='tab10')\nax.legend(*sc.legend_elements(), title='class')",
         {"post": "fc.legend(), fc.legend_title(), fc.scatter(fc.ax(), 0, 6, 'array')", "require": ["legend_elements"]}),
        ("set", "On this ROC plot the chance diagonal isn't at 45° because the Axes isn't square in data units:\n"
                "```python\nfig, ax = plt.subplots(figsize=(6, 3))\nfpr = np.linspace(0, 1, 50)\nax.plot(fpr, fpr ** 0.3)\n"
                "ax.plot([0, 1], [0, 1], 'k--')\n```\nKeep the figure size; make the aspect equal with `ax.set(aspect='equal')`.",
         "fig, ax = plt.subplots(figsize=(6, 3))\nfpr = np.linspace(0, 1, 50)\nax.plot(fpr, fpr ** 0.3)\nax.plot([0, 1], [0, 1], 'k--')\n"
         "ax.set(aspect='equal')", {"post": "fc.lines(fc.ax(), 'ls'), fc.ax().get_aspect(), fc.size()"}),
    ]
    return catalog(rng, items, setup, f"`cm` / `names` are a {k}-class confusion matrix; {desc}; `acc` a 3 × 4 sweep, "
                                      f"`feats` / `imp` importances and `raw` a noisy loss. Fix the plotting bugs.")


@template(M, "hard")
def real_model_figures(rng):
    _warm_sklearn()
    s = seeded(rng) % 1000
    n = rng.choice([150, 200, 240])
    nf = rng.randint(5, 7)
    items = [
        ("confusion_matrix", "End to end: `X, y = make_classification(n, n_features=nf, n_informative=3, n_classes=3, "
                             "random_state=seed)` (given), fit `LogisticRegression(max_iter=500)` on it, compute "
                             "`confusion_matrix(y, clf.predict(X))`, and draw recipe 2: `imshow` (`cmap='Blues'`), class "
                             "names 'A', 'B', 'C' on both axes, the counts centered in each cell.",
         "from sklearn.linear_model import LogisticRegression\nfrom sklearn.metrics import confusion_matrix\n"
         "clf = LogisticRegression(max_iter=500).fit(X, y)\ncm = confusion_matrix(y, clf.predict(X))\nfig, ax = plt.subplots()\n"
         "ax.imshow(cm, cmap='Blues')\nax.set_xticks(range(3), ['A', 'B', 'C'])\nax.set_yticks(range(3), ['A', 'B', 'C'])\n"
         "for (i, j), v in np.ndenumerate(cm):\n    ax.text(j, i, v, ha='center', va='center')",
         {"post": _post("fc.images(fc.ax(), data=True)", "_ticks()", "_cells()"), "require": ["confusion_matrix", "fit"]}),
        ("feature_importances_", "Fit `RandomForestClassifier(n_estimators=20, random_state=0)` on `X, y` and draw its "
                                 "`feature_importances_` sorted (recipe 10) with `barh`, features named `f'f{i}'`, color "
                                 "'C2', x label 'importance'.",
         "from sklearn.ensemble import RandomForestClassifier\nrf = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)\n"
         "imp = rf.feature_importances_\norder = np.argsort(imp)\nfeats = np.array([f'f{i}' for i in range(len(imp))])\n"
         "fig, ax = plt.subplots()\nax.barh(feats[order], imp[order], color='C2')\nax.set_xlabel('importance')",
         {"post": "fc.bars(fc.ax(), 4, 'color'), fc.tick_labels(fc.ax(), 'y'), fc.labels()['xlabel']",
          "require": ["RandomForestClassifier", "feature_importances_", "barh"]}),
        ("roc_curve", "One-vs-rest ROC for class 0: fit `LogisticRegression(max_iter=500)`, score `p = "
                      "clf.predict_proba(X)[:, 0]`, plot `roc_curve(y == 0, p)` labelled `f'class 0 (AUC="
                      "{roc_auc_score(y == 0, p):.2f})'`, the dashed black diagonal, equal aspect, legend lower right.",
         "from sklearn.linear_model import LogisticRegression\nfrom sklearn.metrics import roc_curve, roc_auc_score\n"
         "clf = LogisticRegression(max_iter=500).fit(X, y)\np = clf.predict_proba(X)[:, 0]\nfpr, tpr, _ = roc_curve(y == 0, p)\n"
         "fig, ax = plt.subplots()\nax.plot(fpr, tpr, label=f'class 0 (AUC={roc_auc_score(y == 0, p):.2f})')\n"
         "ax.plot([0, 1], [0, 1], 'k--')\nax.set_aspect('equal')\nax.legend(loc='lower right')",
         {"post": "fc.lines(fc.ax(), 'label', 'ls', nd=4), fc.ax().get_aspect(), fc.legend()", "require": ["roc_curve", "predict_proba"]}),
        ("PCA", "Embedding recipe with a real projection: `PCA(2).fit_transform(X)`, scatter colored by `y` "
                "(`cmap='tab10', s=8`), legend from `legend_elements()` titled 'class', no ticks.",
         "from sklearn.decomposition import PCA\nZ = PCA(2).fit_transform(X)\nfig, ax = plt.subplots()\n"
         "sc = ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap='tab10', s=8)\nax.legend(*sc.legend_elements(), title='class')\n"
         "ax.set_xticks([])\nax.set_yticks([])",
         {"post": _post("fc.scatter(fc.ax(), 0, 4, 'array')", "fc.legend()", "fc.legend_title()", "_ticks()"), "tol": 1e-3,
          "require": ["PCA", "legend_elements"]}),
        ("LinearRegression", "Regression diagnostics for a real model: predict feature 0 of `X` from the other features "
                             "with `LinearRegression`, then two panels (`plt.subplots(1, 2)`): predicted vs actual with the "
                             "dashed black identity line (min to max of the targets), and residuals vs predicted with a "
                             "black zero line.",
         "from sklearn.linear_model import LinearRegression\nt = X[:, 0]\nreg = LinearRegression().fit(X[:, 1:], t)\n"
         "pr = reg.predict(X[:, 1:])\nfig, (a1, a2) = plt.subplots(1, 2)\na1.scatter(t, pr)\nlims = [t.min(), t.max()]\n"
         "a1.plot(lims, lims, 'k--')\na2.scatter(pr, pr - t)\na2.axhline(0, color='k')",
         {"post": "[(fc.scatter(a, 0, 4), fc.lines(a, 'color', 'ls', nd=4)) for a in fc.axes()]", "tol": 1e-3,
          "require": ["LinearRegression", "predict"]}),
        ("precision_recall_curve", "Recipe 13 with a real model: fit `LogisticRegression(max_iter=500)` on the binary "
                                   "task `y == 0`, get `precision, recall, thr = precision_recall_curve(y == 0, p)` with "
                                   "`p` its class-1 (True) probability, plot precision and recall against `thr` (drop the "
                                   "last precision/recall entry, which has no threshold), labels 'precision', 'recall', "
                                   "and a legend.",
         "from sklearn.linear_model import LogisticRegression\nfrom sklearn.metrics import precision_recall_curve\n"
         "clf = LogisticRegression(max_iter=500).fit(X, y == 0)\np = clf.predict_proba(X)[:, 1]\n"
         "precision, recall, thr = precision_recall_curve(y == 0, p)\nfig, ax = plt.subplots()\n"
         "ax.plot(thr, precision[:-1], label='precision')\nax.plot(thr, recall[:-1], label='recall')\nax.legend()",
         {"post": "fc.lines(fc.ax(), 'label', nd=4), fc.legend()", "tol": 1e-4, "require": ["precision_recall_curve"]}),
        ("savefig", "Fit `LogisticRegression(max_iter=500)`, draw its confusion matrix (`confusion_matrix`, `imshow` "
                    "with `cmap='Blues'`) on a 3 × 3 inch figure, save it as `cm.png` at 40 dpi, close all figures, and "
                    "return `(plt.imread('cm.png').shape, len(plt.get_fignums()))`.",
         "from sklearn.linear_model import LogisticRegression\nfrom sklearn.metrics import confusion_matrix\n"
         "clf = LogisticRegression(max_iter=500).fit(X, y)\nfig, ax = plt.subplots(figsize=(3, 3))\n"
         "ax.imshow(confusion_matrix(y, clf.predict(X)), cmap='Blues')\nfig.savefig('cm.png', dpi=40)\nplt.close('all')\n"
         "result = (plt.imread('cm.png').shape, len(plt.get_fignums()))", {"require": ["savefig", "close"]}),
    ]
    return catalog(rng, items, f"from sklearn.datasets import make_classification\nX, y = make_classification({n}, "
                                f"n_features={nf}, n_informative=3, n_redundant=0, n_classes=3, random_state={s})",
                   f"`X`, `y` is a 3-class dataset ({n} samples, {nf} features) from `make_classification`. Fit real "
                   f"scikit-learn models and draw the recipe.")


@template(M, "hard")
def recipe_dashboards(rng):
    code, desc, n = curves(rng, acc=True)
    s = seeded(rng)
    setup = (code + f"\ng = np.random.default_rng({s})\ncm = g.integers(0, 6, (3, 3)) + np.diag(g.integers(20, 40, 3))\n"
             f"names = ['cat', 'dog', 'bird']\nfeats = np.array([f'f{{i}}' for i in range(6)])\nimp = np.round(g.random(6), 3)\n"
             f"steps = np.arange(300)\nlr = np.where(steps < 30, steps / 30, 0.5 * (1 + np.cos(np.pi * (steps - 30) / 270))) * 1e-3")
    items = [
        ("plt.subplots", "A 2 × 2 training report (`plt.subplots(2, 2, figsize=(9, 7), layout='constrained')`): top-left "
                         "both losses on a log y-axis (legend 'train', 'val'); top-right `val_acc`; bottom-left `cm` as "
                         "`imshow` (`cmap='Blues'`) with `names` on both axes; bottom-right the sorted importances "
                         "(`barh`). Titles 'loss', 'accuracy', 'confusion', 'importance'.",
         "fig, axes = plt.subplots(2, 2, figsize=(9, 7), layout='constrained')\n(a, b), (c, d) = axes\n"
         "a.plot(epochs, train_loss, label='train')\na.plot(epochs, val_loss, label='val')\na.set_yscale('log')\na.legend()\n"
         "b.plot(epochs, val_acc)\nc.imshow(cm, cmap='Blues')\nc.set_xticks(range(3), names)\nc.set_yticks(range(3), names)\n"
         "o = np.argsort(imp)\nd.barh(feats[o], imp[o])\nfor ax, t in zip(axes.flat, ['loss', 'accuracy', 'confusion', 'importance']):\n"
         "    ax.set_title(t)",
         {"post": _post("[fc.labels(a)['title'] for a in fc.axes()]", "fc.lines(fc.ax(0), 'label')", "fc.scales(fc.ax(0))",
                        "fc.lines(fc.ax(1))", "fc.images(fc.ax(2), data=True)", "_ticks(fc.ax(2))", "fc.bars(fc.ax(3), 4)",
                        "fc.tick_labels(fc.ax(3), 'y')", "fc.grid_shape()")}),
        ("plt.subplots", "A 1 × 3 run summary (`plt.subplots(1, 3, figsize=(12, 3))`): the LR schedule `lr` vs `steps` with "
                         "scientific y tick labels (`ticklabel_format(axis='y', style='sci', scilimits=(0, 0))`), the "
                         "losses with the best epoch marked by a dashed gray `axvline`, and `val_acc` with a dotted "
                         "horizontal line at its maximum.",
         "fig, (a, b, c) = plt.subplots(1, 3, figsize=(12, 3))\na.plot(steps, lr)\na.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))\n"
         "b.plot(epochs, train_loss)\nb.plot(epochs, val_loss)\nb.axvline(epochs[val_loss.argmin()], ls='--', color='gray')\n"
         "c.plot(epochs, val_acc)\nc.axhline(val_acc.max(), ls=':')",
         {"post": "f = fc.ax(0).yaxis.get_major_formatter()\n[fc.lines(a, 'ls', nd=7) for a in fc.axes()], f._scientific, list(f._powerlimits)"}),
        ("text", "Recipe 2 next to per-class metrics (`plt.subplots(1, 2, figsize=(8, 3.5))`): left `cm` (`cmap='Blues'`) "
                 "with the counts centered in the cells and `names` on both axes; right a grouped bar chart of "
                 "precision (diagonal / column sums, offset -0.2, width 0.4, label 'precision') and recall (diagonal / "
                 "row sums, offset +0.2, label 'recall') per class with `names` as x tick labels and a legend.",
         "fig, (a, b) = plt.subplots(1, 2, figsize=(8, 3.5))\na.imshow(cm, cmap='Blues')\na.set_xticks(range(3), names)\n"
         "a.set_yticks(range(3), names)\nfor (i, j), v in np.ndenumerate(cm):\n    a.text(j, i, v, ha='center', va='center')\n"
         "prec, rec = np.diag(cm) / cm.sum(axis=0), np.diag(cm) / cm.sum(axis=1)\nx = np.arange(3)\n"
         "b.bar(x - 0.2, prec, 0.4, label='precision')\nb.bar(x + 0.2, rec, 0.4, label='recall')\nb.set_xticks(x, names)\nb.legend()",
         {"post": _post("_cells(fc.ax(0))", "_ticks(fc.ax(0))", "fc.bars(fc.ax(1), 4)", "fc.tick_labels(fc.ax(1), 'x')", "fc.legend(fc.ax(1))")}),
        ("suptitle", "Two panels sharing y (`plt.subplots(1, 2, sharey=True, figsize=(8, 3))`): the raw losses left and "
                     "their 3-epoch moving averages (`np.convolve(..., np.ones(3) / 3, mode='valid')`, plotted against "
                     "`epochs[1:-1]`) right, both with 'train' / 'val' legends, and a figure title 'Loss curves' "
                     "(`fig.suptitle`).",
         "fig, (a, b) = plt.subplots(1, 2, sharey=True, figsize=(8, 3))\nfor name, c in (('train', train_loss), ('val', val_loss)):\n"
         "    a.plot(epochs, c, label=name)\n    b.plot(epochs[1:-1], np.convolve(c, np.ones(3) / 3, mode='valid'), label=name)\n"
         "a.legend()\nb.legend()\nfig.suptitle('Loss curves')",
         {"post": "[(fc.lines(a, 'label', nd=5), fc.legend(a)) for a in fc.axes()], fc.suptitles()['suptitle'], "
                  "fc.ax(0).get_shared_y_axes().joined(fc.ax(0), fc.ax(1))", "require": ["suptitle", "convolve"]}),
    ]
    return catalog(rng, items, setup, f"{desc}; `cm` / `names` a confusion matrix, `feats` / `imp` importances and `lr` a "
                                      f"schedule over `steps`. Build each dashboard to the spec.")


@template(M, "hard")
def recipe_drills(rng):
    _warm_sklearn()
    s = seeded(rng)
    k = rng.randint(3, 4)
    names = rng.sample(["logreg", "knn", "tree", "forest", "svm", "mlp"], 3)
    seps = sorted(rng.sample([0.05, 0.15, 0.25, 0.35, 0.5], 3))
    toks = rng.choice([["the", "cat", "sat", "down"], ["I", "like", "big", "models", "!"], ["a", "b", "c", "d", "e"]])
    folds = rng.choice([4, 5, 10])
    g = f"g = np.random.default_rng({s})\n"
    items = [
        ("transpose", "PyTorch hands you images as CHW arrays, normalized per channel: `chw` (3 × 12 × 12) is `(img - mean) "
                      "/ std`. Undo the normalization (broadcast `std` and `mean` with `[:, None, None]`), move the channels "
                      "last with `np.transpose(..., (1, 2, 0))`, clip to 0 … 1, and show the image with `imshow`, axis off.",
         "img = np.clip(np.transpose(chw * std[:, None, None] + mean[:, None, None], (1, 2, 0)), 0, 1)\nfig, ax = plt.subplots()\n"
         "ax.imshow(img)\nax.axis('off')",
         {"setup": g + "mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])\n"
                       "chw = np.round((g.random((3, 12, 12)) - mean[:, None, None]) / std[:, None, None], 4)",
          "post": "fc.images(fc.ax(), 4, data=True), fc.ax().axison", "require": ["transpose", "clip", "imshow"]}),
        ("fill_between", f"Variation across seeds: `runs` holds the validation loss of {k} seeds (one row each, against "
                         f"`epochs`). Plot the mean curve labelled 'mean', a band of ±1 standard deviation around it "
                         f"(`runs.std(axis=0)`; `fill_between` with `alpha=0.3`, labelled '±1 std'), a log y-axis and a "
                         f"legend.",
         "mu, sd = runs.mean(axis=0), runs.std(axis=0)\nfig, ax = plt.subplots()\nax.plot(epochs, mu, label='mean')\n"
         "ax.fill_between(epochs, mu - sd, mu + sd, alpha=0.3, label='±1 std')\nax.set_yscale('log')\nax.legend()",
         {"setup": g + f"epochs = np.arange(1, 31)\nruns = np.round(np.exp(-epochs / g.uniform(6, 10, ({k}, 1))) + 0.1 "
                       f"+ 0.02 * g.random(({k}, 30)), 4)",
          "post": "fc.lines(fc.ax(), 'label'), [(fc.r(c.get_paths()[0].get_extents().bounds, 5), c.get_alpha()) for c in "
                  "fc.ax().collections], fc.legend(), fc.scales()", "require": ["fill_between", "std"]}),
        ("colorbar", f"Attention heads side by side: `att` holds {k} heads, each a T × T matrix over `tokens`. Draw them in "
                     f"one row (`plt.subplots(1, {k}, figsize=({3 * k}, 3))`) on one shared color scale (`cmap='viridis', "
                     f"vmin=0, vmax=att.max()`), titled 'head 0', 'head 1', …, with the tokens as tick labels on both axes, "
                     f"and a single colorbar for the whole row (`fig.colorbar(im, ax=axes)`).",
         f"fig, axes = plt.subplots(1, {k}, figsize=({3 * k}, 3))\nfor i, (ax, a) in enumerate(zip(axes, att)):\n"
         f"    im = ax.imshow(a, cmap='viridis', vmin=0, vmax=att.max())\n    ax.set_title(f'head {{i}}')\n"
         f"    ax.set_xticks(range(len(tokens)), tokens)\n    ax.set_yticks(range(len(tokens)), tokens)\nfig.colorbar(im, ax=axes)",
         {"setup": g + f"tokens = {lit(toks)}\natt = np.tril(g.random(({k}, {len(toks)}, {len(toks)})))\n"
                       f"att = np.round(att / att.sum(-1, keepdims=True), 3)",
          "post": _post("[(fc.labels(a)['title'], fc.images(a, 4), _ticks(a)) for a in fc.axes()]", "len(fc.colorbars())"),
          "require": ["colorbar", "imshow"]}),
        ("roc_curve", "ROC curves of several models: `y` holds binary labels and `scores[name]` each model's scores. For "
                      "every model (in `scores` order) get `fpr, tpr, _ = roc_curve(y, sc)` (from `sklearn.metrics`), its "
                      "AUC as `np.trapezoid(tpr, fpr)`, and plot the curve labelled `f'{name} (AUC={auc:.2f})'`; add the "
                      "chance diagonal `'k--'`, an equal aspect and the legend in the lower right.",
         "from sklearn.metrics import roc_curve\nfig, ax = plt.subplots()\nfor name, sc in scores.items():\n"
         "    fpr, tpr, _ = roc_curve(y, sc)\n    auc = np.trapezoid(tpr, fpr)\n    ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.2f})')\n"
         "ax.plot([0, 1], [0, 1], 'k--')\nax.set_aspect('equal')\nax.legend(loc='lower right')",
         {"setup": g + f"y = g.integers(0, 2, 120)\nscores = {{name: np.round(np.clip(sep * y + g.normal(0.4, 0.25, 120), 0, 1), 3) "
                       f"for name, sep in zip({lit(names)}, {seps})}}",
          "post": "fc.lines(fc.ax(), 'label', 'ls', nd=4), fc.ax().get_aspect(), fc.legend(), fc.ax().get_legend()._loc",
          "require": ["roc_curve", "trapezoid"]}),
        ("digitize", "A reliability diagram: `p` are predicted probabilities and `y` the true labels. Put `p` into 10 equal "
                     "bins over 0 … 1 (`b = np.digitize(p, np.linspace(0, 1, 11)[1:-1])`), and for every non-empty bin, "
                     "in order, plot the mean predicted probability (x) against the fraction of positives (y) as `'o-'` "
                     "labelled 'model'; add the diagonal `'k--'` labelled 'perfect', the axis labels 'predicted' / "
                     "'observed' and a legend.",
         "b = np.digitize(p, np.linspace(0, 1, 11)[1:-1])\nbins = [k for k in range(10) if (b == k).any()]\nfig, ax = plt.subplots()\n"
         "ax.plot([p[b == k].mean() for k in bins], [y[b == k].mean() for k in bins], 'o-', label='model')\n"
         "ax.plot([0, 1], [0, 1], 'k--', label='perfect')\nax.set(xlabel='predicted', ylabel='observed')\nax.legend()",
         {"setup": g + "p = np.round(g.random(200), 3)\ny = (g.random(200) < p ** 1.3).astype(int)   # an overconfident model",
          "post": "fc.lines(fc.ax(), 'label', 'marker', 'ls'), fc.labels(), fc.legend()", "require": ["digitize"]}),
        ("bar_label", "Dead units per layer: `acts` maps layer names to post-ReLU activations. Bar-chart the fraction of "
                      "exact zeros in each layer (`(a == 0).mean()`, layer names on x), write each bar's value with "
                      "`ax.bar_label(bars, fmt='%.2f')`, set the y limits to 0 … 1, and draw a dashed red line at 0.5.",
         "frac = [(a == 0).mean() for a in acts.values()]\nfig, ax = plt.subplots()\nbars = ax.bar(list(acts), frac)\n"
         "ax.bar_label(bars, fmt='%.2f')\nax.set_ylim(0, 1)\nax.axhline(0.5, ls='--', c='red')",
         {"setup": g + f"acts = {{f'layer{{i}}': np.maximum(0, np.round(g.normal(0.5 - 0.4 * i, 1, 400), 3)) for i in range({k})}}",
          "post": "fc.bars(fc.ax(), 4), fc.tick_labels(fc.ax(), 'x'), fc.texts(), fc.limits()['ylim'], fc.lines(fc.ax(), 'ls', 'color')",
          "require": ["bar_label", "bar"]}),
        ("errorbar", f"Cross-validation results: `cv[i]` holds the {folds} fold scores of model `names[i]`. Plot each "
                     f"model's mean score with ±1 standard deviation error bars at x = 0, 1, 2 (`ax.errorbar(x, means, "
                     f"yerr=stds, fmt='o', capsize=4)`, `stds = cv.std(axis=1)`), the model names as x tick labels, and "
                     f"the y label 'accuracy'.",
         "means, stds = cv.mean(axis=1), cv.std(axis=1)\nx = np.arange(len(names))\nfig, ax = plt.subplots()\n"
         "ax.errorbar(x, means, yerr=stds, fmt='o', capsize=4)\nax.set_xticks(x, names)\nax.set_ylabel('accuracy')",
         {"setup": g + f"names = {lit(names)}\ncv = np.round(g.uniform(0.7, 0.9, (3, 1)) + 0.03 * g.normal(size=(3, {folds})), 3)",
          "post": "fc.lines(fc.ax(), 'marker', 'ls')[:1], [fc.r(c.get_segments(), 6) for c in fc.ax().collections], "
                  "fc.tick_labels(fc.ax(), 'x'), fc.labels()['ylabel']", "require": ["errorbar"]}),
    ]
    return catalog(rng, items, "", "Rebuild an ML figure from a blank cell (the guide's drill), to the spec.")


# ================================================================= challenge

@template(M, "challenge")
def roc_from_scores(rng):
    n, s = rng.randint(12, 20), seeded(rng)
    report = rng.choice(["points", "auc", "youden"])

    def gen(seed, n_, nd):
        return (f"g = np.random.default_rng({seed})\nlabels = g.integers(0, 2, {n_})\nlabels[:2] = [0, 1]\n"
                f"scores = np.round(np.clip(0.3 * labels + g.normal(0.35, 0.25, {n_}), 0, 1), {nd})\nreport = {report!r}")
    sol = ("order = np.argsort(-scores, kind='stable')\ns, y = scores[order], labels[order]\n"
           "last = np.r_[s[1:] != s[:-1], True]   # the last sample of each block of equal scores\n"
           "tp, fp = np.cumsum(y)[last], np.cumsum(1 - y)[last]\nP, N = y.sum(), len(y) - y.sum()\n"
           "tpr, fpr = np.r_[0, tp] / P, np.r_[0, fp] / N\n"
           + {"points": "result = np.column_stack([fpr, tpr])",
              "auc": "result = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))",
              "youden": "j = tp * N - fp * P   # = P·N·(TPR − FPR): compare exactly, in integers\n"
                        "result = float(s[last][j.argmax()])   # first max = the highest threshold"}[report])
    want = {"points": "the ROC curve's vertices: (0, 0) and then one (FPR, TPR) point per distinct score, from the "
                      "highest score down (an array of shape (distinct + 1) × 2) — what `roc_curve(..., "
                      "drop_intermediate=False)` returns",
            "auc": "the area under the ROC curve (trapezoids between consecutive vertices; tied scores give a diagonal "
                   "step)",
            "youden": "the threshold (one of the scores) that maximizes Youden's J = TPR − FPR (ties: the highest "
                      "threshold)"}[report]
    return challenge(
        "ROC curve from raw scores",
        "A classifier gave `scores` to samples with binary `labels` (both classes occur). Thresholding at t predicts "
        "positive when `score >= t`; TPR = TP / P, FPR = FP / N. Scores can tie, so walk down the **distinct** scores "
        f"from the highest. Return {want}. (`report` = {report!r}.)",
        gen(s, n, 1), sol, [gen(s + 1, 200000, 3), gen(s + 2, 150000, 2)], "n ≤ 2·10^5 samples",
        speed="O(n log n): sort once, cumulative sums over blocks of tied scores", tol=1e-9)


@template(M, "challenge")
def pr_operating_point(rng):
    n, q, s = rng.randint(14, 22), rng.randint(4, 6), seeded(rng)
    report = rng.choice(["at", "best_f1", "recall_at"])
    pmin = rng.choice([0.6, 0.7, 0.75, 0.8])

    def gen(seed, n_, q_, nd):
        return (f"g = np.random.default_rng({seed})\nlabels = g.integers(0, 2, {n_})\nlabels[:2] = [0, 1]\n"
                f"scores = np.round(np.clip(0.35 * labels + g.normal(0.3, 0.2, {n_}), 0, 1), {nd})\n"
                f"thr = np.round(g.random({q_}), {nd})\np_min = {pmin}\nreport = {report!r}")
    sol = ("s = np.sort(scores)\nys = labels[np.argsort(scores, kind='stable')]\npos_ge = np.r_[np.cumsum(ys[::-1])[::-1], 0]   "
           "# positives with index >= i in sorted order\nP = labels.sum()\n\ndef counts(t):\n"
           "    i = np.searchsorted(s, t, side='left')   # first score >= t\n    tp = pos_ge[i]\n    pp = len(s) - i\n"
           "    return tp, pp\n\n"
           + {"at": ("tp, pp = counts(thr)\nprec = np.where(pp > 0, tp / np.maximum(pp, 1), 1.0)\n"
                     "result = np.column_stack([prec, tp / P])"),
              "best_f1": ("u = np.unique(scores)\ntp, pp = counts(u)\nf1 = 2 * tp / (pp + P)   # = 2TP / (2TP + FP + FN)\n"
                          "i = len(u) - 1 - np.argmax(f1[::-1])   # ties: the highest threshold\nresult = [float(u[i]), float(f1[i])]"),
              "recall_at": ("u = np.unique(scores)\ntp, pp = counts(u)\nok = tp / pp >= p_min\n"
                            "result = float((tp[ok] / P).max()) if ok.any() else 0.0")}[report])
    want = {"at": "for every query threshold `thr[j]`, `(precision, recall)` as a q × 2 array (precision is 1.0 when "
                  "nothing is predicted positive)",
            "best_f1": "`[threshold, F1]` for the threshold among the distinct scores with the best F1 = 2·TP / (2·TP + "
                       "FP + FN) (ties: the highest threshold)",
            "recall_at": "the highest recall reachable with precision ≥ `p_min`, over thresholds at the distinct scores "
                         "(0.0 if none)"}[report]
    return challenge(
        "Choosing an operating point",
        "From recipe 13: a classifier's `scores` and binary `labels` (both classes occur). Thresholding at t predicts "
        "positive when `score >= t`; precision = TP / (TP + FP), recall = TP / P. Compute precision and recall as "
        f"those exact divisions. Return {want}. (`report` = {report!r}.)",
        gen(s, n, q, 2), sol, [gen(s + 1, 200000, 200000, 4), gen(s + 2, 150000, 1000, 3)],
        "n ≤ 2·10^5 samples, q ≤ 2·10^5 thresholds",
        speed="O((n + q) log n): sort the scores, suffix counts of positives, binary search per threshold", tol=1e-9,
        answer_desc=" (a list `[threshold, f1]`)" if report == "best_f1" else "")


@template(M, "challenge")
def layer_histograms(rng):
    L, s = rng.randint(3, 4), seeded(rng)
    report = rng.choice(["counts", "mode", "overlap"])
    B = rng.randint(5, 8)

    def gen(seed, n_, L_, B_):
        return (f"g = np.random.default_rng({seed})\nlayer = g.integers(0, {L_}, {n_})\n"
                f"w = np.round(g.normal(0, 1, {n_}) / (layer + 1), 3)   # one weight per row, deeper layers shrink\n"
                f"L, bins = {L_}, {B_}\nreport = {report!r}")
    sol = ("edges = np.linspace(w.min(), w.max(), bins + 1)   # shared bins over all layers\n"
           "b = np.clip(np.searchsorted(edges, w, side='right') - 1, 0, bins - 1)   # the last bin includes the max\n"
           "counts = np.bincount(layer * bins + b, minlength=L * bins).reshape(L, bins)\n"
           + {"counts": "result = counts",
              "mode": "result = counts.argmax(axis=1)",
              "overlap": "result = np.minimum(counts[:-1], counts[1:]).sum(axis=1)"}[report])
    want = {"counts": "the L × bins matrix of counts (row = layer)",
            "mode": "for each layer, the index of its fullest bin (ties: the lowest index)",
            "overlap": "for each pair of consecutive layers (l, l + 1), the histogram intersection Σ min(count_l, "
                       "count_{l+1}) over the bins (an array of length L − 1)"}[report]
    return challenge(
        "Weight histograms per layer",
        "Recipe 7 compares layers' weight distributions, so all histograms must share the same bins: `bins` equal-width "
        "bins from the smallest to the largest weight over **all** layers, `np.histogram`-style (each bin includes its "
        "left edge; the last one also its right edge). Weight `w[i]` belongs to layer `layer[i]` (0 … L − 1). Return "
        f"{want}. (`report` = {report!r}.)",
        gen(s, rng.randint(30, 50), L, B), sol, [gen(s + 1, 200000, 50, 100), gen(s + 2, 200000, 3, 1000)],
        "n ≤ 2·10^5 weights, L ≤ 50 layers, bins ≤ 1000", speed="O(n log bins): one binary search per weight and a bincount")
