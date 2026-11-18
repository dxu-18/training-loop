"""5.07 Pie, polar & 3D (pie / donut charts, polar Axes and radar charts, mplot3d surfaces, wireframes, scatter,
lines and projected contours)."""

from ... import themes as T
from ._common import catalog, challenge, lit, seeded, template

M = "5.07"
PIE = "fc.patches(fc.ax())"
TXT = "fc.texts(fc.ax())"
LINES = "fc.lines(fc.ax())"
POLAR = "fc.ax().name"
SURF = ("[(c._vec.shape[1], fc.r(c._vec[:3].sum(axis=1), 2), fc.r(c._vec[:3].min(axis=1), 4), "
        "fc.r(c._vec[:3].max(axis=1), 4)) for c in fc.ax().collections if type(c).__name__ == 'Poly3DCollection']")
WIRE = ("[(len(c._segments3d), fc.r(sum(np.asarray(s).sum(axis=0) for s in c._segments3d), 2)) "
        "for c in fc.ax().collections if type(c).__name__ == 'Line3DCollection']")
SC3 = ("[fc.r(np.column_stack([np.ma.getdata(v) for v in c._offsets3d]), 4) for c in fc.ax().collections "
       "if type(c).__name__ == 'Path3DCollection']")
L3 = "[fc.r(l.get_data_3d(), 4) for l in fc.ax().get_lines()]"
CONT3 = ("[(fc.r(c.levels, 4), fc.r(np.concatenate([np.asarray(v) for v, _ in c._3dverts_codes]).min(axis=0), 3), "
         "fc.r(np.concatenate([np.asarray(v) for v, _ in c._3dverts_codes]).max(axis=0), 3)) "
         "for c in fc.ax().collections if hasattr(c, '_3dverts_codes')]")
LAB3 = "[fc.ax().get_xlabel(), fc.ax().get_ylabel(), fc.ax().get_zlabel()]"


def _pix(expr: str, nd: int = 4) -> str:
    """Post code: an array that may be masked as rounded lists (`fc.r` turns rows of 2-D masked arrays into None)."""
    return f"fc.r(np.ma.filled(np.ma.asarray({expr}, float), np.nan), {nd})"


OFF = f"[{_pix('c.get_offsets()')} for c in fc.ax().collections if type(c).__name__ == 'PathCollection']"
CARR = ("[(fc.r(c.get_array(), 4), c.get_cmap().name) for c in fc.ax().collections "
        "if type(c).__name__ == 'PathCollection']")


def _per_axes(expr: str) -> str:
    return f"[{expr} for a in fc.axes()]"


# ---------------------------------------------------------------- data

SPLITS = [(["train", "val", "test"], [70, 15, 15]), (["train", "val", "test", "holdout"], [45, 30, 15, 10]),
          (["train", "dev", "test"], [80, 10, 10]), (["labelled", "unlabelled", "rejected"], [35, 55, 10])]


def _shares(rng, k=(3, 6)):
    """`sizes` (counts) and `labels`: a dataset split or a class distribution."""
    if rng.random() < 0.35:
        labels, sizes = rng.choice(SPLITS)
        what = "dataset split"
    else:
        name, labs = T.labels(rng, k[0])
        n = rng.randint(k[0], min(k[1], len(labs)))
        labels = list(labs[:n])
        sizes = [rng.randint(5, 60) for _ in labels]
        what = f"{name} class distribution"
    return f"sizes = {lit(list(sizes))}\nlabels = {lit(list(labels))}", f"`sizes` and `labels` describe a {what}", \
        list(sizes), list(labels)


def _rose(rng):
    n = rng.choice([60, 100, 150])
    a, b, k = rng.choice([1, 1.5, 2]), rng.choice([0.3, 0.5, 0.8]), rng.choice([3, 4, 5, 6])
    return (f"theta = np.linspace(0, 2 * np.pi, {n})\nr = {a} + {b} * np.sin({k} * theta)",
            f"`theta` holds {n} angles (radians, 0 … 2π) and `r` = {a} + {b}·sin({k}θ) the radius at each")


LOSSES3D = [("W1 ** 2 + W2 ** 2", "a round bowl"), ("W1 ** 2 - W2 ** 2", "a saddle"),
            ("0.5 * W1 ** 2 + 2 * W2 ** 2", "a stretched bowl"), ("np.sin(W1) * np.cos(W2)", "an egg crate"),
            ("np.log1p(W1 ** 2 + 3 * W2 ** 2)", "a log bowl")]


def _landscape(rng, n=(15, 30)):
    nx, ny = rng.randint(*n), rng.randint(*n)
    a = rng.choice([1, 2, 3])
    f, name = rng.choice(LOSSES3D)
    return (f"W1, W2 = np.meshgrid(np.linspace(-{a}, {a}, {nx}), np.linspace(-{a}, {a}, {ny}))\nL = {f}",
            f"`W1`, `W2` are {ny} × {nx} grids of two weights (-{a} … {a}) and `L` is {name} on them", nx, ny)


METRICS = [["acc", "prec", "rec", "f1", "auc"], ["speed", "memory", "accuracy", "robustness", "calibration"],
           ["BLEU", "ROUGE", "chrF", "COMET", "BERTScore", "TER"], ["top1", "top5", "mAP", "IoU"]]


def _models(rng, k=2):
    names = rng.choice(METRICS)
    m = len(names)
    models = rng.sample(["baseline", "ours", "large", "distilled", "v2", "ensemble"], k)
    s = seeded(rng)
    code = (f"metrics = {lit(names)}\nmodels = {lit(models)}\n"
            f"scores = np.round(np.random.default_rng({s}).uniform(0.4, 0.95, ({k}, {m})), 2)")
    return code, (f"`scores[i, j]` is model `models[i]`'s score on metric `metrics[j]` ({k} models × {m} metrics, "
                  f"all in 0 … 1)"), m, models


CLOSE = "angles = np.linspace(0, 2 * np.pi, {m}, endpoint=False)\nth = np.append(angles, angles[0])\n"


# ================================================================= lite

@template(M, "lite")
def pie_options(rng):
    code, desc, sizes, labels = _shares(rng)
    sa = rng.choice([45, 90, 140])
    ex = rng.choice([0.05, 0.1, 0.2])
    w = rng.choice([0.3, 0.4, 0.5])
    rad = rng.choice([0.6, 0.8, 1.2])
    k = len(sizes)
    cols = rng.choice([["C0", "C1", "C2", "C3", "C4", "C5"], ["tab:blue", "tab:orange", "tab:green", "tab:red",
                                                               "tab:purple", "tab:brown"]])[:k]
    title = rng.choice(["Data split", "Classes", "Share"])
    items = [
        (".pie", "Create `fig, ax` and draw `sizes` as a pie chart with `ax.pie`.", "fig, ax = plt.subplots()\nax.pie(sizes)",
         {"post": PIE}),
        ("labels", "Draw `sizes` as a pie with `ax.pie`, labelling each wedge with its entry of `labels`.",
         "fig, ax = plt.subplots()\nax.pie(sizes, labels=labels)", {"post": f"{PIE}, {TXT}", "require": ["pie", "labels"]}),
        ("autopct", "Draw `sizes` as a pie with `ax.pie` and print each wedge's percentage on it with one decimal "
                    "(`autopct='%1.1f%%'`).", "fig, ax = plt.subplots()\nax.pie(sizes, autopct='%1.1f%%')",
         {"post": f"{PIE}, {TXT}", "require": ["pie", "autopct"]}),
        ("startangle", f"Draw `sizes` as a pie with `ax.pie`, starting the first wedge at {sa}° instead of 0° (the "
                       f"positive x-axis).", f"fig, ax = plt.subplots()\nax.pie(sizes, startangle={sa})",
         {"post": PIE, "require": ["pie", "startangle"]}),
        ("counterclock", "Draw `sizes` as a pie with `ax.pie`, laying the wedges out **clockwise** (`counterclock`).",
         "fig, ax = plt.subplots()\nax.pie(sizes, counterclock=False)", {"post": PIE, "require": ["pie", "counterclock"]}),
        ("explode", f"Draw `sizes` as a pie with `ax.pie`, pulling the first wedge out by {ex} of the radius "
                    f"(`explode`, one value per wedge, 0 for the others).",
         f"fig, ax = plt.subplots()\nax.pie(sizes, explode=[{ex}] + [0] * {k - 1})", {"post": PIE, "require": ["pie", "explode"]}),
        ("wedgeprops", f"Draw `sizes` as a **donut** with `ax.pie`: wedges of width {w} (`wedgeprops={{'width': {w}}}`).",
         f"fig, ax = plt.subplots()\nax.pie(sizes, wedgeprops={{'width': {w}}})",
         {"post": f"{PIE}, [p.width for p in fc.ax().patches]", "require": ["pie", "wedgeprops"]}),
        ("colors", f"Draw `sizes` as a pie with `ax.pie` in the colors {cols} (`colors=`).",
         f"fig, ax = plt.subplots()\nax.pie(sizes, colors={cols})",
         {"post": f"{PIE}, [fc.hexcolor(p.get_facecolor()) for p in fc.ax().patches]", "require": ["pie", "colors"]}),
        ("radius", f"Draw `sizes` as a pie with `ax.pie` of radius {rad}.", f"fig, ax = plt.subplots()\nax.pie(sizes, radius={rad})",
         {"post": PIE, "require": ["pie", "radius"]}),
        ("plt.pie", f"With pyplot only, draw `sizes` as a pie labelled with `labels` and title it {title!r}.",
         f"plt.pie(sizes, labels=labels)\nplt.title({title!r})",
         {"post": f"{PIE}, {TXT}, fc.labels(fc.ax())['title']", "require": ["pie", "title"]}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "lite")
def polar_axes(rng):
    code, desc = _rose(rng)
    ticks = rng.choice([[0.5, 1, 1.5], [1, 2], [0.5, 1.5, 2.5]])
    loc = rng.choice(["N", "W", "S"])
    rl = rng.choice([0, 45, 135, 180])
    rmax = rng.choice([2, 3, 4])
    t0, t1 = rng.choice([(0, 180), (0, 90), (-90, 90)])
    al = rng.choice([0.2, 0.3, 0.5])
    items = [
        ("subplot_kw", "Create a polar Axes with `fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})` and "
                       "plot `r` against `theta` on it.",
         "fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})\nax.plot(theta, r)", {"post": f"{POLAR}, {LINES}"}),
        ("add_subplot", "Create `fig = plt.figure()`, add a polar Axes with `fig.add_subplot(projection='polar')` and "
                        "plot `r` against `theta`.", "fig = plt.figure()\nax = fig.add_subplot(projection='polar')\nax.plot(theta, r)",
         {"post": f"{POLAR}, {LINES}", "require": ["add_subplot", "projection"]}),
        ("set_rticks", f"On a polar Axes, plot `r` against `theta` and put the radial ticks at {ticks} "
                       f"(`ax.set_rticks`).",
         f"fig, ax = plt.subplots(subplot_kw={{'projection': 'polar'}})\nax.plot(theta, r)\nax.set_rticks({ticks})",
         {"post": f"{LINES}, fc.r(fc.ax().get_yticks(), 4)"}),
        ("set_theta_zero_location", f"On a polar Axes, plot `r` against `theta` and move angle 0 to the {loc!r} side "
                                    f"(`ax.set_theta_zero_location`).",
         f"fig, ax = plt.subplots(subplot_kw={{'projection': 'polar'}})\nax.plot(theta, r)\nax.set_theta_zero_location({loc!r})",
         {"post": f"{LINES}, fc.r(fc.ax().get_theta_offset(), 6)"}),
        ("set_theta_direction", "On a polar Axes, plot `r` against `theta` with the angles increasing **clockwise** "
                                "(`ax.set_theta_direction(-1)`).",
         "fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})\nax.plot(theta, r)\nax.set_theta_direction(-1)",
         {"post": f"{LINES}, fc.ax().get_theta_direction()"}),
        ("set_rmax", f"On a polar Axes, plot `r` against `theta` and let the radius axis run from 0 to {rmax} "
                     f"(`ax.set_rlim(0, {rmax})` or `ax.set_rmax`).",
         f"fig, ax = plt.subplots(subplot_kw={{'projection': 'polar'}})\nax.plot(theta, r)\nax.set_rlim(0, {rmax})",
         {"post": f"{LINES}, fc.r(fc.ax().get_ylim(), 4)", "require": ["set_rlim|set_rmax|set_ylim"]}),
        ("set_rlabel_position", f"On a polar Axes, plot `r` against `theta` and draw the radius labels along the "
                                f"{rl}° direction (`ax.set_rlabel_position`).",
         f"fig, ax = plt.subplots(subplot_kw={{'projection': 'polar'}})\nax.plot(theta, r)\nax.set_rlabel_position({rl})",
         {"post": f"{LINES}, fc.ax().get_rlabel_position()"}),
        ("set_thetamin", f"Draw a partial polar plot: plot `r` against `theta` and show only the angles {t0}° … {t1}° "
                         f"(`ax.set_thetamin`, `ax.set_thetamax`).",
         f"fig, ax = plt.subplots(subplot_kw={{'projection': 'polar'}})\nax.plot(theta, r)\nax.set_thetamin({t0})\n"
         f"ax.set_thetamax({t1})",
         {"post": f"{LINES}, fc.r([fc.ax().get_thetamin(), fc.ax().get_thetamax()], 4)",
          "require": ["set_thetamin|set_thetalim", "set_thetamax|set_thetalim"]}),
        ("fill", f"On a polar Axes, fill the area enclosed by the curve (`ax.fill(theta, r, alpha={al})`).",
         f"fig, ax = plt.subplots(subplot_kw={{'projection': 'polar'}})\nax.fill(theta, r, alpha={al})",
         {"post": f"{POLAR}, [(p['type'], len(p['xy'])) for p in fc.patches(fc.ax())], [p.get_alpha() for p in fc.ax().patches]"}),
        ("plt.polar", "With pyplot only, draw the curve on a polar plot with `plt.polar(theta, r)`.",
         "plt.polar(theta, r)", {"post": f"{POLAR}, {LINES}"}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "lite")
def three_d_calls(rng):
    code, desc, nx, ny = _landscape(rng)
    rs, cs = rng.choice([(2, 2), (3, 3), (5, 5), (2, 4)])
    cm = rng.choice(["viridis", "plasma", "coolwarm", "magma"])
    el, az = rng.choice([(30, -60), (20, 45), (60, 30), (10, -120)])
    off = rng.choice([-2, -3, -5])
    zl = rng.choice([(-5, 5), (-3, 10), (-10, 10)])
    new = "fig = plt.figure()\nax = fig.add_subplot(projection='3d')\n"
    items = [
        ("add_subplot", "Create a 3-D Axes (`fig = plt.figure()`, `ax = fig.add_subplot(projection='3d')`) and draw "
                        "the surface `L` over `W1`, `W2` with `ax.plot_surface`.", f"{new}ax.plot_surface(W1, W2, L)",
         {"post": SURF, "require": ["projection", "plot_surface"]}),
        ("plot_surface", f"On a 3-D Axes, draw the surface `L` over `W1`, `W2` in the colormap {cm!r}.",
         f"{new}ax.plot_surface(W1, W2, L, cmap={cm!r})",
         {"post": f"{SURF}, [c.get_cmap().name for c in fc.ax().collections]"}),
        ("plot_wireframe", f"On a 3-D Axes, draw `L` over `W1`, `W2` as a wireframe that keeps every {rs}th row and "
                           f"every {cs}th column of the grid (`rstride={rs}, cstride={cs}`).",
         f"{new}ax.plot_wireframe(W1, W2, L, rstride={rs}, cstride={cs})",
         {"post": WIRE, "require": ["plot_wireframe", "rstride", "cstride"]}),
        ("scatter", "On a 3-D Axes, draw the grid points as a 3-D scatter: `ax.scatter(W1.ravel(), W2.ravel(), "
                    "L.ravel())` (x, y and z).", f"{new}ax.scatter(W1.ravel(), W2.ravel(), L.ravel())",
         {"post": SC3, "require": ["scatter"]}),
        ("plot", "On a 3-D Axes, draw the middle row of the surface as a 3-D line: `ax.plot(W1[k], W2[k], L[k])` "
                 "with `k = len(L) // 2`.", f"{new}k = len(L) // 2\nax.plot(W1[k], W2[k], L[k])",
         {"post": L3, "require": ["plot"]}),
        ("view_init", f"On a 3-D Axes, draw the surface `L` and set the camera to elevation {el}° and azimuth {az}° "
                      f"(`ax.view_init`).", f"{new}ax.plot_surface(W1, W2, L)\nax.view_init(elev={el}, azim={az})",
         {"post": f"{SURF}, [fc.ax().elev, fc.ax().azim]"}),
        ("set", "On a 3-D Axes, draw the surface `L` and label the three axes 'w1', 'w2' and 'loss' "
                "(`ax.set(xlabel=..., ylabel=..., zlabel=...)`).",
         f"{new}ax.plot_surface(W1, W2, L)\nax.set(xlabel='w1', ylabel='w2', zlabel='loss')",
         {"post": f"{SURF}, {LAB3}", "require": ["zlabel|set_zlabel"]}),
        ("contour", f"On a 3-D Axes, draw the contour lines of `L` flattened onto the plane z = {off} "
                    f"(`ax.contour(W1, W2, L, zdir='z', offset={off})`).",
         f"{new}ax.contour(W1, W2, L, zdir='z', offset={off})", {"post": CONT3, "require": ["contour", "zdir", "offset"]}),
        ("subplot_kw", "Create the 3-D Axes in one call with `fig, ax = plt.subplots(subplot_kw={'projection': "
                       "'3d'})` and draw the surface `L` over `W1`, `W2`.",
         "fig, ax = plt.subplots(subplot_kw={'projection': '3d'})\nax.plot_surface(W1, W2, L)",
         {"post": f"{SURF}, fc.ax().name", "require": ["subplots", "subplot_kw"]}),
        ("set_zlim", f"On a 3-D Axes, draw the surface `L` and set the z limits to {zl[0]} … {zl[1]}.",
         f"{new}ax.plot_surface(W1, W2, L)\nax.set_zlim({zl[0]}, {zl[1]})",
         {"post": f"{SURF}, fc.r(fc.ax().get_zlim(), 4)", "require": ["set_zlim|zlim"]}),
    ]
    return catalog(rng, items, code, f"{desc}.")


# ================================================================= easy

@template(M, "easy")
def pie_readback(rng):
    code, desc, sizes, labels = _shares(rng)
    sa = rng.choice([0, 90, 30])
    ex = rng.choice([0.1, 0.2])
    fr = rng.choice([[0.2, 0.3], [0.1, 0.25, 0.15], [0.4, 0.2]])
    items = [
        ("theta1", f"Draw `wedges, texts = ax.pie(sizes, startangle={sa})` and return each wedge's start and end angle "
                   f"in degrees, `[[w.theta1, w.theta2], ...]`, rounded to 2 decimals.",
         f"fig, ax = plt.subplots()\nwedges, texts = ax.pie(sizes, startangle={sa})\n"
         f"result = [[round(float(w.theta1), 2), round(float(w.theta2), 2)] for w in wedges]", {"require": ["pie"]}),
        ("autopct", "With `autopct`, `ax.pie` returns three lists. Draw `wedges, texts, autotexts = ax.pie(sizes, "
                    "autopct='%1.0f%%')` and return the percentage texts.",
         "fig, ax = plt.subplots()\nwedges, texts, autotexts = ax.pie(sizes, autopct='%1.0f%%')\n"
         "result = [t.get_text() for t in autotexts]", {"require": ["pie", "autopct"]}),
        ("patches", "Draw `ax.pie(sizes, labels=labels, autopct='%1.1f%%')` and return `[len(ax.patches), "
                    "len(ax.texts)]`: one wedge per size, and a label plus a percentage text per wedge.",
         "fig, ax = plt.subplots()\nax.pie(sizes, labels=labels, autopct='%1.1f%%')\nresult = [len(ax.patches), len(ax.texts)]",
         {"require": ["pie"]}),
        ("explode", f"Draw `wedges, _ = ax.pie(sizes, explode=[{ex}] + [0] * (len(sizes) - 1))` and return the "
                    f"centers of the first two wedges, rounded to 4 decimals: the exploded one is shifted by {ex} "
                    f"along its middle angle.",
         f"fig, ax = plt.subplots()\nwedges, _ = ax.pie(sizes, explode=[{ex}] + [0] * (len(sizes) - 1))\n"
         f"result = [[round(float(c), 4) for c in w.center] for w in wedges[:2]]", {"require": ["pie", "explode"]}),
        ("normalize", f"Shares that sum to **less** than 1 draw a partial pie with `normalize=False`. Draw "
                      f"`wedges, _ = ax.pie({fr}, normalize=False)` and return the total angle the wedges cover, in "
                      f"degrees (sum of `theta2 - theta1`, rounded to 2 decimals).",
         f"fig, ax = plt.subplots()\nwedges, _ = ax.pie({fr}, normalize=False)\n"
         f"result = round(float(sum(w.theta2 - w.theta1 for w in wedges)), 2)", {"require": ["pie", "normalize"]}),
        ("autopct", "`autopct` can be a function of the percentage. Draw the pie with the **counts** printed on the "
                    "wedges instead of percentages: `autopct=lambda p: f'{p * sum(sizes) / 100:.0f}'`, and return "
                    "the printed texts (the third list `ax.pie` returns).",
         "fig, ax = plt.subplots()\n_, _, auto = ax.pie(sizes, autopct=lambda p: f'{p * sum(sizes) / 100:.0f}')\n"
         "result = [t.get_text() for t in auto]", {"require": ["pie", "autopct"]}),
        ("get_label", "`ax.pie(sizes, labels=labels)` also sets each wedge's artist label. Return the labels read back "
                      "from the wedges (`w.get_label()`) — they are what a legend would show.",
         "fig, ax = plt.subplots()\nwedges, _ = ax.pie(sizes, labels=labels)\nresult = [w.get_label() for w in wedges]",
         {"require": ["pie", "get_label"]}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "easy")
def polar_conversions(rng):
    n = rng.choice([8, 10, 12])
    s = seeded(rng)
    code = (f"g = np.random.default_rng({s})\nth = np.round(g.uniform(0, 2 * np.pi, {n}), 3)\n"
            f"rr = np.round(g.uniform(0.5, 2, {n}), 3)\nx, y = np.round(rr * np.cos(th), 3), np.round(rr * np.sin(th), 3)\n"
            f"deg = np.round(np.degrees(th), 1)")
    desc = (f"`th` (radians) and `rr` are {n} points in polar form, `x`, `y` are the same points in Cartesian form "
            f"and `deg` holds `th` in degrees")
    k = rng.choice([6, 8, 12])
    loc = rng.choice(["N", "W", "S", "E"])
    items = [
        ("cos", "Plot the polar points on an ordinary (Cartesian) Axes: compute `rr * np.cos(th)`, `rr * np.sin(th)` "
                "and draw them as circles (`'o'`), with equal scales on both axes (`ax.set_aspect('equal')`).",
         "fig, ax = plt.subplots()\nax.plot(rr * np.cos(th), rr * np.sin(th), 'o')\nax.set_aspect('equal')",
         {"post": f"{LINES}, fc.ax().get_aspect()", "require": ["cos", "sin"]}),
        ("arctan2", "Go the other way: compute each point's angle `np.arctan2(y, x)` and radius `np.hypot(x, y)` and "
                    "draw them as circles (`'o'`) on a **polar** Axes.",
         "fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})\nax.plot(np.arctan2(y, x), np.hypot(x, y), 'o')",
         {"post": f"{POLAR}, {LINES}", "require": ["arctan2", "hypot"]}),
        ("scatter", "On a polar Axes, draw the points `(th, rr)` with `ax.scatter`, colored by their radius (`c=rr`) "
                    "in 'viridis'.",
         "fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})\nax.scatter(th, rr, c=rr, cmap='viridis')",
         {"post": f"{POLAR}, {OFF}, {CARR}", "require": ["scatter"]}),
        ("deg2rad", "Polar Axes expect **radians**. Plot the points with angles `deg` (degrees) and radii `rr` as "
                    "circles (`'o'`) on a polar Axes, converting with `np.deg2rad` (or `np.radians`).",
         "fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})\nax.plot(np.deg2rad(deg), rr, 'o')",
         {"post": f"{POLAR}, {LINES}", "require": ["deg2rad|radians"]}),
        ("get_theta_offset", f"Create a polar Axes, call `ax.set_theta_zero_location({loc!r})` and return "
                             f"`ax.get_theta_offset()`: the rotation of angle 0, in radians, counterclockwise from "
                             f"the east.",
         f"fig, ax = plt.subplots(subplot_kw={{'projection': 'polar'}})\nax.set_theta_zero_location({loc!r})\n"
         f"result = ax.get_theta_offset()", {"require": ["set_theta_zero_location"], "tol": 1e-9}),
        ("bar", f"A polar bar chart: count the points in {k} equal angle bins (`np.histogram(th, bins={k}, range=(0, "
                f"2 * np.pi))`) and draw them with `ax.bar` at the bins' left edges with width 2π/{k} and "
                f"`align='edge'`.",
         f"counts, edges = np.histogram(th, bins={k}, range=(0, 2 * np.pi))\n"
         f"fig, ax = plt.subplots(subplot_kw={{'projection': 'polar'}})\nax.bar(edges[:-1], counts, width=2 * np.pi / {k}, align='edge')",
         {"post": f"{POLAR}, fc.bars(fc.ax(), 4)", "require": ["histogram", "bar"]}),
        ("fill_between", "On a polar Axes, sort the points by angle and shade the area between radius 0 and the "
                         "curve through them (`ax.fill_between(t, 0, r)` with `order = np.argsort(th)`).",
         "order = np.argsort(th)\nfig, ax = plt.subplots(subplot_kw={'projection': 'polar'})\n"
         "ax.fill_between(th[order], 0, rr[order])",
         {"post": f"{POLAR}, [fc.r(c.get_paths()[0].vertices, 3) for c in fc.ax().collections]",
          "require": ["argsort", "fill_between"]}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "easy")
def radar_basics(rng):
    code, desc, m, models = _models(rng, k=1)
    al = rng.choice([0.2, 0.25, 0.3])
    col = rng.choice(["C0", "C1", "tab:green", "purple"])
    ticks = rng.choice([[0.25, 0.5, 0.75], [0.2, 0.4, 0.6, 0.8], [0.5, 1.0]])
    close = CLOSE.format(m=m) + "vals = np.append(scores[0], scores[0][0])\n"
    pol = "fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})\n"
    items = [
        ("append", f"A radar chart puts the {m} metrics at equal angles (`np.linspace(0, 2 * np.pi, {m}, "
                   f"endpoint=False)`). To **close** the shape, repeat the first point at the end. Return the closed "
                   f"angle and value arrays as `[th.tolist(), vals.tolist()]` for model 0 (`scores[0]`).",
         CLOSE.format(m=m) + "vals = np.append(scores[0], scores[0][0])\nresult = [th.tolist(), vals.tolist()]",
         {"require": ["append|concatenate|r_"], "tol": 1e-9}),
        ("plot", "On a polar Axes, draw model 0's radar line: angles evenly spaced (`endpoint=False`), first point "
                 "repeated at the end, style `'o-'`.", f"{close}{pol}ax.plot(th, vals, 'o-')",
         {"post": f"fc.lines(fc.ax(), 'marker', 'ls')", "require": ["plot"]}),
        ("fill", f"On a polar Axes, draw model 0's closed radar line in {col!r} and fill its inside with the same "
                 f"color at opacity {al} (`ax.fill`).",
         f"{close}{pol}ax.plot(th, vals, color={col!r})\nax.fill(th, vals, color={col!r}, alpha={al})",
         {"post": f"fc.lines(fc.ax(), 'color'), fc.patches(fc.ax()), [(fc.hexcolor(p.get_facecolor()), p.get_alpha()) "
                  f"for p in fc.ax().patches]", "require": ["fill"]}),
        ("set_xticks", "On a polar Axes, draw model 0's closed radar line and label the metric directions: "
                       "`ax.set_xticks(angles, labels=metrics)` (the open `angles`, one per metric).",
         f"{close}{pol}ax.plot(th, vals)\nax.set_xticks(angles, labels=metrics)",
         {"post": f"{LINES}, fc.ticks(fc.ax(), 'x', nd=4)", "require": ["set_xticks"]}),
        ("set_ylim", "Scores live in 0 … 1: draw model 0's closed radar line and fix the radius range to 0 … 1 "
                     "(`ax.set_ylim(0, 1)` — on a polar Axes y is the radius).",
         f"{close}{pol}ax.plot(th, vals)\nax.set_ylim(0, 1)",
         {"post": f"{LINES}, fc.r(fc.ax().get_ylim(), 4)", "require": ["set_ylim|set_rlim|set_rmax"]}),
        ("set_rticks", f"Draw model 0's closed radar line, put the radial grid circles at {ticks} (`ax.set_rticks`) "
                       f"and their labels along the 0° direction (`ax.set_rlabel_position(0)`).",
         f"{close}{pol}ax.plot(th, vals)\nax.set_rticks({ticks})\nax.set_rlabel_position(0)",
         {"post": f"{LINES}, fc.r(fc.ax().get_yticks(), 4), fc.ax().get_rlabel_position()",
          "require": ["set_rticks", "set_rlabel_position"]}),
        ("set_theta_zero_location", "Radar charts usually start at the top and go clockwise. Draw model 0's closed "
                                    "radar line with angle 0 at 'N' and the angles increasing clockwise.",
         f"{close}{pol}ax.plot(th, vals)\nax.set_theta_zero_location('N')\nax.set_theta_direction(-1)",
         {"post": f"{LINES}, fc.r(fc.ax().get_theta_offset(), 6), fc.ax().get_theta_direction()",
          "require": ["set_theta_zero_location", "set_theta_direction"]}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "easy")
def surface_prep(rng):
    n = rng.choice([20, 25, 30])
    a = rng.choice([1, 2, 3])
    f, name = rng.choice(LOSSES3D)
    rs = rng.choice([2, 3, 5])
    al = rng.choice([0.5, 0.7, 0.9])
    k = rng.choice([40, 60, 100])
    s = seeded(rng)
    turns = rng.choice([2, 3, 4])
    grid = f"w = np.linspace(-{a}, {a}, {n})"
    sol_grid = f"W1, W2 = np.meshgrid(w, w)\nL = {f}\n"
    new = "fig = plt.figure()\nax = fig.add_subplot(projection='3d')\n"
    items = [
        ("meshgrid", f"`w = np.linspace(-{a}, {a}, {n})`. Build the grid `W1, W2 = np.meshgrid(w, w)`, compute "
                     f"`L = {f}` ({name}) and draw it with `ax.plot_surface` on a 3-D Axes in 'viridis'.",
         f"{sol_grid}{new}ax.plot_surface(W1, W2, L, cmap='viridis')",
         {"setup": grid, "post": SURF, "require": ["meshgrid", "plot_surface"]}),
        ("alpha", f"`w = np.linspace(-{a}, {a}, {n})`. Build the grid with `np.meshgrid`, compute `L = {f}` and draw "
                  f"a smooth see-through surface: `edgecolor='none'`, opacity {al}.",
         f"{sol_grid}{new}ax.plot_surface(W1, W2, L, edgecolor='none', alpha={al})",
         {"setup": grid, "post": f"{SURF}, [c.get_alpha() for c in fc.ax().collections]",
          "require": ["meshgrid", "plot_surface", "alpha"]}),
        ("scatter", f"`P` is a {k} × 3 point cloud. Draw it on a 3-D Axes with `ax.scatter`, colored by its z "
                    f"coordinate (`c=P[:, 2]`) in 'plasma'.",
         f"{new}ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=P[:, 2], cmap='plasma')",
         {"setup": f"P = np.round(np.random.default_rng({s}).normal(size=({k}, 3)), 3)",
          "post": f"{SC3}, [(fc.r(c.get_array(), 4), c.get_cmap().name) for c in fc.ax().collections]",
          "require": ["scatter"]}),
        ("cos", f"Draw a helix as a 3-D line: `t = np.linspace(0, {turns} * 2 * np.pi, 200)`, x = cos t, y = sin t, "
                f"z = t / (2π) (it climbs one unit per turn).",
         f"t = np.linspace(0, {turns} * 2 * np.pi, 200)\n{new}ax.plot(np.cos(t), np.sin(t), t / (2 * np.pi))",
         {"post": L3, "require": ["cos", "sin"]}),
        ("plot_wireframe", f"`w = np.linspace(-{a}, {a}, {n})`. Build the grid, compute `L = {f}` and draw `0.2 * L` "
                           f"as a wireframe with `rstride={rs}, cstride={rs}` and line width 0.5.",
         f"{sol_grid}{new}ax.plot_wireframe(W1, W2, 0.2 * L, rstride={rs}, cstride={rs}, linewidth=0.5)",
         {"setup": grid, "post": WIRE, "require": ["meshgrid", "plot_wireframe"]}),
        ("add_subplot", f"`w = np.linspace(-{a}, {a}, {n})`. Build the grid, compute `L = {f}`, then show it twice "
                        f"side by side: as a surface on a 3-D Axes (`fig.add_subplot(1, 2, 1, projection='3d')`) and "
                        f"as filled contours (`levels=10`) on an ordinary Axes (`fig.add_subplot(1, 2, 2)`).",
         f"{sol_grid}fig = plt.figure()\nax3 = fig.add_subplot(1, 2, 1, projection='3d')\nax3.plot_surface(W1, W2, L)\n"
         f"ax2 = fig.add_subplot(1, 2, 2)\nax2.contourf(W1, W2, L, levels=10)",
         {"setup": grid, "post": "[a.name for a in fc.axes()], [sorted(type(c).__name__ for c in a.collections) for a in fc.axes()]",
          "require": ["add_subplot", "projection"]}),
    ]
    return catalog(rng, items, "", "")


@template(M, "easy")
def translate_polar_3d(rng):
    code, desc = _rose(rng)
    lcode, ldesc, nx, ny = _landscape(rng, n=(12, 20))
    scode, sdesc, sizes, labels = _shares(rng)
    radii = rng.choice([[0.5, 1, 1.5], [1, 2], [0.8, 1.6]])
    title = rng.choice(["Rose", "Polar curve", "Pattern"])
    items = [
        ("subplot_kw", f"Rewrite in the explicit style (`fig, ax = plt.subplots(subplot_kw=...)`, then `ax.` methods):"
                       f"\n```python\nplt.polar(theta, r)\nplt.title({title!r})\n```",
         f"fig, ax = plt.subplots(subplot_kw={{'projection': 'polar'}})\nax.plot(theta, r)\nax.set_title({title!r})",
         {"post": f"{POLAR}, {LINES}, fc.labels(fc.ax())['title']", "require": ["subplot_kw|add_subplot"]}),
        ("plt.rgrids", f"Rewrite with pyplot only (`plt.polar`, `plt.rgrids`):\n```python\nfig, ax = plt.subplots("
                       f"subplot_kw={{'projection': 'polar'}})\nax.plot(theta, r)\nax.set_rticks({radii})\n```",
         f"plt.polar(theta, r)\nplt.rgrids({radii})", {"post": f"{LINES}, fc.r(fc.ax().get_yticks(), 4)",
                                                        "require": ["polar", "rgrids"]}),
        ("plt.thetagrids", "Rewrite with pyplot only (`plt.polar`, `plt.thetagrids` takes **degrees**):\n```python\n"
                           "fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})\nax.plot(theta, r)\n"
                           "ax.set_xticks(np.deg2rad([0, 90, 180, 270]), labels=['E', 'N', 'W', 'S'])\n```",
         "plt.polar(theta, r)\nplt.thetagrids([0, 90, 180, 270], ['E', 'N', 'W', 'S'])",
         {"post": f"{LINES}, fc.ticks(fc.ax(), 'x', nd=4)", "require": ["polar", "thetagrids"]}),
        ("plt.pie", f"{sdesc}. Rewrite with pyplot only:\n```python\nfig, ax = plt.subplots()\nax.pie(sizes, "
                    f"labels=labels, startangle=90)\nax.set_title('share')\n```",
         "plt.pie(sizes, labels=labels, startangle=90)\nplt.title('share')",
         {"setup": scode, "post": f"{PIE}, {TXT}, fc.labels(fc.ax())['title']", "require": ["pie", "title"]}),
        ("add_subplot", f"{ldesc}. Rewrite with `fig = plt.figure()` and `fig.add_subplot(...)` instead of "
                        f"`plt.subplots`:\n```python\nfig, ax = plt.subplots(subplot_kw={{'projection': '3d'}})\n"
                        f"ax.plot_surface(W1, W2, L, cmap='viridis')\n```",
         "fig = plt.figure()\nax = fig.add_subplot(projection='3d')\nax.plot_surface(W1, W2, L, cmap='viridis')",
         {"setup": lcode, "post": f"{SURF}, fc.ax().name", "require": ["figure", "add_subplot"]}),
        ("plt.axes", f"{ldesc}. Pyplot can create a 3-D Axes directly: rewrite with `ax = plt.axes(projection='3d')`:"
                     f"\n```python\nfig = plt.figure()\nax = fig.add_subplot(projection='3d')\nax.plot_wireframe(W1, W2, "
                     f"L)\nax.set_zlabel('loss')\n```",
         "ax = plt.axes(projection='3d')\nax.plot_wireframe(W1, W2, L)\nax.set_zlabel('loss')",
         {"setup": lcode, "post": f"{WIRE}, fc.ax().get_zlabel()", "require": ["axes", "projection"]}),
    ]
    return catalog(rng, items, code, f"{desc}.")


# ================================================================= medium

@template(M, "medium")
def dataset_pies(rng):
    code, desc, sizes, labels = _shares(rng, k=(4, 6))
    k = len(sizes)
    top = rng.choice([2, 3])
    w = rng.choice([0.3, 0.4])
    s = seeded(rng)
    code += f"\nafter = np.round(np.array(sizes) * np.random.default_rng({s}).uniform(0.5, 1.5, {k})).astype(int)"
    desc += " (`after` holds the counts after resampling)"
    items = [
        ("pie", "The guide's pie: draw `sizes` with `labels`, percentages with one decimal (`autopct='%1.1f%%'`), the "
                "first wedge at the top (`startangle=90`), drawn clockwise, with the first wedge pulled out by 0.05.",
         f"fig, ax = plt.subplots()\nax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, "
         f"explode=[0.05] + [0] * {k - 1}, counterclock=False)",
         {"post": f"{PIE}, {TXT}", "require": ["pie", "autopct", "startangle", "explode", "counterclock"]}),
        ("wedgeprops", f"Draw `sizes` as a donut (wedge width {w}, white wedge edges: `wedgeprops={{'width': {w}, "
                       f"'edgecolor': 'w'}}`) labelled with `labels`, and write the total count in the hole: "
                       f"`ax.text(0, 0, f'N={{sum(sizes)}}', ha='center', va='center')`.",
         f"fig, ax = plt.subplots()\nax.pie(sizes, labels=labels, wedgeprops={{'width': {w}, 'edgecolor': 'w'}})\n"
         f"ax.text(0, 0, f'N={{sum(sizes)}}', ha='center', va='center')",
         {"post": f"{PIE}, [(p.width, fc.hexcolor(p.get_edgecolor())) for p in fc.ax().patches], {TXT}",
          "require": ["pie", "wedgeprops", "text"]}),
        ("radius", f"Nested donut: the outer ring shows `sizes` (radius 1), the inner ring `after` (radius 0.7), both "
                   f"with wedge width 0.3 (`wedgeprops={{'width': 0.3}}`). Draw the outer ring first.",
         "fig, ax = plt.subplots()\nax.pie(sizes, radius=1, wedgeprops={'width': 0.3})\n"
         "ax.pie(after, radius=0.7, wedgeprops={'width': 0.3})",
         {"post": f"{PIE}, [p.width for p in fc.ax().patches]", "require": ["pie", "radius"]}),
        ("legend", "Draw `sizes` as a pie **without** labels on the wedges and put the names in a legend instead: "
                   "`wedges, _ = ax.pie(sizes)`, then `ax.legend(wedges, labels, title='class')`.",
         "fig, ax = plt.subplots()\nwedges, _ = ax.pie(sizes)\nax.legend(wedges, labels, title='class')",
         {"post": f"{PIE}, fc.legend(), fc.legend_title()", "require": ["pie", "legend"]}),
        ("subplots", "Before / after resampling: two pies side by side (`plt.subplots(1, 2)`) — `sizes` titled "
                     "'before' and `after` titled 'after' — both labelled with `labels` and percentages "
                     "(`autopct='%1.0f%%'`).",
         "fig, (a1, a2) = plt.subplots(1, 2)\na1.pie(sizes, labels=labels, autopct='%1.0f%%')\na1.set_title('before')\n"
         "a2.pie(after, labels=labels, autopct='%1.0f%%')\na2.set_title('after')",
         {"post": _per_axes("(fc.patches(a), fc.texts(a), a.get_title())"), "require": ["pie"]}),
        ("argsort", f"Too many small wedges: keep the {top} largest classes (largest first) and merge the rest into "
                    f"one 'other' wedge at the end (ties: the earlier class first); draw that pie with labels.",
         f"order = np.argsort(-np.array(sizes), kind='stable')\nkeep = order[:{top}]\nvals = [sizes[i] for i in keep] + "
         f"[sum(sizes[i] for i in order[{top}:])]\nnames = [labels[i] for i in keep] + ['other']\n"
         f"fig, ax = plt.subplots()\nax.pie(vals, labels=names)",
         {"post": f"{PIE}, {TXT}", "require": ["pie"]}),
        ("autopct", "Draw `after` as a pie labelled with `labels`, showing on each wedge both the count and the "
                    "percentage, like '12 (30%)': `autopct=lambda p: f'{round(p * total / 100)} ({p:.0f}%)'` with "
                    "`total = after.sum()`.",
         "total = after.sum()\nfig, ax = plt.subplots()\n"
         "ax.pie(after, labels=labels, autopct=lambda p: f'{round(p * total / 100)} ({p:.0f}%)')",
         {"post": f"{PIE}, {TXT}", "require": ["pie", "autopct"]}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "medium")
def radar_charts(rng):
    k = rng.choice([2, 2, 3])
    code, desc, m, models = _models(rng, k=k)
    al = rng.choice([0.1, 0.2, 0.25])
    pol = "fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})\n"
    ang = f"angles = np.linspace(0, 2 * np.pi, {m}, endpoint=False)\nth = np.append(angles, angles[0])\n"
    loop = "for name, row in zip(models, scores):\n    vals = np.append(row, row[0])\n"
    items = [
        ("legend", f"Drill: compare the {k} models on one radar chart: a closed `'o-'` line per model (first point "
                   f"repeated), labelled with the model name, and a legend.",
         f"{ang}{pol}{loop}    ax.plot(th, vals, 'o-', label=name)\nax.legend()",
         {"post": f"fc.lines(fc.ax(), 'label'), fc.legend()", "require": ["append|concatenate|r_", "legend"]}),
        ("fill", f"Draw each model's closed radar line (labelled with its name) and fill its area at opacity {al}, "
                 f"then label the metric directions with `ax.set_xticks(angles, labels=metrics)`.",
         f"{ang}{pol}{loop}    ax.plot(th, vals, label=name)\n    ax.fill(th, vals, alpha={al})\n"
         f"ax.set_xticks(angles, labels=metrics)",
         {"post": f"fc.lines(fc.ax(), 'label'), fc.patches(fc.ax()), [p.get_alpha() for p in fc.ax().patches], "
                  f"fc.ticks(fc.ax(), 'x', nd=4)['labels']", "require": ["fill", "set_xticks"]}),
        ("set_theta_direction", "Draw each model's closed radar line, labelled, starting at the top (`'N'`) and "
                                "going clockwise, with a legend.",
         f"{ang}{pol}ax.set_theta_zero_location('N')\nax.set_theta_direction(-1)\n{loop}    ax.plot(th, vals, label=name)\n"
         f"ax.legend()",
         {"post": f"fc.lines(fc.ax(), 'label'), fc.r(fc.ax().get_theta_offset(), 6), fc.ax().get_theta_direction(), "
                  f"fc.legend()", "require": ["set_theta_zero_location", "set_theta_direction"]}),
        ("set_rticks", "Draw each model's closed radar line on the fixed radius range 0 … 1 (`ax.set_ylim(0, 1)`), with "
                       "grid circles at 0.2, 0.4, 0.6, 0.8 (`ax.set_rticks`) and the metric names on the spokes.",
         f"{ang}{pol}{loop}    ax.plot(th, vals)\nax.set_ylim(0, 1)\nax.set_rticks([0.2, 0.4, 0.6, 0.8])\n"
         f"ax.set_xticks(angles, labels=metrics)",
         {"post": f"{LINES}, fc.r(fc.ax().get_ylim(), 4), fc.r(fc.ax().get_yticks(), 4), "
                  f"fc.ticks(fc.ax(), 'x', nd=4)['labels']", "require": ["set_rticks"]}),
        ("subplots", f"One radar per model: a row of {k} polar Axes (`plt.subplots(1, {k}, subplot_kw={{'projection': "
                     f"'polar'}})`), each with that model's closed, filled (opacity {al}) radar, titled with its name.",
         f"{ang}fig, axs = plt.subplots(1, {k}, subplot_kw={{'projection': 'polar'}})\n"
         f"for a, name, row in zip(axs, models, scores):\n    vals = np.append(row, row[0])\n    a.plot(th, vals)\n"
         f"    a.fill(th, vals, alpha={al})\n    a.set_title(name)",
         {"post": _per_axes("(a.name, fc.lines(a), len(a.patches), a.get_title())"), "require": ["subplot_kw"]}),
        ("max", "Metrics on a radar should be comparable: divide each metric (column) by its best score over the "
                "models (`scores / scores.max(axis=0)`), then draw each model's closed radar line, labelled, with "
                "a legend.",
         f"norm = scores / scores.max(axis=0)\n{ang}{pol}for name, row in zip(models, norm):\n"
         f"    vals = np.append(row, row[0])\n    ax.plot(th, vals, label=name)\nax.legend()",
         {"post": f"fc.lines(fc.ax(), 'label'), fc.legend()", "require": ["max"]}),
        ("set_rlabel_position", "Draw each model's closed radar line and move the radius labels so they don't sit on "
                                "a spoke: put them at half the angle between the first two spokes "
                                f"(`ax.set_rlabel_position(180 / {m})`).",
         f"{ang}{pol}{loop}    ax.plot(th, vals)\nax.set_rlabel_position(180 / {m})",
         {"post": f"{LINES}, fc.r(fc.ax().get_rlabel_position(), 4)", "require": ["set_rlabel_position"]}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "medium")
def polar_ml_plots(rng):
    n = rng.choice([60, 90, 120])
    k = rng.choice([8, 12, 16])
    s = seeded(rng)
    turns = rng.choice([2, 3])
    code = (f"g = np.random.default_rng({s})\nangles = np.round(np.mod(g.normal(1.0, 0.9, {n}), 2 * np.pi), 4)\n"
            f"hours = g.integers(0, 24, {n})\nnorms = np.round(g.uniform(0.2, 1.0, {n}), 3)\n"
            f"cls = g.integers(0, 3, {n})")
    desc = (f"`angles` (radians, 0 … 2π) are the directions of {n} gradient updates and `norms` their sizes; `hours` "
            f"is the hour (0 … 23) each training job started and `cls` a class label (0, 1, 2) per point")
    pol = "fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})\n"
    items = [
        ("histogram", f"Rose diagram: count `angles` in {k} equal bins over 0 … 2π (`np.histogram`) and draw the "
                      f"counts with `ax.bar` on a polar Axes, at the bin **centers**, with width 2π/{k}.",
         f"counts, edges = np.histogram(angles, bins={k}, range=(0, 2 * np.pi))\ncenters = (edges[:-1] + edges[1:]) / 2\n"
         f"{pol}ax.bar(centers, counts, width=2 * np.pi / {k})",
         {"post": f"{POLAR}, fc.bars(fc.ax(), 4)", "require": ["histogram", "bar"]}),
        ("set_xticks", "Job start clock: count the jobs per hour (`np.bincount(hours, minlength=24)`) and draw them as "
                       "24 polar bars (angle 2π·h/24, width 2π/24) with 0 h at the top ('N'), clockwise, and spokes "
                       "every 3 hours labelled '0h', '3h', …, '21h'.",
         f"counts = np.bincount(hours, minlength=24)\nth = 2 * np.pi * np.arange(24) / 24\n{pol}"
         f"ax.bar(th, counts, width=2 * np.pi / 24)\nax.set_theta_zero_location('N')\nax.set_theta_direction(-1)\n"
         f"ax.set_xticks(th[::3], labels=[f'{{h}}h' for h in range(0, 24, 3)])",
         {"post": f"fc.bars(fc.ax(), 4), fc.ticks(fc.ax(), 'x', nd=4), fc.r(fc.ax().get_theta_offset(), 6), "
                  f"fc.ax().get_theta_direction()", "require": ["bincount", "bar", "set_xticks"]}),
        ("scatter", "Draw the gradient updates as a polar scatter: angle `angles`, radius `norms`, colored by `cls` "
                    "with the colormap 'tab10'.",
         f"{pol}ax.scatter(angles, norms, c=cls, cmap='tab10')",
         {"post": f"{POLAR}, {OFF}, {CARR}", "require": ["scatter"]}),
        ("set_rmax", f"Draw the Archimedean spiral r = θ / (2π) for {turns} full turns (`theta = np.linspace(0, "
                     f"{turns} * 2 * np.pi, 300)`) on a polar Axes, with the radius axis ending at {turns} "
                     f"(`ax.set_rmax`) and radial ticks at the whole numbers 1 … {turns}.",
         f"theta = np.linspace(0, {turns} * 2 * np.pi, 300)\n{pol}ax.plot(theta, theta / (2 * np.pi))\n"
         f"ax.set_rmax({turns})\nax.set_rticks(range(1, {turns + 1}))",
         {"post": f"{LINES}, fc.r(fc.ax().get_ylim(), 4), fc.r(fc.ax().get_yticks(), 4)",
          "require": ["set_rmax|set_rlim|set_ylim", "set_rticks"]}),
        ("set_thetamax", "Gauge chart on a half circle (`set_thetamin(0)`, `set_thetamax(180)`): show the mean of "
                         "`norms` as a single bar at angle π·(1 − mean) (so 0 is on the right… and 1 on the left) "
                         "with radius 1 and width 0.1.",
         f"m = norms.mean()\n{pol}ax.set_thetamin(0)\nax.set_thetamax(180)\nax.bar(np.pi * (1 - m), 1, width=0.1)",
         {"post": f"fc.bars(fc.ax(), 4), fc.r([fc.ax().get_thetamin(), fc.ax().get_thetamax()], 4)",
          "require": ["set_thetamin", "set_thetamax", "bar"]}),
        ("arctan2", "The update vectors are (`norms·cos(angles)`, `norms·sin(angles)`). Average the **vectors** "
                    "(not the angles: angles wrap around) and draw the mean direction as a red line from the center "
                    "to the mean vector's length (`ax.plot([a, a], [0, length], 'r-')`) on top of the polar scatter "
                    "of the updates.",
         "vx, vy = (norms * np.cos(angles)).mean(), (norms * np.sin(angles)).mean()\na = np.arctan2(vy, vx)\n"
         f"{pol}ax.scatter(angles, norms)\nax.plot([a, a], [0, np.hypot(vx, vy)], 'r-')",
         {"post": f"fc.lines(fc.ax(), 'color'), {OFF}", "require": ["arctan2"]}),
        ("fill", f"Draw the direction histogram as a closed **outline** instead of bars: counts of `angles` in {k} bins "
                 f"over 0 … 2π, plotted at the bin centers with the first point repeated at the end, and filled at "
                 f"opacity 0.3.",
         f"counts, edges = np.histogram(angles, bins={k}, range=(0, 2 * np.pi))\nc = (edges[:-1] + edges[1:]) / 2\n"
         f"th, vals = np.append(c, c[0]), np.append(counts, counts[0])\n{pol}ax.plot(th, vals)\nax.fill(th, vals, alpha=0.3)",
         {"post": f"{LINES}, [fc.r(p.get_xy(), 4) for p in fc.ax().patches]", "require": ["histogram", "fill"]}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "medium")
def loss_surface_3d(rng):
    code, desc, nx, ny = _landscape(rng, n=(20, 32))
    cm = rng.choice(["viridis", "plasma", "coolwarm", "cividis"])
    el, az = rng.choice([(30, -60), (25, 45), (40, 120)])
    steps = rng.choice([6, 8, 10])
    new = "fig = plt.figure()\nax = fig.add_subplot(projection='3d')\n"
    off = "off = L.min() - 1\n"
    items = [
        ("offset", f"Drill: draw `L` as a surface in {cm!r} (no edge lines: `edgecolor='none'`) with its contour "
                   f"lines projected **underneath** on the plane z = `L.min() - 1` (`zdir='z'`, `offset=...`, same "
                   f"colormap), and set the z limits to `L.min() - 1` … `L.max()`.",
         f"{off}{new}ax.plot_surface(W1, W2, L, cmap={cm!r}, edgecolor='none')\n"
         f"ax.contour(W1, W2, L, zdir='z', offset=off, cmap={cm!r})\nax.set_zlim(off, L.max())",
         {"post": f"{SURF}, {CONT3}, fc.r(fc.ax().get_zlim(), 4)", "require": ["plot_surface", "contour", "offset"]}),
        ("view_init", f"Draw `L` as a surface in {cm!r}, label the axes 'w1', 'w2', 'loss', and set the view to "
                      f"elevation {el}°, azimuth {az}°.",
         f"{new}ax.plot_surface(W1, W2, L, cmap={cm!r})\nax.set(xlabel='w1', ylabel='w2', zlabel='loss')\n"
         f"ax.view_init(elev={el}, azim={az})",
         {"post": f"{SURF}, {LAB3}, [fc.ax().elev, fc.ax().azim]", "require": ["view_init"]}),
        ("plot", f"Draw `L` as a semi-transparent surface (`alpha=0.6`) and on it the descent path `path` ({steps} "
                 f"points (w1, w2) in a {steps} × 2 array): a red 3-D line with dots, `ax.plot(w1s, w2s, losses, "
                 f"'r.-')`, where the loss of each point is computed with the same formula as `L`.",
         f"{new}ax.plot_surface(W1, W2, L, alpha=0.6)\nw1s, w2s = path[:, 0], path[:, 1]\n"
         f"losses = loss(w1s, w2s)\nax.plot(w1s, w2s, losses, 'r.-')",
         {"post": f"{SURF}, {L3}, fc.lines(fc.ax(), 'color', 'marker', data=False)", "require": ["plot"],
          "setup": f"path = np.round(np.array([[1.8, -1.5]]) * 0.6 ** np.arange({steps})[:, None], 4)\n"
                   f"loss = lambda W1, W2: {code.split(chr(10))[-1][4:]}"}),
        ("add_subplot", "Show `L` two ways side by side: a 3-D surface on the left "
                        "(`fig.add_subplot(1, 2, 1, projection='3d')`) and contour lines with labels (`clabel`) on an "
                        "ordinary Axes on the right (`fig.add_subplot(1, 2, 2)`, `levels=8`).",
         "fig = plt.figure()\nax3 = fig.add_subplot(1, 2, 1, projection='3d')\nax3.plot_surface(W1, W2, L)\n"
         "ax2 = fig.add_subplot(1, 2, 2)\ncs = ax2.contour(W1, W2, L, levels=8)\nax2.clabel(cs)",
         {"post": "[a.name for a in fc.axes()], [len(a.texts) > 0 for a in fc.axes()], "
                  "[fc.r(c.levels, 4) for c in fc.ax(1).collections]", "require": ["add_subplot", "clabel"]}),
        ("unravel_index", "Draw `L` as a wireframe (`rstride=2, cstride=2`) and mark its lowest grid point with a "
                          "red 3-D scatter point (`ax.scatter(x, y, z, color='red')`).",
         f"{new}ax.plot_wireframe(W1, W2, L, rstride=2, cstride=2)\ni, j = np.unravel_index(L.argmin(), L.shape)\n"
         f"ax.scatter(W1[i, j], W2[i, j], L[i, j], color='red')",
         {"post": f"{WIRE}, {SC3}", "require": ["plot_wireframe", "scatter"]}),
        ("colorbar", f"Draw `L` as a surface in {cm!r} and add a colorbar for it, shrunk to 60% "
                     f"(`fig.colorbar(surf, ax=ax, shrink=0.6)`).",
         f"{new}surf = ax.plot_surface(W1, W2, L, cmap={cm!r})\nfig.colorbar(surf, ax=ax, shrink=0.6)",
         {"post": f"{SURF}, len(fc.colorbars())", "require": ["colorbar"]}),
        ("zdir", "Draw `L` as a translucent surface (`alpha=0.5`) and project its contours onto two walls: on the "
                 "plane x = `W1.min()` (`zdir='x'`) and on the plane y = `W2.max()` (`zdir='y'`), using "
                 "`offset=` each time.",
         f"{new}ax.plot_surface(W1, W2, L, alpha=0.5)\nax.contour(W1, W2, L, zdir='x', offset=W1.min())\n"
         f"ax.contour(W1, W2, L, zdir='y', offset=W2.max())",
         {"post": f"{SURF}, {CONT3}", "require": ["contour", "zdir", "offset"]}),
        ("savefig", f"Draw `L` as a surface in {cm!r} on a 4 × 3 inch figure (`plt.figure(figsize=(4, 3))`), save it "
                    f"as 'surface.png' at 40 dpi and close the figure. Return `(plt.imread('surface.png').shape, "
                    f"len(plt.get_fignums()))`.",
         f"fig = plt.figure(figsize=(4, 3))\nax = fig.add_subplot(projection='3d')\nax.plot_surface(W1, W2, L, cmap={cm!r})\n"
         f"fig.savefig('surface.png', dpi=40)\nplt.close(fig)\nresult = (plt.imread('surface.png').shape, len(plt.get_fignums()))",
         {"require": ["savefig", "close"]}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "medium")
def embeddings_3d(rng):
    name, labs = T.labels(rng, 3)
    k = rng.randint(3, min(4, len(labs)))
    n, d = rng.choice([60, 90, 120]), rng.choice([8, 10, 16])
    s = seeded(rng)
    el, az = rng.choice([(20, 30), (30, -45), (15, 120)])
    code = (f"g = np.random.default_rng({s})\ny = g.integers(0, {k}, {n})\nX = np.round(g.normal(size=({n}, {d})) + "
            f"3 * g.normal(size=({k}, {d}))[y], 3)\nclass_names = {lit(list(labs[:k]))}")
    desc = (f"`X` holds {n} embeddings of dimension {d}, `y` their classes (0 … {k - 1}) and `class_names` the {name} "
            f"names")
    pca = "Xc = X - X.mean(axis=0)\nU, S, Vt = np.linalg.svd(Xc, full_matrices=False)\nP = Xc @ Vt[:3].T\n"
    hint = ("Project onto the first 3 principal components (center `X`, `np.linalg.svd(Xc, full_matrices=False)`, "
            "`P = Xc @ Vt[:3].T`)")
    new = "fig = plt.figure()\nax = fig.add_subplot(projection='3d')\n"
    items = [
        ("svd", f"Drill: {hint} and draw `P` as a 3-D scatter colored by class (`c=y`, colormap 'tab10').",
         f"{pca}{new}ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=y, cmap='tab10')",
         {"post": f"[fc.r(np.abs(np.column_stack([np.ma.getdata(v) for v in c._offsets3d])), 3) for c in fc.ax().collections], "
                  f"[(fc.r(c.get_array(), 4), c.get_cmap().name) for c in fc.ax().collections]",
          "require": ["svd", "scatter"]}),
        ("legend", f"{hint}. Draw one 3-D scatter **per class** (`label=` the class name) and add a legend.",
         f"{pca}{new}for c, nm in enumerate(class_names):\n    m = y == c\n    ax.scatter(P[m, 0], P[m, 1], P[m, 2], label=nm)\n"
         f"ax.legend()",
         {"post": "[fc.r(np.abs(np.column_stack([np.ma.getdata(v) for v in c._offsets3d])), 3) for c in fc.ax().collections], "
                  "fc.legend()", "require": ["svd", "legend"]}),
        ("view_init", f"{hint}, draw it as a 3-D scatter colored by class, label the axes 'PC1', 'PC2', 'PC3' and set "
                      f"the view to elevation {el}°, azimuth {az}°.",
         f"{pca}{new}ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=y)\nax.set(xlabel='PC1', ylabel='PC2', zlabel='PC3')\n"
         f"ax.view_init(elev={el}, azim={az})",
         {"post": f"{LAB3}, [fc.ax().elev, fc.ax().azim], len(fc.ax().collections)", "require": ["svd", "view_init"]}),
        ("add_subplot", f"{hint}. Show it two ways side by side: a 3-D scatter of PC1–PC3 (left, "
                        f"`fig.add_subplot(1, 2, 1, projection='3d')`) and a 2-D scatter of PC1 vs PC2 (right), both "
                        f"colored by class (`c=y`).",
         f"{pca}fig = plt.figure()\nax3 = fig.add_subplot(1, 2, 1, projection='3d')\nax3.scatter(P[:, 0], P[:, 1], P[:, 2], c=y)\n"
         f"ax2 = fig.add_subplot(1, 2, 2)\nax2.scatter(P[:, 0], P[:, 1], c=y)",
         {"post": "[a.name for a in fc.axes()], fc.r(np.abs(np.ma.getdata(fc.ax(1).collections[0].get_offsets())), 3)",
          "require": ["svd", "add_subplot"]}),
        ("svd", "How much of the variance do 3 components keep? Return the explained-variance ratio of the first 3 "
                "principal components: `S[:3] ** 2 / (S ** 2).sum()` (a list), from the SVD of the centered `X`.",
         "Xc = X - X.mean(axis=0)\nS = np.linalg.svd(Xc, compute_uv=False)\nresult = (S[:3] ** 2 / (S ** 2).sum()).tolist()",
         {"require": ["svd"], "tol": 1e-6}),
        ("mean", f"{hint}. Draw the points as a small, faint 3-D scatter (`s=5`, `alpha=0.3`) and each class's "
                 f"**centroid** in `P` as a big marker (`s=200`, `marker='*'`) with its class name written next to "
                 f"it with `ax.text(x, y, z, name)`.",
         f"{pca}{new}ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=5, alpha=0.3)\nfor c, nm in enumerate(class_names):\n"
         f"    cx, cy, cz = P[y == c].mean(axis=0)\n    ax.scatter(cx, cy, cz, s=200, marker='*')\n    ax.text(cx, cy, cz, nm)",
         {"post": "[(fc.r(c.get_sizes(), 2), c.get_alpha()) for c in fc.ax().collections], fc.texts(fc.ax()), "
                  "fc.r(np.abs([t.get_position_3d() for t in fc.ax().texts]), 3)", "require": ["svd", "text"]}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "medium")
def three_d_shapes(rng):
    s = seeded(rng)
    nb = rng.choice([3, 4])
    lrs = ["1e-4", "1e-3", "1e-2", "1e-1"][:nb]
    code = (f"g = np.random.default_rng({s})\nscores = np.round(g.uniform(0.6, 0.95, ({nb}, {nb})), 3)\n"
            f"pts = np.round(g.normal(size=(40, 2)), 3)\nlr_names = {lit(lrs)}")
    desc = (f"`scores[i, j]` is the accuracy of a run with learning rate `lr_names[i]` and batch-size index j "
            f"({nb} × {nb}); `pts` holds 40 2-D points")
    new = "fig = plt.figure()\nax = fig.add_subplot(projection='3d')\n"
    bins = rng.choice([4, 5, 6])
    aspect = rng.choice([(1, 1, 0.5), (2, 1, 1), (1, 1, 1)])
    items = [
        ("bar3d", "Draw `scores` as 3-D bars: one bar per cell at x = i, y = j (`np.meshgrid(np.arange(n), "
                  "np.arange(n), indexing='ij')`, raveled), from z = 0, of size 0.5 × 0.5 × score "
                  "(`ax.bar3d(x, y, z0, dx, dy, dz)`).",
         f"x, y = np.meshgrid(np.arange({nb}), np.arange({nb}), indexing='ij')\n{new}"
         f"ax.bar3d(x.ravel(), y.ravel(), np.zeros({nb * nb}), 0.5, 0.5, scores.ravel())",
         {"post": SURF, "require": ["bar3d"]}),
        ("histogram2d", f"3-D histogram: count `pts` in a {bins} × {bins} grid of bins with `np.histogram2d(pts[:, 0], "
                        f"pts[:, 1], bins={bins})` and draw the counts as 3-D bars at the bins' lower-left corners "
                        f"(x from the first edges, y from the second, `indexing='ij'`) with the bins' width and depth.",
         f"H, xe, ye = np.histogram2d(pts[:, 0], pts[:, 1], bins={bins})\nx, y = np.meshgrid(xe[:-1], ye[:-1], indexing='ij')\n"
         f"{new}ax.bar3d(x.ravel(), y.ravel(), np.zeros(H.size), xe[1] - xe[0], ye[1] - ye[0], H.ravel())",
         {"post": SURF, "require": ["histogram2d", "bar3d"]}),
        ("set_box_aspect", f"Draw `scores` as a surface over the grid `np.meshgrid(np.arange({nb}), np.arange({nb}))` "
                           f"and change the box's proportions to {aspect} (`ax.set_box_aspect`).",
         f"X, Y = np.meshgrid(np.arange({nb}), np.arange({nb}))\n{new}ax.plot_surface(X, Y, scores)\nax.set_box_aspect({aspect})",
         {"post": f"{SURF}, fc.r(fc.ax().get_box_aspect(), 4)", "require": ["set_box_aspect"]}),
        ("text", "Draw `scores` as a 3-D scatter at (i, j, scores[i, j]) and write the learning rate name of the best "
                 "run next to it with `ax.text(i, j, score, lr_names[i])`.",
         f"x, y = np.meshgrid(np.arange({nb}), np.arange({nb}), indexing='ij')\n{new}ax.scatter(x.ravel(), y.ravel(), scores.ravel())\n"
         f"i, j = np.unravel_index(scores.argmax(), scores.shape)\nax.text(i, j, scores[i, j], lr_names[i])",
         {"post": f"{SC3}, fc.texts(fc.ax()), [fc.r(t.get_position_3d(), 4) for t in fc.ax().texts]",
          "require": ["text"]}),
        ("plot_trisurf", "Scattered points need no grid: draw a surface through `pts` with heights "
                         "`z = pts[:, 0] ** 2 - pts[:, 1] ** 2` using `ax.plot_trisurf(x, y, z)` (it triangulates).",
         f"{new}ax.plot_trisurf(pts[:, 0], pts[:, 1], pts[:, 0] ** 2 - pts[:, 1] ** 2)",
         {"post": "[(type(c).__name__, fc.r(c._vec[:3].sum(axis=1), 2)) for c in fc.ax().collections]",
          "require": ["plot_trisurf"]}),
        ("plot", f"Stem plot in 3-D: for every run draw a vertical line from z = 0 up to its score at (i, j) "
                 f"(`ax.plot([i, i], [j, j], [0, s], color='gray')`) and a dot on top (`ax.scatter`).",
         f"{new}for i in range({nb}):\n    for j in range({nb}):\n        ax.plot([i, i], [j, j], [0, scores[i, j]], color='gray')\n"
         f"x, y = np.meshgrid(np.arange({nb}), np.arange({nb}), indexing='ij')\nax.scatter(x.ravel(), y.ravel(), scores.ravel())",
         {"post": f"sorted({L3}), {SC3}", "require": ["plot", "scatter"]}),
    ]
    return catalog(rng, items, code, f"{desc}.")


# ================================================================= hard

@template(M, "hard")
def polar_3d_helpers(rng):
    code, desc, m, models = _models(rng, k=2)
    scode, sdesc, sizes, labels = _shares(rng, k=(4, 6))
    lcode, ldesc, nx, ny = _landscape(rng, n=(10, 16))
    s = seeded(rng)
    bins = rng.choice([8, 12])
    pol = "_f, (_a, _b) = plt.subplots(1, 2, subplot_kw={'projection': 'polar'})\n"
    items = [
        ("fill", f"{desc}. Write `radar(ax, values, labels=None, alpha=0.25, **kwargs)` that draws `values` on the "
                 f"polar Axes `ax` as a closed radar line (`ax.plot(..., **kwargs)`, evenly spaced angles, first point "
                 f"repeated), fills it in the **line's color** at opacity `alpha`, labels the spokes with `labels` "
                 f"when given (`ax.set_xticks`), and returns the line.",
         "def radar(ax, values, labels=None, alpha=0.25, **kwargs):\n    n = len(values)\n"
         "    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)\n    th = np.append(angles, angles[0])\n"
         "    vals = np.append(values, values[0])\n    line, = ax.plot(th, vals, **kwargs)\n"
         "    ax.fill(th, vals, color=line.get_color(), alpha=alpha)\n    if labels is not None:\n"
         "        ax.set_xticks(angles, labels=labels)\n    return line",
         {"require": ["fill", "plot"],
          "post": f"{pol}_l = radar(_a, scores[0], metrics, color='red', label='m')\n_m = radar(_b, scores[1], alpha=0.5)\n"
                  "[(fc.lines(a, 'color'), [(fc.hexcolor(p.get_facecolor()), p.get_alpha(), fc.r(p.get_xy(), 4)) "
                  "for p in a.patches]) for a in (_a, _b)], fc.ticks(_a, 'x', nd=4)['labels'], _l.get_label(), "
                  "_m in _b.get_lines()"}),
        ("argsort", f"{sdesc}. Write `pie_top(ax, sizes, labels, k=3)` that draws a pie of the `k` largest entries "
                    f"(largest first; ties: earlier first) plus one 'other' wedge with the sum of the rest (only if "
                    f"something is left), labelled, and returns the wedges.",
         "def pie_top(ax, sizes, labels, k=3):\n    order = np.argsort(-np.asarray(sizes), kind='stable')\n"
         "    vals = [sizes[i] for i in order[:k]]\n    names = [labels[i] for i in order[:k]]\n    if len(order) > k:\n"
         "        vals.append(sum(sizes[i] for i in order[k:]))\n        names.append('other')\n"
         "    wedges, _ = ax.pie(vals, labels=names)\n    return wedges",
         {"setup": scode, "require": ["pie"],
          "post": "_f, (_a, _b) = plt.subplots(1, 2)\n_w = pie_top(_a, sizes, labels, k=2)\n_v = pie_top(_b, sizes, labels, k=len(sizes))\n"
                  "[(fc.patches(a), fc.texts(a)) for a in (_a, _b)], len(_w), len(_v)"}),
        ("offset", f"{ldesc}. Write `surface_with_floor(ax, X, Y, Z, cmap='viridis')` that draws `Z` as a surface on "
                   f"the 3-D Axes `ax` and projects its contour lines onto the floor z = `Z.min() - 1` (`zdir='z'`, "
                   f"`offset=`, same colormap), sets the z limits to `Z.min() - 1` … `Z.max()`, and returns the "
                   f"surface.",
         "def surface_with_floor(ax, X, Y, Z, cmap='viridis'):\n    floor = Z.min() - 1\n"
         "    surf = ax.plot_surface(X, Y, Z, cmap=cmap)\n    ax.contour(X, Y, Z, zdir='z', offset=floor, cmap=cmap)\n"
         "    ax.set_zlim(floor, Z.max())\n    return surf",
         {"setup": lcode, "require": ["plot_surface", "contour", "offset"],
          "post": "_f = plt.figure()\n_a = _f.add_subplot(1, 2, 1, projection='3d')\n_b = _f.add_subplot(1, 2, 2, projection='3d')\n"
                  "_s = surface_with_floor(_a, W1, W2, L)\n_t = surface_with_floor(_b, W1, W2, 2 * L, cmap='magma')\n"
                  "plt.sca(_a)\n_x = " + SURF + "\n_y = " + CONT3 + "\nplt.sca(_b)\n"
                  "(_x, _y, " + CONT3 + ", [c.get_cmap().name for c in _b.collections], fc.r(_a.get_zlim(), 4), "
                  "fc.r(_b.get_zlim(), 4), _s in _a.collections)"}),
        ("legend", "`P` (n × 3) are 3-D embeddings, `y` their classes and `names` the class names. Write "
                   "`scatter_by_class(ax, P, y, names)` that draws one 3-D scatter per class present in `names` "
                   "(class c = `names[c]`, `label=` the name), adds a legend, and returns the list of scatters.",
         "def scatter_by_class(ax, P, y, names):\n    out = []\n    for c, nm in enumerate(names):\n        m = y == c\n"
         "        out.append(ax.scatter(P[m, 0], P[m, 1], P[m, 2], label=nm))\n    ax.legend()\n    return out",
         {"setup": f"g = np.random.default_rng({s})\nP = np.round(g.normal(size=(30, 3)), 3)\ny = g.integers(0, 3, 30)\n"
                   f"names = ['a', 'b', 'c']",
          "require": ["scatter", "legend"],
          "post": "_f = plt.figure()\n_a = _f.add_subplot(projection='3d')\n_r = scatter_by_class(_a, P, y, names)\n"
                  "(" + SC3 + ", fc.legend(_a), len(_r))"}),
        ("bar", f"Write `rose(ax, angles, bins=16)` that draws a rose diagram of `angles` (radians, any real values: "
                f"wrap them into 0 … 2π with `np.mod`) on the polar Axes `ax`: counts in `bins` equal bins over "
                f"0 … 2π, one bar per bin at its center, width 2π/bins. Return the bar container.",
         "def rose(ax, angles, bins=16):\n    a = np.mod(angles, 2 * np.pi)\n"
         "    counts, edges = np.histogram(a, bins=bins, range=(0, 2 * np.pi))\n"
         "    return ax.bar((edges[:-1] + edges[1:]) / 2, counts, width=2 * np.pi / bins)",
         {"setup": f"angles = np.round(np.random.default_rng({s}).normal(0, 3, 80), 4)", "require": ["histogram", "bar"],
          "post": f"{pol}_r = rose(_a, angles)\n_s = rose(_b, angles + 10, bins={bins})\n"
                  "[fc.bars(a, 4) for a in (_a, _b)], len(_r), len(_s)"}),
        ("set_theta_direction", "Write `clock_face(ax, hours=24, step=3)` that turns the polar Axes `ax` into a clock: "
                                "angle 0 at the top, clockwise, spokes every `step` hours (at angle 2π·h/hours) "
                                "labelled like '0h', '3h', …. It returns nothing.",
         "def clock_face(ax, hours=24, step=3):\n    ax.set_theta_zero_location('N')\n    ax.set_theta_direction(-1)\n"
         "    hs = range(0, hours, step)\n    ax.set_xticks([2 * np.pi * h / hours for h in hs], labels=[f'{h}h' for h in hs])",
         {"require": ["set_theta_zero_location", "set_theta_direction"],
          "post": f"{pol}clock_face(_a)\nclock_face(_b, hours=12, step=2)\n"
                  "[(fc.r(a.get_theta_offset(), 6), a.get_theta_direction(), fc.ticks(a, 'x', nd=4)) for a in (_a, _b)]"}),
    ]
    return catalog(rng, items, code, "")


@template(M, "hard")
def debug_polar_3d(rng):
    code, desc = _rose(rng)
    s = seeded(rng)
    nx, ny = rng.choice([(6, 4), (8, 5), (10, 7)])
    m = rng.choice([4, 5, 6])
    items = [
        ("subplot_kw", "This crashes with *AttributeError: 'Axes' object has no attribute 'set_theta_zero_location'*:"
                       "\n```python\nfig, ax = plt.subplots()\nax.plot(theta, r)\nax.set_theta_zero_location('N')\n```\n"
                       "Fix it (an ordinary Axes has no angles).",
         "fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})\nax.plot(theta, r)\nax.set_theta_zero_location('N')",
         {"post": f"{POLAR}, {LINES}, fc.r(fc.ax().get_theta_offset(), 6)", "require": ["subplot_kw|projection"]}),
        ("projection", f"This crashes with *AttributeError: 'Axes' object has no attribute 'plot_surface'*:\n```python\n"
                       f"X, Y = np.meshgrid(np.linspace(-1, 1, {nx}), np.linspace(-1, 1, {ny}))\nfig = plt.figure()\n"
                       f"ax = fig.add_subplot(111)\nax.plot_surface(X, Y, X * Y)\n```\nFix it.",
         f"X, Y = np.meshgrid(np.linspace(-1, 1, {nx}), np.linspace(-1, 1, {ny}))\nfig = plt.figure()\n"
         f"ax = fig.add_subplot(111, projection='3d')\nax.plot_surface(X, Y, X * Y)", {"post": SURF}),
        ("append", f"This radar chart has a **gap** between the last metric and the first:\n```python\n"
                   f"angles = np.linspace(0, 2 * np.pi, {m}, endpoint=False)\n"
                   f"fig, ax = plt.subplots(subplot_kw={{'projection': 'polar'}})\nax.plot(angles, vals, 'o-')\n```\n"
                   f"Close the loop.",
         f"angles = np.linspace(0, 2 * np.pi, {m}, endpoint=False)\nfig, ax = plt.subplots(subplot_kw={{'projection': 'polar'}})\n"
         f"ax.plot(np.append(angles, angles[0]), np.append(vals, vals[0]), 'o-')",
         {"setup": f"vals = np.round(np.random.default_rng({s}).uniform(0.3, 1, {m}), 2)", "post": LINES,
          "require": ["append|concatenate|r_"]}),
        ("deg2rad", "This wind-direction plot looks like random scribbles: the directions are in **degrees**, but a "
                    "polar Axes reads radians.\n```python\nfig, ax = plt.subplots(subplot_kw={'projection': 'polar'})\n"
                    "ax.plot(deg, speed, 'o')\n```\nFix it.",
         "fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})\nax.plot(np.deg2rad(deg), speed, 'o')",
         {"setup": f"g = np.random.default_rng({s})\ndeg = g.integers(0, 360, 12)\nspeed = np.round(g.uniform(1, 9, 12), 1)",
          "post": LINES, "require": ["deg2rad|radians|pi"]}),
        ("exp", "This crashes with *ValueError: Wedge sizes 'x' must be non negative values*:\n```python\n"
                "fig, ax = plt.subplots()\nax.pie(logp, labels=labels)\n```\n`logp` holds log-probabilities. Show "
                "the probabilities (`np.exp(logp)`) instead.",
         "fig, ax = plt.subplots()\nax.pie(np.exp(logp), labels=labels)",
         {"setup": f"labels = ['a', 'b', 'c', 'd']\nlogp = np.log(np.random.default_rng({s}).dirichlet(np.ones(4)))",
          "post": f"{PIE}, {TXT}", "require": ["exp"]}),
        ("scatter", "This 3-D scatter comes out **flat** (every point at z = 0):\n```python\nfig = plt.figure()\n"
                    "ax = fig.add_subplot(projection='3d')\nax.scatter(P[:, 0], P[:, 1], c=P[:, 2])\n```\nThe colors "
                    "are right, but the third coordinate must be passed as z too. Fix it.",
         "fig = plt.figure()\nax = fig.add_subplot(projection='3d')\nax.scatter(P[:, 0], P[:, 1], P[:, 2], c=P[:, 2])",
         {"setup": f"P = np.round(np.random.default_rng({s}).normal(size=(25, 3)), 3)",
          "post": f"{SC3}, [fc.r(c.get_array(), 4) for c in fc.ax().collections]"}),
        ("meshgrid", f"This crashes with *ValueError: shape mismatch: objects cannot be broadcast to a single shape*:\n"
                     f"```python\nx = np.linspace(-1, 1, {nx})\ny = np.linspace(-1, 1, {ny})\nZ = x[None, :] ** 2 + "
                     f"y[:, None] ** 2\nfig = plt.figure()\nax = fig.add_subplot(projection='3d')\n"
                     f"ax.plot_surface(x, y, Z)\n```\n`plot_surface` needs 2-D coordinate grids. Fix it.",
         f"x = np.linspace(-1, 1, {nx})\ny = np.linspace(-1, 1, {ny})\nZ = x[None, :] ** 2 + y[:, None] ** 2\n"
         f"X, Y = np.meshgrid(x, y)\nfig = plt.figure()\nax = fig.add_subplot(projection='3d')\nax.plot_surface(X, Y, Z)",
         {"post": SURF, "require": ["meshgrid"]}),
        ("pie", "This crashes with *ValueError: 'labels' must be of length 'x', not 4*:\n```python\nsizes = [40, 25, 20]\n"
                "labels = ['cat', 'dog', 'bird', 'other']\nfig, ax = plt.subplots()\nax.pie(sizes, labels=labels)\n```\n"
                "The counts come from 100 images; 'other' is whatever is left. Add its size and draw the pie.",
         "sizes = [40, 25, 20]\nlabels = ['cat', 'dog', 'bird', 'other']\nsizes = sizes + [100 - sum(sizes)]\n"
         "fig, ax = plt.subplots()\nax.pie(sizes, labels=labels)", {"post": f"{PIE}, {TXT}"}),
        ("view_init", f"This should show the surface from **straight above** (elevation 90°) with azimuth -90°, but "
                      f"it shows a side view:\n```python\nX, Y = np.meshgrid(np.linspace(-1, 1, {nx}), "
                      f"np.linspace(-1, 1, {ny}))\nfig = plt.figure()\nax = fig.add_subplot(projection='3d')\n"
                      f"ax.plot_surface(X, Y, X * Y)\nax.view_init(-90, 90)\n```\n`view_init`'s first argument is the "
                      f"elevation, the second the azimuth. Fix it (keywords make it clear).",
         f"X, Y = np.meshgrid(np.linspace(-1, 1, {nx}), np.linspace(-1, 1, {ny}))\nfig = plt.figure()\n"
         f"ax = fig.add_subplot(projection='3d')\nax.plot_surface(X, Y, X * Y)\nax.view_init(elev=90, azim=-90)",
         {"post": f"{SURF}, [fc.ax().elev, fc.ax().azim]"}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "hard")
def mixed_projection_figures(rng):
    code, desc, sizes, labels = _shares(rng, k=(4, 4))
    mcode, mdesc, m, models = _models(rng, k=2)
    lcode, ldesc, nx, ny = _landscape(rng, n=(12, 20))
    views = rng.choice([[(30, -60), (30, 30), (60, -60), (10, 120)], [(20, 0), (20, 90), (20, 180), (80, 45)]])
    s = seeded(rng)
    k = len(sizes)
    items = [
        ("subplots", "The guide's pie figure: `fig, axs = plt.subplots(1, 2, figsize=(9, 4))`; on the left `sizes` "
                     "labelled, with one-decimal percentages, starting at 90°, clockwise, first wedge exploded by "
                     "0.05; on the right the same sizes and labels as a donut (wedge width 0.4, white edges) titled "
                     "'donut'.",
         f"fig, axs = plt.subplots(1, 2, figsize=(9, 4))\naxs[0].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, "
         f"explode=[0.05] + [0] * {k - 1}, counterclock=False)\n"
         f"axs[1].pie(sizes, labels=labels, wedgeprops={{'width': 0.4, 'edgecolor': 'w'}})\naxs[1].set_title('donut')",
         {"post": _per_axes("(fc.patches(a), fc.texts(a), [p.width for p in a.patches], a.get_title())") + ", fc.size()",
          "require": ["pie", "wedgeprops"]}),
        ("add_subplot", f"{mdesc}. A one-page report with three projections: `fig = plt.figure(figsize=(12, 4))` and "
                        f"(1) `fig.add_subplot(1, 3, 1, projection='polar')`: both models' closed radar lines; "
                        f"(2) `fig.add_subplot(1, 3, 2)`: a pie of `sizes` with `labels`; (3) "
                        f"`fig.add_subplot(1, 3, 3, projection='3d')`: the surface `L` over `W1`, `W2`.",
         f"fig = plt.figure(figsize=(12, 4))\nax1 = fig.add_subplot(1, 3, 1, projection='polar')\n"
         f"angles = np.linspace(0, 2 * np.pi, {m}, endpoint=False)\nth = np.append(angles, angles[0])\n"
         f"for row in scores:\n    ax1.plot(th, np.append(row, row[0]))\nax2 = fig.add_subplot(1, 3, 2)\n"
         f"ax2.pie(sizes, labels=labels)\nax3 = fig.add_subplot(1, 3, 3, projection='3d')\nax3.plot_surface(W1, W2, L)",
         {"setup": f"{mcode}\n{lcode}",
          "post": "[a.name for a in fc.axes()], fc.lines(fc.ax(0)), fc.patches(fc.ax(1)), fc.texts(fc.ax(1)), "
                  "[(c._vec.shape[1], fc.r(c._vec[:3].sum(axis=1), 2)) for c in fc.ax(2).collections]",
          "require": ["add_subplot", "projection"]}),
        ("view_init", f"Show the surface `L` (over `W1`, `W2`) from four cameras on a 2 × 2 grid of 3-D Axes "
                      f"(`plt.subplots(2, 2, subplot_kw={{'projection': '3d'}})`), using the (elev, azim) pairs "
                      f"{views} in reading order, each titled `f'elev={{e}}, azim={{a}}'`.",
         f"fig, axs = plt.subplots(2, 2, subplot_kw={{'projection': '3d'}})\nfor ax, (e, a) in zip(axs.flat, {views}):\n"
         f"    ax.plot_surface(W1, W2, L)\n    ax.view_init(elev=e, azim=a)\n    ax.set_title(f'elev={{e}}, azim={{a}}')",
         {"setup": lcode, "post": _per_axes("(a.name, a.elev, a.azim, a.get_title(), len(a.collections))"),
          "require": ["view_init", "subplot_kw"]}),
        ("plot_wireframe", "The guide's 3-D figure: `fig = plt.figure(figsize=(10, 4))`. Left "
                           "(`fig.add_subplot(1, 2, 1, projection='3d')`): `L` as a surface in 'viridis' with no "
                           "edges and opacity 0.9, its contours projected on the floor z = `L.min() - 1`, axes "
                           "labelled 'w1', 'w2', 'loss', view (elev 30, azim -60). Right: the points `p` as a 3-D "
                           "scatter colored by their z (`c=p[:, 2]`, 'plasma') plus a wireframe of `0.2 * L` with "
                           "strides 5 and line width 0.5.",
         "fig = plt.figure(figsize=(10, 4))\nax3 = fig.add_subplot(1, 2, 1, projection='3d')\n"
         "ax3.plot_surface(W1, W2, L, cmap='viridis', edgecolor='none', alpha=0.9)\n"
         "ax3.contour(W1, W2, L, zdir='z', offset=L.min() - 1, cmap='viridis')\nax3.set(xlabel='w1', ylabel='w2', zlabel='loss')\n"
         "ax3.view_init(elev=30, azim=-60)\nax4 = fig.add_subplot(1, 2, 2, projection='3d')\n"
         "ax4.scatter(p[:, 0], p[:, 1], p[:, 2], c=p[:, 2], cmap='plasma')\n"
         "ax4.plot_wireframe(W1, W2, L * 0.2, rstride=5, cstride=5, linewidth=0.5)",
         {"setup": f"{lcode}\np = np.round(np.random.default_rng({s}).normal(size=(50, 3)), 3)",
          "post": f"fc.size(), {SURF}, {CONT3}, {LAB3}, [fc.ax(0).elev, fc.ax(0).azim], [c.get_alpha() for c in fc.ax(0).collections], "
                  f"plt.sca(fc.ax(1)), {SC3}, {WIRE}",
          "require": ["plot_surface", "plot_wireframe", "view_init"]}),
        ("set_rmax", "Small multiples: `angles` are update directions (radians) and `cls` their class (0, 1, 2). Draw "
                     "one polar Axes per class in a row, each with a rose diagram of that class's angles (8 bins over "
                     "0 … 2π, bars at the bin centers, width 2π/8) titled 'class c', and give all three the same "
                     "radius range: `set_rmax` = the largest count over all classes.",
         "counts = [np.histogram(angles[cls == c], bins=8, range=(0, 2 * np.pi))[0] for c in range(3)]\n"
         "centers = (np.arange(8) + 0.5) * 2 * np.pi / 8\ntop = max(c.max() for c in counts)\n"
         "fig, axs = plt.subplots(1, 3, subplot_kw={'projection': 'polar'})\nfor c, a in enumerate(axs):\n"
         "    a.bar(centers, counts[c], width=2 * np.pi / 8)\n    a.set_title(f'class {c}')\n    a.set_rmax(top)",
         {"setup": f"g = np.random.default_rng({s})\nangles = np.round(np.mod(g.normal(2, 1.2, 90), 2 * np.pi), 4)\n"
                   f"cls = g.integers(0, 3, 90)",
          "post": _per_axes("(fc.bars(a, 4), a.get_title(), fc.r(a.get_rmax(), 4))"), "require": ["histogram", "set_rmax"]}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "hard")
def geometry_3d(rng):
    R, r_ = rng.choice([(2, 0.5), (3, 1), (2.5, 0.8)])
    nu, nv = rng.choice([(20, 10), (24, 12), (16, 8)])
    s = seeded(rng)
    a = rng.choice([30, 45, 90, 120])
    w = rng.choice([(1.0, -2.0, 1.5), (0.5, 1.0, -2.0), (2.0, 1.0, 1.0)])
    b = rng.choice([0.5, -1.0, 0.0])
    new = "fig = plt.figure()\nax = fig.add_subplot(projection='3d')\n"
    uv = f"u = np.linspace(0, 2 * np.pi, {nu})\nv = np.linspace(0, np.pi, {nv})\n"
    items = [
        ("outer", f"Draw the unit sphere as a wireframe: `u = np.linspace(0, 2π, {nu})`, `v = np.linspace(0, π, "
                  f"{nv})`, x = `np.outer(np.cos(u), np.sin(v))`, y = `np.outer(np.sin(u), np.sin(v))`, z = "
                  f"`np.outer(np.ones_like(u), np.cos(v))`.",
         f"{uv}x = np.outer(np.cos(u), np.sin(v))\ny = np.outer(np.sin(u), np.sin(v))\nz = np.outer(np.ones_like(u), np.cos(v))\n"
         f"{new}ax.plot_wireframe(x, y, z)", {"post": WIRE, "require": ["outer", "plot_wireframe"]}),
        ("plot_surface", f"Draw a torus as a surface: with `u, v = np.meshgrid(np.linspace(0, 2π, {nu}), "
                         f"np.linspace(0, 2π, {nv}))`, x = ({R} + {r_}·cos v)·cos u, y = ({R} + {r_}·cos v)·sin u, "
                         f"z = {r_}·sin v; equal box proportions `ax.set_box_aspect((1, 1, {r_ / R:g}))`.",
         f"u, v = np.meshgrid(np.linspace(0, 2 * np.pi, {nu}), np.linspace(0, 2 * np.pi, {nv}))\n"
         f"x = ({R} + {r_} * np.cos(v)) * np.cos(u)\ny = ({R} + {r_} * np.cos(v)) * np.sin(u)\nz = {r_} * np.sin(v)\n"
         f"{new}ax.plot_surface(x, y, z)\nax.set_box_aspect((1, 1, {r_ / R:g}))",
         {"post": f"{SURF}, fc.r(fc.ax().get_box_aspect(), 4)", "require": ["plot_surface", "set_box_aspect"]}),
        ("scatter", f"A linear classifier in 3-D with weights `w = {list(w)}` and bias {b} separates the points `P` "
                    f"(labels `y`). Draw its decision plane w·p + b = 0 as a surface z = -(b + w0·x + w1·y) / w2 over "
                    f"`np.meshgrid(np.linspace(-2, 2, 10), np.linspace(-2, 2, 10))` with opacity 0.3, and the points "
                    f"as a 3-D scatter colored by `y`.",
         f"w = np.array({list(w)})\nX, Y = np.meshgrid(np.linspace(-2, 2, 10), np.linspace(-2, 2, 10))\n"
         f"Z = -({b} + w[0] * X + w[1] * Y) / w[2]\n{new}ax.plot_surface(X, Y, Z, alpha=0.3)\n"
         f"ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=y)",
         {"setup": f"g = np.random.default_rng({s})\nP = np.round(g.normal(size=(30, 3)), 3)\n"
                   f"y = (P @ np.array({list(w)}) + {b} > 0).astype(int)",
          "post": f"{SURF}, {SC3}, [fc.r(c.get_array(), 4) for c in fc.ax().collections if c.get_array() is not None]",
          "require": ["plot_surface", "scatter"]}),
        ("quiver", "Draw the gradient of f(x, y, z) = x² + y² + z² (that is (2x, 2y, 2z)) as 3-D arrows at the "
                   "points of the grid `np.meshgrid(g1, g1, g1)` with `g1 = np.linspace(-1, 1, 3)`, all the same "
                   "length: `ax.quiver(x, y, z, u, v, w, length=0.3, normalize=True)`.",
         f"g1 = np.linspace(-1, 1, 3)\nx, y, z = np.meshgrid(g1, g1, g1)\n{new}"
         f"ax.quiver(x, y, z, 2 * x, 2 * y, 2 * z, length=0.3, normalize=True)",
         {"post": "[(len(c._segments3d), fc.r(np.abs(np.asarray(c._segments3d)).sum(axis=(0, 1)), 3)) for c in fc.ax().collections]",
          "require": ["quiver", "normalize"]}),
        ("contourf", "Filled contours work on polar Axes too. With `T, Rr = np.meshgrid(np.linspace(0, 2π, 24), "
                     "np.linspace(0, 1, 10))`, draw the dipole field `Rr * np.cos(T)` with `ax.contourf(T, Rr, ..., "
                     "levels=8)` on a polar Axes, and add a colorbar.",
         "T, Rr = np.meshgrid(np.linspace(0, 2 * np.pi, 24), np.linspace(0, 1, 10))\n"
         "fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})\ncf = ax.contourf(T, Rr, Rr * np.cos(T), levels=8)\n"
         "fig.colorbar(cf, ax=ax)",
         {"post": "fc.ax().name, [fc.r(c.levels, 4) for c in fc.ax().collections], len(fc.colorbars())",
          "require": ["contourf", "colorbar"]}),
        ("cos", f"Rotate the point cloud `P` by {a}° about the z axis (rotation matrix [[cos, -sin, 0], [sin, cos, 0], "
                f"[0, 0, 1]], applied as `P @ Rm.T`) and draw the original (gray) and the rotated points (red) as two "
                f"3-D scatters on one Axes.",
         f"t = np.deg2rad({a})\nRm = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])\n"
         f"Q = P @ Rm.T\n{new}ax.scatter(P[:, 0], P[:, 1], P[:, 2], color='gray')\nax.scatter(Q[:, 0], Q[:, 1], Q[:, 2], color='red')",
         {"setup": f"P = np.round(np.random.default_rng({s}).normal(size=(20, 3)), 3)",
          "post": f"{SC3}, [fc.hexcolor(c.get_facecolor()) for c in fc.ax().collections]", "require": ["cos", "sin"]}),
        ("plot", "Draw a trefoil knot as a 3-D line: `t = np.linspace(0, 2 * np.pi, 300)`, x = sin t + 2·sin 2t, "
                 "y = cos t − 2·cos 2t, z = −sin 3t, and label the axes 'x', 'y', 'z'.",
         f"t = np.linspace(0, 2 * np.pi, 300)\n{new}ax.plot(np.sin(t) + 2 * np.sin(2 * t), np.cos(t) - 2 * np.cos(2 * t), "
         f"-np.sin(3 * t))\nax.set(xlabel='x', ylabel='y', zlabel='z')",
         {"post": f"{L3}, {LAB3}", "require": ["plot"]}),
    ]
    return catalog(rng, items, "", "")


# ================================================================= challenge

@template(M, "challenge")
def pie_wedge_lookup(rng):
    n, q, s = rng.randint(4, 7), rng.randint(5, 8), seeded(rng)
    report = rng.choice(["wedge", "click"])
    start, ccw = rng.choice([0, 90, 30, 200]), rng.choice([True, False])
    query = (f"angles = g.integers(-720, 720, {q})" if report == "wedge"
             else f"xs, ys = np.round(g.uniform(-1.2, 1.2, (2, {q})), 3)")
    setup = (f"g = np.random.default_rng({s})\nsizes = g.integers(0, 6, {n})\nsizes[0] += 1\nstart, counterclock = {start}, "
             f"{ccw}\n{query}\nreport = {report!r}")
    sol = ("T = sizes.sum()\nbounds = 360 * np.cumsum(sizes)     # wedge i ends bounds[i] / T degrees after the start\n"
           "sign = 1 if counterclock else -1\n")
    sol += {"wedge": "d = np.mod(sign * (angles - start), 360)   # whole degrees travelled from the start: 0 … 359\n"
                     "result = np.searchsorted(bounds, d * T, side='right')   # exact: compare d·T with 360·C",
            "click": "a = np.degrees(np.arctan2(ys, xs))\nd = np.mod(sign * (a - start), 360)\nd[d >= 360] -= 360\n"
                     "idx = np.searchsorted(bounds / T, d, side='right')\n"
                     "result = np.where(np.hypot(xs, ys) <= 1, idx, -1)"}[report]
    big = ("angles = g.integers(-10 ** 6, 10 ** 6, 200000)" if report == "wedge"
           else "xs, ys = np.round(g.uniform(-1.2, 1.2, (2, 200000)), 3)")
    tests = [f"g = np.random.default_rng({s + i})\nsizes = g.integers(0, {hi}, 200000)\nsizes[0] += 1\n"
             f"start, counterclock = {st}, {cc}\n{big}\nreport = {report!r}"
             for i, (hi, st, cc) in ((1, (1000, 90, False)), (2, (3, 17, True)))]
    want = {"wedge": "for every angle in `angles` (whole degrees, any integer, measured counterclockwise from the "
                     "positive x-axis like Matplotlib's angles), the index of the wedge that contains that direction",
            "click": "for every click `(xs[j], ys[j])` on the figure (the pie has radius 1 and its center at the "
                     "origin), the index of the wedge under it, or -1 if the click is outside the circle (a click "
                     "exactly on the rim is inside)"}[report]
    return challenge(
        "Which wedge was hit?",
        "`ax.pie(sizes, startangle=start, counterclock=counterclock)` gives wedge i the fraction sizes[i] / "
        "sum(sizes) of the circle. The first wedge starts at the angle `start` (degrees) and the wedges follow one "
        "another counterclockwise, or clockwise when `counterclock` is False; a wedge of size 0 covers nothing. A "
        "direction exactly on the border between two wedges belongs to the one that **starts** there (the later in "
        f"drawing order). Return {want}, as an int array. (`report` = {report!r}.)",
        setup, sol, tests, "len(sizes), number of queries ≤ 2·10^5, sizes are integers ≤ 1000",
        speed="O((n + q) log n): cumulative sums and a binary search per query (in integers where you can, so "
              "borders are exact)", answer_desc=" (an int array)")


@template(M, "challenge")
def densest_sector(rng):
    n, s = rng.randint(6, 10), seeded(rng)
    report = rng.choice(["count", "cover", "gap"])
    w, k = rng.choice([3000, 4500, 6000, 9000]), rng.randint(2, 4)
    param = {"count": f"w = {w}", "cover": f"k = {k}", "gap": ""}[report]
    setup = (f"g = np.random.default_rng({s})\na = np.sort(g.integers(0, 36000, {n}))[::-1].copy()\n"
             f"{param + chr(10) if param else ''}report = {report!r}")
    sol = "s = np.sort(a)\nn = len(s)\next = np.concatenate([s, s + 36000])   # unroll the circle once\n"
    sol += {"count": "ends = np.searchsorted(ext, s + w, side='right')\nresult = int((ends - np.arange(n)).max())",
            "cover": "result = int((ext[k - 1:k - 1 + n] - ext[:n]).min())",
            "gap": "result = int((ext[1:n + 1] - ext[:n]).max())"}[report]
    bigp = {"count": "w = 777", "cover": "k = 5000", "gap": ""}[report]
    tests = [f"g = np.random.default_rng({s + i})\na = g.integers(0, 36000, {nn})\n{bigp + chr(10) if bigp else ''}"
             f"report = {report!r}" for i, nn in ((1, 200000), (2, 150000))]
    want = {"count": "the largest number of points a **closed** sector of width `w` can contain (it may start "
                     "anywhere and wrap past 0°)",
            "cover": "the smallest width (in hundredths of a degree) of a closed sector that contains at least "
                     "`k` points",
            "gap": "the width of the widest empty sector: the largest angle between two angularly consecutive "
                   "points (going around the circle, including the step from the last point back to the first)"}[report]
    return challenge(
        "The busiest sector of a polar plot",
        "A polar scatter shows `n` events by direction; `a` holds their angles in hundredths of a degree "
        "(0 ≤ a < 36000, so 36000 means a full turn; several events may share an angle). Before zooming the polar "
        f"Axes onto one sector (`set_thetamin` / `set_thetamax`), you want {want}. Return an int. "
        f"(`report` = {report!r}.)", setup, sol, tests,
        "n ≤ 2·10^5, 1 ≤ k ≤ n, 0 ≤ w < 36000",
        speed="O(n log n): sort, unroll the circle into a doubled array, then two pointers / binary search",
        answer_desc=" (an int)")


@template(M, "challenge")
def polar_zoom_counts(rng):
    n, q, s = rng.randint(10, 16), rng.randint(3, 5), seeded(rng)
    report = rng.choice(["count", "mean"])
    setup = (f"g = np.random.default_rng({s})\ntheta = g.integers(0, 360, {n})\nr = g.integers(0, 10, {n})\n"
             f"t1, t2 = g.integers(0, 360, (2, {q}))\nr1 = g.integers(0, 6, {q})\nr2 = r1 + g.integers(0, 6, {q})\n"
             f"report = {report!r}")
    sol = ("order = np.lexsort((r, theta))            # by angle, then radius\nth, rr = theta[order], r[order]\n"
           "starts = np.searchsorted(th, np.arange(361))   # the points at angle d are th[starts[d]:starts[d + 1]]\n"
           "csum = np.concatenate([[0], np.cumsum(rr)])\ncount = np.zeros(len(t1))\ntotal = np.zeros(len(t1))\n"
           "for d in range(360):\n    lo, hi = starts[d], starts[d + 1]\n    if lo == hi:\n        continue\n"
           "    inside = np.where(t1 <= t2, (t1 <= d) & (d <= t2), (d >= t1) | (d <= t2))\n"
           "    a = lo + np.searchsorted(rr[lo:hi], r1, side='left')\n    b = lo + np.searchsorted(rr[lo:hi], r2, side='right')\n"
           "    count += inside * (b - a)\n    total += inside * (csum[b] - csum[a])\n")
    sol += {"count": "result = count.astype(int)",
            "mean": "result = np.where(count > 0, total / np.maximum(count, 1), 0.0)"}[report]
    tests = [f"g = np.random.default_rng({s + i})\ntheta = g.integers(0, 360, 200000)\nr = g.integers(0, {R}, 200000)\n"
             f"t1, t2 = g.integers(0, 360, (2, 20000))\nr1 = g.integers(0, {R}, 20000)\n"
             f"r2 = r1 + g.integers(0, {R}, 20000)\nreport = {report!r}" for i, R in ((1, 10 ** 6), (2, 50))]
    want = {"count": "how many points stay visible (an int array, one per query)",
            "mean": "the mean radius of the points that stay visible (a float array, one per query; 0.0 when none "
                    "is visible)"}[report]
    return challenge(
        "Zooming into a polar scatter",
        "A polar scatter has points at angles `theta` (whole degrees, 0 … 359) and radii `r` (integers). Query `j` "
        "zooms the Axes to the angles `t1[j]` … `t2[j]` and the radii `r1[j]` … `r2[j]` (all bounds inclusive; when "
        "`t1[j] > t2[j]` the angle range wraps through 0°, e.g. 350 … 10). For every query, return " + want +
        f". (`report` = {report!r}.)", setup, sol, tests,
        "n ≤ 2·10^5 points, q ≤ 2·10^4 queries, radii ≤ 2·10^6",
        speed="about O(360 · q · log n) or better: there are only 360 distinct angles, so group the points by angle "
              "with sorted radii (and prefix sums) and binary-search each group", tol=1e-9,
        answer_desc=" (an array)")
