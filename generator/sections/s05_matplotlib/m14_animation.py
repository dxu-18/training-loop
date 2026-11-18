"""5.14 Animation & live-updating plots (FuncAnimation / ArtistAnimation, PillowWriter GIFs, mutating artists,
interactive live plots with relim / autoscale_view / draw_idle / pause)."""

from ._common import P, catalog, challenge, curves, lit, seeded, template
from ... import themes as T

M = "5.14"

# Animations are graded by calling the learner's update function and reading the artists back, by the animation's
# frame sequence and settings, or by opening the saved GIF with Pillow (frames, size, per-frame duration in ms).
PRE = '''def _gif(path):
    from PIL import Image
    im = Image.open(path)
    d = []
    for i in range(im.n_frames):
        im.seek(i)
        d.append(im.info.get('duration'))
    return [im.n_frames, list(im.size), d]

def _seq(anim):
    import itertools
    out = []
    for f in itertools.islice(anim.new_frame_seq(), 1000):
        out.append(len(f) if isinstance(f, (list, tuple)) else fc.r(f, 4))
    return out

def _call(fn, frames, what):
    out = []
    for k in frames:
        r = fn(k)
        out.append([what(), sorted(type(a).__name__ for a in (r or ()))])
    return out
'''
IMPORT = "Import what you need from `matplotlib.animation` (`FuncAnimation`, `ArtistAnimation`, `PillowWriter`)."
GIF_NOTE = "(GIFs here store each frame's duration in hundredths of a second.)"


def _post(*exprs):
    return PRE + ", ".join(exprs)


def _wave(rng):
    n = rng.choice([30, 40, 60])
    return f"x = np.linspace(0, 2 * np.pi, {n})", n


def _gd(rng):
    """A gradient-descent path on a 2-D quadratic: `loss_fn`, `path` ((T+1) × 2) and a 40 × 40 grid W0, W1."""
    c0, c1 = rng.choice([(1.0, -0.5), (0.5, 0.5), (-1.0, 0.5), (1.5, 0.0)])
    k = rng.choice([3, 5, 8])
    lr = rng.choice([0.05, 0.08, 0.1])
    s0, s1 = rng.choice([(-2.0, 1.5), (2.0, 1.5), (-2.0, -1.5), (0.0, 1.8)])
    T = rng.randint(6, 10)
    code = (f"c0, c1, k, lr = {c0}, {c1}, {k}, {lr}\nloss_fn = lambda w0, w1: (w0 - c0) ** 2 + k * (w1 - c1) ** 2\n"
            f"path = [np.array([{s0}, {s1}])]\nfor _ in range({T}):\n    w = path[-1]\n"
            f"    path.append(w - lr * np.array([2 * (w[0] - c0), 2 * k * (w[1] - c1)]))\n"
            f"path = np.array(path)   # ({T + 1}, 2): the weights after each step\n"
            f"v, w = np.zeros(2), path[0].copy()\nmpath = [w.copy()]\nfor _ in range({T}):\n"
            f"    v = 0.6 * v - lr * np.array([2 * (w[0] - c0), 2 * k * (w[1] - c1)])\n    w = w + v\n    mpath.append(w.copy())\n"
            f"mpath = np.array(mpath)   # the same descent with momentum 0.6\n"
            f"W0, W1 = np.meshgrid(np.linspace(-2.5, 2.5, 40), np.linspace(-2, 2, 40))")
    return code, T + 1, (c0, c1, k, lr)


def _boundary(rng):
    """Two classes `X`, `y` and logistic-regression weights `W` (E × 3: bias, w1, w2) after each epoch."""
    E = rng.randint(5, 8)
    s = seeded(rng)
    lr = rng.choice([0.5, 1.0])
    code = (f"g = np.random.default_rng({s})\nX = np.round(np.vstack([g.normal([-1, -1], 0.8, (25, 2)), "
            f"g.normal([1.5, 1], 0.8, (25, 2))]), 3)\ny = np.repeat([0, 1], 25)\nXb = np.column_stack([np.ones(50), X])\n"
            f"w, W = np.zeros(3), []\nfor epoch in range({E}):\n    p = 1 / (1 + np.exp(-Xb @ w))\n"
            f"    w = w - {lr} * Xb.T @ (p - y) / 50\n    W.append(w.copy())\n"
            f"W = np.array(W)   # ({E}, 3): bias, w1, w2 after each epoch\nxs = np.linspace(-3, 3, 20)")
    return code, E


def _losses(rng):
    N = rng.choice([50, 60, 80, 100])
    s = seeded(rng)
    tau = rng.choice([10, 20, 30])
    return (f"g = np.random.default_rng({s})\nlosses = np.round(2 * np.exp(-np.arange({N}) / {tau}) + 0.1 * g.random({N}), 4)"
            f"   # one loss per training step", N)


# ================================================================= lite

@template(M, "lite")
def animation_objects(rng):
    code, n = _wave(rng)
    N = rng.randint(5, 8)
    I = rng.choice([40, 50, 80, 100, 200])
    fps = rng.choice([5, 10, 20, 25])
    dpi = rng.choice([20, 30, 40])
    base = (f"{code}\nfig, ax = plt.subplots(figsize=(2, 1.5))\nline, = ax.plot(x, np.sin(x))\n"
            f"def update(frame):\n    line.set_ydata(np.sin(x + frame / 4))\n    return (line,)")
    anim = f"\nfrom matplotlib.animation import FuncAnimation\nanim = FuncAnimation(fig, update, frames={N}, interval={I})"
    items = [
        ("FuncAnimation", f"Animate the wave: `anim = FuncAnimation(fig, update, frames={N}, interval={I})` ({N} frames, "
                          f"{I} ms apart).",
         f"from matplotlib.animation import FuncAnimation\nanim = FuncAnimation(fig, update, frames={N}, interval={I})",
         {"post": _post("_seq(anim)", "anim._interval")}),
        ("FuncAnimation", f"Create `anim` with {N} frames and **blitting** on (`blit=True`: only the artists `update` "
                          f"returns are redrawn).",
         f"from matplotlib.animation import FuncAnimation\nanim = FuncAnimation(fig, update, frames={N}, blit=True)",
         {"post": _post("_seq(anim)", "anim._blit"), "require": ["FuncAnimation", "blit"]}),
        ("FuncAnimation", f"Create `anim` with {N} frames that plays **once** (`repeat=False`) at {I} ms per frame.",
         f"from matplotlib.animation import FuncAnimation\nanim = FuncAnimation(fig, update, frames={N}, interval={I}, repeat=False)",
         {"post": _post("_seq(anim)", "anim._interval", "anim._repeat"), "require": ["FuncAnimation", "repeat"]}),
        ("FuncAnimation", f"`frames` can be any sequence: each value is passed to `update`. Create `anim` whose frames "
                          f"are `np.linspace(0, 8, {N})`.",
         f"from matplotlib.animation import FuncAnimation\nanim = FuncAnimation(fig, update, frames=np.linspace(0, 8, {N}))",
         {"post": _post("_seq(anim)")}),
        ("init_func", f"`init` (given) blanks the line before the first frame. Create `anim` with {N} frames, "
                      f"`init_func=init` and `blit=True`.",
         f"from matplotlib.animation import FuncAnimation\nanim = FuncAnimation(fig, update, frames={N}, init_func=init, blit=True)",
         {"post": _post("_seq(anim)", "anim._init_func is init", "anim._blit"),
          "setup": "def init():\n    line.set_ydata(np.zeros_like(x))\n    return (line,)", "require": ["init_func"]}),
        (".save", f"`anim` ({N} frames) exists. Save it as `wave.gif` with `writer=PillowWriter(fps={fps})` and "
                  f"`dpi={dpi}`. {GIF_NOTE}",
         f"from matplotlib.animation import PillowWriter\nanim.save('wave.gif', writer=PillowWriter(fps={fps}), dpi={dpi})",
         {"post": _post("_gif('wave.gif')"), "setup": anim, "require": ["save", "PillowWriter"]}),
        ("PillowWriter", f"Make a writer for {fps} fps GIFs and return `[writer.fps, PillowWriter.isAvailable()]` (Pillow "
                         f"ships with Matplotlib, so no ffmpeg is needed).",
         f"from matplotlib.animation import PillowWriter\nwriter = PillowWriter(fps={fps})\n"
         f"result = [writer.fps, PillowWriter.isAvailable()]"),
        ("ArtistAnimation", f"`frames` (given) is a list of {N} pre-drawn frames, each a list of artists. Turn it into "
                            f"`anim = ArtistAnimation(fig, frames, interval={I})`.",
         f"from matplotlib.animation import ArtistAnimation\nanim = ArtistAnimation(fig, frames, interval={I})",
         {"post": _post("_seq(anim)", "anim._interval"),
          "setup": f"frames = [[ax.imshow(np.random.default_rng(i).random((6, 6)), animated=True)] for i in range({N})]"}),
        (".to_jshtml", f"In Jupyter you'd display `anim` ({N} frames) with `HTML(anim.to_jshtml())`. Call "
                       f"`anim.to_jshtml()` and return how many PNG frames the HTML embeds "
                       f"(`html.count('data:image/png')`).",
         "html = anim.to_jshtml()\nresult = html.count('data:image/png')",
         {"setup": anim + "\nfig.set_dpi(30)"}),
    ]
    return catalog(rng, items, base, f"`fig, ax` show a sine wave `line` over `x`, and `update(frame)` shifts it. {IMPORT}")


@template(M, "lite")
def artist_mutation(rng):
    code, n = _wave(rng)
    k = rng.randint(5, n - 5)
    e = rng.randint(1, 40)
    f = rng.choice([2, 3])
    s = seeded(rng)
    setup = (f"{code}\nfig, ax = plt.subplots(figsize=(3, 2))\nline, = ax.plot(x, np.sin(x))\n"
             f"g = np.random.default_rng({s})")
    items = [
        (".set_ydata", f"Mutate, don't re-plot: change the existing `line` to show `np.cos({f} * x)` with "
                       f"`line.set_ydata`.", f"line.set_ydata(np.cos({f} * x))", {"post": "fc.lines(fc.ax())"}),
        (".set_data", f"Make `line` show only the first {k} points of the wave: `line.set_data(x[:{k}], np.sin(x[:{k}]))`.",
         f"line.set_data(x[:{k}], np.sin(x[:{k}]))", {"post": "fc.lines(fc.ax())"}),
        (".set_xdata", "Shift the existing `line` right by 1 with `line.set_xdata(x + 1)`.", "line.set_xdata(x + 1)",
         {"post": "fc.lines(fc.ax())"}),
        (".set_offsets", f"`sc` is a scatter of 5 points. Move them to `pts` (5 × 2) with `sc.set_offsets`.",
         "sc.set_offsets(pts)",
         {"post": "fc.scatter(fc.ax())", "setup": "sc = ax.scatter(np.zeros(5), np.zeros(5))\npts = np.round(g.random((5, 2)), 3)"}),
        (".set_data", "`im` shows a 6 × 6 image on a second figure. Swap in the new frame `new` with `im.set_data(new)`.",
         "im.set_data(new)",
         {"post": "fc.images(fc.ax(), data=True)",
          "setup": "fig2, ax2 = plt.subplots()\nim = ax2.imshow(np.zeros((6, 6)), vmin=0, vmax=1)\nnew = np.round(g.random((6, 6)), 3)"}),
        (".set_text", f"`txt` is an empty text in the corner. Make it read 'epoch {e}' with `txt.set_text`.",
         f"txt.set_text('epoch {e}')",
         {"post": "fc.texts()", "setup": "txt = ax.text(0.02, 0.9, '', transform=ax.transAxes)"}),
        (".relim", f"After `line.set_data(x, {f + 3} * np.sin(x))` the axes keep their old limits. Update the line, then "
                   f"recompute the data limits with `ax.relim()` and rescale with `ax.autoscale_view()`.",
         f"line.set_data(x, {f + 3} * np.sin(x))\nax.relim()\nax.autoscale_view()",
         {"post": "fc.lines(fc.ax()), fc.limits()", "require": ["relim", "autoscale_view"]}),
        (".set_ylim", "In an animation, fix the y-axis once so it doesn't jump between frames: set it to -1.2 … 1.2.",
         "ax.set_ylim(-1.2, 1.2)", {"post": "fc.limits()['ylim']"}),
        ("plt.ion", "Turn pyplot's interactive mode on with `plt.ion()`, record `plt.isinteractive()`, switch it off again "
                    "with `plt.ioff()` and record it again; return both as a list.",
         "plt.ion()\na = plt.isinteractive()\nplt.ioff()\nresult = [a, plt.isinteractive()]", {"require": ["ion", "ioff"]}),
        (".draw_idle", f"Change the line with `line.set_ydata(np.cos(x))`, then ask the canvas to redraw with "
                       f"`fig.canvas.draw_idle()`. Return `fig.stale` (False once the figure is drawn).",
         "line.set_ydata(np.cos(x))\nfig.canvas.draw_idle()\nresult = fig.stale", {"require": ["draw_idle"]}),
        ("plt.pause", f"Change the line with `line.set_ydata(-np.sin(x))` and let pyplot draw and process events for a "
                      f"moment with `plt.pause(0.001)`. Return `[fig.stale, line.get_ydata()[{k}]]`.",
         f"line.set_ydata(-np.sin(x))\nplt.pause(0.001)\nresult = [fig.stale, line.get_ydata()[{k}]]", {"require": ["pause"]}),
        (".set_height", "`bars = ax.bar(range(4), [1, 1, 1, 1])` exist on a new figure. Set their heights to `h` "
                        "(given) one by one with `set_height`.",
         "for b, v in zip(bars, h):\n    b.set_height(v)",
         {"post": "fc.heights(fc.ax())", "setup": "fig3, ax3 = plt.subplots()\nbars = ax3.bar(range(4), [1, 1, 1, 1])\n"
                                                    "h = np.round(g.random(4), 3)"}),
    ]
    return catalog(rng, items, setup, f"`fig, ax` hold `line`, a sine wave over `x` ({n} points). Animations change "
                                      f"artists in place like this.")


# ================================================================= easy

@template(M, "easy")
def update_functions(rng):
    code, desc, n = curves(rng)
    s = seeded(rng)
    W = rng.randint(5, 9)
    ks = sorted(rng.sample(range(n), 3))
    setup = (code + f"\ng2 = np.random.default_rng({s})\ntraj = np.round(g2.normal(size=(6, 8, 2)).cumsum(axis=0), 3)   # 6 steps × 8 points\n"
             f"probs = np.round(g2.dirichlet(np.ones(4), 6), 3)   # 6 steps × 4 classes\n"
             f"maps = np.round(g2.random((6, 5, 5)), 3)   # 6 frames of a 5 × 5 activation map")
    L = "fc.lines(fc.ax())"
    items = [
        ("set_data", "Grow the training curve: create `fig, ax`, an empty line `line, = ax.plot([], [])`, fix the limits "
                     "(x 0 … `epochs[-1] + 1`, y 0 … `train_loss.max() * 1.1`), and write `update(i)` that shows the "
                     "first i + 1 points and returns `(line,)`.",
         "fig, ax = plt.subplots()\nline, = ax.plot([], [])\nax.set_xlim(0, epochs[-1] + 1)\nax.set_ylim(0, train_loss.max() * 1.1)\n"
         "def update(i):\n    line.set_data(epochs[:i + 1], train_loss[:i + 1])\n    return (line,)",
         {"post": _post(f"_call(update, {ks}, lambda: {L})", "fc.limits()"), "require": ["set_data"]}),
        ("set_data", "A dot that walks along the validation curve: draw `val_loss` against `epochs`, then `dot, = "
                     "ax.plot([], [], 'ro')`; `update(i)` puts the dot on point i (`set_data` wants sequences: "
                     "`[x]`, `[y]`) and returns `(dot,)`.",
         "fig, ax = plt.subplots()\nax.plot(epochs, val_loss)\ndot, = ax.plot([], [], 'ro')\n"
         "def update(i):\n    dot.set_data([epochs[i]], [val_loss[i]])\n    return (dot,)",
         {"post": _post(f"_call(update, {ks}, lambda: {L})")}),
        ("set_data", f"A sliding window: create `fig, ax` with `line, = ax.plot([], [])` and write `update(i)` that shows "
                     f"`train_loss[i:i + {W}]` against `epochs[i:i + {W}]`, moves the x-axis to `epochs[i]` … "
                     f"`epochs[i] + {W}`, and returns `(line,)`.",
         f"fig, ax = plt.subplots()\nline, = ax.plot([], [])\ndef update(i):\n    line.set_data(epochs[i:i + {W}], train_loss[i:i + {W}])\n"
         f"    ax.set_xlim(epochs[i], epochs[i] + {W})\n    return (line,)",
         {"post": _post(f"_call(update, {[k for k in ks if k < n - W] or [0]}, lambda: [{L}, fc.limits()['xlim']])")}),
        ("set_offsets", "Animate an embedding: scatter the 8 points of `traj[0]` (`sc = ax.scatter(...)`) and write "
                        "`update(t)` that moves them to `traj[t]` with `sc.set_offsets` and returns `(sc,)`.",
         "fig, ax = plt.subplots()\nsc = ax.scatter(traj[0][:, 0], traj[0][:, 1])\ndef update(t):\n    sc.set_offsets(traj[t])\n"
         "    return (sc,)", {"post": _post("_call(update, [1, 3, 5], lambda: fc.scatter())")}),
        ("set_height", "Animate class probabilities: `bars = ax.bar(range(4), probs[0])`; `update(t)` sets each bar's "
                       "height to `probs[t]` and returns the bars (`bars.patches`).",
         "fig, ax = plt.subplots()\nbars = ax.bar(range(4), probs[0])\ndef update(t):\n    for b, h in zip(bars, probs[t]):\n"
         "        b.set_height(h)\n    return bars.patches", {"post": _post("_call(update, [2, 5], lambda: fc.heights())")}),
        ("set_text", "Frame counter: plot `train_loss`, add `txt = ax.text(0.98, 0.95, '', transform=ax.transAxes, "
                     "ha='right')`, and write `update(i)` that sets it to `f'epoch {epochs[i]}'` and returns `(txt,)`.",
         "fig, ax = plt.subplots()\nax.plot(epochs, train_loss)\ntxt = ax.text(0.98, 0.95, '', transform=ax.transAxes, ha='right')\n"
         "def update(i):\n    txt.set_text(f'epoch {epochs[i]}')\n    return (txt,)",
         {"post": _post(f"_call(update, {ks}, lambda: fc.texts())")}),
        ("set_data", "Two artists per frame: a growing `train_loss` line **and** a red dot on its newest point. "
                     "`update(i)` updates both and returns `(line, dot)`. Fix the limits to x 0 … `epochs[-1] + 1`, "
                     "y 0 … `train_loss.max() * 1.1`.",
         "fig, ax = plt.subplots()\nline, = ax.plot([], [])\ndot, = ax.plot([], [], 'ro')\nax.set_xlim(0, epochs[-1] + 1)\n"
         "ax.set_ylim(0, train_loss.max() * 1.1)\ndef update(i):\n    line.set_data(epochs[:i + 1], train_loss[:i + 1])\n"
         "    dot.set_data([epochs[i]], [train_loss[i]])\n    return (line, dot)",
         {"post": _post(f"_call(update, {ks}, lambda: fc.lines(fc.ax(), 'marker'))")}),
        ("set_title", "Without blitting the whole figure is redrawn, so `update` may change anything: plot `val_loss`, "
                      "add `vline = ax.axvline(epochs[0], color='gray')`, and write `update(i)` that moves the vertical "
                      "line to `epochs[i]` (`vline.set_xdata([x, x])`) and sets the title to `f'val loss "
                      "{val_loss[i]:.3f}'`.",
         "fig, ax = plt.subplots()\nax.plot(epochs, val_loss)\nvline = ax.axvline(epochs[0], color='gray')\n"
         "def update(i):\n    vline.set_xdata([epochs[i], epochs[i]])\n    ax.set_title(f'val loss {val_loss[i]:.3f}')\n"
         "    return (vline,)",
         {"post": _post(f"_call(update, {ks}, lambda: [[fc.r(l.get_xdata(), 4) for l in fc.ax().get_lines()], fc.ax().get_title()])"),
          "require": ["set_xdata", "set_title"]}),
        ("set_data", "Animate an activation map: `im = ax.imshow(maps[0], vmin=0, vmax=1, cmap='magma')` (fixed color "
                     "range, so frames are comparable) and `update(t)` swaps in `maps[t]` with `im.set_data`, returning "
                     "`(im,)`.",
         "fig, ax = plt.subplots()\nim = ax.imshow(maps[0], vmin=0, vmax=1, cmap='magma')\ndef update(t):\n    im.set_data(maps[t])\n"
         "    return (im,)", {"post": _post("_call(update, [1, 4], lambda: fc.images(fc.ax(), 3, data=True))")}),
        ("set_x", "Highlight the current epoch with a moving band: plot `train_loss`, add `span = ax.axvspan(epochs[0] - "
                  "0.5, epochs[0] + 0.5, color='gray', alpha=0.3)` (a Rectangle) and write `update(i)` that slides it to "
                  "`epochs[i] - 0.5` with `span.set_x(...)`, returning `(span,)`.",
         "fig, ax = plt.subplots()\nax.plot(epochs, train_loss)\nspan = ax.axvspan(epochs[0] - 0.5, epochs[0] + 0.5, color='gray', alpha=0.3)\n"
         "def update(i):\n    span.set_x(epochs[i] - 0.5)\n    return (span,)",
         {"post": _post(f"_call(update, {ks}, lambda: [fc.r([p.get_x(), p.get_width()], 3) for p in fc.ax().patches])"),
          "require": ["axvspan", "set_x"]}),
    ]
    return catalog(rng, items, setup, f"{desc}; `traj` holds 6 steps of 8 moving 2-D points and `probs` 6 steps of 4 "
                                      f"class probabilities. Animations call your `update` with each frame; it's checked "
                                      f"by calling it.")


@template(M, "easy")
def gif_export(rng):
    code, n = _wave(rng)
    N = rng.randint(4, 8)
    fps = rng.choice([5, 10, 20, 25])
    dpi = rng.choice([20, 30, 40])
    w, h = rng.choice([(2, 1.5), (3, 2), (2, 2)])
    name = rng.choice(["wave", "train", "demo", "anim"])
    setup = f"{code}\nfig, ax = plt.subplots(figsize=({w}, {h}))\nline, = ax.plot(x, np.sin(x))\nax.set_ylim(-1.2, 1.2)"
    up = "def update(frame):\n    line.set_ydata(np.sin(x + frame / 3))\n    return (line,)\n"
    imp = "from matplotlib.animation import FuncAnimation, PillowWriter\n"
    G = _post(f"_gif({name + '.gif'!r})")
    items = [
        ("PillowWriter", f"Animate the wave over {N} frames (`update(frame)` shifts it by `frame / 3`) and save it as "
                         f"`{name}.gif` with `PillowWriter(fps={fps})` at `dpi={dpi}`.",
         f"{imp}{up}anim = FuncAnimation(fig, update, frames={N})\nanim.save({name + '.gif'!r}, writer=PillowWriter(fps={fps}), dpi={dpi})",
         {"post": G, "require": ["FuncAnimation", "PillowWriter"]}),
        (".save", f"Same wave animation ({N} frames, shift `frame / 3`), saved by writer **name**: `anim.save("
                  f"{name + '.gif'!r}, writer='pillow', fps={fps}, dpi={dpi})`.",
         f"from matplotlib.animation import FuncAnimation\n{up}anim = FuncAnimation(fig, update, frames={N})\n"
         f"anim.save({name + '.gif'!r}, writer='pillow', fps={fps}, dpi={dpi})", {"post": G, "require": ["save"]}),
        (".grab_frame", f"Record frames by hand, like a training loop would: `writer = PillowWriter(fps={fps})`, then "
                        f"`with writer.saving(fig, {name + '.gif'!r}, dpi={dpi}):` loop over k = 0 … {N - 1}, set the "
                        f"line to `np.sin(x + k / 3)` and call `writer.grab_frame()`.",
         f"from matplotlib.animation import PillowWriter\nwriter = PillowWriter(fps={fps})\n"
         f"with writer.saving(fig, {name + '.gif'!r}, dpi={dpi}):\n    for k in range({N}):\n"
         f"        line.set_ydata(np.sin(x + k / 3))\n        writer.grab_frame()",
         {"post": G, "require": ["saving", "grab_frame"]}),
        ("ArtistAnimation", f"Pre-render {N} frames on a **new** figure `fig2, ax2 = plt.subplots(figsize=({w}, {h}))`: "
                            f"frame k is `[ax2.imshow(np.random.default_rng(k).random((8, 8)), animated=True)]`. Make an "
                            f"`ArtistAnimation` and save it as `{name}.gif` with `PillowWriter(fps={fps})`, `dpi={dpi}`.",
         f"from matplotlib.animation import ArtistAnimation, PillowWriter\nfig2, ax2 = plt.subplots(figsize=({w}, {h}))\n"
         f"frames = [[ax2.imshow(np.random.default_rng(k).random((8, 8)), animated=True)] for k in range({N})]\n"
         f"ArtistAnimation(fig2, frames).save({name + '.gif'!r}, writer=PillowWriter(fps={fps}), dpi={dpi})",
         {"post": G, "require": ["ArtistAnimation", "PillowWriter"]}),
        ("FuncAnimation", f"`frames` may be any iterable: animate the wave with `frames=range(0, {3 * N}, 3)` (update "
                          f"shifts by `frame / 3`) and save `{name}.gif` with `PillowWriter(fps={fps})`, `dpi={dpi}`. "
                          f"Return the GIF's frame count from Pillow: `Image.open(...).n_frames` (`from PIL import Image`).",
         f"{imp}from PIL import Image\n{up}anim = FuncAnimation(fig, update, frames=range(0, {3 * N}, 3))\n"
         f"anim.save({name + '.gif'!r}, writer=PillowWriter(fps={fps}), dpi={dpi})\nresult = Image.open({name + '.gif'!r}).n_frames",
         {"require": ["FuncAnimation", "n_frames"]}),
        ("save_count", f"A generator function makes `frames` endless, so saving needs a limit: `FuncAnimation(fig, update, "
                       f"frames=gen, save_count={N})` with `def gen(): k = 0; while True: yield k; k += 1`. Save "
                       f"`{name}.gif` (`PillowWriter(fps={fps})`, `dpi={dpi}`).",
         f"{imp}{up}def gen():\n    k = 0\n    while True:\n        yield k\n        k += 1\n"
         f"anim = FuncAnimation(fig, update, frames=gen, save_count={N})\n"
         f"anim.save({name + '.gif'!r}, writer=PillowWriter(fps={fps}), dpi={dpi})", {"post": G, "require": ["save_count"]}),
        ("savefig_kwargs", f"Dark-mode GIF: animate the wave over {N} frames and save `{name}.gif` (`PillowWriter(fps={fps})`, "
                           f"`dpi={dpi}`) with a black figure background, passed through to every frame's `savefig` "
                           f"via `savefig_kwargs={{'facecolor': 'black'}}`. Checked: frames and the top-left pixel.",
         f"{imp}{up}anim = FuncAnimation(fig, update, frames={N})\n"
         f"anim.save({name + '.gif'!r}, writer=PillowWriter(fps={fps}), dpi={dpi}, savefig_kwargs={{'facecolor': 'black'}})",
         {"post": PRE + f"from PIL import Image\n_gif({name + '.gif'!r}), Image.open({name + '.gif'!r}).convert('RGB').getpixel((0, 0))",
          "require": ["savefig_kwargs"]}),
    ]
    return catalog(rng, items, setup, f"`fig, ax` ({w} × {h} inches) show a sine wave `line` over `x`. The GIF is "
                                      f"checked with Pillow: frame count, size in pixels (inches × dpi) and per-frame "
                                      f"duration. {GIF_NOTE}")


@template(M, "easy")
def live_loops(rng):
    code, N = _losses(rng)
    k = rng.choice([5, 10, 10, 20])
    W = rng.choice([10, 20])
    setup = code + "\nfig, ax = plt.subplots(figsize=(3, 2))\nline, = ax.plot([], [], label='train')"
    L = "fc.lines(fc.ax(), nd=4), fc.limits()"
    items = [
        (".autoscale_view", f"Live training plot: loop over the steps; every {k} steps (when `step % {k} == 0`) append "
                            f"`(step, losses[step])` to two lists, update `line` with `set_data`, then `ax.relim()` and "
                            f"`ax.autoscale_view()` so the axes follow the data.",
         f"xs, ys = [], []\nfor step in range(len(losses)):\n    if step % {k} == 0:\n        xs.append(step)\n"
         f"        ys.append(losses[step])\n        line.set_data(xs, ys)\n        ax.relim()\n        ax.autoscale_view()",
         {"post": L, "require": ["relim", "autoscale_view", "set_data"]}),
        ("plt.pause", f"The full live loop for scripts: `plt.ion()` first; every {k * 2} steps update `line` (all logged "
                      f"points so far), `ax.relim()`, `ax.autoscale_view()`, `fig.canvas.draw_idle()` and "
                      f"`plt.pause(0.001)`; `plt.ioff()` at the end.",
         f"plt.ion()\nxs, ys = [], []\nfor step in range(len(losses)):\n    if step % {k * 2} == 0:\n        xs.append(step)\n"
         f"        ys.append(losses[step])\n        line.set_data(xs, ys)\n        ax.relim()\n        ax.autoscale_view()\n"
         f"        fig.canvas.draw_idle()\n        plt.pause(0.001)\nplt.ioff()",
         {"post": L + ", plt.isinteractive()", "require": ["ion", "pause", "draw_idle", "ioff"]}),
        (".set_data", f"Show only the **last {W}** logged points: every {k} steps append the loss, then set the line "
                      f"to the last {W} (x = their steps), relim and autoscale.",
         f"xs, ys = [], []\nfor step in range(len(losses)):\n    if step % {k} == 0:\n        xs.append(step)\n"
         f"        ys.append(losses[step])\n        line.set_data(xs[-{W}:], ys[-{W}:])\n        ax.relim()\n        ax.autoscale_view()",
         {"post": L}),
        (".set_text", f"Add `txt = ax.text(0.95, 0.9, '', transform=ax.transAxes, ha='right')` and, every {k} steps, "
                      f"update both the line (all logged points; relim + autoscale) and the text to `f'loss "
                      f"{{losses[step]:.3f}}'`.",
         f"txt = ax.text(0.95, 0.9, '', transform=ax.transAxes, ha='right')\nxs, ys = [], []\n"
         f"for step in range(len(losses)):\n    if step % {k} == 0:\n        xs.append(step)\n        ys.append(losses[step])\n"
         f"        line.set_data(xs, ys)\n        txt.set_text(f'loss {{losses[step]:.3f}}')\n        ax.relim()\n        ax.autoscale_view()",
         {"post": L + ", fc.texts()"}),
        (".plot", f"Add a second line `avg, = ax.plot([], [], label='avg')` showing the mean of the last {k} losses, "
                  f"logged together with the raw loss every {k} steps (from step {k - 1} on: `step % {k} == {k - 1}`), "
                  f"plus a legend. Relim + autoscale after each update.",
         f"avg, = ax.plot([], [], label='avg')\nax.legend()\nxs, ys, ms = [], [], []\nfor step in range(len(losses)):\n"
         f"    if step % {k} == {k - 1}:\n        xs.append(step)\n        ys.append(losses[step])\n"
         f"        ms.append(losses[step - {k - 1}:step + 1].mean())\n        line.set_data(xs, ys)\n        avg.set_data(xs, ms)\n"
         f"        ax.relim()\n        ax.autoscale_view()", {"post": L + ", fc.legend()", "require": ["set_data", "legend"]}),
        (".set_yscale", f"Losses span a wide range: set `ax.set_yscale('log')` once, then every {k} steps update the line "
                        f"with all logged points, relim and autoscale.",
         f"ax.set_yscale('log')\nxs, ys = [], []\nfor step in range(len(losses)):\n    if step % {k} == 0:\n        xs.append(step)\n"
         f"        ys.append(losses[step])\n        line.set_data(xs, ys)\n        ax.relim()\n        ax.autoscale_view()",
         {"post": L + ", fc.scales()"}),
        (".plot", f"Mark the best loss so far: add `best, = ax.plot([], [], 'r*')`; every {k} steps update the line (all "
                  f"logged points) and move `best` to the lowest logged point, then relim + autoscale.",
         f"best, = ax.plot([], [], 'r*')\nxs, ys = [], []\nfor step in range(len(losses)):\n    if step % {k} == 0:\n"
         f"        xs.append(step)\n        ys.append(losses[step])\n        line.set_data(xs, ys)\n        b = int(np.argmin(ys))\n"
         f"        best.set_data([xs[b]], [ys[b]])\n        ax.relim()\n        ax.autoscale_view()",
         {"post": "fc.lines(fc.ax(), 'marker', nd=4), fc.limits()"}),
        ("set_data", f"A slower second metric: add `vl, = ax.plot([], [], label='val')`; every {k} steps update the "
                     f"train line, and every {2 * k} steps also append `losses[step] + 0.1` to the val line; relim + "
                     f"autoscale after each update.",
         f"vl, = ax.plot([], [], label='val')\nxs, ys, vx, vy = [], [], [], []\nfor step in range(len(losses)):\n"
         f"    if step % {k} == 0:\n        xs.append(step)\n        ys.append(losses[step])\n        line.set_data(xs, ys)\n"
         f"        if step % {2 * k} == 0:\n            vx.append(step)\n            vy.append(losses[step] + 0.1)\n"
         f"            vl.set_data(vx, vy)\n        ax.relim()\n        ax.autoscale_view()", {"post": "fc.lines(fc.ax(), 'label', nd=4), fc.limits()"}),
    ]
    return catalog(rng, items, setup, f"`losses` holds {N} training-step losses and `fig, ax` an empty `line`. Update the "
                                      f"**existing** line in a loop (no new `plot` calls per step).")


@template(M, "easy")
def animation_numbers(rng):
    code, n = _wave(rng)
    N = rng.randint(10, 60)
    I = rng.choice([20, 40, 50, 100, 125, 200])
    fps = rng.choice([4, 5, 8, 10, 20, 25, 50])
    secs = rng.choice([2, 3, 4])
    k = rng.choice([3, 4, 5])
    setup = (f"{code}\nfig, ax = plt.subplots(figsize=(2, 1.5))\nline, = ax.plot(x, np.sin(x))\n"
             f"def update(frame):\n    line.set_ydata(np.sin(x + frame / 4))\n    return (line,)")
    items = [
        ("FuncAnimation", f"Create `anim = FuncAnimation(fig, update, frames={N}, interval={I})` and return how long one "
                          f"pass takes in seconds, from the animation's own timer (`anim.event_source.interval` is in ms).",
         f"from matplotlib.animation import FuncAnimation\nanim = FuncAnimation(fig, update, frames={N}, interval={I})\n"
         f"result = {N} * anim.event_source.interval / 1000", {"require": ["FuncAnimation", "event_source"]}),
        ("FuncAnimation", f"A {secs}-second clip at {fps} fps: create `anim` with `interval=1000 / {fps}` and as many "
                          f"frames as that needs, and return `[number of frames, interval]` (count the frames with "
                          f"`len(list(anim.new_frame_seq()))`).",
         f"from matplotlib.animation import FuncAnimation\nanim = FuncAnimation(fig, update, frames={secs * fps}, interval=1000 / {fps})\n"
         f"result = [len(list(anim.new_frame_seq())), 1000 / {fps}]", {"require": ["FuncAnimation", "new_frame_seq"]}),
        ("new_frame_seq", f"Return the frame values `update` would receive for `frames=range(0, {N}, {k})`: build the "
                          f"animation and list `anim.new_frame_seq()`.",
         f"from matplotlib.animation import FuncAnimation\nanim = FuncAnimation(fig, update, frames=range(0, {N}, {k}))\n"
         f"result = list(anim.new_frame_seq())"),
        ("PillowWriter", f"PillowWriter gives every frame `int(1000 / fps)` ms, and the GIF stores hundredths of a second "
                         f"(rounded down). Save a 3-frame animation at {fps} fps (dpi 20) as `t.gif` and return the "
                         f"duration Pillow reads back for the first frame (`Image.open('t.gif').info['duration']`).",
         f"from matplotlib.animation import FuncAnimation, PillowWriter\nfrom PIL import Image\n"
         f"FuncAnimation(fig, update, frames=3).save('t.gif', writer=PillowWriter(fps={fps}), dpi=20)\n"
         f"result = Image.open('t.gif').info['duration']", {"require": ["PillowWriter", "info"]}),
        (".save", f"A GIF is `figsize × dpi` pixels. Save a 3-frame animation of `fig` at `dpi={k * 10}` as `s.gif` and "
                  f"return `Image.open('s.gif').size` as a list — and check it against `fig.get_size_inches() * {k * 10}`.",
         f"from matplotlib.animation import FuncAnimation, PillowWriter\nfrom PIL import Image\n"
         f"FuncAnimation(fig, update, frames=3).save('s.gif', writer=PillowWriter(fps=10), dpi={k * 10})\n"
         f"result = list(Image.open('s.gif').size)", {"require": ["save", "size"]}),
        ("PillowWriter", f"Return the total play time in ms of a {k}-frame GIF saved at {fps} fps (dpi 20): sum the "
                         f"per-frame durations Pillow reads back (`im.seek(i)`, `im.info['duration']` for each of "
                         f"`im.n_frames` frames).",
         f"from matplotlib.animation import FuncAnimation, PillowWriter\nfrom PIL import Image\n"
         f"FuncAnimation(fig, update, frames={k}).save('p.gif', writer=PillowWriter(fps={fps}), dpi=20)\n"
         f"im = Image.open('p.gif')\ntotal = 0\nfor i in range(im.n_frames):\n    im.seek(i)\n    total += im.info['duration']\n"
         f"result = total", {"require": ["PillowWriter", "seek"]}),
        ("FuncAnimation", f"Same animation, twice as fast: create `slow = FuncAnimation(fig, update, frames={N}, "
                          f"interval={I})` and a second one `fast` with the same frames at double speed (half the "
                          f"interval). Return both playback lengths in seconds, `frames × event_source.interval / "
                          f"1000`, as `[slow, fast]`.",
         f"from matplotlib.animation import FuncAnimation\nslow = FuncAnimation(fig, update, frames={N}, interval={I})\n"
         f"fast = FuncAnimation(fig, update, frames={N}, interval={I} / 2)\n"
         f"result = [{N} * slow.event_source.interval / 1000, {N} * fast.event_source.interval / 1000]",
         {"require": ["FuncAnimation", "interval"]}),
        ("FuncAnimation", f"How often does `update` run while saving? Wrap it: `calls = []` and `def counted(f): "
                          f"calls.append(f); return update(f)`. Save `FuncAnimation(fig, counted, frames={k})` as `c.gif` "
                          f"(`PillowWriter(fps=10)`, dpi 20) and return `calls`. (Without an `init_func`, Matplotlib "
                          f"draws the first frame once more to initialize.)",
         f"from matplotlib.animation import FuncAnimation, PillowWriter\ncalls = []\ndef counted(f):\n    calls.append(f)\n"
         f"    return update(f)\nFuncAnimation(fig, counted, frames={k}).save('c.gif', writer=PillowWriter(fps=10), dpi=20)\n"
         f"result = calls", {"require": ["FuncAnimation", "save"]}),
    ]
    return catalog(rng, items, setup, f"`fig, ax` show a wave `line` and `update(frame)` shifts it. Import from "
                                      f"`matplotlib.animation` (and `PIL.Image` for reading GIFs) as needed.")


# ================================================================= medium

@template(M, "medium")
def gd_animation(rng):
    code, F, (c0, c1, k, lr) = _gd(rng)
    fps = rng.choice([5, 10])
    ks = sorted(rng.sample(range(F), 3))
    base = ("fig, ax = plt.subplots(figsize=(3, 2.5))\nax.contour(W0, W1, loss_fn(W0, W1), levels=15, cmap='viridis')\n"
            "pt, = ax.plot([], [], 'ro-', ms=3)\n")
    trail = "def step(i):\n    pt.set_data(path[:i + 1, 0], path[:i + 1, 1])\n    return (pt,)\n"
    C = "[type(c).__name__ for c in fc.ax().collections]"
    items = [
        ("contour", "Drill: draw the loss surface with `ax.contour(W0, W1, loss_fn(W0, W1), levels=15, cmap='viridis')`, "
                    "an empty trail `pt, = ax.plot([], [], 'ro-', ms=3)`, and `step(i)` that shows the path up to step i "
                    "(`path[:i + 1]`) and returns `(pt,)`.",
         base + trail, {"post": _post(f"_call(step, {ks}, lambda: fc.lines(fc.ax(), 'marker', 'color'))", C),
                        "require": ["contour", "set_data"]}),
        ("PillowWriter", f"Drill: the same contour plot and trail (`step(i)` shows `path[:i + 1]`), animated over all "
                         f"`len(path)` steps and saved as `gd.gif` with `PillowWriter(fps={fps})`, `dpi=30`.",
         "from matplotlib.animation import FuncAnimation, PillowWriter\n" + base + trail +
         f"FuncAnimation(fig, step, frames=len(path)).save('gd.gif', writer=PillowWriter(fps={fps}), dpi=30)",
         {"post": _post("_gif('gd.gif')"), "require": ["FuncAnimation", "PillowWriter"]}),
        ("set_title", "Same contour plot and trail; `step(i)` also sets the title to `f'step {i}: loss "
                      "{loss_fn(*path[i]):.3f}'`.",
         base + "def step(i):\n    pt.set_data(path[:i + 1, 0], path[:i + 1, 1])\n"
                "    ax.set_title(f'step {i}: loss {loss_fn(*path[i]):.3f}')\n    return (pt,)",
         {"post": _post(f"_call(step, {ks}, lambda: [fc.lines(fc.ax()), fc.ax().get_title()])")}),
        ("set_data", "Show only where the optimizer **is**, not where it was: a single red dot `dot, = ax.plot([], [], "
                     "'ro')` on the contour plot, moved to `path[i]` by `step(i)`.",
         "fig, ax = plt.subplots(figsize=(3, 2.5))\nax.contour(W0, W1, loss_fn(W0, W1), levels=15, cmap='viridis')\n"
         "dot, = ax.plot([], [], 'ro')\ndef step(i):\n    dot.set_data([path[i, 0]], [path[i, 1]])\n    return (dot,)",
         {"post": _post(f"_call(step, {ks}, lambda: fc.lines(fc.ax()))")}),
        ("plt.subplots", "Two panels (`plt.subplots(1, 2, figsize=(5, 2.5))`): left, the contour plot with the trail "
                         "`pt` ('ro-'); right, the loss per step so far as a line `lc` (x = step 0 … i, y = "
                         "`loss_fn(path[:i + 1, 0], path[:i + 1, 1])`) with x limits 0 … len(path) - 1 and y limits 0 … "
                         "the first loss. `step(i)` updates both and returns `(pt, lc)`.",
         "fig, (a1, a2) = plt.subplots(1, 2, figsize=(5, 2.5))\na1.contour(W0, W1, loss_fn(W0, W1), levels=15, cmap='viridis')\n"
         "pt, = a1.plot([], [], 'ro-', ms=3)\nlc, = a2.plot([], [])\nL = loss_fn(path[:, 0], path[:, 1])\n"
         "a2.set_xlim(0, len(path) - 1)\na2.set_ylim(0, L[0])\ndef step(i):\n    pt.set_data(path[:i + 1, 0], path[:i + 1, 1])\n"
         "    lc.set_data(range(i + 1), L[:i + 1])\n    return (pt, lc)",
         {"post": _post(f"_call(step, {ks}, lambda: [fc.lines(a) for a in fc.axes()])", "fc.limits(fc.ax(1))")}),
        ("contourf", "Filled version: `ax.contourf(W0, W1, loss_fn(W0, W1), levels=15, cmap='magma')`, a white trail "
                     "`pt, = ax.plot([], [], 'o-', color='white', ms=3)`, and `step(i)` showing `path[:i + 1]`.",
         "fig, ax = plt.subplots(figsize=(3, 2.5))\nax.contourf(W0, W1, loss_fn(W0, W1), levels=15, cmap='magma')\n"
         "pt, = ax.plot([], [], 'o-', color='white', ms=3)\n" + trail,
         {"post": _post(f"_call(step, {ks}, lambda: fc.lines(fc.ax(), 'color'))", C), "require": ["contourf"]}),
        ("legend", "Plain GD vs momentum: on the contour plot draw two trails, `pg, = ax.plot([], [], 'o-', color='C3', "
                   "ms=3, label='gd')` for `path` and `pm, = ax.plot([], [], 'o-', color='C1', ms=3, label='momentum')` for "
                   "`mpath`, plus a legend; `step(i)` advances both to step i and returns `(pg, pm)`.",
         "fig, ax = plt.subplots(figsize=(3, 2.5))\nax.contour(W0, W1, loss_fn(W0, W1), levels=15, cmap='viridis')\n"
         "pg, = ax.plot([], [], 'o-', color='C3', ms=3, label='gd')\npm, = ax.plot([], [], 'o-', color='C1', ms=3, label='momentum')\n"
         "ax.legend()\ndef step(i):\n    pg.set_data(path[:i + 1, 0], path[:i + 1, 1])\n    pm.set_data(mpath[:i + 1, 0], mpath[:i + 1, 1])\n"
         "    return (pg, pm)", {"post": _post(f"_call(step, {ks}, lambda: fc.lines(fc.ax(), 'color', 'label'))", "fc.legend()")}),
    ]
    return catalog(rng, items, code, f"Gradient descent with learning rate {lr} on `loss_fn` (minimum at ({c0}, {c1})): "
                                     f"`path` holds the weights after each of the {F - 1} steps (`mpath` the same with momentum) and `W0`, `W1` a 40 × 40 "
                                     f"grid. Name the update function `step`; it's checked by calling it.")


@template(M, "medium")
def boundary_animation(rng):
    code, E = _boundary(rng)
    es = sorted(rng.sample(range(E), 3))
    head = ("fig, ax = plt.subplots(figsize=(3, 3))\nax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=10)\n"
            "ax.set_xlim(-3, 3)\nax.set_ylim(-3, 3)\nline, = ax.plot([], [], 'k-')\n")
    upd = "def update(e):\n    b, w1, w2 = W[e]\n    line.set_data(xs, -(b + w1 * xs) / w2)\n"
    items = [
        ("set_data", "Drill: animate the decision boundary. Scatter `X` colored by `y` (`cmap='coolwarm'`, `s=10`), fix "
                     "both limits to -3 … 3, add `line, = ax.plot([], [], 'k-')`, and write `update(e)` that draws the "
                     "boundary of epoch e — where `b + w1·x1 + w2·x2 = 0`, i.e. `x2 = -(b + w1·xs) / w2` over `xs` — "
                     "and returns `(line,)`.",
         head + upd + "    return (line,)", {"post": _post(f"_call(update, {es}, lambda: fc.lines(fc.ax(), nd=4))", "fc.limits()")}),
        ("set_text", "Same boundary animation, plus a counter `txt = ax.text(0.02, 0.95, '', transform=ax.transAxes, "
                     "va='top')` that `update(e)` sets to `f'epoch {e + 1}'`; return `(line, txt)`.",
         head + "txt = ax.text(0.02, 0.95, '', transform=ax.transAxes, va='top')\n" + upd +
         "    txt.set_text(f'epoch {e + 1}')\n    return (line, txt)",
         {"post": _post(f"_call(update, {es}, lambda: [fc.lines(fc.ax(), nd=4), fc.texts()])")}),
        ("set_array", "Color the points by the model's **prediction** at each epoch: `sc = ax.scatter(X[:, 0], X[:, 1], "
                      "c=np.zeros(50), cmap='coolwarm', vmin=0, vmax=1)`; `update(e)` computes `pred = (Xb @ W[e] > 0)` "
                      "as 0/1 and calls `sc.set_array(pred.astype(float))`; return `(sc,)`.",
         "fig, ax = plt.subplots(figsize=(3, 3))\nsc = ax.scatter(X[:, 0], X[:, 1], c=np.zeros(50), cmap='coolwarm', vmin=0, vmax=1)\n"
         "def update(e):\n    pred = (Xb @ W[e] > 0)\n    sc.set_array(pred.astype(float))\n    return (sc,)",
         {"post": _post(f"_call(update, {es}, lambda: fc.scatter(fc.ax(), 0, 6, 'array', 'clim'))"), "require": ["set_array"]}),
        ("imshow", "Animate the predicted probability as an image: on a 20 × 20 grid (`G0, G1 = np.meshgrid(xs, xs)`), "
                   "`im = ax.imshow(np.zeros((20, 20)), extent=(-3, 3, -3, 3), origin='lower', vmin=0, vmax=1, "
                   "cmap='RdBu_r')`; `update(e)` sets `im.set_data(1 / (1 + np.exp(-(b + w1 * G0 + w2 * G1))))`.",
         "fig, ax = plt.subplots(figsize=(3, 3))\nG0, G1 = np.meshgrid(xs, xs)\n"
         "im = ax.imshow(np.zeros((20, 20)), extent=(-3, 3, -3, 3), origin='lower', vmin=0, vmax=1, cmap='RdBu_r')\n"
         "def update(e):\n    b, w1, w2 = W[e]\n    im.set_data(1 / (1 + np.exp(-(b + w1 * G0 + w2 * G1))))\n    return (im,)",
         {"post": _post(f"_call(update, {es}, lambda: fc.images(fc.ax(), 4, data=True))"), "require": ["imshow", "set_data"],
          "tol": 1e-4}),
        ("contourf", "A ContourSet can't be updated in place, so redraw it: keep the current one in a list `cs = "
                     "[None]`; `update(e)` removes the old one (`cs[0].remove()`) if any and draws `ax.contourf(G0, G1, "
                     "prob, levels=[0, 0.5, 1], colors=['C0', 'C3'], alpha=0.3)` for epoch e's probability on the 20 × "
                     "20 grid `G0, G1 = np.meshgrid(xs, xs)`. After several frames exactly one ContourSet must remain.",
         "fig, ax = plt.subplots(figsize=(3, 3))\nG0, G1 = np.meshgrid(xs, xs)\ncs = [None]\ndef update(e):\n    b, w1, w2 = W[e]\n"
         "    prob = 1 / (1 + np.exp(-(b + w1 * G0 + w2 * G1)))\n    if cs[0] is not None:\n        cs[0].remove()\n"
         "    cs[0] = ax.contourf(G0, G1, prob, levels=[0, 0.5, 1], colors=['C0', 'C3'], alpha=0.3)\n    return []",
         {"post": f"for _e in {es}:\n    update(_e)\n[type(c).__name__ for c in fc.ax().collections], "
                  "[fc.r(c.levels, 3) for c in fc.ax().collections]", "require": ["contourf", "remove"]}),
        ("PillowWriter", "Drill: the boundary animation (scatter colored by `y`, limits -3 … 3, `line` updated per epoch) "
                         "over all `len(W)` epochs, saved as `boundary.gif` with `PillowWriter(fps=5)` at `dpi=30`.",
         "from matplotlib.animation import FuncAnimation, PillowWriter\n" + head + upd + "    return (line,)\n"
         "FuncAnimation(fig, update, frames=len(W)).save('boundary.gif', writer=PillowWriter(fps=5), dpi=30)",
         {"post": _post("_gif('boundary.gif')[0]")}),
        ("plt.subplots", "Boundary and accuracy side by side (`plt.subplots(1, 2, figsize=(5, 2.5))`): left, the scatter "
                         "(colored by `y`), limits -3 … 3 and the boundary `line`; right, the training accuracy per epoch "
                         "so far, `acc = ((Xb @ W.T > 0) == y[:, None]).mean(axis=0)`, as `al, = a2.plot([], [], 'o-')` with "
                         "x limits 0 … len(W) - 1 and y limits 0 … 1. `update(e)` updates both and returns `(line, al)`.",
         "fig, (a1, a2) = plt.subplots(1, 2, figsize=(5, 2.5))\na1.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=10)\n"
         "a1.set_xlim(-3, 3)\na1.set_ylim(-3, 3)\nline, = a1.plot([], [], 'k-')\nacc = ((Xb @ W.T > 0) == y[:, None]).mean(axis=0)\n"
         "al, = a2.plot([], [], 'o-')\na2.set_xlim(0, len(W) - 1)\na2.set_ylim(0, 1)\ndef update(e):\n    b, w1, w2 = W[e]\n"
         "    line.set_data(xs, -(b + w1 * xs) / w2)\n    al.set_data(range(e + 1), acc[:e + 1])\n    return (line, al)",
         {"post": _post(f"_call(update, {es}, lambda: [fc.lines(a, nd=4) for a in fc.axes()])", "[fc.limits(a) for a in fc.axes()]")}),
    ]
    return catalog(rng, items, code, f"A logistic regression trained for {E} epochs on two classes `X`, `y` (`Xb` = X "
                                     f"with a column of ones); `W[e]` = (bias, w1, w2) after epoch e and `xs` = 20 "
                                     f"points from -3 to 3. Name the update function `update`; it's checked by calling it.")


@template(M, "medium")
def live_training_plots(rng):
    code, N = _losses(rng)
    k = rng.choice([10, 10, 20])
    s = seeded(rng)
    setup = code + (f"\nval_losses = np.round(losses + 0.15 + 0.05 * np.random.default_rng({s}).random(len(losses)), 4)\n"
                    f"lrs = np.round(1e-3 * np.minimum(1, (np.arange(len(losses)) + 1) / 20), 7)")
    L = "[fc.lines(a, 'label', nd=5) for a in fc.axes()], [fc.limits(a) for a in fc.axes()]"
    loop = (f"xs, tr = [], []\nfor step in range(len(losses)):\n    if step % {k} == 0:\n        xs.append(step)\n"
            f"        tr.append(losses[step])\n")
    tail = "        ax.relim()\n        ax.autoscale_view()\n        fig.canvas.draw_idle()\n        plt.pause(0.001)\nplt.ioff()"
    items = [
        ("plt.pause", f"Drill: a live loss plot that updates every {k} steps. `plt.ion()`, then `fig, ax = plt.subplots("
                      f"figsize=(3, 2))`, `line, = ax.plot([], [], label='train')`, `ax.legend()`; in the loop, every {k} "
                      f"steps add `(step, losses[step])`, `set_data`, `relim`, `autoscale_view`, `fig.canvas.draw_idle()`, "
                      f"`plt.pause(0.001)`; `plt.ioff()` after.",
         "plt.ion()\nfig, ax = plt.subplots(figsize=(3, 2))\nline, = ax.plot([], [], label='train')\nax.legend()\n"
         + loop + "        line.set_data(xs, tr)\n" + tail,
         {"post": L + ", fc.legend()", "require": ["ion", "pause", "relim", "autoscale_view"]}),
        (".set_data", f"Train **and** validation, live: lines 'train' and 'val' on one Axes (legend); every {k} steps add "
                      f"the train loss, every {2 * k} steps also the validation loss (`val_losses[step]`), update both "
                      f"lines, relim + autoscale, `draw_idle` + `pause(0.001)` (inside `plt.ion()` … `plt.ioff()`).",
         "plt.ion()\nfig, ax = plt.subplots(figsize=(3, 2))\nline, = ax.plot([], [], label='train')\n"
         "vline, = ax.plot([], [], label='val')\nax.legend()\nvx, va = [], []\n" + loop +
         f"        line.set_data(xs, tr)\n        if step % {2 * k} == 0:\n            vx.append(step)\n"
         f"            va.append(val_losses[step])\n            vline.set_data(vx, va)\n" + tail,
         {"post": L + ", fc.legend()", "require": ["set_data", "pause"]}),
        ("plt.subplots", f"A live two-panel dashboard (`fig, (a1, a2) = plt.subplots(2, 1, sharex=True, figsize=(3, 3))`): "
                         f"loss on top, learning rate (`lrs`) below, one line each, both updated every {k} steps with "
                         f"relim + autoscale on **both** Axes, then `fig.canvas.draw_idle()` + `plt.pause(0.001)` "
                         f"(inside `plt.ion()` … `plt.ioff()`).",
         "plt.ion()\nfig, (a1, a2) = plt.subplots(2, 1, sharex=True, figsize=(3, 3))\nl1, = a1.plot([], [])\nl2, = a2.plot([], [])\n"
         f"xs, tr, lr = [], [], []\nfor step in range(len(losses)):\n    if step % {k} == 0:\n        xs.append(step)\n"
         f"        tr.append(losses[step])\n        lr.append(lrs[step])\n        l1.set_data(xs, tr)\n        l2.set_data(xs, lr)\n"
         f"        for a in (a1, a2):\n            a.relim()\n            a.autoscale_view()\n        fig.canvas.draw_idle()\n"
         f"        plt.pause(0.001)\nplt.ioff()", {"post": L, "require": ["subplots", "relim", "pause"]}),
        (".set_text", f"Live plot with a best-so-far readout: every {k} steps update the 'train' line (relim + autoscale, "
                      f"`draw_idle`, `pause(0.001)`, within `plt.ion()` … `plt.ioff()`) and a text `txt = ax.text(0.95, "
                      f"0.9, '', transform=ax.transAxes, ha='right')` reading `f'best {{min(tr):.3f}} @ {{xs[int(np.argmin(tr))]}}'`.",
         "plt.ion()\nfig, ax = plt.subplots(figsize=(3, 2))\nline, = ax.plot([], [], label='train')\n"
         "txt = ax.text(0.95, 0.9, '', transform=ax.transAxes, ha='right')\n" + loop +
         "        line.set_data(xs, tr)\n        txt.set_text(f'best {min(tr):.3f} @ {xs[int(np.argmin(tr))]}')\n" + tail,
         {"post": L + ", fc.texts()"}),
        (".set_yscale", f"Live plot on a log y-axis with an EMA-smoothed line: lines 'raw' and 'ema' (legend), `ax."
                        f"set_yscale('log')`; every {k} steps log the raw loss and `ema = 0.7 * ema + 0.3 * loss` (start "
                        f"`ema` at the first logged loss), update both lines, relim + autoscale, `draw_idle` + "
                        f"`pause(0.001)` (inside `plt.ion()` … `plt.ioff()`).",
         "plt.ion()\nfig, ax = plt.subplots(figsize=(3, 2))\nraw, = ax.plot([], [], label='raw')\nsm, = ax.plot([], [], label='ema')\n"
         "ax.set_yscale('log')\nax.legend()\nem = []\n" + loop +
         "        em.append(tr[-1] if len(em) == 0 else 0.7 * em[-1] + 0.3 * tr[-1])\n        raw.set_data(xs, tr)\n"
         "        sm.set_data(xs, em)\n" + tail, {"post": L + ", fc.scales()", "require": ["set_yscale", "pause"]}),
    ]
    return catalog(rng, items, setup, f"`losses` / `val_losses` hold per-step losses of a {N}-step run and `lrs` the "
                                      f"learning rate per step. Update existing lines in place.")


@template(M, "medium")
def artist_frame_recipes(rng):
    T = rng.randint(4, 6)
    s = seeded(rng)
    fps = rng.choice([5, 10])
    setup = (f"g = np.random.default_rng({s})\nclean = np.outer(np.hanning(12), np.hanning(12))\n"
             f"noisy = [np.round(clean + (1 - t / {T - 1}) * g.normal(0, 0.5, (12, 12)), 3) for t in range({T})]   # denoising steps\n"
             f"weights = [np.round(g.normal(0, 1 / (e + 1), 300), 3) for e in range({T})]   # a layer's weights per epoch")
    fr = ("[[(type(a).__name__, fc.r(a.get_array(), 3) if hasattr(a, 'get_array') and type(a).__name__ == 'AxesImage' else "
          "(a.get_text() if hasattr(a, 'get_text') else None)) for a in f] for f in frames]")
    items = [
        ("ArtistAnimation", "Pre-render a denoising sequence: on `fig, ax = plt.subplots(figsize=(2, 2))` make one frame per "
                            "step t, `[ax.imshow(noisy[t], cmap='gray', animated=True), ax.text(0.5, 1.02, f't={t}', "
                            "transform=ax.transAxes, ha='center', animated=True)]`, collect them in `frames` and build "
                            "`anim = ArtistAnimation(fig, frames, interval=200)`.",
         "from matplotlib.animation import ArtistAnimation\nfig, ax = plt.subplots(figsize=(2, 2))\nframes = []\n"
         "for t in range(len(noisy)):\n    im = ax.imshow(noisy[t], cmap='gray', animated=True)\n"
         "    txt = ax.text(0.5, 1.02, f't={t}', transform=ax.transAxes, ha='center', animated=True)\n    frames.append([im, txt])\n"
         "anim = ArtistAnimation(fig, frames, interval=200)",
         {"post": _post("_seq(anim)", fr, "anim._interval")}),
        ("PillowWriter", f"Same denoising frames (`[imshow(noisy[t], cmap='gray', animated=True)]` per step, `figsize=(2, "
                         f"2)`), as an `ArtistAnimation` saved to `denoise.gif` with `PillowWriter(fps={fps})`, `dpi=30`.",
         f"from matplotlib.animation import ArtistAnimation, PillowWriter\nfig, ax = plt.subplots(figsize=(2, 2))\n"
         f"frames = [[ax.imshow(n, cmap='gray', animated=True)] for n in noisy]\n"
         f"ArtistAnimation(fig, frames).save('denoise.gif', writer=PillowWriter(fps={fps}), dpi=30)",
         {"post": _post("_gif('denoise.gif')")}),
        ("ArtistAnimation", "Histogram per epoch: for each epoch's `weights[e]` draw `ax.hist(w, bins=np.linspace(-3, 3, "
                            "25), color='C0')` on the same Axes; `hist` returns `(counts, edges, bars)` — a frame is "
                            "`list(bars)`. Build `anim = ArtistAnimation(fig, frames)`.",
         "from matplotlib.animation import ArtistAnimation\nfig, ax = plt.subplots(figsize=(3, 2))\nframes = []\n"
         "for w in weights:\n    counts, edges, bars = ax.hist(w, bins=np.linspace(-3, 3, 25), color='C0')\n    frames.append(list(bars))\n"
         "anim = ArtistAnimation(fig, frames)",
         {"post": _post("_seq(anim)", "[[fc.r(p.get_height(), 3) for p in f] for f in frames]"), "require": ["ArtistAnimation", "hist"]}),
        ("ArtistAnimation", "Frames can be plain lines too: on `fig, ax`, for each epoch e draw `ax.plot(np.sort("
                            "weights[e]), color='C1')` (the sorted weights) — each `plot` returns a list, which is already "
                            "a valid frame — and build `anim = ArtistAnimation(fig, frames, interval=150)`.",
         "from matplotlib.animation import ArtistAnimation\nfig, ax = plt.subplots(figsize=(3, 2))\n"
         "frames = [ax.plot(np.sort(w), color='C1') for w in weights]\nanim = ArtistAnimation(fig, frames, interval=150)",
         {"post": _post("_seq(anim)", "[[fc.r(l.get_ydata(), 3) for l in f] for f in frames]", "anim._interval")}),
        ("ArtistAnimation", "Pre-rendered frames with a moving marker: draw the `clean` image once (static), then for each "
                            "step t a marker `ax.plot(2 * t, 6, 'r^')` (a list, so a valid frame); build "
                            "`anim = ArtistAnimation(fig, frames, interval=100)`. Static artists stay in every frame.",
         "from matplotlib.animation import ArtistAnimation\nfig, ax = plt.subplots(figsize=(2, 2))\nax.imshow(clean, cmap='gray')\n"
         "frames = [ax.plot(2 * t, 6, 'r^') for t in range(len(noisy))]\nanim = ArtistAnimation(fig, frames, interval=100)",
         {"post": _post("_seq(anim)", "[[fc.r(l.get_xydata(), 3) for l in f] for f in frames]", "len(fc.ax().get_images())")}),
    ]
    return catalog(rng, items, setup, f"`noisy` holds {T} denoising steps of a 12 × 12 image (`clean` is the target) and "
                                      f"`weights` a layer's weights after each of {T} epochs. Keep the frames list in "
                                      f"`frames`. {IMPORT}")


@template(M, "medium")
def animated_bars(rng):
    s = seeded(rng)
    T = rng.randint(5, 8)
    V = rng.randint(4, 6)
    toks = rng.sample(["the", "cat", "sat", "on", "mat", "dog", "ran", "a", "big", "red"], V)
    ts = sorted(rng.sample(range(T), 3))
    setup = (f"g = np.random.default_rng({s})\ntokens = {lit(toks)}\nlogits = np.round(g.normal(size=({T}, {V})) * 2, 3)   # one row per decoding step\n"
             f"probs = np.round(np.exp(logits) / np.exp(logits).sum(1, keepdims=True), 4)\n"
             f"acts = [np.round(g.normal(0, 1 + e, 200), 3) for e in range({T})]   # activations per epoch\n"
             f"edges = np.linspace(-8, 8, 17)")
    head = "fig, ax = plt.subplots(figsize=(3, 2))\nbars = ax.bar(tokens, probs[0])\nax.set_ylim(0, 1)\n"
    H = "fc.bars(fc.ax(), 4, 'color')"
    items = [
        ("set_height", "Animate the next-token distribution: `bars = ax.bar(tokens, probs[0])`, y limits 0 … 1, and "
                       "`update(t)` that sets each bar's height to `probs[t]` and returns the bars.",
         head + "def update(t):\n    for b, p in zip(bars, probs[t]):\n        b.set_height(p)\n    return bars.patches",
         {"post": _post(f"_call(update, {ts}, lambda: fc.heights(fc.ax(), 4))")}),
        ("set_color", "Same bars, and highlight the most likely token: `update(t)` sets the heights to `probs[t]` and "
                      "colors the argmax bar 'C3', all others 'C0'.",
         head + "def update(t):\n    best = probs[t].argmax()\n    for i, (b, p) in enumerate(zip(bars, probs[t])):\n"
                "        b.set_height(p)\n        b.set_color('C3' if i == best else 'C0')\n    return bars.patches",
         {"post": _post(f"_call(update, {ts}, lambda: {H})"), "require": ["set_height", "set_color"]}),
        ("set_text", "Same bars plus a title-like text `txt = ax.text(0.5, 0.92, '', transform=ax.transAxes, ha='center')`; "
                     "`update(t)` sets the heights and the text to `f'step {t}: {tokens[probs[t].argmax()]}'`, and "
                     "returns the bars and the text.",
         head + "txt = ax.text(0.5, 0.92, '', transform=ax.transAxes, ha='center')\n"
                "def update(t):\n    for b, p in zip(bars, probs[t]):\n        b.set_height(p)\n"
                "    txt.set_text(f'step {t}: {tokens[probs[t].argmax()]}')\n    return [*bars.patches, txt]",
         {"post": _post(f"_call(update, {ts}, lambda: [fc.heights(fc.ax(), 4), fc.texts()])")}),
        ("histogram", "Animate a histogram with **fixed bins**: `counts, _ = np.histogram(acts[0], bins=edges)`, draw "
                      "`bars = ax.bar(edges[:-1], counts, width=np.diff(edges), align='edge')`, set the y limit to 0 … "
                      "200, and `update(e)` recomputes the counts for `acts[e]` and sets the heights.",
         "fig, ax = plt.subplots(figsize=(3, 2))\ncounts, _ = np.histogram(acts[0], bins=edges)\n"
         "bars = ax.bar(edges[:-1], counts, width=np.diff(edges), align='edge')\nax.set_ylim(0, 200)\n"
         "def update(e):\n    counts, _ = np.histogram(acts[e], bins=edges)\n    for b, c in zip(bars, counts):\n"
         "        b.set_height(c)\n    return bars.patches",
         {"post": _post(f"_call(update, {ts}, lambda: fc.bars(fc.ax(), 4))", "fc.limits()['ylim']"), "require": ["histogram", "set_height"]}),
        ("barh", "Horizontal version, sorted once by the **final** step: `order = np.argsort(probs[-1])`, `bars = "
                 "ax.barh(np.array(tokens)[order], probs[0][order])`, x limits 0 … 1; `update(t)` sets the widths to "
                 "`probs[t][order]` (`set_width`).",
         "fig, ax = plt.subplots(figsize=(3, 2))\norder = np.argsort(probs[-1])\nbars = ax.barh(np.array(tokens)[order], probs[0][order])\n"
         "ax.set_xlim(0, 1)\ndef update(t):\n    for b, p in zip(bars, probs[t][order]):\n        b.set_width(p)\n    return bars.patches",
         {"post": _post(f"_call(update, {ts}, lambda: fc.widths(fc.ax(), 4))", "fc.tick_labels(fc.ax(), 'y')"),
          "require": ["barh", "set_width"]}),
    ]
    return catalog(rng, items, setup, f"`probs` ({T} × {V}) holds a language model's next-token probabilities over "
                                      f"`tokens` at each decoding step; `acts` the activations of a layer per epoch and "
                                      f"`edges` fixed histogram bins. `update` is checked by calling it.")


@template(M, "medium")
def embedding_motion(rng):
    s = seeded(rng)
    T, n, k = rng.randint(5, 8), rng.randint(12, 24), rng.randint(2, 4)
    ts = sorted(rng.sample(range(1, T), 3))
    setup = (f"g = np.random.default_rng({s})\nlabels = g.integers(0, {k}, {n})\ncenters = g.uniform(-5, 5, ({k}, 2))\n"
             f"target = centers[labels] + g.normal(0, 0.5, ({n}, 2))\nstart = g.normal(0, 0.3, ({n}, 2))\n"
             f"traj = np.round(np.array([start + (target - start) * (t / {T - 1}) ** 0.5 for t in range({T})]), 3)   # {T} × {n} × 2")
    items = [
        ("set_offsets", "t-SNE-style animation: scatter `traj[0]` colored by `labels` (`cmap='tab10'`, `s=12`), fix the "
                        "limits to cover **all** frames (min/max of `traj[..., 0]` and `traj[..., 1]`, padded by 0.5), and "
                        "`update(t)` moves the points to `traj[t]`, returning `(sc,)`.",
         "fig, ax = plt.subplots(figsize=(3, 3))\nsc = ax.scatter(traj[0][:, 0], traj[0][:, 1], c=labels, cmap='tab10', s=12)\n"
         "ax.set_xlim(traj[..., 0].min() - 0.5, traj[..., 0].max() + 0.5)\nax.set_ylim(traj[..., 1].min() - 0.5, traj[..., 1].max() + 0.5)\n"
         "def update(t):\n    sc.set_offsets(traj[t])\n    return (sc,)",
         {"post": _post(f"_call(update, {ts}, lambda: fc.scatter(fc.ax(), 0, 6, 'array', 'cmap'))", "fc.limits()")}),
        ("set_title", "Same moving scatter (`sc`, colored by `labels`, `cmap='tab10'`); `update(t)` also sets the title "
                      "`f'iteration {t}/{len(traj) - 1}'` (no blitting).",
         "fig, ax = plt.subplots(figsize=(3, 3))\nsc = ax.scatter(traj[0][:, 0], traj[0][:, 1], c=labels, cmap='tab10')\n"
         "def update(t):\n    sc.set_offsets(traj[t])\n    ax.set_title(f'iteration {t}/{len(traj) - 1}')\n    return (sc,)",
         {"post": _post(f"_call(update, {ts}, lambda: [fc.scatter(), fc.ax().get_title()])")}),
        ("set_segments", "Trails: besides the scatter `sc` of `traj[0]`, a `LineCollection` `trails` (gray, empty at first: "
                         "`LineCollection([], colors='gray')`) whose `update(t)` sets one polyline per point covering its "
                         "last 3 positions (`traj[max(0, t - 2):t + 1, i]`) with `trails.set_segments(...)`; the scatter "
                         "moves to `traj[t]`. Return `(sc, trails)`.",
         "from matplotlib.collections import LineCollection\nfig, ax = plt.subplots(figsize=(3, 3))\n"
         "sc = ax.scatter(traj[0][:, 0], traj[0][:, 1])\ntrails = LineCollection([], colors='gray')\nax.add_collection(trails)\n"
         "def update(t):\n    sc.set_offsets(traj[t])\n    trails.set_segments([traj[max(0, t - 2):t + 1, i] for i in range(traj.shape[1])])\n"
         "    return (sc, trails)",
         {"post": _post(f"_call(update, {ts}, lambda: [fc.scatter(), [fc.r(x, 3) for c in fc.ax().collections[1:] for x in c.get_segments()]])"),
          "require": ["LineCollection", "set_segments"]}),
        ("plt.subplots", "Two panels: left, the moving scatter (`sc`, colored by `labels`); right, the mean distance of "
                         "the points to their final position per iteration, growing like a training curve (`dl, = "
                         "a2.plot([], [])`, x limits 0 … len(traj) - 1, y limits 0 … the first distance). `update(t)` "
                         "updates both and returns `(sc, dl)`.",
         "fig, (a1, a2) = plt.subplots(1, 2, figsize=(5, 2.5))\nsc = a1.scatter(traj[0][:, 0], traj[0][:, 1], c=labels)\n"
         "dl, = a2.plot([], [])\nd = np.linalg.norm(traj - traj[-1], axis=2).mean(axis=1)\na2.set_xlim(0, len(traj) - 1)\n"
         "a2.set_ylim(0, d[0])\ndef update(t):\n    sc.set_offsets(traj[t])\n    dl.set_data(range(t + 1), d[:t + 1])\n    return (sc, dl)",
         {"post": _post(f"_call(update, {ts}, lambda: [fc.scatter(fc.ax(0)), fc.lines(fc.ax(1), nd=4)])", "fc.limits(fc.ax(1))")}),
        ("set_array", "Animate k-means on the final embedding `P = traj[-1]`: `assign[t]` / `cents[t]` (given) are the "
                      "cluster of each point and the centroids at iteration t. Scatter `P` with `c=assign[0]`, "
                      "`cmap='tab10', vmin=0, vmax=9`, mark the centroids with `cx, = ax.plot(cents[0][:, 0], "
                      "cents[0][:, 1], 'kx', ms=10)`, and `update(t)` recolors the points (`sc.set_array`) and moves the "
                      "centroids, returning `(sc, cx)`.",
         "fig, ax = plt.subplots(figsize=(3, 3))\nsc = ax.scatter(P[:, 0], P[:, 1], c=assign[0], cmap='tab10', vmin=0, vmax=9)\n"
         "cx, = ax.plot(cents[0][:, 0], cents[0][:, 1], 'kx', ms=10)\ndef update(t):\n    sc.set_array(assign[t])\n"
         "    cx.set_data(cents[t][:, 0], cents[t][:, 1])\n    return (sc, cx)",
         {"post": _post("_call(update, [0, 1, 3], lambda: [fc.scatter(fc.ax(), 0, 6, 'array'), fc.lines(fc.ax(), nd=4)])"),
          "require": ["set_array", "set_data"],
          "setup": "P = traj[-1]\nC, assign = [P[:3].copy()], []\nfor it in range(4):\n"
                   "    a = np.argmin(((P[:, None] - C[-1][None]) ** 2).sum(-1), axis=1)\n    assign.append(a)\n"
                   "    C.append(np.array([P[a == j].mean(0) if (a == j).any() else C[-1][j] for j in range(3)]))\n"
                   "assign, cents = np.array(assign), np.array(C[:-1])   # 4 iterations of 3-means"}),
    ]
    return catalog(rng, items, setup, f"`traj` holds {T} snapshots of a 2-D embedding of {n} points ({k} classes in "
                                      f"`labels`) as it converges. `update(t)` is checked by calling it.")


@template(M, "medium")
def training_replay(rng):
    s = seeded(rng)
    E = rng.randint(5, 7)
    _, labs = T.labels(rng, min_classes=3)
    names = list(labs[:3])
    toks = rng.sample(["the", "cat", "sat", "on", "a", "mat", "it", "was"], 5)
    es = sorted(rng.sample(range(E), 3))
    setup = (f"g = np.random.default_rng({s})\nnames = {lit(names)}\ntrue = g.integers(0, 3, 60)\ncms = []\n"
             f"for e in range({E}):\n    pred = np.where(g.random(60) < 0.4 + 0.5 * e / {E - 1}, true, g.integers(0, 3, 60))\n"
             f"    cm = np.zeros((3, 3), dtype=int)\n    np.add.at(cm, (true, pred), 1)\n    cms.append(cm)\n"
             f"cms = np.array(cms)   # ({E}, 3, 3) confusion matrix after each epoch (rows: true)\n"
             f"tokens = {lit(toks)}\natt = np.round(g.dirichlet(np.ones(5), ({E}, 5)), 3)   # ({E}, 5, 5) attention per layer\n"
             f"Ws = np.round(g.normal(0, 1, (6, 6)) * np.linspace(0.2, 1, {E})[:, None, None], 3)   # a weight matrix per epoch\n"
             f"lr = np.round(3e-4 * np.minimum(np.arange(1, 101) / 10, 1) * np.cos(np.linspace(0, np.pi / 2, 100)), 8)\n"
             f"imgs = np.round(g.random(({E}, 6, 8, 8)), 3)\nlabels6 = g.integers(0, 3, 6)\n"
             f"guess = np.where(g.random(({E}, 6)) < np.linspace(0.3, 0.9, {E})[:, None], labels6, (labels6 + 1) % 3)")
    texts = "sorted((t['xy'], t['text']) for t in fc.text_items())"
    cm_head = ("fig, ax = plt.subplots(figsize=(3, 3))\nim = ax.imshow(cms[0], cmap='Blues', vmin=0, vmax=cms.max())\n"
               "texts = [[ax.text(j, i, '', ha='center', va='center') for j in range(3)] for i in range(3)]\n")
    items = [
        ("set_data", "Replay the confusion matrix over training: `im = ax.imshow(cms[0], cmap='Blues', vmin=0, "
                     "vmax=cms.max())` and a 3 × 3 grid of centered texts (`ax.text(j, i, '', ha='center', "
                     "va='center')`); `update(e)` shows `cms[e]` and writes each count in its cell.",
         cm_head + "def update(e):\n    im.set_data(cms[e])\n    for i in range(3):\n        for j in range(3):\n"
                   "            texts[i][j].set_text(str(cms[e][i, j]))\n    return [im, *sum(texts, [])]",
         {"post": _post(f"_call(update, {es}, lambda: [fc.images(fc.ax(), data=True), {texts}])")}),
        ("set_data", "Same replay, row-normalized (each row divided by its sum, so the diagonal is per-class recall): "
                     "`im = ax.imshow(np.zeros((3, 3)), cmap='Blues', vmin=0, vmax=1)`, centered cell texts formatted "
                     "`f'{v:.2f}'`; `update(e)` shows the normalized `cms[e]`.",
         "fig, ax = plt.subplots(figsize=(3, 3))\nim = ax.imshow(np.zeros((3, 3)), cmap='Blues', vmin=0, vmax=1)\n"
         "texts = [[ax.text(j, i, '', ha='center', va='center') for j in range(3)] for i in range(3)]\n"
         "def update(e):\n    norm = cms[e] / cms[e].sum(axis=1, keepdims=True)\n    im.set_data(norm)\n"
         "    for (i, j), v in np.ndenumerate(norm):\n        texts[i][j].set_text(f'{v:.2f}')\n    return [im, *sum(texts, [])]",
         {"post": _post(f"_call(update, {es}, lambda: [fc.images(fc.ax(), 4, data=True), {texts}])")}),
        ("set_title", "Attention through the layers: `im = ax.imshow(att[0], vmin=0, vmax=1)` with `tokens` as tick labels "
                      "on both axes (positions 0 … 4); `update(l)` shows `att[l]` and sets the title `f'layer {l}'`.",
         "fig, ax = plt.subplots(figsize=(3, 3))\nim = ax.imshow(att[0], vmin=0, vmax=1)\nax.set_xticks(range(5), tokens)\n"
         "ax.set_yticks(range(5), tokens)\ndef update(l):\n    im.set_data(att[l])\n    ax.set_title(f'layer {l}')\n    return (im,)",
         {"post": _post(f"_call(update, {es}, lambda: [fc.images(fc.ax(), data=True), fc.ax().get_title()])",
                        "fc.tick_labels(fc.ax(), 'x'), fc.tick_labels(fc.ax(), 'y')")}),
        (".colorbar", "Watch a weight matrix grow: one **fixed**, symmetric color range for all epochs — `a = "
                      "np.abs(Ws).max()`, `im = ax.imshow(Ws[0], cmap='RdBu_r', vmin=-a, vmax=a)` — a colorbar added "
                      "once (`fig.colorbar(im, ax=ax)`), and `update(e)` showing `Ws[e]`.",
         "fig, ax = plt.subplots(figsize=(3, 3))\na = np.abs(Ws).max()\nim = ax.imshow(Ws[0], cmap='RdBu_r', vmin=-a, vmax=a)\n"
         "fig.colorbar(im, ax=ax)\ndef update(e):\n    im.set_data(Ws[e])\n    return (im,)",
         {"post": _post(f"_call(update, {es}, lambda: fc.images(fc.ax(0), data=True))", "len(fc.colorbars())")}),
        ("set_text", "A cursor on the learning-rate schedule: plot `lr` against `range(100)`, add `dot, = ax.plot([], [], "
                     "'ro')` and `txt = ax.text(0.95, 0.9, '', transform=ax.transAxes, ha='right')`; `update(s)` puts the "
                     "dot on step s and writes `f'lr={lr[s]:.1e}'`.",
         "fig, ax = plt.subplots(figsize=(3, 2))\nax.plot(range(100), lr)\ndot, = ax.plot([], [], 'ro')\n"
         "txt = ax.text(0.95, 0.9, '', transform=ax.transAxes, ha='right')\ndef update(s):\n    dot.set_data([s], [lr[s]])\n"
         "    txt.set_text(f'lr={lr[s]:.1e}')\n    return (dot, txt)",
         {"post": _post("_call(update, [0, 9, 57, 99], lambda: [fc.lines(fc.ax(), nd=9), fc.texts()])")}),
        ("set_height", "Per-class recall bars over training: `bars = ax.bar(names, np.zeros(3))`, y limits 0 … 1; "
                       "`update(e)` sets the heights to `np.diag(cms[e]) / cms[e].sum(axis=1)`.",
         "fig, ax = plt.subplots(figsize=(3, 2))\nbars = ax.bar(names, np.zeros(3))\nax.set_ylim(0, 1)\ndef update(e):\n"
         "    for b, v in zip(bars, np.diag(cms[e]) / cms[e].sum(axis=1)):\n        b.set_height(v)\n    return bars.patches",
         {"post": _post(f"_call(update, {es}, lambda: fc.heights(fc.ax(), 4))", "fc.tick_labels(fc.ax(), 'x')")}),
        ("set_title", "A grid of predictions over epochs (`plt.subplots(2, 3)`): panel j shows `imgs[e][j]` (gray, axes "
                      "hidden) titled `f'{names[guess[e][j]]} ({names[labels6[j]]})'` in 'green' when the guess is right, "
                      "else 'red'; `update(e)` updates the six images and titles.",
         "fig, axes = plt.subplots(2, 3)\nims = []\nfor j, a in enumerate(axes.flat):\n    ims.append(a.imshow(imgs[0][j], cmap='gray'))\n"
         "    a.axis('off')\ndef update(e):\n    for j, a in enumerate(axes.flat):\n        ims[j].set_data(imgs[e][j])\n"
         "        ok = guess[e][j] == labels6[j]\n        a.set_title(f'{names[guess[e][j]]} ({names[labels6[j]]})', "
         "color='green' if ok else 'red')\n    return ims",
         {"post": _post(f"_call(update, {es}, lambda: [(a.get_title(), fc.hexcolor(a.title.get_color()), "
                        f"fc.images(a, 3, data=True)) for a in fc.axes()])")}),
    ]
    return catalog(rng, items, setup, f"A stored training history to replay ({E} epochs): `cms` confusion matrices over "
                                      f"`names`, `att` attention maps over `tokens` per layer, `Ws` a weight matrix per "
                                      f"epoch, `lr` a 100-step schedule, and `imgs` / `guess` six samples' images and "
                                      f"predicted classes (true classes `labels6`). `update` is checked by calling it.")


# ================================================================= hard

@template(M, "hard")
def animation_helpers(rng):
    code, desc, n = curves(rng)
    fps = rng.choice([5, 10])
    m = rng.randint(4, 7)
    items = [
        ("FuncAnimation", "Write `animate_curve(y, ax=None, interval=100)` that draws an empty line on `ax` (default "
                          "`plt.gca()`), fixes the limits to x 0 … len(y) - 1 and y `min(y)` … `max(y)`, and returns "
                          "`FuncAnimation(ax.figure, update, frames=len(y), interval=interval)` where `update(i)` shows "
                          "`y[:i + 1]` against `range(i + 1)`. (Checked by calling it and then its update function.)",
         "from matplotlib.animation import FuncAnimation\ndef animate_curve(y, ax=None, interval=100):\n    ax = ax or plt.gca()\n"
         "    line, = ax.plot([], [])\n    ax.set_xlim(0, len(y) - 1)\n    ax.set_ylim(min(y), max(y))\n"
         "    def update(i):\n        line.set_data(range(i + 1), y[:i + 1])\n        return (line,)\n"
         "    return FuncAnimation(ax.figure, update, frames=len(y), interval=interval)",
         {"post": PRE + "_f, _a = plt.subplots()\n_an = animate_curve(val_loss, ax=_a, interval=60)\n_an._func(4)\n"
                        "_seq(_an), _an._interval, fc.lines(_a), fc.limits(_a)", "require": ["FuncAnimation", "gca"]}),
        ("PillowWriter", f"Write `save_gif(fig, update, n_frames, path, fps=10, dpi=30)` that animates `update` over "
                         f"`n_frames` frames, saves a GIF with `PillowWriter`, closes the figure, and returns `path`. "
                         f"{GIF_NOTE}",
         "from matplotlib.animation import FuncAnimation, PillowWriter\ndef save_gif(fig, update, n_frames, path, fps=10, dpi=30):\n"
         "    FuncAnimation(fig, update, frames=n_frames).save(path, writer=PillowWriter(fps=fps), dpi=dpi)\n"
         "    plt.close(fig)\n    return path",
         {"post": PRE + f"_f, _a = plt.subplots(figsize=(2, 1.5))\n_l, = _a.plot(epochs, train_loss)\n"
                        f"def _u(i):\n    _l.set_ydata(train_loss * (1 + i / 10))\n    return (_l,)\n"
                        f"_p = save_gif(_f, _u, {m}, 'h.gif', fps={fps})\n_p, _gif('h.gif'), plt.fignum_exists(_f.number)",
          "require": ["PillowWriter", "close"]}),
        ("relim", "Write a class `LivePlot` for training scripts: `__init__(self, ax=None, label='loss')` makes one empty "
                  "line on `ax` (default `plt.gca()`) with that label and a legend; `update(self, step, value)` appends the "
                  "point, sets the line's data, calls `relim` and `autoscale_view`, and `draw_idle`s the canvas.",
         "class LivePlot:\n    def __init__(self, ax=None, label='loss'):\n        self.ax = ax or plt.gca()\n"
         "        self.line, = self.ax.plot([], [], label=label)\n        self.ax.legend()\n        self.xs, self.ys = [], []\n\n"
         "    def update(self, step, value):\n        self.xs.append(step)\n        self.ys.append(value)\n"
         "        self.line.set_data(self.xs, self.ys)\n        self.ax.relim()\n        self.ax.autoscale_view()\n"
         "        self.ax.figure.canvas.draw_idle()",
         {"post": "_f, _a = plt.subplots()\n_lp = LivePlot(_a, label='val')\nfor _s in range(0, len(val_loss), 3):\n"
                  "    _lp.update(_s, val_loss[_s])\nfc.lines(_a, 'label'), fc.legend(_a), fc.limits(_a)",
          "require": ["relim", "autoscale_view", "draw_idle"]}),
        ("grab_frame", f"Write `record(fig, draw, n, path, fps=10, dpi=30)` that records a GIF frame by frame: open "
                       f"`PillowWriter(fps=fps).saving(fig, path, dpi)`, and for k = 0 … n - 1 call `draw(k)` then "
                       f"`grab_frame()`. Return the number of frames grabbed.",
         "from matplotlib.animation import PillowWriter\ndef record(fig, draw, n, path, fps=10, dpi=30):\n"
         "    writer = PillowWriter(fps=fps)\n    count = 0\n    with writer.saving(fig, path, dpi):\n        for k in range(n):\n"
         "            draw(k)\n            writer.grab_frame()\n            count += 1\n    return count",
         {"post": PRE + f"_f, _a = plt.subplots(figsize=(2, 1.5))\n_l, = _a.plot(epochs, val_loss)\n"
                        f"_c = record(_f, lambda k: _l.set_ydata(val_loss + k), {m}, 'r.gif', fps={fps})\n_c, _gif('r.gif')",
          "require": ["saving", "grab_frame"]}),
        ("ArtistAnimation", "Write `frames_from_images(images, ax, **kw)` that returns one ArtistAnimation frame per image "
                            "(`[ax.imshow(img, animated=True, **kw)]`), and `image_anim(images, interval=200)` that makes "
                            "its own `fig, ax`, hides the axes and returns `ArtistAnimation(fig, frames_from_images(images, "
                            "ax, cmap='gray'), interval=interval)`.",
         "from matplotlib.animation import ArtistAnimation\ndef frames_from_images(images, ax, **kw):\n"
         "    return [[ax.imshow(img, animated=True, **kw)] for img in images]\n\ndef image_anim(images, interval=200):\n"
         "    fig, ax = plt.subplots()\n    ax.axis('off')\n"
         "    return ArtistAnimation(fig, frames_from_images(images, ax, cmap='gray'), interval=interval)",
         {"post": PRE + "_imgs = [np.outer(train_loss[:5], val_loss[:5]) * k for k in range(1, 4)]\n_an = image_anim(_imgs, interval=90)\n"
                        "_ax = _an._fig.axes[0]\n_seq(_an), _an._interval, _ax.axison, [i.get_cmap().name for i in _ax.get_images()]"}),
        ("contour", "Write `animate_path(path, loss_fn, ax=None)` for any optimizer trace: draw `ax.contour` of `loss_fn` "
                    "on a 40 × 40 grid spanning the path's min … max ± 0.5 on each axis (15 levels), an empty trail "
                    "`'ro-'`, and return a `FuncAnimation` over `len(path)` frames whose update shows `path[:i + 1]`.",
         "from matplotlib.animation import FuncAnimation\ndef animate_path(path, loss_fn, ax=None):\n    ax = ax or plt.gca()\n"
         "    lo, hi = path.min(axis=0) - 0.5, path.max(axis=0) + 0.5\n"
         "    A, B = np.meshgrid(np.linspace(lo[0], hi[0], 40), np.linspace(lo[1], hi[1], 40))\n"
         "    ax.contour(A, B, loss_fn(A, B), levels=15)\n    pt, = ax.plot([], [], 'ro-')\n    def update(i):\n"
         "        pt.set_data(path[:i + 1, 0], path[:i + 1, 1])\n        return (pt,)\n"
         "    return FuncAnimation(ax.figure, update, frames=len(path))",
         {"post": PRE + "_f, _a = plt.subplots()\n_p = np.array([[2.0, 2.0], [1.0, 1.2], [0.4, 0.5], [0.1, 0.1]])\n"
                        "_an = animate_path(_p, lambda u, v: u ** 2 + 3 * v ** 2, ax=_a)\n_an._func(2)\n"
                        "_seq(_an), fc.lines(_a, 'marker', 'color'), [type(c).__name__ for c in _a.collections], fc.limits(_a)",
          "require": ["contour", "FuncAnimation"]}),
        ("Image", "Write `gif_info(path)` that opens a GIF with Pillow (`from PIL import Image`) and returns `{'frames': "
                  "n_frames, 'size': [w, h], 'ms': [duration of each frame]}` (seek to each frame, read "
                  "`info['duration']`).",
         "from PIL import Image\ndef gif_info(path):\n    im = Image.open(path)\n    ms = []\n    for i in range(im.n_frames):\n"
         "        im.seek(i)\n        ms.append(im.info['duration'])\n    return {'frames': im.n_frames, 'size': list(im.size), 'ms': ms}",
         {"post": f"from matplotlib.animation import FuncAnimation, PillowWriter\n_f, _a = plt.subplots(figsize=(2, 1))\n"
                  f"_l, = _a.plot(epochs, train_loss)\ndef _u(k):\n    _l.set_ydata(train_loss * (k + 1))\n    return (_l,)\n"
                  f"FuncAnimation(_f, _u, frames=[0, 1, 1, 2, 3]).save('i.gif', writer=PillowWriter(fps={fps}), dpi=30)\n"
                  f"gif_info('i.gif')", "require": ["Image", "n_frames", "seek"]}),
    ]
    return catalog(rng, items, code, f"{desc}. Reusable animation helpers are checked by calling them. {IMPORT}")


@template(M, "hard")
def debug_animations(rng):
    code, n = _wave(rng)
    N = rng.randint(4, 7)
    fps = rng.choice([5, 10])
    setup = f"{code}\nfig, ax = plt.subplots(figsize=(2, 1.5))\nax.set_ylim(-1.5, 1.5)"
    imp = "from matplotlib.animation import FuncAnimation, PillowWriter\n"
    items = [
        (".set_ydata", "Each frame adds a **new** line instead of moving the old one, so the plot fills up:\n```python\n"
                       "def update(frame):\n    ax.plot(x, np.sin(x + frame / 4))\n```\nCreate one line up front "
                       "(`line, = ax.plot(x, np.sin(x))`) and make `update(frame)` move it and return `(line,)`.",
         "line, = ax.plot(x, np.sin(x))\ndef update(frame):\n    line.set_ydata(np.sin(x + frame / 4))\n    return (line,)",
         {"post": _post("_call(update, [0, 2, 5], lambda: fc.lines(fc.ax()))")}),
        ("FuncAnimation", f"With `blit=True` this crashes (*The animation function must return a sequence of Artist "
                          f"objects*):\n```python\nline, = ax.plot(x, np.sin(x))\ndef update(frame):\n    line.set_ydata("
                          f"np.sin(x + frame / 4))\nanim = FuncAnimation(fig, update, frames={N}, blit=True)\n```\nFix it.",
         f"from matplotlib.animation import FuncAnimation\nline, = ax.plot(x, np.sin(x))\ndef update(frame):\n"
         f"    line.set_ydata(np.sin(x + frame / 4))\n    return (line,)\nanim = FuncAnimation(fig, update, frames={N}, blit=True)",
         {"post": _post("_seq(anim)", "anim._blit", "_call(update, [3], lambda: fc.lines(fc.ax()))")}),
        (".plot", "This crashes with *'list' object has no attribute 'set_ydata'*:\n```python\nline = ax.plot(x, "
                  "np.cos(x))\ndef update(frame):\n    line.set_ydata(np.cos(x - frame / 3))\n    return (line,)\n```\n"
                  "`ax.plot` returns a list of lines. Fix it.",
         "line, = ax.plot(x, np.cos(x))\ndef update(frame):\n    line.set_ydata(np.cos(x - frame / 3))\n    return (line,)",
         {"post": _post("_call(update, [1, 4], lambda: fc.lines(fc.ax()))")}),
        (".set_data", "Animating a single point crashes with *x must be a sequence*:\n```python\ndot, = ax.plot([], [], "
                      "'ro')\ndef update(i):\n    dot.set_data(x[i], np.sin(x[i]))\n    return (dot,)\n```\nFix it.",
         "dot, = ax.plot([], [], 'ro')\ndef update(i):\n    dot.set_data([x[i]], [np.sin(x[i])])\n    return (dot,)",
         {"post": _post("_call(update, [0, 5, 9], lambda: fc.lines(fc.ax()))")}),
        (".relim", "The live plot's line runs off the top of the axes: the limits never follow the data.\n```python\n"
                   "line, = ax.plot([], [])\nys = []\nfor step in range(20):\n    ys.append(step ** 1.5)\n"
                   "    line.set_data(range(len(ys)), ys)\n```\nKeep the loop (on the given `ax`, whose y limits were set "
                   "by hand: undo that by letting autoscaling run) and make the view follow the data after each update.",
         "line, = ax.plot([], [])\nys = []\nfor step in range(20):\n    ys.append(step ** 1.5)\n    line.set_data(range(len(ys)), ys)\n"
         "    ax.relim()\n    ax.autoscale_view()",
         {"post": "fc.lines(fc.ax(), nd=4), fc.limits()", "require": ["relim", "autoscale_view"]}),
        ("PillowWriter", f"This fails in the browser: MP4 needs ffmpeg, which isn't available.\n```python\nline, = "
                         f"ax.plot(x, np.sin(x))\ndef update(f):\n    line.set_ydata(np.sin(x + f / 4))\n    return (line,)\n"
                         f"FuncAnimation(fig, update, frames={N}).save('wave.mp4', fps={fps})\n```\nSave a GIF instead: "
                         f"`wave.gif` with `PillowWriter(fps={fps})`, `dpi=30`.",
         f"{imp}line, = ax.plot(x, np.sin(x))\ndef update(f):\n    line.set_ydata(np.sin(x + f / 4))\n    return (line,)\n"
         f"FuncAnimation(fig, update, frames={N}).save('wave.gif', writer=PillowWriter(fps={fps}), dpi=30)",
         {"post": _post("_gif('wave.gif')")}),
        ("ArtistAnimation", f"This crashes with *'AxesImage' object is not iterable*:\n```python\nframes = [ax.imshow("
                            f"np.random.default_rng(k).random((5, 5)), animated=True) for k in range({N})]\n"
                            f"anim = ArtistAnimation(fig, frames)\n```\nEach frame must be a **list** of artists. Fix it.",
         f"from matplotlib.animation import ArtistAnimation\nframes = [[ax.imshow(np.random.default_rng(k).random((5, 5)), "
         f"animated=True)] for k in range({N})]\nanim = ArtistAnimation(fig, frames)",
         {"post": _post("_seq(anim)")}),
        ("FuncAnimation", f"The saved GIF has a single frame although {N} were rendered — Pillow merges identical "
                          f"consecutive frames, and the update ignores its argument:\n```python\nline, = ax.plot(x, "
                          f"np.sin(x))\nphase = 0\ndef update(frame):\n    line.set_ydata(np.sin(x + phase))\n"
                          f"    return (line,)\nFuncAnimation(fig, update, frames={N}).save('w.gif', "
                          f"writer=PillowWriter(fps={fps}), dpi=30)\n```\nUse the frame number (shift by `frame / 4`).",
         f"{imp}line, = ax.plot(x, np.sin(x))\ndef update(frame):\n    line.set_ydata(np.sin(x + frame / 4))\n    return (line,)\n"
         f"FuncAnimation(fig, update, frames={N}).save('w.gif', writer=PillowWriter(fps={fps}), dpi=30)",
         {"post": _post("_gif('w.gif')")}),
        ("FuncAnimation", f"This crashes with an *IndexError* on the last frame:\n```python\nys = [np.sin(x + k / 4) for k "
                          f"in range({N})]\nline, = ax.plot(x, ys[0])\ndef update(i):\n    line.set_ydata(ys[i + 1])\n"
                          f"    return (line,)\nanim = FuncAnimation(fig, update, frames=len(ys))\n```\nShow `ys[i]` on "
                          f"frame i instead, keeping `frames=len(ys)`.",
         f"from matplotlib.animation import FuncAnimation\nys = [np.sin(x + k / 4) for k in range({N})]\nline, = ax.plot(x, ys[0])\n"
         f"def update(i):\n    line.set_ydata(ys[i])\n    return (line,)\nanim = FuncAnimation(fig, update, frames=len(ys))",
         {"post": _post("_seq(anim)", f"_call(update, [0, {N - 1}], lambda: fc.lines(fc.ax()))")}),
        ("transAxes", f"With blitting only the returned artists are redrawn, and the title sits outside the Axes, so it "
                      f"never changes on screen:\n```python\nline, = ax.plot(x, np.sin(x))\ndef update(frame):\n"
                      f"    line.set_ydata(np.sin(x + frame / 4))\n    ax.set_title(f'frame {{frame}}')\n    return (line,)\n"
                      f"anim = FuncAnimation(fig, update, frames={N}, blit=True)\n```\nKeep `blit=True`; show "
                      f"`f'frame {{frame}}'` in a text **inside** the Axes instead (`txt = ax.text(0.02, 0.9, '', "
                      f"transform=ax.transAxes)`) and return it too.",
         f"from matplotlib.animation import FuncAnimation\nline, = ax.plot(x, np.sin(x))\ntxt = ax.text(0.02, 0.9, '', transform=ax.transAxes)\n"
         f"def update(frame):\n    line.set_ydata(np.sin(x + frame / 4))\n    txt.set_text(f'frame {{frame}}')\n    return (line, txt)\n"
         f"anim = FuncAnimation(fig, update, frames={N}, blit=True)",
         {"post": _post("anim._blit", "_call(update, [2, 3], lambda: [fc.texts(), fc.lines(fc.ax())])"), "require": ["transAxes", "set_text"]}),
        (".set_data", "This live plot piles up a new line (and a new legend entry) every update:\n```python\n"
                      "for step in range(0, 30, 5):\n    ax.plot(range(step + 1), np.sqrt(np.arange(step + 1)), color='C0', "
                      "label='loss')\n    ax.legend()\n```\nCreate one labelled line and a legend before the loop and "
                      "update the line (relim + autoscale) instead.",
         "line, = ax.plot([], [], color='C0', label='loss')\nax.legend()\nfor step in range(0, 30, 5):\n"
         "    line.set_data(range(step + 1), np.sqrt(np.arange(step + 1)))\n    ax.relim()\n    ax.autoscale_view()",
         {"post": "fc.lines(fc.ax(), 'label', nd=4), fc.legend()", "require": ["set_data"]}),
        ("ArtistAnimation", f"The GIF has a single frame: every frame holds the **same** image object, which ends up "
                            f"showing the last data, so all frames are identical (and Pillow merges them):\n```python\n"
                            f"im = ax.imshow(np.zeros((5, 5)), vmin=0, vmax=1)\nframes = []\nfor k in range({N}):\n"
                            f"    im.set_data(np.random.default_rng(k).random((5, 5)))\n    frames.append([im])\n"
                            f"ArtistAnimation(fig, frames).save('f.gif', writer=PillowWriter(fps={fps}), dpi=30)\n```\n"
                            f"Give each frame its own `imshow` artist.",
         f"from matplotlib.animation import ArtistAnimation, PillowWriter\nframes = []\nfor k in range({N}):\n"
         f"    frames.append([ax.imshow(np.random.default_rng(k).random((5, 5)), vmin=0, vmax=1)])\n"
         f"ArtistAnimation(fig, frames).save('f.gif', writer=PillowWriter(fps={fps}), dpi=30)",
         {"post": _post("_gif('f.gif')")}),
    ]
    return catalog(rng, items, setup, f"`x` holds {n} points and `fig, ax` a small empty plot. Debug the animation code "
                                      f"(add the imports from `matplotlib.animation` you need).")


@template(M, "hard")
def animation_specs(rng):
    code, desc, n = curves(rng, acc=True)
    s = seeded(rng)
    ks = sorted(rng.sample(range(n), 3))
    setup = code + f"\nimgs = np.round(np.random.default_rng({s}).random(({n}, 4, 8, 8)), 3)   # 4 feature maps per epoch"
    items = [
        ("plt.subplots", "Three stacked panels sharing x (`plt.subplots(3, 1, sharex=True, figsize=(3, 4))`) for "
                         "`train_loss`, `val_loss`, `val_acc`, one empty line each, x limits 1 … `epochs[-1]` and each "
                         "panel's y limits from its full series' min to max; `update(i)` shows the first i + 1 epochs on "
                         "all three and returns the three lines.",
         "fig, axes = plt.subplots(3, 1, sharex=True, figsize=(3, 4))\nseries = [train_loss, val_loss, val_acc]\nlines = []\n"
         "for a, s in zip(axes, series):\n    lines.append(a.plot([], [])[0])\n    a.set_xlim(1, epochs[-1])\n"
         "    a.set_ylim(s.min(), s.max())\ndef update(i):\n    for l, s in zip(lines, series):\n"
         "        l.set_data(epochs[:i + 1], s[:i + 1])\n    return lines",
         {"post": _post(f"_call(update, {ks}, lambda: [fc.lines(a, nd=4) for a in fc.axes()])", "[fc.limits(a) for a in fc.axes()]")}),
        ("imshow", "A 2 × 2 grid of feature maps (`plt.subplots(2, 2, figsize=(3, 3))`): one `imshow` per panel "
                   "(`vmin=0, vmax=1`, axes hidden), and `update(i)` swaps in `imgs[i][j]` for panel j "
                   "(`im.set_data`), returning the four images.",
         "fig, axes = plt.subplots(2, 2, figsize=(3, 3))\nims = []\nfor j, a in enumerate(axes.flat):\n"
         "    ims.append(a.imshow(imgs[0][j], vmin=0, vmax=1))\n    a.axis('off')\ndef update(i):\n"
         "    for j, im in enumerate(ims):\n        im.set_data(imgs[i][j])\n    return ims",
         {"post": _post(f"_call(update, {ks}, lambda: [fc.images(a, 3, data=True) for a in fc.axes()])", "[a.axison for a in fc.axes()]")}),
        ("axvline", "Linked views (`plt.subplots(1, 2, figsize=(5, 2))`): left, the full `val_loss` curve with a gray "
                    "cursor `cur = a1.axvline(epochs[0], color='gray')`; right, `im = a2.imshow(imgs[0][0], vmin=0, "
                    "vmax=1)`. `update(i)` moves the cursor to `epochs[i]` and shows `imgs[i][0]`.",
         "fig, (a1, a2) = plt.subplots(1, 2, figsize=(5, 2))\na1.plot(epochs, val_loss)\ncur = a1.axvline(epochs[0], color='gray')\n"
         "im = a2.imshow(imgs[0][0], vmin=0, vmax=1)\ndef update(i):\n    cur.set_xdata([epochs[i], epochs[i]])\n"
         "    im.set_data(imgs[i][0])\n    return (cur, im)",
         {"post": _post(f"_call(update, {ks}, lambda: [fc.lines(fc.ax(0), nd=4), fc.images(fc.ax(1), 3, data=True)])")}),
        ("twinx", "Loss and accuracy on twin y-axes, growing together: `l1, = ax.plot([], [], color='C0')`, `ax2 = "
                  "ax.twinx()`, `l2, = ax2.plot([], [], color='C1')`; limits x 1 … `epochs[-1]`, `ax` y 0 … "
                  "`train_loss.max()`, `ax2` y 0 … 1; `update(i)` shows the first i + 1 epochs of `train_loss` / "
                  "`val_acc`.",
         "fig, ax = plt.subplots(figsize=(3, 2))\nl1, = ax.plot([], [], color='C0')\nax2 = ax.twinx()\nl2, = ax2.plot([], [], color='C1')\n"
         "ax.set_xlim(1, epochs[-1])\nax.set_ylim(0, train_loss.max())\nax2.set_ylim(0, 1)\ndef update(i):\n"
         "    l1.set_data(epochs[:i + 1], train_loss[:i + 1])\n    l2.set_data(epochs[:i + 1], val_acc[:i + 1])\n    return (l1, l2)",
         {"post": _post(f"_call(update, {ks}, lambda: [fc.lines(a, 'color', nd=4) for a in fc.axes()])", "[fc.limits(a)['ylim'] for a in fc.axes()]")}),
        ("FuncAnimation", "Animate the training curves and save them: lines 'train' and 'val' (legend) growing over all "
                          "epochs, limits x 1 … `epochs[-1]`, y 0 … `max(train_loss.max(), val_loss.max())`, as "
                          "`FuncAnimation(fig, update, frames=range(0, len(epochs), 4))` saved to `curves.gif` "
                          "(`PillowWriter(fps=10)`, `dpi=20`, figure 2 × 1.5 inches).",
         "from matplotlib.animation import FuncAnimation, PillowWriter\nfig, ax = plt.subplots(figsize=(2, 1.5))\n"
         "lt, = ax.plot([], [], label='train')\nlv, = ax.plot([], [], label='val')\nax.legend()\nax.set_xlim(1, epochs[-1])\n"
         "ax.set_ylim(0, max(train_loss.max(), val_loss.max()))\ndef update(i):\n    lt.set_data(epochs[:i + 1], train_loss[:i + 1])\n"
         "    lv.set_data(epochs[:i + 1], val_loss[:i + 1])\n    return (lt, lv)\n"
         "FuncAnimation(fig, update, frames=range(0, len(epochs), 4)).save('curves.gif', writer=PillowWriter(fps=10), dpi=20)",
         {"post": _post("_gif('curves.gif')", "fc.legend()")}),
        ("histogram", "An image and its pixel histogram, frame by frame (`plt.subplots(1, 2, figsize=(5, 2))`): left `im = "
                      "a1.imshow(imgs[0][0], vmin=0, vmax=1)`; right bars over the fixed bins `edges = np.linspace(0, 1, "
                      "9)` (`a2.bar(edges[:-1], counts, width=0.125, align='edge')`), y limits 0 … 64. `update(i)` shows "
                      "`imgs[i][0]` and sets the bar heights to its `np.histogram` counts.",
         "fig, (a1, a2) = plt.subplots(1, 2, figsize=(5, 2))\nim = a1.imshow(imgs[0][0], vmin=0, vmax=1)\nedges = np.linspace(0, 1, 9)\n"
         "counts, _ = np.histogram(imgs[0][0], bins=edges)\nbars = a2.bar(edges[:-1], counts, width=0.125, align='edge')\n"
         "a2.set_ylim(0, 64)\ndef update(i):\n    im.set_data(imgs[i][0])\n    counts, _ = np.histogram(imgs[i][0], bins=edges)\n"
         "    for b, c in zip(bars, counts):\n        b.set_height(c)\n    return [im, *bars.patches]",
         {"post": _post(f"_call(update, {ks}, lambda: [fc.images(fc.ax(0), 3, data=True), fc.bars(fc.ax(1), 4)])", "fc.limits(fc.ax(1))['ylim']"),
          "require": ["histogram", "set_height"]}),
        ("alpha", "A fading trail on the validation curve: draw `val_loss` in 'lightgray', then five markers `[ax.plot([], "
                  "[], 'o', color='C3', alpha=(j + 1) / 5)[0] for j in range(5)]`; `update(i)` puts marker j on epoch "
                  "index `max(0, i - 4 + j)` (the newest, most opaque one on i) and returns the five markers.",
         "fig, ax = plt.subplots(figsize=(3, 2))\nax.plot(epochs, val_loss, color='lightgray')\n"
         "trail = [ax.plot([], [], 'o', color='C3', alpha=(j + 1) / 5)[0] for j in range(5)]\ndef update(i):\n"
         "    for j, d in enumerate(trail):\n        k = max(0, i - 4 + j)\n        d.set_data([epochs[k]], [val_loss[k]])\n    return trail",
         {"post": _post(f"_call(update, {ks}, lambda: fc.lines(fc.ax(), 'alpha', 'color', nd=4))")}),
    ]
    return catalog(rng, items, setup, f"{desc}; `imgs[i]` holds 4 feature maps (8 × 8) at epoch i. Name the update "
                                      f"function `update`; it's checked by calling it.")


@template(M, "hard")
def frame_schedules(rng):
    code, n = _wave(rng)
    N = rng.randint(30, 80)
    k = rng.randint(6, 12)
    hold = rng.randint(2, 4)
    m = rng.randint(4, 6)
    fps = rng.choice([5, 10])
    setup = (f"{code}\nfig, ax = plt.subplots(figsize=(2, 1.5))\nline, = ax.plot(x, np.sin(x))\nax.set_ylim(-1.2, 1.2)\n"
             f"def update(frame):\n    line.set_ydata(np.sin(x + frame / 4))\n    return (line,)")
    imp = "from matplotlib.animation import FuncAnimation, PillowWriter\n"
    items = [
        ("FuncAnimation", f"Show a {N}-step run in about {k} frames: frames every `{N} // {k}` steps — `range(0, {N}, "
                          f"{N} // {k})` — **plus the last step {N - 1}** if it's missing. Create `anim` with that frame list.",
         f"from matplotlib.animation import FuncAnimation\nfr = list(range(0, {N}, {N} // {k}))\nif fr[-1] != {N - 1}:\n"
         f"    fr.append({N - 1})\nanim = FuncAnimation(fig, update, frames=fr)", {"post": _post("_seq(anim)")}),
        ("PillowWriter", f"Hold the final frame: animate frames `list(range({m})) + [{m - 1}] * {hold}` and save "
                         f"`hold.gif` (`PillowWriter(fps={fps})`, `dpi=20`). Pillow merges identical consecutive frames, "
                         f"so the GIF has {m} frames and the last one lasts longer — return the per-frame durations "
                         f"read back with Pillow.",
         f"{imp}from PIL import Image\nanim = FuncAnimation(fig, update, frames=list(range({m})) + [{m - 1}] * {hold})\n"
         f"anim.save('hold.gif', writer=PillowWriter(fps={fps}), dpi=20)\nim = Image.open('hold.gif')\nd = []\n"
         f"for i in range(im.n_frames):\n    im.seek(i)\n    d.append(im.info['duration'])\nresult = d",
         {"require": ["PillowWriter", "seek"]}),
        ("FuncAnimation", f"Ping-pong playback: frames 0, 1, …, {m - 1} and back down to 1 (so the loop restarts "
                          f"smoothly at 0). Create `anim` with those frames.",
         f"from matplotlib.animation import FuncAnimation\nanim = FuncAnimation(fig, update, frames=list(range({m})) + "
         f"list(range({m - 2}, 0, -1)))", {"post": _post("_seq(anim)")}),
        ("FuncAnimation", f"Ease in and out: {k} frames whose values follow `{N - 1} * (1 - np.cos(np.pi * t)) / 2` for "
                          f"`t = np.linspace(0, 1, {k})`, rounded to ints (`np.round(...).astype(int)`). Create `anim` "
                          f"with those frame values.",
         f"from matplotlib.animation import FuncAnimation\nt = np.linspace(0, 1, {k})\n"
         f"fr = np.round({N - 1} * (1 - np.cos(np.pi * t)) / 2).astype(int)\nanim = FuncAnimation(fig, update, frames=fr)",
         {"post": _post("_seq(anim)")}),
        ("FuncAnimation", f"Make a {m}-frame animation that plays at {fps} fps on screen **and** in the saved GIF: "
                          f"`interval=1000 / {fps}` for the live player, `PillowWriter(fps={fps})` for `sync.gif` "
                          f"(`dpi=20`). Return `[anim._interval, the first frame's GIF duration]`.",
         f"{imp}from PIL import Image\nanim = FuncAnimation(fig, update, frames={m}, interval=1000 / {fps})\n"
         f"anim.save('sync.gif', writer=PillowWriter(fps={fps}), dpi=20)\nresult = [anim._interval, Image.open('sync.gif').info['duration']]",
         {"require": ["interval", "PillowWriter"]}),
        ("FuncAnimation", f"Reverse playback without touching `update`: frames {N - 1}, {N - 2}, …, 0 in steps of {k} "
                          f"(`range({N - 1}, -1, -{k})`). Create `anim` and also return its frame list as `result`.",
         f"from matplotlib.animation import FuncAnimation\nanim = FuncAnimation(fig, update, frames=range({N - 1}, -1, -{k}))\n"
         f"result = list(anim.new_frame_seq())", {"post": _post("_seq(anim)", "result")}),
        ("interval", f"Fast-forward without changing the speed of motion: the full animation would be {N} frames at 40 "
                     f"ms. Show only every {k}-th frame (`range(0, {N}, {k})`) and stretch the interval so it still "
                     f"lasts as long (`40 * {k}` ms).",
         f"from matplotlib.animation import FuncAnimation\nanim = FuncAnimation(fig, update, frames=range(0, {N}, {k}), interval=40 * {k})",
         {"post": _post("_seq(anim)", "anim._interval"), "require": ["FuncAnimation", "interval"]}),
    ]
    return catalog(rng, items, setup, f"`fig, ax` show a wave `line` over `x` and `update(frame)` shifts it by "
                                      f"`frame / 4`. Choose the frames. {IMPORT}")


# ================================================================= challenge

@template(M, "challenge")
def gif_timeline(rng):
    n, q, s = rng.randint(8, 14), rng.randint(4, 6), seeded(rng)
    report = rng.choice(["durations", "shown", "longest"])
    fps = rng.choice([3, 6, 7, 8, 12, 15, 30])

    def gen(seed, n_, q_, kinds):
        return (f"g = np.random.default_rng({seed})\nframe_ids = np.cumsum(g.random({n_}) < 0.6) % {kinds}   # content of each rendered frame\n"
                f"fps = {fps}\nt = g.integers(0, 10 ** 6, {q_})   # playback times in ms\nreport = {report!r}")
    sol = ("d = int(1000 / fps)   # ms per rendered frame (PillowWriter)\n"
           "start = np.flatnonzero(np.r_[True, frame_ids[1:] != frame_ids[:-1]])   # first frame of each run\n"
           "run = np.diff(np.r_[start, len(frame_ids)])\ndur = (run * d) // 10 * 10   # merged, stored in 1/100 s\n"
           + {"durations": "result = dur",
              "shown": "ends = np.cumsum(dur)\nresult = np.searchsorted(ends, t % ends[-1], side='right')",
              "longest": "result = [int(dur.max()), int(dur.argmax())]"}[report])
    want = {"durations": "the duration in ms of each frame of the saved GIF (an int array)",
            "shown": "for every playback time `t[j]` (ms since the GIF started; it loops forever), the index of the GIF "
                     "frame on screen (frame i shows from the sum of the earlier durations up to, not including, that "
                     "plus its own)",
            "longest": "`[duration, index]` of the longest-displayed GIF frame (ties: the first)"}[report]
    return challenge(
        "What does the GIF actually play?",
        "`anim.save(..., writer=PillowWriter(fps=fps))` renders frames whose contents are `frame_ids` (equal ids = "
        "pixel-identical frames) and gives each `int(1000 / fps)` ms. Pillow then **merges consecutive identical "
        "frames** into one, summing their durations, and the GIF format stores a duration in hundredths of a second "
        f"(rounded down: 99 ms → 90 ms). Return {want}. (`report` = {report!r}.)",
        gen(s, n, q, 3), sol, [gen(s + 1, 200000, 200000, 4), gen(s + 2, 150000, 100000, 2)],
        "n ≤ 2·10^5 frames, q ≤ 2·10^5 queries", speed="O((n + q) log n): run lengths, prefix sums, binary search",
        answer_desc=" (a list of two ints)" if report == "longest" else "")


@template(M, "challenge")
def live_window_autoscale(rng):
    n, s = rng.randint(12, 20), seeded(rng)
    report = rng.choice(["ylim", "height", "changes"])
    W, k = rng.randint(3, 6), rng.randint(2, 3)

    def gen(seed, n_, W_, k_):
        return (f"g = np.random.default_rng({seed})\nloss = np.round(np.exp(-np.arange({n_}) / {max(n_ // 3, 1)}) + "
                f"0.3 * g.random({n_}), 4)\nwindow, every = {W_}, {k_}\nreport = {report!r}")
    sol = ("from collections import deque\nlo, hi = [], []\nqmin, qmax = deque(), deque()   # monotonic queues of indices\n"
           "for i, v in enumerate(loss.tolist()):\n    while qmin and loss[qmin[-1]] >= v:\n        qmin.pop()\n"
           "    while qmax and loss[qmax[-1]] <= v:\n        qmax.pop()\n    qmin.append(i)\n    qmax.append(i)\n"
           "    while qmin[0] <= i - window:\n        qmin.popleft()\n    while qmax[0] <= i - window:\n        qmax.popleft()\n"
           "    if (i + 1) % every == 0:\n        lo.append(loss[qmin[0]])\n        hi.append(loss[qmax[0]])\n"
           "lo, hi = np.array(lo), np.array(hi)\nm = 0.05 * (hi - lo)   # axes.ymargin\n"
           + {"ylim": "result = np.column_stack([lo - m, hi + m])",
              "height": "result = hi - lo + 2 * m",
              "changes": "result = int(((lo[1:] != lo[:-1]) | (hi[1:] != hi[:-1])).sum())"}[report])
    want = {"ylim": "the y limits `(bottom, top)` after every update, as a u × 2 array",
            "height": "the height `top - bottom` of the y-axis after every update",
            "changes": "at how many updates (after the first) the y limits differ from the previous update's"}[report]
    return challenge(
        "A live plot's sliding window",
        "A training script logs `loss[i]` at step i (0, 1, 2, …). Every `every` steps — after step i whenever (i + 1) "
        "is a multiple of `every` — it redraws a live plot showing only the last `window` losses (steps i - window + 1 "
        "… i, fewer at the start), then calls `ax.relim()` and `ax.autoscale_view()`: the y-axis spans that window's "
        "min to max plus a 5% margin of the span on each side (`every` ≥ 2 and `window` ≥ 2, so a window is never a single point; values in a window are never all equal). Return "
        f"{want}. (`report` = {report!r}.)",
        gen(s, n, W, k), sol, [gen(s + 1, 200000, 5000, 3), gen(s + 2, 200000, 70000, 2)],
        "n ≤ 2·10^5 steps, window ≤ n", speed="O(n): monotonic queues (or a sparse table) for the sliding min / max",
        answer_desc=" (an int)" if report == "changes" else "", tol=1e-6)


@template(M, "challenge")
def live_decimation(rng):
    n, s = rng.randint(12, 22), seeded(rng)
    report = rng.choice(["size", "envelope", "path"])
    B = rng.randint(3, 5)

    def gen(seed, n_, B_):
        return (f"g = np.random.default_rng({seed})\nloss = np.round(np.exp(-np.arange({n_}) / {max(n_ // 2, 1)}) + "
                f"0.2 * g.random({n_}), 4)\nB = {B_}\nreport = {report!r}")
    sol = ("n = len(loss)\nsize = 1\nwhile -(-n // size) > B:   # buckets double until at most B are needed\n    size *= 2\n"
           "starts = np.arange(0, n, size)\n"
           + {"size": "result = [size, len(starts)]",
              "envelope": "result = np.column_stack([np.minimum.reduceat(loss, starts), np.maximum.reduceat(loss, starts)])",
              "path": ("pts = []\nfor a in starts.tolist():\n    seg = loss[a:a + size]\n"
                       "    i, j = a + int(seg.argmin()), a + int(seg.argmax())\n    for k in sorted({i, j}):\n"
                       "        pts.append((k, loss[k]))\nresult = np.array(pts)")}[report])
    want = {"size": "`[bucket size, number of buckets]` at the end",
            "envelope": "the `(min, max)` of each bucket, in order (a buckets × 2 array)",
            "path": "the vertices the plot draws: for each bucket in order, the point (step, loss) of its minimum and "
                    "of its maximum (first occurrence each), in step order — once if they are the same point (a "
                    "v × 2 array)"}[report]
    return challenge(
        "A live loss plot that never gets slower",
        "A live plot receives `loss[0]`, `loss[1]`, … one value per step and must keep at most `B` buckets. Buckets "
        "cover consecutive steps; all have the same capacity (starting at 1 step) except the last, which may be "
        "partly filled. A new value goes into the last bucket if it has room, else into a new bucket. Whenever that "
        "makes more than `B` buckets, adjacent buckets are merged in pairs (1st+2nd, 3rd+4th, …; an odd one out at the "
        "end stays alone) and the capacity doubles. Each bucket remembers the min and max of its values. After all "
        f"values arrived, return {want}. (`report` = {report!r}.)",
        gen(s, n, B), sol, [gen(s + 1, 200000, 1000), gen(s + 2, 150000, 7)], "n ≤ 2·10^5 values, B ≤ 1000",
        speed="O(n): work out the final bucket layout directly instead of simulating every merge", tol=1e-9)
