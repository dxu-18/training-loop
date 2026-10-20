"""1.06 Views, copies & memory layout.

Many answers here are checked with `post` code that inspects memory, not just values,
so that e.g. a copy can't pass for a view.
"""

from ._common import P, about, arr, challenge, rand_ints, seeded, series, template

M = "1.06"
SWV = "from numpy.lib.stride_tricks import sliding_window_view\n"

VIEW_OR_COPY = (("a[2:5]", "a slice"), ("a[::2]", "a strided slice"), ("a[[0, 2, 4]]", "fancy indexing"),
                ("a.reshape(2, -1)", "reshape"), ("a[a > 3]", "a boolean mask"), ("a.copy()", "copy"),
                ("a.ravel()", "ravel"), ("a.flatten()", "flatten"), ("a[::-1]", "a reversed slice"),
                ("np.sort(a)", "np.sort"), ("a.astype(float)", "astype"), ("a.view()", "view"))


# ================================================================= lite

@template(M, "lite")
def view_or_copy(rng):
    n = rng.choice([6, 8, 10, 12])
    expr, how = rng.choice(VIEW_OR_COPY)
    return P(f"`v` was made from `a` with {how}: `v = {expr}`. Use `np.shares_memory` to check whether `v` "
             f"shares memory with `a` (a **view**) or not (a **copy**). Return the bool.",
             f"a = np.arange({n})\nv = {expr}", "np.shares_memory(v, a)", require=["shares_memory"])


@template(M, "lite")
def independent_copy(rng):
    th, m, _, vals = series(rng, 7, 10)
    v = m.var
    i = rng.randint(0, 3)
    j = rng.randint(i + 2, len(vals))
    return P(f"{about(th, m, len(vals), v)}. Take positions {i} through {j - 1} as an **independent copy** "
             f"named `part` (so writing to it can't change `{v}`), then set `part[0] = 0`.",
             f"{v} = {arr(vals)}", f"part = {v}[{i}:{j}].copy()\npart[0] = 0",
             post=f"{{'{v} unchanged': bool(np.array_equal({v}, {arr(vals)})), 'part': part}}",
             require=["copy|array"])


# ================================================================= easy

@template(M, "easy")
def write_through_view(rng):
    th, m, _, vals = series(rng, 7, 10)
    v = m.var
    if rng.random() < 0.5:
        i = rng.randint(0, 3)
        j = rng.randint(i + 2, len(vals))
        k = rng.randrange(j - i)
        val = -1
        return P(f"{about(th, m, len(vals), v)}. `window = {v}[{i}:{j}]` is a **view**. Set `window[{k}] = {val}` "
                 f"through the view, then return `{v}` to see that the original changed too.",
                 f"{v} = {arr(vals)}\nwindow = {v}[{i}:{j}]", f"window[{k}] = {val}\nresult = {v}")
    return P(f"{about(th, m, len(vals), v)}. `evens = {v}[::2]` is a **view** of every second value. Set all "
             f"of them to 0 through the view (`evens[:] = 0`), then return `{v}`.",
             f"{v} = {arr(vals)}\nevens = {v}[::2]", f"evens[:] = 0\nresult = {v}")


@template(M, "easy")
def base_and_flags(rng):
    r, c = rng.randint(2, 4), rng.randint(3, 5)
    setup = f"M = np.arange({r * c}).reshape({r}, {c})"
    desc, sol, req = rng.choice([
        ("Is `v.base` the array `M` itself? (`v = M[1:]`). Use the `.base` attribute and `is`",
         "v = M[1:]\nresult = v.base is M", "base"),
        ("Is `M.T` C-contiguous (row-major)? Read `flags['C_CONTIGUOUS']`", "M.T.flags['C_CONTIGUOUS']", "flags"),
        ("Is `M` C-contiguous (row-major)? Read `flags['C_CONTIGUOUS']`", "M.flags['C_CONTIGUOUS']", "flags"),
        ("What are the **strides** of `M.T`, in bytes per step along each axis?", "M.T.strides", "strides"),
        ("Does `M[::2]` own its data? Read `flags['OWNDATA']`", "M[::2].flags['OWNDATA']", "flags"),
        ("Does `M.copy()` own its data? Read `flags['OWNDATA']`", "M.copy().flags['OWNDATA']", "flags"),
    ])
    return P(f"`M` is a `({r}, {c})` int64 matrix. {desc}.", setup, sol, require=[req])


@template(M, "easy")
def in_place_update(rng):
    th, m, _, vals = series(rng, 5, 8, positive=True)
    v = m.var
    vals = [float(x) for x in vals]
    s, b = rng.choice([2, 0.5, 10, 0.1]), rng.choice([1.0, -3.0, 100.0])
    lo, hi = sorted(rng.sample(vals, 2))
    desc, sol = rng.choice([
        (f"multiply every value by {s} **in place** (`*=` or `np.multiply(..., out=...)`)", f"{v} *= {s}"),
        (f"add {b} to every value **in place**", f"{v} += {b}"),
        (f"clip the values to [{lo}, {hi}] **in place** with `np.clip(..., out={v})`",
         f"np.clip({v}, {lo}, {hi}, out={v})"),
        (f"take the square root of every value **in place** with `np.sqrt(..., out={v})`",
         f"np.sqrt({v}, out={v})"),
    ])
    return P(f"{about(th, m, len(vals), v)}. `buf` is another reference to the same array (a training loop "
             f"holds on to it). {desc[0].upper() + desc[1:]}, without creating a new array, so `buf` sees "
             f"the change too.", f"{v} = {arr(vals)}\nbuf = {v}", sol,
             post=f"{{'same buffer as buf': bool(np.shares_memory({v}, buf)), 'values': {v}}}")


# ================================================================= medium

@template(M, "medium")
def contiguous_layouts(rng):
    r, c = rng.randint(2, 4), rng.randint(3, 5)
    setup = f"M = np.arange({r * c}).reshape({r}, {c})"
    variant = rng.choice(["c", "f", "zeros"])
    if variant == "c":
        return P(f"`M.T` is a transposed **view**, so it isn't C-contiguous. Some libraries need row-major "
                 f"memory: make `out`, a C-contiguous copy of `M.T` (`np.ascontiguousarray`).", setup,
                 "out = np.ascontiguousarray(M.T)",
                 post="{'C-contiguous': bool(out.flags['C_CONTIGUOUS']), 'values': out}")
    if variant == "f":
        return P(f"Make `out`, a **Fortran-ordered** (column-major) copy of `M` with `np.asfortranarray`, "
                 f"e.g. for a LAPACK routine.", setup, "out = np.asfortranarray(M)",
                 post="{'F-contiguous': bool(out.flags['F_CONTIGUOUS']), 'values': out}")
    return P(f"Create `out`, a `({r}, {c})` array of zeros stored in **column-major (Fortran) order** "
             f"(`order='F'`).", "", f"out = np.zeros(({r}, {c}), order='F')",
             post="{'F-contiguous': bool(out.flags['F_CONTIGUOUS']), 'shape': out.shape, 'values': out}")


@template(M, "medium")
def windows_view(rng):
    s = seeded(rng)
    variant = rng.choice(["1d", "step", "2d"])
    if variant == "2d":
        H, W, k = rng.randint(3, 5), rng.randint(3, 5), 2
        return P(f"`img` is an {H}×{W} image. Build `win`, all overlapping {k}×{k} patches as a **view** (no "
                 f"copy) with `sliding_window_view(img, ({k}, {k}))`: shape `({H - k + 1}, {W - k + 1}, {k}, {k})`.",
                 SWV + f"img = np.arange({H * W}).reshape({H}, {W})", f"win = sliding_window_view(img, ({k}, {k}))",
                 post="{'is a view of img': bool(np.shares_memory(win, img)), 'win': win}",
                 require=["sliding_window_view"])
    n, w = rng.randint(7, 12), rng.randint(2, 4)
    if variant == "step":
        st = rng.randint(2, 3)
        return P(f"`x` holds {n} sensor readings. Build `win`, windows of length {w} that start every {st} "
                 f"readings, as a **view**: take `sliding_window_view(x, {w})` and slice it with step {st}.",
                 SWV + f"x = np.random.default_rng({s}).integers(0, 100, size={n})",
                 f"win = sliding_window_view(x, {w})[::{st}]",
                 post="{'is a view of x': bool(np.shares_memory(win, x)), 'win': win}",
                 require=["sliding_window_view"])
    return P(f"`x` holds {n} sensor readings. Build `win`, all overlapping windows of length {w} (shape "
             f"`({n - w + 1}, {w})`), as a **view** with `sliding_window_view`, so nothing is copied.",
             SWV + f"x = np.random.default_rng({s}).integers(0, 100, size={n})", f"win = sliding_window_view(x, {w})",
             post="{'is a view of x': bool(np.shares_memory(win, x)), 'win': win}", require=["sliding_window_view"])


@template(M, "medium")
def read_only_arrays(rng):
    n = rng.randint(3, 6)
    if rng.random() < 0.5:
        name = rng.choice(["weights", "lookup", "vocab_ids", "cache"])
        return P(f"Protect `{name}` from accidental writes: make it **read-only** (`flags.writeable = False`). "
                 f"Then try `{name}[0] = 1` inside `try`/`except ValueError as e` and store the error message "
                 f"(`str(e)`) in `msg`.", f"{name} = np.arange({n})",
                 f"{name}.flags.writeable = False\ntry:\n    {name}[0] = 1\nexcept ValueError as e:\n    msg = str(e)",
                 post=f"{{'writeable': bool({name}.flags.writeable), 'msg': msg}}")
    return P(f"`b = np.broadcast_to(v, (3, {n}))` repeats `v` without copying, so it is a **read-only view**. "
             f"Try `b[0, 0] = 1` inside `try`/`except ValueError as e` and store the error message in `msg`.",
             f"v = np.arange({n})\nb = np.broadcast_to(v, (3, {n}))",
             "try:\n    b[0, 0] = 1\nexcept ValueError as e:\n    msg = str(e)", post="{'msg': msg}")


@template(M, "medium")
def reinterpret_bytes(rng):
    n = rng.randint(2, 4)
    desc, setup, sol = rng.choice([
        (f"view the `int32` array `x` as raw bytes (`uint8`). The result has 4 bytes per element",
         f"x = np.array({[rng.randint(0, 70000) for _ in range(n)]}, dtype=np.int32)", "x.view(np.uint8)"),
        ("view the `float32` array `x` as `uint32` to inspect the bit patterns of the floats",
         f"x = np.array({[rng.choice([1.0, -2.0, 0.5, 3.25, 0.1]) for _ in range(n)]}, dtype=np.float32)",
         "x.view(np.uint32)"),
        ("view the `int16` array `x` as bytes and reshape to one row of 2 bytes per element",
         f"x = np.array({[rng.randint(-300, 300) for _ in range(n)]}, dtype=np.int16)",
         "x.view(np.uint8).reshape(-1, 2)"),
    ])
    return P(f"Without copying, {desc}. Use `.view(dtype)`.", setup, sol, require=["view"])


# ================================================================= hard

@template(M, "hard")
def image_patches(rng):
    k = rng.choice([2, 3])
    nh, nw = rng.randint(2, 3), rng.randint(2, 3)
    H, W = nh * k, nw * k
    if rng.random() < 0.5:
        return P(f"Split the {H}×{W} image `img` into **non-overlapping {k}×{k} patches** like a Vision "
                 f"Transformer: take `sliding_window_view(img, ({k}, {k}))`, step by {k} along both axes, "
                 f"and flatten to shape `({nh * nw}, {k * k})` (patches in row-major order, each patch "
                 f"flattened row by row).", SWV + f"img = np.arange({H * W}).reshape({H}, {W})",
                 f"p = sliding_window_view(img, ({k}, {k}))[::{k}, ::{k}]\nresult = p.reshape(-1, {k * k})",
                 require=["sliding_window_view|reshape"], no_loops=True)
    H, W = H + 1, W + 1
    return P(f"For the {H}×{W} feature map `fm`, compute a {k}×{k} **max-pool with stride 1**: the max of every "
             f"overlapping {k}×{k} window, shape `({H - k + 1}, {W - k + 1})`. Use `sliding_window_view` and "
             f"reduce over the last two axes.",
             SWV + f"fm = np.random.default_rng({seeded(rng)}).integers(0, 50, size=({H}, {W}))",
             f"result = sliding_window_view(fm, ({k}, {k})).max(axis=(-2, -1))", no_loops=True)


@template(M, "hard")
def rolling_statistics(rng):
    th, m, _, vals = series(rng, 9, 12)
    v = m.var
    w = rng.randint(3, 4)
    word, red = rng.choice([("maximum", "max"), ("minimum", "min"), ("mean", "mean"), ("standard deviation", "std"),
                            ("range (max − min)", "ptp")])
    sol = (f"win = sliding_window_view({v}, {w})\nresult = win.max(axis=1) - win.min(axis=1)" if red == "ptp"
           else f"result = sliding_window_view({v}, {w}).{red}(axis=1)")
    return P(f"{about(th, m, len(vals), v)}, in order. Compute the **rolling {word}** over windows of {w} "
             f"consecutive values (length {len(vals) - w + 1}, full windows only) with `sliding_window_view`. "
             f"No loops.", SWV + f"{v} = {arr(vals)}", sol, no_loops=True)


@template(M, "hard")
def audio_frames(rng):
    F, hop = rng.choice([(4, 2), (4, 4), (6, 3), (8, 4)])
    n = F + hop * rng.randint(2, 4) + rng.randint(0, hop - 1)
    s = seeded(rng)
    nf = (n - F) // hop + 1
    if rng.random() < 0.5:
        return P(f"`audio` holds {n} samples. Cut it into **frames** of {F} samples with a hop of {hop} "
                 f"(frame i starts at sample i·{hop}; keep only full frames, {nf} in total) using "
                 f"`sliding_window_view` plus slicing, then return each frame's **RMS energy** "
                 f"`sqrt(mean(frame**2))`. No loops.",
                 SWV + f"audio = np.random.default_rng({s}).normal(0, 0.1, size={n}).round(3)",
                 f"frames = sliding_window_view(audio, {F})[::{hop}]\nresult = np.sqrt((frames ** 2).mean(axis=1))",
                 no_loops=True)
    return P(f"`audio` holds {n} samples. Cut it into **frames** of {F} samples with a hop of {hop} (frame i "
             f"starts at sample i·{hop}; keep only full frames) using `sliding_window_view` plus slicing, "
             f"multiply every frame by a Hann window `np.hanning({F})`, and return the `({nf}, {F})` array. "
             f"No loops.", SWV + f"audio = np.random.default_rng({s}).normal(0, 0.1, size={n}).round(3)",
             f"frames = sliding_window_view(audio, {F})[::{hop}]\nresult = frames * np.hanning({F})", no_loops=True)


# ================================================================= challenge

@template(M, "challenge")
def longest_budget_window(rng):
    n = rng.randint(9, 13)
    vals = [rng.randint(1, 30) for _ in range(n)]
    S_ = rng.randint(25, 70)
    var, story = rng.choice([
        ("durations", "`durations[i]` is the length in seconds of the i-th audio clip, in recording order. "
                      "A transcription job takes one **contiguous** run of clips whose total duration is at most `S`."),
        ("tokens", "`tokens[i]` is the token count of the i-th chat message, in order. A context window can "
                   "hold one **contiguous** run of messages whose total token count is at most `S`."),
        ("frames_mb", "`frames_mb[i]` is the size in MB of the i-th video frame. A GPU batch holds one "
                      "**contiguous** run of frames whose total size is at most `S` MB."),
    ])
    sol = (f"best = left = total = 0\nvals = {var}.tolist()\nfor right, x in enumerate(vals):\n"
           "    total += x\n    while total > S:\n        total -= vals[left]\n        left += 1\n"
           "    best = max(best, right - left + 1)\nresult = best")
    tests = [rand_ints(var, seeded(rng), 0, 1000, 20000) + "\nS = 250000",
             rand_ints(var, seeded(rng), 1, 50, 20000) + "\nS = 5000",
             f"{var} = np.array([100, 200, 300])\nS = 50",
             f"{var} = np.array([0, 0, 0, 0])\nS = 0"]
    return challenge("Longest batch under budget",
                     story + " What is the **maximum number of items** one job can take? (0 if none fit.)",
                     f"{var} = {arr(vals)}\nS = {S_}", sol, tests,
                     "1 ≤ n ≤ 2·10⁴, 0 ≤ values ≤ 10³, 0 ≤ S ≤ 10⁹")


@template(M, "challenge")
def stable_window(rng):
    n = rng.randint(9, 13)
    vals = [rng.randint(40, 60) for _ in range(n)]
    D = rng.randint(2, 8)
    var, story = rng.choice([
        ("temp_c", "`temp_c[i]` is a GPU's temperature at second i. A run is **stable** while its readings stay "
                   "within `D` degrees of each other."),
        ("latency_ms", "`latency_ms[i]` is the latency of the i-th request to a model endpoint. A stretch of "
                       "requests is **stable** when its slowest and fastest latency differ by at most `D` ms."),
    ])
    sol = (f"from collections import deque\nvals = {var}.tolist()\nmx, mn = deque(), deque()\nleft = best = 0\n"
           "for right, x in enumerate(vals):\n    while mx and vals[mx[-1]] <= x:\n        mx.pop()\n"
           "    mx.append(right)\n    while mn and vals[mn[-1]] >= x:\n        mn.pop()\n    mn.append(right)\n"
           "    while vals[mx[0]] - vals[mn[0]] > D:\n        left += 1\n        if mx[0] < left:\n"
           "            mx.popleft()\n        if mn[0] < left:\n            mn.popleft()\n"
           "    best = max(best, right - left + 1)\nresult = best")
    tests = [rand_ints(var, seeded(rng), 0, 1000, 20000) + "\nD = 900",
             rand_ints(var, seeded(rng), 0, 100, 20000) + "\nD = 3",
             f"{var} = np.arange(20000)\nD = 15000", f"{var} = np.array([5])\nD = 0"]
    return challenge("Longest stable stretch",
                     story + " Find the length of the **longest contiguous stretch** where "
                     "`max − min ≤ D`.", f"{var} = {arr(vals)}\nD = {D}", sol, tests,
                     "1 ≤ n ≤ 2·10⁴, 0 ≤ values ≤ 10⁹, 0 ≤ D ≤ 10⁹",
                     speed="O(n), e.g. two monotonic deques")
