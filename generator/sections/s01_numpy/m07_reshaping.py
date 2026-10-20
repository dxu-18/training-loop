"""1.07 Reshaping & moving axes."""

from ... import themes as T
from ._common import P, arr, challenge, mat, pick_measure, seeded, table, template

M = "1.07"
AXES_REQ = ["transpose|moveaxis|swapaxes|permute_dims|matrix_transpose|T"]


# ================================================================= lite

@template(M, "lite")
def reshape_simple(rng):
    r, c = rng.randint(2, 6), rng.randint(2, 6)
    th = T.theme(rng)
    m = pick_measure(rng, th, r * c)
    vals = T.values(rng, m, r * c)
    v = m.var
    infer = rng.random() < 0.4
    how = "with `-1` for the number of columns so NumPy infers it" if infer else ""
    sol = f"{v}.reshape({r}, -1)" if infer else f"{v}.reshape({r}, {c})"
    return P(f"`{v}` is a flat array of {r * c} {m.label} readings. **Reshape** it into {r} rows "
             f"and {c} columns {how}".strip() + ".", f"{v} = {arr(vals)}", sol, require=["reshape"])


@template(M, "lite")
def expand_and_squeeze(rng):
    n = rng.randint(3, 6)
    desc, setup, sol, req = rng.choice([
        (f"add a leading **batch axis** to `x` (shape `({n},)` → `(1, {n})`) with `np.expand_dims`",
         f"x = np.arange({n})", "np.expand_dims(x, axis=0)", "expand_dims|None|newaxis|reshape"),
        (f"add a **trailing** axis to `x` (shape `({n},)` → `({n}, 1)`) with `np.expand_dims`",
         f"x = np.arange({n})", "np.expand_dims(x, axis=-1)", "expand_dims|None|newaxis|reshape"),
        (f"remove **all** size-1 axes from `y` (shape `(1, {n}, 1)` → `({n},)`) with `squeeze`",
         f"y = np.arange({n}).reshape(1, {n}, 1)", "y.squeeze()", "squeeze|reshape|ravel|flatten"),
        (f"remove **only the first** axis of `y` (shape `(1, {n}, 1)` → `({n}, 1)`) with `squeeze(axis=0)`",
         f"y = np.arange({n}).reshape(1, {n}, 1)", "y.squeeze(axis=0)", "squeeze|reshape"),
        (f"make sure `v` is at least 2-D (`({n},)` → `(1, {n})`) with `np.atleast_2d`",
         f"v = np.arange({n})", "np.atleast_2d(v)", "atleast_2d|None|newaxis|reshape|expand_dims"),
    ])
    return P(f"Shape fixing: {desc}.", setup, sol, require=[req])


# ================================================================= easy

@template(M, "easy")
def transpose_table(rng):
    th, m, ax, rows, cols, data = table(rng)
    v = m.var
    flipped = [list(c) for c in zip(*data)]
    return P(f"`{v}` holds {m.label} with one row per {ax.noun} and one column per {th.noun}. "
             f"**Transpose** it so there is one row per {th.noun} instead.",
             f"{v} = {mat(flipped)}", f"{v}.T", require=AXES_REQ)


@template(M, "easy")
def flatten_table(rng):
    th, m, ax, rows, cols, data = table(rng)
    v = m.var
    return P(f"`{v}` is a 2-D table of {m.label} ({len(rows)} {th.plural} × {len(cols)} {ax.plural}). "
             f"**Flatten** it into a 1-D array, row by row (`ravel`, `flatten` or `reshape(-1)`).",
             f"{v} = {mat(data)}", f"{v}.ravel()", require=["ravel|flatten|reshape"])


@template(M, "easy")
def batch_shapes(rng):
    H, W, C, N = rng.choice([28, 32]), rng.choice([28, 32]), 3, rng.randint(2, 8)
    desc, setup, sol = rng.choice([
        (f"turn the grayscale image `img` (shape `({H}, {W})`) into a batch of one single-channel image, "
         f"shape `(1, 1, {H}, {W})`", f"img = np.zeros(({H}, {W}))", f"img.reshape(1, 1, {H}, {W})"),
        (f"turn the CHW image `img` (shape `({C}, {H}, {W})`) into a batch of one, shape `(1, {C}, {H}, {W})`",
         f"img = np.zeros(({C}, {H}, {W}))", "img[None]"),
        (f"flatten every image of the batch `x` (shape `({N}, {C}, {H}, {W})`) into a vector, giving shape "
         f"`({N}, {C * H * W})` (keep the batch axis; `reshape({N}, -1)`)",
         f"x = np.zeros(({N}, {C}, {H}, {W}))", f"x.reshape({N}, -1)"),
        (f"flatten the batch of token embeddings `x` (shape `({N}, 16, 64)`) to one row per token, shape "
         f"`({N * 16}, 64)`", f"x = np.zeros(({N}, 16, 64))", "x.reshape(-1, 64)"),
    ])
    return P(f"Get the shape a model expects: {desc}. Return the reshaped array.", setup, sol)


# ================================================================= medium

@template(M, "medium")
def move_axes(rng):
    B, H, W = rng.randint(2, 3), rng.randint(2, 4), rng.randint(2, 4)
    variant = rng.choice(["nhwc", "nchw", "heads", "time"])
    if variant == "nhwc":
        return P(f"`x` is an image batch in NHWC layout, shape `({B}, {H}, {W}, 3)`. PyTorch wants NCHW. "
                 f"Rearrange the axes to get shape `({B}, 3, {H}, {W})`.",
                 f"x = np.arange({B * H * W * 3}).reshape({B}, {H}, {W}, 3)", "x.transpose(0, 3, 1, 2)",
                 require=AXES_REQ)
    if variant == "nchw":
        return P(f"`x` is an image batch in NCHW layout, shape `({B}, 3, {H}, {W})`. Matplotlib wants "
                 f"channels last. Rearrange the axes to get NHWC, shape `({B}, {H}, {W}, 3)`.",
                 f"x = np.arange({B * H * W * 3}).reshape({B}, 3, {H}, {W})", "x.transpose(0, 2, 3, 1)",
                 require=AXES_REQ)
    if variant == "heads":
        Tn, nh, hd = rng.randint(2, 4), rng.choice([2, 3]), rng.choice([2, 4])
        return P(f"`x` has shape `({B}, {Tn}, {nh * hd})` = (batch, tokens, heads·head_dim) with {nh} "
                 f"heads of size {hd}. **Split the heads**: reshape to `({B}, {Tn}, {nh}, {hd})` and "
                 f"move the head axis forward to get shape `({B}, {nh}, {Tn}, {hd})`.",
                 f"x = np.arange({B * Tn * nh * hd}).reshape({B}, {Tn}, {nh * hd})",
                 f"x.reshape({B}, {Tn}, {nh}, {hd}).transpose(0, 2, 1, 3)", require=AXES_REQ)
    Tn, D = rng.randint(3, 5), rng.randint(2, 4)
    return P(f"`x` has shape `({B}, {Tn}, {D})` = (batch, time, features). An old RNN API wants "
             f"time first: swap the first two axes to get shape `({Tn}, {B}, {D})`.",
             f"x = np.arange({B * Tn * D}).reshape({B}, {Tn}, {D})", "x.swapaxes(0, 1)", require=AXES_REQ)


@template(M, "medium")
def reshape_then_reduce(rng):
    s = seeded(rng)
    variant = rng.choice(["daily", "weekly", "frames", "tokens"])
    if variant == "daily":
        D = rng.randint(2, 5)
        return P(f"`temp_c` holds hourly GPU temperatures for {D} full days ({D * 24} values, in order). "
                 f"Use `reshape` and a reduction to get the **mean temperature per day** (shape `({D},)`).",
                 f"temp_c = np.random.default_rng({s}).integers(35, 90, size={D * 24})",
                 f"temp_c.reshape({D}, 24).mean(axis=1)", require=["reshape"], no_loops=True)
    if variant == "weekly":
        W = rng.randint(2, 6)
        return P(f"`signups` holds daily sign-ups for an AI app over {W} full weeks ({W * 7} values). "
                 f"Use `reshape` and a reduction to get the **total per week** (shape `({W},)`).",
                 f"signups = np.random.default_rng({s}).integers(0, 500, size={W * 7})",
                 f"signups.reshape({W}, 7).sum(axis=1)", require=["reshape"], no_loops=True)
    if variant == "frames":
        F, k = rng.choice([4, 5, 8]), rng.randint(3, 6)
        return P(f"`audio` is a mono waveform of {F * k} samples. Split it into non-overlapping frames "
                 f"of {F} samples with `reshape`, then compute each frame's **energy** = mean of the "
                 f"squared samples (shape `({k},)`).",
                 f"audio = np.random.default_rng({s}).normal(0, 0.1, size={F * k}).round(3)",
                 f"(audio.reshape(-1, {F}) ** 2).mean(axis=1)", require=["reshape"], no_loops=True)
    B, Tn, Dm = rng.randint(2, 4), rng.randint(3, 6), rng.randint(2, 4)
    return P(f"`flat` holds token features for a batch flattened to shape `({B * Tn}, {Dm})`: "
             f"{B} sequences of {Tn} tokens each, in order. Reshape to `({B}, {Tn}, {Dm})` and "
             f"**average over the tokens** of each sequence (shape `({B}, {Dm})`).",
             f"flat = np.random.default_rng({s}).integers(0, 10, size=({B * Tn}, {Dm}))",
             f"flat.reshape({B}, {Tn}, {Dm}).mean(axis=1)", require=["reshape"], no_loops=True)


@template(M, "medium")
def memory_order(rng):
    r, c = rng.randint(2, 4), rng.randint(2, 4)
    n = r * c
    if rng.random() < 0.5:
        return P(f"`x` holds {n} values that were written **column by column** (Fortran order) from an "
                 f"{r}×{c} matrix, e.g. exported from MATLAB. Rebuild the matrix with "
                 f"`reshape({r}, {c}, order='F')`.", f"x = np.arange({n})", f"x.reshape({r}, {c}, order='F')",
                 require=["reshape"])
    return P(f"Flatten the {r}×{c} matrix `M` **column by column** (first all of column 0, then column 1, …) "
             f"with `ravel(order='F')` (or `flatten`).", f"M = np.arange({n}).reshape({r}, {c})",
             "M.ravel(order='F')", require=["ravel|flatten|reshape"])


@template(M, "medium")
def batched_transpose(rng):
    B, n, m = rng.randint(2, 3), rng.randint(2, 4), rng.randint(2, 4)
    what = rng.choice(["attention matrices", "Jacobians", "weight matrices", "covariance blocks"])
    return P(f"`X` is a stack of {B} {what}, shape `({B}, {n}, {m})`. Transpose **each matrix** (swap only the "
             f"last two axes) to get shape `({B}, {m}, {n})`: `np.matrix_transpose`, `swapaxes(-1, -2)` or "
             f"`transpose(0, 2, 1)`.", f"X = np.arange({B * n * m}).reshape({B}, {n}, {m})",
             "np.matrix_transpose(X)", require=AXES_REQ)


# ================================================================= hard

@template(M, "hard")
def merge_heads(rng):
    B, H, Tn, D = rng.randint(1, 2), rng.choice([2, 3]), rng.randint(2, 4), rng.choice([2, 3])
    return P(f"Multi-head attention produced `out` of shape `({B}, {H}, {Tn}, {D})` = (batch, heads, tokens, "
             f"head_dim). **Merge the heads** back: move the head axis next to head_dim and reshape to "
             f"`({B}, {Tn}, {H * D})`, so token t's vector is the concatenation of every head's output for t.",
             f"out = np.arange({B * H * Tn * D}).reshape({B}, {H}, {Tn}, {D})",
             f"out.transpose(0, 2, 1, 3).reshape({B}, {Tn}, {H * D})", require=AXES_REQ, no_loops=True)


@template(M, "hard")
def patchify(rng):
    p = rng.choice([2, 3])
    nh, nw = rng.randint(2, 3), rng.randint(2, 3)
    H, W, C = nh * p, nw * p, rng.choice([1, 3])
    return P(f"Turn the image `img` (shape `({H}, {W}, {C})`) into **{p}×{p} patches** for a Vision "
             f"Transformer, using only `reshape` and `transpose`: the result has shape `({nh * nw}, "
             f"{p * p * C})`, patches in row-major order, each flattened as (row, col, channel). Hint: reshape "
             f"to `({nh}, {p}, {nw}, {p}, {C})` first.",
             f"img = np.arange({H * W * C}).reshape({H}, {W}, {C})",
             f"img.reshape({nh}, {p}, {nw}, {p}, {C}).transpose(0, 2, 1, 3, 4).reshape(-1, {p * p * C})",
             require=["reshape"], no_loops=True)


@template(M, "hard")
def token_predictions(rng):
    B, Tn, V = rng.randint(2, 3), rng.randint(3, 5), rng.randint(5, 9)
    s = seeded(rng)
    return P(f"A language model returned `logits` flattened to shape `({B * Tn}, {V})`: {B} sequences of "
             f"{Tn} tokens, in order, over a vocabulary of {V}. Return the predicted token id for every position "
             f"as a `({B}, {Tn})` array (argmax over the vocabulary, then reshape).",
             f"logits = np.random.default_rng({s}).normal(size=({B * Tn}, {V})).round(3)",
             f"logits.argmax(axis=1).reshape({B}, {Tn})", require=["reshape"], no_loops=True)


# ================================================================= challenge

@template(M, "challenge")
def sorted_row_shapes(rng):
    n = rng.choice([12, 16, 18, 24])
    c = rng.choice([d for d in range(2, n) if n % d == 0])
    x = []
    for _ in range(n // c):
        x.extend(sorted(rng.randint(0, 20) for _ in range(c)))
    sol = ("vals = x.tolist()\nN = len(vals)\nbad = [i + 1 for i in range(N - 1) if vals[i] > vals[i + 1]]\n"
           "count = 0\nfor c in range(1, N + 1):\n    if N % c == 0 and all(b % c == 0 for b in bad):\n"
           "        count += 1\nresult = count")
    tests = [f"g = np.random.default_rng({seeded(rng)})\nx = np.sort(g.integers(0, 9, size=(55440 // 240, 240)), axis=1).ravel()",
             "x = np.arange(55440)", f"x = np.random.default_rng({seeded(rng)}).integers(0, 10**9, size=50400)",
             "x = np.array([3])"]
    return challenge("Which shapes keep rows sorted?",
                     "`x` is a flat buffer read from disk. For every way to view it as a matrix with "
                     "`x.reshape(r, c)` (all positive r, c with `r·c = n`), check whether **every row** is "
                     "non-decreasing. Count the shapes `(r, c)` for which that holds.",
                     f"x = {arr(x)}", sol, tests, "1 ≤ n ≤ 6·10⁴",
                     speed="about O(n + d(n)·#descents) (d(n) = number of divisors)")


@template(M, "challenge")
def transpose_cycles(rng):
    r, c = rng.randint(2, 5), rng.randint(2, 6)
    sol = ("total = r * c\nseen = bytearray(total)\ncycles = 0\nfor start in range(total):\n"
           "    if not seen[start]:\n        cycles += 1\n        k = start\n        while not seen[k]:\n"
           "            seen[k] = 1\n            i, j = divmod(k, c)\n            k = j * r + i\nresult = cycles")
    tests = ["r, c = 400, 500", "r, c = 1, 200000", "r, c = 449, 443", "r, c = 1, 1"]
    return challenge("Cycles of an in-place transpose",
                     "An r×c matrix is stored row-major in one flat buffer. Transposing it **in place** moves "
                     "the element at flat index `k = i·c + j` to flat index `j·r + i` (its position in the "
                     "row-major c×r result). This is a permutation of the `r·c` positions. Count its "
                     "**cycles** (fixed points count as cycles of length 1).",
                     f"r, c = {r}, {c}", sol, tests, "1 ≤ r, c and r·c ≤ 2·10⁵", speed="O(r·c)")
