"""1.04 Attributes, dtypes & casting."""

from ... import themes as T
from ._common import P, an, arr, challenge, lit, pick_measure, rand_ints, seeded, template

M = "1.04"


# ================================================================= lite

@template(M, "lite")
def array_attribute(rng):
    B = rng.choice([2, 4, 8])
    desc, var, shape, dt = rng.choice([
        ("a batch of RGB images laid out as (batch, height, width, channels)", "images",
         (B, rng.choice([28, 32, 64]), rng.choice([28, 32, 64]), 3), "uint8"),
        ("token embeddings laid out as (batch, sequence length, embedding dim)", "embeddings",
         (B, rng.choice([16, 64, 128]), rng.choice([64, 256, 768])), "float32"),
        ("log-mel spectrograms laid out as (batch, mel bins, frames)", "spectrograms",
         (B, rng.choice([64, 80, 128]), rng.choice([100, 300, 500])), "float32"),
        ("video clips laid out as (batch, frames, channels, height, width)", "clips",
         (B, rng.choice([8, 16]), 3, 64, 64), "uint8"),
        ("attention weights laid out as (batch, heads, queries, keys)", "attn",
         (B, rng.choice([4, 8, 12]), 32, 32), "float16"),
    ])
    attr, what = rng.choice([("shape", "shape (a tuple)"), ("ndim", "number of dimensions"),
                             ("size", "total number of elements"), ("dtype", "data type (dtype)")])
    return P(f"`{var}` holds {desc}. Use an array **attribute** to get its {what}.",
             f"{var} = np.zeros({shape}, dtype=np.{dt})", f"{var}.{attr}", require=[attr])


@template(M, "lite")
def list_to_dtype(rng):
    th = T.theme(rng)
    n = rng.randint(4, 8)
    m = pick_measure(rng, th, n)
    vals = T.values(rng, m, n)
    dt = rng.choice(["float32", "float64", "float16"] if m.decimals else ["int32", "int64", "float32"])
    return P(f"`{m.var}` is a plain Python list with the {m.label} of {n} {th.plural}. Convert it to a "
             f"NumPy array with dtype `{dt}` (pass `dtype=` or use `.astype`).",
             f"{m.var} = {lit(vals)}", f"np.array({m.var}, dtype=np.{dt})",
             require=["array|asarray"], check_dtype=True)


@template(M, "lite")
def bytes_attributes(rng):
    shape = rng.choice([(3, 4), (2, 3, 4), (10, 5), (4, 4, 3), (6,)])
    dt = rng.choice(["float32", "float64", "int16", "uint8", "int64", "float16"])
    name = rng.choice(["batch", "weights", "frames", "grid", "buffer"])
    attr, what = rng.choice([("itemsize", "number of bytes one element takes"),
                             ("nbytes", "total number of bytes of all elements"),
                             ("strides", "strides: bytes to step along each axis (a tuple)")])
    return P(f"`{name}` is an array of shape `{shape}` and dtype `{dt}`. Use an attribute to get the "
             f"**{what}**.", f"{name} = np.zeros({shape}, dtype=np.{dt})", f"{name}.{attr}", require=[attr])


# ================================================================= easy

@template(M, "easy")
def round_and_cast(rng):
    th = T.theme(rng)
    m = rng.choice([x for x in th.measures if x.decimals] or list(th.measures))
    n = rng.randint(5, 8)
    vals = [round(rng.uniform(m.lo, m.hi), 2) for _ in range(n)]
    v = m.var
    dt = rng.choice(["int64", "int32"])
    return P(f"`{v}` holds the {m.label} of {n} {th.plural}. Round each value to the nearest whole number "
             f"with `np.round`, then cast the result to dtype `{dt}` with `.astype`.",
             f"{v} = {arr(vals)}", f"np.round({v}).astype(np.{dt})",
             require=["round|rint|around", "astype"], check_dtype=True)


@template(M, "easy")
def dtype_limits(rng):
    desc, sol, req = rng.choice([
        ("the **largest** value an `int8` can hold (with `np.iinfo`)", "np.iinfo(np.int8).max", "iinfo"),
        ("the **smallest** value an `int16` can hold (with `np.iinfo`)", "np.iinfo(np.int16).min", "iinfo"),
        ("the largest value a `uint8` pixel can hold (with `np.iinfo`)", "np.iinfo(np.uint8).max", "iinfo"),
        ("the gap between 1.0 and the next representable `float32` (machine epsilon, `np.finfo(...).eps`)",
         "np.finfo(np.float32).eps", "finfo"),
        ("the largest finite `float16` value (with `np.finfo`). Activations above it overflow to inf in "
         "fp16 training", "np.finfo(np.float16).max", "finfo"),
        ("the machine epsilon of `float16` (with `np.finfo`)", "np.finfo(np.float16).eps", "finfo"),
        ("the largest value an `int32` can hold (with `np.iinfo`)", "np.iinfo(np.int32).max", "iinfo"),
    ])
    ctx = rng.choice(["Choosing a dtype for a model export:", "Debugging an overflow:",
                      "Sizing a quantized buffer:", "Checking numeric limits:"])
    return P(f"{ctx} get {desc}.", "", sol, require=[req])


@template(M, "easy")
def promotion_rules(rng):
    a, b = rng.choice([("int8", "float32"), ("int32", "float32"), ("uint8", "int8"), ("int16", "float16"),
                       ("int64", "float32"), ("uint8", "uint16"), ("float16", "float32")])
    desc, sol, req = rng.choice([
        (f"the dtype NumPy gives when it combines `{a}` and `{b}` (with `np.result_type`)",
         f"np.result_type(np.{a}, np.{b})", "result_type"),
        (f"the smallest dtype both `{a}` and `{b}` can be safely cast to (with `np.promote_types`)",
         f"np.promote_types(np.{a}, np.{b})", "promote_types"),
        (f"whether `{b}` can be cast to `{a}` without losing information (with `np.can_cast`)",
         f"np.can_cast(np.{b}, np.{a})", "can_cast"),
        (f"whether `{a}` is a floating-point dtype (with `np.issubdtype(..., np.floating)`)",
         f"np.issubdtype(np.{a}, np.floating)", "issubdtype"),
    ])
    return P(f"Mixed-precision code often combines dtypes. Find {desc}.", "", sol, require=[req])


# ================================================================= medium

@template(M, "medium")
def memory_footprint(rng):
    variant = rng.choice(["given", "planned", "params"])
    if variant == "given":
        shape, dt = rng.choice([((1000, 1000), "float64"), ((512, 768), "float32"), ((64, 3, 224, 224), "uint8"),
                                ((4096, 4096), "float16"), ((100, 128, 128), "float32")])
        return P(f"`x` has shape `{shape}` and dtype `{dt}`. Compute its size in **MiB** (1 MiB = 2**20 bytes) "
                 f"from its attributes.", f"x = np.zeros({shape}, dtype=np.{dt})", "x.nbytes / 2**20",
                 require=["nbytes|itemsize|size"])
    if variant == "planned":
        shape, dt = rng.choice([((50000, 768), "float32"), ((2**20, 256), "float16"), ((10**6, 3), "float64"),
                                ((32, 2048, 4096), "float16"), ((128, 512, 512), "int8")])
        return P(f"Before allocating it, compute how many **MiB** an array of shape `{shape}` and dtype "
                 f"`{dt}` would need: `np.prod(shape) * np.dtype(...).itemsize / 2**20`. Don't create "
                 f"the array.", f"shape = {shape}", f"np.prod(shape) * np.dtype(np.{dt}).itemsize / 2**20",
                 require=["itemsize"])
    n = rng.choice([7, 13, 70])
    dt = rng.choice(["float16", "float32"])
    return P(f"A model has {n} billion parameters. Compute the memory for its weights in **GiB** (2**30 "
             f"bytes) when stored as `{dt}`, using `np.dtype(np.{dt}).itemsize`.", f"n_params = {n} * 10**9",
             f"n_params * np.dtype(np.{dt}).itemsize / 2**30", require=["itemsize"])


@template(M, "medium")
def integer_overflow(rng):
    n = rng.randint(3, 6)
    a = [rng.randint(100, 255) for _ in range(n)]
    b = [rng.randint(50, 200) for _ in range(n)]
    setup = f"a = np.array({lit(a)}, dtype=np.uint8)\nb = np.array({lit(b)}, dtype=np.uint8)"
    if rng.random() < 0.5:
        return P(f"`a` and `b` are `uint8` pixel arrays. Add them **as uint8** and return the result: values "
                 f"above 255 wrap around silently (mod 256), a classic image-processing bug.", setup, "a + b")
    return P(f"`a` and `b` are `uint8` pixel arrays. Adding them directly wraps around above 255. Compute the "
             f"**true** sums by casting to `int16` first, and return them.", setup,
             "a.astype(np.int16) + b", check_dtype=True, require=["astype|int16"])


@template(M, "medium")
def select_by_dtype(rng):
    pool = [("token_ids", "np.arange(5)"), ("embeddings", "np.zeros((5, 4), dtype=np.float32)"),
            ("mask", "np.ones(5, dtype=bool)"), ("labels", "np.array([0, 1, 1, 0, 2], dtype=np.int8)"),
            ("loss_curve", "np.linspace(2.0, 0.1, 5)"), ("pixels", "np.zeros((2, 2), dtype=np.uint8)"),
            ("half_weights", "np.ones(3, dtype=np.float16)"), ("names", "np.array(['a', 'b'])")]
    chosen = rng.sample(pool, rng.randint(5, 7))
    kind, cls = rng.choice([("floating-point", "np.floating"), ("integer", "np.integer")])
    setup = "arrays = {\n" + "".join(f"    {k!r}: {v},\n" for k, v in chosen) + "}"
    return P(f"`arrays` maps names to arrays of mixed dtypes (a model checkpoint, say). Return the list of "
             f"names whose array has {an(kind).split()[0]} **{kind}** dtype, in dict order. Use `np.issubdtype(a.dtype, {cls})`.",
             setup, f"[k for k, a in arrays.items() if np.issubdtype(a.dtype, {cls})]", require=["issubdtype"])


# ================================================================= hard

@template(M, "hard")
def quantize_int8(rng):
    n = rng.randint(5, 9)
    s = seeded(rng)
    setup = f"w = np.random.default_rng({s}).normal(0, 0.05, size={n})"
    steps = ("Use **symmetric int8 quantization**: `scale = max(|w|) / 127`, `q = round(w / scale)` cast to "
             "`int8`")
    if rng.random() < 0.5:
        return P(f"`w` holds {n} float weights. {steps}. Return the tuple `(q, scale)`.", setup,
                 "scale = np.abs(w).max() / 127\nq = np.round(w / scale).astype(np.int8)\nresult = (q, scale)",
                 check_dtype=True)
    return P(f"`w` holds {n} float weights. {steps}. Then dequantize (`q * scale`) and return the **largest "
             f"absolute error** compared with `w`.", setup,
             "scale = np.abs(w).max() / 127\nq = np.round(w / scale).astype(np.int8)\n"
             "result = np.abs(q * scale - w).max()")


@template(M, "hard")
def float_image_to_uint8(rng):
    h, w = rng.randint(2, 4), rng.randint(2, 4)
    s = seeded(rng)
    return P(f"`img` is a float image produced by a diffusion model, shape `({h}, {w}, 3)`, nominally in "
             f"[0, 1] but with a few values outside that range. Convert it to a displayable `uint8` image: "
             f"clip to [0, 1], scale by 255, round, and cast to `uint8`.",
             f"img = np.random.default_rng({s}).uniform(-0.1, 1.1, size=({h}, {w}, 3))",
             "result = (np.clip(img, 0, 1) * 255).round().astype(np.uint8)", check_dtype=True, no_loops=True)


@template(M, "hard")
def smallest_fitting_dtype(rng):
    hi = rng.choice([200, 255, 256, 1000, 65535, 70000, 3 * 10**9])
    n = rng.randint(4, 7)
    vals = sorted(rng.randint(0, hi) for _ in range(n - 1)) + [hi]
    rng.shuffle(vals)
    return P(f"`ids` holds non-negative token ids stored as `int64`. To save memory, cast them to the "
             f"**smallest** unsigned dtype among `uint8`, `uint16`, `uint32`, `uint64` whose `np.iinfo(...).max` "
             f"is at least `ids.max()`. Return the cast array.",
             f"ids = np.array({lit(vals)}, dtype=np.int64)",
             "for dt in (np.uint8, np.uint16, np.uint32, np.uint64):\n    if np.iinfo(dt).max >= ids.max():\n"
             "        break\nresult = ids.astype(dt)", check_dtype=True, require=["iinfo|min_scalar_type"])


# ================================================================= challenge

@template(M, "challenge")
def wrapped_sum_windows(rng):
    n = rng.randint(9, 13)
    vals = [rng.randint(0, 255) for _ in range(n)]
    t, L = rng.randint(0, 255), rng.randint(1, 3)
    vals[rng.randrange(n)] = t
    sol = ("vals = x.tolist()\nprefix = [0]\nfor v in vals:\n    prefix.append((prefix[-1] + v) % 256)\n"
           "count = [0] * 256\nans = 0\nfor j in range(L, len(vals) + 1):\n    count[prefix[j - L]] += 1\n"
           "    ans += count[(prefix[j] - t) % 256]\nresult = ans")
    tests = [rand_ints("x", seeded(rng), 0, 256, 20000) + "\nt, L = 17, 1",
             rand_ints("x", seeded(rng), 0, 256, 20000) + "\nt, L = 0, 5000",
             "x = np.zeros(20000, dtype=int)\nt, L = 0, 1", "x = np.array([255])\nt, L = 255, 1"]
    return challenge("Wrapped window sums",
                     "`x` holds byte values (0–255) from a sensor that accumulates them in a `uint8` "
                     "register, so sums wrap around mod 256. Count the **contiguous windows** of length at "
                     "least `L` whose wrapped sum (`sum % 256`) equals `t`.",
                     f"x = {arr(vals)}\nt, L = {t}, {L}", sol, tests, "1 ≤ L ≤ n ≤ 2·10⁴, 0 ≤ x[i], t ≤ 255")


@template(M, "challenge")
def pack_bits_dp(rng):
    n = rng.randint(6, 9)
    vals = [rng.choice([rng.randint(0, 255), rng.randint(0, 65535), rng.randint(0, 2**32 - 1)]) for _ in range(n)]
    H = rng.choice([16, 32, 64])
    sol = ("vals = x.tolist()\n\ndef bits(v):\n    for w in (8, 16, 32, 64):\n        if v < 2 ** w:\n"
           "            return w\n\nn = len(vals)\nINF = float('inf')\ndp = [0] + [INF] * n\nfor i in range(1, n + 1):\n"
           "    mx = 0\n    for j in range(i - 1, -1, -1):\n        mx = max(mx, vals[j])\n"
           "        dp[i] = min(dp[i], dp[j] + H + (i - j) * bits(mx))\nresult = dp[n]")
    # int64 on purpose: Pyodide is 32-bit, so a default-int 2**32 would overflow
    big = lambda s, n_: (f"g = np.random.default_rng({s})\n"  # noqa: E731
                         f"x = g.integers(0, np.array([2**8, 2**16, 2**32], dtype=np.int64)[g.integers(0, 3, size={n_})], "
                         f"dtype=np.int64)")
    tests = [big(seeded(rng), 500) + "\nH = 64", big(seeded(rng), 500) + "\nH = 4096",
             "x = np.full(500, 2**40, dtype=np.int64)\nH = 8", "x = np.array([0])\nH = 16"]
    return challenge("Pack the ids",
                     "You store a stream of non-negative ids `x` in consecutive **blocks**. Every block "
                     "uses one dtype for all its values: the narrowest of 8, 16, 32 or 64 bits that fits the "
                     "block's largest value (`v < 2**bits`). A block costs `H` bits of header plus "
                     "`length · bits`. Split `x` into contiguous blocks to **minimize the total bits**.",
                     f"x = {arr(vals)}\nH = {H}", sol, tests, "1 ≤ n ≤ 500, 0 ≤ x[i] < 2⁶³, 1 ≤ H ≤ 10⁴",
                     speed="O(n²) (that's fine here: n ≤ 500)")
