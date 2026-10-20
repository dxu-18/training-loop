"""3.02 Broadcasting & shape changes."""

from ... import themes as T
from ._common import P, challenge, floats, frows_tt, ftt, lit, pick_theme, seed_line, seeded, shp, template, tt

M = "3.02"


def _factor(rng, n_factors=2, lo=2, hi=5):
    return [rng.randint(lo, hi) for _ in range(n_factors)]


# ================================================================= lite

@template(M, "lite")
def view_reshape(rng):
    a, b, c = _factor(rng, 3, 2, 4)
    task = rng.choice(["view", "minus_one", "reshape3", "flatten", "unflatten"])
    if task == "view":
        return P(f"`x` holds {a * b} numbers in order. View it as a {a}×{b} matrix (`view`).", f"x = torch.arange({a * b})",
                 f"x.view({a}, {b})", require=["view|reshape"])
    if task == "minus_one":
        return P(f"Reshape `x` so that each row has {b} elements and PyTorch infers the number of rows (`-1`).",
                 f"x = torch.arange({a * b})", f"x.view(-1, {b})", require=["view|reshape"])
    if task == "reshape3":
        return P(f"Reshape `x` into shape ({a}, {b}, {c}).", f"x = torch.arange({a * b * c})", f"x.reshape({a}, {b}, {c})",
                 require=["view|reshape"])
    if task == "flatten":
        return P(f"`x` has shape ({a}, {b}, {c}): a batch of {a} items. Flatten everything **except** the batch "
                 f"dimension, giving ({a}, {b * c}) (`flatten(start_dim=1)`).", f"x = torch.arange({a * b * c}).reshape({a}, {b}, {c})",
                 "x.flatten(start_dim=1)", require=["flatten|view|reshape"])
    return P(f"`x` has shape ({a}, {b * c}). Split dimension 1 into ({b}, {c}) with `unflatten`.",
             f"x = torch.arange({a * b * c}).reshape({a}, {b * c})", f"x.unflatten(1, ({b}, {c}))",
             require=["unflatten|view|reshape"])


@template(M, "lite")
def squeeze_unsqueeze(rng):
    dims = [rng.choice([1, rng.randint(2, 5)]) for _ in range(4)]
    dims[rng.randrange(4)] = 1
    dims[rng.randrange(4)] = rng.randint(2, 5)
    setup = f"y = torch.zeros{shp(dims)}"
    task = rng.choice(["squeeze_d", "squeeze_all", "unsqueeze", "none_index"])
    ones = [i for i, d in enumerate(dims) if d == 1]
    if task == "squeeze_d":
        d = rng.choice(ones)
        return P(f"Remove dimension {d} (which has size 1) from `y` with `squeeze({d})` and return the new shape as a "
                 f"tuple.", setup, f"tuple(y.squeeze({d}).shape)", require=["squeeze"])
    if task == "squeeze_all":
        return P("Remove **every** size-1 dimension of `y` and return the new shape as a tuple.", setup,
                 "tuple(y.squeeze().shape)", require=["squeeze"])
    if task == "unsqueeze":
        d = rng.randint(0, 4)
        return P(f"Insert a new size-1 dimension at position {d} of `y` with `unsqueeze` and return the new shape as a "
                 f"tuple.", setup, f"tuple(y.unsqueeze({d}).shape)")
    n = rng.randint(3, 6)
    return P(f"`v` has shape ({n},). Index it with `None` to make it a column of shape ({n}, 1) and return that tensor.",
             f"v = torch.arange({n})", "v[:, None]")


# ================================================================= easy

@template(M, "easy")
def transpose_permute(rng):
    shape = [rng.randint(2, 4) for _ in range(3)]
    setup = f"{seed_line(rng)}\nz = torch.randn{shp(shape)}"
    task = rng.choice(["permute", "transpose", "movedim", "mT"])
    if task == "permute":
        order = [0, 1, 2]
        while order == [0, 1, 2]:
            rng.shuffle(order)
        new = tuple(shape[i] for i in order)
        return P(f"`z` has shape {tuple(shape)}. Reorder its dimensions so the result has shape {new} (`permute`).", setup,
                 f"z.permute({', '.join(map(str, order))})", require=["permute|movedim|transpose"])
    if task == "transpose":
        a, b = rng.sample(range(3), 2)
        return P(f"Swap dimensions {a} and {b} of `z` (`transpose`).", setup, f"z.transpose({a}, {b})",
                 require=["transpose|swapaxes|permute"])
    if task == "movedim":
        return P(f"`z` is channels-last ({', '.join(map(str, shape))}). Move the last dimension to the front (`movedim`), "
                 f"giving shape ({shape[2]}, {shape[0]}, {shape[1]}).", setup, "z.movedim(-1, 0)",
                 require=["movedim|permute"])
    return P("Transpose the last two dimensions of every matrix in the batch `z` (`.mT` or `transpose(-2, -1)`).", setup,
             "z.mT", require=["mT|transpose|permute"])


@template(M, "easy")
def broadcast_result_shape(rng):
    out = [rng.randint(2, 6) for _ in range(rng.randint(2, 4))]
    a = [d if rng.random() < 0.5 else 1 for d in out]
    b = [d if rng.random() < 0.5 or x == 1 else 1 for d, x in zip(out, a)]
    b = b[rng.randint(0, len(b) - 1):]  # b may have fewer dims
    for i in range(len(out)):
        if a[i] == 1 and (len(out) - i > len(b) or b[i - (len(out) - len(b))] == 1):
            a[i] = out[i]
    if rng.random() < 0.3:  # an incompatible pair
        j = rng.randrange(len(b))
        bad = list(b)
        bad[j] = a[len(a) - len(b) + j] + 1 if a[len(a) - len(b) + j] > 1 else 7
        if a[len(a) - len(b) + j] == 1:
            a[len(a) - len(b) + j] = 3
            bad[j] = 4
        return P(f"Can a tensor of shape {tuple(a)} be added to one of shape {tuple(bad)}? Try it (or use "
                 f"`torch.broadcast_shapes`) and set `ok` to True if they broadcast, False if PyTorch raises an error.",
                 "", f"try:\n    torch.broadcast_shapes({tuple(a)}, {tuple(bad)})\n    ok = True\nexcept RuntimeError:\n    ok = False",
                 answer=["ok"])
    return P(f"What shape does `torch.ones{shp(a)} + torch.ones{shp(b)}` have? Return it as a tuple (align shapes from "
             f"the right; `torch.broadcast_shapes` does this for you).", "",
             f"tuple(torch.broadcast_shapes({tuple(a)}, {tuple(b)}))")


@template(M, "easy")
def center_scale(rng):
    th, names = pick_theme(rng, rng.randint(4, 6))
    ms = rng.sample(th.measures, min(3, len(th.measures)))
    rows = [[float(T.value(rng, m)) for m in ms] for _ in names]
    for j in range(len(ms)):
        if len({r[j] for r in rows}) == 1:
            rows[0][j] += 1
    setup = f"X = {frows_tt(rows)}"
    desc = (f"`X` has one row per {th.noun} and columns " + ", ".join(m.label for m in ms))
    task = rng.choice(["center", "unit_rows", "minmax", "weights"])
    if task == "center":
        return P(f"{desc}. Center every column (subtract its mean). Broadcasting makes `X - X.mean(dim=0)` work.", setup,
                 "X - X.mean(dim=0)", no_loops=True, tol=1e-4)
    if task == "unit_rows":
        return P(f"{desc}. Scale every **row** to unit L2 length. Use `keepdim=True` so the (n, 1) norms broadcast.",
                 setup, "X / X.norm(dim=1, keepdim=True)", no_loops=True, tol=1e-5)
    if task == "minmax":
        return P(f"{desc}. Min-max scale every column to [0, 1]: `(X - min) / (max - min)` with per-column min and max.",
                 setup, "lo, hi = X.min(dim=0).values, X.max(dim=0).values\nresult = (X - lo) / (hi - lo)", no_loops=True,
                 tol=1e-5)
    w = [round(rng.uniform(0.1, 2), 1) for _ in ms]
    return P(f"{desc}. Multiply each column by its weight in `w` (one weight per column) using broadcasting.",
             setup + f"\nw = {ftt(w)}", "X * w", no_loops=True, tol=1e-4)


# ================================================================= medium

@template(M, "medium")
def pairwise_distances(rng):
    n, m, d = rng.randint(3, 5), rng.randint(2, 4), rng.randint(2, 3)
    X = [floats(rng, d, -3, 3, 1) for _ in range(n)]
    Y = [floats(rng, d, -3, 3, 1) for _ in range(m)]
    setup = f"X = {frows_tt(X)}\nY = {frows_tt(Y)}"
    desc = f"`X` holds {n} points and `Y` {m} points in {d}-D"
    task = rng.choice(["sq", "cdist", "nearest"])
    if task == "sq":
        return P(f"{desc}. Without loops, return the ({n}, {m}) matrix of **squared** Euclidean distances: insert axes so "
                 f"`X[:, None, :] - Y[None, :, :]` broadcasts to ({n}, {m}, {d}), square, and sum the last dim.", setup,
                 "((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1)", no_loops=True, tol=1e-4)
    if task == "cdist":
        return P(f"{desc}. Return the ({n}, {m}) matrix of Euclidean distances with `torch.cdist`.", setup,
                 "torch.cdist(X, Y)", require=["cdist|norm|sqrt"], tol=1e-4)
    return P(f"{desc}. Without loops, return for every point in `X` the index of its **nearest** point in `Y`.", setup,
             "((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1).argmin(dim=1)", no_loops=True)


@template(M, "medium")
def length_masks(rng):
    B, Tn = rng.randint(2, 4), rng.randint(4, 7)
    lens = [rng.randint(1, Tn) for _ in range(B)]
    lens[0] = Tn
    setup = f"lengths = {tt(lens)}"
    if rng.random() < 0.5:
        return P(f"`lengths` holds the true lengths of {B} sequences padded to length {Tn}. Without loops, build the "
                 f"({B}, {Tn}) boolean padding mask (True = real token): compare `torch.arange({Tn})[None, :]` with "
                 f"`lengths[:, None]`.", setup, f"torch.arange({Tn})[None, :] < lengths[:, None]", no_loops=True)
    vals = [floats(rng, Tn, 0, 9, 1) for _ in range(B)]
    return P(f"`x` ({B}, {Tn}) holds per-token values of {B} padded sequences whose true lengths are in `lengths`. Without "
             f"loops, return each sequence's **mean over its real tokens only**: build the padding mask by broadcasting, "
             f"zero out the padding, and divide by the lengths.", setup + f"\nx = {frows_tt(vals)}",
             f"mask = torch.arange({Tn})[None, :] < lengths[:, None]\nresult = (x * mask).sum(dim=1) / lengths",
             no_loops=True, tol=1e-5)


@template(M, "medium")
def expand_vs_repeat(rng):
    n, m = rng.randint(2, 4), rng.randint(3, 5)
    vals = [rng.randint(1, 9) for _ in range(n)]
    setup = f"col = {tt([[v] for v in vals])}"
    post = "(tuple(_.shape), _.stride(), _.tolist())"
    if rng.random() < 0.5:
        return P(f"`col` has shape ({n}, 1). Make an ({n}, {m}) tensor that repeats it across columns **without copying "
                 f"memory** (`expand`: the new dimension gets stride 0).", setup, f"col.expand({n}, {m})", post=post,
                 require=["expand|broadcast_to"])
    return P(f"`col` has shape ({n}, 1). Make an ({n}, {m}) tensor that repeats it across columns as a **real copy** "
             f"with its own contiguous memory (`repeat`).", setup, f"col.repeat(1, {m})", post=post,
             require=["repeat|tile|contiguous|clone"])


# ================================================================= hard

@template(M, "hard")
def heads_split_merge(rng):
    B, Tn, H, D = rng.randint(1, 2), rng.randint(2, 4), rng.choice([2, 4]), rng.choice([2, 3])
    if rng.random() < 0.5:
        setup = f"{seed_line(rng)}\nx = torch.randn({B}, {Tn}, {H * D})"
        return P(f"`x` ({B}, {Tn}, {H * D}) holds token features of a batch, where the last dim is {H} heads × {D} "
                 f"features. Split it into heads, shape ({B}, {H}, {Tn}, {D}): `view` to ({B}, {Tn}, {H}, {D}), then "
                 f"`transpose(1, 2)`.", setup, f"x.view({B}, {Tn}, {H}, {D}).transpose(1, 2)",
                 require=["view|reshape|unflatten"])
    setup = f"{seed_line(rng)}\nheads = torch.randn({B}, {H}, {Tn}, {D})"
    return P(f"`heads` ({B}, {H}, {Tn}, {D}) is a multi-head attention output. Merge the heads back into ({B}, {Tn}, "
             f"{H * D}): `transpose(1, 2)` then `contiguous().view(...)` (or `reshape`).", setup,
             f"heads.transpose(1, 2).contiguous().view({B}, {Tn}, {H * D})", require=["transpose|permute"])


@template(M, "hard")
def combined_mask(rng):
    B, Tn = rng.randint(2, 3), rng.randint(3, 5)
    lens = [rng.randint(1, Tn) for _ in range(B)]
    lens[0] = Tn
    return P(f"`lengths` holds the lengths of {B} sequences padded to {Tn}. Build the ({B}, 1, {Tn}, {Tn}) boolean "
             f"attention mask where True = **may attend**: query i may attend to key j when j ≤ i (causal) **and** key "
             f"j is a real token. Build the ({Tn}, {Tn}) causal mask with `torch.tril` and the ({B}, {Tn}) padding mask "
             f"from the lengths, then combine them by broadcasting.", f"lengths = {tt(lens)}",
             f"causal = torch.tril(torch.ones({Tn}, {Tn}, dtype=torch.bool))\npad = torch.arange({Tn})[None, :] < lengths[:, None]\n"
             f"result = causal[None, None, :, :] & pad[:, None, None, :]", no_loops=True)


@template(M, "hard")
def einsum_ops(rng):
    task = rng.choice(["scores", "bmm", "trace", "weighted"])
    if task == "scores":
        B, H, Tn, D = rng.randint(1, 2), 2, rng.randint(2, 4), rng.randint(2, 3)
        setup = f"{seed_line(rng)}\nq = torch.randn({B}, {H}, {Tn}, {D})\nk = torch.randn({B}, {H}, {Tn}, {D})"
        return P(f"`q` and `k` are ({B}, {H}, {Tn}, {D}) queries and keys. With one `torch.einsum`, compute the "
                 f"attention scores `scores[b, h, i, j] = sum_d q[b, h, i, d] * k[b, h, j, d]`.", setup,
                 "torch.einsum('bhqd,bhkd->bhqk', q, k)", require=["einsum"], tol=1e-5)
    if task == "bmm":
        B, n, m, p = rng.randint(2, 3), rng.randint(2, 3), rng.randint(2, 4), rng.randint(2, 3)
        setup = f"{seed_line(rng)}\na = torch.randn({B}, {n}, {m})\nb = torch.randn({B}, {m}, {p})"
        return P(f"Multiply the {B} pairs of matrices in `a` ({B}, {n}, {m}) and `b` ({B}, {m}, {p}) batch-wise with "
                 f"`torch.einsum` (the result is ({B}, {n}, {p})).", setup, "torch.einsum('bij,bjk->bik', a, b)",
                 require=["einsum"], tol=1e-5)
    if task == "trace":
        B, n = rng.randint(2, 4), rng.randint(2, 4)
        setup = f"{seed_line(rng)}\nm = torch.randn({B}, {n}, {n})"
        return P(f"Return the trace (sum of the diagonal) of each of the {B} matrices in `m` with one `torch.einsum`.",
                 setup, "torch.einsum('bii->b', m)", require=["einsum"], tol=1e-5)
    n, d = rng.randint(3, 5), rng.randint(2, 3)
    setup = f"{seed_line(rng)}\nw = torch.rand({n})\nx = torch.randn({n}, {d})"
    return P(f"`w` holds {n} weights and `x` {n} vectors of size {d}. Return the weighted sum `sum_i w[i] * x[i]` "
             f"(shape ({d},)) with `torch.einsum`.", setup, "torch.einsum('i,id->d', w, x)", require=["einsum"], tol=1e-5)


# ================================================================= challenge

@template(M, "challenge")
def longest_broadcastable_run(rng):
    n = rng.randint(7, 9)
    rows = [[rng.choice([1, 1, 1, 2, 3]) for _ in range(4)] for _ in range(n)]
    sol = ("from collections import Counter\nrows = shapes.tolist()\ncnt = [Counter() for _ in range(4)]\nbest = lo = 0\n"
           "for hi, r in enumerate(rows):\n    for d, v in enumerate(r):\n        if v != 1:\n            cnt[d][v] += 1\n"
           "    while any(len(c) > 1 for c in cnt):\n        for d, v in enumerate(rows[lo]):\n            if v != 1:\n"
           "                cnt[d][v] -= 1\n                if cnt[d][v] == 0:\n                    del cnt[d][v]\n"
           "        lo += 1\n    best = max(best, hi - lo + 1)\nresult = best")
    gen = "shapes = torch.from_numpy(np.random.default_rng({s}).choice([1, {a}, {b}], size=(20000, 4), p=[{p}, {q}, {q}]))"
    tests = [gen.format(s=seeded(rng), a=2, b=3, p=0.98, q=0.01), gen.format(s=seeded(rng), a=4, b=8, p=0.6, q=0.2),
             gen.format(s=seeded(rng), a=5, b=6, p=0.999, q=0.0005), "shapes = torch.ones(20000, 4, dtype=torch.long)",
             "shapes = torch.tensor([[2, 3, 1, 4]])"]
    return challenge("Broadcast-friendly runs",
                     "Each row of `shapes` is a 4-D tensor shape. A run of **consecutive** shapes can be broadcast "
                     "together exactly when, in every one of the 4 positions, all sizes other than 1 are equal. Return "
                     "the length of the longest such run.", f"shapes = {tt(rows)}", sol, tests, "1 ≤ n ≤ 2·10⁴, sizes ≤ 10⁹",
                     speed="O(n) with two pointers")


@template(M, "challenge")
def count_3d_views(rng):
    sizes = [rng.randint(1, 100) for _ in range(rng.randint(4, 6))]
    sol = ("vals = sizes.tolist()\nN = max(vals)\nspf = np.arange(N + 1)\nfor p in range(2, int(N ** 0.5) + 1):\n"
           "    if spf[p] == p:\n        sl = spf[p * p::p]\n        np.minimum(sl, p, out=sl)\nspf = spf.tolist()\nout = []\n"
           "for n in vals:\n    ways = 1\n    while n > 1:\n        p, e = spf[n], 0\n        while n % p == 0:\n"
           "            n //= p\n            e += 1\n        ways *= (e + 1) * (e + 2) // 2\n    out.append(ways)\nresult = out")
    tests = [f"sizes = torch.from_numpy(np.random.default_rng({seeded(rng)}).integers(1, 10**6 + 1, size=20000))",
             "sizes = torch.full((20000,), 720720)", "sizes = torch.tensor([1, 2, 999983, 10**6])",
             f"sizes = torch.from_numpy(np.random.default_rng({seeded(rng)}).integers(1, 50, size=20000))"]
    return challenge("How many 3-D views?",
                     "A contiguous tensor with `n` elements can be viewed as `x.view(a, b, c)` for **any** positive "
                     "integers with `a·b·c = n`. For each `n` in `sizes`, count the ordered triples `(a, b, c)`. Return "
                     "the list of counts.", f"sizes = {tt(sizes)}", sol, tests, "1 ≤ q ≤ 2·10⁴ sizes, each 1 ≤ n ≤ 10⁶",
                     speed="a smallest-prime-factor sieve, then O(log n) per size")
