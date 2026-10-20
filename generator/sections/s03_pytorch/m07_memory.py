"""3.07 Memory, strides & contiguity."""

from ._common import P, challenge, floats, ftt, lit, seed_line, seeded, shp, template, tt

M = "3.07"


def _grid(rng):
    r, c = rng.randint(2, 5), rng.randint(3, 6)
    return r, c, f"x = torch.arange({r * c}).view({r}, {c})"


# ================================================================= lite

@template(M, "lite")
def strides_basic(rng):
    r, c, setup = _grid(rng)
    task = rng.choice(["stride", "t_stride", "t_contig", "step_stride", "three_d"])
    if task == "stride":
        return P(f"`x` is a contiguous {r}×{c} tensor. Return `x.stride()`: how many elements to skip in memory to step "
                 f"one position along each dimension.", setup, "x.stride()", require=["stride"])
    if task == "t_stride":
        return P(f"`x.t()` transposes `x` without copying: same memory, swapped strides. Return `x.t().stride()`.", setup,
                 "x.t().stride()", require=["stride"])
    if task == "t_contig":
        return P("Return `[x.is_contiguous(), x.t().is_contiguous()]`.", setup, "[x.is_contiguous(), x.t().is_contiguous()]",
                 require=["is_contiguous"])
    if task == "step_stride":
        return P("Slicing with a step makes a view too. Return the strides of `x[:, ::2]`.", setup, "x[:, ::2].stride()",
                 require=["stride"])
    a, b, d = rng.randint(2, 4), rng.randint(2, 4), rng.randint(2, 5)
    return P(f"Return the strides of a contiguous tensor of shape ({a}, {b}, {d}).", f"y = torch.zeros({a}, {b}, {d})",
             "y.stride()", require=["stride"])


@template(M, "lite")
def view_shares_memory(rng):
    r, c, setup = _grid(rng)
    i, k = rng.randrange(r), rng.randint(10, 100)
    task = rng.choice(["view", "clone", "slice"])
    if task == "view":
        return P(f"`row = x[{i}]` is a **view** into `x`. Do `row += {k}` and return `x`: the change shows up in `x` too.",
                 setup, f"row = x[{i}]\nrow += {k}\nresult = x")
    if task == "clone":
        return P(f"Take `safe = x[{i}].clone()` (its own memory), do `safe += {k}`, and return `x`, which is unchanged.",
                 setup, f"safe = x[{i}].clone()\nsafe += {k}\nresult = x", require=["clone"])
    j = rng.randrange(c)
    return P(f"Set column {j} of `x` to {k} in place through the view `x[:, {j}]` (`fill_` or slice assignment) and "
             f"return `x`.", setup, f"x[:, {j}].fill_({k})\nresult = x")


# ================================================================= easy

@template(M, "easy")
def contiguous_fix(rng):
    r, c, setup = _grid(rng)
    task = rng.choice(["t", "permute"])
    if task == "t":
        return P("`t = x.t()` is not contiguous, so `t.view(-1)` raises \"view size is not compatible with input tensor's "
                 "size and stride\". Flatten `t` anyway: `t.contiguous().view(-1)` (or `t.reshape(-1)`, which copies "
                 "when it must).", setup + "\nt = x.t()", "t.contiguous().view(-1)", require=["contiguous|reshape"])
    a, b, d = rng.randint(2, 3), rng.randint(2, 3), rng.randint(2, 4)
    return P(f"`z` ({a}, {b}, {d}) is permuted to `p = z.permute(2, 0, 1)`. Make `p` contiguous and view it as "
             f"({d}, {a * b}).", f"z = torch.arange({a * b * d}).view({a}, {b}, {d})\np = z.permute(2, 0, 1)",
             f"p.contiguous().view({d}, {a * b})", require=["contiguous|reshape"])


@template(M, "easy")
def storage_offsets(rng):
    r, c, setup = _grid(rng)
    task = rng.choice(["rows", "cols", "elem", "shared"])
    if task == "rows":
        i = rng.randint(1, r - 1)
        return P(f"Views can start partway into the shared storage. Return `x[{i}:].storage_offset()`.", setup,
                 f"x[{i}:].storage_offset()", require=["storage_offset"])
    if task == "cols":
        j = rng.randint(1, c - 1)
        return P(f"Return the storage offset of the view `x[:, {j}:]`.", setup, f"x[:, {j}:].storage_offset()",
                 require=["storage_offset"])
    if task == "elem":
        i, j = rng.randrange(r), rng.randrange(c)
        return P(f"Return the storage offset of the single element `x[{i}, {j}]` (as a 0-d view): `{i} * stride(0) + "
                 f"{j} * stride(1)`.", setup, f"x[{i}, {j}].storage_offset()", require=["storage_offset|stride"])
    return P("Return `[x.t().data_ptr() == x.data_ptr(), x.t().contiguous().data_ptr() == x.data_ptr()]`: the transpose "
             "shares memory, the contiguous copy does not.", setup,
             "[x.t().data_ptr() == x.data_ptr(), x.t().contiguous().data_ptr() == x.data_ptr()]", require=["data_ptr"])


@template(M, "easy")
def numpy_sharing(rng):
    n, k = rng.randint(3, 6), rng.randint(1, 9)
    setup = f"arr = np.arange({n}, dtype=np.float64)"
    if rng.random() < 0.5:
        return P(f"Wrap `arr` with `torch.from_numpy` (shared memory), add {k} to the **tensor** in place (`+=`), and "
                 f"return `arr`.", setup, f"t = torch.from_numpy(arr)\nt += {k}\nresult = arr", require=["from_numpy"])
    return P(f"Copy `arr` into a tensor with `torch.tensor(arr)` (a real copy), add {k} to the tensor in place, and return "
             f"`arr`, which is unchanged.", setup, f"t = torch.tensor(arr)\nt += {k}\nresult = arr", require=["tensor"])


# ================================================================= medium

@template(M, "medium")
def expand_stride_zero(rng):
    n, m = rng.randint(2, 5), rng.choice([100, 1000, 512, 4096])
    setup = f"col = torch.randn({n}, 1)\ne = col.expand({n}, {m})"
    if rng.random() < 0.5:
        return P(f"`e = col.expand({n}, {m})` broadcasts without copying. Return `e.stride()`: stride 0 means every "
                 f"position along that dimension reads the same memory.", setup, "e.stride()", require=["stride"])
    return P(f"Return `[logical_bytes, stored_bytes]` for `e`: the bytes it **appears** to hold "
             f"(`element_size() * numel()`) versus the bytes actually stored (those of `col`).", setup,
             "[e.element_size() * e.numel(), col.element_size() * col.numel()]", require=["element_size|nbytes"])


@template(M, "medium")
def as_strided_windows(rng):
    n, w, s = rng.randint(7, 11), rng.randint(2, 4), rng.randint(1, 2)
    vals = [rng.randint(0, 50) for _ in range(n)]
    k = (n - w) // s + 1
    setup = f"x = {tt(vals)}"
    if rng.random() < 0.5:
        return P(f"Build all windows of length {w} with step {s} over `x` **as a view** (no copy) with "
                 f"`torch.as_strided(x, size=({k}, {w}), stride=({s}, 1))`.", setup,
                 f"torch.as_strided(x, size=({k}, {w}), stride=({s}, 1))", require=["as_strided|unfold"])
    return P(f"Build all windows of length {w} with step {s} over `x` with `x.unfold(0, {w}, {s})` (a view with shape "
             f"({k}, {w})).", setup, f"x.unfold(0, {w}, {s})", require=["unfold|as_strided"])


@template(M, "medium")
def memory_estimate(rng):
    b = rng.choice([0.5, 1, 3, 7, 13, 70])
    task = rng.choice(["adam_fp32", "mixed", "inference"])
    setup = f"n_params = {b}e9"
    if task == "adam_fp32":
        return P(f"Estimate the memory (GB, 1 GB = 1e9 bytes) to train a {b}B-parameter model in fp32 with Adam, before "
                 f"activations: 4 bytes of weights + 4 of gradients + 8 of Adam state (two moments) per parameter.", setup,
                 "n_params * (4 + 4 + 8) / 1e9", tol=1e-9)
    if task == "mixed":
        return P(f"Estimate the memory (GB) to train a {b}B-parameter model with bf16 mixed precision and AdamW: per "
                 f"parameter 2 bytes of bf16 weights, 2 of bf16 gradients, 4 of fp32 master weights and 8 of fp32 Adam "
                 f"moments.", setup, "n_params * (2 + 2 + 4 + 8) / 1e9", tol=1e-9)
    return P(f"Estimate the memory (GB) just to hold a {b}B-parameter model's weights for bf16 inference (2 bytes per "
             f"parameter).", setup, "n_params * 2 / 1e9", tol=1e-9)


# ================================================================= hard

@template(M, "hard")
def channels_last_strides(rng):
    N, C, H, W = rng.randint(1, 8), rng.choice([3, 4, 16]), rng.randint(4, 32), rng.randint(4, 32)
    return P(f"`img` is an (N, C, H, W) = ({N}, {C}, {H}, {W}) image batch. Convert it to the channels-last (NHWC) "
             f"memory layout with `.to(memory_format=torch.channels_last)` and return `[strides, "
             f"is_contiguous(), is_contiguous(memory_format=torch.channels_last)]` for the result. (The shape stays "
             f"NCHW; only the strides change, to (H·W·C, 1, W·C, C).)", f"img = torch.zeros({N}, {C}, {H}, {W})",
             "cl = img.to(memory_format=torch.channels_last)\n"
             "result = [cl.stride(), cl.is_contiguous(), cl.is_contiguous(memory_format=torch.channels_last)]",
             require=["channels_last"])


@template(M, "hard")
def sliding_window_stats(rng):
    n, w = rng.randint(8, 12), rng.randint(2, 4)
    vals = floats(rng, n, 0, 10, 1)
    setup = f"x = {ftt(vals)}"
    stat = rng.choice(["mean", "max", "range"])
    if stat == "mean":
        return P(f"Without loops, return the moving average of `x` over windows of {w} (one value per full window): "
                 f"`unfold` makes the windows as a view, then reduce the last dim.", setup, f"x.unfold(0, {w}, 1).mean(-1)",
                 no_loops=True, tol=1e-5)
    if stat == "max":
        return P(f"Without loops, return the rolling maximum of `x` over windows of {w} (one value per full window).",
                 setup, f"x.unfold(0, {w}, 1).amax(-1)", no_loops=True)
    return P(f"Without loops, return `max − min` of each window of {w} consecutive values of `x`.", setup,
             f"win = x.unfold(0, {w}, 1)\nresult = win.amax(-1) - win.amin(-1)", no_loops=True, tol=1e-5)


@template(M, "hard")
def diagonal_view(rng):
    n = rng.randint(3, 6)
    setup = f"x = torch.arange({n * n}).view({n}, {n})"
    if rng.random() < 0.5:
        return P(f"Read the main diagonal of the contiguous {n}×{n} matrix `x` with a stride trick: one element every "
                 f"`{n} + 1` positions, `torch.as_strided(x, size=({n},), stride=({n + 1},))`.", setup,
                 f"torch.as_strided(x, size=({n},), stride=({n + 1},))", require=["as_strided"])
    return P(f"Read the **anti-diagonal** (top-right to bottom-left) of the contiguous {n}×{n} matrix `x` as a view with "
             f"`torch.as_strided`: it starts at storage offset {n - 1} and steps {n - 1} elements each time.", setup,
             f"torch.as_strided(x, size=({n},), stride=({n - 1},), storage_offset={n - 1})", require=["as_strided"])


# ================================================================= challenge

@template(M, "challenge")
def best_fit_allocator(rng):
    N = rng.randint(16, 24)
    ops, ids = [], []
    for _ in range(rng.randint(6, 8)):
        if ids and rng.random() < 0.4:
            ops.append([1, rng.choice(ids), 0])
        else:
            ops.append([0, len(ids), rng.randint(2, 8)])
            ids.append(len(ids))
    sol = ("from bisect import bisect_left, insort\nsize_at, end_at = {0: N}, {N: 0}\n"
           "by_size = [(N, 0)]\nlive, out = {}, []\nfor t, i, s in ops.tolist():\n    if t == 0:\n"
           "        k = bisect_left(by_size, (s, -1))\n        if k == len(by_size):\n            out.append(-1)\n"
           "            continue\n        size, a = by_size.pop(k)\n        del size_at[a], end_at[a + size]\n"
           "        if size > s:\n            size_at[a + s] = size - s\n            end_at[a + size] = a + s\n"
           "            insort(by_size, (size - s, a + s))\n        live[i] = (a, s)\n        out.append(a)\n"
           "    elif i in live:\n        a, s = live.pop(i)\n        if a in end_at:\n            b = end_at.pop(a)\n"
           "            sz = size_at.pop(b)\n            by_size.pop(bisect_left(by_size, (sz, b)))\n"
           "            a, s = b, s + sz\n        if a + s in size_at:\n            sz = size_at.pop(a + s)\n"
           "            del end_at[a + s + sz]\n            by_size.pop(bisect_left(by_size, (sz, a + s)))\n"
           "            s += sz\n        size_at[a] = s\n        end_at[a + s] = a\n        insort(by_size, (s, a))\n"
           "result = out")
    gen = ("g = np.random.default_rng({s})\nN = {N}\nops, nxt = [], 0\nfor _ in range(20000):\n"
           "    if nxt and g.random() < {p}:\n        ops.append((1, int(g.integers(0, nxt)), 0))\n    else:\n"
           "        ops.append((0, nxt, int(g.integers(1, {hi}))))\n        nxt += 1\nops = torch.tensor(ops)")
    tests = [gen.format(s=seeded(rng), N=10**6, p=0.45, hi=5000), gen.format(s=seeded(rng), N=10**5, p=0.3, hi=50),
             gen.format(s=seeded(rng), N=10**9, p=0.6, hi=10**6), "N = 5\nops = torch.tensor([[0, 0, 6], [0, 1, 5], [1, 1, 0], [0, 2, 1]])"]
    return challenge("A best-fit memory allocator",
                     "A caching allocator manages addresses `[0, N)`, all free at first. Process `ops` in order. "
                     "`(0, id, size)` allocates: pick the free block with the **smallest size ≥ size** (ties: lowest "
                     "address), place the allocation at its start, and keep the rest free; if no block fits, the "
                     "allocation fails. `(1, id, 0)` frees allocation `id` if it is currently live (otherwise nothing "
                     "happens); a freed block merges with free neighbours. Return the address of every allocation in "
                     "order (−1 for failures).", f"N = {N}\nops = {tt(ops)}", sol, tests, "N ≤ 10⁹, q ≤ 2·10⁴ ops",
                     speed="O(q log q)-ish: a sorted list of (size, address) plus maps from block starts and ends")


@template(M, "challenge")
def largest_rectangle_hist(rng):
    n = rng.randint(6, 9)
    free = [rng.randint(1, 12) for _ in range(n)]
    sol = ("h = free_mb.tolist() + [0]\nstack, best = [], 0\nfor i, v in enumerate(h):\n    start = i\n"
           "    while stack and stack[-1][1] >= v:\n        start, top = stack.pop()\n        best = max(best, top * (i - start))\n"
           "    stack.append((start, v))\nresult = best")
    big = "free_mb = torch.from_numpy(np.random.default_rng({s}).integers({lo}, {hi}, size=20000))"
    tests = [big.format(s=seeded(rng), lo=0, hi=10**9), big.format(s=seeded(rng), lo=1, hi=5), "free_mb = torch.arange(1, 20001)",
             "free_mb = torch.full((20000,), 10**9)", "free_mb = torch.tensor([0])"]
    return challenge("The biggest job that fits",
                     "`free_mb[t]` is the GPU memory free at time step t. A job that needs `m` MB for `d` consecutive "
                     "steps fits if every one of those steps has at least `m` MB free. Return the largest `m · d` "
                     "(MB-steps) of any job that fits.", f"free_mb = {tt(free)}", sol, tests, "1 ≤ n ≤ 2·10⁴, values ≤ 10⁹",
                     speed="O(n) with a monotonic stack")
