"""3.15 Triton kernels & C++ extensions.

In the browser, kernels run on the CPU through triton-lite, one program instance at a time (like
TRITON_INTERPRET=1): same results, and an unmasked out-of-bounds access raises an error.
"""

from ._common import P, challenge, lit, seed_line, seeded, template, tt

M = "3.15"
IMPORTS = "import triton\nimport triton.language as tl\n"

_ADD = ("@triton.jit\ndef add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):\n"
        "    pid = tl.program_id(axis=0)\n    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)\n"
        "    mask = offsets < n_elements\n    x = tl.load(x_ptr + offsets, mask=mask)\n    y = tl.load(y_ptr + offsets, mask=mask)\n"
        "    tl.store(out_ptr + offsets, x + y, mask=mask)\n")


def D(prompt, setup, solution, **kw):
    return P(prompt, IMPORTS + setup, solution, **kw)


# ================================================================= lite

@template(M, "lite")
def cdiv_grid(rng):
    task = rng.choice(["cdiv", "pow2", "grid2d"])
    if task == "cdiv":
        n, B = rng.randint(100, 100000), rng.choice([128, 256, 512, 1024])
        return D(f"A 1-D kernel processes `BLOCK_SIZE = {B}` elements per program. How many programs does the grid need "
                 f"for {n} elements? Use `triton.cdiv` (ceiling division).", f"n, BLOCK_SIZE = {n}, {B}",
                 "triton.cdiv(n, BLOCK_SIZE)", require=["cdiv"])
    if task == "pow2":
        n = rng.randint(3, 5000)
        return D(f"`tl.arange` sizes must be powers of 2, so a row-wise kernel uses `BLOCK = triton.next_power_of_2(n_cols)`. "
                 f"Return it for `n_cols = {n}`.", f"n_cols = {n}", "triton.next_power_of_2(n_cols)", require=["next_power_of_2"])
    Mm, Nn, BM, BN = rng.randint(100, 5000), rng.randint(100, 5000), rng.choice([32, 64, 128]), rng.choice([32, 64, 128])
    return D(f"A matmul kernel tiles the ({Mm} × {Nn}) output into `BLOCK_M × BLOCK_N = {BM} × {BN}` tiles, one program "
             f"each. Return the total number of programs: `cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N)`.",
             f"M, N, BLOCK_M, BLOCK_N = {Mm}, {Nn}, {BM}, {BN}", "triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)",
             require=["cdiv"])


@template(M, "lite")
def kernel_offsets(rng):
    B = rng.choice([4, 8, 16])
    n = rng.randint(B + 1, 3 * B)
    pid = (n - 1) // B
    return D(f"Simulate one program of a 1-D kernel with plain torch: program `pid = {pid}` of a launch over `n = {n}` "
             f"elements with `BLOCK_SIZE = {B}` handles `offsets = pid * BLOCK_SIZE + arange(BLOCK_SIZE)`, and "
             f"`mask = offsets < n` guards the ragged tail. Return `[offsets, mask]` as tensors.",
             f"pid, n, BLOCK_SIZE = {pid}, {n}, {B}",
             "offsets = pid * BLOCK_SIZE + torch.arange(BLOCK_SIZE)\nresult = [offsets, offsets < n]", require=["arange"])


# ================================================================= easy

@template(M, "easy")
def launch_add_kernel(rng):
    n, B = rng.randint(10, 300), rng.choice([16, 32, 64])
    setup = _ADD + f"\n{seed_line(rng)}\nx, y = torch.randn({n}), torch.randn({n})"
    return D(f"Launch `add_kernel` on `x` and `y`: allocate `out = torch.empty_like(x)`, define the grid as "
             f"`lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)`, call `add_kernel[grid](x, y, out, n, "
             f"BLOCK_SIZE={B})`, and return `out`.", setup,
             f"out = torch.empty_like(x)\nn = out.numel()\ngrid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)\n"
             f"add_kernel[grid](x, y, out, n, BLOCK_SIZE={B})\nresult = out", require=["cdiv"], tol=1e-6)


@template(M, "easy")
def scale_kernel(rng):
    n, B = rng.randint(10, 200), rng.choice([16, 32, 64])
    alpha = rng.choice([0.5, 2.0, -3.0, 1.5])
    op, expr = rng.choice([("x * alpha", "x * alpha"), ("x * alpha + 1", "x * alpha + 1"), ("max(x, 0) * alpha", "tl.maximum(x, 0.0) * alpha")])
    return D(f"Write a Triton kernel `@triton.jit def scale_kernel(x_ptr, out_ptr, alpha, n, BLOCK: tl.constexpr)` that "
             f"stores `{op}` for its block of elements (offsets from `tl.program_id(0)` and `tl.arange`, masked loads and "
             f"stores), launch it over `x` with `BLOCK={B}` and `alpha={alpha}`, and return the output.",
             f"{seed_line(rng)}\nx = torch.randn({n})",
             f"@triton.jit\ndef scale_kernel(x_ptr, out_ptr, alpha, n, BLOCK: tl.constexpr):\n    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)\n"
             f"    mask = offs < n\n    x = tl.load(x_ptr + offs, mask=mask)\n    tl.store(out_ptr + offs, {expr}, mask=mask)\n"
             f"out = torch.empty_like(x)\nscale_kernel[(triton.cdiv({n}, {B}),)](x, out, {alpha}, {n}, BLOCK={B})\nresult = out",
             require=["jit", "program_id"], tol=1e-6)


# ================================================================= medium

@template(M, "medium")
def fused_bias_relu(rng):
    R, C = rng.randint(2, 6), rng.randint(3, 20)
    return D(f"Fuse `relu(x + bias)` into one kernel: `x` is a contiguous ({R}, {C}) matrix and `bias` has {C} entries. "
             f"Launch **one program per row**; each loads its row (`x_ptr + row * stride + cols`, with `cols = "
             f"tl.arange(0, BLOCK)`, `BLOCK = triton.next_power_of_2({C})` and a mask `cols < n_cols`), adds the bias, "
             f"applies `tl.maximum(..., 0.0)`, and stores the result. Return the output tensor.",
             f"{seed_line(rng)}\nx, bias = torch.randn({R}, {C}), torch.randn({C})",
             "@triton.jit\ndef bias_relu_kernel(x_ptr, b_ptr, out_ptr, stride, n_cols, BLOCK: tl.constexpr):\n"
             "    row = tl.program_id(0)\n    cols = tl.arange(0, BLOCK)\n    mask = cols < n_cols\n"
             "    x = tl.load(x_ptr + row * stride + cols, mask=mask)\n    b = tl.load(b_ptr + cols, mask=mask)\n"
             "    tl.store(out_ptr + row * stride + cols, tl.maximum(x + b, 0.0), mask=mask)\nout = torch.empty_like(x)\n"
             "bias_relu_kernel[(x.shape[0],)](x, bias, out, x.stride(0), x.shape[1], BLOCK=triton.next_power_of_2(x.shape[1]))\n"
             "result = out", require=["jit"], tol=1e-6)


@template(M, "medium")
def row_softmax_kernel(rng):
    R, C = rng.randint(2, 5), rng.randint(3, 30)
    return D(f"Write the classic row-wise softmax kernel for a contiguous ({R}, {C}) `x`: one program per row loads it "
             f"with `other=-float('inf')` for masked lanes, subtracts `tl.max(x, axis=0)` for stability, exponentiates, "
             f"divides by `tl.sum(..., axis=0)` and stores. Use `BLOCK = triton.next_power_of_2({C})`. Return the output "
             f"(it must match `torch.softmax(x, dim=1)`).", f"{seed_line(rng)}\nx = torch.randn({R}, {C})",
             "@triton.jit\ndef softmax_kernel(out_ptr, in_ptr, in_stride, out_stride, n_cols, BLOCK: tl.constexpr):\n"
             "    row = tl.program_id(0)\n    cols = tl.arange(0, BLOCK)\n"
             "    x = tl.load(in_ptr + row * in_stride + cols, mask=cols < n_cols, other=-float('inf'))\n"
             "    x = x - tl.max(x, axis=0)\n    num = tl.exp(x)\n"
             "    tl.store(out_ptr + row * out_stride + cols, num / tl.sum(num, axis=0), mask=cols < n_cols)\n"
             "out = torch.empty_like(x)\n"
             "softmax_kernel[(x.shape[0],)](out, x, x.stride(0), out.stride(0), x.shape[1], BLOCK=triton.next_power_of_2(x.shape[1]))\n"
             "result = out", require=["jit", "max", "sum"], tol=1e-5)


@template(M, "medium")
def custom_op_register(rng):
    alpha = rng.choice([2.0, 0.5, 3.0, -1.0])
    n = rng.randint(2, 4)
    ns = rng.choice(["mylib", "kernels", "myops"])
    return D(f"Register `scaled_add(a, b, alpha) = a + alpha * b` as a real PyTorch op with "
             f"`@torch.library.custom_op('{ns}::scaled_add', mutates_args=())` (type-annotate the arguments), add a "
             f"fake implementation with `@scaled_add.register_fake` (returns `torch.empty_like(a)`), and gradients with "
             f"`scaled_add.register_autograd(backward, setup_context=...)` where backward returns `(grad, grad * alpha, "
             f"None)`. Then backpropagate `scaled_add(ta, tb, {alpha}).sum()` for `ta`, `tb` = ones({n}) requiring grad "
             f"and return `[ta.grad, tb.grad]`.", "",
             f"@torch.library.custom_op('{ns}::scaled_add', mutates_args=())\n"
             f"def scaled_add(a: torch.Tensor, b: torch.Tensor, alpha: float) -> torch.Tensor:\n    return a + alpha * b\n"
             f"@scaled_add.register_fake\ndef _(a, b, alpha):\n    return torch.empty_like(a)\n"
             f"def _setup(ctx, inputs, output):\n    ctx.alpha = inputs[2]\ndef _backward(ctx, grad):\n"
             f"    return grad, grad * ctx.alpha, None\nscaled_add.register_autograd(_backward, setup_context=_setup)\n"
             f"ta, tb = torch.ones({n}, requires_grad=True), torch.ones({n}, requires_grad=True)\n"
             f"scaled_add(ta, tb, {alpha}).sum().backward()\nresult = [ta.grad, tb.grad]",
             require=["custom_op", "register_autograd"])


# ================================================================= hard

@template(M, "hard")
def strided_row_sum(rng):
    R, C = rng.randint(3, 6), rng.randint(3, 9)
    return D(f"Kernels see raw memory, so non-contiguous inputs need their **strides**. `xt = x.t()` is a ({C}, {R}) "
             f"view with strides {(1, C)}. Write a kernel that sums each row of `xt` (one program per row, loading "
             f"`x_ptr + row * stride_r + cols * stride_c` with a mask and `other=0.0`, then `tl.sum`) and launch it with "
             f"`xt.stride(0), xt.stride(1)`. Return the ({C},) result (it must equal `xt.sum(dim=1)`).",
             f"{seed_line(rng)}\nx = torch.randn({R}, {C})\nxt = x.t()",
             "@triton.jit\ndef row_sum_kernel(x_ptr, out_ptr, stride_r, stride_c, n_cols, BLOCK: tl.constexpr):\n"
             "    row = tl.program_id(0)\n    cols = tl.arange(0, BLOCK)\n"
             "    vals = tl.load(x_ptr + row * stride_r + cols * stride_c, mask=cols < n_cols, other=0.0)\n"
             "    tl.store(out_ptr + row, tl.sum(vals, axis=0))\nout = torch.empty(xt.shape[0])\n"
             "row_sum_kernel[(xt.shape[0],)](xt, out, xt.stride(0), xt.stride(1), xt.shape[1], BLOCK=triton.next_power_of_2(xt.shape[1]))\n"
             "result = out", require=["stride"], tol=1e-5)


@template(M, "hard")
def atomic_histogram(rng):
    n, bins, B = rng.randint(20, 80), rng.randint(3, 8), rng.choice([8, 16, 32])
    return D(f"Write a histogram kernel: each program loads a block of the int64 `ids` (values in `[0, {bins})`) and "
             f"does `tl.atomic_add(hist_ptr + ids, 1, mask=mask)` into an int64 `hist` of zeros. Programs run "
             f"concurrently on a GPU, which is why the add must be atomic. Launch it with `BLOCK={B}` and return `hist` "
             f"(it must equal `torch.bincount(ids, minlength={bins})`).",
             f"{seed_line(rng)}\nids = torch.randint(0, {bins}, ({n},))",
             f"@triton.jit\ndef hist_kernel(ids_ptr, hist_ptr, n, BLOCK: tl.constexpr):\n"
             f"    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)\n    mask = offs < n\n"
             f"    v = tl.load(ids_ptr + offs, mask=mask, other=0)\n    tl.atomic_add(hist_ptr + v, 1, mask=mask)\n"
             f"hist = torch.zeros({bins}, dtype=torch.long)\nhist_kernel[(triton.cdiv({n}, {B}),)](ids, hist, {n}, BLOCK={B})\n"
             f"result = hist", require=["atomic_add"])


@template(M, "hard")
def tiled_matmul_kernel(rng):
    Mm, Nn, K = rng.choice([8, 12, 16]), rng.choice([8, 12, 16]), rng.choice([8, 16, 24])
    return D(f"Write a tiled matmul kernel for contiguous `a` ({Mm}, {K}) and `b` ({K}, {Nn}) with 8×8 output tiles: "
             f"program `(pid_m, pid_n)` builds `rm = pid_m * 8 + tl.arange(0, 8)`, `rn = ...`, keeps an fp32 "
             f"accumulator `tl.zeros((8, 8), dtype=tl.float32)`, and loops over K in steps of 8 loading the a-tile "
             f"`a_ptr + rm[:, None] * K + rk[None, :]` and the b-tile `b_ptr + rk[:, None] * N + rn[None, :]` (masked) "
             f"and adding `tl.dot(a_tile, b_tile)`; finally it stores the masked tile. Launch a "
             f"`(cdiv(M, 8), cdiv(N, 8))` grid and return the product.",
             f"{seed_line(rng)}\na, b = torch.randn({Mm}, {K}), torch.randn({K}, {Nn})",
             "@triton.jit\ndef matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):\n"
             "    rm = tl.program_id(0) * BM + tl.arange(0, BM)\n    rn = tl.program_id(1) * BN + tl.arange(0, BN)\n"
             "    acc = tl.zeros((BM, BN), dtype=tl.float32)\n    for k0 in range(0, K, BK):\n        rk = k0 + tl.arange(0, BK)\n"
             "        a = tl.load(a_ptr + rm[:, None] * K + rk[None, :], mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)\n"
             "        b = tl.load(b_ptr + rk[:, None] * N + rn[None, :], mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)\n"
             "        acc += tl.dot(a, b)\n"
             "    tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc, mask=(rm[:, None] < M) & (rn[None, :] < N))\n"
             "M, K = a.shape\nN = b.shape[1]\nc = torch.empty(M, N)\n"
             "matmul_kernel[(triton.cdiv(M, 8), triton.cdiv(N, 8))](a, b, c, M, N, K, BM=8, BN=8, BK=8)\nresult = c",
             require=["dot"], tol=1e-4)


# ================================================================= challenge

@template(M, "challenge")
def swizzled_tile_loads(rng):
    pm, pn, G = rng.randint(3, 6), rng.randint(2, 5), rng.randint(2, 3)
    qs = sorted(rng.sample(range(1, pm * pn + 1), rng.randint(3, 4)))
    sol = ("out = []\nper_group = GROUP_M * num_pid_n\nfor P in queries.tolist():\n    g, rem = divmod(P, per_group)\n"
           "    rows, cols = min(g * GROUP_M, num_pid_m), (num_pid_n if g else 0)\n    if rem:\n"
           "        gsm = min(num_pid_m - g * GROUP_M, GROUP_M)\n        rows += min(rem, gsm)\n"
           "        if not g:\n            cols = min(num_pid_n, -(-rem // gsm))\n    out.append(rows + cols)\nresult = out")
    big = ("g = np.random.default_rng({s})\nnum_pid_m, num_pid_n, GROUP_M = {pm}, {pn}, {G}\n"
           "queries = torch.from_numpy(g.integers(1, num_pid_m * num_pid_n + 1, size=20000))")
    tests = [big.format(s=seeded(rng), pm=10**4, pn=10**4, G=8), big.format(s=seeded(rng), pm=7, pn=10**4, G=8),
             big.format(s=seeded(rng), pm=10**4, pn=3, G=1), "num_pid_m, num_pid_n, GROUP_M = 1, 1, 4\nqueries = torch.tensor([1])"]
    return challenge("L2-friendly tile order",
                     "Triton's matmul tutorial launches programs in **grouped order** to reuse cache: with "
                     "`per_group = GROUP_M · num_pid_n`, program `pid` computes the output tile `(pid_m, pid_n)` where "
                     "`group = pid // per_group`, `first = group · GROUP_M`, `size = min(num_pid_m − first, GROUP_M)`, "
                     "`pid_m = first + (pid % per_group) % size` and `pid_n = (pid % per_group) // size`. Tile "
                     "`(pid_m, pid_n)` reads row-block `pid_m` of A and column-block `pid_n` of B. For each P in "
                     "`queries`, the first P programs (pids 0..P−1) have run: return how many **distinct** A row-blocks "
                     "plus distinct B column-blocks they have read.",
                     f"num_pid_m, num_pid_n, GROUP_M = {pm}, {pn}, {G}\nqueries = {tt(qs)}", sol, tests,
                     "num_pid_m, num_pid_n ≤ 10⁴ (up to 10⁸ programs), q ≤ 2·10⁴",
                     speed="O(1) per query: count whole groups, then the partial one")


@template(M, "challenge")
def pow2_block_fit(rng):
    n = rng.randint(6, 10)
    x = [rng.randint(0, 9) for _ in range(n)]
    S = rng.randint(9, 25)
    sol = ("vals = x.tolist()\nn = len(vals)\npref = [0]\nfor v in vals:\n    pref.append(pref[-1] + v)\nbest, B = 0, 1\n"
           "while True:\n    if any(pref[min(i + B, n)] - pref[i] > S for i in range(0, n, B)):\n        break\n    best = B\n"
           "    if B >= n:\n        break\n    B *= 2\nresult = best")
    big = "x = torch.from_numpy(np.random.default_rng({s}).integers(0, {hi}, size=20000))\nS = {S}"
    tests = [big.format(s=seeded(rng), hi=1000, S=10**5), big.format(s=seeded(rng), hi=10**9, S=10**9),
             big.format(s=seeded(rng), hi=2, S=10**9), "x = torch.tensor([5])\nS = 4", "x = torch.zeros(20000, dtype=torch.long)\nS = 0"]
    return challenge("The biggest block that fits",
                     "A reduction kernel splits `x` into **aligned** blocks of `BLOCK` elements (`[0, BLOCK)`, `[BLOCK, "
                     "2·BLOCK)`, ...; the last one may be cut short by the mask), and each program's accumulator can hold a "
                     "sum of at most `S`. Return the largest power-of-two `BLOCK` such that every block's sum is ≤ S, "
                     "counting only blocks up to `next_power_of_2(n)` (or 0 if even `BLOCK = 1` fails).",
                     f"x = {tt(x)}\nS = {S}", sol, tests, "1 ≤ n ≤ 2·10⁴, x ≥ 0", speed="O(n log n) with prefix sums")
