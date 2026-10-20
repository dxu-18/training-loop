"""3.01 Tensors: catalog templates (one API per problem) covering the rest of the guide's tensor API."""

from ._common import STDOUT, catalog, fm, ftt, fv, lit, rows_tt, seeded, template, tt

M = "3.01"


def _fmat(rows):
    return "torch.tensor(" + lit([[float(v) for v in r] for r in rows]) + ")"


# ================================================================= lite

@template(M, "lite")
def creation_catalog(rng):
    r, c = rng.randint(2, 4), rng.randint(2, 5)
    x = fm(rng, r, c)
    fill = rng.choice([0.5, 7.0, -1.5, 2.25])
    k, n = rng.randint(2, 4), rng.randint(3, 5)
    setup = f"x = {_fmat(x)}"
    d64 = {"setup": f"d = torch.tensor({lit(fv(rng, 2))}, dtype=torch.float64)", "check_dtype": True}
    arr = {"setup": f"arr = np.arange({n}.0)"}
    items = [
        ("empty", f"Allocate an **uninitialized** float tensor of shape ({r}, {c}) with `torch.empty` (its values are "
                  f"whatever was in memory, so return just its shape as a tuple).", f"tuple(torch.empty({r}, {c}).shape)"),
        ("empty_like", "Return `[tuple(e.shape), e.dtype]` for `e = torch.empty_like(x)`.",
         "e = torch.empty_like(x)\nresult = [tuple(e.shape), e.dtype]"),
        ("full_like", f"Return a tensor shaped like `x` filled with {fill} (`torch.full_like`).", f"torch.full_like(x, {fill})"),
        ("ones_like", "Return a tensor of ones shaped like `x` (`torch.ones_like`).", "torch.ones_like(x)"),
        ("zeros_like", "Return a tensor of zeros shaped like `x` (`torch.zeros_like`).", "torch.zeros_like(x)"),
        ("logspace", f"Return {n} values spaced evenly on a **log** scale from 10⁰ to 10^{k} (`torch.logspace`).",
         f"torch.logspace(0, {k}, steps={n})", {"tol": 1e-4}),
        ("new_zeros", "`d` is a float64 tensor. Make a (2, 2) tensor of zeros with **the same dtype and device** as `d` "
                      "(`d.new_zeros`).", "d.new_zeros(2, 2)", d64),
        ("new_ones", f"Make {n} ones with the same dtype as `d` (`d.new_ones`).", f"d.new_ones({n})", d64),
        ("new_full", f"Make a length-{n} tensor filled with {fill}, with the same dtype as `d` (`d.new_full`).",
         f"d.new_full(({n},), {fill})", d64),
        ("new_tensor", "Turn the list `[1, 2]` into a tensor with the same dtype as `d` (`d.new_tensor`).",
         "d.new_tensor([1, 2])", d64),
        ("new_empty", f"Return `[tuple(e.shape), e.dtype]` for `e = d.new_empty({n})`.",
         f"e = d.new_empty({n})\nresult = [tuple(e.shape), e.dtype]", d64),
        ("as_tensor", "Make a tensor that **shares memory** with the NumPy array `arr` (`torch.as_tensor`), then set "
                      "`arr[0] = 99` and return the tensor: the change shows through.",
         "t = torch.as_tensor(arr)\narr[0] = 99\nresult = t", arr),
        ("asarray", "Wrap `arr` with `torch.asarray` (array-API style, no copy when possible), set `arr[-1] = -1`, and "
                    "return the tensor.", "t = torch.asarray(arr)\narr[-1] = -1\nresult = t", arr),
    ]
    return catalog(rng, items, setup, "")


@template(M, "lite")
def dtype_catalog(rng):
    a, b, c = rng.randint(2, 4), rng.randint(2, 4), rng.randint(2, 5)
    setup = f"x = torch.arange({a * b * c}, dtype=torch.float32).reshape({a}, {b}, {c})"
    m2 = {"setup": f"m = {_fmat(fm(rng, 2, 3))}"}
    items = [
        ("dtype", "Return the dtype of `x`.", "x.dtype"),
        ("dim", "Return the number of dimensions of `x` with `x.dim()`.", "x.dim()"),
        ("size", "Return `x.size()` (a `torch.Size`, which is a tuple).", "x.size()"),
        ("nbytes", "How many bytes does `x` occupy? Use `x.nbytes`.", "x.nbytes"),
        ("is_floating_point", "Is `x` a floating-point tensor? (`is_floating_point()`)", "x.is_floating_point()"),
        ("is_tensor", "Return `torch.is_tensor(x)`.", "torch.is_tensor(x)"),
        ("requires_grad", "Return `[x.requires_grad, x.is_leaf]` (a freshly created tensor).", "[x.requires_grad, x.is_leaf]",
         {"require": ["requires_grad", "is_leaf"]}),
        ("mT", "Return the shape of `x.mT`, which transposes the last two dimensions of every matrix in the batch.",
         "tuple(x.mT.shape)"),
        ("T", "`m` is a 2-D tensor. Return its transpose with `.T`.", "m.T", m2),
        ("double", "Return the dtype of `x.double()`.", "x.double().dtype"),
        ("int", "Convert `m` to int32 with `.int()` (truncates toward zero).", "m.int()", {**m2, "check_dtype": True}),
        ("bool", "Convert `m` to booleans with `.bool()` (nonzero → True).", "m.bool()", m2),
        ("bfloat16", "Return the dtype of `x.bfloat16()`.", "x.bfloat16().dtype"),
        ("int8", "Convert `m` to int8 (`m.to(torch.int8)`).", "m.to(torch.int8)", {**m2, "check_dtype": True}),
        ("get_default_dtype", "Return PyTorch's default floating dtype (`torch.get_default_dtype()`).",
         "torch.get_default_dtype()"),
        ("promote_types", "Which dtype do an int32 and a float16 tensor promote to? `torch.promote_types`.",
         "torch.promote_types(torch.int32, torch.float16)"),
        ("result_type", "What dtype does `torch.tensor(1) + 1.0` produce? Ask `torch.result_type`.",
         "torch.result_type(torch.tensor(1), 1.0)"),
        ("can_cast", "Can float32 be cast to int32 under PyTorch's casting rules? (`torch.can_cast`)",
         "torch.can_cast(torch.float32, torch.int32)"),
        ("iinfo", "Return the smallest int8 value (`torch.iinfo(torch.int8).min`).", "torch.iinfo(torch.int8).min"),
        ("finfo", "Return float32's machine epsilon, the gap between 1.0 and the next float (`torch.finfo(torch.float32).eps`).",
         "torch.finfo(torch.float32).eps", {"tol": 1e-9}),
        ("layout", "Is `x` a regular dense tensor? Return `x.layout == torch.strided`.", "x.layout == torch.strided"),
        ("device", "Return the device type of `x` as a string (`x.device.type`).", "x.device.type"),
        ("stride", "Return `[x.stride(), x.is_contiguous()]`.", "[x.stride(), x.is_contiguous()]", {"require": ["stride", "is_contiguous"]}),
        ("flatten", "Return the shape of `x.flatten()` as a tuple.", "tuple(x.flatten().shape)"),
        ("grad", "A fresh tensor has no gradient yet: return `x.grad is None`.", "x.grad is None"),
        ("is_available", "Pick a device the portable way: `'cuda' if torch.cuda.is_available() else ('mps' if "
                         "torch.backends.mps.is_available() else 'cpu')`, and return it (the browser has no GPU).",
         "result = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')"),
        ("benchmark", "Turn off cuDNN autotuning for reproducibility (`torch.backends.cudnn.benchmark = False`) and return the "
                      "flag.", "torch.backends.cudnn.benchmark = False\nresult = torch.backends.cudnn.benchmark"),
    ]
    return catalog(rng, items, setup, f"`x` is a float32 tensor of shape ({a}, {b}, {c}).")


@template(M, "lite")
def convert_catalog(rng):
    n = rng.randint(3, 5)
    vals = fv(rng, n, nonzero=True)
    setup = f"t = torch.tensor({lit(vals)}, requires_grad=True)"
    items = [
        ("clone", "Copy `t` with `clone()` (detached first), set the copy's first entry to 100, and return `[t.detach(), "
                  "copy]`: the original is unchanged.", "c = t.detach().clone()\nc[0] = 100\nresult = [t.detach(), c]"),
        ("copy_", f"Copy `t`'s values into an existing buffer `out = torch.empty({n})` with `out.copy_(...)` (detach "
                  f"first) and return `out`.", f"out = torch.empty({n})\nout.copy_(t.detach())\nresult = out"),
        ("numpy", "`t` requires grad, so plain `.numpy()` raises. Return `t.numpy(force=True)` (it detaches/copies as "
                  "needed).", "t.numpy(force=True)", {"require": ["force"]}),
        ("save", "Save `t.detach()` to `t.pt` with `torch.save` and load it back with `torch.load(..., weights_only="
                 "True)`. Return the loaded tensor.", "torch.save(t.detach(), 't.pt')\nresult = torch.load('t.pt', weights_only=True)",
         {"require": ["save", "load"]}),
        ("to", "Return `t.detach()` converted to float64 with `.to(...)`.", "t.detach().to(torch.float64)", {"check_dtype": True}),
        ("type_as", "`i` is an int64 tensor. Return `t.detach()` converted to `i`'s dtype with `type_as`.",
         "t.detach().type_as(i)", {"setup": "i = torch.tensor([1, 2])", "check_dtype": True}),
        ("cpu", "Move `t.detach()` to the CPU with `.cpu()` (a no-op here, but the portable spelling) and return it.",
         "t.detach().cpu()"),
    ]
    return catalog(rng, items, setup, "")


@template(M, "lite")
def compare_catalog(rng):
    n = rng.randint(3, 5)
    a, b = fv(rng, n), fv(rng, n)
    b[0] = a[0]
    setup = f"a = {ftt(a)}\nb = {ftt(b)}"
    special = {"setup": "s = torch.tensor([1.0, float('nan'), float('inf'), -float('inf'), 0.0])"}
    items = [
        ("eq", "Return the elementwise comparison `a == b` written with `torch.eq`.", "torch.eq(a, b)"),
        ("gt", "Return where `a > b`, written with `torch.gt`.", "torch.gt(a, b)"),
        ("equal", "Return whether `a` and `b` have the same shape **and** values (`torch.equal`, a Python bool).",
         "torch.equal(a, b)"),
        ("allclose", "Return `torch.allclose(a, a + 1e-9)`: equal up to a tolerance.", "torch.allclose(a, a + 1e-9)"),
        ("isclose", "Return the elementwise closeness of `a` and `b` (`torch.isclose`).", "torch.isclose(a, b)"),
        ("assert_close", "`torch.testing.assert_close` raises `AssertionError` with a detailed message when tensors differ. "
                         "Call it on `a` and `b` inside try/except and return True if it passed.",
         "try:\n    torch.testing.assert_close(a, b)\n    result = True\nexcept AssertionError:\n    result = False"),
        ("isnan", "Return which entries of `s` are NaN.", "torch.isnan(s)", special),
        ("isinf", "Return which entries of `s` are ±inf.", "torch.isinf(s)", special),
        ("isfinite", "Return which entries of `s` are finite.", "torch.isfinite(s)", special),
        ("logical_and", "Return where both `a > 0` and `b > 0` (`torch.logical_and`).", "torch.logical_and(a > 0, b > 0)"),
        ("", "Return where both `a > 0` and `b > 0` using the `&` operator on boolean tensors.", "(a > 0) & (b > 0)"),
        ("", "Return where `a > 0` is **False**, using `~`.", "~(a > 0)"),
    ]
    return catalog(rng, items, setup, "")


@template(M, "lite")
def arithmetic_catalog(rng):
    n = rng.randint(3, 5)
    a = fv(rng, n, -9, 9, 1)
    b = [abs(v) + 0.5 for v in fv(rng, n, 0, 4, 1)]
    lo, hi = sorted(rng.sample([-2, -1, 0, 1, 2, 3], 2))
    setup = f"a = {ftt(a)}\nb = {ftt(b)}   # all positive"
    items = [
        ("add", "Return `a + 2·b` with one call: `torch.add(..., alpha=2)`.", "torch.add(a, b, alpha=2)", {"require": ["add", "alpha"]}),
        ("div", "Return `a / b` rounded **down** with `torch.div(..., rounding_mode='floor')`.",
         "torch.div(a, b, rounding_mode='floor')", {"require": ["div", "rounding_mode"]}),
        ("remainder", "Return `torch.remainder(a, b)` (the result has the sign of the divisor, like Python's `%`).",
         "torch.remainder(a, b)"),
        ("fmod", "Return `torch.fmod(a, b)` (C-style: the result has the sign of the dividend).", "torch.fmod(a, b)"),
        ("pow", "Return the square root of `b` written as `torch.pow(b, 0.5)`.", "torch.pow(b, 0.5)"),
        ("abs", "Return `torch.abs(a)`.", "torch.abs(a)"),
        ("sign", "Return the sign of each entry of `a` (`torch.sign`).", "torch.sign(a)"),
        ("sqrt", "Return `torch.sqrt(b)`.", "torch.sqrt(b)"),
        ("rsqrt", "Return `1 / sqrt(b)` with `torch.rsqrt`.", "torch.rsqrt(b)"),
        ("square", "Return `torch.square(a)`.", "torch.square(a)"),
        ("floor", "Return `torch.floor(a)`.", "torch.floor(a)"),
        ("ceil", "Return `torch.ceil(a)`.", "torch.ceil(a)"),
        ("round", "Return `a / 3` rounded to 2 decimals (`torch.round(..., decimals=2)`).", "torch.round(a / 3, decimals=2)",
         {"require": ["round", "decimals"]}),
        ("trunc", "Round `a` toward zero with `torch.trunc`.", "torch.trunc(a)"),
        ("frac", "Return the fractional part of `a` (`torch.frac`, same sign as `a`).", "torch.frac(a)"),
        ("clamp", f"Limit `a` to [{lo}, {hi}] with `torch.clamp(..., min=..., max=...)`.", f"torch.clamp(a, min={lo}, max={hi})"),
        ("clip", f"Limit `a` to [{lo}, {hi}] with `torch.clip` (an alias of clamp).", f"torch.clip(a, {lo}, {hi})"),
        ("maximum", "Return the elementwise maximum of `a` and `b` (`torch.maximum`; `torch.max(a, b)` also works).",
         "torch.maximum(a, b)"),
        ("minimum", "Return the elementwise minimum of `a` and `b`.", "torch.minimum(a, b)"),
        ("lerp", "Interpolate 30% of the way from `a` to `b` with `torch.lerp`.", "torch.lerp(a, b, 0.3)"),
        ("hypot", "Return `sqrt(a² + b²)` elementwise with `torch.hypot`.", "torch.hypot(a, b)"),
        ("addcmul", "Return the fused `a + 0.5 · a · b` with `torch.addcmul(..., value=0.5)` (used inside optimizers).",
         "torch.addcmul(a, a, b, value=0.5)"),
        ("addcdiv", "Return the fused `a + 0.5 · a / b` with `torch.addcdiv(..., value=0.5)`.", "torch.addcdiv(a, a, b, value=0.5)"),
        ("add_", "In-place ops end in `_`. On `c = a.clone()`, call `c.add_(1)` then `c.mul_(2)` and return `c`.",
         "c = a.clone()\nc.add_(1)\nc.mul_(2)\nresult = c", {"require": ["add_", "mul_"]}),
        ("clamp_", f"On `c = a.clone()`, clamp in place to [0, {hi + 5}] and return `c`.", f"c = a.clone()\nc.clamp_(0, {hi + 5})\nresult = c"),
        ("zero_", "On `c = a.clone()`, zero it in place and then add `b` in place; return `c`.",
         "c = a.clone()\nc.zero_()\nc.add_(b)\nresult = c", {"require": ["zero_"]}),
        ("fill_", "On `c = a.clone()`, fill it with 3 in place and return `c`.", "c = a.clone()\nc.fill_(3)\nresult = c"),
        ("out", f"Write `a + b` into a preallocated `buf = torch.empty({n})` with `torch.add(a, b, out=buf)` and return "
                f"`buf`.", f"buf = torch.empty({n})\ntorch.add(a, b, out=buf)\nresult = buf", {"require": ["add", "out"]}),
    ]
    return catalog(rng, items, setup, "")


# ================================================================= easy

@template(M, "easy")
def structured_catalog(rng):
    v = fv(rng, 3, -3, 3, 1, nonzero=True)
    n = rng.randint(3, 4)
    Mx = [[float(i * n + j) for j in range(n)] for i in range(n)]
    p, q = [rng.randint(1, 5) for _ in range(2)], [rng.randint(6, 9) for _ in range(2)]
    A = fm(rng, 2, 2)
    setup = f"v = {ftt(v)}\nM = {_fmat(Mx)}"
    items = [
        ("diag", "Build the diagonal matrix with `v` on its diagonal (`torch.diag(v)`).", "torch.diag(v)"),
        ("diag", "Extract the main diagonal of `M` as a vector (`torch.diag(M)`).", "torch.diag(M)"),
        ("diag_embed", "Return `torch.diag_embed(v)` (batched diag: the last dim becomes a diagonal).", "torch.diag_embed(v)"),
        ("diagflat", "Return `torch.diagflat(v)`.", "torch.diagflat(v)"),
        ("tril", "Return the lower triangle of `M` (zeros above the diagonal).", "torch.tril(M)"),
        ("triu", "Return the strictly upper triangle of `M` (`torch.triu(..., diagonal=1)`).", "torch.triu(M, diagonal=1)"),
        ("tril_indices", f"Return the (row, col) indices of the lower triangle of an {n}×{n} matrix (`torch.tril_indices`).",
         f"torch.tril_indices({n}, {n})"),
        ("triu_indices", f"Return the indices of the strictly upper triangle of an {n}×{n} matrix (`torch.triu_indices(..., "
                         f"offset=1)`).", f"torch.triu_indices({n}, {n}, offset=1)"),
        ("meshgrid", f"Return `[xx, yy]` from `torch.meshgrid(torch.arange({n}), torch.arange(2), indexing='ij')`.",
         f"xx, yy = torch.meshgrid(torch.arange({n}), torch.arange(2), indexing='ij')\nresult = [xx, yy]",
         {"require": ["meshgrid", "indexing"]}),
        ("cartesian_prod", f"Return every pair of an element of `torch.tensor({p})` with one of `torch.tensor({q})` "
                           f"(`torch.cartesian_prod`).", f"torch.cartesian_prod(torch.tensor({p}), torch.tensor({q}))"),
        ("combinations", f"Return all unordered pairs of `torch.arange({n + 1})` (`torch.combinations(..., r=2)`).",
         f"torch.combinations(torch.arange({n + 1}), r=2)"),
        ("block_diag", f"Return the block-diagonal matrix of `A = {_fmat(A)}` and a 1×1 block of ones (`torch.block_diag`).",
         f"torch.block_diag({_fmat(A)}, torch.ones(1, 1))"),
        ("vander", "Return the Vandermonde matrix of `v` with 3 columns (`torch.vander(v, N=3)`).", "torch.vander(v, N=3)"),
        ("kron", "Return the Kronecker product `torch.kron(torch.eye(2), M)`.", "torch.kron(torch.eye(2), M)"),
        ("complex", "Return the complex tensor whose real part is `v` and imaginary part is `2 * v` (`torch.complex`).",
         "torch.complex(v, 2 * v)"),
        ("polar", "Return the complex numbers with absolute values `v.abs()` and angle π/2 (`torch.polar(abs, angle)`).",
         "torch.polar(v.abs(), torch.full_like(v, math.pi / 2))"),
    ]
    return catalog(rng, items, setup, "")


@template(M, "easy")
def random_catalog(rng):
    s = seeded(rng)
    n = rng.randint(3, 6)
    setup = f"torch.manual_seed({s})"
    items = [
        ("rand", f"After the seed, draw a (2, {n}) tensor of uniform [0, 1) samples.", f"torch.rand(2, {n})"),
        ("randn", f"After the seed, draw {n} standard-normal samples.", f"torch.randn({n})"),
        ("randint", f"After the seed, draw {n} integers in [0, 10) (`torch.randint(0, 10, ({n},))`).", f"torch.randint(0, 10, ({n},))"),
        ("randperm", f"After the seed, return a random permutation of 0..{n - 1}.", f"torch.randperm({n})"),
        ("normal", f"After the seed, draw {n} samples from N(mean=0, std=2) (`torch.normal(mean=0.0, std=2.0, size=...)`).",
         f"torch.normal(mean=0.0, std=2.0, size=({n},))"),
        ("normal", "After the seed, draw one sample per (mean, std) pair: `torch.normal(torch.zeros(3), torch.ones(3))`.",
         "torch.normal(torch.zeros(3), torch.ones(3))"),
        ("bernoulli", f"After the seed, draw {n} 0/1 values that are 1 with probability 0.3 (`torch.bernoulli(torch.full"
                      f"(({n},), 0.3))`): a dropout-style mask.", f"torch.bernoulli(torch.full(({n},), 0.3))"),
        ("multinomial", "After the seed, sample 4 class ids from probabilities `[0.1, 0.6, 0.3]` with replacement "
                        "(`torch.multinomial`).", "torch.multinomial(torch.tensor([0.1, 0.6, 0.3]), num_samples=4, replacement=True)",
         {"require": ["multinomial", "replacement"]}),
        ("poisson", f"After the seed, draw {n} Poisson counts with rate 4 (`torch.poisson(torch.full(({n},), 4.0))`).",
         f"torch.poisson(torch.full(({n},), 4.0))"),
        ("uniform_", f"After the seed, fill `torch.empty({n})` in place with uniform samples from [-1, 1) (`uniform_`).",
         f"torch.empty({n}).uniform_(-1, 1)"),
        ("normal_", f"After the seed, fill `torch.empty({n})` in place with N(0, 1) samples (`normal_`).", f"torch.empty({n}).normal_(0, 1)"),
        ("exponential_", f"After the seed, fill `torch.empty({n})` with Exponential(1) samples (`exponential_`).",
         f"torch.empty({n}).exponential_()"),
        ("random_", f"After the seed, fill `torch.empty({n})` with integers in [0, 10) (`random_(0, 10)`).", f"torch.empty({n}).random_(0, 10)"),
        ("Generator", f"Use an **independent** random stream: `g = torch.Generator().manual_seed({s + 1})`, then return "
                      f"`torch.rand(2, generator=g)`.", f"g = torch.Generator().manual_seed({s + 1})\nresult = torch.rand(2, generator=g)",
         {"require": ["Generator", "generator"]}),
        ("rand_like", f"After the seed, return uniform samples shaped like `torch.zeros({n}, 2)` (`torch.rand_like`).",
         f"torch.rand_like(torch.zeros({n}, 2))"),
        ("randn_like", f"After the seed, return normal samples shaped like `torch.zeros(2, {n})` (`torch.randn_like`).",
         f"torch.randn_like(torch.zeros(2, {n}))"),
        ("randint_like", f"After the seed, return integers in [0, 10) shaped like `torch.zeros({n})` (`torch.randint_like`).",
         f"torch.randint_like(torch.zeros({n}), 10)"),
        ("manual_seed", "Reseeding repeats a stream: return `[a, b]` where `a = torch.rand(3)` after `torch.manual_seed(0)` "
                        "and `b` is drawn the same way after seeding again.",
         "torch.manual_seed(0)\na = torch.rand(3)\ntorch.manual_seed(0)\nb = torch.rand(3)\nresult = [a, b]"),
        ("use_deterministic_algorithms", "Turn on deterministic algorithms (`torch.use_deterministic_algorithms(True)`), "
                                         "return `torch.are_deterministic_algorithms_enabled()`, then turn them off again.",
         "torch.use_deterministic_algorithms(True)\nresult = torch.are_deterministic_algorithms_enabled()\n"
         "torch.use_deterministic_algorithms(False)"),
    ]
    return catalog(rng, items, setup, "")


@template(M, "easy")
def indexing_catalog(rng):
    base = rng.randint(0, 50)
    t = base + rng.randint(8, 16)
    j = rng.randint(0, 3)
    setup = f"x = torch.arange({base}, {base + 24}).reshape(2, 3, 4)"
    sc = {"setup": f"scores = {_fmat(fm(rng, 3, 4))}"}
    items = [
        ("", "Return `x[..., -1]`: the last element along the final axis (`...` stands for all leading dims).", "x[..., -1]"),
        ("", "Return the shape of `x[:, None]` as a tuple (`None` inserts a new axis).", "tuple(x[:, None].shape)"),
        ("index_select", "Return columns 0 and 2 of every matrix in `x` with `torch.index_select(x, dim=1, index=...)`.",
         "torch.index_select(x, dim=1, index=torch.tensor([0, 2]))"),
        ("take", "Treat `x` as flat and return elements 0, 5 and 23 (`torch.take`).", "torch.take(x, torch.tensor([0, 5, 23]))"),
        ("narrow", "Return the slice of length 2 starting at index 1 along dim 2 (`torch.narrow`).", "torch.narrow(x, dim=2, start=1, length=2)"),
        ("masked_select", f"Return the values of `x` greater than {t} as a 1-D tensor with `torch.masked_select`.",
         f"torch.masked_select(x, x > {t})"),
        ("where", f"With one argument, `torch.where(cond)` returns a tuple of index tensors. Return it for `x > {t}`.",
         f"torch.where(x > {t})"),
        ("nonzero", f"Return the (n, 3) indices of the entries of `x` greater than {t} (`torch.nonzero`).", f"torch.nonzero(x > {t})"),
        ("argwhere", f"Return the indices of the entries greater than {t} with `torch.argwhere`.", f"torch.argwhere(x > {t})"),
        ("nonzero", f"Return the indices of the entries greater than {t} as a tuple of 1-D tensors, one per dimension "
                    f"(`.nonzero(as_tuple=True)`).", f"(x > {t}).nonzero(as_tuple=True)", {"require": ["nonzero", "as_tuple"]}),
        ("clone", f"Masked assignment: copy `x` (`clone`), set every entry greater than {t} to 0, and return the copy.",
         f"y = x.clone()\ny[y > {t}] = 0\nresult = y"),
        ("take_along_dim", "Sort each row of `scores` by gathering with its own argsort: "
                           "`torch.take_along_dim(scores, scores.argsort(dim=1), dim=1)`.",
         "torch.take_along_dim(scores, scores.argsort(dim=1), dim=1)", sc),
        ("scatter_add_", "Accumulate like `bincount`: return `torch.zeros(5).scatter_add_(0, torch.tensor([0, 1, 1, 4]), "
                         "torch.ones(4))`.", "torch.zeros(5).scatter_add_(0, torch.tensor([0, 1, 1, 4]), torch.ones(4))"),
        ("scatter_reduce", "Return the per-slot maximum of `src = torch.tensor([5., 7., 1.])` at slots `[0, 0, 2]`, "
                           "ignoring the initial zeros: `torch.zeros(3).scatter_reduce(0, idx, src, reduce='amax', include_self=False)`.",
         "torch.zeros(3).scatter_reduce(0, torch.tensor([0, 0, 2]), torch.tensor([5., 7., 1.]), reduce='amax', include_self=False)",
         {"require": ["scatter_reduce", "include_self"]}),
        ("index_add_", "On `z = torch.zeros(3, 4)`, add a row of ones into rows 0 and 2 (`z.index_add_(0, torch.tensor([0, 2]), "
                       "torch.ones(2, 4))`) and return `z`.", "z = torch.zeros(3, 4)\nz.index_add_(0, torch.tensor([0, 2]), torch.ones(2, 4))\nresult = z"),
        ("index_fill_", f"On `z = torch.zeros(3, 4)`, fill column {j} with -1 in place (`index_fill_`) and return `z`.",
         f"z = torch.zeros(3, 4)\nz.index_fill_(1, torch.tensor([{j}]), -1.0)\nresult = z"),
        ("index_put_", "On `z = torch.zeros(3, 4)`, add 1 twice at position (0, 1) with `z.index_put_((rows, cols), vals, "
                       "accumulate=True)` and return `z`.",
         "z = torch.zeros(3, 4)\nz.index_put_((torch.tensor([0, 0]), torch.tensor([1, 1])), torch.tensor([1., 1.]), accumulate=True)\nresult = z",
         {"require": ["index_put_", "accumulate"]}),
        ("unbind", "Split `x` into a tuple of its 2 matrices (`torch.unbind(x, dim=0)`).", "torch.unbind(x, dim=0)"),
    ]
    return catalog(rng, items, setup, f"`x` is a (2, 3, 4) int tensor of the numbers {base}..{base + 23}.")


@template(M, "easy")
def joining_catalog(rng):
    a, b = fm(rng, 2, 3), fm(rng, 2, 3)
    n, k = rng.randint(7, 11), rng.randint(1, 3)
    setup = f"a = {_fmat(a)}\nb = {_fmat(b)}\nt = torch.arange({n})"
    items = [
        ("hstack", "Stack `a` and `b` side by side (`torch.hstack`).", "torch.hstack([a, b])"),
        ("vstack", "Stack `a` on top of `b` (`torch.vstack`).", "torch.vstack([a, b])"),
        ("dstack", "Stack `a` and `b` along a new third axis (`torch.dstack`).", "torch.dstack([a, b])"),
        ("column_stack", "Make a 3×2 matrix whose columns are `a[0]` and `b[0]` (`torch.column_stack`).", "torch.column_stack([a[0], b[0]])"),
        ("tensor_split", f"Split `t` into 3 nearly equal parts with `torch.tensor_split`.", "torch.tensor_split(t, 3)"),
        ("split", f"Split `t` into pieces of size 3 (`torch.split(t, 3)`; the last may be smaller).", "torch.split(t, 3)"),
        ("flip", "Reverse `t` with `torch.flip(..., dims=[0])`.", "torch.flip(t, dims=[0])"),
        ("roll", f"Rotate `t` by {k} positions to the right (`torch.roll(..., shifts={k})`).", f"torch.roll(t, shifts={k})"),
        ("roll", "Rotate each row of `a` one step to the right (`torch.roll(a, shifts=1, dims=1)`).", "torch.roll(a, shifts=1, dims=1)"),
        ("rot90", "Rotate `a` by 90° counter-clockwise (`torch.rot90(a, 1, dims=(0, 1))`).", "torch.rot90(a, 1, dims=(0, 1))"),
        ("fliplr", "Mirror `a` left-to-right (`torch.fliplr`).", "torch.fliplr(a)"),
        ("repeat", "Tile `a` 2 times down and 3 times across with `a.repeat(2, 3)` (a copy).", "a.repeat(2, 3)"),
        ("repeat_interleave", f"Repeat each element of `t[:3]` {k + 1} times in place (`torch.repeat_interleave`).",
         f"torch.repeat_interleave(t[:3], {k + 1})"),
        ("tile", "Return `torch.tile(t[:2], (3,))`.", "torch.tile(t[:2], (3,))"),
        ("cat", "Join `a` and `b` into a 2×6 matrix with `torch.cat(..., dim=1)`.", "torch.cat([a, b], dim=1)"),
    ]
    return catalog(rng, items, setup, "")


@template(M, "easy")
def functions_catalog(rng):
    n = rng.randint(3, 5)
    a = fv(rng, n, -2, 2, 2, nonzero=True)
    b = [round(v, 2) for v in fv(rng, n, 0.1, 3, 2)]
    p = [round(v, 2) for v in fv(rng, n, 0.05, 0.95, 2)]
    setup = f"a = {ftt(a)}\nb = {ftt(b)}   # positive\np = {ftt(p)}   # probabilities"
    items = [
        ("exp", "Return `torch.exp(a)`.", "torch.exp(a)"),
        ("log", "Return `torch.log(b)`.", "torch.log(b)"),
        ("log1p", "Return `log(1 + p)` accurately for small values (`torch.log1p`).", "torch.log1p(p)"),
        ("expm1", "Return `exp(a) − 1` accurately near 0 (`torch.expm1`).", "torch.expm1(a)"),
        ("sin", "Return `torch.sin(a)`.", "torch.sin(a)"),
        ("cos", "Return `torch.cos(a)`.", "torch.cos(a)"),
        ("tanh", "Return `torch.tanh(a)`.", "torch.tanh(a)"),
        ("sigmoid", "Return `torch.sigmoid(a)`.", "torch.sigmoid(a)"),
        ("logit", "Return the log-odds of `p` (`torch.logit`, the inverse of sigmoid).", "torch.logit(p)"),
        ("erf", "Return the error function of `a` (`torch.erf`).", "torch.erf(a)"),
        ("atan2", "Return the angle of each point (x = `b`, y = `a`) with `torch.atan2(a, b)`.", "torch.atan2(a, b)"),
        ("logaddexp", "Return `log(exp(a) + exp(b))` computed stably (`torch.logaddexp`).", "torch.logaddexp(a, b)"),
        ("xlogy", "Return `p · log(b)` with `torch.xlogy` (defined as 0 where p = 0).", "torch.xlogy(p, b)"),
        ("nan_to_num", "Clean `s = torch.tensor([float('nan'), 1.0, float('inf')])`: replace NaN with 0 and +inf with 100 "
                       "(`torch.nan_to_num(..., nan=0.0, posinf=100.0)`).",
         "torch.nan_to_num(torch.tensor([float('nan'), 1.0, float('inf')]), nan=0.0, posinf=100.0)"),
    ]
    return catalog(rng, items, setup, "")


@template(M, "easy")
def reduction_catalog(rng):
    r, c = rng.randint(2, 3), rng.randint(3, 4)
    m = fm(rng, r, c, 0.5, 4, 1)
    t = sorted(v for row in m for v in row)[len(m) * c // 2]
    setup = f"M = {_fmat(m)}"
    items = [
        ("sum", "Sum `M` over both dimensions at once (`M.sum(dim=(0, 1))`).", "M.sum(dim=(0, 1))"),
        ("prod", "Return the product of all entries of `M`.", "M.prod()", {"tol": 1e-4}),
        ("var", "Return the (unbiased) variance of all entries of `M`.", "M.var()"),
        ("std", "Return the **population** std of each column (`std(dim=0, correction=0)`).", "M.std(dim=0, correction=0)",
         {"require": ["std", "correction"]}),
        ("std_mean", "Return the std and mean of each column in one call (`torch.std_mean(M, dim=0)`).", "torch.std_mean(M, dim=0)"),
        ("var_mean", "Return the variance and mean of all of `M` in one call (`torch.var_mean`).", "torch.var_mean(M)"),
        ("max", "Return the largest entry of `M` (`M.max()` without `dim`).", "M.max()"),
        ("amax", "Return the largest entry of each row (`M.amax(dim=1)`: values only).", "M.amax(dim=1)"),
        ("amin", "Return the smallest entry of `M` with `amin`.", "M.amin()"),
        ("aminmax", "Return the min and max of `M` together (`torch.aminmax`).", "torch.aminmax(M)"),
        ("argmin", "Return the row index of each column's minimum, keeping the reduced dim (`argmin(dim=0, keepdim=True)`).",
         "M.argmin(dim=0, keepdim=True)", {"require": ["argmin", "keepdim"]}),
        ("all", f"Are all entries of `M` greater than {t}? (`.all()`)", f"(M > {t}).all()"),
        ("any", f"For each column, is any entry greater than {t}? (`.any(dim=0)`)", f"(M > {t}).any(dim=0)"),
        ("count_nonzero", f"Count the entries greater than {t} in each column (`torch.count_nonzero(..., dim=0)`).",
         f"torch.count_nonzero(M > {t}, dim=0)"),
        ("norm", "Return the L2 norm of each row (`M.norm(dim=1)`).", "M.norm(dim=1)"),
        ("vector_norm", "Return the L1 norm of each row with `torch.linalg.vector_norm(M, ord=1, dim=1)`.",
         "torch.linalg.vector_norm(M, ord=1, dim=1)"),
        ("matrix_norm", "Return the Frobenius norm of `M` (`torch.linalg.matrix_norm`).", "torch.linalg.matrix_norm(M)"),
    ]
    return catalog(rng, items, setup, "")


# ================================================================= medium

@template(M, "medium")
def stats_scan_catalog(rng):
    r, c = rng.randint(2, 3), rng.randint(3, 5)
    m = fm(rng, r, c, 0.5, 4, 1)
    vals = [rng.randint(0, 3) for _ in range(5)]
    setup = f"M = {_fmat(m)}"
    nan = {"setup": "w = torch.tensor([1.0, float('nan'), 3.0, 4.0])"}
    items = [
        ("median", "Return the median of all entries of `M` (`M.median()`: for an even count, the lower middle value).",
         "M.median()"),
        ("median", "Return each row's median value (`M.median(dim=1).values`).", "M.median(dim=1).values"),
        ("mode", f"Return the most common value in each row of `k = torch.tensor([{vals}, {vals[::-1]}])` (`mode(dim=1)`).",
         f"torch.tensor([{vals}, {vals[::-1]}]).mode(dim=1).values"),
        ("quantile", "Return the median of `M` with `M.quantile(0.5)` (it interpolates, unlike `median`).", "M.quantile(0.5)"),
        ("quantile", "Return each row's 25th and 75th percentiles: `torch.quantile(M, torch.tensor([0.25, 0.75]), dim=1)`.",
         "torch.quantile(M, torch.tensor([0.25, 0.75]), dim=1)"),
        ("nansum", "Sum `w` ignoring NaN (`nansum`).", "w.nansum()", nan),
        ("nanmean", "Average `w` ignoring NaN (`nanmean`).", "w.nanmean()", nan),
        ("logsumexp", "Return the stable `log(sum(exp(M)))` of each row (`torch.logsumexp(M, dim=1)`).", "torch.logsumexp(M, dim=1)"),
        ("cumsum", "Return the running sum along each row (`torch.cumsum(M, dim=1)`).", "torch.cumsum(M, dim=1)"),
        ("cumprod", "Return the running product along each row (`torch.cumprod`).", "torch.cumprod(M, dim=1)", {"tol": 1e-4}),
        ("cummax", "Return the running maximum along each row (`torch.cummax(M, dim=1).values`).", "torch.cummax(M, dim=1).values"),
        ("logcumsumexp", "Return the running `log(sum(exp(...)))` along each row (`torch.logcumsumexp`).",
         "torch.logcumsumexp(M, dim=1)"),
        ("diff", "Return the differences between neighbouring columns (`torch.diff(M, dim=1)`).", "torch.diff(M, dim=1)"),
        ("trapezoid", "Integrate each row with the trapezoid rule, unit spacing (`torch.trapezoid(M, dim=1)`).",
         "torch.trapezoid(M, dim=1)"),
    ]
    return catalog(rng, items, setup, "")


@template(M, "medium")
def sort_search_catalog(rng):
    v = [rng.randint(0, 6) for _ in range(rng.randint(6, 9))]
    m = fm(rng, 2, 4)
    bounds = sorted(rng.sample(range(1, 10), 3))
    q = [rng.randint(0, 10) for _ in range(3)]
    setup = f"v = {tt(v)}\nM = {_fmat(m)}"
    items = [
        ("sort", "Sort each row of `M` in descending order and return the values (`torch.sort(..., descending=True)`).",
         "torch.sort(M, dim=1, descending=True).values"),
        ("argsort", "Return the indices that sort each row of `M` ascending (`torch.argsort(M, dim=1)`).", "torch.argsort(M, dim=1)"),
        ("topk", "Return the column indices of the 2 largest values in each row of `M` (`torch.topk(M, k=2, dim=1)`).",
         "torch.topk(M, k=2, dim=1).indices"),
        ("kthvalue", "Return the 2nd smallest value of each row of `M` (`torch.kthvalue(M, k=2, dim=1)`).",
         "torch.kthvalue(M, k=2, dim=1).values"),
        ("unique", "Return `torch.unique(v, return_inverse=True, return_counts=True)`: the sorted distinct values, where "
                   "each element maps, and how often each value occurs.", "torch.unique(v, return_inverse=True, return_counts=True)",
         {"require": ["unique", "return_counts"]}),
        ("unique_consecutive", "Collapse runs of repeated values in `v` (`torch.unique_consecutive`).", "torch.unique_consecutive(v)"),
        ("bincount", "Count each value 0..9 in `v` (`torch.bincount(v, minlength=10)`).", "torch.bincount(v, minlength=10)"),
        ("histc", "Histogram `v.float()` into 4 equal bins over [0, 8] (`torch.histc(..., bins=4, min=0, max=8)`).",
         "torch.histc(v.float(), bins=4, min=0, max=8)"),
        ("histogram", "Return the counts and bin edges of `v.float()` in 3 bins (`torch.histogram(..., bins=3)`).",
         "torch.histogram(v.float(), bins=3)"),
        ("searchsorted", f"For each value of `torch.tensor({q})`, return where it would be inserted into the sorted "
                         f"`torch.tensor({bounds})` (`torch.searchsorted`).", f"torch.searchsorted(torch.tensor({bounds}), torch.tensor({q}))"),
        ("bucketize", f"Return the bucket index of each value of `torch.tensor({q})` for boundaries `torch.tensor({bounds})` "
                      f"(`torch.bucketize`).", f"torch.bucketize(torch.tensor({q}), torch.tensor({bounds}))"),
        ("isin", f"Return which entries of `v` are in `torch.tensor({q})` (`torch.isin`).", f"torch.isin(v, torch.tensor({q}))"),
    ]
    return catalog(rng, items, setup, "")


@template(M, "medium")
def products_catalog(rng):
    A, B, C = fm(rng, 3, 4), fm(rng, 4, 2), fm(rng, 3, 2)
    u, w = fv(rng, 4), fv(rng, 4)
    p3, q3 = fv(rng, 3), fv(rng, 3)
    setup = f"A = {_fmat(A)}\nB = {_fmat(B)}\nu = {ftt(u)}\nw = {ftt(w)}"
    batch = {"setup": "torch.manual_seed(0)\nX, Y = torch.randn(2, 3, 4), torch.randn(2, 4, 2)"}
    items = [
        ("matmul", "Return `A @ B` with `torch.matmul`.", "torch.matmul(A, B)"),
        ("mm", "Return `A @ B` with `torch.mm` (2-D only, no broadcasting).", "torch.mm(A, B)"),
        ("bmm", "Multiply the batches `X` (2, 3, 4) and `Y` (2, 4, 2) matrix by matrix (`torch.bmm`).", "torch.bmm(X, Y)", batch),
        ("mv", "Return the matrix-vector product of `A` and `u` (`torch.mv`).", "torch.mv(A, u)"),
        ("dot", "Return the dot product of `u` and `w` (`torch.dot`).", "torch.dot(u, w)"),
        ("inner", "Return the inner product of `u` and `w` (`torch.inner`).", "torch.inner(u, w)"),
        ("outer", "Return the outer product of `u` and `w` (`torch.outer`).", "torch.outer(u, w)"),
        ("vecdot", "Return `torch.linalg.vecdot(u, w)`.", "torch.linalg.vecdot(u, w)"),
        ("einsum", "Return `A @ B` with `torch.einsum('ij,jk->ik', ...)`.", "torch.einsum('ij,jk->ik', A, B)"),
        ("tensordot", "Contract `A`'s dim 1 with `B`'s dim 0 (`torch.tensordot(A, B, dims=([1], [0]))`).",
         "torch.tensordot(A, B, dims=([1], [0]))"),
        ("cross", f"Return the cross product of the 3-vectors `{ftt(p3)}` and `{ftt(q3)}` (`torch.linalg.cross`).",
         f"torch.linalg.cross({ftt(p3)}, {ftt(q3)})"),
        ("addmm", f"Return the fused `0.5·C + 2·(A @ B)` for `C = {_fmat(C)}` (`torch.addmm(C, A, B, beta=0.5, alpha=2.0)`).",
         f"torch.addmm({_fmat(C)}, A, B, beta=0.5, alpha=2.0)", {"require": ["addmm", "beta"]}),
        ("baddbmm", "Return `torch.baddbmm(torch.zeros(2, 3, 2), X, Y)`: a batched `addmm`.", "torch.baddbmm(torch.zeros(2, 3, 2), X, Y)", batch),
        ("multi_dot", "Return `A @ B @ B.T` with `torch.linalg.multi_dot` (it picks the cheapest order).",
         "torch.linalg.multi_dot([A, B, B.T])"),
    ]
    return catalog(rng, items, setup, "", tol=1e-4)


@template(M, "medium")
def spectral_misc_catalog(rng):
    n = rng.choice([4, 8])
    sig = fv(rng, n)
    setup = f"sig = {ftt(sig)}"
    items = [
        ("rfft", f"Return the magnitudes of the real FFT of `sig` (`torch.fft.rfft(sig).abs()`: {n // 2 + 1} frequencies).",
         "torch.fft.rfft(sig).abs()", {"tol": 1e-4}),
        ("fftfreq", f"Return the sample frequencies of a length-{n} FFT (`torch.fft.fftfreq({n})`).", f"torch.fft.fftfreq({n})"),
        ("fft2", "Return the real part of the 2-D FFT of `sig.reshape(2, -1)` (`torch.fft.fft2(...).real`).",
         "torch.fft.fft2(sig.reshape(2, -1)).real", {"tol": 1e-4}),
        ("sparse_coo_tensor", "Build the 2×2 sparse COO tensor with value 3 at (0, 1) and 4 at (1, 0) "
                              "(`torch.sparse_coo_tensor(indices, values, size)`), then return it dense (`.to_dense()`).",
         "torch.sparse_coo_tensor(torch.tensor([[0, 1], [1, 0]]), torch.tensor([3., 4.]), (2, 2)).to_dense()",
         {"require": ["sparse_coo_tensor", "to_dense"]}),
        ("set_printoptions", "Print `sig` with 2 decimals and no scientific notation: "
                             "`torch.set_printoptions(precision=2, sci_mode=False, linewidth=120)`, then `print(sig)`. (Graded on the output.)",
         "torch.set_printoptions(precision=2, sci_mode=False, linewidth=120)\nprint(sig)", {"answer": [STDOUT]}),
    ]
    return catalog(rng, items, setup, "")


# ================================================================= hard

@template(M, "hard")
def decomposition_catalog(rng):
    R = [[rng.randint(-2, 2) for _ in range(3)] for _ in range(3)]
    A = fm(rng, 3, 4)
    b = fv(rng, 3)
    setup = (f"R = {_fmat(R)}\nS = R @ R.T + 3 * torch.eye(3)   # symmetric positive-definite\nA = {_fmat(A)}\n"
             f"b = {ftt(b)}")
    items = [
        ("trace", "Return the trace of `S` (`torch.trace`).", "torch.trace(S)"),
        ("diagonal", "Return the diagonal of `S` with `torch.diagonal`.", "torch.diagonal(S)"),
        ("det", "Return the determinant of `S` (`torch.linalg.det`).", "torch.linalg.det(S)"),
        ("slogdet", "Return `torch.linalg.slogdet(S)`: the sign and log|det| (stable for large determinants).",
         "torch.linalg.slogdet(S)"),
        ("inv", "Return the inverse of `S` (`torch.linalg.inv`).", "torch.linalg.inv(S)"),
        ("pinv", "Return the pseudo-inverse of the 3×4 matrix `A` (`torch.linalg.pinv`).", "torch.linalg.pinv(A)"),
        ("solve", "Solve `S x = b` with `torch.linalg.solve` (better than `inv(S) @ b`).", "torch.linalg.solve(S, b)"),
        ("lstsq", "Find the least-squares solution of `A.T x ≈ c` for `c = torch.ones(4, 1)`: "
                  "`torch.linalg.lstsq(A.T, c).solution`.", "torch.linalg.lstsq(A.T, torch.ones(4, 1)).solution"),
        ("eigh", "Return the eigenvalues of the symmetric `S` in ascending order (`torch.linalg.eigh(S).eigenvalues`).",
         "torch.linalg.eigh(S).eigenvalues"),
        ("eigvalsh", "Return the eigenvalues of `S` with `torch.linalg.eigvalsh`.", "torch.linalg.eigvalsh(S)"),
        ("eig", "`torch.linalg.eig` handles general matrices and returns complex results. Return the real parts of `S`'s "
                "eigenvalues, sorted ascending.", "torch.linalg.eig(S).eigenvalues.real.sort().values"),
        ("svd", "Return the singular values of `A` from `torch.linalg.svd(A, full_matrices=False)`.",
         "torch.linalg.svd(A, full_matrices=False).S", {"require": ["svd", "full_matrices"]}),
        ("svdvals", "Return the singular values of `A` with `torch.linalg.svdvals`.", "torch.linalg.svdvals(A)"),
        ("qr", "Factor `A.T = QR` with `torch.linalg.qr` and return `torch.allclose(Q @ R, A.T, atol=1e-5)`.",
         "Q, R2 = torch.linalg.qr(A.T)\nresult = torch.allclose(Q @ R2, A.T, atol=1e-5)"),
        ("cholesky", "Return the lower-triangular Cholesky factor of `S` (`torch.linalg.cholesky`).", "torch.linalg.cholesky(S)"),
        ("cholesky_solve", "Solve `S x = b` using its Cholesky factor `L`: `torch.cholesky_solve(b[:, None], L)`.",
         "L = torch.linalg.cholesky(S)\nresult = torch.cholesky_solve(b[:, None], L)"),
        ("matrix_rank", "Return the rank of `A` (`torch.linalg.matrix_rank`).", "torch.linalg.matrix_rank(A)"),
        ("matrix_power", "Return `S` cubed with `torch.linalg.matrix_power`.", "torch.linalg.matrix_power(S, 3)", {"tol": 1e-3}),
        ("matrix_exp", "Return the matrix exponential of `S / 10` (`torch.linalg.matrix_exp`).", "torch.linalg.matrix_exp(S / 10)",
         {"tol": 1e-4}),
        ("cond", "Return the condition number of `S` (`torch.linalg.cond`).", "torch.linalg.cond(S)", {"tol": 1e-4}),
        ("svd_lowrank", "Return the 2 largest singular values of `A` from the randomized `torch.svd_lowrank(A, q=2)` "
                        "(`.S`; it returns U, S, V).", "torch.svd_lowrank(A, q=2)[1]", {"tol": 1e-3}),
        ("pca_lowrank", "Return the 2 leading singular values of the centered data `A.T` from `torch.pca_lowrank(A.T, q=2)`.",
         "torch.pca_lowrank(A.T, q=2)[1]", {"tol": 1e-3}),
    ]
    return catalog(rng, items, setup, "", tol=1e-4)
