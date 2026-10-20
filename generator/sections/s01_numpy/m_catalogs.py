"""NumPy catalog templates: the rest of the guide's API, one function per problem (1.01-1.18)."""

from ._common import STDOUT, arr, catalog, floats, lit, mat, seeded, template


def _ints(rng, n, lo=0, hi=9):
    return [rng.randint(lo, hi) for _ in range(n)]


# ================================================================= 1.01 printing

@template("1.01", "lite")
def print_misc_catalog(rng):
    n, p = rng.randint(2, 5), rng.randint(1, 3)
    items = [
        ("printoptions", f"Print `np.pi * np.ones({n})` with {p} decimals **temporarily**: inside `with np.printoptions"
                         f"(precision={p}):` (the options are restored after the block). Graded on the output.",
         f"with np.printoptions(precision={p}):\n    print(np.pi * np.ones({n}))", {"answer": [STDOUT], "require": ["printoptions", "ones"]}),
        ("pi", f"Return `np.pi * np.arange({n})` rounded to {p} decimals (`np.round`).", f"np.round(np.pi * np.arange({n}), {p})"),
    ]
    return catalog(rng, items, "")


# ================================================================= 1.02 creating arrays

@template("1.02", "lite")
def creation_catalog(rng):
    r, c = rng.randint(2, 4), rng.randint(2, 4)
    vals = _ints(rng, 3, 1, 9)
    setup = f"lst = {lit(vals)}\nb = {mat([[float(v) for v in _ints(rng, c, 0, 9)] for _ in range(c)])}"
    items = [
        ("asarray", "Convert `lst` with `np.asarray` (no copy when the input is already an array).", "np.asarray(lst)"),
        ("copy", "Make an independent copy of `b` (`np.copy`), set its [0, 0] to -1, and return `[b, copy]`.",
         "c = np.copy(b)\nc[0, 0] = -1\nresult = [b, c]"),
        ("diagflat", "Build a 2-D array with `lst` on its diagonal using `np.diagflat`.", "np.diagflat(lst)"),
        ("empty", f"Allocate an **uninitialized** ({r}, {c}) array with `np.empty` and return its shape (values are garbage).",
         f"np.empty(({r}, {c})).shape"),
        ("empty_like", "Return `[e.shape, e.dtype]` for `e = np.empty_like(b)`.", "e = np.empty_like(b)\nresult = [e.shape, e.dtype]"),
        ("identity", f"Return the {r}×{r} identity matrix with `np.identity`.", f"np.identity({r})"),
        ("ndim", "Return `[b.ndim, b.size]`.", "[b.ndim, b.size]", {"require": ["ndim", "size"]}),
        ("linspace", f"Return {r + 3} evenly spaced points in [0, 1) **excluding** 1 (`np.linspace(0, 1, {r + 3}, endpoint=False)`).",
         f"np.linspace(0, 1, {r + 3}, endpoint=False)", {"require": ["linspace", "endpoint"]}),
        ("linspace", f"Return both the points and the spacing of `np.linspace(0, 1, {r + 3}, retstep=True)`.",
         f"pts, step = np.linspace(0, 1, {r + 3}, retstep=True)\nresult = [pts, step]", {"require": ["linspace", "retstep"]}),
        ("array", "Make a copy of `b` with `np.array(b, copy=True)`, set its [0, 0] to 99, and return `b[0, 0]` (unchanged).",
         "c = np.array(b, copy=True)\nc[0, 0] = 99\nresult = b[0, 0]", {"require": ["array", "copy"]}),
        ("mgrid", f"Return the shape of the dense grid `np.mgrid[0:{r}, 0:{c}]`.", f"np.mgrid[0:{r}, 0:{c}].shape"),
        ("ogrid", f"`np.ogrid[0:{r}, 0:{c}]` returns an **open** grid of two broadcastable arrays. Return their shapes.",
         f"g = np.ogrid[0:{r}, 0:{c}]\nresult = [g[0].shape, g[1].shape]"),
    ]
    return catalog(rng, items, setup)


# ================================================================= 1.03 random

@template("1.03", "easy")
def random_dist_catalog(rng):
    s, n = seeded(rng), rng.randint(3, 6)
    setup = f"rng = np.random.default_rng({s})"
    items = [
        ("laplace", f"Draw {n} samples from a Laplace(0, 1) distribution (`rng.laplace(0, 1, {n})`).", f"rng.laplace(0, 1, {n})"),
        ("normal", f"Draw {n} samples from N(10, 2²) with keyword arguments (`rng.normal(loc=10, scale=2, size={n})`).",
         f"rng.normal(loc=10, scale=2, size={n})", {"require": ["normal", "loc"]}),
        ("binomial", f"Draw {n} binomial counts of 10 trials with p = 0.5 (`rng.binomial(n=10, p=0.5, size={n})`).",
         f"rng.binomial(n=10, p=0.5, size={n})", {"require": ["binomial"]}),
        ("default_rng", f"Make a fresh generator with `np.random.default_rng(seed=42)` and return `.integers(0, 100, {n})`.",
         f"np.random.default_rng(seed=42).integers(0, 100, {n})", {"require": ["default_rng", "seed"]}),
        ("multinomial", "Split 10 trials over three outcomes with probabilities `[0.2, 0.3, 0.5]` (`rng.multinomial`).",
         "rng.multinomial(10, [0.2, 0.3, 0.5])"),
        ("dirichlet", f"Draw {n} probability vectors from a symmetric Dirichlet over 3 categories (`rng.dirichlet([1, 1, 1], "
                      f"size={n})`); each row sums to 1.", f"rng.dirichlet([1, 1, 1], size={n})"),
    ]
    return catalog(rng, items, setup)


# ================================================================= 1.04 attributes

@template("1.04", "lite")
def attributes_catalog(rng):
    r, c = rng.randint(2, 4), rng.randint(2, 5)
    setup = f"a = np.arange({r * c})\nM = a.reshape({r}, {c})"
    items = [
        ("T", "Return the shape of `M.T`.", "M.T.shape"),
        ("base", "Return `[M.base is a, a.base is None]`: `reshape` returned a view of `a`, which owns its data.",
         "[M.base is a, a.base is None]"),
        ("flags", "Return `[a.flags['OWNDATA'], M.flags['OWNDATA']]`.", "[a.flags['OWNDATA'], M.flags['OWNDATA']]"),
        ("flags", "Return `[M.flags['C_CONTIGUOUS'], M.T.flags['C_CONTIGUOUS']]`: row-major vs its transposed view.",
         "[M.flags['C_CONTIGUOUS'], M.T.flags['C_CONTIGUOUS']]"),
        ("reshape", f"Return whether `a.reshape({c}, {r}).base is a` (reshape returns a view when it can).", f"a.reshape({c}, {r}).base is a"),
        ("astype", "`astype(..., copy=False)` skips the copy when the dtype already matches: return `a.astype(a.dtype, copy=False) is a`.",
         "a.astype(a.dtype, copy=False) is a", {"require": ["astype", "copy"]}),
    ]
    return catalog(rng, items, setup)


# ================================================================= 1.05 indexing

@template("1.05", "easy")
def index_tools_catalog(rng):
    n = 3
    Mx = [[v for v in _ints(rng, n, -9, 15)] for _ in range(n)]
    t = rng.randint(2, 8)
    setup = f"M = {mat(Mx)}"
    items = [
        ("take", "Treat `M` as flat and return elements 0, 4 and 8 (`np.take`).", "np.take(M, [0, 4, 8])"),
        ("put", "On `C = M.copy()`, assign -5 and -6 to flat positions 0 and 1 with `np.put`, and return `C`.",
         "C = M.copy()\nnp.put(C, [0, 1], [-5, -6])\nresult = C", {"require": ["put", "copy"]}),
        ("choose", "Return `np.choose([0, 1, 0], M[:2])`: pick from row 0, then row 1, then row 0 again, column by column.",
         "np.choose([0, 1, 0], M[:2])"),
        ("compress", "Keep rows 0 and 2 of `M` with `np.compress([True, False, True], M, axis=0)`.", "np.compress([True, False, True], M, axis=0)"),
        ("extract", f"Return the entries of `M` greater than {t} with `np.extract`.", f"np.extract(M > {t}, M)"),
        ("place", f"On `C = M.copy()`, set every entry greater than {t} to 0 with `np.place(C, C > {t}, [0])`; return `C`.",
         f"C = M.copy()\nnp.place(C, C > {t}, [0])\nresult = C"),
        ("putmask", "On `C = M.copy()`, clip negatives to 0 in place with `np.putmask(C, C < 0, 0)`; return `C`.",
         "C = M.copy()\nnp.putmask(C, C < 0, 0)\nresult = C"),
        ("zeros", "Return the index array `np.zeros((3, 1), dtype=int)` used with `np.put_along_axis` to target column 0, and "
                  "the result of `np.put_along_axis(C, idx, 99, axis=1)` on `C = M.copy()`.",
         "C = M.copy()\nidx = np.zeros((3, 1), dtype=int)\nnp.put_along_axis(C, idx, 99, axis=1)\nresult = C", {"require": ["zeros", "put_along_axis"]}),
    ]
    return catalog(rng, items, setup)


# ================================================================= 1.06 views & in-place

@template("1.06", "lite")
def inplace_catalog(rng):
    n, k = rng.randint(3, 6), rng.randint(1, 5)
    items = [
        ("ones", f"Create `x = np.ones({n})` and add {k} **in place** with `x += {k}` (no new allocation). Return `x`.",
         f"x = np.ones({n})\nx += {k}\nresult = x"),
        ("sliding_window_view", f"Return every window of 3 consecutive values of `np.arange({n + 3})` as a read-only view "
                                f"(`np.lib.stride_tricks.sliding_window_view(x, window_shape=3)`).",
         f"x = np.arange({n + 3})\nresult = np.lib.stride_tricks.sliding_window_view(x, window_shape=3)", {"require": ["sliding_window_view", "window_shape"]}),
        ("ones", f"Create `x = np.ones({n})`, make it read-only (`x.flags.writeable = False`), and return True if `x[0] = 5` "
                 f"raises ValueError.", f"x = np.ones({n})\nx.flags.writeable = False\ntry:\n    x[0] = 5\n    result = False\n"
                                     f"except ValueError:\n    result = True", {"require": ["ones", "writeable"]}),
    ]
    return catalog(rng, items, "")


# ================================================================= 1.07 reshaping

@template("1.07", "lite")
def reshape_catalog(rng):
    a, b, c = rng.randint(2, 3), rng.randint(2, 4), rng.randint(2, 4)
    setup = f"z = np.arange({a * b * c}).reshape({a}, {b}, {c})"
    items = [
        ("atleast_1d", "Return the shape of `np.atleast_1d(5)`.", "np.atleast_1d(5).shape"),
        ("atleast_3d", f"Return the shape of `np.atleast_3d(np.arange({c}))`.", f"np.atleast_3d(np.arange({c})).shape"),
        ("flatten", "Return `z.flatten()` (always a copy, unlike `ravel`).", "z.flatten()"),
        ("moveaxis", "Move the last axis of `z` to the front with `np.moveaxis` and return the new shape.", "np.moveaxis(z, -1, 0).shape"),
        ("permute_dims", "Return the shape of `np.permute_dims(z, (2, 0, 1))` (the array-API name for transpose).",
         "np.permute_dims(z, (2, 0, 1)).shape"),
        ("resize", f"`np.resize` repeats the data to fill a new shape. Return `np.resize(np.arange(3), ({b}, {c}))`.",
         f"np.resize(np.arange(3), ({b}, {c}))"),
    ]
    return catalog(rng, items, setup)


# ================================================================= 1.08 joining & splitting

@template("1.08", "easy")
def join_split_catalog(rng):
    u, w = _ints(rng, 2, 0, 9), _ints(rng, 2, 0, 9)
    n = rng.choice([4, 6])
    setup = f"u = {arr(u)}\nw = {arr(w)}\nA = np.arange({n * n}).reshape({n}, {n})"
    items = [
        ("c_", "Stack `u` and `w` as columns with the index trick `np.c_[u, w]`.", "np.c_[u, w]"),
        ("column_stack", "Stack `u` and `w` as columns with `np.column_stack`.", "np.column_stack([u, w])"),
        ("concat", "Join `u` and `w` with `np.concat` (the array-API name of concatenate).", "np.concat([u, w])"),
        ("hstack", "Join `u` and `w` end to end with `np.hstack`.", "np.hstack([u, w])"),
        ("vstack", "Stack `u` on top of `w` with `np.vstack`.", "np.vstack([u, w])"),
        ("dstack", "Stack `u` and `w` along a third axis with `np.dstack` and return the shape.", "np.dstack([u, w]).shape"),
        ("hsplit", "Split `A` into 2 equal blocks of columns (`np.hsplit`).", "np.hsplit(A, 2)"),
        ("vsplit", "Split `A` into 2 equal blocks of rows (`np.vsplit`).", "np.vsplit(A, 2)"),
        ("eye", f"Return `np.eye({n}, k=1)`: ones on the first superdiagonal.", f"np.eye({n}, k=1)"),
        ("ones", f"Return `np.block([[np.eye(2), np.ones((2, 2))], [np.zeros((2, 2)), np.eye(2)]])`.",
         "np.block([[np.eye(2), np.ones((2, 2))], [np.zeros((2, 2)), np.eye(2)]])", {"require": ["block", "ones"]}),
        ("pad", "Surround `A` with a border of zeros 1 wide (`np.pad(A, pad_width=1, mode='constant')`).",
         "np.pad(A, pad_width=1, mode='constant')", {"require": ["pad", "pad_width"]}),
        ("trim_zeros", "Strip leading and trailing zeros from `np.concatenate([[0, 0], u + 1, [0]])` (`np.trim_zeros`).",
         "np.trim_zeros(np.concatenate([[0, 0], u + 1, [0]]))"),
    ]
    return catalog(rng, items, setup)


# ================================================================= 1.09 broadcasting

@template("1.09", "easy")
def broadcast_catalog(rng):
    s, n, d = seeded(rng), rng.randint(3, 6), rng.randint(2, 4)
    setup = f"rng = np.random.default_rng({s})\nX = rng.normal(size=({n}, {d}))"
    items = [
        ("norm", "Scale each row of `X` to unit length: divide by `np.linalg.norm(X, axis=1, keepdims=True)`.",
         "X / np.linalg.norm(X, axis=1, keepdims=True)", {"require": ["norm", "keepdims"]}),
        ("normal", f"Draw another ({n}, {d}) batch with `rng.normal(loc=1.0, scale=0.5, size=...)` and return its column means "
                   f"minus `X`'s column means.", f"Y = rng.normal(loc=1.0, scale=0.5, size=({n}, {d}))\nresult = Y.mean(axis=0) - X.mean(axis=0)"),
        ("ones", f"Add a column of ones to `X` (a bias feature) by broadcasting: `np.hstack([X, np.ones((X.shape[0], 1))])`.",
         "np.hstack([X, np.ones((X.shape[0], 1))])", {"require": ["ones"]}),
    ]
    return catalog(rng, items, setup)


# ================================================================= 1.10 ufuncs

@template("1.10", "lite")
def ufunc_math_catalog(rng):
    n = rng.randint(3, 5)
    x = [round(v, 2) for v in floats(rng, n, 0.1, 0.9, 2)]
    y = floats(rng, n, -4, 4, 1)
    setup = f"x = {arr(x)}   # values in (0, 1)\ny = {arr(y)}"
    fns = ["sin", "tan", "sinh", "cosh", "arcsin", "arccos", "acos", "arctan", "exp2", "expm1", "log2", "cbrt", "fabs",
           "abs", "negative", "reciprocal", "sinc", "fix"]
    items = [(f, f"Apply `np.{f}` to {'`y`' if f in ('fabs', 'abs', 'negative', 'fix', 'cbrt') else '`x`'}.",
              f"np.{f}({'y' if f in ('fabs', 'abs', 'negative', 'fix', 'cbrt') else 'x'})") for f in fns]
    items += [
        ("deg2rad", "Convert 180 degrees to radians (`np.deg2rad`).", "np.deg2rad(180)"),
        ("rad2deg", "Convert π radians to degrees (`np.rad2deg(np.pi)`).", "np.rad2deg(np.pi)", {"require": ["rad2deg", "pi"]}),
        ("modf", "Split `y` into fractional and integer parts with `np.modf`.", "np.modf(y)"),
        ("float32", "Return `x` as float32 (`x.astype(np.float32)`).", "x.astype(np.float32)", {"check_dtype": True}),
        ("nan", "Return `np.isnan(np.array([1.0, np.nan]))`.", "np.isnan(np.array([1.0, np.nan]))"),
    ]
    return catalog(rng, items, setup, tol=1e-6)


@template("1.10", "easy")
def ufunc_arith_catalog(rng):
    n = rng.randint(3, 5)
    a, b = _ints(rng, n, -9, 9), _ints(rng, n, 1, 6)
    setup = f"a = {arr(a)}\nb = {arr(b)}   # positive"
    items = [
        ("add", "Return `np.add(a, b)`.", "np.add(a, b)"),
        ("subtract", "Return `np.subtract(a, b)`.", "np.subtract(a, b)"),
        ("multiply", "Return `np.multiply(a, b)`.", "np.multiply(a, b)"),
        ("power", "Return `np.power(b, 2)`.", "np.power(b, 2)"),
        ("pow", "Return `np.pow(b, 3)` (the array-API alias of power).", "np.pow(b, 3)"),
        ("mod", "Return `np.mod(a, b)` (sign of the divisor, like `%`).", "np.mod(a, b)"),
        ("floor_divide", "Return `np.floor_divide(a, b)`.", "np.floor_divide(a, b)"),
        ("divmod", "Return quotient and remainder at once with `np.divmod(a, b)`.", "np.divmod(a, b)"),
        ("fmax", "Return the elementwise maximum of `a` and `b` ignoring NaN (`np.fmax`).", "np.fmax(a, b)"),
        ("fmin", "Return the elementwise minimum of `a` and `b` ignoring NaN (`np.fmin`).", "np.fmin(a, b)"),
        ("copysign", "Give `b` the signs of `a` (`np.copysign`).", "np.copysign(b, a)"),
        ("heaviside", "Return the step function of `a` with value 0.5 at 0 (`np.heaviside(a, 0.5)`).", "np.heaviside(a, 0.5)"),
        ("lcm", "Return the least common multiples of `b` and 4 (`np.lcm`).", "np.lcm(b, 4)"),
        ("atan2", "Return the angle of each point (x = `b`, y = `a`) with `np.atan2(a, b)`.", "np.atan2(a, b)", {"tol": 1e-9}),
        ("min", "Return `np.min(a)` and `np.minimum.reduce(a)` (the same thing).", "[np.min(a), np.minimum.reduce(a)]"),
        ("empty", f"Compute `a + b` into a preallocated `out = np.empty({n}, dtype=a.dtype)` with `np.add(a, b, out=out)` and "
                  f"return `out`.", f"out = np.empty({n}, dtype=a.dtype)\nnp.add(a, b, out=out)\nresult = out", {"require": ["empty", "out"]}),
    ]
    return catalog(rng, items, setup)


@template("1.10", "medium")
def ufunc_tools_catalog(rng):
    v = sorted(floats(rng, 4, -1, 2, 1))
    r, c = rng.randint(2, 4), rng.randint(2, 4)
    setup = f"v = {arr(v)}\nM = np.arange({r * c}).reshape({r}, {c})"
    items = [
        ("vectorize", "Wrap a scalar Python function with `np.vectorize(lambda t: t * 2 if t > 0 else 0)` and apply it to `v`.",
         "np.vectorize(lambda t: t * 2 if t > 0 else 0)(v)"),
        ("frompyfunc", "Turn `lambda t: t * 2` into a ufunc with `np.frompyfunc(..., 1, 1)` and apply it to `np.arange(3)` "
                       "(the result has dtype object).", "np.frompyfunc(lambda t: t * 2, 1, 1)(np.arange(3))"),
        ("piecewise", "Return `np.piecewise(v, [v < 0, v > 1], [0, 1, lambda t: t])`: 0 below 0, 1 above 1, `v` in between.",
         "np.piecewise(v, [v < 0, v > 1], [0, 1, lambda t: t])"),
        ("apply_along_axis", "Return each row's max minus min with `np.apply_along_axis(lambda r: r.max() - r.min(), 1, M)`.",
         "np.apply_along_axis(lambda r: r.max() - r.min(), 1, M)"),
        ("reshape", f"Return `M.reshape(-1)` (a 1-D view).", "M.reshape(-1)"),
    ]
    return catalog(rng, items, setup)


# ================================================================= 1.11 reductions & statistics

@template("1.11", "easy")
def nan_stats_catalog(rng):
    vals = floats(rng, 6, 0, 9, 1)
    vals[rng.randrange(6)] = float("nan")
    setup = f"w = {arr(vals)}"
    fns = ["nansum", "nanmean", "nanmax", "nanprod", "nanstd", "nanvar", "nanmedian", "nanargmin", "nancumsum"]
    items = [(f, f"Return `np.{f}(w)`, which ignores the NaN.", f"np.{f}(w)") for f in fns]
    items += [("nanpercentile", "Return the 90th percentile of `w` ignoring NaN (`np.nanpercentile`).", "np.nanpercentile(w, 90)"),
              ("nan", "Return how many entries of `w` are NaN.", "np.isnan(w).sum()", {"require": ["isnan"]})]
    return catalog(rng, items, setup, tol=1e-9)


@template("1.11", "medium")
def stats_tools_catalog(rng):
    vals = floats(rng, 6, 0, 1, 2)
    setup = f"p = {arr(vals)}   # values in [0, 1]\nM = np.arange(1, 7).reshape(2, 3)"
    items = [
        ("prod", "Return the product of each row of `M` (`np.prod(..., axis=1)`).", "np.prod(M, axis=1)"),
        ("quantile", "Return the 25th and 75th percentiles of `p` with `np.quantile`.", "np.quantile(p, [0.25, 0.75])"),
        ("cumulative_sum", "Return the running sum of `M`'s first row **including** a leading 0 (`np.cumulative_sum(..., "
                           "include_initial=True)`).", "np.cumulative_sum(M[0], include_initial=True)", {"require": ["cumulative_sum", "include_initial"]}),
        ("digitize", "Bucket `p` with boundaries `[0.25, 0.75]` (`np.digitize`: 0 below, 1 between, 2 above).", "np.digitize(p, bins=[0.25, 0.75])"),
        ("histogram_bin_edges", "Return the 4 equal-width bin edges' array for `p` (`np.histogram_bin_edges(p, bins=4)`).",
         "np.histogram_bin_edges(p, bins=4)"),
        ("histogram2d", "Return the 2×2 joint histogram counts of `p[:3]` and `p[3:]` (`np.histogram2d(..., bins=2)[0]`).",
         "np.histogram2d(p[:3], p[3:], bins=2)[0]"),
        ("ravel_multi_index", "Return the flat index of coordinate (1, 2) in an array of `M`'s shape (`np.ravel_multi_index`).",
         "np.ravel_multi_index((1, 2), M.shape)"),
        ("arange", "Return the mean of `np.arange(1, 11)` weighted by itself (`np.average(..., weights=...)`).",
         "x = np.arange(1, 11)\nresult = np.average(x, weights=x)", {"require": ["arange", "average"]}),
        ("histogram", "Return the **density** histogram of `p` over 3 bins (`np.histogram(p, bins=3, density=True)[0]`): it "
                      "integrates to 1.", "np.histogram(p, bins=3, density=True)[0]", {"require": ["histogram", "density"]}),
        ("max", "Return `np.max(p[p > 2], initial=0.0)`: an empty selection would raise without `initial`.", "np.max(p[p > 2], initial=0.0)",
         {"require": ["max", "initial"]}),
        ("average", "Return the weighted mean and the sum of weights at once: `np.average(p, weights=np.arange(1, 7), returned=True)`.",
         "np.average(p, weights=np.arange(1, 7), returned=True)", {"require": ["average", "returned"]}),
        ("float64", "Sum `M` as float64 (`M.sum(dtype=np.float64)`) and return the result.", "M.sum(dtype=np.float64)", {"check_dtype": True}),
    ]
    return catalog(rng, items, setup, tol=1e-9)


# ================================================================= 1.12 logic

@template("1.12", "lite")
def logic_catalog(rng):
    n = rng.randint(3, 5)
    a, b = _ints(rng, n, -3, 3), _ints(rng, n, -3, 3)
    setup = f"a = {arr(a)}\nb = {arr(b)}"
    sp = {"setup": "s = np.array([1.0, np.inf, -np.inf, np.nan])"}
    items = [
        ("equal", "Return `np.equal(a, b)`.", "np.equal(a, b)"),
        ("greater", "Return `np.greater(a, b)`.", "np.greater(a, b)"),
        ("less_equal", "Return `np.less_equal(a, b)`.", "np.less_equal(a, b)"),
        ("logical_and", "Return where both `a > 0` and `b > 0` (`np.logical_and`).", "np.logical_and(a > 0, b > 0)"),
        ("logical_or", "Return where `a > 0` or `b > 0` (`np.logical_or`).", "np.logical_or(a > 0, b > 0)"),
        ("logical_not", "Return where `a > 0` is False (`np.logical_not`).", "np.logical_not(a > 0)"),
        ("array_equiv", "Is `a` equal to the broadcast of `a[:1]`? (`np.array_equiv` broadcasts; `array_equal` doesn't.)",
         "np.array_equiv(a, a[:1])"),
        ("isposinf", "Return where `s` is +inf (`np.isposinf`).", "np.isposinf(s)", sp),
        ("isneginf", "Return where `s` is -inf (`np.isneginf`).", "np.isneginf(s)", sp),
        ("isreal", "Return `np.isreal(np.array([1, 2 + 0j, 1j]))`.", "np.isreal(np.array([1, 2 + 0j, 1j]))"),
        ("iscomplexobj", "Return `[np.iscomplexobj(a), np.iscomplexobj(a + 0j)]`.", "[np.iscomplexobj(a), np.iscomplexobj(a + 0j)]"),
        ("isscalar", "Return `[np.isscalar(3.0), np.isscalar(a)]`.", "[np.isscalar(3.0), np.isscalar(a)]"),
    ]
    return catalog(rng, items, setup)


# ================================================================= 1.13 sorting & sets

@template("1.13", "easy")
def sets_catalog(rng):
    v = _ints(rng, 7, 0, 5)
    k = rng.randint(1, 5)
    setup = f"v = {arr(v)}"
    items = [
        ("copy", "Sort a copy in place: `s = v.copy(); s.sort()`. Return `[v, s]` (the original keeps its order).",
         "s = v.copy()\ns.sort()\nresult = [v, s]"),
        ("partition", f"`np.partition(v, {k})` puts the element that belongs at index {k} there, smaller ones before and larger "
                      f"after (O(n)). Return its element at index {k}.", f"np.partition(v, {k})[{k}]"),
        ("unique_values", "Return the distinct values of `v` (`np.unique_values`).", "np.unique_values(v)"),
        ("sort", "Sort `v` with a **stable** algorithm (`np.sort(v, kind='stable')`): equal keys keep their order.", "np.sort(v, kind='stable')",
         {"require": ["sort", "kind"]}),
        ("unique", "Return `np.unique(v, return_inverse=True)`: the distinct values and where each element maps.",
         "np.unique(v, return_inverse=True)", {"require": ["unique", "return_inverse"]}),
        ("unique_counts", "Return the distinct values and their counts (`np.unique_counts`: a named tuple).",
         "r = np.unique_counts(v)\nresult = [r.values, r.counts]"),
        ("unique_inverse", "Return the distinct values and, for each element, the index of its value (`np.unique_inverse`).",
         "r = np.unique_inverse(v)\nresult = [r.values, r.inverse_indices]"),
        ("unique_all", "Return `[values, indices, inverse_indices, counts]` from `np.unique_all(v)`.",
         "r = np.unique_all(v)\nresult = [r.values, r.indices, r.inverse_indices, r.counts]"),
    ]
    return catalog(rng, items, setup)


# ================================================================= 1.14 linear algebra

@template("1.14", "medium")
def linalg_catalog(rng):
    A = [[float(v) for v in _ints(rng, 3, -3, 4)] for _ in range(3)]
    for i in range(3):
        A[i][i] += 6.0
    v = [float(x) for x in _ints(rng, 3, -2, 3)]
    setup = f"A = {mat(A)}\nv = {arr(v)}"
    items = [
        ("dot", "Return `np.dot(v, v)`.", "np.dot(v, v)"),
        ("vdot", "Return `np.vdot(v, v)` (flattens and conjugates its first argument).", "np.vdot(v, v)"),
        ("vecdot", "Return `np.vecdot(v, v)` (explicit batched vector dot product).", "np.vecdot(v, v)"),
        ("matvec", "Return the matrix-vector product with `np.matvec(A, v)`.", "np.matvec(A, v)"),
        ("vecmat", "Return the vector-matrix product with `np.vecmat(v, A)`.", "np.vecmat(v, A)"),
        ("matmul", "Return `np.matmul(A, A)`.", "np.matmul(A, A)"),
        ("tensordot", "Contract `A`'s axis 1 with `A`'s axis 0 (`np.tensordot(A, A, axes=([1], [0]))`).", "np.tensordot(A, A, axes=([1], [0]))"),
        ("diag", "Return the diagonal of `A` (`np.diag`).", "np.diag(A)"),
        ("einsum", "Return `A @ A @ A` with `np.einsum('ij,jk,kl->il', A, A, A, optimize=True)` (optimize picks the contraction order).",
         "np.einsum('ij,jk,kl->il', A, A, A, optimize=True)", {"require": ["einsum", "optimize"]}),
        ("eigvals", "Return the real parts of `A`'s eigenvalues, sorted (`np.linalg.eigvals`).", "np.sort(np.linalg.eigvals(A).real)"),
        ("eig", "Return the real parts of `A`'s eigenvalues from `np.linalg.eig`, sorted.", "np.sort(np.linalg.eig(A).eigenvalues.real)"),
        ("slogdet", "Return the sign and log|det| of `A` (`np.linalg.slogdet`).", "s, ld = np.linalg.slogdet(A)\nresult = [s, ld]"),
        ("lstsq", "Return the least-squares solution of `A x ≈ v` (`np.linalg.lstsq(A, v, rcond=None)[0]`).",
         "np.linalg.lstsq(A, v, rcond=None)[0]", {"require": ["lstsq", "rcond"]}),
        ("qr", "Factor `A = QR` with `np.linalg.qr` and return whether `Q @ R` reproduces `A` (`np.allclose`).",
         "Q, R = np.linalg.qr(A)\nresult = np.allclose(Q @ R, A)", {"require": ["qr", "allclose"]}),
        ("matrix_norm", "Return the Frobenius norm of `A` with `np.linalg.matrix_norm`.", "np.linalg.matrix_norm(A)"),
        ("vector_norm", "Return the L1 norm of `v` with `np.linalg.vector_norm(v, ord=1)`.", "np.linalg.vector_norm(v, ord=1)"),
        ("ones", "Return the row sums of `A` as a matrix product with `np.ones(3)`.", "A @ np.ones(3)", {"require": ["ones"]}),
        ("reshape", "Return the outer product of `v` with itself via broadcasting: `v.reshape(-1, 1) * v`.", "v.reshape(-1, 1) * v"),
    ]
    return catalog(rng, items, setup, tol=1e-9)


# ================================================================= 1.15 FFT, polynomials & signals

@template("1.15", "medium")
def signal_poly_catalog(rng):
    n = rng.choice([4, 8])
    sig = floats(rng, n, -2, 2, 1)
    coeffs = _ints(rng, 3, 1, 5)
    setup = f"sig = {arr(sig)}\ncoeffs = {lit(coeffs)}   # highest power first"
    items = [
        ("linspace", f"Sample times for a 1-second signal at {n} Hz: `np.linspace(0, 1, {n}, endpoint=False)`.",
         f"np.linspace(0, 1, {n}, endpoint=False)", {"require": ["linspace", "endpoint"]}),
        ("fftfreq", f"Return the sample frequencies of a length-{n} FFT (`np.fft.fftfreq({n})`).", f"np.fft.fftfreq({n})"),
        ("ifft", "Return the real part of `np.fft.ifft(np.fft.fft(sig))`: the round trip gives `sig` back.", "np.fft.ifft(np.fft.fft(sig)).real"),
        ("fft2", "Return the magnitudes of the 2-D FFT of `sig.reshape(2, -1)` (`np.abs(np.fft.fft2(...))`).", "np.abs(np.fft.fft2(sig.reshape(2, -1)))"),
        ("fftn", "Return the magnitudes of `np.fft.fftn(sig.reshape(2, 2, -1))`.", "np.abs(np.fft.fftn(sig.reshape(2, 2, -1)))"),
        ("fftshift", "Center the zero frequency: return `np.fft.fftshift(np.fft.fftfreq(sig.size))`.", "np.fft.fftshift(np.fft.fftfreq(sig.size))",
         {"require": ["fftshift", "size"]}),
        ("poly1d", "Evaluate the polynomial with `coeffs` at 4 (`np.poly1d(coeffs)(4)`).", "np.poly1d(coeffs)(4)"),
        ("deriv", "Return the coefficients of the derivative of `np.poly1d(coeffs)` (`.deriv().coeffs`).", "np.poly1d(coeffs).deriv().coeffs"),
        ("integ", "Return the coefficients of the antiderivative of `np.poly1d(coeffs)` (`.integ().coeffs`).", "np.poly1d(coeffs).integ().coeffs"),
        ("fit", "Fit a degree-2 `Polynomial` to the points x = [0, 1, 2, 3], y = [1, 3, 7, 13] and return the coefficients in "
                "the standard basis, **lowest power first** (`Polynomial.fit(...).convert().coef`).",
         "from numpy.polynomial import Polynomial\nresult = Polynomial.fit([0, 1, 2, 3], [1, 3, 7, 13], deg=2).convert().coef",
         {"require": ["fit", "convert"], "tol": 1e-8}),
        ("polynomial", "Evaluate `numpy.polynomial.Polynomial([1, 2, 3])` (1 + 2x + 3x²) at 2.",
         "from numpy.polynomial import Polynomial\nresult = Polynomial([1, 2, 3])(2)", {"require": ["Polynomial"]}),
        ("correlate", "Return `np.correlate(sig[:3], [0, 1, 0.5], mode='valid')`.", "np.correlate(sig[:3], [0, 1, 0.5], mode='valid')"),
        ("ediff1d", "Return the consecutive differences of `sig` with `np.ediff1d`.", "np.ediff1d(sig)"),
        ("unwrap", "Unwrap the phases `np.array([0.0, 3.0, 6.5])` (`np.unwrap`).", "np.unwrap(np.array([0.0, 3.0, 6.5]))"),
        ("trapezoid", "Integrate `sig` with the trapezoid rule, spacing 0.5 (`np.trapezoid(sig, dx=0.5)`; `np.trapz` before NumPy 2).",
         "np.trapezoid(sig, dx=0.5)"),
        ("eye", "Return the 2-D FFT of `np.eye(2)` (complex).", "np.fft.fft2(np.eye(2))", {"require": ["fft2", "eye"]}),
    ]
    return catalog(rng, items, setup, tol=1e-9)


# ================================================================= 1.16 strings & bits

@template("1.16", "easy")
def strings_bits_catalog(rng):
    a, b = _ints(rng, 3, 1, 15), _ints(rng, 3, 1, 15)
    k = rng.randint(1, 3)
    setup = f"a = {arr(a)}\nb = {arr(b)}"
    items = [
        ("bitwise_xor", "Return `np.bitwise_xor(a, b)`.", "np.bitwise_xor(a, b)"),
        ("left_shift", f"Shift `a` left by {k} bits (`np.left_shift`).", f"np.left_shift(a, {k})"),
        ("rec", "Record arrays allow attribute access: for `people = np.array([('Ana', 31), ('Ben', 25)], dtype=[('name', 'U10'), "
                "('age', 'i4')])`, return `np.rec.array(people).age.mean()`.",
         "people = np.array([('Ana', 31), ('Ben', 25)], dtype=[('name', 'U10'), ('age', 'i4')])\nresult = np.rec.array(people).age.mean()",
         {"require": ["rec", "age"]}),
        ("split", "Split each string of `np.array(['a,b', 'c,d,e'])` on commas with `np.char.split` and return the lists.",
         "np.char.split(np.array(['a,b', 'c,d,e']), ',').tolist()", {"require": ["char", "split"]}),
    ]
    return catalog(rng, items, setup)


# ================================================================= 1.18 testing

@template("1.18", "easy")
def testing_catalog(rng):
    n = rng.randint(3, 5)
    a = floats(rng, n, -3, 3, 1)
    setup = f"a = {arr(a)}\nM = np.arange({n * n}.0).reshape({n}, {n})"
    items = [
        ("allclose", "Return `np.allclose(a, a + 1e-9)`.", "np.allclose(a, a + 1e-9)"),
        ("allclose", "Return `np.allclose(a, a + 1e-4, atol=1e-3)` (a looser absolute tolerance).", "np.allclose(a, a + 1e-4, atol=1e-3)",
         {"require": ["allclose", "atol"]}),
        ("bincount", "Count labels `np.array([0, 2, 2])` into 4 classes, zeros included (`np.bincount(..., minlength=4)`).",
         "np.bincount(np.array([0, 2, 2]), minlength=4)", {"require": ["bincount", "minlength"]}),
        ("assert_equal", "Call `np.testing.assert_equal(a, a.copy())` inside try/except AssertionError and return True if it "
                         "passed.", "try:\n    np.testing.assert_equal(a, a.copy())\n    result = True\nexcept AssertionError:\n    result = False"),
        ("norm", "Return the L2 norm of `a` (`np.linalg.norm`).", "np.linalg.norm(a)"),
        ("ones_like", "Return `np.ones_like(a)`.", "np.ones_like(a)"),
        ("trace", "Return the trace of `M` (`np.trace`).", "np.trace(M)"),
    ]
    return catalog(rng, items, setup, tol=1e-9)


# ================================================================= 1.17 file I/O

@template("1.17", "easy")
def file_io_catalog(rng):
    vals = floats(rng, 4, -3, 3, 1)
    setup = f"a = {arr(vals)}"
    items = [
        ("allow_pickle", "Save an **object** array `np.array([{'a': 1}], dtype=object)` to `o.npy` and load it back; both need "
                         "`allow_pickle=True` (only for files you trust). Return the loaded element.",
         "np.save('o.npy', np.array([{'a': 1}], dtype=object), allow_pickle=True)\nresult = np.load('o.npy', allow_pickle=True)[0]",
         {"require": ["allow_pickle"]}),
        ("mmap_mode", "Save `a` to `a.npy`, then memory-map it read-only with `np.load('a.npy', mmap_mode='r')` and return a copy "
                      "of its first two values.", "np.save('a.npy', a)\nm = np.load('a.npy', mmap_mode='r')\nresult = np.array(m[:2])"),
    ]
    return catalog(rng, items, setup)
