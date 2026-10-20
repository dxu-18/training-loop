"""3.01 Tensors."""

from ... import themes as T
from ._common import P, big_ints, challenge, classes, floats, frows_tt, ftt, lit, pick_theme, rows_tt, seed_line, seeded, \
    shp, template, tt

M = "3.01"


# ================================================================= lite

@template(M, "lite")
def create_tensors(rng):
    th = T.theme(rng)
    m = T.measure(rng, th)
    kind = rng.choice(["list", "zeros", "full", "arange", "linspace", "eye"])
    if kind == "list":
        n = rng.randint(3, 6)
        vals = T.values(rng, m, n)
        if m.decimals or rng.random() < 0.5:
            return P(f"`{m.var}` lists the {m.label} of {n} {th.plural}. Turn it into a **float32** tensor with "
                     f"`torch.tensor(..., dtype=...)`.", f"{m.var} = {lit(vals)}",
                     f"torch.tensor({m.var}, dtype=torch.float32)", require=["tensor"], check_dtype=True)
        return P(f"`{m.var}` lists the {m.label} of {n} {th.plural} (whole numbers). Turn it into a tensor with "
                 f"`torch.tensor` and let PyTorch infer the dtype (int64).", f"{m.var} = {lit(vals)}",
                 f"torch.tensor({m.var})", require=["tensor"], check_dtype=True)
    if kind == "zeros":
        r, c = rng.randint(2, 4), rng.randint(2, 5)
        return P(f"Create a {r}×{c} tensor of zeros (float32) with `torch.zeros`.", "", f"torch.zeros({r}, {c})",
                 require=["zeros"], check_dtype=True)
    if kind == "full":
        r, c = rng.randint(2, 4), rng.randint(2, 5)
        v = rng.choice([0.5, 1.5, 2.5, -1.0, 7.0, 0.25])
        return P(f"Create a {r}×{c} float tensor where every entry is `{v}`, using `torch.full`.", "",
                 f"torch.full(({r}, {c}), {v})", require=["full"], check_dtype=True)
    if kind == "arange":
        a, step = rng.randint(0, 5), rng.randint(2, 4)
        b = a + step * rng.randint(3, 6) + rng.randint(0, step - 1)
        return P(f"Create the integer tensor `{a}, {a + step}, {a + 2 * step}, ...` that stops **before** `{b}` "
                 f"(`torch.arange` with a step).", "", f"torch.arange({a}, {b}, {step})", require=["arange"],
                 check_dtype=True)
    if kind == "linspace":
        a, b, n = rng.randint(0, 3), rng.randint(4, 10), rng.choice([3, 5, 9])
        return P(f"Create {n} evenly spaced floats from `{a}` to `{b}` **inclusive** with `torch.linspace`.", "",
                 f"torch.linspace({a}, {b}, steps={n})", require=["linspace"], check_dtype=True)
    n = rng.randint(2, 5)
    m2 = n + rng.randint(1, 2)
    if rng.random() < 0.5:
        return P(f"Create the {n}×{n} identity matrix with `torch.eye`.", "", f"torch.eye({n})", require=["eye"],
                 check_dtype=True)
    return P(f"Create a {n}×{m2} matrix with ones on the main diagonal and zeros elsewhere (`torch.eye` with two sizes).",
             "", f"torch.eye({n}, {m2})", require=["eye"], check_dtype=True)


@template(M, "lite")
def tensor_attributes(rng):
    shape = tuple(rng.randint(2, 6) for _ in range(rng.randint(2, 4)))
    setup = f"{seed_line(rng)}\nx = torch.randn{shp(shape)}"
    task = rng.choice(["numel", "shape", "size", "ndim", "bytes"])
    if task == "numel":
        return P("How many elements does `x` hold? Use `numel()`.", setup, "x.numel()", require=["numel"])
    if task == "shape":
        return P("Return the shape of `x` as a plain Python tuple.", setup, "tuple(x.shape)", require=["shape|size"])
    if task == "size":
        d = rng.randrange(len(shape))
        return P(f"Return the size of dimension {d} of `x` (`x.size(...)` or `x.shape[...]`).", setup, f"x.size({d})")
    if task == "ndim":
        return P("How many dimensions does `x` have?", setup, "x.ndim", require=["ndim|dim"])
    dt, nbytes = rng.choice([("float16", 2), ("float64", 8), ("int32", 4), ("float32", 4)])
    setup = f"x = torch.zeros({', '.join(map(str, shape))}, dtype=torch.{dt})"
    return P(f"`x` is a {dt} tensor. How many **bytes** do its elements take? Multiply `element_size()` (bytes per "
             f"element) by `numel()`.", setup, "x.element_size() * x.numel()", require=["element_size|nbytes"])


@template(M, "lite")
def convert_tensors(rng):
    n = rng.randint(3, 5)
    vals = floats(rng, n, -5, 5, 1)
    task = rng.choice(["tolist", "item", "numpy", "long", "half", "from_numpy"])
    if task == "tolist":
        return P("Convert `t` to a plain Python list with `tolist()`.", f"t = {ftt(vals)}", "t.tolist()", require=["tolist"])
    if task == "item":
        i = rng.randrange(n)
        return P(f"Return element {i} of `t` as a plain Python float (`.item()`).", f"t = {ftt(vals)}", f"t[{i}].item()",
                 require=["item"])
    if task == "numpy":
        return P("`t` requires grad. Convert it to a NumPy array the safe way: `detach()` first, then `.numpy()`.",
                 f"t = torch.tensor({lit(vals)}, requires_grad=True)", "t.detach().numpy()", require=["detach"])
    if task == "long":
        return P("Convert `t` to an int64 tensor (`.long()`); the fractional part is truncated toward zero.",
                 f"t = {ftt(vals)}", "t.long()", require=["long|to|int64"], check_dtype=True)
    if task == "half":
        return P("Convert `t` to float16 (`.half()` or `.to(torch.float16)`).", f"t = {ftt(vals)}", "t.half()",
                 require=["half|to|float16"], check_dtype=True, tol=1e-2)
    k = rng.randint(4, 7)
    return P(f"`arr` is a float64 NumPy array. Wrap it as a tensor that **shares memory** with it "
             f"(`torch.from_numpy`); keep the dtype.", f"arr = np.arange({k}.0)", "torch.from_numpy(arr)",
             require=["from_numpy|as_tensor"], check_dtype=True)


# ================================================================= easy

@template(M, "easy")
def index_slices(rng):
    th, names = pick_theme(rng, rng.randint(3, 5))
    ax = T.axis(rng, min_labels=4, max_labels=8)
    m = T.measure(rng, th)
    c = rng.randint(4, min(6, len(ax.labels)))
    rows = [[T.value(rng, m) for _ in range(c)] for _ in names]
    rows = [[int(round(v)) for v in r] for r in rows]
    setup = f"{th.var} = {lit(names)}\ncols = {lit(list(ax.labels[:c]))}\nx = {rows_tt(rows)}"
    desc = f"`x` holds the {m.label} (rounded) of the {th.plural} in `{th.var}` (rows) for each of the {ax.plural} in `cols`"
    task = rng.choice(["row", "col", "last", "every_other", "block", "mask", "fancy"])
    if task == "row":
        i = rng.randrange(len(names))
        return P(f"{desc}. Return the row for `{names[i]}`.", setup, f"x[{i}]")
    if task == "col":
        j = rng.randrange(c)
        return P(f"{desc}. Return the column for `{ax.labels[j]}` (all rows).", setup, f"x[:, {j}]")
    if task == "last":
        return P(f"{desc}. Return the **last** column using a negative index.", setup, "x[:, -1]")
    if task == "every_other":
        return P(f"{desc}. Return every other column, starting with the first (a slice with a step).", setup, "x[:, ::2]")
    if task == "block":
        r0 = rng.randrange(len(names) - 1)
        c0 = rng.randrange(c - 2)
        return P(f"{desc}. Return the block of rows {r0}..{r0 + 1} and columns {c0}..{c0 + 2} (inclusive) with one slice "
                 f"per dimension.", setup, f"x[{r0}:{r0 + 2}, {c0}:{c0 + 3}]")
    if task == "mask":
        flat = sorted({v for r in rows for v in r})
        t = flat[len(flat) // 2]
        return P(f"{desc}. Return every value **greater than {t}** as a 1-D tensor (a boolean mask).", setup,
                 f"x[x > {t}]")
    i1, i2 = rng.sample(range(len(names)), 2)
    j1, j2 = rng.randrange(c), rng.randrange(c)
    return P(f"{desc}. With one fancy-indexing expression, return the two values for (`{names[i1]}`, `{ax.labels[j1]}`) "
             f"and (`{names[i2]}`, `{ax.labels[j2]}`).", setup, f"x[[{i1}, {i2}], [{j1}, {j2}]]")


@template(M, "easy")
def reduce_dims(rng):
    th, names = pick_theme(rng, rng.randint(3, 5))
    ax = T.axis(rng, min_labels=3, max_labels=8)
    c = rng.randint(3, min(5, len(ax.labels)))
    rows = [floats(rng, c, 0, 10, 1) for _ in names]
    setup = f"{th.var} = {lit(names)}\nx = {frows_tt(rows)}"
    desc = f"`x` has one row per {th.noun} in `{th.var}` and one column per {ax.noun} ({c} {ax.plural})"
    task = rng.choice(["mean_rows", "max_cols", "argmax", "std", "keepdim", "pop_std"])
    if task == "mean_rows":
        return P(f"{desc}. Return each {th.noun}'s mean over the {ax.plural} (one value per row).", setup, "x.mean(dim=1)",
                 require=["mean"])
    if task == "max_cols":
        return P(f"{desc}. For every {ax.noun}, return the largest value over all {th.plural}. `max(dim=...)` returns "
                 f"`(values, indices)`: return the values.", setup, "x.max(dim=0).values", require=["max|amax"])
    if task == "argmax":
        return P(f"{desc}. For every {th.noun}, return the column index of its largest value (`argmax`).", setup,
                 "x.argmax(dim=1)", require=["argmax"])
    if task == "std":
        return P(f"{desc}. Return the standard deviation of each column. Note `std` is **unbiased** (divides by n − 1) "
                 f"by default.", setup, "x.std(dim=0)", require=["std"])
    if task == "keepdim":
        return P(f"{desc}. Return the row sums as a column of shape ({len(names)}, 1) with `keepdim=True`, so that it "
                 f"broadcasts against `x`.", setup, "x.sum(dim=1, keepdim=True)", require=["keepdim"])
    return P(f"{desc}. Return the **population** standard deviation of each row (divide by n, `correction=0`).", setup,
             "x.std(dim=1, correction=0)", require=["std|var"])


@template(M, "easy")
def join_split(rng):
    r, c = rng.randint(2, 3), rng.randint(2, 4)
    setup = f"a = torch.arange({r * c}).reshape({r}, {c})\nb = torch.arange({100}, {100 + r * c}).reshape({r}, {c})"
    task = rng.choice(["cat0", "cat1", "stack", "chunk", "split"])
    if task == "cat0":
        return P(f"Join `a` and `b` along the **existing** dimension 0 (rows), giving shape ({2 * r}, {c}).", setup,
                 "torch.cat([a, b], dim=0)", require=["cat|concat|vstack"])
    if task == "cat1":
        return P(f"Join `a` and `b` side by side along dimension 1, giving shape ({r}, {2 * c}).", setup,
                 "torch.cat([a, b], dim=1)", require=["cat|concat|hstack"])
    if task == "stack":
        d = rng.choice([0, 2])
        shape = (2, r, c) if d == 0 else (r, c, 2)
        return P(f"Stack `a` and `b` along a **new** dimension {d}, giving shape {shape}.", setup,
                 f"torch.stack([a, b], dim={d})", require=["stack"])
    n = rng.randint(7, 12)
    setup = f"x = torch.arange({n})"
    if task == "chunk":
        k = rng.choice([2, 3, 4])
        return P(f"Split `x` into {k} chunks with `torch.chunk` (the last one may be smaller). Return the tuple of "
                 f"tensors.", setup, f"torch.chunk(x, {k})", require=["chunk"])
    s = rng.randint(2, n - 2)
    return P(f"Split `x` into two pieces of sizes {s} and {n - s} with `torch.split` and a list of sizes.", setup,
             f"torch.split(x, [{s}, {n - s}])", require=["split"])


@template(M, "easy")
def division_kinds(rng):
    vals = [rng.randint(-20, 20) for _ in range(rng.randint(4, 6))]
    vals[0] = -abs(vals[0]) - 1
    k = rng.randint(2, 6)
    setup = f"a = {tt(vals)}"
    task = rng.choice(["true", "floor", "trunc", "mod"])
    if task == "true":
        return P(f"`a` is an int64 tensor. Divide it by {k} with true division (the result is a float tensor).", setup,
                 f"a / {k}", check_dtype=True)
    if task == "floor":
        return P(f"Divide `a` by {k} rounding **down** (floor division; note what happens to negative numbers).", setup,
                 f"a // {k}", check_dtype=True)
    if task == "trunc":
        return P(f"Divide `a` by {k} rounding **toward zero**: `torch.div(..., rounding_mode='trunc')`.", setup,
                 f"torch.div(a, {k}, rounding_mode='trunc')", require=["div"], check_dtype=True)
    return P(f"Return `a` modulo {k} (`%`). In PyTorch, as in Python, the result takes the sign of the divisor.", setup,
             f"a % {k}", check_dtype=True)


# ================================================================= medium

@template(M, "medium")
def gather_targets(rng):
    name, labels = classes(rng)
    n, C = rng.randint(3, 5), len(labels)
    logits = [floats(rng, C, -3, 3, 1) for _ in range(n)]
    targets = [rng.randrange(C) for _ in range(n)]
    setup = f"classes = {lit(labels)}\nlogits = {frows_tt(logits)}\ntargets = {tt(targets)}"
    desc = f"`logits` holds raw {name} scores for {n} examples over the classes in `classes`; `targets` holds each example's true class id"
    task = rng.choice(["gather", "fancy", "logprob"])
    if task == "gather":
        return P(f"{desc}. Pick out each example's logit for its target class with `gather` (index shape (N, 1), then "
                 f"squeeze).", setup, "logits.gather(1, targets.unsqueeze(1)).squeeze(1)", require=["gather"])
    if task == "fancy":
        return P(f"{desc}. Pick out each example's logit for its target class with fancy indexing: "
                 f"`logits[torch.arange(N), targets]`.", setup, f"logits[torch.arange({n}), targets]", require=["arange"])
    return P(f"{desc}. Return each example's **log-probability** of its target class: `F.log_softmax` over the classes, "
             f"then `gather`.", setup, "F.log_softmax(logits, dim=1).gather(1, targets.unsqueeze(1)).squeeze(1)",
             require=["log_softmax"], tol=1e-5)


@template(M, "medium")
def one_hot_ways(rng):
    name, labels = classes(rng)
    C = len(labels)
    n = rng.randint(4, 7)
    t = [rng.randrange(C) for _ in range(n)]
    setup = f"classes = {lit(labels)}\ntargets = {tt(t)}"
    desc = f"`targets` holds {n} {name} class ids (indices into `classes`)"
    way = rng.choice(["one_hot", "scatter", "eye", "counts"])
    if way == "one_hot":
        return P(f"{desc}. Build the ({n}, {C}) one-hot matrix as **float32** with `F.one_hot` (it returns int64).",
                 setup, f"F.one_hot(targets, num_classes={C}).float()", require=["one_hot"], check_dtype=True)
    if way == "scatter":
        return P(f"{desc}. Build the ({n}, {C}) float one-hot matrix by scattering 1.0 into zeros: "
                 f"`torch.zeros(N, C).scatter_(1, targets.unsqueeze(1), 1.0)`.", setup,
                 f"torch.zeros({n}, {C}).scatter_(1, targets.unsqueeze(1), 1.0)", require=["scatter_"], check_dtype=True)
    if way == "eye":
        return P(f"{desc}. Build the ({n}, {C}) float one-hot matrix by indexing the rows of an identity matrix "
                 f"(`torch.eye(C)[targets]`).", setup, f"torch.eye({C})[targets]", require=["eye"], check_dtype=True)
    return P(f"{desc}. Count how many examples fall in each class (length {C}, zeros included) with `torch.bincount` "
             f"(`minlength`).", setup, f"torch.bincount(targets, minlength={C})", require=["bincount|scatter_add_"])


@template(M, "medium")
def masked_ops(rng):
    n = rng.randint(3, 4)
    rows = [floats(rng, n, -4, 4, 1) for _ in range(n)]
    flat = sorted(v for r in rows for v in r)
    t = flat[len(flat) // 2]
    setup = f"x = {frows_tt(rows)}"
    task = rng.choice(["where", "clamp", "count", "causal"])
    if task == "where":
        return P(f"Replace every entry of `x` below {t} with 0 and keep the rest: `torch.where(cond, a, b)` or "
                 f"`masked_fill`.", setup, f"torch.where(x < {t}, torch.zeros_like(x), x)", require=["where|masked_fill"])
    if task == "clamp":
        lo, hi = flat[1], flat[-2]
        return P(f"Clip `x` into the range [{lo}, {hi}] with `clamp`.", setup, f"x.clamp({lo}, {hi})",
                 require=["clamp|clip"])
    if task == "count":
        return P(f"For each row of `x`, count the entries greater than {t} (compare, then `sum(dim=...)`).", setup,
                 f"(x > {t}).sum(dim=1)")
    return P("`x` is a square score matrix. Set every entry **above** the main diagonal to `-inf` (a causal mask): "
             "build the mask with `torch.triu(..., diagonal=1)` and use `masked_fill`.", setup,
             f"x.masked_fill(torch.triu(torch.ones({n}, {n}, dtype=torch.bool), diagonal=1), float('-inf'))",
             require=["masked_fill"])


@template(M, "medium")
def topk_sort(rng):
    th, names = pick_theme(rng, rng.randint(4, 6))
    m = T.measure(rng, th)
    task = rng.choice(["topk_vals", "topk_names", "sort_desc", "kth"])
    if task == "topk_names":
        vals = T.values(rng, m, len(names), distinct=True)
        k = rng.randint(2, 3)
        return P(f"`{m.var}` holds the {m.label} of the {th.plural} in `{th.var}`. Return the names of the {k} "
                 f"{th.plural} with the **highest** values, highest first (`torch.topk`, then index the list).",
                 f"{th.var} = {lit(names)}\n{m.var} = {ftt(vals)}",
                 f"[{th.var}[i] for i in torch.topk({m.var}, {k}).indices.tolist()]", require=["topk|argsort|sort"])
    c = rng.randint(4, 6)
    pool = rng.sample(range(10, 99), len(names) * c)
    rows = [[pool[i * c + j] / 10 for j in range(c)] for i in range(len(names))]
    setup = f"x = {frows_tt(rows)}"
    if task == "topk_vals":
        k = rng.randint(2, 3)
        return P(f"For each row of `x`, return its {k} largest values in descending order (`topk` along dim 1).", setup,
                 f"x.topk({k}, dim=1).values", require=["topk|sort"])
    if task == "sort_desc":
        return P("Sort each row of `x` in **descending** order and return the column indices of the sorted order "
                 "(`sort` returns values and indices).", setup, "x.sort(dim=1, descending=True).indices",
                 require=["sort|argsort"])
    k = rng.randint(2, 3)
    return P(f"For each row of `x`, return its {k}-th **smallest** value (`kthvalue`).", setup,
             f"x.kthvalue({k}, dim=1).values", require=["kthvalue|sort|topk"])


# ================================================================= hard

@template(M, "hard")
def class_centroids(rng):
    name, labels = classes(rng, 2, 4)
    C = len(labels)
    n, D = rng.randint(6, 9), rng.randint(2, 3)
    y = list(range(C)) + [rng.randrange(C) for _ in range(n - C)]
    rng.shuffle(y)
    X = [floats(rng, D, -5, 5, 1) for _ in range(n)]
    setup = f"classes = {lit(labels)}\nX = {frows_tt(X)}\ny = {tt(y)}"
    desc = f"`X` holds {D} features for {n} examples and `y` their {name} class ids (every class appears)"
    if rng.random() < 0.6:
        return P(f"{desc}. Without Python loops, return the ({C}, {D}) matrix of **class centroids** (the mean feature "
                 f"vector of each class): sum the rows per class with `index_add_` and divide by the class counts from "
                 f"`bincount`.", setup,
                 f"sums = torch.zeros({C}, {D}).index_add_(0, y, X)\nresult = sums / torch.bincount(y, minlength={C}).unsqueeze(1)",
                 no_loops=True, require=["index_add_|scatter_add_|index_add"])
    j = rng.randrange(D)
    return P(f"{desc}. Without Python loops, return each class's **largest** value of feature {j}: "
             f"`torch.full((C,), -inf).scatter_reduce(0, y, X[:, {j}], reduce='amax')`.", setup,
             f"torch.full(({C},), float('-inf')).scatter_reduce(0, y, X[:, {j}], reduce='amax')", no_loops=True,
             require=["scatter_reduce"])


@template(M, "hard")
def sequence_logprob(rng):
    B, Tn, V = rng.randint(2, 3), rng.randint(4, 6), rng.randint(5, 8)
    lens = [rng.randint(2, Tn) for _ in range(B)]
    lens[0] = Tn
    mask = [[1.0] * k + [0.0] * (Tn - k) for k in lens]
    setup = (f"{seed_line(rng)}\nlogits = torch.randn({B}, {Tn}, {V})\ntokens = torch.randint(0, {V}, ({B}, {Tn}))\n"
             f"mask = {rows_tt(mask)}")
    desc = (f"`logits` ({B}, {Tn}, {V}) are a language model's scores over a vocabulary of {V} at each position, "
            f"`tokens` ({B}, {Tn}) the observed tokens, and `mask` is 1.0 on real tokens and 0.0 on padding")
    pick = "F.log_softmax(logits, dim=-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)"
    if rng.random() < 0.5:
        return P(f"{desc}. Without loops, return each sequence's **total log-probability** of its tokens, counting only "
                 f"real tokens: log-softmax, gather the token log-probs, multiply by the mask, sum over time.", setup,
                 f"result = ({pick} * mask).sum(-1)", no_loops=True, tol=1e-5)
    return P(f"{desc}. Without loops, return each sequence's **mean** log-probability per real token (masked sum divided "
             f"by the number of real tokens).", setup, f"result = ({pick} * mask).sum(-1) / mask.sum(-1)", no_loops=True,
             tol=1e-5)


@template(M, "hard")
def confusion_bincount(rng):
    name, labels = classes(rng, 2, 4)
    C = len(labels)
    n = rng.randint(8, 12)
    t = list(range(C)) + [rng.randrange(C) for _ in range(n - C)]
    rng.shuffle(t)
    p = [v if rng.random() < 0.6 else rng.randrange(C) for v in t]
    setup = f"classes = {lit(labels)}\ntargets = {tt(t)}\npreds = {tt(p)}"
    desc = f"`targets` and `preds` hold true and predicted {name} ids for {n} examples"
    cm = f"torch.bincount(targets * {C} + preds, minlength={C * C}).reshape({C}, {C})"
    if rng.random() < 0.5:
        return P(f"{desc}. Without loops, build the {C}×{C} **confusion matrix** (row = true class, column = predicted): "
                 f"encode each pair as `target * C + pred`, `bincount` with `minlength=C*C`, and reshape.", setup,
                 f"result = {cm}", no_loops=True, require=["bincount"])
    return P(f"{desc}. Without loops, return each class's **recall** (correct predictions / true examples of the class) "
             f"as a float tensor, via a confusion matrix from `bincount`.", setup,
             f"cm = {cm}\nresult = cm.diag().float() / cm.sum(dim=1)", no_loops=True, require=["bincount"])


# ================================================================= challenge

@template(M, "challenge")
def window_peak_memory(rng):
    n = rng.randint(8, 11)
    mem = [rng.randint(1, 40) for _ in range(n)]
    k = rng.randint(2, 4)
    sol = ("from collections import deque\nvals = mem.tolist()\ndq, out = deque(), []\nfor i, v in enumerate(vals):\n"
           "    while dq and vals[dq[-1]] <= v:\n        dq.pop()\n    dq.append(i)\n    if dq[0] <= i - k:\n"
           "        dq.popleft()\n    if i >= k - 1:\n        out.append(vals[dq[0]])\nresult = out")
    tests = [big_ints("mem", seeded(rng), 1, 10**9, 20000) + "\nk = 1", big_ints("mem", seeded(rng), 1, 10**9, 20000) + "\nk = 5000",
             big_ints("mem", seeded(rng), 1, 100, 20000) + "\nk = 137",
             "mem = torch.arange(20000, 0, -1)\nk = 9999", "mem = torch.tensor([5])\nk = 1"]
    return challenge("Peak memory per window",
                     "`mem` holds the GPU memory (MB) a training job used at each step. For every window of `k` "
                     "consecutive steps (left to right), report the **peak** memory in that window. Return a list of "
                     "`n − k + 1` ints.", f"mem = {tt(mem)}\nk = {k}", sol, tests, "1 ≤ k ≤ n ≤ 2·10⁴",
                     speed="O(n) with a monotonic deque (or a vectorized sliding-window view)")


@template(M, "challenge")
def running_kth_largest(rng):
    n = rng.randint(7, 10)
    k = rng.randint(2, 3)
    s = [rng.randint(1, 50) for _ in range(n)]
    sol = ("import heapq\nheap, out = [], []\nfor v in scores.tolist():\n    if len(heap) < k:\n        heapq.heappush(heap, v)\n"
           "    elif v > heap[0]:\n        heapq.heapreplace(heap, v)\n    if len(heap) == k:\n        out.append(heap[0])\n"
           "result = out")
    tests = [big_ints("scores", seeded(rng), 0, 10**9, 20000) + "\nk = 1", big_ints("scores", seeded(rng), 0, 10**9, 20000) + "\nk = 100",
             big_ints("scores", seeded(rng), 0, 50, 20000) + "\nk = 5000", "scores = torch.arange(20000)\nk = 20000",
             "scores = torch.tensor([3, 3, 3])\nk = 2"]
    return challenge("Leaderboard cut-off",
                     "Evaluation `scores` arrive one at a time. After each arrival, once at least `k` scores have been "
                     "seen, report the **k-th largest score so far** (the cut-off to stay on a top-k leaderboard). "
                     "Return the list of cut-offs, starting with the arrival of the k-th score.",
                     f"scores = {tt(s)}\nk = {k}", sol, tests, "1 ≤ k ≤ n ≤ 2·10⁴",
                     speed="O(n log k) with a heap")
