"""3.18 torch.compile.

In the browser torch.compile has no code generator: compiled code runs eagerly with identical
results, while option checks, `_orig_mod`, `fullgraph=True` errors and `torch._dynamo.explain`
graph-break counts (print, .item(), .tolist(), .numpy(), calls to disabled functions) behave as usual.
"""

from ._common import P, challenge, seed_line, seeded, template, tt

M = "3.18"


def _x(rng, n=None):
    n = n or rng.randint(3, 6)
    return f"{seed_line(rng)}\nx = torch.randn({n})"


# ================================================================= lite

@template(M, "lite")
def compile_function(rng):
    fn = rng.choice([("gelu_tanh", "0.5 * x * (1 + torch.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))"),
                     ("swish", "x * torch.sigmoid(x)"), ("softplus", "torch.log1p(torch.exp(x))"),
                     ("rms", "x * torch.rsqrt(x.pow(2).mean() + 1e-6)")])
    if rng.random() < 0.5:
        return P(f"Decorate `def {fn[0]}(x): return {fn[1]}` with `@torch.compile` and return `{fn[0]}(x)`. (The first "
                 f"call compiles; later calls reuse the compiled code.)", _x(rng),
                 f"@torch.compile\ndef {fn[0]}(x):\n    return {fn[1]}\nresult = {fn[0]}(x)", require=["compile"], tol=1e-5)
    return P(f"Without a decorator: define `def {fn[0]}(x): return {fn[1]}`, wrap it as `fast = torch.compile({fn[0]})`, "
             f"and return `fast(x)`.", _x(rng), f"def {fn[0]}(x):\n    return {fn[1]}\nfast = torch.compile({fn[0]})\nresult = fast(x)",
             require=["compile"], tol=1e-5)


@template(M, "lite")
def compile_modes(rng):
    d = rng.choice([4, 8, 16])
    opt = rng.choice([("mode", "'reduce-overhead'", "CUDA graphs, cutting launch overhead for small batches"),
                      ("mode", "'max-autotune'", "the longest compile and the fastest kernels"),
                      ("backend", "'eager'", "capture only, for debugging"),
                      ("backend", "'aot_eager'", "capture plus AOTAutograd, for debugging"),
                      ("dynamic", "True", "expecting varying input shapes")])
    setup = f"{seed_line(rng)}\nmodel = nn.Sequential(nn.Linear({d}, {d}), nn.GELU(), nn.Linear({d}, 2))\nx = torch.randn(4, {d})"
    return P(f"Compile `model` with `{opt[0]}={opt[1]}` ({opt[2]}) and return `torch.allclose(compiled(x), model(x))`: "
             f"compiling never changes results.", setup,
             f"compiled = torch.compile(model, {opt[0]}={opt[1]})\nresult = torch.allclose(compiled(x), model(x))",
             require=["compile", opt[0]])


# ================================================================= easy

@template(M, "easy")
def orig_mod_keys(rng):
    d = rng.choice([4, 8])
    setup = f"model = nn.Sequential(nn.Linear({d}, {d}), nn.ReLU(), nn.Linear({d}, 1))"
    task = rng.choice(["wrapper", "inplace", "same"])
    if task == "wrapper":
        return P("`torch.compile(model)` returns a wrapper module. Return `list(torch.compile(model).state_dict())`: its "
                 "keys gain an `_orig_mod.` prefix (a common checkpoint-loading surprise).", setup,
                 "list(torch.compile(model).state_dict())", require=["compile"])
    if task == "inplace":
        return P("`model.compile()` compiles **in place** and keeps the state-dict keys clean. Call it and return "
                 "`list(model.state_dict())`.", setup, "model.compile()\nresult = list(model.state_dict())",
                 require=["compile"])
    return P("For `cm = torch.compile(model)`, return `[cm._orig_mod is model, sum(p.numel() for p in cm.parameters()) == "
             "sum(p.numel() for p in model.parameters())]`: the wrapper holds the original module and shares its "
             "parameters.", setup,
             "cm = torch.compile(model)\nresult = [cm._orig_mod is model, sum(p.numel() for p in cm.parameters()) == "
             "sum(p.numel() for p in model.parameters())]", require=["_orig_mod"])


@template(M, "easy")
def compiler_disable(rng):
    k = rng.choice([2, 3, 0.5])
    return P(f"Keep a logging helper out of the graph: decorate `def log_stats(t): print(tuple(t.shape)); return t` with "
             f"`@torch.compiler.disable`, then compile `def f(x): return log_stats(x * {k}) + 1` and return "
             f"`[f(x), torch._dynamo.explain(f)(x).graph_break_count]` (a call into a disabled function is one graph "
             f"break).", _x(rng),
             f"@torch.compiler.disable\ndef log_stats(t):\n    print(tuple(t.shape))\n    return t\n@torch.compile\ndef f(x):\n"
             f"    return log_stats(x * {k}) + 1\nresult = [f(x), torch._dynamo.explain(f)(x).graph_break_count]",
             require=["disable"], tol=1e-6)


# ================================================================= medium

_BREAKS = [("v = y.sum().item()", "y = y + v"), ("print(y.shape)", ""), ("vals = y.tolist()", "y = y * len(vals)"),
           ("arr = y.detach().numpy()", "y = y + float(arr.mean())")]


def _broken_fn(rng, k):
    picks = rng.sample(_BREAKS, k)
    body = ["    y = x.sin()"]
    for a, b in picks:
        body.append("    " + a)
        if b:
            body.append("    " + b)
    body.append("    return y * 2")
    return "def f(x):\n" + "\n".join(body)


@template(M, "medium")
def explain_graph_breaks(rng):
    k = rng.randint(1, 3)
    src = _broken_fn(rng, k)
    what = rng.choice(["graph_break_count", "graph_count"])
    return P(f"Count graph breaks with `torch._dynamo.explain(f)(x)`, which returns an object with `graph_count` and "
             f"`graph_break_count`. Return `.{what}` for this function (`.item()`, `.tolist()`, `.numpy()` and `print` "
             f"each force a break):\n\n```\n{src}\n```", _x(rng) + "\n" + src,
             f"result = torch._dynamo.explain(f)(x).{what}", require=["explain"])


@template(M, "medium")
def fullgraph_error(rng):
    broken = rng.random() < 0.6
    src = _broken_fn(rng, 1) if broken else "def f(x):\n    y = x.sin()\n    return torch.where(y > 0, y, -y) * 2"
    return P(f"`fullgraph=True` turns any graph break into an error. Compile `f` with `torch.compile(f, fullgraph=True)`, "
             f"call it, and return `True` if it raised `torch._dynamo.exc.Unsupported`, else `False`:\n\n```\n{src}\n```",
             _x(rng) + "\n" + src,
             "try:\n    torch.compile(f, fullgraph=True)(x)\n    result = False\nexcept torch._dynamo.exc.Unsupported:\n    result = True",
             require=["fullgraph"])


@template(M, "medium")
def remove_graph_break(rng):
    t = rng.choice([0.0, 0.5, -0.5])
    kind = rng.choice(["const", "exp", "scale"])
    if kind == "const":
        hi, lo = rng.choice([(2.0, -1.0), (1.0, 0.0), (5.0, 3.0)])
        src = f"def slow(x):\n    return torch.tensor([{hi} if v > {t} else {lo} for v in x.tolist()])"
        sol = f"torch.where(x > {t}, torch.full_like(x, {hi}), torch.full_like(x, {lo}))"
    else:
        a, b = ("v.exp()", "v") if kind == "exp" else ("v * 3", "v / 3")
        src = f"def slow(x):\n    return torch.stack([{a} if v.item() > {t} else {b} for v in x])"
        sol = f"torch.where(x > {t}, {a.replace('v', 'x')}, {b.replace('v', 'x')})"
    return P(f"`slow` pulls tensor values into Python (`.item()` / `.tolist()`), which breaks the graph. Write "
             f"`def fixed(x)` that computes the same values with `torch.where` and **no** graph breaks:\n\n```\n{src}\n```",
             _x(rng, rng.randint(4, 7)) + "\n" + src, f"def fixed(x):\n    return {sol}",
             post="(torch._dynamo.explain(fixed)(x).graph_break_count, fixed(x))", require=["where"], tol=1e-5)


# ================================================================= hard

@template(M, "hard")
def compile_training_step(rng):
    d, k = rng.randint(3, 6), rng.randint(2, 4)
    setup = (f"{seed_line(rng)}\nX, y = torch.randn(16, {d}), torch.randn(16, 1)\n"
             f"model = nn.Sequential(nn.Linear({d}, 8), nn.ReLU(), nn.Linear(8, 1))")
    return P(f"Train through a compiled model: `cmodel = torch.compile(model)`; optimizer `torch.optim.AdamW(cmodel."
             f"parameters(), lr=1e-2)` (the wrapper shares the parameters); {k} full-batch MSE steps calling `cmodel(X)`. "
             f"Return `model[0].weight.detach()` (the original module was trained).", setup,
             f"cmodel = torch.compile(model)\nopt = torch.optim.AdamW(cmodel.parameters(), lr=1e-2)\nfor _ in range({k}):\n"
             f"    loss = F.mse_loss(cmodel(X), y)\n    opt.zero_grad()\n    loss.backward()\n    opt.step()\n"
             f"result = model[0].weight.detach()", require=["compile"], tol=1e-5)


@template(M, "hard")
def fused_bias_gelu(rng):
    n, d = rng.randint(2, 4), rng.choice([4, 8])
    setup = f"{seed_line(rng)}\nx, b = torch.randn({n}, {d}), torch.randn({d})"
    return P("Fusion is where `torch.compile` shines: several elementwise ops become one kernel. Write "
             "`@torch.compile def bias_gelu(x, b)` computing the tanh-approximated GELU of `x + b` by its formula "
             "`0.5 z (1 + tanh(sqrt(2/π) (z + 0.044715 z³)))`, and return `[bias_gelu(x, b), torch.allclose(bias_gelu(x, "
             "b), F.gelu(x + b, approximate='tanh'), atol=1e-6)]`.", setup,
             "@torch.compile\ndef bias_gelu(x, b):\n    z = x + b\n    return 0.5 * z * (1 + torch.tanh(math.sqrt(2 / math.pi) * (z + 0.044715 * z ** 3)))\n"
             "result = [bias_gelu(x, b), torch.allclose(bias_gelu(x, b), F.gelu(x + b, approximate='tanh'), atol=1e-6)]",
             require=["compile", "tanh"], tol=1e-5)


# ================================================================= challenge

@template(M, "challenge")
def count_recompiles(rng):
    n = rng.randint(5, 8)
    shapes = [[rng.choice([1, 2, 4, 8]), rng.choice([16, 16, 32]), 3] for _ in range(n)]
    limit = rng.randint(2, 4)
    sol = ("calls = shapes.tolist()\nfirst, dyn, graphs, compiles, fallbacks = None, set(), [], 0, 0\n"
           "def accepts(p, s):\n    return all((q == '*' and v >= 2) or q == v for q, v in zip(p, s))\nfor s in calls:\n"
           "    if any(accepts(p, s) for p in graphs):\n        continue\n    if first is None:\n        first = s\n    else:\n"
           "        dyn |= {i for i, (a, b) in enumerate(zip(first, s)) if a != b}\n"
           "    pat = tuple('*' if i in dyn and v >= 2 else v for i, v in enumerate(s))\n    if compiles < limit:\n"
           "        compiles += 1\n        graphs.append(pat)\n    else:\n        fallbacks += 1\nresult = [compiles, fallbacks]")
    big = ("g = np.random.default_rng({s})\nshapes = torch.from_numpy(np.stack([g.choice({a}, 20000), g.choice({b}, 20000), "
           "g.choice({c}, 20000)], axis=1))\nlimit = {L}")
    tests = [big.format(s=seeded(rng), a=[1, 2, 3, 4, 8], b=[128, 256, 512], c=[3], L=8),
             big.format(s=seeded(rng), a=[1, 1, 1, 7], b=[0, 1, 5], c=[1, 2], L=3),
             big.format(s=seeded(rng), a=list(range(2, 50)), b=[4], c=[4], L=1),
             "shapes = torch.tensor([[2, 2, 2]])\nlimit = 8"]
    return challenge("Counting recompiles",
                     "A compiled function guards on input shapes. The first call compiles a graph specialized to its exact "
                     "shape. Later, a call reuses any compiled graph whose guards accept its shape. Otherwise it is a "
                     "miss: every dimension whose size differs from the **first** call's becomes **dynamic** for good, "
                     "and a new graph is compiled whose pattern uses `*` in dynamic dimensions (accepting any size ≥ 2; "
                     "sizes 0 and 1 are always specialized) and exact sizes elsewhere. Once `limit` graphs exist, misses "
                     "fall back to eager instead. Each row of `shapes` is one call's input shape. Return "
                     "`[graphs compiled, eager fallbacks]`.", f"shapes = {tt(shapes)}\nlimit = {limit}", sol, tests,
                     "1 ≤ n ≤ 2·10⁴ calls, limit ≤ 8", speed="O(n · limit · dims)")


@template(M, "challenge")
def longest_graph_k_fixes(rng):
    n = rng.randint(8, 12)
    ops = [1 if rng.random() < 0.3 else 0 for _ in range(n)]
    k = rng.randint(0, 2)
    sol = ("flags = is_break.tolist()\nbest = lo = breaks = 0\nfor hi, f in enumerate(flags):\n    breaks += f\n"
           "    while breaks > k:\n        breaks -= flags[lo]\n        lo += 1\n    best = max(best, hi - lo + 1)\nresult = best")
    big = "is_break = torch.from_numpy((np.random.default_rng({s}).random(20000) < {p}).astype(np.int64))\nk = {k}"
    tests = [big.format(s=seeded(rng), p=0.1, k=0), big.format(s=seeded(rng), p=0.5, k=40),
             big.format(s=seeded(rng), p=0.001, k=3), "is_break = torch.ones(20000, dtype=torch.long)\nk = 0",
             "is_break = torch.tensor([0])\nk = 0"]
    return challenge("Fixing graph breaks",
                     "`is_break[i]` is 1 if op i of a traced function causes a graph break (a `.item()`, a `print`, ...) "
                     "and 0 otherwise. You may rewrite at most `k` of the breaking ops so they no longer break. A graph is "
                     "a maximal run of consecutive non-breaking ops. Return the length of the **longest graph** you can "
                     "get.", f"is_break = {tt(ops)}\nk = {k}", sol, tests, "1 ≤ n ≤ 2·10⁴, 0 ≤ k ≤ n",
                     speed="O(n) with two pointers")
