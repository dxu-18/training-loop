"""3.10 Hooks & debugging."""

from ._common import P, STDOUT, challenge, floats, lit, seed_line, seeded, template, tt

M = "3.10"


def _net(rng, act="ReLU"):
    a, h, o = rng.randint(3, 8), rng.randint(4, 10), rng.randint(2, 4)
    setup = (f"{seed_line(rng)}\nnet = nn.Sequential(nn.Linear({a}, {h}), nn.{act}(), nn.Linear({h}, {o}))\n"
             f"x = torch.randn({rng.randint(2, 5)}, {a})")
    return a, h, o, setup


# ================================================================= lite

@template(M, "lite")
def forward_hook_capture(rng):
    a, h, o, setup = _net(rng)
    return P("Register a **forward hook** (`hook(module, inputs, output)`) on every `nn.Linear` in `net` that stores "
             "`tuple(output.shape)` in a dict under the module's name from `named_modules()`. Run `net(x)`, remove the "
             "hooks, and return the dict.", setup,
             "shapes = {}\ndef save(name):\n    def hook(module, inputs, output):\n        shapes[name] = tuple(output.shape)\n"
             "    return hook\nhandles = [m.register_forward_hook(save(n)) for n, m in net.named_modules() if isinstance(m, nn.Linear)]\n"
             "net(x)\nfor hd in handles:\n    hd.remove()\nresult = shapes", require=["register_forward_hook"])


@template(M, "lite")
def tensor_hook_scale(rng):
    vals = floats(rng, rng.randint(2, 4), -2, 2, 1)
    k, c = rng.choice([2, 3, 0.5, -1]), rng.randint(2, 5)
    return P(f"Register a **tensor hook** on `x` that multiplies the incoming gradient by {k} "
             f"(`x.register_hook(lambda g: g * {k})`), backpropagate `(x * {c}).sum()`, and return `x.grad`.",
             f"x = torch.tensor({lit(vals)}, requires_grad=True)",
             f"x.register_hook(lambda g: g * {k})\n(x * {c}).sum().backward()\nresult = x.grad", require=["register_hook"])


# ================================================================= easy

@template(M, "easy")
def ablation_hook(rng):
    a, h, o, setup = _net(rng)
    i = rng.choice([0, 1])
    f, what = rng.choice([("output * 0", "zeroes"), ("output * 0.5", "halves"), ("output.clamp(min=0)", "clamps at 0")])
    return P(f"Ablation study: register a forward hook on `net[{i}]` that **returns** `{f}` (a returned value replaces "
             f"the module's output; this one {what} it), run `net(x)`, remove the hook, and return that output.", setup,
             f"hd = net[{i}].register_forward_hook(lambda module, inputs, output: {f})\nresult = net(x)\nhd.remove()",
             require=["register_forward_hook"], tol=1e-5)


@template(M, "easy")
def pre_hook_clamp(rng):
    a, h, o, setup = _net(rng)
    c = rng.choice([0.5, 1.0, 0.25])
    return P(f"A **forward pre-hook** (`hook(module, args)`) can edit a module's inputs: register one on `net[0]` that "
             f"returns `(args[0].clamp(-{c}, {c}),)`, run `net(x)`, remove it, and return the output.", setup,
             f"hd = net[0].register_forward_pre_hook(lambda module, args: (args[0].clamp(-{c}, {c}),))\nresult = net(x)\nhd.remove()",
             require=["register_forward_pre_hook"], tol=1e-5)


@template(M, "easy")
def hook_remove_count(rng):
    a, h, o, setup = _net(rng)
    k, m = rng.randint(2, 4), rng.randint(1, 3)
    return P(f"Register a forward hook on `net[0]` that appends 1 to a list `calls`. Call `net(x)` {k} times, remove the "
             f"hook with its handle, call `net(x)` {m} more times, and return `len(calls)`.", setup,
             f"calls = []\nhd = net[0].register_forward_hook(lambda mod, inp, out: calls.append(1))\nfor _ in range({k}):\n"
             f"    net(x)\nhd.remove()\nfor _ in range({m}):\n    net(x)\nresult = len(calls)", require=["remove"])


# ================================================================= medium

@template(M, "medium")
def grad_norms_per_layer(rng):
    a, h, o, setup = _net(rng)
    return P("Gradient health check: backpropagate `net(x).pow(2).mean()`, then return a dict mapping each parameter's "
             "name (from `named_parameters()`) to its gradient's L2 norm as a float (`p.grad.norm().item()`).", setup,
             "net(x).pow(2).mean().backward()\nresult = {n: p.grad.norm().item() for n, p in net.named_parameters()}",
             require=["named_parameters"], tol=1e-4)


@template(M, "medium")
def backward_hook_gout(rng):
    a, h, o, setup = _net(rng)
    i = rng.choice([0, 2])
    return P(f"Register a **full backward hook** on `net[{i}]` (`hook(module, grad_input, grad_output)`) that saves "
             f"`grad_output[0]` (the gradient of the loss with respect to the module's output). Backpropagate "
             f"`net(x).sum()` and return the saved tensor.", setup,
             f"saved = {{}}\nhd = net[{i}].register_full_backward_hook(lambda m, gin, gout: saved.update(g=gout[0].detach()))\n"
             f"net(x).sum().backward()\nhd.remove()\nresult = saved['g']", require=["register_full_backward_hook"],
             tol=1e-5)


@template(M, "medium")
def activation_stats(rng):
    a, h, o, setup = _net(rng, rng.choice(["ReLU", "Tanh", "GELU"]))
    return P("Record activation statistics for one batch: with forward hooks on every module in `net` (use "
             "`net.named_children()`), store `[output.mean().item(), output.std().item()]` under each child's name. Run "
             "`net(x)`, remove the hooks, and return the dict.", setup,
             "stats = {}\ndef rec(name):\n    def hook(m, inp, out):\n        stats[name] = [out.mean().item(), out.std().item()]\n"
             "    return hook\nhs = [m.register_forward_hook(rec(n)) for n, m in net.named_children()]\nnet(x)\n"
             "for hd in hs:\n    hd.remove()\nresult = stats", require=["register_forward_hook"], tol=1e-4)


# ================================================================= hard

@template(M, "hard")
def find_exploding_layer(rng):
    n_layers = rng.randint(3, 5)
    bad = rng.randrange(n_layers)
    d = rng.randint(4, 8)
    layers = ", ".join(f"nn.Linear({d}, {d}), nn.Tanh()" for _ in range(n_layers))
    setup = (f"{seed_line(rng)}\nnet = nn.Sequential({layers})\nwith torch.no_grad():\n"
             f"    net[{2 * bad}].weight *= 50   # a deliberately bad init somewhere\nx = torch.randn(8, {d})")
    return P("One layer of `net` was badly initialized. Backpropagate `net(x).pow(2).sum()` and return the **name** of "
             "the parameter (from `named_parameters()`) with the largest gradient norm.", setup,
             "net(x).pow(2).sum().backward()\nresult = max(net.named_parameters(), key=lambda kv: kv[1].grad.norm().item())[0]",
             require=["named_parameters"])


@template(M, "hard")
def first_nan_module(rng):
    d = rng.randint(3, 6)
    pos = rng.randint(1, 3)
    mods = [f"nn.Linear({d}, {d})", "nn.ReLU()", f"nn.Linear({d}, {d})", "nn.Tanh()"]
    mods.insert(pos, "Shift()")
    setup = (f"class Shift(nn.Module):\n    def forward(self, x):\n        return (x - 0.5).log()   # NaN for x < 0.5\n\n"
             f"{seed_line(rng)}\nnet = nn.Sequential({', '.join(mods)})\nx = torch.randn(16, {d}) * 3")
    return P("Something in `net` produces NaN. Register a forward hook on **every** module (`named_modules()`) that "
             "records the module's name when its output has any non-finite value (`torch.isfinite(out).all()` is "
             "False). Hooks fire as each module finishes, so the first recorded name is the culprit. Run `net(x)` and "
             "return that name.", setup,
             "bad = []\ndef check(name):\n    def hook(m, inp, out):\n        if not torch.isfinite(out).all():\n"
             "            bad.append(name)\n    return hook\nhs = [m.register_forward_hook(check(n)) for n, m in net.named_modules()]\n"
             "net(x)\nfor hd in hs:\n    hd.remove()\nresult = bad[0]", require=["isfinite|isnan"])


@template(M, "hard")
def shape_logger(rng):
    a, h, o, setup = _net(rng)
    return P("Write a forward hook `shape_logger(m, inp, out)` that prints `f\"{m.__class__.__name__} "
             "{tuple(inp[0].shape)} -> {tuple(out.shape)}\"`, register it on every layer of `net` (iterate over `net`), "
             "run `net(x)` once, and remove the hooks. (Graded on what you print.)", setup,
             "def shape_logger(m, inp, out):\n    print(f\"{m.__class__.__name__} {tuple(inp[0].shape)} -> {tuple(out.shape)}\")\n"
             "hs = [m.register_forward_hook(shape_logger) for m in net]\nnet(x)\nfor hd in hs:\n    hd.remove()",
             answer=[STDOUT], require=["register_forward_hook"])


# ================================================================= challenge

@template(M, "challenge")
def range_max_activation(rng):
    n = rng.randint(6, 9)
    peak = [rng.randint(1, 60) for _ in range(n)]
    qs = [sorted(rng.sample(range(n), 2)) for _ in range(rng.randint(3, 5))]
    sol = ("vals = peak.tolist()\nn = len(vals)\ntable = [vals]\nj = 1\nwhile 2 * j <= n:\n    prev = table[-1]\n"
           "    table.append([max(prev[i], prev[i + j]) for i in range(n - 2 * j + 1)])\n    j *= 2\nout = []\n"
           "for l, r in queries.tolist():\n    k = (r - l + 1).bit_length() - 1\n"
           "    out.append(max(table[k][l], table[k][r - (1 << k) + 1]))\nresult = out")
    big = ("g = np.random.default_rng({s})\npeak = torch.from_numpy(g.integers(0, {hi}, size=20000))\n"
           "queries = torch.from_numpy(np.sort(g.integers(0, 20000, size=(20000, 2)), axis=1))")
    tests = [big.format(s=seeded(rng), hi=10**9), big.format(s=seeded(rng), hi=3),
             "peak = torch.arange(20000)\nqueries = torch.tensor([[0, 19999]] * 20000)", "peak = torch.tensor([4])\nqueries = torch.tensor([[0, 0]])"]
    return challenge("Peak activation over a range of layers",
                     "A forward hook recorded `peak[i]`, the largest activation of layer i. Each row `(l, r)` of "
                     "`queries` asks for the largest peak among layers `l..r` (inclusive). Return the list of answers.",
                     f"peak = {tt(peak)}\nqueries = {tt(qs)}", sol, tests, "1 ≤ n, q ≤ 2·10⁴",
                     speed="O((n + q) log n) or better: a sparse table answers each query in O(1)")


@template(M, "challenge")
def spike_pairs(rng):
    n = rng.randint(6, 9)
    g = [rng.randint(1, 20) for _ in range(n)]
    sol = ("from bisect import bisect_right, insort\nseen, total = [], 0\nfor v in grad_norm.tolist():\n"
           "    total += bisect_right(seen, v // 2)\n    insort(seen, v)\nresult = total")
    big = "grad_norm = torch.from_numpy(np.random.default_rng({s}).integers(1, {hi}, size=20000))"
    tests = [big.format(s=seeded(rng), hi=10**9), big.format(s=seeded(rng), hi=4), "grad_norm = torch.arange(1, 20001)",
             "grad_norm = torch.full((20000,), 7)", "grad_norm = torch.tensor([3])"]
    return challenge("Gradient spikes",
                     "`grad_norm[t]` is the (integer) gradient norm logged at step t. A **spike pair** is a pair of steps "
                     "`i < j` where the later norm is at least double the earlier one: `grad_norm[j] ≥ 2 · grad_norm[i]`. "
                     "Count the spike pairs.", f"grad_norm = {tt(g)}", sol, tests, "1 ≤ n ≤ 2·10⁴, values ≤ 10⁹",
                     speed="O(n log n): binary search in a sorted list (or a Fenwick tree)")
