"""3.03 Autograd, gradient descent & backprop."""

from ._common import P, challenge, floats, frows_tt, ftt, lit, seed_line, seeded, template, tt

M = "3.03"


def _poly(rng):
    a, b, c = rng.randint(1, 4), rng.randint(-3, 3), rng.randint(-5, 5)
    kind = rng.choice(["cubic", "sin", "exp", "ratio"])
    if kind == "cubic":
        return f"{a} * x ** 3 + {b} * x ** 2 + {c} * x"
    if kind == "sin":
        return "torch.sin(x) * x ** 2"
    if kind == "exp":
        return f"torch.exp({a} * x) + {c} * x"
    return f"x ** 2 / (1 + x ** 2) + {a} * x"


# ================================================================= lite

@template(M, "lite")
def simple_derivative(rng):
    v = rng.choice([0.5, 1.0, 1.5, 2.0, -1.0, 3.0, -0.5])
    f = _poly(rng)
    return P(f"`x` is a scalar tensor that requires grad. Compute `y = {f}`, call `y.backward()`, and return `x.grad` "
             f"(dy/dx at x = {v}).", f"x = torch.tensor({v}, requires_grad=True)",
             f"y = {f}\ny.backward()\nresult = x.grad", require=["backward|grad"], tol=1e-5)


@template(M, "lite")
def grad_mode_flags(rng):
    vals = floats(rng, rng.randint(2, 4), -3, 3, 1)
    k = rng.randint(2, 5)
    setup = f"x = torch.tensor({lit(vals)}, requires_grad=True)"
    task = rng.choice(["no_grad", "detach", "leaf", "new_leaf"])
    if task == "no_grad":
        return P(f"Compute `y = x * {k}` inside `with torch.no_grad():` and return `y.requires_grad` (no graph is "
                 f"recorded).", setup, f"with torch.no_grad():\n    y = x * {k}\nresult = y.requires_grad",
                 require=["no_grad"])
    if task == "detach":
        return P(f"Compute `y = (x * {k}).detach()` and return `[y.requires_grad, y.is_leaf]`.", setup,
                 f"y = (x * {k}).detach()\nresult = [y.requires_grad, y.is_leaf]", require=["detach"])
    if task == "leaf":
        return P(f"Compute `y = x * {k}`. Return `[x.is_leaf, y.is_leaf, y.requires_grad]`.", setup,
                 f"y = x * {k}\nresult = [x.is_leaf, y.is_leaf, y.requires_grad]")
    return P("Make `z`: an independent copy of `x`'s values that is a **new leaf** requiring grad "
             "(`x.detach().clone().requires_grad_(True)`).", setup, "z = x.detach().clone().requires_grad_(True)",
             post="(z.is_leaf, z.requires_grad, z.tolist(), z.data_ptr() != x.data_ptr())", require=["detach"])


# ================================================================= easy

@template(M, "easy")
def vector_grad(rng):
    n, d = rng.randint(3, 5), rng.randint(2, 3)
    X = [floats(rng, d, -2, 2, 1) for _ in range(n)]
    y = floats(rng, n, -3, 3, 1)
    setup = f"X = {frows_tt(X)}\ny = {ftt(y)}\nw = torch.zeros({d}, requires_grad=True)"
    if rng.random() < 0.5:
        return P(f"For the linear model `X @ w`, compute the mean squared error against `y`, backpropagate, and return "
                 f"`w.grad`.", setup, "loss = ((X @ w - y) ** 2).mean()\nloss.backward()\nresult = w.grad",
                 require=["backward"], tol=1e-5)
    b = round(rng.uniform(-1, 1), 1)
    return P(f"`b = {b}` is a bias that also requires grad. For the model `X @ w + b`, compute the **mean absolute "
             f"error** against `y`, backpropagate, and return `[w.grad, b.grad]`.",
             setup + f"\nb = torch.tensor({b}, requires_grad=True)",
             "loss = (X @ w + b - y).abs().mean()\nloss.backward()\nresult = [w.grad, b.grad]", require=["backward"],
             tol=1e-5)


@template(M, "easy")
def grad_accumulation(rng):
    v = rng.choice([1.0, 2.0, -1.5, 0.5, 3.0])
    k = rng.randint(2, 4)
    a = rng.randint(2, 5)
    setup = f"x = torch.tensor({v}, requires_grad=True)"
    if rng.random() < 0.5:
        return P(f"Run this {k} times in a loop: compute `y = {a} * x ** 2` and call `y.backward()`, **without** zeroing "
                 f"the gradient. Return `x.grad`: gradients accumulate, so it is {k}× one step's gradient.", setup,
                 f"for _ in range({k}):\n    y = {a} * x ** 2\n    y.backward()\nresult = x.grad", require=["backward"])
    return P(f"Run this {k} times in a loop: compute `y = {a} * x ** 2` and call `y.backward()`, but reset the "
             f"gradient with `x.grad.zero_()` **before** each backward after the first (or set `x.grad = None`). Return "
             f"`x.grad` at the end (one step's gradient).", setup,
             f"for _ in range({k}):\n    x.grad = None\n    y = {a} * x ** 2\n    y.backward()\nresult = x.grad",
             require=["backward"])


@template(M, "easy")
def retain_grad_nonleaf(rng):
    n, d = rng.randint(3, 5), rng.randint(2, 3)
    setup = (f"{seed_line(rng)}\nX = torch.randn({n}, {d})\nw = torch.randn({d}, requires_grad=True)\n"
             f"b = torch.zeros(1, requires_grad=True)")
    if rng.random() < 0.5:
        return P(f"Compute `out = X @ w + b` (a non-leaf), ask PyTorch to keep its gradient with `out.retain_grad()`, "
                 f"then backpropagate `loss = out.pow(2).mean()` and return `out.grad`.", setup,
                 "out = X @ w + b\nout.retain_grad()\nloss = out.pow(2).mean()\nloss.backward()\nresult = out.grad",
                 require=["retain_grad"], tol=1e-5)
    return P(f"Backpropagate `loss = (X @ w + b).pow(2).mean()` and return `b.grad`: the bias is broadcast over the "
             f"{n} rows, so its gradient sums their contributions.", setup,
             "loss = (X @ w + b).pow(2).mean()\nloss.backward()\nresult = b.grad", require=["backward"], tol=1e-5)


# ================================================================= medium

@template(M, "medium")
def gradient_descent_fit(rng):
    a, c = rng.choice([2, 3, -1, 0.5, 1.5]), rng.choice([1, -2, 0.5, 4])
    n = rng.randint(6, 10)
    steps, lr = rng.choice([30, 40, 50]), rng.choice([0.05, 0.1])
    setup = (f"x = torch.linspace(-1, 1, {n})\ny = {a} * x + {c}\nw = torch.zeros(1, requires_grad=True)\n"
             f"b = torch.zeros(1, requires_grad=True)")
    return P(f"Fit `y ≈ w * x + b` by hand-written gradient descent: for {steps} steps, compute the mean squared "
             f"error, `backward()`, update both parameters **inside `torch.no_grad()`** with learning rate {lr} "
             f"(`w -= lr * w.grad`), and zero the gradients. Return `[w.item(), b.item()]`.", setup,
             f"for _ in range({steps}):\n    loss = ((w * x + b - y) ** 2).mean()\n    loss.backward()\n"
             f"    with torch.no_grad():\n        w -= {lr} * w.grad\n        b -= {lr} * b.grad\n    w.grad.zero_()\n"
             f"    b.grad.zero_()\nresult = [w.item(), b.item()]", require=["no_grad", "backward"], tol=1e-4)


@template(M, "medium")
def autograd_grad_fn(rng):
    vals = floats(rng, rng.randint(2, 4), -2, 2, 1)
    p = rng.choice([2, 3])
    f = rng.choice([f"(x ** {p}).sum()", "(x.sin() * x).sum()", "torch.logsumexp(x, dim=0)"])
    return P(f"Compute the gradient of `f = {f}` with respect to `x` using the functional "
             f"`torch.autograd.grad(f, x)` (it returns a tuple and leaves `x.grad` untouched). Return the gradient "
             f"tensor.", f"x = torch.tensor({lit(vals)}, requires_grad=True)",
             f"(result,) = torch.autograd.grad({f}, x)", post="(_, x.grad is None)", require=["grad"], tol=1e-5)


@template(M, "medium")
def vjp_backward(rng):
    vals = floats(rng, rng.randint(3, 4), -2, 2, 1)
    g = floats(rng, len(vals), -1, 1, 1)
    f = rng.choice(["v * 2", "v ** 2", "v.exp()", "v * v.sum()"])
    return P(f"`u = {f}` is a **vector**, so `u.backward()` needs the upstream gradient: call "
             f"`u.backward(gradient=g)` (a vector-Jacobian product) and return `v.grad`.",
             f"v = torch.tensor({lit(vals)}, requires_grad=True)\ng = {ftt(g)}",
             f"u = {f}\nu.backward(gradient=g)\nresult = v.grad", require=["backward"], tol=1e-5)


# ================================================================= hard

@template(M, "hard")
def manual_backprop(rng):
    n, d, h, o = rng.randint(3, 4), rng.randint(2, 3), rng.randint(3, 5), rng.randint(1, 2)
    act = rng.choice(["relu", "tanh"])
    setup = (f"{seed_line(rng)}\nX, Y = torch.randn({n}, {d}), torch.randn({n}, {o})\n"
             f"W1, W2 = torch.randn({d}, {h}), torch.randn({h}, {o})")
    fwd = "H = X @ W1\nA = H.relu()\nP = A @ W2" if act == "relu" else "H = X @ W1\nA = H.tanh()\nP = A @ W2"
    dh = "dH = dA * (H > 0)" if act == "relu" else "dH = dA * (1 - A ** 2)"
    return P(f"A 2-layer network: `H = X @ W1`, `A = H.{act}()`, `P = A @ W2`, `loss = ((P - Y) ** 2).mean()`. Compute "
             f"`dW1` and `dW2` **by hand** with the chain rule (no `backward`): `dP = 2 (P − Y) / P.numel()`, "
             f"`dW2 = A.T @ dP`, `dA = dP @ W2.T`, then through the {act}, `dW1 = X.T @ dH`. Return `[dW1, dW2]`.",
             setup, f"{fwd}\ndP = 2 * (P - Y) / P.numel()\ndW2 = A.T @ dP\ndA = dP @ W2.T\n{dh}\ndW1 = X.T @ dH\n"
                    f"result = [dW1, dW2]", tol=1e-5)


@template(M, "hard")
def per_sample_grads(rng):
    n, d = rng.randint(3, 5), rng.randint(2, 3)
    setup = (f"{seed_line(rng)}\nX, y = torch.randn({n}, {d}), torch.randn({n})\nw = torch.randn({d})\nb = torch.tensor(0.5)")
    return P(f"For the model `pred = x @ w + b` with squared-error loss `(pred − y)²` **per example**, return the "
             f"({n}, {d}) matrix whose row i is the gradient of example i's loss with respect to `w`. Use "
             f"`torch.func.vmap(torch.func.grad(loss_fn))`, a loop of `torch.autograd.grad` calls, or the formula "
             f"`2 (pred − y) x`.", setup,
             "from torch.func import grad, vmap\ndef loss_fn(w, x, t):\n    return (x @ w + b - t) ** 2\n"
             "result = vmap(grad(loss_fn), in_dims=(None, 0, 0))(w, X, y)", tol=1e-5)


@template(M, "hard")
def hessian_matrix(rng):
    vals = floats(rng, rng.randint(2, 3), -2, 2, 1)
    f = rng.choice(["(x ** 3).sum()", "(x ** 2).sum() * x[0]", "(x[0] * x[1]) ** 2 + x.sum() ** 2"])
    return P(f"Return the Hessian (matrix of second derivatives) of `f(x) = {f}` at `x` with `torch.func.hessian` "
             f"(or work it out by hand).", f"x = {ftt(vals)}", f"from torch.func import hessian\nresult = hessian(lambda x: {f})(x)",
             tol=1e-3)


# ================================================================= challenge

_DAG = ("g = np.random.default_rng({s})\nn = {n}\nuv = np.sort(g.integers(0, n, size=({m}, 2)), axis=1)\n"
        "uv = uv[uv[:, 0] != uv[:, 1]]\n")


@template(M, "challenge")
def backprop_dag_mod(rng):
    n = rng.randint(5, 6)
    edges = sorted({(u, v) for u, v in (sorted(rng.sample(range(n), 2)) for _ in range(rng.randint(6, 9)))})
    edges = [[u, v, rng.choice([-3, -2, -1, 1, 2, 3, 4])] for u, v in edges]
    rng.shuffle(edges)
    sol = ("MOD = 10**9 + 7\nout = [[] for _ in range(n)]\nfor u, v, w in edges.tolist():\n    out[u].append((v, w))\n"
           "d = [0] * n\nd[0] = 1\nfor u in range(n):\n    if d[u]:\n        for v, w in out[u]:\n"
           "            d[v] = (d[v] + d[u] * w) % MOD\nresult = d[n - 1] % MOD")
    big = _DAG + "edges = torch.from_numpy(np.column_stack([uv, g.integers(-5, 6, size=len(uv))]))"
    tests = [big.format(s=seeded(rng), n=20000, m=60000), big.format(s=seeded(rng), n=300, m=60000),
             "n = 20000\nedges = torch.tensor([[i, i + 1, 2] for i in range(n - 1)])",
             "n = 2\nedges = torch.tensor([[0, 1, -1]])"]
    return challenge("Backprop through a DAG",
                     "A computation graph has nodes `0..n-1`: node 0 is the input x and node n-1 the loss L. Each row "
                     "`(u, v, w)` of `edges` (always `u < v`) says node v uses node u with local derivative "
                     "∂v/∂u = w. By the chain rule, dL/dx is the **sum over all paths** from 0 to n-1 of the product of "
                     "the edge derivatives. Return dL/dx modulo 10⁹ + 7 (as a number in [0, 10⁹ + 7)).",
                     f"n = {n}\nedges = {tt(edges)}", sol, tests, "2 ≤ n ≤ 2·10⁴, m ≤ 6·10⁴ edges, |w| ≤ 5",
                     speed="O(n + m): one pass in topological order")


@template(M, "challenge")
def backward_schedule(rng):
    n = rng.randint(5, 6)
    perm = list(range(n))
    rng.shuffle(perm)
    pairs = sorted({tuple(sorted(rng.sample(range(n), 2))) for _ in range(rng.randint(4, 7))})
    edges = [[perm[a], perm[b]] for a, b in pairs]
    sol = ("import heapq\nconsumers = [0] * n\nproducers = [[] for _ in range(n)]\nfor a, b in edges.tolist():\n"
           "    consumers[a] += 1\n    producers[b].append(a)\nheap = [-i for i in range(n) if consumers[i] == 0]\n"
           "heapq.heapify(heap)\norder = []\nwhile heap:\n    b = -heapq.heappop(heap)\n    order.append(b)\n"
           "    for a in producers[b]:\n        consumers[a] -= 1\n        if consumers[a] == 0:\n"
           "            heapq.heappush(heap, -a)\nresult = order")
    big = _DAG + "edges = torch.from_numpy(g.permutation(n)[uv])"
    tests = [big.format(s=seeded(rng), n=20000, m=50000), big.format(s=seeded(rng), n=20000, m=5000),
             "n = 20000\nedges = torch.tensor([[i + 1, i] for i in range(n - 1)])", "n = 1\nedges = torch.zeros(0, 2, dtype=torch.long)"]
    return challenge("Backward schedule",
                     "Ops `0..n-1` ran in a forward pass; each row `(a, b)` of `edges` means op b consumed op a's "
                     "output (the graph has no cycles). In the backward pass, an op's backward may run only after the "
                     "backward of **every op that consumed its output**. Whenever several ops are ready, the engine "
                     "runs the one with the **largest id**. Return the order in which all n backward ops run.",
                     f"n = {n}\nedges = {tt(edges)}", sol, tests, "1 ≤ n ≤ 2·10⁴, m ≤ 5·10⁴ edges",
                     speed="O((n + m) log n): Kahn's algorithm with a heap")
