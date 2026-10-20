"""3.06 Custom autograd Functions."""

from ._common import P, challenge, floats, ftt, lit, seeded, template, tt

M = "3.06"
IMPORTS = "from torch.autograd import Function, gradcheck\n"


def D(prompt, setup, solution, **kw):
    return P(prompt, IMPORTS + setup, solution, **kw)


def _vals(rng, n=None):
    vals = floats(rng, n or rng.randint(3, 5), -3, 3, 1)
    return [v if abs(v) > 0.05 else 0.7 for v in vals]  # stay away from kinks at 0


def _probe(cls, vals, weights, extra=""):
    """Post-check: apply cls to a fresh leaf, backprop a weighted sum, return (output, grad)."""
    return (f"_x = torch.tensor({lit(vals)}, requires_grad=True)\n_y = {cls}.apply(_x{extra})\n"
            f"(_y * {ftt(weights)}).sum().backward()\n(_y.detach(), _x.grad)")


# ================================================================= lite

_GIVEN = {
    "MyReLU": ("    @staticmethod\n    def forward(ctx, x):\n        ctx.save_for_backward(x)\n        return x.clamp(min=0)\n"
               "    @staticmethod\n    def backward(ctx, grad_output):\n        (x,) = ctx.saved_tensors\n"
               "        return grad_output * (x > 0).to(grad_output.dtype)"),
    "Cube": ("    @staticmethod\n    def forward(ctx, x):\n        ctx.save_for_backward(x)\n        return x ** 3\n"
             "    @staticmethod\n    def backward(ctx, g):\n        (x,) = ctx.saved_tensors\n        return 3 * x ** 2 * g"),
    "Softsign": ("    @staticmethod\n    def forward(ctx, x):\n        ctx.save_for_backward(x)\n        return x / (1 + x.abs())\n"
                 "    @staticmethod\n    def backward(ctx, g):\n        (x,) = ctx.saved_tensors\n        return g / (1 + x.abs()) ** 2"),
}


@template(M, "lite")
def apply_given_function(rng):
    name = rng.choice(list(_GIVEN))
    vals = _vals(rng)
    setup = f"class {name}(Function):\n{_GIVEN[name]}\n\nx = torch.tensor({lit(vals)}, requires_grad=True)"
    if rng.random() < 0.5:
        return D(f"`{name}` is a custom autograd Function. Call it on `x` the right way (`{name}.apply(x)`, never "
                 f"`{name}()(x)` or `forward` directly), backpropagate the sum of the output, and return `x.grad`.",
                 setup, f"{name}.apply(x).sum().backward()\nresult = x.grad", require=["apply"], tol=1e-5)
    return D(f"`{name}` is a custom autograd Function. Return `[y.detach(), y.requires_grad]` for `y = {name}.apply(x)`: "
             f"the output is part of the graph even though `forward` ran without recording one.", setup,
             f"y = {name}.apply(x)\nresult = [y.detach(), y.requires_grad]", require=["apply"], tol=1e-5)


@template(M, "lite")
def ste_inline(rng):
    vals = floats(rng, rng.randint(3, 5), -3, 3, 2)
    kind = rng.choice(["round", "sign", "floor"])
    return D(f"A **straight-through estimator** uses `{kind}` in the forward pass but pretends it is the identity in "
             f"backward: `y = x + (x.{kind}() - x).detach()`. Compute `y`, backpropagate `y.sum()`, and return "
             f"`[y.detach(), x.grad]`.", f"x = torch.tensor({lit(vals)}, requires_grad=True)",
             f"y = x + (x.{kind}() - x).detach()\ny.sum().backward()\nresult = [y.detach(), x.grad]", require=["detach"],
             tol=1e-6)


# ================================================================= easy

_DERIV = {
    "Square": ("x ** 2", "ctx.save_for_backward(x)", "(x,) = ctx.saved_tensors\n        return 2 * x * g",
               "save `x` with `ctx.save_for_backward` and return `2 * x * grad`"),
    "Cube": ("x ** 3", "ctx.save_for_backward(x)", "(x,) = ctx.saved_tensors\n        return 3 * x ** 2 * g",
             "save `x` and return `3 * x**2 * grad`"),
    "Exp": ("x.exp()", None, "(y,) = ctx.saved_tensors\n        return g * y",
            "save the **output** (the derivative of exp is exp itself) and return `grad * output`"),
    "Sine": ("x.sin()", "ctx.save_for_backward(x)", "(x,) = ctx.saved_tensors\n        return g * x.cos()",
             "save `x` and return `grad * cos(x)`"),
}


@template(M, "easy")
def write_function(rng):
    name = rng.choice(list(_DERIV))
    fwd, save, back, hint = _DERIV[name]
    vals, w = _vals(rng), floats(rng, 4, -2, 2, 1)
    vals = vals[:4] + [0.5] * (4 - len(vals[:4]))
    body = (f"        y = {fwd}\n        ctx.save_for_backward(y)\n        return y" if save is None else
            f"        {save}\n        return {fwd}")
    return D(f"Write `class {name}(Function)` computing `{fwd}` with static `forward(ctx, x)` and `backward(ctx, g)`: "
             f"{hint}. It is checked on `x = {lit(vals)}`.", "",
             f"class {name}(Function):\n    @staticmethod\n    def forward(ctx, x):\n{body}\n    @staticmethod\n"
             f"    def backward(ctx, g):\n        {back}", post=_probe(name, vals, w), require=["Function", "backward"],
             tol=1e-5)


@template(M, "easy")
def grad_reverse(rng):
    lam = rng.choice([0.5, 1.0, 2.0, 0.1])
    vals, w = _vals(rng, 3), floats(rng, 3, -2, 2, 1)
    return D("Write `class GradReverse(Function)` for domain adaptation: `forward(ctx, x, lam)` stores `lam` on `ctx` and "
             "returns `x.view_as(x)` (the identity); `backward(ctx, g)` returns `(-ctx.lam * g, None)`: **one gradient "
             f"per forward input**, None for the non-tensor `lam`. It is checked with `lam={lam}` on `x = {lit(vals)}`.", "",
             "class GradReverse(Function):\n    @staticmethod\n    def forward(ctx, x, lam):\n        ctx.lam = lam\n"
             "        return x.view_as(x)\n    @staticmethod\n    def backward(ctx, g):\n        return -ctx.lam * g, None",
             post=_probe("GradReverse", vals, w, f", {lam}"), require=["Function"], tol=1e-6)


@template(M, "easy")
def scaled_mul_none(rng):
    s = rng.choice([0.5, 2.0, 3.0, -1.0])
    a, b, w = _vals(rng, 3), _vals(rng, 3), floats(rng, 3, -2, 2, 1)
    return D("Write `class ScaledMul(Function)` with `forward(ctx, a, b, scale)` returning `a * b * scale` (save `a` and "
             "`b`, keep `scale` on ctx) and a `backward` that returns `(grad_a, grad_b, None)`, computing a gradient only "
             f"if `ctx.needs_input_grad[i]` is True (otherwise None). It is checked with `scale={s}` on `a = {lit(a)}` "
             f"(requires grad) and `b = {lit(b)}` (doesn't).", "",
             "class ScaledMul(Function):\n    @staticmethod\n    def forward(ctx, a, b, scale):\n"
             "        ctx.save_for_backward(a, b)\n        ctx.scale = scale\n        return a * b * scale\n"
             "    @staticmethod\n    def backward(ctx, g):\n        a, b = ctx.saved_tensors\n"
             "        ga = g * b * ctx.scale if ctx.needs_input_grad[0] else None\n"
             "        gb = g * a * ctx.scale if ctx.needs_input_grad[1] else None\n        return ga, gb, None",
             post=(f"_a = torch.tensor({lit(a)}, requires_grad=True)\n_b = torch.tensor({lit(b)})\n"
                   f"_y = ScaledMul.apply(_a, _b, {s})\n(_y * {ftt(w)}).sum().backward()\n(_y.detach(), _a.grad, _b.grad is None)"),
             require=["needs_input_grad"], tol=1e-6)


# ================================================================= medium

@template(M, "medium")
def sigmoid_gradcheck(rng):
    n = rng.randint(3, 5)
    seed = seeded(rng)
    return D("Write `class MySigmoid(Function)`: `forward` computes `1 / (1 + exp(-x))` and saves the **output** "
             "(σ' = σ(1 − σ) needs only σ); `backward` returns `g * s * (1 - s)`. It must pass "
             f"`gradcheck(MySigmoid.apply, (x,))` on a float64 input (checked on `torch.randn({n})` after "
             f"`torch.manual_seed({seed})`).", "",
             "class MySigmoid(Function):\n    @staticmethod\n    def forward(ctx, x):\n        s = 1 / (1 + torch.exp(-x))\n"
             "        ctx.save_for_backward(s)\n        return s\n    @staticmethod\n    def backward(ctx, g):\n"
             "        (s,) = ctx.saved_tensors\n        return g * s * (1 - s)",
             post=(f"torch.manual_seed({seed})\n_x = torch.randn({n}, dtype=torch.float64, requires_grad=True)\n"
                   f"(gradcheck(MySigmoid.apply, (_x,), eps=1e-6, atol=1e-4), MySigmoid.apply(_x).detach())"),
             require=["save_for_backward"], tol=1e-6)


@template(M, "medium")
def clipped_identity(rng):
    c = rng.choice([1.0, 0.5, 2.0])
    k = rng.choice([3.0, 5.0, -4.0, 10.0])
    vals = _vals(rng, 4)
    return D(f"Gradient clipping as a layer: write `class ClipGrad(Function)` whose `forward(ctx, x)` is the identity "
             f"(return `x.clone()`) and whose `backward(ctx, g)` returns `g.clamp(-{c}, {c})`. It is checked by backpropagating "
             f"`(ClipGrad.apply(x) * {k}).sum()` for `x = {lit(vals)}`.", "",
             f"class ClipGrad(Function):\n    @staticmethod\n    def forward(ctx, x):\n        return x.clone()\n"
             f"    @staticmethod\n    def backward(ctx, g):\n        return g.clamp(-{c}, {c})",
             post=f"_x = torch.tensor({lit(vals)}, requires_grad=True)\n(ClipGrad.apply(_x) * {k}).sum().backward()\n_x.grad",
             require=["clamp|clip"], tol=1e-6)


@template(M, "medium")
def setup_context_style(rng):
    vals, w = _vals(rng, 3), floats(rng, 3, -2, 2, 1)
    f, save, grad = rng.choice([("x.exp()", "output", "g * y"), ("x ** 2", "inputs[0]", "2 * y * g"),
                                ("x.tanh()", "output", "g * (1 - y ** 2)")])
    return D(f"Write a **new-style** Function `class Op(Function)` for `{f}`: `forward(x)` takes no ctx; a separate "
             f"`setup_context(ctx, inputs, output)` saves `{save}` with `ctx.save_for_backward`; `backward(ctx, g)` "
             f"returns the gradient (`y` below is the saved tensor): `{grad}`. It is checked on `x = {lit(vals)}`.", "",
             f"class Op(Function):\n    @staticmethod\n    def forward(x):\n        return {f}\n    @staticmethod\n"
             f"    def setup_context(ctx, inputs, output):\n        ctx.save_for_backward({save})\n    @staticmethod\n"
             f"    def backward(ctx, g):\n        (y,) = ctx.saved_tensors\n        return {grad}",
             post=_probe("Op", vals, w), require=["setup_context"], tol=1e-5)


# ================================================================= hard

@template(M, "hard")
def swish_function(rng):
    seed = seeded(rng)
    return D("Implement Swish `x · σ(x)` as `class Swish(Function)`, saving only `x` for backward. Its derivative is "
             f"`σ(x) + x σ(x)(1 − σ(x))`. It must pass `gradcheck` on float64 inputs (checked on `torch.randn(5)` "
             f"after `torch.manual_seed({seed})`).", "",
             "class Swish(Function):\n    @staticmethod\n    def forward(ctx, x):\n        ctx.save_for_backward(x)\n"
             "        return x * torch.sigmoid(x)\n    @staticmethod\n    def backward(ctx, g):\n        (x,) = ctx.saved_tensors\n"
             "        s = torch.sigmoid(x)\n        return g * (s + x * s * (1 - s))",
             post=(f"torch.manual_seed({seed})\n_x = torch.randn(5, dtype=torch.float64, requires_grad=True)\n"
                   f"(gradcheck(Swish.apply, (_x,), eps=1e-6, atol=1e-4), Swish.apply(_x).detach())"),
             require=["Function"], tol=1e-6)


@template(M, "hard")
def huber_function(rng):
    delta = rng.choice([0.5, 1.0, 1.5])
    seed = seeded(rng)
    return D(f"Write `class Huber(Function)` with `forward(ctx, x, delta)`: elementwise `0.5 x²` where `|x| < delta`, "
             f"else `delta (|x| − 0.5 delta)`. Its derivative is `clamp(x, −delta, delta)`. `backward` must return one "
             f"gradient per input (None for `delta`). It must pass `gradcheck` with `delta={delta}` (checked on "
             f"`2 * torch.randn(6)` after `torch.manual_seed({seed})`).", "",
             "class Huber(Function):\n    @staticmethod\n    def forward(ctx, x, delta):\n        ctx.save_for_backward(x)\n"
             "        ctx.delta = delta\n        return torch.where(x.abs() < delta, 0.5 * x ** 2, delta * (x.abs() - 0.5 * delta))\n"
             "    @staticmethod\n    def backward(ctx, g):\n        (x,) = ctx.saved_tensors\n"
             "        return g * x.clamp(-ctx.delta, ctx.delta), None",
             post=(f"torch.manual_seed({seed})\n_x = (torch.randn(6, dtype=torch.float64) * 2).requires_grad_(True)\n"
                   f"(gradcheck(Huber.apply, (_x, {delta}), eps=1e-6, atol=1e-4), Huber.apply(_x, {delta}).detach())"),
             require=["Function"], tol=1e-6)


@template(M, "hard")
def fake_quant_module(rng):
    scale = rng.choice([0.5, 0.25, 0.1])
    vals = floats(rng, 5, -2, 2, 2)
    return D(f"Quantization-aware training: write `class RoundSTE(Function)` (forward `x.round()`, backward the identity "
             f"`g`) and wrap it in `class FakeQuant(nn.Module)` with `__init__(self, scale)` whose forward returns "
             f"`RoundSTE.apply(x / scale) * scale`: values snap to multiples of `scale`, gradients pass straight through. "
             f"It is checked with `scale={scale}` on `x = {lit(vals)}`.", "",
             "class RoundSTE(Function):\n    @staticmethod\n    def forward(ctx, x):\n        return x.round()\n"
             "    @staticmethod\n    def backward(ctx, g):\n        return g\n\nclass FakeQuant(nn.Module):\n"
             "    def __init__(self, scale):\n        super().__init__()\n        self.scale = scale\n    def forward(self, x):\n"
             "        return RoundSTE.apply(x / self.scale) * self.scale",
             post=(f"_x = torch.tensor({lit(vals)}, requires_grad=True)\n_q = FakeQuant({scale})(_x)\n"
                   f"(_q * 3).sum().backward()\n(_q.detach(), _x.grad)"), require=["Function", "Module"], tol=1e-6)


# ================================================================= challenge

@template(M, "challenge")
def max_grad_scale_chain(rng):
    n = rng.randint(6, 9)
    f = [rng.choice([-3, -2, -1, 0, 1, 2, 3]) for _ in range(n)]
    f[rng.randrange(n)] = rng.choice([2, 3])
    sol = ("vals = factors.tolist()\nbest = hi = lo = vals[0]\nfor a in vals[1:]:\n    c = (a, hi * a, lo * a)\n"
           "    hi, lo = max(c), min(c)\n    best = max(best, hi)\nresult = best")
    big = "factors = torch.from_numpy(np.random.default_rng({s}).integers({lo}, {hi}, size=20000))"
    tests = [big.format(s=seeded(rng), lo=-3, hi=4), big.format(s=seeded(rng), lo=-1, hi=2),
             big.format(s=seeded(rng), lo=-3, hi=0), "factors = torch.full((20000,), -2)", "factors = torch.tensor([0])"]
    return challenge("The strongest gradient chain",
                     "A deep stack of custom Functions scales the gradient: Function i multiplies it by the integer "
                     "`factors[i]`. For a **contiguous** non-empty block of Functions the total scale is the product of "
                     "their factors. Return the **largest** total scale over all blocks, as an exact Python int.",
                     f"factors = {tt(f)}", sol, tests, "1 ≤ n ≤ 2·10⁴, −3 ≤ factor ≤ 3",
                     speed="O(n): track the largest and smallest product ending at each position")


@template(M, "challenge")
def range_grad_products(rng):
    n = rng.randint(6, 8)
    f = [rng.choice([-2, -1, 1, 2, 3, 5, 0]) for _ in range(n)]
    qs = [sorted(rng.sample(range(n), 2)) for _ in range(rng.randint(3, 5))]
    sol = ("MOD = 10**9 + 7\nvals = factors.tolist()\nzeros, pref = [0], [1]\nfor v in vals:\n    zeros.append(zeros[-1] + (v == 0))\n"
           "    pref.append(pref[-1] * (v % MOD if v else 1) % MOD)\nout = []\nfor l, r in queries.tolist():\n"
           "    if zeros[r + 1] - zeros[l]:\n        out.append(0)\n    else:\n"
           "        out.append(pref[r + 1] * pow(pref[l], MOD - 2, MOD) % MOD)\nresult = out")
    big = ("g = np.random.default_rng({s})\nfactors = torch.from_numpy(g.integers({lo}, {hi}, size=20000))\n"
           "queries = torch.from_numpy(np.sort(g.integers(0, 20000, size=(20000, 2)), axis=1))")
    tests = [big.format(s=seeded(rng), lo=-10**9, hi=10**9), big.format(s=seeded(rng), lo=-2, hi=3),
             big.format(s=seeded(rng), lo=1, hi=10**9), "factors = torch.tensor([-1])\nqueries = torch.tensor([[0, 0]])"]
    return challenge("Gradient scale over a range",
                     "Function i in a chain multiplies the gradient by the integer `factors[i]`. Each row `(l, r)` of "
                     "`queries` asks for the total scale through Functions `l..r` (inclusive): the product of their "
                     "factors **modulo 10⁹ + 7**, as a number in [0, 10⁹ + 7). Return the list of answers.",
                     f"factors = {tt(f)}\nqueries = {tt(qs)}", sol, tests, "1 ≤ n, q ≤ 2·10⁴, |factor| ≤ 10⁹",
                     speed="O(n + q log MOD): prefix products, modular inverses, and a prefix count of zeros")
