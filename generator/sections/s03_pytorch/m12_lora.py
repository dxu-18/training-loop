"""3.12 LoRA layer."""

from ._common import P, challenge, lit, seed_line, seeded, template, tt

M = "3.12"

_LORA = ("class LoRALinear(nn.Module):\n    def __init__(self, base, r=8, alpha=16):\n        super().__init__()\n"
         "        self.base = base\n        for p in self.base.parameters():\n            p.requires_grad_(False)\n"
         "        self.A = nn.Parameter(torch.empty(r, base.in_features))\n"
         "        self.B = nn.Parameter(torch.zeros(base.out_features, r))\n"
         "        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))\n        self.scale, self.merged = alpha / r, False\n"
         "    def forward(self, x):\n        out = self.base(x)\n        if not self.merged:\n"
         "            out = out + (x @ self.A.T @ self.B.T) * self.scale\n        return out\n"
         "    @torch.no_grad()\n    def merge(self):\n        self.base.weight += (self.B @ self.A) * self.scale\n"
         "        self.merged = True\n")


# ================================================================= lite

@template(M, "lite")
def lora_param_count(rng):
    din, dout, r = rng.choice([512, 768, 1024, 4096]), rng.choice([512, 768, 1024, 4096]), rng.choice([4, 8, 16, 64])
    setup = f"d_in, d_out, r = {din}, {dout}, {r}"
    if rng.random() < 0.5:
        return P(f"LoRA adds `A` (r × d_in) and `B` (d_out × r) next to a frozen d_out × d_in weight. How many trainable "
                 f"parameters does one adapter add for d_in={din}, d_out={dout}, r={r}?", setup, "r * (d_in + d_out)")
    return P(f"What **percentage** of the base layer's weight count (`d_in · d_out`, ignore the bias) does a rank-{r} "
             f"adapter add? (`100 · r (d_in + d_out) / (d_in · d_out)`)", setup, "100 * r * (d_in + d_out) / (d_in * d_out)",
             tol=1e-9)


@template(M, "lite")
def lora_forward_formula(rng):
    i, o, r = rng.randint(3, 6), rng.randint(2, 5), rng.choice([1, 2])
    alpha = rng.choice([2, 4, 8, 16])
    setup = (f"{seed_line(rng)}\nW, b = torch.randn({o}, {i}), torch.randn({o})   # frozen base layer\n"
             f"A, B = torch.randn({r}, {i}), torch.randn({o}, {r})   # adapter\nalpha, r = {alpha}, {r}\nx = torch.randn(3, {i})")
    return P(f"Return the LoRA layer's output `x W^T + b + (alpha / r) · x A^T B^T`.", setup,
             "x @ W.T + b + (alpha / r) * (x @ A.T @ B.T)", tol=1e-4)


# ================================================================= easy

@template(M, "easy")
def freeze_base(rng):
    d, h = rng.choice([16, 32, 64]), rng.choice([32, 64, 128])
    r = rng.choice([2, 4, 8])
    setup = (f"model = nn.Sequential(nn.Linear({d}, {h}), nn.ReLU(), nn.Linear({h}, {d}))\n"
             f"adapters = [nn.Parameter(torch.zeros({r}, {d})), nn.Parameter(torch.zeros({h}, {r})),\n"
             f"            nn.Parameter(torch.zeros({r}, {h})), nn.Parameter(torch.zeros({d}, {r}))]")
    return P("Freeze every parameter of the pretrained `model` (`model.requires_grad_(False)` or a loop), then return "
             "how many parameters are still trainable among `model.parameters()` and `adapters` together.", setup,
             "model.requires_grad_(False)\nresult = sum(p.numel() for p in list(model.parameters()) + adapters if p.requires_grad)",
             post="(_, any(p.requires_grad for p in model.parameters()))", require=["requires_grad|requires_grad_"])


@template(M, "easy")
def merge_weights(rng):
    i, o, r = rng.randint(3, 6), rng.randint(2, 5), rng.choice([1, 2])
    alpha = rng.choice([2, 4, 8])
    setup = (f"{seed_line(rng)}\nW = torch.randn({o}, {i})\nA, B = torch.randn({r}, {i}), torch.randn({o}, {r})\n"
             f"scale = {alpha} / {r}")
    if rng.random() < 0.5:
        return P("For zero-overhead inference, fold the adapter into the weight: return `W + scale · (B @ A)`.", setup,
                 "W + scale * (B @ A)", tol=1e-5)
    return P("Check that merging is exact: with `x = torch.ones(2, W.shape[1])`, return whether `x @ (W + scale · B @ A).T` "
             "and the unmerged `x @ W.T + scale · x @ A.T @ B.T` agree (`torch.allclose` with `atol=1e-4`).", setup,
             "x = torch.ones(2, W.shape[1])\nresult = torch.allclose(x @ (W + scale * B @ A).T, x @ W.T + scale * x @ A.T @ B.T, atol=1e-4)",
             require=["allclose|isclose|abs"])


# ================================================================= medium

@template(M, "medium")
def lora_linear_class(rng):
    i, o, r, alpha = rng.randint(4, 8), rng.randint(2, 6), rng.choice([2, 4]), rng.choice([4, 8, 16])
    s = seeded(rng)
    post = (f"torch.manual_seed({s})\n_base = nn.Linear({i}, {o})\n_m = LoRALinear(_base, r={r}, alpha={alpha})\n"
            f"with torch.no_grad():\n    _m.A.copy_(torch.randn({r}, {i}))\n    _m.B.copy_(torch.randn({o}, {r}))\n"
            f"(_m(torch.ones(2, {i})), sorted(n for n, p in _m.named_parameters() if p.requires_grad), "
            f"[p.requires_grad for p in _base.parameters()])")
    return P(f"Write `class LoRALinear(nn.Module)` wrapping a pretrained `nn.Linear`: `__init__(self, base, r, alpha)` "
             f"stores `self.base`, **freezes** its parameters, and creates `self.A = nn.Parameter(...)` of shape (r, "
             f"in_features) and `self.B` (out_features, r) initialized to **zeros** (so training starts from the base "
             f"model); `forward(x)` returns `base(x) + (x @ A.T @ B.T) * (alpha / r)`. It is checked with r={r}, "
             f"alpha={alpha} on a {i}→{o} layer.", "",
             "class LoRALinear(nn.Module):\n    def __init__(self, base, r, alpha):\n        super().__init__()\n"
             "        self.base = base\n        for p in base.parameters():\n            p.requires_grad_(False)\n"
             "        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)\n"
             "        self.B = nn.Parameter(torch.zeros(base.out_features, r))\n        self.scale = alpha / r\n"
             "    def forward(self, x):\n        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale",
             post=post, require=["Parameter"], tol=1e-4)


@template(M, "medium")
def add_lora_recursive(rng):
    d = rng.choice([4, 6, 8])
    r = rng.choice([1, 2])
    setup = (_LORA + f"\n{seed_line(rng)}\nmodel = nn.Sequential(nn.Linear({d}, {d}), nn.ReLU(),\n"
             f"                      nn.Sequential(nn.Linear({d}, {d}), nn.Tanh(), nn.Linear({d}, 2)))")
    return P(f"Write `add_lora(model, r)` that walks `model.named_children()` **recursively** and replaces every "
             f"`nn.Linear` child with `LoRALinear(child, r)` (via `setattr(model, name, ...)`; recurse into other "
             f"children), then returns the model. Apply it to `model` with r={r} and return "
             f"`[type(m).__name__ for m in model.modules()]`.", setup,
             "def add_lora(model, r):\n    for name, child in model.named_children():\n        if isinstance(child, nn.Linear):\n"
             "            setattr(model, name, LoRALinear(child, r))\n        else:\n            add_lora(child, r)\n    return model\n"
             f"add_lora(model, {r})\nresult = [type(m).__name__ for m in model.modules()]",
             post="(_, sum(p.numel() for p in model.parameters() if p.requires_grad))", require=["named_children", "setattr"])


@template(M, "medium")
def adapter_state_dict(rng):
    d = rng.choice([4, 6, 8])
    setup = (_LORA + f"\nnet = nn.Sequential(LoRALinear(nn.Linear({d}, {d}), r=2), nn.ReLU(), LoRALinear(nn.Linear({d}, 3), r=2))")
    if rng.random() < 0.5:
        return P("Save **only** the adapter: build `{name: p.detach() for name, p in net.named_parameters() if "
                 "p.requires_grad}` and return its sorted keys.", setup,
                 "adapter_sd = {n: p.detach() for n, p in net.named_parameters() if p.requires_grad}\nresult = sorted(adapter_sd)",
                 require=["requires_grad"])
    return P("What fraction of `net`'s parameters is trainable? Return `trainable / total` as a float.", setup,
             "trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)\ntotal = sum(p.numel() for p in net.parameters())\n"
             "result = trainable / total", require=["requires_grad"], tol=1e-9)


# ================================================================= hard

@template(M, "hard")
def lora_train_merge(rng):
    i, o, k = rng.randint(3, 6), rng.randint(2, 4), rng.randint(3, 6)
    lr = rng.choice([0.1, 0.05])
    setup = (_LORA + f"\n{seed_line(rng)}\nlayer = LoRALinear(nn.Linear({i}, {o}), r=2, alpha=4)\n"
             f"X, Y = torch.randn(16, {i}), torch.randn(16, {o})")
    return P(f"Fine-tune only the adapter: build `torch.optim.SGD` over the parameters that require grad (lr={lr}), take "
             f"{k} full-batch steps on the MSE between `layer(X)` and `Y`, then call `layer.merge()` and return "
             f"`layer.base.weight.detach()`.", setup,
             f"opt = torch.optim.SGD([p for p in layer.parameters() if p.requires_grad], lr={lr})\nfor _ in range({k}):\n"
             f"    loss = F.mse_loss(layer(X), Y)\n    opt.zero_grad()\n    loss.backward()\n    opt.step()\nlayer.merge()\n"
             f"result = layer.base.weight.detach()", require=["requires_grad", "merge"], tol=1e-4)


@template(M, "hard")
def lora_attn_only(rng):
    d = rng.choice([4, 8])
    key = rng.choice(["attn", "q_proj|v_proj"])
    setup = (_LORA + f"\nclass Layer(nn.Module):\n    def __init__(self, d):\n        super().__init__()\n"
             f"        self.attn = nn.ModuleDict({{'q_proj': nn.Linear(d, d), 'k_proj': nn.Linear(d, d), 'v_proj': nn.Linear(d, d)}})\n"
             f"        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Linear(2 * d, d))\n\n"
             f"model = nn.ModuleList([Layer({d}) for _ in range({rng.randint(2, 3)})])")
    if key == "attn":
        rule, cond = "whose full name (from `named_modules()`) contains `'attn'`", "'attn' in name"
    else:
        rule, cond = "whose name ends in `q_proj` or `v_proj` (PEFT's classic `target_modules`)", "name.endswith(('q_proj', 'v_proj'))"
    return P(f"Add LoRA (r=2) only to the `nn.Linear` layers {rule}. First collect the matching `(full_name, module)` "
             f"pairs, then replace each one on its parent (`model.get_submodule(parent_name)`, `setattr`). Return the "
             f"sorted full names of the layers you wrapped.", setup,
             f"targets = [name for name, m in model.named_modules() if isinstance(m, nn.Linear) and {cond}]\n"
             f"for name in targets:\n    parent, _, child = name.rpartition('.')\n    p = model.get_submodule(parent)\n"
             f"    setattr(p, child, LoRALinear(getattr(p, child), r=2))\nresult = sorted(targets)",
             post="(_, sorted(n for n, m in model.named_modules() if isinstance(m, LoRALinear)))", require=["named_modules"])


# ================================================================= challenge

@template(M, "challenge")
def lora_rank_allocation(rng):
    L = rng.randint(3, 4)
    cost = [rng.randint(1, 4) for _ in range(L)]
    gains = [sorted(rng.randint(1, 30) for _ in range(3)) for _ in range(L)]
    budget = rng.randint(10, 40)
    sol = ("ranks = (4, 8, 16)\ncs, gs = cost.tolist(), gains.tolist()\ndp = [0] * (budget + 1)\nfor c, g in zip(cs, gs):\n"
           "    new = dp[:]\n    for r, gain in zip(ranks, g):\n        w = r * c\n        for b in range(w, budget + 1):\n"
           "            v = dp[b - w] + gain\n            if v > new[b]:\n                new[b] = v\n    dp = new\nresult = dp[budget]")
    big = ("g = np.random.default_rng({s})\ncost = torch.from_numpy(g.integers(1, {c}, size={L}))\n"
           "gains = torch.from_numpy(g.integers(0, 10**6, size=({L}, 3)))\nbudget = {B}")
    tests = [big.format(s=seeded(rng), c=10, L=60, B=2000), big.format(s=seeded(rng), c=3, L=60, B=2000),
             big.format(s=seeded(rng), c=50, L=40, B=100), "cost = torch.tensor([5])\ngains = torch.tensor([[1, 2, 3]])\nbudget = 3"]
    return challenge("Spending a LoRA budget",
                     "You can give each of L layers a LoRA rank of 0, 4, 8 or 16. Rank r on layer i costs `r · cost[i]` "
                     "parameters (in thousands) and improves the eval score by `gains[i][k]` for the k-th nonzero rank "
                     "(rank 0 costs and gains nothing). The total cost may not exceed `budget`. Return the best total "
                     "gain.", f"cost = {tt(cost)}\ngains = {tt(gains)}\nbudget = {budget}", sol, tests,
                     "L ≤ 60 layers, budget ≤ 2000", speed="O(L · budget): a multiple-choice knapsack DP")


@template(M, "challenge")
def min_rank_for_tolerance(rng):
    n = rng.randint(5, 7)
    s = sorted((rng.randint(1, 20) for _ in range(n)), reverse=True)
    total = sum(v * v for v in s)
    taus = [rng.randint(0, total) for _ in range(rng.randint(3, 5))]
    sol = ("from bisect import bisect_left\nsq = [v * v for v in sv.tolist()]\nn = len(sq)\ntail = [0] * (n + 1)\n"
           "for i in range(n - 1, -1, -1):\n    tail[i] = tail[i + 1] + sq[i]\nneg = [-t for t in tail]   # increasing\n"
           "result = [bisect_left(neg, -tau) for tau in tolerances.tolist()]")
    big = ("g = np.random.default_rng({s})\nsv = torch.from_numpy(-np.sort(-g.integers(0, {hi}, size=20000)))\n"
           "tolerances = torch.from_numpy(g.integers(0, int((sv.numpy().astype(object) ** 2).sum()) // {div} + 1, size=20000))")
    tests = [big.format(s=seeded(rng), hi=10**6, div=1), big.format(s=seeded(rng), hi=10**6, div=1000),
             big.format(s=seeded(rng), hi=3, div=1), "sv = torch.tensor([7])\ntolerances = torch.tensor([0, 48, 49])"]
    return challenge("The smallest rank that's good enough",
                     "`sv` holds a weight update's singular values, largest first. Keeping the top r of them leaves a "
                     "reconstruction error of `sum(sv[i]² for i ≥ r)`. For each tolerance τ in `tolerances`, return the "
                     "**smallest r** (0 ≤ r ≤ n) whose error is at most τ.", f"sv = {tt(s)}\ntolerances = {tt(taus)}", sol,
                     tests, "1 ≤ n, q ≤ 2·10⁴, values ≤ 10⁶", speed="O((n + q) log n): suffix sums and binary search")
