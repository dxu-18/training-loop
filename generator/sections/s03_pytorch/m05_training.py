"""3.05 MLP training: modules, losses, optimizers & the loop."""

from ._common import P, challenge, classes, floats, ftt, lit, seed_line, seeded, template, tt

M = "3.05"
IMPORTS = "from torch.utils.data import DataLoader, TensorDataset, random_split\n"


# ================================================================= lite

@template(M, "lite")
def linear_layer_shapes(rng):
    i, o = rng.randint(2, 64), rng.randint(1, 32)
    setup = f"lin = nn.Linear({i}, {o})"
    task = rng.choice(["weight", "count", "out", "nobias"])
    if task == "weight":
        return P(f"`lin = nn.Linear({i}, {o})`. Return the shape of its weight matrix as a tuple (PyTorch stores it as "
                 f"(out_features, in_features)).", setup, "tuple(lin.weight.shape)", require=["weight"])
    if task == "count":
        return P(f"How many parameters does `nn.Linear({i}, {o})` have (weights plus biases)? Sum `numel()` over "
                 f"`lin.parameters()`.", setup, "sum(p.numel() for p in lin.parameters())", require=["parameters"])
    if task == "out":
        b, t = rng.randint(2, 8), rng.randint(2, 5)
        return P(f"`lin = nn.Linear({i}, {o})` acts on the **last** dimension. Return the output shape for an input of "
                 f"shape ({b}, {t}, {i}) as a tuple.", setup, f"tuple(lin(torch.randn({b}, {t}, {i})).shape)")
    return P(f"Create `nn.Linear({i}, {o}, bias=False)` and return how many parameters it has.", "",
             f"sum(p.numel() for p in nn.Linear({i}, {o}, bias=False).parameters())", require=["bias"])


@template(M, "lite")
def loss_values(rng):
    n = rng.randint(3, 5)
    task = rng.choice(["mse", "ce", "bce", "l1_sum", "mse_none"])
    if task in ("mse", "l1_sum", "mse_none"):
        pred, target = floats(rng, n, -3, 3, 1), floats(rng, n, -3, 3, 1)
        setup = f"pred = {ftt(pred)}\ntarget = {ftt(target)}"
        if task == "mse":
            return P("Return the mean squared error between `pred` and `target` (`F.mse_loss` or `nn.MSELoss()`).", setup,
                     "F.mse_loss(pred, target)", require=["mse_loss|MSELoss"], tol=1e-5)
        if task == "l1_sum":
            return P("Return the **summed** absolute error between `pred` and `target`: `F.l1_loss` with "
                     "`reduction='sum'`.", setup, "F.l1_loss(pred, target, reduction='sum')", require=["reduction"], tol=1e-5)
        return P("Return the squared error of every element (no averaging): `F.mse_loss(..., reduction='none')`.", setup,
                 "F.mse_loss(pred, target, reduction='none')", require=["reduction"], tol=1e-5)
    if task == "ce":
        name, labels = classes(rng)
        logits = [floats(rng, len(labels), -2, 2, 1) for _ in range(n)]
        t = [rng.randrange(len(labels)) for _ in range(n)]
        return P(f"`logits` ({n}, {len(labels)}) are **raw** {name} scores and `targets` int64 class ids. Return the "
                 f"cross-entropy loss (`F.cross_entropy` applies the softmax itself).",
                 f"logits = {ftt(logits)}\ntargets = {tt(t)}", "F.cross_entropy(logits, targets)",
                 require=["cross_entropy|CrossEntropyLoss"], tol=1e-5)
    logits, t = floats(rng, n, -3, 3, 1), [float(rng.randint(0, 1)) for _ in range(n)]
    return P("`logits` are raw scores of a binary classifier and `targets` float 0/1 labels. Return the loss from "
             "`nn.BCEWithLogitsLoss()` (sigmoid + binary cross-entropy, numerically stable).",
             f"logits = {ftt(logits)}\ntargets = {ftt(t)}", "nn.BCEWithLogitsLoss()(logits, targets)",
             require=["BCEWithLogitsLoss|binary_cross_entropy_with_logits"], tol=1e-5)


@template(M, "lite")
def sequential_build(rng):
    a, h, o = rng.randint(3, 20), rng.randint(8, 64), rng.randint(1, 5)
    act = rng.choice(["ReLU", "GELU", "Tanh"])
    drop = rng.random() < 0.4
    layers = [f"nn.Linear({a}, {h})", f"nn.{act}()"] + (["nn.Dropout(0.1)"] if drop else []) + [f"nn.Linear({h}, {o})"]
    words = f"Linear({a}→{h}), {act}" + (", Dropout(0.1)" if drop else "") + f", Linear({h}→{o})"
    return P(f"Build `model` as an `nn.Sequential` of: {words}.", "", f"model = nn.Sequential({', '.join(layers)})",
             post="[type(m).__name__ for m in model] + [tuple(p.shape) for p in model.parameters()]",
             require=["Sequential"])


# ================================================================= easy

@template(M, "easy")
def module_subclass(rng):
    a, h, o = rng.randint(3, 10), rng.randint(8, 32), rng.randint(1, 4)
    name = rng.choice(["MLP", "Classifier", "Regressor", "TinyNet"])
    act = rng.choice(["relu", "gelu", "tanh"])
    return P(f"Write `class {name}(nn.Module)` whose `__init__(self, d_in, d_hidden, d_out)` calls `super().__init__()` "
             f"and creates `self.fc1 = nn.Linear(d_in, d_hidden)` and `self.fc2 = nn.Linear(d_hidden, d_out)`, and whose "
             f"`forward(x)` returns `fc2({act}(fc1(x)))` (use `F.{act}` or `torch.{act}`).", "",
             f"class {name}(nn.Module):\n    def __init__(self, d_in, d_hidden, d_out):\n        super().__init__()\n"
             f"        self.fc1 = nn.Linear(d_in, d_hidden)\n        self.fc2 = nn.Linear(d_hidden, d_out)\n"
             f"    def forward(self, x):\n        return self.fc2(F.{act}(self.fc1(x)))",
             post=f"_m = {name}({a}, {h}, {o})\n_ref = nn.Sequential(nn.Linear({a}, {h}), nn.Linear({h}, {o}))\n"
                  f"_m.fc1.load_state_dict(_ref[0].state_dict()); _m.fc2.load_state_dict(_ref[1].state_dict())\n"
                  f"_x = torch.randn(3, {a})\n([n for n, _ in _m.named_parameters()], _m(_x), _ref[1](F.{act}(_ref[0](_x))))",
             require=["Module", "forward"], tol=1e-5)


@template(M, "easy")
def count_params(rng):
    a, h, o = rng.randint(4, 32), rng.randint(8, 64), rng.randint(2, 10)
    setup = f"model = nn.Sequential(nn.Linear({a}, {h}), nn.ReLU(), nn.Linear({h}, {h}), nn.ReLU(), nn.Linear({h}, {o}))"
    if rng.random() < 0.5:
        return P("Return the total number of parameters in `model`.", setup, "sum(p.numel() for p in model.parameters())",
                 require=["parameters"])
    return P("Freeze the **first** Linear layer (set `requires_grad = False` on its parameters), then return the number "
             "of **trainable** parameters (those with `requires_grad`).", setup,
             "for p in model[0].parameters():\n    p.requires_grad = False\nresult = sum(p.numel() for p in model.parameters() if p.requires_grad)",
             require=["requires_grad"])


@template(M, "easy")
def optimizer_step(rng):
    vals, target = floats(rng, 3, -2, 2, 1), floats(rng, 3, -2, 2, 1)
    k = rng.randint(2, 4)
    opt = rng.choice([("SGD", "lr=0.1"), ("SGD", "lr=0.1, momentum=0.9"), ("Adam", "lr=0.1"), ("AdamW", "lr=0.1, weight_decay=0.1")])
    return P(f"`w` is an `nn.Parameter` and the loss is `((w - target) ** 2).sum()`. Create "
             f"`torch.optim.{opt[0]}([w], {opt[1]})` and take {k} optimizer steps (zero_grad, loss, backward, step). "
             f"Return `w.detach()`.", f"w = nn.Parameter({ftt(vals)})\ntarget = {ftt(target)}",
             f"opt = torch.optim.{opt[0]}([w], {opt[1]})\nfor _ in range({k}):\n    opt.zero_grad()\n"
             f"    loss = ((w - target) ** 2).sum()\n    loss.backward()\n    opt.step()\nresult = w.detach()",
             require=[opt[0], "step"], tol=1e-5)


# ================================================================= medium

@template(M, "medium")
def train_loop_fixed(rng):
    n, d = rng.randint(16, 32), rng.randint(2, 4)
    steps = rng.randint(5, 15)
    s = seeded(rng)
    if rng.random() < 0.5:
        name, labels = classes(rng, 2, 3)
        C = len(labels)
        setup = (f"torch.manual_seed({s})\nX = torch.randn({n}, {d})\ny = torch.randint(0, {C}, ({n},))\n"
                 f"model = nn.Sequential(nn.Linear({d}, 16), nn.ReLU(), nn.Linear(16, {C}))")
        return P(f"Train `model` (already initialized) to classify `X` into {C} {name} classes: full-batch, "
                 f"`torch.optim.Adam(model.parameters(), lr=0.05)`, cross-entropy loss, {steps} steps of the 5-step loop "
                 f"(forward, loss, zero_grad, backward, step). Return the **final** loss as a float (`loss.item()` from "
                 f"the last step).", setup,
                 f"opt = torch.optim.Adam(model.parameters(), lr=0.05)\nfor _ in range({steps}):\n    logits = model(X)\n"
                 f"    loss = F.cross_entropy(logits, y)\n    opt.zero_grad()\n    loss.backward()\n    opt.step()\n"
                 f"result = loss.item()", require=["backward", "step"], tol=1e-4)
    setup = (f"torch.manual_seed({s})\nX = torch.randn({n}, {d})\ny = X @ torch.randn({d}, 1) + 0.5\n"
             f"model = nn.Linear({d}, 1)")
    return P(f"Fit the linear `model` (already initialized) to `(X, y)`: full-batch, "
             f"`torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)`, MSE loss, {steps} steps of the 5-step "
             f"loop. Return the final loss as a float (from the last step).", setup,
             f"opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)\nfor _ in range({steps}):\n"
             f"    loss = F.mse_loss(model(X), y)\n    opt.zero_grad()\n    loss.backward()\n    opt.step()\nresult = loss.item()",
             require=["backward", "step"], tol=1e-4)


@template(M, "medium")
def lr_schedule_values(rng):
    base = rng.choice([0.1, 0.5, 1.0, 0.01])
    n = rng.randint(5, 9)
    sched = rng.choice([("StepLR", f"step_size={rng.randint(2, 3)}, gamma=0.5"),
                        ("MultiStepLR", f"milestones=[{rng.randint(1, 2)}, {rng.randint(3, 5)}], gamma=0.1"),
                        ("ExponentialLR", f"gamma={rng.choice([0.9, 0.5, 0.8])}"),
                        ("CosineAnnealingLR", f"T_max={n - 1}"),
                        ("LinearLR", f"start_factor=0.25, total_iters={rng.randint(2, 4)}")])
    return P(f"`opt` is an optimizer with base learning rate {base}. Attach "
             f"`torch.optim.lr_scheduler.{sched[0]}(opt, {sched[1]})`, then for {n} epochs record "
             f"`opt.param_groups[0]['lr']` at the **start** of the epoch and call `sched.step()` at its end. Return the "
             f"list of {n} learning rates.", f"opt = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr={base})",
             f"sched = torch.optim.lr_scheduler.{sched[0]}(opt, {sched[1]})\nlrs = []\nfor _ in range({n}):\n"
             f"    lrs.append(opt.param_groups[0]['lr'])\n    sched.step()\nresult = lrs", require=[sched[0]], tol=1e-9)


@template(M, "medium")
def param_groups_decay(rng):
    a, h, o = rng.randint(3, 8), rng.randint(4, 16), rng.randint(1, 3)
    wd = rng.choice([0.1, 0.05, 0.01])
    norm = rng.random() < 0.5
    layers = f"nn.Linear({a}, {h}), " + (f"nn.LayerNorm({h}), " if norm else "") + f"nn.ReLU(), nn.Linear({h}, {o})"
    return P(f"Weight decay should apply to weight **matrices** only, not to biases or norm weights. Build "
             f"`opt = torch.optim.AdamW(...)` for `model` with two parameter groups: parameters with `p.ndim >= 2` "
             f"(weight_decay={wd}) and the rest (weight_decay=0.0), learning rate 3e-4.",
             f"model = nn.Sequential({layers})",
             f"decay = [p for n, p in model.named_parameters() if p.ndim >= 2]\nno_decay = [p for n, p in model.named_parameters() if p.ndim < 2]\n"
             f"opt = torch.optim.AdamW([{{'params': decay, 'weight_decay': {wd}}}, {{'params': no_decay, 'weight_decay': 0.0}}], lr=3e-4)",
             post="sorted((sum(p.numel() for p in g['params']), g['weight_decay'], g['lr']) for g in opt.param_groups)",
             require=["AdamW", "ndim"])


@template(M, "medium")
def state_dict_roundtrip(rng):
    a, o = rng.randint(3, 8), rng.randint(1, 4)
    setup = f"{seed_line(rng)}\nmodel = nn.Sequential(nn.Linear({a}, 8), nn.ReLU(), nn.Linear(8, {o}))\nx = torch.randn(2, {a})"
    if rng.random() < 0.5:
        return P(f"Save `model`'s **weights** (its `state_dict`) to `model.pt` with `torch.save`, create a fresh "
                 f"`m2` with the same architecture, load the file into it (`torch.load(..., weights_only=True)` and "
                 f"`load_state_dict`), and return `m2(x)`.", setup,
                 f"torch.save(model.state_dict(), 'model.pt')\nm2 = nn.Sequential(nn.Linear({a}, 8), nn.ReLU(), nn.Linear(8, {o}))\n"
                 f"m2.load_state_dict(torch.load('model.pt', weights_only=True))\nresult = m2(x)",
                 require=["save", "load_state_dict"], tol=1e-6)
    return P("Return the list of keys of `model.state_dict()` (they name each tensor by its module path).", setup,
             "list(model.state_dict().keys())", require=["state_dict"])


# ================================================================= hard

@template(M, "hard")
def eval_accuracy(rng):
    n, d = rng.choice([30, 40, 50]), rng.randint(3, 5)
    s = seeded(rng)
    bs = rng.choice([8, 16])
    setup = (f"torch.manual_seed({s})\nX = torch.randn({n}, {d})\ny = (X[:, 0] + X[:, 1] > 0).long()\n"
             f"model = nn.Sequential(nn.Linear({d}, 32), nn.ReLU(), nn.Dropout(0.5), nn.Linear(32, 2))\n"
             f"val_dl = DataLoader(TensorDataset(X, y), batch_size={bs})")
    return P(f"Evaluate `model` on `val_dl` the right way: switch to **eval mode** (it has Dropout, which must be off), "
             f"wrap the loop in `torch.no_grad()`, predict with `argmax(dim=1)`, and accumulate correct counts over all "
             f"batches. Return the accuracy as a float.", IMPORTS + setup,
             "model.eval()\ncorrect = total = 0\nwith torch.no_grad():\n    for xb, yb in val_dl:\n"
             "        pred = model(xb).argmax(dim=1)\n        correct += (pred == yb).sum().item()\n        total += yb.numel()\n"
             "result = correct / total", require=["eval", "no_grad"])


@template(M, "hard")
def checkpoint_resume(rng):
    d = rng.randint(2, 4)
    k, m = rng.randint(2, 4), rng.randint(2, 4)
    s = seeded(rng)
    setup = (f"torch.manual_seed({s})\nX, y = torch.randn(16, {d}), torch.randn(16, 1)\n"
             f"def make():\n    torch.manual_seed(0)\n    model = nn.Linear({d}, 1)\n"
             f"    return model, torch.optim.Adam(model.parameters(), lr=0.1)\n"
             f"def train_step(model, opt):\n    loss = F.mse_loss(model(X), y)\n    opt.zero_grad()\n    loss.backward()\n"
             f"    opt.step()\nmodel, opt = make()")
    return P(f"Train `(model, opt)` for {k} steps with `train_step`, then save a resume checkpoint to `ckpt.pt`: a dict "
             f"with the model's and the optimizer's `state_dict()` and the step count. Then build a **fresh** pair with "
             f"`make()` (different weights, empty Adam state), load both state dicts from the file, and train {m} more "
             f"steps. Return the resumed model's `weight.detach()`: it must match {k + m} uninterrupted steps.", setup,
             f"for _ in range({k}):\n    train_step(model, opt)\ntorch.save({{'model': model.state_dict(), 'opt': opt.state_dict(), 'step': {k}}}, 'ckpt.pt')\n"
             f"model2, opt2 = make()\nck = torch.load('ckpt.pt', weights_only=True)\nmodel2.load_state_dict(ck['model'])\n"
             f"opt2.load_state_dict(ck['opt'])\nfor _ in range({m}):\n    train_step(model2, opt2)\nresult = model2.weight.detach()",
             require=["save", "load_state_dict"], tol=1e-5)


@template(M, "hard")
def warmup_cosine_lambda(rng):
    warm, total = rng.randint(2, 4), rng.randint(8, 12)
    base = rng.choice([1.0, 0.1, 3e-4])
    return P(f"The standard LLM schedule: linear **warmup** for {warm} steps, then cosine decay to 0 at step {total}. Write "
             f"the multiplier `f(step) = (step + 1) / {warm}` for `step < {warm}`, else "
             f"`0.5 * (1 + cos(pi * (step - {warm}) / ({total} - {warm})))`, attach it with "
             f"`torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=f)`, and record `opt.param_groups[0]['lr']` before each of "
             f"{total} `sched.step()` calls. Return the list.", f"opt = torch.optim.AdamW([nn.Parameter(torch.zeros(1))], lr={base})",
             f"def f(step):\n    if step < {warm}:\n        return (step + 1) / {warm}\n"
             f"    return 0.5 * (1 + math.cos(math.pi * (step - {warm}) / ({total} - {warm})))\n"
             f"sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=f)\nlrs = []\nfor _ in range({total}):\n"
             f"    lrs.append(opt.param_groups[0]['lr'])\n    sched.step()\nresult = lrs", require=["LambdaLR"], tol=1e-9)


# ================================================================= challenge

@template(M, "challenge")
def longest_improving_checkpoints(rng):
    n = rng.randint(7, 10)
    acc = [rng.randint(50, 80) for _ in range(n)]
    sol = ("from bisect import bisect_left\ntails = []\nfor v in val_acc.tolist():\n    i = bisect_left(tails, v)\n"
           "    if i == len(tails):\n        tails.append(v)\n    else:\n        tails[i] = v\nresult = len(tails)")
    big = "val_acc = torch.from_numpy(np.random.default_rng({s}).integers(0, {hi}, size=20000))"
    tests = [big.format(s=seeded(rng), hi=10**9), big.format(s=seeded(rng), hi=100), "val_acc = torch.arange(20000)",
             "val_acc = torch.arange(20000, 0, -1)", "val_acc = torch.tensor([7])"]
    return challenge("Longest run of improvements",
                     "`val_acc` holds the validation accuracy (×100) of n checkpoints in training order. Choose checkpoints "
                     "(not necessarily consecutive, but in order) whose accuracies **strictly increase**. Return the "
                     "largest number of checkpoints you can choose.", f"val_acc = {tt(acc)}", sol, tests, "1 ≤ n ≤ 2·10⁴",
                     speed="O(n log n): patience sorting with bisect")


@template(M, "challenge")
def tied_param_groups(rng):
    n = rng.randint(5, 7)
    grp = [rng.randrange(3) for _ in range(n)]
    size = [rng.randint(1, 50) for _ in range(3)]
    numel = [size[g] for g in grp]
    ties = []
    for i in range(n):
        for j in range(i + 1, n):
            if grp[i] == grp[j] and rng.random() < 0.6:
                ties.append([i, j])
    if not ties:
        ties = [[0, 0]]
    sol = ("n = len(numel)\nparent = list(range(n))\ndef find(a):\n    while parent[a] != a:\n"
           "        parent[a] = parent[parent[a]]\n        a = parent[a]\n    return a\nfor a, b in ties.tolist():\n"
           "    ra, rb = find(a), find(b)\n    if ra != rb:\n        parent[ra] = rb\nsizes = numel.tolist()\n"
           "roots = [i for i in range(n) if find(i) == i]\nresult = [len(roots), sum(sizes[r] for r in roots)]")
    big = ("g = np.random.default_rng({s})\nn, k = 20000, {k}\ngrp = g.integers(0, k, n)\nnumel = g.integers(1, 10**6, k)[grp]\n"
           "order = np.argsort(grp, kind='stable')\nsame = grp[order][1:] == grp[order][:-1]\npick = same & (g.random(n - 1) < {p})\n"
           "ties = torch.from_numpy(np.column_stack([order[:-1][pick], order[1:][pick]]))\nnumel = torch.from_numpy(numel)")
    tests = [big.format(s=seeded(rng), k=5000, p=0.7), big.format(s=seeded(rng), k=10, p=0.99),
             big.format(s=seeded(rng), k=19000, p=0.5), "numel = torch.tensor([4, 4])\nties = torch.tensor([[1, 0], [0, 1]])"]
    return challenge("Counting tied parameters",
                     "A model has n parameter tensors; `numel[i]` is the size of tensor i. Each row `(a, b)` of `ties` "
                     "says tensors a and b are the **same** tensor (weight tying, like an embedding shared with the LM "
                     "head); ties are transitive. Like `model.parameters()`, count every shared tensor once. Return "
                     "`[number of distinct tensors, total parameter count]`.", f"numel = {tt(numel)}\nties = {tt(ties)}",
                     sol, tests, "1 ≤ n ≤ 2·10⁴, m ≤ 2·10⁴ ties", speed="near-linear with union-find")
