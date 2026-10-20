"""3.13 Automatic mixed precision (AMP)."""

from ._common import P, challenge, floats, lit, seed_line, seeded, template, tt

M = "3.13"


def _model(rng):
    a, h, o = rng.randint(3, 8), rng.randint(4, 12), rng.randint(2, 4)
    return a, o, (f"{seed_line(rng)}\nmodel = nn.Sequential(nn.Linear({a}, {h}), nn.ReLU(), nn.Linear({h}, {o}))\n"
                  f"x = torch.randn(4, {a})")


# ================================================================= lite

@template(M, "lite")
def autocast_dtype(rng):
    a, o, setup = _model(rng)
    task = rng.choice(["out", "weights", "loss"])
    if task == "out":
        return P("Run `model(x)` inside `with torch.autocast(device_type='cpu', dtype=torch.bfloat16):` and return the "
                 "output's dtype (matmuls run in bf16 under autocast).", setup,
                 "with torch.autocast(device_type='cpu', dtype=torch.bfloat16):\n    out = model(x)\nresult = out.dtype",
                 require=["autocast"])
    if task == "weights":
        return P("Autocast casts per operation; it does **not** change the stored weights. Inside a bf16 autocast block, "
                 "run `model(x)` and return `[out.dtype, model[0].weight.dtype]`.", setup,
                 "with torch.autocast(device_type='cpu', dtype=torch.bfloat16):\n    out = model(x)\nresult = [out.dtype, model[0].weight.dtype]",
                 require=["autocast"])
    return P("Keep the loss in fp32: inside a bf16 autocast block compute `out = model(x)` and `loss = F.mse_loss(out."
             "float(), torch.zeros_like(out).float())`, and return `[out.dtype, loss.dtype]`.", setup,
             "with torch.autocast(device_type='cpu', dtype=torch.bfloat16):\n    out = model(x)\n"
             "    loss = F.mse_loss(out.float(), torch.zeros_like(out).float())\nresult = [out.dtype, loss.dtype]",
             require=["autocast", "float"])


@template(M, "lite")
def dtype_ranges(rng):
    q = rng.choice([("torch.finfo(torch.float16).max", "the largest float16 value"),
                    ("torch.finfo(torch.bfloat16).max", "the largest bfloat16 value"),
                    ("torch.finfo(torch.bfloat16).eps", "bfloat16's machine epsilon (gap above 1.0)"),
                    ("torch.finfo(torch.float16).eps", "float16's machine epsilon"),
                    ("torch.finfo(torch.float32).eps", "float32's machine epsilon"),
                    ("torch.finfo(torch.float16).tiny", "the smallest positive normal float16")])
    return P(f"Return {q[1]} with `torch.finfo`.", "", q[0], require=["finfo"], tol=1e-6)


# ================================================================= easy

@template(M, "easy")
def bf16_rounding(rng):
    vals = [round(rng.uniform(-100, 100), 4) for _ in range(rng.randint(3, 5))]
    setup = f"x = torch.tensor({lit(vals)})"
    if rng.random() < 0.5:
        return P("bfloat16 keeps only 8 significant bits. Round-trip `x` through bf16 (`x.to(torch.bfloat16).float()`) "
                 "and return the result.", setup, "x.to(torch.bfloat16).float()", require=["bfloat16"])
    return P("Return the largest absolute rounding error from converting `x` to bfloat16: `(x.bfloat16().float() - "
             "x).abs().max()`.", setup, "(x.bfloat16().float() - x).abs().max()", require=["bfloat16"])


@template(M, "easy")
def fp16_overflow(rng):
    vals = [float(rng.choice([1, 500, 60000, 70000, 1e5, 3e38, -8e4, 12.5])) for _ in range(rng.randint(3, 5))]
    vals[0] = rng.choice([70000.0, 1e5, -9e4])
    setup = f"x = torch.tensor({lit(vals)})"
    if rng.random() < 0.5:
        return P("float16 tops out at 65504. Convert `x` to float16 and return which entries became infinite "
                 "(`torch.isinf(...)`).", setup, "torch.isinf(x.half())", require=["isinf"])
    return P("bfloat16 has float32's range. Return `[torch.isinf(x.half()).sum().item(), torch.isinf(x.bfloat16())"
             ".sum().item()]`: how many entries overflow in each format.", setup,
             "[torch.isinf(x.half()).sum().item(), torch.isinf(x.bfloat16()).sum().item()]", require=["isinf"])


@template(M, "easy")
def autocast_disabled(rng):
    a, o, setup = _model(rng)
    return P("Force an fp32 region inside autocast: within a bf16 `torch.autocast('cpu', ...)` block, compute "
             "`low = model(x)`; then, inside a nested `torch.autocast('cpu', enabled=False)` block, compute `precise = "
             "model(x.float())`. Return `[low.dtype, precise.dtype]`.", setup,
             "with torch.autocast(device_type='cpu', dtype=torch.bfloat16):\n    low = model(x)\n"
             "    with torch.autocast(device_type='cpu', enabled=False):\n        precise = model(x.float())\n"
             "result = [low.dtype, precise.dtype]", require=["enabled"])


# ================================================================= medium

@template(M, "medium")
def grad_scaler_steps(rng):
    n = rng.randint(5, 8)
    bad = sorted(rng.sample(range(n), rng.randint(1, 2)))
    factors = [1e38 if i in bad else 1.0 for i in range(n)]
    S, G = rng.choice([1024.0, 65536.0, 256.0]), rng.choice([2, 3])
    setup = (f"w = nn.Parameter(torch.ones(2))\nopt = torch.optim.SGD([w], lr=0.1)\n"
             f"scaler = torch.amp.GradScaler('cpu', init_scale={S}, growth_interval={G})\n"
             f"factors = {lit(factors)}   # 1e38 makes the scaled gradients overflow")
    return P(f"Run {n} fp16-style steps with the `GradScaler`: for each factor `f`, `opt.zero_grad()`, `loss = (w * "
             f"f).sum()`, `scaler.scale(loss).backward()`, `scaler.step(opt)` (skipped when gradients are inf), "
             f"`scaler.update()` (halves the scale after an overflow, doubles it after {G} clean steps). Return "
             f"`[scaler.get_scale(), w.detach()]`.", setup,
             "for f in factors:\n    opt.zero_grad()\n    loss = (w * f).sum()\n    scaler.scale(loss).backward()\n"
             "    scaler.step(opt)\n    scaler.update()\nresult = [scaler.get_scale(), w.detach()]",
             require=["scale", "update"], tol=1e-6)


@template(M, "medium")
def bf16_training_step(rng):
    a, o, setup = _model(rng)
    return P("One bf16 mixed-precision step: under `torch.autocast('cpu', dtype=torch.bfloat16)` compute `out = "
             "model(x)` and `loss = F.mse_loss(out.float(), torch.ones(out.shape))`; outside the block call "
             "`loss.backward()`. Return `[out.dtype, loss.dtype, model[0].weight.grad.dtype, model[0].weight.dtype]`: "
             "the gradients and master weights stay fp32.", setup,
             "with torch.autocast(device_type='cpu', dtype=torch.bfloat16):\n    out = model(x)\n"
             "    loss = F.mse_loss(out.float(), torch.ones(out.shape))\nloss.backward()\n"
             "result = [out.dtype, loss.dtype, model[0].weight.grad.dtype, model[0].weight.dtype]",
             require=["autocast", "backward"])


@template(M, "medium")
def underflow_demo(rng):
    g = rng.choice([1e-8, 3e-9, 5e-8, 2e-8])
    S = rng.choice([1024, 4096, 65536])
    return P(f"Why loss scaling? A gradient of `{g}` underflows to 0 in float16. Return `[g.half().item(), "
             f"((g * {S}).half().float() / {S}).item()]`: the plain fp16 value, and the value that survives when you "
             f"scale by {S} before the cast and unscale in fp32 afterwards.", f"g = torch.tensor({g})",
             f"[g.half().item(), ((g * {S}).half().float() / {S}).item()]", require=["half"], tol=1e-3)


# ================================================================= hard

@template(M, "hard")
def scaler_unscale_clip(rng):
    a, o, setup = _model(rng)
    n, S = rng.randint(2, 4), rng.choice([128.0, 1024.0])
    setup += (f"\ny = torch.randn(4, {o})\nopt = torch.optim.AdamW(model.parameters(), lr=1e-2)\n"
              f"scaler = torch.amp.GradScaler('cpu', init_scale={S})")
    return P(f"The full AMP step with clipping, {n} times: zero_grad; `loss = F.mse_loss(model(x), y)`; "
             f"`scaler.scale(loss).backward()`; `scaler.unscale_(opt)` (**before** clipping, so you clip true "
             f"gradients); `norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)`; `scaler.step(opt)`; "
             f"`scaler.update()`. Return the list of the {n} pre-clipping norms as floats.", setup,
             f"norms = []\nfor _ in range({n}):\n    opt.zero_grad()\n    loss = F.mse_loss(model(x), y)\n"
             f"    scaler.scale(loss).backward()\n    scaler.unscale_(opt)\n"
             f"    norms.append(nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())\n    scaler.step(opt)\n"
             f"    scaler.update()\nresult = norms", require=["unscale_", "clip_grad_norm_"], tol=1e-4)


@template(M, "hard")
def precision_error_compare(rng):
    n, k, m = rng.randint(3, 6), rng.choice([16, 32, 64]), rng.randint(3, 6)
    setup = f"{seed_line(rng)}\na, b = torch.randn({n}, {k}), torch.randn({k}, {m})"
    return P(f"Measure what bf16 costs: compute `exact = a @ b` in fp32 and `low` = the same product inside a bf16 "
             f"`torch.autocast('cpu', ...)` block. Return `[low.dtype, (low.float() - exact).abs().max().item(), "
             f"((low.float() - exact).abs() / exact.abs().clamp(min=1)).max().item()]` (max absolute and relative "
             f"error).", setup,
             "exact = a @ b\nwith torch.autocast(device_type='cpu', dtype=torch.bfloat16):\n    low = a @ b\n"
             "err = (low.float() - exact).abs()\nresult = [low.dtype, err.max().item(), (err / exact.abs().clamp(min=1)).max().item()]",
             require=["autocast"], tol=1e-5)


# ================================================================= challenge

@template(M, "challenge")
def bf16_collisions(rng):
    n = rng.randint(6, 9)
    x = [rng.randint(250, 300) for _ in range(n)]
    sol = ("from collections import Counter\ndef bf16(v):\n    if v < 256:\n        return v\n    e = v.bit_length() - 8\n"
           "    q, r = divmod(v, 1 << e)\n    half = 1 << (e - 1)\n    if r > half or (r == half and q % 2):\n        q += 1\n"
           "    return q << e\nvals = x.tolist()\nby_round, by_value = Counter(bf16(v) for v in vals), Counter(vals)\n"
           "pairs = lambda c: sum(k * (k - 1) // 2 for k in c.values())\nresult = pairs(by_round) - pairs(by_value)")
    big = "x = torch.from_numpy(np.random.default_rng({s}).integers({lo}, {hi}, size=20000))"
    tests = [big.format(s=seeded(rng), lo=1, hi=10**9), big.format(s=seeded(rng), lo=250, hi=2000),
             "x = torch.arange(1, 20001)", "x = torch.tensor([257, 256, 258])"]
    return challenge("Collisions in bfloat16",
                     "Casting to bfloat16 keeps **8 significant bits**: a positive integer v below 256 is exact; "
                     "otherwise it rounds to the nearest multiple of `2^(bit_length(v) − 8)`, ties to the even "
                     "multiple. Count the pairs `i < j` with `x[i] ≠ x[j]` that become **equal** after the cast.",
                     f"x = {tt(x)}", sol, tests, "1 ≤ n ≤ 2·10⁴, 1 ≤ x ≤ 10⁹", speed="O(n): group by rounded value")


@template(M, "challenge")
def grad_scaler_trace(rng):
    n = rng.randint(6, 9)
    g = [rng.choice([1, 2, 5, 40, 300, 2000]) for _ in range(n)]
    k0, G = rng.randint(4, 8), rng.randint(2, 3)
    sol = ("from fractions import Fraction\nLIMIT = 65504\nk, tracker, skipped = k0, 0, 0\nfor v in grads.tolist():\n"
           "    if v * Fraction(2) ** k > LIMIT:\n        skipped += 1\n        k -= 1\n        tracker = 0\n    else:\n"
           "        tracker += 1\n        if tracker == G:\n            k += 1\n            tracker = 0\nresult = [skipped, k]")
    big = "grads = torch.from_numpy(np.random.default_rng({s}).integers(1, {hi}, size=20000))\nk0, G = {k0}, {G}"
    tests = [big.format(s=seeded(rng), hi=10**6, k0=16, G=50), big.format(s=seeded(rng), hi=100, k0=30, G=1),
             big.format(s=seeded(rng), hi=10**9, k0=0, G=2000), "grads = torch.tensor([65504])\nk0, G = 1, 1"]
    return challenge("Tracing a GradScaler",
                     "A GradScaler keeps the scale `2^k`, starting at `k = k0`. At step t the largest gradient magnitude "
                     "is `grads[t]`. If `grads[t] · 2^k > 65504` the scaled gradients overflow fp16: the step is "
                     "**skipped**, k decreases by 1 and the clean-step counter resets. Otherwise the counter increases, "
                     "and after `G` consecutive clean steps k increases by 1 (and the counter resets). Return "
                     "`[number of skipped steps, final k]`.", f"grads = {tt(g)}\nk0, G = {k0}, {G}", sol, tests,
                     "1 ≤ n ≤ 2·10⁴, grads ≤ 10⁹", speed="O(n) with exact arithmetic (k can go negative)")
