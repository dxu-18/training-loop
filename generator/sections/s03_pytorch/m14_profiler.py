"""3.14 Profiler & benchmarking."""

from ._common import P, challenge, floats, lit, seed_line, seeded, template, tt

M = "3.14"
IMPORTS = "import time\nfrom torch.profiler import profile, record_function, ProfilerActivity, schedule\n"


def D(prompt, setup, solution, **kw):
    return P(prompt, IMPORTS + setup, solution, **kw)


def _model(rng, n_lin=None):
    n_lin = n_lin or rng.randint(2, 4)
    d = rng.choice([16, 32, 64])
    parts = []
    for i in range(n_lin):
        parts.append(f"nn.Linear({d}, {d})")
        if i < n_lin - 1:
            parts.append("nn.ReLU()")
    return n_lin, d, f"{seed_line(rng)}\nmodel = nn.Sequential({', '.join(parts)})\nx = torch.randn(8, {d})"


def _counts():
    return "{e.key: e.count for e in prof.key_averages()}"


# ================================================================= lite

@template(M, "lite")
def record_function_count(rng):
    n_lin, d, setup = _model(rng)
    k = rng.randint(2, 5)
    label = rng.choice(["forward_pass", "inference", "model_fwd", "eval_step"])
    return D(f"Profile {k} forward passes of `model`: open `with profile(activities=[ProfilerActivity.CPU]) as prof:` "
             f"and wrap each `model(x)` in `with record_function('{label}'):`. Return how many times `{label}` was "
             f"recorded: `{{e.key: e.count for e in prof.key_averages()}}['{label}']`.", setup,
             f"with profile(activities=[ProfilerActivity.CPU]) as prof:\n    for _ in range({k}):\n"
             f"        with record_function('{label}'):\n            model(x)\nresult = {_counts()}['{label}']",
             require=["record_function", "key_averages"])


@template(M, "lite")
def timeit_helper(rng):
    warm, iters = rng.randint(2, 5), rng.randint(5, 30)
    return D(f"Write `timeit(fn, iters={iters})` that calls `fn()` {warm} times to **warm up**, then times `iters` "
             f"calls with `time.perf_counter()` and returns the average seconds per call as a float. (On a GPU you "
             f"would also `torch.cuda.synchronize()` before reading the clock.)", "",
             f"def timeit(fn, iters={iters}):\n    for _ in range({warm}):\n        fn()\n    t0 = time.perf_counter()\n"
             f"    for _ in range(iters):\n        fn()\n    return (time.perf_counter() - t0) / iters",
             post=f"_calls = []\n_r = timeit(lambda: _calls.append(1))\n(len(_calls), isinstance(_r, float) and _r >= 0)",
             require=["perf_counter"])


# ================================================================= easy

@template(M, "easy")
def count_linear_calls(rng):
    n_lin, d, setup = _model(rng)
    k = rng.randint(1, 4)
    op = rng.choice(["aten::linear", "aten::relu"])
    return D(f"Profile {k} forward passes of `model` and return how many `{op}` calls the profiler counted "
             f"(`key_averages()`; each entry has `.key` and `.count`).", setup,
             f"with profile(activities=[ProfilerActivity.CPU]) as prof:\n    for _ in range({k}):\n        model(x)\n"
             f"result = {_counts()}['{op}']", require=["key_averages"])


@template(M, "easy")
def schedule_actions(rng):
    w, u, a = rng.randint(0, 2), rng.randint(0, 2), rng.randint(1, 3)
    rep = rng.choice([1, 2, 0])
    n = rng.randint(6, 10)
    return D(f"A profiler `schedule` decides what to do at each step. Build `schedule(wait={w}, warmup={u}, active={a}, "
             f"repeat={rep})` and return the `.name` of its action for steps 0..{n - 1} (NONE, WARMUP, RECORD or "
             f"RECORD_AND_SAVE).", "", f"s = schedule(wait={w}, warmup={u}, active={a}, repeat={rep})\n"
                                       f"result = [s(i).name for i in range({n})]", require=["schedule"])


@template(M, "easy")
def benchmark_timer_runs(rng):
    n_lin, d, setup = _model(rng, 2)
    k = rng.randint(3, 20)
    return D(f"Benchmark `model(x)` with `torch.utils.benchmark.Timer(stmt='m(x)', globals={{'m': model, 'x': x}})` and "
             f"`.timeit({k})`. Return `[measurement.number_per_run, len(measurement.times)]` (the timing itself varies "
             f"run to run, so it isn't graded).", "import torch.utils.benchmark as benchmark\n" + setup,
             f"m = benchmark.Timer(stmt='m(x)', globals={{'m': model, 'x': x}}).timeit({k})\nresult = [m.number_per_run, len(m.times)]",
             require=["Timer", "timeit"])


# ================================================================= medium

@template(M, "medium")
def split_regions(rng):
    n_lin, d, setup = _model(rng, 2)
    k = rng.randint(2, 4)
    setup += f"\ny = torch.randn(8, {d})\nopt = torch.optim.SGD(model.parameters(), lr=0.01)"
    return D(f"Split one training step's time into named regions. Inside `profile(...)`, run {k} steps, each with "
             f"`record_function` regions `'forward'` (model + loss), `'backward'` (`loss.backward()`) and `'optimizer'` "
             f"(`opt.step()` and `opt.zero_grad()`). Return a dict with the recorded count of each of the three "
             f"labels.", setup,
             f"with profile(activities=[ProfilerActivity.CPU]) as prof:\n    for _ in range({k}):\n"
             f"        with record_function('forward'):\n            loss = F.mse_loss(model(x), y)\n"
             f"        with record_function('backward'):\n            loss.backward()\n"
             f"        with record_function('optimizer'):\n            opt.step()\n            opt.zero_grad()\n"
             f"counts = {_counts()}\nresult = {{k: counts[k] for k in ('forward', 'backward', 'optimizer')}}",
             require=["record_function"])


@template(M, "medium")
def on_trace_ready_count(rng):
    n_lin, d, setup = _model(rng, 2)
    w, u, a, rep = rng.randint(0, 1), rng.randint(0, 1), rng.randint(1, 2), rng.choice([1, 2, 3])
    n = rng.randint(4, 9)
    return D(f"Profile a loop of {n} steps with `schedule(wait={w}, warmup={u}, active={a}, repeat={rep})` and an "
             f"`on_trace_ready` callback that appends 1 to a list `traces` (in real code it would export a trace). Call "
             f"`model(x)` and then `prof.step()` in every step. Return how many traces were produced.", setup,
             f"traces = []\nwith profile(activities=[ProfilerActivity.CPU], schedule=schedule(wait={w}, warmup={u}, active={a}, repeat={rep}),\n"
             f"             on_trace_ready=lambda p: traces.append(1)) as prof:\n    for _ in range({n}):\n        model(x)\n"
             f"        prof.step()\nresult = len(traces)", require=["on_trace_ready", "step"])


@template(M, "medium")
def step_time_stats(rng):
    warm = rng.randint(1, 3)
    times = [round(rng.uniform(80, 200), 1) for _ in range(warm)] + [round(rng.uniform(9, 14), 2) for _ in range(rng.randint(5, 8))]
    B, Tn = rng.choice([8, 16, 32]), rng.choice([512, 1024, 2048])
    setup = f"step_ms = {lit(times)}   # measured step times; the first {warm} are warm-up\nbatch, seq_len = {B}, {Tn}"
    return D(f"The first {warm} measurements include warm-up (caches, compilation). Drop them, take the **median** of "
             f"the rest (robust to outliers), and return `[median_ms, tokens_per_second]` where each step processes "
             f"`batch · seq_len` tokens.", setup,
             f"med = torch.tensor(step_ms[{warm}:]).median().item()\nresult = [med, batch * seq_len / (med / 1000)]",
             require=["median"], tol=1e-5)


# ================================================================= hard

@template(M, "hard")
def top_ops_by_calls(rng):
    n_lin, d, setup = _model(rng, rng.randint(3, 4))
    k = rng.randint(1, 3)
    return D(f"Profile {k} forward passes of `model`. From `prof.key_averages()`, return the names of the **3 ops with "
             f"the most calls**, ordered by count descending and then by name.", setup,
             f"with profile(activities=[ProfilerActivity.CPU]) as prof:\n    for _ in range({k}):\n        model(x)\n"
             f"evs = sorted(prof.key_averages(), key=lambda e: (-e.count, e.key))\nresult = [e.key for e in evs[:3]]",
             require=["key_averages"])


@template(M, "hard")
def scheduled_region_counts(rng):
    n_lin, d, setup = _model(rng, 2)
    w, u, a = rng.randint(1, 2), rng.randint(1, 2), rng.randint(2, 3)
    n = w + u + a + rng.randint(1, 3)
    return D(f"With `schedule(wait={w}, warmup={u}, active={a}, repeat=1)`, only the **active** steps are recorded. "
             f"Run {n} steps, each wrapping `model(x)` in `record_function('step')` and ending with `prof.step()`. "
             f"Return `[recorded count of 'step', recorded count of 'aten::linear']`.", setup,
             f"with profile(activities=[ProfilerActivity.CPU], schedule=schedule(wait={w}, warmup={u}, active={a}, repeat=1)) as prof:\n"
             f"    for _ in range({n}):\n        with record_function('step'):\n            model(x)\n        prof.step()\n"
             f"counts = {_counts()}\nresult = [counts['step'], counts['aten::linear']]", require=["schedule"])


# ================================================================= challenge

@template(M, "challenge")
def critical_path_ms(rng):
    n = rng.randint(5, 7)
    dur = [rng.randint(1, 9) for _ in range(n)]
    perm = list(range(n))
    rng.shuffle(perm)
    pairs = sorted({tuple(sorted(rng.sample(range(n), 2))) for _ in range(rng.randint(4, 7))})
    edges = [[perm[a], perm[b]] for a, b in pairs]
    sol = ("n = len(dur_us)\nd = dur_us.tolist()\nindeg, out = [0] * n, [[] for _ in range(n)]\nfor a, b in deps.tolist():\n"
           "    out[a].append(b)\n    indeg[b] += 1\nfinish = d[:]\nready = [i for i in range(n) if indeg[i] == 0]\n"
           "while ready:\n    a = ready.pop()\n    for b in out[a]:\n        finish[b] = max(finish[b], finish[a] + d[b])\n"
           "        indeg[b] -= 1\n        if indeg[b] == 0:\n            ready.append(b)\nresult = max(finish)")
    big = ("g = np.random.default_rng({s})\nn = {n}\nuv = np.sort(g.integers(0, n, size=({m}, 2)), axis=1)\n"
           "uv = uv[uv[:, 0] != uv[:, 1]]\ndeps = torch.from_numpy(g.permutation(n)[uv])\n"
           "dur_us = torch.from_numpy(g.integers(1, 10**6, size=n))")
    tests = [big.format(s=seeded(rng), n=20000, m=50000), big.format(s=seeded(rng), n=20000, m=2000),
             "dur_us = torch.ones(20000, dtype=torch.long)\ndeps = torch.tensor([[i, i + 1] for i in range(19999)])",
             "dur_us = torch.tensor([5])\ndeps = torch.zeros(0, 2, dtype=torch.long)"]
    return challenge("The critical path",
                     "A profiler trace has n ops; `dur_us[i]` is op i's duration. Each row `(a, b)` of `deps` means op b "
                     "can start only after op a finishes (the graph has no cycles). With unlimited parallel streams, "
                     "every op starts as soon as all its dependencies are done. Return the time at which the **last** "
                     "op finishes.", f"dur_us = {tt(dur)}\ndeps = {tt(edges)}", sol, tests,
                     "1 ≤ n ≤ 2·10⁴, m ≤ 5·10⁴", speed="O(n + m): longest path in topological order")


@template(M, "challenge")
def top_self_time(rng):
    n = rng.randint(5, 7)
    parent = [-1] + [rng.randrange(i) for i in range(1, n)]
    self_t = [rng.randint(0, 9) for _ in range(n)]
    total = self_t[:]
    for i in range(n - 1, 0, -1):
        total[parent[i]] += total[i]
    k = rng.randint(2, 3)
    sol = ("tot = total_us.tolist()\nself_us = tot[:]\nfor i, p in enumerate(parent.tolist()):\n    if p >= 0:\n"
           "        self_us[p] -= tot[i]\nresult = sorted(range(len(tot)), key=lambda i: (-self_us[i], i))[:k]")
    big = ("g = np.random.default_rng({s})\nn = 20000\nperm = g.permutation(n)\npar = np.full(n, -1)\n"
           "par[perm[1:]] = perm[(g.random(n - 1) * np.arange(1, n)).astype(np.int64)]\nself_us = g.integers(0, {hi}, size=n)\n"
           "tot = self_us.copy()\nfor i in perm[::-1][:-1]:\n    tot[par[i]] += tot[i]\n"
           "parent, total_us, k = torch.from_numpy(par), torch.from_numpy(tot), {k}")
    tests = [big.format(s=seeded(rng), hi=10**6, k=10), big.format(s=seeded(rng), hi=3, k=20000),
             "parent = torch.tensor([-1, 0, 0])\ntotal_us = torch.tensor([10, 4, 6])\nk = 3"]
    return challenge("Where the time really goes",
                     "A profiler's call tree has n ops; `parent[i]` is op i's caller (−1 for the root) and `total_us[i]` "
                     "its **total** time, including everything it calls. An op's **self time** is its total minus its "
                     "direct children's totals. Return the ids of the `k` ops with the largest self time, largest first "
                     "(ties: smaller id first).", f"parent = {tt(parent)}\ntotal_us = {tt(total)}\nk = {k}", sol, tests,
                     "1 ≤ k ≤ n ≤ 2·10⁴", speed="O(n log n)")
