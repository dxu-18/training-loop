"""3.17 FSDP & checkpointing."""

from ._common import P, challenge, seed_line, seeded, template, tt

M = "3.17"
IMPORTS = ("from torch.utils.checkpoint import checkpoint, checkpoint_sequential\nimport torch.distributed as dist\n"
           "from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy, FSDPModule\n"
           "import torch.distributed.checkpoint as dcp\n"
           "from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict, get_model_state_dict, StateDictOptions\n")
INIT = "dist.init_process_group(backend='gloo', init_method='tcp://127.0.0.1:29500', world_size=1, rank=0)"


def D(prompt, setup, solution, **kw):
    return P(prompt, IMPORTS + setup, solution, **kw)


def _block(rng):
    d = rng.choice([4, 6, 8])
    return d, (f"{seed_line(rng)}\nblock = nn.Sequential(nn.Linear({d}, {2 * d}), nn.GELU(), nn.Linear({2 * d}, {d}))\n"
               f"x = torch.randn(3, {d}, requires_grad=True)")


# ================================================================= lite

@template(M, "lite")
def checkpoint_call(rng):
    d, setup = _block(rng)
    if rng.random() < 0.5:
        return D("Run `block` with **activation checkpointing**: `checkpoint(block, x, use_reentrant=False)` stores no "
                 "intermediate activations and recomputes them during backward. Return the output (identical to "
                 "`block(x)`).", setup, "result = checkpoint(block, x, use_reentrant=False)", require=["checkpoint"],
                 tol=1e-6)
    return D("Backpropagate `checkpoint(block, x, use_reentrant=False).sum()` and return `x.grad`: checkpointing "
             "changes memory use, not gradients.", setup,
             "checkpoint(block, x, use_reentrant=False).sum().backward()\nresult = x.grad", require=["checkpoint"], tol=1e-6)


@template(M, "lite")
def checkpoint_forward_count(rng):
    d = rng.choice([4, 6, 8])
    use = rng.random() < 0.6
    setup = (f"class Counted(nn.Module):\n    def __init__(self):\n        super().__init__()\n"
             f"        self.lin, self.calls = nn.Linear({d}, {d}), 0\n    def forward(self, x):\n        self.calls += 1\n"
             f"        return self.lin(x).relu()\n\nblock = Counted()\nx = torch.randn(2, {d}, requires_grad=True)")
    if use:
        return D("Run `checkpoint(block, x, use_reentrant=False).sum().backward()` and return `block.calls`: how many times "
                 "`forward` ran (it is recomputed during backward).", setup,
                 "checkpoint(block, x, use_reentrant=False).sum().backward()\nresult = block.calls", require=["checkpoint"])
    k = rng.randint(2, 3)
    return D(f"Do {k} training iterations of `checkpoint(block, x, use_reentrant=False).sum().backward()` and return "
             f"`block.calls` (each iteration runs forward twice: once normally, once to recompute).", setup,
             f"for _ in range({k}):\n    checkpoint(block, x, use_reentrant=False).sum().backward()\nresult = block.calls",
             require=["checkpoint"])


# ================================================================= easy

@template(M, "easy")
def checkpointed_block_module(rng):
    d = rng.choice([4, 6, 8, 12])
    act = rng.choice(["Tanh", "GELU", "ReLU", "SiLU"])
    s = seeded(rng)
    post = (f"torch.manual_seed({s})\n_inner = nn.Sequential(nn.Linear({d}, {d}), nn.{act}())\n_ck = CheckpointedBlock(_inner)\n"
            f"_x = torch.randn(2, {d})\n_out = _ck(_x)\n_out.sum().backward()\n"
            f"(isinstance(_ck, nn.Module), _out.detach(), _inner[0].weight.grad)")
    return D(f"Write `class CheckpointedBlock(nn.Module)` that wraps a block (`__init__(self, block)` stores "
             f"`self.block`) and whose `forward(x)` returns `checkpoint(self.block, x, use_reentrant=False)`, so "
             f"`nn.Sequential(*[CheckpointedBlock(b) for b in blocks])` keeps about one block of activations in memory "
             f"at a time. (Checked on a {d}-wide Linear + {act} block.)", "",
             "class CheckpointedBlock(nn.Module):\n    def __init__(self, block):\n        super().__init__()\n"
             "        self.block = block\n    def forward(self, x):\n        return checkpoint(self.block, x, use_reentrant=False)",
             post=post, require=["checkpoint"], tol=1e-6)


@template(M, "easy")
def checkpoint_sequential_call(rng):
    d, L = rng.choice([4, 8]), rng.randint(3, 6)
    segs = rng.randint(2, L - 1)
    setup = (f"{seed_line(rng)}\nseq = nn.Sequential(*[nn.Linear({d}, {d}) for _ in range({L})])\n"
             f"x = torch.randn(2, {d}, requires_grad=True)")
    return D(f"Checkpoint a whole `nn.Sequential` of {L} layers in {segs} segments with `checkpoint_sequential(seq, "
             f"{segs}, x, use_reentrant=False)`, backpropagate the output's sum, and return `[output, x.grad]`.", setup,
             f"out = checkpoint_sequential(seq, {segs}, x, use_reentrant=False)\nout.sum().backward()\nresult = [out, x.grad]",
             require=["checkpoint_sequential"], tol=1e-5)


# ================================================================= medium

@template(M, "medium")
def fully_shard_blocks(rng):
    d, nb = rng.choice([4, 8]), rng.randint(2, 4)
    setup = (f"{INIT}\nclass Net(nn.Module):\n    def __init__(self):\n        super().__init__()\n"
             f"        self.blocks = nn.ModuleList([nn.Linear({d}, {d}) for _ in range({nb})])\n        self.head = nn.Linear({d}, 1)\n"
             f"    def forward(self, x):\n        for b in self.blocks:\n            x = b(x).relu()\n        return self.head(x)\n\nmodel = Net()")
    return D("FSDP2: shard each transformer block, **then** the root module. With `mp = MixedPrecisionPolicy("
             "param_dtype=torch.bfloat16, reduce_dtype=torch.float32)`, call `fully_shard(blk, mp_policy=mp)` for every "
             "block in `model.blocks` and then `fully_shard(model, mp_policy=mp)`. Return `[isinstance(m, FSDPModule) for "
             "m in model.blocks] + [isinstance(model, FSDPModule), isinstance(model.head, FSDPModule)]`.", setup,
             "mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)\nfor blk in model.blocks:\n"
             "    fully_shard(blk, mp_policy=mp)\nfully_shard(model, mp_policy=mp)\n"
             "result = [isinstance(m, FSDPModule) for m in model.blocks] + [isinstance(model, FSDPModule), isinstance(model.head, FSDPModule)]",
             require=["fully_shard", "MixedPrecisionPolicy"])


@template(M, "medium")
def dcp_save_load(rng):
    a, b = rng.randint(2, 6), rng.randint(1, 3)
    setup = f"{seed_line(rng)}\nmodel = nn.Linear({a}, {b})\nx = torch.ones(1, {a})"
    return D(f"Distributed checkpoints save each rank's shard into a **directory**. Save `{{'model': "
             f"model.state_dict()}}` with `dcp.save(..., checkpoint_id='ckpt')`; build `fresh = nn.Linear({a}, {b})`, "
             f"get its state dict `sd`, call `dcp.load({{'model': sd}}, checkpoint_id='ckpt')` (it fills the tensors **in "
             f"place**), then `fresh.load_state_dict(sd)`. Return `fresh(x)`.", setup,
             f"dcp.save({{'model': model.state_dict()}}, checkpoint_id='ckpt')\nfresh = nn.Linear({a}, {b})\nsd = fresh.state_dict()\n"
             f"dcp.load({{'model': sd}}, checkpoint_id='ckpt')\nfresh.load_state_dict(sd)\nresult = fresh(x)",
             require=["dcp"], tol=1e-6)


@template(M, "medium")
def full_state_dict_export(rng):
    d = rng.choice([4, 8])
    wrap = rng.choice(["compile", "ddp"])
    if wrap == "compile":
        setup, desc = f"model = torch.compile(nn.Sequential(nn.Linear({d}, {d}), nn.ReLU(), nn.Linear({d}, 2)))", \
            "`torch.compile` wrapped the model, so `model.state_dict()` keys start with `_orig_mod.`"
    else:
        setup = (f"from torch.nn.parallel import DistributedDataParallel as DDP\n{INIT}\n"
                 f"model = DDP(nn.Sequential(nn.Linear({d}, {d}), nn.ReLU(), nn.Linear({d}, 2)))")
        desc = "DDP wrapped the model, so `model.state_dict()` keys start with `module.`"
    return D(f"{desc}. Export one clean, full state dict (e.g. for a Hugging Face upload) with "
             f"`get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))` and return "
             f"`[sorted(model.state_dict()), sorted(clean)]`.", setup,
             "clean = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))\n"
             "result = [sorted(model.state_dict()), sorted(clean)]", require=["get_model_state_dict"])


# ================================================================= hard

@template(M, "hard")
def resume_full_checkpoint(rng):
    d, k, m = rng.randint(2, 4), rng.randint(2, 3), rng.randint(2, 3)
    setup = (f"{seed_line(rng)}\nX, y = torch.randn(16, {d}), torch.randn(16, 1)\n"
             f"def make():\n    model = nn.Sequential(nn.Linear({d}, 8), nn.Dropout(0.3), nn.Linear(8, 1))\n"
             f"    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)\n"
             f"    return model, opt, torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)\n"
             f"def step(model, opt, sched):\n    loss = F.mse_loss(model(X), y)\n    opt.zero_grad()\n    loss.backward()\n"
             f"    opt.step()\n    sched.step()\ntorch.manual_seed(0)\nmodel, opt, sched = make()")
    return D(f"A complete resume checkpoint. Train {k} steps with `step(...)`, then save `{{'model', 'opt', 'sched' state "
             f"dicts, 'step': {k}, 'rng': torch.get_rng_state()}}` to `ckpt.pt` (the RNG state matters: dropout draws "
             f"random masks). Rebuild with `make()` (after `torch.manual_seed(123)`, so everything differs), load all of "
             f"it back including `torch.set_rng_state`, and train {m} more steps. Return the final first-layer weight "
             f"(`model[0].weight.detach()`) of the resumed run.", setup,
             f"for _ in range({k}):\n    step(model, opt, sched)\n"
             f"torch.save({{'model': model.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(), 'step': {k},\n"
             f"            'rng': torch.get_rng_state()}}, 'ckpt.pt')\ntorch.manual_seed(123)\nmodel2, opt2, sched2 = make()\n"
             f"ck = torch.load('ckpt.pt', weights_only=True)\nmodel2.load_state_dict(ck['model'])\nopt2.load_state_dict(ck['opt'])\n"
             f"sched2.load_state_dict(ck['sched'])\ntorch.set_rng_state(ck['rng'])\nfor _ in range({m}):\n"
             f"    step(model2, opt2, sched2)\nresult = model2[0].weight.detach()", require=["set_rng_state", "load_state_dict"],
             tol=1e-5)


@template(M, "hard")
def activation_memory_estimate(rng):
    L = rng.choice([12, 16, 24, 32, 36, 48])
    a, b = rng.choice([200, 300, 500]), rng.choice([10, 20, 40])
    return D(f"Estimate activation memory for a {L}-layer model where each layer keeps `a = {a}` MB of internal "
             f"activations and its input is `b = {b}` MB. Without checkpointing all layers keep theirs: `L · a`. With "
             f"`checkpoint_sequential` in `s` equal segments, you store the `s` segment inputs plus one segment's "
             f"internals while recomputing it: `s · b + (L / s) · a`. Over the divisors `s` of L, return `[L · a, the "
             f"minimum checkpointed memory, the s that achieves it]` (smallest s on ties).", f"L, a, b = {L}, {a}, {b}",
             "best = min((s * b + L // s * a, s) for s in range(1, L + 1) if L % s == 0)\nresult = [L * a, best[0], best[1]]")


# ================================================================= challenge

@template(M, "challenge")
def checkpoint_segment_size(rng):
    n = rng.randint(6, 9)
    a = [rng.randint(1, 12) for _ in range(n)]
    sol = ("act = a.tolist()\nn = len(act)\npref = [0]\nfor v in act:\n    pref.append(pref[-1] + v)\nbest = None\n"
           "for s in range(1, n + 1):\n    stored = peak = 0\n    for i in range(0, n, s):\n        stored += act[i]\n"
           "        peak = max(peak, pref[min(i + s, n)] - pref[i])\n    cand = (stored + peak, s)\n"
           "    if best is None or cand < best:\n        best = cand\nresult = list(best)")
    big = "a = torch.from_numpy(np.random.default_rng({s}).integers(1, {hi}, size=20000))"
    tests = [big.format(s=seeded(rng), hi=10**6), big.format(s=seeded(rng), hi=3), "a = torch.ones(20000, dtype=torch.long)",
             "a = torch.tensor([9])"]
    return challenge("Choosing a checkpoint segment size",
                     "A network has n layers; `a[i]` is the activation memory of layer i. Checkpointing with segment "
                     "length s splits the layers into consecutive chunks of s (the last may be shorter). Training then "
                     "stores the **first** layer's activation of every chunk, plus, while one chunk is recomputed, all of "
                     "that chunk's activations. So the peak is `sum of chunk-first activations + the largest chunk sum`. "
                     "Return `[minimum peak, the s achieving it]` (smallest s on ties).", f"a = {tt(a)}", sol, tests,
                     "1 ≤ n ≤ 2·10⁴", speed="O(n log n): prefix sums, and only n/s chunks for each s")


@template(M, "challenge")
def fsdp_unit_padding(rng):
    n = rng.randint(4, 7)
    numel = [rng.randint(1, 20) for _ in range(n)]
    r, c = rng.choice([2, 3, 4, 8]), rng.randint(0, 6)
    sol = ("ar = np.arange(world)\nbest = np.full(world, np.inf)\nbest[0] = 0.0\npos, dp = 0, 0.0\nfor v in numel.tolist():\n"
           "    pos = (pos + v) % world\n    dp = overhead + float((best + (ar - pos) % world).min())\n"
           "    best[pos] = min(best[pos], dp)\nresult = int(dp)")
    big = "numel = torch.from_numpy(np.random.default_rng({s}).integers(1, {hi}, size=20000))\nworld, overhead = {w}, {c}"
    tests = [big.format(s=seeded(rng), hi=10**6, w=32, c=1000), big.format(s=seeded(rng), hi=10**6, w=7, c=0),
             big.format(s=seeded(rng), hi=40, w=32, c=10**6), "numel = torch.tensor([5])\nworld, overhead = 4, 2"]
    return challenge("Grouping parameters into FSDP units",
                     "FSDP flattens each unit's parameters into one buffer and shards it across `world` ranks, padding "
                     "the buffer up to a multiple of `world`. You must group the parameters (`numel` in order) into "
                     "**contiguous** units; a unit costs its padding (`(−total) mod world` elements) plus a fixed "
                     "`overhead` per unit. Return the minimum total cost.",
                     f"numel = {tt(numel)}\nworld, overhead = {r}, {c}", sol, tests,
                     "1 ≤ n ≤ 2·10⁴, world ≤ 32", speed="O(n · world): a DP that keeps the best cost per prefix-sum residue")
