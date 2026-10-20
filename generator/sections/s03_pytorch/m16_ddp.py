"""3.16 Distributed Data Parallel (DDP).

The browser runs a single process, so `init_process_group` works with world_size=1 (like
`torchrun --nproc_per_node=1`); multi-rank behaviour is practiced by simulating the ranks.
"""

from ._common import P, challenge, floats, lit, seed_line, seeded, template, tt

M = "3.16"
IMPORTS = ("import torch.distributed as dist\nfrom torch.nn.parallel import DistributedDataParallel as DDP\n"
           "from torch.utils.data import DataLoader, TensorDataset\nfrom torch.utils.data.distributed import DistributedSampler\n")


def D(prompt, setup, solution, **kw):
    return P(prompt, IMPORTS + setup, solution, **kw)


INIT = "dist.init_process_group(backend='gloo', init_method='tcp://127.0.0.1:29500', world_size=1, rank=0)"


# ================================================================= lite

@template(M, "lite")
def init_single_process(rng):
    task = rng.choice(["info", "destroy", "backend", "env", "env"])
    if task == "env":
        world = rng.choice([2, 4, 8, 16])
        per_node = min(world, rng.choice([2, 4, 8]))
        rank = rng.randrange(world)
        return D(f"`torchrun` tells each process who it is through environment variables. They are set below for one "
                 f"process of a {world}-process job ({per_node} per node). Read `RANK`, `LOCAL_RANK` and `WORLD_SIZE` from "
                 f"`os.environ` and return them as a list of ints.",
                 f"import os\nos.environ.update(RANK='{rank}', LOCAL_RANK='{rank % per_node}', WORLD_SIZE='{world}', "
                 f"MASTER_ADDR='127.0.0.1', MASTER_PORT='29500')",
                 "result = [int(os.environ['RANK']), int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_SIZE'])]",
                 require=["environ"])
    if task == "info":
        return D(f"Start a (single-process) process group with `{INIT}` and return `[dist.get_rank(), "
                 f"dist.get_world_size(), dist.is_initialized()]`.", "",
                 f"{INIT}\nresult = [dist.get_rank(), dist.get_world_size(), dist.is_initialized()]",
                 require=["init_process_group"])
    if task == "destroy":
        return D(f"Start a process group with `{INIT}`, then shut it down with `dist.destroy_process_group()`, and return "
                 f"`dist.is_initialized()`.", "", f"{INIT}\ndist.destroy_process_group()\nresult = dist.is_initialized()",
                 require=["destroy_process_group"])
    return D(f"Start a process group with `{INIT}` and return its backend name (`dist.get_backend()`): `gloo` is the CPU "
             f"backend, `nccl` the GPU one.", "", f"{INIT}\nresult = dist.get_backend()", require=["get_backend"])


@template(M, "lite")
def ddp_unwrap_keys(rng):
    a, b = rng.randint(2, 8), rng.randint(1, 4)
    setup = f"{INIT}\nnet = DDP(nn.Linear({a}, {b}))"
    if rng.random() < 0.5:
        return D("`net` is a DDP-wrapped `nn.Linear`. Return `list(net.state_dict())`: DDP keeps the original model as "
                 "`net.module`, so every key gains a `module.` prefix.", setup, "list(net.state_dict())",
                 require=["state_dict"])
    return D("Return the state-dict keys of the **unwrapped** model (`net.module`), the ones to save so the checkpoint "
             "loads into a plain model.", setup, "list(net.module.state_dict())", require=["module"])


# ================================================================= easy

@template(M, "easy")
def distributed_sampler_shards(rng):
    n, r = rng.randint(7, 15), rng.choice([2, 3, 4])
    k = rng.randrange(r)
    drop = rng.random() < 0.4 and n % r
    extra = ", drop_last=True" if drop else ""
    if rng.random() < 0.6:
        return D(f"Rank {k} of {r} gets its shard of a {n}-example dataset from `DistributedSampler(range({n}), "
                 f"num_replicas={r}, rank={k}, shuffle=False{extra})`. Return `list(sampler)`.", "",
                 f"result = list(DistributedSampler(range({n}), num_replicas={r}, rank={k}, shuffle=False{extra}))",
                 require=["DistributedSampler"])
    return D(f"How many examples does each of {r} ranks see per epoch for a {n}-example dataset with "
             f"`DistributedSampler(..., shuffle=False{extra})`? Return `len(sampler)` (without drop_last, the sampler pads "
             f"by repeating examples so every rank gets the same count).", "",
             f"result = len(DistributedSampler(range({n}), num_replicas={r}, rank=0, shuffle=False{extra}))",
             require=["DistributedSampler"])


@template(M, "easy")
def strip_module_prefix(rng):
    a, b = rng.randint(2, 6), rng.randint(1, 3)
    setup = (f"{seed_line(rng)}\nddp_sd = {{'module.' + k: v for k, v in nn.Linear({a}, {b}).state_dict().items()}}  # saved from a DDP model\n"
             f"model = nn.Linear({a}, {b})\nx = torch.ones(1, {a})")
    return D("`ddp_sd` was saved from a DDP-wrapped model, so its keys start with `module.`. Strip the prefix "
             "(`k.removeprefix('module.')`), load the result into the plain `model`, and return `model(x)`.", setup,
             "model.load_state_dict({k.removeprefix('module.'): v for k, v in ddp_sd.items()})\nresult = model(x)",
             require=["load_state_dict"], tol=1e-6)


@template(M, "easy")
def all_reduce_mean(rng):
    r = rng.choice([2, 3, 4, 8])
    vals = [floats(rng, 2, 0, 5, 2) for _ in range(r)]
    setup = f"per_rank = [{', '.join(f'torch.tensor({lit(v)})' for v in vals)}]   # each rank's local metric"
    if rng.random() < 0.5:
        return D(f"Simulate `dist.all_reduce(t, op=dist.ReduceOp.SUM)` followed by `t /= world_size` across the {r} ranks "
                 f"in `per_rank`: return the tensor every rank ends up with.", setup, "torch.stack(per_rank).sum(0) / len(per_rank)",
                 tol=1e-6)
    return D(f"Simulate `dist.all_reduce(t, op=dist.ReduceOp.MAX)` across the {r} ranks in `per_rank`: return the "
             f"elementwise maximum every rank ends up with.", setup, "torch.stack(per_rank).amax(0)", tol=1e-6)


# ================================================================= medium

@template(M, "medium")
def sampler_set_epoch(rng):
    n, r = rng.randint(8, 14), rng.choice([2, 3])
    k, s, e = rng.randrange(r), rng.randint(0, 100), rng.randint(1, 4)
    return D(f"With `shuffle=True`, a `DistributedSampler` shuffles with `seed + epoch`, so you must call "
             f"`sampler.set_epoch(epoch)` every epoch or each epoch repeats the same order. For "
             f"`DistributedSampler(range({n}), num_replicas={r}, rank={k}, shuffle=True, seed={s})`, return "
             f"`[list(sampler) at epoch 0, list(sampler) after set_epoch({e})]`.", "",
             f"sampler = DistributedSampler(range({n}), num_replicas={r}, rank={k}, shuffle=True, seed={s})\n"
             f"first = list(sampler)\nsampler.set_epoch({e})\nresult = [first, list(sampler)]", require=["set_epoch"])


@template(M, "medium")
def grad_accum_no_sync(rng):
    a, A = rng.randint(2, 5), rng.choice([2, 3, 4])
    setup = (f"{INIT}\n{seed_line(rng)}\nnet = DDP(nn.Linear({a}, 1))\n"
             f"micro_x, micro_y = torch.randn({A}, 4, {a}), torch.randn({A}, 4, 1)   # {A} micro-batches")
    return D(f"Gradient accumulation without redundant syncs: for each of the {A} micro-batches compute "
             f"`loss = F.mse_loss(net(xb), yb) / {A}` and call `loss.backward()`, inside `with net.no_sync():` for all "
             f"but the **last** one (which syncs). Return `net.module.weight.grad`.", setup,
             f"for i in range({A}):\n    loss = F.mse_loss(net(micro_x[i]), micro_y[i]) / {A}\n    if i < {A - 1}:\n"
             f"        with net.no_sync():\n            loss.backward()\n    else:\n        loss.backward()\n"
             f"result = net.module.weight.grad", require=["no_sync"], tol=1e-5)


@template(M, "medium")
def simulated_allreduce_step(rng):
    r, d = rng.choice([2, 4]), rng.randint(2, 4)
    lr = rng.choice([0.1, 0.5])
    setup = (f"{seed_line(rng)}\nw = torch.randn({d})          # identical on every rank\n"
             f"local_grads = [torch.randn({d}) for _ in range({r})]   # each rank's gradient from its own shard")
    return D(f"What DDP does during backward: all-reduce the {r} local gradients to their **mean**, so every rank "
             f"applies the same update and the replicas stay identical. Return the updated weights after one SGD step "
             f"with lr={lr}.", setup, f"g = torch.stack(local_grads).mean(0)\nresult = w - {lr} * g", tol=1e-6)


# ================================================================= hard

@template(M, "hard")
def ddp_training_epochs(rng):
    n, a = rng.choice([32, 48]), rng.randint(3, 6)
    epochs, bs = rng.randint(2, 3), rng.choice([8, 16])
    setup = (f"{seed_line(rng)}\nds = TensorDataset(torch.randn({n}, {a}), torch.randint(0, 2, ({n},)))\n"
             f"torch.manual_seed(0)\nmodel = nn.Linear({a}, 2)")
    return D(f"Write the DDP training loop (single process here): `{INIT}`; `rank, world = dist.get_rank(), "
             f"dist.get_world_size()`; a `DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True)`; a "
             f"`DataLoader(ds, batch_size={bs}, sampler=sampler)` (**not** shuffle=True); `net = DDP(model)`; AdamW "
             f"(lr=1e-2). Train {epochs} epochs with cross-entropy, calling `sampler.set_epoch(epoch)` first in each. "
             f"Return `net.module.weight.detach()`.", setup,
             f"{INIT}\nrank, world = dist.get_rank(), dist.get_world_size()\n"
             f"sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True)\n"
             f"dl = DataLoader(ds, batch_size={bs}, sampler=sampler)\nnet = DDP(model)\n"
             f"opt = torch.optim.AdamW(net.parameters(), lr=1e-2)\nfor epoch in range({epochs}):\n    sampler.set_epoch(epoch)\n"
             f"    for xb, yb in dl:\n        loss = F.cross_entropy(net(xb), yb)\n        opt.zero_grad()\n        loss.backward()\n"
             f"        opt.step()\nresult = net.module.weight.detach()", require=["DistributedSampler", "set_epoch"],
             tol=1e-5)


@template(M, "hard")
def rank0_checkpoint(rng):
    a, b = rng.randint(2, 6), rng.randint(1, 3)
    setup = f"{INIT}\n{seed_line(rng)}\nnet = DDP(nn.Linear({a}, {b}))"
    return D(f"Save from **rank 0 only**: `if dist.get_rank() == 0: torch.save(net.module.state_dict(), 'model.pt')`, "
             f"then `dist.barrier()` so no rank reads a half-written file. Load the file into a fresh "
             f"`nn.Linear({a}, {b})` and return `[sorted(state_dict keys), torch.equal(fresh.weight, net.module.weight)]`.",
             setup,
             f"if dist.get_rank() == 0:\n    torch.save(net.module.state_dict(), 'model.pt')\ndist.barrier()\n"
             f"fresh = nn.Linear({a}, {b})\nsd = torch.load('model.pt', weights_only=True)\nfresh.load_state_dict(sd)\n"
             f"result = [sorted(sd), torch.equal(fresh.weight, net.module.weight)]", require=["get_rank", "barrier"])


@template(M, "hard")
def shard_coverage(rng):
    n, r = rng.randint(7, 13), rng.choice([2, 3, 4, 5])
    if n % r == 0:
        n += 1
    return D(f"Check how {r} ranks cover a {n}-example dataset: gather the indices of every rank's "
             f"`DistributedSampler(range({n}), num_replicas={r}, rank=rank, shuffle=False)` and return how many times each "
             f"example is seen per epoch (`torch.bincount(..., minlength={n})`). The padding repeats a few examples.", "",
             f"idx = [i for rank in range({r}) for i in DistributedSampler(range({n}), num_replicas={r}, rank=rank, shuffle=False)]\n"
             f"result = torch.bincount(torch.tensor(idx), minlength={n})", require=["DistributedSampler"])


# ================================================================= challenge

@template(M, "challenge")
def ddp_bucket_timeline(rng):
    n = rng.randint(5, 8)
    size = [rng.randint(1, 9) for _ in range(n)]
    ready = sorted((rng.randint(0, 20) for _ in range(n)), reverse=True)
    cap = rng.randint(6, 14)
    sol = ("sz, rd = size.tolist(), ready.tolist()\nbuckets, cur, cur_sz = [], [], 0\nfor i in range(len(sz) - 1, -1, -1):\n"
           "    if cur and cur_sz + sz[i] > cap:\n        buckets.append(cur)\n        cur, cur_sz = [], 0\n"
           "    cur.append(i)\n    cur_sz += sz[i]\nbuckets.append(cur)\nend = 0\nfor b in buckets:\n"
           "    end = max(end, max(rd[i] for i in b)) + sum(sz[i] for i in b)\nresult = end")
    big = ("g = np.random.default_rng({s})\nsize = torch.from_numpy(g.integers(1, {hi}, size=20000))\n"
           "ready = torch.from_numpy(g.integers(0, 10**9, size=20000))\ncap = {cap}")
    tests = [big.format(s=seeded(rng), hi=10**6, cap=25 * 10**6), big.format(s=seeded(rng), hi=100, cap=1),
             big.format(s=seeded(rng), hi=10**4, cap=10**12), "size = torch.tensor([7])\nready = torch.tensor([3])\ncap = 2"]
    return challenge("DDP gradient buckets",
                     "DDP packs gradients into buckets and all-reduces each bucket as soon as all its gradients exist. "
                     "Walk the parameters from the **last** to the first, adding parameter i (`size[i]` MB) to the "
                     "current bucket; start a new bucket when adding it would push the bucket past `cap` (a bucket always "
                     "takes at least one parameter). Parameter i's gradient is ready at time `ready[i]`. All-reduces run "
                     "one at a time in bucket order; a bucket starts when its gradients are all ready **and** the previous "
                     "all-reduce has finished, and takes as many time units as its size. Return when the last all-reduce "
                     "finishes.", f"size = {tt(size)}\nready = {tt(ready)}\ncap = {cap}", sol, tests,
                     "1 ≤ n ≤ 2·10⁴", speed="O(n)")


@template(M, "challenge")
def world_sizes_low_waste(rng):
    n, d = rng.randint(20, 60), rng.randint(0, 3)
    sol = ("count, r = 0, 1\nwhile r <= n:\n    k = n // r\n    hi = n // k\n    lo = max(r, -(-(n - d) // k))\n"
           "    if lo <= hi:\n        count += hi - lo + 1\n    r = hi + 1\nresult = count")
    tests = ["n, d = 10**10, 0", "n, d = 10**10 - 1, 12345", "n, d = 9876543210, 10**10", "n, d = 1, 0", "n, d = 10**10, 999999"]
    return challenge("World sizes that waste little data",
                     "With `drop_last=True`, `DistributedSampler` over `n` examples and `r` ranks drops `n mod r` "
                     "examples per epoch. Count the world sizes `r` with `1 ≤ r ≤ n` that drop **at most** `d` examples.",
                     f"n, d = {n}, {d}", sol, tests, "1 ≤ n ≤ 10¹⁰, 0 ≤ d ≤ 10¹⁰",
                     speed="O(√n): group the r with the same n // r")
