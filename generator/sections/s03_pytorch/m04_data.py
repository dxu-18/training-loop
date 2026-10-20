"""3.04 Dataset & DataLoader."""

from ... import themes as T
from ._common import P, challenge, classes, floats, frows_tt, lit, seed_line, seeded, template, tt

M = "3.04"
IMPORTS = ("from torch.utils.data import (Dataset, IterableDataset, DataLoader, TensorDataset, random_split, Subset,\n"
           "                              ConcatDataset, WeightedRandomSampler, RandomSampler, SequentialSampler,\n"
           "                              SubsetRandomSampler, BatchSampler, get_worker_info)\n"
           "from torch.nn.utils.rnn import pad_sequence\n")


def D(prompt, setup, solution, **kw):
    return P(prompt, IMPORTS + setup, solution, **kw)


def _tds(rng, n=None, d=None):
    n = n or rng.randint(6, 12)
    d = d or rng.randint(2, 4)
    name, labels = classes(rng, 2, 3)
    return n, d, name, labels, (f"{seed_line(rng)}\nX = torch.randn({n}, {d})\ny = torch.randint(0, {len(labels)}, ({n},))\n"
                                f"ds = TensorDataset(X, y)")


# ================================================================= lite

@template(M, "lite")
def tensor_dataset_basics(rng):
    n, d, name, labels, setup = _tds(rng)
    desc = f"`ds = TensorDataset(X, y)` wraps {n} examples: features `X` ({n}, {d}) and {name} labels `y`"
    task = rng.choice(["len", "item", "tensors"])
    if task == "len":
        return D(f"{desc}. How many examples does `ds` have? Use `len`.", setup, "len(ds)")
    if task == "item":
        i = rng.randrange(n)
        return D(f"{desc}. Return example {i}: indexing a `TensorDataset` gives the tuple `(features, label)`.", setup,
                 f"ds[{i}]")
    return D(f"{desc}. Get the label tensor back out of the dataset through its `tensors` attribute.", setup,
             "ds.tensors[1]", require=["tensors"])


@template(M, "lite")
def dataloader_len(rng):
    n, d, name, labels, setup = _tds(rng, n=rng.randint(10, 40))
    b = rng.randint(3, 8)
    if n % b == 0:
        n += 1
        setup = setup.replace(f"randn({n - 1}", f"randn({n}").replace(f"({n - 1},)", f"({n},)")
    drop = rng.random() < 0.4
    loader = f"DataLoader(ds, batch_size={b}{', drop_last=True' if drop else ''})"
    task = rng.choice(["len", "last_shape", "first_shape"])
    if task == "len":
        return D(f"`ds` holds {n} examples. How many batches does `{loader}` produce? Build it and use `len`.", setup,
                 f"len({loader})", require=["DataLoader"])
    if task == "first_shape":
        return D(f"`ds` holds {n} examples with {d} features. Take the first batch from `{loader}` (`next(iter(...))`) and "
                 f"return `[tuple(xb.shape), tuple(yb.shape)]`.", setup,
                 f"xb, yb = next(iter({loader}))\nresult = [tuple(xb.shape), tuple(yb.shape)]", require=["DataLoader"])
    return D(f"`ds` holds {n} examples. Loop over `{loader}` and return the number of examples in the **last** batch.",
             setup, f"for xb, yb in {loader}:\n    pass\nresult = len(xb)", require=["DataLoader"])


# ================================================================= easy

@template(M, "easy")
def custom_dataset_class(rng):
    th = T.theme(rng)
    ms = rng.sample(th.measures, 2)
    n = rng.randint(4, 6)
    feats = [[float(T.value(rng, m)) for m in ms] for _ in range(n)]
    name, labels = classes(rng, 2, 3)
    ys = [rng.randrange(len(labels)) for _ in range(n)]
    cls = "".join(w.capitalize() for w in th.plural.replace("/", " ").replace("-", " ").split()[-1:]) + "Dataset"
    setup = f"features = {lit(feats)}  # {', '.join(m.label for m in ms)}\nlabels = {lit(ys)}"
    return D(f"`features` holds two measurements for each of {n} {th.plural} and `labels` their {name} ids. Write a "
             f"map-style dataset `class {cls}(Dataset)` whose `__init__(self, X, y)` stores `X` as a float32 tensor and "
             f"`y` as an int64 tensor, with `__len__` and `__getitem__(self, i)` returning `(x_i, y_i)`.", setup,
             f"class {cls}(Dataset):\n    def __init__(self, X, y):\n        self.X = torch.as_tensor(X, dtype=torch.float32)\n"
             f"        self.y = torch.as_tensor(y, dtype=torch.long)\n    def __len__(self):\n        return len(self.X)\n"
             f"    def __getitem__(self, i):\n        return self.X[i], self.y[i]",
             post=f"_ds = {cls}(features, labels)\n(len(_ds), _ds[0], _ds[len(_ds) - 1], str(_ds[1][0].dtype), str(_ds[1][1].dtype))",
             require=["Dataset"])


@template(M, "easy")
def random_split_seeded(rng):
    n = rng.choice([10, 20, 25, 40])
    s = seeded(rng)
    setup = f"ds = TensorDataset(torch.arange({n}))"
    if rng.random() < 0.5:
        a = rng.choice([0.6, 0.7, 0.8])
        return D(f"Split `ds` ({n} examples) into train/validation with fractions `[{a}, {round(1 - a, 1)}]` using "
                 f"`random_split` and a seeded generator `torch.Generator().manual_seed({s})`. Return "
                 f"`[len(train_ds), len(val_ds), val_ds.indices]`.", setup,
                 f"train_ds, val_ds = random_split(ds, [{a}, {round(1 - a, 1)}], generator=torch.Generator().manual_seed({s}))\n"
                 f"result = [len(train_ds), len(val_ds), val_ds.indices]", require=["random_split", "manual_seed"])
    k = rng.randint(2, n // 3)
    return D(f"Split `ds` ({n} examples) into three parts of sizes `[{n - 2 * k}, {k}, {k}]` with `random_split` and "
             f"`generator=torch.Generator().manual_seed({s})`. Return the `indices` of the **last** part.", setup,
             f"parts = random_split(ds, [{n - 2 * k}, {k}, {k}], generator=torch.Generator().manual_seed({s}))\n"
             f"result = parts[-1].indices", require=["random_split", "manual_seed"])


@template(M, "easy")
def subset_concat(rng):
    n1, n2 = rng.randint(4, 8), rng.randint(3, 6)
    setup = f"a = TensorDataset(torch.arange({n1}) * 10)\nb = TensorDataset(torch.arange({n2}) + 100)"
    if rng.random() < 0.5:
        i = rng.randint(n1, n1 + n2 - 1)
        return D(f"Chain `a` ({n1} examples) and `b` ({n2} examples) into one dataset with `ConcatDataset` and return "
                 f"`[len(both), both[{i}]]`.", setup, f"both = ConcatDataset([a, b])\nresult = [len(both), both[{i}]]",
                 require=["ConcatDataset"])
    k = rng.randint(2, n1 // 2)
    return D(f"Make a `Subset` of `a` with its examples at the **odd** positions below {2 * k} (indices 1, 3, ...) and "
             f"return `[len(sub), sub[{k - 1}]]`.", setup, f"sub = Subset(a, range(1, {2 * k}, 2))\nresult = [len(sub), sub[{k - 1}]]",
             require=["Subset"])


# ================================================================= medium

def _seqs(rng, n):
    lens = [rng.randint(1, 6) for _ in range(n)]
    return [([rng.randint(1, 30) for _ in range(k)], rng.randint(0, 1)) for k in lens]


def _seq_setup(seqs):
    return "seq_ds = [\n" + "".join(f"    (torch.tensor({lit(t)}), {y}),\n" for t, y in seqs) + "]"


@template(M, "medium")
def collate_pad(rng):
    seqs = _seqs(rng, rng.randint(5, 8))
    b = rng.randint(2, 4)
    return D(f"`seq_ds` is a list of `(token_ids, label)` pairs with **different lengths**, so the default collate "
             f"can't stack them. Write `collate_pad(batch)` returning `(padded, lengths, labels)`: the ids padded with 0 "
             f"to the longest in the batch (`pad_sequence(..., batch_first=True)`), their lengths, and the labels as "
             f"tensors. Return the first batch of `DataLoader(seq_ds, batch_size={b}, collate_fn=collate_pad)`.",
             _seq_setup(seqs),
             "def collate_pad(batch):\n    seqs, labels = zip(*batch)\n"
             "    return pad_sequence(seqs, batch_first=True, padding_value=0), torch.tensor([len(s) for s in seqs]), torch.tensor(labels)\n"
             f"result = next(iter(DataLoader(seq_ds, batch_size={b}, collate_fn=collate_pad)))",
             require=["collate_fn"])


@template(M, "medium")
def sampler_weights(rng):
    name, labels = classes(rng, 2, 3)
    C = len(labels)
    counts = [rng.randint(1, 3) for _ in range(C)]
    counts[0] = rng.randint(6, 10)
    y = [c for c in range(C) for _ in range(counts[c])]
    rng.shuffle(y)
    setup = f"classes = {lit(labels)}\nlabels = {tt(y)}"
    if rng.random() < 0.5:
        return D(f"`labels` holds imbalanced {name} labels. To rebalance with a `WeightedRandomSampler`, every example "
                 f"needs the weight `1 / (count of its class)`. Return those per-example weights (float) using "
                 f"`bincount` and indexing.", setup, "result = (1.0 / torch.bincount(labels).float())[labels]",
                 require=["bincount"], tol=1e-6)
    s = seeded(rng)
    n = len(y)
    return D(f"`labels` holds imbalanced {name} labels. Build a `WeightedRandomSampler` with per-example weights "
             f"`1 / class count`, `num_samples={n}`, `replacement=True` and `generator=torch.Generator().manual_seed({s})`, "
             f"draw one epoch of indices with `list(sampler)`, and return how many drawn examples belong to each class "
             f"(`bincount` with `minlength={C}`).", setup,
             f"w = (1.0 / torch.bincount(labels).float())[labels]\n"
             f"sampler = WeightedRandomSampler(w, num_samples={n}, replacement=True, generator=torch.Generator().manual_seed({s}))\n"
             f"result = torch.bincount(labels[torch.tensor(list(sampler))], minlength={C})",
             require=["WeightedRandomSampler"])


@template(M, "medium")
def batch_sampler_lists(rng):
    n, b = rng.randint(7, 13), rng.randint(2, 4)
    drop = rng.random() < 0.5
    if rng.random() < 0.6:
        return D(f"Return `list(BatchSampler(SequentialSampler(range({n})), batch_size={b}, drop_last={drop}))`: the "
                 f"lists of indices a DataLoader would fetch for each batch.", "",
                 f"list(BatchSampler(SequentialSampler(range({n})), batch_size={b}, drop_last={drop}))",
                 require=["BatchSampler"])
    s = seeded(rng)
    lo = rng.randint(0, 5)
    return D(f"A `SubsetRandomSampler` shuffles a fixed subset of indices. Return `list(...)` of one over "
             f"`range({lo}, {lo + n})` with `generator=torch.Generator().manual_seed({s})`.", "",
             f"list(SubsetRandomSampler(range({lo}, {lo + n}), generator=torch.Generator().manual_seed({s})))",
             require=["SubsetRandomSampler"])


# ================================================================= hard

@template(M, "hard")
def iterable_worker_shard(rng):
    n, bs, w = rng.randint(7, 14), rng.randint(2, 3), rng.choice([2, 3])
    return D(f"Write `class Stream(IterableDataset)` with `__init__(self, n)` whose `__iter__` yields "
             f"`torch.tensor([float(i)])` for i in `range(n)`, **split across DataLoader workers** so that no item is "
             f"duplicated: use `get_worker_info()` (None in the main process) and let worker `id` take items `id, id + "
             f"num_workers, ...`. Then return `[b.flatten().tolist() for b in DataLoader(Stream({n}), batch_size={bs}, "
             f"num_workers={w})]`.", "",
             "class Stream(IterableDataset):\n    def __init__(self, n):\n        self.n = n\n    def __iter__(self):\n"
             "        info = get_worker_info()\n        start, step = (0, 1) if info is None else (info.id, info.num_workers)\n"
             "        for i in range(start, self.n, step):\n            yield torch.tensor([float(i)])\n"
             f"result = [b.flatten().tolist() for b in DataLoader(Stream({n}), batch_size={bs}, num_workers={w})]",
             post=f"(_, [b.flatten().tolist() for b in DataLoader(Stream({n + 3}), batch_size={bs})])",
             require=["get_worker_info"])


@template(M, "hard")
def collate_attention_mask(rng):
    seqs = _seqs(rng, rng.randint(5, 7))
    b = rng.randint(3, 4)
    return D(f"`seq_ds` holds `(token_ids, label)` pairs of different lengths. Write a `collate_fn` that returns a dict "
             f"with `input_ids` (padded with 0 to the longest in the batch), `attention_mask` (int64, 1 on real tokens, 0 "
             f"on padding) and `labels`. Return the first batch of `DataLoader(seq_ds, batch_size={b}, "
             f"collate_fn=...)`.", _seq_setup(seqs),
             "def collate(batch):\n    seqs, labels = zip(*batch)\n    ids = pad_sequence(seqs, batch_first=True, padding_value=0)\n"
             "    lens = torch.tensor([len(s) for s in seqs])\n    mask = (torch.arange(ids.shape[1])[None, :] < lens[:, None]).long()\n"
             "    return {'input_ids': ids, 'attention_mask': mask, 'labels': torch.tensor(labels)}\n"
             f"result = next(iter(DataLoader(seq_ds, batch_size={b}, collate_fn=collate)))", require=["collate_fn"])


@template(M, "hard")
def train_val_loaders(rng):
    n = rng.choice([20, 30, 40, 50])
    d = rng.randint(2, 4)
    s1, s2, b = seeded(rng), seeded(rng), rng.choice([4, 5, 8])
    setup = f"{seed_line(rng)}\nds = TensorDataset(torch.randn({n}, {d}), torch.arange({n}))"
    return D(f"`ds` holds {n} examples whose labels are their ids. Split it 80/20 with `random_split(..., "
             f"generator=torch.Generator().manual_seed({s1}))`. Build `train_dl` (batch size {b}, `shuffle=True`, "
             f"`generator=torch.Generator().manual_seed({s2})`) and `val_dl` (batch size {2 * b}, no shuffling). Return "
             f"`[len(train_dl), len(val_dl), <labels of the first training batch>]`.", setup,
             f"train_ds, val_ds = random_split(ds, [0.8, 0.2], generator=torch.Generator().manual_seed({s1}))\n"
             f"train_dl = DataLoader(train_ds, batch_size={b}, shuffle=True, generator=torch.Generator().manual_seed({s2}))\n"
             f"val_dl = DataLoader(val_ds, batch_size={2 * b})\nresult = [len(train_dl), len(val_dl), next(iter(train_dl))[1]]",
             require=["random_split", "DataLoader"])


# ================================================================= challenge

@template(M, "challenge")
def dynamic_weighted_sampling(rng):
    n = rng.randint(4, 6)
    w = [rng.randint(0, 9) for _ in range(n)]
    w[0] = max(w[0], 1)
    cur, ops = list(w), []
    for _ in range(rng.randint(5, 7)):
        if rng.random() < 0.4:
            i, nw = rng.randrange(n), rng.randint(0, 9)
            if sum(cur) - cur[i] + nw == 0:
                nw = 1
            cur[i] = nw
            ops.append([0, i, nw])
        else:
            ops.append([1, rng.randrange(sum(cur)), 0])
    if not any(o[0] == 1 for o in ops):
        ops.append([1, 0, 0])
    sol = ("w = weights.tolist()\nn = len(w)\ntree = [0] * (n + 1)\nfor i, v in enumerate(w):\n    j = i + 1\n"
           "    tree[j] += v\n    k = j + (j & -j)\n    if k <= n:\n        tree[k] += tree[j]\nLOG = n.bit_length()\n"
           "out = []\nfor t, a, b in ops.tolist():\n    if t == 0:\n        d, w[a], j = b - w[a], b, a + 1\n"
           "        while j <= n:\n            tree[j] += d\n            j += j & -j\n    else:\n        pos, rem = 0, a\n"
           "        for k in range(LOG, -1, -1):\n            nxt = pos + (1 << k)\n            if nxt <= n and tree[nxt] <= rem:\n"
           "                pos, rem = nxt, rem - tree[nxt]\n        out.append(pos)\nresult = out")
    gen = ("g = np.random.default_rng({s})\nw = g.integers(0, {hi}, size={n})\nw[0] = 1\ncur, total, ops = w.tolist(), int(w.sum()), []\n"
           "for _ in range(20000):\n    if g.random() < {p}:\n        i, nw = int(g.integers(1, {n})), int(g.integers(0, {hi}))\n"
           "        total += nw - cur[i]\n        cur[i] = nw\n        ops.append((0, i, nw))\n    else:\n"
           "        ops.append((1, int(g.integers(0, total)), 0))\nweights, ops = torch.from_numpy(w), torch.tensor(ops)")
    tests = [gen.format(s=seeded(rng), n=20000, hi=100, p=0.5), gen.format(s=seeded(rng), n=20000, hi=10**9, p=0.3),
             gen.format(s=seeded(rng), n=3, hi=5, p=0.5), "weights = torch.tensor([0, 0, 7])\nops = torch.tensor([[1, 6, 0]])"]
    return challenge("A sampler whose weights change",
                     "A weighted sampler picks item i with probability `w[i] / sum(w)`. A draw is given as an integer "
                     "`u` in `[0, sum(w))`: it picks the **smallest i with `w[0] + ... + w[i] > u`**. Process the rows of "
                     "`ops` in order: `(0, i, v)` sets `w[i] = v`; `(1, u, 0)` draws with the current weights (u is "
                     "always valid). Return the list of drawn indices.", f"weights = {tt(w)}\nops = {tt(ops)}", sol, tests,
                     "1 ≤ n ≤ 2·10⁴ items, q ≤ 2·10⁴ ops, 0 ≤ w ≤ 10⁹",
                     speed="O((n + q) log n) with a Fenwick (binary indexed) tree")


@template(M, "challenge")
def bucketed_padding_cost(rng):
    n = rng.randint(5, 7)
    lens = [rng.randint(1, 20) for _ in range(n)]
    B, C = rng.randint(2, 3), rng.randint(1, 15)
    sol = ("L = sorted(lengths.tolist())\nn = len(L)\nS = [0]\nfor v in L:\n    S.append(S[-1] + v)\nINF = float('inf')\n"
           "dp = [0] + [INF] * n\nfor i in range(1, n + 1):\n    mx, best = L[i - 1], INF\n"
           "    for j in range(max(0, i - B), i):\n        c = dp[j] + (i - j) * mx - (S[i] - S[j])\n        if c < best:\n"
           "            best = c\n    dp[i] = best + C\nresult = dp[n]")
    big = "lengths = torch.from_numpy(np.random.default_rng({s}).integers(1, {hi}, size=10000))\nB, C = {B}, {C}"
    tests = [big.format(s=seeded(rng), hi=2048, B=32, C=500), big.format(s=seeded(rng), hi=100, B=8, C=0),
             big.format(s=seeded(rng), hi=10**6, B=1, C=7), "lengths = torch.tensor([5])\nB, C = 4, 3"]
    return challenge("Bucketing to cut padding",
                     "`lengths` holds the token lengths of n sequences. You may **reorder** them freely and split them "
                     "into batches of at most `B` sequences. A batch is padded to its longest sequence, wasting "
                     "`size · max − sum` tokens, and every batch also costs `C` tokens of overhead. Return the minimum "
                     "total cost (wasted padding + C × number of batches).", f"lengths = {tt(lens)}\nB, C = {B}, {C}",
                     sol, tests, "1 ≤ n ≤ 10⁴, 1 ≤ B ≤ 32, lengths ≤ 10⁶",
                     speed="O(n log n + n·B): sort, then a DP over contiguous groups")
