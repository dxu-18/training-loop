"""1.17 File I/O.

Every run happens in a fresh temporary directory (see grader._sandbox), so setups can
write input files and `post` checks can read back what the user's code wrote.
"""

from ._common import P, challenge, lit, seeded, template

M = "1.17"

LOAD_NPZ = "(lambda d: {{k: d[k] for k in sorted(d.files)}})(np.load('{path}'))"


def _csv(header, rows, fmt="{}"):
    return "\n".join([",".join(header)] + [",".join(fmt.format(v) if v != "" else "" for v in r) for r in rows]) + "\n"


def _write(path, text):
    return f"with open('{path}', 'w') as f:\n    f.write({text!r})"


# ================================================================= lite

@template(M, "lite")
def save_and_load(rng):
    r, c = rng.randint(2, 4), rng.randint(2, 4)
    s = seeded(rng)
    name = rng.choice(["weights", "embeddings", "features", "logits"])
    data = f"{name} = np.random.default_rng({s}).normal(size=({r}, {c})).astype(np.float32)"
    if rng.random() < 0.5:
        return P(f"Save the array `{name}` to the file `'{name}.npy'` with `np.save` (the binary `.npy` format "
                 f"keeps the exact dtype and shape).", data, f"np.save('{name}.npy', {name})",
                 post=f"np.load('{name}.npy')", require=["save"])
    return P(f"A previous run saved an array to `'{name}.npy'`. **Load** it with `np.load` and return it.",
             data + f"\nnp.save('{name}.npy', {name})\ndel {name}", f"np.load('{name}.npy')", require=["load"])


@template(M, "lite")
def python_and_bytes(rng):
    n = rng.randint(3, 6)
    dt = rng.choice(["int16", "int32", "float32", "uint8"])
    vals = [rng.randint(0, 200) for _ in range(n)]
    setup = f"x = np.array({lit(vals)}, dtype=np.{dt})"
    desc, setup2, sol, req = rng.choice([
        ("convert `x` to a plain nested Python **list** (e.g. to put it in JSON) with `.tolist()`", setup,
         "x.tolist()", "tolist"),
        ("turn `x` into raw **bytes** with `.tobytes()`", setup, "x.tobytes()", "tobytes"),
        (f"rebuild the array from the raw bytes `raw` (written by `x.tobytes()`; the dtype was `{dt}`) with "
         f"`np.frombuffer`", setup + "\nraw = x.tobytes()\ndel x", f"np.frombuffer(raw, dtype=np.{dt})", "frombuffer"),
    ])
    return P(f"Serialization basics: {desc}.", setup2, sol, require=[req])


# ================================================================= easy

@template(M, "easy")
def savez_bundle(rng):
    n, d = rng.randint(3, 5), rng.randint(2, 3)
    s = seeded(rng)
    setup = (f"g = np.random.default_rng({s})\nX = g.normal(size=({n}, {d})).round(3)\n"
             f"y = g.integers(0, 2, size={n})")
    fn = rng.choice(["savez", "savez_compressed"])
    path = rng.choice(["dataset.npz", "train_split.npz", "batch_000.npz"])
    return P(f"Save the features `X` and labels `y` into **one** file `'{path}'` with `np.{fn}`, stored under the "
             f"names `features` and `labels` (pass them as keyword arguments).", setup,
             f"np.{fn}('{path}', features=X, labels=y)", post=LOAD_NPZ.format(path=path), require=[fn])


@template(M, "easy")
def load_csv(rng):
    cols = rng.sample(["step", "loss", "acc", "lr", "grad_norm", "tokens"], rng.randint(3, 4))
    n = rng.randint(3, 5)
    rows = [[rng.randint(0, 99) for _ in cols] for _ in range(n)]
    setup = _write("metrics.csv", _csv(cols, rows))
    i, j = sorted(rng.sample(range(len(cols)), 2))
    desc, sol = rng.choice([
        ("all of it as a float array (skip the header row with `skiprows=1`)",
         "np.loadtxt('metrics.csv', delimiter=',', skiprows=1)"),
        (f"only the columns `{cols[i]}` and `{cols[j]}` (`usecols=({i}, {j})`), as integers (`dtype=int`)",
         f"np.loadtxt('metrics.csv', delimiter=',', skiprows=1, usecols=({i}, {j}), dtype=int)"),
        (f"only the `{cols[i]}` column as a 1-D float array (`usecols={i}`)",
         f"np.loadtxt('metrics.csv', delimiter=',', skiprows=1, usecols={i})"),
    ])
    return P(f"`metrics.csv` is a training log with the header `{','.join(cols)}`. Load {desc} with `np.loadtxt` "
             f"(`delimiter=','`).", setup, sol, require=["loadtxt|genfromtxt"])


@template(M, "easy")
def save_csv(rng):
    n, d = rng.randint(2, 4), rng.randint(2, 3)
    cols = rng.sample(["p_cat", "p_dog", "p_bird", "score", "margin"], d)
    p = rng.randint(2, 4)
    s = seeded(rng)
    return P(f"Write the `({n}, {d})` predictions `preds` to `'preds.csv'` with `np.savetxt`: comma-separated, "
             f"{p} decimals per value (`fmt='%.{p}f'`), and the header line `{','.join(cols)}` with no `#` in front "
             f"(`comments=''`).", f"preds = np.random.default_rng({s}).random(({n}, {d}))",
             f"np.savetxt('preds.csv', preds, delimiter=',', fmt='%.{p}f', header='{','.join(cols)}', comments='')",
             post="open('preds.csv').read()", require=["savetxt"])


# ================================================================= medium

@template(M, "medium")
def csv_with_gaps(rng):
    cols = rng.sample(["loss", "acc", "lr", "wer", "bleu"], 3)
    n = rng.randint(3, 5)
    rows = [[round(rng.uniform(0, 5), 2) for _ in cols] for _ in range(n)]
    for _ in range(2):
        rows[rng.randrange(n)][rng.randrange(3)] = ""
    setup = _write("eval.csv", _csv(cols, rows))
    variant = rng.choice(["nan", "fill", "names"])
    if variant == "nan":
        return P(f"`eval.csv` (header `{','.join(cols)}`) has **empty cells**. Load it with `np.genfromtxt` "
                 f"(`delimiter=','`, `skip_header=1`); missing values become NaN.", setup,
                 "np.genfromtxt('eval.csv', delimiter=',', skip_header=1)", require=["genfromtxt"])
    if variant == "fill":
        return P(f"`eval.csv` (header `{','.join(cols)}`) has **empty cells**. Load it with `np.genfromtxt` "
                 f"(`delimiter=','`, `skip_header=1`) and fill missing values with -1 (`filling_values=-1`).",
                 setup, "np.genfromtxt('eval.csv', delimiter=',', skip_header=1, filling_values=-1)",
                 require=["genfromtxt"])
    c = rng.choice(cols)
    return P(f"`eval.csv` has a header `{','.join(cols)}` and some empty cells. Load it with `np.genfromtxt` using "
             f"the header as field names (`names=True`, `delimiter=','`), then return the `{c}` column (empty cells "
             f"are NaN).", setup, f"np.genfromtxt('eval.csv', delimiter=',', names=True)['{c}']",
             require=["genfromtxt"])


@template(M, "medium")
def raw_binary(rng):
    r, c = rng.randint(2, 4), rng.randint(2, 4)
    dt = rng.choice(["float32", "int16", "float64", "uint8"])
    s = seeded(rng)
    if rng.random() < 0.5:
        return P(f"An old tool dumped a `({r}, {c})` `{dt}` array to `'dump.bin'` with `x.tofile` (raw bytes: **no "
                 f"shape or dtype is stored**). Read it back with `np.fromfile`, passing the dtype, and restore the "
                 f"shape.", f"np.random.default_rng({s}).integers(0, 100, size=({r}, {c})).astype(np.{dt}).tofile('dump.bin')",
                 f"np.fromfile('dump.bin', dtype=np.{dt}).reshape({r}, {c})", require=["fromfile"])
    return P(f"Write the `({r}, {c})` array `x` to `'dump.bin'` as **raw bytes** with `x.tofile` (no header).",
             f"x = np.random.default_rng({s}).integers(0, 100, size=({r}, {c})).astype(np.{dt})",
             "x.tofile('dump.bin')", post=f"np.fromfile('dump.bin', dtype=np.{dt}).reshape({r}, {c})",
             require=["tofile"])


@template(M, "medium")
def memory_mapped(rng):
    R, C = rng.choice([(1000, 16), (5000, 8), (2000, 32)])
    a = rng.randint(0, R - 20)
    b = a + rng.randint(3, 10)
    return P(f"`'big.npy'` holds a `({R}, {C})` feature matrix. **Memory-map** it with `np.load(..., mmap_mode='r')` "
             f"so nothing is read until you slice, then return rows {a} through {b - 1} as a regular array "
             f"(`np.array(...)`).", f"np.save('big.npy', np.arange({R * C}, dtype=np.float32).reshape({R}, {C}))",
             f"np.array(np.load('big.npy', mmap_mode='r')[{a}:{b}])", require=["mmap_mode"])


@template(M, "medium")
def inspect_npz(rng):
    keys = rng.sample(["W1", "b1", "W2", "b2", "emb", "ln_g", "ln_b"], rng.randint(3, 5))
    setup = "np.savez('ckpt.npz', " + ", ".join(f"{k}=np.zeros(({rng.randint(1, 4)}, {rng.randint(1, 4)}))" for k in keys) + ")"
    k = rng.choice(keys)
    if rng.random() < 0.5:
        return P(f"`'ckpt.npz'` is a saved checkpoint. Open it with `with np.load('ckpt.npz') as data:` and return "
                 f"the **sorted list** of array names it contains (`data.files`).", setup,
                 "with np.load('ckpt.npz') as data:\n    result = sorted(data.files)", require=["load"])
    return P(f"`'ckpt.npz'` is a saved checkpoint. Open it with `with np.load('ckpt.npz') as data:` and return the "
             f"**shape** of the array stored under `'{k}'`.", setup,
             f"with np.load('ckpt.npz') as data:\n    result = data['{k}'].shape", require=["load"])


# ================================================================= hard

@template(M, "hard")
def save_shards(rng):
    n, d, k = rng.randint(7, 12), rng.randint(2, 3), rng.randint(2, 4)
    s = seeded(rng)
    return P(f"Shard the dataset `X` (`({n}, {d})`) for {k} data-loader workers: split it into {k} nearly equal "
             f"consecutive parts with `np.array_split` and save part i to `f'shard_{{i}}.npy'` (i = 0..{k - 1}).",
             f"X = np.random.default_rng({s}).normal(size=({n}, {d})).round(3)",
             f"for i, part in enumerate(np.array_split(X, {k})):\n    np.save(f'shard_{{i}}.npy', part)",
             post=f"[np.load(f'shard_{{i}}.npy') for i in range({k})]", require=["save"])


@template(M, "hard")
def checkpoint_roundtrip(rng):
    s = seeded(rng)
    layers = rng.randint(1, 2)
    shapes = {}
    d = rng.randint(2, 4)
    for i in range(1, layers + 1):
        shapes[f"W{i}"], shapes[f"b{i}"] = (d, d), (d,)
    setup = (f"g = np.random.default_rng({s})\nparams = {{\n" +
             "".join(f"    '{k}': g.normal(size={v}).round(3),\n" for k, v in shapes.items()) + "}")
    return P(f"`params` is a dict of a small model's weights. Save **all of them** into `'ckpt.npz'` in one call "
             f"with `np.savez_compressed` (unpack the dict with `**`), then load the file back into a new dict "
             f"`restored` mapping each name to its array.", setup,
             "np.savez_compressed('ckpt.npz', **params)\nwith np.load('ckpt.npz') as data:\n"
             "    restored = {k: data[k] for k in data.files}",
             post="{'file': " + LOAD_NPZ.format(path="ckpt.npz") +
                  ", 'restored': {k: restored[k] for k in sorted(restored)}}",
             require=["savez_compressed|savez"])


@template(M, "hard")
def csv_column_means(rng):
    cols = rng.sample(["latency_ms", "tokens", "cost_usd", "wer", "bleu", "acc"], 3)
    n = rng.randint(3, 5)
    rows = [[round(rng.uniform(0, 100), 1) for _ in cols] for _ in range(n)]
    rows[rng.randrange(n)][rng.randrange(3)] = ""
    setup = _write("runs.csv", _csv(cols, rows))
    return P(f"`runs.csv` has the header `{','.join(cols)}` and one empty cell. Load it with "
             f"`np.genfromtxt(..., delimiter=',', names=True)` and return a dict mapping **every column name** to the "
             f"column mean, ignoring the missing value (`np.nanmean`). Column names are in `data.dtype.names`.",
             setup, "data = np.genfromtxt('runs.csv', delimiter=',', names=True)\n"
                    "result = {name: float(np.nanmean(data[name])) for name in data.dtype.names}",
             require=["genfromtxt"])


@template(M, "hard")
def chunked_memmap_means(rng):
    n, d, chunk = rng.randint(7, 12), rng.randint(2, 3), rng.randint(2, 4)
    s = seeded(rng)
    return P(f"Out-of-core pattern: save the `({n}, {d})` embeddings `emb` to `'emb.npy'`, reopen the file as a "
             f"read-only **memory map** (`np.load(..., mmap_mode='r')`), and compute the mean of every chunk of "
             f"{chunk} consecutive rows without loading the whole file (the last chunk may be shorter). Return a "
             f"`(number_of_chunks, {d})` array.",
             f"emb = np.random.default_rng({s}).normal(size=({n}, {d})).round(3)",
             f"np.save('emb.npy', emb)\nmm = np.load('emb.npy', mmap_mode='r')\n"
             f"result = np.array([mm[i:i + {chunk}].mean(axis=0) for i in range(0, len(mm), {chunk})])",
             post=f"(np.load('emb.npy'), result)", require=["mmap_mode"])


# ================================================================= challenge

@template(M, "challenge")
def inversions_on_disk(rng):
    n = rng.randint(6, 10)
    x = [rng.randint(0, 20) for _ in range(n)]
    sol = ("a = np.load('ranks.npy').tolist()\n\ndef sort_count(v):\n    if len(v) <= 1:\n        return v, 0\n"
           "    mid = len(v) // 2\n    left, a1 = sort_count(v[:mid])\n    right, a2 = sort_count(v[mid:])\n"
           "    merged, i, j, inv = [], 0, 0, a1 + a2\n    while i < len(left) and j < len(right):\n"
           "        if left[i] <= right[j]:\n            merged.append(left[i])\n            i += 1\n        else:\n"
           "            merged.append(right[j])\n            j += 1\n            inv += len(left) - i\n"
           "    merged += left[i:] + right[j:]\n    return merged, inv\n\nresult = sort_count(a)[1]")
    tests = [f"np.save('ranks.npy', np.random.default_rng({seeded(rng)}).integers(0, 10**9, size=20000))",
             "np.save('ranks.npy', np.arange(20000, 0, -1))", f"np.save('ranks.npy', np.random.default_rng({seeded(rng)}).integers(0, 3, size=20000))",
             "np.save('ranks.npy', np.array([1]))"]
    return challenge("Count inversions on disk",
                     "Two rankers ordered the same documents; `'ranks.npy'` stores, for each document in ranker A's "
                     "order, its position in ranker B's list. Load the file and count the **inversions**: pairs "
                     "`i < j` with `a[i] > a[j]` (a disagreement measure related to Kendall's tau).",
                     f"np.save('ranks.npy', np.array({lit(x)}))", sol, tests, "1 ≤ n ≤ 2·10⁴",
                     speed="O(n log n), e.g. merge sort or a Fenwick tree")


@template(M, "challenge")
def infer_row_length(rng):
    c, r = rng.randint(2, 4), rng.randint(3, 5)
    rows, t = [], rng.randint(0, 5)
    for _ in range(r):
        t += rng.randint(1, 9)
        rows.append([t] + [rng.randint(0, 50) for _ in range(c - 1)])
    flat = [v for row in rows for v in row]
    sol = ("raw = np.fromfile('log.bin', dtype=np.int64)\nn = len(raw)\nresult = -1\nfor c in range(2, n + 1):\n"
           "    if n % c == 0 and np.all(np.diff(raw[::c]) > 0):\n        result = c\n        break")

    def big(s, rows_, cols_):
        return (f"g = np.random.default_rng({s})\nd = g.integers(0, 1000, size=({rows_}, {cols_}))\n"
                f"d[:, 0] = np.cumsum(g.integers(1, 5, size={rows_}))\nd.astype(np.int64).tofile('log.bin')")

    tests = [big(seeded(rng), 5040, 12), big(seeded(rng), 2520, 24), "np.arange(55440, dtype=np.int64).tofile('log.bin')",
             "np.array([5, 4], dtype=np.int64).tofile('log.bin')"]
    return challenge("How wide were the rows?",
                     "A logger dumped a matrix to `'log.bin'` with `tofile` (raw `int64`, row-major, **no shape "
                     "stored**). Column 0 of every row is a strictly increasing timestamp. Load the flat data and "
                     "return the **smallest row length c ≥ 2** that divides the number of values n and makes the "
                     "values at positions `0, c, 2c, …` strictly increasing (−1 if none does).",
                     f"np.array({lit(flat)}, dtype=np.int64).tofile('log.bin')", sol, tests, "2 ≤ n ≤ 6·10⁴",
                     speed="O(Σ n/c over divisors c), vectorized checks with np.diff")
