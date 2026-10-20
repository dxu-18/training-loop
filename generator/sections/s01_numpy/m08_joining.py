"""1.08 Joining, splitting, repeating & padding."""

from ... import themes as T
from ._common import P, about, arr, challenge, lit, mat, pick_measure, seeded, series, template

M = "1.08"


# ================================================================= lite

@template(M, "lite")
def stack_two(rng):
    th = T.theme(rng)
    ax = rng.choice([a for a in T.AXES if a.labels[0][0].isalpha()])
    n = rng.randint(4, 7)
    m = pick_measure(rng, th, n)
    l1, l2 = ax.labels[0], ax.labels[1]
    a, b = T.ident(l1), T.ident(l2)
    va, vb = T.values(rng, m, n), T.values(rng, m, n)
    desc, sol = rng.choice([
        (f"Stack them as the **rows** of a 2-D array with shape `(2, {n})`", f"np.stack([{a}, {b}])"),
        (f"Stack them as the **columns** of a 2-D array with shape `({n}, 2)`", f"np.stack([{a}, {b}], axis=1)"),
        (f"Join them end to end into one 1-D array of length {2 * n} (`{a}` first)", f"np.concatenate([{a}, {b}])"),
    ])
    return P(f"`{a}` and `{b}` hold the {m.label} of the same {n} {th.plural} on {l1} and {l2}. {desc}.",
             f"{a} = {arr(va)}\n{b} = {arr(vb)}", sol)


@template(M, "lite")
def tile_and_repeat(rng):
    n, k = rng.randint(2, 4), rng.randint(2, 4)
    x = [rng.randint(1, 9) for _ in range(n)]
    desc, sol, req = rng.choice([
        (f"the whole array repeated {k} times end to end (`np.tile`), e.g. a pattern of augmentations",
         f"np.tile(x, {k})", "tile"),
        (f"every element repeated {k} times in place (`np.repeat`), e.g. upsampling labels to frames",
         f"np.repeat(x, {k})", "repeat"),
        (f"a `({k}, {n})` array whose rows are all copies of `x` (`np.tile` with a tuple)",
         f"np.tile(x, ({k}, 1))", "tile|repeat|broadcast_to"),
    ])
    return P(f"`x = {x}`. Build {desc}.", f"x = {arr(x)}", sol, require=[req])


# ================================================================= easy

@template(M, "easy")
def split_at_indices(rng):
    th, m, _, vals = series(rng, 8, 12)
    v = m.var
    n = len(vals)
    a = rng.randint(1, n // 2)
    b = rng.randint(a + 1, n - 1)
    if rng.random() < 0.5:
        return P(f"{about(th, m, n, v)}. Use `np.split` with a list of cut indices to split it into three "
                 f"parts: `{v}[:{a}]`, `{v}[{a}:{b}]` and `{v}[{b}:]`. Return the list of parts.",
                 f"{v} = {arr(vals)}", f"np.split({v}, [{a}, {b}])", require=["split|array_split"])
    k = rng.choice([d for d in (2, 3, 4) if n % d == 0] or [1])
    return P(f"{about(th, m, n, v)}. Use `np.split` to split it into {k} **equal** consecutive parts and "
             f"return the list of parts.", f"{v} = {arr(vals)}", f"np.split({v}, {k})", require=["split|array_split"])


@template(M, "easy")
def flip_and_roll(rng):
    n = rng.randint(5, 8)
    k = rng.randint(1, n - 1)
    r, c = rng.randint(2, 3), rng.randint(3, 4)
    desc, setup, sol = rng.choice([
        ("reverse the sequence `x` with `np.flip`", f"x = np.arange({n}) * 10", "np.flip(x)"),
        (f"shift `x` right by {k} positions, wrapping around (`np.roll`)", f"x = np.arange({n}) * 10", f"np.roll(x, {k})"),
        (f"shift `x` left by {k} positions, wrapping around (`np.roll` with a negative shift)",
         f"x = np.arange({n}) * 10", f"np.roll(x, -{k})"),
        ("mirror the image `img` left-to-right (horizontal flip augmentation, `np.fliplr`)",
         f"img = np.arange({r * c}).reshape({r}, {c})", "np.fliplr(img)"),
        ("flip the image `img` upside down (`np.flipud`)", f"img = np.arange({r * c}).reshape({r}, {c})", "np.flipud(img)"),
        ("rotate the image `img` by 90° counter-clockwise (`np.rot90`)", f"img = np.arange({r * c}).reshape({r}, {c})",
         "np.rot90(img)"),
    ])
    return P(f"Augmentation helper: {desc}.", setup, sol, no_loops=True)


@template(M, "easy")
def insert_and_delete(rng):
    th, m, _, vals = series(rng, 5, 8)
    v = m.var
    n = len(vals)
    i, j = sorted(rng.sample(range(n), 2))
    new = T.value(rng, m)
    desc, sol, req = rng.choice([
        (f"insert the value {new} at position {i} (`np.insert`)", f"np.insert({v}, {i}, {new})", "insert"),
        (f"remove the values at positions {i} and {j} (`np.delete`)", f"np.delete({v}, [{i}, {j}])", "delete"),
        (f"append the value {new} at the end (`np.append`)", f"np.append({v}, {new})", "append|concatenate|r_"),
    ])
    return P(f"{about(th, m, n, v)}. Return a new array with {desc}.", f"{v} = {arr(vals)}", sol, require=[req])


# ================================================================= medium

@template(M, "medium")
def pad_arrays(rng):
    s = seeded(rng)
    variant = rng.choice(["right", "left", "border", "reflect", "edge"])
    if variant in ("right", "left"):
        n = rng.randint(3, 7)
        L = n + rng.randint(2, 5)
        pad = rng.choice([0, 1, 50256, -100])
        ids = [rng.randint(2, 32000) for _ in range(n)]
        side = ("at the **end** (right padding)" if variant == "right"
                else "at the **front** (left padding, used for batched generation with decoder-only LMs)")
        width = f"(0, {L} - len(ids))" if variant == "right" else f"({L} - len(ids), 0)"
        return P(f"`ids` holds {n} token ids. Use `np.pad` to pad it to length {L} with the pad id "
                 f"{pad}, {side}.", f"ids = {arr(ids)}",
                 f"np.pad(ids, {width}, constant_values={pad})", require=["pad"])
    h, w, b = rng.randint(2, 4), rng.randint(2, 4), rng.randint(1, 2)
    if variant == "border":
        return P(f"`img` is a {h}×{w} grayscale image. Use `np.pad` to add a border of zeros "
                 f"{b} pixel{'s' if b > 1 else ''} wide on every side (like `padding={b}` in a conv layer).",
                 f"img = np.random.default_rng({s}).integers(0, 256, size=({h}, {w}))",
                 f"np.pad(img, {b})", require=["pad"])
    mode = "reflect" if variant == "reflect" else "edge"
    how = "a reflected border (mirror without repeating the edge)" if mode == "reflect" else "a border that repeats the edge pixels"
    return P(f"`img` is a {h + 1}×{w + 1} grayscale image. Use `np.pad` with `mode='{mode}'` to add "
             f"{how}, 1 pixel wide on every side.",
             f"img = np.random.default_rng({s}).integers(0, 256, size=({h + 1}, {w + 1}))",
             f"np.pad(img, 1, mode='{mode}')", require=["pad"])


@template(M, "medium")
def split_chunks(rng):
    th, m, _, vals = series(rng, 7, 12)
    v = m.var
    k = rng.randint(2, 4)
    if len(vals) % k == 0 or rng.random() < 0.5:
        return P(f"{about(th, m, len(vals), v)}. Use `np.array_split` to split it into {k} nearly equal "
                 f"consecutive chunks; return the **list** of chunks.", f"{v} = {arr(vals)}",
                 f"np.array_split({v}, {k})", require=["array_split|split"])
    return P(f"{about(th, m, len(vals), v)}. Use `np.array_split` to split it into {k} nearly equal "
             f"consecutive chunks, and return a **list of each chunk's mean**.", f"{v} = {arr(vals)}",
             f"[chunk.mean() for chunk in np.array_split({v}, {k})]", require=["array_split|split"])


@template(M, "medium")
def block_matrices(rng):
    n = rng.randint(2, 3)
    s = seeded(rng)
    setup = (f"g = np.random.default_rng({s})\nA = g.integers(0, 9, size=({n}, {n}))\n"
             f"B = g.integers(0, 9, size=({n}, {n}))")
    variant = rng.choice(["block", "diag", "r_"])
    if variant == "block":
        return P(f"Build the `({2 * n}, {2 * n})` block matrix `[[A, B], [B, A]]` with `np.block`.", setup,
                 "np.block([[A, B], [B, A]])", require=["block|vstack|hstack|concatenate"])
    if variant == "diag":
        return P(f"Build the `({2 * n}, {2 * n})` **block-diagonal** matrix with `A` top-left, `B` bottom-right "
                 f"and zeros elsewhere, using `np.block` and `np.zeros_like`.", setup,
                 "Z = np.zeros_like(A)\nresult = np.block([[A, Z], [Z, B]])", require=["block|vstack|hstack|concatenate"])
    a, b = rng.randint(1, 4), rng.randint(5, 9)
    return P(f"Build the 1-D array `[{', '.join(map(str, range(a, b)))}, 0, 0, 99]` in one expression with "
             f"`np.r_` (slices like `{a}:{b}` become ranges).", "", f"np.r_[{a}:{b}, 0, 0, 99]", require=["r_"])


@template(M, "medium")
def oversample_rows(rng):
    what, classes = T.labels(rng, min_classes=2)
    n, d = rng.randint(3, 5), rng.randint(2, 3)
    counts = [rng.randint(1, 3) for _ in range(n)]
    s = seeded(rng)
    if rng.random() < 0.5:
        return P(f"`X` holds {n} training rows and `reps[i]` says how many times row i should appear "
                 f"(oversampling rare {what} classes). Build the oversampled array with `np.repeat` along "
                 f"axis 0.", f"X = np.random.default_rng({s}).integers(0, 9, size=({n}, {d}))\nreps = {arr(counts)}",
                 "np.repeat(X, reps, axis=0)", require=["repeat"])
    n_tr = rng.randint(6, 12)
    return P(f"`x` holds {n_tr * 10} shuffled sample ids. Split it into **train / validation / test** parts of "
             f"70% / 20% / 10% with one `np.split` call (cut indices `int(0.7·n)` and `int(0.9·n)`).",
             f"x = np.random.default_rng({s}).permutation({n_tr * 10})",
             "n = len(x)\nresult = np.split(x, [int(0.7 * n), int(0.9 * n)])", require=["split|array_split"])


# ================================================================= hard

@template(M, "hard")
def center_pad(rng):
    h, w = rng.randint(2, 4), rng.randint(2, 4)
    Th, Tw = h + rng.randint(1, 4), w + rng.randint(1, 4)
    return P(f"`img` is an {h}×{w} image; the model needs {Th}×{Tw}. Zero-pad it with `np.pad` so it ends up "
             f"**centered**: put `(target - size) // 2` rows/columns before and the rest after (so odd "
             f"differences put the extra pixel at the bottom/right).",
             f"img = np.arange(1, {h * w + 1}).reshape({h}, {w})\ntarget_h, target_w = {Th}, {Tw}",
             "h, w = img.shape\ntop, left = (target_h - h) // 2, (target_w - w) // 2\n"
             "result = np.pad(img, ((top, target_h - h - top), (left, target_w - w - left)))",
             require=["pad"], no_loops=True)


@template(M, "hard")
def collate_batch(rng):
    B = rng.randint(2, 4)
    seqs = [[rng.randint(2, 999) for _ in range(rng.randint(1, 5))] for _ in range(B)]
    pad = rng.choice([0, -100])
    return P(f"`seqs` is a list of {B} token-id lists of different lengths. **Collate** them into a batch: "
             f"`ids`, an int array of shape `(B, L)` where L is the longest length, right-padded with {pad}, "
             f"and `mask`, the same shape, 1 for real tokens and 0 for padding.",
             f"seqs = {lit(seqs)}\npad_id = {pad}",
             "L = max(len(s) for s in seqs)\nids = np.array([np.pad(s, (0, L - len(s)), constant_values=pad_id) for s in seqs])\n"
             "mask = np.array([[1] * len(s) + [0] * (L - len(s)) for s in seqs])", answer=["ids", "mask"])


@template(M, "hard")
def pack_sequences(rng):
    docs = [[rng.randint(5, 99) for _ in range(rng.randint(2, 5))] for _ in range(rng.randint(3, 4))]
    eos = rng.choice([0, 1, 2])
    L = rng.randint(3, 5)
    return P(f"Pretraining packs documents into fixed-length rows. `docs` is a list of token-id lists. "
             f"Append the end-of-sequence id {eos} to **every** document, concatenate them all into one array, "
             f"drop the leftover tail that doesn't fill a whole row, and reshape into rows of length {L}.",
             f"docs = {lit(docs)}\neos, L = {eos}, {L}",
             "stream = np.concatenate([np.append(d, eos) for d in docs])\nn_rows = len(stream) // L\n"
             "result = stream[:n_rows * L].reshape(n_rows, L)", require=["concatenate|hstack|append|r_"])


# ================================================================= challenge

@template(M, "challenge")
def merge_segments(rng):
    n = rng.randint(6, 9)
    segs = []
    for _ in range(n):
        a = rng.randint(0, 60)
        segs.append([a, a + rng.randint(1, 12)])
    story = rng.choice([
        "A voice-activity detector and a diarization model produced speech segments `[start, end)` in "
        "seconds, as rows of `segs`, in no particular order and possibly overlapping.",
        "Several annotators marked the time ranges `[start, end)` (in frames) where an object is visible "
        "in a video; the rows of `segs` are unsorted and may overlap.",
    ])
    sol = ("pairs = sorted(map(tuple, segs.tolist()))\nmerged = []\nfor a, b in pairs:\n"
           "    if merged and a <= merged[-1][1]:\n        merged[-1][1] = max(merged[-1][1], b)\n"
           "    else:\n        merged.append([a, b])\nresult = (len(merged), sum(b - a for a, b in merged))")

    def big(s, n_, span):
        return (f"g = np.random.default_rng({s})\nstarts = g.integers(0, {span}, size={n_})\n"
                f"segs = np.stack([starts, starts + g.integers(1, 50, size={n_})], axis=1)")

    tests = [big(seeded(rng), 20000, 10**6), big(seeded(rng), 20000, 5000),
             "segs = np.array([[0, 5], [5, 9]])", "segs = np.array([[3, 4]])"]
    return challenge("Merge the segments",
                     story + " Segments that overlap **or touch** (one ends exactly where the next starts) "
                     "merge into one. Return a tuple `(number_of_merged_segments, total_covered_length)`.",
                     f"segs = {mat(segs)}", sol, tests, "1 ≤ n ≤ 2·10⁴, 0 ≤ start < end ≤ 10⁶")


@template(M, "challenge")
def roll_to_match(rng):
    n = rng.randint(6, 10)
    a = [rng.randint(0, 3) for _ in range(n)]
    k = rng.randrange(n)
    b = a[-k:] + a[:-k] if k else list(a)
    if rng.random() < 0.25:
        b[rng.randrange(n)] += 1
    sol = ("A, B = a.tolist(), b.tolist()\nn = len(A)\nif n != len(B):\n    result = -1\nelse:\n"
           "    pat, text = A, B + B\n    fail = [0] * n\n    k = 0\n    for i in range(1, n):\n"
           "        while k and pat[i] != pat[k]:\n            k = fail[k - 1]\n        if pat[i] == pat[k]:\n"
           "            k += 1\n        fail[i] = k\n    result, k = -1, 0\n    for i, ch in enumerate(text):\n"
           "        while k and ch != pat[k]:\n            k = fail[k - 1]\n        if ch == pat[k]:\n            k += 1\n"
           "        if k == n:\n            result = i - n + 1  # roll(a, s) puts a[0] at index s of b\n            break")
    def big(s, n_, hi, shift, broken=False):
        lines = [f"a = np.random.default_rng({s}).integers(0, {hi}, size={n_})", f"b = np.roll(a, {shift})"]
        if broken:
            lines.append("b[7] += 1")
        return "\n".join(lines)

    tests = [big(seeded(rng), 20000, 2, 12345), big(seeded(rng), 20000, 2, 0), big(seeded(rng), 20000, 3, 777, True),
             "a = np.zeros(20000, dtype=int)\na[-1] = 1\nb = np.roll(a, 19999)", "a = np.array([4])\nb = np.array([4])"]
    return challenge("Undo the rotation",
                     "Two microphones wrote the same circular audio buffer, but the second one started "
                     "recording at a different point. Find the **smallest shift s ≥ 0** such that "
                     "`np.roll(a, s)` equals `b` element for element, or −1 if no shift works.",
                     f"a = {arr(a)}\nb = {arr(b)}", sol, tests, "1 ≤ n ≤ 2·10⁴",
                     speed="O(n), e.g. KMP or the Z-function on b + b")
