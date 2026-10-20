"""3.11 KV cache."""

from ._common import P, challenge, lit, seed_line, seeded, template, tt

M = "3.11"

_CACHED = ("class CachedSelfAttention(nn.Module):\n    def __init__(self, d, h):\n        super().__init__()\n"
           "        self.h, self.dh = h, d // h\n        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)\n"
           "    def forward(self, x, cache=None):   # x: (B, T_new, d); cache: (k, v), each (B, H, T_past, dh)\n"
           "        B, T, C = x.shape\n        q, k, v = self.qkv(x).split(C, dim=-1)\n"
           "        q, k, v = (t.view(B, T, self.h, self.dh).transpose(1, 2) for t in (q, k, v))\n"
           "        if cache is not None:\n            k, v = torch.cat([cache[0], k], dim=2), torch.cat([cache[1], v], dim=2)\n"
           "        y = F.scaled_dot_product_attention(q, k, v, is_causal=(cache is None and T > 1))\n"
           "        return self.proj(y.transpose(1, 2).reshape(B, T, C)), (k, v)\n")


# ================================================================= lite

@template(M, "lite")
def cache_append(rng):
    B, H, Tp, D = rng.randint(1, 2), rng.choice([2, 4]), rng.randint(2, 6), rng.choice([4, 8])
    Tn = rng.choice([1, 1, 2])
    setup = (f"{seed_line(rng)}\npast_k = torch.randn({B}, {H}, {Tp}, {D})   # (B, H, T_past, D)\n"
             f"new_k = torch.randn({B}, {H}, {Tn}, {D})")
    if rng.random() < 0.5:
        return P(f"Append the new keys to the cache along the **sequence** dimension (dim 2) with `torch.cat` and return "
                 f"the updated cache.", setup, "torch.cat([past_k, new_k], dim=2)", require=["cat"])
    return P("Append `new_k` to `past_k` along the sequence dimension and return the shape of the result as a tuple.",
             setup, "tuple(torch.cat([past_k, new_k], dim=2).shape)", require=["cat"])


@template(M, "lite")
def kv_memory_formula(rng):
    layers, B, Hkv, Tn, Dh = rng.choice([24, 32, 40, 80]), rng.choice([1, 4, 8]), rng.choice([8, 32]), rng.choice([2048, 8192, 32768]), 128
    dt, nbytes = rng.choice([("bf16", 2), ("fp32", 4), ("fp8", 1)])
    return P(f"Estimate the KV-cache size in **GiB** (2³⁰ bytes) for {layers} layers, batch {B}, {Hkv} KV heads, "
             f"context {Tn} tokens, head dim {Dh}, stored in {dt} ({nbytes} bytes): "
             f"`2 (keys and values) · layers · batch · kv_heads · tokens · head_dim · bytes / 2**30`.",
             f"layers, batch, kv_heads, tokens, head_dim, nbytes = {layers}, {B}, {Hkv}, {Tn}, {Dh}, {nbytes}",
             "2 * layers * batch * kv_heads * tokens * head_dim * nbytes / 2**30", tol=1e-9)


# ================================================================= easy

@template(M, "easy")
def static_cache_update(rng):
    B, H, L, D = 1, rng.choice([1, 2]), rng.randint(6, 10), rng.choice([2, 4])
    pos, Tn = rng.randint(0, 3), rng.randint(1, 3)
    setup = (f"{seed_line(rng)}\nk_buf = torch.zeros({B}, {H}, {L}, {D})   # preallocated: max_len = {L}\n"
             f"k_buf[:, :, :{pos}] = torch.randn({B}, {H}, {pos}, {D})   # {pos} positions already filled\n"
             f"pos = {pos}\nk_new = torch.randn({B}, {H}, {Tn}, {D})")
    return P(f"A **static** KV cache writes into a preallocated buffer instead of re-allocating with `cat`. Write `k_new` "
             f"({Tn} new positions) into `k_buf` at positions `pos .. pos + {Tn - 1}` along dim 2, then return the filled "
             f"part `k_buf[:, :, :pos + {Tn}]`.", setup,
             f"k_buf[:, :, pos:pos + {Tn}] = k_new\nresult = k_buf[:, :, :pos + {Tn}]")


@template(M, "easy")
def decode_step_attention(rng):
    B, H, Tn, D = rng.randint(1, 2), rng.choice([2, 4]), rng.randint(3, 6), rng.choice([4, 8])
    setup = (f"{seed_line(rng)}\nq = torch.randn({B}, {H}, 1, {D})        # the ONE new token's query\n"
             f"K = torch.randn({B}, {H}, {Tn}, {D})\nV = torch.randn({B}, {H}, {Tn}, {D})   # cached keys/values incl. the new token")
    return P(f"During decoding the single new query may attend to **every** cached position (no causal mask needed). "
             f"Return its attention output, shape ({B}, {H}, 1, {D}): softmax over `q @ K^T / sqrt({D})`, times `V` "
             f"(or `F.scaled_dot_product_attention` without `is_causal`).", setup,
             f"result = (q @ K.transpose(-2, -1) / math.sqrt({D})).softmax(-1) @ V", tol=1e-5)


# ================================================================= medium

@template(M, "medium")
def prefill_decode_match(rng):
    d, h = rng.choice([8, 16]), rng.choice([2, 4])
    Tn, pre = rng.randint(5, 8), rng.randint(2, 4)
    setup = _CACHED + f"\n{seed_line(rng)}\nattn = CachedSelfAttention({d}, {h}).eval()\nseq = torch.randn(1, {Tn}, {d})"
    return P(f"Use the cache: inside `torch.no_grad()`, **prefill** with the first {pre} tokens (`y, cache = "
             f"attn(seq[:, :{pre}])`), then **decode** the remaining tokens one at a time (`y_t, cache = attn(seq[:, "
             f"t:t + 1], cache)`). Return all outputs concatenated along dim 1: it equals the full `attn(seq)[0]`.", setup,
             f"with torch.no_grad():\n    y, cache = attn(seq[:, :{pre}])\n    outs = [y]\n    for t in range({pre}, {Tn}):\n"
             f"        y_t, cache = attn(seq[:, t:t + 1], cache)\n        outs.append(y_t)\nresult = torch.cat(outs, dim=1)",
             require=["cache"], tol=1e-5)


@template(M, "medium")
def sliding_window_cache(rng):
    W, steps, D = rng.randint(2, 4), rng.randint(4, 7), rng.choice([2, 3])
    setup = (f"{seed_line(rng)}\ncache = torch.zeros(1, 1, 0, {D})          # empty: (B, H, T, D)\n"
             f"new_keys = torch.randn({steps}, 1, 1, 1, {D})   # one (B, H, 1, D) key per step")
    return P(f"A sliding-window cache keeps only the last {W} positions. For each of the {steps} new keys, append it "
             f"along dim 2 and then keep `cache[:, :, -{W}:]`. Return the final cache.", setup,
             f"for k_new in new_keys:\n    cache = torch.cat([cache, k_new], dim=2)[:, :, -{W}:]\nresult = cache",
             require=["cat"])


@template(M, "medium")
def chunked_prefill_mask(rng):
    Tp, Tn = rng.randint(2, 6), rng.randint(2, 4)
    return P(f"Chunked prefill: {Tn} new tokens attend to {Tp} cached tokens **plus** themselves. `is_causal=True` would "
             f"align the mask to the top-left, which is wrong here, so build it explicitly: a ({Tn}, {Tp + Tn}) boolean "
             f"mask (True = may attend) where new token i (absolute position {Tp} + i) sees every key j ≤ {Tp} + i. "
             f"Hint: `torch.ones(...).tril(diagonal={Tp})`.", "",
             f"torch.ones({Tn}, {Tp + Tn}, dtype=torch.bool).tril(diagonal={Tp})", require=["tril|triu|arange"])


# ================================================================= hard

@template(M, "hard")
def cached_attention_module(rng):
    d, h, Tn, pre = rng.choice([8, 12]), rng.choice([2, 4]), rng.randint(4, 6), rng.randint(1, 3)
    s = seeded(rng)
    post = (f"_m = CachedSelfAttention({d}, {h}).eval()\ntorch.manual_seed({s})\nwith torch.no_grad():\n"
            f"    for _n, _p in sorted(_m.named_parameters()):\n        _p.copy_(torch.randn_like(_p) * 0.3)\n"
            f"    _seq = torch.randn(2, {Tn}, {d})\n    _full, _ = _m(_seq)\n    _y, _c = _m(_seq[:, :{pre}])\n"
            f"    _outs = [_y]\n    for _t in range({pre}, {Tn}):\n        _yt, _c = _m(_seq[:, _t:_t + 1], _c)\n"
            f"        _outs.append(_yt)\n(_full, torch.cat(_outs, dim=1), tuple(_c[0].shape))")
    return P(f"Write `class CachedSelfAttention(nn.Module)` with `__init__(self, d, h)` creating `self.qkv = "
             f"nn.Linear(d, 3 * d)` and `self.proj = nn.Linear(d, d)`, and `forward(self, x, cache=None)` returning "
             f"`(output, (k, v))`: split q, k, v into heads (B, h, T, d/h); if a cache is given, prepend its keys and "
             f"values along dim 2; use `F.scaled_dot_product_attention` with `is_causal` only when there is no cache "
             f"and T > 1; merge heads and project. It is checked with d={d}, h={h}: prefill + decode must equal the full "
             f"pass.", "", _CACHED.strip(), post=post, require=["cat"], tol=1e-4)


@template(M, "hard")
def kv_cache_generate(rng):
    d, h, V, n_new = rng.choice([8, 16]), 2, rng.randint(6, 12), rng.randint(3, 5)
    prompt = [rng.randrange(V) for _ in range(rng.randint(2, 4))]
    setup = (_CACHED + f"\n{seed_line(rng)}\nattn = CachedSelfAttention({d}, {h}).eval()\nemb = nn.Embedding({V}, {d})\n"
             f"head = nn.Linear({d}, {V})\nidx = {tt([prompt])}")
    return P(f"Greedy generation with a KV cache: prefill with `y, cache = attn(emb(idx))`, then {n_new} times: pick "
             f"`nxt = head(y[:, -1]).argmax(-1)`, append it to the generated list, and run **only the new token** through "
             f"`y, cache = attn(emb(nxt)[:, None], cache)`. Do it under `torch.no_grad()` and return the {n_new} "
             f"generated token ids as a list of ints.", setup,
             f"out = []\nwith torch.no_grad():\n    y, cache = attn(emb(idx))\n    for _ in range({n_new}):\n"
             f"        nxt = head(y[:, -1]).argmax(-1)\n        out.append(nxt.item())\n        y, cache = attn(emb(nxt)[:, None], cache)\n"
             f"result = out", require=["argmax"])


# ================================================================= challenge

@template(M, "challenge")
def optimal_kv_evictions(rng):
    n, C = rng.randint(8, 11), rng.randint(2, 3)
    seq = [rng.randint(0, 5) for _ in range(n)]
    sol = ("import heapq\nseq = blocks.tolist()\nn = len(seq)\nINF = n + 1\nnxt, last = [0] * n, {}\n"
           "for i in range(n - 1, -1, -1):\n    nxt[i] = last.get(seq[i], INF)\n    last[seq[i]] = i\n"
           "cached, heap, misses = {}, [], 0\nfor i, b in enumerate(seq):\n    if b not in cached:\n        misses += 1\n"
           "        if len(cached) == C:\n            while True:\n                nu, bb = heapq.heappop(heap)\n"
           "                if cached.get(bb) == -nu:\n                    break\n            del cached[bb]\n"
           "    cached[b] = nxt[i]\n    heapq.heappush(heap, (-nxt[i], b))\nresult = misses")
    big = "blocks = torch.from_numpy(np.random.default_rng({s}).integers(0, {V}, size=20000))\nC = {C}"
    tests = [big.format(s=seeded(rng), V=3000, C=50), big.format(s=seeded(rng), V=3000, C=1),
             big.format(s=seeded(rng), V=200, C=150), "blocks = torch.arange(20000) % 7\nC = 6", "blocks = torch.tensor([9])\nC = 1"]
    return challenge("Optimal KV-cache eviction",
                     "A paged KV cache holds at most `C` blocks. `blocks` lists the block ids the decoder reads, in "
                     "order. Reading a block that isn't cached is a **miss**: load it, and if the cache is full first "
                     "evict one block. With perfect knowledge of the future, evict the block whose next read is "
                     "**farthest away** (or never). Return the number of misses.", f"blocks = {tt(seq)}\nC = {C}", sol,
                     tests, "1 ≤ C, n ≤ 2·10⁴", speed="O(n log n): precompute next uses, keep a max-heap with lazy deletion")


@template(M, "challenge")
def prefix_cache_reuse(rng):
    base = [rng.randint(1, 9) for _ in range(rng.randint(3, 5))]
    reqs = []
    for _ in range(rng.randint(3, 5)):
        k = rng.randint(0, len(base))
        reqs.append(base[:k] + [rng.randint(1, 9) for _ in range(rng.randint(0, 3))])
    sol = ("root, total = {}, 0\nfor req in requests:\n    node, reused, matching = root, 0, True\n    for tok in req:\n"
           "        if matching and tok in node:\n            reused += 1\n        else:\n            matching = False\n"
           "        node = node.setdefault(tok, {})\n    total += reused\nresult = total")
    big = ("g = np.random.default_rng({s})\nbases = [g.integers(0, {V}, size=int(g.integers(1, 50))).tolist() for _ in range({nb})]\n"
           "requests = []\nfor _ in range({nr}):\n    b = bases[int(g.integers(0, {nb}))]\n    k = int(g.integers(0, len(b) + 1))\n"
           "    requests.append(b[:k] + g.integers(0, {V}, size=int(g.integers(0, 30))).tolist())")
    tests = [big.format(s=seeded(rng), V=50, nb=20, nr=4000), big.format(s=seeded(rng), V=3, nb=5, nr=4000),
             "requests = [list(range(1000))] * 200", "requests = [[], [5], []]"]
    return challenge("Prefix-cache hits",
                     "A serving engine caches the KV state of every prompt it has processed in a prefix tree. When a new "
                     "request arrives, the tokens of its **longest prefix shared with any earlier request** are served "
                     "from the cache. `requests` lists the token-id prompts in arrival order. Return the total number of "
                     "cached tokens reused over all requests.", f"requests = {lit(reqs)}", sol, tests,
                     "≤ 4·10³ requests, ≤ 3.2·10⁵ tokens in total", speed="O(total tokens) with a trie")
