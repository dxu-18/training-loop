"""3.08 Attention."""

from ._common import P, challenge, seed_line, seeded, template, tt

M = "3.08"


def _qk(rng, Tn=None, D=None, v=False):
    Tn, D = Tn or rng.randint(3, 5), D or rng.choice([2, 4, 8])
    s = f"{seed_line(rng)}\nq = torch.randn({Tn}, {D})\nk = torch.randn({Tn}, {D})" + (f"\nv = torch.randn({Tn}, {D})" if v else "")
    return Tn, D, s


def _bhtd(rng, causal_note=True):
    B, H, Tn, D = rng.randint(1, 2), rng.choice([2, 4]), rng.randint(3, 6), rng.choice([4, 8])
    setup = f"{seed_line(rng)}\nq, k, v = (torch.randn({B}, {H}, {Tn}, {D}) for _ in range(3))"
    return B, H, Tn, D, setup


# ================================================================= lite

@template(M, "lite")
def attention_scores(rng):
    Tn, D, setup = _qk(rng)
    task = rng.choice(["scores", "weights", "rowsum"])
    if task == "scores":
        return P(f"`q` and `k` are ({Tn}, {D}) queries and keys of one head. Return the **scaled** attention scores "
                 f"`q @ k.T / sqrt({D})` ({Tn}×{Tn}).", setup, f"q @ k.T / math.sqrt({D})", tol=1e-5)
    if task == "weights":
        return P(f"Return the attention **weights** for queries `q` and keys `k`: softmax over the keys (last dim) of "
                 f"`q @ k.T / sqrt({D})`.", setup, f"(q @ k.T / math.sqrt({D})).softmax(dim=-1)", require=["softmax"],
                 tol=1e-5)
    return P(f"Compute the attention weights `softmax(q @ k.T / sqrt({D}), dim=-1)` and return their row sums (each "
             f"should be 1).", setup, f"(q @ k.T / math.sqrt({D})).softmax(dim=-1).sum(dim=-1)", require=["softmax"],
             tol=1e-5)


@template(M, "lite")
def causal_mask_build(rng):
    Tn = rng.randint(3, 7)
    if rng.random() < 0.5:
        return P(f"Build the {Tn}×{Tn} boolean **causal** mask that is True for *future* positions (above the diagonal), "
                 f"ready for `masked_fill`: `torch.triu(torch.ones({Tn}, {Tn}, dtype=torch.bool), diagonal=1)`.", "",
                 f"torch.triu(torch.ones({Tn}, {Tn}, dtype=torch.bool), diagonal=1)", require=["triu|tril"])
    return P(f"Build the {Tn}×{Tn} boolean mask that is True where query i **may attend** to key j (j ≤ i), the "
             f"convention `F.scaled_dot_product_attention` uses for `attn_mask`.", "",
             f"torch.tril(torch.ones({Tn}, {Tn}, dtype=torch.bool))", require=["triu|tril"])


# ================================================================= easy

@template(M, "easy")
def masked_softmax(rng):
    Tn, D, setup = _qk(rng)
    return P(f"Apply a causal mask to the scores `q @ k.T / sqrt({D})`: set future positions (j > i) to `-inf` with "
             f"`masked_fill`, then softmax over the keys. Return the {Tn}×{Tn} weights (row i is 0 after column i).",
             setup, f"scores = q @ k.T / math.sqrt({D})\nfuture = torch.triu(torch.ones({Tn}, {Tn}, dtype=torch.bool), diagonal=1)\n"
                    f"result = scores.masked_fill(future, float('-inf')).softmax(dim=-1)", require=["masked_fill"],
             tol=1e-5)


@template(M, "easy")
def sdpa_call(rng):
    B, H, Tn, D, setup = _bhtd(rng)
    task = rng.choice(["causal", "scale", "plain"])
    desc = f"`q`, `k`, `v` are ({B}, {H}, {Tn}, {D}) (batch, heads, time, head dim)"
    if task == "causal":
        return P(f"{desc}. Return causal self-attention with the fused `F.scaled_dot_product_attention(..., "
                 f"is_causal=True)`.", setup, "F.scaled_dot_product_attention(q, k, v, is_causal=True)",
                 require=["scaled_dot_product_attention"], tol=1e-5)
    if task == "scale":
        s = rng.choice([0.5, 0.25, 1.0])
        return P(f"{desc}. Return (non-causal) attention from `F.scaled_dot_product_attention` with the softmax scale "
                 f"overridden to {s} instead of 1/sqrt({D}).", setup,
                 f"F.scaled_dot_product_attention(q, k, v, scale={s})", require=["scaled_dot_product_attention"],
                 tol=1e-5)
    return P(f"{desc}. Return full (bidirectional, non-causal) attention with `F.scaled_dot_product_attention`.", setup,
             "F.scaled_dot_product_attention(q, k, v)", require=["scaled_dot_product_attention"], tol=1e-5)


@template(M, "easy")
def attend_values(rng):
    Tn, D, setup = _qk(rng, v=True)
    if rng.random() < 0.5:
        return P(f"Compute single-head attention by hand: weights = `softmax(q @ k.T / sqrt({D}), dim=-1)`, output = "
                 f"`weights @ v`. Return the ({Tn}, {D}) output.", setup,
                 f"w = (q @ k.T / math.sqrt({D})).softmax(dim=-1)\nresult = w @ v", require=["softmax"], tol=1e-5)
    i = rng.randrange(Tn)
    return P(f"Return the attention output of **query {i} only**: its weights over all keys "
             f"(`softmax(q[{i}] @ k.T / sqrt({D}))`) times `v`, a vector of size {D}.", setup,
             f"w = (q[{i}] @ k.T / math.sqrt({D})).softmax(dim=-1)\nresult = w @ v", require=["softmax"], tol=1e-5)


# ================================================================= medium

@template(M, "medium")
def attention_from_scratch(rng):
    B, H, Tn, D, setup = _bhtd(rng)
    return P(f"Implement causal attention **from scratch** for ({B}, {H}, {Tn}, {D}) `q`, `k`, `v` (it must match "
             f"`F.scaled_dot_product_attention(q, k, v, is_causal=True)`): scores `q @ k.transpose(-2, -1) / "
             f"sqrt(D)`, mask the future with `-inf`, softmax over the keys, multiply by `v`.", setup,
             f"scores = q @ k.transpose(-2, -1) / math.sqrt({D})\ncausal = torch.triu(torch.ones({Tn}, {Tn}, dtype=torch.bool), diagonal=1)\n"
             f"result = scores.masked_fill(causal, float('-inf')).softmax(dim=-1) @ v", require=["masked_fill|where"],
             tol=1e-5)


@template(M, "medium")
def padding_mask_attention(rng):
    B, H, Tn, D, setup = _bhtd(rng)
    B = max(B, 2)
    setup = f"{seed_line(rng)}\nq, k, v = (torch.randn({B}, {H}, {Tn}, {D}) for _ in range(3))"
    lens = [Tn] + [rng.randint(1, Tn - 1) for _ in range(B - 1)]
    return P(f"`lengths` holds the real lengths of the {B} sequences (the rest is padding). Build the ({B}, {Tn}) "
             f"padding mask (True = real token), reshape it to ({B}, 1, 1, {Tn}) so it broadcasts over heads and "
             f"queries, and pass it as `attn_mask` to `F.scaled_dot_product_attention` (True = may attend). Return the "
             f"output.", setup + f"\nlengths = {tt(lens)}",
             f"pad = torch.arange({Tn})[None, :] < lengths[:, None]\n"
             f"result = F.scaled_dot_product_attention(q, k, v, attn_mask=pad[:, None, None, :])",
             require=["attn_mask"], tol=1e-5)


@template(M, "medium")
def cross_attention(rng):
    B, Tq, Tk, D = rng.randint(1, 2), rng.randint(2, 4), rng.randint(3, 6), rng.choice([4, 8])
    setup = (f"{seed_line(rng)}\nx = torch.randn({B}, {Tq}, {D})      # decoder states (queries)\n"
             f"enc = torch.randn({B}, {Tk}, {D})    # encoder states (keys and values)\n"
             f"Wq, Wk, Wv = (torch.randn({D}, {D}) / math.sqrt({D}) for _ in range(3))")
    return P(f"Cross-attention: queries come from `x` ({Tq} positions), keys and values from `enc` ({Tk} positions). "
             f"Project `q = x @ Wq`, `k = enc @ Wk`, `v = enc @ Wv`, then return `softmax(q @ k^T / sqrt({D})) @ v` "
             f"(no mask), shape ({B}, {Tq}, {D}).", setup,
             f"q, k, v = x @ Wq, enc @ Wk, enc @ Wv\nresult = (q @ k.transpose(-2, -1) / math.sqrt({D})).softmax(dim=-1) @ v",
             tol=1e-5)


# ================================================================= hard

@template(M, "hard")
def mhsa_module(rng):
    d, h, Tn = rng.choice([8, 12, 16]), rng.choice([2, 4]), rng.randint(3, 6)
    causal = rng.random() < 0.6
    post = (f"torch.manual_seed({seeded(rng)})\n_ref = nn.ModuleDict({{'qkv': nn.Linear({d}, {3 * d}, bias=False), "
            f"'proj': nn.Linear({d}, {d})}})\n_m = MultiHeadSelfAttention({d}, {h})\n_m.load_state_dict(_ref.state_dict())\n"
            f"_m(torch.randn(2, {Tn}, {d}))")
    return P(f"Write `class MultiHeadSelfAttention(nn.Module)` with `__init__(self, d_model, n_heads)` creating "
             f"`self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)` and `self.proj = nn.Linear(d_model, d_model)`. "
             f"`forward(x)` for x of shape (B, T, C): split `self.qkv(x)` into q, k, v along the last dim, reshape each "
             f"to (B, heads, T, C/heads), run `F.scaled_dot_product_attention` ({'causal' if causal else 'not causal'}), "
             f"merge the heads back to (B, T, C), and apply `self.proj`. It is checked with d_model={d}, n_heads={h}.", "",
             f"class MultiHeadSelfAttention(nn.Module):\n    def __init__(self, d_model, n_heads):\n        super().__init__()\n"
             f"        self.h = n_heads\n        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)\n"
             f"        self.proj = nn.Linear(d_model, d_model)\n    def forward(self, x):\n        B, T, C = x.shape\n"
             f"        q, k, v = self.qkv(x).split(C, dim=-1)\n"
             f"        q, k, v = (t.view(B, T, self.h, C // self.h).transpose(1, 2) for t in (q, k, v))\n"
             f"        y = F.scaled_dot_product_attention(q, k, v, is_causal={causal})\n"
             f"        return self.proj(y.transpose(1, 2).reshape(B, T, C))", post=post, require=["Module"], tol=1e-5)


@template(M, "hard")
def gqa_attention(rng):
    B, Hkv, rep, Tn, D = rng.randint(1, 2), rng.choice([1, 2]), rng.choice([2, 4]), rng.randint(3, 6), rng.choice([4, 8])
    H = Hkv * rep
    setup = (f"{seed_line(rng)}\nq = torch.randn({B}, {H}, {Tn}, {D})\nk = torch.randn({B}, {Hkv}, {Tn}, {D})\n"
             f"v = torch.randn({B}, {Hkv}, {Tn}, {D})")
    return P(f"Grouped-query attention: `q` has {H} heads but `k` and `v` only {Hkv}, each shared by {rep} query heads. "
             f"Expand `k` and `v` to {H} heads with `repeat_interleave({rep}, dim=1)` and return causal "
             f"`F.scaled_dot_product_attention`.", setup,
             f"kk, vv = k.repeat_interleave({rep}, dim=1), v.repeat_interleave({rep}, dim=1)\n"
             f"result = F.scaled_dot_product_attention(q, kk, vv, is_causal=True)", require=["repeat_interleave|expand"],
             tol=1e-5)


@template(M, "hard")
def attention_entropy(rng):
    B, H, Tn, D, setup = _bhtd(rng)
    return P(f"How peaked is each query's attention? For causal attention over `q`, `k` ({B}, {H}, {Tn}, {D}), compute the "
             f"weights, then each query's **entropy** `-Σ_j w·log w` (masked positions have w = 0 and contribute 0: use "
             f"`torch.where` or `torch.xlogy`), averaged over heads. Return the ({B}, {Tn}) tensor.", setup,
             f"scores = q @ k.transpose(-2, -1) / math.sqrt({D})\ncausal = torch.triu(torch.ones({Tn}, {Tn}, dtype=torch.bool), diagonal=1)\n"
             f"w = scores.masked_fill(causal, float('-inf')).softmax(dim=-1)\n"
             f"ent = -torch.where(w > 0, w * w.log(), torch.zeros_like(w)).sum(-1)\nresult = ent.mean(dim=1)",
             no_loops=True, tol=1e-5)


# ================================================================= challenge

@template(M, "challenge")
def windowed_attention_peak(rng):
    n = rng.randint(7, 9)
    q = [rng.randint(-9, 9) for _ in range(n)]
    k = [rng.randint(-9, 9) for _ in range(n)]
    w = rng.randint(2, 4)
    sol = ("from collections import deque\nqs, ks = q.tolist(), k.tolist()\nhi, lo, out = deque(), deque(), []\n"
           "for i, x in enumerate(ks):\n    while hi and ks[hi[-1]] <= x:\n        hi.pop()\n    while lo and ks[lo[-1]] >= x:\n"
           "        lo.pop()\n    hi.append(i)\n    lo.append(i)\n    if hi[0] <= i - w:\n        hi.popleft()\n"
           "    if lo[0] <= i - w:\n        lo.popleft()\n    out.append(max(qs[i] * ks[hi[0]], qs[i] * ks[lo[0]]))\nresult = out")
    big = ("g = np.random.default_rng({s})\nq = torch.from_numpy(g.integers(-10**4, 10**4 + 1, 20000))\n"
           "k = torch.from_numpy(g.integers(-10**4, 10**4 + 1, 20000))\nw = {w}")
    tests = [big.format(s=seeded(rng), w=1), big.format(s=seeded(rng), w=777), big.format(s=seeded(rng), w=20000),
             "q = torch.tensor([-3])\nk = torch.tensor([5])\nw = 1"]
    return challenge("Sliding-window attention peaks",
                     "In sliding-window causal attention with 1-D queries and keys, query i sees keys `j` with "
                     "`i − w < j ≤ i`, and its raw score for key j is `q[i] · k[j]`. For every query, return its "
                     "**largest** raw score.", f"q = {tt(q)}\nk = {tt(k)}\nw = {w}", sol, tests,
                     "1 ≤ w ≤ n ≤ 2·10⁴, |q|, |k| ≤ 10⁴",
                     speed="O(n): the best key is the window's max or min key, depending on the sign of q[i]")


@template(M, "challenge")
def causal_top1_counts(rng):
    n = rng.randint(6, 9)
    q = [rng.randint(-5, 5) for _ in range(n)]
    k = [rng.randint(-5, 5) for _ in range(n)]
    sol = ("qs, ks = q.tolist(), k.tolist()\ncounts = [0] * len(ks)\nbest_hi = best_lo = 0\nfor i, x in enumerate(ks):\n"
           "    if x > ks[best_hi]:\n        best_hi = i\n    if x < ks[best_lo]:\n        best_lo = i\n"
           "    if qs[i] > 0:\n        counts[best_hi] += 1\n    elif qs[i] < 0:\n        counts[best_lo] += 1\n"
           "    else:\n        counts[0] += 1\nresult = counts")
    big = ("g = np.random.default_rng({s})\nq = torch.from_numpy(g.integers(-{a}, {a} + 1, 20000))\n"
           "k = torch.from_numpy(g.integers(-{b}, {b} + 1, 20000))")
    tests = [big.format(s=seeded(rng), a=10**9, b=10**9), big.format(s=seeded(rng), a=1, b=3),
             "q = torch.ones(20000, dtype=torch.long)\nk = torch.arange(20000)", "q = torch.tensor([0])\nk = torch.tensor([4])"]
    return challenge("Who gets attended to?",
                     "With causal attention over 1-D queries and keys, query i scores key j (for `j ≤ i`) as "
                     "`q[i] · k[j]` and puts its top weight on the key with the **highest score, earliest j on ties**. "
                     "Return, for every key, how many queries put their top weight on it.",
                     f"q = {tt(q)}\nk = {tt(k)}", sol, tests, "1 ≤ n ≤ 2·10⁴, |q|, |k| ≤ 10⁹",
                     speed="O(n): track the running argmax and argmin of the keys")
