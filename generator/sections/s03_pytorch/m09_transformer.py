"""3.09 Transformer block."""

from ._common import P, challenge, seed_line, seeded, template, tt

M = "3.09"


def _x(rng, n=None, d=None):
    n, d = n or rng.randint(2, 4), d or rng.choice([4, 6, 8])
    return n, d, f"{seed_line(rng)}\nx = torch.randn({n}, {d})"


def _randomize_post(cls_call, s, x_shape):
    """Post-check for 'write a module' tasks: fill parameters by sorted name, so creation order doesn't matter."""
    return (f"_m = {cls_call}\ntorch.manual_seed({s})\nwith torch.no_grad():\n    for _n, _p in sorted(_m.named_parameters()):\n"
            f"        _p.copy_(torch.randn_like(_p) * 0.3)\n(sorted(n for n, _ in _m.named_parameters()), _m(torch.randn{x_shape}))")


# ================================================================= lite

@template(M, "lite")
def rmsnorm_formula(rng):
    n, d, setup = _x(rng)
    eps = rng.choice([1e-6, 1e-5])
    if rng.random() < 0.6:
        return P(f"RMSNorm by hand: `x * rsqrt(mean(x², over the last dim) + eps) * weight` with `eps={eps}` and the "
                 f"given `weight`. Return the result for `x` ({n}, {d}).", setup + f"\nweight = torch.linspace(0.5, 1.5, {d})",
                 f"result = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + {eps}) * weight",
                 require=["rsqrt|sqrt"], tol=1e-5)
    return P(f"Normalize `x` with PyTorch's built-in `nn.RMSNorm({d}, eps={eps})` (its weight starts at 1).", setup,
             f"nn.RMSNorm({d}, eps={eps})(x)", require=["RMSNorm"], tol=1e-5)


@template(M, "lite")
def layernorm_apply(rng):
    n, d, setup = _x(rng)
    if rng.random() < 0.5:
        return P(f"Apply a fresh `nn.LayerNorm({d})` to `x` (weight 1, bias 0, eps 1e-5): each row gets mean 0 and "
                 f"variance 1.", setup, f"nn.LayerNorm({d})(x)", require=["LayerNorm|layer_norm"], tol=1e-5)
    return P("LayerNorm by hand: subtract each row's mean and divide by `sqrt(var + 1e-5)`, where `var` is the "
             "**biased** variance (`unbiased=False`). Return the result.", setup,
             "mu = x.mean(-1, keepdim=True)\nvar = x.var(-1, unbiased=False, keepdim=True)\nresult = (x - mu) / torch.sqrt(var + 1e-5)",
             require=["mean"], tol=1e-5)


# ================================================================= easy

@template(M, "easy")
def gelu_mlp(rng):
    n, d, setup = _x(rng)
    h = 4 * d
    approx = rng.choice(["none", "tanh"])
    setup += f"\nW1, b1 = torch.randn({h}, {d}) * 0.3, torch.zeros({h})\nW2, b2 = torch.randn({d}, {h}) * 0.3, torch.zeros({d})"
    return P(f"The transformer MLP: `fc2(gelu(fc1(x)))` with hidden size 4·d. Using the given weights (stored like "
             f"`nn.Linear`: (out, in)), return `F.linear(F.gelu(F.linear(x, W1, b1), approximate='{approx}'), W2, b2)`.",
             setup, f"F.linear(F.gelu(F.linear(x, W1, b1), approximate='{approx}'), W2, b2)", require=["gelu"], tol=1e-5)


@template(M, "easy")
def swiglu(rng):
    n, d, setup = _x(rng)
    h = rng.choice([2 * d, 3 * d])
    setup += f"\nw1, w3 = torch.randn({h}, {d}) * 0.3, torch.randn({h}, {d}) * 0.3\nw2 = torch.randn({d}, {h}) * 0.3"
    return P(f"LLaMA's SwiGLU feed-forward (no biases): `w2(silu(w1 x) * w3 x)`. With the given (out, in) weight "
             f"matrices, return it for `x`.", setup, "F.linear(F.silu(F.linear(x, w1)) * F.linear(x, w3), w2)",
             require=["silu"], tol=1e-5)


@template(M, "easy")
def residual_prenorm(rng):
    n, d, setup = _x(rng)
    setup += f"\nW = torch.randn({d}, {d}) * 0.3"
    if rng.random() < 0.5:
        return P(f"A **pre-norm** residual step: `x + f(norm(x))`, with `norm` = `F.layer_norm(x, ({d},))` and `f(z) = "
                 f"z @ W.T`. Return the result.", setup, f"result = x + F.layer_norm(x, ({d},)) @ W.T",
                 require=["layer_norm|LayerNorm"], tol=1e-5)
    return P(f"A **post-norm** residual step (the original Transformer): `norm(x + f(x))`, with `norm` = "
             f"`F.layer_norm(..., ({d},))` and `f(z) = z @ W.T`. Return the result.", setup,
             f"result = F.layer_norm(x + x @ W.T, ({d},))", require=["layer_norm|LayerNorm"], tol=1e-5)


# ================================================================= medium

@template(M, "medium")
def sinusoidal_positions(rng):
    Tn, d = rng.randint(3, 6), rng.choice([4, 6, 8])
    return P(f"Build the original Transformer's sinusoidal position encoding `pe` ({Tn}, {d}): for position `pos` and "
             f"even index `i`, the angle is `pos / 10000 ** (i / d)`; `pe[:, 0::2] = sin(angle)`, "
             f"`pe[:, 1::2] = cos(angle)`. Use `pos = torch.arange({Tn})[:, None]` and `i = torch.arange(0, {d}, 2)`.", "",
             f"pos, i = torch.arange({Tn})[:, None], torch.arange(0, {d}, 2)\nangle = pos / (10000 ** (i / {d}))\n"
             f"pe = torch.zeros({Tn}, {d})\npe[:, 0::2], pe[:, 1::2] = torch.sin(angle), torch.cos(angle)\nresult = pe",
             require=["sin", "cos"], tol=1e-5)


@template(M, "medium")
def apply_rope(rng):
    B, H, Tn, D = 1, rng.choice([1, 2]), rng.randint(3, 5), rng.choice([4, 6])
    setup = f"{seed_line(rng)}\nx = torch.randn({B}, {H}, {Tn}, {D})"
    return P(f"Rotary position embeddings (RoPE) for `x` ({B}, {H}, {Tn}, {D}): `inv_freq = 1 / 10000 ** (arange(0, D, "
             f"2) / D)`, angles `ang = arange(T)[:, None] * inv_freq` (T, D/2); with `x1, x2 = x[..., 0::2], x[..., "
             f"1::2]`, return `stack((x1·cos − x2·sin, x1·sin + x2·cos), dim=-1).flatten(-2)`.", setup,
             f"inv_freq = 1.0 / 10000 ** (torch.arange(0, {D}, 2) / {D})\nang = torch.arange({Tn})[:, None] * inv_freq\n"
             f"x1, x2 = x[..., 0::2], x[..., 1::2]\n"
             f"result = torch.stack((x1 * ang.cos() - x2 * ang.sin(), x1 * ang.sin() + x2 * ang.cos()), dim=-1).flatten(-2)",
             require=["cos", "sin"], tol=1e-5)


@template(M, "medium")
def next_token_loss(rng):
    B, Tn, V = rng.randint(1, 3), rng.randint(4, 7), rng.randint(5, 12)
    setup = (f"{seed_line(rng)}\nidx = torch.randint(0, {V}, ({B}, {Tn}))\n"
             f"logits = torch.randn({B}, {Tn - 1}, {V})   # the model's output for idx[:, :-1]")
    return P(f"Next-token prediction: `logits` ({B}, {Tn - 1}, {V}) were computed from the inputs `idx[:, :-1]`, so "
             f"the targets are the tokens shifted by one, `idx[:, 1:]`. Flatten to ({B * (Tn - 1)}, {V}) and "
             f"({B * (Tn - 1)},) and return the cross-entropy loss.", setup,
             f"targets = idx[:, 1:]\nresult = F.cross_entropy(logits.reshape(-1, {V}), targets.reshape(-1))",
             require=["cross_entropy"], tol=1e-5)


# ================================================================= hard

_PARTS = ("class MultiHeadSelfAttention(nn.Module):\n    def __init__(self, d, h):\n        super().__init__()\n"
          "        self.h = h\n        self.qkv = nn.Linear(d, 3 * d, bias=False)\n        self.proj = nn.Linear(d, d)\n"
          "    def forward(self, x):\n        B, T, C = x.shape\n        q, k, v = self.qkv(x).split(C, dim=-1)\n"
          "        q, k, v = (t.view(B, T, self.h, C // self.h).transpose(1, 2) for t in (q, k, v))\n"
          "        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)\n"
          "        return self.proj(y.transpose(1, 2).reshape(B, T, C))\n\n"
          "class MLPBlock(nn.Module):\n    def __init__(self, d, hidden):\n        super().__init__()\n"
          "        self.fc1, self.fc2 = nn.Linear(d, hidden), nn.Linear(hidden, d)\n"
          "    def forward(self, x):\n        return self.fc2(F.gelu(self.fc1(x)))")


@template(M, "hard")
def block_module(rng):
    d, h, Tn = rng.choice([8, 12, 16]), rng.choice([2, 4]), rng.randint(3, 5)
    pre = rng.random() < 0.7
    ratio = rng.choice([2, 4])
    layout = ("**pre-norm** (GPT-2 / LLaMA): `x = x + attn(ln1(x))`, then `x = x + mlp(ln2(x))`" if pre else
              "**post-norm** (the original Transformer): `x = ln1(x + attn(x))`, then `x = ln2(x + mlp(x))`")
    fwd = ("        x = x + self.attn(self.ln1(x))\n        return x + self.mlp(self.ln2(x))" if pre else
           "        x = self.ln1(x + self.attn(x))\n        return self.ln2(x + self.mlp(x))")
    return P(f"Using the given `MultiHeadSelfAttention` and `MLPBlock`, write `class Block(nn.Module)` with "
             f"`__init__(self, d, n_heads)` creating `self.ln1`, `self.ln2` (`nn.LayerNorm(d)`), `self.attn` and "
             f"`self.mlp = MLPBlock(d, {ratio} * d)`, and a {layout}. It is checked with d={d}, n_heads={h}.", _PARTS,
             f"class Block(nn.Module):\n    def __init__(self, d, n_heads):\n        super().__init__()\n"
             f"        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)\n        self.attn = MultiHeadSelfAttention(d, n_heads)\n"
             f"        self.mlp = MLPBlock(d, {ratio} * d)\n    def forward(self, x):\n{fwd}",
             post=_randomize_post(f"Block({d}, {h})", seeded(rng), f"(2, {Tn}, {d})"), require=["Module"], tol=1e-4)


@template(M, "hard")
def top_k_sampling_probs(rng):
    B, V, k = rng.randint(1, 3), rng.randint(6, 12), rng.randint(2, 4)
    temp = rng.choice([0.7, 1.0, 1.5, 0.5])
    setup = f"{seed_line(rng)}\nlogits = torch.randn({B}, {V})   # last-position logits"
    return P(f"Turn next-token `logits` into sampling probabilities with temperature {temp} and **top-k = {k}**: divide "
             f"by the temperature, keep only the {k} largest logits per row (set the rest to `-inf`, using the k-th "
             f"largest from `topk` as the threshold), then softmax. Return the ({B}, {V}) probabilities.", setup,
             f"z = logits / {temp}\nkth = z.topk({k}).values[:, [-1]]\nresult = z.masked_fill(z < kth, float('-inf')).softmax(-1)",
             require=["topk"], tol=1e-5)


@template(M, "hard")
def greedy_generate(rng):
    V, n_new = rng.randint(5, 9), rng.randint(3, 6)
    start = [[rng.randrange(V)] for _ in range(rng.randint(1, 2))]
    setup = (f"{seed_line(rng)}\ntable = torch.randn({V}, {V})   # bigram logits: row = current token\n"
             f"def model(idx):                  # (B, T) token ids -> (B, T, V) logits\n    return table[idx]\n"
             f"idx = {tt(start)}")
    return P(f"Write greedy decoding: for {n_new} steps, run `model(idx)`, take the logits at the **last** position, "
             f"pick `argmax(-1, keepdim=True)`, and append it to `idx` with `torch.cat(..., dim=1)`. Return the final "
             f"`idx` (shape (B, {1 + n_new})).", setup,
             f"for _ in range({n_new}):\n    logits = model(idx)[:, -1, :]\n    nxt = logits.argmax(-1, keepdim=True)\n"
             f"    idx = torch.cat([idx, nxt], dim=1)\nresult = idx", require=["argmax", "cat"])


# ================================================================= challenge

@template(M, "challenge")
def greedy_decode_cycle(rng):
    V = rng.randint(5, 8)
    nxt = [rng.randrange(V) for _ in range(V)]
    s, N = rng.randrange(V), rng.randint(5, 25)
    sol = ("table = next_tok.tolist()\nseen, seq, t = {}, [], start\nwhile t not in seen:\n    seen[t] = len(seq)\n"
           "    seq.append(t)\n    t = table[t]\nmu = seen[t]\nlam = len(seq) - mu\n"
           "result = seq[N] if N < len(seq) else seq[mu + (N - mu) % lam]")
    big = "next_tok = torch.from_numpy(np.random.default_rng({s}).integers(0, {V}, size={V}))\nstart, N = {st}, {N}"
    tests = [big.format(s=seeded(rng), V=20000, st=0, N=10**18), big.format(s=seeded(rng), V=20000, st=123, N=19999),
             "next_tok = torch.cat([torch.arange(1, 20000), torch.tensor([19999])])\nstart, N = 0, 10**18",
             "next_tok = torch.tensor([0])\nstart, N = 0, 1"]
    return challenge("Where does greedy decoding end up?",
                     "A tiny bigram model always predicts `next_tok[t]` after token t. Greedy decoding starts at token "
                     "`start` and generates one token per step. Return the token produced after **N steps** "
                     "(after 0 steps it would be `start`).", f"next_tok = {tt(nxt)}\nstart, N = {s}, {N}", sol, tests,
                     "1 ≤ V ≤ 2·10⁴ tokens, 1 ≤ N ≤ 10¹⁸", speed="O(V): detect the cycle, then use modular arithmetic")


@template(M, "challenge")
def top_p_cutoff_queries(rng):
    V = rng.randint(5, 8)
    w = [rng.randint(1, 20) for _ in range(V)]
    qs = [rng.randint(1, sum(w)) for _ in range(rng.randint(3, 5))]
    sol = ("from bisect import bisect_left\nfrom itertools import accumulate\n"
           "pref = list(accumulate(sorted(weights.tolist(), reverse=True)))\n"
           "result = [bisect_left(pref, p) + 1 for p in thresholds.tolist()]")
    big = ("g = np.random.default_rng({s})\nweights = torch.from_numpy(g.integers(1, {hi}, size=20000))\n"
           "thresholds = torch.from_numpy(g.integers(1, int(weights.sum()) + 1, size=20000))")
    tests = [big.format(s=seeded(rng), hi=10**9), big.format(s=seeded(rng), hi=3),
             "weights = torch.ones(20000, dtype=torch.long)\nthresholds = torch.tensor([1, 20000, 10000])",
             "weights = torch.tensor([5])\nthresholds = torch.tensor([5])"]
    return challenge("Nucleus sizes",
                     "Nucleus (top-p) sampling keeps the **smallest** set of most-likely tokens whose total weight "
                     "reaches a threshold. `weights` holds the integer weights of V tokens; for each threshold `p` in "
                     "`thresholds` (1 ≤ p ≤ total weight), return the smallest k such that the k largest weights sum to "
                     "at least p.", f"weights = {tt(w)}\nthresholds = {tt(qs)}", sol, tests,
                     "1 ≤ V, q ≤ 2·10⁴, weights ≤ 10⁹", speed="O((V + q) log V): sort, prefix sums, binary search")
