"""Catalog templates for the guide syntax of 3.02-3.18 that the other templates don't reach (one API per problem)."""

from ._common import STDOUT, catalog, fm, ftt, fv, lit, seeded, template, tt


def _fmat(rows):
    return "torch.tensor(" + lit([[float(v) for v in r] for r in rows]) + ")"


# ================================================================= 3.02 Broadcasting & shape changes

@template("3.02", "lite")
def shape_api_catalog(rng):
    a, b, c = rng.randint(2, 3), rng.randint(2, 4), rng.randint(2, 5)
    n, m = rng.randint(2, 4), rng.randint(3, 5)
    setup = f"z = torch.arange({a * b * c}.).reshape({a}, {b}, {c})\ncol = torch.arange(1, {n + 1}).reshape({n}, 1)"
    items = [
        ("ravel", "Flatten `z` to 1-D with `torch.ravel`.", "torch.ravel(z)"),
        ("view_as", f"View `z` with the shape of `torch.empty({a * b}, {c})` (`z.view_as(...)`).", f"z.view_as(torch.empty({a * b}, {c}))",
         {"require": ["view_as", "empty"]}),
        ("reshape_as", f"Reshape `z` to the shape of `torch.empty({c}, {a * b})` (`z.reshape_as(...)`).", f"z.reshape_as(torch.empty({c}, {a * b}))"),
        ("atleast_2d", "Return the shape of `torch.atleast_2d(torch.tensor(1.))` as a tuple.", "tuple(torch.atleast_2d(torch.tensor(1.)).shape)"),
        ("swapaxes", "Swap dims 0 and 1 of `z` with `swapaxes` and return the new shape as a tuple.", "tuple(z.swapaxes(0, 1).shape)"),
        ("expand_as", f"Broadcast `col` to the shape of `torch.empty({n}, {m})` without copying (`expand_as`).",
         f"col.expand_as(torch.empty({n}, {m}))"),
        ("broadcast_to", f"Return the shape of `torch.broadcast_to(col, (2, {n}, {m}))` as a tuple.", f"tuple(torch.broadcast_to(col, (2, {n}, {m})).shape)"),
        ("broadcast_tensors", f"Broadcast `torch.ones({n}, 1)` and `torch.ones(1, {m})` against each other with "
                              f"`torch.broadcast_tensors` and return both shapes as tuples.",
         f"p, q = torch.broadcast_tensors(torch.ones({n}, 1), torch.ones(1, {m}))\nresult = [tuple(p.shape), tuple(q.shape)]"),
        ("masked_fill", f"Padding mask over heads and queries: `lengths = torch.tensor([2, {c}])`, `mask = torch.arange({c})[None, :] "
                        f"< lengths[:, None]`. Fill `torch.zeros(2, 3, {c}, {c})` with -inf where `~mask[:, None, None, :]` "
                        f"(it broadcasts to (B, H, T, T)) and return how many entries became -inf.",
         f"lengths = torch.tensor([2, {c}])\nmask = torch.arange({c})[None, :] < lengths[:, None]\n"
         f"scores = torch.zeros(2, 3, {c}, {c}).masked_fill(~mask[:, None, None, :], float('-inf'))\nresult = torch.isinf(scores).sum().item()"),
    ]
    return catalog(rng, items, setup, "")


# ================================================================= 3.03 Autograd

@template("3.03", "easy")
def grad_mode_catalog(rng):
    vals = fv(rng, 3, -2, 2, 1, nonzero=True)
    k = rng.randint(2, 4)
    setup = f"x = torch.tensor({lit(vals)}, requires_grad=True)"
    items = [
        ("enable_grad", f"Inside `torch.no_grad()`, re-enable recording with `torch.enable_grad()` for `y = x * {k}`. Return "
                        f"`y.requires_grad`.", f"with torch.no_grad():\n    with torch.enable_grad():\n        y = x * {k}\nresult = y.requires_grad"),
        ("inference_mode", f"Compute `y = x * {k}` inside `torch.inference_mode()` (faster than no_grad) and return "
                           f"`y.requires_grad`.", f"with torch.inference_mode():\n    y = x * {k}\nresult = y.requires_grad"),
        ("set_grad_enabled", f"Switch recording off with `torch.set_grad_enabled(False)`, compute `y = x * {k}`, switch it back "
                             f"on, and return `y.requires_grad`.",
         f"torch.set_grad_enabled(False)\ny = x * {k}\ntorch.set_grad_enabled(True)\nresult = y.requires_grad"),
        ("no_grad", f"Decorate `def evaluate(t): return t * {k}` with `@torch.no_grad()` and return `evaluate(x).requires_grad`.",
         f"@torch.no_grad()\ndef evaluate(t):\n    return t * {k}\nresult = evaluate(x).requires_grad"),
        ("grad_fn", "Return `[x.grad_fn is None, (x * 2).grad_fn is None]`: leaves have no grad_fn, results of ops do.",
         "[x.grad_fn is None, (x * 2).grad_fn is None]"),
        ("retain_graph", f"Backprop through the same graph twice: `z = (x * {k}).sum()`, `z.backward(retain_graph=True)`, "
                         f"`z.backward()`. Return `x.grad`.", f"z = (x * {k}).sum()\nz.backward(retain_graph=True)\nz.backward()\nresult = x.grad"),
        ("ones_like", "For the vector `u = x * 2`, call `u.backward(gradient=torch.ones_like(u))` and return `x.grad`.",
         "u = x * 2\nu.backward(gradient=torch.ones_like(u))\nresult = x.grad"),
        ("jacobian", "Return the Jacobian of `torch.sin` at `x.detach()` with `torch.autograd.functional.jacobian`.",
         "torch.autograd.functional.jacobian(torch.sin, x.detach())", {"tol": 1e-5}),
        ("allclose", f"Check autograd against the analytic gradient: backprop `(x ** 3).sum()` and return "
                     f"`torch.allclose(x.grad, 3 * x.detach() ** 2, atol=1e-6)`.", "(x ** 3).sum().backward()\nresult = torch.allclose(x.grad, 3 * x.detach() ** 2, atol=1e-6)"),
        ("shape", "Backprop `(x ** 2).sum()` and return the shape of `x.grad` as a tuple (always the shape of `x`).",
         "(x ** 2).sum().backward()\nresult = tuple(x.grad.shape)"),
    ]
    return catalog(rng, items, setup, "")


@template("3.03", "hard")
def func_transforms_catalog(rng):
    vals = fv(rng, 3, -2, 2, 1, nonzero=True)
    n, d = rng.randint(3, 6), rng.randint(2, 4)
    setup = (f"from torch.func import grad, vmap, jacrev, jacfwd, hessian, functional_call\nt = {ftt(vals)}\n"
             f"f = lambda v: (v ** 3).sum()")
    per_sample = (f"torch.manual_seed({seeded(rng)})\nlin = nn.Linear({d}, 1)\nparams = {{k: p.detach() for k, p in lin.named_parameters()}}\n"
                  f"def loss_fn(params, xi, yi):\n    return F.mse_loss(functional_call(lin, params, (xi.unsqueeze(0),)).squeeze(0), yi)\n"
                  f"per_sample = vmap(grad(loss_fn), in_dims=(None, 0, 0))(params, torch.randn({n}, {d}), torch.randn({n}, 1))\n"
                  f"result = tuple(per_sample['weight'].shape)")
    items = [
        ("grad", "`torch.func.grad(f)` returns a function computing the gradient. Return `grad(f)(t)`.", "grad(f)(t)"),
        ("jacrev", "Return the Jacobian of `torch.sin` at `t` with reverse mode (`jacrev(torch.sin)(t)`).", "jacrev(torch.sin)(t)"),
        ("jacfwd", "Return the Jacobian of `torch.sin` at `t` with forward mode (`jacfwd(torch.sin)(t)`).", "jacfwd(torch.sin)(t)"),
        ("hessian", "Return the Hessian of `f` at `t` (`hessian(f)(t)`).", "hessian(f)(t)", {"tol": 1e-3}),
        ("vmap", "`vmap` maps a per-example function over a batch dim. Return `vmap(lambda r: r.sum() ** 2)(torch.stack([t, 2 * t]))`.",
         "vmap(lambda r: r.sum() ** 2)(torch.stack([t, 2 * t]))"),
        ("functional_call", f"Per-sample gradients for `nn.Linear({d}, 1)`: with `params = {{k: p.detach() for k, p in "
                            f"lin.named_parameters()}}` and a `loss_fn(params, xi, yi)` that runs `functional_call(lin, params, "
                            f"(xi.unsqueeze(0),)).squeeze(0)` through `F.mse_loss`, compute `vmap(grad(loss_fn), in_dims=(None, 0, "
                            f"0))` over {n} examples and return the shape of the weight gradients.", per_sample,
         {"require": ["functional_call", "vmap", "grad"]}),
    ]
    return catalog(rng, items, setup, "")


# ================================================================= 3.04 Dataset & DataLoader

@template("3.04", "medium")
def loader_options_catalog(rng):
    n, b, w = rng.choice([12, 16, 20]), rng.choice([3, 4, 5]), rng.choice([2, 3])
    s = seeded(rng)
    setup = (f"from torch.utils.data import DataLoader, TensorDataset, get_worker_info\n"
             f"ds = TensorDataset(torch.arange({n}.).reshape({n}, 1), torch.arange({n}))")
    items = [
        ("persistent_workers", f"Load `ds` with `DataLoader(ds, batch_size={b}, num_workers={w}, persistent_workers=True, "
                               f"prefetch_factor=2)` (parallel loading that keeps its workers alive between epochs) and return "
                               f"the label batches as lists.",
         f"dl = DataLoader(ds, batch_size={b}, num_workers={w}, persistent_workers=True, prefetch_factor=2)\nresult = [yb.tolist() for _, yb in dl]",
         {"require": ["persistent_workers", "prefetch_factor"]}),
        ("worker_init_fn", f"`worker_init_fn` runs once in each worker: pass one that appends its `worker_id` to a list, "
                           f"iterate `DataLoader(ds, batch_size={b}, num_workers={w}, worker_init_fn=...)` once, and return "
                           f"the sorted ids.", f"seen = []\ndl = DataLoader(ds, batch_size={b}, num_workers={w}, worker_init_fn=lambda i: seen.append(i))\n"
                                               f"list(dl)\nresult = sorted(seen)"),
        ("pin_memory", f"`pin_memory` speeds up host-to-GPU copies, so it's set to `torch.cuda.is_available()` (False here). "
                       f"Return the shape of the first feature batch of `DataLoader(ds, batch_size={b}, pin_memory=torch.cuda."
                       f"is_available())`.", f"dl = DataLoader(ds, batch_size={b}, pin_memory=torch.cuda.is_available())\nresult = tuple(next(iter(dl))[0].shape)"),
        ("initial_seed", f"After `torch.manual_seed({s})`, return `torch.initial_seed()`.", f"torch.manual_seed({s})\nresult = torch.initial_seed()"),
        ("seed", f"`seed_worker` gives NumPy a per-process seed: run `np.random.seed(torch.initial_seed() % 2**32)` after "
                 f"`torch.manual_seed({s})` and return `np.random.randint(0, 100, 3)`.",
         f"torch.manual_seed({s})\nnp.random.seed(torch.initial_seed() % 2**32)\nresult = np.random.randint(0, 100, 3)"),
        ("get_worker_info", f"Inside a worker, `get_worker_info()` describes it. Make a dataset whose `__getitem__` returns "
                            f"`get_worker_info().id`, load all {n} items with `DataLoader(..., batch_size={b}, num_workers={w})`, "
                            f"and return the worker id of each batch (batches go round-robin).",
         f"class Who(torch.utils.data.Dataset):\n    def __len__(self):\n        return {n}\n    def __getitem__(self, i):\n"
         f"        return get_worker_info().id\nresult = [int(bt[0]) for bt in DataLoader(Who(), batch_size={b}, num_workers={w})]"),
    ]
    return catalog(rng, items, setup, "")


# ================================================================= 3.06 Custom autograd Functions

@template("3.06", "hard")
def ctx_tools_catalog(rng):
    vals = fv(rng, 4, -2, 2, 1, nonzero=True)
    setup = f"x = torch.tensor({lit(vals)}, requires_grad=True)"
    items = [
        ("mark_non_differentiable", "Write `class ArgmaxDouble(torch.autograd.Function)` whose forward returns `(x * 2, "
                                    "x.argmax())` and marks the index output with `ctx.mark_non_differentiable(idx)`; backward "
                                    "returns `g_y * 2` (ignore the index's gradient). Return `[idx.requires_grad, y.requires_grad]` "
                                    "for `y, idx = ArgmaxDouble.apply(x)`.",
         "class ArgmaxDouble(torch.autograd.Function):\n    @staticmethod\n    def forward(ctx, x):\n        idx = x.argmax()\n"
         "        ctx.mark_non_differentiable(idx)\n        return x * 2, idx\n    @staticmethod\n    def backward(ctx, g_y, g_idx):\n"
         "        return g_y * 2\ny, idx = ArgmaxDouble.apply(x)\nresult = [idx.requires_grad, y.requires_grad]"),
        ("set_materialize_grads", "With `ctx.set_materialize_grads(False)`, unused outputs send `None` (not zeros) to backward. "
                                  "Write `Split(Function)` returning `(x * 2, x * 3)` that calls it, with a backward that adds "
                                  "`2 * g1` and `3 * g2` only for the gradients that are not None. Backprop only the first output's "
                                  "sum and return `x.grad`.",
         "class Split(torch.autograd.Function):\n    @staticmethod\n    def forward(ctx, x):\n        ctx.set_materialize_grads(False)\n"
         "        return x * 2, x * 3\n    @staticmethod\n    def backward(ctx, g1, g2):\n        out = torch.zeros_like(g1 if g1 is not None else g2)\n"
         "        if g1 is not None:\n            out = out + 2 * g1\n        if g2 is not None:\n            out = out + 3 * g2\n        return out\n"
         "a, b = Split.apply(x)\na.sum().backward()\nresult = x.grad"),
        ("mark_dirty", "An in-place Function must call `ctx.mark_dirty(t)`. Write `InplaceDouble` whose forward does `t.mul_(2)`, "
                       "marks it dirty and returns it; backward returns `g * 2`. Apply it to `t = x.detach().clone().requires_grad_"
                       "(True) * 1` (a non-leaf) and return the output.",
         "class InplaceDouble(torch.autograd.Function):\n    @staticmethod\n    def forward(ctx, t):\n        t.mul_(2)\n"
         "        ctx.mark_dirty(t)\n        return t\n    @staticmethod\n    def backward(ctx, g):\n        return g * 2\n"
         "t = x.detach().clone().requires_grad_(True) * 1\nresult = InplaceDouble.apply(t).detach()"),
        ("Function", "Subclass the full path `torch.autograd.Function`: write `Cube` (forward saves x and returns `x ** 3`, "
                     "backward returns `3 x² g`), backprop `Cube.apply(x).sum()` and return `x.grad`.",
         "class Cube(torch.autograd.Function):\n    @staticmethod\n    def forward(ctx, x):\n        ctx.save_for_backward(x)\n"
         "        return x ** 3\n    @staticmethod\n    def backward(ctx, g):\n        (x,) = ctx.saved_tensors\n        return 3 * x ** 2 * g\n"
         "Cube.apply(x).sum().backward()\nresult = x.grad", {"require": ["autograd", "Function"]}),
        ("double", "`gradcheck` needs float64 inputs: make `xd = x.detach().to(torch.double).requires_grad_(True)` and return "
                   "`torch.autograd.gradcheck(torch.sin, (xd,))`.",
         "xd = x.detach().to(torch.double).requires_grad_(True)\nresult = torch.autograd.gradcheck(torch.sin, (xd,))"),
        ("func", "A new-style Function (forward without ctx, plus `setup_context`) works with `torch.func`. Write `Exp` that way "
                 "and return `torch.func.vmap(Exp.apply)(torch.stack([x.detach(), -x.detach()]))`.",
         "class Exp(torch.autograd.Function):\n    generate_vmap_rule = True\n    @staticmethod\n    def forward(x):\n        return x.exp()\n"
         "    @staticmethod\n    def setup_context(ctx, inputs, output):\n        ctx.save_for_backward(output)\n    @staticmethod\n"
         "    def backward(ctx, g):\n        (y,) = ctx.saved_tensors\n        return g * y\n"
         "result = torch.func.vmap(Exp.apply)(torch.stack([x.detach(), -x.detach()]))", {"require": ["vmap", "setup_context"]}),
    ]
    return catalog(rng, items, setup, "")


# ================================================================= 3.07 Memory

@template("3.07", "lite")
def memory_misc_catalog(rng):
    r, c = rng.randint(2, 5), rng.randint(3, 6)
    dt = rng.choice(["float32", "float64", "int16"])
    setup = f"x = torch.zeros({r}, {c}, dtype=torch.{dt})"
    items = [
        ("nelement", "Return the bytes `x` uses: `x.element_size() * x.nelement()` (`nelement` is `numel`).", "x.element_size() * x.nelement()"),
        ("unfold", "Return the shape of the sliding windows `x.unfold(dimension=1, size=2, step=1)` as a tuple (a view).",
         "tuple(x.unfold(dimension=1, size=2, step=1).shape)", {"require": ["unfold", "dimension"]}),
        ("shape", "Return the shape of `x.t()` as a tuple: same memory, swapped strides.", "tuple(x.t().shape)"),
        ("stride", "Return `x.t().stride()`.", "x.t().stride()"),
    ]
    return catalog(rng, items, setup, f"`x` is a contiguous {r}×{c} {dt} tensor.")


# ================================================================= 3.08 Attention

@template("3.08", "medium")
def attention_api_catalog(rng):
    E, H, Tn = rng.choice([8, 16]), rng.choice([2, 4]), rng.randint(3, 6)
    setup = (f"torch.manual_seed({seeded(rng)})\nxq = torch.randn(2, {Tn}, {E})\nq, k, v = (torch.randn(2, {H}, {Tn}, {E // H}) for _ in range(3))")
    mha = f"mha = nn.MultiheadAttention(embed_dim={E}, num_heads={H}, batch_first=True)"
    flex = "from torch.nn.attention.flex_attention import flex_attention, create_block_mask\n"
    items = [
        ("MultiheadAttention", f"Run self-attention with `{mha}`: `y, weights = mha(xq, xq, xq, need_weights=True)`. Return the "
                               f"shapes of `y` and `weights` (averaged over heads) as tuples.",
         f"{mha}\ny, weights = mha(xq, xq, xq, need_weights=True)\nresult = [tuple(y.shape), tuple(weights.shape)]",
         {"require": ["MultiheadAttention", "need_weights"]}),
        ("generate_square_subsequent_mask", f"Return `nn.Transformer.generate_square_subsequent_mask({Tn})`: the float causal "
                                            f"mask (0 on and below the diagonal, -inf above) that `nn.MultiheadAttention` takes.",
         f"nn.Transformer.generate_square_subsequent_mask({Tn})"),
        ("key_padding_mask", f"With `{mha}`, attend causally with `attn_mask=nn.Transformer.generate_square_subsequent_mask({Tn})` "
                             f"and no padding (`key_padding_mask=torch.zeros(2, {Tn}, dtype=torch.bool)`; True would mean "
                             f"ignore). Return the output shape.",
         f"{mha}\ny, _ = mha(xq, xq, xq, attn_mask=nn.Transformer.generate_square_subsequent_mask({Tn}), "
         f"key_padding_mask=torch.zeros(2, {Tn}, dtype=torch.bool))\nresult = tuple(y.shape)", {"require": ["key_padding_mask", "zeros"]}),
        ("MATH", "Force the math backend for `F.scaled_dot_product_attention` with `sdpa_kernel(SDPBackend.MATH)` (from "
                 "`torch.nn.attention`) and return the causal attention output.",
         "from torch.nn.attention import sdpa_kernel, SDPBackend\nwith sdpa_kernel(SDPBackend.MATH):\n"
         "    result = F.scaled_dot_product_attention(q, k, v, is_causal=True)", {"require": ["sdpa_kernel", "MATH"]}),
        ("flex_attention", f"FlexAttention expresses masks as functions: build `create_block_mask(lambda b, h, q_idx, kv_idx: "
                           f"q_idx >= kv_idx, B=None, H=None, Q_LEN={Tn}, KV_LEN={Tn}, device='cpu')` and return whether "
                           f"`flex_attention(q, k, v, block_mask=...)` matches causal SDPA (`torch.allclose(..., atol=1e-5)`).",
         flex + f"bm = create_block_mask(lambda b, h, q_idx, kv_idx: q_idx >= kv_idx, B=None, H=None, Q_LEN={Tn}, KV_LEN={Tn}, device='cpu')\n"
                f"result = torch.allclose(flex_attention(q, k, v, block_mask=bm), F.scaled_dot_product_attention(q, k, v, is_causal=True), atol=1e-5)",
         {"require": ["flex_attention", "block_mask"]}),
        ("score_mod", "`score_mod(score, b, h, q_idx, kv_idx)` edits each raw score. Return `flex_attention(q, k, v, "
                      "score_mod=lambda s, b, h, qi, ki: s + (qi - ki) * 0.1)`: a relative-position bias.",
         flex + "result = flex_attention(q, k, v, score_mod=lambda s, b, h, qi, ki: s + (qi - ki) * 0.1)", {"require": ["flex_attention", "score_mod"]}),
        ("dropout_p", "Return `F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, scale=0.5)` (dropout only during "
                      "training; scale overrides 1/sqrt(D)).", "F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, scale=0.5)"),
        ("contiguous", "Merge heads: return `q.transpose(1, 2).contiguous().view(2, T, H * D)`.",
         f"q.transpose(1, 2).contiguous().view(2, {Tn}, {E})"),
        ("allclose", "Implement causal attention by hand and return whether it matches `F.scaled_dot_product_attention(q, k, v, "
                     "is_causal=True)` with `torch.allclose(..., atol=1e-5)`.",
         f"causal = torch.triu(torch.ones({Tn}, {Tn}, dtype=torch.bool), diagonal=1)\n"
         f"out = (q @ k.transpose(-2, -1) / math.sqrt({E // H})).masked_fill(causal, float('-inf')).softmax(-1) @ v\n"
         f"result = torch.allclose(out, F.scaled_dot_product_attention(q, k, v, is_causal=True), atol=1e-5)"),
    ]
    return catalog(rng, items, setup, "")


# ================================================================= 3.09 Transformer block

@template("3.09", "medium")
def transformer_api_catalog(rng):
    d, h, V, L, Tn = rng.choice([16, 32]), rng.choice([2, 4]), rng.choice([50, 100]), rng.randint(1, 3), rng.randint(4, 8)
    setup = f"torch.manual_seed({seeded(rng)})\nidx = torch.randint(0, {V}, (2, {Tn}))"
    items = [
        ("TransformerEncoderLayer", f"Build `layer = nn.TransformerEncoderLayer(d_model={d}, nhead={h}, dim_feedforward={4 * d}, "
                                    f"dropout=0.1, activation='gelu', batch_first=True, norm_first=True)` and "
                                    f"`nn.TransformerEncoder(layer, num_layers={L})`, and return the output shape for "
                                    f"`torch.randn(2, {Tn}, {d})`.",
         f"layer = nn.TransformerEncoderLayer(d_model={d}, nhead={h}, dim_feedforward={4 * d}, dropout=0.1, activation='gelu', "
         f"batch_first=True, norm_first=True)\nenc = nn.TransformerEncoder(layer, num_layers={L})\nresult = tuple(enc(torch.randn(2, {Tn}, {d})).shape)",
         {"require": ["TransformerEncoderLayer", "TransformerEncoder", "norm_first"]}),
        ("Embedding", f"Token + position embeddings: `tok = nn.Embedding({V}, {d})`, `pos = nn.Embedding(256, {d})`; return the "
                      f"shape of `tok(idx) + pos(torch.arange({Tn}, device=idx.device))`.",
         f"tok, pos = nn.Embedding({V}, {d}), nn.Embedding(256, {d})\nresult = tuple((tok(idx) + pos(torch.arange({Tn}, device=idx.device))).shape)",
         {"require": ["Embedding", "device"]}),
        ("Parameter", f"Weight tying: `tok = nn.Embedding({V}, {d})`, `head = nn.Linear({d}, {V}, bias=False)`, then "
                      f"`head.weight = tok.weight`. Put both in an `nn.ModuleList` and return the parameter count "
                      f"(`numel` over `parameters()`, which counts the shared tensor once).",
         f"tok = nn.Embedding({V}, {d})\nhead = nn.Linear({d}, {V}, bias=False)\nhead.weight = tok.weight\n"
         f"both = nn.ModuleList([tok, head])\nresult = sum(p.numel() for p in both.parameters())", {"require": ["ModuleList", "numel", "parameters"]}),
        ("Parameter", f"An RMSNorm keeps a learnable gain: return the shape and `requires_grad` of `nn.Parameter(torch.ones({d}))`.",
         f"w = nn.Parameter(torch.ones({d}))\nresult = [tuple(w.shape), w.requires_grad]", {"require": ["Parameter", "ones"]}),
        ("multinomial", f"Sample the next token: with `logits = torch.randn(2, {V})`, return the shape of "
                        f"`torch.multinomial(logits.softmax(-1), num_samples=1)`.",
         f"logits = torch.randn(2, {V})\nresult = tuple(torch.multinomial(logits.softmax(-1), num_samples=1).shape)",
         {"require": ["multinomial", "num_samples"]}),
        ("long", "Greedy generation starts from a single token: return the dtype and shape of `torch.zeros(1, 1, dtype=torch.long)`.",
         "start = torch.zeros(1, 1, dtype=torch.long)\nresult = [start.dtype, tuple(start.shape)]"),
        ("size", f"With `logits = torch.randn(2, {Tn}, {V})`, flatten for the loss: return the shape of "
                 f"`logits.reshape(-1, logits.size(-1))`.", f"logits = torch.randn(2, {Tn}, {V})\nresult = tuple(logits.reshape(-1, logits.size(-1)).shape)"),
        ("eval", f"`nn.Dropout(0.5)` is the identity in eval mode. Return `d.eval()(torch.ones(3))` for `d = nn.Dropout(0.5)`.",
         "d = nn.Dropout(0.5)\nresult = d.eval()(torch.ones(3))", {"require": ["Dropout", "eval"]}),
    ]
    return catalog(rng, items, setup, "")


# ================================================================= 3.10 Hooks & debugging

@template("3.10", "medium")
def debug_catalog(rng):
    a, h = rng.randint(3, 6), rng.randint(4, 8)
    setup = f"torch.manual_seed({seeded(rng)})\nnet = nn.Sequential(nn.Linear({a}, {h}), nn.ReLU(), nn.Linear({h}, 2))\nx = torch.randn(4, {a})"
    items = [
        ("clip_grad_norm_", "Measure the global gradient norm **without clipping**: backprop `net(x).sum()` and return "
                            "`nn.utils.clip_grad_norm_(net.parameters(), max_norm=float('inf'))`.",
         "net(x).sum().backward()\nresult = nn.utils.clip_grad_norm_(net.parameters(), max_norm=float('inf'))", {"tol": 1e-4}),
        ("isnan", "Health check: return `[torch.isfinite(net[0].weight).all().item(), torch.isnan(net[0].weight).any().item()]`.",
         "[torch.isfinite(net[0].weight).all().item(), torch.isnan(net[0].weight).any().item()]", {"require": ["isnan", "any"]}),
        ("set_detect_anomaly", "Turn on `torch.autograd.set_detect_anomaly(True)`; backprop `(torch.sqrt(z) * 0).sum()` for "
                               "`z = torch.tensor([0., 1.], requires_grad=True)` (its gradient at 0 is NaN). Return True if anomaly "
                               "mode raised a RuntimeError, then switch it off again.",
         "torch.autograd.set_detect_anomaly(True)\nz = torch.tensor([0., 1.], requires_grad=True)\ntry:\n"
         "    (torch.sqrt(z) * 0).sum().backward()\n    result = False\nexcept RuntimeError:\n    result = True\n"
         "torch.autograd.set_detect_anomaly(False)"),
        ("with_kwargs", "Register a forward hook on `net[0]` with `with_kwargs=True` (signature `hook(module, args, kwargs, "
                        "output)`) that records `len(kwargs)` and `tuple(output.shape)`; run `net(x)` and return the record.",
         "rec = []\nhd = net[0].register_forward_hook(lambda m, args, kwargs, out: rec.append((len(kwargs), tuple(out.shape))), with_kwargs=True)\n"
         "net(x)\nhd.remove()\nresult = rec[0]", {"require": ["register_forward_hook", "with_kwargs"]}),
        ("gradcheck", "Return `torch.autograd.gradcheck(torch.sin, (torch.randn(4, dtype=torch.float64, requires_grad=True),))`.",
         "torch.autograd.gradcheck(torch.sin, (torch.randn(4, dtype=torch.float64, requires_grad=True),))",
         {"require": ["gradcheck", "float64", "sin"]}),
        ("parameters", "After backprop of `net(x).sum()`, return each parameter's name with its gradient norm rounded to 3 "
                       "decimals (`named_parameters()`).", "net(x).sum().backward()\nresult = {n: round(p.grad.norm().item(), 3) for n, p in net.named_parameters()}",
         {"require": ["named_parameters", "grad"]}),
    ]
    return catalog(rng, items, setup, "")


# ================================================================= 3.11 KV cache

@template("3.11", "easy")
def kv_api_catalog(rng):
    B, H, L, D, Tn = 1, rng.choice([2, 4]), rng.randint(8, 16), rng.choice([4, 8]), rng.randint(1, 3)
    setup = (f"class StaticKVCache:\n    def __init__(self, B, H, max_len, Dh, dtype=torch.float32):\n"
             f"        self.k = torch.zeros(B, H, max_len, Dh, dtype=dtype)\n        self.v = torch.zeros_like(self.k)\n        self.pos = 0\n"
             f"    def update(self, k_new, v_new):\n        Tn = k_new.size(2)\n"
             f"        self.k[:, :, self.pos:self.pos + Tn] = k_new\n        self.v[:, :, self.pos:self.pos + Tn] = v_new\n"
             f"        self.pos += Tn\n        return self.k[:, :, :self.pos], self.v[:, :, :self.pos]\n\n"
             f"torch.manual_seed({seeded(rng)})\nkv = StaticKVCache({B}, {H}, {L}, {D})")
    items = [
        ("size", f"Write {Tn} new positions with `kk, vv = kv.update(torch.randn({B}, {H}, {Tn}, {D}), torch.randn({B}, {H}, {Tn}, {D}))` "
                 f"and return `[kk.size(2), kv.pos]`.", f"kk, vv = kv.update(torch.randn({B}, {H}, {Tn}, {D}), torch.randn({B}, {H}, {Tn}, {D}))\n"
                                                       f"result = [kk.size(2), kv.pos]"),
        ("zeros_like", "Return the shape and dtype of `torch.zeros_like(kv.k)`, the buffer the value cache is allocated as.",
         "z = torch.zeros_like(kv.k)\nresult = [tuple(z.shape), z.dtype]"),
        ("float32", f"Return the dtype of a cache built with `StaticKVCache({B}, {H}, {L}, {D}, dtype=torch.float32)`.",
         f"result = StaticKVCache({B}, {H}, {L}, {D}, dtype=torch.float32).k.dtype"),
        ("allclose", f"Two updates of 1 and {Tn} positions must leave the same keys as writing them at once: return "
                     f"`torch.allclose(...)` of the two caches' keys.",
         f"a, b = torch.randn({B}, {H}, 1, {D}), torch.randn({B}, {H}, {Tn}, {D})\nkv.update(a, a)\nk1, _ = kv.update(b, b)\n"
         f"kv2 = StaticKVCache({B}, {H}, {L}, {D})\nk2, _ = kv2.update(torch.cat([a, b], dim=2), torch.cat([a, b], dim=2))\n"
         f"result = torch.allclose(k1, k2, atol=1e-6)", {"require": ["allclose", "atol"]}),
    ]
    return catalog(rng, items, setup, "")


# ================================================================= 3.12 LoRA

@template("3.12", "medium")
def lora_api_catalog(rng):
    d, o, r = rng.choice([8, 16]), rng.choice([4, 8]), rng.choice([2, 4])
    setup = (f"class LoRALinear(nn.Module):\n    def __init__(self, base, r=8, alpha=16, dropout=0.0):\n        super().__init__()\n"
             f"        self.base = base\n        for p in self.base.parameters():\n            p.requires_grad_(False)\n"
             f"        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.1)\n        self.B = nn.Parameter(torch.zeros(base.out_features, r))\n"
             f"        self.scale, self.drop = alpha / r, nn.Dropout(dropout)\n    def forward(self, x):\n"
             f"        return self.base(x) + (self.drop(x) @ self.A.T @ self.B.T) * self.scale\n\n"
             f"torch.manual_seed({seeded(rng)})\nlayer = LoRALinear(nn.Linear({d}, {o}), r={r}, dropout=0.1)")
    items = [
        ("AdamW", "Optimize **only** the adapters: return the number of tensors and of values handed to "
                  "`torch.optim.AdamW([p for p in layer.parameters() if p.requires_grad], lr=1e-3)`.",
         "opt = torch.optim.AdamW([p for p in layer.parameters() if p.requires_grad], lr=1e-3)\n"
         "ps = opt.param_groups[0]['params']\nresult = [len(ps), sum(p.numel() for p in ps)]", {"require": ["AdamW", "requires_grad"]}),
        ("Dropout", f"The adapter path has `nn.Dropout`. With B still zero the output equals the base layer's in any mode; "
                    f"return `torch.allclose(layer.eval()(x), layer.base(x))` for `x = torch.ones(2, {d})`.",
         f"x = torch.ones(2, {d})\nresult = torch.allclose(layer.eval()(x), layer.base(x))", {"require": ["eval", "allclose"]}),
        ("Dropout", "Return the dropout probability stored in `layer.drop` (an `nn.Dropout`).", "layer.drop.p", {"require": ["drop"]}),
    ]
    return catalog(rng, items, setup, "")


# ================================================================= 3.13 AMP

@template("3.13", "easy")
def amp_api_catalog(rng):
    S = rng.choice([1024.0, 65536.0, 256.0])
    setup = f"scaler = torch.amp.GradScaler('cpu', init_scale={S})"
    items = [
        ("set_float32_matmul_precision", "Allow TF32 tensor cores for fp32 matmuls with `torch.set_float32_matmul_precision"
                                         "('high')`, return `torch.get_float32_matmul_precision()`, then set it back to "
                                         "'highest'.", "torch.set_float32_matmul_precision('high')\nresult = torch.get_float32_matmul_precision()\n"
                                                       "torch.set_float32_matmul_precision('highest')"),
        ("state_dict", "Checkpoints should include the scaler: return `scaler.state_dict()['scale']`.", "scaler.state_dict()['scale']"),
        ("set_to_none", "After an AMP step, clear gradients with `opt.zero_grad(set_to_none=True)`: for `w = nn.Parameter("
                        "torch.ones(2))` and `opt = torch.optim.SGD([w], lr=0.1)`, run `scaler.scale((w * 2).sum()).backward()`, "
                        "`scaler.step(opt)`, `scaler.update()`, `opt.zero_grad(set_to_none=True)`, and return `w.grad is None`.",
         "w = nn.Parameter(torch.ones(2))\nopt = torch.optim.SGD([w], lr=0.1)\nscaler.scale((w * 2).sum()).backward()\nscaler.step(opt)\n"
         "scaler.update()\nopt.zero_grad(set_to_none=True)\nresult = w.grad is None"),
        ("zeros", "Create tensors directly on the target device: return the device type of `torch.zeros(2, device='cpu')`.",
         "torch.zeros(2, device='cpu').device.type", {"require": ["zeros", "device"]}),
        ("device", "Pick the autocast device type the portable way: `dev_type = 'cuda' if torch.cuda.is_available() else 'cpu'`, "
                   "then return the dtype of `torch.ones(2, 2) @ torch.ones(2, 2)` computed under `torch.autocast(device_type="
                   "dev_type, dtype=torch.bfloat16)`.",
         "dev_type = 'cuda' if torch.cuda.is_available() else 'cpu'\nwith torch.autocast(device_type=dev_type, dtype=torch.bfloat16):\n"
         "    out = torch.ones(2, 2) @ torch.ones(2, 2)\nresult = out.dtype", {"require": ["autocast", "device_type"]}),
    ]
    return catalog(rng, items, setup, "")


# ================================================================= 3.14 Profiler

@template("3.14", "medium")
def profiler_api_catalog(rng):
    d = rng.choice([16, 32])
    k = rng.randint(1, 3)
    setup = (f"import json, os\nfrom torch.profiler import profile, record_function, ProfilerActivity\ntorch.manual_seed(0)\n"
             f"model = nn.Sequential(nn.Linear({d}, {d}), nn.ReLU(), nn.Linear({d}, 2))\nxp = torch.randn(8, {d})\n"
             f"acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if torch.cuda.is_available() else [])")
    run = f"with profile(activities=acts{{}}) as prof:\n    for _ in range({k}):\n        model(xp)\n"
    items = [
        ("export_chrome_trace", "Profile the forward pass and save a trace with `prof.export_chrome_trace('trace.json')` (open "
                                "it in Perfetto). Return whether the file exists and parses as JSON with a `traceEvents` list.",
         run.format("") + "prof.export_chrome_trace('trace.json')\nresult = os.path.exists('trace.json') and isinstance(json.load(open('trace.json'))['traceEvents'], list)"),
        ("table", "Print `prof.key_averages().table(sort_by='cpu_time_total', row_limit=5)` and return whether the table mentions "
                  "`aten::linear`.", run.format("") + "t = prof.key_averages().table(sort_by='cpu_time_total', row_limit=5)\nresult = 'aten::linear' in t",
         {"require": ["table", "sort_by", "row_limit"]}),
        ("record_shapes", "Profile with `record_shapes=True, profile_memory=True` and return the call count of `aten::linear` "
                          "from `prof.key_averages(group_by_input_shape=True)`.",
         run.format(", record_shapes=True, profile_memory=True") + "result = sum(e.count for e in prof.key_averages(group_by_input_shape=True) if e.key == 'aten::linear')",
         {"require": ["record_shapes", "group_by_input_shape"]}),
        ("CUDA", "Return `acts`: the activities list built portably (CUDA only when a GPU exists).", "acts", {"require": ["acts"]}),
        ("backward", "Profile a full step: inside `profile(activities=acts)`, run `model(xp).sum().backward()` and return the "
                     "number of `aten::linear` calls recorded.",
         "with profile(activities=acts) as prof:\n    model(xp).sum().backward()\nresult = sum(e.count for e in prof.key_averages() if e.key == 'aten::linear')",
         {"require": ["backward", "sum"]}),
        ("sum", f"Total the calls: return the sum of `e.count` over all ops in `prof.key_averages()` whose name starts with 'aten::'.",
         run.format("") + "result = sum(e.count for e in prof.key_averages() if e.key.startswith('aten::'))"),
    ]
    return catalog(rng, items, setup, "")


# ================================================================= 3.15 Triton

@template("3.15", "hard")
def triton_api_catalog(rng):
    n = rng.randint(20, 120)
    setup = f"import triton\nimport triton.language as tl\ntorch.manual_seed({seeded(rng)})\nx = torch.randn({n})"
    items = [
        ("autotune", "Autotuning picks a block size per input size: decorate a masked `x * 2` kernel with `@triton.autotune("
                     "configs=[triton.Config({'BLOCK': 32}, num_warps=4), triton.Config({'BLOCK': 64}, num_warps=4)], "
                     "key=['n'])` above `@triton.jit`, launch it with `grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)` "
                     "(don't pass BLOCK yourself), and return the output.",
         "@triton.autotune(configs=[triton.Config({'BLOCK': 32}, num_warps=4), triton.Config({'BLOCK': 64}, num_warps=4)], key=['n'])\n"
         "@triton.jit\ndef double(x_ptr, out_ptr, n, BLOCK: tl.constexpr):\n    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)\n"
         "    m = offs < n\n    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=m) * 2, mask=m)\nout = torch.empty_like(x)\n"
         f"double[lambda meta: (triton.cdiv({n}, meta['BLOCK']),)](x, out, {n})\nresult = out", {"require": ["autotune", "Config", "configs"]}),
        ("cumsum", "Write a one-program kernel that stores `tl.cumsum` of the (masked, `other=0.0`) whole input with "
                   f"`BLOCK = triton.next_power_of_2({n})`, and return the output (it must equal `torch.cumsum(x, 0)`).",
         "@triton.jit\ndef cs(x_ptr, out_ptr, n, BLOCK: tl.constexpr):\n    offs = tl.arange(0, BLOCK)\n    m = offs < n\n"
         "    tl.store(out_ptr + offs, tl.cumsum(tl.load(x_ptr + offs, mask=m, other=0.0), axis=0), mask=m)\nout = torch.empty_like(x)\n"
         f"cs[(1,)](x, out, {n}, BLOCK=triton.next_power_of_2({n}))\nresult = out"),
        ("where", "Write a kernel storing `tl.where(x > 0, tl.sqrt(x), -x)` with masked loads/stores (BLOCK=32) and return the output.",
         "@triton.jit\ndef k(x_ptr, out_ptr, n, BLOCK: tl.constexpr):\n    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)\n"
         "    m = offs < n\n    v = tl.load(x_ptr + offs, mask=m, other=1.0)\n    tl.store(out_ptr + offs, tl.where(v > 0, tl.sqrt(v), -v), mask=m)\n"
         f"out = torch.empty_like(x)\nk[(triton.cdiv({n}, 32),)](x, out, {n}, BLOCK=32)\nresult = out", {"require": ["where", "sqrt"]}),
        ("assert_close", "Test a kernel the standard way: write a masked `x + 1` kernel (BLOCK=64), run it, and call "
                         "`torch.testing.assert_close(out, x + 1)` inside try/except AssertionError. Return True if it passed.",
         "@triton.jit\ndef inc(x_ptr, out_ptr, n, BLOCK: tl.constexpr):\n    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)\n"
         "    m = offs < n\n    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=m) + 1, mask=m)\nout = torch.empty_like(x)\n"
         f"inc[(triton.cdiv({n}, 64),)](x, out, {n}, BLOCK=64)\ntry:\n    torch.testing.assert_close(out, x + 1)\n    result = True\n"
         f"except AssertionError:\n    result = False"),
        ("do_bench", "`triton.testing.do_bench(fn)` times a callable (milliseconds). Return whether `do_bench(lambda: x * 2)` "
                     "returns a non-negative float (timings vary, so only that is graded).",
         "ms = triton.testing.do_bench(lambda: x * 2)\nresult = isinstance(ms, float) and ms >= 0", {"require": ["do_bench"]}),
    ]
    return catalog(rng, items, setup, "", tol=1e-5)


# ================================================================= 3.16 DDP

@template("3.16", "medium")
def dist_api_catalog(rng):
    vals = fv(rng, 3, 0, 5, 1)
    setup = ("import torch.distributed as dist\ndist.init_process_group(backend='gloo', init_method='tcp://127.0.0.1:29500', "
             f"world_size=1, rank=0)\nworld = dist.get_world_size()\nt = {ftt(vals)}")
    items = [
        ("broadcast", "Send rank 0's tensor to every rank in place: `dist.broadcast(t, src=0)`. Return `t`.", "dist.broadcast(t, src=0)\nresult = t",
         {"require": ["broadcast", "src"]}),
        ("reduce", "Sum `t` onto rank 0 only: `dist.reduce(t, dst=0)`. Return `t` (on rank 0 it holds the sum).", "dist.reduce(t, dst=0)\nresult = t",
         {"require": ["reduce", "dst"]}),
        ("all_gather", "Collect every rank's `t`: `out = [torch.zeros_like(t) for _ in range(world)]`, `dist.all_gather(out, t)`. "
                       "Return `out`.", "out = [torch.zeros_like(t) for _ in range(world)]\ndist.all_gather(out, t)\nresult = out"),
        ("all_gather_into_tensor", "Gather into one flat tensor: `out = torch.empty(world * t.numel())`, "
                                   "`dist.all_gather_into_tensor(out, t)`. Return `out`.",
         "out = torch.empty(world * t.numel())\ndist.all_gather_into_tensor(out, t)\nresult = out"),
        ("clone", "Average a logged loss across ranks without touching the graph: `avg = t.detach().clone()`, "
                  "`dist.all_reduce(avg, op=dist.ReduceOp.SUM)`, `avg /= world`. Return `avg`.",
         "avg = t.detach().clone()\ndist.all_reduce(avg, op=dist.ReduceOp.SUM)\navg /= world\nresult = avg", {"require": ["clone", "all_reduce"]}),
        ("broadcast_object_list", "Share Python objects from rank 0: `objs = [{'lr': 0.1}]`, `dist.broadcast_object_list(objs, "
                                  "src=0)`. Return `objs`.", "objs = [{'lr': 0.1}]\ndist.broadcast_object_list(objs, src=0)\nresult = objs"),
        ("convert_sync_batchnorm", "Return the class names of the modules in `nn.SyncBatchNorm.convert_sync_batchnorm(model)` "
                                   "for `model = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4))`.",
         "model = nn.SyncBatchNorm.convert_sync_batchnorm(nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4)))\nresult = [type(m).__name__ for m in model]"),
        ("device", "Pick the device per rank: `dev = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')`. "
                   "Return `[dev.type, t.to(dev).device.type]`.",
         "dev = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')\nresult = [dev.type, t.to(dev).device.type]",
         {"require": ["device", "to"]}),
        ("device_ids", "Wrap a model for DDP on CPU: `device_ids=None` unless a GPU is used. Return the first state-dict key of "
                       "`DDP(nn.Linear(3, 1), device_ids=None)`.",
         "from torch.nn.parallel import DistributedDataParallel as DDP\nresult = next(iter(DDP(nn.Linear(3, 1), device_ids=None).state_dict()))",
         {"require": ["DistributedDataParallel|DDP", "device_ids"]}),
    ]
    return catalog(rng, items, setup, "")


# ================================================================= 3.17 FSDP & checkpointing

@template("3.17", "medium")
def fsdp_api_catalog(rng):
    d = rng.choice([4, 8])
    setup = ("import torch.distributed as dist\nfrom torch.distributed.fsdp import FullyShardedDataParallel, ShardingStrategy, MixedPrecision\n"
             "from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict\n"
             "dist.init_process_group(backend='gloo', init_method='tcp://127.0.0.1:29500', world_size=1, rank=0)\n"
             f"torch.manual_seed(0)\nmodel = nn.Linear({d}, 2)\nopt = torch.optim.AdamW(model.parameters(), lr=1e-3)")
    items = [
        ("FULL_SHARD", "Legacy FSDP1: wrap `model` with `FullyShardedDataParallel(model, sharding_strategy=ShardingStrategy."
                       "FULL_SHARD, mixed_precision=MixedPrecision(param_dtype=torch.bfloat16), auto_wrap_policy=None)` and return "
                       "the strategy's name.",
         "fm = FullyShardedDataParallel(model, sharding_strategy=ShardingStrategy.FULL_SHARD, mixed_precision=MixedPrecision(param_dtype=torch.bfloat16), "
         "auto_wrap_policy=None)\nresult = fm.sharding_strategy.name", {"require": ["FULL_SHARD", "sharding_strategy", "mixed_precision"]}),
        ("set_state_dict", "Round-trip through the DCP helpers: `msd, osd = get_state_dict(model, opt)`, zero the model's weights, "
                           "then `set_state_dict(model, opt, model_state_dict=msd_copy, optim_state_dict=osd)` with a copy made "
                           "before zeroing. Return `model.weight.detach()`.",
         "msd, osd = get_state_dict(model, opt)\nsaved = {k: v.clone() for k, v in msd.items()}\nwith torch.no_grad():\n"
         "    model.weight.zero_()\nset_state_dict(model, opt, model_state_dict=saved, optim_state_dict=osd)\nresult = model.weight.detach()",
         {"require": ["set_state_dict", "model_state_dict", "optim_state_dict"]}),
    ]
    return catalog(rng, items, setup, "")


# ================================================================= 3.18 torch.compile

@template("3.18", "easy")
def compile_api_catalog(rng):
    n = rng.randint(3, 8)
    setup = f"x = torch.randn(4, {n})\n@torch.compile\ndef f(t):\n    return (t * 2).relu().sum(-1)"
    items = [
        ("mark_dynamic", "Mark dim 0 of `x` as dynamic before calling (`torch._dynamo.mark_dynamic(x, 0)`) so other batch sizes "
                         "don't recompile, then return `f(x)`.", "torch._dynamo.mark_dynamic(x, 0)\nresult = f(x)"),
        ("reset", "Clear compiled caches with `torch.compiler.reset()` and return `f(x)` (it recompiles transparently).",
         "torch.compiler.reset()\nresult = f(x)"),
        ("graph_count", "Return `[e.graph_count, e.graph_break_count]` for `e = torch._dynamo.explain(f)(x)`.",
         "e = torch._dynamo.explain(f)(x)\nresult = [e.graph_count, e.graph_break_count]", {"require": ["explain", "graph_count"]}),
    ]
    return catalog(rng, items, setup, "")
