"""3.05 MLP training: catalog templates for layers, activations, functional ops, losses, optimizers, schedulers,
the nn.Module API, weight init and saving."""

from ._common import catalog, fm, ftt, fv, lit, seeded, template, tt

M = "3.05"


def _fmat(rows):
    return "torch.tensor(" + lit([[float(v) for v in r] for r in rows]) + ")"


# ================================================================= lite

@template(M, "lite")
def layer_shapes_catalog(rng):
    N, C, H, W, L = rng.randint(1, 4), rng.choice([1, 3]), rng.choice([8, 16, 32]), rng.choice([8, 16, 32]), rng.choice([10, 20])
    d, T, V = rng.choice([4, 8, 16]), rng.randint(3, 7), rng.choice([50, 100, 1000])
    setup = (f"img = torch.zeros({N}, {C}, {H}, {W})   # (N, C, H, W)\nseq = torch.zeros({N}, {C}, {L})   # (N, C, L)\n"
             f"x = torch.zeros({N}, {T}, {d})   # (batch, time, features)\nids = torch.zeros({N}, {T}, dtype=torch.long)")
    sh = "tuple({}.shape)"
    items = [
        ("Conv1d", f"Return the output shape of `nn.Conv1d({C}, 8, kernel_size=3, padding=1)` on `seq`.",
         sh.format(f"nn.Conv1d({C}, 8, kernel_size=3, padding=1)(seq)"), {"require": ["Conv1d", "kernel_size"]}),
        ("Conv2d", f"Return the output shape of `nn.Conv2d({C}, 16, kernel_size=3, stride=2, padding=1)` on `img` "
                   f"(out = ⌊(H + 2p − k) / s⌋ + 1).", sh.format(f"nn.Conv2d({C}, 16, kernel_size=3, stride=2, padding=1)(img)"),
         {"require": ["Conv2d", "stride"]}),
        ("Conv2d", f"Return the output shape of `nn.Conv2d({C}, 8, 3, padding='same', dilation=2, groups=1)` on `img`.",
         sh.format(f"nn.Conv2d({C}, 8, 3, padding='same', dilation=2, groups=1)(img)"), {"require": ["Conv2d", "dilation"]}),
        ("pack_padded_sequence", f"Let the RNN skip padding: pack `x` with lengths `[{T}] + [{max(1, T - 2)}] * {N - 1}` using "
                                 f"`pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)`, run "
                                 f"`nn.LSTM({d}, 5, batch_first=True)` on it, unpack with `pad_packed_sequence(out, batch_first=True)` "
                                 f"and return `[tuple(padded.shape), lens.tolist()]`.",
         f"from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence\nlengths = torch.tensor([{T}] + [{max(1, T - 2)}] * {N - 1})\n"
         f"packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)\nout, _ = nn.LSTM({d}, 5, batch_first=True)(packed)\n"
         f"padded, lens = pad_packed_sequence(out, batch_first=True)\nresult = [tuple(padded.shape), lens.tolist()]",
         {"require": ["pack_padded_sequence", "enforce_sorted"]}),
        ("Conv3d", f"Return the output shape of `nn.Conv3d(1, 4, 3)` on a volume `torch.zeros({N}, 1, 6, 7, 8)`.",
         sh.format(f"nn.Conv3d(1, 4, 3)(torch.zeros({N}, 1, 6, 7, 8))")),
        ("ConvTranspose2d", f"A transposed convolution upsamples: return the output shape of `nn.ConvTranspose2d({C}, 3, "
                            f"kernel_size=2, stride=2)` on `img`.", sh.format(f"nn.ConvTranspose2d({C}, 3, kernel_size=2, stride=2)(img)")),
        ("MaxPool2d", "Return the output shape of `nn.MaxPool2d(2)` on `img`.", sh.format("nn.MaxPool2d(2)(img)")),
        ("AvgPool2d", "Return the output shape of `nn.AvgPool2d(2)` on `img`.", sh.format("nn.AvgPool2d(2)(img)")),
        ("AdaptiveAvgPool2d", "Return the output shape of `nn.AdaptiveAvgPool2d((1, 1))` on `img` (global average pooling).",
         sh.format("nn.AdaptiveAvgPool2d((1, 1))(img)")),
        ("AdaptiveMaxPool1d", "Return the output shape of `nn.AdaptiveMaxPool1d(1)` on `seq`.", sh.format("nn.AdaptiveMaxPool1d(1)(seq)")),
        ("Flatten", "Return the output shape of `nn.Flatten(start_dim=1)` on `img`.", sh.format("nn.Flatten(start_dim=1)(img)")),
        ("Unflatten", f"Return the shape of `nn.Unflatten(1, ({C}, {H}, {W}))` applied to `img.flatten(1)`.",
         sh.format(f"nn.Unflatten(1, ({C}, {H}, {W}))(img.flatten(1))")),
        ("Upsample", "Return the output shape of `nn.Upsample(scale_factor=2, mode='bilinear')` on `img`.",
         sh.format("nn.Upsample(scale_factor=2, mode='bilinear')(img)"), {"require": ["Upsample", "scale_factor"]}),
        ("PixelShuffle", "`nn.PixelShuffle(2)` trades 4 channels for a 2× larger image. Return its output shape on "
                         f"`torch.zeros({N}, 8, 3, 3)`.", sh.format(f"nn.PixelShuffle(2)(torch.zeros({N}, 8, 3, 3))")),
        ("Embedding", f"Return the output shape of `nn.Embedding(num_embeddings={V}, embedding_dim={d}, padding_idx=0)` on "
                      f"the integer `ids`.", sh.format(f"nn.Embedding(num_embeddings={V}, embedding_dim={d}, padding_idx=0)(ids)"),
         {"require": ["Embedding", "embedding_dim"]}),
        ("EmbeddingBag", f"`nn.EmbeddingBag({V}, {d}, mode='mean')` averages each row's embeddings. Return its output "
                         f"shape on `ids`.", sh.format(f"nn.EmbeddingBag({V}, {d}, mode='mean')(ids)")),
        ("LSTM", f"Return `[out.shape, h_n.shape]` (as tuples) for `out, (h_n, c_n) = nn.LSTM(input_size={d}, hidden_size=5, "
                 f"num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)(x)`.",
         f"out, (h_n, c_n) = nn.LSTM(input_size={d}, hidden_size=5, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)(x)\n"
         f"result = [tuple(out.shape), tuple(h_n.shape)]", {"require": ["LSTM", "bidirectional"]}),
        ("GRU", f"Return the output shape of `nn.GRU({d}, 6, batch_first=True)` on `x` (the first element it returns).",
         f"tuple(nn.GRU({d}, 6, batch_first=True)(x)[0].shape)", {"require": ["GRU", "batch_first"]}),
        ("RNN", f"Return the shape of the final hidden state from `nn.RNN({d}, 6, nonlinearity='tanh', batch_first=True)` on `x`.",
         f"tuple(nn.RNN({d}, 6, nonlinearity='tanh', batch_first=True)(x)[1].shape)"),
        ("Bilinear", f"Return the output shape of `nn.Bilinear({d}, {d}, 4)` applied to `x[:, 0]` and `x[:, 1]`.",
         sh.format(f"nn.Bilinear({d}, {d}, 4)(x[:, 0], x[:, 1])")),
        ("Identity", "`nn.Identity()` is a placeholder layer. Return its output shape on `img`.", sh.format("nn.Identity()(img)")),
        ("LazyLinear", f"`nn.LazyLinear(32)` infers `in_features` on its first call. Call it on `x` and return the shape of "
                       f"its weight.", f"lin = nn.LazyLinear(32)\nlin(x)\nresult = tuple(lin.weight.shape)"),
        ("Dropout2d", "Return the output shape of `nn.Dropout2d(0.1)` on `img` (it zeroes whole channels).", sh.format("nn.Dropout2d(0.1)(img)")),
        ("BatchNorm2d", f"Return the output shape of `nn.BatchNorm2d({C})` on `img`.", sh.format(f"nn.BatchNorm2d({C})(img)")),
        ("BatchNorm1d", f"Return the output shape of `nn.BatchNorm1d({d})` on `torch.zeros(4, {d})` (a batch of 4 feature "
                        f"vectors; in training mode it needs more than one value per channel).",
         sh.format(f"nn.BatchNorm1d({d})(torch.zeros(4, {d}))")),
        ("GroupNorm", "Return the output shape of `nn.GroupNorm(4, 16)` on `torch.zeros(2, 16, 5, 5)`.",
         sh.format("nn.GroupNorm(4, 16)(torch.zeros(2, 16, 5, 5))")),
        ("InstanceNorm2d", f"Return the output shape of `nn.InstanceNorm2d({C})` on `img`.", sh.format(f"nn.InstanceNorm2d({C})(img)")),
        ("RMSNorm", f"Return the output shape of `nn.RMSNorm({d})` on `x`.", sh.format(f"nn.RMSNorm({d})(x)")),
    ]
    return catalog(rng, items, setup, "")


# ================================================================= easy

@template(M, "easy")
def activation_catalog(rng):
    x = fm(rng, 2, 4, -3, 3, 1)
    setup = f"x = {_fmat(x)}"
    acts = [("ReLU", "nn.ReLU()"), ("LeakyReLU", "nn.LeakyReLU(0.01)"), ("PReLU", "nn.PReLU()"), ("ELU", "nn.ELU()"),
            ("SELU", "nn.SELU()"), ("GELU", "nn.GELU(approximate='tanh')"), ("SiLU", "nn.SiLU()"), ("Mish", "nn.Mish()"),
            ("Sigmoid", "nn.Sigmoid()"), ("Tanh", "nn.Tanh()"), ("Softmax", "nn.Softmax(dim=-1)"),
            ("LogSoftmax", "nn.LogSoftmax(dim=-1)"), ("Softplus", "nn.Softplus()"), ("Hardtanh", "nn.Hardtanh()"),
            ("ReLU6", "nn.ReLU6()"), ("Hardswish", "nn.Hardswish()"), ("GLU", "nn.GLU()")]
    items = [(name, f"Apply the activation layer `{code}` to `x`.", f"{code}(x)") for name, code in acts]
    items += [
        ("relu", "Apply the functional `F.relu` to `x`.", "F.relu(x)"),
        ("gelu", "Apply the exact (erf-based) `F.gelu` to `x`.", "F.gelu(x)"),
        ("silu", "Apply `F.silu` (x · sigmoid(x)) to `x`.", "F.silu(x)"),
        ("softmax", "Apply `F.softmax` over the last dimension of `x`.", "F.softmax(x, dim=-1)"),
        ("log_softmax", "Apply `F.log_softmax` over the last dimension of `x`.", "F.log_softmax(x, dim=-1)"),
        ("logsigmoid", "Apply the stable `F.logsigmoid` to `x`.", "F.logsigmoid(x)"),
        ("softplus", "Apply `F.softplus` to `x`.", "F.softplus(x)"),
    ]
    return catalog(rng, items, setup, "`x` is a 2×4 batch of pre-activations.")


@template(M, "easy")
def module_api_catalog(rng):
    d = rng.choice([3, 4, 5])
    setup = (f"class Scaler(nn.Module):\n    def __init__(self, d):\n        super().__init__()\n"
             f"        self.scale = nn.Parameter(torch.ones(d))                 # trainable, in state_dict\n"
             f"        self.register_buffer('running_mean', torch.zeros(d))     # saved, NOT trainable\n"
             f"        self.layers = nn.ModuleList([nn.Linear(d, d) for _ in range(2)])\n"
             f"        self.heads = nn.ModuleDict({{'a': nn.Linear(d, 1), 'b': nn.Linear(d, 2)}})\n"
             f"    def forward(self, x):\n        for layer in self.layers:\n            x = layer(x)\n"
             f"        return self.heads['a'](x * self.scale)\n\ntorch.manual_seed(0)\nnet = Scaler({d})")
    items = [
        ("buffers", "Return the names of `net`'s buffers (`named_buffers()`; `register_buffer` made one).",
         "[n for n, _ in net.named_buffers()]", {"require": ["named_buffers|buffers"]}),
        ("children", "How many direct children does `net` have? (`net.children()`)", "len(list(net.children()))"),
        ("buffers", "Return the shapes of the tensors in `net.buffers()` (as tuples).", "[tuple(b.shape) for b in net.buffers()]"),
        ("float32", "Return the dtype of `net.scale` after `net.to(torch.float32)`.", "net.to(torch.float32)\nresult = net.scale.dtype"),
        ("reshape", f"Run `net` on a batch shaped `(2, 3, {d})` reshaped to `(6, {d})` with `.reshape(-1, {d})`, and return the "
                    f"output shape.", f"result = tuple(net(torch.ones(2, 3, {d}).reshape(-1, {d})).shape)"),
        ("modules", "Return the class names of every module in `net`, itself included (`net.modules()`).",
         "[type(m).__name__ for m in net.modules()]"),
        ("named_modules", "Return the names of every submodule (`named_modules()`; the root's name is '').",
         "[n for n, _ in net.named_modules()]"),
        ("get_submodule", "Return the weight shape of the module at `'layers.1'` (`net.get_submodule`).",
         "tuple(net.get_submodule('layers.1').weight.shape)"),
        ("get_parameter", "Return the shape of the parameter `'heads.b.bias'` (`net.get_parameter`).",
         "tuple(net.get_parameter('heads.b.bias').shape)"),
        ("apply", "`model.apply(fn)` runs `fn` on every submodule. Write `init_weights(m)` that zeroes every `nn.Linear`'s "
                  "weight and bias with `nn.init.zeros_`, apply it, and return `net.heads['b'].weight`.",
         "def init_weights(m):\n    if isinstance(m, nn.Linear):\n        nn.init.zeros_(m.weight)\n        nn.init.zeros_(m.bias)\n"
         "net.apply(init_weights)\nresult = net.heads['b'].weight.detach()", {"require": ["apply", "zeros_"]}),
        ("train", "Return `[net.training]` after `net.eval()`, then `net.training` after `net.train()` (dropout and "
                  "batch-norm switch on this flag).", "net.eval()\na = net.training\nnet.train()\nresult = [a, net.training]",
         {"require": ["train", "eval"]}),
        ("requires_grad_", "Freeze everything with `net.requires_grad_(False)` and return the number of trainable parameters.",
         "net.requires_grad_(False)\nresult = sum(p.numel() for p in net.parameters() if p.requires_grad)"),
        ("to", "Convert `net` to float64 with `net.to(torch.float64)` and return the dtype of `net.scale`.",
         "net.to(torch.float64)\nresult = net.scale.dtype"),
        ("half", "Return the dtype of `net.scale` after `net.half()`.", "net.half()\nresult = net.scale.dtype"),
        ("float", "Call `net.half()` and then `net.float()`, and return the dtype of `net.scale`.", "net.half()\nnet.float()\nresult = net.scale.dtype"),
        ("cpu", "Return whether `net.cpu()` returns the same module object (moving modules is in place).", "net.cpu() is net"),
        ("ModuleList", "Return `len(net.layers)`: an `nn.ModuleList` registers its layers (a plain list would not).", "len(net.layers)",
         {"require": ["layers"]}),
        ("ModuleDict", "Return the sorted keys of the `nn.ModuleDict` `net.heads`.", "sorted(net.heads.keys())", {"require": ["heads"]}),
        ("state_dict", "Return the sorted keys of `net.state_dict()` (buffers are included).", "sorted(net.state_dict())"),
        ("parameters", "Return the total number of trainable parameters (`sum(p.numel() for p in net.parameters() if "
                       "p.requires_grad)`).", "sum(p.numel() for p in net.parameters() if p.requires_grad)"),
        ("load_state_dict", "Load `{'scale': torch.full((d,), 2.0)}` with `strict=False` and return the sorted missing keys "
                            "from the result.", f"res = net.load_state_dict({{'scale': torch.full(({d},), 2.0)}}, strict=False)\n"
                                                f"result = sorted(res.missing_keys)", {"require": ["load_state_dict", "strict"]}),
        ("zero_grad", "Backpropagate `net(torch.ones(1, d)).sum()`, then call `net.zero_grad(set_to_none=True)` and return "
                      "`net.scale.grad is None`.", f"net(torch.ones(1, {d})).sum().backward()\nnet.zero_grad(set_to_none=True)\n"
                                                    f"result = net.scale.grad is None", {"require": ["zero_grad", "set_to_none"]}),
    ]
    return catalog(rng, items, setup, "")


@template(M, "easy")
def init_catalog(rng):
    r, c = rng.randint(2, 4), rng.randint(4, 6)
    setup = f"torch.manual_seed({seeded(rng)})\nw = torch.empty({r}, {c})"
    items = [
        ("zeros_", "Fill `w` with zeros in place (`nn.init.zeros_`) and return it.", "nn.init.zeros_(w)"),
        ("ones_", "Fill `w` with ones (`nn.init.ones_`) and return it.", "nn.init.ones_(w)"),
        ("constant_", "Fill `w` with 0.5 (`nn.init.constant_`) and return it.", "nn.init.constant_(w, 0.5)"),
        ("eye_", "Make `w` an identity-like matrix (`nn.init.eye_`) and return it.", "nn.init.eye_(w)"),
        ("uniform_", "After the seed, fill `w` from U(-0.1, 0.1) (`nn.init.uniform_`).", "nn.init.uniform_(w, -0.1, 0.1)"),
        ("normal_", "After the seed, fill `w` from N(0, 0.02²) (`nn.init.normal_(w, mean=0.0, std=0.02)`).",
         "nn.init.normal_(w, mean=0.0, std=0.02)"),
        ("trunc_normal_", "After the seed, fill `w` from a truncated normal with std 0.02 (`nn.init.trunc_normal_`).",
         "nn.init.trunc_normal_(w, std=0.02)"),
        ("xavier_uniform_", "After the seed, apply Glorot/Xavier uniform init (`nn.init.xavier_uniform_`).", "nn.init.xavier_uniform_(w)"),
        ("xavier_normal_", "After the seed, apply Xavier normal init (`nn.init.xavier_normal_`).", "nn.init.xavier_normal_(w)"),
        ("kaiming_uniform_", "After the seed, apply He/Kaiming uniform init (`nn.init.kaiming_uniform_`).", "nn.init.kaiming_uniform_(w)"),
        ("kaiming_normal_", "After the seed, apply `nn.init.kaiming_normal_(w, mode='fan_in', nonlinearity='relu')`.",
         "nn.init.kaiming_normal_(w, mode='fan_in', nonlinearity='relu')", {"require": ["kaiming_normal_", "nonlinearity"]}),
        ("orthogonal_", "After the seed, make the rows of `w` orthonormal (`nn.init.orthogonal_`) and return whether "
                        f"`w @ w.T` is close to `torch.eye({r})`.", f"nn.init.orthogonal_(w)\nresult = torch.allclose(w @ w.T, torch.eye({r}), atol=1e-5)"),
    ]
    return catalog(rng, items, setup, f"`w = torch.empty({r}, {c})` is an uninitialized weight.")


# ================================================================= medium

@template(M, "medium")
def functional_catalog(rng):
    t, u = fm(rng, 2, 4), fm(rng, 2, 4)
    img = [[[[float((i * 3 + j) % 7) for j in range(4)] for i in range(4)]]]
    W = fm(rng, 3, 4)
    setup = f"t = {_fmat(t)}\nu = {_fmat(u)}\nimg = torch.tensor({lit(img)})   # (1, 1, 4, 4)\nW = {_fmat(W)}"
    items = [
        ("one_hot", "Return `F.one_hot(torch.tensor([0, 2]), num_classes=3)`.", "F.one_hot(torch.tensor([0, 2]), num_classes=3)",
         {"require": ["one_hot", "num_classes"]}),
        ("normalize", "Scale each row of `t` to unit L2 norm with `F.normalize(t, p=2, dim=-1)`.", "F.normalize(t, p=2, dim=-1)"),
        ("cosine_similarity", "Return the cosine similarity of matching rows of `t` and `u` (`F.cosine_similarity(..., dim=-1)`).",
         "F.cosine_similarity(t, u, dim=-1)"),
        ("pad", "Pad the last dim of `t` with 1 zero on the left and 2 on the right (`F.pad(t, (1, 2))`).", "F.pad(t, (1, 2))"),
        ("pad", "Pad `img` by 1 on left/right and 2 on top/bottom with the value -1 (`F.pad(img, (1, 1, 2, 2), mode='constant', "
                "value=-1)`): pairs start from the LAST dim.", "F.pad(img, (1, 1, 2, 2), mode='constant', value=-1)"),
        ("interpolate", "Resize `img` to 6×6 with `F.interpolate(img, size=(6, 6), mode='bilinear', align_corners=False)`.",
         "F.interpolate(img, size=(6, 6), mode='bilinear', align_corners=False)", {"require": ["interpolate", "align_corners"]}),
        ("dropout", "Return `F.dropout(t, p=0.5, training=False)`: in eval mode dropout is the identity.",
         "F.dropout(t, p=0.5, training=False)", {"require": ["dropout", "training"]}),
        ("linear", "Return `F.linear(t, W)`: `t @ W.T` with the (out, in) weight layout.", "F.linear(t, W)"),
        ("embedding", "Look up rows 1 and 2 of the embedding table `W` with `F.embedding(torch.tensor([1, 2]), W)`.",
         "F.embedding(torch.tensor([1, 2]), W)"),
        ("layer_norm", "Normalize each row of `t` with `F.layer_norm(t, (4,))`.", "F.layer_norm(t, (4,))"),
        ("unfold", "Extract all 3×3 patches of `img` (im2col) with `F.unfold(img, kernel_size=3)` and return its shape.",
         "tuple(F.unfold(img, kernel_size=3).shape)"),
        ("gumbel_softmax", "`F.gumbel_softmax(t, tau=1.0, hard=True)` draws a one-hot sample per row. Return its row sums.",
         "F.gumbel_softmax(t, tau=1.0, hard=True).sum(-1)", {"require": ["gumbel_softmax", "hard"]}),
        ("cdist", "Return the pairwise Euclidean distances between the rows of `t` and of `u` (`torch.cdist`).", "torch.cdist(t, u)"),
    ]
    return catalog(rng, items, setup, "")


@template(M, "medium")
def losses_catalog(rng):
    N, C = rng.randint(3, 4), 3
    logits, other = fm(rng, N, C), fm(rng, N, C)
    target = [rng.randrange(C) for _ in range(N)]
    a, b, y = fv(rng, N), fv(rng, N), [float(rng.choice([1, -1])) for _ in range(N)]
    setup = (f"logits = {_fmat(logits)}\ntarget = {tt(target)}\nother = {_fmat(other)}\na = {ftt(a)}\nb = {ftt(b)}\n"
             f"y = {ftt(y)}   # +1 / -1 labels")
    w3 = [round(rng.uniform(0.5, 2), 1) for _ in range(C)]
    items = [
        ("CrossEntropyLoss", "Return `nn.CrossEntropyLoss()(logits, target)` (raw logits, int64 class ids).", "nn.CrossEntropyLoss()(logits, target)"),
        ("CrossEntropyLoss", f"Ignore the last example (set its target to -100) and smooth labels by 0.1, with class weights "
                             f"{w3}: `nn.CrossEntropyLoss(weight=..., ignore_index=-100, label_smoothing=0.1)`.",
         f"t2 = target.clone()\nt2[-1] = -100\nresult = nn.CrossEntropyLoss(weight=torch.tensor({w3}), ignore_index=-100, "
         f"label_smoothing=0.1)(logits, t2)", {"require": ["CrossEntropyLoss", "label_smoothing"]}),
        ("cross_entropy", "Soft-label targets: return `F.cross_entropy(logits, F.softmax(other, -1))` (probabilities instead "
                          "of class ids).", "F.cross_entropy(logits, F.softmax(other, -1))"),
        ("cross_entropy", "Sequence models put the class dim second: return `F.cross_entropy(logits.T[None], target[None])` "
                          "for inputs shaped (N=1, C, T) and targets (1, T).", "F.cross_entropy(logits.T[None], target[None])"),
        ("NLLLoss", "CrossEntropy = LogSoftmax + NLLLoss: return `nn.NLLLoss()(F.log_softmax(logits, -1), target)`.",
         "nn.NLLLoss()(F.log_softmax(logits, -1), target)"),
        ("BCEWithLogitsLoss", "Binary loss on raw scores `a` with targets `(y > 0).float()`, up-weighting positives 3× "
                              "(`nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]))`).",
         "nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]))(a, (y > 0).float())", {"require": ["BCEWithLogitsLoss", "pos_weight"]}),
        ("BCELoss", "`nn.BCELoss` expects probabilities: return it for `torch.sigmoid(a)` and targets `(y > 0).float()`.",
         "nn.BCELoss()(torch.sigmoid(a), (y > 0).float())"),
        ("MSELoss", "Return `nn.MSELoss()(a, b)`.", "nn.MSELoss()(a, b)"),
        ("L1Loss", "Return `nn.L1Loss()(a, b)`.", "nn.L1Loss()(a, b)"),
        ("SmoothL1Loss", "Return `nn.SmoothL1Loss(beta=1.0)(a, b)`.", "nn.SmoothL1Loss(beta=1.0)(a, b)"),
        ("HuberLoss", "Return `nn.HuberLoss(delta=0.5)(a, b)`.", "nn.HuberLoss(delta=0.5)(a, b)", {"require": ["HuberLoss", "delta"]}),
        ("KLDivLoss", "Return `nn.KLDivLoss(reduction='batchmean')(F.log_softmax(logits, -1), F.softmax(other, -1))`: input as "
                      "log-probs, target as probs.", "nn.KLDivLoss(reduction='batchmean')(F.log_softmax(logits, -1), F.softmax(other, -1))"),
        ("CosineEmbeddingLoss", "Return `nn.CosineEmbeddingLoss()(logits, other, y)`: pull pairs with y = 1 together, push "
                                "y = -1 apart.", "nn.CosineEmbeddingLoss()(logits, other, y)"),
        ("MarginRankingLoss", "Return `nn.MarginRankingLoss(margin=0.5)(a, b, y)`: `a` should beat `b` by 0.5 when y = 1.",
         "nn.MarginRankingLoss(margin=0.5)(a, b, y)", {"require": ["MarginRankingLoss", "margin"]}),
        ("TripletMarginLoss", "Return `nn.TripletMarginLoss(margin=1.0)(logits, other, -other)` (anchor, positive, negative).",
         "nn.TripletMarginLoss(margin=1.0)(logits, other, -other)"),
        ("CTCLoss", "Speech recognition: `lp = F.log_softmax(torch.cat([logits, other]), -1)[:, None]` holds log-probs for "
                    "T = 2N frames of one utterance over 3 symbols (0 = blank). Return `nn.CTCLoss()(lp, torch.tensor([[1, 2]]), "
                    "torch.tensor([lp.shape[0]]), torch.tensor([2]))`.",
         "lp = F.log_softmax(torch.cat([logits, other]), -1)[:, None]\n"
         "result = nn.CTCLoss()(lp, torch.tensor([[1, 2]]), torch.tensor([lp.shape[0]]), torch.tensor([2]))"),
        ("GaussianNLLLoss", "Return `nn.GaussianNLLLoss()(a, b, torch.ones_like(a) * 0.5)` (prediction, target, variance).",
         "nn.GaussianNLLLoss()(a, b, torch.ones_like(a) * 0.5)"),
        ("PoissonNLLLoss", "Return `nn.PoissonNLLLoss()(a, b.abs())` (log-rate prediction, count target).", "nn.PoissonNLLLoss()(a, b.abs())"),
        ("mse_loss", "Return the shape of `F.mse_loss(a, b, reduction='none')` as a tuple (no averaging).",
         "tuple(F.mse_loss(a, b, reduction='none').shape)", {"require": ["mse_loss", "reduction"]}),
    ]
    return catalog(rng, items, setup, "")


@template(M, "medium")
def optimizer_catalog(rng):
    w0, tgt = fv(rng, 3), fv(rng, 3)
    k = rng.randint(2, 4)
    setup = (f"w = nn.Parameter({ftt(w0)})\ntarget = {ftt(tgt)}\ndef loss_fn():\n    return ((w - target) ** 2).sum() + 0.1 * (w ** 4).sum()")
    loop = "for _ in range({k}):\n    opt.zero_grad()\n    loss_fn().backward()\n    opt.step()\nresult = w.detach()"
    opts = [("SGD", "torch.optim.SGD([w], lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4)", "nesterov"),
            ("Adam", "torch.optim.Adam([w], lr=0.1, betas=(0.9, 0.999), eps=1e-8)", "betas"),
            ("AdamW", "torch.optim.AdamW([w], lr=0.1, weight_decay=0.01)", "weight_decay"),
            ("RMSprop", "torch.optim.RMSprop([w], lr=0.01)", ""), ("Adagrad", "torch.optim.Adagrad([w], lr=0.1)", ""),
            ("NAdam", "torch.optim.NAdam([w], lr=0.1)", ""), ("RAdam", "torch.optim.RAdam([w], lr=0.1)", ""),
            ("Adamax", "torch.optim.Adamax([w], lr=0.1)", "")]
    items = [(name, f"Create `opt = {code}` and take {k} steps (zero_grad, `loss_fn().backward()`, step). Return `w.detach()`.",
              f"opt = {code}\n" + loop.format(k=k), {"require": [name] + ([kw] if kw else [])}) for name, code, kw in opts]
    items += [
        ("LBFGS", "L-BFGS needs a **closure** that recomputes the loss: `opt = torch.optim.LBFGS([w], lr=1, max_iter=5)`; "
                  "`def closure(): opt.zero_grad(); l = loss_fn(); l.backward(); return l`; call `opt.step(closure)` twice and "
                  "return `w.detach()`.",
         "opt = torch.optim.LBFGS([w], lr=1, max_iter=5)\ndef closure():\n    opt.zero_grad()\n    l = loss_fn()\n    l.backward()\n"
         "    return l\nfor _ in range(2):\n    opt.step(closure)\nresult = w.detach()", {"require": ["LBFGS", "closure"]}),
        ("param_groups", "Build `opt = torch.optim.SGD([w], lr=0.1)`, then change the learning rate by hand to 0.01 through "
                         "`opt.param_groups` and return `[g['lr'] for g in opt.param_groups]`.",
         "opt = torch.optim.SGD([w], lr=0.1)\nfor g in opt.param_groups:\n    g['lr'] = 0.01\nresult = [g['lr'] for g in opt.param_groups]"),
        ("state_dict", f"After {k} Adam steps (lr=0.1), return the sorted keys of the per-parameter state in "
                       f"`opt.state_dict()['state'][0]` and the stored step count.",
         f"opt = torch.optim.Adam([w], lr=0.1)\nfor _ in range({k}):\n    opt.zero_grad()\n    loss_fn().backward()\n    opt.step()\n"
         f"st = opt.state_dict()['state'][0]\nresult = [sorted(st), float(st['step'])]"),
        ("clip_grad_norm_", "Backpropagate `loss_fn()`, then clip the gradient to norm 1 with `nn.utils.clip_grad_norm_([w], "
                            "max_norm=1.0)`. Return `[total_norm_before_clipping, w.grad.norm()]`.",
         "loss_fn().backward()\ntotal = nn.utils.clip_grad_norm_([w], max_norm=1.0)\nresult = [total, w.grad.norm()]",
         {"require": ["clip_grad_norm_", "max_norm"]}),
    ]
    return catalog(rng, items, setup, "", tol=1e-4)


@template(M, "medium")
def scheduler_catalog(rng):
    n = rng.randint(6, 9)
    base = rng.choice([0.1, 1.0, 0.5])
    setup = f"S = torch.optim.lr_scheduler\nopt = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr={base})"
    rec = (f"lrs = []\nfor _ in range({n}):\n    lrs.append(opt.param_groups[0]['lr'])\n    opt.step()\n    sched.step()\n"
           f"result = lrs")
    scheds = [("StepLR", "S.StepLR(opt, step_size=2, gamma=0.5)"),
              ("MultiStepLR", "S.MultiStepLR(opt, milestones=[2, 5], gamma=0.1)"),
              ("ExponentialLR", "S.ExponentialLR(opt, gamma=0.8)"),
              ("CosineAnnealingLR", f"S.CosineAnnealingLR(opt, T_max={n - 1}, eta_min=0.01)"),
              ("CosineAnnealingWarmRestarts", "S.CosineAnnealingWarmRestarts(opt, T_0=2, T_mult=2)"),
              ("OneCycleLR", f"S.OneCycleLR(opt, max_lr={base * 10}, total_steps={n})"),
              ("LinearLR", "S.LinearLR(opt, start_factor=0.1, total_iters=4)"),
              ("ConstantLR", "S.ConstantLR(opt, factor=0.5, total_iters=3)"),
              ("PolynomialLR", "S.PolynomialLR(opt, total_iters=4, power=2.0)"),
              ("SequentialLR", "S.SequentialLR(opt, schedulers=[S.LinearLR(opt, start_factor=0.25, total_iters=2), "
                               f"S.CosineAnnealingLR(opt, T_max={n})], milestones=[2])")]
    items = [(name, f"Attach `sched = {code}` and record `opt.param_groups[0]['lr']` before each of {n} "
                    f"`opt.step(); sched.step()` rounds. Return the list.", f"sched = {code}\n{rec}") for name, code in scheds]
    metric = [round(1.0 - 0.1 * i if i < 3 else 0.7, 2) for i in range(n)]
    items += [
        ("ReduceLROnPlateau", f"`ReduceLROnPlateau` watches a metric: build `S.ReduceLROnPlateau(opt, mode='min', factor=0.5, "
                              f"patience=1)`, call `sched.step(m)` for each validation loss in `{metric}`, and record the lr "
                              f"after each call.", f"sched = S.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=1)\nlrs = []\n"
                                                   f"for m in {metric}:\n    sched.step(m)\n    lrs.append(opt.param_groups[0]['lr'])\nresult = lrs",
         {"require": ["ReduceLROnPlateau", "patience"]}),
        ("get_last_lr", f"After 3 rounds of `opt.step(); sched.step()` with `sched = S.StepLR(opt, step_size=1, gamma=0.5)`, "
                        f"return `sched.get_last_lr()` and `sched.state_dict()['last_epoch']`.",
         "sched = S.StepLR(opt, step_size=1, gamma=0.5)\nfor _ in range(3):\n    opt.step()\n    sched.step()\n"
         "result = [sched.get_last_lr(), sched.state_dict()['last_epoch']]"),
    ]
    return catalog(rng, items, setup, "", tol=1e-7)


@template(M, "medium")
def persistence_catalog(rng):
    d, o = rng.randint(3, 6), rng.randint(1, 3)
    setup = f"torch.manual_seed({seeded(rng)})\nmodel = nn.Sequential(nn.Linear({d}, 8), nn.ReLU(), nn.Linear(8, {o}))\nx = torch.ones(2, {d})"
    items = [
        ("load", f"Save `model.state_dict()` to `model.pt`, then load it back onto the CPU with `torch.load('model.pt', "
                 f"map_location='cpu', weights_only=True)` into a fresh `m2` of the same architecture, call `m2.eval()`, and "
                 f"return `m2(x)`.", f"torch.save(model.state_dict(), 'model.pt')\nm2 = nn.Sequential(nn.Linear({d}, 8), nn.ReLU(), nn.Linear(8, {o}))\n"
                                     f"m2.load_state_dict(torch.load('model.pt', map_location='cpu', weights_only=True))\nm2.eval()\nresult = m2(x)",
         {"require": ["load", "map_location"]}),
        ("save", "Save a resume checkpoint `{'epoch': 10, 'model': model.state_dict()}` to `ckpt.pt`, load it back, and "
                 "return the epoch to resume from (`ck['epoch'] + 1`).",
         "torch.save({'epoch': 10, 'model': model.state_dict()}, 'ckpt.pt')\nck = torch.load('ckpt.pt', weights_only=True)\nresult = ck['epoch'] + 1"),
        ("export", f"Capture `model` for deployment with `torch.export.export(model, (x,))` and return whether "
                   f"`ep.module()(x)` matches `model(x)`.", "ep = torch.export.export(model, (x,))\nresult = torch.allclose(ep.module()(x), model(x))"),
        ("DataLoader", f"Training reshuffles every epoch: return the number of batches of `DataLoader(TensorDataset(torch.ones("
                       f"10, {d})), batch_size=4, shuffle=True)`.",
         f"from torch.utils.data import DataLoader, TensorDataset\nresult = len(DataLoader(TensorDataset(torch.ones(10, {d})), batch_size=4, shuffle=True))",
         {"require": ["DataLoader", "shuffle"]}),
        ("state_dict", "Return the shapes of `model.state_dict()`'s tensors as `{name: tuple(shape)}`.",
         "{k: tuple(v.shape) for k, v in model.state_dict().items()}"),
    ]
    return catalog(rng, items, setup, "", tol=1e-6)
