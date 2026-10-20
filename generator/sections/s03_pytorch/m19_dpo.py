"""3.19 DPO loss."""

from ._common import P, challenge, floats, ftt, lit, seed_line, seeded, template, tt

M = "3.19"

_SEQLOGP = ("def sequence_logprobs(logits, labels, mask):\n"
            "    logits, labels, mask = logits[:, :-1], labels[:, 1:], mask[:, 1:]   # position t predicts token t+1\n"
            "    logp = torch.log_softmax(logits.float(), dim=-1)\n"
            "    return (logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1) * mask).sum(-1)\n")


def _logps(rng, B=None):
    B = B or rng.randint(3, 5)
    vals = {k: floats(rng, B, -30, -5, 1) for k in ("pi_chosen", "pi_rejected", "ref_chosen", "ref_rejected")}
    setup = "\n".join((f"{k} = torch.tensor({lit(v)}, requires_grad=True)" if k.startswith("pi") else f"{k} = {ftt(v)}")
                      for k, v in vals.items())
    return B, setup


DESC = ("`pi_chosen`/`pi_rejected` are the policy's sequence log-probs of the chosen and rejected responses, and "
        "`ref_chosen`/`ref_rejected` the frozen reference model's")


# ================================================================= lite

@template(M, "lite")
def logratio_basic(rng):
    B, setup = _logps(rng)
    if rng.random() < 0.5:
        return P(f"{DESC}. Return the DPO log-ratio difference `(pi_chosen − pi_rejected) − (ref_chosen − ref_rejected)`: "
                 f"how much more the policy prefers the chosen response than the reference did.", setup,
                 "(pi_chosen - pi_rejected) - (ref_chosen - ref_rejected)", tol=1e-5)
    beta = rng.choice([0.1, 0.5, 0.05])
    return P(f"{DESC}. Return the DPO logits `beta · ((pi_chosen − pi_rejected) − (ref_chosen − ref_rejected))` with "
             f"beta = {beta}.", setup, f"{beta} * ((pi_chosen - pi_rejected) - (ref_chosen - ref_rejected))", tol=1e-5)


@template(M, "lite")
def logsigmoid_loss(rng):
    B = rng.randint(3, 6)
    if rng.random() < 0.3:
        return P(f"At the start of DPO training the policy equals the reference, so all DPO logits are 0. Return "
                 f"`-F.logsigmoid(logits).mean()` for `logits = torch.zeros({B})` (it equals log 2 ≈ 0.693).", "",
                 f"-F.logsigmoid(torch.zeros({B})).mean()", require=["logsigmoid"], tol=1e-6)
    logits = floats(rng, B, -3, 3, 2)
    return P("Return the DPO loss for these logits: `-F.logsigmoid(logits).mean()` (a numerically stable "
             "`-log σ(logits)`).", f"logits = {ftt(logits)}", "-F.logsigmoid(logits).mean()", require=["logsigmoid"],
             tol=1e-6)


# ================================================================= easy

@template(M, "easy")
def dpo_loss_fn(rng):
    B, setup = _logps(rng)
    beta = rng.choice([0.1, 0.2, 0.5])
    return P(f"{DESC}. Write `dpo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta)` returning `(loss, "
             f"chosen_rewards, rejected_rewards)`: `logits = beta · (policy log-ratio − reference log-ratio)`, `loss = "
             f"-logsigmoid(logits).mean()`, and the implicit rewards `beta · (pi − ref).detach()` for each side. Return "
             f"`dpo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected, {beta})`.", setup,
             "def dpo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta):\n"
             "    logits = beta * ((pi_chosen - pi_rejected) - (ref_chosen - ref_rejected))\n"
             "    loss = -F.logsigmoid(logits).mean()\n"
             "    return loss, beta * (pi_chosen - ref_chosen).detach(), beta * (pi_rejected - ref_rejected).detach()\n"
             f"result = dpo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected, {beta})", require=["logsigmoid"],
             tol=1e-5)


@template(M, "easy")
def reward_metrics(rng):
    B = rng.randint(4, 8)
    cr, rr = floats(rng, B, -2, 2, 2), floats(rng, B, -2, 2, 2)
    setup = f"chosen_rewards = {ftt(cr)}\nrejected_rewards = {ftt(rr)}"
    return P("The two curves to watch while training DPO: **reward accuracy** (the fraction of pairs where the chosen "
             "reward beats the rejected one) and **reward margin** (the mean difference). Return both as floats.", setup,
             "result = [(chosen_rewards > rejected_rewards).float().mean().item(), (chosen_rewards - rejected_rewards).mean().item()]",
             tol=1e-6)


@template(M, "easy")
def implicit_rewards(rng):
    B, setup = _logps(rng)
    beta = rng.choice([0.1, 0.5])
    return P(f"{DESC}; `pi_chosen` requires grad. DPO's implicit reward of a response is `beta · (pi − ref)`, logged "
             f"**without** gradients. Return the chosen rewards with beta = {beta} as a tensor detached from the graph.",
             setup, f"result = {beta} * (pi_chosen - ref_chosen).detach()", post="(_, _.requires_grad)",
             require=["detach|no_grad"], tol=1e-5)


# ================================================================= medium

@template(M, "medium")
def sequence_logprobs_fn(rng):
    B, Tn, V = rng.randint(2, 3), rng.randint(4, 6), rng.randint(5, 9)
    prompt = rng.randint(1, Tn - 2)
    setup = (f"{seed_line(rng)}\nlogits = torch.randn({B}, {Tn}, {V})   # model(input_ids)\n"
             f"labels = torch.randint(0, {V}, ({B}, {Tn}))   # = input_ids\nmask = torch.zeros({B}, {Tn})\n"
             f"mask[:, {prompt}:] = 1   # 1 on response tokens, 0 on the prompt")
    return P(f"Write `sequence_logprobs(logits, labels, mask)`: the sum of `log p(token_t | tokens_<t)` over response "
             f"tokens. Position t predicts token t+1, so shift: `logits[:, :-1]`, `labels[:, 1:]`, `mask[:, 1:]`; then "
             f"log-softmax (in fp32), `gather` the label log-probs, multiply by the mask and sum over time. Return it "
             f"for the given tensors (shape ({B},)).", setup,
             _SEQLOGP + "result = sequence_logprobs(logits, labels, mask)", require=["gather"], tol=1e-5)


@template(M, "medium")
def label_smoothing_dpo(rng):
    B, setup = _logps(rng)
    beta, eps = rng.choice([0.1, 0.5]), rng.choice([0.1, 0.2, 0.3])
    return P(f"{DESC}. Conservative DPO assumes a fraction ε = {eps} of the preference labels are flipped: with `logits = "
             f"beta · (policy log-ratio − reference log-ratio)` (beta = {beta}), the per-pair loss is "
             f"`-logsigmoid(logits) · (1 − ε) − logsigmoid(−logits) · ε`. Return its mean.", setup,
             f"logits = {beta} * ((pi_chosen - pi_rejected) - (ref_chosen - ref_rejected))\n"
             f"result = (-F.logsigmoid(logits) * (1 - {eps}) - F.logsigmoid(-logits) * {eps}).mean()",
             require=["logsigmoid"], tol=1e-5)


@template(M, "medium")
def ipo_hinge_variants(rng):
    B, setup = _logps(rng)
    beta = rng.choice([0.1, 0.5, 1.0])
    return P(f"{DESC}. Variants share the log-ratio `h = (pi_chosen − pi_rejected) − (ref_chosen − ref_rejected)`. "
             f"With beta = {beta}, return `[ipo, hinge]` where IPO is `mean((h − 1/(2·beta))²)` and the hinge (SLiC-style) "
             f"loss is `mean(relu(1 − beta · h))`.", setup,
             f"h = (pi_chosen - pi_rejected) - (ref_chosen - ref_rejected)\n"
             f"result = [((h - 1 / (2 * {beta})) ** 2).mean(), torch.relu(1 - {beta} * h).mean()]", tol=1e-4)


# ================================================================= hard

@template(M, "hard")
def simpo_loss(rng):
    B, setup = _logps(rng)
    lc = [rng.randint(5, 40) for _ in range(B)]
    lr_ = [rng.randint(5, 40) for _ in range(B)]
    beta, gamma = rng.choice([2.0, 2.5]), rng.choice([0.5, 1.0, 1.4])
    setup += f"\nlen_chosen = {tt(lc)}\nlen_rejected = {tt(lr_)}"
    return P(f"SimPO is reference-free and **length-normalized**: with beta = {beta} and target margin gamma = {gamma}, "
             f"the loss is `mean(-logsigmoid(beta · (pi_chosen / len_chosen − pi_rejected / len_rejected) − gamma))`. "
             f"Return it (ignore the reference log-probs).", setup,
             f"result = -F.logsigmoid({beta} * (pi_chosen / len_chosen - pi_rejected / len_rejected) - {gamma}).mean()",
             require=["logsigmoid"], tol=1e-5)


@template(M, "hard")
def dpo_step_policy(rng):
    V, d, Tn, B = rng.randint(8, 16), rng.choice([8, 16]), rng.randint(5, 8), rng.randint(2, 3)
    prompt = rng.randint(2, Tn - 2)
    beta = rng.choice([0.1, 0.2])
    setup = (_SEQLOGP + f"\nclass TinyLM(nn.Module):\n    def __init__(self):\n        super().__init__()\n"
             f"        self.emb, self.head = nn.Embedding({V}, {d}), nn.Linear({d}, {V})\n"
             f"    def forward(self, ids):\n        return self.head(self.emb(ids).tanh())\n\n{seed_line(rng)}\npolicy, ref = TinyLM(), TinyLM()\n"
             f"chosen_ids, rejected_ids = torch.randint(0, {V}, ({B}, {Tn})), torch.randint(0, {V}, ({B}, {Tn}))\n"
             f"resp_mask = torch.zeros({B}, {Tn})\nresp_mask[:, {prompt}:] = 1")
    return P(f"One DPO step with a real policy: make `ref` a **frozen copy** of the starting policy "
             f"(`ref.load_state_dict(policy.state_dict())`, then `ref.eval().requires_grad_(False)`). Run chosen and "
             f"rejected in **one** forward pass: `ids = torch.cat([chosen_ids, rejected_ids])` (same for the mask), "
             f"`pi_c, pi_r = sequence_logprobs(policy(ids), ids, mask).chunk(2)`, and the same for `ref` under "
             f"`torch.no_grad()`. Compute the DPO loss with beta = {beta} and call `backward()`. Return "
             f"`[loss.item(), all(p.grad is None for p in ref.parameters())]`; the loss starts at log 2, since policy "
             f"and reference are equal.", setup,
             f"ref.load_state_dict(policy.state_dict())\nref.eval().requires_grad_(False)\n"
             f"ids, mask = torch.cat([chosen_ids, rejected_ids]), torch.cat([resp_mask, resp_mask])\n"
             f"pi_c, pi_r = sequence_logprobs(policy(ids), ids, mask).chunk(2)\nwith torch.no_grad():\n"
             f"    ref_c, ref_r = sequence_logprobs(ref(ids), ids, mask).chunk(2)\n"
             f"loss = -F.logsigmoid({beta} * ((pi_c - pi_r) - (ref_c - ref_r))).mean()\nloss.backward()\n"
             f"result = [loss.item(), all(p.grad is None for p in ref.parameters())]",
             require=["load_state_dict", "chunk"], tol=1e-5)


# ================================================================= challenge

@template(M, "challenge")
def discordant_pairs(rng):
    n = rng.randint(6, 9)
    p = [rng.randint(0, 9) for _ in range(n)]
    r = [rng.randint(0, 9) for _ in range(n)]
    sol = ("ps, rs = policy.tolist(), ref.tolist()\nranks = {v: i + 1 for i, v in enumerate(sorted(set(rs)))}\nm = len(ranks)\n"
           "tree = [0] * (m + 1)\norder = sorted(range(len(ps)), key=lambda i: ps[i])\ntotal = seen = i = 0\n"
           "while i < len(order):\n    j = i\n    while j < len(order) and ps[order[j]] == ps[order[i]]:\n        j += 1\n"
           "    for t in order[i:j]:   # earlier policy groups with a larger reference score\n        k, le = ranks[rs[t]], 0\n"
           "        while k:\n            le += tree[k]\n            k -= k & -k\n        total += seen - le\n"
           "    for t in order[i:j]:\n        k = ranks[rs[t]]\n        while k <= m:\n            tree[k] += 1\n"
           "            k += k & -k\n        seen += 1\n    i = j\nresult = total")
    big = ("g = np.random.default_rng({s})\npolicy = torch.from_numpy(g.integers(0, {hi}, size=20000))\n"
           "ref = torch.from_numpy(g.integers(0, {hi}, size=20000))")
    tests = [big.format(s=seeded(rng), hi=10**9), big.format(s=seeded(rng), hi=5),
             "policy = torch.arange(20000)\nref = torch.arange(20000, 0, -1)", "policy = torch.tensor([1])\nref = torch.tensor([2])"]
    return challenge("Where policy and reference disagree",
                     "`policy[i]` and `ref[i]` are the scores two models give response i. A pair `i < j` is "
                     "**discordant** when the models order it in opposite directions: `(policy[i] − policy[j]) · (ref[i] − "
                     "ref[j]) < 0` (ties in either score don't count). Count the discordant pairs.",
                     f"policy = {tt(p)}\nref = {tt(r)}", sol, tests, "1 ≤ n ≤ 2·10⁴",
                     speed="O(n log n): sort by one score, count inversions of the other with a Fenwick tree")


@template(M, "challenge")
def max_preference_pairs(rng):
    n, m = rng.randint(3, 6), rng.randint(3, 6)
    c = [rng.randint(0, 20) for _ in range(n)]
    r = [rng.randint(0, 20) for _ in range(m)]
    d = rng.randint(0, 5)
    sol = ("cs, rs = sorted(chosen.tolist()), sorted(rejected.tolist())\nj = pairs = 0\nfor v in cs:\n"
           "    if j < len(rs) and v >= rs[j] + margin:\n        pairs += 1\n        j += 1\nresult = pairs")
    big = ("g = np.random.default_rng({s})\nchosen = torch.from_numpy(g.integers(0, {hi}, size=20000))\n"
           "rejected = torch.from_numpy(g.integers(0, {hi}, size={m}))\nmargin = {d}")
    tests = [big.format(s=seeded(rng), hi=10**9, m=20000, d=10**8), big.format(s=seeded(rng), hi=10, m=5000, d=0),
             big.format(s=seeded(rng), hi=100, m=20000, d=99), "chosen = torch.tensor([3])\nrejected = torch.tensor([3])\nmargin = 1"]
    return challenge("Building preference pairs",
                     "To build a DPO dataset you pair responses: `chosen` holds the reward-model scores of candidate "
                     "good responses and `rejected` those of candidate bad ones. A pair (chosen i, rejected j) is usable "
                     "if `chosen[i] ≥ rejected[j] + margin`, and each response may be used in at most one pair. Return "
                     "the **maximum** number of usable pairs.",
                     f"chosen = {tt(c)}\nrejected = {tt(r)}\nmargin = {d}", sol, tests, "1 ≤ n, m ≤ 2·10⁴",
                     speed="O(n log n): sort both, then a greedy two-pointer sweep")
