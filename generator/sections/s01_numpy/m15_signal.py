"""1.15 FFT, polynomials & signal helpers."""

import math

from ... import themes as T
from ._common import P, arr, challenge, floats, mat, pick_measure, rand_ints, seeded, template

M = "1.15"


def _signal_setup(rng, n=None):
    """A 1-second signal sampled at n Hz made of two integer-frequency sines (+ optional noise)."""
    n = n or rng.choice([64, 100, 128, 200])
    f1, f2 = rng.sample(range(2, n // 2 - 1), 2)
    a2 = rng.choice([0.3, 0.5])
    noise = rng.choice([0, 0.1])
    s = seeded(rng)
    setup = (f"sr = {n}\nt = np.arange(sr) / sr\n"
             f"sig = np.sin(2 * np.pi * {f1} * t) + {a2} * np.sin(2 * np.pi * {f2} * t)"
             + (f" + {noise} * np.random.default_rng({s}).normal(size=sr)" if noise else ""))
    return setup, f1, f2


# ================================================================= lite

@template(M, "lite")
def differences(rng):
    th = T.theme(rng)
    ax = T.axis(rng, min_labels=5, max_labels=12)
    who = rng.choice(th.items)
    m = pick_measure(rng, th, len(ax.labels))
    vals = T.values(rng, m, len(ax.labels))
    v = m.var
    if rng.random() < 0.7:
        return P(f"`{v}` holds the {m.label} of the {who} for each of {len(vals)} {ax.plural}, in order. "
                 f"Compute the change from each {ax.noun} to the next (length {len(vals) - 1}) with `np.diff`.",
                 f"{v} = {arr(vals)}", f"np.diff({v})", no_loops=True)
    return P(f"`{v}` holds the {m.label} of the {who} for each of {len(vals)} {ax.plural}, in order. Compute the "
             f"**second difference** (the change of the change, length {len(vals) - 2}) with `np.diff(..., n=2)`.",
             f"{v} = {arr(vals)}", f"np.diff({v}, n=2)", require=["diff"], no_loops=True)


@template(M, "lite")
def interpolate(rng):
    xp = sorted(rng.sample(range(0, 20), rng.randint(3, 5)))
    fp = floats(rng, len(xp), 0, 10, 1)
    q = [round(rng.uniform(xp[0], xp[-1]), 1) for _ in range(rng.randint(2, 4))]
    ctx = rng.choice(["A GPU temperature log has gaps", "A sensor reported only a few timestamps",
                      "A learning-rate schedule is defined at a few steps", "A pitch track has missing frames"])
    return P(f"{ctx}: `xp` holds the known positions and `fp` the values there. **Linearly interpolate** the "
             f"values at the positions `x_new` with `np.interp`.", f"xp = {arr(xp)}\nfp = {arr(fp)}\nx_new = {arr(q)}",
             "np.interp(x_new, xp, fp)", require=["interp"])


@template(M, "lite")
def polynomials(rng):
    if rng.random() < 0.5:
        c = [rng.randint(-3, 3) or 1 for _ in range(rng.randint(3, 4))]
        x = [rng.randint(-3, 3) for _ in range(3)]
        return P(f"`coeffs` holds a polynomial's coefficients, **highest power first** (NumPy's `poly*` "
                 f"convention). Evaluate it at every point of `x` with `np.polyval`.",
                 f"coeffs = {arr(c)}\nx = {arr(x)}", "np.polyval(coeffs, x)", require=["polyval"])
    r1, r2 = rng.sample(range(-6, 7), 2)
    return P(f"Find the roots of `x² + ({-(r1 + r2)})·x + ({r1 * r2})` with `np.roots` (coefficients highest power "
             f"first) and return them sorted ascending.", f"coeffs = {arr([1, -(r1 + r2), r1 * r2])}",
             "np.sort(np.roots(coeffs))", require=["roots"])


# ================================================================= easy

@template(M, "easy")
def fit_line(rng):
    n = rng.randint(5, 8)
    a, b = round(rng.uniform(-3, 3), 1), round(rng.uniform(-5, 5), 1)
    x = sorted(rng.sample(range(0, 30), n))
    y = [round(a * v + b + rng.gauss(0, 0.5), 2) for v in x]
    what = rng.choice(["validation loss vs. log-steps", "latency vs. batch size", "power draw vs. GPU utilization"])
    if rng.random() < 0.5:
        return P(f"Fit a straight line to {what}: use `np.polyfit(x, y, 1)` and return the coefficients "
                 f"`[slope, intercept]`.", f"x = {arr(x)}\ny = {arr(y)}", "np.polyfit(x, y, 1)", require=["polyfit"])
    q = rng.randint(30, 40)
    return P(f"Fit a straight line to {what} with `np.polyfit(x, y, 1)`, then **extrapolate**: predict y at "
             f"x = {q} with `np.polyval`.", f"x = {arr(x)}\ny = {arr(y)}", f"np.polyval(np.polyfit(x, y, 1), {q})",
             require=["polyfit"])


@template(M, "easy")
def trapezoid_area(rng):
    n = rng.randint(5, 8)
    if rng.random() < 0.5:
        y = floats(rng, n, 0, 10, 1)
        dx = rng.choice([0.5, 1.0, 0.1])
        return P(f"`power_w` holds a GPU's power draw sampled every {dx} s. Estimate the **energy** in joules "
                 f"(the area under the curve) with the trapezoid rule: `np.trapezoid(..., dx={dx})`.",
                 f"power_w = {arr(y)}", f"np.trapezoid(power_w, dx={dx})", require=["trapezoid|trapz"])
    fpr = sorted(round(rng.random(), 2) for _ in range(n - 2))
    tpr = sorted(round(rng.random(), 2) for _ in range(n - 2))
    return P(f"`fpr` and `tpr` describe a classifier's ROC curve (sorted by fpr, endpoints included). Compute the "
             f"**AUC** (area under the curve) with `np.trapezoid(tpr, fpr)`.",
             f"fpr = {arr([0.0] + fpr + [1.0])}\ntpr = {arr([0.0] + tpr + [1.0])}", "np.trapezoid(tpr, fpr)",
             require=["trapezoid|trapz"])


@template(M, "easy")
def convolve_modes(rng):
    x = [rng.randint(0, 9) for _ in range(rng.randint(5, 8))]
    k = rng.choice([[1, 0, -1], [1, 2, 1], [0.25, 0.5, 0.25], [1, -1]])
    mode, word = rng.choice([("full", f"the **full** convolution (length {len(x) + len(k) - 1})"),
                             ("same", f"the convolution with the **same** length as `x` ({len(x)})"),
                             ("valid", f"only the positions where the kernel fully overlaps (`'valid'`, length "
                                       f"{len(x) - len(k) + 1})")])
    return P(f"Apply the 1-D filter `kernel` to the signal `x` with `np.convolve`: compute {word}.",
             f"x = {arr(x)}\nkernel = {arr(k)}", f"np.convolve(x, kernel, mode='{mode}')", require=["convolve"])


# ================================================================= medium

@template(M, "medium")
def moving_average(rng):
    n, w = rng.randint(8, 14), rng.randint(3, 5)
    desc, var, vals = rng.choice([
        ("training loss logged at each step", "loss", [round(3 * math.exp(-i / 6) + rng.uniform(0, 0.4), 3) for i in range(n)]),
        ("GPU utilization (%) sampled every minute", "util", [rng.randint(20, 100) for _ in range(n)]),
        ("frame-level speech probability from a voice-activity detector", "vad", [round(rng.random(), 2) for _ in range(n)]),
        ("daily active users (thousands) of a chatbot", "dau_k", [rng.randint(40, 90) for _ in range(n)]),
    ])
    return P(f"`{var}` holds the {desc} ({n} values). Compute its **{w}-point moving average**, keeping "
             f"only full windows (length {n - w + 1}). `np.convolve(..., mode='valid')` with a kernel of "
             f"`{w}` equal weights does it in one call.", f"{var} = {arr(vals)}",
             f"np.convolve({var}, np.ones({w}) / {w}, mode='valid')", no_loops=True)


@template(M, "medium")
def dominant_frequency(rng):
    setup, f1, f2 = _signal_setup(rng)
    if rng.random() < 0.5:
        return P("`sig` is one second of audio sampled at `sr` Hz. Find its **dominant frequency** in Hz: take "
                 "`np.fft.rfft`, get the bin frequencies with `np.fft.rfftfreq(len(sig), d=1/sr)`, and return the "
                 "frequency with the largest magnitude.", setup,
                 "F = np.fft.rfft(sig)\nfreqs = np.fft.rfftfreq(len(sig), d=1 / sr)\nresult = freqs[np.argmax(np.abs(F))]",
                 require=["rfft|fft"])
    return P("`sig` is one second of audio sampled at `sr` Hz, made of two tones. Return the **two strongest "
             "frequencies** in Hz, sorted ascending (`np.fft.rfft` + `np.fft.rfftfreq`).", setup,
             "F = np.abs(np.fft.rfft(sig))\nfreqs = np.fft.rfftfreq(len(sig), d=1 / sr)\n"
             "result = np.sort(freqs[np.argsort(F)[-2:]])", require=["rfft|fft"])


@template(M, "medium")
def numeric_gradient(rng):
    n = rng.randint(5, 8)
    x = sorted(round(v, 1) for v in rng.sample([i / 2 for i in range(30)], n))
    c = [rng.randint(-2, 2) for _ in range(3)]
    y = [round(c[0] * v * v + c[1] * v + c[2], 3) for v in x]
    if rng.random() < 0.5:
        return P(f"`y` was sampled at the (unevenly spaced) points `x`. Estimate the **derivative** dy/dx at every "
                 f"point with `np.gradient(y, x)` (central differences inside, one-sided at the ends).",
                 f"x = {arr(x)}\ny = {arr(y)}", "np.gradient(y, x)", require=["gradient"])
    dx = rng.choice([0.1, 0.5, 2.0])
    return P(f"`y` was sampled every {dx} seconds. Estimate its rate of change at every sample with "
             f"`np.gradient`, passing the spacing.", f"y = {arr(y)}", f"np.gradient(y, {dx})", require=["gradient"])


# ================================================================= hard

@template(M, "hard")
def fft_lowpass(rng):
    setup, f1, f2 = _signal_setup(rng)
    lo, hi = sorted((f1, f2))
    cut = rng.randint(lo + 1, hi)
    return P(f"`sig` is one second sampled at `sr` Hz. Apply an ideal **low-pass filter** in the frequency "
             f"domain: take `np.fft.rfft`, zero every bin whose frequency (from `np.fft.rfftfreq`) is **greater than** "
             f"{cut} Hz, and transform back with `np.fft.irfft(..., n=len(sig))`.", setup,
             f"F = np.fft.rfft(sig)\nfreqs = np.fft.rfftfreq(len(sig), d=1 / sr)\nF[freqs > {cut}] = 0\n"
             "result = np.fft.irfft(F, n=len(sig))", require=["rfft|fft"], tol=1e-5)


@template(M, "hard")
def resample_audio(rng):
    sr1, sr2 = rng.choice([(8, 6), (10, 4), (6, 9), (12, 8)])
    secs = rng.randint(1, 2)
    s = seeded(rng)
    return P(f"`x` holds {secs} s of audio at `sr_in` = {sr1} Hz; a model expects `sr_out` = {sr2} Hz. **Resample** "
             f"with linear interpolation: input times `np.arange(len(x)) / sr_in`, output times "
             f"`np.arange(int(len(x) * sr_out / sr_in)) / sr_out`, then `np.interp`. (Real pipelines low-pass "
             f"first; skip that here.)",
             f"x = np.random.default_rng({s}).normal(size={sr1 * secs}).round(3)\nsr_in, sr_out = {sr1}, {sr2}",
             "t_in = np.arange(len(x)) / sr_in\nt_out = np.arange(int(len(x) * sr_out / sr_in)) / sr_out\n"
             "result = np.interp(t_out, t_in, x)", require=["interp"], no_loops=True)


@template(M, "hard")
def roc_auc_from_scores(rng):
    n = rng.randint(8, 12)
    scores = rng.sample([i / 100 for i in range(100)], n)
    y = [int(rng.random() < s_ * 1.2) for s_ in scores]
    y[0], y[1] = 1, 0
    return P(f"`scores` are a detector's distinct scores for {n} samples and `y` the true labels (1 = positive). "
             f"Compute the **ROC AUC** without loops: sort by score descending; for k = 0..n the point "
             f"(fpr_k, tpr_k) uses the top-k samples as positives (`tpr_k` = positives among them / all positives, "
             f"`fpr_k` likewise for negatives), so start both curves at 0 with `np.r_[0, np.cumsum(...)]`. Then "
             f"integrate with `np.trapezoid(tpr, fpr)`.", f"scores = {arr(scores)}\ny = {arr(y)}",
             "order = np.argsort(-scores)\nys = y[order]\ntpr = np.r_[0, np.cumsum(ys)] / ys.sum()\n"
             "fpr = np.r_[0, np.cumsum(1 - ys)] / (1 - ys).sum()\nresult = np.trapezoid(tpr, fpr)",
             require=["trapezoid|trapz"], no_loops=True)


@template(M, "hard")
def polyfit_predict(rng):
    n, deg = rng.randint(8, 12), rng.randint(2, 3)
    s = seeded(rng)
    q = sorted(round(rng.uniform(0, 5), 1) for _ in range(3))
    return P(f"Fit a degree-{deg} polynomial to the noisy points `(x, y)` with `np.polyfit`, then evaluate the "
             f"fit at the new points `x_new` with `np.polyval`.",
             f"g = np.random.default_rng({s})\nx = np.linspace(0, 5, {n})\ny = (0.5 * x ** {deg} - x + g.normal(0, 0.3, {n})).round(3)\n"
             f"x_new = {arr(q)}", f"result = np.polyval(np.polyfit(x, y, {deg}), x_new)", require=["polyfit"], tol=1e-5)


# ================================================================= challenge

@template(M, "challenge")
def best_window_length_range(rng):
    n = rng.randint(9, 13)
    vals = [rng.randint(-6, 8) for _ in range(n)]
    lo = rng.randint(1, 3)
    hi = rng.randint(lo + 1, min(n, lo + 4))
    var, story = rng.choice([
        ("gain", "`gain[i]` is the accuracy gain (can be negative) from including the i-th data shard."),
        ("reward", "`reward[i]` is the reward of an RL agent at step i (can be negative)."),
    ])
    sol = (f"from collections import deque\nvals = {var}.tolist()\nprefix = [0]\nfor x in vals:\n"
           "    prefix.append(prefix[-1] + x)\nbest = None\ndq = deque()\n"
           "for j in range(L, len(vals) + 1):\n    i = j - L\n"
           "    while dq and prefix[dq[-1]] >= prefix[i]:\n        dq.pop()\n    dq.append(i)\n"
           "    while dq[0] < j - R:\n        dq.popleft()\n    cand = prefix[j] - prefix[dq[0]]\n"
           "    best = cand if best is None else max(best, cand)\nresult = best")
    tests = [rand_ints(var, seeded(rng), -100, 101, 20000) + "\nL, R = 50, 5000",
             rand_ints(var, seeded(rng), -5, 3, 20000) + "\nL, R = 1, 20000",
             f"{var} = np.array([-3, -1, -2])\nL, R = 2, 3",
             f"{var} = np.array([4])\nL, R = 1, 1"]
    return challenge("Best window with a length limit",
                     story + " Choose one **contiguous** window whose length is between `L` and `R` "
                     "(inclusive) and maximize its total. Return that maximum total.",
                     f"{var} = {arr(vals)}\nL, R = {lo}, {hi}", sol, tests,
                     "1 ≤ L ≤ R ≤ n ≤ 2·10⁴, |values| ≤ 10³", speed="O(n): prefix sums + a monotonic deque")


@template(M, "challenge")
def count_weight_combos(rng):
    k, wmax = rng.randint(2, 3), rng.randint(2, 4)
    C = [[rng.randint(0, 2) for _ in range(wmax + 1)] for _ in range(k)]
    for row in C:
        row[rng.randrange(wmax + 1)] += 1
    W = rng.randint(0, k * wmax)
    sol = ("MOD = 10**9 + 7\ndp = np.array([1], dtype=np.int64)\nfor row in counts:\n"
           "    dp = np.convolve(dp, row) % MOD\nresult = int(dp[W]) if W < len(dp) else 0")
    big = lambda s, k_, w_, W_: (f"counts = np.random.default_rng({s}).integers(0, 11, size=({k_}, {w_ + 1}))\nW = {W_}")  # noqa: E731
    tests = [big(seeded(rng), 60, 100, 3000), big(seeded(rng), 60, 100, 5999),
             "counts = np.ones((60, 101), dtype=np.int64)\nW = 6000", "counts = np.array([[0, 1]])\nW = 0"]
    return challenge("Pick one from every group",
                     "An augmentation pipeline has k stages; at each stage you pick **exactly one** augmentation. "
                     "`counts[g][w]` is the number of augmentations in stage g that cost `w` units of compute. "
                     "Count the ways to choose one augmentation per stage with **total cost exactly W**, modulo "
                     "1 000 000 007. (Each stage is a polynomial; the answer is a coefficient of their product.)",
                     f"counts = {mat(C)}\nW = {W}", sol, tests,
                     "1 ≤ k ≤ 60, 1 ≤ len(counts[g]) ≤ 101, 0 ≤ counts[g][w] ≤ 10, 0 ≤ W ≤ 6000",
                     speed="O(k · W · w_max) with vectorized np.convolve (a pure-Python DP is too slow)")
