"""1.14 Linear algebra & einsum."""

from ... import themes as T
from ._common import P, arr, challenge, mat, seeded, template

M = "1.14"


def _g(rng) -> str:
    return f"g = np.random.default_rng({seeded(rng)})\n"


# ================================================================= lite

@template(M, "lite")
def matmul_basic(rng):
    n, d, m = rng.randint(2, 4), rng.randint(2, 4), rng.randint(2, 4)
    g = _g(rng)
    desc, setup, sol = rng.choice([
        (f"the matrix product of `A` `({n}, {d})` and `B` `({d}, {m})` with the `@` operator",
         g + f"A = g.integers(-3, 4, size=({n}, {d}))\nB = g.integers(-3, 4, size=({d}, {m}))", "A @ B"),
        (f"the matrix-vector product of `A` `({n}, {d})` and `v` `({d},)`",
         g + f"A = g.integers(-3, 4, size=({n}, {d}))\nv = g.integers(-3, 4, size={d})", "A @ v"),
        (f"the dot product of the vectors `u` and `v` (length {d})",
         g + f"u = g.integers(-3, 4, size={d})\nv = g.integers(-3, 4, size={d})", "u @ v"),
        (f"the vector-matrix product `v @ A` for `v` `({n},)` and `A` `({n}, {d})`",
         g + f"v = g.integers(-3, 4, size={n})\nA = g.integers(-3, 4, size=({n}, {d}))", "v @ A"),
    ])
    return P(f"Compute {desc}.", setup, sol)


@template(M, "lite")
def trace_det(rng):
    n = rng.randint(2, 4)
    setup = _g(rng) + f"A = g.integers(-4, 5, size=({n}, {n}))"
    desc, sol, req = rng.choice([
        ("the **trace** (sum of the diagonal) with `np.trace`", "np.trace(A)", "trace"),
        ("the **diagonal** as a 1-D array with `np.diagonal`", "np.diagonal(A)", "diagonal|diag"),
        ("the **determinant** with `np.linalg.det`", "np.linalg.det(A)", "det"),
        ("the **rank** with `np.linalg.matrix_rank`", "np.linalg.matrix_rank(A)", "matrix_rank"),
    ])
    return P(f"`A` is a {n}×{n} integer matrix. Compute {desc}.", setup, sol, require=[req])


# ================================================================= easy

@template(M, "easy")
def matmul_scores(rng):
    th = T.theme(rng)
    n, d = rng.randint(3, 6), rng.randint(2, 4)
    X = [[rng.randint(-5, 9) for _ in range(d)] for _ in range(n)]
    w = [round(rng.uniform(-2, 2), 1) for _ in range(d)]
    if rng.random() < 0.5:
        b = round(rng.uniform(-3, 3), 1)
        return P(f"`X` has one row per {th.noun} ({n}) and one column per feature ({d}). A linear "
                 f"model has weights `w` and bias `b`. Compute every row's score `X·w + b` using the "
                 f"`@` operator.", f"X = {mat(X)}\nw = {arr(w)}\nb = {b}", "X @ w + b", no_loops=True)
    k = rng.randint(2, 4)
    W = [[round(rng.uniform(-1, 1), 1) for _ in range(k)] for _ in range(d)]
    return P(f"`X` has one row per {th.noun} ({n}) and one column per feature ({d}). `W` is a "
             f"({d}, {k}) weight matrix of a linear layer with {k} outputs. Compute the ({n}, {k}) "
             f"output with a matrix product.", f"X = {mat(X)}\nW = {mat(W)}", "X @ W", no_loops=True)


@template(M, "easy")
def vector_products(rng):
    n = rng.randint(2, 4)
    g = _g(rng)
    desc, setup, sol, req = rng.choice([
        ("the **outer product** of `u` and `v` (`np.outer`)",
         g + f"u = g.integers(-3, 4, size={n})\nv = g.integers(-3, 4, size={n + 1})", "np.outer(u, v)", "outer"),
        ("the **cross product** of the 3-D vectors `u` and `v` (`np.cross`), e.g. a surface normal",
         g + "u = g.integers(-3, 4, size=3)\nv = g.integers(-3, 4, size=3)", "np.cross(u, v)", "cross"),
        ("the **Kronecker product** `np.kron(np.eye(2), A)`: two copies of `A` on the block diagonal",
         g + f"A = g.integers(-3, 4, size=({n}, {n}))", "np.kron(np.eye(2), A)", "kron"),
        ("the **inner product** of `u` and `v` (`np.inner`)",
         g + f"u = g.integers(-3, 4, size={n})\nv = g.integers(-3, 4, size={n})", "np.inner(u, v)", "inner|dot|vdot"),
    ])
    return P(f"Compute {desc}.", setup, sol, require=[req])


@template(M, "easy")
def batched_at(rng):
    B, n, d, m = rng.randint(2, 3), rng.randint(2, 3), rng.randint(2, 4), rng.randint(2, 3)
    g = _g(rng)
    if rng.random() < 0.5:
        return P(f"`X` holds a batch of {B} sequences of {n} token vectors, shape `({B}, {n}, {d})`, and `W` is "
                 f"a `({d}, {m})` projection. Apply `W` to every token at once with `@` (it broadcasts over the "
                 f"batch axis): result shape `({B}, {n}, {m})`.",
                 g + f"X = g.integers(-2, 3, size=({B}, {n}, {d}))\nW = g.integers(-2, 3, size=({d}, {m}))", "X @ W")
    return P(f"`A` has shape `({B}, {n}, {d})` and `C` has shape `({B}, {d}, {m})`: {B} pairs of matrices. "
             f"Multiply each pair with one `@` (batched matmul): result shape `({B}, {n}, {m})`.",
             g + f"A = g.integers(-2, 3, size=({B}, {n}, {d}))\nC = g.integers(-2, 3, size=({B}, {d}, {m}))", "A @ C")


# ================================================================= medium

@template(M, "medium")
def einsum_ops(rng):
    n, d, k, b = rng.randint(3, 6), rng.randint(2, 5), rng.randint(2, 4), rng.randint(2, 3)
    variant = rng.choice(["rowdot", "outer", "bmm", "scores", "traces"])
    gen = _g(rng)
    if variant == "rowdot":
        return P(f"`Q` and `K` hold query and key vectors for {n} tokens (shape `({n}, {d})`). Use "
                 f"`np.einsum` to compute each token's score: the dot product of row i of `Q` with "
                 f"row i of `K` (shape `({n},)`).",
                 gen + f"Q = g.integers(-3, 4, size=({n}, {d}))\nK = g.integers(-3, 4, size=({n}, {d}))",
                 "np.einsum('ij,ij->i', Q, K)", require=["einsum"])
    if variant == "outer":
        return P(f"`a` (length {n}) and `b` (length {k}) are two vectors. Use `np.einsum` to compute "
                 f"their **outer product** `M[i, j] = a[i] * b[j]`.",
                 gen + f"a = g.integers(-3, 4, size={n})\nb = g.integers(-3, 4, size={k})",
                 "np.einsum('i,j->ij', a, b)", require=["einsum"])
    if variant == "bmm":
        return P(f"`A` has shape `({b}, {n}, {d})` and `B` has shape `({b}, {d}, {k})`: a batch of "
                 f"{b} matrix pairs. Use `np.einsum` to compute the **batched matrix product** "
                 f"(shape `({b}, {n}, {k})`).",
                 gen + f"A = g.integers(-3, 4, size=({b}, {n}, {d}))\nB = g.integers(-3, 4, size=({b}, {d}, {k}))",
                 "np.einsum('bnd,bdk->bnk', A, B)", require=["einsum"])
    if variant == "traces":
        return P(f"`S` is a stack of {b} square matrices, shape `({b}, {n}, {n})`. Use `np.einsum` to compute the "
                 f"**trace of every matrix** (shape `({b},)`).", gen + f"S = g.integers(-3, 4, size=({b}, {n}, {n}))",
                 "np.einsum('bii->b', S)", require=["einsum"])
    return P(f"`Q` and `K` have shape `({b}, {n}, {d})`: (heads, tokens, head_dim). Use `np.einsum` to "
             f"compute attention scores `S[h, i, j] = Q[h, i] · K[h, j]` (shape `({b}, {n}, {n})`).",
             gen + f"Q = g.integers(-3, 4, size=({b}, {n}, {d}))\nK = g.integers(-3, 4, size=({b}, {n}, {d}))",
             "np.einsum('hid,hjd->hij', Q, K)", require=["einsum"])


@template(M, "medium")
def linalg_ops(rng):
    n = rng.randint(2, 4)
    variant = rng.choice(["solve", "norms", "unit", "inv"])
    g = _g(rng)
    if variant == "solve":
        return P(f"Solve the linear system `A @ x = b` for `x` with `np.linalg.solve` "
                 f"(`A` is {n}×{n}; faster and more stable than `inv(A) @ b`).",
                 g + f"A = g.integers(-3, 4, size=({n}, {n})) + {2 * n} * np.eye({n})\n"
                 f"b = g.integers(-9, 10, size={n})", "np.linalg.solve(A, b)", require=["solve"])
    if variant == "inv":
        return P(f"Compute the inverse of the {n}×{n} matrix `A` with `np.linalg.inv`.",
                 g + f"A = g.integers(-3, 4, size=({n}, {n})) + {2 * n} * np.eye({n})",
                 "np.linalg.inv(A)", require=["inv"])
    th = T.theme(rng)
    k, d = rng.randint(3, 6), rng.randint(3, 5)
    setup = g + f"E = g.normal(size=({k}, {d})).round(2)"
    if variant == "norms":
        return P(f"`E` holds {d}-dimensional embeddings of {k} {th.plural}, one per row. Compute the "
                 f"**L2 norm of each row** (shape `({k},)`) with `np.linalg.norm`.", setup,
                 "np.linalg.norm(E, axis=1)", require=["norm"])
    return P(f"`E` holds {d}-dimensional embeddings of {k} {th.plural}, one per row. **L2-normalize** "
             f"every row so it has length 1 (`np.linalg.norm` with `axis=1, keepdims=True`).", setup,
             "E / np.linalg.norm(E, axis=1, keepdims=True)", require=["norm"], no_loops=True)


@template(M, "medium")
def svd_low_rank(rng):
    r, c = rng.randint(3, 5), rng.randint(3, 5)
    setup = _g(rng) + f"W = g.normal(size=({r}, {c})).round(2)"
    variant = rng.choice(["rank1", "svals", "energy"])
    if variant == "rank1":
        return P(f"Compress the `({r}, {c})` weight matrix `W` to its best **rank-1 approximation** with "
                 f"`np.linalg.svd(W, full_matrices=False)`: `S[0] · outer(U[:, 0], Vt[0])`.", setup,
                 "U, S, Vt = np.linalg.svd(W, full_matrices=False)\nresult = S[0] * np.outer(U[:, 0], Vt[0])",
                 require=["svd"])
    if variant == "svals":
        return P(f"Return the **singular values** of `W` (largest first), e.g. to inspect how low-rank a "
                 f"fine-tuning update is (`np.linalg.svd(..., compute_uv=False)` or `np.linalg.svdvals`).",
                 setup, "np.linalg.svd(W, compute_uv=False)", require=["svd|svdvals"])
    k = rng.randint(1, 2)
    return P(f"What fraction of `W`'s energy (sum of squared singular values) do the top {k} singular values "
             f"keep? That's how much a rank-{k} LoRA-style approximation preserves.", setup,
             f"S = np.linalg.svd(W, compute_uv=False)\nresult = (S[:{k}] ** 2).sum() / (S ** 2).sum()",
             require=["svd|svdvals"])


@template(M, "medium")
def symmetric_eigen(rng):
    n = rng.randint(2, 4)
    setup = _g(rng) + f"X = g.normal(size=(10, {n})).round(2)\nC = X.T @ X / 10"
    desc, sol, req = rng.choice([
        ("all its **eigenvalues** in ascending order (`np.linalg.eigvalsh`, for symmetric matrices)",
         "np.linalg.eigvalsh(C)", "eigvalsh|eigh"),
        ("its **largest eigenvalue** (`np.linalg.eigh` returns them ascending)", "np.linalg.eigh(C)[0][-1]", "eigh|eigvalsh"),
        ("its **condition number** (`np.linalg.cond`): large means numerically fragile", "np.linalg.cond(C)", "cond"),
        ("whether it is positive definite: try `np.linalg.cholesky` and return True, or False if it raises "
         "`np.linalg.LinAlgError`",
         "try:\n    np.linalg.cholesky(C)\n    result = True\nexcept np.linalg.LinAlgError:\n    result = False",
         "cholesky"),
    ])
    return P(f"`C` is a `({n}, {n})` symmetric covariance matrix of features. Compute {desc}.", setup, sol,
             require=[req])


@template(M, "medium")
def matrix_utilities(rng):
    n = rng.randint(2, 3)
    g = _g(rng)
    k = rng.randint(2, 5)
    desc, setup, sol, req = rng.choice([
        (f"`A` raised to the power {k} (`np.linalg.matrix_power`), e.g. {k} steps of a Markov chain",
         g + f"A = g.integers(0, 3, size=({n}, {n}))", f"np.linalg.matrix_power(A, {k})", "matrix_power"),
        ("the **Frobenius norm** of `A` (`np.linalg.norm(A, 'fro')`)", g + f"A = g.normal(size=({n}, {n + 1})).round(2)",
         "np.linalg.norm(A, 'fro')", "norm"),
        ("the least-squares solution of the **overdetermined** system `A @ x ≈ b` via the pseudo-inverse "
         "(`np.linalg.pinv(A) @ b`)", g + f"A = g.normal(size=({n + 3}, {n})).round(2)\nb = g.normal(size={n + 3}).round(2)",
         "np.linalg.pinv(A) @ b", "pinv|lstsq"),
        ("the product `A @ B @ C` with `np.linalg.multi_dot` (it picks the cheapest order)",
         g + f"A = g.integers(-2, 3, size=({n}, 6))\nB = g.integers(-2, 3, size=(6, 2))\nC = g.integers(-2, 3, size=(2, 5))",
         "np.linalg.multi_dot([A, B, C])", "multi_dot"),
    ])
    return P(f"Compute {desc}.", setup, sol, require=[req])


# ================================================================= hard

@template(M, "hard")
def cosine_top_k(rng):
    th = T.theme(rng)
    n, d, k = rng.randint(6, 9), rng.randint(4, 6), rng.randint(2, 3)
    names = T.items(rng, th, n)
    s = seeded(rng)
    return P(f"`E` holds a {d}-dimensional embedding for each of {n} {th.plural} (`{th.var}`), and `q` "
             f"is the embedding of a search query. Rank the {th.plural} by **cosine similarity** to `q` "
             f"and return the names of the {k} most similar, most similar first. No loops.",
             f"{th.var} = {arr(names)}\ng = np.random.default_rng({s})\nE = g.normal(size=({n}, {d})).round(2)\n"
             f"q = g.normal(size={d}).round(2)",
             f"sims = E @ q / (np.linalg.norm(E, axis=1) * np.linalg.norm(q))\n"
             f"result = {th.var}[np.argsort(sims)[::-1][:{k}]]", no_loops=True)


@template(M, "hard")
def ridge_regression(rng):
    n, d = rng.randint(8, 12), rng.randint(2, 4)
    lam = rng.choice([0.1, 1.0, 10.0])
    setup = _g(rng) + f"X = g.normal(size=({n}, {d})).round(2)\ny = g.normal(size={n}).round(2)\nlam = {lam}"
    return P(f"Fit **ridge regression** (no intercept) in closed form: `w = (XᵀX + λI)⁻¹ Xᵀy` with λ = `lam`. "
             f"Use `np.linalg.solve` instead of an explicit inverse. Return `w` (shape `({d},)`).", setup,
             "result = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)", require=["solve"], no_loops=True)


@template(M, "hard")
def attention_head(rng):
    Tn, d, dv = rng.randint(3, 5), rng.choice([2, 4]), rng.randint(2, 3)
    causal = rng.random() < 0.5
    setup = _g(rng) + (f"Q = g.normal(size=({Tn}, {d})).round(2)\nK = g.normal(size=({Tn}, {d})).round(2)\n"
                       f"V = g.normal(size=({Tn}, {dv})).round(2)")
    mask_txt = (" Apply a **causal mask** first: set scores where key j > query i to −inf." if causal else "")
    mask_code = "scores = np.where(np.tri(len(Q), dtype=bool), scores, -np.inf)\n" if causal else ""
    return P(f"Implement one **attention head** for {Tn} tokens: scores = `Q @ K.T / sqrt(d)` (d = {d}), a "
             f"row-wise softmax (subtract the row max first), then multiply by `V`.{mask_txt} Return the "
             f"`({Tn}, {dv})` output. No loops.", setup,
             f"scores = Q @ K.T / np.sqrt(Q.shape[1])\n{mask_code}"
             "w = np.exp(scores - scores.max(axis=1, keepdims=True))\nw /= w.sum(axis=1, keepdims=True)\nresult = w @ V",
             no_loops=True)


@template(M, "hard")
def pca_explained(rng):
    n, d, k = rng.randint(8, 12), rng.randint(3, 4), rng.randint(1, 2)
    setup = _g(rng) + f"X = g.normal(size=({n}, {d})) @ g.normal(size=({d}, {d})).round(1)"
    if rng.random() < 0.5:
        return P(f"Run **PCA** on `X` (`({n}, {d})`): center the columns, take the SVD of the centered data, and "
                 f"return the **explained-variance ratio** of every component (`S² / sum(S²)`, largest first).",
                 setup, "Xc = X - X.mean(axis=0)\nS = np.linalg.svd(Xc, compute_uv=False)\nresult = S ** 2 / (S ** 2).sum()",
                 require=["svd|svdvals|eigh|eigvalsh"], no_loops=True)
    return P(f"Run **PCA** on `X` (`({n}, {d})`): center the columns, take the SVD `Xc = U S Vt`, and project "
             f"onto the top {k} component{'s' if k > 1 else ''}. Component signs are arbitrary, so return the "
             f"**squared** projections `(Xc @ Vt[:{k}].T) ** 2` (shape `({n}, {k})`).", setup,
             f"Xc = X - X.mean(axis=0)\nU, S, Vt = np.linalg.svd(Xc, full_matrices=False)\nresult = (Xc @ Vt[:{k}].T) ** 2",
             require=["svd|eigh"], no_loops=True)


# ================================================================= challenge

@template(M, "challenge")
def walks_mod(rng):
    n = rng.randint(3, 4)
    A = [[int(rng.random() < 0.5) for _ in range(n)] for _ in range(n)]
    k = rng.randint(2, 6)
    s_, t_ = rng.randrange(n), rng.randrange(n)
    sol = ("MOD = 10**9 + 7\nAm = [[int(v) for v in row] for row in A.tolist()]\nn = len(Am)\n\n"
           "def mul(X, Y):\n    return [[sum(X[i][l] * Y[l][j] for l in range(n)) % MOD for j in range(n)] for i in range(n)]\n\n"
           "R = [[int(i == j) for j in range(n)] for i in range(n)]\nP_, e = Am, k\nwhile e:\n    if e & 1:\n"
           "        R = mul(R, P_)\n    P_ = mul(P_, P_)\n    e >>= 1\nresult = R[s][t]")
    tests = ["A = np.ones((12, 12), dtype=int)\nk = 10**18\ns, t = 0, 11",
             f"A = (np.random.default_rng({seeded(rng)}).random((12, 12)) < 0.3).astype(int)\nk = 10**18 - 7\ns, t = 3, 5",
             "A = np.eye(12, k=1, dtype=int)\nk = 11\ns, t = 0, 11", "A = np.array([[1]])\nk = 10**18\ns, t = 0, 0"]
    return challenge("Count the walks",
                     "`A` is the adjacency matrix of a state machine (`A[i][j] = 1` if you can go from state i to "
                     "state j in one step; self-loops allowed). Count the **walks of exactly k steps** from state "
                     "`s` to state `t`, modulo 1 000 000 007. (`np.linalg.matrix_power` on int64 would overflow: do "
                     "the arithmetic with Python ints and `% MOD` after every multiplication.)",
                     f"A = {mat(A)}\nk = {k}\ns, t = {s_}, {t_}", sol, tests,
                     "1 ≤ n ≤ 12, 1 ≤ k ≤ 10¹⁸", speed="O(n³ log k) with fast matrix exponentiation")


@template(M, "challenge")
def gf2_rank(rng):
    m, b = rng.randint(4, 7), rng.randint(3, 6)
    B = [[rng.randint(0, 1) for _ in range(b)] for _ in range(m)]
    sol = ("vals = (bits @ (1 << np.arange(bits.shape[1]))).tolist()\nbasis = {}\nfor v in vals:\n"
           "    while v:\n        top = v.bit_length() - 1\n        if top not in basis:\n            basis[top] = v\n"
           "            break\n        v ^= basis[top]\nresult = len(basis)")
    tests = [f"bits = np.random.default_rng({seeded(rng)}).integers(0, 2, size=(20000, 30))",
             f"bits = np.random.default_rng({seeded(rng)}).integers(0, 2, size=(20000, 30)) * np.r_[np.ones(10, int), np.zeros(20, int)]",
             "bits = np.zeros((500, 30), dtype=int)", "bits = np.eye(30, dtype=int)"]
    return challenge("Rank over GF(2)",
                     "`bits` is an m×b matrix of 0s and 1s: each row is a parity check of an error-correcting "
                     "code. Find its **rank over GF(2)**, where addition is XOR (1 + 1 = 0). (`np.linalg.matrix_rank` "
                     "works over the reals and gives the wrong answer.)", f"bits = {mat(B)}", sol, tests,
                     "1 ≤ m ≤ 2·10⁴, 1 ≤ b ≤ 30", speed="O(m·b): pack each row into an int and keep an XOR basis")
