"""2.10 String methods (.str)."""

from ... import themes as T
from ._common import P, challenge, seeded, template

M = "2.10"

FIRST = ["alice", "bob", "carol", "dave", "erin", "frank", "grace", "heidi", "ivan", "judy", "mallory", "oscar"]
LAST = ["smith", "jones", "king", "lee", "garcia", "nguyen", "patel", "kim", "o'neil", "silva"]


def _messy_names(rng, n):
    out = []
    for _ in range(n):
        f, l_ = rng.choice(FIRST), rng.choice(LAST)
        f = rng.choice([f, f.upper(), f.capitalize()])
        l_ = rng.choice([l_, l_.upper(), l_.capitalize()])
        out.append(rng.choice(["", " ", "  "]) + f + rng.choice([" ", "  ", "   "]) + l_ + rng.choice(["", " "]))
    return out


def _items(rng, n=(4, 6)):
    th = T.theme(rng)
    return th, T.items(rng, th, rng.randint(*n))


# ================================================================= lite

@template(M, "lite")
def basic_str(rng):
    names = _messy_names(rng, rng.randint(4, 6))
    desc, sol = rng.choice([
        ("strip the surrounding whitespace", "s.str.strip()"), ("lower-case every value", "s.str.lower()"),
        ("upper-case every value", "s.str.upper()"), ("title-case every value (`str.title`)", "s.str.title()"),
        ("get the length of every value", "s.str.len()"),
    ])
    return P(f"`s` holds names typed into a sign-up form. Using the `.str` accessor, {desc}.",
             f"s = pd.Series({names!r})", sol, require=["str"])


@template(M, "lite")
def str_slices(rng):
    th, items = _items(rng)
    k = rng.randint(2, 4)
    desc, sol = rng.choice([
        (f"the first {k} characters of each name", f"s.str[:{k}]"),
        (f"the last {k} characters of each name", f"s.str[-{k}:]"),
        ("the first character of each name (`str.get(0)` or `str[0]`)", "s.str[0]"),
        (f"characters 1 up to (not including) {k + 1} of each name (`str.slice`)", f"s.str.slice(1, {k + 1})"),
    ])
    return P(f"`s` holds {th.noun} names. Get {desc}.", f"s = pd.Series({items!r})", sol, require=["str"])


# ================================================================= easy

@template(M, "easy")
def str_matching(rng):
    th, items = _items(rng, (5, 7))
    word = items[0].split()[0][:3]
    end = items[1][-2:]
    desc, sol = rng.choice([
        (f"contains `{word.lower()!r}`, ignoring case (`str.contains(..., case=False)`)",
         f"s.str.contains({word.lower()!r}, case=False, regex=False)"),
        (f"starts with `{items[0][0]!r}` (`str.startswith`)", f"s.str.startswith({items[0][0]!r})"),
        (f"ends with `{end!r}` (`str.endswith`)", f"s.str.endswith({end!r})"),
    ])
    return P(f"`s` holds {th.noun} names. Build a boolean mask of the names that {desc}.", f"s = pd.Series({items!r})",
             sol, require=["str"])


@template(M, "easy")
def str_split(rng):
    names = [n.strip() for n in _messy_names(rng, rng.randint(4, 6))]
    names = [" ".join(n.split()) for n in names]
    desc, sol = rng.choice([
        ("split every name into a list of words (`str.split()`)", "s.str.split()"),
        ("get the **first** word of every name (`str.split().str[0]`)", "s.str.split().str[0]"),
        ("get the **last** word of every name (`str.split().str[-1]`)", "s.str.split().str[-1]"),
        ("split each name at the first space into **two columns** (`str.split(' ', n=1, expand=True)`)",
         "s.str.split(' ', n=1, expand=True)"),
    ])
    return P(f"`s` holds full names (already stripped). {desc[0].upper() + desc[1:]}.", f"s = pd.Series({names!r})",
             sol, require=["split"])


@template(M, "easy")
def str_replace(rng):
    if rng.random() < 0.5:
        names = [rng.choice(FIRST) + rng.choice(["-", "_"]) + rng.choice(LAST) for _ in range(rng.randint(4, 6))]
        ch = "-" if any("-" in n for n in names) else "_"
        return P(f"`s` holds user handles. Replace every `{ch!r}` with a space (`str.replace`; a plain string is "
                 f"matched literally by default).", f"s = pd.Series({names!r})", f"s.str.replace({ch!r}, ' ')",
                 require=["replace"])
    names = _messy_names(rng, rng.randint(4, 6))
    return P("`s` holds names with runs of several spaces. Collapse every run of whitespace into **one** space with a "
             "regex (`str.replace(r'\\s+', ' ', regex=True)`), then strip the ends.", f"s = pd.Series({names!r})",
             "s.str.replace(r'\\s+', ' ', regex=True).str.strip()", require=["replace"])


# ================================================================= medium

@template(M, "medium")
def str_extract(rng):
    n = rng.randint(4, 6)
    prefix = rng.choice(["order", "ticket", "run", "job"])
    ids = [f"{prefix}-{rng.randint(1, 9999)}-{rng.choice(['us', 'eu', 'ap'])}" for _ in range(n)]
    if rng.random() < 0.5:
        return P(f"`s` holds ids like `{ids[0]!r}`. Extract the number with `str.extract(r'(\\d+)')` and return it as an "
                 f"**int** Series (`expand=False` gives a Series).", f"s = pd.Series({ids!r})",
                 "s.str.extract(r'(\\d+)', expand=False).astype(int)", require=["extract"])
    return P(f"`s` holds ids like `{ids[0]!r}`. Extract the number and the region into a DataFrame with columns `num` "
             f"and `region`, using **named groups** in one `str.extract` call.", f"s = pd.Series({ids!r})",
             f"s.str.extract(r'{prefix}-(?P<num>\\d+)-(?P<region>\\w+)')", require=["extract"])


@template(M, "medium")
def multi_label_dummies(rng):
    tags = ["ml", "nlp", "cv", "speech", "rl", "data"]
    rows = ["|".join(sorted(rng.sample(tags, rng.randint(1, 3)))) for _ in range(rng.randint(4, 6))]
    if rng.random() < 0.5:
        return P("`tags` holds `'|'`-separated topic tags per paper. One-hot encode them into a 0/1 DataFrame with one "
                 "column per tag (`str.get_dummies(sep='|')`).", f"tags = pd.Series({rows!r})", "tags.str.get_dummies(sep='|')",
                 require=["get_dummies"])
    return P("`tags` holds `'|'`-separated topic tags per paper. Return how many papers carry **each** tag (a Series "
             "indexed by tag), via `str.get_dummies(sep='|')` and a column sum.", f"tags = pd.Series({rows!r})",
             "tags.str.get_dummies(sep='|').sum()", require=["get_dummies|split"])


@template(M, "medium")
def clean_names_chain(rng):
    names = _messy_names(rng, rng.randint(4, 6))
    return P("`s` holds messy names. Clean them in one chain: strip the ends, collapse inner whitespace runs to one "
             "space (regex), and title-case.", f"s = pd.Series({names!r})",
             "s.str.strip().str.replace(r'\\s+', ' ', regex=True).str.title()", require=["str"])


@template(M, "medium")
def str_count_pad(rng):
    th, items = _items(rng)
    ch = rng.choice("aeiou")
    nums = [str(rng.randint(1, 999)) for _ in items]
    desc, setup, sol = rng.choice([
        (f"count how many times `{ch!r}` appears in each name (`str.count`)", f"s = pd.Series({items!r})", f"s.str.count({ch!r})"),
        (f"find the position of the first `{ch!r}` in each name (−1 if absent, `str.find`)", f"s = pd.Series({items!r})",
         f"s.str.find({ch!r})"),
        ("left-pad each id with zeros to 5 characters (`str.zfill`)", f"s = pd.Series({nums!r})", "s.str.zfill(5)"),
        ("right-align each id in a field of 6 characters filled with `'.'` (`str.pad(6, side='left', fillchar='.')`)",
         f"s = pd.Series({nums!r})", "s.str.pad(6, side='left', fillchar='.')"),
    ])
    return P(f"String utilities: {desc}.", setup, sol, require=["str"])


# ================================================================= hard

@template(M, "hard")
def email_domains(rng):
    n = rng.randint(6, 9)
    doms = rng.sample(["gmail.com", "Example.org", "uni.edu", "LAB.ai", "corp.io"], 3)
    emails = [f"{rng.choice(FIRST)}{rng.randint(1, 99)}@{rng.choice(doms)}" for _ in range(n)]
    emails = [e if rng.random() < 0.7 else " " + e.upper() + " " for e in emails]
    return P("`emails` holds sign-up addresses with stray spaces and inconsistent case. Extract each address's "
             "**domain** (after `@`), normalize it (strip, lower-case), and return how many sign-ups each domain has "
             "(`value_counts`).", f"emails = pd.Series({emails!r})",
             "emails.str.strip().str.lower().str.split('@').str[-1].value_counts()", require=["str"])


@template(M, "hard")
def parse_log_fields(rng):
    n = rng.randint(4, 6)
    lines = [f"user={rng.choice(FIRST)} ms={rng.randint(5, 900)} status={rng.choice(['ok', 'err'])}" for _ in range(n)]
    return P("`log` holds lines like `'user=bob ms=123 status=ok'`. Parse them into a DataFrame with columns `user`, "
             "`ms` (int) and `status` using one `str.extract` with named groups, then return only the `'err'` rows.",
             f"log = pd.Series({lines!r})",
             "d = log.str.extract(r'user=(?P<user>\\w+) ms=(?P<ms>\\d+) status=(?P<status>\\w+)')\n"
             "d['ms'] = d['ms'].astype(int)\nresult = d[d['status'] == 'err']", require=["extract"])


@template(M, "hard")
def strip_accents(rng):
    words = rng.sample(["café", "naïve", "résumé", "jalapeño", "crème brûlée", "São Paulo", "Zürich", "façade"], 4)
    words = [rng.choice([w, w.upper(), w.title()]) for w in words]
    return P("`s` holds names with accents. Make ASCII search keys: decompose with `str.normalize('NFKD')`, drop the "
             "accent marks with `str.encode('ascii', errors='ignore')` then `str.decode('ascii')`, and lower-case.",
             f"s = pd.Series({words!r})",
             "s.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('ascii').str.lower()",
             require=["normalize"])


# ================================================================= challenge

@template(M, "challenge")
def shortest_unique_prefix(rng):
    pool = ["dog", "dove", "duck", "deer", "cat", "cow", "crow", "car", "cart", "do"]
    words = rng.sample(pool, rng.randint(4, 6))
    sol = ("w = words.tolist()\norder = sorted(range(len(w)), key=lambda i: w[i])\n\ndef lcp(a, b):\n"
           "    k = 0\n    while k < len(a) and k < len(b) and a[k] == b[k]:\n        k += 1\n    return k\n\n"
           "need = [0] * len(w)\nfor a, b in zip(order, order[1:]):\n    k = lcp(w[a], w[b])\n"
           "    need[a] = max(need[a], k)\n    need[b] = max(need[b], k)\n"
           "result = pd.Series([w[i][:min(len(w[i]), need[i] + 1)] for i in range(len(w))])")

    def big(seed, n_, alpha, length):
        return (f"g = np.random.default_rng({seed})\nletters = np.array(list('{alpha}'))\n"
                f"raw = [''.join(letters[g.integers(0, len(letters), g.integers(1, {length}))]) for _ in range({n_})]\n"
                f"words = pd.Series(list(dict.fromkeys(raw)))")

    tests = [big(seeded(rng), 20000, "abcdefghijklmnopqrstuvwxyz", 12), big(seeded(rng), 20000, "ab", 20),
             "words = pd.Series(['a', 'ab', 'abc'])"]
    return challenge("Shortest unique prefixes",
                     "`words` holds **distinct** command names for an autocomplete box. For every word, return the "
                     "**shortest prefix** that no other word starts with; if the word is itself a prefix of another "
                     "word, the whole word is its answer. Return a Series aligned with `words`.",
                     f"words = pd.Series({words!r})", sol, tests, "≤ 2·10⁴ words, each ≤ 20 characters",
                     answer_desc=" (a Series)", speed="O(total length · log n): sort, then compare neighbours")


@template(M, "challenge")
def overlapping_counts(rng):
    pat = rng.choice(["aa", "aba", "abab", "ana"])
    texts = ["".join(rng.choice("abn") for _ in range(rng.randint(4, 10))) for _ in range(rng.randint(4, 6))]
    texts[0] = pat + pat[-1] + pat
    sol = ("p = pattern\nout = []\nfor t in texts.tolist():\n    count, start = 0, t.find(p)\n    while start != -1:\n"
           "        count += 1\n        start = t.find(p, start + 1)\n    out.append(count)\nresult = pd.Series(out)")

    def big(seed, n_, length, p):
        return (f"g = np.random.default_rng({seed})\nletters = np.array(list('ab'))\n"
                f"texts = pd.Series([''.join(letters[g.integers(0, 2, {length})]) for _ in range({n_})])\npattern = {p!r}")

    tests = [big(seeded(rng), 20000, 60, "aba"), big(seeded(rng), 5000, 200, "aaaa"),
             "texts = pd.Series(['aaaa', ''])\npattern = 'aa'"]
    return challenge("Count overlapping matches",
                     "`texts` holds DNA-like strings and `pattern` a short motif. For every text, count the occurrences of "
                     "`pattern` **including overlapping ones** (`'aaaa'` contains `'aa'` 3 times). Careful: "
                     "`str.count` counts **non-overlapping** regex matches. Return an int Series aligned with `texts`.",
                     f"texts = pd.Series({texts!r})\npattern = {pat!r}", sol, tests, "≤ 2·10⁴ texts, ≤ 200 characters each",
                     answer_desc=" (a Series)", speed="O(total length) per pattern, e.g. repeated `str.find` or a lookahead regex")
