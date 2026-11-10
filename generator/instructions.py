"""Instructions shown with a problem: the functions it is meant to practise (Bash: commands, options, syntax and
keys), each with a one-line description.

Only what is new is listed, so the list stays short. Names the problem requires (`require`) always come first;
then:
  1. names this module introduces that first appear at this problem's tier or the tier before ("recent");
  2. else every name this module introduces that the solution uses;
  3. else (a problem that only reuses older names) the most recently introduced of them, at most two.
"Introduced" means: the first module, and within it the first tier, whose reference solutions use the name
(precomputed by scripts/make_instructions.py, which also writes the descriptions: library docstrings, bash-lite's
manual pages and hand-written notes, in generator/data/instructions.json).
"""

from __future__ import annotations

import ast
import json
import re
from functools import lru_cache
from pathlib import Path

TIERS = ("lite", "easy", "medium", "hard", "challenge")
MAX_ITEMS = 4
MAX_OLDER = 2

# module aliases whose dotted names are kept whole (np.random.default_rng, F.softmax, nn.Linear ...)
ALIASES = ("np", "pd", "torch", "F", "nn", "S", "tl", "triton", "dist", "dcp", "benchmark", "plt", "math", "sklearn")
# plain-Python methods that are not the point of a problem (dict.items, str.join ...)
PY_METHODS = {"append", "extend", "update", "items", "keys", "values", "get", "format", "pop", "add", "copy",
              "setdefault", "remove", "insert", "count", "index", "sort", "split", "join", "strip", "lower", "upper",
              "startswith", "endswith", "replace", "read", "write", "close", "seek", "flush"}
PY_METHODS_KEEP = {"pandas": {"count", "index", "sort", "split", "join", "strip", "lower", "upper", "startswith",
                              "endswith", "replace", "get", "pop", "update", "copy", "add", "values", "keys", "items"},
                   "numpy": {"sort", "copy", "split", "flush", "index", "count"},
                   "pytorch": {"split", "add", "copy", "index", "sort", "count"},
                   "sklearn": {"split", "copy", "index", "sort", "count", "get", "values", "keys", "items"}}
# any import statement (sklearn problems: the imports themselves aren't APIs to describe)
IMPORT_STMT = re.compile(r"^[ \t]*(?:from[ \t]+\S+[ \t]+import[ \t]+(?:\([^)]*\)|[^\n]+)|import[ \t]+[^\n]+)", re.M)
# `from sklearn.preprocessing import StandardScaler, MinMaxScaler as MM` (one line or parenthesized)
SK_IMPORT = re.compile(r"^[ \t]*from[ \t]+sklearn[\w.]*[ \t]+import[ \t]+(\([^)]*\)|[^\n]+)", re.M)


def sk_imported(code: str) -> tuple[dict[str, str], list[tuple[int, int]]]:
    """Names a piece of code imports from sklearn ({local name: real name}) and the spans of those imports."""
    names, spans = {}, []
    for m in SK_IMPORT.finditer(code):
        spans.append(m.span())
        for part in m[1].strip("()").replace("\n", " ").split(","):
            name, _, alias = part.strip().partition(" as ")
            if name.strip().isidentifier():
                names[(alias or name).strip()] = name.strip()
    return names, spans

DATA_FILE = Path(__file__).with_name("data") / "instructions.json"


@lru_cache(maxsize=1)
def data() -> dict:
    try:
        return json.loads(DATA_FILE.read_text())
    except (OSError, ValueError):
        return {}


def canon(key: str, section: str = "") -> str:
    """One name per API: np.cumsum and x.cumsum() are both 'cumsum'; Bash features and syntax are their own name."""
    if section in ("bash", "git", "docker") or key.startswith("syntax:"):
        return key
    return key.rsplit(".", 1)[-1]


BASH_WORD = re.compile(r"^[A-Za-z][\w.+-]*$")
BASH_OPT = re.compile(r"^(-[A-Za-z0-9]|--[a-z][\w-]*)$")


def bash_kept(f: str) -> bool:
    """Feature names worth describing: a command, `cmd -x` / `cmd --long`, `cmd sub` (pip install), or syntax.
    Not literal arguments (`find .`, `cut -f2`, `date +%T`, `./run.sh`)."""
    words = f.split(" ")
    if not f.strip():
        return False
    if not BASH_WORD.match(words[0]):
        return not f.startswith("./") and not f[0].isdigit()
    if len(words) == 1:
        return True
    sub = re.compile(r"^[a-z][a-z-]+$")
    if len(words) == 2:
        return bool(BASH_OPT.match(words[1]) or sub.match(words[1]))
    if len(words) == 3 and words[1] == "-m":                   # python3 -m venv
        return BASH_WORD.match(words[2]) is not None
    if len(words) == 4:                                       # docker compose up -d, gh pr create --title
        return bool(sub.match(words[1]) and sub.match(words[2]) and BASH_OPT.match(words[3]))
    return len(words) == 3 and bool(sub.match(words[1])) and bool(sub.match(words[2]) or BASH_OPT.match(words[2]))


# ---------------------------------------------------------------- Python: names and syntax in a solution

def _blank_strings(code: str) -> str:
    """Comments and string contents out (keeping offsets): 'model.pt' is not an API."""
    def blank(m):
        return m[0][0] + " " * (len(m[0]) - 2) + m[0][-1] if len(m[0]) >= 2 else m[0]
    code = re.sub(r"(?s)(\"\"\"|\'\'\').*?\1", lambda m: " " * len(m[0]), code)
    code = re.sub(r"[rbfu]?(\"[^\"\n]*\"|'[^'\n]*')", blank, code)
    return re.sub(r"#[^\n]*", lambda m: " " * len(m[0]), code)


def py_names(code: str, section: str) -> list[tuple[int, str]]:
    """(position, key) of the API names a solution uses, in order: 'np.cumsum', '.reshape', 'F.softmax' ..."""
    code = _blank_strings(code)
    raw = code
    if section == "sklearn":
        code = IMPORT_STMT.sub(lambda m: " " * len(m[0]), code)
    keep = PY_METHODS_KEEP.get(section, set())
    out = []
    for m in re.finditer(r"(?<![\w.])([A-Za-z_]\w*)((?:\.[A-Za-z_]\w*)+)", code):
        root, rest = m[1], m[2][1:].split(".")
        if root in ALIASES:
            out.append((m.start(), root + "." + ".".join(rest)))
            continue
        pos = m.start() + len(root)
        for attr in rest:
            pos = code.index("." + attr, pos)
            if root in ("self", "ctx") and attr == rest[0]:
                continue
            if attr not in PY_METHODS or attr in keep:
                out.append((pos, "." + attr))
    for m in re.finditer(r"[)\]]((?:\s*\.\s*[A-Za-z_]\w*)+)", code):   # (...).a.b() chains
        for a in re.finditer(r"[A-Za-z_]\w*", m[1]):
            if a[0] not in PY_METHODS or a[0] in keep:
                out.append((m.start(1) + a.start(), "." + a[0]))
    if section == "sklearn":                       # StandardScaler(), cross_val_score(...): imported, used bare
        names, spans = sk_imported(raw)
        for local, real in names.items():
            for m in re.finditer(rf"(?<![\w.]){re.escape(local)}\b(?!\s*=(?!=))", code):
                if not any(a <= m.start() < b for a, b in spans):
                    out.append((m.start(), real))
                    break
    return sorted(set(out))


def py_syntax(code: str) -> list[tuple[int, str]]:
    """(position, key) of the indexing and operator syntax a solution uses: 'syntax:slice-step', 'syntax:@' ..."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    lines = [0]
    for ln in code.split("\n"):
        lines.append(lines[-1] + len(ln) + 1)

    def pos(node):
        return lines[node.lineno - 1] + node.col_offset

    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            idx = node.slice
            parts = idx.elts if isinstance(idx, ast.Tuple) else [idx]
            for part in parts:
                if isinstance(part, ast.Constant) and part.value is None:
                    out.append((pos(node), "syntax:newaxis"))
                elif isinstance(part, ast.Constant) and part.value is Ellipsis:
                    out.append((pos(node), "syntax:ellipsis"))
                elif isinstance(part, ast.Slice) and part.step is not None:
                    out.append((pos(node), "syntax:slice-step"))
                elif isinstance(part, ast.Slice):
                    out.append((pos(node), "syntax:slice"))
                elif isinstance(part, (ast.Compare, ast.BoolOp)) or (
                        isinstance(part, ast.BinOp) and isinstance(part.op, (ast.BitAnd, ast.BitOr))) or (
                        isinstance(part, ast.UnaryOp) and isinstance(part.op, ast.Invert)):
                    out.append((pos(node), "syntax:mask"))
                elif isinstance(part, ast.List):
                    out.append((pos(node), "syntax:fancy"))
            if isinstance(node.ctx, ast.Store):
                out.append((pos(node), "syntax:setitem"))
            if isinstance(idx, ast.Tuple) and len(parts) > 1:
                out.append((pos(node), "syntax:multi-index"))
            elif isinstance(idx, (ast.Constant, ast.Name, ast.UnaryOp)) and not isinstance(node.ctx, ast.Store):
                out.append((pos(node), "syntax:index"))
        elif isinstance(node, ast.BinOp):
            op = {ast.MatMult: "@", ast.Mod: "%", ast.FloorDiv: "//", ast.Pow: "**", ast.BitAnd: "&",
                  ast.BitOr: "|", ast.Add: "arith", ast.Sub: "arith", ast.Mult: "arith",
                  ast.Div: "arith"}.get(type(node.op))
            if op:
                out.append((pos(node), "syntax:" + op))
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Invert):
            out.append((pos(node), "syntax:~"))
        elif isinstance(node, ast.Compare) and any(isinstance(o, ast.In) for o in node.ops):
            out.append((pos(node), "syntax:in"))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "len":
            out.append((pos(node), "syntax:len"))
        elif isinstance(node, ast.Lambda):
            out.append((pos(node), "syntax:lambda"))
        elif isinstance(node, ast.ClassDef):
            out.append((pos(node), "syntax:class"))
        elif isinstance(node, ast.With):
            out.append((pos(node), "syntax:with"))
    return out


# ---------------------------------------------------------------- Bash: commands, options, syntax, keys

def bash_names(problem) -> list[tuple[int, str]]:
    """(position, feature) of what a Bash solution uses: 'grep -c', '|', 'param:%', 'C-r' ..."""
    from .bashlite.features import features
    sol = problem["solution"] if isinstance(problem, dict) else problem.solution
    setup = problem["setup"] if isinstance(problem, dict) else problem.setup
    if "#: keys" in setup:
        from .bashlite.keys import KeyError_, parse_keys
        try:
            keys = parse_keys(sol)
        except KeyError_:
            return []
        out, typed, k = [], [], 0
        for kind, v in keys:
            k += 1
            if kind == "key":
                out.append((k, v))
                if v == "Enter":
                    typed.append("\n")
            else:
                typed.append(v)
        text = "".join(typed)
        return out + [(1000 + p, f) for p, f in _bash_ordered(text, features(text))]
    return _bash_ordered(sol, features(sol))


def _bash_ordered(text: str, feats: set[str]) -> list[tuple[int, str]]:
    out = []
    for f in feats:
        probe = f
        if f.startswith("param:"):
            probe = "${"
        elif f in ("glob", "$name", "array", "array[@]"):
            probe = {"glob": "*", "$name": "$", "array": "(", "array[@]": "[@]"}[f]
        elif " " in f and not f.startswith("[["):
            probe = f.split(" ")[1] if f.split(" ")[1].startswith("-") else f
        i = text.find(probe)
        if i < 0:
            i = text.find(f.split(" ")[0])
        out.append((i if i >= 0 else len(text), f))
    return sorted(out)


# ---------------------------------------------------------------- the list for a problem

def names_of(problem) -> list[str]:
    """The keys a problem's solution uses, in order of first use (one per canonical name)."""
    p = problem if isinstance(problem, dict) else problem.to_dict()
    if p.get("kind") == "bash":
        found = bash_names(p)
    else:
        found = sorted(py_names(p["solution"], p.get("section", "")) + py_syntax(p["solution"]))
    section = "bash" if p.get("kind") == "bash" else p.get("section", "")
    if section == "bash":
        found = [x for x in found if bash_kept(x[1])]
    seen, out = set(), []
    for _, key in found:
        c = canon(key, section)
        if c not in seen:
            seen.add(c)
            out.append(key)
    return out


def describe(section: str, key: str) -> dict | None:
    """{name, sig, desc} for a key, from the data file (falling back to another spelling of the same API)."""
    d = data().get(section, {})
    docs = d.get("docs", {})
    entry = docs.get(key)
    if entry is None:
        alt = d.get("canon", {}).get(canon(key, section))
        entry = docs.get(alt) if alt else None
    if entry is None or not entry.get("d"):
        return None
    name, sig = entry.get("n") or key, entry.get("s", "")
    if sig and (sig == name or sig.startswith(name.rstrip("()") + "(")):
        name, sig = sig, ""       # the signature already shows the name
    return {"name": name, "sig": sig, "desc": entry["d"], "key": key}


def instructions(problem, with_keys: bool = False) -> list[dict]:
    p = problem if isinstance(problem, dict) else problem.to_dict()
    if p.get("vim"):
        return vim_instructions(p, with_keys)
    if p.get("latex"):
        return latex_instructions(p, with_keys)
    if p.get("sql"):
        return sql_instructions(p, with_keys)
    section, module = p.get("section", ""), p.get("module", "")
    tier = TIERS.index(p["tier"]) if p.get("tier") in TIERS else 0
    intro = data().get(section, {}).get("intro", {})
    order = data().get(section, {}).get("modules", [])
    rank = {m: i for i, m in enumerate(order)}
    here = rank.get(module, len(order))
    items = []
    for key in names_of(p):
        doc = describe(section, key)
        if doc is None:
            continue
        where = intro.get(canon(key, section))
        mod_i = rank.get(where[0], here) if where else here
        tier_i = where[1] if where else tier
        items.append((key, doc, mod_i, tier_i))
    required = [x for x in items if canon(x[0], section) in _required(p, section)]
    in_module = [x for x in items if x[2] >= here]
    recent = [x for x in in_module if x[3] >= tier - 1]
    chosen = recent or in_module
    if not chosen and not required:
        older = sorted(items, key=lambda x: (-x[2], -x[3]))[:MAX_OLDER]
        chosen = [x for x in items if x in older]
    chosen = (required + [x for x in chosen if x not in required])[:MAX_ITEMS]
    out = [x[1] for x in chosen]
    if section in ("bash", "git", "docker"):
        out = _merge_commands(out)
    keys = [x["key"] for x in out]     # an accessor (s.str) next to one of its methods says nothing more
    out = [x for x in out if not (x["key"] in (".str", ".dt", ".cat") and len(keys) > 1)]
    return out if with_keys else [{"name": x["name"], "sig": x["sig"], "desc": x["desc"]} for x in out]


def _required(p: dict, section: str) -> set[str]:
    """The names a problem's `require` asks for (every alternative of "a|b")."""
    out = set()
    for req in p.get("require") or []:
        if section in ("bash", "git", "docker"):
            from .bashlite.features import alternatives
            out |= set(alternatives(req))
        else:
            out |= {canon(a, section) for a in req.split("|")}
    return out


def _merge_commands(items: list[dict]) -> list[dict]:
    """`ls` and `ls -l` together read as one entry: ls -l, with both descriptions."""
    keys = {x["key"] for x in items}
    out = []
    for x in items:
        if any(k.startswith(x["key"] + " ") for k in keys):
            continue                              # `ls` is said by `ls -l` (below), `pip` by `pip install`
        cmd = x["key"].split(" ")[0]
        if " " in x["key"] and cmd in keys and x["key"].split(" ")[1].startswith("-"):
            base = next(y for y in items if y["key"] == cmd)
            x = dict(x, desc=f"{base['desc'].rstrip('.')}; {x['desc'][0].lower() + x['desc'][1:]}")
        out.append(x)
    return out


# ---------------------------------------------------------------- Vim (COMP_GUIDE section 9)

VIM_OPERATORS = ("g~", "gu", "gU", "gq", "gw", "zf", "d", "c", "y", ">", "<", "=", "!")
VIM_TRIVIAL = {"<Esc>", "i", ":", "h", "j", "k", "l", "count", "a", "o"}
VIM_UNLESS_REQUIRED = {":", "<Esc>", "count"}     # in nearly every solution: listed only when a problem asks for it
VIM_MAX = MAX_ITEMS


def vim_instructions(p: dict, with_keys: bool = False) -> list[dict]:
    """The commands a Vim problem practises, from the guide's catalog (bashlite/vim/catalog.py): the required ones,
    then those this module introduces that the reference keys use (ciw rather than c and iw), else at most two
    older ones."""
    from .bashlite.vim.catalog import CATALOG, MODULE_OF, shown
    module = p.get("module", "")
    uses = [u for u in (p.get("vim") or {}).get("uses", []) if u in CATALOG]
    covered = set()
    for u in uses:
        for op in VIM_OPERATORS:
            if u.startswith(op) and len(u) > len(op) and u[len(op):] in CATALOG:
                covered |= {op, u[len(op):]}
                break
    req = [a for r in p.get("require") or [] for a in r.split("|") if a in CATALOG]
    here = [u for u in uses if MODULE_OF.get(u) == module and u not in covered and u not in VIM_UNLESS_REQUIRED]
    older = [u for u in uses if MODULE_OF.get(u, "") < module and u not in covered and u not in VIM_TRIVIAL]
    chosen = []
    for k in req + here + (older[::-1][:2] if len(req + here) < 2 else []):
        if k not in chosen:
            chosen.append(k)
    out = [{"name": shown(k), "sig": "", "desc": CATALOG[k]["desc"], "key": k} for k in chosen[:VIM_MAX]]
    return out if with_keys else [{"name": x["name"], "sig": "", "desc": x["desc"]} for x in out]


# ---------------------------------------------------------------- LaTeX (COMP_GUIDE section 11)

def latex_instructions(p: dict, with_keys: bool = False) -> list[dict]:
    """The commands, environments and packages a LaTeX problem practises (latexlite/catalog.py): the required ones,
    then those this module introduces that the reference uses, else at most two older ones."""
    from .latexlite.catalog import CATALOG, MODULE_OF, TRIVIAL, shown
    module = p.get("module", "")
    uses = [u for u in (p.get("latex") or {}).get("uses", []) if u in CATALOG]
    req = [a for r in p.get("require") or [] for a in r.split("|") if a in CATALOG]
    here = [u for u in uses if MODULE_OF.get(u) == module and u not in TRIVIAL]
    older = [u for u in uses if MODULE_OF.get(u, "") < module and u not in TRIVIAL]
    chosen = []
    for k in req + here + (older[::-1][:2] if len(req + here) < 2 else []):
        if k not in chosen:
            chosen.append(k)
    out = [{"name": shown(k), "sig": "", "desc": CATALOG[k], "key": k} for k in chosen[:MAX_ITEMS]]
    return out if with_keys else [{"name": x["name"], "sig": "", "desc": x["desc"]} for x in out]


# ---------------------------------------------------------------- SQL (COMP_GUIDE section 10)

SQL_TRIVIAL = {"SELECT", "AS", "WHERE", "JOIN", "::", "~", "?", "IN", "->"}


def sql_instructions(p: dict, with_keys: bool = False) -> list[dict]:
    """The SQL keywords, functions and psql commands a problem practises (postgreslite/glossary.py): the required
    ones, then those this module introduces that the reference uses, else at most two older ones."""
    from .postgreslite.glossary import CATALOG, MODULE_OF, shown, uses
    module = p.get("module", "")
    used = uses(p.get("solution", ""))
    req = [a for r in p.get("require") or [] for a in r.split("|") if a in CATALOG]
    here = [u for u in used if MODULE_OF.get(u) == module]
    older = [u for u in used if MODULE_OF.get(u, "") < module and u not in SQL_TRIVIAL]
    chosen = []
    for k in req + here + (older[::-1][:2] if len(req + here) < 2 else []):
        if k not in chosen:
            chosen.append(k)
    out = [{"name": shown(k), "sig": "", "desc": CATALOG[k], "key": k} for k in chosen[:MAX_ITEMS]]
    return out if with_keys else [{"name": x["name"], "sig": "", "desc": x["desc"]} for x in out]
