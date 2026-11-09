"""git diff's output: comparing trees, the index and the working tree; unified patches (headers, modes, renames,
binary files, hunks with the function line), --stat, --numstat, --name-only, --name-status, --shortstat and
--word-diff."""

from __future__ import annotations

import re

from ..vfs import LazyData
from .objects import EMPTY_BLOB
from .xdiff import diff_lines

ZERO = "0" * 40
BINARY_PROBE = 8000


def is_binary(data) -> bool:
    return b"\0" in bytes(data[:BINARY_PROBE])


def split_lines(data) -> list[str]:
    text = bytes(data).decode("utf-8", "surrogateescape")
    if not text:
        return []
    lines = text.split("\n")
    out = [ln + "\n" for ln in lines[:-1]]
    if lines[-1]:
        out.append(lines[-1])
    return out


class Change:
    """One path's change: status A M D R T, old/new (mode, sha), similarity for renames."""
    __slots__ = ("status", "old_path", "new_path", "old_mode", "new_mode", "old_sha", "new_sha", "score", "old_data",
                 "new_data")

    def __init__(self, status, old_path, new_path, old_mode, new_mode, old_sha, new_sha, score=0):
        self.status, self.old_path, self.new_path = status, old_path, new_path
        self.old_mode, self.new_mode, self.old_sha, self.new_sha, self.score = old_mode, new_mode, old_sha, new_sha, score
        self.old_data = self.new_data = None

    @property
    def path(self):
        return self.new_path or self.old_path


def compare(old: dict, new: dict, renames: bool = False, db=None, paths=None) -> list[Change]:
    """old/new: path -> (mode str, sha). -> sorted changes."""
    out = []
    keys = sorted(set(old) | set(new))
    for p in keys:
        if paths is not None and not any(_under(p, q) for q in paths):
            continue
        o, n = old.get(p), new.get(p)
        if o == n:
            continue
        if o is None:
            out.append(Change("A", p, p, None, n[0], ZERO, n[1]))
        elif n is None:
            out.append(Change("D", p, p, o[0], None, o[1], ZERO))
        else:
            st = "T" if (o[0] == "120000") != (n[0] == "120000") else "M"
            out.append(Change(st, p, p, o[0], n[0], o[1], n[1]))
    if renames and db is not None:
        out = detect_renames(out, db)
    return out


def _under(p: str, q: str) -> bool:
    q = q.rstrip("/")
    return q in ("", ".") or p == q or p.startswith(q + "/")


def similarity(a, b) -> int:
    """git-like similarity (0-100): the share of the bigger file's lines that also appear in the other."""
    if bytes(a) == bytes(b):
        return 100
    la, lb = split_lines(a), split_lines(b)
    if not la or not lb:
        return 0
    from collections import Counter
    ca, cb = Counter(la), Counter(lb)
    common = sum(min(ca[k], cb[k]) * len(k) for k in ca)
    total = max(sum(len(x) for x in la), sum(len(x) for x in lb))
    return int(100 * common / total) if total else 0


def detect_renames(changes: list[Change], db, threshold: int = 50) -> list[Change]:
    adds = [c for c in changes if c.status == "A"]
    dels = [c for c in changes if c.status == "D"]
    if not adds or not dels or len(adds) * len(dels) > 400:
        return changes
    used_a, used_d = set(), set()
    pairs = []
    for d in dels:                       # exact renames first
        for a in adds:
            if id(a) not in used_a and a.new_sha == d.old_sha:
                pairs.append((d, a, 100))
                used_a.add(id(a))
                used_d.add(id(d))
                break
    cands = []
    for d in dels:
        if id(d) in used_d:
            continue
        od = db.raw(d.old_sha)
        if is_binary(od):
            continue
        for a in adds:
            if id(a) in used_a:
                continue
            nd = db.raw(a.new_sha)
            if is_binary(nd):
                continue
            s = similarity(od, nd)
            if s >= threshold:
                cands.append((s, d, a))
    for s, d, a in sorted(cands, key=lambda x: -x[0]):
        if id(d) in used_d or id(a) in used_a:
            continue
        pairs.append((d, a, s))
        used_a.add(id(a))
        used_d.add(id(d))
    out = [c for c in changes if id(c) not in used_a and id(c) not in used_d]
    for d, a, s in pairs:
        out.append(Change("R", d.old_path, a.new_path, d.old_mode, a.new_mode, d.old_sha, a.new_sha, s))
    return sorted(out, key=lambda c: (c.path, c.old_path))


# ---------------------------------------------------------------------------- unified patches

def func_line(lines: list[str], before: int, rx=None) -> str:
    """The nearest line above `before` that looks like a function header (git's default: starts with a letter, _ or
    $), cut to 80 bytes."""
    i = before - 1
    while i >= 0:
        ln = lines[i]
        if rx is not None:
            m = rx.search(ln.rstrip("\n"))
            if m:
                return (m.group(1) if m.groups() else m.group(0))[:80].rstrip()
        elif ln and (ln[0].isalpha() or ln[0] in "_$"):
            return ln.rstrip("\n")[:80].rstrip()
        i -= 1
    return ""


PYTHON_FUNCNAME = re.compile(r"^[ \t]*((class|(async[ \t]+)?def)[ \t].*)$")


def hunks(a: list[str], b: list[str], ctx: int = 3, inter: int = 0):
    """Group changes into hunks: [(a start, a count, b start, b count, [(tag, line)])]."""
    changes = diff_lines(a, b)
    if not changes:
        return []
    groups = []
    cur = [changes[0]]
    for ch in changes[1:]:
        prev = cur[-1]
        gap = ch[0] - (prev[0] + prev[1])
        if gap <= 2 * ctx + inter:
            cur.append(ch)
        else:
            groups.append(cur)
            cur = [ch]
    groups.append(cur)
    out = []
    for g in groups:
        first, last = g[0], g[-1]
        s1 = max(0, first[0] - ctx)
        e1 = min(len(a), last[0] + last[1] + ctx)
        s2 = first[2] - (first[0] - s1)
        e2 = last[2] + last[3] + (e1 - (last[0] + last[1]))
        body = []
        i1 = s1
        for (c1, n1, c2, n2) in g:
            while i1 < c1:
                body.append((" ", a[i1]))
                i1 += 1
            for k in range(c1, c1 + n1):
                body.append(("-", a[k]))
            for k in range(c2, c2 + n2):
                body.append(("+", b[k]))
            i1 = c1 + n1
        while i1 < e1:
            body.append((" ", a[i1]))
            i1 += 1
        out.append((s1, e1 - s1, s2, e2 - s2, body))
    return out


def _range(start, count):
    s = start + 1 if count else start
    return f"{s}" if count == 1 else f"{s},{count}"


class Colors:
    def __init__(self, on: bool):
        self.on = on

    def __call__(self, code, text):
        if not self.on or not text:
            return text
        return f"\x1b[{code}m{text}\x1b[m"


def patch(ch: Change, db, read_new=None, read_old=None, ctx: int = 3, color: Colors | None = None,
          src_prefix="a/", dst_prefix="b/", funcrx_of=None, abbrev=7, binary_ok=False, word_diff=None,
          full_index=False) -> str:
    """The patch text for one change. read_new/read_old: data for ids that are working-tree files (sha ZERO)."""
    color = color or Colors(False)
    a_path, b_path = ch.old_path, ch.new_path
    old = read_old(ch) if read_old else (db.raw(ch.old_sha) if ch.old_sha != ZERO and ch.status != "A" else b"")
    new = read_new(ch) if read_new else (db.raw(ch.new_sha) if ch.new_sha != ZERO and ch.status != "D" else b"")
    if ch.status == "A":
        old = b""
    if ch.status == "D":
        new = b""
    lines = [color("1", f"diff --git {src_prefix}{a_path} {dst_prefix}{b_path}")]

    def short(s):
        if s == ZERO:
            return "0" * (40 if full_index else abbrev)
        return s if full_index else db.abbrev(s, abbrev) if db.has(s) else s[:abbrev]
    if ch.status == "A":
        lines.append(color("1", f"new file mode {ch.new_mode}"))
        lines.append(color("1", f"index {short(ZERO)}..{short(ch.new_sha)}"))
    elif ch.status == "D":
        lines.append(color("1", f"deleted file mode {ch.old_mode}"))
        lines.append(color("1", f"index {short(ch.old_sha)}..{short(ZERO)}"))
    else:
        if ch.status == "R":
            lines.append(color("1", f"similarity index {ch.score}%"))
            lines.append(color("1", f"rename from {a_path}"))
            lines.append(color("1", f"rename to {b_path}"))
        if ch.old_mode != ch.new_mode:
            lines.append(color("1", f"old mode {ch.old_mode}"))
            lines.append(color("1", f"new mode {ch.new_mode}"))
            if ch.old_sha != ch.new_sha:
                lines.append(color("1", f"index {short(ch.old_sha)}..{short(ch.new_sha)}"))
        elif ch.old_sha != ch.new_sha:
            lines.append(color("1", f"index {short(ch.old_sha)}..{short(ch.new_sha)} {ch.new_mode}"))
    if ch.old_sha == ch.new_sha and ch.status in ("R", "M"):
        return "\n".join(lines) + "\n"
    if (is_binary(old) or is_binary(new)) and not binary_ok:
        aa = "/dev/null" if ch.status == "A" else f"{src_prefix}{a_path}"
        bb = "/dev/null" if ch.status == "D" else f"{dst_prefix}{b_path}"
        lines.append(f"Binary files {aa} and {bb} differ")
        return "\n".join(lines) + "\n"
    la, lb = split_lines(old), split_lines(new)
    hs = hunks(la, lb, ctx)
    if not hs:
        return "\n".join(lines) + "\n"
    lines.append(color("1", "--- " + ("/dev/null" if ch.status == "A" else f"{src_prefix}{a_path}")))
    lines.append(color("1", "+++ " + ("/dev/null" if ch.status == "D" else f"{dst_prefix}{b_path}")))
    rx = funcrx_of(b_path) if funcrx_of else None
    for s1, n1, s2, n2, body in hs:
        fn = func_line(la, s1, rx)
        head = color("36", f"@@ -{_range(s1, n1)} +{_range(s2, n2)} @@") + (f" {fn}" if fn else "")
        lines.append(head)
        if word_diff:
            lines.extend(_word_diff_body(body, word_diff, color))
            continue
        for tag, text in body:
            nl = text.endswith("\n")
            t = text[:-1] if nl else text
            if tag == "-":
                lines.append(color("31", "-" + t))
            elif tag == "+":
                lines.append(color("32", "+" + t))
            else:
                lines.append(" " + t)
            if not nl:
                lines.append("\\ No newline at end of file")
    return "\n".join(lines) + "\n"


def _word_diff_body(body, mode, color):
    """--word-diff=plain (and color): changed runs as [-old-]{+new+} inside their lines."""
    out = []
    i = 0
    while i < len(body):
        tag, text = body[i]
        if tag == " ":
            out.append(" " + text.rstrip("\n") if mode == "porcelain" else text.rstrip("\n"))
            i += 1
            continue
        olds, news = [], []
        while i < len(body) and body[i][0] == "-":
            olds.append(body[i][1])
            i += 1
        while i < len(body) and body[i][0] == "+":
            news.append(body[i][1])
            i += 1
        a = "".join(olds)
        b = "".join(news)
        out.extend(_word_merge(a, b, mode, color).split("\n"))
        if out and out[-1] == "":
            out.pop()
    return out


def _word_merge(a: str, b: str, mode, color) -> str:
    ta = re.findall(r"\S+|\s+", a)
    tb = re.findall(r"\S+|\s+", b)
    wa = [t for t in ta]
    wb = [t for t in tb]
    from difflib import SequenceMatcher
    sm = SequenceMatcher(None, wa, wb, autojunk=False)
    res = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            res.append("".join(wa[i1:i2]))
            continue
        old = "".join(wa[i1:i2])
        new = "".join(wb[j1:j2])
        if old.strip():
            lead = old[:len(old) - len(old.lstrip())]
            res.append(lead)
            core = old.strip()
            res.append(color("31", f"[-{core}-]") if mode != "color" else color("31", core))
            res.append(old[len(old.rstrip()):])
        if new.strip():
            core = new.strip()
            res.append(color("32", f"{{+{core}+}}") if mode != "color" else color("32", core))
            res.append(new[len(new.rstrip()):] if not old.strip() else "")
    return "".join(res).rstrip("\n")


def numstat(ch: Change, db, read_new=None, read_old=None):
    old = read_old(ch) if read_old else (db.raw(ch.old_sha) if ch.status != "A" else b"")
    new = read_new(ch) if read_new else (db.raw(ch.new_sha) if ch.status != "D" else b"")
    if ch.status == "A":
        old = b""
    if ch.status == "D":
        new = b""
    if is_binary(old) or is_binary(new):
        return None, None
    la, lb = split_lines(old), split_lines(new)
    add = rem = 0
    for c1, n1, c2, n2 in diff_lines(la, lb):
        rem += n1
        add += n2
    return add, rem


def stat_name(ch: Change) -> str:
    if ch.status == "R":
        a, b = ch.old_path, ch.new_path
        # common prefix/suffix compression: dir/{old => new}.py
        pre = 0
        while pre < min(len(a), len(b)) and a[pre] == b[pre]:
            pre += 1
        pre = a.rfind("/", 0, pre) + 1
        suf = 0
        while suf < min(len(a), len(b)) - pre and a[-1 - suf] == b[-1 - suf]:
            suf += 1
        while suf and a[len(a) - suf] != "/":
            suf -= 1
        if pre or suf:
            mid_a = a[pre:len(a) - suf]
            mid_b = b[pre:len(b) - suf]
            return f"{a[:pre]}{{{mid_a} => {mid_b}}}{a[len(a) - suf:]}"
        return f"{a} => {b}"
    return ch.path


def _scale(it, width, max_change):
    return 0 if not it else 1 + (it * (width - 1) // max_change)


def stat_block(rows, width: int = 80, color: Colors | None = None, summary_only=False) -> list[str]:
    """rows: [(name, added, removed, binary (old size, new size) or None)] -> git diff --stat lines and the
    summary line (diff.c show_stats: name <= 5/8 of the width, graph <= 3/8, scaled so each change shows)."""
    color = color or Colors(False)
    files = len(rows)
    ins = sum(r[1] or 0 for r in rows if r[3] is None)
    dels = sum(r[2] or 0 for r in rows if r[3] is None)
    summ = f" {files} file{'s' if files != 1 else ''} changed"
    if ins or not dels:
        summ += f", {ins} insertion{'s' if ins != 1 else ''}(+)"
    if dels or not ins:
        summ += f", {dels} deletion{'s' if dels != 1 else ''}(-)"
    if summary_only:
        return [summ]
    if not rows:
        return []
    max_change = max(((r[1] or 0) + (r[2] or 0)) for r in rows if r[3] is None) if any(r[3] is None for r in rows) else 0
    max_len = max(len(r[0]) for r in rows)
    bin_width = 0
    for r in rows:
        if r[3] is not None:
            bin_width = max(bin_width, len(f"Bin {r[3][0]} -> {r[3][1]} bytes"))
    number_width = max(len(str(max_change)), 0)
    if width < 16 + 6 + number_width:
        width = 16 + 6 + number_width
    graph_width = max_change if max_change + 4 > bin_width else bin_width - 4
    name_width = max_len
    if name_width + number_width + 6 + graph_width > width:
        if graph_width > width * 3 // 8 - number_width - 6:
            graph_width = width * 3 // 8 - number_width - 6
            if graph_width < 6:
                graph_width = 6
        if name_width > width - number_width - 6 - graph_width:
            name_width = width - number_width - 6 - graph_width
        else:
            graph_width = width - number_width - 6 - name_width
    out = []
    for name, add, rem, binfo in rows:
        shown, prefix = name, ""
        ln = name_width
        if name_width < len(name):
            prefix = "..."
            ln = max(0, ln - 3)
            shown = name[len(name) - ln:] if ln else ""
            slash = shown.find("/")
            if slash >= 0:
                shown = shown[slash:]
        cell = f" {prefix}{shown:<{ln}} |"
        if binfo is not None:
            out.append(f"{cell} Bin {binfo[0]} -> {binfo[1]} bytes")
            continue
        total = add + rem
        a, d = add, rem
        if graph_width <= max_change:
            t = _scale(add + rem, graph_width, max_change)
            if t < 2 and add and rem:
                t = 2
            if add < rem:
                a = _scale(add, graph_width, max_change)
                d = t - a
            else:
                d = _scale(rem, graph_width, max_change)
                a = t - d
        graph = color("32", "+" * a) + color("31", "-" * d)
        out.append(f"{cell} {total:>{number_width}}" + (f" {graph}" if total else ""))
    out.append(summ)
    return out


def data_of_worktree(repo, rel):
    node = repo.wt_node(rel)
    if node is None:
        return b""
    return repo.file_bytes(node)


def funcrx_for(repo):
    """The diff driver's funcname rule for a path (`*.py diff=python` in .gitattributes)."""
    attrs = repo.attributes() if repo.worktree else []

    def fn(path):
        import fnmatch
        for pat, base, kv in attrs:
            name = path if "/" in pat else path.rsplit("/", 1)[-1]
            if fnmatch.fnmatchcase(name, pat) and "diff=python" in kv:
                return PYTHON_FUNCNAME
        return None
    return fn


__all__ = ["Change", "compare", "patch", "numstat", "stat_block", "stat_name", "hunks", "split_lines", "is_binary",
           "Colors", "ZERO", "EMPTY_BLOB", "LazyData"]
