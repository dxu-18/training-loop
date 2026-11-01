"""Small text helpers: word classes, virtual columns, indentation."""

from __future__ import annotations

import re
import unicodedata


def is_kw(c: str) -> bool:
    """'iskeyword' (@,48-57,_,192-255): letters, digits, underscore."""
    return c.isalnum() or c == "_" or ("À" <= c <= "ÿ")


def cls(c: str, big: bool = False) -> int:
    """0 blank, 1 punctuation, 2 keyword (WORDs: everything non-blank is 2)."""
    if c in " \t" or c == "":
        return 0
    if big:
        return 2
    return 2 if is_kw(c) else 1


def char_width(c: str) -> int:
    if unicodedata.east_asian_width(c) in ("W", "F"):
        return 2
    return 1


def vcol(line: str, col: int, ts: int = 8) -> int:
    """Screen column where character `col` starts (0-based)."""
    v = 0
    for ch in line[:col]:
        if ch == "\t":
            v += ts - v % ts
        elif ord(ch) < 32 or ch == "\x7f":
            v += 2
        else:
            v += char_width(ch)
    return v


def vcol_end(line: str, col: int, ts: int = 8) -> int:
    """Last screen column the character at `col` covers."""
    if col >= len(line):
        return vcol(line, col, ts)
    ch = line[col]
    start = vcol(line, col, ts)
    if ch == "\t":
        return start + (ts - start % ts) - 1
    if ord(ch) < 32 or ch == "\x7f":
        return start + 1
    return start + char_width(ch) - 1


def col_at_vcol(line: str, want: int, ts: int = 8, past_end: bool = False) -> int:
    """The character covering screen column `want` (or the last one; past_end allows len(line))."""
    v = 0
    for i, ch in enumerate(line):
        w = ts - v % ts if ch == "\t" else (2 if ord(ch) < 32 or ch == "\x7f" else char_width(ch))
        if v + w > want:
            return i
        v += w
    if past_end:
        return len(line)
    return max(0, len(line) - 1)


def display_width(line: str, ts: int = 8) -> int:
    return vcol(line, len(line), ts)


def indent_of(line: str, ts: int = 8) -> int:
    """Width of the leading whitespace."""
    m = re.match(r"[ \t]*", line)
    return vcol(m.group(0), len(m.group(0)), ts)


def first_nonblank(line: str) -> int:
    m = re.match(r"[ \t]*", line)
    n = len(m.group(0))
    return min(n, max(0, len(line) - 1)) if line else 0


def make_indent(width: int, expandtab: bool, ts: int = 8) -> str:
    if width <= 0:
        return ""
    if expandtab:
        return " " * width
    return "\t" * (width // ts) + " " * (width % ts)


def set_indent(line: str, width: int, expandtab: bool, ts: int = 8) -> str:
    m = re.match(r"[ \t]*", line)
    return make_indent(width, expandtab, ts) + line[len(m.group(0)):]


def _prevnonblank(lines: list[str], l: int) -> int:
    while l >= 0 and not lines[l].strip():
        l -= 1
    return l


def _blank_strings(s: str) -> str:
    """The line with its comment cut off and string contents blanked, columns kept (for bracket matching)."""
    out, q, i = [], None, 0
    while i < len(s):
        c = s[i]
        if q:
            if c == "\\" and i + 1 < len(s):
                out.append("  ")
                i += 2
                continue
            out.append(c if c == q else " ")
            if c == q:
                q = None
        elif c in "'\"":
            q = c
            out.append(c)
        elif c == "#":
            break
        else:
            out.append(c)
        i += 1
    return "".join(out)


def search_bracket(lines: list[str], lnum: int, col: int = 0, maxoff: int = 50):
    """(line, col) of the innermost unmatched ( [ { before (lnum, col), strings and comments skipped, looking back
    at most `maxoff` lines (python.vim's s:SearchBracket), or None."""
    depth = 0
    for q in range(lnum, max(-1, lnum - maxoff - 1), -1):
        text = _blank_strings(lines[q])
        hi = min(col, len(text)) if q == lnum else len(text)
        for j in range(hi - 1, -1, -1):
            ch = text[j]
            if ch in ")]}":
                depth += 1
            elif ch in "([{":
                if depth == 0:
                    return q, j
                depth -= 1
    return None


def python_indent(lines: list[str], lnum: int, sw: int = 4, ts: int = 8) -> int | None:
    """Indent for line `lnum` of a Python file: Vim's autoload/python.vim (python#GetIndent), or None for its -1
    (keep the indent the line has: a new line has the autoindent by then). A negative result also means keep."""
    def ind(l):
        return indent_of(lines[l], ts)

    def res(x):
        return x if x is not None and x >= 0 else None

    def dedented(expected):
        return ind(lnum) <= expected - sw

    if lnum > 0 and lines[lnum - 1].endswith("\\"):
        if lnum > 1 and lines[lnum - 2].endswith("\\"):
            return ind(lnum - 1)
        return ind(lnum - 1) + 2 * sw
    plnum = _prevnonblank(lines, lnum - 1)
    if plnum < 0:
        return 0
    br = search_bracket(lines, lnum, 0)
    if br is not None:
        bl, bc = br
        if bc != len(lines[bl].rstrip("\n")) - 1:
            return vcol(lines[bl], bc, ts) + 1               # align with the open paren
    pbr = search_bracket(lines, plnum, 0)
    if pbr is not None:
        parlnum = pbr[0]
        plindent, plnumstart = ind(parlnum), parlnum
    else:
        parlnum = None
        plindent, plnumstart = ind(plnum), plnum
    if br is not None:
        p = br[0]
        if p == plnum:
            if search_bracket(lines, br[0], br[1]) is not None:
                return ind(plnum) + sw                        # nested paren
            return ind(plnum) + 2 * sw                        # open paren at the end of the line
        if plnumstart == p:
            return ind(plnum)
        return plindent
    pline = _strip_comment(lines[plnum])
    if re.search(r":\s*$", pline):
        return plindent + sw
    if re.match(r"\s*(break|continue|raise|return|pass)\b", lines[plnum]):
        if dedented(ind(plnum)):
            return None
        return res(ind(plnum) - sw)
    cur = lines[lnum] if lnum < len(lines) else ""
    if re.match(r"\s*(except|finally)\b", cur):
        for q in range(lnum - 1, -1, -1):
            if re.match(r"\s*(try|except)\b", lines[q]):
                return None if ind(q) >= ind(lnum) else ind(q)
        return None
    if re.match(r"\s*(elif|else)\b", cur):
        if re.match(r"\s*(for|if|elif|try)\b", lines[plnumstart]):
            return plindent
        if dedented(plindent):
            return None
        return res(plindent - sw)
    if parlnum is not None:
        return None if dedented(plindent) else plindent
    return None


_YAML_MAPKEY = re.compile(r"^\s*(?!#)\S(?:[^:#]|:(?=\S))*:(?:\s|$)")
_YAML_LIST = re.compile(r"^\s*-(?: |$)")


def yaml_indent(lines: list[str], lnum: int, sw: int = 2, ts: int = 8) -> int | None:
    """Vim's indent/yaml.vim (GetYAMLIndent), without the multiline-scalar options."""
    def ind(l):
        return indent_of(lines[l], ts)

    def prev_less(l, cur=None):
        p = _prevnonblank(lines, l - 1)
        cur = ind(l) if cur is None else cur
        while p >= 0 and ind(p) >= cur and not lines[p].lstrip().startswith("#"):
            p = _prevnonblank(lines, p - 1)
        return p

    def prev_le_matching(l, rx):
        p = prev_less(l, ind(l) + 1)
        while p >= 0 and not rx.match(lines[p]):
            p = prev_less(p)
        return p

    if lnum == 0 or _prevnonblank(lines, lnum - 1) < 0:
        return 0
    prevlnum = _prevnonblank(lines, lnum - 1)
    previndent = ind(prevlnum)
    line = lines[lnum]
    if line.lstrip().startswith("#") and lnum > 0 and lines[lnum - 1].lstrip().startswith("#"):
        return previndent
    if re.match(r"\s*[\]}]", line):
        p = prev_less(lnum)
        return ind(p) if p >= 0 else 0
    while lines[prevlnum].lstrip().startswith("#"):
        prevlnum = _prevnonblank(lines, prevlnum - 1)
        if prevlnum < 0:
            return previndent
    prevline, previndent = lines[prevlnum], ind(prevlnum)
    if re.search(r"[{\[:]$|[:-] [|>][+-]?(?:\s+#.*|\s*)$", prevline):
        return previndent + sw
    if _YAML_LIST.match(line):
        p = prev_le_matching(lnum, _YAML_LIST)
        return ind(p) if p >= 0 else 0
    if _YAML_MAPKEY.match(line):
        p = prev_le_matching(lnum, _YAML_MAPKEY)
        if p < 0:
            return 0
        return ind(p) + 2 if re.match(r"\s*- ", lines[p]) else ind(p)
    return previndent


def _strip_strings(s: str) -> str:
    return re.sub(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"", '""', _strip_comment(s))


def _strip_comment(s: str) -> str:
    out, q = [], None
    i = 0
    while i < len(s):
        c = s[i]
        if q:
            out.append(c)
            if c == "\\" and i + 1 < len(s):
                out.append(s[i + 1])
                i += 2
                continue
            if c == q:
                q = None
        elif c in "'\"":
            q = c
            out.append(c)
        elif c == "#":
            break
        else:
            out.append(c)
        i += 1
    return "".join(out)


def brace_indent(lines: list[str], lnum: int, sw: int = 4, ts: int = 8, cindent: bool = True) -> int:
    """Brace languages and JSON: one shiftwidth more than the line with the innermost open { or [ (two inside an
    open parenthesis), closers line up with that line. cindent (C, C++, Java ...): a statement lines up with the
    previous one in its block (javascript.vim and json.vim don't do that)."""
    br = search_bracket(lines, lnum, 0)
    if br is None:
        return 0
    bl, bc = br
    base = indent_of(lines[bl], ts)
    cur = lines[lnum].strip() if lnum < len(lines) else ""
    if cur and cur[0] in "}])":
        return base
    p = _prevnonblank(lines, lnum - 1)
    if cindent and lines[bl][bc] != "(" and p > bl:
        prev = _blank_strings(lines[p].split("//")[0]).rstrip()
        if prev.endswith(("}", "]")):
            # after a block: line up with the statement the block belongs to
            op = search_bracket(lines, p, len(prev) - 1)
            return indent_of(lines[op[0]], ts) if op is not None else indent_of(lines[p], ts)
        if prev.endswith((";", ",")):
            return indent_of(lines[p], ts)                # the next statement / element: as the previous one
    return base + (2 * sw if lines[bl][bc] == "(" else sw)
