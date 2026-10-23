"""POSIX BRE / ERE (GNU flavour) and grep -P patterns translated to Python `re` syntax."""

from __future__ import annotations

import re
from functools import lru_cache

CLASSES = {
    "alpha": "a-zA-Z", "digit": "0-9", "alnum": "a-zA-Z0-9", "upper": "A-Z", "lower": "a-z",
    "space": " \\t\\n\\r\\f\\v", "blank": " \\t", "punct": "!-/:-@\\[-`{-~", "xdigit": "0-9a-fA-F",
    "cntrl": "\\x00-\\x1f\\x7f", "print": " -~", "graph": "!-~", "word": "a-zA-Z0-9_",
}


class RegexError(ValueError):
    pass


def _bracket(p: str, i: int, nocase_upper=False):
    """[...] starting at p[i]=='['. POSIX: backslash is literal inside brackets. -> (python class, next index)."""
    n = len(p)
    j = i + 1
    neg = False
    if j < n and p[j] == "^":
        neg = True
        j += 1
    items = []
    first = True
    while j < n:
        c = p[j]
        if c == "]" and not first:
            body = "".join(items)
            return ("[^" if neg else "[") + body + "]", j + 1
        first = False
        if c == "[" and j + 1 < n and p[j + 1] in ":=.":
            kind = p[j + 1]
            k = p.find(kind + "]", j + 2)
            if k < 0:
                raise RegexError("unterminated character class")
            name = p[j + 2:k]
            if kind == ":":
                if name not in CLASSES:
                    raise RegexError("invalid character class")
                items.append(CLASSES[name])
            else:
                items.append(re.escape(name))
            j = k + 2
            continue
        if j + 2 < n and p[j + 1] == "-" and p[j + 2] != "]":
            lo, hi = c, p[j + 2]
            if ord(lo) > ord(hi):
                raise RegexError("Invalid range end")
            items.append(_cls_char(lo) + "-" + _cls_char(hi))
            j += 3
            continue
        items.append(_cls_char(c))
        j += 1
    raise RegexError("Unmatched [, [^, [:, [., or [=")


def _cls_char(c: str) -> str:
    return "\\" + c if c in "\\]^-[" else c


def _interval(p, j):
    """Parse m,n} after '{' (ERE) — returns (text, end) or None."""
    m = re.match(r"(\d*)(,(\d*))?\}", p[j:])
    if not m or (m.group(1) == "" and not m.group(2)):
        return None
    lo = m.group(1) or "0"
    if m.group(2):
        return "{" + lo + "," + (m.group(3) or "") + "}", j + m.end()
    return "{" + lo + "}", j + m.end()


@lru_cache(maxsize=2048)
def translate(pat: str, dialect: str = "ere", sed: bool = False) -> str:
    if dialect == "pcre":
        return _pcre(pat)
    out = []
    i, n = 0, len(pat)
    bre = dialect == "bre"
    # positions where '*' is literal / '^' is an anchor: start, after a group open or alternation
    at_start = True
    depth = 0
    while i < n:
        c = pat[i]
        if c == "\\":
            if i + 1 >= n:
                raise RegexError("Trailing backslash")
            d = pat[i + 1]
            i += 2
            if bre and d == "(":
                out.append("(")
                depth += 1
                at_start = True
                continue
            if bre and d == ")":
                out.append(")")
                depth -= 1
                at_start = False
                continue
            if bre and d == "|":
                out.append("|")
                at_start = True
                continue
            if bre and d == "{":
                k = pat.find("\\}", i)
                if k < 0:
                    raise RegexError("Unmatched \\{")
                body = pat[i:k]
                if not re.fullmatch(r"\d*(,\d*)?", body) or body in ("", ","):
                    raise RegexError("Invalid content of \\{\\}")
                lo = body.split(",")[0] or "0"
                out.append("{" + lo + ("," + body.split(",")[1] if "," in body else "") + "}")
                i = k + 2
                at_start = False
                continue
            if bre and d in "+?":
                out.append(d)
                continue
            if d.isdigit() and d != "0":
                out.append("\\" + d)
                at_start = False
                continue
            if d == "<":
                out.append(r"\b(?=\w)")
                continue
            if d == ">":
                out.append(r"\b(?<=\w)")
                continue
            if d in "bBwWsS":
                out.append("\\" + d)
                at_start = False
                continue
            if d == "`":
                out.append(r"\A")
                continue
            if d == "'":
                out.append(r"\Z")
                continue
            if sed and d == "n":
                out.append("\\n")
                at_start = False
                continue
            if sed and d == "t":
                out.append("\\t")
                at_start = False
                continue
            out.append(re.escape(d))
            at_start = False
            continue
        if c == "[":
            cls, i = _bracket(pat, i)
            out.append(cls)
            at_start = False
            continue
        if c == "*":
            if at_start:
                out.append("\\*")
                at_start = False
            else:
                out.append("*")
            i += 1
            continue
        if c == "^":
            if bre and not at_start:
                out.append("\\^")
            else:
                out.append("^")
            i += 1
            continue
        if c == "$":
            if bre:
                rest = pat[i + 1:]
                if rest == "" or rest.startswith("\\)") or rest.startswith("\\|"):
                    out.append("$" if not sed else "(?=\\n?\\Z)" if False else "$")
                else:
                    out.append("\\$")
            else:
                out.append("$")
            i += 1
            at_start = False
            continue
        if not bre and c in "+?":
            if at_start:
                out.append("\\" + c)
            else:
                out.append(c)
            i += 1
            continue
        if not bre and c == "{":
            iv = _interval(pat, i + 1)
            if iv is None or at_start:
                out.append("\\{")
                i += 1
            else:
                out.append(iv[0])
                i = iv[1]
            at_start = False
            continue
        if not bre and c == "(":
            out.append("(")
            depth += 1
            i += 1
            at_start = True
            continue
        if not bre and c == ")":
            if depth == 0:
                out.append("\\)")
            else:
                out.append(")")
                depth -= 1
            i += 1
            at_start = False
            continue
        if not bre and c == "|":
            out.append("|")
            i += 1
            at_start = True
            continue
        out.append(re.escape(c) if c not in "." else ".")
        i += 1
        at_start = False
    if depth > 0:
        raise RegexError("Unmatched ( or \\(")
    return "".join(out)


def _pcre(pat: str) -> str:
    out, i, n = [], 0, len(pat)
    while i < n:
        c = pat[i]
        if c == "\\" and i + 1 < n:
            d = pat[i + 1]
            if d == "Q":
                k = pat.find("\\E", i + 2)
                k = n if k < 0 else k
                out.append(re.escape(pat[i + 2:k]))
                i = k + 2
                continue
            if d == "h":
                out.append("[ \\t]")
                i += 2
                continue
            if d == "K":
                out.append("(?P<_K>)")
                i += 2
                continue
            out.append(pat[i:i + 2])
            i += 2
            continue
        if c == "[":
            j = i + 1
            if j < n and pat[j] == "^":
                j += 1
            if j < n and pat[j] == "]":
                j += 1
            buf = [pat[i:j]]
            while j < n and pat[j] != "]":
                if pat[j] == "[" and j + 1 < n and pat[j + 1] == ":":
                    k = pat.find(":]", j)
                    if k > 0 and pat[j + 2:k] in CLASSES:
                        buf.append(CLASSES[pat[j + 2:k]])
                        j = k + 2
                        continue
                if pat[j] == "\\" and j + 1 < n:
                    buf.append(pat[j:j + 2])
                    j += 2
                    continue
                buf.append(pat[j])
                j += 1
            buf.append("]")
            out.append("".join(buf))
            i = j + 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


@lru_cache(maxsize=2048)
def compile_re(pat: str, dialect: str = "ere", icase: bool = False, sed: bool = False, multiline: bool = False):
    flags = re.S
    if icase:
        flags |= re.I
    if multiline:
        flags |= re.M
    try:
        return re.compile(translate(pat, dialect, sed), flags)
    except re.error as e:
        raise RegexError(_gnu_message(str(e))) from None


def _gnu_message(msg: str) -> str:
    if "missing )" in msg or "unbalanced parenthesis" in msg:
        return "Unmatched ( or \\("
    if "unterminated character set" in msg:
        return "Unmatched [, [^, [:, [., or [="
    if "nothing to repeat" in msg or "multiple repeat" in msg:
        return "Invalid preceding regular expression"
    if "invalid group reference" in msg or "unknown group" in msg:
        return "Invalid back reference"
    return "Invalid regular expression"


def ere_to_python(pat: str) -> str:
    """bash [[ =~ ]] uses POSIX ERE."""
    return translate(pat, "ere")
