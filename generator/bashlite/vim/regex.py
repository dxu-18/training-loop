"""Vim's regular expressions, translated to Python's `re`.

Covered: the four magic levels (\\v \\m \\M \\V), . * \\+ \\= \\? \\{n,m} \\{-n,m} (non-greedy), groups \\( \\) and
\\%( \\), alternation \\|, anchors ^ $, word boundaries \\< \\>, character classes \\s \\S \\d \\D \\w \\W \\a \\A \\l \\L
\\u \\U \\x \\X \\h \\H \\o \\O, [] with POSIX classes, \\_s \\_. (across lines), \\n, \\t \\e \\r, \\zs \\ze (where the
match starts and ends), \\c \\C (case), ~ (the last replacement), lookarounds \\@= \\@! \\@<= \\@<!.

Text is matched as the whole buffer joined with "\\n", with re.M, so ^ and $ work per line and \\n spans lines.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

KW = "0-9A-Za-z_À-ÿ"          # 'iskeyword': @,48-57,_,192-255
CLASSES = {"s": "[ \t]", "S": "[^ \t\n]", "d": "[0-9]", "D": "[^0-9\n]", "w": "[0-9A-Za-z_]", "W": "[^0-9A-Za-z_\n]",
           "a": "[A-Za-z]", "A": "[^A-Za-z\n]", "l": "[a-z]", "L": "[^a-z\n]", "u": "[A-Z]", "U": "[^A-Z\n]",
           "x": "[0-9A-Fa-f]", "X": "[^0-9A-Fa-f\n]", "h": "[A-Za-z_]", "H": "[^A-Za-z_\n]", "o": "[0-7]",
           "O": "[^0-7\n]", "k": f"[{KW}]", "K": f"[{KW}]", "i": f"[{KW}]", "I": f"[{KW}]", "f": "[^ \t\n]",
           "F": "[^ \t\n]", "p": "[ -~]", "P": "[ -~]"}
POSIX = {"alpha": "A-Za-z", "digit": "0-9", "alnum": "0-9A-Za-z", "lower": "a-z", "upper": "A-Z", "space": " \t\n\r\f\v",
         "blank": " \t", "punct": "!-/:-@\\[-`{-~", "xdigit": "0-9A-Fa-f", "cntrl": "\\x00-\\x1f\\x7f",
         "print": " -~", "graph": "!-~", "return": "\r", "tab": "\t", "escape": "\x1b", "backspace": "\b"}
WORD_START = f"(?<![{KW}])(?=[{KW}])"
WORD_END = f"(?<=[{KW}])(?![{KW}])"


class RegexError(Exception):
    pass


@dataclass
class Compiled:
    rx: re.Pattern
    has_zs: bool       # the match is group "zs" (\zs / \ze)
    source: str

    def span(self, m):
        return m.span("zs") if self.has_zs else m.span()

    def group0(self, m):
        return m.group("zs") if self.has_zs else m.group(0)


def translate(pat: str, ignorecase: bool = False, smartcase: bool = False, last_sub: str = "",
              magic: bool = True) -> tuple[str, int, bool]:
    """-> (python pattern, flags, has_zs)."""
    level = "m" if magic else "M"
    out: list[str] = []
    case = None
    zs = ze = False
    i, n = 0, len(pat)
    groups = 0
    atom_end = False   # (unused: kept for clarity of where quantifiers may go)

    def special(c):
        """Does c (without a backslash) have a special meaning at this magic level?"""
        if level == "v":
            return not (c.isalnum() or c == "_")
        if level == "m":
            return c in ".*[~^$"
        if level == "M":
            return c in "^$"
        return False   # \V: nothing is special (but ^ at the start and $ at the end)

    while i < n:
        c = pat[i]
        if c == "\\" and i + 1 < n:
            d = pat[i + 1]
            i += 2
            # backslash flips the meaning of a character at the current level
            if d in "vmMV":
                level = d
                continue
            if d == "c":
                case = "i"
                continue
            if d == "C":
                case = "s"
                continue
            if d == "z" and i < n and pat[i] in "se":
                if pat[i] == "s":
                    zs = True
                    out.append("\x00ZS\x00")
                else:
                    ze = True
                    out.append("\x00ZE\x00")
                i += 1
                continue
            if d == "_" and i < n:
                e = pat[i]
                i += 1
                if e == ".":
                    out.append("(?:.|\n)")
                elif e == "[":
                    cls, i = _bracket(pat, i, extra="\n")
                    out.append(cls)
                elif e in CLASSES:
                    out.append(CLASSES[e][:-1] + "\n]" if CLASSES[e].startswith("[^") is False else CLASSES[e].replace("\n", ""))
                elif e == "^":
                    out.append("^")
                elif e == "$":
                    out.append("$")
                continue
            if d in CLASSES:
                out.append(CLASSES[d])
                continue
            if d == "n":
                out.append("\n")
                continue
            if d == "t":
                out.append("\t")
                continue
            if d == "e":
                out.append("\x1b")
                continue
            if d == "r":
                out.append("\r")
                continue
            if d == "%":
                if i < n and pat[i] == "(":
                    out.append("(?:")
                    i += 1
                    continue
                if i < n and pat[i] in "^$":
                    out.append("\\A" if pat[i] == "^" else "\\Z")
                    i += 1
                    continue
                if i < n and pat[i] == "[":
                    raise RegexError("E70: Empty \\%[]")
                raise RegexError(f"E71: Invalid character after \\%")
            if d.isdigit() and d != "0":
                out.append("\\" + d)
                continue
            # characters whose special meaning needs a backslash at this level
            if level in ("m", "M", "V") and d in "()|+=?{@<>" or level in ("M", "V") and d in ".*[~" or \
                    level == "V" and d in "^$":
                i = _special(d, pat, i, out, level, last_sub)
                groups += d == "("
                continue
            # an escaped character that is special at this level is literal
            out.append(re.escape(d))
            continue
        if special(c):
            if c in "^" and level != "V":
                # ^ is an anchor at the start of the pattern or after \( \| \%(
                prev = "".join(out)
                if not prev or prev.endswith(("(", "|", "(?:")):
                    out.append("^")
                else:
                    out.append(re.escape(c))
                i += 1
                continue
            if c == "$":
                rest = pat[i + 1:]
                if not rest or rest.startswith(("\\|", "\\)")) or (level == "v" and rest.startswith(("|", ")"))):
                    out.append("$")
                else:
                    out.append(re.escape(c))
                i += 1
                continue
            i = _special(c, pat, i + 1, out, level, last_sub)
            groups += c == "("
            continue
        if level == "V" and c == "^" and i == 0:
            out.append("^")
            i += 1
            continue
        if level == "V" and c == "$" and i == n - 1:
            out.append("$")
            i += 1
            continue
        out.append(re.escape(c))
        i += 1

    py = "".join(out)
    if zs or ze:
        a, _, rest = py.partition("\x00ZS\x00") if zs else ("", "", py)
        b, _, cpart = rest.partition("\x00ZE\x00") if ze else (rest, "", "")
        py = (f"(?<={a})" if a and _fixed_width(a) else (f"(?:{a})" if a else "")) + f"(?P<zs>{b})" + (f"(?={cpart})" if ze else "")
        if a and not _fixed_width(a):
            py = f"(?:{a})(?P<zs>{b})" + (f"(?={cpart})" if ze else "")
    flags = re.M
    if case == "i" or (case is None and ignorecase and not (smartcase and any(ch.isupper() for ch in _letters(pat)))):
        flags |= re.I
    return py, flags, bool(zs or ze)


def _letters(pat):
    """Letters of a pattern that count for 'smartcase' (not those after a backslash)."""
    out, i = [], 0
    while i < len(pat):
        if pat[i] == "\\":
            i += 2
            continue
        out.append(pat[i])
        i += 1
    return out


def _fixed_width(p: str) -> bool:
    try:
        re.compile(f"(?<={p})")
        return True
    except re.error:
        return False


def _special(c, pat, i, out, level, last_sub):
    """A character with its special meaning; i is just past it. -> the new i."""
    n = len(pat)
    if c == ".":
        out.append(".")
    elif c == "*":
        out.append("*")
    elif c == "+":
        out.append("+")
    elif c in "=?":
        out.append("?")
    elif c == "(":
        out.append("(")
    elif c == ")":
        out.append(")")
    elif c == "|":
        out.append("|")
    elif c == "<":
        out.append(WORD_START)
    elif c == ">":
        out.append(WORD_END)
    elif c == "~":
        out.append(re.escape(last_sub))
    elif c == "[":
        cls, i2 = _bracket(pat, i)
        if cls is None:
            out.append("\\[")
        else:
            out.append(cls)
            i = i2
    elif c == "{":
        j = pat.find("}", i)
        if j < 0:
            raise RegexError("E554: Syntax error in \\{...}")
        body = pat[i:j]
        if body.endswith("\\"):
            body = body[:-1]
        i = j + 1
        lazy = body.startswith("-")
        body = body.lstrip("-")
        if body == "":
            q = "*"
        elif "," in body:
            lo, hi = body.split(",", 1)
            q = "{" + (lo or "0") + "," + hi + "}"
        else:
            q = "{" + body + "}"
        out.append(q + ("?" if lazy else ""))
    elif c == "@":
        # lookaround: applies to the previous atom; rewrite the last group
        rest = pat[i:]
        kind = None
        for k, py in (("<=", "?<="), ("<!", "?<!"), ("=", "?="), ("!", "?!"), (">", "?>")):
            if rest.startswith(k):
                kind = py
                i += len(k)
                break
        if kind is None:
            raise RegexError("E869: Unknown operator after \\@")
        # find the previous atom (a group)
        text = "".join(out)
        if text.endswith(")"):
            depth, j = 0, len(text) - 1
            while j >= 0:
                if text[j] == ")" and (j == 0 or text[j - 1] != "\\"):
                    depth += 1
                elif text[j] == "(" and (j == 0 or text[j - 1] != "\\"):
                    depth -= 1
                    if depth == 0:
                        break
                j -= 1
            inner = text[j + 1:-1]
            if inner.startswith("?:"):
                inner = inner[2:]
            out.clear()
            out.append(text[:j] + f"({kind}{inner})")
        else:
            last = out.pop() if out else ""
            out.append(f"({kind}{last})")
    elif c == "%" and i < n and pat[i] == "(":
        out.append("(?:")
        i += 1
    else:
        out.append(re.escape(c))
    return i


def _bracket(pat, i, extra=""):
    """A [] collection starting just after "[". -> (python class, index after "]") or (None, i) if not closed."""
    j = i
    n = len(pat)
    parts = []
    neg = False
    if j < n and pat[j] == "^":
        neg = True
        j += 1
    first = True
    while j < n:
        c = pat[j]
        if c == "]" and not first:
            body = "".join(parts) + extra
            if neg:
                body += "\n" if "\n" not in body else ""
            return "[" + ("^" if neg else "") + body + "]", j + 1
        first = False
        if c == "[" and pat.startswith("[:", j):
            k = pat.find(":]", j + 2)
            if k > 0 and pat[j + 2:k] in POSIX:
                parts.append(POSIX[pat[j + 2:k]])
                j = k + 2
                continue
        if c == "\\" and j + 1 < n:
            d = pat[j + 1]
            mapping = {"e": "\\x1b", "t": "\\t", "r": "\\r", "n": "\\n", "b": "\\x08", "\\": "\\\\", "]": "\\]",
                       "^": "\\^", "-": "\\-", "d": "0-9"}
            if d in mapping:
                parts.append(mapping[d])
                j += 2
                continue
            parts.append("\\\\")
            j += 1
            continue
        if c in "\\]^[":
            parts.append("\\" + c)
        elif c == "-" and parts and j + 1 < n and pat[j + 1] != "]":
            parts.append("-")
        elif c == "-":
            parts.append("\\-")
        else:
            parts.append(c)
        j += 1
    return None, i


def compile_(pat: str, ignorecase=False, smartcase=False, last_sub="", magic=True) -> Compiled:
    py, flags, zs = translate(pat, ignorecase, smartcase, last_sub, magic)
    try:
        return Compiled(re.compile(py, flags), zs, pat)
    except re.error as e:
        raise RegexError(f"E486: Pattern not found: {pat}") from e


# ---------------------------------------------------------------- the replacement part of :s

def expand_replacement(rep: str, m, cre: Compiled, last_sub: str = "", magic: bool = True) -> str:
    """Vim's sub-replace-special: & \\0-\\9 ~ \\r (line break) \\n \\t \\\\ \\u \\U \\l \\L \\e \\E."""
    out: list[str] = []
    one = None          # \u or \l: the next character
    many = None         # \U or \L: until \e / \E
    groups = m.groups()

    def add(s):
        nonlocal one
        for ch in s:
            if one == "u":
                ch, one = ch.upper(), None
            elif one == "l":
                ch, one = ch.lower(), None
            elif many == "U":
                ch = ch.upper()
            elif many == "L":
                ch = ch.lower()
            out.append(ch)

    def group(k):
        if k == 0:
            return cre.group0(m) or ""
        # (with \zs the pattern has one extra named group, which comes first in numbering only if it opened first)
        names = m.re.groupindex
        idx = k + (1 if cre.has_zs and names.get("zs", 99) <= k else 0)
        try:
            return m.group(idx) or ""
        except IndexError:
            return ""

    i = 0
    while i < len(rep):
        c = rep[i]
        if c == "\\" and i + 1 < len(rep):
            d = rep[i + 1]
            i += 2
            if d.isdigit():
                add(group(int(d)))
            elif d == "&":
                add("&" if magic else group(0))
            elif d == "~":
                add("~" if magic else last_sub)
            elif d == "r":
                out.append("\n")      # a line break
            elif d == "n":
                out.append("\x00")
            elif d == "t":
                add("\t")
            elif d in "uU":
                if d == "u":
                    one = "u"
                else:
                    many = "U"
            elif d in "lL":
                if d == "l":
                    one = "l"
                else:
                    many = "L"
            elif d in "eE":
                many = None
            else:
                add(d)
            continue
        if c == "&" and magic:
            add(group(0))
        elif c == "~" and magic:
            add(last_sub)
        elif c == "\r":
            out.append("\n")
        else:
            add(c)
        i += 1
    return "".join(out)


def literal_tilde(rep: str, last_sub: str, magic: bool = True) -> str:
    """The replacement with ~ replaced by the previous one (what becomes the new "last substitute string")."""
    if not magic:
        return rep
    out, i = [], 0
    while i < len(rep):
        if rep[i] == "\\" and i + 1 < len(rep):
            out.append(rep[i:i + 2])
            i += 2
            continue
        if rep[i] == "~":
            out.append(last_sub)
        else:
            out.append(rep[i])
        i += 1
    return "".join(out)
