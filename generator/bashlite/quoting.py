"""Bash's quoting styles: xtrace / `declare -p` / `printf %q` / `${x@Q}`."""

from __future__ import annotations

METAS = set(" \t\n'\"\\|&;()<>!{}*[?]^$`")


def contains_metas(s: str) -> bool:
    for i, c in enumerate(s):
        if c in METAS:
            return True
        if c == "~" and (i == 0 or s[i - 1] in "=:"):
            return True
        if c == "#" and i == 0:
            return True
    return False


def needs_ansic(s: str) -> bool:
    return any(ord(c) < 32 or ord(c) == 127 for c in s)


def ansic_quote(s: str) -> str:
    out = ["$'"]
    for c in s:
        o = ord(c)
        if c == "'":
            out.append("\\'")
        elif c == "\\":
            out.append("\\\\")
        elif c == "\n":
            out.append("\\n")
        elif c == "\t":
            out.append("\\t")
        elif c == "\r":
            out.append("\\r")
        elif c == "\x1b":
            out.append("\\E")
        elif c == "\a":
            out.append("\\a")
        elif c == "\b":
            out.append("\\b")
        elif c == "\f":
            out.append("\\f")
        elif c == "\v":
            out.append("\\v")
        elif o < 32 or o == 127:
            out.append("\\%03o" % o)
        else:
            out.append(c)
    out.append("'")
    return "".join(out)


def single_quote(s: str) -> str:
    if s == "'":
        return "\\'"
    return "'" + s.replace("'", "'\\''") + "'"


def xtrace_word(s: str) -> str:
    if s == "":
        return "''"
    if contains_metas(s):
        return single_quote(s)
    if needs_ansic(s):
        return ansic_quote(s)
    return s


def double_quote(s: str) -> str:
    """declare -p style: "value" with \\ before " $ ` \\."""
    out = ['"']
    for c in s:
        if c in '"$`\\':
            out.append("\\")
        out.append(c)
    out.append('"')
    return "".join(out)


def declare_value(s: str) -> str:
    if needs_ansic(s):
        return ansic_quote(s)
    return double_quote(s)


BACKSLASH_SPECIAL = set(" \t\n'\"\\|&;()<>!{}*[?]^$`,#=~")


def backslash_quote(s: str) -> str:
    """printf %q style."""
    if s == "":
        return "''"
    if needs_ansic(s):
        return ansic_quote(s)
    out = []
    for i, c in enumerate(s):
        if c in BACKSLASH_SPECIAL and not (c in "#=~" and i > 0 and not (c == "~" and s[i - 1] in "=:")):
            if c in "=," and True:
                out.append(c)
                continue
            out.append("\\" + c)
        else:
            out.append(c)
    return "".join(out)


def at_q(s: str) -> str:
    """${x@Q}"""
    if needs_ansic(s):
        return ansic_quote(s)
    return "'" + s.replace("'", "'\\''") + "'"
