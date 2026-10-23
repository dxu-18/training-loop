"""Shell glob patterns: matching (case, [[ == ]], ${x#pat}) and pathname expansion.

A *pattern string* uses backslash to mark a literal character, so the expander escapes every quoted
character before calling these functions (`"*".txt` arrives as `\\*.txt`).
"""

from __future__ import annotations

import re
from functools import lru_cache

CLASSES = {
    "alpha": "a-zA-Z", "digit": "0-9", "alnum": "a-zA-Z0-9", "upper": "A-Z", "lower": "a-z", "space": " \\t\\n\\r\\f\\v",
    "blank": " \\t", "punct": "!-/:-@\\[-`{-~", "xdigit": "0-9a-fA-F", "cntrl": "\\x00-\\x1f\\x7f", "print": " -~",
    "graph": "!-~", "word": "a-zA-Z0-9_",
}


def _bracket(pat: str, i: int, pathname: bool):
    """Translate a [...] expression starting at pat[i] == '['. Returns (regex, next index) or None."""
    n = len(pat)
    j = i + 1
    neg = False
    if j < n and pat[j] in "!^":
        neg = True
        j += 1
    items = []
    first = True
    while j < n:
        c = pat[j]
        if c == "]" and not first:
            body = "".join(items)
            if pathname:
                body += "/" if neg else ""
            if not body and not neg:
                return None
            return ("[^" if neg else "[") + body + "]", j + 1
        first = False
        if c == "[" and j + 1 < n and pat[j + 1] == ":":
            k = pat.find(":]", j + 2)
            if k > 0 and pat[j + 2:k] in CLASSES:
                items.append(CLASSES[pat[j + 2:k]])
                j = k + 2
                continue
        if c == "\\" and j + 1 < n:
            items.append(re.escape(pat[j + 1]))
            j += 2
            continue
        if j + 2 < n and pat[j + 1] == "-" and pat[j + 2] != "]":
            lo, hi = c, pat[j + 2]
            if hi == "\\" and j + 3 < n:
                hi = pat[j + 3]
                j += 1
            if lo <= hi:
                items.append(re.escape(lo) + "-" + re.escape(hi))
            j += 3
            continue
        items.append(re.escape(c))
        j += 1
    return None


@lru_cache(maxsize=4096)
def translate(pat: str, pathname: bool = False, extglob: bool = True) -> str:
    out = []
    i, n = 0, len(pat)
    any_ = "[^/]" if pathname else "."
    while i < n:
        c = pat[i]
        if extglob and c in "?*+@!" and i + 1 < n and pat[i + 1] == "(":
            depth, j = 1, i + 2
            while j < n and depth:
                if pat[j] == "\\":
                    j += 2
                    continue
                depth += pat[j] == "("
                depth -= pat[j] == ")"
                j += 1
            if depth == 0:
                alts = _split_alts(pat[i + 2:j - 1])
                inner = "|".join(translate(a, pathname, extglob) for a in alts)
                rx = {"?": f"(?:{inner})?", "*": f"(?:{inner})*", "+": f"(?:{inner})+", "@": f"(?:{inner})",
                      "!": f"(?:(?!(?:{inner})$){any_}*)"}[c]
                out.append(rx)
                i = j
                continue
        if c == "\\":
            if i + 1 < n:
                out.append(re.escape(pat[i + 1]))
                i += 2
            else:
                out.append("\\\\")
                i += 1
        elif c == "*":
            while i + 1 < n and pat[i + 1] == "*":
                i += 1
            out.append(f"{any_}*")
            i += 1
        elif c == "?":
            out.append(any_)
            i += 1
        elif c == "[":
            br = _bracket(pat, i, pathname)
            if br is None:
                out.append("\\[")
                i += 1
            else:
                out.append(br[0])
                i = br[1]
        else:
            out.append(re.escape(c))
            i += 1
    return "".join(out)


def _split_alts(s: str) -> list[str]:
    out, depth, cur, i = [], 0, [], 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s):
            cur.append(s[i:i + 2])
            i += 2
            continue
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        if c == "|" and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(c)
        i += 1
    out.append("".join(cur))
    return out


@lru_cache(maxsize=4096)
def compiled(pat: str, nocase: bool = False, pathname: bool = False, extglob: bool = True):
    return re.compile(translate(pat, pathname, extglob), (re.I if nocase else 0) | re.S)


def fnmatch(pat: str, s: str, nocase: bool = False, extglob: bool = True) -> bool:
    return compiled(pat, nocase, False, extglob).fullmatch(s) is not None


def has_magic(pat: str, extglob: bool = False) -> bool:
    i, n = 0, len(pat)
    while i < n:
        c = pat[i]
        if c == "\\":
            i += 2
            continue
        if c in "*?":
            return True
        if c == "[" and _bracket(pat, i, False) is not None:
            return True
        if extglob and c in "+@!" and i + 1 < n and pat[i + 1] == "(":
            return True
        i += 1
    return False


def unescape(pat: str) -> str:
    out, i, n = [], 0, len(pat)
    while i < n:
        if pat[i] == "\\" and i + 1 < n:
            out.append(pat[i + 1])
            i += 2
        else:
            out.append(pat[i])
            i += 1
    return "".join(out)


def escape(s: str) -> str:
    return "".join("\\" + c if c in "\\*?[]!+@()|" else c for c in s)


# ---------------------------------------------------------------------------- ${x#pat} ${x%pat} ${x/pat/rep}

def strip_prefix(s: str, pat: str, longest: bool, nocase: bool = False) -> str:
    rx = compiled(pat, nocase)
    rng = range(len(s), -1, -1) if longest else range(0, len(s) + 1)
    for k in rng:
        if rx.fullmatch(s, 0, k):
            return s[k:]
    return s


def strip_suffix(s: str, pat: str, longest: bool, nocase: bool = False) -> str:
    rx = compiled(pat, nocase)
    rng = range(0, len(s) + 1) if longest else range(len(s), -1, -1)
    for k in rng:
        if rx.fullmatch(s, k, len(s)):
            return s[:k]
    return s


def _longest_at(rx, s: str, start: int):
    for end in range(len(s), start - 1, -1):
        if rx.fullmatch(s, start, end):
            return end
    return None


def replace(s: str, pat: str, rep: str, mode: str, nocase: bool = False) -> str:
    """mode: '/' first, '//' all, '/#' anchored at start, '/%' anchored at end (longest matches)."""
    if pat == "":
        return s
    rx = compiled(pat, nocase)
    if mode == "/#":
        end = _longest_at(rx, s, 0)
        return s if end is None else rep + s[end:]
    if mode == "/%":
        for k in range(0, len(s) + 1):
            if rx.fullmatch(s, k, len(s)):
                return s[:k] + rep
        return s
    out, i, n = [], 0, len(s)
    while i <= n:
        end = _longest_at(rx, s, i)
        if end is not None and (end > i or True):
            out.append(rep)
            if end == i:   # empty match: keep the character
                if i < n:
                    out.append(s[i])
                i += 1
            else:
                i = end
            if mode == "/":
                out.append(s[i:])
                return "".join(out)
            continue
        if i < n:
            out.append(s[i])
        i += 1
    return "".join(out)


# ---------------------------------------------------------------------------- pathname expansion

def glob(fs, cwd: str, pat: str, dotglob: bool = False, globstar: bool = False, nocase: bool = False,
         extglob: bool = False) -> list[str]:
    """Expand a pattern string against the filesystem. Returns sorted matches ([] if none)."""
    absolute = pat.startswith("/")
    comps = pat.split("/")
    if absolute:
        comps = comps[1:]
    trailing_slash = pat.endswith("/") and len(pat) > 1
    if trailing_slash:
        comps = comps[:-1]
    start = "/" if absolute else ""
    results = _glob_comps(fs, cwd, start, comps, dotglob, globstar, nocase, extglob)
    if trailing_slash:
        out = []
        for r in results:
            try:
                if fs.lookup(r, cwd).type == "d":
                    out.append(r + "/")
            except Exception:  # noqa: BLE001
                pass
        results = out
    seen, uniq = set(), []
    for r in results:
        if r not in seen:
            seen.add(r)
            uniq.append(r)
    return sorted(uniq, key=lambda x: x.encode("utf-8", "surrogateescape"))


def _join(base: str, name: str) -> str:
    if base == "":
        return name
    if base.endswith("/"):
        return base + name
    return base + "/" + name


def _isdir(fs, cwd, path) -> bool:
    try:
        return fs.lookup(path or ".", cwd).type == "d"
    except Exception:  # noqa: BLE001
        return False


def _entries(fs, cwd, path):
    try:
        node = fs.lookup(path or ".", cwd)
    except Exception:  # noqa: BLE001
        return None
    if node.type != "d":
        return None
    return fs.listdir(node)


def _glob_comps(fs, cwd, base, comps, dotglob, globstar, nocase, extglob):
    if not comps:
        return [base] if base else []
    c, rest = comps[0], comps[1:]
    if c == "":
        return _glob_comps(fs, cwd, base, rest, dotglob, globstar, nocase, extglob)
    if globstar and c == "**":
        out = []
        # zero directories
        if rest:
            out += _glob_comps(fs, cwd, base, rest, dotglob, globstar, nocase, extglob)
        else:
            out += [base] if base else []
        for d in _walk_dirs(fs, cwd, base, dotglob):
            if rest:
                out += _glob_comps(fs, cwd, d, rest, dotglob, globstar, nocase, extglob)
            else:
                out.append(d)
        if not rest:   # "**" alone also matches files (and dir/** includes dir/ itself)
            out = [base.rstrip("/") + "/"] if base else []
            for p in _walk_all(fs, cwd, base, dotglob):
                out.append(p)
        return out
    if not has_magic(c, extglob):
        name = unescape(c)
        p = _join(base, name)
        if rest:
            if not _isdir(fs, cwd, p):
                return []
            return _glob_comps(fs, cwd, p, rest, dotglob, globstar, nocase, extglob)
        try:
            fs.lookup(p, cwd, follow=False)
            return [p]
        except Exception:  # noqa: BLE001
            return []
    names = _entries(fs, cwd, base)
    if names is None:
        return []
    rx = compiled(c, nocase, True, extglob)
    explicit_dot = c.startswith(".") or c.startswith("\\.")
    out = []
    for name in names:
        if name.startswith(".") and not (explicit_dot or dotglob):
            continue
        if rx.fullmatch(name):
            p = _join(base, name)
            if rest:
                if _isdir(fs, cwd, p):
                    out += _glob_comps(fs, cwd, p, rest, dotglob, globstar, nocase, extglob)
            else:
                out.append(p)
    return out


def _walk_dirs(fs, cwd, base, dotglob):
    names = _entries(fs, cwd, base)
    if names is None:
        return
    for name in names:
        if name.startswith(".") and not dotglob:
            continue
        p = _join(base, name)
        try:
            node = fs.lookup(p, cwd, follow=False)
        except Exception:  # noqa: BLE001
            continue
        if node.type == "d":
            yield p
            yield from _walk_dirs(fs, cwd, p, dotglob)


def _walk_all(fs, cwd, base, dotglob):
    names = _entries(fs, cwd, base)
    if names is None:
        return
    for name in names:
        if name.startswith(".") and not dotglob:
            continue
        p = _join(base, name)
        yield p
        try:
            node = fs.lookup(p, cwd, follow=False)
        except Exception:  # noqa: BLE001
            continue
        if node.type == "d":
            yield from _walk_all(fs, cwd, p, dotglob)
