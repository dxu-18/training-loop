"""Tokens: TeX's category codes, simplified.

A token is a control sequence (`cs`, the name without the backslash), a character (`ch`, letters and "other"),
a space (`sp`), a paragraph break (`par`, from a blank line), `{` / `}` (`bg` / `eg`), `$` (`math`), `&` (`tab`),
`#` (`param`), `^` / `_` (`sup` / `sub`), `~` (`active`), or raw text read verbatim (`verb`: \\verb, verbatim,
lstlisting, the URL of \\url and \\href). Every token remembers its file, line and column for error messages.

As in TeX: a comment (`%`) runs to the end of the line and swallows it with the next line's leading spaces;
spaces after a control word are skipped; a line's leading spaces are skipped; one newline is a space and a blank
line is \\par.
"""

from __future__ import annotations

VERBATIM_ENVS = ("verbatim", "verbatim*", "lstlisting", "comment", "Verbatim", "minted")


class Tok:
    __slots__ = ("k", "v", "file", "line", "col", "extra")

    def __init__(self, k, v, file="main.tex", line=1, col=0, extra=None):
        self.k, self.v, self.file, self.line, self.col, self.extra = k, v, file, line, col, extra

    def __repr__(self):
        return f"Tok({self.k},{self.v!r}@{self.line})"

    def text(self) -> str:
        """The token as it was written (for error contexts and re-reading arguments as text)."""
        k = self.k
        if k == "cs":
            return "\\" + self.v
        if k == "par":
            return "\n\n"
        if k == "sp":
            return " "
        if k == "verb":
            return self.v
        return self.v

    def same(self, k, v=None):
        return self.k == k and (v is None or self.v == v)


def is_letter(c: str) -> bool:
    return ("a" <= c <= "z") or ("A" <= c <= "Z")


def tokenize(src: str, file: str = "main.tex", at_letter: bool = False) -> list[Tok]:
    out: list[Tok] = []
    lines = src.split("\n")
    li = 0
    while li < len(lines):
        line = lines[li].rstrip("\r")
        i = 0
        n = len(line)
        # leading spaces are skipped
        while i < n and line[i] in " \t":
            i += 1
        if i == n:
            # an empty line: \par (several in a row are one paragraph break)
            if not out or out[-1].k != "par":
                if out and out[-1].k == "sp":
                    out.pop()
                out.append(Tok("par", "", file, li + 1, 0))
            li += 1
            continue
        skip_space = False
        eol_space = True
        while i < n:
            c = line[i]
            col = i
            if c == "%":
                eol_space = False
                break
            if c == "\\":
                if i + 1 >= n:
                    # a backslash at the end of a line: the control symbol \^^M (a space)
                    out.append(Tok("cs", " ", file, li + 1, col))
                    i += 1
                    eol_space = False
                    break
                d = line[i + 1]
                if is_letter(d) or (at_letter and d == "@"):
                    j = i + 1
                    while j < n and (is_letter(line[j]) or (at_letter and line[j] == "@")):
                        j += 1
                    name = line[i + 1:j]
                    i = j
                    if name == "verb" and i < n:
                        star = False
                        if line[i] == "*":
                            star = True
                            i += 1
                        if i < n and line[i] not in " \t":
                            d2 = line[i]
                            k = line.find(d2, i + 1)
                            if k < 0:
                                out.append(Tok("verb_err", line[i + 1:], file, li + 1, col))
                                i = n
                            else:
                                out.append(Tok("verb", line[i + 1:k], file, li + 1, col, "verb*" if star else "verb"))
                                i = k + 1
                            skip_space = False
                            continue
                    tok = Tok("cs", name, file, li + 1, col)
                    out.append(tok)
                    # a verbatim environment: its body is raw text up to \end{name}
                    if name == "begin":
                        m = _env_name(line, i)
                        if m and m[0] in VERBATIM_ENVS:
                            env, after = m
                            out.append(Tok("bg", "{", file, li + 1, i))
                            for ch in env:
                                out.append(Tok("ch", ch, file, li + 1, i))
                            out.append(Tok("eg", "}", file, li + 1, after - 1))
                            rest = line[after:]
                            opt = ""
                            if rest.startswith("["):
                                k = _match_bracket(rest)
                                if k > 0:
                                    opt = rest[1:k]
                                    rest = rest[k + 1:]
                            body, li2, tail = _read_until_end(lines, li, rest, env)
                            out.append(Tok("verb", body, file, li + 2, 0, ("env", env, opt, li2 + 1)))
                            out.append(Tok("cs", "end", file, li2 + 1, 0))
                            out.append(Tok("bg", "{", file, li2 + 1, 0))
                            for ch in env:
                                out.append(Tok("ch", ch, file, li2 + 1, 0))
                            out.append(Tok("eg", "}", file, li2 + 1, 0))
                            li = li2
                            line = lines[li].rstrip("\r")
                            n = len(line)
                            i = n - len(tail)
                            skip_space = False
                            continue
                    if name in ("url", "href") or name == "path":
                        # the URL is read as raw text
                        k = i
                        while k < n and line[k] in " \t":
                            k += 1
                        if k < n and line[k] == "{":
                            depth, e = 0, k
                            while e < n:
                                if line[e] == "{":
                                    depth += 1
                                elif line[e] == "}":
                                    depth -= 1
                                    if depth == 0:
                                        break
                                e += 1
                            if e < n:
                                out.append(Tok("verb", line[k + 1:e], file, li + 1, k, "url"))
                                i = e + 1
                                skip_space = False
                                continue
                    skip_space = True
                    continue
                # a control symbol
                out.append(Tok("cs", d, file, li + 1, col))
                i += 2
                skip_space = d == " "
                continue
            if c in " \t":
                if not skip_space and out and out[-1].k not in ("sp", "par"):
                    out.append(Tok("sp", " ", file, li + 1, col))
                skip_space = True
                i += 1
                continue
            skip_space = False
            k = {"{": "bg", "}": "eg", "$": "math", "&": "tab", "#": "param", "^": "sup", "_": "sub",
                 "~": "active"}.get(c, "ch")
            out.append(Tok(k, c, file, li + 1, col))
            i += 1
        if eol_space and out and out[-1].k not in ("sp", "par") and not (out[-1].k == "cs" and is_letter(out[-1].v[:1])):
            out.append(Tok("sp", " ", file, li + 1, n))
        li += 1
    while out and out[-1].k in ("sp", "par"):
        out.pop()
    return out


def _env_name(line: str, i: int):
    j = i
    while j < len(line) and line[j] in " \t":
        j += 1
    if j < len(line) and line[j] == "{":
        k = line.find("}", j)
        if k > 0:
            return line[j + 1:k], k + 1
    return None


def _match_bracket(s: str) -> int:
    depth = 0
    for i, c in enumerate(s):
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return i
        elif c == "{":
            # skip groups inside the options
            pass
    return -1


def _read_until_end(lines, li, rest, env):
    """Raw lines from `rest` (what follows \\begin{env} on line li) up to \\end{env}.
    -> (body, index of the line holding \\end{env}, what follows \\end{env} on it)."""
    endtag = "\\end{" + env + "}"
    k = rest.find(endtag)
    if k >= 0:
        return rest[:k], li, rest[k + len(endtag):]
    body = [rest] if rest.strip() else []
    j = li + 1
    while j < len(lines):
        ln = lines[j].rstrip("\r")
        k = ln.find(endtag)
        if k >= 0:
            if ln[:k].strip():
                body.append(ln[:k])
            return "\n".join(body) + ("\n" if body else ""), j, ln[k + len(endtag):]
        body.append(ln)
        j += 1
    # no \end: the rest of the file (the engine reports it)
    return "\n".join(body), len(lines) - 1, ""


def detokenize(toks: list[Tok]) -> str:
    """Tokens back to source-like text (arguments read as text: labels, options, file names, colors)."""
    out = []
    for i, t in enumerate(toks):
        s = t.text()
        if t.k == "cs" and t.v and is_letter(t.v[0]) and i + 1 < len(toks) and toks[i + 1].k == "ch" and \
                is_letter(toks[i + 1].v):
            s += " "
        out.append(s)
    return "".join(out)
