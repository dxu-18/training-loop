"""A recursive-descent parser for bash.

Reads one complete command at a time (`parse_next`), like bash does, so aliases defined by one command
apply to the next and an error later in a script doesn't stop earlier commands. `Incomplete` means the
input ended inside a construct (open quote, `if` without `fi`, pending here-document, trailing `|`);
the interactive reader then feeds the next line and parses again.
"""

from __future__ import annotations

import bisect
import re

from .syntax import (AndOr, ArithCmd, ArithFor, ArithSub, Assign, Brace, Case, CmdList, CmdSub, Cond, CondAnd,
                     CondBinary, CondNot, CondOr, CondUnary, CondWord, DQ, Esc, For, FuncDef, HereDoc, If, Lit, Param,
                     Pipeline, ProcSub, Redir, Select, Simple, SQ, Subshell, While, Word)

RESERVED = {"if", "then", "elif", "else", "fi", "do", "done", "case", "esac", "while", "until", "for", "in", "select",
            "function", "time", "{", "}", "!", "[[", "]]", "coproc"}
OPS3 = (";;&", "<<<", "<<-", "&>>")
OPS2 = (";;", ";&", "&&", "||", "|&", "<<", ">>", "<&", ">&", "<>", ">|", "&>")
META = set("|&;()<> \t\n")
REDIR_OPS = {"<", ">", ">>", "<<", "<<-", "<<<", "<&", ">&", "<>", ">|", "&>", "&>>"}
ASSIGN_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)(\[.*\])?(\+?)=", re.S)
ARRAY_PREFIX = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\[[^\]]*\])?\+?=$")
NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
DECLARATION = {"declare", "typeset", "local", "export", "readonly"}
COND_UNARY = {"-a", "-b", "-c", "-d", "-e", "-f", "-g", "-h", "-k", "-p", "-r", "-s", "-t", "-u", "-w", "-x", "-G",
              "-L", "-N", "-O", "-S", "-z", "-n", "-o", "-v", "-R"}
COND_BINARY = {"==", "=", "!=", "=~", "<", ">", "-eq", "-ne", "-lt", "-le", "-gt", "-ge", "-ef", "-nt", "-ot"}
ANSI = {"a": "\a", "b": "\b", "e": "\x1b", "E": "\x1b", "f": "\f", "n": "\n", "r": "\r", "t": "\t", "v": "\v",
        "\\": "\\", "'": "'", '"': '"', "?": "?"}


class ParseError(Exception):
    def __init__(self, msg: str, line: int = 0, token: str = ""):
        super().__init__(msg)
        self.msg, self.line, self.token = msg, line, token


class Incomplete(ParseError):
    """Input ended inside a construct: more lines are needed."""


class Tok:
    __slots__ = ("kind", "val", "word", "start", "end", "io", "line")

    def __init__(self, kind, val="", word=None, start=0, end=0, io=None, line=0):
        self.kind, self.val, self.word, self.start, self.end, self.io, self.line = kind, val, word, start, end, io, line

    def __repr__(self):
        return f"Tok({self.kind},{self.val!r})"

    @property
    def text(self) -> str:
        if self.kind == "nl":
            return "newline"
        if self.kind == "eof":
            return "EOF"
        return self.val


def decode_ansi(s: str) -> str:
    """Decode the body of $'...'."""
    out, i, n = [], 0, len(s)
    while i < n:
        c = s[i]
        if c != "\\" or i + 1 >= n:
            out.append(c)
            i += 1
            continue
        d = s[i + 1]
        i += 2
        if d in ANSI:
            out.append(ANSI[d])
        elif d in "01234567":
            j = i - 1
            while j < n and j < i + 2 and s[j] in "01234567":
                j += 1
            out.append(chr(int(s[i - 1:j], 8) & 0xFF))
            i = j
        elif d == "x":
            j = i
            while j < n and j < i + 2 and s[j] in "0123456789abcdefABCDEF":
                j += 1
            out.append(chr(int(s[i:j], 16)) if j > i else "\\x")
            i = j
        elif d in "uU":
            lim = 4 if d == "u" else 8
            j = i
            while j < n and j < i + lim and s[j] in "0123456789abcdefABCDEF":
                j += 1
            out.append(chr(int(s[i:j], 16)) if j > i else "\\" + d)
            i = j
        elif d == "c" and i < n:
            out.append(chr(ord(s[i]) & 0x1F))
            i += 1
        else:
            out.append("\\" + d)
    return "".join(out)


class Parser:
    def __init__(self, src: str = "", aliases: dict | None = None, interactive: bool = False, line: int = 1):
        self.s = src
        self.n = len(src)
        self.i = 0
        self.base_line = line
        self.aliases = aliases          # None = no alias expansion
        self.interactive = interactive  # EOF inside a construct -> Incomplete
        self.heredocs: list[HereDoc] = []
        self._tok: Tok | None = None
        self._alias_end: list[tuple[str, int]] = []
        self._alias_next = False        # previous alias value ended in a blank
        self.warnings: list[str] = []
        self._index_lines()

    # ------------------------------------------------------------ input management

    def _index_lines(self):
        self._nl = [m.start() for m in re.finditer("\n", self.s)]

    def feed(self, text: str) -> None:
        self.s += text
        self.n = len(self.s)
        self._index_lines()

    def line_of(self, pos: int) -> int:
        return self.base_line + bisect.bisect_left(self._nl, pos)

    def mark(self):
        return (self.i, self._tok, list(self.heredocs), list(self._alias_end), self.s)

    def reset(self, m) -> None:
        self.i, self._tok, self.heredocs, self._alias_end, s = m
        if s != self.s:
            self.s, self.n = s, len(s)
            self._index_lines()

    def at_end(self) -> bool:
        return self._peek_kind() == "eof"

    def _peek_kind(self):
        try:
            return self.peek().kind
        except Incomplete:
            return "incomplete"

    def _eof(self, what: str = ""):
        if self.interactive:
            raise Incomplete("unexpected end of file", self.line_of(self.i))
        if what:
            raise ParseError(f"unexpected EOF while looking for matching `{what}'", self.line_of(self.i))
        raise ParseError("syntax error: unexpected end of file", self.line_of(self.i))

    def _unexpected(self, tok: Tok):
        if tok.kind == "eof":
            self._eof()
        raise ParseError(f"syntax error near unexpected token `{tok.text}'", tok.line or self.line_of(self.i), tok.text)

    # ------------------------------------------------------------ tokens

    def peek(self) -> Tok:
        if self._tok is None:
            self._tok = self._next_token()
        return self._tok

    def advance(self) -> Tok:
        t = self.peek()
        self._tok = None
        return t

    def _skip_blanks(self):
        s, n = self.s, self.n
        while self.i < n:
            c = s[self.i]
            if c in " \t":
                self.i += 1
            elif c == "\\" and self.i + 1 < n and s[self.i + 1] == "\n":
                self.i += 2
            elif c == "\\" and self.i + 1 == n and self.interactive:
                raise Incomplete("continuation", self.line_of(self.i))
            elif c == "#":
                j = s.find("\n", self.i)
                self.i = n if j < 0 else j
            else:
                break

    def _next_token(self) -> Tok:
        self._skip_blanks()
        s, n = self.s, self.n
        while self._alias_end and self.i >= self._alias_end[-1][1]:
            self._alias_end.pop()
        start = self.i
        if self.i >= n:
            return Tok("eof", start=start, end=start, line=self.line_of(start))
        c = s[self.i]
        line = self.line_of(start)
        if c == "\n":
            self.i += 1
            tok = Tok("nl", "\n", start=start, end=self.i, line=line)
            if self.heredocs:
                self._read_heredocs()
            return tok
        io = None
        if c.isdigit():
            j = self.i
            while j < n and s[j].isdigit():
                j += 1
            if j < n and s[j] in "<>" and not (j + 1 < n and s[j + 1] == "("):
                io = int(s[self.i:j])
                self.i = j
                c = s[j]
        if c in "|&;()<>":
            if c in "<>" and self.i + 1 < n and s[self.i + 1] == "(" and io is None:
                return self._word_token(start, line)
            for op in OPS3 + OPS2:
                if s.startswith(op, self.i):
                    self.i += len(op)
                    return Tok("op", op, start=start, end=self.i, io=io, line=line)
            self.i += 1
            return Tok("op", c, start=start, end=self.i, io=io, line=line)
        return self._word_token(start, line)

    def _word_token(self, start, line) -> Tok:
        w = self._word()
        return Tok("word", w.raw, w, start, self.i, line=line)

    # ------------------------------------------------------------ words

    def _word(self, regex: bool = False) -> Word:
        s, n = self.s, self.n
        start = self.i
        parts: list = []
        lit: list[str] = []
        depth = 0

        def flush():
            if lit:
                parts.append(Lit("".join(lit)))
                lit.clear()

        while self.i < n:
            c = s[self.i]
            if c in META:
                if c in "<>" and self.i + 1 < n and s[self.i + 1] == "(" and not parts and not lit and not regex:
                    parts.append(self._procsub())
                    continue
                if c == "(" and not regex and not parts and ARRAY_PREFIX.match("".join(lit)):
                    flush()
                    w = Word(parts, s[start:self.i])
                    w.array = self._array_literal()
                    w.raw = s[start:self.i]
                    return w
                if regex and c in "()|<>" and not (c == ")" and depth == 0):
                    depth += c == "("
                    depth -= c == ")"
                    lit.append(c)
                    self.i += 1
                    continue
                if regex and c in " \t" and depth > 0:
                    lit.append(c)
                    self.i += 1
                    continue
                break
            if c == "\\":
                if self.i + 1 >= n:
                    if self.interactive:
                        raise Incomplete("continuation", self.line_of(self.i))
                    lit.append("\\")
                    self.i += 1
                    continue
                if s[self.i + 1] == "\n":
                    self.i += 2
                    continue
                flush()
                parts.append(Esc(s[self.i + 1]))
                self.i += 2
            elif c == "'":
                flush()
                j = s.find("'", self.i + 1)
                if j < 0:
                    self.i = n
                    self._eof("'")
                parts.append(SQ(s[self.i + 1:j]))
                self.i = j + 1
            elif c == '"':
                flush()
                self.i += 1
                parts.append(DQ(self._dq_parts()))
            elif c == "$":
                flush()
                parts.append(self._dollar(False))
            elif c == "`":
                flush()
                parts.append(self._backtick(False))
            else:
                lit.append(c)
                self.i += 1
        flush()
        return Word(parts, s[start:self.i])

    def _dq_parts(self, heredoc: bool = False) -> list:
        """Parts of a double-quoted string; self.i is just after the opening quote (heredoc: no quotes)."""
        s, n = self.s, self.n
        parts: list = []
        lit: list[str] = []

        def flush():
            if lit:
                parts.append(Lit("".join(lit)))
                lit.clear()

        while True:
            if self.i >= n:
                if heredoc:
                    break
                self._eof('"')
            c = s[self.i]
            if c == '"' and not heredoc:
                self.i += 1
                break
            if c == "\\" and self.i + 1 < n:
                d = s[self.i + 1]
                if d == "\n":
                    self.i += 2
                    continue
                if d in "$`\\" or (d == '"' and not heredoc):
                    flush()
                    parts.append(Esc(d))
                    self.i += 2
                    continue
                lit.append("\\")
                self.i += 1
                continue
            if c == "$":
                flush()
                parts.append(self._dollar(True))
            elif c == "`":
                flush()
                parts.append(self._backtick(True))
            else:
                lit.append(c)
                self.i += 1
        flush()
        return parts

    def _dollar(self, in_dq: bool):
        s, n, i = self.s, self.n, self.i
        nxt = s[i + 1] if i + 1 < n else ""
        if nxt == "'" and not in_dq:
            j, k = i + 2, i + 2
            while True:
                if k >= n:
                    self.i = n
                    self._eof("'")
                if s[k] == "\\" and k + 1 < n:
                    k += 2
                    continue
                if s[k] == "'":
                    break
                k += 1
            self.i = k + 1
            return SQ(decode_ansi(s[j:k]), ansi=True)
        if nxt == '"' and not in_dq:
            self.i = i + 2
            return DQ(self._dq_parts())
        if nxt == "(":
            if i + 2 < n and s[i + 2] == "(":
                end = self._match_arith(i + 3)
                if end is not None:
                    expr = s[i + 3:end]
                    self.i = end + 2
                    return ArithSub(self._sub_word(expr, "arith"), expr)
            self.i = i + 2
            body, src = self._nested_list(")")
            return CmdSub(body, src)
        if nxt == "{":
            return self._brace_param()
        if nxt == "[":   # deprecated $[ expr ]
            j = s.find("]", i + 2)
            if j > 0:
                self.i = j + 1
                return ArithSub(self._sub_word(s[i + 2:j], "arith"), s[i + 2:j])
        m = NAME_RE.match(s, i + 1)
        if m:
            self.i = m.end()
            return Param(m.group())
        if nxt and (nxt.isdigit() or nxt in "@*#?-$!"):
            self.i = i + 2
            return Param(nxt)
        self.i = i + 1
        return Lit("$")

    def _match_arith(self, j: int) -> int | None:
        """Index of the `))` closing an arithmetic expansion starting at j, or None if it's not one."""
        s, n = self.s, self.n
        depth = 0
        while j < n:
            c = s[j]
            if c == "(":
                depth += 1
            elif c == ")":
                if depth == 0:
                    return j if j + 1 < n and s[j + 1] == ")" else None
                depth -= 1
            elif c == "\\":
                j += 1
            elif c in "'\"":
                k = s.find(c, j + 1)
                if k < 0:
                    break
                j = k
            elif c == "$" and j + 1 < n and s[j + 1] == "(" and not (j + 2 < n and s[j + 2] == "("):
                depth += 1
                j += 1
            j += 1
        if self.interactive:
            raise Incomplete("arith", self.line_of(j))
        return None

    def _sub_word(self, text: str, mode: str) -> Word:
        """Parse `text` (an arithmetic expression or heredoc body) into a Word."""
        p = Parser(text, None, False)
        if mode == "heredoc":
            parts = p._dq_parts(heredoc=True)
        else:
            parts = p._loose_parts()
        return Word(parts, text)

    def _loose_parts(self) -> list:
        """Everything to the end of input as literal text, with quotes and $-expansions recognised."""
        s, n = self.s, self.n
        parts, lit = [], []

        def flush():
            if lit:
                parts.append(Lit("".join(lit)))
                lit.clear()
        while self.i < n:
            c = s[self.i]
            if c == "$":
                flush()
                parts.append(self._dollar(False))
            elif c == "`":
                flush()
                parts.append(self._backtick(False))
            elif c == '"':
                flush()
                self.i += 1
                parts.append(DQ(self._dq_parts()))
            elif c == "'":
                flush()
                j = s.find("'", self.i + 1)
                j = n if j < 0 else j
                parts.append(SQ(s[self.i + 1:j]))
                self.i = j + 1
            elif c == "\\" and self.i + 1 < n:
                flush()
                if s[self.i + 1] != "\n":
                    parts.append(Esc(s[self.i + 1]))
                self.i += 2
            else:
                lit.append(c)
                self.i += 1
        flush()
        return parts

    def _nested_list(self, close: str):
        """Parse a command list up to `close` (")") for $( ), <( ) and >( ). self.i is after the "("."""
        saved_tok = self._tok
        self._tok = None
        start = self.i
        body = self.parse_compound_list({close}, allow_empty=True)
        t = self.peek()
        if t.kind != "op" or t.val != close:
            self._unexpected(t)
        src = self.s[start:t.start]
        self._tok = None
        self.i = t.end
        self._tok = saved_tok
        return body, src

    def _procsub(self):
        d = self.s[self.i]
        self.i += 2
        body, src = self._nested_list(")")
        return ProcSub(d, body, src)

    def _backtick(self, in_dq: bool):
        s, n = self.s, self.n
        j = self.i + 1
        out = []
        while True:
            if j >= n:
                self.i = n
                self._eof("`")
            c = s[j]
            if c == "\\" and j + 1 < n and (s[j + 1] in "$`\\" or (in_dq and s[j + 1] == '"')):
                out.append(s[j + 1])
                j += 2
                continue
            if c == "`":
                break
            out.append(c)
            j += 1
        src = "".join(out)
        self.i = j + 1
        p = Parser(src, self.aliases, False, self.line_of(self.i))
        body = p.parse_all()
        return CmdSub(body, src, backtick=True)

    def _brace_param(self):
        s, n = self.s, self.n
        start = self.i
        self.i += 2

        def need(k=0):
            if self.i + k >= n:
                self.i = n
                self._eof("}")
            return s[self.i + k]
        c = need()
        # ${#name} / ${#arr[@]} : length (but ${#} is $#, and ${#-...} etc. are odd)
        if c == "#" and need(1) != "}" and (NAME_RE.match(s, self.i + 1) or s[self.i + 1].isdigit()
                                            or (s[self.i + 1] in "@*#?$!-" and need(2) in "}[")):
            self.i += 1
            name = self._param_name()
            sub = self._subscript() if need() == "[" else None
            if need() != "}":
                raise ParseError(f"${{{s[start + 2:self.i + 1]}: bad substitution", self.line_of(start))
            self.i += 1
            return Param(name, sub, "len", braced=True)
        indirect = False
        if c == "!" and need(1) != "}":
            indirect = True
            self.i += 1
        name = self._param_name()
        if not name:
            j = s.find("}", self.i)
            bad = s[start:j + 1] if j > 0 else s[start:]
            raise ParseError(f"{bad}: bad substitution", self.line_of(start))
        sub = self._subscript() if need() == "[" else None
        c = need()
        if indirect and c in "*@" and need(1) == "}" and sub is None:
            self.i += 2
            return Param(name, None, "!" + c, braced=True)
        if c == "}":
            self.i += 1
            return Param(name, sub, "", indirect=indirect, braced=True)
        two = s[self.i:self.i + 2]
        if two in (":-", ":=", ":?", ":+"):
            self.i += 2
            return Param(name, sub, two, self._param_arg(), indirect=indirect, braced=True)
        if c in "-=?+":
            self.i += 1
            return Param(name, sub, c, self._param_arg(), indirect=indirect, braced=True)
        if c == ":":
            self.i += 1
            off = self._param_arg(stop=":")
            length = None
            if need() == ":":
                self.i += 1
                length = self._param_arg()
            else:
                self.i += 1
            return Param(name, sub, ":", off, length, indirect=indirect, braced=True)
        if c in "#%":
            op = c * 2 if need(1) == c else c
            self.i += len(op)
            return Param(name, sub, op, self._param_arg(pattern=True), indirect=indirect, braced=True)
        if c == "/":
            nxt = need(1)
            op = "/" + nxt if nxt in "/#%" else "/"
            self.i += len(op)
            pat = self._param_arg(stop="/", pattern=True)
            rep = None
            if need() == "/":
                self.i += 1
                rep = self._param_arg()
            else:
                self.i += 1
            return Param(name, sub, op, pat, rep, indirect=indirect, braced=True)
        if c in "^,":
            op = c * 2 if need(1) == c else c
            self.i += len(op)
            return Param(name, sub, op, self._param_arg(pattern=True), indirect=indirect, braced=True)
        if c == "@":
            self.i += 1
            arg = self._param_arg()
            return Param(name, sub, "@", arg, indirect=indirect, braced=True)
        j = s.find("}", self.i)
        raise ParseError(f"{s[start:j + 1] if j > 0 else s[start:]}: bad substitution", self.line_of(start))

    def _param_name(self) -> str:
        s, n = self.s, self.n
        m = NAME_RE.match(s, self.i)
        if m:
            self.i = m.end()
            return m.group()
        j = self.i
        while j < n and s[j].isdigit():
            j += 1
        if j > self.i:
            name = s[self.i:j]
            self.i = j
            return name
        if self.i < n and s[self.i] in "@*#?-$!":
            self.i += 1
            return s[self.i - 1]
        return ""

    def _subscript(self) -> Word:
        s, n = self.s, self.n
        depth = 0
        j = self.i
        while j < n:
            c = s[j]
            if c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    break
            elif c == "\\":
                j += 1
            elif c in "'\"":
                k = s.find(c, j + 1)
                j = k if k > 0 else n
            j += 1
        if j >= n:
            self.i = n
            self._eof("]")
        text = s[self.i + 1:j]
        self.i = j + 1
        return self._sub_word(text, "arith")

    def _param_arg(self, stop: str = "", pattern: bool = False) -> Word:
        """The word after a ${name OP ...} operator, up to `}` (or `stop`)."""
        s, n = self.s, self.n
        start = self.i
        parts, lit = [], []
        depth = 0

        def flush():
            if lit:
                parts.append(Lit("".join(lit)))
                lit.clear()
        while True:
            if self.i >= n:
                self._eof("}")
            c = s[self.i]
            if c == "}" and depth == 0:
                break
            if stop and c == stop and depth == 0:
                break
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            if c == "\\" and self.i + 1 < n:
                if s[self.i + 1] == "\n":
                    self.i += 2
                    continue
                flush()
                parts.append(Esc(s[self.i + 1]))
                self.i += 2
            elif c == "'":
                flush()
                j = s.find("'", self.i + 1)
                if j < 0:
                    self.i = n
                    self._eof("'")
                parts.append(SQ(s[self.i + 1:j]))
                self.i = j + 1
            elif c == '"':
                flush()
                self.i += 1
                parts.append(DQ(self._dq_parts()))
            elif c == "$":
                flush()
                parts.append(self._dollar(False))
            elif c == "`":
                flush()
                parts.append(self._backtick(False))
            elif pattern and c == "[" and stop == "/":   # bracket expressions may hold a /
                k = s.find("]", self.i + 2)
                if k > 0 and "}" not in s[self.i:k]:
                    lit.append(s[self.i:k + 1])
                    self.i = k + 1
                else:
                    lit.append(c)
                    self.i += 1
            else:
                lit.append(c)
                self.i += 1
        flush()
        return Word(parts, s[start:self.i])

    def _array_literal(self) -> list[Word]:
        """name=( a b [k]=v ... ); self.i is at "("."""
        self.i += 1
        items = []
        while True:
            # skip blanks, newlines and comments
            while self.i < self.n and self.s[self.i] in " \t\n":
                self.i += 1
            if self.i < self.n and self.s[self.i] == "#":
                j = self.s.find("\n", self.i)
                self.i = self.n if j < 0 else j
                continue
            if self.i >= self.n:
                self._eof(")")
            if self.s[self.i] == ")":
                self.i += 1
                return items
            if self.s[self.i] in "|&;<>(":
                raise ParseError(f"syntax error near unexpected token `{self.s[self.i]}'", self.line_of(self.i))
            w = self._word()
            if not w.parts:
                self.i += 1
                continue
            items.append(w)

    # ------------------------------------------------------------ here-documents

    def _read_heredocs(self):
        s, n = self.s, self.n
        docs, self.heredocs = self.heredocs, []
        for hd in docs:
            lines = []
            start_line = self.line_of(self.i)
            while True:
                if self.i >= n:
                    if self.interactive:
                        self.heredocs = docs[docs.index(hd):]
                        raise Incomplete("here-document", start_line)
                    self.warnings.append(f"warning: here-document at line {start_line - 1} delimited by end-of-file "
                                         f"(wanted `{hd.delim}')")
                    break
                j = s.find("\n", self.i)
                if j < 0:
                    if self.interactive:
                        self.heredocs = docs[docs.index(hd):]
                        raise Incomplete("here-document", start_line)
                    text, self.i = s[self.i:], n
                else:
                    text, self.i = s[self.i:j], j + 1
                check = text.lstrip("\t") if hd.strip_tabs else text
                if check == hd.delim:
                    break
                lines.append(check)
            hd.text = "".join(ln + "\n" for ln in lines)
            if not hd.quoted:
                hd.body = self._sub_word(hd.text, "heredoc")

    # ------------------------------------------------------------ grammar

    def parse_all(self) -> CmdList:
        """Every command in the input as one list (used for backtick bodies and `eval`-style strings)."""
        items = []
        while True:
            lst = self.parse_next()
            if lst is None:
                break
            items.extend(lst.items)
        return CmdList(items)

    def parse_next(self) -> CmdList | None:
        """The next complete command (ends at a newline), or None at end of input."""
        while self.peek().kind == "nl":
            self.advance()
        if self.peek().kind == "eof":
            if self.heredocs:
                self._read_heredocs()
            return None
        lst = self.parse_list()
        t = self.peek()
        if t.kind == "nl":
            self.advance()
        elif t.kind != "eof":
            self._unexpected(t)
        elif self.heredocs:
            self._read_heredocs()
        return lst

    def parse_list(self) -> CmdList:
        items = []
        while True:
            ao = self.parse_and_or()
            t = self.peek()
            if t.kind == "op" and t.val in (";", "&"):
                self.advance()
                items.append((ao, t.val == "&"))
                nt = self.peek()
                if nt.kind in ("nl", "eof") or (nt.kind == "op" and nt.val == ")"):
                    break
                continue
            items.append((ao, False))
            break
        return CmdList(items)

    def _is_kw(self, t: Tok, words) -> bool:
        return t.kind == "word" and t.val in words and not t.word.is_quoted()

    def parse_compound_list(self, terms: set, allow_empty: bool = False) -> CmdList:
        items = []
        while True:
            while self.peek().kind == "nl":
                self.advance()
            t = self.peek()
            if t.kind == "eof":
                if items or allow_empty and not terms - {")"}:
                    pass
                self._eof()
            if (t.kind == "op" and t.val in terms) or self._is_kw(t, terms):
                break
            ao = self.parse_and_or()
            sep = self.peek()
            if sep.kind == "op" and sep.val in (";", "&"):
                self.advance()
                items.append((ao, sep.val == "&"))
                continue
            if sep.kind == "nl":
                self.advance()
                items.append((ao, False))
                continue
            items.append((ao, False))
            t = self.peek()
            if not ((t.kind == "op" and t.val in terms) or self._is_kw(t, terms)):
                self._unexpected(t)
            break
        if not items and not allow_empty:
            self._unexpected(self.peek())
        return CmdList(items)

    def parse_and_or(self) -> AndOr:
        start = self.peek().start
        first = self.parse_pipeline()
        rest = []
        while True:
            t = self.peek()
            if t.kind == "op" and t.val in ("&&", "||"):
                self.advance()
                while self.peek().kind == "nl":
                    self.advance()
                rest.append((t.val, self.parse_pipeline()))
            else:
                break
        end = self.peek().start if self._tok is not None else self.i
        return AndOr(first, rest, self.s[start:end].strip())

    def parse_pipeline(self) -> Pipeline:
        t = self.peek()
        start, line = t.start, t.line
        timed = posix = negate = False
        if self._is_kw(t, {"time"}):
            self.advance()
            timed = True
            t = self.peek()
            if t.kind == "word" and t.val == "-p":
                self.advance()
                posix = True
            t = self.peek()
        while self._is_kw(t, {"!"}):
            self.advance()
            negate = not negate
            t = self.peek()
        if timed and (t.kind in ("nl", "eof") or (t.kind == "op" and t.val in (";", "&", ")"))):
            return Pipeline([], negate, timed, posix, src=self.s[start:t.start].strip(), line=line)
        cmds = [self.parse_command()]
        pipe_err = []
        while True:
            t = self.peek()
            if t.kind == "op" and t.val in ("|", "|&"):
                self.advance()
                pipe_err.append(t.val == "|&")
                while self.peek().kind == "nl":
                    self.advance()
                cmds.append(self.parse_command())
            else:
                break
        end = self.peek().start if self._tok is not None else self.i
        return Pipeline(cmds, negate, timed, posix, pipe_err, self.s[start:end].strip(), line)

    def _maybe_alias(self, t: Tok) -> bool:
        """Expand an alias in command position. Returns True if the input changed."""
        if self.aliases is None or t.kind != "word" or t.word.is_quoted():
            return False
        name = t.val
        if name not in self.aliases or any(a == name for a, _ in self._alias_end):
            return False
        value = self.aliases[name]
        self.s = self.s[:t.start] + value + self.s[t.end:]
        self.n = len(self.s)
        delta = len(value) - (t.end - t.start)
        self._alias_end = [(a, e + delta) for a, e in self._alias_end]
        self._alias_end.append((name, t.start + len(value)))
        self._index_lines()
        self.i = t.start
        self._tok = None
        self._alias_next = value.endswith((" ", "\t"))
        return True

    def parse_command(self):
        t = self.peek()
        for _ in range(64):
            if not self._maybe_alias(t):
                break
            t = self.peek()
        if t.kind == "word" and not t.word.is_quoted():
            v = t.val
            if v == "{":
                return self._redirs_after(self._brace())
            if v == "if":
                return self._redirs_after(self._if())
            if v in ("while", "until"):
                return self._redirs_after(self._while(v == "until"))
            if v == "for":
                return self._redirs_after(self._for())
            if v == "select":
                return self._redirs_after(self._select())
            if v == "case":
                return self._redirs_after(self._case())
            if v == "[[":
                return self._redirs_after(self._cond())
            if v == "function":
                return self._function_kw()
            if v in ("then", "fi", "do", "done", "esac", "}", "elif", "else", "in", "]]") or (v == "!" and False):
                self._unexpected(t)
        if t.kind == "op" and t.val == "(":
            if self.i < self.n and self.s[self.i] == "(":
                end = self._match_arith(self.i + 1)
                if end is not None:
                    expr = self.s[self.i + 1:end]
                    self._tok = None
                    self.i = end + 2
                    return self._redirs_after(ArithCmd(self._sub_word(expr, "arith"), line=t.line, src=expr))
            self.advance()
            body = self.parse_compound_list({")"})
            close = self.advance()
            if close.kind != "op" or close.val != ")":
                self._unexpected(close)
            return self._redirs_after(Subshell(body, line=t.line))
        return self._simple()

    def _redirs_after(self, node):
        while True:
            t = self.peek()
            if t.kind == "op" and t.val in REDIR_OPS:
                node.redirs.append(self._redirect())
            else:
                return node

    def _redirect(self) -> Redir:
        t = self.advance()
        op = t.val
        tgt = self.peek()
        if tgt.kind != "word":
            self._unexpected(tgt)
        self.advance()
        if op in ("<<", "<<-"):
            delim = tgt.word.literal()
            quoted = tgt.word.is_quoted()
            if delim is None:   # $vars in a delimiter are taken literally
                delim = re.sub(r"['\"\\]", "", tgt.val)
            hd = HereDoc(delim, op == "<<-", quoted)
            self.heredocs.append(hd)
            return Redir(t.io, op, tgt.word, hd)
        return Redir(t.io, op, tgt.word)

    def _simple(self) -> Simple:
        assigns, words, redirs = [], [], []
        first = self.peek()
        start, line = first.start, first.line
        while True:
            t = self.peek()
            if t.kind == "word":
                if not words and t.word.array is None and ASSIGN_RE.match(t.val) and self._assign_ok(t.word):
                    self.advance()
                    assigns.append(self._make_assign(t))
                    continue
                if not words and t.word.array is not None:
                    self.advance()
                    assigns.append(self._make_assign(t))
                    continue
                if words and t.word.array is not None and (words[0].raw not in DECLARATION):
                    raise ParseError("syntax error near unexpected token `('", t.line, "(")
                if not words and not assigns and self._is_funcdef():
                    return self._funcdef(t)
                if words and self._alias_next and self.aliases is not None:
                    self._alias_next = False
                    if self._maybe_alias(t):
                        continue
                self.advance()
                words.append(t.word)
                continue
            if t.kind == "op" and t.val in REDIR_OPS:
                redirs.append(self._redirect())
                continue
            break
        if not (assigns or words or redirs):
            self._unexpected(self.peek())
        end = self.peek().start if self._tok is not None else self.i
        return Simple(assigns, words, redirs, line, self.s[start:end].strip())

    @staticmethod
    def _assign_ok(w: Word) -> bool:
        """`name=` must be unquoted up to the `=`."""
        if not w.parts or not isinstance(w.parts[0], Lit):
            return False
        m = ASSIGN_RE.match(w.raw)
        return m is not None and ("=" in w.parts[0].text)

    def _make_assign(self, t: Tok) -> Assign:
        w = t.word
        m = ASSIGN_RE.match(t.val)
        name, index, append = m.group(1), m.group(2), bool(m.group(3))
        idx_word = self._sub_word(index[1:-1], "arith") if index else None
        if w.array is not None:
            return Assign(name, idx_word, None, w.array, append, t.val)
        # strip "name[...]+=" from the parts
        prefix_len = m.end()
        parts = []
        consumed = 0
        for p in w.parts:
            if consumed >= prefix_len:
                parts.append(p)
                continue
            if isinstance(p, Lit):
                take = prefix_len - consumed
                if len(p.text) > take:
                    parts.append(Lit(p.text[take:]))
                consumed += len(p.text)
            else:
                # the subscript itself contained an expansion: keep the rest after the "="
                consumed = prefix_len
                parts.append(p)
        return Assign(name, idx_word, Word(parts, t.val[prefix_len:]), None, append, t.val)

    def _is_funcdef(self) -> bool:
        """Is the peeked word followed by `(` `)` ?"""
        s, n = self.s, self.n
        j = self.i
        while j < n and s[j] in " \t":
            j += 1
        if j >= n or s[j] != "(":
            return False
        j += 1
        while j < n and s[j] in " \t":
            j += 1
        return j < n and s[j] == ")"

    def _funcdef(self, t: Tok) -> FuncDef:
        self.advance()
        j = self.s.index("(", self.i)
        self.i = self.s.index(")", j) + 1
        self._tok = None
        return self._func_body(t.val, t.start, t.line)

    def _func_body(self, name: str, start: int, line: int, keyword=False) -> FuncDef:
        while self.peek().kind == "nl":
            self.advance()
        t = self.peek()
        body = self.parse_command()
        if isinstance(body, (Simple, FuncDef)):
            self._unexpected(t)
        end = self.peek().start if self._tok is not None else self.i
        return FuncDef(name, body, self.s[start:end].strip(), line, keyword)

    def _function_kw(self) -> FuncDef:
        kw = self.advance()
        name = self.advance()
        if name.kind != "word":
            self._unexpected(name)
        s, n = self.s, self.n
        j = self.i
        while j < n and s[j] in " \t":
            j += 1
        if j < n and s[j] == "(":
            k = j + 1
            while k < n and s[k] in " \t":
                k += 1
            if k < n and s[k] == ")":
                self.i = k + 1
                self._tok = None
        return self._func_body(name.val, kw.start, kw.line, True)

    def _expect_kw(self, word: str) -> Tok:
        t = self.advance()
        if not self._is_kw(t, {word}):
            self._unexpected(t)
        return t

    def _brace(self) -> Brace:
        t = self.advance()
        body = self.parse_compound_list({"}"})
        self._expect_kw("}")
        return Brace(body, line=t.line)

    def _if(self) -> If:
        t = self.advance()
        clauses = []
        cond = self.parse_compound_list({"then"})
        self._expect_kw("then")
        body = self.parse_compound_list({"elif", "else", "fi"})
        clauses.append((cond, body))
        else_ = None
        while True:
            k = self.advance()
            if self._is_kw(k, {"elif"}):
                cond = self.parse_compound_list({"then"})
                self._expect_kw("then")
                body = self.parse_compound_list({"elif", "else", "fi"})
                clauses.append((cond, body))
            elif self._is_kw(k, {"else"}):
                else_ = self.parse_compound_list({"fi"})
                self._expect_kw("fi")
                break
            elif self._is_kw(k, {"fi"}):
                break
            else:
                self._unexpected(k)
        return If(clauses, else_, line=t.line)

    def _while(self, until: bool) -> While:
        t = self.advance()
        cond = self.parse_compound_list({"do"})
        self._expect_kw("do")
        body = self.parse_compound_list({"done"})
        self._expect_kw("done")
        return While(cond, body, until, line=t.line)

    def _do_body(self) -> CmdList:
        while True:
            t = self.peek()
            if t.kind == "nl" or (t.kind == "op" and t.val == ";"):
                self.advance()
                continue
            break
        t = self.peek()
        if self._is_kw(t, {"{"}):
            self.advance()
            body = self.parse_compound_list({"}"})
            self._expect_kw("}")
            return body
        self._expect_kw("do")
        body = self.parse_compound_list({"done"})
        self._expect_kw("done")
        return body

    def _for(self):
        t = self.advance()
        s = self.s
        j = self.i
        while j < self.n and s[j] in " \t":
            j += 1
        if s.startswith("((", j):
            end = self._match_arith(j + 2)
            if end is None:
                raise ParseError("syntax error near unexpected token `('", t.line, "(")
            inner = s[j + 2:end]
            pieces = _split_semis(inner)
            if len(pieces) != 3:
                raise ParseError("syntax error: arithmetic expression required", t.line)
            self.i = end + 2
            self._tok = None
            init, cond, step = (self._sub_word(p, "arith") for p in pieces)
            body = self._do_body()
            return ArithFor(init, cond, step, body, line=t.line)
        name = self.advance()
        if name.kind != "word":
            self._unexpected(name)
        if not NAME_RE.fullmatch(name.val):
            raise ParseError(f"`{name.val}': not a valid identifier", name.line)
        words = None
        while self.peek().kind == "nl":
            self.advance()
        k = self.peek()
        if self._is_kw(k, {"in"}):
            self.advance()
            words = []
            while True:
                w = self.peek()
                if w.kind == "word":
                    self.advance()
                    words.append(w.word)
                    continue
                break
            w = self.peek()
            if not (w.kind == "nl" or (w.kind == "op" and w.val == ";")):
                self._unexpected(w)
        body = self._do_body()
        return For(name.val, words, body, line=t.line)

    def _select(self):
        t = self.advance()
        name = self.advance()
        words = None
        while self.peek().kind == "nl":
            self.advance()
        if self._is_kw(self.peek(), {"in"}):
            self.advance()
            words = []
            while self.peek().kind == "word":
                words.append(self.advance().word)
        body = self._do_body()
        return Select(name.val, words, body, line=t.line)

    def _case(self) -> Case:
        t = self.advance()
        w = self.advance()
        if w.kind != "word":
            self._unexpected(w)
        while self.peek().kind == "nl":
            self.advance()
        self._expect_kw("in")
        items = []
        while True:
            while self.peek().kind == "nl":
                self.advance()
            k = self.peek()
            if self._is_kw(k, {"esac"}):
                self.advance()
                break
            if k.kind == "op" and k.val == "(":
                self.advance()
            pats = []
            while True:
                p = self.advance()
                if p.kind != "word":
                    self._unexpected(p)
                pats.append(p.word)
                sep = self.advance()
                if sep.kind == "op" and sep.val == "|":
                    continue
                if sep.kind == "op" and sep.val == ")":
                    break
                self._unexpected(sep)
            while self.peek().kind == "nl":
                self.advance()
            k = self.peek()
            body = None
            if not ((k.kind == "op" and k.val in (";;", ";&", ";;&")) or self._is_kw(k, {"esac"})):
                body = self.parse_compound_list({";;", ";&", ";;&", "esac"})
            k = self.peek()
            term = ";;"
            if k.kind == "op" and k.val in (";;", ";&", ";;&"):
                self.advance()
                term = k.val
            elif not self._is_kw(k, {"esac"}):
                self._unexpected(k)
            items.append((pats, body, term))
        return Case(w.word, items, line=t.line)

    # ------------------------------------------------------------ [[ ]]

    def _cond(self) -> Cond:
        t = self.advance()
        start = self.i
        expr = self._cond_or()
        k = self.peek()
        if not (k.kind == "word" and k.val == "]]"):
            if k.kind == "eof":
                self._eof()
            raise ParseError(f"syntax error in conditional expression: unexpected token `{k.text}'", k.line, k.text)
        src = self.s[start:k.start].strip()
        self.advance()
        return Cond(expr, line=t.line, src=src)

    def _cond_skip_nl(self):
        while self.peek().kind == "nl":
            self.advance()

    def _cond_or(self):
        left = self._cond_and()
        while True:
            t = self.peek()
            if t.kind == "op" and t.val == "||":
                self.advance()
                self._cond_skip_nl()
                left = CondOr(left, self._cond_and())
            else:
                return left

    def _cond_and(self):
        left = self._cond_not()
        while True:
            t = self.peek()
            if t.kind == "op" and t.val == "&&":
                self.advance()
                self._cond_skip_nl()
                left = CondAnd(left, self._cond_not())
            else:
                return left

    def _cond_not(self):
        t = self.peek()
        if t.kind == "word" and t.val == "!" and not t.word.is_quoted():
            self.advance()
            return CondNot(self._cond_not())
        return self._cond_primary()

    def _cond_primary(self):
        t = self.peek()
        if t.kind == "op" and t.val == "(":
            self.advance()
            e = self._cond_or()
            c = self.advance()
            if c.kind != "op" or c.val != ")":
                raise ParseError(f"syntax error in conditional expression: unexpected token `{c.text}'", c.line, c.text)
            return e
        if t.kind != "word" or t.val == "]]":
            if t.kind == "eof":
                self._eof()
            raise ParseError(f"unexpected token `{t.text}', conditional binary operator expected"
                             if t.kind != "word" else "syntax error in conditional expression", t.line, t.text)
        self.advance()
        if t.val in COND_UNARY and not t.word.is_quoted():
            nt = self.peek()
            if nt.kind == "word" and nt.val != "]]":
                self.advance()
                return CondUnary(t.val, nt.word)
            if nt.kind == "word" and nt.val == "]]":
                return CondWord(t.word)   # [[ -f ]] tests the string "-f"
        nt = self.peek()
        op = None
        if nt.kind == "word" and nt.val in COND_BINARY and not nt.word.is_quoted():
            op = nt.val
        elif nt.kind == "op" and nt.val in ("<", ">"):
            op = nt.val
        if op is None:
            return CondWord(t.word)
        self.advance()
        if op == "=~":
            self._tok = None
            self._skip_blanks()
            right = self._word(regex=True)
            if not right.parts:
                raise ParseError("unexpected argument to conditional binary operator", nt.line)
            return CondBinary(op, t.word, right)
        r = self.advance()
        if r.kind != "word":
            raise ParseError(f"unexpected argument `{r.text}' to conditional binary operator", r.line, r.text)
        return CondBinary(op, t.word, r.word)


def _split_semis(text: str) -> list[str]:
    out, depth, cur = [], 0, []
    for c in text:
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        if c == ";" and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(c)
    out.append("".join(cur))
    return out


def parse(src: str) -> CmdList:
    """Parse a whole script (no aliases). Raises ParseError."""
    return Parser(src).parse_all()
