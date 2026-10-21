"""AST node types produced by parser.py and executed by shell.py."""

from __future__ import annotations


class Node:
    __slots__ = ()

    def __repr__(self):
        fields = ", ".join(f"{k}={getattr(self, k)!r}" for k in self.__slots__ if getattr(self, k, None) not in (None, [], False, ""))
        return f"{type(self).__name__}({fields})"


# ------------------------------------------------------------------ word parts

class Lit(Node):
    """Unquoted literal text (may hold glob, brace and tilde characters)."""
    __slots__ = ("text",)

    def __init__(self, text):
        self.text = text


class Esc(Node):
    """A backslash-escaped character: quoted, so literal."""
    __slots__ = ("text",)

    def __init__(self, text):
        self.text = text


class SQ(Node):
    """'single quoted' or $'ansi-c' text (already decoded)."""
    __slots__ = ("text", "ansi")

    def __init__(self, text, ansi=False):
        self.text, self.ansi = text, ansi


class DQ(Node):
    """"double quoted" parts (Lit parts inside are quoted)."""
    __slots__ = ("parts",)

    def __init__(self, parts):
        self.parts = parts


class Param(Node):
    """$name, ${name...}. op: '' | 'len' | ':-' '-' ':=' '=' ':?' '?' ':+' '+' | '#' '##' '%' '%%' |
    '/' '//' '/#' '/%' | '^' '^^' ',' ',,' | ':' (substring) | '@' (transform) | '!' prefix ops."""
    __slots__ = ("name", "sub", "op", "arg", "arg2", "indirect", "braced")

    def __init__(self, name, sub=None, op="", arg=None, arg2=None, indirect=False, braced=False):
        self.name, self.sub, self.op, self.arg, self.arg2, self.indirect, self.braced = name, sub, op, arg, arg2, indirect, braced


class CmdSub(Node):
    __slots__ = ("body", "src", "backtick")

    def __init__(self, body, src, backtick=False):
        self.body, self.src, self.backtick = body, src, backtick


class ArithSub(Node):
    __slots__ = ("expr", "src")

    def __init__(self, expr, src):
        self.expr, self.src = expr, src


class ProcSub(Node):
    __slots__ = ("direction", "body", "src")

    def __init__(self, direction, body, src):
        self.direction, self.body, self.src = direction, body, src


class Word(Node):
    __slots__ = ("parts", "raw", "array")

    def __init__(self, parts, raw="", array=None):
        self.parts, self.raw, self.array = parts, raw, array

    def is_quoted(self) -> bool:
        return any(not isinstance(p, Lit) for p in self.parts)

    def literal(self) -> str | None:
        """The word's text if it contains no expansions (quotes removed), else None."""
        out = []
        for p in self.parts:
            if isinstance(p, (Lit, Esc)):
                out.append(p.text)
            elif isinstance(p, SQ):
                out.append(p.text)
            elif isinstance(p, DQ):
                for q in p.parts:
                    if isinstance(q, (Lit, Esc)):
                        out.append(q.text)
                    else:
                        return None
            else:
                return None
        return "".join(out)


# ------------------------------------------------------------------ commands

class Assign(Node):
    __slots__ = ("name", "index", "value", "array", "append", "raw")

    def __init__(self, name, index, value, array=None, append=False, raw=""):
        self.name, self.index, self.value, self.array, self.append, self.raw = name, index, value, array, append, raw


class HereDoc(Node):
    __slots__ = ("delim", "strip_tabs", "quoted", "body", "text")

    def __init__(self, delim, strip_tabs, quoted):
        self.delim, self.strip_tabs, self.quoted = delim, strip_tabs, quoted
        self.body = None    # Word (unquoted delimiter: expansions) or None
        self.text = ""      # raw body text


class Redir(Node):
    __slots__ = ("fd", "op", "target", "heredoc")

    def __init__(self, fd, op, target=None, heredoc=None):
        self.fd, self.op, self.target, self.heredoc = fd, op, target, heredoc


class Simple(Node):
    __slots__ = ("assigns", "words", "redirs", "line", "src")

    def __init__(self, assigns, words, redirs, line, src=""):
        self.assigns, self.words, self.redirs, self.line, self.src = assigns, words, redirs, line, src


class Pipeline(Node):
    __slots__ = ("cmds", "negate", "timed", "posix_time", "pipe_err", "src", "line")

    def __init__(self, cmds, negate=False, timed=False, posix_time=False, pipe_err=None, src="", line=0):
        self.cmds, self.negate, self.timed, self.posix_time = cmds, negate, timed, posix_time
        self.pipe_err, self.src, self.line = pipe_err or [], src, line


class AndOr(Node):
    __slots__ = ("first", "rest", "src")

    def __init__(self, first, rest, src=""):
        self.first, self.rest, self.src = first, rest, src


class CmdList(Node):
    """items: [(AndOr, background: bool)]"""
    __slots__ = ("items",)

    def __init__(self, items):
        self.items = items


class Compound(Node):
    __slots__ = ()


class Brace(Compound):
    __slots__ = ("body", "redirs", "line")

    def __init__(self, body, redirs=None, line=0):
        self.body, self.redirs, self.line = body, redirs or [], line


class Subshell(Compound):
    __slots__ = ("body", "redirs", "line")

    def __init__(self, body, redirs=None, line=0):
        self.body, self.redirs, self.line = body, redirs or [], line


class If(Compound):
    __slots__ = ("clauses", "else_", "redirs", "line")

    def __init__(self, clauses, else_, redirs=None, line=0):
        self.clauses, self.else_, self.redirs, self.line = clauses, else_, redirs or [], line


class For(Compound):
    __slots__ = ("var", "words", "body", "redirs", "line")

    def __init__(self, var, words, body, redirs=None, line=0):
        self.var, self.words, self.body, self.redirs, self.line = var, words, body, redirs or [], line


class Select(Compound):
    __slots__ = ("var", "words", "body", "redirs", "line")

    def __init__(self, var, words, body, redirs=None, line=0):
        self.var, self.words, self.body, self.redirs, self.line = var, words, body, redirs or [], line


class ArithFor(Compound):
    __slots__ = ("init", "cond", "step", "body", "redirs", "line")

    def __init__(self, init, cond, step, body, redirs=None, line=0):
        self.init, self.cond, self.step, self.body, self.redirs, self.line = init, cond, step, body, redirs or [], line


class While(Compound):
    __slots__ = ("cond", "body", "until", "redirs", "line")

    def __init__(self, cond, body, until=False, redirs=None, line=0):
        self.cond, self.body, self.until, self.redirs, self.line = cond, body, until, redirs or [], line


class Case(Compound):
    """items: [(patterns: [Word], body: CmdList | None, terminator: ';;' | ';&' | ';;&')]"""
    __slots__ = ("word", "items", "redirs", "line")

    def __init__(self, word, items, redirs=None, line=0):
        self.word, self.items, self.redirs, self.line = word, items, redirs or [], line


class Cond(Compound):
    """[[ expression ]]"""
    __slots__ = ("expr", "redirs", "line", "src")

    def __init__(self, expr, redirs=None, line=0, src=""):
        self.expr, self.redirs, self.line, self.src = expr, redirs or [], line, src


class ArithCmd(Compound):
    """(( expression ))"""
    __slots__ = ("expr", "redirs", "line", "src")

    def __init__(self, expr, redirs=None, line=0, src=""):
        self.expr, self.redirs, self.line, self.src = expr, redirs or [], line, src


class FuncDef(Node):
    __slots__ = ("name", "body", "src", "line", "keyword")

    def __init__(self, name, body, src="", line=0, keyword=False):
        self.name, self.body, self.src, self.line, self.keyword = name, body, src, line, keyword


# ------------------------------------------------------------------ [[ ]] expression nodes

class CondUnary(Node):
    __slots__ = ("op", "arg")

    def __init__(self, op, arg):
        self.op, self.arg = op, arg


class CondBinary(Node):
    __slots__ = ("op", "left", "right")

    def __init__(self, op, left, right):
        self.op, self.left, self.right = op, left, right


class CondAnd(Node):
    __slots__ = ("left", "right")

    def __init__(self, left, right):
        self.left, self.right = left, right


class CondOr(Node):
    __slots__ = ("left", "right")

    def __init__(self, left, right):
        self.left, self.right = left, right


class CondNot(Node):
    __slots__ = ("arg",)

    def __init__(self, arg):
        self.arg = arg


class CondWord(Node):
    __slots__ = ("word",)

    def __init__(self, word):
        self.word = word
