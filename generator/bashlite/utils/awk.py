"""awk (GNU awk flavour): the program is compiled to Python source, so record loops run fast.

Values are Python floats (numbers), str (strings) and StrNum (input data: strings that compare as numbers
when they look numeric). Programs that talk to commands (system, cmd | getline, print | cmd, close) are
compiled to generator functions so the simulated scheduler can run those commands.
"""

from __future__ import annotations

import math
import random
import re

from ..exec import util
from ..regex import RegexError, translate
from ..vfs import FSError


class AwkSyntaxError(Exception):
    def __init__(self, msg, line=1, col=0, src_line=""):
        super().__init__(msg)
        self.msg, self.line, self.col, self.src_line = msg, line, col, src_line


class AwkFatal(Exception):
    pass


class NextRecord(Exception):
    pass


class NextFile(Exception):
    pass


class AwkExit(Exception):
    def __init__(self, status):
        super().__init__(status)
        self.status = status


# ---------------------------------------------------------------------------- values

NUMRE = re.compile(r"[ \t\n\r\f\v]*[+-]?(?:\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)[ \t\n\r\f\v]*\Z")
PREFIX = re.compile(r"[ \t\n\r\f\v]*([+-]?(?:\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?))")
SPECIAL_NUM = re.compile(r"[ \t]*([+-])(inf|nan)", re.I)


class StrNum(str):
    """A string from input data; compares numerically when it looks like a number."""
    __slots__ = ()

    @property
    def isnum(self):
        return NUMRE.match(self) is not None


class Uninit(str):
    __slots__ = ()


UNINIT = Uninit("")


def to_num(v) -> float:
    if type(v) is float:
        return v
    if isinstance(v, (int, bool)):
        return float(v)
    m = PREFIX.match(v)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return 0.0
    m = SPECIAL_NUM.match(v)
    if m:
        return float(m.group(1) + m.group(2))
    return 0.0


def fmt_num(v: float, fmt: str) -> str:
    if v != v:
        return "-nan" if math.copysign(1, v) < 0 else "+nan"
    if v in (math.inf, -math.inf):
        return "+inf" if v > 0 else "-inf"
    if v == int(v):
        return str(int(v))      # gawk prints integral values in full, whatever CONVFMT/OFMT say
    try:
        return fmt % v
    except (TypeError, ValueError):
        return "%.6g" % v


def is_numeric(v) -> bool:
    t = type(v)
    if t is float:
        return True
    if t is Uninit:
        return True
    if t is StrNum:
        return NUMRE.match(v) is not None
    return False


def truthy(v) -> bool:
    t = type(v)
    if t is float:
        return v != 0
    if t is Uninit:
        return False
    if t is StrNum and NUMRE.match(v):
        return to_num(v) != 0
    return v != ""


def make_strnum(s: str):
    return StrNum(s)


# ---------------------------------------------------------------------------- lexer

KEYWORDS = {"BEGIN", "END", "function", "func", "if", "else", "while", "for", "do", "break", "continue", "next",
            "nextfile", "exit", "return", "delete", "in", "getline", "print", "printf", "BEGINFILE", "ENDFILE"}
BUILTINS = {"length", "substr", "index", "split", "sub", "gsub", "match", "sprintf", "sin", "cos", "atan2", "exp",
            "log", "sqrt", "int", "rand", "srand", "tolower", "toupper", "system", "close", "fflush", "gensub",
            "systime", "strftime", "mktime", "asort", "asorti", "and", "or", "xor", "lshift", "rshift", "compl",
            "strtonum", "typeof", "isarray"}
OPS = ["**=", "...", "+=", "-=", "*=", "/=", "%=", "^=", "++", "--", "&&", "||", "==", "!=", "<=", ">=", "!~",
       "**", ">>", "|&", "+", "-", "*", "/", "%", "^", "!", ">", "<", "|", "?", ":", "~", "$", "=", ",", ";", "{",
       "}", "(", ")", "[", "]"]


class Tok:
    __slots__ = ("kind", "val", "line", "col")

    def __init__(self, kind, val, line, col):
        self.kind, self.val, self.line, self.col = kind, val, line, col

    def __repr__(self):
        return f"{self.kind}:{self.val!r}"


def lex(src: str):
    toks = []
    i, n, line = 0, len(src), 1
    line_start = 0

    def regex_ok():
        if not toks:
            return True
        t = toks[-1]
        if t.kind in ("number", "string", "ere", "name", "funcname", "builtin"):
            return t.kind == "builtin" and t.val not in ("length",)
        if t.kind == "op" and t.val in (")", "]", "$", "++", "--"):
            return False
        if t.kind == "kw" and t.val in ("getline",):
            return False
        return True
    while i < n:
        c = src[i]
        if c in " \t\r":
            i += 1
            continue
        if c == "\\" and i + 1 < n and src[i + 1] == "\n":
            i += 2
            line += 1
            line_start = i
            continue
        if c == "\\" and i + 2 < n and src[i + 1] == "\r" and src[i + 2] == "\n":
            i += 3
            line += 1
            continue
        if c == "#":
            while i < n and src[i] != "\n":
                i += 1
            continue
        col = i - line_start
        if c == "\n":
            toks.append(Tok("nl", "\n", line, col))
            i += 1
            line += 1
            line_start = i
            continue
        if c == '"':
            j = i + 1
            out = []
            while True:
                if j >= n or src[j] == "\n":
                    raise AwkSyntaxError("unterminated string", line, col, _src_line(src, line))
                ch = src[j]
                if ch == '"':
                    break
                if ch == "\\" and j + 1 < n:
                    d = src[j + 1]
                    if d == "\n":
                        j += 2
                        continue
                    if d in "01234567":
                        k = j + 1
                        while k < n and k < j + 4 and src[k] in "01234567":
                            k += 1
                        out.append(chr(int(src[j + 1:k], 8)))
                        j = k
                        continue
                    out.append({"n": "\n", "t": "\t", "\\": "\\", '"': '"', "/": "/", "a": "\a", "b": "\b",
                                "f": "\f", "r": "\r", "v": "\v"}.get(d, "\\" + d if d in ".[]()*+?{}|^$" else d))
                    j += 2
                    continue
                out.append(ch)
                j += 1
            toks.append(Tok("string", "".join(out), line, col))
            i = j + 1
            continue
        if c == "/" and regex_ok():
            j = i + 1
            out = []
            in_br = False
            while True:
                if j >= n or src[j] == "\n":
                    raise AwkSyntaxError("unterminated regexp", line, col, _src_line(src, line))
                ch = src[j]
                if ch == "\\" and j + 1 < n:
                    if src[j + 1] == "/":
                        out.append("/")
                    else:
                        out.append(src[j:j + 2])
                    j += 2
                    continue
                if ch == "[" and not in_br:
                    in_br = True
                    out.append(ch)
                    j += 1
                    if j < n and src[j] == "^":
                        out.append("^")
                        j += 1
                    if j < n and src[j] == "]":
                        out.append("]")
                        j += 1
                    continue
                if ch == "]" and in_br:
                    in_br = False
                elif ch == "/" and not in_br:
                    break
                out.append(ch)
                j += 1
            toks.append(Tok("ere", "".join(out), line, col))
            i = j + 1
            continue
        if c.isdigit() or (c == "." and i + 1 < n and src[i + 1].isdigit()):
            m = re.match(r"0[xX][0-9a-fA-F]+|(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?", src[i:])
            text = m.group(0)
            toks.append(Tok("number", float(int(text, 16)) if text[:2].lower() == "0x" else float(text), line, col))
            i += m.end()
            continue
        if c.isalpha() or c == "_":
            m = re.match(r"[A-Za-z_][A-Za-z0-9_]*", src[i:])
            w = m.group(0)
            i += m.end()
            if w in KEYWORDS:
                toks.append(Tok("kw", "function" if w == "func" else w, line, col))
            elif w in BUILTINS:
                toks.append(Tok("builtin", w, line, col))
            elif i < n and src[i] == "(":
                toks.append(Tok("funcname", w, line, col))
            else:
                toks.append(Tok("name", w, line, col))
            continue
        for op in OPS:
            if src.startswith(op, i):
                toks.append(Tok("op", op, line, col))
                i += len(op)
                break
        else:
            raise AwkSyntaxError(f"invalid char '{c}' in expression", line, col, _src_line(src, line))
    toks.append(Tok("eof", None, line, n - line_start))
    return toks


def _src_line(src, line):
    lines = src.split("\n")
    return lines[line - 1] if 0 < line <= len(lines) else ""


# ---------------------------------------------------------------------------- parser -> AST (tuples)

class Parser:
    def __init__(self, src):
        self.src = src
        self.toks = lex(src)
        self.i = 0
        self.funcs = {}
        self.items = []    # (kind, pattern, pattern2, body)
        self.in_func = None
        self.uses_cmd = False
        self.uses_getline = False
        self.uses_nr_end = False

    def peek(self, k=0):
        return self.toks[min(self.i + k, len(self.toks) - 1)]

    def advance(self):
        t = self.toks[self.i]
        self.i += 1
        return t

    def at(self, kind, val=None):
        t = self.peek()
        return t.kind == kind and (val is None or t.val == val)

    def at_op(self, *vals):
        t = self.peek()
        return t.kind == "op" and t.val in vals

    def expect_op(self, val):
        t = self.advance()
        if t.kind != "op" or t.val != val:
            self.error(t)
        return t

    def error(self, t, msg=None):
        if msg is None:
            if t.kind == "eof" or t.kind == "nl":
                msg = "syntax error" if t.kind == "nl" else "unexpected newline or end of string"
            else:
                msg = "syntax error"
        raise AwkSyntaxError(msg, t.line, t.col, _src_line(self.src, t.line))

    def skip_nl(self):
        while self.at("nl"):
            self.advance()

    def skip_terms(self):
        while self.at("nl") or self.at_op(";"):
            self.advance()

    def parse(self):
        self.skip_terms()
        while not self.at("eof"):
            self.item()
            self.skip_terms()
        return self

    def item(self):
        t = self.peek()
        if t.kind == "kw" and t.val in ("BEGIN", "END", "BEGINFILE", "ENDFILE"):
            self.advance()
            self.skip_nl()
            if not self.at_op("{"):
                self.error(self.peek(), f"each rule must have a pattern or an action part")
            self.items.append((t.val, None, None, self.block()))
            return
        if t.kind == "kw" and t.val == "function":
            self.advance()
            nt = self.advance()
            if nt.kind not in ("name", "funcname"):
                self.error(nt)
            name = nt.val
            self.expect_op("(")
            params = []
            while not self.at_op(")"):
                p = self.advance()
                if p.kind != "name":
                    self.error(p)
                params.append(p.val)
                if self.at_op(","):
                    self.advance()
                    self.skip_nl()
            self.expect_op(")")
            self.skip_nl()
            self.in_func = params
            body = self.block()
            self.in_func = None
            self.funcs[name] = (params, body)
            return
        pat = pat2 = None
        if not self.at_op("{"):
            pat = self.expr()
            if self.at_op(","):
                self.advance()
                self.skip_nl()
                pat2 = self.expr()
        if self.at_op("{"):
            body = self.block()
        else:
            body = None
        self.items.append(("main", pat, pat2, body))

    def block(self):
        self.expect_op("{")
        stmts = []
        self.skip_terms()
        while not self.at_op("}"):
            if self.at("eof"):
                self.error(self.peek())
            stmts.append(self.stmt())
            self.skip_terms()
        self.expect_op("}")
        return ("block", stmts)

    def simple_end(self):
        t = self.peek()
        if t.kind == "nl" or (t.kind == "op" and t.val in (";", "}")) or t.kind == "eof":
            if t.kind != "op" or t.val != "}":
                if t.kind != "eof":
                    self.advance()
            return
        self.error(t)

    def stmt(self):
        t = self.peek()
        if t.kind == "op" and t.val == "{":
            return self.block()
        if t.kind == "op" and t.val == ";":
            self.advance()
            return ("block", [])
        if t.kind == "kw":
            v = t.val
            if v == "if":
                self.advance()
                self.expect_op("(")
                cond = self.expr()
                self.expect_op(")")
                self.skip_nl()
                then = self.stmt()
                save = self.i
                self.skip_terms()
                if self.at("kw", "else"):
                    self.advance()
                    self.skip_nl()
                    return ("if", cond, then, self.stmt())
                self.i = save
                return ("if", cond, then, None)
            if v == "while":
                self.advance()
                self.expect_op("(")
                cond = self.expr()
                self.expect_op(")")
                if self.at_op(";"):
                    self.advance()
                    return ("while", cond, ("block", []))
                self.skip_nl()
                return ("while", cond, self.stmt())
            if v == "do":
                self.advance()
                self.skip_nl()
                body = self.stmt()
                self.skip_terms()
                if not self.at("kw", "while"):
                    self.error(self.peek())
                self.advance()
                self.expect_op("(")
                cond = self.expr()
                self.expect_op(")")
                self.simple_end()
                return ("do", body, cond)
            if v == "for":
                self.advance()
                self.expect_op("(")
                # for (k in arr)
                if self.at("name") and self.peek(1).kind == "kw" and self.peek(1).val == "in" and \
                        self.peek(2).kind == "name" and self.peek(3).kind == "op" and self.peek(3).val == ")":
                    var = self.advance().val
                    self.advance()
                    arr = self.advance().val
                    self.expect_op(")")
                    self.skip_nl()
                    return ("forin", var, arr, self.stmt())
                if self.at_op("(") and False:
                    pass
                init = None if self.at_op(";") else self.simple_stmt_expr()
                self.expect_op(";")
                self.skip_nl()
                cond = None if self.at_op(";") else self.expr()
                self.expect_op(";")
                self.skip_nl()
                step = None if self.at_op(")") else self.simple_stmt_expr()
                self.expect_op(")")
                if self.at_op(";"):
                    self.advance()
                    return ("for", init, cond, step, ("block", []))
                self.skip_nl()
                return ("for", init, cond, step, self.stmt())
            if v in ("break", "continue", "next", "nextfile"):
                self.advance()
                self.simple_end()
                return (v,)
            if v == "exit":
                self.advance()
                e = None
                if not (self.at("nl") or self.at_op(";", "}") or self.at("eof")):
                    e = self.expr()
                self.simple_end()
                return ("exit", e)
            if v == "return":
                self.advance()
                e = None
                if not (self.at("nl") or self.at_op(";", "}") or self.at("eof")):
                    e = self.expr()
                self.simple_end()
                return ("return", e)
            if v == "delete":
                self.advance()
                nt = self.advance()
                if nt.kind != "name":
                    self.error(nt)
                if self.at_op("["):
                    self.advance()
                    idx = self.expr_list()
                    self.expect_op("]")
                    self.simple_end()
                    return ("delete", nt.val, idx)
                self.simple_end()
                return ("delete", nt.val, None)
            if v in ("print", "printf"):
                s = self.print_stmt()
                self.simple_end()
                return s
        e = self.expr()
        self.simple_end()
        return ("expr", e)

    def simple_stmt_expr(self):
        return self.expr()

    def print_stmt(self):
        kind = self.advance().val
        args = []
        if self.at_op("(") and self._paren_print():
            self.advance()
            args = self.expr_list(no_gt=False)
            self.expect_op(")")
        elif not (self.at("nl") or self.at_op(";", "}", ">", ">>", "|") or self.at("eof")):
            args = self.expr_list(no_gt=True)
        dest = None
        if self.at_op(">", ">>", "|", "|&"):
            op = self.advance().val
            target = self.concat(no_gt=True)
            dest = (op, target)
            if op == "|":
                self.uses_cmd = True
        if kind == "printf" and not args:
            self.error(self.peek())
        return (kind, args, dest)

    def _paren_print(self):
        """print (a, b) > f : the parenthesised list is the argument list if it closes before the end."""
        depth = 0
        j = self.i
        while j < len(self.toks):
            t = self.toks[j]
            if t.kind == "op" and t.val in "([":
                depth += 1
            elif t.kind == "op" and t.val in ")]":
                depth -= 1
                if depth == 0:
                    nt = self.toks[j + 1]
                    return nt.kind in ("nl", "eof") or (nt.kind == "op" and nt.val in (";", "}", ">", ">>", "|", "|&"))
            elif t.kind in ("nl", "eof"):
                return False
            j += 1
        return False

    def expr_list(self, no_gt=False):
        out = [self.expr(no_gt)]
        while self.at_op(","):
            self.advance()
            self.skip_nl()
            out.append(self.expr(no_gt))
        return out

    # expressions -----------------------------------------------------------

    def expr(self, no_gt=False):
        return self.ternary(no_gt)

    def ternary(self, no_gt):
        cond = self.or_(no_gt)
        if self.at_op("?"):
            self.advance()
            self.skip_nl()
            a = self.ternary(no_gt)
            self.skip_nl()
            self.expect_op(":")
            self.skip_nl()
            b = self.ternary(no_gt)
            return ("?:", cond, a, b)
        if self.at_op("=", "+=", "-=", "*=", "/=", "%=", "^=", "**="):
            if cond[0] in ("var", "field", "index"):
                op = self.advance().val
                self.skip_nl()
                rhs = self.ternary(no_gt)
                return ("assign", op, cond, rhs)
        return cond

    def or_(self, no_gt):
        left = self.and_(no_gt)
        while self.at_op("||"):
            self.advance()
            self.skip_nl()
            left = ("||", left, self.and_(no_gt))
        return left

    def and_(self, no_gt):
        left = self.in_(no_gt)
        while self.at_op("&&"):
            self.advance()
            self.skip_nl()
            left = ("&&", left, self.in_(no_gt))
        return left

    def in_(self, no_gt):
        left = self.match(no_gt)
        while self.at("kw", "in"):
            self.advance()
            arr = self.advance()
            if arr.kind != "name":
                self.error(arr)
            left = ("in", [left] if left[0] != "group" else left[1], arr.val)
        return left

    def match(self, no_gt):
        left = self.comparison(no_gt)
        while self.at_op("~", "!~"):
            op = self.advance().val
            right = self.comparison(no_gt)
            left = (op, left, right)
        return left

    def comparison(self, no_gt):
        left = self.pipe_getline(no_gt)
        t = self.peek()
        if t.kind == "op" and t.val in ("<", "<=", "!=", "==", ">=") or (t.kind == "op" and t.val == ">" and not no_gt):
            op = self.advance().val
            right = self.pipe_getline(no_gt)
            return ("cmp", op, left, right)
        return left

    def pipe_getline(self, no_gt):
        left = self.concat(no_gt)
        while self.at_op("|", "|&") and self.peek(1).kind == "kw" and self.peek(1).val == "getline":
            self.advance()
            self.advance()
            var = None
            if self.at("name") or self.at_op("$"):
                var = self.primary_lvalue()
            self.uses_cmd = True
            left = ("cmdgetline", left, var)
        return left

    def concat(self, no_gt):
        left = self.additive(no_gt)
        while self._starts_operand():
            right = self.additive(no_gt)
            left = ("concat", left, right)
        return left

    def _starts_operand(self):
        t = self.peek()
        if t.kind in ("number", "string", "ere", "name", "funcname", "builtin"):
            return True
        if t.kind == "kw" and t.val == "getline":
            return False
        if t.kind == "op" and t.val in ("$", "!", "(", "-", "+", "++", "--"):
            if t.val in ("-", "+"):
                return False   # binary +/- bind tighter than concatenation
            if t.val == "!":
                return False
            return True
        return False

    def additive(self, no_gt):
        left = self.mult(no_gt)
        while self.at_op("+", "-"):
            op = self.advance().val
            self.skip_nl() if False else None
            left = (op, left, self.mult(no_gt))
        return left

    def mult(self, no_gt):
        left = self.unary(no_gt)
        while self.at_op("*", "/", "%"):
            op = self.advance().val
            left = (op, left, self.unary(no_gt))
        return left

    def unary(self, no_gt):
        if self.at_op("!"):
            self.advance()
            return ("!", self.unary(no_gt))
        if self.at_op("-"):
            self.advance()
            return ("neg", self.unary(no_gt))
        if self.at_op("+"):
            self.advance()
            return ("pos", self.unary(no_gt))
        return self.power(no_gt)

    def power(self, no_gt):
        base = self.postfix(no_gt)
        if self.at_op("^", "**"):
            self.advance()
            exp = self.power_rhs(no_gt)
            return ("^", base, exp)
        return base

    def power_rhs(self, no_gt):
        if self.at_op("-"):
            self.advance()
            return ("neg", self.power_rhs(no_gt))
        if self.at_op("+"):
            self.advance()
            return self.power_rhs(no_gt)
        if self.at_op("!"):
            self.advance()
            return ("!", self.power_rhs(no_gt))
        return self.power(no_gt)

    def postfix(self, no_gt):
        if self.at_op("++", "--"):
            op = self.advance().val
            lv = self.primary_lvalue()
            return ("preinc", 1 if op == "++" else -1, lv)
        e = self.primary(no_gt)
        if e[0] in ("var", "field", "index") and self.at_op("++", "--"):
            op = self.advance().val
            return ("postinc", 1 if op == "++" else -1, e)
        return e

    def primary_lvalue(self):
        if self.at_op("$"):
            self.advance()
            return ("field", self.field_operand())
        t = self.advance()
        if t.kind != "name":
            self.error(t)
        if self.at_op("["):
            self.advance()
            idx = self.expr_list()
            self.expect_op("]")
            return ("index", t.val, idx)
        return ("var", t.val)

    def field_operand(self):
        if self.at_op("++", "--"):
            op = self.advance().val
            lv = self.primary_lvalue()
            return ("preinc", 1 if op == "++" else -1, lv)
        if self.at_op("-"):
            self.advance()
            return ("neg", self.field_operand())
        return self.primary(False)

    def primary(self, no_gt):
        t = self.advance()
        k = t.kind
        if k == "number":
            return ("num", t.val)
        if k == "string":
            return ("str", t.val)
        if k == "ere":
            return ("regex", t.val)
        if k == "op" and t.val == "$":
            return ("field", self.field_operand())
        if k == "op" and t.val == "(":
            self.skip_nl()
            e = self.expr()
            if self.at_op(","):
                items = [e]
                while self.at_op(","):
                    self.advance()
                    self.skip_nl()
                    items.append(self.expr())
                self.expect_op(")")
                if not self.at("kw", "in"):
                    self.error(self.peek())
                return ("group", items)
            self.skip_nl()
            self.expect_op(")")
            return ("paren", e)
        if k == "op" and t.val in ("-", "+", "!"):
            inner = self.unary(no_gt)
            return {"-": ("neg", inner), "+": ("pos", inner), "!": ("!", inner)}[t.val]
        if k == "name":
            if self.at_op("["):
                self.advance()
                idx = self.expr_list()
                self.expect_op("]")
                return ("index", t.val, idx)
            return ("var", t.val)
        if k == "funcname":
            self.expect_op("(")
            args = []
            self.skip_nl()
            while not self.at_op(")"):
                args.append(self.expr())
                self.skip_nl()
                if self.at_op(","):
                    self.advance()
                    self.skip_nl()
            self.expect_op(")")
            return ("call", t.val, args)
        if k == "builtin":
            name = t.val
            if name == "length" and not self.at_op("("):
                return ("builtin", "length", [])
            self.expect_op("(")
            args = []
            self.skip_nl()
            while not self.at_op(")"):
                args.append(self.expr())
                self.skip_nl()
                if self.at_op(","):
                    self.advance()
                    self.skip_nl()
            self.expect_op(")")
            if name in ("system", "close", "fflush"):
                self.uses_cmd = True
            return ("builtin", name, args)
        if k == "kw" and t.val == "getline":
            var = None
            if self.at("name") or self.at_op("$"):
                var = self.primary_lvalue()
            src = None
            if self.at_op("<"):
                self.advance()
                src = self.primary(True)
            self.uses_getline = True
            return ("getline", var, src)
        self.error(t)


# ---------------------------------------------------------------------------- code generation

class Gen:
    def __init__(self, parser: Parser, gen_mode: bool):
        self.p = parser
        self.lines: list[str] = []
        self.gen_mode = gen_mode
        self.tmp = 0
        self.locals: set[str] | None = None
        self.array_params = {}
        self.regexes: list[str] = []
        self.loop_depth = 0
        self._infer_arrays()

    def _infer_arrays(self):
        """Which function parameters are used as arrays (fixpoint over calls)."""
        funcs = self.p.funcs
        arr = {f: set() for f in funcs}

        def scan(node, params, fname):
            if isinstance(node, tuple):
                if node and node[0] in ("index",) and node[1] in params:
                    arr[fname].add(node[1])
                if node and node[0] in ("in",) and node[2] in params:
                    arr[fname].add(node[2])
                if node and node[0] in ("forin",) and node[2] in params:
                    arr[fname].add(node[2])
                if node and node[0] == "delete" and node[1] in params:
                    arr[fname].add(node[1])
                if node and node[0] == "builtin" and node[1] in ("split", "asort", "asorti") and len(node[2]) > 1:
                    a = node[2][1]
                    if a[0] == "var" and a[1] in params:
                        arr[fname].add(a[1])
                    if node[1] in ("asort", "asorti") and node[2] and node[2][0][0] == "var" and node[2][0][1] in params:
                        arr[fname].add(node[2][0][1])
                if node and node[0] == "builtin" and node[1] == "match" and len(node[2]) > 2:
                    a = node[2][2]
                    if a[0] == "var" and a[1] in params:
                        arr[fname].add(a[1])
                if node and node[0] == "call" and node[1] in funcs:
                    cp = funcs[node[1]][0]
                    for k, a in enumerate(node[2]):
                        if k < len(cp) and cp[k] in arr[node[1]] and a[0] == "var" and a[1] in params:
                            arr[fname].add(a[1])
                for x in node[1:]:
                    scan(x, params, fname)
            elif isinstance(node, list):
                for x in node:
                    scan(x, params, fname)
        for _ in range(len(funcs) + 2):
            for f, (params, body) in funcs.items():
                scan(body, set(params), f)
        self.array_params = arr
        # global arrays: any name used with a subscript / in / delete / split target anywhere
        glob = set()

        def scan_g(node):
            if isinstance(node, tuple):
                if node and node[0] == "index":
                    glob.add(node[1])
                if node and node[0] == "in":
                    glob.add(node[2])
                if node and node[0] == "forin":
                    glob.add(node[2])
                if node and node[0] == "delete":
                    glob.add(node[1])
                if node and node[0] == "builtin" and node[1] in ("split", "asort", "asorti") and len(node[2]) > 1:
                    if node[2][1][0] == "var":
                        glob.add(node[2][1][1])
                if node and node[0] == "builtin" and node[1] in ("asort", "asorti") and node[2] and node[2][0][0] == "var":
                    glob.add(node[2][0][1])
                if node and node[0] == "builtin" and node[1] == "match" and len(node[2]) > 2 and node[2][2][0] == "var":
                    glob.add(node[2][2][1])
                if node and node[0] == "call" and node[1] in funcs:
                    cp = funcs[node[1]][0]
                    for k, a in enumerate(node[2]):
                        if k < len(cp) and cp[k] in arr[node[1]] and a[0] == "var":
                            glob.add(a[1])
                for x in node[1:]:
                    scan_g(x)
            elif isinstance(node, list):
                for x in node:
                    scan_g(x)
        scan_g([it[3] for it in self.p.items] + [it[1] for it in self.p.items] + [b for _, b in funcs.values()])
        self.global_arrays = glob | {"ENVIRON", "ARGV", "PROCINFO", "SYMTAB"}

    def emit(self, line, ind):
        self.lines.append("    " * ind + line)

    def new_tmp(self):
        self.tmp += 1
        return f"_t{self.tmp}"

    def yf(self, call: str) -> str:
        return f"(yield from {call})" if self.gen_mode else call

    # ---------------- variables

    def is_local(self, name):
        return self.locals is not None and name in self.locals

    def var_get(self, name):
        if self.is_local(name):
            return f"L_{name}"
        if name == "NF":
            return "rt.get_nf()"
        return f"G.get({name!r}, U)"

    def var_set(self, name, val):
        if self.is_local(name):
            return f"_setl(L, {name!r}, {val})" if False else f"_assign_local(locals_box, {name!r}, {val})"
        if name == "NF":
            return f"rt.set_nf({val})"
        if name in ("FS", "RS", "OFS", "ORS", "SUBSEP", "CONVFMT", "OFMT", "IGNORECASE"):
            return f"rt.set_special({name!r}, {val})"
        return f"_gset(G, {name!r}, {val})"

    def array_ref(self, name):
        if self.is_local(name):
            return f"L_{name}"
        return f"rt.arr({name!r})"

    def subscript(self, idx):
        if len(idx) == 1:
            return f"_key({self.expr(idx[0])}, rt)"
        return "rt.subsep().join([" + ", ".join(f"_key({self.expr(e)}, rt)" for e in idx) + "])"

    # ---------------- expressions

    def regex_obj(self, pat: str):
        self.regexes.append(pat)
        return f"R[{len(self.regexes) - 1}]"

    def match_expr(self, e):
        """An expression used as a regex (literal or dynamic)."""
        if e[0] == "regex":
            return self.regex_obj(e[1])
        return f"rt.dyn_re({self.expr(e)})"

    def expr(self, e) -> str:
        k = e[0]
        if k == "num":
            return repr(float(e[1]))
        if k == "str":
            return repr(e[1])
        if k == "regex":
            return f"(1.0 if {self.regex_obj(e[1])}.search(rt.get0()) else 0.0)"
        if k == "var":
            return self.var_get(e[1])
        if k == "field":
            return f"rt.getf({self.expr(e[1])})"
        if k == "index":
            return f"_aget({self.array_ref(e[1])}, {self.subscript(e[2])})"
        if k == "paren":
            return self.expr(e[1])
        if k == "group":
            return "rt.subsep().join([" + ", ".join(f"_key({self.expr(x)}, rt)" for x in e[1]) + "])"
        if k == "assign":
            return self.assign(e[1], e[2], e[3])
        if k in ("preinc", "postinc"):
            return self.incr(e[2], e[1], k == "postinc")
        if k == "?:":
            return f"({self.expr(e[2])} if _truthy({self.expr(e[1])}) else {self.expr(e[3])})"
        if k == "||":
            return f"(1.0 if (_truthy({self.expr(e[1])}) or _truthy({self.expr(e[2])})) else 0.0)"
        if k == "&&":
            return f"(1.0 if (_truthy({self.expr(e[1])}) and _truthy({self.expr(e[2])})) else 0.0)"
        if k == "!":
            return f"(0.0 if _truthy({self.expr(e[1])}) else 1.0)"
        if k == "in":
            return f"(1.0 if {self.subscript(e[1])} in {self.array_ref(e[2])} else 0.0)"
        if k in ("~", "!~"):
            m = f"{self.match_expr(e[2])}.search(_str({self.expr(e[1])}, rt))"
            return f"(1.0 if {m} else 0.0)" if k == "~" else f"(0.0 if {m} else 1.0)"
        if k == "cmp":
            return f"_cmp({e[1]!r}, {self.expr(e[2])}, {self.expr(e[3])}, rt)"
        if k == "concat":
            parts = []

            def flat(x):
                if x[0] == "concat":
                    flat(x[1])
                    flat(x[2])
                else:
                    parts.append(x)
            flat(e)
            return "(" + " + ".join(f"_str({self.expr(p)}, rt)" for p in parts) + ")"
        if k in ("+", "-", "*"):
            return f"(_num({self.expr(e[1])}) {k} _num({self.expr(e[2])}))"
        if k == "/":
            return f"_div(_num({self.expr(e[1])}), _num({self.expr(e[2])}))"
        if k == "%":
            return f"_mod(_num({self.expr(e[1])}), _num({self.expr(e[2])}))"
        if k == "^":
            return f"_pow(_num({self.expr(e[1])}), _num({self.expr(e[2])}))"
        if k == "neg":
            return f"(-_num({self.expr(e[1])}))"
        if k == "pos":
            return f"(_num({self.expr(e[1])}))"
        if k == "call":
            return self.call(e[1], e[2])
        if k == "builtin":
            return self.builtin(e[1], e[2])
        if k == "getline":
            var, src = e[1], e[2]
            setter = self.setter_lambda(var) if var is not None else "None"
            if src is None:
                return f"rt.getline_main({setter})"
            return f"rt.getline_file(_str({self.expr(src)}, rt), {setter})"
        if k == "cmdgetline":
            setter = self.setter_lambda(e[2]) if e[2] is not None else "None"
            return self.yf(f"rt.getline_cmd(_str({self.expr(e[1])}, rt), {setter})")
        raise AwkSyntaxError(f"cannot compile {k}")

    def setter_lambda(self, lv):
        return f"(lambda _v: {self.store(lv, '_v')})"

    def store(self, lv, val):
        k = lv[0]
        if k == "var":
            return self.var_set(lv[1], val)
        if k == "field":
            return f"rt.setf({self.expr(lv[1])}, {val})"
        if k == "index":
            return f"_aset({self.array_ref(lv[1])}, {self.subscript(lv[2])}, {val})"
        raise AwkSyntaxError("assignment to non-lvalue")

    def assign(self, op, lv, rhs):
        r = self.expr(rhs)
        if op == "=":
            if rhs[0] in ("var", "field", "index", "getline") or True:
                r = f"_copy({r})"
            return self.store(lv, r)
        cur = self.expr(lv)
        a, b = f"_num({cur})", f"_num({r})"
        expr = {"+=": f"({a} + {b})", "-=": f"({a} - {b})", "*=": f"({a} * {b})", "/=": f"_div({a}, {b})",
                "%=": f"_mod({a}, {b})", "^=": f"_pow({a}, {b})", "**=": f"_pow({a}, {b})"}[op]
        if lv[0] == "index":
            arr = self.array_ref(lv[1])
            key = self.subscript(lv[2])
            return f"_aop({arr}, {key}, lambda _c: {expr.replace(cur, '_c')})"
        return self.store(lv, expr)

    def incr(self, lv, delta, post):
        if lv[0] == "index":
            return f"_ainc({self.array_ref(lv[1])}, {self.subscript(lv[2])}, {float(delta)}, {post})"
        if lv[0] == "var":
            name = lv[1]
            if self.is_local(name):
                return f"_linc(locals_box, {name!r}, {float(delta)}, {post})"
            if name == "NF":
                return f"rt.inc_nf({float(delta)}, {post})"
            return f"_ginc(G, {name!r}, {float(delta)}, {post})"
        return f"rt.incf({self.expr(lv[1])}, {float(delta)}, {post})"

    def call(self, name, args):
        if name not in self.p.funcs:
            raise AwkFatal(f"function `{name}' not defined")
        params = self.p.funcs[name][0]
        if len(args) > len(params):
            raise AwkFatal(f"function `{name}' called with more arguments than declared")
        arrs = self.array_params.get(name, set())
        out = []
        for k, a in enumerate(args):
            pname = params[k]
            if pname in arrs:
                if a[0] != "var":
                    raise AwkFatal(f"attempt to use scalar parameter `{pname}' as an array")
                out.append(self.array_ref(a[1]) if not self.is_local(a[1]) else f"_ensure_arr(locals_box, {a[1]!r})")
            else:
                out.append(self.expr(a))
        return self.yf(f"F_{name}(rt, G, {', '.join(out)}{',' if out else ''})")

    def builtin(self, name, args):
        ex = [self.expr(a) for a in args]
        n = len(args)
        if name == "length":
            if n == 0:
                return "float(len(rt.get0()))"
            a = args[0]
            if a[0] == "var" and (a[1] in self.global_arrays and not self.is_local(a[1]) or
                                  (self.is_local(a[1]) and self.locals_arrays and a[1] in self.locals_arrays)):
                return f"float(len({self.array_ref(a[1])}))"
            return f"float(len(_str({ex[0]}, rt)))"
        if name == "substr":
            return f"_substr(_str({ex[0]}, rt), _num({ex[1]}){', _num(' + ex[2] + ')' if n > 2 else ''})"
        if name == "index":
            return f"float(_str({ex[0]}, rt).find(_str({ex[1]}, rt)) + 1)"
        if name == "split":
            arr = self.array_ref(args[1][1]) if args[1][0] == "var" else None
            if arr is None:
                raise AwkFatal("split: second argument is not an array")
            sep = ("None" if n < 3 else (self.regex_obj(args[2][1]) if args[2][0] == "regex" else ex[2]))
            seps = self.array_ref(args[3][1]) if n > 3 else "None"
            return f"rt.split(_str({ex[0]}, rt), {arr}, {sep}, {seps})"
        if name in ("sub", "gsub"):
            rx = self.match_expr(args[0])
            target = args[2] if n > 2 else ("field", ("num", 0.0))
            cur = self.expr(target)
            t = self.new_tmp()
            setter = self.setter_lambda(target)
            return f"rt.subst({rx}, _str({ex[1]}, rt), {cur}, {setter}, {name == 'gsub'})"
        if name == "gensub":
            rx = self.match_expr(args[0])
            target = ex[3] if n > 3 else "rt.get0()"
            return f"rt.gensub({rx}, _str({ex[1]}, rt), {ex[2]}, _str({target}, rt))"
        if name == "match":
            rx = self.match_expr(args[1])
            arr = self.array_ref(args[2][1]) if n > 2 else "None"
            return f"rt.match(_str({ex[0]}, rt), {rx}, {arr})"
        if name == "sprintf":
            return f"rt.sprintf(_str({ex[0]}, rt), [{', '.join(ex[1:])}])"
        if name in ("sin", "cos", "exp", "sqrt", "log"):
            return f"_math_{name}(_num({ex[0]}))"
        if name == "atan2":
            return f"math.atan2(_num({ex[0]}), _num({ex[1]}))"
        if name == "int":
            return f"float(int(_num({ex[0]})))"
        if name == "rand":
            return "rt.rand()"
        if name == "srand":
            return f"rt.srand({ex[0] if n else 'None'})"
        if name == "tolower":
            return f"_str({ex[0]}, rt).lower()"
        if name == "toupper":
            return f"_str({ex[0]}, rt).upper()"
        if name == "system":
            return self.yf(f"rt.system(_str({ex[0]}, rt))")
        if name == "close":
            return self.yf(f"rt.close(_str({ex[0]}, rt))")
        if name == "fflush":
            return self.yf("rt.fflush()")
        if name == "systime":
            return "rt.systime()"
        if name == "strftime":
            return f"rt.strftime({', '.join(ex)})"
        if name == "mktime":
            return f"rt.mktime(_str({ex[0]}, rt))"
        if name in ("asort", "asorti"):
            src = self.array_ref(args[0][1])
            dst = self.array_ref(args[1][1]) if n > 1 else "None"
            return f"rt.{name}({src}, {dst})"
        if name in ("and", "or", "xor"):
            op = {"and": "&", "or": "|", "xor": "^"}[name]
            return "float(" + f" {op} ".join(f"int(_num({x}))" for x in ex) + ")"
        if name == "lshift":
            return f"float(int(_num({ex[0]})) << int(_num({ex[1]})))"
        if name == "rshift":
            return f"float(int(_num({ex[0]})) >> int(_num({ex[1]})))"
        if name == "compl":
            return f"float(~int(_num({ex[0]})) & ((1 << 53) - 1))"
        if name == "strtonum":
            return f"_strtonum({ex[0]})"
        if name == "typeof":
            return f"_typeof({ex[0]})"
        if name == "isarray":
            return f"(1.0 if isinstance({ex[0]}, dict) else 0.0)"
        raise AwkFatal(f"function `{name}' not supported")

    locals_arrays: set | None = None

    # ---------------- statements

    def stmts(self, node, ind):
        k = node[0]
        if k == "block":
            if not node[1]:
                self.emit("pass", ind)
            for s in node[1]:
                self.stmts(s, ind)
            return
        if k == "expr":
            self.emit(self.expr(node[1]), ind)
            return
        if k in ("print", "printf"):
            args = [self.expr(a) for a in node[1]]
            dest = node[2]
            dest_s = "None" if dest is None else f"({dest[0]!r}, _str({self.expr(dest[1])}, rt))"
            if k == "print":
                call = f"rt.do_print([{', '.join(args)}], {dest_s})"
            else:
                call = f"rt.do_printf(_str({args[0]}, rt), [{', '.join(args[1:])}], {dest_s})"
            self.emit(call, ind)
            return
        if k == "if":
            self.emit(f"if _truthy({self.expr(node[1])}):", ind)
            self.stmts(node[2], ind + 1)
            if node[3] is not None:
                self.emit("else:", ind)
                self.stmts(node[3], ind + 1)
            return
        if k == "while":
            self.emit(f"while _truthy({self.expr(node[1])}):", ind)
            self.emit("rt.tick()", ind + 1)
            self.loop_depth += 1
            self.stmts(node[2], ind + 1)
            self.loop_depth -= 1
            return
        if k == "do":
            self.emit("while True:", ind)
            self.emit("rt.tick()", ind + 1)
            self.loop_depth += 1
            # continue must still test the condition: wrap the body in a one-shot loop
            self.emit("for _once in (0,):", ind + 1)
            self.stmts(node[1], ind + 2)
            self.emit("else:", ind + 1)
            self.emit("pass", ind + 2)
            self.emit(f"if not _truthy({self.expr(node[2])}):", ind + 1)
            self.emit("break", ind + 2)
            self.loop_depth -= 1
            return
        if k == "for":
            init, cond, step, body = node[1:]
            if init is not None:
                self.emit(self.expr(init), ind)
            self.emit("while True:", ind)
            self.emit("rt.tick()", ind + 1)
            if cond is not None:
                self.emit(f"if not _truthy({self.expr(cond)}):", ind + 1)
                self.emit("break", ind + 2)
            self.loop_depth += 1
            # `continue` jumps to the step: body inside a one-shot loop whose `continue` = break
            self.emit("for _once in (0,):", ind + 1)
            self._in_for_body = getattr(self, "_in_for_body", 0) + 1
            self.stmts(body, ind + 2)
            self._in_for_body -= 1
            self.emit("else:", ind + 1)
            self.emit("pass", ind + 2)
            self.emit("if rt.brk:", ind + 1)
            self.emit("rt.brk = False", ind + 2)
            self.emit("break", ind + 2)
            if step is not None:
                self.emit(self.expr(step), ind + 1)
            self.loop_depth -= 1
            return
        if k == "forin":
            var, arr, body = node[1:]
            t = self.new_tmp()
            self.emit(f"for {t} in rt.keys({self.array_ref(arr)}):", ind)
            self.emit(self.store(("var", var), f"StrNum({t})") if not self.is_local(var) else
                      f"_assign_local(locals_box, {var!r}, StrNum({t}))", ind + 1)
            if self.is_local(var):
                self.emit(f"L_{var} = locals_box[{var!r}]", ind + 1)
            self.loop_depth += 1
            saved = getattr(self, "_in_for_body", 0)
            self._in_for_body = 0
            self.stmts(body, ind + 1)
            self._in_for_body = saved
            self.loop_depth -= 1
            return
        if k == "break":
            if getattr(self, "_in_for_body", 0):
                self.emit("rt.brk = True", ind)
            self.emit("break", ind)
            return
        if k == "continue":
            self.emit("break" if getattr(self, "_in_for_body", 0) else "continue", ind)
            return
        if k == "next":
            self.emit("raise NextRecord()", ind)
            return
        if k == "nextfile":
            self.emit("raise NextFile()", ind)
            return
        if k == "exit":
            self.emit(f"raise AwkExit({self.expr(node[1]) if node[1] is not None else 'None'})", ind)
            return
        if k == "return":
            self.emit(f"return {self.expr(node[1]) if node[1] is not None else 'U'}", ind)
            return
        if k == "delete":
            if node[2] is None:
                self.emit(f"{self.array_ref(node[1])}.clear()", ind)
            else:
                self.emit(f"{self.array_ref(node[1])}.pop({self.subscript(node[2])}, None)", ind)
            return
        raise AwkSyntaxError(f"cannot compile statement {k}")

    def function(self, name, params, body):
        arrs = self.array_params.get(name, set())
        self.locals = set(params)
        self.locals_arrays = arrs
        self.emit(f"def F_{name}(rt, G, {', '.join('L_' + p + '=None' for p in params)}):", 0)
        for p in params:
            if p in arrs:
                self.emit(f"if L_{p} is None: L_{p} = {{}}", 1)
            else:
                self.emit(f"if L_{p} is None: L_{p} = U", 1)
        self.emit("locals_box = {" + ", ".join(f"{p!r}: L_{p}" for p in params) + "}", 1)
        # local scalars are read through Python locals; writes go through locals_box then refresh
        self._func_body(body, 1)
        self.emit("return U", 1)
        if self.gen_mode:
            self.emit("yield", 1)
        self.locals = None
        self.locals_arrays = None

    def _func_body(self, body, ind):
        start = len(self.lines)
        self.stmts(body, ind)
        # rewrite local reads/writes: L_x reads must see updates made via locals_box
        for i in range(start, len(self.lines)):
            line = self.lines[i]
            for p in self.locals:
                if p not in (self.locals_arrays or ()):
                    line = re.sub(rf"\bL_{p}\b(?!\s*=[^=])", f"locals_box[{p!r}]", line)
            self.lines[i] = line


def compile_program(src: str):
    p = Parser(src).parse()
    gen_mode = p.uses_cmd
    g = Gen(p, gen_mode)
    for name, (params, body) in p.funcs.items():
        g.function(name, params, body)
    rules = []
    for idx, (kind, pat, pat2, body) in enumerate(p.items):
        fname = f"rule_{idx}"
        g.emit(f"def {fname}(rt, G):", 0)
        if kind == "main":
            if pat2 is not None:
                g.emit(f"if not rt.in_range({idx}, lambda: _truthy({g.expr(pat)}), lambda: _truthy({g.expr(pat2)})):", 1)
                g.emit("return", 2)
            elif pat is not None:
                g.emit(f"if not _truthy({g.expr(pat)}):", 1)
                g.emit("return", 2)
        if body is None:
            g.emit("rt.do_print([], None)", 1)
        else:
            g.stmts(body, 1)
        if gen_mode:
            g.emit("return", 1)
            g.emit("yield", 1)
        rules.append((kind, fname))
    code = "\n".join(g.lines) + "\n"
    ns = dict(RUNTIME_NS)
    ns["R"] = [compile_ere(x) for x in g.regexes]
    try:
        exec(compile(code, "<awk>", "exec"), ns)
    except SyntaxError as e:  # a bug in the generator: report the generated code
        raise AwkFatal(f"internal compiler error: {e}\n{code}") from None
    return p, [(k, ns[f]) for k, f in rules], gen_mode, code


def compile_ere(pat: str, icase=False):
    try:
        return re.compile(translate(pat, "ere", sed=True), re.S | (re.I if icase else 0))
    except (re.error, RegexError):
        raise AwkFatal(f"invalid regexp: /{pat}/") from None


# ---------------------------------------------------------------------------- runtime helpers

def _num(v):
    return v if type(v) is float else to_num(v)


def _str(v, rt):
    if type(v) is float:
        return fmt_num(v, rt.convfmt)
    return v


def _key(v, rt):
    if type(v) is float:
        return fmt_num(v, rt.convfmt)
    return str(v)


def _truthy(v):
    return truthy(v)


def _copy(v):
    return v


def _div(a, b):
    if b == 0:
        raise AwkFatal("division by zero attempted")
    return a / b


def _mod(a, b):
    if b == 0:
        raise AwkFatal("division by zero attempted in `%'")
    return math.fmod(a, b)


def _pow(a, b):
    try:
        return float(a ** b) if not (a < 0 and b != int(b)) else float("nan")
    except OverflowError:
        return float("inf")
    except ZeroDivisionError:
        return float("inf")


def _cmp(op, a, b, rt):
    if is_numeric(a) and is_numeric(b):
        x, y = _num(a), _num(b)
    else:
        x, y = _str(a, rt), _str(b, rt)
    if op == "<":
        r = x < y
    elif op == "<=":
        r = x <= y
    elif op == ">":
        r = x > y
    elif op == ">=":
        r = x >= y
    elif op == "==":
        r = x == y
    else:
        r = x != y
    return 1.0 if r else 0.0


def _gset(G, name, v):
    G[name] = v
    return v


def _ginc(G, name, delta, post):
    old = _num(G.get(name, UNINIT))
    G[name] = old + delta
    return old if post else old + delta


def _assign_local(box, name, v):
    box[name] = v
    return v


def _linc(box, name, delta, post):
    old = _num(box[name])
    box[name] = old + delta
    return old if post else old + delta


def _ensure_arr(box, name):
    v = box.get(name)
    if not isinstance(v, dict):
        v = {}
        box[name] = v
    return v


def _aget(arr, key):
    v = arr.get(key)
    if v is None:
        arr[key] = UNINIT
        return UNINIT
    return v


def _aset(arr, key, v):
    arr[key] = v
    return v


def _aop(arr, key, fn):
    v = fn(_num(arr.get(key, UNINIT)))
    arr[key] = v
    return v


def _ainc(arr, key, delta, post):
    old = _num(arr.get(key, UNINIT))
    arr[key] = old + delta
    return old if post else old + delta


def _substr(s, m, n=None):
    # gawk: start and length are truncated toward zero; a start below 1 counts as 1 (the length is not reduced)
    def trunc(x):
        if x != x:
            return 0
        if x in (math.inf, -math.inf):
            return x
        return math.trunc(x)
    start = trunc(m)
    if start < 1:
        start = 1
    if start == math.inf:
        return ""
    if n is None:
        return s[int(start) - 1:]
    ln = trunc(n)
    if ln <= 0:
        return ""
    if ln == math.inf:
        return s[int(start) - 1:]
    return s[int(start) - 1:int(start) - 1 + int(ln)]


def _math_sqrt(x):
    return math.sqrt(x) if x >= 0 else float("nan")


def _math_log(x):
    if x == 0:
        return float("-inf")
    return math.log(x) if x > 0 else float("nan")


def _math_exp(x):
    try:
        return math.exp(x)
    except OverflowError:
        return float("inf")


def _math_sin(x):
    return math.sin(x)


def _math_cos(x):
    return math.cos(x)


def _strtonum(v):
    if type(v) is float:
        return v
    s = str(v).strip()
    try:
        if s.lower().startswith("0x"):
            return float(int(s, 16))
        if len(s) > 1 and s.startswith("0") and s.isdigit():
            return float(int(s, 8))
    except ValueError:
        pass
    return to_num(s)


def _typeof(v):
    if isinstance(v, dict):
        return "array"
    if type(v) is Uninit:
        return "untyped"
    if type(v) is float:
        return "number"
    if type(v) is StrNum:
        return "strnum" if v.isnum else "string"
    return "string"


RUNTIME_NS = {k: v for k, v in globals().items() if k.startswith("_") and callable(v)}
RUNTIME_NS.update({"U": UNINIT, "StrNum": StrNum, "math": math, "NextRecord": NextRecord, "NextFile": NextFile,
                   "AwkExit": AwkExit, "AwkFatal": AwkFatal})


def awk_printf(fmt: str, args, rt) -> str:
    out = []
    i, n = 0, len(fmt)
    ai = 0
    while i < n:
        c = fmt[i]
        if c != "%":
            out.append(c)
            i += 1
            continue
        m = re.match(r"%([-+ #0]*)(\*|\d+)?(?:\.(\*|\d*))?([cdiouxXeEfFgGs%])", fmt[i:])
        if not m:
            out.append(c)
            i += 1
            continue
        i += m.end()
        flags, width, prec, conv = m.groups()
        if conv == "%":
            out.append("%")
            continue
        if width == "*":
            width = str(int(_num(args[ai]))) if ai < len(args) else ""
            ai += 1
            if width.startswith("-"):
                flags += "-"
                width = width[1:]
        if prec == "*":
            prec = str(max(int(_num(args[ai])), 0)) if ai < len(args) else ""
            ai += 1
        if ai >= len(args):
            pointer = " " * (i - 1)
            raise AwkFatal(f"not enough arguments to satisfy format string\n\t`{fmt}'\n\t{pointer}^ ran out for this one")
        arg = args[ai]
        ai += 1
        w = width or ""
        p = ("." + prec) if prec is not None else ""
        if conv == "c":
            if type(arg) is float:
                ch = chr(int(arg) % 0x110000) if int(arg) >= 0 else ""
            else:
                ch = str(arg)[:1]
            out.append(("%" + flags.replace("0", "") + w + "s") % ch)
        elif conv in "di":
            v = _num(arg)
            if v != v or v in (math.inf, -math.inf):
                out.append(("%" + flags.replace("0", "") + w + "s") % fmt_num(v, "%.6g"))
            else:
                out.append(("%" + flags + w + p + "d") % int(v))
        elif conv in "ouxX":
            v = int(_num(arg))
            if v < 0:
                v += 1 << 64
            s = ("%" + flags.replace("#", "") + w + p + {"o": "o", "u": "d", "x": "x", "X": "X"}[conv]) % v
            if "#" in flags and conv in "xX" and v:
                s = ("%" + flags + w + p + conv) % v
            elif "#" in flags and conv == "o":
                s = ("%" + flags.replace("#", "") + w + "s") % ("0" + "%o" % v)
            out.append(s)
        elif conv in "eEfFgG":
            v = _num(arg)
            if v != v or v in (math.inf, -math.inf):
                out.append(("%" + flags.replace("0", "") + w + "s") % fmt_num(v, "%.6g"))
            else:
                out.append(("%" + flags + w + p + conv) % v)
        else:
            s = _str(arg, rt) if not isinstance(arg, dict) else ""
            out.append(("%" + flags.replace("0", "") + w + p + "s") % s)
    return "".join(out)


class Runtime:
    def __init__(self, ctx, G, rules, gen_mode, parser):
        self.ctx, self.G, self.gen_mode = ctx, G, gen_mode
        self.fields = [""]
        self.split_done = True
        self.nf = 0
        self.convfmt = "%.6g"
        self.ofmt = "%.6g"
        self.fs = " "
        self.rs = "\n"
        self.ofs = " "
        self.ors = "\n"
        self._subsep = "\x1c"
        self.out_files: dict = {}
        self.cmd_out: dict = {}
        self.read_files: dict = {}
        self.cmd_in: dict = {}
        self.ranges: dict = {}
        self.records = None
        self.rng = random.Random(0)
        self.seed = 0.0
        self.prev_seed = 0.0
        self.brk = False
        self.icase = False
        self.steps = 0
        self.dyn_cache: dict = {}
        self.parser = parser

    # ---- special variables
    def set_special(self, name, v):
        self.G[name] = v
        s = _str(v, self)
        if name == "FS":
            self.fs = s
        elif name == "RS":
            self.rs = s
        elif name == "OFS":
            self.ofs = s
        elif name == "ORS":
            self.ors = s
        elif name == "SUBSEP":
            self._subsep = s
        elif name == "CONVFMT":
            self.convfmt = s
        elif name == "OFMT":
            self.ofmt = s
        elif name == "IGNORECASE":
            self.icase = truthy(v)
            self.dyn_cache.clear()
        return v

    def subsep(self):
        return self._subsep

    def tick(self):
        self.steps += 1
        if self.steps & 0x3FFF == 0:
            k = self.ctx.k
            k.steps += 0x4000 // 8
            if k.steps > k.budget:
                from ..kernel import TimeLimit
                raise TimeLimit()

    # ---- records and fields
    def set_record(self, text):
        self.fields = [StrNum(text)]
        self.split_done = False

    def get0(self):
        return self.fields[0]

    def _split(self):
        if self.split_done:
            return
        rec = self.fields[0]
        parts = self.split_fields(rec, self.fs, paragraph=self.rs == "")
        self.fields = [self.fields[0]] + [StrNum(x) for x in parts]
        self.nf = len(parts)
        self.split_done = True

    def split_fields(self, rec, fs, paragraph=False):
        if fs == " ":
            return rec.split()
        if paragraph and fs != " ":
            out = []
            for line in rec.split("\n"):
                out += self.split_fields(line, fs)
            return out
        if rec == "":
            return []
        if fs == "":
            return list(rec)
        if len(fs) == 1 and fs != "\\":
            if self.icase and fs.isalpha():
                return re.split(re.escape(fs), rec, flags=re.I)
            return rec.split(fs)
        rx = self.dyn_re(fs)
        return rx.split(rec) if rx.groups == 0 else [x for x in re.split(f"(?:{rx.pattern})", rec)]

    def getf(self, i):
        i = int(_num(i))
        if i == 0:
            return self.fields[0]
        if i < 0:
            raise AwkFatal(f"attempt to access field {i}")
        self._split()
        if i <= self.nf:
            return self.fields[i]
        return UNINIT

    def setf(self, i, v):
        i = int(_num(i))
        if i < 0:
            raise AwkFatal(f"attempt to access field {i}")
        if i == 0:
            self.fields = [StrNum(_str(v, self)) if type(v) is not str else v]
            self.fields[0] = v if isinstance(v, str) else _str(v, self)
            self.split_done = False
            return v
        self._split()
        while self.nf < i:
            self.fields.append(UNINIT)
            self.nf += 1
        self.fields[i] = v
        self._rebuild()
        return v

    def incf(self, i, delta, post):
        old = _num(self.getf(i))
        self.setf(i, old + delta)
        return old if post else old + delta

    def _rebuild(self):
        self.fields[0] = self.ofs.join(_str(x, self) for x in self.fields[1:self.nf + 1])
        self.split_done = True

    def get_nf(self):
        self._split()
        return float(self.nf)

    def set_nf(self, v):
        self._split()
        n = int(_num(v))
        if n < 0:
            raise AwkFatal(f"NF set to negative value")
        if n < self.nf:
            self.fields = self.fields[:n + 1]
        else:
            self.fields += [UNINIT] * (n - self.nf)
        self.nf = n
        self._rebuild()
        return v

    def inc_nf(self, delta, post):
        old = self.get_nf()
        self.set_nf(old + delta)
        return old if post else old + delta

    def arr(self, name):
        v = self.G.get(name)
        if not isinstance(v, dict):
            if v is not None and type(v) is not Uninit:
                raise AwkFatal(f"attempt to use scalar `{name}' as an array")
            v = {}
            self.G[name] = v
        return v

    def keys(self, d):
        ks = list(d)
        order = self.G.get("PROCINFO", {}).get("sorted_in") if isinstance(self.G.get("PROCINFO"), dict) else None
        if order:
            o = str(order)
            num = lambda k: to_num(k)  # noqa: E731
            if o == "@ind_str_asc":
                ks.sort()
            elif o == "@ind_str_desc":
                ks.sort(reverse=True)
            elif o == "@ind_num_asc":
                ks.sort(key=num)
            elif o == "@ind_num_desc":
                ks.sort(key=num, reverse=True)
            elif o == "@val_num_asc":
                ks.sort(key=lambda k: (_num(d[k]), k))
            elif o == "@val_num_desc":
                ks.sort(key=lambda k: (-_num(d[k]), k))
            elif o == "@val_str_asc":
                ks.sort(key=lambda k: (_str(d[k], self), k))
            elif o == "@val_str_desc":
                ks.sort(key=lambda k: (_str(d[k], self), k), reverse=True)
        return ks

    def dyn_re(self, v):
        pat = _str(v, self)
        key = (pat, self.icase)
        rx = self.dyn_cache.get(key)
        if rx is None:
            rx = compile_ere(pat, self.icase)
            self.dyn_cache[key] = rx
        return rx

    def in_range(self, idx, start, end):
        active = self.ranges.get(idx, False)
        if not active:
            if start():
                if not end():
                    self.ranges[idx] = True
                return True
            return False
        if end():
            self.ranges[idx] = False
        return True

    # ---- builtins
    def split(self, s, arr, sep, seps):
        arr.clear()
        if sep is None:
            parts = self.split_fields(s, self.fs)
        elif isinstance(sep, re.Pattern):
            parts = sep.split(s) if s else []
        else:
            sp = _str(sep, self)
            parts = self.split_fields(s, sp) if s else []
        for i, p in enumerate(parts, 1):
            arr[str(i)] = StrNum(p)
        return float(len(parts))

    def subst(self, rx, repl, cur, setter, glob):
        s = _str(cur, self)
        count = 0

        def rep(m):
            nonlocal count
            count += 1
            out = []
            i = 0
            while i < len(repl):
                c = repl[i]
                if c == "\\" and i + 1 < len(repl) and repl[i + 1] in "&\\":
                    out.append(repl[i + 1])
                    i += 2
                    continue
                if c == "&":
                    out.append(m.group(0))
                else:
                    out.append(c)
                i += 1
            return "".join(out)
        new = rx.sub(rep, s, count=0 if glob else 1)
        if count:
            setter(new)
        return float(count)

    def gensub(self, rx, repl, how, target):
        h = _str(how, self)
        glob = h[:1] in ("g", "G")
        nth = 0 if glob else max(int(_num(how)) if not glob else 0, 1)
        count = 0

        def rep(m):
            nonlocal count
            count += 1
            if not glob and count != nth:
                return m.group(0)
            out = []
            i = 0
            while i < len(repl):
                c = repl[i]
                if c == "\\" and i + 1 < len(repl):
                    d = repl[i + 1]
                    if d.isdigit():
                        g = int(d)
                        out.append((m.group(g) or "") if g <= (rx.groups or 0) else "")
                    elif d == "&":
                        out.append("&")
                    else:
                        out.append(d)
                    i += 2
                    continue
                out.append(m.group(0) if c == "&" else c)
                i += 1
            return "".join(out)
        return rx.sub(rep, target)

    def match(self, s, rx, arr):
        m = rx.search(s)
        if arr is not None:
            arr.clear()
        if not m:
            self.G["RSTART"], self.G["RLENGTH"] = 0.0, -1.0
            return 0.0
        self.G["RSTART"] = float(m.start() + 1)
        self.G["RLENGTH"] = float(m.end() - m.start())
        if arr is not None:
            arr["0"] = StrNum(m.group(0))
            arr["0" + self._subsep + "start"] = float(m.start() + 1)
            arr["0" + self._subsep + "length"] = float(m.end() - m.start())
            for gi in range(1, (rx.groups or 0) + 1):
                if m.group(gi) is not None:
                    arr[str(gi)] = StrNum(m.group(gi))
                    arr[str(gi) + self._subsep + "start"] = float(m.start(gi) + 1)
                    arr[str(gi) + self._subsep + "length"] = float(m.end(gi) - m.start(gi))
        return float(m.start() + 1)

    def sprintf(self, fmt, args):
        return awk_printf(fmt, args, self)

    def rand(self):
        return self.rng.random()

    def srand(self, seed):
        prev = self.seed
        if seed is None:
            self.seed = float(self.ctx.now // 1_000_000_000)
        else:
            self.seed = _num(seed)
        self.rng = random.Random(self.seed)
        return prev

    def systime(self):
        return float(self.ctx.now // 1_000_000_000)

    def strftime(self, fmt=None, ts=None, utc=None):
        from ..timeutil import local_dt, strftime
        f = _str(fmt, self) if fmt is not None else "%a %b %e %H:%M:%S %Z %Y"
        t = int(_num(ts)) if ts is not None else self.ctx.now // 1_000_000_000
        tz = "UTC" if (utc is not None and truthy(utc)) else self.ctx.tz
        return strftime(f, local_dt(t * 1_000_000_000, tz))

    def mktime(self, spec):
        from datetime import datetime
        from ..timeutil import localize, to_ns
        parts = spec.split()
        try:
            y, mo, d, h, mi, s = (int(x) for x in parts[:6])
            return float(to_ns(localize(datetime(y, mo, d, h, mi, s), self.ctx.tz)) // 1_000_000_000)
        except (ValueError, IndexError):
            return -1.0

    def asort(self, src, dst):
        vals = sorted(src.values(), key=lambda v: (0, _num(v), "") if is_numeric(v) else (1, 0, _str(v, self)))
        target = dst if dst is not None else src
        target.clear()
        for i, v in enumerate(vals, 1):
            target[str(i)] = v
        return float(len(vals))

    def asorti(self, src, dst):
        ks = sorted(src)
        target = dst if dst is not None else src
        target.clear()
        for i, k in enumerate(ks, 1):
            target[str(i)] = StrNum(k)
        return float(len(ks))

    # ---- output
    def write_to(self, dest, text):
        if dest is None:
            self.ctx.w(text)
            return
        op, name = dest
        if op in ("|", "|&"):
            self.cmd_out.setdefault(name, []).append(text)
            return
        if name in ("/dev/stdout", "-"):
            self.ctx.w(text)
            return
        if name == "/dev/stderr":
            self.ctx.e(text)
            return
        of = self.out_files.get(name)
        if of is None:
            try:
                of = self.ctx.open(name, "a" if op == ">>" else "w")
            except FSError as e:
                raise AwkFatal(f"can't redirect to `{name}' ({e.msg})") from None
            self.out_files[name] = of
        node = of.inode
        if node is not None:
            node.data += text.encode("utf-8", "surrogateescape")
            node.mtime = self.ctx.now

    def do_print(self, args, dest):
        if not args:
            text = _str(self.get0(), self)
        else:
            text = self.ofs.join(fmt_num(a, self.ofmt) if type(a) is float else a for a in args)
        self.write_to(dest, text + self.ors)

    def do_printf(self, fmt, args, dest):
        self.write_to(dest, awk_printf(fmt, args, self))

    def flush_cmd(self, name):
        """Run a `print | cmd` pipe with everything written to it."""
        text = "".join(self.cmd_out.pop(name, []))
        yield from self.ctx.flush()
        from ..fileio import anon_file
        fds = dict(self.ctx.proc.fds)
        fds[0] = anon_file(text.encode("utf-8", "surrogateescape"))
        st = yield from self.ctx.run_shell(name, fds=fds)
        return float(st)

    def close(self, name):
        if name in self.cmd_out:
            return (yield from self.flush_cmd(name))
        if name in self.out_files:
            del self.out_files[name]
            return 0.0
        if name in self.read_files:
            del self.read_files[name]
            return 0.0
        if name in self.cmd_in:
            del self.cmd_in[name]
            return 0.0
        return -1.0

    def fflush(self):
        yield from self.ctx.flush()
        return 0.0

    def system(self, cmd):
        yield from self.ctx.flush()
        for name in list(self.cmd_out):
            yield from self.flush_cmd(name)
        st = yield from self.ctx.run_shell(cmd)
        return float(st)

    # ---- input
    def getline_main(self, setter):
        try:
            rec = next(self.records)
        except StopIteration:
            return 0.0
        self.G["NR"] = _num(self.G.get("NR", 0.0)) + 1
        self.G["FNR"] = _num(self.G.get("FNR", 0.0)) + 1
        if setter is None:
            self.set_record(rec)
        else:
            setter(StrNum(rec))
        return 1.0

    def getline_file(self, name, setter):
        st = self.read_files.get(name)
        if st is None:
            try:
                data = bytes(self.ctx.fs.lookup(name, self.ctx.cwd).data).decode("utf-8", "surrogateescape") \
                    if name not in ("-", "/dev/stdin") else ""
            except FSError:
                return -1.0
            st = iter(split_records(data, self.rs))
            self.read_files[name] = st
        try:
            rec = next(st)
        except StopIteration:
            return 0.0
        if setter is None:
            self.set_record(rec)
            self.G["NR"] = _num(self.G.get("NR", 0.0))
        else:
            setter(StrNum(rec))
        return 1.0

    def getline_cmd(self, cmd, setter):
        st = self.cmd_in.get(cmd)
        if st is None:
            yield from self.ctx.flush()
            from ..fileio import read_all
            from ..kernel import OpenFile, Pipe
            pipe = Pipe()
            w = OpenFile("pipe", wr=True, pipe=pipe)
            r = OpenFile("pipe", rd=True, pipe=pipe)
            fds = dict(self.ctx.proc.fds)
            fds[1] = w
            p = self.ctx.spawn(["sh", "-c", cmd], fds=fds)
            r.ref()
            data = yield from read_all(self.ctx.k, r)
            r.unref(self.ctx.k)
            if p is not None:
                yield from self.ctx.wait([p])
            st = iter(split_records(data.decode("utf-8", "surrogateescape"), self.rs))
            self.cmd_in[cmd] = st
        try:
            rec = next(st)
        except StopIteration:
            return 0.0
        self.G["NR"] = _num(self.G.get("NR", 0.0)) + 1
        if setter is None:
            self.set_record(rec)
        else:
            setter(StrNum(rec))
        return 1.0


def split_records(data: str, rs: str):
    if data == "":
        return []
    if rs == "\n":
        recs = data.split("\n")
        if recs and recs[-1] == "":
            recs.pop()
        return recs
    if rs == "":
        stripped = data.strip("\n")
        if not stripped:
            return []
        return re.split(r"\n\n+", stripped)
    if len(rs) == 1:
        recs = data.split(rs)
        if recs and recs[-1] in ("", "\n"):
            recs.pop()
        return recs
    recs = re.split(translate(rs, "ere", sed=True), data)
    if recs and recs[-1] in ("", "\n"):
        recs.pop()
    return recs


def _unescape_assign(v: str) -> str:
    out = []
    i = 0
    while i < len(v):
        c = v[i]
        if c == "\\" and i + 1 < len(v):
            d = v[i + 1]
            out.append({"n": "\n", "t": "\t", "\\": "\\", '"': '"', "/": "/", "a": "\a", "b": "\b", "f": "\f",
                        "r": "\r", "v": "\v"}.get(d, "\\" + d))
            i += 2
            continue
        out.append(c)
        i += 1
    return "".join(out)


AWK_USAGE = """Usage: awk [POSIX or GNU style options] -f progfile [--] file ...
Usage: awk [POSIX or GNU style options] [--] 'program' file ...
POSIX options:\t\tGNU long options: (standard)
\t-f progfile\t\t--file=progfile
\t-F fs\t\t\t--field-separator=fs
\t-v var=val\t\t--assign=var=val
"""


@util("gawk", "mawk", "nawk")
def u_awk(ctx):
    a = ctx.argv[1:]
    fs = None
    assigns = []
    progfiles = []
    i = 0
    while i < len(a):
        x = a[i]
        if x == "--":
            i += 1
            break
        if x == "-F" and i + 1 < len(a):
            fs = a[i + 1]
            i += 2
            continue
        if x.startswith("-F") and len(x) > 2:
            fs = x[2:]
            i += 1
            continue
        if x.startswith("--field-separator="):
            fs = x.split("=", 1)[1]
            i += 1
            continue
        if x == "-v" and i + 1 < len(a):
            assigns.append(a[i + 1])
            i += 2
            continue
        if x.startswith("-v") and len(x) > 2:
            assigns.append(x[2:])
            i += 1
            continue
        if x.startswith("--assign="):
            assigns.append(x.split("=", 1)[1])
            i += 1
            continue
        if x == "-f" and i + 1 < len(a):
            progfiles.append(a[i + 1])
            i += 2
            continue
        if x in ("--version", "-V"):
            ctx.w("GNU Awk 5.2.1, API 3.2, PMA Avon 8-g1, (GNU MPFR 4.2.1, GNU MP 6.3.0)\n")
            return 0
        if x in ("--help", "-h"):
            ctx.w(AWK_USAGE)
            return 0
        if x in ("--posix", "-P", "--traditional", "-c", "--re-interval", "-r", "--sandbox", "-S", "-b",
                 "--characters-as-bytes", "-n", "--non-decimal-data", "-e"):
            i += 1
            if x == "-e" and i < len(a):
                progfiles.append(("src", a[i]))
                i += 1
            continue
        if x.startswith("-") and len(x) > 1 and not progfiles:
            ctx.e(f"awk: not an option: {x}\n" if x.startswith("--") else f"awk: invalid option -- '{x[1]}'\n")
            ctx.e(AWK_USAGE)
            return 2
        break
    rest = a[i:]
    if progfiles:
        srcs = []
        for pf in progfiles:
            if isinstance(pf, tuple):
                srcs.append(pf[1])
                continue
            try:
                srcs.append((yield from ctx.text(pf)))
            except FSError as e:
                ctx.e(f"awk: fatal: can't open source file `{pf}' for reading: {e.msg}\n")
                return 2
        src = "\n".join(srcs)
    else:
        if not rest:
            ctx.e(AWK_USAGE)
            return 2
        src = rest[0]
        rest = rest[1:]
    try:
        parser, rules, gen_mode, code = compile_program(src)
    except AwkSyntaxError as e:
        where = "cmd. line" if not progfiles else progfiles[0]
        ctx.e(f"awk: {where}:{e.line}: {e.src_line}\nawk: {where}:{e.line}: {' ' * e.col}^ {e.msg}\n")
        return 1
    except AwkFatal as e:
        ctx.e(f"awk: cmd. line:1: fatal: {e}\n")
        return 2
    G = {"NR": 0.0, "FNR": 0.0, "NF": 0.0, "FS": " ", "OFS": " ", "ORS": "\n", "RS": "\n", "SUBSEP": "\x1c",
         "RSTART": 0.0, "RLENGTH": -1.0, "FILENAME": "", "CONVFMT": "%.6g", "OFMT": "%.6g"}
    G["ENVIRON"] = {k: StrNum(v) for k, v in ctx.env.items()}
    G["ARGV"] = {"0": "awk"}
    for k, x in enumerate(rest, 1):
        G["ARGV"][str(k)] = StrNum(x)
    G["ARGC"] = float(len(rest) + 1)
    G["PROCINFO"] = {"version": "5.2.1", "pid": float(ctx.proc.pid)}
    rt = Runtime(ctx, G, rules, gen_mode, parser)
    if fs is not None:
        rt.set_special("FS", "\t" if fs == "t" else _unescape_assign(fs))
    for asg in assigns:
        if "=" not in asg:
            ctx.e(f"awk: `{asg}' argument to `-v' not in `var=value' form\n")
            return 2
        k, v = asg.split("=", 1)
        val = StrNum(_unescape_assign(v))
        if k in ("FS", "RS", "OFS", "ORS", "SUBSEP", "CONVFMT", "OFMT", "IGNORECASE"):
            rt.set_special(k, val)
        else:
            G[k] = val
    status = 0
    try:
        status = yield from _drive(ctx, rt, rules, gen_mode)
    except AwkFatal as e:
        yield from ctx.flush()
        fnr = int(_num(G.get('FNR', 0.0)))
        where = f"(FILENAME={_str(G.get('FILENAME', ''), rt) or '-'} FNR={fnr}) " if fnr > 0 else ""
        ctx.e(f"awk: cmd. line:1: {where}fatal: {e}\n")
        return 2
    return status


def _prepare_records(ctx, rt):
    """Read every input named in ARGV (stdin if none) and return a record iterator (None on a fatal error)."""
    G = rt.G
    sources = []
    argc = int(_num(G.get("ARGC", 1.0)))
    argv = G.get("ARGV", {})
    any_file = False
    for k in range(1, argc):
        v = argv.get(str(k))
        if v is None or _str(v, rt) == "":
            continue
        s = _str(v, rt)
        m = re.match(r"([A-Za-z_][A-Za-z0-9_]*)=(.*)\Z", s, re.S)
        if m:
            sources.append(("assign", m.group(1), m.group(2)))
            continue
        any_file = True
        sources.append(("file", s))
    if not any_file:
        sources.append(("file", "-"))
    datas = []
    for src in sources:
        if src[0] == "file":
            name = src[1]
            try:
                if name in ("-", "/dev/stdin"):
                    data = yield from ctx.read_fd(0)
                else:
                    node = ctx.fs.lookup(name, ctx.cwd)
                    if node.type == "d":
                        ctx.e(f"awk: warning: command line argument `{name}' is a directory: skipped\n")
                        continue
                    data = yield from ctx.read_path(name)
                datas.append(("file", name, data.decode("utf-8", "surrogateescape")))
            except FSError as e:
                yield from ctx.flush()
                ctx.e(f"awk: fatal: cannot open file `{name}' for reading: {e.msg}\n")
                return None
        else:
            datas.append(src)

    def records():
        for src in datas:
            if src[0] == "assign":
                val = StrNum(_unescape_assign(src[2]))
                if src[1] in ("FS", "RS", "OFS", "ORS", "SUBSEP", "CONVFMT", "OFMT"):
                    rt.set_special(src[1], val)
                else:
                    G[src[1]] = val
                continue
            G["FILENAME"] = src[1] if src[1] != "-" else ""
            G["FNR"] = 0.0
            rt.cur_file = src[1]
            for rec in split_records(src[2], rt.rs):
                yield rec
                if getattr(rt, "skip_file", False):
                    rt.skip_file = False
                    break
    return records()


def _drive(ctx, rt, rules, gen_mode):
    G = rt.G
    begin = [f for k, f in rules if k == "BEGIN"]
    main = [f for k, f in rules if k == "main"]
    end = [f for k, f in rules if k == "END"]

    def call(f):
        if gen_mode:
            yield from f(rt, G)
        else:
            f(rt, G)
    exit_status = None
    if rt.parser.uses_getline:
        rt.records = yield from _prepare_records(ctx, rt)
        if rt.records is None:
            return 2
    try:
        for f in begin:
            yield from call(f)
    except AwkExit as e:
        exit_status = int(_num(e.status)) if e.status is not None else 0
        main = []
    except NextRecord:
        raise AwkFatal("`next' used in BEGIN action")
    if exit_status is None and (main or end):
        if rt.records is None:
            rt.records = yield from _prepare_records(ctx, rt)
            if rt.records is None:
                return 2
        try:
            if main:
                while True:
                    try:
                        rec = next(rt.records)
                    except StopIteration:
                        break
                    G["NR"] = _num(G.get("NR", 0.0)) + 1
                    G["FNR"] = _num(G.get("FNR", 0.0)) + 1
                    rt.set_record(rec)
                    rt.tick()
                    try:
                        for f in main:
                            if gen_mode:
                                yield from f(rt, G)
                            else:
                                f(rt, G)
                    except NextRecord:
                        pass
                    except NextFile:
                        rt.skip_file = True
                    if rt.steps % 2048 == 0:
                        yield from ctx.maybe_flush()
            else:
                for rec in rt.records:
                    G["NR"] = _num(G.get("NR", 0.0)) + 1
                    G["FNR"] = _num(G.get("FNR", 0.0)) + 1
                    rt.set_record(rec)
        except AwkExit as e:
            exit_status = int(_num(e.status)) if e.status is not None else 0
    try:
        if end and (exit_status is None or True):
            for f in end:
                yield from call(f)
    except AwkExit as e:
        exit_status = int(_num(e.status)) if e.status is not None else (exit_status or 0)
    for name in list(rt.cmd_out):
        yield from rt.flush_cmd(name)
    return exit_status or 0
