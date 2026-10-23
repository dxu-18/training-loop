"""Shell arithmetic: $(( )), (( )), let, array subscripts and `declare -i`. 64-bit wraparound integers."""

from __future__ import annotations

import re


class ArithError(Exception):
    def __init__(self, msg: str, token: str = ""):
        super().__init__(msg)
        self.msg, self.token = msg, token


MASK = (1 << 64) - 1


def wrap(v: int) -> int:
    v &= MASK
    return v - (1 << 64) if v >> 63 else v


TOKEN = re.compile(r"""\s*(?:
    (?P<num>0[xX][0-9a-fA-F]+|[0-9]+\#[0-9a-zA-Z@_]+|[0-9]+)|
    (?P<name>[A-Za-z_][A-Za-z0-9_]*)|
    (?P<op>\*\*=|<<=|>>=|&&|\|\||\+\+|--|<<|>>|<=|>=|==|!=|\*\*|[-+*/%&^|]=|[-+*/%<>=!~&^|?:,()\[\]])
    )""", re.X)


def parse_number(text: str) -> int:
    t = text
    if "#" in t:
        base_s, digits = t.split("#", 1)
        base = int(base_s)
        if not 2 <= base <= 64:
            raise ArithError("invalid arithmetic base", t)
        val = 0
        for ch in digits:
            if ch.isdigit():
                d = int(ch)
            elif "a" <= ch <= "z":
                d = ord(ch) - 87
            elif "A" <= ch <= "Z":
                d = ord(ch) - 29 if base > 36 else ord(ch) - 55
            elif ch == "@":
                d = 62
            elif ch == "_":
                d = 63
            else:
                raise ArithError("invalid number", t)
            if d >= base:
                raise ArithError("value too great for base", t)
            val = val * base + d
        return wrap(val)
    if t[:2] in ("0x", "0X"):
        return wrap(int(t[2:], 16))
    if len(t) > 1 and t[0] == "0":
        if any(c in "89" for c in t):
            raise ArithError("value too great for base", t)
        return wrap(int(t, 8))
    return wrap(int(t))


class Evaluator:
    """get(name, index_or_None) -> str ; set(name, index_or_None, int) ; index_of(name, expr_text) for a[...]"""

    def __init__(self, get, set_, depth: int = 0, is_assoc=None):
        self.get, self.set = get, set_
        self.depth = depth
        self.is_assoc = is_assoc or (lambda name: False)

    def eval(self, text: str) -> int:
        if self.depth > 1024:
            raise ArithError("expression recursion level exceeded", text)
        toks = self._tokens(text)
        if not toks:
            return 0
        self.toks, self.pos, self.text = toks, 0, text
        val = self._comma(True)
        if self.pos < len(self.toks):
            tok = self.toks[self.pos]
            raise ArithError("syntax error in expression", self._rest(tok))
        return val

    def _rest(self, tok) -> str:
        return self.text[tok[2]:]

    def _tokens(self, text: str):
        out, i, n = [], 0, len(text)
        while i < n:
            if text[i].isspace():
                i += 1
                continue
            m = TOKEN.match(text, i)
            if not m or m.end() == i:
                raise ArithError("syntax error: invalid arithmetic operator", text[i:].strip())
            kind = m.lastgroup
            val = m.group(kind)
            start = m.start(kind)
            if kind == "num" and m.end() < n and (text[m.end()].isalpha() or text[m.end()] == "_"):
                j = m.end()
                while j < n and (text[j].isalnum() or text[j] in "_#@"):
                    j += 1
                raise ArithError("value too great for base" if not text[start].isdigit() or True else "", text[start:j])
            out.append((kind, val, start))
            i = m.end()
        return out

    # ------------------------------------------------------------ helpers

    def _peek(self):
        return self.toks[self.pos] if self.pos < len(self.toks) else None

    def _accept(self, op):
        t = self._peek()
        if t and t[0] == "op" and t[1] == op:
            self.pos += 1
            return True
        return False

    def _expect(self, op):
        if not self._accept(op):
            t = self._peek()
            raise ArithError(f"syntax error: `{op}' expected" if op != ")" else "missing `)'",
                             self._rest(t) if t else "")

    def _value_of(self, name, index, ev):
        raw = self.get(name, index)
        if raw is None or raw == "":
            return 0
        raw = raw.strip()
        if re.fullmatch(r"-?[0-9]+", raw):
            return wrap(int(raw))
        sub = Evaluator(self.get, self.set, self.depth + 1, self.is_assoc)
        return sub.eval(raw) if ev else 0

    # ------------------------------------------------------------ grammar (lowest precedence first)

    def _comma(self, ev):
        v = self._assign(ev)
        while self._accept(","):
            v = self._assign(ev)
        return v

    def _lvalue(self):
        """If the next tokens are `name` or `name[expr]` followed by an assignment op, return (name, index)."""
        t = self._peek()
        if not t or t[0] != "name":
            return None
        save = self.pos
        self.pos += 1
        index = None
        if self._accept("["):
            depth, start = 1, self.pos
            while self.pos < len(self.toks) and depth:
                tt = self.toks[self.pos]
                if tt[1] == "[":
                    depth += 1
                elif tt[1] == "]":
                    depth -= 1
                self.pos += 1
            if depth:
                self.pos = save
                return None
            index = (start, self.pos - 1)
        nt = self._peek()
        if nt and nt[0] == "op" and nt[1] in ("=", "+=", "-=", "*=", "/=", "%=", "<<=", ">>=", "&=", "^=", "|=", "**="):
            return t[1], index
        self.pos = save
        return None

    def _index_value(self, span, ev):
        start, end = span
        saved = (self.toks, self.pos)
        sub_toks = self.toks[start:end]
        self.toks, self.pos = sub_toks, 0
        v = self._comma(ev) if sub_toks else 0
        self.toks, self.pos = saved
        return v

    def _assign(self, ev):
        lv = self._lvalue()
        if lv is None:
            return self._ternary(ev)
        name, span = lv
        op = self.toks[self.pos][1]
        self.pos += 1
        index = (self._raw(span) if self.is_assoc(name) else self._index_value(span, ev)) if span else None
        rhs = self._assign(ev)
        if not ev:
            return rhs
        if op == "=":
            val = rhs
        else:
            cur = self._value_of(name, index, ev)
            val = self._binop(op[:-1], cur, rhs)
        self.set(name, index, val)
        return val

    def _ternary(self, ev):
        cond = self._lor(ev)
        if self._accept("?"):
            a = self._assign(ev and cond != 0)
            self._expect(":")
            b = self._assign(ev and cond == 0)
            return a if cond else b
        return cond

    def _lor(self, ev):
        v = self._land(ev)
        while self._accept("||"):
            r = self._land(ev and not v)
            v = 1 if (v or r) else 0
        return v

    def _land(self, ev):
        v = self._bor(ev)
        while self._accept("&&"):
            r = self._bor(ev and v)
            v = 1 if (v and r) else 0
        return v

    def _bor(self, ev):
        v = self._bxor(ev)
        while True:
            t = self._peek()
            if t and t[1] == "|" :
                self.pos += 1
                v = wrap(v | self._bxor(ev))
            else:
                return v

    def _bxor(self, ev):
        v = self._band(ev)
        while self._accept("^"):
            v = wrap(v ^ self._band(ev))
        return v

    def _band(self, ev):
        v = self._eq(ev)
        while True:
            t = self._peek()
            if t and t[1] == "&":
                self.pos += 1
                v = wrap(v & self._eq(ev))
            else:
                return v

    def _eq(self, ev):
        v = self._rel(ev)
        while True:
            if self._accept("=="):
                v = int(v == self._rel(ev))
            elif self._accept("!="):
                v = int(v != self._rel(ev))
            else:
                return v

    def _rel(self, ev):
        v = self._shift(ev)
        while True:
            t = self._peek()
            if t and t[0] == "op" and t[1] in ("<", ">", "<=", ">="):
                self.pos += 1
                r = self._shift(ev)
                v = int({"<": v < r, ">": v > r, "<=": v <= r, ">=": v >= r}[t[1]])
            else:
                return v

    def _shift(self, ev):
        v = self._add(ev)
        while True:
            t = self._peek()
            if t and t[0] == "op" and t[1] in ("<<", ">>"):
                self.pos += 1
                v = self._binop(t[1], v, self._add(ev))
            else:
                return v

    def _add(self, ev):
        v = self._mul(ev)
        while True:
            t = self._peek()
            if t and t[0] == "op" and t[1] in ("+", "-"):
                self.pos += 1
                r = self._mul(ev)
                v = wrap(v + r if t[1] == "+" else v - r)
            else:
                return v

    def _mul(self, ev):
        v = self._pow(ev)
        while True:
            t = self._peek()
            if t and t[0] == "op" and t[1] in ("*", "/", "%"):
                self.pos += 1
                rt = self._peek()
                r = self._pow(ev)
                v = self._binop(t[1], v, r, ev, rt)
            else:
                return v

    def _pow(self, ev):
        base = self._unary(ev)
        if self._accept("**"):
            exp = self._pow(ev)   # right associative
            if exp < 0:
                if ev:
                    raise ArithError("exponent less than 0", str(exp))
                return 0
            return wrap(pow(base, exp, 1 << 64)) if ev else 0
        return base

    def _binop(self, op, a, b, ev=True, tok=None):
        if op in ("/", "%"):
            if b == 0:
                if not ev:
                    return 0
                raise ArithError("division by 0", self._rest(tok) if tok else "")
            q = abs(a) // abs(b)
            if (a < 0) != (b < 0):
                q = -q
            return wrap(q) if op == "/" else wrap(a - b * q)
        if op == "+":
            return wrap(a + b)
        if op == "-":
            return wrap(a - b)
        if op == "*":
            return wrap(a * b)
        if op == "<<":
            return wrap(a << (b & 63))
        if op == ">>":
            return wrap(a >> (b & 63))
        if op == "&":
            return wrap(a & b)
        if op == "|":
            return wrap(a | b)
        if op == "^":
            return wrap(a ^ b)
        if op == "**":
            return wrap(pow(a, b, 1 << 64)) if b >= 0 else 0
        raise ArithError("syntax error", op)

    def _unary(self, ev):
        t = self._peek()
        if t is None:
            raise ArithError("syntax error: operand expected", "")
        if t[0] == "op":
            op = t[1]
            if op in ("++", "--"):
                self.pos += 1
                n = self._peek()
                if not n or n[0] != "name":
                    raise ArithError("syntax error: operand expected", self._rest(n) if n else "")
                name, index = self._name_ref(ev)
                v = self._value_of(name, index, ev) + (1 if op == "++" else -1)
                if ev:
                    self.set(name, index, wrap(v))
                return wrap(v)
            if op in ("+", "-", "!", "~"):
                self.pos += 1
                v = self._unary(ev)
                return {"+": v, "-": wrap(-v), "!": int(not v), "~": wrap(~v)}[op]
            if op == "(":
                self.pos += 1
                v = self._comma(ev)
                self._expect(")")
                return v
            raise ArithError("syntax error: operand expected", self._rest(t))
        if t[0] == "num":
            self.pos += 1
            return parse_number(t[1])
        # a variable, maybe with ++/--
        name, index = self._name_ref(ev)
        n = self._peek()
        if n and n[0] == "op" and n[1] in ("++", "--"):
            self.pos += 1
            v = self._value_of(name, index, ev)
            if ev:
                self.set(name, index, wrap(v + (1 if n[1] == "++" else -1)))
            return v
        return self._value_of(name, index, ev)

    def _name_ref(self, ev):
        t = self.toks[self.pos]
        self.pos += 1
        index = None
        if self._accept("["):
            depth, start = 1, self.pos
            while self.pos < len(self.toks) and depth:
                tt = self.toks[self.pos]
                if tt[1] == "[":
                    depth += 1
                elif tt[1] == "]":
                    depth -= 1
                self.pos += 1
            if depth:
                raise ArithError("syntax error: `]' expected", "")
            span = (start, self.pos - 1)
            index = self._raw(span) if self.is_assoc(t[1]) else self._index_value(span, ev)
        return t[1], index

    def _raw(self, span) -> str:
        start, end = span
        if start >= end:
            return ""
        return self.text[self.toks[start][2]:self.toks[end][2]].strip()
