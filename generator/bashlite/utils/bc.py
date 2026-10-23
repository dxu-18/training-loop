"""bc: arbitrary-precision decimal calculator (GNU bc subset: scale, ibase/obase, functions, -l math library)."""

from __future__ import annotations

import re
from decimal import ROUND_DOWN, Decimal, getcontext, localcontext

from ..exec import util

getcontext().prec = 400


class BcError(Exception):
    pass


class Num:
    """A bc number: value with a scale (digits after the point)."""
    __slots__ = ("v", "scale")

    def __init__(self, v, scale=0):
        self.v = Decimal(v)
        self.scale = scale

    @staticmethod
    def parse(text: str, ibase=10):
        if ibase == 10:
            if "." in text:
                return Num(Decimal(text), len(text.split(".")[1]))
            return Num(Decimal(text), 0)
        digits = "0123456789ABCDEF"
        whole, _, frac = text.partition(".")
        v = Decimal(0)
        for ch in whole:
            v = v * ibase + digits.index(ch)
        if frac:
            f = Decimal(0)
            for k, ch in enumerate(frac, 1):
                f += Decimal(digits.index(ch)) / (Decimal(ibase) ** k)
            v += f
        return Num(v, len(frac))


def trunc(v: Decimal, scale: int) -> Decimal:
    q = Decimal(1).scaleb(-scale)
    with localcontext() as c:
        c.prec = 500
        return v.quantize(q, rounding=ROUND_DOWN)


def fmt(n: Num, obase=10) -> str:
    v = trunc(n.v, n.scale) if n.scale >= 0 else n.v
    if v == 0:
        return "0"
    if obase != 10:
        neg = v < 0
        v = abs(v)
        whole = int(v)
        digits = "0123456789ABCDEF"
        s = ""
        if whole == 0:
            s = "0"
        while whole:
            s = digits[whole % obase] + s if obase <= 16 else f" {whole % obase:02d}" + s
            whole //= obase
        return ("-" if neg else "") + s
    s = format(v, "f")
    if "." in s:
        whole, frac = s.split(".")
        frac = frac[:n.scale].ljust(n.scale, "0") if n.scale else ""
        s = whole + ("." + frac if frac else "")
    if s.startswith("0.") and len(s) > 2:
        s = s[1:]
    elif s.startswith("-0.") and len(s) > 3:
        s = "-" + s[2:]
    if s in ("-0", "-.") or re.fullmatch(r"-\.?0*", s):
        s = "0"
    # bc wraps long numbers at 70 characters with a backslash
    if len(s) > 69:
        lines = [s[k:k + 69] for k in range(0, len(s), 69)]
        s = "\\\n".join(lines)
    return s


TOKEN = re.compile(r"""\s*(?:
    (?P<num>[0-9A-F]+(?:\.[0-9A-F]*)?|\.[0-9A-F]+)|
    (?P<name>[a-z][a-z0-9_]*)|
    (?P<str>"[^"]*")|
    (?P<op>\+\+|--|\+=|-=|\*=|/=|%=|\^=|==|!=|<=|>=|&&|\|\||[-+*/%^=<>!(){}\[\];,\n])
    )""", re.X)


class Interp:
    def __init__(self, mathlib=False):
        self.vars = {"scale": Num(20 if mathlib else 0), "ibase": Num(10), "obase": Num(10), "last": Num(0)}
        self.arrays: dict = {}
        self.funcs: dict = {}
        self.out: list[str] = []
        self.mathlib = mathlib
        if mathlib:
            self.load_mathlib()

    @property
    def scale(self):
        return int(self.vars["scale"].v)

    def load_mathlib(self):
        pass

    # ---- tokenizer / parser (recursive descent producing closures)
    def tokenize(self, src):
        src = re.sub(r"/\*.*?\*/", " ", src, flags=re.S)
        src = re.sub(r"#[^\n]*", "", src)
        src = src.replace("\\\n", "")
        toks = []
        i = 0
        while i < len(src):
            if src[i] in " \t":
                i += 1
                continue
            m = TOKEN.match(src, i)
            if not m or m.end() == i:
                raise BcError(f"illegal character: {src[i]}")
            kind = m.lastgroup
            toks.append((kind, m.group(kind)))
            i = m.end()
        toks.append(("eof", None))
        return toks

    def run(self, src):
        self.toks = self.tokenize(src)
        self.i = 0
        while self.peek()[0] != "eof":
            if self.peek() == ("op", "\n") or self.peek() == ("op", ";"):
                self.i += 1
                continue
            stmt = self.statement()
            try:
                stmt()
            except _Quit:
                return
            except BcError as e:
                self.out.append(f"__ERR__Runtime error (func=(main), adr=3): {e}\n")

    def peek(self):
        return self.toks[self.i]

    def take(self, kind=None, val=None):
        t = self.toks[self.i]
        if (kind and t[0] != kind) or (val is not None and t[1] != val):
            raise BcError(f"syntax error")
        self.i += 1
        return t

    def accept(self, val):
        if self.peek()[1] == val and self.peek()[0] in ("op", "name"):
            self.i += 1
            return True
        return False

    def statement(self):
        t = self.peek()
        if t == ("op", "{"):
            self.i += 1
            stmts = []
            while not self.accept("}"):
                if self.peek()[1] in ("\n", ";"):
                    self.i += 1
                    continue
                stmts.append(self.statement())

            def run_block():
                for s in stmts:
                    s()
            return run_block
        if t[0] == "name":
            w = t[1]
            if w == "quit":
                self.i += 1

                def q():
                    raise _Quit()
                return q
            if w == "print":
                self.i += 1
                items = []
                while True:
                    if self.peek()[0] == "str":
                        items.append(("s", self.take()[1][1:-1]))
                    else:
                        items.append(("e", self.expr()))
                    if not self.accept(","):
                        break

                def pr():
                    for kind, x in items:
                        if kind == "s":
                            self.out.append(x.replace("\\n", "\n").replace("\\t", "\t"))
                        else:
                            self.out.append(fmt(x(), int(self.vars["obase"].v)))
                return pr
            if w == "if":
                self.i += 1
                self.take("op", "(")
                cond = self.expr()
                self.take("op", ")")
                self._skip_nl()
                body = self.statement()
                save = self.i
                self._skip_nl()
                if self.peek() == ("name", "else"):
                    self.i += 1
                    self._skip_nl()
                    other = self.statement()
                else:
                    self.i = save
                    other = None

                def iff():
                    if cond().v != 0:
                        body()
                    elif other:
                        other()
                return iff
            if w == "while":
                self.i += 1
                self.take("op", "(")
                cond = self.expr()
                self.take("op", ")")
                self._skip_nl()
                body = self.statement()

                def wh():
                    n = 0
                    while cond().v != 0:
                        n += 1
                        if n > 1_000_000:
                            raise BcError("loop limit")
                        try:
                            body()
                        except _Break:
                            break
                return wh
            if w == "for":
                self.i += 1
                self.take("op", "(")
                init = self.expr() if self.peek() != ("op", ";") else None
                self.take("op", ";")
                cond = self.expr() if self.peek() != ("op", ";") else None
                self.take("op", ";")
                step = self.expr() if self.peek() != ("op", ")") else None
                self.take("op", ")")
                self._skip_nl()
                body = self.statement()

                def fo():
                    if init:
                        init()
                    n = 0
                    while cond is None or cond().v != 0:
                        n += 1
                        if n > 1_000_000:
                            raise BcError("loop limit")
                        try:
                            body()
                        except _Break:
                            break
                        if step:
                            step()
                return fo
            if w == "break":
                self.i += 1

                def br():
                    raise _Break()
                return br
            if w == "return":
                self.i += 1
                e = None
                if self.accept("("):
                    e = self.expr() if self.peek() != ("op", ")") else None
                    self.take("op", ")")
                elif self.peek()[1] not in ("\n", ";", "}") and self.peek()[0] != "eof":
                    e = self.expr()

                def ret():
                    raise _Return(e() if e else Num(0))
                return ret
            if w == "define":
                self.i += 1
                name = self.take("name")[1]
                self.take("op", "(")
                params = []
                while not self.accept(")"):
                    params.append(self.take("name")[1])
                    if self.accept("["):
                        self.take("op", "]")
                    self.accept(",")
                self._skip_nl()
                self.take("op", "{")
                self._skip_nl()
                autos = []
                if self.peek() == ("name", "auto"):
                    self.i += 1
                    while self.peek()[1] not in ("\n", ";"):
                        autos.append(self.take("name")[1])
                        self.accept(",")
                self.i -= 0
                stmts = []
                while not self.accept("}"):
                    if self.peek()[1] in ("\n", ";"):
                        self.i += 1
                        continue
                    stmts.append(self.statement())
                self.funcs[name] = (params, autos, stmts)
                return lambda: None
        if t[0] == "str":
            self.i += 1
            s = t[1][1:-1]
            return lambda: self.out.append(s)
        e = self.expr()
        is_assign = getattr(e, "assign", False)

        def show():
            v = e()
            if not is_assign:
                self.vars["last"] = v
                self.out.append(fmt(v, int(self.vars["obase"].v)) + "\n")
        return show

    def _skip_nl(self):
        while self.peek() == ("op", "\n"):
            self.i += 1

    # expressions (bc precedence: || && ! relational assignment +- */% ^ unary ++--)
    def expr(self):
        return self.or_()

    def or_(self):
        left = self.and_()
        while self.accept("||"):
            r = self.and_()
            left = (lambda a, b: lambda: Num(1 if a().v != 0 or b().v != 0 else 0))(left, r)
        return left

    def and_(self):
        left = self.not_()
        while self.accept("&&"):
            r = self.not_()
            left = (lambda a, b: lambda: Num(1 if a().v != 0 and b().v != 0 else 0))(left, r)
        return left

    def not_(self):
        if self.accept("!"):
            e = self.not_()
            return lambda: Num(1 if e().v == 0 else 0)
        return self.rel()

    def rel(self):
        left = self.assign()
        t = self.peek()
        if t[0] == "op" and t[1] in ("==", "!=", "<", "<=", ">", ">="):
            self.i += 1
            r = self.assign()
            op = t[1]
            ops = {"==": lambda x, y: x == y, "!=": lambda x, y: x != y, "<": lambda x, y: x < y,
                   "<=": lambda x, y: x <= y, ">": lambda x, y: x > y, ">=": lambda x, y: x >= y}
            f = ops[op]
            return lambda: Num(1 if f(left().v, r().v) else 0)
        return left

    def assign(self):
        start = self.i
        t = self.peek()
        if t[0] == "name" and t[1] not in ("sqrt", "length", "scale", "s", "c", "a", "l", "e", "j") or (
                t[0] == "name" and t[1] == "scale" and self.toks[self.i + 1][1] != "("):
            name = t[1]
            j = self.i + 1
            idx = None
            if self.toks[j] == ("op", "["):
                self.i += 2
                idx = self.expr()
                self.take("op", "]")
                j = self.i
            nt = self.toks[j]
            if nt[0] == "op" and nt[1] in ("=", "+=", "-=", "*=", "/=", "%=", "^="):
                self.i = j + 1
                rhs = self.assign()
                op = nt[1]

                def do():
                    v = rhs()
                    if op != "=":
                        cur = self.get(name, idx)
                        v = self.binop(op[0], cur, v)
                    self.set(name, idx, v)
                    return v
                do.assign = True
                return do
            self.i = start
        return self.additive()

    def get(self, name, idx):
        if idx is not None:
            return self.arrays.get(name, {}).get(int(idx().v), Num(0))
        return self.vars.get(name, Num(0))

    def set(self, name, idx, v):
        if idx is not None:
            self.arrays.setdefault(name, {})[int(idx().v)] = v
            return
        if name == "scale":
            v = Num(int(v.v))
        self.vars[name] = v

    def binop(self, op, a, b):
        s = self.scale
        if op == "+":
            return Num(a.v + b.v, max(a.scale, b.scale))
        if op == "-":
            return Num(a.v - b.v, max(a.scale, b.scale))
        if op == "*":
            sc = min(a.scale + b.scale, max(s, a.scale, b.scale))
            return Num(trunc(a.v * b.v, sc), sc)
        if op == "/":
            if b.v == 0:
                raise BcError("Divide by zero")
            return Num(trunc(a.v / b.v, s), s)
        if op == "%":
            if b.v == 0:
                raise BcError("Divide by zero")
            q = trunc(a.v / b.v, s)
            r = a.v - q * b.v
            sc = max(s + b.scale, a.scale)
            return Num(trunc(r, sc), sc)
        if op == "^":
            e = int(b.v)
            if b.v != e:
                raise BcError("Non-zero scale in exponent") if False else None
            if e >= 0:
                sc = min(a.scale * e, max(s, a.scale))
                return Num(trunc(a.v ** e, sc), sc)
            if a.v == 0:
                raise BcError("Divide by zero")
            return Num(trunc(Decimal(1) / (a.v ** -e), s), s)
        raise BcError(op)

    def additive(self):
        left = self.mult()
        while self.peek()[0] == "op" and self.peek()[1] in ("+", "-"):
            op = self.take()[1]
            r = self.mult()
            left = (lambda a, b, o: lambda: self.binop(o, a(), b()))(left, r, op)
        return left

    def mult(self):
        left = self.power()
        while self.peek()[0] == "op" and self.peek()[1] in ("*", "/", "%"):
            op = self.take()[1]
            r = self.power()
            left = (lambda a, b, o: lambda: self.binop(o, a(), b()))(left, r, op)
        return left

    def power(self):
        base = self.unary()
        if self.accept("^"):
            e = self.power()
            return lambda: self.binop("^", base(), e())
        return base

    def unary(self):
        if self.accept("-"):
            e = self.unary()
            return lambda: (lambda v: Num(-v.v, v.scale))(e())
        if self.peek()[0] == "op" and self.peek()[1] in ("++", "--"):
            op = self.take()[1]
            name = self.take("name")[1]

            def pre():
                v = self.get(name, None)
                nv = Num(v.v + (1 if op == "++" else -1), v.scale)
                self.set(name, None, nv)
                return nv
            return pre
        return self.postfix()

    def postfix(self):
        e = self.primary()
        if self.peek()[0] == "op" and self.peek()[1] in ("++", "--") and getattr(e, "lname", None):
            op = self.take()[1]
            name = e.lname

            def post():
                v = self.get(name, None)
                self.set(name, None, Num(v.v + (1 if op == "++" else -1), v.scale))
                return v
            return post
        return e

    def primary(self):
        t = self.take()
        if t[0] == "num":
            text = t[1]

            def num():
                return Num.parse(text, int(self.vars["ibase"].v))
            return num
        if t == ("op", "("):
            e = self.expr()
            self.take("op", ")")
            return e
        if t[0] == "name":
            name = t[1]
            if self.accept("("):
                args = []
                while not self.accept(")"):
                    args.append(self.expr())
                    self.accept(",")
                return lambda: self.call(name, args)
            if self.accept("["):
                idx = self.expr()
                self.take("op", "]")
                return lambda: self.get(name, idx)
            f = lambda: self.get(name, None)  # noqa: E731
            f.lname = name
            return f
        raise BcError("syntax error")

    def call(self, name, args):
        vals = [a() for a in args]
        s = self.scale
        if name == "sqrt":
            v = vals[0]
            if v.v < 0:
                raise BcError("Square root of a negative number")
            sc = max(s, v.scale)
            with localcontext() as c:
                c.prec = sc + 50
                return Num(trunc(v.v.sqrt(), sc), sc)
        if name == "length":
            d = format(abs(vals[0].v), "f").replace(".", "").lstrip("0")
            return Num(max(len(d), 1))
        if name == "scale":
            return Num(vals[0].scale)
        if self.mathlib and name in ("s", "c", "a", "l", "e") and name not in self.funcs:
            import math
            x = float(vals[0].v)
            fn = {"s": math.sin, "c": math.cos, "a": math.atan, "l": math.log, "e": math.exp}[name]
            if name == "l" and x <= 0:
                raise BcError("log of non-positive")
            r = Decimal(repr(fn(x)))
            return Num(trunc(r, s), s)
        if name not in self.funcs:
            raise BcError(f"Function {name} not defined.")
        params, autos, stmts = self.funcs[name]
        saved = {p: self.vars.get(p) for p in params + autos}
        for p, v in zip(params, vals):
            self.vars[p] = v
        for a in autos:
            self.vars[a] = Num(0)
        try:
            for st in stmts:
                st()
            result = Num(0)
        except _Return as r:
            result = r.value
        finally:
            for p, v in saved.items():
                if v is None:
                    self.vars.pop(p, None)
                else:
                    self.vars[p] = v
        return result


class _Quit(Exception):
    pass


class _Break(Exception):
    pass


class _Return(Exception):
    def __init__(self, value):
        super().__init__()
        self.value = value


@util("bc")
def u_bc(ctx):
    a = ctx.argv[1:]
    mathlib = any(x in ("-l", "--mathlib") or (x.startswith("-") and "l" in x and not x.startswith("--")) for x in a)
    files = [x for x in a if not x.startswith("-")]
    srcs = []
    for f in files:
        srcs.append((yield from ctx.text(f)))
    data = yield from ctx.read_fd(0)
    srcs.append(data.decode("utf-8", "surrogateescape"))
    it = Interp(mathlib)
    status = 0
    for src in srcs:
        try:
            it.run(src)
        except BcError as e:
            ctx.e(f"(standard_in) 1: {e}\n")
            status = 1
            break
    for piece in it.out:
        if piece.startswith("__ERR__"):
            ctx.e(piece[7:])
        else:
            ctx.w(piece)
    return status
