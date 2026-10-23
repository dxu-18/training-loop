"""GNU sed (and BSD `sed -i ''` on simulated macOS hosts)."""

from __future__ import annotations

import re

from ..exec import util
from ..regex import RegexError, compile_re
from ..vfs import FSError


class SedError(Exception):
    pass


class Cmd:
    __slots__ = ("a1", "a2", "neg", "name", "arg", "rx", "rep", "flags", "nth", "glob", "pflag", "wfile",
                 "label", "block_end", "y", "in_range", "idx", "range_end_line", "eflag")

    def __init__(self):
        self.a1 = self.a2 = None
        self.neg = False
        self.name = ""
        self.arg = None
        self.rx = None
        self.rep = None
        self.flags = ""
        self.nth = 1
        self.glob = False
        self.pflag = 0
        self.wfile = None
        self.label = None
        self.block_end = None
        self.y = None
        self.in_range = False
        self.idx = 0
        self.range_end_line = None
        self.eflag = False


class Script:
    def __init__(self, text: str, ere: bool, expr_no=1, posix=False):
        self.text = text
        self.i = 0
        self.ere = ere
        self.cmds: list[Cmd] = []
        self.labels: dict[str, int] = {}
        self.expr_no = expr_no
        self.last_rx = None

    def err(self, msg, at=None):
        pos = (self.i if at is None else at)
        raise SedError(f"-e expression #{self.expr_no}, char {pos}: {msg}")

    def parse(self):
        t = self.text
        n = len(t)
        stack = []
        while True:
            self._skip_ws_semis()
            if self.i >= n:
                break
            if t[self.i] == "#":
                j = t.find("\n", self.i)
                self.i = n if j < 0 else j + 1
                continue
            c = Cmd()
            c.a1 = self._address()
            if c.a1 is not None:
                self._skip_blanks()
                if self.i < n and t[self.i] == ",":
                    self.i += 1
                    self._skip_blanks()
                    c.a2 = self._address(second=True)
                    if c.a2 is None:
                        self.err("unexpected `,'")
            self._skip_blanks()
            while self.i < n and t[self.i] == "!":
                c.neg = True
                self.i += 1
                self._skip_blanks()
            if self.i >= n:
                self.err("missing command")
            name = t[self.i]
            self.i += 1
            c.name = name
            if name == "{":
                stack.append(len(self.cmds))
                self.cmds.append(c)
                continue
            if name == "}":
                if not stack:
                    self.err("unexpected `}'")
                if c.a1 is not None:
                    self.err("} doesn't want any addresses")
                start = stack.pop()
                self.cmds[start].block_end = len(self.cmds)
                c.name = "}"
                self.cmds.append(c)
                self._end_cmd()
                continue
            if name == "s":
                self._subst(c)
            elif name == "y":
                self._y(c)
            elif name in "aic":
                c.arg = self._text_arg()
            elif name in ":":
                if c.a1 is not None:
                    self.err(": doesn't want any addresses")
                lab = self._label()
                if not lab:
                    self.err("\":\" lacks a label")
                self.labels[lab] = len(self.cmds)
                c.label = lab
            elif name in "btT":
                c.label = self._label()
            elif name in "rRwW":
                c.arg = self._filename()
                if not c.arg and name in "rRwW":
                    self.err(f"missing filename in r/R/w/W commands")
            elif name in "qQlL":
                self._skip_blanks()
                m = re.match(r"\d+", t[self.i:])
                if m:
                    c.arg = int(m.group())
                    self.i += m.end()
                self._end_cmd()
            elif name in "=dDgGhHnNpPxzF":
                self._end_cmd()
            elif name == "e":
                c.arg = self._rest_of_line()
            elif name == "v":
                self._label()
            else:
                self.err(f"unknown command: `{name}'", self.i)
            self.cmds.append(c)
        if stack:
            raise SedError(f"-e expression #{self.expr_no}, char 0: unmatched `{{'")
        for c in self.cmds:
            if c.name in "btT" and c.label and c.label not in self.labels:
                raise SedError(f"can't find label for jump to `{c.label}'")
        return self

    def _skip_ws_semis(self):
        t = self.text
        while self.i < len(t) and t[self.i] in " \t\n;":
            self.i += 1

    def _skip_blanks(self):
        t = self.text
        while self.i < len(t) and t[self.i] in " \t":
            self.i += 1

    def _end_cmd(self):
        self._skip_blanks()
        t = self.text
        if self.i < len(t) and t[self.i] not in ";\n}#":
            self.err(f"extra characters after command")
        if self.i < len(t) and t[self.i] in ";\n":
            self.i += 1

    def _address(self, second=False):
        t = self.text
        n = len(t)
        if self.i >= n:
            return None
        c = t[self.i]
        if c.isdigit():
            m = re.match(r"(\d+)(?:~(\d+))?", t[self.i:])
            self.i += m.end()
            if m.group(2) is not None:
                return ("step", int(m.group(1)), int(m.group(2)))
            return ("line", int(m.group(1)))
        if c == "$":
            self.i += 1
            return ("last",)
        if second and c in "+~":
            m = re.match(r"[+~](\d+)", t[self.i:])
            if m:
                self.i += m.end()
                return ("plus" if c == "+" else "mult", int(m.group(1)))
        if c == "/" or c == "\\":
            if c == "\\":
                self.i += 1
                delim = t[self.i]
            else:
                delim = "/"
            self.i += 1
            pat = self._delimited(delim)
            flags = ""
            while self.i < n and t[self.i] in "IM":
                flags += t[self.i]
                self.i += 1
            return ("re", self._compile(pat, flags))
        return None

    def _delimited(self, delim):
        t = self.text
        out = []
        while True:
            if self.i >= len(t):
                self.err(f"unterminated address regex" if delim == "/" else "unterminated `s' command")
            ch = t[self.i]
            if ch == "\\" and self.i + 1 < len(t):
                nxt = t[self.i + 1]
                if nxt == delim:
                    out.append(delim if delim not in ".*[]^$\\+?(){}|" else "\\" + delim)
                elif nxt == "n":
                    out.append("\\n")
                else:
                    out.append("\\" + nxt)
                self.i += 2
                continue
            if ch == "[":
                # bracket expressions may contain the delimiter
                j = self.i + 1
                if j < len(t) and t[j] == "^":
                    j += 1
                if j < len(t) and t[j] == "]":
                    j += 1
                while j < len(t) and t[j] != "]":
                    if t[j] == "[" and j + 1 < len(t) and t[j + 1] in ":.=":
                        k = t.find(t[j + 1] + "]", j + 2)
                        j = k + 2 if k > 0 else j + 1
                        continue
                    j += 1
                if j < len(t):
                    out.append(t[self.i:j + 1])
                    self.i = j + 1
                    continue
            if ch == delim:
                self.i += 1
                return "".join(out)
            if ch == "\n" and delim != "\n":
                self.err("unterminated address regex")
            out.append(ch)
            self.i += 1

    def _compile(self, pat, flags):
        if pat == "":
            return None   # reuse the last regex
        try:
            return compile_re(pat, "ere" if self.ere else "bre", "I" in flags, True, "M" in flags)
        except RegexError as e:
            raise SedError(f"-e expression #{self.expr_no}, char {self.i}: {e}") from None

    def _subst(self, c):
        t = self.text
        if self.i >= len(t):
            self.err("unterminated `s' command")
        delim = t[self.i]
        if delim in "\n\\":
            self.err("unterminated `s' command")
        self.i += 1
        pat = self._delimited(delim)
        rep = []
        while True:
            if self.i >= len(t):
                self.err("unterminated `s' command")
            ch = t[self.i]
            if ch == "\\" and self.i + 1 < len(t):
                nxt = t[self.i + 1]
                if nxt == delim:
                    rep.append(("lit", delim))
                elif nxt == "\n":
                    rep.append(("lit", "\n"))
                elif nxt.isdigit():
                    rep.append(("grp", int(nxt)))
                elif nxt == "&":
                    rep.append(("lit", "&"))
                elif nxt == "n":
                    rep.append(("lit", "\n"))
                elif nxt == "t":
                    rep.append(("lit", "\t"))
                elif nxt in "LUluE":
                    rep.append(("case", nxt))
                elif nxt == "\\":
                    rep.append(("lit", "\\"))
                else:
                    rep.append(("lit", nxt))
                self.i += 2
                continue
            if ch == delim:
                self.i += 1
                break
            if ch == "&":
                rep.append(("grp", 0))
            else:
                rep.append(("lit", ch))
            self.i += 1
        flags = ""
        while self.i < len(t) and t[self.i] in "gpiImMe0123456789w":
            ch = t[self.i]
            if ch == "w":
                self.i += 1
                c.wfile = self._filename()
                break
            flags += ch
            self.i += 1
        c.glob = "g" in flags
        c.pflag = flags.count("p")
        m = re.search(r"\d+", flags)
        c.nth = int(m.group()) if m else 1
        if m and int(m.group()) == 0:
            self.err("number option to `s' command may not be zero")
        c.eflag = "e" in flags
        icase = "i" in flags or "I" in flags
        c.flags = ("I" if icase else "") + ("M" if ("m" in flags or "M" in flags) else "")
        c.rx = self._compile(pat, c.flags)
        if c.rx is not None:
            ng = c.rx.groups
            for kind, v in rep:
                if kind == "grp" and v > ng:
                    raise SedError(f"-e expression #{self.expr_no}, char {self.i}: invalid reference \\{v} on `s' command's RHS")
        c.rep = rep
        if c.wfile is None:
            self._end_cmd()

    def _y(self, c):
        t = self.text
        delim = t[self.i]
        self.i += 1

        def part():
            out = []
            while True:
                if self.i >= len(t):
                    self.err("unterminated `y' command")
                ch = t[self.i]
                if ch == "\\" and self.i + 1 < len(t):
                    nxt = t[self.i + 1]
                    out.append({"n": "\n", "t": "\t", "\\": "\\"}.get(nxt, nxt))
                    self.i += 2
                    continue
                if ch == delim:
                    self.i += 1
                    return "".join(out)
                out.append(ch)
                self.i += 1
        a, b = part(), part()
        if len(a) != len(b):
            self.err("strings for `y' command are different lengths")
        c.y = str.maketrans(a, b)
        self._end_cmd()

    def _text_arg(self):
        """Text for a/i/c: GNU one-liner `a text` or the classic `a\\` + following lines."""
        t = self.text
        self._skip_blanks()
        if self.i < len(t) and t[self.i] == "\\":
            self.i += 1
            if self.i < len(t) and t[self.i] == "\n":
                self.i += 1
        out = []
        while self.i < len(t):
            ch = t[self.i]
            if ch == "\\" and self.i + 1 < len(t):
                nxt = t[self.i + 1]
                if nxt == "\n":
                    out.append("\n")
                    self.i += 2
                    continue
                out.append({"t": "\t", "n": "\n"}.get(nxt, nxt))
                self.i += 2
                continue
            if ch == "\n":
                self.i += 1
                break
            out.append(ch)
            self.i += 1
        return "".join(out)

    def _label(self):
        self._skip_blanks()
        t = self.text
        m = re.match(r"[^;\n}\s]*", t[self.i:])
        lab = m.group().strip()
        self.i += m.end()
        self._skip_blanks()
        if self.i < len(t) and t[self.i] in ";\n":
            self.i += 1
        return lab

    def _filename(self):
        self._skip_blanks()
        t = self.text
        j = t.find("\n", self.i)
        j = len(t) if j < 0 else j
        name = t[self.i:j]
        self.i = j + 1
        return name

    def _rest_of_line(self):
        t = self.text
        j = t.find("\n", self.i)
        j = len(t) if j < 0 else j
        s = t[self.i:j].strip()
        self.i = j + 1
        return s


def expand_rep(rep, m):
    out = []
    mode = None       # None / "L" / "U"
    once = None       # "l" / "u"
    for kind, v in rep:
        if kind == "case":
            if v in "LU":
                mode = v
                once = None
            elif v == "E":
                mode = None
                once = None
            else:
                once = v
            continue
        s = v if kind == "lit" else (m.group(v) or "")
        if not s:
            continue
        if mode == "L":
            s = s.lower()
        elif mode == "U":
            s = s.upper()
        if once:
            s = (s[0].lower() if once == "l" else s[0].upper()) + s[1:]
            once = None
        out.append(s)
    return "".join(out)


class Runner:
    def __init__(self, ctx, script: Script, quiet: bool, separate: bool, inplace, zero: bool, posix=False,
                 unbuffered=False):
        self.ctx, self.s, self.quiet, self.separate, self.inplace, self.zero = ctx, script, quiet, separate, inplace, zero
        self.nl = "\0" if zero else "\n"
        self.hold = ""
        self.out: list[str] = []
        self.wfiles: dict[str, object] = {}
        self.lineno = 0
        self.last_rx = None
        self.quit_status = None

    def regex(self, rx):
        if rx is None:
            if self.last_rx is None:
                raise SedError("no previous regular expression")
            return self.last_rx
        self.last_rx = rx
        return rx

    def match_addr(self, a, ps, is_last):
        k = a[0]
        if k == "line":
            return self.lineno == a[1]
        if k == "last":
            return is_last
        if k == "re":
            return self.regex(a[1]).search(ps) is not None
        if k == "step":
            first, step = a[1], a[2]
            if step <= 0:
                return self.lineno == first
            return self.lineno >= first and (self.lineno - first) % step == 0
        return False

    def selected(self, c, ps, is_last):
        if c.a1 is None:
            return True
        if c.a2 is None:
            r = self.match_addr(c.a1, ps, is_last)
            return r != c.neg
        # range
        if not c.in_range:
            start = (c.a1[0] == "line" and c.a1[1] == 0 and self.lineno >= 1) or self.match_addr(c.a1, ps, is_last)
            if c.a1[0] == "line" and c.a1[1] == 0:
                # 0,/re/: the regex may match on the very first line
                if self.lineno == 1 or c.idx == 0:
                    c.idx = 1
                    if c.a2[0] == "re" and self.regex(c.a2[1]).search(ps):
                        return not c.neg
                    c.in_range = True
                    return not c.neg
                return c.neg
            if start:
                a2 = c.a2
                if a2[0] == "line":
                    if a2[1] <= self.lineno:
                        return not c.neg   # one-line range
                    c.in_range = True
                elif a2[0] == "plus":
                    c.range_end_line = self.lineno + a2[1]
                    if a2[1] > 0:
                        c.in_range = True
                elif a2[0] == "mult":
                    if a2[1] > 0 and self.lineno % a2[1] != 0:
                        c.in_range = True
                elif a2[0] == "last":
                    if not is_last:
                        c.in_range = True
                else:
                    c.in_range = True
                return not c.neg
            return c.neg
        a2 = c.a2
        end = False
        if a2[0] == "line":
            end = self.lineno >= a2[1]
            if self.lineno > a2[1]:
                c.in_range = False
                return c.neg
        elif a2[0] == "plus":
            end = self.lineno >= c.range_end_line
        elif a2[0] == "mult":
            end = self.lineno % a2[1] == 0
        elif a2[0] == "last":
            end = is_last
        elif a2[0] == "re":
            end = self.regex(a2[1]).search(ps) is not None
        if end:
            c.in_range = False
        return not c.neg

    def run_file_set(self, inputs):
        """inputs: list of (name, text). Returns output text (for one in-place file or the whole stream)."""
        lines = []
        for name, text in inputs:
            if not text:
                continue                 # an empty input has no lines at all
            parts = text.split(self.nl)
            trailing = text.endswith(self.nl)
            if trailing:
                parts.pop()
            for i, p in enumerate(parts):
                lines.append((p, name, (i == len(parts) - 1) and not trailing))
        return self.process(lines)

    def process(self, lines):
        s = self.s
        cmds = s.cmds
        out = self.out
        nl = self.nl
        idx = 0
        n = len(lines)
        ps = None
        append_q = []
        while idx < n or ps is not None:
            if ps is None:
                if idx >= n:
                    break
                ps, fname, no_eol = lines[idx]
                idx += 1
                self.lineno += 1
            is_last = idx >= n
            tflag = False
            pc = 0
            deleted = False
            restart = False
            while pc < len(cmds):
                c = cmds[pc]
                pc += 1
                if c.name == "}":
                    continue
                if c.name == ":":
                    continue
                if not self.selected(c, ps, is_last):
                    if c.name == "{":
                        pc = c.block_end + 1
                    continue
                nm = c.name
                if nm == "{":
                    continue
                if nm == "s":
                    rx = self.regex(c.rx)
                    count = 0
                    done = False

                    def repl(m):
                        nonlocal count, done
                        count += 1
                        if count < c.nth or (done and not c.glob):
                            return m.group(0)
                        done = True
                        return expand_rep(c.rep, m)
                    new = rx.sub(repl, ps)
                    if done:
                        ps = new
                        tflag = True
                        if c.pflag:
                            out.append((ps + nl) * c.pflag)
                        if c.wfile:
                            self.write_file(c.wfile, ps + nl)
                elif nm == "y":
                    ps = ps.translate(c.y)
                elif nm == "p":
                    out.append(ps + nl)
                elif nm == "P":
                    out.append(ps.split("\n", 1)[0] + nl)
                elif nm == "d":
                    deleted = True
                    break
                elif nm == "D":
                    if "\n" not in ps:
                        deleted = True
                        break
                    ps = ps.split("\n", 1)[1]
                    restart = True
                    break
                elif nm == "h":
                    self.hold = ps
                elif nm == "H":
                    self.hold = self.hold + "\n" + ps
                elif nm == "g":
                    ps = self.hold
                elif nm == "G":
                    ps = ps + "\n" + self.hold
                elif nm == "x":
                    ps, self.hold = self.hold, ps
                elif nm == "n":
                    if not self.quiet:
                        out.append(ps + nl)
                    if idx >= n:
                        deleted = True     # GNU: no next line -> quit without printing again
                        self.flush_appends(append_q)
                        append_q = []
                        ps = None
                        return self.finish(out)
                    self.flush_appends(append_q)
                    append_q = []
                    ps, fname, no_eol = lines[idx]
                    idx += 1
                    self.lineno += 1
                    is_last = idx >= n
                elif nm == "N":
                    if idx >= n:
                        if not self.quiet:
                            out.append(ps + nl)
                        self.flush_appends(append_q)
                        return self.finish(out)
                    nxt, fname, no_eol = lines[idx]
                    idx += 1
                    self.lineno += 1
                    ps = ps + "\n" + nxt
                    is_last = idx >= n
                elif nm == "a":
                    append_q.append(("text", c.arg + "\n"))
                elif nm == "i":
                    out.append(c.arg + "\n")
                elif nm == "c":
                    if c.a2 is None or not c.in_range:
                        out.append(c.arg + "\n")
                    deleted = True
                    break
                elif nm == "=":
                    out.append(f"{self.lineno}\n")
                elif nm == "l":
                    out.append(_l_format(ps, c.arg if c.arg is not None else 70) + "\n")
                elif nm == "r":
                    append_q.append(("file", c.arg))
                elif nm == "R":
                    append_q.append(("rline", c.arg))
                elif nm == "w":
                    self.write_file(c.arg, ps + nl)
                elif nm == "W":
                    self.write_file(c.arg, ps.split("\n", 1)[0] + nl)
                elif nm == "z":
                    ps = ""
                elif nm == "F":
                    out.append(("-" if fname == "-" else fname) + "\n")
                elif nm == "q":
                    if not self.quiet:
                        out.append(ps + nl)
                    self.flush_appends(append_q)
                    self.quit_status = c.arg or 0
                    return self.finish(out)
                elif nm == "Q":
                    self.quit_status = c.arg or 0
                    return self.finish(out)
                elif nm == "b":
                    pc = s.labels[c.label] if c.label else len(cmds)
                elif nm == "t":
                    if tflag:
                        tflag = False
                        pc = s.labels[c.label] if c.label else len(cmds)
                elif nm == "T":
                    if not tflag:
                        pc = s.labels[c.label] if c.label else len(cmds)
                    else:
                        tflag = False
                elif nm == "e":
                    pass
            if restart:
                if not self.quiet and False:
                    pass
                self.flush_appends(append_q)
                append_q = []
                continue
            if not deleted and not self.quiet:
                out.append(ps + ("" if (no_eol and is_last) else nl))
            self.flush_appends(append_q)
            append_q = []
            ps = None
        return self.finish(out)

    def finish(self, out):
        text = "".join(out)
        self.out = []
        return text

    def flush_appends(self, q):
        for kind, v in q:
            if kind == "text":
                self.out.append(v)
            elif kind == "file":
                try:
                    data = bytes(self.ctx.fs.lookup(v, self.ctx.cwd).data).decode("utf-8", "surrogateescape")
                    self.out.append(data)
                except FSError:
                    pass
            elif kind == "rline":
                st = self.__dict__.setdefault("_rlines", {})
                if v not in st:
                    try:
                        st[v] = bytes(self.ctx.fs.lookup(v, self.ctx.cwd).data).decode().splitlines(True)
                    except FSError:
                        st[v] = []
                if st[v]:
                    line = st[v].pop(0)
                    self.out.append(line if line.endswith("\n") else line + "\n")

    def write_file(self, name, text):
        if name in ("/dev/stdout",):
            self.out.append(text)
            return
        if name == "/dev/stderr":
            self.ctx.e(text)
            return
        of = self.wfiles.get(name)
        if of is None:
            of = self.ctx.open(name, "w")
            self.wfiles[name] = of
        node = of.inode
        node.data += text.encode("utf-8", "surrogateescape")


def _l_format(s, width):
    out = []
    for ch in s:
        o = ord(ch)
        if ch == "\\":
            out.append("\\\\")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\t":
            out.append("\\t")
        elif ch == "\a":
            out.append("\\a")
        elif ch == "\b":
            out.append("\\b")
        elif ch == "\f":
            out.append("\\f")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\v":
            out.append("\\v")
        elif o < 32 or o >= 127:
            for b in ch.encode("utf-8", "surrogateescape"):
                out.append("\\%03o" % b)
        else:
            out.append(ch)
    text = "".join(out)
    if width and width > 1:
        lines = []
        cur = ""
        for piece in re.findall(r"\\[0-7]{3}|\\.|.", text):
            if len(cur) + len(piece) > width - 1:
                lines.append(cur + "\\")
                cur = ""
            cur += piece
        lines.append(cur + "$")
        return "\n".join(lines)
    return text + "$"


@util("sed", "gsed")
def u_sed(ctx):
    a = ctx.argv[1:]
    bsd = ctx.host.os == "macos" and ctx.name == "sed"
    scripts = []
    quiet = separate = zero = ere = False
    inplace = None
    files = []
    i = 0
    while i < len(a):
        x = a[i]
        if x == "--":
            files += a[i + 1:]
            break
        if x in ("-n", "--quiet", "--silent"):
            quiet = True
        elif x in ("-E", "-r", "--regexp-extended"):
            ere = True
        elif x in ("-s", "--separate"):
            separate = True
        elif x in ("-z", "--null-data"):
            zero = True
        elif x in ("-u", "--unbuffered", "--posix", "--debug", "--sandbox", "--follow-symlinks"):
            pass
        elif x == "-e" or x == "--expression":
            if i + 1 >= len(a):
                ctx.err("option requires an argument -- 'e'")
                return ctx.usage()
            scripts.append(("e", a[i + 1]))
            i += 1
        elif x.startswith("--expression="):
            scripts.append(("e", x.split("=", 1)[1]))
        elif x == "-f" or x == "--file":
            try:
                txt = yield from ctx.text(a[i + 1])
            except FSError as e:
                ctx.err(f"couldn't open file {a[i + 1]}: {e.msg}")
                return 1
            scripts.append(("f", txt.rstrip("\n")))
            i += 1
        elif x == "-i" and bsd:
            if i + 1 < len(a):
                inplace = a[i + 1]
                i += 1
        elif x.startswith("-i") or x.startswith("--in-place"):
            inplace = x[2:] if x.startswith("-i") else (x.split("=", 1)[1] if "=" in x else "")
            if bsd and x != "-i":
                inplace = x[2:]
        elif x.startswith("-l") or x.startswith("--line-length"):
            if x in ("-l",):
                i += 1
        elif x == "--help":
            ctx.w("Usage: sed [OPTION]... {script-only-if-no-other-script} [input-file]...\n\n"
                  "  -n, --quiet, --silent\n                 suppress automatic printing of pattern space\n"
                  "  -e script, --expression=script\n                 add the script to the commands to be executed\n"
                  "  -i[SUFFIX], --in-place[=SUFFIX]\n                 edit files in place (makes backup if SUFFIX supplied)\n"
                  "  -E, -r, --regexp-extended\n                 use extended regular expressions in the script\n")
            return 0
        elif x == "--version":
            ctx.w("sed (GNU sed) 4.9\n")
            return 0
        elif x.startswith("-") and len(x) > 1 and not x[1:].isdigit():
            combo = x[1:]
            ok = True
            for k, ch in enumerate(combo):
                if ch == "n":
                    quiet = True
                elif ch in "Er":
                    ere = True
                elif ch == "s":
                    separate = True
                elif ch == "z":
                    zero = True
                elif ch == "u":
                    pass
                elif ch == "i":
                    inplace = combo[k + 1:]
                    break
                elif ch == "e":
                    rest = combo[k + 1:]
                    if rest:
                        scripts.append(("e", rest))
                    else:
                        scripts.append(("e", a[i + 1]))
                        i += 1
                    break
                else:
                    ctx.err(f"invalid option -- '{ch}'")
                    ctx.e("Usage: sed [OPTION]... {script-only-if-no-other-script} [input-file]...\n\n")
                    return 1
                ok = ok
        else:
            files.append(x)
        i += 1
    if not scripts:
        if not files:
            ctx.e("Usage: sed [OPTION]... {script-only-if-no-other-script} [input-file]...\n\n"
                  "  -n, --quiet, --silent\n                 suppress automatic printing of pattern space\n")
            return 1
        scripts.append(("e", files.pop(0)))
    text = "\n".join(s for _, s in scripts)
    try:
        script = Script(text, ere).parse()
    except SedError as e:
        ctx.err(str(e))
        return 1
    if inplace is not None and not files:
        ctx.err("no input files")
        return 1 if not bsd else 1
    status = 0
    if inplace is not None:
        for f in files:
            try:
                node = ctx.lookup(f)
                if node.type != "f":
                    ctx.err(f"couldn't edit {f}: not a regular file")
                    status = 4
                    continue
                data = bytes(node.data).decode("utf-8", "surrogateescape")
            except FSError as e:
                ctx.err(f"can't read {f}: {e.msg}")
                status = 2
                continue
            r = Runner(ctx, script, quiet, True, inplace, zero)
            for c in script.cmds:
                c.in_range = False
                c.idx = 0
            try:
                result = r.run_file_set([(f, data)])
            except SedError as e:
                ctx.err(str(e))
                return 4
            if inplace:
                if "*" in inplace:
                    base = f.rsplit("/", 1)
                    bname = inplace.replace("*", base[-1])
                    backup = (base[0] + "/" + bname) if len(base) > 1 and "/" not in inplace else bname
                else:
                    backup = f + inplace
                ctx.fs.write_file(backup, bytes(node.data), ctx.cwd, node.mode, node.uid, node.gid, node.mtime)
            node.data[:] = result.encode("utf-8", "surrogateescape")
            node.mtime = node.ctime = ctx.now
            if r.quit_status is not None:
                return r.quit_status
        return status
    inputs = []
    for f in files or ["-"]:
        try:
            data = yield from ctx.read_path(f)
            inputs.append((f, data.decode("utf-8", "surrogateescape")))
        except FSError as e:
            ctx.err(f"can't read {f}: {e.msg}")
            status = 2
    r = Runner(ctx, script, quiet, separate, None, zero)
    try:
        if separate:
            for inp in inputs:
                for c in script.cmds:
                    c.in_range = False
                ctx.w(r.run_file_set([inp]))
                if r.quit_status is not None:
                    break
        else:
            ctx.w(r.run_file_set(inputs))
    except SedError as e:
        ctx.err(str(e))
        return 4
    if r.quit_status is not None:
        return r.quit_status
    return status
