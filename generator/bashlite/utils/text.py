"""Text tools: cat tac nl head tail wc sort uniq cut tr paste join split comm cmp diff column rev fold tee shuf
md5sum sha*sum base64 xxd hexdump od strings less more."""

from __future__ import annotations

import hashlib
import math
import re

from ..exec import UsageError, gnu_getopt, util
from ..fileio import read, readline
from ..kernel import NS
from ..vfs import FSError

ENC = "utf-8"


def dec(b: bytes) -> str:
    return b.decode(ENC, "surrogateescape")


def enc(s: str) -> bytes:
    return s.encode(ENC, "surrogateescape")


def _open_err(ctx, path, e):
    ctx.err(f"{path}: {e.msg}")


def read_inputs(ctx, paths):
    """Generator yielding nothing useful; returns list of (path, bytes or None on error)."""
    out = []
    for p in paths or ["-"]:
        try:
            data = yield from ctx.read_path(p)
            out.append((p, data))
        except FSError as e:
            _open_err(ctx, p, e)
            out.append((p, None))
    return out


def split_lines(text: str):
    """Lines without terminators; a final unterminated line counts."""
    if not text:
        return []
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()
    return lines


# ---------------------------------------------------------------------------- cat tac nl rev

@util("cat")
def u_cat(ctx):
    try:
        opts, ops = gnu_getopt(ctx.argv[1:], "AbeEnstTuv", ["show-all", "number-nonblank", "show-ends", "number",
                                                           "squeeze-blank", "show-tabs", "show-nonprinting", "help",
                                                           "version"], name="cat")
    except UsageError as e:
        ctx.err(e.msg)
        return ctx.usage()
    num = nonblank = ends = tabs = squeeze = nonprint = False
    for k, _ in opts:
        if k in ("-n", "--number"):
            num = True
        elif k in ("-b", "--number-nonblank"):
            nonblank = True
        elif k in ("-E", "--show-ends"):
            ends = True
        elif k in ("-T", "--show-tabs"):
            tabs = True
        elif k in ("-s", "--squeeze-blank"):
            squeeze = True
        elif k in ("-v", "--show-nonprinting"):
            nonprint = True
        elif k in ("-A", "--show-all"):
            nonprint = ends = tabs = True
        elif k == "-e":
            nonprint = ends = True
        elif k == "-t":
            nonprint = tabs = True
        elif k == "--help":
            ctx.w("Usage: cat [OPTION]... [FILE]...\nConcatenate FILE(s) to standard output.\n")
            return 0
    plain = not (num or nonblank or ends or tabs or squeeze or nonprint)
    status = 0
    lineno = 0
    prev_blank = False
    pending_line_start = True
    for p in ops or ["-"]:
        try:
            of = ctx.proc.fds.get(0) if p == "-" else ctx.open(p)
            if of is None:
                raise FSError("EBADF")
            if of.kind == "dir":
                ctx.err(f"{p}: Is a directory")
                status = 1
                continue
        except FSError as e:
            _open_err(ctx, p, e)
            status = 1
            continue
        o1 = ctx.proc.fds.get(1)
        if of.kind == "file" and o1 is not None and o1.kind == "file" and o1.inode is of.inode and of.inode.data:
            ctx.err(f"{p}: input file is output file")
            status = 1
            continue
        while True:
            data = yield from read(ctx.k, of, 65536)
            if not data:
                break
            if plain:
                ctx.w(data)
                yield from ctx.maybe_flush()
                continue
            text = dec(data)
            out = []
            for piece in re.split(r"(\n)", text):
                if piece == "":
                    continue
                if piece == "\n":
                    if pending_line_start:
                        # an empty line
                        blank = True
                        if squeeze and prev_blank:
                            continue
                        prev_blank = True
                        if num and not nonblank:
                            lineno += 1
                            out.append(f"{lineno:6d}\t")
                    else:
                        prev_blank = False
                    out.append("$\n" if ends else "\n")
                    pending_line_start = True
                    continue
                line = piece
                if nonprint:
                    line = _show_nonprinting(line)
                if tabs:
                    line = line.replace("\t", "^I")
                if pending_line_start and (num or nonblank):
                    lineno += 1
                    out.append(f"{lineno:6d}\t")
                pending_line_start = False
                prev_blank = False
                out.append(line)
            ctx.w("".join(out))
            yield from ctx.maybe_flush()
    return status


def _show_nonprinting(s: str) -> str:
    out = []
    for ch in enc(s):
        if ch >= 128:
            out.append("M-")
            ch -= 128
        if ch < 32 and ch not in (9, 10):
            out.append("^" + chr(ch + 64))
        elif ch == 127:
            out.append("^?")
        else:
            out.append(chr(ch))
    return "".join(out)


@util("tac")
def u_tac(ctx):
    ops = [x for x in ctx.argv[1:] if not x.startswith("-") or x == "-"]
    status = 0
    for p in ops or ["-"]:
        try:
            data = yield from ctx.read_path(p)
        except FSError as e:
            ctx.err(f"failed to open '{p}' for reading: {e.msg}")
            status = 1
            continue
        text = dec(data)
        parts = text.split("\n")
        if parts and parts[-1] == "":
            parts.pop()
            ctx.w("".join(x + "\n" for x in reversed(parts)))
        else:
            last = parts.pop() if parts else ""
            ctx.w(last + "".join(x + "\n" for x in reversed(parts))[0:] if False else last + "".join(
                "\n" + x for x in reversed(parts)) + ("\n" if parts else ""))
    return status


@util("nl")
def u_nl(ctx):
    a = ctx.argv[1:]
    style, width, sep, fmt, start, incr = "t", 6, "\t", "rn", 1, 1
    ops = []
    i = 0
    while i < len(a):
        x = a[i]
        m = re.fullmatch(r"-([bwsnvi])(.*)", x)
        if m and m.group(1) in "bwsnvi":
            val = m.group(2) or (a[i + 1] if i + 1 < len(a) else "")
            if not m.group(2):
                i += 1
            c = m.group(1)
            if c == "b":
                style = val
            elif c == "w":
                width = int(val)
            elif c == "s":
                sep = val
            elif c == "n":
                fmt = val
            elif c == "v":
                start = int(val)
            elif c == "i":
                incr = int(val)
        elif x.startswith("--body-numbering="):
            style = x.split("=", 1)[1]
        elif x.startswith("--number-width="):
            width = int(x.split("=", 1)[1])
        else:
            ops.append(x)
        i += 1
    status = 0
    n = start
    for p, data in (yield from read_inputs(ctx, ops)):
        if data is None:
            status = 1
            continue
        for line in split_lines(dec(data)):
            number = style == "a" or (style == "t" and line.strip() != "") or (style.startswith("p") and re.search(style[1:], line))
            if number:
                num = {"ln": f"{n:<{width}}", "rn": f"{n:>{width}}", "rz": f"{n:0{width}}"}.get(fmt, f"{n:>{width}}")
                ctx.w(f"{num}{sep}{line}\n")
                n += incr
            else:
                ctx.w(" " * (width + len(sep)) + line + "\n" if style != "t" or True else line + "\n")
        yield from ctx.maybe_flush()
    return status


@util("rev")
def u_rev(ctx):
    status = 0
    for p, data in (yield from read_inputs(ctx, ctx.argv[1:])):
        if data is None:
            status = 1
            continue
        text = dec(data)
        ctx.w("".join(line[::-1] + "\n" for line in split_lines(text)))
    return status


# ---------------------------------------------------------------------------- head tail

def _num_arg(v: str, name: str):
    m = re.fullmatch(r"([+-]?)(\d+)([bkKMG]?B?|[kKMG]?i?B?)", v)
    if not m:
        raise ValueError(v)
    n = int(m.group(2))
    mult = {"": 1, "b": 512, "k": 1024, "K": 1024, "kB": 1000, "KB": 1000, "M": 1 << 20, "MB": 10 ** 6,
            "G": 1 << 30, "GB": 10 ** 9, "KiB": 1024, "MiB": 1 << 20, "GiB": 1 << 30}.get(m.group(3), 1)
    return m.group(1), n * mult


def _legacy_count(args):
    """head -5 / tail -5 / tail +5 -> -n 5."""
    out = []
    for i, x in enumerate(args):
        if i == 0 and re.fullmatch(r"-\d+[lcb]?", x):
            out += ["-c" if x.endswith(("c", "b")) else "-n", x[1:].rstrip("lcb")]
        elif i == 0 and re.fullmatch(r"\+\d+", x):
            out += ["-n", x]
        else:
            out.append(x)
    return out


@util("head")
def u_head(ctx):
    args = _legacy_count(ctx.argv[1:])
    try:
        opts, ops = gnu_getopt(args, "c:n:qvz", ["bytes=", "lines=", "quiet", "silent", "verbose", "zero-terminated",
                                                "help", "version"], name="head")
    except UsageError as e:
        ctx.err(e.msg)
        return ctx.usage()
    mode, count, neg = "lines", 10, False
    quiet = verbose = False
    for k, v in opts:
        if k in ("-n", "--lines", "-c", "--bytes"):
            try:
                sign, count = _num_arg(v, "head")
            except ValueError:
                ctx.err(f"invalid number of {'lines' if k in ('-n', '--lines') else 'bytes'}: '{v}'")
                return 1
            neg = sign == "-"
            mode = "lines" if k in ("-n", "--lines") else "bytes"
        elif k in ("-q", "--quiet", "--silent"):
            quiet = True
        elif k in ("-v", "--verbose"):
            verbose = True
    files = ops or ["-"]
    status = 0
    for i, p in enumerate(files):
        try:
            of = ctx.proc.fds.get(0) if p == "-" else ctx.open(p)
            if of.kind == "dir":
                ctx.err(f"error reading '{p}': Is a directory")
                status = 1
                continue
        except FSError as e:
            ctx.err(f"cannot open '{p}' for reading: {e.msg}")
            status = 1
            continue
        if (len(files) > 1 and not quiet) or verbose:
            ctx.w(("\n" if i else "") + f"==> {'standard input' if p == '-' else p} <==\n")
        if neg:
            data = yield from _read_all(ctx, of)
            if mode == "bytes":
                ctx.w(data[:max(len(data) - count, 0)])
            else:
                lines = data.split(b"\n")
                trailing = data.endswith(b"\n")
                if trailing:
                    lines.pop()
                keep = lines[:max(len(lines) - count, 0)]
                ctx.w(b"".join(x + b"\n" for x in keep))
            continue
        if mode == "bytes":
            got = bytearray()
            while len(got) < count:
                data = yield from read(ctx.k, of, min(65536, count - len(got)))
                if not data:
                    break
                got += data
            ctx.w(bytes(got))
            continue
        n = 0
        if of.kind == "file":
            node = of.inode
            pos = of.pos
            data = node.data
            end = pos
            while n < count:
                j = data.find(b"\n", end)
                if j < 0:
                    end = len(data)
                    break
                end = j + 1
                n += 1
            ctx.w(bytes(data[pos:end]))
            of.pos = end
            continue
        while n < count:
            line = yield from readline(ctx.k, of)
            if not line:
                break
            ctx.w(line)
            n += 1
            if n % 256 == 0:
                yield from ctx.maybe_flush()
    return status


def _read_all(ctx, of):
    chunks = []
    while True:
        data = yield from read(ctx.k, of, -1)
        if not data:
            return b"".join(chunks)
        chunks.append(data)


@util("tail")
def u_tail(ctx):
    args = _legacy_count(ctx.argv[1:])
    try:
        opts, ops = gnu_getopt(args, "c:fFn:qs:vz", ["bytes=", "follow=?", "lines=", "quiet", "silent", "verbose",
                                                    "retry", "sleep-interval=", "pid=", "zero-terminated", "help",
                                                    "version"], name="tail")
    except UsageError as e:
        ctx.err(e.msg)
        return ctx.usage()
    mode, count, from_start = "lines", 10, False
    follow = quiet = verbose = False
    interval = 1.0
    pid = None
    for k, v in opts:
        if k in ("-n", "--lines", "-c", "--bytes"):
            try:
                sign, count = _num_arg(v, "tail")
            except ValueError:
                ctx.err(f"invalid number of {'lines' if k in ('-n', '--lines') else 'bytes'}: '{v}'")
                return 1
            from_start = sign == "+"
            mode = "lines" if k in ("-n", "--lines") else "bytes"
        elif k in ("-f", "--follow", "-F"):
            follow = True
        elif k in ("-q", "--quiet", "--silent"):
            quiet = True
        elif k in ("-v", "--verbose"):
            verbose = True
        elif k in ("-s", "--sleep-interval"):
            interval = float(v)
        elif k == "--pid":
            pid = int(v)
    files = ops or ["-"]
    status = 0
    sizes = {}
    multi = (len(files) > 1 and not quiet) or verbose
    last_header = None
    for i, p in enumerate(files):
        try:
            of = ctx.proc.fds.get(0) if p == "-" else ctx.open(p)
            if of.kind == "dir":
                ctx.err(f"error reading '{p}': Is a directory")
                status = 1
                continue
        except FSError as e:
            ctx.err(f"cannot open '{p}' for reading: {e.msg}")
            status = 1
            continue
        data = yield from _read_all(ctx, of)
        if multi:
            ctx.w(("\n" if i else "") + f"==> {'standard input' if p == '-' else p} <==\n")
            last_header = p
        ctx.w(_tail_select(data, mode, count, from_start))
        if of.kind == "file":
            sizes[p] = (of.inode, len(of.inode.data))
    if not follow or not sizes:
        return status
    # follow: print what is appended until Ctrl-C (foreground) or until killed
    yield from ctx.flush()
    k = ctx.k
    tty = ctx.world.tty_of(ctx.proc)
    quiet_for = 0.0
    limit = ctx.world.ctrl_c_after
    fg = ctx.foreground()
    while True:
        yield ("sleep", k.now + int(interval * NS))
        grew = False
        for p, (node, size) in list(sizes.items()):
            if p != "-":
                try:
                    cur = ctx.fs.lookup(p, ctx.cwd)
                except FSError:
                    continue
                if cur is not node:     # rotated (-F reopens)
                    node, size = cur, 0
            n = len(node.data)
            if n > size:
                if multi and last_header != p:
                    ctx.w(f"\n==> {p} <==\n")
                    last_header = p
                ctx.w(bytes(node.data[size:n]))
                grew = True
            elif n < size:
                ctx.e(f"tail: {p}: file truncated\n")
            sizes[p] = (node, n)
        yield from ctx.flush()
        quiet_for = 0.0 if grew else quiet_for + interval
        if pid is not None and pid not in ctx.host.procs:
            return 0
        if fg and quiet_for >= limit:
            ctx.world.tty.write("echo", b"^C\n")
            return 130


def _tail_select(data: bytes, mode, count, from_start):
    if mode == "bytes":
        if from_start:
            return data[max(count - 1, 0):]
        return data[-count:] if count else b""
    if from_start:
        lines = data.split(b"\n")
        idx = max(count - 1, 0)
        # keep the original terminators
        pos = 0
        for _ in range(idx):
            j = data.find(b"\n", pos)
            if j < 0:
                return b""
            pos = j + 1
        return data[pos:]
    if count == 0:
        return b""
    end = len(data)
    pos = end
    if data.endswith(b"\n"):
        pos -= 1
    n = 0
    while pos > 0:
        j = data.rfind(b"\n", 0, pos)
        n += 1
        if n == count:
            return data[j + 1:] if j >= 0 else data
        if j < 0:
            return data
        pos = j
    return data


# ---------------------------------------------------------------------------- wc

@util("wc")
def u_wc(ctx):
    try:
        opts, ops = gnu_getopt(ctx.argv[1:], "clLmw", ["bytes", "chars", "lines", "max-line-length", "words",
                                                      "files0-from=", "total=", "help", "version"], name="wc")
    except UsageError as e:
        ctx.err(e.msg)
        return ctx.usage()
    want = set()
    total_mode = "auto"
    for k, v in opts:
        want.add({"-c": "c", "--bytes": "c", "-m": "m", "--chars": "m", "-l": "l", "--lines": "l",
                  "-L": "L", "--max-line-length": "L", "-w": "w", "--words": "w"}.get(k, ""))
        if k == "--total":
            total_mode = v
    want.discard("")
    if not want:
        want = {"l", "w", "c"}
    order = [c for c in "lwmcL" if c in want]
    files = ops or ["-"]
    results = []
    status = 0
    regular_total = 0
    any_nonreg = False
    for p in files:
        try:
            if p == "-":
                of = ctx.proc.fds.get(0)
                data = yield from _read_all(ctx, of)
                if of.kind != "file":
                    any_nonreg = True
                else:
                    regular_total += len(of.inode.data)
            else:
                node = ctx.lookup(p)
                if node.type == "d":
                    ctx.err(f"{p}: Is a directory")
                    results.append((p, None))
                    status = 1
                    continue
                data = yield from ctx.read_path(p)
                if node.type == "f":
                    regular_total += len(data)
                else:
                    any_nonreg = True
        except FSError as e:
            ctx.err(f"{p}: {e.msg}")
            status = 1
            continue
        text = dec(data)
        counts = {"l": data.count(b"\n"), "w": len(text.split()), "c": len(data), "m": len(text),
                  "L": max((len(x.expandtabs(8)) for x in text.split("\n")), default=0)}
        results.append((p, counts))
    good = [r for r in results if r[1] is not None]
    width = 1
    if len(order) == 1 and len(files) == 1:
        width = 1
    else:
        t = regular_total
        while t >= 10:
            width += 1
            t //= 10
        if any_nonreg:
            width = max(width, 7)
    out = []
    for p, c in results:
        if c is None:
            continue
        vals = " ".join(f"{c[k]:>{width}}" for k in order)
        out.append(vals + ("" if p == "-" and not ops else f" {p}"))
    show_total = (total_mode == "always") or (total_mode == "auto" and len(files) > 1)
    if total_mode == "only":
        tot = {k: sum(c[k] for _, c in good) if k != "L" else max((c[k] for _, c in good), default=0) for k in order}
        out = [" ".join(str(tot[k]) for k in order)]
    elif show_total:
        tot = {k: sum(c[k] for _, c in good) if k != "L" else max((c[k] for _, c in good), default=0) for k in order}
        out.append(" ".join(f"{tot[k]:>{width}}" for k in order) + " total")
    ctx.w("".join(x + "\n" for x in out))
    return status


# ---------------------------------------------------------------------------- sort

MONTHS = {m: i + 1 for i, m in enumerate("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split())}
SUFFIX = {"": 0, "K": 1, "k": 1, "M": 2, "G": 3, "T": 4, "P": 5, "E": 6, "Z": 7, "Y": 8}


class KeySpec:
    def __init__(self, f1, c1, f2, c2, opts):
        self.f1, self.c1, self.f2, self.c2 = f1, c1, f2, c2
        self.opts = opts   # set of letters

    @property
    def reverse(self):
        return "r" in self.opts


def parse_key(spec: str, global_opts: set):
    m = re.fullmatch(r"(\d+)(?:\.(\d+))?([bdfgiMhnRrV]*)(?:,(\d+)(?:\.(\d+))?([bdfgiMhnRrV]*))?", spec)
    if not m:
        raise ValueError(spec)
    f1 = int(m.group(1))
    if f1 == 0:
        raise ValueError(spec)
    c1 = int(m.group(2)) if m.group(2) else 1
    f2 = int(m.group(4)) if m.group(4) else None
    c2 = int(m.group(5)) if m.group(5) else 0
    opts = set(m.group(3) or "") | set(m.group(6) or "")
    b_start = "b" in (m.group(3) or "")
    b_end = "b" in (m.group(6) or "")
    if not opts:
        opts = set(global_opts)
        b_start = b_end = "b" in opts
    k = KeySpec(f1, c1, f2, c2, opts)
    k.b_start, k.b_end = b_start, b_end
    return k


def _fields(line: str, sep):
    """(start, end) spans of fields; without -t a field includes its leading blanks."""
    spans = []
    if sep is not None:
        pos = 0
        while True:
            j = line.find(sep, pos)
            if j < 0:
                spans.append((pos, len(line)))
                break
            spans.append((pos, j))
            pos = j + 1
        return spans
    i, n = 0, len(line)
    while i < n or not spans:
        start = i
        while i < n and line[i] in " \t":
            i += 1
        while i < n and line[i] not in " \t":
            i += 1
        spans.append((start, i))
        if i >= n:
            break
    return spans


def key_text(line: str, k: KeySpec, sep):
    spans = _fields(line, sep)
    n = len(line)
    if k.f1 - 1 >= len(spans):
        return ""
    s0, e0 = spans[k.f1 - 1]
    start = s0
    if k.b_start:
        while start < e0 and line[start] in " \t":
            start += 1
    start = min(start + k.c1 - 1, e0 if k.c1 > 1 else start)
    if k.c1 > 1:
        start = min(s0 + (start - s0), n)
    if k.f2 is None:
        end = n
    else:
        if k.f2 - 1 >= len(spans):
            end = n
        else:
            s2, e2 = spans[k.f2 - 1]
            if k.c2 == 0:
                end = e2
            else:
                b = s2
                if k.b_end:
                    while b < e2 and line[b] in " \t":
                        b += 1
                end = min(b + k.c2, e2)
    if end < start:
        return ""
    return line[start:end]


_NUM = re.compile(r"[ \t]*(-?)(\d*)(?:\.(\d*))?")


def num_value(s: str):
    m = _NUM.match(s)
    if not m or not (m.group(2) or m.group(3)):
        return 0.0
    v = float((m.group(2) or "0") + "." + (m.group(3) or "0"))
    return -v if m.group(1) else v


def general_value(s: str):
    m = re.match(r"[ \t]*([+-]?(?:inf(?:inity)?|nan|(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?))", s, re.I)
    if not m:
        return (0, 0.0)   # non-numbers sort first
    v = float(m.group(1))
    if v != v:
        return (1, 0.0)
    return (2, v)


def human_value(s: str):
    m = re.match(r"[ \t]*(-?)(\d*)(?:\.(\d*))?([KkMGTPEZY]?)", s)
    if not m or not (m.group(2) or m.group(3)):
        return (0, 0.0)
    v = float((m.group(2) or "0") + "." + (m.group(3) or "0"))
    neg = bool(m.group(1))
    rank = SUFFIX.get(m.group(4), 0)
    if neg:
        return (-rank, -v)
    return (rank, v) if v else (0, 0.0)


def version_value(s: str):
    parts = re.findall(r"\d+|\D+", s)
    return [(0, int(p), "") if p.isdigit() else (1, 0, p) for p in parts]


def make_keyfunc(opts: set, sep, keyspec=None):
    def transform(t: str):
        if "d" in opts:
            t = "".join(c for c in t if c.isalnum() or c in " \t")
        if "i" in opts:
            t = "".join(c for c in t if 32 <= ord(c) < 127)
        if "f" in opts:
            t = t.upper()
        return t

    def f(line):
        t = key_text(line, keyspec, sep) if keyspec is not None else line
        if "b" in opts and keyspec is None:
            t = t.lstrip(" \t")
        if "n" in opts:
            return num_value(t)
        if "g" in opts:
            return general_value(t)
        if "h" in opts:
            return human_value(t)
        if "M" in opts:
            return MONTHS.get(t.strip()[:3].upper(), 0)
        if "V" in opts:
            return version_value(t)
        return enc(transform(t))
    return f


@util("sort")
def u_sort(ctx):
    try:
        opts, ops = gnu_getopt(ctx.argv[1:], "bcCdfghik:mMno:rRsS:t:T:uVz", [
            "ignore-leading-blanks", "check=?", "dictionary-order", "ignore-case", "general-numeric-sort",
            "human-numeric-sort", "ignore-nonprinting", "key=", "merge", "month-sort", "numeric-sort", "output=",
            "reverse", "random-sort", "stable", "buffer-size=", "field-separator=", "temporary-directory=", "unique",
            "version-sort", "zero-terminated", "parallel=", "random-source=", "sort=", "help", "version",
            "debug", "files0-from="], name="sort")
    except UsageError as e:
        ctx.err(e.msg)
        return ctx.usage(status=2)
    g = set()
    keys_raw = []
    sep = None
    out_path = None
    unique = stable = check = random_ = False
    zero = False
    for k, v in opts:
        letter = {"-b": "b", "--ignore-leading-blanks": "b", "-d": "d", "--dictionary-order": "d", "-f": "f",
                  "--ignore-case": "f", "-g": "g", "--general-numeric-sort": "g", "-h": "h",
                  "--human-numeric-sort": "h", "-i": "i", "--ignore-nonprinting": "i", "-M": "M",
                  "--month-sort": "M", "-n": "n", "--numeric-sort": "n", "-r": "r", "--reverse": "r", "-V": "V",
                  "--version-sort": "V"}.get(k)
        if letter:
            g.add(letter)
        elif k in ("-k", "--key"):
            keys_raw.append(v)
        elif k in ("-t", "--field-separator"):
            if v == "\\t":
                v = "\t"
            if len(v) != 1:
                if v == "\\0":
                    v = "\0"
                else:
                    ctx.err("multi-character tab '" + v + "'")
                    return 2
            sep = v
        elif k in ("-o", "--output"):
            out_path = v
        elif k in ("-u", "--unique"):
            unique = True
        elif k in ("-s", "--stable"):
            stable = True
        elif k in ("-c", "-C", "--check"):
            check = k if k != "--check" else ("-C" if v in ("quiet", "silent") else "-c")
        elif k in ("-R", "--random-sort"):
            random_ = True
        elif k == "--sort":
            g.add({"numeric": "n", "general-numeric": "g", "human-numeric": "h", "month": "M", "version": "V"}.get(v, ""))
        elif k in ("-z", "--zero-terminated"):
            zero = True
        elif k == "--help":
            ctx.w("Usage: sort [OPTION]... [FILE]...\nWrite sorted concatenation of all FILE(s) to standard output.\n")
            return 0
    g.discard("")
    try:
        keys = [parse_key(x, g) for x in keys_raw]
    except ValueError as e:
        ctx.err(f"invalid number at field start: invalid count at start of '{e.args[0]}'")
        return 2
    lines = []
    status = 0
    for p, data in (yield from read_inputs(ctx, ops)):
        if data is None:
            return 2
        text = dec(data)
        lines += (text.split("\0")[:-1] if text.endswith("\0") else text.split("\0")) if zero else split_lines(text)
    nl = "\0" if zero else "\n"
    reverse_all = "r" in g
    funcs = [(make_keyfunc(k.opts, sep, k), k.reverse) for k in keys] if keys else [(make_keyfunc(g, sep), reverse_all)]
    if check:
        for i in range(1, len(lines)):
            a, b = lines[i - 1], lines[i]
            if _cmp_lines(a, b, funcs, stable or unique, reverse_all, keys) > (0 if not unique else -1):
                if check == "-c":
                    ctx.err(f"{ops[0] if ops else '-'}:{i + 1}: disorder: {b}")
                return 1
        return 0
    if random_:
        rng = ctx.world.rng
        lines = sorted(lines, key=lambda x: (hashlib.md5(enc(x) + str(rng.random()).encode()).digest()))
    else:
        if not stable and not unique:
            lines.sort(key=enc, reverse=reverse_all)
        for fn, rev in reversed(funcs):
            lines.sort(key=fn, reverse=rev)
    if unique:
        out = []
        prev_keys = None
        for ln in lines:
            ks = tuple(fn(ln) for fn, _ in funcs)
            if ks != prev_keys:
                out.append(ln)
                prev_keys = ks
        lines = out
    text = "".join(ln + nl for ln in lines)
    if out_path:
        try:
            of = ctx.open(out_path, "w")
            from ..fileio import write
            yield from write(ctx.k, of, enc(text))
        except FSError as e:
            ctx.err(f"open failed: {out_path}: {e.msg}")
            return 2
        return status
    ctx.w(text)
    return status


def _cmp_lines(a, b, funcs, no_last, reverse_all, keys):
    for fn, rev in funcs:
        x, y = fn(a), fn(b)
        if x != y:
            r = -1 if x < y else 1
            return -r if rev else r
    if no_last:
        return 0
    if a == b:
        return 0
    r = -1 if enc(a) < enc(b) else 1
    return -r if reverse_all else r


# ---------------------------------------------------------------------------- uniq

@util("uniq")
def u_uniq(ctx):
    a = _legacy_uniq(ctx.argv[1:])
    try:
        opts, ops = gnu_getopt(a, "cdDf:is:uw:z", ["count", "repeated", "all-repeated=?", "skip-fields=",
                                                  "ignore-case", "skip-chars=", "unique", "check-chars=",
                                                  "zero-terminated", "group=?", "help", "version"], name="uniq")
    except UsageError as e:
        ctx.err(e.msg)
        return ctx.usage()
    count = dup = alldup = uniq_only = icase = False
    skipf = skipc = 0
    width = None
    for k, v in opts:
        if k in ("-c", "--count"):
            count = True
        elif k in ("-d", "--repeated"):
            dup = True
        elif k in ("-D", "--all-repeated"):
            alldup = True
        elif k in ("-u", "--unique"):
            uniq_only = True
        elif k in ("-i", "--ignore-case"):
            icase = True
        elif k in ("-f", "--skip-fields"):
            skipf = int(v)
        elif k in ("-s", "--skip-chars"):
            skipc = int(v)
        elif k in ("-w", "--check-chars"):
            width = int(v)
    if len(ops) > 2:
        ctx.err(f"extra operand '{ops[2]}'")
        return ctx.usage()
    src = ops[0] if ops else "-"
    try:
        data = yield from ctx.read_path(src)
    except FSError as e:
        ctx.err(f"{src}: {e.msg}")
        return 1
    lines = split_lines(dec(data))

    def key(line):
        s = line
        for _ in range(skipf):
            s = re.sub(r"^[ \t]*[^ \t]*", "", s, count=1)
        s = s[skipc:]
        if width is not None:
            s = s[:width]
        return s.lower() if icase else s
    out = []
    i = 0
    while i < len(lines):
        j = i + 1
        k0 = key(lines[i])
        while j < len(lines) and key(lines[j]) == k0:
            j += 1
        n = j - i
        if alldup:
            if n > 1:
                out += lines[i:j]
        elif (dup and n > 1) or (uniq_only and n == 1) or (not dup and not uniq_only):
            out.append(f"{n:7d} {lines[i]}" if count else lines[i])
        i = j
    text = "".join(x + "\n" for x in out)
    if len(ops) == 2:
        try:
            of = ctx.open(ops[1], "w")
            from ..fileio import write
            yield from write(ctx.k, of, enc(text))
        except FSError as e:
            ctx.err(f"{ops[1]}: {e.msg}")
            return 1
    else:
        ctx.w(text)
    return 0


def _legacy_uniq(a):
    out = []
    for x in a:
        if re.fullmatch(r"-\d+", x):
            out += ["-f", x[1:]]
        elif re.fullmatch(r"\+\d+", x):
            out += ["-s", x[1:]]
        else:
            out.append(x)
    return out


# ---------------------------------------------------------------------------- cut

def parse_list(spec: str):
    ranges = []
    for part in spec.split(","):
        if not part:
            raise ValueError("fields and positions are numbered from 1")
        m = re.fullmatch(r"(\d*)-(\d*)|(\d+)", part)
        if not m:
            raise ValueError(f"invalid field value '{part}'" if not part.replace("-", "").isdigit() else "invalid range")
        if m.group(3):
            n = int(m.group(3))
            if n == 0:
                raise ValueError("fields and positions are numbered from 1")
            ranges.append((n, n))
        else:
            lo = int(m.group(1)) if m.group(1) else 1
            hi = int(m.group(2)) if m.group(2) else 10 ** 9
            if not m.group(1) and not m.group(2):
                raise ValueError("invalid range with no endpoint: -")
            if lo == 0:
                raise ValueError("fields and positions are numbered from 1")
            if hi < lo:
                raise ValueError("invalid decreasing range")
            ranges.append((lo, hi))
    return ranges


def in_list(i, ranges):
    return any(lo <= i <= hi for lo, hi in ranges)


@util("cut")
def u_cut(ctx):
    try:
        opts, ops = gnu_getopt(ctx.argv[1:], "b:c:d:f:nsz", ["bytes=", "characters=", "delimiter=", "fields=",
                                                            "only-delimited", "complement", "output-delimiter=",
                                                            "zero-terminated", "help", "version"], name="cut")
    except UsageError as e:
        ctx.err(e.msg)
        return ctx.usage()
    mode = spec = None
    delim = "\t"
    only = complement = False
    outd = None
    for k, v in opts:
        if k in ("-b", "--bytes", "-c", "--characters", "-f", "--fields"):
            if mode is not None:
                ctx.err("only one list may be specified")
                return ctx.usage()
            mode = {"-b": "b", "--bytes": "b", "-c": "c", "--characters": "c", "-f": "f", "--fields": "f"}[k]
            spec = v
        elif k in ("-d", "--delimiter"):
            if len(v) != 1:
                ctx.err("the delimiter must be a single character")
                return ctx.usage()
            delim = v
        elif k in ("-s", "--only-delimited"):
            only = True
        elif k == "--complement":
            complement = True
        elif k == "--output-delimiter":
            outd = v
    if mode is None:
        ctx.err("you must specify a list of bytes, characters, or fields")
        return ctx.usage()
    if mode != "f" and any(k in ("-d", "--delimiter") for k, _ in opts):
        ctx.err("an input delimiter may be specified only when operating on fields")
        return ctx.usage()
    try:
        ranges = parse_list(spec)
    except ValueError as e:
        ctx.err(str(e))
        return ctx.usage()
    status = 0
    for p, data in (yield from read_inputs(ctx, ops)):
        if data is None:
            status = 1
            continue
        out = []
        for line in split_lines(dec(data)):
            if mode == "f":
                if delim not in line:
                    if not only:
                        out.append(line)
                    continue
                parts = line.split(delim)
                sel = [x for i, x in enumerate(parts, 1) if in_list(i, ranges) != complement]
                out.append((outd if outd is not None else delim).join(sel))
            else:
                chars = line if mode == "c" else line
                if outd is not None and not complement:
                    pieces, cur = [], []
                    prev = False
                    for i, ch in enumerate(chars, 1):
                        sel = in_list(i, ranges)
                        if sel:
                            if not prev and cur:
                                pieces.append("".join(cur))
                                cur = []
                            cur.append(ch)
                        prev = sel if sel else prev and False
                    if cur:
                        pieces.append("".join(cur))
                    out.append(outd.join(pieces))
                else:
                    out.append("".join(ch for i, ch in enumerate(chars, 1) if in_list(i, ranges) != complement))
        ctx.w("".join(x + "\n" for x in out))
        yield from ctx.maybe_flush()
    return status


# ---------------------------------------------------------------------------- tr

TR_CLASSES = {
    "alpha": [chr(c) for c in range(65, 91)] + [chr(c) for c in range(97, 123)],
    "upper": [chr(c) for c in range(65, 91)], "lower": [chr(c) for c in range(97, 123)],
    "digit": [chr(c) for c in range(48, 58)],
    "alnum": [chr(c) for c in range(48, 58)] + [chr(c) for c in range(65, 91)] + [chr(c) for c in range(97, 123)],
    "space": list(" \t\n\r\f\v"), "blank": list(" \t"),
    "punct": [chr(c) for c in range(33, 127) if not chr(c).isalnum()],
    "print": [chr(c) for c in range(32, 127)], "graph": [chr(c) for c in range(33, 127)],
    "cntrl": [chr(c) for c in range(32)] + [chr(127)], "xdigit": list("0123456789ABCDEFabcdef"),
}


class TrError(ValueError):
    pass


def tr_expand(s: str, is_set2=False, fill_to=None):
    """Expand a tr SET into a list of characters (and markers for [c*])."""
    out = []
    i, n = 0, len(s)
    classes = []

    def esc(j):
        c = s[j + 1] if j + 1 < n else "\\"
        if c in "01234567":
            k = j + 1
            while k < n and k < j + 4 and s[k] in "01234567":
                k += 1
            return chr(int(s[j + 1:k], 8)), k
        m = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", "a": "\a", "b": "\b", "f": "\f", "v": "\v"}
        if j + 1 >= n:
            return "\\", j + 1
        return m.get(c, c), j + 2
    while i < n:
        if s.startswith("[:", i):
            j = s.find(":]", i + 2)
            if j > 0:
                name = s[i + 2:j]
                if name not in TR_CLASSES:
                    raise TrError(f"invalid character class '{name}'")
                classes.append((len(out), name))
                out.extend(TR_CLASSES[name])
                i = j + 2
                continue
        if s.startswith("[=", i) and s.find("=]", i + 2) == i + 3:
            out.append(s[i + 2])
            i += 5
            continue
        if s[i] == "[" and is_set2:
            m = re.match(r"\[(\\?.)\*(\d*)\]", s[i:])
            if m:
                ch = m.group(1)
                ch = esc(0)[0] if False else (ch[1] if ch.startswith("\\") and len(ch) == 2 else ch)
                cnt = m.group(2)
                out.append(("REPEAT", ch, int(cnt, 8 if cnt.startswith("0") else 10) if cnt else None))
                i += m.end()
                continue
        if s[i] == "\\":
            c, i = esc(i)
        else:
            c = s[i]
            i += 1
        if i + 1 < n and s[i] == "-" and not (i + 1 < n and s[i + 1] == "\\" and False):
            hi_c = s[i + 1]
            j = i + 2
            if hi_c == "\\":
                hi_c, j = esc(i + 1)
            if ord(hi_c) < ord(c):
                raise TrError(f"range-endpoints of '{c}-{hi_c}' are in reverse collating sequence order")
            out.extend(chr(x) for x in range(ord(c), ord(hi_c) + 1))
            i = j
            continue
        out.append(c)
    return out, classes


@util("tr")
def u_tr(ctx):
    a = ctx.argv[1:]
    flags = set()
    sets = []
    for x in a:
        if x.startswith("-") and len(x) > 1 and not sets and x != "-":
            if x == "--":
                continue
            for c in x[1:]:
                if c not in "cCdst":
                    ctx.err(f"invalid option -- '{c}'")
                    return ctx.usage()
                flags.add("c" if c == "C" else c)
        else:
            sets.append(x)
    delete, squeeze, comp, trunc = "d" in flags, "s" in flags, "c" in flags, "t" in flags
    if not sets:
        ctx.err("missing operand")
        return ctx.usage()
    if delete and not squeeze and len(sets) > 1:
        ctx.err(f"extra operand '{sets[1]}'")
        ctx.e("Only one string may be given when deleting without squeezing repeats.\n")
        ctx.e(f"Try '{ctx.name} --help' for more information.\n")
        return 1
    if not delete and not squeeze and len(sets) < 2:
        ctx.err(f"missing operand after '{sets[0]}'")
        ctx.e("Two strings must be given when translating.\n")
        ctx.e(f"Try '{ctx.name} --help' for more information.\n")
        return 1
    if len(sets) > 2:
        ctx.err(f"extra operand '{sets[2]}'")
        return ctx.usage()
    try:
        s1, cls1 = tr_expand(sets[0])
        s2, cls2 = tr_expand(sets[1], True) if len(sets) > 1 else ([], [])
    except TrError as e:
        ctx.err(str(e))
        return 1
    data = yield from ctx.read_fd(0)
    text = dec(data)
    set1 = set(s1)
    if comp:
        allchars = sorted(set(text) | set(chr(c) for c in range(256)))
        s1 = [c for c in allchars if c not in set1]
        set1 = set(s1)
    out = []
    if delete:
        text = "".join(c for c in text if c not in set1)
        if squeeze and len(sets) > 1:
            sq = set(x for x in s2 if isinstance(x, str))
            text = _squeeze(text, sq)
        ctx.w(text)
        return 0
    if len(sets) == 1:   # squeeze only
        ctx.w(_squeeze(text, set1))
        return 0
    # expand repeats in set2
    flat = []
    for x in s2:
        if isinstance(x, tuple):
            flat.append(x)
        else:
            flat.append(x)
    fixed = sum(1 for x in flat if not isinstance(x, tuple)) + sum(x[2] or 0 for x in flat if isinstance(x, tuple))
    s2x = []
    for x in flat:
        if isinstance(x, tuple):
            cnt = x[2] if x[2] else max(len(s1) - fixed, 0)
            s2x.extend([x[1]] * cnt)
        else:
            s2x.append(x)
    # case conversion with classes: [:lower:] -> [:upper:]
    if cls1 and cls2 and len(cls1) == 1 and len(cls2) == 1 and {cls1[0][1], cls2[0][1]} <= {"lower", "upper"}:
        pass
    if trunc:
        s1 = s1[:len(s2x)]
    elif len(s2x) < len(s1):
        if not s2x:
            ctx.err("when not truncating set1, string2 must be non-empty")
            return 1
        s2x = s2x + [s2x[-1]] * (len(s1) - len(s2x))
    table = {}
    for c1, c2 in zip(s1, s2x):
        table[c1] = c2
    if comp:
        last = s2x[-1] if s2x else ""
        table = {c: last for c in s1}
    text = "".join(table.get(c, c) for c in text)
    if squeeze:
        text = _squeeze(text, set(s2x))
    ctx.w(text)
    return 0


def _squeeze(text, chars):
    out = []
    prev = None
    for c in text:
        if c == prev and c in chars:
            continue
        out.append(c)
        prev = c
    return "".join(out)


# ---------------------------------------------------------------------------- paste join split comm

def _paste_delims(spec):
    out = []
    i = 0
    while i < len(spec):
        c = spec[i]
        if c == "\\" and i + 1 < len(spec):
            d = spec[i + 1]
            out.append({"n": "\n", "t": "\t", "\\": "\\", "0": ""}.get(d, d))
            i += 2
        else:
            out.append(c)
            i += 1
    return out or [""]


@util("paste")
def u_paste(ctx):
    try:
        opts, ops = gnu_getopt(ctx.argv[1:], "d:sz", ["delimiters=", "serial", "zero-terminated", "help", "version"],
                               name="paste")
    except UsageError as e:
        ctx.err(e.msg)
        return ctx.usage()
    delims = ["\t"]
    serial = False
    for k, v in opts:
        if k in ("-d", "--delimiters"):
            delims = _paste_delims(v)
        elif k in ("-s", "--serial"):
            serial = True
    files = ops or ["-"]
    contents = []
    stdin_lines = None
    for p in files:
        if p == "-":
            if stdin_lines is None:
                data = yield from ctx.read_fd(0)
                stdin_lines = split_lines(dec(data))
            contents.append("-")
        else:
            try:
                data = yield from ctx.read_path(p)
            except FSError as e:
                ctx.err(f"{p}: {e.msg}")
                return 1
            contents.append(split_lines(dec(data)))
    out = []
    if serial:
        for c in contents:
            lines = stdin_lines if c == "-" else c
            if c == "-":
                stdin_lines = []
            s = ""
            for i, ln in enumerate(lines):
                if i:
                    s += delims[(i - 1) % len(delims)]
                s += ln
            out.append(s)
    else:
        # several "-" operands take turns reading stdin lines
        cols = []
        n_dash = contents.count("-")
        for idx, c in enumerate(contents):
            if c == "-":
                k = sum(1 for x in contents[:idx] if x == "-")
                cols.append(stdin_lines[k::n_dash] if stdin_lines else [])
            else:
                cols.append(c)
        rows = max((len(c) for c in cols), default=0)
        for r in range(rows):
            s = ""
            for i, c in enumerate(cols):
                if i:
                    s += delims[(i - 1) % len(delims)]
                s += c[r] if r < len(c) else ""
            out.append(s)
    ctx.w("".join(x + "\n" for x in out))
    return 0


@util("join")
def u_join(ctx):
    try:
        opts, ops = gnu_getopt(ctx.argv[1:], "a:e:i1:2:j:o:t:v:z", ["ignore-case", "check-order", "nocheck-order",
                                                                   "header", "zero-terminated", "help", "version"],
                               name="join")
    except UsageError as e:
        ctx.err(e.msg)
        return ctx.usage()
    f1 = f2 = 1
    sep = None
    unpaired = set()
    only = set()
    empty = None
    fmt = None
    icase = header = False
    check = "default"
    for k, v in opts:
        if k == "-1":
            f1 = int(v)
        elif k == "-2":
            f2 = int(v)
        elif k == "-j":
            f1 = f2 = int(v)
        elif k == "-t":
            sep = v
        elif k == "-a":
            unpaired.add(int(v))
        elif k == "-v":
            only.add(int(v))
        elif k == "-e":
            empty = v
        elif k == "-o":
            fmt = v
        elif k in ("-i", "--ignore-case"):
            icase = True
        elif k == "--header":
            header = True
        elif k == "--check-order":
            check = "strict"
        elif k == "--nocheck-order":
            check = "none"
    if len(ops) != 2:
        ctx.err("missing operand" if len(ops) < 2 else f"extra operand '{ops[2]}'")
        return ctx.usage()
    data = []
    for p in ops:
        try:
            d = yield from ctx.read_path(p)
        except FSError as e:
            ctx.err(f"{p}: {e.msg}")
            return 1
        data.append(split_lines(dec(d)))

    def split(line):
        if sep is not None:
            return line.split(sep)
        return line.split()

    def key(fields, f):
        k = fields[f - 1] if f - 1 < len(fields) else ""
        return k.lower() if icase else k
    A = [split(x) for x in data[0]]
    B = [split(x) for x in data[1]]
    outsep = sep if sep is not None else " "
    out = []
    if header and A and B:
        out.append(_join_line(A[0], B[0], f1, f2, fmt, empty, outsep, True, True))
        A, B = A[1:], B[1:]
    i = j = 0
    warned = False
    status = 0

    def disorder(lst, idx, f, which):
        nonlocal warned, status
        if check == "none" or warned:
            return
        if idx > 0 and enc(key(lst[idx], f)) < enc(key(lst[idx - 1], f)):
            ctx.err(f"{ops[which - 1]}:{idx + 1 + (1 if header else 0)}: is not sorted: {(sep or ' ').join(lst[idx])}")
            warned = True
            status = 1
    while i < len(A) or j < len(B):
        if i < len(A):
            disorder(A, i, f1, 1)
        if j < len(B):
            disorder(B, j, f2, 2)
        if i >= len(A):
            if 2 in unpaired or 2 in only:
                out.append(_join_line(None, B[j], f1, f2, fmt, empty, outsep, False, True))
            j += 1
            continue
        if j >= len(B):
            if 1 in unpaired or 1 in only:
                out.append(_join_line(A[i], None, f1, f2, fmt, empty, outsep, True, False))
            i += 1
            continue
        ka, kb = enc(key(A[i], f1)), enc(key(B[j], f2))
        if ka < kb:
            if 1 in unpaired or 1 in only:
                out.append(_join_line(A[i], None, f1, f2, fmt, empty, outsep, True, False))
            i += 1
        elif ka > kb:
            if 2 in unpaired or 2 in only:
                out.append(_join_line(None, B[j], f1, f2, fmt, empty, outsep, False, True))
            j += 1
        else:
            i2 = i
            while i2 < len(A) and enc(key(A[i2], f1)) == ka:
                i2 += 1
            j2 = j
            while j2 < len(B) and enc(key(B[j2], f2)) == kb:
                j2 += 1
            if not only:
                for x in A[i:i2]:
                    for y in B[j:j2]:
                        out.append(_join_line(x, y, f1, f2, fmt, empty, outsep, True, True))
            i, j = i2, j2
    if warned and check != "strict":
        pass
    ctx.w("".join(x + "\n" for x in out))
    if warned:
        ctx.err("input is not in sorted order")
    return status


def _join_line(a, b, f1, f2, fmt, empty, sep, has_a, has_b):
    e = empty if empty is not None else ""
    if fmt:
        cells = []
        for spec in re.split(r"[ ,]+", fmt.strip()):
            if spec == "0":
                k = (a[f1 - 1] if a and f1 - 1 < len(a) else (b[f2 - 1] if b and f2 - 1 < len(b) else e))
                cells.append(k)
                continue
            if spec == "auto":
                return _join_line(a, b, f1, f2, None, empty, sep, has_a, has_b)
            fi, _, col = spec.partition(".")
            src = a if fi == "1" else b
            c = int(col)
            cells.append(src[c - 1] if src and c - 1 < len(src) else e)
        return sep.join(cells)
    if a is not None and b is not None:
        k = a[f1 - 1] if f1 - 1 < len(a) else e
        rest = [x for i, x in enumerate(a) if i != f1 - 1] + [x for i, x in enumerate(b) if i != f2 - 1]
        return sep.join([k] + rest)
    src, f = (a, f1) if a is not None else (b, f2)
    k = src[f - 1] if f - 1 < len(src) else e
    return sep.join([k] + [x for i, x in enumerate(src) if i != f - 1])


@util("split")
def u_split(ctx):
    a = ctx.argv[1:]
    lines = bytes_ = chunks = None
    numeric = False
    suffix_len = 2
    add_suffix = ""
    verbose = False
    ops = []
    i = 0
    while i < len(a):
        x = a[i]
        if x in ("-l", "-b", "-n", "-a") and i + 1 < len(a):
            v = a[i + 1]
            i += 2
            if x == "-l":
                lines = int(v)
            elif x == "-b":
                from .files import _size_arg
                bytes_ = _size_arg(v)
            elif x == "-n":
                chunks = int(v)
            else:
                suffix_len = int(v)
            continue
        m = re.fullmatch(r"-([lbna])(\d+\w*)", x)
        if m:
            if m.group(1) == "l":
                lines = int(m.group(2))
            elif m.group(1) == "b":
                from .files import _size_arg
                bytes_ = _size_arg(m.group(2))
            elif m.group(1) == "n":
                chunks = int(m.group(2))
            else:
                suffix_len = int(m.group(2))
        elif x in ("-d", "--numeric-suffixes"):
            numeric = True
        elif x.startswith("--additional-suffix="):
            add_suffix = x.split("=", 1)[1]
        elif x == "--verbose":
            verbose = True
        elif x.startswith("--lines="):
            lines = int(x.split("=", 1)[1])
        else:
            ops.append(x)
        i += 1
    src = ops[0] if ops else "-"
    prefix = ops[1] if len(ops) > 1 else "x"
    try:
        data = yield from ctx.read_path(src)
    except FSError as e:
        ctx.err(f"cannot open '{src}' for reading: {e.msg}")
        return 1
    pieces = []
    if bytes_:
        pieces = [data[k:k + bytes_] for k in range(0, len(data), bytes_)]
    elif chunks:
        size = len(data) // chunks
        pieces = [data[k * size:(k + 1) * size if k < chunks - 1 else len(data)] for k in range(chunks)]
    else:
        n = lines or 1000
        ls = data.split(b"\n")
        if data.endswith(b"\n"):
            ls.pop()
            ls = [x + b"\n" for x in ls]
        else:
            ls = [x + b"\n" for x in ls[:-1]] + [ls[-1]]
        pieces = [b"".join(ls[k:k + n]) for k in range(0, len(ls), n)]
    letters = "abcdefghijklmnopqrstuvwxyz"
    for idx, piece in enumerate(pieces):
        if numeric:
            suf = f"{idx:0{suffix_len}d}"
        else:
            digits = []
            v = idx
            for _ in range(suffix_len):
                digits.append(letters[v % 26])
                v //= 26
            suf = "".join(reversed(digits))
        name = prefix + suf + add_suffix
        try:
            of = ctx.open(name, "w")
            of.inode.data[:] = piece
        except FSError as e:
            ctx.err(f"{name}: {e.msg}")
            return 1
        if verbose:
            ctx.w(f"creating file '{name}'\n")
    return 0


@util("comm")
def u_comm(ctx):
    a = ctx.argv[1:]
    sup = set()
    outd = "\t"
    ops = []
    check = True
    for x in a:
        if re.fullmatch(r"-[123]+", x):
            sup |= set(x[1:])
        elif x.startswith("--output-delimiter="):
            outd = x.split("=", 1)[1]
        elif x == "--nocheck-order":
            check = False
        elif x in ("--check-order", "--total", "-i"):
            pass
        else:
            ops.append(x)
    if len(ops) != 2:
        ctx.err("missing operand" if len(ops) < 2 else f"extra operand '{ops[2]}'")
        return ctx.usage()
    data = []
    for p in ops:
        try:
            d = yield from ctx.read_path(p)
        except FSError as e:
            ctx.err(f"{p}: {e.msg}")
            return 1
        data.append(split_lines(dec(d)))
    A, B = data
    out = []
    i = j = 0
    status = 0
    warned = [False, False]

    def emit(col, line):
        if str(col) in sup:
            return
        indent = sum(1 for c in range(1, col) if str(c) not in sup)
        out.append(outd * indent + line)
    while i < len(A) or j < len(B):
        for which, lst, idx in ((0, A, i), (1, B, j)):
            if check and idx < len(lst) and idx > 0 and enc(lst[idx]) < enc(lst[idx - 1]) and not warned[which]:
                warned[which] = True
                status = 1
        if j >= len(B) or (i < len(A) and enc(A[i]) < enc(B[j])):
            emit(1, A[i])
            i += 1
        elif i >= len(A) or enc(A[i]) > enc(B[j]):
            emit(2, B[j])
            j += 1
        else:
            emit(3, A[i])
            i += 1
            j += 1
    ctx.w("".join(x + "\n" for x in out))
    for which in (0, 1):
        if warned[which]:
            ctx.err(f"file {which + 1} is not in sorted order")
    if any(warned):
        ctx.err("input is not in sorted order")
    return status


# ---------------------------------------------------------------------------- cmp diff

@util("cmp")
def u_cmp(ctx):
    a = ctx.argv[1:]
    silent = "-s" in a or "--silent" in a or "--quiet" in a
    listing = "-l" in a or "--verbose" in a
    ops = [x for x in a if not x.startswith("-") or x == "-"]
    if len(ops) < 2:
        ops.append("-")
    datas = []
    for p in ops[:2]:
        try:
            datas.append((yield from ctx.read_path(p)))
        except FSError as e:
            ctx.err(f"{p}: {e.msg}")
            return 2
    x, y = datas
    line = 1
    diffs = []
    for i, (c1, c2) in enumerate(zip(x, y)):
        if c1 != c2:
            if listing:
                diffs.append(f"{i + 1} {c1:3o} {c2:3o}")
                continue
            if not silent:
                ctx.w(f"{ops[0]} {ops[1]} differ: char {i + 1}, line {line}\n")
            return 1
        if c1 == 10:
            line += 1
    if listing and diffs:
        width = len(str(min(len(x), len(y))))
        ctx.w("".join(d + "\n" for d in diffs) if True else "")
    if len(x) != len(y):
        short = ops[0] if len(x) < len(y) else ops[1]
        n = min(len(x), len(y))
        if not silent:
            if n == 0:
                ctx.err(f"EOF on {short} which is empty")
            else:
                lines = (x if len(x) < len(y) else y).count(b"\n")
                ctx.err(f"EOF on {short} after byte {n}, line {lines + (0 if (x if len(x) < len(y) else y).endswith(b'\n') else 1)}"
                        if True else "")
        return 1
    return 1 if diffs else 0


def lcs_ops(a, b):
    """Myers' O(ND) diff -> list of ('=', i, j) / ('-', i, None) / ('+', None, j)."""
    n, m = len(a), len(b)
    # trim common prefix/suffix
    pre = 0
    while pre < n and pre < m and a[pre] == b[pre]:
        pre += 1
    suf = 0
    while suf < n - pre and suf < m - pre and a[n - 1 - suf] == b[m - 1 - suf]:
        suf += 1
    A, B = a[pre:n - suf], b[pre:m - suf]
    N, M = len(A), len(B)
    ops = [("=", i, i) for i in range(pre)]
    if N == 0:
        ops += [("+", None, pre + j) for j in range(M)]
    elif M == 0:
        ops += [("-", pre + i, None) for i in range(N)]
    else:
        maxd = N + M
        v = {1: 0}
        trace = []
        found = False
        for d in range(maxd + 1):
            trace.append(dict(v))
            for k in range(-d, d + 1, 2):
                if k == -d or (k != d and v.get(k - 1, -1) < v.get(k + 1, -1)):
                    x = v.get(k + 1, 0)
                else:
                    x = v.get(k - 1, 0) + 1
                y = x - k
                while x < N and y < M and A[x] == B[y]:
                    x += 1
                    y += 1
                v[k] = x
                if x >= N and y >= M:
                    found = True
                    break
            if found:
                break
        # backtrack
        x, y = N, M
        rev = []
        for d in range(len(trace) - 1, -1, -1):
            vv = trace[d]
            k = x - y
            if d == 0:
                while x > 0 and y > 0:
                    rev.append(("=", pre + x - 1, pre + y - 1))
                    x -= 1
                    y -= 1
                break
            if k == -d or (k != d and vv.get(k - 1, -1) < vv.get(k + 1, -1)):
                pk = k + 1
            else:
                pk = k - 1
            px = vv.get(pk, 0)
            py = px - pk
            while x > px and y > py:
                rev.append(("=", pre + x - 1, pre + y - 1))
                x -= 1
                y -= 1
            if x == px:
                rev.append(("+", None, pre + y - 1))
            else:
                rev.append(("-", pre + x - 1, None))
            x, y = px, py
        ops += list(reversed(rev))
    ops += [("=", n - suf + i, m - suf + i) for i in range(suf)]
    return ops


def hunks(ops):
    """Group ops into change hunks: (a_start, a_end, b_start, b_end) half-open over original indices."""
    out = []
    i = 0
    ai = bi = 0
    while i < len(ops):
        op = ops[i]
        if op[0] == "=":
            ai, bi = op[1] + 1, op[2] + 1
            i += 1
            continue
        a0, b0 = ai, bi
        while i < len(ops) and ops[i][0] != "=":
            if ops[i][0] == "-":
                ai = ops[i][1] + 1
            else:
                bi = ops[i][2] + 1
            i += 1
        out.append((a0, ai, b0, bi))
    return out


def _rng(s, e):
    """normal-diff range text for half-open [s, e) (1-based output)."""
    if e - s <= 1:
        return str(s + 1 if e > s else s)
    return f"{s + 1},{e}"


def _diff_files(ctx, la, lb, names, o, out, label_a, label_b, times):
    def norm(x):
        if o.get("i"):
            x = x.lower()
        if o.get("w"):
            x = re.sub(r"[ \t]+", "", x)
        elif o.get("b"):
            x = re.sub(r"[ \t]+", " ", x).rstrip(" \t")
        return x
    ka = [norm(x) for x in la]
    kb = [norm(x) for x in lb]
    ops = lcs_ops(ka, kb)
    hs = hunks(ops)
    if o.get("B"):
        hs = [h for h in hs if any(la[i].strip() for i in range(h[0], h[1])) or any(lb[j].strip() for j in range(h[2], h[3]))]
    if not hs:
        return 0
    if o.get("q"):
        out.append(f"Files {names[0]} and {names[1]} differ\n")
        return 1
    na = names[2] if len(names) > 2 else None
    ta = la.no_nl if hasattr(la, "no_nl") else False

    def line_a(i, prefix):
        s = prefix + la[i] + "\n"
        if i == len(la) - 1 and getattr(la, "no_eol", False):
            s += "\\ No newline at end of file\n"
        return s

    def line_b(j, prefix):
        s = prefix + lb[j] + "\n"
        if j == len(lb) - 1 and getattr(lb, "no_eol", False):
            s += "\\ No newline at end of file\n"
        return s
    if o.get("u") is not None:
        ctxn = o["u"]
        out.append(f"--- {label_a}\t{times[0]}\n+++ {label_b}\t{times[1]}\n")
        groups = []
        for h in hs:
            if groups and h[0] - groups[-1][-1][1] <= 2 * ctxn:
                groups[-1].append(h)
            else:
                groups.append([h])
        for g in groups:
            a0 = max(g[0][0] - ctxn, 0)
            a1 = min(g[-1][1] + ctxn, len(la))
            b0 = max(g[0][2] - ctxn, 0)
            b1 = min(g[-1][3] + ctxn, len(lb))

            def ur(s, e):
                cnt = e - s
                if cnt == 1:
                    return str(s + 1)
                if cnt == 0:
                    return f"{s},0"
                return f"{s + 1},{cnt}"
            out.append(f"@@ -{ur(a0, a1)} +{ur(b0, b1)} @@\n")
            ai, bi = a0, b0
            for h in g:
                while ai < h[0]:
                    out.append(line_a(ai, " "))
                    ai += 1
                    bi += 1
                for i in range(h[0], h[1]):
                    out.append(line_a(i, "-"))
                for j in range(h[2], h[3]):
                    out.append(line_b(j, "+"))
                ai, bi = h[1], h[3]
            while ai < a1:
                out.append(line_a(ai, " "))
                ai += 1
        return 1
    if o.get("y"):
        width = o.get("W", 130)
        col = (width - 3) // 2
        half = col
        for op in ops:
            if op[0] == "=":
                left = la[op[1]]
                out.append(f"{left[:half - 1]:<{half - 1}}" + "   " + lb[op[2]][:half - 1] + "\n" if False else
                           (left[:half].ljust(half + 2) + "  " + lb[op[2]][:half]).rstrip() + "\n")
        return 1
    for a0, a1, b0, b1 in hs:
        if a0 == a1:
            out.append(f"{a0}a{_rng(b0, b1)}\n")
            out += [line_b(j, "> ") for j in range(b0, b1)]
        elif b0 == b1:
            out.append(f"{_rng(a0, a1)}d{b0}\n")
            out += [line_a(i, "< ") for i in range(a0, a1)]
        else:
            out.append(f"{_rng(a0, a1)}c{_rng(b0, b1)}\n")
            out += [line_a(i, "< ") for i in range(a0, a1)]
            out.append("---\n")
            out += [line_b(j, "> ") for j in range(b0, b1)]
    return 1


class Lines(list):
    no_eol = False


def _lines_of(data: bytes) -> Lines:
    text = dec(data)
    ls = Lines(text.split("\n"))
    if text.endswith("\n"):
        ls.pop()
    elif text:
        ls.no_eol = True
    else:
        ls.pop()
    return ls


@util("diff")
def u_diff(ctx):
    try:
        opts, ops = gnu_getopt(ctx.argv[1:], "abBcC:diNqrsTuU:wW:yx:", [
            "text", "ignore-space-change", "ignore-blank-lines", "context=?", "ignore-case", "new-file", "brief",
            "recursive", "report-identical-files", "initial-tab", "unified=?", "ignore-all-space", "width=",
            "side-by-side", "exclude=", "label=", "color=?", "strip-trailing-cr", "normal", "help", "version",
            "suppress-common-lines", "no-dereference", "minimal", "speed-large-files"], name="diff")
    except UsageError as e:
        ctx.err(e.msg)
        return ctx.usage(status=2)
    o = {}
    labels = []
    exclude = []
    for k, v in opts:
        if k in ("-u", "--unified"):
            o["u"] = int(v) if v else 3
        elif k == "-U":
            o["u"] = int(v)
        elif k in ("-q", "--brief"):
            o["q"] = True
        elif k in ("-r", "--recursive"):
            o["r"] = True
        elif k in ("-i", "--ignore-case"):
            o["i"] = True
        elif k in ("-w", "--ignore-all-space"):
            o["w"] = True
        elif k in ("-b", "--ignore-space-change"):
            o["b"] = True
        elif k in ("-B", "--ignore-blank-lines"):
            o["B"] = True
        elif k in ("-s", "--report-identical-files"):
            o["s"] = True
        elif k in ("-N", "--new-file"):
            o["N"] = True
        elif k in ("-y", "--side-by-side"):
            o["y"] = True
        elif k in ("-W", "--width"):
            o["W"] = int(v)
        elif k == "--label":
            labels.append(v)
        elif k in ("-x", "--exclude"):
            exclude.append(v)
        elif k in ("-c", "-C", "--context"):
            o["u"] = int(v) if v else 3
    if len(ops) != 2:
        ctx.err("missing operand after '" + ops[0] + "'" if len(ops) == 1 else "missing operand" if not ops else f"extra operand '{ops[2]}'")
        ctx.e("diff: Try 'diff --help' for more information.\n")
        return 2
    out = []
    status = yield from _diff_paths(ctx, ops[0], ops[1], o, out, labels, exclude, top=True)
    ctx.w("".join(out))
    return status


def _diff_paths(ctx, p1, p2, o, out, labels=(), exclude=(), top=False):
    fs = ctx.fs
    nodes = []
    for p in (p1, p2):
        if p == "-" or p.startswith(("/dev/fd/", "/proc/self/fd/")) or p == "/dev/stdin":
            nodes.append(None)
            continue
        try:
            nodes.append(fs.lookup(p, ctx.cwd))
        except FSError as e:
            if o.get("N"):
                nodes.append("missing")
                continue
            ctx.err(f"{p}: {e.msg}")
            return 2
    n1, n2 = nodes
    if isinstance(n1, object) and n1 not in (None, "missing") and n2 not in (None, "missing") and n1.type == "d" and n2.type == "d":
        if not o.get("r") and not top:
            out.append(f"Common subdirectories: {p1} and {p2}\n")
            return 0
        if not o.get("r") and top:
            pass
        names1, names2 = set(fs.listdir(n1)), set(fs.listdir(n2))
        status = 0
        for name in sorted(names1 | names2, key=lambda s: s.encode()):
            if any(_g(e, name) for e in exclude):
                continue
            a, b = _join(p1, name), _join(p2, name)
            if name not in names2:
                if o.get("N"):
                    st = yield from _diff_paths(ctx, a, b, o, out, exclude=exclude)
                    status = max(status, st)
                else:
                    out.append(f"Only in {p1}: {name}\n")
                    status = max(status, 1)
                continue
            if name not in names1:
                if o.get("N"):
                    st = yield from _diff_paths(ctx, a, b, o, out, exclude=exclude)
                    status = max(status, st)
                else:
                    out.append(f"Only in {p2}: {name}\n")
                    status = max(status, 1)
                continue
            c1, c2 = n1.entries[name], n2.entries[name]
            if c1.type == "d" and c2.type == "d":
                if o.get("r"):
                    st = yield from _diff_paths(ctx, a, b, o, out, exclude=exclude)
                else:
                    out.append(f"Common subdirectories: {a} and {b}\n")
                    st = 0
                status = max(status, st)
                continue
            if (c1.type == "d") != (c2.type == "d"):
                out.append(f"File {a if c1.type == 'd' else b} is a directory while file {b if c1.type == 'd' else a} is a regular file\n")
                status = max(status, 1)
                continue
            sub = []
            st = yield from _diff_paths(ctx, a, b, o, sub, exclude=exclude)
            if sub and not o.get("q"):
                flags = " ".join(f for f in (["-r"] if o.get("r") else []) + (["-u"] if o.get("u") is not None else []))
                out.append(f"diff {flags + ' ' if flags else ''}{a} {b}\n")
            out += sub
            status = max(status, st)
        return status
    if n1 not in (None, "missing") and n1.type == "d":
        p1 = _join(p1, p2.rstrip("/").rsplit("/", 1)[-1])
        try:
            n1 = fs.lookup(p1, ctx.cwd)
        except FSError as e:
            ctx.err(f"{p1}: {e.msg}")
            return 2
    elif n2 not in (None, "missing") and n2.type == "d":
        p2 = _join(p2, p1.rstrip("/").rsplit("/", 1)[-1])
        try:
            n2 = fs.lookup(p2, ctx.cwd)
        except FSError as e:
            ctx.err(f"{p2}: {e.msg}")
            return 2
    datas = []
    times = []
    from ..timeutil import stat_time
    for p, n in ((p1, n1), (p2, n2)):
        if n == "missing":
            datas.append(b"")
            times.append("1970-01-01 00:00:00.000000000 +0000")
            continue
        try:
            d = yield from ctx.read_path(p)
        except FSError as e:
            ctx.err(f"{p}: {e.msg}")
            return 2
        datas.append(d)
        times.append(stat_time(n.mtime if n is not None else ctx.now, ctx.tz))
    if datas[0] == datas[1]:
        if o.get("s"):
            out.append(f"Files {p1} and {p2} are identical\n")
        return 0
    if b"\0" in datas[0][:8000] or b"\0" in datas[1][:8000]:
        out.append(f"Binary files {p1} and {p2} differ\n")
        return 1
    la, lb = _lines_of(datas[0]), _lines_of(datas[1])
    lab = list(labels) + [p1, p2]
    return _diff_files(ctx, la, lb, (p1, p2), o, out, lab[0] if labels else p1, lab[1] if len(labels) > 1 else p2, times)
    yield


def _g(pat, name):
    from ..pattern import fnmatch
    return fnmatch(pat, name)


def _join(a, b):
    return a + b if a.endswith("/") else a + "/" + b


# ---------------------------------------------------------------------------- column fold tee shuf

@util("column")
def u_column(ctx):
    a = ctx.argv[1:]
    table = False
    seps = None
    outsep = "  "
    names = None
    right = set()
    width = ctx.columns
    across = False
    ops = []
    i = 0
    while i < len(a):
        x = a[i]
        if x in ("-s", "--separator", "-o", "--output-separator", "-N", "--table-columns", "-R",
                 "--table-right", "-c", "--output-width", "-l") and i + 1 < len(a):
            v = a[i + 1]
            if x in ("-s", "--separator"):
                seps = v
            elif x in ("-o", "--output-separator"):
                outsep = v
            elif x in ("-N", "--table-columns"):
                names = v.split(",")
            elif x in ("-R", "--table-right"):
                right = set(v.split(","))
            elif x in ("-c", "--output-width"):
                width = int(v)
            i += 2
            continue
        if x.startswith("-") and len(x) > 1:
            for k, c in enumerate(x[1:], 1):
                if c == "t":
                    table = True
                elif c == "x":
                    across = True
                elif c in "so":
                    val = x[k + 1:]
                    if not val:
                        val = a[i + 1] if i + 1 < len(a) else " "
                        i += 1
                    if c == "s":
                        seps = val
                    else:
                        outsep = val
                    break
            i += 1
            continue
        ops.append(x)
        i += 1
    lines = []
    for p, data in (yield from read_inputs(ctx, ops)):
        if data is None:
            return 1
        lines += split_lines(dec(data))
    if table:
        rows = []
        for ln in lines:
            if seps is None:
                cells = ln.split()
            else:
                cells = re.split("[" + re.escape(seps) + "]", ln)
            if not cells or (len(cells) == 1 and cells[0] == "" and seps is None):
                continue
            rows.append(cells)
        if names:
            rows.insert(0, names)
        if not rows:
            return 0
        nc = max(len(r) for r in rows)
        widths = [max((len(r[c]) for r in rows if c < len(r)), default=0) for c in range(nc)]
        out = []
        for r in rows:
            cells = []
            for c, cell in enumerate(r):
                last = c == len(r) - 1
                key = str(c + 1) if not names else (names[c] if c < len(names) else str(c + 1))
                if key in right or str(c + 1) in right:
                    cells.append(cell.rjust(widths[c]))
                elif last:
                    cells.append(cell)
                else:
                    cells.append(cell.ljust(widths[c]))
            out.append(outsep.join(cells).rstrip() if not right else outsep.join(cells))
        ctx.w("".join(x + "\n" for x in out))
        return 0
    items = [ln for ln in lines if ln.strip()]
    if not items:
        return 0
    maxlen = max(len(x) for x in items)
    colw = (maxlen // 8 + 1) * 8
    ncols = max(1, width // colw)
    nrows = math.ceil(len(items) / ncols)
    out = []
    for r in range(nrows):
        row = []
        idxs = range(r * ncols, min(len(items), (r + 1) * ncols)) if across else range(r, len(items), nrows)
        idxs = list(idxs)
        for k, idx in enumerate(idxs):
            it = items[idx]
            if k == len(idxs) - 1:
                row.append(it)
            else:
                pad = colw - len(it)
                tabs = math.ceil(pad / 8)
                row.append(it + "\t" * tabs)
        out.append("".join(row))
    ctx.w("".join(x + "\n" for x in out))
    return 0


@util("fold")
def u_fold(ctx):
    a = ctx.argv[1:]
    width = 80
    spaces = False
    ops = []
    i = 0
    while i < len(a):
        x = a[i]
        if x == "-w" and i + 1 < len(a):
            width = int(a[i + 1])
            i += 2
            continue
        if re.fullmatch(r"-w\d+", x):
            width = int(x[2:])
        elif x == "-s":
            spaces = True
        elif x == "-b":
            pass
        elif re.fullmatch(r"-\d+", x):
            width = int(x[1:])
        else:
            ops.append(x)
        i += 1
    for p, data in (yield from read_inputs(ctx, ops)):
        if data is None:
            return 1
        out = []
        for line in split_lines(dec(data)):
            while len(line) > width:
                cut = width
                if spaces:
                    k = line.rfind(" ", 0, width)
                    if k >= 0:
                        cut = k + 1
                out.append(line[:cut])
                line = line[cut:]
            out.append(line)
        ctx.w("".join(x + "\n" for x in out))
    return 0


@util("tee")
def u_tee(ctx):
    a = ctx.argv[1:]
    append = "-a" in a or "--append" in a
    ignore_int = "-i" in a
    files = [x for x in a if x not in ("-a", "--append", "-i", "--ignore-interrupts", "-p")]
    ofs = []
    status = 0
    for p in files:
        try:
            ofs.append(ctx.open(p, "a" if append else "w"))
        except FSError as e:
            ctx.err(f"{p}: {e.msg}")
            status = 1
    if ignore_int:
        ctx.proc.ignored.add(2)
    from ..fileio import write
    of0 = ctx.proc.fds.get(0)
    while True:
        data = yield from read(ctx.k, of0, 65536)
        if not data:
            break
        for of in ofs:
            yield from write(ctx.k, of, data)
        yield from ctx.out(data)
    return status


@util("shuf")
def u_shuf(ctx):
    a = ctx.argv[1:]
    count = None
    rng_spec = None
    echo = False
    repeat = False
    items = []
    ops = []
    i = 0
    while i < len(a):
        x = a[i]
        if x == "-n" and i + 1 < len(a):
            count = int(a[i + 1])
            i += 2
            continue
        if x.startswith("-n") and len(x) > 2:
            count = int(x[2:])
        elif x == "-i" and i + 1 < len(a):
            rng_spec = a[i + 1]
            i += 2
            continue
        elif x.startswith("--input-range="):
            rng_spec = x.split("=", 1)[1]
        elif x in ("-e", "--echo"):
            echo = True
        elif x in ("-r", "--repeat"):
            repeat = True
        elif x.startswith("--random-source="):
            pass
        else:
            (items if echo else ops).append(x)
        i += 1
    if rng_spec:
        lo, hi = rng_spec.split("-")
        lines = [str(v) for v in range(int(lo), int(hi) + 1)]
    elif echo:
        lines = items
    else:
        lines = []
        for p, data in (yield from read_inputs(ctx, ops)):
            if data is None:
                return 1
            lines += split_lines(dec(data))
    rng = ctx.world.rng
    if repeat:
        n = count if count is not None else 1000
        out = [rng.choice(lines) for _ in range(n)] if lines else []
    else:
        out = list(lines)
        rng.shuffle(out)
        if count is not None:
            out = out[:count]
    ctx.w("".join(x + "\n" for x in out))
    return 0


# ---------------------------------------------------------------------------- checksums & encodings

def _hash_util(algo, bsd_name=None):
    def run(ctx):
        a = ctx.argv[1:]
        check = "-c" in a or "--check" in a
        quiet = "--quiet" in a
        status_only = "--status" in a
        tag = "--tag" in a
        ops = [x for x in a if not x.startswith("-") or x == "-"]
        status = 0
        if check:
            bad = 0
            missing = 0
            for p, data in (yield from read_inputs(ctx, ops)):
                if data is None:
                    return 1
                for line in split_lines(dec(data)):
                    m = re.fullmatch(r"([0-9a-fA-F]+) [ *](.+)", line) or re.fullmatch(r"\w+ \((.+)\) = ([0-9a-f]+)", line)
                    if not m:
                        continue
                    if line.startswith(("MD5", "SHA")):
                        name, want = m.group(1), m.group(2)
                    else:
                        want, name = m.group(1), m.group(2)
                    try:
                        content = yield from ctx.read_path(name)
                    except FSError as e:
                        ctx.err(f"{name}: {e.msg}")
                        if not status_only:
                            ctx.w(f"{name}: FAILED open or read\n")
                        missing += 1
                        continue
                    ok = hashlib.new(algo, content).hexdigest() == want.lower()
                    if not ok:
                        bad += 1
                    if not status_only and not (quiet and ok):
                        ctx.w(f"{name}: {'OK' if ok else 'FAILED'}\n")
            if missing:
                ctx.err(f"WARNING: {missing} listed file{'s' if missing > 1 else ''} could not be read")
            if bad:
                ctx.err(f"WARNING: {bad} computed checksum{'s' if bad > 1 else ''} did NOT match")
            return 1 if bad or missing else 0
        for p, data in (yield from read_inputs(ctx, ops)):
            if data is None:
                status = 1
                continue
            h = hashlib.new(algo, data).hexdigest()
            if tag:
                ctx.w(f"{algo.upper()} ({p}) = {h}\n")
            else:
                ctx.w(f"{h}  {p}\n")
        return status
    return run


for _algo, _names in (("md5", ("md5sum",)), ("sha1", ("sha1sum",)), ("sha224", ("sha224sum",)),
                      ("sha256", ("sha256sum",)), ("sha384", ("sha384sum",)), ("sha512", ("sha512sum",)),
                      ("blake2b", ("b2sum",))):
    util(*_names)(_hash_util(_algo))


@util("md5")
def u_md5(ctx):
    a = ctx.argv[1:]
    quiet = "-q" in a
    ops = [x for x in a if not x.startswith("-")]
    if "-s" in a:
        s = a[a.index("-s") + 1]
        h = hashlib.md5(enc(s)).hexdigest()
        ctx.w(f"{h}\n" if quiet else f'MD5 ("{s}") = {h}\n')
        return 0
    if not ops:
        data = yield from ctx.read_fd(0)
        ctx.w(hashlib.md5(data).hexdigest() + "\n")
        return 0
    status = 0
    for p, data in (yield from read_inputs(ctx, ops)):
        if data is None:
            status = 1
            continue
        h = hashlib.md5(data).hexdigest()
        ctx.w(f"{h}\n" if quiet else f"MD5 ({p}) = {h}\n")
    return status


@util("shasum")
def u_shasum(ctx):
    a = ctx.argv[1:]
    algo = "1"
    ops = []
    check = False
    i = 0
    while i < len(a):
        if a[i] == "-a" and i + 1 < len(a):
            algo = a[i + 1]
            i += 2
            continue
        if a[i] in ("-c", "--check"):
            check = True
        elif not a[i].startswith("-") or a[i] == "-":
            ops.append(a[i])
        i += 1
    name = {"1": "sha1", "224": "sha224", "256": "sha256", "384": "sha384", "512": "sha512"}.get(algo)
    if name is None:
        ctx.e(f"shasum: Unrecognized algorithm\n")
        return 1
    fn = _hash_util(name)
    sub = type(ctx)(ctx.proc, ["shasum"] + (["-c"] if check else []) + ops, ctx.env, ctx.shell)
    st = yield from fn(sub)
    yield from sub.flush()
    return st


@util("base64")
def u_base64(ctx):
    import base64
    a = ctx.argv[1:]
    decode = "-d" in a or "--decode" in a
    wrap = 76
    ops = []
    i = 0
    while i < len(a):
        x = a[i]
        if x in ("-w", "--wrap") and i + 1 < len(a):
            wrap = int(a[i + 1])
            i += 2
            continue
        if x.startswith("--wrap="):
            wrap = int(x.split("=")[1])
        elif x not in ("-d", "--decode", "-i", "--ignore-garbage"):
            ops.append(x)
        i += 1
    data = yield from ctx.read_path(ops[0] if ops else "-")
    if decode:
        try:
            ctx.w(base64.b64decode(re.sub(rb"\s", b"", data), validate=True))
        except Exception:  # noqa: BLE001
            ctx.err("invalid input")
            return 1
        return 0
    s = base64.b64encode(data).decode()
    if wrap:
        s = "\n".join(s[k:k + wrap] for k in range(0, len(s), wrap))
    ctx.w(s + ("\n" if s else ""))
    return 0


@util("xxd")
def u_xxd(ctx):
    a = ctx.argv[1:]
    plain = include = reverse = False
    upper = False
    length = None
    seek = 0
    cols = 16
    group = 2
    ops = []
    i = 0
    while i < len(a):
        x = a[i]
        if x in ("-l", "-len", "-s", "-seek", "-c", "-cols", "-g", "-groupsize") and i + 1 < len(a):
            v = int(a[i + 1], 0)
            if x in ("-l", "-len"):
                length = v
            elif x in ("-s", "-seek"):
                seek = v
            elif x in ("-c", "-cols"):
                cols = v
            else:
                group = v
            i += 2
            continue
        if x in ("-p", "-ps", "-postscript", "-plain"):
            plain = True
        elif x in ("-i", "-include"):
            include = True
        elif x in ("-r", "-revert"):
            reverse = True
        elif x in ("-u",):
            upper = True
        else:
            ops.append(x)
        i += 1
    data = yield from ctx.read_path(ops[0] if ops else "-")
    if reverse:
        text = dec(data)
        out = bytearray()
        if plain:
            out = bytes.fromhex(re.sub(r"\s", "", text))
        else:
            for line in text.splitlines():
                m = re.match(r"[0-9a-fA-F]+:\s*((?:[0-9a-fA-F]{2,4} ?)+)", line)
                if m:
                    hexpart = m.group(1)[:cols * 2 + cols]
                    out += bytes.fromhex(re.sub(r"\s", "", hexpart))
        ctx.w(bytes(out))
        return 0
    data = data[seek:]
    if length is not None:
        data = data[:length]
    hx = (lambda b: f"{b:02X}") if upper else (lambda b: f"{b:02x}")
    if plain:
        s = "".join(hx(b) for b in data)
        ctx.w("".join(s[k:k + 60] + "\n" for k in range(0, len(s), 60)))
        return 0
    if include:
        name = re.sub(r"[^A-Za-z0-9]", "_", ops[0]) if ops else "stdin"
        body = ",".join(("\n  " if k % 12 == 0 else " ") + f"0x{b:02x}" for k, b in enumerate(data))
        ctx.w(f"unsigned char {name}[] = {{{body}\n}};\nunsigned int {name}_len = {len(data)};\n")
        return 0
    out = []
    for off in range(0, len(data), cols):
        chunk = data[off:off + cols]
        groups = []
        for g in range(0, len(chunk), group):
            groups.append("".join(hx(b) for b in chunk[g:g + group]))
        hexs = " ".join(groups)
        full = cols * 2 + (cols // group) - 1 if group else cols * 2
        text = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        out.append(f"{off + seek:08x}: {hexs:<{full}}  {text}")
    ctx.w("".join(x + "\n" for x in out))
    return 0


@util("hexdump", "hd")
def u_hexdump(ctx):
    a = ctx.argv[1:]
    canonical = "-C" in a or ctx.name == "hd"
    ops = [x for x in a if not x.startswith("-")]
    data = yield from ctx.read_path(ops[0] if ops else "-")
    out = []
    if canonical:
        prev = None
        star = False
        for off in range(0, len(data), 16):
            chunk = data[off:off + 16]
            if chunk == prev and len(chunk) == 16:
                if not star:
                    out.append("*")
                    star = True
                continue
            star = False
            prev = chunk
            h1 = " ".join(f"{b:02x}" for b in chunk[:8])
            h2 = " ".join(f"{b:02x}" for b in chunk[8:])
            text = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
            out.append(f"{off:08x}  {h1:<23}  {h2:<23}  |{text}|")
        out.append(f"{len(data):08x}")
    else:
        for off in range(0, len(data), 16):
            chunk = data[off:off + 16]
            words = []
            for k in range(0, len(chunk), 2):
                pair = chunk[k:k + 2]
                v = pair[0] | ((pair[1] if len(pair) > 1 else 0) << 8)
                words.append(f"{v:04x}")
            out.append(f"{off:07x} " + " ".join(words))
        out.append(f"{len(data):07x}")
    ctx.w("".join(x + "\n" for x in out))
    return 0


@util("od")
def u_od(ctx):
    a = ctx.argv[1:]
    fmt = "o2"
    addr = "o"
    ops = []
    i = 0
    while i < len(a):
        x = a[i]
        if x == "-c":
            fmt = "c"
        elif x == "-An":
            addr = "n"
        elif x == "-A" and i + 1 < len(a):
            addr = a[i + 1]
            i += 1
        elif x.startswith("-A"):
            addr = x[2:]
        elif x == "-t" and i + 1 < len(a):
            fmt = a[i + 1]
            i += 1
        elif x.startswith("-t"):
            fmt = x[2:]
        elif x == "-x":
            fmt = "x2"
        elif x == "-b":
            fmt = "o1"
        else:
            ops.append(x)
        i += 1
    data = yield from ctx.read_path(ops[0] if ops else "-")
    out = []

    def a_txt(off):
        if addr == "n":
            return ""
        return {"o": f"{off:07o}", "x": f"{off:06x}", "d": f"{off:07d}"}.get(addr, f"{off:07o}")
    show_z = fmt.endswith("z")
    f = fmt.rstrip("z")
    for off in range(0, len(data), 16):
        chunk = data[off:off + 16]
        if f == "c":
            cells = []
            for b in chunk:
                c = {0: "\\0", 7: "\\a", 8: "\\b", 9: "\\t", 10: "\\n", 11: "\\v", 12: "\\f", 13: "\\r"}.get(b)
                cells.append(f"{c if c else (chr(b) if 32 <= b < 127 else f'{b:03o}'):>4}")
            out.append(a_txt(off) + "".join(cells))
        elif f in ("x1", "o1", "d1", "u1"):
            base = f[0]
            w = {"x": 3, "o": 4, "d": 4, "u": 4}[base]
            cells = [f"{b:>{w - 1}{'x' if base == 'x' else ('o' if base == 'o' else 'd')}}".rjust(w) if base != "x" else f" {b:02x}" for b in chunk]
            line = a_txt(off) + "".join(cells)
            if show_z:
                line += "  >" + "".join(chr(b) if 32 <= b < 127 else "." for b in chunk) + "<"
            out.append(line)
        else:
            words = []
            for k in range(0, len(chunk), 2):
                pair = chunk[k:k + 2]
                v = pair[0] | ((pair[1] if len(pair) > 1 else 0) << 8)
                words.append(f" {v:06o}" if f.startswith("o") else f" {v:04x}")
            out.append(a_txt(off) + "".join(words))
    if addr != "n":
        out.append(a_txt(len(data)))
    ctx.w("".join(x + "\n" for x in out))
    return 0


@util("strings")
def u_strings(ctx):
    a = ctx.argv[1:]
    n = 4
    ops = []
    i = 0
    while i < len(a):
        x = a[i]
        if x == "-n" and i + 1 < len(a):
            n = int(a[i + 1])
            i += 2
            continue
        if re.fullmatch(r"-\d+", x):
            n = int(x[1:])
        elif x.startswith("-n") and x[2:].isdigit():
            n = int(x[2:])
        elif x.startswith("--bytes="):
            n = int(x.split("=")[1])
        else:
            ops.append(x)
        i += 1
    status = 0
    for p, data in (yield from read_inputs(ctx, ops)):
        if data is None:
            status = 1
            continue
        for m in re.finditer(rb"[\x20-\x7e\t]{%d,}" % n, data):
            ctx.w(m.group(0) + b"\n")
    return status


@util("less", "more", "most", "pager")
def u_less(ctx):
    """A pager on the simulated terminal shows the whole file and quits (as if you pressed q at the end)."""
    a = ctx.argv[1:]
    ops = [x for x in a if not x.startswith("-") and not x.startswith("+")]
    numbers = "-N" in a
    status = 0
    if not ops:
        if ctx.stdin_is_tty():
            ctx.e("Missing filename (\"less --help\" for help)\n")
            return 1
        data = yield from ctx.read_fd(0)
        datas = [("-", data)]
    else:
        datas = []
        for p in ops:
            try:
                node = ctx.lookup(p)
                if node.type == "d":
                    ctx.e(f"{p} is a directory\n")
                    status = 1
                    continue
                datas.append((p, (yield from ctx.read_path(p))))
            except FSError as e:
                ctx.e(f"{p}: {e.msg}\n")
                status = 1
    for p, data in datas:
        if numbers:
            ctx.w("".join(f"{i:>7} {ln}\n" for i, ln in enumerate(split_lines(dec(data)), 1)))
        else:
            ctx.w(data)
            if data and not data.endswith(b"\n"):
                ctx.w(b"\n")
    return status
