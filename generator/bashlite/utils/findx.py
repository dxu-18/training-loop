"""find and xargs."""

from __future__ import annotations

import math
import re

from ..exec import util
from ..kernel import NS
from ..pattern import fnmatch
from ..timeutil import DateError, local_dt, parse_date, strftime
from ..vfs import FSError, mode_string


def q(ctx, s: str) -> str:
    """find's locale quoting: ‘x’ in UTF-8 locales, 'x' otherwise."""
    loc = ctx.env.get("LC_ALL") or ctx.env.get("LC_MESSAGES") or ctx.env.get("LANG", "")
    if "UTF-8" in loc.upper() or "UTF8" in loc.upper():
        return f"‘{s}’"
    return f"'{s}'"


class FindError(Exception):
    pass


TESTS_WITH_ARG = {"-name", "-iname", "-path", "-ipath", "-wholename", "-iwholename", "-regex", "-iregex", "-type",
                  "-xtype", "-size", "-mtime", "-mmin", "-atime", "-amin", "-ctime", "-cmin", "-newer", "-anewer",
                  "-cnewer", "-newermt", "-perm", "-user", "-group", "-uid", "-gid", "-links", "-inum", "-samefile",
                  "-maxdepth", "-mindepth", "-printf", "-fprint", "-fprintf", "-regextype", "-lname", "-ilname",
                  "-used", "-fls", "-fprint0", "-context"}
NO_ARG = {"-print", "-print0", "-delete", "-prune", "-quit", "-true", "-false", "-empty", "-depth", "-mount",
          "-xdev", "-readable", "-writable", "-executable", "-nouser", "-nogroup", "-ls", "-follow", "-noleaf",
          "-daystart", "-ignore_readdir_race"}
ACTIONS = {"-print", "-print0", "-printf", "-fprint", "-fprintf", "-ls", "-fls", "-delete", "-exec", "-execdir",
           "-ok", "-okdir", "-quit", "-prune", "-fprint0"}


def parse_expr(tokens, ctx):
    """-> AST of tuples: ('and', a, b) ('or', a, b) ('not', a) ('test', name, arg) ('exec', kind, argv, plus)."""
    pos = [0]
    opts = {"maxdepth": None, "mindepth": 0, "depth": False, "has_action": False, "follow": False,
            "daystart": False, "regextype": "emacs"}

    def peek():
        return tokens[pos[0]] if pos[0] < len(tokens) else None

    def take():
        pos[0] += 1
        return tokens[pos[0] - 1]

    def parse_or():
        left = parse_and()
        while peek() in ("-o", "-or"):
            take()
            right = parse_and()
            left = ("or", left, right)
        return left

    def parse_and():
        left = parse_not()
        while True:
            t = peek()
            if t is None or t in ("-o", "-or", ")", ","):
                return left
            if t in ("-a", "-and"):
                take()
            right = parse_not()
            left = ("and", left, right)

    def parse_not():
        t = peek()
        if t in ("!", "-not"):
            take()
            return ("not", parse_not())
        return parse_primary()

    def parse_primary():
        t = take()
        if t is None:
            raise FindError("expected an expression")
        if t == "(":
            e = parse_or()
            if peek() != ")":
                raise FindError("invalid expression; I was expecting to find a ')' somewhere but did not see one.")
            take()
            return e
        if t in ("-exec", "-execdir", "-ok", "-okdir"):
            argv = []
            while True:
                x = take()
                if x is None:
                    raise FindError(f"missing argument to `{t}'")
                if x == ";":
                    plus = False
                    break
                if x == "+" and argv and argv[-1] == "{}":
                    plus = True
                    argv.pop()
                    break
                argv.append(x)
            if not argv:
                raise FindError(f"missing argument to `{t}'")
            opts["has_action"] = True
            return ("exec", t, argv, plus)
        if t in TESTS_WITH_ARG:
            arg = take()
            if arg is None:
                raise FindError(f"missing argument to `{t}'")
            if t == "-maxdepth":
                opts["maxdepth"] = int(arg)
                return ("test", "-true", None)
            if t == "-mindepth":
                opts["mindepth"] = int(arg)
                return ("test", "-true", None)
            if t == "-regextype":
                opts["regextype"] = arg
                return ("test", "-true", None)
            if t in ACTIONS:
                opts["has_action"] = True
            if t == "-type" or t == "-xtype":
                for ch in arg.split(","):
                    if ch not in "fdlpscbD":
                        raise FindError(f"Unknown argument to {t}: {arg}")
            if t == "-newer" or t == "-anewer" or t == "-cnewer":
                try:
                    ref = ctx.fs.lookup(arg, ctx.cwd)
                except FSError as e:
                    raise FindError(f"{q(ctx, arg)}: {e.msg}") from None
                arg = ref.mtime
            if t == "-newermt":
                try:
                    arg = parse_date(arg, ctx.now, ctx.tz)
                except DateError:
                    raise FindError(f"I cannot figure out how to interpret {q(ctx, arg)} as a date or time") from None
            if t in ("-size",) and not re.fullmatch(r"[+-]?\d+[cwbkMG]?", arg):
                raise FindError(f"invalid -size type `{arg[-1:]}'" if arg and arg[-1].isalpha() else f"Invalid argument `{arg}' to -size")
            if t in ("-mtime", "-mmin", "-atime", "-amin", "-ctime", "-cmin", "-links", "-uid", "-gid", "-inum",
                     "-used") and not re.fullmatch(r"[+-]?\d+(\.\d+)?", arg):
                raise FindError(f"invalid argument `{arg}' to `{t}'")
            if t in ("-regex", "-iregex"):
                from ..regex import compile_re
                arg = compile_re(arg, "ere" if opts["regextype"] in ("posix-extended", "egrep", "findutils-default", "emacs") else "bre",
                                 t == "-iregex")
            return ("test", t, arg)
        if t in NO_ARG:
            if t == "-depth":
                opts["depth"] = True
                return ("test", "-true", None)
            if t == "-follow":
                opts["follow"] = True
                return ("test", "-true", None)
            if t == "-daystart":
                opts["daystart"] = True
                return ("test", "-true", None)
            if t in ("-mount", "-xdev", "-noleaf", "-ignore_readdir_race"):
                return ("test", "-true", None)
            if t in ACTIONS:
                opts["has_action"] = True
            if t == "-delete":
                opts["depth"] = True
            return ("test", t, None)
        if t.startswith("-"):
            raise FindError(f"unknown predicate `{t}'")
        raise FindError(f"paths must precede expression: `{t}'")
    if not tokens:
        return ("test", "-true", None), opts
    e = parse_or()
    while peek() == ",":
        take()
        e = ("comma", e, parse_or())
    if pos[0] < len(tokens):
        t = tokens[pos[0]]
        if t == ")":
            raise FindError("invalid expression; you have too many ')'")
        raise FindError(f"paths must precede expression: `{t}'" if not t.startswith("-") else f"unknown predicate `{t}'")
    return e, opts


def _num_cmp(spec: str, value) -> bool:
    if spec.startswith("+"):
        return value > float(spec[1:])
    if spec.startswith("-"):
        return value < float(spec[1:])
    return value == float(spec)


SIZE_UNITS = {"c": 1, "w": 2, "b": 512, "k": 1024, "M": 1 << 20, "G": 1 << 30}


class Finder:
    def __init__(self, ctx, expr, opts):
        self.ctx, self.expr, self.opts = ctx, expr, opts
        self.status = 0
        self.quit = False
        self.batches: dict = {}
        self.out = []

    def evaluate(self, e, path, node, depth, start):
        kind = e[0]
        if kind == "and":
            ok = yield from self.evaluate(e[1], path, node, depth, start)
            if not ok:
                return False
            return (yield from self.evaluate(e[2], path, node, depth, start))
        if kind == "or":
            ok = yield from self.evaluate(e[1], path, node, depth, start)
            if ok:
                return True
            return (yield from self.evaluate(e[2], path, node, depth, start))
        if kind == "not":
            ok = yield from self.evaluate(e[1], path, node, depth, start)
            return not ok
        if kind == "comma":
            yield from self.evaluate(e[1], path, node, depth, start)
            return (yield from self.evaluate(e[2], path, node, depth, start))
        if kind == "exec":
            return (yield from self.do_exec(e, path, node, start))
        return (yield from self.test(e[1], e[2], path, node, depth, start))

    def test(self, name, arg, path, node, depth, start):
        ctx = self.ctx
        base = path.rstrip("/").rsplit("/", 1)[-1] if path not in ("/",) else "/"
        if name == "-true":
            return True
        if name == "-false":
            return False
        if name in ("-name", "-iname"):
            return fnmatch(arg, base, name == "-iname")
        if name in ("-path", "-ipath", "-wholename", "-iwholename"):
            return fnmatch(arg, path, name in ("-ipath", "-iwholename"))
        if name in ("-lname", "-ilname"):
            return node.type == "l" and fnmatch(arg, node.target, name == "-ilname")
        if name in ("-regex", "-iregex"):
            return arg.fullmatch(path) is not None
        if name in ("-type", "-xtype"):
            t = node.type
            if name == "-xtype" and t == "l":
                try:
                    t = ctx.fs.lookup(path, ctx.cwd).type
                except FSError:
                    t = "l"
            return t in [{"f": "f", "d": "d", "l": "l", "p": "p", "s": "s", "c": "c", "b": "b"}[c] for c in arg.split(",") if c != "D"]
        if name == "-size":
            m = re.fullmatch(r"([+-]?)(\d+)([cwbkMG]?)", arg)
            unit = SIZE_UNITS[m.group(3) or "b"]
            size = node.size if node.type != "d" else 4096
            units = math.ceil(size / unit)
            return _num_cmp(m.group(1) + m.group(2), units)
        if name in ("-mtime", "-atime", "-ctime", "-mmin", "-amin", "-cmin"):
            t = {"m": node.mtime, "a": node.atime, "c": node.ctime}[name[1]]
            age = (ctx.now - t) / NS
            if name.endswith("min"):
                val = math.floor(age / 60) if not arg.startswith(("+", "-")) else age / 60
                if arg.startswith("-"):
                    return age / 60 < float(arg[1:])
                if arg.startswith("+"):
                    return age / 60 > float(arg[1:])
                return math.ceil(age / 60) == float(arg) if age > 0 else float(arg) == 0
            days = math.floor(age / 86400)
            return _num_cmp(arg, days)
        if name in ("-newer", "-anewer", "-cnewer", "-newermt"):
            t = {"-newer": node.mtime, "-anewer": node.atime, "-cnewer": node.ctime, "-newermt": node.mtime}[name]
            return t > arg
        if name == "-empty":
            if node.type == "d":
                return not node.entries
            return node.type == "f" and not node.data
        if name == "-perm":
            from .files import parse_mode
            mode = node.mode & 0o7777
            if arg.startswith("-"):
                want = parse_mode(arg[1:], 0, node.type == "d", 0)
                return mode & want == want
            if arg.startswith("/"):
                want = parse_mode(arg[1:], 0, node.type == "d", 0)
                return want == 0 or bool(mode & want)
            return mode == parse_mode(arg, 0, node.type == "d", 0)
        if name == "-user":
            return ctx.host.user_name(node.uid) == arg or str(node.uid) == arg
        if name == "-group":
            return ctx.host.group_name(node.gid) == arg or str(node.gid) == arg
        if name == "-uid":
            return _num_cmp(arg, node.uid)
        if name == "-gid":
            return _num_cmp(arg, node.gid)
        if name == "-nouser":
            return node.uid not in ctx.host.users
        if name == "-nogroup":
            return node.gid not in ctx.host.groups
        if name == "-links":
            return _num_cmp(arg, node.nlink)
        if name == "-inum":
            return _num_cmp(arg, node.ino)
        if name == "-samefile":
            try:
                return ctx.fs.lookup(arg, ctx.cwd) is node
            except FSError:
                return False
        if name == "-readable":
            return ctx.fs.can(node, 4, ctx.uid, ctx.proc.gids)
        if name == "-writable":
            return ctx.fs.can(node, 2, ctx.uid, ctx.proc.gids)
        if name == "-executable":
            return ctx.fs.can(node, 1, ctx.uid, ctx.proc.gids) and (node.type == "d" or bool(node.mode & 0o111))
        if name == "-print":
            self.out.append(path + "\n")
            return True
        if name == "-print0":
            self.out.append(path + "\0")
            return True
        if name in ("-printf", "-fprintf"):
            self.out.append(self.printf(arg, path, node, depth, start))
            return True
        if name == "-ls":
            self.out.append(self.ls_line(path, node))
            return True
        if name == "-prune":
            self.pruned = True
            return True
        if name == "-quit":
            self.quit = True
            return True
        if name == "-delete":
            yield from self.flush()
            try:
                if node.type == "d":
                    ctx.fs.rmdir(path, ctx.cwd, ctx.uid, ctx.proc.gids)
                else:
                    ctx.fs.unlink(path, ctx.cwd, ctx.uid, ctx.proc.gids)
            except FSError as e:
                ctx.err(f"cannot delete {q(ctx, path)}: {e.msg}")
                self.status = 1
                return False
            return True
        return False
        yield

    def printf(self, fmt, path, node, depth, start):
        ctx = self.ctx
        out = []
        i = 0
        while i < len(fmt):
            c = fmt[i]
            if c == "\\" and i + 1 < len(fmt):
                d = fmt[i + 1]
                out.append({"n": "\n", "t": "\t", "0": "\0", "\\": "\\", "a": "\a", "r": "\r"}.get(d, "\\" + d))
                i += 2
                continue
            if c != "%" or i + 1 >= len(fmt):
                out.append(c)
                i += 1
                continue
            m = re.match(r"%(-?\d*)(?:\.(\d+))?([A-Za-z%])", fmt[i:])
            if not m:
                out.append(c)
                i += 1
                continue
            width, prec, k = m.group(1), m.group(2), m.group(3)
            i += m.end()
            if k in "TAC" and i < len(fmt):
                tk = fmt[i]
                i += 1
                t = {"T": node.mtime, "A": node.atime, "C": node.ctime}[k]
                d = local_dt(t, ctx.tz)
                v = str(t // NS) + f".{t % NS:09d}0" if tk == "@" else strftime("%" + tk, d, t % NS)
            elif k == "p":
                v = path
            elif k == "f":
                v = path.rstrip("/").rsplit("/", 1)[-1] if path != "/" else "/"
            elif k == "h":
                v = path.rsplit("/", 1)[0] if "/" in path.rstrip("/") else "."
            elif k == "P":
                v = path[len(start):].lstrip("/") if path.startswith(start) else path
            elif k == "H":
                v = start
            elif k == "s":
                v = str(node.size)
            elif k == "k":
                v = str((node.blocks + 1) // 2)
            elif k == "b":
                v = str(node.blocks)
            elif k == "m":
                v = f"{node.mode & 0o7777:o}"
            elif k == "M":
                v = mode_string(node)
            elif k == "u":
                v = ctx.host.user_name(node.uid)
            elif k == "U":
                v = str(node.uid)
            elif k == "g":
                v = ctx.host.group_name(node.gid)
            elif k == "G":
                v = str(node.gid)
            elif k == "d":
                v = str(depth)
            elif k == "y":
                v = {"f": "f", "d": "d", "l": "l", "c": "c", "p": "p", "s": "s"}[node.type]
            elif k == "l":
                v = node.target if node.type == "l" else ""
            elif k == "i":
                v = str(node.ino)
            elif k == "n":
                v = str(node.nlink)
            elif k in "tac":
                t = {"t": node.mtime, "a": node.atime, "c": node.ctime}[k]
                v = strftime("%a %b %e %H:%M:%S.%N0 %Y", local_dt(t, ctx.tz), t % NS) if False else \
                    strftime("%a %b %e %H:%M:%S", local_dt(t, ctx.tz)) + f".{t % NS:09d}0 " + strftime("%Y", local_dt(t, ctx.tz))
            elif k == "%":
                v = "%"
            else:
                v = "%" + k
            if prec:
                v = v[:int(prec)]
            if width:
                w = int(width)
                v = v.ljust(-w) if w < 0 else v.rjust(w)
            out.append(v)
        return "".join(out)

    def ls_line(self, path, node):
        ctx = self.ctx
        from ..timeutil import ls_time
        name = path + (" -> " + node.target if node.type == "l" else "")
        return (f"{node.ino:>9} {max((node.blocks + 1) // 2, 0):>6} {mode_string(node)} {node.nlink:>3} "
                f"{ctx.host.user_name(node.uid):<8} {ctx.host.group_name(node.gid):<8} {node.size:>8} "
                f"{ls_time(node.mtime, ctx.now, ctx.tz)} {name}\n")

    def flush(self):
        if self.out:
            self.ctx.w("".join(self.out))
            self.out = []
        yield from self.ctx.maybe_flush()

    def do_exec(self, e, path, node, start):
        ctx = self.ctx
        _, kind, argv, plus = e
        if kind in ("-execdir", "-okdir"):
            name = "./" + path.rstrip("/").rsplit("/", 1)[-1]
            cwd = path.rsplit("/", 1)[0] if "/" in path else ctx.cwd
            if not cwd.startswith("/"):
                cwd = ctx.abspath(cwd)
        else:
            name, cwd = path, None
        if plus:
            key = id(e)
            self.batches.setdefault(key, (argv, cwd, []))[2].append(name)
            return True
        cmd = [a.replace("{}", name) for a in argv]
        yield from self.flush()
        if kind in ("-ok", "-okdir"):
            ctx.e(f"< {' '.join(cmd)} ... {path} > ? ")
            from ..fileio import readline
            ans = yield from readline(ctx.k, ctx.proc.fds.get(0))
            if not ans.decode(errors="replace").strip().lower().startswith("y"):
                return False
        st = yield from ctx.run(cmd, cwd=cwd)
        if st == 127 and ctx.spawn is not None:
            pass
        return st == 0

    def run_batches(self):
        ctx = self.ctx
        yield from self.flush()
        for argv, cwd, names in self.batches.values():
            if not names:
                continue
            cmd = [a for a in argv] + names
            st = yield from ctx.run(cmd, cwd=cwd)
            if st != 0:
                self.status = 1

    def walk(self, start, path, node, depth):
        if self.quit:
            return
        ctx = self.ctx
        o = self.opts
        maxd, mind = o["maxdepth"], o["mindepth"]
        is_dir = node.type == "d"
        if o["follow"] and node.type == "l":
            try:
                node = ctx.fs.lookup(path, ctx.cwd)
                is_dir = node.type == "d"
            except FSError:
                pass
        self.pruned = False
        if not o["depth"] and depth >= mind:
            yield from self.evaluate(self.expr, path, node, depth, start)
            if not o["has_action"] and self._last_ok:
                pass
        pruned = self.pruned
        if is_dir and not pruned and (maxd is None or depth < maxd) and not self.quit:
            if not ctx.fs.can(node, 5, ctx.uid, ctx.proc.gids):
                yield from self.flush()
                ctx.err(f"{q(ctx, path)}: Permission denied")
                self.status = 1
            else:
                for name in ctx.fs.listdir(node):
                    if self.quit:
                        break
                    child = node.entries.get(name)
                    if child is None:
                        continue
                    cp = path + name if path.endswith("/") else path + "/" + name
                    yield from self.walk(start, cp, child, depth + 1)
                    if len(self.out) > 256:
                        yield from self.flush()
        if o["depth"] and depth >= mind and not self.quit:
            yield from self.evaluate(self.expr, path, node, depth, start)

    _last_ok = True


@util("find")
def u_find(ctx):
    a = ctx.argv[1:]
    follow = False
    while a and a[0] in ("-H", "-L", "-P", "-O0", "-O1", "-O2", "-O3"):
        if a[0] == "-L":
            follow = True
        a = a[1:]
    paths = []
    while a and not (a[0].startswith("-") and len(a[0]) > 1) and a[0] not in ("(", "!", ")", ","):
        paths.append(a.pop(0))
    if not paths:
        paths = ["."]
    try:
        expr, opts = parse_expr(a, ctx)
    except FindError as e:
        ctx.err(str(e))
        return 1
    except Exception as e:  # noqa: BLE001
        ctx.err(str(e))
        return 1
    opts["follow"] = opts["follow"] or follow
    if not opts["has_action"]:
        expr = ("and", expr, ("test", "-print", None))
    f = Finder(ctx, expr, opts)
    for p in paths:
        try:
            node = ctx.fs.lookup(p, ctx.cwd, follow)
        except FSError as e:
            yield from f.flush()
            ctx.err(f"{q(ctx, p)}: {e.msg}")
            f.status = 1
            continue
        yield from f.walk(p, p, node, 0)
        if f.quit:
            break
    yield from f.run_batches()
    yield from f.flush()
    return f.status


# ---------------------------------------------------------------------------- xargs

def split_items(text: str, ctx):
    """xargs default input parsing: blanks and newlines separate; quotes and backslash escape."""
    items, cur, i, n = [], [], 0, len(text)
    have = False
    while i < n:
        c = text[i]
        if c in " \t\n":
            if have:
                items.append("".join(cur))
                cur, have = [], False
            i += 1
            continue
        if c in "'\"":
            j = text.find(c, i + 1)
            nl = text.find("\n", i + 1)
            if j < 0 or (0 <= nl < j):
                kind = "single" if c == "'" else "double"
                raise ValueError(f"unmatched {kind} quote; by default quotes are special to xargs unless you use the -0 option")
            cur.append(text[i + 1:j])
            have = True
            i = j + 1
            continue
        if c == "\\" and i + 1 < n:
            cur.append(text[i + 1])
            have = True
            i += 2
            continue
        cur.append(c)
        have = True
        i += 1
    if have:
        items.append("".join(cur))
    return items


@util("xargs")
def u_xargs(ctx):
    a = ctx.argv[1:]
    max_args = None
    max_lines = None
    replace = None
    parallel = 1
    trace = False
    no_empty = False
    null = False
    delim = None
    argfile = None
    eof = None
    i = 0
    while i < len(a):
        x = a[i]
        nxt = a[i + 1] if i + 1 < len(a) else None
        if x in ("-n", "--max-args") and nxt is not None:
            max_args = int(nxt)
            i += 2
            continue
        if re.fullmatch(r"-n\d+", x):
            max_args = int(x[2:])
        elif x.startswith("--max-args="):
            max_args = int(x.split("=")[1])
        elif x in ("-L", "--max-lines") and nxt is not None:
            max_lines = int(nxt)
            i += 2
            continue
        elif re.fullmatch(r"-L\d+", x):
            max_lines = int(x[2:])
        elif x == "-I" and nxt is not None:
            replace = nxt
            i += 2
            continue
        elif x.startswith("-I") and len(x) > 2:
            replace = x[2:]
        elif x in ("-i", "--replace") or x.startswith("--replace="):
            replace = x.split("=", 1)[1] if "=" in x else "{}"
        elif x in ("-P", "--max-procs") and nxt is not None:
            parallel = int(nxt)
            i += 2
            continue
        elif re.fullmatch(r"-P\d+", x):
            parallel = int(x[2:])
        elif x in ("-t", "--verbose"):
            trace = True
        elif x in ("-r", "--no-run-if-empty"):
            no_empty = True
        elif x in ("-0", "--null"):
            null = True
        elif x in ("-d", "--delimiter") and nxt is not None:
            delim = nxt.encode().decode("unicode_escape")
            i += 2
            continue
        elif x in ("-a", "--arg-file") and nxt is not None:
            argfile = nxt
            i += 2
            continue
        elif x in ("-E",) and nxt is not None:
            eof = nxt
            i += 2
            continue
        elif x in ("-s", "--max-chars") and nxt is not None:
            i += 2
            continue
        elif x in ("-x", "--exit", "-p", "--interactive", "--process-slot-var"):
            pass
        elif x == "--":
            i += 1
            break
        elif x.startswith("-") and len(x) > 1:
            ctx.err(f"invalid option -- '{x[1]}'")
            ctx.e("Usage: xargs [OPTION]... COMMAND [INITIAL-ARGS]...\nTry 'xargs --help' for more information.\n")
            return 1
        else:
            break
        i += 1
    cmd = a[i:] or ["echo"]
    try:
        data = yield from (ctx.read_path(argfile) if argfile else ctx.read_fd(0))
    except Exception as e:  # noqa: BLE001
        ctx.err(f"{argfile}: No such file or directory")
        return 1
    text = data.decode("utf-8", "surrogateescape")
    if null:
        items = [x for x in text.split("\0")]
        if items and items[-1] == "":
            items.pop()
    elif delim is not None:
        items = text.split(delim)
        if items and items[-1] in ("", "\n"):
            items.pop()
    elif replace is not None or max_lines is not None:
        items = [ln.lstrip(" \t") for ln in text.split("\n")]
        if items and items[-1] == "":
            items.pop()
        items = [x for x in items if x.strip() or replace is None]
        if max_lines is not None and replace is None:
            # each line may hold several blank-separated arguments
            groups = []
            for k in range(0, len(items), max_lines):
                g = []
                for ln in items[k:k + max_lines]:
                    try:
                        g += split_items(ln, ctx)
                    except ValueError as e:
                        ctx.err(str(e))
                        return 1
                groups.append(g)
            return (yield from _run_groups(ctx, cmd, groups, parallel, trace))
    else:
        try:
            items = split_items(text, ctx)
        except ValueError as e:
            ctx.err(str(e))
            return 1
    if eof is not None and eof in items:
        items = items[:items.index(eof)]
    if replace is not None:
        groups = [[a.replace(replace, it) for a in cmd[1:]] for it in items]
        cmds = [[cmd[0].replace(replace, it)] + g for g, it in zip(groups, items)]
        return (yield from _run_cmds(ctx, cmds, parallel, trace))
    if not items:
        if no_empty:
            return 0
        groups = [[]]
    elif max_args:
        groups = [items[k:k + max_args] for k in range(0, len(items), max_args)]
    else:
        groups = [items]
    return (yield from _run_groups(ctx, cmd, groups, parallel, trace))


def _run_groups(ctx, cmd, groups, parallel, trace):
    return (yield from _run_cmds(ctx, [cmd + g for g in groups], parallel, trace))


def _run_cmds(ctx, cmds, parallel, trace):
    status = 0
    running = []
    yield from ctx.flush()

    def account(p):
        nonlocal status
        st = p.status
        if p.term_sig:
            ctx.err(f"{p.argv[0]}: terminated by signal {p.term_sig}")
            status = 125
        elif st == 255:
            ctx.err(f"{p.argv[0]}: exited with status 255; aborting")
            status = 124
        elif 1 <= st <= 125:
            status = max(status, 123) if status != 124 else status
        elif st in (126, 127):
            status = st
    for c in cmds:
        if trace:
            ctx.e(" ".join(c) + "\n")
        p = ctx.spawn(c)
        if p is None:
            ctx.err(f"{c[0]}: No such file or directory")
            return 127
        running.append(p)
        if len(running) >= max(parallel, 1) and parallel != 0:
            while all(x.state != "Z" for x in running):
                yield ("wait", running)
            done = [x for x in running if x.state == "Z"]
            for x in done:
                account(x)
                running.remove(x)
            if status in (124, 125):
                return status
    while running:
        while all(x.state != "Z" for x in running):
            yield ("wait", running)
        done = [x for x in running if x.state == "Z"]
        for x in done:
            account(x)
            running.remove(x)
    return status
