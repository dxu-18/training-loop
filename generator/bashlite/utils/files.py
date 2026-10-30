"""File management: ls tree cp mv rm mkdir rmdir touch ln readlink realpath basename dirname stat file chmod chown
chgrp du df mktemp dd truncate install."""

from __future__ import annotations

import math
import posixpath
import re

from ..exec import UsageError, gnu_getopt, util
from ..kernel import NS
from ..timeutil import DateError, ls_time, parse_date, stat_time
from ..vfs import FSError, mode_string

DEVNUM = {"null": (1, 3), "zero": (1, 5), "full": (1, 7), "random": (1, 8), "urandom": (1, 9), "tty": (5, 0)}
SAFE = re.compile(r"[A-Za-z0-9_./,:@%+=\-]+\Z")


def shell_quote(name: str) -> str:
    """GNU ls/stat 'shell-escape' quoting."""
    if name and SAFE.match(name) and not name.startswith("~") and not name.startswith("#"):
        return name
    if any(ord(c) < 32 for c in name):
        out = []
        for c in name:
            if ord(c) < 32:
                out.append("'$'\\" + {"\n": "n", "\t": "t", "\r": "r"}.get(c, "%03o" % ord(c)) + "''")
            else:
                out.append(c)
        return ("'" + "".join(out) + "'").replace("''", "")
    if "'" in name:
        if not any(c in name for c in '"$`\\!'):
            return '"' + name + '"'
        return "'" + name.replace("'", "'\\''") + "'"
    return "'" + name + "'"


def human_size(n: float, si=False) -> str:
    """GNU -h: ceiling, one decimal below 10."""
    base = 1000 if si else 1024
    units = "kMGTPE" if si else "KMGTPE"
    if n < base:
        return str(int(n))
    x = float(n)
    for u in units:
        x /= base
        if x < base or u == units[-1]:
            if x < 10:
                v = math.ceil(x * 10 - 1e-9) / 10
                if v >= 10:
                    return f"{int(math.ceil(x))}{u}"
                return f"{v:.1f}{u}"
            v = math.ceil(x - 1e-9)
            if v >= base and u != units[-1]:
                continue
            return f"{v}{u}"
    return str(n)


def _err_fs(ctx, what, path, e):
    ctx.err(f"{what} '{path}': {e.msg}" if what else f"'{path}': {e.msg}")


# ---------------------------------------------------------------------------- ls

class Entry:
    __slots__ = ("name", "path", "node", "target_node", "err")

    def __init__(self, name, path, node, target_node=None, err=None):
        self.name, self.path, self.node, self.target_node, self.err = name, path, node, target_node, err


def _ls_indicator(node, target, style):
    if style is None:
        return ""
    t = node.type
    if t == "d":
        return "/"
    if style == "slash":
        return ""
    if t == "l":
        return "@" if style == "classify" else ""
    if t == "p":
        return "|"
    if t == "s":
        return "="
    if t == "f" and node.mode & 0o111 and style == "classify":
        return "*"
    return ""


COLORS = {"d": "01;34", "l": "01;36", "x": "01;32", "c": "40;33;01", "p": "40;33", "s": "01;35"}
ARCHIVE_EXT = (".tar", ".tgz", ".gz", ".zip", ".xz", ".bz2", ".7z", ".zst")
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".webp")


def _color(node, name):
    t = node.type
    if t == "d":
        if node.mode & 0o1000 and node.mode & 0o002:
            return "30;42"
        return COLORS["d"]
    if t == "l":
        return COLORS["l"]
    if t in ("c", "p", "s"):
        return COLORS[t]
    if node.mode & 0o111:
        return COLORS["x"]
    low = name.lower()
    if low.endswith(ARCHIVE_EXT):
        return "01;31"
    if low.endswith(IMAGE_EXT):
        return "01;35"
    if low.endswith((".mp3", ".wav", ".flac", ".ogg")):
        return "00;36"
    return None


@util("ls", "dir", "vdir")
def u_ls(ctx):
    try:
        opts, ops = gnu_getopt(ctx.argv[1:], "aAbBcCdfFghiklLmnopqQrRsStuUvxX1T:w:I:",
                               ["all", "almost-all", "color=?", "colour=?", "directory", "classify=?", "file-type",
                                "human-readable", "si", "inode", "dereference", "numeric-uid-gid", "reverse",
                                "recursive", "size", "sort=", "time=", "time-style=", "full-time", "help", "version",
                                "group-directories-first", "indicator-style=", "literal", "quote-name",
                                "quoting-style=", "width=", "ignore=", "hide=", "block-size=", "kibibytes",
                                "escape", "no-group", "author", "zero", "hyperlink=?", "tabsize=", "show-control-chars",
                                "dereference-command-line", "format="], name="ls")
    except UsageError as e:
        ctx.err(e.msg)
        return ctx.usage(status=2)
    tty = ctx.stdout_is_tty()
    o = {"all": False, "almost": False, "long": False, "human": False, "si": False, "inode": False, "size": False,
         "dir": False, "recursive": False, "reverse": False, "sort": "name", "time": "mtime", "ind": None,
         "fmt": "columns" if tty else "single", "owner": True, "group": True, "numeric": False, "color": False,
         "deref": False, "style": None, "quote": "shell-escape" if tty else "literal", "dirs_first": False,
         "width": ctx.columns, "ignore": [], "hide": [], "kib": False, "tab": 8}
    for k, v in opts:
        if k in ("-a", "--all"):
            o["all"] = True
            o["almost"] = False
        elif k in ("-A", "--almost-all"):
            o["almost"] = True
            o["all"] = False
        elif k in ("-l",):
            o["fmt"] = "long"
        elif k == "-g":
            o["fmt"] = "long"
            o["owner"] = False
        elif k == "-o":
            o["fmt"] = "long"
            o["group"] = False
        elif k in ("-G", "--no-group"):
            o["group"] = False
        elif k in ("-n", "--numeric-uid-gid"):
            o["fmt"] = "long"
            o["numeric"] = True
        elif k in ("-h", "--human-readable"):
            o["human"] = True
        elif k == "--si":
            o["human"] = o["si"] = True
        elif k in ("-i", "--inode"):
            o["inode"] = True
        elif k in ("-s", "--size"):
            o["size"] = True
        elif k in ("-d", "--directory"):
            o["dir"] = True
        elif k in ("-R", "--recursive"):
            o["recursive"] = True
        elif k in ("-r", "--reverse"):
            o["reverse"] = True
        elif k == "-t":
            o["sort"] = "time"
        elif k == "-S":
            o["sort"] = "size"
        elif k == "-X":
            o["sort"] = "extension"
        elif k == "-v":
            o["sort"] = "version"
        elif k == "-U":
            o["sort"] = "none"
        elif k == "-f":
            o["sort"] = "none"
            o["all"] = True
        elif k == "--sort":
            o["sort"] = {"none": "none", "size": "size", "time": "time", "version": "version", "extension": "extension",
                         "name": "name", "width": "width"}.get(v, "name")
        elif k == "-c":
            o["time"] = "ctime"
        elif k == "-u":
            o["time"] = "atime"
        elif k == "--time":
            o["time"] = {"atime": "atime", "access": "atime", "use": "atime", "ctime": "ctime", "status": "ctime",
                         "birth": "mtime", "creation": "mtime"}.get(v, "mtime")
        elif k == "-1":
            o["fmt"] = "single"
        elif k == "-C":
            o["fmt"] = "columns"
        elif k == "-x":
            o["fmt"] = "across"
        elif k == "-m":
            o["fmt"] = "commas"
        elif k == "--format":
            o["fmt"] = {"long": "long", "verbose": "long", "single-column": "single", "commas": "commas",
                        "across": "across", "horizontal": "across", "vertical": "columns"}.get(v, "columns")
        elif k in ("-F", "--classify"):
            if v in (None, "always", "yes", "force") or (v in ("auto", "tty", "if-tty") and tty):
                o["ind"] = "classify"
        elif k == "--file-type":
            o["ind"] = "file-type"
        elif k == "-p":
            o["ind"] = "slash"
        elif k == "--indicator-style":
            o["ind"] = {"none": None, "slash": "slash", "file-type": "file-type", "classify": "classify"}.get(v)
        elif k in ("--color", "--colour"):
            o["color"] = v in ("always", "yes", "force") or (v in (None, "auto", "tty", "if-tty") and tty) if v != "never" else False
            if v is None:
                o["color"] = True
        elif k in ("-L", "--dereference"):
            o["deref"] = True
        elif k == "--time-style":
            o["style"] = v.replace("posix-", "")
        elif k == "--full-time":
            o["fmt"] = "long"
            o["style"] = "full-iso"
        elif k in ("-N", "--literal"):
            o["quote"] = "literal"
        elif k in ("-Q", "--quote-name"):
            o["quote"] = "c"
        elif k == "-q":
            pass
        elif k == "-b":
            o["quote"] = "escape"
        elif k == "--quoting-style":
            o["quote"] = v
        elif k in ("-w", "--width"):
            o["width"] = int(v) or 10 ** 9
        elif k == "--group-directories-first":
            o["dirs_first"] = True
        elif k in ("-I", "--ignore"):
            o["ignore"].append(v)
        elif k == "--hide":
            o["hide"].append(v)
        elif k in ("-k", "--kibibytes"):
            o["kib"] = True
        elif k in ("-T", "--tabsize"):
            o["tab"] = int(v)
        elif k == "--help":
            ctx.w("Usage: ls [OPTION]... [FILE]...\nList information about the FILEs (the current directory by default).\n"
                  "Sort entries alphabetically if none of -cftuvSUX nor --sort is specified.\n\n"
                  "  -a, --all                  do not ignore entries starting with .\n"
                  "  -A, --almost-all           do not list implied . and ..\n"
                  "  -d, --directory            list directories themselves, not their contents\n"
                  "  -F, --classify[=WHEN]      append indicator (one of */=>@|) to entries WHEN\n"
                  "  -h, --human-readable       with -l and -s, print sizes like 1K 234M 2G etc.\n"
                  "  -i, --inode                print the index number of each file\n"
                  "  -l                         use a long listing format\n"
                  "  -r, --reverse              reverse order while sorting\n"
                  "  -R, --recursive            list subdirectories recursively\n"
                  "  -s, --size                 print the allocated size of each file, in blocks\n"
                  "  -S                         sort by file size, largest first\n"
                  "  -t                         sort by time, newest first; see --time\n"
                  "  -1                         list one file per line\n"
                  "      --help        display this help and exit\n")
            return 0
        elif k == "--version":
            ctx.w("ls (GNU coreutils) 9.4\n")
            return 0
    if o["fmt"] == "long" and o["style"] is None:
        o["style"] = ctx.env.get("TIME_STYLE")
    if not ops:
        ops = ["."]
    ls = LS(ctx, o)
    status = 0
    files, dirs = [], []
    for op in ops:
        try:
            node = ctx.fs.lookup(op, ctx.cwd, False)
        except FSError as e:
            ctx.err(f"cannot access '{op}': {e.msg}")
            status = 2
            continue
        target = node
        if node.type == "l":
            try:
                target = ctx.fs.lookup(op, ctx.cwd, True)
            except FSError:
                target = None
        follow = o["deref"] or (o["fmt"] != "long" and not o["ind"] and not o["dir"]) or (
            o["fmt"] == "long" and op.endswith("/"))
        if node.type == "l" and target is not None and target.type == "d" and not o["dir"] and (follow or o["fmt"] != "long"):
            dirs.append((op, target))
            continue
        if node.type == "d" and not o["dir"]:
            dirs.append((op, node))
        else:
            shown = target if (o["deref"] and target is not None) else node
            files.append(Entry(op, op, shown, target))
    first = True
    if files:
        ls.print_entries(ls.sort(files), None)
        first = False
    multi = len(ops) > 1 or o["recursive"]
    for path, node in ls.sort_dirs(dirs):
        st = yield from ls.list_dir(path, node, multi, first)
        status = max(status, st)
        first = False
    return status


class LS:
    def __init__(self, ctx, o):
        self.ctx, self.o = ctx, o
        self.fs = ctx.fs
        self.host = ctx.host

    def sort_dirs(self, dirs):
        ents = [Entry(p, p, n) for p, n in dirs]
        return [(e.path, e.node) for e in self.sort(ents)]

    def sort(self, ents):
        o = self.o
        key_name = lambda e: e.name.encode("utf-8", "surrogateescape")  # noqa: E731
        s = o["sort"]
        if s == "none":
            res = list(ents)
        elif s == "time":
            res = sorted(ents, key=lambda e: (-getattr(e.node, o["time"]), key_name(e)))
        elif s == "size":
            res = sorted(ents, key=lambda e: (-e.node.size, key_name(e)))
        elif s == "extension":
            res = sorted(ents, key=lambda e: (posixpath.splitext(e.name)[1] if not e.name.startswith(".") or e.name.count(".") > 1 else "", key_name(e)))
        elif s == "version":
            res = sorted(ents, key=lambda e: version_key(e.name))
        elif s == "width":
            res = sorted(ents, key=lambda e: (len(e.name), key_name(e)))
        else:
            res = sorted(ents, key=key_name)
        if o["reverse"]:
            res.reverse()
        if o["dirs_first"]:
            res = [e for e in res if (e.target_node or e.node).type == "d"] + [e for e in res if (e.target_node or e.node).type != "d"]
        return res

    def name_text(self, e: Entry, raw=False):
        o = self.o
        q = o["quote"]
        name = e.name
        if q == "shell-escape" or q == "shell" or q == "shell-always" or q == "shell-escape-always":
            text = shell_quote(name) if q != "shell-always" else "'" + name + "'"
        elif q == "c":
            text = '"' + name.replace("\\", "\\\\").replace('"', '\\"') + '"'
        elif q == "escape":
            text = re.sub(r"([ \\])", r"\\\1", name)
        else:
            text = name
        node = e.node
        ind = _ls_indicator(node, e.target_node, o["ind"]) if node is not None else ""
        if o["color"] and node is not None:
            code = _color(node, name)
            if code:
                text = f"\x1b[{code}m{text}\x1b[0m"
        return text + ind

    def visible_len(self, s):
        return len(re.sub(r"\x1b\[[0-9;]*m", "", s))

    def list_dir(self, path, node, show_header, first):
        ctx, o = self.ctx, self.o
        status = 0
        if not self.fs.can(node, 4, ctx.uid, ctx.proc.gids):
            ctx.err(f"cannot open directory '{path}': Permission denied")
            return 2
        if not first:
            ctx.w("\n")
        if show_header:
            ctx.w((shell_quote(path) if o["quote"] == "shell-escape" else path) + ":\n")
        names = list(node.entries) if o["sort"] == "none" else self.fs.listdir(node)
        ents = []
        if o["all"]:
            ents.append(Entry(".", path, node))
            ents.append(Entry("..", path, node.parent))
        for n in names:
            if n.startswith(".") and not (o["all"] or o["almost"]):
                continue
            if any(_glob(p, n) for p in o["ignore"]) or (not (o["all"] or o["almost"]) and any(_glob(p, n) for p in o["hide"])):
                continue
            child = node.entries[n]
            target = child
            if child.type == "l":
                try:
                    target = self.fs.lookup(_join(path, n), ctx.cwd, True)
                except FSError:
                    target = None
            shown = target if (o["deref"] and target is not None) else child
            ents.append(Entry(n, _join(path, n), shown, target))
        ents = self.sort(ents)
        self.print_entries(ents, node)
        yield from ctx.maybe_flush()
        if o["recursive"]:
            for e in ents:
                if e.name in (".", "..") or e.node.type != "d":
                    continue
                st = yield from self.list_dir(e.path, e.node, True, False)
                status = max(status, st)
        return status

    def print_entries(self, ents, dirnode):
        ctx, o = self.ctx, self.o
        fmt = o["fmt"]
        if fmt == "long" or o["size"]:
            if dirnode is not None:
                blocks = sum(e.node.blocks for e in ents if e.node is not None) // 2
                ctx.w(f"total {self.size_text(blocks * 1024) if o['human'] else blocks}\n")
        texts = [self.name_text(e) for e in ents]
        if o["quote"] == "shell-escape" and any(t.startswith(("'", '"')) or "\x1b[" in t and "'" in t[:12] for t in texts):
            texts = [t if t.startswith(("'", '"')) else " " + t for t in texts]
        prefix = [""] * len(ents)
        if o["inode"]:
            w = max((len(str(e.node.ino)) for e in ents), default=0)
            prefix = [p + f"{e.node.ino:>{w}} " for p, e in zip(prefix, ents)]
        if o["size"]:
            sizes = [self.size_text(e.node.blocks * 512) if o["human"] else str(e.node.blocks // 2) for e in ents]
            w = max((len(s) for s in sizes), default=0)
            prefix = [p + f"{s:>{w}} " for p, s in zip(prefix, sizes)]
        if fmt == "long":
            self.long(ents, texts, prefix)
        elif fmt == "single":
            for p, t in zip(prefix, texts):
                ctx.w(p + t + "\n")
        elif fmt == "commas":
            line = ", ".join(p + t for p, t in zip(prefix, texts))
            ctx.w(_wrap_commas([p + t for p, t in zip(prefix, texts)], o["width"]) if texts else "")
            _ = line
        else:
            self.columns([p + t for p, t in zip(prefix, texts)], fmt == "columns")

    def size_text(self, n):
        return human_size(n, self.o["si"])

    def long(self, ents, texts, prefix):
        ctx, o = self.ctx, self.o
        rows = []
        for e, t in zip(ents, texts):
            node = e.node
            if node is None:
                continue
            mode = mode_string(node)
            nl = str(node.nlink)
            owner = str(node.uid) if o["numeric"] else self.host.user_name(node.uid)
            group = str(node.gid) if o["numeric"] else self.host.group_name(node.gid)
            if node.type == "c":
                ma, mi = DEVNUM.get(node.dev, (1, 3))
                size = f"{ma}, {mi}"
            else:
                size = self.size_text(node.size) if o["human"] else str(node.size)
            when = ls_time(getattr(node, o["time"]), ctx.now, ctx.tz, o["style"])
            name = t
            if node.type == "l":
                tgt = node.target
                tnode = e.target_node
                shown_t = shell_quote(tgt) if o["quote"] == "shell-escape" else tgt
                if o["color"]:
                    if tnode is None:
                        shown_t = f"\x1b[40;31;01m{shown_t}\x1b[0m"
                    else:
                        c = _color(tnode, tgt)
                        if c:
                            shown_t = f"\x1b[{c}m{shown_t}\x1b[0m"
                name = name + " -> " + shown_t
                if o["ind"] == "classify" and tnode is not None:
                    name += _ls_indicator(tnode, None, "classify") if tnode.type != "l" else ""
            rows.append((mode, nl, owner, group, size, when, name))
        if not rows:
            return
        wn = max(len(r[1]) for r in rows)
        wo = max(len(r[2]) for r in rows)
        wg = max(len(r[3]) for r in rows)
        ws = max(len(r[4]) for r in rows)
        out = []
        for p, (mode, nl, owner, group, size, when, name) in zip(prefix, rows):
            cols = [mode, nl.rjust(wn)]
            if o["owner"]:
                cols.append(owner.ljust(wo))
            if o["group"]:
                cols.append(group.ljust(wg))
            cols += [size.rjust(ws), when, name]
            out.append(p + " ".join(cols))
        ctx.w("".join(x + "\n" for x in out))

    def columns(self, texts, vertical):
        ctx, o = self.ctx, self.o
        n = len(texts)
        if n == 0:
            return
        width = o["width"]
        lens = [self.visible_len(t) for t in texts]
        max_cols = max(1, min(n, width // 3))
        best = 1
        col_arrs = {}
        for cols in range(1, max_cols + 1):
            rows = (n + cols - 1) // cols
            arr = [3] * cols
            line_len = 3 * cols
            valid = True
            for i, L in enumerate(lens):
                idx = i // rows if vertical else i % cols
                real = L + (0 if idx == cols - 1 else 2)
                if arr[idx] < real:
                    line_len += real - arr[idx]
                    arr[idx] = real
                    if line_len >= width:
                        valid = False
                        break
            if valid:
                best = cols
                col_arrs[cols] = arr
        cols = best
        arr = col_arrs.get(cols, [max(lens)])
        rows = (n + cols - 1) // cols
        tab = o["tab"]
        out = []
        for r in range(rows):
            pos = 0
            line = []
            if vertical:
                idxs = list(range(r, n, rows))
            else:
                idxs = list(range(r * cols, min(n, (r + 1) * cols)))
            for c, i in enumerate(idxs):
                line.append(texts[i])
                if c == len(idxs) - 1:
                    break
                frm = pos + lens[i]
                to = pos + arr[c]
                line.append(_indent(frm, to, tab))
                pos = to
            out.append("".join(line))
        ctx.w("".join(x + "\n" for x in out))


def _indent(frm, to, tab):
    out = []
    while frm < to:
        if tab and to // tab > (frm + 1) // tab:
            out.append("\t")
            frm += tab - frm % tab
        else:
            out.append(" ")
            frm += 1
    return "".join(out)


def _wrap_commas(items, width):
    lines, cur = [], ""
    for i, it in enumerate(items):
        piece = it + ("," if i < len(items) - 1 else "")
        if cur and len(cur) + 1 + len(piece) > width:
            lines.append(cur)
            cur = piece
        else:
            cur = (cur + " " + piece) if cur else piece
    lines.append(cur)
    return "\n".join(lines) + "\n"


def _glob(pat, name):
    from ..pattern import fnmatch
    return fnmatch(pat, name)


def _join(base, name):
    if base.endswith("/"):
        return base + name
    return base + "/" + name


def version_key(s: str):
    parts = re.split(r"(\d+)", s)
    return [(0, int(p)) if p.isdigit() else (1, p) for p in parts if p != ""]


# ---------------------------------------------------------------------------- tree

@util("tree")
def u_tree(ctx):
    a = ctx.argv[1:]
    level = None
    show_all = dirs_only = full = noreport = classify = False
    ignore = []
    pattern = None
    paths = []
    i = 0
    while i < len(a):
        x = a[i]
        if x == "-L" and i + 1 < len(a):
            try:
                level = int(a[i + 1])
            except ValueError:
                ctx.e("tree: Invalid level, must be greater than 0.\n")
                return 1
            i += 2
            continue
        if x == "-I" and i + 1 < len(a):
            ignore += a[i + 1].split("|")
            i += 2
            continue
        if x == "-P" and i + 1 < len(a):
            pattern = a[i + 1]
            i += 2
            continue
        if x.startswith("-") and len(x) > 1 and not x.startswith("--"):
            for c in x[1:]:
                if c == "a":
                    show_all = True
                elif c == "d":
                    dirs_only = True
                elif c == "f":
                    full = True
                elif c == "F":
                    classify = True
                elif c in "Cnhspug":
                    pass
                else:
                    ctx.e(f"tree: Invalid argument -`{c}'.\n")
                    return 1
            i += 1
            continue
        if x == "--noreport":
            noreport = True
            i += 1
            continue
        if x == "--dirsfirst" or x.startswith("--"):
            i += 1
            continue
        paths.append(x)
        i += 1
    if not paths:
        paths = ["."]
    ndirs = nfiles = 0
    for root in paths:
        try:
            node = ctx.fs.lookup(root, ctx.cwd)
        except FSError:
            ctx.w(f"{root}  [error opening dir]\n")
            continue
        ctx.w(root + "\n")
        if node.type != "d":
            continue

        def walk(n, path, prefix, depth):
            nonlocal ndirs, nfiles
            names = [x for x in ctx.fs.listdir(n) if show_all or not x.startswith(".")]
            names = [x for x in names if not any(_glob(p, x) for p in ignore)]
            items = []
            for name in names:
                child = n.entries[name]
                target = child
                if child.type == "l":
                    try:
                        target = ctx.fs.lookup(_join(path, name), ctx.cwd)
                    except FSError:
                        target = None
                is_dir = target is not None and target.type == "d"
                if dirs_only and not is_dir:
                    continue
                if pattern and not is_dir and not _glob(pattern, name):
                    continue
                items.append((name, child, target, is_dir))
            for k, (name, child, target, is_dir) in enumerate(items):
                last = k == len(items) - 1
                shown = _join(path, name) if full else name
                mark = ""
                if classify and target is not None:      # -F: / for directories, * for executables
                    mark = "/" if is_dir else "*" if target.type == "f" and target.mode & 0o111 else ""
                if child.type == "l":
                    shown += " -> " + child.target + mark
                else:
                    shown += mark
                ctx.w(prefix + ("└── " if last else "├── ") + shown + "\n")
                if is_dir:
                    ndirs += 1
                    if child.type != "l" and (level is None or depth + 1 < level):
                        walk(child, _join(path, name), prefix + ("    " if last else "│   "), depth + 1)
                else:
                    nfiles += 1
        walk(node, root, "", 0)
    if not noreport:
        d = f"{ndirs} director{'y' if ndirs == 1 else 'ies'}"
        if dirs_only:
            ctx.w(f"\n{d}\n")
        else:
            ctx.w(f"\n{d}, {nfiles} file{'' if nfiles == 1 else 's'}\n")
    return 0
    yield


# ---------------------------------------------------------------------------- mkdir rmdir touch

def parse_mode(spec: str, old: int, is_dir: bool, umask: int) -> int:
    """chmod-style mode (octal or symbolic). Raises ValueError."""
    if re.fullmatch(r"[0-7]{1,4}", spec):
        new = int(spec, 8)
        if is_dir and len(spec) < 4:
            new |= old & 0o6000   # GNU keeps setuid/setgid bits on directories for short octal modes
        return new
    mode = old
    for clause in spec.split(","):
        m = re.fullmatch(r"([ugoa]*)((?:[-+=](?:[rwxXst]*|[ugo]))+)", clause)
        if not m:
            raise ValueError(spec)
        who = m.group(1) or ""
        for op, perms in re.findall(r"([-+=])([rwxXst]*|[ugo])", m.group(2)):
            if perms in ("u", "g", "o"):
                src = {"u": (mode >> 6) & 7, "g": (mode >> 3) & 7, "o": mode & 7}[perms]
                bits_rwx = src
                bits = 0
                for w in (who or "ugo"):
                    shift = {"u": 6, "g": 3, "o": 0, "a": None}[w]
                    if shift is None:
                        bits |= (src << 6) | (src << 3) | src
                    else:
                        bits |= src << shift
                _ = bits_rwx
            else:
                bits = 0
                targets = who or "a"
                for w in targets:
                    shifts = (6, 3, 0) if w == "a" else ({"u": 6, "g": 3, "o": 0}[w],)
                    for sh in shifts:
                        for p in perms:
                            if p == "r":
                                bits |= 4 << sh
                            elif p == "w":
                                bits |= 2 << sh
                            elif p == "x":
                                bits |= 1 << sh
                            elif p == "X":
                                if is_dir or old & 0o111:
                                    bits |= 1 << sh
                    if "s" in perms:
                        if w in ("u", "a"):
                            bits |= 0o4000
                        if w in ("g", "a"):
                            bits |= 0o2000
                    if "t" in perms and w in ("o", "a"):
                        bits |= 0o1000
                if not who:
                    bits &= ~umask | 0o7000
            if op == "+":
                mode |= bits
            elif op == "-":
                mode &= ~bits
            else:
                clear = 0
                for w in (who or "a"):
                    shifts = (6, 3, 0) if w == "a" else ({"u": 6, "g": 3, "o": 0}[w],)
                    for sh in shifts:
                        clear |= 7 << sh
                    if w in ("u", "a"):
                        clear |= 0o4000
                    if w in ("g", "a"):
                        clear |= 0o2000 if not is_dir else 0
                    if w in ("o", "a"):
                        clear |= 0o1000
                mode = (mode & ~clear) | bits
    return mode


@util("mkdir")
def u_mkdir(ctx):
    try:
        opts, ops = gnu_getopt(ctx.argv[1:], "pvm:Z", ["parents", "verbose", "mode=", "help", "version"], name="mkdir")
    except UsageError as e:
        ctx.err(e.msg)
        return ctx.usage()
    parents = any(k in ("-p", "--parents") for k, _ in opts)
    verbose = any(k in ("-v", "--verbose") for k, _ in opts)
    mode_s = next((v for k, v in opts if k in ("-m", "--mode")), None)
    if not ops:
        ctx.err("missing operand")
        return ctx.usage()
    status = 0
    um = ctx.proc.umask
    for path in ops:
        mode = 0o777 & ~um
        if mode_s:
            try:
                mode = parse_mode(mode_s, 0o777 & ~um if not re.fullmatch(r"[0-7]+", mode_s) else 0, True, um) if not re.fullmatch(r"[0-7]+", mode_s) else int(mode_s, 8)
            except ValueError:
                ctx.err(f"invalid mode '{mode_s}'")
                return 1
        try:
            if parents:
                created = []
                p = "" if not path.startswith("/") else "/"
                for c in [x for x in path.split("/") if x]:
                    p = p + c if p in ("", "/") else p + "/" + c
                    if not ctx.fs.exists(p, ctx.cwd):
                        created.append(p)
                node = ctx.fs.mkdir(path, ctx.cwd, 0o777 & ~um, ctx.uid, ctx.proc.gid, parents=True, gids=ctx.proc.gids)
                for c in created:
                    n = ctx.fs.lookup(c, ctx.cwd)
                    n.uid, n.gid = ctx.uid, ctx.proc.gid
                    if verbose:
                        ctx.w(f"mkdir: created directory '{c}'\n")
                if mode_s:
                    node.mode = mode
            else:
                node = ctx.fs.mkdir(path, ctx.cwd, mode, ctx.uid, ctx.proc.gid, gids=ctx.proc.gids)
                node.mode = mode
                if verbose:
                    ctx.w(f"mkdir: created directory '{path}'\n")
        except FSError as e:
            ctx.err(f"cannot create directory '{path}': {e.msg}")
            status = 1
    return status
    yield


@util("rmdir")
def u_rmdir(ctx):
    a = ctx.argv[1:]
    parents = "-p" in a or "--parents" in a
    verbose = "-v" in a or "--verbose" in a
    ignore = "--ignore-fail-on-non-empty" in a
    ops = [x for x in a if not x.startswith("-")]
    if not ops:
        ctx.err("missing operand")
        return ctx.usage()
    status = 0
    for path in ops:
        targets = [path]
        if parents:
            p = path.rstrip("/")
            while "/" in p:
                p = p.rsplit("/", 1)[0]
                if p:
                    targets.append(p)
        for t in targets:
            try:
                if verbose:
                    ctx.w(f"rmdir: removing directory, '{t}'\n")
                ctx.fs.rmdir(t, ctx.cwd, ctx.uid, ctx.proc.gids)
            except FSError as e:
                if not (ignore and e.code == "ENOTEMPTY"):
                    ctx.err(f"failed to remove '{t}': {e.msg}")
                    status = 1
                break
    return status
    yield


def _touch_stamp(s: str, now_ns: int, tz):
    """-t [[CC]YY]MMDDhhmm[.ss]"""
    from datetime import datetime
    from ..timeutil import localize, local_dt, to_ns
    m = re.fullmatch(r"(\d{2,4})?(\d{2})(\d{2})(\d{2})(\d{2})(?:\.(\d{2}))?", s)
    if not m or len(s.split(".")[0]) not in (8, 10, 12):
        raise DateError(s)
    head = s.split(".")[0]
    year = local_dt(now_ns, tz).year
    if len(head) == 12:
        year = int(head[:4])
        head = head[4:]
    elif len(head) == 10:
        yy = int(head[:2])
        year = 1900 + yy if yy >= 69 else 2000 + yy
        head = head[2:]
    mo, d, h, mi = int(head[:2]), int(head[2:4]), int(head[4:6]), int(head[6:8])
    sec = int(m.group(6) or 0)
    return to_ns(localize(datetime(year, mo, d, h, mi, sec), tz))


@util("touch")
def u_touch(ctx):
    try:
        opts, ops = gnu_getopt(ctx.argv[1:], "acd:fhmr:t:", ["no-create", "date=", "reference=", "time=",
                                                            "no-dereference", "help", "version"], name="touch")
    except UsageError as e:
        ctx.err(e.msg)
        return ctx.usage()
    when = ctx.now
    no_create = only_a = only_m = False
    for k, v in opts:
        if k in ("-c", "--no-create"):
            no_create = True
        elif k == "-a":
            only_a = True
        elif k == "-m":
            only_m = True
        elif k in ("-d", "--date"):
            try:
                when = parse_date(v, ctx.now, ctx.tz)
            except DateError:
                ctx.err(f"invalid date format '{v}'")
                return 1
        elif k == "-t":
            try:
                when = _touch_stamp(v, ctx.now, ctx.tz)
            except DateError:
                ctx.err(f"invalid date format '{v}'")
                return 1
        elif k in ("-r", "--reference"):
            try:
                when = ctx.lookup(v).mtime
            except FSError as e:
                ctx.err(f"failed to get attributes of '{v}': {e.msg}")
                return 1
        elif k == "--time":
            if v in ("access", "atime", "use"):
                only_a = True
            else:
                only_m = True
    if not ops:
        ctx.err("missing file operand")
        return ctx.usage()
    status = 0
    for path in ops:
        try:
            node = ctx.fs.lookup(path, ctx.cwd, True, ctx.uid, ctx.proc.gids)
        except FSError as e:
            if e.code != "ENOENT":
                ctx.err(f"cannot touch '{path}': {e.msg}")
                status = 1
                continue
            if no_create:
                continue
            try:
                node = ctx.fs.create(path, ctx.cwd, 0o666 & ~ctx.proc.umask, ctx.uid, ctx.proc.gid, ctx.proc.gids)
            except FSError as e2:
                ctx.err(f"cannot touch '{path}': {e2.msg}")
                status = 1
                continue
        else:
            if node.uid != ctx.uid and ctx.uid != 0 and not ctx.fs.can(node, 2, ctx.uid, ctx.proc.gids):
                ctx.err(f"cannot touch '{path}': Permission denied")
                status = 1
                continue
        if not only_m:
            node.atime = when
        if not only_a:
            node.mtime = when
        node.ctime = ctx.now
    return status
    yield


# ---------------------------------------------------------------------------- cp mv rm ln

def _prompt(ctx, question: str):
    """Ask on stderr, read the answer from stdin (GNU coreutils). True for y/Y."""
    from ..fileio import readline
    ctx.e(question)
    line = yield from readline(ctx.k, ctx.proc.fds.get(0))
    ans = line.decode("utf-8", "replace").strip()
    return ans[:1] in ("y", "Y")


def _copy_node(ctx, src_node, src_path, dst_path, opts, depth=0):
    """Copy one file or tree. Returns status."""
    fs = ctx.fs
    status = 0
    preserve = opts.get("preserve")
    try:
        dst = fs.lookup(dst_path, ctx.cwd, not opts.get("no_deref_dst"))
    except FSError:
        dst = None
    if src_node.type == "d":
        if not opts.get("recursive"):
            ctx.err(f"-r not specified; omitting directory '{src_path}'")
            return 1
        if dst is not None and dst.type != "d":
            ctx.err(f"cannot overwrite non-directory '{dst_path}' with directory '{src_path}'")
            return 1
        if dst is None:
            try:
                dst = fs.mkdir(dst_path, ctx.cwd, 0o700, ctx.uid, ctx.proc.gid, gids=ctx.proc.gids)
                dst.mode = src_node.mode & 0o7777 if preserve else src_node.mode & 0o777 & ~ctx.proc.umask
            except FSError as e:
                ctx.err(f"cannot create directory '{dst_path}': {e.msg}")
                return 1
            if opts.get("verbose"):
                ctx.w(f"'{src_path}' -> '{dst_path}'\n")
        p = dst
        while True:
            if p is src_node:
                ctx.err(f"cannot copy a directory, '{src_path}', into itself, '{dst_path}'")
                return 1
            if p is fs.root:
                break
            p = p.parent
        for name in fs.listdir(src_node):
            child = src_node.entries[name]
            st = yield from _copy_node(ctx, child, _join(src_path, name), _join(dst_path, name), opts, depth + 1)
            status = max(status, st)
        if preserve:
            dst.mtime, dst.atime = src_node.mtime, src_node.atime
        return status
    if src_node.type == "l" and (opts.get("no_deref") or (depth > 0 and opts.get("recursive") and not opts.get("deref"))):
        if dst is not None:
            if opts.get("no_clobber"):
                return 0
            fs.unlink(dst_path, ctx.cwd)
        fs.symlink(src_node.target, dst_path, ctx.cwd, ctx.uid, ctx.proc.gid)
        if opts.get("verbose"):
            ctx.w(f"'{src_path}' -> '{dst_path}'\n")
        return 0
    if src_node.type == "l":
        try:
            src_node = fs.lookup(src_path, ctx.cwd, True)
        except FSError as e:
            ctx.err(f"cannot stat '{src_path}': {e.msg}")
            return 1
    if dst is not None:
        if dst is src_node:
            ctx.err(f"'{src_path}' and '{dst_path}' are the same file")
            return 1
        if dst.type == "d":
            ctx.err(f"cannot overwrite directory '{dst_path}' with non-directory")
            return 1
        if opts.get("no_clobber"):
            if opts.get("verbose"):
                ctx.w(f"skipped '{dst_path}'\n")
            return 0
        if opts.get("update") and dst.mtime >= src_node.mtime:
            return 0
        if opts.get("interactive"):
            ok = yield from _prompt(ctx, f"{ctx.name}: overwrite '{dst_path}'? ")
            if not ok:
                return 0
        if opts.get("backup"):
            try:
                fs.rename(dst_path, dst_path + "~", ctx.cwd)
            except FSError:
                pass
            dst = None
    if not fs.can(src_node, 4, ctx.uid, ctx.proc.gids):
        ctx.err(f"cannot open '{src_path}' for reading: Permission denied")
        return 1
    if opts.get("link"):
        try:
            fs.link(src_path, dst_path, ctx.cwd)
        except FSError as e:
            ctx.err(f"cannot create hard link '{dst_path}' to '{src_path}': {e.msg}")
            return 1
        return 0
    if opts.get("symlink"):
        try:
            fs.symlink(src_path, dst_path, ctx.cwd, ctx.uid, ctx.proc.gid)
        except FSError as e:
            ctx.err(f"cannot create symbolic link '{dst_path}': {e.msg}")
            return 1
        return 0
    try:
        if dst is None:
            node = fs.create(dst_path, ctx.cwd, 0o600, ctx.uid, ctx.proc.gid, ctx.proc.gids)
            node.mode = src_node.mode & 0o7777 if preserve else src_node.mode & 0o777 & ~ctx.proc.umask
        else:
            node = dst
            if not fs.can(node, 2, ctx.uid, ctx.proc.gids):
                ctx.err(f"cannot create regular file '{dst_path}': Permission denied")
                return 1
    except FSError as e:
        ctx.err(f"cannot create regular file '{dst_path}': {e.msg}")
        return 1
    node.data[:] = src_node.data if src_node.type == "f" else b""
    node.mtime = node.ctime = ctx.now
    if preserve:
        node.mtime, node.atime = src_node.mtime, src_node.atime
        node.mode = src_node.mode
        if ctx.uid == 0:
            node.uid, node.gid = src_node.uid, src_node.gid
    if opts.get("verbose"):
        ctx.w(f"'{src_path}' -> '{dst_path}'\n")
    return 0
    yield


def _targets(ctx, ops, target_dir, no_target_dir, name):
    """(sources, destination, dest_is_dir) or an error message."""
    if target_dir is not None:
        try:
            if ctx.fs.lookup(target_dir, ctx.cwd).type != "d":
                return f"target '{target_dir}' is not a directory"
        except FSError as e:
            return f"target directory '{target_dir}': {e.msg}" if e.code != "ENOENT" else f"target '{target_dir}': No such file or directory"
        return ops, target_dir, True
    if len(ops) < 2:
        if not ops:
            return "missing file operand"
        return f"missing destination file operand after '{ops[0]}'"
    *srcs, dst = ops
    is_dir = False
    try:
        is_dir = ctx.fs.lookup(dst, ctx.cwd).type == "d" and not no_target_dir
    except FSError:
        pass
    if len(srcs) > 1 and not is_dir:
        return f"target '{dst}' is not a directory"
    return srcs, dst, is_dir


@util("cp")
def u_cp(ctx):
    try:
        opts, ops = gnu_getopt(ctx.argv[1:], "abdfHilLnPprRsSt:Tuvx", [
            "archive", "backup=?", "force", "interactive", "link", "dereference", "no-clobber", "no-dereference",
            "preserve=?", "recursive", "symbolic-link", "target-directory=", "no-target-directory", "update=?",
            "verbose", "one-file-system", "parents", "help", "version", "suffix=", "sparse=", "reflink=?",
            "attributes-only", "strip-trailing-slashes", "remove-destination"], name="cp")
    except UsageError as e:
        ctx.err(e.msg)
        return ctx.usage()
    o = {}
    target_dir = None
    no_target = False
    for k, v in opts:
        if k in ("-r", "-R", "--recursive"):
            o["recursive"] = True
        elif k in ("-a", "--archive"):
            o["recursive"] = o["preserve"] = o["no_deref"] = True
        elif k in ("-p",):
            o["preserve"] = True
        elif k == "--preserve":
            o["preserve"] = True
        elif k in ("-i", "--interactive"):
            o["interactive"] = True
            o.pop("no_clobber", None)
        elif k in ("-n", "--no-clobber"):
            o["no_clobber"] = True
            o.pop("interactive", None)
        elif k in ("-f", "--force"):
            o["force"] = True
        elif k in ("-v", "--verbose"):
            o["verbose"] = True
        elif k in ("-u", "--update"):
            o["update"] = True
        elif k in ("-l", "--link"):
            o["link"] = True
        elif k in ("-s", "--symbolic-link"):
            o["symlink"] = True
        elif k in ("-d", "-P", "--no-dereference"):
            o["no_deref"] = True
        elif k in ("-L", "--dereference"):
            o["deref"] = True
        elif k in ("-b", "--backup"):
            o["backup"] = True
        elif k in ("-t", "--target-directory"):
            target_dir = v
        elif k in ("-T", "--no-target-directory"):
            no_target = True
        elif k == "--help":
            ctx.w("Usage: cp [OPTION]... [-T] SOURCE DEST\n  or:  cp [OPTION]... SOURCE... DIRECTORY\n"
                  "  or:  cp [OPTION]... -t DIRECTORY SOURCE...\nCopy SOURCE to DEST, or multiple SOURCE(s) to DIRECTORY.\n")
            return 0
    t = _targets(ctx, ops, target_dir, no_target, "cp")
    if isinstance(t, str):
        ctx.err(t)
        return ctx.usage() if "missing" in t else 1
    srcs, dst, is_dir = t
    status = 0
    for src in srcs:
        try:
            node = ctx.fs.lookup(src, ctx.cwd, not o.get("no_deref"), ctx.uid, ctx.proc.gids)
        except FSError as e:
            ctx.err(f"cannot stat '{src}': {e.msg}")
            status = 1
            continue
        target = _join(dst, posixpath.basename(src.rstrip("/")) or src) if is_dir else dst
        st = yield from _copy_node(ctx, node, src.rstrip("/") if src not in ("/",) else src, target, o)
        status = max(status, st)
    return status


@util("mv")
def u_mv(ctx):
    try:
        opts, ops = gnu_getopt(ctx.argv[1:], "bfinSt:TuvZ", ["backup=?", "force", "interactive", "no-clobber",
                                                            "target-directory=", "no-target-directory", "update=?",
                                                            "verbose", "help", "version", "suffix=",
                                                            "strip-trailing-slashes"], name="mv")
    except UsageError as e:
        ctx.err(e.msg)
        return ctx.usage()
    o = {}
    target_dir = None
    no_target = False
    for k, v in opts:
        if k in ("-i", "--interactive"):
            o["interactive"] = True
            o.pop("no_clobber", None)
            o.pop("force", None)
        elif k in ("-n", "--no-clobber"):
            o["no_clobber"] = True
            o.pop("interactive", None)
        elif k in ("-f", "--force"):
            o["force"] = True
            o.pop("interactive", None)
            o.pop("no_clobber", None)
        elif k in ("-v", "--verbose"):
            o["verbose"] = True
        elif k in ("-u", "--update"):
            o["update"] = True
        elif k in ("-b", "--backup"):
            o["backup"] = True
        elif k in ("-t", "--target-directory"):
            target_dir = v
        elif k in ("-T", "--no-target-directory"):
            no_target = True
    t = _targets(ctx, ops, target_dir, no_target, "mv")
    if isinstance(t, str):
        ctx.err(t)
        return ctx.usage() if "missing" in t else 1
    srcs, dst, is_dir = t
    fs = ctx.fs
    status = 0
    for src in srcs:
        try:
            node = fs.lookup(src, ctx.cwd, False)
        except FSError as e:
            ctx.err(f"cannot stat '{src}': {e.msg}")
            status = 1
            continue
        target = _join(dst, posixpath.basename(src.rstrip("/"))) if is_dir else dst
        try:
            existing = fs.lookup(target, ctx.cwd, False)
        except FSError:
            existing = None
        if existing is not None:
            if existing is node:
                ctx.err(f"'{src}' and '{target}' are the same file")
                status = 1
                continue
            if o.get("no_clobber"):
                continue
            if o.get("update") and existing.mtime >= node.mtime:
                continue
            if o.get("interactive"):
                ok = yield from _prompt(ctx, f"mv: overwrite '{target}'? ")
                if not ok:
                    continue
            if existing.type == "d" and node.type != "d":
                ctx.err(f"cannot overwrite directory '{target}' with non-directory")
                status = 1
                continue
            if existing.type != "d" and node.type == "d":
                ctx.err(f"cannot overwrite non-directory '{target}' with directory '{src}'")
                status = 1
                continue
            if o.get("backup"):
                fs.rename(target, target + "~", ctx.cwd)
        try:
            fs.rename(src.rstrip("/") or src, target, ctx.cwd, ctx.uid, ctx.proc.gids)
            node.ctime = ctx.now
        except FSError as e:
            if e.code == "EINVAL":
                ctx.err(f"cannot move '{src}' to a subdirectory of itself, '{target}'")
            elif e.code == "ENOTEMPTY":
                ctx.err(f"cannot move '{src}' to '{target}': Directory not empty")
            else:
                ctx.err(f"cannot move '{src}' to '{target}': {e.msg}")
            status = 1
            continue
        if o.get("verbose"):
            ctx.w(f"renamed '{src}' -> '{target}'\n")
    return status


def _kind_name(node):
    if node.type == "d":
        return "directory"
    if node.type == "l":
        return "symbolic link"
    if node.type == "f":
        return "regular empty file" if not node.data else "regular file"
    if node.type == "c":
        return "character special file"
    if node.type == "p":
        return "fifo"
    return "file"


@util("rm")
def u_rm(ctx):
    try:
        opts, ops = gnu_getopt(ctx.argv[1:], "dfiIrRv", ["force", "interactive=?", "recursive", "dir", "verbose",
                                                        "one-file-system", "no-preserve-root", "preserve-root=?",
                                                        "help", "version"], name="rm")
    except UsageError as e:
        ctx.err(e.msg)
        return ctx.usage()
    force = recursive = interactive = verbose = dirs = once = False
    for k, v in opts:
        if k in ("-f", "--force"):
            force = True
            interactive = False
        elif k in ("-i",):
            interactive = True
            force = False
        elif k == "--interactive":
            interactive = v in (None, "always")
            once = v == "once"
        elif k == "-I":
            once = True
        elif k in ("-r", "-R", "--recursive"):
            recursive = True
        elif k in ("-d", "--dir"):
            dirs = True
        elif k in ("-v", "--verbose"):
            verbose = True
        elif k == "--help":
            ctx.w("Usage: rm [OPTION]... [FILE]...\nRemove (unlink) the FILE(s).\n")
            return 0
    if not ops:
        if force:
            return 0
        ctx.err("missing operand")
        return ctx.usage()
    if once and (len(ops) > 3 or recursive):
        what = f"remove {len(ops)} argument{'s' if len(ops) > 1 else ''}{' recursively' if recursive else ''}?"
        ok = yield from _prompt(ctx, f"rm: {what} ")
        if not ok:
            return 0
    fs = ctx.fs
    status = 0

    def remove(path, node):
        nonlocal status
        if node.type == "d":
            if not recursive:
                if dirs:
                    if node.entries:
                        ctx.err(f"cannot remove '{path}': Directory not empty")
                        status = 1
                        return
                else:
                    ctx.err(f"cannot remove '{path}': Is a directory")
                    status = 1
                    return
            if interactive and node.entries:
                ok = yield from _prompt(ctx, f"rm: descend into directory '{path}'? ")
                if not ok:
                    return
            for name in fs.listdir(node):
                yield from remove(_join(path, name), node.entries[name])
            if interactive:
                ok = yield from _prompt(ctx, f"rm: remove directory '{path}'? ")
                if not ok:
                    return
            try:
                fs.rmdir(path, ctx.cwd, ctx.uid, ctx.proc.gids)
            except FSError as e:
                if not (force and e.code == "ENOENT"):
                    ctx.err(f"cannot remove '{path}': {e.msg}")
                    status = 1
                return
            if verbose:
                ctx.w(f"removed directory '{path}'\n")
            return
        if interactive:
            ok = yield from _prompt(ctx, f"rm: remove {_kind_name(node)} '{path}'? ")
            if not ok:
                return
        elif not force and node.type == "f" and not fs.can(node, 2, ctx.uid, ctx.proc.gids) and ctx.stdin_is_tty():
            ok = yield from _prompt(ctx, f"rm: remove write-protected {_kind_name(node)} '{path}'? ")
            if not ok:
                return
        try:
            fs.unlink(path, ctx.cwd, ctx.uid, ctx.proc.gids)
        except FSError as e:
            ctx.err(f"cannot remove '{path}': {e.msg}")
            status = 1
            return
        if verbose:
            ctx.w(f"removed '{path}'\n")
    for path in ops:
        base = path.rstrip("/").rsplit("/", 1)[-1]
        if base in (".", ".."):
            ctx.err(f"refusing to remove '.' or '..' directory: skipping '{path}'")
            status = 1
            continue
        if path.rstrip("/") == "" and path.startswith("/"):
            ctx.err("it is dangerous to operate recursively on '/'")
            ctx.err("use --no-preserve-root to override this failsafe")
            status = 1
            continue
        try:
            node = fs.lookup(path, ctx.cwd, False)
        except FSError as e:
            if not force:
                ctx.err(f"cannot remove '{path}': {e.msg}")
                status = 1
            continue
        yield from remove(path, node)
    return status


@util("ln")
def u_ln(ctx):
    try:
        opts, ops = gnu_getopt(ctx.argv[1:], "bdfFinrsSt:TvLP", ["backup=?", "force", "interactive", "no-dereference",
                                                                 "relative", "symbolic", "target-directory=",
                                                                 "no-target-directory", "verbose", "logical", "physical",
                                                                 "help", "version", "suffix="], name="ln")
    except UsageError as e:
        ctx.err(e.msg)
        return ctx.usage()
    sym = force = verbose = relative = no_deref = False
    target_dir = None
    for k, v in opts:
        if k in ("-s", "--symbolic"):
            sym = True
        elif k in ("-f", "--force"):
            force = True
        elif k in ("-v", "--verbose"):
            verbose = True
        elif k in ("-r", "--relative"):
            relative = True
        elif k in ("-n", "--no-dereference"):
            no_deref = True
        elif k in ("-t", "--target-directory"):
            target_dir = v
    fs = ctx.fs
    if not ops:
        ctx.err("missing file operand")
        return ctx.usage()
    if target_dir is None and len(ops) == 1:
        srcs, dst, is_dir = ops, ".", True
    else:
        if target_dir is not None:
            srcs, dst = ops, target_dir
        else:
            srcs, dst = ops[:-1], ops[-1]
        try:
            node = fs.lookup(dst, ctx.cwd, not no_deref)
            is_dir = node.type == "d"
        except FSError:
            is_dir = False
        if len(srcs) > 1 and not is_dir:
            ctx.err(f"target '{dst}' is not a directory")
            return 1
    status = 0
    for src in srcs:
        link = _join(dst, posixpath.basename(src.rstrip("/"))) if is_dir else dst
        target = src
        if relative and sym:
            target = posixpath.relpath(ctx.abspath(src), posixpath.dirname(ctx.abspath(link)))
        if fs.exists(link, ctx.cwd, follow=False):
            if not force:
                ctx.err(f"failed to create {'symbolic' if sym else 'hard'} link '{link}'"
                        + (f" -> '{src}'" if False else "") + ": File exists")
                status = 1
                continue
            try:
                n = fs.lookup(link, ctx.cwd, False)
                if n.type == "d":
                    ctx.err(f"'{link}': cannot overwrite directory")
                    status = 1
                    continue
                fs.unlink(link, ctx.cwd)
            except FSError:
                pass
        try:
            if sym:
                fs.symlink(target, link, ctx.cwd, ctx.uid, ctx.proc.gid, ctx.proc.gids)
            else:
                fs.link(src, link, ctx.cwd, ctx.uid, ctx.proc.gids)
        except FSError as e:
            if not sym and e.code == "ENOENT":
                ctx.err(f"failed to access '{src}': {e.msg}")
            elif not sym and e.code == "EPERM":
                ctx.err(f"'{src}': hard link not allowed for directory")
            else:
                ctx.err(f"failed to create {'symbolic' if sym else 'hard'} link '{link}': {e.msg}")
            status = 1
            continue
        if verbose:
            ctx.w(f"'{link}' -> '{target}'\n" if sym else f"'{link}' => '{src}'\n")
    return status
    yield


# ---------------------------------------------------------------------------- names and paths

def canonicalize(fs, path, cwd, mode):
    """readlink -f/-e/-m and realpath. mode: 'e' all exist, 'f' all but last, 'm' none."""
    full = path if path.startswith("/") else (cwd.rstrip("/") + "/" + path)
    comps = [c for c in full.split("/") if c not in ("", ".")]
    resolved = []
    seen = 0
    i = 0
    while i < len(comps):
        c = comps[i]
        if c == "..":
            if resolved:
                resolved.pop()
            i += 1
            continue
        cur = "/" + "/".join(resolved + [c])
        last = i == len(comps) - 1
        try:
            node = fs.lookup(cur, "/", False)
        except FSError as e:
            if mode == "m" or (mode == "f" and last and e.code == "ENOENT"):
                resolved.append(c)
                i += 1
                continue
            raise
        if node.type == "l":
            seen += 1
            if seen > 40:
                raise FSError("ELOOP")
            t = node.target
            tcomps = [x for x in t.split("/") if x not in ("", ".")]
            if t.startswith("/"):
                resolved = []
            comps = comps[:i] + tcomps + comps[i + 1:]
            continue
        if not last and node.type != "d":
            if mode == "m":
                resolved.append(c)
                i += 1
                continue
            raise FSError("ENOTDIR")
        resolved.append(c)
        i += 1
    return "/" + "/".join(resolved)


@util("readlink")
def u_readlink(ctx):
    a = ctx.argv[1:]
    mode = None
    nl = True
    ops = []
    for x in a:
        if x in ("-f", "--canonicalize"):
            mode = "f"
        elif x in ("-e", "--canonicalize-existing"):
            mode = "e"
        elif x in ("-m", "--canonicalize-missing"):
            mode = "m"
        elif x in ("-n", "--no-newline"):
            nl = False
        elif x in ("-v", "-q", "-s", "--verbose", "--quiet", "--silent", "-z"):
            pass
        else:
            ops.append(x)
    if not ops:
        ctx.err("missing operand")
        return ctx.usage()
    status = 0
    for p in ops:
        try:
            if mode:
                out = canonicalize(ctx.fs, p, ctx.cwd, mode)
            else:
                node = ctx.fs.lookup(p, ctx.cwd, False)
                if node.type != "l":
                    status = 1
                    continue
                out = node.target
        except FSError:
            status = 1
            continue
        ctx.w(out + ("\n" if nl or len(ops) > 1 else ""))
    return status
    yield


@util("realpath")
def u_realpath(ctx):
    a = ctx.argv[1:]
    mode = "f"
    strip = False
    rel_to = rel_base = None
    quiet = False
    ops = []
    for x in a:
        if x in ("-e", "--canonicalize-existing"):
            mode = "e"
        elif x in ("-m", "--canonicalize-missing"):
            mode = "m"
        elif x in ("-s", "--strip", "--no-symlinks"):
            strip = True
        elif x.startswith("--relative-to="):
            rel_to = x.split("=", 1)[1]
        elif x.startswith("--relative-base="):
            rel_base = x.split("=", 1)[1]
        elif x in ("-q", "--quiet"):
            quiet = True
        elif x in ("-L", "-P", "-z"):
            pass
        else:
            ops.append(x)
    if not ops:
        ctx.err("missing operand")
        return ctx.usage()
    status = 0
    for p in ops:
        try:
            if strip:
                full = posixpath.normpath(p if p.startswith("/") else ctx.cwd.rstrip("/") + "/" + p)
                if mode == "e" and not ctx.fs.exists(full):
                    raise FSError("ENOENT")
                out = full
            else:
                out = canonicalize(ctx.fs, p, ctx.cwd, mode)
        except FSError as e:
            if not quiet:
                ctx.err(f"{p}: {e.msg}")
            status = 1
            continue
        if rel_to:
            base = canonicalize(ctx.fs, rel_to, ctx.cwd, "m")
            out = posixpath.relpath(out, base)
        elif rel_base:
            base = canonicalize(ctx.fs, rel_base, ctx.cwd, "m")
            if out == base or out.startswith(base.rstrip("/") + "/"):
                out = posixpath.relpath(out, base)
        ctx.w(out + "\n")
    return status
    yield


@util("basename")
def u_basename(ctx):
    a = ctx.argv[1:]
    multi = False
    suffix = None
    zero = False
    ops = []
    i = 0
    while i < len(a):
        x = a[i]
        if x in ("-a", "--multiple"):
            multi = True
        elif x in ("-s",) and i + 1 < len(a):
            suffix = a[i + 1]
            multi = True
            i += 1
        elif x.startswith("--suffix="):
            suffix = x.split("=", 1)[1]
            multi = True
        elif x in ("-z", "--zero"):
            zero = True
        elif x == "--":
            ops += a[i + 1:]
            break
        elif x.startswith("-") and len(x) > 1 and not ops:
            ctx.err(f"invalid option -- '{x[1]}'")
            return ctx.usage()
        else:
            ops.append(x)
        i += 1
    if not ops:
        ctx.err("missing operand")
        return ctx.usage()
    if not multi:
        if len(ops) > 2:
            ctx.err(f"extra operand '{ops[2]}'")
            return ctx.usage()
        if len(ops) == 2:
            suffix = ops[1]
        ops = ops[:1]
    end = "\0" if zero else "\n"
    for p in ops:
        b = _basename(p)
        if suffix and b.endswith(suffix) and b != suffix:
            b = b[:-len(suffix)]
        ctx.w(b + end)
    return 0
    yield


def _basename(p):
    if p == "":
        return ""
    s = p.rstrip("/")
    if s == "":
        return "/"
    return s.rsplit("/", 1)[-1]


def _dirname(p):
    s = p.rstrip("/")
    if s == "":
        return "/" if p.startswith("/") else "."
    if "/" not in s:
        return "."
    d = s.rsplit("/", 1)[0].rstrip("/")
    return d or "/"


@util("dirname")
def u_dirname(ctx):
    ops = [x for x in ctx.argv[1:] if x not in ("-z", "--zero", "--")]
    if not ops:
        ctx.err("missing operand")
        return ctx.usage()
    for p in ops:
        ctx.w(_dirname(p) + "\n")
    return 0
    yield


# ---------------------------------------------------------------------------- stat file

def _ftype(node):
    if node.type == "f":
        return "regular empty file" if not node.data else "regular file"
    return {"d": "directory", "l": "symbolic link", "c": "character special file", "p": "fifo", "s": "socket"}[node.type]


def stat_format(ctx, fmt, path, node, follow_name=True):
    h = ctx.host

    def q(name):
        return shell_quote(name)
    out = []
    i = 0
    while i < len(fmt):
        c = fmt[i]
        if c == "\\" and i + 1 < len(fmt):
            d = fmt[i + 1]
            out.append({"n": "\n", "t": "\t", "\\": "\\", '"': '"'}.get(d, "\\" + d))
            i += 2
            continue
        if c != "%" or i + 1 >= len(fmt):
            out.append(c)
            i += 1
            continue
        m = re.match(r"%(-?\d*)(?:\.(\d+))?([a-zA-Z%])", fmt[i:])
        if not m:
            out.append(c)
            i += 1
            continue
        width, k = m.group(1), m.group(3)
        i += m.end()
        if k == "%":
            v = "%"
        elif k == "n":
            v = path
        elif k == "N":
            v = q(path) + (" -> " + q(node.target) if node.type == "l" else "")
        elif k == "s":
            v = str(node.size)
        elif k == "b":
            v = str(node.blocks)
        elif k == "B":
            v = "512"
        elif k == "o":
            v = "4096"
        elif k == "a":
            v = f"{node.mode & 0o7777:o}"
        elif k == "A":
            v = mode_string(node)
        elif k == "f":
            from ..vfs import TYPE_BITS
            v = f"{TYPE_BITS[node.type] | node.mode:x}"
        elif k == "F":
            v = _ftype(node)
        elif k == "u":
            v = str(node.uid)
        elif k == "U":
            v = h.user_name(node.uid)
        elif k == "g":
            v = str(node.gid)
        elif k == "G":
            v = h.group_name(node.gid)
        elif k == "h":
            v = str(node.nlink)
        elif k == "i":
            v = str(node.ino)
        elif k in ("x", "y", "z", "w"):
            t = {"x": node.atime, "y": node.mtime, "z": node.ctime, "w": node.mtime}[k]
            v = stat_time(t, ctx.tz)
        elif k in ("X", "Y", "Z", "W"):
            t = {"X": node.atime, "Y": node.mtime, "Z": node.ctime, "W": node.mtime}[k]
            v = str(t // NS)
        elif k == "d":
            v = "2050"
        elif k == "D":
            v = "802"
        elif k == "m":
            v = "/"
        elif k == "t":
            v = "1" if node.type == "c" else "0"
        elif k == "T":
            v = "3" if node.type == "c" else "0"
        else:
            v = "?"
        if width:
            w = int(width)
            v = v.ljust(-w) if w < 0 else v.rjust(w)
        out.append(v)
    return "".join(out)


DEFAULT_STAT = ("  File: %N\n  Size: %-10s\tBlocks: %-10b IO Block: %-6o %F\nDevice: %Hd,%Ld\tInode: %-11i Links: %h\n"
                "Access: (%04a/%10.10A)  Uid: (%5u/%8U)   Gid: (%5g/%8G)\nAccess: %x\nModify: %y\nChange: %z\n Birth: %w\n")


@util("stat")
def u_stat(ctx):
    try:
        opts, ops = gnu_getopt(ctx.argv[1:], "c:fLt", ["format=", "printf=", "dereference", "file-system", "terse",
                                                       "help", "version", "cached="], name="stat")
    except UsageError as e:
        ctx.err(e.msg)
        return ctx.usage()
    fmt = None
    deref = False
    fsys = False
    printf = False
    for k, v in opts:
        if k in ("-c", "--format"):
            fmt = v + "\n"
        elif k == "--printf":
            fmt = v
            printf = True
        elif k in ("-L", "--dereference"):
            deref = True
        elif k in ("-f", "--file-system"):
            fsys = True
    if not ops:
        ctx.err("missing operand")
        return ctx.usage()
    status = 0
    for p in ops:
        try:
            node = ctx.fs.lookup(p, ctx.cwd, deref)
        except FSError as e:
            ctx.err(f"cannot stat{'x' if False else ''} '{p}': {e.msg}")
            status = 1
            continue
        if fsys:
            ctx.w(f"  File: \"{p}\"\n    ID: 9a3f0c21b7e4d5f6 Namelen: 255     Type: ext2/ext3\n"
                  f"Block size: 4096       Fundamental block size: 4096\n"
                  f"Blocks: Total: 122722275  Free: 84899204   Available: 78653427\n"
                  f"Inodes: Total: 31227904   Free: 29875310\n")
            continue
        if fmt is None:
            text = DEFAULT_STAT.replace("%Hd,%Ld", "8,2" if node.type != "c" else "0,5")
            text = re.sub(r"%04a", f"{node.mode & 0o7777:04o}", text)
            text = text.replace("%10.10A", mode_string(node)[:10].rjust(10))
            if node.type == "c":
                ma, mi = DEVNUM.get(node.dev, (1, 3))
                text = text.replace("Links: %h\n", f"Links: %h     Device type: {ma},{mi}\n")
            ctx.w(stat_format(ctx, text, p, node))
        else:
            ctx.w(stat_format(ctx, fmt, p, node))
    return status
    yield


def file_type(ctx, path, node, data: bytes) -> str:
    if node.type == "d":
        return "directory"
    if node.type == "l":
        tgt = node.target
        if not ctx.fs.exists(path, ctx.cwd):
            return f"broken symbolic link to {tgt}"
        return f"symbolic link to {tgt}"
    if node.type == "c":
        from .files import DEVNUM as DN
        ma, mi = DN.get(node.dev, (1, 3))
        return f"character special ({ma}/{mi})"
    if node.type == "p":
        return "fifo (named pipe)"
    if not data:
        return "empty"
    if data.startswith(b"\x7fELF"):
        return ("ELF 64-bit LSB pie executable, x86-64, version 1 (SYSV), dynamically linked, interpreter "
                "/lib64/ld-linux-x86-64.so.2, BuildID[sha1]=4f1a0b4c0e5dfa71b6f4cbbd2b2b61e5d6a39b9c, for GNU/Linux 3.2.0, stripped")
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 26:
        w = int.from_bytes(data[16:20], "big")
        hgt = int.from_bytes(data[20:24], "big")
        depth = data[24]
        ctype = {0: "grayscale", 2: "RGB", 3: "colormap", 4: "gray+alpha", 6: "RGBA"}.get(data[25], "RGB")
        return f"PNG image data, {w} x {hgt}, {depth}-bit/color {ctype}, non-interlaced"
    if data.startswith(b"\xff\xd8\xff"):
        return "JPEG image data, JFIF standard 1.01, aspect ratio, density 1x1, segment length 16, baseline, precision 8"
    if data.startswith(b"GIF8"):
        return "GIF image data, version 89a"
    if data.startswith(b"\x1f\x8b"):
        name = ""
        if len(data) > 3 and data[3] & 8:
            end = data.find(b"\0", 10)
            name = data[10:end].decode("latin-1")
        import time as _t
        mt = int.from_bytes(data[4:8], "little")
        from ..timeutil import local_dt, strftime
        when = strftime("%a %b %e %H:%M:%S %Y", local_dt(mt * NS, "UTC")) if mt else ""
        parts = ["gzip compressed data"]
        if name:
            parts.append(f'was "{name}"')
        if mt:
            parts.append(f"last modified: {when}")
        parts.append("from Unix")
        _ = _t
        return ", ".join(parts)
    if data.startswith(b"PK\x03\x04"):
        if b"data.pkl" in data[:2000] or b"archive/" in data[:400]:
            return "Zip archive data, at least v0.0 to extract, compression method=store"
        return "Zip archive data, at least v2.0 to extract, compression method=deflate"
    if data.startswith(b"\xfd7zXZ"):
        return "XZ compressed data, checksum CRC64"
    if data.startswith(b"BZh"):
        return "bzip2 compressed data, block size = 900k"
    if len(data) > 262 and data[257:262] == b"ustar":
        return "POSIX tar archive (GNU)" if data[257:265] == b"ustar  \0" else "POSIX tar archive"
    if data.startswith(b"%PDF-"):
        return f"PDF document, version {data[5:8].decode('latin-1')}"
    if data.startswith(b"\x93NUMPY"):
        return "NumPy data file, version 1.0, description {'descr': '<f4', 'fortran_order': False, 'shape': (1000,), }"
    if data.startswith(b"\x80\x02") or data.startswith(b"\x80\x04"):
        return "data"
    if data.startswith(b"SQLite format 3"):
        return "SQLite 3.x database, last written using SQLite version 3045001"
    if data.startswith(b"RIFF") and data[8:12] == b"WAVE":
        return "RIFF (little-endian) data, WAVE audio, Microsoft PCM, 16 bit, mono 16000 Hz"
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return "data"
    if any(ord(c) < 32 and c not in "\n\t\r\f\b\x1b" for c in text[:4096]):
        return "data"
    ascii_ = all(ord(c) < 128 for c in text)
    enc = "ASCII text" if ascii_ else "Unicode text, UTF-8 text"
    first = text.split("\n", 1)[0]
    if "\r\n" in text:
        enc += ", with CRLF line terminators"
    long_line = max((len(x) for x in text.split("\n")), default=0) > 300
    if long_line:
        enc += ", with very long lines (" + str(max(len(x) for x in text.split("\n"))) + ")"
    if not text.endswith("\n") and "\n" not in text:
        enc += ", with no line terminators"
    exe = ", ASCII text executable" if ascii_ else ", Unicode text, UTF-8 text executable"
    if first.startswith("#!"):
        interp = first[2:].strip().split()
        prog = interp[-1] if interp and interp[0].endswith("/env") else (interp[0].rsplit("/", 1)[-1] if interp else "")
        prog = prog.rsplit("/", 1)[-1]
        kind = {"bash": "Bourne-Again shell script", "sh": "POSIX shell script", "python3": "Python script",
                "python": "Python script", "perl": "Perl script", "node": "Node.js script", "zsh": "Paul Falstad's zsh script"}.get(prog, f"a {prog} script")
        return kind + (exe if node.mode & 0o111 or True else ", " + enc)
    stripped = text.strip()
    if (stripped.startswith("{") and stripped.endswith("}")) or (stripped.startswith("[") and stripped.endswith("]")):
        import json
        try:
            json.loads(stripped)
            return "JSON text data"
        except ValueError:
            pass
    if stripped.startswith("<?xml"):
        return "XML 1.0 document, " + enc
    if stripped.lower().startswith("<!doctype html") or stripped.lower().startswith("<html"):
        return "HTML document, " + enc
    lines = [ln for ln in text.split("\n") if ln]
    if len(lines) >= 2 and all("," in ln for ln in lines[:20]):
        counts = {ln.count(",") for ln in lines[:20]}
        if len(counts) == 1:
            return "CSV " + enc
    if re.search(r"^\s*(import |from \S+ import |def \w+\(|class \w+[:(])", text, re.M) and path.endswith(".py"):
        return "Python script, " + enc
    return enc


@util("file")
def u_file(ctx):
    a = ctx.argv[1:]
    brief = "-b" in a or "--brief" in a
    mime = "-i" in a or "--mime" in a or "--mime-type" in a
    follow = "-L" in a
    ops = [x for x in a if not x.startswith("-")]
    if not ops:
        ctx.e("Usage: file [-bcCdEhikLlNnprsSvzZ0] [--apple] [--extension] [--mime-encoding]\n"
              "            [--mime-type] [-e <testname>] [-F <separator>]  [-f <namefile>]\n"
              "            [-m <magicfiles>] [-P <parameter=value>] [--exclude-quiet]\n"
              "            <file> ...\n       file -C [-m <magicfiles>]\n       file [--help]\n")
        return 1
    width = max(len(p) for p in ops) + 1
    for p in ops:
        try:
            node = ctx.fs.lookup(p, ctx.cwd, follow)
            data = bytes(node.data[:65536]) if node.type == "f" else b""
            desc = file_type(ctx, p, node, data)
        except FSError as e:
            desc = f"cannot open `{p}' (No such file or directory)" if e.code == "ENOENT" else f"cannot open `{p}' ({e.msg})"
        if mime:
            desc = _mime(desc)
        if brief:
            ctx.w(desc + "\n")
        else:
            ctx.w(f"{(p + ':').ljust(width)} {desc}\n")
    return 0
    yield


def _mime(desc):
    d = desc.lower()
    if d.startswith("directory"):
        return "inode/directory; charset=binary"
    if d.startswith("empty"):
        return "inode/x-empty; charset=binary"
    if "symbolic link" in d:
        return "inode/symlink; charset=binary"
    if "png" in d:
        return "image/png; charset=binary"
    if "jpeg" in d:
        return "image/jpeg; charset=binary"
    if "gzip" in d:
        return "application/gzip; charset=binary"
    if "zip" in d:
        return "application/zip; charset=binary"
    if "tar archive" in d:
        return "application/x-tar; charset=binary"
    if "json" in d:
        return "application/json; charset=us-ascii"
    if "csv" in d:
        return "text/csv; charset=us-ascii"
    if "python" in d:
        return "text/x-script.python; charset=us-ascii"
    if "shell script" in d:
        return "text/x-shellscript; charset=us-ascii"
    if "utf-8" in d:
        return "text/plain; charset=utf-8"
    if "text" in d:
        return "text/plain; charset=us-ascii"
    if "elf" in d:
        return "application/x-pie-executable; charset=binary"
    return "application/octet-stream; charset=binary"


# ---------------------------------------------------------------------------- chmod chown chgrp

@util("chmod")
def u_chmod(ctx):
    a = ctx.argv[1:]
    recursive = verbose = changes = quiet = False
    rest = []
    for x in a:
        if x in ("-R", "--recursive"):
            recursive = True
        elif x in ("-v", "--verbose"):
            verbose = True
        elif x in ("-c", "--changes"):
            changes = True
        elif x in ("-f", "--silent", "--quiet"):
            quiet = True
        elif x.startswith("--reference="):
            rest.append(x)
        else:
            rest.append(x)
    if len(rest) < 2:
        ctx.err("missing operand" if not rest else f"missing operand after '{rest[0]}'")
        return ctx.usage()
    spec, paths = rest[0], rest[1:]
    if spec.startswith("--reference="):
        try:
            ref = ctx.lookup(spec.split("=", 1)[1])
            spec = f"{ref.mode & 0o7777:o}"
        except FSError as e:
            ctx.err(f"failed to get attributes of '{spec.split('=', 1)[1]}': {e.msg}")
            return 1
    try:
        parse_mode(spec, 0o644, False, ctx.proc.umask)
    except ValueError:
        ctx.err(f"invalid mode: '{spec}'")
        return ctx.usage()
    status = 0
    for p in paths:
        try:
            node = ctx.fs.lookup(p, ctx.cwd, True, ctx.uid, ctx.proc.gids)
        except FSError as e:
            if not quiet:
                ctx.err(f"cannot access '{p}': {e.msg}")
            status = 1
            continue
        todo = [(p, node)]
        if recursive and node.type == "d":
            todo = [(path, n) for path, n in ctx.fs.walk(node, p) if n.type != "l"]
        for path, n in todo:
            if ctx.uid != 0 and n.uid != ctx.uid:
                if not quiet:
                    ctx.err(f"changing permissions of '{path}': Operation not permitted")
                status = 1
                continue
            old = n.mode
            new = parse_mode(spec, old, n.type == "d", ctx.proc.umask)
            n.mode = new
            n.ctime = ctx.now
            if verbose or (changes and old != new):
                if old == new:
                    ctx.w(f"mode of '{path}' retained as {old:04o} ({mode_string(n)[1:]})\n")
                else:
                    tmp = n.mode
                    n.mode = old
                    olds = mode_string(n)[1:]
                    n.mode = tmp
                    ctx.w(f"mode of '{path}' changed from {old:04o} ({olds}) to {new:04o} ({mode_string(n)[1:]})\n")
    return status
    yield


def _chown_like(ctx, group_only):
    a = ctx.argv[1:]
    recursive = verbose = no_deref = False
    rest = []
    for x in a:
        if x in ("-R", "--recursive"):
            recursive = True
        elif x in ("-v", "--verbose", "-c", "--changes"):
            verbose = True
        elif x in ("-h", "--no-dereference"):
            no_deref = True
        elif x in ("-f", "--silent", "--quiet", "-H", "-L", "-P"):
            pass
        else:
            rest.append(x)
    if len(rest) < 2:
        ctx.err("missing operand" if not rest else f"missing operand after '{rest[0]}'")
        return ctx.usage()
    spec, paths = rest[0], rest[1:]
    h = ctx.host
    uid = gid = None
    if group_only:
        gid = h.gid_of(spec)
        if gid is None:
            ctx.err(f"invalid group: '{spec}'")
            return 1
    else:
        user, sep, group = spec.partition(":") if ":" in spec else spec.partition(".")
        if user:
            uid = h.uid_of(user)
            if uid is None:
                ctx.err(f"invalid user: '{spec}'")
                return 1
        if sep and group:
            gid = h.gid_of(group)
            if gid is None:
                ctx.err(f"invalid group: '{spec}'")
                return 1
        elif sep and user:
            gid = h.users[uid].gid if uid in h.users else None
    status = 0
    for p in paths:
        try:
            node = ctx.fs.lookup(p, ctx.cwd, not no_deref)
        except FSError as e:
            ctx.err(f"cannot access '{p}': {e.msg}")
            status = 1
            continue
        todo = [(p, node)] if not (recursive and node.type == "d") else list(ctx.fs.walk(node, p))
        for path, n in todo:
            if ctx.uid != 0 and (uid is not None and uid != n.uid or (gid is not None and gid not in ctx.proc.gids) or n.uid != ctx.uid):
                what = "group" if group_only else "ownership"
                ctx.err(f"changing {what} of '{path}': Operation not permitted")
                status = 1
                continue
            if uid is not None:
                n.uid = uid
            if gid is not None:
                n.gid = gid
            n.ctime = ctx.now
            if verbose:
                ctx.w(f"changed ownership of '{path}' to {spec}\n" if not group_only else f"changed group of '{path}' to {spec}\n")
    return status


@util("chown")
def u_chown(ctx):
    return _chown_like(ctx, False)
    yield


@util("chgrp")
def u_chgrp(ctx):
    return _chown_like(ctx, True)
    yield


# ---------------------------------------------------------------------------- du df

@util("du")
def u_du(ctx):
    try:
        opts, ops = gnu_getopt(ctx.argv[1:], "abcd:hHklLmsSxB:0", [
            "all", "apparent-size", "bytes", "total", "max-depth=", "human-readable", "si", "summarize", "exclude=",
            "one-file-system", "block-size=", "count-links", "dereference", "separate-dirs", "help", "version",
            "threshold=", "time=?", "null", "inodes"], name="du")
    except UsageError as e:
        ctx.err(e.msg)
        return ctx.usage()
    o = {"all": False, "apparent": False, "total": False, "depth": None, "human": False, "si": False, "unit": 1024,
         "exclude": [], "summarize": False}
    for k, v in opts:
        if k in ("-a", "--all"):
            o["all"] = True
        elif k == "--apparent-size":
            o["apparent"] = True
        elif k in ("-b", "--bytes"):
            o["apparent"] = True
            o["unit"] = 1
        elif k in ("-c", "--total"):
            o["total"] = True
        elif k in ("-d", "--max-depth"):
            o["depth"] = int(v)
        elif k in ("-h", "--human-readable"):
            o["human"] = True
        elif k == "--si":
            o["human"] = o["si"] = True
        elif k in ("-s", "--summarize"):
            o["summarize"] = True
            o["depth"] = 0
        elif k == "--exclude":
            o["exclude"].append(v)
        elif k == "-k":
            o["unit"] = 1024
        elif k == "-m":
            o["unit"] = 1024 * 1024
        elif k in ("-B", "--block-size"):
            m = re.fullmatch(r"(\d*)([KMG]?)", v)
            o["unit"] = (int(m.group(1) or 1)) * {"": 1, "K": 1024, "M": 1 << 20, "G": 1 << 30}[m.group(2)] if m else 1024
    if not ops:
        ops = ["."]
    seen = set()
    total = 0
    status = 0

    def fmt(nbytes):
        if o["human"]:
            return human_size(nbytes, o["si"])
        return str(math.ceil(nbytes / o["unit"])) if nbytes else "0"

    def walk(node, path, depth):
        nonlocal status
        if node.ino in seen:
            return 0
        seen.add(node.ino)
        size = node.size if o["apparent"] else node.blocks * 512
        if node.type == "d" and o["apparent"]:
            size = node.size
        if node.type == "d":
            if not ctx.fs.can(node, 5, ctx.uid, ctx.proc.gids):
                ctx.err(f"cannot read directory '{path}': Permission denied")
                status = 1
            else:
                for name in ctx.fs.listdir(node):
                    if any(_glob(p, name) for p in o["exclude"]):
                        continue
                    child = node.entries[name]
                    size += walk(child, _join(path, name), depth + 1)
            if o["depth"] is None or depth <= o["depth"]:
                ctx.w(f"{fmt(size)}\t{path}\n")
        elif o["all"] and (o["depth"] is None or depth <= o["depth"]) or depth == 0:
            ctx.w(f"{fmt(size)}\t{path}\n")
        return size
    for p in ops:
        try:
            node = ctx.fs.lookup(p, ctx.cwd, True)
        except FSError as e:
            ctx.err(f"cannot access '{p}': {e.msg}")
            status = 1
            continue
        total += walk(node, p, 0)
        yield from ctx.maybe_flush()
    if o["total"]:
        ctx.w(f"{fmt(total)}\ttotal\n")
    return status


@util("df")
def u_df(ctx):
    a = ctx.argv[1:]
    human = "-h" in a or "--human-readable" in a or any(x.startswith("-") and not x.startswith("--") and "h" in x for x in a)
    si = "-H" in a or "--si" in a
    typ = "-T" in a or "--print-type" in a or any(x.startswith("-") and not x.startswith("--") and "T" in x for x in a)
    inodes = "-i" in a
    ops = [x for x in a if not x.startswith("-")]
    h = ctx.host
    mounts = h.extra.get("mounts") or default_mounts(h)
    if ops:
        chosen = []
        for p in ops:
            try:
                full = canonicalize(ctx.fs, p, ctx.cwd, "e")
            except FSError as e:
                ctx.err(f"{p}: {e.msg}")
                continue
            best = max((m for m in mounts if full == m[1] or full.startswith(m[1].rstrip("/") + "/") or m[1] == "/"),
                       key=lambda m: len(m[1]))
            if best not in chosen:
                chosen.append(best)
        mounts = chosen
    rows = []
    for dev, mnt, fstype, size, used in mounts:
        if callable(used):
            used = used()
        avail = max(size - used - (size // 20 if fstype == "ext4" else 0), 0)
        pct = "-" if size == 0 else f"{math.ceil(100 * used / (used + avail)) if used + avail else 0}%"
        if human or si:
            vals = [human_size(size, si), human_size(used, si) if used else "0", human_size(avail, si)]
        else:
            vals = [str(size // 1024), str(used // 1024), str(avail // 1024)]
        rows.append([dev] + ([fstype] if typ else []) + vals + [pct, mnt])
    head = ["Filesystem"] + (["Type"] if typ else []) + (["Size", "Used", "Avail", "Use%", "Mounted on"] if (human or si)
                                                          else ["1K-blocks", "Used", "Available", "Use%", "Mounted on"])
    allrows = [head] + rows
    nc = len(head)
    # coreutils' minimum field widths: Filesystem 14, Type 4, sizes 5, Use% 4
    minw = [14] + ([4] if typ else []) + [5, 5, 5, 4, 0]
    widths = [max([minw[i]] + [len(r[i]) for r in allrows]) for i in range(nc)]
    out = []
    for r in allrows:
        cells = []
        for i, c in enumerate(r):
            if i == 0 or (typ and i == 1):
                cells.append(c.ljust(widths[i]))
            elif i == nc - 1:
                cells.append(c)
            else:
                cells.append(c.rjust(widths[i]))
        out.append(" ".join(cells))
    ctx.w("".join(x + "\n" for x in out))
    return 0
    yield


def default_mounts(h):
    G = 1 << 30
    fs = h.fs

    def used_root():
        return 121 * G + fs.du_blocks(fs.root) * 512

    def used_data():
        try:
            return 1200 * G + fs.du_blocks(fs.lookup("/data")) * 512
        except FSError:
            return 1200 * G
    return [("tmpfs", "/run", "tmpfs", 3_336_000 * 1024, 2_150 * 1024),
            ("/dev/nvme0n1p2", "/", "ext4", 490_617_772 * 1024, used_root),
            ("tmpfs", "/dev/shm", "tmpfs", 16_680_000 * 1024, 0),
            ("tmpfs", "/run/lock", "tmpfs", 5120 * 1024, 4 * 1024),
            ("/dev/nvme0n1p1", "/boot/efi", "vfat", 1_098_632 * 1024, 6_220 * 1024),
            ("/dev/sda1", "/data", "ext4", 3_844_550_000 * 1024, used_data)]


# ---------------------------------------------------------------------------- mktemp dd truncate

ALNUM = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


@util("mktemp")
def u_mktemp(ctx):
    a = ctx.argv[1:]
    is_dir = dry = quiet = False
    tmpdir = None
    use_t = False
    suffix = ""
    templates = []
    i = 0
    while i < len(a):
        x = a[i]
        if x in ("-d", "--directory"):
            is_dir = True
        elif x in ("-u", "--dry-run"):
            dry = True
        elif x in ("-q", "--quiet"):
            quiet = True
        elif x == "-t":
            use_t = True
        elif x == "-p" and i + 1 < len(a):
            tmpdir = a[i + 1]
            i += 1
        elif x.startswith("--tmpdir"):
            tmpdir = x.split("=", 1)[1] if "=" in x else ctx.env.get("TMPDIR", "/tmp")
        elif x.startswith("--suffix="):
            suffix = x.split("=", 1)[1]
        elif x.startswith("-") and len(x) > 1:
            for c in x[1:]:
                if c == "d":
                    is_dir = True
                elif c == "u":
                    dry = True
                elif c == "q":
                    quiet = True
                elif c == "t":
                    use_t = True
        else:
            templates.append(x)
        i += 1
    tmpl = templates[0] if templates else "tmp.XXXXXXXXXX"
    m = re.search(r"X{3,}", tmpl)
    if not m or m.end() != len(tmpl) and not suffix and not re.search(r"X{3,}$", tmpl):
        if m is None:
            ctx.err(f"too few X's in template '{tmpl}'")
            return 1
    if not templates or use_t or tmpdir is not None:
        base = tmpdir if tmpdir is not None else ctx.env.get("TMPDIR", "/tmp")
        if not tmpl.startswith("/"):
            tmpl = base.rstrip("/") + "/" + tmpl
    rng = ctx.world.rng
    for _ in range(100):
        name = re.sub(r"X{3,}(?!.*X{3,})", lambda mm: "".join(rng.choice(ALNUM) for _ in mm.group(0)), tmpl) + suffix
        if not ctx.fs.exists(name, ctx.cwd, False):
            break
    if not dry:
        try:
            if is_dir:
                node = ctx.fs.mkdir(name, ctx.cwd, 0o700, ctx.uid, ctx.proc.gid, gids=ctx.proc.gids)
            else:
                node = ctx.fs.create(name, ctx.cwd, 0o600, ctx.uid, ctx.proc.gid, ctx.proc.gids, excl=True)
            node.mode = 0o700 if is_dir else 0o600
        except FSError as e:
            if not quiet:
                ctx.err(f"failed to create {'directory' if is_dir else 'file'} via template '{tmpl}': {e.msg}")
            return 1
    ctx.w(name + "\n")
    return 0
    yield


def _size_arg(s: str) -> int:
    m = re.fullmatch(r"(\d+)([kKMGTbcw]?)(B|iB)?", s)
    if not m:
        raise ValueError(s)
    n = int(m.group(1))
    unit = m.group(2)
    mult = {"": 1, "c": 1, "w": 2, "b": 512, "k": 1024, "K": 1024, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40}[unit]
    if m.group(3) == "B":
        mult = {"": 1, "k": 1000, "K": 1000, "M": 10 ** 6, "G": 10 ** 9, "T": 10 ** 12}.get(unit, mult)
    return n * mult


@util("dd")
def u_dd(ctx):
    params = {}
    for x in ctx.argv[1:]:
        if "=" not in x:
            ctx.err(f"unrecognized operand '{x}'")
            return ctx.usage()
        k, v = x.split("=", 1)
        params[k] = v
    try:
        bs = _size_arg(params.get("bs", params.get("ibs", "512")))
        count = _size_arg(params["count"]) if "count" in params else None
        skip = _size_arg(params.get("skip", "0"))
        seek = _size_arg(params.get("seek", "0"))
    except ValueError as e:
        ctx.err(f"invalid number: '{e.args[0]}'")
        return 1
    src = params.get("if", "-")
    try:
        of_in = ctx.proc.fds.get(0) if src == "-" else ctx.open(src)
    except FSError as e:
        ctx.err(f"failed to open '{src}': {e.msg}")
        return 1
    from ..fileio import read
    total = 0
    nrec = 0
    partial = 0
    chunks = []
    if skip:
        if of_in.kind == "file":
            of_in.pos += skip * bs
        else:
            for _ in range(skip):
                yield from read(ctx.k, of_in, bs)
    while count is None or nrec + partial < count:
        data = yield from read(ctx.k, of_in, bs)
        if not data:
            break
        if len(data) == bs:
            nrec += 1
        else:
            partial += 1
        total += len(data)
        chunks.append(data)
        if total > 512 * 1024 * 1024 or ctx.fs.quota_bytes is not None and total > ctx.fs.quota_bytes:
            break
    blob = b"".join(chunks)
    conv = params.get("conv", "")
    if "ucase" in conv:
        blob = blob.upper()
    if "lcase" in conv:
        blob = blob.lower()
    dst = params.get("of")
    if dst is None:
        yield from ctx.out(blob)
    else:
        try:
            of = ctx.open(dst, "w" if "notrunc" not in conv else "rw")
        except FSError as e:
            ctx.err(f"failed to open '{dst}': {e.msg}")
            return 1
        node = of.inode
        if node is not None:
            start = seek * bs
            ctx.fs.charge(max(0, start + len(blob) - len(node.data)))
            if len(node.data) < start:
                node.data.extend(b"\0" * (start - len(node.data)))
            node.data[start:start + len(blob)] = blob
            node.mtime = ctx.now
    status = params.get("status", "")
    if status not in ("none", "noxfer"):
        ctx.e(f"{nrec}+{partial} records in\n{nrec}+{partial} records out\n")
        secs = 0.000123 + total / 2e9
        rate = total / secs
        ctx.e(f"{total} bytes ({_dd_h(total, 1000)}, {_dd_h(total, 1024, True)}) copied, {secs:.6g} s, {_dd_h(rate, 1000)}/s\n"
              if total >= 1000 else f"{total} bytes copied, {secs:.6g} s, {_dd_h(rate, 1000)}/s\n")
    return 0


def _dd_h(n, base, iec=False):
    units = ["", "kB", "MB", "GB", "TB"] if not iec else ["", "KiB", "MiB", "GiB", "TiB"]
    i = 0
    x = float(n)
    while x >= base and i < len(units) - 1:
        x /= base
        i += 1
    if i == 0:
        return f"{int(x)} B"
    return f"{x:.{1 if x < 10 else 0}f} {units[i]}" if x < 100 else f"{x:.0f} {units[i]}"


@util("truncate")
def u_truncate(ctx):
    a = ctx.argv[1:]
    size = None
    ops = []
    i = 0
    while i < len(a):
        if a[i] == "-s" and i + 1 < len(a):
            size = a[i + 1]
            i += 2
            continue
        if a[i].startswith("--size="):
            size = a[i].split("=", 1)[1]
        elif a[i] != "-c":
            ops.append(a[i])
        i += 1
    if size is None:
        ctx.err("you must specify either '--size' or '--reference'")
        return ctx.usage()
    rel = size[0] if size[0] in "+-" else ""
    try:
        n = _size_arg(size.lstrip("+-"))
    except ValueError:
        ctx.err(f"Invalid number: '{size}'")
        return 1
    for p in ops:
        try:
            node = ctx.fs.lookup(p, ctx.cwd)
        except FSError:
            node = ctx.fs.create(p, ctx.cwd, 0o644 & ~ctx.proc.umask, ctx.uid, ctx.proc.gid)
        cur = len(node.data)
        new = cur + n if rel == "+" else (max(cur - n, 0) if rel == "-" else n)
        if new < cur:
            del node.data[new:]
        else:
            ctx.fs.charge(new - cur)
            node.data.extend(b"\0" * (new - cur))
        node.mtime = ctx.now
    return 0
    yield
