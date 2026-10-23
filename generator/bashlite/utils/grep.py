"""grep / egrep / fgrep / zgrep / rg."""

from __future__ import annotations

import re

from ..exec import UsageError, gnu_getopt, util
from ..fileio import readline
from ..pattern import fnmatch
from ..regex import RegexError, compile_re, translate
from ..vfs import FSError

C_MATCH, C_FILE, C_SEP, C_LINE = "\x1b[01;31m\x1b[K", "\x1b[35m\x1b[K", "\x1b[36m\x1b[K", "\x1b[32m\x1b[K"
C_END = "\x1b[m\x1b[K"

LONG = ["extended-regexp", "fixed-strings", "basic-regexp", "perl-regexp", "regexp=", "file=", "ignore-case",
        "no-ignore-case", "word-regexp", "line-regexp", "null-data", "no-messages", "invert-match", "version", "help",
        "max-count=", "byte-offset", "line-number", "no-line-number", "line-buffered", "with-filename",
        "no-filename", "label=", "only-matching", "quiet", "silent", "binary-files=", "text", "directories=",
        "devices=", "recursive", "dereference-recursive", "include=", "exclude=", "exclude-from=", "exclude-dir=",
        "files-without-match", "files-with-matches", "count", "initial-tab", "null", "before-context=",
        "after-context=", "context=", "color=?", "colour=?", "binary"]


def _legacy_context(args):
    out = []
    for x in args:
        m = re.fullmatch(r"-(\d+)", x)
        out += ["-C", m.group(1)] if m else [x]
    return out


@util("grep", "egrep", "fgrep", "ggrep")
def u_grep(ctx):
    return (yield from grep_main(ctx, ctx.argv[1:]))


def grep_main(ctx, args, decompress=False):
    try:
        opts, ops = gnu_getopt(_legacy_context(args), "EFGPe:f:iywxzsvVm:bnHhoqaIdD:rRLlcTZA:B:C:U", LONG,
                               name="grep")
    except UsageError as e:
        ctx.err(e.msg)
        ctx.e("Usage: grep [OPTION]... PATTERNS [FILE]...\nTry 'grep --help' for more information.\n")
        return 2
    o = {"mode": "E" if ctx.name == "egrep" else ("F" if ctx.name == "fgrep" else "G"), "pats": [], "i": False,
         "w": False, "x": False, "v": False, "c": False, "l": False, "L": False, "n": False, "H": None, "o": False,
         "q": False, "s": False, "m": None, "A": 0, "B": 0, "r": False, "include": [], "exclude": [],
         "exdir": [], "b": False, "color": False, "z": False, "Z": False, "label": "(standard input)", "T": False,
         "binary": "binary", "lb": False}
    have_pat = False
    for k, v in opts:
        if k in ("-E", "--extended-regexp"):
            o["mode"] = "E"
        elif k in ("-F", "--fixed-strings"):
            o["mode"] = "F"
        elif k in ("-G", "--basic-regexp"):
            o["mode"] = "G"
        elif k in ("-P", "--perl-regexp"):
            if ctx.host.os == "macos" and ctx.name == "grep":
                ctx.err("invalid option -- P")
                ctx.e("usage: grep [-abcdDEFGHhIiJLlMmnOopqRSsUVvwXxZz] [-A num] [-B num] [-C[num]]\n"
                      "\t[-e pattern] [-f file] [--binary-files=value] [--color=when]\n"
                      "\t[--context[=num]] [--directories=action] [--label] [--line-buffered]\n"
                      "\t[--null] [pattern] [file ...]\n")
                return 2
            o["mode"] = "P"
        elif k in ("-e", "--regexp"):
            o["pats"] += v.split("\n")
            have_pat = True
        elif k in ("-f", "--file"):
            try:
                text = yield from ctx.text(v)
            except FSError as e:
                ctx.err(f"{v}: {e.msg}")
                return 2
            o["pats"] += [x for x in text.split("\n")[:-1 if text.endswith("\n") else None]]
            have_pat = True
        elif k in ("-i", "-y", "--ignore-case"):
            o["i"] = True
        elif k == "--no-ignore-case":
            o["i"] = False
        elif k in ("-w", "--word-regexp"):
            o["w"] = True
        elif k in ("-x", "--line-regexp"):
            o["x"] = True
        elif k in ("-v", "--invert-match"):
            o["v"] = True
        elif k in ("-c", "--count"):
            o["c"] = True
        elif k in ("-l", "--files-with-matches"):
            o["l"] = True
        elif k in ("-L", "--files-without-match"):
            o["L"] = True
        elif k in ("-n", "--line-number"):
            o["n"] = True
        elif k == "--no-line-number":
            o["n"] = False
        elif k in ("-H", "--with-filename"):
            o["H"] = True
        elif k in ("-h", "--no-filename"):
            o["H"] = False
        elif k in ("-o", "--only-matching"):
            o["o"] = True
        elif k in ("-q", "--quiet", "--silent"):
            o["q"] = True
        elif k in ("-s", "--no-messages"):
            o["s"] = True
        elif k in ("-m", "--max-count"):
            o["m"] = int(v)
        elif k in ("-A", "--after-context"):
            o["A"] = int(v)
        elif k in ("-B", "--before-context"):
            o["B"] = int(v)
        elif k in ("-C", "--context"):
            o["A"] = o["B"] = int(v)
        elif k in ("-r", "--recursive", "-R", "--dereference-recursive"):
            o["r"] = True
        elif k == "--include":
            o["include"].append(v)
        elif k == "--exclude":
            o["exclude"].append(v)
        elif k == "--exclude-dir":
            o["exdir"].append(v)
        elif k in ("-b", "--byte-offset"):
            o["b"] = True
        elif k in ("--color", "--colour"):
            o["color"] = v in ("always", "yes", "force") or (v in (None, "auto", "tty", "if-tty") and ctx.stdout_is_tty())
        elif k in ("-z", "--null-data"):
            o["z"] = True
        elif k in ("-Z", "--null"):
            o["Z"] = True
        elif k == "--label":
            o["label"] = v
        elif k in ("-T", "--initial-tab"):
            o["T"] = True
        elif k in ("-a", "--text"):
            o["binary"] = "text"
        elif k == "-I":
            o["binary"] = "without-match"
        elif k == "--binary-files":
            o["binary"] = v
        elif k == "--line-buffered":
            o["lb"] = True
        elif k in ("-V", "--version"):
            ctx.w("grep (GNU grep) 3.11\n")
            return 0
        elif k == "--help":
            ctx.w("Usage: grep [OPTION]... PATTERNS [FILE]...\nSearch for PATTERNS in each FILE.\n"
                  "Example: grep -i 'hello world' menu.h main.c\nPATTERNS can contain multiple patterns separated by newlines.\n\n"
                  "Pattern selection and interpretation:\n  -E, --extended-regexp     PATTERNS are extended regular expressions\n"
                  "  -F, --fixed-strings       PATTERNS are strings\n  -i, --ignore-case         ignore case distinctions in patterns and data\n"
                  "  -w, --word-regexp         match only whole words\n  -x, --line-regexp         match only whole lines\n\n"
                  "Miscellaneous:\n  -v, --invert-match        select non-matching lines\n\nOutput control:\n"
                  "  -n, --line-number         print line number with output lines\n  -o, --only-matching       show only nonempty parts of lines that match\n"
                  "  -r, --recursive           like --directories=recurse\n  -l, --files-with-matches  print only names of FILEs with selected lines\n"
                  "  -c, --count               print only a count of selected lines per FILE\n\n"
                  "Context control:\n  -B, --before-context=NUM  print NUM lines of leading context\n"
                  "  -A, --after-context=NUM   print NUM lines of trailing context\n  -C, --context=NUM         print NUM lines of output context\n")
            return 0
    if not have_pat:
        if not ops:
            ctx.e("Usage: grep [OPTION]... PATTERNS [FILE]...\nTry 'grep --help' for more information.\n")
            return 2
        o["pats"] = ops[0].split("\n")
        ops = ops[1:]
    try:
        rx = build_regex(o)
    except RegexError as e:
        ctx.err(str(e))
        return 2
    files = ops
    if not files:
        files = ["."] if o["r"] else ["-"]
        implicit_dot = o["r"]
    else:
        implicit_dot = False
    targets = []
    errors = False
    for f in files:
        if f == "-":
            targets.append(("-", None))
            continue
        try:
            node = ctx.fs.lookup(f, ctx.cwd)
        except FSError as e:
            if not o["s"]:
                ctx.err(f"{f}: {e.msg}")
            errors = True
            continue
        if node.type == "d":
            if not o["r"]:
                if not o["s"]:
                    ctx.err(f"{f}: Is a directory")
                errors = True
                continue
            for path, n in _walk(ctx, f, node, o):
                shown = path[2:] if implicit_dot and path.startswith("./") else path
                targets.append((shown, n))
        else:
            if o["include"] and not any(fnmatch(g, f.rsplit("/", 1)[-1]) for g in o["include"]) and o["r"]:
                continue
            targets.append((f, node))
    show_name = o["H"] if o["H"] is not None else (len(targets) > 1 or o["r"])
    any_match = False
    for name, node in targets:
        try:
            if name == "-":
                of = ctx.proc.fds.get(0)
                stream = name
            else:
                of = ctx.open(name if not implicit_dot or name.startswith("/") else name)
                stream = name
        except FSError as e:
            if not o["s"]:
                ctx.err(f"{name}: {e.msg}")
            errors = True
            continue
        label = o["label"] if name == "-" else name
        matched = yield from grep_stream(ctx, of, label, rx, o, show_name, decompress)
        any_match = any_match or matched
        if o["q"] and any_match:
            return 0
    if errors and not (o["q"] and any_match):
        return 2
    return 0 if any_match else 1


def build_regex(o):
    pats = o["pats"]
    mode = o["mode"]
    if mode == "F":
        parts = [re.escape(p) for p in pats]
    else:
        dialect = {"G": "bre", "E": "ere", "P": "pcre"}[mode]
        parts = []
        for p in pats:
            parts.append(translate(p, dialect))
    if any(p == "" for p in pats):
        body = "|".join(f"(?:{p})" for p in parts) if len(parts) > 1 else parts[0]
    else:
        body = "|".join(f"(?:{p})" for p in parts) if len(parts) > 1 else parts[0]
    if o["x"]:
        body = f"^(?:{body})$"
    elif o["w"]:
        body = rf"(?<![A-Za-z0-9_])(?:{body})(?![A-Za-z0-9_])"
    flags = re.I if o["i"] else 0
    try:
        return re.compile(body, flags)
    except re.error as e:
        msg = str(e)
        if "unterminated character set" in msg:
            raise RegexError("Unmatched [, [^, [:, [., or [=") from None
        if "missing )" in msg:
            raise RegexError("Unmatched ( or \\(") from None
        raise RegexError("Invalid regular expression") from None


def _walk(ctx, path, node, o):
    fs = ctx.fs
    for name in fs.listdir(node):
        child = node.entries[name]
        p = path.rstrip("/") + "/" + name
        if child.type == "l":
            continue   # -r does not follow symlinks found while recursing
        if child.type == "d":
            if any(fnmatch(g, name) for g in o["exdir"]):
                continue
            yield from _walk(ctx, p, child, o)
        elif child.type == "f":
            if o["include"] and not any(fnmatch(g, name) for g in o["include"]):
                continue
            if any(fnmatch(g, name) for g in o["exclude"]):
                continue
            yield p, child


def grep_stream(ctx, of, name, rx, o, show_name, decompress=False):
    """Search one input. Returns True if a line was selected."""
    data = None
    if of.kind == "file":
        data = bytes(of.inode.data[of.pos:])
        of.pos = len(of.inode.data)
    elif not o["lb"] and not (of.kind == "pipe" and (o["q"] or o["m"])):
        chunks = []
        from ..fileio import read
        while True:
            d = yield from read(ctx.k, of, -1)
            if not d:
                break
            chunks.append(d)
        data = b"".join(chunks)
    if decompress and data is not None and data[:2] == b"\x1f\x8b":
        import gzip
        try:
            data = gzip.decompress(data)
        except Exception:  # noqa: BLE001
            pass
    sep = b"\0" if o["z"] else b"\n"
    binary = data is not None and b"\0" in data[:32768] and o["binary"] != "text" and not o["z"]
    if binary and o["binary"] == "without-match":
        return False
    color = o["color"]
    count = 0
    matched_any = False
    last_printed = -1
    before = []
    after_left = 0
    lineno = 0
    offset = 0
    printed_sep = False
    out = []

    def prefix(ln, off, sep_char):
        parts = []
        if show_name:
            parts.append((C_FILE + name + C_END) if color else name)
        if o["n"]:
            parts.append((C_LINE + str(ln) + C_END) if color else str(ln))
        if o["b"]:
            parts.append(str(off))
        if not parts:
            return ""
        s = (C_SEP + sep_char + C_END if color else sep_char).join(parts)
        if show_name and o["Z"]:
            s = s.replace(name + (C_END if color else "") + (C_SEP + sep_char + C_END if color else sep_char), name + "\0", 1)
            return s if o["n"] or o["b"] else name + "\0"
        return s + ((C_SEP + sep_char + C_END) if color else sep_char) + ("\t" if o["T"] else "")

    def emit_ctx(ln, off, text):
        nonlocal last_printed, printed_sep
        if last_printed >= 0 and ln > last_printed + 1 and (o["A"] or o["B"]):
            out.append((C_SEP + "--" + C_END if color else "--") + "\n")
        out.append(prefix(ln, off, "-") + text + "\n")
        last_printed = ln

    def lines_iter():
        if data is not None:
            pos = 0
            n = len(data)
            while pos < n:
                j = data.find(sep, pos)
                if j < 0:
                    yield data[pos:], pos
                    break
                yield data[pos:j], pos
                pos = j + 1
    gen = lines_iter() if data is not None else None
    stop = False
    while not stop:
        if gen is not None:
            try:
                raw, off = next(gen)
            except StopIteration:
                break
        else:
            line_b = yield from readline(ctx.k, of, sep)
            if not line_b:
                break
            raw = line_b[:-1] if line_b.endswith(sep) else line_b
            off = offset
            offset += len(line_b)
        lineno += 1
        text = raw.decode("utf-8", "surrogateescape")
        m = rx.search(text)
        selected = (m is not None) != o["v"]
        if selected:
            count += 1
            matched_any = True
            if o["q"]:
                return True
            if o["l"] or o["L"]:
                break
            if binary:
                if not o["c"]:
                    pass
                if o["m"] is not None and count >= o["m"]:
                    break
                continue
            if not o["c"]:
                for bl_ln, bl_off, bl_text in before:
                    if bl_ln > last_printed:
                        emit_ctx(bl_ln, bl_off, bl_text)
                before = []
                if last_printed >= 0 and lineno > last_printed + 1 and (o["A"] or o["B"]):
                    out.append((C_SEP + "--" + C_END if color else "--") + "\n")
                if o["o"]:
                    if not o["v"]:
                        for mm in rx.finditer(text):
                            s0 = mm.start("_K") if "_K" in rx.groupindex and mm.group("_K") is not None else mm.start()
                            piece = text[s0:mm.end()]
                            if not piece:
                                continue
                            out.append(prefix(lineno, off + len(text[:s0].encode()), ":") +
                                       ((C_MATCH + piece + C_END) if color else piece) + "\n")
                else:
                    shown = text
                    if color and not o["v"]:
                        shown = rx.sub(lambda mm: (C_MATCH + mm.group(0) + C_END) if mm.group(0) else "", text)
                    out.append(prefix(lineno, off, ":") + shown + "\n")
                last_printed = lineno
                after_left = o["A"]
            if o["m"] is not None and count >= o["m"]:
                # trailing context still printed
                stop = True
                if o["A"] and not o["c"]:
                    k = o["A"]
                    while k > 0:
                        if gen is not None:
                            try:
                                raw2, off2 = next(gen)
                            except StopIteration:
                                break
                        else:
                            lb = yield from readline(ctx.k, of, sep)
                            if not lb:
                                break
                            raw2, off2 = (lb[:-1] if lb.endswith(sep) else lb), offset
                            offset += len(lb)
                        lineno += 1
                        t2 = raw2.decode("utf-8", "surrogateescape")
                        if (rx.search(t2) is not None) != o["v"]:
                            break
                        emit_ctx(lineno, off2, t2)
                        k -= 1
                break
        else:
            if after_left > 0 and not o["c"]:
                emit_ctx(lineno, off, text)
                after_left -= 1
            elif o["B"]:
                before.append((lineno, off, text))
                if len(before) > o["B"]:
                    before.pop(0)
        if out and (o["lb"] or len(out) > 512):
            ctx.w("".join(out))
            out = []
            yield from ctx.flush()
    if o["c"]:
        if show_name:
            out.append(((C_FILE + name + C_END + C_SEP + ":" + C_END) if color else name + ("\0" if o["Z"] else ":")) + f"{count}\n")
        else:
            out.append(f"{count}\n")
    if o["l"] and matched_any:
        out.append(((C_FILE + name + C_END) if color else name) + ("\0" if o["Z"] else "\n"))
    if o["L"] and not matched_any:
        out.append(((C_FILE + name + C_END) if color else name) + ("\0" if o["Z"] else "\n"))
    if binary and matched_any and not (o["c"] or o["l"] or o["L"] or o["q"]):
        ctx.w("".join(out))
        out = []
        ctx.e(f"grep: {name}: binary file matches\n")
    if out:
        ctx.w("".join(out))
    if o["lb"]:
        yield from ctx.flush()
    return matched_any


@util("zgrep")
def u_zgrep(ctx):
    return (yield from grep_main(ctx, ctx.argv[1:], decompress=True))


# ---------------------------------------------------------------------------- ripgrep

RG_TYPES = {"py": ["*.py", "*.pyi"], "python": ["*.py", "*.pyi"], "js": ["*.js", "*.jsx", "*.mjs"],
            "ts": ["*.ts", "*.tsx"], "md": ["*.md", "*.markdown"], "markdown": ["*.md", "*.markdown"],
            "json": ["*.json"], "yaml": ["*.yaml", "*.yml"], "sh": ["*.sh", "*.bash", ".bashrc", "*.zsh"],
            "c": ["*.c", "*.h"], "cpp": ["*.cpp", "*.cc", "*.hpp", "*.h"], "rust": ["*.rs"], "go": ["*.go"],
            "txt": ["*.txt"], "csv": ["*.csv"], "toml": ["*.toml"], "html": ["*.html", "*.htm"], "ipynb": ["*.ipynb"]}


def _gitignore_rules(ctx, dirpath, node):
    rules = []
    gi = node.entries.get(".gitignore")
    if gi is not None and gi.type == "f":
        for line in gi.data.decode("utf-8", "replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rules.append(line)
    return rules


def _ignored(name, is_dir, rules):
    ignored = False
    for r in rules:
        neg = r.startswith("!")
        pat = r[1:] if neg else r
        dir_only = pat.endswith("/")
        pat = pat.rstrip("/").lstrip("/")
        if dir_only and not is_dir:
            continue
        if fnmatch(pat, name):
            ignored = not neg
    return ignored


@util("rg")
def u_rg(ctx):
    a = ctx.argv[1:]
    o = {"mode": "E", "pats": [], "i": False, "w": False, "x": False, "v": False, "c": False, "l": False,
         "L": False, "n": None, "H": None, "o": False, "q": False, "s": True, "m": None, "A": 0, "B": 0, "r": True,
         "include": [], "exclude": [], "exdir": [], "b": False, "color": False, "z": False, "Z": False,
         "label": "<stdin>", "T": False, "binary": "binary", "lb": False}
    types, globs = [], []
    hidden = no_ignore = False
    smart = False
    heading = None
    rest = []
    i = 0
    while i < len(a):
        x = a[i]
        nxt = a[i + 1] if i + 1 < len(a) else None
        if x in ("-t", "--type") and nxt:
            types.append(nxt)
            i += 2
            continue
        if x in ("-g", "--glob") and nxt:
            globs.append(nxt)
            i += 2
            continue
        if x in ("-e", "--regexp") and nxt:
            o["pats"].append(nxt)
            i += 2
            continue
        if x in ("-m", "--max-count") and nxt:
            o["m"] = int(nxt)
            i += 2
            continue
        if x in ("-A", "-B", "-C") and nxt:
            if x in ("-A", "-C"):
                o["A"] = int(nxt)
            if x in ("-B", "-C"):
                o["B"] = int(nxt)
            i += 2
            continue
        if x.startswith("--") or (x.startswith("-") and len(x) > 1):
            flags = [x] if x.startswith("--") else ["-" + c for c in x[1:]]
            for f in flags:
                if f in ("-i", "--ignore-case"):
                    o["i"] = True
                elif f in ("-S", "--smart-case"):
                    smart = True
                elif f in ("-w", "--word-regexp"):
                    o["w"] = True
                elif f in ("-x", "--line-regexp"):
                    o["x"] = True
                elif f in ("-v", "--invert-match"):
                    o["v"] = True
                elif f in ("-c", "--count"):
                    o["c"] = True
                elif f in ("-l", "--files-with-matches"):
                    o["l"] = True
                elif f == "--files-without-match":
                    o["L"] = True
                elif f in ("-n", "--line-number"):
                    o["n"] = True
                elif f in ("-N", "--no-line-number"):
                    o["n"] = False
                elif f in ("-o", "--only-matching"):
                    o["o"] = True
                elif f in ("-q", "--quiet"):
                    o["q"] = True
                elif f in ("-F", "--fixed-strings"):
                    o["mode"] = "F"
                elif f in ("-P", "--pcre2"):
                    o["mode"] = "P"
                elif f in ("-H", "--with-filename"):
                    o["H"] = True
                elif f in ("-I", "--no-filename"):
                    o["H"] = False
                elif f in ("--hidden", "-."):
                    hidden = True
                elif f in ("--no-ignore", "-u"):
                    no_ignore = True
                elif f == "--heading":
                    heading = True
                elif f == "--no-heading":
                    heading = False
                elif f == "--version":
                    ctx.w("ripgrep 14.1.0\n")
                    return 0
            i += 1
            continue
        rest.append(x)
        i += 1
    if not o["pats"]:
        if not rest:
            ctx.e("rg: ripgrep requires at least one pattern to execute a search\n")
            return 2
        o["pats"] = [rest[0]]
        rest = rest[1:]
    if smart and not any(c.isupper() for c in "".join(o["pats"])):
        o["i"] = True
    try:
        rx = build_regex(o)
    except RegexError as e:
        ctx.e(f"rg: regex parse error:\n    {o['pats'][0]}\nerror: {e}\n")
        return 2
    tty = ctx.stdout_is_tty()
    if o["n"] is None:
        o["n"] = tty
    if heading is None:
        heading = tty
    patterns = [g for t in types for g in RG_TYPES.get(t, [f"*.{t}"])]
    paths = rest or ["."]
    targets = []
    for p in paths:
        try:
            node = ctx.fs.lookup(p, ctx.cwd)
        except FSError as e:
            ctx.e(f"rg: {p}: {e.msg} (os error 2)\n")
            continue
        if node.type != "d":
            targets.append((p, node))
            continue

        def walk(path, n, rules):
            rules = rules + _gitignore_rules(ctx, path, n) if not no_ignore else rules
            for name in ctx.fs.listdir(n):
                child = n.entries[name]
                cp = name if path == "." else path.rstrip("/") + "/" + name
                if name.startswith(".") and not hidden:
                    continue
                if name == ".git":
                    continue
                if not no_ignore and _ignored(name, child.type == "d", rules):
                    continue
                if child.type == "d":
                    yield from walk(cp, child, rules)
                elif child.type == "f":
                    if patterns and not any(fnmatch(g, name) for g in patterns):
                        continue
                    if globs:
                        incl = [g for g in globs if not g.startswith("!")]
                        excl = [g[1:] for g in globs if g.startswith("!")]
                        if incl and not any(fnmatch(g, name) or fnmatch(g, cp) for g in incl):
                            continue
                        if any(fnmatch(g, name) or fnmatch(g, cp) for g in excl):
                            continue
                    yield cp, child
        targets += list(walk(p, node, []))
    show_name = o["H"] if o["H"] is not None else (len(targets) > 1 or any(ctx.fs.exists(p, ctx.cwd) and ctx.fs.lookup(p, ctx.cwd).type == "d" for p in paths))
    found = False
    first = True
    for name, node in targets:
        data = bytes(node.data)
        if b"\0" in data[:8192]:
            continue
        sub = dict(o)
        if heading and show_name and not (o["c"] or o["l"]):
            buf_ctx = _Capture(ctx)
            matched = yield from grep_stream(buf_ctx, ctx.open(name), name, rx, dict(sub, color=False), False)
            if matched:
                if not first:
                    ctx.w("\n")
                ctx.w(name + "\n" + buf_ctx.text())
                first = False
        else:
            matched = yield from grep_stream(ctx, ctx.open(name), name, rx, sub, show_name)
        found = found or matched
        if o["q"] and found:
            return 0
    return 0 if found else 1


class _Capture:
    """Stand-in Ctx that collects grep_stream output (for rg --heading)."""

    def __init__(self, ctx):
        self.__dict__.update({k: v for k, v in ctx.__dict__.items()})
        self._parts = []
        self._real = ctx

    def w(self, s):
        self._parts.append(s if isinstance(s, str) else s.decode())

    def e(self, s):
        self._real.e(s)

    def flush(self):
        return
        yield

    def text(self):
        return "".join(self._parts)

    def stdout_is_tty(self):
        return False
