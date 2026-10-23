"""man whatis apropos tldr info · crontab · shellcheck · apt · brew."""

from __future__ import annotations

import re

from ..exec import util
from ..vfs import FSError
from .manpages import PAGES, man_text, tldr_text, whatis_line

BUILTIN_NAMES = {"cd", "help", "type", "alias", "export", "source", "history", "jobs", "fg", "bg", "read", "set", "shopt",
                 "trap", "declare", "local", "eval", "exec", "exit", "unset", "getopts", "printf", "echo", "test", "pwd"}


@util("man")
def u_man(ctx):
    a = [x for x in ctx.argv[1:]]
    if not a:
        ctx.e("What manual page do you want?\nFor example, try 'man man'.\n")
        return 1
    if a[0] in ("-k", "--apropos"):
        return (yield from _apropos(ctx, a[1:]))
    if a[0] in ("-f", "--whatis"):
        return (yield from _whatis(ctx, a[1:]))
    names = [x for x in a if not x.startswith("-") and not x.isdigit()]
    status = 0
    for n in names:
        text = man_text(n)
        if text is None:
            if n in BUILTIN_NAMES:
                text = man_text("bash")
            else:
                ctx.e(f"No manual entry for {n}\n")
                status = 16
                continue
        ctx.w(text)
    return status
    yield


def _whatis(ctx, names):
    status = 0
    for n in names:
        line = whatis_line(n)
        if line is None:
            ctx.w(f"{n}: nothing appropriate.\n")
            status = 16
        else:
            ctx.w(line + "\n")
    return status
    yield


def _apropos(ctx, words):
    if not words:
        ctx.w("apropos what?\n")
        return 1
    hits = []
    for name, page in PAGES.items():
        text = name + " " + page[1]
        if all(re.search(w, text, re.I) for w in words) or any(re.search(" ".join(words), text, re.I) for _ in [0]):
            hits.append(name)
    if not hits:
        ctx.w(f"{' '.join(words)}: nothing appropriate.\n")
        return 16
    for n in sorted(hits):
        ctx.w(whatis_line(n) + "\n")
    return 0
    yield


@util("whatis")
def u_whatis(ctx):
    return (yield from _whatis(ctx, [x for x in ctx.argv[1:] if not x.startswith("-")]))


@util("apropos")
def u_apropos(ctx):
    return (yield from _apropos(ctx, [x for x in ctx.argv[1:] if not x.startswith("-")]))


@util("info")
def u_info(ctx):
    names = [x for x in ctx.argv[1:] if not x.startswith("-")]
    text = man_text(names[0]) if names else None
    if text is None:
        ctx.e(f"info: No menu item '{names[0] if names else ''}' in node '(dir)Top'\n")
        return 1
    ctx.w(text)
    return 0
    yield


@util("tldr")
def u_tldr(ctx):
    names = [x for x in ctx.argv[1:] if not x.startswith("-")]
    if not names:
        ctx.w("Usage: tldr command\n")
        return 1
    page = tldr_text(names[0])
    if page is None:
        ctx.e(f"Page not found. Feel free to send a pull request to: https://github.com/tldr-pages/tldr\n")
        return 1
    ctx.w(page)
    return 0
    yield


# ---------------------------------------------------------------------------- crontab

CRON_FIELD = re.compile(r"(\*|\d+(-\d+)?)(/\d+)?(,(\*|\d+(-\d+)?)(/\d+)?)*\Z")
NAMES = re.compile(r"(?i)(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|sun|mon|tue|wed|thu|fri|sat)(-\w+)?(,\w+)*\Z")


def validate_crontab(text):
    for n, raw in enumerate(text.split("\n"), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if re.match(r"[A-Za-z_][A-Za-z0-9_]*\s*=", line):
            continue
        if line.startswith("@"):
            if line.split()[0] not in ("@reboot", "@yearly", "@annually", "@monthly", "@weekly", "@daily", "@midnight", "@hourly"):
                return n, "bad time specifier"
            if len(line.split(None, 1)) < 2:
                return n, "bad command"
            continue
        parts = line.split(None, 5)
        if len(parts) < 6:
            return n, "bad command" if len(parts) == 5 else "bad minute" if len(parts) < 1 else ["bad minute", "bad hour", "bad day-of-month", "bad month", "bad day-of-week", "bad command"][min(len(parts), 5)]
        labels = ["minute", "hour", "day-of-month", "month", "day-of-week"]
        limits = [(0, 59), (0, 23), (1, 31), (1, 12), (0, 7)]
        for k in range(5):
            f = parts[k]
            if not (CRON_FIELD.match(f) or (k >= 3 and NAMES.match(f))):
                return n, f"bad {labels[k]}"
            for num in re.findall(r"\d+", re.sub(r"/\d+", "", f)):
                lo, hi = limits[k]
                if not lo <= int(num) <= hi:
                    return n, f"bad {labels[k]}"
    return None


@util("crontab")
def u_crontab(ctx):
    a = ctx.argv[1:]
    user = ctx.host.user_name(ctx.uid)
    if "-u" in a:
        i = a.index("-u")
        user = a[i + 1]
        del a[i:i + 2]
        if ctx.uid != 0 and user != ctx.host.user_name(ctx.uid):
            ctx.e("must be privileged to use -u\n")
            return 1
    path = f"/var/spool/cron/crontabs/{user}"
    fs = ctx.fs

    def current():
        try:
            return bytes(fs.lookup(path).data).decode()
        except FSError:
            return None

    def install(text):
        err = validate_crontab(text)
        if err:
            n, msg = err
            ctx.e(f'"-":{n - 1 if False else n - 1}: {msg}\nerrors in crontab file, can\'t install.\n')
            return 1
        node = fs.write_file(path, text, "/", 0o600, ctx.host.uid_of(user) or 0, 101)
        node.mode = 0o600
        return 0
    if not a or a == ["-"]:
        if not a and ctx.stdin_is_tty():
            pass
        data = yield from ctx.read_fd(0)
        return install(data.decode("utf-8", "replace"))
    flag = a[0]
    if flag == "-l":
        cur = current()
        if cur is None:
            ctx.e(f"no crontab for {user}\n")
            return 1
        ctx.w(cur)
        return 0
    if flag == "-r":
        if current() is None:
            ctx.e(f"no crontab for {user}\n")
            return 1
        fs.unlink(path)
        return 0
    if flag == "-e":
        cur = current()
        if cur is None:
            ctx.e(f"no crontab for {user} - using an empty one\n")
            cur = ""
        tmp = f"/tmp/crontab.{ctx.world.rng.randrange(10**6):06d}/crontab"
        fs.write_file(tmp, cur, "/", 0o600, ctx.uid, ctx.proc.gid)
        editor = ctx.env.get("VISUAL") or ctx.env.get("EDITOR") or "vi"
        st = yield from ctx.run_shell(f"{editor} {tmp}")
        if st != 0:
            ctx.e("crontab: editor exited with an error; crontab not changed\n")
            return 1
        new = bytes(fs.lookup(tmp).data).decode()
        if new == cur:
            ctx.e("crontab: no changes made to crontab\n")
            return 0
        rc = install(new)
        if rc == 0:
            ctx.e("crontab: installing new crontab\n")
        return rc
    if flag.startswith("-"):
        ctx.e(f"crontab: invalid option -- '{flag[1:2]}'\ncrontab: usage error: unrecognized option\n"
              "usage:\tcrontab [-u user] file\n\tcrontab [ -u user ] [ -i ] { -e | -l | -r }\n")
        return 1
    try:
        text = yield from ctx.text(flag)
    except FSError as e:
        ctx.e(f"{flag}: {e.msg}\n")
        return 1
    return install(text)


# ---------------------------------------------------------------------------- shellcheck

SC = {
    "SC2148": ("error", "Tips depend on target shell and yours is unknown. Add a shebang or a 'shell' directive."),
    "SC2086": ("info", "Double quote to prevent globbing and word splitting."),
    "SC2046": ("warning", "Quote this to prevent word splitting."),
    "SC2006": ("style", "Use $(...) notation instead of legacy backticks `...`."),
    "SC2164": ("warning", "Use 'cd ... || exit' or 'cd ... || return' in case cd fails."),
    "SC2162": ("info", "read without -r will mangle backslashes."),
    "SC2045": ("error", "Iterating over ls output is fragile. Use globs."),
    "SC2068": ("error", "Double quote array expansions to avoid re-splitting elements."),
    "SC2155": ("warning", "Declare and assign separately to avoid masking return values."),
    "SC2002": ("style", "Useless cat. Consider 'cmd < file | ..' or 'cmd file | ..' instead."),
    "SC2009": ("info", "Consider using pgrep instead of grepping ps output."),
    "SC2034": ("warning", "{name} appears unused. Verify use (or export if used externally)."),
    "SC2010": ("warning", "Don't use ls | grep. Use a glob or a for loop with a condition to allow non-alphanumeric filenames."),
}


def _scan_line(line):
    """Yield (col, kind, text) for $-expansions with quoting context: kind = 'q' (inside "") or 'u' (unquoted)."""
    i, n = 0, len(line)
    in_dq = False
    while i < n:
        c = line[i]
        if c == "\\" and i + 1 < n:
            i += 2
            continue
        if c == "'" and not in_dq:
            j = line.find("'", i + 1)
            i = n if j < 0 else j + 1
            continue
        if c == '"':
            in_dq = not in_dq
            i += 1
            continue
        if c == "#" and not in_dq and (i == 0 or line[i - 1] in " \t;"):
            return
        if c == "`":
            j = line.find("`", i + 1)
            yield (i, "bt", line[i:j + 1 if j > 0 else n])
            i = n if j < 0 else j + 1
            continue
        if c == "$" and i + 1 < n:
            if line[i + 1] == "(" and line[i + 2:i + 3] != "(":
                depth, j = 0, i + 1
                while j < n:
                    if line[j] == "(":
                        depth += 1
                    elif line[j] == ")":
                        depth -= 1
                        if depth == 0:
                            break
                    j += 1
                yield (i, "cmd" + ("q" if in_dq else "u"), line[i:j + 1])
                i = j + 1
                continue
            if line[i + 1] == "{":
                j = line.find("}", i)
                yield (i, "q" if in_dq else "u", line[i:j + 1 if j > 0 else n])
                i = n if j < 0 else j + 1
                continue
            m = re.match(r"\$([A-Za-z_][A-Za-z0-9_]*|[0-9@*#?$!-])", line[i:])
            if m:
                yield (i, "q" if in_dq else "u", m.group(0))
                i += m.end()
                continue
        i += 1


def shellcheck_findings(src):
    findings = []
    lines = src.split("\n")
    if not src.startswith("#!"):
        findings.append((1, 0, 1, "SC2148", None))
    assigned = {}
    used = set()
    for ln, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        for m in re.finditer(r"(?:^|[\s;&|(])(?:local\s+|export\s+|declare\s+(?:-\w+\s+)*|readonly\s+)?([A-Za-z_][A-Za-z0-9_]*)=", line):
            assigned.setdefault(m.group(1), (ln, m.start(1)))
        for _, _, text in _scan_line(line):
            for name in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text.lstrip("$").lstrip("{").split("}")[0][:40]):
                used.add(name)
                break
            for name in re.findall(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)", text):
                used.add(name)
        in_cond = "[[" in line or "((" in line
        for col, kind, text in _scan_line(line):
            before = line[:col]
            if re.search(r"(^|[\s;])\w+(\[[^\]]*\])?=$", before):   # assignment value
                continue
            if re.search(r"\bcase\s+$", before) or (in_cond and not re.search(r"\]\]|\)\)", line[:col])):
                continue
            if kind == "bt":
                findings.append((ln, col, col + len(text), "SC2006", line[:col] + "$(" + text[1:-1] + ")" + line[col + len(text):]))
            elif kind == "cmdu":
                if re.search(r"\bfor\s+\w+\s+in\s+$", before) and text.startswith("$(ls"):
                    findings.append((ln, col, col + len(text), "SC2045", None))
                elif not re.search(r"\b(local|export|declare|readonly)\s+\w+=$", before):
                    findings.append((ln, col, col + len(text), "SC2046", line[:col] + '"' + text + '"' + line[col + len(text):]))
            elif kind == "u":
                if re.search(r"\[[@]\]\}?$", text) or text in ("$@",):
                    findings.append((ln, col, col + len(text), "SC2068", None))
                elif text not in ("$?", "$#", "$$", "$!", "$-") and not re.fullmatch(r"\$\{#\w+(\[@\])?\}", text):
                    findings.append((ln, col, col + len(text), "SC2086", line[:col] + '"' + text + '"' + line[col + len(text):]))
        if re.search(r"(^|[;&|]\s*|\bthen\s+|\bdo\s+)cd\s+[^;&|]+\s*$", line) or re.search(r"(^|[;]\s*)cd\s+\S+\s*;", line):
            if "||" not in line[line.find("cd"):]:
                c = line.find("cd")
                findings.append((ln, c, c + len(line[c:].split(";")[0].rstrip()), "SC2164", None))
        m = re.search(r"\bread\s+(?!.*-\w*r)(-\w+\s+)*\w", line)
        if m and not re.search(r"\bread\s+-\w*r", line):
            c = m.start()
            findings.append((ln, c, c + 4, "SC2162", None))
        m = re.search(r"\b(local|export|declare|readonly)\s+(\w+)=\$\(", line)
        if m:
            findings.append((ln, m.start(2), m.start(2) + len(m.group(2)), "SC2155", None))
        m = re.search(r"\bcat\s+([^\s|<>;&]+)\s*\|", line)
        if m:
            findings.append((ln, m.start(1), m.end(1), "SC2002", None))
        m = re.search(r"\bps\s+[^|]*\|\s*grep\b", line)
        if m:
            c = line.find("ps", m.start())
            findings.append((ln, c, m.end(), "SC2009", None))
        m = re.search(r"\bls\b[^|]*\|\s*grep\b", line)
        if m:
            findings.append((ln, m.start(), m.end(), "SC2010", None))
    for name, (ln, col) in assigned.items():
        if name not in used and not name.isupper():
            findings.append((ln, col, col + len(name), "SC2034", name))
    findings.sort(key=lambda f: (f[0], f[1]))
    return findings


@util("shellcheck")
def u_shellcheck(ctx):
    files = [x for x in ctx.argv[1:] if not x.startswith("-")]
    fmt = "tty"
    for x in ctx.argv[1:]:
        if x.startswith("--format=") or x.startswith("-f"):
            fmt = x.split("=", 1)[1] if "=" in x else "gcc"
    if not files:
        ctx.e("No files specified.\n\nUsage: shellcheck [OPTIONS...] FILES...\n")
        return 3
    status = 0
    for f in files:
        try:
            src = yield from ctx.text(f)
        except FSError as e:
            ctx.e(f"{f}: {f}: openBinaryFile: does not exist (No such file or directory)\n")
            status = 2
            continue
        fnd = shellcheck_findings(src)
        if not fnd:
            continue
        status = max(status, 1)
        lines = src.split("\n")
        if fmt == "gcc":
            for ln, c0, c1, code, extra in fnd:
                level, msg = SC[code]
                ctx.w(f"{f}:{ln}:{c0 + 1}: {level}: {msg.format(name=extra or '')} [{code}]\n")
            continue
        out = []
        groups = {}
        for item in fnd:
            groups.setdefault(item[0], []).append(item)
        for ln, items in groups.items():
            out.append(f"\nIn {f} line {ln}:")
            out.append(lines[ln - 1])
            fix = None
            for _, c0, c1, code, extra in items:
                level, msg = SC[code]
                width = max(c1 - c0, 1)
                mark = "^--" if width <= 1 else "^" + "-" * (width - 2) + "^"
                out.append(" " * c0 + f"{mark} {code} ({level}): {msg.format(name=extra or '')}")
                if code in ("SC2086", "SC2046", "SC2006") and extra:
                    fix = extra if fix is None else fix
            if fix and len(items) == 1:
                out.append("")
                out.append("Did you mean:")
                out.append(fix)
        out.append("")
        out.append("For more information:")
        seen = []
        for _, _, _, code, extra in fnd:
            if code in seen:
                continue
            seen.append(code)
            msg = SC[code][1].format(name=extra or "")
            short = msg if len(msg) <= 40 else msg[:msg.rfind(" ", 0, 40)] + " ..."
            out.append(f"  https://www.shellcheck.net/wiki/{code} -- {short}")
        ctx.w("\n".join(out) + "\n")
    return status
    yield


# ---------------------------------------------------------------------------- apt / brew

APT_PACKAGES = {"tree": "2.1.1-2ubuntu3", "htop": "3.3.0-4build1", "jq": "1.7.1-3build1", "ripgrep": "14.1.0-1",
                "tldr": "1.6.1-1", "tmux": "3.4-1build1", "shellcheck": "0.9.0-1", "nvtop": "3.0.2-1", "bc": "1.07.1-3build1",
                "zip": "3.0-13build1", "unzip": "6.0-28ubuntu4", "curl": "8.5.0-2ubuntu10", "wget": "1.21.4-1ubuntu4",
                "rsync": "3.2.7-1ubuntu1", "git": "1:2.43.0-1ubuntu7", "python3-venv": "3.12.3-0ubuntu2",
                "python3-pip": "24.0+dfsg-1ubuntu1", "build-essential": "12.10ubuntu1", "vim": "2:9.1.0016-1ubuntu7"}


@util("apt", "apt-get")
def u_apt(ctx):
    a = ctx.argv[1:]
    if not a:
        ctx.w("apt 2.7.14 (amd64)\nUsage: apt [options] command\n")
        return 1
    cmd = a[0]
    pkgs = [x for x in a[1:] if not x.startswith("-")]
    if cmd in ("install", "remove", "update", "upgrade", "purge") and ctx.uid != 0:
        ctx.e("E: Could not open lock file /var/lib/dpkg/lock-frontend - open (13: Permission denied)\n"
              "E: Unable to acquire the dpkg frontend lock (/var/lib/dpkg/lock-frontend), are you root?\n")
        return 100
    if cmd == "update":
        ctx.w("Hit:1 http://archive.ubuntu.com/ubuntu noble InRelease\nGet:2 http://security.ubuntu.com/ubuntu noble-security InRelease [126 kB]\n"
              "Fetched 126 kB in 1s (126 kB/s)\nReading package lists... Done\nBuilding dependency tree... Done\nReading state information... Done\n"
              "All packages are up to date.\n")
        return 0
    if cmd == "install":
        ctx.w("Reading package lists... Done\nBuilding dependency tree... Done\nReading state information... Done\n")
        missing = [p for p in pkgs if p not in APT_PACKAGES]
        if missing:
            ctx.e(f"E: Unable to locate package {missing[0]}\n")
            return 100
        new = [p for p in pkgs if not ctx.fs.exists(f"/var/lib/bashlite/apt/{p}")]
        if not new:
            for p in pkgs:
                ctx.w(f"{p} is already the newest version ({APT_PACKAGES[p]}).\n")
            ctx.w("0 upgraded, 0 newly installed, 0 to remove and 0 not upgraded.\n")
            return 0
        ctx.w("The following NEW packages will be installed:\n  " + " ".join(new) + "\n")
        ctx.w(f"0 upgraded, {len(new)} newly installed, 0 to remove and 0 not upgraded.\n")
        for p in new:
            ctx.w(f"Selecting previously unselected package {p}.\nPreparing to unpack .../{p}_{APT_PACKAGES[p]}_amd64.deb ...\n"
                  f"Unpacking {p} ({APT_PACKAGES[p]}) ...\nSetting up {p} ({APT_PACKAGES[p]}) ...\n")
            ctx.fs.write_file(f"/var/lib/bashlite/apt/{p}", APT_PACKAGES[p] + "\n", "/", 0o644, 0, 0)
        return 0
    if cmd in ("list",):
        for p in sorted(APT_PACKAGES):
            if "--installed" in a and not ctx.fs.exists(f"/var/lib/bashlite/apt/{p}"):
                continue
            ctx.w(f"{p}/noble {APT_PACKAGES[p]} amd64\n")
        return 0
    if cmd == "search":
        for p in sorted(APT_PACKAGES):
            if any(q in p for q in pkgs):
                ctx.w(f"{p}/noble {APT_PACKAGES[p]} amd64\n\n")
        return 0
    ctx.w("Reading package lists... Done\n")
    return 0
    yield


BREW = {"coreutils": "9.5", "gnu-sed": "4.9", "grep": "3.11", "findutils": "4.10.0", "bash": "5.2.37", "jq": "1.7.1",
        "tree": "2.1.3", "htop": "3.3.0", "tmux": "3.5a", "ripgrep": "14.1.1", "tldr": "1.6.1", "wget": "1.24.5", "shellcheck": "0.10.0"}
GNU_PROGS = {"coreutils": ["ls", "cat", "date", "sort", "head", "tail", "cut", "wc", "stat", "du", "cp", "mv", "rm", "readlink", "realpath", "sha256sum", "md5sum"],
             "gnu-sed": ["sed"], "grep": ["grep", "egrep", "fgrep"], "findutils": ["find", "xargs"]}


@util("brew")
def u_brew(ctx):
    a = ctx.argv[1:]
    if not a:
        ctx.w("Example usage:\n  brew search TEXT|/REGEX/\n  brew info [FORMULA|CASK...]\n  brew install FORMULA|CASK...\n")
        return 1
    cmd, pkgs = a[0], [x for x in a[1:] if not x.startswith("-")]
    if cmd == "install":
        for p in pkgs:
            if p not in BREW:
                ctx.e(f"Warning: No available formula with the name \"{p}\".\n")
                return 1
            ctx.w(f"==> Downloading https://ghcr.io/v2/homebrew/core/{p}/manifests/{BREW[p]}\n==> Pouring {p}--{BREW[p]}.arm64_sequoia.bottle.tar.gz\n"
                  f"🍺  /opt/homebrew/Cellar/{p}/{BREW[p]}: 1 file, 1MB\n")
            from ..exec import stub_bytes
            for prog in GNU_PROGS.get(p, [p]):
                name = ("g" + prog) if p in GNU_PROGS else prog
                node = ctx.fs.write_file(f"/opt/homebrew/bin/{name}", stub_bytes(name), "/", 0o755, ctx.uid, ctx.proc.gid)
                node.mode = 0o755
            ctx.fs.mkdir(f"/opt/homebrew/Cellar/{p}/{BREW[p]}", "/", 0o755, ctx.uid, ctx.proc.gid, parents=True)
        return 0
    if cmd == "list":
        for p in sorted(BREW):
            if ctx.fs.exists(f"/opt/homebrew/Cellar/{p}"):
                ctx.w(p + "\n")
        return 0
    if cmd == "--prefix":
        ctx.w("/opt/homebrew\n")
        return 0
    return 0
    yield
