"""Small coreutils and system-information commands."""

from __future__ import annotations

import re

from ..exec import UsageError, exec_program, find_in_path, gnu_getopt, util
from ..kernel import NS
from ..printf import escapes, format_printf
from ..timeutil import DateError, local_dt, parse_date, strftime
from ..vfs import FSError


@util("true")
def u_true(ctx):
    return 0
    yield


@util("false")
def u_false(ctx):
    return 1
    yield


@util("echo")
def u_echo(ctx):
    a = ctx.argv[1:]
    nl, esc = True, False
    i = 0
    while i < len(a) and re.fullmatch(r"-[neE]+", a[i]):
        for c in a[i][1:]:
            nl = nl and c != "n"
            esc = True if c == "e" else (False if c == "E" else esc)
        i += 1
    text = " ".join(a[i:])
    if esc:
        text, stop = escapes(text, echo=True)
        if stop:
            nl = False
    ctx.w(text + ("\n" if nl else ""))
    return 0
    yield


@util("printf")
def u_printf(ctx):
    a = ctx.argv[1:]
    if a and a[0] == "--":
        a = a[1:]
    if not a:
        ctx.err("missing operand")
        return ctx.usage(status=1)
    text, errors, _ = format_printf(a[0], a[1:], None, "printf")
    ctx.w(text)
    for e in errors:
        ctx.e(e + "\n")
    return 1 if errors else 0
    yield


@util("test", "[")
def u_test(ctx):
    from ..builtins import TestError, _test_expr

    class Fake:
        fs, proc, host = ctx.fs, ctx.proc, ctx.host
        opts = set()

        def lookup(self, n):
            return None

        def is_set(self, n):
            return n in ctx.env

        def arith(self, s):
            return int(s)
    a = ctx.argv[1:]
    if ctx.name == "[":
        if not a or a[-1] != "]":
            ctx.err("missing ']'")
            return 2
        a = a[:-1]
    try:
        return 0 if _test_expr(Fake(), a) else 1
    except TestError as e:
        ctx.err(str(e))
        return 2
    yield


def parse_duration(s: str) -> float:
    m = re.fullmatch(r"(\d+(?:\.\d*)?|\.\d+|inf|infinity)([smhd]?)", s)
    if not m:
        raise ValueError(s)
    if m.group(1).startswith("inf"):
        return float("inf")
    return float(m.group(1)) * {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}[m.group(2)]


@util("sleep")
def u_sleep(ctx):
    a = ctx.argv[1:]
    if not a:
        ctx.err("missing operand")
        return ctx.usage()
    total = 0.0
    for x in a:
        try:
            total += parse_duration(x)
        except ValueError:
            ctx.err(f"invalid time interval '{x}'")
            return ctx.usage()
    if total == float("inf"):
        while True:
            yield ("sleep", ctx.k.max_time)
    yield ("sleep", ctx.k.now + int(total * NS))
    return 0


@util("yes")
def u_yes(ctx):
    text = (" ".join(ctx.argv[1:]) if len(ctx.argv) > 1 else "y") + "\n"
    chunk = text * max(1, 8192 // len(text))
    while True:
        yield from ctx.out(chunk)


def _seq_num(s):
    if not re.fullmatch(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?", s):
        raise ValueError(s)
    return float(s) if any(c in s for c in ".eE") else int(s)


@util("seq")
def u_seq(ctx):
    a = ctx.argv[1:]
    sep, width, fmt = "\n", False, None
    nums = []
    i = 0
    while i < len(a):
        x = a[i]
        if x in ("-s", "--separator") and i + 1 < len(a):
            sep = a[i + 1]
            i += 2
            continue
        if x.startswith("-s") and len(x) > 2:
            sep = x[2:]
            i += 1
            continue
        if x in ("-w", "--equal-width"):
            width = True
            i += 1
            continue
        if x in ("-f", "--format") and i + 1 < len(a):
            fmt = a[i + 1]
            i += 2
            continue
        nums = a[i:]
        break
    if not nums:
        ctx.err("missing operand")
        return ctx.usage()
    if len(nums) > 3:
        ctx.err(f"extra operand '{nums[3]}'")
        return ctx.usage()
    try:
        vals = [_seq_num(x) for x in nums]
    except ValueError as e:
        ctx.err(f"invalid floating point argument: '{e.args[0]}'")
        return ctx.usage()
    first, step, last = (1, 1, vals[0]) if len(vals) == 1 else ((vals[0], 1, vals[1]) if len(vals) == 2 else vals)
    if step == 0:
        ctx.err("invalid Zero increment value: '0'")
        return ctx.usage()
    decimals = max((len(x.split(".")[1]) if "." in x and "e" not in x.lower() else 0) for x in nums)
    is_int = all(isinstance(v, int) for v in vals)
    out = []
    v = first
    n = 0
    texts = []
    while (step > 0 and v <= last + 1e-12) or (step < 0 and v >= last - 1e-12):
        if fmt:
            texts.append(format_printf(fmt, [repr(float(v))])[0])
        elif is_int:
            texts.append(str(v))
        else:
            texts.append(f"{v:.{decimals}f}")
        n += 1
        v = first + n * step
        if n > 2_000_000:
            break
    if width and texts:
        w = max(len(t) for t in texts)
        texts = [("-" + t[1:].rjust(w - 1, "0")) if t.startswith("-") else t.rjust(w, "0") for t in texts]
    for k, t in enumerate(texts):
        out.append(t)
        if len(out) >= 4096:
            ctx.w(sep.join(out) + (sep if k < len(texts) - 1 else ""))
            out = []
            yield from ctx.maybe_flush()
    if out:
        ctx.w(sep.join(out))
    if texts:
        ctx.w("\n")
    return 0


@util("env")
def u_env(ctx):
    a = ctx.argv[1:]
    env = dict(ctx.env)
    i = 0
    null = False
    chdir = None
    while i < len(a):
        x = a[i]
        if x in ("-i", "--ignore-environment", "-"):
            env = {}
            i += 1
        elif x in ("-u", "--unset") and i + 1 < len(a):
            env.pop(a[i + 1], None)
            i += 2
        elif x.startswith("-u") and len(x) > 2:
            env.pop(x[2:], None)
            i += 1
        elif x in ("-0", "--null"):
            null = True
            i += 1
        elif x in ("-C", "--chdir") and i + 1 < len(a):
            chdir = a[i + 1]
            i += 2
        elif x == "--":
            i += 1
            break
        elif x.startswith("-") and len(x) > 1 and "=" not in x:
            ctx.err(f"invalid option -- '{x[1]}'")
            return ctx.usage(status=125)
        else:
            break
    while i < len(a) and "=" in a[i] and not a[i].startswith("="):
        k, v = a[i].split("=", 1)
        env[k] = v
        i += 1
    if i >= len(a):
        end = "\0" if null else "\n"
        ctx.w("".join(f"{k}={v}{end}" for k, v in env.items()))
        return 0
    if chdir:
        ctx.proc.cwd = ctx.abspath(chdir)
    path = find_in_path(ctx.fs, a[i], env.get("PATH", ctx.env.get("PATH", "")), ctx.cwd)
    if path is None:
        ctx.err(f"'{a[i]}': No such file or directory")
        return 127
    yield from ctx.flush()
    ctx.proc.argv = a[i:]
    return (yield from exec_program(ctx.proc, path, a[i:], env, None))


@util("printenv")
def u_printenv(ctx):
    a = [x for x in ctx.argv[1:] if not x.startswith("-")]
    if not a:
        ctx.w("".join(f"{k}={v}\n" for k, v in ctx.env.items()))
        return 0
    status = 0
    for k in a:
        if k in ctx.env:
            ctx.w(ctx.env[k] + "\n")
        else:
            status = 1
    return status
    yield


DATE_HELP = "Usage: date [OPTION]... [+FORMAT]\n  or:  date [-u|--utc|--universal] [MMDDhhmm[[CC]YY][.ss]]\nDisplay date and time in the given FORMAT.\n"


@util("date", "gdate")
def u_date(ctx):
    try:
        opts, ops = gnu_getopt(ctx.argv[1:], "d:f:I::r:R uv:jf:s:", ["date=", "file=", "iso-8601=?", "reference=", "rfc-email",
                                                                   "rfc-3339=", "utc", "universal", "debug", "help",
                                                                   "set="], name="date")
    except UsageError as e:
        ctx.err(e.msg)
        return ctx.usage()
    tz = ctx.env.get("TZ")
    when = ctx.now
    fmt = None
    bsd = ctx.host.os == "macos" and ctx.name == "date"
    adjust = []
    for o, v in opts:
        if o in ("-u", "--utc", "--universal"):
            tz = "UTC"
        elif o in ("-d", "--date"):
            if bsd:
                ctx.err("illegal option -- d")
                ctx.e("usage: date [-jnRu] [-I[date|hours|minutes|seconds]] [-f input_fmt]\n            "
                      "[-r filename|seconds] [-v[+|-]val[y|m|w|d|H|M|S]]\n            [[[[mm]dd]HH]MM[[cc]yy][.SS] | new_date] [+output_fmt]\n")
                return 1
            try:
                when = parse_date(v, ctx.now, tz)
            except DateError:
                ctx.err(f"invalid date '{v}'")
                return 1
        elif o in ("-r", "--reference"):
            try:
                when = ctx.lookup(v).mtime
            except FSError as e:
                ctx.err(f"{v}: {e.msg}")
                return 1
        elif o in ("-R", "--rfc-email"):
            fmt = "%a, %d %b %Y %H:%M:%S %z"
        elif o in ("-I", "--iso-8601"):
            spec = (v or "date").lower()
            fmt = {"date": "%Y-%m-%d", "hours": "%Y-%m-%dT%H%:z", "minutes": "%Y-%m-%dT%H:%M%:z",
                   "seconds": "%Y-%m-%dT%H:%M:%S%:z", "ns": "%Y-%m-%dT%H:%M:%S,%N%:z"}.get(spec, "%Y-%m-%d")
        elif o == "--rfc-3339":
            fmt = {"date": "%Y-%m-%d", "seconds": "%Y-%m-%d %H:%M:%S%:z", "ns": "%Y-%m-%d %H:%M:%S.%N%:z"}.get(v, "%Y-%m-%d")
        elif o == "-v":
            adjust.append(v)
        elif o == "--help":
            ctx.w(DATE_HELP)
            return 0
    for x in ops:
        if x.startswith("+"):
            fmt = x[1:]
        else:
            ctx.err(f"invalid date '{x}'")
            return 1
    if adjust and not bsd:
        ctx.err(f"invalid option -- 'v'")
        return ctx.usage()
    for adj in adjust:   # BSD date -v+1d
        m = re.fullmatch(r"([+-]?)(\d+)([ymwdHMS])", adj)
        if not m:
            ctx.err(f"-v{adj}: Cannot apply date adjustment")
            return 1
        n = int(m.group(2)) * (-1 if m.group(1) == "-" else 1)
        unit = m.group(3)
        if unit in "ym":
            from ..timeutil import parse_date as pd
            when = pd(f"{n} {'year' if unit == 'y' else 'month'}", when, tz)
        else:
            when += n * {"w": 604800, "d": 86400, "H": 3600, "M": 60, "S": 1}[unit] * NS
    d = local_dt(when, tz)
    if fmt is None:
        fmt = "%a %b %e %H:%M:%S %Z %Y"
    text = strftime(fmt.replace("%:z", "%z" if False else "%:z"), d, when % NS)
    if "%:z" in fmt:
        z = strftime("%z", d)
        text = strftime(fmt.replace("%:z", z[:3] + ":" + z[3:]), d, when % NS)
    ctx.w(text + "\n")
    return 0
    yield


UNAME = {"s": "Linux", "o": "GNU/Linux", "m": "x86_64", "p": "x86_64", "i": "x86_64"}


@util("uname")
def u_uname(ctx):
    h = ctx.host
    a = ctx.argv[1:]
    flags = set()
    for x in a:
        if x == "--all":
            flags.add("a")
        elif x.startswith("-") and not x.startswith("--"):
            flags |= set(x[1:])
        elif x.startswith("--"):
            m = {"--kernel-name": "s", "--nodename": "n", "--kernel-release": "r", "--kernel-version": "v",
                 "--machine": "m", "--processor": "p", "--hardware-platform": "i", "--operating-system": "o"}
            if x not in m:
                ctx.err(f"unrecognized option '{x}'")
                return ctx.usage()
            flags.add(m[x])
        else:
            ctx.err(f"extra operand '{x}'")
            return ctx.usage()
    if h.os == "macos":
        vals = {"s": "Darwin", "n": h.name, "r": "24.6.0", "v": "Darwin Kernel Version 24.6.0", "m": "arm64",
                "p": "arm", "i": "arm64", "o": "Darwin"}
    else:
        vals = {"s": "Linux", "n": h.name, "r": h.kernel_release, "v": "#45-Ubuntu SMP PREEMPT_DYNAMIC Fri Aug 30 12:02:04 UTC 2024",
                "m": "x86_64", "p": "x86_64", "i": "x86_64", "o": "GNU/Linux"}
    if not flags:
        flags = {"s"}
    order = "snrvmpio"
    if "a" in flags:
        out = [vals[c] for c in "snrvm"] + ([vals["p"], vals["i"]] if h.os != "macos" else []) + [vals["o"]]
    else:
        out = [vals[c] for c in order if c in flags]
    ctx.w(" ".join(out) + "\n")
    return 0
    yield


@util("arch")
def u_arch(ctx):
    ctx.w("x86_64\n")
    return 0
    yield


@util("whoami")
def u_whoami(ctx):
    ctx.w(ctx.host.user_name(ctx.proc.uid) + "\n")
    return 0
    yield


@util("logname")
def u_logname(ctx):
    ctx.w(ctx.env.get("LOGNAME", "ada") + "\n")
    return 0
    yield


@util("id")
def u_id(ctx):
    h = ctx.host
    a = ctx.argv[1:]
    flags = "".join(x[1:] for x in a if x.startswith("-"))
    users = [x for x in a if not x.startswith("-")]
    uid = ctx.proc.uid
    if users:
        u = h.uid_of(users[0])
        if u is None or u not in h.users:
            ctx.err(f"'{users[0]}': no such user")
            return 1
        uid = u
    user = h.users.get(uid)
    gid = user.gid if user else ctx.proc.gid
    gids = sorted(h.gids_of(uid), key=lambda g: (g != gid, g)) if users or uid != ctx.proc.uid else \
        sorted(ctx.proc.gids | {gid}, key=lambda g: (g != gid, g))
    names = "n" in flags
    if "u" in flags:
        ctx.w((h.user_name(uid) if names else str(uid)) + "\n")
    elif "g" in flags and "G" not in flags:
        ctx.w((h.group_name(gid) if names else str(gid)) + "\n")
    elif "G" in flags:
        ctx.w(" ".join(h.group_name(g) if names else str(g) for g in gids) + "\n")
    else:
        groups = ",".join(f"{g}({h.group_name(g)})" for g in gids)
        ctx.w(f"uid={uid}({h.user_name(uid)}) gid={gid}({h.group_name(gid)}) groups={groups}\n")
    return 0
    yield


@util("groups")
def u_groups(ctx):
    h = ctx.host
    users = ctx.argv[1:]
    if not users:
        gids = sorted(ctx.proc.gids, key=lambda g: (g != ctx.proc.gid, g))
        ctx.w(" ".join(h.group_name(g) for g in gids) + "\n")
        return 0
    for u in users:
        uid = h.uid_of(u)
        if uid is None:
            ctx.err(f"'{u}': no such user")
            continue
        gid = h.users[uid].gid
        gids = sorted(h.gids_of(uid), key=lambda g: (g != gid, g))
        ctx.w(f"{u} : " + " ".join(h.group_name(g) for g in gids) + "\n")
    return 0
    yield


@util("hostname")
def u_hostname(ctx):
    a = ctx.argv[1:]
    h = ctx.host
    if "-I" in a or "--all-ip-addresses" in a:
        ctx.w(h.ip + " \n")
    elif "-i" in a:
        ctx.w("127.0.1.1\n")
    elif "-s" in a or "--short" in a:
        ctx.w(h.name.split(".")[0] + "\n")
    elif "-f" in a or "--fqdn" in a:
        ctx.w(h.name + "\n")
    elif a and not a[0].startswith("-"):
        if ctx.proc.uid != 0:
            ctx.err("you must be root to change the host name")
            return 1
        h.name = a[0]
    else:
        ctx.w(h.name + "\n")
    return 0
    yield


@util("nproc")
def u_nproc(ctx):
    n = ctx.host.cpus
    if "--all" not in ctx.argv:
        omp = ctx.env.get("OMP_NUM_THREADS", "")
        if omp.split(",")[0].isdigit():
            n = int(omp.split(",")[0])
    for i, x in enumerate(ctx.argv):
        if x.startswith("--ignore="):
            n = max(1, n - int(x.split("=")[1]))
    ctx.w(f"{n}\n")
    return 0
    yield


def _uptime_text(ctx):
    h = ctx.host
    up = (ctx.now - h.boot_ns) // NS
    days, rem = divmod(up, 86400)
    hours, rem = divmod(rem, 3600)
    mins = rem // 60
    d = local_dt(ctx.now, ctx.env.get("TZ"))
    users = 1
    dpart = f"{days} day{'s' if days != 1 else ''}, " if days else ""
    return (f" {d.hour:02d}:{d.minute:02d}:{d.second:02d} up {dpart}{hours:2d}:{mins:02d},  {users} user,  "
            f"load average: {h.load[0]:.2f}, {h.load[1]:.2f}, {h.load[2]:.2f}")


@util("uptime")
def u_uptime(ctx):
    h = ctx.host
    if "-p" in ctx.argv or "--pretty" in ctx.argv:
        up = (ctx.now - h.boot_ns) // NS
        w, rem = divmod(up, 604800)
        d, rem = divmod(rem, 86400)
        hr, rem = divmod(rem, 3600)
        m = rem // 60
        parts = []
        for v, nm in ((w, "week"), (d, "day"), (hr, "hour"), (m, "minute")):
            if v:
                parts.append(f"{v} {nm}{'s' if v != 1 else ''}")
        ctx.w("up " + ", ".join(parts) + "\n")
    elif "-s" in ctx.argv or "--since" in ctx.argv:
        ctx.w(strftime("%Y-%m-%d %H:%M:%S", local_dt(h.boot_ns, ctx.env.get("TZ"))) + "\n")
    else:
        ctx.w(_uptime_text(ctx) + "\n")
    return 0
    yield


def human(n: float, si=False, suffixes="BKMGTPE", width=0) -> str:
    """free -h / df -h / du -h style sizes."""
    base = 1000 if si else 1024
    if n < base:
        return f"{int(n)}{suffixes[0] if suffixes[0] != 'B' else ''}" if suffixes[0] != "B" else f"{int(n)}"
    for s in suffixes[1:]:
        n /= base
        if n < base or s == suffixes[-1]:
            if n < 10:
                import math
                v = math.ceil(n * 10) / 10
                if v >= 10:
                    return f"{math.ceil(n)}{s}"
                return f"{v:.1f}{s}"
            import math
            return f"{math.ceil(n)}{s}"
    return str(n)


@util("free")
def u_free(ctx):
    a = ctx.argv[1:]
    h = ctx.host
    total = h.mem_kib
    used = int(total * 0.22)
    free_ = int(total * 0.61)
    shared = 12 * 1024
    cache = total - used - free_
    avail = int(total * 0.78)
    swap = 2097148
    rows = [("Mem:", total, used, free_, shared, cache, avail), ("Swap:", swap, 0, swap)]

    def fmt(v):
        if "-h" in a or "--human" in a:
            return _free_human(v * 1024)
        if "-m" in a or "--mega" in a:
            return str(v // 1024)
        if "-g" in a or "--giga" in a:
            return str(v // (1024 * 1024))
        if "-b" in a or "--bytes" in a:
            return str(v * 1024)
        return str(v)
    lines = [" " * 8 + "".join(f"{h:>12}" for h in ("total", "used", "free", "shared", "buff/cache", "available"))]
    for r in rows:
        lines.append(f"{r[0]:<8}" + "".join(f"{fmt(v):>12}" for v in r[1:]))
    ctx.w("\n".join(lines) + "\n")
    return 0
    yield


def _free_human(b):
    for unit, size in (("Gi", 1 << 30), ("Mi", 1 << 20), ("Ki", 1 << 10)):
        if b >= size:
            v = b / size
            return f"{v:.1f}{unit}" if v < 10 else f"{round(v)}{unit}"
    return f"{b}B"


@util("lscpu")
def u_lscpu(ctx):
    h = ctx.host
    rows = [("Architecture:", "x86_64"), ("  CPU op-mode(s):", "32-bit, 64-bit"), ("  Address sizes:", "46 bits physical, 57 bits virtual"),
            ("  Byte Order:", "Little Endian"), ("CPU(s):", str(h.cpus)), ("  On-line CPU(s) list:", f"0-{h.cpus - 1}"),
            ("Vendor ID:", "GenuineIntel"), ("  Model name:", h.cpu_model), ("    CPU family:", "6"),
            ("    Model:", "106"), ("    Thread(s) per core:", "2"), ("    Core(s) per socket:", str(h.cpus // 2)),
            ("    Socket(s):", "1"), ("    CPU max MHz:", "3200.0000"), ("    CPU min MHz:", "800.0000"),
            ("Caches (sum of all):", ""), ("  L1d:", f"{h.cpus // 2 * 48} KiB ({h.cpus // 2} instances)"),
            ("  L2:", f"{h.cpus // 2 * 1.25:g} MiB ({h.cpus // 2} instances)"), ("  L3:", "48 MiB (1 instance)"),
            ("NUMA:", ""), ("  NUMA node(s):", "1"), ("  NUMA node0 CPU(s):", f"0-{h.cpus - 1}")]
    ctx.w("".join(f"{k:<38}{v}".rstrip() + "\n" for k, v in rows))
    return 0
    yield


@util("clear")
def u_clear(ctx):
    ctx.w("\x1b[H\x1b[2J\x1b[3J")
    return 0
    yield


@util("reset")
def u_reset(ctx):
    ctx.w("\x1bc")
    return 0
    yield


@util("tput")
def u_tput(ctx):
    a = ctx.argv[1:]
    if not a:
        ctx.err("usage: tput [-V] [-S] [-T term] capname")
        return 2
    cap = a[0]
    m = {"clear": "\x1b[H\x1b[2J", "cols": "80\n", "lines": "24\n", "bold": "\x1b[1m", "sgr0": "\x1b(B\x1b[m",
         "smul": "\x1b[4m", "rmul": "\x1b[24m", "rev": "\x1b[7m", "colors": "256\n", "cup": "\x1b[H"}
    if cap in ("setaf", "setab") and len(a) > 1:
        ctx.w(f"\x1b[{3 if cap == 'setaf' else 4}{a[1]}m")
        return 0
    if cap not in m:
        ctx.err(f"unknown terminfo capability '{cap}'")
        return 4
    ctx.w(m[cap])
    return 0
    yield


@util("tty")
def u_tty(ctx):
    of = ctx.proc.fds.get(0)
    if of is not None and of.kind == "tty":
        ctx.w(f"/dev/{of.tty.name}\n")
        return 0
    ctx.w("not a tty\n")
    return 1
    yield


@util("which")
def u_which(ctx):
    a = ctx.argv[1:]
    all_ = False
    names = []
    for x in a:
        if x == "-a":
            all_ = True
        elif x == "-s":
            pass
        else:
            names.append(x)
    status = 0
    path = ctx.env.get("PATH", "")
    for n in names:
        found = []
        if "/" in n:
            try:
                node = ctx.lookup(n)
                if node.type == "f" and node.mode & 0o111:
                    found.append(n)
            except FSError:
                pass
        else:
            for d in path.split(":"):
                cand = (d or ".").rstrip("/") + "/" + n
                try:
                    node = ctx.fs.lookup(cand, ctx.cwd)
                except FSError:
                    continue
                if node.type == "f" and node.mode & 0o111 and cand not in found:
                    found.append(cand)
                    if not all_:
                        break
        if not found:
            status = 1
        for f in found:
            ctx.w(f + "\n")
    return status
    yield


@util("sudo")
def u_sudo(ctx):
    a = ctx.argv[1:]
    user = "root"
    env = dict(ctx.env)
    keep_env = False
    login = False
    while a and a[0].startswith("-"):
        o = a.pop(0)
        if o == "-u" and a:
            user = a.pop(0)
        elif o in ("-E", "--preserve-env"):
            keep_env = True
        elif o in ("-i", "--login"):
            login = True
        elif o in ("-s", "-H", "-n", "-k", "-S"):
            pass
        elif o == "-v":
            return 0
        elif o == "--":
            break
        else:
            ctx.err(f"invalid option -- '{o[1:2]}'")
            ctx.e("usage: sudo [-u user] command\n")
            return 1
    h = ctx.host
    me = h.user_name(ctx.proc.uid)
    sudoers = h.groups.get(h.gid_of("sudo") or 27, ("sudo", set()))[1]
    if ctx.proc.uid != 0 and me not in sudoers:
        ctx.e(f"{me} is not in the sudoers file.\n")
        return 1
    uid = h.uid_of(user)
    if uid is None:
        ctx.err(f"unknown user {user}")
        return 1
    if not a:
        if login:
            a = ["bash", "-i"]
        else:
            ctx.e("usage: sudo -h | -K | -k | -V\nusage: sudo [-u user] command\n")
            return 1
    u = h.users[uid]
    if not keep_env:
        env = {k: v for k, v in env.items() if k in ("TERM", "PATH", "LANG", "DISPLAY", "COLORTERM")}
        env.update({"HOME": u.home, "USER": u.name, "LOGNAME": u.name, "SHELL": u.shell, "SUDO_USER": me,
                    "SUDO_UID": str(ctx.proc.uid), "SUDO_COMMAND": " ".join(a),
                    "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/snap/bin"})
    path = find_in_path(ctx.fs, a[0], env.get("PATH", ""), ctx.cwd)
    if path is None:
        ctx.e(f"sudo: {a[0]}: command not found\n")
        return 1
    ctx.proc.uid = uid
    ctx.proc.gid = u.gid
    ctx.proc.gids = h.gids_of(uid)
    ctx.proc.argv = a
    return (yield from exec_program(ctx.proc, path, a, env, None))


@util("expr")
def u_expr(ctx):
    a = ctx.argv[1:]
    try:
        v = _expr(a)
    except ZeroDivisionError:
        ctx.err("division by zero")
        return 2
    except (ValueError, IndexError):
        ctx.err("syntax error")
        return 2
    ctx.w(f"{v}\n")
    return 0 if v not in ("", 0, "0") else 1
    yield


def _expr(a):
    # precedence: | & (= > >= < <= !=) (+ -) (* / %) :
    def num(x):
        return int(x)

    def parse(toks, ops_list):
        if not ops_list:
            if toks and toks[0] == "(":
                depth = 0
                for i, t in enumerate(toks):
                    depth += t == "("
                    depth -= t == ")"
                    if depth == 0:
                        return parse(toks[1:i], LEVELS)
            if len(toks) == 1:
                return toks[0]
            if len(toks) == 2 and toks[0] == "length":
                return len(toks[1])
            raise ValueError
        ops = ops_list[0]
        depth = 0
        for i in range(len(toks) - 1, -1, -1):
            t = toks[i]
            depth += t == ")"
            depth -= t == "("
            if depth == 0 and t in ops and 0 < i < len(toks) - 1:
                left = parse(toks[:i], ops_list)
                right = parse(toks[i + 1:], ops_list[1:])
                return apply(t, left, right)
        return parse(toks, ops_list[1:])

    def apply(op, x, y):
        if op == "|":
            return x if x not in ("", 0, "0") else (y if y not in ("", 0, "0") else 0)
        if op == "&":
            return x if x not in ("", 0, "0") and y not in ("", 0, "0") else 0
        if op in ("=", "!=", "<", "<=", ">", ">="):
            try:
                xv, yv = int(x), int(y)
            except ValueError:
                xv, yv = str(x), str(y)
            return int({"=": xv == yv, "!=": xv != yv, "<": xv < yv, "<=": xv <= yv, ">": xv > yv, ">=": xv >= yv}[op])
        if op in "+-*/%":
            xv, yv = num(x), num(y)
            if op == "+":
                return xv + yv
            if op == "-":
                return xv - yv
            if op == "*":
                return xv * yv
            q = abs(xv) // abs(yv)
            if (xv < 0) != (yv < 0):
                q = -q
            return q if op == "/" else xv - yv * q
        if op == ":":
            from ..regex import compile_re
            m = compile_re(str(y), "bre").match(str(x))
            if m is None:
                return "" if compile_re(str(y), "bre").groups else 0
            return m.group(1) if m.re.groups else m.end()
        raise ValueError
    LEVELS = [["|"], ["&"], ["=", "!=", "<", "<=", ">", ">="], ["+", "-"], ["*", "/", "%"], [":"]]
    return parse(a, LEVELS)
