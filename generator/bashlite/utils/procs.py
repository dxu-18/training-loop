"""Processes: ps pgrep pkill kill killall pidof top htop nohup timeout nice renice watch lsof."""

from __future__ import annotations

import re

from ..exec import exec_program, find_in_path, util
from ..kernel import NS, SIG_DESC, SIGHUP, SIGKILL, SIGNALS, SIGNUM, SIGTERM
from ..timeutil import local_dt, strftime
from .basic import parse_duration


def proc_list(host):
    return sorted((p for p in host.procs.values() if p.state != "Z"), key=lambda p: p.pid)


def stat_of(p, tty) -> str:
    s = {"R": "R", "S": "S", "T": "T", "Z": "Z", "I": "S"}.get(p.state, "S")
    if p.stat_extra:
        return s + p.stat_extra
    if p.nice > 0:
        s += "N"
    elif p.nice < 0:
        s += "<"
    if p.shell is not None and p.shell.toplevel_pid == p.pid and p.tty != "?":
        s += "s"
    if tty is not None and p.tty != "?" and p.pgid == tty.fg_pgid and p.state != "T":
        s += "+"
    return s


def start_text(p, now):
    d = local_dt(p.start, None)
    n = local_dt(now, None)
    if (now - p.start) < 86400 * NS and d.day == n.day:
        return strftime("%H:%M", d)
    if d.year == n.year:
        return strftime("%b%d", d)
    return str(d.year)


def cpu_time(p, now, long=False):
    secs = int(p.cputime)
    if long:
        h, rem = divmod(secs, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
    m, s = divmod(secs, 60)
    return f"{m}:{s:02d}"


def etime(p, now):
    secs = max((now - p.start) // NS, 0)
    d, rem = divmod(secs, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    if d:
        return f"{d}-{h:02d}:{m:02d}:{s:02d}"
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def cmdline(p):
    if not p.argv:
        return f"[{p.comm}]"
    return " ".join(p.argv)


def comm_of(p):
    a = p.argv[0] if p.argv else "?"
    if a.startswith("-"):
        a = a[1:]
    name = a.rsplit("/", 1)[-1]
    if p.extra_comm if hasattr(p, "extra_comm") else False:
        return p.extra_comm
    return name[:15]


FIELDS = {
    "pid": ("PID", lambda p, c: str(p.pid), True), "ppid": ("PPID", lambda p, c: str(p.ppid), True),
    "pgid": ("PGID", lambda p, c: str(p.pgid), True), "user": ("USER", lambda p, c: c.host.user_name(p.uid), False),
    "uid": ("UID", lambda p, c: str(p.uid), True), "comm": ("COMMAND", lambda p, c: comm_of(p), False),
    "ucmd": ("CMD", lambda p, c: comm_of(p), False),
    "cmd": ("CMD", lambda p, c: cmdline(p), False), "args": ("COMMAND", lambda p, c: cmdline(p), False),
    "command": ("COMMAND", lambda p, c: cmdline(p), False),
    "%cpu": ("%CPU", lambda p, c: f"{p.cpu:.1f}", True), "pcpu": ("%CPU", lambda p, c: f"{p.cpu:.1f}", True),
    "%mem": ("%MEM", lambda p, c: f"{p.mem:.1f}", True), "pmem": ("%MEM", lambda p, c: f"{p.mem:.1f}", True),
    "rss": ("RSS", lambda p, c: str(p.rss), True), "vsz": ("VSZ", lambda p, c: str(p.vsz), True),
    "etime": ("ELAPSED", lambda p, c: etime(p, c.now), True), "time": ("TIME", lambda p, c: cpu_time(p, c.now, True), True),
    "stat": ("STAT", lambda p, c: stat_of(p, c.world.tty), False), "s": ("S", lambda p, c: stat_of(p, c.world.tty)[0], False),
    "tty": ("TT", lambda p, c: p.tty, False), "tt": ("TT", lambda p, c: p.tty, False),
    "start": ("STARTED", lambda p, c: start_text(p, c.now), True), "stime": ("STIME", lambda p, c: start_text(p, c.now), False),
    "lstart": ("STARTED", lambda p, c: strftime("%a %b %e %H:%M:%S %Y", local_dt(p.start, c.tz)), False),
    "nice": ("NI", lambda p, c: str(p.nice), True), "ni": ("NI", lambda p, c: str(p.nice), True),
    "etimes": ("ELAPSED", lambda p, c: str(max((c.now - p.start) // NS, 0)), True),
}
SORT_KEYS = {"pid": lambda p: p.pid, "ppid": lambda p: p.ppid, "%cpu": lambda p: p.cpu, "pcpu": lambda p: p.cpu,
             "%mem": lambda p: p.mem, "pmem": lambda p: p.mem, "rss": lambda p: p.rss, "vsz": lambda p: p.vsz,
             "start_time": lambda p: p.start, "etime": lambda p: -p.start, "cmd": lambda p: cmdline(p),
             "comm": lambda p: comm_of(p), "user": lambda p: p.uid, "time": lambda p: p.cputime, "uid": lambda p: p.uid}


def table(rows, rights, sep=" "):
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    out = []
    for r in rows:
        cells = []
        for i, c in enumerate(r):
            if i == len(r) - 1:
                cells.append(c)
            elif rights[i]:
                cells.append(c.rjust(widths[i]))
            else:
                cells.append(c.ljust(widths[i]))
        out.append(sep.join(cells).rstrip() if not rights[-1] else sep.join(cells))
    return out


@util("ps")
def u_ps(ctx):
    a = ctx.argv[1:]
    bsd = False
    all_ = False
    full = False
    fmt = None
    pids = None
    users = None
    names = None
    sort = None
    headers = True
    i = 0
    while i < len(a):
        x = a[i]
        nxt = a[i + 1] if i + 1 < len(a) else None
        if x in ("aux", "-aux", "axu", "auxww", "auxf", "aux --forest") or (not x.startswith("-") and set(x) <= set("auxwf")):
            bsd = "u" in x
            all_ = all_ or "a" in x or "x" in x
            i += 1
            continue
        if x in ("-e", "-A", "ax", "-ax"):
            all_ = True
        elif x == "-f":
            full = True
        elif x in ("-ef", "-fe", "-eF"):
            all_ = full = True
        elif x in ("-o", "o", "--format") and nxt is not None:
            fmt = (fmt or []) + [f for f in nxt.split(",") if f]
            i += 1
        elif x.startswith("-o") and len(x) > 2:
            fmt = (fmt or []) + x[2:].split(",")
        elif x in ("-p", "p", "--pid") and nxt is not None:
            pids = [int(y) for y in re.split(r"[ ,]+", nxt) if y.isdigit()]
            i += 1
        elif x in ("-u", "-U", "--user") and nxt is not None:
            users = nxt.split(",")
            i += 1
        elif x == "-C" and nxt is not None:
            names = nxt.split(",")
            i += 1
        elif x.startswith("--sort"):
            if "=" in x:
                sort = x.split("=", 1)[1]
            else:
                sort = nxt
                i += 1
        elif x in ("--no-headers", "--no-heading", "h"):
            headers = False
        elif x in ("-l", "-ly", "-j", "--forest", "-H", "f", "w", "ww", "-w"):
            pass
        elif x.isdigit():
            pids = (pids or []) + [int(x)]
        else:
            ctx.e(f"error: unsupported option (BSD syntax)\n\nUsage:\n ps [options]\n\n Try 'ps --help <simple|list|output|threads|misc|all>'\n"
                  f"  or 'ps --help <s|l|o|t|m|a>'\n for additional help text.\n\nFor more details see ps(1).\n")
            return 1
        i += 1
    procs = proc_list(ctx.host)
    me = ctx.proc
    tty_name = me.tty
    if pids is not None:
        procs = [p for p in procs if p.pid in pids]
    elif users is not None:
        procs = [p for p in procs if ctx.host.user_name(p.uid) in users or str(p.uid) in users]
    elif names is not None:
        procs = [p for p in procs if comm_of(p) in names]
    elif not all_:
        procs = [p for p in procs if p.tty == tty_name and p.uid == me.uid] if tty_name != "?" else \
            [p for p in procs if p.uid == me.uid and p.tty == "?" and (p.pgid == me.pgid or p is me)]
    if sort:
        for key in reversed(sort.split(",")):
            rev = key.startswith("-")
            k = key.lstrip("+-")
            fn = SORT_KEYS.get(k)
            if fn:
                procs.sort(key=fn, reverse=rev)
    out = []
    if fmt:
        cols = []
        for f in fmt:
            name, _, hdr = f.partition("=")
            spec = FIELDS.get(name)
            if spec is None:
                ctx.e(f"error: unknown user-defined format specifier \"{name}\"\n\nUsage:\n ps [options]\n")
                return 1
            cols.append((hdr if _ else spec[0], spec[1], spec[2]))
        rows = [[h for h, _, _ in cols]] if headers and any(h for h, _, _ in cols) else []
        for p in procs:
            rows.append([fn(p, ctx) for _, fn, _ in cols])
        if rows:
            out = table(rows, [r for _, _, r in cols])
    elif bsd:
        rows = [["USER", "PID", "%CPU", "%MEM", "VSZ", "RSS", "TTY", "STAT", "START", "TIME", "COMMAND"]]
        for p in procs:
            rows.append([ctx.host.user_name(p.uid), str(p.pid), f"{p.cpu:.1f}", f"{p.mem:.1f}", str(p.vsz), str(p.rss),
                         p.tty, stat_of(p, ctx.world.tty), start_text(p, ctx.now), cpu_time(p, ctx.now), cmdline(p)])
        mins = [8, 7, 4, 4, 6, 5, 8, 4, 5, 6, 0]
        widths = [max(max(len(r[i]) for r in rows), mins[i]) for i in range(11)]
        right = [False, True, True, True, True, True, False, False, True, True, False]
        for r in rows:
            cells = [r[i].rjust(widths[i]) if right[i] else r[i].ljust(widths[i]) for i in range(10)] + [r[10]]
            out.append(" ".join(cells))
    elif full:
        rows = [["UID", "PID", "PPID", "C", "STIME", "TTY", "TIME", "CMD"]]
        for p in procs:
            rows.append([ctx.host.user_name(p.uid), str(p.pid), str(p.ppid), str(int(p.cpu)), start_text(p, ctx.now),
                         p.tty, cpu_time(p, ctx.now, True), cmdline(p)])
        mins = [8, 7, 7, 2, 5, 8, 8, 0]
        widths = [max(max(len(r[i]) for r in rows), mins[i]) for i in range(8)]
        right = [False, True, True, True, False, False, True, False]
        for r in rows:
            out.append(" ".join([r[i].rjust(widths[i]) if right[i] else r[i].ljust(widths[i]) for i in range(7)] + [r[7]]))
    else:
        rows = [["PID", "TTY", "TIME", "CMD"]]
        for p in procs:
            rows.append([str(p.pid), p.tty, cpu_time(p, ctx.now, True), comm_of(p)])
        for k, r in enumerate(rows):
            out.append(f"{r[0]:>7} {r[1]:<8} {r[2]:>8} {r[3]}")
    if not headers and out and not fmt:
        out = out[1:]
    ctx.w("".join(x + "\n" for x in out))
    return 0 if procs or not pids else 1
    yield


def _match_procs(ctx, pattern, full, exact, user, newest, oldest, ignore_case=False):
    from ..regex import RegexError, compile_re
    try:
        rx = compile_re(pattern, "ere", ignore_case)
    except RegexError as e:
        ctx.err(f"regex error: {e}")
        return None
    me = ctx.proc
    hits = []
    for p in proc_list(ctx.host):
        if p is me:
            continue
        if user is not None and ctx.host.user_name(p.uid) not in user and str(p.uid) not in user:
            continue
        text = cmdline(p) if full else comm_of(p)
        if exact:
            ok = rx.fullmatch(text) is not None
        else:
            ok = rx.search(text) is not None
        if ok:
            hits.append(p)
    if newest and hits:
        hits = [max(hits, key=lambda p: (p.start, p.pid))]
    if oldest and hits:
        hits = [min(hits, key=lambda p: (p.start, p.pid))]
    return hits


def _pgrep_args(ctx, a, is_kill):
    o = {"full": False, "exact": False, "user": None, "newest": False, "oldest": False, "list": False, "all": False,
         "count": False, "delim": "\n", "sig": SIGTERM, "icase": False, "pattern": None}
    i = 0
    while i < len(a):
        x = a[i]
        nxt = a[i + 1] if i + 1 < len(a) else None
        if is_kill and re.fullmatch(r"-\d+|-[A-Z]+|-SIG[A-Z]+", x):
            s = x[1:]
            from ..builtins import parse_signal
            sig = parse_signal(s)
            if sig is None:
                ctx.err(f"Unknown signal \"{s}\".")
                return None
            o["sig"] = sig
        elif x in ("--signal",) and nxt:
            from ..builtins import parse_signal
            o["sig"] = parse_signal(nxt)
            i += 1
        elif x in ("-u", "-U", "--euid", "--uid") and nxt:
            o["user"] = nxt.split(",")
            i += 1
        elif x in ("-d", "--delimiter") and nxt:
            o["delim"] = nxt
            i += 1
        elif x.startswith("-") and len(x) > 1 and not x.startswith("--"):
            for c in x[1:]:
                if c == "f":
                    o["full"] = True
                elif c == "x":
                    o["exact"] = True
                elif c == "n":
                    o["newest"] = True
                elif c == "o":
                    o["oldest"] = True
                elif c == "l":
                    o["list"] = True
                elif c == "a":
                    o["all"] = True
                elif c == "c":
                    o["count"] = True
                elif c == "i":
                    o["icase"] = True
                elif c == "e":
                    pass
                else:
                    ctx.err(f"invalid option -- '{c}'")
                    ctx.e(f"\nUsage:\n {ctx.name} [options] <pattern>\n")
                    return None
        elif x == "--full":
            o["full"] = True
        elif x == "--list-full":
            o["all"] = True
        elif x == "--count":
            o["count"] = True
        else:
            if o["pattern"] is not None:
                ctx.err("only one pattern can be provided")
                return None
            o["pattern"] = x
        i += 1
    if o["pattern"] is None and o["user"] is None:
        ctx.err("no matching criteria specified")
        ctx.e(f"Try `{ctx.name} --help' for more information.\n")
        return None
    return o


@util("pgrep")
def u_pgrep(ctx):
    o = _pgrep_args(ctx, ctx.argv[1:], False)
    if o is None:
        return 2
    hits = _match_procs(ctx, o["pattern"] or "", o["full"], o["exact"], o["user"], o["newest"], o["oldest"], o["icase"])
    if hits is None:
        return 2
    if o["count"]:
        ctx.w(f"{len(hits)}\n")
        return 0 if hits else 1
    items = []
    for p in hits:
        if o["all"]:
            items.append(f"{p.pid} {cmdline(p)}")
        elif o["list"]:
            items.append(f"{p.pid} {comm_of(p)}")
        else:
            items.append(str(p.pid))
    if items:
        ctx.w(o["delim"].join(items) + "\n")
    return 0 if hits else 1
    yield


@util("pkill")
def u_pkill(ctx):
    o = _pgrep_args(ctx, ctx.argv[1:], True)
    if o is None:
        return 2
    hits = _match_procs(ctx, o["pattern"] or "", o["full"], o["exact"], o["user"], o["newest"], o["oldest"], o["icase"])
    if hits is None:
        return 2
    killed = 0
    for p in hits:
        if ctx.uid != 0 and p.uid != ctx.uid:
            ctx.err(f"killing pid {p.pid} failed: Operation not permitted")
            continue
        ctx.k.signal(p, o["sig"])
        killed += 1
    if o["count"]:
        ctx.w(f"{killed}\n")
    yield ("yield",)
    return 0 if killed else 1


@util("killall")
def u_killall(ctx):
    a = ctx.argv[1:]
    sig = SIGTERM
    names = []
    for x in a:
        if re.fullmatch(r"-\d+|-[A-Z]+", x):
            from ..builtins import parse_signal
            sig = parse_signal(x[1:])
        elif x in ("-s",):
            continue
        else:
            names.append(x)
    status = 0
    for n in names:
        hits = [p for p in proc_list(ctx.host) if comm_of(p) == n and p is not ctx.proc]
        if not hits:
            ctx.e(f"{n}: no process found\n")
            status = 1
        for p in hits:
            ctx.k.signal(p, sig)
    yield ("yield",)
    return status


@util("pidof")
def u_pidof(ctx):
    names = [x for x in ctx.argv[1:] if not x.startswith("-")]
    pids = [str(p.pid) for p in reversed(proc_list(ctx.host)) if comm_of(p) in names]
    if pids:
        ctx.w(" ".join(pids) + "\n")
    return 0 if pids else 1
    yield


@util("kill")
def u_kill(ctx):
    """procps kill(1) (the bash builtin usually runs instead)."""
    from ..builtins import parse_signal, signal_listing
    a = ctx.argv[1:]
    sig = SIGTERM
    if not a:
        ctx.e("\nUsage:\n kill [options] <pid> [...]\n")
        return 1
    if a[0] in ("-l", "-L", "--list"):
        ctx.w(signal_listing() if len(a) == 1 else "".join(SIGNALS.get(int(x) % 128, "?") + "\n" for x in a[1:]))
        return 0
    if a[0] in ("-s", "--signal") and len(a) > 1:
        sig = parse_signal(a[1])
        a = a[2:]
    elif re.fullmatch(r"-(\d+|[A-Za-z]+)", a[0]):
        sig = parse_signal(a[0][1:])
        a = a[1:]
        if sig is None:
            ctx.err(f"unknown signal name")
            return 1
    status = 0
    for x in a:
        try:
            pid = int(x)
        except ValueError:
            ctx.err(f"cannot find process \"{x}\"")
            status = 1
            continue
        p = ctx.host.procs.get(pid)
        if p is None:
            ctx.err(f"({pid}): No such process")
            status = 1
            continue
        if ctx.uid != 0 and p.uid != ctx.uid:
            ctx.err(f"({pid}): Operation not permitted")
            status = 1
            continue
        ctx.k.signal(p, sig)
    yield ("yield",)
    return status


# ---------------------------------------------------------------------------- top / htop

def top_frame(ctx, limit=None):
    from .basic import _uptime_text
    h = ctx.host
    procs = proc_list(h)
    running = sum(1 for p in procs if p.state == "R" or p.cpu > 50)
    stopped = sum(1 for p in procs if p.state == "T")
    total = len(procs) + 180
    tot_mib = h.mem_kib / 1024
    used = tot_mib * 0.22
    free_ = tot_mib * 0.61
    cache = tot_mib - used - free_
    busy = sum(p.cpu for p in procs) / h.cpus
    lines = [
        "top - " + _uptime_text(ctx).strip().replace(" up ", " up ", 1),
        f"Tasks: {total:3d} total, {running:3d} running, {total - running - stopped:3d} sleeping, {stopped:3d} stopped,   0 zombie",
        f"%Cpu(s): {busy * 0.9:4.1f} us, {busy * 0.1 + 0.5:4.1f} sy,  0.0 ni, {max(100 - busy - 0.5, 0):4.1f} id,  0.1 wa,  0.0 hi,  0.1 si,  0.0 st",
        f"MiB Mem : {tot_mib:8.1f} total, {free_:8.1f} free, {used:8.1f} used, {cache:8.1f} buff/cache",
        f"MiB Swap: {2048.0:8.1f} total, {2048.0:8.1f} free, {0.0:8.1f} used. {tot_mib * 0.78:8.1f} avail Mem",
        "",
        "    PID USER      PR  NI    VIRT    RES    SHR S  %CPU  %MEM     TIME+ COMMAND",
    ]
    rows = sorted(procs, key=lambda p: (-p.cpu, p.pid))
    if limit:
        rows = rows[:limit]
    for p in rows:
        res = p.rss
        res_s = f"{res / 1048576:.1f}g" if res > 9_999_999 else str(res)
        secs = p.cputime
        m, s = divmod(secs, 60)
        lines.append(f"{p.pid:7d} {ctx.host.user_name(p.uid)[:8]:<8}  20 {p.nice:3d} {p.vsz:7d} {res_s:>7} {min(res // 3, 999999):6d} "
                     f"{stat_of(p, None)[0]} {p.cpu:5.1f} {p.mem:5.1f} {int(m):6d}:{s:05.2f} {comm_of(p)}")
    return "\n".join(lines) + "\n"


@util("top")
def u_top(ctx):
    a = ctx.argv[1:]
    batch = "-b" in a
    n = None
    for i, x in enumerate(a):
        if x == "-n" and i + 1 < len(a):
            n = int(a[i + 1])
        elif x.startswith("-n") and x[2:].isdigit():
            n = int(x[2:])
    if batch or n is not None:
        for k in range(n or 1):
            ctx.w(top_frame(ctx) + ("\n" if k < (n or 1) - 1 else ""))
            if k < (n or 1) - 1:
                yield from ctx.flush()
                yield from ctx.sleep(3)
        return 0
    return (yield from until_interrupted(ctx, lambda: "\x1b[H\x1b[2J" + top_frame(ctx, 20), 3.0))


@util("htop")
def u_htop(ctx):
    def frame():
        h = ctx.host
        procs = proc_list(h)
        lines = []
        per = [min(sum(p.cpu for p in procs) / h.cpus + (i * 7) % 11, 100.0) for i in range(h.cpus)]
        for i in range(0, h.cpus, 2):
            cells = []
            for j in (i, i + 1):
                if j < h.cpus:
                    bars = int(per[j] / 100 * 25)
                    cells.append(f"{j:3d}[{'|' * bars}{' ' * (25 - bars)}{per[j]:5.1f}%]")
            lines.append("  ".join(cells))
        tot = h.mem_kib / 1048576
        lines.append(f"  Mem[{'|' * 12}{' ' * 18}{tot * 0.22:.2f}G/{tot:.1f}G]   Tasks: {len(procs) + 60}, 312 thr; 1 running")
        lines.append(f"  Swp[{' ' * 30}0K/2.00G]   Load average: {h.load[0]:.2f} {h.load[1]:.2f} {h.load[2]:.2f}")
        lines.append("")
        lines.append("    PID USER       PRI  NI  VIRT   RES   SHR S  CPU%  MEM%   TIME+  Command")
        for p in sorted(procs, key=lambda p: (-p.cpu, p.pid))[:20]:
            lines.append(f"{p.pid:7d} {ctx.host.user_name(p.uid)[:9]:<9}   20 {p.nice:3d} {p.vsz // 1024:4d}M {p.rss // 1024:4d}M "
                         f"{p.rss // 3072:4d}M {stat_of(p, None)[0]} {p.cpu:5.1f} {p.mem:5.1f} {int(p.cputime) // 60:4d}:{int(p.cputime) % 60:02d}  {cmdline(p)}")
        lines.append("F1Help  F2Setup F3Search F4Filter F5Tree  F6SortBy F7Nice -F8Nice +F9Kill  F10Quit")
        return "\x1b[H\x1b[2J" + "\n".join(lines) + "\n"
    return (yield from until_interrupted(ctx, frame, 1.5))


def until_interrupted(ctx, frame_fn, interval):
    """Redraw `frame_fn()` every `interval` seconds until Ctrl-C (simulated after world.ctrl_c_after s in the foreground)."""
    start = ctx.now
    tty = ctx.world.tty_of(ctx.proc)
    fg = ctx.foreground()
    limit = ctx.world.ctrl_c_after
    while True:
        ctx.w(frame_fn())
        yield from ctx.flush()
        if fg and (ctx.now - start) / NS + interval > limit:
            ctx.world.tty.write("echo", b"^C\n")
            return 130
        yield from ctx.sleep(interval)


# ---------------------------------------------------------------------------- nohup timeout nice renice

@util("nohup")
def u_nohup(ctx):
    a = ctx.argv[1:]
    if a and a[0] == "--":
        a = a[1:]
    if not a:
        ctx.err("missing operand")
        ctx.e("Try 'nohup --help' for more information.\n")
        return 125
    from ..fileio import open_path
    from ..kernel import OpenFile
    proc = ctx.proc
    msgs = []
    if proc.fds.get(0) is not None and proc.fds[0].kind == "tty":
        proc.set_fd(0, OpenFile("null", rd=True, wr=True))
        msgs.append("ignoring input")
    if proc.fds.get(1) is not None and proc.fds[1].kind == "tty":
        try:
            of = open_path(proc, "nohup.out", "a")
            target = "nohup.out"
        except Exception:  # noqa: BLE001
            of = open_path(proc, ctx.env.get("HOME", "/") + "/nohup.out", "a")
            target = ctx.env.get("HOME", "/") + "/nohup.out"
        of.inode.mode = 0o600
        proc.set_fd(1, of)
        msgs.append(f"appending output to '{target}'")
        if proc.fds.get(2) is not None and proc.fds[2].kind == "tty":
            proc.set_fd(2, of)
    elif proc.fds.get(2) is not None and proc.fds[2].kind == "tty":
        pass
    if msgs:
        text = "nohup: " + " and ".join(msgs) + "\n"
        if "ignoring input" in msgs and len(msgs) == 1:
            text = "nohup: ignoring input\n"
        ctx.world.tty.write("err", text.encode())
    proc.ignored.add(SIGHUP)
    path = find_in_path(ctx.fs, a[0], ctx.env.get("PATH", ""), ctx.cwd)
    if path is None:
        ctx.err(f"failed to run command '{a[0]}': No such file or directory")
        return 127
    proc.argv = a
    return (yield from exec_program(proc, path, a, ctx.env, None))


@util("timeout")
def u_timeout(ctx):
    a = ctx.argv[1:]
    sig = SIGTERM
    kill_after = None
    preserve = False
    foreground = False
    while a and a[0].startswith("-") and len(a[0]) > 1:
        x = a.pop(0)
        if x in ("-s", "--signal") and a:
            from ..builtins import parse_signal
            sig = parse_signal(a.pop(0))
        elif x.startswith("--signal="):
            from ..builtins import parse_signal
            sig = parse_signal(x.split("=", 1)[1])
        elif x in ("-k", "--kill-after") and a:
            kill_after = parse_duration(a.pop(0))
        elif x == "--preserve-status":
            preserve = True
        elif x == "--foreground":
            foreground = True
        elif x == "--":
            break
    if len(a) < 2:
        ctx.err("missing operand")
        ctx.e("Try 'timeout --help' for more information.\n")
        return 125
    try:
        dur = parse_duration(a[0])
    except ValueError:
        ctx.err(f"invalid time interval '{a[0]}'")
        ctx.e("Try 'timeout --help' for more information.\n")
        return 125
    cmd = a[1:]
    p = ctx.spawn(cmd)
    if p is None:
        ctx.err(f"failed to run command '{cmd[0]}': No such file or directory")
        return 127
    p.pgid = ctx.proc.pgid
    deadline = ctx.now + int(dur * NS) if dur > 0 else None
    timed_out = False
    while p.state != "Z":
        if deadline is not None and ctx.now >= deadline and not timed_out:
            timed_out = True
            ctx.k.signal(p, sig)
            if kill_after:
                deadline = ctx.now + int(kill_after * NS)
                sig = SIGKILL
                timed_out = False
                kill_after = None
            else:
                deadline = None
            continue
        yield ("wait", [p], deadline)
    if timed_out or (p.term_sig and p.term_sig == sig and deadline is None):
        if preserve:
            return p.status
        return 137 if p.term_sig == SIGKILL else 124
    return p.status


@util("nice")
def u_nice(ctx):
    a = ctx.argv[1:]
    adj = 10
    if not a:
        ctx.w(f"{ctx.proc.nice}\n")
        return 0
    if a[0] in ("-n", "--adjustment") and len(a) > 1:
        adj = int(a[1])
        a = a[2:]
    elif re.fullmatch(r"-\d+", a[0]):
        adj = int(a[0][1:])
        a = a[1:]
    elif a[0].startswith("--adjustment="):
        adj = int(a[0].split("=")[1])
        a = a[1:]
    if adj < 0 and ctx.uid != 0:
        ctx.err("cannot set niceness: Permission denied")
        adj = 0
    ctx.proc.nice = max(-20, min(19, ctx.proc.nice + adj))
    path = find_in_path(ctx.fs, a[0], ctx.env.get("PATH", ""), ctx.cwd)
    if path is None:
        ctx.err(f"'{a[0]}': No such file or directory")
        return 127
    ctx.proc.argv = a
    return (yield from exec_program(ctx.proc, path, a, ctx.env, None))


@util("renice")
def u_renice(ctx):
    a = ctx.argv[1:]
    prio = None
    pids = []
    i = 0
    while i < len(a):
        x = a[i]
        if x in ("-n", "--priority") and i + 1 < len(a):
            prio = int(a[i + 1])
            i += 2
            continue
        if x in ("-p", "--pid"):
            i += 1
            continue
        if re.fullmatch(r"[+-]?\d+", x):
            if prio is None:
                prio = int(x)
            else:
                pids.append(int(x))
        i += 1
    status = 0
    for pid in pids:
        p = ctx.host.procs.get(pid)
        if p is None:
            ctx.err(f"failed to get priority for {pid} (process ID): No such process")
            status = 1
            continue
        old = p.nice
        if prio < old and ctx.uid != 0:
            ctx.err(f"failed to set priority for {pid} (process ID): Permission denied")
            status = 1
            continue
        p.nice = prio
        ctx.w(f"{pid} (process ID) old priority {old}, new priority {prio}\n")
    return status
    yield


# ---------------------------------------------------------------------------- watch

@util("watch")
def u_watch(ctx):
    a = ctx.argv[1:]
    interval = 2.0
    no_title = False
    diff = False
    while a and a[0].startswith("-"):
        x = a.pop(0)
        if x in ("-n", "--interval") and a:
            interval = float(a.pop(0))
        elif x.startswith("-n") and len(x) > 2:
            interval = float(x[2:])
        elif x.startswith("--interval="):
            interval = float(x.split("=")[1])
        elif x in ("-t", "--no-title"):
            no_title = True
        elif x in ("-d", "--differences"):
            diff = True
        elif x in ("-e", "-g", "-c", "--color", "-x", "--exec", "-p"):
            pass
        elif x == "--":
            break
    if not a:
        ctx.e("Usage:\n watch [options] command\n")
        return 1
    command = " ".join(a)
    from ..fileio import read_all
    from ..kernel import OpenFile, Pipe
    start = ctx.now
    tty = ctx.world.tty_of(ctx.proc)
    fg = ctx.foreground()
    limit = ctx.world.ctrl_c_after
    while True:
        pipe = Pipe()
        w = OpenFile("pipe", wr=True, pipe=pipe)
        r = OpenFile("pipe", rd=True, pipe=pipe)
        fds = dict(ctx.proc.fds)
        fds[1] = w
        fds[2] = w
        p = ctx.spawn(["sh", "-c", command], fds=fds)
        r.ref()
        data = yield from read_all(ctx.k, r)
        r.unref(ctx.k)
        if p is not None:
            yield from ctx.wait([p])
        header = ""
        if not no_title:
            left = f"Every {interval:.1f}s: {command}"
            right = f"{ctx.host.name}: " + strftime("%a %b %e %H:%M:%S %Y", local_dt(ctx.now, ctx.tz))
            gap = max(ctx.columns - len(left) - len(right), 1)
            header = left + " " * gap + right + "\n\n"
        ctx.w("\x1b[H\x1b[2J" + header + data.decode("utf-8", "replace"))
        yield from ctx.flush()
        if fg and (ctx.now - start) / NS + interval > limit:
            ctx.world.tty.write("echo", b"^C\n")
            return 130
        yield from ctx.sleep(interval)


# ---------------------------------------------------------------------------- lsof

@util("lsof")
def u_lsof(ctx):
    a = ctx.argv[1:]
    terse = False
    port = None
    net = False
    pid = None
    files = []
    i = 0
    while i < len(a):
        x = a[i]
        if x == "-t":
            terse = True
        elif x == "-i" or x.startswith("-i"):
            net = True
            spec = x[2:] if len(x) > 2 else (a[i + 1] if i + 1 < len(a) and not a[i + 1].startswith("-") else "")
            if len(x) == 2 and spec:
                i += 1
            m = re.search(r":(\d+)", spec)
            if m:
                port = int(m.group(1))
        elif x == "-p" and i + 1 < len(a):
            pid = int(a[i + 1])
            i += 1
        elif x in ("-n", "-P", "-nP", "-Pn", "-w", "-c"):
            pass
        elif x.startswith("-"):
            pass
        else:
            files.append(x)
        i += 1
    rows = []
    for p in proc_list(ctx.host):
        if pid is not None and p.pid != pid:
            continue
        user = ctx.host.user_name(p.uid)
        if net or (pid is None and not files):
            for prt in p.ports:
                if port is not None and prt != port:
                    continue
                rows.append((comm_of(p)[:9], p.pid, user, "3u", "IPv4", str(30000 + prt), "0t0", "TCP", f"*:{prt} (LISTEN)"))
        if files or (pid is not None and not net):
            for fd, of in sorted(p.fds.items()):
                if of.kind == "file" and of.inode is not None:
                    try:
                        path = ctx.fs.path_of(of.inode)
                    except Exception:  # noqa: BLE001
                        continue
                    if files and not any(ctx.abspath(f) == path for f in files):
                        continue
                    mode = "w" if of.wr and not of.rd else ("u" if of.wr else "r")
                    rows.append((comm_of(p)[:9], p.pid, user, f"{fd}{mode}", "REG", "259,2", str(of.inode.size), str(of.inode.ino), path))
    if not rows:
        return 1
    if terse:
        seen = []
        for r in rows:
            if r[1] not in seen:
                seen.append(r[1])
        ctx.w("".join(f"{x}\n" for x in seen))
        return 0
    head = ("COMMAND", "PID", "USER", "FD", "TYPE", "DEVICE", "SIZE/OFF", "NODE", "NAME")
    all_rows = [head] + [tuple(str(x) for x in r) for r in rows]
    w = [max(len(r[k]) for r in all_rows) for k in range(9)]
    right = [False, True, False, False, False, True, True, True, False]
    for r in all_rows:
        ctx.w(" ".join(r[k].rjust(w[k]) if right[k] else r[k].ljust(w[k]) for k in range(8)) + " " + r[8] + "\n")
    return 0
    yield
