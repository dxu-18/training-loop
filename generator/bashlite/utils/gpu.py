"""nvidia-smi, nvtop and fuser: the GPUs of a host and the processes holding their memory.

A process uses a GPU when its `gpu` list has (gpu index, MiB) entries: Python code that imports torch and touches
CUDA gets one while it runs, and setup directives can start long-running jobs that hold some.
"""

from __future__ import annotations

import re

from ..exec import gnu_getopt, util, UsageError
from ..kernel import NS
from ..timeutil import local_dt, strftime
from .procs import until_interrupted

DRIVER = "570.172.08"
CUDA = "12.8"

MODELS = {
    "a100": ("NVIDIA A100-SXM4-80GB", 81920, 400, 62),
    "h100": ("NVIDIA H100 80GB HBM3", 81559, 700, 71),
    "4090": ("NVIDIA GeForce RTX 4090", 24564, 450, 21),
    "3090": ("NVIDIA GeForce RTX 3090", 24576, 350, 19),
    "t4": ("Tesla T4", 15360, 70, 10),
    "l4": ("NVIDIA L4", 23034, 72, 16),
    "v100": ("Tesla V100-SXM2-32GB", 32768, 300, 43),
}
BUS = ["00000000:07:00.0", "00000000:0F:00.0", "00000000:47:00.0", "00000000:4E:00.0",
       "00000000:87:00.0", "00000000:90:00.0", "00000000:B7:00.0", "00000000:BD:00.0"]


def setup_gpus(host, count, model="a100"):
    """Give `host` `count` GPUs of one model, plus the driver's tools and device files."""
    from ..exec import stub_bytes
    from ..world import GPU
    name, mem, cap, idle_w = MODELS.get(model.lower(), MODELS["a100"]) if model.lower() in MODELS else \
        (model, 81920, 400, 62)
    rng = host.world.rng
    host.gpus.clear()
    for i in range(count):
        uid = "-".join("".join(rng.choice("0123456789abcdef") for _ in range(n)) for n in (8, 4, 4, 4, 12))
        consumer = "GeForce" in name
        host.gpus.append(GPU(i, name, mem, 1 if not consumer else 18 + i, 0, 33 + (i * 3) % 5, idle_w + i % 3, cap,
                             BUS[i % len(BUS)] if not consumer else f"00000000:{1 + i:02X}:00.0", f"GPU-{uid}"))
    fs = host.fs
    for tool in ("nvidia-smi", "nvtop"):
        node = fs.write_file("/usr/bin/" + tool, stub_bytes(tool), "/", 0o755, 0, 0)
        node.mode = 0o755
    for i in range(count):
        if not fs.exists(f"/dev/nvidia{i}"):
            fs.mknod(f"/dev/nvidia{i}", f"nvidia{i}", 0o666)
    for extra in ("nvidiactl", "nvidia-uvm"):
        if not fs.exists("/dev/" + extra):
            fs.mknod("/dev/" + extra, extra, 0o666)
    fs.write_file("/proc/driver/nvidia/version",
                  f"NVRM version: NVIDIA UNIX x86_64 Kernel Module  {DRIVER}  Sun Jul 13 09:57:09 UTC 2025\n"
                  "GCC version:  gcc version 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04)\n", "/", 0o444, 0, 0)


def users_of(host, index):
    """[(proc, MiB)] of live processes holding memory on GPU `index`, oldest first."""
    out = []
    for p in sorted(host.procs.values(), key=lambda p: p.pid):
        if p.state == "Z":
            continue
        for g, mib in p.gpu:
            if g == index:
                out.append((p, mib))
    return out


def stats(host, g):
    """(used MiB, util %, temp C, power W, pstate) for a GPU right now."""
    users = users_of(host, g.index)
    used = g.mem_used + sum(m for _, m in users)
    busy = bool(users)
    util = max(g.util, 97 - g.index % 3) if busy else g.util
    temp = g.temp + (29 if busy else 0)
    power = g.power + (int(g.power_cap * 0.68) if busy else 0)
    pstate = "P0" if busy or "GeForce" not in g.name else "P8"
    return used, util, temp, min(power, g.power_cap), pstate


def proc_name(p):
    name = p.argv[0] if p.argv else "?"
    return name


def smi_table(ctx, gpus):
    host = ctx.host
    when = strftime("%a %b %e %H:%M:%S %Y", local_dt(ctx.now, ctx.tz))
    bar = "+" + "-" * 89 + "+"
    mid = "|" + "-" * 41 + "+" + "-" * 24 + "+" + "-" * 22 + "+"
    eq = "|" + "=" * 41 + "+" + "=" * 24 + "+" + "=" * 22 + "|"
    sep = "+" + "-" * 41 + "+" + "-" * 24 + "+" + "-" * 22 + "+"
    head = f" NVIDIA-SMI {DRIVER:<22}Driver Version: {DRIVER:<14}CUDA Version: {CUDA:<9}"
    lines = [when + "       ", bar, "|" + head.ljust(89) + "|", mid,
             "| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |",
             "| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |",
             "|                                         |                        |               MIG M. |", eq]
    for k, g in enumerate(gpus):
        used, util, temp, power, pstate = stats(host, g)
        consumer = "GeForce" in g.name
        pm = "Off" if consumer else "On"
        fan = f"{30 + util // 4}%" if consumer else "N/A"
        ecc = "N/A" if consumer else "0"
        mig = "N/A" if consumer else "Disabled"
        s1 = f"{g.index:>4}  {g.name[:29]:<29}{pm:>4}  "
        s2 = f"{g.bus:>19} {'On' if consumer and g.index == 0 else 'Off':>3} "
        s3 = f"{ecc:>21} "
        r1 = f" {fan:<4} {temp:>3}C    {pstate}" + f"{str(power) + 'W / ' + f'{g.power_cap:>4}W':>24} "
        r2 = f"{used:>8}MiB / {g.mem_total:>6}MiB "
        r3 = f"{util:>7}%  {'Default':>11} "
        lines += [f"|{s1}|{s2}|{s3}|", f"|{r1[:41]:<41}|{r2:>24}|{r3:>22}|", f"|{'':41}|{'':24}|{mig:>21} |"]
        lines.append(sep if k == len(gpus) - 1 else mid)
    lines.append(" " * 91)
    lines += [bar, "| Processes:" + " " * 78 + "|",
              "|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |",
              "|        ID   ID                                                               Usage      |",
              "|" + "=" * 89 + "|"]
    rows = []
    for g in gpus:
        for p, mib in users_of(host, g.index):
            name = proc_name(p)
            if len(name) > 34:
                name = "..." + name[-31:]
            rows.append(f"|  {g.index:>3}   N/A  N/A  {p.pid:>15}      C   {name:<34}{str(mib) + 'MiB':>12} |")
    if not rows:
        rows = ["|  No running processes found" + " " * 61 + "|"]
    lines += rows + [bar]
    return "\n".join(lines) + "\n"


GPU_FIELDS = {
    "index": ("index", lambda c, g, s: str(g.index)),
    "name": ("name", lambda c, g, s: g.name),
    "gpu_name": ("name", lambda c, g, s: g.name),
    "uuid": ("uuid", lambda c, g, s: g.uuid),
    "gpu_uuid": ("uuid", lambda c, g, s: g.uuid),
    "pci.bus_id": ("pci.bus_id", lambda c, g, s: g.bus),
    "gpu_bus_id": ("pci.bus_id", lambda c, g, s: g.bus),
    "driver_version": ("driver_version", lambda c, g, s: DRIVER),
    "count": ("count", lambda c, g, s: str(len(c.host.gpus))),
    "memory.total": ("memory.total [MiB]", lambda c, g, s: (g.mem_total, "MiB")),
    "memory.used": ("memory.used [MiB]", lambda c, g, s: (s[0], "MiB")),
    "memory.free": ("memory.free [MiB]", lambda c, g, s: (g.mem_total - s[0] - (g.mem_total // 160), "MiB")),
    "memory.reserved": ("memory.reserved [MiB]", lambda c, g, s: (g.mem_total // 160, "MiB")),
    "utilization.gpu": ("utilization.gpu [%]", lambda c, g, s: (s[1], "%")),
    "utilization.memory": ("utilization.memory [%]", lambda c, g, s: (s[1] // 2, "%")),
    "temperature.gpu": ("temperature.gpu", lambda c, g, s: str(s[2])),
    "power.draw": ("power.draw [W]", lambda c, g, s: (f"{s[3]:.2f}", "W")),
    "power.limit": ("power.limit [W]", lambda c, g, s: (f"{g.power_cap:.2f}", "W")),
    "fan.speed": ("fan.speed [%]", lambda c, g, s: (30 + s[1] // 4, "%") if "GeForce" in g.name else "[N/A]"),
    "pstate": ("pstate", lambda c, g, s: s[4]),
    "persistence_mode": ("persistence_mode", lambda c, g, s: "Disabled" if "GeForce" in g.name else "Enabled"),
    "compute_mode": ("compute_mode", lambda c, g, s: "Default"),
    "clocks.sm": ("clocks.current.sm [MHz]", lambda c, g, s: (1410 if s[1] else 210, "MHz")),
    "clocks.mem": ("clocks.current.memory [MHz]", lambda c, g, s: (1593, "MHz")),
    "timestamp": ("timestamp", lambda c, g, s: strftime("%Y/%m/%d %H:%M:%S", local_dt(c.now, c.tz)) + ".123"),
}
APP_FIELDS = {
    "pid": ("pid", lambda c, g, p, m: str(p.pid)),
    "process_name": ("process_name", lambda c, g, p, m: proc_name(p)),
    "name": ("process_name", lambda c, g, p, m: proc_name(p)),
    "used_memory": ("used_gpu_memory [MiB]", lambda c, g, p, m: (m, "MiB")),
    "used_gpu_memory": ("used_gpu_memory [MiB]", lambda c, g, p, m: (m, "MiB")),
    "gpu_uuid": ("gpu_uuid", lambda c, g, p, m: g.uuid),
    "gpu_bus_id": ("gpu_bus_id", lambda c, g, p, m: g.bus),
    "gpu_name": ("gpu_name", lambda c, g, p, m: g.name),
    "timestamp": ("timestamp", lambda c, g, p, m: strftime("%Y/%m/%d %H:%M:%S", local_dt(c.now, c.tz)) + ".123"),
}


def _cell(v, units):
    if isinstance(v, tuple):
        return f"{v[0]} {v[1]}" if units else str(v[0])
    return v


def _query(ctx, fields, table, rows_fn, fmt):
    if not fmt or "csv" not in fmt:
        ctx.e('"--format=" switch is missing. Please use --format=csv\n' if not fmt else
              f'Invalid combination of input arguments. Please run \'nvidia-smi -h\' for help.\n')
        return None
    opts = fmt.split(",")
    header, units = "noheader" not in opts, "nounits" not in opts
    names = [f.strip() for f in fields.split(",") if f.strip()]
    for n in names:
        if n not in table:
            ctx.e(f'Field "{n}" is not a valid field to query.\n\n')
            return None
    out = []
    if header:
        heads = [table[n][0] for n in names]
        if not units:
            heads = [re.sub(r" \[.*\]$", "", h) for h in heads]
        out.append(", ".join(heads))
    for row in rows_fn():
        out.append(", ".join(_cell(table[n][1](*row), units) for n in names))
    return "".join(x + "\n" for x in out)


@util("nvidia-smi")
def u_nvidia_smi(ctx):
    host = ctx.host
    try:
        opts, rest = gnu_getopt(ctx.argv[1:], "hLqi:l:f:", ["help", "list-gpus", "query", "id=", "loop=", "query-gpu=",
                                                            "query-compute-apps=", "format=", "loop-ms=", "filename=",
                                                            "version"])
    except UsageError as e:
        ctx.e(f"Invalid combination of input arguments. Please run 'nvidia-smi -h' for help.\n")
        return 2
    if rest:
        ctx.e(f"Invalid combination of input arguments. Please run 'nvidia-smi -h' for help.\n")
        return 2
    o = {}
    for k, v in opts:
        o[k] = v if v is not None else True
    if not host.gpus:
        ctx.w("No devices were found\n")
        return 6
    if "-h" in o or "--help" in o:
        ctx.w(f"NVIDIA System Management Interface -- v{DRIVER}\n\nNVSMI provides monitoring information for Tesla and "
              "select Quadro devices.\n\n    nvidia-smi [OPTION1 [ARG1]] [OPTION2 [ARG2]] ...\n\n"
              "    -h,   --help                Print usage information and exit.\n"
              "    -L,   --list-gpus           Display a list of GPUs connected to the system.\n"
              "    -i,   --id=                 Target a specific GPU.\n"
              "    -l,   --loop=               Probe until Ctrl+C at specified second interval.\n"
              "    -q,   --query               Display GPU or Unit info.\n"
              "    --query-gpu=                Information about GPU.\n"
              "    --query-compute-apps=       List of active compute processes.\n"
              "    --format=                   Comma separated list of format options: csv, noheader, nounits.\n")
        return 0
    if "--version" in o:
        ctx.w(f"NVIDIA-SMI version  : {DRIVER}\nNVML version        : {DRIVER.split('.')[0]}.172\n"
              f"DRIVER version      : {DRIVER}\nCUDA Version        : {CUDA}\n")
        return 0
    gpus = list(host.gpus)
    sel = o.get("-i") or o.get("--id")
    if sel is not None:
        picked = []
        for part in str(sel).split(","):
            part = part.strip()
            g = next((g for g in host.gpus if str(g.index) == part or g.uuid == part or g.bus == part), None)
            if g is None:
                ctx.w(f"No devices were found\n" if False else "")
                ctx.e(f"Invalid combination of input arguments. Please run 'nvidia-smi -h' for help.\n"
                      if not part.isdigit() else "No devices were found\n")
                return 6
            picked.append(g)
        gpus = picked
    if "-L" in o or "--list-gpus" in o:
        ctx.w("".join(f"GPU {g.index}: {g.name} (UUID: {g.uuid})\n" for g in gpus))
        return 0
    loop = o.get("-l") or o.get("--loop")
    fmt = o.get("--format")

    def frame():
        if "--query-gpu" in o:
            return _query(ctx, o["--query-gpu"], GPU_FIELDS, lambda: [(ctx, g, stats(host, g)) for g in gpus], fmt)
        if "--query-compute-apps" in o:
            return _query(ctx, o["--query-compute-apps"], APP_FIELDS,
                          lambda: [(ctx, g, p, m) for g in gpus for p, m in users_of(host, g.index)], fmt)
        if "-q" in o or "--query" in o:
            return detail(ctx, gpus)
        return smi_table(ctx, gpus)
    first = frame()
    if first is None:
        return 2
    if loop is not None:
        try:
            interval = max(1, int(loop))
        except ValueError:
            ctx.e(f"Invalid combination of input arguments. Please run 'nvidia-smi -h' for help.\n")
            return 2
        shown = [first]

        def nxt():
            if shown:
                return shown.pop()
            return frame() or ""
        return (yield from until_interrupted(ctx, nxt, interval))
    ctx.w(first)
    return 0


def detail(ctx, gpus):
    host = ctx.host
    when = strftime("%a %b %e %H:%M:%S %Y", local_dt(ctx.now, ctx.tz))
    out = ["", "==============NVSMI LOG==============", "", f"{'Timestamp':<42}: {when}",
           f"{'Driver Version':<42}: {DRIVER}", f"{'CUDA Version':<42}: {CUDA}", "", f"{'Attached GPUs':<42}: {len(host.gpus)}"]
    for g in gpus:
        used, util, temp, power, pstate = stats(host, g)
        out += [f"GPU {g.bus}", f"    {'Product Name':<38}: {g.name}", f"    {'Persistence Mode':<38}: Enabled",
                f"    {'GPU UUID':<38}: {g.uuid}", f"    {'Minor Number':<38}: {g.index}",
                f"    FB Memory Usage", f"        {'Total':<34}: {g.mem_total} MiB",
                f"        {'Reserved':<34}: {g.mem_total // 160} MiB", f"        {'Used':<34}: {used} MiB",
                f"        {'Free':<34}: {g.mem_total - used - g.mem_total // 160} MiB",
                f"    Utilization", f"        {'Gpu':<34}: {util} %", f"        {'Memory':<34}: {util // 2} %",
                f"    Temperature", f"        {'GPU Current Temp':<34}: {temp} C",
                f"    {'Performance State':<38}: {pstate}", ""]
    return "\n".join(out) + "\n"


@util("nvtop")
def u_nvtop(ctx):
    host = ctx.host
    if not host.gpus:
        ctx.e("No GPU to monitor.\n")
        return 1

    def frame():
        lines = ["\x1b[H\x1b[2J"]
        for g in host.gpus:
            used, util, temp, power, pstate = stats(host, g)
            bar = lambda frac, w: "|" * int(frac * w) + " " * (w - int(frac * w))
            lines.append(f" Device {g.index} [{g.name}] PCIe GEN 4@16x RX: 0.000 KiB/s TX: 0.000 KiB/s")
            lines.append(f" GPU {1410 if util else 210}MHz MEM 1593MHz TEMP {temp:>3}°C FAN N/A% POW {power:>3} / {g.power_cap} W")
            lines.append(f" GPU[{bar(util / 100, 30)}{util:>3}%] MEM[{bar(used / g.mem_total, 22)}"
                         f"{used / 1024:.3f}Gi/{g.mem_total / 1024:.3f}Gi]")
        lines.append("")
        lines.append(f"{'PID':>8} {'USER':<8} {'DEV':>3} {'TYPE':<8} {'GPU':>4} {'GPU MEM':>12} {'CPU':>4} {'HOST MEM':>9} Command")
        for g in host.gpus:
            for p, mib in users_of(host, g.index):
                user = host.user_name(p.uid)
                lines.append(f"{p.pid:>8} {user[:8]:<8} {g.index:>3} {'Compute':<8} {stats(host, g)[1]:>3}% "
                             f"{str(mib) + 'MiB':>8} {mib * 100 // g.mem_total:>2}% {int(p.cpu or 100):>3}% "
                             f"{p.rss // 1024:>6}MiB {' '.join(p.argv)}")
        return "\n".join(lines) + "\n"
    return (yield from until_interrupted(ctx, frame, 1))


# ---------------------------------------------------------------------------- fuser

@util("fuser")
def u_fuser(ctx):
    a = ctx.argv[1:]
    verbose = kill = False
    sig = 9
    ns = None
    names = []
    i = 0
    while i < len(a):
        x = a[i]
        if x == "-n" and i + 1 < len(a):
            ns = a[i + 1]
            i += 2
            continue
        if x.startswith("-") and len(x) > 1 and not x[1:].isdigit() and x[1:] not in ("TERM", "KILL", "INT", "HUP"):
            for c in x[1:]:
                if c == "v":
                    verbose = True
                elif c == "k":
                    kill = True
                elif c in "muaiw":
                    pass
                else:
                    ctx.e(f"fuser: invalid option -- '{c}'\n")
                    return 1
        elif x.startswith("-"):
            from ..builtins import parse_signal
            sig = parse_signal(x[1:]) or 9
        else:
            names.append(x)
        i += 1
    if not names:
        ctx.e("No process specification given\nUsage: fuser [-fIMuvw] [-a|-s] [-4|-6] [-c|-m|-n SPACE]\n"
              "             [-k [-i] [-SIGNAL]] NAME...\n")
        return 1
    host = ctx.host
    found_any = False
    table = []
    for name in names:
        m = re.match(r"^(\d+)/(tcp|udp)$", name)
        port = int(m.group(1)) if m else (int(name) if ns in ("tcp", "udp") and name.isdigit() else None)
        users = []
        if port is not None:
            users = [(p, "F") for p in sorted(host.procs.values(), key=lambda p: p.pid) if port in p.ports and p.state != "Z"]
            label = f"{port}/{m.group(2) if m else ns}"
        else:
            path = ctx.abspath(name)
            label = name
            if not ctx.fs.exists(path):
                ctx.e(f"{name}: No such file or directory\n" if not verbose else f"Specified filename {name} does not exist.\n")
                continue
            dm = re.match(r"^/dev/nvidia(\d+)$", path)
            for p in sorted(host.procs.values(), key=lambda p: p.pid):
                if p.state == "Z":
                    continue
                if dm and any(g == int(dm.group(1)) for g, _ in p.gpu):
                    users.append((p, "F...m"))
                    continue
                if path in ("/dev/nvidiactl", "/dev/nvidia-uvm") and p.gpu:
                    users.append((p, "F...m"))
                    continue
                if p.cwd == path:
                    users.append((p, "..c.."))
                    continue
                for of in p.fds.values():
                    if of.kind == "file" and of.inode is not None:
                        try:
                            if ctx.fs.path_of(of.inode) == path:
                                users.append((p, "F...."))
                                break
                        except Exception:  # noqa: BLE001
                            pass
        if users:
            found_any = True
        if verbose:
            if not table:
                table.append(f"{'':21}{'USER':<12}{'PID':>4} ACCESS COMMAND")
            first = True
            for p, acc in users:
                head = f"{label + ':':<20}" if first else " " * 20
                table.append(f"{head} {host.user_name(p.uid):<10}{p.pid:>6} {acc:<6} {(p.argv[0].rsplit('/', 1)[-1] if p.argv else '?')[:15]}")
                first = False
            if not users:
                table.append(f"{label + ':':<20}")
        else:
            ctx.e(f"{label}:")
            yield from ctx.flush()
            ctx.w("".join(f"{p.pid:>6}{'m' if acc.endswith('m') else ('c' if 'c' in acc else '')}" for p, acc in users))
            yield from ctx.flush()
            ctx.e("\n")
        if kill:
            for p, _ in users:
                if p is not ctx.proc:
                    ctx.k.signal(p, sig)
    if verbose and table:
        ctx.e("".join(x + "\n" for x in table))
    return 0 if found_any else 1
