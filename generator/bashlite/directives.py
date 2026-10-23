"""`#:` lines in a problem's setup: they describe the machine the code runs on (the site hides them).

    #: gpus 4 [a100|h100|4090|3090|t4|l4|v100]   GPUs (nvidia-smi, CUDA_VISIBLE_DEVICES, torch.cuda)
    #: date 2026-03-14 [09:30[:00]]               the clock at the start (UTC unless `tz`)
    #: tz Europe/Berlin                           TZ for the session
    #: os macos                                   BSD flavours of sed -i, date -v, md5 ...
    #: hostname gpu-box                           the local machine's name
    #: cwd ~/project                              where the session starts (created if missing)
    #: host NAME [ip=10.0.0.5] [user=ada] [key] [password=pw] [port=2222] [private] [via=bastion] [gpus=8]
    #:      [model=h100] [os=macos] [motd=Welcome...] [up=30]   a machine reachable with ssh/scp/rsync/curl/ping
#:                                                 (`up`: it only answers after that many seconds)
    #:                                                 (`key`: ~/.ssh/id_ed25519 exists and is authorized there)
    #: dns ALIAS HOST                             another name for a host
    #: url URL [status=200] [type=text/plain] (FILE | TEXT...)   what curl/wget get (TEXT may use \n)
    #: proc [user=U] [gpu=1:20480,2:20480] [cpu=99.5] [mem=3.1] [port=8000] [ago=3600] [host=H] -- CMD ARGS
    #:                                            a long-running process (training job, server ...)
    #: env NAME=VALUE                             an environment variable in the session
    #: history COMMAND LINE                       a line already in the shell history (oldest first)
    #: venv PATH [PKG[==VER] ...]                 a virtual environment with packages
    #: activate PATH                              ... activated when the session starts
    #: conda [noinit]                             Miniconda in ~/miniconda3 (conda init run unless noinit)
    #: conda-env NAME [python=3.12] [PKG ...]     a conda environment
    #: uv                                         uv installed in ~/.local/bin
    #: ctrlc SECONDS                              how long a run-until-Ctrl-C command runs (default 10)
    #: columns N                                  terminal width
    #: script FILE                                the answer is the file FILE (made executable), not typed
    #: run COMMAND                                typed after the answer (repeatable); default for a
    #:                                            script: ./FILE
    #: type LINE                                  a line typed after the answer (read by prompts)
    #: keys                                       the answer is keystrokes (see keys.py)
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from datetime import datetime, timezone

DIRECTIVE = re.compile(r"^\s*#:\s?(.*)$")


@dataclass
class Spec:
    gpus: int = 0
    model: str = "a100"
    date: datetime | None = None
    tz: str | None = None
    os: str = "linux"
    hostname: str = "laptop"
    cwd: str | None = None
    hosts: list = field(default_factory=list)       # [(name, {opt: value})]
    dns: list = field(default_factory=list)
    urls: list = field(default_factory=list)        # [(url, opts, body)]
    procs: list = field(default_factory=list)       # [(opts, argv)]
    env: list = field(default_factory=list)
    history: list = field(default_factory=list)
    venvs: list = field(default_factory=list)       # [(path, [pkgs])]
    activate: str | None = None
    conda: str | None = None                        # None | "init" | "noinit"
    conda_envs: list = field(default_factory=list)  # [(name, [specs])]
    uv: bool = False
    ctrlc: float | None = None
    columns: int | None = None
    script: str | None = None
    run: list = field(default_factory=list)
    typed: list = field(default_factory=list)
    keys: str | None = None


class DirectiveError(ValueError):
    pass


def split(setup: str) -> tuple[list[str], str]:
    """-> (directive lines, the setup's bash code)."""
    dirs, code = [], []
    for line in setup.split("\n"):
        m = DIRECTIVE.match(line)
        if m:
            dirs.append(m.group(1).strip())
        else:
            code.append(line)
    return dirs, "\n".join(code).strip("\n")


def visible(setup: str) -> str:
    """The setup as shown to the user (without directive lines)."""
    return split(setup)[1]


def _opts(words):
    opts, rest = {}, []
    for k, w in enumerate(words):
        if w == "--":
            rest = words[k + 1:]
            break
        m = re.match(r"^([a-z_]+)=(.*)$", w)
        if m:
            opts[m.group(1)] = m.group(2)
        elif re.match(r"^[a-z]+$", w) and not rest and w in ("key", "private", "noinit"):
            opts[w] = True
        else:
            rest = words[k:]
            break
    return opts, rest


def parse(lines: list[str]) -> Spec:
    sp = Spec()
    for line in lines:
        if not line:
            continue
        name, _, arg = line.partition(" ")
        arg = arg.strip()
        if name == "gpus":
            parts = arg.split()
            sp.gpus = int(parts[0])
            if len(parts) > 1:
                sp.model = parts[1]
        elif name == "date":
            parts = arg.split()
            day = parts[0]
            tm = parts[1] if len(parts) > 1 else "09:41:07"
            if tm.count(":") == 1:
                tm += ":00"
            sp.date = datetime.fromisoformat(f"{day}T{tm}").replace(tzinfo=timezone.utc)
        elif name == "tz":
            sp.tz = arg
        elif name == "os":
            sp.os = arg
        elif name == "hostname":
            sp.hostname = arg
        elif name == "cwd":
            sp.cwd = arg
        elif name == "host":
            words = shlex.split(arg)
            opts, _ = _opts(words[1:])
            sp.hosts.append((words[0], opts))
        elif name == "dns":
            a, b = arg.split()
            sp.dns.append((a, b))
        elif name == "url":
            words = arg.split(" ")
            url = words[0]
            opts, rest = {}, []
            for k, w in enumerate(words[1:]):
                m = re.match(r"^(status|type)=(\S+)$", w)
                if m and not rest:
                    opts[m.group(1)] = m.group(2)
                else:
                    rest = words[1 + k:]
                    break
            sp.urls.append((url, opts, " ".join(rest)))
        elif name == "proc":
            opts, argv = _opts(shlex.split(arg))
            if not argv:
                raise DirectiveError(f"#: proc needs a command: {line}")
            sp.procs.append((opts, argv))
        elif name == "env":
            k, _, v = arg.partition("=")
            sp.env.append((k, v))
        elif name == "history":
            sp.history.append(arg)
        elif name == "venv":
            parts = arg.split()
            sp.venvs.append((parts[0], parts[1:]))
        elif name == "activate":
            sp.activate = arg
        elif name == "conda":
            sp.conda = "noinit" if arg == "noinit" else "init"
        elif name == "conda-env":
            parts = arg.split()
            sp.conda_envs.append((parts[0], parts[1:]))
            sp.conda = sp.conda or "init"
        elif name == "uv":
            sp.uv = True
        elif name == "ctrlc":
            sp.ctrlc = float(arg)
        elif name == "columns":
            sp.columns = int(arg)
        elif name == "script":
            sp.script = arg
        elif name == "run":
            sp.run.append(arg)
        elif name == "type":
            sp.typed.append(arg)
        elif name == "keys":
            sp.keys = arg or "readline"
        else:
            raise DirectiveError(f"unknown directive: #: {line}")
    return sp


# ---------------------------------------------------------------------------- building the machine

def ensure_user(host, name):
    """A login account (home directory with the default dotfiles) on `host`."""
    uid = host.uid_of(name)
    if uid is not None and uid in host.users:
        return uid
    from .basefs import BASHRC, PROFILE
    uid = 1000 + len([u for u in host.users if u >= 1000])
    while uid in host.users:
        uid += 1
    host.add_user(name, uid, uid, f"{name},,,", groups=())
    fs = host.fs
    home = host.users[uid].home
    fs.mkdir(home, "/", 0o750, uid, uid, parents=True)
    fs.write_file(home + "/.bashrc", BASHRC, "/", 0o644, uid, uid)
    fs.write_file(home + "/.profile", PROFILE, "/", 0o644, uid, uid)
    for path, line in (("/etc/passwd", f"{name}:x:{uid}:{uid}:{name},,,:{home}:/bin/bash\n"),
                       ("/etc/group", f"{name}:x:{uid}:\n")):
        try:
            old = bytes(fs.lookup(path).data).decode()
        except Exception:  # noqa: BLE001
            old = ""
        fs.write_file(path, old + line, "/", 0o644, 0, 0)
    return uid


def _local_key(world, user="ada"):
    from .utils.net import write_keypair
    host = world.local
    uid = host.uid_of(user)
    home = host.users[uid].home
    path = f"{home}/.ssh/id_ed25519"
    if host.fs.exists(path + ".pub"):
        return bytes(host.fs.lookup(path + ".pub").data).decode()
    return write_keypair(host.fs, path, f"{user}@{host.name}", uid, host.users[uid].gid, f"{world.seed}:local-key")


def _add_host(world, name, o):
    ip = o.get("ip") or f"10.0.{len(world.hosts) % 250}.{10 + len(world.hosts)}"
    h = world.add_host(name, ip, os_name=o.get("os", "linux"))
    user = o.get("user", "ada")
    uid = ensure_user(h, user)
    home = h.users[uid].home
    if "port" in o:
        h.ssh_port = int(o["port"])
    if o.get("password"):
        h.extra["password"] = o["password"]
    if o.get("private"):
        h.extra["private"] = True
        h.extra["reachable_from"] = [x for x in o.get("via", "").split(",") if x]
    if o.get("up"):           # still booting: unreachable for this many seconds
        from .kernel import NS
        h.extra["up_after"] = world.kernel.now + int(float(o["up"]) * NS)
    if o.get("motd"):
        h.motd = o["motd"].replace("\\n", "\n") + "\n"
    if o.get("gpus"):
        from .utils.gpu import setup_gpus
        setup_gpus(h, int(o["gpus"]), o.get("model", "a100"))
    if o.get("key"):
        pub = _local_key(world)
        fs = h.fs
        fs.mkdir(home + "/.ssh", "/", 0o700, uid, uid, parents=True)
        path = home + "/.ssh/authorized_keys"
        old = bytes(fs.lookup(path).data).decode() if fs.exists(path) else ""
        node = fs.write_file(path, old + pub, "/", 0o600, uid, uid)
        node.mode = 0o600
    # the local machine already knows the host key (no "authenticity of host" prompt)
    known = world.local.users[world.local.uid_of("ada")].home + "/.ssh/known_hosts"
    fs = world.local.fs
    if not fs.exists(known.rsplit("/", 1)[0]):
        fs.mkdir(known.rsplit("/", 1)[0], "/", 0o700, 1000, 1000, parents=True)
    old = bytes(fs.lookup(known).data).decode() if fs.exists(known) else ""
    import base64
    import hashlib
    blob = base64.b64encode(b"\x00\x00\x00\x0bssh-ed25519\x00\x00\x00 " + hashlib.sha256(name.encode()).digest()).decode()
    node = fs.write_file(known, old + f"{name},{ip} ssh-ed25519 {blob}\n", "/", 0o644, 1000, 1000)
    node.mode = 0o644
    return h


def _url(world, url, opts, body):
    from .utils.net import HttpResult
    status = int(opts.get("status", 200))
    ctype = opts.get("type") or ("application/json" if url.endswith(".json") or body.lstrip().startswith(("{", "["))
                                 else "text/csv" if url.endswith(".csv") else "text/plain")
    local = world.local

    def handler(method, path, headers, req_body, body=body):
        if body.startswith("/") and " " not in body:
            try:
                data = bytes(local.fs.lookup(body).data)
            except Exception:  # noqa: BLE001
                return HttpResult(404, {"Content-Type": "text/html"}, b"<html><body>404 Not Found</body></html>\n")
        else:
            data = body.replace("\\n", "\n").encode()
            if data and not data.endswith(b"\n"):
                data += b"\n"
        return HttpResult(status, {"Content-Type": ctype}, data if method != "HEAD" else b"")
    world.web[url] = handler


def _proc(world, opts, argv):
    from .kernel import NS
    from .world import Service, forever
    host = world.hosts[opts["host"]] if opts.get("host") else world.local
    user = opts.get("user", "ada")
    uid = ensure_user(host, user) if user != "root" else 0
    gid = host.users[uid].gid
    k = world.kernel
    p = k.spawn(host, argv, None, None, cwd=opts.get("cwd") or host.users[uid].home, uid=uid, gid=gid)
    p.ppid = 1
    p.daemon = True
    p.cpu = float(opts.get("cpu", 0.0))
    p.mem = float(opts.get("mem", 0.1))
    p.rss = int(opts.get("rss", 20480 if not opts.get("gpu") else 3145728))
    p.vsz = p.rss * 3
    p.tty = opts.get("tty", "?")
    ago = float(opts.get("ago", 3600))
    p.start = k.now - int(ago * NS)
    p.cputime = int(ago * p.cpu / 100)
    if opts.get("nice"):
        p.nice = int(opts["nice"])
    if opts.get("gpu"):
        for part in opts["gpu"].split(","):
            g, _, mib = part.partition(":")
            p.gpu.append((int(g), int(mib or 2048)))
    if opts.get("port"):
        port = int(opts["port"])
        p.ports.append(port)
        from .utils.net import HttpResult
        host.services[port] = Service(port, p, lambda m, path, h, b: HttpResult(200, {"Content-Type": "text/plain"},
                                                                                b"OK\n"), argv[0])
    p.env = {"HOME": host.users[uid].home, "PATH": "/usr/local/bin:/usr/bin:/bin", "USER": user}
    k.start(p, forever())
    p.state = "S"
    p.waiting = ("forever",)
    if p in k.runq:
        k.runq.remove(p)
    return p


def build(spec: Spec, seed: int = 0):
    """-> (world, session) ready for the setup code."""
    from . import utils  # noqa: F401  (registers the commands)
    from .world import Session, World
    world = World(seed=seed, start=spec.date, hostname=spec.hostname, gpus=spec.gpus, os_name=spec.os)
    if spec.gpus and spec.model != "a100":
        from .utils.gpu import setup_gpus
        setup_gpus(world.local, spec.gpus, spec.model)
    if spec.ctrlc is not None:
        world.ctrl_c_after = spec.ctrlc
    if spec.columns:
        world.tty.width = spec.columns
    for name, opts in spec.hosts:
        _add_host(world, name, opts)
    for alias, target in spec.dns:
        world.dns[alias] = target
    for url, opts, body in spec.urls:
        _url(world, url, opts, body)
    fs = world.local.fs
    home = "/home/ada"
    if spec.conda:
        from .utils.pyenv import install_conda
        install_conda(world.local, init=spec.conda == "init")
    if spec.uv:
        from .exec import stub_bytes
        fs.mkdir(home + "/.local/bin", "/", 0o755, 1000, 1000, parents=True)
        node = fs.write_file(home + "/.local/bin/uv", stub_bytes("uv"), "/", 0o755, 1000, 1000)
        node.mode = 0o755
    for opts, argv in spec.procs:
        _proc(world, opts, argv)
    sess = Session(world, cwd=home)
    sh = sess.sh
    for k_, v in spec.env:
        sh.set_var(k_, v)
        sh.globals[k_].attrs.add("x")
    if spec.tz:
        sh.set_var("TZ", spec.tz)
        sh.globals["TZ"].attrs.add("x")
    if spec.columns:
        sh.set_var("COLUMNS", str(spec.columns))
    return world, sess


def _cd_line(spec: Spec) -> str:
    if not spec.cwd:
        return "cd ~"
    c = spec.cwd
    q = ("~/" + shlex.quote(c[2:]) if c.startswith("~/") and len(c) > 2 else "~" if c == "~" else shlex.quote(c))
    return f"mkdir -p {q} && cd {q}"


def prepare(sess, spec: Spec):
    """Before the setup code: go to the start directory and create the environments there."""
    lines = [_cd_line(spec)]
    for path, pkgs in spec.venvs:
        lines.append(f"python3 -m venv {shlex.quote(path)}")
        if pkgs:
            lines.append(f"{shlex.quote(path)}/bin/pip install -q " + " ".join(shlex.quote(x) for x in pkgs) + " 2>/dev/null")
    for name, specs in spec.conda_envs:
        lines.append(f"~/miniconda3/bin/conda create -y -n {shlex.quote(name)} " + " ".join(shlex.quote(x) for x in specs)
                     + " >/dev/null")
    sess.run("\n".join(lines), quiet=True, history=False)


def start_session(sess, spec: Spec):
    """After the setup code: back to the start directory, activate the environment, load the history."""
    lines = [_cd_line(spec)]
    if spec.activate:
        lines.append(f"source {shlex.quote(spec.activate)}/bin/activate")
    sess.run("\n".join(lines), quiet=True, history=False)
    sh = sess.sh
    sh.history = list(spec.history)
    sh.prompt_no = 0
    sh.status = 0
