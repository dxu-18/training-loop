"""The simulated machine(s): hosts, users, an Ubuntu-like base filesystem, the clock, and the terminal Session."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .kernel import NS, TTY, Kernel, OpenFile
from .vfs import FS

DEFAULT_START = datetime(2026, 10, 20, 9, 41, 7, tzinfo=timezone.utc)
DEFAULT_PATH = "/home/ada/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


@dataclass
class User:
    name: str
    uid: int
    gid: int
    gecos: str
    home: str
    shell: str = "/bin/bash"


@dataclass
class GPU:
    index: int
    name: str = "NVIDIA A100-SXM4-80GB"
    mem_total: int = 81920       # MiB
    mem_used: int = 0
    util: int = 0
    temp: int = 34
    power: int = 62
    power_cap: int = 400
    bus: str = ""
    uuid: str = ""


@dataclass
class Service:
    """Something listening on a TCP port: a process (lsof) and/or an HTTP handler(method, path, headers, body)."""
    port: int
    proc: object = None
    handler: object = None
    name: str = ""


class Host:
    def __init__(self, world, name: str, ip: str, pid_base: int = 4000, os_name: str = "linux"):
        self.world = world
        self.name = name
        self.ip = ip
        self.os = os_name
        self.fs = FS(lambda: world.kernel.now)
        self.procs: dict[int, object] = {}
        self._pid = pid_base
        self.users: dict[int, User] = {}
        self.groups: dict[int, tuple[str, set[str]]] = {}
        self.gpus: list[GPU] = []
        self.services: dict[int, Service] = {}
        self.cpus = 8
        self.mem_kib = 32 * 1024 * 1024
        self.boot_ns = world.kernel.now - (12 * 86400 + 3 * 3600 + 4 * 60) * NS
        self.kernel_release = "6.8.0-45-generic"
        self.cpu_model = "Intel(R) Xeon(R) Gold 6338 CPU @ 2.00GHz"
        self.load = (0.52, 0.58, 0.59)
        self.authorized_keys_required = True
        self.ssh_port = 22
        self.motd = ""
        self.extra: dict = {}

    def alloc_pid(self) -> int:
        self._pid += 1
        while self._pid in self.procs:
            self._pid += 1
        return self._pid

    # ------------------------------------------------------------ users

    def add_user(self, name, uid, gid, gecos="", home=None, shell="/bin/bash", groups=()):
        home = home or (f"/home/{name}" if uid else "/root")
        self.users[uid] = User(name, uid, gid, gecos, home, shell)
        if gid not in self.groups:
            self.groups[gid] = (name, set())
        for g in groups:
            gid2 = self.gid_of(g)
            if gid2 is not None:
                self.groups[gid2][1].add(name)

    def user_name(self, uid: int) -> str:
        u = self.users.get(uid)
        return u.name if u else str(uid)

    def group_name(self, gid: int) -> str:
        g = self.groups.get(gid)
        return g[0] if g else str(gid)

    def uid_of(self, name: str):
        if name.isdigit():
            return int(name)
        return next((u.uid for u in self.users.values() if u.name == name), None)

    def gid_of(self, name: str):
        if name.isdigit():
            return int(name)
        return next((g for g, (n, _) in self.groups.items() if n == name), None)

    def user_home(self, name: str):
        u = next((u for u in self.users.values() if u.name == name), None)
        return u.home if u else None

    def home_of(self, uid: int) -> str:
        u = self.users.get(uid)
        return u.home if u else "/"

    def gids_of(self, uid: int) -> set[int]:
        u = self.users.get(uid)
        if u is None:
            return {0}
        return {u.gid} | {g for g, (_, members) in self.groups.items() if u.name in members}


class World:
    def __init__(self, seed: int = 0, start: datetime | None = None, hostname: str = "laptop", gpus: int = 0,
                 os_name: str = "linux"):
        self.seed = seed
        self.rng = random.Random(seed)
        start = start or DEFAULT_START
        self.kernel = Kernel(self, int(start.timestamp()) * NS)
        self.tty = TTY()
        self.hosts: dict[str, Host] = {}
        self.dns: dict[str, str] = {}          # name -> host name (for ssh/ping/curl)
        self.web: dict = {}                    # (host, port, path) or url -> response
        self.toplevel_ppid: dict[int, int] = {}
        self.ctrl_c_after = 10                 # seconds a run-until-Ctrl-C command runs in the foreground
        self.local = self.add_host(hostname, "192.168.1.23", pid_base=4000 + self.rng.randrange(800), os_name=os_name)
        for i in range(gpus):
            self.local.gpus.append(GPU(i))

    def add_host(self, name, ip, pid_base=None, os_name="linux") -> Host:
        from .basefs import build_base
        h = Host(self, name, ip, pid_base if pid_base is not None else 2000 + self.rng.randrange(3000), os_name)
        self.hosts[name] = h
        self.dns[name] = name
        self.dns[ip] = name
        build_base(h)
        boot_procs(h)
        return h

    def resolve(self, name: str):
        hn = self.dns.get(name)
        return self.hosts.get(hn) if hn else None

    def tty_of(self, proc):
        return proc.ctty if proc is not None and proc.ctty is not None else self.tty


# ---------------------------------------------------------------------------- shell start-up

def login_env(host: Host, user: str = "ada", cwd: str | None = None) -> dict[str, str]:
    uid = host.uid_of(user)
    u = host.users[uid]
    return {
        "HOME": u.home, "USER": u.name, "LOGNAME": u.name, "SHELL": u.shell,
        "PATH": DEFAULT_PATH if uid else "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "LANG": "C.UTF-8", "TERM": "xterm-256color", "PWD": cwd or u.home, "SHLVL": "0",
        "EDITOR": "vim", "MAIL": f"/var/mail/{u.name}", "XDG_RUNTIME_DIR": f"/run/user/{uid}",
    }


def init_shell(sh, env: dict, interactive: bool = False):
    """Variables a new bash starts with: the environment (exported) plus bash's own."""
    from .shell import Var
    for k, v in env.items():
        if k.startswith("BASH_FUNC_") and k.endswith("%%"):
            continue
        sh.globals[k] = Var(v, "x")
    proc = sh.proc
    host = sh.host

    def setv(name, value, attrs=""):
        if name not in sh.globals:
            sh.globals[name] = Var(value, attrs)
    setv("BASH", "/usr/bin/bash")
    setv("BASH_VERSION", "5.2.21(1)-release")
    setv("HOSTTYPE", "x86_64")
    setv("MACHTYPE", "x86_64-pc-linux-gnu")
    setv("OSTYPE", "linux-gnu")
    setv("IFS", " \t\n")
    setv("PS2", "> ")
    setv("PS4", "+ ")
    setv("OPTERR", "1")
    sh.globals["OPTIND"] = Var("1")
    setv("HOME", host.home_of(proc.uid))
    setv("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")
    try:
        lvl = int(env.get("SHLVL", "0") or 0) + 1
    except ValueError:
        lvl = 1
    sh.globals["SHLVL"] = Var(str(lvl), "x")
    pwd = env.get("PWD")
    if not pwd or not _same_dir(sh, pwd, proc.cwd):
        pwd = proc.cwd
    sh.globals["PWD"] = Var(pwd, "x")
    sh.dir_logical = pwd
    if interactive:
        setv("PS1", "\\u@\\h:\\w\\$ ")
        setv("HISTFILE", host.home_of(proc.uid) + "/.bash_history")
        setv("HISTSIZE", "500")
        setv("HISTFILESIZE", "500")
        setv("COLUMNS", str(sh.world.tty.width))
        setv("LINES", str(sh.world.tty.height))
    for k, v in env.items():
        if k.startswith("BASH_FUNC_") and k.endswith("%%"):
            name = k[len("BASH_FUNC_"):-2]
            from .parser import Parser
            try:
                lst = Parser(f"{name} {v}").parse_all()
                node = lst.items[0][0].first.cmds[0]
                sh.funcs[name] = node
                sh.exported_funcs = getattr(sh, "exported_funcs", set()) | {name}
            except Exception:  # noqa: BLE001
                pass


def _same_dir(sh, a, b) -> bool:
    try:
        return sh.fs.lookup(a) is sh.fs.lookup(b)
    except Exception:  # noqa: BLE001
        return False


class Session:
    """An interactive terminal session on a host: run(src) types `src` line by line at the prompt."""

    def __init__(self, world: World, user: str = "ada", cwd: str | None = None, host: Host | None = None,
                 rc: bool = True):
        from .shell import Shell
        self.world = world
        host = host or world.local
        self.host = host
        k = world.kernel
        uid = host.uid_of(user)
        u = host.users[uid]
        tty = world.tty
        fds = {0: OpenFile("tty", rd=True, tty=tty, stream="in"),
               1: OpenFile("tty", wr=True, tty=tty, stream="out"),
               2: OpenFile("tty", wr=True, tty=tty, stream="err")}
        cwd = cwd or u.home
        self.proc = k.spawn(host, ["-bash"], None, None, fds=fds, cwd=cwd, uid=uid, gid=u.gid)
        self.proc.gids = host.gids_of(uid)
        self.proc.tty = tty.name
        self.proc.ctty = tty
        self.proc.state = "I"
        self.sh = Shell(world, self.proc, interactive=True)
        self.sh.arg0 = "bash"
        world.toplevel_ppid[self.proc.pid] = self.proc.pid - 7
        init_shell(self.sh, login_env(host, user, cwd), interactive=True)
        if rc:
            rcfile = u.home + "/.bashrc"
            if host.fs.exists(rcfile):
                self.run(f". {rcfile}", quiet=True, history=False)

    @property
    def alive(self) -> bool:
        return self.proc.state != "Z"

    def run(self, src: str, quiet: bool = False, history: bool = True) -> dict:
        """Type `src` at the prompt. -> {segments, stdout, stderr, status, error}"""
        k = self.world.kernel
        tty = self.world.tty
        saved_out, tty.out, tty.size = tty.out, [], 0
        tty.mute = quiet
        error = None
        p = self.proc
        if p.state == "Z":
            raise RuntimeError("the session has ended")
        lines = src.split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        tty.lines.extend(lines)
        p.gen = self.sh.repl(history=history)
        p.state = "R"
        k.runq.append(p)
        from .kernel import Deadlock, OutputLimit, TimeLimit
        try:
            k.run(lambda: p.state in ("I", "Z"))
        except TimeLimit:
            error = "Time limit exceeded: the commands ran too long (an infinite loop?)."
        except OutputLimit:
            error = "Output limit exceeded: the commands printed too much."
        except Deadlock:
            error = "Every process is waiting for input that will never come (deadlock)."
        if error is not None and p.state not in ("I", "Z"):
            # abandon the run: stop everything this run started
            for proc in list(self.host.procs.values()):
                if proc is not p and not proc.daemon:
                    k.kill_now(proc, 9)
            p.gen = None
            p.state = "I"
            k.runq.clear()
        tty.lines.clear()
        tty.input.clear()
        out = tty.out
        tty.out, tty.mute = saved_out, False
        if quiet:
            out = []
        return {
            "segments": [(s, t) for s, t in out],
            "stdout": "".join(t for s, t in out if s == "out"),
            "stderr": "".join(t for s, t in out if s == "err"),
            "text": "".join(t for s, t in out),
            "status": p.status if p.state == "Z" else self.sh.status,
            "exited": p.state == "Z",
            "error": error,
        }


def forever():
    while True:
        yield ("forever",)


DAEMONS = [(1, 0, "/sbin/init splash", 0.0, 0.1, 167324, 11808), (412, 0, "/usr/lib/systemd/systemd-journald", 0.0, 0.1, 50300, 16512),
           (488, 0, "/usr/lib/systemd/systemd-udevd", 0.0, 0.0, 26808, 7040), (902, 0, "/usr/sbin/cron -f -P", 0.0, 0.0, 6696, 2688),
           (1102, 0, "sshd: /usr/sbin/sshd -D [listener] 0 of 10-100 startups", 0.0, 0.0, 12016, 7424),
           (1187, 0, "/usr/sbin/rsyslogd -n -iNONE", 0.0, 0.0, 222400, 5248)]


def boot_procs(host):
    """Long-running system processes, so ps/top look like a real machine."""
    k = host.world.kernel
    saved = host._pid
    for pid, uid, cmd, cpu, mem, vsz, rss in DAEMONS:
        host._pid = pid - 1
        p = k.spawn(host, cmd.split(" "), None, None, cwd="/", uid=uid, gid=uid)
        p.argv = [cmd] if cmd.startswith("sshd:") else cmd.split(" ")
        p.cpu, p.mem, p.vsz, p.rss = cpu, mem, vsz, rss
        p.start = host.boot_ns + (pid // 50) * NS
        p.cputime = 12 if pid == 1 else 3
        p.daemon = True
        p.ppid = 0 if pid == 1 else 1
        if "sshd" in cmd:
            p.ports.append(22)
        k.start(p, forever())
        p.state = "S"
        p.waiting = ("forever",)
        k.runq.remove(p)
    host._pid = saved
    host.services[22] = Service(22, None, None, "ssh")
