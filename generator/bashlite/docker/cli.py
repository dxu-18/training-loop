"""The `docker` command: dispatch, flag parsing, the daemon checks, and the container commands (run, create,
start/stop/restart/kill, rm, ps, logs, exec, cp, inspect, top, port, stats, wait, rename, pause, prune).
Image, volume, network, system and registry commands are in cmd_images.py / cmd_misc.py; build in build.py;
compose in compose.py."""

from __future__ import annotations

import json
import posixpath
import re
import shlex

from ..exec import util
from ..kernel import NS
from ..vfs import FSError
from . import gotmpl
from .engine import ENGINE, API, DockerError, ago, daemon_of, human_size, parse_size

COMMANDS: dict = {}


def cmd(*names):
    def reg(fn):
        for n in names:
            COMMANDS[n] = fn
        return fn
    return reg


class Out:
    """What a docker command writes (stdout / stderr), flushed as it goes."""

    def __init__(self, ctx):
        self.ctx = ctx

    def out(self, s):
        self.ctx.w(s)

    def err(self, s):
        self.ctx._flush_nb()
        self.ctx.e(s)


# ---------------------------------------------------------------------------- flags

def flags(args, spec, stop_at_positional=True):
    """spec: {"-d|--detach": None (bool) | "v" (one value) | "m" (repeatable)}. -> (opts, rest)."""
    table = {}
    for key, kind in spec.items():
        names = key.split("|")
        longs = [n for n in names if n.startswith("--")]
        canon = (longs[0] if longs else names[-1]).lstrip("-")
        for n in names:
            table[n] = (canon, kind)
    opts: dict = {}
    rest = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--":
            rest += args[i + 1:]
            break
        if a.startswith("--") and len(a) > 2:
            name, eq, val = a.partition("=")
            if name not in table:
                raise DockerError(f"unknown flag: {name}\nSee 'docker --help'.", 125)
            canon, kind = table[name]
            if kind is None:
                opts[canon] = val.lower() != "false" if eq else True
            else:
                if not eq:
                    if i + 1 >= len(args):
                        raise DockerError(f"flag needs an argument: {name}", 125)
                    i += 1
                    val = args[i]
                if kind == "m":
                    opts.setdefault(canon, []).append(val)
                else:
                    opts[canon] = val
            i += 1
            continue
        if a.startswith("-") and len(a) > 1 and a != "-":
            j = 1
            while j < len(a):
                n = "-" + a[j]
                if n not in table:
                    raise DockerError(f"unknown shorthand flag: '{a[j]}' in {a}\nSee 'docker --help'.", 125)
                canon, kind = table[n]
                if kind is None:
                    opts[canon] = True
                    j += 1
                    continue
                val = a[j + 1:]
                if val.startswith("="):
                    val = val[1:]
                if not val:
                    if i + 1 >= len(args):
                        raise DockerError(f"flag needs an argument: '{a[j]}' in {a}", 125)
                    i += 1
                    val = args[i]
                if kind == "m":
                    opts.setdefault(canon, []).append(val)
                else:
                    opts[canon] = val
                break
            i += 1
            continue
        if stop_at_positional:
            rest += args[i:]
            break
        rest.append(a)
        i += 1
    return opts, rest


# ---------------------------------------------------------------------------- the entry point

USAGE = """
Usage:  docker [OPTIONS] COMMAND

A self-sufficient runtime for containers

Common Commands:
  run         Create and run a new container from an image
  exec        Execute a command in a running container
  ps          List containers
  build       Build an image from a Dockerfile
  pull        Download an image from a registry
  push        Upload an image to a registry
  images      List images
  login       Authenticate to a registry
  logout      Log out from a registry
  version     Show the Docker version information
  info        Display system-wide information

Management Commands:
  builder     Manage builds
  buildx*     Docker Buildx
  compose*    Docker Compose
  container   Manage containers
  image       Manage images
  network     Manage networks
  scout*      Docker Scout
  system      Manage Docker
  volume      Manage volumes

Commands:
  cp          Copy files/folders between a container and the local filesystem
  create      Create a new container
  inspect     Return low-level information on Docker objects
  kill        Kill one or more running containers
  logs        Fetch the logs of a container
  pause       Pause all processes within one or more containers
  port        List port mappings or a specific mapping for the container
  rename      Rename a container
  restart     Restart one or more containers
  rm          Remove one or more containers
  rmi         Remove one or more images
  save        Save one or more images to a tar archive (streamed to STDOUT by default)
  load        Load an image from a tar archive or STDIN
  start       Start one or more stopped containers
  stats       Display a live stream of container(s) resource usage statistics
  stop        Stop one or more running containers
  tag         Create a tag TARGET_IMAGE that refers to SOURCE_IMAGE
  top         Display the running processes of a container
  unpause     Unpause all processes within one or more containers
  wait        Block until one or more containers stop, then print their exit codes

Run 'docker COMMAND --help' for more information on a command.
"""

MANAGEMENT = {"container": {"ls": "ps", "list": "ps", "run": "run", "create": "create", "start": "start",
                            "stop": "stop", "restart": "restart", "kill": "kill", "rm": "rm", "remove": "rm",
                            "logs": "logs", "exec": "exec", "cp": "cp", "inspect": "inspect", "top": "top",
                            "port": "port", "stats": "stats", "prune": "container-prune", "wait": "wait",
                            "rename": "rename", "pause": "pause", "unpause": "unpause"},
              "image": {"ls": "images", "list": "images", "pull": "pull", "push": "push", "tag": "tag", "rm": "rmi",
                        "remove": "rmi", "prune": "image-prune", "history": "history", "inspect": "image-inspect",
                        "save": "save", "load": "load", "build": "build"},
              "volume": None, "network": None, "system": None, "builder": None, "buildx": None, "compose": None,
              "scout": None, "context": None}


@util("docker")
def u_docker(ctx):
    a = list(ctx.argv[1:])
    o = Out(ctx)
    if not a or a[0] in ("--help", "-h", "help"):
        ctx.w(USAGE)
        return 0 if a else 0
    if a[0] in ("--version", "-v"):
        ctx.w(f"Docker version {ENGINE}, build ce12230\n")
        return 0
    if a[0] == "version":
        return (yield from _version(ctx, a[1:]))
    sub = a[0]
    rest = a[1:]
    if sub in MANAGEMENT and MANAGEMENT[sub] is not None and rest and not rest[0].startswith("-"):
        name = MANAGEMENT[sub].get(rest[0])
        if name is None:
            o.err(f"docker: '{sub} {rest[0]}' is not a docker command.\nSee 'docker --help'\n")
            return 1
        sub, rest = name, rest[1:]
    elif sub in ("container", "image") and (not rest or rest[0] in ("--help", "-h")):
        ctx.w(f"\nUsage:  docker {sub} COMMAND\n\nManage {sub}s\n\nCommands:\n" +
              "".join(f"  {k}\n" for k in MANAGEMENT[sub]) + f"\nRun 'docker {sub} COMMAND --help' for more "
              f"information on a command.\n")
        return 0
    if sub == "buildx" and rest and rest[0] == "build":
        sub, rest = "build", rest[1:]
    fn = COMMANDS.get(sub)
    if fn is None:
        o.err(f"docker: '{sub}' is not a docker command.\nSee 'docker --help'\n")
        return 1
    if "--help" in rest or "-h" in rest and sub not in ("run", "exec", "create", "compose"):
        from .helptext import HELP
        ctx.w(HELP.get(sub, f"\nUsage:  docker {sub} [OPTIONS]\n\nRun 'docker --help' for the list of commands.\n"))
        return 0
    try:
        check_daemon(ctx)
        r = fn(ctx, o, rest)
        if hasattr(r, "send"):
            r = yield from r
        return r if isinstance(r, int) else 0
    except DockerError as e:
        if e.msg:
            o.err(e.msg.rstrip("\n") + "\n")
        return e.status


def check_daemon(ctx):
    d = daemon_of(ctx.world, ctx.host)
    if not d.installed:
        raise DockerError("", 127)
    if ctx.uid != 0 and 999 not in ctx.proc.gids:
        raise DockerError("permission denied while trying to connect to the Docker daemon socket at unix:///var/run/"
                          "docker.sock: Get \"http://%2Fvar%2Frun%2Fdocker.sock/v1.47/containers/json\": dial unix "
                          "/var/run/docker.sock: connect: permission denied", 1)
    if not d.running:
        raise DockerError("Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon "
                          "running?", 1)
    return d


def _version(ctx, args):
    opts, _ = flags(args, {"-f|--format": "v"})
    d = daemon_of(ctx.world, ctx.host)
    client = {"Version": ENGINE, "ApiVersion": API, "GoVersion": "go1.22.7", "GitCommit": "ce12230",
              "Os": "linux", "Arch": "amd64", "Context": "default"}
    server = {"Version": ENGINE, "ApiVersion": API, "MinAPIVersion": "1.24", "GoVersion": "go1.22.7",
              "Os": "linux", "Arch": "amd64"}
    ok = True
    err = ""
    try:
        check_daemon(ctx)
    except DockerError as e:
        ok, err = False, e.msg
    if opts.get("format"):
        ctx.w(gotmpl.render(opts["format"], {"Client": client, "Server": server if ok else None}) + "\n")
        return 0 if ok else 1
    ctx.w(f"Client: Docker Engine - Community\n Version:           {ENGINE}\n API version:       {API}\n Go version:   "
          f"     go1.22.7\n Git commit:        ce12230\n Built:             Fri Sep 20 11:41:00 2024\n OS/Arch:        "
          f"   linux/amd64\n Context:           default\n")
    if not ok:
        ctx.e(err + "\n")
        return 1
    ctx.w(f"\nServer: Docker Engine - Community\n Engine:\n  Version:          {ENGINE}\n  API version:      {API} "
          f"(minimum version 1.24)\n  Go version:       go1.22.7\n  Git commit:       41ca978\n  Built:            "
          f"Fri Sep 20 11:41:00 2024\n  OS/Arch:          linux/amd64\n  Experimental:     false\n containerd:\n  "
          f"Version:          1.7.22\n runc:\n  Version:          1.1.14\n docker-init:\n  Version:          0.19.0\n")
    return 0
    yield


# ---------------------------------------------------------------------------- run / create

RUN_FLAGS = {"-d|--detach": None, "-i|--interactive": None, "-t|--tty": None, "--rm": None, "--name": "v",
             "-p|--publish": "m", "-P|--publish-all": None, "-e|--env": "m", "--env-file": "m", "-v|--volume": "m",
             "--mount": "m", "--tmpfs": "m", "-w|--workdir": "v", "-u|--user": "v", "--entrypoint": "v",
             "--network|--net": "v", "--network-alias|--net-alias": "m", "--add-host": "m", "--gpus": "v",
             "--shm-size": "v", "--cpus": "v", "-m|--memory": "v", "--restart": "v", "-h|--hostname": "v",
             "--init": None, "-l|--label": "m", "--platform": "v", "--pull": "v", "--stop-timeout": "v",
             "--ipc": "v", "--privileged": None, "--health-cmd": "v", "--health-interval": "v",
             "--health-retries": "v", "--read-only": None, "-q|--quiet": None, "--runtime": "v",
             "--stop-signal": "v", "--memory-swap": "v", "--cap-add": "m", "--device": "m", "--pid": "v",
             "--ulimit": "m", "--security-opt": "m", "--log-driver": "v", "--expose": "m"}


def parse_publish(spec, img):
    """-p [ip:][hostPort:]containerPort[/proto] -> [(ip, host port, container port, proto)]"""
    proto = "tcp"
    if "/" in spec:
        spec, proto = spec.rsplit("/", 1)
    parts = spec.split(":")
    if len(parts) == 1:
        return [("0.0.0.0", 32768, int(parts[0]), proto)]
    if len(parts) == 2:
        return [("0.0.0.0", int(parts[0]) if parts[0] else 32768, int(parts[1]), proto)]
    return [(parts[0] or "0.0.0.0", int(parts[1]) if parts[1] else 32768, int(parts[2]), proto)]


def host_path(ctx, p):
    if p.startswith("~"):
        p = ctx.env.get("HOME", "/root") + p[1:]
    return posixpath.normpath(p if p.startswith("/") else posixpath.join(ctx.cwd, p))


def parse_volume(ctx, spec):
    parts = spec.split(":")
    ro = False
    if len(parts) >= 3 and parts[-1] in ("ro", "rw", "z", "Z", "ro,z"):
        ro = parts[-1].startswith("ro")
        parts = parts[:-1]
    if len(parts) == 1:
        return {"type": "volume", "source": None, "dest": parts[0], "ro": ro}
    src, dest = parts[0], parts[1]
    if not dest.startswith("/"):
        raise DockerError(f"docker: Error response from daemon: invalid volume specification: '{spec}': invalid mount "
                          f"config for type \"volume\": invalid mount path: '{dest}' mount path must be absolute.", 125)
    if src.startswith(("/", ".", "~")):
        return {"type": "bind", "source": host_path(ctx, src), "dest": dest, "ro": ro}
    return {"type": "volume", "source": src, "dest": dest, "ro": ro}


def parse_mount(ctx, spec):
    kv = {}
    for item in spec.split(","):
        k, _, v = item.partition("=")
        kv[k.strip()] = v.strip() if v else "true"
    t = kv.get("type", "volume")
    src = kv.get("source") or kv.get("src")
    dst = kv.get("target") or kv.get("destination") or kv.get("dst")
    if not dst:
        raise DockerError("docker: invalid argument \"" + spec + "\" for \"--mount\" flag: target is required", 125)
    ro = kv.get("readonly", kv.get("ro", "false")) in ("true", "1", "")
    if "readonly" in kv and kv["readonly"] == "true":
        ro = True
    if t == "bind":
        return {"type": "bind", "source": host_path(ctx, src or "."), "dest": dst, "ro": ro, "long": True}
    if t == "tmpfs":
        return {"type": "tmpfs", "dest": dst}
    return {"type": "volume", "source": src, "dest": dst, "ro": ro}


def read_env_file(ctx, path):
    try:
        text = bytes(ctx.fs.lookup(host_path(ctx, path)).data).decode("utf-8", "replace")
    except FSError:
        raise DockerError(f"docker: open {path}: no such file or directory.\nSee 'docker run --help'.", 125)
    out = []
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            v = ctx.env.get(line)
            if v is not None:
                out.append(f"{line}={v}")
            continue
        out.append(line)
    return out


def container_config(ctx, d, opts, img, command, image_ref):
    cfg = {"image_ref": image_ref, "name": opts.get("name"), "tty": bool(opts.get("tty")),
           "interactive": bool(opts.get("interactive")), "rm": bool(opts.get("rm"))}
    if command:
        cfg["cmd"] = list(command)
    if "entrypoint" in opts:
        ep = opts["entrypoint"]
        cfg["entrypoint"] = shlex.split(ep) if ep else []
        cfg["entrypoint_set"] = True
        if not command:
            cfg["cmd"] = []
    env = []
    for f in opts.get("env-file", []):
        env += read_env_file(ctx, f)
    for e in opts.get("env", []):
        if "=" in e:
            env.append(e)
        elif e in ctx.env:
            env.append(f"{e}={ctx.env[e]}")
    cfg["env"] = env
    if opts.get("workdir"):
        cfg["workdir"] = opts["workdir"]
    if opts.get("user"):
        cfg["user"] = opts["user"]
    if opts.get("hostname"):
        cfg["hostname"] = opts["hostname"]
    cfg["network"] = opts.get("network") or "bridge"
    cfg["aliases"] = list(opts.get("network-alias", []))
    cfg["add_hosts"] = []
    for h in opts.get("add-host", []):
        n, _, ip = h.partition(":") if "=" not in h else h.partition("=")
        cfg["add_hosts"].append((n, ip))
    ports = []
    for p in opts.get("publish", []):
        try:
            ports += parse_publish(p, img)
        except ValueError:
            raise DockerError(f"docker: invalid containerPort: {p}.\nSee 'docker run --help'.", 125)
    if opts.get("publish-all"):
        base = 32768
        for i, ep in enumerate(sorted((img.config.get("ExposedPorts") or {}))):
            ports.append(("0.0.0.0", base + i, int(ep.split("/")[0]), "tcp"))
    used = {s for s in d.host.services}
    fixed = []
    for hip, hport, cport, proto in ports:
        if hport == 32768:
            hport = 32768
            while hport in used:
                hport += 1
            used.add(hport)
        fixed.append((hip, hport, cport, proto))
    cfg["ports"] = fixed
    mounts = []
    for v in opts.get("volume", []):
        mounts.append(parse_volume(ctx, v))
    for m in opts.get("mount", []):
        mounts.append(parse_mount(ctx, m))
    for t in opts.get("tmpfs", []):
        mounts.append({"type": "tmpfs", "dest": t.split(":")[0]})
    cfg["mounts"] = mounts
    if opts.get("gpus"):
        g = opts["gpus"].strip("'\"")
        if g == "all":
            cfg["gpus"] = "all"
        else:
            m = re.match(r'^"?device=([\d,]+)"?$', g)
            if m:
                cfg["gpus"] = [int(x) for x in m.group(1).split(",")]
            elif g.isdigit():
                cfg["gpus"] = list(range(int(g)))
            else:
                raise DockerError(f"docker: invalid argument \"{opts['gpus']}\" for \"--gpus\" flag: unexpected key "
                                  f"'{g.split('=')[0]}' in '{g}'.", 125)
    if opts.get("shm-size"):
        cfg["shm_mb"] = parse_size(opts["shm-size"]) // (1024 * 1024)
    if opts.get("ipc") == "host":
        cfg["shm_mb"] = 16384
    if opts.get("memory"):
        cfg["memory"] = parse_size(opts["memory"])
    if opts.get("cpus"):
        cfg["cpus"] = float(opts["cpus"])
    if opts.get("restart"):
        r = opts["restart"]
        if r.split(":")[0] not in ("no", "always", "unless-stopped", "on-failure"):
            raise DockerError(f"docker: invalid restart policy '{r}'.", 125)
        if r != "no" and cfg["rm"]:
            raise DockerError("docker: Conflicting options: --restart and --rm.\nSee 'docker run --help'.", 125)
        cfg["restart"] = r
    if opts.get("init"):
        cfg["init"] = True
    if opts.get("stop-timeout"):
        cfg["stop_timeout"] = int(opts["stop-timeout"])
    if opts.get("stop-signal"):
        cfg["stop_signal"] = opts["stop-signal"]
    labels = {}
    for lab in opts.get("label", []):
        k, _, v = lab.partition("=")
        labels[k] = v
    cfg["labels"] = labels
    if opts.get("health-cmd"):
        cfg["healthcheck"] = {"Test": ["CMD-SHELL", opts["health-cmd"]],
                              "Interval": int(_dur(opts.get("health-interval", "30s")) * NS),
                              "Retries": int(opts.get("health-retries", 3))}
    return cfg


def _dur(s):
    m = re.fullmatch(r"(\d+(?:\.\d+)?)(ms|s|m|h)?", str(s))
    if not m:
        return 30
    return float(m.group(1)) * {"ms": 0.001, "s": 1, "m": 60, "h": 3600, None: 1}[m.group(2)]


def ensure_image(ctx, o, d, ref, quiet=False, platform=None, pull="missing"):
    """The image for a reference, pulling it first when it isn't here ("Unable to find image ...")."""
    from .cmd_images import pull_image, normalize
    img = d.find_image(ref)
    if img is not None and pull != "always":
        return img
    if pull == "never":
        raise DockerError(f"docker: Error response from daemon: No such image: {normalize(ref)}.", 125)
    if img is None and not quiet:
        o.err(f"Unable to find image '{normalize(ref)}' locally\n")
    img = yield from pull_image(ctx, o, d, ref, quiet=quiet, platform=platform, from_run=True)
    return img


@cmd("run", "create")
def docker_run(ctx, o, args):
    create_only = ctx.argv[1] == "create" or (len(ctx.argv) > 2 and ctx.argv[2] == "create")
    opts, rest = flags(args, RUN_FLAGS)
    d = daemon_of(ctx.world, ctx.host)
    if not rest:
        raise DockerError(f"\"docker {'create' if create_only else 'run'}\" requires at least 1 argument.\nSee 'docker "
                          f"{'create' if create_only else 'run'} --help'.\n\nUsage:  docker run [OPTIONS] IMAGE "
                          f"[COMMAND] [ARG...]\n\nCreate and run a new container from an image", 125)
    image_ref, command = rest[0], rest[1:]
    img = yield from ensure_image(ctx, o, d, image_ref, quiet=bool(opts.get("quiet")), platform=opts.get("platform"),
                                  pull=opts.get("pull", "missing"))
    if img.arch != "amd64":
        o.err(f"WARNING: The requested image's platform (linux/{img.arch}) does not match the detected host platform "
              f"(linux/amd64/v3) and no specific platform was requested\n")
    cfg = container_config(ctx, d, opts, img, command, image_ref)
    c = d.create(img, cfg)
    if create_only:
        ctx.w(c.id + "\n")
        return 0
    detach = bool(opts.get("detach"))
    attach = None
    if not detach:
        attach = {1: ctx.proc.fds.get(1), 2: ctx.proc.fds.get(2), "pgid": ctx.proc.pgid}
        if opts.get("interactive"):
            attach[0] = ctx.proc.fds.get(0)
        if opts.get("tty") and not ctx.stdin_is_tty():
            d.remove(c, force=True)
            raise DockerError("the input device is not a TTY", 1)
    try:
        yield from ctx.flush()
        p = d.start(c, attach)
    except DockerError as e:
        code = e.status
        if c.id in d.containers:
            c.state = "created"
            c.exit_code = code
            if cfg.get("rm"):
                d.remove(c, force=True)
        raise DockerError(e.msg, code)
    if detach:
        ctx.w(c.id + "\n")
        return 0
    status = yield from wait_attached(ctx, d, c, p)
    return status


def wait_attached(ctx, d, c, p):
    """Wait for an attached container (docker run without -d): its exit status, or ^C after the session's timeout
    (the interrupt goes to the container, as docker run proxies signals)."""
    cli = ctx.proc

    def forward(proc, c=c):
        if c.running and c.proc is not None:
            c.stopping = True
            d.world.kernel.kill_now(c.proc, 2)
    cli.on_exit.append(forward)
    fg = ctx.foreground() and not c.cfg.get("interactive")    # an interactive session ends when you leave it
    deadline = ctx.now + ctx.world.ctrl_c_after * NS if fg else None
    while p.state != "Z":
        if deadline is not None and ctx.now >= deadline:
            ctx.world.tty.write("echo", b"^C")
            c.stopping = True
            d.world.kernel.kill_now(p, 2)
            break
        step = ctx.now + int(0.25 * NS)
        if deadline is not None and deadline != float("inf"):
            step = min(step, int(deadline))
        yield ("sleep", step)
    if forward in cli.on_exit:
        cli.on_exit.remove(forward)
    status = c.exit_code
    return status


# ---------------------------------------------------------------------------- start / stop / kill / rm

@cmd("start")
def docker_start(ctx, o, args):
    opts, rest = flags(args, {"-a|--attach": None, "-i|--interactive": None}, stop_at_positional=False)
    d = daemon_of(ctx.world, ctx.host)
    if not rest:
        raise DockerError("\"docker start\" requires at least 1 argument.", 1)
    st = 0
    for ref in rest:
        c = d.find_container(ref)
        if c is None:
            o.err(f"Error response from daemon: No such container: {ref}\nError: failed to start containers: {ref}\n")
            st = 1
            continue
        if c.running:
            if not opts.get("attach"):
                ctx.w(ref + "\n")
            continue
        attach = None
        if opts.get("attach"):
            attach = {1: ctx.proc.fds.get(1), 2: ctx.proc.fds.get(2), "pgid": ctx.proc.pgid}
            if opts.get("interactive"):
                attach[0] = ctx.proc.fds.get(0)
        try:
            p = d.start(c, attach)
        except DockerError as e:
            o.err(e.msg.replace("docker: ", "") + f"\nError: failed to start containers: {ref}\n")
            st = 1
            continue
        if attach:
            return (yield from wait_attached(ctx, d, c, p))
        ctx.w(ref + "\n")
    return st


@cmd("stop", "restart")
def docker_stop(ctx, o, args):
    restart = ctx.argv[1] == "restart" or (len(ctx.argv) > 2 and ctx.argv[2] == "restart")
    opts, rest = flags(args, {"-t|--time|--timeout": "v", "-s|--signal": "v"}, stop_at_positional=False)
    d = daemon_of(ctx.world, ctx.host)
    if not rest:
        raise DockerError(f"\"docker {'restart' if restart else 'stop'}\" requires at least 1 argument.", 1)
    st = 0
    for ref in rest:
        c = d.find_container(ref)
        if c is None:
            o.err(f"Error response from daemon: No such container: {ref}\n")
            st = 1
            continue
        t = int(opts["time"]) if opts.get("time") else None
        if c.running:
            yield from d.stop(c, t, opts.get("signal"))
        if restart:
            c.stopping = False
            try:
                d.start(c)
                c.restart_count += 0
            except DockerError as e:
                o.err(e.msg + "\n")
                st = 1
                continue
        ctx.w(ref + "\n")
    return st


SIGNALS = {"SIGKILL": 9, "KILL": 9, "9": 9, "SIGTERM": 15, "TERM": 15, "15": 15, "SIGINT": 2, "INT": 2, "2": 2,
           "SIGHUP": 1, "HUP": 1, "1": 1, "SIGUSR1": 10, "USR1": 10, "SIGQUIT": 3, "QUIT": 3}


@cmd("kill")
def docker_kill(ctx, o, args):
    opts, rest = flags(args, {"-s|--signal": "v"}, stop_at_positional=False)
    d = daemon_of(ctx.world, ctx.host)
    sig = SIGNALS.get(str(opts.get("signal", "KILL")).upper(), 9)
    st = 0
    for ref in rest:
        c = d.find_container(ref)
        if c is None:
            o.err(f"Error response from daemon: No such container: {ref}\n")
            st = 1
            continue
        if not c.running:
            o.err(f"Error response from daemon: cannot kill container: {ref}: container {c.id} is not running\n")
            st = 1
            continue
        if sig in (1, 10):
            ctx.w(ref + "\n")
            continue
        d.kill(c, sig)
        ctx.w(ref + "\n")
    return st


@cmd("rm")
def docker_rm(ctx, o, args):
    opts, rest = flags(args, {"-f|--force": None, "-v|--volumes": None, "-l|--link": None}, stop_at_positional=False)
    d = daemon_of(ctx.world, ctx.host)
    if not rest:
        raise DockerError("\"docker rm\" requires at least 1 argument.\nSee 'docker rm --help'.\n\nUsage:  docker rm "
                          "[OPTIONS] CONTAINER [CONTAINER...]\n\nRemove one or more containers", 1)
    st = 0
    for ref in rest:
        c = d.find_container(ref)
        if c is None:
            o.err(f"Error response from daemon: No such container: {ref}\n")
            st = 1
            continue
        try:
            d.remove(c, force=bool(opts.get("force")), volumes=bool(opts.get("volumes")))
        except DockerError as e:
            o.err(e.msg + "\n")
            st = 1
            continue
        ctx.w(ref + "\n")
    return st
    yield


@cmd("pause", "unpause")
def docker_pause(ctx, o, args):
    d = daemon_of(ctx.world, ctx.host)
    un = "unpause" in ctx.argv[1:3]
    for ref in args:
        c = d.get_container(ref)
        if not c.running:
            raise DockerError(f"Error response from daemon: container {c.id} is not running", 1)
        c.state = "running" if un else "paused"
        ctx.w(ref + "\n")
    return 0
    yield


@cmd("rename")
def docker_rename(ctx, o, args):
    d = daemon_of(ctx.world, ctx.host)
    if len(args) != 2:
        raise DockerError("\"docker rename\" requires exactly 2 arguments.", 1)
    c = d.get_container(args[0])
    if d.find_container(args[1]) is not None:
        raise DockerError(f'Error response from daemon: Conflict. The container name "/{args[1]}" is already in use.',
                          1)
    c.name = args[1]
    return 0
    yield


@cmd("wait")
def docker_wait(ctx, o, args):
    d = daemon_of(ctx.world, ctx.host)
    for ref in args:
        c = d.get_container(ref)
        while c.running:
            yield ("sleep", ctx.now + int(0.25 * NS))
        ctx.w(f"{c.exit_code}\n")
    return 0


# ---------------------------------------------------------------------------- ps

def short_cmd(c, trunc=True):
    ep = c.cfg.get("entrypoint")
    if ep is None:
        ep = c.image.config.get("Entrypoint") or []
    cmd_ = c.cfg.get("cmd")
    if cmd_ is None:
        cmd_ = c.image.config.get("Cmd") or []
    s = " ".join(list(ep) + list(cmd_))
    if trunc and len(s) > 20:
        s = s[:19] + "…"
    return f'"{s}"'


def ports_text(c):
    out = []
    for hip, hport, cport, proto in c.ports:
        if c.running:
            out.append(f"{hip}:{hport}->{cport}/{proto}")
            if hip == "0.0.0.0":
                out.append(f"[::]:{hport}->{cport}/{proto}")
    exposed = [p for p in (c.image.config.get("ExposedPorts") or {})
               if not any(f"{cp}/{pr}" == p for _, _, cp, pr in c.ports)]
    if c.running:
        out += exposed
    return ", ".join(out)


def ps_row(d, c, now, trunc=True):
    ref = c.cfg.get("image_ref") or (c.image.repo_tags[0] if c.image.repo_tags else c.image.short)
    return {"ID": c.short if trunc else c.id, "Image": ref, "Command": short_cmd(c, trunc),
            "CreatedAt": _ts(c.created), "RunningFor": f"{ago(now - c.created)} ago", "Ports": ports_text(c),
            "State": c.state, "Status": c.status_text(now), "Names": c.name,
            "Labels": ",".join(f"{k}={v}" for k, v in sorted(c.labels.items())),
            "Mounts": ",".join(m.get("name") or m["source"] for m in c.mounts if m["type"] != "tmpfs"),
            "Networks": ",".join(c.networks), "Size": "0B"}


def _ts(ns):
    import datetime
    t = datetime.datetime.fromtimestamp(ns / NS, tz=datetime.timezone.utc)
    return t.strftime("%Y-%m-%d %H:%M:%S +0000 UTC")


def _match_filter(d, c, f):
    k, _, v = f.partition("=")
    if k == "status":
        return c.state == v
    if k == "name":
        return re.search(v, c.name) is not None
    if k == "ancestor":
        img = d.find_image(v)
        return img is not None and c.image is img
    if k == "label":
        lk, _, lv = v.partition("=")
        return lk in c.labels and (not lv or c.labels[lk] == lv)
    if k == "exited":
        return c.state == "exited" and str(c.exit_code) == v
    if k == "network":
        return v in c.networks
    if k == "id":
        return c.id.startswith(v)
    if k == "volume":
        return any(m.get("name") == v or m["dest"] == v for m in c.mounts)
    if k == "health":
        return (c.health or "none") == v
    raise DockerError(f"Error response from daemon: invalid filter '{k}'", 1)


PS_HEAD = {"ID": "CONTAINER ID", "Image": "IMAGE", "Command": "COMMAND", "RunningFor": "CREATED", "Status": "STATUS",
           "Ports": "PORTS", "Names": "NAMES", "State": "STATE", "CreatedAt": "CREATED AT", "Labels": "LABELS",
           "Mounts": "MOUNTS", "Networks": "NETWORKS", "Size": "SIZE"}


@cmd("ps")
def docker_ps(ctx, o, args):
    opts, rest = flags(args, {"-a|--all": None, "-q|--quiet": None, "-n|--last": "v", "-l|--latest": None,
                              "-f|--filter": "m", "--format": "v", "--no-trunc": None, "-s|--size": None},
                       stop_at_positional=False)
    d = daemon_of(ctx.world, ctx.host)
    now = ctx.now
    cs = sorted(d.containers.values(), key=lambda c: -c.created)
    if not opts.get("all") and not opts.get("last") and not opts.get("latest"):
        cs = [c for c in cs if c.running]
    for f in opts.get("filter", []):
        cs = [c for c in cs if _match_filter(d, c, f)]
    if opts.get("latest"):
        cs = cs[:1]
    if opts.get("last"):
        cs = cs[:int(opts["last"])]
    trunc = not opts.get("no-trunc")
    if opts.get("quiet"):
        ctx.w("".join((c.short if trunc else c.id) + "\n" for c in cs))
        return 0
    fmt = opts.get("format")
    rows = [ps_row(d, c, now, trunc) for c in cs]
    if fmt and fmt != "table":
        if fmt == "json":
            ctx.w("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows))
        elif fmt.startswith("table"):
            ctx.w(gotmpl.table(fmt, rows, PS_HEAD))
        else:
            ctx.w("".join(gotmpl.render(fmt.replace("\\t", "\t").replace("\\n", "\n"), r) + "\n" for r in rows))
        return 0
    table = [["CONTAINER ID", "IMAGE", "COMMAND", "CREATED", "STATUS", "PORTS", "NAMES"]]
    for r in rows:
        table.append([r["ID"], r["Image"], r["Command"], r["RunningFor"], r["Status"], r["Ports"], r["Names"]])
    ctx.w(gotmpl.tabulate(table))
    return 0
    yield


# ---------------------------------------------------------------------------- logs

@cmd("logs")
def docker_logs(ctx, o, args):
    opts, rest = flags(args, {"-f|--follow": None, "-n|--tail": "v", "-t|--timestamps": None, "--since": "v",
                              "--until": "v", "--details": None}, stop_at_positional=False)
    d = daemon_of(ctx.world, ctx.host)
    if len(rest) != 1:
        raise DockerError("\"docker logs\" requires exactly 1 argument.\nSee 'docker logs --help'.\n\nUsage:  docker "
                          "logs [OPTIONS] CONTAINER\n\nFetch the logs of a container", 1)
    c = d.get_container(rest[0])
    since = _since(ctx, opts.get("since"))
    lines = _log_lines(c, since)
    tail = opts.get("tail", "all")
    if tail != "all":
        n = int(tail)
        lines = lines[-n:] if n > 0 else []
    for t, stream, text in lines:
        _emit(ctx, o, t, stream, text, opts.get("timestamps"))
    if not opts.get("follow"):
        return 0
    seen = len(c.logs)
    fg = ctx.foreground()
    deadline = ctx.now + ctx.world.ctrl_c_after * NS if fg else float("inf")
    while True:
        new = c.logs[seen:]
        seen = len(c.logs)
        for t, stream, data in _split_entries(new):
            _emit(ctx, o, t, stream, data, opts.get("timestamps"))
        yield from ctx.flush()
        if not c.running and c.state != "restarting":
            return 0
        if ctx.now >= deadline:
            ctx.world.tty.write("echo", b"^C\n")
            return 130
        nxt = ctx.now + int(0.25 * NS)
        if deadline != float("inf"):
            nxt = min(nxt, int(deadline))
        yield ("sleep", nxt)


def _since(ctx, s):
    if not s:
        return None
    m = re.fullmatch(r"(\d+)([smh])", s)
    if m:
        return ctx.now - int(m.group(1)) * {"s": 1, "m": 60, "h": 3600}[m.group(2)] * NS
    return None


def _split_entries(entries):
    out = []
    buf = {"out": b"", "err": b""}
    times = {}
    for t, stream, data in entries:
        if not buf[stream]:
            times[stream] = t
        buf[stream] += data
        while b"\n" in buf[stream]:
            line, _, rest = buf[stream].partition(b"\n")
            out.append((times[stream], stream, line + b"\n"))
            buf[stream] = rest
            times[stream] = t
    for s in ("out", "err"):
        if buf[s]:
            out.append((times[s], s, buf[s]))
    return out


def _log_lines(c, since=None):
    lines = _split_entries(c.logs)
    if since is not None:
        lines = [x for x in lines if x[0] >= since]
    return lines


def _emit(ctx, o, t, stream, data, stamps):
    text = data.decode("utf-8", "replace")
    if stamps:
        import datetime
        ts = datetime.datetime.fromtimestamp(t / NS, tz=datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")
        text = f"{ts}000Z {text}"
    if stream == "out":
        ctx.w(text)
    else:
        ctx._flush_nb()
        ctx.e(text)


# ---------------------------------------------------------------------------- exec

@cmd("exec")
def docker_exec(ctx, o, args):
    opts, rest = flags(args, {"-i|--interactive": None, "-t|--tty": None, "-d|--detach": None, "-u|--user": "v",
                              "-w|--workdir": "v", "-e|--env": "m", "--privileged": None})
    d = daemon_of(ctx.world, ctx.host)
    if len(rest) < 2:
        raise DockerError("\"docker exec\" requires at least 2 arguments.\nSee 'docker exec --help'.\n\nUsage:  docker "
                          "exec [OPTIONS] CONTAINER COMMAND [ARG...]\n\nExecute a command in a running container", 1)
    c = d.get_container(rest[0])
    if c.state == "paused":
        raise DockerError(f"Error response from daemon: Container {c.id} is paused, unpause the container before exec",
                          1)
    if not c.running:
        raise DockerError(f"Error response from daemon: container {c.id} is not running", 1)
    if opts.get("tty") and not ctx.stdin_is_tty():
        raise DockerError("the input device is not a TTY", 1)
    argv = rest[1:]
    from ..exec import exec_program, find_in_path
    h = c.host
    env = dict(c.proc.env) if c.proc is not None else {}
    env = {k: v for k, v in env.items() if not k.startswith("_")}
    for e in opts.get("env", []):
        k, _, v = e.partition("=")
        env[k] = v
    if opts.get("tty"):
        env["TERM"] = "xterm"
    user = opts.get("user") or c.cfg.get("user") or c.image.config.get("User") or "root"
    uid, gid = d.user_ids(h, user)
    env["HOME"] = h.users[uid].home if uid in h.users else "/"
    workdir = opts.get("workdir") or c.cfg.get("workdir") or c.image.config.get("WorkingDir") or "/"
    path = find_in_path(h.fs, argv[0], env.get("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:"
                                                          "/bin"), workdir)
    if path is None or not h.fs.exists(path):
        o.err(f"OCI runtime exec failed: exec failed: unable to start container process: exec: \"{argv[0]}\": "
              f"executable file not found in $PATH: unknown\n")
        return 126 if "/" not in argv[0] else 127
    fds = {1: ctx.proc.fds.get(1), 2: ctx.proc.fds.get(2)}
    if opts.get("interactive"):
        fds[0] = ctx.proc.fds.get(0)
    else:
        from ..kernel import OpenFile
        fds[0] = OpenFile("null", rd=True)
    if opts.get("detach"):
        from ..kernel import OpenFile
        null = OpenFile("null", wr=True)
        fds = {0: OpenFile("null", rd=True), 1: null, 2: null}
    yield from ctx.flush()
    k = ctx.k
    p = k.spawn(h, argv, None, None, fds={n: f for n, f in fds.items() if f is not None}, cwd=workdir, env=env,
                uid=uid, gid=gid)
    p.pgid = ctx.proc.pgid
    k.start(p, exec_program(p, path, argv, env, None))
    if opts.get("detach"):
        p.daemon = True
        return 0
    yield from ctx.wait([p])
    return p.status


# ---------------------------------------------------------------------------- cp

@cmd("cp")
def docker_cp(ctx, o, args):
    opts, rest = flags(args, {"-a|--archive": None, "-L|--follow-link": None, "-q|--quiet": None},
                       stop_at_positional=False)
    d = daemon_of(ctx.world, ctx.host)
    if len(rest) != 2:
        raise DockerError("\"docker cp\" requires exactly 2 arguments.\nSee 'docker cp --help'.\n\nUsage:  docker cp "
                          "[OPTIONS] CONTAINER:SRC_PATH DEST_PATH|-\n\tdocker cp [OPTIONS] SRC_PATH|- "
                          "CONTAINER:DEST_PATH\n\nCopy files/folders between a container and the local filesystem", 1)
    src, dst = rest

    def split(x):
        m = re.match(r"^([a-zA-Z0-9][a-zA-Z0-9_.-]*):(.*)$", x)
        if m and not x.startswith("/"):
            return d.get_container(m.group(1)), m.group(2)
        return None, x
    sc, sp = split(src)
    dc, dp = split(dst)
    if (sc is None) == (dc is None):
        raise DockerError("must specify at least one container source" if sc is None else
                          "copying between containers is not supported", 1)
    if sc is not None:
        sfs, spath = sc.host.fs, sp if sp.startswith("/") else "/" + sp
        dfs, dpath = ctx.fs, host_path(ctx, dp)
    else:
        sfs, spath = ctx.fs, host_path(ctx, sp)
        dfs, dpath = dc.host.fs, dp if dp.startswith("/") else (dc.cfg.get("workdir") or "/") + "/" + dp
    try:
        node = sfs.lookup(spath)
    except FSError:
        where = f"{sc.name}:" if sc else ""
        raise DockerError(f"Error response from daemon: Could not find the file {spath} in container {sc.name}"
                          if sc else f"lstat {spath}: no such file or directory", 1)
    target = dpath
    try:
        tnode = dfs.lookup(dpath)
        if tnode.type == "d":
            target = dpath.rstrip("/") + "/" + posixpath.basename(spath.rstrip("/"))
    except FSError:
        if dpath.endswith("/"):
            raise DockerError(f"Error response from daemon: Could not find the file {dpath}", 1)
    uid = ctx.uid if sc is not None else 0
    gid = ctx.proc.gid if sc is not None else 0
    try:
        if node.type == "d":
            sfs.copy_tree(node, dfs, target, uid, gid)
        else:
            new = dfs.create(target, "/", node.mode & 0o777, uid, gid)
            new.data = bytearray(node.data) if not hasattr(node.data, "clone") else node.data.clone()
            new.mode = node.mode
    except FSError as e:
        raise DockerError(f"Error response from daemon: {e.msg}", 1)
    if not opts.get("quiet") and ctx.stdin_is_tty():
        size = sfs.tree_size(node)
        ctx.e(f"Successfully copied {human_size(size)} to {dst}\n")
    return 0
    yield


# ---------------------------------------------------------------------------- inspect

def container_json(d, c):
    img = c.image
    env = list(img.config.get("Env") or [])
    for e in c.cfg.get("env", []):
        k = e.split("=", 1)[0]
        env = [x for x in env if not x.startswith(k + "=")] + [e]
    ep = c.cfg.get("entrypoint", img.config.get("Entrypoint"))
    cmd_ = c.cfg.get("cmd", img.config.get("Cmd") if not c.cfg.get("entrypoint_set") else [])
    argv = list(ep or []) + list(cmd_ or [])
    nets = {}
    for n, ip in c.networks.items():
        net = d.networks.get(n)
        nets[n] = {"IPAddress": ip, "Gateway": net.gateway if net else "", "Aliases": list(c.cfg.get("aliases", [])) or None,
                   "MacAddress": "02:42:ac:11:00:02", "NetworkID": net.id if net else ""}
    ports = {}
    for p in (img.config.get("ExposedPorts") or {}):
        ports[p] = None
    for hip, hport, cport, proto in c.ports:
        ports.setdefault(f"{cport}/{proto}", [])
        if c.running:
            ports[f"{cport}/{proto}"] = (ports[f"{cport}/{proto}"] or []) + [{"HostIp": hip, "HostPort": str(hport)}]
    restart = c.cfg.get("restart", "no")
    health = None
    if c.health is not None:
        fails = 0
        for _, ok in reversed(c.health_log):
            if ok:
                break
            fails += 1
        health = {"Status": c.health, "FailingStreak": fails,
                  "Log": [{"ExitCode": 0 if ok else 1} for _, ok in c.health_log[-5:]]}
    st = {"Status": c.state, "Running": c.running, "Paused": c.state == "paused",
          "Restarting": c.state == "restarting", "OOMKilled": c.oom, "Dead": False,
          "Pid": c.proc.pid if c.proc is not None else 0, "ExitCode": c.exit_code, "Error": c.error,
          "StartedAt": _iso(c.started), "FinishedAt": _iso(c.finished)}
    if health:
        st["Health"] = health
    return {
        "Id": c.id, "Created": _iso(c.created), "Path": argv[0] if argv else "", "Args": argv[1:],
        "State": st, "Image": img.id, "Name": "/" + c.name, "RestartCount": c.restart_count,
        "Platform": "linux",
        "HostConfig": {"Binds": [f"{m['source']}:{m['dest']}" + (":ro" if m.get("ro") else "") for m in c.mounts
                                 if m["type"] == "bind"] or None,
                       "NetworkMode": c.cfg.get("network") or "bridge",
                       "PortBindings": {f"{cp}/{pr}": [{"HostIp": "" if hip == "0.0.0.0" else hip,
                                                         "HostPort": str(hp)}] for hip, hp, cp, pr in c.ports},
                       "RestartPolicy": {"Name": restart.split(":")[0],
                                         "MaximumRetryCount": int(restart.split(":")[1]) if ":" in restart else 0},
                       "AutoRemove": bool(c.cfg.get("rm")), "Memory": c.cfg.get("memory", 0),
                       "NanoCpus": int(c.cfg.get("cpus", 0) * 1e9), "ShmSize": c.cfg.get("shm_mb", 64) * 1024 * 1024,
                       "Init": c.cfg.get("init"),
                       "DeviceRequests": None if c.cfg.get("gpus") is None else [
                           {"Driver": "", "Count": -1 if c.cfg["gpus"] == "all" else 0,
                            "DeviceIDs": None if c.cfg["gpus"] == "all" else [str(x) for x in c.cfg["gpus"]],
                            "Capabilities": [["gpu"]]}],
                       "ExtraHosts": [f"{n}:{ip}" for n, ip in c.cfg.get("add_hosts", [])] or None},
        "Mounts": [{"Type": m["type"], **({"Name": m["name"]} if m.get("name") else {}), "Source": m.get("source", ""),
                    "Destination": m["dest"], "Driver": "local" if m["type"] == "volume" else "",
                    "Mode": "ro" if m.get("ro") else ("z" if m["type"] == "volume" else ""),
                    "RW": not m.get("ro"), "Propagation": "" if m["type"] == "volume" else "rprivate"}
                   for m in c.mounts],
        "Config": {"Hostname": c.host.name if c.host else c.short, "User": c.cfg.get("user", img.config.get("User", "")),
                   "Tty": bool(c.cfg.get("tty")), "OpenStdin": bool(c.cfg.get("interactive")), "Env": env,
                   "Cmd": cmd_, "Image": c.cfg.get("image_ref"), "Volumes": img.config.get("Volumes") or None,
                   "WorkingDir": c.cfg.get("workdir", img.config.get("WorkingDir", "")), "Entrypoint": ep,
                   "Labels": {**(img.config.get("Labels") or {}), **c.labels},
                   "ExposedPorts": img.config.get("ExposedPorts") or None,
                   **({"Healthcheck": img.config.get("Healthcheck") or c.cfg.get("healthcheck")}
                      if (img.config.get("Healthcheck") or c.cfg.get("healthcheck")) else {}),
                   **({"StopSignal": img.config["StopSignal"]} if img.config.get("StopSignal") else {})},
        "NetworkSettings": {"IPAddress": c.networks.get("bridge", ""), "Ports": ports, "Networks": nets,
                            "Gateway": d.networks["bridge"].gateway if "bridge" in c.networks else ""},
    }


def _iso(ns):
    import datetime
    if not ns:
        return "0001-01-01T00:00:00Z"
    t = datetime.datetime.fromtimestamp(ns / NS, tz=datetime.timezone.utc)
    return t.strftime("%Y-%m-%dT%H:%M:%S.%f") + "000Z"


@cmd("inspect")
def docker_inspect(ctx, o, args):
    from .cmd_images import image_json
    from .cmd_misc import network_json, volume_json
    opts, rest = flags(args, {"-f|--format": "v", "--type": "v", "-s|--size": None}, stop_at_positional=False)
    d = daemon_of(ctx.world, ctx.host)
    if not rest:
        raise DockerError("\"docker inspect\" requires at least 1 argument.", 1)
    objs, st = [], 0
    t = opts.get("type")
    for ref in rest:
        obj = None
        if t in (None, "container"):
            c = d.find_container(ref)
            if c is not None:
                obj = container_json(d, c)
        if obj is None and t in (None, "image"):
            img = d.find_image(ref)
            if img is not None:
                obj = image_json(d, img)
        if obj is None and t in (None, "volume") and ref in d.volumes:
            obj = volume_json(d, d.volumes[ref])
        if obj is None and t in (None, "network") and d.network(ref) is not None:
            obj = network_json(d, d.network(ref))
        if obj is None:
            o.err(f"Error: No such object: {ref}\n")
            st = 1
            continue
        objs.append(obj)
    fmt = opts.get("format")
    if fmt:
        for ob in objs:
            try:
                ctx.w(gotmpl.render(fmt, ob) + "\n")
            except gotmpl.TemplateError as e:
                raise DockerError(f"template parsing error: {e}", 1)
        return st
    if objs:
        ctx.w(json.dumps(objs, indent=4) + "\n")
    elif st:
        ctx.w("[]\n")
    return st
    yield


# ---------------------------------------------------------------------------- top / port / stats

@cmd("top")
def docker_top(ctx, o, args):
    d = daemon_of(ctx.world, ctx.host)
    if not args:
        raise DockerError("\"docker top\" requires at least 1 argument.", 1)
    c = d.get_container(args[0])
    if not c.running:
        raise DockerError(f"Error response from daemon: container {c.id} is not running", 1)
    rows = [["UID", "PID", "PPID", "C", "STIME", "TTY", "TIME", "CMD"]]
    base = 40000 + int(c.id[:4], 16) % 9000
    for p in sorted(c.host.procs.values(), key=lambda p: p.pid):
        if p.state == "Z":
            continue
        user = c.host.users[p.uid].name if p.uid in c.host.users and c.host.users[p.uid].name else str(p.uid)
        rows.append([user, str(base + p.pid), str(base + p.ppid if p.pid != 1 else base - 21), "0", "09:41", "?",
                     "00:00:00", p.cmdline])
    ctx.w(gotmpl.tabulate(rows, pad=3))
    return 0
    yield


@cmd("port")
def docker_port(ctx, o, args):
    d = daemon_of(ctx.world, ctx.host)
    if not args:
        raise DockerError("\"docker port\" requires at least 1 argument.", 1)
    c = d.get_container(args[0])
    want = args[1] if len(args) > 1 else None
    for hip, hport, cport, proto in c.ports:
        if not c.running:
            continue
        key = f"{cport}/{proto}"
        if want and want not in (key, str(cport)):
            continue
        if want:
            ctx.w(f"{hip}:{hport}\n" + (f"[::]:{hport}\n" if hip == "0.0.0.0" else ""))
        else:
            ctx.w(f"{key} -> {hip}:{hport}\n" + (f"{key} -> [::]:{hport}\n" if hip == "0.0.0.0" else ""))
    return 0
    yield


def _mem_used(c):
    base = {"nginx": 3.4, "postgres": 38.2, "python": 22.5, "uvicorn": 310.0, "start-notebook.py": 140.0,
            "sleep": 0.4, "bash": 1.9, "sh": 0.6}
    prog = posixpath.basename(c.proc.argv[0]) if c.proc else ""
    return base.get(prog, 6.1) * 1024 * 1024


@cmd("stats")
def docker_stats(ctx, o, args):
    opts, rest = flags(args, {"--no-stream": None, "-a|--all": None, "--format": "v", "--no-trunc": None},
                       stop_at_positional=False)
    d = daemon_of(ctx.world, ctx.host)
    cs = [d.get_container(r) for r in rest] if rest else [c for c in d.containers.values() if c.running or
                                                          opts.get("all")]
    rows = []
    total = d.host.mem_kib * 1024
    for c in sorted(cs, key=lambda c: c.created):
        used = _mem_used(c) if c.running else 0
        limit = c.cfg.get("memory") or total
        rows.append({"Container": c.short, "ID": c.short, "Name": c.name,
                     "CPUPerc": f"{(int(c.id[:2], 16) % 7) / 10 if c.running else 0:.2f}%",
                     "MemUsage": f"{_mib(used)} / {_mib(limit)}", "MemPerc": f"{100 * used / limit:.2f}%",
                     "NetIO": "1.2kB / 0B" if c.running else "0B / 0B", "BlockIO": "0B / 0B",
                     "PIDs": str(len([p for p in c.host.procs.values() if p.state != 'Z']) if c.running else 0)})
    fmt = opts.get("format")
    if fmt:
        out = gotmpl.table(fmt, rows, {"Container": "CONTAINER", "Name": "NAME", "CPUPerc": "CPU %",
                                       "MemUsage": "MEM USAGE / LIMIT", "MemPerc": "MEM %", "NetIO": "NET I/O",
                                       "BlockIO": "BLOCK I/O", "PIDs": "PIDS", "ID": "CONTAINER ID"}) \
            if fmt.startswith("table") else "".join(gotmpl.render(fmt.replace("\\t", "\t"), r) + "\n" for r in rows)
    else:
        t = [["CONTAINER ID", "NAME", "CPU %", "MEM USAGE / LIMIT", "MEM %", "NET I/O", "BLOCK I/O", "PIDS"]]
        for r in rows:
            t.append([r["ID"], r["Name"], r["CPUPerc"], r["MemUsage"], r["MemPerc"], r["NetIO"], r["BlockIO"], r["PIDs"]])
        out = gotmpl.tabulate(t)
    ctx.w(out)
    if opts.get("no-stream"):
        return 0
    yield from ctx.flush()
    if ctx.foreground():
        yield ("sleep", ctx.now + ctx.world.ctrl_c_after * NS)
        ctx.world.tty.write("echo", b"^C\n")
        return 130
    return 0


def _mib(n):
    if n >= 1024 ** 3:
        return f"{n / 1024 ** 3:.2f}GiB"
    return f"{n / 1024 ** 2:.2f}MiB" if n >= 1024 ** 2 else f"{n / 1024:.0f}KiB"


@cmd("container-prune")
def docker_container_prune(ctx, o, args):
    opts, rest = flags(args, {"-f|--force": None, "--filter": "m"}, stop_at_positional=False)
    d = daemon_of(ctx.world, ctx.host)
    if not opts.get("force"):
        yield from ctx.flush()
        ok = yield from confirm(ctx, "WARNING! This will remove all stopped containers.\nAre you sure you want to "
                                     "continue? [y/N] ")
        if not ok:
            return 0
    gone = [c for c in list(d.containers.values()) if not c.running]
    for c in gone:
        d.remove(c)
    if gone:
        ctx.w("Deleted Containers:\n" + "".join(c.id + "\n" for c in gone) + "\n")
    ctx.w("Total reclaimed space: " + (human_size(sum(1024 for _ in gone)) if gone else "0B") + "\n")
    return 0


def confirm(ctx, question):
    ctx.w(question)
    yield from ctx.flush()
    data = yield from _readline(ctx)
    return data.strip().lower() in ("y", "yes")


def _readline(ctx):
    r = ctx.reader("-")
    line = yield from r.readline()
    return (line or b"").decode("utf-8", "replace")
