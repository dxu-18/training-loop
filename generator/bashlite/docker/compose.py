"""docker compose (v2): compose.yaml (a YAML subset parser), variable interpolation from .env and the
environment, and up/down/ps/logs/exec/run/start/stop/restart/build/pull/config/images/ls.

A project is the directory's name; its containers are <project>-<service>-1, its default network
<project>_default (services find each other by service name) and its volumes <project>_<name>. depends_on with
condition: service_healthy waits for the dependency's health check."""

from __future__ import annotations

import json
import posixpath
import re
import shlex

from ..kernel import NS
from ..vfs import FSError
from . import gotmpl
from .cli import cmd, flags, host_path
from .engine import DockerError, ago, daemon_of


# ---------------------------------------------------------------------------- YAML (the part compose files use)

class YamlError(Exception):
    pass


def _strip_comment(line):
    out, q = [], None
    for i, ch in enumerate(line):
        if q:
            if ch == q:
                q = None
        elif ch in "'\"":
            q = ch
        elif ch == "#" and (i == 0 or line[i - 1] in " \t"):
            break
        out.append(ch)
    return "".join(out).rstrip()


def _scalar(s):
    s = s.strip()
    if s == "":
        return None
    if s[0] in "'\"" and s[-1] == s[0] and len(s) >= 2:
        return s[1:-1] if s[0] == "'" else json.loads(s)
    if s.startswith("[") or s.startswith("{"):
        return _flow(s)
    if s in ("null", "~"):
        return None
    if s in ("true", "True", "yes"):
        return True
    if s in ("false", "False", "no"):
        return False
    if re.fullmatch(r"-?\d+", s):
        return int(s)
    if re.fullmatch(r"-?\d+\.\d+", s):
        return float(s)
    return s


def _flow(s):
    s = s.strip()
    pos = [0]

    def ws():
        while pos[0] < len(s) and s[pos[0]] in " \t\n":
            pos[0] += 1

    def item():
        ws()
        ch = s[pos[0]]
        if ch == "[":
            pos[0] += 1
            out = []
            ws()
            if s[pos[0]] == "]":
                pos[0] += 1
                return out
            while True:
                out.append(item())
                ws()
                if s[pos[0]] == ",":
                    pos[0] += 1
                    continue
                if s[pos[0]] == "]":
                    pos[0] += 1
                    return out
                raise YamlError("bad flow sequence")
        if ch == "{":
            pos[0] += 1
            out = {}
            ws()
            if s[pos[0]] == "}":
                pos[0] += 1
                return out
            while True:
                k = item()
                ws()
                if s[pos[0]] == ":":
                    pos[0] += 1
                    v = item()
                else:
                    v = None
                out[str(k)] = v
                ws()
                if s[pos[0]] == ",":
                    pos[0] += 1
                    continue
                if s[pos[0]] == "}":
                    pos[0] += 1
                    return out
                raise YamlError("bad flow mapping")
        if ch in "'\"":
            end = pos[0] + 1
            while end < len(s) and s[end] != ch:
                end += 2 if s[end] == "\\" and ch == '"' else 1
            tok = s[pos[0]:end + 1]
            pos[0] = end + 1
            return _scalar(tok)
        m = re.match(r"[^,\]\}:]+(?::(?![ ,\]\}])[^,\]\}:]*)*", s[pos[0]:])
        tok = m.group(0) if m else ""
        pos[0] += len(tok)
        return _scalar(tok)
    v = item()
    return v


def yaml_load(text):
    lines = []
    for n, raw in enumerate(text.replace("\t", "  ").split("\n"), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if raw.strip() in ("---",):
            continue
        lines.append((len(raw) - len(raw.lstrip(" ")), _strip_comment(raw).strip(), n, raw))
    pos = [0]

    def block(indent):
        if pos[0] >= len(lines):
            return None
        ind, body, n, raw = lines[pos[0]]
        if body.startswith("- ") or body == "-":
            return seq(ind)
        return mapping(ind)

    def seq(indent):
        out = []
        while pos[0] < len(lines):
            ind, body, n, raw = lines[pos[0]]
            if ind < indent or not (body.startswith("- ") or body == "-"):
                if ind > indent:
                    raise YamlError(f"line {n}: did not find expected '-' indicator")
                break
            if ind > indent:
                raise YamlError(f"line {n}: bad indentation of a sequence entry")
            rest = body[1:].strip()
            pos[0] += 1
            if not rest:
                out.append(block(indent + 1) if pos[0] < len(lines) and lines[pos[0]][0] > indent else None)
            elif re.match(r"^[^'\"\[\{][^:]*:(\s|$)", rest) and not rest.startswith(("http:", "https:")):
                # a mapping inside a sequence item: its first key is on the dash line
                sub_indent = ind + 2
                lines.insert(pos[0], (sub_indent, rest, n, raw))
                out.append(mapping(sub_indent))
            else:
                out.append(_scalar(rest))
        return out

    def mapping(indent):
        out = {}
        while pos[0] < len(lines):
            ind, body, n, raw = lines[pos[0]]
            if ind < indent:
                break
            if ind > indent:
                raise YamlError(f"line {n}: mapping values are not allowed in this context")
            m = re.match(r"^(\"[^\"]*\"|'[^']*'|[^:]+?)\s*:(?:\s+(.*)|$)", body)
            if not m:
                if body.startswith("- "):
                    raise YamlError(f"line {n}: did not find expected key")
                raise YamlError(f"line {n}: could not find expected ':'")
            key = _scalar(m.group(1))
            key = str(key) if key is not None else ""
            val = m.group(2)
            pos[0] += 1
            if val is None or val == "":
                if pos[0] < len(lines) and lines[pos[0]][0] > indent:
                    out[key] = block(lines[pos[0]][0])
                elif pos[0] < len(lines) and lines[pos[0]][0] == indent and lines[pos[0]][1].startswith("- "):
                    out[key] = seq(indent)
                else:
                    out[key] = None
            elif val in ("|", ">", "|-", ">-"):
                parts = []
                while pos[0] < len(lines) and lines[pos[0]][0] > indent:
                    parts.append(lines[pos[0]][3].strip())
                    pos[0] += 1
                out[key] = ("\n" if val.startswith("|") else " ").join(parts) + ("\n" if not val.endswith("-") else "")
            else:
                out[key] = _scalar(val)
            if key in list(out)[:-1]:
                raise YamlError(f"line {n}: mapping key \"{key}\" already defined")
        return out
    if not lines:
        return {}
    return block(lines[0][0])


def yaml_dump(v, indent=0):
    pad = "  " * indent
    out = []
    if isinstance(v, dict):
        for k, x in v.items():
            if isinstance(x, (dict, list)) and x:
                out.append(f"{pad}{k}:")
                out.append(yaml_dump(x, indent + 1))
            else:
                out.append(f"{pad}{k}: {_dump_scalar(x)}")
    elif isinstance(v, list):
        for x in v:
            if isinstance(x, dict) and x:
                first = yaml_dump(x, indent + 1).split("\n")
                out.append(f"{pad}- " + first[0].lstrip())
                out += first[1:]
            else:
                out.append(f"{pad}- {_dump_scalar(x)}")
    return "\n".join(o for o in out if o != "")


def _dump_scalar(x):
    if x is None:
        return "{}"
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, (int, float)):
        return str(x)
    if isinstance(x, (dict, list)):
        return "{}" if isinstance(x, dict) else "[]"
    s = str(x)
    if re.search(r"[:#{}\[\],&*?|<>=!%@`'\"]|^\s|\s$|^[-\d]", s) or s in ("true", "false", "null", ""):
        return json.dumps(s)
    return s


# ---------------------------------------------------------------------------- the project

FILES = ("compose.yaml", "compose.yml", "docker-compose.yaml", "docker-compose.yml")
KEYS = {"image", "build", "ports", "environment", "env_file", "volumes", "depends_on", "healthcheck", "command",
        "entrypoint", "working_dir", "user", "restart", "networks", "deploy", "container_name", "profiles", "shm_size",
        "labels", "expose", "init", "stop_grace_period", "hostname", "extra_hosts", "tty", "stdin_open", "ipc",
        "gpus", "platform", "pull_policy", "develop", "secrets", "configs", "logging", "mem_limit", "cpus",
        "stop_signal", "network_mode", "read_only", "tmpfs", "devices", "cap_add", "privileged", "ulimits"}


def interpolate(text, env):
    def repl(m):
        if m.group(0) == "$$":
            return "$"
        name = m.group(1) or m.group(2)
        op, word = m.group(3), m.group(4)
        val = env.get(name)
        if op in (":-", "-"):
            return val if (val if op == ":-" else val is not None) else (word or "")
        if op in (":?", "?"):
            if not val:
                raise DockerError(f"error while interpolating: required variable {name} is missing a value: "
                                  f"{word or ''}".rstrip(": "), 1)
            return val
        if val is None:
            return ""
        return val
    return re.sub(r"\$\$|\$(?:(\w+)|\{(\w+)(?:(:?[-?])([^}]*))?\})", repl, text)


class Project:
    def __init__(self, ctx, d, files, name=None, profiles=()):
        self.ctx, self.d = ctx, d
        self.files = files
        self.dir = posixpath.dirname(files[0])
        env = dict(ctx.env)
        envfile = self.dir + "/.env"
        self.dotenv = {}
        try:
            for line in bytes(ctx.fs.lookup(envfile).data).decode("utf-8", "replace").split("\n"):
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    self.dotenv[k.strip()] = v.strip().strip('"\'')
        except FSError:
            pass
        for k, v in self.dotenv.items():
            env.setdefault(k, v)
        merged = {}
        for f in files:
            try:
                text = bytes(ctx.fs.lookup(f).data).decode("utf-8", "replace")
            except FSError:
                raise DockerError(f"open {f}: no such file or directory", 1)
            try:
                data = yaml_load(interpolate(text, env)) or {}
            except YamlError as e:
                raise DockerError(f"yaml: {e}", 1)
            if not isinstance(data, dict):
                raise DockerError(f"validating {f}: top-level object must be a mapping", 1)
            merged = _merge(merged, data)
        self.data = merged
        raw_name = name or ctx.env.get("COMPOSE_PROJECT_NAME") or merged.get("name") or posixpath.basename(self.dir)
        self.name = re.sub(r"[^a-z0-9_-]", "", str(raw_name).lower()) or "default"
        self.services = merged.get("services") or {}
        if not isinstance(self.services, dict):
            raise DockerError(f"validating {files[0]}: services must be a mapping", 1)
        for sname, svc in self.services.items():
            if not isinstance(svc, dict):
                raise DockerError(f"validating {files[0]}: services.{sname} must be a mapping", 1)
            for k in svc:
                if k not in KEYS:
                    raise DockerError(f"validating {files[0]}: services.{sname} additional properties '{k}' not "
                                      f"allowed", 15)
            if "image" not in svc and "build" not in svc:
                raise DockerError(f"service \"{sname}\" has neither an image nor a build context specified: invalid "
                                  f"compose project", 1)
            if "ports" in svc and not isinstance(svc["ports"], list):
                raise DockerError(f"validating {files[0]}: services.{sname}.ports must be a array", 15)
            for dep in _deps(svc):
                if dep not in self.services:
                    raise DockerError(f"service \"{sname}\" depends on undefined service \"{dep}\": invalid compose "
                                      f"project", 1)
        self.active = [s for s, v in self.services.items() if not v.get("profiles") or
                       set(v.get("profiles") or []) & set(profiles)]
        self.volumes = merged.get("volumes") or {}
        self.networks = merged.get("networks") or {}

    def container_name(self, s):
        return self.services[s].get("container_name") or f"{self.name}-{s}-1"

    def image_name(self, s):
        svc = self.services[s]
        return svc.get("image") or f"{self.name}-{s}"

    def containers(self):
        return [c for c in self.d.containers.values() if c.labels.get("com.docker.compose.project") == self.name]

    def find(self, s):
        name = self.container_name(s)
        return self.d.find_container(name)

    def order(self, services):
        """Dependencies first."""
        out, seen = [], set()

        def visit(s, stack=()):
            if s in seen:
                return
            if s in stack:
                raise DockerError(f"dependency cycle detected: {' -> '.join(stack + (s,))}", 1)
            for dep in _deps(self.services[s]):
                visit(dep, stack + (s,))
            seen.add(s)
            out.append(s)
        for s in services:
            visit(s)
        return out


def _deps(svc):
    dep = svc.get("depends_on") or []
    return list(dep) if isinstance(dep, (list, dict)) else []


def _merge(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        out = dict(a)
        for k, v in b.items():
            out[k] = _merge(a.get(k), v) if k in a else v
        return out
    return b


def find_files(ctx, given):
    if given:
        return [host_path(ctx, f) for f in given]
    d = ctx.cwd
    while True:
        for f in FILES:
            if ctx.fs.exists(f"{d}/{f}"):
                out = [f"{d}/{f}"]
                for ov in ("compose.override.yaml", "compose.override.yml", "docker-compose.override.yml",
                           "docker-compose.override.yaml"):
                    if ctx.fs.exists(f"{d}/{ov}") and f.startswith(ov.split(".")[0].replace("-override", "")):
                        out.append(f"{d}/{ov}")
                        break
                return out
        if d == "/":
            break
        d = posixpath.dirname(d)
    raise DockerError("no configuration file provided: not found", 1)


# ---------------------------------------------------------------------------- creating a service's container

def _env_list(p, svc):
    out = []
    files = svc.get("env_file") or []
    if isinstance(files, (str, dict)):
        files = [files]
    for f in files:
        path = f["path"] if isinstance(f, dict) else f
        full = path if path.startswith("/") else posixpath.normpath(posixpath.join(p.dir, path))
        try:
            text = bytes(p.ctx.fs.lookup(full).data).decode("utf-8", "replace")
        except FSError:
            raise DockerError(f"env file {full} not found: stat {full}: no such file or directory", 1)
        for line in text.split("\n"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                out.append(line)
    env = svc.get("environment") or []
    if isinstance(env, dict):
        for k, v in env.items():
            out.append(f"{k}={'' if v is None else ('true' if v is True else 'false' if v is False else v)}")
    else:
        for e in env:
            out.append(str(e) if "=" in str(e) else f"{e}={p.ctx.env.get(str(e), '')}")
    return out


def _cfg_for(p, s, img):
    ctx = p.ctx
    svc = p.services[s]
    from .cli import parse_publish
    cfg = {"image_ref": p.image_name(s), "name": p.container_name(s), "env": _env_list(p, svc),
           "labels": {"com.docker.compose.project": p.name, "com.docker.compose.service": s,
                      "com.docker.compose.container-number": "1",
                      "com.docker.compose.project.working_dir": p.dir,
                      "com.docker.compose.project.config_files": ",".join(p.files),
                      **({k: str(v) for k, v in (svc.get("labels") or {}).items()} if isinstance(svc.get("labels"),
                                                                                                dict) else {})},
           "network": None, "aliases": [s], "tty": bool(svc.get("tty")),
           "interactive": bool(svc.get("stdin_open"))}
    c_ = svc.get("command")
    if c_ is not None:
        cfg["cmd"] = shlex.split(c_) if isinstance(c_, str) else [str(x) for x in c_]
    e_ = svc.get("entrypoint")
    if e_ is not None:
        cfg["entrypoint"] = shlex.split(e_) if isinstance(e_, str) else [str(x) for x in e_]
        cfg["entrypoint_set"] = True
    for k_src, k_dst in (("working_dir", "workdir"), ("user", "user"), ("hostname", "hostname")):
        if svc.get(k_src):
            cfg[k_dst] = str(svc[k_src])
    if svc.get("restart"):
        cfg["restart"] = str(svc["restart"])
    if svc.get("init"):
        cfg["init"] = True
    ports = []
    for x in svc.get("ports") or []:
        if isinstance(x, dict):
            ports.append((x.get("host_ip", "0.0.0.0"), int(x.get("published", 32768)), int(x["target"]),
                          x.get("protocol", "tcp")))
        else:
            ports += parse_publish(str(x), img)
    cfg["ports"] = ports
    mounts = []
    for v in svc.get("volumes") or []:
        if isinstance(v, dict):
            src = v.get("source")
            t = v.get("type", "volume")
            if t == "bind":
                src = posixpath.normpath(posixpath.join(p.dir, src)) if not src.startswith("/") else src
                mounts.append({"type": "bind", "source": src, "dest": v["target"], "ro": bool(v.get("read_only"))})
            else:
                mounts.append({"type": t, "source": f"{p.name}_{src}" if src else None, "dest": v["target"],
                               "ro": bool(v.get("read_only"))})
            continue
        parts = str(v).split(":")
        ro = len(parts) == 3 and parts[2].startswith("ro")
        if len(parts) == 1:
            mounts.append({"type": "volume", "source": None, "dest": parts[0], "ro": False})
            continue
        src, dest = parts[0], parts[1]
        if src.startswith((".", "/", "~")):
            full = src if src.startswith("/") else posixpath.normpath(posixpath.join(p.dir, src)) if not \
                src.startswith("~") else ctx.env.get("HOME", "/root") + src[1:]
            mounts.append({"type": "bind", "source": full, "dest": dest, "ro": ro})
        else:
            if src not in p.volumes:
                raise DockerError(f"service \"{s}\" refers to undefined volume {src}: invalid compose project", 1)
            ext = (p.volumes.get(src) or {}).get("external") if isinstance(p.volumes.get(src), dict) else None
            vname = (p.volumes.get(src) or {}).get("name") if isinstance(p.volumes.get(src), dict) else None
            mounts.append({"type": "volume", "source": vname or (src if ext else f"{p.name}_{src}"), "dest": dest,
                           "ro": ro})
    cfg["mounts"] = mounts
    if svc.get("shm_size"):
        from .engine import parse_size
        cfg["shm_mb"] = parse_size(str(svc["shm_size"])) // (1024 * 1024)
    if svc.get("ipc") == "host":
        cfg["shm_mb"] = 16384
    if svc.get("mem_limit"):
        from .engine import parse_size
        cfg["memory"] = parse_size(str(svc["mem_limit"]))
    devs = (((svc.get("deploy") or {}).get("resources") or {}).get("reservations") or {}).get("devices") or []
    for dv in devs:
        if isinstance(dv, dict) and (dv.get("driver") == "nvidia" or "gpu" in str(dv.get("capabilities"))):
            if dv.get("device_ids"):
                cfg["gpus"] = [int(x) for x in dv["device_ids"]]
            elif dv.get("count") in (None, "all"):
                cfg["gpus"] = "all"
            else:
                cfg["gpus"] = list(range(int(dv["count"])))
    if svc.get("gpus"):
        cfg["gpus"] = "all"
    hc = svc.get("healthcheck")
    if isinstance(hc, dict) and not hc.get("disable"):
        from .cli import _dur
        test = hc.get("test")
        if isinstance(test, str):
            test = ["CMD-SHELL", test]
        cfg["healthcheck"] = {"Test": test or ["NONE"], "Interval": int(_dur(str(hc.get("interval", "30s"))) * NS),
                              "Retries": int(hc.get("retries", 3)),
                              "StartPeriod": int(_dur(str(hc.get("start_period", "0s"))) * NS)}
    nets = svc.get("networks")
    if svc.get("network_mode"):
        cfg["network"] = svc["network_mode"]
    elif nets:
        names = list(nets) if isinstance(nets, (list, dict)) else []
        cfg["network"] = _net_name(p, names[0])
        cfg["networks_extra"] = [_net_name(p, n) for n in names[1:]]
    else:
        cfg["network"] = f"{p.name}_default"
    extra = svc.get("extra_hosts") or []
    cfg["add_hosts"] = [tuple(str(x).split(":", 1)) if "=" not in str(x) else tuple(str(x).split("=", 1))
                        for x in extra] if isinstance(extra, list) else [(k, v) for k, v in extra.items()]
    return cfg


def _net_name(p, n):
    spec = p.networks.get(n) if isinstance(p.networks, dict) else None
    if isinstance(spec, dict) and spec.get("external"):
        return spec.get("name") or n
    if isinstance(spec, dict) and spec.get("name"):
        return spec["name"]
    return f"{p.name}_{n}"


# ---------------------------------------------------------------------------- the command

def _line(o, mark, kind, name, state, secs):
    label = f" {mark} {kind} {name}"
    o.err(f"{label:<40}{state:<10}{secs:>28.1f}s\n")       # 79 columns: fits an 80-column terminal


@cmd("compose")
def docker_compose(ctx, o, args):
    opts, rest = flags(args, {"-f|--file": "m", "-p|--project-name": "v", "--profile": "m", "--env-file": "v",
                              "--project-directory": "v", "--progress": "v", "--ansi": "v"})
    d = daemon_of(ctx.world, ctx.host)
    if not rest or rest[0] in ("--help", "-h", "help"):
        ctx.w("\nUsage:  docker compose [OPTIONS] COMMAND\n\nDefine and run multi-container applications with Docker\n\n"
              "Commands:\n  build       Build or rebuild services\n  config      Parse, resolve and render compose file "
              "in canonical format\n  down        Stop and remove containers, networks\n  exec        Execute a "
              "command in a running container\n  images      List images used by the created containers\n  logs        "
              "View output from containers\n  ls          List running compose projects\n  ps          List "
              "containers\n  pull        Pull service images\n  restart     Restart service containers\n  run         "
              "Run a one-off command on a service\n  start       Start services\n  stop        Stop services\n  up    "
              "      Create and start containers\n\nRun 'docker compose COMMAND --help' for more information on a "
              "command.\n")
        return 0
    if rest[0] in ("version", "--version"):
        ctx.w("Docker Compose version v2.29.7\n")
        return 0
    sub, rest = rest[0], rest[1:]
    if sub == "ls":
        projects = {}
        for c in d.containers.values():
            pn = c.labels.get("com.docker.compose.project")
            if pn:
                st = projects.setdefault(pn, [0, 0, c.labels.get("com.docker.compose.project.config_files", "")])
                st[0 if c.running else 1] += 1
        rows = [["NAME", "STATUS", "CONFIG FILES"]]
        for pn, (run, ex, files) in sorted(projects.items()):
            status = ", ".join(x for x in ([f"running({run})"] if run else []) + ([f"exited({ex})"] if ex else []))
            rows.append([pn, status, files])
        ctx.w(gotmpl.tabulate(rows))
        return 0
    p = Project(ctx, d, find_files(ctx, opts.get("file")), opts.get("project-name"), opts.get("profile", []))
    fn = {"up": _up, "down": _down, "ps": _ps, "logs": _logs, "exec": _exec, "run": _run, "start": _start,
          "stop": _stop, "restart": _restart, "build": _build, "pull": _pull, "config": _config,
          "images": _images, "rm": _rm, "create": _up, "top": _top}.get(sub)
    if fn is None:
        raise DockerError(f"unknown docker command: \"compose {sub}\"", 1)
    r = fn(ctx, o, d, p, rest)
    if hasattr(r, "send"):
        r = yield from r
    return r or 0


def _ensure_network(o, d, p, n, report):
    if d.network(n) is None:
        d.create_network(n, labels={"com.docker.compose.project": p.name})
        report.append(("✔", "Network", n, "Created", 0.1))


def _service_image(ctx, o, d, p, s, force_build=False, report=None):
    svc = p.services[s]
    name = p.image_name(s)
    img = d.find_image(name)
    if svc.get("build") and (img is None or force_build):
        b = svc["build"]
        context = b if isinstance(b, str) else (b.get("context") or ".")
        cdir = context if context.startswith("/") else posixpath.normpath(posixpath.join(p.dir, context))
        bargs = ["-t", name]
        if isinstance(b, dict):
            if b.get("dockerfile"):
                bargs += ["-f", posixpath.join(cdir, b["dockerfile"])]
            if b.get("target"):
                bargs += ["--target", b["target"]]
            for k, v in (b.get("args") or {}).items() if isinstance(b.get("args"), dict) else \
                    [(x.split("=", 1) + [""])[:2] for x in (b.get("args") or [])]:
                bargs += ["--build-arg", f"{k}={v}"]
        from .build import docker_build
        saved = ctx.cwd
        r = docker_build(ctx, o, bargs + [cdir])
        img = None
        res = yield from r
        if res != 0:
            raise DockerError(f"failed to solve: service {s} failed to build", 1)
        img = d.find_image(name)
        if report is not None:
            report.append(("✔", "Service", s, "Built", 0.0))
        del saved
        return img
    if img is None:
        from .cmd_images import pull_image
        from .cli import Out

        class Q(Out):
            def out(self, x):
                pass

            def err(self, x):
                pass
        img = yield from pull_image(ctx, Q(ctx), d, name, quiet=True)
        o.err(f"[+] Pulling 1/1\n ✔ {s} Pulled{'':>58}{4.2:.1f}s\n")
    return img


def _up(ctx, o, d, p, args):
    opts, svcs = flags(args, {"-d|--detach": None, "--build": None, "--no-build": None, "--force-recreate": None,
                              "--remove-orphans": None, "--wait": None, "--no-deps": None, "--abort-on-container-exit":
                              None, "-V|--renew-anon-volumes": None, "--pull": "v", "--no-start": None,
                              "--quiet-pull": None, "-y|--yes": None}, stop_at_positional=False)
    targets = svcs or p.active
    for s in targets:
        if s not in p.services:
            raise DockerError(f"no such service: {s}", 1)
    order = p.order(targets) if not opts.get("no-deps") else targets
    report = []
    # networks and volumes
    nets = {f"{p.name}_default"} if any(not p.services[s].get("networks") and not p.services[s].get("network_mode")
                                        for s in order) else set()
    for s in order:
        for n in (p.services[s].get("networks") or []):
            nets.add(_net_name(p, n))
    for n in sorted(nets):
        spec = p.networks.get(n.split("_", 1)[-1]) if isinstance(p.networks, dict) else None
        if isinstance(spec, dict) and spec.get("external"):
            if d.network(n) is None:
                raise DockerError(f"network {n} declared as external, but could not be found", 1)
            continue
        _ensure_network(o, d, p, n, report)
    for vname, vspec in (p.volumes or {}).items():
        if isinstance(vspec, dict) and vspec.get("external"):
            if (vspec.get("name") or vname) not in d.volumes:
                raise DockerError(f"external volume \"{vspec.get('name') or vname}\" not found", 1)
            continue
        full = (vspec or {}).get("name") if isinstance(vspec, dict) and vspec.get("name") else f"{p.name}_{vname}"
        if full not in d.volumes and any(f"{vname}:" in str(v) or (isinstance(v, dict) and v.get("source") == vname)
                                         for s in order for v in (p.services[s].get("volumes") or [])):
            d.volume(full, labels={"com.docker.compose.project": p.name, "com.docker.compose.volume": vname})
            report.append(("✔", "Volume", f'"{full}"', "Created", 0.0))
    started = []
    for s in order:
        img = yield from _service_image(ctx, o, d, p, s, bool(opts.get("build")), report)
        c = p.find(s)
        cfg = _cfg_for(p, s, img)
        if c is not None and (opts.get("force-recreate") or c.image is not img or opts.get("build")):
            if c.running:
                yield from d.stop(c, 2)
            d.remove(c, force=True)
            c = None
            state = "Recreated"
        else:
            state = "Created" if c is None else ("Running" if c.running else "Started")
        if c is None:
            c = d.create(img, cfg)
            report.append(("✔", "Container", c.name, "Created", 0.1))
        if opts.get("no-start"):
            continue
        # wait for dependencies' conditions
        deps = p.services[s].get("depends_on") or []
        if isinstance(deps, dict):
            for dep, cond in deps.items():
                cond = (cond or {}).get("condition", "service_started") if isinstance(cond, dict) else \
                    "service_started"
                dc = p.find(dep)
                if dc is None:
                    continue
                if cond == "service_healthy":
                    waited = 0
                    while dc.health not in ("healthy", "unhealthy") and waited < 600:
                        yield ("sleep", ctx.now + int(0.5 * NS))
                        waited += 1
                        if not dc.running:
                            break
                    if dc.health != "healthy":
                        _flush(o, report)
                        raise DockerError(f"dependency failed to start: container {dc.name} is "
                                          f"{dc.health or 'not running'}", 1)
                    report.append(("✔", "Container", dc.name, "Healthy", 5.6))
                elif cond == "service_completed_successfully":
                    while dc.running:
                        yield ("sleep", ctx.now + int(0.5 * NS))
                    if dc.exit_code != 0:
                        _flush(o, report)
                        raise DockerError(f"service \"{dep}\" didn't complete successfully: exit {dc.exit_code}", 1)
        if not c.running:
            try:
                d.start(c)
            except DockerError as e:
                _flush(o, report)
                raise DockerError(f"Error response from daemon: {e.msg.split('Error response from daemon: ')[-1]}",
                                  1)
            report = [r for r in report if not (r[2] == c.name and r[3] == "Created")] + \
                [("✔", "Container", c.name, "Started", 0.4)]
            started.append(c)
        else:
            report.append(("✔", "Container", c.name, "Running", 0.0))
        yield ("sleep", ctx.now + int(0.2 * NS))
    _flush(o, report)
    if opts.get("detach") or opts.get("no-start"):
        if opts.get("wait"):
            for s in order:
                c = p.find(s)
                while c is not None and c.health == "starting":
                    yield ("sleep", ctx.now + int(0.5 * NS))
        return 0
    # attached: follow every service's logs until ^C, then stop them
    cs = [p.find(s) for s in order if p.find(s) is not None]
    o.out("Attaching to " + ", ".join(c.name for c in cs) + "\n")
    seen = {c.id: 0 for c in cs}
    width = max(len(_prefix(c)) for c in cs)
    fg = ctx.foreground()
    deadline = ctx.now + ctx.world.ctrl_c_after * NS if fg else float("inf")
    while True:
        for c in cs:
            from .cli import _split_entries
            for t, stream, data in _split_entries(c.logs[seen[c.id]:]):
                ctx.w(f"{_prefix(c):<{width}}  | " + data.decode("utf-8", "replace"))
            seen[c.id] = len(c.logs)
        yield from ctx.flush()
        if all(not c.running for c in cs):
            for c in cs:
                ctx.w(f"{_prefix(c)} exited with code {c.exit_code}\n")
            return 0
        if ctx.now >= deadline:
            ctx.world.tty.write("echo", b"^C")
            o.out("Gracefully stopping... (press Ctrl+C again to force)\n")
            for c in reversed(cs):
                if c.running:
                    yield from d.stop(c, 10)
            return 130
        nxt = ctx.now + int(0.25 * NS)
        if deadline != float("inf"):
            nxt = min(nxt, int(deadline))
        yield ("sleep", nxt)


def _prefix(c):
    return f"{c.labels.get('com.docker.compose.service', c.name)}-1"


def _flush(o, report):
    if not report:
        return
    o.err(f"[+] Running {len(report)}/{len(report)}\n")
    for mark, kind, name, state, secs in report:
        _line(o, mark, kind, name, state, secs)
    report.clear()


def _down(ctx, o, d, p, args):
    opts, rest = flags(args, {"-v|--volumes": None, "--rmi": "v", "--remove-orphans": None, "-t|--timeout": "v"},
                       stop_at_positional=False)
    report = []
    for c in sorted(p.containers(), key=lambda c: -c.created):
        if c.running:
            yield from d.stop(c, int(opts.get("timeout", 10)))
            report.append(("✔", "Container", c.name, "Stopped", 0.3))
        d.remove(c, force=True)
        report = [r for r in report if r[2] != c.name] + [("✔", "Container", c.name, "Removed", 0.3)]
    if opts.get("volumes"):
        for v in list(d.volumes.values()):
            if v.labels.get("com.docker.compose.project") == p.name and not d.volume_users(v.name):
                d.remove_volume(v.name)
                report.append(("✔", "Volume", v.name, "Removed", 0.0))
    for n in list(d.networks.values()):
        if n.labels.get("com.docker.compose.project") == p.name and \
                not any(n.name in c.networks for c in d.containers.values()):
            del d.networks[n.name]
            report.append(("✔", "Network", n.name, "Removed", 0.1))
    if opts.get("rmi") == "all":
        for s in p.services:
            img = d.find_image(p.image_name(s))
            if img is not None and not d.image_in_use(img):
                for t in list(img.repo_tags):
                    d.untag(t)
                d.images.pop(img.id, None)
                report.append(("✔", "Image", p.image_name(s), "Removed", 0.0))
    if not report:
        o.err("")
    _flush(o, report)
    return 0


def _ps(ctx, o, d, p, args):
    opts, rest = flags(args, {"-a|--all": None, "-q|--quiet": None, "--format": "v", "--services": None,
                              "--status": "m"}, stop_at_positional=False)
    cs = sorted(p.containers(), key=lambda c: c.name)
    if not opts.get("all"):
        cs = [c for c in cs if c.running]
    if rest:
        cs = [c for c in cs if c.labels.get("com.docker.compose.service") in rest]
    if opts.get("services"):
        ctx.w("".join(c.labels["com.docker.compose.service"] + "\n" for c in cs))
        return 0
    if opts.get("quiet"):
        ctx.w("".join(c.id + "\n" for c in cs))
        return 0
    from .cli import ps_row
    rows = []
    for c in cs:
        r = ps_row(d, c, ctx.now)
        r["Service"] = c.labels.get("com.docker.compose.service", "")
        r["Name"] = c.name
        r["Health"] = c.health or ""
        rows.append(r)
    if opts.get("format"):
        fmt = opts["format"]
        if fmt == "json":
            ctx.w("".join(json.dumps(r) + "\n" for r in rows))
        elif fmt.startswith("table"):
            ctx.w(gotmpl.table(fmt, rows, {"Name": "NAME", "Image": "IMAGE", "Service": "SERVICE", "Status": "STATUS",
                                           "Ports": "PORTS", "State": "STATE", "Health": "HEALTH"}))
        else:
            ctx.w("".join(gotmpl.render(fmt.replace("\\t", "\t"), r) + "\n" for r in rows))
        return 0
    table = [["NAME", "IMAGE", "COMMAND", "SERVICE", "CREATED", "STATUS", "PORTS"]]
    for r in rows:
        table.append([r["Name"], r["Image"], r["Command"], r["Service"], r["RunningFor"], r["Status"], r["Ports"]])
    ctx.w(gotmpl.tabulate(table))
    return 0
    yield


def _logs(ctx, o, d, p, args):
    opts, rest = flags(args, {"-f|--follow": None, "-n|--tail": "v", "-t|--timestamps": None, "--no-color": None,
                              "--no-log-prefix": None}, stop_at_positional=False)
    cs = [c for c in sorted(p.containers(), key=lambda c: c.name)
          if not rest or c.labels.get("com.docker.compose.service") in rest]
    if not cs:
        return 0
    from .cli import _split_entries
    width = max(len(_prefix(c)) for c in cs)
    seen = {}
    for c in cs:
        lines = _split_entries(c.logs)
        if opts.get("tail") and opts["tail"] != "all":
            lines = lines[-int(opts["tail"]):] if int(opts["tail"]) > 0 else []
        for t, stream, data in lines:
            pre = "" if opts.get("no-log-prefix") else f"{_prefix(c):<{width}}  | "
            ctx.w(pre + data.decode("utf-8", "replace"))
        seen[c.id] = len(c.logs)
    if not opts.get("follow"):
        return 0
    fg = ctx.foreground()
    deadline = ctx.now + ctx.world.ctrl_c_after * NS if fg else float("inf")
    while True:
        for c in cs:
            for t, stream, data in _split_entries(c.logs[seen[c.id]:]):
                pre = "" if opts.get("no-log-prefix") else f"{_prefix(c):<{width}}  | "
                ctx.w(pre + data.decode("utf-8", "replace"))
            seen[c.id] = len(c.logs)
        yield from ctx.flush()
        if all(not c.running for c in cs):
            return 0
        if ctx.now >= deadline:
            ctx.world.tty.write("echo", b"^C\n")
            return 130
        nxt = ctx.now + int(0.25 * NS)
        if deadline != float("inf"):
            nxt = min(nxt, int(deadline))
        yield ("sleep", nxt)


def _exec(ctx, o, d, p, args):
    opts, rest = flags(args, {"-T|--no-TTY": None, "-u|--user": "v", "-w|--workdir": "v", "-e|--env": "m",
                              "-d|--detach": None, "-i|--interactive": None, "--index": "v"})
    if len(rest) < 2:
        raise DockerError("\"docker compose exec\" requires at least 2 arguments.", 1)
    s = rest[0]
    if s not in p.services:
        raise DockerError(f"no such service: {s}", 1)
    c = p.find(s)
    if c is None or not c.running:
        raise DockerError(f"service \"{s}\" is not running", 1)
    from .cli import docker_exec
    eargs = ["-i"]
    if not opts.get("no-TTY") and ctx.stdin_is_tty():
        eargs.append("-t")
    for k in ("user", "workdir"):
        if opts.get(k):
            eargs += [f"--{k}", opts[k]]
    for e in opts.get("env", []):
        eargs += ["-e", e]
    r = docker_exec(ctx, o, eargs + [c.name] + rest[1:])
    return (yield from r)


def _run(ctx, o, d, p, args):
    opts, rest = flags(args, {"--rm": None, "-d|--detach": None, "-e|--env": "m", "--entrypoint": "v",
                              "-T|--no-TTY": None, "--no-deps": None, "-u|--user": "v", "-w|--workdir": "v",
                              "--name": "v", "-p|--publish": "m", "--service-ports": None, "--build": None,
                              "-v|--volume": "m"})
    if not rest:
        raise DockerError("\"docker compose run\" requires at least 1 argument.", 1)
    s = rest[0]
    if s not in p.services:
        raise DockerError(f"no such service: {s}", 1)
    # dependencies come up first
    deps = [x for x in p.order([s]) if x != s]
    if deps and not opts.get("no-deps"):
        r = yield from _up(ctx, o, d, p, ["-d"] + deps)
        if r:
            return r
    img = yield from _service_image(ctx, o, d, p, s, bool(opts.get("build")))
    cfg = _cfg_for(p, s, img)
    import random as _r
    cfg["name"] = opts.get("name") or f"{p.name}-{s}-run-{''.join(_r.Random(ctx.now).choice('0123456789abcdef') for _ in range(12))}"
    cfg["labels"]["com.docker.compose.oneoff"] = "True"
    for n in [cfg.get("network")] + list(cfg.get("networks_extra", [])):
        if n and n not in ("host", "none") and d.network(n) is None:
            _ensure_network(o, d, p, n, [])
    for m in cfg.get("mounts", []):
        if m["type"] == "volume" and m.get("source") and m["source"] not in d.volumes:
            d.volume(m["source"], labels={"com.docker.compose.project": p.name})
    if not opts.get("service-ports"):
        cfg["ports"] = []
    if len(rest) > 1:
        cfg["cmd"] = rest[1:]
    if opts.get("entrypoint") is not None:
        cfg["entrypoint"] = shlex.split(opts["entrypoint"])
        cfg["entrypoint_set"] = True
    cfg["env"] = cfg.get("env", []) + opts.get("env", [])
    cfg["rm"] = bool(opts.get("rm"))
    cfg["tty"] = not opts.get("no-TTY") and ctx.stdin_is_tty()
    cfg["interactive"] = True
    for k in ("user", "workdir"):
        if opts.get(k):
            cfg[k] = opts[k]
    c = d.create(img, cfg)
    if opts.get("detach"):
        d.start(c)
        ctx.w(c.name + "\n")
        return 0
    attach = {0: ctx.proc.fds.get(0), 1: ctx.proc.fds.get(1), 2: ctx.proc.fds.get(2), "pgid": ctx.proc.pgid}
    yield from ctx.flush()
    proc = d.start(c, attach)
    from .cli import wait_attached
    return (yield from wait_attached(ctx, d, c, proc))


def _start(ctx, o, d, p, args):
    report = []
    for s in (args or p.order(p.active)):
        c = p.find(s)
        if c is not None and not c.running:
            d.start(c)
            report.append(("✔", "Container", c.name, "Started", 0.3))
    _flush(o, report)
    return 0
    yield


def _stop(ctx, o, d, p, args):
    opts, rest = flags(args, {"-t|--timeout": "v"}, stop_at_positional=False)
    report = []
    for c in sorted(p.containers(), key=lambda c: -c.created):
        if rest and c.labels.get("com.docker.compose.service") not in rest:
            continue
        if c.running:
            yield from d.stop(c, int(opts.get("timeout", 10)))
            report.append(("✔", "Container", c.name, "Stopped", 0.3))
    _flush(o, report)
    return 0


def _restart(ctx, o, d, p, args):
    opts, rest = flags(args, {"-t|--timeout": "v"}, stop_at_positional=False)
    report = []
    for s in (rest or p.order(p.active)):
        c = p.find(s)
        if c is None:
            continue
        if c.running:
            yield from d.stop(c, int(opts.get("timeout", 10)))
        d.start(c)
        report.append(("✔", "Container", c.name, "Started", 0.6))
    _flush(o, report)
    return 0


def _rm(ctx, o, d, p, args):
    opts, rest = flags(args, {"-f|--force": None, "-s|--stop": None, "-v|--volumes": None}, stop_at_positional=False)
    gone = [c for c in p.containers() if not c.running and (not rest or c.labels.get("com.docker.compose.service")
                                                            in rest)]
    for c in gone:
        d.remove(c)
    _flush(o, [("✔", "Container", c.name, "Removed", 0.0) for c in gone])
    return 0
    yield


def _build(ctx, o, d, p, args):
    opts, rest = flags(args, {"--no-cache": None, "--pull": None, "-q|--quiet": None}, stop_at_positional=False)
    for s in (rest or p.active):
        if p.services[s].get("build"):
            yield from _service_image(ctx, o, d, p, s, True)
    return 0


def _pull(ctx, o, d, p, args):
    from .cmd_images import pull_image
    from .cli import Out

    class Q(Out):
        def out(self, x):
            pass

        def err(self, x):
            pass
    n = 0
    lines = []
    for s in (args or p.active):
        svc = p.services[s]
        if svc.get("image"):
            try:
                yield from pull_image(ctx, Q(ctx), d, svc["image"], quiet=True)
                lines.append(f" ✔ {s} Pulled")
                n += 1
            except DockerError:
                if svc.get("build"):
                    lines.append(f" ! {s} Warning pull access denied for {svc['image']}, repository does not exist "
                                 f"or may require 'docker login'")
                    continue
                raise
    o.err(f"[+] Pulling {n}/{n}\n" + "".join(f"{x:<70}{3.1:>5.1f}s\n" for x in lines))
    return 0


def _config(ctx, o, d, p, args):
    opts, rest = flags(args, {"-q|--quiet": None, "--services": None, "--volumes": None, "--format": "v",
                              "--images": None}, stop_at_positional=False)
    if opts.get("quiet"):
        return 0
    if opts.get("services"):
        ctx.w("".join(s + "\n" for s in p.services))
        return 0
    if opts.get("volumes"):
        ctx.w("".join(v + "\n" for v in p.volumes))
        return 0
    if opts.get("images"):
        ctx.w("".join(p.image_name(s) + "\n" for s in p.services))
        return 0
    out = {"name": p.name, "services": {}}
    for s, svc in sorted(p.services.items()):
        x = {}
        for k in sorted(svc):
            v = svc[k]
            if k == "build":
                ctxdir = v if isinstance(v, str) else v.get("context", ".")
                x[k] = {"context": posixpath.normpath(posixpath.join(p.dir, ctxdir)), "dockerfile": "Dockerfile"
                        if isinstance(v, str) else v.get("dockerfile", "Dockerfile")}
            elif k == "environment":
                x[k] = dict(e.split("=", 1) for e in _env_list(p, {"environment": v}))
            elif k == "ports" and isinstance(v, list):
                from .cli import parse_publish
                x[k] = [{"mode": "ingress", "target": t, "published": str(h), "protocol": pr}
                        for item in v for ip, h, t, pr in parse_publish(str(item), None)]
            else:
                x[k] = v
        x.setdefault("networks", {"default": None})
        out["services"][s] = x
    if not any(isinstance(v, dict) and v.get("networks") and v["networks"] != {"default": None}
               for v in out["services"].values()):
        out["networks"] = {"default": {"name": f"{p.name}_default"}}
    if p.volumes:
        out["volumes"] = {k: {"name": f"{p.name}_{k}"} for k in p.volumes}
    if opts.get("format") == "json":
        ctx.w(json.dumps(out, indent=2) + "\n")
    else:
        ctx.w(yaml_dump(out) + "\n")
    return 0
    yield


def _images(ctx, o, d, p, args):
    rows = [["CONTAINER", "REPOSITORY", "TAG", "IMAGE ID", "SIZE"]]
    from .engine import human_size
    for c in sorted(p.containers(), key=lambda c: c.name):
        ref = c.cfg.get("image_ref", "")
        repo, _, tag = (ref if ":" in ref.rsplit("/", 1)[-1] else ref + ":latest").rpartition(":")
        rows.append([c.name, repo, tag, c.image.short, human_size(c.image.size)])
    ctx.w(gotmpl.tabulate(rows))
    return 0
    yield


def _top(ctx, o, d, p, args):
    from .cli import docker_top
    for c in sorted(p.containers(), key=lambda c: c.name):
        if c.running:
            ctx.w(c.name + "\n")
            r = docker_top(ctx, o, [c.name])
            yield from r
            ctx.w("\n")
    return 0


__all__ = ["yaml_load", "Project", "docker_compose", "ago"]
