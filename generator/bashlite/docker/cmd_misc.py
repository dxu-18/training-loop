"""volume, network, system (df, prune, events, info), builder prune, login/logout, scout — and `systemctl` for
the docker service."""

from __future__ import annotations

import json
import re

from ..exec import util
from ..kernel import NS
from ..vfs import FSError
from . import gotmpl
from .cli import cmd, confirm, flags
from .cmd_images import registries, registry_of
from .engine import ENGINE, DockerError, ago, daemon_of, human_size
from .fsutil import tree_bytes


def _iso(ns):
    from .cli import _iso as iso
    return iso(ns)


# ---------------------------------------------------------------------------- volumes

def volume_json(d, v):
    return {"CreatedAt": _iso(v.created).split(".")[0] + "Z", "Driver": "local", "Labels": v.labels or None,
            "Mountpoint": v.mountpoint, "Name": v.name, "Options": None, "Scope": "local"}


@cmd("volume")
def docker_volume(ctx, o, args):
    d = daemon_of(ctx.world, ctx.host)
    sub = args[0] if args else ""
    rest = args[1:]
    if sub == "create":
        opts, r = flags(rest, {"--name": "v", "-l|--label": "m", "-d|--driver": "v", "-o|--opt": "m"},
                        stop_at_positional=False)
        name = opts.get("name") or (r[0] if r else None)
        labels = dict(x.partition("=")[::2] for x in opts.get("label", []))
        v = d.volume(name, labels=labels)
        ctx.w(v.name + "\n")
        return 0
    if sub in ("ls", "list"):
        opts, r = flags(rest, {"-q|--quiet": None, "-f|--filter": "m", "--format": "v"}, stop_at_positional=False)
        vs = sorted(d.volumes.values(), key=lambda v: v.name)
        for f in opts.get("filter", []):
            k, _, val = f.partition("=")
            if k == "dangling":
                vs = [v for v in vs if (not d.volume_users(v.name)) == (val == "true")]
            elif k == "name":
                vs = [v for v in vs if val in v.name]
        if opts.get("quiet"):
            ctx.w("".join(v.name + "\n" for v in vs))
            return 0
        rows = [{"Driver": "local", "Name": v.name, "Mountpoint": v.mountpoint, "Scope": "local"} for v in vs]
        if opts.get("format"):
            fmt = opts["format"]
            ctx.w(gotmpl.table(fmt, rows, {"Driver": "DRIVER", "Name": "VOLUME NAME", "Mountpoint": "MOUNTPOINT"})
                  if fmt.startswith("table") else "".join(gotmpl.render(fmt.replace("\\t", "\t"), r) + "\n"
                                                          for r in rows))
            return 0
        ctx.w(gotmpl.tabulate([["DRIVER", "VOLUME NAME"]] + [["local", v.name] for v in vs], pad=5))
        return 0
    if sub == "inspect":
        opts, r = flags(rest, {"-f|--format": "v"}, stop_at_positional=False)
        out, st = [], 0
        for name in r:
            v = d.volumes.get(name)
            if v is None:
                o.err(f"Error response from daemon: get {name}: no such volume\n")
                st = 1
                continue
            out.append(volume_json(d, v))
        if opts.get("format"):
            ctx.w("".join(gotmpl.render(opts["format"], x) + "\n" for x in out))
        else:
            ctx.w(json.dumps(out, indent=4) + "\n")
        return st
    if sub in ("rm", "remove"):
        opts, r = flags(rest, {"-f|--force": None}, stop_at_positional=False)
        st = 0
        for name in r:
            v = d.volumes.get(name)
            if v is None:
                o.err(f"Error response from daemon: get {name}: no such volume\n")
                st = 1
                continue
            users = d.volume_users(name)
            if users:
                o.err(f"Error response from daemon: remove {name}: volume is in use - [{users[0].id}]\n")
                st = 1
                continue
            d.remove_volume(name)
            ctx.w(name + "\n")
        return st
    if sub == "prune":
        opts, r = flags(rest, {"-a|--all": None, "-f|--force": None, "--filter": "m"}, stop_at_positional=False)
        if not opts.get("force"):
            ok = yield from confirm(ctx, "WARNING! This will remove anonymous local volumes not used by at least one "
                                         "container.\nAre you sure you want to continue? [y/N] " if not opts.get("all")
                                    else "WARNING! This will remove all local volumes not used by at least one "
                                         "container.\nAre you sure you want to continue? [y/N] ")
            if not ok:
                return 0
        gone = [v for v in list(d.volumes.values()) if not d.volume_users(v.name) and (v.anonymous or opts.get("all"))]
        total = 0
        for v in gone:
            try:
                total += tree_bytes(d.host.fs.lookup(v.mountpoint))
            except FSError:
                pass
            d.remove_volume(v.name)
        if gone:
            ctx.w("Deleted Volumes:\n" + "".join(v.name + "\n" for v in gone) + "\n")
        ctx.w(f"Total reclaimed space: {human_size(total) if total else '0B'}\n")
        return 0
    ctx.w("\nUsage:  docker volume COMMAND\n\nManage volumes\n\nCommands:\n  create      Create a volume\n  inspect     "
          "Display detailed information on one or more volumes\n  ls          List volumes\n  prune       Remove "
          "unused local volumes\n  rm          Remove one or more volumes\n\nRun 'docker volume COMMAND --help' for "
          "more information on a command.\n")
    return 0 if not sub else 1


# ---------------------------------------------------------------------------- networks

def network_json(d, n):
    cs = {c.id: {"Name": c.name, "IPv4Address": f"{ip}/16", "MacAddress": "02:42:ac:12:00:02"}
          for c in d.containers.values() if c.running for net, ip in c.networks.items() if net == n.name}
    return {"Name": n.name, "Id": n.id, "Created": _iso(n.created), "Scope": "local", "Driver": n.driver,
            "EnableIPv6": False, "IPAM": {"Driver": "default", "Config": [{"Subnet": n.subnet, "Gateway": n.gateway}]
                                          if n.subnet else []},
            "Internal": n.internal, "Attachable": False, "Containers": cs, "Options": {}, "Labels": n.labels}


@cmd("network")
def docker_network(ctx, o, args):
    d = daemon_of(ctx.world, ctx.host)
    sub = args[0] if args else ""
    rest = args[1:]
    if sub == "create":
        opts, r = flags(rest, {"-d|--driver": "v", "--internal": None, "--subnet": "m", "--label": "m",
                               "--attachable": None}, stop_at_positional=False)
        if not r:
            raise DockerError("\"docker network create\" requires exactly 1 argument.", 1)
        n = d.create_network(r[0], opts.get("driver", "bridge"), bool(opts.get("internal")))
        ctx.w(n.id + "\n")
        return 0
    if sub in ("ls", "list"):
        opts, r = flags(rest, {"-q|--quiet": None, "-f|--filter": "m", "--format": "v", "--no-trunc": None},
                        stop_at_positional=False)
        ns = sorted(d.networks.values(), key=lambda n: n.name)
        if opts.get("quiet"):
            ctx.w("".join(n.id[:12] + "\n" for n in ns))
            return 0
        rows = [{"ID": n.id[:12], "Name": n.name, "Driver": n.driver, "Scope": "local"} for n in ns]
        if opts.get("format"):
            fmt = opts["format"]
            ctx.w(gotmpl.table(fmt, rows, {"ID": "NETWORK ID", "Name": "NAME", "Driver": "DRIVER", "Scope": "SCOPE"})
                  if fmt.startswith("table") else "".join(gotmpl.render(fmt.replace("\\t", "\t"), x) + "\n"
                                                          for x in rows))
            return 0
        ctx.w(gotmpl.tabulate([["NETWORK ID", "NAME", "DRIVER", "SCOPE"]] +
                              [[x["ID"], x["Name"], x["Driver"], x["Scope"]] for x in rows]))
        return 0
    if sub == "inspect":
        opts, r = flags(rest, {"-f|--format": "v"}, stop_at_positional=False)
        out, st = [], 0
        for name in r:
            n = d.network(name)
            if n is None:
                o.err(f"Error response from daemon: network {name} not found\n")
                st = 1
                continue
            out.append(network_json(d, n))
        if opts.get("format"):
            ctx.w("".join(gotmpl.render(opts["format"], x) + "\n" for x in out))
        else:
            ctx.w(json.dumps(out, indent=4) + "\n")
        return st
    if sub in ("rm", "remove"):
        st = 0
        for name in rest:
            n = d.network(name)
            if n is None:
                o.err(f"Error response from daemon: network {name} not found\n")
                st = 1
                continue
            if n.name in ("bridge", "host", "none"):
                o.err(f"Error response from daemon: {n.name} is a pre-defined network and cannot be removed\n")
                st = 1
                continue
            users = [c for c in d.containers.values() if c.running and n.name in c.networks]
            if users:
                o.err(f"Error response from daemon: error while removing network: network {n.name} id {n.id} has "
                      f"active endpoints\n")
                st = 1
                continue
            del d.networks[n.name]
            ctx.w(name + "\n")
        return st
    if sub in ("connect", "disconnect"):
        opts, r = flags(rest, {"--alias": "m", "--ip": "v", "-f|--force": None}, stop_at_positional=False)
        if len(r) != 2:
            raise DockerError(f"\"docker network {sub}\" requires exactly 2 arguments.", 1)
        n = d.network(r[0])
        if n is None:
            raise DockerError(f"Error response from daemon: network {r[0]} not found", 1)
        c = d.get_container(r[1])
        if sub == "connect":
            if n.name in c.networks:
                raise DockerError(f"Error response from daemon: endpoint with name {c.name} already exists in "
                                  f"network {n.name}", 1)
            ip = n.alloc()
            c.networks[n.name] = ip
            c.cfg.setdefault("aliases", [])
            c.cfg["aliases"] += opts.get("alias", [])
            ctx.world.dns[ip] = f"docker:{c.id}"
        else:
            if n.name not in c.networks:
                raise DockerError(f"Error response from daemon: container {c.id} is not connected to network "
                                  f"{n.name}", 1)
            del c.networks[n.name]
        return 0
    if sub == "prune":
        opts, r = flags(rest, {"-f|--force": None}, stop_at_positional=False)
        if not opts.get("force"):
            ok = yield from confirm(ctx, "WARNING! This will remove all custom networks not used by at least one "
                                         "container.\nAre you sure you want to continue? [y/N] ")
            if not ok:
                return 0
        gone = [n for n in list(d.networks.values()) if n.name not in ("bridge", "host", "none") and
                not any(n.name in c.networks for c in d.containers.values())]
        for n in gone:
            del d.networks[n.name]
        if gone:
            ctx.w("Deleted Networks:\n" + "".join(n.name + "\n" for n in gone))
        return 0
    ctx.w("\nUsage:  docker network COMMAND\n\nManage networks\n\nCommands:\n  connect     Connect a container to a "
          "network\n  create      Create a network\n  disconnect  Disconnect a container from a network\n  inspect     "
          "Display detailed information on one or more networks\n  ls          List networks\n  prune       Remove "
          "all unused networks\n  rm          Remove one or more networks\n\nRun 'docker network COMMAND --help' for "
          "more information on a command.\n")
    return 0 if not sub else 1


# ---------------------------------------------------------------------------- system

def disk_usage(d):
    used_imgs = {c.image.id for c in d.containers.values()}
    imgs = sum(i.size for i in d.images.values())
    imgs_r = sum(i.size for i in d.images.values() if i.id not in used_imgs)
    from .fsutil import diff_bytes
    cts = [(c, diff_bytes(c.image.fs.root, c.host.fs.root) if c.host else 0) for c in d.containers.values()]
    ct = sum(x for _, x in cts)
    ct_r = sum(x for c, x in cts if not c.running)
    vols = []
    for v in d.volumes.values():
        try:
            vols.append((v, tree_bytes(d.host.fs.lookup(v.mountpoint))))
        except FSError:
            vols.append((v, 0))
    vb = sum(x for _, x in vols)
    vb_r = sum(x for v, x in vols if not d.volume_users(v.name))
    cache = sum(e[1] for e in d.build_cache.values())
    return [("Images", len(d.images), len([i for i in d.images.values() if i.id in used_imgs]), imgs, imgs_r),
            ("Containers", len(d.containers), len([c for c in d.containers.values() if c.running]), ct, ct_r),
            ("Local Volumes", len(d.volumes), len([v for v, _ in vols if d.volume_users(v.name)]), vb, vb_r),
            ("Build Cache", len(d.build_cache), 0, cache, cache)]


@cmd("system")
def docker_system(ctx, o, args):
    d = daemon_of(ctx.world, ctx.host)
    sub = args[0] if args else ""
    rest = args[1:]
    if sub == "df":
        opts, r = flags(rest, {"-v|--verbose": None, "--format": "v"}, stop_at_positional=False)
        rows = []
        for kind, total, active, size, rec in disk_usage(d):
            pct = f" ({100 * rec // size if size else 0}%)"
            rows.append([kind, str(total), str(active), human_size(size) if size else "0B",
                         (human_size(rec) if rec else "0B") + (pct if size else "")])
        if opts.get("format"):
            for kind, total, active, size, rec in disk_usage(d):
                ctx.w(gotmpl.render(opts["format"], {"Type": kind, "TotalCount": total, "Active": active,
                                                     "Size": human_size(size) if size else "0B",
                                                     "Reclaimable": human_size(rec) if rec else "0B"}) + "\n")
            return 0
        ctx.w(gotmpl.tabulate([["TYPE", "TOTAL", "ACTIVE", "SIZE", "RECLAIMABLE"]] + rows))
        return 0
    if sub == "prune":
        opts, r = flags(rest, {"-a|--all": None, "-f|--force": None, "--volumes": None, "--filter": "m"},
                        stop_at_positional=False)
        if not opts.get("force"):
            what = ["  - all stopped containers", "  - all networks not used by at least one container"]
            if opts.get("volumes"):
                what.append("  - all anonymous volumes not used by at least one container")
            what.append("  - all images without at least one container associated to them" if opts.get("all") else
                        "  - all dangling images")
            what.append("  - unused build cache" if opts.get("all") else "  - all dangling build cache")
            ok = yield from confirm(ctx, "WARNING! This will remove:\n" + "\n".join(what) + "\n\nAre you sure you want "
                                                                                           "to continue? [y/N] ")
            if not ok:
                return 0
        total = 0
        lines = []
        cs = [c for c in list(d.containers.values()) if not c.running]
        for c in cs:
            d.remove(c)
        if cs:
            lines.append("Deleted Containers:\n" + "".join(c.id + "\n" for c in cs) + "\n")
        nets = [n for n in list(d.networks.values()) if n.name not in ("bridge", "host", "none") and
                not any(n.name in c.networks for c in d.containers.values())]
        for n in nets:
            del d.networks[n.name]
        if nets:
            lines.append("Deleted Networks:\n" + "".join(n.name + "\n" for n in nets) + "\n")
        if opts.get("volumes"):
            vs = [v for v in list(d.volumes.values()) if not d.volume_users(v.name) and (v.anonymous or opts.get("all"))]
            for v in vs:
                try:
                    total += tree_bytes(d.host.fs.lookup(v.mountpoint))
                except FSError:
                    pass
                d.remove_volume(v.name)
            if vs:
                lines.append("Deleted Volumes:\n" + "".join(v.name + "\n" for v in vs) + "\n")
        used = {c.image.id for c in d.containers.values()}
        imgs = [i for i in list(d.images.values()) if i.id not in used and (opts.get("all") or not i.repo_tags)]
        if imgs:
            out = []
            for i in imgs:
                for t in list(i.repo_tags):
                    d.untag(t)
                    out.append(f"untagged: {t}\n")
                d.images.pop(i.id, None)
                out.append(f"deleted: {i.id}\n")
                total += i.size
            lines.append("Deleted Images:\n" + "".join(out) + "\n")
        cache = sum(e[1] for e in d.build_cache.values())
        if d.build_cache:
            lines.append("Deleted build cache objects:\n" + "".join(k[:25] + "\n" for k in list(d.build_cache)[:8])
                         + "\n")
            d.build_cache.clear()
            total += cache
        ctx.w("".join(lines) + f"Total reclaimed space: {human_size(total) if total else '0B'}\n")
        return 0
    if sub == "info":
        from .cli import COMMANDS
        r = COMMANDS["info"](ctx, o, rest)
        if hasattr(r, "send"):
            r = yield from r
        return r
    if sub == "events":
        from .cli import COMMANDS
        r = COMMANDS["events"](ctx, o, rest)
        if hasattr(r, "send"):
            r = yield from r
        return r
    ctx.w("\nUsage:  docker system COMMAND\n\nManage Docker\n\nCommands:\n  df          Show docker disk usage\n  "
          "events      Get real time events from the server\n  info        Display system-wide information\n  prune     "
          "  Remove unused data\n\nRun 'docker system COMMAND --help' for more information on a command.\n")
    return 0 if not sub else 1


@cmd("info")
def docker_info(ctx, o, args):
    opts, r = flags(args, {"-f|--format": "v"}, stop_at_positional=False)
    d = daemon_of(ctx.world, ctx.host)
    h = d.host
    runtimes = {"io.containerd.runc.v2": {}, "runc": {}}
    if d.gpu_toolkit and h.gpus:
        runtimes["nvidia"] = {}
    info = {"ID": d._id("daemon")[:36], "Containers": len(d.containers),
            "ContainersRunning": len([c for c in d.containers.values() if c.state == "running"]),
            "ContainersPaused": len([c for c in d.containers.values() if c.state == "paused"]),
            "ContainersStopped": len([c for c in d.containers.values() if c.state in ("exited", "created")]),
            "Images": len(d.images), "Driver": "overlay2", "CgroupDriver": "systemd", "CgroupVersion": "2",
            "KernelVersion": h.kernel_release, "OperatingSystem": "Ubuntu 24.04.1 LTS", "OSType": "linux",
            "Architecture": "x86_64", "NCPU": h.cpus, "MemTotal": h.mem_kib * 1024, "Name": h.name,
            "ServerVersion": ENGINE, "DockerRootDir": "/var/lib/docker", "Runtimes": runtimes,
            "DefaultRuntime": "runc", "LoggingDriver": "json-file", "Swarm": {"LocalNodeState": "inactive"},
            "IndexServerAddress": "https://index.docker.io/v1/"}
    if opts.get("format"):
        ctx.w(gotmpl.render(opts["format"], info) + "\n")
        return 0
    mem = h.mem_kib * 1024 / 1024 ** 3
    ctx.w(f"Client: Docker Engine - Community\n Version:    {ENGINE}\n Context:    default\n Debug Mode: false\n Plugins:\n"
          f"  buildx: Docker Buildx (Docker Inc.)\n    Version:  v0.17.1\n    Path:     /usr/libexec/docker/cli-plugins/"
          f"docker-buildx\n  compose: Docker Compose (Docker Inc.)\n    Version:  v2.29.7\n    Path:     "
          f"/usr/libexec/docker/cli-plugins/docker-compose\n\nServer:\n Containers: {info['Containers']}\n  Running: "
          f"{info['ContainersRunning']}\n  Paused: {info['ContainersPaused']}\n  Stopped: {info['ContainersStopped']}\n"
          f" Images: {info['Images']}\n Server Version: {ENGINE}\n Storage Driver: overlay2\n Logging Driver: json-file\n"
          f" Cgroup Driver: systemd\n Cgroup Version: 2\n Runtimes: {' '.join(sorted(runtimes))}\n Default Runtime: "
          f"runc\n Kernel Version: {h.kernel_release}\n Operating System: Ubuntu 24.04.1 LTS\n OSType: linux\n "
          f"Architecture: x86_64\n CPUs: {h.cpus}\n Total Memory: {mem:.2f}GiB\n Name: {h.name}\n Docker Root Dir: "
          f"/var/lib/docker\n Debug Mode: false\n Experimental: false\n Insecure Registries:\n  127.0.0.0/8\n Live "
          f"Restore Enabled: false\n")
    return 0
    yield


@cmd("events")
def docker_events(ctx, o, args):
    opts, r = flags(args, {"--since": "v", "--until": "v", "-f|--filter": "m", "--format": "v"},
                    stop_at_positional=False)
    d = daemon_of(ctx.world, ctx.host)
    import datetime
    seen = 0
    if opts.get("since"):
        for t, text in d.events:
            ts = datetime.datetime.fromtimestamp(t / NS, tz=datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")
            ctx.w(f"{ts}000Z {text}\n")
    seen = len(d.events)
    if opts.get("until"):
        return 0
    fg = ctx.foreground()
    deadline = ctx.now + ctx.world.ctrl_c_after * NS if fg else float("inf")
    while True:
        for t, text in d.events[seen:]:
            ts = datetime.datetime.fromtimestamp(t / NS, tz=datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")
            ctx.w(f"{ts}000Z {text}\n")
        seen = len(d.events)
        yield from ctx.flush()
        if ctx.now >= deadline:
            ctx.world.tty.write("echo", b"^C\n")
            return 130
        nxt = ctx.now + int(0.25 * NS)
        if deadline != float("inf"):
            nxt = min(nxt, int(deadline))
        yield ("sleep", nxt)


@cmd("builder")
def docker_builder(ctx, o, args):
    d = daemon_of(ctx.world, ctx.host)
    sub = args[0] if args else ""
    if sub == "prune":
        opts, r = flags(args[1:], {"-a|--all": None, "-f|--force": None}, stop_at_positional=False)
        if not opts.get("force"):
            ok = yield from confirm(ctx, "WARNING! This will remove all dangling build cache. Are you sure you want "
                                         "to continue? [y/N] ")
            if not ok:
                return 0
        total = sum(e[1] for e in d.build_cache.values())
        n = len(d.build_cache)
        d.build_cache.clear()
        d.cache_mounts.clear()
        ctx.w(f"Total:\t{human_size(total) if total else '0B'}\n")
        return 0
    if sub == "ls" or sub == "du":
        ctx.w(f"Reclaimable:\t{human_size(sum(e[1] for e in d.build_cache.values()))}\nTotal:\t\t"
              f"{human_size(sum(e[1] for e in d.build_cache.values()))}\n")
        return 0
    ctx.w("\nUsage:  docker builder COMMAND\n\nManage builds\n\nCommands:\n  build       Build an image from a "
          "Dockerfile\n  prune       Remove build cache\n")
    return 0


# ---------------------------------------------------------------------------- login / logout

@cmd("login")
def docker_login(ctx, o, args):
    opts, r = flags(args, {"-u|--username": "v", "-p|--password": "v", "--password-stdin": None},
                    stop_at_positional=False)
    d = daemon_of(ctx.world, ctx.host)
    server = r[0] if r else "docker.io"
    reg = "docker.io" if server in ("docker.io", "index.docker.io", "registry-1.docker.io", "https://index.docker.io/v1/") \
        else server.replace("https://", "").rstrip("/")
    user = opts.get("username")
    pw = opts.get("password")
    if opts.get("password-stdin"):
        if not user:
            raise DockerError("Must provide --username with --password-stdin", 1)
        data = yield from ctx.read_fd(0)
        pw = data.decode("utf-8", "replace").strip()
    if pw and opts.get("password"):
        o.err("WARNING! Using --password via the CLI is insecure. Use --password-stdin.\n")
    if not user:
        ctx.w(f"Username: ")
        yield from ctx.flush()
        from .cli import _readline
        user = (yield from _readline(ctx)).strip()
    if not pw:
        ctx.w("Password: ")
        yield from ctx.flush()
        from .cli import _readline
        pw = (yield from _readline(ctx)).strip()
        ctx.w("\n")
    ok = check_credentials(ctx.world, reg, user, pw)
    if not ok:
        raise DockerError("Error response from daemon: Get \"https://" + (reg if reg != "docker.io" else
                                                                           "registry-1.docker.io") +
                          "/v2/\": unauthorized: incorrect username or password" if reg == "docker.io" else
                          f"Error response from daemon: Get \"https://{reg}/v2/\": denied: denied", 1)
    d.auth[reg] = user
    home = ctx.env.get("HOME", "/root")
    cfg_path = f"{home}/.docker/config.json"
    try:
        cfg = json.loads(bytes(ctx.fs.lookup(cfg_path).data).decode())
    except (FSError, ValueError):
        cfg = {}
    import base64
    key = "https://index.docker.io/v1/" if reg == "docker.io" else reg
    cfg.setdefault("auths", {})[key] = {"auth": base64.b64encode(f"{user}:{pw}".encode()).decode()}
    ctx.fs.mkdir(f"{home}/.docker", "/", 0o700, ctx.uid, ctx.proc.gid, parents=True)
    ctx.fs.write_file(cfg_path, json.dumps(cfg, indent=1) + "\n", "/", 0o600, ctx.uid, ctx.proc.gid)
    o.err(f"WARNING! Your password will be stored unencrypted in {cfg_path}.\nConfigure a credential helper to "
          f"remove this warning. See\nhttps://docs.docker.com/engine/reference/commandline/login/#credential-stores\n\n")
    ctx.w("Login Succeeded\n")
    return 0


def check_credentials(world, reg, user, pw):
    if reg == "ghcr.io":
        gh = world.__dict__.get("github")
        if gh is None:
            return False
        acc = gh.accounts.get(user) or next((a for a in gh.accounts.values() if a.login.lower() == user.lower()), None)
        return acc is not None and pw in acc.tokens
    accounts = world.__dict__.get("docker_hub_accounts", {})
    return accounts.get(user) == pw


@cmd("logout")
def docker_logout(ctx, o, args):
    d = daemon_of(ctx.world, ctx.host)
    server = args[0] if args else "docker.io"
    reg = "docker.io" if server in ("docker.io", "index.docker.io") else server
    shown = "https://index.docker.io/v1/" if reg == "docker.io" else reg
    if reg not in d.auth:
        ctx.w(f"Not logged in to {shown}\n")
        return 0
    del d.auth[reg]
    home = ctx.env.get("HOME", "/root")
    try:
        cfg = json.loads(bytes(ctx.fs.lookup(f"{home}/.docker/config.json").data).decode())
        cfg.get("auths", {}).pop(shown if reg == "docker.io" else reg, None)
        ctx.fs.write_file(f"{home}/.docker/config.json", json.dumps(cfg, indent=1) + "\n", "/", 0o600, ctx.uid,
                          ctx.proc.gid)
    except (FSError, ValueError):
        pass
    ctx.w(f"Removing login credentials for {shown}\n")
    return 0
    yield


# ---------------------------------------------------------------------------- scout

CVES = {"debian": (0, 1, 2, 23), "ubuntu24": (0, 0, 2, 7), "ubuntu22": (0, 1, 5, 18), "alpine": (0, 0, 0, 1),
        "busybox": (0, 0, 0, 0), None: (0, 0, 0, 0)}
OLD = {"python:3.11-slim": (1, 4, 9, 30), "python:3.12": (2, 11, 41, 160), "debian:bookworm": (0, 2, 9, 60),
       "nginx:1.27": (0, 2, 6, 31), "postgres:16": (0, 3, 8, 40)}


@cmd("scout")
def docker_scout(ctx, o, args):
    from .images import IMAGES
    d = daemon_of(ctx.world, ctx.host)
    sub = args[0] if args else ""
    rest = [a for a in args[1:] if not a.startswith("-")]
    if sub not in ("cves", "quickview", "qv") or not rest:
        ctx.w("\nUsage:  docker scout COMMAND\n\nCommand line tool for Docker Scout\n\nCommands:\n  cves        "
              "Display CVEs identified in a software artifact\n  quickview   Quick overview of an image\n")
        return 0
    img = d.find_image(rest[0])
    if img is None:
        raise DockerError(f"ERROR image not found: {rest[0]}", 1)
    base = img.base
    spec = IMAGES.get(base or "", {})
    c, h, m, lo = OLD.get(base, CVES.get(spec.get("distro"), (0, 0, 0, 0)))
    extra = sum(1 for x in img.history if "pip install" in x[0])
    m += extra
    total = c + h + m + lo
    ctx.w(f"    ✓ SBOM of image already cached, {120 + 7 * len(img.history)} packages indexed\n")
    ctx.w(f"    {'✗' if total else '✓'} Detected {total} vulnerable packages with a total of {total} "
          f"vulnerabilities\n\n")
    ctx.w(f"  Target   │  {(img.repo_tags[0] if img.repo_tags else img.short)}\n    digest │  {img.short}\n"
          f"  Base image │  {base or 'unknown'}\n\n")
    ctx.w(f"{total} vulnerabilities found in {total} packages\n  CRITICAL  {c}\n  HIGH      {h}\n  MEDIUM    {m}\n"
          f"  LOW       {lo}\n")
    if base in OLD:
        better = {"python:3.11-slim": "python:3.12-slim", "python:3.12": "python:3.12-slim",
                  "debian:bookworm": "debian:bookworm-slim", "nginx:1.27": "nginx:alpine",
                  "postgres:16": "postgres:16"}[base]
        ctx.w(f"\nRecommended fix: change the base image to {better}\n")
    return 0
    yield


# ---------------------------------------------------------------------------- systemctl (the docker service)

@util("systemctl")
def u_systemctl(ctx):
    a = [x for x in ctx.argv[1:] if not x.startswith("-")]
    if not a:
        ctx.e("Too few arguments.\n")
        return 1
    verb, units = a[0], [u.removesuffix(".service") for u in a[1:]]
    if verb in ("start", "stop", "restart", "enable", "disable", "reload") and ctx.uid != 0:
        ctx.e(f"Failed to {verb} {units[0] if units else ''}.service: Access denied\nSee system logs and 'systemctl "
              f"status {units[0] if units else ''}.service' for details.\n")
        return 1
    unit = units[0] if units else ""
    if unit not in ("docker", "containerd", "docker.socket", "ssh", "cron"):
        ctx.e(f"Unit {unit}.service could not be found.\n" if verb == "status" else
              f"Failed to {verb} {unit}.service: Unit {unit}.service not found.\n")
        return 4 if verb == "status" else 5
    if unit not in ("docker", "docker.socket", "containerd"):
        if verb == "status":
            ctx.w(f"● {unit}.service\n     Loaded: loaded\n     Active: active (running)\n")
        return 0
    d = daemon_of(ctx.world, ctx.host)
    if not d.installed:
        ctx.e(f"Unit docker.service could not be found.\n" if verb == "status" else
              f"Failed to {verb} docker.service: Unit docker.service not found.\n")
        return 4 if verb == "status" else 5
    if verb in ("start", "restart", "reload"):
        if verb == "restart":
            for c in list(d.containers.values()):
                if c.running:
                    d.kill(c, 15)
                    c.state = "exited"
        d.running = True
        if d.host.extra.get("nvidia_configured"):
            d.gpu_toolkit = True
        for c in list(d.containers.values()):
            pol = c.cfg.get("restart", "no")
            if not c.running and pol in ("always",) or (pol == "unless-stopped" and not c.stopping):
                try:
                    d.start(c)
                except DockerError:
                    pass
        yield ("sleep", ctx.now + int(1.2 * NS))
        return 0
    if verb == "stop":
        for c in list(d.containers.values()):
            if c.running:
                d.kill(c, 15)
        d.running = False
        return 0
    if verb in ("enable", "disable"):
        if verb == "enable":
            ctx.e("Synchronizing state of docker.service with SysV service script with /usr/lib/systemd/"
                  "systemd-sysv-install.\nExecuting: /usr/lib/systemd/systemd-sysv-install enable docker\n")
        return 0
    if verb == "is-active":
        ctx.w("active\n" if d.running else "inactive\n")
        return 0 if d.running else 3
    if verb == "status":
        state = "active (running)" if d.running else "inactive (dead)"
        ctx.w(f"● docker.service - Docker Application Container Engine\n     Loaded: loaded (/usr/lib/systemd/system/"
              f"docker.service; enabled; preset: enabled)\n     Active: {state}\n       Docs: https://docs.docker.com\n"
              + (f"   Main PID: 1234 (dockerd)\n      Tasks: 18\n     CGroup: /system.slice/docker.service\n             "
                 f"└─1234 /usr/bin/dockerd -H fd:// --containerd=/run/containerd/containerd.sock\n" if d.running else ""))
        return 0 if d.running else 3
    return 0
