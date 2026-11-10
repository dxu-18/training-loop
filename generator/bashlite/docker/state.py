"""The Docker problems' checker: print the engine's state so that two sessions that reach the same state print the
same text, whatever the timing and whatever ids Docker generated.

    /usr/lib/bashlite/dockerstate SECTION...

Sections: images | image=REF | containers (named ones) | containers-all (unnamed ones too, by image and command) |
          container=NAME | volumes | volume=NAME (and its files) | networks | registry | auth | daemon | compose=PROJECT
"""

from __future__ import annotations

import hashlib

from ..exec import util
from ..vfs import FSError
from .engine import daemon_of


def _img_ref(d, img):
    return img.repo_tags[0] if img.repo_tags else "<none>"


def image_lines(d, img, ref=None):
    cfg = img.config
    base_env = set()
    if img.base:
        from .images import IMAGES
        base_env = set(IMAGES.get(img.base, {}).get("env") or [])
    env = [e for e in (cfg.get("Env") or []) if e not in base_env and not e.startswith("PATH=")]
    out = [f"image {ref or _img_ref(d, img)}"]
    out.append(f"  base={img.base} user={cfg.get('User') or 'root'} workdir={cfg.get('WorkingDir') or '/'}")
    out.append(f"  entrypoint={cfg.get('Entrypoint')} cmd={cfg.get('Cmd')}")
    if env:
        out.append(f"  env={sorted(env)}")
    if cfg.get("ExposedPorts"):
        out.append(f"  expose={sorted(cfg['ExposedPorts'])}")
    if cfg.get("Labels"):
        out.append(f"  labels={sorted(cfg['Labels'].items())}")
    if cfg.get("Healthcheck"):
        out.append(f"  healthcheck={cfg['Healthcheck'].get('Test')}")
    if cfg.get("Volumes"):
        out.append(f"  volumes={sorted(cfg['Volumes'])}")
    out.append(f"  platform=linux/{img.arch}")
    return out


def container_lines(d, c, named=True):
    cfg = c.cfg
    name = c.name if named else "<unnamed>"
    state = c.state if c.state != "restarting" else "restarting"
    out = [f"container {name} image={cfg.get('image_ref')} state={state}" +
           (f" exit={c.exit_code}" if c.state == "exited" else "") + (" oom" if c.oom else "") +
           (f" health={c.health}" if c.health else "")]
    if cfg.get("cmd") is not None:
        out.append(f"  cmd={cfg['cmd']}")
    if cfg.get("entrypoint_set"):
        out.append(f"  entrypoint={cfg.get('entrypoint')}")
    if cfg.get("env"):
        out.append(f"  env={sorted(cfg['env'])}")
    if c.ports:
        out.append(f"  ports={sorted(f'{ip}:{h}->{p}/{pr}' for ip, h, p, pr in c.ports)}")
    for m in sorted(c.mounts, key=lambda m: m["dest"]):
        if m["type"] == "volume" and d.volumes.get(m.get("name")) is not None and d.volumes[m["name"]].anonymous:
            out.append(f"  mount volume <anonymous> -> {m['dest']}")
            continue
        src = m.get("name") if m["type"] == "volume" else m.get("source", "")
        out.append(f"  mount {m['type']} {src} -> {m['dest']}{' (ro)' if m.get('ro') else ''}")
    nets = sorted(n for n in c.networks if n != "bridge") if (cfg.get("network") or "bridge") == "bridge" else \
        sorted(c.networks) or [cfg.get("network")]
    if nets:
        out.append(f"  networks={nets}")
    for k, v in (("user", cfg.get("user")), ("workdir", cfg.get("workdir")), ("restart", cfg.get("restart")),
                 ("gpus", cfg.get("gpus")), ("shm", f"{cfg['shm_mb']}m" if cfg.get("shm_mb") else None),
                 ("memory", cfg.get("memory")), ("cpus", cfg.get("cpus")), ("init", cfg.get("init")),
                 ("autoremove", cfg.get("rm") or None), ("hostname", cfg.get("hostname"))):
        if v:
            out.append(f"  {k}={v}")
    hosts = cfg.get("add_hosts")
    if hosts:
        out.append(f"  add_hosts={sorted(hosts)}")
    labels = {k: v for k, v in c.labels.items() if not k.startswith("com.docker.compose.project.")}
    if labels:
        out.append(f"  labels={sorted(labels.items())}")
    return out


def tree_lines(fs, node, prefix=""):
    out = []
    for name in sorted(node.entries):
        ch = node.entries[name]
        path = f"{prefix}/{name}"
        if ch.type == "d":
            out.append(f"    {path}/ uid={ch.uid}")
            out += tree_lines(fs, ch, path)
        elif ch.type == "f":
            h = hashlib.sha1(bytes(ch.data[:1_000_000])).hexdigest()[:10]
            out.append(f"    {path} {len(ch.data)}B {h} uid={ch.uid}")
        else:
            out.append(f"    {path} -> {ch.target}")
    return out


@util("bashlite-dockerstate")
def u_dockerstate(ctx):
    d = daemon_of(ctx.world, ctx.host)
    out = []
    for sec in ctx.argv[1:] or ["images", "containers", "volumes", "networks"]:
        key, _, val = sec.partition("=")
        if key == "daemon":
            out.append(f"daemon installed={d.installed} running={d.running} gpu-toolkit={d.gpu_toolkit}")
        elif key == "images":
            for t in sorted(d.tags):
                img = d.images.get(d.tags[t])
                if img is not None and img.base is None or (img is not None and val == "all"):
                    out += image_lines(d, img, t)
                elif img is not None:
                    out.append(f"image {t} (pulled, base={img.base})")
            dangling = [i for i in d.images.values() if not i.repo_tags]
            if dangling:
                out.append(f"dangling images: {len(dangling)}")
        elif key == "image":
            img = d.find_image(val)
            out += image_lines(d, img, val) if img is not None else [f"image {val}: none"]
        elif key in ("containers", "containers-all"):
            named = [c for c in d.containers.values() if c.cfg.get("name") or c.labels.get("com.docker.compose.project")]
            for c in sorted(named, key=lambda c: c.name):
                out += container_lines(d, c)
            others = [c for c in d.containers.values() if c not in named]
            if key == "containers-all":
                rows = sorted("\n".join(container_lines(d, c, named=False)) for c in others)
                out += rows
            elif others:
                out.append(f"unnamed containers: {len(others)}")
        elif key == "container":
            c = d.find_container(val)
            out += container_lines(d, c) if c is not None else [f"container {val}: none"]
        elif key == "volumes":
            for name in sorted(d.volumes):
                v = d.volumes[name]
                if v.anonymous:
                    continue
                out.append(f"volume {name}")
            anon = [v for v in d.volumes.values() if v.anonymous]
            if anon:
                out.append(f"anonymous volumes: {len(anon)}")
        elif key == "volume":
            v = d.volumes.get(val)
            if v is None:
                out.append(f"volume {val}: none")
                continue
            out.append(f"volume {val}")
            try:
                out += tree_lines(d.host.fs, d.host.fs.lookup(v.mountpoint))
            except FSError:
                pass
        elif key == "networks":
            for n in sorted(d.networks.values(), key=lambda n: n.name):
                if n.name in ("bridge", "host", "none"):
                    continue
                members = sorted(c.name for c in d.containers.values() if n.name in c.networks)
                out.append(f"network {n.name} driver={n.driver} internal={n.internal} containers={members}")
        elif key == "registry":
            from .cmd_images import registries
            for reg, repos in sorted(registries(ctx.world).items()):
                for name in sorted(repos):
                    img = repos[name]
                    out.append(f"pushed {name}")
                    out += ["  " + x for x in image_lines(d, img, name)[1:]]
                    if getattr(img, "platforms", None) and len(img.platforms) > 1:
                        out.append(f"  platforms={img.platforms}")
        elif key == "auth":
            out.append(f"logged in: {sorted(d.auth.items())}")
        elif key == "compose":
            cs = [c for c in d.containers.values() if c.labels.get("com.docker.compose.project") == val]
            for c in sorted(cs, key=lambda c: c.name):
                out += container_lines(d, c)
            vols = sorted(v.name for v in d.volumes.values() if v.labels.get("com.docker.compose.project") == val)
            nets = sorted(n.name for n in d.networks.values() if n.labels.get("com.docker.compose.project") == val)
            out.append(f"compose {val} volumes={vols} networks={nets}")
        else:
            out.append(f"unknown section {key}")
    ctx.w("\n".join(out) + "\n")
    return 0
    yield
