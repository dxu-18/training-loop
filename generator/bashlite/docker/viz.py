"""What the terminal's Docker panel shows: images, containers (state, ports, health), volumes, networks, compose
projects and the last build's steps — JSON-ready, recomputed after each command."""

from __future__ import annotations

from .engine import ago, human_size


def snapshot(world) -> dict | None:
    host = world.local
    d = host.extra.get("dockerd")
    if d is None:
        return None
    now = world.kernel.now
    imgs = []
    for img in sorted(d.images.values(), key=lambda i: -i.created):
        for t in (img.repo_tags or ["<none>:<none>"]):
            imgs.append({"ref": t, "id": img.short, "size": human_size(img.size), "built": img.base is None or
                         t.split(":")[0] not in (img.base or "").split(":")[0], "age": ago(now - img.created)})
    cs = []
    for c in sorted(d.containers.values(), key=lambda c: -c.created):
        cs.append({"name": c.name, "id": c.short, "image": c.cfg.get("image_ref"), "state": c.state,
                   "status": c.status_text(now), "exit": c.exit_code if c.state == "exited" else None,
                   "oom": c.oom, "health": c.health,
                   "ports": [f"{h}->{p}" if ip in ("0.0.0.0", "") else f"{ip}:{h}->{p}" for ip, h, p, _ in c.ports],
                   "mounts": [f"{m.get('name') or m.get('source') or 'tmpfs'}:{m['dest']}{':ro' if m.get('ro') else ''}"
                              for m in c.mounts if not (m['type'] == 'volume' and d.volumes.get(m.get('name')) is not None
                                                        and d.volumes[m['name']].anonymous)],
                   "nets": sorted(n for n in c.networks if n != "bridge"),
                   "project": c.labels.get("com.docker.compose.project"),
                   "gpus": "all" if c.cfg.get("gpus") == "all" else c.cfg.get("gpus")})
    vols = [{"name": v.name, "used": [c.name for c in d.volume_users(v.name)]}
            for v in sorted(d.volumes.values(), key=lambda v: v.name) if not v.anonymous]
    nets = [{"name": n.name, "members": sorted(c.name for c in d.containers.values() if n.name in c.networks)}
            for n in sorted(d.networks.values(), key=lambda n: n.name) if n.name not in ("bridge", "host", "none")]
    build = getattr(d, "last_build", None)
    return {"running": d.running and d.installed, "images": imgs[:30], "containers": cs[:30], "volumes": vols,
            "networks": nets, "build": build, "auth": sorted(d.auth)}
