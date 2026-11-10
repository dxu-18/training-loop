"""Image commands and the registries: pull, push, images, tag, rmi, image prune, history, image inspect,
save/load, commit.

Registries (Docker Hub = docker.io, GitHub Container Registry = ghcr.io, quay.io) hold the catalog images and
whatever users push. Pushing needs `docker login` to that registry and a name in your namespace: docker.io/<you>/x
or ghcr.io/<you>/x (GHCR logs in with a GitHub-lite token)."""

from __future__ import annotations

import hashlib
import json
import re

from ..kernel import NS
from ..vfs import FSError, LazyData
from . import gotmpl
from .cli import Out, cmd, confirm, flags, host_path
from .engine import DockerError, Image, ago, daemon_of, human_size
from .fsutil import clone_fs
from .images import ALIASES, IMAGES, digest_of, layer_ids


def normalize(ref: str) -> str:
    """'nginx' -> 'nginx:latest', 'docker.io/library/x:1' -> 'x:1' (the name docker shows locally)."""
    r = ref.split("@", 1)[0]
    for pre in ("docker.io/library/", "index.docker.io/library/", "docker.io/"):
        if r.startswith(pre):
            r = r[len(pre):]
    last = r.rsplit("/", 1)[-1]
    return r if ":" in last else r + ":latest"


def registry_of(name: str) -> str:
    first = name.split("/", 1)[0]
    return first if "/" in name and ("." in first or ":" in first or first == "localhost") else "docker.io"


def remote_path(name: str) -> str:
    """How the registry names a repository: library/nginx, ada/app, ghcr.io's ada/app."""
    reg = registry_of(name)
    repo = name.rsplit(":", 1)[0]
    if reg != "docker.io":
        repo = repo.split("/", 1)[1]
    elif "/" not in repo:
        repo = "library/" + repo
    return repo


def registries(world) -> dict:
    r = world.__dict__.get("registries")
    if r is None:
        r = {"docker.io": {}, "ghcr.io": {}, "quay.io": {}}
        world.registries = r
    return r


def recipe_of(name: str):
    n = normalize(name)
    if n in IMAGES:
        return n
    if n in ALIASES:
        return ALIASES[n]
    base = n.rsplit(":", 1)[0]
    if n.endswith(":latest") and base in ALIASES:
        return ALIASES[base]
    return None


def lookup_remote(world, name: str):
    """(image or catalog recipe name, digest) available from a registry, or (None, None)."""
    n = normalize(name)
    reg = registry_of(n)
    pushed = registries(world).get(reg, {}).get(n)
    if pushed is not None:
        return pushed, pushed.digest
    key = n[len("quay.io/"):] if n.startswith("quay.io/") else n
    rec = recipe_of(n) or (recipe_of("quay.io/" + key) if reg == "quay.io" else None)
    if rec is None and n.startswith("quay.io/"):
        rec = recipe_of(n)
    if rec is not None:
        return rec, digest_of(rec)
    return None, None


def pull_image(ctx, o, d, ref, quiet=False, platform=None, from_run=False, show_name=True):
    """docker pull (generator): the registry's progress lines; -> the local Image."""
    name = normalize(ref)
    src, digest = lookup_remote(ctx.world, name)
    if src is None:
        reg = registry_of(name)
        what = remote_path(name)
        msg = (f"Error response from daemon: pull access denied for {name.rsplit(':', 1)[0]}, repository does not "
               f"exist or may require 'docker login': denied: requested access to the resource is denied"
               if reg == "docker.io" else
               f"Error response from daemon: Head \"https://{reg}/v2/{what}/manifests/{name.rsplit(':', 1)[1]}\": "
               f"denied")
        if reg == "docker.io" and recipe_of(name.rsplit(":", 1)[0] + ":latest") and not name.endswith(":latest"):
            msg = f"Error response from daemon: manifest for {name} not found: manifest unknown: manifest unknown"
        raise DockerError(("docker: " if from_run else "") + msg + ("\nSee 'docker run --help'." if from_run else ""),
                          125 if from_run else 1)
    tag = name.rsplit(":", 1)[1]
    arch = (platform or "linux/amd64").split("/")[1] if platform else "amd64"
    if isinstance(src, str):
        img = d.find_image(f"sha256:{hashlib.sha256(f'image:{src}'.encode()).hexdigest()}") \
            if arch == "amd64" else None
        if img is None:
            img = d.catalog_image(src)
            if arch != "amd64":
                img.id = "sha256:" + hashlib.sha256(f"image:{src}:{arch}".encode()).hexdigest()
                img.arch = arch
        n_layers = IMAGES[src].get("layers", 1)
        size = IMAGES[src]["size"]
        lids = layer_ids(src, n_layers)
    else:
        img = src.copy_for(d)
        n_layers = max(1, len([h for h in img.history if not h[2]]))
        size = img.size
        lids = layer_ids(src.id, n_layers)
    had = d.find_image(name)
    if not quiet and show_name and not from_run and ":" not in ref.rsplit("/", 1)[-1] and "@" not in ref:
        o.out("Using default tag: latest\n")
    if had is not None and had.id == img.id:
        if not quiet:
            o.out(f"{tag}: Pulling from {remote_path(name)}\nDigest: {digest}\nStatus: Image is up to date for "
                  f"{name}\n")
            if not from_run:
                o.out(f"{_full(name)}\n")
        elif not from_run:
            o.out(f"{_full(name)}\n")
        return had
    # time: about 40 MB/s
    secs = max(0.3, size / 40e6)
    yield ("sleep", ctx.now + int(min(secs, 60) * NS))
    if not quiet:
        out = (o.err if from_run else o.out)
        lines = [f"{tag}: Pulling from {remote_path(name)}\n"]
        known = {lid[:12] for i in d.images.values() for lid in getattr(i, "layer_ids", [])}
        for lid in lids:
            lines.append(f"{lid[:12]}: {'Already exists' if lid[:12] in known else 'Pull complete'}\n")
        lines.append(f"Digest: {digest}\nStatus: Downloaded newer image for {name}\n")
        out("".join(lines))
        if not from_run:
            o.out(f"{_full(name)}\n")
    elif not from_run:
        o.out(f"{_full(name)}\n")
    img.layer_ids = [x[:12] for x in lids]
    d.add_image(img, [name])
    img.repo_digests = sorted(set(img.repo_digests + [f"{name.rsplit(':', 1)[0]}@{digest}"]))
    d.event(f"image pull {name}")
    return img


def _full(name):
    reg = registry_of(name)
    if reg == "docker.io":
        return "docker.io/" + (name if "/" in name else "library/" + name)
    return name


# ---------------------------------------------------------------------------- pull / push

@cmd("pull")
def docker_pull(ctx, o, args):
    opts, rest = flags(args, {"-q|--quiet": None, "--platform": "v", "-a|--all-tags": None},
                       stop_at_positional=False)
    d = daemon_of(ctx.world, ctx.host)
    if len(rest) != 1:
        raise DockerError("\"docker pull\" requires exactly 1 argument.\nSee 'docker pull --help'.\n\nUsage:  docker "
                          "pull [OPTIONS] NAME[:TAG|@DIGEST]\n\nDownload an image from a registry", 1)
    yield from pull_image(ctx, o, d, rest[0], quiet=bool(opts.get("quiet")), platform=opts.get("platform"))
    return 0


def _image_copy_for(self, d):
    img = Image(self.id, json.loads(json.dumps(self.config)), clone_fs(self.fs, lambda: d.world.kernel.now),
                self.size, list(self.history), self.created, self.arch, self.base)
    img.cuda = self.cuda
    img.platforms = list(self.platforms)
    return img


Image.copy_for = _image_copy_for


@cmd("push")
def docker_push(ctx, o, args):
    opts, rest = flags(args, {"-a|--all-tags": None, "-q|--quiet": None, "--platform": "v"}, stop_at_positional=False)
    d = daemon_of(ctx.world, ctx.host)
    if len(rest) != 1:
        raise DockerError("\"docker push\" requires exactly 1 argument.", 1)
    ref = rest[0]
    tags = []
    if opts.get("all-tags") or ":" not in ref.rsplit("/", 1)[-1]:
        base = normalize(ref).rsplit(":", 1)[0]
        tags = sorted(t for t in d.tags if t.rsplit(":", 1)[0] == base)
        if not opts.get("all-tags") and not tags:
            tags = [normalize(ref)]
        if not opts.get("all-tags") and normalize(ref) in d.tags:
            tags = [normalize(ref)]
    else:
        tags = [normalize(ref)]
    for name in tags:
        img = d.find_image(name)
        if img is None:
            raise DockerError(f"An image does not exist locally with the tag: {name.rsplit(':', 1)[0] if name.endswith(':latest') else name}", 1)
        yield from push_one(ctx, o, d, name, img, quiet=bool(opts.get("quiet")))
    return 0


def push_one(ctx, o, d, name, img, quiet=False):
    reg = registry_of(name)
    path = remote_path(name)
    owner = path.split("/")[0]
    user = d.auth.get(reg)
    if not quiet:
        o.out(f"The push refers to repository [{_full(name).rsplit(':', 1)[0]}]\n")
    layers = [h for h in img.history if not h[2]]
    lids = [hashlib.sha256(f"{img.id}:{i}".encode()).hexdigest()[:12] for i in range(len(layers))]
    if reg == "docker.io":
        ok = user is not None and owner == user
        if not ok:
            if not quiet:
                o.out("".join(f"{lid}: Preparing\n" for lid in lids[-4:]))
            raise DockerError("denied: requested access to the resource is denied", 1)
    else:
        if user is None:
            if not quiet:
                o.out("".join(f"{lid}: Preparing\n" for lid in lids[-4:]))
            raise DockerError("unauthorized: unauthenticated: User cannot be authenticated with the token provided.", 1)
        if owner.lower() != user.lower():
            raise DockerError(f"denied: permission_denied: The token provided does not match expected scopes.", 1)
    store = registries(ctx.world).setdefault(reg, {})
    have = {getattr(i, "layer_set", None) for i in store.values()}
    del have
    secs = max(0.3, min(30, img.size / 50e6))
    yield ("sleep", ctx.now + int(secs * NS))
    copy = img.copy_for(d)
    copy.digest = "sha256:" + hashlib.sha256(f"push:{img.id}".encode()).hexdigest()
    existed = name in store
    store[name] = copy
    if not quiet:
        out = []
        for lid in lids[-8:]:
            out.append(f"{lid}: {'Layer already exists' if existed else 'Pushed'}\n")
        tag = name.rsplit(":", 1)[1]
        out.append(f"{tag}: digest: {copy.digest} size: {512 + 270 * len(lids)}\n")
        o.out("".join(out))
    img.repo_digests = sorted(set(img.repo_digests + [f"{name.rsplit(':', 1)[0]}@{copy.digest}"]))
    d.event(f"image push {name}")


# ---------------------------------------------------------------------------- images

IMG_HEAD = {"Repository": "REPOSITORY", "Tag": "TAG", "ID": "IMAGE ID", "CreatedSince": "CREATED", "Size": "SIZE",
            "Digest": "DIGEST", "CreatedAt": "CREATED AT"}


def image_rows(d, all_=False, dangling=None, ref=None, now=0, digests=False):
    rows = []
    for img in sorted(d.images.values(), key=lambda i: -i.created):
        tags = img.repo_tags or (["<none>:<none>"])
        if dangling is True and img.repo_tags:
            continue
        if dangling is False and not img.repo_tags:
            continue
        if not img.repo_tags and not all_ and dangling is None and getattr(img, "intermediate", False):
            continue
        for t in tags:
            repo, _, tag = t.rpartition(":")
            if ref and not _ref_match(ref, repo, tag):
                continue
            rows.append({"Repository": repo, "Tag": tag, "ID": img.short, "CreatedSince": f"{ago(now - img.created)} ago",
                         "CreatedAt": str(img.created), "Size": human_size(img.size),
                         "Digest": (img.repo_digests[0].split("@")[1] if img.repo_digests else "<none>")})
    return rows


def _ref_match(ref, repo, tag):
    import fnmatch
    r, _, t = ref.partition(":")
    if r.startswith("docker.io/library/"):
        r = r[len("docker.io/library/"):]
    return fnmatch.fnmatch(repo, r) and (not t or fnmatch.fnmatch(tag, t))


@cmd("images")
def docker_images(ctx, o, args):
    opts, rest = flags(args, {"-a|--all": None, "-q|--quiet": None, "-f|--filter": "m", "--format": "v",
                              "--no-trunc": None, "--digests": None}, stop_at_positional=False)
    d = daemon_of(ctx.world, ctx.host)
    dangling = None
    ref = rest[0] if rest else None
    for f in opts.get("filter", []):
        k, _, v = f.partition("=")
        if k == "dangling":
            dangling = v == "true"
        elif k == "reference":
            ref = v
    rows = image_rows(d, bool(opts.get("all")), dangling, ref, ctx.now)
    if opts.get("quiet"):
        seen = []
        for r in rows:
            if r["ID"] not in seen:
                seen.append(r["ID"])
        ctx.w("".join(x + "\n" for x in seen))
        return 0
    fmt = opts.get("format")
    if fmt:
        if fmt == "json":
            ctx.w("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows))
        elif fmt.startswith("table"):
            ctx.w(gotmpl.table(fmt, rows, IMG_HEAD))
        else:
            ctx.w("".join(gotmpl.render(fmt.replace("\\t", "\t"), r) + "\n" for r in rows))
        return 0
    head = ["REPOSITORY", "TAG"] + (["DIGEST"] if opts.get("digests") else []) + ["IMAGE ID", "CREATED", "SIZE"]
    table = [head]
    for r in rows:
        table.append([r["Repository"], r["Tag"]] + ([r["Digest"]] if opts.get("digests") else []) +
                     [r["ID"], r["CreatedSince"], r["Size"]])
    ctx.w(gotmpl.tabulate(table))
    return 0
    yield


@cmd("tag")
def docker_tag(ctx, o, args):
    d = daemon_of(ctx.world, ctx.host)
    if len(args) != 2:
        raise DockerError("\"docker tag\" requires exactly 2 arguments.\nSee 'docker tag --help'.\n\nUsage:  docker tag "
                          "SOURCE_IMAGE[:TAG] TARGET_IMAGE[:TAG]\n\nCreate a tag TARGET_IMAGE that refers to "
                          "SOURCE_IMAGE", 1)
    img = d.find_image(args[0])
    if img is None:
        raise DockerError(f"Error response from daemon: No such image: {normalize(args[0])}", 1)
    t = args[1]
    if re.search(r"[A-Z]", t.split(":")[0]):
        raise DockerError(f"Error parsing reference: \"{t}\" is not a valid repository/tag: invalid reference format: "
                          f"repository name (library/{t.split(':')[0]}) must be lowercase", 1)
    d.tag(img, normalize(t))
    return 0
    yield


@cmd("rmi")
def docker_rmi(ctx, o, args):
    opts, rest = flags(args, {"-f|--force": None, "--no-prune": None}, stop_at_positional=False)
    d = daemon_of(ctx.world, ctx.host)
    if not rest:
        raise DockerError("\"docker rmi\" requires at least 1 argument.", 1)
    st = 0
    for ref in rest:
        img = d.find_image(ref)
        if img is None:
            o.err(f"Error response from daemon: No such image: {normalize(ref) if not re.fullmatch('[0-9a-f]+', ref) else ref}\n")
            st = 1
            continue
        by_id = not (normalize(ref) in img.repo_tags)
        users = d.image_in_use(img)
        if users and not opts.get("force"):
            c = users[0]
            if by_id or len(img.repo_tags) <= 1:
                o.err(f"Error response from daemon: conflict: unable to {'delete ' + img.short if by_id else 'remove repository reference ' + chr(34) + normalize(ref) + chr(34)} "
                      f"(must be forced) - container {c.short} is using its referenced image {img.short}\n"
                      if not by_id else
                      f"Error response from daemon: conflict: unable to delete {img.short} (cannot be forced) - image "
                      f"is being used by running container {c.short}\n" if c.running else
                      f"Error response from daemon: conflict: unable to delete {img.short} (must be forced) - image is "
                      f"being used by stopped container {c.short}\n")
                st = 1
                continue
        if by_id and len(img.repo_tags) > 1 and not opts.get("force"):
            o.err(f"Error response from daemon: conflict: unable to delete {img.short} (must be forced) - image is "
                  f"referenced in multiple repositories\n")
            st = 1
            continue
        if not by_id and len(img.repo_tags) > 1:
            d.untag(ref)
            ctx.w(f"Untagged: {normalize(ref)}\n")
            continue
        for t in list(img.repo_tags):
            d.untag(t)
            ctx.w(f"Untagged: {t}\n")
        for dg in img.repo_digests:
            ctx.w(f"Untagged: {dg}\n")
        d.images.pop(img.id, None)
        ctx.w(f"Deleted: {img.id}\n")
        d.event(f"image delete {img.id}")
    return st
    yield


@cmd("image-prune")
def docker_image_prune(ctx, o, args):
    opts, rest = flags(args, {"-a|--all": None, "-f|--force": None, "--filter": "m"}, stop_at_positional=False)
    d = daemon_of(ctx.world, ctx.host)
    if not opts.get("force"):
        q = ("WARNING! This will remove all images without at least one container associated to them.\nAre you sure "
             "you want to continue? [y/N] " if opts.get("all") else
             "WARNING! This will remove all dangling images.\nAre you sure you want to continue? [y/N] ")
        ok = yield from confirm(ctx, q)
        if not ok:
            return 0
    used = {c.image.id for c in d.containers.values()}
    gone = [i for i in list(d.images.values()) if i.id not in used and (opts.get("all") or not i.repo_tags)]
    total = 0
    lines = []
    for img in gone:
        for t in list(img.repo_tags):
            d.untag(t)
            lines.append(f"untagged: {t}\n")
        d.images.pop(img.id, None)
        lines.append(f"deleted: {img.id}\n")
        total += img.size
    if lines:
        ctx.w("Deleted Images:\n" + "".join(lines) + "\n")
    ctx.w(f"Total reclaimed space: {human_size(total) if total else '0B'}\n")
    return 0


@cmd("history")
def docker_history(ctx, o, args):
    opts, rest = flags(args, {"--no-trunc": None, "-q|--quiet": None, "--format": "v", "-H|--human": None},
                       stop_at_positional=False)
    d = daemon_of(ctx.world, ctx.host)
    if len(rest) != 1:
        raise DockerError("\"docker history\" requires exactly 1 argument.", 1)
    img = d.find_image(rest[0])
    if img is None:
        raise DockerError(f"Error response from daemon: No such image: {normalize(rest[0])}", 1)
    rows = []
    hist = list(reversed(img.history))
    for i, (by, size, empty, created) in enumerate(hist):
        shown = by if opts.get("no-trunc") or len(by) <= 45 else by[:44] + "…"
        rows.append({"ID": img.short if i == 0 else "<missing>", "CreatedSince": f"{ago(ctx.now - created)} ago",
                     "CreatedBy": shown, "Size": human_size(size) if size else "0B",
                     "Comment": "buildkit.dockerfile.v0" if by.startswith(("RUN", "COPY", "ADD", "WORKDIR", "ENV",
                                                                           "ARG", "CMD", "ENTRYPOINT", "USER",
                                                                           "EXPOSE", "LABEL", "HEALTHCHECK",
                                                                           "VOLUME")) and img.base is None else ""})
    fmt = opts.get("format")
    if opts.get("quiet"):
        ctx.w("".join(r["ID"] + "\n" for r in rows))
        return 0
    if fmt:
        ctx.w(gotmpl.table(fmt, rows, {"ID": "IMAGE", "CreatedSince": "CREATED", "CreatedBy": "CREATED BY",
                                       "Size": "SIZE", "Comment": "COMMENT"}) if fmt.startswith("table") else
              "".join(gotmpl.render(fmt.replace("\\t", "\t"), r) + "\n" for r in rows))
        return 0
    t = [["IMAGE", "CREATED", "CREATED BY", "SIZE", "COMMENT"]]
    for r in rows:
        t.append([r["ID"], r["CreatedSince"], r["CreatedBy"], r["Size"], r["Comment"]])
    ctx.w(gotmpl.tabulate(t))
    return 0
    yield


def image_json(d, img):
    cfg = dict(img.config)
    return {"Id": img.id, "RepoTags": list(img.repo_tags), "RepoDigests": list(img.repo_digests),
            "Parent": "", "Comment": "buildkit.dockerfile.v0" if img.base is None else "",
            "Created": _iso(img.created), "DockerVersion": "", "Author": "",
            "Config": {"Hostname": "", "User": cfg.get("User", ""), "ExposedPorts": cfg.get("ExposedPorts") or None,
                       "Env": cfg.get("Env"), "Cmd": cfg.get("Cmd"), "Healthcheck": cfg.get("Healthcheck"),
                       "Image": "", "Volumes": cfg.get("Volumes") or None, "WorkingDir": cfg.get("WorkingDir", ""),
                       "Entrypoint": cfg.get("Entrypoint"), "OnBuild": cfg.get("OnBuild"),
                       "Labels": cfg.get("Labels") or None,
                       **({"StopSignal": cfg["StopSignal"]} if cfg.get("StopSignal") else {}),
                       **({"Shell": cfg["Shell"]} if cfg.get("Shell") else {})},
            "Architecture": img.arch, "Os": "linux", "Size": img.size,
            "RootFS": {"Type": "layers", "Layers": ["sha256:" + hashlib.sha256(f"{img.id}:{i}".encode()).hexdigest()
                                                    for i, h in enumerate(img.history) if not h[2]]},
            "Metadata": {"LastTagTime": _iso(img.created)}}


def _iso(ns):
    from .cli import _iso as iso
    return iso(ns)


@cmd("image-inspect")
def docker_image_inspect(ctx, o, args):
    from .cli import docker_inspect
    r = docker_inspect(ctx, o, ["--type", "image"] + list(args))
    if hasattr(r, "send"):
        r = yield from r
    return r


# ---------------------------------------------------------------------------- save / load / commit

@cmd("save")
def docker_save(ctx, o, args):
    opts, rest = flags(args, {"-o|--output": "v", "--platform": "v"}, stop_at_positional=False)
    d = daemon_of(ctx.world, ctx.host)
    if not rest:
        raise DockerError("\"docker save\" requires at least 1 argument.", 1)
    imgs = []
    for ref in rest:
        img = d.find_image(ref)
        if img is None:
            raise DockerError(f"Error response from daemon: reference does not exist", 1)
        imgs.append((normalize(ref), img))
    if not opts.get("output") and ctx.stdin_is_tty() and ctx.proc.fds.get(1) and ctx.proc.fds[1].kind == "tty":
        raise DockerError("cowardly refusing to save to a terminal. Use the -o flag or redirect", 1)
    store = ctx.world.__dict__.setdefault("docker_tars", {})
    key = hashlib.sha256(("|".join(n + i.id for n, i in imgs)).encode()).hexdigest()
    store[key] = [(n, i) for n, i in imgs]
    size = sum(i.size for _, i in imgs)
    head = b"manifest.json\0" + b" " * 100 + f"bashlite-docker-image:{key}\n".encode()
    data = LazyData(size=size, seed=f"save:{key}", head=head)
    if opts.get("output"):
        path = host_path(ctx, opts["output"])
        try:
            node = ctx.fs.create(path, "/", 0o644, ctx.uid, ctx.proc.gid, ctx.proc.gids)
        except FSError as e:
            raise DockerError(f"failed to save image: open {opts['output']}: {e.msg.lower()}", 1)
        node.data = data
    else:
        from ..fileio import write
        yield from write(ctx.k, ctx.proc.fds.get(1), bytes(data[:4096]))
    yield ("sleep", ctx.now + int(max(0.2, size / 200e6) * NS))
    return 0


def tar_images(world, data):
    head = bytes(data[:200])
    m = re.search(rb"bashlite-docker-image:([0-9a-f]{64})", head)
    if not m:
        return None
    return world.__dict__.get("docker_tars", {}).get(m.group(1).decode())


@cmd("load")
def docker_load(ctx, o, args):
    opts, rest = flags(args, {"-i|--input": "v", "-q|--quiet": None}, stop_at_positional=False)
    d = daemon_of(ctx.world, ctx.host)
    if opts.get("input"):
        try:
            data = ctx.fs.lookup(host_path(ctx, opts["input"])).data
        except FSError:
            raise DockerError(f"open {opts['input']}: no such file or directory", 1)
    else:
        data = yield from ctx.read_fd(0)
    imgs = tar_images(ctx.world, data)
    if imgs is None:
        raise DockerError("Error response from daemon: unexpected EOF" if not len(data) else
                          "Error response from daemon: open /var/lib/docker/tmp/docker-import-1234/manifest.json: no "
                          "such file or directory", 1)
    for name, img in imgs:
        copy = img.copy_for(d)
        d.add_image(copy, [name])
        if not opts.get("quiet"):
            for i, h in enumerate([h for h in img.history if not h[2]][-3:]):
                ctx.w(f"{hashlib.sha256(f'{img.id}:{i}'.encode()).hexdigest()[:12]}: Loading layer  "
                      f"{human_size(h[1])}/{human_size(h[1])}\n")
        ctx.w(f"Loaded image: {name}\n")
    return 0


@cmd("commit")
def docker_commit(ctx, o, args):
    opts, rest = flags(args, {"-m|--message": "v", "-a|--author": "v", "-c|--change": "m", "-p|--pause": None},
                       stop_at_positional=False)
    d = daemon_of(ctx.world, ctx.host)
    if not rest:
        raise DockerError("\"docker commit\" requires at least 1 and at most 2 arguments.", 1)
    c = d.get_container(rest[0])
    from .fsutil import diff_bytes
    added = diff_bytes(c.image.fs.root, c.host.fs.root)
    fs = clone_fs(c.host.fs, lambda: ctx.world.kernel.now)
    cfg = json.loads(json.dumps(c.image.config))
    hist = list(c.image.history) + [(short_cmd_for(c), added, False, ctx.now)]
    iid = "sha256:" + hashlib.sha256(f"commit:{c.id}:{ctx.now}".encode()).hexdigest()
    img = Image(iid, cfg, fs, c.image.size + added, hist, ctx.now, c.image.arch, c.image.base)
    img.cuda = c.image.cuda
    d.add_image(img, [normalize(rest[1])] if len(rest) > 1 else [])
    ctx.w(img.id + "\n")
    return 0
    yield


def short_cmd_for(c):
    from .cli import short_cmd
    return short_cmd(c, False).strip('"')


__all__ = ["pull_image", "normalize", "image_json", "Out", "push_one", "registries"]
