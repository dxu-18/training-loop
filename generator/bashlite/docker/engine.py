"""docker-lite's daemon: images, containers, volumes, networks and registries, for one simulated machine.

A container is a small simulated host of its own (`Host`): its filesystem is a copy of the image's (contents
shared copy-on-write), its processes run in the world's kernel with PID 1 the image's entrypoint + command, its
stdout/stderr go to the container's log (and to the terminal when attached), and it gets an IP on each network it
joins. Published ports are services on the machine that forward to the container's own listening service;
containers on a user-defined network find each other by name (the default bridge has no such DNS). Bind mounts
and volumes graft a directory of the machine's filesystem into the container's (fsutil.graft), so files written
there are the machine's files.
"""

from __future__ import annotations

import hashlib
import json
import posixpath
import re

from ..kernel import NS
from ..vfs import FS, FSError
from .fsutil import clone_fs, clone_node, graft, tree_bytes
from .images import IMAGES, build_rootfs, canonical, digest_of, layer_ids

ENGINE = "27.3.1"
API = "1.47"


class DockerError(Exception):
    def __init__(self, msg, status=125):
        super().__init__(msg)
        self.msg, self.status = msg, status


class Image:
    def __init__(self, iid, config, fs, size, history, created, arch="amd64", base=None):
        self.id = iid                      # sha256:...
        self.config = config               # Cmd Entrypoint Env WorkingDir User ExposedPorts Labels Volumes ...
        self.fs = fs                       # frozen filesystem (containers start from a copy)
        self.size = size
        self.history = history             # [(created_by, size, empty_layer, created ns)]
        self.created = created
        self.arch = arch
        self.base = base                   # catalog name it starts from
        self.repo_tags: list[str] = []
        self.repo_digests: list[str] = []
        self.cuda = None
        self.platforms = [f"linux/{arch}"]

    @property
    def short(self) -> str:
        return self.id.split(":", 1)[1][:12]


class Container:
    def __init__(self, cid, name, image, cfg):
        self.id, self.name, self.image, self.cfg = cid, name, image, cfg
        self.state = "created"
        self.exit_code = 0
        self.oom = False
        self.error = ""
        self.created = 0
        self.started = 0
        self.finished = 0
        self.host = None
        self.proc = None
        self.logs: list[tuple[int, str, bytes]] = []
        self.ports: list[tuple[str, int, int, str]] = []        # (host ip, host port, container port, proto)
        self.mounts: list[dict] = []
        self.networks: dict[str, str] = {}                       # network name -> ip
        self.restart_count = 0
        self.health = None                                       # starting | healthy | unhealthy
        self.health_log: list = []
        self.stopping = False
        self.labels = dict(cfg.get("labels", {}))
        self.log_waiters: list = []

    @property
    def short(self) -> str:
        return self.id[:12]

    @property
    def running(self) -> bool:
        return self.state in ("running", "paused", "restarting")

    def status_text(self, now) -> str:
        if self.state == "running":
            h = f" ({self.health})" if self.health else ""
            return f"Up {ago(now - self.started)}{h}"
        if self.state == "paused":
            return f"Up {ago(now - self.started)} (Paused)"
        if self.state == "restarting":
            return f"Restarting ({self.exit_code}) {ago(now - self.finished)} ago"
        if self.state == "exited":
            return f"Exited ({self.exit_code}) {ago(now - self.finished)} ago"
        return "Created"


class Volume:
    def __init__(self, name, created, labels=None, anonymous=False):
        self.name, self.created, self.labels, self.anonymous = name, created, labels or {}, anonymous

    @property
    def mountpoint(self):
        return f"/var/lib/docker/volumes/{self.name}/_data"


class Network:
    def __init__(self, name, nid, driver, subnet, internal=False, created=0, labels=None):
        self.name, self.id, self.driver, self.subnet = name, nid, driver, subnet
        self.internal, self.created, self.labels = internal, created, labels or {}
        self.next_ip = 2

    @property
    def gateway(self):
        return self.subnet.rsplit(".", 2)[0] + ".0.1"

    def alloc(self) -> str:
        ip = self.subnet.rsplit(".", 2)[0] + f".0.{self.next_ip}"
        self.next_ip += 1
        return ip


def ago(ns: int) -> str:
    """Docker's units.HumanDuration."""
    s = max(0, ns // NS)
    if s < 1:
        return "Less than a second"
    if s == 1:
        return "1 second"
    if s < 60:
        return f"{s} seconds"
    m = s // 60
    if m == 1:
        return "About a minute"
    if m < 60:
        return f"{m} minutes"
    h = round(s / 3600)
    if h == 1:
        return "About an hour"
    if h < 48:
        return f"{h} hours"
    if h < 24 * 7 * 2:
        return f"{h // 24} days"
    if h < 24 * 30 * 2:
        return f"{h // 24 // 7} weeks"
    if h < 24 * 365 * 2:
        return f"{h // 24 // 30} months"
    return f"{h // 24 // 365} years"


def human_size(n: float) -> str:
    """Docker's units.HumanSize: decimal units, 3 significant digits."""
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if n < 1000 or unit == "TB":
            if unit == "B":
                return f"{int(n)}B"
            s = f"{n:.3g}"
            if "e" in s:
                s = f"{n:.0f}"
            return f"{s}{unit}"
        n /= 1000
    return f"{n:.3g}TB"


def parse_size(s: str) -> int:
    m = re.match(r"^(\d+(?:\.\d+)?)([bkmgt]?)b?$", str(s).strip().lower())
    if not m:
        raise DockerError(f'invalid size: \'{s}\'')
    return int(float(m.group(1)) * {"": 1, "b": 1, "k": 1024, "m": 1024 ** 2, "g": 1024 ** 3,
                                     "t": 1024 ** 4}[m.group(2)])


NAMES_L = ["admiring", "brave", "clever", "dazzling", "eager", "festive", "gallant", "happy", "jolly", "keen",
           "loving", "modest", "nifty", "optimistic", "practical", "quirky", "relaxed", "serene", "tender", "upbeat",
           "vibrant", "wizardly", "youthful", "zealous"]
NAMES_R = ["albattani", "babbage", "curie", "darwin", "einstein", "fermi", "galileo", "hopper", "johnson", "kepler",
           "lovelace", "meitner", "noether", "pasteur", "ritchie", "shannon", "turing", "wozniak", "yalow", "zhukovsky"]


class Daemon:
    def __init__(self, world, host):
        self.world, self.host = world, host
        self.images: dict[str, Image] = {}
        self.tags: dict[str, str] = {}            # "repo:tag" -> image id
        self.containers: dict[str, Container] = {}
        self.volumes: dict[str, Volume] = {}
        self.networks: dict[str, Network] = {}
        self.build_cache: dict[str, tuple] = {}   # cache key -> (fs, size, history entry)
        self.cache_mounts: dict[str, object] = {}
        self.auth: dict[str, str] = {}            # registry -> user
        self.running = True
        self.installed = True
        self.gpu_toolkit = True
        self.events: list[tuple[int, str]] = []
        self.seq = 0
        self.base_fs: dict[str, FS] = {}
        self.dangling_seq = 0
        for name, driver, subnet in (("bridge", "bridge", "172.17.0.0/16"), ("host", "host", ""),
                                     ("none", "null", "")):
            self.networks[name] = Network(name, self._id("net:" + name), driver, subnet, created=world.kernel.now)
        self._setup_host()

    # -------------------------------------------------- the machine side
    def _setup_host(self):
        fs = self.host.fs
        for p, mode in (("/var/lib/docker", 0o710), ("/var/lib/docker/volumes", 0o701),
                        ("/var/lib/docker/containers", 0o710), ("/etc/docker", 0o755)):
            n = fs.mkdir(p, "/", mode, 0, 0, parents=True)
            n.mode = mode
        sock = fs.write_file("/run/docker.sock", b"", "/", 0o660, 0, 999)
        sock.type = "s"
        sock.mode, sock.gid = 0o660, 999
        if not fs.exists("/var/run"):
            fs.symlink("/run", "/var/run")

    @property
    def now(self):
        return self.world.kernel.now

    def _id(self, seed: str) -> str:
        return hashlib.sha256(f"{self.world.seed}:{seed}".encode()).hexdigest()

    def new_id(self, kind="c") -> str:
        self.seq += 1
        return self._id(f"{kind}:{self.seq}")

    def event(self, text):
        self.events.append((self.now, text))

    # -------------------------------------------------- images
    def find_image(self, ref: str):
        """An image by reference (name[:tag], digest or id prefix) or None."""
        if not ref:
            return None
        from .cmd_images import normalize
        r = normalize(ref)
        iid = self.tags.get(r)
        if iid:
            return self.images.get(iid)
        if "@sha256:" in ref:
            for img in self.images.values():
                if any(d.endswith(ref.split("@", 1)[1]) for d in img.repo_digests):
                    return img
        x = ref[7:] if ref.startswith("sha256:") else ref
        if re.fullmatch(r"[0-9a-f]{4,64}", x):
            hits = [i for i in self.images.values() if i.id[7:].startswith(x)]
            if len(hits) == 1:
                return hits[0]
        return None

    def catalog_image(self, name: str) -> Image:
        """The local copy of a catalog image (made when pulled)."""
        spec = IMAGES[name]
        fs = self.base_fs.get(name)
        if fs is None:
            fs = build_rootfs(self.world, name)
            self.base_fs[name] = fs
        cfg = {"Cmd": spec.get("cmd"), "Entrypoint": spec.get("entrypoint"),
               "Env": list(spec.get("env") or ["PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"]),
               "WorkingDir": spec.get("workdir", ""), "User": spec.get("user", ""), "Labels": {},
               "ExposedPorts": {p: {} for p in spec.get("expose", [])},
               "Volumes": {v: {} for v in spec.get("volumes", [])}, "StopSignal": spec.get("stopsignal")}
        n = spec.get("layers", 1)
        hist = [(f"/bin/sh -c #(nop) ADD file:{layer_ids(name, n)[0][:16]} in / ", spec["size"] // max(1, n), False,
                 self.now - 30 * 86400 * NS)]
        for i in range(1, n):
            hist.append((f"RUN /bin/sh -c set -eux; step {i} # buildkit", spec["size"] // max(1, n), False,
                         self.now - 30 * 86400 * NS))
        hist.append((f"CMD {json.dumps(spec.get('cmd'))}", 0, True, self.now - 30 * 86400 * NS))
        img = Image("sha256:" + hashlib.sha256(f"image:{name}".encode()).hexdigest(), cfg, fs, spec["size"], hist,
                    self.now - 30 * 86400 * NS, base=name)
        img.cuda = spec.get("cuda")
        img.repo_digests = [f"{name.rsplit(':', 1)[0]}@{digest_of(name)}"]
        return img

    def add_image(self, img: Image, tags=()):
        self.images[img.id] = img
        for t in tags:
            self.tag(img, t)
        return img

    def tag(self, img: Image, ref: str):
        from .cmd_images import normalize
        r = normalize(ref)
        old = self.tags.get(r)
        if old and old != img.id and old in self.images:
            o = self.images[old]
            if r in o.repo_tags:
                o.repo_tags.remove(r)
        self.tags[r] = img.id
        if r not in img.repo_tags:
            img.repo_tags.append(r)

    def untag(self, ref: str):
        from .cmd_images import normalize
        r = normalize(ref)
        iid = self.tags.pop(r, None)
        if iid and iid in self.images and r in self.images[iid].repo_tags:
            self.images[iid].repo_tags.remove(r)
        return iid

    def image_in_use(self, img: Image):
        return [c for c in self.containers.values() if c.image is img]

    # -------------------------------------------------- volumes and networks
    def volume(self, name=None, create=True, labels=None):
        anonymous = name is None
        if name is None:
            name = self.new_id("v")
        v = self.volumes.get(name)
        if v is None and create:
            v = Volume(name, self.now, labels, anonymous)
            self.volumes[name] = v
            d = self.host.fs.mkdir(v.mountpoint, "/", 0o755, 0, 0, parents=True)
            d.mode = 0o755
            vol_dir = self.host.fs.lookup(f"/var/lib/docker/volumes/{name}")
            vol_dir.mode = 0o701
            self.event(f"volume create {name}")
        return v

    def volume_users(self, name):
        return [c for c in self.containers.values() if any(m.get("name") == name for m in c.mounts)]

    def network(self, name):
        n = self.networks.get(name)
        if n is None:
            for x in self.networks.values():
                if x.id.startswith(name):
                    return x
        return n

    def create_network(self, name, driver="bridge", internal=False, labels=None):
        if name in self.networks:
            raise DockerError(f"Error response from daemon: network with name {name} already exists", 1)
        k = len([n for n in self.networks.values() if n.driver == "bridge"])
        n = Network(name, self._id("net:" + name), driver, f"172.{17 + k}.0.0/16", internal, self.now, labels)
        self.networks[name] = n
        self.event(f"network create {name}")
        return n

    # -------------------------------------------------- containers
    def find_container(self, ref: str):
        if not ref:
            return None
        ref = ref.lstrip("/")
        for c in self.containers.values():
            if c.name == ref:
                return c
        hits = [c for c in self.containers.values() if c.id.startswith(ref)]
        return hits[0] if len(hits) == 1 else None

    def get_container(self, ref: str) -> Container:
        c = self.find_container(ref)
        if c is None:
            raise DockerError(f"Error response from daemon: No such container: {ref}", 1)
        return c

    def random_name(self):
        rng = self.world.rng
        for _ in range(100):
            n = f"{rng.choice(NAMES_L)}_{rng.choice(NAMES_R)}"
            if not self.find_container(n):
                return n
        return f"container_{self.seq}"

    def create(self, img: Image, cfg: dict) -> Container:
        """docker create: the container's filesystem, mounts and network addresses (not started)."""
        name = cfg.get("name") or self.random_name()
        if cfg.get("name"):
            if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", name):
                raise DockerError(f'Error response from daemon: Invalid container name ({name}), only '
                                  f'[a-zA-Z0-9][a-zA-Z0-9_.-] are allowed', 125)
            other = self.find_container(name)
            if other is not None and other.name == name:
                raise DockerError(f'Error response from daemon: Conflict. The container name "/{name}" is already in '
                                  f'use by container "{other.id}". You have to remove (or rename) that container to '
                                  f'be able to reuse that name.', 125)
        net = cfg.get("network") or "bridge"
        if net not in ("host", "none") and self.network(net) is None:
            raise DockerError(f"Error response from daemon: network {net} not found", 125)
        c = Container(self.new_id("c"), name, img, cfg)
        c.created = self.now
        self.containers[c.id] = c
        self._make_host(c)
        for spec in cfg.get("ports", []):
            c.ports.append(spec)
        self.event(f"container create {c.short} (image={cfg.get('image_ref')}, name={name})")
        return c

    def _make_host(self, c: Container):
        from ..world import Host
        img = c.image
        h = Host(self.world, c.short if not c.cfg.get("hostname") else c.cfg["hostname"], "0.0.0.0", pid_base=0)
        h.fs = clone_fs(img.fs, lambda: self.world.kernel.now)
        h.extra["container"] = c
        h.extra["daemon"] = self
        h.extra["resolver"] = lambda name, c=c: self.resolve_for(c, name)
        h.cpus = int(c.cfg.get("cpus") or self.host.cpus) or 1
        mem = c.cfg.get("memory")
        h.mem_kib = (mem // 1024) if mem else self.host.mem_kib
        h.kernel_release = self.host.kernel_release
        h.cpu_model = self.host.cpu_model
        h.boot_ns = self.world.kernel.now
        h.authorized_keys_required = False
        try:
            text = bytes(h.fs.lookup("/etc/passwd").data).decode()
        except FSError:
            text = "root:x:0:0:root:/root:/bin/sh\n"
        from ..world import User
        for line in text.split("\n"):
            parts = line.split(":")
            if len(parts) >= 7 and parts[2].isdigit():
                h.users[int(parts[2])] = User(parts[0], int(parts[2]), int(parts[3]), parts[4], parts[5], parts[6])
                h.groups.setdefault(int(parts[3]), (parts[0], set()))
        if 0 not in h.users:
            h.users[0] = User("root", 0, 0, "root", "/root", "/bin/sh")
        fs = h.fs
        hostname = c.cfg.get("hostname") or c.short
        fs.write_file("/etc/hostname", hostname + "\n", "/", 0o644, 0, 0)
        # networks
        net = c.cfg.get("network") or "bridge"
        if net == "host":
            h.ip = self.host.ip
        elif net != "none":
            n = self.network(net)
            ip = n.alloc()
            c.networks[n.name] = ip
            h.ip = ip
            aliases = c.cfg.get("aliases", [])
            for extra in c.cfg.get("networks_extra", []):
                n2 = self.network(extra)
                if n2 is not None:
                    c.networks[n2.name] = n2.alloc()
            del aliases
        hosts = f"127.0.0.1\tlocalhost\n::1\tlocalhost ip6-localhost ip6-loopback\n{h.ip}\t{hostname}\n"
        for name, ip in c.cfg.get("add_hosts", []):
            hosts += f"{ip}\t{name}\n"
        fs.write_file("/etc/hosts", hosts, "/", 0o644, 0, 0)
        key = f"docker:{c.id}"
        self.world.hosts[key] = h
        if h.ip not in ("0.0.0.0", self.host.ip):
            self.world.dns[h.ip] = key
            for ip in c.networks.values():
                self.world.dns[ip] = key
        # GPUs: the NVIDIA runtime gives the container the devices (and nvidia-smi)
        gpus = c.cfg.get("gpus")
        if gpus is not None:
            h.gpus = [g for g in self.host.gpus if gpus == "all" or g.index in gpus]
            from ..exec import stub_bytes
            n_ = fs.write_file("/usr/bin/nvidia-smi", stub_bytes("nvidia-smi"), "/", 0o755, 0, 0)
            n_.mode = 0o755
        # mounts
        for m in c.cfg.get("mounts", []):
            self._mount(c, h, m)
        # image volumes become anonymous volumes
        for vpath in (c.image.config.get("Volumes") or {}):
            if not any(m["dest"] == vpath for m in c.mounts):
                self._mount(c, h, {"type": "volume", "source": None, "dest": vpath, "ro": False})
        shm = c.cfg.get("shm_mb", 64)
        h.extra["shm_mb"] = shm
        c.host = h

    def _mount(self, c, h, m):
        fs = h.fs
        dest = m["dest"]
        if m["type"] == "bind":
            src = m["source"]
            try:
                node = self.host.fs.lookup(src)
            except FSError:
                if m.get("long"):
                    raise DockerError(f"docker: Error response from daemon: invalid mount config for type \"bind\": "
                                      f"bind source path does not exist: {src}", 125)
                node = self.host.fs.mkdir(src, "/", 0o755, 0, 0, parents=True)
            graft(fs, dest, node, m.get("ro", False))
            c.mounts.append({"type": "bind", "source": src, "dest": dest, "ro": m.get("ro", False)})
        elif m["type"] == "volume":
            v = self.volume(m.get("source"))
            data = self.host.fs.lookup(v.mountpoint)
            # an empty volume starts with what the image has at that path
            if not data.entries:
                try:
                    old = fs.lookup(dest)
                    if old.type == "d":
                        for k, child in old.entries.items():
                            data.entries[k] = clone_node(child, self.host.fs, data)
                        data.uid, data.gid, data.mode = old.uid, old.gid, old.mode
                except FSError:
                    pass
            graft(fs, dest, data, m.get("ro", False))
            c.mounts.append({"type": "volume", "name": v.name, "source": v.mountpoint, "dest": dest,
                             "ro": m.get("ro", False)})
        elif m["type"] == "tmpfs":
            n = fs.mkdir(dest, "/", 0o1777, 0, 0, parents=True)
            n.mode = 0o1777
            n.entries.clear()
            c.mounts.append({"type": "tmpfs", "source": "", "dest": dest, "ro": False})

    def resolve_for(self, c: Container, name: str):
        """Name resolution inside container c: localhost, its own name, peers on its user-defined networks (by
        container name, service name or alias), extra /etc/hosts entries, then the outside world."""
        if name in ("localhost", "127.0.0.1", c.host.name, c.short):
            return c.host
        for hname, ip in c.cfg.get("add_hosts", []):
            if hname == name:
                if ip in ("host-gateway", self.networks["bridge"].gateway, self.host.ip):
                    return self.host
                return self.world.resolve(ip)
        for net in c.networks:
            n = self.networks.get(net)
            if n is None or net == "bridge":
                continue
            for o in self.containers.values():
                if o is c or not o.running or net not in o.networks:
                    continue
                names = {o.name, o.short} | set(o.cfg.get("aliases", [])) | (
                    {o.labels.get("com.docker.compose.service")} if o.labels.get("com.docker.compose.service") else set())
                if name in names:
                    return o.host
        if c.cfg.get("network") == "none":
            return None
        if c.cfg.get("network") == "host":
            if name in ("localhost", "127.0.0.1"):
                return self.host
        hit = self.world.dns.get(name)
        if hit and hit.startswith("docker:"):
            return self.world.hosts.get(hit)
        if name == self.host.name:
            return None
        return self.world.resolve(name) if hit else None

    # -------------------------------------------------- running
    def start(self, c: Container, attach=None):
        """Start PID 1 (entrypoint + command). attach: {0: stdin, 1: out, 2: err} file objects to also write to."""
        from ..exec import exec_program, find_in_path
        if c.running:
            return
        if c.state == "exited" or c.state == "created" and c.proc is not None:
            pass
        img = c.image
        ep = c.cfg.get("entrypoint")
        if ep is None:
            ep = img.config.get("Entrypoint") or []
        cmd = c.cfg.get("cmd")
        if cmd is None:
            cmd = img.config.get("Cmd") or [] if not c.cfg.get("entrypoint_set") else []
        argv = list(ep) + list(cmd)
        if not argv:
            raise DockerError("docker: Error response from daemon: no command specified", 125)
        env = dict(e.split("=", 1) for e in img.config.get("Env") or [] if "=" in e)
        for e in c.cfg.get("env", []):
            if "=" in e:
                k, v = e.split("=", 1)
                env[k] = v
        env.setdefault("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")
        env["HOSTNAME"] = c.host.name
        h = c.host
        user = c.cfg.get("user") or img.config.get("User") or "root"
        uid, gid = self.user_ids(h, user)
        env.setdefault("HOME", h.users[uid].home if uid in h.users else "/")
        if c.cfg.get("tty"):
            env.setdefault("TERM", "xterm")
        if h.gpus:
            env["_GPUS"] = ",".join(str(g.index) for g in h.gpus)
        env["_SHM_MB"] = str(h.extra.get("shm_mb", 64))
        workdir = c.cfg.get("workdir") or img.config.get("WorkingDir") or "/"
        if not h.fs.exists(workdir):
            h.fs.mkdir(workdir, "/", 0o755, uid, gid, parents=True)
        # the program must exist in the image
        prog = argv[0]
        path = find_in_path(h.fs, prog, env["PATH"], workdir)
        if path is not None and not path.startswith("/"):       # ./start.sh is relative to the working directory
            path = posixpath.normpath(posixpath.join(workdir, path))
        if path is None or not h.fs.exists(path):
            c.state = "created" if c.state == "created" else "exited"
            c.exit_code = 127
            raise DockerError(
                f"docker: Error response from daemon: failed to create task for container: failed to create shim "
                f"task: OCI runtime create failed: runc create failed: unable to start container process: error "
                f"during container init: exec: \"{prog}\": {'stat ' + prog + ': no such file or directory' if '/' in prog else 'executable file not found in $PATH'}: unknown.", 127)
        node = h.fs.lookup(path)
        if node.type == "d" or not node.mode & 0o111:
            c.exit_code = 126
            raise DockerError(
                f"docker: Error response from daemon: failed to create task for container: failed to create shim "
                f"task: OCI runtime create failed: runc create failed: unable to start container process: error "
                f"during container init: exec: \"{prog}\": permission denied: unknown.", 126)
        if self.needs_cuda(c):
            raise DockerError(self.needs_cuda(c), 125)
        for hip, hport, cport, proto in c.ports:
            self.publish(c, hip, hport, cport)
        sink = LogSink(c, attach)
        from ..kernel import OpenFile
        out = OpenFile("log", wr=True, stream="out")
        out.tty = sink
        err = OpenFile("log", wr=True, stream="err")
        err.tty = sink
        fds = {1: out, 2: err}
        if attach and 0 in attach and attach[0] is not None:
            fds[0] = attach[0]
        else:
            fds[0] = OpenFile("null", rd=True)
        if attach and c.cfg.get("tty"):
            for n in (1, 2):
                if attach.get(n) is not None:
                    fds[n] = attach[n]
        k = self.world.kernel
        p = k.spawn(h, argv, None, None, fds=fds, cwd=workdir, env=env, uid=uid, gid=gid)
        p.daemon = True
        p.pgid = attach.get("pgid", p.pid) if attach else p.pid    # attached: the terminal's foreground job
        c.proc = p
        c.state = "running"
        c.started = self.now
        c.oom = False
        c.stopping = False
        c.error = ""
        if img.config.get("Healthcheck") or c.cfg.get("healthcheck"):
            c.health = "starting"
        p.on_exit.append(lambda proc, c=c: self._exited(c, proc))
        k.start(p, exec_program(p, path, argv, env, None))
        self.event(f"container start {c.short} (name={c.name})")
        if c.health is not None:
            self._schedule_health(c)
        if self.check_oom(c):
            pass
        return p

    def needs_cuda(self, c):
        img = c.image
        need = img.cuda
        if c.cfg.get("gpus") is not None:
            if not self.gpu_toolkit or not self.host.gpus:
                return ('docker: Error response from daemon: could not select device driver "" with capabilities: '
                        '[[gpu]].')
            if need and float(need) > float(self.host.extra.get("cuda_driver", "12.4")):
                return ("docker: Error response from daemon: failed to create task for container: failed to create "
                        "shim task: OCI runtime create failed: runc create failed: unable to start container process: "
                        "error during container init: error running prestart hook #0: exit status 1, stdout: , stderr: "
                        f"Auto-detected mode as 'legacy'\nnvidia-container-cli: requirement error: unsatisfied "
                        f"condition: cuda>={need}, please update your driver to a newer version, or use an earlier "
                        f"cuda container: unknown.")
        return None

    def check_oom(self, c):
        return False

    def user_ids(self, h, user):
        name, _, group = str(user).partition(":")
        if name.isdigit():
            uid = int(name)
        else:
            uid = next((u.uid for u in h.users.values() if u.name == name), None)
            if uid is None:
                raise DockerError(f"docker: Error response from daemon: unable to find user {name}: no matching "
                                  f"entries in passwd file.", 126)
        if group.isdigit():
            gid = int(group)
        elif group:
            gid = next((g for g, (n, _) in h.groups.items() if n == group), 0)
        else:
            gid = h.users[uid].gid if uid in h.users else 0
        if uid not in h.users:
            from ..world import User
            h.users[uid] = User("", uid, gid, "", "/", "/bin/sh")
        return uid, gid

    def _exited(self, c: Container, proc):
        c.exit_code = proc.status
        c.finished = self.now
        c.proc = None
        for hip, hport, cport, proto in c.ports:
            svc = self.host.services.get(hport)
            if svc is not None and getattr(svc, "container", None) is c:
                del self.host.services[hport]
        # other processes in the container go with PID 1
        for p in list(c.host.procs.values()):
            if p.state != "Z":
                self.world.kernel.kill_now(p, 9)
        c.host.services.clear()
        self.event(f"container die {c.short} (exitCode={c.exit_code}, name={c.name})")
        policy = c.cfg.get("restart", "no")
        restart = not c.stopping and (policy in ("always", "unless-stopped") or
                                      policy.startswith("on-failure") and c.exit_code != 0)
        if restart and policy.startswith("on-failure:"):
            restart = c.restart_count < int(policy.split(":")[1])
        if restart:
            c.state = "restarting"
            c.restart_count += 1
            delay = min(60, 0.1 * 2 ** min(c.restart_count, 9))

            def again(c=c):
                if c.state == "restarting" and c.id in self.containers:
                    c.state = "exited"
                    try:
                        self.start(c)
                    except DockerError:
                        pass
            self.world.kernel.add_timer(self.now + int(delay * NS), again)
            return
        c.state = "exited"
        c.health = None if c.health is None else c.health
        if c.cfg.get("rm"):
            self.remove(c, force=True)

    def publish(self, c, hip, hport, cport):
        from ..world import Service
        cur = self.host.services.get(hport)
        if cur is not None and (getattr(cur, "container", None) is None or cur.container is not c):
            other = getattr(cur, "container", None)
            if other is None or other.running:
                raise DockerError(f"docker: Error response from daemon: driver failed programming external "
                                  f"connectivity on endpoint {c.name} ({c.id[:64]}): Bind for {hip or '0.0.0.0'}:"
                                  f"{hport} failed: port is already allocated.", 125)

        def handler(method, path, headers, body, c=c, cport=cport):
            from ..utils.net import HttpError
            if not c.running or c.state == "paused":
                raise HttpError(56, "Recv failure: Connection reset by peer")
            svc = c.host.services.get(cport)
            if svc is None or svc.handler is None:
                raise HttpError(52, "Empty reply from server")
            if getattr(svc, "loopback", False):
                raise HttpError(56, "Recv failure: Connection reset by peer")
            return svc.handler(method, path, headers, body)
        s = Service(hport, None, handler, f"docker-proxy:{c.short}:{cport}")
        s.container = c
        s.bind = hip or "0.0.0.0"
        self.host.services[hport] = s

    def stop(self, c: Container, timeout=None, sig=None):
        """docker stop: the stop signal, then SIGKILL after the timeout (generator: waits)."""
        if not c.running or c.proc is None:
            return
        c.stopping = True
        p = c.proc
        signame = sig or c.cfg.get("stop_signal") or c.image.config.get("StopSignal") or "SIGTERM"
        graceful = getattr(p, "graceful", None) or graceful_program(c)
        k = self.world.kernel
        if graceful:
            yield ("sleep", self.now + int(0.3 * NS))
            if p.state != "Z":
                k.kill_now(p, 15)
                c.exit_code = 143 if graceful == "143" else 0
            return
        t = c.cfg.get("stop_timeout", 10) if timeout is None else timeout
        yield ("sleep", self.now + int(t * NS))
        if p.state != "Z":
            k.kill_now(p, 9)

    def kill(self, c: Container, sig=9):
        if c.proc is not None and c.proc.state != "Z":
            c.stopping = sig in (9, 15)
            self.world.kernel.kill_now(c.proc, sig)

    def remove(self, c: Container, force=False, volumes=False):
        if c.running and not force:
            raise DockerError(f"Error response from daemon: cannot remove container \"/{c.name}\": container is "
                              f"running: stop the container before removing or force remove", 1)
        if c.running:
            c.stopping = True
            self.kill(c)
        self.containers.pop(c.id, None)
        self.world.hosts.pop(f"docker:{c.id}", None)
        for ip in list(c.networks.values()):
            if self.world.dns.get(ip) == f"docker:{c.id}":
                del self.world.dns[ip]
        for m in c.mounts:
            if m["type"] == "volume" and (volumes or c.cfg.get("rm")):
                v = self.volumes.get(m.get("name"))
                if v is not None and v.anonymous and not self.volume_users(v.name):
                    self.remove_volume(v.name)
        self.event(f"container destroy {c.short} (name={c.name})")

    def remove_volume(self, name):
        from ..git.repo import rmtree
        self.volumes.pop(name, None)
        rmtree(self.host.fs, f"/var/lib/docker/volumes/{name}")

    def _schedule_health(self, c):
        hc = c.cfg.get("healthcheck") or c.image.config.get("Healthcheck") or {}
        interval = hc.get("Interval", 30 * NS)
        retries = hc.get("Retries", 3)
        test = hc.get("Test") or []
        k = self.world.kernel
        if not test or test[0] == "NONE":
            c.health = None
            return

        def probe():
            if c.state != "running":
                return
            ok = health_probe(self, c, test)
            c.health_log.append((self.now, ok))
            fails = 0
            for _, r in reversed(c.health_log):
                if r:
                    break
                fails += 1
            if ok:
                c.health = "healthy"
            elif fails >= retries:
                c.health = "unhealthy"
            k.add_timer(self.now + interval, probe)
        k.add_timer(self.now + min(interval, hc.get("StartPeriod", 0) + interval), probe)


def graceful_program(c: Container):
    """Does PID 1 exit promptly on the stop signal? Programs that handle it (nginx, postgres, uvicorn, jupyter)
    do, when they are PID 1 (exec form); a shell running them (`sh -c`, shell-form CMD) doesn't pass it on."""
    if c.proc is None:
        return None
    img = c.image
    ep = c.cfg.get("entrypoint", img.config.get("Entrypoint")) or []
    cmd = c.cfg.get("cmd", img.config.get("Cmd") if not c.cfg.get("entrypoint_set") else []) or []
    argv = list(ep) + list(cmd)
    if c.cfg.get("init"):
        return "143"
    if len(argv) >= 3 and posixpath.basename(argv[0]) in ("sh", "bash", "dash") and argv[1] == "-c":
        inner = argv[2].strip()
        if not inner.startswith("exec "):
            return None                      # a shell is PID 1 and doesn't pass the signal on
        argv = inner.split()[1:]
    prog = posixpath.basename(argv[0]) if argv else ""
    if prog in ("nginx", "postgres", "uvicorn", "start-notebook.py", "tini", "docker-entrypoint.sh",
                "/docker-entrypoint.sh", "redis-server", "hello", "gunicorn"):
        return "0"
    return None


def health_probe(daemon, c, test):
    """Run a health check: CMD probes are judged from what they do (an HTTP request to a listening port, a
    pg_isready against a ready server, a command that exists)."""
    cmd = " ".join(test[1:]) if test[0] in ("CMD", "CMD-SHELL") else " ".join(test)
    h = c.host
    if "pg_isready" in cmd:
        return 5432 in h.services
    m = re.search(r"localhost:(\d+)", cmd) or re.search(r"127\.0\.0\.1:(\d+)", cmd)
    if m:
        tool = cmd.split()[0]
        if tool in ("curl", "wget") and not any(h.fs.exists(f"{d}/{tool}") for d in ("/usr/bin", "/usr/local/bin",
                                                                                     "/bin")):
            return False
        return int(m.group(1)) in h.services
    return True


class LogSink:
    """Where a container's stdout and stderr go: its log, and the attached terminal or pipes when attached."""

    def __init__(self, c, attach):
        self.c, self.attach = c, attach

    def write(self, stream, data):
        c = self.c
        c.logs.append((c.host.world.kernel.now if c.host else 0, stream, bytes(data)))
        if len(c.logs) > 5000:
            del c.logs[:1000]
        if self.attach:
            of = self.attach.get(1 if stream == "out" else 2)
            if of is not None:
                from ..fileio import write_nb
                try:
                    write_nb(c.host.world.kernel, of, bytes(data))
                except Exception:  # noqa: BLE001
                    pass
        k = c.host.world.kernel if c.host else None
        ws, c.log_waiters = c.log_waiters, []
        for p in ws:
            if k is not None:
                k.poke(p)


def daemon_of(world, host=None):
    """The docker daemon of a host (the machine by default), created on first use."""
    host = host or world.local
    d = host.extra.get("dockerd")
    if d is None:
        d = Daemon(world, host)
        host.extra["dockerd"] = d
    return d


__all__ = ["Daemon", "DockerError", "Image", "Container", "daemon_of", "human_size", "ago", "parse_size",
           "tree_bytes", "json"]
