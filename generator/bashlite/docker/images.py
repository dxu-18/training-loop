"""The images registries serve before anyone builds anything: official bases (ubuntu, debian, alpine, busybox,
python, nginx, postgres, hello-world, curl), the NVIDIA CUDA and PyTorch images, Jupyter's scipy-notebook, and a
few teaching images under `mlgym/`. Each is a recipe: the distribution's programs (bash-lite utilities installed
as /usr/bin stubs), extra files, and the image config (Cmd, Entrypoint, Env, ExposedPorts, User, WorkingDir...).
Sizes are what `docker images` shows for the real images, roughly."""

from __future__ import annotations

import hashlib

COMMON = ["cat", "ls", "cp", "mv", "rm", "mkdir", "rmdir", "touch", "echo", "printf", "test", "[", "true", "false",
          "env", "printenv", "id", "whoami", "hostname", "uname", "date", "sleep", "head", "tail", "grep", "egrep",
          "fgrep", "sed", "cut", "sort", "uniq", "wc", "tr", "tee", "find", "xargs", "tar", "gzip", "gunzip", "zcat",
          "chmod", "chown", "chgrp", "ln", "readlink", "realpath", "basename", "dirname", "df", "du", "stat", "seq",
          "yes", "mktemp", "nproc", "kill", "timeout", "nice", "nohup", "groups", "tty", "split", "tac", "nl", "od",
          "base64", "md5sum", "sha256sum", "sha1sum", "cmp", "comm", "join", "paste", "expr", "fold", "shuf",
          "truncate", "dd", "more", "mawk", "awk", "logname", "pwd"]
DEBIAN = COMMON + ["bash", "sh", "dash", "apt", "apt-get", "which", "diff", "useradd", "groupadd", "usermod",
                   "userdel", "adduser", "su"]
ALPINE = COMMON + ["sh", "apk", "wget", "ps", "top", "vi", "which", "adduser", "addgroup", "su", "pkill", "pgrep",
                   "less"]
BUSYBOX = [p for p in ALPINE if p not in ("apk",)]

OS_RELEASE = {
    "debian": 'PRETTY_NAME="Debian GNU/Linux 12 (bookworm)"\nNAME="Debian GNU/Linux"\nVERSION_ID="12"\n'
              'VERSION_CODENAME=bookworm\nID=debian\n',
    "ubuntu24": 'PRETTY_NAME="Ubuntu 24.04.1 LTS"\nNAME="Ubuntu"\nVERSION_ID="24.04"\nVERSION_CODENAME=noble\n'
                'ID=ubuntu\nID_LIKE=debian\n',
    "ubuntu22": 'PRETTY_NAME="Ubuntu 22.04.5 LTS"\nNAME="Ubuntu"\nVERSION_ID="22.04"\nVERSION_CODENAME=jammy\n'
                'ID=ubuntu\nID_LIKE=debian\n',
    "alpine": 'NAME="Alpine Linux"\nID=alpine\nVERSION_ID=3.20.3\nPRETTY_NAME="Alpine Linux v3.20"\n',
    "busybox": "",
}
PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
PY_ENV = [f"PATH={PATH}", "LANG=C.UTF-8", "GPG_KEY=7169605F62C751356D054A26A821E680E5FA6305", "PYTHON_VERSION=3.12.7",
          "PYTHON_SHA256=24887b92e2afd4a2ac602419ad4b596372f67ac9b077190f459aba390faf5550"]

MB = 1000 * 1000

# name -> recipe. distro: program set and /etc/os-release; extra: more programs; python: where Python lives
IMAGES = {
    "hello-world:latest": dict(distro=None, size=13_256, cmd=["/hello"], layers=1, extra=["hello"], env=[f"PATH={PATH}"]),
    "busybox:latest": dict(distro="busybox", size=4_270_000, cmd=["sh"], layers=1),
    "alpine:3.20": dict(distro="alpine", size=7_800_000, cmd=["/bin/sh"], layers=1),
    "ubuntu:24.04": dict(distro="ubuntu24", size=78_100_000, cmd=["/bin/bash"], layers=1),
    "ubuntu:22.04": dict(distro="ubuntu22", size=77_900_000, cmd=["/bin/bash"], layers=1),
    "debian:bookworm-slim": dict(distro="debian", size=74_800_000, cmd=["bash"], layers=1),
    "debian:bookworm": dict(distro="debian", size=117_000_000, cmd=["bash"], layers=1, extra=["curl", "wget"]),
    "python:3.12-slim": dict(distro="debian", size=124_000_000, cmd=["python3"], layers=4, python="local",
                             env=PY_ENV),
    "python:3.11-slim": dict(distro="debian", size=130_000_000, cmd=["python3"], layers=4, python="local",
                             env=[e.replace("3.12.7", "3.11.10") for e in PY_ENV]),
    "python:3.12": dict(distro="debian", size=1_020_000_000, cmd=["python3"], layers=7, python="local", env=PY_ENV,
                        extra=["curl", "wget", "git", "less", "ps", "vim"]),
    "python:3.12-alpine": dict(distro="alpine", size=50_400_000, cmd=["python3"], layers=4, python="local",
                               env=PY_ENV),
    "nginx:1.27": dict(distro="debian", size=192_000_000, cmd=["nginx", "-g", "daemon off;"], layers=7,
                       entrypoint=["/docker-entrypoint.sh"], expose=["80/tcp"], stopsignal="SIGQUIT",
                       extra=["nginx", "curl"], env=[f"PATH={PATH}", "NGINX_VERSION=1.27.2"]),
    "nginx:alpine": dict(distro="alpine", size=47_900_000, cmd=["nginx", "-g", "daemon off;"], layers=8,
                         entrypoint=["/docker-entrypoint.sh"], expose=["80/tcp"], stopsignal="SIGQUIT",
                         extra=["nginx", "curl"], env=[f"PATH={PATH}", "NGINX_VERSION=1.27.2"]),
    "postgres:16": dict(distro="debian", size=435_000_000, cmd=["postgres"], layers=14,
                        entrypoint=["docker-entrypoint.sh"], expose=["5432/tcp"], stopsignal="SIGINT",
                        extra=["postgres", "psql", "pg_isready", "createdb", "ps"], volumes=["/var/lib/postgresql/data"],
                        env=[f"PATH={PATH}:/usr/lib/postgresql/16/bin", "PG_MAJOR=16", "PG_VERSION=16.4-1.pgdg120+2",
                             "PGDATA=/var/lib/postgresql/data", "LANG=en_US.utf8"]),
    "curlimages/curl:8.10.1": dict(distro="alpine", size=21_900_000, cmd=["curl"], entrypoint=["/entrypoint.sh"],
                                   layers=9, extra=["curl"], user="curl_user"),
    "nvidia/cuda:12.4.1-base-ubuntu22.04": dict(distro="ubuntu22", size=243_000_000, cmd=["/bin/bash"], layers=5,
                                                cuda="12.4", env=[f"PATH={PATH}", "NVIDIA_VISIBLE_DEVICES=all",
                                                                  "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
                                                                  "CUDA_VERSION=12.4.1"]),
    "nvidia/cuda:12.6.2-base-ubuntu22.04": dict(distro="ubuntu22", size=247_000_000, cmd=["/bin/bash"], layers=5,
                                                cuda="12.6", env=[f"PATH={PATH}", "NVIDIA_VISIBLE_DEVICES=all",
                                                                  "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
                                                                  "CUDA_VERSION=12.6.2"]),
    "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime": dict(
        distro="ubuntu22", size=7_600_000_000, cmd=["/bin/bash"], layers=4, python="conda", cuda="12.4",
        packages=[("torch", "2.5.1"), ("numpy", "2.1.2"), ("pillow", "10.4.0"), ("pyyaml", "6.0.2")],
        workdir="/workspace", env=["PATH=/opt/conda/bin:" + PATH, "NVIDIA_VISIBLE_DEVICES=all", "PYTORCH_VERSION=2.5.1"]),
    "quay.io/jupyter/scipy-notebook:latest": dict(
        distro="ubuntu24", size=4_210_000_000, cmd=["start-notebook.py"], entrypoint=["tini", "-g", "--",
                                                                                     "start.sh"], layers=30,
        python="conda", user="1000", workdir="/home/jovyan", expose=["8888/tcp"],
        extra=["start-notebook.py", "start.sh", "tini", "jupyter", "jupyter-lab", "git", "less", "ps"],
        packages=[("numpy", "2.1.2"), ("pandas", "2.2.3"), ("matplotlib", "3.9.2"), ("scikit-learn", "1.5.2"),
                  ("jupyterlab", "4.2.5")],
        env=["PATH=/opt/conda/bin:" + PATH, "HOME=/home/jovyan", "NB_USER=jovyan", "NB_UID=1000"]),
    # teaching images (fictional, on Docker Hub under mlgym/)
    "mlgym/trainer:1.0": dict(distro="debian", size=1_450_000_000, cmd=["python", "train.py", "--workers", "4"],
                              layers=6, python="local", workdir="/workspace", env=PY_ENV,
                              packages=[("torch", "2.4.1"), ("numpy", "2.0.2")], files="trainer"),
    "mlgym/memhog:1.0": dict(distro="debian", size=80_100_000, cmd=["memhog", "1500"], layers=2, extra=["memhog"]),
    "mlgym/crasher:1.0": dict(distro="debian", size=80_000_000, cmd=["sh", "-c", "exit 0"], layers=2,
                              extra=["segv"]),
}
ALIASES = {"hello-world": "hello-world:latest", "busybox": "busybox:latest", "alpine": "alpine:3.20",
           "alpine:latest": "alpine:3.20", "ubuntu": "ubuntu:24.04", "ubuntu:latest": "ubuntu:24.04",
           "ubuntu:noble": "ubuntu:24.04", "ubuntu:jammy": "ubuntu:22.04", "debian": "debian:bookworm",
           "debian:latest": "debian:bookworm", "python": "python:3.12", "python:latest": "python:3.12",
           "python:3": "python:3.12", "python:slim": "python:3.12-slim", "python:3-slim": "python:3.12-slim",
           "python:3.12-slim-bookworm": "python:3.12-slim", "nginx": "nginx:1.27", "nginx:latest": "nginx:1.27",
           "nginx:stable": "nginx:1.27", "postgres": "postgres:16", "postgres:latest": "postgres:16",
           "postgres:16-bookworm": "postgres:16", "curlimages/curl": "curlimages/curl:8.10.1",
           "curlimages/curl:latest": "curlimages/curl:8.10.1",
           "quay.io/jupyter/scipy-notebook": "quay.io/jupyter/scipy-notebook:latest",
           "jupyter/scipy-notebook": "quay.io/jupyter/scipy-notebook:latest"}

TRAINER = '''"""Train a small classifier; DataLoader workers need shared memory (/dev/shm)."""
import argparse
import os
import sys

p = argparse.ArgumentParser()
p.add_argument("--epochs", type=int, default=2)
p.add_argument("--workers", type=int, default=4)
p.add_argument("--out", default="outputs")
a = p.parse_args()
shm = int(os.environ.get("_SHM_MB", "64"))
if a.workers > 0 and shm < 512:
    print("ERROR: Unexpected bus error encountered in worker. This might be caused by insufficient shared "
          "memory (shm).", file=sys.stderr)
    sys.exit(1)
gpus = os.environ.get("_GPUS", "")
print(f"devices: {gpus or 'cpu'}")
os.makedirs(a.out, exist_ok=True)
for e in range(1, a.epochs + 1):
    print(f"epoch {e}/{a.epochs} loss {1.0 / e:.3f}")
with open(os.path.join(a.out, "model.pt"), "w") as f:
    f.write("weights\\n")
print(f"saved {a.out}/model.pt")
'''


def canonical(ref: str) -> str:
    """'nginx' -> 'nginx:1.27' (a known image), else the reference with ':latest' added."""
    r = ref
    for pre in ("docker.io/library/", "docker.io/", "index.docker.io/library/"):
        if r.startswith(pre):
            r = r[len(pre):]
    if "@" in r:
        r = r.split("@", 1)[0]
    if r in ALIASES:
        return ALIASES[r]
    if r in IMAGES:
        return r
    last = r.rsplit("/", 1)[-1]
    return r if ":" in last else r + ":latest"


def digest_of(name: str) -> str:
    return "sha256:" + hashlib.sha256(f"manifest:{name}".encode()).hexdigest()


def layer_ids(name: str, n: int) -> list[str]:
    return [hashlib.sha256(f"layer:{name}:{i}".encode()).hexdigest() for i in range(n)]


def build_rootfs(world, name: str):
    """A fresh filesystem for a catalog image."""
    from ..exec import stub_bytes
    from ..vfs import FS
    spec = IMAGES[name]
    fs = FS(lambda: world.kernel.now)
    distro = spec.get("distro")

    def d(path, mode=0o755, uid=0, gid=0):
        n = fs.mkdir(path, "/", mode, uid, gid, parents=True)
        n.mode, n.uid, n.gid = mode, uid, gid
        return n

    def f(path, text, mode=0o644, uid=0, gid=0):
        n = fs.write_file(path, text, "/", mode, uid, gid)
        n.mode = mode
        return n

    def prog(path, util):
        n = fs.write_file(path, stub_bytes(util), "/", 0o755, 0, 0)
        n.mode = 0o755
    if distro:
        for p in ("/usr/bin", "/usr/sbin", "/usr/lib", "/usr/local/bin", "/usr/local/lib", "/etc", "/var/lib",
                  "/var/log", "/opt", "/srv", "/mnt", "/home", "/run", "/proc", "/sys", "/dev"):
            d(p)
        for link, target in (("/bin", "usr/bin"), ("/sbin", "usr/sbin"), ("/lib", "usr/lib")):
            fs.symlink(target, link)
        d("/root", 0o700)
        d("/tmp", 0o1777)
        d("/dev/shm", 0o1777)
        for n_ in ("null", "zero", "urandom", "random", "tty"):
            fs.mknod("/dev/" + n_, n_, 0o666)
        progs = {"debian": DEBIAN, "ubuntu24": DEBIAN, "ubuntu22": DEBIAN, "alpine": ALPINE,
                 "busybox": BUSYBOX}[distro]
        for p in progs + list(spec.get("extra", [])):
            prog(f"/usr/bin/{p}", "dash" if p == "sh" and distro.startswith(("debian", "ubuntu")) else p)
        users = [("root", 0, 0, "/root", "/bin/bash" if distro not in ("alpine", "busybox") else "/bin/sh"),
                 ("daemon", 1, 1, "/usr/sbin", "/usr/sbin/nologin"), ("www-data", 33, 33, "/var/www",
                                                                         "/usr/sbin/nologin"),
                 ("nobody", 65534, 65534, "/nonexistent", "/usr/sbin/nologin")]
        if "postgres" in spec.get("extra", []):
            users.append(("postgres", 999, 999, "/var/lib/postgresql", "/bin/bash"))
            d("/var/lib/postgresql/data", 0o700, 999, 999)
        if spec.get("user") == "1000":
            users.append(("jovyan", 1000, 100, "/home/jovyan", "/bin/bash"))
            d("/home/jovyan/work", 0o755, 1000, 100)
        if spec.get("user") == "curl_user":
            users.append(("curl_user", 100, 101, "/home/curl_user", "/sbin/nologin"))
        f("/etc/passwd", "".join(f"{u}:x:{i}:{g}:{u}:{h}:{s}\n" for u, i, g, h, s in users))
        f("/etc/group", "root:x:0:\ndaemon:x:1:\nwww-data:x:33:\nusers:x:100:\nnogroup:x:65534:\n" +
          ("postgres:x:999:\n" if "postgres" in spec.get("extra", []) else ""))
        f("/etc/hostname", "localhost\n")
        f("/etc/hosts", "127.0.0.1\tlocalhost\n::1\tlocalhost ip6-localhost ip6-loopback\n")
        f("/etc/os-release", OS_RELEASE.get(distro, ""))
        f("/etc/resolv.conf", "nameserver 127.0.0.11\noptions ndots:0\n")
        if distro.startswith(("debian", "ubuntu")):
            f("/root/.bashrc", "PS1='${debian_chroot:+($debian_chroot)}\\u@\\h:\\w\\$ '\n")
            f("/etc/debian_version", "12.7\n" if distro == "debian" else "trixie/sid\n")
            d("/var/lib/apt/lists")
            d("/var/lib/bashlite/apt")
    else:
        d("/")
    if spec.get("python") == "local":
        for p in ("python3.12", "pip"):
            prog(f"/usr/local/bin/{p}", p)
        for alias, target in (("python3", "python3.12"), ("python", "python3.12"), ("pip3", "pip")):
            fs.symlink(target, f"/usr/local/bin/{alias}")
        d("/usr/local/lib/python3.12/site-packages")
        _dist(fs, "/usr/local/lib/python3.12/site-packages", "pip", "24.3.1")
        for pkg, ver in spec.get("packages", []):
            _dist(fs, "/usr/local/lib/python3.12/site-packages", pkg, ver)
    if spec.get("python") == "conda":
        for p in ("python3.12", "pip", "conda"):
            prog(f"/opt/conda/bin/{p}", p)
        for alias in ("python3", "python"):
            fs.symlink("python3.12", f"/opt/conda/bin/{alias}")
        d("/opt/conda/conda-meta")
        site = "/opt/conda/lib/python3.12/site-packages"
        d(site)
        for pkg, ver in [("pip", "24.2")] + list(spec.get("packages", [])):
            _dist(fs, site, pkg, ver)
            if pkg == "torch":
                prog("/opt/conda/bin/torchrun", "torchrun")
        if "start-notebook.py" in spec.get("extra", []):
            for p in ("jupyter", "jupyter-lab"):
                prog(f"/opt/conda/bin/{p}", p)
    for p in spec.get("extra", []):
        if p == "hello":
            prog("/hello", "hello")
        if p == "nginx":
            d("/usr/share/nginx/html")
            f("/usr/share/nginx/html/index.html", NGINX_INDEX)
            f("/usr/share/nginx/html/50x.html", "<!DOCTYPE html>\n<html><body><h1>An error occurred.</h1></body>"
                                                "</html>\n")
            f("/etc/nginx/nginx.conf", NGINX_CONF)
            f("/etc/nginx/conf.d/default.conf", NGINX_DEFAULT)
            d("/var/log/nginx")
            fs.symlink("/dev/stdout", "/var/log/nginx/access.log")
            fs.symlink("/dev/stderr", "/var/log/nginx/error.log")
            prog("/docker-entrypoint.sh", "docker-entrypoint")
        if p == "postgres":
            prog("/usr/local/bin/docker-entrypoint.sh", "docker-entrypoint")
            fs.symlink("usr/local/bin/docker-entrypoint.sh", "/docker-entrypoint.sh")
        if p == "curl" and spec.get("user") == "curl_user":
            prog("/entrypoint.sh", "docker-entrypoint")
        if p == "start-notebook.py":
            prog("/usr/local/bin/start-notebook.py", "start-notebook.py")
            prog("/usr/local/bin/start.sh", "docker-entrypoint")
            prog("/usr/bin/tini", "tini")
    if spec.get("files") == "trainer":
        d("/workspace")
        f("/workspace/train.py", TRAINER)
        f("/workspace/requirements.txt", "torch==2.4.1\nnumpy==2.0.2\n")
    return fs


def _dist(fs, site, name, ver):
    d = f"{site}/{name.replace('-', '_')}-{ver}.dist-info"
    fs.mkdir(d, "/", 0o755, 0, 0, parents=True)
    fs.write_file(f"{d}/METADATA", f"Metadata-Version: 2.1\nName: {name}\nVersion: {ver}\n", "/", 0o644, 0, 0)
    fs.write_file(f"{d}/INSTALLER", "pip\n", "/", 0o644, 0, 0)


NGINX_INDEX = ("<!DOCTYPE html>\n<html>\n<head>\n<title>Welcome to nginx!</title>\n</head>\n<body>\n"
               "<h1>Welcome to nginx!</h1>\n<p>If you see this page, the nginx web server is installed and working.</p>"
               "\n</body>\n</html>\n")
NGINX_CONF = ("user  nginx;\nworker_processes  auto;\n\nerror_log  /var/log/nginx/error.log notice;\npid        "
              "/var/run/nginx.pid;\n\nevents {\n    worker_connections  1024;\n}\n\nhttp {\n    include       "
              "/etc/nginx/mime.types;\n    default_type  application/octet-stream;\n    access_log  "
              "/var/log/nginx/access.log;\n    sendfile        on;\n    keepalive_timeout  65;\n    include "
              "/etc/nginx/conf.d/*.conf;\n}\n")
NGINX_DEFAULT = ("server {\n    listen       80;\n    server_name  localhost;\n\n    location / {\n        root   "
                 "/usr/share/nginx/html;\n        index  index.html index.htm;\n    }\n\n    error_page   500 502 503 "
                 "504  /50x.html;\n    location = /50x.html {\n        root   /usr/share/nginx/html;\n    }\n}\n")
