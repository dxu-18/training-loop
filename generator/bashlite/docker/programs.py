"""Programs that live inside images: hello (hello-world), the entrypoint scripts, nginx, the postgres server,
uvicorn (a FastAPI app's routes), Jupyter's start-notebook.py, tini, huggingface-cli, the teaching programs memhog
and segv, and the account / package tools images use (useradd, adduser, groupadd, usermod, apk)."""

from __future__ import annotations

import datetime
import json
import posixpath
import re

from ..exec import util
from ..kernel import NS
from ..vfs import FSError


def _container(ctx):
    return ctx.host.extra.get("container")


def _log(ctx, text, stream="out"):
    (ctx.w if stream == "out" else ctx.e)(text)


def _stamp(ctx, fmt="%Y/%m/%d %H:%M:%S"):
    return datetime.datetime.fromtimestamp(ctx.now / NS, tz=datetime.timezone.utc).strftime(fmt)


def _forever(ctx):
    yield from ctx.flush()
    while True:
        yield ("forever",)


# ---------------------------------------------------------------------------- hello-world

HELLO = """
Hello from Docker!
This message shows that your installation appears to be working correctly.

To generate this message, Docker took the following steps:
 1. The Docker client contacted the Docker daemon.
 2. The Docker daemon pulled the "hello-world" image from the Docker Hub.
    (amd64)
 3. The Docker daemon created a new container from that image which runs the
    executable that produced the output you are currently reading.
 4. The Docker daemon streamed that output to the Docker client, which sent it
    to your terminal.

To try something more ambitious, you can run an Ubuntu container with:
 $ docker run -it ubuntu bash

Share images, automate workflows, and more with a free Docker ID:
 https://hub.docker.com/

For more examples and ideas, visit:
 https://docs.docker.com/get-started/

"""


@util("hello")
def u_hello(ctx):
    ctx.w(HELLO)
    return 0
    yield


# ---------------------------------------------------------------------------- entrypoints

@util("docker-entrypoint")
def u_entrypoint(ctx):
    """The images' entrypoint scripts: set things up, then become the command (exec "$@")."""
    from ..exec import exec_program, find_in_path
    args = ctx.argv[1:]
    script = posixpath.basename(ctx.argv[0])
    env = ctx.env
    if not args:
        return 0
    prog = args[0]
    if script == "docker-entrypoint.sh" and prog == "postgres":
        if not env.get("POSTGRES_PASSWORD") and env.get("POSTGRES_HOST_AUTH_METHOD") != "trust":
            ctx.e("Error: Database is uninitialized and superuser password is not specified.\n"
                  "       You must specify POSTGRES_PASSWORD to a non-empty value for the\n"
                  "       superuser. For example, \"-e POSTGRES_PASSWORD=password\" on \"docker run\".\n\n"
                  "       You may also use \"POSTGRES_HOST_AUTH_METHOD=trust\" to allow all\n"
                  "       connections without a password. This is *not* recommended.\n\n"
                  "       See PostgreSQL documentation about \"trust\":\n"
                  "       https://www.postgresql.org/docs/current/auth-trust.html\n")
            return 1
    if script == "docker-entrypoint.sh" and prog == "postgres" and ctx.uid == 0:
        ctx.proc.uid = ctx.proc.gid = 999                 # the real entrypoint drops to the postgres user (gosu)
        ctx.proc.gids = {999}
        env = dict(env, HOME="/var/lib/postgresql")
        try:
            n = ctx.fs.lookup(env.get("PGDATA", "/var/lib/postgresql/data"))
            n.uid = n.gid = 999
        except FSError:
            pass
    if script == "docker-entrypoint.sh" and prog.startswith("nginx"):
        ctx.w("/docker-entrypoint.sh: /docker-entrypoint.d/ is not empty, will attempt to perform configuration\n"
              "/docker-entrypoint.sh: Looking for shell scripts in /docker-entrypoint.d/\n"
              "/docker-entrypoint.sh: Launching /docker-entrypoint.d/10-listen-on-ipv6-by-default.sh\n"
              "10-listen-on-ipv6-by-default.sh: info: Enabled listen on IPv6 in /etc/nginx/conf.d/default.conf\n"
              "/docker-entrypoint.sh: Configuration complete; ready for start up\n")
    yield from ctx.flush()
    path = find_in_path(ctx.fs, prog, env.get("PATH", ""), ctx.cwd)
    if path is None or not ctx.fs.exists(path):
        ctx.e(f"{ctx.argv[0]}: line 1: exec: {prog}: not found\n")
        return 127
    ctx.proc.argv = list(args)
    return (yield from exec_program(ctx.proc, path, list(args), dict(env), None))


@util("tini")
def u_tini(ctx):
    from ..exec import exec_program, find_in_path
    args = [a for a in ctx.argv[1:] if a not in ("-g", "-s", "--")]
    if not args:
        ctx.e("tini (tini version 0.19.0)\nUsage: tini [OPTIONS] PROGRAM -- [ARGS]\n")
        return 1
    path = find_in_path(ctx.fs, args[0], ctx.env.get("PATH", ""), ctx.cwd)
    if path is None:
        ctx.e(f"[FATAL tini (7)] exec {args[0]} failed: No such file or directory\n")
        return 127
    return (yield from exec_program(ctx.proc, path, args, dict(ctx.env), None))


# ---------------------------------------------------------------------------- nginx

def _nginx_conf(fs):
    """(listen port, root, [(location prefix, proxy upstream)]) from conf.d/*.conf."""
    port, root, proxies = 80, "/usr/share/nginx/html", []
    try:
        d = fs.lookup("/etc/nginx/conf.d")
        texts = [bytes(n.data).decode("utf-8", "replace") for k, n in sorted(d.entries.items())
                 if k.endswith(".conf") and n.type == "f"]
    except FSError:
        texts = []
    for t in texts:
        m = re.search(r"listen\s+(\d+)", t)
        if m:
            port = int(m.group(1))
        m = re.search(r"location\s+/\s*\{[^}]*?root\s+([^;\s]+)\s*;", t, re.S) or re.search(r"\broot\s+([^;\s]+)\s*;", t)
        if m:
            root = m.group(1)
        for lm in re.finditer(r"location\s+(\S+)\s*\{([^}]*)\}", t, re.S):
            pm = re.search(r"proxy_pass\s+http://([\w.-]+)(?::(\d+))?(/?\S*?)\s*;", lm.group(2))
            if pm:
                proxies.append((lm.group(1), pm.group(1), int(pm.group(2) or 80)))
    return port, root, proxies


@util("nginx")
def u_nginx(ctx):
    a = ctx.argv[1:]
    if "-v" in a or "-V" in a:
        ctx.e("nginx version: nginx/1.27.2\n")
        return 0
    if "-t" in a:
        ctx.e("nginx: the configuration file /etc/nginx/nginx.conf syntax is ok\nnginx: configuration file "
              "/etc/nginx/nginx.conf test is successful\n")
        return 0
    if "-s" in a:
        return 0
    port, root, proxies = _nginx_conf(ctx.fs)
    host = ctx.host
    if port in host.services:
        ctx.e(f"{_stamp(ctx)} [emerg] 1#1: bind() to 0.0.0.0:{port} failed (98: Address already in use)\n")
        return 1
    from ..world import Service
    fs = host.fs
    world = ctx.world
    c = _container(ctx)

    def access(code, size, method, path, headers):
        ip = "172.17.0.1"
        ua = headers.get("User-Agent", "curl/8.5.0")
        line = (f'{ip} - - [{_stamp(ctx, "%d/%b/%Y:%H:%M:%S +0000")}] "{method} {path} HTTP/1.1" {code} {size} '
                f'"-" "{ua}" "-"\n')
        if c is not None:
            c.logs.append((world.kernel.now, "out", line.encode()))

    def handler(method, path, headers, body):
        from ..utils.net import HttpResult, mime_of, http_request
        p = path.split("?", 1)[0]
        for prefix, up, upport in proxies:
            if p.startswith(prefix):
                resolver = host.extra.get("resolver")
                target = resolver(up) if resolver else world.resolve(up)
                svc = target.services.get(upport) if target is not None else None
                if svc is None or svc.handler is None:
                    access(502, 157, method, path, headers)
                    return HttpResult(502, {"Content-Type": "text/html", "Server": "nginx/1.27.2"},
                                      b"<html>\r\n<head><title>502 Bad Gateway</title></head>\r\n<body>\r\n<center><h1>"
                                      b"502 Bad Gateway</h1></center>\r\n<hr><center>nginx/1.27.2</center>\r\n</body>\r\n"
                                      b"</html>\r\n")
                res = svc.handler(method, path, headers, body)
                access(res.status, len(res.body), method, path, headers)
                return res
        fpath = posixpath.normpath(root + "/" + p.lstrip("/"))
        try:
            node = fs.lookup(fpath)
            if node.type == "d":
                fpath = fpath.rstrip("/") + "/index.html"
                node = fs.lookup(fpath)
            if not node.mode & 0o004 and node.uid != 101:
                access(403, 153, method, path, headers)
                return HttpResult(403, {"Content-Type": "text/html", "Server": "nginx/1.27.2"},
                                  b"<html>\r\n<head><title>403 Forbidden</title></head>\r\n<body>\r\n<center><h1>403 "
                                  b"Forbidden</h1></center>\r\n<hr><center>nginx/1.27.2</center>\r\n</body>\r\n"
                                  b"</html>\r\n")
            data = bytes(node.data)
            access(200, len(data), method, path, headers)
            return HttpResult(200, {"Content-Type": mime_of(fpath), "Server": "nginx/1.27.2"},
                              b"" if method == "HEAD" else data)
        except FSError:
            access(404, 153, method, path, headers)
            return HttpResult(404, {"Content-Type": "text/html", "Server": "nginx/1.27.2"},
                              b"<html>\r\n<head><title>404 Not Found</title></head>\r\n<body>\r\n<center><h1>404 Not "
                              b"Found</h1></center>\r\n<hr><center>nginx/1.27.2</center>\r\n</body>\r\n</html>\r\n")
    host.services[port] = Service(port, ctx.proc, handler, "nginx")
    ctx.e(f"{_stamp(ctx)} [notice] 1#1: using the \"epoll\" event method\n{_stamp(ctx)} [notice] 1#1: nginx/1.27.2\n"
          f"{_stamp(ctx)} [notice] 1#1: OS: Linux {host.kernel_release}\n{_stamp(ctx)} [notice] 1#1: start worker "
          f"processes\n")
    ctx.proc.on_exit.append(lambda p: host.services.pop(port, None))
    yield from _forever(ctx)


# ---------------------------------------------------------------------------- postgres

@util("postgres")
def u_postgres(ctx):
    from ...postgreslite.server import Cluster
    host = ctx.host
    env = ctx.env
    if ctx.uid == 0:
        ctx.e('"root" execution of the PostgreSQL server is not permitted.\nThe server must be started under an '
              'unprivileged user ID to prevent\npossible system security compromise.  See the documentation for\nmore '
              'information on how to properly start the server.\n')
        return 1
    pgdata = env.get("PGDATA", "/var/lib/postgresql/data")
    fs = host.fs
    fresh = True
    try:
        fresh = "PG_VERSION" not in fs.lookup(pgdata).entries
    except FSError:
        fs.mkdir(pgdata, "/", 0o700, ctx.uid, ctx.proc.gid, parents=True)
    store = ctx.world.__dict__.setdefault("pg_volumes", {})
    data_node = fs.lookup(pgdata)
    cluster = store.get(id(data_node))
    if fresh:
        ctx.w("The files belonging to this database system will be owned by user \"postgres\".\nThis user must also "
              "own the server process.\n\nThe database cluster will be initialized with locale \"en_US.utf8\".\n"
              "The default database encoding has accordingly been set to \"UTF8\".\n\nfixing permissions on existing "
              f"directory {pgdata} ... ok\ncreating subdirectories ... ok\nperforming post-bootstrap initialization "
              "... ok\nsyncing data to disk ... ok\n\n\nSuccess. You can now start the database server using:\n\n"
              f"    pg_ctl -D {pgdata} -l logfile start\n\n")
        fs.write_file(pgdata + "/PG_VERSION", "16\n", "/", 0o600, ctx.uid, ctx.proc.gid)
        cluster = Cluster()
        store[id(data_node)] = cluster
        db = env.get("POSTGRES_DB")
        if db and db not in cluster.databases:
            try:
                cluster.create_database(db) if hasattr(cluster, "create_database") else None
            except Exception:  # noqa: BLE001
                pass
        yield ("sleep", ctx.now + int(1.5 * NS))
        ctx.w("PostgreSQL init process complete; ready for start up.\n\n")
    elif cluster is None:
        cluster = Cluster()
        store[id(data_node)] = cluster
        ctx.w("\nPostgreSQL Database directory appears to contain a database; Skipping initialization\n\n")
    from ..world import Service
    host.extra["pg"] = cluster
    if env.get("POSTGRES_PASSWORD"):
        host.extra["pg_password"] = env["POSTGRES_PASSWORD"]
    ts = _stamp(ctx, "%Y-%m-%d %H:%M:%S.000 UTC")
    ctx.e(f"{ts} [1] LOG:  starting PostgreSQL 16.4 (Debian 16.4-1.pgdg120+2) on x86_64-pc-linux-gnu, compiled by gcc "
          f"(Debian 12.2.0-14) 12.2.0, 64-bit\n{ts} [1] LOG:  listening on IPv4 address \"0.0.0.0\", port 5432\n"
          f"{ts} [1] LOG:  listening on Unix socket \"/var/run/postgresql/.s.PGSQL.5432\"\n"
          f"{ts} [1] LOG:  database system is ready to accept connections\n")
    host.services[5432] = Service(5432, ctx.proc, None, "postgres")
    ctx.proc.on_exit.append(lambda p: host.services.pop(5432, None))
    yield from _forever(ctx)


# ---------------------------------------------------------------------------- uvicorn (a FastAPI app)

def _routes(text):
    """[(method, path, function name, body text)] from FastAPI decorators."""
    out = []
    for m in re.finditer(r"@\w+\.(get|post|put|delete)\(\s*[\"']([^\"']+)[\"']\s*\)\s*\n(?:async\s+)?def\s+(\w+)\(([^)]*)\)"
                         r":\s*\n((?:[ \t]+.*\n?)*)", text):
        out.append((m.group(1).upper(), m.group(2), m.group(3), m.group(5), m.group(4)))
    return out


def _sentiment(text):
    neg = ("hate", "terrible", "awful", "bad", "worst", "boring", "slow", "broken", "sad", "angry")
    t = text.lower()
    n = sum(w in t for w in neg)
    return {"label": "NEGATIVE" if n else "POSITIVE", "score": round(0.9991 - (len(t) % 9) / 1000 if n else
                                                                     0.9998 - (len(t) % 7) / 10000, 4)}


@util("uvicorn")
def u_uvicorn(ctx):
    a = ctx.argv[1:]
    if not a or a[0] in ("--help", "-h"):
        ctx.w("Usage: uvicorn [OPTIONS] APP\n")
        return 0 if a else 2
    app = next((x for x in a if not x.startswith("-") and ":" in x), None)
    port = int(a[a.index("--port") + 1]) if "--port" in a else 8000
    hostbind = a[a.index("--host") + 1] if "--host" in a else "127.0.0.1"
    if app is None:
        ctx.e("Usage: uvicorn [OPTIONS] APP\nTry 'uvicorn --help' for help.\n\nError: Missing argument 'APP'.\n")
        return 2
    mod, _, attr = app.partition(":")
    path = mod.replace(".", "/") + ".py"
    try:
        text = bytes(ctx.fs.lookup(path, ctx.cwd).data).decode("utf-8", "replace")
    except FSError:
        ctx.e(f"ERROR:    Error loading ASGI app. Could not import module \"{mod}\".\n")
        return 1
    # imports must be installed
    from ..utils.pyenv import env_of, installed
    root, kind = env_of(ctx, "/usr/local/bin/python")
    have = installed(ctx, root, kind)
    for m in re.finditer(r"^\s*(?:from|import)\s+(\w+)", text, re.M):
        top = m.group(1)
        dist = {"fastapi": "fastapi", "pydantic": "pydantic", "transformers": "transformers", "torch": "torch"}.get(top)
        if dist and dist not in have:
            ctx.e(f"Traceback (most recent call last):\n  File \"/usr/local/bin/uvicorn\", line 8, in <module>\n"
                  f"ModuleNotFoundError: No module named '{top}'\n")
            return 1
    model = None
    mm = re.search(r"pipeline\(\s*[\"']([\w-]+)[\"']\s*,\s*model=[\"']([^\"']+)[\"']", text)
    if mm:
        model = mm.group(2)
        hf = ctx.env.get("HF_HOME") or (ctx.env.get("HOME", "/root") + "/.cache/huggingface")
        cache = f"{hf}/hub/models--{model.replace('/', '--')}"
        if not ctx.fs.exists(cache):
            if ctx.host.extra.get("offline") or (_container(ctx) and _container(ctx).cfg.get("network") == "none"):
                ctx.e(f"OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find "
                      f"them in the cached files.\nCheck your internet connection or see how to run the library in "
                      f"offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.\n")
                return 1
            ctx.e(f"config.json: 100%|██████████| 629/629 [00:00<00:00, 3.1MB/s]\nmodel.safetensors: 100%|██████████| "
                  f"268M/268M [00:06<00:00, 41.2MB/s]\ntokenizer_config.json: 100%|██████████| 48.0/48.0 [00:00<00:00]\n"
                  f"vocab.txt: 100%|██████████| 232k/232k [00:00<00:00, 9.80MB/s]\n")
            _hf_download(ctx, model, hf)
            yield ("sleep", ctx.now + int(6 * NS))
        ctx.e("Device set to use cpu\n")
    routes = _routes(text)
    pid = ctx.proc.pid
    ctx.e(f"INFO:     Started server process [{pid}]\nINFO:     Waiting for application startup.\nINFO:     "
          f"Application startup complete.\nINFO:     Uvicorn running on http://{hostbind}:{port} (Press CTRL+C to "
          f"quit)\n")
    from ..world import Service
    host = ctx.host
    c = _container(ctx)
    world = ctx.world

    def handler(method, path, headers, body):
        from ..utils.net import HttpResult
        p = path.split("?", 1)[0]
        for meth, rpath, fname, fbody, params in routes:
            if rpath == p and meth == method:
                status, payload = _call_route(fbody, params, body, model)
                code_line = {200: "200 OK", 422: "422 Unprocessable Entity", 500: "500 Internal Server Error"}[status]
                line = f'INFO:     172.17.0.1:51234 - "{method} {path} HTTP/1.1" {code_line}\n'
                if c is not None:
                    c.logs.append((world.kernel.now, "err", line.encode()))
                return HttpResult(status, {"content-type": "application/json", "server": "uvicorn"},
                                  json.dumps(payload, separators=(",", ":")).encode())
        allowed = [m for m, rp, *_ in routes if rp == p]
        st = 405 if allowed else 404
        if c is not None:
            c.logs.append((world.kernel.now, "err", f'INFO:     172.17.0.1:51234 - "{method} {path} HTTP/1.1" '
                                                     f'{st} {"Method Not Allowed" if allowed else "Not Found"}\n'.encode()))
        return HttpResult(st, {"content-type": "application/json", "server": "uvicorn"},
                          b'{"detail":"Method Not Allowed"}' if allowed else b'{"detail":"Not Found"}')
    svc = Service(port, ctx.proc, handler, "uvicorn")
    svc.loopback = hostbind in ("127.0.0.1", "localhost")     # reachable only from inside the container
    host.services[port] = svc
    ctx.proc.on_exit.append(lambda p: host.services.pop(port, None))
    yield from _forever(ctx)


def _call_route(fbody, params, body, model):
    """What a simple route returns: a literal dict, the pipeline's prediction for req.text, or a list of them."""
    lit = re.search(r"return\s+(\{[^\n]*\})", fbody)
    try:
        data = json.loads(body.decode() or "null") if body else None
    except ValueError:
        return 422, {"detail": [{"type": "json_invalid", "loc": ["body", 0], "msg": "JSON decode error"}]}
    if "clf(" in fbody or "pipeline" in fbody or "classifier(" in fbody:
        if not isinstance(data, dict):
            return 422, {"detail": [{"type": "missing", "loc": ["body"], "msg": "Field required", "input": None}]}
        if "texts" in fbody and isinstance(data.get("texts"), list):
            return 200, [_sentiment(t) for t in data["texts"]]
        if "text" not in data:
            return 422, {"detail": [{"type": "missing", "loc": ["body", "text"], "msg": "Field required",
                                     "input": data}]}
        return 200, _sentiment(str(data["text"]))
    if lit:
        try:
            return 200, json.loads(lit.group(1).replace("'", '"'))
        except ValueError:
            pass
    return 200, None


def _hf_download(ctx, model, hf):
    from ..vfs import LazyData
    base = f"{hf}/hub/models--{model.replace('/', '--')}/snapshots/714eb0fa89d2f80546fda750413ed43d93601a13"
    for name, size in (("config.json", 629), ("model.safetensors", 267_832_558), ("tokenizer_config.json", 48),
                       ("vocab.txt", 231_508)):
        n = ctx.fs.write_file(f"{base}/{name}", b"", "/", 0o644, ctx.uid, ctx.proc.gid)
        n.data = LazyData(size=size, seed=f"hf:{model}:{name}", head=b"{" if name.endswith(".json") else b"")


@util("huggingface-cli")
def u_hf_cli(ctx):
    a = ctx.argv[1:]
    if not a or a[0] != "download" or len(a) < 2:
        ctx.w("usage: huggingface-cli <command> [<args>]\n\npositional arguments:\n  {download,login,whoami,...}\n")
        return 0 if not a else 2
    model = a[1]
    if ctx.host.extra.get("offline"):
        ctx.e("huggingface_hub.errors.LocalEntryNotFoundError: An error happened while trying to locate the files on "
              "the Hub and we cannot find the appropriate snapshot folder for the specified revision on the local "
              "disk. Please check your internet connection and try again.\n")
        return 1
    hf = ctx.env.get("HF_HOME") or (ctx.env.get("HOME", "/root") + "/.cache/huggingface")
    ctx.e("Fetching 4 files: 100%|██████████| 4/4 [00:05<00:00,  1.36s/it]\n")
    _hf_download(ctx, model, hf)
    yield ("sleep", ctx.now + int(5 * NS))
    ctx.w(f"{hf}/hub/models--{model.replace('/', '--')}/snapshots/714eb0fa89d2f80546fda750413ed43d93601a13\n")
    return 0


# ---------------------------------------------------------------------------- Jupyter

@util("start-notebook.py")
def u_start_notebook(ctx):
    token = __import__("hashlib").sha256(f"jupyter:{ctx.world.seed}:{ctx.host.name}".encode()).hexdigest()[:48]
    host = ctx.host
    ts = _stamp(ctx, "%Y-%m-%d %H:%M:%S.000")
    ctx.e(f"Entered start.sh with args: jupyter lab\nExecuting the command: jupyter lab\n[I {ts} ServerApp] "
          f"jupyterlab | extension was successfully loaded.\n[I {ts} ServerApp] Serving notebooks from local "
          f"directory: /home/jovyan\n[I {ts} ServerApp] Jupyter Server 2.14.2 is running at:\n[I {ts} ServerApp] "
          f"http://{host.name}:8888/lab?token={token}\n[I {ts} ServerApp]     http://127.0.0.1:8888/lab?token={token}\n"
          f"[I {ts} ServerApp] Use Control-C to stop this server and shut down all kernels (twice to skip "
          f"confirmation).\n[C {ts} ServerApp] \n\n    To access the server, open this file in a browser:\n        "
          f"file:///home/jovyan/.local/share/jupyter/runtime/jpserver-7-open.html\n    Or copy and paste one of these "
          f"URLs:\n        http://{host.name}:8888/lab?token={token}\n        http://127.0.0.1:8888/lab?token={token}\n")
    from ..world import Service

    def handler(method, path, headers, body):
        from ..utils.net import HttpResult
        if f"token={token}" in path or path.startswith("/api"):
            return HttpResult(200, {"Content-Type": "text/html"}, b"<!DOCTYPE HTML>\n<html><head><title>JupyterLab"
                                                                  b"</title></head><body></body></html>\n")
        return HttpResult(302, {"Location": "/login?next=%2Flab"}, b"")
    host.services[8888] = Service(8888, ctx.proc, handler, "jupyter")
    ctx.proc.on_exit.append(lambda p: host.services.pop(8888, None))
    yield from _forever(ctx)


# ---------------------------------------------------------------------------- teaching programs

@util("memhog")
def u_memhog(ctx):
    """memhog MB: allocates MB megabytes (and holds them), as a training job loading a dataset would."""
    mb = int(ctx.argv[1]) if len(ctx.argv) > 1 and ctx.argv[1].isdigit() else 1500
    c = _container(ctx)
    limit = c.cfg.get("memory") if c is not None else None
    ctx.w(f"loading dataset into memory ({mb} MiB)...\n")
    yield from ctx.flush()
    yield ("sleep", ctx.now + int(1.0 * NS))
    if limit and mb * 1024 * 1024 > limit:
        if c is not None:
            c.oom = True
        ctx.k.kill_now(ctx.proc, 9)
        return 137
    ctx.w("dataset loaded, training...\nepoch 1/3\nepoch 2/3\nepoch 3/3\ndone\n")
    return 0


@util("segv")
def u_segv(ctx):
    ctx.w("starting native extension...\n")
    yield from ctx.flush()
    ctx.k.kill_now(ctx.proc, 11)
    return 139


# ---------------------------------------------------------------------------- accounts and packages in images

def _passwd(fs):
    try:
        return [ln.split(":") for ln in bytes(fs.lookup("/etc/passwd").data).decode().split("\n") if ln.count(":") >= 6]
    except FSError:
        return []


def _groups(fs):
    try:
        return [ln.split(":") for ln in bytes(fs.lookup("/etc/group").data).decode().split("\n") if ln.count(":") >= 3]
    except FSError:
        return []


def _add_account(ctx, name, uid=None, gid=None, home=None, shell="/bin/sh", create_home=False, system=False,
                 group=None):
    from ..world import User
    fs = ctx.fs
    users = _passwd(fs)
    if any(u[0] == name for u in users):
        return f"user '{name}' already exists"
    used = {int(u[2]) for u in users if u[2].isdigit()}
    if uid is None:
        uid = 100 if system else 1000
        while uid in used:
            uid += 1
    groups = _groups(fs)
    if gid is None:
        g = next((x for x in groups if x[0] == (group or name)), None)
        if g is not None:
            gid = int(g[2])
        else:
            gid = uid
            if any(int(x[2]) == gid for x in groups if x[2].isdigit()):
                gid = max(int(x[2]) for x in groups if x[2].isdigit()) + 1
            text = bytes(fs.lookup("/etc/group").data).decode() if fs.exists("/etc/group") else ""
            fs.write_file("/etc/group", text + f"{group or name}:x:{gid}:\n", "/", 0o644, 0, 0)
    home = home or f"/home/{name}"
    text = bytes(fs.lookup("/etc/passwd").data).decode() if fs.exists("/etc/passwd") else ""
    fs.write_file("/etc/passwd", text + f"{name}:x:{uid}:{gid}::{home}:{shell}\n", "/", 0o644, 0, 0)
    if create_home:
        n = fs.mkdir(home, "/", 0o750, uid, gid, parents=True)
        n.uid, n.gid, n.mode = uid, gid, 0o750
    ctx.host.users[uid] = User(name, uid, gid, "", home, shell)
    ctx.host.groups.setdefault(gid, (group or name, set()))
    return None


@util("useradd")
def u_useradd(ctx):
    a = ctx.argv[1:]
    if ctx.uid != 0:
        ctx.e("useradd: Permission denied.\nuseradd: cannot lock /etc/passwd; try again later.\n")
        return 1
    uid = gid = None
    home, shell, create, system, group = None, "/bin/sh", False, False, None
    i = 0
    name = None
    while i < len(a):
        x = a[i]
        if x in ("-m", "--create-home"):
            create = True
        elif x in ("-r", "--system"):
            system = True
        elif x in ("-u", "--uid"):
            uid = int(a[i + 1])
            i += 1
        elif x in ("-g", "--gid"):
            g = a[i + 1]
            i += 1
            if g.isdigit():
                gid = int(g)
            else:
                grp = next((y for y in _groups(ctx.fs) if y[0] == g), None)
                if grp is None:
                    ctx.e(f"useradd: group '{g}' does not exist\n")
                    return 6
                gid = int(grp[2])
        elif x in ("-d", "--home-dir"):
            home = a[i + 1]
            i += 1
        elif x in ("-s", "--shell"):
            shell = a[i + 1]
            i += 1
        elif x in ("-U", "--user-group", "-N", "-M", "--no-create-home", "-l"):
            pass
        elif x.startswith("-"):
            ctx.e(f"useradd: invalid option -- '{x.lstrip('-')}'\n")
            return 2
        else:
            name = x
        i += 1
    if not name:
        ctx.e("Usage: useradd [options] LOGIN\n")
        return 2
    err = _add_account(ctx, name, uid, gid, home, shell, create, system)
    if err:
        ctx.e(f"useradd: {err}\n")
        return 9
    return 0
    yield


@util("groupadd", "addgroup")
def u_groupadd(ctx):
    a = [x for x in ctx.argv[1:] if not x.startswith("-")]
    gid = None
    if "-g" in ctx.argv:
        gid = int(ctx.argv[ctx.argv.index("-g") + 1])
        a = [x for x in a if x != str(gid)]
    if not a:
        ctx.e("Usage: groupadd [options] GROUP\n")
        return 2
    fs = ctx.fs
    groups = _groups(fs)
    if any(g[0] == a[0] for g in groups):
        ctx.e(f"groupadd: group '{a[0]}' already exists\n")
        return 9
    if gid is None:
        gid = 1000
        while any(g[2] == str(gid) for g in groups):
            gid += 1
    text = bytes(fs.lookup("/etc/group").data).decode() if fs.exists("/etc/group") else ""
    fs.write_file("/etc/group", text + f"{a[0]}:x:{gid}:\n", "/", 0o644, 0, 0)
    ctx.host.groups[gid] = (a[0], set())
    return 0
    yield


@util("adduser")
def u_adduser(ctx):
    """Debian's adduser (asks for details) or busybox's (adduser -D name)."""
    a = ctx.argv[1:]
    busy = ctx.fs.exists("/sbin/apk") or ctx.fs.exists("/usr/bin/apk") or not ctx.fs.exists("/usr/bin/useradd")
    name = None
    for i, x in enumerate(a):
        if not x.startswith("-") and (i == 0 or a[i - 1] not in ("-u", "-G", "-h", "-s", "-g", "--uid", "--gecos")):
            name = x
    if not name:
        ctx.e("adduser: Only one or two names allowed.\n")
        return 1
    uid = int(a[a.index("-u") + 1]) if "-u" in a else None
    err = _add_account(ctx, name, uid, None, None, "/bin/sh" if busy else "/bin/bash", True)
    if err:
        ctx.e(f"adduser: {err}\n" if busy else f"fatal: The user `{name}' already exists.\n")
        return 1
    if not busy:
        ctx.w(f"info: Adding user `{name}' ...\ninfo: Selecting UID/GID from range 1000 to 59999 ...\ninfo: Adding new "
              f"group `{name}' ...\ninfo: Adding new user `{name}' ...\ninfo: Creating home directory `/home/{name}' "
              f"...\n")
    return 0
    yield


@util("usermod")
def u_usermod(ctx):
    """usermod -aG GROUP USER: membership is in /etc/group (and takes effect at the next login)."""
    a = ctx.argv[1:]
    if ctx.uid != 0:
        ctx.e("usermod: Permission denied.\nusermod: cannot lock /etc/passwd; try again later.\n")
        return 1
    if "-aG" in a or ("-a" in a and "-G" in a) or "-G" in a:
        flag = "-aG" if "-aG" in a else "-G"
        grp = a[a.index(flag) + 1]
        user = [x for x in a if not x.startswith("-") and x != grp][-1]
        fs = ctx.fs
        text = bytes(fs.lookup("/etc/group").data).decode()
        lines = text.split("\n")
        found = False
        for i, ln in enumerate(lines):
            p = ln.split(":")
            if p[0] == grp and len(p) >= 4:
                members = [m for m in p[3].split(",") if m]
                if user not in members:
                    members.append(user)
                p[3] = ",".join(members)
                lines[i] = ":".join(p)
                found = True
                gid = int(p[2])
                if gid in ctx.host.groups:
                    ctx.host.groups[gid][1].add(user)
        if not found:
            ctx.e(f"usermod: group '{grp}' does not exist\n")
            return 6
        fs.write_file("/etc/group", "\n".join(lines), "/", 0o644, 0, 0)
        return 0
    return 0
    yield


@util("userdel")
def u_userdel(ctx):
    return 0
    yield


@util("apk")
def u_apk(ctx):
    """Alpine's package manager (apk add / apk update)."""
    from ..exec import stub_bytes
    a = ctx.argv[1:]
    sub = a[0] if a else ""
    pkgs = [x for x in a[1:] if not x.startswith("-")]
    if sub == "update":
        ctx.w("fetch https://dl-cdn.alpinelinux.org/alpine/v3.20/main/x86_64/APKINDEX.tar.gz\nfetch https://dl-cdn."
              "alpinelinux.org/alpine/v3.20/community/x86_64/APKINDEX.tar.gz\nv3.20.3-161-g3d0d8b2e8c3 [https://dl-cdn."
              "alpinelinux.org/alpine/v3.20/main]\nOK: 24165 distinct packages available\n")
        return 0
    if sub == "add":
        if ctx.uid != 0:
            ctx.e("ERROR: Unable to lock database: Permission denied\nERROR: Failed to open apk database: Permission "
                  "denied\n")
            return 99
        known = {"bash", "curl", "git", "jq", "vim", "less", "python3", "py3-pip", "tree", "nano", "wget", "ca-certificates",
                 "tini", "postgresql-client", "htop"}
        for i, p in enumerate(pkgs):
            if p not in known:
                ctx.e(f"ERROR: unable to select packages:\n  {p} (no such package):\n    required by: world[{p}]\n")
                return 1
        for i, p in enumerate(pkgs):
            ctx.w(f"({i + 1}/{len(pkgs)}) Installing {p} ({'8.10.1-r0' if p == 'curl' else '1.0-r0'})\n")
            progs = {"python3": ["python3.12"], "py3-pip": ["pip"], "postgresql-client": ["psql", "pg_isready"]}.get(p, [p])
            for prog in progs:
                n = ctx.fs.write_file(f"/usr/bin/{prog}", stub_bytes(prog), "/", 0o755, 0, 0)
                n.mode = 0o755
            if p == "python3":
                ctx.fs.symlink("python3.12", "/usr/bin/python3") if not ctx.fs.exists("/usr/bin/python3") else None
        ctx.w(f"OK: {12 + 3 * len(pkgs)} MiB in {20 + len(pkgs)} packages\n")
        return 0
    ctx.w("apk-tools 2.14.4, compiled for x86_64.\n\nusage: apk [<OPTIONS>...] COMMAND [<ARGUMENTS>...]\n")
    return 0
    yield


# ---------------------------------------------------------------------------- sg / newgrp (a new group without logging out)

def _member(ctx, group):
    try:
        for ln in bytes(ctx.fs.lookup("/etc/group").data).decode().split("\n"):
            p = ln.split(":")
            if p[0] == group and len(p) >= 4:
                user = ctx.host.user_name(ctx.uid)
                return int(p[2]), user in p[3].split(",") or ctx.uid == 0
    except FSError:
        pass
    return None, False


@util("sg")
def u_sg(ctx):
    from ..exec import exec_program
    a = ctx.argv[1:]
    if not a:
        ctx.e("Usage: sg group [[-c] command]\n")
        return 1
    gid, ok = _member(ctx, a[0])
    if gid is None:
        ctx.e(f"sg: group '{a[0]}' does not exist\n")
        return 1
    if not ok:
        ctx.e("Password: \nsg: failed to crypt password with previous salt: Invalid argument\n")
        return 1
    cmd = a[2] if len(a) > 2 and a[1] == "-c" else " ".join(a[1:])
    ctx.proc.gids = set(ctx.proc.gids) | {gid}
    ctx.proc.gid = gid
    return (yield from exec_program(ctx.proc, "/bin/sh", ["sh", "-c", cmd], dict(ctx.env), None))


@util("newgrp")
def u_newgrp(ctx):
    from ..exec import exec_program
    a = ctx.argv[1:]
    group = a[0] if a else "ada"
    gid, ok = _member(ctx, group)
    if gid is None:
        ctx.e(f"newgrp: group '{group}' does not exist\n")
        return 1
    if not ok:
        ctx.e("Password: \nnewgrp: failed to crypt password with previous salt: Invalid argument\n")
        return 1
    ctx.proc.gids = set(ctx.proc.gids) | {gid}
    ctx.proc.gid = gid
    return (yield from exec_program(ctx.proc, "/bin/bash", ["bash"], dict(ctx.env), None))


@util("nvidia-ctk")
def u_nvidia_ctk(ctx):
    """nvidia-ctk runtime configure --runtime=docker: register the nvidia runtime in /etc/docker/daemon.json."""
    a = ctx.argv[1:]
    if a[:2] != ["runtime", "configure"]:
        ctx.w("NAME:\n   NVIDIA Container Toolkit CLI\n\nUSAGE:\n   nvidia-ctk [global options] command\n")
        return 0
    if ctx.uid != 0:
        ctx.e("ERRO[0000] unable to flush config: unable to open /etc/docker/daemon.json for writing: open "
              "/etc/docker/daemon.json: permission denied\n")
        return 1
    ctx.fs.write_file("/etc/docker/daemon.json", '{\n    "runtimes": {\n        "nvidia": {\n            "args": [],\n'
                      '            "path": "nvidia-container-runtime"\n        }\n    }\n}\n', "/", 0o644, 0, 0)
    ctx.world.local.extra["nvidia_configured"] = True
    ctx.e("INFO[0000] Config file does not exist; using empty config\nINFO[0000] Wrote updated config to "
          "/etc/docker/daemon.json\nINFO[0000] It is recommended that docker daemon be restarted.\n")
    return 0
    yield


@util("pwd")
def u_pwd(ctx):
    """/usr/bin/pwd (coreutils): what a container runs for `docker run IMAGE pwd`."""
    ctx.w(ctx.cwd + "\n")
    return 0
    yield
