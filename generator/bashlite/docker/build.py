"""docker build: the Dockerfile parser and a BuildKit-like builder.

Every instruction runs for real on a copy of the stage's filesystem: RUN starts `/bin/sh -c ...` (or the exec form)
in a build container, so pip, apt-get, useradd... do what they do in the terminal; COPY/ADD take files from the
build context (minus .dockerignore). Each step has a cache key (its parent's key, the instruction, the content of
the files it copies, the build args it sees), so an unchanged prefix of the Dockerfile is CACHED. Layers record
the bytes they add; installed packages count with their real installed sizes, apt lists and pip's download cache
too when they are left behind in a layer. Multi-stage builds, --target, --build-arg, --secret, RUN --mount=type=
cache|secret|bind, --no-cache, --platform and buildx multi-platform pushes are supported."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import posixpath
import re
import shlex

from ..kernel import NS
from ..vfs import FSError, LazyData
from .cli import cmd, flags, host_path
from .engine import DockerError, Image, daemon_of, human_size
from .fsutil import clone_fs, diff_bytes

INSTRUCTIONS = {"FROM", "RUN", "CMD", "LABEL", "MAINTAINER", "EXPOSE", "ENV", "ADD", "COPY", "ENTRYPOINT", "VOLUME",
                "USER", "WORKDIR", "ARG", "ONBUILD", "STOPSIGNAL", "HEALTHCHECK", "SHELL"}
APT_MB = {"git": 18.4, "curl": 1.6, "wget": 3.7, "build-essential": 233.0, "gcc": 68.0, "vim": 38.0, "less": 0.5,
          "procps": 2.1, "iputils-ping": 0.2, "ca-certificates": 0.9, "jq": 1.1, "unzip": 0.4, "zip": 0.7,
          "htop": 0.4, "tree": 0.1, "nano": 0.9, "libgl1": 48.0, "ffmpeg": 290.0, "python3": 32.0,
          "python3-pip": 42.0, "postgresql-client": 14.0, "nginx": 3.2, "tini": 0.9, "git-lfs": 11.0}


class Instr:
    def __init__(self, cmd_, args, raw, line, flags_=None, json_=None, heredocs=None):
        self.cmd, self.args, self.raw, self.line = cmd_, args, raw, line
        self.flags = flags_ or {}
        self.json = json_
        self.heredocs = heredocs or []


def parse_dockerfile(text: str, name="Dockerfile"):
    """-> [Instr]. Comments only start at the beginning of a line; a trailing \\ joins the next line."""
    lines = text.split("\n")
    out = []
    i = 0
    escape = "\\"
    # parser directives (# syntax=..., # escape=...) at the very top
    while i < len(lines):
        m = re.match(r"^#\s*(\w+)\s*=\s*(\S+)\s*$", lines[i])
        if not m:
            break
        if m.group(1).lower() == "escape":
            escape = m.group(2)
        i += 1
    while i < len(lines):
        start = i
        line = lines[i]
        i += 1
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = [line]
        while parts[-1].rstrip().endswith(escape) and i < len(lines):
            parts[-1] = parts[-1].rstrip()[:-1]
            nxt = lines[i]
            i += 1
            if nxt.lstrip().startswith("#"):
                parts.append("")
                continue
            parts.append(nxt)
        joined = " ".join(p.strip() if k else p.strip() for k, p in enumerate(parts) if p.strip() or k == 0)
        m = re.match(r"^\s*(\S+)\s*(.*)$", joined)
        word, rest = m.group(1), m.group(2)
        cmd_ = word.upper()
        if cmd_ not in INSTRUCTIONS:
            raise DockerError(f"ERROR: failed to solve: dockerfile parse error on line {start + 1}: unknown "
                              f"instruction: {word}" + (" (did you mean " + _suggest(cmd_) + "?)" if _suggest(cmd_)
                                                         else ""), 1)
        fl = {}
        while rest.startswith("--") and cmd_ in ("RUN", "COPY", "ADD", "FROM", "HEALTHCHECK"):
            m2 = re.match(r"^--([\w-]+)(?:=(\S+))?\s*(.*)$", rest)
            if not m2:
                break
            key, val, rest = m2.group(1), m2.group(2), m2.group(3)
            fl.setdefault(key, []).append(val if val is not None else "true")
        heredocs = []
        hm = re.search(r"<<-?\s*(['\"]?)(\w+)\1", rest)
        if hm and cmd_ in ("RUN", "COPY"):
            body = []
            while i < len(lines) and lines[i].strip() != hm.group(2):
                body.append(lines[i])
                i += 1
            i += 1
            heredocs.append((hm.group(2), "\n".join(body) + "\n"))
        js = None
        if rest.startswith("[") and cmd_ in ("RUN", "CMD", "ENTRYPOINT", "SHELL", "COPY", "ADD", "VOLUME",
                                             "HEALTHCHECK"):
            try:
                js = json.loads(rest)
                if not isinstance(js, list) or not all(isinstance(x, str) for x in js):
                    js = None
            except ValueError:
                js = None
        out.append(Instr(cmd_, rest, joined, start + 1, fl, js, heredocs))
    return out


def _suggest(word):
    import difflib
    m = difflib.get_close_matches(word, sorted(INSTRUCTIONS), n=1, cutoff=0.6)
    return m[0].lower() if m else ""


def substitute(s, env):
    def repl(m):
        name = m.group(1) or m.group(2)
        op, word = m.group(3), m.group(4)
        val = env.get(name)
        if op == ":-":
            return val if val else (word or "")
        if op == ":+":
            return (word or "") if val else ""
        return val or ""
    return re.sub(r"\$(?:(\w+)|\{(\w+)(?:(:[-+])([^}]*))?\})", repl, s)


def env_pairs(args):
    """ENV/LABEL/ARG: k=v pairs (quotes allowed), or the old `ENV key value` form."""
    try:
        toks = shlex.split(args, posix=True)
    except ValueError:
        toks = args.split()
    if toks and "=" not in toks[0]:
        return [(toks[0], args.split(None, 1)[1] if len(args.split(None, 1)) > 1 else "")]
    out = []
    for t in toks:
        if "=" not in t:
            raise DockerError(f"Syntax error - can't find = in \"{t}\". Must be of the form: name=value", 1)
        k, _, v = t.partition("=")
        out.append((k, v))
    return out


def read_ignore(fs, ctxdir):
    try:
        text = bytes(fs.lookup(ctxdir + "/.dockerignore").data).decode("utf-8", "replace")
    except FSError:
        return []
    out = []
    for line in text.split("\n"):
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def ignored(rel, pats):
    """.dockerignore matching (Go filepath.Match plus **), last match wins, ! re-includes."""
    hit = False
    for p in pats:
        neg = p.startswith("!")
        pat = p[1:] if neg else p
        pat = pat.lstrip("/").rstrip("/")
        if pat.startswith("./"):
            pat = pat[2:]
        rx = "^" + re.escape(pat).replace(r"\*\*/", "(.*/)?").replace(r"\*\*", ".*").replace(r"\*", "[^/]*") \
            .replace(r"\?", "[^/]") + "(/.*)?$"
        if re.match(rx, rel):
            hit = not neg
    return hit


def context_files(fs, ctxdir, pats):
    """{relative path: inode} of the build context after .dockerignore (files and empty dirs)."""
    out = {}

    def walk(node, rel):
        for name in sorted(node.entries):
            child = node.entries[name]
            r = f"{rel}/{name}" if rel else name
            if ignored(r, pats) and not any(p.startswith("!") and p[1:].lstrip("/").startswith(r + "/")
                                            for p in pats):
                continue
            if child.type == "d":
                out[r + "/"] = child
                walk(child, r)
            else:
                out[r] = child
    walk(fs.lookup(ctxdir), "")
    return out


def content_hash(files: dict, prefix=""):
    h = hashlib.sha256()
    for rel in sorted(files):
        node = files[rel]
        h.update(rel.encode())
        if node.type == "f":
            d = node.data
            if isinstance(d, LazyData) and not d.materialized and d._src is None:
                h.update(f"lazy:{d._seed}:{len(d)}".encode())
            else:
                h.update(bytes(d))
            h.update(str(node.mode & 0o111).encode())
    return h.hexdigest()


class Stage:
    def __init__(self, name, index, base_ref):
        self.name, self.index, self.base_ref = name, index, base_ref
        self.fs = None
        self.config = None
        self.history = []
        self.size = 0
        self.key = ""
        self.args = {}
        self.base = None
        self.cuda = None
        self.arch = "amd64"
        self.steps = []


@cmd("build")
def docker_build(ctx, o, args):
    opts, rest = flags(args, {"-t|--tag": "m", "-f|--file": "v", "--build-arg": "m", "--target": "v",
                              "--no-cache": None, "--progress": "v", "--platform": "v", "-q|--quiet": None,
                              "--pull": None, "--secret": "m", "--push": None, "--load": None, "--label": "m",
                              "--network": "v", "--builder": "v", "--cache-from": "m", "--cache-to": "m",
                              "-o|--output": "m", "--ssh": "m"}, stop_at_positional=False)
    d = daemon_of(ctx.world, ctx.host)
    if len(rest) != 1:
        raise DockerError("ERROR: docker: 'docker buildx build' requires 1 argument\n\nUsage:  docker buildx build "
                          "[OPTIONS] PATH | URL | -\n\nRun 'docker buildx build --help' for more information", 1)
    ctxdir = host_path(ctx, rest[0])
    fs = ctx.fs
    try:
        if fs.lookup(ctxdir).type != "d":
            raise FSError("ENOTDIR")
    except FSError:
        raise DockerError(f"ERROR: unable to prepare context: path \"{rest[0]}\" not found", 1)
    dfile = host_path(ctx, opts["file"]) if opts.get("file") else ctxdir + "/Dockerfile"
    try:
        text = bytes(fs.lookup(dfile).data).decode("utf-8", "replace")
    except FSError:
        raise DockerError("ERROR: failed to solve: failed to read dockerfile: open Dockerfile: no such file or "
                          "directory", 1)
    platforms = (opts.get("platform") or "linux/amd64").split(",")
    if len(platforms) > 1 and getattr(d, "buildx_driver", "docker") == "docker":
        raise DockerError("ERROR: Multi-platform build is not supported for the docker driver.\nSwitch to a different "
                          "driver, or turn on the containerd image store, and try again.\nLearn more at "
                          "https://docs.docker.com/go/build-multi-platform/", 1)
    b = Builder(ctx, o, d, ctxdir, dfile, text, opts, platforms)
    img = yield from b.run()
    return 0 if img is not None else 1


class Builder:
    def __init__(self, ctx, o, d, ctxdir, dfile, text, opts, platforms):
        self.ctx, self.o, self.d = ctx, o, d
        self.ctxdir, self.dfile, self.text, self.opts = ctxdir, dfile, text, opts
        self.platforms = platforms
        self.plain = opts.get("progress") == "plain" or not ctx.stdin_is_tty() and opts.get("progress") != "tty"
        self.quiet = bool(opts.get("quiet"))
        self.lines = []          # the summary lines (tty style)
        self.n = 0               # #N counter (plain style)
        self.elapsed = 0.0
        self.build_args = dict(a.partition("=")[::2] for a in opts.get("build-arg", []))
        for k in list(self.build_args):
            if "=" not in [x for x in opts.get("build-arg", []) if x.startswith(k)][0]:
                self.build_args[k] = ctx.env.get(k, "")
        self.used_args = set()
        self.secrets = {}
        for s in opts.get("secret", []):
            kv = dict(x.partition("=")[::2] for x in s.split(","))
            sid = kv.get("id")
            if "env" in kv:
                self.secrets[sid] = ctx.env.get(kv["env"], "").encode()
            elif "src" in kv or "source" in kv:
                try:
                    self.secrets[sid] = bytes(ctx.fs.lookup(host_path(ctx, kv.get("src") or kv["source"])).data)
                except FSError:
                    raise DockerError(f"ERROR: failed to stat {kv.get('src') or kv['source']}: no such file", 1)
            else:
                self.secrets[sid] = ctx.env.get(sid.upper(), ctx.env.get(sid, "")).encode()
        self.log = []
        self.steps = []
        d.last_build = {"steps": self.steps, "tags": list(opts.get("tag", [])), "ok": None}

    # ----- output helpers
    def say(self, text, secs=0.0, cached=False, plain_body=None):
        if text.startswith("["):
            self.steps.append({"step": text, "cached": cached, "secs": round(secs, 1)})
        self.n += 1
        self.elapsed += secs
        if self.quiet:
            return
        if self.plain:
            self.o.err(f"#{self.n} {text}\n")
            for ln in (plain_body or []):
                self.o.err(f"#{self.n} {ln}\n")
            self.o.err(f"#{self.n} {'CACHED' if cached else f'DONE {secs:.1f}s'}\n\n")
        else:
            label = f" => {'CACHED ' if cached else ''}{text}"
            self.lines.append(f"{label:<74}{secs:>5.1f}s" if not cached else f"{label:<74}  0.0s")

    def sub(self, text, secs=0.0):
        if not self.quiet and not self.plain:
            label = f" => => {text}"
            self.lines.append(f"{label:<74}{secs:>5.1f}s")

    def flush_summary(self, ok=True, total=None):
        if self.quiet or self.plain:
            return
        steps = self.n
        head = f"[+] Building {self.elapsed:.1f}s ({steps}/{steps}) {'FINISHED' if ok else 'FAILED'}"
        self.o.err(f"{head:<65}docker:default\n" + "\n".join(self.lines) + "\n")

    def fail(self, step_label, instr, output, msg):
        lines = self.text.split("\n")
        ctxlines = []
        for k in range(max(1, instr.line - 2), min(len(lines), instr.line + 2) + 1):
            mark = ">>>" if k == instr.line else "   "
            ctxlines.append(f" {k:>3} | {mark} {lines[k - 1]}")
        if not self.quiet:
            if not self.plain:
                self.lines.append(f" => ERROR {step_label:<66}{0.4:>5.1f}s")
                self.flush_summary(False)
            body = "".join(f"{(i + 1) * 0.317:.3f} {ln}\n" for i, ln in enumerate(output.rstrip("\n").split("\n"))
                           if output.strip())
            self.o.err(f"------\n > {step_label}:\n{body}------\n{posixpath.basename(self.dfile)}:{instr.line}\n"
                       f"--------------------\n" + "\n".join(ctxlines) + "\n--------------------\n")
        self.d.last_build["ok"] = False
        self.steps.append({"step": "ERROR " + step_label, "cached": False, "secs": 0.0})
        raise DockerError(f"ERROR: failed to solve: {msg}", 1)

    # ----- the build
    def run(self):
        ctx, d = self.ctx, self.d
        instrs = parse_dockerfile(self.text, self.dfile)
        if not instrs or instrs[0].cmd not in ("FROM", "ARG"):
            raise DockerError("ERROR: failed to solve: no build stage in current context" if not instrs else
                              f"ERROR: failed to solve: dockerfile parse error on line {instrs[0].line}: "
                              f"{instrs[0].cmd.lower()} must be preceded by a FROM", 1)
        pats = read_ignore(ctx.fs, self.ctxdir)
        files = context_files(ctx.fs, self.ctxdir, pats)
        ctx_bytes = sum(len(n.data) for n in files.values() if n.type == "f")
        self.context = files
        # global ARGs (before the first FROM)
        globals_ = {}
        k = 0
        while k < len(instrs) and instrs[k].cmd == "ARG":
            for name, val in _arg_decl(instrs[k].args):
                globals_[name] = self.build_args.get(name, val)
                if name in self.build_args:
                    self.used_args.add(name)
            k += 1
        stages = []
        cur = None
        for ins in instrs[k:]:
            if ins.cmd == "FROM":
                toks = substitute(ins.args, globals_).split()
                if len(toks) not in (1, 3) or (len(toks) == 3 and toks[1].upper() != "AS"):
                    self.fail(f"[internal] load build definition from {posixpath.basename(self.dfile)}", ins, "",
                              f"dockerfile parse error on line {ins.line}: FROM requires either one or three "
                              f"arguments")
                cur = Stage(toks[2].lower() if len(toks) == 3 else None, len(stages), toks[0])
                cur.args = dict(globals_)
                stages.append(cur)
            else:
                cur.steps.append(ins)
        target = self.opts.get("target")
        if target and not any(s.name == target for s in stages):
            raise DockerError(f"ERROR: failed to solve: target stage \"{target}\" could not be found", 1)
        # which stages are needed
        last = next(i for i, s in enumerate(stages) if s.name == target) if target else len(stages) - 1
        need = self._needed(stages, last)
        # the build context and the base images
        self.say(f"[internal] load build definition from {posixpath.basename(self.dfile)}", 0.0)
        self.sub(f"transferring dockerfile: {len(self.text.encode())}B", 0.0)
        bases = []
        for s in stages:
            if s.index in need and s.base_ref.lower() not in [x.name for x in stages if x.name] and \
                    s.base_ref != "scratch" and s.base_ref not in bases:
                bases.append(s.base_ref)
        for ref in bases:
            secs = 0.4 + (0.0 if d.find_image(ref) else 0.8)
            self.say(f"[internal] load metadata for {_full_ref(ref)}", secs)
        self.say("[internal] load .dockerignore", 0.0)
        self.sub(f"transferring context: {len(chr(10).join(pats).encode()) + 2 if pats else 2}B", 0.0)
        ctx_secs = ctx_bytes / 150e6
        yield ("sleep", ctx.now + int((0.3 + sum(0.4 for _ in bases) + ctx_secs) * NS))
        self.multi = len(need) > 1
        for s in stages:
            if s.index not in need:
                continue
            self.total = len([x for x in s.steps if x.cmd in ("RUN", "COPY", "ADD", "WORKDIR")]) + 1
            self.stepno = 0
            ok = yield from self.stage(s, stages, ctx_bytes)
            if not ok:
                return None
            if s.index == last:
                break
        final = stages[last]
        img = self.export(final)
        return img

    def _needed(self, stages, last):
        names = {s.name: s.index for s in stages if s.name}
        need = {last}
        todo = [last]
        while todo:
            i = todo.pop()
            s = stages[i]
            if s.base_ref.lower() in names and names[s.base_ref.lower()] not in need:
                need.add(names[s.base_ref.lower()])
                todo.append(names[s.base_ref.lower()])
            for ins in s.steps:
                for v in ins.flags.get("from", []):
                    j = names.get(v.lower(), int(v) if v.isdigit() else None)
                    if j is not None and j not in need:
                        need.add(j)
                        todo.append(j)
                for m in ins.flags.get("mount", []):
                    mm = re.search(r"from=([\w.-]+)", m)
                    if mm and mm.group(1).lower() in names and names[mm.group(1).lower()] not in need:
                        need.add(names[mm.group(1).lower()])
                        todo.append(names[mm.group(1).lower()])
        return need

    def label(self, s, ins_text):
        self.stepno += 1
        tag = f"{s.name} " if s.name and self.multi else ""
        return f"[{tag}{self.stepno}/{self.total}] {ins_text}"

    def stage(self, s, stages, ctx_bytes):
        ctx, d = self.ctx, self.d
        names = {x.name: x for x in stages if x.name}
        # FROM
        if s.base_ref == "scratch":
            from ..vfs import FS
            s.fs = FS(lambda: ctx.world.kernel.now)
            s.config = {"Env": [], "Cmd": None, "Entrypoint": None, "WorkingDir": "", "User": "", "Labels": {},
                        "ExposedPorts": {}, "Volumes": {}}
            s.key = "scratch"
            s.size = 0
        elif s.base_ref.lower() in names and names[s.base_ref.lower()].index < s.index:
            parent = names[s.base_ref.lower()]
            s.fs = clone_fs(parent.fs, lambda: ctx.world.kernel.now)
            s.config = json.loads(json.dumps(parent.config))
            s.history = list(parent.history)
            s.size, s.key, s.base, s.cuda = parent.size, parent.key, parent.base, parent.cuda
            self.label(s, f"FROM {parent.name}")
            self.stepno -= 1
        else:
            img = d.find_image(s.base_ref)
            if img is None or self.opts.get("pull"):
                from .cmd_images import pull_image
                from .cli import Out

                class Quiet(Out):
                    def out(self, x):
                        pass

                    def err(self, x):
                        pass
                try:
                    img = yield from pull_image(ctx, Quiet(ctx), d, s.base_ref, quiet=True)
                except DockerError:
                    self.flush_summary(False)
                    raise DockerError(f"ERROR: failed to solve: {s.base_ref}: failed to resolve source metadata for "
                                      f"{_full_ref(s.base_ref)}: pull access denied, repository does not exist or "
                                      f"may require authorization: server message: insufficient_scope: authorization "
                                      f"failed", 1)
            text = f"FROM {_full_ref(s.base_ref)}@{img.repo_digests[0].split('@')[1] if img.repo_digests else img.id}"
            lab = self.label(s, text)
            self.say(lab, 2.1 if not getattr(img, "_built_used", False) else 0.0)
            img._built_used = True
            s.fs = clone_fs(img.fs, lambda: ctx.world.kernel.now)
            s.config = json.loads(json.dumps(img.config))
            s.history = list(img.history)
            s.size = img.size
            s.key = img.id
            s.base = img.base
            s.cuda = img.cuda
        env = dict(e.split("=", 1) for e in s.config.get("Env") or [] if "=" in e)
        args = {}
        context_shown = False
        for ins in s.steps:
            subst_env = {**s.args, **args, **env}
            c = ins.cmd
            if c == "ARG":
                for name, val in _arg_decl(ins.args):
                    if name in self.build_args:
                        args[name] = self.build_args[name]
                        self.used_args.add(name)
                    elif name in s.args and val is None:
                        args[name] = s.args[name]
                    else:
                        args[name] = substitute(val or "", subst_env)
                s.key = _h(s.key, ins.raw, json.dumps(args, sort_keys=True))
                continue
            if c in ("ENV", "LABEL"):
                pairs = env_pairs(ins.args)
                for k_, v_ in pairs:
                    v_ = substitute(v_, subst_env)
                    if c == "ENV":
                        env[k_] = v_
                        s.config["Env"] = [e for e in s.config.get("Env") or [] if not e.startswith(k_ + "=")] + \
                            [f"{k_}={v_}"]
                    else:
                        s.config.setdefault("Labels", {})
                        s.config["Labels"] = dict(s.config.get("Labels") or {}, **{k_: v_})
                s.history.append((f"{c} " + " ".join(f"{k_}={substitute(v_, subst_env)}" for k_, v_ in pairs), 0, True,
                                  ctx.now))
                s.key = _h(s.key, ins.raw)
                continue
            if c == "MAINTAINER":
                s.history.append((f"MAINTAINER {ins.args}", 0, True, ctx.now))
                continue
            if c == "EXPOSE":
                for p in substitute(ins.args, subst_env).split():
                    if not re.fullmatch(r"\d+(-\d+)?(/(tcp|udp|sctp))?", p):
                        self.fail(self.label(s, ins.raw), ins, "",
                                  f"dockerfile parse error on line {ins.line}: invalid containerPort: {p}")
                    s.config.setdefault("ExposedPorts", {})
                    s.config["ExposedPorts"] = dict(s.config.get("ExposedPorts") or {},
                                                    **{p if "/" in p else p + "/tcp": {}})
                s.history.append((f"EXPOSE map[{ins.args}/tcp:{{}}]", 0, True, ctx.now))
                s.key = _h(s.key, ins.raw)
                continue
            if c == "USER":
                s.config["User"] = substitute(ins.args.strip(), subst_env)
                s.history.append((f"USER {s.config['User']}", 0, True, ctx.now))
                s.key = _h(s.key, ins.raw)
                continue
            if c in ("CMD", "ENTRYPOINT"):
                val = ins.json if ins.json is not None else ["/bin/sh", "-c", ins.args]
                s.config["Cmd" if c == "CMD" else "Entrypoint"] = val
                if c == "ENTRYPOINT" and "Cmd" in s.config and not getattr(s, "_cmd_set", False):
                    s.config["Cmd"] = None
                if c == "CMD":
                    s._cmd_set = True
                s.history.append((f"{c} {json.dumps(val)}", 0, True, ctx.now))
                s.key = _h(s.key, ins.raw)
                continue
            if c == "VOLUME":
                vols = ins.json if ins.json is not None else ins.args.split()
                s.config["Volumes"] = dict(s.config.get("Volumes") or {}, **{v: {} for v in vols})
                s.history.append((f"VOLUME {json.dumps(vols)}", 0, True, ctx.now))
                continue
            if c == "STOPSIGNAL":
                s.config["StopSignal"] = ins.args.strip()
                continue
            if c == "SHELL":
                s.config["Shell"] = ins.json
                continue
            if c == "ONBUILD":
                s.config["OnBuild"] = (s.config.get("OnBuild") or []) + [ins.args]
                continue
            if c == "HEALTHCHECK":
                rest = ins.args
                if rest.strip().upper() == "NONE":
                    s.config["Healthcheck"] = {"Test": ["NONE"]}
                else:
                    m = re.match(r"^(?:CMD|cmd)\s+(.*)$", rest.strip())
                    if not m:
                        self.fail(self.label(s, ins.raw), ins, "", f"dockerfile parse error on line {ins.line}: "
                                                                   f"Unknown type \"{rest.split()[0]}\" in HEALTHCHECK")
                    body = m.group(1)
                    try:
                        js = json.loads(body) if body.startswith("[") else None
                    except ValueError:
                        js = None
                    test = ["CMD"] + js if js else ["CMD-SHELL", body]
                    hc = {"Test": test}
                    for k_, v_ in ins.flags.items():
                        from .cli import _dur
                        if k_ in ("interval", "timeout", "start-period"):
                            hc[{"interval": "Interval", "timeout": "Timeout", "start-period": "StartPeriod"}[k_]] = \
                                int(_dur(v_[0]) * NS)
                        if k_ == "retries":
                            hc["Retries"] = int(v_[0])
                    s.config["Healthcheck"] = hc
                s.history.append((f"HEALTHCHECK &{{{json.dumps(s.config['Healthcheck'].get('Test'))}}}", 0, True,
                                  ctx.now))
                s.key = _h(s.key, ins.raw)
                continue
            if c == "WORKDIR":
                wd = substitute(ins.args.strip(), subst_env)
                if not wd.startswith("/"):
                    wd = posixpath.join(s.config.get("WorkingDir") or "/", wd)
                s.config["WorkingDir"] = wd
                lab = self.label(s, f"WORKDIR {wd}")
                key = _h(s.key, ins.raw)
                if key in self.d.build_cache and not self.opts.get("no-cache"):
                    self.say(lab, cached=True)
                else:
                    self.say(lab, 0.0)
                s.fs.mkdir(wd, "/", 0o755, 0, 0, parents=True)
                s.history.append((f"WORKDIR {wd}", 0, False, ctx.now))
                self.d.build_cache[key] = (None, 0, None)
                s.key = key
                continue
            if c in ("COPY", "ADD"):
                if not context_shown and "from" not in ins.flags:
                    self.say("[internal] load build context", 0.1)
                    self.sub(f"transferring context: {human_size(ctx_bytes) if ctx_bytes else '2B'}", 0.1)
                    context_shown = True
                ok = yield from self.copy(s, ins, stages, subst_env)
                if not ok:
                    return False
                continue
            if c == "RUN":
                ok = yield from self.run_step(s, ins, env, args)
                if not ok:
                    return False
                continue
        s.config["Env"] = [f"{k_}={v_}" for k_, v_ in env.items()] if env else s.config.get("Env")
        return True

    def copy(self, s, ins, stages, subst_env):
        ctx = self.ctx
        toks = ins.json if ins.json is not None else shlex.split(substitute(ins.args, subst_env))
        if ins.heredocs:
            toks = [t for t in toks if not t.startswith("<<")]
        if len(toks) < 2 and not ins.heredocs:
            self.fail(self.label(s, ins.raw), ins, "", f"dockerfile parse error on line {ins.line}: {ins.cmd} requires "
                                                       f"at least two arguments, but only one was provided. "
                                                       f"Destination could not be determined")
        srcs, dest = toks[:-1], toks[-1]
        wd = s.config.get("WorkingDir") or "/"
        if not dest.startswith("/"):
            dest = posixpath.join(wd, dest) + ("/" if dest.endswith("/") or dest in (".", "./") else "")
        frm = ins.flags.get("from", [None])[0]
        if frm:
            names = {x.name: x for x in stages if x.name}
            src_stage = names.get(frm.lower()) or (stages[int(frm)] if frm.isdigit() and int(frm) < len(stages)
                                                   else None)
            if src_stage is not None:
                src_fs = src_stage.fs
            else:
                img = self.d.find_image(frm)
                if img is None:
                    from .cmd_images import pull_image
                    from .cli import Out

                    class Q(Out):
                        def out(self, x):
                            pass

                        def err(self, x):
                            pass
                    img = yield from pull_image(ctx, Q(ctx), self.d, frm, quiet=True)
                src_fs = img.fs
            picked = {}
            for src in srcs:
                p = src if src.startswith("/") else "/" + src
                try:
                    node = src_fs.lookup(p)
                except FSError:
                    self.fail(self.label(s, ins.raw), ins, "", f"failed to compute cache key: failed to calculate "
                                                               f"checksum of ref: \"{p}\": not found")
                picked[p] = node
            key = _h(s.key, ins.raw, content_hash({k: v for k, v in picked.items()}))
        else:
            picked = {}
            for src in srcs:
                if src.startswith(("http://", "https://")) and ins.cmd == "ADD":
                    picked[src] = None
                    continue
                pat = posixpath.normpath(src.lstrip("/")) if src not in (".", "./") else ""
                hits = {r: n for r, n in self.context.items() if
                        (pat == "" and not r.endswith("/")) or fnmatch.fnmatchcase(r.rstrip("/"), pat) or
                        (r.startswith(pat + "/") and pat)}
                if not hits and pat:
                    self.fail(self.label(s, ins.raw), ins, "", f"failed to compute cache key: failed to calculate "
                                                               f"checksum of ref {self.d._id('ref')[:25]}::"
                                                               f"{self.d._id('x')[:25]}: \"/{pat}\": not found")
                picked[src] = hits
            for name, body in ins.heredocs:
                picked[f"<<{name}"] = body
            key = _h(s.key, ins.raw, json.dumps(ins.flags, sort_keys=True),
                     content_hash({f"{k}:{r}": n for k, v in picked.items() if isinstance(v, dict)
                                   for r, n in v.items() if n.type == "f"}))
        lab = self.label(s, f"{ins.cmd} " + " ".join(f"--{k}={v[0]}" for k, v in ins.flags.items()) +
                         (" " if ins.flags else "") + " ".join(toks))
        lab = re.sub(r"\s+", " ", lab)
        cached = self.d.build_cache.get(key)
        if cached is not None and cached[0] is not None and not self.opts.get("no-cache"):
            self.say(lab, cached=True)
            s.fs = clone_fs(cached[0], lambda: ctx.world.kernel.now)
            s.history.append(cached[2])
            s.size += cached[1]
            s.key = key
            return True
        before = clone_fs(s.fs, lambda: ctx.world.kernel.now)
        chown = ins.flags.get("chown", [None])[0]
        uid = gid = 0
        if chown:
            u, _, g = chown.partition(":")
            uid = _uid_in(s.fs, u)
            gid = _gid_in(s.fs, g or u) if (g or u) else uid
        chmod = ins.flags.get("chmod", [None])[0]
        multi = len(srcs) > 1 or any(isinstance(v, dict) and any(ch in src for ch in "*?[") and
                                     len([r for r in v if not r.endswith("/")]) > 1 for src, v in picked.items())
        if multi and not dest.endswith("/") and not _is_dir(s.fs, dest):
            self.fail(lab, ins, "", f"dockerfile parse error on line {ins.line}: When using {ins.cmd} with more than "
                                    f"one source file, the destination must be a directory and end with a /")
        try:
            for src, val in picked.items():
                if frm:
                    self._put(val, s.fs, dest if not dest.endswith("/") or val.type == "d" else
                              dest + posixpath.basename(src.rstrip("/")), uid, gid, chmod, frm=True,
                              src_fs=src_fs)
                    continue
                if isinstance(val, str):        # heredoc
                    target = dest if not dest.endswith("/") else dest + src[2:]
                    n = s.fs.write_file(target, val, "/", 0o644, uid, gid)
                    n.mode = 0o644
                    continue
                if val is None:                 # ADD <url>
                    target = dest if not dest.endswith("/") else dest + posixpath.basename(src)
                    s.fs.write_file(target, f"downloaded from {src}\n", "/", 0o600, uid, gid)
                    continue
                pat = posixpath.normpath(src.lstrip("/")) if src not in (".", "./") else ""
                for rel, node in sorted(val.items()):
                    if pat and rel.rstrip("/") == pat and node.type == "d":
                        continue
                    if pat and rel.startswith(pat + "/"):
                        sub = rel[len(pat) + 1:]
                    elif pat and rel.rstrip("/") == pat:
                        sub = posixpath.basename(rel) if (dest.endswith("/") or _is_dir(s.fs, dest)) else ""
                    elif not pat:
                        sub = rel
                    else:
                        sub = posixpath.basename(rel.rstrip("/"))
                    target = posixpath.join(dest, sub) if sub else dest.rstrip("/") or "/"
                    if rel.endswith("/"):
                        n = s.fs.mkdir(target, "/", 0o755, uid, gid, parents=True)
                        n.uid, n.gid = uid, gid
                        continue
                    self._put(node, s.fs, target, uid, gid, chmod)
        except FSError as e:
            self.fail(lab, ins, "", f"failed to copy files: {e.msg}")
        added = diff_bytes(before.root, s.fs.root)
        secs = 0.1 + added / 300e6
        self.say(lab, secs)
        yield ("sleep", ctx.now + int(secs * NS))
        hist = (f"{ins.cmd} " + " ".join(toks) + f" # buildkit", added, False, ctx.now)
        s.history.append(hist)
        s.size += added
        s.key = key
        self.d.build_cache[key] = (clone_fs(s.fs, lambda: ctx.world.kernel.now), added, hist)
        return True

    def _put(self, node, fs, target, uid, gid, chmod, frm=False, src_fs=None):
        from .fsutil import clone_node
        parent, _, name = target.rstrip("/").rpartition("/")
        pdir = fs.mkdir(parent or "/", "/", 0o755, 0, 0, parents=True)
        new = clone_node(node, fs, pdir)
        if not frm or uid or gid:
            _chown_tree(new, uid, gid)
        if chmod:
            new.mode = (new.mode & ~0o7777) | int(chmod, 8)
        old = pdir.entries.get(name)
        if old is not None and old.type == "d" and new.type == "d":
            for k, v in new.entries.items():
                old.entries[k] = v
                if v.type == "d":
                    v.parent = old
        else:
            pdir.entries[name] = new
            if new.type == "d":
                new.parent = pdir

    def run_step(self, s, ins, env, args):
        ctx, d = self.ctx, self.d
        shell = s.config.get("Shell") or ["/bin/sh", "-c"]
        body = ins.args
        if ins.heredocs:
            body = ins.heredocs[0][1]
            argv = list(shell) + [body]
            shown = f"RUN {ins.args}"
        elif ins.json is not None:
            argv = list(ins.json)
            shown = "RUN " + json.dumps(ins.json)
        else:
            argv = list(shell) + [body]
            shown = f"RUN {ins.args}"
        mounts = ins.flags.get("mount", [])
        lab = self.label(s, " ".join(["RUN"] + [f"--mount={m}" for m in mounts] + [shown[4:]]))
        used_args = {k: v for k, v in args.items() if re.search(rf"\$\{{?{k}\b", body)}
        key = _h(s.key, ins.raw, json.dumps(used_args, sort_keys=True), json.dumps(env, sort_keys=True))
        cached = d.build_cache.get(key)
        if cached is not None and cached[0] is not None and not self.opts.get("no-cache"):
            self.say(lab, cached=True)
            s.fs = clone_fs(cached[0], lambda: ctx.world.kernel.now)
            s.history.append(cached[2])
            s.size += cached[1]
            s.key = key
            return True
        before = clone_fs(s.fs, lambda: ctx.world.kernel.now)
        run_env = dict(env)
        run_env.update(args)
        run_env.setdefault("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")
        run_env.setdefault("HOME", "/root")
        status, out, secs = yield from run_in_fs(ctx, d, s, argv, run_env, mounts, self.secrets,
                                                 ins.flags.get("network", [None])[0])
        if status != 0:
            self.fail(lab, ins, out, f"process \"{' '.join(argv) if ins.json is None else json.dumps(argv)}\" did not "
                                     f"complete successfully: exit code: {status}")
        added = diff_bytes(before.root, s.fs.root) + virtual_bytes(before, s.fs, body, run_env, mounts)
        self.say(lab, secs, plain_body=[f"{(i + 1) * 0.211:.3f} {ln}" for i, ln in
                                        enumerate(out.rstrip("\n").split("\n")) if out.strip()])
        hist = (f"RUN {'|' + str(len(used_args)) + ' ' + ' '.join(f'{k}={v}' for k, v in used_args.items()) + ' ' if used_args else ''}"
                f"{'/bin/sh -c ' if ins.json is None else ''}{body if ins.json is None else json.dumps(ins.json)}"
                f" # buildkit", added, False, ctx.now)
        s.history.append(hist)
        s.size += added
        s.key = key
        d.build_cache[key] = (clone_fs(s.fs, lambda: ctx.world.kernel.now), added, hist)
        return True

    def export(self, s):
        ctx, d = self.ctx, self.d
        cfg = s.config
        for lab in self.opts.get("label", []):
            k, _, v = lab.partition("=")
            cfg["Labels"] = dict(cfg.get("Labels") or {}, **{k: v})
        iid = "sha256:" + hashlib.sha256(f"built:{s.key}:{json.dumps(cfg, sort_keys=True)}".encode()).hexdigest()
        existing = d.images.get(iid)
        img = existing or Image(iid, cfg, clone_fs(s.fs, lambda: ctx.world.kernel.now), s.size, s.history, ctx.now,
                                self.platforms[0].split("/")[-1] if self.platforms else "amd64", None)
        img.base = img.base or s.base
        img.cuda = s.cuda
        img.platforms = list(self.platforms)
        tags = [t for t in self.opts.get("tag", [])]
        from .cmd_images import normalize
        old_ids = {normalize(t): d.tags.get(normalize(t)) for t in tags}
        d.add_image(img, [normalize(t) for t in tags])
        for t, oid in old_ids.items():
            if oid and oid != img.id and oid in d.images and not d.images[oid].repo_tags:
                d.images[oid].intermediate = False      # now a dangling <none>:<none> image
        d.last_build["ok"] = True
        d.last_build["size"] = human_size(img.size)
        self.say("exporting to image", 0.3)
        self.sub("exporting layers", 0.2)
        self.sub(f"writing image {img.id}", 0.0)
        for t in tags:
            self.sub(f"naming to {_full_ref(normalize(t))}", 0.0)
        self.flush_summary(True)
        if self.quiet:
            ctx.w(img.id + "\n")
        d.event(f"image build {img.short}")
        if self.opts.get("push"):
            from .cmd_images import push_one
            for t in tags:
                yield_push = push_one(ctx, self.o, d, normalize(t), img, quiet=True)
                for _ in yield_push:
                    pass
            img.pushed_platforms = list(self.platforms)
        return img


def _arg_decl(text):
    out = []
    for tok in text.split():
        k, eq, v = tok.partition("=")
        out.append((k, v.strip('"\'') if eq else None))
    return out


def _full_ref(ref):
    from .cmd_images import normalize, registry_of
    n = normalize(ref)
    if registry_of(n) == "docker.io":
        return "docker.io/" + (n if "/" in n else "library/" + n)
    return n


def _h(*parts):
    return hashlib.sha256("\x00".join(str(p) for p in parts).encode()).hexdigest()


def _is_dir(fs, p):
    try:
        return fs.lookup(p).type == "d"
    except FSError:
        return False


def _chown_tree(n, uid, gid):
    n.uid, n.gid = uid, gid
    if n.type == "d":
        for c in n.entries.values():
            _chown_tree(c, uid, gid)


def _uid_in(fs, name):
    if name.isdigit():
        return int(name)
    try:
        for line in bytes(fs.lookup("/etc/passwd").data).decode().split("\n"):
            p = line.split(":")
            if p[0] == name and len(p) > 2:
                return int(p[2])
    except FSError:
        pass
    raise DockerError(f"ERROR: failed to solve: unable to convert uid/gid chown string to host mapping: can't find "
                      f"uid for user {name}: no such user: {name}", 1)


def _gid_in(fs, name):
    if name.isdigit():
        return int(name)
    try:
        for line in bytes(fs.lookup("/etc/group").data).decode().split("\n"):
            p = line.split(":")
            if p[0] == name and len(p) > 2:
                return int(p[2])
    except FSError:
        pass
    return _uid_in(fs, name)


def virtual_bytes(before, after, body, env, mounts):
    """Bytes a RUN adds that the simulation doesn't write out: installed packages' real sizes, apt lists left in the
    layer, pip's download cache left in the layer."""
    from ..utils.pyenv import WHEEL_MB
    total = 0.0
    site_dirs = ["/usr/local/lib/python3.12/site-packages", "/opt/conda/lib/python3.12/site-packages"]
    new_pkgs = []
    for sd in site_dirs:
        try:
            a = set(after.lookup(sd).entries)
        except FSError:
            continue
        try:
            b = set(before.lookup(sd).entries)
        except FSError:
            b = set()
        for name in a - b:
            m = re.match(r"(.+)-[^-]+\.dist-info$", name)
            if m:
                new_pkgs.append(m.group(1).replace("_", "-").lower())
    cpu = "download.pytorch.org/whl/cpu" in body
    for p in new_pkgs:
        mb = WHEEL_MB.get(p, 0.35) * 2.4
        if p == "torch":
            mb = 185.0 * 2.4 if cpu else 4300.0
        total += mb * 1e6
    cache_mounted = any("type=cache" in m and ("pip" in m or "uv" in m) for m in mounts)
    if new_pkgs and not cache_mounted and "--no-cache-dir" not in body and not env.get("PIP_NO_CACHE_DIR") and \
            "uv " not in body:
        total += sum((WHEEL_MB.get(p, 0.35) if p != "torch" else (185.0 if cpu else 2400.0)) for p in new_pkgs) * 1e6
    try:
        a = set(after.lookup("/var/lib/bashlite/apt").entries)
        try:
            b = set(before.lookup("/var/lib/bashlite/apt").entries)
        except FSError:
            b = set()
        for p in a - b:
            total += APT_MB.get(p, 2.0) * 1e6
    except FSError:
        pass
    try:
        lists = after.lookup("/var/lib/apt/lists")
        had = len(before.lookup("/var/lib/apt/lists").entries)
        if lists.entries and not had:
            total += 47e6
    except FSError:
        pass
    return int(total)


def run_in_fs(ctx, d, s, argv, env, mounts, secrets, network=None):
    """Run a RUN step in a throwaway build container on the stage's filesystem (generator) -> (status, output,
    seconds)."""
    from ..exec import exec_program, find_in_path
    from ..kernel import OpenFile
    from ..world import Host, User
    from .fsutil import graft
    h = Host(ctx.world, "buildkitsandbox", "172.17.0.99", pid_base=0)
    h.fs = s.fs
    h.extra["offline"] = network == "none"
    h.extra["build"] = True
    try:
        text = bytes(s.fs.lookup("/etc/passwd").data).decode()
    except FSError:
        text = "root:x:0:0:root:/root:/bin/sh\n"
    for line in text.split("\n"):
        p = line.split(":")
        if len(p) >= 7 and p[2].isdigit():
            h.users[int(p[2])] = User(p[0], int(p[2]), int(p[3]), p[4], p[5], p[6])
            h.groups.setdefault(int(p[3]), (p[0], set()))
    if 0 not in h.users:
        h.users[0] = User("root", 0, 0, "root", "/root", "/bin/sh")
    user = s.config.get("User") or "root"
    name, _, grp = user.partition(":")
    uid = int(name) if name.isdigit() else next((u.uid for u in h.users.values() if u.name == name), None)
    if uid is None:
        return 1, f"unable to find user {name}: no matching entries in passwd file\n", 0.1
    gid = int(grp) if grp.isdigit() else (h.users[uid].gid if uid in h.users else 0)
    if uid in h.users:
        env["HOME"] = h.users[uid].home
    temp = []
    for m in mounts:
        kv = dict(x.partition("=")[::2] for x in m.split(","))
        t = kv.get("type", "bind")
        target = kv.get("target") or kv.get("dst") or kv.get("destination")
        if t == "cache":
            key = kv.get("id") or target
            node = d.cache_mounts.get(key)
            if node is None:
                node = d.host.fs._new("d", 0o755, 0, 0)
                node.parent = node
                d.cache_mounts[key] = node
            graft(s.fs, target, node)
            temp.append(target)
        elif t == "secret":
            sid = kv.get("id")
            tgt = target or f"/run/secrets/{sid}"
            if sid not in secrets:
                if kv.get("required") == "true":
                    return 1, f"secret {sid}: not found\n", 0.1
                continue
            n = s.fs.write_file(tgt, secrets[sid], "/", 0o400, 0, 0)
            n.mode = 0o400
            temp.append(tgt)
        elif t == "bind":
            pass
    out = []

    class Sink:
        def write(self, stream, data):
            out.append(bytes(data))
    sink = Sink()
    o1 = OpenFile("log", wr=True, stream="out")
    o1.tty = sink
    o2 = OpenFile("log", wr=True, stream="err")
    o2.tty = sink
    wd = s.config.get("WorkingDir") or "/"
    path = find_in_path(s.fs, argv[0], env.get("PATH"), wd)
    if path is None or not s.fs.exists(path):
        return 127, f"/bin/sh: 1: {argv[-1].split()[0] if argv else ''}: not found\n" if argv[0] != "/bin/sh" else \
            "runc run failed: unable to start container process: exec: \"/bin/sh\": stat /bin/sh: no such file or " \
            "directory\n", 0.1
    k = ctx.world.kernel
    p = k.spawn(h, argv, None, None, fds={0: OpenFile("null", rd=True), 1: o1, 2: o2}, cwd=wd, env=env, uid=uid,
                gid=gid)
    p.daemon = True
    t0 = k.now
    k.start(p, exec_program(p, path, argv, env, None))
    yield from ctx.wait([p])
    # what the RUN mounted is not part of the layer
    for tgt in temp:
        parent, _, nm = tgt.rstrip("/").rpartition("/")
        try:
            pnode = s.fs.lookup(parent or "/")
            pnode.entries.pop(nm, None)
        except FSError:
            pass
        binds = s.fs.__dict__.get("binds")
        if binds:
            for key_, val in list(binds.items()):
                if val[1] == tgt:
                    del binds[key_]
    for q in list(h.procs.values()):
        if q.state != "Z":
            k.kill_now(q, 9)
    secs = max(0.2, (k.now - t0) / NS)
    return p.status, b"".join(out).decode("utf-8", "replace"), secs


@cmd("buildx")
def docker_buildx(ctx, o, args):
    """docker buildx create --use (a docker-container builder: multi-platform builds) / ls / version."""
    d = daemon_of(ctx.world, ctx.host)
    sub = args[0] if args else ""
    if sub == "create":
        opts, rest = flags(args[1:], {"--use": None, "--name": "v", "--driver": "v", "--bootstrap": None},
                           stop_at_positional=False)
        name = opts.get("name") or "festive_kepler"
        d.buildx_driver = "docker-container"
        d.buildx_name = name
        ctx.w(name + "\n")
        return 0
    if sub == "use":
        d.buildx_driver = "docker-container" if args[1:] and args[1] != "default" else "docker"
        return 0
    if sub == "ls":
        rows = [["NAME/NODE", "DRIVER/ENDPOINT", "STATUS", "BUILDKIT", "PLATFORMS"],
                ["default" + ("" if getattr(d, "buildx_driver", "docker") != "docker" else "*"), "docker", "", "", ""],
                [" \\_ default", " \\_ default", "running", "v0.16.0", "linux/amd64, linux/amd64/v2, linux/386"]]
        if getattr(d, "buildx_driver", "docker") != "docker":
            rows += [[getattr(d, "buildx_name", "builder") + "*", "docker-container", "", "", ""],
                     [" \\_ " + getattr(d, "buildx_name", "builder") + "0", " \\_ unix:///var/run/docker.sock",
                      "running", "v0.17.1", "linux/amd64, linux/arm64, linux/arm/v7"]]
        from . import gotmpl
        ctx.w(gotmpl.tabulate(rows))
        return 0
    if sub == "version":
        ctx.w("github.com/docker/buildx v0.17.1 257815a\n")
        return 0
    ctx.w("\nUsage:  docker buildx [OPTIONS] COMMAND\n\nExtended build capabilities with BuildKit\n\nCommands:\n  "
          "build       Start a build\n  create      Create a new builder instance\n  ls          List builder "
          "instances\n  use         Set the current builder instance\n")
    return 0
    yield


__all__ = ["parse_dockerfile", "docker_build", "Builder"]
