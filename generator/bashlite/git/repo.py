"""A repository on the simulated filesystem: the .git directory (HEAD, refs, packed-refs, reflogs, config, the
binary index), the working tree, and .gitignore rules. Objects live in the world's table (objects.odb), but loose
objects are also written as real zlib files under .git/objects, so `ls .git/objects` looks the way it does."""

from __future__ import annotations

import fnmatch
import hashlib
import posixpath
import re
import struct

from ..vfs import FSError, LazyData
from .objects import EMPTY_TREE, GitError, Person, hash_object, loose_bytes, odb

ZERO = "0" * 40
SAMPLE_HOOKS = ("applypatch-msg", "commit-msg", "fsmonitor-watchman", "post-update", "pre-applypatch", "pre-commit",
                "pre-merge-commit", "pre-push", "pre-rebase", "pre-receive", "prepare-commit-msg", "push-to-checkout",
                "sendemail-validate", "update")


# ---------------------------------------------------------------------------- config

def _unquote(v: str) -> str:
    out = []
    q = False
    i = 0
    while i < len(v):
        c = v[i]
        if c == '"':
            q = not q
        elif c == "\\" and i + 1 < len(v):
            i += 1
            out.append({"n": "\n", "t": "\t", "b": "\b", "\\": "\\", '"': '"'}.get(v[i], v[i]))
        elif c in "#;" and not q:
            break
        else:
            out.append(c)
        i += 1
    return "".join(out).strip() if not any(ch == '"' for ch in v) else "".join(out).rstrip()


def _quote(v: str) -> str:
    s = v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\t", "\\t")
    if s != s.strip() or "#" in s or ";" in s:
        s = f'"{s}"'
    return s


def canon_key(key: str) -> str:
    """section.Sub.Key -> section.Sub.key (section and key are case-insensitive, a subsection isn't)."""
    parts = key.split(".")
    if len(parts) < 2:
        return key.lower()
    return ".".join([parts[0].lower()] + parts[1:-1] + [parts[-1].lower()])


def parse_config(text: str):
    """-> [(key, value, line number)]; value None for a bare `key` (true)."""
    out = []
    section = ""
    for n, raw in enumerate(text.split("\n"), 1):
        line = raw.strip()
        if not line or line[0] in "#;":
            continue
        if line.startswith("["):
            m = re.match(r'\[\s*([A-Za-z0-9.-]+)(?:\s+"((?:[^"\\]|\\.)*)")?\s*\]\s*(.*)$', line)
            if not m:
                continue
            sec = m.group(1).lower()
            if m.group(2) is not None:
                sec += "." + re.sub(r"\\(.)", r"\1", m.group(2))
            elif "." in sec:
                a, _, b = sec.partition(".")
                sec = a + "." + b
            section = sec
            line = m.group(3).strip()
            if not line:
                continue
        m = re.match(r"([A-Za-z][A-Za-z0-9-]*)\s*(=\s*(.*))?$", line)
        if not m or not section:
            continue
        key = section + "." + m.group(1).lower()
        out.append((key, None if m.group(2) is None else _unquote(m.group(3)), n))
    return out


class ConfigFile:
    def __init__(self, fs, path: str):
        self.fs, self.path = fs, path

    def text(self) -> str:
        try:
            return bytes(self.fs.lookup(self.path).data).decode("utf-8", "surrogateescape")
        except FSError:
            return ""

    def exists(self) -> bool:
        return self.fs.exists(self.path)

    def entries(self):
        return parse_config(self.text())

    def write(self, text: str, uid: int = 1000, gid: int = 1000):
        self.fs.write_file(self.path, text, "/", 0o644, uid, gid)

    @staticmethod
    def _section_header(sec: str) -> str:
        if "." in sec:
            a, _, b = sec.partition(".")
            b = b.replace("\\", "\\\\").replace('"', '\\"')
            return f'[{a} "{b}"]'
        return f"[{sec}]"

    def set(self, key: str, value: str, uid=1000, gid=1000, add=False, replace_all=False, value_re=None):
        key = canon_key(key)
        sec, _, name = key.rpartition(".")
        text = self.text()
        lines = text.split("\n") if text else []
        if lines and lines[-1] == "":
            lines.pop()
        hits = [n for k, v, n in parse_config(text) if canon_key(k) == key
                and (value_re is None or re.search(value_re, v or ""))]
        new_line = f"\t{name} = {_quote(value)}"
        if hits and not add:
            if replace_all:
                for n in reversed(hits[:-1]):
                    del lines[n - 1]
                last = hits[-1] - (len(hits) - 1)
                lines[last - 1] = new_line
            else:
                lines[hits[-1] - 1] = new_line
        else:
            # the end of the last section with this name, else a new section at the end
            sec_line = None
            current = None
            for i, raw in enumerate(lines):
                s = raw.strip()
                m = re.match(r'\[\s*([A-Za-z0-9.-]+)(?:\s+"((?:[^"\\]|\\.)*)")?\s*\]', s)
                if m:
                    cs = m.group(1).lower() + ("." + m.group(2) if m.group(2) is not None else "")
                    current = cs
                    if cs == sec:
                        sec_line = i
                elif current == sec and s:
                    sec_line = i
            if sec_line is None:
                lines.append(self._section_header(sec))
                lines.append(new_line)
            else:
                lines.insert(sec_line + 1, new_line)
        self.write("\n".join(lines) + "\n", uid, gid)

    def unset(self, key: str, uid=1000, gid=1000, all_=False, value_re=None) -> int:
        """-> how many lines were removed (git exits 5 when nothing matched)."""
        key = canon_key(key)
        text = self.text()
        lines = text.split("\n")
        hits = [n for k, v, n in parse_config(text) if canon_key(k) == key
                and (value_re is None or re.search(value_re, v or ""))]
        if not hits:
            return 0
        if len(hits) > 1 and not all_:
            return -1
        for n in reversed(hits):
            del lines[n - 1]
        self._drop_empty_sections(lines)
        self.write("\n".join(lines).rstrip("\n") + "\n" if any(x.strip() for x in lines) else "", uid, gid)
        return len(hits)

    @staticmethod
    def _drop_empty_sections(lines):
        i = 0
        while i < len(lines):
            if lines[i].strip().startswith("["):
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                if j >= len(lines) or lines[j].strip().startswith("["):
                    del lines[i]
                    continue
            i += 1

    def remove_section(self, sec: str, uid=1000, gid=1000) -> bool:
        lines = self.text().split("\n")
        out, skip, found = [], False, False
        for raw in lines:
            s = raw.strip()
            m = re.match(r'\[\s*([A-Za-z0-9.-]+)(?:\s+"((?:[^"\\]|\\.)*)")?\s*\]', s)
            if m:
                cs = m.group(1).lower() + ("." + m.group(2) if m.group(2) is not None else "")
                skip = cs == sec
                found = found or skip
                if skip:
                    continue
            if not skip:
                out.append(raw)
        if found:
            self.write("\n".join(out).rstrip("\n") + "\n" if any(x.strip() for x in out) else "", uid, gid)
        return found

    def rename_section(self, old: str, new: str, uid=1000, gid=1000) -> bool:
        lines = self.text().split("\n")
        found = False
        for i, raw in enumerate(lines):
            s = raw.strip()
            m = re.match(r'\[\s*([A-Za-z0-9.-]+)(?:\s+"((?:[^"\\]|\\.)*)")?\s*\]', s)
            if m:
                cs = m.group(1).lower() + ("." + m.group(2) if m.group(2) is not None else "")
                if cs == old:
                    lines[i] = self._section_header(new)
                    found = True
        if found:
            self.write("\n".join(lines), uid, gid)
        return found


class Config:
    """system (/etc/gitconfig), global (~/.gitconfig, ~/.config/git/config), local (.git/config), -c overrides."""

    def __init__(self, fs, home: str, gitdir: str | None, overrides=(), env=None):
        self.fs = fs
        self.home = home
        env = env or {}
        gl = env.get("GIT_CONFIG_GLOBAL")
        self.files = [("system", ConfigFile(fs, "/etc/gitconfig"))]
        xdg = (env.get("XDG_CONFIG_HOME") or f"{home}/.config") + "/git/config"
        if gl:
            self.files.append(("global", ConfigFile(fs, gl)))
        else:
            self.files.append(("global", ConfigFile(fs, xdg)))
            self.files.append(("global", ConfigFile(fs, f"{home}/.gitconfig")))
        if gitdir:
            self.files.append(("local", ConfigFile(fs, f"{gitdir}/config")))
            wt = f"{gitdir}/config.worktree"
            if fs.exists(wt):
                self.files.append(("worktree", ConfigFile(fs, wt)))
        self.overrides = list(overrides)

    def global_file(self) -> ConfigFile:
        """Where --global writes: ~/.gitconfig (or the XDG file when only it exists)."""
        files = [f for scope, f in self.files if scope == "global"]
        if len(files) == 1:
            return files[0]
        xdg, home = files
        return xdg if xdg.exists() and not home.exists() else home

    def local_file(self) -> ConfigFile | None:
        for scope, f in self.files:
            if scope == "local":
                return f
        return None

    def all(self):
        """-> [(scope, path, key, value)] in precedence order (later wins)."""
        out = []
        for scope, f in self.files:
            for k, v, _ in f.entries():
                out.append((scope, f.path, k, v))
        for k, v in self.overrides:
            out.append(("command", "", canon_key(k), v))
        return out

    def get_all(self, key: str) -> list:
        key = canon_key(key)
        return [v for _, _, k, v in self.all() if canon_key(k) == key]

    def get(self, key: str, default=None):
        vals = self.get_all(key)
        if not vals:
            return default
        v = vals[-1]
        return "true" if v is None else v

    def get_bool(self, key: str, default=False) -> bool:
        v = self.get(key)
        if v is None:
            return default
        return v.lower() in ("true", "yes", "on", "1", "")

    def get_int(self, key: str, default=0) -> int:
        v = self.get(key)
        if v is None:
            return default
        m = re.match(r"^(-?\d+)([kmg]?)$", v.strip().lower())
        if not m:
            return default
        return int(m.group(1)) * {"": 1, "k": 1024, "m": 1 << 20, "g": 1 << 30}[m.group(2)]

    def section_names(self, section: str) -> list:
        """Subsection names of a section (remote "origin", branch "main" ...), in order of appearance."""
        out = []
        for _, _, k, _ in self.all():
            parts = k.split(".")
            if len(parts) >= 3 and parts[0] == section:
                sub = ".".join(parts[1:-1])
                if sub not in out:
                    out.append(sub)
        return out


# ---------------------------------------------------------------------------- the index

class Entry:
    __slots__ = ("path", "sha", "mode", "stage", "ctime", "mtime", "dev", "ino", "uid", "gid", "size", "flags2")

    def __init__(self, path, sha, mode, stage=0, ctime=0, mtime=0, dev=0, ino=0, uid=0, gid=0, size=0):
        self.path, self.sha, self.mode, self.stage = path, sha, mode, stage
        self.ctime, self.mtime, self.dev, self.ino, self.uid, self.gid, self.size = ctime, mtime, dev, ino, uid, gid, size
        self.flags2 = 0      # extended flags: intent-to-add (0x2000) / skip-worktree (0x4000)

    @property
    def mode_str(self) -> str:
        return f"{self.mode:o}"


class Index:
    """The staging area: path -> Entry (stage 0), plus conflict stages 1-3 during a merge."""

    def __init__(self):
        self.entries: dict[tuple[str, int], Entry] = {}

    def get(self, path, stage=0):
        return self.entries.get((path, stage))

    def add(self, e: Entry):
        if e.stage == 0:
            for s in (1, 2, 3):
                self.entries.pop((e.path, s), None)
        self.entries[(e.path, e.stage)] = e

    def remove(self, path):
        for s in (0, 1, 2, 3):
            self.entries.pop((path, s), None)

    def paths(self):
        return sorted({p for p, _ in self.entries})

    def stage0(self) -> dict:
        return {p: e for (p, s), e in self.entries.items() if s == 0}

    def conflicts(self) -> dict:
        out = {}
        for (p, s), e in self.entries.items():
            if s:
                out.setdefault(p, {})[s] = e
        return out

    def flat(self) -> dict:
        """path -> (mode str, sha) for stage-0 entries (what write-tree stores)."""
        return {p: (e.mode_str, e.sha) for p, e in self.stage0().items() if not (e.flags2 & 0x2000)}

    # ----- the real binary format (DIRC version 2, or 3 for extended flags)
    def encode(self) -> bytes:
        ents = sorted(self.entries.values(), key=lambda e: (e.path.encode("utf-8", "surrogateescape"), e.stage))
        ext = any(e.flags2 for e in ents)
        out = bytearray(b"DIRC" + struct.pack(">II", 3 if ext else 2, len(ents)))
        for e in ents:
            name = e.path.encode("utf-8", "surrogateescape")
            flags = min(len(name), 0xFFF) | (e.stage << 12) | (0x4000 if e.flags2 else 0)
            start = len(out)
            out += struct.pack(">IIIIIIIIII", int(e.ctime // 10**9) & 0xFFFFFFFF, int(e.ctime % 10**9),
                               int(e.mtime // 10**9) & 0xFFFFFFFF, int(e.mtime % 10**9), e.dev & 0xFFFFFFFF,
                               e.ino & 0xFFFFFFFF, e.mode, e.uid, e.gid, e.size & 0xFFFFFFFF)
            out += bytes.fromhex(e.sha) + struct.pack(">H", flags)
            if e.flags2:
                out += struct.pack(">H", e.flags2)
            out += name
            n = len(out) - start
            out += b"\0" * (8 - (n % 8) if n % 8 else 8)
        out += hashlib.sha1(out).digest()
        return bytes(out)

    @staticmethod
    def decode(data: bytes) -> "Index":
        idx = Index()
        if len(data) < 12 or data[:4] != b"DIRC":
            return idx
        version, n = struct.unpack(">II", data[4:12])
        pos = 12
        for _ in range(n):
            f = struct.unpack(">IIIIIIIIII", data[pos:pos + 40])
            sha = data[pos + 40:pos + 60].hex()
            flags = struct.unpack(">H", data[pos + 60:pos + 62])[0]
            p = pos + 62
            flags2 = 0
            if version >= 3 and flags & 0x4000:
                flags2 = struct.unpack(">H", data[p:p + 2])[0]
                p += 2
            nul = data.index(b"\0", p)
            path = data[p:nul].decode("utf-8", "surrogateescape")
            e = Entry(path, sha, f[6], (flags >> 12) & 3, f[0] * 10**9 + f[1], f[2] * 10**9 + f[3], f[4], f[5],
                      f[7], f[8], f[9])
            e.flags2 = flags2
            idx.entries[(path, e.stage)] = e
            ent_len = nul - pos            # fixed part + name; then 1-8 NULs pad it to a multiple of 8
            pos += (ent_len + 8) & ~7
        return idx


# ---------------------------------------------------------------------------- .gitignore

class IgnoreRule:
    __slots__ = ("pattern", "negate", "dir_only", "anchored", "base", "source", "line", "rx")

    def __init__(self, pattern, base, source, line):
        self.source, self.line, self.base = source, line, base
        p = pattern
        self.pattern = pattern
        self.negate = p.startswith("!")
        if self.negate:
            p = p[1:]
        self.dir_only = p.endswith("/")
        p = p.rstrip("/")
        self.anchored = "/" in p
        p = p.lstrip("/") if p.startswith("/") else p
        self.rx = re.compile(_glob_rx(p) if self.anchored else _glob_rx(p))

    def matches(self, rel: str, is_dir: bool) -> bool:
        """rel: path relative to this rule's directory."""
        if self.dir_only and not is_dir:
            return False
        if self.anchored:
            return bool(self.rx.fullmatch(rel))
        return bool(self.rx.fullmatch(rel.rsplit("/", 1)[-1]))


def _glob_rx(p: str) -> str:
    out = []
    i = 0
    n = len(p)
    while i < n:
        c = p[i]
        if p.startswith("**/", i) and (i == 0 or p[i - 1] == "/"):
            out.append("(?:.*/)?")
            i += 3
            continue
        if p.startswith("/**", i) and i + 3 == n:
            out.append("/.*")
            i += 3
            continue
        if c == "*":
            if p.startswith("**", i):
                out.append(".*")
                i += 2
                continue
            out.append("[^/]*")
        elif c == "?":
            out.append("[^/]")
        elif c == "[":
            j = p.find("]", i + 1)
            if j < 0:
                out.append(re.escape(c))
            else:
                body = p[i + 1:j]
                if body.startswith("!"):
                    body = "^" + body[1:]
                out.append("[" + body.replace("\\", "\\\\") + "]")
                i = j
        elif c == "\\" and i + 1 < n:
            i += 1
            out.append(re.escape(p[i]))
        else:
            out.append(re.escape(c))
        i += 1
    return "".join(out)


def parse_ignore(text: str, base: str, source: str):
    rules = []
    for n, raw in enumerate(text.split("\n"), 1):
        line = raw.rstrip("\r")
        if not line or line.startswith("#"):
            continue
        # trailing spaces are ignored unless escaped
        stripped = re.sub(r"(?<!\\)\s+$", "", line)
        if not stripped:
            continue
        if stripped.startswith("\\#") or stripped.startswith("\\!"):
            stripped = stripped[1:]
            r = IgnoreRule(stripped, base, source, n)
            r.negate = False
            r.pattern = raw.rstrip()
            rules.append(r)
            continue
        rules.append(IgnoreRule(stripped, base, source, n))
    return rules


# ---------------------------------------------------------------------------- the repository

class Repo:
    """gitdir: the .git directory (absolute); worktree: the top of the working tree (None for a bare repo)."""

    def __init__(self, ctx, gitdir: str, worktree: str | None, commondir: str | None = None):
        self.ctx = ctx
        self.fs = ctx.fs
        self.gitdir = gitdir
        self.common = commondir or gitdir      # shared by linked worktrees: objects, refs (but not HEAD), config
        self.worktree = worktree
        self.odb = odb(ctx.world)
        self._index = None
        self._config = None
        self._ignores = None

    # -------------------------------------------------- environment
    @property
    def env(self):
        return self.ctx.env

    @property
    def home(self) -> str:
        return self.env.get("HOME") or self.ctx.host.home_of(self.ctx.uid)

    @property
    def bare(self) -> bool:
        return self.worktree is None

    @property
    def config(self) -> Config:
        if self._config is None:
            self._config = Config(self.fs, self.home, self.common, getattr(self.ctx, "git_overrides", ()), self.env)
        return self._config

    def reload_config(self):
        self._config = None

    def now(self) -> int:
        return self.ctx.now // 1_000_000_000

    def tz(self, when: int | None = None) -> str:
        from ..timeutil import local_dt
        d = local_dt((when if when is not None else self.now()) * 1_000_000_000, self.env.get("TZ"))
        off = d.utcoffset()
        mins = int(off.total_seconds() // 60) if off else 0
        sign = "+" if mins >= 0 else "-"
        mins = abs(mins)
        return f"{sign}{mins // 60:02d}{mins % 60:02d}"

    def ident(self, role: str = "author", strict: bool = True) -> Person:
        """GIT_AUTHOR_NAME / user.name ..., and the date (GIT_AUTHOR_DATE or now)."""
        R = role.upper()
        env = self.env
        name = env.get(f"GIT_{R}_NAME") or self.config.get("user.name")
        email = env.get(f"GIT_{R}_EMAIL") or self.config.get("user.email") or env.get("EMAIL")
        if strict and (not name or not email):
            user = self.ctx.host.user_name(self.ctx.uid)
            raise GitError(
                "Author identity unknown\n\n*** Please tell me who you are.\n\nRun\n\n"
                '  git config --global user.email "you@example.com"\n  git config --global user.name "Your Name"\n\n'
                "to set your account's default identity.\nOmit --global to set the identity only in this repository.\n\n"
                f"fatal: unable to auto-detect email address (got '{user}@{self.ctx.host.name}.(none)')")
        name = name or self.ctx.host.user_name(self.ctx.uid)
        email = email or f"{self.ctx.host.user_name(self.ctx.uid)}@{self.ctx.host.name}"
        date = env.get(f"GIT_{R}_DATE")
        when, tz = self.now(), None
        if date:
            when, tz = parse_git_date(date, self)
        return Person(name, email, when, tz or self.tz(when))

    # -------------------------------------------------- files in .git
    def path(self, rel: str, common: bool = False) -> str:
        base = self.common if common else self.gitdir
        return f"{base}/{rel}"

    def read(self, rel: str, common: bool = False) -> str | None:
        try:
            return bytes(self.fs.lookup(self.path(rel, common)).data).decode("utf-8", "surrogateescape")
        except FSError:
            return None

    def write(self, rel: str, text, common: bool = False, mode: int = 0o644):
        self.fs.write_file(self.path(rel, common), text, "/", mode, self.ctx.uid, self.ctx.proc.gid)

    def remove(self, rel: str, common: bool = False):
        try:
            self.fs.unlink(self.path(rel, common))
        except FSError:
            pass

    def exists(self, rel: str, common: bool = False) -> bool:
        return self.fs.exists(self.path(rel, common))

    def rmtree(self, rel: str, common: bool = False):
        rmtree(self.fs, self.path(rel, common))

    # -------------------------------------------------- objects
    def store(self, kind: str, data, loose: bool = True) -> str:
        sha = self.odb.put(kind, data)
        if loose:
            p = f"objects/{sha[:2]}/{sha[2:]}"
            if not self.exists(p, True) and not isinstance(data, LazyData):
                self.write(p, loose_bytes(kind, data), True, 0o444)
        return sha

    def blob_of_file(self, data) -> str:
        if isinstance(data, LazyData) and data.oid:
            self.odb.objects.setdefault(data.oid, ("blob", data))
            return data.oid
        return self.store("blob", data)

    # -------------------------------------------------- refs
    def ref_names(self, prefix: str = "refs/") -> dict:
        """refname -> sha for every loose and packed ref under prefix."""
        out = {}
        packed = self.read("packed-refs", True) or ""
        for line in packed.split("\n"):
            if not line or line[0] in "#^":
                continue
            sha, _, name = line.partition(" ")
            if name.startswith(prefix):
                out[name] = sha
        base = self.path("refs", True)
        try:
            root = self.fs.lookup(base)
        except FSError:
            root = None
        if root is not None:
            for p, node in self.fs.walk(root, base):
                if node.type == "f":
                    name = "refs" + p[len(base):]
                    if name.startswith(prefix):
                        val = bytes(node.data).decode().strip()
                        if val.startswith("ref: "):
                            continue
                        out[name] = val
        return dict(sorted(out.items()))

    def read_ref_raw(self, name: str):
        """A ref's own value: ('sha', x) | ('sym', target) | None."""
        common = name.startswith("refs/") and not name.startswith("refs/bisect")
        text = self.read(name, common)
        if text is not None:
            text = text.strip()
            if text.startswith("ref: "):
                return ("sym", text[5:].strip())
            if text:
                return ("sha", text.split()[0])
        if name.startswith("refs/"):
            packed = self.read("packed-refs", True) or ""
            for line in packed.split("\n"):
                if line.endswith(" " + name) and line[0] not in "#^":
                    return ("sha", line.split(" ")[0])
        return None

    def resolve_ref(self, name: str, depth: int = 0):
        """Follow symbolic refs: -> sha or None."""
        r = self.read_ref_raw(name)
        if r is None or depth > 5:
            return None
        if r[0] == "sym":
            return self.resolve_ref(r[1], depth + 1)
        return r[1]

    def head_ref(self) -> str | None:
        """'refs/heads/main' (even if the branch has no commits yet), or None when detached."""
        r = self.read_ref_raw("HEAD")
        if r and r[0] == "sym":
            return r[1]
        return None

    def shallow(self) -> set:
        """Boundary commits of a shallow clone (.git/shallow)."""
        text = self.read("shallow", True) or ""
        return {ln.strip() for ln in text.split("\n") if ln.strip()}

    def head_branch(self) -> str | None:
        ref = self.head_ref()
        return ref[len("refs/heads/"):] if ref and ref.startswith("refs/heads/") else None

    def head(self) -> str | None:
        return self.resolve_ref("HEAD")

    def head_commit(self):
        h = self.head()
        return self.odb.commit(h) if h else None

    def head_tree(self) -> str:
        c = self.head_commit()
        return c.tree if c else EMPTY_TREE

    def set_head_symbolic(self, ref: str, msg: str | None = None):
        old = self.head()
        self.write("HEAD", f"ref: {ref}\n")
        new = self.resolve_ref(ref)
        if msg and (old or new):
            self.reflog_append("HEAD", old or ZERO, new or ZERO, msg)

    def set_head_detached(self, sha: str, msg: str | None = None):
        old = self.head()
        self.write("HEAD", sha + "\n")
        if msg:
            self.reflog_append("HEAD", old or ZERO, sha, msg)

    def update_ref(self, name: str, sha: str, msg: str | None = None, old: str | None = None):
        """Point a ref at sha (logging it, and HEAD's log too when HEAD is on that branch)."""
        prev = self.resolve_ref(name) if old is None else old
        if name == "HEAD":
            hr = self.head_ref()
            if hr:
                return self.update_ref(hr, sha, msg)
            self.set_head_detached(sha, msg)
            return
        common = name.startswith("refs/")
        self._unpack_ref(name)
        self.write(name, sha + "\n", common)
        if msg is not None and (name.startswith(("refs/heads/", "refs/remotes/", "refs/stash")) or
                                self.exists(f"logs/{name}", True)):
            self.reflog_append(name, prev or ZERO, sha, msg)
            if self.head_ref() == name:
                self.reflog_append("HEAD", prev or ZERO, sha, msg)

    def delete_ref(self, name: str):
        self.remove(name, True)
        self._unpack_ref(name)
        rmtree(self.fs, self.path(f"logs/{name}", True))
        prune_empty_dirs(self.fs, self.path(posixpath.dirname(name), True), self.path("refs", True))

    def _unpack_ref(self, name: str):
        packed = self.read("packed-refs", True)
        if packed and (" " + name + "\n") in packed + "\n":
            lines = packed.split("\n")
            out = []
            skip_peel = False
            for line in lines:
                if skip_peel and line.startswith("^"):
                    skip_peel = False
                    continue
                skip_peel = False
                if line.endswith(" " + name) and not line.startswith("#"):
                    skip_peel = True
                    continue
                out.append(line)
            self.write("packed-refs", "\n".join(out), True)

    def write_packed_refs(self, refs: dict):
        lines = ["# pack-refs with: peeled fully-peeled sorted "]
        for name, sha in sorted(refs.items()):
            lines.append(f"{sha} {name}")
            if name.startswith("refs/tags/") and self.odb.has(sha) and self.odb.kind(sha) == "tag":
                lines.append("^" + self.peel(sha))
        self.write("packed-refs", "\n".join(lines) + "\n", True)

    def peel(self, sha: str) -> str:
        """Tags -> the object they point at (recursively)."""
        seen = 0
        while sha and self.odb.has(sha) and self.odb.kind(sha) == "tag" and seen < 10:
            sha = self.odb.tag(sha).object
            seen += 1
        return sha

    def branches(self) -> dict:
        return {k[len("refs/heads/"):]: v for k, v in self.ref_names("refs/heads/").items()}

    def tags(self) -> dict:
        return {k[len("refs/tags/"):]: v for k, v in self.ref_names("refs/tags/").items()}

    def remote_branches(self) -> dict:
        out = {}
        for k, v in self.ref_names("refs/remotes/").items():
            out[k[len("refs/remotes/"):]] = v
        return out

    def remotes(self) -> list:
        return self.config.section_names("remote")

    def upstream(self, branch: str):
        """(remote, merge ref) configured for a branch, or None."""
        remote = self.config.get(f"branch.{branch}.remote")
        merge = self.config.get(f"branch.{branch}.merge")
        if remote and merge:
            return remote, merge
        return None

    def upstream_ref(self, branch: str) -> str | None:
        """refs/remotes/origin/main for a branch tracking origin/main (or refs/heads/x for a local upstream)."""
        up = self.upstream(branch)
        if not up:
            return None
        remote, merge = up
        if remote == ".":
            return merge
        short = merge[len("refs/heads/"):] if merge.startswith("refs/heads/") else merge
        return f"refs/remotes/{remote}/{short}"

    # -------------------------------------------------- reflog
    def reflog_append(self, name: str, old: str, new: str, msg: str):
        if not self.config.get_bool("core.logAllRefUpdates", True) and not self.bare:
            return
        who = self.ident("committer", strict=False)
        line = f"{old} {new} {who.name} <{who.email}> {who.when} {who.tz}\t{msg}\n"
        common = name != "HEAD"
        p = self.path(f"logs/{name}", common)
        try:
            node = self.fs.lookup(p)
            node.data += line.encode("utf-8", "surrogateescape")
            node.mtime = self.ctx.now
        except FSError:
            self.fs.write_file(p, line, "/", 0o644, self.ctx.uid, self.ctx.proc.gid)

    def reflog(self, name: str) -> list:
        """Newest first: [(old, new, Person, msg)]."""
        common = name != "HEAD"
        text = self.read(f"logs/{name}", common) or ""
        out = []
        for line in text.split("\n"):
            if not line.strip():
                continue
            head, _, msg = line.partition("\t")
            parts = head.split(" ", 2)
            if len(parts) < 3:
                continue
            out.append((parts[0], parts[1], Person.parse(parts[2]), msg))
        return out[::-1]

    # -------------------------------------------------- index
    @property
    def index(self) -> Index:
        if self._index is None:
            try:
                data = bytes(self.fs.lookup(self.path("index")).data)
            except FSError:
                data = b""
            self._index = Index.decode(data) if data else Index()
        return self._index

    @property
    def index_mtime(self) -> int:
        try:
            return self.fs.lookup(self.path("index")).mtime
        except FSError:
            return 0

    def write_index(self):
        if self._index is not None:
            self.write("index", self._index.encode())

    def reset_index_to(self, tree: str, keep_stat=True):
        old = self.index.stage0()
        idx = Index()
        for path, (mode, sha) in self.odb.flat(tree).items():
            e = Entry(path, sha, int(mode, 8))
            o = old.get(path)
            if keep_stat and o is not None and o.sha == sha and o.mode == e.mode:
                e = o
            idx.add(e)
        self._index = idx

    # -------------------------------------------------- the working tree
    def wt_path(self, rel: str) -> str:
        return f"{self.worktree}/{rel}" if rel else self.worktree

    def rel(self, abs_path: str) -> str:
        """An absolute path -> relative to the top of the working tree."""
        p = posixpath.normpath(abs_path)
        top = self.worktree
        if p == top:
            return ""
        if p.startswith(top + "/"):
            return p[len(top) + 1:]
        raise GitError(f"fatal: {abs_path}: '{abs_path}' is outside repository at '{top}'")

    def wt_node(self, rel: str):
        try:
            return self.fs.lookup(self.wt_path(rel), follow=False)
        except FSError:
            return None

    def file_mode(self, node) -> int:
        if node.type == "l":
            return 0o120000
        if node.type == "d":
            return 0o160000
        return 0o100755 if node.mode & 0o111 and self.config.get_bool("core.fileMode", True) else 0o100644

    def file_bytes(self, node):
        if node.type == "l":
            return node.target.encode("utf-8", "surrogateescape")
        return node.data

    def wt_sha(self, rel: str, write: bool = False) -> tuple[str, int] | None:
        """(blob id, mode) of a working-tree file, or None when it's missing."""
        node = self.wt_node(rel)
        if node is None or (node.type == "d" and not self._is_submodule(rel, node)):
            return None
        if node.type == "d":
            sub = self._submodule_head(rel)
            return (sub or ZERO), 0o160000
        mode = self.file_mode(node)
        data = self.file_bytes(node)
        data = self.clean_filter(rel, data)
        if write:
            return self.blob_of_file(data), mode
        if isinstance(data, LazyData) and data.oid:
            return data.oid, mode
        return hash_object("blob", data), mode

    def stat_matches(self, e: Entry, node) -> bool:
        """Stat data says the file is unchanged. A file modified no earlier than the index was written is "racily
        clean" and must be compared by content, as in git."""
        if node is None or node.type == "d":
            return False
        if e.mtime >= self.index_mtime:
            return False
        size = len(node.data) if node.type == "f" else len(node.target.encode())
        return e.mtime == node.mtime and e.size == size and e.ino == node.ino

    def index_entry_for(self, rel: str, sha: str, mode: int, node) -> Entry:
        size = len(node.data) if node is not None and node.type == "f" else (
            len(node.target.encode()) if node is not None and node.type == "l" else 0)
        return Entry(rel, sha, mode, 0, node.ctime if node else 0, node.mtime if node else 0, 2049,
                     node.ino if node else 0, node.uid if node else 0, node.gid if node else 0, size)

    def _is_submodule(self, rel, node) -> bool:
        return node.type == "d" and ".git" in (node.entries or {})

    def _submodule_head(self, rel):
        from .discover import open_repo_at
        try:
            sub = open_repo_at(self.ctx, self.wt_path(rel))
            return sub.head() if sub else None
        except GitError:
            return None

    def clean_filter(self, rel: str, data):
        """`git add` filters: Git LFS turns a tracked big file into a pointer; nbstripout empties notebook outputs."""
        if self.lfs_tracked(rel) and not bytes(data[:40]).startswith(b"version https://git-lfs"):
            from .lfs import pointer_for
            return pointer_for(self, data)
        filt = self.attr_value(rel, "filter")
        if filt and filt != "lfs" and "nbstripout" in (self.config.get(f"filter.{filt}.clean") or ""):
            from .tools import strip_notebook
            out = strip_notebook(bytes(data))
            return data if out is None else out
        return data

    def attr_value(self, rel: str, key: str):
        """The value an attribute has for a path (`filter=lfs` -> "lfs"; `-text` -> False; unset -> None)."""
        val = None
        for pat, base, kv in self.attributes():
            r = rel[len(base) + 1:] if base and rel.startswith(base + "/") else (rel if not base else None)
            if r is None:
                continue
            name = r if "/" in pat else r.rsplit("/", 1)[-1]
            if not fnmatch.fnmatchcase(name, pat.lstrip("/")):
                continue
            for a in kv:
                if a.startswith(key + "="):
                    val = a.split("=", 1)[1]
                elif a == key:
                    val = True
                elif a in ("-" + key, "!" + key):
                    val = False if a[0] == "-" else None
        return val

    def lfs_tracked(self, rel: str) -> bool:
        attrs = self.attributes()
        hit = False
        for pat, base, kv in attrs:
            r = rel[len(base) + 1:] if base and rel.startswith(base + "/") else (rel if not base else None)
            if r is None:
                continue
            name = r if "/" in pat else r.rsplit("/", 1)[-1]
            if fnmatch.fnmatchcase(name, pat.lstrip("/")):
                if "filter=lfs" in kv:
                    hit = True
                elif "-filter" in kv or "!filter" in kv:
                    hit = False
        return hit

    def attributes(self):
        """[(pattern, base dir, [attrs])] from .gitattributes files and .git/info/attributes."""
        out = []
        if self.bare:
            return out
        for base, text in self._files_named(".gitattributes"):
            for line in text.split("\n"):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                out.append((parts[0], base, parts[1:]))
        info = self.read("info/attributes", True)
        if info:
            for line in info.split("\n"):
                parts = line.split()
                if parts and not parts[0].startswith("#"):
                    out.append((parts[0], "", parts[1:]))
        return out

    def _files_named(self, name: str):
        """(dir relative to the top, text) for every `name` in the working tree (outside .git), top first."""
        out = []
        if self.bare:
            return out

        def walk(rel):
            node = self.wt_node(rel)
            if node is None or node.type != "d":
                return
            f = node.entries.get(name)
            if f is not None and f.type == "f":
                out.append((rel, bytes(f.data).decode("utf-8", "replace")))
            for child in sorted(node.entries):
                c = node.entries[child]
                if c.type == "d" and child != ".git" and not (rel == "" and child == ".git"):
                    if ".git" in (c.entries or {}):
                        continue
                    walk(f"{rel}/{child}" if rel else child)
        walk("")
        return out

    # -------------------------------------------------- ignore rules
    def ignore_rules(self):
        if self._ignores is None:
            rules = []
            excludes = self.config.get("core.excludesFile")
            path = excludes.replace("~", self.home, 1) if excludes else \
                (self.env.get("XDG_CONFIG_HOME") or f"{self.home}/.config") + "/git/ignore"
            try:
                text = bytes(self.fs.lookup(path).data).decode("utf-8", "replace")
                rules += parse_ignore(text, "", path if excludes else path)
            except FSError:
                pass
            info = self.read("info/exclude", True)
            if info:
                rules += parse_ignore(info, "", ".git/info/exclude")
            for base, text in self._files_named(".gitignore"):
                rules += parse_ignore(text, base, (base + "/" if base else "") + ".gitignore")
            self._ignores = rules
        return self._ignores

    def ignored_by(self, rel: str, is_dir: bool):
        """The rule that decides `rel` (its last matching rule, or a parent directory's), or None."""
        parts = rel.split("/")
        for i in range(1, len(parts)):
            d = "/".join(parts[:i])
            r = self._last_match(d, True)
            if r is not None and not r.negate:
                return r
        return self._last_match(rel, is_dir)

    def _last_match(self, rel: str, is_dir: bool):
        hit = None
        for r in self.ignore_rules():
            if r.base:
                if not rel.startswith(r.base + "/"):
                    continue
                sub = rel[len(r.base) + 1:]
            else:
                sub = rel
            if r.matches(sub, is_dir):
                hit = r
        return hit

    def is_ignored(self, rel: str, is_dir: bool = False) -> bool:
        r = self.ignored_by(rel, is_dir)
        return r is not None and not r.negate

    # -------------------------------------------------- walking the working tree
    def worktree_files(self, include_ignored=False, under: str = ""):
        """Every file in the working tree (not .git, not ignored unless asked): rel -> node."""
        out = {}
        if self.bare:
            return out
        idx = self.index

        def walk(rel, node):
            for name in sorted(node.entries):
                if rel == "" and name == ".git":
                    continue
                child = node.entries[name]
                p = f"{rel}/{name}" if rel else name
                if child.type == "d":
                    if ".git" in (child.entries or {}) and p != "":
                        out[p] = child
                        continue
                    tracked_inside = any(k.startswith(p + "/") for k, _ in idx.entries)
                    if not include_ignored and not tracked_inside and self.is_ignored(p, True):
                        continue
                    walk(p, child)
                else:
                    if not include_ignored and idx.get(p) is None and self.is_ignored(p, False):
                        continue
                    out[p] = child
        root = self.wt_node(under) if under else self.fs.lookup(self.worktree)
        if root is not None and root.type == "d":
            walk(under, root)
        return out

    # -------------------------------------------------- writing the working tree
    def checkout_file(self, rel: str, sha: str, mode: str | int, force=True):
        mode = int(mode, 8) if isinstance(mode, str) else mode
        p = self.wt_path(rel)
        parent = posixpath.dirname(p)
        # a file where a directory must go (or the reverse) is removed first
        self._make_dirs(parent)
        node = self.wt_node(rel)
        if node is not None and node.type == "d" and mode != 0o160000:
            rmtree(self.fs, p)
            node = None
        if mode == 0o160000:
            self.fs.mkdir(p, "/", 0o755, self.ctx.uid, self.ctx.proc.gid, parents=True)
            return
        data = self.odb.raw(sha)
        data = self.smudge_filter(rel, data)
        if mode == 0o120000:
            if node is not None:
                self.fs.unlink(p)
            self.fs.symlink(bytes(data).decode("utf-8", "surrogateescape"), p, "/", self.ctx.uid, self.ctx.proc.gid)
            return
        if node is not None and node.type == "l":
            self.fs.unlink(p)
        node = self.fs.create(p, "/", 0o755 if mode == 0o100755 else 0o644, self.ctx.uid, self.ctx.proc.gid)
        if isinstance(data, LazyData):
            self.fs.charge(-len(node.data) if not isinstance(node.data, LazyData) else 0)
            node.data = data.clone()
        else:
            raw = bytes(data)
            self.fs.charge(len(raw) - (len(node.data) if not isinstance(node.data, LazyData) else 0))
            node.data = LazyData(src=raw) if len(raw) > 4096 else bytearray(raw)
        node.mode = (node.mode & ~0o777) | (0o755 if mode == 0o100755 else 0o644)
        node.mtime = node.ctime = self.ctx.now

    def smudge_filter(self, rel: str, data):
        if self.lfs_tracked(rel) and bytes(data[:40]).startswith(b"version https://git-lfs"):
            from .lfs import smudge
            return smudge(self, data)
        return data

    def remove_file(self, rel: str):
        p = self.wt_path(rel)
        node = self.wt_node(rel)
        if node is None:
            return
        if node.type == "d":
            rmtree(self.fs, p)
        else:
            self.fs.unlink(p)
        # drop directories left empty, up to the top
        prune_empty_dirs(self.fs, posixpath.dirname(p), self.worktree)

    def _make_dirs(self, d: str):
        if d == self.worktree or self.fs.exists(d):
            node = self.fs.lookup(d, follow=False) if self.fs.exists(d, follow=False) else None
            if node is not None and node.type != "d":
                self.fs.unlink(d)
            else:
                return
        parent = posixpath.dirname(d)
        if parent != d:
            self._make_dirs(parent)
        if not self.fs.exists(d):
            self.fs.mkdir(d, "/", 0o755, self.ctx.uid, self.ctx.proc.gid)


def rmtree(fs, path: str):
    try:
        node = fs.lookup(path, follow=False)
    except FSError:
        return
    if node.type == "d":
        for name in list(node.entries):
            rmtree(fs, path.rstrip("/") + "/" + name)
        fs.rmdir(path)
    else:
        fs.unlink(path)


def prune_empty_dirs(fs, d: str, stop: str):
    while d and d != stop and d.startswith(stop):
        try:
            node = fs.lookup(d, follow=False)
        except FSError:
            d = posixpath.dirname(d)
            continue
        if node.type != "d" or node.entries:
            break
        try:
            fs.rmdir(d)
        except FSError:
            break
        d = posixpath.dirname(d)


def parse_git_date(text: str, repo) -> tuple[int, str | None]:
    """GIT_AUTHOR_DATE / --date: '@1700000000 +0100', '1700000000 +0100', ISO 8601, RFC 2822, or approxidate."""
    t = text.strip()
    m = re.match(r"^@?(\d{9,})(?:\s+([+-]\d{4}))?$", t)
    if m:
        return int(m.group(1)), m.group(2)
    m = re.match(r"^(\d{4}-\d\d-\d\d)[T ](\d\d:\d\d(?::\d\d)?)\s*(Z|[+-]\d\d:?\d\d)?$", t)
    if m:
        from datetime import datetime, timezone, timedelta
        tm = m.group(2) if m.group(2).count(":") == 2 else m.group(2) + ":00"
        d = datetime.fromisoformat(f"{m.group(1)}T{tm}")
        z = m.group(3)
        if z and z != "Z":
            z = z.replace(":", "")
            off = (1 if z[0] == "+" else -1) * (int(z[1:3]) * 60 + int(z[3:5]))
            d = d.replace(tzinfo=timezone(timedelta(minutes=off)))
            return int(d.timestamp()), z
        if z == "Z":
            return int(d.replace(tzinfo=timezone.utc).timestamp()), "+0000"
        from ..timeutil import localize
        ld = localize(d, repo.env.get("TZ"))
        return int(ld.timestamp()), None
    from ..timeutil import parse_date
    ns = parse_date(t, repo.ctx.now, repo.env.get("TZ"))
    return ns // 1_000_000_000, None
