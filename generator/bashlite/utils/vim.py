"""vim, vi, view: Vim on the machine (the editor itself is generator/bashlite/vim).

In the live terminal the process asks the world to show Vim (world.vim) and sleeps until Vim quits; the terminal
sends it every key and draws its screen. Files, shell commands (:!cmd, :r !cmd, filters, :terminal) and the clock
come from the machine through VimIO.
"""

from __future__ import annotations

import posixpath
import re

from ..exec import util
from ..kernel import TimeLimit
from ..vfs import FSError


class VimIO:
    """What Vim needs from the machine, as the user who started it."""

    def __init__(self, world, proc, env):
        self.world = world
        self.host = world.local
        self.fs = self.host.fs
        self.proc = proc
        self.uid, self.gid = proc.uid, proc.gid
        self.gids = proc.gids or {proc.gid}
        self.env = env
        self.sess = None

    def _node(self, path):
        try:
            return self.fs.lookup(path, "/", True, self.uid, self.gids)
        except FSError:
            return None

    def read(self, path):
        node = self._node(path)
        if node is None or node.type != "f":
            return None
        if not self.fs.can(node, 4, self.uid, self.gids):
            return None
        return bytes(node.data).decode("utf-8", "surrogateescape")

    def writable(self, path):
        node = self._node(path)
        if node is None:
            d = self._node(posixpath.dirname(path) or "/")
            return d is not None and self.fs.can(d, 3, self.uid, self.gids)
        return self.fs.can(node, 2, self.uid, self.gids)

    def write(self, path, text, force=False):
        data = text.encode("utf-8", "surrogateescape")
        node = self._node(path)
        try:
            if node is not None:
                if node.type == "d":
                    return "is a directory"
                if not self.fs.can(node, 2, self.uid, self.gids):
                    return "Permission denied"
                self.fs.charge(len(data) - len(node.data))
                node.data[:] = data
                node.mtime = node.ctime = self.world.kernel.now
            else:
                d = self._node(posixpath.dirname(path) or "/")
                if d is None:
                    return "No such file or directory"
                if not self.fs.can(d, 3, self.uid, self.gids):
                    return "Permission denied"
                self.fs.write_file(path, data, "/", 0o666 & ~getattr(self.proc, "umask", 0o022), self.uid, self.gid)
        except FSError as e:
            return e.msg
        return None

    def exists(self, path):
        return self._node(path) is not None

    def isdir(self, path):
        node = self._node(path)
        return node is not None and node.type == "d"

    def listdir(self, path):
        node = self._node(path)
        if node is None or node.type != "d":
            return []
        out = []
        for name in self.fs.listdir(node):
            child = self._node(path.rstrip("/") + "/" + name)
            if child is None:
                continue
            out.append((name, child.type == "d", child.type == "f" and bool(child.mode & 0o111)))
        return out

    def glob(self, cwd, pattern):
        """Shell-style patterns, with ** for any depth (:vimgrep)."""
        base = cwd.rstrip("/")
        rx = re.compile("^" + re.escape(pattern).replace(r"\*\*/", "(?:.*/)?").replace(r"\*\*", ".*")
                        .replace(r"\*", "[^/]*").replace(r"\?", "[^/]") + "$")
        out = []

        def walk(path, rel, depth):
            if depth > 12:
                return
            for name, isdir, _ in self.listdir(path):
                if name.startswith("."):
                    continue
                r = f"{rel}/{name}" if rel else name
                full = path.rstrip("/") + "/" + name
                if isdir:
                    walk(full, r, depth + 1)
                elif rx.match(r):
                    out.append(r)
        if pattern.startswith("/"):
            return [pattern] if self.exists(pattern) else []
        walk(base, "", 0)
        return sorted(out)

    def complete(self, cwd, part):
        d, _, stem = part.rpartition("/")
        path = posixpath.normpath(posixpath.join(cwd, d)) if d else cwd
        out = []
        for name, isdir, _ in self.listdir(path):
            if name.startswith(stem) and (not name.startswith(".") or stem.startswith(".")):
                out.append((d + "/" if d else "") + name + ("/" if isdir else ""))
        return sorted(out)

    def _session(self, cwd):
        from ..world import Session
        if self.sess is None or not self.sess.alive:
            self.sess = Session(self.world, cwd=cwd)
        return self.sess

    def run(self, cmd, stdin=None, cwd=None, keep_cwd=False):
        """Run a shell command line on the machine. stdin: text fed to it. -> {stdout, stderr, status, cwd}"""
        k = self.world.kernel
        s = self._session(cwd or self.proc.cwd)
        if cwd and not keep_cwd:
            s.run(f"cd {_q(cwd)}", quiet=True, history=False)
        line = cmd
        if stdin is not None:
            delim = "VIMEOF"
            while delim in stdin:
                delim += "_"
            line = f"{cmd} <<'{delim}'\n{stdin if stdin.endswith(chr(10)) else stdin + chr(10)}{delim}"
        saved = k.budget
        k.budget = k.steps + 3_000_000
        try:
            r = s.run(line, history=False)
        except TimeLimit:
            r = {"stdout": "", "stderr": "", "status": 124, "error": "the command ran too long and was stopped"}
        finally:
            k.budget = saved
        err = r.get("stderr", "")
        if r.get("error"):
            err += r["error"] + "\n"
        return {"stdout": r.get("stdout", ""), "stderr": err, "status": r.get("status", 0),
                "cwd": s.sh.get("PWD", cwd) if s.alive else cwd}

    def prompt(self, cwd):
        s = self._session(cwd)
        ps = s.sh.get("PS1", "") or "$ "
        try:
            return re.sub(r"\x1b\[[0-9;]*m|\x01|\x02", "", s.sh.prompt_expand(ps))
        except Exception:  # noqa: BLE001
            return "$ "


def _q(s):
    return "'" + s.replace("'", "'\\''") + "'"


def parse_args(argv):
    """vim [+N] [+/pat] [-O|-o|-p] [-R] files... -> dict."""
    out = {"files": [], "line": None, "split": None, "readonly": argv[0].rsplit("/", 1)[-1] == "view", "search": None}
    a = argv[1:]
    i = 0
    while i < len(a):
        x = a[i]
        if x == "--":
            out["files"] += a[i + 1:]
            break
        if x.startswith("+"):
            rest = x[1:]
            if rest == "" or rest == "$":
                out["line"] = "$"
            elif rest.isdigit():
                out["line"] = int(rest)
            elif rest.startswith("/"):
                out["search"] = rest[1:]
        elif x in ("-O", "-o", "-p"):
            out["split"] = x[1]
        elif x == "-R":
            out["readonly"] = True
        elif x in ("-u", "-c", "--cmd") and i + 1 < len(a):
            i += 1
        elif x.startswith("-"):
            pass
        else:
            out["files"].append(x)
        i += 1
    return out


@util("vim", "vi", "view", "vim.basic", "vimtutor")
def u_vim(ctx):
    world = ctx.world
    if not getattr(world, "editor_ok", False) or not ctx.stdin_is_tty():
        ctx.e("Vim: Warning: Input is not from a terminal\n" if not ctx.stdin_is_tty() else
              "Vim: this terminal can't show Vim here\n")
        return 1
    from ..vim import Vim
    args = parse_args(ctx.argv)
    if ctx.name == "vimtutor":
        args["files"] = ["tutor"]
    io = VimIO(world, ctx.proc, ctx.env)
    term = getattr(world, "vim_term", {}) or {}
    home = ctx.env.get("HOME", "/home/ada")
    width = getattr(world.tty, "width", 80) or 80
    v = Vim(io, args["files"], cwd=ctx.proc.cwd, home=home, width=width, height=24,
            now=term.get("now", lambda: 0)(), noquit=term.get("noquit", False), line=args["line"],
            split={"O": "O", "o": "o"}.get(args["split"]), readonly=args["readonly"], target=term.get("target"))
    v.dark = term.get("dark", False)
    if args["split"] == "p" and len(args["files"]) > 1:
        for name in args["files"][1:]:
            v.new_tab(v.load_buffer(name))
        v.goto_tab(0)
    if args["search"]:
        from ..vim.cmdline import split_search
        try:
            pos = v.search(split_search(args["search"], "/")[0], 1, 1, (0, 0), show=False)
            v.set_cursor(*pos)
        except Exception:  # noqa: BLE001
            pass
    v.pid = ctx.proc.pid
    if args["line"] is not None:
        v.used["vim +N"] += 1
    if args["split"] == "O":
        v.used["vim -O"] += 1
    if args["readonly"]:
        v.used["view"] += 1
    world.vim = v
    world.vim_last = v                       # what the grader looks at (also after Vim has quit)
    while v.quit is None:
        yield ("forever",)
    if world.vim is v:
        world.vim = None
    return v.quit
