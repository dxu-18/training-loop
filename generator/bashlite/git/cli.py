"""The `git` program: global options (-C, -c, --git-dir, --no-pager, --version), aliases (alias.lg, !shell
aliases), built-in commands (cmd_*.py), external `git-foo` programs on PATH (git-lfs, git-filter-repo), and the
helpers commands share: option parsing with git's messages, pathspecs, colors, the editor and hooks."""

from __future__ import annotations

import fnmatch
import posixpath
import re
import shlex

from ..exec import util
from ..vfs import FSError
from .discover import find_repo
from .objects import GitError

VERSION = "2.43.0"
COMMANDS: dict = {}
NEEDS_WT = set()


def run_maybe(x):
    """`yield from` a command helper whether or not it is a generator (some never wait for input)."""
    if hasattr(x, "send"):
        return (yield from x)
    return x


def command(*names, wt: bool = False):
    def reg(fn):
        for n in names:
            COMMANDS[n] = fn
            if wt:
                NEEDS_WT.add(n)
        return fn
    return reg


class Usage(GitError):
    def __init__(self, msg: str):
        super().__init__(msg, 129)


class Opts:
    """Parsed options: .flags (name -> value or True; repeated values in .lists), .args (positionals),
    .paths (after --), .order (names in the order given)."""

    def __init__(self):
        self.flags: dict = {}
        self.lists: dict = {}
        self.args: list = []
        self.paths: list | None = None
        self.order: list = []

    def __contains__(self, k):
        return k in self.flags

    def get(self, k, default=None):
        return self.flags.get(k, default)

    def all(self, k):
        return self.lists.get(k, [])


def parse(args, spec: str, usage: str = "", stop_at_nonopt: bool = False, numbers: bool = False):
    """spec: space-separated option names; a trailing '=' takes a value, '=?' an optional (--opt[=v]) value, ':'
    marks a short option taking a value ("m: message= amend all|a ..."). '|' joins aliases (first is the key).
    numbers: -5 means -n 5 (log)."""
    keys = {}
    takes = {}
    for item in spec.split():
        names = item.split("|")
        key = names[0].rstrip("=:?")
        # the value marker (on the last alias) applies to every spelling: "message|m=" -> --message and -m
        want = "opt" if item.endswith("=?") else ("req" if item.endswith(("=", ":")) else "no")
        for n in names:
            bare = n.rstrip("=:?")
            keys[bare] = key
            takes[bare] = want
    o = Opts()
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--":
            o.paths = list(args[i + 1:])
            break
        if numbers and re.fullmatch(r"-\d+", a):
            _set(o, "n", a[1:])
            i += 1
            continue
        if a.startswith("--") and len(a) > 2:
            name, eq, val = a[2:].partition("=")
            neg = False
            if name not in keys and name.startswith("no-") and name[3:] in keys:
                neg, name = True, name[3:]
            if name not in keys:
                # unambiguous prefixes are accepted
                cands = [k for k in keys if len(k) > 1 and k.startswith(name)]
                if len(cands) == 1:
                    name = cands[0]
                else:
                    raise Usage(f"error: unknown option `{a[2:].partition('=')[0]}'" + (f"\n{usage}" if usage else ""))
            key = keys[name]
            if neg:
                _set(o, key, False)
            elif takes[name] == "req":
                if eq:
                    _set(o, key, val)
                elif i + 1 < len(args):
                    i += 1
                    _set(o, key, args[i])
                else:
                    raise Usage(f"error: option `{name}' requires a value")
            elif takes[name] == "opt":
                _set(o, key, val if eq else True)
            else:
                if eq:
                    raise Usage(f"error: option `{name}' takes no value")
                _set(o, key, True)
            i += 1
            continue
        if a.startswith("-") and len(a) > 1:
            j = 1
            while j < len(a):
                c = a[j]
                if c not in keys:
                    raise Usage(f"error: unknown switch `{c}'" + (f"\n{usage}" if usage else ""))
                key = keys[c]
                if takes[c] in ("req", "opt"):
                    rest = a[j + 1:]
                    if rest:
                        _set(o, key, rest)
                    elif takes[c] == "req":
                        if i + 1 >= len(args):
                            raise Usage(f"error: switch `{c}' requires a value")
                        i += 1
                        _set(o, key, args[i])
                    else:
                        _set(o, key, True)
                    break
                _set(o, key, True)
                j += 1
            i += 1
            continue
        o.args.append(a)
        i += 1
        if stop_at_nonopt:
            o.args.extend(args[i:])
            break
    return o


def _set(o: Opts, key, val):
    o.flags[key] = val
    o.lists.setdefault(key, []).append(val)
    o.order.append(key)


class G:
    """What a git command works with: the process context, the repository (found on first use), output."""

    def __init__(self, ctx, overrides=()):
        self.ctx = ctx
        self.fs = ctx.fs
        self.world = ctx.world
        ctx.git_overrides = list(overrides)
        self._repo = None
        self.color_cache = None

    @property
    def env(self):
        return self.ctx.env

    @property
    def cwd(self):
        return self.ctx.cwd

    def repo(self, wt: bool = False):
        if self._repo is None:
            self._repo = find_repo(self.ctx, work_tree_needed=wt)
            self._repo.odb.cut = self._repo.shallow()       # a shallow clone's history ends at its boundary
        if wt and self._repo.bare:
            raise GitError("fatal: this operation must be run in a work tree")
        return self._repo

    def maybe_repo(self):
        try:
            return self.repo()
        except GitError:
            return None

    def out(self, text: str):
        self.ctx.w(text)

    def err(self, text: str):
        self.ctx._flush_nb()          # what was printed to stdout comes first
        self.ctx.e(text)

    def color(self, name: str = "ui") -> bool:
        """color.<cmd> / color.ui: auto -> on a terminal."""
        r = self.maybe_repo()
        from .repo import Config
        cfg = r.config if r else Config(self.fs, self.env.get("HOME", "/"), None, self.ctx.git_overrides, self.env)
        v = cfg.get(f"color.{name}") or cfg.get("color.ui") or "auto"
        v = v.lower()
        if v in ("false", "never", "no", "off"):
            return False
        if v in ("always",):
            return True
        return self.ctx.stdout_is_tty() and self.env.get("TERM", "dumb") != "dumb"

    def columns(self) -> int:
        return self.ctx.columns if self.ctx.stdout_is_tty() else 80

    # ---- paths: what the user typed, relative to the current directory
    def prefix(self, repo) -> str:
        """The current directory relative to the top of the working tree ('' at the top)."""
        if repo.bare:
            return ""
        cwd = posixpath.normpath(self.cwd)
        if cwd == repo.worktree:
            return ""
        if cwd.startswith(repo.worktree + "/"):
            return cwd[len(repo.worktree) + 1:]
        return ""

    def to_rel(self, repo, path: str) -> str:
        """A path the user typed -> relative to the top of the repository."""
        if path.startswith("/"):
            ab = posixpath.normpath(path)
        else:
            ab = posixpath.normpath(posixpath.join(self.cwd, path))
        return repo.rel(ab)

    def shown(self, repo, rel: str) -> str:
        """A repository path as git shows it: relative to the current directory."""
        pre = self.prefix(repo)
        if not pre:
            return rel
        return posixpath.relpath(rel or ".", pre) + ("/" if rel.endswith("/") else "")


def pathspec_matcher(g: G, repo, specs):
    """Pathspecs (files, directories, globs where * also crosses /, ':(exclude)x' or ':!x') -> match(rel)."""
    if not specs:
        return None
    inc, exc = [], []
    pre = g.prefix(repo)
    for s in specs:
        neg = False
        if s.startswith(":(exclude)") or s.startswith(":!") or s.startswith(":^"):
            neg = True
            s = s.split(")", 1)[1] if s.startswith(":(") else s[2:]
        top = False
        if s.startswith(":/"):
            top, s = True, s[2:]
        if not top:
            base = posixpath.normpath(posixpath.join(pre, s)) if s not in ("", ".") else (pre or ".")
        else:
            base = s or "."
        if base in (".", ""):
            base = ""
        (exc if neg else inc).append(base)

    def match(rel: str) -> bool:
        def hit(p):
            if p == "":
                return True
            if any(ch in p for ch in "*?["):
                return fnmatch.fnmatchcase(rel, p) or fnmatch.fnmatchcase(rel, p + "/*")
            return rel == p or rel.startswith(p + "/")
        ok = any(hit(p) for p in inc) if inc else True
        return ok and not any(hit(p) for p in exc)
    match.specs = inc
    return match


def launch_editor(g: G, path: str):
    """GIT_EDITOR, core.editor, VISUAL, EDITOR (else vi): run it on `path` and wait (generator) -> status."""
    repo = g.maybe_repo()
    from .repo import Config
    cfg = repo.config if repo else Config(g.fs, g.env.get("HOME", "/"), None, (), g.env)
    editor = g.env.get("GIT_EDITOR") or cfg.get("core.editor") or g.env.get("VISUAL") or g.env.get("EDITOR") or "vi"
    if editor == ":":
        return 0
    g.ctx._flush_nb()
    st = yield from g.ctx.run_shell(f"{editor} {shlex.quote(path)}")
    if st:
        g.err(f"error: There was a problem with the editor '{editor}'.\n")
    return st


def run_hook(g: G, repo, name: str, args=(), stdin: bytes | None = None):
    """Run .git/hooks/<name> (or core.hooksPath) if it's there and executable. -> exit status (0 when absent)."""
    hooks = repo.config.get("core.hooksPath")
    if hooks:
        hooks = hooks if hooks.startswith("/") else posixpath.join(repo.worktree or repo.gitdir, hooks)
    else:
        hooks = repo.path("hooks", True)
    p = f"{hooks}/{name}"
    try:
        node = g.fs.lookup(p)
    except FSError:
        return 0
    if node.type != "f" or not node.mode & 0o111:
        return 0
    yield from g.ctx.flush()
    env = dict(g.env)
    env["GIT_DIR"] = repo.gitdir if not repo.worktree else ".git" if repo.gitdir == repo.worktree + "/.git" else repo.gitdir
    env.setdefault("GIT_INDEX_FILE", repo.path("index"))
    st = yield from g.ctx.run([p, *args], env=env, cwd=repo.worktree or repo.gitdir)
    return st


@util("git")
def u_git(ctx):
    a = list(ctx.argv[1:])
    overrides = []
    while a and a[0].startswith("-"):
        x = a.pop(0)
        if x in ("--version", "-v"):
            ctx.w(f"git version {VERSION}\n")
            return 0
        if x in ("-h", "--help"):
            a = ["help"] + a
            break
        if x == "-C":
            if not a:
                ctx.e("error: no directory given for -C\n")
                return 129
            d = a.pop(0)
            target = d if d.startswith("/") else posixpath.normpath(posixpath.join(ctx.cwd, d))
            try:
                node = ctx.fs.lookup(target)
                if node.type != "d":
                    raise FSError("ENOTDIR")
            except FSError:
                ctx.e(f"fatal: cannot change to '{d}': No such file or directory\n")
                return 128
            ctx.proc.cwd = target
            continue
        if x == "-c":
            if not a:
                ctx.e("error: -c expects a configuration string\n")
                return 129
            k, eq, v = a.pop(0).partition("=")
            overrides.append((k, v if eq else "true"))
            continue
        if x.startswith("--git-dir="):
            ctx.env = dict(ctx.env, GIT_DIR=x.split("=", 1)[1])
            continue
        if x.startswith("--work-tree="):
            ctx.env = dict(ctx.env, GIT_WORK_TREE=x.split("=", 1)[1])
            continue
        if x in ("--no-pager", "-P", "--paginate", "-p", "--no-replace-objects", "--literal-pathspecs", "--bare"):
            continue
        if x == "--exec-path":
            ctx.w("/usr/lib/git-core\n")
            return 0
        ctx.e(f"unknown option: {x}\nusage: git [-v | --version] [-h | --help] [-C <path>] [-c <name>=<value>]\n"
              "           [--exec-path[=<path>]] [--html-path] [--man-path] [--info-path]\n"
              "           [-p | --paginate | -P | --no-pager] [--no-replace-objects] [--bare]\n"
              "           [--git-dir=<path>] [--work-tree=<path>] [--namespace=<name>]\n"
              "           [--super-prefix=<path>] [--config-env=<name>=<envvar>]\n"
              "           <command> [<args>]\n")
        return 129
    if not a:
        from .help import usage_text
        ctx.w(usage_text())
        return 1
    g = G(ctx, overrides)
    name = a[0]
    args = a[1:]
    # aliases (an alias can't override a built-in)
    if name not in COMMANDS:
        from .repo import Config
        repo = g.maybe_repo()
        cfg = repo.config if repo else Config(ctx.fs, ctx.env.get("HOME", "/"), None, overrides, ctx.env)
        alias = cfg.get(f"alias.{name}")
        if alias:
            if alias.startswith("!"):
                cmd = alias[1:]
                return (yield from ctx.run(["sh", "-c", f"{cmd} \"$@\"", cmd] + args,
                                           cwd=repo.worktree if repo and repo.worktree else ctx.cwd))
            parts = shlex.split(alias)
            name, args = parts[0], parts[1:] + args
    fn = COMMANDS.get(name)
    if fn is None:
        # git-<name> on PATH
        from ..exec import find_in_path
        prog = find_in_path(ctx.fs, f"git-{name}", ctx.env.get("PATH", ""), ctx.cwd)
        if prog:
            return (yield from ctx.run([f"git-{name}", *args]))
        from .help import suggest
        ctx.e(f"git: '{name}' is not a git command. See 'git --help'.\n" + suggest(name))
        return 1
    if "-h" in args and name not in ("grep",):
        from .help import short_usage
        ctx.w(short_usage(name))
        return 129
    if "--help" in args:
        from .help import man
        ctx.w(man(name))
        return 0
    from .objects import odb
    try:
        st = fn(g, args)
        if hasattr(st, "send"):
            st = yield from st
        return st if isinstance(st, int) else 0
    except GitError as e:
        ctx._flush_nb()
        if e.msg:
            ctx.e(e.msg if e.msg.endswith("\n") else e.msg + "\n")
        return e.status
    except FSError as e:
        ctx._flush_nb()
        if e.code == "ENOSPC":
            ctx.e("fatal: unable to write file: No space left on device\n")
            return 128
        ctx.e(f"fatal: {e.msg}\n")
        return 128
    except Exception as e:  # noqa: BLE001  (a bug in bash-lite's git: say so instead of failing silently)
        ctx._flush_nb()
        ctx.e(f"fatal: bash-lite git internal error in '{name}': {type(e).__name__}: {e}\n")
        return 128
    finally:
        odb(ctx.world).cut = set()
