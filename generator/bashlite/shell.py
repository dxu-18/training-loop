"""The bash interpreter: variables, expansions, redirections, pipelines, functions, jobs and traps.

Everything that can wait (command substitution, pipelines, `wait`, reading a pipe) is a generator, so
each Shell runs inside a kernel process and the scheduler interleaves it with other processes.
"""

from __future__ import annotations

import re

from . import pattern as P
from .arith import ArithError, Evaluator
from .fileio import anon_file, open_path, read, read_all, readline, write, write_nb
from .kernel import (NS, Interrupted, SIG_DESC, SIGCONT, SIGHUP, SIGINT, SIGKILL, SIGNALS, SIGNUM, SIGTERM, SIGTSTP, OpenFile,
                     Pipe, ProcExit, Signal, TimeLimit)
from .parser import ASSIGN_RE, DECLARATION, Incomplete, ParseError, Parser
from .quoting import xtrace_word
from .syntax import (AndOr, ArithCmd, ArithFor, ArithSub, Assign, Brace, Case, CmdList, CmdSub, Cond, CondAnd,
                     CondBinary, CondNot, CondOr, CondUnary, CondWord, DQ, Esc, For, FuncDef, If, Lit, Param, Pipeline,
                     ProcSub, Select, Simple, SQ, Subshell, While, Word)
from .vfs import FSError

Q, LIT, EXP = 0, 1, 2
BREAK = (None, -1)
NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


# ---------------------------------------------------------------------------- control flow

class ShellAbort(BaseException):
    """An error (already reported) that abandons the current command line; fatal ones end a script."""

    def __init__(self, status: int = 1, fatal: bool = False):
        super().__init__(status)
        self.status, self.fatal = status, fatal


class BreakLoop(BaseException):
    def __init__(self, n):
        super().__init__(n)
        self.n = n


class ContinueLoop(BaseException):
    def __init__(self, n):
        super().__init__(n)
        self.n = n


class ReturnFunc(BaseException):
    def __init__(self, status):
        super().__init__(status)
        self.status = status


class ExitShell(BaseException):
    def __init__(self, status):
        super().__init__(status)
        self.status = status


# ---------------------------------------------------------------------------- variables

class Var:
    __slots__ = ("value", "attrs")

    def __init__(self, value=None, attrs=""):
        self.value = value      # str | dict (indexed: int keys, assoc: str keys) | None (declared, unset)
        self.attrs = set(attrs)

    def copy(self):
        v = Var(dict(self.value) if isinstance(self.value, dict) else self.value)
        v.attrs = set(self.attrs)
        return v

    @property
    def is_array(self):
        return "a" in self.attrs or "A" in self.attrs


def fnv_bucket(key: str, buckets: int = 1024) -> int:
    i = 2166136261
    for c in key.encode("utf-8", "surrogateescape"):
        i = (i * 16777619) & 0xFFFFFFFF
        i ^= c
    return i & (buckets - 1)


def assoc_keys(d: dict) -> list[str]:
    """Bash's iteration order for associative arrays: by hash bucket, newest first within a bucket."""
    order = {k: i for i, k in enumerate(d)}
    return sorted(d, key=lambda k: (fnv_bucket(k), -order[k]))


class Job:
    def __init__(self, num, procs, text):
        self.num, self.procs, self.text = num, procs, text
        self.notified = False
        self.report_at = None            # prompt number at which "Done"/"Terminated" is reported
        self.died_in_wait = False
        self.died_line = None
        self.reported_state = "Running"
        self.disowned = False
        self.nohup = False
        self.fg_seen = False

    @property
    def pgid(self):
        return self.procs[0].pid

    def state(self) -> str:
        if all(p.state == "Z" for p in self.procs):
            return "Done"
        if any(p.state == "T" for p in self.procs) and all(p.state in ("T", "Z") for p in self.procs):
            return "Stopped"
        return "Running"

    def status_text(self) -> str:
        st = self.state()
        if st == "Done":
            last = self.procs[-1]
            if last.term_sig:
                return SIG_DESC.get(last.term_sig, f"Signal {last.term_sig}")
            return "Done" if last.status == 0 else f"Exit {last.status}"
        if st == "Stopped":
            sig = next((p.term_sig for p in self.procs if p.state == "T"), 20)
            return SIG_DESC.get(sig, "Stopped")
        return "Running"


class Shell:
    """One bash process's state (a subshell is a copy)."""

    def __init__(self, world, proc, interactive=False, name="bash"):
        self.world = world
        self.kernel = world.kernel
        self.proc = proc
        self.host = proc.host
        self.fs = proc.host.fs
        self.globals: dict[str, Var] = {}
        self.scopes: list[dict[str, Var]] = []
        self.funcs: dict[str, FuncDef] = {}
        self.aliases: dict[str, str] = {}
        self.opts: set[str] = {"braceexpand", "hashall", "interactive-comments"}
        self.shopts: set[str] = {"cmdhist", "expand_aliases" if interactive else "sourcepath", "sourcepath",
                                 "extquote", "progcomp", "promptvars", "patsub_replacement", "checkwinsize",
                                 "globasciiranges", "globskipdots", "complete_fullquote", "force_fignore"}
        if interactive:
            self.opts |= {"histexpand", "history", "monitor", "emacs"}
        self.positional: list[str] = []
        self.arg0 = name
        self.status = 0
        self.last_bg = None
        self.last_arg = ""
        self.traps: dict[str, str] = {}
        self.ignored_sigs: set[int] = set()
        self.pending_traps: list[int] = []
        self.jobs: dict[int, Job] = {}
        self.prompt_no = 0                 # primary prompts shown so far (job notices are timed by them)
        self.typed_log: list[str] = []
        self.awaiting = None               # "PS1"/"PS2" while a persistent REPL waits for a line
        self.cmd_hook = None               # live terminal: called with ("start", raw line) / ("line", ..) / ("end", status)
        self.interrupted = False           # Ctrl-C arrived while a command ran (interactive shell)
        self.cancel_pending = False        # live terminal: Ctrl-C at the PS2 prompt
        self._in_wait_builtin = False
        self.job_order: list[int] = []     # most recent last (+ = current, - = previous)
        self.dirstack: list[str] = []
        self.history: list[str] = []
        self.hist_base = 1
        self.interactive = interactive
        self.script = None          # script name for error messages ("run.sh"), None = interactive / -c
        self.c_mode = False
        self.lineno = 0
        self.funcnames: list[str] = []
        self.sources: list[str] = []
        self.loop_depth = 0
        self.subshell = 0
        self.no_errexit = 0
        self.in_trap = False
        self.cmdsub_status = None
        self.getopts_pos = 0
        self.hashed: dict[str, str] = {}
        self.toplevel_pid = proc.pid
        self.rand_seed = (world.seed * 2654435761 + proc.pid) & 0x7FFFFFFF or 1
        self.last_rand = 0
        self.seconds_base = self.kernel.now
        self.pipestatus = [0]
        self.dir_logical = "/"
        self.exec_ok = False        # the next simple external command may replace this process
        self.trace_depth = 0
        self.pending_heredoc = None
        self.on_command = None      # hook for keys mode
        proc.shell = self
        proc.trapper = self._trap_signal

    # ------------------------------------------------------------ process helpers

    def fork(self, proc) -> "Shell":
        sub = Shell.__new__(Shell)
        sub.__dict__.update(self.__dict__)
        sub.globals = {k: v.copy() for k, v in self.globals.items()}
        sub.scopes = [{k: v.copy() for k, v in s.items()} for s in self.scopes]
        sub.funcs = dict(self.funcs)
        sub.aliases = dict(self.aliases)
        sub.opts = set(self.opts) - {"monitor"}
        sub.shopts = set(self.shopts)
        sub.positional = list(self.positional)
        sub.traps = {k: v for k, v in self.traps.items() if v == ""}   # ignored signals stay ignored
        sub.ignored_sigs = set(self.ignored_sigs)
        sub.pending_traps = []
        sub.jobs, sub.job_order = {}, []
        sub.dirstack = list(self.dirstack)
        sub.history = list(self.history)
        sub.funcnames = list(self.funcnames)
        sub.sources = list(self.sources)
        sub.interactive = False
        sub.subshell = self.subshell + 1
        sub.proc = proc
        sub.exec_ok = False
        sub.pipestatus = list(self.pipestatus)
        proc.shell = sub
        proc.trapper = sub._trap_signal
        proc.ignored = set(self.ignored_sigs)
        return sub

    def spawn_subshell(self, body, fds=None, argv=None, extra=None) -> "object":
        """Fork a subshell process running `body` (a node or a callable(shell) -> generator)."""
        k = self.kernel
        child = k.spawn(self.host, argv or list(self.proc.argv), None, self.proc, fds=fds)
        sub = self.fork(child)
        if extra:
            extra(sub)
        k.start(child, sub._subshell_main(body))
        return child

    def _subshell_main(self, body):
        status = 0
        try:
            if callable(body):
                status = yield from body(self)
            else:
                self.exec_ok = True
                status = yield from self.run_node(body)
        except ExitShell as e:
            status = e.status
        except ShellAbort as e:
            status = e.status
        except (BreakLoop, ContinueLoop):
            status = 0
        except ReturnFunc as e:
            status = e.status
        status = yield from self.run_exit_trap(status)
        return status

    def wait_for(self, procs, interruptible=False, fg=True):
        """Wait until every process exits (or one stops). Returns True if a trap interrupted the wait."""
        while True:
            if self.interrupted:
                self.interrupted = False
                raise Interrupted()
            if all(p.state == "Z" for p in procs):
                return False
            if fg and any(p.state == "T" for p in procs) and all(p.state in ("T", "Z") for p in procs):
                return False
            if interruptible and self.pending_traps:
                return True
            yield ("wait", procs)

    # ------------------------------------------------------------ output

    def out(self, text: str):
        yield from write(self.kernel, self.proc.fds.get(1), text.encode("utf-8", "surrogateescape"))

    def err(self, text: str):
        of = self.proc.fds.get(2)
        if of is None:
            return
        try:
            if not write_nb(self.kernel, of, text.encode("utf-8", "surrogateescape")):
                yield from write(self.kernel, of, text.encode())
        except FSError:
            pass

    def prefix(self) -> str:
        if self.interactive:
            return "bash: "
        if self.script is not None:
            return f"{self.script}: line {self.lineno}: "
        return f"{self.arg0 if self.c_mode else 'bash'}: line {self.lineno}: "

    def error(self, msg: str):
        yield from self.err(self.prefix() + msg + "\n")

    def error_now(self, msg: str):
        """Report an error without yielding (stderr never blocks)."""
        of = self.proc.fds.get(2)
        if of is not None:
            try:
                write_nb(self.kernel, of, (self.prefix() + msg + "\n").encode("utf-8", "surrogateescape"))
            except (FSError, Signal):
                pass

    # ------------------------------------------------------------ variables

    def lookup(self, name: str) -> Var | None:
        for scope in reversed(self.scopes):
            v = scope.get(name)
            if v is not None:
                return v
        return self.globals.get(name)

    def special(self, name: str):
        """Dynamic variables; None if `name` isn't one."""
        if name == "RANDOM":
            v = self.lookup("RANDOM")
            if v is not None and "_seeded" in v.attrs:
                pass
            return str(self.next_random())
        if name == "SECONDS":
            return str((self.kernel.now - self.seconds_base) // NS)
        if name == "EPOCHSECONDS":
            return str(self.kernel.now // NS)
        if name == "EPOCHREALTIME":
            return f"{self.kernel.now // NS}.{(self.kernel.now % NS) // 1000:06d}"
        if name == "LINENO":
            return str(self.lineno)
        if name == "BASHPID":
            return str(self.proc.pid)
        if name == "PPID":
            p = self.kernel.world.toplevel_ppid.get(self.toplevel_pid)
            return str(p if p is not None else self.proc.ppid)
        if name == "BASH_SUBSHELL":
            return str(self.subshell)
        if name == "UID" or name == "EUID":
            return str(self.proc.uid)
        if name == "HOSTNAME":
            return self.host.name
        return None

    def next_random(self) -> int:
        while True:
            ret = self.rand_seed or 123459876
            h = ret // 127773
            l = ret - 127773 * h
            t = 16807 * l - 2836 * h
            self.rand_seed = t + 0x7FFFFFFF if t < 0 else t
            rv = self.rand_seed & 32767
            if rv != self.last_rand:
                self.last_rand = rv
                return rv

    def get(self, name: str, default=None):
        """Scalar value of a variable (element 0 of an array), or default if unset."""
        sp = self.special(name)
        if sp is not None:
            return sp
        v = self.lookup(name)
        if v is None or v.value is None:
            return default
        if isinstance(v.value, dict):
            if "A" in v.attrs:
                return v.value.get("0", default)
            return v.value.get(0, default)
        return v.value

    def is_set(self, name: str) -> bool:
        if self.special(name) is not None:
            return True
        v = self.lookup(name)
        return v is not None and v.value is not None and (not isinstance(v.value, dict) or bool(v.value) or True)

    def _container(self, name, local=False):
        if local and self.scopes:
            return self.scopes[-1]
        for scope in reversed(self.scopes):
            if name in scope:
                return scope
        return self.globals

    def set_var(self, name: str, value, index=None, append=False, local=False, attrs=None, check=True):
        """Assign a scalar (index None) or one array element. Raises ShellAbort on readonly."""
        if name == "RANDOM" and index is None:
            try:
                self.rand_seed = int(value) & 0xFFFFFFFF
            except ValueError:
                self.rand_seed = 0
            self.last_rand = 0
            return
        if name == "SECONDS" and index is None:
            try:
                self.seconds_base = self.kernel.now - int(value) * NS
            except ValueError:
                pass
            return
        cont = self._container(name, local)
        var = cont.get(name)
        if var is None:
            var = Var()
            cont[name] = var
            if "allexport" in self.opts:
                var.attrs.add("x")
        if attrs:
            var.attrs |= set(attrs)
        if "r" in var.attrs and check:
            self.error_now(f"{name}: readonly variable")
            raise ShellAbort(1)
        if index is not None or var.is_array:
            if not isinstance(var.value, dict):
                old = var.value
                var.value = {} if old is None else ({0: old} if "A" not in var.attrs else {"0": old})
                if "A" not in var.attrs:
                    var.attrs.add("a")
            arr = var.value
            if "A" in var.attrs:
                key = "0" if index is None else str(index)
            else:
                if index is None:
                    key = 0
                else:
                    key = int(index)
                    if key < 0:
                        top = (max(arr) + 1) if arr else 0
                        if top + key < 0:
                            self.error_now(f"{name}[{key}]: bad array subscript")
                            raise ShellAbort(1)
                        key = top + key
            value = self._convert(var, value, arr.get(key) if append else None, append)
            arr[key] = value
        else:
            value = self._convert(var, value, var.value if append else None, append)
            var.value = value
        if name == "PWD":
            self.dir_logical = var.value if isinstance(var.value, str) else self.dir_logical
        elif name == "PATH":
            self.hashed.clear()

    def _convert(self, var, value, old, append):
        if "i" in var.attrs:
            n = self.arith(value)
            if append and old not in (None, ""):
                n = self.arith(old) + n
            value = str(n)
        elif append and old is not None:
            value = old + value
        if "l" in var.attrs:
            value = value.lower()
        elif "u" in var.attrs:
            value = value.upper()
        return value

    def unset_var(self, name: str, index=None):
        cont = self._container(name)
        var = cont.get(name)
        if var is None:
            if name in self.funcs and index is None:
                pass
            return True
        if "r" in var.attrs:
            self.error_now(f"unset: {name}: cannot unset: readonly variable")
            return False
        if index is not None and isinstance(var.value, dict):
            key = str(index) if "A" in var.attrs else int(index)
            if isinstance(key, int) and key < 0 and var.value:
                key = max(var.value) + 1 + key
            var.value.pop(key, None)
            return True
        if cont is self.globals or not self.scopes:
            del cont[name]
        else:
            # unsetting a local exposes nothing (bash marks it unset in that scope)
            var.value = None
            var.attrs.discard("x")
        return True

    def array_items(self, name: str):
        """(keys, values) of an array (a scalar is a one-element array)."""
        v = self.lookup(name)
        if name == "@" or name == "*":
            return list(range(1, len(self.positional) + 1)), list(self.positional)
        if v is None or v.value is None:
            if name in ("BASH_REMATCH", "PIPESTATUS", "FUNCNAME", "BASH_SOURCE", "DIRSTACK", "GROUPS", "BASH_VERSINFO"):
                return self._special_array(name)
            return [], []
        if isinstance(v.value, dict):
            if "A" in v.attrs:
                ks = assoc_keys(v.value)
                return ks, [v.value[k] for k in ks]
            ks = sorted(v.value)
            return ks, [v.value[k] for k in ks]
        return [0], [v.value]

    def _special_array(self, name):
        if name == "PIPESTATUS":
            vals = [str(s) for s in self.pipestatus]
        elif name == "FUNCNAME":
            vals = list(reversed(self.funcnames)) + (["main"] if self.funcnames else [])
        elif name == "BASH_SOURCE":
            vals = list(reversed(self.sources))
        elif name == "DIRSTACK":
            vals = [self.tilde_short(self.dir_logical)] + [self.tilde_short(d) for d in reversed(self.dirstack)]
        elif name == "GROUPS":
            vals = [str(g) for g in sorted(self.proc.gids)]
        elif name == "BASH_VERSINFO":
            vals = ["5", "2", "21", "1", "release", "x86_64-pc-linux-gnu"]
        else:
            vals = []
        return list(range(len(vals))), vals

    def is_assoc(self, name: str) -> bool:
        v = self.lookup(name)
        return v is not None and "A" in v.attrs

    def env(self) -> dict[str, str]:
        out = {}
        for scope in [self.globals] + self.scopes:
            for k, v in scope.items():
                if "x" in v.attrs and isinstance(v.value, str):
                    out[k] = v.value
                elif "x" in v.attrs and v.value is None:
                    out.pop(k, None)
        return out

    def tilde_short(self, path: str) -> str:
        home = self.get("HOME", "")
        if home and home != "/" and (path == home or path.startswith(home + "/")):
            return "~" + path[len(home):]
        return path

    # ------------------------------------------------------------ arithmetic

    def arith(self, text: str) -> int:
        def get(name, index):
            if index is None:
                sp = self.special(name)
                if sp is not None:
                    return sp
                v = self.lookup(name)
                if v is None or v.value is None:
                    if "nounset" in self.opts:
                        self.error_now(f"{name}: unbound variable")
                        raise ShellAbort(1, fatal=True)
                    return None
                if isinstance(v.value, dict):
                    return v.value.get("0" if "A" in v.attrs else 0)
                return v.value
            v = self.lookup(name)
            if v is None or not isinstance(v.value, dict):
                if index in (0, "0") and v is not None:
                    return v.value
                return None
            if "A" in v.attrs:
                return v.value.get(str(index))
            idx = int(index)
            if idx < 0 and v.value:
                idx = max(v.value) + 1 + idx
            return v.value.get(idx)

        def set_(name, index, val):
            self.set_var(name, str(val), index)
        ev = Evaluator(get, set_, 0, self.is_assoc)
        try:
            return ev.eval(text)
        except ArithError as e:
            tok = e.token
            shown = text.lstrip()
            self.error_now(f"{self._arith_ctx}{shown}: {e.msg} (error token is \"{tok}\")")
            raise ShellAbort(1, fatal=(not self._arith_cmd) and (self.c_mode or self.interactive and False)) from None

    _arith_ctx = ""
    _arith_cmd = False

    # ------------------------------------------------------------ expansion: words -> fields

    def expand_words(self, words, declaration=False):
        out = []
        for w in words:
            if declaration and w.array is not None:
                out.append(w)            # name=(...) passed through to the builtin
                continue
            if declaration and ASSIGN_RE.match(w.raw) and isinstance(w.parts[0], Lit) and "=" in w.parts[0].text:
                s = yield from self.expand_assignment_word(w)
                out.append(s)
                continue
            fields = yield from self.expand_word(w)
            out.extend(fields)
        return out

    def expand_word(self, w: Word, glob=True, split=True):
        """Full expansion of one word into fields."""
        parts = w.parts
        if len(parts) == 1 and type(parts[0]) is SQ:
            return [parts[0].text]
        if all(type(p) is Lit for p in parts):
            text = "".join(p.text for p in parts)
            if not any(c in text for c in "{~*?[") or (not glob and not any(c in text for c in "{~")):
                return [text]
        alts = self.brace_expand(parts) if "braceexpand" in self.opts else [parts]
        result = []
        for alt in alts:
            chunks = []
            yield from self._expand_parts(self._tilde(alt), False, chunks)
            fields = self._split(chunks) if split else [self._join_fields(chunks)]
            for f in fields:
                if glob and "noglob" not in self.opts:
                    result.extend(self._glob_field(f))
                else:
                    result.append("".join(t for t, _ in f))
        return result

    @staticmethod
    def _join_fields(chunks):
        out = []
        for c in chunks:
            if c is BREAK:
                out.append((" ", Q))
            else:
                out.append(c)
        return out

    def expand_string(self, w: Word | None, quoted_ctx=False):
        """Expansion without splitting or globbing (assignments, redirect targets, here-strings, case words)."""
        if w is None:
            return ""
        parts = w.parts
        if all(type(p) in (Lit, SQ, Esc) for p in parts) and not (parts and type(parts[0]) is Lit and parts[0].text.startswith("~")):
            return "".join(p.text for p in parts)
        chunks = []
        yield from self._expand_parts(self._tilde(parts), quoted_ctx, chunks)
        return "".join(" " if c is BREAK else c[0] for c in chunks)

    def expand_assignment_word(self, w: Word):
        """`name=value` argument of declare/export/local: tilde after = and :, no splitting/globbing."""
        chunks = []
        yield from self._expand_parts(self._tilde(w.parts, assign=True), False, chunks)
        return "".join(" " if c is BREAK else c[0] for c in chunks)

    def expand_pattern(self, w: Word | None):
        """A pattern word: quoted characters are escaped so they match literally."""
        if w is None:
            return ""
        chunks = []
        yield from self._expand_parts(self._tilde(w.parts), False, chunks)
        out = []
        for c in chunks:
            if c is BREAK:
                out.append(" ")
                continue
            t, f = c
            out.append(P.escape(t) if f == Q else t)
        return "".join(out)

    def _tilde(self, parts, assign=False):
        if not parts or type(parts[0]) is not Lit:
            return parts
        first = parts[0].text
        if not first.startswith("~") and not (assign and ("=~" in first or ":~" in first)):
            return parts
        if assign:
            # expand ~ after the first = and after each :
            segs = re.split(r"([=:])", first)
            out, eq_seen = [], False
            for s in segs:
                if s in ("=", ":"):
                    out.append(s)
                    if s == "=":
                        eq_seen = True
                    continue
                if s.startswith("~") and (not out or out[-1] in ("=", ":")) and (eq_seen or not out):
                    out.append(self._tilde_text(s))
                else:
                    out.append(s)
            return [Lit("".join(out))] + list(parts[1:])
        name_end = first.find("/")
        prefix = first if name_end < 0 else first[:name_end]
        if name_end < 0 and len(parts) > 1:
            return parts   # ~"user" etc.: no expansion
        exp = self._tilde_word(prefix)
        if exp is None:
            return parts
        rest = first[len(prefix):]
        new = [SQ(exp)]
        if rest:
            new.append(Lit(rest))
        return new + list(parts[1:])

    def _tilde_text(self, s: str) -> str:
        j = s.find("/")
        prefix = s if j < 0 else s[:j]
        exp = self._tilde_word(prefix)
        return s if exp is None else exp + s[len(prefix):]

    def _tilde_word(self, prefix: str):
        if prefix == "~":
            return self.get("HOME", self.host.home_of(self.proc.uid))
        if prefix == "~+":
            return self.get("PWD", self.proc.cwd)
        if prefix == "~-":
            return self.get("OLDPWD")
        user = prefix[1:]
        return self.host.user_home(user)

    # -------- brace expansion

    def brace_expand(self, parts):
        items = []
        for p in parts:
            if type(p) is Lit:
                items.extend(("c", ch) for ch in p.text)
            else:
                items.append(("p", p))
        if not any(it == ("c", "{") for it in items):
            return [parts]
        return [self._items_to_parts(x) for x in self._brace_items(items)]

    def _brace_items(self, items):
        n = len(items)
        i = 0
        while i < n:
            if items[i] == ("c", "{"):
                depth, j, commas = 0, i, []
                while j < n:
                    it = items[j]
                    if it == ("c", "{"):
                        depth += 1
                    elif it == ("c", "}"):
                        depth -= 1
                        if depth == 0:
                            break
                    elif it == ("c", ",") and depth == 1:
                        commas.append(j)
                    j += 1
                if j < n:
                    pre, post = items[:i], items[j + 1:]
                    if commas:
                        bounds = [i] + commas + [j]
                        res = []
                        for a, b in zip(bounds, bounds[1:]):
                            for alt in self._brace_items(items[a + 1:b]):
                                res.extend(self._brace_items(pre + alt + post))
                        return res
                    inner = items[i + 1:j]
                    if all(t == "c" for t, _ in inner):
                        seq = _brace_sequence("".join(c for _, c in inner))
                        if seq is not None:
                            res = []
                            for s in seq:
                                res.extend(self._brace_items(pre + [("c", ch) for ch in s] + post))
                            return res
                    # not a brace expression: keep "{" literally and look further
            i += 1
        return [items]

    @staticmethod
    def _items_to_parts(items):
        parts, buf = [], []
        for t, v in items:
            if t == "c":
                buf.append(v)
            else:
                if buf:
                    parts.append(Lit("".join(buf)))
                    buf = []
                parts.append(v)
        if buf:
            parts.append(Lit("".join(buf)))
        return parts

    # -------- parts -> chunks

    def _expand_parts(self, parts, quoted, out):
        for p in parts:
            t = type(p)
            if t is Lit:
                out.append((p.text, Q if quoted else LIT))
            elif t is SQ or t is Esc:
                out.append((p.text, Q))
            elif t is DQ:
                sub = []
                yield from self._expand_parts(p.parts, True, sub)
                if not sub and any(type(q) is Param and self._is_at(q) for q in p.parts):
                    continue
                out.append(("", Q))
                out.extend(sub)
            elif t is Param:
                yield from self._expand_param(p, quoted, out)
            elif t is CmdSub:
                s = yield from self.command_subst(p)
                out.append((s, Q if quoted else EXP))
            elif t is ArithSub:
                text = yield from self.expand_string(p.expr, True)
                v = self.arith(text)
                out.append((str(v), Q if quoted else EXP))
            elif t is ProcSub:
                path = self.proc_subst(p)
                out.append((path, Q))
            else:
                raise RuntimeError(f"cannot expand {p!r}")

    @staticmethod
    def _is_at(p: Param) -> bool:
        if p.op in ("len",) or p.indirect and p.op not in ("!@",):
            return False
        if p.name == "@" and p.sub is None:
            return True
        if p.sub is not None and p.sub.raw == "@":
            return True
        return p.op == "!@" or (p.indirect and p.sub is not None and p.sub.raw == "@")

    # -------- split & glob

    def _split(self, chunks):
        ifs = self.get("IFS")
        if ifs is None:
            ifs = " \t\n"
        ws = set(c for c in ifs if c in " \t\n")
        other = set(c for c in ifs if c not in " \t\n")
        fields, cur, started, ws_pending = [], [], False, False
        for c in chunks:
            if c is BREAK:
                if started:
                    fields.append(cur)
                cur, started, ws_pending = [], False, False
                continue
            text, flag = c
            if flag != EXP or not ifs:
                cur.append(c)
                if flag == Q or text:
                    started = True
                    ws_pending = False
                continue
            buf = []
            for ch in text:
                if ch in ws:
                    if buf:
                        cur.append(("".join(buf), EXP))
                        buf = []
                    if started:
                        fields.append(cur)
                        cur, started = [], False
                        ws_pending = True
                elif ch in other:
                    if buf:
                        cur.append(("".join(buf), EXP))
                        buf = []
                    if started:
                        fields.append(cur)
                        cur, started = [], False
                    elif not ws_pending:
                        fields.append([])
                    ws_pending = False
                else:
                    buf.append(ch)
                    started = True
                    ws_pending = False
            if buf:
                cur.append(("".join(buf), EXP))
        if started:
            fields.append(cur)
        return fields

    def _glob_field(self, field):
        if not any(f != Q and any(ch in t for ch in "*?[") for t, f in field) and not (
                "extglob" in self.shopts and any(f != Q and "(" in t for t, f in field)):
            return ["".join(t for t, _ in field)]
        pat = "".join(P.escape(t) if f == Q else t for t, f in field)
        if not P.has_magic(pat, "extglob" in self.shopts):
            return [P.unescape(pat)]
        matches = P.glob(self.fs, self.proc.cwd, pat, "dotglob" in self.shopts, "globstar" in self.shopts,
                         "nocaseglob" in self.shopts, "extglob" in self.shopts)
        if matches:
            return matches
        if "failglob" in self.shopts:
            self.error_now(f"no match: {P.unescape(pat)}")
            raise ShellAbort(1)
        if "nullglob" in self.shopts:
            return []
        return ["".join(t for t, _ in field)]

    # -------- parameters

    def _param_values(self, p: Param):
        """-> (values: list[str] | None, is_list: bool, star: bool, name_for_errors)."""
        name = p.name
        if p.indirect and p.op not in ("!*", "!@"):
            if p.sub is not None and p.sub.raw in ("@", "*"):
                ks, _ = self.array_items(name)
                return [str(k) for k in ks], True, p.sub.raw == "*", name
            target = self.get(name) if not name.isdigit() else self._positional(int(name))
            if target is None:
                if "nounset" in self.opts:
                    self.error_now(f"{name}: unbound variable")
                    raise ShellAbort(self.fatal_status(), fatal=True)
                return None, False, False, name
            m = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\[(.*)\]", target)
            if m:
                fake = Param(m.group(1), Word([Lit(m.group(2))], m.group(2)), "")
                return (yield from self._param_values(fake))
            if not NAME.match(target) and not target.isdigit() and target not in "@*#?-$!":
                self.error_now(f"{target}: invalid variable name")
                raise ShellAbort(1, fatal=True)
            return (yield from self._param_values(Param(target)))
        if p.op in ("!*", "!@"):
            names = sorted(set(k for k in list(self.globals) + [k for s in self.scopes for k in s]
                               if k.startswith(name) and self.lookup(k) is not None and self.lookup(k).value is not None))
            return names, True, p.op == "!*", name
        if p.sub is not None:
            sub = p.sub.raw
            if sub in ("@", "*"):
                if name in ("@", "*"):
                    return list(self.positional), True, sub == "*", name
                ks, vals = self.array_items(name)
                return vals, True, sub == "*", name
            v = self.lookup(name)
            label = f"{name}[{sub}]"
            if v is not None and "A" in v.attrs:
                key = yield from self.expand_string(p.sub, True)
                return [v.value.get(key) if isinstance(v.value, dict) else None], False, False, label
            idx_text = yield from self.expand_string(p.sub, True)
            idx = self.arith(idx_text) if idx_text.strip() else 0
            if v is None or v.value is None:
                ks, vals = self._special_array(name)
                d = dict(zip(ks, vals))
            elif isinstance(v.value, dict):
                d = v.value
            else:
                return [v.value if idx in (0, -1) else None], False, False, label
            if idx < 0:
                top = (max(d) + 1) if d else 0
                if top + idx < 0:
                    self.error_now(f"{name}: bad array subscript")
                    raise ShellAbort(1)
                idx = top + idx
            return [d.get(idx)], False, False, label
        if name in ("@", "*"):
            return list(self.positional), True, name == "*", name
        if name == "#":
            return [str(len(self.positional))], False, False, name
        if name == "?":
            return [str(self.status)], False, False, name
        if name == "$":
            return [str(self.toplevel_pid)], False, False, name
        if name == "!":
            return [str(self.last_bg) if self.last_bg is not None else None], False, False, name
        if name == "-":
            return [self.flags()], False, False, name
        if name == "_":
            return [self.last_arg], False, False, name
        if name.isdigit():
            return [self._positional(int(name))], False, False, name
        return [self.get(name)], False, False, name

    def _positional(self, n: int):
        if n == 0:
            return self.arg0
        return self.positional[n - 1] if n <= len(self.positional) else None

    FLAG_LETTERS = (("allexport", "a"), ("notify", "b"), ("errexit", "e"), ("noglob", "f"), ("hashall", "h"),
                    ("interactive", "i"), ("keyword", "k"), ("monitor", "m"), ("noexec", "n"), ("privileged", "p"),
                    ("onecmd", "t"), ("nounset", "u"), ("verbose", "v"), ("xtrace", "x"), ("braceexpand", "B"),
                    ("noclobber", "C"), ("errtrace", "E"), ("histexpand", "H"), ("physical", "P"), ("functrace", "T"))

    def fatal_status(self) -> int:
        """Exit status after a fatal expansion error: 127 for bash -c, else 1."""
        return 127 if self.c_mode else 1

    def flags(self) -> str:
        f = "".join(ch for opt, ch in self.FLAG_LETTERS if opt in self.opts or (opt == "interactive" and self.interactive))
        if self.interactive:
            f += "s"
        if self.c_mode:
            f += "c"
        return f

    def _expand_param(self, p: Param, quoted, out):
        op = p.op
        if op == "len":
            if p.name in ("@", "*") and p.sub is None:
                out.append((str(len(self.positional)), Q if quoted else EXP))
                return
            if p.sub is not None and p.sub.raw in ("@", "*"):
                _, vals = self.array_items(p.name)
                out.append((str(len(vals)), Q if quoted else EXP))
                return
            vals, _, _, nm = yield from self._param_values(Param(p.name, p.sub))
            v = vals[0] if vals else None
            if v is None and "nounset" in self.opts:
                self.error_now(f"{nm}: unbound variable")
                raise ShellAbort(self.fatal_status(), fatal=True)
            out.append((str(len(v or "")), Q if quoted else EXP))
            return
        vals, is_list, star, nm = yield from self._param_values(p)
        unset = vals is None or (not is_list and vals[0] is None)
        if is_list:
            unset = False
        if op in (":-", "-", ":=", "=", ":?", "?", ":+", "+"):
            if is_list:
                null = not vals or (len(vals) == 1 and vals[0] == "")
                use_default = null if op.startswith(":") else not vals
            else:
                v = vals[0] if vals else None
                use_default = v is None or (op.startswith(":") and v == "")
            if op in (":+", "+"):
                if not use_default:
                    yield from self._expand_parts(self._tilde(p.arg.parts), quoted, out)
                return
            if not use_default:
                self._emit_values(vals, is_list, star, quoted, out)
                return
            if op in (":-", "-"):
                yield from self._expand_parts(self._tilde(p.arg.parts), quoted, out)
                return
            if op in (":=", "="):
                if p.name.isdigit() or p.name in "@*#?-$!":
                    self.error_now(f"${p.name}: cannot assign in this way")
                    raise ShellAbort(1, fatal=True)
                val = yield from self.expand_string(p.arg, quoted)
                if p.sub is not None:
                    idx = yield from self.expand_string(p.sub, True)
                    self.set_var(p.name, val, idx if self.is_assoc(p.name) else self.arith(idx))
                else:
                    self.set_var(p.name, val)
                out.append((val, Q if quoted else EXP))
                return
            # :? and ?
            msg = yield from self.expand_string(p.arg, True)
            self.error_now(f"{nm}: {msg if msg else 'parameter null or not set'}")
            raise ShellAbort(self.fatal_status(), fatal=True)
        if unset and "nounset" in self.opts and not (is_list and p.name in ("@", "*")):
            # bash names an unbraced positional parameter with its dollar: "$1: unbound variable"
            self.error_now(f"{'$' if nm.isdigit() and not p.braced else ''}{nm}: unbound variable")
            raise ShellAbort(self.fatal_status(), fatal=True)
        if op == "":
            self._emit_values(vals, is_list, star, quoted, out)
            return
        if vals is None:
            vals = [None]
        src = [v for v in vals if v is not None] if is_list else [vals[0] or ""]
        if op in ("#", "##", "%", "%%"):
            pat = yield from self.expand_pattern(p.arg)
            nocase = "nocasematch" in self.shopts and False
            if op[0] == "#":
                res = [P.strip_prefix(s, pat, op == "##", nocase) for s in src]
            else:
                res = [P.strip_suffix(s, pat, op == "%%", nocase) for s in src]
        elif op in ("/", "//", "/#", "/%"):
            pat = yield from self.expand_pattern(p.arg)
            rep_chunks = []
            if p.arg2 is not None:
                yield from self._expand_parts(self._tilde(p.arg2.parts), quoted, rep_chunks)
            res = []
            for s in src:
                res.append(self._patsub(s, pat, rep_chunks, op))
        elif op in ("^", "^^", ",", ",,"):
            pat = (yield from self.expand_pattern(p.arg)) if p.arg is not None and p.arg.parts else "?"
            res = [self._case_mod(s, op, pat) for s in src]
        elif op == ":":
            off_s = yield from self.expand_string(p.arg, True)
            off = self.arith(off_s) if off_s.strip() else 0
            ln = None
            if p.arg2 is not None:
                ln_s = yield from self.expand_string(p.arg2, True)
                ln = self.arith(ln_s) if ln_s.strip() else 0
            if is_list:
                if p.name in ("@", "*") and p.sub is None:
                    seq = [self.arg0] + list(self.positional)
                    if off < 0:
                        off = len(seq) + off
                        if off < 1:
                            off = 0
                    res = seq[off:] if off >= 0 else []
                else:
                    ks, vs = self.array_items(p.name)
                    if off < 0:
                        top = (max(ks) + 1) if ks and not self.is_assoc(p.name) else len(vs)
                        off = top + off
                        if off < 0:
                            res = []
                            ks = []
                    res = [v for k, v in zip(ks, vs) if (k if isinstance(k, int) else 0) >= off] if not self.is_assoc(p.name) else vs[off:]
                if ln is not None:
                    if ln < 0:
                        self.error_now(f"{ln}: substring expression < 0")
                        raise ShellAbort(1, fatal=True)
                    res = res[:ln]
                self._emit_values(res, True, star, quoted, out)
                return
            s = src[0]
            if off < 0:
                off = max(len(s) + off, 0) if len(s) + off >= 0 else None
                if off is None:
                    out.append(("", Q if quoted else EXP))
                    return
            if ln is None:
                r = s[off:]
            elif ln < 0:
                end = len(s) + ln
                if end < off:
                    self.error_now(f"{ln}: substring expression < 0")
                    raise ShellAbort(1, fatal=True)
                r = s[off:end]
            else:
                r = s[off:off + ln]
            res = [r]
        elif op == "@":
            code = (p.arg.literal() or "") if p.arg is not None else ""
            res = [self._transform(s, code, p) for s in src]
            if code == "A" and is_list:
                res = [self._transform_array(p.name)]
                is_list = False
        else:
            raise RuntimeError(f"bad op {op}")
        self._emit_values(res, is_list, star, quoted, out)

    def _emit_values(self, vals, is_list, star, quoted, out):
        flag = Q if quoted else EXP
        if not is_list:
            v = vals[0] if vals else None
            out.append((v or "", flag))
            return
        vals = [v for v in vals if v is not None]
        if star and quoted:
            ifs = self.get("IFS")
            sep = " " if ifs is None else ifs[:1]
            out.append((sep.join(vals), flag))
            return
        for i, v in enumerate(vals):
            if i:
                out.append(BREAK)
            out.append((v, flag))

    def _patsub(self, s, pat, rep_chunks, op):
        # patsub_replacement: an unquoted & in the replacement is the matched text
        has_amp = "patsub_replacement" in self.shopts and any(f != Q and "&" in t for t, f in rep_chunks if t is not None)
        if not has_amp:
            rep = "".join(" " if c is BREAK else c[0] for c in rep_chunks)
            return P.replace(s, pat, rep, op)
        # do it the slow way: find each match, build replacement with the matched text
        rx = P.compiled(pat)
        out, i, n = [], 0, len(s)

        def rep_for(m):
            r = []
            for c in rep_chunks:
                if c is BREAK:
                    r.append(" ")
                    continue
                t, f = c
                if f != Q:
                    r.append(re.sub(r"\\(.)|&", lambda mm: mm.group(1) if mm.group(1) else m, t))
                else:
                    r.append(t)
            return "".join(r)
        if op == "/#":
            for end in range(n, -1, -1):
                if rx.fullmatch(s, 0, end):
                    return rep_for(s[:end]) + s[end:]
            return s
        if op == "/%":
            for k in range(0, n + 1):
                if rx.fullmatch(s, k, n):
                    return s[:k] + rep_for(s[k:])
            return s
        while i <= n:
            end = None
            for e in range(n, i - 1, -1):
                if rx.fullmatch(s, i, e):
                    end = e
                    break
            if end is not None:
                out.append(rep_for(s[i:end]))
                if end == i:
                    if i < n:
                        out.append(s[i])
                    i += 1
                else:
                    i = end
                if op == "/":
                    out.append(s[i:])
                    return "".join(out)
                continue
            if i < n:
                out.append(s[i])
            i += 1
        return "".join(out)

    @staticmethod
    def _case_mod(s, op, pat):
        rx = P.compiled(pat)
        conv = str.upper if op[0] == "^" else str.lower
        if len(op) == 1:
            if s and rx.fullmatch(s[0]):
                return conv(s[0]) + s[1:]
            return s
        return "".join(conv(c) if rx.fullmatch(c) else c for c in s)

    def _transform(self, s, code, p):
        from .quoting import at_q
        if code == "Q":
            return at_q(s)
        if code == "U":
            return s.upper()
        if code == "u":
            return s[:1].upper() + s[1:]
        if code == "L":
            return s.lower()
        if code == "E":
            from .parser import decode_ansi
            return decode_ansi(s)
        if code == "a":
            v = self.lookup(p.name)
            return "".join(sorted(v.attrs & set("aAilnrtux"), key="aAilnrtux".index)) if v else ""
        if code == "A":
            from .quoting import single_quote
            v = self.lookup(p.name)
            flags = "".join(sorted(v.attrs & set("aAilnrtux"), key="aAilnrtux".index)) if v else ""
            return f"declare -{flags or '-'} {p.name}={single_quote(s)}" if flags else f"{p.name}={single_quote(s)}"
        if code == "P":
            return self.prompt_expand(s)
        if code == "K":
            from .quoting import at_q
            return at_q(s)
        return s

    def _transform_array(self, name):
        from .builtins import declare_line
        return declare_line(self, name, self.lookup(name)).replace("declare -a ", "declare -a ", 1)

    # -------- command and process substitution

    def command_subst(self, node: CmdSub):
        body = node.body
        # $(< file) reads the file without a subshell
        if len(body.items) == 1:
            ao, bg = body.items[0]
            if not bg and not ao.rest and len(ao.first.cmds) == 1 and not ao.first.negate:
                cmd = ao.first.cmds[0]
                if type(cmd) is Simple and not cmd.words and not cmd.assigns and len(cmd.redirs) == 1 \
                        and cmd.redirs[0].op == "<" and cmd.redirs[0].fd is None:
                    path = yield from self.expand_string(cmd.redirs[0].target)
                    try:
                        of = open_path(self.proc, path, "r")
                        data = yield from read_all(self.kernel, of)
                    except FSError as e:
                        yield from self.error(f"{path}: {e.msg}")
                        self.cmdsub_status = 1
                        return ""
                    self.cmdsub_status = 0
                    return _strip_nl(data.decode("utf-8", "surrogateescape"))
        pipe = Pipe()
        w = OpenFile("pipe", wr=True, pipe=pipe)
        r = OpenFile("pipe", rd=True, pipe=pipe)
        fds = dict(self.proc.fds)
        fds[1] = w
        self.trace_depth += 1
        child = self.spawn_subshell(body, fds=fds, extra=lambda sub: setattr(sub, "trace_depth", self.trace_depth))
        self.trace_depth -= 1
        r.ref()
        data = yield from read_all(self.kernel, r)
        r.unref(self.kernel)
        yield from self.wait_for([child], fg=False)
        self.cmdsub_status = child.status
        self.status = child.status
        text = data.decode("utf-8", "surrogateescape")
        if "\0" in text:
            text = text.replace("\0", "")
            self.error_now("warning: command substitution: ignored null byte in input")
        return _strip_nl(text)

    def proc_subst(self, node: ProcSub) -> str:
        pipe = Pipe()
        w = OpenFile("pipe", wr=True, pipe=pipe)
        r = OpenFile("pipe", rd=True, pipe=pipe)
        fds = dict(self.proc.fds)
        if node.direction == "<":
            fds[1] = w
            mine = r
        else:
            fds[0] = r
            mine = w
        self.spawn_subshell(node.body, fds=fds)
        fd = 63
        while fd in self.proc.fds:
            fd -= 1
        self.proc.set_fd(fd, mine)
        self._procsub_fds.append(fd)
        return f"/dev/fd/{fd}"

    @property
    def _procsub_fds(self):
        if not hasattr(self, "_psfds"):
            self._psfds = []
        return self._psfds

    def close_procsubs(self):
        if getattr(self, "_psfds", None):
            for fd in self._psfds:
                self.proc.set_fd(fd, None)
            self._psfds = []

    # ------------------------------------------------------------ execution

    def tick(self):
        if self.interrupted:
            self.interrupted = False
            raise Interrupted()
        k = self.kernel
        k.steps += 1
        if k.steps > k.budget:
            raise TimeLimit()

    def run_node(self, node):
        t = type(node)
        if t is CmdList:
            return (yield from self.run_list(node))
        if t is AndOr:
            return (yield from self.run_andor(node))
        if t is Pipeline:
            return (yield from self.run_pipeline(node))
        return (yield from self.run_command(node))

    def run_list(self, lst: CmdList):
        status = self.status
        items = lst.items
        last_i = len(items) - 1
        exec_ok = self.exec_ok
        for i, (ao, bg) in enumerate(items):
            self.exec_ok = exec_ok and i == last_i and not bg and not self.traps.get("EXIT")
            if bg:
                yield from self.run_background(ao)
                status = 0
            else:
                status = yield from self.run_andor(ao)
            if self.pending_traps:
                yield from self.run_pending_traps()
        self.exec_ok = False
        return status

    def run_andor(self, ao: AndOr):
        exec_ok = self.exec_ok
        if ao.rest:
            self.exec_ok = False
            self.no_errexit += 1
            try:
                status = yield from self.run_pipeline(ao.first)
            finally:
                self.no_errexit -= 1
            n = len(ao.rest)
            for i, (op, pl) in enumerate(ao.rest):
                if (op == "&&") == (status == 0):
                    last = i == n - 1
                    if not last:
                        self.no_errexit += 1
                    try:
                        status = yield from self.run_pipeline(pl, errexit_ok=last)
                    finally:
                        if not last:
                            self.no_errexit -= 1
            return status
        self.exec_ok = exec_ok
        return (yield from self.run_pipeline(ao.first))

    def run_pipeline(self, pl: Pipeline, errexit_ok=True):
        t0 = self.kernel.now
        if pl.negate:
            self.no_errexit += 1
        try:
            if not pl.cmds:
                status = 0
            elif len(pl.cmds) == 1:
                status = yield from self.run_command(pl.cmds[0])
                self.pipestatus = [status]
            else:
                status = yield from self._run_stages(pl)
        finally:
            if pl.negate:
                self.no_errexit -= 1
        if pl.negate:
            status = 0 if status else 1
        self.status = status
        if pl.timed:
            yield from self._report_time(pl, t0)
        if status != 0 and not pl.negate and errexit_ok:
            yield from self.after_failure(status)
        return status

    def after_failure(self, status):
        """ERR trap and `set -e` for a failed command."""
        if self.no_errexit or self.in_trap:
            return
        if "ERR" in self.traps and self.traps["ERR"]:
            yield from self.run_trap_action("ERR", status)
        if "errexit" in self.opts:
            raise ExitShell(status)

    def _report_time(self, pl, t0):
        dt = (self.kernel.now - t0) / NS
        user = sum(getattr(p, "cputime", 0) for p in []) if False else min(dt, 0.0) + 0.001 * bool(pl.cmds)
        if pl.posix_time:
            text = f"real {dt:.2f}\nuser {user:.2f}\nsys {0.0:.2f}\n"
        else:
            fmt = self.get("TIMEFORMAT")

            def mm(x):
                m, s = divmod(x, 60)
                return f"{int(m)}m{s:.3f}s"
            if fmt is not None:
                text = (fmt.replace("%3lR", mm(dt)).replace("%R", f"{dt:.3f}").replace("%U", f"{user:.3f}")
                        .replace("%S", "0.000").replace("%P", "0.00")) + "\n"
            else:
                text = f"\nreal\t{mm(dt)}\nuser\t{mm(user)}\nsys\t{mm(0.0)}\n"
        yield from self.err(text)

    def _run_stages(self, pl: Pipeline):
        cmds = pl.cmds
        n = len(cmds)
        procs = []
        prev_r = None
        for i, cmd in enumerate(cmds):
            fds = dict(self.proc.fds)
            if prev_r is not None:
                fds[0] = prev_r
            nxt_r = None
            if i < n - 1:
                pipe = Pipe()
                w = OpenFile("pipe", wr=True, pipe=pipe)
                nxt_r = OpenFile("pipe", rd=True, pipe=pipe)
                fds[1] = w
                if pl.pipe_err[i]:
                    fds[2] = w
            argv = self._literal_argv(cmd)
            child = self.spawn_subshell(cmd, fds=fds, argv=argv)
            if "monitor" in self.opts:
                child.pgid = procs[0].pid if procs else child.pid
            procs.append(child)
            if prev_r is not None and prev_r.refs == 0:
                prev_r.ref()
                prev_r.unref(self.kernel)
            if i < n - 1 and w.refs == 0:
                w.ref()
                w.unref(self.kernel)
            prev_r = nxt_r
        tty = self.world.tty_of(self.proc)
        old_fg = tty.fg_pgid if tty else None
        if tty is not None and self.interactive:
            tty.fg_pgid = procs[0].pid
        yield from self.wait_for(procs)
        if tty is not None and self.interactive:
            tty.fg_pgid = old_fg
        if any(p.state == "T" for p in procs):
            from .unparse import job_text
            job = self.add_job(procs, job_text(pl))
            job.reported_state = "Stopped"
            yield from self.err(f"\n[{job.num}]+  Stopped                 {job.text}\n")
            return 128 + SIGTSTP
        statuses = [p.status for p in procs]
        self.pipestatus = statuses
        if "pipefail" in self.opts:
            nz = [s for s in statuses if s != 0]
            return nz[-1] if nz else 0
        if self.interactive and procs[-1].term_sig == SIGINT:
            yield from self.err("\n")
            raise Interrupted()
        return statuses[-1]

    def _literal_argv(self, cmd):
        """Show a pipeline stage in ps as its command when its words are plain literals."""
        if type(cmd) is Simple and cmd.words:
            lits = [w.literal() for w in cmd.words]
            if all(x is not None for x in lits) and lits[0] not in self.funcs:
                return lits
        return None

    def run_background(self, ao: AndOr):
        fds = dict(self.proc.fds)
        if (fds.get(0) is not None and fds[0].kind == "tty") and "monitor" not in self.opts:
            fds[0] = OpenFile("null", rd=True, wr=True)
        argv = None
        if not ao.rest and len(ao.first.cmds) == 1 and not ao.first.negate:
            argv = self._literal_argv(ao.first.cmds[0])
        child = self.spawn_subshell(ao, fds=fds, argv=argv)
        if "monitor" in self.opts:
            child.pgid = child.pid
        else:
            child.bg = True
        self.last_bg = child.pid
        from .unparse import job_text
        job = self.add_job([child], job_text(ao) + " &")
        if self.interactive:
            yield from self.err(f"[{job.num}] {child.pid}\n")
        return 0

    def add_job(self, procs, text) -> Job:
        num = 1
        while num in self.jobs:
            num += 1
        if self.jobs:
            num = max(self.jobs) + 1
        job = Job(num, procs, text)
        self.jobs[num] = job
        self.job_order.append(num)

        def ended(_p, job=job):
            if job.report_at is None and job.state() == "Done":
                # bash notices a dead child at once while it waits (for a foreground job or in `wait`); otherwise the
                # SIGCHLD arrives after the next prompt is already up, so the notice comes one prompt later
                me = self.proc
                blocked = me.state == "S" and bool(me.waiting) and me.waiting[0] == "wait"
                job.report_at = self.prompt_no + (1 if blocked else 2)
                job.died_in_wait = blocked and self._in_wait_builtin
                # a death we caused this instant (kill) only lands once the command line has moved on: a `wait`
                # later on the same line still sees the job alive and reaps it
                job.died_line = None if blocked else self.prompt_no
        for p in procs:
            p.job = job
            p.on_exit.append(ended)
        return job

    def run_command(self, cmd):
        t = type(cmd)
        if t is Simple:
            return (yield from self.run_simple(cmd))
        if t is FuncDef:
            self.funcs[cmd.name] = cmd
            return 0
        saved = None
        if cmd.redirs:
            saved = yield from self.apply_redirs(cmd.redirs)
            if saved is None:
                return 1
        try:
            exec_ok = self.exec_ok
            self.exec_ok = False
            if t is Brace:
                self.exec_ok = exec_ok and not cmd.redirs
                return (yield from self.run_list(cmd.body))
            if t is Subshell:
                child = self.spawn_subshell(cmd.body)
                yield from self.wait_for([child])
                self.status = child.status
                return child.status
            if t is If:
                return (yield from self._run_if(cmd))
            if t is While:
                return (yield from self._run_while(cmd))
            if t is For:
                return (yield from self._run_for(cmd))
            if t is ArithFor:
                return (yield from self._run_arith_for(cmd))
            if t is Case:
                return (yield from self._run_case(cmd))
            if t is Cond:
                return (yield from self._run_cond(cmd))
            if t is ArithCmd:
                return (yield from self._run_arith_cmd(cmd))
            if t is Select:
                return (yield from self._run_select(cmd))
            raise RuntimeError(f"cannot run {cmd!r}")
        finally:
            if saved is not None:
                self.restore_redirs(saved)

    # -------- compound commands

    def _cond_list(self, lst):
        self.no_errexit += 1
        try:
            return (yield from self.run_list(lst))
        finally:
            self.no_errexit -= 1

    def _run_if(self, cmd: If):
        for cond, body in cmd.clauses:
            st = yield from self._cond_list(cond)
            if st == 0:
                return (yield from self.run_list(body))
        if cmd.else_ is not None:
            return (yield from self.run_list(cmd.else_))
        return 0

    def _loop(self):
        self.loop_depth += 1

    def _run_while(self, cmd: While):
        status = 0
        self.loop_depth += 1
        try:
            while True:
                self.tick()
                st = yield from self._cond_list(cmd.cond)
                if (st == 0) == cmd.until:
                    break
                try:
                    status = yield from self.run_list(cmd.body)
                except BreakLoop as b:
                    if b.n > 1:
                        b.n -= 1
                        raise
                    status = 0
                    break
                except ContinueLoop as c:
                    if c.n > 1:
                        c.n -= 1
                        raise
                    continue
        finally:
            self.loop_depth -= 1
        return status

    def _run_for(self, cmd: For):
        if cmd.words is None:
            items = list(self.positional)
        else:
            items = yield from self.expand_words(cmd.words)
        status = 0
        self.loop_depth += 1
        try:
            for item in items:
                self.tick()
                if "xtrace" in self.opts:
                    yield from self.trace(f"for {cmd.var} in {' '.join(xtrace_word(x) for x in items)}")
                self.set_var(cmd.var, item)
                try:
                    status = yield from self.run_list(cmd.body)
                except BreakLoop as b:
                    if b.n > 1:
                        b.n -= 1
                        raise
                    break
                except ContinueLoop as c:
                    if c.n > 1:
                        c.n -= 1
                        raise
                    continue
        finally:
            self.loop_depth -= 1
        return status

    def _arith_eval_word(self, w: Word, cmd_ctx=True):
        text = yield from self.expand_string(w, True)
        if "xtrace" in self.opts:
            yield from self.trace(f"(( {text.strip()} ))")
        self._arith_cmd = True
        self._arith_ctx = "((: "
        try:
            return self.arith(text) if text.strip() else None
        finally:
            self._arith_cmd = False
            self._arith_ctx = ""

    def _run_arith_for(self, cmd: ArithFor):
        status = 0
        try:
            yield from self._arith_eval_word(cmd.init)
        except ShellAbort as e:
            return e.status
        self.loop_depth += 1
        try:
            while True:
                self.tick()
                try:
                    v = yield from self._arith_eval_word(cmd.cond)
                except ShellAbort as e:
                    return e.status
                if v is not None and v == 0:
                    break
                try:
                    status = yield from self.run_list(cmd.body)
                except BreakLoop as b:
                    if b.n > 1:
                        b.n -= 1
                        raise
                    break
                except ContinueLoop as c:
                    if c.n > 1:
                        c.n -= 1
                        raise
                try:
                    yield from self._arith_eval_word(cmd.step)
                except ShellAbort as e:
                    return e.status
        finally:
            self.loop_depth -= 1
        return status

    def _run_case(self, cmd: Case):
        word = yield from self.expand_string(cmd.word)
        if "xtrace" in self.opts:
            yield from self.trace(f"case {word} in")
        nocase = "nocasematch" in self.shopts
        status = 0
        fall = False
        items = cmd.items
        i = 0
        while i < len(items):
            pats, body, term = items[i]
            matched = fall
            if not matched:
                for pw in pats:
                    pat = yield from self.expand_pattern(pw)
                    if P.fnmatch(pat, word, nocase, "extglob" in self.shopts or True):
                        matched = True
                        break
            if matched:
                if body is not None:
                    status = yield from self.run_list(body)
                else:
                    status = 0
                if term == ";;":
                    return status
                if term == ";&":
                    fall = True
                    i += 1
                    continue
                fall = False   # ;;& : keep testing the next patterns
            i += 1
        return status

    def _run_arith_cmd(self, cmd: ArithCmd):
        text = yield from self.expand_string(cmd.expr, True)
        if "xtrace" in self.opts:
            yield from self.trace(f"(( {text} ))")
        self._arith_cmd = True
        self._arith_ctx = "((: "
        try:
            v = self.arith(text)
        except ShellAbort as e:
            return e.status
        finally:
            self._arith_cmd = False
            self._arith_ctx = ""
        return 0 if v != 0 else 1

    def _run_select(self, cmd: Select):
        items = list(self.positional) if cmd.words is None else (yield from self.expand_words(cmd.words))
        if not items:
            return 0
        status = 0
        while True:
            menu = "".join(f"{i + 1}) {it}\n" for i, it in enumerate(items))
            yield from self.err(menu)
            yield from self.err(self.get("PS3", "#? "))
            line = yield from readline(self.kernel, self.proc.fds.get(0))
            if not line:
                yield from self.err("\n")
                return 1
            reply = line.decode().rstrip("\n")
            self.set_var("REPLY", reply)
            choice = items[int(reply) - 1] if reply.isdigit() and 1 <= int(reply) <= len(items) else ""
            self.set_var(cmd.var, choice)
            try:
                status = yield from self.run_list(cmd.body)
            except BreakLoop:
                return 0
            except ContinueLoop:
                continue
        return status

    # -------- [[ ]]

    def _run_cond(self, cmd: Cond):
        try:
            ok = yield from self._cond_eval(cmd.expr, "xtrace" in self.opts)
        except ShellAbort as e:
            return 2 if not e.fatal else e.status
        return 0 if ok else 1

    def _cond_eval(self, e, trace):
        t = type(e)
        if t is CondAnd:
            a = yield from self._cond_eval(e.left, trace)
            return a and (yield from self._cond_eval(e.right, trace))
        if t is CondOr:
            a = yield from self._cond_eval(e.left, trace)
            return a or (yield from self._cond_eval(e.right, trace))
        if t is CondNot:
            return not (yield from self._cond_eval(e.arg, trace))
        if t is CondWord:
            s = yield from self.expand_string(e.word)
            if trace:
                yield from self.trace(f"[[ {s} ]]")
            return s != ""
        if t is CondUnary:
            s = yield from self.expand_string(e.arg)
            if trace:
                yield from self.trace(f"[[ {e.op} {s} ]]")
            from .builtins import unary_test
            return unary_test(self, e.op, s)
        if t is CondBinary:
            left = yield from self.expand_string(e.left)
            op = e.op
            if op in ("==", "=", "!="):
                pat = yield from self.expand_pattern(e.right)
                if trace:
                    yield from self.trace(f"[[ {left} {op} {pat} ]]")
                m = P.fnmatch(pat, left, "nocasematch" in self.shopts, True)
                return m if op != "!=" else not m
            if op == "=~":
                rx_src = yield from self._regex_from(e.right)
                if trace:
                    yield from self.trace(f"[[ {left} =~ {rx_src} ]]")
                from .regex import ere_to_python
                try:
                    rx = re.compile(ere_to_python(rx_src), re.I if "nocasematch" in self.shopts else 0)
                except re.error:
                    return False
                m = rx.search(left)
                if m is None:
                    self.set_var("BASH_REMATCH", "", check=False)
                    v = self.lookup("BASH_REMATCH")
                    v.value = {}
                    v.attrs.add("a")
                    return False
                v = self.lookup("BASH_REMATCH")
                if v is None:
                    self.globals["BASH_REMATCH"] = v = Var({}, "a")
                v.attrs.add("a")
                v.value = {0: m.group(0)}
                for gi in range(1, (rx.groups or 0) + 1):
                    v.value[gi] = m.group(gi) or ""
                return True
            right = yield from self.expand_string(e.right)
            if trace:
                yield from self.trace(f"[[ {left} {op} {right} ]]")
            from .builtins import binary_test
            return binary_test(self, op, left, right, cond=True)
        raise RuntimeError(repr(e))

    def _regex_from(self, w: Word):
        """Regex operand of =~: quoted parts match literally."""
        chunks = []
        yield from self._expand_parts(w.parts, False, chunks)
        out = []
        for c in chunks:
            if c is BREAK:
                out.append(" ")
                continue
            t, f = c
            out.append(re.sub(r"([\\.^$*+?()\[\]{}|])", r"\\\1", t) if f == Q else t)
        return "".join(out)

    # -------- simple commands

    def trace(self, text: str):
        ps4 = self.get("PS4", "+ ")
        if ps4:
            lvl = 1 + self.trace_depth
            first, rest = (ps4[0], ps4[1:]) if ps4 else ("", "")
            prefix = first * lvl + rest
        else:
            prefix = ""
        yield from self.err(prefix + text + "\n")

    def run_simple(self, cmd: Simple):
        self.lineno = cmd.line
        k = self.kernel
        k.steps += 1
        if k.steps > k.budget:
            raise TimeLimit()
        if k.steps % 4096 == 0:
            yield ("yield",)
        exec_ok, self.exec_ok = self.exec_ok, False
        self.cmdsub_status = None
        words = cmd.words
        if words:
            first = words[0].literal()
            decl = first in DECLARATION and first not in self.funcs
            args = yield from self.expand_words(words, decl)
        else:
            args = []
        xtrace = "xtrace" in self.opts
        if not args:
            for a in cmd.assigns:
                yield from self.do_assign(a, xtrace)
            status = self.cmdsub_status if self.cmdsub_status is not None else 0
            if cmd.redirs:
                saved = yield from self.apply_redirs(cmd.redirs)
                if saved is None:
                    return 1
                self.restore_redirs(saved)
            self.close_procsubs()
            self.status = status
            return status
        # temporary assignments for this command
        tmp = []
        for a in cmd.assigns:
            val = ""
            if a.value is not None:
                val = (yield from self.expand_assignment_word(Word([Lit("x=")] + list(a.value.parts))))[2:]
            tmp.append((a.name, val, a.append))
        if xtrace:
            line = " ".join(f"{n}{'+' if ap else ''}={xtrace_word(v) if v else ''}" for n, v, ap in tmp)
            words_s = " ".join(xtrace_word(a) if isinstance(a, str) else a.raw for a in args)
            yield from self.trace((line + " " if line else "") + words_s)
        name = args[0] if isinstance(args[0], str) else args[0].raw
        self.last_arg = args[-1] if isinstance(args[-1], str) else args[-1].raw
        try:
            status = yield from self.dispatch(name, args, tmp, cmd.redirs, exec_ok)
        finally:
            self.close_procsubs()
        self.status = status
        return status

    def do_assign(self, a: Assign, xtrace=False):
        if a.array is not None:
            if xtrace:
                yield from self.trace(f"{a.raw}")
            yield from self.assign_array(a.name, a.array, a.append)
            return
        val = (yield from self.expand_assignment_word(Word([Lit("x=")] + list(a.value.parts)))) if a.value else "x="
        val = val[2:]
        index = None
        if a.index is not None:
            if self.is_assoc(a.name):
                index = yield from self.expand_string(a.index, True)
            else:
                idx_text = yield from self.expand_string(a.index, True)
                index = self.arith(idx_text)
        if xtrace:
            sub = f"[{index}]" if index is not None else ""
            yield from self.trace(f"{a.name}{sub}{'+' if a.append else ''}={xtrace_word(val) if val else ''}")
        self.set_var(a.name, val, index, a.append)

    def assign_array(self, name, elems, append=False, local=False, assoc=None):
        """name=(a b [k]=v)."""
        var = self.lookup(name) if not local else (self.scopes[-1].get(name) if self.scopes else self.globals.get(name))
        if var is not None and "r" in var.attrs:
            self.error_now(f"{name}: readonly variable")
            raise ShellAbort(1)
        is_assoc = assoc if assoc is not None else (var is not None and "A" in var.attrs)
        values = {}
        if append and var is not None and isinstance(var.value, dict):
            values = dict(var.value)
        elif append and var is not None and isinstance(var.value, str):
            values = {0: var.value}
        nxt = (max(values) + 1) if values and not is_assoc else 0
        pairs = []
        for w in elems:
            m = re.match(r"\[([^\]]*)\](\+?)=", w.raw)
            if m and w.parts and type(w.parts[0]) is Lit and w.parts[0].text.startswith("["):
                key_raw = m.group(1)
                rest_text = w.parts[0].text[m.end():]
                value_word = Word(([Lit(rest_text)] if rest_text else []) + list(w.parts[1:]), w.raw[m.end():])
                key_w = Parser(key_raw)._sub_word(key_raw, "arith")
                val = yield from self.expand_string(value_word)
                if is_assoc:
                    key = yield from self.expand_string(key_w, True)
                else:
                    kt = yield from self.expand_string(key_w, True)
                    key = self.arith(kt)
                    nxt = key + 1
                if m.group(2) and key in values:
                    values[key] = values[key] + val
                else:
                    values[key] = val
                continue
            if is_assoc:
                pairs.extend((yield from self.expand_word(w)))   # bash 5.1+: assoc=(k1 v1 k2 v2)
                continue
            vals = yield from self.expand_word(w)
            for v in vals:
                values[nxt] = v
                nxt += 1
        for i in range(0, len(pairs), 2):
            values[pairs[i]] = pairs[i + 1] if i + 1 < len(pairs) else ""
        cont = self._container(name, local)
        v = cont.get(name)
        if v is None:
            v = Var()
            cont[name] = v
        v.attrs.discard("a")
        v.attrs.discard("A")
        v.attrs.add("A" if is_assoc else "a")
        if "i" in v.attrs:
            values = {k: str(self.arith(x)) for k, x in values.items()}
        v.value = values

    def dispatch(self, name, args, tmp, redirs, exec_ok=False):
        from .builtins import BUILTINS, SPECIAL
        fn = self.funcs.get(name)
        if fn is not None:
            return (yield from self.call_function(fn, args, tmp, redirs))
        b = BUILTINS.get(name)
        if b is not None:
            saved = {}
            if redirs:
                saved = yield from self.apply_redirs(redirs)
                if saved is None:
                    return 1
            pushed = False
            if tmp:
                if name in SPECIAL and False:
                    for n, v, ap in tmp:
                        self.set_var(n, v, append=ap)
                else:
                    self.scopes.append({})
                    pushed = True
                    for n, v, ap in tmp:
                        cur = self.get(n) if ap else None
                        var = Var((cur or "") + v if ap else v, "x")
                        self.scopes[-1][n] = var
            if name == "exec" and len(args) == 1:
                for old in saved.values():      # redirections stay in effect
                    if old is not None:
                        old.unref(self.kernel)
                saved = {}
            try:
                status = yield from b(self, [a if isinstance(a, str) else a for a in args])
            finally:
                if pushed:
                    self.scopes.pop()
                self.restore_redirs(saved)
            return status if status is not None else 0
        return (yield from self.run_external(args, tmp, redirs, exec_ok))

    def call_function(self, fn: FuncDef, args, tmp, redirs):
        if len(self.funcnames) >= 1000:
            yield from self.error(f"{fn.name}: maximum function nesting level exceeded (1000)")
            raise ShellAbort(1)
        saved_redirs = None
        if redirs:
            saved_redirs = yield from self.apply_redirs(redirs)
            if saved_redirs is None:
                return 1
        scope = {}
        for n, v, ap in tmp:
            scope[n] = Var(v, "x")
        self.scopes.append(scope)
        saved_pos = self.positional
        self.positional = [a if isinstance(a, str) else a.raw for a in args[1:]]
        self.funcnames.append(fn.name)
        saved_opt = self.getopts_pos
        status = 0
        try:
            status = yield from self.run_command(fn.body)
        except ReturnFunc as r:
            status = r.status
        finally:
            self.funcnames.pop()
            self.positional = saved_pos
            self.scopes.pop()
            self.getopts_pos = saved_opt if False else self.getopts_pos
            if saved_redirs is not None:
                self.restore_redirs(saved_redirs)
        if "RETURN" in self.traps and self.traps["RETURN"]:
            yield from self.run_trap_action("RETURN", status)
        self.status = status
        return status

    # -------- external commands

    def find_command(self, name: str):
        """Path of an executable for `name` (None if not found)."""
        if "/" in name:
            return name
        if name in self.hashed:
            p = self.hashed[name]
            if self.fs.exists(p):
                return p
        path = self.get("PATH", "")
        for d in path.split(":"):
            d = d or "."
            cand = d.rstrip("/") + "/" + name
            try:
                node = self.fs.lookup(cand, self.proc.cwd)
            except FSError:
                continue
            if node.type == "f" and node.mode & 0o111:
                if "hashall" in self.opts:
                    self.hashed[name] = cand
                return cand
        return None

    def run_external(self, args, tmp, redirs, exec_ok=False):
        name = args[0] if isinstance(args[0], str) else args[0].raw
        args = [a if isinstance(a, str) else a.raw for a in args]
        path = self.find_command(name)
        if path is None:
            if name == "":
                yield from self.error(": command not found")
            else:
                yield from self.error(f"{name}: command not found")
            return 127
        env = self.env()
        for n, v, ap in tmp:
            env[n] = (env.get(n, "") + v) if ap else v
        from .exec import exec_program
        if exec_ok:
            # this (sub)shell process becomes the program
            if redirs:
                saved = yield from self.apply_redirs(redirs)
                if saved is None:
                    return 1
                for old in saved.values():      # never restored: drop the saved descriptors (pipe ends!)
                    if old is not None:
                        old.unref(self.kernel)
            self.proc.argv = list(args)
            self.proc.env = env
            st = yield from exec_program(self.proc, path, args, env, self)
            raise ExitShell(st)
        fds = dict(self.proc.fds)
        saved = None
        if redirs:
            saved = yield from self.apply_redirs(redirs)
            if saved is None:
                return 1
            fds = dict(self.proc.fds)
        child = self.kernel.spawn(self.host, args, None, self.proc, fds=fds, env=env)
        if saved is not None:
            self.restore_redirs(saved)
        self.kernel.start(child, exec_program(child, path, args, env, self))
        if "monitor" in self.opts:
            child.pgid = child.pid
        tty = self.world.tty_of(self.proc)
        old_fg = tty.fg_pgid if tty else None
        if tty is not None and self.interactive:
            tty.fg_pgid = child.pgid
        yield from self.wait_for([child])
        if tty is not None and self.interactive:
            tty.fg_pgid = old_fg
        if child.state == "T":
            job = self.add_job([child], " ".join(args))
            job.reported_state = "Stopped"
            yield from self.err(f"\n[{job.num}]+  Stopped                 {job.text}\n")
            return 128 + SIGTSTP
        if self.interactive and child.term_sig == SIGINT:
            yield from self.err("\n")         # after the terminal's ^C
            raise Interrupted()               # like bash: `sleep 9; echo hi` + Ctrl-C doesn't print hi
        elif child.term_sig and child.term_sig not in (SIGINT, 13):
            msg = SIG_DESC.get(child.term_sig, f"Signal {child.term_sig}")
            yield from self.err(f"{msg}\n" if self.interactive else f"{self.prefix()}{child.pid} {msg}" +
                                f"{'' if self.interactive else ''}                 {' '.join(args)}\n")
        return child.status

    # -------- redirections

    def apply_redirs(self, redirs):
        """Apply redirections to this process's fd table. Returns the undo record (None on error)."""
        saved = {}
        proc = self.proc
        for r in redirs:
            op = r.op
            fd = r.fd
            try:
                if op in ("<<", "<<-"):
                    hd = r.heredoc
                    if hd.quoted or hd.body is None:
                        text = hd.text
                    else:
                        text = yield from self.expand_string(hd.body, True)
                    of = anon_file(text.encode("utf-8", "surrogateescape"), self.kernel.now)
                    self._redirect(saved, 0 if fd is None else fd, of)
                    continue
                if op == "<<<":
                    text = yield from self.expand_string(r.target)
                    of = anon_file((text + "\n").encode("utf-8", "surrogateescape"), self.kernel.now)
                    self._redirect(saved, 0 if fd is None else fd, of)
                    continue
                if op in (">&", "<&"):
                    tgt = yield from self.expand_string(r.target)
                    dflt = 1 if op == ">&" else 0
                    if tgt == "-":
                        self._redirect(saved, dflt if fd is None else fd, None)
                        continue
                    if tgt.isdigit() or (tgt.endswith("-") and tgt[:-1].isdigit()):
                        src = int(tgt.rstrip("-"))
                        of = proc.fds.get(src)
                        if of is None:
                            yield from self.error(f"{src}: Bad file descriptor")
                            self.restore_redirs(saved)
                            return None
                        self._redirect(saved, dflt if fd is None else fd, of)
                        if tgt.endswith("-"):
                            self._redirect(saved, src, None)
                        continue
                    if op == ">&" and fd is None:
                        op = "&>"
                        target = tgt
                    else:
                        yield from self.error(f"{tgt}: ambiguous redirect")
                        self.restore_redirs(saved)
                        return None
                else:
                    fields = yield from self.expand_word(r.target)
                    if len(fields) != 1:
                        yield from self.error(f"{r.target.raw}: ambiguous redirect")
                        self.restore_redirs(saved)
                        return None
                    target = fields[0]
                    if target == "":
                        yield from self.error(": No such file or directory")
                        self.restore_redirs(saved)
                        return None
                if op == "<":
                    of = open_path(proc, target, "r")
                    self._redirect(saved, 0 if fd is None else fd, of)
                elif op in (">", ">|"):
                    of = open_path(proc, target, "w", noclobber=("noclobber" in self.opts and op == ">"))
                    self._redirect(saved, 1 if fd is None else fd, of)
                elif op == ">>":
                    of = open_path(proc, target, "a")
                    self._redirect(saved, 1 if fd is None else fd, of)
                elif op == "<>":
                    of = open_path(proc, target, "rw")
                    self._redirect(saved, 0 if fd is None else fd, of)
                elif op in ("&>", "&>>"):
                    of = open_path(proc, target, "a" if op == "&>>" else "w",
                                   noclobber=("noclobber" in self.opts and op == "&>"))
                    self._redirect(saved, 1, of)
                    self._redirect(saved, 2, of)
                else:
                    raise RuntimeError(op)
            except FSError as e:
                if e.code == "NOCLOBBER":
                    yield from self.error(f"{target}: cannot overwrite existing file")
                else:
                    yield from self.error(f"{target}: {e.msg}")
                self.restore_redirs(saved)
                return None
        return saved

    def _redirect(self, saved, fd, of):
        proc = self.proc
        if fd not in saved:
            old = proc.fds.get(fd)
            saved[fd] = old.ref() if old is not None else None
        proc.set_fd(fd, of)

    def restore_redirs(self, saved):
        if not saved:
            return
        proc = self.proc
        for fd, old in saved.items():
            proc.set_fd(fd, old)
            if old is not None:
                old.unref(self.kernel)

    # ------------------------------------------------------------ traps & signals

    def _trap_signal(self, sig: int) -> bool:
        """Kernel callback: True if this shell handles (or ignores) the signal itself."""
        name = SIGNALS.get(sig, str(sig))
        if name in self.traps:
            if self.traps[name] == "":
                return True
            self.pending_traps.append(sig)
            p = self.proc
            if p.state == "S" and p.waiting and p.waiting[0] in ("wait",) and getattr(self, "_in_wait_builtin", False):
                p.state = "R"
                p.token += 1
                self.kernel.runq.append(p)
            return True
        if self.interactive and sig == SIGINT and self.awaiting is None:
            # bash abandons the command line it is running (a loop of builtins, `read`, `wait`)
            self.interrupted = True
            p = self.proc
            if p.state == "S" and p.waiting and p.waiting[0] in ("tty", "wait"):
                self.kernel.poke(p)
            return True
        if self.interactive and sig in (SIGINT, SIGTERM, 3, SIGTSTP, 21, 22):
            return True
        return False

    def run_pending_traps(self):
        while self.pending_traps:
            sig = self.pending_traps.pop(0)
            name = SIGNALS.get(sig, str(sig))
            action = self.traps.get(name)
            if action:
                yield from self.run_trap_action(name, self.status)

    def run_trap_action(self, name, status):
        if self.in_trap:
            return
        action = self.traps.get(name)
        if not action:
            return
        self.in_trap = True
        saved = self.status
        self.status = status
        try:
            yield from self.eval_string(action, "trap")
        finally:
            self.in_trap = False
        if name != "EXIT":
            self.status = saved

    def run_exit_trap(self, status):
        action = self.traps.pop("EXIT", None)
        if action:
            self.status = status
            self.in_trap = True
            try:
                yield from self.eval_string(action, "trap")
            except ExitShell as e:
                status = e.status
            except ShellAbort as e:
                status = e.status
            finally:
                self.in_trap = False
        self.close_procsubs()
        return status

    # ------------------------------------------------------------ running source text

    def eval_string(self, src: str, what="eval"):
        """Parse and run a string in this shell (eval, trap actions, source, bash -c)."""
        p = Parser(src, self.aliases if "expand_aliases" in self.shopts else None, line=self.lineno or 1)
        status = 0
        while True:
            try:
                cmd = p.parse_next()
            except ParseError as e:
                yield from self.error(e.msg)
                return 2 if what != "source" else 1
            if cmd is None:
                return status
            status = yield from self.run_list(cmd)

    def run_script_text(self, src: str, interactive_lines=False):
        """Run source text command by command (aliases and syntax errors take effect in order)."""
        p = Parser(src, self.aliases if "expand_aliases" in self.shopts else None)
        status = 0
        while True:
            try:
                cmd = p.parse_next()
            except ParseError as e:
                self.lineno = e.line
                yield from self.error(e.msg)
                if not self.interactive and self.script is not None:
                    bad = src.splitlines()[e.line - 1] if 0 < e.line <= len(src.splitlines()) else ""
                    if "unexpected token" in e.msg:
                        yield from self.err(f"{self.prefix()}`{bad.strip()}'\n")
                return 2
            for wl, w in p.warnings:
                self.lineno = wl
                yield from self.error(w)
            p.warnings.clear()
            if cmd is None:
                return status
            p.aliases = self.aliases if "expand_aliases" in self.shopts else None
            try:
                status = yield from self.run_list(cmd)
            except ShellAbort as e:
                status = e.status
                self.status = status
                if e.fatal and not self.interactive:
                    raise ExitShell(status)

    # ------------------------------------------------------------ the interactive prompt

    def add_history(self, line: str):
        ctl = (self.get("HISTCONTROL") or "").split(":")
        if ("ignorespace" in ctl or "ignoreboth" in ctl) and line.startswith(" "):
            return
        if ("ignoredups" in ctl or "ignoreboth" in ctl) and self.history and self.history[-1] == line:
            return
        if "erasedups" in ctl:
            self.history = [h for h in self.history if h != line]
        if line.strip():
            self.history.append(line)

    def repl(self, history: bool = True, idle: bool = True, persistent: bool = False):
        """Read `src` line by line like a terminal: history expansion, job notices, one command at a time."""
        from .builtins import notify_jobs
        from .history import HistError, expand, join_lines
        tty = self.world.tty_of(self.proc)

        def fresh():
            return Parser("", self.aliases if "expand_aliases" in self.shopts else None, interactive=True)
        p = fresh()
        pending: list[str] = []
        hd_delims: list = []           # here-documents whose bodies are still being typed
        while True:
            if tty.input:      # leftover typeahead from a program that read less than a line
                raw = tty.input.decode("utf-8", "surrogateescape").rstrip("\n")
                tty.input.clear()
            elif tty.lines:
                raw = tty.lines.popleft()
            elif tty.source is not None and tty.source.lines:
                raw = tty.source.lines.popleft()
            elif persistent:
                if tty.eof_pending:              # Ctrl-D on an empty line: the shell exits
                    tty.eof_pending = False
                    if pending:
                        yield from self.err("bash: syntax error: unexpected end of file\n")
                        p, pending, hd_delims = fresh(), [], []
                        continue
                    yield from self.err("exit\n")
                    status = yield from self.run_exit_trap(self.status)
                    self.status = status
                    self._exited = True
                    return status
                self.awaiting = "PS2" if pending else "PS1"     # the live terminal shows this prompt
                yield ("tty", tty)
                self.awaiting = None
                if self.cancel_pending:          # Ctrl-C at the > prompt drops the unfinished command
                    self.cancel_pending = False
                    p, pending, hd_delims = fresh(), [], []
                continue
            else:
                break
            if self.interactive and not pending:
                if self.jobs:
                    # background work that is ready (not sleeping) runs "while the prompt is drawn"
                    for _ in range(64):
                        yield ("yield",)
                        if not any(q is not self.proc and q.state == "R" for q in self.kernel.runq):
                            break
                self.prompt_no += 1
                if self.jobs:
                    msg = notify_jobs(self)
                    if msg:
                        yield from self.err(msg)
            self.interrupted = False
            if self.cmd_hook is not None and not pending:
                self.cmd_hook("start", raw)      # the live terminal attributes output to commands
            if tty.echo and self.interactive:
                # the transcript shows each typed line after its prompt, like a terminal
                ps = self.get("PS1", "") if not pending else self.get("PS2", "> ")
                tty.write("cmd", (self.prompt_expand(ps or "") + raw + "\n").encode("utf-8", "surrogateescape"))
            line = raw
            in_body = bool(hd_delims)
            if in_body:                 # a here-document line: no history expansion; maybe its end
                d, strip = hd_delims[0]
                if (line.lstrip("\t") if strip else line) == d:
                    hd_delims.pop(0)
            if not in_body and self.interactive and "histexpand" in self.opts and ("!" in line or line.startswith("^")):
                try:
                    new, changed, print_only = expand(line, self.history + ([join_lines(pending)] if False else []),
                                                      quick=not pending)
                except HistError as e:
                    yield from self.err(f"bash: {e}\n")
                    p, pending = fresh(), []
                    continue
                if changed:
                    yield from self.err(new + "\n")
                    line = new
                if print_only:
                    if history:
                        self.add_history(line)
                    continue
            if not in_body:
                for m_ in re.finditer(r"<<(-?)[ \t]*(['\"]?)([^\s'\";|&<>()]+)\2", line.replace("<<<", "   ")):
                    hd_delims.append((m_.group(3), m_.group(1) == "-"))
            pending.append(line)
            self.typed_log.append(line)     # what was run, after history expansion (the keys simulator reads it)
            if self.cmd_hook is not None:
                self.cmd_hook("line", line)
            p.aliases = self.aliases if "expand_aliases" in self.shopts else None
            p.feed(line + "\n")
            while True:
                m = p.mark()
                try:
                    cmd = p.parse_next()
                except Incomplete:
                    p.reset(m)
                    break
                except ParseError as e:
                    yield from self.err(f"bash: {e.msg}\n")
                    if history and pending:
                        self.add_history(join_lines(pending))
                    p, pending = fresh(), []
                    break
                if cmd is None:
                    # nothing left to run: a blank or comment-only line ends here (a comment still goes in
                    # the history), so the next prompt is PS1 again
                    if pending and not hd_delims and p.at_end():
                        if history and any(x.strip() for x in pending):
                            self.add_history(join_lines(pending))
                        pending = []
                    break
                if pending:
                    if history:
                        self.add_history(join_lines(pending))
                    pending = []
                try:
                    yield from self.run_list(cmd)
                except ExitShell as e:
                    status = yield from self.run_exit_trap(e.status)
                    self.status = status
                    self._exited = True
                    return status
                except ShellAbort as e:
                    self.status = e.status
                except (BreakLoop, ContinueLoop, ReturnFunc):
                    pass
                except Interrupted:
                    self.status = 130
                    self.interrupted = False
                    if self.cmd_hook is not None:
                        self.cmd_hook("end", self.status)
                    p, pending, hd_delims = fresh(), [], []
                    break
                if self.cmd_hook is not None:
                    self.cmd_hook("end", self.status)
                if self.pending_traps:
                    yield from self.run_pending_traps()
                p.aliases = self.aliases if "expand_aliases" in self.shopts else None
        if pending and any(x.strip() for x in pending):
            text = "\n".join(pending) + "\n"
            try:
                Parser(text).parse_all()
            except ParseError as e:
                yield from self.err(f"bash: {e.msg}\n")
                if "matching" in e.msg:
                    yield from self.err("bash: syntax error: unexpected end of file\n")
            if history:
                self.add_history(join_lines(pending))
        if idle:
            yield ("idle",)
        return self.status

    def prompt_expand(self, s: str) -> str:
        """PS1/PS2 decoding: backslash escapes (\\u \\h \\w \\033 \\e ...), then parameter expansion (promptvars)."""
        user = self.host.user_name(self.proc.uid)
        cwd = self.get("PWD", self.proc.cwd)
        rep = {"u": user, "h": self.host.name.split(".")[0], "H": self.host.name, "w": self.tilde_short(cwd),
               "W": "~" if cwd == self.get("HOME") else (cwd.rsplit("/", 1)[-1] or "/"),
               "$": "#" if self.proc.uid == 0 else "$", "n": "\n", "r": "\r", "s": "bash", "v": "5.2", "V": "5.2.21",
               "\\": "\\", "e": "\x1b", "a": "\x07", "[": "", "]": ""}

        def one(m):
            g = m.group(1)
            if g[0] in "01234567":
                return chr(int(g, 8))
            return rep.get(g, "\\" + g)
        out = re.sub(r"\\([0-7]{1,3}|.)", one, s)
        if "$" in out and "promptvars" in self.shopts:
            try:
                out = self.expand_prompt_vars(out)
            except Exception:  # noqa: BLE001 - a broken PS1 shows as typed
                pass
        return out

    def expand_prompt_vars(self, text: str) -> str:
        """Parameter expansion of an already-decoded prompt (no command substitution side effects needed here)."""
        from .parser import Parser
        word = Parser('"' + text.replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`") + '"').parse_all()
        cmd = word.items[0][0].first.cmds[0]
        gen = self.expand_word(cmd.words[0]) if hasattr(self, "expand_word") else None
        if gen is None:
            return text
        try:
            while True:
                next(gen)
        except StopIteration as stop:
            vals = stop.value
        return " ".join(vals) if isinstance(vals, list) else str(vals)


def _brace_sequence(text: str):
    m = re.fullmatch(r"(-?\d+)\.\.(-?\d+)(?:\.\.(-?\d+))?", text)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        step = abs(int(m.group(3))) if m.group(3) else 1
        step = step or 1
        width = 0
        sa, sb = m.group(1).lstrip("-"), m.group(2).lstrip("-")
        if (len(sa) > 1 and sa[0] == "0") or (len(sb) > 1 and sb[0] == "0"):
            width = max(len(m.group(1)), len(m.group(2)))
        seq = range(a, b + 1, step) if a <= b else range(a, b - 1, -step)
        out = []
        for x in seq:
            if width:
                s = f"{abs(x):0{width - (1 if x < 0 else 0)}d}"
                out.append("-" + s if x < 0 else s)
            else:
                out.append(str(x))
        return out
    m = re.fullmatch(r"([^.\\])\.\.([^.\\])(?:\.\.(-?\d+))?", text)
    if m:
        a, b = ord(m.group(1)), ord(m.group(2))
        step = abs(int(m.group(3))) if m.group(3) else 1
        step = step or 1
        seq = range(a, b + 1, step) if a <= b else range(a, b - 1, -step)
        return [chr(x) for x in seq]
    return None


def _strip_nl(s: str) -> str:
    return s.rstrip("\n")
