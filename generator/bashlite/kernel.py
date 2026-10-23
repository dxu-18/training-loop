"""Processes, pipes, signals, the terminal and a virtual clock, run by a cooperative scheduler.

Every process is a Python generator. It yields only when it must wait: ("sleep", until_ns),
("read", pipe), ("write", pipe), ("wait", [procs], until_ns | None), ("tty",) or ("yield",).
CPU work takes no virtual time, so output never depends on how fast the machine is; only sleeps,
timeouts and simulated delays move the clock.
"""

from __future__ import annotations

import heapq
import itertools
from collections import deque

SIGNALS = {1: "HUP", 2: "INT", 3: "QUIT", 4: "ILL", 5: "TRAP", 6: "ABRT", 7: "BUS", 8: "FPE", 9: "KILL", 10: "USR1",
           11: "SEGV", 12: "USR2", 13: "PIPE", 14: "ALRM", 15: "TERM", 16: "STKFLT", 17: "CHLD", 18: "CONT",
           19: "STOP", 20: "TSTP", 21: "TTIN", 22: "TTOU", 23: "URG", 24: "XCPU", 25: "XFSZ", 26: "VTALRM",
           27: "PROF", 28: "WINCH", 29: "IO", 30: "PWR", 31: "SYS"}
SIGNUM = {v: k for k, v in SIGNALS.items()}
SIGHUP, SIGINT, SIGQUIT, SIGKILL, SIGPIPE, SIGTERM, SIGCHLD, SIGCONT, SIGSTOP, SIGTSTP = 1, 2, 3, 9, 13, 15, 17, 18, 19, 20
STOP_SIGNALS = {19, 20, 21, 22}
IGNORED_BY_DEFAULT = {17, 23, 28}
# how bash describes a job killed by a signal
SIG_DESC = {1: "Hangup", 2: "Interrupt", 3: "Quit", 4: "Illegal instruction", 6: "Aborted", 7: "Bus error",
            8: "Floating point exception", 9: "Killed", 10: "User defined signal 1", 11: "Segmentation fault",
            12: "User defined signal 2", 13: "Broken pipe", 14: "Alarm clock", 15: "Terminated", 24: "CPU time limit exceeded",
            19: "Stopped (signal)", 20: "Stopped", 21: "Stopped (tty input)", 22: "Stopped (tty output)"}

NS = 1_000_000_000


class Signal(BaseException):
    """Thrown into a process generator to deliver a fatal signal."""

    def __init__(self, sig: int):
        super().__init__(sig)
        self.sig = sig


class ProcExit(BaseException):
    """Raised inside a process to exit immediately with `status`."""

    def __init__(self, status: int):
        super().__init__(status)
        self.status = status


class TimeLimit(BaseException):
    """The run used up its step budget (an infinite loop?) or simulated time limit."""


class OutputLimit(BaseException):
    """The run printed too much."""


class Deadlock(Exception):
    pass


# ---------------------------------------------------------------------------- pipes & open files

class Pipe:
    CAP = 65536
    __slots__ = ("buf", "readers", "writers", "waiters")

    def __init__(self):
        self.buf = bytearray()
        self.readers = 0
        self.writers = 0
        self.waiters: list = []


class OpenFile:
    """An open file description (shared by dup'ed and inherited descriptors)."""
    __slots__ = ("kind", "inode", "pos", "append", "rd", "wr", "pipe", "refs", "path", "stream", "tty", "rng", "fs")

    def __init__(self, kind, rd=False, wr=False, inode=None, pipe=None, path="", stream="out", tty=None, append=False):
        self.kind, self.rd, self.wr, self.inode, self.pipe = kind, rd, wr, inode, pipe
        self.pos, self.append, self.refs, self.path, self.stream, self.tty = 0, append, 0, path, stream, tty
        self.rng = None
        self.fs = None
        if pipe is not None:
            if rd:
                pipe.readers += 1
            if wr:
                pipe.writers += 1

    def ref(self):
        self.refs += 1
        return self

    def unref(self, kernel):
        self.refs -= 1
        if self.refs <= 0 and self.pipe is not None:
            p = self.pipe
            if self.rd:
                p.readers -= 1
            if self.wr:
                p.writers -= 1
            self.pipe = None
            kernel.wake(p)

    @property
    def is_tty(self) -> bool:
        return self.kind == "tty"


class TTY:
    """The session's terminal. `out` collects (stream, text) in order: stream is out / err / echo."""

    def __init__(self, width: int = 80, height: int = 24, name: str = "pts/0"):
        self.width, self.height, self.name = width, height, name
        self.out: list[list] = []
        self.size = 0
        self.limit = 2_000_000
        self.input = bytearray()     # typeahead (keys typed while a command runs)
        self.lines = deque()         # lines typed at the terminal that nobody has read yet
        self.input_eof = True        # when nothing is left to read, the terminal gives EOF (like Ctrl-D)
        self.fg_pgid = 0
        self.waiters: list = []
        self.mute = False

    def write(self, stream: str, data: bytes) -> None:
        if self.mirror is not None:
            self.mirror.write(stream, data)
        if self.mute:
            return
        text = data.decode("utf-8", "replace")
        self.size += len(text)
        if self.size > self.limit:
            raise OutputLimit()
        if self.out and self.out[-1][0] == stream:
            self.out[-1][1] += text
        else:
            self.out.append([stream, text])

    echo = False        # show typed lines (after a prompt) in the transcript
    source = None       # a pane bridged to another terminal reads that terminal's typed lines
    mirror = None       # ... and copies its output there
    on_starve = None

    def take_line(self) -> bool:
        """Move the next typed line into the input buffer. False if nothing is left."""
        if not self.lines:
            if self.source is not None and self.source.lines:
                self.input += (self.source.lines.popleft() + "\n").encode("utf-8", "surrogateescape")
                return True
            return False
        line = self.lines.popleft()
        if self.echo:
            self.write("in", (line + "\n").encode("utf-8", "surrogateescape"))
        self.input += (line + "\n").encode("utf-8", "surrogateescape")
        return True

    def text(self, streams=("out", "err", "echo", "cmd", "in")) -> str:
        return "".join(t for s, t in self.out if s in streams)


# ---------------------------------------------------------------------------- processes

class Proc:
    _seq = itertools.count()

    def __init__(self, kernel, host, pid: int, argv: list[str], gen=None, parent=None, uid=1000, gid=1000):
        self.kernel, self.host, self.pid = kernel, host, pid
        self.argv = argv
        self.gen = gen
        self.parent = parent
        self.ppid = parent.pid if parent is not None else 1
        self.pgid = pid
        self.uid, self.gid = uid, gid
        self.gids = {gid}
        self.fds: dict[int, OpenFile] = {}
        self.cwd = "/"
        self.env: dict[str, str] = {}
        self.umask = 0o022
        self.state = "R"            # R runnable · S blocked · T stopped · Z exited
        self.status = 0
        self.term_sig = 0
        self.resume = None
        self.pending: list[int] = []
        self.ignored: set[int] = set()
        self.trapper = None          # shell with traps: callable(sig) -> handled?
        self.token = 0               # bumps every time the process blocks (stale wakeups are ignored)
        self.waiting = None
        self.saved_wait = None       # what a stopped process was waiting on
        self.waiters: list = []      # procs waiting for this one to change state
        self.start = kernel.now
        self.end = None
        self.tty = "?"
        self.stat_extra = ""
        self.cpu = 0.0               # %CPU and %MEM shown by ps/top
        self.mem = 0.0
        self.vsz = 8964
        self.rss = 5120
        self.cputime = 0             # seconds of CPU time shown by ps
        self.ports: list[int] = []
        self.gpu: list[tuple[int, int]] = []   # (gpu index, MiB)
        self.nice = 0
        self.daemon = False          # keeps running when the session ends
        self.shell = None
        self.job = None
        self.on_exit = []
        self.ctty = parent.ctty if parent is not None else None
        self.bg = parent.bg if parent is not None else False    # started with & while job control is off

    @property
    def cmdline(self) -> str:
        return " ".join(self.argv)

    @property
    def comm(self) -> str:
        a = self.argv[0] if self.argv else "?"
        return a.rsplit("/", 1)[-1][:15]

    def alive(self) -> bool:
        return self.state != "Z"

    def fd(self, n: int) -> OpenFile | None:
        return self.fds.get(n)

    def set_fd(self, n: int, of: OpenFile | None) -> None:
        old = self.fds.pop(n, None)
        if of is not None:
            self.fds[n] = of.ref()
        if old is not None:
            old.unref(self.kernel)

    def close_all(self) -> None:
        fds, self.fds = self.fds, {}
        for of in fds.values():
            of.unref(self.kernel)


class Kernel:
    def __init__(self, world, start_ns: int):
        self.world = world
        self.now = start_ns
        self.runq: deque = deque()
        self.timers: list = []       # heap of (when, seq, token, proc | callable)
        self.current: Proc | None = None
        self.steps = 0
        self.budget = 4_000_000
        self.max_time = start_ns + 400 * 86400 * NS
        self.strict = False          # re-raise Python errors from utilities (tests)
        self.errors: list[str] = []

    # ------------------------------------------------------------ process lifecycle

    def spawn(self, host, argv, gen, parent=None, fds=None, cwd=None, env=None, uid=None, gid=None) -> Proc:
        pid = host.alloc_pid()
        p = Proc(self, host, pid, list(argv), gen, parent,
                 uid if uid is not None else (parent.uid if parent else 1000),
                 gid if gid is not None else (parent.gid if parent else 1000))
        if parent is not None:
            p.gids = set(parent.gids)
            p.tty = parent.tty
            p.umask = parent.umask
            p.pgid = parent.pgid
            p.cwd = parent.cwd if cwd is None else cwd
            src = parent.fds if fds is None else fds
        else:
            p.cwd = cwd or "/"
            src = fds or {}
        for n, of in src.items():
            p.fds[n] = of.ref()
        if env is not None:
            p.env = env
        host.procs[pid] = p
        if gen is not None:
            self.runq.append(p)
        return p

    def start(self, p: Proc, gen) -> Proc:
        """Give a spawned process its program and make it runnable."""
        p.gen = gen
        if p.state == "R":
            self.runq.append(p)
        return p

    def poke(self, p: Proc) -> None:
        """Make a blocked process re-check whatever it is waiting for."""
        if p.state == "S":
            p.state = "R"
            p.token += 1
            self.runq.append(p)

    def wake(self, obj) -> None:
        ws = obj.waiters
        if not ws:
            return
        obj.waiters = []
        for p, tok in ws:
            if p.state == "S" and p.token == tok:
                p.state = "R"
                self.runq.append(p)

    def _exit(self, p: Proc, status: int, sig: int = 0) -> None:
        if p.state == "Z":
            return
        p.state = "Z"
        p.status = status & 0xFF if not sig else 128 + sig
        p.term_sig = sig
        p.end = self.now
        p.close_all()
        p.host.procs.pop(p.pid, None)
        for cb in p.on_exit:
            cb(p)
        self.wake(p)

    def kill_now(self, p: Proc, sig: int) -> None:
        """Terminate p immediately (SIGKILL semantics)."""
        if p.state == "Z":
            return
        gen = p.gen
        p.gen = None
        if gen is not None and p is not self.current:
            try:
                gen.close()
            except BaseException:  # noqa: BLE001
                pass
        self._exit(p, 128 + sig, sig)

    # ------------------------------------------------------------ signals

    def signal(self, p: Proc, sig: int) -> None:
        if p.state == "Z" or sig == 0:
            return
        if sig == SIGKILL:
            if p is self.current:
                p.pending.insert(0, SIGKILL)
            else:
                self.kill_now(p, sig)
            return
        if sig == SIGCONT:
            if p.state == "T":
                self._resume_stopped(p)
            if p.trapper is not None:
                p.trapper(sig)
            return
        if sig in STOP_SIGNALS:
            if sig != SIGSTOP and (sig in p.ignored or (p.trapper is not None and p.trapper(sig))):
                return
            self._stop(p, sig)
            return
        if sig in p.ignored:
            return
        if p.trapper is not None and p.trapper(sig):
            return
        if sig in IGNORED_BY_DEFAULT:
            return
        p.pending.append(sig)
        if p.state == "S":
            p.state = "R"
            p.token += 1
            self.runq.append(p)
        elif p.state == "T":
            pass   # delivered when continued (KILL/TERM to a stopped job: bash sends CONT too)

    def _stop(self, p: Proc, sig: int) -> None:
        if p.state in ("T", "Z"):
            return
        p.saved_wait = (p.state, p.waiting)
        p.state = "T"
        p.token += 1
        p.term_sig = sig
        if p in self.runq:
            self.runq.remove(p)
        self.wake(p)

    def _resume_stopped(self, p: Proc) -> None:
        saved = p.saved_wait
        p.term_sig = 0
        p.saved_wait = None
        if saved is not None and saved[0] == "S" and saved[1]:
            # it was blocked (sleeping, reading, waiting): it goes back to that, like a restarted system call
            p.state = "R"
            self._block(p, saved[1])
        else:
            p.state = "R"
            self.runq.append(p)
        self.wake(p)

    # ------------------------------------------------------------ scheduling

    def add_timer(self, when: int, fn) -> None:
        heapq.heappush(self.timers, (when, next(Proc._seq), -1, fn))

    def _block(self, p: Proc, req) -> None:
        kind = req[0]
        if kind == "yield":
            self.runq.append(p)
            return
        if kind == "idle":       # an interactive shell waiting for the next run
            p.state = "I"
            return
        if kind == "forever":    # a daemon: sleeps until a signal arrives
            p.state = "S"
            p.token += 1
            p.waiting = req
            return
        p.state = "S"
        p.token += 1
        tok = p.token
        p.waiting = req
        if kind == "sleep":
            heapq.heappush(self.timers, (req[1], next(Proc._seq), tok, p))
        elif kind in ("read", "write"):
            obj = req[1]
            if (kind == "read" and (obj.buf or obj.writers == 0)) or (kind == "write" and (len(obj.buf) < obj.CAP or obj.readers == 0)):
                p.state = "R"
                self.runq.append(p)
                return
            obj.waiters.append((p, tok))
        elif kind == "wait":
            procs, until = req[1], req[2] if len(req) > 2 else None
            # children that already exited don't count: block on the live ones (else `true | sleep 1` spins)
            live = [c for c in procs if c.state != "Z"]
            if not live or any(c.state == "T" for c in live):
                p.state = "R"
                self.runq.append(p)
                return
            for c in live:
                c.waiters.append((p, tok))
            if until is not None:
                heapq.heappush(self.timers, (until, next(Proc._seq), tok, p))
        elif kind == "tty":
            tty = req[1]
            tty.waiters.append((p, tok))
            hook = getattr(tty, "on_starve", None)
            if hook is not None:
                hook()
        else:
            raise RuntimeError(f"bad block request {req!r}")

    def _step(self, p: Proc) -> None:
        if p.state != "R" or p.gen is None:
            return
        self.current = p
        try:
            if p.pending:
                sig = p.pending.pop(0)
                if sig == SIGKILL:
                    self.current = None
                    self.kill_now(p, SIGKILL)
                    return
                req = p.gen.throw(Signal(sig))
            else:
                v, p.resume = p.resume, None
                req = p.gen.send(v)
        except StopIteration as e:
            self.current = None
            self._exit(p, e.value if isinstance(e.value, int) else 0)
            return
        except Signal as e:
            self.current = None
            self._exit(p, 128 + e.sig, e.sig)
            return
        except ProcExit as e:
            self.current = None
            self._exit(p, e.status)
            return
        except (TimeLimit, OutputLimit):
            self.current = None
            raise
        except Exception as e:  # noqa: BLE001  (a bug in a simulated command: don't take the whole run down)
            self.current = None
            if self.strict:
                raise
            self.errors.append(f"{p.cmdline}: {type(e).__name__}: {e}")
            tty = next((of.tty for n, of in p.fds.items() if n == 2 and of.kind == "tty"), None)
            if tty is not None:
                tty.write("err", f"bash-lite: internal error in `{p.comm}`: {type(e).__name__}: {e}\n".encode())
            self._exit(p, 70)
            return
        self.current = None
        if p.state == "T":     # stopped itself (kill -STOP $BASHPID)
            p.saved_wait = ("R", req)
            return
        if p.pending and SIGKILL in p.pending:
            self.kill_now(p, SIGKILL)
            return
        if p.pending:
            self.runq.append(p)
            return
        self._block(p, req)

    def run(self, done) -> None:
        """Run processes until done() is true. Raises Deadlock if nothing can make progress."""
        while not done():
            self.steps += 1
            if self.steps > self.budget:
                raise TimeLimit()
            if self.runq:
                self._step(self.runq.popleft())
                continue
            # nothing runnable: advance the clock to the next timer
            while self.timers:
                when, _, tok, obj = heapq.heappop(self.timers)
                if callable(obj) and not isinstance(obj, Proc):
                    if when > self.now:
                        self.now = when
                    obj()
                    break
                if obj.state == "S" and obj.token == tok:
                    if when > self.now:
                        self.now = when
                    if self.now > self.max_time:
                        raise TimeLimit()
                    obj.state = "R"
                    self.runq.append(obj)
                    break
            else:
                raise Deadlock()
