"""Running programs: executable lookup results become processes running a simulated utility or a script.

Simulated binaries in /usr/bin are small files starting with an ELF header and `BASHLITE:<name>`; running
one calls the Python implementation registered under that name (utils/).
"""

from __future__ import annotations

from .fileio import read, readline, write, write_nb
from .kernel import ProcExit, Signal
from .vfs import FSError

ELF = b"\x7fELF\x02\x01\x01\x00"
MARK = b"BASHLITE:"
UTILS: dict = {}


def util(*names):
    def reg(fn):
        for n in names:
            UTILS[n] = fn
        return fn
    return reg


def stub_bytes(name: str) -> bytes:
    return ELF + b"\x00" * 8 + b"\x02\x00\x3e\x00\x01\x00\x00\x00" + MARK + name.encode() + b"\n"


def program_of(data: bytes):
    if not data.startswith(ELF):
        return None
    i = data.find(MARK)
    if i < 0:
        return ""
    j = data.find(b"\n", i)
    return data[i + len(MARK):j].decode()


class UsageError(Exception):
    def __init__(self, msg: str):
        super().__init__(msg)
        self.msg = msg


class Ctx:
    """What a simulated command sees: argv, environment, its process (fds, cwd, uid) and helpers."""

    def __init__(self, proc, argv, env, shell=None):
        self.proc = proc
        self.argv = list(argv)
        self.env = env
        self.k = proc.kernel
        self.world = self.k.world
        self.host = proc.host
        self.fs = proc.host.fs
        self.name = argv[0].rsplit("/", 1)[-1] if argv else "?"
        self.shell = shell
        self._buf: list[bytes] = []
        self._bufsize = 0

    # ------------------------------------------------------------ properties

    @property
    def cwd(self) -> str:
        return self.proc.cwd

    @property
    def uid(self) -> int:
        return self.proc.uid

    @property
    def now(self) -> int:
        return self.k.now

    @property
    def tz(self):
        return self.env.get("TZ")

    def stdout_is_tty(self) -> bool:
        of = self.proc.fds.get(1)
        return of is not None and of.kind == "tty"

    def foreground(self) -> bool:
        """Is this process in the terminal's foreground (where a simulated Ctrl-C would reach it)?"""
        tty = self.world.tty_of(self.proc)
        return tty is not None and tty.fg_pgid in (self.proc.pgid, 0) and not self.proc.bg

    def stdin_is_tty(self) -> bool:
        of = self.proc.fds.get(0)
        return of is not None and of.kind == "tty"

    @property
    def columns(self) -> int:
        try:
            return int(self.env.get("COLUMNS", "") or 80)
        except ValueError:
            return 80

    # ------------------------------------------------------------ output

    def w(self, text) -> None:
        """Write to stdout (C stdio buffering: line-buffered on a terminal, else block-buffered)."""
        data = text.encode("utf-8", "surrogateescape") if isinstance(text, str) else bytes(text)
        of = self.proc.fds.get(1)
        if of is None:
            raise FSError("EBADF")
        if of.kind == "tty":
            self._flush_nb()
            of.tty.write(of.stream, data)
            return
        self._buf.append(data)
        self._bufsize += len(data)
        if of.kind != "pipe" and self._bufsize >= 8192:
            self._flush_nb()

    def _flush_nb(self):
        if not self._buf:
            return
        of = self.proc.fds.get(1)
        data = b"".join(self._buf)
        self._buf, self._bufsize = [], 0
        if of is not None and of.kind != "pipe":
            write_nb(self.k, of, data)
        else:
            self._buf, self._bufsize = [data], len(data)

    def flush(self):
        if not self._buf:
            return
        data = b"".join(self._buf)
        self._buf, self._bufsize = [], 0
        yield from write(self.k, self.proc.fds.get(1), data)

    def maybe_flush(self):
        if self._bufsize >= 4096:
            yield from self.flush()

    def out(self, text):
        """Write and flush (generator)."""
        self.w(text)
        yield from self.flush()

    def e(self, text: str) -> None:
        """Write to stderr (unbuffered). Flushes pending stdout first when both go to the same place."""
        of = self.proc.fds.get(2)
        if of is None:
            return
        o1 = self.proc.fds.get(1)
        if o1 is of and of.kind != "pipe":
            self._flush_nb()
        try:
            write_nb(self.k, of, text.encode("utf-8", "surrogateescape"))
        except (FSError, Signal):
            pass

    def err(self, msg: str) -> None:
        self.e(f"{self.name}: {msg}\n")

    def usage(self, msg: str | None = None, status: int = 1, help_hint: bool = True) -> int:
        if msg:
            self.err(msg)
        if help_hint:
            self.e(f"Try '{self.name} --help' for more information.\n")
        return status

    # ------------------------------------------------------------ input

    def open(self, path: str, mode: str = "r"):
        from .fileio import open_path
        return open_path(self.proc, path, mode)

    def read_fd(self, fd: int = 0):
        """All remaining bytes from a file descriptor."""
        of = self.proc.fds.get(fd)
        if of is None:
            raise FSError("EBADF")
        chunks = []
        while True:
            data = yield from read(self.k, of, -1)
            if not data:
                return b"".join(chunks)
            chunks.append(data)

    def read_path(self, path: str):
        """Whole file ('-' = stdin). Raises FSError."""
        if path == "-":
            return (yield from self.read_fd(0))
        of = self.open(path)
        if of.kind == "dir":
            raise FSError("EISDIR")
        if of.kind == "file":
            data = bytes(of.inode.data)
            of.inode.atime = self.k.now
            return data
        chunks = []
        while True:
            data = yield from read(self.k, of, -1)
            if not data:
                return b"".join(chunks)
            chunks.append(data)

    def text(self, path: str):
        data = yield from self.read_path(path)
        return data.decode("utf-8", "surrogateescape")

    def reader(self, path: str = "-"):
        of = self.proc.fds.get(0) if path == "-" else self.open(path)
        if of is None:
            raise FSError("EBADF")
        if of.kind == "dir":
            raise FSError("EISDIR")
        return LineReader(self.k, of)

    # ------------------------------------------------------------ files

    def lookup(self, path: str, follow: bool = True):
        return self.fs.lookup(path, self.cwd, follow, self.uid, self.proc.gids)

    def abspath(self, path: str) -> str:
        import posixpath
        return posixpath.normpath(posixpath.join(self.cwd, path)) if not path.startswith("//") else path

    def sleep(self, seconds: float):
        if seconds > 0:
            yield ("sleep", self.k.now + int(seconds * 1_000_000_000))

    # ------------------------------------------------------------ running other programs

    def spawn(self, argv, fds=None, env=None, cwd=None, uid=None):
        """Start a program (PATH lookup like execvp). Returns the Proc, or None if not found."""
        path = find_in_path(self.fs, argv[0], (env or self.env).get("PATH", ""), cwd or self.cwd)
        if path is None:
            return None
        p = self.k.spawn(self.host, argv, None, self.proc, fds=fds, cwd=cwd, env=dict(env or self.env), uid=uid)
        if uid is not None:
            p.gids = {0} if uid == 0 else set(self.proc.gids)
            p.gid = 0 if uid == 0 else self.proc.gid
        self.k.start(p, exec_program(p, path, list(argv), p.env, None))
        return p

    def run(self, argv, fds=None, env=None, cwd=None, uid=None, missing_status=127):
        """Run a program and wait for it. Returns its exit status."""
        p = self.spawn(argv, fds=fds, env=env, cwd=cwd, uid=uid)
        if p is None:
            self.err(f"{argv[0]}: No such file or directory" if "/" in argv[0] else f"'{argv[0]}': No such file or directory")
            return missing_status
        yield from self.wait([p])
        return p.status

    def wait(self, procs):
        while any(p.state != "Z" for p in procs):
            yield ("wait", procs)

    def run_shell(self, script: str, fds=None, env=None, argv0="sh", args=()):
        """sh -c SCRIPT (used by xargs, find -exec sh -c, awk system())."""
        return (yield from self.run(["sh", "-c", script, argv0, *args], fds=fds, env=env))


class LineReader:
    def __init__(self, kernel, of):
        self.k, self.of = kernel, of
        self.buf = bytearray()
        self.eof = False
        if of.kind == "file":
            node = of.inode
            self.buf = bytearray(node.data[of.pos:])
            of.pos = len(node.data)
            self.eof = True

    def readline(self, delim: bytes = b"\n"):
        while True:
            j = self.buf.find(delim)
            if j >= 0:
                line = bytes(self.buf[:j + len(delim)])
                del self.buf[:j + len(delim)]
                return line
            if self.eof:
                line = bytes(self.buf)
                self.buf.clear()
                return line
            data = yield from read(self.k, self.of, 65536)
            if not data:
                self.eof = True
            else:
                self.buf += data

    def read_all(self):
        while not self.eof:
            data = yield from read(self.k, self.of, 65536)
            if not data:
                self.eof = True
            else:
                self.buf += data
        data = bytes(self.buf)
        self.buf.clear()
        return data

    def lines(self):
        """All lines (blocking until EOF)."""
        data = yield from self.read_all()
        return data.decode("utf-8", "surrogateescape").splitlines(True)


def find_in_path(fs, name: str, path: str, cwd: str):
    if "/" in name:
        return name
    for d in path.split(":"):
        cand = (d or ".").rstrip("/") + "/" + name
        try:
            node = fs.lookup(cand, cwd)
        except FSError:
            continue
        if node.type == "f" and node.mode & 0o111:
            return cand
    return None


def gnu_getopt(args, short: str, long=(), permute=True, name="?"):
    """GNU getopt_long. short: "abn:" (colon = argument, double colon = optional). long: ["all", "key=", "color=?"].
    -> (opts [(opt, value)], operands). Raises UsageError with GNU's message."""
    longs = {}
    for spec in long:
        if spec.endswith("=?"):
            longs[spec[:-2]] = "opt"
        elif spec.endswith("="):
            longs[spec[:-1]] = "req"
        else:
            longs[spec] = "no"
    takes = {}
    i = 0
    while i < len(short):
        c = short[i]
        if i + 2 < len(short) + 1 and short[i + 1:i + 3] == "::":
            takes[c] = "opt"
            i += 3
        elif short[i + 1:i + 2] == ":":
            takes[c] = "req"
            i += 2
        else:
            takes[c] = "no"
            i += 1
    opts, operands = [], []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--":
            operands.extend(args[i + 1:])
            break
        if a.startswith("--") and longs:
            body = a[2:]
            nm, eq, val = body.partition("=")
            cands = [k for k in longs if k.startswith(nm)]
            if nm in longs:
                cands = [nm]
            if not cands:
                raise UsageError(f"unrecognized option '{a}'")
            if len(cands) > 1:
                raise UsageError(f"option '--{nm}' is ambiguous; possibilities:" + "".join(f" '--{c}'" for c in cands))
            nm = cands[0]
            kind = longs[nm]
            if kind == "no":
                if eq:
                    raise UsageError(f"option '--{nm}' doesn't allow an argument")
                opts.append(("--" + nm, None))
            elif kind == "req":
                if not eq:
                    if i + 1 >= len(args):
                        raise UsageError(f"option '--{nm}' requires an argument")
                    i += 1
                    val = args[i]
                opts.append(("--" + nm, val))
            else:
                opts.append(("--" + nm, val if eq else None))
            i += 1
            continue
        if a.startswith("-") and len(a) > 1:
            j = 1
            while j < len(a):
                c = a[j]
                kind = takes.get(c)
                if kind is None:
                    raise UsageError(f"invalid option -- '{c}'")
                if kind == "no":
                    opts.append(("-" + c, None))
                    j += 1
                    continue
                rest = a[j + 1:]
                if rest:
                    opts.append(("-" + c, rest))
                elif kind == "req":
                    if i + 1 >= len(args):
                        raise UsageError(f"option requires an argument -- '{c}'")
                    i += 1
                    opts.append(("-" + c, args[i]))
                else:
                    opts.append(("-" + c, None))
                break
            i += 1
            continue
        if not permute:
            operands.extend(args[i:])
            break
        operands.append(a)
        i += 1
    return opts, operands


def exec_program(proc, path, argv, env, shell=None, depth=0):
    """Generator: become the program at `path`. Returns its exit status."""
    fs = proc.host.fs
    prefix = shell.prefix() if shell is not None else ""
    shown = argv[0] if argv else path

    def fail(msg, status):
        of = proc.fds.get(2)
        if of is not None:
            try:
                write_nb(proc.kernel, of, f"{prefix}{shown}: {msg}\n".encode())
            except (FSError, Signal):
                pass
        return status
    try:
        node = fs.lookup(path, proc.cwd, True, proc.uid, proc.gids)
    except FSError as e:
        return fail(e.msg, 127 if e.code == "ENOENT" else 126)
    if node.type == "d":
        return fail("Is a directory", 126)
    if not fs.can(node, 1, proc.uid, proc.gids) or not node.mode & 0o111:
        return fail("Permission denied", 126)
    data = bytes(node.data[:4096])
    prog = program_of(data)
    proc.env = env
    if prog is not None:
        fn = UTILS.get(prog)
        if fn is None:
            return fail("cannot execute binary file: Exec format error", 126)
        if not proc.argv:
            proc.argv = list(argv)
        ctx = Ctx(proc, argv, env, shell)
        ctx.path = path
        try:
            status = yield from fn(ctx)
        except FSError as e:
            if e.code == "EBADF":
                ctx.e(f"{ctx.name}: write error: Bad file descriptor\n")
                return 1
            raise
        except ProcExit as e:
            status = e.status
        try:
            yield from ctx.flush()
        except FSError:
            ctx.e(f"{ctx.name}: write error: Bad file descriptor\n")
            return 1
        return status if isinstance(status, int) else 0
    if data.startswith(b"#!"):
        line = data[2:].split(b"\n", 1)[0].decode("utf-8", "replace").strip()
        parts = line.split(None, 1)
        if not parts:
            return fail("cannot execute: required file not found", 126)
        interp = parts[0]
        optarg = parts[1] if len(parts) > 1 else None
        new_argv = [interp] + ([optarg] if optarg else []) + [path] + list(argv[1:])
        if depth > 4:
            return fail("too many levels of interpreters", 126)
        try:
            inode = fs.lookup(interp, proc.cwd)
            if inode.type != "f":
                raise FSError("EACCES")
        except FSError:
            return fail(f"{interp}: bad interpreter: No such file or directory", 126)
        proc.argv = [interp.rsplit("/", 1)[-1]] + new_argv[1:]
        return (yield from exec_program(proc, interp, new_argv, env, None, depth + 1))
    # no #! line: bash runs it as a shell script
    proc.argv = ["bash"] + list(argv)
    return (yield from exec_program(proc, "/bin/bash", ["bash", path] + list(argv[1:]), env, None, depth + 1))
