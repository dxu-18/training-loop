"""Reading, writing and opening files for simulated processes (generators: `x = yield from read(...)`)."""

from __future__ import annotations

import random

from .kernel import SIGPIPE, Interrupted, OpenFile, Signal
from .vfs import FSError


def write(kernel, of: OpenFile | None, data: bytes):
    if of is None or not of.wr:
        raise FSError("EBADF")
    k = of.kind
    if k == "tty":
        of.tty.write(of.stream, data)
        return
    if k == "file":
        node = of.inode
        if of.append:
            of.pos = len(node.data)
        end = of.pos + len(data)
        if of.fs is not None or node is not None:
            fs = of.fs or getattr(kernel.current, "host", None) and kernel.current.host.fs
            if fs is not None:
                fs.charge(max(0, end - len(node.data)))
        if of.pos > len(node.data):
            node.data.extend(b"\0" * (of.pos - len(node.data)))
        node.data[of.pos:end] = data
        of.pos = end
        node.mtime = node.ctime = kernel.now
        return
    if k in ("null", "zero", "rand"):
        return
    if k == "pipe":
        p = of.pipe
        while data:
            if p is None or p.readers == 0:
                raise Signal(SIGPIPE)
            space = p.CAP - len(p.buf)
            if space <= 0:
                yield ("write", p)
                continue
            p.buf += data[:space]
            data = data[space:]
            kernel.wake(p)
        return
    raise FSError("EBADF")


def write_nb(kernel, of: OpenFile | None, data: bytes) -> bool:
    """Non-blocking write for destinations that never block (tty, file, null). False if it would need to wait."""
    if of is None or not of.wr:
        raise FSError("EBADF")
    k = of.kind
    if k == "pipe":
        p = of.pipe
        if p is None or p.readers == 0:
            raise Signal(SIGPIPE)
        p.buf += data            # stderr into a pipe: small, allowed past the capacity
        kernel.wake(p)
        return True
    gen = write(kernel, of, data)
    try:
        next(gen)
    except StopIteration:
        return True
    return False


def read(kernel, of: OpenFile | None, n: int = -1):
    """Up to n bytes (all available if n < 0); b"" at end of file."""
    if of is None or not of.rd:
        raise FSError("EBADF")
    k = of.kind
    if k == "file":
        node = of.inode
        if node.type == "d":
            raise FSError("EISDIR")
        data = node.data[of.pos:] if n < 0 else node.data[of.pos:of.pos + n]
        of.pos += len(data)
        return bytes(data)
    if k == "pipe":
        while True:
            p = of.pipe
            if p is None:
                return b""
            if p.buf:
                if n < 0 or n >= len(p.buf):
                    data = bytes(p.buf)
                    p.buf.clear()
                else:
                    data = bytes(p.buf[:n])
                    del p.buf[:n]
                kernel.wake(p)
                return data
            if p.writers == 0:
                return b""
            yield ("read", p)
    if k == "null":
        return b""
    if k == "zero":
        return b"\0" * (n if n > 0 else 65536)
    if k == "rand":
        if of.rng is None:
            of.rng = random.Random(kernel.world.seed * 7919 + 17)
        m = n if n > 0 else 4096
        return bytes(of.rng.getrandbits(8) for _ in range(m))
    if k == "tty":
        tty = of.tty
        if _background_reader(kernel, tty):
            return b""
        while True:
            if tty.input:
                data = bytes(tty.input if n < 0 else tty.input[:n])
                del tty.input[:len(data)]
                return data
            if tty.take_line():
                continue
            if tty.input_eof:
                return b""
            if tty.eof_pending:
                tty.eof_pending = False
                return b""
            _check_interrupt(kernel)
            yield ("tty", tty)
    if k == "dir":
        raise FSError("EISDIR")
    raise FSError("EBADF")


def read_all(kernel, of):
    chunks = []
    while True:
        data = yield from read(kernel, of)
        if not data:
            return b"".join(chunks)
        chunks.append(data)


def readline(kernel, of, delim: bytes = b"\n"):
    """One line including the delimiter (b"" at EOF). Never reads past the delimiter."""
    if of is None or not of.rd:
        raise FSError("EBADF")
    k = of.kind
    if k == "file":
        node = of.inode
        if node.type == "d":
            raise FSError("EISDIR")
        j = node.data.find(delim, of.pos)
        end = len(node.data) if j < 0 else j + len(delim)
        data = bytes(node.data[of.pos:end])
        of.pos = end
        return data
    if k == "pipe":
        out = bytearray()
        while True:
            p = of.pipe
            if p is None:
                return bytes(out)
            if p.buf:
                j = p.buf.find(delim)
                if j >= 0:
                    out += p.buf[:j + len(delim)]
                    del p.buf[:j + len(delim)]
                    kernel.wake(p)
                    return bytes(out)
                out += p.buf
                p.buf.clear()
                kernel.wake(p)
                continue
            if p.writers == 0:
                return bytes(out)
            yield ("read", p)
    if k == "tty":
        tty = of.tty
        if _background_reader(kernel, tty):
            return b""
        out = bytearray()
        while True:
            if tty.input:
                j = tty.input.find(delim)
                if j >= 0:
                    out += tty.input[:j + len(delim)]
                    del tty.input[:j + len(delim)]
                    return bytes(out)
                out += tty.input
                tty.input.clear()
                continue
            if tty.take_line():
                continue
            if tty.input_eof:
                return bytes(out)
            if tty.eof_pending:
                tty.eof_pending = False
                return bytes(out)
            _check_interrupt(kernel)
            yield ("tty", tty)
    data = yield from read(kernel, of, 1 << 16)
    return data


def _check_interrupt(kernel):
    """`read` in an interactive shell: Ctrl-C abandons it (and the command line)."""
    sh = getattr(kernel.current, "shell", None)
    if sh is not None and sh.interrupted:
        sh.interrupted = False
        raise Interrupted()


def _background_reader(kernel, tty) -> bool:
    """A background job that reads the terminal is stopped with SIGTTIN (job control)."""
    p = kernel.current
    if p is None or not tty.fg_pgid or p.pgid == tty.fg_pgid or p.shell is not None and p.shell.interactive:
        return False
    kernel.signal(p, 21)
    return True


DEVICES = {"/dev/null": "null", "/dev/zero": "zero", "/dev/urandom": "rand", "/dev/random": "rand", "/dev/full": "null"}


def open_path(proc, path: str, mode: str = "r", noclobber: bool = False, excl: bool = False) -> OpenFile:
    """Open `path` for a process. mode: r, w (truncate), a (append), rw (read+write, create). Raises FSError."""
    fs = proc.host.fs
    dev = None
    if path in DEVICES:
        dev = DEVICES[path]
    elif path == "/dev/tty":
        tty = proc.kernel.world.tty_of(proc)
        if tty is None:
            raise FSError("ENXIO")
        of = OpenFile("tty", rd="r" in mode, wr=mode != "r", tty=tty, path=path, stream="out")
        return of
    else:
        fdnum = _fd_path(path)
        if fdnum is not None:
            src = proc.fds.get(fdnum)
            if src is None:
                raise FSError("ENOENT")
            if src.kind == "file" and mode != "r":
                of = OpenFile("file", rd="r" in mode, wr=True, inode=src.inode, path=src.path, append=mode == "a")
                of.pos = len(src.inode.data) if mode == "a" else 0
                if mode == "w":
                    src.inode.data.clear()
                return of
            return src   # pipes and terminals: the same description (like Linux's /proc/self/fd)
    if dev is not None:
        return OpenFile(dev, rd=True, wr=True, path=path)
    uid, gids = proc.uid, proc.gids
    if mode == "r":
        node = fs.lookup(path, proc.cwd, True, uid, gids)
        if node.type == "d":
            of = OpenFile("dir", rd=True, inode=node, path=path)
            return of
        if not fs.can(node, 4, uid, gids):
            raise FSError("EACCES")
        if node.type == "c":
            return OpenFile(node.dev or "null", rd=True, wr=True, path=path)
        if node.type == "p":
            raise FSError("ENXIO")
        of = OpenFile("file", rd=True, inode=node, path=path)
        node.atime = proc.kernel.now
        return of
    # writing
    try:
        node = fs.lookup(path, proc.cwd, True, uid, gids)
        exists = True
    except FSError as e:
        if e.code != "ENOENT":
            raise
        node, exists = None, False
    if exists:
        if excl:
            raise FSError("EEXIST")
        if node.type == "d":
            raise FSError("EISDIR")
        if node.type == "c":
            return OpenFile(node.dev or "null", rd=True, wr=True, path=path)
        if noclobber and mode == "w" and node.type == "f":
            raise FSError("NOCLOBBER")
        if not fs.can(node, 2, uid, gids):
            raise FSError("EACCES")
    else:
        node = fs.create(path, proc.cwd, 0o666 & ~proc.umask, uid, proc.gid, gids)
    of = OpenFile("file", rd="r" in mode, wr=True, inode=node, path=path, append=mode == "a")
    if mode == "w":
        node.data.clear()
        node.mtime = node.ctime = proc.kernel.now
    elif mode == "a":
        of.pos = len(node.data)
    return of


def _fd_path(path: str):
    for prefix in ("/dev/fd/", "/proc/self/fd/"):
        if path.startswith(prefix) and path[len(prefix):].isdigit():
            return int(path[len(prefix):])
    return {"/dev/stdin": 0, "/dev/stdout": 1, "/dev/stderr": 2}.get(path)


def anon_file(data: bytes, now: int = 0) -> OpenFile:
    """A read-only in-memory file (here-documents, here-strings)."""
    from .vfs import Inode
    node = Inode(0, "f", 0o600, 0, 0, now)
    node.data[:] = data
    return OpenFile("file", rd=True, inode=node, path="")
