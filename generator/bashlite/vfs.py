"""An in-memory POSIX-like filesystem: inodes, directories, symlinks, devices, owners, modes and times.

Paths are resolved component by component (symlinks followed like the kernel does, so `..` after a
symlink is physical). Directory listings come back sorted by name (bytewise), which keeps every
command's output deterministic.
"""

from __future__ import annotations

import posixpath

MESSAGES = {
    "ENOENT": "No such file or directory", "EEXIST": "File exists", "ENOTDIR": "Not a directory",
    "EISDIR": "Is a directory", "EACCES": "Permission denied", "ENOTEMPTY": "Directory not empty",
    "EINVAL": "Invalid argument", "ELOOP": "Too many levels of symbolic links", "EPERM": "Operation not permitted",
    "EBUSY": "Device or resource busy", "EXDEV": "Invalid cross-device link", "EROFS": "Read-only file system",
    "EBADF": "Bad file descriptor", "ENOSPC": "No space left on device", "ENAMETOOLONG": "File name too long",
    "ESPIPE": "Illegal seek", "ENXIO": "No such device or address", "ETXTBSY": "Text file busy",
}


class FSError(Exception):
    def __init__(self, code: str, path: str = ""):
        super().__init__(code)
        self.code = code
        self.path = path

    @property
    def msg(self) -> str:
        return MESSAGES.get(self.code, self.code)

    def __str__(self) -> str:
        return self.msg


BLOCK = 4096


class Inode:
    __slots__ = ("ino", "type", "mode", "uid", "gid", "atime", "mtime", "ctime", "nlink", "data", "entries",
                 "target", "parent", "dev", "xattr")

    def __init__(self, ino: int, type_: str, mode: int, uid: int, gid: int, now: int):
        self.ino, self.type, self.mode, self.uid, self.gid = ino, type_, mode, uid, gid
        self.atime = self.mtime = self.ctime = now
        self.nlink = 2 if type_ == "d" else 1
        self.data = bytearray() if type_ == "f" else None
        self.entries: dict | None = {} if type_ == "d" else None
        self.target = ""        # symlink target
        self.parent = None      # directories only
        self.dev = ""           # device name for type "c" (null, zero, urandom, tty)
        self.xattr = None       # simulator metadata (e.g. the program a /usr/bin stub runs)

    @property
    def size(self) -> int:
        if self.type == "f":
            return len(self.data)
        if self.type == "d":
            return BLOCK
        if self.type == "l":
            return len(self.target.encode())
        return 0

    @property
    def blocks(self) -> int:
        """512-byte blocks allocated (ext4-style 4 KiB allocation units)."""
        if self.type == "f":
            return -(-len(self.data) // BLOCK) * 8
        if self.type == "d":
            return 8
        return 0

    def is_dir(self) -> bool:
        return self.type == "d"


TYPE_BITS = {"f": 0o100000, "d": 0o040000, "l": 0o120000, "c": 0o020000, "p": 0o010000, "s": 0o140000}
TYPE_CHAR = {"f": "-", "d": "d", "l": "l", "c": "c", "p": "p", "s": "s"}


def mode_string(node: Inode) -> str:
    """-rwxr-x--- style string, with setuid/setgid/sticky letters."""
    m = node.mode
    out = [TYPE_CHAR[node.type]]
    for shift, special, ch in ((6, 0o4000, "s"), (3, 0o2000, "s"), (0, 0o1000, "t")):
        r = "r" if m & (4 << shift) else "-"
        w = "w" if m & (2 << shift) else "-"
        x = m & (1 << shift)
        if m & special:
            x = ch if x else ch.upper()
        else:
            x = "x" if x else "-"
        out += [r, w, x]
    return "".join(out)


class FS:
    """One host's filesystem. `clock()` returns the current time in nanoseconds."""

    def __init__(self, clock):
        self.clock = clock
        self._ino = 1
        self.root = self._new("d", 0o755, 0, 0)
        self.root.parent = self.root
        self.mounts: list[tuple[str, str, str, int]] = []   # (device, mountpoint, fstype, size bytes)

    def _new(self, type_: str, mode: int, uid: int, gid: int) -> Inode:
        self._ino += 1
        return Inode(self._ino, type_, mode, uid, gid, self.clock())

    # ------------------------------------------------------------ resolution

    def _split(self, path: str, cwd: str) -> tuple[Inode, list[str]]:
        if not path:
            raise FSError("ENOENT", path)
        if path.startswith("/"):
            return self.root, path.split("/")
        base, comps = self._split(cwd, "/") if cwd != "/" else (self.root, [])
        node = self._walk(base, comps, True, 0, 0, None)
        return node, path.split("/")

    def _walk(self, node: Inode, comps: list[str], follow_last: bool, depth: int, uid: int, gids) -> Inode:
        comps = [c for c in comps]
        n = len(comps)
        for i, c in enumerate(comps):
            if c in ("", "."):
                if node.type != "d":
                    raise FSError("ENOTDIR")
                continue
            if node.type != "d":
                raise FSError("ENOTDIR")
            if gids is not None and not self.can(node, 1, uid, gids):
                raise FSError("EACCES")
            if c == "..":
                node = node.parent
                continue
            child = node.entries.get(c)
            if child is None:
                raise FSError("ENOENT")
            last = all(x in ("", ".") for x in comps[i + 1:])
            trailing = last and i < n - 1   # "link/" always follows
            if child.type == "l" and (not last or follow_last or trailing):
                if depth > 40:
                    raise FSError("ELOOP")
                t = child.target
                start, tcomps = (self.root, t.split("/")) if t.startswith("/") else (node, t.split("/"))
                child = self._walk(start, tcomps, True, depth + 1, uid, gids)
            node = child
        return node

    def lookup(self, path: str, cwd: str = "/", follow: bool = True, uid: int = 0, gids=None) -> Inode:
        """The inode at `path` (FSError on failure). Pass uid/gids to enforce search permission."""
        base, comps = self._split(path, cwd)
        return self._walk(base, comps, follow, 0, uid, gids)

    def exists(self, path: str, cwd: str = "/", follow: bool = True) -> bool:
        try:
            self.lookup(path, cwd, follow)
            return True
        except FSError:
            return False

    def parent_of(self, path: str, cwd: str = "/", uid: int = 0, gids=None) -> tuple[Inode, str]:
        """(directory inode, final name) for creating/removing `path`."""
        stripped = path.rstrip("/")
        if not stripped:
            return self.root, "."
        head, _, name = stripped.rpartition("/")
        if not _:
            head = cwd
        elif not head:
            head = "/"
        else:
            head = head if stripped.startswith("/") else posixpath.join(cwd, head)
        d = self.lookup(head, "/", True, uid, gids)
        if d.type != "d":
            raise FSError("ENOTDIR")
        return d, name

    def realpath(self, path: str, cwd: str = "/") -> str:
        """Canonical absolute path (all symlinks resolved); FSError if a component is missing."""
        node = self.lookup(path, cwd)
        return self.path_of(node)

    def path_of(self, node: Inode) -> str:
        parts = []
        while node is not self.root:
            parent = node.parent
            if parent is None:
                raise FSError("ENOENT")
            name = next((k for k, v in parent.entries.items() if v is node), None)
            if name is None:
                raise FSError("ENOENT")
            parts.append(name)
            node = parent
        return "/" + "/".join(reversed(parts))

    def find_name(self, parent: Inode, node: Inode) -> str | None:
        return next((k for k, v in parent.entries.items() if v is node), None)

    # ------------------------------------------------------------ permissions

    @staticmethod
    def can(node: Inode, want: int, uid: int, gids) -> bool:
        """want: 4 read, 2 write, 1 execute/search."""
        if uid == 0:
            return want != 1 or node.type == "d" or bool(node.mode & 0o111)
        if node.uid == uid:
            bits = (node.mode >> 6) & 7
        elif node.gid in gids:
            bits = (node.mode >> 3) & 7
        else:
            bits = node.mode & 7
        return bits & want == want

    # ------------------------------------------------------------ creation

    def mkdir(self, path: str, cwd: str = "/", mode: int = 0o755, uid: int = 0, gid: int = 0, parents=False,
              gids=None) -> Inode:
        if parents:
            node = self.root if path.startswith("/") else self.lookup(cwd)
            for c in path.split("/"):
                if c in ("", "."):
                    continue
                if c == "..":
                    node = node.parent
                    continue
                child = node.entries.get(c)
                if child is not None and child.type == "l":
                    child = self.lookup(self.path_of(node) + "/" + c)
                if child is None:
                    if gids is not None and not self.can(node, 3, uid, gids):
                        raise FSError("EACCES")
                    child = self._link_new(node, c, "d", mode, uid, gid)
                elif child.type != "d":
                    raise FSError("EEXIST" if c == path.rstrip("/").split("/")[-1] else "ENOTDIR")
                node = child
            return node
        d, name = self.parent_of(path, cwd, uid, gids)
        if name in d.entries or name in (".", ".."):
            raise FSError("EEXIST")
        if gids is not None and not self.can(d, 3, uid, gids):
            raise FSError("EACCES")
        return self._link_new(d, name, "d", mode, uid, gid)

    def _link_new(self, d: Inode, name: str, type_: str, mode: int, uid: int, gid: int) -> Inode:
        node = self._new(type_, mode, uid, gid)
        d.entries[name] = node
        if type_ == "d":
            node.parent = d
            d.nlink += 1
        d.mtime = d.ctime = self.clock()
        return node

    def create(self, path: str, cwd: str = "/", mode: int = 0o644, uid: int = 0, gid: int = 0, gids=None,
               excl: bool = False) -> Inode:
        """Open-for-write semantics: return the existing file (following symlinks) or create it."""
        try:
            node = self.lookup(path, cwd, True, uid, gids)
            if excl:
                raise FSError("EEXIST")
            if node.type == "d":
                raise FSError("EISDIR")
            return node
        except FSError as e:
            if e.code != "ENOENT":
                raise
        if path.endswith("/"):
            raise FSError("EISDIR")
        d, name = self.parent_of(path, cwd, uid, gids)
        if name in d.entries:   # dangling symlink: create its target
            link = d.entries[name]
            if link.type == "l":
                t = link.target if link.target.startswith("/") else self.path_of(d) + "/" + link.target
                return self.create(t, "/", mode, uid, gid, gids)
        if gids is not None and not self.can(d, 3, uid, gids):
            raise FSError("EACCES")
        return self._link_new(d, name, "f", mode, uid, gid)

    def write_file(self, path: str, data, cwd: str = "/", mode: int = 0o644, uid: int = 1000, gid: int = 1000,
                   mtime: int | None = None) -> Inode:
        """Setup helper: create parents, write the whole file."""
        if isinstance(data, str):
            data = data.encode("utf-8", "surrogateescape")
        head = posixpath.dirname(posixpath.join(cwd, path))
        self.mkdir(head, "/", 0o755, uid, gid, parents=True)
        node = self.create(path, cwd, mode, uid, gid)
        node.data[:] = data
        node.mtime = node.ctime = self.clock() if mtime is None else mtime
        return node

    def symlink(self, target: str, path: str, cwd: str = "/", uid: int = 0, gid: int = 0, gids=None) -> Inode:
        d, name = self.parent_of(path, cwd, uid, gids)
        if name in d.entries:
            raise FSError("EEXIST")
        if gids is not None and not self.can(d, 3, uid, gids):
            raise FSError("EACCES")
        node = self._link_new(d, name, "l", 0o777, uid, gid)
        node.target = target
        return node

    def link(self, src: str, path: str, cwd: str = "/", uid: int = 0, gids=None) -> Inode:
        node = self.lookup(src, cwd, False, uid, gids)
        if node.type == "d":
            raise FSError("EPERM")
        d, name = self.parent_of(path, cwd, uid, gids)
        if name in d.entries:
            raise FSError("EEXIST")
        d.entries[name] = node
        node.nlink += 1
        node.ctime = d.mtime = self.clock()
        return node

    def mknod(self, path: str, dev: str, mode: int = 0o666) -> Inode:
        d, name = self.parent_of(path)
        node = self._link_new(d, name, "c", mode, 0, 0)
        node.dev = dev
        return node

    # ------------------------------------------------------------ removal / rename

    def unlink(self, path: str, cwd: str = "/", uid: int = 0, gids=None) -> None:
        d, name = self.parent_of(path, cwd, uid, gids)
        node = d.entries.get(name)
        if node is None:
            raise FSError("ENOENT")
        if node.type == "d":
            raise FSError("EISDIR")
        self._check_delete(d, node, uid, gids)
        del d.entries[name]
        node.nlink -= 1
        d.mtime = d.ctime = self.clock()

    def rmdir(self, path: str, cwd: str = "/", uid: int = 0, gids=None) -> None:
        d, name = self.parent_of(path, cwd, uid, gids)
        if name in (".", ".."):
            raise FSError("EINVAL" if name == "." else "ENOTEMPTY")
        node = d.entries.get(name)
        if node is None:
            raise FSError("ENOENT")
        if node.type != "d":
            raise FSError("ENOTDIR")
        if node.entries:
            raise FSError("ENOTEMPTY")
        if node is self.root:
            raise FSError("EBUSY")
        self._check_delete(d, node, uid, gids)
        del d.entries[name]
        d.nlink -= 1
        d.mtime = d.ctime = self.clock()

    def _check_delete(self, d: Inode, node: Inode, uid: int, gids) -> None:
        if gids is None or uid == 0:
            return
        if not self.can(d, 3, uid, gids):
            raise FSError("EACCES")
        if d.mode & 0o1000 and node.uid != uid and d.uid != uid:   # sticky /tmp
            raise FSError("EPERM")

    def rename(self, src: str, dst: str, cwd: str = "/", uid: int = 0, gids=None) -> None:
        sd, sname = self.parent_of(src, cwd, uid, gids)
        node = sd.entries.get(sname)
        if node is None:
            raise FSError("ENOENT")
        dd, dname = self.parent_of(dst, cwd, uid, gids)
        if node.type == "d":   # can't move a directory inside itself
            p = dd
            while True:
                if p is node:
                    raise FSError("EINVAL")
                if p is self.root:
                    break
                p = p.parent
        existing = dd.entries.get(dname)
        if existing is node:
            return
        if existing is not None:
            if existing.type == "d":
                if node.type != "d":
                    raise FSError("EISDIR")
                if existing.entries:
                    raise FSError("ENOTEMPTY")
                dd.nlink -= 1
            elif node.type == "d":
                raise FSError("ENOTDIR")
            existing.nlink -= 1
        if gids is not None:
            self._check_delete(sd, node, uid, gids)
            if not self.can(dd, 3, uid, gids):
                raise FSError("EACCES")
        del sd.entries[sname]
        dd.entries[dname] = node
        if node.type == "d":
            node.parent = dd
            sd.nlink -= 1
            dd.nlink += 1
        now = self.clock()
        sd.mtime = sd.ctime = dd.mtime = dd.ctime = node.ctime = now

    # ------------------------------------------------------------ listing / walking

    @staticmethod
    def listdir(node: Inode) -> list[str]:
        return sorted(node.entries, key=lambda s: s.encode("utf-8", "surrogateescape"))

    def walk(self, node: Inode, prefix: str):
        """Yield (path, inode) for node and everything under it, depth first, names sorted."""
        yield prefix, node
        if node.type == "d":
            for name in self.listdir(node):
                child = node.entries[name]
                yield from self.walk(child, prefix.rstrip("/") + "/" + name if prefix != "/" else "/" + name)

    def du_blocks(self, node: Inode, seen=None) -> int:
        """512-byte blocks used by a tree (hard links counted once)."""
        seen = set() if seen is None else seen
        if node.ino in seen:
            return 0
        seen.add(node.ino)
        total = node.blocks
        if node.type == "d":
            for child in node.entries.values():
                total += self.du_blocks(child, seen)
        return total

    def tree_size(self, node: Inode) -> int:
        total = node.size if node.type != "d" else 0
        if node.type == "d":
            for child in node.entries.values():
                total += self.tree_size(child)
        return total

    def copy_tree(self, node: Inode, into: "FS", path: str, uid: int, gid: int, keep: bool = True) -> Inode:
        """Deep copy `node` to `path` in `into` (possibly another host's filesystem)."""
        if node.type == "d":
            new = into.mkdir(path, "/", node.mode, uid, gid, parents=True)
            for name in self.listdir(node):
                self.copy_tree(node.entries[name], into, path.rstrip("/") + "/" + name, uid, gid, keep)
        elif node.type == "l":
            if into.exists(path, follow=False):
                into.unlink(path)
            new = into.symlink(node.target, path, "/", uid, gid)
        else:
            new = into.create(path, "/", node.mode, uid, gid)
            new.data[:] = node.data or b""
            new.mode = node.mode
        if keep:
            new.mtime, new.atime = node.mtime, node.atime
        return new
