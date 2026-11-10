"""Filesystems for images and containers: an image keeps a frozen filesystem, a container works on a copy of it
(file contents are shared copy-on-write, so a copy costs inodes, not bytes). Bind mounts and volumes graft a node
of another filesystem (the host's) into the container's tree."""

from __future__ import annotations

from ..vfs import FS, LazyData


def clone_node(src, dst_fs: FS, parent=None):
    """A deep copy of `src` made of new inodes in `dst_fs` (file data shared until written)."""
    n = dst_fs._new(src.type, src.mode, src.uid, src.gid)
    n.atime, n.mtime, n.ctime = src.atime, src.mtime, src.ctime
    n.target, n.dev, n.xattr = src.target, src.dev, src.xattr
    if src.type == "f":
        d = src.data
        if isinstance(d, LazyData):
            n.data = d.clone()
        elif len(d) > 4096:
            n.data = LazyData(src=bytes(d))
        else:
            n.data = bytearray(d)
    elif src.type == "d":
        n.parent = parent if parent is not None else n
        for name, child in src.entries.items():
            n.entries[name] = clone_node(child, dst_fs, n)
    return n


def clone_fs(src: FS, clock) -> FS:
    fs = FS(clock)
    fs.root = clone_node(src.root, fs)
    fs.root.parent = fs.root
    return fs


def tree_bytes(node) -> int:
    if node.type == "f":
        return len(node.data)
    if node.type == "d":
        return sum(tree_bytes(c) for c in node.entries.values())
    return 0


def diff_bytes(old, new) -> int:
    """Bytes a layer adds: files new or changed in `new` compared with `old` (both directory inodes or None)."""
    if new is None:
        return 0
    if old is None or old.type != new.type:
        return tree_bytes(new)
    if new.type == "f":
        if old.data is new.data or (isinstance(old.data, LazyData) and isinstance(new.data, LazyData) and
                                    old.data.oid and old.data.oid == new.data.oid):
            return 0
        if len(old.data) == len(new.data) and old.mtime == new.mtime:
            return 0
        return len(new.data)
    if new.type == "d":
        return sum(diff_bytes(old.entries.get(k), v) for k, v in new.entries.items())
    return 0


def graft(fs: FS, path: str, node, ro: bool = False):
    """Make `node` (a directory or file of another filesystem) appear at `path` in `fs` (a mount)."""
    parent_path, _, name = path.rstrip("/").rpartition("/")
    parent = fs.mkdir(parent_path or "/", "/", 0o755, 0, 0, parents=True) if parent_path else fs.root
    parent.entries[name] = node
    mounts = fs.__dict__.setdefault("binds", {})
    mounts[id(node)] = (parent, path, ro)
    return node


def mount_info(fs: FS, node):
    return getattr(fs, "binds", {}).get(id(node))
