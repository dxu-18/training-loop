"""Git LFS: big files stored outside git's history. A tracked file (`git lfs track "*.pt"` writes the pattern to
.gitattributes with filter=lfs) is committed as a small pointer; its content goes to .git/lfs/objects and, on push,
to the LFS store of the remote (GitHub-lite). The clean/smudge filter only runs once `git lfs install` has set
filter.lfs.* in the global config, as with the real git-lfs."""

from __future__ import annotations

import hashlib
import re

from ..vfs import LazyData

POINTER_HEAD = b"version https://git-lfs.github.com/spec/v1\n"


def active(repo) -> bool:
    cfg = repo.config
    return bool(cfg.get("filter.lfs.clean") or cfg.get("filter.lfs.process"))


def oid_of(data) -> str:
    if isinstance(data, LazyData) and not data.materialized and data._src is None:
        # a generated big file: its LFS id comes from its recipe (hashing 100 MB in the browser is slow)
        return hashlib.sha256(f"lfs:{data._seed}:{len(data)}".encode()).hexdigest()
    return hashlib.sha256(bytes(data)).hexdigest()


def pointer_text(oid: str, size: int) -> bytes:
    return POINTER_HEAD + f"oid sha256:{oid}\nsize {size}\n".encode()


def parse_pointer(data) -> tuple[str, int] | None:
    head = bytes(data[:300])
    if not head.startswith(POINTER_HEAD):
        return None
    m = re.search(rb"oid sha256:([0-9a-f]{64})\nsize (\d+)", head)
    if not m:
        return None
    return m.group(1).decode(), int(m.group(2))


def store(world) -> dict:
    s = world.__dict__.get("lfs_store")
    if s is None:
        s = {}
        world.lfs_store = s
    return s


def local_path(repo, oid: str) -> str:
    return f"lfs/objects/{oid[:2]}/{oid[2:4]}/{oid}"


def pointer_for(repo, data):
    """The clean filter: keep the content in .git/lfs/objects, return the pointer to commit."""
    if not active(repo):
        return data
    oid = oid_of(data)
    size = len(data)
    store(repo.ctx.world)[oid] = data if isinstance(data, LazyData) else LazyData(src=bytes(data))
    p = repo.path(local_path(repo, oid), True)
    if not repo.fs.exists(p):
        node = repo.fs.write_file(p, b"", "/", 0o644, repo.ctx.uid, repo.ctx.proc.gid)
        node.data = data.clone() if isinstance(data, LazyData) else LazyData(src=bytes(data))
    return pointer_text(oid, size)


def have_locally(repo, oid: str) -> bool:
    return repo.fs.exists(repo.path(local_path(repo, oid), True))


def smudge(repo, data):
    """The smudge filter: a pointer becomes its content when that content is here (else the pointer stays)."""
    if not active(repo):
        return data
    p = parse_pointer(data)
    if p is None:
        return data
    oid, size = p
    obj = store(repo.ctx.world).get(oid)
    if obj is None:
        return data
    if not have_locally(repo, oid):
        node = repo.fs.write_file(repo.path(local_path(repo, oid), True), b"", "/", 0o644, repo.ctx.uid,
                                  repo.ctx.proc.gid)
        node.data = obj.clone() if isinstance(obj, LazyData) else LazyData(src=bytes(obj))
    return obj.clone() if isinstance(obj, LazyData) else obj
