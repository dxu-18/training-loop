"""Git objects as git stores them: blobs, trees, commits and tags, named by the SHA-1 of their real encoding (so
`git hash-object`, commit ids and tree ids are the ids real git computes for the same content, names and dates).

The object store is the world's (`odb(world)`): every repository on every simulated host, and GitHub-lite, reads
the same content-addressed table. Objects never change, so sharing is safe, and cloning or fetching only has to
copy refs. A repository's own objects are the ones reachable from its refs (and those it wrote as loose files).
"""

from __future__ import annotations

import hashlib
import zlib
from dataclasses import dataclass

from ..vfs import LazyData

EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
EMPTY_BLOB = "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"


class GitError(Exception):
    """A git-level failure: message (already in git's words) and exit status."""

    def __init__(self, msg: str, status: int = 128):
        super().__init__(msg)
        self.msg, self.status = msg, status


def hash_object(kind: str, data) -> str:
    if isinstance(data, LazyData) and data.oid and kind == "blob":
        return data.oid
    raw = bytes(data)
    return hashlib.sha1(f"{kind} {len(raw)}".encode() + b"\0" + raw).hexdigest()


@dataclass(frozen=True)
class Person:
    name: str
    email: str
    when: int          # unix seconds
    tz: str            # "+0200"

    def line(self) -> str:
        return f"{self.name} <{self.email}> {self.when} {self.tz}"

    @staticmethod
    def parse(text: str) -> "Person":
        lt, gt = text.find("<"), text.rfind(">")
        name = text[:lt].strip()
        email = text[lt + 1:gt]
        rest = text[gt + 1:].split()
        when = int(rest[0]) if rest else 0
        tz = rest[1] if len(rest) > 1 else "+0000"
        return Person(name, email, when, tz)


@dataclass
class Commit:
    sha: str
    tree: str
    parents: tuple
    author: Person
    committer: Person
    message: str
    extra: tuple = ()          # other header lines, e.g. ("gpgsig", ...), ("encoding", ...)

    @property
    def subject(self) -> str:
        lines = self.message.split("\n")
        out = []
        for ln in lines:
            if not ln.strip():
                break
            out.append(ln.strip())
        return " ".join(out)

    @property
    def body(self) -> str:
        parts = self.message.split("\n\n", 1)
        return parts[1] if len(parts) > 1 else ""


@dataclass
class Tag:
    sha: str
    object: str
    type: str
    tag: str
    tagger: Person | None
    message: str


def encode_tree(entries) -> bytes:
    """entries: [(mode str, name, sha)] -> git's tree encoding, sorted the way git sorts (a directory as name/)."""
    def key(e):
        mode, name, _ = e
        return (name + "/" if mode == "40000" else name).encode("utf-8", "surrogateescape")
    out = bytearray()
    for mode, name, sha in sorted(entries, key=key):
        out += f"{mode} {name}".encode("utf-8", "surrogateescape") + b"\0" + bytes.fromhex(sha)
    return bytes(out)


def decode_tree(data: bytes):
    """-> [(mode, name, sha)] in stored order."""
    out = []
    i = 0
    n = len(data)
    while i < n:
        sp = data.index(b" ", i)
        nul = data.index(b"\0", sp)
        mode = data[i:sp].decode()
        name = data[sp + 1:nul].decode("utf-8", "surrogateescape")
        sha = data[nul + 1:nul + 21].hex()
        out.append((mode, name, sha))
        i = nul + 21
    return out


def encode_commit(tree: str, parents, author: Person, committer: Person, message: str, extra=()) -> bytes:
    lines = [f"tree {tree}"] + [f"parent {p}" for p in parents]
    lines += [f"author {author.line()}", f"committer {committer.line()}"]
    for k, v in extra:
        vv = v.replace("\n", "\n ")
        lines.append(f"{k} {vv}")
    return ("\n".join(lines) + "\n\n" + message).encode("utf-8", "surrogateescape")


def decode_commit(sha: str, data: bytes) -> Commit:
    text = data.decode("utf-8", "surrogateescape")
    head, _, message = text.partition("\n\n")
    tree = ""
    parents = []
    author = committer = None
    extra = []
    for line in head.split("\n"):
        if line.startswith(" ") and extra:
            k, v = extra[-1]
            extra[-1] = (k, v + "\n" + line[1:])
            continue
        k, _, v = line.partition(" ")
        if k == "tree":
            tree = v
        elif k == "parent":
            parents.append(v)
        elif k == "author":
            author = Person.parse(v)
        elif k == "committer":
            committer = Person.parse(v)
        else:
            extra.append((k, v))
    return Commit(sha, tree, tuple(parents), author, committer, message, tuple(extra))


def encode_tag(obj: str, kind: str, tag: str, tagger: Person | None, message: str) -> bytes:
    lines = [f"object {obj}", f"type {kind}", f"tag {tag}"]
    if tagger is not None:
        lines.append(f"tagger {tagger.line()}")
    return ("\n".join(lines) + "\n\n" + message).encode("utf-8", "surrogateescape")


def decode_tag(sha: str, data: bytes) -> Tag:
    text = data.decode("utf-8", "surrogateescape")
    head, _, message = text.partition("\n\n")
    f = {}
    for line in head.split("\n"):
        k, _, v = line.partition(" ")
        f[k] = v
    tagger = Person.parse(f["tagger"]) if "tagger" in f else None
    return Tag(sha, f.get("object", ""), f.get("type", "commit"), f.get("tag", ""), tagger, message)


class ODB:
    """The world's object table: sha -> (kind, data). data is bytes, or LazyData for big generated blobs."""

    def __init__(self):
        self.cut: set = set()          # a shallow clone's boundary commits: their parents are out of reach
        self.objects: dict[str, tuple[str, object]] = {}
        self._commits: dict[str, Commit] = {}
        self._trees: dict[str, list] = {}
        self._flat: dict[str, dict] = {}
        self._sorted: list[str] | None = None

    def put(self, kind: str, data) -> str:
        sha = hash_object(kind, data)
        if sha not in self.objects:
            if not isinstance(data, LazyData):
                data = bytes(data)
            self.objects[sha] = (kind, data)
            self._sorted = None
        return sha

    def put_blob_lazy(self, data: LazyData) -> str:
        """A big generated blob: its id is its recipe's (data.oid), its bytes made only when read."""
        sha = data.oid or hash_object("blob", data)
        data.oid = sha
        if sha not in self.objects:
            self.objects[sha] = ("blob", data)
            self._sorted = None
        return sha

    def has(self, sha: str) -> bool:
        return sha in self.objects

    def kind(self, sha: str) -> str:
        try:
            return self.objects[sha][0]
        except KeyError:
            raise GitError(f"fatal: bad object {sha}") from None

    def raw(self, sha: str):
        try:
            return self.objects[sha][1]
        except KeyError:
            raise GitError(f"fatal: bad object {sha}") from None

    def size(self, sha: str) -> int:
        return len(self.raw(sha))

    def commit(self, sha: str) -> Commit:
        c = self._commits.get(sha)
        if c is None:
            kind, data = self.objects.get(sha, (None, None))
            if kind != "commit":
                raise GitError(f"fatal: {sha} is not a commit" if kind else f"fatal: bad object {sha}")
            c = decode_commit(sha, data)
            self._commits[sha] = c
        return c

    def tree(self, sha: str) -> list:
        t = self._trees.get(sha)
        if t is None:
            kind, data = self.objects.get(sha, (None, None))
            if kind != "tree":
                raise GitError(f"fatal: not a tree object: {sha}" if kind else f"fatal: bad object {sha}")
            t = decode_tree(data)
            self._trees[sha] = t
        return t

    def tag(self, sha: str) -> Tag:
        kind, data = self.objects.get(sha, (None, None))
        if kind != "tag":
            raise GitError(f"fatal: {sha} is not a tag object")
        return decode_tag(sha, data)

    def flat(self, tree_sha: str) -> dict:
        """path -> (mode, sha) for every file (blob, symlink, gitlink) under a tree."""
        if tree_sha in self._flat:
            return self._flat[tree_sha]
        out = {}

        def walk(sha, prefix):
            for mode, name, s in self.tree(sha):
                p = prefix + name
                if mode == "40000":
                    walk(s, p + "/")
                else:
                    out[p] = (mode, s)
        if tree_sha and tree_sha != EMPTY_TREE:
            walk(tree_sha, "")
        self._flat[tree_sha] = out
        return out

    def write_tree(self, flat: dict) -> str:
        """path -> (mode, sha) back into trees; returns the root tree's id."""
        dirs: dict[str, list] = {"": []}
        for path in flat:
            parts = path.split("/")
            for i in range(1, len(parts)):
                d = "/".join(parts[:i])
                dirs.setdefault(d, [])
        for path, (mode, sha) in flat.items():
            parent, _, name = path.rpartition("/")
            dirs[parent].append((mode, name, sha))
        order = sorted((d for d in dirs if d), key=lambda x: x.count("/"), reverse=True)
        ids = {}
        for d in order:
            sha = self.put("tree", encode_tree(dirs[d]))
            ids[d] = sha
            parent, _, name = d.rpartition("/")
            dirs[parent].append(("40000", name, sha))
        root = self.put("tree", encode_tree(dirs[""]))
        self._flat[root] = dict(flat)
        return root

    def put_commit(self, tree, parents, author, committer, message, extra=()) -> str:
        sha = self.put("commit", encode_commit(tree, parents, author, committer, message, extra))
        return sha

    def resolve_prefix(self, prefix: str) -> list[str]:
        if self._sorted is None:
            self._sorted = sorted(self.objects)
        import bisect
        i = bisect.bisect_left(self._sorted, prefix)
        out = []
        while i < len(self._sorted) and self._sorted[i].startswith(prefix):
            out.append(self._sorted[i])
            i += 1
        return out

    def abbrev(self, sha: str, n: int = 7) -> str:
        """The shortest unique prefix of at least n characters (git's core.abbrev auto: 7)."""
        if self._sorted is None:
            self._sorted = sorted(self.objects)
        import bisect
        i = bisect.bisect_left(self._sorted, sha)
        need = n
        for j in (i - 1, i + 1):
            if 0 <= j < len(self._sorted) and self._sorted[j] != sha:
                other = self._sorted[j]
                k = 0
                while k < len(sha) and k < len(other) and sha[k] == other[k]:
                    k += 1
                need = max(need, k + 1)
        return sha[:need]

    # -------------------------------------------------- walking history
    def parents(self, sha: str) -> tuple:
        return () if sha in self.cut else self.commit(sha).parents

    def ancestors(self, heads, stop=()):
        """Every commit reachable from heads (not from stop), newest first by committer date."""
        import heapq
        seen = set()
        excluded = self.reachable(stop) if stop else set()
        heap = []
        n = 0
        for h in heads:
            if h and h not in seen and h not in excluded:
                seen.add(h)
                c = self.commit(h)
                heapq.heappush(heap, (-c.committer.when, n, h))     # ties: first in, first out (as git's queue)
                n += 1
        out = []
        while heap:
            _, _, sha = heapq.heappop(heap)
            out.append(sha)
            for p in self.parents(sha):
                if p not in seen and p not in excluded and p in self.objects:
                    seen.add(p)
                    heapq.heappush(heap, (-self.commit(p).committer.when, n, p))
                    n += 1
        return out

    def reachable(self, heads) -> set:
        seen = set()
        stack = [h for h in heads if h]
        while stack:
            s = stack.pop()
            if s in seen or s not in self.objects:
                continue
            seen.add(s)
            stack.extend(self.parents(s))
        return seen

    def is_ancestor(self, a: str, b: str) -> bool:
        """Is commit a an ancestor of (or equal to) b?"""
        if a == b:
            return True
        return a in self.reachable([b])

    def merge_bases(self, a: str, b: str) -> list[str]:
        """Best common ancestors (git merge-base --all)."""
        common = self.reachable([a]) & self.reachable([b])
        if not common:
            return []
        # the commons that aren't ancestors of other commons
        below = self.reachable([p for c in common for p in self.parents(c)])
        return sorted(common - below, key=lambda s: -self.commit(s).committer.when)

    def objects_of(self, heads) -> set:
        """Every object reachable from these commits (commits, trees, blobs)."""
        out = set()
        for c in self.reachable(heads):
            out.add(c)
            stack = [self.commit(c).tree]
            while stack:
                t = stack.pop()
                if t in out:
                    continue
                out.add(t)
                for mode, _, s in self.tree(t):
                    if mode == "40000":
                        stack.append(s)
                    elif mode != "160000":
                        out.add(s)
        return out


def odb(world) -> ODB:
    db = world.__dict__.get("git_odb")
    if db is None:
        db = ODB()
        world.git_odb = db
    return db


def loose_bytes(kind: str, data) -> bytes:
    """A loose object file's content: zlib of 'kind size\\0data'."""
    raw = bytes(data)
    return zlib.compress(f"{kind} {len(raw)}".encode() + b"\0" + raw)
