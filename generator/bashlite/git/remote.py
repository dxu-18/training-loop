"""Talking to other repositories: URL forms, opening the far side, and the progress lines git prints.

A remote is a repository on this machine (a path or file:// URL), on another simulated host (user@host:path over
ssh), or on GitHub-lite (git@github.com:owner/repo.git, ssh://git@github.com/..., https://github.com/owner/repo).
Objects live in the world's shared table, so "transferring" them only means counting what the other side lacks
(for the progress lines); refs are what actually moves.
"""

from __future__ import annotations

import posixpath
import re

from ..vfs import FSError
from .objects import GitError


class Endpoint:
    """Where a URL points. kind: local | github | ssh-host."""

    def __init__(self, kind, url, path=None, host=None, user=None, owner=None, repo=None, protocol="file"):
        self.kind, self.url, self.path, self.host, self.user = kind, url, path, host, user
        self.owner, self.repo, self.protocol = owner, repo, protocol
        self.shown_url = None

    @property
    def anonymized(self) -> str:
        """The URL without a user name (git's transport_anonymize_url)."""
        import re as _re
        u = self.shown_url or self.url
        u = _re.sub(r"^(https?://)[^@/]+@", r"\1", u)
        u = _re.sub(r"^(ssh://)?[\w.-]+@", r"\1", u)
        return u

    @property
    def display_to(self) -> str:
        """`To ...` in push output."""
        return self.anonymized

    @property
    def display(self) -> str:
        """`From ...` in fetch output: also without a trailing .git."""
        u = self.anonymized.rstrip("/")
        return u[:-4] if u.endswith(".git") and len(u) > 4 else u


def parse_url(ctx, url: str) -> Endpoint:
    u = url.strip()
    m = re.match(r"^(?:ssh://)?([\w.-]+)@github\.com[:/](?P<o>[\w.-]+)/(?P<r>[\w.-]+?)(?:\.git)?/?$", u)
    if m:
        return Endpoint("github", u, owner=m.group("o"), repo=m.group("r"), user=m.group(1), protocol="ssh")
    m = re.match(r"^https?://(?:[^@/]+@)?github\.com/(?P<o>[\w.-]+)/(?P<r>[\w.-]+?)(?:\.git)?/?$", u)
    if m:
        return Endpoint("github", u, owner=m.group("o"), repo=m.group("r"), protocol="https")
    m = re.match(r"^(?:ssh://)?(?:([\w.-]+)@)?([\w.-]+):(?!//)(.+)$", u)
    if m and not u.startswith("/") and "/" not in m.group(2):
        return Endpoint("ssh-host", u, path=m.group(3), host=m.group(2), user=m.group(1), protocol="ssh")
    if u.startswith("file://"):
        u = u[7:]
    path = u if u.startswith("/") else posixpath.normpath(posixpath.join(ctx.cwd, u))
    return Endpoint("local", url, path=path)


class FarCtx:
    """A stand-in process context for reading/writing a repository on another host's filesystem."""

    def __init__(self, ctx, host, uid=1000, gid=1000, home="/root"):
        self.world = ctx.world
        self.k = ctx.k
        self.host = host
        self.fs = host.fs
        self._uid, self._gid = uid, gid
        self.env = {"HOME": home, "TZ": ctx.env.get("TZ", "")}
        self.cwd = "/"
        self.git_overrides = []

        class _P:
            pass
        self.proc = _P()
        self.proc.gid = gid
        self.proc.gids = {gid}
        self.proc.cwd = "/"

    @property
    def uid(self):
        return self._uid

    @property
    def now(self):
        return self.k.now


def open_far_repo(ctx, ep: Endpoint):
    """A Repo object for a local or ssh-host remote (bare or not), or raises GitError like git does."""
    from .discover import _gitdir_of, _is_gitdir
    from .repo import Repo
    if ep.kind == "local":
        far = ctx
        path = ep.path
        fs = ctx.fs
    else:
        host = ctx.world.resolve(ep.host)
        if host is None:
            raise GitError(f"ssh: Could not resolve hostname {ep.host}: Name or service not known\n"
                           "fatal: Could not read from remote repository.\n\nPlease make sure you have the correct "
                           "access rights\nand the repository exists.")
        user = ep.user or ctx.host.user_name(ctx.uid)
        uid = host.uid_of(user)
        if uid is None:
            raise GitError(f"{user}@{ep.host}: Permission denied (publickey).\nfatal: Could not read from remote "
                           "repository.\n\nPlease make sure you have the correct access rights\nand the repository "
                           "exists.")
        u = host.users[uid]
        far = FarCtx(ctx, host, uid, u.gid, u.home)
        path = ep.path if ep.path.startswith("/") else posixpath.join(u.home, ep.path)
        fs = host.fs
    for cand in (path, path + ".git"):
        try:
            node = fs.lookup(cand)
        except FSError:
            continue
        if node.type != "d":
            continue
        if _is_gitdir(fs, cand):
            return Repo(far, cand, None)
        found = _gitdir_of(fs, cand + "/.git")
        if found:
            return Repo(far, found[0], cand, found[1])
    raise GitError(f"fatal: '{ep.url}' does not appear to be a git repository\nfatal: Could not read from remote "
                   "repository.\n\nPlease make sure you have the correct access rights\nand the repository exists.")


class Far:
    """The far side of a fetch/push, whatever it is: refs, HEAD, and ref updates."""

    def __init__(self, ep: Endpoint, repo=None, gh=None, ghrepo=None, account=None):
        self.ep, self.repo, self.gh, self.ghrepo, self.account = ep, repo, gh, ghrepo, account

    def refs(self) -> dict:
        if self.ghrepo is not None:
            return dict(self.ghrepo.all_refs())
        return dict(self.repo.ref_names("refs/"))

    def head_symref(self) -> str | None:
        if self.ghrepo is not None:
            return f"refs/heads/{self.ghrepo.default_branch}" if self.ghrepo.refs else None
        return self.repo.head_ref()

    def update(self, ref: str, new: str | None, old: str | None, force: bool, who=None):
        """Apply one ref update. -> None (ok) or a rejection reason."""
        if self.ghrepo is not None:
            return self.ghrepo.receive(ref, new, old, force, self.account)
        r = self.repo
        cur = r.resolve_ref(ref)
        if new is None:
            if cur is None:
                return "remote ref does not exist"
            if not r.bare and r.head_ref() == ref:
                return "deletion of the current branch prohibited"
            r.delete_ref(ref)
            return None
        if not r.bare and r.head_ref() == ref and r.config.get("receive.denyCurrentBranch", "refuse") == "refuse":
            return "branch is currently checked out"
        r.update_ref(ref, new, "push" if r.exists(f"logs/{ref}", True) else None)
        return None


def human_bytes(n: float) -> str:
    for unit in ("bytes", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{int(n)} bytes" if unit == "bytes" else f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} GiB"


def object_stats(db, objs) -> tuple[int, int]:
    """(count, approximate packed bytes) for a set of object ids."""
    size = 0
    for s in objs:
        try:
            n = db.size(s)
        except GitError:
            continue
        size += max(30, int(n * 0.45)) if db.kind(s) == "blob" else max(40, int(n * 0.6))
    return len(objs), size


def missing_objects(db, wanted_heads, have_heads) -> set:
    want = db.objects_of([h for h in wanted_heads if h and db.has(h) and db.kind(h) == "commit"])
    have = db.objects_of([h for h in have_heads if h and db.has(h) and db.kind(h) == "commit"])
    return want - have


def remote_progress(n: int, size: int, deltas: int = 0) -> str:
    """The `remote: ...` and `Receiving objects` lines of a fetch or clone."""
    if not n:
        return ""
    comp = max(1, int(n * 0.6))
    rate = human_bytes(max(size, 1) * 3.7) + "/s"
    lines = [f"remote: Enumerating objects: {n}, done.",
             f"remote: Counting objects: 100% ({n}/{n}), done.",
             f"remote: Compressing objects: 100% ({comp}/{comp}), done.",
             f"remote: Total {n} (delta {deltas}), reused {n - deltas} (delta {deltas}), pack-reused 0",
             f"Receiving objects: 100% ({n}/{n}), {human_bytes(size)} | {rate}, done." if size > 32 * 1024 else
             f"Unpacking objects: 100% ({n}/{n}), {human_bytes(size)} | {rate}, done."]
    if deltas and size > 32 * 1024:
        lines.append(f"Resolving deltas: 100% ({deltas}/{deltas}), done.")
    return "\n".join(lines) + "\n"


def stderr_is_tty(ctx) -> bool:
    """git shows progress only on a terminal."""
    of = ctx.proc.fds.get(2)
    return of is not None and of.kind == "tty"


def push_progress(n: int, size: int) -> str:
    if not n:
        return ""
    comp = max(1, n // 2)
    return (f"Enumerating objects: {n}, done.\nCounting objects: 100% ({n}/{n}), done.\n"
            f"Delta compression using up to 8 threads\nCompressing objects: 100% ({comp}/{comp}), done.\n"
            f"Writing objects: 100% ({n}/{n}), {human_bytes(size)} | {human_bytes(size * 4.1)}/s, done.\n"
            f"Total {n} (delta 0), reused 0 (delta 0), pack-reused 0\n")


def ref_line(flag: str, summary: str, src: str, dst: str, width: int, note: str = "") -> str:
    """` * [new branch]      feature -> origin/feature` (git pads the summary to 17; fetch also pads the source
    to the batch's widest, push doesn't: width 0)."""
    s = f"{src:<{width}}" if width else src
    return f" {flag} {summary:<17} {s} -> {dst}" + (f" ({note})" if note else "")
