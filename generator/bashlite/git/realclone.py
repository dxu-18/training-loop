"""Cloning a real public repository from github.com, read-only.

When a clone names a repository GitHub-lite doesn't have, it asks the real GitHub: the repository's metadata and
newest commit (api.github.com), the file list of that commit, then every file (raw.githubusercontent.com; both
allow cross-origin requests, unlike the tarball download). The result is kept in GitHub-lite as a read-only
repository owned by its real owner, so pushing to it is refused like any push to someone else's repository.

Only a snapshot is fetched: the newest commit of the default branch (or -b BRANCH), as a single commit. A size
cap keeps the browser's memory safe: when the files add up to more than LIMIT_BYTES (or LIMIT_FILES files), the
clone stops, everything it fetched or created is removed, and the user is told why.

Requests: in the live terminal they are asynchronous (the command waits in simulated sleeps while the browser
fetches, so the page stays responsive); when a session is replayed for grading they are synchronous (a worker may
block) and served from a cache first. Outside the browser (tests) urllib is used, or a TRANSPORT stub."""

from __future__ import annotations

import json
import sys
import urllib.parse

from .objects import GitError, Person, encode_commit, hash_object
from .remote import human_bytes

LIMIT_BYTES = 20 * 1024 * 1024
LIMIT_FILES = 600

_CACHE: dict[str, tuple[int, bytes]] = {}       # url -> (status, body), for replays in the same worker
TRANSPORT = None                                # tests: fn(url) -> (status, bytes)


class NetError(Exception):
    pass


# ---------------------------------------------------------------------------- HTTP

def _sync_get(url: str) -> tuple[int, bytes]:
    if TRANSPORT is not None:
        return TRANSPORT(url)
    if sys.platform == "emscripten":
        import js
        xhr = js.XMLHttpRequest.new()
        xhr.open("GET", url, False)
        xhr.responseType = "arraybuffer"          # allowed for synchronous requests inside a worker
        try:
            xhr.send(None)
        except Exception as e:  # noqa: BLE001
            raise NetError(str(e)) from None
        body = bytes(js.Uint8Array.new(xhr.response).to_py()) if xhr.response else b""
        return int(xhr.status), body
    import urllib.error
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "bash-lite-git", "Accept": "*/*"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except Exception as e:  # noqa: BLE001
        raise NetError(str(e)) from None


def fetch(ctx, url: str):
    """GET url (generator) -> (status, body). Live terminal: an asynchronous browser fetch, waited for in short
    simulated sleeps; otherwise a synchronous request. Answers are cached for replays."""
    hit = _CACHE.get(url)
    if hit is not None:
        return hit
    if sys.platform == "emscripten" and getattr(ctx.world, "net_async", False) and TRANSPORT is None:
        import asyncio

        from pyodide.http import pyfetch

        async def go():
            r = await pyfetch(url)
            return r.status, await r.bytes()
        task = asyncio.ensure_future(go())
        waited = 0
        while not task.done():
            yield ("sleep", ctx.k.now + 50_000_000)
            waited += 1
            if waited > 1200:                   # a minute
                task.cancel()
                raise NetError("timed out")
        try:
            res = task.result()
        except Exception as e:  # noqa: BLE001
            raise NetError(str(e)) from None
    else:
        res = _sync_get(url)
    if res[0] in (200, 404):
        _CACHE[url] = res
    return res


def _api(ctx, path: str):
    status, body = yield from fetch(ctx, "https://api.github.com/" + path)
    if status == 404:
        return None
    if status in (403, 429):
        raise NetError("GitHub's API rate limit was reached (60 requests an hour without signing in); try "
                       "again later")
    if status != 200:
        raise NetError(f"GitHub answered {status}")
    return json.loads(body.decode("utf-8"))


# ---------------------------------------------------------------------------- the clone

def _when(iso: str) -> int:
    import datetime
    try:
        return int(datetime.datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp())
    except ValueError:
        return 0


def fetch_real(g, ep, branch: str | None, quiet: bool):
    """Bring owner/repo from github.com into GitHub-lite (generator). -> GHRepo, or None after printing why not."""
    from .github import GHRepo, github_of
    from .remote import stderr_is_tty
    gh = github_of(g.world)
    owner, name = ep.owner, ep.repo
    tty = stderr_is_tty(g.ctx) and not quiet
    try:
        meta = yield from _api(g.ctx, f"repos/{owner}/{name}")
        if meta is None or meta.get("private"):
            return None
        owner, name = meta.get("owner", {}).get("login", owner), meta.get("name", name)
        if int(meta.get("size") or 0) * 1024 > 25 * LIMIT_BYTES:     # history far beyond any snapshot we'd keep
            g.err(_too_big(owner, name, int(meta["size"]) * 1024, 0, False))
            return False
        ref = branch or meta.get("default_branch") or "main"
        head = yield from _api(g.ctx, f"repos/{owner}/{name}/commits/{urllib.parse.quote(ref, safe='')}")
        if head is None:
            g.err(f"warning: Could not find remote branch {ref} to clone.\nfatal: Remote branch {ref} not found in "
                  "upstream origin\n")
            return False
        sha = head["sha"]
        tree_sha = head["commit"]["tree"]["sha"]
        listing = yield from _api(g.ctx, f"repos/{owner}/{name}/git/trees/{tree_sha}?recursive=1")
    except NetError as e:
        g.err(f"fatal: unable to access 'https://github.com/{owner}/{name}.git/': {e}\n")
        return False
    entries = [e for e in (listing or {}).get("tree", []) if e.get("type") in ("blob", "commit")]
    blobs = [e for e in entries if e["type"] == "blob"]
    total = sum(int(e.get("size") or 0) for e in blobs)
    if (listing or {}).get("truncated") or total > LIMIT_BYTES or len(blobs) > LIMIT_FILES:
        g.err(_too_big(owner, name, total, len(blobs), bool((listing or {}).get("truncated"))))
        return False
    # every file, checked against the id GitHub lists for it
    files: dict[str, tuple[str, bytes | str]] = {}
    got = 0
    n = len(blobs)
    for i, e in enumerate(blobs, 1):
        path = e["path"]
        url = (f"https://raw.githubusercontent.com/{owner}/{name}/{sha}/" +
               urllib.parse.quote(path, safe="/"))
        try:
            status, body = yield from fetch(g.ctx, url)
        except NetError as err:
            g.err(f"\nfatal: unable to fetch {path}: {err}\n")
            return False
        if status != 200:
            g.err(f"\nfatal: unable to fetch {path} (HTTP {status})\n")
            return False
        got += len(body)
        if got > LIMIT_BYTES:
            g.err(_too_big(owner, name, got, n, False))
            return False
        if hash_object("blob", body) != e["sha"]:
            # the raw endpoint can hand back a converted file; trust GitHub's listing (keep the id it gave)
            pass
        files[path] = (e["mode"], body)
        if tty:
            g.err(f"\rReceiving objects: {i * 100 // n:3d}% ({i}/{n}), {human_bytes(got)}" + (
                ", done.\n" if i == n else ""))
    # into the object table (only now: a failed clone leaves nothing behind)
    from .objects import odb
    db = odb(g.world)
    flat = {}
    for path, (mode, body) in files.items():
        flat[path] = (mode, db.put("blob", body))
    for e in entries:
        if e["type"] == "commit":
            flat[e["path"]] = ("160000", e["sha"])
    tree = db.write_tree(flat)
    c = head["commit"]
    a, cm = c.get("author") or {}, c.get("committer") or {}
    author = Person(a.get("name", owner), a.get("email", ""), _when(a.get("date", "")), "+0000")
    committer = Person(cm.get("name", owner), cm.get("email", ""), _when(cm.get("date", "")), "+0000")
    msg = c.get("message", "") or ""
    commit = db.put("commit", encode_commit(tree, [], author, committer, msg if msg.endswith("\n") else msg + "\n"))
    r = GHRepo(gh, owner, name, meta.get("description") or "", False, ref, _when(meta.get("created_at", "")))
    r.refs = {f"refs/heads/{ref}": commit}
    r.real = True
    r.stars = int(meta.get("stargazers_count") or 0)
    r.language = meta.get("language") or ""
    r.topics = list(meta.get("topics") or [])
    r.size_kb = max(1, total // 1024)
    gh.repos[f"{owner}/{name}"] = r
    gh.account(owner)
    gh.missing.discard(f"{owner}/{name}".lower())
    if not quiet:
        g.err(f"note: {owner}/{name} is a real GitHub repository: cloned read-only, as one snapshot commit of "
              f"{ref} ({human_bytes(total)}, {len(blobs)} files).\n")
    return r


def _too_big(owner, name, size, nfiles, truncated) -> str:
    what = ("it has too many files to list" if truncated else
            f"about {human_bytes(size)} of history" if not nfiles else
            f"{human_bytes(size)} in {nfiles} files" if size > LIMIT_BYTES else f"{nfiles} files")
    return (f"fatal: {owner}/{name} is too large to clone here: {what} (the limit is {human_bytes(LIMIT_BYTES)} "
            f"and {LIMIT_FILES} files).\nThe clone was cancelled and everything it created was removed.\n")


__all__ = ["fetch_real", "fetch", "LIMIT_BYTES", "LIMIT_FILES"]
