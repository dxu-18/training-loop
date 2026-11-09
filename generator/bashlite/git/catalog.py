"""The practice repositories GitHub-lite serves: 30 small ones (1-5 MB) and 5 medium ones (about 100 MB), each
modelled on a well-known kind of project (an HTTP library, a dotfiles repo, a Pages site, an ML training repo, a
kernel with a long history...). Recipes live in catalog_light.py and catalog_medium.py.

Nothing is built until a problem or a command needs a repository (`gh.get` -> materialize), and a build is
deterministic: the same recipe always gives the same commit ids, whatever session builds it. Big files are
LazyData (a size and a seed; bytes are generated only when someone reads them) so a 100 MB repository costs a few
MB of memory until its large files are actually opened."""

from __future__ import annotations

import hashlib
import random

from ..vfs import LazyData
from .objects import Person, encode_commit, encode_tag, odb

# 2024-01-08 09:00 UTC: histories start after this, at a recipe-specific offset
EPOCH = 1_704_704_400

PEOPLE = {
    # login: (name, email, timezone)
    "mira-chen": ("Mira Chen", "mira.chen@example.org", "+0800"),
    "tomas-ruiz": ("Tomas Ruiz", "tomas@ruiz.dev", "+0100"),
    "priya-n": ("Priya Nair", "priya.nair@example.edu", "+0530"),
    "jberg": ("Jonas Berg", "jonas.berg@example.se", "+0100"),
    "aiko-t": ("Aiko Tanaka", "aiko@tanaka.example.jp", "+0900"),
    "sokafor": ("Sam Okafor", "sam.okafor@example.com", "+0000"),
    "lena-f": ("Lena Fischer", "lena.fischer@example.de", "+0100"),
    "dalvarez": ("Diego Alvarez", "diego@alvarez.example.mx", "-0600"),
    "noor-h": ("Noor Haddad", "noor.haddad@example.com", "+0300"),
    "kwame-a": ("Kwame Asante", "kwame@asante.example.gh", "+0000"),
    "yuki-s": ("Yuki Sato", "yuki.sato@example.jp", "+0900"),
    "elif-k": ("Elif Kaya", "elif.kaya@example.com.tr", "+0300"),
    "ravi-k": ("Ravi Kumar", "ravi.kumar@example.in", "+0530"),
    "ana-p": ("Ana Popescu", "ana.popescu@example.ro", "+0200"),
    "ben-o": ("Ben O'Neill", "ben.oneill@example.ie", "+0000"),
    "chloe-m": ("Chloe Martin", "chloe.martin@example.fr", "+0100"),
    "omar-s": ("Omar Saleh", "omar.saleh@example.eg", "+0200"),
    "grace-l": ("Grace Liu", "grace.liu@example.ca", "-0500"),
    "felix-w": ("Felix Wagner", "felix@wagner.example.at", "+0100"),
    "zara-q": ("Zara Qureshi", "zara.q@example.pk", "+0500"),
    "dependabot[bot]": ("dependabot[bot]", "49699333+dependabot[bot]@users.noreply.github.com", "+0000"),
}

ORGS = {"minilibs", "mini-ml", "toolsmiths", "open-learning", "acme-corp", "pixel-forge", "git-gym", "openmaps",
        "kernel-lab"}


class Lazy:
    """A big generated file: `size` bytes after `head` (a file signature), made from `seed` when read."""

    def __init__(self, size: int, head: bytes = b"", seed: str = ""):
        self.size, self.head, self.seed = size, head, seed


class LFS(Lazy):
    """Like Lazy, but committed as a Git LFS pointer; the content goes to the LFS store."""


class Recipe:
    def __init__(self, owner, name, description, build, language="", topics=(), kind="light", homepage="",
                 stars=0, about="", default_branch="main", license="MIT"):
        self.owner, self.name, self.description, self.build = owner, name, description, build
        self.language, self.topics, self.kind, self.homepage = language, list(topics), kind, homepage
        self.stars, self.about, self.default_branch, self.license = stars, about, default_branch, license

    @property
    def full(self):
        return f"{self.owner}/{self.name}"


RECIPES: dict[str, Recipe] = {}


def recipe(owner, name, description, **kw):
    def reg(fn):
        r = Recipe(owner, name, description, fn, **kw)
        RECIPES[r.full.lower()] = r
        return fn
    return reg


def _load():
    from . import catalog_light, catalog_light2, catalog_medium  # noqa: F401  (registers the recipes)


# ---------------------------------------------------------------------------- building a history

class Builder:
    """Writes a repository's history straight into the world's object table: commits on branches, merges, tags,
    and the GitHub side (pull requests, issues, releases)."""

    def __init__(self, gh, repo, recipe: Recipe):
        self.gh, self.repo, self.recipe = gh, repo, recipe
        self.db = odb(gh.world)
        seed = int(hashlib.sha256(recipe.full.encode()).hexdigest()[:8], 16)
        self.rng = random.Random(seed)
        self.t = EPOCH + seed % (86400 * 120)
        self.heads: dict[str, str | None] = {recipe.default_branch: None}
        self.trees: dict[str, dict] = {recipe.default_branch: {}}
        self.cur = recipe.default_branch
        self.tags: dict[str, str] = {}
        self.blob_cache: dict = {}
        self.lfs_bytes = 0
        self.lazy_bytes = 0
        self.commits = 0

    # ----- people and time
    def person(self, login, when=None) -> Person:
        name, email, tz = PEOPLE.get(login, (login, f"{login}@users.noreply.github.com", "+0000"))
        when = self.t if when is None else when
        return Person(name, email, when, tz)

    def tick(self, lo=1, hi=48):
        """Advance the clock by lo..hi hours (and a few minutes)."""
        self.t += self.rng.randint(lo * 3600, hi * 3600) + self.rng.randint(0, 3599)
        return self.t

    # ----- content
    def blob(self, content, path="") -> tuple[str, str]:
        if isinstance(content, tuple):
            return content
        mode = "100755" if path.endswith((".sh",)) or path.startswith(("bin/", "scripts/")) and not \
            path.endswith((".py", ".md", ".txt")) else "100644"
        if isinstance(content, LFS):
            from .lfs import pointer_text, store
            seed = content.seed or f"{self.recipe.full}:{path}:{content.size}"
            data = LazyData(size=content.size, seed=seed, head=content.head)
            oid = hashlib.sha256(f"lfs:{seed}:{content.size}".encode()).hexdigest()
            store(self.gh.world)[oid] = data
            self.repo.lfs[oid] = data
            self.lfs_bytes += content.size
            return "100644", self.db.put("blob", pointer_text(oid, content.size))
        if isinstance(content, Lazy):
            seed = content.seed or f"{self.recipe.full}:{path}:{content.size}"
            oid = hashlib.sha1(f"blob-recipe:{seed}:{content.size}".encode()).hexdigest()
            if not self.db.has(oid):
                self.lazy_bytes += content.size
            return "100644", self.db.put_blob_lazy(LazyData(size=content.size, seed=seed, head=content.head, oid=oid))
        if isinstance(content, str):
            content = content.encode()
        return mode, self.db.put("blob", content)

    def tree(self, branch=None) -> dict:
        return self.trees[branch or self.cur]

    def read(self, path, branch=None) -> str:
        hit = self.trees[branch or self.cur].get(path)
        return bytes(self.db.raw(hit[1])).decode("utf-8", "replace") if hit else ""

    # ----- commits
    def commit(self, msg, files=None, delete=(), by="mira-chen", committer=None, hours=(1, 48), at=None,
               rename=None) -> str:
        """Commit on the current branch: files {path: content} added or changed, paths deleted, renames {old: new}."""
        if at is not None:
            self.t = at
        elif hours:
            self.tick(*hours)
        flat = dict(self.trees[self.cur])
        for old, new in (rename or {}).items():
            if old in flat:
                flat[new] = flat.pop(old)
        for p, c in (files or {}).items():
            flat[p] = self.blob(c, p)
        for p in delete:
            if p.endswith("/"):
                for k in [k for k in flat if k.startswith(p)]:
                    del flat[k]
            else:
                flat.pop(p, None)
        tree = self.db.write_tree(flat)
        parent = self.heads[self.cur]
        author = self.person(by)
        comm = self.person(committer or by)
        sha = self.db.put("commit", encode_commit(tree, [parent] if parent else [], author, comm,
                                                   msg if msg.endswith("\n") else msg + "\n"))
        self.heads[self.cur] = sha
        self.trees[self.cur] = flat
        self.commits += 1
        return sha

    def edit(self, path, fn, msg, by="mira-chen", **kw) -> str:
        """Commit a change to one text file: fn(old text) -> new text."""
        return self.commit(msg, {path: fn(self.read(path))}, by=by, **kw)

    def branch(self, name, start=None):
        """Create a branch at the current (or another branch's) tip and switch to it."""
        src = start or self.cur
        self.heads[name] = self.heads[src]
        self.trees[name] = dict(self.trees[src])
        self.cur = name
        return self

    def switch(self, name):
        self.cur = name
        return self

    def merge(self, name, msg=None, by="mira-chen", committer=None, ff=False, delete=False) -> str:
        """Merge branch `name` into the current one (a merge commit unless ff and possible). Path-level three-way:
        what the other branch changed since the merge base wins."""
        ours, theirs = self.heads[self.cur], self.heads[name]
        if ff and (ours is None or self.db.is_ancestor(ours, theirs)):
            self.heads[self.cur] = theirs
            self.trees[self.cur] = dict(self.trees[name])
            if delete:
                self.drop(name)
            return theirs
        bases = self.db.merge_bases(ours, theirs)
        base = dict(self.db.flat(self.db.commit(bases[0]).tree)) if bases else {}
        flat = dict(self.trees[self.cur])
        other = self.trees[name]
        for p in set(base) | set(other):
            if base.get(p) != other.get(p):
                if p in other:
                    flat[p] = other[p]
                else:
                    flat.pop(p, None)
        self.tick(0, 6)
        tree = self.db.write_tree(flat)
        msg = msg or f"Merge branch '{name}'" + ("" if self.cur in ("main", "master") else f" into {self.cur}")
        sha = self.db.put("commit", encode_commit(tree, [ours, theirs], self.person(by),
                                                   self.person(committer or by), msg + "\n"))
        self.heads[self.cur] = sha
        self.trees[self.cur] = flat
        self.commits += 1
        if delete:
            self.drop(name)
        return sha

    def merge_pr(self, number, name, title, by="mira-chen", author=None, method="merge", delete=True, body="",
                 reviews=(), labels=()):
        """A pull request that was merged on GitHub (merge commit by the GitHub bot, or a squash)."""
        login = author or by
        self.tick(1, 30)
        head_sha = self.heads[name]
        base_before = self.heads[self.cur]
        if method == "squash":
            bases = self.db.merge_bases(base_before, head_sha)
            n = len([c for c in self.db.ancestors([head_sha], [bases[0]] if bases else [])])
            flat = dict(self.trees[self.cur])
            base = dict(self.db.flat(self.db.commit(bases[0]).tree)) if bases else {}
            for p in set(base) | set(self.trees[name]):
                if base.get(p) != self.trees[name].get(p):
                    if p in self.trees[name]:
                        flat[p] = self.trees[name][p]
                    else:
                        flat.pop(p, None)
            msgs = [self.db.commit(c).subject for c in reversed(self.db.ancestors([head_sha],
                                                                                    [bases[0]] if bases else []))]
            text = f"{title} (#{number})\n" + ("\n" + "".join(f"* {m}\n" for m in msgs) if n > 1 else "")
            tree = self.db.write_tree(flat)
            sha = self.db.put("commit", encode_commit(tree, [base_before], self.person(login),
                                                       Person("GitHub", "noreply@github.com", self.t, "+0000"), text))
            self.heads[self.cur] = sha
            self.trees[self.cur] = flat
            self.commits += 1
        else:
            sha = self.merge(name, f"Merge pull request #{number} from {login}/{name}\n\n{title}", by=by)
            # GitHub's merge button commits as the GitHub bot
            c = self.db.commit(sha)
            data = encode_commit(c.tree, list(c.parents), self.person(by),
                                 Person("GitHub", "noreply@github.com", self.t, "+0000"), c.message)
            sha = self.db.put("commit", data)
            self.heads[self.cur] = sha
        pr = self._pr(number, title, body, name, self.cur, login, head_sha, labels)
        pr.state = "MERGED"
        pr.merged_by = by
        pr.merge_commit = sha
        pr.merge_method = method
        pr.reviews = list(reviews)
        if delete:
            self.drop(name)
        return sha

    def open_pr(self, number, name, title, author, base=None, body="", draft=False, reviews=(), comments=(),
                labels=(), closed=False, head_owner=None):
        pr = self._pr(number, title, body, name, base or self.recipe.default_branch, author, self.heads.get(name),
                      labels, draft, head_owner)
        pr.reviews = list(reviews)
        pr.comments = list(comments)
        if closed:
            pr.state = "CLOSED"
        return pr

    def _pr(self, number, title, body, head, base, author, head_sha, labels=(), draft=False, head_owner=None):
        from .github import PR
        repo_full = f"{head_owner}/{self.repo.name}" if head_owner else self.repo.full
        pr = PR(number, title, body, repo_full, head, base, author, draft, self.t)
        pr.head_sha = head_sha
        pr.labels = list(labels)
        self.gh.pulls_of(self.repo).append(pr)
        self.gh.counter[self.repo.full] = max(self.gh.counter.get(self.repo.full, 0), number)
        return pr

    def issue(self, number, title, author, body="", labels=(), closed=False, comments=(), assignees=()):
        from .github import Issue
        self.tick(0, 12)
        it = Issue(number, title, body, author, self.t, labels)
        it.state = "CLOSED" if closed else "OPEN"
        it.comments = list(comments)
        it.assignees = list(assignees)
        self.gh.issues_of(self.repo).append(it)
        self.gh.counter[self.repo.full] = max(self.gh.counter.get(self.repo.full, 0), number)
        return it

    def release(self, tag, title=None, notes="", by="mira-chen", prerelease=False):
        from .github import Release
        self.repo.releases.append(Release(tag, title or tag, notes, by, self.t, prerelease=prerelease))

    def tag(self, name, msg=None, by="mira-chen", at=None):
        """A tag on the current tip: annotated when msg is given."""
        target = at or self.heads[self.cur]
        if msg is None:
            self.tags[name] = target
        else:
            self.tags[name] = self.db.put("tag", encode_tag(target, "commit", name, self.person(by), msg + "\n"))
        return self.tags[name]

    def drop(self, name):
        self.heads.pop(name, None)
        self.trees.pop(name, None)

    def finish(self):
        refs = {}
        for b, s in self.heads.items():
            if s:
                refs[f"refs/heads/{b}"] = s
        for t, s in self.tags.items():
            refs[f"refs/tags/{t}"] = s
        self.repo.refs = refs
        text = sum(self.db.size(s) for p, (m, s) in self.trees[self.recipe.default_branch].items()
                   if m != "160000" and not isinstance(self.db.raw(s), LazyData))
        hist = self.commits * 900
        self.repo.size_kb = max(1, (int(text * 0.45) + hist + self.lazy_bytes + self.lfs_bytes) // 1024)


def materialize(gh, owner: str, name: str, as_owner: str | None = None, as_name: str | None = None):
    """Build owner/name from the catalog on first use (optionally as another owner/name, e.g. the user's own
    copy). -> GHRepo or None when it isn't one of ours."""
    _load()
    rec = RECIPES.get(f"{owner}/{name}".lower())
    if rec is None:
        return None
    from .github import GHRepo
    o, n = as_owner or rec.owner, as_name or rec.name
    repo = GHRepo(gh, o, n, rec.description, False, rec.default_branch, EPOCH)
    repo.topics = list(rec.topics)
    repo.homepage = rec.homepage
    repo.stars = rec.stars
    repo.language = rec.language
    repo.license = rec.license
    gh.repos[f"{o}/{n}"] = repo
    acc = gh.account(o, org=o in ORGS)
    if o in PEOPLE:
        acc.name, acc.email = PEOPLE[o][0], PEOPLE[o][1]
    for login, (nm, em, _) in PEOPLE.items():
        a = gh.account(login)
        a.name, a.email = nm, em
    b = Builder(gh, repo, rec)
    rec.build(b)
    b.finish()
    repo.created = EPOCH
    repo.pushed = b.t
    return repo


def catalog_list(kind=None):
    """[(full name, description, kind, language)] for the docs and the problem generator."""
    _load()
    return [(r.full, r.description, r.kind, r.language) for r in RECIPES.values() if kind in (None, r.kind)]
