"""GitHub-lite: the GitHub a simulated machine talks to. Accounts (SSH keys, tokens), repositories (refs in the
world's object table, generated on first use from the practice catalog), pull requests and issues (one numbering),
releases, Actions runs, Pages, protected branches; plus authentication for git over SSH and HTTPS and `ssh -T`.

Nothing here touches the real github.com: pushes land in this model. (Cloning a real public repository, read-only
and size-capped, is handled by realclone.py.)
"""

from __future__ import annotations

import base64
import hashlib
import re

from .objects import GitError, Person, encode_commit

HOST_IP = "140.82.121.3"
# GitHub's published ED25519 host key (docs: "GitHub's SSH key fingerprints")
HOST_KEY = "AAAAC3NzaC1lZDI1NTE5AAAAIOMqqnkVzrm0SdG6UOoqKLsabgH5C9okWi0dh2l9GKJl"
HOST_FP = "SHA256:+DiY3wvvV6TuJJhbpZisF/zLDA0zPMSvHdkr4UvCOqU"
BOT = Person("GitHub", "noreply@github.com", 0, "+0000")


class Account:
    def __init__(self, login, name="", email="", org=False):
        self.login, self.name, self.email, self.org = login, name or login, email or f"{login}@users.noreply.github.com", org
        self.keys: list[tuple[str, bytes]] = []          # (title, public key blob)
        self.tokens: set[str] = set()
        self.followers = 0


class PR:
    def __init__(self, number, title, body, head_repo, head_branch, base, author, draft=False, created=0):
        self.number, self.title, self.body = number, title, body
        self.head_repo, self.head_branch, self.base = head_repo, head_branch, base
        self.author, self.draft, self.created = author, draft, created
        self.state = "OPEN"               # OPEN | CLOSED | MERGED
        self.merged_by = None
        self.merge_commit = None
        self.merge_method = None
        self.reviews: list[tuple[str, str, str]] = []   # (login, APPROVED | CHANGES_REQUESTED | COMMENTED, body)
        self.comments: list[tuple[str, str]] = []
        self.head_sha = None
        self.labels: list[str] = []


class Issue:
    def __init__(self, number, title, body, author, created=0, labels=()):
        self.number, self.title, self.body, self.author, self.created = number, title, body, author, created
        self.state = "OPEN"
        self.labels = list(labels)
        self.comments: list[tuple[str, str]] = []
        self.assignees: list[str] = []


class Release:
    def __init__(self, tag, name, notes, author, created, draft=False, prerelease=False):
        self.tag, self.name, self.notes, self.author, self.created = tag, name, notes, author, created
        self.draft, self.prerelease = draft, prerelease


class Run:
    def __init__(self, rid, workflow, event, branch, sha, created, fails, title):
        self.id, self.workflow, self.event, self.branch, self.sha = rid, workflow, event, branch, sha
        self.created, self.fails, self.title = created, fails, title
        self.duration = 25 + (int(sha[:2], 16) % 40)

    def status(self, now):
        el = now - self.created
        if el < 3:
            return "queued", ""
        if el < self.duration:
            return "in_progress", ""
        return "completed", "failure" if self.fails else "success"


class GHRepo:
    def __init__(self, gh, owner, name, description="", private=False, default_branch="main", created=0):
        self.gh, self.owner, self.name = gh, owner, name
        self.description, self.private, self.default_branch = description, private, default_branch
        self.refs: dict[str, str] = {}
        self.created = self.pushed = created
        self.fork_of: str | None = None
        self.stars = 0
        self.topics: list[str] = []
        self.homepage = ""
        self.protected: dict[str, dict] = {}            # branch -> {"pr": True, "reviews": n}
        self.collaborators: set[str] = set()
        self.pages: dict | None = None                  # {"branch": "main", "path": "/docs"}
        self.runs: list[Run] = []
        self.releases: list[Release] = []
        self.lfs: dict[str, object] = {}
        self.size_kb = 0
        self.real = False                               # cloned from the real github.com (read-only)
        self.language = ""
        self.license = ""

    @property
    def full(self) -> str:
        return f"{self.owner}/{self.name}"

    @property
    def db(self):
        from .objects import odb
        return odb(self.gh.world)

    def all_refs(self):
        out = dict(self.refs)
        for pr in self.gh.pulls_of(self):
            if pr.head_sha:
                out[f"refs/pull/{pr.number}/head"] = pr.head_sha
        return dict(sorted(out.items()))

    def can_push(self, login) -> bool:
        if self.real:
            return False
        acc = self.gh.accounts.get(self.owner)
        if login == self.owner or login in self.collaborators:
            return True
        return bool(acc and acc.org and login in self.gh.members.get(self.owner, set()))

    def receive(self, ref, new, old, force, account):
        """A push's ref update. -> None, or the reason git prints for a rejection."""
        login = account.login if account else None
        if not self.can_push(login):
            return "permission"
        cur = self.refs.get(ref)
        branch = ref[len("refs/heads/"):] if ref.startswith("refs/heads/") else None
        if branch and branch in self.protected:
            rule = self.protected[branch]
            if new is None:
                return "protected:delete"
            if rule.get("pr"):
                return "protected:pr"
            if cur and not force and not self.db.is_ancestor(cur, new):
                return "protected:force"
        if new is None:
            if cur is None:
                return "remote ref does not exist"
            if branch == self.default_branch:
                return "default:delete"
            del self.refs[ref]
            return None
        if cur and not force and self.db.has(new) and self.db.kind(new) == "commit" and self.db.kind(cur) == "commit" \
                and not self.db.is_ancestor(cur, new):
            return "non-fast-forward"
        if ref.startswith("refs/tags/") and cur and not force and cur != new:
            return "tag:exists"
        self.refs[ref] = new
        self.pushed = self.gh.now()
        if not self.refs.get(f"refs/heads/{self.default_branch}") and branch:
            self.default_branch = branch
        if branch:
            for pr in self.gh.pulls_of_head(self, branch):
                pr.head_sha = new
            self.gh.trigger_runs(self, "push", branch, new)
        return None


class GitHub:
    def __init__(self, world):
        self.world = world
        self.accounts: dict[str, Account] = {}
        self.members: dict[str, set] = {}
        self.repos: dict[str, GHRepo] = {}
        self.pulls: dict[str, list[PR]] = {}
        self.issues: dict[str, list[Issue]] = {}
        self.counter: dict[str, int] = {}
        self.run_ids = 6_000_000_000
        self.me: str | None = None                    # the login the simulated user signs in as
        self.missing: set[str] = set()                # names that don't exist (not in the catalog, offline)

    def now(self) -> int:
        return self.world.kernel.now // 1_000_000_000

    # -------------------------------------------------- accounts
    def account(self, login, **kw) -> Account:
        a = self.accounts.get(login)
        if a is None:
            a = Account(login, **kw)
            self.accounts[login] = a
        return a

    def account_for_key(self, blob: bytes):
        for a in self.accounts.values():
            if any(k == blob for _, k in a.keys):
                return a
        return None

    def account_for_token(self, token: str):
        for a in self.accounts.values():
            if token in a.tokens:
                return a
        return None

    def new_token(self, login) -> str:
        acc = self.account(login)
        tok = "gho_" + base64.b32encode(hashlib.sha256(f"{login}:{len(acc.tokens)}".encode()).digest()).decode()[:36]
        acc.tokens.add(tok)
        return tok

    # -------------------------------------------------- repositories
    def get(self, owner: str, name: str, materialize: bool = True):
        key = f"{owner}/{name}".lower()
        for k, r in self.repos.items():
            if k.lower() == key:
                return r
        if materialize and key not in self.missing:
            from .catalog import materialize as from_catalog
            r = from_catalog(self, owner, name)
            if r is not None:
                return r
            self.missing.add(key)
        return None

    def create(self, owner, name, description="", private=False, default_branch="main") -> GHRepo:
        if self.get(owner, name):
            raise GitError(f"GraphQL: Name already exists on this account (createRepository)")
        r = GHRepo(self, owner, name, description, private, default_branch, self.now())
        self.repos[f"{owner}/{name}"] = r
        self.account(owner)
        return r

    def fork(self, src: GHRepo, owner: str) -> GHRepo:
        existing = self.get(owner, src.name, materialize=False)
        if existing:
            return existing
        r = GHRepo(self, owner, src.name, src.description, False, src.default_branch, self.now())
        r.refs = {k: v for k, v in src.refs.items() if k.startswith(("refs/heads/", "refs/tags/"))}
        r.fork_of = src.full
        r.lfs = dict(src.lfs)
        self.repos[f"{owner}/{src.name}"] = r
        return r

    def delete(self, r: GHRepo):
        self.repos.pop(r.full, None)
        self.missing.add(r.full.lower())

    # -------------------------------------------------- pull requests and issues
    def next_number(self, r: GHRepo) -> int:
        self.counter[r.full] = self.counter.get(r.full, 0) + 1
        return self.counter[r.full]

    def pulls_of(self, r: GHRepo) -> list:
        return self.pulls.setdefault(r.full, [])

    def issues_of(self, r: GHRepo) -> list:
        return self.issues.setdefault(r.full, [])

    def pulls_of_head(self, head_repo: GHRepo, branch: str):
        for prs in self.pulls.values():
            for pr in prs:
                if pr.head_repo == head_repo.full and pr.head_branch == branch and pr.state == "OPEN":
                    yield pr

    def open_pr(self, base_repo, head_repo, head_branch, base, title, body, author, draft=False) -> PR:
        sha = head_repo.refs.get(f"refs/heads/{head_branch}")
        if sha is None:
            raise GitError(f"pull request create failed: GraphQL: Head sha can't be blank, Base sha can't be blank, "
                           f"No commits between {base} and {head_branch}, Head ref must be a branch (createPullRequest)")
        base_sha = base_repo.refs.get(f"refs/heads/{base}")
        if base_sha and self_db(self).is_ancestor(sha, base_sha):
            raise GitError(f"pull request create failed: GraphQL: No commits between {base} and {head_branch} "
                           "(createPullRequest)")
        pr = PR(self.next_number(base_repo), title, body, head_repo.full, head_branch, base, author, draft, self.now())
        pr.head_sha = sha
        self.pulls_of(base_repo).append(pr)
        self.trigger_runs(base_repo, "pull_request", head_branch, sha, pr=pr)
        return pr

    def find_pr(self, r: GHRepo, number: int):
        for pr in self.pulls_of(r):
            if pr.number == number:
                return pr
        return None

    def merge_pr(self, r: GHRepo, pr: PR, method: str, merger: str, subject=None, body=None):
        """Merge a pull request the way github.com does. -> the new tip of the base branch."""
        db = self_db(self)
        base_ref = f"refs/heads/{pr.base}"
        base = r.refs.get(base_ref)
        head = pr.head_sha
        if pr.state != "OPEN":
            raise GitError(f"Pull request #{pr.number} ({pr.title}) can't be merged: it is {pr.state.lower()}")
        rule = r.protected.get(pr.base, {})
        if rule.get("reviews") and sum(1 for _, st, _ in pr.reviews if st == "APPROVED") < rule["reviews"]:
            raise GitError(f"X Pull request #{pr.number} is not mergeable: the base branch policy prohibits the merge.\n"
                           "To have the pull request merged after all the requirements have been met, add the `--auto` "
                           "flag.\nTo use administrator privileges to immediately merge the pull request, add the "
                           "`--admin` flag.")
        when = self.now()
        acc = self.account(merger)
        who = Person(acc.name, acc.email, when, "+0000")
        bot = Person(BOT.name, BOT.email, when, "+0000")
        from .merge import merge_trees
        bases = db.merge_bases(base, head) if base else []
        if base and db.is_ancestor(head, base):
            raise GitError(f"Pull request #{pr.number} has nothing to merge")
        if method == "merge":
            res = merge_trees(db, db.commit(bases[0]).tree if bases else None, db.commit(base).tree,
                              db.commit(head).tree)
            if res.conflicts:
                raise GitError(f"X Pull request #{pr.number} is not mergeable: the merge commit cannot be cleanly "
                               "created.")
            tree = db.write_tree(res.flat)
            msg = subject or f"Merge pull request #{pr.number} from {pr.head_repo.split('/')[0]}/{pr.head_branch}"
            msg += "\n\n" + (body if body is not None else pr.title) + "\n"
            new = db.put("commit", encode_commit(tree, [base, head], who, bot, msg))
        elif method == "squash":
            res = merge_trees(db, db.commit(bases[0]).tree if bases else None, db.commit(base).tree,
                              db.commit(head).tree)
            if res.conflicts:
                raise GitError(f"X Pull request #{pr.number} is not mergeable: the merge commit cannot be cleanly "
                               "created.")
            tree = db.write_tree(res.flat)
            commits = list(reversed(db.ancestors([head], [base])))
            msg = subject or f"{pr.title} (#{pr.number})"
            details = body if body is not None else "\n\n".join(f"* {db.commit(c).subject}" for c in commits)
            msg += ("\n\n" + details if details else "") + "\n"
            author = db.commit(commits[-1]).author if commits else who
            new = db.put("commit", encode_commit(tree, [base], Person(author.name, author.email, when, "+0000"),
                                                 bot, msg))
        else:   # rebase
            commits = list(reversed(db.ancestors([head], [base])))
            tip = base
            for c in commits:
                cc = db.commit(c)
                pt = db.commit(cc.parents[0]).tree if cc.parents else None
                res = merge_trees(db, pt, db.commit(tip).tree, cc.tree)
                if res.conflicts:
                    raise GitError(f"X Pull request #{pr.number} is not mergeable: this branch can't be rebased.")
                tree = db.write_tree(res.flat)
                tip = db.put("commit", encode_commit(tree, [tip], cc.author, bot, cc.message))
            new = tip
        r.refs[base_ref] = new
        r.pushed = when
        pr.state = "MERGED"
        pr.merged_by = merger
        pr.merge_commit = new
        pr.merge_method = method
        self.trigger_runs(r, "push", pr.base, new)
        return new

    # -------------------------------------------------- Actions
    def workflows(self, r: GHRepo, sha: str):
        """Workflow files in .github/workflows of a commit: [(path, name, events)]."""
        db = self_db(self)
        out = []
        if not sha or not db.has(sha):
            return out
        flat = db.flat(db.commit(sha).tree)
        for p, (mode, s) in sorted(flat.items()):
            if p.startswith(".github/workflows/") and p.endswith((".yml", ".yaml")):
                text = bytes(db.raw(s)).decode("utf-8", "replace")
                name = p.rsplit("/", 1)[-1].rsplit(".", 1)[0]
                events = set()
                for line in text.split("\n"):
                    ln = line.strip()
                    if ln.startswith("name:") and not line.startswith(" "):
                        name = ln.split(":", 1)[1].strip().strip("'\"") or name
                    if ln.startswith("on:"):
                        rest = ln[3:].strip()
                        for ev in ("push", "pull_request", "workflow_dispatch", "release", "schedule"):
                            if ev in rest:
                                events.add(ev)
                    for ev in ("push:", "pull_request:", "workflow_dispatch:", "release:"):
                        if ln.startswith(ev) and line.startswith(" "):
                            events.add(ev[:-1])
                out.append((p, name, events or {"push"}))
        return out

    def trigger_runs(self, r: GHRepo, event, branch, sha, pr=None):
        db = self_db(self)
        for path, name, events in self.workflows(r, sha):
            if event not in events:
                continue
            fails = _tests_fail(db, sha)
            self.run_ids += 1
            title = db.commit(sha).subject if db.has(sha) else branch
            if pr is not None:
                title = pr.title
            r.runs.append(Run(self.run_ids, name, event, branch, sha, self.now(), fails, title))
        if r.pages and r.pages.get("branch") == branch:
            r.pages["built"] = sha

    # -------------------------------------------------- ssh
    def ssh_known(self, ctx) -> bool:
        home = ctx.env.get("HOME", "")
        try:
            text = bytes(ctx.fs.lookup(f"{home}/.ssh/known_hosts").data).decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            return False
        return any(line.split()[:1] and "github.com" in line.split()[0] for line in text.split("\n") if line.strip())


def self_db(gh):
    from .objects import odb
    return odb(gh.world)


def _tests_fail(db, sha) -> bool:
    """A run fails when the commit's tests can't pass: a test with `assert False`, or a FAILING marker file."""
    try:
        flat = db.flat(db.commit(sha).tree)
    except GitError:
        return False
    if ".ci-failing" in flat:
        return True
    for p, (mode, s) in flat.items():
        if p.startswith("tests/") and p.endswith(".py"):
            data = bytes(db.raw(s)[:200_000])
            if b"assert False" in data or b"raise NotImplementedError  # TODO" in data:
                return True
        if p.startswith(("test/", "tests/", "__tests__/")) and p.endswith((".js", ".ts")):
            if b"assert.fail(" in bytes(db.raw(s)[:200_000]):
                return True
    return False


def github_of(world) -> GitHub:
    gh = world.__dict__.get("github")
    if gh is None:
        gh = GitHub(world)
        world.github = gh
        # github.com and its ssh endpoint
        if "github.com" not in world.hosts:
            from ..world import Host
            h = Host(world, "github.com", HOST_IP)
            h.extra["github"] = gh
            world.hosts["github.com"] = h
            world.dns["github.com"] = "github.com"
            world.dns[HOST_IP] = "github.com"
    return gh


def known_hosts_line() -> str:
    return f"github.com ssh-ed25519 {HOST_KEY}\n"


# ---------------------------------------------------------------------------- authentication

def ssh_account(ctx, gh: GitHub):
    """The account the client's SSH keys identify (or None)."""
    from ..utils.net import identity_pubs
    for kind, blob, _ in identity_pubs(ctx, None):
        acc = gh.account_for_key(blob)
        if acc is not None:
            return acc
    return None


def confirm_host(ctx, gh: GitHub):
    """First contact: ssh asks to trust github.com's host key (generator). -> True when trusted."""
    if gh.ssh_known(ctx):
        return True
    tty = ctx.world.tty_of(ctx.proc)
    ctx.e(f"The authenticity of host 'github.com ({HOST_IP})' can't be established.\n"
          f"ED25519 key fingerprint is {HOST_FP}.\nThis key is not known by any other names.\n"
          "Are you sure you want to continue connecting (yes/no/[fingerprint])? ")
    if tty is None:
        return False
    rd = ctx.reader("-")
    line = yield from rd.readline()
    ans = (line or b"").decode("utf-8", "replace").strip()
    if ans != "yes" and ans != HOST_FP:
        ctx.e("Host key verification failed.\n")
        return False
    home = ctx.env.get("HOME", "")
    path = f"{home}/.ssh/known_hosts"
    try:
        node = ctx.fs.lookup(path)
        node.data += known_hosts_line().encode()
    except Exception:  # noqa: BLE001
        ctx.fs.mkdir(f"{home}/.ssh", "/", 0o700, ctx.uid, ctx.proc.gid, parents=True)
        ctx.fs.write_file(path, known_hosts_line(), "/", 0o644, ctx.uid, ctx.proc.gid)
    ctx.e("Warning: Permanently added 'github.com' (ED25519) to the list of known hosts.\n")
    return True


def https_account(ctx, gh: GitHub, url: str, need: bool):
    """The account an HTTPS request authenticates as (credential helper, or prompts). Generator.
    -> (account or None, error message or None)."""
    from .repo import Config
    cfg = Config(ctx.fs, ctx.env.get("HOME", "/"), None, getattr(ctx, "git_overrides", ()), ctx.env)
    helpers = [h for h in cfg.get_all("credential.helper") if h]
    if helpers and helpers[-1]:
        h = helpers[-1]
        if "gh auth git-credential" in h and gh.me and gh.accounts.get(gh.me) and gh.accounts[gh.me].tokens:
            return gh.accounts[gh.me], None
        if h in ("store",) or h.startswith("store"):
            home = ctx.env.get("HOME", "")
            try:
                text = bytes(ctx.fs.lookup(f"{home}/.git-credentials").data).decode()
            except Exception:  # noqa: BLE001
                text = ""
            import re
            for line in text.split("\n"):
                m = re.match(r"https://([^:]+):([^@]+)@github\.com", line.strip())
                if m:
                    acc = gh.account_for_token(m.group(2))
                    if acc:
                        return acc, None
    if not need:
        return None, None
    tty = ctx.world.tty_of(ctx.proc)
    if tty is None:
        return None, "fatal: could not read Username for 'https://github.com': terminal prompts disabled"
    ctx.e("Username for 'https://github.com': ")
    rd = ctx.reader("-")
    user = (yield from rd.readline()).decode("utf-8", "replace").strip()
    from ..utils.net import read_password
    pw = yield from read_password(ctx, f"Password for 'https://{user}@github.com': ")
    acc = gh.account_for_token(pw or "")
    if acc is not None:
        return acc, None
    return None, ("remote: Support for password authentication was removed on August 13, 2021.\n"
                  "remote: Use a personal access token (or `gh auth login`) instead of your password.\n"
                  f"fatal: Authentication failed for '{url.rstrip('/')}/'")


def ssh_command(ctx, gh: GitHub, command):
    """`ssh git@github.com` (and `ssh -T`): a greeting when a key matches, else permission denied. Generator."""
    ok = yield from confirm_host(ctx, gh)
    if not ok:
        return 255
    acc = ssh_account(ctx, gh)
    if acc is None:
        ctx.e("git@github.com: Permission denied (publickey).\n")
        return 255
    if command:
        ctx.e("Invalid command: '" + " ".join(command) + "'\n  You appear to be using ssh to clone a git:// URL.\n"
              "  Make sure your core.gitProxy config option and the\n  GIT_PROXY_COMMAND environment variable are NOT "
              "set.\n")
        return 128
    ctx.e(f"Hi {acc.login}! You've successfully authenticated, but GitHub does not provide shell access.\n")
    return 1


# ---------------------------------------------------------------------------- the web side (curl / wget)

PAGES_404 = (b"<!DOCTYPE html>\n<html>\n<head><title>Site not found &middot; GitHub Pages</title></head>\n<body>\n"
             b"<h1>404</h1>\n<p><strong>There isn't a GitHub Pages site here.</strong></p>\n</body>\n</html>\n")


def _md_page(title: str, text: str) -> bytes:
    """What Pages' Jekyll build makes of a Markdown index (headings and paragraphs only)."""
    import html
    out = []
    para = []

    def flush():
        if para:
            out.append("<p>" + html.escape(" ".join(para)) + "</p>")
            para.clear()
    for ln in text.split("\n"):
        m = re.match(r"^(#{1,6})\s+(.*)$", ln)
        if m:
            flush()
            n = len(m.group(1))
            out.append(f"<h{n}>{html.escape(m.group(2))}</h{n}>")
        elif not ln.strip():
            flush()
        else:
            para.append(ln.strip())
    flush()
    return (f"<!DOCTYPE html>\n<html lang=\"en-US\">\n<head>\n<meta charset=\"UTF-8\">\n<title>{html.escape(title)}"
            f"</title>\n</head>\n<body>\n" + "\n".join(out) + "\n</body>\n</html>\n").encode()


def _pages_file(r: GHRepo, rest: str):
    """(bytes, file name) of a Pages site file, or None."""
    db = self_db(r.gh)
    sha = r.pages.get("built")
    if not sha or not db.has(sha):
        return None
    flat = db.flat(db.commit(sha).tree)
    base = (r.pages.get("path") or "/").strip("/")
    pre = base + "/" if base else ""
    rest = rest.split("?", 1)[0]
    cands = [rest + "index.html"] if rest.endswith("/") or not rest else [rest, rest + ".html", rest + "/index.html"]
    for c in cands:
        hit = flat.get(pre + c)
        if hit and hit[0] != "160000":
            return bytes(db.raw(hit[1])), c
    if not rest or rest.endswith("/"):
        for md in ("index.md", "README.md"):
            hit = flat.get(pre + rest + md)
            if hit:
                return _md_page(r.name, bytes(db.raw(hit[1])).decode("utf-8", "replace")), "index.html"
    return None


def web(ctx, host: str, path: str, method: str = "GET"):
    """GitHub-lite over HTTP: Pages sites (*.github.io), raw.githubusercontent.com files, and repository pages.
    -> HttpResult, or None when the request isn't about a repository here."""
    from ..utils.net import HttpResult, mime_of
    host = host.lower()
    if host.endswith(".github.io"):
        gh = github_of(ctx.world)
        owner = host[:-len(".github.io")]
        parts = path.lstrip("/").split("/", 1)
        tries = []
        if parts[0]:
            tries.append((parts[0], parts[1] if len(parts) > 1 else None))
        tries.append((host, path.lstrip("/")))
        for name, rest in tries:
            r = gh.get(owner, name)
            if r is None or not r.pages or (r.private and name != host):
                continue
            if rest is None:                   # /<repo> -> /<repo>/
                return HttpResult(301, {"Location": f"https://{host}/{name}/", "Content-Type": "text/html"},
                                  b"<html>\r\n<head><title>301 Moved Permanently</title></head>\r\n</html>\r\n")
            got = _pages_file(r, rest)
            if got is None:
                continue
            data, fname = got
            return HttpResult(200, {"Content-Type": mime_of(fname) + ("; charset=utf-8" if fname.endswith(".html")
                                                                      else ""), "Server": "GitHub.com"},
                              data)
        return HttpResult(404, {"Content-Type": "text/html; charset=utf-8", "Server": "GitHub.com"}, PAGES_404)
    gh = ctx.world.__dict__.get("github")
    if gh is None:
        return None
    if host == "raw.githubusercontent.com":
        m = re.match(r"^/([\w.-]+)/([\w.-]+)/(?:refs/heads/)?([^/]+)/(.+)$", path)
        if not m:
            return None
        r = gh.get(m.group(1), m.group(2), materialize=False)
        if r is None or r.private:
            return None if r is None else HttpResult(404, {"Content-Type": "text/plain"}, b"404: Not Found")
        db = self_db(gh)
        ref = m.group(3)
        sha = r.refs.get(f"refs/heads/{ref}") or r.refs.get(f"refs/tags/{ref}") or (ref if db.has(ref) else None)
        if sha and db.kind(sha) == "tag":
            sha = db.tag(sha).object
        hit = db.flat(db.commit(sha).tree).get(m.group(4)) if sha and db.kind(sha) == "commit" else None
        if not hit:
            return HttpResult(404, {"Content-Type": "text/plain; charset=utf-8"}, b"404: Not Found")
        return HttpResult(200, {"Content-Type": "text/plain; charset=utf-8"}, bytes(db.raw(hit[1])))
    if host == "github.com":
        m = re.match(r"^/([\w.-]+)/([\w.-]+?)(?:\.git)?/?$", path)
        if not m:
            return None
        r = gh.get(m.group(1), m.group(2), materialize=False)
        if r is None:
            return None
        if r.private:
            return HttpResult(404, {"Content-Type": "text/html"}, b"<html><body>Not Found</body></html>\n")
        body = (f"<!DOCTYPE html>\n<html lang=\"en\">\n<head><title>GitHub - {r.full}: {r.description}</title></head>\n"
                f"<body>\n<h1>{r.full}</h1>\n<p>{r.description}</p>\n</body>\n</html>\n").encode()
        return HttpResult(200, {"Content-Type": "text/html; charset=utf-8", "Server": "GitHub.com"}, body)
    return None
