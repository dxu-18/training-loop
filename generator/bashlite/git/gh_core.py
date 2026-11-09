"""The gh CLI against GitHub-lite: auth, repo, ssh-key, browse, config, api (the Pages endpoint). Pull requests,
issues, runs and releases are in gh_pr.py. Prompts are line-based (type the answer, or Enter for the default)."""

from __future__ import annotations

import posixpath
import re

from ..exec import util
from ..vfs import FSError
from .cli import G, parse, run_maybe
from .github import github_of
from .objects import GitError

GH_COMMANDS: dict = {}


def gh_command(*names):
    def reg(fn):
        for n in names:
            GH_COMMANDS[n] = fn
        return fn
    return reg


class GhError(Exception):
    def __init__(self, msg, status=1):
        super().__init__(msg)
        self.msg, self.status = msg, status


def ask(g, question, default=""):
    """`? question (default)` and a typed line (generator)."""
    g.out(f"? {question}" + (f" ({default}) " if default else " "))
    g.ctx._flush_nb()
    rd = g.ctx.reader("-")
    line = yield from rd.readline()
    ans = (line or b"").decode("utf-8", "replace").strip()
    return ans or default


def logged_in(gh):
    acc = gh.accounts.get(gh.me) if gh.me else None
    return acc if acc is not None and acc.tokens else None


def need_login(gh):
    acc = logged_in(gh)
    if acc is None:
        raise GhError("To get started with GitHub CLI, please run:  gh auth login\nAlternatively, populate the GH_TOKEN "
                      "environment variable with a GitHub API authentication token.", 4)
    return acc


def protocol(g, gh):
    return gh.__dict__.get("git_protocol") or "https"


def clone_url(gh, g, full):
    return f"git@github.com:{full}.git" if protocol(g, gh) == "ssh" else f"https://github.com/{full}.git"


def current_repo(g, gh, flag=None):
    """-R owner/repo, or the GitHub repository the current git repository's remote points to."""
    if flag:
        owner, _, name = flag.partition("/")
        r = gh.get(owner, name)
        if r is None:
            raise GhError(f"GraphQL: Could not resolve to a Repository with the name '{flag}'. (repository)")
        return r
    repo = g.maybe_repo()
    if repo is None:
        raise GhError("failed to run git: fatal: not a git repository (or any of the parent directories): .git")
    from .remote import parse_url
    names = repo.remotes()
    order = [n for n in ("upstream", "github", "origin") if n in names] + [n for n in names if n not in
                                                                           ("upstream", "github", "origin")]
    for n in order:
        url = repo.config.get(f"remote.{n}.url") or ""
        ep = parse_url(g.ctx, url)
        if ep.kind == "github":
            r = gh.get(ep.owner, ep.repo)
            if r is not None:
                return r
    raise GhError("none of the git remotes configured for this repository point to a known GitHub host. To tell gh "
                  "about a new GitHub host, please use `gh auth login`")


def write_hosts(g, gh, acc, proto):
    home = g.env.get("HOME", "/")
    text = f"github.com:\n    users:\n        {acc.login}:\n    git_protocol: {proto}\n    user: {acc.login}\n"
    g.fs.mkdir(f"{home}/.config/gh", "/", 0o755, g.ctx.uid, g.ctx.proc.gid, parents=True)
    g.fs.write_file(f"{home}/.config/gh/hosts.yml", text, "/", 0o600, g.ctx.uid, g.ctx.proc.gid)


@util("gh")
def u_gh(ctx):
    a = list(ctx.argv[1:])
    g = G(ctx)
    gh = github_of(ctx.world)
    if not a or a[0] in ("-h", "--help", "help"):
        ctx.w("Work seamlessly with GitHub from the command line.\n\nUSAGE\n  gh <command> <subcommand> [flags]\n\n"
              "CORE COMMANDS\n  auth:        Authenticate gh and git with GitHub\n  browse:      Open the repository in "
              "the browser\n  issue:       Manage issues\n  pr:          Manage pull requests\n  release:     Manage "
              "releases\n  repo:        Manage repositories\n\nGITHUB ACTIONS COMMANDS\n  run:         View details "
              "about workflow runs\n  workflow:    View details about GitHub Actions workflows\n\nADDITIONAL COMMANDS\n"
              "  api:         Make an authenticated GitHub API request\n  config:      Manage configuration for gh\n"
              "  ssh-key:     Manage SSH keys\n")
        return 0
    if a[0] in ("--version", "version"):
        ctx.w("gh version 2.45.0 (2024-03-04)\nhttps://github.com/cli/cli/releases/tag/v2.45.0\n")
        return 0
    group = a[0]
    sub = a[1] if len(a) > 1 else ""
    fn = GH_COMMANDS.get(f"{group} {sub}") or GH_COMMANDS.get(group)
    rest = a[2:] if f"{group} {sub}" in GH_COMMANDS else a[1:]
    if fn is None:
        ctx.e(f"unknown command \"{group}\" for \"gh\"\n" if group not in {k.split()[0] for k in GH_COMMANDS} else
              f"unknown command \"{sub}\" for \"gh {group}\"\n")
        return 1
    try:
        st = fn(g, gh, rest)
        if hasattr(st, "send"):
            st = yield from st
        return st if isinstance(st, int) else 0
    except GhError as e:
        ctx._flush_nb()
        ctx.e(e.msg + "\n")
        return e.status
    except GitError as e:
        ctx._flush_nb()
        ctx.e(e.msg + "\n")
        return 1


# ---------------------------------------------------------------------------- auth

@gh_command("auth login")
def gh_auth_login(g, gh, args):
    o = parse(args, "hostname|h= git-protocol|p= web|w with-token skip-ssh-key scopes|s= insecure-storage")
    login = gh.__dict__.get("login_as") or "ada"
    acc = gh.account(login)
    if o.get("with-token"):
        data = yield from g.ctx.read_fd(0)
        tok = data.decode().strip()
        found = gh.account_for_token(tok)
        if found is None:
            raise GhError("error validating token: HTTP 401: Bad credentials (https://api.github.com/)")
        gh.me = found.login
        gh.git_protocol = o.get("git-protocol") or "https"
        write_hosts(g, gh, found, gh.git_protocol)
        return 0
    proto = (o.get("git-protocol") or "").lower()
    if not proto:
        g.out("? Where do you use GitHub? GitHub.com\n")
        ans = yield from ask(g, "What is your preferred protocol for Git operations on this host? [SSH/HTTPS]", "SSH")
        proto = "https" if ans.lower().startswith("h") else "ssh"
    else:
        g.out("? Where do you use GitHub? GitHub.com\n")
    home = g.env.get("HOME", "/")
    upload = None
    if proto == "ssh" and not o.get("skip-ssh-key"):
        pubs = [f"{home}/.ssh/{n}" for n in ("id_ed25519.pub", "id_rsa.pub", "id_ecdsa.pub")
                if g.fs.exists(f"{home}/.ssh/{n}")]
        if pubs:
            ans = yield from ask(g, "Upload your SSH public key to your GitHub account?", pubs[0])
            if ans.lower() not in ("skip", "n", "no"):
                upload = ans
        else:
            ans = yield from ask(g, "Generate a new SSH key to add to your GitHub account? [Y/n]", "Y")
            if not ans.lower().startswith("n"):
                yield from g.ctx.run(["ssh-keygen", "-t", "ed25519", "-N", "", "-f", f"{home}/.ssh/id_ed25519", "-q"])
                upload = f"{home}/.ssh/id_ed25519.pub"
        if upload:
            yield from ask(g, "Title for your SSH key:", "GitHub CLI")
    if not o.get("web"):
        yield from ask(g, "How would you like to authenticate GitHub CLI? [web/token]", "Login with a web browser")
    code = "4B2A-91C7"
    g.out(f"\n! First copy your one-time code: {code}\nPress Enter to open github.com in your browser... ")
    g.ctx._flush_nb()
    rd = g.ctx.reader("-")
    yield from rd.readline()
    g.out("✓ Authentication complete.\n")
    gh.me = acc.login
    gh.new_token(acc.login)
    gh.git_protocol = proto
    write_hosts(g, gh, acc, proto)
    g.out(f"- gh config set -h github.com git_protocol {proto}\n✓ Configured git protocol\n")
    if upload:
        try:
            text = bytes(g.ctx.lookup(upload).data).decode()
        except FSError:
            raise GhError(f"could not read {upload}: no such file or directory")
        from ..utils.net import parse_pub
        p = parse_pub(text)
        if p:
            if not any(k == p[1] for _, k in acc.keys):
                acc.keys.append(("GitHub CLI", p[1]))
            g.out(f"✓ Uploaded the SSH key to your GitHub account: {upload}\n")
    g.out(f"✓ Logged in as {acc.login}\n")
    return 0


@gh_command("auth status")
def gh_auth_status(g, gh, args):
    acc = logged_in(gh)
    if acc is None:
        g.err("You are not logged into any GitHub hosts. To log in, run: gh auth login\n")
        return 1
    tok = sorted(acc.tokens)[0]
    g.out(f"github.com\n  ✓ Logged in to github.com account {acc.login} (keyring)\n  - Active account: true\n"
          f"  - Git operations protocol: {protocol(g, gh)}\n  - Token: {tok[:4]}{'*' * (len(tok) - 4)}\n"
          "  - Token scopes: 'gist', 'read:org', 'repo', 'workflow'\n")
    return 0


@gh_command("auth logout")
def gh_auth_logout(g, gh, args):
    acc = logged_in(gh)
    if acc is None:
        raise GhError("not logged in to any hosts")
    acc.tokens.clear()
    gh.me = None
    g.out(f"✓ Logged out of github.com account {acc.login}\n")
    return 0


@gh_command("auth token")
def gh_auth_token(g, gh, args):
    acc = need_login(gh)
    g.out(sorted(acc.tokens)[0] + "\n")
    return 0


@gh_command("auth setup-git")
def gh_auth_setup_git(g, gh, args):
    need_login(gh)
    from .repo import ConfigFile
    home = g.env.get("HOME", "/")
    cf = ConfigFile(g.fs, f"{home}/.gitconfig")
    cf.set("credential.https://github.com.helper", "", g.ctx.uid, g.ctx.proc.gid)
    cf.set("credential.https://github.com.helper", "!/usr/bin/gh auth git-credential", g.ctx.uid, g.ctx.proc.gid,
           add=True)
    cf.set("credential.helper", "!/usr/bin/gh auth git-credential", g.ctx.uid, g.ctx.proc.gid)
    return 0


# ---------------------------------------------------------------------------- ssh-key, config, browse

@gh_command("ssh-key add")
def gh_ssh_key_add(g, gh, args):
    o = parse(args, "title|t= type=")
    acc = need_login(gh)
    if not o.args:
        raise GhError("public key file missing")
    try:
        text = bytes(g.ctx.lookup(o.args[0]).data).decode()
    except FSError:
        raise GhError(f"open {o.args[0]}: no such file or directory")
    from ..utils.net import parse_pub
    p = parse_pub(text)
    if p is None or o.args[0].endswith(("_ed25519", "_rsa")):
        raise GhError("HTTP 422: Validation Failed (https://api.github.com/user/keys)\nkey is invalid. You must supply a "
                      "key in OpenSSH public key format")
    if any(k == p[1] for _, k in acc.keys):
        g.out("✓ Public key already exists on your account\n")
        return 0
    acc.keys.append((o.get("title") or "", p[1]))
    g.out("✓ Public key added to your account\n")
    return 0


@gh_command("ssh-key list")
def gh_ssh_key_list(g, gh, args):
    acc = need_login(gh)
    for title, blob in acc.keys:
        import base64
        g.out(f"{title or '(no title)'}\tssh-ed25519 {base64.b64encode(blob).decode()[:20]}...\tauthentication\n")
    return 0


@gh_command("config set")
def gh_config_set(g, gh, args):
    o = parse(args, "host|h=")
    if len(o.args) < 2:
        raise GhError("usage: gh config set <key> <value>")
    if o.args[0] == "git_protocol":
        gh.git_protocol = o.args[1]
    return 0


@gh_command("config get")
def gh_config_get(g, gh, args):
    o = parse(args, "host|h=")
    if o.args and o.args[0] == "git_protocol":
        g.out(protocol(g, gh) + "\n")
    return 0


@gh_command("browse")
def gh_browse(g, gh, args):
    o = parse(args, "repo|R= no-browser|n settings|s wiki|w projects|p branch|b= commit|c=?")
    r = current_repo(g, gh, o.get("repo"))
    path = "/settings" if o.get("settings") else ""
    url = f"github.com/{r.full}{path}"
    if o.get("no-browser"):
        g.out(f"https://{url}\n")
    else:
        g.out(f"Opening {url} in your browser.\n")
    return 0


# ---------------------------------------------------------------------------- repo

@gh_command("repo create")
def gh_repo_create(g, gh, args):
    o = parse(args, "public private internal source|s= remote|r= push description|d= clone|c homepage|h= "
                    "add-readme gitignore|g= license|l= template|p= disable-issues disable-wiki team|t=")
    acc = need_login(gh)
    name = o.args[0] if o.args else None
    if o.get("source") and not name:
        src = o.get("source")
        path = src if src.startswith("/") else posixpath.normpath(posixpath.join(g.cwd, src))
        name = posixpath.basename(path)
    if not name:
        raise GhError("at least one argument required in non-interactive mode")
    owner = acc.login
    if "/" in name:
        owner, name = name.split("/", 1)
    if not (o.get("public") or o.get("private") or o.get("internal")):
        raise GhError("--public, --private, or --internal required when not running interactively")
    r = gh.create(owner, name, o.get("description") or "", private=not o.get("public"))
    g.out(f"✓ Created repository {r.full} on GitHub\n  https://github.com/{r.full}\n")
    if o.get("add-readme"):
        from .objects import Person, encode_commit, odb
        db = odb(g.world)
        blob = db.put("blob", f"# {name}\n".encode())
        tree = db.write_tree({"README.md": ("100644", blob)})
        who = Person(acc.name, acc.email, gh.now(), "+0000")
        c = db.put("commit", encode_commit(tree, [], who, who, "Initial commit\n"))
        r.refs["refs/heads/main"] = c
    if o.get("source"):
        src = o.get("source")
        path = src if src.startswith("/") else posixpath.normpath(posixpath.join(g.cwd, src))
        remote = o.get("remote") or "origin"
        url = clone_url(gh, g, r.full)
        st = yield from g.ctx.run(["git", "-C", path, "remote", "add", remote, url])
        if st:
            return st
        g.out(f"✓ Added remote {url}\n")
        if o.get("push"):
            g.ctx._flush_nb()
            st = yield from g.ctx.run(["git", "-C", path, "push", "--set-upstream", remote, "HEAD"])
            if st:
                return st
            g.out(f"✓ Pushed commits to {url}\n")
    elif o.get("clone"):
        st = yield from g.ctx.run(["git", "clone", clone_url(gh, g, r.full)])
        return st
    return 0


@gh_command("repo clone")
def gh_repo_clone(g, gh, args):
    o = parse(args, "upstream-remote-name|u=", stop_at_nonopt=False)
    if not o.args:
        raise GhError("cannot clone: repository argument required")
    spec = o.args[0]
    if "/" not in spec:
        acc = need_login(gh)
        spec = f"{acc.login}/{spec}"
    m = re.match(r"^(?:https://github\.com/|git@github\.com:)?([\w.-]+)/([\w.-]+?)(?:\.git)?$", spec)
    if not m:
        raise GhError(f"invalid repository: {spec}")
    owner, name = m.group(1), m.group(2)
    r = gh.get(owner, name)
    extra = list(o.args[1:]) + list(o.paths or [])
    url = clone_url(gh, g, f"{owner}/{name}")
    if r is None:
        url = f"https://github.com/{owner}/{name}.git"
    st = yield from g.ctx.run(["git", "clone", url] + extra)
    if st == 0 and r is not None and r.fork_of:
        target = extra[0] if extra and not extra[0].startswith("-") else name
        up = clone_url(gh, g, r.fork_of)
        yield from g.ctx.run(["git", "-C", target, "remote", "add", o.get("upstream-remote-name") or "upstream", up])
        yield from g.ctx.run(["git", "-C", target, "fetch", "-q", o.get("upstream-remote-name") or "upstream"])
        g.err(f"! Repository {r.fork_of} set as the default repository. To learn more or change this selection, see: "
              "gh repo set-default\n")
    return st


@gh_command("repo view")
def gh_repo_view(g, gh, args):
    o = parse(args, "web|w branch|b= json= jq|q=")
    r = current_repo(g, gh, o.args[0] if o.args else None)
    if o.get("web"):
        g.out(f"Opening github.com/{r.full} in your browser.\n")
        return 0
    from .objects import odb
    db = odb(g.world)
    readme = ""
    head = r.refs.get(f"refs/heads/{r.default_branch}")
    if head:
        flat = db.flat(db.commit(head).tree)
        for cand in ("README.md", "README.rst", "README.txt", "README"):
            if cand in flat:
                readme = bytes(db.raw(flat[cand][1])[:4000]).decode("utf-8", "replace")
                break
    g.out(f"{r.full}\n{r.description or 'No description provided'}\n\n" + (readme.rstrip() + "\n\n" if readme else
          "This repository does not have a README\n\n") + f"View this repository on GitHub: https://github.com/{r.full}\n")
    return 0


@gh_command("repo fork")
def gh_repo_fork(g, gh, args):
    o = parse(args, "clone remote remote-name= fork-name= default-branch-only org=")
    acc = need_login(gh)
    in_repo = not o.args
    src = current_repo(g, gh, o.args[0] if o.args else None)
    if src.owner == acc.login:
        raise GhError(f"failed to fork: {src.full} cannot be forked: it is your own repository")
    fork = gh.fork(src, o.get("org") or acc.login)
    g.out(f"✓ Created fork {fork.full}\n")
    add_remote = bool(o.get("remote"))
    if in_repo and not add_remote:
        ans = yield from ask(g, "Would you like to add a remote for the fork? [Y/n]", "Y")
        add_remote = not ans.lower().startswith("n")
    if in_repo and add_remote:
        repo = g.repo()
        url = clone_url(gh, g, fork.full)
        name = o.get("remote-name") or "origin"
        if name in repo.remotes():
            yield from g.ctx.run(["git", "remote", "rename", name, "upstream"])
            g.out(f"✓ Renamed remote {name} to upstream\n")
        yield from g.ctx.run(["git", "remote", "add", name, url])
        g.out(f"✓ Added remote {name}\n")
    elif o.get("clone"):
        st = yield from g.ctx.run(["git", "clone", clone_url(gh, g, fork.full)])
        if st:
            return st
        yield from g.ctx.run(["git", "-C", fork.name, "remote", "add", "upstream", clone_url(gh, g, src.full)])
        yield from g.ctx.run(["git", "-C", fork.name, "fetch", "-q", "upstream"])
        g.out(f"✓ Cloned fork\n")
    return 0


@gh_command("repo list")
def gh_repo_list(g, gh, args):
    o = parse(args, "limit|L= public private fork source language|l= json= visibility=")
    owner = o.args[0] if o.args else need_login(gh).login
    repos = [r for r in gh.repos.values() if r.owner == owner]
    if not repos:
        g.out(f"\nThere are no repositories in @{owner}\n\n")
        return 0
    g.out(f"\nShowing {len(repos)} of {len(repos)} repositories in @{owner}\n\n")
    w = max(len(r.full) for r in repos)
    g.out(f"{'NAME':<{w}}  {'DESCRIPTION':<30}  INFO\n")
    for r in sorted(repos, key=lambda r: -r.pushed):
        info = "private" if r.private else ("public, fork" if r.fork_of else "public")
        g.out(f"{r.full:<{w}}  {(r.description or '')[:30]:<30}  {info}\n")
    return 0


@gh_command("repo delete")
def gh_repo_delete(g, gh, args):
    o = parse(args, "yes|confirm")
    acc = need_login(gh)
    r = current_repo(g, gh, o.args[0] if o.args else None)
    if r.owner != acc.login:
        raise GhError(f"HTTP 403: Must have admin rights to Repository. (https://api.github.com/repos/{r.full})")
    if not o.get("yes"):
        ans = yield from ask(g, f"Type {r.full} to confirm deletion:")
        if ans != r.full:
            raise GhError("confirmation did not match repository name")
    gh.delete(r)
    g.out(f"✓ Deleted repository {r.full}\n")
    return 0


@gh_command("repo sync")
def gh_repo_sync(g, gh, args):
    o = parse(args, "source|s= branch|b= force")
    acc = need_login(gh)
    r = current_repo(g, gh, o.args[0] if o.args else None) if o.args else None
    repo = g.maybe_repo()
    if r is None and repo is not None:
        # local: fast-forward the default branch from the remote
        st = yield from g.ctx.run(["git", "pull", "--ff-only", "-q"])
        if st == 0:
            g.out(f"✓ Synced the \"{repo.head_branch()}\" branch from \"origin\" to local repository\n")
        return st
    if r is None or not r.fork_of:
        raise GhError("can't determine source repository; use --source")
    parent = gh.get(*r.fork_of.split("/"))
    b = o.get("branch") or parent.default_branch
    r.refs[f"refs/heads/{b}"] = parent.refs[f"refs/heads/{b}"]
    g.out(f"✓ Synced the \"{r.owner}:{b}\" branch from \"{parent.full}:{b}\"\n")
    return 0


# ---------------------------------------------------------------------------- api (Pages)

@gh_command("api")
def gh_api(g, gh, args):
    o = parse(args, "method|X= field|f= raw-field|F= header|H= jq|q= paginate silent")
    need_login(gh)
    if not o.args:
        raise GhError("usage: gh api <endpoint> [flags]")
    ep = o.args[0].lstrip("/")
    m = re.match(r"^repos/([\w.-]+)/([\w.-]+)/pages$", ep)
    fields = {}
    for f in o.all("field") + o.all("raw-field"):
        k, _, v = f.partition("=")
        fields[k] = v
    if m:
        r = gh.get(m.group(1), m.group(2))
        if r is None:
            raise GhError("gh: Not Found (HTTP 404)")
        method = (o.get("method") or ("POST" if fields else "GET")).upper()
        if method in ("POST", "PUT"):
            br = fields.get("source[branch]") or fields.get("source.branch") or r.default_branch
            path = fields.get("source[path]") or fields.get("source.path") or "/"
            r.pages = {"branch": br, "path": path, "built": r.refs.get(f"refs/heads/{br}")}
            g.out('{"url":"https://api.github.com/repos/%s/pages","status":"queued","html_url":"https://%s.github.io/%s/",'
                  '"source":{"branch":"%s","path":"%s"}}\n' % (r.full, r.owner, r.name, br, path))
            return 0
        if r.pages is None:
            raise GhError("gh: Not Found (HTTP 404)")
        g.out('{"html_url":"https://%s.github.io/%s/","status":"built","source":{"branch":"%s","path":"%s"}}\n'
              % (r.owner, r.name, r.pages.get("branch"), r.pages.get("path")))
        return 0
    if ep == "user":
        acc = need_login(gh)
        g.out('{"login":"%s","name":"%s"}\n' % (acc.login, acc.name))
        return 0
    raise GhError(f"gh: Not Found (HTTP 404)")


__all__ = ["GH_COMMANDS", "gh_command", "GhError", "current_repo", "need_login", "clone_url", "run_maybe"]
