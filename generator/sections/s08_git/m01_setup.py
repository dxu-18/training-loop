"""8.01 Setup & configuration."""

from ._common import GH, GIT, PEOPLE, P, Repo, catalog, ml_files, q, setup, template

M = "8.01"


def _who(rng):
    name, email = rng.choice(PEOPLE[1:])
    return name, email


# ================================================================= lite

@template(M, "lite")
def set_identity(rng):
    name, email = _who(rng)
    editor = rng.choice(["vim", "nano", "code --wait"])
    branch = rng.choice(["main", "trunk", "develop"])
    items = [
        ("git config", f"Set your name for every repository to `{name}` (global config).",
         f"git config --global user.name {q(name)}", {"post": "git config --global user.name"}),
        ("git config", f"Set your email for every repository to `{email}`.",
         f"git config --global user.email {email}", {"post": "git config --global user.email"}),
        ("git config", f"Make `{branch}` the name of the first branch in every new repository.",
         f"git config --global init.defaultBranch {branch}", {"post": "git config --global init.defaultBranch"}),
        ("git config", f"Make `{editor}` the editor git opens for commit messages.",
         f"git config --global core.editor {q(editor)}", {"post": "git config --global core.editor"}),
        ("git config", "Make `git pull` rebase your local commits instead of creating merge commits.",
         "git config --global pull.rebase true", {"post": "git config --global pull.rebase"}),
        ("git config", "Make the first `git push` of a new branch set its upstream automatically.",
         "git config --global push.autoSetupRemote true", {"post": "git config --global push.autoSetupRemote"}),
        ("git config", "Make every fetch forget remote branches that were deleted on the server.",
         "git config --global fetch.prune true", {"post": "git config --global fetch.prune"}),
        ("git config", "Turn on rerere, so git remembers how you resolved a conflict.",
         "git config --global rerere.enabled true", {"post": "git config --global rerere.enabled"}),
        ("git config", "Show the common ancestor in conflict markers (conflict style `zdiff3`).",
         "git config --global merge.conflictstyle zdiff3", {"post": "git config --global merge.conflictstyle"}),
    ]
    return catalog(rng, items, setup("#: git noident"), answer=["<post>"])


@template(M, "lite")
def read_config(rng):
    name, email = _who(rng)
    items = [
        ("git config", "Print the email git will use for commits (just the value).", "git config user.email"),
        ("git config", "Print your global user name.", "git config --global user.name"),
        ("git config --list", "List every setting git sees.", "git config --list"),
        ("git config --global", "List only your global settings.", "git config --global --list",
         {"require": ["git config --global"]}),
        ("git config", "Print the default branch name new repositories get.", "git config init.defaultBranch"),
        ("git --version", "Print which version of git is installed.", "git --version"),
    ]
    return catalog(rng, items, setup(f"#: git name={name.replace(' ', '_')} email={email} branch=main"))


# ================================================================= easy

@template(M, "easy")
def local_override(rng):
    name, email = _who(rng)
    work = f"{name.split()[0].lower()}@{rng.choice(['acme.dev', 'lab.example.org', 'corp.example.com'])}"
    r = Repo(rng).commit("Initial commit", ml_files(rng, 2))
    s = setup(GIT, "#: cwd ~/work/project", r.code())
    return P(f"This is a work project. Give **this repository only** the email `{work}` (your global email stays "
             f"`ada@example.com`), then print the email git will use here.",
             s, f"git config user.email {work}\ngit config user.email",
             post="git config --local user.email; git config --global user.email", answer=["<stdout>", "<post>"],
             require=["git config"])


@template(M, "easy")
def alias(rng):
    pick_ = rng.choice([("lg", "log --oneline --graph --decorate --all"), ("st", "status -sb"), ("co", "checkout"),
                        ("last", "log -1 HEAD"), ("unstage", "restore --staged")])
    a, cmd = pick_
    s = setup(GIT, "#: cwd ~/proj", Repo(rng).commit("Initial commit", ml_files(rng, 2)).commit(
        "Add notes", {"notes.md": "todo\n"}).code())
    use = {"lg": "git lg", "st": "git st", "co": "git co main", "last": "git last", "unstage": "git unstage notes.md"}[a]
    return P(f"Create a global alias so that `git {a}` runs `git {cmd}`, then try it with `{use}`.",
             s, f"git config --global alias.{a} {q(cmd)}\n{use}", post=f"git config --global alias.{a}",
             answer=["<post>"], require=["git config"])


# ================================================================= medium

@template(M, "medium")
def ssh_to_github(rng):
    title = rng.choice(["laptop", "work-laptop", "desktop", "lab-machine"])
    s = setup(GIT, "#: github login")
    return P(f"Connect this machine to GitHub over SSH: make an ed25519 key at `~/.ssh/id_ed25519` (empty "
             f"passphrase, comment `ada@example.com`), add its public key to your GitHub account with `gh` (title "
             f"`{title}`), then test the connection with `ssh -T git@github.com` (accept GitHub's host key).",
             s, f"ssh-keygen -t ed25519 -C ada@example.com -f ~/.ssh/id_ed25519 -N ''\n"
                f"gh ssh-key add ~/.ssh/id_ed25519.pub --title {title}\nssh -o StrictHostKeyChecking=accept-new -T "
                f"git@github.com",
             post=f"{_state('ghme')}; test -f ~/.ssh/id_ed25519 && echo private key here; grep -c github.com "
                  f"~/.ssh/known_hosts", answer=["<post>"], require=["ssh-keygen", "ssh -T"])


def _state(*secs):
    from ._common import STATE
    return f"{STATE} {' '.join(secs)}"


@template(M, "medium")
def fix_config(rng):
    name, email = _who(rng)
    bad = email.replace("@", "@@")
    s = setup(f"#: git name={name.replace(' ', '_')} email={bad}",
              "git config --global alias.co 'checkot'", "git config --global core.editor nan")
    items = [
        ("git config", f"Your global email was typed wrong (`{bad}`). Set it to `{email}`, and remove the broken "
                       f"`co` alias completely.",
         f"git config --global user.email {email}\ngit config --global --unset alias.co",
         {"post": "git config --global --list", "require": ["git config --global"]}),
        ("git config", "Your global editor is set to a program that doesn't exist (`nan`). Set it to `nano`, and fix "
                       "the `co` alias so it runs `checkout`.",
         "git config --global core.editor nano\ngit config --global alias.co checkout",
         {"post": "git config --global --list"}),
    ]
    return catalog(rng, items, s, answer=["<post>"])


# ================================================================= hard

@template(M, "hard")
def fresh_machine(rng):
    name, email = _who(rng)
    branch = rng.choice(["main", "trunk"])
    d = rng.choice(["curriculum", "notes", "experiments", "thesis-code"])
    return P(f"A fresh machine. Configure git globally: name `{name}`, email `{email}`, default branch `{branch}`, "
             f"editor `vim`, and `git pull` should rebase. Then create a repository in a new directory `~/{d}`.",
             setup("#: git noident noconfig"),
             f"git config --global user.name {q(name)}\ngit config --global user.email {email}\n"
             f"git config --global init.defaultBranch {branch}\ngit config --global core.editor vim\n"
             f"git config --global pull.rebase true\ngit init ~/{d}",
             post=f"git config --global --list | sort; cd ~/{d} && git symbolic-ref HEAD", answer=["<post>"],
             require=["git config --global", "git init"])


@template(M, "hard")
def show_origin(rng):
    name, email = _who(rng)
    r = Repo(rng).commit("Initial commit", ml_files(rng, 2)).run(f"git config user.name {q(name)}")
    s = setup(GIT, "#: cwd ~/proj", r.code())
    items = [
        ("git config --show-origin", "Commits here come out with an unexpected author name. Show where git "
                                     "reads `user.name` from (the file it comes from, and the value).",
         "git config --show-origin user.name"),
        ("git config --show-origin", "List every setting together with the file it comes from, to find why this "
                                     "repository uses a different author name.",
         "git config --list --show-origin", {"require": ["git config --show-origin", "git config --list"]}),
        ("git config", "This repository overrides your name. Remove the repository's own `user.name`, so the "
                       "global one applies, and print the name git uses now.",
         "git config --unset user.name\ngit config user.name"),
    ]
    return catalog(rng, items, s)


# ================================================================= challenge

@template(M, "challenge")
def identity_unknown(rng):
    name, email = _who(rng)
    r = Repo(rng, init=True)
    for p, t in ml_files(rng, 3).items():
        r.write(p, t)
    r.run("git add -A")
    s = setup("#: git noident", "#: cwd ~/project", r.code())
    return P(f"**First commit.** Everything in `~/project` is staged, but committing fails because git doesn't "
             f"know who you are. Set your global identity to `{name}` / `{email}` and make the commit with the "
             f"message `Initial commit`.",
             s, f"git config --global user.name {q(name)}\ngit config --global user.email {email}\n"
                f"git commit -m 'Initial commit'",
             check=(["head", "log", "index"], "."), require=["git config", "git commit"])


@template(M, "challenge")
def ready_for_github(rng):
    repo = rng.choice(["dotfiles", "ml-notes", "playground", "comp-guide"])
    s = setup(GIT, "#: github login", "#: cwd ~/" + repo,
              Repo(rng).commit("Initial commit", ml_files(rng, 3)).code())
    return P(f"**Ready for GitHub.** You're signed in with `gh` but have no SSH key. Create an ed25519 key (empty "
             f"passphrase), add it to GitHub, set the alias `git lg` = `log --oneline --graph --decorate --all`, "
             f"create a public GitHub repository `{repo}` from this folder with `gh` (remote `origin`) and push "
             f"`main` over SSH.",
             s, "ssh-keygen -t ed25519 -C ada@example.com -f ~/.ssh/id_ed25519 -N ''\n"
                "gh ssh-key add ~/.ssh/id_ed25519.pub --title laptop\n"
                "git config --global alias.lg 'log --oneline --graph --decorate --all'\n"
                f"gh repo create {repo} --public --source=. --remote=origin --push",
             post=f"{_state('ghme', f'gh=ada/{repo}')}; git config --global alias.lg; git remote get-url origin",
             answer=["<post>"], require=["ssh-keygen", "gh repo create"])
