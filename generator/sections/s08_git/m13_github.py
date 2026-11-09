"""8.13 GitHub: gh CLI, pull requests, Actions & Pages."""

from ._common import GH, GIT, P, Repo, catalog, heredoc, ml_files, q, setup, template, WORDS

M = "8.13"


def _own(name, src):
    return "\n".join([GIT, GH, f"#: ghrepo ada/{name} from={src}", f"#: cwd ~/{name}",
                      f"git clone -q git@github.com:ada/{name}.git ."])


# ================================================================= lite

@template(M, "lite")
def gh_look(rng):
    items = [
        ("gh repo view", "Show the GitHub page of `minilibs/requests-mini` in the terminal (`gh`).",
         "gh repo view minilibs/requests-mini"),
        ("gh pr list", "List the open pull requests of `acme-corp/kanban-board`.",
         "gh pr list -R acme-corp/kanban-board", {"require": ["gh pr list"]}),
        ("gh issue list", "List the open issues of `acme-corp/kanban-board`.", "gh issue list -R acme-corp/kanban-board",
         {"require": ["gh issue list"]}),
        ("gh pr view", "Show pull request #15 of `minilibs/requests-mini`.", "gh pr view 15 -R minilibs/requests-mini",
         {"require": ["gh pr view"]}),
        ("gh auth status", "Check which GitHub account `gh` is signed in as.", "gh auth status"),
        ("gh release list", "List the releases of `minilibs/requests-mini`.",
         "gh release list -R minilibs/requests-mini", {"require": ["gh release list"]}),
    ]
    return catalog(rng, items, setup(GIT, GH))


@template(M, "lite")
def gh_create(rng):
    name = rng.choice(["comp-guide", "ml-notes", "dsa-practice", "my-site"])
    vis = rng.choice(["public", "private"])
    r = Repo(rng).commit("Initial commit", ml_files(rng, 3))
    return P(f"Create a {vis} GitHub repository `{name}` from this folder with `gh`, adding it as the remote "
             f"`origin` and pushing in the same command.", setup(GIT, GH, f"#: cwd ~/{name}", r.code()),
             f"gh repo create {name} --{vis} --source=. --remote=origin --push",
             check=(["remotes", "upstream", f"gh=ada/{name}"], "."), require=["gh repo create"])


# ================================================================= easy

@template(M, "easy")
def open_pr(rng):
    b = rng.choice(["docs/usage", "feature/retries-config", "fix/typo"])
    title = rng.choice(["Document the timeout option", "Make retries configurable", "Fix a typo in the README"])
    s = _own("requests-mini", "minilibs/requests-mini")
    return P(f"Make a branch `{b}`, append the line `{title}.` to `README.md`, commit it as `{title}`, push the "
             f"branch (with upstream) and open a pull request into main titled `{title}` with the body `See the "
             f"README.`.", s,
             f"git switch -c {b}\necho '{title}.' >> README.md\ngit commit -am {q(title)}\ngit push -u origin {b}\n"
             f"gh pr create --title {q(title)} --body 'See the README.' --base main",
             check=(["head", "upstream", "gh=ada/requests-mini"], "."), require=["gh pr create"])


@template(M, "easy")
def issues(rng):
    t = rng.choice(["Broken link in the README", "Add a CONTRIBUTING guide", "Cards flicker when dragged"])
    items = [
        ("gh issue create", f"Open an issue titled `{t}` with the body `Found while testing.`.",
         f"gh issue create --title {q(t)} --body 'Found while testing.'"),
        ("gh issue close", "Issue #8 (the typo) is fixed. Close it.", "gh issue close 8"),
        ("gh issue comment", "Comment `I can take this one.` on issue #6.",
         "gh issue comment 6 --body 'I can take this one.'"),
    ]
    return catalog(rng, items, _own("kanban-board", "acme-corp/kanban-board"),
                   check=(["gh=ada/kanban-board"], "."))


# ================================================================= medium

@template(M, "medium")
def review_merge(rng):
    how = rng.choice([("--squash", "squash"), ("--merge", "merge"), ("--rebase", "rebase")])
    return P(f"You maintain `ada/kanban-board`. Approve pull request #11 (`Drag and drop columns`) with `gh`, then "
             f"merge it with the {how[1]} method and delete its branch.",
             _own("kanban-board", "acme-corp/kanban-board"),
             f"gh pr review 11 --approve\ngh pr merge 11 {how[0]} --delete-branch",
             check=(["head", "files", "gh=ada/kanban-board"], "."), require=["gh pr review", "gh pr merge"])


@template(M, "medium")
def pr_local(rng):
    items = [
        ("gh pr checkout", "Check out pull request #11's branch locally to try it.", "gh pr checkout 11",
         {"check": (["head", "branches"], ".")}),
        ("gh pr diff", "Show the diff of pull request #12.", "gh pr diff 12", {"answer": ["<stdout>"]}),
        ("gh pr status", "Show the status of pull requests relevant to you.", "gh pr status",
         {"answer": ["<stdout>"]}),
    ]
    return catalog(rng, items, setup(GIT, GH, "#: cwd ~/kanban-board",
                                     "git clone -q git@github.com:acme-corp/kanban-board.git ."))


# ================================================================= hard

@template(M, "hard")
def pages(rng):
    name = rng.choice(["portfolio", "course-notes", "project-site"])
    title = rng.choice(["Hello, Pages", "My notes", "Project home"])
    r = Repo(rng).commit("Initial commit", ml_files(rng, 2))
    s = setup(GIT, GH, f"#: ghrepo ada/{name}", f"#: cwd ~/{name}", r.code(),
              f"git remote add origin git@github.com:ada/{name}.git", "git push -q -u origin main")
    return P(f"Publish a one-page site: add `docs/index.html` containing `<h1>{title}</h1>`, commit it as `Add "
             f"site`, push, and turn on GitHub Pages for `main` with the `/docs` folder (`gh api -X POST "
             f"repos/ada/{name}/pages -f \"source[branch]=main\" -f \"source[path]=/docs\"`). Then fetch "
             f"`https://ada.github.io/{name}/` with curl.", s,
             f"mkdir docs\necho '<h1>{title}</h1>' > docs/index.html\ngit add docs\ngit commit -m 'Add site'\n"
             f"git push\ngh api -X POST repos/ada/{name}/pages -f 'source[branch]=main' -f 'source[path]=/docs'\n"
             f"curl -s https://ada.github.io/{name}/",
             post=f"/usr/lib/bashlite/gitstate gh=ada/{name}; curl -s https://ada.github.io/{name}/",
             answer=["<post>"], require=["gh api", "curl"])


@template(M, "hard")
def fix_ci(rng):
    return P("The pull request for `feature/pagination` in your `ada/express-api` fails its check: "
             "`test/pagination.test.js` calls `assert.fail`. Check out the branch, replace that test file's "
             "`assert.fail(...)` line with `assert.ok(true);`, commit as `test: fix pagination test`, and push.",
             _own("express-api", "acme-corp/express-api"),
             "git switch feature/pagination\nsed -i \"s/  assert.fail('pagination returns page 1 twice');/  "
             "assert.ok(true);/\" test/pagination.test.js\ngit commit -am 'test: fix pagination test'\ngit push",
             check=(["head", "files", "gh=ada/express-api"], "."), require=["git push"])


# ================================================================= challenge

@template(M, "challenge")
def full_flow(rng):
    name = rng.choice(["comp-guide", "ml-utils", "notes-site"])
    b = rng.choice(["docs/readme", "feature/cli"])
    r = Repo(rng).commit("Initial commit", ml_files(rng, 3))
    return P(f"**Ship it.** Create a public GitHub repository `{name}` from this folder with `gh` (remote "
             f"`origin`, pushed). Then on a branch `{b}`, append `More docs.` to `README.md`, commit `docs: more "
             f"docs`, push it, open a PR titled `More docs` (body `Adds docs.`), squash-merge it deleting the "
             f"branch, and create a release `v1.0` with the notes `First public version`.",
             setup(GIT, GH, f"#: cwd ~/{name}", r.code() + "\necho '# Project' > README.md && git add README.md && "
                                                           "git commit -qm 'Add README'"),
             f"gh repo create {name} --public --source=. --remote=origin --push\ngit switch -c {b}\n"
             f"echo 'More docs.' >> README.md\ngit commit -am 'docs: more docs'\ngit push -u origin {b}\n"
             f"gh pr create --title 'More docs' --body 'Adds docs.'\ngh pr merge --squash --delete-branch\n"
             f"gh release create v1.0 --notes 'First public version'",
             check=(["head", "files", f"gh=ada/{name}"], "."), require=["gh pr create", "gh pr merge"])


@template(M, "challenge")
def first_contribution(rng):
    name = rng.choice(["Ada Lovelace", "A. Lovelace"])
    return P(f"**Your first open-source PR.** Fork `open-learning/first-contributions` with `gh` and clone your "
             f"fork, make a branch `add-ada`, add the line `- {name}` at the end of `Contributors.md`, commit `Add "
             f"{name} to Contributors`, push the branch to your fork, and open a pull request against the original "
             f"repository.", setup(GIT, GH),
             "gh repo fork open-learning/first-contributions --clone\ncd first-contributions\ngit switch -c add-ada\n"
             f"echo '- {name}' >> Contributors.md\ngit commit -am 'Add {name} to Contributors'\n"
             "git push -u origin add-ada\ngh pr create --repo open-learning/first-contributions --head ada:add-ada "
             f"--title 'Add {name} to Contributors' --body 'My first contribution.'",
             post="/usr/lib/bashlite/gitstate -C ~/first-contributions head remotes ghme "
                  "gh=open-learning/first-contributions | grep -v '^  pr #[0-9]* .*MERGED'",
             answer=["<post>"], require=["gh repo fork", "gh pr create"])
