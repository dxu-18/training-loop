"""8.14 Git for ML projects & commit conventions."""

from ._common import GH, GIT, P, Repo, catalog, heredoc, ml_files, q, setup, template, WORDS

M = "8.14"
VENV = "#: venv /home/ada/.venv {pkgs}\n#: activate /home/ada/.venv"


def _proj(rng):
    r = Repo(rng).commit("Initial commit", {**ml_files(rng, 3), ".gitignore": "outputs/\n"})
    r.append("train.py", "# warmup").commit("Add warmup")
    return r


# ================================================================= lite

@template(M, "lite")
def record_run(rng):
    run = f"outputs/run_{rng.randint(10, 99)}"
    r = _proj(rng)
    s = setup(GIT, "#: cwd ~/proj", r.code(), f"mkdir -p {run}", "echo '# tweak' >> train.py")
    items = [
        ("git rev-parse", f"Save the short hash of the current commit in `{run}/commit.txt`.",
         f"git rev-parse --short HEAD > {run}/commit.txt", {"post": f"cat {run}/commit.txt"}),
        ("git diff", f"Save your uncommitted changes as a patch in `{run}/uncommitted.patch`.",
         f"git diff > {run}/uncommitted.patch", {"post": f"cat {run}/uncommitted.patch"}),
        ("git rev-parse", f"Record the full commit hash in `{run}/commit.txt` and the uncommitted changes in "
                          f"`{run}/uncommitted.patch`.",
         f"git rev-parse HEAD > {run}/commit.txt\ngit diff > {run}/uncommitted.patch",
         {"post": f"cat {run}/commit.txt {run}/uncommitted.patch", "require": ["git rev-parse", "git diff"]}),
    ]
    return catalog(rng, items, s, answer=["<post>"])


@template(M, "lite")
def tag_experiment(rng):
    lr, bs = rng.choice(["3e-4", "1e-3", "5e-4"]), rng.choice([32, 64, 128])
    acc = rng.choice(["91.2", "93.4", "88.7"])
    r = _proj(rng)
    items = [
        ("git tag", f"Tag the commit behind your reported result as `exp/lr{lr}-bs{bs}`.", f"git tag exp/lr{lr}-bs{bs}"),
        ("git tag -a", f"Make an annotated tag `exp/lr{lr}-bs{bs}` with the message `val acc {acc}`.",
         f"git tag -a exp/lr{lr}-bs{bs} -m 'val acc {acc}'"),
    ]
    return catalog(rng, items, setup(GIT, "#: cwd ~/proj", r.code()), check=(["tags"], "."))


# ================================================================= easy

@template(M, "easy")
def conventional(rng):
    r = _proj(rng)
    kinds = [("feat", "add cosine scheduler", "a new learning-rate scheduler (a feature)"),
             ("fix", "mask padding in attention", "a bug fix in the attention masking"),
             ("docs", "explain the config options", "documentation only"),
             ("refactor", "split data loading into its own module", "a restructuring that changes no behaviour"),
             ("test", "cover the tokenizer edge cases", "new tests"),
             ("perf", "cache the tokenized dataset", "a speed-up"),
             ("chore", "bump torch to 2.4", "maintenance (a dependency bump)"),
             ("ci", "run tests on pull requests", "a change to the CI workflow")]
    t, desc, what = rng.choice(kinds)
    s = setup(GIT, "#: cwd ~/proj", r.code(), "echo '# change' >> model.py 2>/dev/null || echo x > model.py",
              "git add -A")
    return P(f"The staged change is {what}. Commit it with a Conventional Commits message whose description is "
             f"`{desc}` (type, colon, space, description).", s, f"git commit -m '{t}: {desc}'",
             check=(["log", "index"], "."), require=["git commit"])


@template(M, "easy")
def nbstripout_install(rng):
    s = setup(GIT, GH, VENV.format(pkgs="nbstripout"), "#: cwd ~/titanic-eda",
              "git clone -q git@github.com:mini-ml/titanic-eda.git .")
    items = [
        ("nbstripout --install", "Set up nbstripout as a git filter in this repository, so notebook outputs never "
                                 "get committed.", "nbstripout --install"),
        ("nbstripout --install", "Install the nbstripout filter here, then check that it's installed (`--status`).",
         "nbstripout --install\nnbstripout --status"),
    ]
    return catalog(rng, items, s, post="git config filter.nbstripout.clean; git config diff.ipynb.textconv; "
                                       "cat .git/info/attributes", answer=["<post>"])


# ================================================================= medium

@template(M, "medium")
def dvc_add(rng):
    f = rng.choice(["data/raw.csv", "data/train.parquet", "data/images.zip"])
    r = _proj(rng)
    s = setup(GIT, VENV.format(pkgs="dvc"), "#: cwd ~/proj", r.code(), "mkdir -p data",
              f"head -c 300000 /dev/urandom > {f}")
    d, base = f.rsplit("/", 1)
    return P(f"Version `{f}` with DVC instead of git: initialise DVC in this repository, add the file to DVC, and "
             f"commit what DVC asks you to track (plus its own files) as `Track {base} with DVC`.",
             s, f"dvc init\ndvc add {f}\ngit add {f}.dvc {d}/.gitignore\n"
                f"git commit -m 'Track {base} with DVC'",
             check=(["log", "files", "index"], "."), require=["dvc init", "dvc add"])


@template(M, "medium")
def reword_conventional(rng):
    r = Repo(rng).commit("Initial commit", ml_files(rng, 2))
    w1, w2 = rng.sample(WORDS, 2)
    r.write(f"{w1}.py", "a = 1").commit(f"added {w1}")
    r.write(f"{w2}.py", "b = 2").commit(f"fixed {w2} bug")
    return P(f"Reword the last two commits in Conventional Commits style: `added {w1}` -> `feat: add {w1}` and "
             f"`fixed {w2} bug` -> `fix: {w2} bug` (this branch isn't pushed). The rebase todo list opens in vim, "
             f"or set `GIT_SEQUENCE_EDITOR`.", setup(GIT, "#: cwd ~/proj", r.code()),
             f"GIT_SEQUENCE_EDITOR=\"sed -i '1,2s/^pick/edit/'\" git rebase -i HEAD~2\n"
             f"git commit --amend -m 'feat: add {w1}'\ngit rebase --continue\n"
             f"git commit --amend -m 'fix: {w2} bug'\ngit rebase --continue",
             check=(["head", "log", "files", "state"], "."), require=["git rebase -i|git commit --amend"])


# ================================================================= hard

@template(M, "hard")
def dvc_pull(rng):
    s = setup(GIT, GH, VENV.format(pkgs="dvc"), "#: cwd ~/sentiment-classifier",
              "git clone -q git@github.com:mini-ml/sentiment-classifier.git .")
    return P("This clone has `data/reviews.csv.dvc` but not the dataset itself. Download it from the project's DVC "
             "remote, then print its first three lines.", s, "dvc pull\nhead -3 data/reviews.csv",
             post="ls data; /usr/lib/bashlite/gitstate worktree untracked", answer=["<stdout>", "<post>"],
             require=["dvc pull"])


@template(M, "hard")
def dvc_remote(rng):
    url = rng.choice(["s3://my-bucket/dvc", "s3://ml-data/store", "gs://team-dvc/cache"])
    r = _proj(rng)
    s = setup(GIT, VENV.format(pkgs="dvc"), "#: cwd ~/proj", r.code(), "mkdir -p data",
              "head -c 50000 /dev/urandom > data/raw.csv", "/home/ada/.venv/bin/dvc init -q >/dev/null", "/home/ada/.venv/bin/dvc add -q data/raw.csv >/dev/null",
              "git add -A && git commit -qm 'Track data with DVC'")
    return P(f"Add a default DVC remote called `store` at `{url}`, commit the DVC config change as `Configure DVC "
             f"remote`, and push the data to the remote.", s,
             f"dvc remote add -d store {url}\ngit add .dvc/config\ngit commit -m 'Configure DVC remote'\ndvc push",
             post="cat .dvc/config; /usr/lib/bashlite/gitstate log", answer=["<post>"],
             require=["dvc remote add", "dvc push"])


# ================================================================= challenge

@template(M, "challenge")
def strip_notebooks(rng):
    s = setup(GIT, GH, VENV.format(pkgs="nbstripout"), "#: ghrepo ada/titanic-eda from=mini-ml/titanic-eda",
              "#: cwd ~/titanic-eda", "git clone -q git@github.com:ada/titanic-eda.git .")
    return P("**Notebook hygiene.** The notebooks were committed with all their outputs. Install the nbstripout git "
             "filter, strip the outputs from every notebook in `notebooks/`, commit that as `chore: strip notebook "
             "outputs`, and push.", s,
             "nbstripout --install\nnbstripout notebooks/*.ipynb\ngit commit -am 'chore: strip notebook outputs'\n"
             "git push", check=(["log", "files", "worktree", "gh=ada/titanic-eda"], "."),
             require=["nbstripout"])


@template(M, "challenge")
def experiment_release(rng):
    lr = rng.choice(["3e-4", "1e-3"])
    acc = rng.choice(["93.1", "94.0", "91.8"])
    s = setup(GIT, GH, "#: ghrepo ada/nanogpt-mini from=mini-ml/nanogpt-mini", "#: cwd ~/nanogpt-mini",
              "git clone -q git@github.com:ada/nanogpt-mini.git .", "mkdir -p outputs/run_1")
    return P(f"**A reproducible result.** Set `learning_rate = {lr}` in `config/train_shakespeare.py` (replace the "
             f"existing line), commit it as `exp: lr {lr}`, record that commit's full hash in "
             f"`outputs/run_1/commit.txt`, tag the commit `exp/lr{lr}` (annotated, message `val acc {acc}`), and push "
             f"the commit and the tag.", s,
             f"sed -i 's/^learning_rate = .*/learning_rate = {lr}/' config/train_shakespeare.py\n"
             f"git commit -am 'exp: lr {lr}'\ngit rev-parse HEAD > outputs/run_1/commit.txt\n"
             f"git tag -a exp/lr{lr} -m 'val acc {acc}'\ngit push origin main exp/lr{lr}",
             post=f"/usr/lib/bashlite/gitstate log tags gh=ada/nanogpt-mini; git log -1 --format=%s $(cat "
                  f"outputs/run_1/commit.txt)", answer=["<post>"], require=["git tag", "git push"])
