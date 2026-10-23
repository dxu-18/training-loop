"""7.15 Python environments."""

from ._common import DATASETS, MODELS, NAMES, WORDS, P, catalog, challenge, files, q, template

M = "7.15"
PKGS = ["numpy", "pandas", "requests", "tqdm", "pyyaml", "matplotlib", "scipy", "einops", "torch", "jinja2"]
FREEZE = "pip freeze"


def _pkgs(rng, k):
    return rng.sample(PKGS, k)


# ================================================================= lite

@template(M, "lite")
def venv_pip(rng):
    a, b, c = _pkgs(rng, 3)
    env = rng.choice([".venv", "venv", "env"])
    v = {"numpy": "2.2.6", "pandas": "2.2.3", "torch": "2.8.0", "scipy": "1.16.3", "pip": "25.2"}
    pinned = rng.choice(["numpy", "pandas", "torch"])
    act = f"#: venv {env}\n#: activate {env}"
    items = [
        ("python3 -m", f"Create a virtual environment in `{env}` with the `venv` module, activate it, and print which "
                       f"`python` you get.", f"python3 -m venv {env}\nsource {env}/bin/activate\nwhich python",
         {"require": ["python3 -m venv", "source|."]}),
        ("deactivate", f"`{env}` is active. Leave it with `deactivate` and print which `python3` you get now.",
         "deactivate\nwhich python3", {"setup": act}),
        ("pip install", f"Install `{a}` and `{b}` into the active environment.", f"pip install {a} {b}",
         {"setup": act, "post": FREEZE, "answer": ["<post>"]}),
        ("pip install", f"Install exactly version {v[pinned]} of `{pinned}`.", f"pip install {pinned}=={v[pinned]}",
         {"setup": act, "post": FREEZE, "answer": ["<post>"]}),
        ("python -m", "Upgrade pip itself inside the environment (`python -m pip install -U pip`), then print `pip "
                      "--version`'s first word and version.", "python -m pip install -U pip\npip --version | cut -d' ' -f1,2",
         {"setup": act, "answer": ["<post>"], "post": "pip --version | cut -d' ' -f1,2", "require": ["python -m pip|pip install -U"]}),
        ("pip install -r", "Install everything listed in `requirements.txt`.", "pip install -r requirements.txt",
         {"setup": act + "\n" + files({"requirements.txt": f"{a}\n{b}=={'2.2.6' if b == 'numpy' else ''}\n".replace("==\n", "\n")}),
          "post": FREEZE, "answer": ["<post>"]}),
        ("pip freeze", "Write the installed packages, pinned (`name==version`), to `requirements.txt` and print it.",
         "pip freeze > requirements.txt\ncat requirements.txt", {"setup": f"#: venv {env} {a} {b}\n#: activate {env}"}),
        ("pip list", "Show which installed packages have newer versions available.", "pip list --outdated",
         {"setup": f"#: venv {env} numpy==2.2.6 {c}\n#: activate {env}", "require": ["pip list", "pip --outdated"]}),
        ("pip show", f"Show the metadata (version, location, dependencies) of the installed `{a}`.", f"pip show {a}",
         {"setup": f"#: venv {env} {a}\n#: activate {env}"}),
        ("pip uninstall", f"Uninstall `{a}` without being asked for confirmation.", f"pip uninstall -y {a}",
         {"setup": f"#: venv {env} {a} {b}\n#: activate {env}", "post": FREEZE, "answer": ["<post>"],
          "require": ["pip uninstall"]}),
    ]
    return catalog(rng, items)


@template(M, "lite")
def check_install(rng):
    env = rng.choice([".venv", "venv"])
    kind = rng.choice(["torch-cuda", "versions", "pep668", "which"])
    if kind == "torch-cuda":
        g = rng.choice([0, 1, 2, 4])
        return P("Print torch's version and whether CUDA is available, from a one-line Python command: "
                 "`python -c \"import torch; print(torch.__version__, torch.cuda.is_available())\"`.",
                 (f"#: gpus {g}\n" if g else "") + f"#: venv {env} torch\n#: activate {env}",
                 'python -c "import torch; print(torch.__version__, torch.cuda.is_available())"', require=["python -c"])
    if kind == "versions":
        a, b = _pkgs(rng, 2)
        mods = {"pyyaml": "yaml"}
        return P(f"Print the versions of `{a}` and `{b}` (their `__version__`), one per line, with `python -c`.",
                 f"#: venv {env} {a} {b}\n#: activate {env}",
                 f'python -c "import {mods.get(a, a)}, {mods.get(b, b)}; print({mods.get(a, a)}.__version__); print({mods.get(b, b)}.__version__)"',
                 require=["python -c"])
    if kind == "pep668":
        a = rng.choice(PKGS)
        return P(f"Try to `pip install {a}` with the SYSTEM Python (no environment active) and see why Ubuntu refuses; "
                 f"then print the exit status.", "", f"pip install {a}\necho $?", answer=["<stdout>", "<stderr>"],
                 require=["pip install"])
    return P(f"Activate `{env}`, then show that `python` and `pip` now both come from it (`which python pip`).",
             f"#: venv {env}", f"source {env}/bin/activate\nwhich python pip", require=["source|.", "which"])


# ================================================================= easy

@template(M, "easy")
def uv_workflows(rng):
    a, b = _pkgs(rng, 2)
    proj = rng.choice(["demo", "mlapp", "trainer", "nanogpt"])
    msg = rng.choice(WORDS)
    setup = "#: uv\nexport PATH=\"$HOME/.local/bin:$PATH\""
    items = [
        ("uv venv", f"Create a virtual environment with `uv` (it goes into `.venv`), then install `{a}` into it with "
                    f"`uv pip install` from `requirements.txt`.", "uv venv\nuv pip install -r requirements.txt",
         {"setup": setup + "\n" + files({"requirements.txt": f"{a}\n"}), "post": ".venv/bin/pip freeze", "answer": ["<post>"],
          "require": ["uv venv", "uv pip"]}),
        ("uv init", f"Start a new uv project called `{proj}` in a new directory (`uv init {proj}`), then print its "
                    f"`pyproject.toml`.", f"uv init {proj}\ncat {proj}/pyproject.toml", {"setup": setup}),
        ("uv add", f"In this uv project, add `{a}` and `{b}` as dependencies (`uv add`), then print the dependency lines "
                   f"of `pyproject.toml`.", f"uv add {a} {b}\ngrep -A3 dependencies pyproject.toml",
         {"setup": setup + f"\nuv init -q\n#: cwd ~/{proj}", "post": "grep -A3 dependencies pyproject.toml",
          "answer": ["<post>"]}),
        ("uv sync", "Install the project's locked dependencies into `.venv` with `uv sync`, then list them with "
                    "`uv pip list`.", "uv sync\nuv pip list",
         {"setup": setup + f"\nuv init -q && uv add -q {a} && rm -rf .venv\n#: cwd ~/{proj}",
          "post": ".venv/bin/pip freeze 2>/dev/null || uv pip freeze", "answer": ["<post>"]}),
        ("uv run", f"Run `main.py` in the project's environment with `uv run`.", "uv run main.py",
         {"setup": setup + f"\nuv init -q\n#: cwd ~/{proj}\n" + files({f"main.py": f"print('{msg} from uv')\n"})}),
        ("uv lock", f"Update the project's lockfile with `uv lock`, then print the package names in `uv.lock`.",
         "uv lock\ngrep '^name' uv.lock", {"setup": setup + f"\nuv init -q\n#: cwd ~/{proj}\n"
                                                           f"sed -i 's/dependencies = \\[\\]/dependencies = [\"{a}\"]/' pyproject.toml",
                                           "post": "grep '^name' uv.lock", "answer": ["<post>"]}),
        ("uv python", "Install Python 3.12 with uv (`uv python install`).", "uv python install 3.12",
         {"setup": setup, "post": "ls ~/.local/share/uv/python", "answer": ["<post>"]}),
    ]
    return catalog(rng, items)


@template(M, "easy")
def conda_envs(rng):
    name = rng.choice(["ml", "torch", "research", "nlp", "cv"])
    py = rng.choice(["3.11", "3.12"])
    a, b = rng.sample(["numpy", "pandas", "scipy", "matplotlib"], 2)
    base = "#: conda"
    have = f"#: conda-env {name} python={py} {a}"
    items = [
        ("conda create", f"Create a conda environment `{name}` with Python {py}, answering yes automatically (`-y`).",
         f"conda create -n {name} python={py} -y", {"setup": base, "post": "conda env list", "answer": ["<post>"]}),
        ("conda activate", f"Activate the conda environment `{name}`, then print which `python` is used.",
         f"conda activate {name}\nwhich python", {"setup": have}),
        ("conda install", f"Activate `{name}` and install `{b}` into it from the `conda-forge` channel (`-y`).",
         f"conda activate {name}\nconda install -c conda-forge {b} -y",
         {"setup": have, "post": f"conda list -n {name} | grep -E '^({a}|{b}) ' | cut -d' ' -f1", "answer": ["<post>"],
          "require": ["conda install", "conda -c"]}),
        ("conda env", f"Export the environment `{name}` to `environment.yml`, then print the file.",
         f"conda env export -n {name} > environment.yml\ncat environment.yml", {"setup": have}),
        ("conda env", f"Recreate the environment described in `environment.yml` (`conda env create -f`), then list the "
                      f"environments.", "conda env create -f environment.yml\nconda env list",
         {"setup": have + f"\nconda env export -n {name} > environment.yml && conda remove -n {name} --all -y > /dev/null",
          "post": "conda env list", "answer": ["<post>"], "require": ["conda env", "conda -f"]}),
        ("conda env", "List your conda environments.", "conda env list", {"setup": have}),
        ("conda remove", f"Delete the environment `{name}` completely (`--all`, `-y`), then list environments.",
         f"conda remove -n {name} --all -y\nconda env list", {"setup": have, "post": "conda env list", "answer": ["<post>"]}),
    ]
    return catalog(rng, items)


# ================================================================= medium

@template(M, "medium")
def reproduce(rng):
    pk = _pkgs(rng, 3)
    env = rng.choice([".venv", "venv"])
    kind = rng.choice(["freeze-restore", "fresh-machine"])
    if kind == "freeze-restore":
        return P(f"Create a venv `{env}`, activate it, install `{' '.join(pk)}`, and freeze the exact versions into "
                 f"`requirements.txt`.", "",
                 f"python3 -m venv {env}\nsource {env}/bin/activate\npip install {' '.join(pk)}\npip freeze > requirements.txt",
                 post="cat requirements.txt", answer=["<post>"], require=["python3 -m venv", "pip install", "pip freeze"])
    reqs = f"#: venv old {' '.join(pk)}\nold/bin/pip freeze > requirements.txt\nrm -rf old"
    return P(f"`requirements.txt` came from another machine. Recreate that environment in a new venv `{env}` here "
             f"(create, activate, install from the file).", reqs,
             f"python3 -m venv {env}\nsource {env}/bin/activate\npip install -r requirements.txt",
             post="pip freeze | diff - requirements.txt && echo identical", answer=["<post>"], require=["pip install -r"])


@template(M, "medium")
def editable_and_kernel(rng):
    pkg = rng.choice(["mylib", "toolkit", "nanotrain", "datakit"])
    fn = rng.choice(WORDS)
    env = ".venv"
    proj = files({f"{pkg}/pyproject.toml": f'[project]\nname = "{pkg}"\nversion = "0.{rng.randint(1, 9)}.0"\n'
                                           f'dependencies = ["tqdm"]\n',
                  f"{pkg}/src/{pkg}/__init__.py": f"def {fn}():\n    return '{fn} works'\n"})
    kind = rng.choice(["editable", "kernel", "jupyter"])
    if kind == "editable":
        return P(f"Install the local project in `{pkg}/` in editable mode (`pip install -e`), then call "
                 f"`{pkg}.{fn}()` from Python and print the result.", f"#: venv {env}\n#: activate {env}\n" + proj,
                 f"pip install -e {pkg}\npython -c \"import {pkg}; print({pkg}.{fn}())\"",
                 post=f'python -c "import {pkg}; print({pkg}.{fn}())"; pip freeze | grep -c "^-e"', answer=["<post>"],
                 require=["pip install -e"])
    if kind == "kernel":
        name = rng.choice(["ml", "torch-env", "research"])
        return P(f"Register the active venv as a Jupyter kernel named `{name}` for your user "
                 f"(`python -m ipykernel install`), then list the kernels.",
                 f"#: venv {env} ipykernel jupyterlab\n#: activate {env}",
                 f"python -m ipykernel install --user --name {name}\njupyter kernelspec list",
                 require=["python -m ipykernel|python3 -m ipykernel"])
    port = rng.choice([8888, 8889, 9999])
    return P(f"Start JupyterLab without opening a browser on port {port} (it runs until Ctrl-C, after 10 s here); then "
             f"check that nothing listens on port {port} anymore (`lsof -i :{port} || echo free`).",
             f"#: venv {env} jupyterlab\n#: activate {env}", f"jupyter lab --no-browser --port {port}\nlsof -i :{port} || echo free",
             answer=["<stdout>"], require=["jupyter"])


# ================================================================= hard

@template(M, "hard")
def two_envs(rng):
    a = rng.choice(["numpy", "pandas", "torch"])
    vers = {"numpy": ("2.2.6", "2.4.6"), "pandas": ("2.2.3", "3.0.2"), "torch": ("2.4.1", "2.11.0")}[a]
    o, n = rng.choice([("old", "new"), ("env-a", "env-b"), ("stable", "latest"), ("v1", "v2")])
    return P(f"Make two venvs, `{o}` with `{a}=={vers[0]}` and `{n}` with the latest `{a}`, then print `{o}: V` and "
             f"`{n}: V` with each environment's own Python (`{o}/bin/python -c ...`), without activating either.", "",
             f"python3 -m venv {o}\npython3 -m venv {n}\n{o}/bin/pip install -q {a}=={vers[0]}\n{n}/bin/pip install -q {a}\n"
             f'echo "{o}: $({o}/bin/python -c "import {a}; print({a}.__version__)")"\n'
             f'echo "{n}: $({n}/bin/python -c "import {a}; print({a}.__version__)")"', answer=["<stdout>"],
             require=["python3 -m venv"])


@template(M, "hard")
def which_env(rng):
    envs = rng.sample(["ml", "nlp", "cv", "audio", "rl"], 2)
    pk = {e: rng.sample(["numpy", "pandas", "scipy", "matplotlib", "torch"], 2) for e in envs}
    setup = "\n".join(f"#: conda-env {e} python=3.12 {' '.join(pk[e])}" for e in envs)
    want = rng.choice(pk[envs[1]])
    return P(f"Which of your conda environments have `{want}` installed? Print their names, one per line "
             f"(check `conda list -n ENV`).", setup,
             f"for e in $(conda env list | grep -v '^#' | awk 'NF {{print $1}}'); do\n"
             f"  conda list -n \"$e\" | grep -q '^{want} ' && echo \"$e\"\ndone", require=["conda"])


# ================================================================= challenge

def _reqs(rng, n):
    lines = []
    for p in rng.sample(PKGS + ["requests", "wandb", "pydantic", "fastapi"], n):
        op = rng.choice(["", "==", ">=", "<"])
        ver = {"numpy": ["2.2.6", "2.0", "3.0"], "pandas": ["2.2.3", "3.0"], "torch": ["2.4.1", "2.8.0", "3.0"]}.get(p, ["1.0", "99.0"])
        lines.append(p + (op + rng.choice(ver) if op else ""))
    return files({"requirements.txt": "\n".join(lines) + "\n"}) + "\n#: venv .venv\n#: activate .venv"


@template(M, "challenge")
def resolve_report(rng):
    setup = _reqs(rng, 4)
    hidden = [_reqs(rng, rng.randint(8, 12)) for _ in range(2)]
    sol = ('while read -r req; do\n  if pip install -q "$req" > /dev/null 2>&1; then\n'
           '    name=$(sed -E "s/[<>=].*//" <<< "$req")\n    echo "ok $req -> $(pip show "$name" | grep ^Version | cut -d" " -f2)"\n'
           '  else\n    echo "FAILED $req"\n  fi\ndone < requirements.txt')
    return challenge("Which requirements resolve?", "Install the lines of `requirements.txt` one at a time into the "
                     "active venv (hide pip's output). For each, print `ok REQ -> VERSION` with the version that got "
                     "installed (`pip show`), or `FAILED REQ` when pip can't satisfy it.", setup, sol, hidden,
                     "up to 12 requirements", require=["pip install", "while|for"])


def _sweep_envs(rng, n):
    lines = []
    for i in range(n):
        p = rng.sample(["numpy", "pandas", "scipy", "einops", "tqdm", "pyyaml"], rng.randint(1, 3))
        lines.append(f"#: venv envs/e{i} {' '.join(p)}")
    return "\n".join(lines)


@template(M, "challenge")
def package_matrix(rng):
    setup = _sweep_envs(rng, 2)
    hidden = [_sweep_envs(rng, rng.randint(4, 6)) for _ in range(2)]
    sol = ('for e in envs/*/; do\n  e=${e%/}\n  "$e/bin/pip" freeze | sed "s|^|${e##*/} |"\ndone | '
           "awk '{split($2, a, \"==\"); n[a[1]]++; envs[a[1]] = envs[a[1]] \" \" $1} END {for (p in n) print p, n[p] \":\" envs[p]}' | sort")
    return challenge("Package matrix", "`envs/` holds several venvs. For every package installed in at least one of "
                     "them (as `pip freeze` lists it), print `PACKAGE COUNT: ENV ENV ...` — how many envs have it and "
                     "which (env directory names, in alphabetical order) — sorted by package name.", setup, sol, hidden,
                     "up to 6 environments")
