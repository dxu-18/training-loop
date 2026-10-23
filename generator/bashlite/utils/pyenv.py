"""python / pip / venv / uv / conda / jupyter / tensorboard / torchrun.

Python code really runs (CPython or Pyodide), with open()/os/pathlib redirected to the simulated filesystem, sys.argv,
os.environ, stdin and the clock from the simulation; third-party imports follow what is "installed" in the active env.
"""

from __future__ import annotations

import builtins
import io
import os
import re
import sys
import traceback

from ..exec import stub_bytes, util
from ..kernel import NS
from ..timeutil import local_dt, strftime
from ..vfs import FSError

PY_VERSION = "3.12.3"
SITE = "lib/python3.12/site-packages"

# name -> (latest version, older versions, dependencies, summary)
INDEX = {
    "numpy": ("2.4.6", ["2.0.2", "2.2.6", "2.3.5"], [], "Fundamental package for array computing in Python"),
    "pandas": ("3.0.2", ["2.2.3"], ["numpy", "python-dateutil", "tzdata"], "Powerful data structures for data analysis, time series, and statistics"),
    "python-dateutil": ("2.9.0.post0", [], ["six"], "Extensions to the standard Python datetime module"),
    "six": ("1.17.0", [], [], "Python 2 and 3 compatibility utilities"),
    "tzdata": ("2025.2", [], [], "Provider of IANA time zone data"),
    "torch": ("2.11.0", ["2.4.1", "2.8.0"], ["filelock", "typing-extensions", "sympy", "networkx", "jinja2", "fsspec"],
              "Tensors and Dynamic neural networks in Python with strong GPU acceleration"),
    "filelock": ("3.20.0", [], [], "A platform independent file lock."),
    "typing-extensions": ("4.15.0", [], [], "Backported and Experimental Type Hints for Python 3.9+"),
    "sympy": ("1.14.0", [], ["mpmath"], "Computer algebra system (CAS) in Python"),
    "mpmath": ("1.3.0", [], [], "Python library for arbitrary-precision floating-point arithmetic"),
    "networkx": ("3.5", [], [], "Python package for creating and manipulating graphs and networks"),
    "jinja2": ("3.1.6", [], ["markupsafe"], "A very fast and expressive template engine."),
    "markupsafe": ("3.0.3", [], [], "Safely add untrusted strings to HTML/XML markup."),
    "fsspec": ("2025.9.0", [], [], "File-system specification"),
    "scikit-learn": ("1.8.0", ["1.5.2"], ["numpy", "scipy", "joblib", "threadpoolctl"], "A set of python modules for machine learning and data mining"),
    "scipy": ("1.16.3", [], ["numpy"], "Fundamental algorithms for scientific computing in Python"),
    "joblib": ("1.5.2", [], [], "Lightweight pipelining with Python functions"),
    "threadpoolctl": ("3.6.0", [], [], "threadpoolctl"),
    "matplotlib": ("3.10.8", [], ["numpy", "pillow", "pyparsing", "cycler", "kiwisolver", "packaging", "python-dateutil"],
                   "Python plotting package"),
    "pillow": ("12.0.0", [], [], "Python Imaging Library (fork)"),
    "pyparsing": ("3.2.5", [], [], "pyparsing - Classes and methods to define and execute parsing grammars"),
    "cycler": ("0.12.1", [], [], "Composable style cycles"),
    "kiwisolver": ("1.4.9", [], [], "A fast implementation of the Cassowary constraint solver"),
    "packaging": ("25.0", [], [], "Core utilities for Python packages"),
    "requests": ("2.32.5", [], ["charset-normalizer", "idna", "urllib3", "certifi"], "Python HTTP for Humans."),
    "charset-normalizer": ("3.4.4", [], [], "The Real First Universal Charset Detector."),
    "idna": ("3.11", [], [], "Internationalized Domain Names in Applications (IDNA)"),
    "urllib3": ("2.5.0", [], [], "HTTP library with thread-safe connection pooling, file post, and more."),
    "certifi": ("2025.10.5", [], [], "Python package for providing Mozilla's CA Bundle."),
    "tqdm": ("4.67.1", [], [], "Fast, Extensible Progress Meter"),
    "pyyaml": ("6.0.3", [], [], "YAML parser and emitter for Python"),
    "transformers": ("4.57.1", [], ["huggingface-hub", "numpy", "packaging", "pyyaml", "regex", "requests", "safetensors", "tokenizers", "tqdm", "filelock"],
                     "State-of-the-art Machine Learning for JAX, PyTorch and TensorFlow"),
    "huggingface-hub": ("0.35.3", [], ["filelock", "fsspec", "packaging", "pyyaml", "requests", "tqdm", "typing-extensions"],
                        "Client library to download and publish models, datasets and other repos on the huggingface.co hub"),
    "regex": ("2025.9.18", [], [], "Alternative regular expression module, to replace re."),
    "safetensors": ("0.6.2", [], [], ""),
    "tokenizers": ("0.22.1", [], ["huggingface-hub"], ""),
    "datasets": ("4.2.0", [], ["numpy", "pandas", "requests", "tqdm", "fsspec", "huggingface-hub", "pyyaml"], "HuggingFace community-driven open-source library of datasets"),
    "accelerate": ("1.10.1", [], ["numpy", "packaging", "pyyaml", "torch", "huggingface-hub", "safetensors"], "Accelerate"),
    "wandb": ("0.22.2", [], ["requests", "pyyaml", "packaging"], "A CLI and library for interacting with the Weights & Biases API."),
    "tensorboard": ("2.20.0", [], ["numpy", "packaging"], "TensorBoard lets you watch Tensors Flow"),
    "jupyterlab": ("4.4.9", [], ["jinja2", "packaging", "tornado", "ipykernel"], "JupyterLab computational environment"),
    "ipykernel": ("6.30.1", [], ["packaging", "tornado"], "IPython Kernel for Jupyter"),
    "tornado": ("6.5.2", [], [], "Tornado is a Python web framework and asynchronous networking library."),
    "einops": ("0.8.1", [], [], "A new flavour of deep learning operations"),
    "pip": ("25.2", ["24.0"], [], "The PyPA recommended tool for installing Python packages."),
    "uv": ("0.9.4", [], [], "An extremely fast Python package and project manager, written in Rust."),
    "setuptools": ("80.9.0", [], [], "Easily download, build, install, upgrade, and uninstall Python packages"),
    "pytest": ("8.4.2", [], ["packaging"], "pytest: simple powerful testing with Python"),
    "black": ("25.9.0", [], ["packaging"], "The uncompromising code formatter."),
    "ruff": ("0.14.1", [], [], "An extremely fast Python linter and code formatter, written in Rust."),
    "fastapi": ("0.119.0", [], ["pydantic"], "FastAPI framework, high performance, easy to learn, fast to code, ready for production"),
    "pydantic": ("2.12.2", [], ["typing-extensions"], "Data validation using Python type hints"),
    "uvicorn": ("0.37.0", [], [], "The lightning-fast ASGI server."),
}


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


# how a project spells its own name in its metadata
INDEX_NAME = {"pyyaml": "PyYAML", "markupsafe": "MarkupSafe", "jinja2": "Jinja2", "pillow": "pillow"}


def env_of(ctx, path=None):
    """The Python environment a python/pip binary belongs to: (root, kind) where kind is venv | system | conda."""
    p = path or getattr(ctx, "path", None) or ""
    if p and "/bin/" in p and not p.startswith(("/usr/bin", "/bin/")):
        root = p.rsplit("/bin/", 1)[0]
        if not root.startswith("/"):
            root = ctx.abspath(root)
        if ctx.fs.exists(root + "/pyvenv.cfg"):
            return root, "venv"
        if ctx.fs.exists(root + "/conda-meta"):
            return root, "conda"
    venv = ctx.env.get("VIRTUAL_ENV")
    if venv and ctx.fs.exists(venv + "/pyvenv.cfg") and not p:
        return venv, "venv"
    return "/usr", "system"


def site_dir(root, kind):
    return "/usr/lib/python3/dist-packages" if kind == "system" else f"{root}/{SITE}"


def installed(ctx, root, kind):
    """{normalized name: (display name, version)} from *.dist-info directories."""
    out = {}
    d = site_dir(root, kind)
    try:
        node = ctx.fs.lookup(d)
    except FSError:
        return out
    for name in ctx.fs.listdir(node):
        m = re.match(r"(.+)-([^-]+)\.dist-info$", name)
        if m:
            out[norm(m.group(1))] = (m.group(1).replace("_", "-"), m.group(2))
    return out


def set_installed(ctx, root, kind, name, version, editable=None):
    d = site_dir(root, kind)
    for n, (disp, ver) in installed(ctx, root, kind).items():
        if n == norm(name):
            remove_installed(ctx, root, kind, name)
    info = INDEX.get(norm(name), (version, [], [], ""))
    meta = (f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\nSummary: {info[3]}\n"
            + "".join(f"Requires-Dist: {r}\n" for r in info[2]))
    base = f"{d}/{name.replace('-', '_')}-{version}.dist-info"
    ctx.fs.write_file(base + "/METADATA", meta, "/", 0o644, ctx.uid, ctx.proc.gid)
    if editable:
        ctx.fs.write_file(base + "/direct_url.json", f'{{"url": "file://{editable}", "dir_info": {{"editable": true}}}}\n',
                          "/", 0o644, ctx.uid, ctx.proc.gid)
    else:
        pkgdir = f"{d}/{name.replace('-', '_')}"
        ctx.fs.write_file(pkgdir + "/__init__.py", f"__version__ = {version!r}\n", "/", 0o644, ctx.uid, ctx.proc.gid)
    write_console_scripts(ctx, root, kind, name)


# console scripts a package installs into the environment's bin/: script name -> (entry module, simulated program)
CONSOLE = {"pip": [("pip", "pip._internal.cli.main"), ("pip3", "pip._internal.cli.main"), ("pip3.12", "pip._internal.cli.main")],
           "jupyterlab": [("jupyter", "jupyter_core.command"), ("jupyter-lab", "jupyterlab.labapp")],
           "torch": [("torchrun", "torch.distributed.run")], "tensorboard": [("tensorboard", "tensorboard.main")],
           "uv": [("uv", "uv")]}
ENTRY = {"pip._internal.cli.main": "pip", "jupyter_core.command": "jupyter", "jupyterlab.labapp": "jupyter-lab",
         "torch.distributed.run": "torchrun", "tensorboard.main": "tensorboard", "uv": "uv"}


def console_script(python, module):
    return (f"#!{python}\n# -*- coding: utf-8 -*-\nimport re\nimport sys\nfrom {module} import main\n"
            f"if __name__ == '__main__':\n    sys.argv[0] = re.sub(r'(-script\\.pyw|\\.exe)?$', '', sys.argv[0])\n"
            f"    sys.exit(main())\n")


def write_console_scripts(ctx, root, kind, name):
    bindir = "/usr/local/bin" if kind == "system" else f"{root}/bin"
    python = "/usr/bin/python3" if kind == "system" else f"{root}/bin/python{'3' if kind == 'venv' else ''}"
    for script, module in CONSOLE.get(norm(name), []):
        node = ctx.fs.write_file(f"{bindir}/{script}", console_script(python, module), "/", 0o755, ctx.uid, ctx.proc.gid)
        node.mode = 0o755


def remove_installed(ctx, root, kind, name):
    d = site_dir(root, kind)
    try:
        node = ctx.fs.lookup(d)
    except FSError:
        return False
    removed = False
    for entry in list(node.entries):
        m = re.match(r"(.+)-([^-]+)\.dist-info$", entry)
        if (m and norm(m.group(1)) == norm(name)) or entry == name.replace("-", "_"):
            _rm_tree(ctx.fs, f"{d}/{entry}")
            removed = True
    if removed:
        bindir = "/usr/local/bin" if kind == "system" else f"{root}/bin"
        for script, _ in CONSOLE.get(norm(name), []):
            if ctx.fs.exists(f"{bindir}/{script}", follow=False):
                ctx.fs.unlink(f"{bindir}/{script}")
    return removed


def _rm_tree(fs, path):
    try:
        node = fs.lookup(path, follow=False)
    except FSError:
        return
    if node.type == "d":
        for child in list(node.entries):
            _rm_tree(fs, path + "/" + child)
        fs.rmdir(path)
    else:
        fs.unlink(path)


def make_venv(ctx, path, prompt=None, quiet=True, kind="venv"):
    fs = ctx.fs
    full = ctx.abspath(path)
    for d in ("bin", "include/python3.12", SITE):
        node = fs.mkdir(f"{full}/{d}", "/", 0o755, ctx.uid, ctx.proc.gid, parents=True)
    fs.symlink("lib", f"{full}/lib64") if not fs.exists(f"{full}/lib64", follow=False) else None
    name = prompt or full.rstrip("/").rsplit("/", 1)[-1]
    fs.write_file(f"{full}/pyvenv.cfg", f"home = /usr/bin\ninclude-system-site-packages = false\nversion = {PY_VERSION}\n"
                                        f"executable = /usr/bin/python3.12\ncommand = /usr/bin/python3 -m venv {full}\n"
                                        + (f"prompt = {prompt}\n" if prompt else ""), "/", 0o644, ctx.uid, ctx.proc.gid)
    if not fs.exists(f"{full}/bin/python3", follow=False):
        fs.symlink("/usr/bin/python3", f"{full}/bin/python3")
        fs.symlink("python3", f"{full}/bin/python")
        fs.symlink("python3", f"{full}/bin/python3.12")
    act = ACTIVATE.replace("@@VENV@@", full).replace("@@PROMPT@@", name)
    fs.write_file(f"{full}/bin/activate", act, "/", 0o644, ctx.uid, ctx.proc.gid)
    fs.write_file(f"{full}/bin/activate.csh", f"# This file must be used with \"source bin/activate.csh\" *from csh*.\n"
                  f"setenv VIRTUAL_ENV \"{full}\"\n", "/", 0o644, ctx.uid, ctx.proc.gid)
    fs.write_file(f"{full}/bin/activate.fish", f"# This file must be used with \"source <venv>/bin/activate.fish\" *from fish*\n"
                  f"set -gx VIRTUAL_ENV \"{full}\"\n", "/", 0o644, ctx.uid, ctx.proc.gid)
    fs.write_file(f"{full}/bin/Activate.ps1", "<#\n.Synopsis\nActivate a Python virtual environment for the current PowerShell session.\n#>\n",
                  "/", 0o644, ctx.uid, ctx.proc.gid)
    set_installed(ctx, full, "venv", "pip", "24.0")


ACTIVATE = r'''# This file must be used with "source bin/activate" *from bash*
# You cannot run it directly

deactivate () {
    # reset old environment variables
    if [ -n "${_OLD_VIRTUAL_PATH:-}" ] ; then
        PATH="${_OLD_VIRTUAL_PATH:-}"
        export PATH
        unset _OLD_VIRTUAL_PATH
    fi
    if [ -n "${_OLD_VIRTUAL_PYTHONHOME:-}" ] ; then
        PYTHONHOME="${_OLD_VIRTUAL_PYTHONHOME:-}"
        export PYTHONHOME
        unset _OLD_VIRTUAL_PYTHONHOME
    fi

    # Call hash to forget past commands. Without forgetting
    # past commands the $PATH changes we made may not be respected
    hash -r 2> /dev/null

    if [ -n "${_OLD_VIRTUAL_PS1:-}" ] ; then
        PS1="${_OLD_VIRTUAL_PS1:-}"
        export PS1
        unset _OLD_VIRTUAL_PS1
    fi

    unset VIRTUAL_ENV
    unset VIRTUAL_ENV_PROMPT
    if [ ! "${1:-}" = "nondestructive" ] ; then
    # Self destruct!
        unset -f deactivate
    fi
}

# unset irrelevant variables
deactivate nondestructive

# on Windows, a path can contain colons and backslashes and has to be converted:
if [ "${OSTYPE:-}" = "cygwin" ] || [ "${OSTYPE:-}" = "msys" ] ; then
    # transform D:\path\to\venv to /d/path/to/venv on MSYS
    # and to /cygdrive/d/path/to/venv on Cygwin
    export VIRTUAL_ENV=$(cygpath "@@VENV@@")
else
    # use the path as-is
    export VIRTUAL_ENV="@@VENV@@"
fi

_OLD_VIRTUAL_PATH="$PATH"
PATH="$VIRTUAL_ENV/bin:$PATH"
export PATH

# unset PYTHONHOME if set
# this will fail if PYTHONHOME is set to the empty string (which is bad anyway)
# could use `if (set -u; : $PYTHONHOME) ;` in bash
if [ -n "${PYTHONHOME:-}" ] ; then
    _OLD_VIRTUAL_PYTHONHOME="${PYTHONHOME:-}"
    unset PYTHONHOME
fi

if [ -z "${VIRTUAL_ENV_DISABLE_PROMPT:-}" ] ; then
    _OLD_VIRTUAL_PS1="${PS1:-}"
    PS1="(@@PROMPT@@) ${PS1:-}"
    export PS1
    VIRTUAL_ENV_PROMPT="(@@PROMPT@@) "
    export VIRTUAL_ENV_PROMPT
fi

# Call hash to forget past commands. Without forgetting
# past commands the $PATH changes we made may not be respected
hash -r 2> /dev/null
'''


# ---------------------------------------------------------------------------- requirement resolution

SPEC = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)(\[[^\]]*\])?\s*((?:[<>=!~]=?|===)\s*[^;,\s]+(?:\s*,\s*(?:[<>=!~]=?)\s*[^;,\s]+)*)?")


def vkey(v):
    return [int(x) if x.isdigit() else x for x in re.split(r"[.+-]", v)]


def pick_version(name, spec):
    info = INDEX.get(norm(name))
    if info is None:
        return None
    versions = sorted([info[0]] + info[1], key=vkey)
    if not spec:
        return info[0]
    ok = []
    for v in versions:
        good = True
        for clause in spec.split(","):
            m = re.match(r"\s*(===|==|>=|<=|~=|!=|<|>)\s*(\S+)", clause)
            if not m:
                continue
            op, target = m.group(1), m.group(2)
            a, b = vkey(v), vkey(target.rstrip(".*"))
            if op in ("==", "===") and not (v == target or (target.endswith(".*") and v.startswith(target[:-1]))):
                good = False
            elif op == "!=" and v == target:
                good = False
            elif op == ">=" and a < b:
                good = False
            elif op == "<=" and a > b:
                good = False
            elif op == ">" and a <= b:
                good = False
            elif op == "<" and a >= b:
                good = False
            elif op == "~=" and (a < b or a[:len(b) - 1] != b[:len(b) - 1]):
                good = False
        if good:
            ok.append(v)
    if not ok:
        if "==" in spec and info:
            m = re.search(r"==\s*(\S+)", spec)
            return ("ERR", m.group(1), versions)
        return ("ERR", spec, versions)
    return ok[-1]


def parse_requirements(ctx, path, seen=None):
    seen = seen or set()
    reqs = []
    try:
        text = bytes(ctx.lookup(path).data).decode()
    except FSError as e:
        raise PipError(f"ERROR: Could not open requirements file: [Errno 2] No such file or directory: '{path}'") from None
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith(("-r ", "--requirement ")):
            reqs += parse_requirements(ctx, line.split(None, 1)[1], seen)
            continue
        if line.startswith(("-e ", "--editable ")):
            reqs.append(("-e", line.split(None, 1)[1]))
            continue
        if line.startswith("-"):
            continue
        reqs.append(line)
    return reqs


class PipError(Exception):
    pass


def pyproject_info(ctx, d):
    for fname in ("pyproject.toml", "setup.cfg", "setup.py"):
        try:
            text = bytes(ctx.lookup(f"{d}/{fname}").data).decode()
        except FSError:
            continue
        name = re.search(r'^\s*name\s*=\s*["\']([^"\']+)', text, re.M)
        ver = re.search(r'^\s*version\s*=\s*["\']([^"\']+)', text, re.M)
        deps = []
        m = re.search(r"dependencies\s*=\s*\[(.*?)\]", text, re.S)
        if m:
            deps = re.findall(r'["\']([^"\']+)["\']', m.group(1))
        return (name.group(1) if name else d.rstrip("/").rsplit("/", 1)[-1]), (ver.group(1) if ver else "0.1.0"), deps
    return None


WHEEL_MB = {"numpy": 16.6, "pandas": 12.4, "torch": 888.1, "scipy": 35.7, "matplotlib": 8.7, "scikit-learn": 9.5,
            "transformers": 12.0, "pillow": 7.0, "sympy": 6.3, "networkx": 2.0, "tokenizers": 3.3, "pip": 1.8,
            "setuptools": 1.2, "jupyterlab": 12.3, "tensorboard": 5.5, "datasets": 0.5, "wandb": 22.8, "regex": 0.8,
            "pydantic": 0.5, "ruff": 13.1, "black": 1.8, "kiwisolver": 1.5, "tornado": 0.4, "pyyaml": 0.8, "mpmath": 0.5}


def wheel_size(key):
    if key in WHEEL_MB:
        return f"{WHEEL_MB[key]:.1f} MB"
    return f"{20 + sum(map(ord, key)) * 7 % 480} kB"


def resolve(ctx, specs, have, upgrade=False):
    """pip's resolver, simplified: -> list of dicts (name as written, key, version, status, parent, depth) in the order
    pip reports them. status: satisfied | install | upgrade. Raises PipError."""
    items, seen = [], {}
    queue = []
    for s in specs:
        m = SPEC.match(s)
        if not m or m.end() < len(s.split(";")[0].strip()):
            raise PipError(f"ERROR: Invalid requirement: '{s}'")
        queue.append((m.group(1), (m.group(3) or "").replace(" ", ""), None, 0))
    while queue:
        name, spec, parent, depth = queue.pop(0)
        key = norm(name)
        if key in seen:
            it = seen[key]
            it["depth"] = max(it["depth"], depth)
            continue
        info = INDEX.get(key)
        shown = name + spec
        if info is None:
            raise PipError(f"ERROR: Could not find a version that satisfies the requirement {shown} (from versions: none)\n"
                           f"ERROR: No matching distribution found for {shown}")
        cur = have.get(key)
        if cur is not None and (not upgrade or parent is not None) and _satisfies(cur[1], spec):
            it = dict(name=name, key=key, version=cur[1], status="satisfied", parent=parent, depth=depth, spec=spec)
        else:
            v = pick_version(name, spec)
            if isinstance(v, tuple):
                raise PipError(f"ERROR: Could not find a version that satisfies the requirement {shown} "
                               f"(from versions: {', '.join(v[2])})\nERROR: No matching distribution found for {shown}")
            if cur is not None and cur[1] == v:
                it = dict(name=name, key=key, version=v, status="satisfied", parent=parent, depth=depth, spec=spec)
            else:
                it = dict(name=name, key=key, version=v, status="install" if cur is None else "upgrade", parent=parent,
                          depth=depth, spec=spec)
        seen[key] = it
        items.append(it)
        for dep in info[2]:
            queue.append((dep, "", name, depth + 1))
    return items


def install_order(items):
    todo = [it for it in items if it["status"] != "satisfied"]
    return sorted(todo, key=lambda it: (it["depth"], it["key"]), reverse=True)


def pip_install(ctx, rest, root, kind, have):
    upgrade = any(x in ("-U", "--upgrade") for x in rest)
    quiet = any(x in ("-q", "--quiet", "-qq") for x in rest)
    if kind == "system" and "--break-system-packages" not in rest and "--user" not in rest or \
            kind == "system" and "--user" in rest and "--break-system-packages" not in rest:
        ctx.e(PEP668)
        return 1
    specs, editables = [], []
    i = 0
    while i < len(rest):
        x = rest[i]
        if x in ("-r", "--requirement") and i + 1 < len(rest) or x.startswith("-r") and len(x) > 2 and not x.startswith("--"):
            path = rest[i + 1] if x in ("-r", "--requirement") else x[2:]
            try:
                for r in parse_requirements(ctx, path):
                    (editables.append(r[1]) if isinstance(r, tuple) else specs.append(r))
            except PipError as e:
                ctx.e(str(e) + "\n")
                return 1
            i += 2 if x in ("-r", "--requirement") else 1
            continue
        if x in ("-e", "--editable") and i + 1 < len(rest):
            editables.append(rest[i + 1])
            i += 2
            continue
        if x in ("-i", "--index-url", "--extra-index-url", "-c", "--constraint", "-t", "--target") and i + 1 < len(rest):
            i += 2
            continue
        if x.startswith("-"):
            i += 1
            continue
        specs.append(x)
        i += 1
    if not specs and not editables:
        ctx.e("ERROR: You must give at least one requirement to install (see \"pip help install\")\n")
        return 1
    out = []
    projects = []
    for e in editables:
        d = ctx.abspath(e)
        info = pyproject_info(ctx, d)
        if info is None:
            ctx.e(f"ERROR: file://{d} does not appear to be a Python project: neither 'setup.py' nor 'pyproject.toml' found.\n")
            return 1
        out += [f"Obtaining file://{d}", "  Installing build dependencies: started", "  Installing build dependencies: finished with status 'done'",
                "  Checking if build backend supports build_editable: started",
                "  Checking if build backend supports build_editable: finished with status 'done'",
                "  Getting requirements to build editable: started", "  Getting requirements to build editable: finished with status 'done'",
                "  Preparing editable metadata (pyproject.toml): started",
                "  Preparing editable metadata (pyproject.toml): finished with status 'done'"]
        projects.append((info[0], info[1], d, info[2]))
    try:
        items = resolve(ctx, specs + [dep for p in projects for dep in p[3]], have, upgrade)
    except PipError as e:
        ctx.w("".join(x + "\n" for x in out))
        ctx.e(str(e) + "\n")
        return 1
    site = shown_site(ctx, root, kind)
    for it in items:
        frm = f" (from {it['parent']})" if it["parent"] else ""
        if it["status"] == "satisfied":
            out.append(f"Requirement already satisfied: {it['name']}{it['spec']} in {site}{frm} ({it['version']})")
        else:
            out.append(f"Collecting {it['name']}{it['spec']}{frm}")
            out.append(f"  Downloading {it['key'].replace('-', '_')}-{it['version']}-py3-none-any.whl.metadata "
                       f"({1 + sum(map(ord, it['key'])) % 60}.{len(it['key']) % 10} kB)")
    new = [it for it in items if it["status"] != "satisfied"]
    for it in new:
        out.append(f"Downloading {it['key'].replace('-', '_')}-{it['version']}-py3-none-any.whl ({wheel_size(it['key'])})")
    order = install_order(items)
    names = [it["name"] for it in order] + [p[0] for p in projects]
    if names:
        out.append("Installing collected packages: " + ", ".join(names))
    for it in order:
        cur = have.get(it["key"])
        if cur is not None:
            out += [f"  Attempting uninstall: {cur[0]}", f"    Found existing installation: {cur[0]} {cur[1]}",
                    f"    Uninstalling {cur[0]}-{cur[1]}:", f"      Successfully uninstalled {cur[0]}-{cur[1]}"]
        set_installed(ctx, root, kind, INDEX_NAME.get(it["key"], it["key"]), it["version"])
    for name, ver, d, _ in projects:
        out.append(f"  Running setup.py develop for {name}" if False else f"  Building editable for {name} (pyproject.toml): started")
        out.append(f"  Building editable for {name} (pyproject.toml): finished with status 'done'")
        set_installed(ctx, root, kind, name, ver, editable=d)
    done = sorted([(INDEX_NAME.get(it["key"], it["key"]), it["version"]) for it in order] + [(p[0], p[1]) for p in projects])
    if done:
        out.append("Successfully installed " + " ".join(f"{n}-{v}" for n, v in done))
    if not quiet:
        ctx.w("".join(x + "\n" for x in out))
    pipver = installed(ctx, root, kind).get("pip", ("pip", "24.0"))[1]
    if kind != "system" and vkey(pipver) < vkey(INDEX["pip"][0]):
        yield from ctx.flush()
        ctx.e(f"\n[notice] A new release of pip is available: {pipver} -> {INDEX['pip'][0]}\n"
              f"[notice] To update, run: pip install --upgrade pip\n")
    return 0


def _satisfies(v, spec):
    if not spec:
        return True
    for clause in spec.split(","):
        m = re.match(r"\s*(===|==|>=|<=|~=|!=|<|>)\s*(\S+)", clause)
        if not m:
            continue
        op, t = m.groups()
        a, b = vkey(v), vkey(t)
        if (op in ("==", "===") and v != t) or (op == ">=" and a < b) or (op == "<=" and a > b) or \
                (op == ">" and a <= b) or (op == "<" and a >= b) or (op == "!=" and v == t):
            return False
    return True


def shown_site(ctx, root, kind):
    d = site_dir(root, kind)
    cwd = ctx.cwd.rstrip("/")
    if d.startswith(cwd + "/"):
        return "./" + d[len(cwd) + 1:]
    return d


PEP668 = """error: externally-managed-environment

× This environment is externally managed
╰─> To install Python packages system-wide, try apt install
    python3-xyz, where xyz is the package you are trying to
    install.

    If you wish to install a non-Debian-packaged Python package,
    create a virtual environment using python3 -m venv path/to/venv.
    Then use path/to/venv/bin/python and path/to/venv/bin/pip. Make
    sure you have python3-full installed.

    If you wish to install a non-Debian packaged Python application,
    it may be easiest to use pipx install xyz, which will manage a
    virtual environment for you. Make sure you have pipx installed.

    See /usr/share/doc/python3.12/README.venv for more information.

note: If you believe this is a mistake, please contact your Python installation or OS distribution provider. You can override this, at the risk of breaking your Python installation or OS, by passing --break-system-packages.
hint: See PEP 668 for the detailed specification.
"""


def pip_main(ctx, args, root, kind, prog="pip"):
    if not args:
        ctx.w("\nUsage:   \n  pip <command> [options]\n\nCommands:\n  install                     Install packages.\n"
              "  uninstall                   Uninstall packages.\n  freeze                      Output installed packages in requirements format.\n"
              "  list                        List installed packages.\n  show                        Show information about installed packages.\n")
        return 0
    cmd, rest = args[0], args[1:]
    have = installed(ctx, root, kind)
    if cmd in ("--version", "-V"):
        ver = have.get("pip", ("pip", "24.0"))[1]
        ctx.w(f"pip {ver} from {site_dir(root, kind)}/pip (python 3.12)\n")
        return 0
    if cmd == "install":
        return (yield from pip_install(ctx, rest, root, kind, have))
    if cmd == "uninstall":
        yes = any(x in ("-y", "--yes") for x in rest)
        names = [x for x in rest if not x.startswith("-")]
        if kind == "system":
            ctx.e(PEP668)
            return 1
        for n in names:
            cur = have.get(norm(n))
            if cur is None:
                ctx.w(f"WARNING: Skipping {n} as it is not installed.\n")
                continue
            ctx.w(f"Found existing installation: {cur[0]} {cur[1]}\nUninstalling {cur[0]}-{cur[1]}:\n")
            if not yes:
                ctx.w(f"  Would remove:\n    {site_dir(root, kind)}/{cur[0].replace('-', '_')}-{cur[1]}.dist-info/*\n"
                      f"    {site_dir(root, kind)}/{cur[0].replace('-', '_')}/*\nProceed (Y/n)? ")
                yield from ctx.flush()
                from ..fileio import readline
                ans = (yield from readline(ctx.k, ctx.proc.fds.get(0))).decode(errors="replace").strip()
                if ans.lower() not in ("", "y", "yes"):
                    continue
            remove_installed(ctx, root, kind, n)
            ctx.w(f"  Successfully uninstalled {cur[0]}-{cur[1]}\n")
        return 0
    if cmd == "freeze":
        lines = []
        for key, (disp, ver) in sorted(have.items()):
            if key in ("pip", "setuptools", "wheel", "distribute"):
                continue
            d = site_dir(root, kind)
            try:
                du = bytes(ctx.fs.lookup(f"{d}/{disp.replace('-', '_')}-{ver}.dist-info/direct_url.json").data).decode()
                path = re.search(r'file://([^"]+)', du).group(1)
                lines.append(f"# Editable install with no version control ({disp}=={ver})\n-e {path}")
                continue
            except (FSError, AttributeError):
                pass
            lines.append(f"{disp}=={ver}")
        ctx.w("".join(x + "\n" for x in lines))
        return 0
    if cmd == "list":
        outdated = any(x in ("-o", "--outdated") for x in rest)
        fmt_free = "--format=freeze" in rest
        rows = []
        for key, (disp, ver) in sorted(have.items()):
            latest = INDEX.get(key, (ver,))[0]
            if outdated and vkey(latest) <= vkey(ver):
                continue
            rows.append((disp, ver, latest))
        if fmt_free:
            ctx.w("".join(f"{a}=={b}\n" for a, b, _ in rows))
            return 0
        if outdated:
            if not rows:
                return 0
            hdr = ("Package", "Version", "Latest", "Type")
            data = [(a, b, c, "wheel") for a, b, c in rows]
        else:
            hdr = ("Package", "Version")
            data = [(a, b) for a, b, _ in rows]
        widths = [max(len(r[k]) for r in [hdr] + data) for k in range(len(hdr))]
        lines = [" ".join(h.ljust(w) for h, w in zip(hdr, widths)).rstrip(), " ".join("-" * w for w in widths)]
        for r in data:
            lines.append(" ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip())
        ctx.w("".join(x + "\n" for x in lines))
        return 0
    if cmd == "show":
        status = 0
        names = [x for x in rest if not x.startswith("-")]
        for k, n in enumerate(names):
            cur = have.get(norm(n))
            if cur is None:
                ctx.e(f"WARNING: Package(s) not found: {n}\n")
                status = 1
                continue
            info = INDEX.get(norm(n), ("", [], [], ""))
            req_by = sorted(disp for key, (disp, _) in have.items() if norm(n) in [norm(x) for x in INDEX.get(key, ("", [], [], ""))[2]])
            if k:
                ctx.w("---\n")
            ctx.w(f"Name: {cur[0]}\nVersion: {cur[1]}\nSummary: {info[3]}\nHome-page: \nAuthor: \nAuthor-email: \nLicense: \n"
                  f"Location: {site_dir(root, kind)}\nRequires: {', '.join(info[2])}\nRequired-by: {', '.join(req_by)}\n")
        return status
    if cmd in ("help", "-h", "--help"):
        return (yield from pip_main(ctx, [], root, kind))
    if cmd == "check":
        ctx.w("No broken requirements found.\n")
        return 0
    ctx.e(f'ERROR: unknown command "{cmd}"\n')
    return 1


@util("pip", "pip3")
def u_pip(ctx):
    root, kind = env_of(ctx)
    return (yield from pip_main(ctx, ctx.argv[1:], root, kind))


# ---------------------------------------------------------------------------- running Python code

class PySandbox:
    """Redirect open()/os/pathlib to the simulated filesystem while Python code runs."""

    def __init__(self, ctx, stdin_text, root, kind):
        self.ctx, self.fs = ctx, ctx.fs
        self.stdin_text = stdin_text
        self.root, self.kind = root, kind
        self.events = []        # (virtual offset seconds, stream, text)
        self.offset = 0.0
        self.cwd = ctx.cwd
        self.site = site_dir(root, kind)
        self.python_path = "/usr/bin/python3" if kind == "system" else f"{root}/bin/python"
        self.writers = {}
        self.filedata = {}

    def emit(self, stream, text):
        self.events.append((self.offset, stream, text))

    def is_tty(self, fd):
        of = self.ctx.proc.fds.get(fd)
        return of is not None and of.kind == "tty"

    def abspath(self, p):
        p = os.fspath(p)
        if isinstance(p, bytes):
            p = p.decode()
        if not p.startswith("/"):
            p = self.cwd.rstrip("/") + "/" + p
        import posixpath
        return posixpath.normpath(p)

    # ---- file API
    def open(self, file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None):
        if isinstance(file, int):
            if file in self.writers:
                return self.writers[file]
            raise OSError(9, "Bad file descriptor")
        path = self.abspath(file)
        fs = self.fs
        binary = "b" in mode
        if any(c in mode for c in "wax+"):
            try:
                node = fs.lookup(path)
                if node.type == "d":
                    raise IsADirectoryError(21, "Is a directory", os.fspath(file))
                if "x" in mode:
                    raise FileExistsError(17, "File exists", os.fspath(file))
                if not fs.can(node, 2, self.ctx.uid, self.ctx.proc.gids):
                    raise PermissionError(13, "Permission denied", os.fspath(file))
            except FSError:
                try:
                    parent = fs.lookup(path.rsplit("/", 1)[0] or "/")
                except FSError:
                    raise FileNotFoundError(2, "No such file or directory", os.fspath(file)) from None
                if not fs.can(parent, 3, self.ctx.uid, self.ctx.proc.gids):
                    raise PermissionError(13, "Permission denied", os.fspath(file)) from None
                node = fs.create(path, "/", 0o666 & ~self.ctx.proc.umask, self.ctx.uid, self.ctx.proc.gid)
            if "w" in mode:
                self.filedata[node] = b""
                self.emit(("file", node), b"")
            start = self.data_of(node) if ("a" in mode or "+" in mode) else b""
            return _VFSFile(self, node, start, binary, os.fspath(file), append="a" in mode)
        try:
            node = fs.lookup(path)
        except FSError as e:
            raise FileNotFoundError(2, "No such file or directory", os.fspath(file)) from None
        if node.type == "d":
            raise IsADirectoryError(21, "Is a directory", os.fspath(file))
        if not fs.can(node, 4, self.ctx.uid, self.ctx.proc.gids):
            raise PermissionError(13, "Permission denied", os.fspath(file))
        data = self.data_of(node)
        if binary:
            return io.BytesIO(data)
        return io.StringIO(data.decode(encoding or "utf-8", errors or "strict"), newline=None)

    def data_of(self, node):
        """A file's contents as this program sees them (its own writes are released at their simulated times)."""
        return self.filedata.get(node, bytes(node.data))

    def exists(self, p):
        try:
            self.fs.lookup(self.abspath(p))
            return True
        except FSError:
            return False

    def isfile(self, p):
        try:
            return self.fs.lookup(self.abspath(p)).type == "f"
        except FSError:
            return False

    def isdir(self, p):
        try:
            return self.fs.lookup(self.abspath(p)).type == "d"
        except FSError:
            return False

    def listdir(self, p="."):
        try:
            node = self.fs.lookup(self.abspath(p))
        except FSError:
            raise FileNotFoundError(2, "No such file or directory", os.fspath(p)) from None
        if node.type != "d":
            raise NotADirectoryError(20, "Not a directory", os.fspath(p))
        return self.fs.listdir(node)

    def makedirs(self, p, mode=0o777, exist_ok=False):
        path = self.abspath(p)
        if self.exists(path) and not exist_ok:
            raise FileExistsError(17, "File exists", os.fspath(p))
        self.fs.mkdir(path, "/", 0o755, self.ctx.uid, self.ctx.proc.gid, parents=True)

    def mkdir(self, p, mode=0o777, *a, **k):
        path = self.abspath(p)
        try:
            self.fs.mkdir(path, "/", 0o755, self.ctx.uid, self.ctx.proc.gid)
        except FSError as e:
            if e.code == "EEXIST":
                raise FileExistsError(17, "File exists", os.fspath(p)) from None
            raise FileNotFoundError(2, "No such file or directory", os.fspath(p)) from None

    def remove(self, p, *a, **k):
        try:
            self.fs.unlink(self.abspath(p))
        except FSError:
            raise FileNotFoundError(2, "No such file or directory", os.fspath(p)) from None

    def rename(self, a, b, *x, **k):
        try:
            self.fs.rename(self.abspath(a), self.abspath(b))
        except FSError:
            raise FileNotFoundError(2, "No such file or directory", os.fspath(a)) from None

    def getsize(self, p):
        try:
            n = self.fs.lookup(self.abspath(p))
            return len(self.filedata[n]) if n in self.filedata else n.size
        except FSError:
            raise FileNotFoundError(2, "No such file or directory", os.fspath(p)) from None

    def getmtime(self, p):
        try:
            return self.fs.lookup(self.abspath(p)).mtime / NS
        except FSError:
            raise FileNotFoundError(2, "No such file or directory", os.fspath(p)) from None

    def stat(self, p, *a, **k):
        try:
            n = self.fs.lookup(self.abspath(p))
        except FSError:
            raise FileNotFoundError(2, "No such file or directory", os.fspath(p)) from None
        import stat as S
        from ..vfs import TYPE_BITS
        return os.stat_result((TYPE_BITS[n.type] | n.mode, n.ino, 2050, n.nlink, n.uid, n.gid, n.size,
                               int(n.atime / NS), int(n.mtime / NS), int(n.ctime / NS)))

    def scandir(self, p="."):
        sb = self
        base = self.abspath(p)

        class Entry:
            def __init__(self, name):
                self.name = name
                self.path = os.path.join(os.fspath(p), name) if os.fspath(p) != "." else name

            def is_dir(self, follow_symlinks=True):
                return sb.isdir(base + "/" + self.name)

            def is_file(self, follow_symlinks=True):
                return sb.isfile(base + "/" + self.name)

            def is_symlink(self):
                return False

            def stat(self, follow_symlinks=True):
                return sb.stat(base + "/" + self.name)

            def inode(self):
                return 0

            def __fspath__(self):
                return self.path

        class It:
            def __init__(self, names):
                self.names = names

            def __iter__(self):
                return iter([Entry(n) for n in self.names])

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def close(self):
                pass
        return It(self.listdir(p))

    def walk(self, top, topdown=True, onerror=None, followlinks=False):
        try:
            names = self.listdir(top)
        except OSError as e:
            if onerror:
                onerror(e)
            return
        dirs = [n for n in names if self.isdir(os.path.join(top, n))]
        files = [n for n in names if not self.isdir(os.path.join(top, n))]
        if topdown:
            yield top, dirs, files
        for d in dirs:
            yield from self.walk(os.path.join(top, d), topdown, onerror, followlinks)
        if not topdown:
            yield top, dirs, files

    def getcwd(self):
        return self.cwd

    def chdir(self, p):
        path = self.abspath(p)
        if not self.isdir(path):
            raise FileNotFoundError(2, "No such file or directory", os.fspath(p))
        self.cwd = path


class _StreamWriter(io.TextIOBase):
    """sys.stdout / sys.stderr: CPython's buffering (line-buffered on a terminal, 8 KiB blocks otherwise; stderr is
    always line-buffered; -u / PYTHONUNBUFFERED turn buffering off)."""

    def __init__(self, sb, stream, mode):
        self.sb, self.stream, self.mode = sb, stream, mode
        self.pending = []
        self.size = 0

    def write(self, s):
        if isinstance(s, bytes):
            s = s.decode("utf-8", "replace")
        if not isinstance(s, str):
            raise TypeError(f"write() argument must be str, not {type(s).__name__}")
        if not s:
            return 0
        self.pending.append(s)
        self.size += len(s)
        if self.mode == "none":
            self.flush()
        elif self.mode == "line":
            if "\n" in s:
                text = "".join(self.pending)
                cut = text.rfind("\n") + 1
                self.pending, self.size = ([text[cut:]] if cut < len(text) else []), len(text) - cut
                self.sb.emit(self.stream, text[:cut])
        elif self.size >= 8192:
            self.flush()
        return len(s)

    def flush(self):
        if self.pending:
            text = "".join(self.pending)
            self.pending, self.size = [], 0
            self.sb.emit(self.stream, text)

    def isatty(self):
        return self.sb.is_tty(1 if self.stream == "out" else 2)

    def writable(self):
        return True

    @property
    def encoding(self):
        return "utf-8"

    def fileno(self):
        return 1 if self.stream == "out" else 2

    @property
    def buffer(self):
        w = self

        class B:
            def write(self, b):
                return w.write(bytes(b).decode("utf-8", "replace"))

            def flush(self):
                w.flush()
        return B()


class _TTYStdin(io.TextIOBase):
    """sys.stdin on the terminal: reads the lines typed after the command, one at a time."""

    def __init__(self, tty):
        self.tty = tty

    def readline(self, size=-1):
        tty = self.tty
        while True:
            j = tty.input.find(b"\n")
            if j >= 0:
                data = bytes(tty.input[:j + 1])
                del tty.input[:j + 1]
                return data.decode("utf-8", "replace")
            if not tty.take_line():
                data = bytes(tty.input)
                tty.input.clear()
                return data.decode("utf-8", "replace")

    def read(self, size=-1):
        out = []
        while True:
            line = self.readline()
            if not line:
                return "".join(out)
            out.append(line)

    def __iter__(self):
        while True:
            line = self.readline()
            if not line:
                return
            yield line

    def readable(self):
        return True

    def isatty(self):
        return True

    def fileno(self):
        return 0


class _VFSFile(io.TextIOBase):
    """A file opened for writing: every write lands in the simulated file straight away."""

    def __init__(self, sb, node, start, binary, path, append=False):
        self.sb, self.node, self.binary, self.name = sb, node, binary, path
        self.buf = io.BytesIO() if binary else io.StringIO()
        if start:
            self.buf.write(start if binary else start.decode("utf-8", "replace"))
        if not append:
            self.buf.seek(0)
        self.mode = "wb" if binary else "w"

    def write(self, s):
        if self.binary and isinstance(s, str):
            raise TypeError("a bytes-like object is required, not 'str'")
        if not self.binary and not isinstance(s, str):
            raise TypeError(f"write() argument must be str, not {type(s).__name__}")
        n = self.buf.write(s)
        self._sync()
        return n

    def writelines(self, lines):
        for ln in lines:
            self.buf.write(ln)
        self._sync()

    def _sync(self):
        v = self.buf.getvalue()
        data = v if self.binary else v.encode("utf-8", "surrogateescape")
        if self.sb.filedata.get(self.node) != data:
            self.sb.filedata[self.node] = data
            self.sb.emit(("file", self.node), data)

    def read(self, *a):
        return self.buf.read(*a)

    def readable(self):
        return True

    def writable(self):
        return True

    def seek(self, *a):
        return self.buf.seek(*a)

    def tell(self):
        return self.buf.tell()

    def close(self):
        if not self.closed:
            self._sync()
        super().close()

    def isatty(self):
        return False


# ---------------------------------------------------------------------------- imports

IMPORT_NAMES = {"sklearn": "scikit-learn", "yaml": "pyyaml", "PIL": "pillow", "dateutil": "python-dateutil",
                "huggingface_hub": "huggingface-hub", "charset_normalizer": "charset-normalizer",
                "typing_extensions": "typing-extensions", "jupyterlab": "jupyterlab"}
# packages that are always simulated (too big / not available where the site runs)
STUB_ONLY = {"torch", "transformers", "wandb", "accelerate", "datasets", "tensorboard", "jupyterlab", "ipykernel",
             "sklearn", "scipy", "matplotlib", "tokenizers", "safetensors", "huggingface_hub", "einops", "tqdm"}


class Anything:
    """A permissive stand-in: any attribute, call, index or iteration works and returns another stand-in."""

    def __init__(self, name):
        object.__setattr__(self, "_name", name)

    def __getattr__(self, k):
        if k.startswith("__") and k.endswith("__"):
            raise AttributeError(k)
        return Anything(f"{self._name}.{k}")

    def __call__(self, *a, **k):
        return Anything(f"{self._name}()")

    def __getitem__(self, k):
        return Anything(f"{self._name}[]")

    def __iter__(self):
        return iter(())

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __repr__(self):
        return f"<{self._name}>"

    def __bool__(self):
        return True

    def __float__(self):
        return 0.0

    def __int__(self):
        return 0

    __index__ = __int__

    def __format__(self, spec):
        return format(0.0, spec) if spec and spec[-1] in "eEfFgG%" else format(repr(self), spec)


class _StubModule:
    """Module type for simulated third-party packages: real attributes first, then permissive stand-ins."""


def _stub(top, version, ctx, env, sb):
    import types
    m = types.ModuleType(top)
    m.__version__ = version
    m.__file__ = f"{sb.site}/{top}/__init__.py"
    m.__path__ = []
    m.__getattr__ = lambda k: Anything(f"{top}.{k}") if not (k.startswith("__") and k.endswith("__")) else \
        (_ for _ in ()).throw(AttributeError(k))
    subs = {}
    if top == "torch":
        gpus = ctx.host.gpus
        vis = env.get("CUDA_VISIBLE_DEVICES")
        if vis is None:
            visible = list(range(len(gpus)))
        else:
            visible = []
            for x in vis.split(","):
                x = x.strip()
                if not x.isdigit() or int(x) >= len(gpus):
                    break           # CUDA stops at the first invalid entry
                visible.append(int(x))
        m._visible = visible
        cuda = types.ModuleType("torch.cuda")

        def need(i):
            if not visible:
                raise RuntimeError("No CUDA GPUs are available")
            if not 0 <= i < len(visible):
                raise RuntimeError("CUDA error: invalid device ordinal\nCUDA kernel errors might be asynchronously reported at "
                                   "some other API call, so the stacktrace below might be incorrect.")
            return gpus[visible[i]]
        cuda.is_available = lambda: bool(visible)
        cuda.device_count = lambda: len(visible)
        cuda.get_device_name = lambda i=0: need(i if isinstance(i, int) else 0).name
        cuda.current_device = lambda: (need(0), 0)[1]
        cuda.is_bf16_supported = lambda: bool(visible)
        cuda.mem_get_info = lambda i=0: ((need(i).memory - need(i).used) * 2 ** 20, need(i).memory * 2 ** 20)
        cuda.get_device_properties = lambda i=0: types.SimpleNamespace(name=need(i).name, total_memory=need(i).memory * 2 ** 20,
                                                                       major=9, minor=0, multi_processor_count=132)
        cuda.synchronize = lambda *a: None
        cuda.empty_cache = lambda: None
        cuda.__getattr__ = lambda k: Anything(f"torch.cuda.{k}")
        m.cuda = cuda
        subs["torch.cuda"] = cuda
        m.version = types.SimpleNamespace(cuda="12.8", git_version="0" * 40, hip=None)
        mps = types.SimpleNamespace(is_available=lambda: False, is_built=lambda: False)
        m.backends = types.SimpleNamespace(mps=mps, cudnn=types.SimpleNamespace(version=lambda: 91002, enabled=True, benchmark=False))

        class device:
            def __init__(self, kind, index=None):
                self.type, _, idx = str(kind).partition(":")
                self.index = int(idx) if idx else index

            def __repr__(self):
                return f"device(type='{self.type}'" + (f", index={self.index}" if self.index is not None else "") + ")"

            def __str__(self):
                return self.type + (f":{self.index}" if self.index is not None else "")
        m.device = device
        dist = types.ModuleType("torch.distributed")
        dist.is_available = lambda: True
        dist.is_initialized = lambda: "RANK" in env
        dist.init_process_group = lambda *a, **k: None
        dist.get_rank = lambda *a: int(env.get("RANK", "0"))
        dist.get_world_size = lambda *a: int(env.get("WORLD_SIZE", "1"))
        dist.destroy_process_group = lambda *a: None
        dist.barrier = lambda *a, **k: None
        dist.__getattr__ = lambda k: Anything(f"torch.distributed.{k}")
        m.distributed = dist
        subs["torch.distributed"] = dist
        m.manual_seed = lambda s: None
    if top == "tqdm":
        def tqdm(it=None, *a, **k):
            return it if it is not None else Anything("tqdm")
        m.tqdm = tqdm
        m.trange = lambda *a, **k: range(*a)
    if top == "matplotlib":
        plt = types.ModuleType("matplotlib.pyplot")
        plt.__getattr__ = lambda k: Anything(f"matplotlib.pyplot.{k}")

        def savefig(fname, *a, **k):
            path = sb.abspath(fname)
            png = (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x02\x80\x00\x00\x01\xe0\x08\x06\x00\x00\x00"
                   + bytes(range(64)) * 40)
            with sb.open(path, "wb") as f:
                f.write(png)
        plt.savefig = savefig
        plt.show = lambda *a, **k: None
        plt.subplots = lambda *a, **k: (Anything("Figure"), Anything("Axes"))
        m.pyplot = plt
        m.use = lambda *a, **k: None
        subs["matplotlib.pyplot"] = plt
    for name, sub in subs.items():
        sub.__file__ = f"{sb.site}/{name.replace('.', '/')}.py"
    return m, subs


class _Importer:
    """builtins.__import__ while user code runs: imports from user code see the simulated environment."""

    def __init__(self, sb, ctx, env, have, search):
        self.sb, self.ctx, self.env, self.have, self.search = sb, ctx, env, have, search
        self.orig = builtins.__import__
        self.cuda_visible = []   # physical GPUs this run's torch can see
        self.added = []          # sys.modules entries created for this run
        self.patched = []        # (module, old __version__)
        self.ready = set()

    def __call__(self, name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and globals is not None and globals.get("__vfs__"):
            top = name.partition(".")[0]
            if top not in self.ready:
                self.prepare(top)
        return self.orig(name, globals, locals, fromlist, level)

    def prepare(self, top):
        # a module or package on the simulated sys.path (the script's folder, PYTHONPATH, editable installs);
        # packages pip "installed" are simulated below, other code in site-packages is loaded from there
        dist = norm(IMPORT_NAMES.get(top, top))
        known = dist in INDEX and top not in ("pip", "setuptools")
        for d in self.search if not known else [x for x in self.search if x != self.sb.site]:
            for cand, is_pkg in ((f"{d}/{top}.py", False), (f"{d}/{top}/__init__.py", True)):
                if self.sb.isfile(cand):
                    self.load_vfs(top, cand, is_pkg)
                    self.ready.add(top)
                    return
        if not known:
            self.ready.add(top)
            return
        if dist not in self.have:
            raise ModuleNotFoundError(f"No module named '{top}'", name=top)
        version = self.have[dist][1]
        real = None
        if top not in STUB_ONLY:
            real = sys.modules.get(top)
            if real is None:
                import importlib.util
                try:
                    if importlib.util.find_spec(top) is not None:
                        real = self.orig(top)
                except Exception:  # noqa: BLE001
                    real = None
        if real is not None:
            self.patched.append((real, getattr(real, "__version__", None)))
            try:
                real.__version__ = version
            except (AttributeError, TypeError):
                pass
        else:
            saved = {k: sys.modules[k] for k in sys.modules if k == top or k.startswith(top + ".")}
            mod, subs = _stub(top, version, self.ctx, self.env, self.sb)
            if top == "torch":
                self.cuda_visible = getattr(mod, "_visible", [])
            self.added.append((top, saved))
            sys.modules[top] = mod
            for k, v in subs.items():
                sys.modules[k] = v
        self.ready.add(top)

    def load_vfs(self, top, path, is_pkg):
        import types
        code = bytes(self.sb.fs.lookup(path).data).decode("utf-8", "replace")
        mod = types.ModuleType(top)
        mod.__file__ = path
        mod.__vfs__ = True
        mod.__builtins__ = builtins
        if is_pkg:
            mod.__path__ = [path.rsplit("/", 1)[0]]
        saved = {k: sys.modules[k] for k in sys.modules if k == top or k.startswith(top + ".")}
        self.added.append((top, saved))
        sys.modules[top] = mod
        exec(compile(code, path, "exec"), mod.__dict__)

    def undo(self):
        for mod, old in self.patched:
            try:
                if old is None:
                    del mod.__version__
                else:
                    mod.__version__ = old
            except (AttributeError, TypeError):
                pass
        for top, saved in reversed(self.added):
            for k in [k for k in sys.modules if k == top or k.startswith(top + ".")]:
                del sys.modules[k]
            sys.modules.update(saved)


def search_path(sb, ctx, env, root, kind, filename):
    """The simulated sys.path entries that hold user code."""
    first = sb.cwd if filename.startswith("<") else sb.abspath(filename).rsplit("/", 1)[0] or "/"
    out = [first]
    out += [sb.abspath(p) for p in env.get("PYTHONPATH", "").split(":") if p]
    d = site_dir(root, kind)
    try:
        for name in ctx.fs.listdir(ctx.fs.lookup(d)):
            if name.endswith(".dist-info"):
                try:
                    du = bytes(ctx.fs.lookup(f"{d}/{name}/direct_url.json").data).decode()
                except FSError:
                    continue
                m = re.search(r'file://([^"]+)', du)
                if m:
                    out += [m.group(1), m.group(1) + "/src"]
    except FSError:
        pass
    out.append(d)
    return out


class _Run:
    """Patch the interpreter so user code sees the simulated machine; restore everything afterwards."""

    def __init__(self, ctx, env, root, kind, argv, filename, stdin_text):
        self.ctx, self.env = ctx, env
        sb = self.sb = PySandbox(ctx, stdin_text, root, kind)
        unbuffered = bool(env.get("PYTHONUNBUFFERED"))
        self.out = _StreamWriter(sb, "out", "none" if unbuffered else ("line" if sb.is_tty(1) else "block"))
        self.err = _StreamWriter(sb, "err", "none" if unbuffered else "line")
        sb.writers = {1: self.out, 2: self.err}
        of = ctx.proc.fds.get(0)
        if of is not None and of.kind == "tty" and stdin_text is None:
            self.stdin = _TTYStdin(of.tty)
        else:
            self.stdin = io.StringIO(stdin_text or "")
        have = installed(ctx, root, kind)
        self.importer = _Importer(sb, ctx, env, have, search_path(sb, ctx, env, root, kind, filename))
        self.argv = argv

    def __enter__(self):
        import time as _time
        sb, ctx = self.sb, self.ctx
        world_now = ctx.now

        def fake_sleep(s):
            if s < 0:
                raise ValueError("sleep length must be non-negative")
            sb.offset += float(s)

        def fake_time():
            return world_now / NS + sb.offset

        def fake_perf():
            return 1000.0 + sb.offset
        self.saved = []
        patches = [(builtins, "open", sb.open), (io, "open", sb.open), (os, "listdir", sb.listdir),
                   (os, "makedirs", sb.makedirs), (os, "mkdir", sb.mkdir), (os, "remove", sb.remove),
                   (os, "unlink", sb.remove), (os, "rename", sb.rename), (os, "replace", sb.rename),
                   (os, "getcwd", sb.getcwd), (os, "chdir", sb.chdir), (os, "stat", sb.stat), (os, "scandir", sb.scandir),
                   (os, "walk", sb.walk), (os.path, "exists", sb.exists), (os.path, "isfile", sb.isfile),
                   (os.path, "isdir", sb.isdir), (os.path, "getsize", sb.getsize), (os.path, "getmtime", sb.getmtime),
                   (os, "getpid", lambda: ctx.proc.pid), (os, "getppid", lambda: ctx.proc.ppid),
                   (os, "cpu_count", lambda: len(ctx.host.extra.get("cpus", [])) or ctx.host.extra.get("nproc", 8)),
                   (_time, "sleep", fake_sleep), (_time, "time", fake_time), (_time, "perf_counter", fake_perf),
                   (_time, "monotonic", fake_perf), (sys, "stdin", self.stdin), (sys, "stdout", self.out),
                   (sys, "stderr", self.err), (sys, "argv", list(self.argv)), (builtins, "__import__", self.importer),
                   (sys, "executable", self.sb.python_path)]
        for obj, attr, val in patches:
            self.saved.append((obj, attr, getattr(obj, attr)))
            setattr(obj, attr, val)
        self.saved_env = dict(os.environ)
        os.environ.clear()
        os.environ.update(self.env)
        return self

    def __exit__(self, *exc):
        try:
            self.out.flush()
            self.err.flush()
        finally:
            for obj, attr, val in reversed(self.saved):
                setattr(obj, attr, val)
            os.environ.clear()
            os.environ.update(self.saved_env)
            self.importer.undo()
        return False


def format_traceback(e, code, filename):
    tb = traceback.extract_tb(e.__traceback__)
    lines = ["Traceback (most recent call last):"]
    shown = [f for f in tb if "bashlite" not in f.filename]
    for f in shown:
        lines.append(f'  File "{f.filename}", line {f.lineno}, in {f.name}')
        src = None
        if f.filename == filename:
            src = code.split("\n")
        elif f.line:
            src = None
            lines.append("    " + f.line.strip())
            continue
        if src is not None and 0 < f.lineno <= len(src) and filename not in ("<stdin>",) and src[f.lineno - 1].strip():
            lines.append("    " + src[f.lineno - 1].strip())
    if isinstance(e, SyntaxError) and not shown:
        lines = []
        lines.append(f'  File "{e.filename}", line {e.lineno}')
        if e.text:
            t = e.text.rstrip("\n")
            stripped = t.lstrip()
            lines.append("    " + stripped)
            if e.offset:
                start = e.offset - (len(t) - len(stripped))
                end = (e.end_offset or e.offset + 1) - (len(t) - len(stripped))
                lines.append("    " + " " * (start - 1) + "^" * max(1, end - start))
    lines += [x.rstrip("\n") for x in traceback.format_exception_only(type(e), e)][-1:]
    return "\n".join(lines) + "\n"


def run_python(ctx, code, filename, argv, env, root, kind, stdin_text=None, module=None):
    """Execute Python source on the simulated machine. Returns (status, events, offset)."""
    status = 0
    run = _Run(ctx, env, root, kind, argv, filename, stdin_text)
    ns = {"__name__": "__main__", "__file__": filename, "__builtins__": builtins, "__vfs__": True}
    if filename.startswith("<"):
        del ns["__file__"]
    with run:
        try:
            compiled = compile(code, filename, "exec")
            exec(compiled, ns)
        except SystemExit as e:
            c = e.code
            if c is None:
                status = 0
            elif isinstance(c, int):
                status = c & 0xFF
            else:
                run.err.write(f"{c}\n")
                status = 1
        except KeyboardInterrupt:
            run.err.write("Traceback (most recent call last):\nKeyboardInterrupt\n")
            status = 130
        except BaseException as e:  # noqa: BLE001
            run.err.write(format_traceback(e, code, filename))
            status = 1
    # code that imported torch and talks about CUDA holds memory on its (first visible) GPU while it runs
    vis = run.importer.cuda_visible
    if vis and re.search(r"cuda|\.to\(\s*device|device\s*=", code):
        rank = int(env.get("LOCAL_RANK", "0") or 0) if str(env.get("LOCAL_RANK", "0")).isdigit() else 0
        mib = 2048 + 512 * (len(code) % 7)
        ctx.gpu_claim = getattr(ctx, "gpu_claim", []) + [(vis[rank % len(vis)], mib)]
    return status, run.sb.events, run.sb.offset


def replay(ctx, events, start):
    """Release recorded output and file writes at their virtual times."""
    k = ctx.k
    for t, stream, data in events:
        due = start + int(t * NS)
        if due > k.now:
            yield ("sleep", due)
        if stream == "out":
            ctx.w(data)
            yield from ctx.flush()
        elif stream == "err":
            ctx.e(data)
        else:
            node = stream[1]
            node.data[:] = data
            node.mtime = node.ctime = k.now
    yield from ctx.flush()


# ---------------------------------------------------------------------------- python

PY_BANNER = (f"Python {PY_VERSION} (main, Sep 11 2024, 14:17:37) [GCC 13.2.0] on linux\n"
             "Type \"help\", \"copyright\", \"credits\" or \"license\" for more information.\n")
PY_USAGE = ("usage: python3 [option] ... [-c cmd | -m mod | file | -] [arg] ...\n"
            "Try `python -h' for more information.\n")


@util("python3.12")
def u_python(ctx):
    root, kind = env_of(ctx)
    a = ctx.argv[1:]
    env = dict(ctx.env)
    i = 0
    while i < len(a):
        x = a[i]
        if x in ("-V", "--version"):
            ctx.w(f"Python {(conda_pyver(ctx.fs, root) if kind == 'conda' else None) or PY_VERSION}\n")
            return 0
        if x == "-VV":
            ctx.w(f"Python {PY_VERSION} (main, Sep 11 2024, 14:17:37) [GCC 13.2.0]\n")
            return 0
        if x.startswith("-c"):
            code = x[2:] or (a[i + 1] if i + 1 < len(a) else None)
            if code is None:
                ctx.e("Argument expected for the -c option\n" + PY_USAGE)
                return 2
            rest = a[i + (1 if x[2:] else 2):]
            return (yield from _exec_code(ctx, code, "<string>", ["-c"] + rest, env, root, kind))
        if x.startswith("-m"):
            mod = x[2:] or (a[i + 1] if i + 1 < len(a) else None)
            if mod is None:
                ctx.e("Argument expected for the -m option\n" + PY_USAGE)
                return 2
            return (yield from _run_module(ctx, mod, a[i + (1 if x[2:] else 2):], env, root, kind))
        if x in ("-h", "--help", "-?"):
            ctx.w("usage: python3 [option] ... [-c cmd | -m mod | file | -] [arg] ...\nOptions (and corresponding environment variables):\n"
                  "-c cmd : program passed in as string (terminates option list)\n-m mod : run library module as a script (terminates option list)\n"
                  "-u     : force the stdout and stderr streams to be unbuffered;\n-V     : print the Python version number and exit (also --version)\n")
            return 0
        if x.startswith("-") and x != "-" and len(x) > 1:
            for c in x[1:]:
                if c == "u":
                    env["PYTHONUNBUFFERED"] = "1"
                elif c not in "BdEhiIOqsSvx":
                    ctx.e(f"Unknown option: -{c}\n" + PY_USAGE)
                    return 2
            i += 1
            continue
        break
    rest = a[i:]
    if not rest or rest[0] == "-":
        if ctx.stdin_is_tty() and not rest:
            return (yield from _repl(ctx, env, root, kind))
        data = yield from ctx.read_fd(0)
        return (yield from _exec_code(ctx, data.decode("utf-8", "replace"), "<stdin>", ["-"] + rest[1:], env, root, kind, ""))
    script = rest[0]
    prog = ctx.argv[0].rsplit("/", 1)[-1]
    try:
        node = ctx.lookup(script)
    except FSError:
        ctx.e(f"{prog}: can't open file '{ctx.abspath(script)}': [Errno 2] No such file or directory\n")
        return 2
    if node.type == "d":
        main = ctx.abspath(script).rstrip("/") + "/__main__.py"
        if not ctx.fs.exists(main):
            ctx.e(f"{prog}: can't find '__main__' module in '{script}'\n")
            return 1
        node = ctx.fs.lookup(main)
    code = bytes(node.data).decode("utf-8", "replace")
    m = re.search(r"^from (\S+) import main$", code[:400], re.M)
    if m and m.group(1) in ENTRY and code.startswith("#!"):
        # an installed console script (pip, jupyter, torchrun, ...): run the simulated program for its environment
        from ..exec import Ctx, UTILS
        sub = Ctx(ctx.proc, rest, env, ctx.shell)
        sub.path = ctx.abspath(script)
        sub.name = rest[0].rsplit("/", 1)[-1]
        status = yield from UTILS[ENTRY[m.group(1)]](sub)
        yield from sub.flush()
        return status
    return (yield from _exec_code(ctx, code, script, rest, env, root, kind))


def _exec_code(ctx, code, filename, argv, env, root, kind, stdin_text=None):
    if stdin_text is None:
        of = ctx.proc.fds.get(0)
        if of is not None and of.kind != "tty" and re.search(r"\bstdin\b|\binput\s*\(|fileinput", code):
            data = yield from ctx.read_fd(0)
            stdin_text = data.decode("utf-8", "replace")
        elif of is None or of.kind != "tty":
            stdin_text = ""
    start = ctx.now
    ctx.gpu_claim = []
    status, events, offset = run_python(ctx, code, filename, argv, env, root, kind, stdin_text)
    return (yield from _play(ctx, events, start, offset, status))


def _play(ctx, events, start, offset, status):
    """Release a finished run's output at its simulated times, holding its GPU memory meanwhile."""
    held = ctx.proc.gpu
    ctx.proc.gpu = held + ctx.gpu_claim
    try:
        yield from replay(ctx, events, start)
        end = start + int(offset * NS)
        if end > ctx.now:
            yield ("sleep", end)
    finally:
        ctx.proc.gpu = held
    return status


def _repl(ctx, env, root, kind):
    """The interactive interpreter, reading the lines typed after `python3`."""
    import code as codemod
    from ..fileio import readline
    ctx.e(PY_BANNER)
    ns = {"__name__": "__main__", "__builtins__": builtins, "__vfs__": True}
    lines = []
    prompt = ">>> "
    while True:
        ctx.w(prompt)
        yield from ctx.flush()
        line = yield from readline(ctx.k, ctx.proc.fds.get(0))
        if not line:
            ctx.w("\n")
            yield from ctx.flush()
            return 0
        text = line.decode("utf-8", "replace").rstrip("\n")
        lines.append(text)
        src = "\n".join(lines)
        try:
            compiled = codemod.compile_command(src, "<stdin>", "single")
        except (SyntaxError, OverflowError, ValueError) as e:
            ctx.e(format_traceback(e, src, "<stdin>"))
            lines, prompt = [], ">>> "
            continue
        if compiled is None:
            prompt = "... "
            continue
        lines, prompt = [], ">>> "
        status, events, offset, exited = _run_single(ctx, compiled, src, ns, env, root, kind)
        start = ctx.now
        yield from replay(ctx, events, start)
        if start + int(offset * NS) > ctx.now:
            yield ("sleep", start + int(offset * NS))
        if exited:
            return status


def _run_single(ctx, compiled, src, ns, env, root, kind):
    run = _Run(ctx, env, root, kind, [""], "<stdin>", None)
    run.out.mode = "line"
    status, exited = 0, False

    def hook(v):
        if v is not None:
            builtins._ = v
            run.out.write(repr(v) + "\n")
    with run:
        saved = sys.displayhook
        sys.displayhook = hook
        try:
            exec(compiled, ns)
        except SystemExit as e:
            exited = True
            status = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
            if e.code is not None and not isinstance(e.code, int):
                run.err.write(f"{e.code}\n")
        except BaseException as e:  # noqa: BLE001
            run.err.write(format_traceback(e, src, "<stdin>"))
        finally:
            sys.displayhook = saved
    return status, run.sb.events, run.sb.offset, exited


def _run_module(ctx, mod, args, env, root, kind):
    if mod == "venv":
        opts = [x for x in args if x.startswith("-")]
        dirs = [x for x in args if not x.startswith("-")]
        if not dirs:
            ctx.e("usage: venv [-h] [--system-site-packages] [--symlinks | --copies] [--clear] [--upgrade] [--without-pip]\n"
                  "            [--prompt PROMPT] [--upgrade-deps]\n            ENV_DIR [ENV_DIR ...]\nvenv: error: the following arguments are required: ENV_DIR\n")
            return 2
        prompt = None
        if "--prompt" in args:
            prompt = args[args.index("--prompt") + 1]
            dirs = [d for d in dirs if d != prompt]
        for d in dirs:
            make_venv(ctx, d, prompt)
        return 0
    if mod == "pip":
        return (yield from pip_main(ctx, args, root, kind, prog="python -m pip"))
    if mod == "http.server":
        port = int(args[0]) if args and args[0].isdigit() else 8000
        directory = ctx.cwd
        if "--directory" in args or "-d" in args:
            k = args.index("--directory") if "--directory" in args else args.index("-d")
            directory = ctx.abspath(args[k + 1])
        bind = "0.0.0.0"
        if "--bind" in args or "-b" in args:
            k = args.index("--bind") if "--bind" in args else args.index("-b")
            bind = args[k + 1]
        return (yield from serve(ctx, port, f"Serving HTTP on {bind} port {port} (http://{bind if bind != '0.0.0.0' else '0.0.0.0'}:{port}/) ...\n",
                                 "http.server", directory=directory))
    if mod == "ipykernel":
        if norm("ipykernel") not in installed(ctx, root, kind):
            ctx.e(f"{ctx.argv[0].rsplit('/', 1)[-1]}: No module named ipykernel\n")
            return 1
        name = args[args.index("--name") + 1] if "--name" in args else "python3"
        display = args[args.index("--display-name") + 1] if "--display-name" in args else name
        base = (ctx.env.get("HOME", "") + "/.local/share/jupyter/kernels/") if "--user" in args else "/usr/local/share/jupyter/kernels/"
        exe = (root + "/bin/python") if kind != "system" else "/usr/bin/python3"
        spec = ('{\n "argv": [\n  "' + exe + '",\n  "-m",\n  "ipykernel_launcher",\n  "-f",\n  "{connection_file}"\n ],\n'
                f' "display_name": "{display}",\n "language": "python",\n "metadata": {{\n  "debugger": true\n }}\n}}\n')
        ctx.fs.write_file(base + name + "/kernel.json", spec, "/", 0o644, ctx.uid, ctx.proc.gid)
        ctx.w(f"Installed kernelspec {name} in {base}{name}\n")
        return 0
    if mod in ("json.tool", "timeit", "this", "calendar", "base64", "zipfile", "tarfile", "platform", "site", "uuid"):
        import runpy
        code = f"import runpy, sys; sys.argv = {[mod] + list(args)!r}; runpy.run_module({mod!r}, run_name='__main__', alter_sys=False)"
        return (yield from _exec_code(ctx, code, f"<{mod}>", [mod] + list(args), env, root, kind))
    # a module in the current directory / project
    cand = ctx.abspath(mod.replace(".", "/") + ".py")
    if ctx.fs.exists(cand):
        code = bytes(ctx.fs.lookup(cand).data).decode()
        return (yield from _exec_code(ctx, code, cand, [cand] + list(args), env, root, kind))
    ctx.e(f"{ctx.argv[0].rsplit('/', 1)[-1]}: No module named {mod}\n")
    return 1


def serve(ctx, port, banner, kind, directory=None, handler=None, stderr_banner=False, shutdown="\nKeyboard interrupt received, exiting.\n"):
    """Run a server: listen on `port`, run until Ctrl-C in the foreground or forever in the background."""
    from ..world import Service
    from .net import static_handler
    h = ctx.host
    existing = h.services.get(port)
    if existing is not None and (existing.proc is None or existing.proc.state != "Z"):
        ctx.e(f"OSError: [Errno 98] Address already in use\n")
        return 1
    logs = []

    def logging_handler(inner):
        def wrapped(method, path, headers, body):
            res = inner(method, path, headers, body)
            when = strftime("%d/%b/%Y %H:%M:%S", local_dt(ctx.now, ctx.tz))
            if kind == "http.server":
                ctx.e(f'127.0.0.1 - - [{when}] "{method} {path} HTTP/1.1" {res.status} -\n')
            return res
        return wrapped
    base = handler or static_handler(h, directory or ctx.cwd)
    h.services[port] = Service(port, ctx.proc, logging_handler(base), kind)
    ctx.proc.ports.append(port)
    if stderr_banner:
        ctx.e(banner)
    else:
        ctx.w(banner)
    yield from ctx.flush()
    tty = ctx.world.tty_of(ctx.proc)
    fg = ctx.foreground()
    try:
        if fg:
            yield ("sleep", ctx.now + ctx.world.ctrl_c_after * NS)
            ctx.world.tty.write("echo", b"^C")
            ctx.e(shutdown) if stderr_banner else ctx.w(shutdown)
            return 130 if kind != "http.server" else 0
        while True:
            yield ("forever",)
    finally:
        if h.services.get(port) is not None and h.services[port].proc is ctx.proc:
            del h.services[port]


# ---------------------------------------------------------------------------- jupyter / tensorboard / torchrun

def _need(ctx, root, kind, dist, cmd):
    if norm(dist) not in installed(ctx, root, kind):
        ctx.e(f"Command '{cmd}' not found, did you mean:\n  command 'jupyter' from deb jupyter-core (5.7.2-1)\nTry: sudo apt install <deb name>\n"
              if cmd == "jupyter" else f"{cmd}: command not found\n")
        return False
    return True


@util("jupyter", "jupyter-lab")
def u_jupyter(ctx):
    root, kind = env_of(ctx)
    a = ctx.argv[1:] if ctx.name == "jupyter" else ["lab"] + ctx.argv[1:]
    if not a:
        ctx.w("usage: jupyter [-h] [--version] [--config-dir] [--data-dir] [--runtime-dir] [--paths] [--json] [--debug] [subcommand]\n")
        return 1
    sub = a[0]
    if sub == "kernelspec" and len(a) > 1 and a[1] == "list":
        home = ctx.env.get("HOME", "")
        rows = [("python3", f"{root}/share/jupyter/kernels/python3" if kind != "system" else "/usr/share/jupyter/kernels/python3")]
        try:
            node = ctx.fs.lookup(f"{home}/.local/share/jupyter/kernels")
            rows = [(n, f"{home}/.local/share/jupyter/kernels/{n}") for n in ctx.fs.listdir(node)] + rows
        except FSError:
            pass
        w = max(len(r[0]) for r in rows) + 4
        ctx.w("Available kernels:\n" + "".join(f"  {n.ljust(w)}{p}\n" for n, p in rows))
        return 0
    if sub in ("lab", "notebook", "server"):
        if norm("jupyterlab") not in installed(ctx, root, kind):
            ctx.e(f"Jupyter command `jupyter-{sub}` not found.\n")
            return 1
        port = 8888
        for x in a:
            m = re.match(r"--port[= ]?(\d+)", x)
            if m:
                port = int(m.group(1))
        if "--port" in a:
            port = int(a[a.index("--port") + 1])
        token = "".join(ctx.world.rng.choice("0123456789abcdef") for _ in range(48))
        when = strftime("%Y-%m-%d %H:%M:%S", local_dt(ctx.now, ctx.tz)) + ".123"

        def handler(method, path, headers, body):
            from .net import HttpResult
            if path.startswith("/api/status"):
                return HttpResult(200, {"Content-Type": "application/json"}, b'{"connections": 0, "kernels": 0, "started": "2026-10-20T09:41:07Z"}\n')
            return HttpResult(200, {"Content-Type": "text/html"}, b"<!DOCTYPE html>\n<html><head><title>JupyterLab</title></head><body>JupyterLab</body></html>\n")
        banner = "".join(f"[I {when} ServerApp] {x}\n" for x in [
            "jupyter_lsp | extension was successfully linked.", "jupyter_server_terminals | extension was successfully linked.",
            "jupyterlab | extension was successfully linked.", f"Serving notebooks from local directory: {ctx.cwd}",
            "Jupyter Server 2.17.0 is running at:", f"http://localhost:{port}/lab?token={token}", f"    http://127.0.0.1:{port}/lab?token={token}",
            "Use Control-C to stop this server and shut down all kernels (twice to skip confirmation)."])
        banner += (f"[C {when} ServerApp] \n    \n    To access the server, open this file in a browser:\n"
                   f"        file://{ctx.env.get('HOME', '')}/.local/share/jupyter/runtime/jpserver-{ctx.proc.pid}-open.html\n"
                   f"    Or copy and paste one of these URLs:\n        http://localhost:{port}/lab?token={token}\n"
                   f"        http://127.0.0.1:{port}/lab?token={token}\n")
        return (yield from serve(ctx, port, banner, "jupyter", handler=handler, stderr_banner=True,
                                 shutdown=f"\n[C {when} ServerApp] received signal 2, stopping\n[I {when} ServerApp] Shutting down 5 extensions\n"))
    if sub == "--version":
        ctx.w("jupyter_core     : 5.8.1\njupyterlab       : 4.4.9\n")
        return 0
    ctx.e(f"Jupyter command `jupyter-{sub}` not found.\n")
    return 1


@util("tensorboard")
def u_tensorboard(ctx):
    root, kind = env_of(ctx)
    a = ctx.argv[1:]
    port = 6006
    logdir = None
    for k, x in enumerate(a):
        if x.startswith("--port"):
            port = int(x.split("=", 1)[1] if "=" in x else a[k + 1])
        if x.startswith("--logdir"):
            logdir = x.split("=", 1)[1] if "=" in x else a[k + 1]
    if logdir is None:
        ctx.e("Error: A logdir or db must be specified. For example `tensorboard --logdir mylogdir` or `tensorboard --db sqlite:~/.tensorboard.db`. Run `tensorboard --helpfull` for details and examples.\n")
        return 1

    def handler(method, path, headers, body):
        from .net import HttpResult
        return HttpResult(200, {"Content-Type": "text/html"}, b"<!doctype html><title>TensorBoard</title>\n")
    banner = ("TensorFlow installation not found - running with reduced feature set.\n"
              "Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all\n"
              f"TensorBoard 2.20.0 at http://localhost:{port}/ (Press CTRL+C to quit)\n")
    return (yield from serve(ctx, port, banner, "tensorboard", handler=handler, stderr_banner=True, shutdown="\n"))


@util("torchrun")
def u_torchrun(ctx):
    root, kind = env_of(ctx)
    a = ctx.argv[1:]
    nproc = 1
    i = 0
    while i < len(a) and a[i].startswith("--"):
        x = a[i]
        k, _, v = x.partition("=")
        if k == "--nproc_per_node" or k == "--nproc-per-node":
            nproc = int(v or a[i + 1])
            if not v:
                i += 1
        elif not v and k in ("--nnodes", "--node_rank", "--master_port", "--master_addr", "--rdzv_backend", "--rdzv_endpoint"):
            i += 1
        i += 1
    if i >= len(a):
        ctx.e("torchrun: error: the following arguments are required: training_script, training_script_args\n")
        return 2
    script, sargs = a[i], a[i + 1:]
    if norm("torch") not in installed(ctx, root, kind):
        ctx.e("Traceback (most recent call last):\n  File \"" + (root if kind != "system" else "/usr/local") +
              "/bin/torchrun\", line 5, in <module>\n    from torch.distributed.run import main\nModuleNotFoundError: No module named 'torch'\n")
        return 1
    try:
        code = bytes(ctx.lookup(script).data).decode()
    except FSError:
        ctx.e(f"[Errno 2] No such file or directory: '{script}'\n")
        return 1
    when = strftime("%m%d %H:%M:%S", local_dt(ctx.now, ctx.tz)) + ".123000"
    if nproc > 1:
        star = "*" * 41
        msg = ("Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being "
               "overloaded, please further tune the variable for optimal performance in your application as needed. ")
        for line in ("", star, msg, star):
            ctx.e(f"W{when} {ctx.proc.pid} torch/distributed/run.py:774] {line}\n")
    status = 0
    start = ctx.now
    all_events = []
    longest = 0.0
    ctx.gpu_claim = []
    for rank in range(nproc):
        env = dict(ctx.env, RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(nproc), LOCAL_WORLD_SIZE=str(nproc),
                   MASTER_ADDR="127.0.0.1", MASTER_PORT="29500", GROUP_RANK="0", ROLE_RANK=str(rank), OMP_NUM_THREADS="1")
        st, events, off = run_python(ctx, code, script, [script] + sargs, env, root, kind)
        status = status or st
        all_events += events
        longest = max(longest, off)
    all_events.sort(key=lambda e: e[0])
    return (yield from _play(ctx, all_events, start, longest, status))


# ---------------------------------------------------------------------------- uv

@util("uv")
def u_uv(ctx):
    a = ctx.argv[1:]
    if any(x in ("-q", "--quiet") for x in a):
        a = [x for x in a if x not in ("-q", "--quiet")]
        loud = ctx.e
        ctx.e = lambda text: loud(text) if text.startswith(("error", "  ×", "warning")) else None
    if not a or a[0] in ("-h", "--help"):
        ctx.w("An extremely fast Python package manager.\n\nUsage: uv [OPTIONS] <COMMAND>\n\nCommands:\n  run      Run a command or script\n"
              "  init     Create a new project\n  add      Add dependencies to the project\n  sync     Update the project's environment\n"
              "  lock     Update the project's lockfile\n  venv     Create a virtual environment\n  pip      Manage Python packages with a pip-compatible interface\n"
              "  python   Manage Python versions and installations\n")
        return 0
    cmd, rest = a[0], a[1:]
    if cmd in ("--version", "-V", "version"):
        ctx.w("uv 0.9.4\n")
        return 0
    venv = ctx.abspath(".venv")
    if cmd == "venv":
        target = next((x for x in rest if not x.startswith("-")), ".venv")
        ctx.e(f"Using CPython {PY_VERSION} interpreter at: /usr/bin/python3\nCreating virtual environment at: {target}\n"
              f"Activate with: source {target}/bin/activate\n")
        make_venv(ctx, target)
        remove_installed(ctx, ctx.abspath(target), "venv", "pip")
        return 0
    if cmd == "pip":
        if not rest:
            ctx.e("error: A subcommand is required\n")
            return 2
        root = ctx.env.get("VIRTUAL_ENV") or (venv if ctx.fs.exists(venv + "/pyvenv.cfg") else None)
        if "--system" in rest:
            root = None
        if root is None:
            ctx.e("error: No virtual environment found; run `uv venv` to create an environment, or pass `--system` to install into a non-virtual environment\n")
            return 2
        sub = rest[0]
        if sub in ("install", "sync"):
            specs = []
            k = 1
            while k < len(rest):
                if rest[k] in ("-r", "--requirement") and k + 1 < len(rest):
                    try:
                        specs += [r for r in parse_requirements(ctx, rest[k + 1]) if not isinstance(r, tuple)]
                    except PipError as e:
                        ctx.e(f"error: File not found: `{rest[k + 1]}`\n")
                        return 2
                    k += 2
                    continue
                if not rest[k].startswith("-"):
                    specs.append(rest[k])
                k += 1
            have = installed(ctx, root, "venv")
            try:
                plan = resolve(ctx, specs, have, "-U" in rest or "--upgrade" in rest)
            except PipError as e:
                name = re.search(r"requirement (\S+)", str(e))
                ctx.e(f"  × No solution found when resolving dependencies:\n  ╰─▶ Because {name.group(1) if name else 'it'} was not found in the package registry and you require {name.group(1) if name else 'it'}, we can conclude that your requirements are unsatisfiable.\n")
                return 1
            new = [(it["key"], it["version"]) for it in plan if it["status"] != "satisfied"]
            ctx.e(f"Resolved {len(plan)} package{'s' if len(plan) != 1 else ''} in 18ms\n")
            if new:
                ctx.e(f"Prepared {len(new)} package{'s' if len(new) != 1 else ''} in 210ms\n"
                      f"Installed {len(new)} package{'s' if len(new) != 1 else ''} in 9ms\n")
                for n, v in sorted(new):
                    set_installed(ctx, root, "venv", INDEX_NAME.get(n, n), v)
                    ctx.e(f" + {n}=={v}\n")
            else:
                ctx.e(f"Audited {len(plan)} package{'s' if len(plan) != 1 else ''} in 0.21ms\n")
            return 0
        sub_ctx_args = rest
        return (yield from pip_main(ctx, sub_ctx_args, root, "venv", prog="uv pip"))
    if cmd == "init":
        name = next((x for x in rest if not x.startswith("-")), None)
        base = ctx.abspath(name) if name else ctx.cwd
        pname = base.rstrip("/").rsplit("/", 1)[-1]
        if ctx.fs.exists(base + "/pyproject.toml"):
            ctx.e(f"error: Project is already initialized in `{base}` (`pyproject.toml` file exists)\n")
            return 2
        fs = ctx.fs
        fs.mkdir(base, "/", 0o755, ctx.uid, ctx.proc.gid, parents=True)
        fs.write_file(base + "/pyproject.toml", f'[project]\nname = "{pname}"\nversion = "0.1.0"\ndescription = "Add your description here"\n'
                                                f'readme = "README.md"\nrequires-python = ">=3.12"\ndependencies = []\n', "/", 0o644, ctx.uid, ctx.proc.gid)
        fs.write_file(base + "/.python-version", "3.12\n", "/", 0o644, ctx.uid, ctx.proc.gid)
        fs.write_file(base + "/README.md", "", "/", 0o644, ctx.uid, ctx.proc.gid)
        fs.write_file(base + "/main.py", f'def main():\n    print("Hello from {pname}!")\n\n\nif __name__ == "__main__":\n    main()\n',
                      "/", 0o644, ctx.uid, ctx.proc.gid)
        if not fs.exists(base + "/.gitignore"):
            fs.write_file(base + "/.gitignore", "# Python-generated files\n__pycache__/\n*.py[oc]\nbuild/\ndist/\nwheels/\n*.egg-info\n\n# Virtual environments\n.venv\n",
                          "/", 0o644, ctx.uid, ctx.proc.gid)
        ctx.e(f"Initialized project `{pname}`" + (f" at `{base}`" if name else "") + "\n")
        return 0
    if cmd in ("add", "remove", "sync", "lock", "run"):
        proj = ctx.cwd
        pp = proj + "/pyproject.toml"
        if not ctx.fs.exists(pp):
            ctx.e("error: No `project` table found in: `" + pp + "`\n" if False else
                  f"error: No `pyproject.toml` found in current directory or any parent directory\n")
            return 2
        text = bytes(ctx.fs.lookup(pp).data).decode()
        m = re.search(r"dependencies\s*=\s*\[(.*?)\]", text, re.S)
        deps = re.findall(r'"([^"]+)"', m.group(1)) if m else []
        if cmd in ("add", "remove"):
            names = [x for x in rest if not x.startswith("-")]
            if cmd == "add":
                for n in names:
                    mm = SPEC.match(n)
                    if mm is None or norm(mm.group(1)) not in INDEX:
                        ctx.e(f"  × No solution found when resolving dependencies:\n  ╰─▶ Because {n} was not found in the package registry and your project depends on {n}, we can conclude that your project's requirements are unsatisfiable.\n")
                        return 1
                    base_name = norm(mm.group(1))
                    deps = [d for d in deps if norm(SPEC.match(d).group(1)) != base_name]
                    spec = mm.group(3) or f">={INDEX[base_name][0]}"
                    deps.append(f"{mm.group(1)}{spec}")
            else:
                deps = [d for d in deps if norm(SPEC.match(d).group(1)) not in [norm(x) for x in names]]
            body = "dependencies = [\n" + "".join(f'    "{d}",\n' for d in deps) + "]" if deps else "dependencies = []"
            text = re.sub(r"dependencies\s*=\s*\[.*?\]", body, text, count=1, flags=re.S) if m else text + body + "\n"
            ctx.fs.write_file(pp, text, "/", 0o644, ctx.uid, ctx.proc.gid)
        # lock + sync
        created = False
        if not ctx.fs.exists(venv + "/pyvenv.cfg") and cmd != "lock":
            ctx.e(f"Using CPython {PY_VERSION} interpreter at: /usr/bin/python3\nCreating virtual environment at: .venv\n")
            make_venv(ctx, ".venv")
            remove_installed(ctx, venv, "venv", "pip")
            created = True
        try:
            plan = resolve(ctx, deps, {}, False)
        except PipError as e:
            ctx.e(str(e) + "\n")
            return 1
        pname = re.search(r'name\s*=\s*"([^"]+)"', text)
        pname = pname.group(1) if pname else "project"
        lock = "version = 1\nrevision = 3\nrequires-python = \">=3.12\"\n\n" + "".join(
            f'[[package]]\nname = "{it["key"]}"\nversion = "{it["version"]}"\nsource = {{ registry = "https://pypi.org/simple" }}\n\n'
            for it in sorted(plan, key=lambda it: it["key"]))
        ctx.fs.write_file(proj + "/uv.lock", lock, "/", 0o644, ctx.uid, ctx.proc.gid)
        ctx.e(f"Resolved {len(plan) + 1} packages in 12ms\n")
        if cmd == "lock":
            return 0
        have = installed(ctx, venv, "venv")
        want = {it["key"]: it["version"] for it in plan}
        new = [(n, v) for n, v in want.items() if have.get(n, (None, None))[1] != v]
        gone = [n for n in have if n not in want and n != "pip"]
        if new or gone:
            if new:
                ctx.e(f"Prepared {len(new)} package{'s' if len(new) != 1 else ''} in 350ms\n")
            if gone:
                ctx.e(f"Uninstalled {len(gone)} package{'s' if len(gone) != 1 else ''} in 3ms\n")
            if new:
                ctx.e(f"Installed {len(new)} package{'s' if len(new) != 1 else ''} in 11ms\n")
            for n in sorted(gone):
                remove_installed(ctx, venv, "venv", n)
                ctx.e(f" - {n}=={have[n][1]}\n")
            for n, v in sorted(new):
                set_installed(ctx, venv, "venv", INDEX_NAME.get(n, n), v)
                ctx.e(f" + {n}=={v}\n")
        elif cmd == "sync":
            ctx.e(f"Audited {len(want)} package{'s' if len(want) != 1 else ''} in 0.15ms\n")
        _ = created
        if cmd == "run":
            if not rest:
                ctx.e("error: the following required arguments were not provided:\n  <COMMAND>\n")
                return 2
            args = [x for x in rest]
            while args and args[0].startswith("-"):
                args = args[2:] if args[0] in ("--with", "--python", "-p") else args[1:]
            if not args:
                ctx.e("error: the following required arguments were not provided:\n  <COMMAND>\n")
                return 2
            env = dict(ctx.env, VIRTUAL_ENV=venv, PATH=f"{venv}/bin:" + ctx.env.get("PATH", ""))
            env.pop("PYTHONHOME", None)
            argv = [f"{venv}/bin/python"] + args if args[0].endswith(".py") else args
            from ..exec import find_in_path
            if find_in_path(ctx.fs, argv[0], env["PATH"], ctx.cwd) is None:
                ctx.e(f"error: Failed to spawn: `{argv[0]}`\n  Caused by: No such file or directory (os error 2)\n")
                return 2
            yield from ctx.flush()
            st = yield from ctx.run(argv, env=env)
            return st
        return 0
    if cmd == "python":
        sub = rest[0] if rest else ""
        if sub == "install":
        	vers = [x for x in rest[1:] if not x.startswith("-")] or ["3.13"]
        	for v in vers:
        	    full = {"3.12": "3.12.12", "3.13": "3.13.9", "3.11": "3.11.14", "3.14": "3.14.0", "3.10": "3.10.19"}.get(v, v)
        	    ctx.e(f"Installed Python {full} in 1.21s\n + cpython-{full}-linux-x86_64-gnu\n")
        	    ctx.fs.mkdir(ctx.env.get("HOME", "") + f"/.local/share/uv/python/cpython-{full}-linux-x86_64-gnu/bin", "/", 0o755,
        	                 ctx.uid, ctx.proc.gid, parents=True)
        	return 0
        if sub == "list":
            ctx.w(f"cpython-3.12.3-linux-x86_64-gnu     /usr/bin/python3.12\n")
            return 0
        ctx.e("error: unrecognized subcommand\n")
        return 2
    ctx.e(f"error: unrecognized subcommand '{cmd}'\n")
    return 2


# ---------------------------------------------------------------------------- conda

CONDA_PY = {"3.13": "3.13.7", "3.12": "3.12.11", "3.11": "3.11.13", "3.10": "3.10.18", "3.9": "3.9.23"}


def conda_hook(base):
    return f"""export CONDA_EXE='{base}/bin/conda'
export _CE_M=''
export _CE_CONDA=''
export CONDA_PYTHON_EXE='{base}/bin/python'

__conda_exe() (
    "$CONDA_EXE" $_CE_M $_CE_CONDA "$@"
)

__conda_hashr() {{
    \\hash -r
}}

__conda_activate() {{
    \\local ask_conda
    ask_conda="$(PS1="${{PS1:-}}" __conda_exe shell.posix "$@")" || \\return
    \\eval "$ask_conda"
    __conda_hashr
}}

__conda_reactivate() {{
    \\local ask_conda
    ask_conda="$(PS1="${{PS1:-}}" __conda_exe shell.posix reactivate)" || \\return
    \\eval "$ask_conda"
    __conda_hashr
}}

conda() {{
    \\local cmd="${{1-__missing__}}"
    case "$cmd" in
        activate|deactivate)
            __conda_activate "$@"
            ;;
        install|update|upgrade|remove|uninstall)
            __conda_exe "$@" || \\return
            __conda_reactivate
            ;;
        *)
            __conda_exe "$@"
            ;;
    esac
}}

if [ -z "${{CONDA_SHLVL+x}}" ]; then
    \\export CONDA_SHLVL=0
    if [ -n "${{_CE_CONDA:+x}}" ] && [ -n "${{WINDIR+x}}" ]; then
        PATH="$(\\dirname "$CONDA_EXE")/condabin${{PATH:+":${{PATH}}"}}"
    else
        PATH="$(\\dirname "$(\\dirname "$CONDA_EXE")")/condabin${{PATH:+":${{PATH}}"}}"
    fi
    \\export PATH
    if [ -z "${{PS1+x}}" ]; then
        PS1=
    fi
fi

conda activate base
"""


CONDA_INIT = """
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('@@BASE@@/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "@@BASE@@/etc/profile.d/conda.sh" ]; then
        . "@@BASE@@/etc/profile.d/conda.sh"
    else
        export PATH="@@BASE@@/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
"""


def make_conda_env(fs, path, pyver, uid, gid):
    for d in ("conda-meta", "bin", SITE, "lib"):
        fs.mkdir(f"{path}/{d}", "/", 0o755, uid, gid, parents=True)
    fs.write_file(f"{path}/conda-meta/history", f"==> 2026-10-01 10:00:00 <==\n# cmd: conda create\n", "/", 0o644, uid, gid)
    fs.write_file(f"{path}/conda-meta/python-{pyver}-h5148396_0.json", f'{{"name": "python", "version": "{pyver}"}}\n',
                  "/", 0o644, uid, gid)
    for tool, prog in (("python3.12", "python3.12"), ("pip", "pip")):
        n = fs.write_file(f"{path}/bin/{tool}", stub_bytes(prog), "/", 0o755, uid, gid)
        n.mode = 0o755
    for alias in ("python", "python3"):
        if not fs.exists(f"{path}/bin/{alias}", follow=False):
            fs.symlink("python3.12", f"{path}/bin/{alias}")


def install_conda(host, user="ada", init=True):
    """A Miniconda install in ~/miniconda3 (with `conda init bash` run, unless init=False)."""
    fs = host.fs
    uid = host.uid_of(user)
    gid = host.users[uid].gid
    home = host.users[uid].home
    base = f"{home}/miniconda3"
    make_conda_env(fs, base, "3.13.7", uid, gid)
    for d in ("envs", "pkgs", "condabin", "etc/profile.d"):
        fs.mkdir(f"{base}/{d}", "/", 0o755, uid, gid, parents=True)
    for p in (f"{base}/bin/conda", f"{base}/condabin/conda"):
        n = fs.write_file(p, stub_bytes("conda"), "/", 0o755, uid, gid)
        n.mode = 0o755
    fs.write_file(f"{base}/etc/profile.d/conda.sh", conda_hook(base).replace("\nconda activate base\n", "\n"), "/", 0o644, uid, gid)
    fs.write_file(f"{base}/conda-meta/conda-25.7.0-py313h06a4308_0.json", '{"name": "conda", "version": "25.7.0"}\n',
                  "/", 0o644, uid, gid)
    if init:
        rc = f"{home}/.bashrc"
        old = bytes(fs.lookup(rc).data).decode() if fs.exists(rc) else ""
        fs.write_file(rc, old + CONDA_INIT.replace("@@BASE@@", base), "/", 0o644, uid, gid)


def conda_pyver(fs, path):
    try:
        for name in fs.listdir(fs.lookup(path + "/conda-meta")):
            m = re.match(r"python-(\d+\.\d+\.\d+)-", name)
            if m:
                return m.group(1)
    except FSError:
        pass
    return None


def read_env_yaml(text):
    """name, channels and dependencies (conda specs, plus pip requirements) from an environment.yml."""
    name, channels, deps, pip = None, [], [], []
    section = None
    in_pip = False
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        body = line.strip()
        if indent == 0:
            in_pip = False
            key, _, val = body.partition(":")
            section = key.strip()
            if section == "name":
                name = val.strip().strip("'\"")
            continue
        if not body.startswith("- "):
            continue
        item = body[2:].strip().strip("'\"")
        if section == "channels":
            channels.append(item)
        elif section == "dependencies":
            if item == "pip:":
                in_pip = True
                pip_indent = indent
                continue
            if in_pip and indent > pip_indent:
                pip.append(item)
            else:
                in_pip = False
                deps.append(item)
    return name, channels, deps, pip


def _conda_specs(specs):
    """'numpy=2.2', 'python=3.12', 'scikit-learn' -> [(name, version-or-None)]."""
    out = []
    for s in specs:
        m = re.match(r"^([A-Za-z0-9_.-]+)(?:\s*(==?|>=|<=)\s*([^=\s]+))?", s)
        if m:
            out.append((m.group(1), m.group(3)))
    return out


def _conda_pick(name, ver):
    if ver is None:
        return pick_version(name, "")
    v = pick_version(name, f"=={ver}")
    if v is None or isinstance(v, tuple):
        info = INDEX.get(norm(name))
        if info:
            cands = [x for x in sorted([info[0]] + info[1], key=vkey) if x == ver or x.startswith(ver.rstrip("*").rstrip(".") + ".")]
            return cands[-1] if cands else None
        return None
    return v


@util("conda")
def u_conda(ctx):
    a = ctx.argv[1:]
    fs = ctx.fs
    exe = getattr(ctx, "path", "") or ""
    base = exe.rsplit("/", 2)[0] if exe.endswith(("/bin/conda", "/condabin/conda")) else ctx.env.get("HOME", "") + "/miniconda3"
    envs_dir = base + "/envs"
    if not a or a[0] in ("-h", "--help"):
        ctx.w("usage: conda [-h] [-v] [--no-plugins] [-V] COMMAND ...\n\nconda is a tool for managing and deploying applications, "
              "environments and packages.\n\npositional arguments:\n  COMMAND\n    activate     Activate a conda environment.\n"
              "    create       Create a new conda environment from a list of specified packages.\n"
              "    deactivate   Deactivate the current active conda environment.\n    env          Create and manage conda environments.\n"
              "    info         Display information about current conda install.\n    install      Install a list of packages into a specified conda environment.\n"
              "    list         List installed packages in a conda environment.\n    remove       Remove a list of packages from a specified conda environment.\n")
        return 0
    cmd, rest = a[0], a[1:]

    def env_path(name):
        return base if name == "base" else (name if name.startswith("/") else f"{envs_dir}/{name}")

    def env_name(path):
        return "base" if path == base else path.rstrip("/").rsplit("/", 1)[-1]

    def current():
        return ctx.env.get("CONDA_PREFIX") or base

    def opt(*names):
        for k, x in enumerate(rest):
            for n in names:
                if x == n and k + 1 < len(rest):
                    return rest[k + 1]
                if n.startswith("--") and x.startswith(n + "="):
                    return x.split("=", 1)[1]
        return None

    def positional():
        out, k = [], 0
        while k < len(rest):
            x = rest[k]
            if x in ("-n", "--name", "-p", "--prefix", "-c", "--channel", "-f", "--file"):
                k += 2
                continue
            if not x.startswith("-"):
                out.append(x)
            k += 1
        return out

    def confirm():
        if any(x in ("-y", "--yes") for x in rest):
            return True
        ctx.w("\nProceed ([y]/n)? ")
        yield from ctx.flush()
        from ..fileio import readline
        ans = (yield from readline(ctx.k, ctx.proc.fds.get(0))).decode(errors="replace").strip().lower()
        ctx.w("\n")
        if ans not in ("", "y", "yes"):
            ctx.w("\nCondaSystemExit: Exiting.\n\n")
            return False
        return True

    def channels_block(chans):
        return "Channels:\n" + "".join(f" - {c}\n" for c in (chans or ["defaults"])) + "Platform: linux-64\n" \
            "Collecting package metadata (repodata.json): done\nSolving environment: done\n"

    def plan_block(path, specs, new):
        width = max([len(n) for n, _ in new] + [10]) + 2
        text = f"\n## Package Plan ##\n\n  environment location: {path}\n\n"
        if specs:
            text += "  added / updated specs:\n" + "".join(f"    - {s}\n" for s in specs) + "\n"
        if new:
            text += "\nThe following NEW packages will be INSTALLED:\n\n" + "".join(
                f"  {n:<{width}} pkgs/main/linux-64::{n}-{v}-py312_0\n" for n, v in new) + "\n"
        return text
    if cmd in ("--version", "-V"):
        ctx.w("conda 25.7.0\n")
        return 0
    if cmd in ("shell.bash", "shell.posix"):
        if not rest:
            return 2
        sub = rest[0]
        if sub == "hook":
            ctx.w(conda_hook(base))
            return 0
        q = lambda v: "'" + v.replace("'", "'\\''") + "'"
        path_parts = [p for p in ctx.env.get("PATH", "").split(":") if p]
        old = ctx.env.get("CONDA_PREFIX")
        if old:
            path_parts = [p for p in path_parts if p != old + "/bin"]
        ps1 = ctx.env.get("PS1", "")
        mod = ctx.env.get("CONDA_PROMPT_MODIFIER", "")
        if mod and ps1.startswith(mod):
            ps1 = ps1[len(mod):]
        try:
            lvl = int(ctx.env.get("CONDA_SHLVL", "0") or 0)
        except ValueError:
            lvl = 0

        def activate_lines(path, new_lvl, extra=""):
            cb = base + "/condabin"
            parts = list(path_parts)
            k = parts.index(cb) + 1 if cb in parts else 0
            parts.insert(k, path + "/bin")
            shown = env_name(path)
            return (f"PS1={q(f'({shown}) ' + ps1)}\n{extra}export PATH={q(':'.join(parts))}\nexport CONDA_PREFIX={q(path)}\n"
                    f"export CONDA_SHLVL='{new_lvl}'\nexport CONDA_DEFAULT_ENV={q(shown)}\n"
                    f"export CONDA_PROMPT_MODIFIER={q(f'({shown}) ')}\n")
        if sub in ("activate", "reactivate"):
            name = (rest[1] if len(rest) > 1 else "base") if sub == "activate" else (old or base)
            path = env_path(name)
            if not fs.exists(path + "/conda-meta"):
                ctx.e(f"\nEnvironmentNameNotFound: Could not find conda environment: {name}\n"
                      "You can list all discoverable environments with `conda info --envs`.\n\n")
                return 1
            if sub == "reactivate" or path == old:
                ctx.w(activate_lines(path, max(lvl, 1)))
                return 0
            extra = f"export CONDA_PREFIX_{lvl}={q(old)}\n" if old and lvl >= 1 else ""
            ctx.w(activate_lines(path, lvl + 1, extra))
            return 0
        if sub == "deactivate":
            if not old or lvl <= 0:
                return 0
            if lvl == 1:
                ctx.w(f"PS1={q(ps1)}\nexport PATH={q(':'.join(path_parts))}\nunset CONDA_PREFIX\nunset CONDA_DEFAULT_ENV\n"
                      f"unset CONDA_PROMPT_MODIFIER\nexport CONDA_SHLVL='0'\n")
                return 0
            prev = ctx.env.get(f"CONDA_PREFIX_{lvl - 1}") or base
            ctx.w(f"unset CONDA_PREFIX_{lvl - 1}\n" + activate_lines(prev, lvl - 1))
            return 0
        return 2
    if cmd in ("activate", "deactivate"):
        ctx.e("\nCondaError: Run 'conda init' before 'conda " + cmd + "'\n\n")
        return 1
    if cmd == "init":
        rc = ctx.env.get("HOME", "") + "/.bashrc"
        old = bytes(fs.lookup(rc).data).decode() if fs.exists(rc) else ""
        if ">>> conda initialize >>>" in old:
            ctx.w(f"no change     {base}/condabin/conda\nno change     {rc}\nNo action taken.\n")
            return 0
        fs.write_file(rc, old + CONDA_INIT.replace("@@BASE@@", base), "/", 0o644, ctx.uid, ctx.proc.gid)
        ctx.w(f"no change     {base}/condabin/conda\nno change     {base}/bin/conda\nmodified      {rc}\n\n"
              "==> For changes to take effect, close and re-open your current shell. <==\n\n")
        return 0
    if cmd == "env" and rest[:1] == ["create"] or cmd == "create":
        from_file = cmd == "env"
        if from_file:
            rest = rest[1:]
        name = opt("-n", "--name")
        prefix = opt("-p", "--prefix")
        chans = [rest[k + 1] for k, x in enumerate(rest[:-1]) if x in ("-c", "--channel")]
        pip_reqs = []
        if from_file:
            file = opt("-f", "--file") or "environment.yml"
            try:
                text = bytes(ctx.lookup(file).data).decode()
            except FSError:
                ctx.e(f"\nEnvironmentFileNotFound: '{ctx.abspath(file)}' file not found\n\n")
                return 1
            fname, fchans, specs, pip_reqs = read_env_yaml(text)
            name = name or fname
            chans = chans or fchans
        else:
            specs = positional()
        if not name and not prefix:
            ctx.e("\nCondaValueError: one of the arguments -n/--name -p/--prefix is required\n\n" if not from_file else
                  "\nCondaValueError: Unable to determine environment\n\nPlease re-run this command with one of the following options:\n\n"
                  "* Provide an environment name via --name or -n\n* Re-run this command inside an activated conda environment.\n\n")
            return 1
        path = ctx.abspath(prefix) if prefix else env_path(name)
        if fs.exists(path):
            if from_file:
                ctx.e(f"\nCondaValueError: prefix already exists: {path}\n\n")
                return 1
            ctx.w(f"WARNING: A conda environment already exists at '{path}'\n\nRemove existing environment?\nThis will remove ALL directories contained within this specified prefix directory.\n")
            ok = yield from confirm()
            if not ok:
                return 1
            _rm_tree(fs, path)
        pyver = None
        new = []
        for n, v in _conda_specs(specs):
            if n == "python":
                pyver = CONDA_PY.get(v or "3.13", v if v and v.count(".") == 2 else None) or "3.13.7"
                continue
            if n == "pip":
                continue
            vv = _conda_pick(n, v)
            if vv is None:
                ctx.w(channels_block(chans).replace("Solving environment: done", "Solving environment: failed"))
                ctx.e(f"\nPackagesNotFoundError: The following packages are not available from current channels:\n\n  - {n}\n\n"
                      f"Current channels:\n\n  - defaults\n\nTo search for alternate channels that may provide the conda package you're\n"
                      f"looking for, navigate to\n\n    https://anaconda.org\n\nand use the search bar at the top of the page.\n\n\n")
                return 1
            new.append((norm(n), vv))
        listed = ([("python", pyver)] if pyver else []) + new
        ctx.w(channels_block(chans) + plan_block(path, specs, listed + ([("pip", "25.2")] if pyver else [])))
        if not from_file:                # `conda env create` never asks
            ok = yield from confirm()
            if not ok:
                return 1
        ctx.w("\nDownloading and Extracting Packages:\n\nPreparing transaction: done\nVerifying transaction: done\nExecuting transaction: done\n")
        if pyver:
            make_conda_env(fs, path, pyver, ctx.uid, ctx.proc.gid)
            set_installed(ctx, path, "conda", "pip", "25.2")
        else:
            fs.mkdir(path + "/conda-meta", "/", 0o755, ctx.uid, ctx.proc.gid, parents=True)
        for n, v in new:
            set_installed(ctx, path, "conda", INDEX_NAME.get(n, n), v)
        if pip_reqs:
            ctx.w("Installing pip dependencies: | Ran pip subprocess with arguments:\n"
                  f"['{path}/bin/python', '-m', 'pip', 'install', '-U', '-r', '{ctx.cwd}/condaenv.requirements.txt', '--exists-action=b']\n")
            try:
                items = resolve(ctx, pip_reqs, installed(ctx, path, "conda"))
            except PipError as e:
                ctx.e(str(e) + "\n")
                return 1
            for it in install_order(items):
                set_installed(ctx, path, "conda", INDEX_NAME.get(it["key"], it["key"]), it["version"])
            ctx.w("Successfully installed " + " ".join(sorted(f"{it['key']}-{it['version']}" for it in items if it["status"] != "satisfied"))
                  + "\n\ndone\n")
        shown = env_name(path) if not prefix else path
        ctx.w(f"#\n# To activate this environment, use\n#\n#     $ conda activate {shown}\n#\n"
              f"# To deactivate an active environment, use\n#\n#     $ conda deactivate\n\n")
        return 0
    if cmd in ("install", "update", "upgrade"):
        target = env_path(opt("-n", "--name")) if opt("-n", "--name") else (ctx.abspath(opt("-p", "--prefix")) if opt("-p", "--prefix") else current())
        if not fs.exists(target + "/conda-meta"):
            ctx.e(f"\nEnvironmentLocationNotFound: Not a conda environment: {target}\n\n")
            return 1
        chans = [rest[k + 1] for k, x in enumerate(rest[:-1]) if x in ("-c", "--channel")]
        specs = positional()
        if not specs:
            ctx.e("\nCondaValueError: too few arguments, must supply command line package specs or --file\n\n")
            return 1
        have = installed(ctx, target, "conda")
        new = []
        for n, v in _conda_specs(specs):
            vv = _conda_pick(n, v)
            if vv is None:
                ctx.w(channels_block(chans + ["defaults"] if chans else []).replace("Solving environment: done", "Solving environment: failed"))
                ctx.e(f"\nPackagesNotFoundError: The following packages are not available from current channels:\n\n  - {n}\n\n")
                return 1
            if have.get(norm(n), (None, None))[1] != vv:
                new.append((norm(n), vv))
                for dep in INDEX.get(norm(n), ("", [], []))[2]:
                    if norm(dep) not in have and norm(dep) not in dict(new):
                        new.append((norm(dep), INDEX[norm(dep)][0]))
        if not new:
            ctx.w(channels_block(chans + ["defaults"] if chans else []) + "\n# All requested packages already installed.\n\n")
            return 0
        ctx.w(channels_block(chans + ["defaults"] if chans else []) + plan_block(target, specs, sorted(new)))
        ok = yield from confirm()
        if not ok:
            return 1
        ctx.w("\nDownloading and Extracting Packages:\n\nPreparing transaction: done\nVerifying transaction: done\nExecuting transaction: done\n")
        for n, v in new:
            set_installed(ctx, target, "conda", INDEX_NAME.get(n, n), v)
        return 0
    if cmd == "env" and rest[:1] == ["export"]:
        rest = rest[1:]
        target = env_path(opt("-n", "--name")) if opt("-n", "--name") else current()
        if not fs.exists(target + "/conda-meta"):
            ctx.e(f"\nEnvironmentLocationNotFound: Not a conda environment: {target}\n\n")
            return 1
        pyver = conda_pyver(fs, target)
        hist = "--from-history" in rest
        lines = [f"name: {env_name(target)}", "channels:", "  - defaults", "dependencies:"]
        pkgs = installed(ctx, target, "conda")
        if pyver:
            lines.append(f"  - python={'.'.join(pyver.split('.')[:2]) if hist else pyver + '=h5148396_0'}")
        for key, (disp, ver) in sorted(pkgs.items()):
            if key == "pip" and hist:
                continue
            lines.append(f"  - {key}" if hist else f"  - {key}={ver}=py312_0")
        lines.append(f"prefix: {target}")
        ctx.w("\n".join(lines) + "\n")
        return 0
    if cmd == "env" and rest[:1] == ["list"] or cmd == "info" and any(x in ("-e", "--envs") for x in rest):
        cur = current() if ctx.env.get("CONDA_PREFIX") else None
        rows = [("base", base)]
        try:
            rows += [(n, f"{envs_dir}/{n}") for n in fs.listdir(fs.lookup(envs_dir))]
        except FSError:
            pass
        ctx.w("# conda environments:\n#\n" + "".join(f"{n:<20} {'*' if p == cur else ' '}  {p}\n" for n, p in rows) + "\n")
        return 0
    if cmd == "info":
        cur = ctx.env.get("CONDA_PREFIX")
        ctx.w(f"\n     active environment : {env_name(cur) if cur else 'None'}\n" + (f"    active env location : {cur}\n" if cur else "") +
              f"       user config file : {ctx.env.get('HOME', '')}/.condarc\n          conda version : 25.7.0\n"
              f"         python version : 3.13.7.final.0\n       base environment : {base}  (writable)\n"
              f"          envs directories : {envs_dir}\n               platform : linux-64\n\n")
        return 0
    if cmd in ("remove", "uninstall") or cmd == "env" and rest[:1] == ["remove"]:
        if cmd == "env":
            rest = rest[1:]
            rest = rest + ["--all"]
        name, prefix = opt("-n", "--name"), opt("-p", "--prefix")
        target = env_path(name) if name else (ctx.abspath(prefix) if prefix else current())
        if "--all" in rest:
            if target == base:
                ctx.e("\nCondaEnvironmentError: cannot remove root environment, add -n NAME or -p PREFIX option\n\n")
                return 1
            if not fs.exists(target):
                ctx.e(f"\nEnvironmentLocationNotFound: Not a conda environment: {target}\n\n")
                return 1
            if ctx.env.get("CONDA_PREFIX") == target:
                ctx.e("\nCondaEnvironmentError: Cannot remove current environment. Deactivate and run conda remove again\n\n")
                return 1
            ctx.w(f"\nRemove all packages in environment {target}:\n\n")
            ok = yield from confirm()
            if not ok:
                return 1
            ctx.w("\nPreparing transaction: done\nVerifying transaction: done\nExecuting transaction: done\n" if False else "")
            _rm_tree(fs, target)
            return 0
        specs = positional()
        have = installed(ctx, target, "conda")
        missing = [s for s in specs if norm(s) not in have]
        if missing:
            ctx.w(channels_block([]).replace("Solving environment: done", "Solving environment: failed"))
            ctx.e("\nPackagesNotFoundError: The following packages are missing from the target environment:\n\n" +
                  "".join(f"  - {m}\n" for m in missing) + "\n\n")
            return 1
        ctx.w(channels_block([]) + f"\n## Package Plan ##\n\n  environment location: {target}\n\n  removed specs:\n" +
              "".join(f"    - {s}\n" for s in specs) + "\n\nThe following packages will be REMOVED:\n\n" +
              "".join(f"  {norm(s)}-{have[norm(s)][1]}-py312_0\n" for s in specs) + "\n")
        ok = yield from confirm()
        if not ok:
            return 1
        ctx.w("\nPreparing transaction: done\nVerifying transaction: done\nExecuting transaction: done\n")
        for s in specs:
            remove_installed(ctx, target, "conda", s)
        return 0
    if cmd == "list":
        target = env_path(opt("-n", "--name")) if opt("-n", "--name") else current()
        if not fs.exists(target + "/conda-meta"):
            ctx.e(f"\nEnvironmentLocationNotFound: Not a conda environment: {target}\n\n")
            return 1
        pkgs = dict((k, v[1]) for k, v in installed(ctx, target, "conda").items())
        pyver = conda_pyver(fs, target)
        if pyver:
            pkgs["python"] = pyver
        rows = sorted(pkgs.items())
        pat = positional()
        if pat:
            rows = [r for r in rows if re.search(pat[0], r[0])]
        ctx.w(f"# packages in environment at {target}:\n#\n# Name                    Version                   Build  Channel\n" +
              "".join(f"{n:<25} {v:<25} {'py312_0':>6}\n" for n, v in rows))
        return 0
    ctx.e(f"usage: conda [-h] [-v] [--no-plugins] [-V] COMMAND ...\nconda: error: argument COMMAND: invalid choice: '{cmd}' "
          "(choose from activate, clean, commands, compare, config, create, deactivate, env, export, info, init, install, list, "
          "notices, package, remove, rename, run, search, update, upgrade)\n")
    return 2
