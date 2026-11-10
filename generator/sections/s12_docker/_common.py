"""Helpers shared by the Docker templates (section 12). Problems run in bash-lite's live terminal with docker-lite
(generator/bashlite/docker): kind "bash", graded by bashlite/check.py.

Most problems are graded on what the session leaves behind, checked after it ends (`post`): the engine's state as
/usr/lib/bashlite/dockerstate prints it (containers by name, images by tag, volumes, networks, registry pushes —
no ids, no times), and behaviour: what the built image prints when run, which layers stay CACHED after a code
change, who a container runs as, what curl gets from a published port. Problems about reading output compare
what the answer prints, using --format/inspect fields that don't depend on timing."""

from __future__ import annotations

import random
import shlex

from ...core import Problem, pick, template as _template

S = "docker"
STATE = "/usr/lib/bashlite/dockerstate"
APPS = ("api", "web", "worker", "trainer", "notebook", "inference", "scraper", "dashboard")
NAMES = ("web", "api", "cache", "db", "proxy", "worker", "app", "server")
WORDS = ("alpha", "bravo", "cobalt", "delta", "ember", "falcon", "garnet", "harbor", "indigo", "juniper")
MODELS = ("resnet18", "bert-base", "vit-small", "gpt2", "distilbert", "t5-small")


def template(module: str, tier: str):
    return _template(S, module, tier)


def q(s) -> str:
    return shlex.quote(str(s))


def docker(*flags: str) -> str:
    """The #: docker directive."""
    return "#: docker" + "".join(" " + f for f in flags if f)


def P(prompt: str, setup: str, solution: str, check=None, **kw) -> Problem:
    """check: dockerstate sections (graded after the session); post=...: any bash check; else output compared."""
    kw.setdefault("timeout", 30.0)
    if check is not None:
        kw["post"] = f"{STATE} {' '.join(check)}" + (f"; {kw.pop('post')}" if kw.get("post") else "")
    if kw.get("post"):
        kw.setdefault("answer", ["<post>"])
    return Problem(prompt=prompt.strip(), setup=setup.strip("\n"), solution=solution.strip("\n"), kind="bash",
                   lang="bash", **kw)


def catalog(rng: random.Random, items: list, setup: str = "", intro: str = "", **kw) -> Problem:
    """One task from a list. Item: (feature, prompt, solution[, {options}]); options may add `setup` lines, a
    `check`, `post`, and any Problem field (`require` defaults to [feature] when feature is given)."""
    item = pick(rng, items)
    feat, prompt, sol = item[:3]
    opts = dict(item[3]) if len(item) > 3 else {}
    req = opts.pop("require", kw.pop("require", None) or ([feat] if feat else []))
    extra = opts.pop("setup", "")
    full_setup = "\n".join(x for x in (setup, extra) if x)
    merged = {**kw, **opts}
    return P(f"{intro} {prompt}".strip(), full_setup, sol, require=req, **merged)


def heredoc(path: str, text: str) -> str:
    if text and not text.endswith("\n"):
        text += "\n"
    delim = "EOF"
    while delim in text.split("\n"):
        delim += "_"
    pre = f"mkdir -p {q(path.rsplit('/', 1)[0])} && " if "/" in path else ""
    return f"{pre}cat > {q(path)} <<'{delim}'\n{text}{delim}"


def files(spec: dict) -> str:
    return "\n".join(heredoc(p, t) for p, t in spec.items())


def setup(*parts: str) -> str:
    return "\n".join(p for p in parts if p)


def dockerfile(*lines: str) -> str:
    return "\n".join(lines) + "\n"


def py_app(rng, name="app.py", deps=("requests",)):
    """A tiny Python project: app.py printing a line, requirements.txt."""
    word = rng.choice(WORDS)
    return {name: f'import sys\nprint("{word} ready", *sys.argv[1:])\n',
            "requirements.txt": "".join(f"{d}\n" for d in deps)}, word


SERVE = '''from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()
clf = pipeline("text-classification", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")


class Request(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: Request):
    return clf(req.text)[0]
'''

API_DOCKERFILE = dockerfile(
    "FROM python:3.12-slim",
    "WORKDIR /app",
    'RUN pip install --no-cache-dir fastapi "uvicorn[standard]" transformers torch --extra-index-url '
    "https://download.pytorch.org/whl/cpu",
    "COPY serve.py .",
    "EXPOSE 8000",
    'CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]')
