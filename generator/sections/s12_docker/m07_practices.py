"""12.07 Best practices & multi-stage builds."""

from ._common import P, catalog, docker, dockerfile, files, heredoc, py_app, setup, template

M = "12.07"
PY = "python:3.12-slim"
SIZE = "docker image inspect -f '{{.Size}}'"


def _proj(rng, deps=("requests",), extra=None):
    fs, word = py_app(rng, deps=deps)
    fs.update(extra or {})
    return setup(docker(f"pull={PY},python:3.12"), "#: cwd ~/app", files(fs)), word


def _w(*lines):
    return heredoc("Dockerfile", dockerfile(*lines))


@template(M, "lite")
def slim_vs_full(rng):
    s, word = _proj(rng)
    return P("Build the same app twice, from `python:3.12` as `fat:1` and from `python:3.12-slim` as `slim:1` (copy "
             "`app.py`, run it), and print both sizes as `REPO:TAG SIZE` (`docker images --format`).", s,
             _w("FROM python:3.12", "COPY app.py .", 'CMD ["python", "app.py"]') + "\ndocker build -t fat:1 .\n" +
             _w(f"FROM {PY}", "COPY app.py .", 'CMD ["python", "app.py"]') + "\ndocker build -t slim:1 .\n"
             "docker images --format '{{.Repository}}:{{.Tag}} {{.Size}}' fat\ndocker images --format "
             "'{{.Repository}}:{{.Tag}} {{.Size}}' slim",
             post="docker images --format '{{.Repository}}:{{.Tag}} {{.Size}}' | grep -E '^(fat|slim):' | sort",
             require=["docker build"])


@template(M, "lite")
def pin_base(rng):
    s, word = _proj(rng)
    items = [
        ("FROM", "Pin the base image: write a Dockerfile `FROM python:3.12-slim` (a specific version, not `latest`) "
                 "that copies and runs `app.py`, and build `pinned:1`.",
         _w(f"FROM {PY}", "COPY app.py .", 'CMD ["python", "app.py"]') + "\ndocker build -t pinned:1 ."),
    ]
    return catalog(rng, items, s, check=["image=pinned:1"], require=["docker build"])


@template(M, "easy")
def apt_one_layer(rng):
    pkg = rng.choice(["git", "curl", "jq"])
    s, word = _proj(rng)
    s = setup(s, _w(f"FROM {PY}", "RUN apt-get update", f"RUN apt-get install -y {pkg}", "COPY app.py .",
                    'CMD ["python", "app.py"]'))
    return P(f"This Dockerfile installs `{pkg}` in two RUNs and leaves apt's package lists in the image. Install it in "
             f"**one** RUN that updates, installs with `--no-install-recommends`, and removes `/var/lib/apt/lists/*`; "
             f"build `tools:1`.", s,
             _w(f"FROM {PY}", f"RUN apt-get update && apt-get install -y --no-install-recommends {pkg} && rm -rf "
                f"/var/lib/apt/lists/*", "COPY app.py .", 'CMD ["python", "app.py"]') + "\ndocker build -t tools:1 .",
             post=f"docker run --rm tools:1 sh -c 'command -v {pkg}; ls /var/lib/apt/lists | wc -l'; docker history "
                  f"--format '{{{{.CreatedBy}}}}' tools:1 | grep -c apt-get", require=["docker build"])


@template(M, "easy")
def no_pip_cache(rng):
    s, word = _proj(rng, deps=("requests", "numpy"))
    items = [
        ("pip", "Build `lean:1` from `python:3.12-slim` that installs `requirements.txt` **without** keeping pip's "
                "download cache in the image (`--no-cache-dir`), then copies and runs `app.py`.",
         _w(f"FROM {PY}", "COPY requirements.txt .", "RUN pip install --no-cache-dir -r requirements.txt",
            "COPY app.py .", 'CMD ["python", "app.py"]') + "\ndocker build -t lean:1 ."),
        ("ENV", "Build `lean:1` from `python:3.12-slim` with `PIP_NO_CACHE_DIR=1` set in an ENV before installing "
                "`requirements.txt`, then copy and run `app.py`.",
         _w(f"FROM {PY}", "ENV PIP_NO_CACHE_DIR=1", "COPY requirements.txt .", "RUN pip install -r requirements.txt",
            "COPY app.py .", 'CMD ["python", "app.py"]') + "\ndocker build -t lean:1 ."),
    ]
    return catalog(rng, items, s, post=f"docker run --rm lean:1; [ $({SIZE} lean:1) -lt 177000000 ] && echo small",
                   require=["docker build"])


@template(M, "medium")
def multi_stage(rng):
    s, word = _proj(rng)
    return P("Convert to a **multi-stage** build: a `builder` stage (from `python:3.12-slim`) creates a virtual "
             "environment at `/app/.venv` and installs `requirements.txt` into it; a `runtime` stage (from "
             "`python:3.12-slim`) copies only `/app/.venv` and `app.py` from the builder, puts the venv first on "
             "PATH, and runs `python app.py`. Build it as `ms:1`.", s,
             _w(f"FROM {PY} AS builder", "WORKDIR /app", "RUN python -m venv /app/.venv",
                "COPY requirements.txt .", "RUN /app/.venv/bin/pip install --no-cache-dir -r requirements.txt",
                "", f"FROM {PY} AS runtime", "WORKDIR /app", "COPY --from=builder /app/.venv /app/.venv",
                "COPY app.py .", 'ENV PATH="/app/.venv/bin:$PATH"', 'CMD ["python", "app.py"]') +
             "\ndocker build -t ms:1 .",
             post="docker run --rm ms:1; docker run --rm ms:1 sh -c 'command -v python; ls /app'; docker run --rm ms:1 "
                  "python -c 'import requests; print(requests.__name__)'", require=["docker build"])


@template(M, "medium")
def cache_mount(rng):
    s, word = _proj(rng, deps=("requests", "numpy"))
    return P("Speed up rebuilds: install `requirements.txt` with a BuildKit **cache mount** for pip's cache "
             "(`RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt`), so downloads are "
             "reused across builds without ending up in the image. Copy and run `app.py`; build `cached:1`.", s,
             _w("# syntax=docker/dockerfile:1", f"FROM {PY}", "COPY requirements.txt .",
                "RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt", "COPY app.py .",
                'CMD ["python", "app.py"]') + "\ndocker build -t cached:1 .",
             post="docker run --rm cached:1; grep -c 'type=cache' Dockerfile; docker run --rm cached:1 sh -c 'ls "
                  "/root/.cache/pip 2>/dev/null | wc -l'", require=["docker build"])


@template(M, "hard")
def build_secret(rng):
    token = f"hf_{rng.randint(10 ** 9, 10 ** 10)}"
    s, word = _proj(rng)
    s = setup(s, f"export HF_TOKEN={token}")
    return P("The build must read a Hugging Face token (it's in `$HF_TOKEN`) but the token must never end up in the "
             "image or its history. Use a secret mount: `RUN --mount=type=secret,id=hf_token sh -c 'test -s "
             "/run/secrets/hf_token && echo token ok'`, and pass it with `--secret id=hf_token,env=HF_TOKEN`. "
             "Also copy and run `app.py`; build `secret:1`.", s,
             _w("# syntax=docker/dockerfile:1", f"FROM {PY}",
                "RUN --mount=type=secret,id=hf_token sh -c 'test -s /run/secrets/hf_token && echo token ok'",
                "COPY app.py .", 'CMD ["python", "app.py"]') +
             "\ndocker build --secret id=hf_token,env=HF_TOKEN -t secret:1 .",
             post=f"docker run --rm secret:1; docker history --no-trunc secret:1 | grep -c {token}; docker run --rm "
                  f"secret:1 sh -c 'ls /run/secrets 2>&1 | wc -l'", require=["docker build --secret"])


@template(M, "hard")
def scout(rng):
    s, word = _proj(rng)
    s = setup(s, _w("FROM python:3.11-slim", "COPY app.py .", 'CMD ["python", "app.py"]'),
              "docker build -q -t old:1 . >/dev/null")
    return P("`docker scout cves old:1` reports critical vulnerabilities coming from its base image "
             "(`python:3.11-slim`). Follow scout's recommendation: switch the base to `python:3.12-slim`, rebuild as "
             "`old:1`, and check scout again.", s,
             "docker scout cves old:1\nsed -i 's/python:3.11-slim/python:3.12-slim/' Dockerfile\ndocker build -t old:1 "
             ".\ndocker scout cves old:1", post="docker scout cves old:1 | grep CRITICAL; docker run --rm old:1",
             require=["docker scout"])


@template(M, "challenge")
def optimize(rng):
    s, word = _proj(rng, extra={"data/raw.csv": "x\n" * 2000, ".env": "KEY=fake\n"})
    s = setup(s, _w("FROM python:3.12", "COPY . /app", "WORKDIR /app", "RUN pip install -r requirements.txt",
                    'CMD ["python", "app.py"]'), "docker build -q -t bloated:1 . >/dev/null")
    return P("**Put it on a diet.** `bloated:1` is over 1 GB, runs as root, ships `.env` and `data/`, and reinstalls "
             "dependencies on every code change. Rebuild it as `lean:1`: a slim base, a `.dockerignore` for `.env` and "
             "`data/`, dependencies installed (no pip cache) before the code is copied, and a non-root user `app`.",
             s, "printf '.env\\ndata/\\n' > .dockerignore\n" +
             _w(f"FROM {PY}", "WORKDIR /app", "COPY requirements.txt .",
                "RUN pip install --no-cache-dir -r requirements.txt", "COPY . .", "RUN useradd --create-home app",
                "USER app", 'CMD ["python", "app.py"]') + "\ndocker build -t lean:1 .",
             post=f"docker run --rm lean:1; docker run --rm lean:1 whoami; docker run --rm lean:1 ls -a; [ $({SIZE} "
                  f"lean:1) -lt 200000000 ] && echo small; echo '#' >> app.py; docker build -t lean:1 . 2>&1 | grep -c "
                  "'CACHED.*pip install'", require=["docker build"])


@template(M, "challenge")
def uv_multistage(rng):
    s, word = _proj(rng)
    s = setup(s, files({"pyproject.toml": f'[project]\nname = "{word}"\nversion = "0.1.0"\nrequires-python = ">=3.12"\n'
                                          'dependencies = ["requests"]\n'}))
    return P("**The guide's recipe.** Build `svc:1` in two stages: `builder` (from `python:3.12-slim`) installs `uv` "
             "with pip, creates `/app/.venv` with `uv venv` and installs `requests` into it with `uv pip install` (with "
             "the venv activated); "
             "`runtime` (from `python:3.12-slim`) copies `/app/.venv` and `app.py`, puts the venv on PATH, runs as a "
             "user `app`, and starts `python app.py`.", s,
             _w(f"FROM {PY} AS builder", "WORKDIR /app", "RUN pip install --no-cache-dir uv",
                "RUN uv venv /app/.venv && . /app/.venv/bin/activate && uv pip install requests", "",
                f"FROM {PY} AS runtime", "WORKDIR /app", "COPY --from=builder /app/.venv /app/.venv", "COPY app.py .",
                'ENV PATH="/app/.venv/bin:$PATH"', "RUN useradd -m app", "USER app", 'CMD ["python", "app.py"]') +
             "\ndocker build -t svc:1 .",
             post="docker run --rm svc:1; docker run --rm svc:1 whoami; docker run --rm svc:1 sh -c 'command -v uv || "
                  "echo no uv'", require=["docker build"])
