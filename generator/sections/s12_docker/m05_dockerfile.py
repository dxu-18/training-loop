"""12.05 Dockerfile instructions."""

from ._common import WORDS, P, catalog, docker, dockerfile, files, heredoc, py_app, setup, template

M = "12.05"
PY = "python:3.12-slim"


def _write(lines):
    return heredoc("Dockerfile", dockerfile(*lines))


def _proj(rng, deps=("requests",)):
    fs, word = py_app(rng, deps=deps)
    return setup(docker(f"pull={PY}"), "#: cwd ~/app", files(fs)), word


@template(M, "lite")
def first_dockerfile(rng):
    s, word = _proj(rng)
    tag = rng.choice(["hello-app:1.0", "my-app:dev", "word-app:0.1"])
    items = [
        ("docker build", f"Write a `Dockerfile` that starts from `{PY}`, works in `/app`, copies `app.py` there and "
                         f"runs `python app.py` by default (exec form). Build it as `{tag}`.",
         _write([f"FROM {PY}", "WORKDIR /app", "COPY app.py .", 'CMD ["python", "app.py"]']) +
         f"\ndocker build -t {tag} .", {"post": f"docker run --rm {tag}; docker image inspect -f "
                                                 f"'{{{{.Config.WorkingDir}}}} {{{{json .Config.Cmd}}}}' {tag}"}),
        ("docker build", f"Write a `Dockerfile` (FROM `{PY}`, WORKDIR `/srv`) that copies the whole folder in, "
                         f"installs `requirements.txt` with pip (no cache) and runs `python app.py`. Build it as "
                         f"`{tag}`.",
         _write([f"FROM {PY}", "WORKDIR /srv", "COPY . .", "RUN pip install --no-cache-dir -r requirements.txt",
                 'CMD ["python", "app.py"]']) + f"\ndocker build -t {tag} .",
         {"post": f"docker run --rm {tag}; docker run --rm {tag} pip show requests | head -2"}),
    ]
    return catalog(rng, items, s)


@template(M, "lite")
def metadata(rng):
    s, word = _proj(rng)
    v = rng.choice(["1.0", "2.3", "0.4"])
    port = rng.choice([8000, 8080, 5000])
    items = [
        ("ENV", f"Build `meta:1` from `{PY}` with the environment variables `PYTHONUNBUFFERED=1` and `MODE={word}` "
                f"set in the image (one ENV), and `python app.py` as the command.",
         _write([f"FROM {PY}", f"ENV PYTHONUNBUFFERED=1 MODE={word}", "COPY app.py .", 'CMD ["python", "app.py"]'])
         + "\ndocker build -t meta:1 .", {"require": ["docker build"]}),
        ("LABEL", f"Build `meta:1` from `{PY}` with the label `org.opencontainers.image.version={v}` and port "
                  f"{port} documented with EXPOSE.",
         _write([f"FROM {PY}", f"LABEL org.opencontainers.image.version={v}", f"EXPOSE {port}"]) +
         "\ndocker build -t meta:1 .", {"require": ["docker build"]}),
        ("ARG", f"Build `meta:1` from `{PY}` with a build argument `APP_VERSION` (default `dev`) that ends up in the "
                f"environment variable `VERSION`, passing `--build-arg APP_VERSION={v}`.",
         _write([f"FROM {PY}", "ARG APP_VERSION=dev", "ENV VERSION=$APP_VERSION"]) +
         f"\ndocker build --build-arg APP_VERSION={v} -t meta:1 .", {"require": ["docker build --build-arg"]}),
    ]
    return catalog(rng, items, s, check=["image=meta:1"])


@template(M, "easy")
def entrypoint_cmd(rng):
    s, word = _proj(rng)
    n = rng.randint(2, 9)
    return P(f"Make `cli:1` (from `{PY}`, `app.py` copied to `/app`) always run `python /app/app.py` (ENTRYPOINT, exec "
             f"form) with default arguments `--epochs 10` (CMD), so that `docker run --rm cli:1 --epochs {n}` "
             f"replaces only the arguments.", s,
             _write([f"FROM {PY}", "WORKDIR /app", "COPY app.py .", 'ENTRYPOINT ["python", "/app/app.py"]',
                     'CMD ["--epochs", "10"]']) + "\ndocker build -t cli:1 .",
             post=f"docker run --rm cli:1; docker run --rm cli:1 --epochs {n}", require=["docker build"])


@template(M, "easy")
def non_root(rng):
    s, word = _proj(rng)
    user = rng.choice(["appuser", "ml", "runner"])
    return P(f"Build `safe:1` from `{PY}` that copies `app.py` to `/app` and runs it as a non-root user `{user}` "
             f"(create it with `useradd --create-home`).", s,
             _write([f"FROM {PY}", "WORKDIR /app", "COPY app.py .", f"RUN useradd --create-home {user}",
                     f"USER {user}", 'CMD ["python", "app.py"]']) + "\ndocker build -t safe:1 .",
             post="docker run --rm safe:1; docker run --rm safe:1 whoami; docker run --rm safe:1 sh -c 'echo $HOME'",
             require=["docker build"])


@template(M, "medium")
def cache_order(rng):
    s, word = _proj(rng)
    bad = _write([f"FROM {PY}", "WORKDIR /app", "COPY . .", "RUN pip install --no-cache-dir -r requirements.txt",
                  'CMD ["python", "app.py"]'])
    return P("Every code change reinstalls the dependencies because the `Dockerfile` copies everything before "
             "`pip install`. Reorder it so `requirements.txt` is copied and installed first and the code after, then "
             "build it as `app:1`.", setup(s, bad),
             _write([f"FROM {PY}", "WORKDIR /app", "COPY requirements.txt .",
                     "RUN pip install --no-cache-dir -r requirements.txt", "COPY . .", 'CMD ["python", "app.py"]'])
             + "\ndocker build -t app:1 .",
             post="docker run --rm app:1; echo '# change' >> app.py; docker build -t app:1 . 2>&1 | grep -c "
                  "'CACHED.*pip install'", require=["docker build"])


@template(M, "medium")
def trailing_comments(rng):
    s, word = _proj(rng)
    port = rng.choice([8000, 8080])
    bad = _write([f"FROM {PY}", f"ENV MODE=prod # production settings", f"EXPOSE {port} # the API port",
                  "COPY app.py .", 'CMD ["python", "app.py"]'])
    return P("`docker build -t fixme:1 .` fails: in a Dockerfile `#` only starts a comment at the beginning of a "
             "line. Move the two trailing comments onto their own lines (keep them) and build `fixme:1`.",
             setup(s, bad), _write([f"FROM {PY}", "# production settings", "ENV MODE=prod", "# the API port",
                                    f"EXPOSE {port}", "COPY app.py .", 'CMD ["python", "app.py"]']) +
             "\ndocker build -t fixme:1 .", check=["image=fixme:1"], post="grep -c '^#' Dockerfile",
             require=["docker build"])


@template(M, "hard")
def healthcheck(rng):
    port = rng.choice([8000, 8080])
    s = setup(docker(f"pull={PY}"), "#: cwd ~/api",
              files({"serve.py": "from fastapi import FastAPI\n\napp = FastAPI()\n\n\n@app.get(\"/health\")\ndef "
                                 "health():\n    return {\"status\": \"ok\"}\n"}))
    hc = (f"HEALTHCHECK --interval=5s CMD python -c \"import urllib.request; "
          f"urllib.request.urlopen('http://localhost:{port}/health')\" || exit 1")
    return P(f"Build `api:1`: `{PY}`, install `fastapi` and `uvicorn` (no pip cache), copy `serve.py` to `/app`, "
             f"serve it with `uvicorn serve:app --host 0.0.0.0 --port {port}` (exec form), and add a HEALTHCHECK every "
             f"5s that requests `http://localhost:{port}/health` with Python's urllib. Then run it detached as `api` and "
             f"wait until Docker reports it healthy.", s,
             _write([f"FROM {PY}", "WORKDIR /app", "RUN pip install --no-cache-dir fastapi uvicorn", "COPY serve.py .",
                     hc, f'CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "{port}"]']) +
             "\ndocker build -t api:1 .\ndocker run -d --name api api:1\nsleep 12\ndocker inspect -f "
             "'{{.State.Health.Status}}' api",
             check=["image=api:1", "containers"], require=["docker build"])


@template(M, "hard")
def exec_form(rng):
    port = rng.choice([8000, 8080])
    s = setup(docker(f"pull={PY}"), "#: cwd ~/api",
              files({"serve.py": "from fastapi import FastAPI\n\napp = FastAPI()\n"}),
              _write([f"FROM {PY}", "WORKDIR /app", "RUN pip install --no-cache-dir fastapi uvicorn", "COPY serve.py .",
                      f"CMD uvicorn serve:app --host 0.0.0.0 --port {port}"]))
    return P("`docker stop` on this API takes 10 seconds and ends with exit code 137: its CMD is in *shell form*, so "
             "`/bin/sh` is PID 1 and never passes SIGTERM on. Rewrite the CMD in exec form, build `api:2`, run it "
             "detached as `api`, stop it, and print its exit code.", s,
             _write([f"FROM {PY}", "WORKDIR /app", "RUN pip install --no-cache-dir fastapi uvicorn", "COPY serve.py .",
                     f'CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "{port}"]']) +
             "\ndocker build -t api:2 .\ndocker run -d --name api api:2\ndocker stop api\n"
             "docker inspect -f '{{.State.ExitCode}}' api",
             check=["image=api:2"], post="docker inspect -f '{{.State.ExitCode}}' api", require=["docker build"])


@template(M, "challenge")
def training_image(rng):
    epochs = rng.randint(3, 20)
    s = setup(docker(f"pull={PY}"), "#: cwd ~/train",
              files({"train.py": "import argparse\np = argparse.ArgumentParser()\np.add_argument('--epochs', type=int, "
                                 "default=1)\na = p.parse_args()\nprint(f'training for {a.epochs} epochs')\n",
                     "requirements.txt": "numpy\n", "data/big.csv": "x\n" * 50, ".env": "WANDB_API_KEY=fake\n"}))
    return P(f"**A training image.** Write a Dockerfile and a `.dockerignore` and build `trainer:1`: base `{PY}`, "
             "`PYTHONUNBUFFERED=1`, work in `/workspace`, dependencies installed before the code is copied (no pip "
             "cache), run as a non-root user `ml`, ENTRYPOINT `python train.py` and default arguments `--epochs "
             f"{epochs}`. `data/` and `.env` must stay out of the image.", s,
             "printf 'data/\\n.env\\n' > .dockerignore\n" +
             _write([f"FROM {PY}", "ENV PYTHONUNBUFFERED=1", "WORKDIR /workspace", "COPY requirements.txt .",
                     "RUN pip install --no-cache-dir -r requirements.txt", "COPY . .", "RUN useradd --create-home ml",
                     "USER ml", 'ENTRYPOINT ["python", "train.py"]', f'CMD ["--epochs", "{epochs}"]']) +
             "\ndocker build -t trainer:1 .",
             post="docker run --rm trainer:1; docker run --rm trainer:1 --epochs 2; docker run --rm --entrypoint ls "
                  "trainer:1 -a; docker run --rm --entrypoint whoami trainer:1; echo '#' >> train.py; docker build -t "
                  "trainer:1 . 2>&1 | grep -c 'CACHED.*pip install'", require=["docker build"])


@template(M, "challenge")
def broken_build(rng):
    s, word = _proj(rng)
    bad = _write([f"FORM {PY}", "WORKDIR /app", "COPY requirement.txt .",
                  "RUN pip install --no-cache-dir -r requirements.txt", "COPY . .", 'CMD ["python", "app.py"]'])
    return P("**Make it build.** This Dockerfile has two mistakes that stop `docker build -t fixed:1 .`. Read the "
             "errors, fix the Dockerfile and build `fixed:1`.", setup(s, bad),
             _write([f"FROM {PY}", "WORKDIR /app", "COPY requirements.txt .",
                     "RUN pip install --no-cache-dir -r requirements.txt", "COPY . .", 'CMD ["python", "app.py"]']) +
             "\ndocker build -t fixed:1 .", check=["image=fixed:1"], post="docker run --rm fixed:1",
             require=["docker build"])
