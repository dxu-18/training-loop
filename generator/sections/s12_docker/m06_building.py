"""12.06 Building images & .dockerignore."""

from ._common import P, catalog, docker, dockerfile, files, heredoc, py_app, setup, template

M = "12.06"
PY = "python:3.12-slim"
DF = dockerfile(f"FROM {PY} AS base", "ARG APP_VERSION=dev", "ENV APP_VERSION=$APP_VERSION", "WORKDIR /app",
                "COPY . .", 'CMD ["python", "app.py"]', "", "FROM base AS dev", "ENV DEBUG=1")


def _proj(rng, extra=None):
    fs, word = py_app(rng)
    fs["Dockerfile"] = DF
    fs["docker/Dockerfile.dev"] = dockerfile(f"FROM {PY}", "ENV DEBUG=1", "WORKDIR /app", "COPY app.py .",
                                             'CMD ["python", "app.py", "--dev"]')
    fs.update(extra or {})
    return setup(docker(f"pull={PY}"), "#: cwd ~/app", files(fs)), word


@template(M, "lite")
def build_flags(rng):
    s, word = _proj(rng)
    v = rng.choice(["1.0", "1.4", "2.0"])
    items = [
        ("docker build -t", f"Build the image in this folder and name it `my-app:{v}`.", f"docker build -t my-app:{v} .",
         {"check": [f"image=my-app:{v}"]}),
        ("docker build -f", "Build `docker/Dockerfile.dev` (the build context is still this folder) as `my-app:dev`.",
         "docker build -t my-app:dev -f docker/Dockerfile.dev .", {"check": ["image=my-app:dev"]}),
        ("docker build --build-arg", f"Build `my-app:{v}` with the build argument `APP_VERSION={v}`.",
         f"docker build --build-arg APP_VERSION={v} -t my-app:{v} .", {"check": [f"image=my-app:{v}"]}),
        ("docker build --target", "Build only up to the stage named `base`, as `my-app:base`.",
         "docker build --target base -t my-app:base .", {"check": ["image=my-app:base"]}),
        ("docker build --no-cache", "Rebuild `my-app:latest` from scratch, ignoring the build cache, with plain "
                                    "progress output.", "docker build --no-cache --progress=plain -t my-app .",
         {"check": ["image=my-app:latest"], "require": ["docker build --no-cache"]}),
    ]
    return catalog(rng, items, s)


@template(M, "lite")
def quiet_and_id(rng):
    s, word = _proj(rng)
    items = [
        ("docker build -q", "Build the image quietly (`-q`) as `quiet:1`: only its id is printed.",
         "docker build -q -t quiet:1 .", {"check": ["image=quiet:1"]}),
        ("docker build", "Build `my-app:1`, then run it once (removed afterwards).",
         "docker build -t my-app:1 .\ndocker run --rm my-app:1", {"post": "docker images --format "
                                                                          "'{{.Repository}}:{{.Tag}}' | sort"}),
    ]
    return catalog(rng, items, s)


@template(M, "easy")
def dockerignore(rng):
    s, word = _proj(rng, {".env": "OPENAI_API_KEY=fake-123\n", "data/train.csv": "a,b\n" * 100,
                          "checkpoints/epoch_9.pt": "w" * 3000, "__pycache__/app.cpython-312.pyc": "x\n",
                          ".venv/pyvenv.cfg": "home = /usr/bin\n"})
    return P("`COPY . .` currently puts `.env`, `data/`, `checkpoints/`, `__pycache__/` and `.venv/` into the image. "
             "Write a `.dockerignore` that keeps all five out, then build `clean:1`.", s,
             "printf '.env\\ndata/\\ncheckpoints/\\n__pycache__/\\n.venv/\\n' > .dockerignore\ndocker build -t clean:1 .",
             post="docker run --rm clean:1 ls -a", require=["docker build"])


@template(M, "easy")
def whats_in(rng):
    s, word = _proj(rng, {".env": "TOKEN=fake\n", ".dockerignore": ".env\n"})
    items = [
        ("docker run", "Build `check:1`, then list the files that ended up in its `/app` (run `ls -A` in it).",
         "docker build -t check:1 .\ndocker run --rm check:1 ls -A"),
        ("docker run", "Build `check:1`, then show whether `/app/.env` is in the image (`ls -a /app`).",
         "docker build -t check:1 .\ndocker run --rm check:1 ls -a /app"),
    ]
    return catalog(rng, items, s)


@template(M, "medium")
def big_context(rng):
    s, word = _proj(rng, {"checkpoints/big.pt": "", "README.md": "# app\n"})
    s = setup(s, "head -c 30000000 /dev/zero > checkpoints/big.pt")
    return P("The build sends a 30 MB checkpoint to the builder every time (`transferring context`), although the "
             "image never uses it. Keep `checkpoints/` and every `*.pt` file out of the build context and build "
             "`small:1`.", s, "printf 'checkpoints/\\n*.pt\\n' > .dockerignore\ndocker build -t small:1 .",
             post="docker run --rm small:1 ls; docker build -t small:1 . 2>&1 | grep 'transferring context' | tail -1 "
                  "| awk '{print $4}'", require=["docker build"])


@template(M, "medium")
def build_cache(rng):
    s, word = _proj(rng)
    s = setup(s, "docker build -q -t a:1 . >/dev/null", "docker build -q --no-cache -t a:2 . >/dev/null")
    items = [
        ("docker builder prune", "Clear Docker's build cache (no prompt).", "docker builder prune -f"),
        ("docker system df", "Show how much space images, containers, volumes and the build cache take.",
         "docker system df"),
    ]
    return catalog(rng, items, s, post="docker system df --format '{{.Type}} {{.TotalCount}}'")


@template(M, "hard")
def platforms(rng):
    s, word = _proj(rng)
    items = [
        ("docker build --platform", "You're on an Apple Silicon laptop but the server is x86: build `my-app:amd64` "
                                    "for `linux/amd64` explicitly.",
         "docker build --platform linux/amd64 -t my-app:amd64 .", {"check": ["image=my-app:amd64"]}),
        ("docker buildx build", "Build for **both** `linux/amd64` and `linux/arm64` with buildx and push it as "
                                "`ghcr.io/ada/my-app:1.0`. Multi-platform builds need a `docker-container` builder "
                                "(`docker buildx create --use`) and you must be logged in to ghcr.io (use your "
                                "`gh auth token`).",
         "docker buildx create --use\ngh auth token | docker login ghcr.io -u ada --password-stdin\n"
         "docker buildx build --platform linux/amd64,linux/arm64 -t ghcr.io/ada/my-app:1.0 --push .",
         {"check": ["registry"], "setup": "#: github key login", "require": ["docker buildx build"]}),
    ]
    return catalog(rng, items, s)


@template(M, "hard")
def from_stage(rng):
    s, word = _proj(rng)
    items = [
        ("docker build --target", "Build the `base` stage of the Dockerfile as `my-app:base` and check it doesn't "
                                  "have the `DEBUG` variable the `dev` stage adds (`docker run --rm my-app:base env`).",
         "docker build --target base -t my-app:base .\ndocker run --rm my-app:base env",
         {"check": ["image=my-app:base"]}),
        ("docker build --target", "The Dockerfile has two stages, `base` and `dev`. Build the `base` stage as "
                                  "`my-app:prod` and the `dev` stage as `my-app:dev` (two builds with `--target`).",
         "docker build --target base -t my-app:prod .\ndocker build --target dev -t my-app:dev .",
         {"check": ["image=my-app:prod", "image=my-app:dev"]}),
    ]
    return catalog(rng, items, s)


@template(M, "challenge")
def leaked_secret(rng):
    key = f"sk-live-{rng.randint(100000, 999999)}"
    s, word = _proj(rng, {".env": f"API_KEY={key}\n"})
    s = setup(s, "docker build -q -t leaky:1 . >/dev/null")
    return P("**A secret in the image.** `leaky:1` was built with `COPY . .` and anyone who pulls it can read `.env` "
             "(try `docker run --rm leaky:1 cat .env`). Make sure future builds never include `.env`, rebuild the "
             "image under the same name, and remove the old (now dangling) image.", s,
             "echo .env > .dockerignore\ndocker build -t leaky:1 .\ndocker image prune -f",
             post="docker run --rm leaky:1 ls -a; docker images -q --filter dangling=true | wc -l; grep -c '^.env$' "
                  ".dockerignore", require=["docker build"])


@template(M, "challenge")
def versioned(rng):
    v = rng.choice(["1.2.0", "2.0.1", "0.9.3"])
    s, word = _proj(rng)
    return P(f"**Stamp the version.** Build the image with `APP_VERSION={v}`, tag it as `my-app:{v}` and "
             f"`my-app:latest` in one build (two `-t`), with the label `org.opencontainers.image.version={v}` added "
             f"from the command line (`--label`).", s,
             f"docker build --build-arg APP_VERSION={v} --label org.opencontainers.image.version={v} -t my-app:{v} "
             f"-t my-app:latest .", check=[f"image=my-app:{v}", "image=my-app:latest"],
             require=["docker build --build-arg", "docker build --label"])
