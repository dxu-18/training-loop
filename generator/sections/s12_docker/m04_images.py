"""12.04 Images."""

from ._common import WORDS, P, catalog, docker, dockerfile, files, q, setup, template

M = "12.04"
PRE = "pull=python:3.12-slim,ubuntu:24.04,nginx,alpine:3.20,hello-world"


def _built(rng, tag="my-app:1.0"):
    word = rng.choice(WORDS)
    return setup(docker(PRE), "#: cwd ~/app",
                 files({"app.py": f'print("{word}")\n', "Dockerfile": dockerfile(
                     "FROM python:3.12-slim", "WORKDIR /app", "COPY app.py .", 'CMD ["python", "app.py"]')}),
                 f"docker build -q -t {tag} . >/dev/null")


@template(M, "lite")
def list_pull(rng):
    items = [
        ("docker images", "List your images as `REPOSITORY:TAG`, one per line.",
         "docker images --format '{{.Repository}}:{{.Tag}}'", {"unordered": True}),
        ("docker pull", "Download `busybox:latest` without running it.", "docker pull busybox:latest",
         {"post": "docker images --format '{{.Repository}}:{{.Tag}}' | sort"}),
        ("docker pull", "Download the `python:3.11-slim` image.", "docker pull python:3.11-slim",
         {"post": "docker images --format '{{.Repository}}:{{.Tag}}' | sort"}),
        ("docker images -q", "Print just the image ids (each once).", "docker images -q", {"unordered": True}),
        ("docker pull --platform", "Download the **x86** build of `alpine:3.20` explicitly (`linux/amd64`), even on "
                                   "an Apple Silicon Mac.", "docker pull --platform linux/amd64 alpine:3.20",
         {"post": "docker image inspect -f '{{.Architecture}}' alpine:3.20"}),
    ]
    return catalog(rng, items, setup(docker(PRE)))


@template(M, "lite")
def tag_rm(rng):
    v = rng.choice(["1.0", "2.1", "0.9"])
    items = [
        ("docker tag", f"Give `my-app:1.0` a second name: `ghcr.io/ada/my-app:{v}`.",
         f"docker tag my-app:1.0 ghcr.io/ada/my-app:{v}"),
        ("docker tag", "Tag `my-app:1.0` as `my-app:latest` too.", "docker tag my-app:1.0 my-app:latest"),
        ("docker rmi", "Remove the `hello-world` image.", "docker rmi hello-world"),
        ("docker rmi", "Remove the `alpine:3.20` and `nginx` images in one command.", "docker rmi alpine:3.20 nginx"),
    ]
    return catalog(rng, items, _built(rng), post="docker images --format '{{.Repository}}:{{.Tag}}' | sort")


@template(M, "easy")
def history(rng):
    items = [
        ("docker history", "Show the layers of `my-app:1.0` and the instruction that made each.",
         "docker history my-app:1.0"),
        ("docker history", "Print just the instruction of each layer of `my-app:1.0` (`--format '{{.CreatedBy}}'`, "
                           "untruncated).", "docker history --no-trunc --format '{{.CreatedBy}}' my-app:1.0"),
        ("docker image inspect", "Print `my-app:1.0`'s default command (`.Config.Cmd`) as JSON.",
         "docker image inspect -f '{{json .Config.Cmd}}' my-app:1.0"),
        ("docker image inspect", "Print `nginx`'s exposed ports (`.Config.ExposedPorts`) as JSON.",
         "docker image inspect -f '{{json .Config.ExposedPorts}}' nginx"),
        ("docker image inspect", "Print `python:3.12-slim`'s environment variables, one per line.",
         "docker image inspect -f '{{range .Config.Env}}{{println .}}{{end}}' python:3.12-slim"),
    ]
    return catalog(rng, items, _built(rng))


@template(M, "easy")
def image_prune(rng):
    s = setup(_built(rng), "echo 'print(2)' > app.py && docker build -q -t my-app:1.0 . >/dev/null",
              "echo 'print(3)' > app.py && docker build -q -t my-app:1.0 . >/dev/null")
    items = [
        ("docker image prune", "Rebuilding `my-app:1.0` left untagged (`<none>`) images behind. Remove those dangling "
                               "images (no prompt).", "docker image prune -f"),
        ("docker images", "List only the dangling images' ids (`--filter dangling=true -q`).",
         "docker images --filter dangling=true -q", {"unordered": True,
                                                       "require": ["docker images --filter|docker images -f"]}),
    ]
    return catalog(rng, items, s, post="docker images --format '{{.Repository}}:{{.Tag}}' | sort")


@template(M, "medium")
def save_load(rng):
    return P("There's no registry between your laptop and the GPU server. Save `my-app:1.0` to `~/my-app.tar`, copy it "
             "to `gpu-server` with scp, and load it there (`ssh gpu-server docker load -i my-app.tar`).",
             setup(_built(rng), "#: host gpu-server key docker"),
             "docker save my-app:1.0 -o ~/my-app.tar\nscp ~/my-app.tar gpu-server:\nssh gpu-server docker load -i my-app.tar",
             post="ssh gpu-server docker images --format '{{.Repository}}:{{.Tag}}'; ls ~/my-app.tar",
             require=["docker save", "docker load"])


@template(M, "medium")
def save_rmi_load(rng):
    return P("Back up `my-app:1.0` to `~/backup.tar`, delete the image, then restore it from the tar file.",
             _built(rng), "docker save -o ~/backup.tar my-app:1.0\ndocker rmi my-app:1.0\ndocker load -i ~/backup.tar",
             post="docker images --format '{{.Repository}}:{{.Tag}}' | sort; docker run --rm my-app:1.0",
             require=["docker save", "docker load"])


@template(M, "hard")
def biggest(rng):
    items = [
        ("docker images", "Print the repository:tag of your largest image (sort by size; images show sizes like "
                          "`124MB`/`7.8MB`/`13.3kB`, so compare carefully, or use `docker image inspect -f '{{.Size}}'`).",
         "for i in $(docker images --format '{{.Repository}}:{{.Tag}}'); do echo \"$(docker image inspect -f "
         "'{{.Size}}' $i) $i\"; done | sort -n | tail -1 | cut -d' ' -f2", {"require": ["docker image inspect"]}),
        ("docker history", "Print the size of the largest layer of `my-app:1.0`'s history (`--format '{{.Size}}'`, the "
                           "first line after sorting human sizes with `sort -h`).",
         "docker history --format '{{.Size}}' my-app:1.0 | sort -h | tail -1"),
    ]
    return catalog(rng, items, _built(rng))


@template(M, "hard")
def rmi_conflict(rng):
    return P("`docker rmi my-app:1.0` fails because a stopped container `old-run` still uses it. Remove that "
             "container, then the image.",
             setup(_built(rng), "docker run -q --name old-run my-app:1.0 >/dev/null"),
             "docker rm old-run\ndocker rmi my-app:1.0", check=["containers-all"],
             post="docker images --format '{{.Repository}}:{{.Tag}}' | sort", require=["docker rmi"])


@template(M, "challenge")
def commit_vs_dockerfile(rng):
    pkg = rng.choice(["curl", "git", "jq"])
    return P(f"**Reproducible, please.** Someone ran `apt-get install -y {pkg}` inside an `ubuntu:24.04` container and "
             f"saved it with `docker commit`. Do it properly instead: write a `Dockerfile` (FROM `ubuntu:24.04`, one "
             f"RUN that updates, installs `{pkg}` without recommends, and removes `/var/lib/apt/lists/*`) and build it "
             f"as `tools:1.0`.", setup(docker("pull=ubuntu:24.04"), "#: cwd ~/tools"),
             "cat > Dockerfile <<'EOF'\nFROM ubuntu:24.04\nRUN apt-get update && apt-get install -y "
             f"--no-install-recommends {pkg} && rm -rf /var/lib/apt/lists/*\nEOF\ndocker build -t tools:1.0 .",
             post=f"docker run --rm tools:1.0 sh -c 'command -v {pkg}; ls /var/lib/apt/lists | wc -l'; docker history "
                  f"--no-trunc --format '{{{{.CreatedBy}}}}' tools:1.0 | grep -c 'apt-get update && apt-get install'",
             require=["docker build"])


@template(M, "challenge")
def migrate_registry(rng):
    v = rng.choice(["1.0", "1.1"])
    return P(f"**Rename for release.** Tag `my-app:1.0` as `ghcr.io/ada/my-app:{v}` and `ghcr.io/ada/my-app:latest`, "
             f"then remove the old `my-app:1.0` name (the image itself must stay).", _built(rng),
             f"docker tag my-app:1.0 ghcr.io/ada/my-app:{v}\ndocker tag my-app:1.0 ghcr.io/ada/my-app:latest\n"
             "docker rmi my-app:1.0",
             post="docker images --format '{{.Repository}}:{{.Tag}}' | sort; docker run --rm ghcr.io/ada/my-app:latest",
             require=["docker tag", "docker rmi"])
