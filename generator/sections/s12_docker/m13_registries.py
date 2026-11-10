"""12.13 Registries: Docker Hub, GHCR, pushing, pulling elsewhere."""

from ._common import WORDS, P, catalog, docker, dockerfile, files, setup, template

M = "12.13"
HUB_TOKEN = "dckr_pat_7Qm2xLk9Vb"
GH_LOGIN = "gh auth token | docker login ghcr.io -u ada --password-stdin"
HUB_LOGIN = 'echo "$DOCKERHUB_TOKEN" | docker login -u ada --password-stdin'


def _built(rng, tag="my-app:1.0", hub=False, gh=False, host=False):
    word = rng.choice(WORDS)
    return setup(docker("pull=python:3.12-slim", f"hub=ada:{HUB_TOKEN}" if hub else ""),
                 "#: github key login" if gh else "", "#: host gpu-server key docker" if host else "", "#: cwd ~/app",
                 files({"app.py": f'print("{word} ready")\n', "Dockerfile": dockerfile(
                     "FROM python:3.12-slim", "WORKDIR /app", "COPY app.py .", 'CMD ["python", "app.py"]')}),
                 f"docker build -q -t {tag} . >/dev/null", f"export DOCKERHUB_TOKEN={HUB_TOKEN}" if hub else "")


@template(M, "lite")
def hub_push(rng):
    v = rng.choice(["1.0", "0.3.1", "2.1"])
    return P(f"Publish `my-app:1.0` on Docker Hub as `ada/my-app:{v}`: log in as `ada` with the access token in "
             f"`$DOCKERHUB_TOKEN` (piped to `--password-stdin`), tag the image, and push it.",
             _built(rng, hub=True), f"{HUB_LOGIN}\ndocker tag my-app:1.0 ada/my-app:{v}\ndocker push ada/my-app:{v}",
             check=["registry"], require=["docker login", "docker push"])


@template(M, "lite")
def login_logout(rng):
    s = _built(rng, gh=True)
    items = [
        ("docker login", "Log in to the GitHub Container Registry (`ghcr.io`) as `ada`, piping your GitHub token "
                         "(`gh auth token`) to `--password-stdin`.", GH_LOGIN),
        ("docker logout", "You're logged in to `ghcr.io`. Log out of it.", "docker logout ghcr.io",
         {"setup": f"{GH_LOGIN} >/dev/null 2>&1"}),
    ]
    return catalog(rng, items, s, check=["auth"])


@template(M, "easy")
def ghcr_push(rng):
    v = rng.choice(["1.0.0", "0.2.0", "1.4.2"])
    return P(f"Push `my-app:1.0` to GitHub's registry as `ghcr.io/ada/my-app:{v}`: log in with `gh auth token`, tag, "
             f"push.", _built(rng, gh=True),
             f"{GH_LOGIN}\ndocker tag my-app:1.0 ghcr.io/ada/my-app:{v}\ndocker push ghcr.io/ada/my-app:{v}",
             check=["registry"], require=["docker push"])


@template(M, "easy")
def pull_back(rng):
    s = setup(_built(rng, gh=True), f"{GH_LOGIN} >/dev/null 2>&1", "docker tag my-app:1.0 ghcr.io/ada/my-app:1.0",
              "docker push -q ghcr.io/ada/my-app:1.0 >/dev/null", "docker rmi -f my-app:1.0 ghcr.io/ada/my-app:1.0 "
                                                                   ">/dev/null")
    return P("You pushed `ghcr.io/ada/my-app:1.0` and then deleted it locally. Pull it back from the registry and "
             "run it (removed afterwards).", s, "docker pull ghcr.io/ada/my-app:1.0\ndocker run --rm ghcr.io/ada/my-app:1.0",
             post="docker images --format '{{.Repository}}:{{.Tag}}' | grep ghcr", require=["docker pull|docker run"])


@template(M, "medium")
def denied(rng):
    items = [
        ("docker push", "`docker push my-app:1.0` fails with *denied: requested access to the resource is denied*: "
                        "without a namespace it means `docker.io/library/my-app`, which isn't yours. You're logged in "
                        "to Docker Hub as `ada`; push the image as `ada/my-app:1.0`.",
         "docker tag my-app:1.0 ada/my-app:1.0\ndocker push ada/my-app:1.0",
         {"setup": f"{HUB_LOGIN} >/dev/null 2>&1"}),
        ("docker login", "`docker push ghcr.io/ada/my-app:1.0` fails with *unauthorized*: you never logged in to "
                         "ghcr.io. Log in as `ada` with your GitHub token (`gh auth token`) and push it again.",
         f"{GH_LOGIN}\ndocker push ghcr.io/ada/my-app:1.0",
         {"setup": "docker tag my-app:1.0 ghcr.io/ada/my-app:1.0"}),
        ("docker tag", "`docker push ghcr.io/team-ml/my-app:1.0` is denied: your token can only push to your own "
                       "namespace. Publish it as `ghcr.io/ada/my-app:1.0` instead (you're logged in).",
         "docker tag my-app:1.0 ghcr.io/ada/my-app:1.0\ndocker push ghcr.io/ada/my-app:1.0",
         {"setup": f"{GH_LOGIN} >/dev/null 2>&1\ndocker tag my-app:1.0 ghcr.io/team-ml/my-app:1.0"}),
    ]
    return catalog(rng, items, _built(rng, hub=True, gh=True), check=["registry"], require=["docker push"])


@template(M, "medium")
def multi_tag(rng):
    v = rng.choice(["1.2.0", "2.0.1", "0.9.0"])
    return P(f"Release version {v} on ghcr.io: tag `my-app:1.0` as both `ghcr.io/ada/my-app:{v}` and "
             f"`ghcr.io/ada/my-app:latest`, and push **all** tags of that repository in one command. You're logged "
             f"in.", setup(_built(rng, gh=True), f"{GH_LOGIN} >/dev/null 2>&1"),
             f"docker tag my-app:1.0 ghcr.io/ada/my-app:{v}\ndocker tag my-app:1.0 ghcr.io/ada/my-app:latest\n"
             "docker push --all-tags ghcr.io/ada/my-app", check=["registry"], require=["docker push --all-tags|docker push -a"])


@template(M, "hard")
def deploy_remote(rng):
    return P("Ship the image to the GPU server through Docker Hub: log in as `ada` (token in `$DOCKERHUB_TOKEN`), "
             "push `my-app:1.0` as `ada/my-app:1.0`, then on `gpu-server` (over ssh) pull it and run it once "
             "(removed afterwards).", _built(rng, hub=True, host=True),
             f"{HUB_LOGIN}\ndocker tag my-app:1.0 ada/my-app:1.0\ndocker push ada/my-app:1.0\n"
             "ssh gpu-server docker pull ada/my-app:1.0\nssh gpu-server docker run --rm ada/my-app:1.0",
             check=["registry"], post="ssh gpu-server docker images --format '{{.Repository}}:{{.Tag}}'",
             require=["docker push", "ssh"])


@template(M, "hard")
def by_digest(rng):
    s = setup(_built(rng, gh=True), f"{GH_LOGIN} >/dev/null 2>&1", "docker tag my-app:1.0 ghcr.io/ada/my-app:1.0",
              "docker push -q ghcr.io/ada/my-app:1.0 >/dev/null")
    items = [
        ("docker inspect", "Tags can move; digests can't. Print the registry digest of `ghcr.io/ada/my-app:1.0` "
                           "(its first `RepoDigests` entry, `name@sha256:...`).",
         "docker inspect -f '{{index .RepoDigests 0}}' ghcr.io/ada/my-app:1.0"),
        ("docker run", "Run the pushed image **by its digest** (`ghcr.io/ada/my-app@sha256:...`, from `docker images "
                       "--digests` or `.RepoDigests`), removed afterwards.",
         "docker run --rm \"$(docker inspect -f '{{index .RepoDigests 0}}' ghcr.io/ada/my-app:1.0)\"",
         {"require": ["docker run"]}),
    ]
    return catalog(rng, items, s)


@template(M, "challenge")
def release(rng):
    v = rng.choice(["1.3.0", "2.0.0", "0.5.2"])
    s = setup(_built(rng, gh=True), f"echo {v} > VERSION")
    return P(f"**Release script.** Write `release.sh` that reads the version from `VERSION`, builds the image in this "
             f"folder as `ghcr.io/ada/my-app:<version>` and `ghcr.io/ada/my-app:latest` (one build, two `-t`), logs in "
             f"to ghcr.io with `gh auth token`, pushes both tags, and logs out. Run it.", s,
             "cat > release.sh <<'EOF'\n#!/bin/sh\nset -e\nv=$(cat VERSION)\ndocker build -t ghcr.io/ada/my-app:$v -t "
             "ghcr.io/ada/my-app:latest .\ngh auth token | docker login ghcr.io -u ada --password-stdin\ndocker push "
             "ghcr.io/ada/my-app:$v\ndocker push ghcr.io/ada/my-app:latest\ndocker logout ghcr.io\nEOF\n"
             "chmod +x release.sh\n./release.sh",
             check=["registry", "auth"], post="grep -c 'docker push' release.sh", require=[])


@template(M, "challenge")
def both_registries(rng):
    v = rng.choice(["1.0", "1.1", "3.0"])
    return P(f"**Mirror it.** Publish `my-app:1.0` as version {v} on **both** Docker Hub (`ada/my-app:{v}`, token in "
             f"`$DOCKERHUB_TOKEN`) and GHCR (`ghcr.io/ada/my-app:{v}`, `gh auth token`), then log out of both.",
             _built(rng, hub=True, gh=True),
             f"{HUB_LOGIN}\n{GH_LOGIN}\ndocker tag my-app:1.0 ada/my-app:{v}\ndocker tag my-app:1.0 ghcr.io/ada/my-app:{v}\n"
             f"docker push ada/my-app:{v}\ndocker push ghcr.io/ada/my-app:{v}\ndocker logout\ndocker logout ghcr.io",
             check=["registry", "auth"], require=["docker push", "docker logout"])
