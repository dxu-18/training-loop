"""12.01 Setup & first commands."""

from ._common import P, catalog, docker, setup, template

M = "12.01"


@template(M, "lite")
def first_commands(rng):
    items = [
        ("docker run", "Check that Docker works: run the `hello-world` image.", "docker run hello-world",
         {"post": "docker ps -a --format '{{.Image}} {{.State}}'; docker images --format '{{.Repository}}:{{.Tag}}'"}),
        ("docker version", "Print just the Docker Engine (server) version, using `--format`.",
         "docker version --format '{{.Server.Version}}'"),
        ("docker info", "Print how many CPUs the Docker host has (`docker info --format`).",
         "docker info --format '{{.NCPU}}'"),
        ("docker info", "Print the storage driver Docker uses (`docker info --format`).",
         "docker info --format '{{.Driver}}'"),
        ("docker --version", "Print the Docker client's version line.", "docker --version"),
    ]
    return catalog(rng, items, setup(docker()))


@template(M, "lite")
def help_pages(rng):
    items = [
        ("docker run --help", "Show the help for `docker run`.", "docker run --help"),
        ("docker --help", "Show docker's list of commands.", "docker --help"),
        ("docker images --help", "Show the help for `docker images`.", "docker images --help"),
    ]
    return catalog(rng, items, setup(docker()))


@template(M, "easy")
def not_in_group(rng):
    return P("`docker ps` fails with *permission denied* on `/var/run/docker.sock`: your user isn't in the `docker` "
             "group. Add `ada` to it (use sudo), then run `docker ps` with that group in effect without logging out "
             "(`newgrp docker` starts a shell with it; `sg docker -c 'docker ps'` works too).",
             setup(docker("nogroup")), "sudo usermod -aG docker ada\nsg docker -c 'docker ps'",
             post="grep '^docker:' /etc/group", require=["usermod"])


@template(M, "easy")
def daemon_down(rng):
    return P("Docker says it *cannot connect to the Docker daemon*. Start the docker service (it needs sudo), then "
             "run `hello-world`.", setup(docker("stopped")),
             "sudo systemctl start docker\ndocker run hello-world",
             post="systemctl is-active docker; docker ps -a --format '{{.Image}} {{.State}}'",
             require=["systemctl"])


@template(M, "medium")
def runtimes(rng):
    items = [
        ("docker info", "Print the container runtimes Docker knows about as JSON (`{{json .Runtimes}}`).",
         "docker info --format '{{json .Runtimes}}'"),
        ("docker info", "Print Docker's root directory (where images and volumes live).",
         "docker info --format '{{.DockerRootDir}}'"),
        ("docker version", "Print the client and server API versions, as `client=X server=Y` (one `--format`).",
         "docker version --format 'client={{.Client.ApiVersion}} server={{.Server.ApiVersion}}'"),
    ]
    return catalog(rng, items, setup(docker(), "#: gpus 2"))


@template(M, "medium")
def whoami_in_container(rng):
    img = rng.choice(["ubuntu:24.04", "debian:bookworm-slim", "alpine:3.20", "python:3.12-slim"])
    items = [
        ("docker run", f"Run `cat /etc/os-release` in a throwaway `{img}` container (removed when it exits).",
         f"docker run --rm {img} cat /etc/os-release", {"require": ["docker run --rm"],
                                                         "post": "docker ps -aq | wc -l"}),
        ("docker run", f"Print which user a `{img}` container runs as (`id`), removing the container afterwards.",
         f"docker run --rm {img} id", {"require": ["docker run --rm"]}),
        ("docker run", f"Print the hostname a `{img}` container named `probe` sees, keeping no container around.",
         f"docker run --rm --name probe --hostname probe {img} hostname", {"require": ["docker run --rm"]}),
    ]
    return catalog(rng, items, setup(docker(f"pull={img}")))


@template(M, "hard")
def install_docker(rng):
    return P("Docker isn't installed on this machine. Install the `docker.io` package with apt (sudo), make sure "
             "the service runs, and run `hello-world`.", setup(docker("missing")),
             "sudo apt install -y docker.io\nsudo systemctl start docker\ndocker run hello-world",
             post="docker ps -a --format '{{.Image}} {{.State}}'", require=["apt|apt-get"])


@template(M, "hard")
def context_info(rng):
    items = [
        ("docker system df", "Show how much disk Docker's images, containers, volumes and build cache use.",
         "docker system df", {"setup": "docker pull -q python:3.12-slim >/dev/null"}),
        ("docker info", "Print how many containers exist and how many are running, as `N total, M running`.",
         "docker info --format '{{.Containers}} total, {{.ContainersRunning}} running'",
         {"setup": "docker run -q hello-world >/dev/null; docker run -dq --name web nginx >/dev/null"}),
    ]
    return catalog(rng, items, setup(docker()))


@template(M, "challenge")
def fresh_machine(rng):
    return P("**A fresh GPU box.** Docker is installed but stopped, and you aren't in the `docker` group. Start the "
             "service, add `ada` to the group, and then (through `sg docker -c ...`) run `hello-world` and list "
             "the GPUs from a container: `nvidia/cuda:12.4.1-base-ubuntu22.04` running `nvidia-smi -L` with every "
             "GPU (`--rm`).", setup(docker("stopped", "nogroup"), "#: gpus 2"),
             "sudo systemctl start docker\nsudo usermod -aG docker ada\nsg docker -c 'docker run hello-world'\n"
             "sg docker -c 'docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi -L'",
             post="systemctl is-active docker; grep '^docker:' /etc/group; docker images --format "
                  "'{{.Repository}}:{{.Tag}}' | sort", require=["usermod", "systemctl"])


@template(M, "challenge")
def diagnose(rng):
    which = rng.choice(["stopped", "nogroup"])
    fix = "sudo systemctl start docker" if which == "stopped" else "sudo usermod -aG docker ada"
    run = "docker run hello-world" if which == "stopped" else "sg docker -c 'docker run hello-world'"
    return P("**What's wrong?** `docker run hello-world` fails on this machine. Read the error, fix the cause (one "
             "command with sudo), and run `hello-world` successfully (if the fix is a new group membership, run it "
             "through `sg docker -c '...'`).", setup(docker(which)), f"{fix}\n{run}",
             post="systemctl is-active docker; grep '^docker:' /etc/group; docker ps -a --format '{{.Image}} "
                  "{{.State}}'")
