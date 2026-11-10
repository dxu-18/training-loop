"""12.03 Managing containers."""

from ._common import NAMES, WORDS, P, catalog, docker, files, q, setup, template

M = "12.03"


def _web(name="web", port=8080):
    return f"docker run -dq --name {name} -p {port}:80 nginx >/dev/null"


def _stack(rng):
    """A few containers in different states."""
    return setup(docker("pull=nginx,ubuntu:24.04,python:3.12-slim"), _web("web"),
                 "docker run -dq --name worker ubuntu:24.04 sleep infinity >/dev/null",
                 "docker run -q --name job ubuntu:24.04 sh -c 'echo done' >/dev/null",
                 "docker run -q --name broken ubuntu:24.04 sh -c 'exit 3' >/dev/null || true",
                 "curl -s localhost:8080 >/dev/null; curl -s localhost:8080/missing >/dev/null")


@template(M, "lite")
def listing(rng):
    items = [
        ("docker ps", "Print the names of the running containers, one per line (`docker ps --format`).",
         "docker ps --format '{{.Names}}'", {"unordered": True}),
        ("docker ps -a", "Print every container's name and state (running or not) as `NAME STATE`.",
         "docker ps -a --format '{{.Names}} {{.State}}'", {"unordered": True}),
        ("docker ps -q", "Print how many containers are running (count the ids `docker ps -q` prints).",
         "docker ps -q | wc -l"),
        ("docker ps --filter", "Print the names of the containers that have exited.",
         "docker ps -a --filter status=exited --format '{{.Names}}'", {"unordered": True,
                                                                        "require": ["docker ps --filter|docker ps -f"]}),
    ]
    return catalog(rng, items, _stack(rng))


@template(M, "lite")
def lifecycle(rng):
    items = [
        ("docker stop", "Stop the `worker` container.", "docker stop worker"),
        ("docker rm", "Remove the stopped containers `job` and `broken` (one command).", "docker rm job broken"),
        ("docker rm -f", "Remove the **running** container `worker` in one command (force).", "docker rm -f worker"),
        ("docker restart", "Restart the `web` container.", "docker restart web", {"check": ["containers"],
                                                                                 "post": "curl -so /dev/null -w '%{http_code}\\n' localhost:8080"}),
        ("docker start", "Start the exited container `job` again.", "docker start job"),
        ("docker kill", "Kill the `worker` container immediately (SIGKILL).", "docker kill worker"),
    ]
    return catalog(rng, items, _stack(rng), check=["containers"])


@template(M, "easy")
def logs(rng):
    n = rng.randint(2, 4)
    items = [
        ("docker logs", "Show `web`'s logs (stdout and stderr).", "docker logs web"),
        ("docker logs", f"Show only the last {n} lines of `web`'s logs.", f"docker logs --tail {n} web",
         {"require": ["docker logs --tail|docker logs -n"]}),
        ("docker logs", "Show `job`'s output (it has already exited).", "docker logs job"),
        ("docker logs", "Find the requests in `web`'s access log that got a 404 (grep the logs for ` 404 `).",
         "docker logs web 2>/dev/null | grep -c ' 404 '"),
    ]
    return catalog(rng, items, _stack(rng))


@template(M, "easy")
def exec_inside(rng):
    items = [
        ("docker exec", "List the files nginx serves (`/usr/share/nginx/html`) inside the running `web` container.",
         "docker exec web ls /usr/share/nginx/html"),
        ("docker exec", "Print the `server_name` line of `/etc/nginx/conf.d/default.conf` inside `web`.",
         "docker exec web grep server_name /etc/nginx/conf.d/default.conf"),
        ("docker exec -it", "Open a shell in `web` (`docker exec -it ... bash`), print its hostname, and leave.",
         "docker exec -it web bash\nhostname\nexit", {"require": ["docker exec -it|docker exec -i"]}),
        ("docker exec", "Create `/usr/share/nginx/html/ping.html` containing `pong` inside `web` (use `sh -c`).",
         "docker exec web sh -c 'echo pong > /usr/share/nginx/html/ping.html'",
         {"post": "curl -s localhost:8080/ping.html"}),
    ]
    return catalog(rng, items, _stack(rng))


@template(M, "medium")
def copy_files(rng):
    word = rng.choice(WORDS)
    s = setup(_stack(rng), f"echo '<h1>{word}</h1>' > ~/page.html", "head -c 4000 /dev/urandom > ~/model.pt")
    items = [
        ("docker cp", "Copy nginx's config `/etc/nginx/nginx.conf` out of `web` into your home directory.",
         "docker cp web:/etc/nginx/nginx.conf ~/", {"post": "head -3 ~/nginx.conf"}),
        ("docker cp", "Copy `~/page.html` into `web` as `/usr/share/nginx/html/index.html` (replacing the default "
                      "page).", "docker cp ~/page.html web:/usr/share/nginx/html/index.html",
         {"post": "curl -s localhost:8080"}),
        ("docker cp", "Copy `~/model.pt` into the `worker` container's `/models/` directory (create the directory "
                      "first with exec).", "docker exec worker mkdir -p /models\ndocker cp ~/model.pt worker:/models/",
         {"post": "docker exec worker ls -l /models | awk '{print $5, $9}'"}),
        ("docker cp", "`job` has exited, but its `/etc/hostname` is still there. Copy it out to `~/job-hostname`.",
         "docker cp job:/etc/hostname ~/job-hostname", {"post": "wc -c < ~/job-hostname"}),
    ]
    return catalog(rng, items, s)


@template(M, "medium")
def inspect(rng):
    items = [
        ("docker inspect", "Print `web`'s state (`running`, `exited`...) with `docker inspect -f`.",
         "docker inspect -f '{{.State.Status}}' web"),
        ("docker inspect", "Print `broken`'s exit code.", "docker inspect -f '{{.State.ExitCode}}' broken"),
        ("docker inspect", "Print `web`'s IP address on the default bridge network.",
         "docker inspect -f '{{.NetworkSettings.IPAddress}}' web"),
        ("docker port", "Show which host port `web`'s port 80 is published on.", "docker port web 80"),
        ("docker inspect", "Print the image `worker` was created from (as given to `docker run`).",
         "docker inspect -f '{{.Config.Image}}' worker"),
        ("docker inspect", "Print `web`'s restart policy name.", "docker inspect -f '{{.HostConfig.RestartPolicy.Name}}' web"),
    ]
    return catalog(rng, items, _stack(rng))


@template(M, "hard")
def prune(rng):
    items = [
        ("docker container prune", "Remove every stopped container in one command (no prompt).",
         "docker container prune -f"),
        ("docker rm", "Remove every **exited** container using `docker ps -aq --filter` inside `$( )`.",
         "docker rm $(docker ps -aq --filter status=exited)", {"require": ["docker rm", "$("]}),
        ("docker stop", "Stop every running container with one command line (`docker stop $(docker ps -q)`).",
         "docker stop $(docker ps -q)", {"require": ["docker stop", "$("]}),
    ]
    return catalog(rng, items, _stack(rng), check=["containers"])


@template(M, "hard")
def top_stats(rng):
    items = [
        ("docker top", "Show the processes running inside `worker`.", "docker top worker"),
        ("docker stats", "Print a one-time snapshot (no streaming) of each running container's name and CPU% "
                         "(`--format '{{.Name}} {{.CPUPerc}}'`).",
         "docker stats --no-stream --format '{{.Name}} {{.CPUPerc}}'", {"unordered": True}),
        ("docker top", "`ubuntu` images have no `ps`. Show what runs inside `worker` with `docker top`, printing "
                       "only the command (`awk 'NR>1{print $8, $9}'`).",
         "docker top worker | awk 'NR>1{print $8, $9}'", {"require": ["docker top"]}),
    ]
    return catalog(rng, items, _stack(rng))


@template(M, "challenge")
def recover_file(rng):
    word = rng.choice(WORDS)
    s = setup(docker("pull=python:3.12-slim"),
              f"docker run -q --name trainer python:3.12-slim sh -c 'mkdir -p /out && echo \"loss=0.{rng.randint(10, 99)} "
              f"tag={word}\" > /out/metrics.txt' >/dev/null")
    return P("**Rescue the results.** A training container `trainer` wrote `/out/metrics.txt` and exited; nothing was "
             "mounted. Copy the file to `~/results/metrics.txt` (create the folder), then remove the container.", s,
             "mkdir -p ~/results\ndocker cp trainer:/out/metrics.txt ~/results/\ndocker rm trainer",
             check=["containers-all"], post="cat ~/results/metrics.txt", require=["docker cp"])


@template(M, "challenge")
def clean_slate(rng):
    keep = rng.choice(["web", "worker"])
    return P(f"**Tidy up.** Remove every container except `{keep}` (running or not), and leave `{keep}` running.",
             _stack(rng), f"docker ps -a --format '{{{{.Names}}}}' | grep -vx {keep} | xargs docker rm -f",
             check=["containers-all"])
