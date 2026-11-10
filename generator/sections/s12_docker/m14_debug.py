"""12.14 Debugging & cleanup."""

from ._common import WORDS, P, catalog, docker, dockerfile, files, heredoc, setup, template

M = "12.14"
PY = "python:3.12-slim"
STATE = "docker inspect -f '{{.State.ExitCode}} {{.State.OOMKilled}}'"


def _app(rng, *lines, word=None, extra=None):
    word = word or rng.choice(WORDS)
    fs = {"app.py": f'print("{word} ready")\n',
          "Dockerfile": dockerfile(*(lines or (f"FROM {PY}", "WORKDIR /app", "COPY app.py .",
                                               'CMD ["python", "app.py"]')))}
    fs.update(extra or {})
    return setup(docker(f"pull={PY},nginx,mlgym/memhog:1.0,mlgym/crasher:1.0"), "#: cwd ~/app", files(fs))


@template(M, "lite")
def read_logs(rng):
    n = rng.choice([2, 3, 5])
    s = setup(_app(rng), "docker run -dq --name api nginx >/dev/null", "sleep 1",
              *[f"docker exec api curl -s -o /dev/null localhost/{p}"
                for p in ("", "health", "missing", "docs", "favicon.ico", "index.html")])
    items = [
        ("docker logs", f"Show only the last {n} lines of `api`'s logs.", f"docker logs --tail {n} api"),
        ("docker logs", "Show `api`'s log lines for requests that got a 404.", "docker logs api 2>&1 | grep ' 404 '"),
        ("docker ps", "List the names of all containers that have **exited** (`--filter status=exited`).",
         "docker ps -a --filter status=exited --format '{{.Names}}'",
         {"setup": "docker run -q --name job1 ubuntu:24.04 true >/dev/null 2>&1; docker run -q --name job2 "
                   "mlgym/crasher:1.0 >/dev/null 2>&1", "unordered": True}),
    ]
    return catalog(rng, items, s)


@template(M, "lite")
def override_entrypoint(rng):
    s = setup(_app(rng), "docker build -q -t my-app:1.0 . >/dev/null")
    items = [
        ("docker run --entrypoint", "Instead of the image's default command, list `/app` in a throwaway `my-app:1.0` "
                                    "container: override the entrypoint with `ls` and pass `/app`.",
         "docker run --rm --entrypoint ls my-app:1.0 /app"),
        ("docker run --entrypoint", "Start `my-app:1.0` with `sh` as the entrypoint and run `-c 'python --version; pwd'` "
                                    "instead of the app (removed afterwards).",
         "docker run --rm --entrypoint sh my-app:1.0 -c 'python --version; pwd'"),
        ("docker exec", "`my-app:1.0` is running as `app` (kept alive with `sleep infinity`). Print `/app/app.py` "
                        "from inside it.", "docker exec app cat /app/app.py",
         {"setup": "docker run -dq --name app my-app:1.0 sleep infinity >/dev/null"}),
    ]
    return catalog(rng, items, s)


@template(M, "easy")
def exit_code(rng):
    items = [
        ("docker inspect", "The training job `trainer` (`mlgym/memhog:1.0`, run with `-m 512m`) died. Print its exit "
                           "code and whether it was OOM-killed: `.State.ExitCode` and `.State.OOMKilled`.",
         f"{STATE} trainer",
         {"setup": "docker run --name trainer -m 512m mlgym/memhog:1.0 >/dev/null 2>&1"}),
        ("docker inspect", "The container `ext` (`mlgym/crasher:1.0` running `segv`) died. Print its exit code and "
                           "OOM flag (`.State.ExitCode .State.OOMKilled`).", f"{STATE} ext",
         {"setup": "docker run --name ext mlgym/crasher:1.0 segv >/dev/null 2>&1"}),
        ("docker ps", "Print each container's name and status (`{{.Names}} {{.Status}}`) for the containers that "
                      "exited with code 137 (`--filter exited=137`).",
         "docker ps -a --filter exited=137 --format '{{.Names}}'",
         {"setup": "docker run --name big -m 256m mlgym/memhog:1.0 >/dev/null 2>&1\ndocker run --name ok "
                   "mlgym/memhog:1.0 >/dev/null 2>&1"}),
    ]
    return catalog(rng, items, _app(rng))


@template(M, "easy")
def disk_usage(rng):
    s = setup(_app(rng), "docker build -q -t my-app:1.0 . >/dev/null", "docker volume create data >/dev/null",
              "docker run -q --name old nginx true >/dev/null 2>&1")
    items = [
        ("docker system df", "Show how much disk Docker uses for images, containers, volumes and build cache.",
         "docker system df"),
        ("docker image prune", "Remove the **dangling** images (untagged leftovers of rebuilds), without asking for "
                               "confirmation.", "docker image prune -f",
         {"setup": "echo '#' >> app.py\ndocker build -q -t my-app:1.0 . >/dev/null", "check": ["images"]}),
        ("docker container prune", "Remove all stopped containers, without a confirmation prompt.",
         "docker container prune -f", {"check": ["containers"]}),
    ]
    return catalog(rng, items, s)


@template(M, "medium")
def exits_immediately(rng):
    word = rng.choice(WORDS)
    s = setup(_app(rng, f"FROM {PY}", "WORKDIR /srv", "COPY app.py /app/", 'CMD ["python", "app.py"]', word=word),
              "docker build -q -t my-app:1.0 . >/dev/null")
    return P("`docker run my-app:1.0` exits immediately with code 2. Find out why by looking inside the image with a "
             "shell instead of the default command (`--entrypoint sh` and `-c 'pwd; ls /app'`), then fix the "
             "Dockerfile so the working directory is where `app.py` was copied, rebuild `my-app:1.0`, and run it.", s,
             "docker run --rm --entrypoint sh my-app:1.0 -c 'pwd; ls /app'\n" + heredoc("Dockerfile", dockerfile(f"FROM {PY}", "WORKDIR /app", "COPY app.py /app/",
                                                'CMD ["python", "app.py"]'))
             + "\ndocker build -t my-app:1.0 .\ndocker run --rm my-app:1.0",
             post="docker run --rm my-app:1.0", require=["docker run --entrypoint", "docker build"])


@template(M, "medium")
def why_killed(rng):
    limit = rng.choice(["512m", "1g"])
    s = setup(_app(rng), f"docker run --name trainer -m {limit} mlgym/memhog:1.0 >/dev/null 2>&1")
    return P(f"`trainer` (`mlgym/memhog:1.0`, run with `-m {limit}`) stopped by itself. Find out why (its exit code and "
             f"`OOMKilled`; the logs show how much memory it loads), then run it again in the foreground as "
             f"`trainer2` with a 2 GB memory limit so it finishes.", s,
             f"{STATE} trainer\ndocker logs trainer\ndocker run --name trainer2 -m 2g mlgym/memhog:1.0",
             post=f"{STATE} trainer2", require=["docker inspect|docker ps", "docker run -m|docker run --memory"])


@template(M, "hard")
def exec_failures(rng):
    items = [
        ("chmod", "`docker run my-app:1.0` fails with exit **126**: *exec: \"./start.sh\": permission denied*. The "
                  "script was copied without its executable bit. Fix the Dockerfile (`COPY --chmod=755`, or `RUN chmod "
                  "+x start.sh`), rebuild `my-app:1.0`, and run it.",
         {"lines": (f"FROM {PY}", "WORKDIR /app", "COPY app.py start.sh ./", 'ENTRYPOINT ["./start.sh"]'),
          "fixed": (f"FROM {PY}", "WORKDIR /app", "COPY app.py ./", "COPY --chmod=755 start.sh ./",
                    'ENTRYPOINT ["./start.sh"]')}),
        ("docker build", "`docker run my-app:1.0` fails with exit **127**: *exec: \"pyhton\": executable file not "
                         "found in $PATH*. Fix the Dockerfile's command, rebuild `my-app:1.0`, and run it.",
         {"lines": (f"FROM {PY}", "WORKDIR /app", "COPY app.py .", 'CMD ["pyhton", "app.py"]'),
          "fixed": (f"FROM {PY}", "WORKDIR /app", "COPY app.py .", 'CMD ["python", "app.py"]')}),
    ]
    feat, prompt, spec = rng.choice(items)
    word = rng.choice(WORDS)
    extra = {"start.sh": "#!/bin/sh\nexec python app.py \"$@\"\n"}
    s = setup(_app(rng, *spec["lines"], word=word, extra=extra), "chmod -x start.sh",
              "docker build -q -t my-app:1.0 . >/dev/null")
    return P(prompt, s, heredoc("Dockerfile", dockerfile(*spec["fixed"])) +
             "\ndocker build -t my-app:1.0 .\ndocker run --rm my-app:1.0", post="docker run --rm my-app:1.0",
             require=["docker build"])


@template(M, "hard")
def slow_stop(rng):
    s = setup(_app(rng, "FROM nginx", "CMD nginx -g 'daemon off;'"), "docker build -q -t web:1 . >/dev/null",
              "docker run -dq --name web web:1 >/dev/null")
    return P("`docker stop web` hangs for 10 seconds and the container ends with code 137: its shell-form `CMD` makes "
             "`/bin/sh` PID 1, which never passes SIGTERM on to nginx, so Docker has to SIGKILL it. Switch the "
             "Dockerfile to **exec form** (`CMD [\"nginx\", \"-g\", \"daemon off;\"]`), rebuild as `web:2`, run it "
             "detached as `web2`, stop it, and print its exit code.", s,
             heredoc("Dockerfile", dockerfile("FROM nginx", 'CMD ["nginx", "-g", "daemon off;"]')) +
             "\ndocker build -t web:2 .\ndocker run -d --name web2 web:2\ndocker stop web2\n"
             "docker inspect -f '{{.State.ExitCode}}' web2",
             post="docker inspect -f '{{.State.ExitCode}}' web2", require=["docker build", "docker stop"])


@template(M, "challenge")
def reclaim_safely(rng):
    s = setup(_app(rng), "docker build -q -t my-app:1.0 . >/dev/null", "echo '#' >> app.py",
              "docker build -q -t my-app:1.0 . >/dev/null", "echo '##' >> app.py",
              "docker build -q -t my-app:1.0 . >/dev/null",
              "docker volume create pgdata >/dev/null", "docker run -q --rm -v pgdata:/data nginx sh -c 'echo rows > "
                                                        "/data/table' >/dev/null 2>&1",
              "docker run -q --name scratch1 nginx true >/dev/null 2>&1",
              "docker run -q --name scratch2 mlgym/crasher:1.0 >/dev/null 2>&1",
              "docker run -dq --name web nginx >/dev/null")
    return P("**Reclaim disk space safely.** Remove the stopped containers, the dangling images, unused networks and "
             "the build cache, but keep the running `web` container, every tagged image, and the unused volume "
             "`pgdata` (it holds the only copy of a database). `docker system prune -a --volumes` would be a disaster "
             "here. Check `docker system df` before and after.", s,
             "docker system df\ndocker system prune -f\ndocker system df",
             check=["containers", "volumes"],
             post="docker images --format '{{.Repository}}:{{.Tag}}' | sort; docker images -f dangling=true -q | wc -l; "
                  "docker run --rm -v pgdata:/data nginx cat /data/table",
             require=["docker system prune|docker container prune|docker image prune"])


@template(M, "challenge")
def postmortem(rng):
    s = setup(_app(rng), "docker run --name etl -m 256m mlgym/memhog:1.0 >/dev/null 2>&1",
              "docker run --name native mlgym/crasher:1.0 segv >/dev/null 2>&1",
              "docker run --name report ubuntu:24.04 true >/dev/null 2>&1",
              "docker run --name typo ubuntu:24.04 pyhton >/dev/null 2>&1",
              "docker run -dq --name web nginx >/dev/null")
    return P("**Post-mortem.** Several jobs ran overnight. For every container that has **exited**, print one line: "
             "`.Name`, `.State.ExitCode` and `.State.OOMKilled` as `docker inspect -f` prints them (e.g. `/etl 137 "
             "true`). Loop over `docker ps -aq --filter status=exited` (the running `web` must not appear). Then explain "
             "each code to yourself: 137 OOM, 139 segfault, 127 not found, 0 fine.", s,
             "for c in $(docker ps -aq --filter status=exited); do docker inspect -f '{{.Name}} {{.State.ExitCode}} "
             "{{.State.OOMKilled}}' \"$c\"; done", unordered=True, require=["docker inspect"])
