"""12.08 Volumes & bind mounts."""

from ._common import WORDS, P, catalog, docker, files, setup, template

M = "12.08"
U = "ubuntu:24.04"


@template(M, "lite")
def volume_commands(rng):
    name = rng.choice(["hf-cache", "pgdata", "datasets", "models"])
    s = setup(docker(f"pull={U}"), "docker volume create old-cache >/dev/null; docker volume create scratch "
                                   ">/dev/null")
    items = [
        ("docker volume create", f"Create a named volume `{name}`.", f"docker volume create {name}",
         {"check": ["volumes"]}),
        ("docker volume ls", "List the volumes' names only.", "docker volume ls -q", {"unordered": True}),
        ("docker volume rm", "Remove the volume `scratch`.", "docker volume rm scratch", {"check": ["volumes"]}),
        ("docker volume inspect", "Print where the volume `old-cache` lives on the host (`Mountpoint`).",
         "docker volume inspect -f '{{.Mountpoint}}' old-cache"),
        ("docker volume prune", "Remove every volume no container uses (`-a` includes named ones; no prompt).",
         "docker volume prune -a -f", {"check": ["volumes"]}),
    ]
    return catalog(rng, items, s)


@template(M, "lite")
def named_volume(rng):
    word = rng.choice(WORDS)
    return P(f"Write `{word}` into `/cache/token.txt` from one throwaway `{U}` container using a named volume "
             f"`hf-cache` mounted at `/cache`, then read it back from a **second** throwaway container.",
             setup(docker(f"pull={U}")),
             f"docker run --rm -v hf-cache:/cache {U} sh -c 'echo {word} > /cache/token.txt'\n"
             f"docker run --rm -v hf-cache:/cache {U} cat /cache/token.txt",
             check=["volume=hf-cache"], require=["docker run -v|docker run --mount"])


@template(M, "easy")
def ro_rw(rng):
    n = rng.randint(3, 9)
    s = setup(docker("pull=python:3.12-slim"), "#: cwd ~/exp",
              files({"data/numbers.txt": "\n".join(str(i) for i in range(n)) + "\n",
                     "count.py": "import pathlib\nn = len(pathlib.Path('/data/numbers.txt').read_text().split())\n"
                                 "pathlib.Path('/outputs/count.txt').write_text(str(n) + '\\n')\nprint('counted', n)\n"}))
    return P("Run `count.py` in a throwaway `python:3.12-slim` container with `./data` mounted **read-only** at "
             "`/data`, `./outputs` mounted read-write at `/outputs`, and the script itself mounted at `/count.py`.",
             s, 'mkdir -p outputs\ndocker run --rm -v "$(pwd)/data":/data:ro -v "$(pwd)/outputs":/outputs -v '
                '"$(pwd)/count.py":/count.py python:3.12-slim python /count.py',
             post="cat outputs/count.txt; docker run --rm -v \"$(pwd)/data\":/data:ro python:3.12-slim touch /data/x "
                  "2>&1 | tail -1", require=["docker run -v|docker run --mount"])


@template(M, "easy")
def mount_flag(rng):
    s = setup(docker(f"pull={U}"), "#: cwd ~/proj", files({"notes.md": "hello\n"}))
    items = [
        ("docker run --mount", "Bind-mount the current folder at `/workspace` with the explicit long form "
                               "(`--mount type=bind,...`) and list it from a throwaway `ubuntu:24.04` container.",
         f'docker run --rm --mount type=bind,source="$(pwd)",target=/workspace {U} ls /workspace'),
        ("docker run --tmpfs", "Run a throwaway `ubuntu:24.04` container with an in-memory `/scratch` (tmpfs) and "
                               "show it with `ls -ld /scratch`.",
         f"docker run --rm --tmpfs /scratch {U} ls -ld /scratch"),
        ("docker run --mount", "Mount a named volume `cache` at `/cache` read-only with `--mount` (long form) and "
                               "list `/cache` from a throwaway `ubuntu:24.04` container.",
         f"docker run --rm --mount type=volume,source=cache,target=/cache,readonly {U} ls /cache"),
    ]
    return catalog(rng, items, s)


@template(M, "medium")
def where_it_lives(rng):
    word = rng.choice(WORDS)
    s = setup(docker(f"pull={U}"), f"docker run --rm -q -v models:/m {U} sh -c 'echo {word} > /m/best.txt' >/dev/null")
    return P("Find where the volume `models` is stored on the host (`docker volume inspect`), then read "
             "`best.txt` straight from that directory (it belongs to root: use sudo).", s,
             "docker volume inspect -f '{{.Mountpoint}}' models\nsudo cat /var/lib/docker/volumes/models/_data/best.txt",
             require=["docker volume inspect"])


@template(M, "medium")
def compare_mounts(rng):
    s = setup(docker(f"pull={U}"), "#: cwd ~/proj", files({"a.txt": "a\n"}),
              f"docker run -dq --name box -v data:/data -v \"$(pwd)\":/src:ro {U} sleep infinity >/dev/null")
    items = [
        ("docker inspect", "Print each mount of `box` as `TYPE SOURCE -> DESTINATION` (range over `.Mounts`).",
         "docker inspect -f '{{range .Mounts}}{{.Type}} {{.Source}} -> {{.Destination}}{{println}}{{end}}' box"),
        ("docker inspect", "Print whether each of `box`'s mounts is writable, as `DESTINATION RW`.",
         "docker inspect -f '{{range .Mounts}}{{.Destination}} {{.RW}}{{println}}{{end}}' box"),
    ]
    return catalog(rng, items, s)


@template(M, "hard")
def backup_volume(rng):
    files_ = rng.sample(["a.bin", "b.bin", "c.bin", "d.bin"], 2)
    s = setup(docker(f"pull={U}"), "#: cwd ~/backups",
              f"docker run --rm -q -v pgdata:/data {U} sh -c 'echo x > /data/{files_[0]}; echo y > /data/{files_[1]}' "
              ">/dev/null")
    return P("Back up the volume `pgdata` to `./pgdata.tgz` on the host: a throwaway `ubuntu:24.04` container mounts "
             "the volume at `/data` and the current folder at `/backup`, and runs `tar czf /backup/pgdata.tgz -C /data .`.",
             s, f'docker run --rm -v pgdata:/data -v "$(pwd)":/backup {U} tar czf /backup/pgdata.tgz -C /data .',
             post="tar tzf pgdata.tgz | sort", require=["docker run -v|docker run --mount"])


@template(M, "hard")
def restore_volume(rng):
    s = setup(docker(f"pull={U}"), "#: cwd ~/backups",
              f"docker run --rm -q -v src:/data {U} sh -c 'echo weights > /data/model.bin' >/dev/null",
              f"docker run --rm -q -v src:/data -v \"$(pwd)\":/b {U} tar czf /b/models.tgz -C /data . >/dev/null",
              "docker volume rm src >/dev/null")
    return P("`models.tgz` is a backup of a volume. Restore it into a new volume `models` (extract it in a throwaway "
             "`ubuntu:24.04` container with the volume at `/data` and this folder at `/backup`).", s,
             f'docker volume create models\ndocker run --rm -v models:/data -v "$(pwd)":/backup {U} tar xzf '
             "/backup/models.tgz -C /data", check=["volume=models"], require=["docker run -v|docker run --mount"])


@template(M, "challenge")
def persist_db(rng):
    table = rng.choice(["runs", "metrics", "experiments"])
    s = setup(docker("pull=postgres:16"))
    return P(f"**Survive a restart.** Run Postgres 16 detached as `db` (password `secret`) with its data in a named "
             f"volume `pgdata` (at `/var/lib/postgresql/data`), create a table `{table}` (`psql -U postgres -c "
             f"'CREATE TABLE {table} (id int)'` via `docker exec`), then remove the container and start a **new** "
             f"`db` on the same volume: the table must still be there.", s,
             "docker run -d --name db -e POSTGRES_PASSWORD=secret -v pgdata:/var/lib/postgresql/data postgres:16\n"
             f"sleep 3\ndocker exec db psql -U postgres -c 'CREATE TABLE {table} (id int)'\ndocker rm -f db\n"
             "docker run -d --name db -e POSTGRES_PASSWORD=secret -v pgdata:/var/lib/postgresql/data postgres:16\n"
             "sleep 3", check=["containers", "volumes"],
             post="docker exec db psql -U postgres -c '\\dt' | grep -o 'public | [a-z]*'", require=["docker exec"])


@template(M, "challenge")
def shared_cache(rng):
    word = rng.choice(WORDS)
    return P(f"**One cache, two jobs.** A named volume `hf-cache` should be shared by two detached `ubuntu:24.04` "
             f"containers, `downloader` (which writes `{word}` to `/cache/model.txt` and then sleeps: `sh -c 'echo "
             f"{word} > /cache/model.txt && sleep infinity'`) and `reader` (which only needs to read it: mount it "
             f"read-only, `sleep infinity`). Then print the file from `reader` with exec.",
             setup(docker(f"pull={U}")),
             f"docker run -d --name downloader -v hf-cache:/cache {U} sh -c 'echo {word} > /cache/model.txt && sleep "
             f"infinity'\ndocker run -d --name reader -v hf-cache:/cache:ro {U} sleep infinity\nsleep 1\n"
             "docker exec reader cat /cache/model.txt", check=["containers", "volume=hf-cache"],
             require=["docker exec"])
