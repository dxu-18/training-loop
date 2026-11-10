"""12.02 Running containers."""

from ._common import NAMES, WORDS, P, catalog, docker, files, q, setup, template

M = "12.02"
PY = "python:3.12-slim"


@template(M, "lite")
def run_basics(rng):
    word = rng.choice(WORDS)
    name = rng.choice(NAMES)
    port = rng.choice([8080, 8081, 9000, 3000])
    items = [
        ("docker run", f"Run `echo {word}` in an `ubuntu:24.04` container.", f"docker run ubuntu:24.04 echo {word}",
         {"post": "docker ps -a --format '{{.Image}} {{.State}} {{.Command}}'"}),
        ("docker run --rm", f"Run `echo {word}` in an `ubuntu:24.04` container that is deleted when it exits.",
         f"docker run --rm ubuntu:24.04 echo {word}", {"post": "docker ps -aq | wc -l"}),
        ("docker run -d", f"Start nginx in the background as a container named `{name}`, publishing host port {port} "
                          f"to the container's port 80.", f"docker run -d --name {name} -p {port}:80 nginx",
         {"check": ["containers"], "post": f"curl -s localhost:{port} | grep -o '<h1>.*</h1>'",
          "require": ["docker run -d|docker run --detach", "docker run -p|docker run --publish"]}),
        ("docker run -e", f"Run `{PY}` with the environment variable `MODE={word}` set, printing it with "
                          f"`python -c \"import os; print(os.environ['MODE'])\"` (remove the container afterwards).",
         f"docker run --rm -e MODE={word} {PY} python -c \"import os; print(os.environ['MODE'])\""),
        ("docker run -w", f"Print the working directory of an `ubuntu:24.04` container started in `/srv/{word}` "
                          f"(`pwd`, with `-w`, removed afterwards).", f"docker run --rm -w /srv/{word} ubuntu:24.04 pwd"),
    ]
    return catalog(rng, items, setup(docker("pull=ubuntu:24.04,nginx,python:3.12-slim")))


@template(M, "lite")
def interactive(rng):
    n = rng.randint(2, 9)
    items = [
        ("docker run -it", f"Start Python 3.12 interactively without installing it (`-it`, removed on exit), compute "
                           f"`2**{n}` at the `>>>` prompt, then quit.",
         f"docker run -it --rm {PY} python\n2**{n}\nexit()", {"require": ["docker run -it|docker run -i"]}),
        ("docker run -it", "Open a bash shell in a throwaway `ubuntu:24.04` container, print `/etc/os-release`'s "
                           "PRETTY_NAME line with grep, and exit.",
         "docker run -it --rm ubuntu:24.04 bash\ngrep PRETTY_NAME /etc/os-release\nexit",
         {"require": ["docker run -it|docker run -i"]}),
    ]
    return catalog(rng, items, setup(docker(f"pull={PY},ubuntu:24.04")), post="docker ps -aq | wc -l")


@template(M, "easy")
def mount_and_run(rng):
    word = rng.choice(WORDS)
    n = rng.randint(3, 9)
    script = f'import sys\ntotal = sum(range({n}))\nprint("{word}", total)\nopen("result.txt", "w").write(str(total))\n'
    return P(f"Run `script.py` from the current folder in a `{PY}` container: mount the current directory at "
             f"`/workspace`, work there, and remove the container when it's done. It writes `result.txt` next to "
             f"the script.", setup(docker(f"pull={PY}"), "#: cwd ~/proj", files({"script.py": script})),
             f'docker run --rm -v "$(pwd)":/workspace -w /workspace {PY} python script.py',
             post="cat result.txt; docker ps -aq | wc -l", require=["docker run -v|docker run --mount"])


@template(M, "easy")
def env_file(rng):
    level = rng.choice(["debug", "info", "warning"])
    s = setup(docker("pull=python:3.12-slim"), "#: cwd ~/svc",
              files({".env": f"LOG_LEVEL={level}\nMODEL=resnet18\nBATCH=32\n",
                     "show.py": 'import os\nfor k in ("LOG_LEVEL", "MODEL", "BATCH", "RUN"):\n    print(k, os.environ.get(k))\n'}))
    run = rng.choice(["exp1", "exp7", "baseline"])
    return P(f"Run `show.py` (mounted at `/w`, working there) in a throwaway `{PY}` container with all the variables "
             f"from `.env` plus `RUN={run}`.", s,
             f'docker run --rm --env-file .env -e RUN={run} -v "$(pwd)":/w -w /w {PY} python show.py',
             require=["docker run --env-file"])


@template(M, "medium")
def serve_folder(rng):
    port = rng.choice([8080, 8088, 9090])
    title = rng.choice(["Lab notes", "Results", "Model card", "Docs"])
    s = setup(docker("pull=nginx"), "#: cwd ~/site",
              files({"site/index.html": f"<h1>{title}</h1>\n", "site/about.html": "<p>about</p>\n"}))
    return P(f"Serve the `site/` folder with nginx on host port {port}: a detached container `site` with `./site` "
             f"mounted **read-only** at `/usr/share/nginx/html`.", s,
             f'docker run -d --name site -p {port}:80 -v "$(pwd)/site":/usr/share/nginx/html:ro nginx',
             check=["containers"], post=f"curl -s localhost:{port}; curl -s localhost:{port}/about.html",
             require=["docker run -v|docker run --mount"])


@template(M, "medium")
def owned_files(rng):
    fname = rng.choice(["out.txt", "metrics.json", "report.csv"])
    s = setup(docker("pull=python:3.12-slim"), "#: cwd ~/work")
    return P(f"Create `{fname}` in the current folder from a `{PY}` container (mount `.` at `/w`, work there, "
             f"`touch {fname}`) so that the file is owned by **you**, not root.", s,
             f'docker run --rm --user "$(id -u):$(id -g)" -v "$(pwd)":/w -w /w {PY} touch {fname}',
             post=f"stat -c '%U %G %n' {fname}", require=["docker run --user|docker run -u"])


@template(M, "hard")
def gpus(rng):
    a, b = sorted(rng.sample(range(4), 2))
    items = [
        ("docker run --gpus", "List the GPUs a `nvidia/cuda:12.4.1-base-ubuntu22.04` container sees when given all of "
                              "them (`nvidia-smi -L`, removed afterwards).",
         "docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi -L"),
        ("docker run --gpus", f"List the GPUs a CUDA container sees when given only GPUs {a} and {b} "
                              f"(`nvidia-smi -L` in `nvidia/cuda:12.4.1-base-ubuntu22.04`, removed afterwards).",
         f"docker run --rm --gpus '\"device={a},{b}\"' nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi -L"),
        ("docker run --gpus", "In `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` with every GPU, print "
                              "`torch.cuda.device_count()`.",
         "docker run --rm --gpus all pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime python -c \"import torch; "
         "print(torch.cuda.device_count())\""),
    ]
    return catalog(rng, items, setup(docker("pull=nvidia/cuda:12.4.1-base-ubuntu22.04,pytorch/pytorch:2.5.1-cuda12.4-"
                                            "cudnn9-runtime"), "#: gpus 4"))


@template(M, "hard")
def limits(rng):
    cpus = rng.choice(["2", "4", "1.5"])
    mem = rng.choice(["2g", "4g", "512m"])
    name = rng.choice(["worker", "cache", "svc"])
    items = [
        ("docker run --memory", f"Start a detached worker: `ubuntu:24.04` running `sleep infinity`, named "
                                f"`{name}`, limited to {cpus} CPUs and {mem} of memory, restarting unless stopped.",
         f"docker run -d --name {name} --cpus {cpus} --memory {mem} --restart unless-stopped ubuntu:24.04 sleep infinity",
         {"require": ["docker run --memory|docker run -m", "docker run --cpus", "docker run --restart"]}),
        ("docker run --restart", f"Start `ubuntu:24.04` running `sleep infinity` detached as `{name}`, set to always "
                                 f"restart, with hostname `{name}-host`.",
         f"docker run -d --name {name} --restart always --hostname {name}-host ubuntu:24.04 sleep infinity"),
    ]
    return catalog(rng, items, setup(docker("pull=ubuntu:24.04")), check=["containers"])


@template(M, "challenge")
def train_on_gpus(rng):
    a, b = rng.sample(range(4), 2)
    epochs = rng.randint(2, 5)
    s = setup(docker("pull=mlgym/trainer:1.0"), "#: gpus 4", "#: cwd ~/runs")
    return P(f"**Train in a container.** Run `mlgym/trainer:1.0` (its command is `python train.py --workers 4`) on "
             f"GPUs {a} and {b} only, with enough shared memory for DataLoader workers (8 GB), the folder "
             f"`./outputs` mounted at `/workspace/outputs`, and the command `python train.py --epochs {epochs} --out "
             f"outputs`; remove the container afterwards. The checkpoint must land in `./outputs`.", s,
             f'docker run --rm --gpus \'"device={a},{b}"\' --shm-size=8g -v "$(pwd)/outputs":/workspace/outputs '
             f"mlgym/trainer:1.0 python train.py --epochs {epochs} --out outputs",
             post="ls outputs; docker ps -aq | wc -l", require=["docker run --gpus", "docker run --shm-size|docker run --ipc"])


@template(M, "challenge")
def port_conflict(rng):
    port = rng.choice([8080, 8000, 9000])
    other = port + 1
    s = setup(docker("pull=nginx"), f"docker run -dq --name old -p {port}:80 nginx >/dev/null")
    return P(f"**Port taken.** Port {port} is already used by the container `old`. Start a second nginx named `new` "
             f"detached on host port {other} instead (leave `old` running), then fetch the page from `new`.", s,
             f"docker run -d --name new -p {other}:80 nginx\ncurl -s localhost:{other}",
             check=["containers"], post=f"curl -s -o /dev/null -w '%{{http_code}}\\n' localhost:{other}")
