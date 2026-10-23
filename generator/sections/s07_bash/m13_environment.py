"""7.13 Environment, PATH & config files."""

from ._common import DATASETS, MODELS, NAMES, WORDS, P, catalog, challenge, files, q, template

M = "7.13"


# ================================================================= lite

@template(M, "lite")
def inspect_env(rng):
    var = rng.choice(["HOME", "USER", "SHELL", "LANG", "TERM"])
    hf = rng.choice(["/data/hf_cache", "/scratch/hf", "/mnt/models/hf"])
    gpus = rng.choice(["0", "0,1", "1,2", "2,3"])
    zone = rng.choice(["UTC", "Asia/Tokyo", "America/New_York", "Europe/Berlin"])
    setup = f"#: date 2026-{rng.randint(1, 12):02d}-{rng.randint(10, 28)} {rng.randint(10, 23)}:{rng.randint(10, 59)}:{rng.randint(10, 59)}"
    items = [
        ("printenv", "Print the value of `PATH` with `printenv`.", "printenv PATH"),
        ("tr", "Print the directories of `$PATH` one per line.", "echo \"$PATH\" | tr ':' '\\n'", {"require": ["tr", "$name"]}),
        ("env", f"Print the line for `{var}` from the full environment listing (`env`, filtered with grep `^{var}=`).",
         f"env | grep '^{var}='", {"require": ["env", "grep"]}),
        ("export", f"Export `HF_HOME={hf}` and show that a child process sees it (`bash -c 'echo $HF_HOME'`).",
         f"export HF_HOME={hf}\nbash -c 'echo $HF_HOME'"),
        ("export", f"Make only GPUs `{gpus}` visible to programs you start: export `CUDA_VISIBLE_DEVICES`, then print "
                   f"it with `printenv`.", f"export CUDA_VISIBLE_DEVICES={gpus}\nprintenv CUDA_VISIBLE_DEVICES",
         {"require": ["export", "printenv"]}),
        ("$name", f"Set `HF_HOME={hf}` for ONE command only: run `printenv HF_HOME` with it, then print `[$HF_HOME]` "
                  f"(still empty).", f"HF_HOME={hf} printenv HF_HOME\necho \"[$HF_HOME]\"", {"require": ["printenv"]}),
        ("date", "Print the current date and time as `YYYY-MM-DD_HH-MM-SS`.", "date +%Y-%m-%d_%H-%M-%S"),
        ("date", f"Print the current time in `{zone}` (set `TZ` for just this `date`).", f"TZ={zone} date"),
    ]
    return catalog(rng, items, setup)


@template(M, "lite")
def aliases(rng):
    name = rng.choice(["ll", "lh", "l2", "lsa"])
    other = rng.choice(["gs", "gst", "gss"])
    items = [
        ("alias", f"Create an alias `{name}` for `ls -lah`, then show its definition with `alias {name}`.",
         f"alias {name}='ls -lah'\nalias {name}"),
        ("alias", f"Create the alias `{other}` for `git status`, then ask bash what `{other}` is with `type`.",
         f"alias {other}='git status'\ntype {other}", {"require": ["alias", "type"]}),
        ("unalias", "The Ubuntu `.bashrc` defines `la`. Remove it with `unalias`, then ask `type la` (it fails now).",
         "unalias la\ntype la", {"answer": ["<stdout>", "<stderr>"]}),
        ("alias", "List every alias currently defined.", "alias"),
        ("alias", "Create `gpu` as an alias for `watch -n 1 nvidia-smi`, then print its definition.",
         "alias gpu='watch -n 1 nvidia-smi'\nalias gpu"),
    ]
    return catalog(rng, items)


# ================================================================= easy

@template(M, "easy")
def extend_path(rng):
    tool = rng.choice(["hello", "mytool", "greet", "train-helper"])
    d = rng.choice(["$HOME/bin", "$HOME/.local/bin", "$HOME/tools"])
    msg = rng.choice(WORDS)
    real = d.replace("$HOME", "/home/ada")
    setup = files({f"{real}/{tool}": f"#!/bin/bash\necho \"{msg} from {tool}\"\n"}) + f"\nchmod +x {real}/{tool}"
    kind = rng.choice(["now", "permanent", "which"])
    if kind == "now":
        return P(f"`{tool}` lives in `{d}`, which isn't on your PATH. Prepend that directory to `PATH` (export it) and "
                 f"run `{tool}`.", setup + f'\nexport PATH=$(echo "$PATH" | sed "s|{real}:||g")',
                 f'export PATH="{d}:$PATH"\n{tool}', require=["export"])
    if kind == "which":
        return P(f"Prepend `{d}` to your PATH, then print where bash finds `{tool}` (`which`).",
                 setup + f'\nexport PATH=$(echo "$PATH" | sed "s|{real}:||g")', f'export PATH="{d}:$PATH"\nwhich {tool}',
                 require=["export", "which|command -v|type"])
    return P(f"Make the change permanent: append `export PATH=\"{d}:$PATH\"` (literally, with the `$`) to "
             f"`~/.bashrc`, then reload it with `source` and run `{tool}`.", setup,
             f"echo 'export PATH=\"{d}:$PATH\"' >> ~/.bashrc\nsource ~/.bashrc\n{tool}",
             post="tail -1 ~/.bashrc", require=["source|."])


@template(M, "easy")
def dotenv(rng):
    pairs = {"WANDB_PROJECT": rng.choice(WORDS), "DATA_DIR": f"/data/{rng.choice(DATASETS)}",
             "BATCH_SIZE": str(rng.choice([16, 32, 64])), "MODEL": rng.choice(MODELS)}
    keys = rng.sample(list(pairs), 3)
    env = "".join(f"{k}={pairs[k]}\n" for k in keys)
    setup = files({".env": env})
    k = keys[0]
    kind = rng.choice(["export-all", "child"])
    if kind == "export-all":
        return P("Load every `KEY=value` line of `.env` into your shell AND export them (`set -a; source .env; set +a`), "
                 f"then print `{k}` from a child process (`bash -c 'echo ${k}'`).", setup,
                 f"set -a\nsource .env\nset +a\nbash -c 'echo ${k}'", require=["set -a", "set +a", "source|."])
    return P(f"Show why `set -a` matters: `source .env` WITHOUT it and print `[${k}]` from a child (`bash -c`), then "
             f"with `set -a ... set +a` and print it from a child again.", setup,
             f"source .env\nbash -c 'echo \"[${k}]\"'\nset -a\nsource .env\nset +a\nbash -c 'echo \"[${k}]\"'",
             require=["set -a"])


# ================================================================= medium

@template(M, "medium")
def cron_jobs(rng):
    h, m = rng.randint(0, 6), rng.choice([0, 15, 30, 45])
    script = rng.choice(["backup.sh", "sync.sh", "cleanup.sh", "report.sh"])
    log = f"/tmp/{script[:-3]}.log"
    setup = files({f"/home/ada/{script}": "#!/bin/bash\ndate\n"}) + f"\nchmod +x ~/{script}"
    kind = rng.choice(["install", "add", "remove"])
    line = f"{m} {h} * * * /home/ada/{script} >> {log} 2>&1"
    if kind == "install":
        return P(f"Schedule `~/{script}` to run every day at {h:02d}:{m:02d}, appending its output and errors to "
                 f"`{log}`. Install the crontab from standard input (`crontab -`), then list it.", setup,
                 f"echo '{line}' | crontab -\ncrontab -l", require=["crontab"])
    other = f"*/10 * * * * /home/ada/{rng.choice(['ping.sh', 'poll.sh'])}"
    if kind == "add":
        return P(f"Your crontab already has an entry. ADD a job running `~/{script}` every Monday at {h:02d}:{m:02d} "
                 f"(weekday 1), keeping the existing line, then list the crontab.", setup + f"\necho '{other}' | crontab -",
                 f"(crontab -l; echo '{m} {h} * * 1 /home/ada/{script}') | crontab -\ncrontab -l", require=["crontab"])
    return P(f"Remove every crontab line that mentions `{script}`, keeping the rest, then list the crontab.",
             setup + f"\nprintf '%s\\n' '{line}' '{other}' | crontab -", f"crontab -l | grep -v {script} | crontab -\ncrontab -l",
             require=["crontab"])


@template(M, "medium")
def per_command_env(rng):
    gpus = rng.choice(["1", "0,1", "2"])
    setup = "#: gpus 4\n#: venv .venv torch\n#: activate .venv"
    kind = rng.choice(["torch", "python-env", "two-runs"])
    if kind == "torch":
        return P(f"Run `python -c \"import torch; print(torch.cuda.device_count())\"` twice: once normally, once with "
                 f"`CUDA_VISIBLE_DEVICES={gpus}` set just for that command.", setup,
                 f'python -c "import torch; print(torch.cuda.device_count())"\n'
                 f'CUDA_VISIBLE_DEVICES={gpus} python -c "import torch; print(torch.cuda.device_count())"',
                 require=["python"])
    if kind == "python-env":
        hf = rng.choice(["/data/hf", "/scratch/cache"])
        return P(f"Start Python with `HF_HOME={hf}` set only for it and print `os.environ['HF_HOME']` from inside Python.",
                 setup, f'HF_HOME={hf} python -c "import os; print(os.environ[\'HF_HOME\'])"', require=["python"])
    return P("Hide all GPUs from one command: run `python -c \"import torch; print(torch.cuda.is_available())\"` with "
             "`CUDA_VISIBLE_DEVICES` set to an empty string just for that command.", setup,
             'CUDA_VISIBLE_DEVICES= python -c "import torch; print(torch.cuda.is_available())"', require=["python"])


# ================================================================= hard

@template(M, "hard")
def dedupe_path(rng):
    dirs = ["/usr/local/bin", "/usr/bin", "/bin", "/home/ada/.local/bin", "/home/ada/bin", "/opt/conda/bin", "/snap/bin"]
    path = [rng.choice(dirs) for _ in range(rng.randint(6, 10))]
    setup = f"export PATH={':'.join(path)}"
    return P("Your `PATH` has duplicates. Print a cleaned-up `PATH` (colon-separated) keeping only the FIRST "
             "occurrence of each directory, in order.", setup,
             "echo \"$PATH\" | tr ':' '\\n' | awk '!seen[$0]++' | paste -sd:", require=["$name"])


@template(M, "hard")
def idempotent_rc(rng):
    alias = rng.choice(["ll='ls -lah'", "gs='git status'", "gpu='watch -n 1 nvidia-smi'"])
    exp = rng.choice(['export HF_HOME=/data/hf', 'export WANDB_MODE=offline', 'export PIP_REQUIRE_VIRTUALENV=true'])
    line = f"alias {alias}" if rng.random() < 0.5 else exp
    return P(f"Write a function `addrc` that appends a line to `~/.bashrc` only if that exact line isn't there yet "
             f"(`grep -qxF`). Call `addrc \"{line}\"` twice, then print how many times the line occurs in `~/.bashrc` "
             f"(`grep -cxF`).", "", f'addrc() {{ grep -qxF "$1" ~/.bashrc || echo "$1" >> ~/.bashrc; }}\n'
             f'addrc "{line}"\naddrc "{line}"\ngrep -cxF "{line}" ~/.bashrc', require=["function", "grep"])


# ================================================================= challenge

def _path_world(rng, n):
    dirs = [f"/home/ada/{rng.choice(['bin', 'tools', 'opt', 'apps', 'sdk'])}{i}" for i in range(n)]
    spec = {}
    for d in dirs:
        if rng.random() < 0.7:
            for j in range(rng.randint(0, 4)):
                spec[f"{d}/tool{j}"] = "#!/bin/sh\n"
    made = sorted({p.rsplit("/", 1)[0] for p in spec})
    lines = [files(spec)] if spec else []
    exe = [p for p in spec if rng.random() < 0.7]
    if exe:
        lines.append("chmod +x " + " ".join(exe))
    order = dirs + [rng.choice(dirs) for _ in range(2)]
    rng.shuffle(order)
    lines.append(f"export PATH={':'.join(order)}:/usr/bin")
    _ = made
    return "\n".join(lines)


@template(M, "challenge")
def path_audit(rng):
    setup = _path_world(rng, 4)
    hidden = [_path_world(rng, rng.randint(8, 12)) for _ in range(2)]
    sol = ('declare -A seen\nIFS=: read -ra dirs <<< "$PATH"\nfor d in "${dirs[@]}"; do\n'
           '  if [[ -n ${seen[$d]} ]]; then echo "$d duplicate"; continue; fi\n  seen[$d]=1\n'
           '  if [[ ! -d $d ]]; then echo "$d missing"; continue; fi\n  n=0\n'
           '  for f in "$d"/*; do [[ -f $f && -x $f ]] && ((n++)); done\n  echo "$d $n executables"\ndone')
    return challenge("PATH audit", "Walk through `$PATH` in order. For each directory print `DIR missing` if it "
                     "doesn't exist, `DIR duplicate` if it appeared earlier in PATH, otherwise `DIR N executables` "
                     "(regular files with execute permission directly inside it).", setup, sol, hidden,
                     "up to 15 PATH entries")


def _dotenv(rng, n):
    keys = [f"{w.upper()}_{i}" for i, w in enumerate(rng.sample(WORDS, n))]
    lines = []
    for i, k in enumerate(keys):
        if i and rng.random() < 0.5:
            ref = rng.choice(keys[:i])
            lines.append(f"{k}=${{{ref}}}/{rng.choice(WORDS)}")
        else:
            lines.append(f"{k}={rng.choice(['/data', '/scratch', 'demo', str(rng.randint(1, 512))])}")
        if rng.random() < 0.2:
            lines.append(f"# note about {k}")
    return files({".env": "\n".join(lines) + "\n"})


@template(M, "challenge")
def resolve_dotenv(rng):
    setup = _dotenv(rng, 4)
    hidden = [_dotenv(rng, rng.randint(10, 20)) for _ in range(2)]
    sol = ("set -a\nsource .env\nset +a\ngrep -v '^#' .env | cut -d= -f1 | sort | while read -r k; do\n"
           '  echo "$k=$(printenv "$k")"\ndone')
    return challenge("Resolve a .env file", "`.env` has `KEY=value` lines (and `#` comments); values may refer to "
                     "earlier keys like `${DATA_0}/cache`. Load it so every key is EXPORTED, then print `KEY=VALUE` for "
                     "each key in the file, with references resolved, sorted by key, reading the values back from the "
                     "environment (`printenv`).", setup, sol, hidden, "up to 20 keys", require=["set -a"])
