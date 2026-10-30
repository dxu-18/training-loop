"""7.16 ML workflow one-liners."""

from ._common import DATASETS, MODELS, NAMES, WORDS, P, catalog, challenge, files, q, template, train_log

M = "7.16"

TRAIN_PY = ("import os, sys, time\n"
            "args = dict(zip(sys.argv[1::2], sys.argv[2::2]))\n"
            "lr = float(args.get('--lr', 0.001)); seed = int(args.get('--seed', 0))\n"
            "gpu = os.environ.get('CUDA_VISIBLE_DEVICES', 'all')\n"
            "for e in range(3):\n"
            "    time.sleep(1)\n"
            "    acc = round(0.5 + (e + 1) * 0.1 - abs(lr - 0.0003) * 50 + seed * 0.003, 4)\n"
            "    print(f'epoch={e} loss={1 / (e + 2) + lr:.4f} val_acc={acc}', flush=True)\n"
            "print(f'done lr={lr} seed={seed} gpu={gpu}', flush=True)\n")


def _gpu_setup(rng, n=None, procs=True):
    n = n or rng.choice([2, 4, 8])
    model = rng.choice(["a100", "h100", "4090"])
    lines = [f"#: gpus {n} {model}"]
    used = []
    if procs:
        for i in range(rng.randint(1, 3)):
            g = rng.randrange(n)
            mib = rng.choice([4096, 10240, 20480, 30720])
            used.append(g)
            lines.append(f"#: proc cpu={rng.choice([99.0, 98.4, 87.5])} gpu={g}:{mib} -- python3 train.py --run r{i}")
    return "\n".join(lines), n, used


# ================================================================= lite

@template(M, "lite")
def gpus(rng):
    setup, n, used = _gpu_setup(rng)
    g = rng.randrange(n)
    k = rng.choice([2, 4]) if n >= 4 else 2
    script = files({"train.py": "import torch\nprint('visible GPUs:', torch.cuda.device_count())\n"
                                "x = torch.zeros(1).cuda()\nprint('ok')\n",
                    "ddp.py": "import os\nprint(f\"rank {os.environ['RANK']} of {os.environ['WORLD_SIZE']}\")\n"})
    env = "#: venv .venv torch\n#: activate .venv\n"
    items = [
        ("nvidia-smi", "Show the GPU status table.", "nvidia-smi"),
        ("nvidia-smi --query-gpu", "Print one CSV line per GPU with its index, name, used memory, total memory and "
                                   "utilization (`--query-gpu=index,name,memory.used,memory.total,utilization.gpu "
                                   "--format=csv`).",
         "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv",
         {"require": ["nvidia-smi --query-gpu", "nvidia-smi --format"]}),
        ("nvidia-smi --query-compute-apps", "Which processes hold GPU memory? Print their PID and used memory as CSV.",
         "nvidia-smi --query-compute-apps=pid,used_memory --format=csv"),
        ("nvidia-smi -l", "Refresh the GPU table every second with `nvidia-smi -l 1` (it runs until you press Ctrl-C).",
         "nvidia-smi -l 1", {"setup": "#: ctrlc 3"}),
        ("nvtop", "Watch the GPUs with `nvtop` (it runs until you press Ctrl-C).", "nvtop", {"setup": "#: ctrlc 2"}),
        ("python", f"Run `python train.py` pinned to GPU {g} only (`CUDA_VISIBLE_DEVICES` for that one command).",
         f"CUDA_VISIBLE_DEVICES={g} python train.py", {"setup": env + script, "require": ["python"]}),
        ("torchrun", f"Launch `ddp.py` on {k} GPUs of this machine with `torchrun --nproc_per_node={k}`.",
         f"torchrun --nproc_per_node={k} ddp.py", {"setup": env + script, "answer": ["<stdout>"]}),
    ]
    return catalog(rng, items, setup)


@template(M, "lite")
def log_oneliners(rng):
    logs = {f"logs/lr{lr}_s{s}.log": train_log(rng, rng.randint(3, 6), noise=False)
            for lr in rng.sample(["1e-3", "3e-4", "1e-4"], 2) for s in range(2)}
    main_log = train_log(rng, rng.randint(4, 8))
    ckpts = {f"checkpoints/step_{i:05d}.pt": "w\n" for i in range(1000, 1000 * rng.randint(5, 8), 1000)}
    touch = "\n".join(f"touch -d '2026-10-{10 + j:02d} 12:00' {p}" for j, p in enumerate(sorted(ckpts)))
    caches = {f"/home/ada/.cache/huggingface/hub/m{i}.bin": "x" * rng.randint(1, 9) * 4000 + "\n" for i in range(2)}
    caches |= {"/home/ada/.cache/torch/hub/w.pt": "x" * 9000 + "\n", "/home/ada/.cache/pip/http/a": "x\n"}
    setup = "#: cwd ~/exp\n" + files({**logs, "train.log": main_log, **ckpts, **caches}) + "\n" + touch
    items = [
        ("grep -o", "Extract every loss value from `train.log` into `losses.txt` (one number per line: "
                    "`grep -o \"loss=[0-9.]*\"` then `cut`).", 'grep -o "loss=[0-9.]*" train.log | cut -d= -f2 > losses.txt',
         {"post": "cat losses.txt", "require": ["grep -o", "cut"]}),
        ("grep -h", "Print the single best `val_acc` line across all the logs in `logs/` (no file names: `grep -h`; "
                    "sort numerically on the value after `=`).", 'grep -h "val_acc" logs/*.log | sort -t= -k3 -n | tail -1',
         {"require": ["grep -h", "sort"]}),
        ("du -sh", "How big are the Hugging Face, torch and pip caches in `~/.cache`? (one `du -sh`)",
         "du -sh ~/.cache/huggingface ~/.cache/torch ~/.cache/pip"),
        ("xargs", "Keep only the 3 newest checkpoints in `checkpoints/`: delete the rest with `ls -t | tail -n +4 | "
                  "xargs rm -f`.", "ls -t checkpoints/*.pt | tail -n +4 | xargs rm -f", {"post": "ls checkpoints",
                                                                                         "require": ["ls -t", "tail", "xargs"]}),
        ("tail -f", "Follow all the logs and show only lines containing `epoch`, as they arrive "
                    "(`tail -f ... | grep --line-buffered`). It runs until you press Ctrl-C.",
         'tail -f logs/*.log | grep --line-buffered "epoch"', {"setup": "#: ctrlc 2",
                                                              "require": ["tail -f", "grep --line-buffered"]}),
    ]
    return catalog(rng, items, setup)


# ================================================================= easy

@template(M, "easy")
def launch(rng):
    cfg = f"configs/{rng.choice(MODELS)}.yaml"
    setup = (f"#: date 2026-{rng.randint(1, 12):02d}-{rng.randint(10, 28)} {rng.randint(10, 23)}:{rng.randint(10, 59)}\n"
             + files({"train.py": TRAIN_PY, cfg: "lr: 0.001\n"}) + "\nmkdir -p logs")
    kind = rng.choice(["nohup-dated", "server", "tensorboard"])
    if kind == "nohup-dated":
        return P(f"Start `python3 train.py --config {cfg}` with `nohup` in the background, logging stdout and stderr to "
                 f"`logs/run_DATE_TIME.log` where the name comes from `date +%F_%H%M`.", setup,
                 f'nohup python3 train.py --config {cfg} > "logs/run_$(date +%F_%H%M).log" 2>&1 &',
                 post="sleep 10; ls logs; cat logs/*.log", answer=["<post>"], require=["nohup", "$(", "date"])
    if kind == "server":
        port = rng.choice([8000, 8080, 9000])
        return P(f"Serve the current folder over HTTP on port {port} (`python3 -m http.server`) in the background, then "
                 f"fetch `http://localhost:{port}/{cfg}` with `curl -s`.", setup,
                 f"python3 -m http.server {port} &\nsleep 1\ncurl -s http://localhost:{port}/{cfg}",
                 require=["python3 -m http.server|python -m http.server", "curl"])
    return P("Start TensorBoard on the `runs` directory, port 6006 (it runs until you press Ctrl-C).",
             "#: venv .venv tensorboard\n#: activate .venv\nmkdir -p runs", "tensorboard --logdir runs --port 6006",
             require=["tensorboard"], answer=["<stdout>"])


@template(M, "easy")
def sweep(rng):
    lrs = rng.sample(["1e-3", "3e-4", "1e-4", "3e-3"], 2)
    seeds = rng.sample(range(3), 2)
    setup = files({"train.py": TRAIN_PY}) + "\nmkdir -p logs"
    return P(f"Run a sweep, one run at a time: for every learning rate in `{' '.join(lrs)}` and every seed in "
             f"`{' '.join(map(str, seeds))}`, run `python3 train.py --lr LR --seed SEED` with all output in "
             f"`logs/lrLR_sSEED.log`. Then print the last line of every log (`tail -n 1 logs/*.log`).", setup,
             f'for lr in {" ".join(lrs)}; do for seed in {" ".join(map(str, seeds))}; do\n'
             f'  python3 train.py --lr "$lr" --seed "$seed" > "logs/lr${{lr}}_s${{seed}}.log" 2>&1\ndone; done\n'
             "tail -n 1 logs/*.log", require=["for"])


# ================================================================= medium

@template(M, "medium")
def parallel_gpus(rng):
    n = rng.choice([2, 3, 4])
    lr = rng.choice(["1e-3", "3e-4", "1e-4", "3e-3"])
    d = rng.choice(["logs", "runs", "out"])
    setup = f"#: gpus {n}\n" + files({"train.py": TRAIN_PY}) + f"\nmkdir -p {d}"
    return P(f"Run one training per GPU in parallel: for k in 0..{n - 1}, run `python3 train.py --lr {lr} --seed k` "
             f"with `CUDA_VISIBLE_DEVICES=k` and all output in `{d}/seedk.log`, using `seq` and "
             f"`xargs -P {n} -I{{}} sh -c`. Then print the last line of each log.", setup,
             f"seq 0 {n - 1} | xargs -P {n} -I{{}} sh -c 'CUDA_VISIBLE_DEVICES={{}} python3 train.py --lr {lr} --seed {{}} > {d}/seed{{}}.log 2>&1'\n"
             f"tail -qn 1 {d}/*.log", require=["xargs -P", "xargs -I{}", "seq"])


@template(M, "medium")
def pick_gpu(rng):
    setup, n, used = _gpu_setup(rng, rng.choice([4, 8]))
    kind = rng.choice(["freest", "idle-list", "count-busy"])
    if kind == "freest":
        return P("Print the index of the GPU with the least memory in use (query `index,memory.used` as CSV without "
                 "header and units, sort numerically on the memory).", setup,
                 "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1",
                 require=["nvidia-smi --query-gpu"])
    if kind == "idle-list":
        return P("Print the indices of all GPUs at 0% utilization, comma-separated (ready for `CUDA_VISIBLE_DEVICES`).",
                 setup, "nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits | "
                        "awk -F', ' '$2 == 0 {print $1}' | paste -sd,", require=["nvidia-smi --query-gpu"])
    return P("How many GPUs are busy (have at least one compute process)? Print just the number.", setup,
             "nvidia-smi --query-compute-apps=gpu_bus_id --format=csv,noheader | sort -u | wc -l",
             require=["nvidia-smi --query-compute-apps"])


# ================================================================= hard

@template(M, "hard")
def free_a_gpu(rng):
    setup, n, used = _gpu_setup(rng, 4)
    g = used[0]
    return P(f"Kill every process using GPU {g} (`nvidia-smi --query-compute-apps=pid -i {g}`, no header), then show the "
             f"compute apps that are left.", setup,
             f"nvidia-smi --query-compute-apps=pid --format=csv,noheader -i {g} | xargs -r kill\nsleep 1\n"
             "nvidia-smi --query-compute-apps=pid,used_memory --format=csv", require=["nvidia-smi", "kill"])


@template(M, "hard")
def plot_losses(rng):
    log = train_log(rng, rng.randint(3, 6), noise=False)
    out = rng.choice(["loss.png", "curve.png", "losses.png"])
    setup = "#: venv .venv matplotlib\n#: activate .venv\n" + files({"train.log": log})
    return P(f"Extract the loss values from `train.log` into `losses.txt` (one per line), then plot them with a 3-line "
             f"Python script (read the numbers, `plt.plot`, `plt.savefig(\"{out}\")`) run from a here-document, and "
             f"print what `file {out}` says.", setup,
             f"grep -o 'loss=[0-9.]*' train.log | cut -d= -f2 > losses.txt\npython <<'EOF'\nimport matplotlib.pyplot as plt\n"
             f"plt.plot([float(x) for x in open('losses.txt')])\nplt.savefig('{out}')\nEOF\nfile {out}",
             post="wc -l < losses.txt", require=["grep", "python"])


# ================================================================= challenge

def _sweep_logs(rng, lrs, seeds):
    spec = {}
    for lr in lrs:
        for s in seeds:
            spec[f"logs/lr{lr}_s{s}.log"] = train_log(rng, rng.randint(3, 6), noise=rng.random() < 0.5)
    return files(spec)


@template(M, "challenge")
def sweep_table(rng):
    def mk(rng, nl, ns):
        return _sweep_logs(rng, rng.sample(["1e-2", "3e-3", "1e-3", "3e-4", "1e-4", "3e-5"], nl), list(range(ns)))
    setup = mk(rng, 2, 2)
    hidden = [mk(rng, rng.randint(4, 6), rng.randint(3, 5)) for _ in range(2)]
    sol = ("for f in logs/*.log; do\n  lr=${f#logs/lr}; lr=${lr%%_s*}\n"
           "  best=$(grep -oP 'val_acc=\\K[0-9.]+' \"$f\" | sort -n | tail -1)\n  echo \"$lr $best\"\ndone | "
           "awk '{s[$1] += $2; n[$1]++} END {for (k in s) printf \"%s %.4f %d\\n\", k, s[k] / n[k], n[k]}' | sort -k2,2nr -k1,1")
    return challenge("Sweep summary", "`logs/lrLR_sSEED.log` are the logs of a learning-rate × seed sweep. For each run "
                     "take its BEST `val_acc`; then for each learning rate print `LR MEAN RUNS`: the mean of those best "
                     "values over its seeds (4 decimals) and how many runs it had, best mean first (ties: by LR text).",
                     setup, sol, hidden, "up to 6 learning rates × 5 seeds")


def _ckpts(rng, n_runs):
    spec, touches = {}, []
    for r in range(n_runs):
        run = f"runs/{rng.choice(MODELS)}-{r}"
        steps = sorted(rng.sample(range(500, 20000, 500), rng.randint(2, 7)))
        for s in steps:
            val = f"{rng.uniform(0.5, 0.95):.4f}"
            p = f"{run}/step{s:05d}_val{val}.pt"
            spec[p] = "w\n"
            touches.append(f"touch -d @{1790000000 + s * 60} {p}")
    return files(spec) + "\n" + "\n".join(touches)


@template(M, "challenge")
def retention(rng):
    k = rng.choice([2, 3])
    setup = _ckpts(rng, 2)
    hidden = [_ckpts(rng, rng.randint(4, 7)) for _ in range(2)]
    sol = ('for d in runs/*/; do\n'
           '  best=$(ls "$d" | sed -n \'s/^\\(.*_val\\([0-9.]*\\)\\.pt\\)$/\\2 \\1/p\' | sort -k1,1nr | head -1 | cut -d" " -f2)\n'
           f'  ls -t "$d" | tail -n +{k + 1} | while read -r f; do\n    [[ $f == "$best" ]] || rm -f "$d$f"\n  done\ndone\n'
           'find runs -name "*.pt" | sort')
    return challenge("Checkpoint retention", f"Each run directory in `runs/` has checkpoints `stepNNNNN_valX.pt` (X is the "
                     f"validation score). In every run keep the {k} most recently modified checkpoints AND the one with the "
                     f"highest X; delete the rest. Then print all remaining checkpoint paths, sorted.", setup, sol, hidden,
                     "up to 7 runs × 7 checkpoints")
