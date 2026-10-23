"""7.14 Remote work: ssh, scp, rsync, curl & tmux."""

import json

from ._common import DATASETS, MODELS, NAMES, WORDS, P, catalog, challenge, files, q, template

M = "7.14"
HOSTS = ("gpu-box", "trainer", "a100-node", "lab-server", "cluster01")


def _host(rng):
    return rng.choice(HOSTS)


# ================================================================= lite

@template(M, "lite")
def ssh_basics(rng):
    h = _host(rng)
    port = rng.choice([2222, 2200, 8022])
    email = f"{rng.choice(NAMES)}@example.com"
    alias = rng.choice(["gpu", "box", "lab", "train"])
    ip = f"10.0.{rng.randint(0, 9)}.{rng.randint(2, 250)}"
    items = [
        ("ssh", f"Run `hostname` on `{h}` over ssh (you can log in with your key).", f"ssh {h} hostname",
         {"setup": f"#: host {h} key"}),
        ("ssh", f"Log in to `{h}` as `ada`, run `nvidia-smi -L` there, then log out with `exit`.",
         f"ssh ada@{h}\nnvidia-smi -L\nexit", {"setup": f"#: host {h} key gpus=2"}),
        ("ssh -p", f"`{h}` runs sshd on port {port}. Run `whoami` there as `ada`.", f"ssh -p {port} ada@{h} whoami",
         {"setup": f"#: host {h} key port={port}"}),
        ("ssh -i", f"Your key for `{h}` is `~/.ssh/work_key` (not the default name). Run `hostname` there with it.",
         f"ssh -i ~/.ssh/work_key {h} hostname",
         {"setup": f"#: host {h} key\nmv ~/.ssh/id_ed25519 ~/.ssh/work_key && mv ~/.ssh/id_ed25519.pub ~/.ssh/work_key.pub"}),
        ("ssh-keygen", f"Generate an ed25519 key pair with the comment `{email}`, saved as `~/.ssh/id_ed25519` with an "
                       f"empty passphrase (`-N \"\"`), then print the public key.",
         f'ssh-keygen -t ed25519 -C "{email}" -f ~/.ssh/id_ed25519 -N ""\ncat ~/.ssh/id_ed25519.pub',
         {"require": ["ssh-keygen -t", "ssh-keygen -C"], "answer": ["<post>"], "post": "ls ~/.ssh; cut -d' ' -f1,3 ~/.ssh/id_ed25519.pub"}),
        ("ssh-copy-id", f"Install your public key on `{h}` with `ssh-copy-id` (its password is `hunter2`: type it "
                        f"when asked), then run `hostname` there without a password.",
         f"ssh-copy-id {h}\nhunter2\nssh {h} hostname",
         {"setup": f"#: host {h} password=hunter2\nssh-keygen -q -t ed25519 -f ~/.ssh/id_ed25519 -N ''"}),
        ("ssh-agent", "Start an ssh agent in this shell (`eval \"$(ssh-agent -s)\"`), add your key with `ssh-add`, and "
                      "list the agent's keys (`ssh-add -l`).", 'eval "$(ssh-agent -s)"\nssh-add ~/.ssh/id_ed25519\nssh-add -l',
         {"setup": "ssh-keygen -q -t ed25519 -f ~/.ssh/id_ed25519 -N '' -C ada@laptop", "require": ["ssh-agent", "ssh-add", "eval"],
          "answer": ["<post>"], "post": "ssh-add -l | cut -d' ' -f3-"}),
        ("ssh", f"Add a host alias `{alias}` to `~/.ssh/config` for `HostName {ip}`, `User ada` and "
                f"`IdentityFile ~/.ssh/id_ed25519`, then run `hostname` on it with just `ssh {alias}`.",
         f"cat >> ~/.ssh/config <<'EOF'\nHost {alias}\n    HostName {ip}\n    User ada\n    IdentityFile ~/.ssh/id_ed25519\nEOF\n"
         f"ssh {alias} hostname", {"setup": f"#: host {h} key ip={ip}", "require": ["ssh", "<<"]}),
    ]
    return catalog(rng, items)


@template(M, "lite")
def copying(rng):
    h = _host(rng)
    model = f"{rng.choice(MODELS)}.pt"
    exp = f"exp{rng.randint(1, 9)}"
    spec = {model: "weights\n", "data/train.csv": "a,b\n", "data/val.csv": "c,d\n", ".venv/bin/python": "x\n",
            "ckpt/step100.pt": "w\n", "src/train.py": "print(1)\n", "README.md": "# hi\n"}
    setup = f"#: host {h} key\n#: cwd ~/project\n" + files(spec) + f"\nssh {h} 'mkdir -p models runs/{exp}/plots && echo 0.91 > runs/{exp}/acc.txt && " \
                                              f"echo png > runs/{exp}/plots/loss.png'"
    items = [
        ("scp", f"Copy `{model}` into `~/models/` on `{h}` (create the directory there first).",
         f"ssh {h} mkdir -p models\nscp {model} {h}:~/models/", {"post": f"ssh {h} ls models"}),
        ("scp -r", f"Copy the whole `~/runs/{exp}` directory from `{h}` into the current directory.",
         f"scp -r {h}:~/runs/{exp} ./", {"post": f"find {exp} | sort"}),
        ("rsync", f"Sync the CONTENTS of `data/` into `~/data/` on `{h}` with `rsync -avzP` (mind the trailing slash).",
         f"rsync -avzP data/ {h}:~/data/", {"post": f"ssh {h} ls data", "require": ["rsync -a", "rsync -z", "rsync -P"]}),
        ("rsync --exclude", f"Sync the current project to `~/project/` on `{h}`, leaving out `.venv` and every `*.pt` file.",
         f"rsync -avzP --exclude '.venv' --exclude '*.pt' ./ {h}:~/project/", {"post": f"ssh {h} 'cd project && find . | sort'"}),
        ("rsync -n", "Mirror `src/` into `backup/`, deleting files that no longer exist in src, but first do a DRY run "
                     "(`-n`) to see what would happen.", "rsync -avn --delete src/ backup/",
         {"setup": "mkdir -p backup && echo old > backup/old.py", "require": ["rsync -n", "rsync --delete"]}),
    ]
    return catalog(rng, items, setup)


# ================================================================= easy

@template(M, "easy")
def http(rng):
    ds = rng.choice(DATASETS)
    text = rng.choice(WORDS)
    label = rng.choice(["positive", "negative", "neutral"])
    setup = "\n".join([
        f"#: url https://example.com/file.zip zip-bytes-{rng.randint(100, 999)}",
        f"#: url https://example.com/{ds}.tar {ds}-archive",
        f"#: url http://localhost:8000/predict {{\"text\": \"{text}\", \"label\": \"{label}\", \"score\": 0.{rng.randint(50, 99)}}}",
    ])
    items = [
        ("curl -O", "Download `https://example.com/file.zip` keeping its remote file name (`-O`).",
         "curl -O https://example.com/file.zip", {"post": "ls; cat file.zip", "answer": ["<post>"]}),
        ("curl -L", "`https://example.com/download` redirects to the real file. Follow redirects and save it as "
                    "`data.zip` (`-L -o`).", "curl -L -o data.zip https://example.com/download",
         {"post": "ls -1", "answer": ["<post>"], "require": ["curl -L", "curl -o"]}),
        ("curl -s", "Fetch `https://api.github.com/repos/pytorch/pytorch` quietly and print just its "
                    "`stargazers_count` with jq.", "curl -s https://api.github.com/repos/pytorch/pytorch | jq .stargazers_count",
         {"require": ["curl -s", "jq"]}),
        ("curl -X", f"Test your model API: POST the JSON `{{\"text\": \"{text}\"}}` to `http://localhost:8000/predict` with a "
                    f"`Content-Type: application/json` header.",
         f'curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d \'{{"text": "{text}"}}\'',
         {"require": ["curl -X", "curl -H", "curl -d"]}),
        ("curl -I", "Print only the response headers of `https://example.com/` (`-I`).", "curl -I https://example.com/"),
        ("wget", f"Download `https://example.com/{ds}.tar` with wget, quietly (`-q`), then list the directory.",
         f"wget -q https://example.com/{ds}.tar\nls", {"require": ["wget -q"]}),
        ("wget -c", f"Resume an interrupted download of `https://example.com/{ds}.tar` (a partial `{ds}.tar` is already "
                    f"here) with `wget -c`, quietly, then print the file.", f"wget -c -q https://example.com/{ds}.tar\ncat {ds}.tar",
         {"setup": f"printf '{ds}-' > {ds}.tar", "require": ["wget -c"]}),
    ]
    return catalog(rng, items, setup)


@template(M, "easy")
def tunnels(rng):
    h = _host(rng)
    port, prog = rng.choice([(8888, "jupyter-lab --no-browser --port 8888"), (6006, "tensorboard --logdir runs --port 6006"),
                             (8000, "python3 -m http.server 8000")])
    local = rng.choice([port, port + 1, 9000 + port % 100])
    setup = f"#: host {h} key\n#: proc host={h} port={port} -- {prog}"
    kind = rng.choice(["forward", "background", "jump"])
    if kind == "forward":
        return P(f"`{prog.split()[0]}` runs on `{h}` port {port}. Forward local port {local} to it (`ssh -L`) as a "
                 f"background tunnel with no shell (`-N -f`), then fetch `http://localhost:{local}/` with `curl -s`.",
                 setup, f"ssh -N -f -L {local}:localhost:{port} {h}\ncurl -s http://localhost:{local}/",
                 require=["ssh -L", "ssh -N", "ssh -f"])
    if kind == "background":
        return P(f"Open a background tunnel (no shell) from local port {port} to port {port} on `{h}`, then show "
                 f"what's listening on local port {port} (`lsof -i`).", setup,
                 f"ssh -N -f -L {port}:localhost:{port} {h}\nlsof -i :{port}", require=["ssh -L", "lsof"])
    inner = rng.choice(["internal-box", "db-node", "gpu-inner"])
    return P(f"`{inner}` is only reachable through the jump host `bastion`. Run `hostname` on `{inner}` by hopping "
             f"through it (`ssh -J`).", f"#: host bastion key\n#: host {inner} key private via=bastion",
             f"ssh -J bastion {inner} hostname", require=["ssh -J"])


# ================================================================= medium

@template(M, "medium")
def tmux_sessions(rng):
    name = rng.choice(["train", "exp", "sweep", "notebook", "work"])
    other = rng.choice(["eval", "data", "monitor"])
    kind = rng.choice(["new-ls", "kill", "attach", "send"])
    if kind == "new-ls":
        return P(f"Create a detached tmux session named `{name}` and another named `{other}`, then list the sessions.",
                 "", f"tmux new -s {name} -d\ntmux new -s {other} -d\ntmux ls", require=["tmux new", "tmux ls"])
    if kind == "kill":
        return P(f"Sessions `{name}` and `{other}` exist. Kill `{name}`, then list what's left.",
                 f"tmux new -d -s {name}\ntmux new -d -s {other}", f"tmux kill-session -t {name}\ntmux ls",
                 require=["tmux kill-session"])
    if kind == "attach":
        return P(f"Attach to the existing session `{name}` (`tmux attach -t`), run `echo inside $TMUX_PANE` in it; then "
                 f"(back outside, after the input ends) list the sessions.", f"tmux new -d -s {name}",
                 f"tmux attach -t {name}\necho inside $TMUX_PANE", post="tmux ls", require=["tmux attach"])
    cmd = rng.choice(["python3 -c 'print(42)'", "echo training started", "date +%F"])
    return P(f"Without attaching, type `{cmd}` + Enter into session `{name}` (`tmux send-keys ... Enter`), then print "
             f"the pane's contents (`tmux capture-pane -p -t {name}`).", f"#: date 2026-10-20\ntmux new -d -s {name}",
             f"tmux send-keys -t {name} {q(cmd)} Enter\ntmux capture-pane -p -t {name}", require=["tmux"])


@template(M, "medium")
def deploy(rng):
    h = _host(rng)
    spec = {"src/train.py": "print('train')\n", "src/model.py": "x = 1\n", "configs/base.yaml": "lr: 0.1\n",
            ".venv/lib/x.py": "venv\n", f"checkpoints/{rng.choice(MODELS)}.pt": "w\n", "notes.md": "n\n",
            "wandb/run-1/log.txt": "w\n"}
    setup = f"#: host {h} key\n#: cwd ~/project\n" + files(spec)
    excl = rng.sample([".venv", "*.pt", "wandb", "notes.md"], 2)
    flags = " ".join(f"--exclude {q(e)}" for e in excl)
    kind = rng.choice(["sync", "dry"])
    if kind == "sync":
        return P(f"Sync this project to `~/project/` on `{h}`, excluding {' and '.join(f'`{e}`' for e in excl)}.", setup,
                 f"rsync -avzP {flags} ./ {h}:~/project/", post=f"ssh {h} 'cd project && find . | sort'",
                 answer=["<post>"], require=["rsync --exclude"])
    return P(f"Before syncing to `{h}:~/project/` (excluding {' and '.join(f'`{e}`' for e in excl)}), do a dry run "
             f"(`-n`) to see which files WOULD be sent.", setup, f"rsync -avzn {flags} ./ {h}:~/project/",
             require=["rsync -n", "rsync --exclude"])


# ================================================================= hard

@template(M, "hard")
def remote_training(rng):
    h = _host(rng)
    epochs = rng.randint(2, 4)
    body = (f"import time\nfor e in range({epochs}):\n    time.sleep(1)\n    print(f'epoch {{e}} loss={{1/(e+1):.3f}}', flush=True)\n"
            "print('done', flush=True)\n")
    setup = f"#: host {h} key\n" + files({"train.py": body})
    return P(f"Copy `train.py` to your home on `{h}`, start it there in the background with `nohup` (all output to "
             f"`train.log`, so it survives the ssh session ending), then wait 10 seconds and print the last 2 lines of "
             f"the remote log.", setup,
             f"scp train.py {h}:~/\nssh {h} 'nohup python3 train.py > train.log 2>&1 &'\nsleep 10\nssh {h} tail -2 train.log",
             require=["scp", "ssh", "nohup"])


@template(M, "hard")
def api_digest(rng):
    repos = rng.sample(["pytorch/pytorch", "huggingface/transformers", "openai/whisper", "karpathy/nanoGPT",
                        "numpy/numpy", "pandas-dev/pandas"], 3)
    data = [{"name": r.split("/")[1], "full_name": r, "stargazers_count": rng.randint(1000, 90000),
             "open_issues_count": rng.randint(10, 5000), "language": rng.choice(["Python", "C++", "Jupyter Notebook"])}
            for r in repos]
    org = rng.choice(["ml-lab", "acme-ai"])
    setup = f"#: url https://api.github.com/orgs/{org}/repos {json.dumps(data)}"
    kind = rng.choice(["table", "top", "sum"])
    if kind == "table":
        return P(f"`https://api.github.com/orgs/{org}/repos` returns a JSON array of repos. Print `NAME STARS` lines "
                 f"sorted by stars, most first (curl quietly + jq + sort).", setup,
                 f"curl -s https://api.github.com/orgs/{org}/repos | jq -r '.[] | \"\\(.name) \\(.stargazers_count)\"' | sort -k2,2nr",
                 require=["curl", "jq"])
    if kind == "top":
        return P(f"Print the full name of the repo with the most open issues from `https://api.github.com/orgs/{org}/repos`.",
                 setup, f"curl -s https://api.github.com/orgs/{org}/repos | jq -r 'max_by(.open_issues_count).full_name'",
                 require=["curl", "jq"])
    return P(f"Print the total number of stars across the repos at `https://api.github.com/orgs/{org}/repos`.", setup,
             f"curl -s https://api.github.com/orgs/{org}/repos | jq 'map(.stargazers_count) | add'", require=["curl", "jq"])


# ================================================================= challenge

def _mirror(rng, n):
    h = "gpu-box"
    local, remote = {}, {}
    for i in range(n):
        p = f"{rng.choice(['', 'src/', 'configs/', 'src/utils/'])}{rng.choice(WORDS)}_{i}.{rng.choice(['py', 'yaml', 'txt'])}"
        body = f"{rng.choice(WORDS)} {i}\n"
        r = rng.random()
        if r < 0.15:
            local[p] = body
        elif r < 0.3:
            remote[p] = body
        else:
            local[p] = body
            remote[p] = body if rng.random() < 0.6 else body + "changed\n"
    lines = [f"#: host {h} key", "#: cwd ~/project", files(local) if local else ":"]
    staged = {f"/tmp/rseed/{p}": b for p, b in remote.items()}
    lines.append(files(staged, mkdirs=["/tmp/rseed"]))
    lines.append(f"rsync -a /tmp/rseed/ {h}:~/project/ > /dev/null && rm -rf /tmp/rseed")
    return "\n".join(lines)


@template(M, "challenge")
def mirror_check(rng):
    setup = _mirror(rng, 6)
    hidden = [_mirror(rng, rng.randint(20, 30)) for _ in range(2)]
    sol = ("comm -3 <(find . -type f | sort) <(ssh gpu-box 'cd project && find . -type f | sort') | "
           "sed -e 's|^\\t\\./|only remote: |' -e 's|^\\./|only local: |' | sort\n"
           "join <(find . -type f -exec md5sum {} + | awk '{print $2, $1}' | sort) "
           "<(ssh gpu-box 'cd project && find . -type f -exec md5sum {} +' | awk '{print $2, $1}' | sort) | "
           "awk '$2 != $3 {sub(/^\\.\\//, \"\", $1); print \"differs: \" $1}'")
    return challenge("Is the remote copy in sync?", "Compare `~/project` here with `~/project` on `gpu-box` (you can "
                     "ssh there with your key). Print `only local: PATH` and `only remote: PATH` lines (together, "
                     "sorted), then `differs: PATH` for files on both sides with different contents (sorted by path). "
                     "Paths are relative to the project, without `./`.", setup, sol, hidden, "up to 30 files")


def _pages(rng, n_pages):
    lines, all_items = [], []
    for page in range(1, n_pages + 1):
        items = [{"run": f"{rng.choice(MODELS)}-{page}-{i}", "acc": round(rng.uniform(0.5, 0.99), 3)}
                 for i in range(rng.randint(2, 4))]
        all_items += items
        nxt = f"https://api.example.com/runs?page={page + 1}" if page < n_pages else None
        lines.append(f"#: url https://api.example.com/runs?page={page} {json.dumps({'items': items, 'next': nxt})}")
    return "\n".join(lines)


@template(M, "challenge")
def paginate(rng):
    setup = _pages(rng, 2)
    hidden = [_pages(rng, rng.randint(5, 9)) for _ in range(2)]
    sol = ('url="https://api.example.com/runs?page=1"\n: > all.jsonl\nwhile [[ $url != null ]]; do\n'
           '  page=$(curl -s "$url")\n  jq -c ".items[]" <<< "$page" >> all.jsonl\n  url=$(jq -r ".next" <<< "$page")\ndone\n'
           'jq -s "length" all.jsonl\njq -rs "max_by(.acc) | \\"\\(.run) \\(.acc)\\"" all.jsonl')
    return challenge("Follow the pages", "`https://api.example.com/runs?page=1` returns `{\"items\": [...], \"next\": URL}`; "
                     "keep fetching `next` until it is `null`. Then print the total number of runs and `RUN ACC` for the "
                     "run with the highest `acc` (the first one on ties).", setup, sol, hidden, "up to 9 pages",
                     require=["curl", "jq", "while"])
