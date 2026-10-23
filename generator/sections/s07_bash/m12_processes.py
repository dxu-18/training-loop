"""7.12 Processes & jobs."""

from ._common import DATASETS, MODELS, NAMES, WORDS, P, catalog, challenge, files, q, template

M = "7.12"


def _procs(rng, n=3, gpu=False):
    """`#: proc` lines for a few long-running ML jobs."""
    lines, cmds = [], []
    scripts = rng.sample(["train.py", "eval.py", "preprocess.py", "serve.py", "finetune.py", "sweep.py"], n)
    for i, sc in enumerate(scripts):
        args = f"--lr {rng.choice(['1e-3', '3e-4'])}" if sc in ("train.py", "finetune.py") else \
            f"--data {rng.choice(DATASETS)}"
        cmd = f"python3 {sc} {args}"
        opts = f"cpu={rng.choice([99.5, 97.1, 45.0, 12.3, 3.2])} mem={rng.choice([12.5, 4.1, 0.8])} ago={rng.randint(60, 9000)}"
        if gpu:
            opts += f" gpu={i % 2}:{rng.choice([20480, 40960, 8192])}"
        lines.append(f"#: proc {opts} -- {cmd}")
        cmds.append(cmd)
    return "\n".join(lines), scripts, cmds


# ================================================================= lite

@template(M, "lite")
def inspect_system(rng):
    setup, scripts, _ = _procs(rng)
    sc = scripts[0]
    items = [
        ("ps", "List all processes in BSD style and keep only the lines mentioning `python` (`ps aux | grep`).",
         "ps aux | grep python", {"require": ["ps", "grep"]}),
        ("pgrep -f", f"Print the PID and full command line of the process running `{sc}` (`pgrep -af`).",
         f"pgrep -af {sc}", {"require": ["pgrep -a", "pgrep -f"]}),
        ("top", "Show one batch-mode snapshot of `top` (`-b -n 1`), first 12 lines.", "top -b -n 1 | head -12",
         {"require": ["top"]}),
        ("uptime", "How long has the machine been up, and what's the load? (`uptime`)", "uptime"),
        ("htop", "Open `htop` to look at the processes (it runs until Ctrl-C; after 2 s here).", "htop",
         {"setup": "#: ctrlc 2"}),
        ("watch -n", "Re-run `nvidia-smi -L` every second with `watch` (it runs until Ctrl-C; after 3 s here).",
         "watch -n 1 nvidia-smi -L", {"setup": "#: ctrlc 3\n#: gpus 2"}),
        ("free -h", "Show memory usage in human-readable units.", "free -h"),
        ("nproc", "Print the number of CPU cores available.", "nproc"),
        ("lscpu", "Show the CPU details (first 8 lines of `lscpu`).", "lscpu | head -8", {"require": ["lscpu"]}),
        ("uname -a", "Print the full kernel/system information.", "uname -a"),
        ("whoami", "Print your user name and the machine's name, on two lines.", "whoami\nhostname",
         {"require": ["whoami", "hostname"]}),
    ]
    return catalog(rng, items, setup)


@template(M, "lite")
def job_control(rng):
    t = rng.randint(30, 300)
    t2 = rng.randint(2, 9)
    items = [
        ("&", f"Start `sleep {t}` in the background, then list your jobs.", f"sleep {t} &\njobs", {"require": ["&", "jobs"]}),
        ("jobs -l", f"Start `sleep {t}` and `sleep {t + 5}` in the background, then list your jobs with their PIDs.",
         f"sleep {t} &\nsleep {t + 5} &\njobs -l"),
        ("wait", f"Start `sleep {t2}` in the background, wait for all background jobs, then print `all done` and the "
                 f"time (`date +%T`).", f"sleep {t2} &\nwait\necho \"all done $(date +%T)\"", {"require": ["&", "wait"]}),
        ("kill", f"Start `sleep {t}` in the background, kill it by job number (`kill %1`), then wait for it and print "
                 f"its exit status.", f"sleep {t} &\nkill %1\nwait %1\necho $?", {"require": ["kill"]}),
        ("fg", f"Start `sleep {t2}` in the background, then bring it to the foreground with `fg %1` (it finishes), "
               f"then print the time.", f"sleep {t2} &\nfg %1\ndate +%T", {"require": ["fg"]}),
        ("bg", f"Start `sleep {t}` in the background, stop it with `kill -STOP %1`, list jobs, resume it in the "
               f"background with `bg %1`, and list jobs again.", f"sleep {t} &\nkill -STOP %1\njobs\nbg %1\njobs",
         {"require": ["bg"]}),
        ("disown", f"Start `sleep {t}` in the background, `disown -h %1` it (it won't get a hangup when you log out), "
                   f"then list jobs with `jobs -l`.", f"sleep {t} &\ndisown -h %1\njobs -l", {"require": ["disown -h"]}),
        ("$!", f"Start `sleep {t}` in the background, save its PID (`$!`) in `job.pid`, then kill it using that file.",
         f'sleep {t} &\necho $! > job.pid\nkill "$(cat job.pid)"', {"post": 'pgrep -c sleep || echo "no sleep left"',
                                                                  "require": ["$!", "kill"]}),
    ]
    return catalog(rng, items)


# ================================================================= easy

@template(M, "easy")
def stop_processes(rng):
    setup, scripts, cmds = _procs(rng)
    sc = scripts[0]
    check = f'pgrep -af "{sc}" || echo "{sc} stopped"'
    kind = rng.choice(["pkill", "kill-pgrep", "kill9", "pidfile"])
    if kind == "pkill":
        return P(f"Stop the process running `{sc}` by matching its command line (`pkill -f`).", setup,
                 f"pkill -f {sc}", post=check, require=["pkill -f"])
    if kind == "kill-pgrep":
        return P(f"Stop the `{sc}` process with `kill`, looking its PID up with `pgrep -f` in a command substitution.",
                 setup, f'kill "$(pgrep -f {sc})"', post=check, require=["kill", "pgrep|ps"])
    if kind == "kill9":
        return P(f"The `{sc}` process ignores polite requests. Force-kill it (`kill -9`) using `pgrep -f` to find it.",
                 setup, f'kill -9 $(pgrep -f {sc})',
                 post=check, require=["kill -9"])
    setup2 = setup + f"\npgrep -f {sc} > train.pid"
    return P(f"`train.pid` holds the PID of `{sc}`. Stop it politely (`kill`) using the file.", setup2,
             'kill "$(cat train.pid)"', post=check, require=["kill"])


@template(M, "easy")
def nohup_pidfile(rng):
    epochs = rng.randint(3, 5)
    body = ("import time\n" f"for e in range({epochs}):\n" "    time.sleep(1)\n"
            "    print(f'epoch {e} done', flush=True)\n" "print('finished')\n")
    setup = files({"train.py": body})
    kind = rng.choice(["nohup", "pid"])
    if kind == "nohup":
        return P("Start `python3 train.py` with `nohup` in the background, sending stdout and stderr to `train.log`, so "
                 "it survives closing the terminal.", setup, "nohup python3 train.py > train.log 2>&1 &",
                 post="sleep 10; cat train.log", require=["nohup", "&", "2>&1"])
    return P("Start `python3 train.py` in the background with `nohup` (output to `train.log`), save its PID in "
             "`train.pid`, then print `started` and whether that PID is alive (`kill -0 PID && echo alive`).", setup,
             "nohup python3 train.py > train.log 2>&1 &\necho $! > train.pid\necho started\n"
             'kill -0 "$(cat train.pid)" && echo alive', post="sleep 10; cat train.log", require=["nohup", "$!"])


# ================================================================= medium

@template(M, "medium")
def ports(rng):
    port = rng.choice([8888, 6006, 8000, 5000, 7860])
    prog = {8888: "jupyter-lab --port 8888", 6006: "tensorboard --logdir runs --port 6006",
            8000: "python3 -m http.server 8000", 5000: "python3 app.py --port 5000",
            7860: "python3 demo.py --port 7860"}[port]
    setup = f"#: proc cpu=2.0 port={port} -- {prog}"
    kind = rng.choice(["who", "kill", "pid"])
    check = f'lsof -i :{port} > /dev/null && echo "port {port} busy" || echo "port {port} free"'
    if kind == "who":
        return P(f"What's listening on port {port}? Show it with `lsof -i`.", setup, f"lsof -i :{port}",
                 require=["lsof -i"])
    if kind == "pid":
        return P(f"Print just the PID of the process using port {port} (`lsof -t`).", setup, f"lsof -t -i :{port}",
                 require=["lsof -t"])
    return P(f"Free port {port}: kill whatever is listening on it, using `lsof -t` in a command substitution.", setup,
             f'kill "$(lsof -t -i :{port})"', post=check, require=["kill", "lsof -t"])


@template(M, "medium")
def time_limits(rng):
    t = rng.randint(3, 9)
    lim = rng.randint(1, t - 1)
    slow = files({"slow.py": f"import time\ntime.sleep({t})\nprint('finished')\n"})
    kind = rng.choice(["timeout", "time", "nice", "sleep"])
    if kind == "timeout":
        return P(f"`slow.py` takes {t} s. Run it with a {lim}-second limit (`timeout`) and print the exit status "
                 f"(124 means it was cut off).", slow, f"timeout {lim} python3 slow.py\necho $?", require=["timeout"])
    if kind == "time":
        return P("Measure how long `python3 slow.py` takes with `time` (the timing goes to stderr).", slow,
                 "time python3 slow.py", answer=["<stdout>", "<stderr>"], require=["time"])
    if kind == "nice":
        n = rng.choice([5, 10, 15, 19])
        return P(f"Start `sleep 300` in the background at a lower priority (niceness {n}, `nice -n`), then print its "
                 f"niceness with `ps -o ni= -p $!`.", "", f"nice -n {n} sleep 300 &\nps -o ni= -p $!", require=["nice -n"])
    return P(f"Wait {t} seconds with `sleep`, then print the time.", "", f"sleep {t}\ndate +%T", require=["sleep"])


# ================================================================= hard

@template(M, "hard")
def busiest(rng):
    setup, scripts, cmds = _procs(rng, rng.randint(3, 5))
    kind = rng.choice(["cpu", "mem", "user"])
    if kind == "cpu":
        return P("Kill the process using the most CPU (`ps` sorted by `%cpu`; skip the header).", setup,
                 "kill $(ps -eo pid --sort=-%cpu --no-headers | head -1)", post="ps -eo pcpu,args --sort=-%cpu | head -3",
                 require=["kill", "ps"])
    if kind == "mem":
        return P("Print the command line of the process using the most memory (`ps -eo ... --sort=-%mem`).", setup,
                 "ps -eo args --sort=-%mem --no-headers | head -1", require=["ps"])
    return P("Print how many processes YOU (`$USER`) are running right now, according to `ps -u`.", setup,
             'ps -u "$USER" --no-headers | wc -l', require=["ps"])


@template(M, "hard")
def parallel_jobs(rng):
    n = rng.randint(3, 4)
    durs = [rng.randint(1, 6) for _ in range(n)]
    codes = [rng.choice([0, 0, 1, 2]) for _ in range(n)]
    setup = files({"task.sh": '#!/bin/bash\nsleep "$1"\necho "task $2 done"\nexit "$3"\n'}) + "\nchmod +x task.sh"
    launches = " ".join(f"{d}:{c}" for d, c in zip(durs, codes))
    sol = ('pids=()\ni=0\nfor spec in ' + launches + '; do\n  ./task.sh "${spec%:*}" $i "${spec#*:}" &\n  pids+=($!)\n'
           '  ((i++))\ndone\nfor j in "${!pids[@]}"; do wait "${pids[j]}"; echo "task $j exit $?"; done')
    return P(f"Run `./task.sh SECONDS ID CODE` for these (SECONDS:CODE) specs in parallel, IDs 0, 1, ...: "
             f"`{launches}`. Collect each PID, then wait for each one in launch order and print `task ID exit CODE`.",
             setup, sol, require=["&", "wait", "$!"])


# ================================================================= challenge

def _users_procs(rng, n):
    lines = []
    for i in range(n):
        u = rng.choice(["ada", "bob", "carol", "dave"])
        lines.append(f"#: proc user={u} cpu={rng.choice([0.5, 1.5, 12.0, 25.0, 50.0, 99.9])} -- "
                     f"python3 job_{i}.py")
    return "\n".join(lines)


@template(M, "challenge")
def cpu_by_user(rng):
    setup = _users_procs(rng, 5)
    hidden = [_users_procs(rng, rng.randint(15, 30)) for _ in range(2)]
    sol = ("ps -eo user=,pcpu=,comm= | awk '$3 == \"python3\" {c[$1] += $2; n[$1]++} END {for (u in c) "
           "printf \"%s %d %.1f\\n\", u, n[u], c[u]}' | sort -k3,3nr -k1,1")
    return challenge("Who's hogging the CPU?", "For every user running `python3` processes, print `USER COUNT TOTAL`: "
                     "how many python3 processes they run and their summed %CPU (one decimal), highest total first "
                     "(ties: by user name). Ignore all other processes.", setup, sol, hidden, "up to 30 processes",
                     require=["ps"])


def _tasks(rng, n):
    return files({"tasks.txt": "\n".join(f"t{i} {rng.randint(1, 9)}" for i in range(n)) + "\n"})


@template(M, "challenge")
def bounded_parallelism(rng):
    k = rng.choice([2, 3])
    setup = _tasks(rng, 4)
    hidden = [_tasks(rng, rng.randint(8, 14)) for _ in range(2)]
    sol = (f'SECONDS=0\nrunning=0\nwhile read -r name secs; do\n  if (( running == {k} )); then wait -n; ((running--)); fi\n'
           '  (sleep "$secs"; echo "$name finished at ${SECONDS}s") &\n  ((running++))\ndone < tasks.txt\nwait\n'
           'echo "total ${SECONDS}s"')
    return challenge("At most K at a time", f"Each line of `tasks.txt` is `NAME SECONDS`. Run the tasks (a task is "
                     f"`sleep SECONDS`) in file order, with at most {k} running at once: start the next one as soon as "
                     f"ANY running task ends (`wait -n`). When a task ends print `NAME finished at Ts` (seconds since the "
                     f"start, from `$SECONDS`), and finally `total Ts`.", setup, sol, hidden, "up to 14 tasks",
                     require=["&", "wait"])
