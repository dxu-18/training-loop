"""7.10 Conditionals & loops."""

from ._common import DATASETS, MODELS, NAMES, TEAMS, WORDS, P, catalog, challenge, csv_text, files, q, template

M = "7.10"


# ================================================================= lite

@template(M, "lite")
def tests(rng):
    f = rng.choice(["data.csv", "train.csv", "notes.md"])
    d = rng.choice(["data", "runs", "logs"])
    empty = rng.choice(["empty.txt", "blank.log"])
    script = rng.choice(["run.sh", "go.sh"])
    setup = files({f: "x,y\n", f"{d}/a.txt": "a\n", empty: None, script: "echo hi\n"}) + f"\nchmod +x {script}"
    path = rng.choice([f, d, "missing.txt"])
    n = rng.randint(1, 10)
    lim = rng.randint(3, 8)
    s = rng.choice(["", rng.choice(WORDS)])
    items = [
        ("if", f"Print `file` if `{path}` is a regular file, `dir` if it's a directory, `missing` otherwise "
               f"(`if [[ -f ... ]]; elif [[ -d ... ]]; else ...; fi`).",
         f'if [[ -f {path} ]]; then echo "file"; elif [[ -d {path} ]]; then echo "dir"; else echo "missing"; fi',
         {"require": ["if", "[[ -f", "[[ -d"]}),
        ("[[ -e", f"Print `yes` if `{path}` exists at all (file or directory), else `no`.",
         f"[[ -e {path} ]] && echo yes || echo no"),
        ("[[ -s", f"Print `has data` if `{empty}` is non-empty, else `empty`.",
         f'[[ -s {empty} ]] && echo "has data" || echo "empty"'),
        ("[[ -x", f"Print `runnable` if `{script}` is executable, else `not runnable`.",
         f'[[ -x {script} ]] && echo "runnable" || echo "not runnable"'),
        ("[[ -z", f"`s` holds `{s}`. Print `empty` if it's an empty string, else `set to $s`.",
         f'if [[ -z $s ]]; then echo "empty"; else echo "set to $s"; fi', {"setup": f"s={q(s)}"}),
        ("[[ -n", f"`s` holds `{s}`. Print `non-empty` if the string isn't empty, else `nothing`.",
         f'[[ -n $s ]] && echo "non-empty" || echo "nothing"', {"setup": f"s={q(s)}"}),
        ("[[ -eq", f"`n={n}`. Print `equal` if n equals {lim}, else `different` (`-eq`).",
         f'[[ $n -eq {lim} ]] && echo equal || echo different', {"setup": f"n={n}"}),
        ("[[ -gt", f"`n={n}`. Print `big` if n is greater than {lim}, else `small` (`-gt`).",
         f'if [[ $n -gt {lim} ]]; then echo big; else echo small; fi', {"setup": f"n={n}"}),
        ("((", f"`n={n}`. Print `big` if n > {lim} using arithmetic `(( ))`, else `small`.",
         f"if (( n > {lim} )); then echo big; else echo small; fi", {"setup": f"n={n}"}),
        ("[[ !", f"Print `ok` if `{f}` exists and `{f}.bak` does NOT exist (`&&` and `!` inside `[[ ]]`), else `no`.",
         f'[[ -f {f} && ! -f {f}.bak ]] && echo ok || echo no', {"require": ["[[ !", "[[ -f"]}),
    ]
    return catalog(rng, items, setup)


@template(M, "lite")
def loops(rng):
    seeds = sorted(rng.sample(range(10), rng.randint(2, 4)))
    n = rng.randint(3, 7)
    csvs = [f"data/{d}.csv" for d in rng.sample(DATASETS, 3)]
    lines = [f"{rng.choice(WORDS)} {rng.choice(WORDS)}" for _ in range(rng.randint(3, 5))]
    skip, stop = sorted(rng.sample(range(2, 9), 2))
    setup = files({**{c: "x\n" for c in csvs}, "list.txt": "\n".join(lines) + "\n"})
    items = [
        ("for", f"Print `seed N` for each seed in `{' '.join(map(str, seeds))}` with a for loop.",
         f'for seed in {" ".join(map(str, seeds))}; do echo "seed $seed"; done'),
        ("for", "Print the path of every `.csv` file in `data/` with a for loop over a glob.",
         'for f in data/*.csv; do echo "$f"; done', {"require": ["for", "glob"]}),
        ("for((", f"Print 0 to {n - 1} with a C-style `for ((...))` loop.", f'for ((i = 0; i < {n}; i++)); do echo "$i"; done'),
        ("while", "Print each line of `list.txt` with `while read -r`.", 'while read -r line; do echo "$line"; done < list.txt',
         {"require": ["while", "read -r"]}),
        ("continue", f"Loop over 1..10; skip {skip} (`continue`) and stop the loop at {stop} (`break`); print the others.",
         f'for i in {{1..10}}; do [[ $i -eq {skip} ]] && continue; [[ $i -eq {stop} ]] && break; echo "$i"; done',
         {"require": ["continue", "break"]}),
        ("until", f"Count from 1 to {n} with an `until` loop.", f'i=1\nuntil (( i > {n} )); do echo "$i"; ((i++)); done'),
    ]
    return catalog(rng, items, setup)


# ================================================================= easy

@template(M, "easy")
def case_dispatch(rng):
    cmds = rng.sample(["train", "eval", "test", "serve", "export", "clean"], 3)
    arg = rng.choice(cmds + ["deploy"])
    script = (f'case "$1" in\n  {cmds[0]}) echo "running {cmds[0]}" ;;\n  {cmds[1]}|{cmds[2]}) echo "running {cmds[1]}/{cmds[2]}" ;;\n'
              f'  *) echo "usage: $0 {{{"|".join(cmds)}}}" ;;\nesac')
    return P(f"Write a script `run.sh` whose `case` statement prints `running {cmds[0]}` for the argument `{cmds[0]}`, "
             f"`running {cmds[1]}/{cmds[2]}` for `{cmds[1]}` or `{cmds[2]}` (one pattern with `|`), and "
             f"`usage: ./run.sh {{{'|'.join(cmds)}}}` otherwise (use `$0`). It's run as `bash run.sh {arg}` and "
             f"`bash run.sh`.", f"#: script run.sh\n#: run bash ./run.sh {arg}\n#: run bash ./run.sh", script,
             require=["case"])


@template(M, "easy")
def string_matching(rng):
    name = rng.choice([f"run_{rng.randint(1, 999)}", f"{rng.choice(WORDS)}.csv", f"exp-{rng.randint(1, 9)}", "notes.md"])
    kind = rng.choice(["glob", "regex", "compare", "not-equal"])
    setup = f"s={q(name)}"
    if kind == "glob":
        return P("`s` holds a file name. Print `csv` if it ends in `.csv` (a glob pattern inside `[[ ]]`), else `other`.",
                 setup, '[[ $s == *.csv ]] && echo csv || echo other', require=["[[ =="])
    if kind == "regex":
        return P("If `s` looks like `run_` followed by digits, print the number (from `BASH_REMATCH`), otherwise print "
                 "`no match`. Use `=~` with a capture group.", setup,
                 'if [[ $s =~ ^run_([0-9]+)$ ]]; then echo "${BASH_REMATCH[1]}"; else echo "no match"; fi',
                 require=["[[ =~"])
    other = rng.choice(WORDS)
    if kind == "compare":
        return P(f"Print `before` if `$s` sorts before `{other}` (string `<` inside `[[ ]]`), else `after`.", setup,
                 f'[[ $s < {other} ]] && echo before || echo after', require=["[[ <"])
    return P(f"Print `changed` if `$s` is not `notes.md` (`!=`), else `same`.", setup,
             '[[ $s != notes.md ]] && echo changed || echo same', require=["[[ !="])


# ================================================================= medium

@template(M, "medium")
def parse_csv(rng):
    rows = [(rng.randint(100, 999), rng.choice(["cat", "dog", "bird", "fish"])) for _ in range(rng.randint(3, 6))]
    setup = files({"labels.csv": csv_text(("id", "label"), rows)})
    kind = rng.choice(["arrow", "filter", "sentence"])
    if kind == "arrow":
        return P("Print `ID -> LABEL` for each row of `labels.csv`, skipping the header, with "
                 "`while IFS=, read -r ... done < <(tail -n +2 labels.csv)`.", setup,
                 'while IFS=, read -r id label; do echo "$id -> $label"; done < <(tail -n +2 labels.csv)',
                 require=["while", "read -r", "<("])
    want = rng.choice(["cat", "dog"])
    if kind == "filter":
        return P(f"Print the ids labelled `{want}` (skip the header), reading the CSV with a `while read` loop.", setup,
                 f'while IFS=, read -r id label; do [[ $label == {want} ]] && echo "$id"; done < <(tail -n +2 labels.csv)',
                 require=["while", "read"])
    return P("For each row of `labels.csv` (skip the header) print a sentence: `Image ID shows a LABEL.`", setup,
             'tail -n +2 labels.csv | while IFS=, read -r id label; do echo "Image $id shows a $label."; done',
             require=["while", "read"])


@template(M, "medium")
def and_or_chains(rng):
    d = rng.choice(["out", "results", "build"])
    kind = rng.choice(["mkcd", "fallback", "check-each"])
    if kind == "mkcd":
        return P(f"Create `{d}` (no error if it exists) and, only if that worked, `cd` into it; if anything fails, "
                 f"`exit 1`. Then print the current directory.", "", f"mkdir -p {d} && cd {d} || exit 1\npwd",
                 require=["&&", "||"])
    if kind == "fallback":
        f = rng.choice(["config.local.yaml", "settings.json"])
        return P(f"Print `{f}` if it exists; otherwise print `using defaults` (one line with `&&` / `||`).",
                 rng.choice(["", files({f: "lr: 0.1\n"})]), f'[[ -f {f} ]] && cat {f} || echo "using defaults"',
                 require=["&&", "||"])
    paths = rng.sample(["data.csv", "model.pt", "README.md", "src", "notes"], 3)
    setup = files({paths[0]: "x\n"}, mkdirs=[paths[1]])
    return P(f"For each of `{' '.join(paths)}` print `NAME exists` or `NAME missing`.", setup,
             f'for p in {" ".join(paths)}; do [[ -e $p ]] && echo "$p exists" || echo "$p missing"; done',
             require=["for", "[[ -e|[[ -f"])


# ================================================================= hard

@template(M, "hard")
def wait_for_it(rng):
    host = rng.choice(["gpu-box", "db", "api-server", "trainer"])
    up = rng.choice([7, 12, 18, 26])
    every = rng.choice([3, 5])
    kind = rng.choice(["until-ping", "count-tries"])
    setup = f"#: host {host} up={up}"
    if kind == "until-ping":
        return P(f"`{host}` is rebooting. Wait until it answers one ping (`ping -c1`, output hidden), checking every "
                 f"{every} s, then print `{host} is up at HH:MM:SS` (`date +%T`).", setup,
                 f'until ping -c1 {host} > /dev/null 2>&1; do sleep {every}; done\necho "{host} is up at $(date +%T)"',
                 require=["until", "ping -c"])
    return P(f"Ping `{host}` once every {every} s (`ping -c1`, output hidden) until it answers, then print how many "
             f"failed tries there were, as `N failed pings`.", setup,
             f'n=0\nuntil ping -c1 {host} > /dev/null 2>&1; do ((n++)); sleep {every}; done\necho "$n failed pings"',
             require=["until|while"])


@template(M, "hard")
def line_counts(rng):
    logs = {f"logs/{rng.choice(MODELS)}_{i}.log": "".join(f"{rng.choice(WORDS)}\n" for _ in range(rng.randint(1, 30)))
            for i in range(rng.randint(3, 5))}
    setup = files(logs)
    lim = rng.randint(8, 20)
    kind = rng.choice(["all", "long"])
    if kind == "all":
        return P("Loop over the `.log` files in `logs/` and print each name with its line count, like "
                 "`logs/x.log: 12 lines`.", setup,
                 'for f in logs/*.log; do echo "$f: $(wc -l < "$f") lines"; done', require=["for"])
    return P(f"Print the names of the `.log` files in `logs/` that have more than {lim} lines (a loop and a numeric "
             f"test).", setup, f'for f in logs/*.log; do (( $(wc -l < "$f") > {lim} )) && echo "$f"; done',
             require=["for"])


# ================================================================= challenge

def _flaky(rng):
    need = rng.randint(1, 6)
    return (f"cat > flaky.sh <<'EOF'\n#!/bin/bash\nn=$(cat .attempts 2>/dev/null || echo 0)\necho $((n + 1)) > .attempts\n"
            f"if (( n + 1 < {need} )); then echo \"attempt $((n + 1)) failed\" >&2; exit 1; fi\necho \"attempt $((n + 1)) ok\"\nEOF\n"
            "chmod +x flaky.sh")


@template(M, "challenge")
def retry_backoff(rng):
    tries = rng.randint(4, 6)
    setup = _flaky(rng)
    hidden = [_flaky(rng) for _ in range(3)]
    sol = (f'delay=1\nfor ((i = 1; i <= {tries}; i++)); do\n  if ./flaky.sh 2>/dev/null; then echo "succeeded on try $i"; break; fi\n'
           f'  if (( i == {tries} )); then echo "giving up after {tries} tries"; break; fi\n  echo "waiting ${{delay}}s"\n'
           '  sleep "$delay"\n  delay=$((delay * 2))\ndone')
    return challenge("Retry with backoff", f"`./flaky.sh` fails a few times before it works. Run it up to {tries} times "
                     f"(hide its errors). After a failure, print `waiting Ns` and sleep N seconds, doubling N each time "
                     f"(1, 2, 4, ...). On success print what it printed, then `succeeded on try K`; if all {tries} tries fail, "
                     f"print `giving up after {tries} tries` instead of waiting again.", setup, sol, hidden,
                     f"at most {tries} tries")


def _kv(rng, n):
    lines = []
    for _ in range(n):
        k = rng.choice(["lr", "epochs", "batch_size", "dropout", "name", "seed"])
        v = rng.choice([str(rng.randint(1, 512)), f"0.{rng.randint(1, 9)}", rng.choice(WORDS), "", "-3"])
        lines.append(f"{k}={v}" if rng.random() < 0.9 else f"# {k} comment")
    return files({"params.env": "\n".join(lines) + "\n"})


@template(M, "challenge")
def validate_params(rng):
    setup = _kv(rng, 6)
    hidden = [_kv(rng, rng.randint(30, 50)) for _ in range(2)]
    sol = ('n=0\nwhile IFS='"'"'='"'"' read -r key val; do\n  ((n++))\n  [[ $key == \\#* ]] && continue\n'
           '  case $key in\n    epochs|batch_size|seed) [[ $val =~ ^[0-9]+$ ]] || echo "line $n: $key must be a non-negative integer" ;;\n'
           '    lr|dropout) [[ $val =~ ^[0-9]*\\.?[0-9]+$ ]] || echo "line $n: $key must be a number" ;;\n'
           '    name) [[ -n $val ]] || echo "line $n: name must not be empty" ;;\n  esac\ndone < params.env')
    return challenge("Validate parameters", "Check every `KEY=VALUE` line of `params.env` (skip lines starting with "
                     "`#`, but count them in the line numbers). `epochs`, `batch_size` and `seed` must be non-negative "
                     "integers; `lr` and `dropout` must be numbers like `3`, `0.5` or `.5` (no sign); `name` must not "
                     "be empty. For each violation print `line N: KEY must be a non-negative integer` / "
                     "`... must be a number` / `line N: name must not be empty`.", setup, sol, hidden, "up to 50 lines")
