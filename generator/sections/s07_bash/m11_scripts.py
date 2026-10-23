"""7.11 Functions & scripts."""

from ._common import DATASETS, MODELS, NAMES, WORDS, P, catalog, challenge, files, q, template

M = "7.11"


def _runs(lines):
    return "\n".join(f"#: run {x}" for x in lines)


# ================================================================= lite

@template(M, "lite")
def functions(rng):
    name = rng.choice(NAMES)
    words = rng.sample(WORDS, 3)
    n = rng.randint(2, 9)
    items = [
        ("function", f"Define a function `greet` that prints `Hello, NAME!` for its first argument, then call "
                     f"`greet {name}`.", f'greet() {{ echo "Hello, $1!"; }}\ngreet {name}'),
        ("$#", f"Define a function `count` that prints how many arguments it got, then call `count {' '.join(words)}`.",
         f'count() {{ echo "$#"; }}\ncount {" ".join(words)}'),
        ("array[@]", f"Define `each` that prints each of its arguments on its own line (loop over `\"$@\"`), then call "
                     f"`each {words[0]} \"{words[1]} {words[2]}\"`.",
         f'each() {{ for a in "$@"; do echo "$a"; done; }}\neach {words[0]} "{words[1]} {words[2]}"', {"require": ["function", "for"]}),
        ("$*", f"Define `log` that prints `[log] ` followed by all its arguments joined with spaces (`$*`), then call "
               f"`log {' '.join(words)}`.", f'log() {{ echo "[log] $*"; }}\nlog {" ".join(words)}'),
        ("local", f"`x` is `outer`. Define `f` that sets a LOCAL `x={n}` and prints it; call `f`, then print `$x` "
                  f"(still `outer`).", f'x=outer\nf() {{ local x={n}; echo "$x"; }}\nf\necho "$x"'),
        ("return", f"Define `is_even` that returns 0 when its argument is even and 1 otherwise (`return`), then print "
                   f"`even` or `odd` for {n} using it.", f'is_even() {{ return $(( $1 % 2 )); }}\nis_even {n} && echo even || echo odd'),
        ("$?", "Run `false`, then print the exit status it left in `$?`; then run `true` and print `$?` again.",
         "false\necho $?\ntrue\necho $?"),
        ("printf", f"Define `log` that prints its arguments after the current time in brackets, like `[09:41:07] start`: "
                   f"`printf '[%s] %s\\n' \"$(date +%T)\" \"$*\"`. Then call `log {' '.join(words[:2])}`.",
         f'log() {{ printf \'[%s] %s\\n\' "$(date +%T)" "$*"; }}\nlog {" ".join(words[:2])}',
         {"setup": f"#: date 2026-10-{n + 10} {n + 8}:{n * 5:02d}:0{n}", "require": ["function", "printf", "date"]}),
        ("python", f"Set `epochs={n}` and define `train`, which stores its first argument in a LOCAL variable `cfg` and "
                   f"runs `python train.py --config CFG --epochs EPOCHS`. Call `train configs/{words[0]}.yaml`.",
         f'epochs={n}\ntrain() {{\n  local cfg=$1\n  python train.py --config "$cfg" --epochs "$epochs"\n}}\n'
         f"train configs/{words[0]}.yaml",
         {"setup": files({"train.py": "import sys\nprint('training with', ' '.join(sys.argv[1:]))\n"}),
          "require": ["function", "local", "python"]}),
        ("chmod", f"Write `go.sh` with a here-document containing the line `echo \"args: $*\"`, make it executable, and "
                  f"run it as `./go.sh -e {n} configs/base.yaml`.",
         f"cat > go.sh <<'EOF'\necho \"args: $*\"\nEOF\nchmod +x go.sh\n./go.sh -e {n} configs/base.yaml",
         {"require": ["chmod", "<<"]}),
    ]
    return catalog(rng, items)


@template(M, "lite")
def script_args(rng):
    a = rng.sample(WORDS, rng.randint(2, 4))
    kind = rng.choice(["first", "count", "all", "name", "shebang"])
    run = "./run.sh " + " ".join(a)
    head = "#: script run.sh\n" + _runs([run])
    if kind == "first":
        return P(f"Write `run.sh` so it prints its first argument. It's run as `{run}`.", head, 'echo "$1"', require=["$1"])
    if kind == "count":
        return P(f"Write `run.sh` so it prints `got N arguments`. It's run as `{run}`.", head,
                 'echo "got $# arguments"', require=["$#"])
    if kind == "all":
        return P(f"Write `run.sh` so it prints each argument on its own line, numbered like `1: x`. It's run as "
                 f"`{run}`.", head, 'i=1\nfor a in "$@"; do echo "$i: $a"; ((i++)); done', require=["for"])
    if kind == "name":
        return P(f"Write `run.sh` so it prints the name it was called by (`$0`) and then its last argument "
                 f"(`${{!#}}`). It's run as `{run}`.", head, 'echo "$0"\necho "${!#}"', require=["$0"])
    return P(f"Write `run.sh` as a proper bash script (start with the `#!/usr/bin/env bash` shebang) that prints "
             f"`hello from bash`. It's run as `./run.sh`.", "#: script run.sh\n#: run ./run.sh\n#: run head -1 run.sh",
             '#!/usr/bin/env bash\necho "hello from bash"', require=["echo"])


# ================================================================= easy

@template(M, "easy")
def getopts_script(rng):
    e_def, g_def = rng.choice([10, 20, 50]), 0
    cfg = f"configs/{rng.choice(MODELS)}.yaml"
    runs = [f"./run.sh {cfg}", f"./run.sh -e {rng.randint(2, 9)} {cfg}", f"./run.sh -g {rng.randint(1, 3)} -e 3 {cfg}"]
    sol = (f'epochs={e_def}; gpu={g_def}\nwhile getopts "e:g:" opt; do\n  case $opt in\n    e) epochs=$OPTARG ;;\n'
           '    g) gpu=$OPTARG ;;\n  esac\ndone\nshift $((OPTIND - 1))\necho "config=$1 epochs=$epochs gpu=$gpu"')
    return P(f"Write `run.sh`: it takes options `-e EPOCHS` (default {e_def}) and `-g GPU` (default {g_def}) parsed "
             f"with `getopts`, then one config file argument. It prints `config=FILE epochs=N gpu=G`.",
             "#: script run.sh\n" + _runs(runs), sol, require=["getopts", "shift"])


@template(M, "easy")
def source_vs_run(rng):
    var = rng.choice(["DATA_DIR", "WANDB_PROJECT", "HF_HOME", "EXP_NAME"])
    val = rng.choice(["/data/hf", "demo", "baseline", "/scratch/ada"])
    setup = files({"env.sh": f"{var}={val}\nexport PATH=\"$HOME/tools:$PATH\"\n"}) + "\nchmod +x env.sh"
    kind = rng.choice(["source", "both"])
    if kind == "source":
        return P(f"`env.sh` sets `{var}`. Load it into the CURRENT shell (so the variable stays set) and print "
                 f"`${var}`.", setup, f"source env.sh\necho \"${var}\"", require=["source|."])
    return P(f"Show the difference: run `./env.sh` as a child process and print `[${var}]` (empty), then `source` it "
             f"and print `[${var}]` again.", setup, f"./env.sh\necho \"[${var}]\"\nsource env.sh\necho \"[${var}]\"",
             require=["source|."])


# ================================================================= medium

@template(M, "medium")
def strict_mode(rng):
    v = rng.choice(["epochs", "config", "out_dir"])
    kind = rng.choice(["errexit", "nounset", "pipefail"])
    if kind == "errexit":
        f = rng.choice(["missing.csv", "nope.txt"])
        return P(f"Write `job.sh` that starts with `set -euo pipefail`, prints `start`, runs `cat {f}` (which fails), "
                 f"and then prints `end`. Thanks to `-e` it stops after the error. It's run as `bash job.sh; echo "
                 f"\"exit $?\"`.", "#: script job.sh\n#: run bash job.sh; echo \"exit $?\"",
                 f'set -euo pipefail\necho start\ncat {f}\necho end', answer=["<stdout>", "<stderr>"],
                 require=["set -e|set -euo"])
    if kind == "nounset":
        return P(f"Write `job.sh` with `set -u` that prints `using ${v}` without setting `{v}`: bash must stop with "
                 f"an `unbound variable` error. It's run as `bash job.sh; echo \"exit $?\"`.",
                 "#: script job.sh\n#: run bash job.sh; echo \"exit $?\"", f'set -u\necho "using ${v}"',
                 answer=["<stdout>", "<stderr>"], require=["set -u|set -euo"])
    return P("Write `job.sh` that turns on `pipefail`, runs `grep needle /nonexistent | sort`, and prints `pipeline "
             "status N` with the pipeline's exit status. It's run as `bash job.sh`.", "#: script job.sh\n#: run bash job.sh",
             'set -o pipefail\ngrep needle /nonexistent | sort\necho "pipeline status $?"', answer=["<stdout>"],
             require=["set -o"])


@template(M, "medium")
def cleanup_trap(rng):
    word = rng.choice(WORDS)
    kind = rng.choice(["exit-trap", "on-error"])
    if kind == "exit-trap":
        sol = ('tmp=$(mktemp -d)\ntrap \'rm -rf "$tmp"; echo "cleaned up"\' EXIT\necho "' + word + '" > "$tmp/data.txt"\n'
               'cat "$tmp/data.txt"\necho "$tmp" > /tmp/last_tmp')
        return P(f"Write `job.sh`: make a temporary directory with `mktemp -d`, set an `EXIT` trap that deletes it and "
                 f"prints `cleaned up`, write `{word}` into a file inside it, print that file, and save the directory "
                 f"path in `/tmp/last_tmp`. The test then checks the directory is gone.",
                 "#: script job.sh\n#: run bash job.sh\n#: run [[ -d $(cat /tmp/last_tmp) ]] && echo still there || echo gone",
                 sol, require=["trap", "mktemp -d"])
    sol = ('trap \'echo "failed at line $LINENO" >&2\' ERR\nset -e\necho step1\nfalse\necho step2')
    return P("Write `job.sh` with an `ERR` trap that prints `failed at line N` (use `$LINENO`) to stderr, then "
             "`set -e`, print `step1`, run `false`, and print `step2` (never reached).",
             "#: script job.sh\n#: run bash job.sh", sol, answer=["<stdout>", "<stderr>"], require=["trap"])


# ================================================================= hard

@template(M, "hard")
def trace_debug(rng):
    xs = rng.sample(range(1, 9), rng.randint(2, 3))
    body = f'total=0\nfor x in {" ".join(map(str, xs))}; do\n  total=$((total + x))\ndone\necho "total=$total"\n'
    setup = files({"sum.sh": body})
    kind = rng.choice(["trace", "set-x"])
    if kind == "trace":
        return P("Run `sum.sh` under `bash -x` to see every command it executes (the trace goes to stderr).", setup,
                 "bash -x sum.sh", answer=["<stdout>", "<stderr>"], require=["bash -x"])
    return P("Trace only the loop of `sum.sh`: copy it to `traced.sh` with `set -x` added before the loop and "
             "`set +x` after it (use sed or rewrite it), then run `bash traced.sh`.", setup,
             "sed -e 's/^for /set -x\\nfor /' -e 's/^done$/done\\nset +x/' sum.sh > traced.sh\nbash traced.sh",
             answer=["<stdout>", "<stderr>"], require=["bash"])


@template(M, "hard")
def lint_fix(rng):
    d = rng.choice(["$dir", "$target", "$src"])
    name = d[1:]
    bad = f'#!/bin/bash\n{name}="my data"\nmkdir -p {d}\nfor f in $(ls *.csv); do\n  cp $f {d}/\ndone\n'
    good = f'#!/bin/bash\n{name}="my data"\nmkdir -p "{d}"\nfor f in *.csv; do\n  cp "$f" "{d}/"\ndone\n'
    setup = files({"copy.sh": bad, "a.csv": "1\n", "b c.csv": "2\n"})
    kind = rng.choice(["lint", "fix"])
    if kind == "lint":
        return P("Lint `copy.sh` with `shellcheck` and show its report.", setup, "shellcheck copy.sh",
                 require=["shellcheck"])
    return P("`copy.sh` breaks on names with spaces (shellcheck complains about unquoted variables and `$(ls)`). "
             f"Write a fixed version to `copy.sh` that shellcheck accepts (quote `{d}` and `$f`, loop over a glob), then "
             "run `shellcheck copy.sh && echo clean` and `bash copy.sh`.", setup,
             f"cat > copy.sh <<'EOF'\n{good}EOF\nshellcheck copy.sh && echo clean\nbash copy.sh",
             post=f'ls "my data"', require=["shellcheck"])


# ================================================================= challenge

def _launch_runs(rng, n):
    lines = []
    cfgs = [f"configs/{m}.yaml" for m in rng.sample(MODELS, 3)]
    for _ in range(n):
        args = []
        if rng.random() < 0.5:
            args += ["-e", str(rng.randint(1, 99))]
        if rng.random() < 0.4:
            args += ["-g", str(rng.randint(0, 7))]
        if lines and rng.random() < 0.1:
            args += ["-x"]
        args += rng.sample(cfgs, 1 if not lines else rng.choice([0, 1, 1, 1, 1, 2]))
        lines.append("./launch.sh " + " ".join(args) + '; echo "exit $?"')
    return "#: script launch.sh\n" + _runs(lines)


@template(M, "challenge")
def launcher(rng):
    setup = _launch_runs(rng, 4)
    hidden = [_launch_runs(rng, rng.randint(12, 18)) for _ in range(2)]
    sol = ('usage() { echo "usage: $0 [-e epochs] [-g gpu] config.yaml" >&2; exit 1; }\nepochs=10; gpu=0\n'
           'while getopts "e:g:" opt; do\n  case $opt in\n    e) epochs=$OPTARG ;;\n    g) gpu=$OPTARG ;;\n    *) usage ;;\n'
           '  esac\ndone\nshift $((OPTIND - 1))\n[[ $# -eq 1 ]] || usage\n'
           'echo "CUDA_VISIBLE_DEVICES=$gpu python train.py --config $1 --epochs $epochs"')
    return challenge("A training launcher", "Write `launch.sh`. It takes `-e EPOCHS` (default 10) and `-g GPU` "
                     "(default 0) via getopts, then exactly ONE config file. It prints the command it would run: "
                     "`CUDA_VISIBLE_DEVICES=G python train.py --config FILE --epochs N`. On a bad option or a wrong "
                     "number of arguments it prints `usage: ./launch.sh [-e epochs] [-g gpu] config.yaml` (with `$0`) "
                     "to stderr and exits 1 (getopts prints its own error line first for a bad option).", setup, sol,
                     hidden, "up to 18 invocations", answer=["<stdout>", "<stderr>"], require=["getopts"])


def _slug_runs(rng, n):
    titles = []
    for _ in range(n):
        ws = rng.sample(WORDS, rng.randint(1, 4))
        t = rng.choice([" ", "  ", "_", "-", " - "]).join(w.capitalize() if rng.random() < 0.5 else w.upper() for w in ws)
        if rng.random() < 0.3:
            t = rng.choice(["  ", "--", "!"]) + t + rng.choice(["!", "?", "  ", "."])
        titles.append(t)
    return "#: script lib.sh\n" + _runs([f"source ./lib.sh; slugify {q(t)}" for t in titles])


@template(M, "challenge")
def slug_library(rng):
    setup = _slug_runs(rng, 4)
    hidden = [_slug_runs(rng, rng.randint(15, 25)) for _ in range(2)]
    sol = ('slugify() {\n  local s=${1,,}\n  s=$(printf "%s" "$s" | tr -cs "a-z0-9" "-")\n  s=${s#-}\n  s=${s%-}\n'
           '  echo "$s"\n}')
    return challenge("A slug library", "Write `lib.sh` defining a function `slugify` (it's sourced, then called with one "
                     "title). It prints the title in lower case with every run of characters other than `a-z` and `0-9` "
                     "turned into a single `-`, and no `-` at either end: `  Hello  World!` becomes `hello-world`.",
                     setup, sol, hidden, "up to 25 titles", require=["function"])
