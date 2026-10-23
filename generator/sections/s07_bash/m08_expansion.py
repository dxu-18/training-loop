"""7.08 Variables, quoting & expansion."""

from ._common import DATASETS, MODELS, NAMES, WORDS, P, catalog, challenge, files, q, template

M = "7.08"


def _path(rng):
    root = rng.choice(["experiments", "runs", "outputs", "checkpoints"])
    run = f"{rng.choice(['run', 'exp', 'trial'])}_{rng.randint(1, 999):03d}"
    name = rng.choice(MODELS).replace("-", "_")
    ext = rng.choice(["pt", "ckpt", "bin", "safetensors"])
    return root, run, name, ext, f"{root}/{run}/{name}.{ext}"


# ================================================================= lite

@template(M, "lite")
def parameter_ops(rng):
    root, run, name, ext, s = _path(rng)
    k = rng.randint(3, len(root))
    setup = f"s={q(s)}"
    new_ext = rng.choice(["onnx", "pt2", "tflite"])
    items = [
        ("param:len", "Print the length of `$s` (number of characters).", 'echo "${#s}"'),
        ("param:sub", f"Print the first {k} characters of `$s` (substring expansion).", f'echo "${{s:0:{k}}}"'),
        ("param:#", "Print `$s` without its first path component (strip the shortest prefix matching `*/`).",
         'echo "${s#*/}"'),
        ("param:##", "Print just the file name of `$s` with parameter expansion (strip the longest prefix matching "
                     "`*/`).", 'echo "${s##*/}"'),
        ("param:%", "Print the directory part of `$s` with parameter expansion (strip the shortest suffix `/*`).",
         'echo "${s%/*}"'),
        ("param:%", f"Print `$s` with its `.{ext}` extension swapped for `.{new_ext}`.", f'echo "${{s%.{ext}}}.{new_ext}"'),
        ("param:/", f"Print `$s` with the first `_` replaced by `-`.", 'echo "${s/_/-}"'),
        ("param://", "Print `$s` with every `_` replaced by `-`.", 'echo "${s//_/-}"'),
        ("param:^^", "Print `$s` in upper case (parameter expansion, no `tr`).", 'echo "${s^^}"'),
        ("param:,,", "`$s` is set to an upper-case version of the path. Print it in lower case (parameter expansion).",
         'echo "${s,,}"', {"setup": f"s={q(s.upper())}"}),
    ]
    return catalog(rng, items, setup)


@template(M, "lite")
def variables_and_quotes(rng):
    name = rng.choice(NAMES).capitalize()
    proj = rng.choice(WORDS)
    f = rng.choice(["data.csv", "train.csv", "labels.csv"])
    setup = files({f: "".join(f"{i},{rng.choice(WORDS)}\n" for i in range(rng.randint(5, 20))),
                   "a.csv": "x\n", "b.csv": "y\n"}) + "\n#: date 2026-0" + f"{rng.randint(1, 9)}-{rng.randint(10, 28)}"
    items = [
        ("$name", f"Store `{name}` in a variable `name` (no spaces around `=`), then print it with `echo \"$name\"`.",
         f'name="{name}"\necho "$name"', {"require": ["$name", "echo"]}),
        ("${", f"Set `name={name}` and print `{name}_v2` by putting braces around the variable name.",
         f'name="{name}"\necho "${{name}}_v2"', {"require": ["${"]}),
        ("echo", f"Set `name={name}`, then print the literal text `$name` (single quotes stop expansion).",
         f"name=\"{name}\"\necho '$name'", {"require": ["echo"]}),
        ("readonly", f"Make `PROJECT={proj}` read-only, try to change it to `other`, then print its value.",
         f"readonly PROJECT={proj}\nPROJECT=other\necho \"$PROJECT\"", {"answer": ["<stdout>", "<stderr>"]}),
        ("unset", f"Set `name={name}`, remove the variable with `unset`, then print `[$name]`.",
         f"name={name}\nunset name\necho \"[$name]\""),
        ("export", f"Export `WANDB_PROJECT={proj}` so child processes see it, then print it from a child shell "
                   f"(`bash -c 'echo $WANDB_PROJECT'`).", f"export WANDB_PROJECT={proj}\nbash -c 'echo $WANDB_PROJECT'"),
        ("$(", "Store today's date (`date +%F`) in `today` with command substitution, then print `run_` followed by it.",
         'today=$(date +%F)\necho "run_$today"', {"require": ["$(", "date"]}),
        ("$(", f"Store the number of lines of `{f}` in `n` (use `wc -l < {f}` so you get just the number), then print "
               f"`{f} has N lines`.", f'n=$(wc -l < {f})\necho "{f} has $n lines"', {"require": ["$(", "wc"]}),
        ("$(", "Store the list of `.csv` files (`ls *.csv`) in `files` with command substitution and print it.",
         'files=$(ls *.csv)\necho "$files"', {"require": ["$("]}),
    ]
    return catalog(rng, items, setup)


# ================================================================= easy

@template(M, "easy")
def defaults(rng):
    dflt = rng.choice(["./out", "./results", "/tmp/run", "./logs"])
    v = rng.choice(["OUT_DIR", "RESULTS", "LOG_DIR", "SAVE_TO"])
    given = rng.choice(["/data/out", "./custom"])
    kind = rng.choice(["minus-unset", "minus-set", "assign", "error", "empty"])
    if kind == "minus-unset":
        return P(f"`{v}` is not set. Print its value, or `{dflt}` if it's unset or empty, with `${{VAR:-default}}`.",
                 "", f'echo "${{{v}:-{dflt}}}"', require=["param::-"])
    if kind == "minus-set":
        return P(f"Print `{v}`, or `{dflt}` if it's unset or empty, with `${{VAR:-default}}`. (Here it's set.)",
                 f"{v}={given}", f'echo "${{{v}:-{dflt}}}"', require=["param::-"])
    if kind == "assign":
        return P(f"Give `{v}` the default `{dflt}` only if it's unset (`: \"${{VAR:=default}}\"`), then print it.",
                 "", f': "${{{v}:={dflt}}}"\necho "${v}"', require=["param::="])
    if kind == "error":
        return P(f"Make bash stop with the message `{v} must be set` when `{v}` is unset, using `${{VAR:?message}}` "
                 f"in an `echo`, then (after the error) print `next`.", "",
                 f'echo "${{{v}:?{v} must be set}}"\necho next', answer=["<stdout>", "<stderr>"], require=["param::?"])
    return P(f"`{v}` is set but empty. Print `[{dflt}]` using the default-value expansion (with the colon, so empty "
             f"counts as unset).", f"{v}=", f'echo "[${{{v}:-{dflt}}}]"', require=["param::-"])


@template(M, "easy")
def braces_and_globs(rng):
    n = rng.randint(4, 9)
    a, b = sorted(rng.sample("abcdefghij", 2))
    step = rng.choice([2, 3, 5])
    top = step * rng.randint(3, 6)
    w = rng.randint(2, 3)
    runs = rng.sample(["a", "b", "c", "x", "y", "z", "lr1", "lr2"], 3)
    pyfiles = {f"{d}/{rng.choice(WORDS)}.py": "x\n" for d in ["src", "src/core", "tests", "."] for _ in range(1)}
    csvs = {f"data_{i:02d}.csv": "x\n" for i in rng.sample(range(1, 40), 3)} | {"data_1.csv": "x\n", "data_100.csv": "x\n"}
    setup = files({**pyfiles, **csvs, "alpha.txt": "x\n", "beta.txt": "x\n", "delta.txt": "x\n", "cat.txt": "x\n"})
    items = [
        ("{..}", f"Print the numbers 1 to {n} on one line with brace expansion (no `seq`).", f"echo {{1..{n}}}"),
        ("{..}", f"Print the letters `{a}` to `{b}` on one line with brace expansion.", f"echo {{{a}..{b}}}"),
        ("{..}", f"Print 0 to {top} in steps of {step} with brace expansion.", f"echo {{0..{top}..{step}}}"),
        ("{..}", f"Print the zero-padded numbers `{'0' * (w - 1)}1` to `{n:0{w}d}` with brace expansion.",
         f"echo {{{'0' * (w - 1)}1..{n:0{w}d}}}"),
        ("{,}", f"Create the directories `run_{runs[0]}`, `run_{runs[1]}` and `run_{runs[2]}` with one `mkdir -p`.",
         f"mkdir -p run_{{{','.join(runs)}}}", {"post": "ls -d run_*"}),
        ("glob", "List the `.py` files in the current directory (a glob).", "ls *.py"),
        ("glob", "List the files named `data_` + exactly two characters + `.csv` (use `?`).", "ls data_??.csv"),
        ("glob", "List the files whose names start with `a`, `b` or `c` (a bracket expression).", "ls [abc]*"),
        ("**", "Turn on `globstar` and list every `.py` file in all subdirectories with `**`.",
         "shopt -s globstar\nls **/*.py", {"require": ["shopt -s", "**"]}),
    ]
    return catalog(rng, items, setup)


# ================================================================= medium

@template(M, "medium")
def printf_formatting(rng):
    rows = [(f"{rng.choice(DATASETS)}.csv", rng.randint(10, 99999)) for _ in range(rng.randint(2, 4))]
    pi = rng.choice(["3.14159", "2.71828", "1.41421", "0.57721"])
    width = rng.choice([5, 6, 8])
    kind = rng.choice(["sentence", "pad", "table", "percent"])
    if kind == "sentence":
        f, n = rows[0]
        return P(f"Print `{f} has {n} rows` with `printf` and a format string (`%s` and `%d`).", "",
                 f'printf "%s has %d rows\\n" "{f}" {n}', require=["printf"])
    if kind == "pad":
        return P(f"Print {pi} rounded to 2 decimals, zero-padded to {width} characters (like `0{pi[:4]}`).", "",
                 f'printf "%0{width}.2f\\n" {pi}', require=["printf"])
    if kind == "table":
        args = " ".join(f"{q(f)} {n}" for f, n in rows)
        return P("Print each file name left-aligned in 15 characters followed by its row count right-aligned in 6, "
                 f"one per line, with a single `printf` (the format is reused for each pair): {args}.", "",
                 f'printf "%-15s %6d\\n" {args}', require=["printf"])
    a, b = rng.randint(1, 50), rng.randint(51, 200)
    return P(f"`done={a}` and `total={b}`. Print the percentage with one decimal and a `%` sign, like `12.5%` "
             f"(use `printf` and `%%`; compute with `awk` or `bc`).", f"done={a}\ntotal={b}",
             'printf "%.1f%%\\n" "$(echo "scale=4; 100 * $done / $total" | bc)"', require=["printf"])


@template(M, "medium")
def rename_loop(rng):
    old, new = rng.choice([("pt", "onnx"), ("txt", "md"), ("jpeg", "jpg"), ("yml", "yaml")])
    names = rng.sample(WORDS, rng.randint(3, 5))
    setup = files({f"{n}.{old}": "x\n" for n in names} | {"keep.csv": "x\n"})
    kind = rng.choice(["ext", "prefix", "upper"])
    if kind == "ext":
        return P(f"Rename every `.{old}` file here to `.{new}` (same base name), with a loop and `${{f%.{old}}}`.",
                 setup, f'for f in *.{old}; do mv "$f" "${{f%.{old}}}.{new}"; done', post="ls",
                 require=["for", "param:%"])
    if kind == "prefix":
        return P(f"Add the prefix `v1_` to every `.{old}` file name.", setup,
                 f'for f in *.{old}; do mv "$f" "v1_$f"; done', post="ls", require=["for", "mv"])
    return P(f"Rename every `.{old}` file so its base name is in upper case (`apple.{old}` -> `APPLE.{old}`), using "
             f"`${{var^^}}`.", setup, f'for f in *.{old}; do b=${{f%.{old}}}; mv "$f" "${{b^^}}.{old}"; done', post="ls",
             require=["param:^^"])


# ================================================================= hard

@template(M, "hard")
def path_surgery(rng):
    root, run, name, ext, s = _path(rng)
    full = f"/data/{s}"
    kind = rng.choice(["three", "sibling", "relative"])
    setup = f"ckpt={q(full)}"
    if kind == "three":
        return P(f"From `$ckpt` (`{full}`) print, on three lines, the file name, the directory, and the name without "
                 f"its extension, using only parameter expansion.", setup,
                 'echo "${ckpt##*/}"\necho "${ckpt%/*}"\nb=${ckpt##*/}\necho "${b%.*}"', require=["param:##", "param:%"])
    if kind == "sibling":
        return P("Print the path of a file `config.yaml` in the same directory as `$ckpt` (parameter expansion only).",
                 setup, 'echo "${ckpt%/*}/config.yaml"', require=["param:%"])
    return P("Print the run directory name (the parent directory of the checkpoint, e.g. `run_042`) using only "
             "parameter expansion.", setup, 'd=${ckpt%/*}\necho "${d##*/}"', require=["param:%", "param:##"])


@template(M, "hard")
def spaces_in_names(rng):
    names = [f"{rng.choice(WORDS)} {rng.choice(WORDS)}.tmp" for _ in range(rng.randint(2, 3))] + \
            [f"{rng.choice(WORDS)}.tmp", "keep me.txt", f"{rng.choice(WORDS)}.log"]
    setup = files({n: "x\n" for n in names})
    kind = rng.choice(["delete", "count", "list"])
    if kind == "delete":
        return P("Delete every `.tmp` file here. Some names contain spaces, so quote the variable in your loop.",
                 setup, 'for f in *.tmp; do rm -- "$f"; done', post="ls -1", require=["for"])
    if kind == "count":
        return P("Print each `.tmp` file name followed by `: N bytes` (from `wc -c < FILE`). Names contain spaces.",
                 setup, 'for f in *.tmp; do echo "$f: $(wc -c < "$f") bytes"; done', require=["for", "$("])
    return P("Store all `.tmp` file names in an array (a glob) and print how many there are, then each name in "
             "brackets on its own line.", setup, 'files=(*.tmp)\necho "${#files[@]}"\nprintf "[%s]\\n" "${files[@]}"',
             require=["arr=(", "array[@]"])


# ================================================================= challenge

def _imgs(rng, n):
    names = set()
    while len(names) < n:
        names.add(f"{rng.choice(['img', 'frame', 'scan'])}_{rng.randint(0, 1200)}.{rng.choice(['png', 'jpg'])}")
    return files({f"shots/{x}": "x\n" for x in sorted(names)} | {"shots/README": "notes\n"})


@template(M, "challenge")
def zero_pad(rng):
    setup = _imgs(rng, 5)
    hidden = [_imgs(rng, rng.randint(25, 40)) for _ in range(2)]
    sol = ('cd shots\nfor f in *_*.*; do\n  stem=${f%_*}; rest=${f##*_}; num=${rest%.*}; ext=${rest##*.}\n'
           '  new=$(printf "%s_%05d.%s" "$stem" "$((10#$num))" "$ext")\n  [[ $f != "$new" ]] && mv "$f" "$new"\ndone\ncd ..')
    return challenge("Zero-pad the frames", "Files in `shots/` are named `PREFIX_NUMBER.EXT` (e.g. `img_7.png`). "
                     "Rename them so the number has 5 digits (`img_00007.png`), leaving other files alone. Sorting by "
                     "name will then sort by number.", setup, sol, hidden, "up to 40 files", post="ls -1 shots")


def _bumps(rng, n):
    v = f"{rng.randint(0, 3)}.{rng.randint(0, 12)}.{rng.randint(0, 20)}"
    return files({"VERSION": v + "\n", "bumps.txt": "\n".join(rng.choice(["patch", "patch", "minor", "major"])
                                                           for _ in range(n)) + "\n"})


@template(M, "challenge")
def version_bumps(rng):
    setup = _bumps(rng, 4)
    hidden = [_bumps(rng, rng.randint(20, 40)) for _ in range(2)]
    sol = ('v=$(cat VERSION)\nwhile read -r b; do\n  IFS=. read -r ma mi pa <<< "$v"\n  case $b in\n'
           '    major) v="$((ma + 1)).0.0" ;;\n    minor) v="$ma.$((mi + 1)).0" ;;\n    patch) v="$ma.$mi.$((pa + 1))" ;;\n'
           '  esac\n  echo "$b -> $v"\ndone < bumps.txt')
    return challenge("Semantic version bumps", "`VERSION` holds `MAJOR.MINOR.PATCH`. Apply each bump in `bumps.txt` "
                     "in order (`major` resets minor and patch to 0, `minor` resets patch) and print `BUMP -> VERSION` "
                     "after each one.", setup, sol, hidden, "up to 40 bumps")
