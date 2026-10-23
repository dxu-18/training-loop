"""7.07 Pipes & redirection."""

from ._common import DATASETS, MODELS, NAMES, WORDS, P, catalog, challenge, files, q, template

M = "7.07"


def _noisy(rng, name="job.sh", marker=True):
    """A script that writes some lines to stdout and some to stderr (and leaves a marker file)."""
    lines = []
    for i in range(rng.randint(3, 6)):
        if rng.random() < 0.6:
            lines.append(f'echo "step {i}: {rng.choice(WORDS)} ok"')
        else:
            lines.append(f'echo "warning {i}: {rng.choice(WORDS)} slow" >&2')
    lines.insert(0, 'echo "starting"')
    lines.append('echo "error: disk almost full" >&2')
    if marker:
        lines.append("date +%s > ran.marker")
    body = "#!/bin/bash\n" + "\n".join(lines) + "\n"
    return files({name: body}) + f"\nchmod +x {name}"


def _train_py(rng):
    epochs = rng.randint(2, 4)
    body = ("import sys\n"
            f"for e in range({epochs}):\n"
            f"    print(f'epoch {{e}} loss={{0.9 / (e + 1):.3f}}')\n"
            "    print(f'grad warning at epoch {e}', file=sys.stderr)\n"
            "print('done')\n")
    return files({"train.py": body})


# ================================================================= lite

@template(M, "lite")
def redirects(rng):
    setup = _noisy(rng) + "\n" + files({"names.txt": "\n".join(rng.sample(NAMES, 6)) + "\n",
                                        "old.txt": "previous line\n"})
    out = rng.choice(["out.txt", "result.txt", "output.log"])
    items = [
        (">", f"Run `./job.sh` and save its standard output in `{out}` (overwrite).", f"./job.sh > {out}",
         {"post": f"cat {out}"}),
        (">>", "Run `./job.sh` and append its standard output to `old.txt`.", "./job.sh >> old.txt",
         {"post": "cat old.txt"}),
        ("2>", "Run `./job.sh` and send only its error output to `err.txt` (stdout stays on the screen).",
         "./job.sh 2> err.txt", {"answer": ["<stdout>", "<post>"], "post": "cat err.txt"}),
        ("2>&1", f"Run `./job.sh` with both stdout and stderr going into `{out}` (redirect stdout first, then `2>&1`).",
         f"./job.sh > {out} 2>&1", {"post": f"cat {out}"}),
        ("&>", f"Same as `> file 2>&1`, with bash's shorthand: run `./job.sh` with all output into `{out}`.",
         f"./job.sh &> {out}", {"post": f"cat {out}"}),
        (">", "Run `./job.sh` discarding all its output (stdout and stderr), then print `finished`.",
         "./job.sh > /dev/null 2>&1\necho finished", {"answer": ["<stdout>", "<stderr>", "<post>"], "post": "ls ran.marker",
                                                      "require": [">", "2>&1|&>"]}),
        ("<", "Sort `names.txt` by feeding it to `sort` on standard input (`<`).", "sort < names.txt"),
        ("<", "Print the number of lines in `names.txt` without the file name (read it from stdin).", "wc -l < names.txt"),
        ("|", "Print how many lines of `./job.sh`'s standard output contain `ok`, with a three-command pipeline "
              "(`./job.sh | grep ... | wc -l`).", "./job.sh 2>/dev/null | grep ok | wc -l", {"require": ["|", "grep", "wc"]}),
    ]
    return catalog(rng, items, setup)


@template(M, "lite")
def tee_heredoc(rng):
    batch = rng.choice([16, 32, 64, 128])
    lr = rng.choice(["0.001", "3e-4", "1e-4"])
    word = rng.choice(WORDS)
    text_lines = [f"{rng.choice(WORDS)} {rng.choice(WORDS)}" for _ in range(4)] + [f"{word} found here"]
    setup = _train_py(rng) + "\n" + files({"a.txt": "\n".join(rng.sample(NAMES, 5)) + "\n",
                                           "b.txt": "\n".join(rng.sample(NAMES, 5)) + "\n",
                                           "train.log": "old run\n"})
    items = [
        ("tee", "Run `python3 train.py` so that stdout AND stderr appear on the screen and are saved in `train.log` "
                "(overwrite it).", "python3 train.py 2>&1 | tee train.log", {"post": "cat train.log",
                                                                             "require": ["2>&1", "tee"]}),
        ("tee -a", "Run `python3 train.py`, showing its stdout and appending it to `train.log`.",
         "python3 train.py | tee -a train.log", {"post": "cat train.log", "answer": ["<stdout>", "<post>"]}),
        ("<<", f"Write `config.yaml` with a here-document: `lr: {lr}` and `batch_size: ` followed by `$BATCH`, "
               f"defaulting to {batch} when BATCH isn't set (`${{BATCH:-{batch}}}`).",
         f"cat > config.yaml <<EOF\nlr: {lr}\nbatch_size: ${{BATCH:-{batch}}}\nEOF", {"post": "cat config.yaml",
                                                                                   "require": ["<<", "param::-"]}),
        ("<<", "Write `script.sh` containing the line `echo \"home is $HOME\"` literally (with a quoted here-document "
               "delimiter, so `$HOME` is not expanded).", "cat > script.sh <<'EOF'\necho \"home is $HOME\"\nEOF",
         {"post": "cat script.sh"}),
        ("<<<", f"The variable `text` holds several lines. Print the ones containing `{word}`, feeding the variable to "
                f"grep with a here-string.", f'grep {word} <<< "$text"',
         {"setup": "text=" + q("\n".join(text_lines)), "require": ["<<<"]}),
        ("<(", "Compare `a.txt` and `b.txt` sorted, without creating temporary files (`diff` + process substitution).",
         "diff <(sort a.txt) <(sort b.txt)", {"require": ["<(", "diff"]}),
    ]
    return catalog(rng, items, setup)


# ================================================================= easy

@template(M, "easy")
def separate_streams(rng):
    setup = _noisy(rng)
    kind = rng.choice(["count-errors", "errors-only", "both-files", "swap"])
    if kind == "count-errors":
        return P("How many lines does `./job.sh` write to standard ERROR? Print just the number (throw stdout away "
                 "and pipe only stderr).", setup, "./job.sh 2>&1 >/dev/null | wc -l", require=["2>&1", "|"])
    if kind == "errors-only":
        return P("Show only the error output of `./job.sh` (hide its normal output).", setup, "./job.sh > /dev/null",
                 answer=["<stdout>", "<stderr>"], require=[">"])
    if kind == "both-files":
        return P("Run `./job.sh` once, saving stdout in `out.log` and stderr in `err.log`; then print the line counts "
                 "of both files (`wc -l out.log err.log`).", setup, "./job.sh > out.log 2> err.log\nwc -l out.log err.log",
                 require=[">", "2>"])
    return P("Make `./job.sh`'s error lines go through a pipe into `grep -c warning` while its normal output "
             "stays on the screen: print the normal output, then the number of warnings.", setup,
             "./job.sh 2>&1 >/dev/tty | grep -c warning", require=["2>&1", "|"])


@template(M, "easy")
def pipe_status(rng):
    f = rng.choice(["missing.txt", "nope.csv", "gone.log"])
    kind = rng.choice(["pipefail", "pipestatus", "no-pipefail"])
    if kind == "pipefail":
        return P(f"`cat {f} | sort` exits 0 even though `cat` fails. Turn on the option that makes a pipeline fail when "
                 f"ANY stage fails, run the pipeline, and print its exit status with `echo $?`.", "",
                 f"set -o pipefail\ncat {f} | sort\necho $?", require=["set -o", "$?"])
    if kind == "pipestatus":
        return P(f"Run `cat {f} | sort | wc -l`, then print the exit status of every stage (`PIPESTATUS`).", "",
                 f'cat {f} | sort | wc -l\necho "${{PIPESTATUS[@]}}"', require=["array[@]"])
    return P(f"Print the exit status of the pipeline `cat {f} | sort` (with default settings), then of `cat {f}` "
             f"alone.", "", f"cat {f} | sort\necho $?\ncat {f}\necho $?", require=["$?"])


# ================================================================= medium

@template(M, "medium")
def generate_configs(rng):
    models = rng.sample(MODELS, rng.randint(2, 4))
    ds = rng.choice(DATASETS)
    lr = rng.choice(["1e-3", "3e-4"])
    sol = (f'for m in {" ".join(models)}; do\n  cat > "configs/$m.yaml" <<EOF\nmodel: $m\ndataset: {ds}\nlr: {lr}\nEOF\ndone')
    return P(f"Create `configs/` and, for each model in `{' '.join(models)}`, write `configs/MODEL.yaml` with three "
             f"lines: `model: MODEL`, `dataset: {ds}` and `lr: {lr}` (a loop and a here-document).",
             "", "mkdir -p configs\n" + sol, post="head configs/*.yaml", require=["<<", "for"])


@template(M, "medium")
def log_wrapper(rng):
    setup = _noisy(rng, marker=False) + "\n#: date 2026-10-20 14:05:09"
    kind = rng.choice(["timestamped", "tee-both", "append-twice"])
    if kind == "timestamped":
        return P("Run `./job.sh` and prefix every line it prints (stdout and stderr together) with the current time "
                 "in brackets, like `[14:05:09] starting`, using `date +%T`.", setup,
                 './job.sh 2>&1 | while IFS= read -r line; do echo "[$(date +%T)] $line"; done',
                 require=["2>&1", "|", "while"])
    if kind == "tee-both":
        return P("Run `./job.sh` so that everything it prints (stdout and stderr) shows on the screen AND is saved in "
                 "`run.log`; then print the number of lines saved.", setup,
                 "./job.sh 2>&1 | tee run.log\nwc -l < run.log", require=["tee", "2>&1"])
    return P("Run `./job.sh` twice, appending all its output (stdout and stderr) to `all.log` each time, then print "
             "`all.log`.", setup, "./job.sh >> all.log 2>&1\n./job.sh >> all.log 2>&1\ncat all.log",
             require=[">>", "2>&1"])


# ================================================================= hard

@template(M, "hard")
def compare_outputs(rng):
    a = rng.sample(NAMES, 7)
    b = rng.sample(NAMES, 7)
    setup = files({"server_a.txt": "".join(f"{n} {rng.randint(1, 9)}\n" for n in a),
                   "server_b.txt": "".join(f"{n} {rng.randint(1, 9)}\n" for n in b)})
    kind = rng.choice(["users-diff", "common", "paste"])
    if kind == "users-diff":
        return P("Compare the user names (first column) of `server_a.txt` and `server_b.txt`, sorted, with `diff` "
                 "and process substitution (no temp files).", setup,
                 "diff <(cut -d' ' -f1 server_a.txt | sort) <(cut -d' ' -f1 server_b.txt | sort)", require=["<(", "diff"])
    if kind == "common":
        return P("Print the users (first column) present on both servers, sorted (`comm` + process substitution).",
                 setup, "comm -12 <(cut -d' ' -f1 server_a.txt | sort) <(cut -d' ' -f1 server_b.txt | sort)",
                 require=["<(", "comm"])
    return P("Print the sorted user names of both servers side by side, tab-separated (`paste` + process "
             "substitution).", setup,
             "paste <(cut -d' ' -f1 server_a.txt | sort) <(cut -d' ' -f1 server_b.txt | sort)", require=["<(", "paste"])


@template(M, "hard")
def here_string_parsing(rng):
    run = f"{rng.choice(MODELS)}-{rng.randint(1, 99)}"
    fields = {"model": run.rsplit("-", 1)[0], "acc": f"{rng.uniform(0.6, 0.99):.3f}", "epochs": rng.randint(5, 90)}
    line = " ".join(f"{k}={v}" for k, v in fields.items())
    key = rng.choice(list(fields))
    kind = rng.choice(["read", "grep"])
    setup = f"summary={q(line)}"
    if kind == "read":
        return P(f"`$summary` holds `KEY=VALUE` pairs separated by spaces. Read them into an array with "
                 f"`read -a` and a here-string, then print one pair per line.", setup,
                 'read -ra pairs <<< "$summary"\nprintf "%s\\n" "${pairs[@]}"', require=["<<<", "read"])
    return P(f"Print just the value of `{key}` from `$summary` (a here-string into `grep -oP` or `sed`).", setup,
             f'grep -oP "{key}=\\K\\S+" <<< "$summary"', require=["<<<"])


# ================================================================= challenge

def _mixed_script(rng, n):
    lines = ["#!/bin/bash"]
    for i in range(n):
        lvl = rng.choice(["INFO", "INFO", "WARN", "ERROR"])
        target = ">&2" if lvl != "INFO" else ""
        lines.append(f'echo "{lvl} {rng.choice(WORDS)} {i}" {target}'.rstrip())
    return files({"service.sh": "\n".join(lines) + "\n"}) + "\nchmod +x service.sh"


@template(M, "challenge")
def stream_split(rng):
    setup = _mixed_script(rng, 6)
    hidden = [_mixed_script(rng, rng.randint(40, 80)) for _ in range(2)]
    sol = ("./service.sh > out.log 2> err.log\necho \"stdout $(wc -l < out.log)\"\necho \"stderr $(wc -l < err.log)\"\n"
           "grep -c '^ERROR' err.log\ngrep '^ERROR' err.log | tail -1")
    return challenge("Split the streams", "`./service.sh` writes INFO lines to stdout and WARN/ERROR lines to stderr. "
                     "Run it once, saving stdout in `out.log` and stderr in `err.log`, then print `stdout N`, "
                     "`stderr N` (line counts), the number of ERROR lines, and the last ERROR line.", setup, sol,
                     hidden, "up to 80 lines", post="cat out.log err.log | sort | md5sum",
                     answer=["<stdout>", "<post>"])


def _reqs(rng, n):
    pkgs = ["numpy", "pandas", "torch", "scipy", "Pillow", "requests", "tqdm", "PyYAML", "matplotlib", "jinja2",
            "einops", "wandb", "transformers", "datasets", "accelerate", "safetensors"]
    a = rng.sample(pkgs, min(n, len(pkgs)))
    b = rng.sample(pkgs, min(n, len(pkgs)))
    fmt = lambda p: (p.upper() if rng.random() < 0.2 else p) + rng.choice(["", f"=={rng.randint(1, 3)}.{rng.randint(0, 9)}",
                                                                           f">={rng.randint(1, 2)}.0"])
    return files({"prod.txt": "\n".join(fmt(p) for p in a) + "\n", "dev.txt": "\n".join(fmt(p) for p in b) + "\n"})


@template(M, "challenge")
def requirement_drift(rng):
    setup = _reqs(rng, 6)
    hidden = [_reqs(rng, rng.randint(10, 16)) for _ in range(2)]
    names = "sed -E 's/[<>=].*//' {} | tr 'A-Z' 'a-z' | sort -u"
    sol = (f"comm -23 <({names.format('prod.txt')}) <({names.format('dev.txt')}) | sed 's/^/only in prod: /'\n"
           f"comm -13 <({names.format('prod.txt')}) <({names.format('dev.txt')}) | sed 's/^/only in dev: /'")
    return challenge("Requirement drift", "`prod.txt` and `dev.txt` are requirements files (`name`, `name==1.2` or "
                     "`name>=1.0`, any capitalisation). Compare the package NAMES case-insensitively and print "
                     "`only in prod: NAME` lines, then `only in dev: NAME` lines, names in lower case and sorted. "
                     "Don't create temporary files.", setup, sol, hidden, "up to 16 packages per file",
                     require=["<("])
