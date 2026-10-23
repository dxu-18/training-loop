"""7.01 Shell basics & getting help."""

from ._common import NAMES, WORDS, P, catalog, challenge, files, q, template

M = "7.01"

KINDS = {  # name -> what `type -t` says on the simulated Ubuntu box
    "cd": "builtin", "echo": "builtin", "pwd": "builtin", "read": "builtin", "export": "builtin", "printf": "builtin",
    "source": "builtin", "alias": "builtin", "type": "builtin", "history": "builtin", "test": "builtin",
    "if": "keyword", "for": "keyword", "while": "keyword", "case": "keyword", "then": "keyword", "[[": "keyword",
    "ll": "alias", "la": "alias", "l": "alias",
    "ls": "file", "grep": "file", "python3": "file", "tar": "file", "ssh": "file", "curl": "file", "awk": "file",
    "sed": "file", "find": "file", "git": "", "nvim": "", "conda": "",
}
HELP_LINES = [  # (command, phrase in its --help output, what the option does)
    ("ls", "sort by file size", "sorts by file size"),
    ("ls", "sort by time", "sorts by modification time"),
    ("ls", "do not list implied", "shows hidden files but not `.` and `..`"),
    ("ls", "with -l and -s, print sizes like", "prints human-readable sizes"),
    ("grep", "ignore case distinctions in patterns", "ignores case"),
    ("grep", "select non-matching lines", "inverts the match"),
    ("grep", "print line number", "prints line numbers"),
    ("sort", "reverse the result", "reverses the order"),
    ("sort", "compare according to string numerical value", "sorts numerically"),
    ("head", "print the first NUM lines", "sets how many lines to print"),
    ("wc", "print the newline counts", "counts lines"),
    ("cp", "explain what is being done", "explains what is being done"),
    ("rm", "remove directories and their contents recursively", "removes directories recursively"),
    ("mkdir", "no error if existing", "creates parent directories as needed"),
    ("tail", "output appended data as the file grows", "follows a growing file"),
]


# ================================================================= lite

@template(M, "lite")
def basics(rng):
    name = rng.choice(NAMES).capitalize()
    word = rng.choice(WORDS)
    builtin = rng.choice([k for k, v in KINDS.items() if v == "builtin" and k not in ("type",)])
    kw = rng.choice([k for k, v in KINDS.items() if v == "keyword" and k != "[["])
    alias = rng.choice(["ll", "la", "l"])
    prog = rng.choice(["ls", "grep", "python3", "tar", "ssh", "curl", "sed", "find"])
    code = rng.randint(2, 9)
    items = [
        ("echo", f"Print the line `Hello, {name}. Welcome to the {word} lab`.", f'echo "Hello, {name}. Welcome to the {word} lab"'),
        ("echo", "Print the path of your login shell (the `SHELL` environment variable).", 'echo "$SHELL"',
         {"require": ["echo", "$name"]}),
        ("bash --version", "Print only the first line of bash's `--version` output.", "bash --version | head -1",
         {"require": ["bash --version", "head"]}),
        ("type", f"Ask bash what kind of command `{builtin}` is.", f"type {builtin}"),
        ("type", f"Ask bash what kind of command `{kw}` is.", f"type {kw}"),
        ("type", f"`{alias}` is defined in `~/.bashrc`. Ask bash what it is.", f"type {alias}"),
        ("type", f"Ask bash what `{prog}` is and where it lives.", f"type {prog}"),
        ("which", f"Print the full path of the `{prog}` program with `which`.", f"which {prog}"),
        ("command -v", f"Print how `{prog}` would be found, POSIX style, with `command -v`.", f"command -v {prog}"),
        ("whatis", f"Print the one-line description of `{prog}` from the manual.", f"whatis {prog}"),
        ("help", f"`{builtin}` is a builtin, so it has no man page. Print the first line of its built-in help "
                 f"(the usage line).", f"help {builtin} | head -1", {"require": ["help", "head"]}),
        ("clear", "Clear the terminal screen.", "clear"),
        ("exit", f"Leave the shell with exit status {code}.", f"exit {code}", {"answer": ["<status>"]}),
    ]
    return catalog(rng, items)


@template(M, "lite")
def find_a_flag(rng):
    cmd, phrase, does = rng.choice(HELP_LINES)
    style = rng.choice(["help", "man"])
    if style == "help":
        return P(f"Which option of `{cmd}` {does}? Print the line of `{cmd} --help` that contains "
                 f"`{phrase}` (pipe the help into `grep`).", "", f'{cmd} --help | grep "{phrase}"',
                 require=[f"{cmd} --help", "grep"])
    return P(f"Which option of `{cmd}` {does}? Search its manual page: print the lines of `man {cmd}` that "
             f"contain `{phrase}`.", "", f'man {cmd} | grep "{phrase}"', require=["man", "grep"])


# ================================================================= easy

@template(M, "easy")
def lookups(rng):
    names = rng.sample([k for k in KINDS if KINDS[k] and k != "[["], 4)
    prog = rng.choice(["python3", "grep", "ls", "awk"])
    word = rng.choice(["compress", "copy", "directory", "archive", "pattern"])
    hist = rng.sample(["ls -la", "cd data", "python3 train.py --lr 0.01", "nvidia-smi", "git status", "du -sh *",
                       "tail -f train.log", "pip install torch", "htop", "cat notes.md", "mkdir -p runs/exp1"], 6)
    k = rng.randint(2, 4)
    hset = "\n".join(f"#: history {h}" for h in hist)
    tool = rng.choice(["tar", "grep", "find", "curl", "ssh", "rsync"])
    items = [
        ("type -t", f"Print one word per command saying what kind each of `{' '.join(names)}` is "
                    f"(builtin, keyword, alias or file), in that order, using a single `type` command.",
         f"type -t {' '.join(names)}", {"require": ["type -t"]}),
        ("which -a", f"`{prog}` may be on the PATH more than once. Print every match, in PATH order.",
         f"which -a {prog}"),
        ("type -a", f"`echo` is both a builtin and a program. Print every place bash finds it.", "type -a echo"),
        ("apropos", f"How many manual page descriptions mention `{word}`? Print just the count.",
         f"apropos {word} | wc -l", {"require": ["apropos", "wc"]}),
        ("history", f"Show the last {k} lines of your shell history (the command you type counts too).",
         f"history | tail -{k}", {"setup": hset, "require": ["history", "tail"]}),
        ("history", f"Which of your earlier commands mention `{hist[2].split()[0]}`? Search your history with grep.",
         f"history | grep {q(hist[2].split()[0])}", {"setup": hset, "require": ["history", "grep"]}),
        ("tldr", f"Print the first 6 lines of the `tldr` page for `{tool}`.", f"tldr {tool} | head -6",
         {"require": ["tldr", "head"]}),
    ]
    return catalog(rng, items)


@template(M, "easy")
def mac_vs_linux(rng):
    old, new = rng.sample(WORDS, 2)
    f = rng.choice(["config.txt", "notes.md", "params.yaml", "README.md"])
    text = "".join(f"{rng.choice(WORDS)} {old} {rng.choice(WORDS)}\n" for _ in range(3))
    setup = files({f: text})
    days = rng.randint(1, 9)
    kind = rng.choice(["sed-mac", "sed-linux", "date-mac", "date-linux", "md5-mac"])
    if kind == "sed-mac":
        return P(f"You're on a **Mac** (BSD tools). Replace the first `{old}` on each line of `{f}` with `{new}`, "
                 f"editing the file in place.", "#: os macos\n" + setup, f"sed -i '' 's/{old}/{new}/' {f}",
                 post=f"cat {f}", require=["sed -i"])
    if kind == "sed-linux":
        return P(f"You're on **Linux** (GNU tools). Replace the first `{old}` on each line of `{f}` with `{new}`, "
                 f"editing the file in place.", setup, f"sed -i 's/{old}/{new}/' {f}", post=f"cat {f}", require=["sed -i"])
    if kind == "date-mac":
        return P(f"You're on a **Mac**. Print the date {days} days from now as `YYYY-MM-DD` with BSD `date -v`.",
                 "#: os macos\n#: date 2026-03-14", f"date -v+{days}d +%F", require=["date -v"])
    if kind == "date-linux":
        return P(f"You're on **Linux**. Print the date {days} days from now as `YYYY-MM-DD` with GNU `date -d`.",
                 "#: date 2026-03-14", f"date -d '+{days} days' +%F", require=["date -d"])
    return P(f"You're on a **Mac**, which has `md5` instead of `md5sum`. Print just the MD5 hash of `{f}` "
             f"(`md5 -q`).", "#: os macos\n" + setup, f"md5 -q {f}", require=["md5"])


# ================================================================= medium

@template(M, "medium")
def classify_commands(rng):
    names = rng.sample([k for k in KINDS if k != "[["], rng.randint(5, 7))
    setup = files({"commands.txt": "\n".join(names) + "\n"})
    style = rng.choice(["loop", "missing"])
    if style == "loop":
        return P("`commands.txt` lists command names, one per line. For each one print `NAME: KIND`, where KIND is "
                 "what `type -t` says (empty when bash doesn't know the name).", setup,
                 'while read -r c; do echo "$c: $(type -t "$c")"; done < commands.txt',
                 require=["type -t", "while|for"])
    return P("`commands.txt` lists command names, one per line. Print the names bash can't find at all "
             "(`type` fails for them), one per line.", setup,
             'while read -r c; do type "$c" > /dev/null 2>&1 || echo "$c"; done < commands.txt',
             require=["type|command -v"])


@template(M, "medium")
def manual_search(rng):
    word = rng.choice(["copy", "compress", "directory", "archive", "pattern", "files", "list", "remove"])
    style = rng.choice(["names", "count", "sections"])
    if style == "names":
        return P(f"Print just the command names (first word) of every manual page whose description matches "
                 f"`{word}`, one per line.", "", f"apropos {word} | cut -d' ' -f1",
                 require=["apropos|man -k", "cut|awk"])
    if style == "count":
        cmd = rng.choice(["ls", "grep", "sort", "tar", "find"])
        return P(f"How many lines of `man {cmd}` mention `{word}`? Print just the number.", "",
                 f"man {cmd} | grep -c {word}", require=["man", "grep"])
    return P(f"Print the descriptions (`whatis`) of `ls`, `cp`, `mv` and `rm`, sorted alphabetically.", "",
             "whatis ls cp mv rm | sort", require=["whatis", "sort"])


# ================================================================= hard

@template(M, "hard")
def describe_function(rng):
    names = rng.sample(["cd", "echo", "ls", "grep", "if", "ll", "for", "python3", "tar", "pwd", "la", "case"], 4)
    fn = rng.choice(["kind", "what", "describe"])
    return P(f"Write a function `{fn}` that prints, for each argument, `NAME is a KIND` (KIND from `type -t`), or "
             f"`NAME is unknown` when bash can't find it. Then call it as `{fn} {' '.join(names)} nosuchcmd`.", "",
             f'{fn}() {{\n  for c in "$@"; do\n    t=$(type -t "$c")\n    echo "$c is ${{t:-unknown}}"\n  done\n}}\n'
             f'{fn} {" ".join(names)} nosuchcmd', require=["function", "type -t"])


@template(M, "hard")
def builtin_usage(rng):
    bs = rng.sample(["cd", "echo", "pwd", "read", "export", "printf", "source", "alias", "history", "test", "type",
                     "kill", "jobs", "wait", "shift", "trap", "umask"], rng.randint(3, 5))
    setup = files({"builtins.txt": "\n".join(bs) + "\n"})
    return P("`builtins.txt` lists bash builtins. For each one print its usage line (the first line of `help NAME`) "
             "and finally the total number of builtins listed, as `N builtins`.", setup,
             'while read -r b; do help "$b" | head -1; done < builtins.txt\necho "$(wc -l < builtins.txt) builtins"',
             require=["help", "while|for"])


# ================================================================= challenge

def _kind_counts(rng, n):
    pool = [k for k in KINDS if k not in ("[[",)] + ["zsh", "fish", "kubectl", "terraform"]
    return [rng.choice(pool) for _ in range(n)]


@template(M, "challenge")
def kind_histogram(rng):
    names = _kind_counts(rng, 8)
    hidden = [files({"names.txt": "\n".join(_kind_counts(rng, rng.randint(30, 60))) + "\n"}) for _ in range(2)]
    setup = files({"names.txt": "\n".join(names) + "\n"})
    sol = ('while read -r c; do t=$(type -t "$c"); echo "${t:-missing}"; done < names.txt | sort | uniq -c | '
           "sort -k1,1nr -k2,2")
    return challenge("Command census", "`names.txt` lists command names (with repeats). Classify each with "
                     "`type -t` (`missing` when bash doesn't know it) and print how many names fall in each class, "
                     "like `uniq -c` does, most frequent first; ties alphabetically by class.", setup, sol, hidden,
                     "up to 60 names", require=["type -t"])


@template(M, "challenge")
def path_shadowing(rng):
    def world(rng, n_tools):
        tools = rng.sample(["python3", "grep", "ls", "tar", "jq", "sort", "awk", "curl", "fmt", "train", "eval-model",
                            "plot", "sync-data"], n_tools)
        dirs = ["/home/ada/bin", "/home/ada/.local/bin", "/home/ada/tools/bin"]
        spec = {}
        for t in tools:
            for d in rng.sample(dirs, rng.randint(0, 2)):
                spec[f"{d}/{t}"] = "#!/bin/sh\necho stub\n"
        order = rng.sample(dirs, 3)
        lines = [files(spec), "chmod +x " + " ".join(sorted(spec)) if spec else "",
                 f'export PATH="{":".join(order)}:$PATH"', files({"tools.txt": "\n".join(tools) + "\n"})]
        return "\n".join(x for x in lines if x)
    setup = world(rng, 5)
    hidden = [world(rng, rng.randint(9, 13)) for _ in range(2)]
    sol = ('while read -r t; do\n  n=$(type -ap "$t" | wc -l)\n  echo "$t $(command -v "$t" || echo none) $n"\n'
           "done < tools.txt")
    return challenge("Who wins the PATH?", "Some tools exist in several directories of `$PATH` (the setup puts "
                     "three personal bin directories in front). For each tool in `tools.txt` print `TOOL WINNER COUNT`: "
                     "the path bash would run (`none` if not found) and how many executables of that name are on the "
                     "PATH in total.", setup, sol, hidden, "up to 13 tools", require=["command -v|which|type"])
