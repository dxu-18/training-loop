"""7.17 Keyboard shortcuts & history.

Keystroke problems use the readline/tmux simulator (bashlite/keys.py): the answer is keys like `C-a "sudo " Enter`.
History problems are ordinary bash, typed into a shell whose history is preloaded with `#: history` lines.
"""

from ._common import DATASETS, MODELS, NAMES, WORDS, K, P, catalog, challenge, files, q, template

M = "7.17"
KEYS_HINT = ("Press the keys in the terminal: `C-a` is Ctrl-A, `M-b` is Option-B (Alt-B), and `C-b d` means Ctrl-B, "
             "then d. Only what you press counts, so if a key goes wrong, press Reset and try again.")


def _cmd(rng):
    return rng.choice([
        f"python train.py --lr {rng.choice(['0.1', '0.01', '3e-4'])} --epochs {rng.randint(5, 50)}",
        f"rsync -avz data/ {rng.choice(['gpu', 'lab'])}:~/data/",
        f"grep -rn {rng.choice(['TODO', 'FIXME', 'import'])} src",
        f"tar -czf {rng.choice(WORDS)}.tar.gz results",
        f"du -sh {rng.choice(['data', 'checkpoints', 'logs'])}",
        f"ssh {rng.choice(['gpu-box', 'trainer'])} nvidia-smi",
    ])


def _hist(rng, n=5, must=()):
    pool = ["ls -la", "cd ~/project", "git status", "nvidia-smi", "du -sh *", "htop", "cat notes.md",
            "pip install -r requirements.txt", "tail -f train.log", "python eval.py --ckpt best.pt"]
    lines = rng.sample(pool, n) + list(must)
    rng.shuffle(lines)
    return lines


def hist_directives(lines):
    return "\n".join(f"#: history {h}" for h in lines)


# ================================================================= lite

@template(M, "lite")
def line_editing(rng):
    c = _cmd(rng)
    words = c.split()
    flag = rng.choice([" --verbose", " --dry-run", " | less", " > out.log"])
    items = [
        ("C-a", "The command line holds the text shown. Jump to the START of the line, add `sudo ` there, and run "
                "it (Ctrl-A).", 'C-a "sudo " Enter', {"setup": f"#: buffer {c}", "require": ["C-a|Home"],
                                                      "forbid": ["Left"]}),
        ("C-e", f"The cursor is at the start of the line. Jump to the END (Ctrl-E) and append `{flag}` (don't run it).",
         f'C-e "{flag}"', {"setup": f"#: buffer {c}\n#: cursor 0", "require": ["C-e|End"], "forbid": ["Right"]}),
        ("C-w", "Delete the last word on the line (Ctrl-W), without running anything.", "C-w",
         {"setup": f"#: buffer {c}", "require": ["C-w"]}),
        ("C-u", "Delete everything before the cursor (the cursor is at the end, so: the whole line) with one key.",
         "C-u", {"setup": f"#: buffer {c}", "require": ["C-u"]}),
        ("C-k", "The cursor is at the start. Delete from the cursor to the end of the line with one key.", "C-k",
         {"setup": f"#: buffer {c}\n#: cursor 0", "require": ["C-k"]}),
        ("M-b", f"Move back {min(2, len(words) - 1)} words (Alt-B), then delete from there to the end of the line "
                f"(Ctrl-K).", " ".join(["M-b"] * min(2, len(words) - 1)) + " C-k",
         {"setup": f"#: buffer {c}", "require": ["M-b", "C-k"]}),
        ("M-f", "The cursor is at the start. Move forward one word (Alt-F) and type ` -v` there.",
         'M-f " -v"', {"setup": f"#: buffer {c}\n#: cursor 0", "require": ["M-f"]}),
        ("C-y", "Cut the last word (Ctrl-W), go to the start of the line (Ctrl-A) and paste it back there (Ctrl-Y) "
                "followed by a space.", 'C-w C-a C-y " "', {"setup": f"#: buffer {c}", "require": ["C-w", "C-y"]}),
        ("C-c", "Abandon this half-typed command without running it (and without deleting it key by key), then run "
                "`pwd`.", 'C-c "pwd" Enter', {"setup": f"#: buffer {c}", "require": ["C-c"]}),
        ("C-l", "Clear the screen with a shortcut, keeping the line, then run it.", "C-l Enter",
         {"setup": f"#: buffer echo {rng.choice(WORDS)}", "require": ["C-l"]}),
    ]
    return _keys_catalog(rng, items)


def _keys_catalog(rng, items):
    from ...core import pick
    feat, prompt, sol, opts = pick(rng, items)
    opts = dict(opts)
    setup = opts.pop("setup", "")
    return K(f"{prompt}\n\n{KEYS_HINT}", setup, sol, **opts)


@template(M, "lite")
def history_keys(rng):
    target = _cmd(rng)
    hist = _hist(rng, 5, [target])
    word = target.split()[0] if rng.random() < 0.5 else target.split()[1].lstrip("-")[:4]
    last = hist[-1]
    files_setup = "mkdir -p ~/data && touch ~/data/train_2026.csv ~/data/test.csv ~/data/notes.md"
    items = [
        ("C-r", f"Search your history backwards for the latest command containing `{word}` (Ctrl-R, type `{word}`) "
                f"and run it with Enter.", f'C-r "{word}" Enter', {"setup": hist_directives(hist), "require": ["C-r"]}),
        ("Up", "Bring back the previous command (Up arrow or Ctrl-P) and run it again.", "Up Enter",
         {"setup": hist_directives(hist), "require": ["Up|C-p"]}),
        ("M-.", f"Type `ls -l ` and insert the LAST argument of the previous command with Alt-. , then run it.",
         '"ls -l " M-. Enter', {"setup": hist_directives(hist) + "\nmkdir -p results ~/project src", "require": ["M-."]}),
        ("Tab", "Type `cat ~/data/tr` and let Tab complete the file name, then run it.", '"cat ~/data/tr" Tab Enter',
         {"setup": files_setup, "require": ["Tab"]}),
        ("Tab", "Type `ls ~/data/t` and press Tab twice to list the possible completions (don't run anything).",
         '"ls ~/data/t" Tab Tab', {"setup": files_setup, "require": ["Tab"]}),
        ("C-d", "Log out of the shell with Ctrl-D on an empty line.", "C-d", {"setup": "", "require": ["C-d"]}),
    ]
    return _keys_catalog(rng, items)


# ================================================================= easy

@template(M, "easy")
def bang_history(rng):
    d = rng.choice(["results", "logs", "runs"])
    f = rng.choice(["notes.md", "README.md", "config.yaml"])
    setup = files({f"{d}/a.txt": "a\n", f: f"# {rng.choice(WORDS)}\n"})
    cmd = f"ls {d}"
    old, new = d, rng.choice([x for x in ["src", "data", "docs"]])
    items = [
        ("!!", f"Your last command was `{cmd}`. Run it again with `!!`.", "!!",
         {"setup": setup + f"\nmkdir -p {new}\n#: history {cmd}"}),
        ("!!", "Your last command was `cat /etc/shadow` and it failed (permission denied). Re-run it with sudo "
               "using `sudo !!`.", "sudo !!", {"setup": "#: history cat /etc/shadow", "require": ["!!", "sudo"]}),
        ("!$", f"Your last command was `mkdir -p {d}/new`. `cd` into that same path with `!$`, then print `pwd`.",
         "cd !$\npwd", {"setup": f"mkdir -p {d}/new\n#: history mkdir -p {d}/new", "require": ["!$"]}),
        ("!str", f"Re-run the most recent command that started with `cat` using `!cat`.", "!cat",
         {"setup": setup + f"\n#: history cat {f}\n#: history ls -la\n#: history echo hi", "require": ["!str"]}),
        ("^^", f"Your last command was `{cmd}`. Re-run it with `{old}` replaced by `{new}` using `^old^new`.",
         f"^{old}^{new}", {"setup": setup + f"\nmkdir -p {new} && touch {new}/b.txt\n#: history {cmd}", "require": ["^^"]}),
        ("history", "Print your last 3 history entries (including this command).", "history | tail -3",
         {"setup": "\n".join(f"#: history {h}" for h in _hist(rng, 5)), "require": ["history"]}),
    ]
    return catalog(rng, items)


@template(M, "easy")
def tab_completion(rng):
    stems = rng.sample(["train", "test", "valid", "transform", "tools"], 3)
    exts = rng.sample(["csv", "py", "json", "md"], 2)
    names = [f"{s}_{rng.randint(1, 9)}.{rng.choice(exts)}" for s in stems]
    setup = files({f"work/{n}": "x\n" for n in names} | {"work/models/m.pt": "w\n"})
    pick = rng.choice(names)
    pre = pick[:2] if sum(n.startswith(pick[:2]) for n in names) == 1 else pick[:pick.index("_") + 1]
    kind = rng.choice(["file", "dir", "command"])
    if kind == "file":
        return K(f"Type `cat work/{pre}` and complete the file name with Tab, then run it.\n\n{KEYS_HINT}", setup,
                 f'"cat work/{pre}" Tab Enter', require=["Tab"])
    if kind == "dir":
        return K(f"Type `ls work/mo`, complete the directory name with Tab, and run it.\n\n{KEYS_HINT}", setup,
                 '"ls work/mo" Tab Enter', require=["Tab"])
    prog = rng.choice(["nvidia-smi", "whoami", "hostname", "uptime"])
    return K(f"Type the first 4 letters of `{prog}`, let Tab complete the command name, and run it.\n\n{KEYS_HINT}",
             "#: gpus 1" if prog == "nvidia-smi" else "", f'"{prog[:4]}" Tab Enter', require=["Tab"])


# ================================================================= medium

@template(M, "medium")
def job_keys(rng):
    t = rng.randint(30, 300)
    kind = rng.choice(["suspend-bg", "suspend-fg-interrupt", "interrupt"])
    if kind == "suspend-bg":
        return K(f"Run `sleep {t}`, suspend it with Ctrl-Z, resume it in the background with `bg`, then list jobs."
                 f"\n\n{KEYS_HINT}", "", f'"sleep {t}" Enter C-z "bg" Enter "jobs" Enter', require=["C-z"])
    if kind == "suspend-fg-interrupt":
        return K(f"Run `sleep {t}`, suspend it (Ctrl-Z), bring it back to the foreground with `fg`, then interrupt it "
                 f"with Ctrl-C and print `$?`.\n\n{KEYS_HINT}", "",
                 f'"sleep {t}" Enter C-z "fg" Enter C-c "echo $?" Enter', require=["C-z", "C-c"])
    return K(f"Start `sleep {t}` and stop it with Ctrl-C, then print its exit status with `echo $?`.\n\n{KEYS_HINT}",
             "", f'"sleep {t}" Enter C-c "echo $?" Enter', require=["C-c"])


@template(M, "medium")
def fix_previous(rng):
    good = rng.choice(MODELS)
    typo = good.replace("-", "", 1) if "-" in good else good[:-1]
    d = rng.choice(DATASETS)
    hist = [f"echo train {typo} on {d}"]
    kind = rng.choice(["edit-up", "quick-sub"])
    if kind == "edit-up":
        return K(f"Your last command was `{hist[0]}`: the model name is wrong (it should be `{good}`). Bring it back "
                 f"with Up, fix it with editing keys (e.g. move back words with Alt-B, delete with Ctrl-W/Alt-D or "
                 f"Backspace), and run it.\n\n{KEYS_HINT}", hist_directives(hist),
                 f'Up M-b M-b M-b C-w "{good} " Enter', require=["Up|C-p|C-r"])
    return P(f"Your last command was `{hist[0]}` with a typo: fix `{typo}` to `{good}` and re-run it with a quick "
             f"substitution (`^old^new`).", hist_directives(hist), f"^{typo}^{good}", require=["^^"])


# ================================================================= hard

@template(M, "hard")
def vi_mode(rng):
    c = _cmd(rng)
    kind = rng.choice(["delete-word", "change-word", "append-end"])
    if kind == "delete-word":
        return K(f"Switch the shell to vi editing (`set -o vi`, Enter). Then type `echo {rng.choice(WORDS)} extra`, "
                 f"press Esc for normal mode, delete the last word with `b` then `dw`, and run it.\n\n{KEYS_HINT}", "",
                 f'"set -o vi" Enter "echo {rng.choice(WORDS)} extra" Esc b d w Enter', require=["Esc"])
    if kind == "change-word":
        w1, w2 = rng.sample(WORDS, 2)
        return K(f"Turn on vi mode (`set -o vi`, Enter), type `echo {w1}`, press Esc, then change the word under the "
                 f"cursor's start: `b`, `cw`, type `{w2}`, Enter.\n\n{KEYS_HINT}", "",
                 f'"set -o vi" Enter "echo {w1}" Esc b c w "{w2}" Enter', require=["Esc"])
    return K(f"In vi mode (`set -o vi`, Enter), type `{c}`, press Esc, jump to the start with `0`, then append "
             f"` --help` at the END with `A` and run it.\n\n{KEYS_HINT}", "",
             f'"set -o vi" Enter "{c}" Esc 0 A " --help" Enter', require=["Esc"])


@template(M, "hard")
def tmux_keys(rng):
    name = rng.choice(["train", "work", "exp"])
    kind = rng.choice(["detach", "windows", "split", "rename"])
    head = f"You're attached to the tmux session `{name}`; the prefix is Ctrl-B (`C-b`). "
    if kind == "detach":
        return K(head + f"Run `echo started` in it, detach (prefix, `d`), then list sessions with `tmux ls`."
                 f"\n\n{KEYS_HINT}", f"#: tmux {name}", '"echo started" Enter C-b d "tmux ls" Enter', tmux=True,
                 require=["C-b"])
    if kind == "windows":
        n = rng.randint(2, 3)
        return K(head + f"Create {n} new windows (prefix, `c`), then go back to window 0 (prefix, `0`).\n\n{KEYS_HINT}",
                 f"#: tmux {name}", " ".join(["C-b c"] * n) + " C-b 0", tmux=True, require=["C-b"])
    if kind == "split":
        return K(head + "Split the window left/right (prefix, `%`), then split the new pane top/bottom (prefix, `\"`), "
                 f"then zoom that pane (prefix, `z`).\n\n{KEYS_HINT}", f"#: tmux {name}", 'C-b % C-b " C-b z', tmux=True,
                 require=["C-b"])
    new = rng.choice(["logs", "gpu", "editor"])
    return K(head + f"Rename the current window to `{new}` (prefix, `,`, clear the old name with Ctrl-U, type the new "
             f"one, Enter).\n\n{KEYS_HINT}", f"#: tmux {name}", f'C-b , C-u "{new}" Enter', tmux=True, require=["C-b"])


# ================================================================= challenge

def _hist_world(rng, n):
    """History ending with several python runs; files the commands print."""
    lines = []
    for i in range(n):
        lr = rng.choice(["0.1", "0.01", "0.001"])
        if rng.random() < 0.5:
            lines.append(f"echo python train.py --lr {lr} --seed {i}")
        else:
            lines.append(rng.choice(["ls", "pwd", "echo hi", "date +%F", "whoami"]))
    lines.append(f"echo python train.py --lr {rng.choice(['0.1', '0.01'])} --seed {n}")
    lines.append(rng.choice(["pwd", "whoami", "date +%F"]))
    return "#: date 2026-10-20\n" + hist_directives(lines)


@template(M, "challenge")
def history_surgery(rng):
    setup = _hist_world(rng, 4)
    hidden = [_hist_world(rng, rng.randint(10, 30)) for _ in range(2)]
    sol = "!echo:s/train.py/eval.py/"
    return challenge("History surgery", "Your history contains several `echo python train.py --lr LR --seed S` runs "
                     "(echoed, so nothing really trains). With ONE history expansion (no retyping the command), re-run "
                     "the most recent command that started with `echo`, with `train.py` replaced by `eval.py`. (Hint: "
                     "`!string` plus a `:s/old/new/` modifier.)", setup, sol, hidden,
                     "the history has up to 32 entries", require=["!str"])


@template(M, "challenge")
def word_designators(rng):
    setup = _hist_world(rng, 3)
    hidden = [_hist_world(rng, rng.randint(8, 25)) for _ in range(2)]
    sol = "echo !echo:3-4 !echo:$"
    return challenge("Pick the words", "With history expansion only, print (with `echo`) the 3rd and 4th words and the "
                     "last word of the most recent command that started with `echo` (word 0 is `echo` itself), e.g. "
                     "`echo !echo:3-4 ...`.", setup, sol, hidden, "the history has up to 27 entries",
                     require=["!str"])
