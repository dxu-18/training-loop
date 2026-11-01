"""9.11 Command-line power moves: :!cmd and :!python %, :w !cmd, :r !cmd, filters (:%!..., :'<,'>!sort), :sort,
:%norm, :g/^$/d, line ranges with :d :m :t, :w !sudo tee %, :set ff=unix, spell checking, :help."""

import json

from ...core import pick
from ._common import DATASETS, MODELS, VARS, V, count_keys, drill, template, train_py

M = "9.11"


# ================================================================= lite

@template(M, "lite")
def shell_commands(rng):
    a, b = rng.randint(3, 9), rng.randint(11, 19)
    script = f"print({a} * {b})\n"
    csv = "epoch,loss\n" + "".join(f"{e},{rng.uniform(0.1, 2):.3f}\n" for e in range(1, rng.randint(5, 9)))
    n = csv.count("\n")
    items = [
        ("Run this script without leaving Vim (`:!python %`: `%` is the current file), read what it prints, press "
         "Enter to come back, and add the printed number as a comment on a new last line: `# result: N`.",
         f":!python %<CR><CR>Go# result: {a * b}<Esc>", {"files": {"calc.py": script}, "require": [":!"]}),
        ("Insert the output of `python calc.py` below the last line (`G` then `:r !python calc.py`).",
         "G:r !python calc.py<CR>", {"files": {"calc.py": script}, "require": [":r !"]}),
        ("Insert the line count of this file, as `wc -l` prints it, at the end of the file (`G`, then "
         "`:r !wc -l %`).", "G:r !wc -l %<CR>", {"files": {"loss.csv": csv}, "require": [":r !"]}),
        ("Run the buffer through Python WITHOUT saving it (`:w !python`), then add its output as a last line "
         "`# N`.", f":w !python<CR><CR>Go# {a * b}<Esc>", {"files": {"calc.py": script}, "require": [":w !"]}),
    ]
    return drill(rng, items, {"calc.py": script})


@template(M, "lite")
def sorting(rng):
    words = rng.sample(MODELS, 6)
    dup = words[:4] + [words[1], words[3]]
    rng.shuffle(dup)
    nums = [str(rng.randint(1, 200)) for _ in range(6)]
    imports = ["import torch", "import os", "import numpy as np", "import sys", "import json"]
    rng.shuffle(imports)
    items = [
        ("Sort the lines alphabetically (`:sort`).", ":sort<CR>", {"files": {"models.txt": "\n".join(words) + "\n"},
                                                                   "require": [":sort"]}),
        ("Sort the lines and drop the duplicates (`:sort u`).", ":sort u<CR>",
         {"files": {"models.txt": "\n".join(dup) + "\n"}, "require": [":sort"]}),
        ("Sort the numbers numerically (`:sort n`; plain `:sort` would put 100 before 20).", ":sort n<CR>",
         {"files": {"steps.txt": "\n".join(nums) + "\n"}, "require": [":sort"]}),
        ("Sort just the import block (lines 1-5) with a range: `:1,5sort`.", ":1,5sort<CR>",
         {"files": {"main.py": "\n".join(imports) + "\n\nprint('ok')\n"}, "require": [":sort"]}),
    ]
    return drill(rng, items, {}, show="target")


# ================================================================= easy

@template(M, "lite")
def help_pages(rng):
    text, f = train_py(rng)
    topic, what = rng.choice([("ciw", "ciw"), (":s", ":s"), ("registers", "the registers"), ("gq", "gq"),
                              (":g", ":global")])
    return V(f"Look {what} up in Vim's own manual: `:help {topic}` opens the help in a window above (you can read "
             f"it, search it, and close it later with `:q`). Leave the help window open and Submit.",
             {"train.py": text}, f":help {topic}<CR>", check=("layout",), require=[":help"], show=None)


@template(M, "easy")
def line_ranges(rng):
    text, f = train_py(rng)
    lines = text.split("\n")
    files = {"train.py": text}
    n = len(lines) - 1
    a = 5
    items = [
        ("Delete lines 5-7 (the three constants) with an Ex range: `:5,7d`.", ":5,7d<CR>", {"require": [":d"]}),
        ("Move lines 5-7 (the constants) to the end of the file: `:5,7m$`.", ":5,7m$<CR>", {"require": [":m"]}),
        ("Copy lines 1-3 (the imports) to the end of the file: `:1,3t$`.", ":1,3t$<CR>", {"require": [":t"]}),
        ("Copy the cursor's line (`.`) to just below line 1: `:t1`.", ":t1<CR>", {"cursor": (a + 1, 1),
                                                                                  "require": [":t"]}),
        ("Delete from the cursor's line to the end of the file with a range: `:.,$d`.", ":.,$d<CR>",
         {"cursor": (n - 3, 1), "require": [":d"]}),
        ("Move the cursor's line to the top of the file: `:m0`.", ":m0<CR>", {"cursor": (a, 1), "require": [":m"]}),
    ]
    return drill(rng, items, files)


@template(M, "easy")
def norm_and_blank(rng):
    names = rng.sample(VARS, 5)
    text = "".join(f"{n}\n" + ("\n" if i % 2 == 0 else "") for i, n in enumerate(names))
    items = [
        ("Append a comma to every line: `:%norm A,`.", ":%norm A,<CR>", {"require": [":%normal"]}),
        ("Delete all the empty lines: `:g/^$/d`.", ":g/^$/d<CR>", {"require": [":g"]}),
        ("Put `- ` in front of every line with `:%norm I- ` (empty lines too).", ":%norm I- <CR>",
         {"require": [":%normal"]}),
    ]
    return drill(rng, items, {"names.txt": text}, show="target")


# ================================================================= medium

@template(M, "medium")
def filters(rng):
    cfg = {"model": rng.choice(MODELS), "lr": float(rng.choice(("0.001", "0.0003"))), "epochs": rng.randint(5, 50),
           "datasets": rng.sample(DATASETS, 2)}
    minified = json.dumps(cfg, separators=(",", ":")) + "\n"
    words = rng.sample(DATASETS, 6)
    text = "# datasets\n" + "\n".join(words) + "\n# end\n"
    items = [
        ("This JSON is minified onto one line. Pretty-print it in place by filtering the whole buffer through "
         "Python: `:%!python -m json.tool`.", ":%!python -m json.tool<CR>",
         {"files": {"config.json": minified}, "require": [":range!"]}),
        ("Sort the six dataset names (lines 2-7) by filtering them through the shell's `sort`: select them "
         "(`2GV7G`) and type `:!sort` (Vim makes it `:'<,'>!sort`).", "2GV7G:!sort<CR>",
         {"files": {"datasets.txt": text}, "require": [":range!"]}),
        ("Uppercase lines 2-7 by filtering them through `tr a-z A-Z` (`:2,7!tr a-z A-Z`).", ":2,7!tr a-z A-Z<CR>",
         {"files": {"datasets.txt": text}, "require": [":range!"]}),
        ("Number lines 2-7 with the shell: `:2,7!nl -w1 -s'. '` (nl numbers lines).", ":2,7!nl -w1 -s'. '<CR>",
         {"files": {"datasets.txt": text}, "require": [":range!"]}),
    ]
    return drill(rng, items, {}, show="target")


@template(M, "medium")
def root_file(rng):
    key = rng.choice(("max_jobs", "gpu_quota", "timeout"))
    old, new = rng.randint(2, 9), rng.randint(10, 40)
    setup = (f"sudo sh -c \"printf '# cluster limits\\n{key} = {old}\\n' > /home/ada/limits.conf && "
             f"chown root:root /home/ada/limits.conf && chmod 644 /home/ada/limits.conf\"")
    keys = f"2G$ciw{new}<Esc>:w !sudo tee %<CR><CR>"
    return V(f"`limits.conf` belongs to root, and you opened it without sudo (`:w` fails: E45 / permission denied). "
             f"Change `{key} = {old}` to `{key} = {new}` and save it anyway through sudo: `:w !sudo tee %` (the buffer "
             f"goes to `sudo tee`, which writes the file). Vim then notices the file changed on disk: press Enter (or "
             f"`L` to load it).", {}, keys, start="limits.conf", extra_setup=setup, check=("files",),
             check_files=["limits.conf"], require=[":w !sudo tee %"], show=None)


# ================================================================= hard

@template(M, "hard")
def line_endings(rng):
    names = rng.sample(VARS, 4)
    text = "".join(f"{n} = {i}\r\n" for i, n in enumerate(names))
    keys = ":set ff=unix<CR>:w<CR>"
    return V("This file was saved on Windows: its lines end in CRLF (Vim shows `[dos]` in the file message). Convert "
             "it to Unix line endings and save: `:set ff=unix` then `:w`.", {"vars.py": text}, keys,
             check=("files",), require=[":set fileformat"], show=None)


@template(M, "hard")
def spelling(rng):
    from ...bashlite.vim.spell import suggestions

    class _NoWords:
        spell_good = set()

    bad = [("teh", "the"), ("seperate", "separate"), ("modle", "model"), ("trainning", "training"),
           ("validaton", "validation")]
    b1, b2 = rng.sample(bad, 2)
    pick1, pick2 = (suggestions(_NoWords, w).index(c) + 1 for w, c in (b1, b2))
    text = f"# Notes\n\nWe check {b1[0]} results daily.\nThe {b2[0]} step is slow.\n"
    items = [
        (f"Turn on spell checking (`:set spell`), jump to the first misspelled word (`]s`), and fix it with `z=`: "
         f"type the number of `{b1[1]}` in the list and press Enter.", f":set spell<CR>]sz={pick1}<CR>",
         {"require": [":set spell", "]s", "z="]}),
        (f"Spell checking: fix BOTH misspelled words with `]s` and `z=` (`{b1[0]}` → `{b1[1]}`, `{b2[0]}` → "
         f"`{b2[1]}`).", f":set spell<CR>]sz={pick1}<CR>]sz={pick2}<CR>", {"require": [":set spell", "]s", "z="]}),
        (f"Spell checking is on and the cursor is at the end. Go BACK to the previous misspelled word with `[s` and "
         f"fix it with `z=` (`{b2[0]}` → `{b2[1]}`).", f"[sz={pick2}<CR>",
         {"prep": [(5, ":set spell<CR>G$")], "require": ["[s", "z="]}),
        (f"`{b1[0]}` is a project term here, not a typo. With spell checking on, mark it as a good word with `zg` "
         f"(on the word), so `]s` skips it; then fix the next misspelling (`]s`, `z=`: `{b2[0]}` → `{b2[1]}`).",
         f":set spell<CR>]szg]sz={pick2}<CR>", {"require": ["zg", "]s", "z="]}),
    ]
    return drill(rng, items, {"notes.md": text}, show="diff")


# ================================================================= challenge

@template(M, "challenge")
def sort_imports(rng):
    imps = ["import torch", "import os", "import numpy as np", "import sys", "from pathlib import Path",
            "import json", "import torch"]
    rng.shuffle(imps)
    body = "\n\ndef main():\n    print(Path.cwd(), np.pi, torch.__version__)\n"
    text = "\n".join(imps) + body
    keys = ":1,7sort u<CR>"
    return V("**Tidy the imports.** The import block (the first 7 lines) is unsorted and has a duplicate. Sort it "
             "alphabetically and drop the duplicate, without touching the rest of the file.\n\n**Constraints:** one "
             f"Ex command; par {count_keys(keys)} keys.", {"main.py": text}, keys, require=[":sort"],
             par=count_keys(keys), show="diff")


@template(M, "challenge")
def report_from_shell(rng):
    logs = "".join(f"{rng.choice(['INFO', 'INFO', 'WARN', 'ERROR'])} step={i}\n" for i in range(1, 13))
    n_err = logs.count("ERROR")
    files = {"report.md": "# Report\n\nErrors:\n", "train.log": logs}
    keys = "G:r !grep -c ERROR train.log<CR>"
    return V("**Report.** Fill in `report.md` (open now): under `Errors:` insert the number of ERROR lines in "
             f"`train.log` (it's {n_err}, but get it from the shell, not by counting: `grep -c ERROR train.log`).\n\n"
             "**Constraints:** the number must come from a command's output read into the buffer (`:r !cmd`).",
             files, keys, start="report.md", require=[":r !"], show="diff")
