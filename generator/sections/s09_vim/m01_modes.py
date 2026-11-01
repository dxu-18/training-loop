"""9.01 Modes, opening, saving & quitting.

The one module where Vim may be quit: some problems start at the bash prompt (open Vim, edit, save, quit), others
start in Vim with unsaved changes to keep or throw away.
"""

from ...core import pick
from ._common import MODELS, WORDS, V, config_yaml, count_keys, drill, notes_md, template, train_py

M = "9.01"


def _bullets(rng, n=4):
    items = rng.sample(["check the data loader", "plot the loss curve", "tune the learning rate", "add a val split",
                        "try mixed precision", "log grad norms", "save best checkpoint", "write the README",
                        "profile one step", "fix the seed"], n)
    return "# TODO\n" + "".join(f"- {x}\n" for x in items)


# ================================================================= lite

@template(M, "lite")
def save_and_quit(rng):
    text = _bullets(rng)
    item = rng.choice(["clean the cache", "email the team", "book GPU time", "rerun eval", "update deps"])
    f = {"notes.md": text}
    common = dict(shell=True, check=("files", "quit"), show="target")
    items = [
        (f"Open `notes.md` in Vim (`vim notes.md`), add the line `- {item}` at the END of the file, then save and "
         f"quit with `:wq`. (In Normal mode `G` jumps to the last line and `o` opens a new line below it; Esc gets "
         f"you back to Normal mode.)", f"vim notes.md<CR>Go- {item}<Esc>:wq<CR>", {"require": [":wq"]}),
        (f"Open `notes.md`, add `- {item}` as a new last line, then save and quit with `ZZ` (capital Z twice).",
         f"vim notes.md<CR>Go- {item}<Esc>ZZ", {"require": ["ZZ"]}),
        (f"Open `notes.md`, add `- {item}` as a new last line, then save and quit with `:x`.",
         f"vim notes.md<CR>Go- {item}<Esc>:x<CR>", {"require": [":x"]}),
        (f"Open `notes.md`, insert `- {item}` as a new line right under `# TODO` (`o` opens a line below the "
         f"cursor's), then save and quit.", f"vim notes.md<CR>o- {item}<Esc>:wq<CR>", {"require": [":wq|:x|ZZ"]}),
        (f"Open `notes.md`, put `URGENT ` at the very start of the first line (i inserts before the cursor), then "
         f"save and quit.", "vim notes.md<CR>iURGENT <Esc>:wq<CR>", {"require": ["i", ":wq|:x|ZZ"]}),
    ]
    return drill(rng, items, f, **common)


@template(M, "lite")
def quit_without_saving(rng):
    text, ps = notes_md(rng, paras=2, per=2)
    junk = rng.choice(["asdf", "jjjjkkkk", "qqqq", "xxxxx", ":wq"])
    prep = [(5, f"ggi{junk}<Esc>Gdd")]
    common = dict(quit_ok=True, check=("quit", "files"), prep=prep, show=None)
    a, b = rng.sample(MODELS, 2)
    items = [
        ("Someone typed into `notes.md` while you were away: the buffer has unsaved changes you don't want. Quit "
         "Vim WITHOUT saving them (`:q!`).", ":q!<CR>", {"require": [":q!"]}),
        ("The buffer has unsaved junk in it. Quit without saving, using the Normal-mode shortcut `ZQ`.", "ZQ",
         {"require": ["ZQ"]}),
        ("Try `:q` first: Vim refuses (E37: No write since last change) because the buffer has unsaved changes. "
         "Then quit for real, throwing the changes away.", ":q<CR>:q!<CR>",
         {"require": [":q", ":q!|ZQ"], "errors_ok": True}),
        (f"Two windows are open (`{a}.md` and `{b}.md`) and both have unsaved junk. Quit everything without saving, "
         f"with one command (`:qa!`).", ":qa!<CR>",
         {"require": [":qa!"], "files": {f"{a}.md": text, f"{b}.md": text}, "start": f"-O {a}.md {b}.md",
          "prep": [(5, f"ggi{junk}<Esc><C-w>wGdd<C-w>w")]}),
    ]
    return drill(rng, items, {"notes.md": text}, **common)


# ================================================================= easy

@template(M, "easy")
def modes_tour(rng):
    text, f = train_py(rng)
    lines = text.split("\n")
    word = rng.choice(["fast", "tiny", "debug", "full"])
    new_lr = rng.choice([x for x in ("0.002", "0.005", "0.009") if len(x) == len(f["lr"])] or ["X" * len(f["lr"])])
    lr_line = lines.index(f"LR = {f['lr']}") + 1
    ep_line = lines.index(f"EPOCHS = {f['epochs']}") + 1
    files = {"train.py": text}
    common = dict(check=("text", "mode"))
    items = [
        (f"The cursor is on `train` in `def train(...)`. Press `i`, type `{word}_` (it goes in front of the cursor), "
         f"then Esc back to Normal mode.", f"i{word}_<Esc>", {"cursor": (10, 5), "require": ["i"]}),
        (f"The cursor is on the first character of `{f['lr']}`. Overwrite the value with `{new_lr}` using Replace "
         f"mode (`R`, type, Esc).", f"R{new_lr}<Esc>", {"cursor": (lr_line, 6), "require": ["R"]})
        if new_lr[0] != "X" else
        (f"The cursor is on `E` of `EPOCHS`. Overwrite `EPOCHS` with `ROUNDS` using Replace mode (`R`).",
         "RROUNDS<Esc>", {"cursor": (ep_line, 1), "require": ["R"]}),
        ("The cursor is at the start of `optimizer.zero_grad()`. Select `optimizer` in Visual mode (`v`, then `e` "
         "moves to the end of the word) and delete the selection with `d`.", "ved",
         {"cursor": (16, 13), "require": ["v", "v_d"]}),
        (f"Delete the line `LR = {f['lr']}` (the cursor is on it) using Visual Line mode: `V` selects the whole line, "
         f"`d` deletes it.", "Vd", {"cursor": (lr_line, 1), "require": ["V", "v_d"]}),
        ("The cursor is on the first `i` of the `import` lines. Delete the first character of lines 1-3 at once with "
         "Visual Block mode: Ctrl-V, `2j` (down two lines), `d`.", "<C-v>2jd",
         {"cursor": (1, 1), "require": ["<C-v>", "v_d"]}),
        ("The buffer has an unsaved change. Save it from Command-line mode: type `:w` and press Enter.", ":w<CR>",
         {"prep": [(5, f"{ep_line}GA  # changed<Esc>")], "check": ("files",), "require": [":w"]}),
    ]
    return drill(rng, items, files, **common)


@template(M, "easy")
def files_and_buffers(rng):
    text, f = train_py(rng)
    cfg, c = config_yaml(rng)
    new_ep = c["epochs"] * 2
    lines = text.split("\n")
    ep_line = lines.index(f"EPOCHS = {f['epochs']}") + 1
    name = rng.choice(WORDS)
    files = {"train.py": text, "config.yaml": cfg}
    items = [
        (f"Save a copy of this buffer as `train_{name}.py` with `:w train_{name}.py` (train.py itself stays as it is).",
         f":w train_{name}.py<CR>", {"check": ("files",), "require": [":w"]}),
        (f"Open `config.yaml` with `:e config.yaml`, change `epochs: {c['epochs']}` to `epochs: {new_ep}` (the "
         f"cursor can sit on the number: `cw` changes a word, or use `R`), then save it.",
         f":e config.yaml<CR>5G$ciw{new_ep}<Esc>:w<CR>", {"check": ("files",), "require": [":e"]}),
        ("You've made a mess of this buffer (the changes aren't saved). Revert it to the saved file without "
         "quitting: `:e!`.", ":e!<CR>",
         {"prep": [(5, "ggdGiAAAAA<Esc>")], "check": ("text",), "require": [":e!"], "show": None}),
        ("Both windows have unsaved changes. Save them all with one command (`:wa`).", ":wa<CR>",
         {"start": "-O train.py config.yaml", "prep": [(5, f"{ep_line}GA  # 1<Esc><C-w>wGo# 2<Esc><C-w>w")],
          "check": ("files",), "require": [":wa"], "show": None}),
        ("Write the buffer to a new file `draft.py` (`:w draft.py`), then open that file with `:e draft.py`: you "
         "should end up editing draft.py, with train.py unchanged.", ":w draft.py<CR>:e draft.py<CR>",
         {"check": ("files", "buffers"), "require": [":w", ":e"], "show": None}),
    ]
    return drill(rng, items, files)


# ================================================================= medium

@template(M, "medium")
def open_from_shell(rng):
    text, f = train_py(rng)
    model, cfg_t = rng.choice(MODELS), config_yaml(rng)[0]
    lines = text.split("\n")
    n = rng.choice([i + 1 for i, ln in enumerate(lines) if ln.strip().startswith(("print(", "total +=", "loss.backward"))])
    csv = "epoch,loss,acc\n" + "".join(f"{e},{2.0 / e:.3f},{0.5 + e / 40:.3f}\n" for e in range(1, 9))
    files = {"train.py": text, "model.py": f"import torch\n\nMODEL = '{model}'\n", "results.csv": csv,
             "config.yaml": cfg_t}
    common = dict(shell=True, show=None)
    items = [
        (f"From the shell, open `train.py` straight at line {n} (`vim +{n} train.py`), delete that line (`dd`), then "
         f"save and quit.", f"vim +{n} train.py<CR>dd:wq<CR>",
         {"check": ("files", "quit"), "require": ["vim +N"]}),
        ("Open `model.py` and `train.py` side by side in one Vim (`vim -O model.py train.py`). Leave Vim open with "
         "both windows showing, then Submit.", "vim -O model.py train.py<CR>",
         {"check": ("layout",), "require": ["vim -O"]}),
        ("Open `results.csv` read-only with `view results.csv`, then quit Vim.", "view results.csv<CR>:q<CR>",
         {"check": ("quit", "files"), "require": ["view"]}),
        ("Open `config.yaml` at its last line (`vim + config.yaml` jumps to the end; `vim +N` to line N), add a line "
         "`notes: baseline` under it, save and quit.", "vim + config.yaml<CR>onotes: baseline<Esc>:wq<CR>",
         {"check": ("files", "quit"), "require": ["vim +N"]}),
    ]
    return drill(rng, items, files, **common)


@template(M, "medium")
def keep_or_discard(rng):
    a_t, _ = train_py(rng)
    b_t, _ = config_yaml(rng)
    files = {"train.py": a_t, "config.yaml": b_t}
    prep = [(10, "<C-w>wGoextra: 1<Esc><C-w>wGo# left edit<Esc>")]
    common = dict(start="-O train.py config.yaml", quit_ok=True, prep=prep, check=("files", "quit"), show=None)
    items = [
        ("Both windows have unsaved changes (the cursor is in the left one, train.py). Save train.py but throw away "
         "the changes to config.yaml, and quit Vim.", ":w<CR>:qa!<CR>", {"require": [":w"]}),
        ("Save both files and quit, with one command.", ":wqa<CR>", {"require": [":wqa|:xa"]}),
        ("Throw away all the changes and quit, with one command.", ":qa!<CR>", {"require": [":qa!"]}),
        ("Save config.yaml (the right window: Ctrl-W w moves there), revert train.py to its saved version, then quit.",
         "<C-w>w:w<CR>:qa!<CR>", {}),
    ]
    return drill(rng, items, files, **common)


# ================================================================= hard

@template(M, "hard")
def fast_exit(rng):
    text = _bullets(rng, 5)
    item = rng.choice(["ship it", "tag v1", "freeze deps", "archive runs"])
    files = {"todo.md": text}
    items = [
        (f"Append the line `- {item}` at the end of the file, save and quit.", f"Go- {item}<Esc>ZZ"),
        (f"Insert `- {item}` as a new line right under the heading (line 1), save and quit.", f"o- {item}<Esc>ZZ"),
        ("Delete the last line, save and quit.", "GddZZ"),
        (f"Insert `# {item}` as a new FIRST line, save and quit.", f"O# {item}<Esc>ZZ"),
    ]
    prompt, keys = pick(rng, items)
    return V(prompt + " Golf it: every key counts.", files, keys, quit_ok=True, check=("files", "quit"),
             par=count_keys(keys), show="target")


@template(M, "hard")
def readonly_rescue(rng):
    cfg, c = config_yaml(rng)
    new_bs = c["bs"] * 2
    files = {"config.yaml": cfg, "train.log": "\n".join(f"step {i} ok" for i in range(1, 9)) + "\n"}
    items = [
        (f"Open `config.yaml` with `view` (read-only), change `batch_size: {c['bs']}` to `batch_size: {new_bs}`, "
         f"then save anyway (`:w!` overrides read-only) and quit.",
         f"view config.yaml<CR>5G$ciw{new_bs}<Esc>:w!<CR>:q<CR>", {"require": ["view", ":w!"]}),
        ("Open `train.log`, delete its first line, save the result as `train_trim.log` (train.log itself must stay "
         "unchanged on disk), then quit without saving train.log.",
         "vim train.log<CR>dd:w train_trim.log<CR>:q!<CR>", {"require": [":q!|ZQ"]}),
        ("Open `config.yaml`, delete the comment line at the top, then quit with a command that saves only if "
         "something changed (`:x` or `ZZ`, not `:wq`).", "vim config.yaml<CR>dd:x<CR>",
         {"require": [":x|ZZ"], "forbid": [":wq"]}),
    ]
    return drill(rng, items, files, shell=True, check=("files", "quit"), show=None)


# ================================================================= challenge

@template(M, "challenge")
def exit_gauntlet(rng):
    text, f = train_py(rng)
    lines = text.split("\n")
    n = lines.index("        print(f\"epoch {epoch}: loss {total / len(loader):.4f}\")") + 1
    files = {"train.py": text}
    keys = f"vim +{n} train.py<CR>dd:w train_quiet.py<CR>:q!<CR>"
    return V(f"**Quiet copy.** From the shell: open `train.py` directly at line {n} (the `print`), delete that line, "
             f"save the result as `train_quiet.py`, and quit WITHOUT changing train.py on disk.\n\n"
             f"**Constraints:** open at the line from the command line (`vim +{n}`): no moving around inside Vim. "
             f"Only the keys pressed inside Vim count toward the par.",
             files, keys, shell=True, check=("files", "quit"), require=["vim +N", ":q!|ZQ"],
             forbid=["j", "k", "G", "gg", ":N"], par=count_keys(keys.split("<CR>", 1)[1]) + 2, show=None)


@template(M, "challenge")
def two_files(rng):
    a, b = rng.sample(["notes.md", "todo.md", "ideas.md", "log.md"], 2)
    ta, tb = _bullets(rng, 3), _bullets(rng, 3)
    x, y = rng.sample(["rerun ablation", "share results", "fix NaNs", "add tests", "clean data"], 2)
    files = {a: ta, b: tb}
    keys = f"vim -O {a} {b}<CR>Go- {x}<Esc><C-w>wGo- {y}<Esc>:wqa<CR>"
    return V(f"**Two lists.** From the shell, open `{a}` and `{b}` side by side in one Vim, add `- {x}` at the end "
             f"of {a} and `- {y}` at the end of {b} (`Ctrl-W w` moves to the other window), then save both and quit "
             f"with ONE command.",
             files, keys, shell=True, check=("files", "quit"), require=["vim -O", ":wqa|:xa"], show=None)
