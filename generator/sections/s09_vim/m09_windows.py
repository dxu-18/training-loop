"""9.09 Buffers, windows & tabs: :e :ls :b :bn :bp :bd Ctrl-^, :sp :vsp and Ctrl-W (s v h j k l w q o = _ |),
:resize, tab pages (:tabnew gt gT :tabclose), netrw (:Explore :Vexplore) and :terminal."""

from ...core import pick
from ._common import MODELS, V, config_yaml, drill, model_py, template, train_py

M = "9.09"


def _project(rng):
    train, f = train_py(rng)
    model, _ = model_py(rng)
    cfg, _ = config_yaml(rng)
    utils = "def seed_everything(seed):\n    import random\n    random.seed(seed)\n"
    return {"model.py": model, "train.py": train, "utils.py": utils, "config.yaml": cfg}


ARGS = "model.py train.py utils.py"


# ================================================================= lite

@template(M, "lite")
def buffers(rng):
    files = _project(rng)
    cur = ("text", "cursor")
    items = [
        ("Three files are open as buffers (only one shows). List them with `:ls`, then switch to buffer 3 "
         "(`:b 3`).", ":ls<CR><CR>:b 3<CR>", {"require": [":ls", ":b"]}),
        ("Switch to the `utils.py` buffer by part of its name: `:b util`.", ":b util<CR>", {"require": [":b"]}),
        ("Go to the next buffer twice (`:bn`), then back one (`:bp`).", ":bn<CR>:bn<CR>:bp<CR>",
         {"require": [":bn", ":bp"]}),
        ("You switched to utils.py a moment ago. Toggle back to the file you were in before with Ctrl-^ (the "
         "alternate file).", "<C-^>", {"prep": [(5, ":b 3<CR>")], "require": ["<C-^>"]}),
        ("Close the current buffer (model.py) with `:bd`: it leaves the buffer list.", ":bd<CR>",
         {"check": ("buffers",), "require": [":bd"]}),
    ]
    return drill(rng, items, files, start=ARGS, check=cur, show=None, target=False)


@template(M, "lite")
def splits(rng):
    files = _project(rng)
    items = [
        ("Split the window horizontally (`:sp`): both halves show model.py.", ":sp<CR>", {"require": [":sp"]}),
        ("Open train.py in a vertical split (`:vsp train.py`).", ":vsp train.py<CR>", {"require": [":vsp"]}),
        ("Split the window with Ctrl-W s (horizontal), then split the new top window with Ctrl-W v (vertical).",
         "<C-w>s<C-w>v", {"require": ["<C-w>s", "<C-w>v"]}),
        ("Open config.yaml in a horizontal split (`:sp config.yaml`).", ":sp config.yaml<CR>", {"require": [":sp"]}),
    ]
    return drill(rng, items, files, check=("layout",), show=None)


# ================================================================= easy

@template(M, "easy")
def window_moves(rng):
    files = _project(rng)
    note = rng.choice(["# reviewed", "# TODO: tests", "# ok"])
    items = [
        (f"model.py and train.py are side by side; the cursor is in the left one. Move to the RIGHT window (Ctrl-W l) "
         f"and add `{note}` as its last line.", f"<C-w>lGo{note}<Esc>",
         {"start": "-O model.py train.py", "require": ["<C-w>l"]}),
        (f"The cursor is in the right window (train.py). Move to the LEFT one (Ctrl-W h) and add `{note}` as its "
         f"last line.", f"<C-w>hGo{note}<Esc>",
         {"start": "-O model.py train.py", "prep": [(5, "<C-w>l")], "require": ["<C-w>h"]}),
        (f"model.py is above train.py. Move DOWN to train.py's window (Ctrl-W j) and add `{note}` as its first "
         f"line.", f"<C-w>jO{note}<Esc>", {"start": "-o model.py train.py", "require": ["<C-w>j"]}),
        (f"The cursor is in the bottom window. Move UP (Ctrl-W k) and add `{note}` at the top of that file.",
         f"<C-w>kggO{note}<Esc>", {"start": "-o model.py train.py", "prep": [(5, "<C-w>j")], "require": ["<C-w>k"]}),
        (f"Three windows are open side by side. Cycle to the THIRD one with Ctrl-W w (twice) and add `{note}` as its "
         f"last line.", f"<C-w>w<C-w>wGo{note}<Esc>", {"start": "-O model.py train.py utils.py",
                                                       "require": ["<C-w>w"]}),
    ]
    return drill(rng, items, files, check=("text", "layout"), show=None)


@template(M, "easy")
def close_windows(rng):
    files = _project(rng)
    prep3 = [(5, ":sp train.py<CR>:vsp utils.py<CR>")]
    items = [
        ("Close the current window (Ctrl-W q); the others stay.", "<C-w>q", {"prep": prep3, "require": ["<C-w>q"]}),
        ("Close every window except the current one (Ctrl-W o).", "<C-w>o", {"prep": prep3, "require": ["<C-w>o"]}),
        ("Close the window showing model.py (move there first: Ctrl-W j), keep the other two.", "<C-w>j<C-w>q",
         {"prep": prep3, "require": ["<C-w>q"]}),
        ("Make the window showing train.py the only one: go to it (Ctrl-W h) and close the others with `:only`.",
         "<C-w>h:only<CR>", {"prep": prep3, "require": [":only|<C-w>o"]}),
    ]
    return drill(rng, items, files, check=("layout",), show=None)


# ================================================================= medium

@template(M, "medium")
def window_sizes(rng):
    files = _project(rng)
    h = rng.choice((5, 6, 8))
    w = rng.choice((30, 35, 40))
    items = [
        ("The two windows have different sizes. Make them equal (Ctrl-W =).", "<C-w>=",
         {"prep": [(5, ":sp train.py<CR>:resize 4<CR>")], "require": ["<C-w>="]}),
        ("Make the current window as tall as possible (Ctrl-W _).", "<C-w>_",
         {"prep": [(5, ":sp train.py<CR>")], "require": ["<C-w>_"]}),
        ("Make the current window as wide as possible (Ctrl-W |).", "<C-w>|",
         {"start": "-O model.py train.py", "require": ["<C-w>|"]}),
        (f"Set the current window's height to {h} lines (`:resize {h}`).", f":resize {h}<CR>",
         {"prep": [(5, ":sp train.py<CR>")], "require": [":resize"]}),
        (f"Set the current window's width to {w} columns (`:vertical resize {w}`).", f":vertical resize {w}<CR>",
         {"start": "-O model.py train.py", "require": [":vertical resize"]}),
    ]
    return drill(rng, items, files, check=("layout", "sizes"), show=None)


@template(M, "medium")
def tabs(rng):
    files = _project(rng)
    items = [
        ("Open train.py in a new tab page (`:tabnew train.py`).", ":tabnew train.py<CR>", {"require": [":tabnew"]}),
        ("Open train.py and then utils.py in new tabs, then go back to the FIRST tab (`gt` wraps around, `gT` goes "
         "back).", ":tabnew train.py<CR>:tabnew utils.py<CR>gt", {"require": [":tabnew", "gt|gT"]}),
        ("Three tabs are open and the third is current. Go to the previous tab with `gT`.", "gT",
         {"prep": [(5, ":tabnew train.py<CR>:tabnew utils.py<CR>")], "require": ["gT"]}),
        ("Three tabs are open. Close the current tab with `:tabclose`.", ":tabclose<CR>",
         {"prep": [(5, ":tabnew train.py<CR>:tabnew utils.py<CR>gt")], "require": [":tabclose"]}),
        ("Open config.yaml in a new tab, then go back to the first tab with `gt`.", ":tabnew config.yaml<CR>gt",
         {"require": [":tabnew", "gt"]}),
    ]
    return drill(rng, items, files, check=("layout",), show=None)


# ================================================================= hard

@template(M, "hard")
def file_browser(rng):
    files = _project(rng)
    target = rng.choice(["utils.py", "config.yaml", "train.py"])
    items = [
        (f"Open the built-in file browser with `:Explore` (`:Ex`), move to `{target}` (search with `/{target}` "
         f"Enter, or j/k) and open it with Enter.", f":Explore<CR>/{target}<CR><CR>",
         {"check": ("text", "cursor"), "require": [":Explore"], "target": False}),
        ("Open the file browser in a vertical split on the left (`:Vexplore`) and leave it open.", ":Vexplore<CR>",
         {"check": ("layout",), "require": [":Vexplore"]}),
    ]
    return drill(rng, items, files, show=None)


@template(M, "hard")
def terminal_split(rng):
    files = _project(rng)
    items = [
        ("Open a shell in a split window with `:terminal`, run `python --version` in it, then go back to the "
         "model.py window (Ctrl-W w) and leave both open.", ":terminal<CR>python --version<CR><C-w>w",
         {"require": [":terminal"]}),
        ("Open a `:terminal` split, then switch it to Normal mode with Ctrl-W N (so you could scroll and yank its "
         "output); leave the cursor there.", ":terminal<CR><C-w>N", {"require": [":terminal", "<C-w>N"]}),
    ]
    return drill(rng, items, files, check=("layout",), show=None)


# ================================================================= challenge

@template(M, "challenge")
def build_layout(rng):
    files = _project(rng)
    keys = ":vsp train.py<CR>:sp config.yaml<CR>"
    return V("**Workbench.** Arrange the windows like this (model.py is open now):\n\n```\n+-------------+-----------+\n"
             "| config.yaml |           |\n+-------------+ model.py  |\n| train.py    |           |\n"
             "+-------------+-----------+\n```\n\nThe cursor should end in config.yaml. (Without `splitright`, `:vsp` "
             "opens the new window on the LEFT; `:sp` opens it ABOVE.)", files, keys, check=("layout",),
             require=[":vsp|<C-w>v", ":sp|<C-w>s"], show=None)


@template(M, "challenge")
def three_files(rng):
    files = _project(rng)
    keys = ":set hidden<CR>ggO# reviewed<Esc>:bn<CR>ggO# reviewed<Esc>:bn<CR>ggO# reviewed<Esc>:wa<CR>"
    return V("**Review stamp.** Three files are open as buffers. Put a first line `# reviewed` at the top of ALL "
             "THREE, and save them.\n\n**Constraints:** stay in one window and move between buffers (`:bn`, `:b`, "
             "Ctrl-^). Note: Vim won't leave a buffer with unsaved changes (E37) unless you save it first (`:w`) or "
             "turn on `:set hidden`, which lets changed buffers stay open in the background (then `:wa` saves them "
             "all).", files, keys, start=ARGS, check=("files",), require=[":bn|:b|:bp"],
             forbid=[":sp", ":vsp", ":tabnew"], show=None)
