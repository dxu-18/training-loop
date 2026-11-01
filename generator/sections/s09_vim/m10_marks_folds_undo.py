"""9.10 Marks, folds & undo history: m ' ` '' `` :marks, global marks, zf zo zc za zR zM and foldmethod=indent,
:earlier / :later (by time and by count), g- g+ through undo branches, and 'undofile'."""

from ...core import pick
from ._common import V, drill, long_py, model_py, template, train_py

M = "9.10"


def _find(lines, needle, start=0):
    for i in range(start, len(lines)):
        j = lines[i].find(needle)
        if j >= 0:
            return i + 1, j + 1
    raise ValueError(needle)


# ================================================================= lite

@template(M, "lite")
def marks_basic(rng):
    text, f = train_py(rng)
    lines = text.split("\n")
    files = {"train.py": text}
    tl, tc = _find(lines, "def train")
    ll, lc = _find(lines, "loss.backward")
    rl, _ = _find(lines, "return model")
    items = [
        ("Set mark `a` at the cursor (`ma`). Submit checks the mark.", "ma",
         {"cursor": (ll, lc + 5), "check": ("text", "marks"), "marks": "a", "require": ["m"]}),
        ("Mark `b` was set on the `def train` line. Jump to that LINE with `'b` (apostrophe: the line's first "
         "non-blank).", "'b", {"prep": [(5, f"{tl}G5|mbG")], "check": ("text", "cursor"), "require": ["'"]}),
        ("Mark `b` was set on the word `train` in `def train`. Jump to its EXACT spot with `` `b `` (backtick).",
         "`b", {"prep": [(5, f"{tl}G5|mbG")], "check": ("text", "cursor"), "require": ["`"]}),
        ("Mark `a` is on the `def train` line. Delete from the cursor's line up to mark a's line, both included "
         "(`d'a`).", "d'a", {"prep": [(5, f"{tl}GmaG")], "cursor": (rl + 1, 1), "require": ["'"]}),
    ]
    return drill(rng, items, files, show=None)


@template(M, "lite")
def jump_back(rng):
    text, f = train_py(rng)
    lines = text.split("\n")
    files = {"train.py": text}
    ll, lc = _find(lines, "loss.backward")
    tl, _ = _find(lines, "def train")
    letter = rng.choice("cdefgh")
    items = [
        ("You just jumped to the end of the file with `G`. Jump back to the line you were on with `''` (two "
         "apostrophes).", "''", {"prep": [(5, f"{ll}G{lc + 3}|G")], "require": ["''"]}),
        ("You just jumped to the end of the file with `G`. Jump back to the exact spot you were on with ``` `` ``` "
         "(two backticks).", "``", {"prep": [(5, f"{ll}G{lc + 3}|G")], "require": ["''"]}),
        ("Several marks are set. List them with `:marks`, find the one on the `def train` line, and jump to it "
         "with `'` and its letter.", f":marks<CR><CR>'{letter}",
         {"prep": [(5, f"{ll}Gmx{tl}Gm{letter}3GmyG")], "require": [":marks", "'"]}),
    ]
    return drill(rng, items, files, check=("text", "cursor"), show=None)


# ================================================================= easy

@template(M, "easy")
def global_marks(rng):
    train, f = train_py(rng)
    model, m = model_py(rng)
    files = {"model.py": model, "train.py": train}
    ml = model.split("\n").index("    def forward(self, x):") + 1
    tl = train.split("\n").index(next(x for x in train.split("\n") if x.startswith("def train"))) + 1
    items = [
        ("A global (uppercase) mark `T` was set in train.py. You're in model.py now: jump straight to the mark with "
         "`'T` (it switches files).", "'T", {"start": "train.py model.py",
                                             "prep": [(5, f"{tl}GmT:bn<CR>")], "require": ["'"]}),
        (f"Set a global mark `F` on the `def forward` line (line {ml}: the cursor is on it) with `mF`, switch to "
         f"train.py (`:bn`), then come back with `'F`.", "mF:bn<CR>'F",
         {"start": "model.py train.py", "cursor": (ml, 5), "require": ["mA", "'"]}),
    ]
    return drill(rng, items, files, check=("text", "cursor"), show=None, target=False)


@template(M, "easy")
def manual_folds(rng):
    text, m = model_py(rng)
    lines = text.split("\n")
    il = lines.index("    def __init__(self, dim={dim}, hidden={hidden}, n_classes={n}):".format(**m)) + 1
    fl = lines.index("    def forward(self, x):") + 1
    files = {"model.py": text}
    items = [
        ("Fold the `__init__` method: on its `def` line, create a fold to the end of the paragraph with `zf}`.",
         "zf}", {"cursor": (il, 5), "require": ["zf"]}),
        ("Fold the `forward` method with `zfap` (a fold over the paragraph and the blank line after it).", "zfap",
         {"cursor": (fl, 5), "require": ["zf"]}),
        ("The `__init__` method is folded. Open the fold with `zo`.", "zo",
         {"cursor": (il, 1), "prep": [(5, f"{il}Gzf}}")], "require": ["zo"]}),
        ("The `__init__` method is folded and open. Close the fold again with `zc`.", "zc",
         {"cursor": (il, 1), "prep": [(5, f"{il}Gzf}}zo")], "require": ["zc"]}),
        ("Toggle the fold on `__init__` (it's closed) with `za`, then toggle the one on `forward` (also closed).",
         f"za{fl}Gza", {"cursor": (il, 1), "prep": [(5, f"{il}Gzf}}{fl}Gzf}}")], "require": ["za"]}),
    ]
    return drill(rng, items, files, check=("text", "folds"), show=None)


# ================================================================= medium

@template(M, "medium")
def indent_folds(rng):
    text, defs = long_py(rng, 6)
    files = {"metrics.py": text}
    k = rng.randrange(1, len(defs))
    items = [
        ("Fold the file by indentation: `:set foldmethod=indent` (every function body folds and starts closed).",
         ":set foldmethod=indent<CR>", {"require": [":set foldmethod"]}),
        (f"Fold by indentation (`:set foldmethod=indent`), then open just the body of the function defined on "
         f"line {defs[k]}: the fold is the indented body, so go to the folded line under the `def` "
         f"(`{defs[k] + 1}G`) and press `zo` (on the `def` line itself: E490, no fold there).",
         f":set foldmethod=indent<CR>{defs[k] + 1}Gzo",
         {"require": [":set foldmethod", "zo"]}),
        ("Folding by indentation is on and everything is closed. Open ALL the folds with `zR`.", "zR",
         {"prep": [(5, ":set foldmethod=indent<CR>")], "require": ["zR"]}),
        ("Folding by indentation is on and all folds are open. Close them all with `zM`.", "zM",
         {"prep": [(5, ":set foldmethod=indent<CR>zR")], "require": ["zM"]}),
    ]
    return drill(rng, items, files, check=("text", "folds", "options"), options=("foldmethod",), show=None)


@template(M, "medium")
def time_travel(rng):
    text, f = train_py(rng)
    files = {"train.py": text}
    lines = text.split("\n")
    el, _ = _find(lines, f"EPOCHS = {f['epochs']}")
    prep = [(5400, f"{el}GA  # 90 minutes ago<Esc>"), (2700, "ggOimport os  # 45 minutes ago<Esc>"),
            (60, "Gdd")]
    items = [
        ("Three changes were made: 90 minutes ago, 45 minutes ago and a minute ago (the last line was deleted). Go "
         "back to how the file looked 20 minutes ago with `:earlier 20m`.", ":earlier 20m<CR>",
         {"require": [":earlier"]}),
        ("Go all the way back to the file as it was before every change (`:earlier 2h`), then forward again by "
         "one change (`:later 1`).", ":earlier 2h<CR>:later 1<CR>", {"require": [":earlier", ":later"]}),
        ("Go back 2 changes in time with `:earlier 2` (a count is a number of changes, like `u`).",
         ":earlier 2<CR>", {"require": [":earlier"]}),
    ]
    return drill(rng, items, files, prep=prep, show="diff",
                 hint="Vim is open on `train.py` in the terminal. The times count from when this problem opened, "
                      "so answer within ~15 minutes. Submit checks the result and the commands you used; Reset "
                      "starts over.")


# ================================================================= hard

@template(M, "hard")
def undo_branches(rng):
    text, f = train_py(rng)
    files = {"train.py": text}
    a, b = rng.sample(["import os", "import sys", "import json", "import math"], 2)
    prep = [(300, f"ggO{a}<Esc>"), (200, "u"), (100, f"ggO{b}<Esc>")]
    items = [
        (f"You added `{a}` at the top, undid it, then added `{b}` instead. `u` and Ctrl-R can't reach the "
         f"`{a}` version any more (it's on another branch of the undo tree), but `g-` steps back through EVERY "
         f"state in time order. Get the `{a}` version back.", "g-", {"require": ["g-"]}),
        (f"You added `{a}`, undid it, added `{b}` instead, then went back in time with `g-` twice (to the original "
         f"file). Step FORWARD in time with `g+` to the `{a}` version.", "g+",
         {"prep": prep + [(50, "g-g-")], "require": ["g+"]}),
    ]
    return drill(rng, items, files, prep=prep, show="diff")


@template(M, "hard")
def undo_file(rng):
    text, f = train_py(rng)
    files = {"train.py": text}
    keys = ":set undofile<CR>:w<CR>:bd<CR>:e train.py<CR>u"
    return V("A line was added at the top (not saved yet). Normally the undo history is lost when a buffer is "
             "unloaded. Turn on `:set undofile`, save, close the buffer (`:bd`), open train.py again (`:e train.py`), "
             "and undo the added line with `u`: the history survived.", files, keys,
             prep=[(30, "ggO# added<Esc>")], check=("text", "options"), options=("undofile",),
             require=[":set undofile", ":bd", "u"], show=None)


# ================================================================= challenge

@template(M, "challenge")
def marks_move(rng):
    text, f = train_py(rng)
    lines = text.split("\n")
    files = {"train.py": text}
    fl, _ = _find(lines, "for inputs, targets")
    tl, _ = _find(lines, "total += loss")
    keys = "'ad'bGp"
    return V(f"**Move a block between marks.** Mark `a` is on line {fl} and mark `b` on line {tl}. Cut the lines "
             f"from mark a to mark b (both included) and paste them at the end of the file.\n\n**Constraints:** use "
             f"the marks, not line numbers; par 7 keys.", files, keys, prep=[(5, f"{fl}Gma{tl}Gmbgg")],
             require=["'"], forbid=[":N"], par=7)


@template(M, "challenge")
def restore_and_save(rng):
    text, f = train_py(rng)
    lines = text.split("\n")
    files = {"train.py": text}
    el, _ = _find(lines, f"EPOCHS = {f['epochs']}")
    prep = [(3000, f"{el}Gdd"), (1500, "Gdk"), (40, "ggdd")]
    keys = ":earlier 1h<CR>:w<CR>"
    return V("**Undo the damage.** Over the last hour someone deleted three parts of train.py (50, 25 and 1 minutes "
             "ago) and the deletions aren't saved... yet. Bring the file back to how it was an hour ago and SAVE it.\n\n"
             "**Constraints:** one time-travel command, then save.", files, keys, prep=prep, check=("files",),
             require=[":earlier"], show=None)
