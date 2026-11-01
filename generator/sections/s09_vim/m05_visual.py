"""9.05 Visual mode: v V Ctrl-V, o gv, operators on the selection, :'<,'> commands, and block edits (I A $A c)."""

from ...core import pick
from ._common import DATASETS, MODELS, VARS, WORDS, V, count_keys, drill, template, train_py

M = "9.05"


def _find(lines, needle, start=0):
    for i in range(start, len(lines)):
        j = lines[i].find(needle)
        if j >= 0:
            return i + 1, j + 1
    raise ValueError(needle)


# ================================================================= lite

@template(M, "lite")
def visual_ops(rng):
    text, f = train_py(rng)
    lines = text.split("\n")
    files = {"train.py": text}
    el, _ = _find(lines, f"EPOCHS = {f['epochs']}")
    ol, oc = _find(lines, "optimizer = torch")
    zl, zc = _find(lines, "optimizer.zero_grad()")
    items = [
        ("Delete the three constant lines (EPOCHS, LR, BATCH_SIZE): select them with `V` and `jj`, then `d`.",
         "Vjjd", (el, 1), ["V", "v_d"]),
        ("Uppercase the word `optimizer` under the cursor: select it with `v` and `e`, then `U`.", "veU",
         (ol, oc), ["v", "v_U"]),
        ("Lowercase the three constant lines: `V`, `jj`, then `u`.", "Vjju", (el, 1), ["V", "v_u"]),
        ("Toggle the case of the cursor's line with `V` then `~`.", "V~", (el, 1), ["V", "v_~"]),
        ("Join the EPOCHS and LR lines into one: `Vj` then `J`.", "VjJ", (el, 1), ["V", "v_J"]),
        ("Replace the three constant lines with a single line `CONFIG = load_config()`: select them (`Vjj`) and "
         "change them (`c`).", "Vjjc" + "CONFIG = load_config()<Esc>", (el, 1), ["V", "v_c"]),
        ("Indent the `optimizer.zero_grad()` line and the two below it: `Vjj` then `>`.", "Vjj>", (zl, 1),
         ["V", "v_>"]),
        ("Copy the word `optimizer` (select it with `ve`, yank with `y`), then paste it at the end of the line "
         "(`$p`).", "vey$p", (ol, oc), ["v", "v_y"]),
    ]
    prompt, keys, cur, req = pick(rng, items)
    return V(prompt, files, keys, cursor=cur, require=req)


@template(M, "lite")
def visual_objects(rng):
    a, b = rng.sample(VARS, 2)
    model = rng.choice(MODELS)
    text = (f"def run({a}, {b}):\n    out = forward({a}, {b}, training=True)\n    name = \"{model}\"\n\n"
            f"    total = sum({a})\n    mean = total / len({a})\n    return out, mean\n")
    lines = text.split("\n")
    files = {"run.py": text}
    items = [
        (f"Uppercase the word `{model}` with `viw` (select inner word) then `U`.", "viwU",
         (3, lines[2].index(model) + 2), ["v", "iw", "v_U"]),
        ("Delete the arguments of `forward(...)` (keep the parentheses): `vi(` then `d`.", "vi(d",
         (2, lines[1].index("training") + 1), ["v", "i(", "v_d"]),
        ("Indent the paragraph of lines 5-7 one more level: `vip` then `>`.", "vip>", (5, 5), ["v", "ip", "v_>"]),
        ("A selection was made earlier and abandoned. Reselect it with `gv` and delete it.", "gvd", None, ["gv"]),
        ("Select from the cursor to the end of the line (`v$`), jump to the other end of the selection (`o`), extend "
         "it back over the previous word (`b`), then delete (`d`).", "v$obd",
         (2, lines[1].index("training") + 1), ["v_o"]),
    ]
    prompt, keys, cur, req = pick(rng, items)
    prep = [(5, f"6G0wve<Esc>gg")] if keys == "gvd" else ()
    return V(prompt, files, keys, cursor=cur, require=req, prep=prep)


# ================================================================= easy

@template(M, "easy")
def block_comment(rng):
    text, f = train_py(rng)
    lines = text.split("\n")
    zl, zc = _find(lines, "optimizer.zero_grad()")
    files = {"train.py": text}
    n = rng.choice((2, 3, 4))
    commented = lines[:]
    for i in range(zl - 1, zl - 1 + n + 1):
        commented[i] = commented[i][:zc - 1] + "# " + commented[i][zc - 1:]
    js = "function f() {\n" + "".join(f"  step{i}();\n" for i in range(1, 6)) + "}\n"
    items = [
        (f"Comment out the {n + 1} lines starting at `optimizer.zero_grad()`: Ctrl-V, `{n}j`, then `I# ` and Esc "
         f"(the text goes in front of the block on every line).", f"<C-v>{n}jI# <Esc>", (zl, zc),
         ["<C-v>", "v_I"], files),
        (f"These {n + 1} lines are commented out with `# `. Uncomment them: select the two columns `# ` with Ctrl-V "
         f"(`{n}j` and `l`), then `d`.", f"<C-v>{n}jld", (zl, zc), ["<C-v>", "v_d"],
         {"train.py": "\n".join(commented)}),
        ("Comment out the five `step` calls with `// ` using a block insert.", "<C-v>4jI// <Esc>", (2, 3),
         ["<C-v>", "v_I"], {"steps.js": js}),
    ]
    prompt, keys, cur, req, fl = pick(rng, items)
    return V(prompt, fl, keys, cursor=cur, require=req)


@template(M, "easy")
def block_append(rng):
    names = rng.sample(VARS, 5)
    text = "".join(f"{n} = None\n" for n in names)
    lvls = "".join(f"INFO {w}\n" for w in rng.sample(WORDS, 4))
    items = [
        ("Append `  # TODO` at the END of all five lines (they have different lengths): Ctrl-V, `4j`, `$`, then "
         "`A  # TODO` and Esc.", "<C-v>4j$A  # TODO<Esc>", {"files": {"vars.py": text}, "require": ["<C-v>", "v_A"]}),
        ("Put `self.` in front of all five names with ONE block insert.", "<C-v>4jIself.<Esc>",
         {"files": {"vars.py": text}, "require": ["<C-v>", "v_I"]}),
        ("Change `INFO` into `WARN` on all four lines at once: select the four `INFO` words as a block (Ctrl-V, `3j`, "
         "`e`), then `c`, type `WARN`, Esc.", "<C-v>3jecWARN<Esc>",
         {"files": {"levels.log": lvls}, "require": ["<C-v>", "v_c"]}),
        ("Append `,` after `None` on all five lines with a block append (`$A`).", "<C-v>4j$A,<Esc>",
         {"files": {"vars.py": text}, "require": ["<C-v>", "v_A"]}),
    ]
    return drill(rng, items, {"vars.py": text}, show="target")


# ================================================================= medium

@template(M, "medium")
def visual_indent(rng):
    py = ("def step(model, batch):\nx, y = batch\nout = model(x)\nloss = (out - y).pow(2).mean()\nreturn loss\n")
    deep = ("class Trainer:\n    def fit(self):\n        for epoch in range(3):\n" +
            "".join(f"        log({i})\n" for i in range(3)))
    items = [
        ("The four lines of the function body aren't indented. Select them (`Vjjj` or `vip` from line 2) and indent "
         "them with `>`.", "jVjjj>", {"files": {"step.py": py}, "require": ["v_>"]}),
        ("The three `log(...)` lines belong inside the `for` loop: select them and indent them one level with `>`.",
         "4GVjj>", {"files": {"trainer.py": deep}, "require": ["v_>"]}),
        ("Indent the three `log(...)` lines by 3 more levels with ONE command: select them, then `3>`.", "4GVjj3>",
         {"files": {"trainer.py": deep}, "require": ["v_>"]}),
        ("Indent the function body (lines 2-5) one level, then indent it a second time by reselecting it with `gv` "
         "and pressing `>` again.", "jVjjj>gv>", {"files": {"step.py": py}, "require": ["v_>", "gv"]}),
        ("Let Vim indent the function body: select lines 2-5 and press `=`.", "jVjjj=",
         {"files": {"step.py": py}, "require": ["v_="]}),
        ("Move the three `log(...)` lines out of the `for` loop: select them and dedent them one level with `<`.",
         "4GVjj<",
         {"files": {"trainer.py": deep}, "require": ["v_<"]}),
    ]
    return drill(rng, items, {"step.py": py}, show="target")


@template(M, "medium")
def visual_ex(rng):
    names = rng.sample(VARS, 6)
    imports = ["import torch", "import numpy as np", "import os", "import sys", "import json"]
    rng.shuffle(imports)
    text = "\n".join(imports) + "\n\n" + "".join(f"{n} = 0\n" for n in names) + "print('done')\n"
    items = [
        ("Sort the five import lines: select them (`Vip` from line 1, or `V4j`), then `:sort` (Vim pre-fills "
         "`:'<,'>`).", "Vip:sort<CR>", {"require": ["v_:", ":sort"]}),
        ("Replace `= 0` with `= None` on the six assignment lines only: select them, then `:s/0/None/`.",
         "7GVip:s/0/None/<CR>", {"require": ["v_:", ":'<,'>s"]}),
        ("Append ` # init` to the six assignment lines: select them, then `:normal A # init`.",
         "7GVip:normal A # init<CR>", {"require": ["v_:", ":normal"]}),
        ("Delete the six assignment lines with an Ex command on a selection (`:d`).", "7GVip:d<CR>",
         {"require": ["v_:"]}),
    ]
    return drill(rng, items, {"setup.py": text}, show="target")


# ================================================================= hard

@template(M, "hard")
def table_columns(rng):
    rows = [(e, round(rng.uniform(0.2, 1.5), 3), round(rng.uniform(0.5, 0.95), 3)) for e in range(1, 6)]
    text = "epoch  loss   acc\n" + "".join(f"{e:<5}  {l:.3f}  {a:.3f}\n" for e, l, a in rows)
    items = [
        ("Delete the `loss` column (header and values, plus the two spaces after it) with a block delete: start on "
         "`l` of `loss`, Ctrl-V, `5j`, `6l`, `d`.", "fl<C-v>5j6ld"),
        ("Delete the `epoch` column and the two spaces after it (a 7-column block: Ctrl-V, `5j`, `6l`, `d`).",
         "<C-v>5j6ld"),
        ("Append ` | ok` to the end of every line of the table with one block append.", "<C-v>5j$A | ok<Esc>"),
    ]
    prompt, keys = pick(rng, items)
    return V(prompt, {"metrics.txt": text}, keys, require=["<C-v>"], show="target")


@template(M, "hard")
def visual_golf(rng):
    names = rng.sample(VARS, 4)
    text = "".join(f"{n:<8} = compute(x) # noqa\n" for n in names)
    items = [
        ("Delete the aligned ` # noqa` comments (and the space before them) from all four lines.",
         "f#h<C-v>3j$d"),
        ("Uppercase the first 8 columns of all four lines (the padded variable names) with one block command.",
         "<C-v>3j7lU"),
        ("Duplicate the whole block of four lines below itself with Visual Line mode.", "V3jyGp"),
    ]
    prompt, keys = pick(rng, items)
    return V(prompt + " Golf it.", {"calls.py": text}, keys, par=count_keys(keys), show="target")


# ================================================================= challenge

@template(M, "challenge")
def quote_list(rng):
    words = rng.sample(DATASETS, 5)
    text = "\n".join(words) + "\n"
    keys = "<C-v>4jI\"<Esc>gv$A\",<Esc>"
    return V("**Quote them.** Turn the five dataset names into quoted list items: each line `name` becomes "
             "`\"name\",`.\n\n**Constraints:** block insert for the opening quotes, then reselect (`gv`), extend to the "
             f"line ends (`$`) and block-append; par {count_keys(keys)} keys.", {"datasets.txt": text}, keys,
             require=["v_I", "v_A", "gv"], par=count_keys(keys), show="target")


@template(M, "challenge")
def comment_toggle(rng):
    text, f = train_py(rng)
    lines = text.split("\n")
    fl, _ = _find(lines, "for inputs, targets")
    keys = f"{fl}G<C-v>6jI# <Esc>"
    return V("**Disable the inner loop.** Comment out the whole inner `for` loop (its header and its 5-line body: "
             "7 lines) by putting `# ` in front of the code, at the indentation of the `for`.\n\n**Constraints:** "
             f"one block insert; par {count_keys(keys)} keys (jumping to the line included).",
             {"train.py": text}, keys, require=["<C-v>", "v_I"], par=count_keys(keys), show="diff")
