"""9.07 Registers & the clipboard: "a "A "0 "_ "+ ". ": "/ "% :reg, numbered registers, Ctrl-R in Insert mode and
the expression register."""

from ...core import pick
from ._common import DATASETS, MODELS, VARS, V, count_keys, drill, template, train_py

M = "9.07"


def _find(lines, needle, start=0):
    for i in range(start, len(lines)):
        j = lines[i].find(needle)
        if j >= 0:
            return i + 1, j + 1
    raise ValueError(needle)


# ================================================================= lite

@template(M, "lite")
def named_registers(rng):
    text, f = train_py(rng)
    lines = text.split("\n")
    files = {"train.py": text}
    el, _ = _find(lines, f"EPOCHS = {f['epochs']}")
    ll, lc = _find(lines, f"LR = {f['lr']}")
    items = [
        ("Yank the cursor's line into register `a` (`\"ayy`). Submit checks register a.", "\"ayy",
         {"cursor": (el, 1), "check": ("text", "regs"), "regs": "a", "require": ['"a']}),
        ("Yank the cursor's line into register `a`, then paste it from register a at the end of the file "
         "(`G\"ap`).", "\"ayyG\"ap", {"cursor": (el, 1), "require": ['"a']}),
        ("Register `a` already holds the EPOCHS line. APPEND the LR line (the cursor's) to it with `\"Ayy`.",
         "\"Ayy", {"cursor": (ll, 1), "prep": [(5, f"{el}G\"ayy")], "check": ("text", "regs"), "regs": "a",
                   "require": ['"A']}),
        (f"Yank the word `{f['opt']}` under the cursor into register `b` (`\"byiw`).", "\"byiw",
         {"cursor": (_find(lines, f['opt'])[0], _find(lines, f['opt'])[1] + 1), "check": ("text", "regs"),
          "regs": "b", "require": ['"b']}),
    ]
    return drill(rng, items, files)


@template(M, "lite")
def yank_vs_delete(rng):
    text, f = train_py(rng)
    lines = text.split("\n")
    files = {"train.py": text}
    el, _ = _find(lines, f"EPOCHS = {f['epochs']}")
    pl, _ = _find(lines, "print(f")
    bl, _ = _find(lines, f"BATCH_SIZE = {f['bs']}")
    items = [
        ("You yanked the EPOCHS line, then deleted a line (which replaced the unnamed register). Paste the line you "
         "YANKED below the cursor's line with `\"0p` (register 0 keeps the last yank).", "\"0p",
         {"prep": [(5, f"{el}Gyy{pl}Gdd{bl}G")], "require": ['"0']}),
        ("You yanked the EPOCHS line (it's in the unnamed register). Delete the `print(...)` line WITHOUT losing it "
         "(`\"_dd`: the black hole register), then paste the EPOCHS line below the cursor's line with `p`.",
         "\"_ddp", {"prep": [(5, f"{el}Gyy{pl}G")], "require": ['"_']}),
        ("Delete the cursor's line into the black hole (`\"_dd`), so the unnamed register still holds the BATCH_SIZE "
         "line you yanked; then paste that at the top of the file (`ggP`).", "\"_ddggP",
         {"prep": [(5, f"{bl}Gyy{pl}G")], "require": ['"_']}),
    ]
    return drill(rng, items, files)


# ================================================================= easy

@template(M, "easy")
def clipboard_and_list(rng):
    text, f = train_py(rng)
    lines = text.split("\n")
    files = {"train.py": text}
    el, _ = _find(lines, f"EPOCHS = {f['epochs']}")
    il, _ = _find(lines, "from torch import nn")
    reg = rng.choice("cdefg")
    items = [
        ("Copy the cursor's line to the system clipboard (`\"+yy`).", "\"+yy",
         {"cursor": (el, 1), "check": ("text", "regs"), "regs": "+", "require": ['"+']}),
        ("Some text was copied to the clipboard earlier. Paste it below the last line (`G\"+p`).", "G\"+p",
         {"prep": [(5, f"{il}G\"+yygg")], "require": ['"+']}),
        ("A named register holds the line `from torch import nn`. Find out which one with `:reg`, then paste it "
         "below line 1 (`\"xp` with the right letter).", f":reg<CR><CR>\"{reg}p",
         {"prep": [(5, f"{il}G\"{reg}yygg")], "require": [":reg"], "cursor": (1, 1)}),
    ]
    return drill(rng, items, files)


@template(M, "easy")
def special_registers(rng):
    text, f = train_py(rng)
    lines = text.split("\n")
    files = {"train.py": text}
    w = rng.choice(["warmup", "cosine", "linear"])
    items = [
        ("Insert the current file's name at the end of line 1: `$` then `\"%p` (register % is the file name).",
         "$\"%p", {"cursor": (1, 1), "require": ['"%']}),
        ("Add the last search pattern as a new last line: `Go`, Esc, then `\"/p`.", "Go<Esc>\"/p",
         {"prep": [(5, "/backward<CR>gg")], "require": ['"/']}),
        ("Put the text you last inserted at the end of the last line: `G$` then `\".p` (register . is the last "
         "inserted text; careful: `o` then Esc would replace it with nothing).", "G$\".p",
         {"prep": [(5, f"ggO# {w} schedule<Esc>u")], "require": ['".']}),
        ("You ran an Ex command a moment ago. Paste the command's text as a new last line (`Go`, Esc, `\":p`).",
         "Go<Esc>\":p", {"prep": [(5, ":set nonumber<CR>")], "require": ['":']}),
    ]
    return drill(rng, items, files)


# ================================================================= medium

@template(M, "medium")
def insert_registers(rng):
    text, f = train_py(rng)
    lines = text.split("\n")
    files = {"train.py": text}
    a, b = rng.choice((64, 128, 256)), rng.choice((4, 8, 16))
    hl, hc = _find(lines, "hub.load")
    items = [
        ("The model name was yanked into register `a`. Open a line below line 1 and type `MODEL = \"`, insert "
         "register a with Ctrl-R a, then type the closing `\"`.", "oMODEL = \"<C-r>a\"<Esc>",
         {"prep": [(5, f"{hl}G$F\"b\"ayiwgg")], "cursor": (1, 1), "require": ["i_<C-r>"]}),
        (f"Open a new last line and type `STEPS = `, then insert the value of {a}*{b} computed by Vim: Ctrl-R = "
         f"`{a}*{b}` Enter (the expression register).", f"GoSTEPS = <C-r>={a}*{b}<CR><Esc>",
         {"require": ["i_<C-r>="]}),
        ("Yank the word `train` from `def train` into register `t` (`\"tyiw`), then at the end of the file open "
         "a line and type `__all__ = [\"`, Ctrl-R t, `\"]`.", "\"tyiwGo__all__ = [\"<C-r>t\"]<Esc>",
         {"cursor": (_find(lines, "def train")[0], 5), "require": ["i_<C-r>", '"t']}),
    ]
    return drill(rng, items, files)


@template(M, "medium")
def three_snippets(rng):
    a, b, c = rng.sample(DATASETS, 3)
    text = f"load({a!r})\nload({b!r})\nload({c!r})\n\n# reversed:\n"
    keys = "\"ayyj\"byyj\"cyyG\"cp\"bp\"ap"
    return V("Yank the three `load(...)` lines into registers `a`, `b` and `c` (one each), then paste them under "
             "`# reversed:` in REVERSE order (c, b, a).", {"data.py": text}, keys,
             check=("text", "regs"), regs="abc", require=['"a', '"b', '"c'], show="target")


# ================================================================= hard

@template(M, "hard")
def numbered_registers(rng):
    names = rng.sample(VARS, 5)
    text = "".join(f"{n} = {i}\n" for i, n in enumerate(names))
    prep = [(5, "dd"), (4, "dd"), (3, "dd")]
    items = [
        ("Three lines were deleted one after the other (the newest is in register 1, then 2, then 3). Put the one "
         "deleted FIRST back at the top of the file (`gg\"3P`).", "gg\"3P", {"prep": prep}),
        ("Three lines were deleted one by one (newest in register 1). Put all three back at the top in their "
         "ORIGINAL order (paste above, newest first: `gg\"1P\"2P\"3P`).", "gg\"1P\"2P\"3P", {"prep": prep}),
    ]
    return drill(rng, items, {"vars.py": text}, show="target")


@template(M, "hard")
def registers_golf(rng):
    a, b = rng.sample(MODELS, 2)
    text = f"first = '{a}'\nmiddle = 0\nlast = '{b}'\n"
    items = [
        ("Move the first line to the end of the file.", "ddGp"),
        (f"Copy `'{a}'` (with its quotes) from line 1 over the `0` on line 2: yank it (`ya'`), then change the `0` "
         f"and insert register 0 with Ctrl-R 0.", "f'ya'jf0ciw<C-r>0<Esc>"),
    ]
    prompt, keys = pick(rng, items)
    return V(prompt + " Golf it.", {"swap.py": text}, keys, cursor=(1, 1), par=count_keys(keys), show="target")


# ================================================================= challenge

@template(M, "challenge")
def assemble(rng):
    a, b, c = rng.sample(MODELS, 3)
    text = (f"# candidates\nmodel_a = '{a}'\nmodel_b = '{b}'\nmodel_c = '{c}'\n\n# chosen (best first):\n")
    order = rng.sample(["a", "b", "c"], 3)
    keys = "".join(f"{2 + 'abc'.index(r)}G\"{r}yy" for r in "abc") + "G" + "".join(f"\"{r}p" for r in order)
    return V(f"**Ranking.** Under `# chosen (best first):` put copies of the three model lines in the order "
             f"{', '.join('model_' + r for r in order)}.\n\n**Constraints:** yank each line into a named register "
             f"(a, b, c) first, then paste from the registers; par {count_keys(keys)} keys.",
             {"models.py": text}, keys, require=['"a', '"b', '"c'], par=count_keys(keys), show="target")


@template(M, "challenge")
def computed_values(rng):
    bs, acc = rng.choice((32, 64)), rng.choice((2, 4, 8))
    steps = rng.choice((1000, 2000, 5000))
    text = f"batch_size = {bs}\ngrad_accum = {acc}\neffective_batch = TODO\nsteps = {steps}\nsamples_seen = TODO\n"
    keys = f"3GfTciw<C-r>={bs}*{acc}<CR><Esc>5GfTciw<C-r>={bs}*{acc}*{steps}<CR><Esc>"
    return V("**Fill in the numbers.** Replace each `TODO` with the value it stands for, computed by Vim's expression "
             "register (`Ctrl-R =` in Insert mode): `effective_batch` = batch_size × grad_accum, `samples_seen` = "
             "effective_batch × steps.\n\n**Constraints:** no typing the results yourself: the expression register.",
             {"numbers.py": text}, keys, require=["i_<C-r>="], show="target")
