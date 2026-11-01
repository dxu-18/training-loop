"""9.03 Inserting & basic editing: i a I A o O, x X r s S, dd yy p P, undo, the dot command, J ~ >> ==, Ctrl-A,
and the Insert-mode keys (Ctrl-W, Ctrl-U, completion, Ctrl-R)."""

from ...core import pick
from ._common import MODELS, VARS, WORDS, V, count_keys, drill, template, train_py

M = "9.03"


def _find(lines, needle, start=0):
    for i in range(start, len(lines)):
        j = lines[i].find(needle)
        if j >= 0:
            return i + 1, j + 1
    raise ValueError(needle)


# ================================================================= lite

@template(M, "lite")
def insert_commands(rng):
    text, f = train_py(rng)
    lines = text.split("\n")
    files = {"train.py": text}
    w = rng.choice(["fast", "debug", "full", "tiny"])
    tl, tc = _find(lines, "def train")
    ll, lc = _find(lines, "loss.backward()")
    el, _ = _find(lines, f"EPOCHS = {f['epochs']}")
    items = [
        (f"The cursor is on `t` of `train`. Insert `{w}_` before it (`i`).", f"i{w}_<Esc>",
         {"cursor": (tl, tc + 4), "require": ["i"]}),
        ("The cursor is on the last letter of `train`. Append `_one_epoch` right after it (`a` inserts AFTER the "
         "cursor).", "a_one_epoch<Esc>", {"cursor": (tl, tc + 8), "require": ["a"]}),
        ("Comment out the cursor's line: put `# ` in front of the code, after the indentation (`I` inserts at the "
         "first non-blank).", "I# <Esc>", {"cursor": (ll, lc + 5), "require": ["I"]}),
        ("Append `  # TODO: try more` at the end of the cursor's line (`A`).", "A  # TODO: try more<Esc>",
         {"cursor": (el, 1), "require": ["A"]}),
        ("Open a new line BELOW the cursor's and type `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` "
         "(`o`: Vim indents it for you).", "otorch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)<Esc>",
         {"cursor": (ll, lc), "require": ["o"]}),
        ("Open a new line ABOVE the cursor's and type `WARMUP = 500` (`O`).", "OWARMUP = 500<Esc>",
         {"cursor": (el, 1), "require": ["O"]}),
    ]
    return drill(rng, items, files)


@template(M, "lite")
def delete_replace(rng):
    text, f = train_py(rng)
    lines = text.split("\n")
    ol, oc = _find(lines, "optimizer = torch")
    typo = lines[ol - 1].replace("optimizer =", "optimmizer =")
    lines_t = lines[:ol - 1] + [typo] + lines[ol:]
    text_t = "\n".join(lines_t)
    bl, bc = _find(lines, f"BATCH_SIZE = {f['bs']}")
    pl, pc = _find(lines, "print(f")
    ll, lc = _find(lines, "loss.backward()")
    files = {"train.py": text}
    items = [
        ("There's a typo: `optimmizer`. The cursor is on the extra `m`: delete it with `x`.", "x",
         {"files": {"train.py": text_t}, "cursor": (ol, oc + 5), "require": ["x"]}),
        ("There's a typo: `optimmizer`. The cursor is on the `i` after the double m: delete the character BEFORE the "
         "cursor with `X`.", "X", {"files": {"train.py": text_t}, "cursor": (ol, oc + 6), "require": ["X"]}),
        (f"The cursor is on the first digit of `{f['bs']}`. Replace that one character with `9` (`r9`).", "r9",
         {"cursor": (bl, bc + 13), "require": ["r"]}),
        ("The cursor is on the `B` of `BATCH_SIZE`. Replace that single character with `MINI_B` (`s` deletes the "
         "character and starts Insert mode).", "sMINI_B<Esc>", {"cursor": (bl, bc), "require": ["s"]}),
        ("Replace the whole cursor line with `pass`, keeping its indentation (`S`).", "Spass<Esc>",
         {"cursor": (ll, lc + 3), "require": ["S"]}),
        ("Delete the cursor's line (`dd`).", "dd", {"cursor": (pl, pc), "require": ["dd"]}),
        ("Duplicate the cursor's line: yank it (`yy`) and put the copy below (`p`).", "yyp",
         {"cursor": (ll, lc), "require": ["yy", "p"]}),
        ("Copy the cursor's line ABOVE itself: `yy`, then `P` puts before.", "yyP",
         {"cursor": (bl, 1), "require": ["yy", "P"]}),
    ]
    return drill(rng, items, files)


# ================================================================= easy

@template(M, "easy")
def undo_redo(rng):
    text, f = train_py(rng)
    lines = text.split("\n")
    el, _ = _find(lines, f"EPOCHS = {f['epochs']}")
    ll, _ = _find(lines, "loss.backward()")
    files = {"train.py": text}
    prep3 = [(300, f"{el}Gdd"), (200, f"{ll - 1}GA  # x<Esc>"), (100, "ggOimport os<Esc>")]
    items = [
        ("Three changes were made (a line deleted, a comment added, an import added). Undo the LAST one (`u`).", "u",
         {"prep": prep3, "require": ["u"]}),
        ("Three changes were made. Undo all three with a count (`3u`).", "3u", {"prep": prep3, "require": ["u"]}),
        ("Three changes were made, then two were undone. Redo ONE of them (Ctrl-R).", "<C-r>",
         {"prep": prep3 + [(50, "uu")], "require": ["<C-r>"]}),
        ("Several edits were made to the `BATCH_SIZE` line. Undo all of them at once with `U` (undo line).", "U",
         {"prep": [(100, f"{el + 2}G$ciw7<Esc>0iMY_<Esc>A  # tuned<Esc>")], "require": ["U"]}),
        ("Two changes were made. Undo both, then redo the first one again.", "uu<C-r>",
         {"prep": prep3[:2], "require": ["u", "<C-r>"]}),
    ]
    return drill(rng, items, files)


@template(M, "easy")
def dot_repeat(rng):
    names = rng.sample(VARS, 5)
    body = "".join(f"{n} = compute_{n}(batch)\n" for n in names)
    log = "".join(f"step {i} done\n" for i in range(1, 6))
    files = {"step.py": "def step(batch):\n" + "".join("    " + ln + "\n" for ln in body.splitlines()) + "    return loss\n"}
    items = [
        ("Append `  # cached` to the end of each of the 5 assignment lines: do it once on the first (`A`...Esc), then "
         "`j` and `.` for each of the others.", "jA  # cached<Esc>j.j.j.j.",
         {"require": ["."], "par": count_keys("jA  # cached<Esc>j.j.j.j.")}),
        ("Put `self.` in front of the variable name on each of the 5 assignment lines (`I` on the first, then "
         "`j.` for the others).", "jIself.<Esc>j.j.j.j.",
         {"require": ["."], "par": count_keys("jIself.<Esc>j.j.j.j.")}),
        ("Indent the 5 assignment lines one more level: `>>` on the first, then `j.` four times.", "j>>j.j.j.j.",
         {"require": [">>", "."]}),
        ("Delete lines 2-4 with one `dd` and two dots.", "jdd..", {"require": ["dd", "."]}),
        ("The log has 5 lines. Delete the word `step ` at the start of each line with `dw` once and `j.` for the rest.",
         "dwj.j.j.j.", {"files": {"run.log": log}, "require": ["dw", "."]}),
    ]
    return drill(rng, items, files)


# ================================================================= medium

@template(M, "medium")
def join_case_indent(rng):
    model = rng.choice(MODELS)
    text = (f"def build(cfg):\n    name = (\"{model}\"\n        + \"_v2\")\nurl = \"https://example.com/\"\n"
            f"\"checkpoints/{model}.pt\"\nopt = \"sgd\"\n    if cfg.debug:\n    print(cfg)\n  return name\n")
    files = {"build.py": text}
    items = [
        ("Join line 3 onto line 2 (`J` on line 2: it adds a space and removes the indentation).", "jJ",
         {"cursor": (1, 1), "require": ["J"]}),
        ("The URL string is split over lines 4 and 5. Join them WITHOUT inserting a space (`gJ` on line 4).",
         "gJ", {"cursor": (4, 1), "require": ["gJ"]}),
        ("Uppercase `sgd` on line 6 with `~` and a count (the cursor is on the `s`).", "3~",
         {"cursor": (6, 8), "require": ["~"]}),
        ("Indent line 8 (`print(cfg)`) one level with `>>`.", ">>", {"cursor": (8, 5), "require": [">>"]}),
        ("Dedent line 7 (`if cfg.debug:`) one level with `<<`.", "<<", {"cursor": (7, 5), "require": ["<<"]}),
        ("Let Vim fix the indentation of line 9 (`return name`) with `==`.", "==", {"cursor": (9, 3), "require": ["=="]}),
        ("Indent lines 7 and 8 together with `2>>`.", "2>>", {"cursor": (7, 1), "require": [">>", "count"]}),
    ]
    return drill(rng, items, files)


@template(M, "medium")
def numbers(rng):
    ep = rng.choice((10, 20, 25))
    add = rng.choice((5, 10, 20))
    warm = rng.choice((500, 1000, 2000))
    sub = rng.choice((100, 250))
    bs = rng.choice((16, 32, 64))
    ver = (1, rng.randint(0, 9), rng.randint(0, 8))
    text = (f"epochs = {ep}\nwarmup_steps = {warm}\nbatch_size = {bs}\nversion = \"{ver[0]}.{ver[1]}.{ver[2]}\"\n"
            f"mask = 0x0f\nseed = 7\n")
    files = {"config.py": text}
    no_typing = ["insert", "r", "R", "s", "x"]
    items = [
        (f"Change `epochs = {ep}` to `epochs = {ep + add}` with Ctrl-A and a count (the cursor can be anywhere before "
         f"the number: Ctrl-A finds it).", f"{add}<C-a>", {"cursor": (1, 1), "require": ["<C-a>"], "forbid": no_typing}),
        (f"Lower `warmup_steps` from {warm} to {warm - sub} with Ctrl-X.", f"{sub}<C-x>",
         {"cursor": (2, 1), "require": ["<C-x>"], "forbid": no_typing}),
        (f"Double `batch_size` ({bs} → {bs * 2}) with one Ctrl-A command.", f"{bs}<C-a>",
         {"cursor": (3, 1), "require": ["<C-a>"], "forbid": no_typing}),
        (f"Bump the PATCH number of the version ({'.'.join(map(str, ver))} → {ver[0]}.{ver[1]}.{ver[2] + 1}). Careful: "
         f"Ctrl-A from the start of the line would bump the first number it finds.", "2f.<C-a>",
         {"cursor": (4, 1), "require": ["<C-a>"], "forbid": no_typing}),
        ("Increment the hex mask (0x0f → 0x10): Ctrl-A knows hex.", "<C-a>",
         {"cursor": (5, 1), "require": ["<C-a>"], "forbid": no_typing}),
        ("Set `seed` to 42 using Ctrl-A (count + Ctrl-A).", "35<C-a>",
         {"cursor": (6, 1), "require": ["<C-a>"], "forbid": no_typing}),
    ]
    return drill(rng, items, files)


# ================================================================= hard

@template(M, "hard")
def insert_mode_keys(rng):
    text, f = train_py(rng)
    lines = text.split("\n")
    sched = rng.choice(["cosine_warmup_scheduler", "linear_decay_schedule", "polynomial_lr_policy"])
    text = text.replace("\n\n\ndef train", f"\n{sched} = None\n\n\ndef train", 1)
    lines = text.split("\n")
    sl, _ = _find(lines, f"{sched} = None")
    ll, lc = _find(lines, "loss.backward()")
    files = {"train.py": text}
    pre = sched[:3]
    items = [
        ("Open a line below the last line and type `learning rate warmup`; then, still in Insert mode, delete "
         "`warmup` with Ctrl-W and type `decay`, so the line reads `learning rate decay`.",
         "Golearning rate warmup<C-w>decay<Esc>", {"require": ["i_<C-w>"]}),
        ("Open a line below `loss.backward()` and type `print(loss)`; then (still in Insert mode) wipe what you typed "
         "with Ctrl-U and type `optimizer.zero_grad()` instead.", "oprint(loss)<C-u>optimizer.zero_grad()<Esc>",
         {"cursor": (ll, lc), "require": ["i_<C-u>"]}),
        (f"Open a line below line {sl} and type `{pre}`, then complete the long name `{sched}` with Ctrl-N "
         f"(Insert-mode completion from words in the file), then type ` = 1`.",
         f"o{pre}<C-n> = 1<Esc>", {"cursor": (sl, 1), "require": ["i_<C-n>"]}),
        (f"At the end of the file, open a new line and type `{pre}`, then complete it with Ctrl-P (it searches "
         f"backwards), then type `.step()`.", f"Go{pre}<C-p>.step()<Esc>", {"require": ["i_<C-p>"]}),
        ("A word was yanked into register `a`. Open a line below the first line and type `MODEL = \"`, then insert "
         "the register with Ctrl-R a, then type `\"`.", "oMODEL = \"<C-r>a\"<Esc>",
         {"prep": [(10, f"{_find(lines, 'hub.load')[0]}G$F\"b\"ayiwgg")], "require": ["i_<C-r>"]}),
        ("On the line `total = 0.0`, go to its end in Insert mode (`A`) and indent the line one more level with "
         "Ctrl-T, then Esc.", "A<C-t><Esc>", {"cursor": (_find(lines, "total = 0.0")[0], 1), "require": ["i_<C-t>"]}),
        ("On the line `total = 0.0`, go to its end in Insert mode (`A`) and DEDENT it one level with Ctrl-D, then "
         "Esc.", "A<C-d><Esc>", {"cursor": (_find(lines, "total = 0.0")[0], 1), "require": ["i_<C-d>"]}),
    ]
    return drill(rng, items, files)


@template(M, "hard")
def golf_edits(rng):
    a, b = rng.sample(WORDS, 2)
    text = (f"first = \"{a}\"\nsecond = \"{b}\"\nprint(teh_result)\nx = 1\ny = 2\nz = 3\nw = 4\n"
            f"last = True\n")
    files = {"golf.py": text}
    items = [
        ("Duplicate line 1 three times (so there are four copies).", "yy3p", (1, 1)),
        ("Swap lines 1 and 2.", "ddp", (1, 1)),
        ("Fix the typo `teh` → `the` (the cursor is on the `e`).", "xp", (3, 8)),
        ("Delete lines 4-7.", "4dd", (4, 1)),
        ("Move the last line to the top of the file.", "ddggP", (8, 1)),
        ("Join lines 4-7 into one line.", "4J", (4, 1)),
    ]
    prompt, keys, cur = pick(rng, items)
    return V(prompt + " Golf it.", files, keys, cursor=cur, par=count_keys(keys), show="diff")


# ================================================================= challenge

@template(M, "challenge")
def refactor_combo(rng):
    text, f = train_py(rng)
    lines = text.split("\n")
    el, _ = _find(lines, f"EPOCHS = {f['epochs']}")
    add = rng.choice((5, 10))
    keys = f"{add}<C-a>yyp$ciwFalse<Esc>0cwDEBUG<Esc>"
    files = {"train.py": text}
    return V(f"**Config bump.** The cursor is on `EPOCHS = {f['epochs']}`. Raise it to {f['epochs'] + add} without "
             f"typing the number, then add a line `DEBUG = False` right below it by copying the line and changing "
             f"the copy.\n\n**Constraints:** Ctrl-A for the number; `yy`/`p` for the copy; par {count_keys(keys)}.",
             files, keys, cursor=(el, 1), require=["<C-a>", "yy", "p"], par=count_keys(keys))


@template(M, "challenge")
def dot_golf(rng):
    names = rng.sample(VARS, 6)
    text = "".join(f"{n} = {n}.detach()\n" for n in names)
    keys = "A.cpu()<Esc>j.j.j.j.j."
    return V("**Move to CPU.** Append `.cpu()` to the end of all six lines.\n\n**Constraints:** one change, then "
             f"repeat it; par {count_keys(keys)} keys.", {"tensors.py": text}, keys, require=["."],
             par=count_keys(keys), show="target")
