"""9.02 Moving around.

Cursor drills: the highlighted cell is the target, the text must stay as it is, a motion is required and a par keeps
the solution short. The hard and challenge tiers are VimGolf-style navigation with the slow keys (h j k l, arrows)
forbidden.
"""

from ...core import pick
from ._common import VARS, V, count_keys, long_py, template, train_py

M = "9.02"
NAV = ("text", "cursor")
SLOW = ["hjkl", "arrows"]


def _nav(prompt, files, keys, cursor, req=(), forbid=(), par=None, check=NAV, **kw):
    return V(prompt, files, keys, cursor=cursor, check=check, require=list(req), forbid=list(forbid),
             par=par if par is not None else count_keys(keys), show=None, **kw)


def _find(lines, needle):
    """(line, col) of the first occurrence, 1-based."""
    for i, ln in enumerate(lines):
        j = ln.find(needle)
        if j >= 0:
            return i + 1, j + 1
    raise ValueError(needle)


# ================================================================= lite

@template(M, "lite")
def line_motions(rng):
    text, f = train_py(rng)
    lines = text.split("\n")
    n = len(lines) - 1
    files = {"train.py": text}
    loss_l, _ = _find(lines, "loss = loss_fn")
    opt_l, _ = _find(lines, "optimizer = torch")
    d = rng.randint(3, 6)
    target = rng.randint(12, n - 2)
    lines_ws = lines[:]
    lines_ws[loss_l - 1] += "    "
    files_ws = {"train.py": "\n".join(lines_ws)}
    items = [
        ("Jump to the END of the cursor's line with one key.", "$", (loss_l, 13), ["$"]),
        ("Jump to the very START of the line (column 1, before the indentation).", "0", (loss_l, 20), ["0"]),
        ("Jump to the first NON-BLANK character of the line (where the code starts, after the indentation).", "^",
         (loss_l, 30), ["^"]),
        (f"Move {d} lines down with a count (not by pressing j {d} times).", f"{d}j", (opt_l, 5), ["count", "j"]),
        (f"Move {d} lines up with a count.", f"{d}k", (opt_l + d, 9), ["count", "k"]),
        ("Jump to the first line of the file.", "gg", (target, 3), ["gg"]),
        ("Jump to the last line of the file.", "G", (4, 1), ["G"]),
        (f"Jump to line {target} with `{target}G`.", f"{target}G", (2, 1), ["G"]),
        (f"Jump to line {target} with the Ex command `:{target}`.", f":{target}<CR>", (2, 1), [":N"]),
        ("Move 4 characters to the right with a count (4l).", "4l", (loss_l, 13), ["count", "l"]),
        ("The cursor's line ends in a few invisible trailing spaces. Jump to its last NON-BLANK character with "
         "`g_` (`$` would land on the last space).", "g_", (loss_l, 13), ["g_"]),
    ]
    prompt, keys, cur, req = pick(rng, items)
    return _nav(prompt, files_ws if keys == "g_" else files, keys, cur, req)


@template(M, "lite")
def word_motions(rng):
    a, b = rng.sample(VARS, 2)
    line = f"    {a} = self.encoder(x[:, 0], {b}.mask, dim=-1)  # first token"
    files = {"encode.py": f"def encode(self, x, {b}):\n{line}\n    return {a}\n"}
    L = 2
    col = line.index("self") + 1
    items = [
        ("Move forward 3 words with `3w` (punctuation counts as its own word: `self`, `.`, `encoder` ...).", "3w",
         (L, col), ["w"]),
        ("Move back 2 words with `2b`.", "2b", (L, line.index("mask") + 1), ["b"]),
        ("Move to the END of the current word with `e`.", "e", (L, col), ["e"]),
        ("Move to the end of the PREVIOUS word with `ge`.", "ge", (L, line.index("encoder") + 1), ["ge"]),
        ("Move forward one WORD with `W` (WORDs are separated only by spaces, so `self.encoder(x[:,` is one WORD).",
         "W", (L, col), ["W"]),
        ("Move back one WORD with `B`.", "B", (L, line.index("mask") + 1), ["B"]),
        ("Move to the end of the WORD with `E`.", "E", (L, col), ["E"]),
        ("Move to the start of the second WORD after the cursor with `2W`.", "2W", (L, col), ["W"]),
    ]
    prompt, keys, cur, req = pick(rng, items)
    return _nav(prompt, files, keys, cur, req)


# ================================================================= easy

@template(M, "easy")
def screen_motions(rng):
    text, defs = long_py(rng, 13)
    files = {"metrics.py": text}
    mid = rng.randint(40, 60)
    items = [
        ("Scroll half a page DOWN (Ctrl-D): the cursor moves with the text.", "<C-d>", (5, 1), ["<C-d>"], NAV),
        ("Scroll half a page UP (Ctrl-U).", "<C-u>", (mid, 1), ["<C-u>"], NAV),
        ("Scroll a whole page DOWN (Ctrl-F).", "<C-f>", (3, 1), ["<C-f>"], NAV),
        ("Scroll a whole page UP (Ctrl-B).", "<C-b>", (mid + 20, 1), ["<C-b>"], NAV),
        ("Jump to the line at the TOP of the window (H).", "H", (mid, 1), ["H"], NAV),
        ("Jump to the line in the MIDDLE of the window (M).", "M", (mid, 1), ["M"], NAV),
        ("Jump to the line at the BOTTOM of the window (L).", "L", (mid, 1), ["L"], NAV),
        ("Scroll the window so the cursor's line is in the MIDDLE of the screen, without moving the cursor (zz).",
         "zz", (mid, 1), ["zz"], ("text", "cursor", "topline")),
        ("Scroll so the cursor's line is at the TOP of the window (zt).", "zt", (mid, 1), ["zt"],
         ("text", "cursor", "topline")),
        ("Scroll so the cursor's line is at the BOTTOM of the window (zb).", "zb", (mid, 1), ["zb"],
         ("text", "cursor", "topline")),
    ]
    prompt, keys, cur, req, check = pick(rng, items)
    return _nav(prompt, files, keys, cur, req, check=check)


@template(M, "easy")
def jumps(rng):
    text, f = train_py(rng)
    lines = text.split("\n")
    files = {"train.py": text}
    use_l, use_c = _find(lines, "total += loss")
    tot_l, tot_c = _find(lines, "total = 0.0")
    loss_l, loss_c = _find(lines, "loss.backward")
    far = len(lines) - 3
    items = [
        (f"Jump to line {far}, then jump back to where you were with Ctrl-O (the jump list).", f"{far}G<C-o>",
         (use_l, 13), ["G|:N", "<C-o>"]),
        (f"Jump to line {far}, back with Ctrl-O, then forward again with Tab (same as Ctrl-I).",
         f"{far}G<C-o><Tab>", (use_l, 13), ["<C-o>", "<Tab>"]),
        ("The cursor is on `total` in `total += loss.item()`. Go to where the variable is defined in this function "
         "(gd: go to local definition).", "gd", (use_l, use_c), ["gd"]),
        ("The cursor is on `loss` in `loss.backward()`. Jump to the NEXT occurrence of this word with `*`.", "*",
         (loss_l, loss_c), ["*"]),
        ("The cursor is on `loss` in `loss.backward()`. Jump to the PREVIOUS occurrence of this word with `#`.", "#",
         (loss_l, loss_c), ["#"]),
        ("The cursor is on `total` in `total = 0.0`. Jump to its next occurrence (`*`), then the one after (`n`).",
         "*n", (tot_l, tot_c), ["*"]),
    ]
    prompt, keys, cur, req = pick(rng, items)
    return _nav(prompt, files, keys, cur, req)


# ================================================================= medium

@template(M, "medium")
def find_char(rng):
    a, b, c = rng.sample(VARS, 3)
    line = (f"    out = model.generate({a}, max_new_tokens=64, top_k=50, temperature=0.8, "
            f"stop=[{b}, {c}])  # sample")
    files = {"sample.py": f"def sample(model, {a}, {b}, {c}):\n{line}\n    return out\n"}
    L = 2
    start = line.index("model") + 1
    gen = line.index("generate") + 1
    items = [
        ("Jump to the next `(` on the line with `f(`.", "f(", (L, start), ["f"], []),
        ("Jump to the THIRD comma after the cursor: `f,` then `;` repeats the find.", "f,;;", (L, start), ["f", ";"],
         []),
        ("Jump to the character just BEFORE the next `=` with `t=`.", "t=", (L, gen), ["t"], []),
        ("Jump BACK to the `(` before the cursor with `F(`.", "F(", (L, line.index("temperature") + 1), ["F"], []),
        ("Jump back to just AFTER the previous `=` with `T=`.", "T=", (L, line.index("0.8") + 3), ["T"], []),
        ("Find the second `_` after the cursor (`f_;`), then go back one with `,`.", "f_;,", (L, gen), ["f", ";", ","],
         []),
        ("Jump to the `[` of the stop list without h/l/w/e: one find command.", "f[", (L, gen), ["f"], SLOW),
    ]
    prompt, keys, cur, req, forbid = pick(rng, items)
    return _nav(prompt, files, keys, cur, req, forbid)


@template(M, "medium")
def structure(rng):
    text, f = train_py(rng)
    lines = text.split("\n")
    files = {"train.py": text}
    call_l = next(i + 1 for i, ln in enumerate(lines) if ln.strip().startswith("train(model, load_"))
    open_c = lines[call_l - 1].index("(") + 1
    pr_l = next(i + 1 for i, ln in enumerate(lines) if "print(f" in ln)
    items = [
        ("The cursor is on the `(` after `train`. Jump to its matching `)` with `%`.", "%", (call_l, open_c), ["%"]),
        ("Jump to the next blank line (the end of this paragraph of code) with `}`.", "}", (1, 1), ["}"]),
        ("Jump forward 3 paragraphs with a count: `3}`.", "3}", (1, 1), ["}"]),
        ("Jump back to the previous blank line with `{`.", "{", (pr_l, 9), ["{"]),
        ("The cursor is at the start of the `print(...)` line. Jump to the parenthesis that closes `print(` "
         "(f( then %).", "f(%", (pr_l, 1), ["%"]),
    ]
    prompt, keys, cur, req = pick(rng, items)
    return _nav(prompt, files, keys, cur, req)


# ================================================================= hard

@template(M, "hard")
def open_file_under_cursor(rng):
    helper = rng.choice(("utils", "metrics", "schedule", "augment"))
    fn = rng.choice(("warmup", "cosine", "clip", "smooth"))
    main = (f"# see {helper}.py for the helpers\nfrom {helper} import {fn}\n\n\ndef step(x):\n"
            f"    return {fn}(x)\n")
    other = f"def {fn}(x):\n    return x * 0.5\n\n\ndef unused(y):\n    return y\n"
    files = {"main.py": main, f"{helper}.py": other}
    col = main.index(helper) + 1
    items = [
        (f"The cursor is on `{helper}.py` in the first line. Open that file with `gf` (go to file).", "gf",
         ["gf"], ("text", "cursor")),
        (f"Open `{helper}.py` with `gf`, then jump back to main.py with Ctrl-O.", "gf<C-o>", ["gf", "<C-o>"],
         ("text", "cursor")),
        (f"Open `{helper}.py` with `gf`, jump to its last line (G), then go back to main.py with Ctrl-^ (the "
         f"alternate file).", "gfG<C-^>", ["gf", "<C-^>"], ("text", "cursor")),
    ]
    prompt, keys, req, check = pick(rng, items)
    return V(prompt, files, keys, cursor=(1, col), check=check, require=req, par=count_keys(keys) + 1, show=None,
             target=False)


@template(M, "hard")
def golf_targets(rng):
    text, f = train_py(rng)
    lines = text.split("\n")
    files = {"train.py": text}
    cands = []
    for i, ln in enumerate(lines):
        stripped = ln.lstrip()
        if len(stripped) < 15 or i < 3:
            continue
        fnb = len(ln) - len(stripped)
        for col in range(fnb + 1, len(ln)):
            c = ln[col]
            if c.isalnum() or c == " ":
                continue
            if ln.find(c, fnb + 1) == col:        # the first such character after the first non-blank
                cands.append((i, col, c))
    i, col, c = rng.choice(cands)
    start = (rng.choice([k for k in range(1, len(lines) - 1) if abs(k - (i + 1)) > 4]), 1)
    keys = f"{i + 1}Gf{c}"
    return _nav(f"Get the cursor to the highlighted `{c}` on line {i + 1} without h, j, k, l or the arrow keys. "
                f"(Jump to the line, then find the character.)", files, keys, start, forbid=SLOW)


# ================================================================= challenge

@template(M, "challenge")
def gauntlet(rng):
    text, defs = long_py(rng, 12)
    lines = text.split("\n")
    files = {"metrics.py": text}
    k = rng.randrange(len(defs) - 2) + 2
    L = defs[k] + 4                                    # `total += abs(x - y)`
    ln = lines[L - 1]
    col = ln.index("(") + 1                           # after abs
    target_c = rng.choice(["(", "-", ")"])
    col = ln.index(target_c)
    keys = f"{L}Gf{target_c}"
    return _nav(f"**Long jump.** Reach the highlighted `{target_c}` on line {L} from the top of the file.\n\n"
                f"**Constraints:** no h/j/k/l or arrows; par {count_keys(keys)} keys (a line number + G, then a "
                f"find).", files, keys, (1, 1), forbid=SLOW)


@template(M, "challenge")
def relative_numbers(rng):
    text, f = train_py(rng)
    lines = text.split("\n")
    files = {"train.py": text}
    start = rng.randint(12, 16)
    cand = [i + 1 for i, ln in enumerate(lines) if "=" in ln and abs(i + 1 - start) >= 3 and ln.strip()]
    L = rng.choice(cand)
    d = L - start
    keys = (f"{d}j" if d > 0 else f"{-d}k") + "f="
    return V(f"**Relative jump.** Line numbers are relative to the cursor (`:set relativenumber`, already on). Using "
             f"the number shown next to line `{lines[L - 1].strip()[:30]}…`, jump there with a count and `j`/`k`, then "
             f"land on its `=` with a find.\n\n**Constraints:** no G, gg, :N, / or ?; par {count_keys(keys)} keys.",
             files, keys, cursor=(start, 1), prep=[(0, ":set number relativenumber<CR>")],
             check=NAV, require=["count", "f"], forbid=["G", "gg", ":N", "/", "?", "<Down>", "<Up>"],
             par=count_keys(keys), show=None)
