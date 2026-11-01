"""9.04 The grammar: operators × motions × text objects (d c y > < = gu gU g~ gq with w e b $ G f t / and iw aw is
as ip ap i( a( i[ i{ a{ i< i" a" i' i` it at)."""

from ...core import pick
from ._common import MODELS, VARS, WORDS, V, count_keys, drill, notes_md, template, train_py

M = "9.04"


def _find(lines, needle, start=0):
    for i in range(start, len(lines)):
        j = lines[i].find(needle)
        if j >= 0:
            return i + 1, j + 1
    raise ValueError(needle)


# ================================================================= lite

@template(M, "lite")
def operator_motion(rng):
    text, f = train_py(rng)
    lines = text.split("\n")
    files = {"train.py": text}
    ol, oc = _find(lines, "optimizer = torch")
    ll, lc = _find(lines, "loss = loss_fn")
    pl, pc = _find(lines, "print(f")
    w = rng.choice(["net", "clf", "backbone"])
    items = [
        ("Delete the word under the cursor and the space after it (`dw`).", "dw", (ol, oc), ["dw"]),
        ("Delete from the cursor to the end of the line (`d$`, or its shortcut `D`).", "D", (ll, lc + 7), ["d$|D"]),
        ("Delete from the cursor's line to the end of the file (`dG`).", "dG", (pl + 2, 1), ["dG"]),
        (f"The cursor is on `model` in `model(inputs)`. Change the word to `{w}` (`cw`, type, Esc).", f"cw{w}<Esc>",
         (ll, lines[ll - 1].index("model(") + 1), ["cw"]),
        ("Change from the cursor to the end of the line into `None` (`c$` or `C`).", "CNone<Esc>",
         (ll, lc + 7), ["c$|C"]),
        ("Change the whole line (keeping its indentation) into `loss = criterion(outputs, targets)` (`cc`).",
         "ccloss = criterion(outputs, targets)<Esc>", (ll, lc + 3), ["cc"]),
        ("Delete 3 lines, starting at the cursor's, with ONE command (`3dd`).", "3dd", (ol, 1), ["dd", "count"]),
        ("Delete the next 3 words with ONE command (`d3w`).", "d3w", (ol, oc), ["dw", "count"]),
    ]
    prompt, keys, cur, req = pick(rng, items)
    return V(prompt, files, keys, cursor=cur, require=req, par=count_keys(keys) + 2 if len(keys) < 6 else None)


@template(M, "lite")
def text_objects(rng):
    a, b = rng.sample(VARS, 2)
    model = rng.choice(MODELS)
    lr = rng.choice(("1e-3", "3e-4", "0.01"))
    text = (f"def run({a}, {b}):  # quick sanity run\n    name = \"{model}\"\n    cfg = dict(lr={lr}, steps=[100, 200, 300])\n"
            f"    out = train({a}, {b}, lr={lr})\n    return {{\"name\": name, \"out\": out}}\n")
    files = {"run.py": text}
    lines = text.split("\n")
    nm = rng.choice(["vit_b16", "gpt2", "t5_small"])
    items = [
        (f"The cursor is in the middle of `{a}` on line 4. Change that whole word to `inputs` (`ciw`: from anywhere in "
         f"the word).", "ciwinputs<Esc>", (4, lines[3].index(a) + 2), ["ciw"]),
        ("Delete the word `quick` in the comment on line 1, together with the space after it (`daw`, from anywhere "
         "in the word).", "daw", (1, lines[0].index("quick") + 3), ["daw"]),
        (f"Change the text inside the quotes on line 2 to `{nm}` (`ci\"`, from anywhere on the line before it).",
         f"ci\"{nm}<Esc>", (2, 5), ['ci"']),
        ("Replace every argument of `train(...)` on line 4 with `batch` (`ci(` from inside the parentheses).",
         "ci(batch<Esc>", (4, lines[3].index("lr=") + 1), ["ci("]),
        ("Delete the call's parentheses AND their contents on line 4: `train(...)` becomes `train` (`da(`).", "da(",
         (4, lines[3].index("lr=") + 1), ["da("]),
        ("Delete the string on line 2, quotes included, so the line reads `name = ` (`da\"`).", "da\"",
         (2, 12), ['d', 'a"']),
        ("Delete the dict on line 5, braces included, so the line reads `return ` (`da{`).", "da{",
         (5, lines[4].index("name") + 1), ["d", "a{"]),
        ("Delete the contents of the list `[100, 200, 300]` (keep the brackets) with `di[`.", "di[",
         (3, lines[2].index("200") + 1), ["d", "i["]),
        ("Yank the text inside the braces on line 5 with `yi{` (Submit checks the unnamed register).", "yi{",
         (5, lines[4].index("name") + 1), ["yi{"]),
    ]
    prompt, keys, cur, req = pick(rng, items)
    check = ("text", "regs") if keys == "yi{" else ("text",)
    return V(prompt, files, keys, cursor=cur, require=req, check=check, regs='"' if keys == "yi{" else "")


# ================================================================= easy

@template(M, "easy")
def until_and_search(rng):
    a, b, c = rng.sample(VARS, 3)
    text = (f"logits = model({a}, {b}, mask={c}, dropout=0.1)\nprint(\"loss:\", loss.item(), \"acc:\", acc)\n"
            f"path = \"runs/exp1/checkpoints/best.pt\"\nresult = compute(x)  # TODO remove this debug call\n")
    files = {"snippet.py": text}
    lines = text.split("\n")
    items = [
        (f"On line 1, delete from the cursor up to (not including) the next comma: remove `{a}` (`dt,`).", "dt,",
         (1, lines[0].index(a) + 1), ["dt"]),
        ("On line 1, change from the cursor THROUGH the next `)` into `x)` (`cf)`).", "cf)x)<Esc>",
         (1, lines[0].index("(") + 2), ["cf"]),
        ("On line 4, delete from the cursor up to the word `TODO` (`d/TODO` then Enter).", "d/TODO<CR>",
         (4, lines[3].index("#") + 3), ["d/"]),
        ("On line 3, delete back to just after the previous `/` (`dT/`): the cursor is on the `.` of `best.pt`.",
         "dT/", (3, lines[2].index(".pt") + 1), ["d"]),
        ("On line 3, change the text up to the closing quote into `latest.pt` (`ct\"`): the cursor is on `b` of "
         "`best`.", "ct\"latest.pt<Esc>", (3, lines[2].index("best") + 1), ["c"]),
        ("On line 2, yank from the cursor up to (not including) the comma, then paste it at the end of the line "
         "(`yt,`, then `$p`).", "yt,$p", (2, lines[1].index("loss.") + 1), ["y"]),
    ]
    prompt, keys, cur, req = pick(rng, items)
    return V(prompt, files, keys, cursor=cur, require=req)


@template(M, "easy")
def case_ops(rng):
    name = rng.choice(("max_steps", "warmup_ratio", "num_heads", "hidden_dim"))
    text = (f"{name} = 8\nconst = 'gpt'\nprint('Training Started')\nlevel = 'debug'\nTITLE = 'Results Table'\n")
    files = {"case.py": text}
    items = [
        (f"Uppercase the word `{name}` (`gUiw`, from anywhere in it) to make it a constant.", "gUiw", (1, 3),
         ["gUiw"]),
        ("Lowercase the whole line 3 (`guu`).", "guu", (3, 1), ["gu"]),
        ("Toggle the case of the word `Training` (`g~iw`).", "g~iw", (3, 9), ["g~"]),
        ("Uppercase the whole line 4 (`gUU`).", "gUU", (4, 1), ["gU"]),
        ("Lowercase from the cursor to the end of line 5 (`gu$`).", "gu$", (5, 10), ["gu"]),
        ("Uppercase the next word only, from the cursor (`gUw`).", "gUw", (2, 10), ["gU"]),
    ]
    prompt, keys, cur, req = pick(rng, items)
    return V(prompt, files, keys, cursor=cur, require=req, par=count_keys(keys) + 1)


# ================================================================= medium

@template(M, "medium")
def indent_ops(rng):
    fn = rng.choice(("train", "evaluate", "fit"))
    js = (f"function {fn}(model, loader) {{\n    for (const batch of loader) {{\n  const out = model(batch);\n"
          f"          out.loss.backward();\n            step(model);\n    }}\n    return model;\n}}\n")
    py = (f"def {fn}(model, data):\n    for x, y in data:\n        pred = model(x)\n        loss = (pred - y) ** 2\n"
          f"    return model\n\n\ndef helper(z):\n    return z * 2\n")
    items = [
        ("The three lines inside the `for` block are badly indented. Fix them with `=i{` (the cursor is inside the "
         "block).", "=i{", {"files": {"train.js": js}, "cursor": (3, 3), "require": ["=i{"]}),
        ("Indent the `helper` function (lines 8-9, a paragraph) one level with `>ip`.", ">ip",
         {"files": {"model.py": py}, "cursor": (8, 1), "require": [">ip"]}),
        ("Dedent the body of the `for` loop (lines 3-4) one level with `<` and a motion (`<j`).", "<j",
         {"files": {"model.py": py}, "cursor": (3, 1), "require": ["<"]}),
        ("Indent everything from the cursor's line to the end of the file one level (`>G`).", ">G",
         {"files": {"model.py": py}, "cursor": (8, 1), "require": [">"]}),
        ("Re-indent the whole JavaScript file (`gg=G`).", "gg=G",
         {"files": {"train.js": js}, "cursor": (1, 1), "require": ["="]}),
    ]
    return drill(rng, items, {"model.py": py})


@template(M, "medium")
def more_objects(rng):
    text, ps = notes_md(rng, paras=3, per=3)
    tag = rng.choice(("b", "em", "code"))
    html = (f"<table>\n  <tr><td>loss</td><td><{tag}>0.412</{tag}></td></tr>\n  <tr><td>acc</td><td>91.3%</td></tr>\n"
            f"</table>\n")
    code = "x: List<Tensor> = [t.cpu() for t in batch]\ncmd = `nvidia-smi` + ' --query'\nname = 'run_01'\n"
    lines = text.split("\n")
    s2 = ps[0][1]
    c2 = lines[2].index(s2) + 3
    items = [
        ("On line 3, delete the SECOND sentence of the paragraph and the space after it (`das`, from inside it).",
         "das", {"files": {"notes.md": text}, "cursor": (3, c2), "require": ["d", "as"]}),
        ("Change the second sentence on line 3 into `TBD.` (`cis`), keeping the spaces around it.", "cisTBD.<Esc>",
         {"files": {"notes.md": text}, "cursor": (3, c2), "require": ["c", "is"]}),
        ("Delete the whole second paragraph (line 5) and the blank line after it (`dap`).", "dap",
         {"files": {"notes.md": text}, "cursor": (5, 4), "require": ["dap"]}),
        (f"Change the value inside the `<{tag}>` tag to `0.398` (`cit`).", "cit0.398<Esc>",
         {"files": {"table.html": html}, "cursor": (2, html.split(chr(10))[1].index("0.412") + 2),
          "require": ["c", "it"]}),
        ("Delete the whole second `<td>...</td>` cell on line 3, tags included (`dat`, from inside it).", "dat",
         {"files": {"table.html": html}, "cursor": (3, html.split(chr(10))[2].index("91") + 1),
          "require": ["d", "at"]}),
        ("Change the type inside `List<...>` to `int` (`ci<`).", "ci<int<Esc>",
         {"files": {"types.py": code}, "cursor": (1, 9), "require": ["c", "i<"]}),
        ("Change the command between the backticks on line 2 to `ls` (`` ci` ``).", "ci`ls<Esc>",
         {"files": {"types.py": code}, "cursor": (2, 9), "require": ["c", "i`"]}),
        ("Delete the string `'run_01'` on line 3, quotes included (`da'`).", "da'",
         {"files": {"types.py": code}, "cursor": (3, 10), "require": ["d"]}),
        ("Change the text inside the single quotes on line 3 to `run_02` (`ci'`).", "ci'run_02<Esc>",
         {"files": {"types.py": code}, "cursor": (3, 1), "require": ["c", "i'"]}),
    ]
    return drill(rng, items, {"notes.md": text})


# ================================================================= hard

@template(M, "hard")
def wrap_text(rng):
    text, ps = notes_md(rng, paras=2, per=4)
    tw = rng.choice((40, 50, 60))
    items = [
        (f"`textwidth` is set to {tw}. Re-wrap the first paragraph (line 3) so no line is longer than {tw} "
         f"characters (`gqap`, or `gqip`).", "gqap", (3, 1), ["gq"]),
        (f"`textwidth` is {tw}. Re-wrap just line 5 (`gqq`).", "gqq", (5, 1), ["gq"]),
        (f"`textwidth` is {tw}. Re-wrap everything from the cursor to the end of the file (`gqG`).", "gqG", (3, 1),
         ["gq"]),
    ]
    prompt, keys, cur, req = pick(rng, items)
    return V(prompt, {"notes.md": text}, keys, cursor=cur, require=req, prep=[(0, f":set tw={tw}<CR>")],
             show="target")


@template(M, "hard")
def refactor_golf(rng):
    a, b = rng.sample(VARS, 2)
    model = rng.choice(MODELS)
    text = (f"result = trainer.fit(model, {a}, {b}, epochs=10, verbose=True)\n"
            f"config = {{\"name\": \"{model}\", \"layers\": [64, 128, 256]}}\n"
            f"if not os.path.exists(path) and retries > 3 and verbose:\n    raise RuntimeError(\"give up\")\n")
    lines = text.split("\n")
    items = [
        ("Replace all the arguments of `fit(...)` with `loader` (the cursor is on `r` of `result`).",
         "f(ci(loader<Esc>", (1, 1)),
        ("Delete `, verbose=True` from the call on line 1 (the cursor is at the start of the line).",
         "f,;;;dt)", (1, 1)),
        ("Change the model name inside its quotes on line 2 into `bert` (cursor at the start of the line).",
         "3f\"ci\"bert<Esc>", (2, 1)),
        ("Delete the list `[64, 128, 256]` on line 2, brackets included, leaving `\"layers\": }` (cursor at the "
         "start of the line).", "f[da[", (2, 1)),
        ("Change the error message on line 4 into `retry` (cursor at the start of the line).", "ci\"retry<Esc>",
         (4, 1)),
    ]
    prompt, keys, cur = pick(rng, items)
    return V(prompt + " Golf it.", {"refactor.py": text}, keys, cursor=cur, par=count_keys(keys))


# ================================================================= challenge

@template(M, "challenge")
def grammar_gauntlet(rng):
    text, f = train_py(rng)
    lines = text.split("\n")
    tl, _ = _find(lines, "def train")
    keys = "ci(net, loader<Esc>jf(ci(net.parameters(), lr=LR<Esc>"
    return V("**Rename the model.** In `def train(...)`, replace the parameter list with `net, loader`, and on the "
             "next line make the optimizer read `torch.optim...(net.parameters(), lr=LR)`.\n\n**Constraints:** text "
             f"objects for both changes; par {count_keys(keys)} keys (the cursor starts on `def`).",
             {"train.py": text}, keys, cursor=(tl, 1), require=["ci("], par=count_keys(keys))


@template(M, "challenge")
def rename_with_dot(rng):
    old, new = rng.sample(["out", "res", "acc", "tmp"], 2)
    text = (f"def evaluate(model, data):\n    {old} = 0\n    for x, y in data:\n        {old} += (model(x) == y)\n"
            f"    print({old})\n    return {old} / len(data)\n")
    keys = f"*ciw{new}<Esc>n.n.n."
    return V(f"**Rename.** Rename the variable `{old}` to `{new}` everywhere in the function (4 places). The cursor "
             f"is on the first one.\n\n**Constraints:** `*` (or `/`) to find the next one, `ciw` once, then `n` and "
             f"`.` for the rest; par {count_keys(keys)} keys.", {"evaluate.py": text}, keys,
             cursor=(2, 5), require=["ciw", ".", "n|*"], par=count_keys(keys), show="target")
