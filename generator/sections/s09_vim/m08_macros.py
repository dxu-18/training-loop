"""9.08 Macros & the dot command: q{reg} … q, @{reg} @@ counts, :normal @a on ranges, editing a macro as text, and
the cgn + . workflow."""

from ...core import pick
from ._common import DATASETS, MODELS, NAMES, VARS, V, count_keys, drill, template

M = "9.08"


# ================================================================= lite

@template(M, "lite")
def record_play(rng):
    names = rng.sample(VARS, 4)
    text = "".join(f"{n} = compute_{n}()\n" for n in names)
    items = [
        ("Record a macro in register `a` that appends `;` to the line and moves down (`qa`, `A;` Esc, `j`, `q`), "
         "then play it on the next 3 lines with a count (`3@a`).", "qaA;<Esc>jq3@a", ["q", "@"]),
        ("Record a macro in `a` that uppercases the first word and moves down (`qa`, `gUiw`, `j`, `q`); then play "
         "it once (`@a`) and repeat it twice more with `@@`.", "qagUiwjq@a@@@@", ["q", "@", "@@"]),
        ("Record a macro in `q` that puts `self.` in front of the line and moves down, then run it on the other "
         "three lines.", "qqIself.<Esc>jq3@q", ["q", "@"]),
    ]
    prompt, keys, req = pick(rng, items)
    return V(prompt, {"calls.py": text}, keys, require=req, show="target")


@template(M, "lite")
def replay_counts(rng):
    n = rng.randint(6, 9)
    words = rng.sample(MODELS, n) if n <= len(MODELS) else MODELS[:n]
    text = "".join(f"{w}\n" for w in words)
    items = [
        (f"Wrap every line in `load(\"...\")`: record it once on line 1 (`qa`, `Iload(\"` Esc, `A\")` Esc, `j`, `q`), "
         f"then play it on the remaining {n - 1} lines with ONE count.", f"qaIload(\"<Esc>A\")<Esc>jq{n - 1}@a",
         ["q", "@", "count"]),
        ("Record a macro that deletes the line's first character and moves down; play it on all the other lines "
         "with a big count (`99@a`: the macro stops by itself when `j` fails on the last line).",
         "qaxjq99@a", ["q", "@"]),
    ]
    prompt, keys, req = pick(rng, items)
    return V(prompt, {"models.txt": text}, keys, require=req, show="target")


# ================================================================= easy

@template(M, "easy")
def recipe(rng):
    n = rng.randint(5, 8)
    rows = [(rng.choice(NAMES), rng.randint(60, 99)) for _ in range(n)]
    text = "".join(f"{a} {b}\n" for a, b in rows)
    items = [
        ("Turn each `name score` line into `name: score/100`. Recipe: on line 1 `qa`, make the edit, go to the next "
         "line's start (`j0`), `q`, then `99@a`.", "qaf i:<Esc>A/100<Esc>j0q99@a",
         ["q", "@"]),
        ("Turn each `name score` line into `print(\"name\", score)` with one recorded macro, played on the rest.",
         "qaIprint(\"<Esc>f i\",<Esc>A)<Esc>j0q99@a", ["q", "@"]),
    ]
    prompt, keys, req = pick(rng, items)
    return V(prompt, {"scores.txt": text}, keys, require=req, show="target")


@template(M, "easy")
def cgn_dot(rng):
    old, new = rng.sample(["tmp", "val", "res", "buf"], 2)
    text = (f"def step(x):\n    {old} = x * 2\n    {old} = {old} + 1\n    print({old})\n    return {old}\n")
    k = text.count(old)
    keys = f"/{old}<CR>cgn{new}<Esc>" + "." * (k - 1)
    items = [
        (f"Rename every `{old}` to `{new}` with the dot workflow: search for it (`/{old}`), change the match with "
         f"`cgn{new}` Esc, then press `.` for each of the remaining {k - 1} matches.", keys, ["cgn", "."]),
        (f"Rename only the FIRST three `{old}` to `{new}` (`/{old}`, `cgn{new}` Esc, then `.` twice).",
         f"/{old}<CR>cgn{new}<Esc>..", ["cgn", "."]),
    ]
    prompt, keys, req = pick(rng, items)
    return V(prompt, {"step.py": text}, keys, require=req, show="target")


# ================================================================= medium

@template(M, "medium")
def normal_range(rng):
    names = rng.sample(VARS, 8)
    text = "# settings\n" + "".join(f"{n} = None\n" for n in names) + "# end\n"
    items = [
        ("Register `a` holds a macro that appends `  # set me` to a line. Run it on lines 2-9 (the eight "
         "assignments) with `:2,9normal @a`.", ":2,9normal @a<CR>", ["@", ":normal"]),
        ("Register `a` holds a macro that appends `  # set me`. Select the eight assignment lines (`2GV9G`) and run the macro on the selection (`:normal @a`; Vim pre-fills `:'<,'>`).",
         "2GV9G:normal @a<CR>", ["@", ":normal"]),
    ]
    prompt, keys, req = pick(rng, items)
    return V(prompt, {"settings.py": text}, keys, require=req, prep=[(5, "qaA  # set me<Esc>qu")], show="target")


@template(M, "medium")
def edit_macro(rng):
    names = rng.sample(VARS, 5)
    text = "".join(f"{n} = 0\n" for n in names)
    keys = "Go<Esc>\"apF;r,0\"ay$ddgg5@a"
    return V("Register `a` holds a macro that was recorded wrong: it appends `;` (it should append `,`). Fix the "
             "MACRO, not the lines: paste it on a new last line (`Go` Esc `\"ap`), change the `;` to `,` (after `p` the cursor is at the END of the pasted text), yank the "
             "line back into `a` (`0\"ay$`), delete that scratch line, then run the fixed macro on all five lines.",
             {"zeros.py": text}, keys, prep=[(5, "qaA;<Esc>jqu")], check=("text", "regs"), regs="a",
             require=['"a', "@"], show="target", expect=[f"{n} = 0," for n in names])


# ================================================================= hard

@template(M, "hard")
def dict_literal(rng):
    keys_vals = [("lr", rng.choice(("0.01", "3e-4"))), ("epochs", str(rng.randint(5, 50))),
                 ("batch_size", str(rng.choice((32, 64)))), ("dropout", rng.choice(("0.1", "0.2"))),
                 ("warmup", str(rng.choice((100, 500)))), ("seed", str(rng.randint(0, 9))),
                 ("shuffle", "True"), ("amp", rng.choice(("True", "False")))]
    rng.shuffle(keys_vals)
    text = "".join(f"{k}={v}\n" for k, v in keys_vals)
    n = len(keys_vals)
    keys = f"qaI    \"<Esc>f=i\"<Esc>lr:a <Esc>A,<Esc>jq{n - 1}@aggOconfig = {{<Esc>Go}}<Esc>"
    return V("Turn the `key=value` lines into a Python dict literal:\n\n```\nconfig = {\n    \"lr\": 0.01,\n    ...\n}\n"
             "```\n\nRecord the change for one line as a macro and play it on the others; then add the first and "
             "last lines.", {"config.txt": text}, keys, require=["q", "@"], show="target")


@template(M, "hard")
def quoted_list(rng):
    words = rng.sample(DATASETS, rng.randint(5, 7))
    text = "".join(f"{w}\n" for w in words)
    keys = f"qaI\"<Esc>A\",<Esc>jq{len(words) - 1}@aggVGJ$x"
    return V("Turn the column of names into ONE line: a quoted, comma-separated list (`\"a\", \"b\", \"c\"`, no "
             "trailing comma). A macro for the quotes and commas, then join the lines (`VGJ` from the top) and fix "
             "the end.", {"names.txt": text}, keys, require=["q", "@"], show="target")


# ================================================================= challenge

@template(M, "challenge")
def csv_to_markdown(rng):
    rows = [("model", "acc", "params")] + [(m, f"{rng.uniform(70, 95):.1f}", f"{rng.randint(5, 300)}M")
                                          for m in rng.sample(MODELS, 4)]
    text = "".join(",".join(r) + "\n" for r in rows)
    keys = "qaI| <Esc>:s/,/ | /g<CR>A |<Esc>jq4@aggo| --- | --- | --- |<Esc>"
    return V("**CSV to Markdown.** Turn the CSV into a Markdown table: each row `a,b,c` becomes `| a | b | c |`, and "
             "a separator row `| --- | --- | --- |` goes right under the header.\n\n**Constraints:** record the row "
             "conversion once (a `:s` inside a macro is fine) and replay it.", {"results.csv": text}, keys,
             require=["q", "@"], show="target")


@template(M, "challenge")
def numbering(rng):
    items = rng.sample(["load data", "build model", "train", "evaluate", "save checkpoint", "plot curves",
                        "write report", "tag release", "clean up", "profile", "tune", "deploy"], rng.randint(8, 12))
    text = "1. " + items[0] + "\n" + "".join(f"0. {x}\n" for x in items[1:])
    keys = f"qayiwj0viwp<C-a>q{len(items) - 2}@a"
    return V(f"**Number the steps.** Line 1 is numbered `1.`; every other line starts with `0.`. Make each number "
             f"the step's position (1, 2, 3 ... {len(items)}).\n\n**Constraints:** a macro that copies the number "
             f"above (yank it, then paste it over the `0`) and increments it with Ctrl-A; played with a count; par "
             f"{count_keys(keys)} keys.", {"steps.md": text}, keys, require=["q", "@", "<C-a>"],
             par=count_keys(keys), show="target")
