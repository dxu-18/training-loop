"""9.06 Search & replace: / ? n N, \\< \\> \\c \\v, :s with ranges and flags (g c n e), & \\r \\1, :g / :v (with
:normal), :vimgrep and the quickfix list (:copen :cn :cp :cdo)."""

from ...core import pick
from ._common import MODELS, VARS, V, count_keys, drill, log_lines, template, train_py

M = "9.06"


def _find(lines, needle, start=0):
    for i in range(start, len(lines)):
        j = lines[i].find(needle)
        if j >= 0:
            return i + 1, j + 1
    raise ValueError(needle)


# ================================================================= lite

@template(M, "lite")
def search_basic(rng):
    text, f = train_py(rng)
    lines = text.split("\n")
    files = {"train.py": text}
    nav = ("text", "cursor")
    items = [
        ("Search forward for `backward` (`/backward` then Enter): the cursor lands on the match.", "/backward<CR>",
         {"require": ["/"]}),
        ("The cursor is at the end of the file. Search BACKWARD for `zero_grad` (`?zero_grad`).", "?zero_grad<CR>",
         {"cursor": (len(lines) - 1, 1), "require": ["?"]}),
        ("Search for `loss` (`/loss`), then jump to the third match with `n` twice.", "/loss<CR>nn",
         {"require": ["/", "n"]}),
        ("Search for `loss`, then go back to the PREVIOUS match with `N` (it wraps around to the end of the file).",
         "/loss<CR>N", {"require": ["N"]}),
        ("Jump to the next WHOLE word `loss` (`loss_fn` doesn't count): search for `\\<loss\\>`.",
         "/\\<loss\\><CR>", {"cursor": (_find(lines, "loss_fn = ")[0], 1), "require": ["\\<"]}),
        (f"Find `{f['opt'].lower()}` ignoring case (`\\c` anywhere in the pattern): `/{f['opt'].lower()}\\c`.",
         f"/{f['opt'].lower()}\\c<CR>", {"require": ["\\c"]}),
        ("`hlsearch` is on and every `loss` is highlighted from an earlier search. Search for `backward`, then clear "
         "the highlighting with `:noh` (`:nohlsearch`: it stays off until the next search).",
         "/backward<CR>:noh<CR>", {"prep": [(5, ":set hlsearch<CR>/loss<CR>gg")], "require": [":noh"]}),
    ]
    return drill(rng, items, files, check=nav, show=None)


@template(M, "lite")
def subst_basic(rng):
    old, new = rng.sample(["x", "inp", "data", "batch"], 2)
    m_old, m_new = rng.sample(MODELS, 2)
    text = (f"def run({old}):\n    y = model({old}) + bias({old})\n    print({old}, y)\n    return y\n\n"
            f"MODEL = \"{m_old}\"  # was {m_old}\n")
    files = {"run.py": text}
    items = [
        (f"On line 2, replace the FIRST `{old}` with `{new}` (`:s/{old}/{new}/`).", f":s/{old}/{new}/<CR>",
         {"cursor": (2, 1), "require": [":s"]}),
        (f"On line 2, replace EVERY `{old}` with `{new}` (the `g` flag).", f":s/{old}/{new}/g<CR>",
         {"cursor": (2, 1), "require": [":s", ":s_g"]}),
        (f"Replace every `{old}` in the whole file with `{new}` (`:%s/{old}/{new}/g`).", f":%s/{old}/{new}/g<CR>",
         {"require": [":%s", ":s_g"]}),
        (f"On the last line, replace both `{m_old}` with `{m_new}`.", f":s/{m_old}/{m_new}/g<CR>",
         {"cursor": (6, 1), "require": [":s_g"]}),
    ]
    return drill(rng, items, files)


# ================================================================= easy

@template(M, "easy")
def subst_ranges(rng):
    text, f = train_py(rng)
    lines = text.split("\n")
    files = {"train.py": text}
    a, b = _find(lines, "def train")[0], _find(lines, "return model")[0]
    m_lines = [i + 1 for i, ln in enumerate(lines) if "model" in ln]
    lr_count = sum(ln.count("LR") for ln in lines)
    items = [
        (f"Rename `model` to `net` only inside the train function: lines {a}-{b} (`:{a},{b}s/model/net/g`).",
         f":{a},{b}s/model/net/g<CR>", {"require": [":ranges"]}),
        (f"Select lines {a}-{b} (the train function) in Visual Line mode, then rename `model` to `net` in the "
         f"selection (`:` pre-fills `:'<,'>`).", f"{a}GV{b}G:s/model/net/g<CR>", {"require": [":'<,'>s"]}),
        (f"Rename `LR` to `LEARNING_RATE`, but confirm each match (`c` flag): there are {lr_count}; accept the "
         f"first, skip the second (`y`, `n`).", f":%s/\\<LR\\>/LEARNING_RATE/gc<CR>yn" + "n" * max(0, lr_count - 2),
         {"require": [":s_c"]}),
        (f"From the cursor's line to the end of the file (`.,$`), replace `model` with `net`.",
         ":.,$s/model/net/g<CR>", {"cursor": (b, 1), "require": [":ranges"]}),
    ]
    return drill(rng, items, files)


@template(M, "easy")
def flags_and_specials(rng):
    todos = rng.randint(2, 5)
    body = []
    for i in range(6):
        body.append(f"step_{i} = {rng.randint(1, 99)}" + ("  # TODO" if i < todos else "") + ("   " if i % 2 else ""))
    text = "\n".join(body) + "\n"
    args = "result = train(model, loader, optimizer, scheduler)\n"
    items = [
        (f"Count the TODOs with `:%s/TODO//gn` (the `n` flag only counts: nothing changes), then add a last line "
         f"`# {todos} TODOs` with the number it reports.", f":%s/TODO//gn<CR>Go# {todos} TODOs<Esc>",
         {"require": [":s_n"]}),
        ("Strip the trailing spaces from every line: `:%s/\\s\\+$//e` (`e`: no error if a line has none).",
         ":%s/\\s\\+$//e<CR>", {"require": [":%s", ":s_e"]}),
        ("Wrap every value in parentheses (`step_3 = 42` becomes `step_3 = (42)`): `:%s/\\<\\d\\+\\>/(&)/g`. `&` in "
         "the replacement is the whole match, and `\\<` `\\>` keep the digits inside names out of it.",
         ":%s/\\<\\d\\+\\>/(&)/g<CR>", {"require": ["&"]}),
        ("Put each argument of `train(...)` on its own line: replace `, ` with `,` and a line break (`\\r` in the "
         "replacement).", ":s/, /,\\r/g<CR>", {"files": {"call.py": args}, "require": ["\\r"]}),
    ]
    return drill(rng, items, {"steps.py": text})


# ================================================================= medium

@template(M, "medium")
def global_cmds(rng):
    text, f = train_py(rng)
    text = text.replace("        print(f", "        print(\"debug\", epoch)\n        print(f", 1)
    text = text.replace("            loss.backward()", "            print(loss)  # TODO remove\n            loss.backward()", 1)
    logs = "\n".join(log_lines(rng, 12)) + "\n"
    items = [
        ("Delete every line that contains `print(` (`:g/print(/d`).", ":g/print(/d<CR>",
         {"files": {"train.py": text}, "require": [":g"]}),
        ("Keep only the WARN lines: delete every line that does NOT contain `WARN` (`:v/WARN/d`).", ":v/WARN/d<CR>",
         {"files": {"train.log": logs}, "require": [":v"]}),
        ("Append `  # check` to every line that contains `TODO`, with `:g/TODO/normal A  # check`.",
         ":g/TODO/normal A  # check<CR>", {"files": {"train.py": text}, "require": [":g_normal"]}),
        ("Delete every DEBUG line from the log.", ":g/DEBUG/d<CR>", {"files": {"train.log": logs}, "require": [":g"]}),
        ("Put `# ` in front of every line containing `print`: `:g/print/normal I# `.", ":g/print/normal I# <CR>",
         {"files": {"train.py": text}, "require": [":g_normal"]}),
    ]
    if "WARN" not in logs:
        items = [it for it in items if "WARN" not in it[0]]
    if "DEBUG" not in logs:
        items = [it for it in items if "DEBUG" not in it[0]]
    return drill(rng, items, {"train.py": text})


@template(M, "medium")
def capture_groups(rng):
    pairs = rng.sample(VARS, 6)
    swap = "".join(f"{pairs[i]} = {pairs[i + 1]}\n" for i in range(0, 6, 2))
    runs = "".join(f"run_{rng.randint(1, 99)}: acc {rng.randint(50, 99)}\n" for _ in range(4))
    names = "".join(f"{n}\n" for n in rng.sample(MODELS, 4))
    items = [
        ("Swap both sides of every assignment (`a = b` becomes `b = a`) with capture groups: "
         "`:%s/\\(\\w\\+\\) = \\(\\w\\+\\)/\\2 = \\1/`.", ":%s/\\(\\w\\+\\) = \\(\\w\\+\\)/\\2 = \\1/<CR>",
         {"files": {"swap.py": swap}, "require": ["\\1"]}),
        ("Swap both sides of every assignment, this time in very magic mode (no backslashes before the groups; a "
         "literal `=` is written `\\=`): `:%s/\\v(\\w+) \\= (\\w+)/\\2 = \\1/`.",
         ":%s/\\v(\\w+) \\= (\\w+)/\\2 = \\1/<CR>", {"files": {"swap.py": swap}, "require": ["\\v"]}),
        ("Rewrite `run_N: acc M` lines as `M,N` (accuracy first, then the run number).",
         ":%s/\\vrun_(\\d+): acc (\\d+)/\\2,\\1/<CR>", {"files": {"runs.txt": runs}, "require": ["\\1"]}),
        ("Turn each model name into a quoted, comma-terminated item: `name` → `\"name\",` with `&`.",
         ":%s/.*/\"&\",/<CR>", {"files": {"models.txt": names}, "require": ["&"]}),
    ]
    return drill(rng, items, {"swap.py": swap}, show="target")


# ================================================================= hard

def _project(rng):
    renames = {"load_data": "read_dataset", "get_batch": "next_batch", "make_model": "build_model",
               "init_weights": "reset_parameters", "log_metrics": "report_metrics", "save_ckpt": "save_checkpoint",
               "eval_step": "validation_step", "set_seed": "seed_everything"}
    old = rng.choice(sorted(renames))
    new = renames[old]
    files = {
        "main.py": f"from utils import {old}\n\n\ndef main():\n    x = {old}()  # TODO cache it\n    return x\n",
        "utils.py": f"def {old}():\n    return [1, 2, 3]\n",
        "tests/test_utils.py": f"from utils import {old}\n\n\ndef test_it():\n    assert {old}()  # TODO more\n",
        "README.md": f"Call `{old}()` to get the data.\n",
    }
    return files, old, new


@template(M, "hard")
def quickfix(rng):
    files, old, new = _project(rng)
    items = [
        ("Find every `TODO` in the project's Python files with `:vimgrep /TODO/ **/*.py` (it jumps to the first "
         "one), then go to the SECOND result with `:cn`.", ":vimgrep /TODO/ **/*.py<CR>:cn<CR>",
         {"check": ("text", "cursor"), "require": [":vimgrep", ":cn"], "target": False}),
        ("Search the Python files for `TODO` with `:vimgrep`, open the results list (`:copen`), and leave it "
         "open.", ":vimgrep /TODO/ **/*.py<CR>:copen<CR>",
         {"check": ("layout",), "require": [":vimgrep", ":copen"]}),
        (f"`:vimgrep` for `{old}` in all the Python files, step forward twice with `:cn`, then back once with "
         f"`:cp`.", f":vimgrep /{old}/ **/*.py<CR>:cn<CR>:cn<CR>:cp<CR>",
         {"check": ("text", "cursor"), "require": [":vimgrep", ":cp"], "target": False}),
        (f"Rename `{old}` to `{new}` in every Python file: `:vimgrep /{old}/ **/*.py`, then "
         f"`:cdo s/{old}/{new}/g | update` (update saves each changed file). README.md must stay as it is.",
         f":vimgrep /{old}/ **/*.py<CR>:cdo s/{old}/{new}/g | update<CR>",
         {"check": ("files",), "require": [":vimgrep", ":cdo"]}),
    ]
    return drill(rng, items, files, show=None)


@template(M, "hard")
def regex_golf(rng):
    keys_txt = "".join(f"{k}: {v}\n" for k, v in (("epochs", rng.randint(5, 50)), ("lr", rng.choice(("0.1", "0.01"))),
                                                  ("batch_size", rng.choice((32, 64))), ("seed", rng.randint(0, 9))))
    snake = "".join(f"{w}\n" for w in rng.sample(["max_seq_len", "num_attention_heads", "hidden_dropout_prob",
                                                   "layer_norm_eps", "vocab_size", "intermediate_size"], 4))
    numbered = "".join(f"{i}. {w}\n" for i, w in enumerate(rng.sample(MODELS, 4), 1))
    items = [
        ("Turn `key: value` lines into `--key=value` command-line flags.", ":%s/\\v(\\w+): /--\\1=/<CR>",
         {"files": {"config.txt": keys_txt}}),
        ("Convert every snake_case name to camelCase (`max_seq_len` → `maxSeqLen`): `\\u` uppercases the next "
         "character of the replacement.", ":%s/_\\(\\a\\)/\\u\\1/g<CR>", {"files": {"names.txt": snake}}),
        ("Remove the `N. ` numbering at the start of every line.", ":%s/^\\d\\+\\. //<CR>",
         {"files": {"list.md": numbered}}),
    ]
    prompt, keys, opts = pick(rng, items)
    return V(prompt + " Golf it: one substitute command.", opts["files"], keys, par=count_keys(keys) + 2,
             show="target")


# ================================================================= challenge

@template(M, "challenge")
def project_rename(rng):
    files, old, new = _project(rng)
    keys = f":vimgrep /\\<{old}\\>/ **/*.py<CR>:cdo s/\\<{old}\\>/{new}/g | update<CR>"
    return V(f"**Project-wide rename.** Rename the function `{old}` to `{new}` in every Python file of the project "
             f"(definition, imports, calls, tests) and save them. README.md is documentation: leave it alone.\n\n"
             f"**Constraints:** do it from this Vim, without opening the files one by one: `:vimgrep` + `:cdo`.",
             files, keys, check=("files",), require=[":vimgrep", ":cdo"], show=None)


@template(M, "challenge")
def log_to_csv(rng):
    lines = [f"INFO step={100 * i} loss={rng.uniform(0.2, 2):.4f} lr={rng.choice(('1e-3', '5e-4'))}"
             for i in range(1, 7)]
    text = "\n".join(lines) + "\n"
    keys = ":%s/\\vINFO step\\=(\\d+) loss\\=(\\S+) .*/\\1,\\2/<CR>ggOstep,loss<Esc>"
    return V("**Log to CSV.** Turn the log into a CSV with a header: the first line `step,loss`, then one `step,loss` "
             "row per log line (drop `INFO` and the lr).\n\n**Constraints:** one substitution with capture groups "
             f"(very magic mode helps), then the header; par {count_keys(keys)} keys.",
             {"train.log": text}, keys, require=["\\1"], par=count_keys(keys), show="target")
