"""9.12 A starter .vimrc & Vim keys everywhere: :set options, mapleader and mappings (nnoremap, inoremap jk),
autocmd, ~/.vimrc and :source, and vi keys in bash (set -o vi)."""

from ...core import pick
from ._common import VARS, V, drill, template, train_py

M = "9.12"
OPTS = ("number", "relativenumber", "ignorecase", "smartcase", "hlsearch", "incsearch", "expandtab", "shiftwidth",
        "tabstop", "colorcolumn", "cursorline", "splitright", "splitbelow", "scrolloff", "undofile")

STARTER = """set nocompatible
syntax on
filetype plugin indent on
set number relativenumber
set tabstop=4 shiftwidth=4 expandtab autoindent smartindent
set ignorecase smartcase incsearch hlsearch
set scrolloff=5 splitright splitbelow undofile
set colorcolumn=88 cursorline
let mapleader = " "
nnoremap <leader>w :w<CR>
nnoremap <leader>h :nohlsearch<CR>
inoremap jk <Esc>
nnoremap <C-h> <C-w>h
nnoremap <C-l> <C-w>l
autocmd BufWritePre *.py :%s/\\s\\+$//e
"""


def _typed(text: str) -> str:
    """Text typed literally in Insert mode or on the command line: `<` written as <lt> in the key notation."""
    return text.replace("<", "<lt>").replace("\n", "<CR>")


# ================================================================= lite

@template(M, "lite")
def set_options(rng):
    text, f = train_py(rng)
    files = {"train.py": text}
    sw = rng.choice((2, 4))
    items = [
        ("Turn on line numbers (`:set number`).", ":set number<CR>", ["number"], [":set number"]),
        ("Turn on relative line numbers too (`:set number relativenumber`): the current line shows its real number, "
         "the others their distance from it.", ":set number relativenumber<CR>", ["number", "relativenumber"],
         [":set relativenumber"]),
        ("Make searches ignore case unless the pattern has a capital letter: `:set ignorecase smartcase`.",
         ":set ignorecase smartcase<CR>", ["ignorecase", "smartcase"], [":set ignorecase", ":set smartcase"]),
        ("Highlight all search matches and show matches while typing: `:set hlsearch incsearch`.",
         ":set hlsearch incsearch<CR>", ["hlsearch", "incsearch"], [":set hlsearch"]),
        (f"Make Tab insert spaces and indent by {sw}: `:set expandtab shiftwidth={sw} tabstop={sw}`.",
         f":set expandtab shiftwidth={sw} tabstop={sw}<CR>", ["expandtab", "shiftwidth", "tabstop"],
         [":set shiftwidth"]),
        ("Show a guide at column 88 and highlight the cursor's line: `:set colorcolumn=88 cursorline`.",
         ":set colorcolumn=88 cursorline<CR>", ["colorcolumn", "cursorline"], [":set"]),
    ]
    prompt, keys, opts, req = pick(rng, items)
    return V(prompt, files, keys, check=("text", "options"), options=opts, require=req, show=None)


@template(M, "lite")
def options_in_action(rng):
    names = rng.sample(VARS, 3)
    text = "def f():\n" + "".join(f"\treturn_{n} = {n}\n" for n in names) + "\nModel = 1\nmodel = 2\n"
    items = [
        ("Turn on `ignorecase` and search for `MODEL`: with it on, the search finds `Model` on line 6.",
         ":set ignorecase<CR>/MODEL<CR>", ("text", "cursor", "options"), ["ignorecase"], [":set ignorecase"]),
        ("Turn on `ignorecase smartcase`, then search for `model` (all lowercase: it ignores case, so it stops at "
         "`Model` first).", ":set ignorecase smartcase<CR>/model<CR>", ("text", "cursor", "options"),
         ["ignorecase", "smartcase"], [":set smartcase"]),
        ("Indent with spaces instead of tabs: `:set expandtab shiftwidth=4`, then re-indent lines 2-4 so their tabs "
         "become spaces (`:2,4retab` converts them).", ":set expandtab shiftwidth=4<CR>:2,4retab<CR>",
         ("text", "options"), ["expandtab", "shiftwidth"], [":set expandtab"]),
    ]
    prompt, keys, check, opts, req = pick(rng, items)
    return V(prompt, {"f.py": text}, keys, check=check, options=opts, require=req, show=None, target=False)


# ================================================================= easy

@template(M, "easy")
def leader_mappings(rng):
    text, f = train_py(rng)
    files = {"train.py": text}
    items = [
        ("Make Space the leader key (`:let mapleader = \" \"`), then map `<leader>w` to save: "
         "`:nnoremap <leader>w :w<CR>` (type the `<leader>` and `<CR>` literally).",
         ":let mapleader = \" \"<CR>:nnoremap <lt>leader>w :w<lt>CR><CR>", [":let mapleader", ":nnoremap"]),
        ("Make Space the leader key, then map `<leader>h` to clear the search highlighting: "
         "`:nnoremap <leader>h :nohlsearch<CR>`.",
         ":let mapleader = \" \"<CR>:nnoremap <lt>leader>h :nohlsearch<lt>CR><CR>", [":let mapleader", ":nnoremap"]),
        ("Map Ctrl-H and Ctrl-L to move to the window on the left / right: `:nnoremap <C-h> <C-w>h` and "
         "`:nnoremap <C-l> <C-w>l`.", ":nnoremap <lt>C-h> <lt>C-w>h<CR>:nnoremap <lt>C-l> <lt>C-w>l<CR>",
         [":nnoremap"]),
    ]
    prompt, keys, req = pick(rng, items)
    return V(prompt, files, keys, check=("text", "maps"), require=req, show=None)


@template(M, "easy")
def jk_escape(rng):
    text, f = train_py(rng)
    lines = text.split("\n")
    ll = next(i + 1 for i, x in enumerate(lines) if "loss.backward" in x)
    files = {"train.py": text}
    keys = ":inoremap jk <lt>Esc><CR>otorch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)jk"
    return V("Map `jk` in Insert mode to Escape (`:inoremap jk <Esc>`, typed literally), then USE it: open a line below "
             "`loss.backward()` (`o`), type `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` and leave Insert "
             "mode by typing `jk` quickly.", files, keys, cursor=(ll, 1), check=("text", "maps", "mode"),
             require=[":inoremap"], show="diff")


# ================================================================= medium

@template(M, "medium")
def autocmd_strip(rng):
    names = rng.sample(VARS, 4)
    text = "".join(f"{n} = {i}" + " " * rng.randint(1, 4) + "\n" for i, n in enumerate(names))
    keys = ":autocmd BufWritePre *.py :%s/\\s\\+$//e<CR>:w<CR>"
    return V("Every line of this file ends in invisible trailing spaces. Add an autocommand that strips trailing "
             "whitespace from Python files before each save (`:autocmd BufWritePre *.py :%s/\\s\\+$//e`), then save "
             "with `:w` and let it do the work.", {"vars.py": text}, keys, check=("files",), require=[":autocmd"],
             show=None)


@template(M, "medium")
def write_vimrc(rng):
    text, f = train_py(rng)
    sw = rng.choice((2, 4))
    lines = ["set number", f"set shiftwidth={sw} expandtab", "set ignorecase smartcase"]
    keys = ":e ~/.vimrc<CR>i" + _typed("\n".join(lines)) + "<Esc>:w<CR>:source %<CR>"
    return V("Start a `~/.vimrc`: open it (`:e ~/.vimrc`), write these three lines, save, and load it right away with "
             "`:source %` (`%` is the file you're editing):\n\n```vim\n" + "\n".join(lines) + "\n```",
             {"train.py": text}, keys, check=("files", "options"), check_files=[".vimrc"],
             options=("number", "shiftwidth", "expandtab", "ignorecase", "smartcase"), require=[":source"],
             show=None)


# ================================================================= hard

@template(M, "hard")
def run_mapping(rng):
    script = f"print({rng.randint(2, 9)} * {rng.randint(10, 20)})\n"
    keys = ":let mapleader = \" \"<CR>:nnoremap <lt>leader>r :!python %<lt>CR><CR> r<CR>"
    return V("Make Space the leader and map `<leader>r` to run the current Python file (`:nnoremap <leader>r "
             ":!python %<CR>`), then try it: press Space r (and Enter to come back).", {"calc.py": script}, keys,
             check=("text", "maps"), require=[":let mapleader", ":nnoremap", ":!"], show=None)


@template(M, "hard")
def window_mappings(rng):
    text, f = train_py(rng)
    note = rng.choice(["# reviewed", "# ok"])
    files = {"model.py": "import torch\n", "train.py": text}
    keys = f":nnoremap <lt>C-l> <lt>C-w>l<CR><C-l>Go{note}<Esc>"
    return V(f"model.py and train.py are side by side. Map Ctrl-L to `<C-w>l` (`:nnoremap <C-l> <C-w>l`), then use "
             f"your new Ctrl-L to move to the right window and add `{note}` as its last line.", files, keys,
             start="-O model.py train.py", check=("text", "maps", "layout"), require=[":nnoremap", "key:<C-l>"],
             show=None)


# ================================================================= challenge

@template(M, "challenge")
def starter_vimrc(rng):
    text, f = train_py(rng)
    keys = ":e ~/.vimrc<CR>i" + _typed(STARTER.rstrip("\n")) + "<Esc>:w<CR>:source %<CR>"
    return V("**The guide's starter `.vimrc`.** Write it to `~/.vimrc` (`:e ~/.vimrc`), save it, and load it with "
             "`:source %`. Submit checks the options and mappings it sets (clipboard and mouse are left out here):\n\n"
             "```vim\n" + STARTER + "```", {"train.py": text}, keys, check=("options", "maps"),
             options=OPTS, require=[":source"], show=None, timeout=30.0)


@template(M, "challenge")
def vi_everywhere(rng):
    name = rng.choice(["notes.md", "todo.md", "habits.md", "setup.md"])
    todo = rng.sample(["try vi mode in bash", "install the vimrc", "use jk for a day", "map <leader>r",
                       "turn on the VS Code Vim extension", "try IdeaVim", "learn :help"], 3)
    done = rng.choice(["set -o vi", "bindkey -v (zsh)", "Vim keys in the shell"])
    notes = "# TODO\n" + "".join(f"- {x}\n" for x in todo)
    keys = f"set -o vi<CR>vim {name}<CR>Go- done: {done}<Esc>:wq<CR>"
    return V(f"**Vim keys everywhere.** At the bash prompt, switch bash's line editing to vi mode (`set -o vi`: Esc "
             f"then gives you Normal mode on the command line; zsh's equivalent is `bindkey -v`). Then open `{name}` "
             f"in Vim, add `- done: {done}` as its last line, save and quit.", {name: notes}, keys,
             shell=True, check=("files", "quit", "bash_vi"), require=[":wq|:x|ZZ"], show=None)
