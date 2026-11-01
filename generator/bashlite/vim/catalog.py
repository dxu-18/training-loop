"""Every Vim command the guide (COMP_GUIDE section 9) teaches, with a one-line description.

The keys are the names the engine records as commands run (Vim.used): what problems `require`, what the
Instructions list, and what scripts/guide_coverage.py checks is practised.
"""

SECTIONS = {"starting": "Starting and leaving Vim", "editing": "Files", "motion": "Moving around",
            "insert": "Inserting text", "change": "Changing text", "undo": "Undo and redo", "visual": "Visual mode",
            "pattern": "Patterns and searching", "registers": "Registers", "windows": "Windows and buffers",
            "tabpage": "Tab pages", "fold": "Folding", "options": "Options", "map": "Key mappings",
            "autocmd": "Automatic commands", "quickfix": "The quickfix list", "terminal": "The terminal window",
            "spell": "Spell checking", "various": "Various commands"}


def _c(desc, help_="various"):
    return {"desc": desc, "help": help_}


CATALOG = {
    # 9.01 modes, opening, saving, quitting
    "<Esc>": _c("back to Normal mode (also <C-[>)", "insert"),
    "i": _c("Insert mode, before the cursor", "insert"),
    "v": _c("Visual mode: select characters", "visual"),
    "V": _c("Visual mode: select whole lines", "visual"),
    "<C-v>": _c("Visual mode: select a rectangular block", "visual"),
    ":": _c("Command-line mode: type an Ex command, run it with <CR>", "various"),
    "R": _c("Replace mode: what you type overwrites the text", "insert"),
    ":w": _c("write (save) the file; :w new.py saves as new.py", "editing"),
    ":wa": _c("write all changed buffers", "editing"),
    ":q": _c("quit (fails if there are unsaved changes)", "editing"),
    ":q!": _c("quit and throw away the changes", "editing"),
    "ZQ": _c("quit without saving (same as :q!)", "editing"),
    ":wq": _c("write and quit", "editing"),
    ":x": _c("write if changed, then quit", "editing"),
    "ZZ": _c("write if changed, then quit (same as :x)", "editing"),
    ":qa!": _c("quit everything, discarding all changes", "editing"),
    ":e": _c("edit another file (:e other.py)", "editing"),
    ":e!": _c("revert the buffer to the saved file", "editing"),
    "vim +N": _c("open a file at line N: vim +42 file.py", "starting"),
    "vim -O": _c("open files side by side: vim -O a.py b.py", "starting"),
    "view": _c("open a file read-only: view file.txt", "starting"),
    # 9.02 moving around
    "h": _c("left", "motion"), "j": _c("down a line", "motion"), "k": _c("up a line", "motion"),
    "l": _c("right", "motion"),
    "count": _c("a number before a command repeats it: 5j moves 5 lines down", "motion"),
    "w": _c("to the start of the next word", "motion"), "b": _c("back to the start of a word", "motion"),
    "e": _c("to the end of the word", "motion"), "ge": _c("back to the end of the previous word", "motion"),
    "W": _c("next WORD (only white space separates WORDs: my.var[0] is one WORD)", "motion"),
    "B": _c("back a WORD", "motion"), "E": _c("to the end of the WORD", "motion"),
    "0": _c("to the start of the line", "motion"), "^": _c("to the first non-blank character", "motion"),
    "$": _c("to the end of the line", "motion"), "g_": _c("to the last non-blank character", "motion"),
    "gg": _c("to the first line (42gg: line 42)", "motion"), "G": _c("to the last line (42G: line 42)", "motion"),
    ":N": _c("go to line N: :42", "motion"),
    "<C-d>": _c("scroll half a page down", "motion"), "<C-u>": _c("scroll half a page up", "motion"),
    "<C-f>": _c("one page down", "motion"), "<C-b>": _c("one page up", "motion"),
    "H": _c("to the top of the screen", "motion"), "M": _c("to the middle of the screen", "motion"),
    "L": _c("to the bottom of the screen", "motion"),
    "zz": _c("scroll so the cursor line is in the middle", "motion"),
    "zt": _c("scroll so the cursor line is at the top", "motion"),
    "zb": _c("scroll so the cursor line is at the bottom", "motion"),
    "f": _c("to the next occurrence of a character on the line: f(", "motion"),
    "F": _c("to the previous occurrence of a character: F(", "motion"),
    "t": _c("to just before the next occurrence: t(", "motion"),
    "T": _c("to just after the previous occurrence: T(", "motion"),
    ";": _c("repeat the last f t F T", "motion"), ",": _c("repeat the last f t F T backwards", "motion"),
    "%": _c("to the matching ( ) [ ] { }", "motion"),
    "{": _c("back to the previous blank line (paragraph)", "motion"),
    "}": _c("on to the next blank line (paragraph)", "motion"),
    "*": _c("search forward for the word under the cursor", "pattern"),
    "#": _c("search backward for the word under the cursor", "pattern"),
    "gd": _c("go to the local definition of the word under the cursor", "pattern"),
    "gf": _c("edit the file whose name is under the cursor", "editing"),
    "<C-o>": _c("back to where you were before the last jump", "motion"),
    "<Tab>": _c("forward again in the jump list (<C-i>)", "motion"),
    # 9.03 inserting and basic editing
    "a": _c("Insert mode after the cursor", "insert"), "I": _c("insert at the start of the line", "insert"),
    "A": _c("append at the end of the line", "insert"), "o": _c("open a new line below", "insert"),
    "O": _c("open a new line above", "insert"), "x": _c("delete the character under the cursor", "change"),
    "X": _c("delete the character before the cursor", "change"), "r": _c("replace one character: rX", "change"),
    "s": _c("delete the character and start inserting", "change"),
    "S": _c("delete the whole line and start inserting", "change"),
    "dd": _c("delete (cut) the line", "change"), "yy": _c("yank (copy) the line", "change"),
    "p": _c("put (paste) after the cursor / below the line", "change"),
    "P": _c("put before the cursor / above the line", "change"),
    "u": _c("undo", "undo"), "<C-r>": _c("redo", "undo"), "U": _c("undo all changes on the last changed line", "undo"),
    ".": _c("repeat the last change", "change"),
    "J": _c("join the next line onto this one (with a space)", "change"),
    "gJ": _c("join without adding a space", "change"), "~": _c("toggle the case of the character", "change"),
    ">>": _c("indent the line (5>> indents 5 lines)", "change"), "<<": _c("dedent the line", "change"),
    "==": _c("re-indent the line", "change"),
    "<C-a>": _c("add to the number under (or after) the cursor: 10<C-a> adds 10", "change"),
    "<C-x>": _c("subtract from the number under the cursor", "change"),
    "i_<C-w>": _c("(Insert mode) delete the word before the cursor", "insert"),
    "i_<C-u>": _c("(Insert mode) delete back to the start of the line", "insert"),
    "i_<C-n>": _c("(Insert mode) complete the word: the next match", "insert"),
    "i_<C-p>": _c("(Insert mode) complete the word: the previous match", "insert"),
    "i_<C-r>": _c("(Insert mode) put a register's text: <C-r>a", "insert"),
    "i_<C-t>": _c("(Insert mode) indent the line", "insert"), "i_<C-d>": _c("(Insert mode) dedent the line", "insert"),
    # 9.04 operators, motions, text objects
    "d": _c("operator: delete", "change"), "c": _c("operator: change (delete, then insert)", "change"),
    "y": _c("operator: yank (copy)", "change"), ">": _c("operator: indent", "change"),
    "<": _c("operator: dedent", "change"), "=": _c("operator: re-indent", "change"),
    "gu": _c("operator: lowercase", "change"), "gU": _c("operator: uppercase", "change"),
    "g~": _c("operator: toggle case", "change"), "gq": _c("operator: wrap text to 'textwidth'", "change"),
    "iw": _c("text object: inner word", "visual"), "aw": _c("text object: a word and the space after it", "visual"),
    "is": _c("text object: inner sentence", "visual"), "as": _c("text object: a sentence and its space", "visual"),
    "ip": _c("text object: inner paragraph", "visual"), "ap": _c("text object: a paragraph and the blank line after", "visual"),
    "i(": _c("text object: inside ( )", "visual"), "a(": _c("text object: ( ) and what's inside", "visual"),
    "i[": _c("text object: inside [ ]", "visual"), "i{": _c("text object: inside { }", "visual"),
    "a{": _c("text object: { } and what's inside", "visual"), "i<": _c("text object: inside < >", "visual"),
    'i"': _c('text object: inside " "', "visual"), 'a"': _c('text object: " " and what\'s inside', "visual"),
    "i'": _c("text object: inside ' '", "visual"), "i`": _c("text object: inside ` `", "visual"),
    "it": _c("text object: inside an HTML/XML tag", "visual"), "at": _c("text object: a whole tag element", "visual"),
    "dw": _c("delete to the next word", "change"), "d$": _c("delete to the end of the line (also D)", "change"),
    "D": _c("delete to the end of the line", "change"), "dG": _c("delete to the end of the file", "change"),
    "cw": _c("change to the end of the word", "change"), "c$": _c("change to the end of the line (also C)", "change"),
    "C": _c("change to the end of the line", "change"), "cc": _c("change the whole line", "change"),
    "ciw": _c("change the word under the cursor", "change"), 'ci"': _c("change inside the quotes", "change"),
    "ci(": _c("change inside the parentheses (function arguments)", "change"),
    "da(": _c("delete the parentheses and what's inside", "change"), "dap": _c("delete a paragraph", "change"),
    "yi{": _c("yank inside the braces", "change"), "dt": _c("delete up to (not including) a character: dt,", "change"),
    "cf": _c("change through a character: cf)", "change"), "d/": _c("delete up to a search match: d/foo", "change"),
    "gUiw": _c("uppercase the word", "change"), "=i{": _c("re-indent inside the braces", "change"),
    "gqap": _c("re-wrap the paragraph", "change"), ">ip": _c("indent the paragraph", "change"),
    # 9.05 visual mode
    "v_o": _c("(Visual) jump to the other end of the selection", "visual"),
    "gv": _c("reselect the last selection", "visual"),
    "v_d": _c("(Visual) delete the selection", "visual"), "v_y": _c("(Visual) yank the selection", "visual"),
    "v_c": _c("(Visual) change the selection", "visual"), "v_>": _c("(Visual) indent (3> indents 3 times)", "visual"),
    "v_<": _c("(Visual) dedent", "visual"), "v_=": _c("(Visual) re-indent", "visual"),
    "v_~": _c("(Visual) toggle case", "visual"), "v_u": _c("(Visual) lowercase", "visual"),
    "v_U": _c("(Visual) uppercase", "visual"), "v_J": _c("(Visual) join the lines", "visual"),
    "v_:": _c("(Visual) an Ex command on the selected lines (:'<,'>)", "visual"),
    "v_I": _c("(Visual block) insert in front of the block on every line", "visual"),
    "v_A": _c("(Visual block) append after the block on every line ($A: at each line's end)", "visual"),
    # 9.06 search and replace
    "/": _c("search forward", "pattern"), "?": _c("search backward", "pattern"),
    "n": _c("next match", "pattern"), "N": _c("previous match", "pattern"),
    ":noh": _c("clear the search highlighting", "pattern"),
    "\\<": _c("pattern: start of a word (\\<loss\\> matches the whole word only)", "pattern"),
    "\\c": _c("pattern: ignore case", "pattern"),
    "\\v": _c('pattern: "very magic": ( ) + ? | without backslashes', "pattern"),
    ":s": _c("substitute on the current line: :s/old/new/", "change"),
    ":s_g": _c("the g flag: every match on the line", "change"),
    ":%s": _c("substitute in the whole file: :%s/old/new/g", "change"),
    ":s_c": _c("the c flag: confirm each replacement (y/n/a/q)", "change"),
    ":s_n": _c("the n flag: count the matches, change nothing", "change"),
    ":s_e": _c("the e flag: no error when nothing matches", "change"),
    ":ranges": _c("only some lines: :10,20s/a/b/g", "change"),
    ":'<,'>s": _c("substitute in the visual selection", "change"),
    "\\1": _c("in the replacement: the first captured group \\( \\) (\\2 the second)", "change"),
    "&": _c("in the replacement: the whole match", "change"), "\\r": _c("in the replacement: a line break", "change"),
    ":g": _c("run a command on every line that matches: :g/pattern/d", "change"),
    ":v": _c("run a command on every line that does NOT match", "change"),
    ":g_normal": _c("normal-mode keys on matching lines: :g/TODO/normal A  # check", "change"),
    ":vimgrep": _c("search files: :vimgrep /TODO/ **/*.py", "quickfix"),
    ":copen": _c("show the results list", "quickfix"), ":cn": _c("the next result", "quickfix"),
    ":cp": _c("the previous result", "quickfix"),
    ":cdo": _c("a command on every result: :cdo s/old/new/g | update", "quickfix"),
    # 9.07 registers
    '"a': _c('use register a: "ayy yanks into it, "ap puts from it', "registers"),
    '"A': _c('append to register a: "Ayy', "registers"), '"+': _c("the system clipboard: \"+y \"+p", "registers"),
    '"0': _c("the last yank (still there after deletes)", "registers"),
    '"_': _c('the black hole: "_d deletes without touching your yank', "registers"),
    '".': _c("register .: the last inserted text", "registers"), '":': _c("register :: the last command", "registers"),
    '"/': _c("register /: the last search", "registers"), '"%': _c("register %: the current file name", "registers"),
    ":reg": _c("show the registers", "registers"),
    "i_<C-r>=": _c("(Insert mode) <C-r>=6*7<CR> inserts 42 (the expression register)", "insert"),
    # 9.08 macros
    "q": _c("record a macro: qa ... q records into register a", "various"),
    "@": _c("play a macro: @a (20@a: 20 times)", "various"), "@@": _c("play the last macro again", "various"),
    ":normal": _c("normal-mode keys on a range of lines: :5,20normal @a", "various"),
    "cgn": _c("change the next search match; . repeats on the next one", "change"),
    # 9.09 buffers, windows, tabs
    ":ls": _c("list the buffers", "windows"), ":b": _c("switch buffer: :b 2 or :b part_of_name", "windows"),
    ":bn": _c("next buffer", "windows"), ":bp": _c("previous buffer", "windows"),
    ":bd": _c("delete (close) a buffer", "windows"), "<C-^>": _c("toggle between the last two files", "windows"),
    ":sp": _c("split the window horizontally (:sp file)", "windows"), ":vsp": _c("split vertically", "windows"),
    "<C-w>s": _c("split the window", "windows"), "<C-w>v": _c("split vertically", "windows"),
    "<C-w>h": _c("to the window on the left", "windows"), "<C-w>j": _c("to the window below", "windows"),
    "<C-w>k": _c("to the window above", "windows"), "<C-w>l": _c("to the window on the right", "windows"),
    "<C-w>w": _c("to the next window", "windows"), "<C-w>q": _c("close this window", "windows"),
    "<C-w>o": _c("close every other window", "windows"), "<C-w>=": _c("make the windows equal sizes", "windows"),
    "<C-w>_": _c("make this window as tall as possible", "windows"), "<C-w>|": _c("make it as wide as possible", "windows"),
    ":resize": _c("set the window's height: :resize 20", "windows"),
    ":vertical resize": _c("set its width: :vertical resize 80", "windows"),
    ":tabnew": _c("open a new tab page (:tabnew file)", "tabpage"), "gt": _c("next tab", "tabpage"),
    "gT": _c("previous tab", "tabpage"), ":tabclose": _c("close the tab", "tabpage"),
    ":Explore": _c("the built-in file browser (netrw); :Ex", "windows"),
    ":Vexplore": _c("the file browser in a vertical split", "windows"),
    ":terminal": _c("a shell in a split window", "terminal"),
    "<C-w>N": _c("(terminal) Normal mode inside the terminal window", "terminal"),
    # 9.10 marks, folds, undo history
    "m": _c("set a mark: ma", "motion"), "'": _c("jump to a mark's line: 'a", "motion"),
    "`": _c("jump to a mark's exact spot: `a", "motion"), ":marks": _c("list the marks", "motion"),
    "mA": _c("uppercase marks are global: they jump between files", "motion"),
    "''": _c("back to where you were before the last jump (`` for the exact spot)", "motion"),
    "zf": _c("create a fold: zf} folds to the end of the paragraph", "fold"), "zo": _c("open a fold", "fold"),
    "zc": _c("close a fold", "fold"), "za": _c("toggle a fold", "fold"), "zR": _c("open all folds", "fold"),
    "zM": _c("close all folds", "fold"), ":set foldmethod": _c("fold by indentation: :set foldmethod=indent (good for Python)", "fold"),
    ":earlier": _c("go back in time: :earlier 10m (or :earlier 3 for 3 changes)", "undo"),
    ":later": _c("forward in time: :later 5m", "undo"),
    "g-": _c("to the previous state of the text (through undo branches)", "undo"),
    "g+": _c("to the next state of the text", "undo"),
    ":set undofile": _c("keep the undo history after closing the file", "undo"),
    # 9.11 command-line power moves
    ":!": _c("run a shell command: :!ls, :!python %", "various"),
    ":w !": _c("send the buffer to a command without saving: :w !python", "various"),
    ":r !": _c("insert a command's output: :r !date", "various"),
    ":range!": _c("filter lines through a command: :%!python -m json.tool, :'<,'>!sort", "various"),
    ":sort": _c("sort lines: :sort, :sort u (unique), :sort n (numeric)", "change"),
    ":%normal": _c("keys on every line: :%norm A,", "various"),
    ":d": _c("delete lines: :10,20d", "change"), ":m": _c("move lines: :10,20m 30", "change"),
    ":t": _c("copy lines: :10,20t$", "change"),
    ":w !sudo tee %": _c("save a root-owned file you opened without sudo", "editing"),
    ":set fileformat": _c("convert Windows line endings: :set ff=unix", "options"),
    ":set spell": _c("spell checking", "spell"), "]s": _c("next misspelled word", "spell"),
    "[s": _c("previous misspelled word", "spell"), "z=": _c("spelling suggestions", "spell"),
    "zg": _c("mark the word as good", "spell"), ":help": _c("the manual: :help ciw, :help :s", "help"),
    # 9.12 .vimrc
    ":set number": _c("line numbers", "options"), ":set relativenumber": _c("numbers relative to the cursor", "options"),
    ":set ignorecase": _c("searches ignore case...", "options"),
    ":set smartcase": _c("...unless the pattern has a capital letter", "options"),
    ":set hlsearch": _c("highlight every match", "options"), ":set incsearch": _c("show the match while typing", "options"),
    ":set expandtab": _c("Tab inserts spaces", "options"), ":set shiftwidth": _c("columns per indent level", "options"),
    ":set tabstop": _c("columns a tab character takes", "options"),
    ":let mapleader": _c('the <leader> key: let mapleader = " "', "map"),
    ":nnoremap": _c("a Normal-mode mapping: nnoremap <leader>w :w<CR>", "map"),
    ":inoremap": _c("an Insert-mode mapping: inoremap jk <Esc>", "map"),
    ":autocmd": _c("run a command on an event: autocmd BufWritePre *.py :%s/\\s\\+$//e", "autocmd"),
    ":source": _c("read a vimrc again: :source ~/.vimrc (or :so %)", "starting"),
    "set -o vi": _c("vi keys in bash (bindkey -v in zsh)", "starting"),
}


# the first command of each module in CATALOG (it is in guide order): which module introduces a command
_FIRST = {"<Esc>": "9.01", "h": "9.02", "a": "9.03", "d": "9.04", "v_o": "9.05", "/": "9.06", '"a': "9.07",
          "q": "9.08", ":ls": "9.09", "m": "9.10", ":!": "9.11", ":set number": "9.12"}


def _module_of() -> dict:
    out, cur = {}, None
    for k in CATALOG:
        cur = _FIRST.get(k, cur)
        out[k] = cur
    return out


MODULE_OF = _module_of()

# how a command is written in the Instructions (when it isn't its name)
DISPLAY = {
    "count": "{count}", "f": "f{char}", "F": "F{char}", "t": "t{char}", "T": "T{char}", "r": "r{char}",
    "m": "m{a-z}", "'": "'{a-z}", "`": "`{a-z}", "q": "q{a-z} … q", "@": "@{a-z}", "dt": "dt{char}",
    "cf": "cf{char}", "d/": "d/{pattern}", ":s_g": ":s/old/new/g", ":s_c": ":s/old/new/gc", ":s_n": ":s/pat//gn",
    ":s_e": ":s/pat/new/e", ":ranges": ":10,20s/a/b/g", ":g_normal": ":g/pattern/normal {keys}",
    ":N": ":{number}", "vim +N": "vim +42 file", "vim -O": "vim -O a.py b.py", "view": "view file",
    '"a': '"{a-z}', '"A': '"{A-Z}', ":set foldmethod": ":set foldmethod=indent", ":set fileformat": ":set ff=unix",
    ":%normal": ":%norm {keys}", ":range!": ":{range}!{command}", "i_<C-r>=": "<C-r>=",
    "<Tab>": "<Tab> / <C-i>", "''": "'' / ``", "mA": "m{A-Z}",
}


def shown(key: str) -> str:
    if key in DISPLAY:
        return DISPLAY[key]
    for pre in ("i_", "v_"):
        if key.startswith(pre) and len(key) > 2:
            return key[2:]
    return key
