# training-loop: project spec & session notes

This file is the source of truth for what we're building. It records the original
request and the decisions made since, so work can pick up in a new chat session.
Read it together with `COMP_GUIDE.md`, which defines the topics.

---

## 1. Original request (session 1, 2026-10-19)

Lightly cleaned up from the prompt; every requirement is kept.

**Goal.** Build a system that teaches the basic tools of the ML industry and the
common developer tools to anyone who already knows Python syntax: classes,
inheritance, dicts, sets, tuples and so on. It must eventually cover **every topic
in `COMP_GUIDE.md`**, which has 12 sections: NumPy, pandas, PyTorch, scikit-learn,
Matplotlib, Hugging Face, Bash, Git, Vim, SQL, LaTeX and Docker.

**Hosting.** The site will eventually be hosted on GitHub Pages (github.io). Start
with the scripts. The frontend should be minimal but elegant, and the exact style
can wait.

**User experience (like AoPS Alcumus, with differences).**
- Show one problem at a time. The user types the answer into a code box, which is a
  real coding editor that supports Python.
- There is **no progress bar**. The page shows only the problem and the code box.
- A "compiler" panel on the side runs the code. It shows the error message when the
  code fails, prints the output when it runs, and shows **right or wrong**.

**Question generation.**
- Each question says in plain English which feature or concept to type out or use.
- **Only the first 10 questions of a section show an example**, until the pattern
  sinks in. After that, an **"Example"** button reveals the example on demand. (Session 9 replaced the example with
  an **Instructions** dropdown, same open/closed rule: see §2, session 9.)
- Questions are **randomly generated**, with as much variety as possible while still
  testing the same syntax. To get that variety, the repo ships themed datasets, such
  as AI companies, pets and household appliances, plus data concepts, speech tech
  and image/video creation and recognition. The questions draw on these real-world
  things, since this is an ML practice tool.
- **Difficulty rises within each section, gently**, by question number:
  - 1–10: **lite**, 11–35: **easy**, 36–70: **medium**, 71–95: **hard**, 96–100: **challenge**, then the medal
    (session 10, 2026-10-30: cut from 250 questions, 1–50 lite … 246–250 challenge, because the templates are too
    few for 250 without repeats; the medal, answer reveal and the rest below that say 250 now mean 100)
  - 101 onward: random, using the same distribution, so a challenge still turns up
    now and then.
  - Difficulty means more logical complexity, so a problem needs more thinking.
    Lite to medium are still syntax questions, with medium a bit harder. Hard is
    syntax plus a little application. Challenge is full application at about
    **Codeforces 1800 (Expert)**.
- **Question counter.** The user sees a plain number, never a progress bar. At **250
  solved in a section**, the user gets a **surprise medal** for that section.
- **A wrong answer keeps the question number.** The generator rerolls a new question
  at the **same difficulty**.
- **After question 250** in a section, the user can choose to **reveal the answer**.

**Main page.** A list of all topics, one row each. The topic title is on the left.
On the right are the number solved in that topic and its 250-solved badge, which is
unique to each section.

**Deferred: medals.** Design unique, cute medal images for each topic later. For now
the slot is a placeholder.

**Working style.** Make subdirectories as needed. Be mindful of usage: the plan
was at 68% used when session 1 started. Nothing has to be done in one shot, but
session 1 should produce a working result.

### Session 2 request (2026-10-20)
Build the github.io page and make sure the whole frontend works, so people can start practicing.

### Session 3 request (2026-10-20): one page per subsection
**Every `###` subsection of a toolkit is its own practice page.** For example, clicking NumPy
on the main menu opens NumPy's page: a list of its 18 subsections (1.01 Setup, printing & help,
1.02 Creating arrays, …, 1.18 Vectorized ML patterns & testing). **Each subsection has its own
250 goal** (counter, difficulty ladder, surprise medal) and unlimited, randomly generated
questions. This supersedes the per-topic counter of session 1. The difficulty schedule
(1–50 lite … 246–250 challenge, then random) now applies within each subsection.

### Session 4 request (2026-10-20)
"Do pandas now." Same structure as NumPy: every pandas subsection (2.01–2.19) is its own page with
its own 250 goal; medal art is still deferred.

### Session 5 request (2026-10-20)
"Do PyTorch" ("the limit reset, just do the whole thing"): all 19 subsections (3.01–3.19) at the
NumPy/pandas standard, running in the browser.

---

## 2. Architecture decisions (sessions 1–5)

| Decision | Choice | Why |
|---|---|---|
| Where code runs | In the browser via **Pyodide 314.0.7** (CPython 3.14 in WebAssembly), inside a Web Worker | GitHub Pages is static and has no server. Pyodide 314.0.7 ships numpy 2.4.6, pandas 3.0.2, scikit-learn 1.8.0 and matplotlib 3.10.8, the same versions as COMP_GUIDE. The worker can be killed on infinite loops. |
| Generator language | **Python** (`generator/` package) | The same code runs locally (CLI, tests) and in the browser (Pyodide). Reference answers are computed by really running the reference solution. |
| Grading | Run the reference solution and the user's code on the same setup, then compare the answer values | Robust to many correct spellings. No hand-written expected outputs that can go stale. |
| How the answer is read | A variable named in the problem (default `result`), or else the value of the **last expression** in the code | Feels like a REPL, and short syntax answers can be one line. |
| Syntax enforcement | Optional `require` (names that must appear in the code, `a\|b` = either) and `no_loops` (AST check) | Stops "right value, wrong skill" answers, e.g. hard-coding `(3, 4)` instead of `.shape`. |
| Anti-hard-coding | `tests`: extra hidden setups the code must also pass. Used for every challenge problem. | Works like Codeforces hidden tests. |
| Instructions (was: Examples) | The functions a problem practises, each with a one-line description, only the new ones (session 9). Until then: another random instance of the same template, with its solution | Examples were long and often unrelated to the problem at hand; the function list says what to reach for without solving it. |
| Template registry | `REGISTRY[section][module][tier]`, registered with `@template(module, tier)` (the section is fixed per package). A module is **live** once all 5 tiers have templates. | Matches the one-page-per-subsection structure (session 3). |
| Template choice | Uniform over the templates registered for (section, module, tier), then a uniform theme, then random parameters | Gives every template an equal chance and keeps the questions varied. |
| State | `localStorage` key `training-loop:v2`: per module (`"numpy/1.05"`) solved count, current seed and tier, medal, code draft. The home page's Download / Upload save all of it to a `.tlsave` file and restore it (session 20). | A static site has no accounts. The session-1 key `training-loop:v1` (one counter per topic) is left untouched but no longer read. |
| Catalog | `scripts/build.py` parses COMP_GUIDE's `##`/`###` headings into `docs/build.json` (topics → modules, titles, live flag, template counts) | The site lists every subsection, even unfinished ones ("soon"), straight from the guide. `10.00 Practice schema` is skipped (reference material). |
| Printed-output answers | `answer=["<stdout>"]` grades what the code printed | Needed for 1.01 (print options, `np.info`). |
| Post checks | `post`: code run after the user's; its value is graded (e.g. `np.load('x.npy')`, `np.shares_memory(v, a)`) | Lets 1.06 check views vs copies and 1.17 check files the code wrote. |
| Sandboxed runs | Each run gets a fresh temp working directory; NumPy print options, `np.seterr` and pandas `display.*` options are restored and Matplotlib figures closed afterwards | The reference's files can't leak into the user's run, and global settings in one answer can't change the next. |
| pandas comparisons | `assert_frame_equal` / `assert_series_equal` with `check_names=False`, `check_index_type=False`, `check_column_type=False`, `check_freq=False` (and `check_dtype` only when asked); nullable arrays, `pd.NA`, `NaT` handled | Grade the data, not incidental metadata such as a Series name from `groupby`. |
| `_` in post checks | `post` code can refer to the answer value as `_` | e.g. Styler answers are checked with `_.set_uuid('chk').to_html()`. |
| Plot grading | Setup imports `matplotlib.pyplot as plt`; `post` inspects `plt.gca()` / `plt.gcf()` (bar heights, line data, labels, titles, saved PNG size). The grader sets `MPLBACKEND=Agg`. | Works for any plotting style that draws on the current figure. |
| Per-module packages | `TOPICS[...].extra` in `docs/app.js` (e.g. pandas `2.18` → matplotlib, jinja2) | Heavy packages download only on the page that needs them. |
| Editor | CodeMirror 5 (cdnjs), with the mode set per topic | Simple script tags and no build step. |
| Site folder | `docs/` on `main`, served by GitHub Pages ("deploy from branch", `/docs`). Live at https://dxu-18.github.io/training-loop/ | `docs/generator.zip` + `docs/build.json` are **committed**: run `scripts/build.py` before committing generator changes (`tests/test_bundle.py` fails if stale). An Actions deploy is ready in `ci/pages.yml` but needs a `gh` token with the `workflow` scope (see the file header). |
| PyTorch in the browser | **torch-lite**: a NumPy-backed `torch` package in `generator/shims/torch` (plus `triton-lite` in `generator/shims/triton`), put on `sys.path` by `generator/__init__.py` only when real torch is missing | Pyodide has no torch build. The shim follows PyTorch semantics for what the templates use (see "torch-lite" below), so prompts, solutions and printed tensors look exactly like PyTorch. |
| torch state in the sandbox | Before and after every run the grader calls `torch.manual_seed(0)`, re-enables grad mode, clears autocast/profilers, destroys the process group and resets `torch.compiler`; it also restores `os.environ` | The reference and the user's run see the same random numbers, and modes one answer leaves on can't leak into the next. |
| "Write a module" problems | `post` fills the user's module's parameters **by sorted name** from a fixed seed (or loads a reference `state_dict`), then compares outputs | Grading doesn't depend on the order the user created layers in, only on the names and the forward pass the prompt asks for. |
| Worker type | **Module** worker (`new Worker(..., {type: "module"})`) importing `pyodide.mjs` | Classic `importScripts` of the CDN script failed in current Chromium (session 2). Module workers work in current Chrome, Firefox and Safari. |

### Pyodide gotchas (found by the in-browser self-test, session 3)
- Pyodide is **32-bit WebAssembly**: NumPy's default integer is `int32` (e.g. `2 ** np.array([32])`
  overflows to 0). Use explicit `dtype=np.int64` in setups that need big integers.
- **Writable memory maps don't work** (`np.memmap(..., mode='w+')` raises `OSError: Bad file
  descriptor`). Read-only `np.load(..., mmap_mode='r')` works.
- Always run `scripts/e2e_check.py`: it grades every template's own solution inside Pyodide, **with exactly
  the packages its page loads** (a self-test on a page with extra packages hid a bug once), and checks
  that the slowest grade fits its time limit.
- `pd.reset_option("all")` imports matplotlib (the plotting options' callback), so the sandbox resets only
  `^display\.` options.
- `sqlite3` is bundled with Pyodide 314. **Not available:** `tabulate` (`df.to_markdown`) and `openpyxl`
  (Excel), so those parts of 2.03 aren't practised. Parquet is skipped too: loading `pyarrow` would switch
  pandas' default string storage mid-session.

### Non-Python sections: planned checkers
- **SQL**: built as postgres-lite (session 24): PostgreSQL 16 + psql 16 in Python; results and end states are
  compared on the shown database and hidden ones.
- **Bash / Git / Docker CLI**: a `command` checker that tokenizes with `shlex` and
  compares program, subcommand, flags (aliases, `-la` = `-l -a`, order-insensitive)
  and positional args. A simulated shell may come later.
- **Vim**: compare keystroke strings against a list of accepted alternatives.
- **LaTeX**: built as latex-lite (session 23): the answer is compiled and its document compared with the
  reference's.
- **Dockerfile / compose.yaml**: line-wise normalized comparison or a parsed-structure check.

### torch-lite (session 5): what it covers and where it differs from PyTorch
Decision: build a NumPy-backed `torch` (option (a) of the old open problem, extended to all of 3.01–3.19).
- **Covered:** tensors with PyTorch dtypes/printing (float32/int64 defaults, emulated bfloat16), broadcasting,
  real views and strides (`view` errors on incompatible strides, `contiguous`, `as_strided`, `channels_last`),
  reverse-mode autograd (hooks, `retain_grad`, `no_grad`, in-place-on-leaf errors), `torch.autograd.Function`
  (old and `setup_context` style, gradient count/shape checks, `gradcheck`), `torch.func` (grad, vmap,
  functional_call, jacrev, hessian), `nn` (Module registration, hooks, state_dict, Linear/Conv/Norm/Embedding/
  MultiheadAttention/TransformerEncoder, all common losses), `optim` (exact SGD/Adam/AdamW/RMSprop/Adagrad
  formulas, param groups, state_dict) and LR schedulers, `utils.data` (DataLoader with simulated workers and
  `get_worker_info`, samplers, `random_split`, DistributedSampler), checkpointing, autocast + GradScaler,
  profiler (record_function, schedules, key_averages), benchmark.Timer, `torch.library.custom_op`/`torch.ops`,
  `torch.compile` (option checks, `_orig_mod`, `fullgraph` errors, `explain` graph-break counts),
  `torch.distributed` (a real **single-process** group), FSDP2 `fully_shard` (identity sharding) and DCP.
  `triton-lite` runs `@triton.jit` kernels one program at a time with masked loads/stores, atomics and `tl.dot`;
  an unmasked out-of-bounds access raises a clear error.
- **Differs:** random numbers are reproducible under `manual_seed` but are not PyTorch's stream; no
  `create_graph` (higher-order autograd; `torch.func.hessian` uses finite differences of exact gradients);
  `torch.compile` runs eagerly (no speed-up); profiler op times are not meaningful (counts are); no CUDA, LSTM/GRU,
  real multi-process collectives or C++ extensions (`load_inline` explains why). Anything unsupported raises
  "not available in the browser's torch-lite" instead of silently misbehaving.
- **Testing:** `tests/` run against torch-lite too (the `.venv` has no torch), and the Pyodide self-test grades
  every PyTorch template in the browser.

### Guide coverage (session 6)
"Add the missing syntax as much as possible": `scripts/guide_coverage.py` lists every API in a module's guide code and
checks that some problem of that module uses it; **catalog templates** (`core.pick`, one API per problem, every item
graded by `tests/test_catalogs.py` and by the Pyodide self-test) close the gaps. Now 100% of guide APIs are practised for
NumPy (537), pandas (473) and PyTorch (855); browser-impossible names are listed with reasons in `EXCLUDED`, and
`tests/test_coverage.py` keeps it at 100%. `scripts/diff_torch.py --torch-python <real-torch venv>` compares every
deterministic PyTorch problem with real PyTorch (1,384 agree). Extra page packages: 2.03 jinja2, 2.11 tzdata.

### Session 7 (2026-10-20 → 10-23): Bash, live
Request: "skip scikit-learn for now, and do bash first. ensure everything in comp_guide.md is included for bash."
Decision: instead of the planned shlex `command` checker, build **bash-lite** (`generator/bashlite/`), a simulated
bash running in Pyodide. User code really runs, so the console shows real output, and grading compares stdout (or a
`post` check) between the reference and the user, on the same simulated machine.
- **Written, but untested (no builtins or commands yet, so nothing runs):**
  - `vfs.py`: in-memory filesystem with inodes, symlinks, modes, owners, times and sorted listings.
  - `parser.py`: full bash grammar, incremental `parse_next`, heredocs, aliases, `Incomplete` for interactive input.
    It parses all 189 bash blocks of guide section 7.
  - `syntax.py`: AST.
  - `kernel.py`: generator-based processes, pipes with SIGPIPE, signals, a virtual clock and the TTY.
  - `fileio.py`: read, write and open, including `/dev/fd`.
  - `arith.py`, `pattern.py` (globs, `${x#pat}`, pathname expansion), `quoting.py`.
  - `shell.py`: interpreter with expansions, splitting, redirections, pipelines, functions, jobs and traps.
- **Written since (2026-10-23, still not live):**
  - `builtins.py` (all bash builtins), `exec.py` (shebangs, `/usr/bin` stubs `\x7fELF...BASHLITE:<name>`),
    `regex.py` (BRE/ERE/PCRE → Python `re`), `world.py` + `basefs.py` (World/Host/Session, Ubuntu-like base FS,
    user `ada`, fixed virtual clock), `history.py`, `printf.py`, `timeutil.py`, `unparse.py`, `helptext.py`.
  - `utils/`: coreutils, grep/rg, sed, awk (compiled to Python), find/xargs, jq, bc, tar/zip/gzip/xz, ps/kill/top,
    man/tldr/shellcheck/crontab, ssh/scp/rsync/curl/wget (`net.py`), tmux, and `pyenv.py` (python that really runs
    against the simulated FS, venv, pip with PEP 668, uv, conda with `conda init` hooks, jupyter, tensorboard,
    torchrun, `python -m http.server`).
  - Job notices now follow real bash's timing (checked in a pty): a job that dies during `wait` or a foreground
    command is reported at the next prompt, one killed by `kill` a prompt later.
  - Differential corpora (`tests/bash_corpus/`, `scripts/diff_bash.py`) agree with real bash/GNU tools except jq
    1.8 vs Ubuntu's 1.7.1 wording (bash-lite follows 1.7.1).
- **Finished 2026-10-23 (live):**
  - `utils/gpu.py` (nvidia-smi table/queries/`-l`, nvtop, fuser; Python code that touches CUDA holds GPU memory
    while it runs), `directives.py` (`#:` setup lines: GPUs, date, hosts, URL fixtures, long-running processes, venvs,
    conda, history, script mode ...), `keys.py` (a readline editor in emacs and vi mode with kill ring, history search,
    completion, plus tmux prefix keys), `features.py` (what a solution uses: commands, `cmd -flag`, syntax), and
    `check.py`, the `kind="bash"` grader: setup runs quietly in the user's interactive session, the answer is typed
    line by line, and stdout (or stderr, exit status, or a `post` check) is compared with the reference solution's.
  - `generator/sections/s07_bash/`: 170 templates for 7.01–7.17 (≥ 2 per tier, 2 challenges per module), ~500
    distinct tasks. Every curated guide feature (469: commands, flags, syntax, 7.17 keys) is practised
    (`scripts/guide_coverage.py bash`, enforced by tests/test_coverage.py). Every challenge is cross-checked
    against an independent Python brute force (tests/test_bash_challenges.py).
  - Frontend (replaced in session 8 by the live terminal): `#:` lines hidden, a terminal transcript in the
    console, shell highlighting, plain-text editor for keystroke answers.
  - Lesson: a simulator bug can make the reference and the answer fail identically, so "the reference passes" proves
    little. The templates were reviewed transcript by transcript, and a scan flags reference solutions that print
    errors; that found a dozen simulator bugs (e.g. `true | sleep 1` spinning, `a[$k]=v` not parsed as an assignment,
    `set -euo pipefail`, fd leaks with `2>&1 >/dev/null` in pipelines, sed on empty input).
- **Reference behaviour checked against real bash 5.2.37** (assoc-array order is FNV-1 hash bucket order, xtrace,
  `type`/`declare -f` printing, `printf`, `read`, `getopts`). Recreate the reference tools with micromamba:
  `micromamba create -p <dir> -c conda-forge bash coreutils sed gawk grep findutils diffutils tar gzip jq bc xz zip unzip file`.

### Session 8 (2026-10-30): the Bash terminal
Request: replace the code box on the Bash pages with "an actual box to emulate the font, functionality, of an actual
Linux terminal ... with the fonts and styles of a mac m1 cli"; keep the files people create in the browser (with
safeguards); check that a task's commands were all used, in any order; a directory-tree display beside it, reset per
problem; more hands-on problems (build this tree from `~`, turn tree A into tree B ...).
- **The look** is Terminal.app's Basic profile, read from the local profile plist (SF Mono, block cursor, 80×24,
  the Basic light and dark colours), in a window with the title bar and traffic lights; the title follows
  Terminal's `dir — process — cols×rows`. SF Mono can't be served as a web font (licence), so the font stack is
  SF Mono / ui-monospace / Menlo, then JetBrains Mono (Google Fonts) elsewhere. Rendering: xterm.js 5.5 (DOM
  renderer), Option as Meta; ⌘K clears, ⌘+/− resize the font, ⌘. is Ctrl-C, ⌘Enter submits.
- **One engine for typing and grading** (`generator/bashlite/live.py`, `LiveTerm`): the page sends keys; the
  shell's line editing is keys.Editor (the same readline the 7.17 grader uses), programs get a cooked-mode line
  (echo, Ctrl-C/Z/D/\\), and the machine runs on the real clock in slices (`Kernel.run_slice`), so `sleep 3` takes
  3 s and `tail -f` runs until Ctrl-C. Ctrl-C now does what bash does: it abandons the running command line
  (`Interrupted`), including loops of builtins, `read` and `wait`; at the `>` prompt it drops the unfinished command.
- **nano** (`utils/editor.py` + the page's `Nano` in `docs/terminal.js`): GNU nano 7.2's screen on xterm's
  alternate screen, with ^O/^S/^X/^K/^U/^W/^C; saves (whole text, maybe a new name) go to the machine. vim and
  emacs say "edit files with nano instead".
- **Every key is logged** with where it went: the prompt, a running program (with the ms since the command
  started), or nano. The log is kept per module in localStorage (capped at 1.5 MB, oldest sessions evicted), so a
  reload rebuilds the session by replaying it at the recorded times. Safeguards: 3000 inodes and 8 MB of file data
  per session (`VFS.set_quota`: commands fail with "No space left on device"), 20,000 log events, bounded
  transcripts; a command that hangs the worker is dropped from the log and the session rebuilt without it.
- **Grading a session** (`check.term_checker`): the log is replayed on the visible machine and on every hidden one,
  as fast as the commands allow (prompt keys as soon as the prompt is back, keys for a running program at the same
  moment of that command), so thinking time doesn't change `ls -l` times. Everything typed counts for
  require/forbid, in any order. The expected output must come from some commands in a row (exploring first is
  fine); if it doesn't, the reference is retyped at each of the last 12 commands of the session, so exploring that
  shifted process ids, history or the directory still passes. A command stopped with Ctrl-C passes with a prefix of
  the output. `post` checks run in the session's shell afterwards; script problems are written in the terminal
  (nano) and run by the grader. 7.17 sessions are graded like keys answers (commands run, line left, tmux state).
  Typed-text answers (tests, the old path) still work.
- **Files panel** (right column): the problem's start directory, live, with folder/file/executable/link icons,
  "you are here", new items in green, deleted ones struck through, hidden files behind a toggle.
- **New 7.02 problems**: build a tree from `~` (lite: `mkdir`/`touch`; easy: `mkdir -p`), find your way through a
  maze with `cd`, reshape/reorganize a project (tree A → tree B with `mkdir touch rm rmdir mv cp cp -r rm -r`, each
  required, any order; contents checked by md5), and clean-up tasks that also run on hidden trees. Trees are drawn
  `tree -F` style in the prompt; a test checks each drawing against what the reference builds.
- Fixed on the way: aliases inside `$( )` in double quotes broke parsing (the parser rewrote its text in place);
  comment-only lines left the shell at the `>` prompt; Ubuntu's `.bashrc` colour prompt and `ls`/`grep` aliases;
  `tree -F`.
- Tests: `tests/test_bash_terminal.py` (every template's reference as a session, exploring first, wrong and empty
  sessions, Ctrl-C, nano, quotas, reload). The in-browser self-test grades bash solutions as sessions.

### Session 9 (2026-10-30): Instructions instead of examples; the solution after a wrong answer
Request: "instead of long, often unrelated example problems, change it to an Instructions dropdown to show the user
what functions they are intended to use and a description of that function ... only list the new functions that are
recently introduced"; and "when Wrong is displayed, show the correct solution (labeled Solution: ...)".
- `generator/instructions.py` reads the names a reference solution uses, in order: APIs as written (`np.cumsum`,
  `.reshape`, `F.softmax`), indexing and operator syntax from the AST (`a[:, None]`, masks, `@`, `len(df)` …), and for
  Bash the features (`grep -c`, `|`, `${var%pat}`, `[[ -f`, 7.17 keys). It lists **only what is new**: the names
  introduced at this problem's module and at its tier or the one before; else every name the module introduces;
  else (a problem that only reuses older names) the two most recent. At most 4. "Introduced" = the first module whose
  COMP_GUIDE code lists the name (else whose solutions use it), and the first tier there whose solutions use it.
  A Bash command and its option read as one entry (`ls -l`: both descriptions).
- `generator/data/instructions.json` (~340 KB, in the bundle): names, where they are introduced, and descriptions.
  Built by `scripts/make_instructions.py --python <venv with numpy, pandas, torch, matplotlib>`: the real libraries'
  docstrings (first sentence; signatures cut to 3 parameters; "Alias for …" followed), bash-lite's manual pages and
  `help` for commands, options and builtins, and `scripts/instruction_notes.py` (syntax, constants, keys, Bash
  syntax, commands without a manual page; an empty note hides a name). Rerun it after adding templates; names it
  can't describe simply aren't listed.
- The page: an **Instructions** dropdown (open for questions 1–10, like the example was), and after ✗ Wrong the
  reference solution under **Solution:** in the verdict panel. The example (a second instance of the template) is
  gone, which also halves generation work.
- Tests: every problem lists ≤ 4 described names, under 3% of problems list none, and a problem never mixes new and
  older names (tests/test_templates.py); e2e checks the dropdown and the Solution box.

### Session 11 (2026-10-30): Part III, Data Structures & Algorithms
Request (before scikit-learn, Hugging Face, …), condensed with every requirement kept:
- **A new Part III.** Step-by-step guides that teach beginners the basic data structures and algorithms. Unlike the
  100-problem sections, a guide walks the student through the exact steps to build one structure. The linked list,
  for example (default null pointers), goes: the node with data and a pointer to next → contracts before writing the
  code → insert at the start → insert at the end → delete head → delete node → lookup → traverse/print. Each step can be
  broken into smaller steps, down to concise micro details: **very concrete, literal steps, not high-level ideas.**
- **Status labels.** On the main page each tutorial shows **Not Started / In Progress / Almost! / Completed** instead of
  a count.
- **Languages.** Python, C++, Java and Rust. Unlike LeetCode there are **no templates** (no `class LinkedList` stub):
  the student writes everything from scratch.
- **The 12 tutorials:** linked lists, stacks, queues, unbounded arrays, hash tables, binary search trees, dictionaries
  and sets, heaps, priority queues, binary search, merge sort, quicksort.
- **Options before starting.** Every tutorial lets the student pick the element type: one specific type, or generic
  (`void*` in C, templates and generics elsewhere). Then options per tutorial:

  | Tutorial | Options |
  |---|---|
  | Linked lists | NULL-terminated or dummy node; singly or doubly; circular or not |
  | Stacks | via linked lists |
  | Queues | via linked lists or two stacks |
  | Unbounded arrays | the resizing mechanism |
  | Hash tables | buckets (separate chaining), or open addressing with linear or quadratic probing |
  | Binary search trees | plain, AVL or splay |
  | Dictionaries and sets | on BSTs or on hash tables |
  | Heaps | — |
  | Priority queues | — |
  | Binary search | iterative or recursive |
  | Merge sort | one way: a merge function plus recursive mergesort |
  | Quicksort | one way: a partition function plus quicksort |

- **Live complexity boxes.** While choosing, boxes show the complexity overall and per operation (insert, lookup, find
  minimum, …), updating in real time. Operations a structure doesn't have are left out.
- **Conventions.** Segments are `[lo, hi)`: `lo` included, `hi` excluded. Variables are the structure's capital first
  letter: array `A`, queue `Q`, stack `S`, hash table or dictionary `H`, list `L`.
- **The final challenge.** When every step is done the tutorial moves to **Almost!** and asks the student to implement
  the structure once more with no hints: only the problem and a completely blank code box. Passing it gives
  **Completed**. Reset/restart at any point, with a confirmation.
- **Plan.** Research all of them first (the best way to build each, always preferring the optimal implementation),
  write it into PROJECT_SPEC and COMP_GUIDE, and **build linked lists first**. The rest show "soon".

**Decisions.**

| Decision | Choice | Why |
|---|---|---|
| Where code runs | Python: in the page's Pyodide worker (`generator/api.py: dsa_run`). C++, Java and Rust: **Compiler Explorer** (godbolt.org) `POST /api/compiler/<id>/compile` with `executorRequest`. Compilers: GCC 15.2 `g152` (`-std=c++20 -O1`), JDK 25 `java2500`, rustc 1.94 `r1940` (`--edition 2021`). | GitHub Pages is static. CE's API accepts requests from any origin (checked: `Access-Control-Allow-Origin: *`) and executes all three. There is no in-browser rustc. Options weighed: in-browser clang (`@yowasp/clang`, ~25 MB) would cover only C++; Java in the browser (CheerpJ) is free only for non-commercial use and can't be self-hosted. **The user's C++/Java/Rust code is sent to godbolt.org**, and the page says so. |
| One source, four languages | Each function is written once as micro-steps in a tiny statement language (`docs/dsa/lang.js`: `n.next = L.head`, `if L.tail == null`, `let x = take n`, …) and rendered to Python, C++, Java and Rust. | The step text, the reference solution, the tests' helpers and the tests' `main` all come from it, so they can't disagree. 8 layouts × 2 element types × 4 languages = 64 variants of every step. |
| API style | Free functions that take the structure: `insert_front(L, x)`; Java `insertFront(L, x)` as static methods inside `class LinkedList`; C++ `LinkedList& L`. | The requested naming convention (`L`), 15-122/CLRS style, and every step reads the same in all four languages (`L.head`; node fields are `n->next` in C++ and `(*n).next` in Rust). |
| Generic elements | C++ `template <typename T>`, Java `<T>` (comparisons with `.equals`), Rust `<T>` with only the bounds used (`PartialEq`, `Display`, `Default` for dummies), Python `class Node[T]` | Native, type-checked, zero-cost. `void*` (C1 style) is explained but not used. Generic tests use `String` values, and Java's `new String(...)` catches `==`. |
| Rust pointers | Raw pointers (`*mut Node<T>`) inside `unsafe { }`; nodes via `Box::into_raw`, freed with `Box::from_raw` | Tail pointers, `prev`, dummies and cycles share nodes; `Box` can't. `std::collections::LinkedList` does the same. |
| Linked-list layouts | NULL-linear: `head`, `tail`. NULL-circular: singly `tail` only (`tail.next` is first), doubly `head` only (`head.prev` is last). Dummy-linear: `head` and `dummy`, the list is `[head, dummy)`. Dummy-circular: `dummy` only, `[dummy.next, dummy)`. All keep `size`. | The fewest pointers with O(1) at both ends. The dummy-end lists use 15-122's trick for `insert_back` (write into the dummy, add a fresh dummy); the doubly circular one is CLRS's sentinel. |
| Contracts | `is_segment(start, end, n)`: exactly `n` steps from `start` reach `end`, never meeting null or `end` early, plus `p.next.prev == p` when doubly. `is_list(L)` per layout. `assert is_list(L)` at the start and end of every operation (`debug_assert!` in Rust); `delete_head` also requires non-empty. | The requested "contracts before code", and `[lo, hi)` segments. The checker turns Java's assertions on (`setDefaultAssertionStatus` before the user's classes load). |
| Tests | Each step's tests: the user's file plus a `main` the checker adds. The checker's own helpers (`ref_is_segment`, `ref_insert_back`, `show`) are built on the user's types; `show` prints the contents and the first broken invariant. Checking step k reruns steps 1..k. The final adds a seeded random sequence of 60 operations. The page compares every line and shows the first difference. | Precise feedback ("L.tail.next is not null, so L.tail is not the last node") instead of right/wrong. A user's own `main` is renamed so the checker's runs. Nothing is put before the user's code, so error line numbers match the editor. |
| Status | localStorage `training-loop:dsa:v1`: options, phase (`setup`, `steps`, `final`), steps passed, the code, the final's code and result. Not Started = no options confirmed; In Progress; Almost! = all steps passed; Completed = final passed. | Same storage style as the problem sections. Reset clears it after a confirmation dialog. |

**Built: 13.01 Linked lists** (`docs/dsa/linked-list.js`):

- **Options:** language, int or generic, NULL or dummy, singly or doubly, linear or circular.
- **Complexity boxes:** space, insert at start and end, delete head, delete last, unlink a held node, delete by value,
  lookup, find minimum, k-th, traverse, length, `is_list`. The ones that aren't tutorial functions are faded. Changed
  values flash.
- **A drawing of the chosen layout.**
- **8 steps,** with the final challenge's problem generated from the same interface.

The other 11 are researched in COMP_GUIDE §13 (interface, invariant, options with their complexity, steps, pitfalls)
and listed as "soon".

**Testing.**

- `scripts/dsa_selftest.mjs` runs every reference solution, step by step and as the final: 576 runs with local
  clang++, javac and rustc, and Pyodide's `dsa_run` for Python.
- `--mutants` deletes each statement of every Python reference in turn: all 630 mutants are caught. Strengthening
  this added broken-list cases, `is_segment` walks past the end and around a circle, and printing what a Python
  function really returned (`None` is not `false`).
- `tests/test_dsa.py` covers the Python runner and runs the Python references.
- `scripts/dsa_e2e.py` clicks through the site:
  - home status labels;
  - options updating the boxes and the diagram;
  - all 8 steps and the final in Python, C++, Java and Rust;
  - wrong, empty and non-compiling code, and Run;
  - Reset (cancel and confirm), reload, and the phone layout.

  Its Compiler Explorer requests are answered by compiling locally, so no code leaves the machine during tests.

### Session 12 (2026-10-30): micro-steps as bubbles that follow the typing
Request: the step instructions took so much room that the code box was pushed far down (worst at step 2). Now:
- The top of a step is a **high-level overview** only: "Step 1 of 8 · The node" and the goal paragraph.
- The detailed micro-steps appear as **translucent chat bubbles** while the user types. The page keeps checking (not
  compiling) whether the current micro-step's code has been typed. When it has, the bubble fades out and the next one
  appears; when the user deletes progress, the bubbles go back.

Built (`docs/dsa/page.js`, "the coach"):

| Part | How |
|---|---|
| Targets | A step's micro-steps are flattened into ordered targets (an instruction and its line(s)). Wordless parts (an `else` branch, closing braces) join the target before them. Things with nothing to type (Python and Java free memory themselves) are told with the next target. Java's `class LinkedList` brace is not a target, since it closes after all the functions. |
| Matching | 120 ms after each edit the file's lines are compared, in order, with the targets: whitespace and comments ignored, Python quotes unified, Python type hints optional, `x++` = `x += 1` = `x = x + 1`, `NULL` = `nullptr`, `std::ptr::` = `ptr::`. A function header matches by its name. The bubble shows the first target not yet typed. |
| Bubble | Under the cursor's line (or the last typed line when the editor isn't focused), at the right of the editor. Translucent and blurred, with clicks passing through to the code. It shows "k / n", the instruction and the code. Changes fade out and in. |
| Escape hatches | A "skip ›" button for a line deliberately written another way. The whole list stays under a closed "All micro-steps" dropdown. When every target is typed, the bubble says to press Check; there's no bubble on steps already passed. |

Tested in `scripts/dsa_e2e.py`:
- the first bubble;
- typing advances it, and deleting brings it back;
- "press Check" when the step is done;
- for every layout × language × step (512), the reference code satisfies all of its bubbles, and typing it line by line
  never moves the bubbles backwards.

### Session 13 (2026-10-30): stacks and queues
Request: "now go ahead and implement stacks and queues." Built:

- **13.02 Stacks** (`docs/dsa/stack.js`): a stack on a linked list, the requested option. Options: language and
  int/generic. Steps:
  1. node;
  2. stack and contracts (`is_segment`, `is_stack`, `stack_empty`);
  3. `push`;
  4. `pop`;
  5. `peek` and `stack_size`.
- **13.03 Queues** (`docs/dsa/queue.js`), in two versions.
  - On a linked list with a dummy node (15-122: [front, back)). Steps:
    1. node;
    2. queue and contracts;
    3. `enq`;
    4. `deq`;
    5. `queue_peek` and `queue_size`.
  - On two stacks (`inbox` and `outbox`, reusing 13.02's steps; amortized O(1) `deq`). Steps:
    1. node;
    2. stack;
    3. `push`, `pop`, `peek`;
    4. queue and contracts;
    5. `enq`;
    6. `refill` and `deq`;
    7. `queue_peek` and `queue_size`.

  The options change the steps, the complexity boxes ("Dequeue (amortized)", "one dequeue, worst case" O(1) vs
  O(n)) and the diagram (two rows for two stacks).

**Groundwork for more tutorials.**

| Piece | What changed |
|---|---|
| `docs/dsa/common.js` | Holds what every tutorial shares: step text, the node step, rendering, the reference program (Java functions placed inside their class), test compilation and the harness. A tutorial is a definition passed to `DSACommon.make()`, and linked lists moved onto it. |
| Structures in `lang.js` | Named structures (`R:Stack` reference parameters, `V:Stack` values), so a queue can hold two stacks. Rust borrows a value field as `&mut Q.inbox`. |
| Java calls | A function in another class is called as `Stack.push(...)`. |
| C++ templates | Calls passing `nullptr` name the template argument (`is_segment<T>(S.top, nullptr, S.size)`). |
| `peek` | Rust's generic `peek` returns `&T`. |
| Options | Function specs, structures and the step count may depend on the options. |

**Testing.**
- `dsa_selftest.mjs`: 736 runs (all three tutorials × options × languages × steps, plus the final); 718 mutants, all
  caught.
- `dsa_e2e.py`:
  - stacks in Python (with a wrong step) and C++ generic;
  - queues on two stacks in Python generic and Java, and on a list in Rust generic;
  - the queue's options changing steps, boxes and diagram;
  - the hint-bubble consistency check across every tutorial.

### Session 14 (2026-10-30): Part III reordered; binary search
Request: "move binary search, mergesort and quicksort to 13.01, 13.02, and 13.03, and shift everything down. rename merge
sort without the space -> mergesort. then implement binary search".

**New order.**

| Number | Tutorial |
|---|---|
| 13.01 | Binary search |
| 13.02 | Mergesort |
| 13.03 | Quicksort |
| 13.04 | Linked lists |
| 13.05 | Stacks |
| 13.06 | Queues |
| 13.07 | Unbounded arrays |
| 13.08 | Hash tables |
| 13.09 | Binary search trees |
| 13.10 | Dictionaries & sets |
| 13.11 | Heaps |
| 13.12 | Priority queues |

Where the new order lives:
- The home page reads it from `DSAPage.LIST`. Saved progress is keyed by tutorial id, so the move changes nothing for
  learners.
- COMP_GUIDE §13 and its cross-references are renumbered, and the not-yet-built `merge-sort` id is now `mergesort`.
- Session notes 11–13 above keep the numbers of their time (linked lists were 13.01 then).

**13.01 Binary search** (`docs/dsa/binary-search.js`). Options: iterative or recursive; int or generic. Steps:
1. sorted arrays and their contracts (`is_sorted(A, lo, hi)`, `is_in(A, lo, hi, x)`);
2. the loop invariant (`all_smaller`, `all_bigger`: `A[0, lo) < x < A[hi, n)`);
3. `binsearch(A, n, x)`, with an overflow-safe `mid` and -1 when absent (or `binsearch_rec` plus `binsearch`);
4. `lower_bound(A, n, x)`, after `all_at_least` (or its recursive pair).

In Java the functions are static methods in `class BinarySearch`.

**Arrays in `lang.js`.**

| Addition | Detail |
|---|---|
| Expressions | indexing `A[i]`, array literals in tests, `*`, `/` (`//` in Python), `%`, negative literals, `for i = lo to hi` |
| Parameter kinds | `Arr`: `list[T]` / `const std::vector<T>&` / `T[]` / `&[T]`. `Tref`: Rust `&T`, read as `*x` and passed on as `x` |
| `index` return | an index or -1; Rust `isize`, positions `usize`, so `mid as isize` |
| Ordering on `T` | Java `compareTo` with `<T extends Comparable<T>>`; Rust `PartialOrd`. A function's spec can declare bounds that callers inherit (`bounds: ["PartialOrd"]`). |

The hints skip an empty Java class's closing brace, which comes after all the functions.

**Testing.**
- `dsa_selftest.mjs`: 80 runs and 90 mutants for binary search, all passing and all caught.
- The other tutorials still pass.
- `dsa_e2e.py`:
  - the new home order;
  - the style option changing the space box;
  - binary search in Python (with a wrong step), C++ recursive generic, Rust iterative generic and Java recursive
    generic;
  - the hint bubbles matching their reference code for every tutorial.

### Session 15 (2026-10-30): mergesort and quicksort
Request: "go ahead and do mergesort and quicksort".

**13.02 Mergesort** (`docs/dsa/mergesort.js`). Options: language, int/generic. Steps:
1. sorted segments (`is_sorted`, reused from binary search);
2. `merge(A, lo, mid, hi)`: two cursors into a temporary growable `B`, ties from the left (stable), then copy back;
3. `mergesort(A, lo, hi)`.

**13.03 Quicksort** (`docs/dsa/quicksort.js`). Options: language, int/generic, pivot (middle or first; the boxes show
sorted input at O(n log n) vs O(n²)). Steps:
1. contracts (`is_sorted`, `all_at_most`, `all_bigger`);
2. `swap`: Python tuple, C++/Java temporary, Rust `A.swap`;
3. `partition(A, lo, pi, hi)`: 15-122's two-sided version with its loop invariant;
4. `quicksort`.

**More in `lang.js`.**
- In-place arrays (`ArrMut`: `std::vector<T>&`, `&mut [T]`).
- A growable array: `vec()` and `push B, x`; Java's `ArrayList` reads with `.get`.
- Rust clones a generic element copied out of an array, and adds `Clone` to the bounds.
- A `swap X, i, j` statement for the checker (`java.util.Collections.swap` on lists).
- Steps limited to some languages with `only`. Raw per-language code is allowed inside functions.

**Tests.**
- Partition prints each side sorted, through the checker's own insertion sort, so any correct partition passes.
- The mutation check skips statements that exist only in other languages.
- `dsa_selftest.mjs`: 112 runs and 126 mutants for the two sorts, all passing and all caught.
- `dsa_e2e.py`:
  - mergesort in Python (with a wrong step) and Java generic;
  - quicksort's pivot option changing the boxes;
  - quicksort in Python (with a wrong step), Rust generic with a first-element pivot, and C++ generic.

A diagram label longer than its box now hangs over the next cells instead of widening its column.

### Session 16 (2026-10-31): unbounded arrays
Request: "go ahead and do unbounded arrays (they should be able to choose resizing mechanism, among other options)".

**13.07 Unbounded arrays** (`docs/dsa/unbounded-array.js`).
- Options:
  - language and int/generic;
  - **growth**: ×2, ×1.5 or +4;
  - **shrinking**: at ¼ full, below ½ full (thrashes), or never.
- The complexity boxes follow the options: amortized add O(1) vs O(n), copies for n adds (< 2n / < 3n / ≈ n²/8),
  amortized remove O(1) vs O(n), space O(n) vs O(peak).
- Steps:
  1. struct and contracts (`is_uba`: `size < limit`, `len(data) == limit`; `uba_len`);
  2. `uba_get` and `uba_set`;
  3. `uba_resize`;
  4. `uba_add`;
  5. `uba_rem`.

A first draft had `uba_resize` require `is_uba`, which fails exactly when `uba_add` calls it (a full block); it now
requires `size <= limit`, as in 15-122.

**More in `lang.js`.**
- `alloc(n)`: a fixed block. Python `[0] * n`, C++ `std::vector<T>(n)`, Java `new int[n]` / `(T[]) new Object[n]`,
  Rust `vec![T::default(); n]` (adds `Clone + Default`).
- `len(X)`.
- Rust's generic getter returns `&A.data[i]`.

**Tests.**
- The checker prints `size` and `limit` after every operation, so the growth and shrink policies are checked, against
  a JavaScript model of the chosen policy.
- `dsa_selftest.mjs`: 432 runs and 282 mutants for unbounded arrays, all passing and all caught.
- The self-test now retries a failed compiled run once, on its own: under load an occasional javac or JVM run timed
  out. Compilers get 120 s, programs 60 s.
- `dsa_e2e.py`:
  - the growth and shrink options changing the boxes;
  - Python with a wrong step;
  - Rust generic (×1.5, ½), Java generic (+4, never) and C++.

### Session 17 (2026-10-31): hash tables
Request: "go ahead and do hash tables".

**13.08 Hash tables** (`docs/dsa/hash-table.js`): key → int value.
- Options:
  - language;
  - **keys**: int or string, instead of int/generic, because a hash table needs a concrete hash function;
  - **collisions**: separate chaining, linear probing or quadratic probing.
- The hash function is written by hand: the key mod m, never negative, or a polynomial `h * 31 + code` reduced at
  every step. It gives the same numbers in every language, so the tests print the exact table and check it against a
  JavaScript model.
- Chaining: chains of `Entry {key, value, next}`, front insertion, doubling when `size > capacity`.
- Open addressing: tombstones (`H.tomb`), `used` counting entries plus tombstones, doubling when
  `used * 2 > capacity`, capacity a power of two (so the triangular quadratic steps reach every slot).
- Steps:
  1. table and hash (and `ht_probe`);
  2. contract (`is_ht`, `ht_size`; `ht_find_slot` first for open addressing);
  3. lookup;
  4. resize;
  5. insert;
  6. delete.

**More in `lang.js` and `common.js`.**
- A tutorial can name its own node type (`Entry` with `key`, `value`, `next`) and its own element type (string keys,
  not generic) through `ctxExtra`.
- `new(k, v)` with several arguments.
- `allocn(n)`: a block of null pointers.
- Functions returning an entry (ret `N`).
- A value kind `V` (Rust `i32`, where positions are `usize`).
- Rust casts an index (`isize`, maybe -1) for indexing and comparisons.
- Java compares string keys with `equals`.

**Mutation check.**
- It first left 18 mutants alive. Strengthening the tests caught them:
  - the open-addressing invariant became "`ht_find_slot` finds each entry at its own slot" (simpler, and it also
    catches duplicate keys);
  - a capacity that doesn't match an empty table;
  - a resize after tombstones, checking that `used` resets;
  - the cut-off-entry break, which is now always generated.
- Steps marked `noMutate` are skipped: Python's implicit `return None`, and a final `return -1` that a half-empty
  table can't reach.
- 252 mutants, all caught; 168 runs, all passing.

### Session 18 (2026-10-31): the Contracts option
Request: "create an option in each datastructure/alg tutorial to toggle contracts. if this option is unselected
(untoggled is default), then the user need not implement things like representation invariants and such, and straight
up practice implementing the data structure itself and its functionalities".

**Built.** `DSACommon.make()` adds a **Contracts** option (off | on, default off) to every tutorial. Off means:
- Function specs marked `contract: true` disappear, and so does every `assert` statement.
  - The marked functions: `is_list`, `is_segment`, `is_stack`, `is_queue`, `is_uba`, `is_ht`; binary search's
    `is_sorted`, `is_in` and `all_*`; the sorts' `is_sorted` and `all_*`.
- A step left with nothing to write is dropped, and its declarations move into the next step (e.g. binary search's
  setup into `binsearch`). Its tests leave the later steps' lists.
- Test groups with a `break` op are dropped. Ops a tutorial lists in `CONTRACT_OPS` (is_list, segment, …) are removed.
- The checker's own helpers (`ref_is_list`, `show`'s BROKEN messages) stay, so structure errors are still reported.
- Complexity boxes with ids `is_*` or `contracts` are hidden.
- The final problem drops the contract functions and its "Check …" sentence.
- Step texts that talk about contracts have a no-contracts wording (via `C.contractsOn(o)`, and `ctx.contracts` in
  step text).

**Steps without contracts.**

| Tutorial | Steps without | Steps with |
|---|---|---|
| Binary search | 2 | 4 |
| Mergesort | 2 | 3 |
| Quicksort | 3 | 4 |
| Linked lists, stacks, queues, unbounded arrays, hash tables | same | same |

The list, stack, queue and table steps stay, keeping their struct and `is_empty` / `size`.

**Old saves.** Progress saved before the option existed (phase past setup, no `contracts` in the saved options) is read
as contracts on, so step counts don't shift for anyone mid-tutorial.

**Testing.**
- `dsa_selftest.mjs`: every option combination now includes both modes: 3000 runs, 0 failures. Mutants: see the
  release.
- `dsa_e2e.py`:
  - the option on all 8 tutorials, off by default;
  - binary search 2 ↔ 4 steps, and the contracts box;
  - the setup moving into the first remaining step, and a final without contract functions;
  - full no-contracts runs: binary search in C++, hash tables in Rust, quicksort in Java, the two-stack queue in
    Python, unbounded arrays in C++.

  The earlier flows run with contracts on (`start()` defaults to on).

### Session 19 (2026-10-31): binary search trees
Request: "go ahead and do binary search trees (default, AVL, splay)".

**13.09 Binary search trees** (`docs/dsa/bst.js`).
- Options:
  - **tree**: plain, AVL or splay;
  - int or generic;
  - language and Contracts.
- The tree variable is `B`, because `T` would clash with the type parameter.
- Node functions are recursive and return the subtree's new root.
- Steps:
  - plain: tree, contracts, lookup, insert, min/in-order, delete;
  - AVL adds height/fix_height, rotations and rebalance;
  - splay adds rotations and splay, and does lookup, insert and delete by splaying.
- The checker prints the tree's shape (with heights for AVL) after every change, against a JavaScript model of each
  variant.

**More in `lang.js`.** A generic Rust element copied out of a node field is cloned (`n.data = m.data.clone()`, adding
`Clone`), as array slots already were.

**Mutation check.**
- It first left 68 alive. It now deletes nodes of every kind, chosen from the model's shape.
- Each broken tree trips exactly one ordering check.
- AVL heights are printed, and a correct-heights-but-unbalanced tree is tested.
- Splay tests reach both null-child guards.
- Two splay checks that could never be true were deleted. A splay-insert `return` and the AVL one-child delete branch
  are marked `noMutate` (provably equivalent).
- 696 mutants, all caught; 376 runs, all passing.

### Session 20 (2026-10-31): saving progress to a file
Request: progress lives only in the browser, so clearing its data loses it. Add Download and Upload buttons to the
top right of the main page. Choose the best file format. Download should open the system's Save dialog so the file
can be renamed. Upload should accept only that file type.

**Format: `.tlsave`, UTF-8 JSON** (`docs/progress.js`).
```
{ "format": "training-loop-progress", "version": 1, "savedAt": "<ISO time>", "site": "<page URL>",
  "stores": { "<localStorage key>": <its value, parsed>, ... } }
```
- `stores` holds every `training-loop:*` key except the retired `v1` counter:
  - Part I/II modules (`v2`);
  - Part III tutorials (`dsa:v1`);
  - the terminal font size;
  - open Bash terminal sessions (`term:<module>`).
- New stores are included without code changes. Each store's key carries its own version, so the envelope's
  `version` changes only if the envelope does.
- The file is written one store per line: readable and diffable, but terminal logs don't explode into thousands of
  lines.
- Why JSON with its own extension, rather than gzip or a binary format:
  - progress is a few kilobytes;
  - a person (or a future version) can read and repair the file;
  - the extension lets the picker filter on it and tells the file apart from any other `.json`.

**Download.**
- A small box asks for the file name, suggesting `training-loop-progress-<date>`. The file is then an ordinary
  download, to the downloads folder, or to wherever the browser asks if it is set to ask where to save.
  - Path and reserved characters become `-`.
  - A typed `.tlsave` isn't doubled.
  - The blob is `application/octet-stream`, so Safari doesn't append `.json`.
- At first, Chrome and Edge used the system Save dialog (`showSaveFilePicker`). It was dropped the same day: that API
  gives the site edit access to the chosen file, and Chrome shows a warning that the site can see edits made to it.
  The site should never have access to anything on the person's disk.
- Upload uses a plain file input, which lets the page read only the one chosen file. The browser test checks that
  neither button ever calls the File System Access API.

**Upload.**
- `<input type=file accept=".tlsave">`. The content is what's checked:
  - the format marker;
  - the version: a newer one says to reload;
  - every known store's shape, since the pages read stores without checking.
- Keys that aren't the site's are ignored. A refused file changes nothing.
- The confirmation box compares this browser with the file (problems solved, medals, tutorials completed and started).
  - **Merge** (the default): per module and per tutorial, whichever is further along wins whole. A tie keeps this
    browser's, with its draft. A terminal session is kept only if it belongs to the winning record's current problem.
  - **Replace**: the browser matches the file exactly.
  - With no local progress there is a single **Restore**.
- Then the page reloads, because both `app.js` and `dsa/page.js` read storage once at load, and a toast says what
  happened.
- Progress is written first. If storage is full, terminal sessions are the part that gets dropped, and the toast says
  so.

**Tests.** `scripts/progress_e2e.py`:
- Download through the name box, with the file compared with storage; no File System Access calls.
- Restore, Merge, Replace and Cancel, and the pages reading restored progress.
- Rejected files: not JSON, not a progress file, a newer version, damaged.
- Phone and dark layouts.

### Session 21 (2026-10-31): the coach waits for whole headers; dictionaries & sets, heaps, priority queues
Request:
* Fix the hint bubble moving on as soon as `def binsearch()` appears. The editor closes brackets as they're opened, so the
  bubble must wait until the header is done (its colon, or brace).
* Then build dictionaries and sets, heaps and priority queues, after planning them properly.

**The coach.**
* A function's header is matched by the function's name, so the parameters may be named differently.
* It now counts only once it's whole:
  * as many parameters as the reference's, counted at the top level of the brackets;
  * the `:` (Python) or `{` (C++/Java/Rust) after them.
* `scripts/dsa_e2e.py` types every function header of every tutorial, option and language partially (`f()`, and the
  header without its `:`/`{`) and checks the bubble stays put: 3078 headers.

**13.10 Dictionaries & sets** (`docs/dsa/dict-set.js`).
* Options: built on a hash table (chaining) or a plain BST; int or string keys.
* `dict_contains`, `dict_lookup`, the sets, the set operations and the counting client are identical code on both
  implementations: the abstraction boundary.
* Steps: structure, contracts, lookup and printing, insert, delete, sets, set operations, client (8).

**13.11 Heaps** (`docs/dsa/heap.js`).
* Options: min or max (only `ok_above` changes) and 0- or 1-indexed (only `parent`/`left`/`right` and the bounds
  change).
* Fixed capacity: `Heap(capacity)`.
* Steps: the heap, order and swap, contracts, sift up and add, sift down and remove, heapify and heapsort (6).

**13.12 Priority queues** (`docs/dsa/priority-queue.js`).
* Options: binary heap, sorted list or unsorted array, the classic trade-off in the boxes.
* Entries carry a sequence number, so equal priorities are first in, first out.
* Arrays start at 4 and double.
* Steps: entries, contracts, add, peek, remove (5).

**Changes to the statement language (`lang.js`):**
* `make Heap H(n)`: constructors with arguments.
* Rust clones a `String` key copied out of a node or array, including into a by-value argument, and never clones a
  pointer.
* Java orders strings with `compareTo`.
* C++ and Rust spell generic entry arrays `Entry<T>*`.
* A function whose signature never mentions the element type, like `parent(int i)`, isn't made generic: nothing could
  infer its `T`.
* Every program the nine earlier tutorials generate (3376, all languages and options) was compared before and after:
  byte-identical.

**Generic Java arrays.**
* A generic heap's block is `(T[]) new Comparable[n]`.
* `<T extends Comparable<T>>` methods read a `T[]` field as a `Comparable[]`, and an `Object[]` fails that cast
  (checked with javac).
* The checker never indexes `H.data` from its concrete-typed `main`.

**Mutation check.**
* Heaps: the size check survived because a non-full heap's placeholder already broke the order. It's tested on a full
  heap now.
* Priority queues: the list's `n > size` guard could never fire, since a loop in a strictly ordered list fails the
  order check first. It was deleted.
* Dictionaries: added 13.08's "capacity on an empty table" test and 13.09's "only child with a child" deletes.

### Session 22 (2026-10-31): Vim (9.01–9.12) on bash-lite
Request:
* Build Part II's Vim section on bash-lite.
* The user must not be able to leave Vim for the shell, except in problems that practise quitting.
* Each exercise detects the Vim keys the user typed, with problems designed for each section.
* Cover every section thoroughly: by the end, people should handle Vim like a competitive programming pro.

**Design.**
* **Vim is a bash-lite program** (`generator/bashlite/vim/`, run as `vim`/`vi`/`view`), written in Python like the
  rest of the machine. The live terminal (live.py) sends it every key while it is open, and it draws its screen
  into xterm.js with escape sequences on the alternate screen.
* The same engine runs live, on reload (log replay) and in the grader, so the reference keys and the user's keys see
  identical behaviour.
* **Behaviour target:** Ubuntu's vim 9.1 without a vimrc: defaults.vim (scrolloff=5, incsearch, nrformats=bin,hex,
  backspace=indent,eol,start, showcmd, ruler), plus the Python and text filetype settings.
* **Tests:** a differential corpus against real Vim 9.1 (`vim -u DEFAULTS -s keys`), in the same way bash-lite was
  checked against bash: `scripts/diff_vim.py`, `tests/vim_corpus/`.
* **Pages:**
  * Vim problems start inside Vim on the problem's files (`#: vim file …`). `:q` and friends are refused there.
  * 9.01's quitting problems start at the bash prompt, so `vim +42 f`, `vim -O a b` and `view f` are typed, and
    quitting returns to the shell.
* **Grading** replays the key log and compares the end state with the reference keys' end state: the buffers'
  text, the files on disk, the cursor (for motion drills, where the target cell is highlighted on screen),
  windows/tabs, folds, marks, options, and whether Vim quit and what it saved.
* **Key checks:**
  * `require` and `forbid` use canonical Vim commands the engine records as they run (`ciw`, `dd`, `.`, `:%s`,
    `q`, `@`, `<C-v>`, `i_<C-w>` …).
  * `max_keys` is a VimGolf-style keystroke par.
  * The terminal shows a live keystroke count and whether the buffer already matches the target.
* **Templates:** `generator/sections/s09_vim/` m01–m12, at least 2 per tier per module. Every key and command in
  COMP_GUIDE §9 is practised (guide coverage).

**Built.**
* **Engine** (`generator/bashlite/vim/`, ~10k lines):
  * normal / insert / visual / replace / command-line modes, counts, registers, marks, the jump list;
  * undo trees with `:earlier`/`:later`/`g-`/`g+` and `undofile`; the dot command; macros; mappings with timeouts;
  * `:s` with every flag, `:g`/`:v`/`:normal`, ranges, filters through bash-lite (`:%!python -m json.tool`, `:r !cmd`,
    `:w !sudo tee %`);
  * windows, tabs, buffers, folds, netrw, quickfix (`:vimgrep`/`:cdo`), `:terminal`, spell checking, `:help` built
    from the command catalog (`vim/catalog.py`: every §9 command with a one-line description, the Instructions'
    source);
  * Python, YAML and brace-language indentation ported from Vim's runtime (autoload/python.vim, indent/yaml.vim,
    a cindent sketch) with `indentkeys`.
* **Differential tests:** 631 cases against real Vim 9.1 (`scripts/diff_vim.py`, recorded in `tests/vim_corpus/`).
  `scripts/diff_vim_templates.py` also runs every exercise's reference keys in real Vim, and all 732 comparable runs
  (two seeds) match. Skipped are those that:
  * start at the shell, quit, or call shell commands;
  * travel undo by time;
  * use netrw (not loaded with `-u DEFAULTS`), the system clipboard, or spelling suggestions picked by number.
* Things measured against real Vim along the way:
  * Python's ftplugin uses `tabstop=4` in 9.1, and `autoindent` comes before indentexpr.
  * When a jump scrolls the window: the cursor is centred once it is `height/2 + 2` lines past the last line shown,
    or `height/2 - 1` above topline + 'scrolloff'.
  * `v$` then `o` keeps the line break selected.
  * 'undofile' history comes back after `:bd` and `:e`.
* **Live terminal** (live.py):
  * `#: vim args` opens Vim before the session starts; `#: vimquit` allows quitting.
  * `#: vimtarget L C` highlights a cursor target; `#: vimprep SECONDS keys` types earlier edits, so there is undo
    history to travel through (Vim's clock runs ahead by that much).
  * The page sends keys, and Vim's frames go to xterm's alternate screen.
  * A bar under the window shows the keys pressed, the par, and whether Vim already matches the target.
  * On Windows/Linux, Alt-W/N/T send Ctrl-W/N/T, which the browser keeps for itself.
* **Grader** (`bashlite/vimcheck.py`):
  * replays the log, types the reference into a fresh machine, and compares what `vim["check"]` names: text, files,
    quit, cursor, line, topline, layout, sizes, buffers, regs, marks, folds, options, maps, mode, bash_vi;
  * a missing required command, a forbidden one, or keys over par is "Not yet" (Reset and retry); a wrong result is
    ✗ with a message naming the first difference.
* **121 templates** (25 lite, 24 per other tier), each a catalog of drills. `scripts/vim_templates.py` checks every
  item: the reference passes and an idle session fails. `tests/test_vim_section.py` covers the same, plus guide
  coverage, blocked quitting, and replay. `scripts/vim_e2e.py` types each module's reference keys into the real page.

### Session 23 (2026-10-31): LaTeX (11.01–11.18) on latex-lite
Request: build the LaTeX section from the guide. It needs an embedded display of what the source compiles to, and a
**Compile** button (instead of Run) next to Submit.

**Decision: latex-lite, not a TeX engine in WebAssembly.** pdfTeX in WebAssembly (SwiftLaTeX, BusyTeX) needs
100+ MB of TeX Live per visitor and seconds per compile, worse with TikZ, pgfplots and Beamer. It also can't run in
the Python grader or tests. So, as with bash-lite and the Vim engine, `generator/latexlite/` interprets LaTeX in
Python (it runs in Pyodide):
* **Pipeline:** tokens (category codes, comments, verbatim) → macro expansion (\newcommand, \renewcommand,
  \DeclareMathOperator, \newenvironment, \newtheorem, \def) → a document model (headings, paragraphs of styled
  runs, lists, display math, floats, tabulars, theorems, algorithms, listings, boxes, TikZ, pgfplots, Beamer
  frames, bibliography) → HTML/SVG.
* **The page looks like the PDF:** sizes in pt from the article class (10/11/12pt), Computer Modern (KaTeX's
  fonts), the paper and geometry margins. Section, equation, figure, table and theorem numbering, footnotes at the
  page bottom, the title block, abstract, table of contents. Math is typeset by KaTeX in the page; equation
  numbers and labels are resolved by latex-lite (all references resolved, as latexmk does). Beamer frames become
  slides, one per overlay step. TikZ and pgfplots become SVG.
* **Packages matter:** a command works only if its class or package is loaded (\mathbb needs amssymb, \toprule
  booktabs, align amsmath, \cref cleveref …), as in real LaTeX.
* **Errors use TeX's wording and line numbers:** `! Undefined control sequence. l.12 \textbff`, `! Missing $
  inserted.`, `! LaTeX Error: Environment … undefined`, `File `x.sty' not found`, mismatched \begin/\end, extra
  `}`, a misplaced `&`, citation/reference warnings. Like TeX, it recovers and keeps going, so the page still shows.
* **Grading** (`latexlite/check.py`) compares the user's compiled document model with the reference's, not the
  source. Equivalent markup passes (`\textbf{x}` / `{\bfseries x}`, `x^2` / `x^{2}`): text runs with their
  styles, canonical math, numbering, lists, tables, floats, figures. `require`/`forbid` name commands or
  environments used. A compile error is "Doesn't compile" (the problem is kept).
* **Problems** edit the body (the preamble is given and shown), a whole document, or just the preamble. Some
  start from given code (Troubleshooting: fix this). Given files: references.bib, CSV data, images. Many show the
  target as a compiled preview.
* **Page:** CodeMirror's stex mode; Compile shows the compiled page (and the log) beside the editor; Submit grades.
  Errors list Overleaf-style above the page (file:line, the message, the source context, a hint); the log is
  pdfTeX's. Pages and slides zoom to fit the panel (phones too).
* **180 templates** in `generator/sections/s11_latex/` (10 per module, 2 per tier), each a catalog of drills. Answer
  modes: the body, a whole document, the preamble, or references.bib. Templates that recreate a block of the guide
  from memory (its SGD algorithm, its talk, the equation gallery) are one fixed text on purpose (`@fixed`); every
  other template varies by seed. Every command in latexlite/catalog.py (the Instructions) is practised.
* **Checks:** `scripts/latex_templates.py` (every item: the reference passes, an empty or starter answer fails;
  `--katex` typesets every target formula with KaTeX in Node); `tests/test_latex_section.py` (the same, catalog
  coverage, pdflatex's error wording, page rendering, grading messages); `scripts/latex_e2e.py` (every module in
  the browser: Compile, the PDF, Submit; errors and the log; reload; phone); `scripts/e2e_check.py` self-tests
  every template in Pyodide.
* **Faithful details** checked along the way: \label before \caption takes the section's number; \textbf is
  \long (an unclosed brace reads to the end of the file); a math-only command in text gives "Missing $ inserted.";
  \include clears the page before and after; \newpage on an empty page does nothing; a twocolumn article is
  410pt wide and fills the left column first; parboxes (minipage, p{} cells) have no paragraph indent;
  algpseudocode puts \Require/\Ensure in the label column.

### Session 24 (2026-11-06 → 11-07): SQL (10.01–10.16) on postgres-lite
Request: build the SQL section like the other developer-tools sections: a live psql, problems from COMP_GUIDE §10.

**Decision: postgres-lite, not SQLite and not PGlite.** The guide teaches PostgreSQL (ILIKE, DISTINCT ON, FILTER,
JSONB, generated columns, MERGE, EXPLAIN, `\d`). SQLite answers differently (types, NULL ordering, integer
division, no EXPLAIN plans worth reading); PGlite (Postgres in WebAssembly) is ~3 MB per visitor plus a separate
runtime, can't run in the Python grader or tests, and has no psql. So, as with bash-lite and latex-lite,
`generator/postgreslite/` implements PostgreSQL 16 in Python:
* **Engine:** lexer and parser after gram.y (precedence, every statement the guide uses), a type system and
  overload resolution driven by PostgreSQL's own catalog (`pg_catalog.txt`: 487 functions, 396 operators, 229
  casts, 95 types; func_select_candidate / select_common_type), numeric scale rules, float8 shortest output,
  intervals, timestamps and time zones, arrays, JSON/JSONB (key order, `->`, `@>`, jsonb_set ...), regexes,
  to_char/to_date, aggregates (Youngs–Cramer for float statistics), windows with frames, grouping sets, recursive
  CTEs, LATERAL, set operations, DML with RETURNING, ON CONFLICT and MERGE (rows move to the heap's end as in
  PostgreSQL), constraints and foreign-key actions, sequences, identity and generated columns, views and
  materialized views, schemas, transactions with savepoints and the aborted state (25P02), COPY.
* **psql 16:** prompts (`shop=>`, `shop-*>`, `shop!>`), the scanner (quotes, dollar quotes, `\` commands mid-line),
  the printer (aligned, expanded, unaligned, csv, `\pset`), errors with LINE and caret, `\d` family, `\l`, `\dn`,
  `\df`, `\x`, `\timing`, `\copy`, `\i`, `\o`, `\e` (opens nano), `\set`/variables, `\h`, `\?`, `-c/-f/-v/-A/-t`.
* **EXPLAIN** after costsize.c / subselect.c / selfuncs.c: Seq / Index / Index Only (heap fetches until VACUUM) /
  Bitmap scans with btree sizes including deduplication, hash joins with PostgreSQL's cost formula (inner-unique and
  semi/anti factors, bucket statistics), nested loops over parameterized index scans, eqjoinsel with MCV lists,
  EXISTS / NOT EXISTS / IN pulled up into semi / anti joins (or the subquery made unique, or a plain join when it is
  unique anyway), InitPlans and (hashed) SubPlans numbered as PostgreSQL does, CTE inlining, quals pushed into
  subqueries, aggregates, windows, sorts, limits; EXPLAIN ANALYZE runs the query (actual rows, simulated times).
* **Checked against a real PostgreSQL 16.2:** `scripts/sql_corpus.py` replays `tests/sql_corpus/*.sql` (results,
  errors, the guide's statements, plans) against the recorded server output; `scripts/sql_templates.py --oracle`
  runs every template's database and reference on a real server (389 of 402 problems compared; byte-identical except
  that ARM FMA can change a float's last digit, and plan costs within 2%).
* **The terminal:** psql runs as a bash-lite program (`#: psql shop` in a setup opens it; `\q` is off except in
  10.15, which starts at the bash prompt). The page shows a Database panel (tables, columns, row counts, indexes;
  changed tables green) and a bar under the terminal with the last result's shape and whether it matches the target.
* **Grading** (`bashlite/sqlcheck.py`): the session is replayed on the shown database and on hidden ones built by
  the template with traps (NULLs, ties, duplicates, empty groups, boundary values); `query` problems need one
  statement whose result equals the reference's on all of them (order only when asked; column names checked after
  values, with specific messages), `state` problems compare the end state of named tables (rows, types, nullability,
  constraints, defaults, indexes) and probes, `explain` problems check the plan's nodes, `psql` problems check
  settings, files written (\copy) and SQLite files. A hard-coded result fails on the hidden databases.
* **10.15** adds the command line: `psql -c / -f / -l`, `createdb`/`dropdb`/`pg_isready`, a `sqlite3` shell
  (bashlite/utils/sqlitecli.py: modes, .import, .schema ...) and Python's sqlite3 on the simulated filesystem (a
  connection's database is loaded from and written back to the machine's file); pandas `read_sql`/`to_sql` in a venv.
* **182 templates** in `generator/sections/s10_sql/` (every module ≥ 2 per tier), mostly catalogs of drills; the
  Instructions come from `postgreslite/glossary.py` (every entry practised). If random data leaves a task without an
  answer on the shown database, the builder retries with a derived seed.
* **Checks:** `tests/test_sql_section.py` (every item of every template: the reference passes, an idle session fails;
  glossary coverage; the PostgreSQL corpus; psql's errors; grading messages; hidden databases; the command line).

### Session 25 (2026-11-01 → 11-08): Git & GitHub (8.01–8.14) on bash-lite, with GitHub-lite
Request: the git section on bash-lite, with a git that can clone, fetch, pull and push against a simulated GitHub
(never the real one), SSH preferred over HTTPS; 30 small (1–5 MB) and 5 medium (~100 MB) practice repositories,
loaded only when a problem uses one; cloning real public repositories read-only under a size cap; and a visualizer
for repositories and GitHub (branches, merges, pull requests) like the Files panel.

* **git in Python** (`generator/bashlite/git/`): real SHA-1 objects in a world-wide object table, the real binary
  index, refs/packed-refs/reflogs, config files, .gitignore, a port of xdiff (Myers with git's heuristics), 3-way
  merges (merge/diff3/zdiff3, `-X`), and the porcelain: status/add/commit/restore/reset/rm/mv/clean, log (graph,
  -S/-G/-L, --follow, formats)/show/diff/blame/shortlog/grep/bisect, branch/switch/checkout/merge/rebase (-i with
  GIT_SEQUENCE_EDITOR or the editor, --onto, --autosquash)/cherry-pick/revert/stash/tag/describe, remotes, clone
  (--depth, -b, --bare, --recurse-submodules), fetch/pull/push (leases, rejections, hints), worktree, submodule,
  sparse-checkout, plumbing (rev-parse, cat-file, ls-tree, for-each-ref, archive, fsck...). Help texts are our own.
* **Checked against real git 2.39:** `scripts/diff_git.py` runs `tests/git_corpus/*.sh` in both (fixed dates and
  identity), command by command; 9 cases, 0 differences apart from lines marked `#~` (version wording).
* **GitHub-lite** (`git/github.py`): accounts with SSH keys and tokens, the SSH greeting and host-key prompt, HTTPS
  token auth (password auth refused as on github.com), repositories with protection rules, pull requests and issues
  sharing numbers, reviews, merges by merge/squash/rebase (committed by the GitHub bot), releases, Actions runs from
  `.github/workflows` (pass/fail from the tests), Pages served to curl at `*.github.io`, raw.githubusercontent.com,
  the 100 MB file limit (GH001). The `gh` CLI covers auth, repo (create --source --push, clone, fork, view),
  pr (create, list, view, checkout, diff, review, merge, status, checks), issue, run, workflow, release, api, ssh-key.
* **Tools:** `git-lfs` (apt; track/install/pull/push/ls-files/migrate, LFS uploads on push), and pip packages
  `git-filter-repo`, `pre-commit` (its common hooks, ruff and ruff-format), `nbstripout` (as a git filter too),
  `dvc` (init/add/remote/push/pull/status) and `ruff`.
* **The practice repositories** (`git/catalog*.py`): 35 recipes (30 light, 5 medium) modelled on familiar projects
  (an HTTP library, a microframework, leftpad, a GPT training repo, dotfiles, a Pages site, a thesis, a kernel with
  ~500 commits and maintainer merges, a monorepo, a game with LFS art...), with fictional contributors, tags,
  releases, merged/open PRs and issues, and git-gym repositories built around one skill (a regression to bisect,
  conflicting branches, secrets in history, WIP commits, submodules, LFS, hooks). Built on first use only and
  deterministically (same commit ids every session); big files are LazyData (size + seed), so a 100 MB repository
  costs a few MB until its files are read.
* **Real repositories** (`git/realclone.py`): a clone of an unknown name asks api.github.com and fetches the files
  from raw.githubusercontent.com (both allow cross-origin requests; codeload tarballs don't). Read-only, one snapshot
  commit, at most 20 MB / 600 files: over the cap the clone stops, nothing is kept, and the user is told why. Live
  sessions fetch asynchronously (the command sleeps while the browser fetches); replays fetch synchronously, from a
  cache first.
* **Panels:** Files | Git | GitHub tabs beside the terminal (`docs/terminal.js`). Git: the three areas (working
  tree, staging area, repository), an SVG commit graph with branch/remote/tag labels, branches with ahead/behind,
  stash, remotes, tags, an in-progress banner. GitHub: account, repository, branches (default, protected), PRs with
  state, checks and reviews, issues, Actions runs, releases, Pages. The snapshot (`git/viz.py`) is recomputed after
  each command.
* **Grading:** most problems compare the state `/usr/lib/bashlite/gitstate` prints (`git/gitstate.py`): branches,
  history, index, working tree, stash, remotes, the GitHub-lite side, with commit ids made from content (message,
  author, tree, parents) and never dates, so commits typed slower than the reference still match. History-reading
  problems compare output on commits the setup made (fixed ids).
* **140 templates** in `generator/sections/s08_git/` (10 per module, 2 per tier), mostly catalogs; the Instructions
  come from `scripts/instruction_notes.py` (GIT). `scripts/git_templates.py` / `tests/test_git_section.py`: every item
  of every template passes with its reference session, prints no fatal/error, has a clean setup, and fails idle.

### Session 26 (2026-11-08 → 11-09): Docker (12.01–12.14) on bash-lite, with docker-lite
Request: the Docker module, after reading COMP_GUIDE §12 and planning interesting, extensive problems across the whole
curriculum.

* **docker-lite** (`generator/bashlite/docker/`, ~7.5k lines): the daemon and CLI in Python on bash-lite's kernel.
  A container is a simulated host whose filesystem is a copy-on-write clone of its image's; PID 1 runs as a normal
  bash-lite process (signals, exit codes, `--init`), its output goes to a log; published ports are host services
  proxied to the container. run/create/start/stop/kill/rm/ps/logs/exec/inspect (Go templates in `gotmpl.py`)/top/
  stats/cp/port/events, the real error texts and exit codes (125, 126, 127, 137 with OOM, 139, 143), graceful stop
  (exec form vs. `sh -c` PID 1, the 10 s SIGKILL). Help texts are our own.
* **Images** (`images.py`): 22 catalog images built from the distributions' program sets (hello-world, busybox,
  alpine, ubuntu, debian, python 3.11/3.12 (slim, alpine), nginx, postgres:16, curl, nvidia/cuda, pytorch, Jupyter's
  scipy-notebook, and teaching images `mlgym/trainer|memhog|crasher`). **Builds** (`build.py`): the Dockerfile
  instructions incl. multi-stage, ARG/--build-arg, `--target`, heredocs, `.dockerignore`, COPY --chown/--chmod/
  --from, `RUN --mount=type=cache|secret` (secrets never in a layer or history), BuildKit-style output, a layer cache
  keyed by instruction, file contents and args (so a code change keeps the dependency layers CACHED), virtual sizes
  for pip/apt/caches, history, buildx multi-platform builds, `docker scout cves`.
* **Volumes, mounts, networks:** named volumes, bind mounts grafted into the container's filesystem (`:ro` gives
  EROFS, `..` works), tmpfs; the default bridge (no names) vs. user networks (DNS by name, aliases, `--internal`),
  `host.docker.internal:host-gateway`, localhost-only publishing. Postgres and psql run across containers.
* **Compose** (`compose.py`): a YAML subset; up/down (-v)/ps/logs/exec/run/restart/config/build, project naming
  (`<proj>-<svc>-1`, `<proj>_default`), env_file, `depends_on: condition: service_healthy` with health checks.
* **GPUs:** `--gpus all|N|"device=..."` against the machine's GPUs (`#: gpus N`), the CUDA driver version vs. the
  image's requirement, a missing NVIDIA Container Toolkit (install, `nvidia-ctk runtime configure`, restart docker),
  the DataLoader bus error from a 64 MB `/dev/shm` (`--shm-size`, `--ipc=host`).
* **Registries:** Docker Hub, ghcr.io (logins with GitHub-lite tokens from `gh auth token`) and quay.io, world-wide
  so another host (`#: host NAME key docker`, over ssh) can pull what you pushed; denied/unauthorized pushes, digests,
  `--all-tags`; save/load. Model-serving recipes run a FastAPI/uvicorn sentiment API (/health, /predict, routes added
  by the user), Jupyter prints its token URL, `huggingface-cli download` bakes weights for `--network none`.
* **Panel:** a Docker tab beside Files (`docs/terminal.js` DockerPanel; snapshot `docker/viz.py`): images,
  containers with state, ports, health, mounts and GPUs, volumes, networks, compose projects, the last build's steps.
* **Grading:** most problems compare `/usr/lib/bashlite/dockerstate` sections (`docker/state.py`: containers by name,
  images by tag, volumes, networks, registry pushes, logins, compose projects; no ids or times) plus behaviour after
  the session (`post`): what the built image prints, which layers stay CACHED after a code change, the image size, who
  a container runs as, what curl gets from a published port. Reading problems compare `--format`/inspect output.
* **140 templates** in `generator/sections/s12_docker/` (10 per module, 2 per tier); the Instructions come from
  `scripts/instruction_notes.py` (DOCKER, plus the bash notes for the shell around it). Docker's feature names stop
  at the image (`docker run IMG cmd -x` is not `docker run -x`). `scripts/docker_templates.py` /
  `tests/test_docker_section.py`: every item of every template passes with its reference session, prints no error,
  has a clean setup, and fails idle.

### Session 27 (2026-11-09 → 11-10): scikit-learn (4.01–4.20)
Request: the scikit-learn section, with diverse and interesting problems and far more distinct problems than the 100
questions of a module (lite 10, easy 25, medium 35, hard 25, challenge 5), so learners who go past 100 (random
tiers) rarely see a repeat.

* **Runtime:** Pyodide's scikit-learn 1.8.0 (the version the guide is written against; the same version is in the
  local venv for the tests). The sklearn page loads numpy, pandas and scikit-learn (matplotlib only for 4.17). The
  setup imports NumPy/pandas and builds the data; learners write their own sklearn imports, and the Instructions show
  each name with its full import path (`sklearn.preprocessing.StandardScaler(...)`). The grader now also restores
  scikit-learn's global config after every run (a `set_config(transform_output='pandas')` can't leak).
* **Problems** (`generator/sections/s04_sklearn/`, one file per module): 691 templates, most of them catalogs, giving
  3,184 distinct problem kinds (distinct solutions; every catalog item and task variant counts) — at least
  15 / 30 / 40 / 30 / 6 per module for lite / easy / medium / hard / challenge. Data: the bundled datasets (shuffled
  where rows are sorted by class), seeded `make_*` generators, and `table()`: ten themed DataFrame scenarios (churn,
  houses, loans, students, training runs, servers, patients, flights, bike rentals, phones) with numeric and
  categorical columns, optional NaNs, generated from a seed. Problems range from one API at a time to leakage fixes,
  imbalanced data, grouped/temporal splits, by-hand reimplementations checked against sklearn, interpretation and
  debugging; every reference is deterministic (`random_state` given in the prompt) and fast enough for Pyodide.
* **Challenges** (92 templates, 169 kinds, ≥ 6 per module): algorithmic tasks about how methods work (prefix-sum CV and
  standardization queries, merging scaler shards, the best stump split, k-means++ seeding, PAVA, ROC AUC with ties,
  ridge leave-one-out shortcuts, TF-IDF from scratch, Fenwick-tree ECDFs ...), with 2–3 variants each stored in the
  setup; large hidden tests; brute forces in `tests/sklearn_brutes/` (one file per module, often scikit-learn itself).
* **Instructions:** `generator/instructions.py` tracks names imported from sklearn and used bare;
  `scripts/_docstrings.py` describes them from scikit-learn's docstrings (the public API index, fitted attributes and
  result fields from the numpydoc sections, the tree structure), plus hand-written notes for dataset Bunch fields.
* **Coverage:** `scripts/guide_coverage.py` handles bare sklearn names (the guide's §4 imports are the vocabulary);
  every guide API of every module is practised (`tests/test_coverage.py` covers sklearn). Excluded: the `fetch_*`
  downloaders (no network) and UMAP (not in Pyodide).
* **Checks:** `scripts/sklearn_templates.py [modules]` grades every catalog item's reference (twice, which catches
  nondeterminism), checks that wrong answers fail, variety over seeds, the Instructions, Pyodide hazards (`n_jobs`,
  `fetch_*`), reference speed, and prints distinct kinds per tier next to the questions per tier.
  `scripts/sklearn_e2e.py` checks the pages in a browser and grades every template (every catalog item) inside
  Pyodide: 2,238 grades, 0 failures, the slowest 1.6 s against a 15–20 s limit.
* **Pyodide is 32-bit:** `np.intp` is int32 there, so `np.repeat(a, rng.integers(...))` and `np.bincount` on int64
  arrays raise in the browser but work on 64-bit machines. `grader.numpy_32bit_compat()` makes both accept int64
  input whose values fit (a no-op on 64-bit; applied on every run, since NumPy may load after the grader). Also
  seed `liblinear`/`saga` models: without `random_state` they draw from NumPy's global RNG.

### Open problem: Hugging Face in the browser
Pyodide has no `transformers`. Options: extend the shim idea (a tiny tokenizer/pipeline layer over torch-lite),
AST and pattern checks, or an optional "run locally" mode. **Not decided yet.**

---

## 3. Repo layout

```
generator/            Python package: question generation + grading (runs locally & in Pyodide)
  core.py             Problem dataclass, tier schedule, template registry, generate()
  grader.py           run / grade user code, value comparison, AST checks
  themes.py           themed datasets (AI labs, pets, appliances, speech, vision, ...)
  sections/           one package per COMP_GUIDE section
    s01_numpy/        _common.py (helpers, challenge()) + one file per subsection: m01_printing.py … m18_ml_patterns.py
    s02_pandas/       _common.py (themed DataFrame builder `frame()`, challenge()) + m01_core.py … m19_performance.py
    s03_pytorch/      _common.py (HEADER imports torch/nn/F, tensor literals, challenge()) + m01_tensors.py … m19_dpo.py
    s04_sklearn/      _common.py (P/catalog/challenge, datasets: builtin/synth_clf/synth_reg, table() = themed
                      DataFrames) + m01_estimators.py … m20_custom.py (auto-imported in order)
    s07_bash/         _common.py (P/K problem helpers, files() heredocs, catalog(), challenge()) + m01_basics.py … m17_keys_history.py
    s08_git/          _common.py (P() problem helper with check=(gitstate sections, dir), Repo() setup builder, catalog())
                      + m01_setup.py … m14_ml.py
    s09_vim/          _common.py (V() problem helper: runs the reference once for the target text, cursor target and
                      Instructions; drill(); ML-themed texts) + m01_modes.py … m12_vimrc.py
    s10_sql/          _common.py (Q() problem helper: runs the reference for the target, drill(), retries), _data.py (the
                      shop, ML-runs, events ... databases with hidden variants) + m01_select.py … m16_analytics.py
    s11_latex/        _common.py (L() problem helper: compiles the reference once for the target preview and the
                      Instructions; drill(); @fixed) + m01_skeleton.py … m18_troubleshooting.py
    s12_docker/       _common.py (P() problem helper with check=(dockerstate sections), catalog(), files(), the model API
                      app) + m01_setup.py … m14_debug.py
  bashlite/           bash-lite: a simulated Ubuntu machine in pure Python (vfs, parser, kernel, shell, builtins, utils/ commands,
                      world/basefs, directives, check = the bash grader, keys = readline/tmux simulator, features,
                      live = the live terminal: key-by-key sessions on the real clock, logs, replays;
                      vim/ = the Vim engine, utils/vim.py = the `vim` command, vimcheck = the Vim grader;
                      git/ = git + GitHub-lite + gh + git-lfs/filter-repo/pre-commit/nbstripout/dvc, the practice
                      repositories (catalog*.py), realclone, viz = the panels' data, gitstate = the git grader;
                      docker/ = docker-lite: engine, images (catalog), build, compose, cli + cmd_*, programs inside
                      images, gotmpl, viz = the Docker panel, state = the docker grader)
  postgreslite/       postgres-lite: PostgreSQL 16 + psql 16 in Python (lexer, parser, types + pg_catalog.txt, coerce,
                      analyze, executor, funcs, aggs, dml, ddl, server = cluster/sessions/transactions, psql, describe,
                      explain = the planner, glossary = the Instructions); bashlite/utils/psqlprog.py runs it in the
                      terminal, bashlite/sqlcheck.py grades, bashlite/sqlsetup.py builds a problem's databases
  latexlite/          latex-lite: LaTeX in Python (tokens, engine + commands/features/floats/tables/bib/algo, mathmode,
                      tikz, pgfplots, beamer, packages = what each package provides, render + render_table = HTML,
                      compile = compile_doc(), check = the LaTeX grader, catalog = the Instructions' descriptions)
  shims/torch/        torch-lite: NumPy-backed torch (core in __init__.py, more ops in _extra.py; nn, optim, utils, func,
                      autograd, amp, profiler, distributed (+ fsdp, checkpoint), library, _compile, ...)
  shims/triton/       triton-lite: @triton.jit kernels interpreted on the CPU
  api.py              JSON entry points called by docs/worker.js (generate, run, grade, info, self_test)
scripts/
  build.py            zip generator/ into docs/generator.zip + write docs/build.json catalog (rerun after every change)
  preview.py          print sample questions from the CLI
  e2e_check.py        Playwright click-through of the site + Pyodide self-test of every template
  latex_templates.py  check every LaTeX template item (+ --katex: typeset every target formula); latex_e2e.py: the pages
ci/pages.yml          optional GitHub Actions deploy (inactive, see its header)
tests/                pytest: every template passes its own grader; every challenge matches a brute force
docs/                 the static website (GitHub Pages root): app.js (pages), terminal.js (the Bash terminal, nano,
                      Files panel), worker.js (Pyodide), progress.js (Download / Upload of a .tlsave progress file),
                      style.css, latex.css (latex-lite's page: the article class, Beamer slides; KaTeX from cdnjs)
  dsa/                Part III tutorials: lang.js (the statement language -> Python/C++/Java/Rust), common.js (what
                      every tutorial shares: rendering, reference programs, tests), run.js (Compiler Explorer + Pyodide
                      runner, the tests around the user's file, output diff), page.js (setup, steps with the typing
                      coach, final, status, reset); one file per tutorial: binary-search.js, mergesort.js, quicksort.js, linked-list.js,
                      stack.js, queue.js, unbounded-array.js, hash-table.js, bst.js,
                      dict-set.js, heap.js, priority-queue.js
scripts/dsa_selftest.mjs  every tutorial reference x option x language, step by step (+ --mutants)
scripts/dsa_e2e.py        browser click-through of Part III (Compiler Explorer answered by local compilers)
scripts/progress_e2e.py   browser check of the progress file: Download, Upload, Merge / Replace, rejected files
scripts/diff_vim.py       bash-lite's Vim vs real Vim 9.1 on tests/vim_corpus (--record saves real Vim's states)
scripts/diff_vim_templates.py  every Vim exercise's reference keys in real Vim vs bash-lite's Vim
scripts/vim_templates.py  every Vim template item: the reference passes, an idle session fails (-v prints prompts)
scripts/vim_e2e.py        browser check of the Vim pages
scripts/sql_templates.py  every SQL template item: the reference passes, an idle session fails (--oracle: real PostgreSQL)
scripts/sql_corpus.py     postgres-lite vs PostgreSQL 16 on tests/sql_corpus (--record saves a real server's output)
scripts/git_templates.py  every Git template item: the reference passes, no fatal/error, an idle session fails
scripts/diff_git.py       bash-lite's git vs real git on tests/git_corpus (command by command)
```

Local dev:
```
uv venv --python 3.14 .venv && uv pip install --python .venv/bin/python numpy pandas pytest
.venv/bin/python -m pytest -q
.venv/bin/python scripts/preview.py numpy --module 1.05 --tier hard -n 3 --solution
.venv/bin/python scripts/build.py && python3 -m http.server -d docs 8000   # open http://localhost:8000
.venv/bin/python scripts/e2e_check.py /tmp/shots [https://dxu-18.github.io/training-loop]  # browser click-through
```

---

## 4. Status & roadmap

Legend: ✅ done · 🟡 partial · ⬜ not started

| # | Section | Templates | Checker | Notes |
|---|---|---|---|---|
| 1 | NumPy | ✅ **all 18 modules live**: 277 templates in `generator/sections/s01_numpy/` (every module ≥ 2 templates per tier, 2 challenges each) | exec (Pyodide) | session 3 |
| 2 | pandas | ✅ **all 19 modules live**: 261 templates in `generator/sections/s02_pandas/` (every module ≥ 2 templates per tier, 2 challenges each) | exec (Pyodide) + figure/Styler inspection for 2.18 | session 4 |
| 3 | PyTorch | ✅ **all 19 modules live**: 240 templates in `generator/sections/s03_pytorch/` (every module ≥ 2 templates per tier, 2 challenges each) | exec (Pyodide) on torch-lite + triton-lite | session 5 |
| 4 | scikit-learn | ✅ **all 20 modules live**: 691 templates in `generator/sections/s04_sklearn/` (3,184 distinct problem kinds; every tier of every module has more kinds than questions) | exec in Pyodide's scikit-learn 1.8: the reference reruns on the same seeded data and values are compared; challenges have hidden tests and brute forces | session 27 |
| 5 | Matplotlib | ⬜ | exec + figure inspection | |
| 6 | Hugging Face | ⬜ | **open problem** | |
| 7 | Bash | ✅ **all 17 modules live**: 176 templates in `generator/sections/s07_bash/` (every module ≥ 2 templates per tier, 2 challenges each), answered in a live terminal | bash-lite: simulated Ubuntu machine (`generator/bashlite/`); the terminal session is replayed and its output / files compared; keystrokes for 7.17 | sessions 7, 8 |
| 8 | Git & GitHub | ✅ **all 14 modules live**: 140 templates in `generator/sections/s08_git/` (every module ≥ 2 templates per tier), answered in a live terminal beside Files, Git and GitHub panels | bash-lite's git and GitHub-lite (`generator/bashlite/git/`, checked against real git 2.39): repository and GitHub state compared with date-independent commit ids, or the output of history commands | session 25 |
| 9 | Vim | ✅ **all 12 modules live**: 121 templates in `generator/sections/s09_vim/` (every module ≥ 2 templates per tier), answered in Vim in the live terminal | bash-lite's Vim (`generator/bashlite/vim/`, checked against real Vim 9.1); Vim's end state compared with the reference keys', plus required / forbidden commands and a keystroke par | session 22 |
| 10 | SQL | ✅ **all 16 modules live**: 182 templates in `generator/sections/s10_sql/` (every module ≥ 2 templates per tier), answered in a live psql beside a Database panel | postgres-lite (`generator/postgreslite/`, checked against a real PostgreSQL 16.2): results / end states compared on the shown and hidden databases, plans, psql settings, files | session 24 |
| 11 | LaTeX | ✅ **all 18 modules live**: 180 templates in `generator/sections/s11_latex/` (10 per module), answered in an editor with Compile (the PDF beside it) and Submit | latex-lite (`generator/latexlite/`): LaTeX interpreted in Python, math by KaTeX, TikZ/pgfplots as SVG, Beamer slides; the compiled documents compared, plus required / forbidden commands | session 23 |
| 12 | Docker | ✅ **all 14 modules live**: 140 templates in `generator/sections/s12_docker/` (every module ≥ 2 templates per tier), answered in a live terminal beside Files and Docker panels | docker-lite (`generator/bashlite/docker/`): engine state (containers, images, volumes, networks, registries, compose) compared without ids or times, plus behaviour: the built image's output, cached layers, sizes, users, HTTP responses | session 26 |
| 13 | Data Structures & Algorithms (Part III) | ✅ **all 12 live** (binary search, mergesort, quicksort, linked lists, stacks, queues, unbounded arrays, hash tables, BSTs, dictionaries & sets, heaps, priority queues; all options × element types × Python/C++/Java/Rust, contracts optional) | Pyodide (Python), Compiler Explorer (C++/Java/Rust); line-by-line output diff | session 11 |

### What exists (as of session 5; the Bash topic and its terminal are described in §2, sessions 7–8)
- `generator/`: core (tiers, registry, Instructions), grader (answer comparison, AST checks,
  hidden tests, clean tracebacks), themes (20 themes, 10 axes, 10 label sets), and the NumPy section.
- `tests/`: 2,463 tests. Every template × 25 seeds (6 for challenges) must pass its own grader,
  obviously wrong answers must fail, and variety is checked. Every challenge (112: 36 NumPy, 38 pandas,
  38 PyTorch) is cross-checked against a brute force on 60 small inputs. `test_*_modules_complete` enforces
  ≥ 2 templates per tier for NumPy, pandas and PyTorch.
- `docs/`: the live site, three levels: topic list (`#/`: total solved + medal count per topic,
  "soon" for topics with no live module) → topic page (`#/numpy`: its modules with per-module
  count and medal) → practice (`#/numpy/1.05`: problem, Given, Instructions, editor, Run/Submit,
  console, verdict). Pyodide worker with a time limit, per-module localStorage state, a
  placeholder medal per module and the surprise modal at 250.
- **Verified in a real browser (session 2)** with `scripts/e2e_check.py` (Playwright + bundled
  Chromium). In the Claude Code sandbox, run it with the sandbox disabled: the system Chrome
  has no network there, but Playwright's bundled Chromium does.

### UX decisions made in session 1 (change if you disagree)
- Submitting code that **doesn't parse** (SyntaxError) does not use up the problem. The user
  fixes it and resubmits. Any other wrong submission shows ✗, the reference solution under **"Solution:"**
  (session 9), and a **"New problem →"** button, which rerolls at the same number and tier.
- **Run** (⌘/Ctrl+Enter) executes without grading. **Submit** is Shift+Enter. On the Bash pages there is no Run (the
  terminal runs everything) and no editor: Submit (or ⌘Enter) grades the session, Reset starts the problem's
  machine over, and "nothing to check yet" (no commands, no script file) keeps the problem like a syntax error.
- The number shown on the practice page is the current question number of that module
  (solved + 1). The topic page shows each module's solved count and medal; the home page shows
  the topic's total solved and how many module medals it holds (`×N`).
- Refreshing does not reroll: the current seed and tier are saved, and so is a draft of the code.
- Challenge prompts show the sample answer for the visible input, like Codeforces. Hidden tests
  are large (n = 2·10⁴), so quadratic solutions time out.

### Next steps (suggested order)
0. Part III, one tutorial at a time, in COMP_GUIDE §13's order (stacks and queues reuse the linked-list node), each as
   a `docs/dsa/<name>.js` like `linked-list.js`, plus its reference/mutant self-test.
1. Matplotlib (reuse the 2.18 figure-inspection `post` pattern, and 4.17's Display/Axes value checks).
2. SQL section (sqlite3 checker); Git and Docker on top of bash-lite (a simulated `git` and `docker`).
3. Hugging Face (§2 open problem; torch-lite is a starting point).
4. Medal art, then the style pass.

**Adding a section:** copy the `s01_numpy/` layout (`_common.py` with a `template(module, tier)` wrapper that
fixes the section, one file per subsection), import the package in `generator/sections/__init__.py`, and give every
subsection ≥ 1 template per tier (it goes live module by module). For each challenge add a brute
force to `tests/test_challenges.py`. Then `scripts/build.py`, `pytest`, `scripts/e2e_check.py`, push.

Other TODO:
- ⬜ Medal art: a unique, cute medal image for each **subsection** (195 across the 12 topics;
  deferred by request). The placeholder varies the colour per topic and the ribbon + number per module.
- ⬜ Final visual style pass on the frontend.
- ✅ GitHub Pages deploy (session 2): https://dxu-18.github.io/training-loop/
- ✅ Per-module pages, counters and medals (session 3). 🟡 Content: NumPy 18/18, pandas 19/19 and PyTorch 19/19
  modules; the other 9 topics (139 modules) are listed as "soon".
