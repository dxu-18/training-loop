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
  - 1–50: **lite**
  - 51–120: **easy**
  - 121–200: **medium**
  - 201–245: **hard**
  - 246–250: **challenge**
  - 251 onward: random, using the same distribution, so a challenge still turns up
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
| State | `localStorage` key `training-loop:v2`: per module (`"numpy/1.05"`) solved count, current seed and tier, medal, code draft | A static site has no accounts. The session-1 key `training-loop:v1` (one counter per topic) is left untouched but no longer read. |
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
- **SQL**: run for real with `sqlite3`, which is part of Pyodide's stdlib. Compare
  result sets.
- **Bash / Git / Docker CLI**: a `command` checker that tokenizes with `shlex` and
  compares program, subcommand, flags (aliases, `-la` = `-l -a`, order-insensitive)
  and positional args. A simulated shell may come later.
- **Vim**: compare keystroke strings against a list of accepted alternatives.
- **LaTeX**: normalized token comparison with accepted alternatives. KaTeX could
  render the user's math later.
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
    s07_bash/         _common.py (P/K problem helpers, files() heredocs, catalog(), challenge()) + m01_basics.py … m17_keys_history.py
  bashlite/           bash-lite: a simulated Ubuntu machine in pure Python (vfs, parser, kernel, shell, builtins, utils/ commands,
                      world/basefs, directives, check = the bash grader, keys = readline/tmux simulator, features,
                      live = the live terminal: key-by-key sessions on the real clock, logs, replays)
  shims/torch/        torch-lite: NumPy-backed torch (core in __init__.py, more ops in _extra.py; nn, optim, utils, func,
                      autograd, amp, profiler, distributed (+ fsdp, checkpoint), library, _compile, ...)
  shims/triton/       triton-lite: @triton.jit kernels interpreted on the CPU
  api.py              JSON entry points called by docs/worker.js (generate, run, grade, info, self_test)
scripts/
  build.py            zip generator/ into docs/generator.zip + write docs/build.json catalog (rerun after every change)
  preview.py          print sample questions from the CLI
  e2e_check.py        Playwright click-through of the site + Pyodide self-test of every template
ci/pages.yml          optional GitHub Actions deploy (inactive, see its header)
tests/                pytest: every template passes its own grader; every challenge matches a brute force
docs/                 the static website (GitHub Pages root): app.js (pages), terminal.js (the Bash terminal, nano,
                      Files panel), worker.js (Pyodide), style.css
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
| 4 | scikit-learn | ⬜ | exec | |
| 5 | Matplotlib | ⬜ | exec + figure inspection | |
| 6 | Hugging Face | ⬜ | **open problem** | |
| 7 | Bash | ✅ **all 17 modules live**: 176 templates in `generator/sections/s07_bash/` (every module ≥ 2 templates per tier, 2 challenges each), answered in a live terminal | bash-lite: simulated Ubuntu machine (`generator/bashlite/`); the terminal session is replayed and its output / files compared; keystrokes for 7.17 | sessions 7, 8 |
| 8 | Git & GitHub | ⬜ | command | |
| 9 | Vim | ⬜ | keystrokes | |
| 10 | SQL | ⬜ | sqlite3 exec | |
| 11 | LaTeX | ⬜ | normalized / KaTeX | |
| 12 | Docker | ⬜ | command / Dockerfile | |

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
1. scikit-learn (exec checker; add `"scikit-learn"` packages, already listed in `docs/app.js`) and
   Matplotlib (reuse the 2.18 figure-inspection `post` pattern).
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
