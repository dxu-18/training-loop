# training-loop

One problem at a time: muscle memory for the ML stack (NumPy, pandas, PyTorch, scikit-learn,
Matplotlib, Hugging Face) and developer tools (Bash, Git, Vim, SQL, LaTeX, Docker).

**Practice:** https://dxu-18.github.io/training-loop/

Every topic is split into its COMP_GUIDE subsections (NumPy has 18), and each subsection is its own
page with its own counter, a medal at 100, and endless randomly generated questions that get harder
as you go. Your code runs in the browser (Pyodide), so there is nothing to install. Live now: NumPy,
pandas, PyTorch (on torch-lite, a NumPy-backed `torch` that runs in the browser) and Bash (on bash-lite,
a simulated Ubuntu machine with its own shell, files, processes, GPUs, ssh hosts and Python environments, used
through a Terminal.app-style terminal with nano and a live view of your files);
the other topics are listed as "soon". Topics come from [COMP_GUIDE.md](COMP_GUIDE.md); the design and roadmap are in
[PROJECT_SPEC.md](PROJECT_SPEC.md).

**Part III, Data Structures & Algorithms**, is different: step-by-step tutorials that walk you through building each
structure from a blank file, in Python, C++, Java or Rust. Choose the options (say, a circular doubly linked list with a
dummy node) and the complexity boxes update. Then follow concrete steps, each checked before the next opens, and
finally build it again with no hints. Binary search, mergesort, quicksort, linked lists, stacks, queues, unbounded arrays and hash tables are live; Python runs in the browser, while C++, Java and Rust
are compiled and run on [Compiler Explorer](https://godbolt.org).

## Develop

```
uv venv --python 3.14 .venv && uv pip install --python .venv/bin/python numpy pandas pytest
.venv/bin/python -m pytest -q                          # every template must pass its own grader
.venv/bin/python scripts/preview.py numpy --module 1.05 --tier hard -n 3 --solution
python3 scripts/build.py && python3 -m http.server -d docs 8000   # http://localhost:8000
node scripts/dsa_selftest.mjs                          # Part III: every reference x option x language (+ --mutants)
```

GitHub Pages serves `docs/` from `main`. Run `python3 scripts/build.py` before committing
generator changes (a test fails if the committed bundle is stale).
