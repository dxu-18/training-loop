# COMP_GUIDE — Muscle-memory reference for the ML stack & developer tools

A syntax-first practice guide covering 12 tools. **Part I** covers NumPy, pandas, PyTorch, scikit-learn, Matplotlib and Hugging Face. **Part II** covers Bash, Git, Vim, SQL, LaTeX and Docker. **Part III** is different: step-by-step tutorials that build the classic data structures and algorithms from scratch, in Python, C++, Java or Rust. Within each section, numbered modules go from fundamentals to advanced usage; the PyTorch modules follow the `01_tensors` → `19_dpo_loss` curriculum. Each module has a dense, commented code block, notes on common mistakes where needed, and short **Drills**. Each `###` module is designed to become one practice page or exercise set on the companion website.

**How to practice.** Type every line yourself instead of copy-pasting, and predict each output before you run it. Trailing comments show expected results and shapes, such as `(B, T, D)` = batch, sequence, features. Do the drills from memory and check against the block only afterwards. Revisit a section a few days later and redo its drills faster.

**Versions.** Python 3.12 · NumPy 2.4 · pandas 3.0 · scikit-learn 1.8 · Matplotlib 3.10 · PyTorch 2.x (2.6+ for FSDP2) · Transformers 5.x · TRL 1.x · PostgreSQL 15+ · TeX Live · Docker Engine with Compose v2. The NumPy, pandas, scikit-learn and Matplotlib code was executed against these versions. The PyTorch and Hugging Face code was syntax-checked against them, and every LaTeX block was compiled with pdflatex.

## Contents

**[Part I — The Python ML stack](#part-i--the-python-ml-stack)**

- **[1. NumPy](#1-numpy)** — [1.01 Setup, printing & help](#101-setup-printing--help) · [1.02 Creating arrays](#102-creating-arrays) · [1.03 Random numbers](#103-random-numbers) · [1.04 Attributes, dtypes & casting](#104-attributes-dtypes--casting) · [1.05 Indexing & slicing](#105-indexing--slicing) · [1.06 Views, copies & memory layout](#106-views-copies--memory-layout) · [1.07 Reshaping & moving axes](#107-reshaping--moving-axes) · [1.08 Joining, splitting, repeating & padding](#108-joining-splitting-repeating--padding) · [1.09 Broadcasting](#109-broadcasting) · [1.10 Elementwise math & ufuncs](#110-elementwise-math--ufuncs) · [1.11 Reductions & statistics](#111-reductions--statistics) · [1.12 Comparisons, logic & missing values](#112-comparisons-logic--missing-values) · [1.13 Sorting, searching, sets & counting](#113-sorting-searching-sets--counting) · [1.14 Linear algebra & einsum](#114-linear-algebra--einsum) · [1.15 FFT, polynomials & signal helpers](#115-fft-polynomials--signal-helpers) · [1.16 Strings, dates, structured arrays & bits](#116-strings-dates-structured-arrays--bits) · [1.17 File I/O](#117-file-io) · [1.18 Vectorized ML patterns & testing](#118-vectorized-ml-patterns--testing)
- **[2. pandas](#2-pandas)** — [2.01 Setup, options & core objects](#201-setup-options--core-objects) · [2.02 Creating DataFrames & dtypes](#202-creating-dataframes--dtypes) · [2.03 Reading & writing data](#203-reading--writing-data) · [2.04 Inspecting data](#204-inspecting-data) · [2.05 Selecting & filtering](#205-selecting--filtering) · [2.06 Adding, modifying & removing columns](#206-adding-modifying--removing-columns) · [2.07 Sorting, ranking & duplicates](#207-sorting-ranking--duplicates) · [2.08 Missing data](#208-missing-data) · [2.09 Applying functions](#209-applying-functions) · [2.10 String methods (.str)](#210-string-methods-str) · [2.11 Dates, times & time series](#211-dates-times--time-series) · [2.12 Categoricals & binning](#212-categoricals--binning) · [2.13 GroupBy (split-apply-combine)](#213-groupby-split-apply-combine) · [2.14 Reshaping](#214-reshaping) · [2.15 Combining frames](#215-combining-frames) · [2.16 Window functions](#216-window-functions) · [2.17 MultiIndex](#217-multiindex) · [2.18 Plotting & styling](#218-plotting--styling) · [2.19 Performance & handing data to models](#219-performance--handing-data-to-models)
- **[3. PyTorch](#3-pytorch)** — [3.01 Tensors](#301-tensors) · [3.02 Broadcasting & shape changes](#302-broadcasting--shape-changes) · [3.03 Autograd, gradient descent & backprop](#303-autograd-gradient-descent--backprop) · [3.04 Dataset & DataLoader](#304-dataset--dataloader) · [3.05 MLP training: modules, losses, optimizers & the loop](#305-mlp-training-modules-losses-optimizers--the-loop) · [3.06 Custom autograd Functions](#306-custom-autograd-functions) · [3.07 Memory, strides & contiguity](#307-memory-strides--contiguity) · [3.08 Attention](#308-attention) · [3.09 Transformer block](#309-transformer-block) · [3.10 Hooks & debugging](#310-hooks--debugging) · [3.11 KV cache](#311-kv-cache) · [3.12 LoRA layer](#312-lora-layer) · [3.13 Automatic mixed precision (AMP)](#313-automatic-mixed-precision-amp) · [3.14 Profiler & benchmarking](#314-profiler--benchmarking) · [3.15 Triton kernels & C++ extensions](#315-triton-kernels--c-extensions) · [3.16 Distributed Data Parallel (DDP)](#316-distributed-data-parallel-ddp) · [3.17 FSDP & checkpointing](#317-fsdp--checkpointing) · [3.18 torch.compile](#318-torchcompile) · [3.19 DPO loss](#319-dpo-loss)
- **[4. scikit-learn](#4-scikit-learn)** — [4.01 The estimator API](#401-the-estimator-api) · [4.02 Datasets](#402-datasets) · [4.03 Splitting & cross-validation](#403-splitting--cross-validation) · [4.04 Scaling & feature transforms](#404-scaling--feature-transforms) · [4.05 Encoding categories & imputing missing values](#405-encoding-categories--imputing-missing-values) · [4.06 Pipelines & ColumnTransformer](#406-pipelines--columntransformer) · [4.07 Feature selection](#407-feature-selection) · [4.08 Linear models](#408-linear-models) · [4.09 SVMs, nearest neighbours & naive Bayes](#409-svms-nearest-neighbours--naive-bayes) · [4.10 Trees & ensembles](#410-trees--ensembles) · [4.11 Clustering & mixture models](#411-clustering--mixture-models) · [4.12 Dimensionality reduction & embeddings](#412-dimensionality-reduction--embeddings) · [4.13 Classification metrics](#413-classification-metrics) · [4.14 Regression, clustering & distance metrics](#414-regression-clustering--distance-metrics) · [4.15 Hyperparameter search](#415-hyperparameter-search) · [4.16 Probability calibration & decision thresholds](#416-probability-calibration--decision-thresholds) · [4.17 Model inspection & built-in plots](#417-model-inspection--built-in-plots) · [4.18 Text features](#418-text-features) · [4.19 Outliers, multi-output, semi-supervised & other models](#419-outliers-multi-output-semi-supervised--other-models) · [4.20 Custom estimators, persistence & configuration](#420-custom-estimators-persistence--configuration)
- **[5. Matplotlib](#5-matplotlib)** — [5.01 Setup, backends & the two interfaces](#501-setup-backends--the-two-interfaces) · [5.02 Figures, subplots & layouts](#502-figures-subplots--layouts) · [5.03 Line & scatter plots](#503-line--scatter-plots) · [5.04 Bars, histograms & distributions](#504-bars-histograms--distributions) · [5.05 Areas, error bars, reference lines & spans](#505-areas-error-bars-reference-lines--spans) · [5.06 Images, heatmaps, contours & vector fields](#506-images-heatmaps-contours--vector-fields) · [5.07 Pie, polar & 3D](#507-pie-polar--3d) · [5.08 Titles, labels, limits, scales & ticks](#508-titles-labels-limits-scales--ticks) · [5.09 Legends, annotations, text & math](#509-legends-annotations-text--math) · [5.10 Colors, colormaps, norms & colorbars](#510-colors-colormaps-norms--colorbars) · [5.11 Styles & rcParams](#511-styles--rcparams) · [5.12 Saving & exporting](#512-saving--exporting) · [5.13 Patches, collections & coordinate systems](#513-patches-collections--coordinate-systems) · [5.14 Animation & live-updating plots](#514-animation--live-updating-plots) · [5.15 ML visualization recipes](#515-ml-visualization-recipes)
- **[6. Hugging Face](#6-hugging-face)** — [6.01 The ecosystem at a glance](#601-the-ecosystem-at-a-glance) · [6.02 The Hub: auth, download, upload & cache](#602-the-hub-auth-download-upload--cache) · [6.03 Pipelines](#603-pipelines) · [6.04 Tokenizers & chat templates](#604-tokenizers--chat-templates) · [6.05 Models & configs](#605-models--configs) · [6.06 Text generation](#606-text-generation) · [6.07 Datasets](#607-datasets) · [6.08 Trainer & TrainingArguments](#608-trainer--trainingarguments) · [6.09 Custom loops with Accelerate](#609-custom-loops-with-accelerate) · [6.10 PEFT: LoRA & QLoRA](#610-peft-lora--qlora) · [6.11 TRL: SFT, DPO & GRPO](#611-trl-sft-dpo--grpo) · [6.12 Evaluate, safetensors, embeddings & diffusion](#612-evaluate-safetensors-embeddings--diffusion) · [6.13 Inspecting & visualizing models](#613-inspecting--visualizing-models)

**[Part II — Developer tools](#part-ii--developer-tools)**

- **[7. Bash & the command line](#7-bash--the-command-line)** — [7.01 Shell basics & getting help](#701-shell-basics--getting-help) · [7.02 Navigating & managing files](#702-navigating--managing-files) · [7.03 Viewing & comparing files](#703-viewing--comparing-files) · [7.04 Permissions, archives & disk usage](#704-permissions-archives--disk-usage) · [7.05 Finding files & searching text](#705-finding-files--searching-text) · [7.06 Text processing: cut, sort, uniq, tr, sed, awk, jq](#706-text-processing-cut-sort-uniq-tr-sed-awk-jq) · [7.07 Pipes & redirection](#707-pipes--redirection) · [7.08 Variables, quoting & expansion](#708-variables-quoting--expansion) · [7.09 Arrays & arithmetic](#709-arrays--arithmetic) · [7.10 Conditionals & loops](#710-conditionals--loops) · [7.11 Functions & scripts](#711-functions--scripts) · [7.12 Processes & jobs](#712-processes--jobs) · [7.13 Environment, PATH & config files](#713-environment-path--config-files) · [7.14 Remote work: ssh, scp, rsync, curl & tmux](#714-remote-work-ssh-scp-rsync-curl--tmux) · [7.15 Python environments](#715-python-environments) · [7.16 ML workflow one-liners](#716-ml-workflow-one-liners) · [7.17 Keyboard shortcuts & history](#717-keyboard-shortcuts--history)
- **[8. Git & GitHub](#8-git--github)** — [8.01 Setup & configuration](#801-setup--configuration) · [8.02 Creating repositories & .gitignore](#802-creating-repositories--gitignore) · [8.03 The daily loop: status, add, commit](#803-the-daily-loop-status-add-commit) · [8.04 History & diffs](#804-history--diffs) · [8.05 Branches](#805-branches) · [8.06 Merging & resolving conflicts](#806-merging--resolving-conflicts) · [8.07 Rebasing & rewriting history](#807-rebasing--rewriting-history) · [8.08 Remotes: fetch, pull, push & forks](#808-remotes-fetch-pull-push--forks) · [8.09 Stash, tags & cherry-pick](#809-stash-tags--cherry-pick) · [8.10 Undoing things](#810-undoing-things) · [8.11 Debugging & search](#811-debugging--search) · [8.12 Advanced: worktrees, submodules, LFS, hooks & history surgery](#812-advanced-worktrees-submodules-lfs-hooks--history-surgery) · [8.13 GitHub: gh CLI, pull requests, Actions & Pages](#813-github-gh-cli-pull-requests-actions--pages) · [8.14 Git for ML projects & commit conventions](#814-git-for-ml-projects--commit-conventions)
- **[9. Vim](#9-vim)** — [9.01 Modes, opening, saving & quitting](#901-modes-opening-saving--quitting) · [9.02 Moving around](#902-moving-around) · [9.03 Inserting & basic editing](#903-inserting--basic-editing) · [9.04 The grammar: operators × motions × text objects](#904-the-grammar-operators--motions--text-objects) · [9.05 Visual mode](#905-visual-mode) · [9.06 Search & replace](#906-search--replace) · [9.07 Registers & the clipboard](#907-registers--the-clipboard) · [9.08 Macros & the dot command](#908-macros--the-dot-command) · [9.09 Buffers, windows & tabs](#909-buffers-windows--tabs) · [9.10 Marks, folds & undo history](#910-marks-folds--undo-history) · [9.11 Command-line power moves](#911-command-line-power-moves) · [9.12 A starter .vimrc & Vim keys everywhere](#912-a-starter-vimrc--vim-keys-everywhere)
- **[10. SQL](#10-sql)** — [10.00 Practice schema](#1000-practice-schema) · [10.01 SELECT: filtering, sorting & limiting](#1001-select-filtering-sorting--limiting) · [10.02 Expressions: CASE, NULL handling & casting](#1002-expressions-case-null-handling--casting) · [10.03 Aggregation: GROUP BY & HAVING](#1003-aggregation-group-by--having) · [10.04 Joins](#1004-joins) · [10.05 Subqueries, CTEs & set operations](#1005-subqueries-ctes--set-operations) · [10.06 Window functions](#1006-window-functions) · [10.07 String functions](#1007-string-functions) · [10.08 Dates & times](#1008-dates--times) · [10.09 Math & numeric functions](#1009-math--numeric-functions) · [10.10 Creating & changing tables (DDL)](#1010-creating--changing-tables-ddl) · [10.11 Inserting, updating, deleting & upserting (DML)](#1011-inserting-updating-deleting--upserting-dml) · [10.12 Transactions, views & indexes](#1012-transactions-views--indexes) · [10.13 JSON & arrays](#1013-json--arrays) · [10.14 Query performance](#1014-query-performance) · [10.15 SQL from Python & the command line](#1015-sql-from-python--the-command-line) · [10.16 Analytics & ML patterns](#1016-analytics--ml-patterns)
- **[11. LaTeX](#11-latex)** — [11.01 Document skeleton & compiling](#1101-document-skeleton--compiling) · [11.02 Structure](#1102-structure) · [11.03 Text formatting, special characters & spacing](#1103-text-formatting-special-characters--spacing) · [11.04 Lists](#1104-lists) · [11.05 Math mode basics](#1105-math-mode-basics) · [11.06 Symbols reference](#1106-symbols-reference) · [11.07 Multi-line equations & numbering](#1107-multi-line-equations--numbering) · [11.08 Matrices, cases, brackets & custom operators](#1108-matrices-cases-brackets--custom-operators) · [11.09 ML equations gallery](#1109-ml-equations-gallery) · [11.10 Figures](#1110-figures) · [11.11 Tables](#1111-tables) · [11.12 Cross-references, links & citations](#1112-cross-references-links--citations) · [11.13 Algorithms & code listings](#1113-algorithms--code-listings) · [11.14 Theorems & custom macros](#1114-theorems--custom-macros) · [11.15 Page layout, colors & boxes](#1115-page-layout-colors--boxes) · [11.16 TikZ & pgfplots: diagrams and plots](#1116-tikz--pgfplots-diagrams-and-plots) · [11.17 Beamer slides](#1117-beamer-slides) · [11.18 Troubleshooting](#1118-troubleshooting)
- **[12. Docker](#12-docker)** — [12.01 Setup & first commands](#1201-setup--first-commands) · [12.02 Running containers](#1202-running-containers) · [12.03 Managing containers](#1203-managing-containers) · [12.04 Images](#1204-images) · [12.05 Dockerfile instructions](#1205-dockerfile-instructions) · [12.06 Building images & .dockerignore](#1206-building-images--dockerignore) · [12.07 Best practices & multi-stage builds](#1207-best-practices--multi-stage-builds) · [12.08 Volumes & bind mounts](#1208-volumes--bind-mounts) · [12.09 Networking](#1209-networking) · [12.10 Docker Compose](#1210-docker-compose) · [12.11 GPUs & ML images](#1211-gpus--ml-images) · [12.12 Recipes: Jupyter & a model API](#1212-recipes-jupyter--a-model-api) · [12.13 Registries](#1213-registries) · [12.14 Debugging & cleanup](#1214-debugging--cleanup)

**[Part III — Data Structures & Algorithms](#part-iii--data-structures--algorithms)**

- **[13. Data Structures & Algorithms](#13-data-structures--algorithms)** — [13.01 Binary search](#1301-binary-search) · [13.02 Mergesort](#1302-mergesort) · [13.03 Quicksort](#1303-quicksort) · [13.04 Linked lists](#1304-linked-lists) · [13.05 Stacks](#1305-stacks) · [13.06 Queues](#1306-queues) · [13.07 Unbounded arrays](#1307-unbounded-arrays) · [13.08 Hash tables](#1308-hash-tables) · [13.09 Binary search trees](#1309-binary-search-trees) · [13.10 Dictionaries and sets](#1310-dictionaries-and-sets) · [13.11 Heaps](#1311-heaps) · [13.12 Priority queues](#1312-priority-queues)

---

# Part I — The Python ML stack

## 1. NumPy

`import numpy as np` · written against NumPy 2.x · the `ndarray` is the base type for pandas, scikit-learn, Matplotlib and (via `.numpy()`) PyTorch, so every later section reuses this one.

### 1.01 Setup, printing & help

```python
import numpy as np
np.__version__                                   # '2.x.y'
np.set_printoptions(precision=3, suppress=True, linewidth=120)   # suppress = no scientific notation for small floats
np.set_printoptions(threshold=50, edgeitems=2)   # summarize big arrays with "..."
np.get_printoptions()["precision"]
with np.printoptions(precision=1): print(np.pi * np.ones(3))   # temporary options, restored after the block
np.info(np.clip)                                 # docstring (Jupyter: np.clip? or help(np.clip))
np.set_printoptions(precision=8, suppress=False, threshold=1000, edgeitems=3, linewidth=75)   # back to defaults
```

**Drills:** (1) Print `np.pi * np.arange(5)` with 2 decimals without changing global options. (2) Show the docstring of `np.where` from code. (3) Make a 1,000-element array print only 3 items at each end.

### 1.02 Creating arrays

```python
# from Python data
a = np.array([1, 2, 3])                          # int64 on Linux/macOS
b = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
np.asarray(a)                                    # no copy if already an ndarray of the right dtype
np.array(a, copy=True)                           # always copies (NumPy 2: copy=False RAISES if a copy is required)
np.array(5).ndim                                 # 0 -> a 0-d "scalar array"
# filled arrays
np.zeros((2, 3)); np.ones((2, 3), dtype=int); np.full((2, 3), 7.5)
np.empty((2, 3))                                 # uninitialized memory: values are garbage until you write them
np.zeros_like(b); np.ones_like(b); np.full_like(b, -1); np.empty_like(b)   # same shape & dtype as b
np.eye(3); np.eye(3, 4, k=1); np.identity(3)     # k shifts the diagonal
# ranges
np.arange(5)                                     # [0 1 2 3 4]   stop is EXCLUSIVE
np.arange(2, 10, 3)                              # [2 5 8]
np.arange(0, 1, 0.25)                            # float steps work, but rounding can add/drop the last point
np.linspace(0, 1, 5)                             # [0. 0.25 0.5 0.75 1.]  stop is INCLUSIVE
np.linspace(0, 1, 5, endpoint=False, retstep=True)   # (array, step)
np.logspace(0, 3, 4)                             # [1. 10. 100. 1000.] (exponents of 10)
np.geomspace(1, 1000, 4)                         # same, specified by endpoints
# grids
xs, ys = np.meshgrid(np.arange(3), np.arange(2)) # each (2, 3); indexing="ij" for matrix-style order
np.mgrid[0:3, 0:2].shape                         # (2, 3, 2) dense grid
np.ogrid[0:3, 0:2]                               # open grid: (3,1) and (1,2), broadcastable
np.indices((2, 3)).shape                         # (2, 2, 3)
# special matrices
np.diag([1, 2, 3]); np.diag(b)                   # build diagonal matrix / extract a diagonal
np.diagflat([[1, 2], [3, 4]]); np.tri(3); np.tril(np.ones((3, 3))); np.triu(np.ones((3, 3)), k=1)
np.vander([1, 2, 3], N=3)                        # columns x^2, x, 1
# from functions, iterables, bytes
np.fromfunction(lambda i, j: i * 10 + j, (2, 3), dtype=int)
np.fromiter((x * x for x in range(5)), dtype=float)
np.frombuffer(b"\x01\x02\x03", dtype=np.uint8)
np.copy(a); a.copy()
```

**Watch out:** `np.empty` is not zeros. `np.arange` excludes its stop value; `np.linspace` includes it.

**Drills:** (1) Build a 5×5 matrix with 1s on the border and 0s inside. (2) Create 11 evenly spaced points from −1 to 1 with `linspace`, then try the same with `arange`. (3) Build a 4×4 multiplication table with `np.fromfunction`. (4) Make a 3×3 identity matrix in `float16`.

### 1.03 Random numbers

```python
rng = np.random.default_rng(seed=42)             # modern API: create ONE Generator and pass it around
rng.random((2, 3))                               # uniform [0, 1)
rng.uniform(-1, 1, size=5)
rng.integers(0, 10, size=(2, 3))                 # [0, 10); endpoint=True includes 10
rng.normal(loc=0, scale=1, size=3); rng.standard_normal((2, 2))
rng.choice(10, size=3, replace=False)            # sample without replacement
rng.choice(["a", "b", "c"], size=5, p=[0.5, 0.3, 0.2])
rng.permutation(5)                               # shuffled copy of arange(5)
rng.permutation(np.arange(10).reshape(5, 2))     # shuffles ROWS (axis 0) of a copy
arr = np.arange(10); rng.shuffle(arr)            # shuffles in place
rng.permuted(np.arange(12).reshape(3, 4), axis=1)    # shuffle within each row independently
rng.binomial(n=10, p=0.5, size=4); rng.poisson(lam=3, size=4); rng.exponential(scale=2, size=3)
rng.beta(2, 5, 3); rng.gamma(shape=2, scale=1, size=3); rng.lognormal(0, 1, 3); rng.laplace(0, 1, 3)
rng.multinomial(10, [0.2, 0.3, 0.5]); rng.dirichlet([1, 1, 1])
rng.multivariate_normal(mean=[0, 0], cov=[[1, 0.8], [0.8, 1]], size=100).shape   # (100, 2)
workers = rng.spawn(4)                           # 4 independent child generators (e.g. one per worker)
# legacy global-state API (common in older code; avoid in new code)
np.random.seed(0); np.random.rand(2, 3); np.random.randn(2); np.random.randint(0, 5, 3); np.random.choice(5, 2)
```

**Drills:** (1) Draw 1,000 samples from N(5, 2) with a seeded Generator and check the mean and std. (2) Sample 3 distinct indices out of 10, then 10 labels with probabilities [0.7, 0.2, 0.1]. (3) Shuffle the rows of a (6, 2) array without mixing values within a row. (4) Create 4 independent generators for 4 data-loader workers.

### 1.04 Attributes, dtypes & casting

```python
x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
x.shape, x.ndim, x.size, x.dtype, x.itemsize, x.nbytes   # (2, 3, 4) 3 24 float32 4 96
x.strides                                        # (48, 16, 4): bytes to step along each axis
len(x)                                           # 2 (length of the first axis)
x.T.shape                                        # (4, 3, 2): .T reverses ALL axes
x.base, x.flags["OWNDATA"]
# dtypes: int8 int16 int32 int64 uint8 ... float16 float32 float64 complex64 complex128 bool str_ object_ datetime64
np.dtype("float32"), np.dtype("i8"), np.dtype("<U10")    # string codes
x.astype(np.int32)                               # new array
x.astype("float32", copy=False)                  # no copy when already that dtype
np.array([1, 2, 3.5]).dtype                      # float64 (upcast to fit everything)
np.array([1, "a"]).dtype                         # <U21: everything became a string
np.iinfo(np.int8)                                # min -128, max 127
np.finfo(np.float32).eps, np.finfo(np.float16).max   # 1.19e-07, 65504.0
np.result_type(np.int8, np.float32), np.promote_types("i4", "f4")   # float32, float64
np.can_cast(np.float64, np.float32)              # False (would lose precision)
np.issubdtype(x.dtype, np.floating)              # True
np.array([250], dtype=np.uint8) + np.array([10], dtype=np.uint8)    # [4]: integer overflow wraps silently
np.float32(3) + 3.0                              # NumPy 2 keeps float32: Python scalars are "weak"
```

**Watch out:** Integer arrays overflow without raising an error. Mixing in a Python string turns the whole array into strings.

**Drills:** (1) Find the gap between 1.0 and the next representable `float32`. (2) Compute 200 + 100 in `uint8` and explain the result. (3) Compute the size of a (1000, 1000) `float64` array in MB from its attributes.

### 1.05 Indexing & slicing

```python
a = np.arange(10)                                # [0 1 2 3 4 5 6 7 8 9]
a[0], a[-1], a[2:5], a[:3], a[7:], a[::2], a[::-1], a[1:-1]
M = np.arange(12).reshape(3, 4)
M[1, 2], M[1][2]                                 # 6 both ways; the first form is faster
M[0], M[:, 0], M[:, -1], M[1:, :2], M[::2, ::-1]
M[..., 0]                                        # ... = "all remaining axes"
M[:, np.newaxis].shape, M[None].shape            # (3, 1, 4), (1, 3, 4)
# boolean masks: always return copies
M[M > 5]                                         # 1-D: [ 6  7  8  9 10 11]
M[(M > 2) & (M < 8)]                             # combine with & | ~ and parentheses (never `and`/`or`)
M[M.sum(axis=1) > 10]                            # keep rows whose sum > 10
B = M.copy(); B[B % 2 == 0] = -1                 # masked assignment
# integer ("fancy") indexing: copies
M[[0, 2]]                                        # rows 0 and 2
M[:, [3, 0]]                                     # columns 3 and 0, in that order
M[[0, 1, 2], [1, 2, 3]]                          # paired coordinates -> [ 1  6 11]
M[np.ix_([0, 2], [1, 3])]                        # sub-grid: rows {0,2} x cols {1,3} -> (2, 2)
probs = np.random.default_rng(0).random((3, 4)); labels = np.array([2, 0, 3])
probs[np.arange(3), labels]                      # one entry per row: score of each true class
# where / take / put / select
np.where(M > 5, M, 0)                            # if/else per element
np.where(M > 5)                                  # tuple of index arrays (rows, cols)
np.nonzero(M > 5); np.argwhere(M > 5); np.flatnonzero(M > 5)
np.take(M, [0, 5, 11]); np.take(M, [0, 2], axis=1)
order = np.argsort(M, axis=1); np.take_along_axis(M, order, axis=1)    # gather along an axis with an index array
C = M.copy(); np.put(C, [0, 1], [-5, -6])                              # assign by flat index
np.put_along_axis(C, np.zeros((3, 1), dtype=int), 99, axis=1)          # set column 0 of each row
np.select([M < 3, M < 6], [0, 1], default=2)    # multi-branch if/elif/else
np.choose([0, 1, 0], [[1, 2, 3], [10, 20, 30]]) # [ 1 20  3]
np.compress([True, False, True], M, axis=0); np.extract(M > 5, M)
np.place(C, C > 10, [0]); np.putmask(C, C < 0, 0)
```

**Watch out:** Slices return views, so writing to them changes the original. Masks and fancy indexing return copies. `M[[0, 1], [1, 2]]` selects the elements (0,1) and (1,2), not a 2×2 block; use `np.ix_` for a block.

**Drills:** (1) From `np.arange(36).reshape(6, 6)` extract the central 2×2 block, the last column, and every other row in reverse. (2) Replace negative values in a random array with 0, once with a mask and once with `np.where`. (3) Given `probs (N, C)` and `labels (N,)`, pull out each row's true-class probability. (4) List the (row, col) of every value > 0.9.

### 1.06 Views, copies & memory layout

```python
a = np.arange(6)
v = a[1:4]                                       # slice -> VIEW (shares memory)
v[0] = 100; a                                    # a is now [  0 100   2   3   4   5]
c = a[[1, 2, 3]]                                 # fancy index -> COPY
c[0] = -1; a[1]                                  # still 100
np.shares_memory(a, v), np.shares_memory(a, c)   # (True, False)
v.base is a                                      # True
a.reshape(2, 3).base is a                        # reshape returns a view when it can
a.copy()                                         # independent copy
a.view(np.int32).shape                           # (12,): same bytes reinterpreted as a different dtype
M = np.arange(12).reshape(3, 4)
M.flags["C_CONTIGUOUS"], M.T.flags["C_CONTIGUOUS"]   # (True, False): row-major vs transposed view
M.strides, M.T.strides                           # (32, 8) vs (8, 32)
np.ascontiguousarray(M.T).flags["C_CONTIGUOUS"]  # True (copies into row-major order)
np.asfortranarray(M).flags["F_CONTIGUOUS"]; np.zeros((3, 4), order="F")
from numpy.lib.stride_tricks import sliding_window_view
sliding_window_view(np.arange(6), window_shape=3)        # overlapping windows, no copy (read-only)
sliding_window_view(np.arange(16).reshape(4, 4), (2, 2)).shape   # (3, 3, 2, 2) image patches
x = np.ones(3); x += 1                           # in place: no new allocation
np.multiply(x, 2, out=x)                         # ufunc writes into an existing buffer
x.flags.writeable = False                        # read-only from now on (x[0] = 5 raises ValueError)
```

**Drills:** (1) Use `np.shares_memory` to find which of `a[::2]`, `a[[0, 2]]`, `a.reshape(2, -1)`, `a.T` and `a.flatten()` are views. (2) Make an array read-only and catch the error when writing to it. (3) Build overlapping windows of length 4 over `np.arange(10)` without copying.

### 1.07 Reshaping & moving axes

```python
a = np.arange(12)
a.reshape(3, 4); a.reshape(3, -1); a.reshape(-1, 2, 3)    # -1 = infer this dimension
a.reshape(3, 4, order="F")                       # fill column by column
M = a.reshape(3, 4)
M.ravel()                                        # 1-D, view when possible
M.flatten()                                      # 1-D, always a copy
M.T; M.transpose(); np.transpose(M)
X = np.zeros((2, 3, 4))
X.transpose(2, 0, 1).shape                       # (4, 2, 3)
np.swapaxes(X, 0, 2).shape                       # (4, 3, 2)
np.moveaxis(X, -1, 0).shape                      # (4, 2, 3)   e.g. HWC image -> CHW
np.permute_dims(X, (1, 0, 2)).shape              # (3, 2, 4)   array-API name (2.0+)
np.matrix_transpose(X).shape                     # (2, 4, 3)   swap the last two axes (2.0+)
np.expand_dims(a, axis=0).shape                  # (1, 12)
np.expand_dims(a, axis=(0, 2)).shape             # (1, 12, 1)
np.zeros((1, 3, 1)).squeeze().shape              # (3,)
np.zeros((1, 3, 1)).squeeze(axis=0).shape        # (3, 1)
np.atleast_1d(5).shape, np.atleast_2d([1, 2]).shape, np.atleast_3d(M).shape   # (1,), (1, 2), (3, 4, 1)
np.resize(np.arange(3), (2, 4))                  # repeats the data to fill the new shape
```

**Drills:** (1) Turn a (32, 32, 3) image into (3, 32, 32) in two different ways. (2) Turn a (28, 28) image into a batch of shape (1, 1, 28, 28). (3) Flatten a (10, 3, 4) batch to (10, 12).

### 1.08 Joining, splitting, repeating & padding

```python
a, b = np.array([[1, 2], [3, 4]]), np.array([[5, 6]])
np.concatenate([a, b], axis=0)                   # (3, 2): joins along an EXISTING axis
np.concatenate([a, a], axis=1)                   # (2, 4)
np.concatenate([a, b], axis=None)                # flattened
np.concat([a, b])                                # array-API alias (2.0+)
np.stack([a, a], axis=0).shape                  # (2, 2, 2): creates a NEW axis
np.vstack([a, b]); np.hstack([a, a]); np.dstack([a, a]).shape   # rows / columns / depth
np.column_stack([[1, 2], [3, 4]])                # 1-D inputs become columns -> [[1 3] [2 4]]
np.block([[a, a], [a, a]]).shape                 # (4, 4) block matrix
np.r_[1:4, 0, [9, 9]]                            # [1 2 3 0 9 9]
np.c_[np.array([1, 2]), np.array([3, 4])]        # [[1 3] [2 4]]
x = np.arange(9)
np.split(x, 3)                                   # 3 equal parts (must divide evenly)
np.split(x, [2, 5])                              # cut at indices: x[:2], x[2:5], x[5:]
np.array_split(x, 4)                             # unequal parts allowed
np.hsplit(np.arange(16).reshape(4, 4), 2); np.vsplit(np.arange(16).reshape(4, 4), [1])
np.tile([1, 2], 3)                               # [1 2 1 2 1 2]
np.tile([1, 2], (2, 2))                          # 2x2 grid of copies
np.repeat([1, 2], 3)                             # [1 1 1 2 2 2]
np.repeat([[1, 2], [3, 4]], [1, 2], axis=0)      # repeat each row a different number of times
np.append([1, 2], [3, 4])                        # copies everything; in loops, collect a list then call np.array once
np.insert(x, 1, 99); np.delete(x, [0, 1]); np.delete(np.eye(3), 1, axis=0)
np.flip(x); np.flip(np.eye(3), axis=0); np.fliplr(np.eye(3)); np.flipud(np.eye(3))
np.roll(x, 2); np.roll(np.eye(3), -1, axis=1); np.rot90(np.arange(4).reshape(2, 2))
np.pad(np.ones((2, 2)), pad_width=1)             # zero border -> (4, 4)
np.pad(np.ones((2, 2)), ((1, 0), (2, 3)), mode="constant", constant_values=9)   # (before, after) per axis
np.pad([1, 2, 3], 2, mode="edge"); np.pad([1, 2, 3], 2, mode="reflect"); np.pad([1, 2, 3], 2, mode="wrap")
np.trim_zeros(np.array([0, 0, 1, 2, 0]))         # [1 2]
```

**Drills:** (1) Stack five (3,) vectors into shape (5, 3) and into (3, 5). (2) Split `np.arange(100)` into 70/20/10 train/val/test parts. (3) Zero-pad a (28, 28) image to (32, 32), keeping it centered. (4) Build a 4×4 block matrix from two 2×2 matrices.

### 1.09 Broadcasting

```python
# Rule: line shapes up from the RIGHT; each pair of dims must be equal or one must be 1; missing dims count as 1.
np.ones((3, 4)) + np.arange(4)                   # (3,4) + (4,)   -> (3,4): add a row to every row
np.ones((3, 4)) + np.arange(3)[:, None]          # (3,4) + (3,1)  -> (3,4): add a column to every column
np.arange(3)[:, None] * np.arange(4)[None, :]    # (3,1) * (1,4)  -> (3,4): outer product
np.ones((2, 1, 5)) + np.ones((3, 1))             # (2,1,5) + (3,1) -> (2,3,5)
np.broadcast_shapes((2, 1, 5), (3, 1))           # (2, 3, 5): check shapes without computing
np.broadcast_to(np.arange(3), (4, 3))            # read-only view, no copy
xx, yy = np.broadcast_arrays(np.arange(3)[:, None], np.arange(2))
# np.ones((3, 4)) + np.arange(3)                 # ValueError: (3,4) vs (3,) don't align from the right
X = np.random.default_rng(0).normal(size=(100, 5))
Xc = X - X.mean(axis=0)                          # center each column: (100,5) - (5,)
Xs = (X - X.mean(axis=0)) / X.std(axis=0)        # standardize
unit_rows = X / np.linalg.norm(X, axis=1, keepdims=True)   # keepdims gives (100,1), which broadcasts per row
D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=-1))   # (100,100) distances via (100,1,5) - (1,100,5)
```

**Watch out:** A (N,) array minus an (N, 1) array broadcasts silently to (N, N). This is a common source of wrong losses.

**Drills:** (1) Predict the shape of `(8,1,6,1) + (7,1,5)`, then verify it. (2) Min-max scale every column of a (100, 4) matrix in one line. (3) Build a 5×5 matrix with `M[i, j] = i - j` without loops. (4) Compute all pairwise L1 distances between the rows of A (10, 3) and B (4, 3), giving shape (10, 4).

### 1.10 Elementwise math & ufuncs

```python
x = np.array([1.0, 4.0, 9.0]); y = np.array([2.0, 2.0, 2.0])
x + y, x - y, x * y, x / y, x // y, x % y, x ** y, -x, abs(-x)   # operators call ufuncs
np.add(x, y); np.subtract(x, y); np.multiply(x, y); np.divide(x, y); np.floor_divide(x, y); np.mod(x, y)
np.power(x, 2); np.negative(x); np.reciprocal(x); np.sqrt(x); np.cbrt(x); np.square(x)
np.exp(x); np.exp2(x); np.expm1(x); np.log(x); np.log2(x); np.log10(x); np.log1p(x)
np.sin(x); np.cos(x); np.tan(x); np.arcsin(0.5); np.arccos(0.5); np.arctan(1.0); np.arctan2(1.0, 1.0); np.hypot(3, 4)
np.sinh(x); np.cosh(x); np.tanh(x); np.deg2rad(180); np.rad2deg(np.pi)
np.pow(2, 3); np.acos(0.5); np.atan2(1, 1)      # array-API aliases (2.0+)
np.abs(-x); np.fabs(-x); np.sign([-2, 0, 3]); np.heaviside([-1, 0, 1], 0.5)
np.floor(2.7), np.ceil(2.1), np.trunc(-2.7), np.fix(-2.5)
np.rint(2.5), np.round(3.5), np.round(2.675, 2)  # 2.0 4.0 2.67: round-half-to-even, plus float representation error
np.clip(x, 2, 5); np.maximum(x, 5); np.minimum(x, 5)   # elementwise max/min of two arrays
np.fmax(x, np.nan); np.fmin(x, np.nan)           # like maximum/minimum but ignore NaN
np.gcd(12, 18), np.lcm(4, 6), np.divmod(7, 2), np.modf(2.5)
np.logaddexp(0.0, 0.0); np.copysign(3, -1); np.sinc(0.5)
# ufunc keyword arguments
out = np.empty(3); np.add(x, y, out=out)         # write into a preallocated array
a, b = np.array([1.0, 2.0, 3.0]), np.array([0.0, 2.0, 0.0])
np.divide(a, b, out=np.zeros_like(a), where=b != 0)   # safe division: 0 wherever b == 0
np.sqrt(x, dtype=np.float32)
# ufunc methods
np.add.reduce([1, 2, 3])                         # 6        (= sum)
np.add.accumulate([1, 2, 3])                     # [1 3 6]  (= cumsum)
np.maximum.accumulate([1, 3, 2, 5])              # [1 3 3 5] running max
np.multiply.outer([1, 2], [3, 4])                # [[3 4] [6 8]]
counts = np.zeros(4); np.add.at(counts, [0, 1, 1, 3], 1)   # [1. 2. 0. 1.]: repeated indices accumulate
np.add.reduceat(np.arange(6), [0, 2, 4])         # segment sums [1 5 9]
np.logical_or.reduce([False, True])
# wrapping your own Python functions
f = np.vectorize(lambda p, q: p if p > q else q) # convenience only: still a Python loop underneath
f(np.array([1, 5]), 3)
np.frompyfunc(lambda t: t * 2, 1, 1)(np.arange(3))
v = np.array([-2.0, 0.5, 3.0])
np.piecewise(v, [v < 0, v > 1], [0, 1, lambda t: t])   # different function per region -> [0.  0.5 1. ]
np.apply_along_axis(lambda r: r.max() - r.min(), 1, np.arange(12).reshape(3, 4))   # [3 3 3]
```

**Drills:** (1) Write a numerically safe softplus `log(1 + e^x)` with `np.logaddexp`. (2) Count occurrences of each index in `[0, 1, 1, 3, 3, 3]` with `np.add.at`. (3) Compute the running maximum of a random walk. (4) Divide `a` by `b`, writing 0 where `b == 0`, with no warnings.

### 1.11 Reductions & statistics

```python
M = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
M.sum(), M.sum(axis=0), M.sum(axis=1)            # 21.0, [5. 7. 9.] (per column), [ 6. 15.] (per row)
M.sum(axis=1, keepdims=True)                     # [[ 6.] [15.]] keeps a size-1 axis so it broadcasts
M.sum(axis=(0, 1)); np.sum(M, dtype=np.float64)
M.mean(axis=0); M.std(); M.std(ddof=1); M.var(axis=1)    # ddof=1 = sample std (pandas' default)
M.min(), M.max(axis=0), np.ptp(M, axis=1)        # ptp = max - min
M.argmin(), M.argmax(axis=1), np.argmax(M, axis=1, keepdims=True)
np.unravel_index(M.argmax(), M.shape)            # (1, 2): flat index -> coordinates
np.ravel_multi_index((1, 2), M.shape)            # 5: coordinates -> flat index
M.prod(); M.cumsum(); M.cumsum(axis=1); M.cumprod(axis=0)
np.cumulative_sum(M, axis=1, include_initial=True)   # prepends 0 (2.1+)
np.median(M); np.median(M, axis=0); np.percentile(M, [25, 50, 75]); np.quantile(M, 0.9, axis=1)
np.average(M, axis=1, weights=[1, 0, 1])         # weighted mean
np.average([1, 2, 3], weights=[3, 1, 1], returned=True)   # (mean, sum of weights)
np.corrcoef(M)                                   # each ROW is a variable -> (2, 2)
np.cov(M, rowvar=False)                          # each COLUMN is a variable -> (3, 3)
np.histogram(np.arange(10), bins=5)              # (counts, bin_edges)
np.histogram(np.arange(10), bins=[0, 3, 7, 10], density=True)
np.histogram_bin_edges(np.arange(10), bins="auto"); np.histogram2d([0, 1, 1], [1, 0, 1], bins=2)
np.bincount([0, 1, 1, 3]); np.bincount([0, 1, 1], weights=[0.5, 1.0, 2.0], minlength=4)
np.digitize([0.1, 0.5, 0.9], bins=[0.25, 0.75]) # bucket ids [0 1 2]
np.count_nonzero(M > 2, axis=0); (M > 2).sum(); (M > 2).mean()   # True counts as 1
np.all(M > 0); np.any(M > 5, axis=1)
np.mean(M, where=M > 1)                          # reduce only over selected elements
np.sum([[1, 2], [3, 4]], axis=0, initial=10)     # [14 16]
# NaN-aware versions skip NaN
N = np.array([1.0, np.nan, 3.0])
N.sum(), np.nansum(N), np.nanmean(N), np.nanstd(N), np.nanmax(N), np.nanargmin(N), np.nanmedian(N)
np.nanpercentile(N, 50); np.nancumsum(N); np.nanvar(N); np.nanprod(N)
```

**Watch out:** `axis=0` collapses the rows and leaves one value per column. NumPy's `std` uses `ddof=0` while pandas uses `ddof=1`.

**Drills:** (1) For a (5, 3) matrix compute row means as a (5, 1) column. (2) Find the (row, col) of the global maximum. (3) Compute the 10th, 50th and 90th percentile of each column. (4) Compute the mean ignoring NaNs and the fraction of values that are NaN.

### 1.12 Comparisons, logic & missing values

```python
a, b = np.array([1, 2, 3]), np.array([1, 5, 3])
a == b, a != b, a < b, a >= 2                    # elementwise -> boolean arrays
np.equal(a, b); np.greater(a, b); np.less_equal(a, b)
np.array_equal(a, b); np.array_equal([1, np.nan], [1, np.nan], equal_nan=True)   # False, True
np.array_equiv([1, 1], [[1, 1], [1, 1]])         # True (equal after broadcasting)
np.allclose([1.0, 2.0], [1.0, 2.0 + 1e-9]); np.isclose(0.1 + 0.2, 0.3)   # never use == on floats
np.logical_and(a > 1, b > 1); np.logical_or(a > 2, b > 2); np.logical_not(a > 1); np.logical_xor(a > 1, b > 1)
(a > 1) & (b > 1), (a > 2) | (b > 2), ~(a > 1), (a > 1) ^ (b > 1)   # operator forms need parentheses
# `if a > 1:` raises "truth value of an array is ambiguous" -> use (a > 1).any() or (a > 1).all()
x = np.array([1.0, np.nan, np.inf, -np.inf])
np.isnan(x); np.isinf(x); np.isfinite(x); np.isposinf(x); np.isneginf(x)
np.nan == np.nan                                 # False! test NaN with np.isnan
x[~np.isnan(x)]                                  # drop NaN
np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
np.where(np.isnan(x), 0, x)
m = np.ma.masked_invalid(x)                      # masked array: operations skip masked entries
m.mean(), m.filled(0), m.compressed()
np.ma.masked_where(np.array([1, 2, 3]) > 2, [1, 2, 3])
with np.errstate(divide="ignore", invalid="ignore"): r = np.array([1.0, 0.0]) / 0.0   # silence warnings locally
np.isreal(np.array([1 + 0j, 1 + 1j])); np.iscomplexobj(np.array([1j])); np.isscalar(3.0)
```

**Drills:** (1) Return True if any row of a matrix is all zeros. (2) Check two float arrays are equal within 1e-6. (3) Replace each NaN with its column's mean.

### 1.13 Sorting, searching, sets & counting

```python
a = np.array([3, 1, 2, 3, 0])
np.sort(a); s = a.copy(); s.sort()               # sorted copy vs in-place sort
np.sort(a)[::-1]                                 # descending
np.argsort(a); np.argsort(-a); np.argsort(a, kind="stable")
M = np.array([[3, 1], [2, 4]]); np.sort(M, axis=0); np.sort(M, axis=None)
M[M[:, 1].argsort()[::-1]]                       # sort ROWS by column 1, descending
names, ages = np.array(["b", "a", "c"]), np.array([30, 30, 20])
np.lexsort((names, ages))                        # [2 1 0]: sort by ages, ties by names (LAST key is primary)
np.partition(a, 2)                               # element at index 2 lands in sorted position; O(n)
v = np.array([5, 1, 9, 7, 3]); k = 2
top = np.argpartition(v, -k)[-k:]                # indices of the top-k (unordered), O(n)
top[np.argsort(-v[top])]                         # ...now sorted largest first
np.searchsorted([1, 3, 5, 7], 4)                 # 2: insertion index that keeps the order
np.searchsorted([1, 3, 5, 7], [1, 7], side="right")
np.unique([3, 1, 2, 3, 1])                       # [1 2 3]
vals, first_idx, inverse, counts = np.unique([3, 1, 2, 3, 1], return_index=True, return_inverse=True, return_counts=True)
dict(zip(vals.tolist(), counts.tolist()))        # {1: 2, 2: 1, 3: 2}
np.unique(np.array([[1, 2], [1, 2], [3, 4]]), axis=0)   # unique rows
np.unique_values([3, 1, 3]); np.unique_counts([3, 1, 3]); np.unique_inverse([3, 1, 3]); np.unique_all([3, 1, 3])   # 2.0+
np.isin([1, 2, 5], [2, 5, 7]); np.isin([1, 2], [2], invert=True)
np.intersect1d([1, 2, 3], [2, 3, 4]); np.union1d([1, 2], [2, 3]); np.setdiff1d([1, 2, 3], [2]); np.setxor1d([1, 2], [2, 3])
np.argmax(a == 3)                                # index of the FIRST match (also 0 when there is no match; check .any())
np.bincount(a).argmax()                          # most frequent non-negative integer
```

**Drills:** (1) Get the indices of the 3 largest values, largest first. (2) Return a dict of unique values and their counts. (3) Sort the rows of a matrix by its second column, descending. (4) Find the elements of `a` that are not in `b`.

### 1.14 Linear algebra & einsum

```python
A = np.array([[2.0, 1.0], [1.0, 3.0]]); B = np.arange(6.0).reshape(2, 3); v = np.array([1.0, 2.0])
A @ B; np.matmul(A, B); np.dot(A, B)             # matrix product (prefer @)
A @ v, v @ A, v @ v                              # matrix-vector, vector-matrix, dot product
np.dot(v, v); np.vdot(v, v); np.inner(v, v); np.outer(v, v)
np.vecdot(v, v); np.matvec(A, v); np.vecmat(v, A)   # explicit batched versions (2.0+/2.2+)
X = np.random.default_rng(0).normal(size=(10, 3, 4)); W = np.ones((4, 5))
(X @ W).shape                                    # (10, 3, 5): @ broadcasts over leading batch dims
np.tensordot(B, B.T, axes=([1], [0])).shape      # (2, 2): sum over the chosen axes
np.kron(np.eye(2), A).shape; np.cross([1, 0, 0], [0, 1, 0]); np.trace(A); np.diagonal(A)
np.linalg.det(A); np.linalg.inv(A); np.linalg.pinv(B)
x = np.linalg.solve(A, v)                        # solves A x = v; faster and more stable than inv(A) @ v
np.allclose(A @ x, v)
sol, residuals, rank, sv = np.linalg.lstsq(B.T, np.ones(3), rcond=None)   # least squares
w, V = np.linalg.eig(A); w, V = np.linalg.eigh(A)   # eigh: symmetric/Hermitian; real values sorted ascending
np.linalg.eigvals(A); np.linalg.eigvalsh(A)
U, S, Vt = np.linalg.svd(B, full_matrices=False)    # B == U @ np.diag(S) @ Vt
np.allclose(B, U @ np.diag(S) @ Vt)
np.linalg.svdvals(B)                             # singular values only (2.0+)
Q, R = np.linalg.qr(B.T); L = np.linalg.cholesky(A)   # cholesky needs a symmetric positive-definite matrix
np.linalg.matrix_rank(B); np.linalg.matrix_power(A, 3); np.linalg.cond(A); np.linalg.slogdet(A)
np.linalg.norm(v); np.linalg.norm(v, ord=1); np.linalg.norm(A, "fro"); np.linalg.norm(B, axis=1)
np.linalg.vector_norm(B, axis=1); np.linalg.matrix_norm(A, ord=2)   # explicit versions (2.0+)
np.linalg.multi_dot([A, A, B])                   # chooses the cheapest multiplication order
# einsum: label each axis; repeated labels are multiplied; labels missing from the output are summed
np.einsum("ij,jk->ik", A, B)                     # matmul
np.einsum("ii->", A), np.einsum("ii->i", A)      # trace, diagonal
np.einsum("ij->ji", B); np.einsum("ij->", B); np.einsum("ij->j", B)   # transpose, total sum, column sums
np.einsum("i,j->ij", v, v); np.einsum("i,i->", v, v)                  # outer, dot
np.einsum("bij,bjk->bik", X, np.ones((10, 4, 2))).shape               # batched matmul (10, 3, 2)
np.einsum("bij,jk->bik", X, W, optimize=True).shape                   # optimize=True picks a fast contraction order
np.einsum("...ij,...jk->...ik", X, np.ones((4, 2))).shape             # ... = any leading dims
```

**Drills:** (1) Solve a random 3×3 system and verify `A @ x ≈ b`. (2) Build the rank-1 approximation of a matrix from its SVD. (3) Write a batched matmul (B,N,D)@(B,D,M) with both `einsum` and `@`. (4) Compute the trace of every matrix in a (B, N, N) stack with `einsum`.

### 1.15 FFT, polynomials & signal helpers

```python
t = np.linspace(0, 1, 500, endpoint=False)
sig = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 40 * t)
F = np.fft.rfft(sig); freqs = np.fft.rfftfreq(sig.size, d=t[1] - t[0])
freqs[np.argsort(np.abs(F))[-2:]]                # the two dominant frequencies: [40.  5.]
np.fft.irfft(F, n=sig.size)                      # back to the signal
np.fft.fft(sig); np.fft.ifft(np.fft.fft(sig)).real; np.fft.fftfreq(8, d=0.1)
np.fft.fftshift(np.fft.fftfreq(8)); np.fft.fft2(np.eye(4)); np.fft.fftn(np.ones((2, 2, 2)))
# polynomials
coeffs = np.polyfit([0, 1, 2, 3], [1, 3, 7, 13], deg=2)   # [1. 1. 1.]: highest power FIRST
np.polyval(coeffs, 4)                            # 21.0
p = np.poly1d(coeffs); p(4), p.deriv(), p.integ()
np.roots([1, -3, 2])                             # roots of x^2 - 3x + 2 -> [2. 1.]
from numpy.polynomial import Polynomial
P = Polynomial.fit([0, 1, 2, 3], [1, 3, 7, 13], deg=2); P(4); P.convert().coef   # LOWEST power first
# 1-D signal helpers
np.convolve(np.ones(10), np.ones(3) / 3, mode="valid")   # 3-point moving average
np.convolve([1, 2, 3], [0, 1, 0.5], mode="full"); np.correlate([1, 2, 3], [0, 1, 0.5], mode="valid")
np.interp(2.5, [1, 2, 3], [10, 20, 30])          # 25.0 linear interpolation
np.diff([1, 4, 9, 16]); np.diff([1, 4, 9, 16], n=2); np.ediff1d([1, 4, 9]); np.gradient([1.0, 4.0, 9.0, 16.0])
np.trapezoid([1, 2, 3], dx=1.0)                  # 4.0: integral by the trapezoid rule (np.trapz before 2.0)
np.unwrap(np.array([0.0, 3.0, 6.5]))             # phase unwrapping
```

**Drills:** (1) Find the dominant frequency of a noisy 7 Hz sine wave. (2) Fit a cubic to noisy data and evaluate it at 10 new points. (3) Compute a 5-point moving average with `np.convolve`.

### 1.16 Strings, dates, structured arrays & bits

```python
s = np.array(["Apple", "banana ", " Cherry"])
np.strings.upper(s); np.strings.strip(s); np.strings.str_len(s)    # np.strings (2.0+); np.char is the older namespace
np.strings.startswith(s, "A"); np.strings.find(s, "an"); np.strings.replace(s, "a", "A"); np.strings.add(s, "!")
np.char.lower(s); np.char.split(np.array(["a,b", "c,d"]), ",")
# datetimes
d = np.datetime64("2024-01-15"); d + np.timedelta64(30, "D")       # numpy.datetime64('2024-02-14')
np.arange("2024-01", "2024-05", dtype="datetime64[M]")             # months Jan..Apr
np.datetime64("2024-03-01") - np.datetime64("2024-02-01")          # 29 days (leap year)
np.datetime64("2024-01-15T13:45").astype("datetime64[D]")          # truncate to the day
(np.datetime64("2024-01-02") - np.datetime64("2024-01-01")) / np.timedelta64(1, "h")   # 24.0 hours
np.busday_count("2024-01-01", "2024-02-01"); np.is_busday("2024-01-06"); np.busday_offset("2024-01-05", 1, roll="forward")
# structured arrays: a lightweight table (use pandas for real tabular work)
people = np.array([("Ana", 31, 1.62), ("Ben", 25, 1.80)], dtype=[("name", "U10"), ("age", "i4"), ("height", "f4")])
people["age"]; people[people["age"] > 28]["name"]; np.sort(people, order="age")
np.rec.array(people).age.mean()
# bits & bytes
np.packbits(np.array([1, 0, 1, 1, 0, 0, 0, 1], dtype=np.uint8)); np.unpackbits(np.array([177], dtype=np.uint8))
x = np.array([5], dtype=np.uint8); np.left_shift(x, 1); x >> 1; x & 0b100; np.bitwise_xor(x, 1); np.invert(x)
```

**Drills:** (1) Count the business days in March 2024. (2) Make an array of the first day of every month in 2024. (3) Build a structured array of (name, score) records and sort it by score.

### 1.17 File I/O

```python
a = np.arange(6).reshape(2, 3)
np.save("a.npy", a); np.load("a.npy")            # binary; keeps exact dtype and shape (fastest)
np.savez("arrays.npz", features=a, labels=np.array([0, 1])); np.savez_compressed("arrays_c.npz", features=a)
with np.load("arrays.npz") as data: print(data.files, data["features"].shape, data["labels"])
np.savetxt("a.csv", a, delimiter=",", fmt="%d", header="x,y,z", comments="")
np.loadtxt("a.csv", delimiter=",", skiprows=1)
np.loadtxt("a.csv", delimiter=",", skiprows=1, usecols=(0, 2), dtype=int)
np.genfromtxt("a.csv", delimiter=",", names=True)   # handles missing values and named columns
np.load("a.npy", mmap_mode="r")                  # memory-map: slice without loading the whole file
mm = np.memmap("big.dat", dtype="float32", mode="w+", shape=(1000, 128)); mm[:10] = 1.0; mm.flush()
a.tofile("a.bin"); np.fromfile("a.bin", dtype=a.dtype).reshape(2, 3)   # raw bytes: no shape/dtype stored!
a.tolist(); a.tobytes(); np.frombuffer(a.tobytes(), dtype=a.dtype)
# np.load(..., allow_pickle=True) executes pickle: only for files you trust
```

**Drills:** (1) Save features and labels into one `.npz` and load them back. (2) Write a (100, 3) float array to CSV with a header and 4 decimals, then read it back. (3) Memory-map a `.npy` file and read only rows 10–20.

### 1.18 Vectorized ML patterns & testing

```python
rng = np.random.default_rng(0)
logits = rng.normal(size=(4, 3)); y = np.array([0, 2, 1, 2])
z = logits - logits.max(axis=1, keepdims=True)   # subtract the row max for numerical stability
probs = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)           # softmax
log_probs = z - np.log(np.exp(z).sum(axis=1, keepdims=True))       # log-softmax
loss = -log_probs[np.arange(len(y)), y].mean()   # cross-entropy
acc = (probs.argmax(axis=1) == y).mean()
one_hot = np.eye(3)[y]                           # (4, 3); one_hot.argmax(1) recovers y
sigmoid = lambda t: 1 / (1 + np.exp(-t))
p = sigmoid(np.array([-2.0, 0.0, 3.0])); target = np.array([0, 1, 1])
bce = -np.mean(target * np.log(p + 1e-12) + (1 - target) * np.log(1 - p + 1e-12))
# a linear layer: forward and backward by hand
X = rng.normal(size=(32, 5)); W = rng.normal(size=(5, 2)) * 0.1; b = np.zeros(2)
out = X @ W + b                                  # (32, 2)
dout = np.ones_like(out)                         # upstream gradient dL/dout
dW, db, dX = X.T @ dout, dout.sum(axis=0), dout @ W.T
# linear regression: closed form vs gradient descent
true_w = np.array([2.0, -1.0, 0.5, 0.0, 3.0]); yr = X @ true_w + 0.1 * rng.normal(size=32)
w_closed = np.linalg.lstsq(X, yr, rcond=None)[0]
w = np.zeros(5)
for _ in range(500):
    grad = 2 / len(X) * X.T @ (X @ w - yr)
    w -= 0.05 * grad
np.allclose(w, w_closed, atol=1e-3)
# pairwise squared distances without building an (n, m, d) intermediate
A, B = rng.normal(size=(100, 8)), rng.normal(size=(50, 8))
d2 = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T   # (100, 50)
nearest = d2.argmin(axis=1)                      # 1-nearest-neighbour index
cos = (A / np.linalg.norm(A, axis=1, keepdims=True)) @ (B / np.linalg.norm(B, axis=1, keepdims=True)).T
# shuffled minibatches, train/val split, standardization with TRAIN statistics only
idx = rng.permutation(len(X)); batches = [idx[i:i + 8] for i in range(0, len(idx), 8)]; xb = X[batches[0]]
cut = int(0.8 * len(X)); X_tr, X_val = X[idx[:cut]], X[idx[cut:]]
mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-8
X_tr, X_val = (X_tr - mu) / sd, (X_val - mu) / sd
# image batches: uint8 -> float in [0, 1], per-channel normalize, NHWC -> NCHW
imgs = rng.integers(0, 256, size=(8, 32, 32, 3), dtype=np.uint8)
xim = imgs.astype(np.float32) / 255.0
xim = (xim - xim.mean(axis=(0, 1, 2))) / xim.std(axis=(0, 1, 2))
xim.transpose(0, 3, 1, 2).shape                  # (8, 3, 32, 32)
# confusion matrix without loops
y_true, y_pred = np.array([0, 1, 2, 2, 1]), np.array([0, 2, 2, 2, 1])
cm = np.zeros((3, 3), dtype=int); np.add.at(cm, (y_true, y_pred), 1)
np.bincount(3 * y_true + y_pred, minlength=9).reshape(3, 3)   # same matrix
# assertions for tests and sanity checks
np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-6)
np.testing.assert_array_equal(one_hot.argmax(1), y)
np.testing.assert_equal(cm.trace(), 4)
```

**Drills:** (1) Implement softmax and cross-entropy for (N, C) logits and check them against a Python loop. (2) Write a 1-nearest-neighbour classifier with the d² trick. (3) Train logistic regression with 200 steps of gradient descent in pure NumPy. (4) Build a confusion matrix with `np.bincount`.

Full reference: https://numpy.org/doc/stable/reference/

## 2. pandas

`import pandas as pd` · written against pandas 3.x (nearly all of it also runs on 2.2). A Series is a labeled 1-D array; a DataFrame is a table of Series that share one index. pandas 3 changed three things you will hit right away. Copy-on-Write is always on, so chained assignment like `df[mask]["col"] = x` never modifies `df`. Text columns default to the `str` dtype instead of `object`. Frequency aliases are now `"ME"`, `"h"`, `"min"` and `"s"` (not `"M"`, `"H"`, `"T"` or `"S"`).

### 2.01 Setup, options & core objects

```python
import numpy as np
import pandas as pd
pd.__version__
pd.set_option("display.max_columns", 50); pd.set_option("display.width", 120); pd.set_option("display.precision", 3)
pd.set_option("display.float_format", "{:,.2f}".format); pd.reset_option("display.float_format")
with pd.option_context("display.max_rows", 10): print(pd.DataFrame({"a": range(100)}))   # temporary option
pd.get_option("display.max_rows")
s = pd.Series([10, 20, 30], index=["a", "b", "c"], name="points")   # values + index + name
s.index, s.values, s.array, s.name, s.dtype
s["b"], s.iloc[0], s[["a", "c"]], s[s > 15]
pd.Series({"a": 1, "b": 2}); pd.Series(5, index=range(3)); pd.Series(np.arange(3), dtype="float32")
idx = pd.Index(["x", "y", "z"], name="key"); idx.get_loc("y"), idx.is_unique, idx.tolist()
pd.RangeIndex(5); pd.date_range("2024-01-01", periods=3); pd.CategoricalIndex(["a", "b", "a"])
```

**Drills:** (1) Build a Series of 5 temperatures indexed by weekday names and select Tue–Thu. (2) Temporarily show 3 decimals inside a `with` block. (3) Find the position of label "y" in an Index.

### 2.02 Creating DataFrames & dtypes

```python
def make_df():                                   # the practice table used throughout this section
    return pd.DataFrame({
        "name":   ["Ana", "Ben", "Cal", "Dee", "Eve", "Fay"],
        "team":   ["red", "blue", "red", "green", "blue", "red"],
        "score":  [88.0, 92.5, np.nan, 75.0, 92.5, 60.0],
        "age":    [23, 35, 31, 42, 28, 35],
        "joined": pd.to_datetime(["2023-01-15", "2022-06-01", "2023-03-20", "2021-11-11", "2024-02-29", "2022-06-01"]),
    })
df = make_df()                                   # dict of columns: the most common constructor
pd.DataFrame([{"a": 1, "b": 2}, {"a": 3, "b": 4}])           # list of row dicts (e.g. parsed JSON)
pd.DataFrame([[1, 2], [3, 4]], columns=["a", "b"], index=["r1", "r2"])   # list of rows
pd.DataFrame(np.random.default_rng(0).normal(size=(3, 2)), columns=["x", "y"])
pd.DataFrame.from_dict({"r1": {"a": 1}, "r2": {"a": 2}}, orient="index")
pd.DataFrame.from_records([(1, "x"), (2, "y")], columns=["id", "tag"])
df.dtypes                                        # name/team: str (pandas 3) or object (pandas 2), score: float64, age: int64, joined: datetime64
df.astype({"age": "int32", "team": "category"}).dtypes
pd.array([1, None, 3], dtype="Int64")            # nullable integer: holds pd.NA instead of becoming float
pd.Series([True, None], dtype="boolean"); pd.Series(["a", None], dtype="string"); pd.Series([1.5, None], dtype="Float64")
df.convert_dtypes().dtypes                       # best nullable dtype for every column
df.infer_objects()                               # fix object columns that actually hold numbers
pd.NA, pd.NaT, np.nan                            # missing markers: nullable dtypes / datetimes / floats
```

**Drills:** (1) Build a 3-row DataFrame from a list of dicts where one dict is missing a key. What appears? (2) Store ages with a missing value without turning them into floats. (3) Convert `team` to `category` and compare memory use.

### 2.03 Reading & writing data

```python
df = make_df()
df.to_csv("people.csv", index=False)             # index=False unless the index carries meaning
pd.read_csv("people.csv")
pd.read_csv("people.csv", usecols=["name", "score"], dtype={"name": "string"}, nrows=3)
pd.read_csv("people.csv", parse_dates=["joined"], index_col="name")
pd.read_csv("people.csv", na_values=["?", "NA", ""], skiprows=[1], header=0)
pd.read_csv("people.csv", names=["n", "t", "s", "a", "j"], header=0)   # rename columns while reading
for chunk in pd.read_csv("people.csv", chunksize=2): print(chunk.shape)   # stream files bigger than RAM
pd.read_csv("people.csv", sep=",", encoding="utf-8", thousands=",", comment="#", on_bad_lines="skip")
df.to_parquet("people.parquet", index=False); pd.read_parquet("people.parquet", columns=["name", "age"])   # typed, compressed, columnar: best for ML data (needs pyarrow)
df.to_json("people.jsonl", orient="records", lines=True, date_format="iso"); pd.read_json("people.jsonl", lines=True)
df.to_pickle("people.pkl"); pd.read_pickle("people.pkl")             # only unpickle files you trust
df.to_excel("people.xlsx", sheet_name="data", index=False); pd.read_excel("people.xlsx", sheet_name="data")   # needs openpyxl
with pd.ExcelWriter("multi.xlsx") as xw: df.to_excel(xw, sheet_name="a"); df.head(2).to_excel(xw, sheet_name="b")
pd.read_excel("multi.xlsx", sheet_name=None).keys()                   # dict of ALL sheets
records = [{"id": 1, "user": {"name": "Ana", "tags": ["x"]}}, {"id": 2, "user": {"name": "Ben", "tags": []}}]
pd.json_normalize(records)                                            # nested dicts -> columns id, user.name, user.tags
import sqlite3
con = sqlite3.connect(":memory:"); df.to_sql("people", con, index=False, if_exists="replace")
pd.read_sql("SELECT team, AVG(score) AS avg_score FROM people GROUP BY team", con)
pd.read_sql("SELECT * FROM people WHERE age > ?", con, params=(30,))   # parameters, never f-strings
df.to_dict(orient="records")[:2]; df.to_dict(orient="list"); df.to_numpy()
df.to_markdown(index=False); df.to_string(); df.to_html(); df.to_latex(index=False)   # tables for READMEs, reports, papers
# also: pd.read_table (TSV) · read_fwf · read_html · read_xml · read_feather · read_orc · read_stata · read_spss · read_clipboard
```

**Drills:** (1) Write `df` to CSV and read back only `name` and `age`, with `name` as the index. (2) Round-trip `df` through JSON Lines and check that the dtypes survived. (3) Load a table into SQLite and query the average score per team.

### 2.04 Inspecting data

```python
df = make_df()
df.head(); df.head(3); df.tail(2); df.sample(2, random_state=0); df.sample(frac=0.5, random_state=0)
df.shape, len(df), df.size, df.ndim, df.columns, df.index, df.empty
df.info(); df.info(memory_usage="deep")
df.describe()                                    # count, mean, std, min, quartiles, max for numeric columns
df.describe(include="all"); df.describe(exclude="number"); df.describe(percentiles=[0.1, 0.9])
df["team"].value_counts(); df["team"].value_counts(normalize=True); df["score"].value_counts(bins=3, sort=False)
df.value_counts(["team", "age"])                 # counts of column combinations
df["team"].unique(); df["team"].nunique(); df.nunique()
df.isna().sum(); df.isna().mean().round(2); df.notna().all()
df.duplicated().sum(); df.duplicated(subset=["score"]).sum()
df.memory_usage(deep=True)
df["score"].agg(["min", "max", "mean", "median", "std"])
df.corr(numeric_only=True); df["score"].corr(df["age"])
df["age"].quantile([0.25, 0.5, 0.75]); df["age"].mode(); df.loc[df["score"].idxmax(), "name"]
df["age"].is_unique, df["age"].is_monotonic_increasing, df.index.is_unique
df.T                                             # transpose: rows <-> columns
```

**Drills:** (1) Find the share of missing values in every column. (2) Return the name of the oldest person. (3) Count the rows for each (team, age) pair.

### 2.05 Selecting & filtering

```python
df = make_df()
df["name"]                                       # one column -> Series
df[["name", "score"]]                            # list of columns -> DataFrame
df.name                                          # attribute access: simple names only, and never for assignment
df[1:3]                                          # a slice inside [] selects ROWS by position
# .loc = labels (slice end INCLUDED) · .iloc = integer positions (slice end excluded)
df.loc[0], df.loc[0, "name"]
df.loc[0:2, ["name", "age"]]; df.loc[:, "team":"age"]
df.iloc[0]; df.iloc[-1]; df.iloc[0:2, 0:3]; df.iloc[[0, 2], [1, 3]]; df.iloc[:, -1]
df.at[0, "name"], df.iat[0, 0]                   # fast single-cell access
# boolean masks
df[df["age"] > 30]
df[(df["age"] > 30) & (df["team"] == "red")]     # & | ~ with parentheses around each condition
df[~df["team"].isin(["red", "blue"])]
df[df["score"].between(70, 90)]                  # inclusive at both ends by default
df[df["name"].str.startswith("A")]
df[df["score"].isna()]; df[df["score"].notna()]
df.loc[df["age"] > 30, ["name", "score"]]        # rows by mask, columns by label
df.loc[lambda d: d["score"] > 80]                # callable: works mid-chain
# query strings
df.query("age > 30 and team == 'red'")
min_age = 30; df.query("age > @min_age")         # @ references a Python variable
df.query("team in ['red', 'green']")
# column-selection helpers
df.filter(items=["name", "age"]); df.filter(like="a"); df.filter(regex=r"^(name|age)$")
df.select_dtypes(include="number"); df.select_dtypes(exclude="number")
df.nlargest(2, "score"); df.nsmallest(2, "age"); df.get("missing_col", default=None)
df.where(df["age"] > 30)                         # same shape; rows that fail become NaN
df["score"].where(df["score"] > 70, other=0); df["score"].mask(df["score"] > 70, other=100)
```

**Watch out:** `df.loc[0:2]` returns 3 rows, because label slices include the end. `df.iloc[0:2]` returns 2.

**Drills:** (1) Select the names of red-team players older than 25. (2) Select the 2nd–4th rows and the last two columns by position. (3) Write the same filter once with a boolean mask and once with `query`.

### 2.06 Adding, modifying & removing columns

```python
df = make_df()
df["age_months"] = df["age"] * 12                # new column from vectorized math
df["senior"] = df["age"] >= 35
df["grade"] = np.where(df["score"] >= 80, "A", "B")   # if/else per row
df.insert(1, "id", range(len(df)))               # insert at a position
df = df.assign(score_z=lambda d: (d["score"] - d["score"].mean()) / d["score"].std(),
               name_len=lambda d: d["name"].str.len())   # returns a NEW frame, good for chaining
df.loc[df["team"] == "red", "score"] += 1        # conditional update: mask + column in ONE .loc
df.loc[0, "score"] = 90.0                        # single cell
df = df.rename(columns={"name": "player", "team": "squad"})
df = df.rename(columns=str.upper).rename(columns=str.lower)   # a function is applied to every label
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]   # clean all names at once
df = df.drop(columns=["age_months", "name_len"])
df.drop(index=[0, 1]); df.drop(columns=["missing"], errors="ignore")
senior = df.pop("senior")                        # remove a column and return it
df.add_prefix("f_").columns; df.add_suffix("_v1").columns
df = df.set_index("id"); df = df.reset_index()   # move a column into / out of the index
df.reset_index(drop=True)
df.reindex(columns=["player", "squad", "score"]) # select/reorder columns (missing ones become NaN)
df[["squad", "player"] + [c for c in df.columns if c not in ("squad", "player")]]   # move columns to the front
df.copy()                                        # explicit (deep) copy
sub = df[df["age"] > 30]; sub["score"] = 0       # Copy-on-Write: modifies sub only, never df
# df[df["age"] > 30]["score"] = 0                # chained assignment: does NOTHING in pandas 3 -> use df.loc[mask, "score"] = 0
```

**Drills:** (1) Add a `bmi` column from `weight` and `height` columns that you create first. (2) Rename every column to upper case in one call. (3) Give everyone under 30 a 5-point score bonus in a single statement.

### 2.07 Sorting, ranking & duplicates

```python
df = make_df()
df.sort_values("score"); df.sort_values("score", ascending=False, na_position="first")
df.sort_values(["team", "score"], ascending=[True, False])
df.sort_values("name", key=lambda s: s.str.lower())   # sort by a transformed key
df.sort_index(); df.sort_index(axis=1)           # by row labels / by column names
df["score"].rank(); df["score"].rank(method="dense", ascending=False); df["score"].rank(pct=True)
df.groupby("team")["score"].rank(ascending=False)    # rank within each group
df.drop_duplicates(); df.drop_duplicates(subset=["score"], keep="last"); df.drop_duplicates(subset=["team"], keep=False)
df[df.duplicated(subset=["joined"], keep=False)]     # show ALL rows involved in a duplicate
df.sort_values("joined").drop_duplicates("team", keep="last")   # latest row per team
```

**Drills:** (1) Sort by team A→Z, then score high→low. (2) Keep only each team's first-joined member. (3) Add a dense rank of score (1 = best).

### 2.08 Missing data

```python
df = make_df()
df.isna(); df.notna(); df["score"].hasnans
df.isna().sum(); df[df.isna().any(axis=1)]        # rows containing any missing value
df.dropna(); df.dropna(subset=["score"]); df.dropna(how="all"); df.dropna(axis=1); df.dropna(thresh=5)
df.fillna({"score": df["score"].median()})        # fill value per column
df["score"].fillna(0); df["score"].fillna(df["score"].mean())
df["score"].ffill(); df["score"].bfill(); df["score"].ffill(limit=1)   # carry the previous/next value
df["score"].interpolate()
df.set_index("joined").sort_index()["score"].interpolate(method="time")   # time-aware interpolation
df.groupby("team")["score"].transform(lambda s: s.fillna(s.mean()))   # fill with each group's mean
df.replace({"team": {"green": np.nan}}); df.replace(-999, np.nan); df.replace(r"^\s*$", np.nan, regex=True)
df["score"].combine_first(pd.Series([0.0] * 6))  # take values from another Series where missing
pd.isna(np.nan), pd.isna(None), pd.isna(pd.NA), pd.isna(pd.NaT)   # all True
pd.Series([1, None, 3], dtype="Int64").sum()     # 4: aggregations skip NA by default
df["score"].sum(skipna=False)                    # nan
```

**Drills:** (1) Drop rows where `score` is missing, then count how many were dropped. (2) Fill missing scores with each team's median. (3) Treat the string `"?"` as missing while reading a CSV.

### 2.09 Applying functions

```python
df = make_df()
# 1) vectorized operations first (fastest)
df["score"] * 2; np.log1p(df["age"]); df["score"].clip(lower=70); df["score"].round(0)
# 2) map: element-wise on a Series
df["team"].map({"red": "R", "blue": "B", "green": "G"})   # values missing from the dict -> NaN
df["team"].map(str.upper); df["name"].map(len)
df["team"].replace({"red": "R"})                 # like map, but leaves unmatched values unchanged
# 3) apply: any Python function; axis=1 runs per row (slow)
df["age"].apply(lambda a: "30+" if a >= 30 else "<30")
df[["score", "age"]].apply(lambda col: col.max() - col.min())        # per column
df.apply(lambda row: f"{row['name']} ({row['team']})", axis=1)      # per row (prefer vectorized string ops)
df[["score", "age"]].map(lambda v: f"{v:.1f}")                      # every cell (formerly applymap)
df[["score", "age"]].agg(["mean", "max"]); df.agg({"score": "mean", "age": ["min", "max"]})
df[["score", "age"]].transform(lambda c: c - c.mean())               # same shape as the input
# multi-branch conditions
df["tier"] = np.select([df["score"] >= 90, df["score"] >= 75], ["gold", "silver"], default="bronze")
df["tier2"] = pd.Series("bronze", index=df.index).case_when([(df["score"] >= 90, "gold"), (df["score"] >= 75, "silver")])   # 2.2+
# pipe: your own functions inside method chains
def add_ratio(d, num, den): return d.assign(ratio=d[num] / d[den])
df.pipe(add_ratio, "score", "age").head(2)
df.eval("ratio = score / age")                   # expression strings (column names as variables)
(df.query("age > 25").assign(score=lambda d: d["score"].fillna(0)).sort_values("score", ascending=False).head(3))
```

**Drills:** (1) Map team names to integer ids with a dict. (2) Build a `label` column like "Ana (red)" without `apply`. (3) Write a `clean(df)` function and call it with `.pipe`.

### 2.10 String methods (.str)

```python
s = pd.Series(["  Alice Smith ", "bob jones", "CAROL  king", None, "dave-o'neil"])
s.str.strip(); s.str.lower(); s.str.upper(); s.str.title(); s.str.capitalize(); s.str.swapcase(); s.str.len()
s.str.strip().str.split()                        # lists of tokens
s.str.strip().str.split(" ", n=1, expand=True)   # expand=True -> DataFrame columns
s.str.strip().str.split().str[0]                 # .str[] indexes into each list/string
s.str.strip().str.split().str.get(-1); s.str.rsplit(" ", n=1)
s.str.contains("jo"); s.str.contains("JO", case=False, na=False); s.str.contains(r"\s{2,}", regex=True)
s.str.startswith("b"); s.str.endswith("ll"); s.str.match(r"\s*[A-Z]"); s.str.fullmatch(r"[a-z ]+")
s.str.replace("-", " "); s.str.replace(r"\s+", " ", regex=True)    # regex=False is the default
s.str.strip().str.extract(r"(?P<first>\w+)\s+(?P<last>\w+)")       # named groups -> columns
pd.Series(["a1b22", "c333"]).str.extractall(r"(\d+)"); pd.Series(["a1b22"]).str.findall(r"\d+")
s.str.count("o"); s.str.find("o"); s.str.slice(0, 3); s.str[:3]; s.str[-2:]
s.str.pad(15, side="left", fillchar="."); pd.Series(["7", "42"]).str.zfill(4); s.str.center(15, "*")
s.str.cat(sep=" | ", na_rep="?")                 # join all values into one string
s.str.strip().str.cat(pd.Series(["!"] * 5))      # element-wise concatenation
s.str.removeprefix("  "); s.str.removesuffix(" "); s.str.repeat(2); s.str.wrap(5)
s.str.isalpha(); s.str.isdigit(); s.str.isnumeric(); s.str.isspace(); s.str.islower()
pd.Series(["a|b", "b|c"]).str.get_dummies(sep="|")   # multi-label one-hot
pd.Series(["café", "naïve"]).str.normalize("NFKD").str.encode("ascii", errors="ignore").str.decode("ascii")   # strip accents
s.str.partition(" "); pd.Series([["a", "b"]]).str.join("-")
```

**Drills:** (1) Clean names: strip, collapse repeated spaces, title-case. (2) Extract the numbers from `"order-123"`-style ids into an int column. (3) One-hot encode a `"tag1|tag2"` column.

### 2.11 Dates, times & time series

```python
ts = pd.to_datetime(["2024-01-15 08:30", "2024-02-29 17:05", "2024-03-03 23:59"])
pd.to_datetime("15/01/2024", format="%d/%m/%Y"); pd.to_datetime(["2024-01-01", "oops"], errors="coerce")   # bad -> NaT
pd.to_datetime(1_700_000_000, unit="s"); pd.to_datetime(pd.DataFrame({"year": [2024], "month": [5], "day": [1]}))
s = pd.Series(ts)
s.dt.year; s.dt.month; s.dt.day; s.dt.hour; s.dt.minute; s.dt.dayofweek; s.dt.day_name(); s.dt.month_name()
s.dt.quarter; s.dt.dayofyear; s.dt.is_month_end; s.dt.days_in_month; s.dt.isocalendar().week
s.dt.date; s.dt.time; s.dt.normalize(); s.dt.floor("h"); s.dt.ceil("D"); s.dt.round("15min"); s.dt.to_period("M")
s.dt.strftime("%Y-%m-%d %H:%M"); s.dt.tz_localize("UTC").dt.tz_convert("America/New_York")
s.diff(); (s - s.min()).dt.days; s.diff().dt.total_seconds() / 3600
pd.Timestamp("2024-01-31") + pd.Timedelta(days=1, hours=3)
pd.Timestamp("2024-01-31") + pd.DateOffset(months=1)   # 2024-02-29: calendar-aware
pd.Timestamp("2024-01-15") + pd.offsets.MonthEnd(0); pd.Timestamp.now(); pd.Timestamp.today().normalize()
pd.to_timedelta(["1 days", "2h", "30min"]); pd.Timedelta("1h30min").total_seconds()
# frequencies: D day · h hour · min minute · s second · B business day · W-MON weekly · ME month end · MS month start · QE quarter end · YE year end
pd.date_range("2024-01-01", periods=5, freq="D"); pd.date_range("2024-01-01", "2024-06-30", freq="ME"); pd.bdate_range("2024-01-01", periods=5)
pd.period_range("2024-01", periods=3, freq="M"); pd.Period("2024-03", freq="M").end_time
rng = np.random.default_rng(0)
daily = pd.DataFrame({"sales": rng.integers(50, 150, 90)}, index=pd.date_range("2024-01-01", periods=90, freq="D"))
daily.loc["2024-02"]; daily.loc["2024-01-10":"2024-01-15"]       # partial-string date slicing
daily.resample("W").sum(); daily.resample("ME").agg(["sum", "mean", "max"]); daily.resample("MS").first()
daily.resample("h").ffill().head()               # upsample: daily -> hourly
daily["sales"].shift(1); daily["sales"].shift(-1); daily["sales"].shift(1, freq="D")   # lag / lead / shift the index
daily["sales"].diff(); daily["sales"].pct_change(); daily["sales"].diff(7)              # change vs yesterday / last week
daily.asfreq("W-MON"); daily.groupby(daily.index.month)["sales"].mean(); daily.tz_localize("UTC")
make_df().groupby(pd.Grouper(key="joined", freq="YE"))["score"].mean()   # time bins on a column
```

**Drills:** (1) Parse `"03/15/2024 14:00"` strings and extract the weekday name. (2) Resample daily sales to weekly totals and month-end means. (3) Add 7-day lag and week-over-week change columns.

### 2.12 Categoricals & binning

```python
df = make_df()
df["team"] = df["team"].astype("category")
df["team"].cat.categories; df["team"].cat.codes  # integer codes: handy as model inputs
df["team"].cat.rename_categories({"red": "R"}); df["team"].cat.add_categories(["yellow"]); df["team"].cat.remove_unused_categories()
df["team"].cat.set_categories(["red", "blue", "green", "black"]); df["team"].cat.reorder_categories(["green", "blue", "red"])
size = pd.Categorical(["M", "S", "L", "M"], categories=["S", "M", "L"], ordered=True)
size.min(), size.max(); pd.Series(size) > "S"; pd.Series(size).sort_values()   # ordered comparisons & sorting
pd.CategoricalDtype(["low", "high"], ordered=True)
df.groupby("team", observed=True)["score"].mean()   # observed=True skips empty categories
pd.cut(df["age"], bins=3)                           # 3 equal-WIDTH bins
pd.cut(df["age"], bins=[0, 30, 40, 120], labels=["young", "mid", "older"], right=False)
pd.qcut(df["age"], q=4); pd.qcut(df["score"], q=[0, 0.5, 1], labels=["low", "high"]); pd.qcut(df["age"], 3, labels=False)   # equal-COUNT bins
codes, uniques = pd.factorize(make_df()["team"])    # label-encode any column
pd.get_dummies(df["team"], prefix="team", dtype=int); pd.get_dummies(df, columns=["team"], drop_first=True)
make_df()["team"].memory_usage(deep=True), df["team"].memory_usage(deep=True)   # categories shrink repeated strings
```

**Drills:** (1) Make an ordered `size` column (S < M < L) and filter sizes ≥ M. (2) Bin ages into quartiles labeled Q1–Q4. (3) One-hot encode `team`, dropping the first level.

### 2.13 GroupBy (split-apply-combine)

```python
df = make_df()
g = df.groupby("team")
g["score"].mean(); g["score"].agg(["mean", "max", "count"])
g.size(); g["score"].count()                     # size counts rows (including NaN); count skips NaN
g[["score", "age"]].mean(); df.groupby(["team", "age"])["score"].sum()
df.groupby("team", as_index=False)["score"].mean()   # keep team as a regular column
df.groupby("team").agg(n=("name", "size"), avg_score=("score", "mean"), oldest=("age", "max"))   # named aggregation
df.groupby("team").agg({"score": ["mean", "std"], "age": "median"})
df.groupby("team")["score"].agg(lambda s: s.max() - s.min())
df.groupby("team", dropna=False, sort=False)["age"].sum()
# transform: one result per ORIGINAL row (same length), aligned back automatically
df["team_avg"] = df.groupby("team")["score"].transform("mean")
df["score_vs_team"] = df["score"] - df["team_avg"]
df["rank_in_team"] = df.groupby("team")["score"].rank(ascending=False, method="first")
df["share_of_team_age"] = df["age"] / df.groupby("team")["age"].transform("sum")
df.groupby("team").filter(lambda d: len(d) >= 2)            # keep or drop WHOLE groups
df.groupby("team")[["score", "age"]].apply(lambda d: d.nlargest(1, "age"))   # arbitrary function per group (slowest)
g = df.groupby("team")
g.head(1); g.nth(0); g.first(); g.last(); g.cumcount(); g.ngroup()
g["score"].cumsum(); g["score"].shift(1); g["score"].diff()
df.loc[g["score"].idxmax()]                      # the best-scoring row of each team
g.get_group("red"); g.groups; g.ngroups
for team, frame in g: print(team, frame.shape)
df.groupby(df["joined"].dt.year)["score"].mean()             # group by a derived Series
df.groupby(pd.cut(df["age"], [0, 30, 100]), observed=True)["score"].mean()
df.groupby("team")["score"].describe(); df.groupby("team")["name"].agg(", ".join)
df.groupby("team")["score"].rolling(2, min_periods=1).mean()   # rolling window within each group
df.groupby("team").sample(n=1, random_state=0)               # stratified sample
```

**Watch out:** `transform` keeps the original row count, so the result can be assigned back as a column. `agg` returns one row per group.

**Drills:** (1) Per team: member count, mean score and youngest age, with readable column names. (2) Add each person's z-score within their team. (3) Keep only teams whose mean score is above 80.

### 2.14 Reshaping

```python
long = pd.DataFrame({"student": ["a", "a", "b", "b"], "subject": ["math", "art", "math", "art"], "grade": [90, 75, 60, 85]})
wide = long.pivot(index="student", columns="subject", values="grade")   # long -> wide (pairs must be unique)
wide.reset_index().melt(id_vars="student", var_name="subject", value_name="grade")   # wide -> long
df = make_df(); df["over30"] = df["age"] > 30
df.pivot_table(index="team", values="score", aggfunc="mean")
df.pivot_table(index="team", columns="over30", values="score", aggfunc="mean", fill_value=0, margins=True)
pd.crosstab(df["team"], df["over30"]); pd.crosstab(df["team"], df["over30"], normalize="index")
stacked = wide.stack()                           # columns -> innermost row level (a Series)
stacked.unstack(); stacked.unstack(level=0); wide.T
tags = pd.DataFrame({"id": [1, 2], "tags": [["ml", "nlp"], ["cv"]]})
tags.explode("tags", ignore_index=True)          # one row per list element
pd.get_dummies(df[["team"]], dtype=int); pd.from_dummies(pd.get_dummies(df["team"]))
pd.wide_to_long(pd.DataFrame({"id": [1, 2], "score2023": [5, 6], "score2024": [7, 8]}), stubnames="score", i="id", j="year")
df.set_index(["team", "name"])["score"].unstack()   # quick wide table
long.pivot_table(index="student", columns="subject", values="grade").rename_axis(columns=None).reset_index()   # flat & tidy
```

**Drills:** (1) Convert a wide table with `q1..q4` columns to long format. (2) Make a team × over30 table of mean scores. (3) Explode a list-of-tags column and count each tag.

### 2.15 Combining frames

```python
people = make_df()[["name", "team"]]
teams = pd.DataFrame({"team": ["red", "blue", "gold"], "coach": ["Kim", "Lee", "Max"]})
pd.merge(people, teams, on="team")               # inner join (default): matching keys only
people.merge(teams, on="team", how="left")       # keep every person; coach is NaN for "green"
people.merge(teams, on="team", how="right"); people.merge(teams, on="team", how="outer", indicator=True)   # _merge column
people.merge(teams.rename(columns={"team": "squad"}), left_on="team", right_on="squad")
people.merge(people, on="team", suffixes=("_a", "_b"))   # self-join: all pairs within a team
people.merge(teams, on="team", how="left", validate="many_to_one")   # raises if teams has duplicate keys
people.merge(teams, how="cross").shape           # cartesian product (18, 4)
m = people.merge(teams, on="team", how="left", indicator=True); m[m["_merge"] == "left_only"]   # anti-join
people[~people["team"].isin(teams["team"])]      # anti-join, simpler
stats, info = make_df().set_index("name")[["score"]], make_df().set_index("name")[["age"]]
stats.join(info); stats.join(info, how="inner")  # join on the index
pd.concat([people.head(2), people.tail(2)]); pd.concat([people.head(2), people.tail(2)], ignore_index=True)   # stack rows
pd.concat([people, make_df()[["score"]]], axis=1)   # side by side, aligned on the index
pd.concat({"train": people.head(3), "test": people.tail(3)}, names=["split"])   # dict keys -> outer index level
pd.concat([people, teams], join="inner")         # only the shared columns
trades = pd.DataFrame({"t": pd.to_datetime(["2024-01-01 10:00:03", "2024-01-01 10:00:07"]), "qty": [5, 3]})
quotes = pd.DataFrame({"t": pd.to_datetime(["2024-01-01 10:00:00", "2024-01-01 10:00:05"]), "price": [100.0, 101.0]})
pd.merge_asof(trades, quotes, on="t", direction="backward")   # nearest earlier key (both sorted)
pd.Series([1, np.nan, 3]).combine_first(pd.Series([9, 2, 9]))   # [1. 2. 3.]
a = make_df(); b = a.copy(); b.loc[0, "score"] = 0; a.compare(b)   # cell-level diff
a.update(b)                                      # overwrite a in place with b's non-NA values
```

**Watch out:** after a merge, check `len(result)`. Duplicate keys on both sides multiply rows. `validate=` catches this before it happens.

**Drills:** (1) Left-join coaches onto people and list people without a coach. (2) Stack two monthly files and tag each row with its source month. (3) Attach to each trade the last quote before it.

### 2.16 Window functions

```python
rng = np.random.default_rng(1)
s = pd.Series(rng.normal(size=60).cumsum(), index=pd.date_range("2024-01-01", periods=60, freq="D"))
s.rolling(7).mean()                              # first 6 values are NaN
s.rolling(7, min_periods=1).mean(); s.rolling(7, center=True).median()
s.rolling("7D").sum()                            # time-based window (needs a DatetimeIndex)
s.rolling(7).agg(["mean", "std", "min", "max"]); s.rolling(7).quantile(0.9)
s.rolling(7).apply(lambda w: w[-1] - w[0], raw=True)   # custom function (raw=True passes NumPy arrays)
s.rolling(5, win_type="gaussian").mean(std=2)    # weighted window (needs SciPy)
s.expanding().max(); s.expanding(min_periods=3).mean(); s.cummax()   # everything up to each point
s.ewm(span=10).mean(); s.ewm(alpha=0.3, adjust=False).mean(); s.ewm(halflife="3D", times=s.index).mean()
s.rolling(14).corr(s.shift(1))                   # rolling correlation
pd.DataFrame({"a": s, "b": s * 2}).rolling(3, step=2).mean()
# leakage-safe ML features: lag and rolling stats per entity, computed from PAST rows only
feat = pd.DataFrame({"user": ["u1"] * 5 + ["u2"] * 5, "value": range(10)})
feat["lag1"] = feat.groupby("user")["value"].shift(1)
feat["roll3_mean"] = feat.groupby("user")["value"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
```

**Drills:** (1) Compute a 30-day rolling mean and std of a price series. (2) Add an EMA with span 12 and with span 26, and their difference. (3) Build per-user lag-1 and lag-7 features without leaking the current row.

### 2.17 MultiIndex

```python
df = make_df()
mi = df.set_index(["team", "name"]).sort_index()
mi.index.names; mi.index.levels; mi.index.get_level_values("team")
mi.loc["red"]; mi.loc[("red", "Ana")]; mi.loc[("red", "Ana"), "score"]
mi.xs("Ana", level="name"); mi.loc[pd.IndexSlice[:, ["Ana", "Ben"]], :]
mi.swaplevel().sort_index(); mi.droplevel("name"); mi.reset_index(level="name")
mi.groupby(level="team")["score"].mean(); mi["score"].unstack(level="team")
pd.MultiIndex.from_tuples([("a", 1), ("a", 2)], names=["letter", "num"])
pd.MultiIndex.from_product([["x", "y"], [1, 2]]); pd.MultiIndex.from_arrays([["a", "b"], [1, 2]])
agg = df.groupby("team").agg({"score": ["mean", "max"], "age": ["min"]})   # MultiIndex COLUMNS
agg.columns.tolist()                             # [('score', 'mean'), ('score', 'max'), ('age', 'min')]
agg.columns = ["_".join(c) for c in agg.columns] # flatten -> score_mean, score_max, age_min
mi.rename_axis(index={"team": "squad"})
```

**Drills:** (1) Index by (team, name) and pull every player on "blue". (2) Flatten the columns of a multi-function `agg` result. (3) Turn the `team` level into columns with `unstack`.

### 2.18 Plotting & styling

```python
import matplotlib.pyplot as plt
df = make_df()
ax = df.plot(x="name", y="score", kind="bar", title="Scores", legend=False)   # returns a Matplotlib Axes
df.plot.scatter(x="age", y="score", c="age", colormap="viridis")
df["score"].plot.hist(bins=5, alpha=0.7); df["score"].plot.kde(); df[["score", "age"]].plot.box()
df.groupby("team")["score"].mean().sort_values().plot.barh()
ts = pd.DataFrame(np.random.default_rng(0).normal(size=(30, 2)).cumsum(axis=0), columns=["a", "b"], index=pd.date_range("2024-01-01", periods=30))
ts.plot(figsize=(8, 3), grid=True, style=["-", "--"]); ts.plot(subplots=True, layout=(2, 1), sharex=True)
ts.plot.area(stacked=False, alpha=0.4); ts.plot(secondary_y="b")
df["team"].value_counts().plot.pie(autopct="%1.0f%%")
df.hist(column=["score", "age"], bins=5, figsize=(8, 3)); df.boxplot(column="score", by="team")
pd.plotting.scatter_matrix(df[["score", "age"]], diagonal="kde", figsize=(5, 5))
ax = df.plot.scatter(x="age", y="score"); ax.set_title("Age vs score"); ax.figure.savefig("scatter.png", dpi=150)
df.style.format({"score": "{:.1f}"}).background_gradient(subset=["score"], cmap="Greens").highlight_null()
df.style.highlight_max(subset=["score", "age"]).bar(subset=["age"], color="#9cf").hide(axis="index").to_html()
plt.close("all")
```

**Drills:** (1) Plot mean score per team as a sorted horizontal bar chart. (2) Plot two time series on separate y-axes. (3) Render a table with a color gradient on `score` for a notebook report.

### 2.19 Performance & handing data to models

```python
df = make_df()
X = df[["score", "age"]].to_numpy(dtype=np.float32)   # model-ready float32 matrix (NaN stays NaN)
y = df["team"].astype("category").cat.codes.to_numpy()  # integer class labels
df["age"].to_list(); df.to_numpy()                  # .values is the legacy spelling
# PyTorch: torch.from_numpy(X) or torch.tensor(df[cols].to_numpy(dtype="float32"))
big = pd.DataFrame({"a": np.arange(100_000), "b": np.arange(100_000) * 2.0})
big["c"] = big["a"] * big["b"]                      # vectorized: fastest
big["d"] = [a + b for a, b in zip(big["a"], big["b"])]   # list comprehension: fine for complex Python logic
for row in big.head(3).itertuples(index=False): print(row.a, row.b)   # quick row iteration (namedtuples)
for i, row in big.head(3).iterrows(): print(row["a"])  # slowest: builds a Series per row
big["a"] = pd.to_numeric(big["a"], downcast="integer"); big["b"] = pd.to_numeric(big["b"], downcast="float")
big.memory_usage(deep=True).sum() / 1e6             # MB
pd.read_csv("people.csv", dtype={"team": "category"}, usecols=["team", "score"])   # shrink at load time
summary = (make_df()                                # method chain: readable, no temporary variables
    .dropna(subset=["score"])
    .assign(age_group=lambda d: pd.cut(d["age"], [0, 30, 100], labels=["<30", "30+"]))
    .groupby(["team", "age_group"], observed=True)
    .agg(n=("name", "size"), mean_score=("score", "mean"))
    .reset_index()
    .sort_values("mean_score", ascending=False))
pd.testing.assert_frame_equal(summary, summary.copy()); pd.testing.assert_series_equal(df["age"], df["age"])
pd.api.types.is_numeric_dtype(df["age"]), pd.api.types.is_string_dtype(df["name"]), pd.api.types.is_datetime64_any_dtype(df["joined"])
val_names = set(df["name"].unique()[:2]); is_val = df["name"].isin(val_names)   # split by GROUP so no entity is in both sets
train_df, val_df = df[~is_val], df[is_val]
```

**Drills:** (1) Time a row-wise `apply` against the vectorized version on 1M rows. (2) Cut a DataFrame's memory in half with downcasting and categoricals. (3) Rewrite a 6-step cleaning script as a single method chain.

Full reference: https://pandas.pydata.org/docs/reference/

## 3. PyTorch

`import torch` · written against PyTorch 2.x. Modules 3.01–3.19 follow the curriculum order `01_tensors` → `19_dpo_loss`. The first five build everyday fluency. Modules 06–12 cover mechanisms you implement yourself, and 13–19 cover performance and scale. Snippets run on CPU unless marked GPU.

### 3.01 Tensors

#### Creating tensors

```python
import math
import numpy as np
import torch
torch.tensor([1, 2, 3])                          # copies data; infers dtype (int64)
torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float32, device="cpu", requires_grad=True)
arr = np.arange(6.0)
torch.as_tensor(arr)                             # shares memory with NumPy when possible
torch.from_numpy(arr)                            # always shares memory (CPU): editing one edits the other
torch.asarray(arr)                               # array-API style; avoids copies when it can
torch.zeros(2, 3); torch.zeros((2, 3))           # size as varargs or as a tuple
torch.ones(2, 3); torch.full((2, 3), 7.0); torch.empty(2, 3)   # empty = uninitialized memory
torch.eye(3); torch.eye(2, 3)
x = torch.randn(2, 3)
torch.zeros_like(x); torch.ones_like(x); torch.full_like(x, 0.5); torch.empty_like(x)
torch.rand_like(x); torch.randn_like(x); torch.randint_like(x, 10)
x.new_zeros(2, 2); x.new_ones(3); x.new_full((2,), 9.0); x.new_tensor([1, 2]); x.new_empty(4)   # same dtype & device as x
torch.arange(5); torch.arange(1, 10, 2); torch.arange(0, 1, 0.25)   # end exclusive
torch.linspace(0, 1, steps=5); torch.logspace(0, 3, steps=4)        # end inclusive; logspace -> 1, 10, 100, 1000
xx, yy = torch.meshgrid(torch.arange(3), torch.arange(2), indexing="ij")
torch.cartesian_prod(torch.tensor([1, 2]), torch.tensor([3, 4])); torch.combinations(torch.arange(4), r=2)
M = torch.arange(9.).reshape(3, 3); v = torch.tensor([1., 2., 3.])
torch.diag(v); torch.diag(M); torch.diag_embed(v); torch.diagflat(v)
torch.tril(M); torch.triu(M, diagonal=1); torch.tril_indices(3, 3); torch.triu_indices(3, 3, offset=1)
torch.block_diag(torch.ones(2, 2), torch.ones(1, 1)); torch.vander(v, N=3); torch.kron(torch.eye(2), M)
torch.complex(torch.tensor(1.), torch.tensor(2.)); torch.polar(torch.tensor(1.), torch.tensor(math.pi / 2))
```

#### Random sampling & reproducibility

```python
torch.manual_seed(0)                             # seeds CPU and all GPUs
torch.rand(2, 3)                                 # uniform [0, 1)
torch.randn(2, 3)                                # standard normal
torch.randint(0, 10, (2, 3))                     # integers in [0, 10)
torch.randperm(5)                                # random permutation of 0..4
torch.normal(mean=0.0, std=2.0, size=(3,)); torch.normal(torch.zeros(3), torch.ones(3))
torch.bernoulli(torch.full((5,), 0.3))           # 0/1 draws (dropout-style masks)
torch.multinomial(torch.tensor([0.1, 0.6, 0.3]), num_samples=4, replacement=True)   # sample class ids (token sampling)
torch.poisson(torch.full((3,), 4.0))
torch.empty(3).uniform_(-1, 1); torch.empty(3).normal_(0, 1); torch.empty(3).exponential_(); torch.empty(3).random_(0, 10)
g = torch.Generator().manual_seed(42); torch.rand(2, generator=g)   # an independent random stream
torch.use_deterministic_algorithms(True); torch.use_deterministic_algorithms(False)   # error on non-deterministic kernels
torch.backends.cudnn.benchmark = False            # True = autotune conv algorithms (faster, less reproducible)
```

#### dtypes, devices & attributes

```python
# float32 (float) · float64 (double) · float16 (half) · bfloat16 · int64 (long) · int32 (int) · int16 · int8 · uint8 · bool · complex64 · float8_e4m3fn
x = torch.randn(2, 3, 4)
x.shape, x.size(), x.size(1), x.ndim, x.dim(), x.numel(), len(x)   # (2,3,4) twice, 3, 3, 3, 24, 2
x.dtype, x.device, x.layout, x.requires_grad, x.is_leaf, x.grad
x.element_size(), x.nbytes, x.stride(), x.is_contiguous()
x.mT.shape                                       # (2, 4, 3): transpose of the last two dims (batched)
x.to(torch.float16); x.float(); x.double(); x.half(); x.bfloat16(); x.long(); x.int(); x.bool()
torch.tensor([1, 2]).dtype, torch.tensor([1., 2]).dtype   # int64, float32
torch.tensor([1, 2]) / 2                         # true division -> float tensor([0.5, 1.0])
torch.tensor([5, 7]) // 2                        # floor division -> tensor([2, 3])
torch.get_default_dtype(); torch.promote_types(torch.int32, torch.float16); torch.result_type(torch.tensor(1), 1.0)
torch.can_cast(torch.float32, torch.int32)       # False
torch.finfo(torch.float16).max, torch.finfo(torch.bfloat16).eps, torch.iinfo(torch.int8).min
x.is_floating_point(); torch.is_tensor(x)
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
torch.cuda.is_available(); torch.cuda.device_count()     # GPU checks (mps = Apple Silicon)
y = torch.zeros(3, device=device); y.to("cpu"); y.cpu(); y.device
# GPU only: x.cuda() · x.to("cuda", non_blocking=True) · x.pin_memory() · torch.cuda.get_device_name(0)
with torch.device(device): z = torch.ones(3)     # default device for tensors created inside the block
```

#### Converting

```python
t = torch.tensor([[1.5, 2.5], [3.5, 4.5]], requires_grad=True)
t[0, 0].item()                                   # one-element tensor -> Python number
t.tolist()                                       # nested Python lists
t.detach().cpu().numpy()                         # the safe universal recipe -> NumPy
t.numpy(force=True)                              # same: detaches / moves / copies as needed
t.clone(); t.detach(); torch.empty(2, 2).copy_(t) # copy / cut from graph / copy into existing storage
torch.save(t.detach(), "t.pt"); torch.load("t.pt", weights_only=True)
```

#### Indexing, gather & scatter

```python
x = torch.arange(24).reshape(2, 3, 4)
x[0], x[0, 1], x[0, 1, 2], x[:, 1], x[..., -1], x[:, :, ::2], x[:, None].shape   # None adds an axis
x[x > 20]; x[x % 2 == 0]                         # boolean masks -> 1-D result
y = x.clone(); y[y > 10] = 0                     # masked assignment
idx = torch.tensor([0, 2]); x[:, idx]; x[[0, 1], [1, 2]]   # fancy indexing (copies)
torch.index_select(x, dim=1, index=idx); torch.take(x, torch.tensor([0, 5, 23])); torch.narrow(x, dim=2, start=1, length=2)
torch.masked_select(x, x > 20); x.masked_fill(x > 20, -1)
torch.where(x > 10, x, torch.zeros_like(x)); torch.where(x > 20)   # elementwise select / tuple of indices
torch.nonzero(x > 20); torch.argwhere(x > 20); (x > 20).nonzero(as_tuple=True)
logits = torch.randn(4, 10); targets = torch.tensor([3, 0, 9, 1])
logits.gather(1, targets.unsqueeze(1)).squeeze(1)    # out[i] = logits[i, targets[i]]
logits[torch.arange(4), targets]                     # same thing with fancy indexing
torch.zeros(4, 10).scatter_(1, targets.unsqueeze(1), 1.0)   # one-hot by scattering (== F.one_hot)
torch.zeros(5).scatter_add_(0, torch.tensor([0, 1, 1, 4]), torch.ones(4))   # bincount-style accumulation
torch.zeros(3).scatter_reduce(0, torch.tensor([0, 0, 2]), torch.tensor([5., 7., 1.]), reduce="amax", include_self=False)
z = torch.zeros(3, 4); z.index_add_(0, torch.tensor([0, 2]), torch.ones(2, 4)); z.index_fill_(1, torch.tensor([3]), -1.0)
z.index_put_((torch.tensor([0, 0]), torch.tensor([1, 1])), torch.tensor([1., 1.]), accumulate=True)
torch.take_along_dim(logits, logits.argsort(dim=1), dim=1)   # gather with an index tensor of the same rank
```

#### Joining, splitting & rearranging

```python
a, b = torch.ones(2, 3), torch.zeros(2, 3)
torch.cat([a, b], dim=0).shape, torch.cat([a, b], dim=1).shape   # (4, 3), (2, 6): EXISTING dim
torch.stack([a, b], dim=0).shape                 # (2, 2, 3): NEW dim
torch.hstack([a, b]); torch.vstack([a, b]); torch.dstack([a, b]); torch.column_stack([a[0], b[0]])
x = torch.arange(10)
torch.split(x, 3); torch.split(x, [2, 8]); torch.chunk(x, 3); torch.tensor_split(x, 3); torch.unbind(torch.eye(3), dim=0)
torch.flip(x, dims=[0]); torch.roll(x, shifts=2); torch.rot90(torch.eye(3), 1, dims=(0, 1)); torch.fliplr(torch.eye(3))
torch.tensor([[1, 2]]).repeat(2, 3)              # tile data (copies) -> (2, 6)
torch.repeat_interleave(torch.tensor([1, 2]), 3) # tensor([1, 1, 1, 2, 2, 2])
torch.tile(torch.tensor([1, 2]), (2,))
```

#### Math, reductions & comparisons

```python
a, b = torch.tensor([1., 4., 9.]), torch.tensor([2., 2., 2.])
a + b, a - b, a * b, a / b, a // b, a % b, a ** 2, -a, a @ b   # @ on 1-D = dot product
torch.add(a, b, alpha=2)                         # a + 2*b
torch.div(a, b, rounding_mode="floor"); torch.remainder(a, b); torch.fmod(a, b); torch.pow(a, 0.5)
torch.abs(a); torch.sign(a); torch.sqrt(a); torch.rsqrt(a); torch.square(a); torch.exp(a); torch.log(a); torch.log1p(a); torch.expm1(a)
torch.sin(a); torch.cos(a); torch.tanh(a); torch.sigmoid(a); torch.logit(torch.tensor([0.2])); torch.erf(a); torch.atan2(a, b)
torch.floor(a); torch.ceil(a); torch.round(a / 3, decimals=2); torch.trunc(a); torch.frac(a)
torch.clamp(a, min=2, max=5); torch.clip(a, 2, 5); torch.maximum(a, b); torch.minimum(a, b)
torch.lerp(a, b, 0.3); torch.hypot(a, b); torch.logaddexp(a, b); torch.xlogy(a, b); torch.nan_to_num(torch.tensor([float("nan")]), nan=0.0)
torch.addcmul(a, a, b, value=0.5); torch.addcdiv(a, a, b, value=0.5)   # fused a + v*x*y and a + v*x/y (used in optimizers)
c = a.clone(); c.add_(1); c.mul_(2); c.clamp_(0, 10); c.zero_(); c.fill_(3)   # trailing underscore = in-place
torch.add(a, b, out=torch.empty(3))
M = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
M.sum(), M.sum(dim=0), M.sum(dim=1, keepdim=True), M.sum(dim=(0, 1))
M.mean(dim=0); M.std(); M.std(dim=0, correction=0); M.var(); torch.std_mean(M, dim=0); torch.var_mean(M)   # std is unbiased (correction=1) by default
M.max(); M.max(dim=1)                            # with dim: namedtuple (values, indices)
values, indices = M.max(dim=1); M.amax(dim=1); M.amin(); torch.aminmax(M)
M.argmax(); M.argmax(dim=1); M.argmin(dim=0, keepdim=True)
M.median(); M.median(dim=1); M.mode(dim=1); M.quantile(0.5); torch.quantile(M, torch.tensor([0.25, 0.75]), dim=1)
M.prod(); M.nansum(); M.nanmean(); torch.logsumexp(M, dim=1)   # logsumexp = stable log(sum(exp(x)))
torch.cumsum(M, dim=1); torch.cumprod(M, dim=1); torch.cummax(M, dim=1); torch.logcumsumexp(M, dim=1)
M.all(); (M > 2).any(dim=0); torch.count_nonzero(M > 2, dim=0)
torch.linalg.vector_norm(M, ord=2, dim=1); torch.linalg.matrix_norm(M); M.norm(dim=1)
torch.diff(M, dim=1); torch.trapezoid(M, dim=1)
torch.sort(M, dim=1, descending=True); torch.argsort(M, dim=1); torch.topk(M, k=2, dim=1); torch.kthvalue(M, k=2, dim=1)
torch.unique(torch.tensor([3, 1, 3, 2]), return_inverse=True, return_counts=True); torch.unique_consecutive(torch.tensor([1, 1, 2, 2, 1]))
torch.bincount(torch.tensor([0, 1, 1, 3]), minlength=5); torch.histc(M, bins=4, min=0, max=8); torch.histogram(M.flatten(), bins=4)
torch.searchsorted(torch.tensor([1, 3, 5, 7]), torch.tensor([4, 7])); torch.bucketize(torch.tensor([0.1, 0.6]), torch.tensor([0.25, 0.5, 0.75]))
torch.isin(torch.tensor([1, 2, 5]), torch.tensor([2, 5]))
a == b; torch.eq(a, b); torch.gt(a, b); torch.equal(a, a)   # torch.equal: same shape AND values -> bool
torch.allclose(a, a + 1e-9); torch.isclose(a, b); torch.testing.assert_close(a, a.clone())   # assert_close gives detailed errors
torch.isnan(a); torch.isinf(a); torch.isfinite(a); torch.logical_and(a > 1, b > 1); (a > 1) & (b > 1); ~(a > 1)
```

#### Linear algebra

```python
A, B = torch.randn(3, 4), torch.randn(4, 5)
A @ B; torch.matmul(A, B); torch.mm(A, B)        # mm: 2-D only, no broadcasting
torch.bmm(torch.randn(10, 3, 4), torch.randn(10, 4, 5)).shape     # (10, 3, 5) batched
torch.matmul(torch.randn(10, 3, 4), torch.randn(4, 5)).shape      # @ broadcasts batch dims -> (10, 3, 5)
u, w = torch.randn(4), torch.randn(4)
torch.mv(A, u); torch.dot(u, w); torch.inner(u, w); torch.outer(u, w); torch.linalg.vecdot(u, w)
torch.einsum("ij,jk->ik", A, B); torch.einsum("bij,bjk->bik", torch.randn(2, 3, 4), torch.randn(2, 4, 5))
torch.tensordot(A, B, dims=([1], [0])); torch.linalg.cross(torch.randn(3), torch.randn(3))
torch.addmm(torch.zeros(3, 5), A, B, beta=1.0, alpha=1.0); torch.baddbmm(torch.zeros(2, 3, 5), torch.randn(2, 3, 4), torch.randn(2, 4, 5))
S = torch.randn(3, 3); S = S @ S.T + 3 * torch.eye(3)   # symmetric positive-definite
torch.trace(S); torch.diagonal(S); torch.linalg.det(S); torch.linalg.slogdet(S); torch.linalg.inv(S); torch.linalg.pinv(A)
bvec = torch.randn(3); torch.linalg.solve(S, bvec)       # prefer solve over inv(S) @ b
torch.linalg.lstsq(A.T, torch.randn(4, 1)).solution
torch.linalg.eigh(S)                             # symmetric: real eigenvalues (ascending) and eigenvectors
torch.linalg.eig(S); torch.linalg.eigvalsh(S)    # eig returns complex tensors
U, Sv, Vh = torch.linalg.svd(A, full_matrices=False); torch.linalg.svdvals(A); torch.svd_lowrank(A, q=2)
Q, R = torch.linalg.qr(A); L = torch.linalg.cholesky(S); torch.cholesky_solve(bvec[:, None], L)
torch.linalg.matrix_rank(A); torch.linalg.matrix_power(S, 3); torch.linalg.matrix_exp(S); torch.linalg.cond(S)
torch.linalg.multi_dot([A, B, torch.randn(5, 2)]); torch.pca_lowrank(torch.randn(20, 5), q=2)
torch.fft.rfft(torch.randn(16)); torch.fft.fftfreq(8); torch.fft.fft2(torch.randn(4, 4))
torch.sparse_coo_tensor(torch.tensor([[0, 1], [1, 0]]), torch.tensor([3., 4.]), (2, 2)).to_dense()
torch.set_printoptions(precision=3, sci_mode=False, linewidth=120)
```

**Drills:** (1) Create a 4×4 tensor of the integers 0–15 and select the anti-diagonal. (2) Build a one-hot (N, C) matrix 3 different ways (`F.one_hot`, `scatter_`, indexing `torch.eye`). (3) Gather the log-probability of each target token from (B, T, V) logits. (4) Convert a GPU tensor that requires grad into a NumPy array.

### 3.02 Broadcasting & shape changes

```python
x = torch.arange(12)
x.view(3, 4); x.view(-1, 6); x.reshape(2, 2, 3); x.reshape(-1)   # view needs compatible memory; reshape copies if needed
x.view(2, 2, 3).flatten(start_dim=1).shape       # (2, 6)
torch.ravel(x); x.view(3, 4).unflatten(1, (2, 2)).shape   # (3, 2, 2)
y = torch.zeros(1, 3, 1, 4)
y.squeeze().shape, y.squeeze(0).shape, y.squeeze((0, 2)).shape   # (3, 4), (3, 1, 4), (3, 4)
y.unsqueeze(0).shape, x[None, :, None].shape     # (1, 1, 3, 1, 4), (1, 12, 1)
z = torch.randn(2, 3, 4)
z.transpose(0, 2).shape, z.swapaxes(0, 1).shape, z.permute(2, 0, 1).shape, z.movedim(-1, 0).shape, z.mT.shape
z.view_as(torch.empty(6, 4)); z.reshape_as(torch.empty(4, 6)); torch.atleast_2d(torch.tensor(1.)).shape
col = torch.tensor([[1], [2], [3]])              # (3, 1)
col.expand(3, 4); col.expand(-1, 4); col.expand_as(torch.empty(3, 4))   # NO copy (stride 0)
col.repeat(1, 4)                                 # real copy
torch.broadcast_to(col, (2, 3, 4)).shape
# Broadcasting rule: align shapes from the RIGHT; each dim pair must be equal or one of them 1; missing dims count as 1.
(torch.ones(3, 1) + torch.ones(1, 4)).shape      # (3, 4)
(torch.ones(5, 3, 1) + torch.ones(4)).shape      # (5, 3, 4)
torch.broadcast_shapes((5, 1, 4), (3, 1))        # torch.Size([5, 3, 4])
p, q = torch.broadcast_tensors(torch.ones(3, 1), torch.ones(1, 4))
# torch.ones(2, 3) + torch.ones(3, 2)            # RuntimeError: size mismatch
X = torch.randn(100, 5)
X - X.mean(dim=0)                                # (100,5) - (5,): center each column
X / X.norm(dim=1, keepdim=True)                  # keepdim -> (100,1) so it broadcasts per row
((X[:, None, :] - X[None, :, :]) ** 2).sum(-1).shape   # (100, 100) pairwise squared distances
torch.cdist(X, X).shape                          # same thing, efficient, with sqrt
lengths = torch.tensor([2, 4])
mask = torch.arange(5)[None, :] < lengths[:, None]     # (2, 5) padding mask from lengths
scores = torch.randn(2, 4, 5, 5).masked_fill(~mask[:, None, None, :], float("-inf"))   # (B,1,1,T) broadcasts over heads & queries
torch.einsum("bhqd,bhkd->bhqk", torch.randn(2, 4, 5, 8), torch.randn(2, 4, 5, 8)).shape   # attention scores
torch.einsum("...ij,...jk->...ik", torch.randn(7, 2, 3), torch.randn(3, 4)).shape         # ... = any batch dims
# einops (pip install einops): rearrange(x, "b t (h d) -> b h t d", h=4) · reduce(x, "b c h w -> b c", "mean") · repeat(x, "h w -> b h w", b=8)
```

**Watch out:** If `pred` has shape (N, 1) and `target` has shape (N,), then `pred - target` broadcasts to (N, N) and the loss is silently wrong. `squeeze()` with no argument also removes a batch dimension of size 1, so always pass `dim`.

**Drills:** (1) Normalize each row of a (B, D) tensor to unit length. (2) Build a causal mask (T, T) and a padding mask (B, T), then combine them into (B, 1, T, T). (3) Convert (B, T, H·D) to (B, H, T, D) and back. (4) Predict the result shape of (8,1,6,1) + (7,1,5).

### 3.03 Autograd, gradient descent & backprop

```python
import torch.nn.functional as F
x = torch.tensor(2.0, requires_grad=True)
y = x ** 3 + 2 * x                               # builds a graph while computing
y.backward()                                     # dy/dx = 3x^2 + 2 = 14
x.grad                                           # tensor(14.)
x.grad.zero_()                                   # gradients ACCUMULATE: reset them between steps
w = torch.randn(3, requires_grad=True); b = torch.zeros(1, requires_grad=True)
out = torch.randn(5, 3) @ w + b
out.retain_grad()                                # keep .grad on a non-leaf tensor (call before backward)
loss = out.pow(2).mean(); loss.backward()
w.grad.shape, b.grad, out.grad.shape, out.grad_fn, out.is_leaf, w.is_leaf
v = torch.randn(3, requires_grad=True); u = v * 2
u.backward(gradient=torch.ones_like(u))          # non-scalar output: pass the upstream gradient (vector-Jacobian product)
x = torch.tensor([1.0, 2.0], requires_grad=True)
(g,) = torch.autograd.grad((x ** 2).sum(), x, create_graph=True)   # functional: returns grads, leaves .grad alone
(h,) = torch.autograd.grad(g.sum(), x)           # second derivative: tensor([2., 2.])
z = (x * 3).sum(); z.backward(retain_graph=True); z.backward()     # backprop twice through the same graph
with torch.no_grad(): y = x * 2                  # no graph is recorded (evaluation, weight updates)
with torch.inference_mode(): y = x * 2           # faster still; outputs can never enter autograd later
with torch.enable_grad(): y = x * 2
torch.set_grad_enabled(False); torch.set_grad_enabled(True)
@torch.no_grad()
def evaluate(model, inputs): return model(inputs)
x.detach(); x.detach().clone(); x.detach().requires_grad_(True)   # cut from the graph / independent copy / new leaf
# gradient descent by hand
true_w = torch.tensor([1.0, -2.0, 0.5]); Xd = torch.randn(100, 3); yd = Xd @ true_w
w = torch.zeros(3, requires_grad=True)
for step in range(200):
    loss = ((Xd @ w - yd) ** 2).mean()
    loss.backward()
    with torch.no_grad():
        w -= 0.1 * w.grad                        # update outside the graph
    w.grad.zero_()
torch.allclose(w, true_w, atol=1e-3)             # True
# backpropagation by hand vs autograd for a 2-layer network
torch.manual_seed(0)
Xb, Yb = torch.randn(4, 3), torch.randn(4, 2)
W1, W2 = torch.randn(3, 5, requires_grad=True), torch.randn(5, 2, requires_grad=True)
H = Xb @ W1; A = H.relu(); P = A @ W2
loss = ((P - Yb) ** 2).mean(); loss.backward()
with torch.no_grad():
    dP = 2 * (P - Yb) / P.numel()                # dL/dP
    dW2 = A.T @ dP                               # chain rule through P = A @ W2
    dA = dP @ W2.T
    dH = dA * (H > 0)                            # ReLU passes gradient only where H > 0
    dW1 = Xb.T @ dH
torch.allclose(dW1, W1.grad), torch.allclose(dW2, W2.grad)   # (True, True)
# torch.func: composable, functional transforms
from torch.func import grad, vmap, jacrev, jacfwd, hessian, functional_call
f = lambda t: (t ** 3).sum()
grad(f)(torch.tensor([1., 2.]))                  # tensor([ 3., 12.])
jacrev(torch.sin)(torch.tensor([0., 1.])); jacfwd(torch.sin)(torch.tensor([0., 1.])); hessian(f)(torch.tensor([1., 2.]))
torch.autograd.functional.jacobian(torch.sin, torch.tensor([0., 1.]))
lin = torch.nn.Linear(3, 1)
params = {k: p.detach() for k, p in lin.named_parameters()}
def loss_fn(params, xi, yi):
    return F.mse_loss(functional_call(lin, params, (xi.unsqueeze(0),)).squeeze(0), yi)
per_sample = vmap(grad(loss_fn), in_dims=(None, 0, 0))(params, torch.randn(8, 3), torch.randn(8, 1))
per_sample["weight"].shape                       # (8, 1, 3): one gradient per example
```

**Watch out:** Forgetting `zero_grad` makes gradients pile up across steps. Changing a tensor in place after it was used in the forward pass raises "a variable needed for gradient computation has been modified by an inplace operation".

**Drills:** (1) Compute d/dx of `sin(x)·x²` at x = 1 with autograd and check it analytically. (2) Fit y = 3x + 2 by hand-written gradient descent. (3) Backprop a 2-layer MLP manually and match autograd. (4) Compute per-sample gradients with `vmap(grad(...))`.

### 3.04 Dataset & DataLoader

```python
from torch.utils.data import (Dataset, IterableDataset, DataLoader, TensorDataset, random_split, Subset, ConcatDataset,
                              WeightedRandomSampler, RandomSampler, SequentialSampler, SubsetRandomSampler, BatchSampler, get_worker_info)
from torch.nn.utils.rnn import pad_sequence
class TabularDataset(Dataset):                   # map-style: implement __len__ and __getitem__
    def __init__(self, X, y, transform=None):
        self.X = torch.as_tensor(X, dtype=torch.float32); self.y = torch.as_tensor(y, dtype=torch.long); self.transform = transform
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        x = self.X[i]
        if self.transform: x = self.transform(x)
        return x, self.y[i]
ds = TabularDataset(np.random.randn(100, 5), np.random.randint(0, 2, 100))
len(ds); x0, y0 = ds[0]
dl = DataLoader(ds, batch_size=16, shuffle=True, num_workers=0, drop_last=False, pin_memory=torch.cuda.is_available())
xb, yb = next(iter(dl)); xb.shape, yb.shape, len(dl)   # (16, 5), (16,), 7 batches
for batch_idx, (xb, yb) in enumerate(dl): pass
tds = TensorDataset(torch.randn(100, 5), torch.randint(0, 2, (100,)))   # wrap ready-made tensors
train_ds, val_ds = random_split(tds, [0.8, 0.2], generator=torch.Generator().manual_seed(42))
Subset(tds, range(10)); ConcatDataset([tds, tds])
DataLoader(tds, batch_size=32, shuffle=True, num_workers=4, persistent_workers=True, prefetch_factor=2)   # parallel loading
def collate_pad(batch):                          # variable-length sequences -> padded batch + lengths
    seqs, labels = zip(*batch)
    return pad_sequence(seqs, batch_first=True, padding_value=0), torch.tensor([len(s) for s in seqs]), torch.tensor(labels)
seq_ds = [(torch.randint(1, 100, (int(n),)), int(n) % 2) for n in torch.randint(3, 10, (20,))]
padded, lengths, labels = next(iter(DataLoader(seq_ds, batch_size=4, collate_fn=collate_pad)))
labels_all = tds.tensors[1]
weights = 1.0 / torch.bincount(labels_all).float()
sampler = WeightedRandomSampler(weights[labels_all], num_samples=len(labels_all), replacement=True)   # rebalance classes
DataLoader(tds, batch_size=32, sampler=sampler)  # sampler and shuffle=True are mutually exclusive
RandomSampler(tds); SequentialSampler(tds); SubsetRandomSampler(range(50)); list(BatchSampler(range(10), batch_size=4, drop_last=False))
class Stream(IterableDataset):                   # iterable-style: data streams, shards, generators
    def __init__(self, n): self.n = n
    def __iter__(self):
        info = get_worker_info()                 # split the work so that workers don't duplicate items
        start, step = (0, 1) if info is None else (info.id, info.num_workers)
        for i in range(start, self.n, step): yield torch.tensor([float(i)])
[b.shape for b in DataLoader(Stream(10), batch_size=4)]
def seed_worker(worker_id):                      # make NumPy/random differ per worker
    np.random.seed(torch.initial_seed() % 2**32)
DataLoader(tds, num_workers=2, worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(0))
# torchvision: datasets.MNIST(root, train=True, download=True, transform=transforms.ToTensor())
```

**Watch out:** With `num_workers > 0` on macOS and Windows, scripts need an `if __name__ == "__main__":` guard. Move each batch to the device inside the training loop, not inside `__getitem__`.

**Drills:** (1) Write a Dataset over a CSV that returns (features, label) tensors. (2) Build train/val loaders with an 80/20 split and a fixed seed. (3) Write a `collate_fn` that pads token sequences and returns an attention mask. (4) Oversample a 5%-minority class with `WeightedRandomSampler`.

### 3.05 MLP training: modules, losses, optimizers & the loop

#### The canonical training loop

```python
import torch.nn as nn
torch.manual_seed(0)
X = torch.randn(2000, 20); y = (X[:, 0] + X[:, 1] ** 2 > 1).long()
train_ds, val_ds = random_split(TensorDataset(X, y), [1600, 400])
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True); val_dl = DataLoader(val_ds, batch_size=256)
class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, p=0.1):
        super().__init__()                       # ALWAYS call first
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(d_hidden, d_out))
    def forward(self, x):
        return self.net(x)                       # return raw logits; the loss applies softmax
model = MLP(20, 64, 2).to(device)
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
for epoch in range(10):
    model.train()                                # dropout/batchnorm in training mode
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)                       # 1. forward
        loss = loss_fn(logits, yb)               # 2. loss
        opt.zero_grad(set_to_none=True)          # 3. clear old gradients
        loss.backward()                          # 4. backprop
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)   # (optional) clip
        opt.step()                               # 5. update weights
    sched.step()                                 # per-epoch scheduler
    model.eval(); correct = total = 0
    with torch.no_grad():
        for xb, yb in val_dl:
            pred = model(xb.to(device)).argmax(dim=1)
            correct += (pred == yb.to(device)).sum().item(); total += yb.numel()
    print(f"epoch {epoch} loss {loss.item():.3f} val_acc {correct / total:.3f} lr {sched.get_last_lr()[0]:.2e}")
```

#### nn.Module API

```python
model.parameters(); model.named_parameters(); model.buffers(); model.children(); model.modules(); model.named_modules()
sum(p.numel() for p in model.parameters() if p.requires_grad)   # trainable parameter count
print(model)                                     # architecture summary
sd = model.state_dict(); model.load_state_dict(sd, strict=True)   # returns missing_keys / unexpected_keys
model.train(); model.eval(); model.training
model.to(device); model.to(torch.float32); model.float(); model.half(); model.cpu()
model.zero_grad(set_to_none=True); model.requires_grad_(False); model.requires_grad_(True)   # freeze / unfreeze everything
for p in model.net[0].parameters(): p.requires_grad = False      # freeze one layer
model.get_submodule("net.0"); model.get_parameter("net.0.weight"); model.net[-1]
class Scaler(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d))                 # trainable, in state_dict
        self.register_buffer("running_mean", torch.zeros(d))     # NOT trainable, but saved and moved by .to()
        self.layers = nn.ModuleList([nn.Linear(d, d) for _ in range(2)])   # a plain Python list would NOT register these
        self.heads = nn.ModuleDict({"a": nn.Linear(d, 1), "b": nn.Linear(d, 2)})
    def forward(self, x):
        for layer in self.layers: x = layer(x)
        return self.heads["a"](x * self.scale)
Scaler(4)(torch.randn(2, 4)).shape
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None: nn.init.zeros_(m.bias)
model.apply(init_weights)                        # runs on every submodule
w = torch.empty(64, 32)
nn.init.kaiming_normal_(w, mode="fan_in", nonlinearity="relu"); nn.init.kaiming_uniform_(w); nn.init.xavier_normal_(w)
nn.init.normal_(w, mean=0.0, std=0.02); nn.init.trunc_normal_(w, std=0.02); nn.init.uniform_(w, -0.1, 0.1)
nn.init.zeros_(w); nn.init.ones_(w); nn.init.constant_(w, 0.5); nn.init.orthogonal_(w); nn.init.eye_(w)
```

#### Layers & activations

```python
nn.Linear(16, 32)                                # (*, 16) -> (*, 32)
nn.Bilinear(8, 8, 4); nn.Identity(); nn.LazyLinear(32)   # Lazy: in_features inferred at first call
nn.Conv1d(3, 8, kernel_size=3, padding=1)        # (N, C, L)
conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)   # (N, C, H, W); out = (H + 2p - k)/s + 1
conv(torch.randn(8, 3, 32, 32)).shape            # (8, 16, 32, 32)
nn.Conv2d(3, 16, 3, padding="same", dilation=2, groups=1); nn.Conv3d(1, 4, 3); nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2)
nn.MaxPool2d(2); nn.AvgPool2d(2); nn.AdaptiveAvgPool2d((1, 1)); nn.AdaptiveMaxPool1d(1)
nn.BatchNorm1d(64); nn.BatchNorm2d(16); nn.LayerNorm(64); nn.GroupNorm(4, 16); nn.InstanceNorm2d(16); nn.RMSNorm(64)
nn.Dropout(0.1); nn.Dropout2d(0.1)
emb = nn.Embedding(num_embeddings=1000, embedding_dim=64, padding_idx=0)
emb(torch.randint(0, 1000, (2, 7))).shape        # (2, 7, 64): integer ids -> vectors
nn.EmbeddingBag(1000, 64, mode="mean")
lstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
out, (h_n, c_n) = lstm(torch.randn(4, 7, 10)); out.shape, h_n.shape   # (4, 7, 40), (4, 4, 20) = (layers*dirs, N, H)
nn.GRU(10, 20, batch_first=True); nn.RNN(10, 20, nonlinearity="tanh", batch_first=True)
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
packed = pack_padded_sequence(torch.randn(2, 5, 10), torch.tensor([5, 3]), batch_first=True, enforce_sorted=False)
out, _ = lstm(packed); out, lens = pad_packed_sequence(out, batch_first=True)   # RNN skips the padding
nn.Flatten(start_dim=1); nn.Unflatten(1, (16, 4, 4)); nn.Upsample(scale_factor=2, mode="bilinear"); nn.PixelShuffle(2)
nn.ReLU(); nn.LeakyReLU(0.01); nn.PReLU(); nn.ELU(); nn.SELU(); nn.GELU(approximate="tanh"); nn.SiLU(); nn.Mish()
nn.Sigmoid(); nn.Tanh(); nn.Softmax(dim=-1); nn.LogSoftmax(dim=-1); nn.Softplus(); nn.Hardtanh(); nn.ReLU6(); nn.Hardswish(); nn.GLU()
t = torch.randn(4, 10)
F.relu(t); F.gelu(t); F.silu(t); F.softmax(t, dim=-1); F.log_softmax(t, dim=-1); F.logsigmoid(t); F.softplus(t)
F.one_hot(torch.tensor([0, 2]), num_classes=3); F.normalize(t, p=2, dim=-1); F.cosine_similarity(t, t, dim=-1)
F.pad(t, (1, 2)); F.pad(torch.randn(1, 3, 8, 8), (1, 1, 2, 2), mode="constant", value=0)   # pads the LAST dim first
F.interpolate(torch.randn(1, 3, 8, 8), size=(16, 16), mode="bilinear", align_corners=False)
F.dropout(t, p=0.5, training=True); F.linear(t, torch.randn(5, 10)); F.embedding(torch.tensor([1, 2]), torch.randn(10, 4))
F.layer_norm(t, (10,)); F.unfold(torch.randn(1, 3, 8, 8), kernel_size=3).shape   # im2col patches (1, 27, 36)
F.gumbel_softmax(t, tau=1.0, hard=True); torch.cdist(t, t)
```

#### Losses (input shapes matter)

```python
logits, target = torch.randn(8, 5), torch.randint(0, 5, (8,))
nn.CrossEntropyLoss()(logits, target)            # logits (N, C) RAW, targets (N,) int64 class ids
nn.CrossEntropyLoss(weight=torch.ones(5), ignore_index=-100, label_smoothing=0.1)
F.cross_entropy(torch.randn(2, 5, 7), torch.randint(0, 5, (2, 7)))   # sequences: (N, C, T) with targets (N, T)
lm_logits, lm_targets = torch.randn(2, 7, 100), torch.randint(0, 100, (2, 7))
F.cross_entropy(lm_logits.reshape(-1, 100), lm_targets.reshape(-1))   # language models: flatten (B, T, V) -> (B*T, V)
F.cross_entropy(logits, F.softmax(torch.randn(8, 5), -1))   # soft-label targets (N, C) probabilities
nn.NLLLoss()(F.log_softmax(logits, -1), target)  # CrossEntropy = LogSoftmax + NLLLoss
nn.BCEWithLogitsLoss()(torch.randn(8), torch.randint(0, 2, (8,)).float())   # binary/multi-label: raw logits, float targets
nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]))   # up-weight the rare positives
nn.BCELoss()(torch.sigmoid(torch.randn(8)), torch.ones(8))   # expects probabilities (less stable: prefer WithLogits)
nn.MSELoss()(torch.randn(8), torch.randn(8)); nn.L1Loss(); nn.SmoothL1Loss(beta=1.0); nn.HuberLoss(delta=1.0)
nn.KLDivLoss(reduction="batchmean")(F.log_softmax(logits, -1), F.softmax(torch.randn(8, 5), -1))   # input = log-probs, target = probs
nn.CosineEmbeddingLoss(); nn.MarginRankingLoss(margin=0.5); nn.TripletMarginLoss(margin=1.0); nn.CTCLoss(); nn.GaussianNLLLoss(); nn.PoissonNLLLoss()
F.mse_loss(torch.randn(8), torch.randn(8), reduction="none").shape   # reduction: "mean" (default) | "sum" | "none"
```

#### Optimizers & LR schedulers

```python
params = list(model.parameters())
torch.optim.SGD(params, lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4)
torch.optim.Adam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
torch.optim.AdamW(params, lr=3e-4, weight_decay=0.01)   # decoupled weight decay: the transformer default
torch.optim.RMSprop(params, lr=1e-3); torch.optim.Adagrad(params); torch.optim.NAdam(params); torch.optim.RAdam(params); torch.optim.Adamax(params)
lbfgs = torch.optim.LBFGS(params, lr=1)          # needs a closure that re-computes the loss
def closure():
    lbfgs.zero_grad(); l = loss_fn(model(X[:64]), y[:64]); l.backward(); return l
lbfgs.step(closure)
decay = [p for n, p in model.named_parameters() if p.ndim >= 2]      # weight matrices
no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]    # biases & norm weights
opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}], lr=3e-4)   # parameter groups
opt.param_groups[0]["lr"]; [g["lr"] for g in opt.param_groups]
for group in opt.param_groups: group["lr"] = 1e-4   # change the LR by hand
opt.state_dict(); opt.load_state_dict(opt.state_dict())
S = torch.optim.lr_scheduler
S.StepLR(opt, step_size=30, gamma=0.1); S.MultiStepLR(opt, milestones=[30, 80], gamma=0.1); S.ExponentialLR(opt, gamma=0.95)
S.CosineAnnealingLR(opt, T_max=100, eta_min=1e-6); S.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)
S.OneCycleLR(opt, max_lr=1e-2, total_steps=1000)  # call .step() every BATCH
S.LinearLR(opt, start_factor=0.1, total_iters=100); S.ConstantLR(opt, factor=0.5, total_iters=10); S.PolynomialLR(opt, total_iters=100, power=2.0)
plateau = S.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5); plateau.step(0.42)   # pass the monitored metric
warmup, total = 100, 1000
def warmup_cosine(step):                         # the standard LLM schedule as a multiplier of the base LR
    if step < warmup: return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))
sched = S.LambdaLR(opt, lr_lambda=warmup_cosine)
S.SequentialLR(opt, schedulers=[S.LinearLR(opt, start_factor=0.01, total_iters=100), S.CosineAnnealingLR(opt, T_max=900)], milestones=[100])
sched.step(); sched.get_last_lr(); sched.state_dict()
```

#### Saving & loading

```python
torch.save(model.state_dict(), "model.pt")       # save WEIGHTS, not the pickled module
m2 = MLP(20, 64, 2); m2.load_state_dict(torch.load("model.pt", map_location="cpu", weights_only=True)); m2.eval()
ckpt = {"epoch": 10, "model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict()}
torch.save(ckpt, "ckpt.pt")                      # everything needed to RESUME training
ck = torch.load("ckpt.pt", map_location="cpu"); model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); start_epoch = ck["epoch"] + 1
# weights_only=True (the default since 2.6) refuses arbitrary pickled objects; pass weights_only=False only for files you trust
# safetensors: from safetensors.torch import save_file, load_file; save_file(model.state_dict(), "model.safetensors")
exported = torch.export.export(m2, (torch.randn(1, 20),))   # graph capture for deployment (ONNX: torch.onnx.export(m2, (x,), "m.onnx", dynamo=True))
```

**Watch out:** `CrossEntropyLoss` expects raw logits; adding a softmax before it degrades training. Call `model.eval()` and wrap evaluation in `torch.no_grad()`, otherwise dropout stays active and memory grows.

**Drills:** (1) Write the 5-line training step from memory. (2) Overfit one batch to ~0 loss as a sanity check. (3) Train with a warmup + cosine schedule and plot the LR per step. (4) Save a checkpoint mid-training, reload it, and resume with the same loss curve.

### 3.06 Custom autograd Functions

```python
from torch.autograd import gradcheck
class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)                 # stash tensors needed in backward
        return x.clamp(min=0)
    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return grad_output * (x > 0).to(grad_output.dtype)   # one gradient per forward input
x = torch.randn(5, requires_grad=True)
MyReLU.apply(x).sum().backward(); x.grad         # ALWAYS call through .apply
class ScaledMul(torch.autograd.Function):        # several inputs, a non-tensor argument
    @staticmethod
    def forward(ctx, a, b, scale: float):
        ctx.save_for_backward(a, b); ctx.scale = scale   # non-tensors go straight onto ctx
        return a * b * scale
    @staticmethod
    def backward(ctx, g):
        a, b = ctx.saved_tensors
        ga = g * b * ctx.scale if ctx.needs_input_grad[0] else None
        gb = g * a * ctx.scale if ctx.needs_input_grad[1] else None
        return ga, gb, None                      # None for inputs that need no gradient
inputs = (torch.randn(4, dtype=torch.double, requires_grad=True), torch.randn(4, dtype=torch.double, requires_grad=True), 2.0)
gradcheck(ScaledMul.apply, inputs, eps=1e-6, atol=1e-4)   # finite-difference check (use float64!)
class RoundSTE(torch.autograd.Function):         # straight-through estimator (quantization)
    @staticmethod
    def forward(ctx, x): return x.round()
    @staticmethod
    def backward(ctx, g): return g               # pretend round() is the identity
ste_inline = lambda t: t + (t.round() - t).detach()   # the same trick without a Function class
class GradReverse(torch.autograd.Function):      # gradient reversal (domain adaptation)
    @staticmethod
    def forward(ctx, x, lam): ctx.lam = lam; return x.view_as(x)
    @staticmethod
    def backward(ctx, g): return -ctx.lam * g, None
class Exp(torch.autograd.Function):              # new style: forward without ctx + setup_context (works with torch.func/vmap)
    generate_vmap_rule = True
    @staticmethod
    def forward(x): return x.exp()
    @staticmethod
    def setup_context(ctx, inputs, output): ctx.save_for_backward(output)
    @staticmethod
    def backward(ctx, g):
        (y,) = ctx.saved_tensors
        return g * y
Exp.apply(torch.tensor([0., 1.], requires_grad=True)).sum().backward()
class STEQuant(nn.Module):                       # wrap in a Module to use it like a layer
    def forward(self, x): return RoundSTE.apply(x)
# other ctx tools: ctx.mark_non_differentiable(t) · ctx.mark_dirty(t) (in-place) · ctx.set_materialize_grads(False)
```

**Drills:** (1) Implement a custom `Sigmoid` Function and pass `gradcheck`. (2) Implement a clipped-gradient identity (forward = x, backward = clamp(g, −1, 1)). (3) Implement Swish `x·σ(x)` saving only what backward needs.

### 3.07 Memory, strides & contiguity

```python
x = torch.arange(12).view(3, 4)
x.stride(), x.is_contiguous()                    # (4, 1): elements to skip per step along each dim; True
t = x.t(); t.shape, t.stride(), t.is_contiguous()   # (4, 3), (1, 4), False: same memory, new strides
# t.view(-1)                                     # RuntimeError: view size is not compatible with stride
t.reshape(-1); t.contiguous().view(-1)           # reshape copies only when it must; contiguous() copies explicitly
x.data_ptr() == t.data_ptr(), x[1:].storage_offset()   # True (shared storage), 4
e = torch.randn(3, 1).expand(3, 1000); e.stride()      # (1, 0): stride 0 = broadcast without memory
x.element_size() * x.nelement()                  # bytes used
torch.as_strided(torch.arange(6), size=(2, 2), stride=(1, 1))   # [[0, 1], [1, 2]]: windows by stride tricks
x.unfold(dimension=1, size=2, step=1).shape      # (3, 3, 2) sliding windows (a view)
row = x[0]; row += 100; x[0]                     # views share memory: x changed too
safe = x[0].clone(); safe += 1                   # clone breaks the sharing
arr = np.zeros(3); tt = torch.from_numpy(arr); tt += 1; arr   # array([1., 1., 1.]): shared with NumPy
img = torch.randn(8, 3, 32, 32).to(memory_format=torch.channels_last)   # NHWC layout: faster convs on GPUs
img.is_contiguous(memory_format=torch.channels_last), img.stride()
# GPU memory: torch.cuda.memory_allocated() · memory_reserved() · max_memory_allocated() · reset_peak_memory_stats()
#             torch.cuda.empty_cache() · torch.cuda.mem_get_info() · print(torch.cuda.memory_summary())
# fragmentation: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# training memory ≈ weights + grads + optimizer state (Adam: 2 extra copies) + activations (∝ batch × seq × layers)
n_params = 7e9; print(f"7B params, fp32 Adam ≈ {n_params * (4 + 4 + 8) / 1e9:.0f} GB before activations")
```

**Drills:** (1) Make a non-contiguous tensor three different ways and fix each with `.contiguous()`. (2) Show that editing a `view` changes the base but editing a `clone` doesn't. (3) Estimate the training memory of a 1B-parameter model with AdamW in bf16 mixed precision.

### 3.08 Attention

```python
B, H, T, D = 2, 4, 8, 16                         # batch, heads, sequence length, head dim
q, k, v = (torch.randn(B, H, T, D) for _ in range(3))
scores = q @ k.transpose(-2, -1) / math.sqrt(D)  # (B, H, T, T)
causal = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)   # True above the diagonal = future positions
attn = scores.masked_fill(causal, float("-inf")).softmax(dim=-1)      # each row sums to 1
out = attn @ v                                   # (B, H, T, D)
out2 = F.scaled_dot_product_attention(q, k, v, is_causal=True)       # fused kernel (Flash / memory-efficient / math)
torch.allclose(out, out2, atol=1e-5)             # True
F.scaled_dot_product_attention(q, k, v, attn_mask=~causal)            # bool mask: True = ATTEND
pad = torch.tensor([[1] * 8, [1] * 5 + [0] * 3], dtype=torch.bool)    # (B, T): True = real token
F.scaled_dot_product_attention(q, k, v, attn_mask=pad[:, None, None, :])   # broadcast over heads and queries
F.scaled_dot_product_attention(q, k, v, dropout_p=0.1, scale=0.5)     # dropout only during training; scale overrides 1/sqrt(D)
xs = torch.randn(B, T, H * D)
split = xs.view(B, T, H, D).transpose(1, 2)      # (B, H, T, D): split into heads
merged = split.transpose(1, 2).contiguous().view(B, T, H * D)         # merge heads back
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.d = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)       # one fused projection for q, k, v
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = dropout
    def forward(self, x, causal=True):
        Bx, Tx, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q, k, v = (t.view(Bx, Tx, self.h, self.d).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=causal, dropout_p=self.dropout if self.training else 0.0)
        return self.proj(y.transpose(1, 2).reshape(Bx, Tx, C))
MultiHeadSelfAttention(64, 4)(torch.randn(2, 10, 64)).shape         # (2, 10, 64)
mha = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
xq = torch.randn(2, 10, 64)
y, weights = mha(xq, xq, xq, need_weights=True)  # weights (2, 10, 10), averaged over heads
y, _ = mha(xq, xq, xq, attn_mask=nn.Transformer.generate_square_subsequent_mask(10), key_padding_mask=torch.zeros(2, 10, dtype=torch.bool))
enc = torch.randn(2, 12, 64); y, _ = mha(xq, enc, enc)   # cross-attention: queries from decoder, keys/values from encoder
Hkv = 2; kg = torch.randn(B, Hkv, T, D)          # grouped-query attention: fewer K/V heads
F.scaled_dot_product_attention(q, kg.repeat_interleave(H // Hkv, dim=1), kg.repeat_interleave(H // Hkv, dim=1))
from torch.nn.attention import sdpa_kernel, SDPBackend
with sdpa_kernel(SDPBackend.MATH): F.scaled_dot_product_attention(q, k, v)   # force a backend (FLASH_ATTENTION needs GPU + fp16/bf16)
# FlexAttention (GPU, compiled): from torch.nn.attention.flex_attention import flex_attention, create_block_mask
#   block_mask = create_block_mask(lambda b, h, q_idx, kv_idx: q_idx >= kv_idx, B=None, H=None, Q_LEN=T, KV_LEN=T)
#   flex_attention(q, k, v, block_mask=block_mask, score_mod=lambda score, b, h, qi, ki: score + (qi - ki))
```

**Watch out:** The boolean mask convention differs between APIs. In `masked_fill` True means "block". In `scaled_dot_product_attention` True means "attend". In `nn.MultiheadAttention`, True in `key_padding_mask` and in a boolean `attn_mask` means "ignore". Always check which one an API uses.

**Drills:** (1) Implement attention from scratch and match `F.scaled_dot_product_attention`. (2) Add a padding mask built from sequence lengths. (3) Implement cross-attention between two sequences of different lengths. (4) Return the attention weights and plot one head as a heatmap.

### 3.09 Transformer block

```python
class RMSNorm(nn.Module):                        # nn.RMSNorm exists; written out here for practice
    def __init__(self, d, eps=1e-6):
        super().__init__(); self.eps = eps; self.weight = nn.Parameter(torch.ones(d))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
class MLPBlock(nn.Module):
    def __init__(self, d, hidden):
        super().__init__(); self.fc1 = nn.Linear(d, hidden); self.fc2 = nn.Linear(hidden, d)
    def forward(self, x): return self.fc2(F.gelu(self.fc1(x)))
class SwiGLU(nn.Module):                         # the LLaMA-style feed-forward
    def __init__(self, d, hidden):
        super().__init__()
        self.w1, self.w3 = nn.Linear(d, hidden, bias=False), nn.Linear(d, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d, bias=False)
    def forward(self, x): return self.w2(F.silu(self.w1(x)) * self.w3(x))
class Block(nn.Module):                          # pre-norm residual block (GPT-2 / LLaMA layout)
    def __init__(self, d, n_heads, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = MultiHeadSelfAttention(d, n_heads, dropout)
        self.mlp = MLPBlock(d, mlp_ratio * d)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        x = x + self.drop(self.attn(self.ln1(x)))   # x + f(norm(x))
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x
class TinyGPT(nn.Module):
    def __init__(self, vocab, d=128, n_heads=4, n_layers=2, max_len=256):
        super().__init__()
        self.max_len = max_len
        self.tok, self.pos = nn.Embedding(vocab, d), nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList(Block(d, n_heads) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight      # weight tying
    def forward(self, idx, targets=None):
        Bi, Ti = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(Ti, device=idx.device))
        for blk in self.blocks: x = blk(x)
        logits = self.head(self.ln_f(x))         # (B, T, V)
        loss = None if targets is None else F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss
gpt = TinyGPT(vocab=100)
idx = torch.randint(0, 100, (2, 16)); logits, loss = gpt(idx[:, :-1], idx[:, 1:])   # next-token prediction: targets shifted by one
sum(p.numel() for p in gpt.parameters())
@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -model.max_len:])
        logits = logits[:, -1, :] / temperature  # last position only
        if top_k is not None:
            kth = logits.topk(top_k).values[:, [-1]]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        nxt = torch.multinomial(logits.softmax(-1), num_samples=1)   # sample; use argmax(-1, keepdim=True) for greedy
        idx = torch.cat([idx, nxt], dim=1)
    return idx
generate(gpt.eval(), torch.zeros(1, 1, dtype=torch.long), max_new_tokens=5, top_k=10).shape   # (1, 6)
def sinusoidal_positions(Tn, d):                 # the original Transformer encoding
    pos, i = torch.arange(Tn)[:, None], torch.arange(0, d, 2)
    angle = pos / (10000 ** (i / d)); pe = torch.zeros(Tn, d)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(angle), torch.cos(angle)
    return pe
def apply_rope(x, base=10000.0):                 # rotary embeddings: rotate (even, odd) pairs by a position angle; x: (B, H, T, D)
    Tn, Dn = x.shape[-2], x.shape[-1]
    inv_freq = 1.0 / base ** (torch.arange(0, Dn, 2, device=x.device) / Dn)
    ang = torch.arange(Tn, device=x.device)[:, None] * inv_freq   # (T, D/2)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * ang.cos() - x2 * ang.sin(), x1 * ang.sin() + x2 * ang.cos()), dim=-1).flatten(-2)
apply_rope(torch.randn(2, 4, 8, 16)).shape
layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=512, dropout=0.1, activation="gelu", batch_first=True, norm_first=True)
nn.TransformerEncoder(layer, num_layers=2)(torch.randn(2, 10, 128)).shape   # built-in equivalent (encoder)
```

**Drills:** (1) Write a pre-norm block from memory and count its parameters for d = 512. (2) Swap LayerNorm for RMSNorm and GELU MLP for SwiGLU. (3) Train TinyGPT on a character dataset until it overfits one paragraph. (4) Add RoPE to q and k inside the attention.

### 3.10 Hooks & debugging

```python
net = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 2))
activations = {}
def save_activation(name):
    def hook(module, inputs, output):            # forward hook signature
        activations[name] = output.detach()
    return hook
handles = [m.register_forward_hook(save_activation(n)) for n, m in net.named_modules() if isinstance(m, nn.Linear)]
net(torch.randn(4, 10)); {k: v.shape for k, v in activations.items()}   # {'0': (4, 20), '2': (4, 2)}
for h in handles: h.remove()                     # ALWAYS remove hooks when you're done
h = net[1].register_forward_hook(lambda m, i, o: o * 0); h.remove()   # returning a value REPLACES the output (ablation)
h = net[0].register_forward_pre_hook(lambda m, args: (args[0].clamp(-3, 3),)); h.remove()   # edit inputs before forward
grads = {}
h = net[0].register_full_backward_hook(lambda m, gin, gout: grads.update(l0=gout[0].detach()))
net(torch.randn(4, 10)).sum().backward(); h.remove(); grads["l0"].shape
xg = torch.randn(3, requires_grad=True)
hx = xg.register_hook(lambda g: g * 2)           # tensor hook: inspect or replace the gradient flowing through
(xg * 1).sum().backward(); xg.grad; hx.remove()  # tensor([2., 2., 2.])
# with_kwargs: m.register_forward_hook(hook, with_kwargs=True) -> hook(module, args, kwargs, output)
torch.autograd.set_detect_anomaly(True)          # points to the op that produced NaN/Inf in backward (slow; debug only)
torch.autograd.set_detect_anomaly(False)
for n, p in net.named_parameters():              # per-layer gradient health check
    if p.grad is not None: print(f"{n:10s} grad_norm={p.grad.norm():.2e} weight_norm={p.norm():.2e}")
total_norm = nn.utils.clip_grad_norm_(net.parameters(), max_norm=float("inf"))   # measures the global norm without clipping
torch.isfinite(net[0].weight).all(); torch.isnan(net[0].weight).any()
def shape_logger(m, inp, out): print(f"{m.__class__.__name__:8s} {tuple(inp[0].shape)} -> {tuple(out.shape)}")
hs = [m.register_forward_hook(shape_logger) for m in net]; net(torch.randn(2, 10)); [h.remove() for h in hs]
gradcheck(torch.sin, (torch.randn(4, dtype=torch.float64, requires_grad=True),))
# breakpoint() inside forward() · torchinfo.summary(net, input_size=(2, 10)) · overfit ONE batch first: loss must reach ~0
```

**Watch out:** Common errors and their usual cause:
- "mat1 and mat2 shapes cannot be multiplied": wrong `in_features`.
- "Expected all tensors to be on the same device": a tensor wasn't moved with `.to(device)`.
- "expected scalar type Long": cross-entropy targets must be `int64`.
- "does not require grad": you're inside `no_grad` or computing on detached tensors.
- "backward through the graph a second time": add `retain_graph=True` or recompute the forward pass.

**Drills:** (1) Record the mean and std of every layer's activations for one batch. (2) Find the layer whose gradient norm explodes in a deliberately bad init. (3) Zero-ablate one attention head with a hook and measure the loss change.

### 3.11 KV cache

```python
class CachedSelfAttention(nn.Module):
    def __init__(self, d, h):
        super().__init__(); self.h, self.dh = h, d // h
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
    def forward(self, x, cache=None):            # x: (B, T_new, d); cache: (k, v), each (B, H, T_past, dh)
        Bx, Tx, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q, k, v = (t.view(Bx, Tx, self.h, self.dh).transpose(1, 2) for t in (q, k, v))
        if cache is not None:                    # append the new keys/values along the sequence dim
            k, v = torch.cat([cache[0], k], dim=2), torch.cat([cache[1], v], dim=2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=(cache is None and Tx > 1))   # one new token may see everything cached
        return self.proj(y.transpose(1, 2).reshape(Bx, Tx, C)), (k, v)
attn = CachedSelfAttention(64, 4).eval()
seq = torch.randn(1, 10, 64)
with torch.no_grad():
    full, _ = attn(seq)                          # whole sequence in one pass (no cache)
    y, cache = attn(seq[:, :6])                  # prefill: process the prompt, keep its keys/values
    outs = [y]
    for t in range(6, 10):                       # decode: ONE token per step, reusing the cache
        y_t, cache = attn(seq[:, t:t + 1], cache)
        outs.append(y_t)
torch.allclose(torch.cat(outs, dim=1), full, atol=1e-5)   # True: identical results, far less compute per step
cache[0].shape                                   # (1, 4, 10, 16): grows by one position per token
class StaticKVCache:                             # preallocated buffer: no re-allocation per step (CUDA-graph friendly)
    def __init__(self, B, H, max_len, Dh, dtype=torch.float32, device=None):
        self.k = torch.zeros(B, H, max_len, Dh, dtype=dtype, device=device); self.v = torch.zeros_like(self.k); self.pos = 0
    def update(self, k_new, v_new):
        Tn = k_new.size(2)
        self.k[:, :, self.pos:self.pos + Tn] = k_new; self.v[:, :, self.pos:self.pos + Tn] = v_new; self.pos += Tn
        return self.k[:, :, :self.pos], self.v[:, :, :self.pos]
kv = StaticKVCache(1, 4, 32, 16); kk, vv = kv.update(torch.randn(1, 4, 3, 16), torch.randn(1, 4, 3, 16)); kk.shape
layers, Bn, Hkv, Tn, Dh, bytes_per = 32, 1, 8, 8192, 128, 2   # e.g. an 8B-class model with GQA, bf16
print(f"KV cache ≈ {2 * layers * Bn * Hkv * Tn * Dh * bytes_per / 2**30:.1f} GiB")   # 2 = keys + values
# Hugging Face: out = model(input_ids, use_cache=True); out.past_key_values (a Cache object)
#   next step: model(next_token_ids, past_key_values=out.past_key_values, use_cache=True)
#   model.generate(..., use_cache=True) handles all of this; cache_implementation="static" preallocates
```

**Watch out:** With `is_causal=True`, SDPA aligns the mask to the top-left. That is wrong when several new tokens attend to a longer cached past (chunked prefill), so build an explicit mask in that case.

**Drills:** (1) Add a KV cache to TinyGPT's attention and verify generation matches the uncached version. (2) Measure tokens/second with and without the cache at 512 tokens. (3) Compute the KV-cache memory for a 70B model at 32k context.

### 3.12 LoRA layer

```python
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters(): p.requires_grad_(False)   # freeze the pretrained W and b
        self.A = nn.Parameter(torch.empty(r, base.in_features))     # down-projection
        self.B = nn.Parameter(torch.zeros(base.out_features, r))    # up-projection, zero-init -> starts as a no-op
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale, self.drop, self.merged = alpha / r, nn.Dropout(dropout), False
    def forward(self, x):
        out = self.base(x)
        if not self.merged:
            out = out + (self.drop(x) @ self.A.T @ self.B.T) * self.scale   # W x + (alpha/r) * B A x
        return out
    @torch.no_grad()
    def merge(self):                             # fold the update into W for zero-overhead inference
        self.base.weight += (self.B @ self.A) * self.scale; self.merged = True
def add_lora(model, should_wrap=lambda name, m: isinstance(m, nn.Linear), r=8, alpha=16):
    for name, child in model.named_children():   # recursively replace matching Linear layers
        if should_wrap(name, child): setattr(model, name, LoRALinear(child, r, alpha))
        else: add_lora(child, should_wrap, r, alpha)
    return model
base_net = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4))
xin = torch.randn(5, 16); before = base_net(xin)
lora_net = add_lora(base_net, r=4)
torch.allclose(lora_net(xin), before)            # True: B = 0 so nothing changes at init
trainable = sum(p.numel() for p in lora_net.parameters() if p.requires_grad); total_p = sum(p.numel() for p in lora_net.parameters())
print(f"trainable {trainable} / {total_p} ({100 * trainable / total_p:.1f}%)")
opt = torch.optim.AdamW([p for p in lora_net.parameters() if p.requires_grad], lr=1e-3)   # optimize ONLY the adapters
F.mse_loss(lora_net(xin), torch.randn(5, 4)).backward(); opt.step()
adapter_sd = {n: p.detach() for n, p in lora_net.named_parameters() if p.requires_grad}   # save just the adapter (tiny)
unmerged = lora_net(xin); [m.merge() for m in lora_net.modules() if isinstance(m, LoRALinear)]
torch.allclose(lora_net(xin), unmerged, atol=1e-6)   # True: merging doesn't change outputs
# PEFT library: LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.05) -> get_peft_model(model, config)
```

**Drills:** (1) Add LoRA only to layers whose name contains `"attn"`. (2) Check the trainable fraction of parameters for r = 4, 8 and 64. (3) Train, merge and verify that outputs match. (4) Save and load only the adapter weights.

### 3.13 Automatic mixed precision (AMP)

```python
amp_model = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 512)).to(device)
opt = torch.optim.AdamW(amp_model.parameters(), lr=1e-3)
xa, ya = torch.randn(64, 512, device=device), torch.randn(64, 512, device=device)
dev_type = "cuda" if device == "cuda" else "cpu"
with torch.autocast(device_type=dev_type, dtype=torch.bfloat16):   # bf16: same range as fp32, no scaler needed
    out = amp_model(xa)                          # matmuls/convs run in bf16
    loss = F.mse_loss(out.float(), ya)           # keep the loss in fp32
out.dtype                                        # torch.bfloat16
loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
use_fp16 = device == "cuda"                      # fp16 has a tiny range: needs gradient scaling
scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
for step in range(3):
    with torch.autocast(device_type=dev_type, dtype=torch.float16, enabled=use_fp16):
        loss = F.mse_loss(amp_model(xa), ya)
    scaler.scale(loss).backward()                # scale up the loss so small grads don't underflow to 0
    scaler.unscale_(opt)                         # unscale BEFORE clipping or inspecting grads
    nn.utils.clip_grad_norm_(amp_model.parameters(), 1.0)
    scaler.step(opt)                             # skips the step if grads contain inf/NaN
    scaler.update()                              # adjusts the scale factor
    opt.zero_grad(set_to_none=True)
with torch.autocast(device_type=dev_type, dtype=torch.bfloat16):
    with torch.autocast(device_type=dev_type, enabled=False):
        precise = amp_model(xa.float())          # force an fp32 region inside an autocast block
torch.set_float32_matmul_precision("high")       # allow TF32 tensor cores for plain fp32 matmuls (Ampere+)
# torch.cuda.is_bf16_supported() · checkpoints should include scaler.state_dict() · old spelling torch.cuda.amp.* is deprecated
```

**Watch out:** Use autocast instead of `model.half()` for training. Pure fp16 weights lose small updates. Autocast keeps fp32 master weights and casts per operation.

**Drills:** (1) Convert the 3.05 loop to bf16 autocast and compare speed and loss. (2) Do the same with fp16 and a GradScaler. (3) Print the dtype of activations, weights and gradients inside and outside autocast.

### 3.14 Profiler & benchmarking

```python
import time
from torch.profiler import profile, record_function, ProfilerActivity, schedule
prof_model = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024, 10)); xp = torch.randn(256, 1024)
acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if torch.cuda.is_available() else [])
with profile(activities=acts, record_shapes=True, profile_memory=True) as prof:
    with record_function("forward_pass"):        # label a region in the trace
        prof_model(xp)
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))   # GPU: sort_by="cuda_time_total"
print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=5))
prof.export_chrome_trace("trace.json")           # open in https://ui.perfetto.dev or chrome://tracing
with profile(activities=acts, schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
             on_trace_ready=lambda p: p.export_chrome_trace(f"trace_{p.step_num}.json")) as prof:
    for step in range(6):
        prof_model(xp).sum().backward()
        prof.step()                              # tell the profiler a training step ended
def timeit(fn, iters=20):                        # correct wall-clock timing
    for _ in range(3): fn()                      # warm up (caches, cudnn autotune, compilation)
    if torch.cuda.is_available(): torch.cuda.synchronize()   # GPU calls are ASYNC: wait before reading the clock
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    if torch.cuda.is_available(): torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters
print(f"{timeit(lambda: prof_model(xp)) * 1e3:.2f} ms/iter")
import torch.utils.benchmark as benchmark
print(benchmark.Timer(stmt="m(x)", globals={"m": prof_model, "x": xp}).timeit(50))   # handles warmup & sync for you
# GPU events: s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True); s.record(); fn(); e.record(); torch.cuda.synchronize(); s.elapsed_time(e)
# memory timeline: torch.cuda.memory._record_memory_history(); ...; torch.cuda.memory._dump_snapshot("mem.pickle") -> https://pytorch.org/memory_viz
# system level: nsys profile -o report python train.py
```

**Drills:** (1) Profile one training step and name the 3 most expensive ops. (2) Time a matmul at fp32 vs bf16 correctly (with warmup and sync). (3) Use `record_function` to split the time into data loading, forward, backward and optimizer step.

### 3.15 Triton kernels & C++ extensions

```python
# --- Triton (GPU only; ships with CUDA builds of PyTorch on Linux) ---
import triton
import triton.language as tl
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)                  # which block this program instance handles
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements                  # guard the ragged tail
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)
def triton_add(x, y):
    out = torch.empty_like(x); n = out.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)   # number of program instances
    add_kernel[grid](x, y, out, n, BLOCK_SIZE=1024)
    return out
@triton.jit
def softmax_kernel(out_ptr, in_ptr, in_stride, out_stride, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)                       # one program per row
    cols = tl.arange(0, BLOCK)
    x = tl.load(in_ptr + row * in_stride + cols, mask=cols < n_cols, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    tl.store(out_ptr + row * out_stride + cols, num / tl.sum(num, axis=0), mask=cols < n_cols)
def triton_softmax(x):
    out = torch.empty_like(x)
    softmax_kernel[(x.shape[0],)](out, x, x.stride(0), out.stride(0), x.shape[1], BLOCK=triton.next_power_of_2(x.shape[1]))
    return out
# usage (GPU): a = torch.randn(4096, device="cuda"); torch.testing.assert_close(triton_add(a, a), a + a)
# more tl ops: tl.dot · tl.where · tl.zeros((M, N), dtype=tl.float32) · tl.cumsum · tl.sqrt · tl.atomic_add
# autotune: @triton.autotune(configs=[triton.Config({"BLOCK_SIZE": 512}, num_warps=4), ...], key=["n_elements"])
# benchmark: triton.testing.do_bench(lambda: triton_add(a, a))
# --- C++ extension compiled at runtime ---
from torch.utils.cpp_extension import load_inline
cpp_src = """
torch::Tensor scaled_add(torch::Tensor a, torch::Tensor b, double alpha) {
    TORCH_CHECK(a.sizes() == b.sizes(), "shape mismatch");
    return a + alpha * b;
}
"""
# ext = load_inline(name="scaled_add_ext", cpp_sources=cpp_src, functions=["scaled_add"])   # needs a C++ compiler
# ext.scaled_add(torch.ones(3), torch.ones(3), 2.0)   -> tensor([3., 3., 3.])
# files instead: load(name="my_ext", sources=["my_ext.cpp", "kernel.cu"]) or setup.py with CppExtension/CUDAExtension + BuildExtension
# in my_ext.cpp:  PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("scaled_add", &scaled_add, "a + alpha * b"); }
# --- register as a real PyTorch op (works with autograd and torch.compile) ---
@torch.library.custom_op("mylib::scaled_add", mutates_args=())
def scaled_add(a: torch.Tensor, b: torch.Tensor, alpha: float) -> torch.Tensor:
    return a + alpha * b                         # could call your Triton kernel or C++ extension here
@scaled_add.register_fake
def _(a, b, alpha): return torch.empty_like(a)   # shape/dtype propagation for the compiler
def _setup(ctx, inputs, output): ctx.alpha = inputs[2]
def _backward(ctx, grad): return grad, grad * ctx.alpha, None
scaled_add.register_autograd(_backward, setup_context=_setup)
ta = torch.ones(3, requires_grad=True); tb = torch.ones(3, requires_grad=True)
scaled_add(ta, tb, 2.0).sum().backward(); tb.grad   # tensor([2., 2., 2.])
```

**Drills:** (1) Write a Triton kernel for fused `relu(x + bias)` and test it against PyTorch. (2) Benchmark your softmax kernel against `torch.softmax` across row sizes. (3) Register the kernel with `torch.library.custom_op` and call it inside a compiled function.

### 3.16 Distributed Data Parallel (DDP)

```python
# train_ddp.py — launch with:  torchrun --standalone --nproc_per_node=4 train_ddp.py
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
def main():
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])   # torchrun sets RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
    dev = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    if dev.type == "cuda": torch.cuda.set_device(dev)
    ds = TensorDataset(torch.randn(1024, 20), torch.randint(0, 2, (1024,)))
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True)   # each rank gets a different shard
    dl = DataLoader(ds, batch_size=32, sampler=sampler)          # NOT shuffle=True
    net = DDP(nn.Linear(20, 2).to(dev), device_ids=[local_rank] if dev.type == "cuda" else None)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    for epoch in range(3):
        sampler.set_epoch(epoch)                 # reshuffle differently every epoch
        for xb, yb in dl:
            loss = F.cross_entropy(net(xb.to(dev)), yb.to(dev))
            opt.zero_grad(); loss.backward(); opt.step()   # gradients are all-reduced during backward
        avg = loss.detach().clone(); dist.all_reduce(avg, op=dist.ReduceOp.SUM); avg /= world
        if rank == 0: print(f"epoch {epoch} mean loss {avg.item():.4f}")   # log and save from rank 0 only
    if rank == 0: torch.save(net.module.state_dict(), "model.pt")   # .module unwraps DDP
    dist.barrier(); dist.destroy_process_group()
if __name__ == "__main__" and "LOCAL_RANK" in os.environ: main()
# collectives: dist.broadcast(t, src=0) · all_reduce(t) · reduce(t, dst=0) · all_gather(list_out, t) · all_gather_into_tensor(out, t)
#              reduce_scatter_tensor(out, t) · all_to_all_single(out, t) · barrier() · broadcast_object_list(objs, src=0)
# gradient accumulation without redundant syncs:
#   with net.no_sync(): (loss / accum).backward()   # on all but the last micro-batch
# multi-node: torchrun --nnodes=2 --node_rank=0 --nproc_per_node=8 --master_addr=10.0.0.1 --master_port=29500 train_ddp.py
# debug: NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DEBUG=DETAIL · BatchNorm: nn.SyncBatchNorm.convert_sync_batchnorm(model)
```

**Drills:** (1) Convert the 3.05 loop to DDP and run it with 2 CPU processes (`gloo`). (2) Average a validation metric across ranks with `all_reduce`. (3) Add gradient accumulation with `no_sync`.

### 3.17 FSDP & checkpointing

```python
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
class CheckpointedBlock(nn.Module):              # activation checkpointing: store less, recompute in backward
    def __init__(self, block): super().__init__(); self.block = block
    def forward(self, x): return checkpoint(self.block, x, use_reentrant=False)
ck_net = nn.Sequential(*[CheckpointedBlock(Block(64, 4)) for _ in range(2)])
ck_net(torch.randn(2, 10, 64)).sum().backward()  # same gradients, ~1 block of activations in memory at a time
checkpoint_sequential(nn.Sequential(*[nn.Linear(32, 32) for _ in range(4)]), 2, torch.randn(4, 32, requires_grad=True), use_reentrant=False)
# Hugging Face models: model.gradient_checkpointing_enable()
# FSDP2 (inside a torchrun job): shard params, grads and optimizer state across ranks
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
def shard_model(model):
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for blk in model.blocks: fully_shard(blk, mp_policy=mp)   # shard each transformer block...
    fully_shard(model, mp_policy=mp)             # ...then the root module
    return model                                 # create the optimizer AFTER sharding
# legacy FSDP1: FullyShardedDataParallel(model, auto_wrap_policy=..., mixed_precision=..., sharding_strategy=ShardingStrategy.FULL_SHARD)
import torch.distributed.checkpoint as dcp       # distributed checkpoints: every rank saves/loads its own shard
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict, get_model_state_dict, StateDictOptions
def save_sharded(model, opt, path):
    model_sd, optim_sd = get_state_dict(model, opt)
    dcp.save({"model": model_sd, "optim": optim_sd}, checkpoint_id=path)
def load_sharded(model, opt, path):
    model_sd, optim_sd = get_state_dict(model, opt)
    dcp.load({"model": model_sd, "optim": optim_sd}, checkpoint_id=path)   # loads in place, resharding if needed
    set_state_dict(model, opt, model_state_dict=model_sd, optim_state_dict=optim_sd)
def export_full(model):                          # one regular state_dict (e.g. for Hugging Face upload)
    return get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
# single-process resume checkpoint: see 3.05 (model + optimizer + scheduler + scaler + step + RNG state)
```

**Drills:** (1) Measure peak memory of TinyGPT with and without activation checkpointing. (2) Wrap TinyGPT with `fully_shard` per block in a 2-process run. (3) Save a DCP checkpoint, kill the job, and resume from it.

### 3.18 torch.compile

```python
cmodel = torch.compile(gpt)                      # default: TorchDynamo graph capture + Inductor codegen
torch.compile(gpt, mode="reduce-overhead")       # CUDA graphs: cuts launch overhead for small batches / inference
torch.compile(gpt, mode="max-autotune")          # longest compile, fastest kernels
torch.compile(gpt, fullgraph=True)               # raise an error on any graph break
torch.compile(gpt, dynamic=True)                 # expect varying input shapes
torch.compile(gpt, backend="eager")              # debug: capture only (also "aot_eager")
@torch.compile
def gelu_tanh(x): return 0.5 * x * (1 + torch.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))
gelu_tanh(torch.randn(8))                        # first call compiles (slow); later calls are fast
explanation = torch._dynamo.explain(gelu_tanh)(torch.randn(8))
explanation.graph_count, explanation.graph_break_count
@torch.compiler.disable
def log_stats(x): print(x.shape)                 # keep a function out of the graph
torch._dynamo.mark_dynamic(torch.randn(4, 8), 0) # mark dim 0 dynamic to avoid recompiles
torch.compiler.reset()                           # clear compiled caches
# graph-break causes: .item(), print, .tolist(), data-dependent Python if/for on tensor VALUES, unsupported libraries
# inspect: TORCH_LOGS="graph_breaks,recompiles" python train.py · TORCH_LOGS="output_code" shows generated Triton
# gpt.compile() compiles in place and keeps state_dict keys clean; torch.compile(gpt) returns a wrapper whose keys start with "_orig_mod."
```

**Drills:** (1) Compile TinyGPT and benchmark (with warmup) against eager mode. (2) Write a function with a graph break, find it with `explain`, and remove it. (3) Trigger recompiles by changing input shapes, then fix them with `dynamic=True`.

### 3.19 DPO loss

```python
def sequence_logprobs(logits, labels, mask):
    """Sum of log p(token_t | tokens_<t) over the response tokens.
    logits (B, T, V) from model(input_ids); labels (B, T) = input_ids; mask (B, T) = 1 on response tokens."""
    logits, labels, mask = logits[:, :-1], labels[:, 1:], mask[:, 1:]   # position t predicts token t+1
    logp = torch.log_softmax(logits.float(), dim=-1)
    token_logp = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)     # (B, T-1)
    return (token_logp * mask).sum(-1)           # (B,)
def dpo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta=0.1, label_smoothing=0.0):
    pi_logratios = pi_chosen - pi_rejected       # how much the policy prefers chosen over rejected
    ref_logratios = ref_chosen - ref_rejected    # ...and how much the frozen reference did
    logits = beta * (pi_logratios - ref_logratios)
    losses = -F.logsigmoid(logits) * (1 - label_smoothing) - F.logsigmoid(-logits) * label_smoothing
    chosen_rewards = beta * (pi_chosen - ref_chosen).detach()     # implicit rewards, for logging
    rejected_rewards = beta * (pi_rejected - ref_rejected).detach()
    return losses.mean(), chosen_rewards, rejected_rewards
Bp = 4
pc, pr = torch.randn(Bp, requires_grad=True), torch.randn(Bp, requires_grad=True)
rc, rr = torch.randn(Bp), torch.randn(Bp)
loss, cr, rj = dpo_loss(pc, pr, rc, rr); loss.backward()
(cr > rj).float().mean(), (cr - rj).mean()       # reward accuracy and reward margin (the two curves to watch)
policy, ref = TinyGPT(vocab=100), TinyGPT(vocab=100)   # one step: chosen and rejected in ONE forward pass
ref.load_state_dict(policy.state_dict()); ref.eval().requires_grad_(False)   # reference = frozen copy of the starting policy
chosen_ids, rejected_ids = torch.randint(0, 100, (Bp, 12)), torch.randint(0, 100, (Bp, 12))
resp_mask = torch.zeros(Bp, 12); resp_mask[:, 6:] = 1   # prompt = first 6 tokens, response = the rest
ids, mask = torch.cat([chosen_ids, rejected_ids]), torch.cat([resp_mask, resp_mask])
pi_c, pi_r = sequence_logprobs(policy(ids)[0], ids, mask).chunk(2)
with torch.no_grad(): ref_c, ref_r = sequence_logprobs(ref(ids)[0], ids, mask).chunk(2)
loss, *_ = dpo_loss(pi_c, pi_r, ref_c, ref_r, beta=0.1); loss.backward()
loss.item()                                      # ≈ log(2) = 0.693 at the start (policy == reference)
lr_ = pi_c - pi_r - (ref_c - ref_r)              # variants share the same log-ratio:
ipo = ((lr_ - 1 / (2 * 0.1)) ** 2).mean()        # IPO: squared loss toward a margin of 1/(2*beta)
hinge = torch.relu(1 - 0.1 * lr_).mean()         # hinge / SLiC style
# SimPO (reference-free): -logsigmoid(beta * (pi_c / len_c - pi_r / len_r) - gamma)
# production: trl.DPOTrainer(model, args=DPOConfig(beta=0.1), train_dataset=prompt/chosen/rejected rows, processing_class=tokenizer)
```

**Drills:** (1) Derive why the loss starts at log 2. (2) Implement length-normalized log-probs and SimPO. (3) Train TinyGPT on synthetic preferences, e.g. "prefer sequences ending in token 7", and plot reward accuracy.

Full reference: https://docs.pytorch.org/docs/stable/index.html

## 4. scikit-learn

`import sklearn` · written against scikit-learn 1.8. Every model follows the same contract: construct with hyperparameters, then `.fit(X, y)`, then `.predict(X)` / `.transform(X)`. `X` is a 2-D array or DataFrame of shape (n_samples, n_features). Learned attributes end with an underscore (`coef_`, `classes_`).

### 4.01 The estimator API

```python
import numpy as np, pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone, is_classifier
X, y = load_iris(return_X_y=True)
clf = LogisticRegression(max_iter=1000)          # 1. construct with hyperparameters
clf.fit(X, y)                                    # 2. learn (returns self, so calls chain)
clf.predict(X[:3])                               # class labels
clf.predict_proba(X[:3]).round(3)                # class probabilities (rows sum to 1)
clf.predict_log_proba(X[:3]); clf.decision_function(X[:3])   # log-probs / raw margins
clf.score(X, y)                                  # default metric: accuracy (R² for regressors)
clf.classes_, clf.coef_.shape, clf.intercept_, clf.n_features_in_, clf.n_iter_
clf.get_params(); clf.set_params(C=0.5)
fresh = clone(clf)                               # same hyperparameters, NOT fitted
is_classifier(clf)
scaler = StandardScaler()                        # transformers: fit learns statistics, transform applies them
Xs = scaler.fit_transform(X)                     # = scaler.fit(X).transform(X)
scaler.mean_, scaler.scale_; scaler.inverse_transform(Xs[:2]); scaler.get_feature_names_out()
```

**Drills:** (1) Fit a classifier and print its classes, coefficients and training accuracy. (2) Clone it with a different `C` and show that the clone is unfitted. (3) Standardize a matrix and invert the transform.

### 4.02 Datasets

```python
from sklearn import datasets
X, y = datasets.load_iris(return_X_y=True)                     # (150, 4), 3 classes
iris = datasets.load_iris(as_frame=True); iris.frame.head(); iris.feature_names; iris.target_names
datasets.load_digits(return_X_y=True); datasets.load_wine(return_X_y=True)
datasets.load_breast_cancer(return_X_y=True); datasets.load_diabetes(return_X_y=True)
# downloaded on first use (cached in ~/scikit_learn_data):
# fetch_california_housing(as_frame=True) · fetch_openml("mnist_784", version=1, as_frame=False) · fetch_20newsgroups(subset="train")
datasets.make_classification(n_samples=500, n_features=10, n_informative=4, n_redundant=2, n_classes=3, random_state=0)
datasets.make_classification(n_samples=1000, weights=[0.95, 0.05], random_state=0)   # imbalanced binary
datasets.make_regression(n_samples=200, n_features=5, noise=10.0, coef=True, random_state=0)
datasets.make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=0)
datasets.make_moons(n_samples=200, noise=0.1, random_state=0); datasets.make_circles(n_samples=200, noise=0.05, factor=0.5, random_state=0)
datasets.make_swiss_roll(n_samples=500, noise=0.1, random_state=0); datasets.make_multilabel_classification(n_samples=100, n_classes=4, random_state=0)
```

**Drills:** (1) Load breast cancer as a DataFrame and list its 5 first features. (2) Generate a 3-class dataset with 2 informative features. (3) Generate imbalanced data (1% positives) and check the class counts.

### 4.03 Splitting & cross-validation

```python
from sklearn.model_selection import (train_test_split, KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit, ShuffleSplit,
    StratifiedShuffleSplit, LeaveOneOut, RepeatedStratifiedKFold, cross_val_score, cross_validate, cross_val_predict)
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42, stratify=y_train)   # 60/20/20
model = RandomForestClassifier(n_estimators=50, random_state=0, n_jobs=-1)
cross_val_score(model, X_train, y_train, cv=5)                   # 5 accuracy scores
cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc").mean()
res = cross_validate(model, X_train, y_train, cv=5, scoring=["accuracy", "f1"], return_train_score=True)
res["test_f1"].mean(), res["train_accuracy"].mean(), res["fit_time"]
oof = cross_val_predict(model, X_train, y_train, cv=5, method="predict_proba")[:, 1]   # out-of-fold predictions
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):
    Xa, Xb, ya, yb = X_train[tr_idx], X_train[va_idx], y_train[tr_idx], y_train[va_idx]
KFold(n_splits=5, shuffle=True, random_state=0); ShuffleSplit(n_splits=10, test_size=0.2, random_state=0); StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
groups = np.arange(len(X_train)) % 20            # e.g. patient or user ids
cross_val_score(model, X_train, y_train, groups=groups, cv=GroupKFold(n_splits=5))   # a group never sits on both sides
for tr, va in TimeSeriesSplit(n_splits=3).split(np.arange(12)): print(tr, va)        # train on the past, validate on the future
RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0); LeaveOneOut()
```

**Watch out:** Use `GroupKFold` when several rows come from the same person or source, and `TimeSeriesSplit` for temporal data. A plain shuffled split leaks information in both cases.

**Drills:** (1) Make a stratified 70/15/15 split. (2) Run 5-fold CV reporting accuracy and F1 together. (3) Write the manual `StratifiedKFold` loop and collect fold scores in a list.

### 4.04 Scaling & feature transforms

```python
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, Normalizer, QuantileTransformer,
    PowerTransformer, PolynomialFeatures, SplineTransformer, FunctionTransformer, KBinsDiscretizer, Binarizer)
X = np.array([[1.0, 200.0], [2.0, 300.0], [3.0, 10_000.0], [4.0, 400.0]])
StandardScaler().fit_transform(X)                # (x - mean) / std per column
MinMaxScaler(feature_range=(0, 1)).fit_transform(X)
RobustScaler().fit_transform(X)                  # median / IQR: robust to outliers
MaxAbsScaler().fit_transform(X)                  # no centering: keeps sparse matrices sparse
Normalizer(norm="l2").fit_transform(X)           # scales each ROW to unit length
QuantileTransformer(n_quantiles=4, output_distribution="normal").fit_transform(X)
PowerTransformer(method="yeo-johnson").fit_transform(X)   # makes skewed features more Gaussian
PolynomialFeatures(degree=2, include_bias=False).fit_transform(X[:, :1])   # x, x^2
PolynomialFeatures(degree=2, interaction_only=True, include_bias=False).fit(X).get_feature_names_out(["a", "b"])
SplineTransformer(n_knots=4, degree=3).fit_transform(X[:, :1]).shape
FunctionTransformer(np.log1p, inverse_func=np.expm1).fit_transform(X)
KBinsDiscretizer(n_bins=2, encode="ordinal", strategy="uniform").fit_transform(X)
Binarizer(threshold=2.5).fit_transform(X)
X_train, X_test = X[:3], X[3:]                   # THE rule: fit on train, transform train AND test with it
sc = StandardScaler().fit(X_train); X_train_s, X_test_s = sc.transform(X_train), sc.transform(X_test)
```

**Drills:** (1) Compare StandardScaler and RobustScaler on a column with one huge outlier. (2) Create degree-3 polynomial features with readable names. (3) Log-transform a skewed column with `FunctionTransformer`.

### 4.05 Encoding categories & imputing missing values

```python
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder, LabelBinarizer, MultiLabelBinarizer, TargetEncoder
from sklearn.impute import SimpleImputer, KNNImputer, MissingIndicator
from sklearn.experimental import enable_iterative_imputer  # noqa: F401 (unlocks IterativeImputer)
from sklearn.impute import IterativeImputer
cat = pd.DataFrame({"color": ["red", "blue", "red", "green"], "size": ["S", "M", "L", "M"]})
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(cat)
ohe.transform(pd.DataFrame({"color": ["purple"], "size": ["S"]}))   # unseen category -> all zeros for color
ohe.get_feature_names_out(); ohe.categories_
OneHotEncoder(drop="first", sparse_output=False).fit_transform(cat)   # drop one level per feature (linear models)
OneHotEncoder(min_frequency=2, handle_unknown="infrequent_if_exist", sparse_output=False).fit_transform(cat)   # rare levels grouped
OrdinalEncoder(categories=[["red", "green", "blue"], ["S", "M", "L"]]).fit_transform(cat)   # explicit order
OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(cat).transform(pd.DataFrame({"color": ["x"], "size": ["S"]}))
LabelEncoder().fit_transform(["cat", "dog", "cat"])   # for TARGETS (y) only, not features
LabelBinarizer().fit_transform(["cat", "dog", "bird"]); MultiLabelBinarizer().fit_transform([["a", "b"], ["c"]])
TargetEncoder(target_type="continuous", cv=2).fit_transform(cat[["color"]], np.array([1.0, 3.0, 2.0, 4.0]))   # cross-fitted target means
Xm = np.array([[1.0, np.nan], [2.0, 4.0], [np.nan, 6.0], [4.0, 8.0]])
SimpleImputer(strategy="mean").fit_transform(Xm)  # also "median", "most_frequent", "constant"
SimpleImputer(strategy="constant", fill_value=0, add_indicator=True).fit_transform(Xm)   # adds missing-flag columns
KNNImputer(n_neighbors=2).fit_transform(Xm); IterativeImputer(random_state=0).fit_transform(Xm); MissingIndicator().fit_transform(Xm)
```

**Drills:** (1) One-hot encode two columns and check that an unseen category doesn't crash. (2) Ordinal-encode `size` with S < M < L. (3) Median-impute and add missing-indicator columns.

### 4.06 Pipelines & ColumnTransformer

```python
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector, TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn import set_config
rng = np.random.default_rng(0)
df = pd.DataFrame({"age": rng.integers(18, 70, 200).astype(float), "income": rng.normal(50, 15, 200),
                   "city": rng.choice(["NYC", "SF", "LA"], 200), "plan": rng.choice(["free", "pro"], 200)})
df.loc[::17, "income"] = np.nan
y = (df["income"].fillna(50) + rng.normal(0, 10, 200) > 55).astype(int)
num_cols, cat_cols = ["age", "income"], ["city", "plan"]
preprocess = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
], remainder="drop")                             # "passthrough" keeps unlisted columns
model = Pipeline([("prep", preprocess), ("clf", LogisticRegression(max_iter=1000))])
model.fit(df, y); model.predict(df.head()); model.score(df, y)   # ONE object: no train/test leakage
model.named_steps["clf"].coef_; model["prep"]; model[-1]; model[:-1].transform(df.head(2))   # index & slice steps
model.set_params(clf__C=0.1)                     # <step>__<param>: the same syntax grid search uses
model[:-1].get_feature_names_out()               # ['num__age', 'num__income', 'cat__city_LA', ...]
quick = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))   # auto step names: standardscaler, logisticregression
ct = make_column_transformer((StandardScaler(), make_column_selector(dtype_include="number")),
                             (OneHotEncoder(handle_unknown="ignore"), make_column_selector(dtype_exclude="number")))
ct.fit_transform(df.fillna(0)).shape
preprocess.set_output(transform="pandas"); preprocess.fit_transform(df).head()   # DataFrame output with feature names
set_config(transform_output="pandas"); set_config(transform_output="default")   # global switch
Xi, yi = load_iris(return_X_y=True)
FeatureUnion([("pca", PCA(n_components=2)), ("best", SelectKBest(k=1))]).fit_transform(Xi, yi).shape   # (150, 3)
TransformedTargetRegressor(regressor=Ridge(), func=np.log1p, inverse_func=np.expm1)   # model a log-target transparently
# in Jupyter, just display `model` to get an interactive diagram of the pipeline
```

**Watch out:** Never fit a scaler or encoder on the full dataset before splitting. Put preprocessing inside the pipeline so that cross-validation refits it on each training fold.

**Drills:** (1) Build a pipeline: impute → scale numerics, one-hot categoricals → logistic regression. (2) Change the classifier's `C` with `set_params`. (3) Print the feature names that come out of the preprocessing step.

### 4.07 Feature selection

```python
from sklearn.feature_selection import (SelectKBest, SelectPercentile, f_classif, chi2, mutual_info_classif,
    VarianceThreshold, RFE, RFECV, SelectFromModel, SequentialFeatureSelector)
from sklearn.neighbors import KNeighborsClassifier
X, y = load_breast_cancer(return_X_y=True); Xs = StandardScaler().fit_transform(X)
skb = SelectKBest(score_func=f_classif, k=10).fit(X, y); skb.get_support(indices=True); skb.scores_[:3]; skb.transform(X).shape
SelectKBest(chi2, k=5).fit_transform(X, y)       # chi2 needs non-negative features
SelectPercentile(mutual_info_classif, percentile=20).fit_transform(X, y).shape
VarianceThreshold(threshold=0.01).fit_transform(X).shape
RFE(LogisticRegression(max_iter=5000), n_features_to_select=5, step=2).fit(Xs, y).ranking_
RFECV(LogisticRegression(max_iter=5000), cv=3, scoring="accuracy", min_features_to_select=3).fit(Xs, y).n_features_
SelectFromModel(RandomForestClassifier(n_estimators=50, random_state=0), threshold="median").fit(X, y).get_support().sum()
SequentialFeatureSelector(KNeighborsClassifier(3), n_features_to_select=3, direction="forward", cv=3).fit(Xs[:, :8], y).get_support()
```

**Drills:** (1) Keep the 5 best features by mutual information and print their names. (2) Run RFECV and plot score vs number of features. (3) Drop near-constant columns.

### 4.08 Linear models

```python
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet, RidgeCV, LassoCV, LogisticRegressionCV, SGDClassifier,
    SGDRegressor, HuberRegressor, QuantileRegressor, PoissonRegressor, BayesianRidge, RidgeClassifier, Perceptron)
from sklearn.datasets import make_regression
Xr, yr, true_coef = make_regression(n_samples=200, n_features=5, n_informative=3, noise=5, coef=True, random_state=0)
lr = LinearRegression().fit(Xr, yr); lr.coef_, lr.intercept_, lr.score(Xr, yr)   # score = R²
Ridge(alpha=1.0).fit(Xr, yr); Lasso(alpha=0.5).fit(Xr, yr).coef_   # L2 shrinks weights; L1 zeroes some out
ElasticNet(alpha=0.5, l1_ratio=0.5).fit(Xr, yr)
RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(Xr, yr).alpha_; LassoCV(cv=5, random_state=0).fit(Xr, yr).alpha_
HuberRegressor().fit(Xr, yr); QuantileRegressor(quantile=0.9, alpha=0.0).fit(Xr, yr); BayesianRidge().fit(Xr, yr)
PoissonRegressor().fit(Xr, np.abs(yr).round())   # count targets
SGDRegressor(max_iter=1000, tol=1e-3, alpha=1e-4).fit(StandardScaler().fit_transform(Xr), yr)
Xc, yc = load_breast_cancer(return_X_y=True); Xc = StandardScaler().fit_transform(Xc)
LogisticRegression(C=1.0, max_iter=1000).fit(Xc, yc)   # C = 1 / regularization strength (smaller C = stronger)
LogisticRegression(C=0.5, l1_ratio=1.0, solver="saga", max_iter=5000).fit(Xc, yc)   # L1 (1.8+: l1_ratio replaces penalty=)
LogisticRegression(C=0.5, l1_ratio=0.5, solver="saga", max_iter=5000).fit(Xc, yc)   # elastic net
LogisticRegression(C=np.inf, max_iter=1000); LogisticRegression(class_weight="balanced", max_iter=1000)   # no penalty / reweight rare classes
LogisticRegressionCV(Cs=10, cv=5, max_iter=2000).fit(Xc, yc).C_
SGDClassifier(loss="log_loss", alpha=1e-4, random_state=0).fit(Xc, yc)   # loss="hinge" = linear SVM
sgd = SGDClassifier(loss="log_loss", random_state=0)
for i in range(0, len(Xc), 100): sgd.partial_fit(Xc[i:i + 100], yc[i:i + 100], classes=np.array([0, 1]))   # streaming / out-of-core
RidgeClassifier().fit(Xc, yc); Perceptron().fit(Xc, yc)
```

**Drills:** (1) Fit Lasso at 3 alphas and count the non-zero coefficients. (2) Fit L1-regularized logistic regression and list the surviving features. (3) Train an `SGDClassifier` in mini-batches with `partial_fit`.

### 4.09 SVMs, nearest neighbours & naive Bayes

```python
from sklearn.svm import SVC, LinearSVC, SVR, OneClassSVM
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors, NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB, CategoricalNB
svc = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=0).fit(Xc, yc)
svc.predict_proba(Xc[:2]); svc.support_vectors_.shape; svc.decision_function(Xc[:2])
SVC(kernel="linear"); SVC(kernel="poly", degree=3); LinearSVC(C=0.5).fit(Xc, yc)   # LinearSVC scales to large data
SVR(kernel="rbf", C=10, epsilon=0.1).fit(Xr, yr); OneClassSVM(nu=0.05).fit(Xc).predict(Xc[:5])   # +1 inlier / -1 outlier
knn = KNeighborsClassifier(n_neighbors=5, weights="distance", metric="minkowski", p=2).fit(Xc, yc); knn.predict(Xc[:3])
KNeighborsRegressor(n_neighbors=3).fit(Xr, yr)
nn_index = NearestNeighbors(n_neighbors=3, metric="cosine").fit(Xc)
dist, idx = nn_index.kneighbors(Xc[:2])          # similarity search / retrieval
NearestCentroid().fit(Xc, yc)
GaussianNB().fit(Xc, yc).predict_proba(Xc[:2])   # continuous features
counts = np.random.default_rng(0).integers(0, 5, size=(100, 20)); yb = np.random.default_rng(1).integers(0, 2, 100)
MultinomialNB(alpha=1.0).fit(counts, yb); ComplementNB().fit(counts, yb); BernoulliNB().fit(counts > 0, yb); CategoricalNB().fit(counts, yb)
```

**Watch out:** SVMs and k-NN depend on distances, so always scale the features first.

**Drills:** (1) Compare linear and RBF SVC accuracy with 5-fold CV. (2) Find the 5 nearest neighbours of a query row. (3) Classify word-count vectors with `MultinomialNB`.

### 4.10 Trees & ensembles

```python
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier,
    HistGradientBoostingRegressor, AdaBoostClassifier, BaggingClassifier, VotingClassifier, StackingClassifier)
data = load_breast_cancer(); X, y = data.data, data.target
tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, criterion="gini", random_state=0).fit(X, y)
print(export_text(tree, feature_names=list(data.feature_names)))   # the learned rules as text
tree.get_depth(); tree.get_n_leaves(); tree.feature_importances_[:5]
tree.cost_complexity_pruning_path(X, y).ccp_alphas[:5]; DecisionTreeClassifier(ccp_alpha=0.01)   # pruning strength
rf = RandomForestClassifier(n_estimators=200, max_features="sqrt", oob_score=True, n_jobs=-1, random_state=0).fit(X, y)
rf.oob_score_; rf.feature_importances_.argsort()[::-1][:5]; len(rf.estimators_)
ExtraTreesClassifier(n_estimators=200, random_state=0).fit(X, y)
hgb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, early_stopping=True, random_state=0).fit(X, y)
hgb.n_iter_                                      # boosting rounds actually used; NaN inputs handled natively
HistGradientBoostingClassifier(categorical_features="from_dtype")   # pandas category columns used natively
GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, subsample=0.8, random_state=0).fit(X, y)
AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=100, random_state=0).fit(X, y)
BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=20, max_samples=0.8, random_state=0).fit(X, y)
VotingClassifier([("lr", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),
                  ("rf", RandomForestClassifier(random_state=0))], voting="soft").fit(X, y)
StackingClassifier([("rf", RandomForestClassifier(n_estimators=50, random_state=0)), ("svc", make_pipeline(StandardScaler(), SVC(random_state=0)))],
                   final_estimator=LogisticRegression(), cv=5).fit(X, y)
RandomForestRegressor(n_estimators=100, random_state=0).fit(Xr, yr); HistGradientBoostingRegressor(loss="absolute_error").fit(Xr, yr)
```

**Drills:** (1) Print a depth-2 tree's rules. (2) Compare RF, HistGradientBoosting and a soft-voting ensemble with 5-fold CV. (3) Use early stopping and report the number of boosting rounds used.

### 4.11 Clustering & mixture models

```python
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN, HDBSCAN, AgglomerativeClustering, SpectralClustering, MeanShift, estimate_bandwidth, Birch, OPTICS
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.datasets import make_blobs, make_moons
Xb, yb = make_blobs(n_samples=300, centers=4, cluster_std=0.7, random_state=0)
km = KMeans(n_clusters=4, n_init="auto", random_state=0).fit(Xb)
km.labels_[:10]; km.cluster_centers_; km.inertia_; km.predict(Xb[:3]); km.transform(Xb[:3])   # transform = distance to each center
inertias = [KMeans(k, n_init="auto", random_state=0).fit(Xb).inertia_ for k in range(1, 8)]   # data for an elbow plot
MiniBatchKMeans(n_clusters=4, batch_size=256, random_state=0).fit(Xb)
Xm, _ = make_moons(n_samples=300, noise=0.05, random_state=0)
db = DBSCAN(eps=0.2, min_samples=5).fit(Xm); set(db.labels_)   # label -1 = noise; no k needed
HDBSCAN(min_cluster_size=10).fit(Xm).labels_; OPTICS(min_samples=10).fit(Xm)
AgglomerativeClustering(n_clusters=2, linkage="single").fit_predict(Xm)   # linkage: ward (default), complete, average, single
SpectralClustering(n_clusters=2, affinity="nearest_neighbors", random_state=0).fit_predict(Xm)
MeanShift(bandwidth=estimate_bandwidth(Xb, quantile=0.2)).fit(Xb).cluster_centers_.shape; Birch(n_clusters=4).fit_predict(Xb)
gmm = GaussianMixture(n_components=4, covariance_type="full", random_state=0).fit(Xb)
gmm.predict_proba(Xb[:2]).round(3); gmm.means_; gmm.bic(Xb); gmm.aic(Xb); gmm.sample(5)[0]   # soft assignments, model selection, sampling
BayesianGaussianMixture(n_components=10, random_state=0).fit(Xb).weights_.round(2)   # switches off unneeded components
```

**Drills:** (1) Choose k for KMeans with the elbow method and with silhouette scores. (2) Cluster two moons with DBSCAN and count the noise points. (3) Choose a GMM's number of components by BIC.

### 4.12 Dimensionality reduction & embeddings

```python
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA, TruncatedSVD, NMF, FastICA, FactorAnalysis, LatentDirichletAllocation
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, SpectralEmbedding
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_digits
X, y = load_digits(return_X_y=True)              # (1797, 64)
pca = PCA(n_components=2).fit(X); Z = pca.transform(X)   # (1797, 2)
pca.explained_variance_ratio_; pca.components_.shape; pca.inverse_transform(Z).shape
PCA(n_components=0.95).fit(X).n_components_      # keep 95% of the variance
np.cumsum(PCA().fit(X).explained_variance_ratio_)[:10]   # cumulative variance (scree plot data)
PCA(n_components=10, whiten=True, random_state=0).fit_transform(X)
IncrementalPCA(n_components=10, batch_size=200).fit(X)   # partial_fit for data that doesn't fit in RAM
KernelPCA(n_components=2, kernel="rbf", gamma=1e-3).fit_transform(X[:300])
TruncatedSVD(n_components=20, random_state=0).fit_transform(X)   # works on sparse input (TF-IDF -> LSA)
NMF(n_components=10, init="nndsvda", max_iter=500, random_state=0).fit_transform(X)   # non-negative parts
FastICA(n_components=10, random_state=0).fit_transform(X); FactorAnalysis(n_components=5).fit_transform(X)
LatentDirichletAllocation(n_components=5, random_state=0).fit_transform(X)   # topics from count data
LinearDiscriminantAnalysis(n_components=2).fit_transform(X, y)   # supervised: maximizes class separation
TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=0).fit_transform(X[:500])   # for plots only: no .transform()
Isomap(n_neighbors=10, n_components=2).fit_transform(X[:500]); SpectralEmbedding(n_components=2, random_state=0).fit_transform(X[:500])
LocallyLinearEmbedding(n_neighbors=10, n_components=2, random_state=0).fit_transform(X[:500])
# UMAP is a separate package: pip install umap-learn -> umap.UMAP(n_neighbors=15, min_dist=0.1).fit_transform(X)
```

**Drills:** (1) How many PCA components keep 90% of the digits' variance? (2) Project digits to 2-D with PCA and with t-SNE and compare the plots. (3) Reconstruct one digit from 10 components and measure the error.

### 4.13 Classification metrics

```python
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, log_loss,
    brier_score_loss, matthews_corrcoef, cohen_kappa_score, top_k_accuracy_score, hamming_loss, jaccard_score, precision_recall_fscore_support)
X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, random_state=0)
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)
y_pred = clf.predict(X_te); y_prob = clf.predict_proba(X_te)[:, 1]   # P(class 1)
accuracy_score(y_te, y_pred); balanced_accuracy_score(y_te, y_pred)
precision_score(y_te, y_pred); recall_score(y_te, y_pred); f1_score(y_te, y_pred); fbeta_score(y_te, y_pred, beta=2)   # beta > 1 favors recall
print(classification_report(y_te, y_pred, target_names=["malignant", "benign"], digits=3))
classification_report(y_te, y_pred, output_dict=True)["macro avg"]["f1-score"]
confusion_matrix(y_te, y_pred); confusion_matrix(y_te, y_pred, normalize="true")   # rows = true class, cols = predicted
tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()
roc_auc_score(y_te, y_prob); average_precision_score(y_te, y_prob)   # need SCORES / probabilities, not hard labels
fpr, tpr, thr = roc_curve(y_te, y_prob); prec, rec, thr_pr = precision_recall_curve(y_te, y_prob)
log_loss(y_te, y_prob); brier_score_loss(y_te, y_prob); matthews_corrcoef(y_te, y_pred); cohen_kappa_score(y_te, y_pred)
(y_prob >= 0.3).astype(int)                      # a custom decision threshold
Xi, yi = load_iris(return_X_y=True); mc = LogisticRegression(max_iter=1000).fit(Xi, yi)
f1_score(yi, mc.predict(Xi), average="macro")    # multiclass averaging: "macro", "micro", "weighted", None (per class)
roc_auc_score(yi, mc.predict_proba(Xi), multi_class="ovr"); top_k_accuracy_score(yi, mc.predict_proba(Xi), k=2)
precision_recall_fscore_support(yi, mc.predict(Xi), average=None)
Y_true, Y_pred = np.array([[1, 0, 1], [0, 1, 0]]), np.array([[1, 0, 0], [0, 1, 0]])   # multilabel
hamming_loss(Y_true, Y_pred); jaccard_score(Y_true, Y_pred, average="samples"); f1_score(Y_true, Y_pred, average="micro")
```

**Watch out:** On imbalanced data, accuracy can look excellent while the model misses every positive. Report precision, recall and F1 or PR-AUC as well.

**Drills:** (1) Compute precision, recall and F1 by hand from `tn, fp, fn, tp` and match sklearn. (2) Find the threshold that maximizes F1 using `precision_recall_curve`. (3) Report macro and weighted F1 for a 3-class model.

### 4.14 Regression, clustering & distance metrics

```python
from sklearn.metrics import (mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error,
    median_absolute_error, explained_variance_score, max_error, mean_squared_log_error, silhouette_score, silhouette_samples,
    adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, davies_bouldin_score, calinski_harabasz_score, pairwise_distances, ndcg_score)
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances, rbf_kernel, linear_kernel
y_true, y_hat = np.array([3.0, 5.0, 2.5, 7.0]), np.array([2.5, 5.0, 3.0, 8.0])
mean_squared_error(y_true, y_hat); root_mean_squared_error(y_true, y_hat)   # RMSE (replaces squared=False)
mean_absolute_error(y_true, y_hat); median_absolute_error(y_true, y_hat); r2_score(y_true, y_hat)
mean_absolute_percentage_error(y_true, y_hat); explained_variance_score(y_true, y_hat); max_error(y_true, y_hat); mean_squared_log_error(y_true, y_hat)
Xb, yb = make_blobs(n_samples=300, centers=3, random_state=0); labels = KMeans(3, n_init="auto", random_state=0).fit_predict(Xb)
silhouette_score(Xb, labels); silhouette_samples(Xb, labels)[:5]; davies_bouldin_score(Xb, labels); calinski_harabasz_score(Xb, labels)   # no labels needed
adjusted_rand_score(yb, labels); normalized_mutual_info_score(yb, labels); homogeneity_score(yb, labels)   # vs ground truth
A, B = np.random.default_rng(0).normal(size=(4, 3)), np.random.default_rng(1).normal(size=(2, 3))
cosine_similarity(A, B).shape; euclidean_distances(A, B); manhattan_distances(A, B); rbf_kernel(A, B, gamma=0.5); linear_kernel(A, B)
pairwise_distances(A, metric="cosine"); pairwise_distances(A, B, metric="euclidean", n_jobs=-1)
ndcg_score([[3, 2, 0, 1]], [[0.9, 0.7, 0.1, 0.5]])   # ranking quality (search, recommendation)
```

**Drills:** (1) Compute RMSE and MAE by hand and match sklearn. (2) Pick k for KMeans by the best silhouette score. (3) Find the most similar pair of rows by cosine similarity.

### 4.15 Hyperparameter search

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.metrics import make_scorer, get_scorer_names
from scipy.stats import loguniform
X, y = load_breast_cancer(return_X_y=True)
pipe = Pipeline([("scale", StandardScaler()), ("clf", SVC())])
grid = GridSearchCV(pipe, param_grid={"clf__C": [0.1, 1, 10], "clf__kernel": ["linear", "rbf"]}, cv=5, scoring="f1", n_jobs=-1)
grid.fit(X, y); grid.best_params_; grid.best_score_; grid.predict(X[:3])   # refit=True: best model retrained on all data
pd.DataFrame(grid.cv_results_)[["params", "mean_test_score", "std_test_score", "rank_test_score"]].sort_values("rank_test_score")
GridSearchCV(pipe, [{"clf__kernel": ["linear"], "clf__C": [1, 10]}, {"clf__kernel": ["rbf"], "clf__gamma": [1e-3, 1e-2]}], cv=3)   # list of grids
rand = RandomizedSearchCV(pipe, {"clf__C": loguniform(1e-3, 1e3), "clf__gamma": loguniform(1e-4, 1e-1)},
                          n_iter=20, cv=3, scoring="roc_auc", random_state=0, n_jobs=-1).fit(X, y)
rand.best_params_
GridSearchCV(pipe, {"clf__C": [1, 10]}, scoring={"auc": "roc_auc", "f1": "f1"}, refit="auc", cv=3).fit(X, y).best_score_   # several metrics
HalvingGridSearchCV(pipe, {"clf__C": [0.1, 1, 10, 100]}, cv=3, factor=2, random_state=0).fit(X, y).best_params_   # successive halving
GridSearchCV(pipe, {"clf__C": [1, 10]}, scoring=make_scorer(fbeta_score, beta=2), cv=3)   # custom metric
"neg_root_mean_squared_error" in get_scorer_names()   # error metrics are negated so that higher is always better
cross_val_score(GridSearchCV(pipe, {"clf__C": [0.1, 1, 10]}, cv=3), X, y, cv=5).mean()   # nested CV: honest estimate of a tuned model
```

**Drills:** (1) Grid-search `max_depth` and `n_estimators` for a random forest. (2) Random-search `C` on a log scale with 30 draws. (3) Run nested CV and compare it with the plain `best_score_`.

### 4.16 Probability calibration & decision thresholds

```python
from sklearn.calibration import CalibratedClassifierCV, calibration_curve, CalibrationDisplay
from sklearn.model_selection import TunedThresholdClassifierCV, FixedThresholdClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=2000, weights=[0.9, 0.1], random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, random_state=0)
cal = CalibratedClassifierCV(GaussianNB(), method="isotonic", cv=5).fit(X_tr, y_tr)   # or method="sigmoid" (Platt scaling)
prob_true, prob_pred = calibration_curve(y_te, cal.predict_proba(X_te)[:, 1], n_bins=10)
fitted = GaussianNB().fit(X_tr, y_tr)
CalibratedClassifierCV(FrozenEstimator(fitted), method="sigmoid").fit(X_te[:200], y_te[:200])   # calibrate an already-trained model
tuned = TunedThresholdClassifierCV(LogisticRegression(max_iter=1000), scoring="f1", cv=5).fit(X_tr, y_tr)
tuned.best_threshold_, tuned.best_score_         # learns the probability cut-off that maximizes F1
FixedThresholdClassifier(LogisticRegression(max_iter=1000), threshold=0.3).fit(X_tr, y_tr).predict(X_te[:5])
```

**Drills:** (1) Plot a reliability curve before and after isotonic calibration. (2) Tune a threshold for recall ≥ 0.9 and report the precision you get. (3) Wrap a trained model with a fixed 0.2 threshold.

### 4.17 Model inspection & built-in plots

```python
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance, PartialDependenceDisplay, DecisionBoundaryDisplay
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay, DetCurveDisplay, PredictionErrorDisplay
from sklearn.model_selection import LearningCurveDisplay, ValidationCurveDisplay
data = load_breast_cancer(as_frame=True); X, y = data.data, data.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, random_state=0)
clf = RandomForestClassifier(n_estimators=100, random_state=0).fit(X_tr, y_tr)
ConfusionMatrixDisplay.from_estimator(clf, X_te, y_te, display_labels=["malignant", "benign"], cmap="Blues", normalize="true")
ConfusionMatrixDisplay.from_predictions(y_te, clf.predict(X_te))
RocCurveDisplay.from_estimator(clf, X_te, y_te); PrecisionRecallDisplay.from_estimator(clf, X_te, y_te); DetCurveDisplay.from_estimator(clf, X_te, y_te)
fig, ax = plt.subplots()                         # overlay several models on one Axes
RocCurveDisplay.from_estimator(clf, X_te, y_te, ax=ax, name="RF")
RocCurveDisplay.from_estimator(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X_tr, y_tr), X_te, y_te, ax=ax, name="LR")
CalibrationDisplay.from_estimator(clf, X_te, y_te, n_bins=10)
r = permutation_importance(clf, X_te, y_te, n_repeats=10, random_state=0, n_jobs=-1)
pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False).head(5)   # model-agnostic, on held-out data
pd.Series(clf.feature_importances_, index=X.columns).nlargest(5)   # impurity-based (biased toward high-cardinality features)
PartialDependenceDisplay.from_estimator(clf, X_te, features=["mean radius", ("mean radius", "mean texture")], kind="average")
PartialDependenceDisplay.from_estimator(clf, X_te, features=["worst area"], kind="both", subsample=50, random_state=0)   # PDP + ICE lines
X2, y2 = make_moons(n_samples=200, noise=0.2, random_state=0); svm2 = SVC(gamma=2).fit(X2, y2)
disp = DecisionBoundaryDisplay.from_estimator(svm2, X2, response_method="predict", alpha=0.4)
disp.ax_.scatter(X2[:, 0], X2[:, 1], c=y2, edgecolor="k")
LearningCurveDisplay.from_estimator(clf, X, y, cv=3, train_sizes=np.linspace(0.2, 1.0, 4), score_type="both", n_jobs=-1)
ValidationCurveDisplay.from_estimator(SVC(), StandardScaler().fit_transform(X), y, param_name="C", param_range=np.logspace(-2, 2, 5), cv=3)
PredictionErrorDisplay.from_predictions(yr, Ridge().fit(Xr, yr).predict(Xr), kind="actual_vs_predicted")
plt.close("all")
```

**Drills:** (1) Plot ROC curves for 3 models on one Axes. (2) Rank features by permutation importance on the test set. (3) Draw the decision boundary of k-NN for k = 1 and k = 30.

### 4.18 Text features

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer, HashingVectorizer
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
docs = ["the cat sat on the mat", "the dog sat", "cats and dogs play", "a dog chased the cat"]
cv = CountVectorizer(); Xc = cv.fit_transform(docs)   # sparse (n_docs, vocab_size)
cv.get_feature_names_out(); cv.vocabulary_["cat"]; Xc.toarray()
CountVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.9, stop_words="english", max_features=1000, binary=True).fit_transform(docs)
CountVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit_transform(docs).shape   # character n-grams: robust to typos
tfidf = TfidfVectorizer(sublinear_tf=True, norm="l2"); Xt = tfidf.fit_transform(docs); tfidf.idf_.round(2)
TfidfTransformer().fit_transform(Xc); HashingVectorizer(n_features=2**10).transform(docs).shape   # hashing: stateless, no vocabulary
text_clf = make_pipeline(TfidfVectorizer(), LogisticRegression()).fit(docs, [1, 0, 0, 1]); text_clf.predict(["the cat"])
DictVectorizer(sparse=False).fit_transform([{"city": "NYC", "temp": 20}, {"city": "SF", "temp": 15}])
FeatureHasher(n_features=8, input_type="string").transform([["a", "b"], ["c"]]).toarray()
```

**Drills:** (1) Build a TF-IDF + logistic regression spam classifier in one pipeline. (2) Print the 10 highest-weighted words for each class. (3) Compare word and character n-gram features with CV.

### 4.19 Outliers, multi-output, semi-supervised & other models

```python
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor, ClassifierChain
from sklearn.semi_supervised import SelfTrainingClassifier, LabelSpreading
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.isotonic import IsotonicRegression
from sklearn.kernel_approximation import RBFSampler, Nystroem
from sklearn.dummy import DummyClassifier, DummyRegressor
X, y = load_breast_cancer(return_X_y=True); Xs = StandardScaler().fit_transform(X)
IsolationForest(contamination=0.02, random_state=0).fit(Xs).predict(Xs[:5])   # +1 inlier, -1 outlier
LocalOutlierFactor(n_neighbors=20).fit_predict(Xs)[:5]; LocalOutlierFactor(novelty=True).fit(Xs).predict(Xs[:5])
EllipticEnvelope(random_state=0).fit(Xs[:, :5]).predict(Xs[:5, :5])
OneVsRestClassifier(SVC()).fit(*load_iris(return_X_y=True)); OneVsOneClassifier(LinearSVC()).fit(*load_iris(return_X_y=True))
Y = np.column_stack([y, 1 - y])                  # two binary targets
MultiOutputClassifier(LogisticRegression(max_iter=1000)).fit(Xs, Y).predict(Xs[:2])
ClassifierChain(LogisticRegression(max_iter=1000), order=[0, 1]).fit(Xs, Y)
Xr2, Yr2 = make_regression(n_samples=100, n_features=4, n_targets=2, random_state=0); MultiOutputRegressor(Ridge()).fit(Xr2, Yr2)
y_partial = y.copy(); y_partial[100:] = -1       # -1 marks unlabeled rows
SelfTrainingClassifier(SVC(probability=True, random_state=0)).fit(Xs, y_partial); LabelSpreading(kernel="knn", n_neighbors=7).fit(Xs, y_partial)
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4, learning_rate_init=1e-3, max_iter=300, early_stopping=True, random_state=0).fit(Xs, y)
mlp.loss_curve_[-1], mlp.best_validation_score_
MLPRegressor(hidden_layer_sizes=(32,), max_iter=2000, random_state=0).fit(Xr2, Yr2[:, 0])
xg = np.linspace(0, 10, 30)[:, None]; yg = np.sin(xg).ravel()
gp = GaussianProcessRegressor(kernel=ConstantKernel() * RBF(length_scale=1.0) + WhiteKernel(), normalize_y=True).fit(xg, yg)
mean, std = gp.predict(xg, return_std=True)      # predictions with uncertainty
QuadraticDiscriminantAnalysis().fit(Xs[:, :5], y); IsotonicRegression(out_of_bounds="clip").fit(np.arange(10), np.arange(10) ** 0.5)
RBFSampler(gamma=0.1, random_state=0).fit_transform(Xs).shape; Nystroem(kernel="rbf", n_components=100, random_state=0).fit_transform(Xs).shape
DummyClassifier(strategy="most_frequent").fit(Xs, y).score(Xs, y); DummyRegressor(strategy="mean")   # ALWAYS compare against a baseline
```

**Drills:** (1) Flag the top 1% anomalies with IsolationForest. (2) Train with 90% of labels hidden using self-training. (3) Check that your model beats `DummyClassifier`.

### 4.20 Custom estimators, persistence & configuration

```python
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.utils import shuffle, resample
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn import set_config, config_context
import joblib
class Clipper(TransformerMixin, BaseEstimator):  # mixins go to the LEFT of BaseEstimator
    def __init__(self, quantile=0.01):           # only store params here; no logic, no validation
        self.quantile = quantile
    def fit(self, X, y=None):
        X = validate_data(self, X)               # checks input and sets n_features_in_
        self.low_, self.high_ = np.quantile(X, [self.quantile, 1 - self.quantile], axis=0)
        return self                              # always return self
    def transform(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        return np.clip(X, self.low_, self.high_)
class MajorityClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        X, y = validate_data(self, X, y)
        self.classes_, counts = np.unique(y, return_counts=True); self.majority_ = self.classes_[counts.argmax()]
        return self
    def predict(self, X):
        check_is_fitted(self); X = validate_data(self, X, reset=False)
        return np.full(len(X), self.majority_)
X, y = load_breast_cancer(return_X_y=True)
make_pipeline(Clipper(0.05), StandardScaler(), LogisticRegression(max_iter=1000)).fit(X, y).score(X, y)
MajorityClassifier().fit(X, y).score(X, y)       # ClassifierMixin provides .score (accuracy)
GridSearchCV(make_pipeline(Clipper(), LogisticRegression(max_iter=1000)), {"clipper__quantile": [0.0, 0.05]}, cv=3).fit(X, y).best_params_
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X, y)
joblib.dump(model, "model.joblib"); loaded = joblib.load("model.joblib"); loaded.score(X, y)   # pickle-based: trusted files only, same sklearn version
with config_context(assume_finite=True): model.predict(X[:5])
shuffle(X, y, random_state=0)[0].shape; resample(X, y, n_samples=100, stratify=y, random_state=0)[0].shape
compute_class_weight("balanced", classes=np.unique(y), y=y); compute_sample_weight("balanced", y)[:5]
# check_estimator(Clipper()) from sklearn.utils.estimator_checks runs the full API-compliance test suite
```

**Drills:** (1) Write a transformer that adds `log1p` copies of selected columns. (2) Tune its parameter inside a grid search. (3) Save and reload a fitted pipeline and verify the predictions match.

Full reference: https://scikit-learn.org/stable/api/index.html

## 5. Matplotlib

`import matplotlib.pyplot as plt` · written against Matplotlib 3.10. Every plot is a **Figure** (the canvas) holding one or more **Axes** (individual plots). Each Axes owns its lines, labels, ticks and legend. Use the explicit `fig, ax = plt.subplots()` style for anything you'll keep.

### 5.01 Setup, backends & the two interfaces

```python
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.__version__
# mpl.use("Agg")  -> headless servers/scripts that only save files (call before any figure is created)
# Jupyter: %matplotlib inline (static) · %matplotlib widget (interactive, needs ipympl) · %config InlineBackend.figure_format = "retina"
x = np.linspace(0, 2 * np.pi, 100)
plt.figure(figsize=(6, 3))                       # 1) implicit "pyplot" style: acts on the CURRENT axes
plt.plot(x, np.sin(x), label="sin"); plt.plot(x, np.cos(x), label="cos")
plt.title("Implicit API"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.grid(True)
plt.savefig("implicit.png"); plt.close()         # in scripts plt.show() opens a window
fig, ax = plt.subplots(figsize=(6, 3), layout="constrained")   # 2) explicit object-oriented style (recommended)
ax.plot(x, np.sin(x), label="sin"); ax.plot(x, np.cos(x), label="cos")
ax.set(title="Explicit API", xlabel="x", ylabel="y"); ax.legend(); ax.grid(True)
fig.savefig("explicit.png", dpi=150)
plt.gcf(), plt.gca(); plt.sca(ax)                # current figure / axes (bridge between the two styles)
plt.close(fig); plt.close("all")                 # free memory, especially in loops
```

| pyplot (implicit) | Axes method (explicit) |
|---|---|
| `plt.plot / scatter / bar / hist / imshow` | `ax.plot / scatter / bar / hist / imshow` |
| `plt.title`, `plt.xlabel`, `plt.ylabel` | `ax.set_title`, `ax.set_xlabel`, `ax.set_ylabel` |
| `plt.xlim`, `plt.ylim`, `plt.xscale` | `ax.set_xlim`, `ax.set_ylim`, `ax.set_xscale` |
| `plt.xticks(ticks, labels)` | `ax.set_xticks(ticks, labels)` |
| `plt.legend`, `plt.grid`, `plt.text` | `ax.legend`, `ax.grid`, `ax.text` |
| `plt.colorbar()` | `fig.colorbar(mappable, ax=ax)` |
| `plt.savefig` | `fig.savefig` |

**Drills:** (1) Draw the same 2-line plot in both styles. (2) Save it as PNG at 200 dpi and as PDF. (3) Create and close 50 figures in a loop without a memory warning.

### 5.02 Figures, subplots & layouts

```python
x = np.linspace(0, 2 * np.pi, 100)
fig, axes = plt.subplots(2, 3, figsize=(10, 5), sharex=True, sharey="row", layout="constrained")
axes.shape; axes[0, 1].set_title("row 0, col 1")
for i, ax in enumerate(axes.flat): ax.plot(np.random.default_rng(i).normal(size=20).cumsum())
for ax in axes.flat: ax.label_outer()            # hide inner tick labels on shared grids
fig.suptitle("Grid of subplots"); fig.supxlabel("step"); fig.supylabel("value")
fig, (ax1, ax2) = plt.subplots(1, 2, width_ratios=[3, 1], figsize=(8, 3))   # unpack directly
fig, axd = plt.subplot_mosaic([["main", "side"], ["main", "bottom"]], figsize=(8, 5), layout="constrained")
axd["main"].plot(x, np.sin(x)); axd["side"].hist(np.random.default_rng(0).normal(size=200)); axd["bottom"].bar([1, 2, 3], [3, 1, 2])
fig, axd = plt.subplot_mosaic("AAB;CCB", figsize=(8, 4))   # string shorthand; "." leaves a cell empty
fig = plt.figure(figsize=(8, 6))
gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
ax_top = fig.add_subplot(gs[0, :]); ax_left = fig.add_subplot(gs[1:, :2]); ax_right = fig.add_subplot(gs[1:, 2])
fig = plt.figure(); ax = fig.add_subplot(2, 2, 1); ax_polar = fig.add_subplot(222, projection="polar")
ax_free = fig.add_axes((0.55, 0.1, 0.4, 0.3))    # [left, bottom, width, height] in figure fractions
fig = plt.figure(figsize=(9, 3), layout="constrained")
left, right = fig.subfigures(1, 2, width_ratios=[2, 1]); left.subplots(1, 2); right.subplots(2, 1); left.suptitle("subfigure A")
fig, ax = plt.subplots(); ax.plot(x, np.sin(x))
ins = ax.inset_axes([0.6, 0.6, 0.35, 0.35]); ins.plot(x, np.sin(x)); ins.set_xlim(1, 2); ins.set_ylim(0.8, 1.05)
ax.indicate_inset_zoom(ins, edgecolor="k")       # zoom box + connector lines
ax_r = ax.twinx(); ax_r.plot(x, 100 * np.cos(x), color="C1"); ax_r.set_ylabel("cos × 100", color="C1")   # second y-axis
ax.secondary_xaxis("top", functions=(np.rad2deg, np.deg2rad)).set_xlabel("degrees")   # second scale for the same data
fig.subplots_adjust(left=0.1, right=0.85, wspace=0.3); fig.tight_layout(); fig.align_labels()
fig.savefig("layouts.png"); plt.close("all")
```

**Drills:** (1) Build a 2×2 grid with shared x-axes and a figure-level title. (2) Make a big main plot with a narrow histogram on the right using `subplot_mosaic`. (3) Plot loss and learning rate on twin y-axes.

### 5.03 Line & scatter plots

```python
rng = np.random.default_rng(0); x = np.linspace(0, 10, 50)
fig, ax = plt.subplots()
ax.plot(x, np.sin(x))                            # default style; colors cycle C0, C1, C2, ...
ax.plot(x, np.cos(x), "r--")                     # fmt string = color + linestyle + marker: "go-", "k:", "b^", "s--"
ax.plot(x, 0.5 * np.sin(x), color="tab:green", linestyle="-.", linewidth=2, marker="o", markersize=4,
        markerfacecolor="white", markevery=5, alpha=0.8, label="half", zorder=3)
ax.plot(x, np.column_stack([np.sin(x + p) for p in (1, 2)]))   # 2-D y: one line per column
line, = ax.plot(x, x / 10)                       # keep the Line2D handle to update it later
line.set_color("purple"); line.set_ydata(x / 20); line.set_label("updated")
ax.plot(rng.normal(size=30).cumsum(), drawstyle="steps-post")
fig, axs = plt.subplots(1, 3, figsize=(10, 3))
axs[0].semilogy(x, np.exp(x)); axs[1].semilogx(x + 1, x); axs[2].loglog(x + 1, (x + 1) ** 2)
fig, ax = plt.subplots()
pts = rng.normal(size=(200, 2)); vals = pts[:, 0] + pts[:, 1]
sc = ax.scatter(pts[:, 0], pts[:, 1], c=vals, cmap="viridis", s=20 + 30 * np.abs(vals), alpha=0.7, edgecolors="k", linewidths=0.3)
fig.colorbar(sc, ax=ax, label="x + y")           # c = numbers -> colormap + colorbar
classes = rng.integers(0, 3, 200)
sc2 = ax.scatter(pts[:, 0] + 6, pts[:, 1], c=classes, cmap="tab10", s=15, marker="^")
ax.legend(*sc2.legend_elements(), title="class", loc="upper right")   # legend for color-coded categories
fig, ax = plt.subplots()
for k in range(3):                               # alternative: one call per class gives automatic legend entries
    xk, yk = pts[classes == k].T
    ax.scatter(xk, yk, s=10, label=f"class {k}")
ax.legend(markerscale=2)
fig.savefig("lines_scatter.png"); plt.close("all")
```

**Drills:** (1) Plot sin, cos and tan (clipped to ±3) with different line styles and a legend. (2) Scatter 2-D points colored by a continuous value with a colorbar. (3) Scatter 3 classes with a legend using `legend_elements`.

### 5.04 Bars, histograms & distributions

```python
cats, vals = ["A", "B", "C", "D"], [23, 45, 12, 30]
fig, axs = plt.subplots(2, 3, figsize=(12, 6), layout="constrained")
bars = axs[0, 0].bar(cats, vals, color=["C0", "C1", "C2", "C3"], edgecolor="black", width=0.6)
axs[0, 0].bar_label(bars, fmt="%d", padding=2)   # value labels on top of the bars
axs[0, 1].barh(cats, vals, xerr=[2, 3, 1, 2], capsize=3)   # horizontal bars with error bars
xpos, w = np.arange(len(cats)), 0.38; a, b = np.array(vals), np.array(vals[::-1])
axs[0, 2].bar(xpos - w / 2, a, w, label="2023"); axs[0, 2].bar(xpos + w / 2, b, w, label="2024")   # grouped bars
axs[0, 2].set_xticks(xpos, cats); axs[0, 2].legend()
axs[1, 0].bar(cats, a, label="x"); axs[1, 0].bar(cats, b, bottom=a, label="y"); axs[1, 0].legend()   # stacked bars
data = np.random.default_rng(0).normal(size=1000)
counts, edges, patches = axs[1, 1].hist(data, bins=30, density=True, alpha=0.6, color="C0", edgecolor="white")
axs[1, 1].hist(data * 0.5 + 1, bins=edges, histtype="step", linewidth=2)   # reuse the bin edges to compare
axs[1, 2].hist([data, data + 2], bins=20, stacked=True, label=["a", "b"]); axs[1, 2].legend()
fig, axs = plt.subplots(1, 4, figsize=(14, 3), layout="constrained")
groups = [np.random.default_rng(i).normal(i, 1, 100) for i in range(3)]
axs[0].boxplot(groups, tick_labels=["g0", "g1", "g2"], showmeans=True, patch_artist=True, notch=True)   # tick_labels: 3.9+
axs[1].violinplot(groups, showmeans=True, showmedians=True)
axs[2].ecdf(data); axs[2].hist(data, bins=50, cumulative=True, density=True, histtype="step")   # empirical CDF two ways
xy = np.random.default_rng(1).multivariate_normal([0, 0], [[1, 0.8], [0.8, 1]], 5000)
axs[3].hexbin(xy[:, 0], xy[:, 1], gridsize=30, cmap="Blues", mincnt=1)   # density of many points
fig, ax = plt.subplots(); h = ax.hist2d(xy[:, 0], xy[:, 1], bins=40, cmap="magma"); fig.colorbar(h[3], ax=ax)
fig, ax = plt.subplots(); ax.boxplot(groups, orientation="horizontal")   # 3.10+ (older: vert=False)
fig.savefig("distributions.png"); plt.close("all")
```

**Drills:** (1) Draw grouped bars for train/val/test accuracy of 3 models with value labels. (2) Overlay two normalized histograms with shared bins. (3) Compare 4 distributions with box and violin plots side by side.

### 5.05 Areas, error bars, reference lines & spans

```python
x = np.linspace(0, 10, 100); y = np.sin(x); err = 0.2 + 0.1 * np.abs(np.cos(x))
fig, axs = plt.subplots(2, 2, figsize=(10, 6), layout="constrained")
ax = axs[0, 0]
ax.plot(x, y); ax.fill_between(x, y - err, y + err, alpha=0.25, label="± err")   # uncertainty band (mean ± std over seeds)
ax.fill_between(x, 0, y, where=y > 0, color="green", alpha=0.2, interpolate=True)
ax.axhline(0, color="k", linewidth=0.8); ax.axvline(np.pi, color="r", linestyle="--", label="π")
ax.axhspan(-0.25, 0.25, color="gray", alpha=0.1); ax.axvspan(6, 7, color="orange", alpha=0.15)   # shaded regions
ax.axline((0, 0), slope=0.1, color="purple", linestyle=":")   # infinite line through a point
ax.hlines([0.5, -0.5], xmin=0, xmax=2, colors="C2"); ax.vlines([8, 9], ymin=-1, ymax=1, colors="C3")
ax.fill_betweenx(np.linspace(-1, 1, 10), 9, 10, alpha=0.1)
ax.broken_barh([(1, 2), (4, 1)], (-1.2, 0.2), facecolors="tab:blue")   # Gantt-style segments
ax.eventplot([0.5, 2.5, 7.1], lineoffsets=-1.4, linelengths=0.2); ax.legend()
axs[0, 1].errorbar(x[::10], y[::10], yerr=err[::10], fmt="o", capsize=4, ecolor="gray", elinewidth=1)
axs[0, 1].errorbar(x[::10], y[::10] + 1, xerr=0.2, yerr=[err[::10] * 0.5, err[::10]], fmt="s-")   # asymmetric errors
axs[1, 0].stackplot(x, np.abs(y), np.abs(np.cos(x)), labels=["|sin|", "|cos|"], alpha=0.8); axs[1, 0].legend(loc="upper left")
axs[1, 1].step(x[::10], y[::10], where="mid"); axs[1, 1].stairs([3, 1, 4, 1], edges=[0, 1, 2, 3, 4], fill=True, alpha=0.3)
axs[1, 1].stem(x[::10], y[::10])
fig.savefig("areas.png"); plt.close("all")
```

**Drills:** (1) Plot the mean ± std of 5 noisy training runs as a line with a band. (2) Shade the epochs where the learning rate was warming up. (3) Mark the best epoch with a vertical line and a label.

### 5.06 Images, heatmaps, contours & vector fields

```python
img = np.random.default_rng(0).random((28, 28))
fig, axs = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
im = axs[0, 0].imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest"); fig.colorbar(im, ax=axs[0, 0])
rgb = np.random.default_rng(1).random((16, 16, 3)); axs[0, 1].imshow(rgb)   # (H, W, 3): floats in [0, 1] or uint8 in [0, 255]
axs[0, 1].imshow(img, extent=[0, 10, 0, 5], origin="lower", aspect="auto", alpha=0.3)   # map pixels onto data coordinates
M = np.corrcoef(np.random.default_rng(2).normal(size=(5, 50)))
axs[0, 2].matshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
X, Y = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
Z = np.exp(-(X ** 2 + Y ** 2)) - 0.5 * np.exp(-((X - 1) ** 2 + (Y - 1) ** 2))
pc = axs[1, 0].pcolormesh(X, Y, Z, cmap="viridis", shading="auto"); fig.colorbar(pc, ax=axs[1, 0])
cf = axs[1, 1].contourf(X, Y, Z, levels=20, cmap="coolwarm"); cs = axs[1, 1].contour(X, Y, Z, levels=8, colors="k", linewidths=0.5)
axs[1, 1].clabel(cs, inline=True, fontsize=7); fig.colorbar(cf, ax=axs[1, 1])
U, V = np.meshgrid(np.linspace(-2, 2, 15), np.linspace(-2, 2, 15)); dU, dV = -V, U
axs[1, 2].quiver(U, V, dU, dV, np.hypot(dU, dV), cmap="plasma"); axs[1, 2].streamplot(U, V, dU, dV, density=0.8, color="gray", linewidth=0.6)
fig, ax = plt.subplots(); ax.spy(np.random.default_rng(3).random((20, 20)) > 0.9)   # sparsity pattern of a matrix
fig.savefig("images.png"); plt.close("all")
```

**Drills:** (1) Show a 4×4 grid of random 28×28 "images" with no axes. (2) Plot a loss landscape `f(w1, w2)` with filled contours and labeled contour lines. (3) Show a correlation matrix with a diverging colormap centered at 0.

### 5.07 Pie, polar & 3D

```python
fig, axs = plt.subplots(1, 2, figsize=(9, 4))
sizes, labels = [45, 30, 15, 10], ["train", "val", "test", "holdout"]
axs[0].pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90, explode=[0.05, 0, 0, 0], counterclock=False)
axs[1].pie(sizes, labels=labels, wedgeprops={"width": 0.4, "edgecolor": "w"}); axs[1].set_title("donut")
fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
theta = np.linspace(0, 2 * np.pi, 200); ax.plot(theta, 1 + 0.5 * np.sin(5 * theta)); ax.set_rticks([0.5, 1, 1.5]); ax.set_theta_zero_location("N")
angles = np.linspace(0, 2 * np.pi, 5, endpoint=False); scores = np.array([0.8, 0.6, 0.9, 0.7, 0.5])
fig, ax = plt.subplots(subplot_kw={"projection": "polar"})   # radar chart: close the loop by repeating the first point
ax.plot(np.append(angles, angles[0]), np.append(scores, scores[0]), "o-"); ax.fill(np.append(angles, angles[0]), np.append(scores, scores[0]), alpha=0.25)
ax.set_xticks(angles, ["acc", "prec", "rec", "f1", "auc"])
fig = plt.figure(figsize=(10, 4))
ax3 = fig.add_subplot(1, 2, 1, projection="3d")
W1, W2 = np.meshgrid(np.linspace(-2, 2, 40), np.linspace(-2, 2, 40)); L = W1 ** 2 - W2 ** 2   # a saddle, loss-landscape style
ax3.plot_surface(W1, W2, L, cmap="viridis", edgecolor="none", alpha=0.9); ax3.contour(W1, W2, L, zdir="z", offset=-4, cmap="viridis")
ax3.set(xlabel="w1", ylabel="w2", zlabel="loss"); ax3.view_init(elev=30, azim=-60)
ax4 = fig.add_subplot(1, 2, 2, projection="3d")
p = np.random.default_rng(0).normal(size=(100, 3)); ax4.scatter(p[:, 0], p[:, 1], p[:, 2], c=p[:, 2], cmap="plasma")
ax4.plot_wireframe(W1, W2, L * 0.2, rstride=5, cstride=5, linewidth=0.5)
fig.savefig("3d.png"); plt.close("all")
```

**Drills:** (1) Draw a radar chart comparing two models on 5 metrics. (2) Plot the surface of a quadratic loss with its contours projected underneath. (3) Make a 3-D scatter of PCA embeddings colored by class.

### 5.08 Titles, labels, limits, scales & ticks

```python
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
x = np.linspace(0.01, 100, 400)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(x, np.log(x))
ax.set_title("Title", fontsize=14, fontweight="bold", loc="left", pad=10)
ax.set_xlabel("x (units)", fontsize=11); ax.set_ylabel("log(x)", labelpad=8)
ax.set_xlim(0, 100); ax.set_ylim(bottom=-5); ax.margins(x=0); ax.autoscale(axis="y", tight=True)
ax.set_xscale("linear"); ax.set_yscale("linear")   # also "log", "symlog", "logit"
ax.set_xticks([0, 50, 100], labels=["start", "mid", "end"]); ax.set_xticks(np.arange(0, 101, 10), minor=True)
ax.tick_params(axis="x", labelrotation=45, labelsize=8, direction="in", length=6); ax.tick_params(axis="y", colors="gray")
ax.xaxis.set_major_locator(mticker.MultipleLocator(20)); ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, integer=True))
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, pos: f"{v:.0f}%"))
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f")); ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.2f}"))
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0)); ax.yaxis.set_major_formatter(mticker.EngFormatter(unit="B"))
ax.grid(True, which="major", linestyle="--", alpha=0.5); ax.minorticks_on(); ax.grid(True, which="minor", alpha=0.15)
ax.spines[["top", "right"]].set_visible(False); ax.spines["left"].set_position(("outward", 5))
ax.invert_yaxis(); ax.invert_yaxis()             # flip an axis (rank plots, image coordinates)
ax.set_box_aspect(0.6); ax.set_facecolor("#fafafa")
fig, ax = plt.subplots(); ax.plot([1e5, 2e5, 3e5]); ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))   # ×10^5 offset label
ax.set_aspect("equal"); ax.axis("off"); ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])   # equal scales / hide everything / hide ticks
fig, ax = plt.subplots()
dates = np.arange("2024-01-01", "2024-07-01", dtype="datetime64[D]")
ax.plot(dates, np.random.default_rng(0).normal(size=len(dates)).cumsum())
ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y")); fig.autofmt_xdate()
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))   # compact automatic date labels
fig.savefig("ticks.png"); plt.close("all")
```

**Drills:** (1) Plot validation accuracy with a y-axis formatted as percentages. (2) Plot loss on a log y-axis with minor grid lines. (3) Plot a daily metric with month labels like "Jan 2024".

### 5.09 Legends, annotations, text & math

```python
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
x = np.linspace(0, 10, 200)
fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
ax.plot(x, np.sin(x), label="train"); ax.plot(x, np.sin(x) * 0.8, label="val")
ax.legend()                                      # picks up every artist's label=
ax.legend(loc="lower left", ncols=2, frameon=False, fontsize="small", title="split")
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))   # outside the axes
handles, labels = ax.get_legend_handles_labels(); ax.legend(handles[::-1], labels[::-1])   # reorder entries
custom = [Line2D([0], [0], color="k", ls="--", label="baseline"), Patch(facecolor="C2", alpha=0.3, label="region")]
first = ax.legend(loc="upper left"); ax.add_artist(first); ax.legend(handles=custom, loc="upper right")   # two legends
fig.legend(loc="outside upper center", ncols=2)  # figure-level legend placed outside (constrained layout)
ax.annotate("peak", xy=(np.pi / 2, 1), xytext=(3, 1.3), arrowprops=dict(arrowstyle="->", color="k"), fontsize=10)
ax.annotate("note", xy=(0.02, 0.95), xycoords="axes fraction", va="top")   # (0, 0) = bottom-left, (1, 1) = top-right of the axes
ax.annotate(f"{np.sin(5):.2f}", xy=(5, np.sin(5)), xytext=(10, -15), textcoords="offset points", arrowprops=dict(arrowstyle="-"))
ax.text(5, -1.2, "centered text", ha="center", va="bottom", fontsize=9, color="gray")
ax.text(0.98, 0.02, "corner", transform=ax.transAxes, ha="right", bbox=dict(boxstyle="round", fc="w", ec="0.7", alpha=0.9))
ax.set_title(r"$\mathcal{L}(\theta)=-\frac{1}{N}\sum_i y_i\log\hat{y}_i$")   # mathtext: a LaTeX subset, no TeX install needed
ax.set_xlabel(r"learning rate $\eta$ ($\times 10^{-3}$)"); ax.set_ylabel(r"$\Vert\nabla_\theta\Vert_2$")
fig.savefig("annotations.png")
plt.rcParams["text.usetex"]                      # set True to render ALL text with a real LaTeX install (matches papers)
plt.close("all")
```

**Drills:** (1) Annotate the minimum of a loss curve with an arrow and its value. (2) Put the legend outside the plot on the right without clipping it. (3) Write the softmax formula in a title with mathtext.

### 5.10 Colors, colormaps, norms & colorbars

```python
from matplotlib.colors import LogNorm, TwoSlopeNorm, CenteredNorm, BoundaryNorm, ListedColormap, LinearSegmentedColormap, to_rgba, to_hex
# color specs: "red" · "tab:blue" · "C0".."C9" (current cycle) · "#1f77b4" · (0.1, 0.2, 0.5) · (0.1, 0.2, 0.5, 0.3) RGBA · "0.6" (gray level)
to_rgba("C0"), to_hex((1, 0, 0)), mpl.colormaps["viridis"](0.5)
list(mpl.colormaps)[:5]
# sequential: viridis plasma inferno magma cividis · diverging: RdBu_r coolwarm bwr PiYG · qualitative: tab10 tab20 Set2 · cyclic: twilight hsv
cmap = mpl.colormaps["viridis"]; cmap.reversed(); mpl.colormaps["viridis_r"]; cmap.resampled(5)   # 5 discrete colors
colors = mpl.colormaps["plasma"](np.linspace(0, 1, 6))   # N colors sampled from a colormap (one per line)
ListedColormap(["#2b83ba", "#fdae61", "#d7191c"]); LinearSegmentedColormap.from_list("white_navy", ["white", "navy"])
data = np.random.default_rng(0).lognormal(size=(20, 20)); signed = np.random.default_rng(1).normal(size=(20, 20))
fig, axs = plt.subplots(1, 4, figsize=(14, 3), layout="constrained")
im0 = axs[0].imshow(data, cmap="viridis", norm=LogNorm(vmin=data.min(), vmax=data.max())); fig.colorbar(im0, ax=axs[0])   # log color scale
im1 = axs[1].imshow(signed, cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0)); fig.colorbar(im1, ax=axs[1])   # white exactly at 0
im2 = axs[2].imshow(signed, cmap="RdBu_r", norm=CenteredNorm()); cb = fig.colorbar(im2, ax=axs[2], shrink=0.8, label="z")
cb.set_ticks([-2, 0, 2]); cb.ax.tick_params(labelsize=7)
im3 = axs[3].imshow(signed, cmap=mpl.colormaps["RdBu_r"].resampled(4), norm=BoundaryNorm([-3, -1, 0, 1, 3], 4))   # discrete levels
fig.colorbar(im3, ax=axs[3], orientation="horizontal", extend="both")
fig, axs = plt.subplots(1, 2, layout="constrained")
ims = [a.imshow(np.random.default_rng(i).random((5, 5)), vmin=0, vmax=1) for i, a in enumerate(axs)]
fig.colorbar(ims[0], ax=axs, shrink=0.6)         # ONE shared colorbar: use the same vmin/vmax on every panel
fig, ax = plt.subplots()
for i, c in enumerate(colors): ax.plot(np.arange(10) * (i + 1), color=c)
ax.set_prop_cycle(color=["k", "r", "b"])         # custom color cycle for later plots on this Axes
fig.savefig("colors.png"); plt.close("all")
```

**Drills:** (1) Plot 8 learning-rate runs colored by a sequential colormap. (2) Show a weight-update matrix with a diverging colormap centered at 0. (3) Share one colorbar across a 2×2 grid of heatmaps.

### 5.11 Styles & rcParams

```python
plt.style.available[:5]                          # built-in styles
plt.style.use("ggplot"); plt.style.use("default")   # also "seaborn-v0_8-whitegrid", "fivethirtyeight", "bmh", "dark_background", "tableau-colorblind10"
with plt.style.context("seaborn-v0_8-darkgrid"): fig, ax = plt.subplots(); ax.plot([1, 3, 2])
plt.rcParams["font.size"] = 11; plt.rcParams["axes.spines.top"] = False; plt.rcParams["figure.dpi"] = 110
mpl.rcParams.update({"axes.titlesize": 13, "axes.labelsize": 11, "legend.frameon": False, "lines.linewidth": 2,
                     "axes.grid": True, "grid.alpha": 0.3, "figure.figsize": (6, 4), "savefig.bbox": "tight", "savefig.dpi": 200})
plt.rc("axes", prop_cycle=plt.cycler(color=["#0072B2", "#E69F00", "#009E73", "#D55E00"]))   # colorblind-friendly cycle
with plt.rc_context({"font.family": "serif", "mathtext.fontset": "cm"}): fig, ax = plt.subplots(); ax.set_title(r"$\alpha$ paper style")
with plt.xkcd(): fig, ax = plt.subplots(); ax.plot([1, 2, 1])
mpl.rcdefaults()                                 # reset everything to defaults
mpl.get_configdir(), mpl.matplotlib_fname()      # where a persistent matplotlibrc lives; custom file: plt.style.use("./paper.mplstyle")
plt.close("all")
```

**Drills:** (1) Make a `paper.mplstyle` with serif fonts and no top/right spines, then apply it. (2) Temporarily switch styles for one figure only. (3) Set a colorblind-safe color cycle globally.

### 5.12 Saving & exporting

```python
fig, ax = plt.subplots(figsize=(6, 4))           # inches × dpi = pixels: 6×4 in at 100 dpi = 600×400 px
ax.plot([1, 2, 3], [2, 1, 3])
fig.savefig("plot.png", dpi=300, bbox_inches="tight", pad_inches=0.05)   # raster: slides, web
fig.savefig("plot.pdf"); fig.savefig("plot.svg") # vector: papers (\includegraphics{plot.pdf}) and websites
fig.savefig("plot_transparent.png", transparent=True); fig.savefig("plot_white.png", facecolor="white")
fig.savefig("plot.jpg", pil_kwargs={"quality": 90})
fig.set_size_inches(8, 3); fig.get_size_inches(), fig.dpi
import io
buf = io.BytesIO(); fig.savefig(buf, format="png"); buf.getvalue()[:8]   # in-memory PNG (web apps, loggers)
fig.canvas.draw(); rgba = np.asarray(fig.canvas.buffer_rgba()); rgba.shape   # figure -> NumPy image (e.g. for TensorBoard)
from matplotlib.backends.backend_pdf import PdfPages
with PdfPages("report.pdf") as pdf:              # multi-page PDF report
    for k in range(3):
        f, a = plt.subplots(); a.plot(np.arange(10) ** k); a.set_title(f"page {k}"); pdf.savefig(f); plt.close(f)
plt.imsave("array.png", np.random.default_rng(0).random((32, 32)), cmap="gray"); plt.imread("array.png").shape
plt.close("all")                                 # in loops: create -> save -> close
```

**Drills:** (1) Export the same figure at 72 and 300 dpi and compare file sizes. (2) Write a 3-page PDF report with one plot per page. (3) Convert a figure to a NumPy RGBA array.

### 5.13 Patches, collections & coordinate systems

```python
from matplotlib.patches import Rectangle, Circle, Ellipse, Polygon, FancyBboxPatch, FancyArrowPatch, Wedge, Arc
from matplotlib.collections import LineCollection, PatchCollection
import matplotlib.transforms as mtransforms
fig, ax = plt.subplots(figsize=(6, 6))
ax.add_patch(Rectangle((0.1, 0.1), 0.3, 0.2, fill=False, edgecolor="red", linewidth=2))   # (x, y) = lower-left corner
ax.add_patch(Circle((0.7, 0.7), 0.1, color="C0", alpha=0.5)); ax.add_patch(Ellipse((0.3, 0.7), 0.3, 0.1, angle=30, color="C1"))
ax.add_patch(Polygon([[0.6, 0.1], [0.9, 0.1], [0.75, 0.35]], closed=True, color="C2")); ax.add_patch(Wedge((0.5, 0.5), 0.1, 0, 120, color="C3"))
ax.add_patch(FancyBboxPatch((0.4, 0.85), 0.2, 0.08, boxstyle="round,pad=0.02", fc="lightyellow", ec="k"))
ax.add_patch(FancyArrowPatch((0.2, 0.45), (0.5, 0.45), arrowstyle="-|>", mutation_scale=15)); ax.add_patch(Arc((0.5, 0.2), 0.2, 0.2, theta1=0, theta2=180))
segs = [[(0, i / 10), (1, i / 10 + 0.05)] for i in range(10)]
ax.add_collection(LineCollection(segs, colors=mpl.colormaps["viridis"](np.linspace(0, 1, 10)), linewidths=0.8, alpha=0.4))   # many lines, fast
ax.add_collection(PatchCollection([Circle((0.9, yy), 0.02) for yy in np.linspace(0.1, 0.9, 5)], facecolor="k"))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
ax.text(0.5, 1.02, "axes coords", transform=ax.transAxes, ha="center")   # transData (default) · transAxes (0-1 of axes) · fig.transFigure
ax.plot([0, 1], [0.5, 0.5], transform=ax.transAxes, ls=":", color="gray")
trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)   # x in data units, y in axes units
ax.add_patch(Rectangle((0.2, 0), 0.1, 1, transform=trans, color="gray", alpha=0.1))
fig.text(0.01, 0.01, "figure footnote", fontsize=8)
fig.savefig("patches.png")
img = np.random.default_rng(0).random((100, 150, 3))   # object-detection boxes over an image
boxes = [(10, 20, 50, 40, "cat", 0.92), (80, 30, 60, 55, "dog", 0.71)]   # x, y, w, h, label, score
fig, ax = plt.subplots(); ax.imshow(img)
for bx, by, bw, bh, lbl, score in boxes:
    ax.add_patch(Rectangle((bx, by), bw, bh, fill=False, edgecolor="lime", linewidth=2))
    ax.text(bx, by - 2, f"{lbl} {score:.2f}", color="black", fontsize=8, bbox=dict(facecolor="lime", pad=1, edgecolor="none"))
ax.axis("off"); fig.savefig("boxes.png"); plt.close("all")
```

**Drills:** (1) Draw predicted vs ground-truth bounding boxes in two colors on an image. (2) Draw a simple neural-net diagram with circles and arrows. (3) Shade a vertical band spanning the full axes height with a blended transform.

### 5.14 Animation & live-updating plots

```python
from matplotlib.animation import FuncAnimation, ArtistAnimation, PillowWriter
fig, ax = plt.subplots()
x = np.linspace(0, 2 * np.pi, 200); line, = ax.plot(x, np.sin(x)); ax.set_ylim(-1.2, 1.2)
def update(frame):
    line.set_ydata(np.sin(x + frame / 10))       # mutate existing artists instead of re-plotting
    return (line,)
anim = FuncAnimation(fig, update, frames=30, interval=50, blit=True)
anim.save("wave.gif", writer=PillowWriter(fps=20))   # .mp4 needs ffmpeg: anim.save("wave.mp4", fps=30)
# Jupyter: from IPython.display import HTML; HTML(anim.to_jshtml())
loss_fn = lambda w0, w1: (w0 - 1) ** 2 + 5 * (w1 + 0.5) ** 2   # gradient descent path on a contour plot
path = [np.array([-2.0, 1.5])]
for _ in range(30): path.append(path[-1] - 0.08 * np.array([2 * (path[-1][0] - 1), 10 * (path[-1][1] + 0.5)]))
path = np.array(path)
W0, W1 = np.meshgrid(np.linspace(-2.5, 2.5, 100), np.linspace(-1.5, 2, 100))
fig, ax = plt.subplots(); ax.contour(W0, W1, loss_fn(W0, W1), levels=20, cmap="viridis")
pt, = ax.plot([], [], "ro-", ms=3)
def step(i):
    pt.set_data(path[:i + 1, 0], path[:i + 1, 1])
    return (pt,)
FuncAnimation(fig, step, frames=len(path), interval=100).save("gd.gif", writer=PillowWriter(fps=10))
fig, ax = plt.subplots()
frames = [[ax.imshow(np.random.default_rng(i).random((8, 8)), animated=True)] for i in range(5)]
ArtistAnimation(fig, frames, interval=200).save("frames.gif", writer=PillowWriter(fps=5))   # pre-rendered frames
plt.ion()                                        # live training curve (scripts; in Jupyter use clear_output(wait=True) + display(fig))
fig, ax = plt.subplots(); (train_line,) = ax.plot([], [], label="train"); ax.legend()
losses = []
for it in range(10):
    losses.append(np.exp(-it / 5) + 0.05 * np.random.default_rng(it).random())
    train_line.set_data(range(len(losses)), losses); ax.relim(); ax.autoscale_view()
    fig.canvas.draw_idle(); plt.pause(0.001)
plt.ioff(); plt.close("all")
```

**Drills:** (1) Animate gradient descent on a 2-D quadratic and save it as a GIF. (2) Animate how a decision boundary changes over training epochs. (3) Build a live loss plot that updates every 10 steps.

### 5.15 ML visualization recipes

```python
rng = np.random.default_rng(0)
# 1) training curves: log-scale loss, best epoch marked
epochs = np.arange(1, 51); train_loss = 2.0 * np.exp(-epochs / 10) + 0.05 * rng.random(50)
val_loss = train_loss + 0.1 + 0.002 * np.clip(epochs - 25, 0, None) ** 1.5
fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
ax.plot(epochs, train_loss, label="train"); ax.plot(epochs, val_loss, label="val"); ax.set_yscale("log")
best = val_loss.argmin(); ax.axvline(epochs[best], ls="--", c="gray")
ax.scatter(epochs[best], val_loss[best], c="red", zorder=3, label=f"best @ {epochs[best]}"); ax.set(xlabel="epoch", ylabel="loss"); ax.legend()
raw = np.exp(-np.arange(500) / 150) + 0.1 * rng.normal(size=500); ema = np.zeros_like(raw); ema[0] = raw[0]
for i in range(1, len(raw)): ema[i] = 0.9 * ema[i - 1] + 0.1 * raw[i]   # EMA smoothing like TensorBoard's slider
fig, ax = plt.subplots(); ax.plot(raw, alpha=0.3, color="C0"); ax.plot(ema, color="C0", label="EMA 0.9"); ax.legend()
# 2) confusion matrix with counts in each cell
cm = np.array([[50, 2, 3], [5, 40, 5], [0, 4, 46]]); names = ["cat", "dog", "bird"]
fig, ax = plt.subplots(figsize=(4.5, 4), layout="constrained"); im = ax.imshow(cm, cmap="Blues"); fig.colorbar(im, ax=ax)
ax.set_xticks(range(3), names); ax.set_yticks(range(3), names); ax.set(xlabel="predicted", ylabel="true")
for i in range(3):
    for j in range(3):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
# 3) decision boundary of any 2-D classifier
from sklearn.datasets import make_moons
from sklearn.neighbors import KNeighborsClassifier
Xm, ym = make_moons(300, noise=0.25, random_state=0); clf = KNeighborsClassifier(15).fit(Xm, ym)
xx, yy = np.meshgrid(np.linspace(Xm[:, 0].min() - 0.5, Xm[:, 0].max() + 0.5, 300), np.linspace(Xm[:, 1].min() - 0.5, Xm[:, 1].max() + 0.5, 300))
proba = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
fig, ax = plt.subplots(); ax.contourf(xx, yy, proba, levels=20, cmap="RdBu_r", alpha=0.6); ax.contour(xx, yy, proba, levels=[0.5], colors="k")
ax.scatter(Xm[:, 0], Xm[:, 1], c=ym, cmap="RdBu_r", edgecolors="k", s=15)
# 4) image grid with predicted (true) labels; green = correct
imgs = rng.random((16, 28, 28)); y_true = rng.integers(0, 10, 16); y_pred = np.where(rng.random(16) < 0.8, y_true, (y_true + 1) % 10)
fig, axes = plt.subplots(2, 8, figsize=(12, 3.5))
for ax, im_, t, p in zip(axes.flat, imgs, y_true, y_pred):
    ax.imshow(im_, cmap="gray"); ax.set_title(f"{p} ({t})", color="green" if p == t else "red", fontsize=9); ax.axis("off")
# PyTorch images are CHW: show with img.permute(1, 2, 0).cpu().numpy() (undo normalization first)
# 5) attention heatmap with token labels
tokens = ["The", "cat", "sat", "on", "the", "mat"]; att = np.tril(rng.random((6, 6))); att /= att.sum(1, keepdims=True)
fig, ax = plt.subplots(figsize=(5, 4.5), layout="constrained"); im = ax.imshow(att, cmap="viridis"); fig.colorbar(im, ax=ax)
ax.set_xticks(range(6), tokens, rotation=45, ha="right"); ax.set_yticks(range(6), tokens); ax.set(xlabel="key", ylabel="query")
# 6) 2-D embedding scatter colored by class (after PCA / t-SNE / UMAP)
emb = np.concatenate([rng.normal(c, 0.6, size=(100, 2)) for c in [(0, 0), (3, 1), (1, 4)]]); lab = np.repeat([0, 1, 2], 100)
fig, ax = plt.subplots(); sc = ax.scatter(emb[:, 0], emb[:, 1], c=lab, cmap="tab10", s=8, alpha=0.8)
ax.legend(*sc.legend_elements(), title="class"); ax.set_xticks([]); ax.set_yticks([])
# 7) weight / gradient histograms per layer (spot dead or exploding layers)
layer_w = {f"layer{i}": rng.normal(0, 1 / (i + 1), 5000) for i in range(4)}
fig, axes = plt.subplots(1, 4, figsize=(12, 2.5), sharey=True, layout="constrained")
for ax, (name, wv) in zip(axes, layer_w.items()): ax.hist(wv, bins=60); ax.set_title(f"{name} σ={wv.std():.2f}", fontsize=9)
# 8) learning-rate schedule: warmup + cosine
steps = np.arange(1000); warm = 100
lr = np.where(steps < warm, steps / warm, 0.5 * (1 + np.cos(np.pi * (steps - warm) / (1000 - warm)))) * 3e-4
fig, ax = plt.subplots(figsize=(6, 2.5)); ax.plot(steps, lr); ax.set(xlabel="step", ylabel="lr"); ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
# 9) ROC curves for several models + chance diagonal
fig, ax = plt.subplots(figsize=(4.5, 4.5)); fpr = np.linspace(0, 1, 100)
for name, sharp in [("model A", 3), ("model B", 1.5)]: ax.plot(fpr, fpr ** (1 / sharp), label=f"{name} (AUC={sharp / (sharp + 1):.2f})")
ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set(xlabel="FPR", ylabel="TPR", aspect="equal"); ax.legend(loc="lower right")
# 10) sorted feature importances
feats = np.array([f"f{i}" for i in range(10)]); imp = rng.random(10); order = np.argsort(imp)
fig, ax = plt.subplots(figsize=(5, 4)); ax.barh(feats[order], imp[order], color="C2"); ax.set_xlabel("importance")
# 11) regression diagnostics: predicted vs actual, residuals
y_act = rng.normal(size=200); y_hat = y_act + rng.normal(0, 0.3, 200)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
a1.scatter(y_act, y_hat, s=8, alpha=0.6); lims = [y_act.min(), y_act.max()]; a1.plot(lims, lims, "k--"); a1.set(xlabel="actual", ylabel="predicted")
a2.scatter(y_hat, y_hat - y_act, s=8, alpha=0.6); a2.axhline(0, color="k"); a2.set(xlabel="predicted", ylabel="residual")
# 12) correlation heatmap with the upper triangle masked
corr = np.corrcoef(rng.normal(size=(6, 100)) + rng.normal(size=100) * 0.5)
masked = np.ma.masked_where(np.triu(np.ones_like(corr, dtype=bool), k=1), corr)
fig, ax = plt.subplots(); im = ax.imshow(masked, cmap="RdBu_r", vmin=-1, vmax=1); fig.colorbar(im, ax=ax)
# 13) precision & recall vs threshold (choose an operating point)
thr = np.linspace(0, 1, 101); precision = 0.5 + 0.5 * thr ** 0.5; recall = 1 - thr ** 2
fig, ax = plt.subplots(); ax.plot(thr, precision, label="precision"); ax.plot(thr, recall, label="recall"); ax.axvline(0.5, ls=":", c="gray"); ax.legend()
# 14) hyperparameter sweep heatmap with annotated cells
lrs, wds = np.logspace(-4, -2, 5), np.logspace(-5, -3, 4); acc = rng.uniform(0.7, 0.95, (4, 5))
fig, ax = plt.subplots(); im = ax.imshow(acc, cmap="viridis", origin="lower")
ax.set_xticks(range(5), [f"{v:.0e}" for v in lrs]); ax.set_yticks(range(4), [f"{v:.0e}" for v in wds]); ax.set(xlabel="lr", ylabel="weight decay")
for (i, j), val in np.ndenumerate(acc): ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color="w")
fig.savefig("sweep.png"); plt.close("all")
```

**Drills:** Rebuild each of the 14 recipes from a blank cell, timing yourself. Then do them again using real outputs: your own training logs, a real classifier's confusion matrix, and attention weights from 3.08.

Full reference: https://matplotlib.org/stable/api/index.html

## 6. Hugging Face

`pip install transformers datasets accelerate peft trl evaluate safetensors sentence-transformers huggingface_hub` · written against Transformers 5.x and TRL 1.x. v5 changes you'll see in old tutorials:
- `from_pretrained(dtype=...)` replaces `torch_dtype=`.
- `Trainer(processing_class=tok)` replaces `tokenizer=`.
- `eval_strategy` replaces `evaluation_strategy`.
- TensorFlow and Flax support is gone.
- The pipeline task list no longer includes summarization, translation or extractive QA; use `text-generation` with an instruct model.

Model ids below are real public checkpoints, so every snippet downloads weights the first time it runs.

### 6.01 The ecosystem at a glance

```python
import torch
import transformers
transformers.__version__                         # '5.x'
# transformers        models, tokenizers, processors, pipelines, Trainer, generate()
# datasets            load / process / stream datasets (Arrow-backed, memory-mapped)
# tokenizers          fast Rust tokenizers (used under the hood)
# huggingface_hub     download/upload files, repos, the `hf` CLI
# accelerate          one training script for CPU / GPU / multi-GPU / TPU
# peft                parameter-efficient fine-tuning (LoRA, QLoRA, ...)
# trl                 post-training: SFT, DPO, GRPO, reward modeling
# evaluate            metrics          · safetensors  safe, fast weight files
# sentence-transformers  embeddings & semantic search   · diffusers  image/video diffusion pipelines
```

### 6.02 The Hub: auth, download, upload & cache

```bash
hf auth login                                    # token from huggingface.co/settings/tokens (older CLI: huggingface-cli login)
hf auth whoami
hf download openai-community/gpt2 config.json    # one file -> prints its cached path
hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir ./qwen   # a whole repo into a folder
hf upload my-user/my-model ./outputs .           # upload a folder to the repo root
export HF_HOME=/data/hf                          # move the cache (models + datasets) to a big disk
export HF_TOKEN=hf_xxx                           # auth in CI / scripts
export HF_HUB_OFFLINE=1                          # use only what's already cached
```

```python
from huggingface_hub import login, hf_hub_download, snapshot_download, HfApi, create_repo, upload_folder, scan_cache_dir
login()                                          # interactive; or login(token="hf_...")
cfg_path = hf_hub_download(repo_id="openai-community/gpt2", filename="config.json")
local_dir = snapshot_download("openai-community/gpt2", allow_patterns=["*.json", "*.safetensors"], revision="main")
api = HfApi(); api.whoami()
[m.id for m in api.list_models(search="bert", sort="downloads", limit=5)]
create_repo("my-user/my-model", private=True, exist_ok=True)
upload_folder(repo_id="my-user/my-model", folder_path="./outputs", commit_message="add weights")
api.upload_file(path_or_fileobj="README.md", path_in_repo="README.md", repo_id="my-user/my-model")
scan_cache_dir().size_on_disk / 1e9              # GB used by the local cache
# model.push_to_hub("my-user/my-model") · tokenizer.push_to_hub(...) · dataset.push_to_hub(...)
```

**Drills:** (1) Download only `config.json` of a model and print its `hidden_size`. (2) Create a private repo and upload a file to it. (3) Find out how much disk your HF cache uses.

### 6.03 Pipelines

```python
from transformers import pipeline
clf = pipeline("text-classification", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
clf("I love this course!"); clf(["great", "awful"], top_k=None)   # top_k=None -> scores for every label
ner = pipeline("token-classification", model="dslim/bert-base-NER", aggregation_strategy="simple")   # merges word pieces into entities
ner("Ada Lovelace worked in London.")
fill = pipeline("fill-mask", model="google-bert/bert-base-uncased"); fill("Paris is the [MASK] of France.", top_k=3)
zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
zs("The GPU ran out of memory", candidate_labels=["hardware", "cooking", "sports"])
gen = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct", dtype="auto", device_map="auto")
gen("Once upon a time", max_new_tokens=40, do_sample=True, temperature=0.7, return_full_text=False)
gen([{"role": "user", "content": "Summarize in one sentence: tensors are n-dimensional arrays..."}], max_new_tokens=60)   # chat input
emb = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2"); emb("hello")   # nested list (1, T, 384)
asr = pipeline("automatic-speech-recognition", model="openai/whisper-small")   # asr("clip.wav")
img_clf = pipeline("image-classification", model="google/vit-base-patch16-224")   # img_clf("cat.jpg")
det = pipeline("object-detection", model="facebook/detr-resnet-50")   # det("street.jpg") -> boxes, labels, scores
zsi = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")   # zsi("cat.jpg", candidate_labels=["cat", "dog"])
depth = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
vlm = pipeline("image-text-to-text", model="HuggingFaceTB/SmolVLM-256M-Instruct")
# vlm(text=[{"role": "user", "content": [{"type": "image", "url": "cat.jpg"}, {"type": "text", "text": "Describe the image."}]}])
pipeline("text-classification", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", device=0, batch_size=32)   # GPU 0, batched
```

Tasks in v5: text-classification · token-classification · fill-mask · text-generation · zero-shot-classification · feature-extraction · table- and document-question-answering · automatic-speech-recognition · audio-classification · text-to-audio · zero-shot-audio-classification · image-classification · image-feature-extraction · image-segmentation · object-detection · zero-shot-image-classification · zero-shot-object-detection · depth-estimation · mask-generation · image-text-to-text · video-classification · keypoint-matching · any-to-any.

**Drills:** (1) Classify the sentiment of 5 sentences in one call. (2) Extract the people and places from a paragraph. (3) Use zero-shot classification to route support tickets into 4 categories. (4) Generate a reply from a chat message list.

### 6.04 Tokenizers & chat templates

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
enc = tok("Hello world!", "A second sentence.", padding="max_length", truncation=True, max_length=16, return_tensors="pt")
enc.keys()                                       # input_ids, token_type_ids, attention_mask
tok.tokenize("Tokenization isn't trivial")       # ['token', '##ization', 'isn', "'", 't', 'trivial']
ids = tok.encode("hello world"); tok.decode(ids); tok.decode(ids, skip_special_tokens=True)
tok.convert_tokens_to_ids(["hello"]); tok.convert_ids_to_tokens(ids)
batch = tok(["short", "a much longer sentence here"], padding=True, truncation=True, return_tensors="pt")   # pad to the longest
batch["attention_mask"]                          # 1 = real token, 0 = padding
tok.batch_decode(batch["input_ids"], skip_special_tokens=True)
tok.vocab_size, len(tok), tok.model_max_length
tok.pad_token, tok.cls_token, tok.sep_token, tok.mask_token, tok.unk_token, tok.all_special_tokens
tok("New York", return_offsets_mapping=True)["offset_mapping"]   # character spans per token
words = ["Hugging", "Face", "rocks"]; tok(words, is_split_into_words=True).word_ids()   # token -> word index (label alignment for NER)
long_text = "a long document " * 500
windows = tok(long_text, truncation=True, max_length=128, stride=32, return_overflowing_tokens=True)   # overlapping chunks
len(windows["input_ids"])
tok.add_tokens(["<ROBOT>"]); tok.add_special_tokens({"additional_special_tokens": ["<|tool|>"]})
# after adding tokens: model.resize_token_embeddings(len(tok))
lm_tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")   # decoder-only (GPT-style) model
lm_tok.padding_side = "left"                     # left padding for BATCHED generation
if lm_tok.pad_token is None: lm_tok.pad_token = lm_tok.eos_token
messages = [{"role": "system", "content": "You are concise."}, {"role": "user", "content": "What is a tensor?"}]
prompt = lm_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)   # the model's own chat format as a string
inputs = lm_tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
tok.save_pretrained("./my_tokenizer")
new_tok = tok.train_new_from_iterator((line for line in ["your corpus text"] * 100), vocab_size=500)   # same algorithm, new vocab
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
bpe = Tokenizer(models.BPE(unk_token="[UNK]")); bpe.pre_tokenizer = pre_tokenizers.Whitespace()
bpe.train_from_iterator(["some text here", "more text"], trainer=trainers.BpeTrainer(vocab_size=200, special_tokens=["[UNK]", "[PAD]"]))
bpe.encode("some text").tokens
```

**Watch out:** Different models use different tokenizers. Always load the tokenizer from the same checkpoint as the model, and build prompts with `apply_chat_template` instead of hand-writing special tokens.

**Drills:** (1) Tokenize a batch with padding and explain every entry of `attention_mask`. (2) Align word-level NER labels to sub-word tokens with `word_ids()`. (3) Print the exact chat-formatted prompt string for a 3-turn conversation. (4) Train a 1k-vocab BPE tokenizer on a text file.

### 6.05 Models & configs

```python
from transformers import (AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoModelForCausalLM, AutoModelForTokenClassification,
                          AutoModelForMaskedLM, AutoModelForImageClassification, AutoProcessor, BitsAndBytesConfig)
bert = AutoModel.from_pretrained("google-bert/bert-base-uncased")   # backbone only -> hidden states
out = bert(**tok("hello world", return_tensors="pt"))
out.last_hidden_state.shape, out.pooler_output.shape   # (1, T, 768), (1, 768)
clf_model = AutoModelForSequenceClassification.from_pretrained(
    "google-bert/bert-base-uncased", num_labels=3, id2label={0: "neg", 1: "neu", 2: "pos"}, label2id={"neg": 0, "neu": 1, "pos": 2})   # new, randomly initialized head
o = clf_model(**tok(["good", "bad"], padding=True, return_tensors="pt"), labels=torch.tensor([2, 0]))
o.loss, o.logits.shape                           # passing labels= makes the model compute the loss
lm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", dtype="auto", device_map="auto", attn_implementation="sdpa")
lm.config.hidden_size, lm.num_parameters(), lm.dtype, lm.device, lm.generation_config
cfg = AutoConfig.from_pretrained("google-bert/bert-base-uncased", num_hidden_layers=4)
small = AutoModel.from_config(cfg)               # same architecture, fewer layers, RANDOM weights
eager = AutoModel.from_pretrained("google-bert/bert-base-uncased", attn_implementation="eager")   # eager = attention weights can be returned
o = eager(**tok("hello world", return_tensors="pt"), output_hidden_states=True, output_attentions=True)
len(o.hidden_states), o.attentions[0].shape      # 13 (embeddings + 12 layers), (B, heads, T, T)
for p in clf_model.base_model.parameters(): p.requires_grad = False   # freeze the backbone, train only the head
clf_model.save_pretrained("./my_model"); tok.save_pretrained("./my_model")   # safetensors + config.json + tokenizer files
AutoModelForSequenceClassification.from_pretrained("./my_model")
clf_model.eval(); clf_model.train(); clf_model.gradient_checkpointing_enable(); clf_model.get_input_embeddings(); clf_model.config.to_dict()
q4 = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
# AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", quantization_config=q4, device_map="auto")   # 4-bit, GPU + bitsandbytes
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")   # text + image preprocessing in one object
# other heads: AutoModelForTokenClassification · AutoModelForMaskedLM · AutoModelForImageClassification · AutoModelForSpeechSeq2Seq
# from_pretrained options: revision="main" · cache_dir=... · local_files_only=True · token=... · trust_remote_code=True (runs repo code: read it first)
```

**Drills:** (1) Load a classifier with your own `id2label` and run one forward pass with labels. (2) Count the parameters of BERT-base and of a 4-layer config of it. (3) Freeze everything except the classification head and count the trainable parameters.

### 6.06 Text generation

```python
from transformers import GenerationConfig, TextStreamer
inputs = lm_tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(lm.device)
out = lm.generate(**inputs, max_new_tokens=64)   # greedy decoding
new_tokens = out[:, inputs["input_ids"].shape[1]:]   # generate() returns prompt + continuation: slice the prompt off
lm_tok.batch_decode(new_tokens, skip_special_tokens=True)[0]
lm.generate(**inputs, max_new_tokens=64, do_sample=True, temperature=0.7, top_p=0.9, top_k=50)   # sampling
lm.generate(**inputs, max_new_tokens=64, num_beams=4, early_stopping=True, no_repeat_ngram_size=3)   # beam search
lm.generate(**inputs, max_new_tokens=64, do_sample=True, num_return_sequences=3, repetition_penalty=1.1, min_new_tokens=5)
lm.generate(**inputs, max_new_tokens=64, eos_token_id=lm_tok.eos_token_id, pad_token_id=lm_tok.pad_token_id)
lm.generate(**inputs, max_new_tokens=64, stop_strings=["\n\n"], tokenizer=lm_tok)   # stop on a string
res = lm.generate(**inputs, max_new_tokens=20, return_dict_in_generate=True, output_scores=True)
res.sequences.shape, len(res.scores)             # one (B, V) score tensor per generated token
lm.compute_transition_scores(res.sequences, res.scores, normalize_logits=True)   # log-prob of each chosen token
lm.generate(**inputs, max_new_tokens=64, streamer=TextStreamer(lm_tok, skip_prompt=True))   # print tokens as they arrive
gc = GenerationConfig(max_new_tokens=128, do_sample=True, temperature=0.8); lm.generate(**inputs, generation_config=gc)
chats = [[{"role": "user", "content": q}] for q in ["Hi!", "What is 2+2?"]]
batch_in = lm_tok.apply_chat_template(chats, add_generation_prompt=True, padding=True, return_tensors="pt", return_dict=True).to(lm.device)
lm.generate(**batch_in, max_new_tokens=32)       # batched generation (left padding!)
ids, past = inputs["input_ids"], None            # what generate() does: greedy loop with a KV cache (see 3.11)
with torch.no_grad():
    for _ in range(10):
        step = lm(input_ids=ids if past is None else ids[:, -1:], past_key_values=past, use_cache=True)
        past = step.past_key_values
        ids = torch.cat([ids, step.logits[:, -1].argmax(-1, keepdim=True)], dim=1)
```

**Drills:** (1) Generate 3 samples at temperatures 0.2, 0.7 and 1.2 and compare them. (2) Return only the new tokens for a batch of 4 prompts. (3) Compute the average log-prob of a generated answer with `compute_transition_scores`.

### 6.07 Datasets

```python
from datasets import load_dataset, Dataset, DatasetDict, ClassLabel, Audio, Image, concatenate_datasets, interleave_datasets, load_from_disk
ds = load_dataset("stanfordnlp/imdb")            # DatasetDict with "train", "test", "unsupervised" splits
ds["train"].features, ds["train"].column_names, ds["train"].num_rows
ds["train"][0]; ds["train"][:3]; ds["train"]["text"][:2]   # one row (dict) / a slice (dict of lists) / column values
train = load_dataset("stanfordnlp/imdb", split="train[:10%]")   # split slicing: "train[:1000]", "train[10%:20%]", "train+test"
csv_ds = load_dataset("csv", data_files={"train": "train.csv", "test": "test.csv"})
json_ds = load_dataset("json", data_files="data.jsonl"); pq_ds = load_dataset("parquet", data_files="data/*.parquet")
local_audio = load_dataset("audiofolder", data_dir="./clips"); local_imgs = load_dataset("imagefolder", data_dir="./images")   # label = subfolder name
stream = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True)   # IterableDataset: nothing fully downloaded
next(iter(stream)); list(stream.take(3)); stream.skip(100).shuffle(seed=0, buffer_size=10_000)
tokenized = train.map(lambda b: tok(b["text"], truncation=True, max_length=256), batched=True, num_proc=4, remove_columns=["text"])
train.filter(lambda ex: len(ex["text"]) > 200); train.select(range(100)); train.shuffle(seed=42); train.sort("label")
split = train.train_test_split(test_size=0.1, seed=42, stratify_by_column="label"); split["train"], split["test"]
train.rename_column("label", "labels"); train.remove_columns(["text"]); train.add_column("idx", list(range(len(train))))
train.cast_column("label", ClassLabel(names=["neg", "pos"])); train.features["label"].int2str(1)
train.unique("label"); train.map(lambda ex, i: {"id": i}, with_indices=True)
tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])   # rows come back as tensors
for batch in train.iter(batch_size=32): break
train.to_pandas().head(); Dataset.from_dict({"text": ["a", "b"], "label": [0, 1]}); Dataset.from_list([{"text": "a", "label": 0}])
Dataset.from_generator(lambda: ({"x": i} for i in range(10)))   # Dataset.from_pandas(df) for DataFrames
concatenate_datasets([ds["train"], ds["test"]]); interleave_datasets([ds["train"], ds["test"]], probabilities=[0.7, 0.3], seed=0)
DatasetDict({"train": split["train"], "validation": split["test"]}).push_to_hub("my-user/imdb-small", private=True)
train.save_to_disk("./imdb_train"); load_from_disk("./imdb_train")
# audio: ds.cast_column("audio", Audio(sampling_rate=16_000)) resamples on access · images: ds.cast_column("image", Image())
# into PyTorch: DataLoader(tokenized, batch_size=16, collate_fn=DataCollatorWithPadding(tok))
```

**Drills:** (1) Load 1% of a dataset, tokenize it with `batched=True`, and drop the raw text. (2) Make a stratified 90/10 split. (3) Stream the first 1,000 rows of a huge dataset without downloading it. (4) Turn a pandas DataFrame into a Dataset and push it to the Hub.

### 6.08 Trainer & TrainingArguments

```python
import numpy as np
import evaluate
from transformers import (TrainingArguments, Trainer, DataCollatorWithPadding, EarlyStoppingCallback,
                          DataCollatorForLanguageModeling, DataCollatorForTokenClassification, DataCollatorForSeq2Seq)
tok = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
raw = load_dataset("stanfordnlp/imdb")
enc = raw.map(lambda b: tok(b["text"], truncation=True, max_length=256), batched=True)
model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased", num_labels=2)
accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    return accuracy.compute(predictions=np.argmax(logits, axis=-1), references=labels)
args = TrainingArguments(
    output_dir="out/imdb", num_train_epochs=2, learning_rate=2e-5, weight_decay=0.01,
    per_device_train_batch_size=16, per_device_eval_batch_size=64, gradient_accumulation_steps=2,
    lr_scheduler_type="cosine", warmup_steps=100,
    eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
    load_best_model_at_end=True, metric_for_best_model="accuracy",
    logging_steps=50, bf16=torch.cuda.is_available(), report_to="none", seed=42,   # report_to: "wandb", "tensorboard", "mlflow", ...
)
trainer = Trainer(model=model, args=args, train_dataset=enc["train"], eval_dataset=enc["test"].shuffle(seed=0).select(range(2000)),
                  processing_class=tok, data_collator=DataCollatorWithPadding(tok), compute_metrics=compute_metrics,
                  callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])
trainer.train()                                  # resume after a crash: trainer.train(resume_from_checkpoint=True)
trainer.evaluate()
preds = trainer.predict(enc["test"].select(range(100))); preds.predictions.shape, preds.metrics
trainer.save_model("out/imdb/best")              # trainer.push_to_hub() uploads it
trainer.state.log_history[:3]                    # list of dicts (loss, learning_rate, eval_accuracy, ...): plot it, see 6.13
DataCollatorForLanguageModeling(tok, mlm=True, mlm_probability=0.15)   # masked LM; mlm=False builds causal-LM labels
DataCollatorForTokenClassification(tok); DataCollatorForSeq2Seq(tok, model=model)
# custom loss: subclass Trainer and override compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None)
```

**Drills:** (1) Fine-tune DistilBERT on 2,000 IMDB reviews and report accuracy. (2) Add F1 to `compute_metrics`. (3) Turn on early stopping and keep only the best checkpoint. (4) Resume an interrupted run from its last checkpoint.

### 6.09 Custom loops with Accelerate

```python
from accelerate import Accelerator
from torch.utils.data import DataLoader
accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=4)   # the same script runs on CPU, 1 GPU, many GPUs
train_dl = DataLoader(enc["train"].remove_columns(["text"]), batch_size=16, shuffle=True, collate_fn=DataCollatorWithPadding(tok))
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=1000)
model, optimizer, train_dl, scheduler = accelerator.prepare(model, optimizer, train_dl, scheduler)   # moves and wraps everything
model.train()
for batch in train_dl:
    with accelerator.accumulate(model):          # handles gradient accumulation and sync
        loss = model(**batch).loss
        accelerator.backward(loss)               # instead of loss.backward()
        if accelerator.sync_gradients: accelerator.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); scheduler.step(); optimizer.zero_grad()
accelerator.print("done"); accelerator.is_main_process; accelerator.wait_for_everyone()
# eval: preds, refs = accelerator.gather_for_metrics((logits.argmax(-1), batch["labels"]))   # collect from all processes
accelerator.save_state("ckpt"); accelerator.load_state("ckpt")   # model + optimizer + scheduler + RNG
accelerator.unwrap_model(model).save_pretrained("out/final", save_function=accelerator.save, is_main_process=accelerator.is_main_process)
```

```bash
accelerate config                                # answer questions once (GPUs, mixed precision, DeepSpeed/FSDP)
accelerate launch --num_processes 2 train.py     # multi-GPU launch
accelerate env                                   # print your setup for bug reports
```

**Drills:** (1) Port the 3.05 loop to Accelerate with bf16. (2) Add gradient accumulation of 8 steps. (3) Save and resume full training state.

### 6.10 PEFT: LoRA & QLoRA

```python
from peft import LoraConfig, get_peft_model, PeftModel, TaskType, prepare_model_for_kbit_training
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.bfloat16)
lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type=TaskType.CAUSAL_LM)
# target_modules="all-linear" adapts every linear layer; modules_to_save=["lm_head"] also trains (and saves) full copies of listed modules
peft_model = get_peft_model(base, lora_cfg)
peft_model.print_trainable_parameters()          # trainable params: ~1% of all params
# ...train with Trainer / SFTTrainer / your own loop exactly as usual...
peft_model.save_pretrained("lora-adapter")       # saves ONLY the adapter (a few MB)
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.bfloat16)
peft_model = PeftModel.from_pretrained(base, "lora-adapter")   # attach the adapter to a fresh base
merged = peft_model.merge_and_unload(); merged.save_pretrained("merged-model")   # plain model, no PEFT needed at inference
peft_model.load_adapter("lora-adapter", adapter_name="second"); peft_model.set_adapter("second")   # switch between adapters
with peft_model.disable_adapter(): base_logits = peft_model(**inputs).logits   # compare with and without the adapter
# QLoRA (GPU): 4-bit frozen base + LoRA adapters in bf16
# qbase = AutoModelForCausalLM.from_pretrained(name, quantization_config=q4, device_map="auto")
# qbase = prepare_model_for_kbit_training(qbase); qmodel = get_peft_model(qbase, lora_cfg)
```

**Drills:** (1) Compare the trainable-parameter count for r = 4, 16 and 64. (2) Fine-tune with LoRA, then merge and check that the logits match the unmerged model. (3) Hot-swap two adapters on the same base model.

### 6.11 TRL: SFT, DPO & GRPO

```python
from trl import SFTTrainer, SFTConfig, DPOTrainer, DPOConfig, GRPOTrainer, GRPOConfig
sft_data = load_dataset("trl-lib/Capybara", split="train")   # conversational rows: {"messages": [{"role": ..., "content": ...}, ...]}
sft = SFTTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",              # a model id or an already-loaded model
    train_dataset=sft_data,
    args=SFTConfig(output_dir="out/sft", max_length=1024, packing=False, assistant_only_loss=False,
                   learning_rate=2e-5, per_device_train_batch_size=4, report_to="none"),
    peft_config=LoraConfig(r=16, lora_alpha=32, target_modules="all-linear"),   # optional: LoRA instead of full fine-tuning
)
sft.train()
pref = load_dataset("trl-lib/ultrafeedback_binarized", split="train")   # rows: "chosen" and "rejected" conversations
dpo = DPOTrainer(model="Qwen/Qwen2.5-0.5B-Instruct", train_dataset=pref,
                 args=DPOConfig(output_dir="out/dpo", beta=0.1, max_length=1024, learning_rate=5e-7, report_to="none"))
dpo.train()                                      # logs rewards/chosen, rewards/rejected, rewards/accuracies, rewards/margins (compare with 3.19)
def reward_len(completions, **kwargs):           # GRPO: online RL; rewards come from your own functions
    return [-abs(50 - len(c)) / 50 for c in completions]
grpo = GRPOTrainer(model="Qwen/Qwen2.5-0.5B-Instruct", reward_funcs=reward_len, train_dataset=load_dataset("trl-lib/tldr", split="train"),
                   args=GRPOConfig(output_dir="out/grpo", num_generations=8, max_completion_length=128, report_to="none"))
grpo.train()
```

```bash
trl sft --model_name_or_path Qwen/Qwen2.5-0.5B --dataset_name trl-lib/Capybara --output_dir out/sft   # the same from the CLI
```

**Drills:** (1) SFT a small model on 1,000 conversations with LoRA. (2) Run DPO and plot `rewards/accuracies` from the log history. (3) Write a GRPO reward that favors answers containing a number.

### 6.12 Evaluate, safetensors, embeddings & diffusion

```python
acc = evaluate.load("accuracy"); acc.compute(predictions=[0, 1, 1], references=[0, 1, 0])   # {'accuracy': 0.667}
evaluate.load("f1").compute(predictions=[0, 1], references=[0, 1], average="macro")
clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"]); clf_metrics.compute(predictions=[0, 1], references=[0, 1])
evaluate.load("rouge").compute(predictions=["the cat sat"], references=["a cat sat down"])   # also "sacrebleu", "seqeval", "bertscore", "perplexity"
for preds_b, refs_b in [([0, 1], [0, 1]), ([1], [0])]: acc.add_batch(predictions=preds_b, references=refs_b)
acc.compute()                                    # accumulate batches, then compute once
from safetensors.torch import save_file, load_file, safe_open
save_file({"w": torch.randn(3, 3)}, "weights.safetensors"); load_file("weights.safetensors")["w"].shape   # no pickle -> safe to load
with safe_open("weights.safetensors", framework="pt") as f: print(list(f.keys()), f.get_tensor("w").dtype)   # lazy, per-tensor loading
from sentence_transformers import SentenceTransformer, util
st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
E = st.encode(["How do I bake bread?", "Bread baking tips", "CUDA out of memory"], normalize_embeddings=True, batch_size=32)
st.similarity(E, E)                              # (3, 3) cosine similarities
util.semantic_search(st.encode(["bread recipe"]), E, top_k=2)
from diffusers import DiffusionPipeline          # GPU
# pipe = DiffusionPipeline.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16).to("cuda")
# pipe("a watercolor fox", num_inference_steps=1, guidance_scale=0.0).images[0].save("fox.png")
```

**Drills:** (1) Build a 20-document semantic search with sentence-transformers. (2) Compute ROUGE between model summaries and references. (3) Save a state_dict as safetensors and load one tensor lazily.

### 6.13 Inspecting & visualizing models

```python
import matplotlib.pyplot as plt
import pandas as pd
enc1 = tok("The cat sat on the mat", return_tensors="pt")
with torch.no_grad(): out = eager(**enc1, output_attentions=True, output_hidden_states=True)
att = out.attentions[-1][0, 0]                   # last layer, batch item 0, head 0 -> (T, T)
tokens = tok.convert_ids_to_tokens(enc1["input_ids"][0])
fig, ax = plt.subplots(); ax.imshow(att, cmap="viridis"); ax.set_xticks(range(len(tokens)), tokens, rotation=90); ax.set_yticks(range(len(tokens)), tokens)
cls_by_layer = torch.stack([h[0, 0] for h in out.hidden_states])   # (13, 768): how the [CLS] vector evolves through layers
mask = enc1["attention_mask"][..., None]
mean_emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1)     # mean pooling -> one sentence embedding
hist = pd.DataFrame(trainer.state.log_history)   # Trainer logs -> DataFrame
ax = hist.dropna(subset=["loss"]).plot(x="step", y="loss"); hist.dropna(subset=["eval_loss"]).plot(x="step", y="eval_loss", ax=ax)
with torch.no_grad(): next_probs = lm(**inputs).logits[0, -1].softmax(-1)   # next-token distribution
top = next_probs.topk(10)
fig, ax = plt.subplots(); ax.barh([repr(lm_tok.decode(int(i))) for i in top.indices], top.values.float().cpu())
```

**Drills:** (1) Plot all 12 heads of one BERT layer as a 3×4 grid of heatmaps. (2) Plot train and eval loss from `log_history` on one Axes. (3) Show how the top-5 next-token probabilities change when you change one word of the prompt.

Full references: https://huggingface.co/docs/transformers · /datasets · /peft · /trl · /huggingface_hub


---

# Part II — Developer tools

## 7. Bash & the command line

Everything here works in bash on Linux. On macOS the default shell is zsh, and almost all of this works there too. Two differences: zsh arrays start at 1, and macOS ships BSD versions of tools like `sed`, `date` and `grep`. For GNU behavior, `brew install coreutils gnu-sed grep findutils` gives you `gls`, `gsed`, `ggrep` and `gfind`.

### 7.01 Shell basics & getting help

```bash
echo "hello"; echo "$SHELL"; bash --version
man ls                                           # full manual (q to quit, /word to search)
ls --help; help cd; type ls; which python; command -v git; whatis grep; apropos "copy files"
tldr tar                                         # short, practical examples (install: brew/apt install tldr)
clear; history | tail -20; exit
# macOS BSD vs GNU:  sed -i '' 's/a/b/' f  vs  sed -i 's/a/b/' f  ·  date -v+1d  vs  date -d tomorrow  ·  BSD grep has no -P
```

**Drills:** (1) Find the flag of `ls` that sorts by size. (2) Check whether a command is a builtin, an alias or a file. (3) Look up 3 `tar` examples with `tldr`.

### 7.02 Navigating & managing files

```bash
pwd; ls; ls -la; ls -lh; ls -lt; ls -lS; ls -1; ls -d */; tree -L 2
cd ~; cd -; cd ..; cd ../..; cd /abs/path; pushd /tmp; popd
mkdir data; mkdir -p exp/{train,val,test}; touch notes.md; rmdir empty_dir
cp a.txt b.txt; cp -r src/ backup/; cp -i a.txt b.txt; cp -p a.txt b.txt; cp config.yaml{,.bak}   # -i ask, -p keep timestamps, {,.bak} = backup copy
mv old.txt new.txt; mv *.csv data/; mv -n a.txt b.txt   # -n never overwrite
rm file.txt; rm -i *.tmp; rm -r old_dir; rm -rf build/   # no trash can, no undo: be careful with -rf
ln -s /data/datasets datasets; readlink datasets; realpath notes.md
basename /a/b/model.pt .pt; dirname /a/b/model.pt   # model, /a/b
file img.png; stat model.pt; wc -l train.csv
```

**Drills:** (1) Create `project/{data,src,notebooks}` in one command. (2) Back up `config.yaml` to `config.yaml.bak` with brace expansion. (3) Symlink a big dataset directory into your project.

### 7.03 Viewing & comparing files

```bash
cat file.txt; cat -n file.txt; tac file.txt; nl file.txt
less big.log                                     # space/b page, /word search, n/N next/prev, g/G top/bottom, F follow, q quit
head file.txt; head -n 20 file.txt; head -c 100 file.bin
tail -n 50 train.log; tail -n +2 data.csv        # everything EXCEPT the header line
tail -f train.log; tail -F train.log             # follow a growing log (-F survives log rotation)
diff a.txt b.txt; diff -u a.txt b.txt; diff -r dir1 dir2; cmp a.bin b.bin
comm -12 <(sort a.txt) <(sort b.txt)             # lines present in both files
md5sum model.pt; sha256sum model.pt              # macOS: md5 / shasum -a 256
xxd model.bin | head; strings model.bin | head
column -s, -t data.csv | less -S                 # view a CSV as aligned columns (-S: don't wrap)
```

**Drills:** (1) Print lines 100–120 of a file (two ways). (2) Follow a training log and stop with Ctrl-C. (3) Verify a download's checksum.

### 7.04 Permissions, archives & disk usage

```bash
ls -l                                            # -rwxr-x--- : type | owner rwx | group r-x | others ---
chmod +x run.sh; chmod u+x,g-w file; chmod 755 run.sh; chmod 600 ~/.ssh/id_ed25519; chmod -R u+rw dir/
chown user:group file; sudo chown -R "$USER" /data/project
tar -czvf archive.tar.gz dir/                    # c create, z gzip, v verbose, f file
tar -xzvf archive.tar.gz; tar -xzf archive.tar.gz -C dest/; tar -tzf archive.tar.gz; tar -cJf archive.tar.xz dir/
zip -r out.zip dir/; unzip out.zip -d dest/; unzip -l out.zip
gzip big.csv; gunzip big.csv.gz; zcat big.csv.gz | head; zgrep "error" logs.gz
du -sh data/; du -sh * | sort -h; du -h -d 1 ~; df -h   # folder sizes / largest last / one level deep / free space per disk
```

**Drills:** (1) Make a script executable and run it. (2) Archive a results folder as `.tar.gz` and list its contents without extracting. (3) Find the 5 largest directories in your home folder.

### 7.05 Finding files & searching text

```bash
find . -name "*.py"; find . -iname "*readme*"; find . -type d -name "__pycache__"
find . -type f -size +100M                       # files bigger than 100 MB
find . -mtime -1                                 # modified in the last 24 h
find . -maxdepth 2 -name "*.ckpt"; find . -name "*.py" -not -path "./.venv/*"
find . -name "*.pyc" -delete; find . -name "__pycache__" -type d -exec rm -rf {} +
find . -name "*.py" -exec wc -l {} +             # run a command on the results
find . -name "*.jpg" -print0 | xargs -0 -n 100 cp -t images/   # safe with spaces in names (GNU cp -t)
grep "error" train.log; grep -i "error" train.log; grep -n "loss" train.log; grep -v "DEBUG" train.log
grep -c "epoch" train.log; grep -l "TODO" *.py; grep -w "lr" config.py; grep -o "loss=[0-9.]*" train.log
grep -r "import torch" --include="*.py" .; grep -rn "TODO" src/ --exclude-dir=.venv
grep -E "error|warning" train.log; grep -A 3 -B 2 "Traceback" train.log; grep -F "a.b[0]" file.py
grep -oP "loss=\K[0-9.]+" train.log              # Perl regex, GNU grep only (\K drops the prefix)
rg "pattern"; rg -t py "torch.compile"; rg -l "wandb"   # ripgrep: fast, respects .gitignore
```

**Drills:** (1) Delete every `__pycache__` folder under the current directory. (2) Find which Python files import `pandas`. (3) Print the last 5 `val_acc` values from a log.

### 7.06 Text processing: cut, sort, uniq, tr, sed, awk, jq

```bash
cut -d, -f1,3 data.csv; cut -c1-10 file.txt
sort names.txt; sort -n nums.txt; sort -r; sort -u; sort -h sizes.txt
sort -t, -k3,3nr data.csv | head                 # by 3rd column, numeric, descending
sort | uniq -c | sort -rn | head                 # the classic "count occurrences" pipeline
uniq -d dupes.txt; tr 'a-z' 'A-Z' < f.txt; tr -d '\r' < win.txt > unix.txt; tr -s ' ' < f.txt; tr ',' '\t' < a.csv
sed 's/old/new/' f.txt; sed 's/old/new/g' f.txt # first per line / all per line
sed -i 's/lr: 0.1/lr: 0.01/' config.yaml         # in place (macOS: sed -i '' ...)
sed -n '10,20p' f.txt; sed '1d' data.csv; sed '/^#/d' config.txt; sed -E 's/([0-9]+)/<\1>/g' f.txt
awk '{print $1}' f.txt                           # first whitespace-separated field
awk -F, '{print $2, $NF}' data.csv               # comma separator; NF = number of fields ($NF = last)
awk -F, 'NR > 1 {sum += $3; n++} END {print sum / n}' data.csv   # mean of column 3, skipping the header
awk '$3 > 0.9' results.txt                       # rows where column 3 > 0.9
awk '{count[$1]++} END {for (k in count) print k, count[k]}' f.txt   # group-by count
awk -v OFS='\t' '{print $1, $2}' f.txt; awk 'length($0) > 80' f.py; awk '{printf "%-10s %.3f\n", $1, $2}' f.txt
paste -d, a.txt b.txt; paste -sd+ nums.txt | bc; join -t, a.csv b.csv; split -l 1000 big.csv chunk_
jq '.' data.json                                 # pretty-print JSON
jq '.items[] | {name, score}' data.json; jq -r '.[].name' data.json; jq 'map(select(.score > 0.9))' data.json
jq -s '.' rows.jsonl; jq -c '.[]' data.json; jq 'length' data.json
seq 1 5 | xargs -I{} echo "run {}"; cat urls.txt | xargs -n 1 -P 4 wget -q
```

**Drills:** (1) Count the unique values in the 2nd column of a CSV, most frequent first. (2) Replace `batch_size: 32` with `64` in a YAML file in place. (3) Compute the average of the `loss` column with awk. (4) Extract every `name` field from a JSON array with jq.

### 7.07 Pipes & redirection

```bash
cmd > out.txt                                    # stdout to a file (overwrite)
cmd >> out.txt                                   # append
cmd 2> err.txt                                   # stderr only
cmd > all.txt 2>&1                               # both (order matters: redirect stdout first)
cmd &> all.txt                                   # both (bash shorthand)
cmd > /dev/null 2>&1                             # discard everything
cmd < input.txt; cmd1 | cmd2 | cmd3
python train.py 2>&1 | tee train.log             # watch AND save
python train.py | tee -a train.log
cat > config.yaml <<EOF
lr: 0.001
batch_size: ${BATCH:-32}
EOF
cat > script.sh <<'EOF'
echo "quoted delimiter: $HOME is NOT expanded here"
EOF
grep "x" <<< "$text"                             # here-string: feed a variable as stdin
diff <(sort a.txt) <(sort b.txt)                 # process substitution: a command's output as a "file"
set -o pipefail                                  # a pipeline fails if ANY stage fails (not just the last)
```

**Drills:** (1) Run a script so that stdout and stderr both go to the screen and to a log. (2) Write a 3-line YAML config from a script with a heredoc that uses a variable. (3) Compare two unsorted files without creating temp files.

### 7.08 Variables, quoting & expansion

```bash
name="Ada"                                       # NO spaces around =
echo "$name" "${name}_v2" '$name'                # double quotes expand; single quotes are literal
readonly PI=3.14; unset name; export WANDB_PROJECT=demo
today=$(date +%F); n=$(wc -l < data.csv); files=$(ls *.csv)   # command substitution
echo "${missing:-default}"                       # use a default if unset or empty
: "${OUT_DIR:=./out}"                            # assign a default if unset
echo "${REQUIRED:?must be set}"                  # exit with an error if unset
s="experiments/run_042/model.pt"
echo "${#s}"                                     # 29         length
echo "${s:0:11}"                                 # experiments substring (offset, length)
echo "${s#*/}"                                   # run_042/model.pt   strip shortest prefix matching */
echo "${s##*/}"                                  # model.pt           strip longest prefix  (basename)
echo "${s%/*}"                                   # experiments/run_042 strip shortest suffix (dirname)
echo "${s%.pt}.onnx"                             # swap the extension
echo "${s/run/RUN}" "${s//_/-}" "${s^^}" "${s,,}"   # replace first / replace all / UPPER / lower
echo {1..5} {a..e} {0..10..2} {01..03}; mkdir -p run_{a,b,c}
ls *.py; ls data_??.csv; ls [abc]*; shopt -s globstar; ls **/*.py
printf "%s has %d rows\n" "data.csv" 120; printf "%05.2f\n" 3.14159
```

**Watch out:** Always quote variables (`"$file"`). An unquoted variable containing a space becomes two arguments, which can make `rm` delete the wrong thing.

**Drills:** (1) From `/data/runs/exp7/ckpt_best.pt` get the file name, the directory and the name without its extension. (2) Make `OUT` default to `./results` when it isn't set. (3) Create `fold_1` … `fold_5` directories in one command.

### 7.09 Arrays & arithmetic

```bash
arr=(alpha beta gamma); arr+=(delta)
echo "${arr[0]}" "${arr[-1]}" "${arr[@]}" "${#arr[@]}" "${!arr[@]}" "${arr[@]:1:2}"   # first, last, all, count, indices, slice
for x in "${arr[@]}"; do echo "$x"; done
declare -A lr=([small]=1e-3 [large]=3e-4)        # associative array (bash 4+; macOS system bash is 3.2 -> brew install bash)
echo "${lr[small]}"; for k in "${!lr[@]}"; do echo "$k=${lr[$k]}"; done
mapfile -t lines < file.txt; echo "${#lines[@]}" # read lines into an array
echo $((3 + 4 * 2)) $((17 / 5)) $((17 % 5)) $((2 ** 10))   # integer arithmetic only
i=0; ((i++)); ((i += 5)); echo "$i"
echo "scale=4; 1/3" | bc; awk 'BEGIN {print 1/3}'; python3 -c "print(1/3)"   # floating point
```

**Drills:** (1) Loop over an array of learning rates and print each one. (2) Map model names to batch sizes with an associative array. (3) Compute 7/3 to 3 decimals.

### 7.10 Conditionals & loops

```bash
if [[ -f data.csv ]]; then echo "file"; elif [[ -d data ]]; then echo "dir"; else echo "missing"; fi
# tests: -e exists · -f file · -d dir · -s non-empty · -r/-w/-x permissions · -z empty string · -n non-empty string
# strings: == != < > · glob: [[ $f == *.csv ]] · regex: [[ $s =~ ^run_([0-9]+)$ ]] && echo "${BASH_REMATCH[1]}"
# numbers: -eq -ne -lt -le -gt -ge  or arithmetic  (( n > 5 ))  · combine: && || !
n=7; if (( n > 5 )); then echo big; fi; [[ -f a.txt && ! -f b.txt ]] || echo "no"
mkdir -p out && cd out || exit 1                 # && runs on success, || on failure
case "$1" in
  train) echo "training" ;;
  eval|test) echo "evaluating" ;;
  *) echo "usage: $0 {train|eval}" ;;
esac
for f in data/*.csv; do echo "$f"; done
for seed in 0 1 2; do echo "seed $seed"; done
for ((i = 0; i < 5; i++)); do echo "$i"; done
while read -r line; do echo "$line"; done < file.txt
while IFS=, read -r id label; do echo "$id -> $label"; done < <(tail -n +2 labels.csv)   # parse a CSV, skipping the header
until ping -c1 example.com > /dev/null 2>&1; do sleep 5; done
for i in {1..10}; do [[ $i -eq 3 ]] && continue; [[ $i -eq 6 ]] && break; echo "$i"; done
```

**Drills:** (1) Print "exists" or "missing" for each of 3 paths. (2) Loop over all `.log` files and print each name with its line count. (3) Read a two-column CSV and echo a sentence per row.

### 7.11 Functions & scripts

```bash
#!/usr/bin/env bash
set -euo pipefail                                # exit on error · error on unset variables · fail on any pipe stage
usage() { echo "usage: $0 [-e epochs] [-g gpu] config.yaml" >&2; exit 1; }
log() { printf '[%s] %s\n' "$(date +%T)" "$*"; }
epochs=10; gpu=0
while getopts "e:g:h" opt; do
  case $opt in
    e) epochs=$OPTARG ;;
    g) gpu=$OPTARG ;;
    *) usage ;;
  esac
done
shift $((OPTIND - 1))
[[ $# -eq 1 ]] || usage
config=$1
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT      # cleanup always runs: success, error or Ctrl-C
train() {
  local cfg=$1                                   # local: doesn't leak outside the function
  CUDA_VISIBLE_DEVICES=$gpu python train.py --config "$cfg" --epochs "$epochs"
}
log "start $config"; train "$config"; log "done"
```

```bash
chmod +x run.sh && ./run.sh -e 5 configs/base.yaml
bash -x run.sh configs/base.yaml                 # trace each command as it runs (debugging)
shellcheck run.sh                                # lint: catches quoting bugs and common mistakes
source env.sh                                    # run in the CURRENT shell (keeps variables); ./env.sh runs in a child shell
# special variables: $0 script · $1..$9 args · $# count · "$@" all args · $? last exit code · $$ PID · $! last background PID
```

**Drills:** (1) Write a script that takes `--seed`-style options with `getopts` and validates them. (2) Add a cleanup trap that deletes a temp dir. (3) Run it under `bash -x` and read the trace.

### 7.12 Processes & jobs

```bash
ps aux | grep python; pgrep -af train.py; top; htop
kill 12345; kill -9 12345; pkill -f train.py     # polite stop / force kill / by command line
python train.py &                                # run in the background
jobs -l; fg %1; bg %1; disown -h %1; wait        # Ctrl-Z suspends the foreground job; bg resumes it in the background
nohup python train.py > train.log 2>&1 &         # survives closing the terminal
echo $! > train.pid; kill "$(cat train.pid)"
time python script.py; timeout 2h python train.py; sleep 10
watch -n 1 nvidia-smi                            # re-run a command every second
lsof -i :8888; kill "$(lsof -t -i :8888)"        # what's using port 8888, then kill it
nice -n 10 python preprocess.py                  # lower priority
uptime; free -h; nproc; lscpu; uname -a; whoami; hostname
```

**Drills:** (1) Start a long job with `nohup`, log out, log back in and check it. (2) Find and kill the process holding port 6006. (3) Suspend a job, resume it in the background, then bring it to the foreground.

### 7.13 Environment, PATH & config files

```bash
env; printenv PATH; echo "$PATH" | tr ':' '\n'
export PATH="$HOME/.local/bin:$PATH"             # prepend a directory
export CUDA_VISIBLE_DEVICES=0,1; export HF_HOME=/data/hf_cache
CUDA_VISIBLE_DEVICES=1 python train.py           # set a variable for ONE command only
# ~/.bashrc: interactive shells (Linux) · ~/.bash_profile: login shells · ~/.zshrc: macOS default · reload: source ~/.bashrc
alias ll='ls -lah'; alias gs='git status'; alias gpu='watch -n 1 nvidia-smi'; unalias ll; alias
set -a; source .env; set +a                      # export every KEY=value in a .env file
crontab -e                                       # "min hour day month weekday command", e.g.
# 0 3 * * * /home/ada/backup.sh >> /tmp/backup.log 2>&1      (every day at 03:00)
date +%Y-%m-%d_%H-%M-%S; TZ=UTC date
```

**Drills:** (1) Add `~/bin` to your PATH permanently. (2) Create 3 aliases for your most-used commands. (3) Schedule a nightly job that writes the date to a log.

### 7.14 Remote work: ssh, scp, rsync, curl & tmux

```bash
ssh ada@gpu-box; ssh -p 2222 ada@gpu-box; ssh -i ~/.ssh/id_ed25519 ada@gpu-box
ssh-keygen -t ed25519 -C "ada@example.com"; ssh-copy-id ada@gpu-box   # password-less login
eval "$(ssh-agent -s)"; ssh-add ~/.ssh/id_ed25519
ssh -L 8888:localhost:8888 ada@gpu-box           # local port forward: open http://localhost:8888 for the remote Jupyter
ssh -N -f -L 6006:localhost:6006 ada@gpu-box     # background tunnel (TensorBoard), no shell
ssh -J bastion ada@internal-box                  # hop through a jump host
scp model.pt gpu:~/models/; scp -r gpu:~/runs/exp1 ./
rsync -avzP data/ gpu:~/data/                    # trailing slash = copy the CONTENTS; resumable; sends only changes
rsync -avzP --exclude '.venv' --exclude '*.pt' ./ gpu:~/project/
rsync -avn --delete src/ backup/                 # -n dry run first! --delete mirrors deletions
curl -O https://example.com/file.zip; curl -L -o data.zip https://example.com/download   # -L follows redirects
curl -s https://api.github.com/repos/pytorch/pytorch | jq .stargazers_count
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"text": "hello"}'   # test your model API
wget -c https://example.com/big.tar              # -c resumes a partial download
tmux new -s train; tmux ls; tmux attach -t train; tmux kill-session -t train
# inside tmux (prefix Ctrl-b): d detach · c new window · n/p next/prev window · % split left/right · " split top/bottom
#   arrows move between panes · z zoom pane · [ scroll mode (q exits) · x kill pane · , rename window
```

```text
# ~/.ssh/config  ->  then just: ssh gpu
Host gpu
    HostName 10.0.0.5
    User ada
    IdentityFile ~/.ssh/id_ed25519
    LocalForward 8888 localhost:8888
```

**Drills:** (1) Set up key-based login and an `~/.ssh/config` alias for a server. (2) Open a remote Jupyter in your local browser through a tunnel. (3) Start training inside tmux, detach, reattach. (4) Sync a project to a server excluding `.venv` and checkpoints.

### 7.15 Python environments

```bash
python3 -m venv .venv; source .venv/bin/activate; deactivate; which python
python -m pip install -U pip; pip install numpy torch; pip install -r requirements.txt; pip install -e .
pip freeze > requirements.txt; pip list --outdated; pip show torch; pip uninstall -y torch
uv venv; uv pip install -r requirements.txt      # uv: very fast pip/venv replacement
uv init; uv add numpy torch; uv sync; uv run train.py; uv python install 3.12; uv lock
conda create -n ml python=3.12 -y; conda activate ml; conda install -c conda-forge numpy
conda env export > environment.yml; conda env create -f environment.yml; conda env list; conda remove -n ml --all
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
jupyter lab --no-browser --port 8888; python -m ipykernel install --user --name ml   # register the env as a notebook kernel
```

**Drills:** (1) Create a venv, install 3 packages and freeze them to `requirements.txt`. (2) Recreate it from that file on a new machine. (3) Register the venv as a Jupyter kernel.

### 7.16 ML workflow one-liners

```bash
nvidia-smi; nvidia-smi -l 1; nvtop
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=pid,used_memory --format=csv   # which processes hold GPU memory
CUDA_VISIBLE_DEVICES=0 python train.py           # pin a run to GPU 0
torchrun --nproc_per_node=4 train.py             # multi-GPU (see 3.16)
nohup python train.py --config cfg.yaml > "logs/run_$(date +%F_%H%M).log" 2>&1 &
for lr in 1e-3 3e-4 1e-4; do for seed in 0 1 2; do python train.py --lr "$lr" --seed "$seed" > "logs/lr${lr}_s${seed}.log" 2>&1; done; done
seq 0 3 | xargs -P 4 -I{} sh -c 'CUDA_VISIBLE_DEVICES={} python train.py --seed {} > logs/seed{}.log 2>&1'   # one run per GPU in parallel
grep -h "val_acc" logs/*.log | sort -t= -k2 -n | tail -1   # best line across runs
grep -o "loss=[0-9.]*" train.log | cut -d= -f2 > losses.txt   # extract a metric to plot
tail -f logs/*.log | grep --line-buffered "epoch"
du -sh ~/.cache/huggingface ~/.cache/torch ~/.cache/pip
ls -t checkpoints/*.pt | tail -n +4 | xargs rm -f   # keep only the 3 newest checkpoints
tensorboard --logdir runs --port 6006
python -m http.server 8000                       # serve the current folder: preview your github.io site at localhost:8000
```

**Drills:** (1) Launch a 3×3 learning-rate × seed sweep and find the best run from the logs. (2) Kill every process using GPU 1. (3) Plot losses extracted from a log with a 3-line Python script.

### 7.17 Keyboard shortcuts & history

```text
Tab / Tab Tab         complete / list possible completions
Ctrl-A / Ctrl-E       jump to start / end of line          Alt-B / Alt-F   back / forward one word
Ctrl-U / Ctrl-K       delete to start / end of line         Ctrl-W          delete previous word   Ctrl-Y  paste it back
Ctrl-R                search history backwards (press again for older matches; Enter runs, Esc edits)
Ctrl-L clear screen   Ctrl-C interrupt   Ctrl-D end of input / exit   Ctrl-Z suspend
!!                    repeat previous command (sudo !! = previous command with sudo)
!$  /  Alt-.          last argument of the previous command
!grep                 most recent command starting with "grep"
^old^new              re-run the previous command with old replaced by new
set -o vi             vi-style editing on the command line (Esc for normal mode; see Vim)
```

Full reference: https://www.gnu.org/software/bash/manual/

## 8. Git & GitHub

Git is version control for your code: snapshots (commits), parallel lines of work (branches), and collaboration through remotes such as GitHub. The mental model is three areas. Edits happen in the **working directory**, `git add` moves them to the **staging area**, and `git commit` writes them into the **repository**.

### 8.01 Setup & configuration

```bash
git config --global user.name "Ada Lovelace"
git config --global user.email "ada@example.com"
git config --global init.defaultBranch main
git config --global core.editor "vim"            # or "code --wait"
git config --global pull.rebase true             # pull = fetch + rebase (linear history)
git config --global push.autoSetupRemote true    # first push of a new branch sets its upstream
git config --global fetch.prune true             # forget remote branches deleted on the server
git config --global rerere.enabled true          # remember how you resolved conflicts
git config --global merge.conflictstyle zdiff3   # show the common ancestor in conflicts
git config --global alias.lg "log --oneline --graph --decorate --all"
git config --list --show-origin; git config user.email   # a repo's own setting overrides --global
ssh-keygen -t ed25519 -C "ada@example.com"       # add ~/.ssh/id_ed25519.pub at GitHub -> Settings -> SSH keys
ssh -T git@github.com                            # test the connection
```

**Drills:** (1) Configure name, email, default branch and editor on a fresh machine. (2) Create a `git lg` alias and use it. (3) Connect to GitHub over SSH.

### 8.02 Creating repositories & .gitignore

```bash
git init; git init my-project
git clone git@github.com:user/repo.git; git clone https://github.com/user/repo.git my-dir
git clone --depth 1 https://github.com/user/repo.git   # shallow: latest snapshot only
git clone -b dev --recurse-submodules git@github.com:user/repo.git
git check-ignore -v checkpoints/best.pt          # which .gitignore rule matches this path?
git rm -r --cached venv && git commit -m "chore: stop tracking venv"   # already committed it? untrack, keep the files
```

```gitignore
# .gitignore for an ML project (paths are relative to the repo root)
venv/
.venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
.DS_Store
.env
data/
!data/README.md
checkpoints/
*.pt
*.pth
*.ckpt
*.safetensors
wandb/
runs/
mlruns/
outputs/
```

**Watch out:** Never commit secrets (`.env`, tokens) or large binaries. Removing them later requires rewriting history (8.12).

**Drills:** (1) Initialize a repo for your curriculum folder and ignore `venv/`. (2) Check why `model.pt` is being ignored. (3) Untrack a folder you committed by mistake.

### 8.03 The daily loop: status, add, commit

```bash
git status; git status -s                        # what changed? (short format: M modified, A added, ?? untracked)
git add file.py; git add src/; git add .; git add -A; git add -u   # -u: only already-tracked files
git add -p                                       # stage hunk by hunk (y yes, n no, s split, e edit, q quit)
git restore --staged file.py                     # unstage (keeps your edits)
git commit -m "feat: add attention module"
git commit -m "fix: mask padding in attention" -m "Padding tokens were attended to when batch had mixed lengths."
git commit -am "docs: update README"             # stage tracked files + commit in one step
git commit --amend --no-edit                     # add staged changes to the last commit (only if not pushed)
git commit --amend -m "better message"
git rm old.py; git mv utils.py helpers.py
```

**Drills:** (1) Commit two unrelated changes in one file as two separate commits using `git add -p`. (2) Fix a typo in your last commit message. (3) Unstage a file you added by mistake.

### 8.04 History & diffs

```bash
git log; git log --oneline; git log --oneline --graph --decorate --all; git log -n 5
git log -p file.py; git log --stat; git log --follow -- src/model.py   # history of one file (across renames)
git log --author="Ada" --since="2 weeks ago" --until="yesterday"; git log --grep="fix"
git log -S "def forward"                         # "pickaxe": commits that added/removed this string
git log main..feature                            # commits on feature that are not on main
git log --pretty=format:"%h %an %ar %s"
git show HEAD; git show abc123; git show abc123:src/model.py; git show --stat HEAD
git diff                                         # working directory vs staging area
git diff --staged                                # staging area vs last commit (what you're about to commit)
git diff HEAD; git diff main..feature; git diff main...feature   # ... = changes since the branches diverged
git diff --stat; git diff --name-only HEAD~3; git diff --word-diff; git diff abc123 -- file.py
git blame file.py; git blame -L 10,30 file.py; git shortlog -sn
# revisions: HEAD · HEAD~1 (parent) · HEAD~3 · HEAD^2 (2nd parent of a merge) · @{u} (upstream) · HEAD@{2} (reflog) · :/fix (newest commit message matching "fix")
```

**Drills:** (1) Show every commit that touched `train.py`, with its diff. (2) Find the commit that introduced the string `lr_scheduler`. (3) Compare your branch with main since it branched off.

### 8.05 Branches

```bash
git branch; git branch -a; git branch -vv        # local / all (incl. remote) / with upstream & last commit
git switch -c feature/kv-cache                   # create + switch (older: git checkout -b)
git switch main; git switch -                    # switch / jump back to the previous branch
git branch -m old-name new-name; git branch -d merged-branch; git branch -D unmerged-branch
git branch --merged; git branch --no-merged; git branch -u origin/main
git switch --detach v1.0                         # look around an old version
git switch --orphan gh-pages                     # new branch with no history (e.g. a site branch)
```

**Drills:** (1) Create a feature branch, commit, and switch back to main. (2) Delete branches that are already merged. (3) List branches with their upstream status.

### 8.06 Merging & resolving conflicts

```bash
git switch main; git merge feature/kv-cache      # fast-forward if possible, else a merge commit
git merge --no-ff feature/kv-cache               # always create a merge commit
git merge --squash feature/kv-cache && git commit -m "feat: KV cache"   # one squashed commit
git merge --abort                                # back out of a conflicted merge
git diff --name-only --diff-filter=U             # list conflicted files
git checkout --ours file.py; git checkout --theirs file.py   # take one side wholesale
git add file.py; git merge --continue            # after fixing conflicts by hand
git merge -X theirs feature/kv-cache             # resolve every conflict in favor of the incoming branch
git mergetool
```

```text
<<<<<<< HEAD
lr = 1e-3            # your branch
||||||| base
lr = 1e-2            # common ancestor (shown with conflictstyle zdiff3)
=======
lr = 3e-4            # incoming branch
>>>>>>> feature/kv-cache
```

Edit the file to the final version, delete all marker lines, then `git add` the file and continue.

**Drills:** (1) Create a conflict on purpose (both branches edit the same line) and resolve it. (2) Abort a merge midway. (3) Squash-merge a feature branch.

### 8.07 Rebasing & rewriting history

```bash
git switch feature/kv-cache; git rebase main     # replay your commits on top of the latest main
git rebase --continue; git rebase --skip; git rebase --abort
git rebase -i HEAD~4                             # interactive: pick · reword · edit · squash · fixup · drop · reorder lines
git commit --fixup abc123; git rebase -i --autosquash main   # mark a fix for an older commit, then fold it in automatically
git rebase --onto main old-base feature/kv-cache # move a branch onto a new base
git pull --rebase
git push --force-with-lease                      # required after rewriting a pushed branch; refuses if someone else pushed
```

**Watch out:** Never rebase or force-push a branch others build on (like `main`). Rewriting history is for your own feature branches.

**Drills:** (1) Squash 4 WIP commits into one clean commit. (2) Reword an older commit message. (3) Rebase a feature branch on main and force-push it safely.

### 8.08 Remotes: fetch, pull, push & forks

```bash
git remote -v; git remote add origin git@github.com:user/repo.git; git remote set-url origin git@github.com:user/new.git
git remote rename origin upstream; git remote remove old; git remote show origin
git fetch; git fetch --all --prune               # download without changing your files
git pull; git pull --rebase                      # fetch + integrate
git push -u origin main                          # first push: set the upstream
git push; git push origin feature/kv-cache; git push origin --delete feature/kv-cache; git push --tags
git remote add upstream https://github.com/original/repo.git   # fork workflow: keep your fork in sync
git fetch upstream; git rebase upstream/main; git push --force-with-lease
```

**Drills:** (1) Push a new repo to GitHub for the first time. (2) Sync a fork with the original repository. (3) Delete a remote branch.

### 8.09 Stash, tags & cherry-pick

```bash
git stash; git stash push -m "wip attention" -- src/attn.py; git stash -u   # -u includes untracked files
git stash list; git stash show -p stash@{0}; git stash pop; git stash apply stash@{1}; git stash drop stash@{1}; git stash clear
git stash branch try-fix                         # new branch from a stash
git tag; git tag v1.0; git tag -a v1.0 -m "first release"; git push origin v1.0; git push --tags
git tag -d v1.0; git push origin --delete v1.0; git describe --tags
git cherry-pick abc123; git cherry-pick -x abc123; git cherry-pick a1b2c3..d4e5f6; git cherry-pick --no-commit abc123; git cherry-pick --abort
```

**Drills:** (1) Stash half-done work, fix a bug on main, then restore the work. (2) Tag a release and push the tag. (3) Copy one bug-fix commit from a feature branch to main.

### 8.10 Undoing things

| I want to… | Command |
|---|---|
| discard unstaged edits in a file | `git restore file.py` |
| unstage a file | `git restore --staged file.py` |
| get an old version of one file | `git restore --source=HEAD~2 file.py` |
| undo the last commit, keep changes staged | `git reset --soft HEAD~1` |
| undo the last commit, keep changes unstaged | `git reset HEAD~1` |
| undo the last commit and throw the changes away | `git reset --hard HEAD~1` |
| make my branch identical to the remote | `git fetch && git reset --hard origin/main` |
| undo a pushed commit safely (new inverse commit) | `git revert abc123` |
| undo a pushed merge | `git revert -m 1 <merge-sha>` |
| delete untracked files (dry run first!) | `git clean -n` then `git clean -fd` (`-x` also ignored files) |
| recover "lost" commits after a bad reset | `git reflog` then `git reset --hard HEAD@{3}` or `git branch rescue <sha>` |

**Watch out:** `reset --hard`, `restore` and `clean -f` delete uncommitted work permanently. Committed work is almost always recoverable through `git reflog`.

**Drills:** (1) Make 3 commits, reset the last 2 softly, and recommit them as one. (2) Revert a pushed commit. (3) "Lose" a commit with `reset --hard` and recover it from the reflog.

### 8.11 Debugging & search

```bash
git bisect start; git bisect bad; git bisect good v1.0   # binary search for the commit that broke something
git bisect good                                  # ...after testing each checkout it offers (or: git bisect bad)
git bisect run pytest tests/test_model.py        # automate: exit code 0 = good, non-zero = bad
git bisect reset
git grep "TODO"; git grep -n "learning_rate" -- "*.py"
git log -L :forward:src/model.py                 # history of a single function
git rev-parse HEAD; git rev-parse --short HEAD; git rev-parse --show-toplevel
git count-objects -vH; git fsck
```

**Drills:** (1) Use `git bisect run` to find the commit where a test started failing. (2) Show the history of one function. (3) Print the repo root from any subfolder.

### 8.12 Advanced: worktrees, submodules, LFS, hooks & history surgery

```bash
git worktree add ../repo-exp experiment; git worktree list; git worktree remove ../repo-exp   # two branches checked out at once
git submodule add https://github.com/user/lib.git external/lib; git submodule update --init --recursive; git submodule foreach git pull
git lfs install; git lfs track "*.pt"; git add .gitattributes; git lfs ls-files; git lfs pull   # large files stored outside git history
git sparse-checkout set src docs; git sparse-checkout disable   # check out only some folders of a huge repo
pip install git-filter-repo
git filter-repo --path data/huge.csv --invert-paths   # purge a file from ALL history (then force-push; collaborators re-clone)
git archive --format=zip HEAD > snapshot.zip
pip install pre-commit; pre-commit install; pre-commit run --all-files   # run linters/formatters before every commit
```

```yaml
# .pre-commit-config.yaml (pin rev to the latest release tags)
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9
    hooks:
      - id: ruff
      - id: ruff-format
  - repo: https://github.com/kynan/nbstripout
    rev: 0.7.1
    hooks:
      - id: nbstripout
```

**Drills:** (1) Check out an experiment branch in a second worktree. (2) Track `*.safetensors` with LFS. (3) Add pre-commit with ruff and nbstripout, then run it on all files.

### 8.13 GitHub: gh CLI, pull requests, Actions & Pages

```bash
gh auth login
gh repo create comp-guide --public --source=. --remote=origin --push   # create on GitHub from the current folder
gh repo clone user/repo; gh repo view --web
git switch -c docs/numpy-drills; git push -u origin docs/numpy-drills
gh pr create --fill; gh pr create --title "Add NumPy drills" --body "Adds 18 modules" --base main
gh pr list; gh pr checkout 12; gh pr diff 12; gh pr view 12 --web
gh pr review 12 --approve; gh pr merge 12 --squash --delete-branch
gh issue create --title "Broken link" --body "..."; gh issue list
gh run list; gh run watch; gh release create v1.0 --notes "First public version"
```

```yaml
# .github/workflows/pages.yml : publish the ./site folder to https://<user>.github.io/<repo>/
# (repo Settings -> Pages -> Source: "GitHub Actions"; check the actions' pages for newer major versions)
name: Deploy site
on:
  push:
    branches: [main]
  workflow_dispatch:
permissions:
  contents: read
  pages: write
  id-token: write
concurrency:
  group: pages
  cancel-in-progress: true
jobs:
  deploy:
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/configure-pages@v5
      - uses: actions/upload-pages-artifact@v3
        with:
          path: ./site
      - id: deployment
        uses: actions/deploy-pages@v4
```

```yaml
# .github/workflows/test.yml : run tests (e.g. the guide's code snippets) on every push and pull request
name: tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -r requirements.txt
      - run: pytest -q
```

The simplest Pages setup needs no workflow at all. Use Settings → Pages → "Deploy from a branch" and choose `main` with the `/docs` folder.

**Drills:** (1) Create this guide's repo with `gh repo create` and push it. (2) Open a PR from a branch, review it, and squash-merge it. (3) Publish a one-page site to github.io.

### 8.14 Git for ML projects & commit conventions

```bash
git rev-parse --short HEAD > outputs/run_42/commit.txt   # record the exact code version of every experiment
git diff > outputs/run_42/uncommitted.patch      # ...plus any uncommitted changes
git tag exp/lr3e-4-bs64                          # tag the commit behind a result you report
nbstripout --install                             # strip notebook outputs on commit (small, reviewable diffs)
dvc init; dvc add data/raw.csv; git add data/raw.csv.dvc .gitignore   # DVC: version data next to code
dvc remote add -d store s3://my-bucket/dvc; dvc push; dvc pull
```

Commit messages: use an imperative subject of 50 characters or fewer ("add", not "added"), a blank line, then a body wrapped at 72 characters that explains why. The Conventional Commits prefixes are `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `perf:`, `chore:` and `ci:`. One logical change per commit makes `revert`, `bisect` and `cherry-pick` work cleanly.

**Drills:** (1) Make every training run save its commit hash. (2) Rewrite your last 5 commit messages in Conventional Commits style (on an unpushed branch). (3) Put a 1 GB dataset under DVC instead of git.

Full reference: https://git-scm.com/docs

## 9. Vim

Vim is modal. You spend most of your time in **normal mode**, where keys are commands that compose like a grammar: *operator* + *count* + *motion/text object*. For example, `d2w` deletes two words and `ci"` changes the text inside quotes. Learn the grammar and you rarely need to memorize individual commands. Run `vimtutor` from a shell for a 30-minute interactive start.

### 9.01 Modes, opening, saving & quitting

```text
vim file.py            open (vim +42 file.py -> at line 42 · vim -O a.py b.py -> side by side · view file -> read-only)
Esc  (or Ctrl-[)       back to NORMAL mode: navigate and run commands
i                      INSERT mode: type text
v  V  Ctrl-v           VISUAL mode: select characters / lines / a block
:                      COMMAND-LINE mode: Ex commands like :w
R                      REPLACE mode: overtype
:w   :w new.py         save / save as          :wa   save all buffers
:q   :q!   ZQ          quit / quit discarding changes
:wq  :x   ZZ           save and quit           :qa!  quit everything without saving
:e other.py            open another file       :e!   revert to the saved version
```

**Drills:** (1) Open a file, add a line, save, quit, in under 5 seconds. (2) Make a change and quit without saving. (3) Open two files side by side.

### 9.02 Moving around

```text
h j k l                left down up right            5j = 5 lines down (counts work with almost everything)
w  b  e  ge            next word / previous word / end of word / end of previous word
W  B  E                same, but WORDs are separated only by whitespace (my.var[0] is one WORD)
0  ^  $  g_            line start / first non-blank / line end / last non-blank
gg  G  42G  :42        top / bottom / line 42
Ctrl-d Ctrl-u          half page down / up           Ctrl-f Ctrl-b  full page down / up
H  M  L                top / middle / bottom of the screen
zz  zt  zb             scroll so the cursor line is centered / at top / at bottom
f(  F(  t(  T(         jump to next ( / previous ( / just before next ( / just after previous (
;  ,                   repeat / reverse the last f t F T
%                      jump to the matching ( ) [ ] { }
{  }                   previous / next blank line (paragraph or code block)
*  #                   next / previous occurrence of the word under the cursor
gd                     go to local definition        gf  open the file path under the cursor
Ctrl-o  Ctrl-i         back / forward through your jump history
```

**Drills:** (1) Navigate a Python file using only `w b e f ; % { }`, no arrow keys. (2) Jump to line 120, then back to where you were. (3) Jump between the matching brackets of a long function call.

### 9.03 Inserting & basic editing

```text
i  a                   insert before / after the cursor
I  A                   insert at line start / line end
o  O                   open a new line below / above
x  X                   delete character under / before the cursor
r<c>                   replace one character with <c>
s  S                   delete character / whole line and start inserting
dd  yy                 delete (cut) / yank (copy) the current line
p  P                   paste after / before the cursor
u  Ctrl-r  U           undo / redo / undo all changes on the line
.                      repeat the last change (the most powerful key in Vim)
J  gJ                  join the next line onto this one (with / without a space)
~                      toggle case of a character
>>  <<  ==             indent / dedent / auto-indent the line  (5>> indents 5 lines)
Ctrl-a  Ctrl-x         increment / decrement the number under the cursor (10 Ctrl-a adds 10)
Insert mode:           Ctrl-w delete previous word · Ctrl-u delete to line start · Ctrl-n / Ctrl-p complete a word · Ctrl-r a paste register a
```

**Drills:** (1) Duplicate a line 3 times (`yy3p`). (2) Change `epochs = 10` to `epochs = 30` with `Ctrl-a`. (3) Swap two lines (`ddp`).

### 9.04 The grammar: operators × motions × text objects

```text
OPERATORS     d delete · c change (delete + insert) · y yank · > indent · < dedent · = auto-format
              gu lowercase · gU uppercase · g~ toggle case · gq wrap text
MOTIONS       anything from 9.02: w e b $ 0 } G f, t, /pattern ...
TEXT OBJECTS  (after an operator or in visual mode)  i = inner, a = around (includes delimiters/space)
              iw aw word · is as sentence · ip ap paragraph
              i( a(  i[ a[  i{ a{  i< a<  brackets · i" a"  i' a'  i` a`  quotes · it at  HTML/XML tags

dw     delete to the next word          d$ / D   delete to end of line       dG   delete to end of file
cw     change to end of word            c$ / C   change to end of line       cc   change the whole line
ciw    change the word under the cursor ci"      change inside quotes        ci(  change function arguments
da(    delete parentheses and contents  dap      delete a paragraph          yi{  yank inside braces
dt,    delete up to (not including) ,   cf)      change through the next )   d/foo  delete up to "foo"
3dd    delete 3 lines                   d3w      delete 3 words              >ip  indent a paragraph
gUiw   uppercase a word                 =i{      re-indent inside braces     gqap rewrap a paragraph
```

**Drills:** (1) Rename a variable: `ciw` then `n .` on each search match. (2) Replace every argument of `train(model, data, lr=1e-3)` with `ci(`. (3) Delete a whole Python function with `dap`.

### 9.05 Visual mode

```text
v  V  Ctrl-v           select characters / lines / a rectangular block
o                      jump to the other end of the selection        gv  reselect the last selection
then:  d y c > < = ~ u U J   act on the selection
:                      run an Ex command on the selected lines (Vim pre-fills :'<,'>)
viw  vi(  vip          select a word / inside parentheses / a paragraph
Block edits (Ctrl-v):  select a column, then
    I#<Esc>            insert "#" at the start of every selected line (comment out code)
    A,<Esc>            append "," after the block on every line
    $A;<Esc>           append ";" at the END of every line
    c<text><Esc>       replace the block on every line
```

**Drills:** (1) Comment out 10 lines with a block insert, then uncomment them. (2) Indent a selection 3 levels with `3>`. (3) Uppercase a selected paragraph.

### 9.06 Search & replace

```text
/pattern  ?pattern     search forward / backward;  n N next / previous match;  :noh clear highlighting
/\<loss\>              whole word only          /pattern\c   case-insensitive
/\vrun_(\d+)           \v "very magic": regex syntax close to Python (no backslashes before ( ) + ? |)
:s/old/new/            first match on the current line
:s/old/new/g           every match on the current line
:%s/old/new/g          every match in the file              :%s/old/new/gc   ask before each
:%s/old//gn            count matches without changing anything
:10,20s/a/b/g          only lines 10-20                     :'<,'>s/a/b/g    only the visual selection
:%s/\v(\w+) \= (\w+)/\2 = \1/g     capture groups \1 \2 (in \v mode a literal = is written \=)
:%s/\s\+$//e           strip trailing whitespace (e: no error if there's none)
& in the replacement   the whole match;  \r inserts a newline
:g/pattern/d           delete every line that matches       :v/pattern/d   delete every line that does NOT match
:g/TODO/normal A  # check     run normal-mode keys on every matching line
:vimgrep /TODO/ **/*.py        search the whole project -> :copen shows results, :cn / :cp to move
:cdo s/old/new/g | update      replace in every file from the results list and save
```

**Drills:** (1) Rename `lr` to `learning_rate` in a file, confirming each change. (2) Delete all lines containing `print(`. (3) Swap `a = b` to `b = a` on every line with a capture-group substitution.

### 9.07 Registers & the clipboard

```text
"ayy  "ap              yank into register a / paste from register a     "Ayy  APPEND to register a
"+y  "+p               system clipboard ("* on some systems; check :echo has('clipboard'))
"0p                    paste your last YANK, even after deleting other text
"_d                    delete into the "black hole" (keeps your yank intact)
".  ":  "/  "%         last inserted text / last command / last search / current file name
:reg                   show all registers
Ctrl-r a               (insert mode) paste register a      Ctrl-r =6*7<Enter>   inserts 42 (expression register)
```

**Drills:** (1) Yank 3 different snippets into registers a, b, c and paste them in reverse order. (2) Delete lines without losing what you yanked earlier. (3) Copy a line to the system clipboard.

### 9.08 Macros & the dot command

```text
qa ... q               record keystrokes into register a
@a  @@  20@a           play once / replay the last macro / play 20 times
:5,20normal @a         run the macro on lines 5-20     :'<,'>normal @a   on a selection
Recipe: move to the first target line -> qa -> make the edit -> move to the next line (j0) -> q -> 99@a
Edit a macro: "ap (paste it as text), fix it, then 0"ay$ (yank it back into a)
Dot workflow: /word -> cgn new<Esc> -> . . . (each . changes the next match)
```

**Drills:** (1) Convert 20 `key=value` lines into a Python dict literal with one macro. (2) Turn a column of names into a quoted, comma-separated list. (3) Use `cgn` + `.` to rename 5 occurrences.

### 9.09 Buffers, windows & tabs

```text
:e file  :ls  :b 2  :b part_of_name  :bn  :bp  :bd      open / list / switch / next / previous / close buffers
Ctrl-^                 toggle between the last two files
:sp file   :vsp file   split horizontally / vertically     (Ctrl-w s / Ctrl-w v split the current file)
Ctrl-w h/j/k/l         move to the window left/down/up/right    Ctrl-w w   cycle windows
Ctrl-w q  Ctrl-w o     close this window / close all others      Ctrl-w =   equal sizes   Ctrl-w _ / Ctrl-w |  maximize
:resize 20   :vertical resize 80
:tabnew file   gt  gT   :tabclose                    tabs = layouts of windows
:Explore (:Ex)  :Vexplore                            built-in file browser (netrw)
:terminal                                            a shell in a split (Ctrl-w N for normal mode inside it)
```

**Drills:** (1) Open `model.py` and `train.py` in a vertical split and move between them. (2) Jump between the last two files with `Ctrl-^`. (3) Run your script in a `:terminal` split.

### 9.10 Marks, folds & undo history

```text
ma   'a   `a            set mark a / jump to its line / jump to its exact spot     :marks
mA                      uppercase marks are global (they work across files)
''   ``                 jump back to where you were before the last jump
zf}  zo  zc  za         create a fold / open / close / toggle
zR  zM                  open / close all folds       :set foldmethod=indent   (good for Python)
:earlier 10m   :later 5m   g-  g+                     travel through the undo history by time or by state
:set undofile                                         keep undo history after closing the file
```

**Drills:** (1) Mark two locations in a file and bounce between them. (2) Fold every function in a Python file and open one. (3) Undo back to how the file looked 5 minutes ago.

### 9.11 Command-line power moves

```text
:!python %             run the current file              :!ls   any shell command
:w !python             run the buffer without saving      :r !date   insert a command's output
:%!python -m json.tool pipe the whole file through a command (pretty-print JSON)
:'<,'>!sort            sort the selected lines            :sort   :sort u (unique)   :sort n (numeric)
:%norm A,              append a comma to every line       :g/^$/d   delete empty lines
:10,20d   :10,20m 30   :10,20t$                           delete / move / copy line ranges  (. = current line, $ = last, % = all)
:w !sudo tee %         save a root-owned file you opened without sudo
:set ff=unix           convert Windows line endings      :set spell   ]s next misspelling   z= suggestions
:help ciw   :help :s   :help registers                   the built-in manual is excellent
```

**Drills:** (1) Sort the import lines of a file. (2) Pretty-print a minified JSON file in place. (3) Run the current Python file without leaving Vim.

### 9.12 A starter .vimrc & Vim keys everywhere

```vim
" ~/.vimrc
set nocompatible
syntax on
filetype plugin indent on
set number relativenumber          " relative numbers make 5j and 3dd easy to count
set tabstop=4 shiftwidth=4 expandtab autoindent smartindent
set ignorecase smartcase incsearch hlsearch
set clipboard=unnamedplus          " share the system clipboard (macOS: unnamed)
set mouse=a scrolloff=5 splitright splitbelow undofile
set colorcolumn=88 cursorline
let mapleader = " "
nnoremap <leader>w :w<CR>
nnoremap <leader>h :nohlsearch<CR>
inoremap jk <Esc>
nnoremap <C-h> <C-w>h
nnoremap <C-l> <C-w>l
autocmd BufWritePre *.py :%s/\s\+$//e
```

Vim keys elsewhere:
- **Shell:** `set -o vi` in bash, `bindkey -v` in zsh.
- **Editors:** the VS Code "Vim" extension, IdeaVim in PyCharm, `jupyterlab-vim` in JupyterLab.
- **Neovim:** a modern Vim fork configured in `~/.config/nvim/init.lua`.

**Drills:** (1) Install this `.vimrc` and use `jk` to leave insert mode for a day. (2) Map `<leader>r` to run the current Python file. (3) Turn on Vim mode in your editor and use it for a week.

Full reference: `:help` inside Vim, or https://vimhelp.org/

## 10. SQL

Written in the PostgreSQL dialect, the most common choice for ML/analytics work. Most of it also runs unchanged in DuckDB, Snowflake and BigQuery, and much of it in SQLite and MySQL. Remember the logical order a query runs in: `FROM` → `JOIN` → `WHERE` → `GROUP BY` → `HAVING` → `SELECT` → `DISTINCT` → `ORDER BY` → `LIMIT`. It explains most "why can't I use that alias here?" errors.

### 10.00 Practice schema

Run this once (in `psql`, DuckDB, or an online Postgres playground). Every example below uses these two tables.

```sql
CREATE TABLE users (
  id          INTEGER PRIMARY KEY,
  name        TEXT NOT NULL,
  country     TEXT,
  signup_date DATE NOT NULL
);
CREATE TABLE orders (
  id         INTEGER PRIMARY KEY,
  user_id    INTEGER NOT NULL REFERENCES users(id),
  amount     NUMERIC(10, 2) NOT NULL CHECK (amount >= 0),
  status     TEXT NOT NULL DEFAULT 'paid',
  created_at TIMESTAMP NOT NULL
);
INSERT INTO users VALUES
  (1, 'Ana', 'US', '2024-01-05'), (2, 'Ben', 'UK', '2024-01-20'), (3, 'Cal', 'US', '2024-02-11'),
  (4, 'Dee', NULL, '2024-03-02'), (5, 'Eve', 'DE', '2024-03-15');
INSERT INTO orders VALUES
  (1, 1, 120.00, 'paid', '2024-01-06 10:00'), (2, 1, 35.50, 'paid', '2024-02-01 09:30'),
  (3, 2, 80.00, 'refunded', '2024-02-03 14:00'), (4, 3, 15.00, 'paid', '2024-02-12 08:15'),
  (5, 3, 220.00, 'paid', '2024-03-01 19:45'), (6, 3, 42.00, 'paid', '2024-03-20 12:00'),
  (7, 5, 99.99, 'pending', '2024-03-16 16:20');
```

### 10.01 SELECT: filtering, sorting & limiting

```sql
SELECT * FROM users;
SELECT name, country FROM users;
SELECT name AS customer, signup_date AS joined FROM users;
SELECT DISTINCT country FROM users;
SELECT * FROM orders WHERE amount > 50;
SELECT * FROM orders WHERE status = 'paid' AND amount BETWEEN 20 AND 150;   -- BETWEEN is inclusive
SELECT * FROM users WHERE country IN ('US', 'UK');
SELECT * FROM users WHERE country IS NULL;            -- never "= NULL" (that is always unknown)
SELECT * FROM users WHERE name LIKE 'A%';             -- % any characters, _ exactly one; ILIKE ignores case
SELECT * FROM users WHERE country <> 'US';            -- NULL countries are NOT returned (NULL <> 'US' is unknown)
SELECT * FROM users WHERE country IS DISTINCT FROM 'US';   -- NULL-safe "not equal"
SELECT * FROM orders ORDER BY amount DESC;
SELECT * FROM orders ORDER BY status, amount DESC;
SELECT * FROM users ORDER BY country NULLS LAST;
SELECT * FROM orders ORDER BY created_at DESC LIMIT 3;
SELECT * FROM orders ORDER BY id LIMIT 3 OFFSET 3;    -- page 2 (standard SQL: OFFSET 3 ROWS FETCH FIRST 3 ROWS ONLY)
-- line comment      /* block comment */
```

**Drills:** (1) List paid orders over $50, newest first. (2) Find users whose country is missing. (3) Show page 3 of orders with 2 orders per page.

### 10.02 Expressions: CASE, NULL handling & casting

```sql
SELECT id, amount,
       amount * 1.2 AS with_tax,
       ROUND(amount * 0.9, 2) AS discounted,
       CASE WHEN amount >= 100 THEN 'large'
            WHEN amount >= 40  THEN 'medium'
            ELSE 'small' END AS size
FROM orders;
SELECT name, COALESCE(country, 'unknown') AS country FROM users;   -- first non-NULL argument
SELECT id, NULLIF(status, 'pending') AS status FROM orders;         -- NULL when the two are equal
SELECT CAST(amount AS INTEGER), amount::TEXT, '2024-01-01'::DATE FROM orders;
SELECT 7 / 2 AS int_div, 7 / 2.0 AS dec_div, 7::NUMERIC / 2 AS cast_div;   -- 3, 3.5, 3.5: integer division trap
SELECT name || ' (' || COALESCE(country, '?') || ')' AS label FROM users;   -- || concatenates (NULL || 'x' is NULL)
SELECT NULL = NULL AS eq, NULL IS NULL AS is_null, 1 + NULL AS plus;  -- NULL, true, NULL: NULL propagates
SELECT GREATEST(1, 5, 3), LEAST(1, 5, 3);
```

**Drills:** (1) Bucket orders into small/medium/large. (2) Display "unknown" for missing countries. (3) Compute the share of each order in percent with correct decimals.

### 10.03 Aggregation: GROUP BY & HAVING

```sql
SELECT COUNT(*), COUNT(country), COUNT(DISTINCT country) FROM users;   -- all rows / non-NULL / distinct non-NULL
SELECT SUM(amount), AVG(amount), MIN(amount), MAX(amount) FROM orders;
SELECT status, COUNT(*) AS n, SUM(amount) AS revenue FROM orders GROUP BY status ORDER BY revenue DESC;
SELECT user_id, COUNT(*) AS n_orders FROM orders GROUP BY user_id HAVING COUNT(*) >= 2;   -- HAVING filters GROUPS
SELECT user_id,
       COUNT(*) FILTER (WHERE status = 'paid') AS paid_orders,              -- conditional aggregate
       SUM(CASE WHEN status = 'paid' THEN amount ELSE 0 END) AS paid_revenue   -- portable equivalent
FROM orders GROUP BY user_id;
SELECT DATE_TRUNC('month', created_at) AS month, SUM(amount) FROM orders GROUP BY 1 ORDER BY 1;   -- GROUP BY / ORDER BY position
SELECT STRING_AGG(name, ', ' ORDER BY name) AS names, ARRAY_AGG(id) AS ids FROM users;
SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY amount) AS median, STDDEV(amount), VARIANCE(amount) FROM orders;
SELECT u.country, o.status, SUM(o.amount) FROM orders o JOIN users u ON u.id = o.user_id
GROUP BY ROLLUP (u.country, o.status);               -- subtotals per country plus a grand total (also: CUBE, GROUPING SETS)
SELECT BOOL_AND(amount > 0) AS all_positive, BOOL_OR(status = 'refunded') AS any_refund FROM orders;
```

**Watch out:** `WHERE` filters rows before grouping and `HAVING` filters groups after it. Every selected column must either appear in `GROUP BY` or be inside an aggregate.

**Drills:** (1) Revenue and order count per status. (2) Users with more than $200 of paid orders. (3) The median order amount per month.

### 10.04 Joins

```sql
SELECT u.name, o.amount FROM users u INNER JOIN orders o ON o.user_id = u.id;   -- only matching rows
SELECT u.name, o.amount FROM users u LEFT JOIN orders o ON o.user_id = u.id;    -- every user; NULL amount for Dee
SELECT u.name, o.id FROM users u RIGHT JOIN orders o ON o.user_id = u.id;
SELECT u.name, o.id FROM users u FULL OUTER JOIN orders o ON o.user_id = u.id;
SELECT u.name, o.id FROM users u CROSS JOIN orders o;                           -- every pair: 5 × 7 = 35 rows
SELECT u.name FROM users u LEFT JOIN orders o ON o.user_id = u.id WHERE o.id IS NULL;           -- ANTI-join: users with no orders
SELECT u.name FROM users u WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.user_id = u.id);    -- same idea, usually faster
SELECT u.name FROM users u WHERE EXISTS (SELECT 1 FROM orders o WHERE o.user_id = u.id AND o.amount > 100);   -- SEMI-join
SELECT a.name, b.name FROM users a JOIN users b ON a.country = b.country AND a.id < b.id;       -- self-join: pairs per country
SELECT u.name, o.amount FROM users u LEFT JOIN orders o ON o.user_id = u.id AND o.status = 'paid';
-- ^ condition in ON keeps users without paid orders; the same condition in WHERE would drop them
SELECT u.name, latest.amount, latest.created_at
FROM users u
LEFT JOIN LATERAL (SELECT amount, created_at FROM orders o WHERE o.user_id = u.id ORDER BY created_at DESC LIMIT 1) latest ON TRUE;   -- per-row subquery
```

**Drills:** (1) List every user with their total spend, including users who never ordered (show 0). (2) Find users who never ordered, two ways. (3) Show each user's most recent order.

### 10.05 Subqueries, CTEs & set operations

```sql
SELECT * FROM orders WHERE amount > (SELECT AVG(amount) FROM orders);   -- scalar subquery
SELECT name FROM users WHERE id IN (SELECT user_id FROM orders WHERE status = 'refunded');
SELECT o.* FROM orders o
WHERE o.amount = (SELECT MAX(o2.amount) FROM orders o2 WHERE o2.user_id = o.user_id);   -- correlated: each user's largest order
SELECT t.user_id, t.total
FROM (SELECT user_id, SUM(amount) AS total FROM orders GROUP BY user_id) AS t
WHERE t.total > 100;                                -- derived table
WITH user_totals AS (
  SELECT user_id, SUM(amount) AS total FROM orders WHERE status = 'paid' GROUP BY user_id
), ranked AS (
  SELECT *, RANK() OVER (ORDER BY total DESC) AS rnk FROM user_totals
)
SELECT u.name, r.total, r.rnk FROM ranked r JOIN users u ON u.id = r.user_id ORDER BY r.rnk;   -- CTEs read top to bottom
WITH RECURSIVE nums(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM nums WHERE n < 10)
SELECT * FROM nums;                                 -- recursive CTE (hierarchies, sequences)
SELECT country FROM users UNION SELECT 'FR';        -- union, duplicates removed
SELECT user_id FROM orders UNION ALL SELECT id FROM users;   -- keeps duplicates (faster)
SELECT id FROM users INTERSECT SELECT user_id FROM orders;   -- users with at least one order
SELECT id FROM users EXCEPT SELECT user_id FROM orders;      -- users with none
```

**Watch out:** `NOT IN (subquery)` returns no rows if the subquery yields a NULL. Prefer `NOT EXISTS`.

**Drills:** (1) Orders above their user's average order. (2) Rewrite a nested subquery as two CTEs. (3) Generate the numbers 1–12 with a recursive CTE.

### 10.06 Window functions

```sql
SELECT id, user_id, amount,
       ROW_NUMBER()   OVER (PARTITION BY user_id ORDER BY created_at) AS order_n,   -- 1, 2, 3 within each user
       RANK()         OVER (ORDER BY amount DESC) AS rnk,          -- ties share a rank, then a gap
       DENSE_RANK()   OVER (ORDER BY amount DESC) AS dense_rnk,    -- ties share a rank, no gap
       NTILE(4)       OVER (ORDER BY amount) AS quartile,
       PERCENT_RANK() OVER (ORDER BY amount) AS pct_rank
FROM orders;
SELECT id, user_id, created_at, amount,
       LAG(amount)  OVER w AS prev_amount,                         -- previous row in the window
       LEAD(amount) OVER w AS next_amount,
       amount - LAG(amount) OVER w AS change,
       FIRST_VALUE(amount) OVER w AS first_amount,
       SUM(amount) OVER (PARTITION BY user_id ORDER BY created_at ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_total,
       AVG(amount) OVER (PARTITION BY user_id ORDER BY created_at ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS moving_avg_3,
       SUM(amount) OVER (PARTITION BY user_id) AS user_total,      -- no ORDER BY: the whole partition
       amount / SUM(amount) OVER () AS share_of_all                -- empty OVER (): the whole table
FROM orders
WINDOW w AS (PARTITION BY user_id ORDER BY created_at);
WITH ranked AS (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY amount DESC) AS rn FROM orders
)
SELECT * FROM ranked WHERE rn = 1;                  -- top-N per group: window functions can't go in WHERE, so wrap them
-- DuckDB / Snowflake / BigQuery shortcut: SELECT * FROM orders QUALIFY ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY amount DESC) = 1
-- LAST_VALUE needs a full frame: LAST_VALUE(x) OVER (... ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)
```

**Drills:** (1) Running total of revenue by date. (2) Each user's second order. (3) Day-over-day change in daily revenue.

### 10.07 String functions

```sql
SELECT LENGTH(name), UPPER(name), LOWER(name), INITCAP('hello world'), TRIM('  x  '), LTRIM('xxa', 'x'), RTRIM('a  ') FROM users;
SELECT SUBSTRING(name FROM 1 FOR 2), LEFT(name, 2), RIGHT(name, 1), POSITION('e' IN name), STRPOS(name, 'e') FROM users;
SELECT REPLACE('a-b-c', '-', '_'), SPLIT_PART('2024-03-15', '-', 2), CONCAT('a', NULL, 'b'), CONCAT_WS('/', 'data', 'raw', 'x.csv');
-- CONCAT skips NULLs ('ab') while || returns NULL
SELECT LPAD('7', 3, '0'), RPAD('ab', 4, '.'), REVERSE('abc'), REPEAT('ab', 3), FORMAT('%s has %s orders', 'Ana', 2);
SELECT REGEXP_REPLACE('order-123-x', '[^0-9]', '', 'g') AS digits,
       'abc123' ~ '^[a-z]+[0-9]+$' AS matches,          -- ~ regex match (~* case-insensitive)
       SUBSTRING('run_042' FROM '[0-9]+') AS run_id;
SELECT name FROM users WHERE name ILIKE 'a%' OR name ~* '^b';
```

**Drills:** (1) Extract the domain from emails. (2) Zero-pad ids to 6 digits. (3) Keep only the digits of a messy phone-number column.

### 10.08 Dates & times

```sql
SELECT CURRENT_DATE, CURRENT_TIMESTAMP, NOW();
SELECT DATE_TRUNC('month', created_at), EXTRACT(YEAR FROM created_at), EXTRACT(DOW FROM created_at), DATE_PART('hour', created_at) FROM orders;
SELECT created_at + INTERVAL '7 days', created_at - INTERVAL '1 hour', created_at::DATE FROM orders;
SELECT AGE(DATE '2024-03-15', DATE '2024-01-05') AS age_interval, DATE '2024-03-15' - DATE '2024-01-05' AS days_between;
SELECT TO_CHAR(created_at, 'YYYY-MM-DD HH24:MI'), TO_DATE('15/03/2024', 'DD/MM/YYYY'), TO_TIMESTAMP(1700000000) FROM orders;
SELECT * FROM orders WHERE created_at >= DATE '2024-02-01' AND created_at < DATE '2024-03-01';   -- half-open range = all of February
SELECT d::DATE AS day FROM GENERATE_SERIES(DATE '2024-01-01', DATE '2024-01-07', INTERVAL '1 day') AS d;   -- a date spine
SELECT created_at AT TIME ZONE 'UTC' AT TIME ZONE 'America/New_York' AS local_time FROM orders;
```

**Drills:** (1) Orders per weekday. (2) Days from signup to first order per user. (3) Revenue for the last 30 days relative to a fixed date.

### 10.09 Math & numeric functions

```sql
SELECT ROUND(3.14159, 2), CEIL(2.1), FLOOR(2.9), ABS(-5), POWER(2, 10), SQRT(16), MOD(17, 5), 17 % 5, LN(10), LOG(100), EXP(1);
SELECT SIGN(-3), TRUNC(2.789, 1), RANDOM();
SELECT ROUND(AVG(amount), 2) AS avg_amount,
       ROUND(100.0 * COUNT(*) FILTER (WHERE status = 'paid') / COUNT(*), 1) AS pct_paid   -- 100.0 forces decimal division
FROM orders;
SELECT WIDTH_BUCKET(amount, 0, 250, 5) AS bucket, COUNT(*) FROM orders GROUP BY 1 ORDER BY 1;   -- histogram bins
SELECT CORR(amount, EXTRACT(EPOCH FROM created_at)) AS corr, REGR_SLOPE(amount, EXTRACT(EPOCH FROM created_at)) AS slope FROM orders;
```

**Drills:** (1) The percentage of refunded orders, rounded to 1 decimal. (2) A 10-bin histogram of order amounts. (3) Z-scores of amounts using window `AVG` and `STDDEV`.

### 10.10 Creating & changing tables (DDL)

```sql
CREATE TABLE products (
  id         BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,   -- auto-increment (older Postgres: SERIAL)
  sku        TEXT UNIQUE NOT NULL,
  name       VARCHAR(200) NOT NULL,
  price      NUMERIC(10, 2) NOT NULL CHECK (price > 0),
  tags       TEXT[] DEFAULT '{}',
  attrs      JSONB,
  in_stock   BOOLEAN DEFAULT TRUE,
  created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE TABLE order_items (
  order_id   INTEGER REFERENCES orders(id) ON DELETE CASCADE,   -- delete items when their order is deleted
  product_id BIGINT REFERENCES products(id),
  qty        INTEGER NOT NULL DEFAULT 1,
  PRIMARY KEY (order_id, product_id)                             -- composite key
);
ALTER TABLE users ADD COLUMN email TEXT;
ALTER TABLE users RENAME COLUMN email TO contact_email;
ALTER TABLE users DROP COLUMN contact_email;
ALTER TABLE users ALTER COLUMN country SET DEFAULT 'US';
ALTER TABLE products ALTER COLUMN price TYPE NUMERIC(12, 2);
ALTER TABLE users ADD CONSTRAINT users_name_unique UNIQUE (name);
CREATE TABLE big_orders AS SELECT * FROM orders WHERE amount > 100;   -- table from a query
CREATE TEMP TABLE scratch (x INT);                                    -- gone when the session ends
CREATE SCHEMA analytics;
CREATE TABLE analytics.daily (day DATE PRIMARY KEY, revenue NUMERIC);
TRUNCATE TABLE scratch;
DROP TABLE IF EXISTS scratch;
DROP TABLE big_orders;
-- common types: INTEGER BIGINT SMALLINT · NUMERIC(p, s) exact (money) · REAL DOUBLE PRECISION · TEXT VARCHAR(n) · BOOLEAN
--               DATE TIMESTAMP TIMESTAMPTZ INTERVAL · UUID · JSONB · arrays like INTEGER[]
```

**Drills:** (1) Design an `experiments` table (id, name, config JSONB, metrics JSONB, created_at). (2) Add a NOT NULL column with a default to an existing table. (3) Create a table from the result of a query.

### 10.11 Inserting, updating, deleting & upserting (DML)

```sql
INSERT INTO products (sku, name, price) VALUES ('A-1', 'GPU', 1999.00), ('B-2', 'Cable', 9.50);
INSERT INTO products (sku, name, price) VALUES ('C-3', 'SSD', 120.00) RETURNING id, created_at;   -- get generated values back
INSERT INTO analytics.daily (day, revenue)
SELECT created_at::DATE, SUM(amount) FROM orders GROUP BY 1;     -- insert the result of a query
UPDATE orders SET status = 'paid' WHERE id = 7;
UPDATE products SET price = price * 0.9, in_stock = FALSE WHERE sku = 'B-2' RETURNING *;
UPDATE orders o SET amount = o.amount * 1.1
FROM users u WHERE u.id = o.user_id AND u.country = 'UK';       -- update using another table
INSERT INTO products (sku, name, price) VALUES ('A-1', 'GPU v2', 2099.00)
ON CONFLICT (sku) DO UPDATE SET name = EXCLUDED.name, price = EXCLUDED.price;   -- UPSERT; EXCLUDED = the row you tried to insert
INSERT INTO products (sku, name, price) VALUES ('A-1', 'duplicate', 1.00) ON CONFLICT (sku) DO NOTHING;
MERGE INTO products p
USING (VALUES ('B-2', 8.99), ('D-4', 15.00)) AS s(sku, price) ON p.sku = s.sku
WHEN MATCHED THEN UPDATE SET price = s.price
WHEN NOT MATCHED THEN INSERT (sku, name, price) VALUES (s.sku, 'new item', s.price);   -- standard SQL upsert (Postgres 15+)
DELETE FROM orders WHERE status = 'refunded';
DELETE FROM products WHERE NOT EXISTS (SELECT 1 FROM order_items oi WHERE oi.product_id = products.id) AND in_stock = FALSE;
```

**Watch out:** Run an `UPDATE` or `DELETE` as a `SELECT` with the same `WHERE` first, or do it inside a transaction (10.12). A missing `WHERE` changes every row.

**Drills:** (1) Insert 3 rows and return their ids. (2) Apply a 10% discount to products never ordered. (3) Upsert daily revenue so re-running the job doesn't duplicate rows.

### 10.12 Transactions, views & indexes

```sql
BEGIN;
UPDATE orders SET amount = amount - 10 WHERE id = 1;
UPDATE orders SET amount = amount + 10 WHERE id = 2;
COMMIT;                                               -- both changes or neither (ROLLBACK undoes everything since BEGIN)
BEGIN;
SAVEPOINT before_delete;
DELETE FROM orders;
ROLLBACK TO SAVEPOINT before_delete;                  -- undo only back to the savepoint
COMMIT;
CREATE VIEW paid_orders AS SELECT * FROM orders WHERE status = 'paid';   -- a saved query
SELECT * FROM paid_orders WHERE amount > 50;
CREATE MATERIALIZED VIEW monthly_revenue AS
  SELECT DATE_TRUNC('month', created_at) AS month, SUM(amount) AS revenue FROM orders GROUP BY 1;   -- a stored result
REFRESH MATERIALIZED VIEW monthly_revenue;
CREATE INDEX idx_orders_user ON orders (user_id);
CREATE INDEX idx_orders_user_time ON orders (user_id, created_at DESC);   -- multi-column: column order matters
CREATE UNIQUE INDEX idx_users_lower_name ON users (LOWER(name));          -- expression index
CREATE INDEX idx_orders_paid ON orders (created_at) WHERE status = 'paid'; -- partial index
DROP INDEX idx_orders_paid;
```

**Drills:** (1) Move $20 between two orders atomically, then deliberately roll back. (2) Create a view of each user's lifetime value. (3) Index the columns your most frequent filter uses.

### 10.13 JSON & arrays

```sql
CREATE TABLE events (id SERIAL PRIMARY KEY, payload JSONB);
INSERT INTO events (payload) VALUES ('{"user": "ana", "action": "click", "meta": {"x": 10, "tags": ["a", "b"]}}');
SELECT payload->'meta' AS meta_json,                 -- -> returns JSON
       payload->>'user' AS user_text,                -- ->> returns text
       payload#>>'{meta,x}' AS x,                    -- path lookup
       payload->'meta'->'tags'->>0 AS first_tag      -- JSON arrays are 0-indexed
FROM events;
SELECT * FROM events WHERE payload @> '{"action": "click"}';   -- containment (fast with a GIN index)
SELECT * FROM events WHERE payload ? 'meta';                   -- key exists
SELECT JSONB_BUILD_OBJECT('id', id, 'name', name) FROM users;
SELECT JSONB_AGG(JSONB_BUILD_OBJECT('id', id, 'amount', amount)) FROM orders;   -- rows -> one JSON array
SELECT e.id, t.tag FROM events e, JSONB_ARRAY_ELEMENTS_TEXT(e.payload->'meta'->'tags') AS t(tag);   -- one row per array element
SELECT ARRAY[1, 2, 3] AS arr, (ARRAY[10, 20, 30])[1] AS first, ARRAY_LENGTH(ARRAY[1, 2], 1) AS len, 2 = ANY(ARRAY[1, 2, 3]) AS has_two;   -- SQL arrays are 1-indexed
SELECT UNNEST(ARRAY['a', 'b', 'c']) AS letter;
CREATE INDEX idx_events_payload ON events USING GIN (payload);
```

**Drills:** (1) Store model configs as JSONB and query runs where `lr` = 0.001. (2) Flatten a JSON array of tags into rows and count each tag. (3) Return a user's orders as one JSON array.

### 10.14 Query performance

```sql
EXPLAIN SELECT * FROM orders WHERE user_id = 3;          -- the plan: Seq Scan vs Index Scan, estimated costs
EXPLAIN ANALYZE SELECT * FROM orders WHERE user_id = 3;  -- actually runs it: real time and row counts
ANALYZE orders;                                          -- refresh the planner's statistics
SELECT * FROM orders WHERE created_at >= DATE '2024-03-01';                    -- index-friendly
SELECT * FROM orders WHERE DATE_TRUNC('month', created_at) = DATE '2024-03-01'; -- function on the column: a plain index can't be used
```

Habits that keep queries fast:
- Select only the columns you need.
- Filter as early as possible.
- Index the columns you join and filter on.
- Use `EXISTS` for existence checks.
- Don't hide duplicate rows from a bad join with `DISTINCT`.
- Always `LIMIT` exploratory queries on big tables.

**Drills:** (1) Compare `EXPLAIN ANALYZE` before and after adding an index. (2) Rewrite a non-sargable date filter as a range. (3) Spot the sequential scan in a slow join.

### 10.15 SQL from Python & the command line

```python
import sqlite3
import pandas as pd
con = sqlite3.connect("shop.db")                 # a file database; ":memory:" for a throwaway one
cur = con.cursor()
cur.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT, country TEXT)")
cur.executemany("INSERT INTO users (name, country) VALUES (?, ?)", [("Ana", "US"), ("Ben", "UK"), ("Cal", "US")])
con.commit()
cur.execute("SELECT * FROM users WHERE country = ?", ("US",))   # ALWAYS use parameters: f-strings with user input = SQL injection
rows = cur.fetchall()
df = pd.read_sql("SELECT country, COUNT(*) AS n FROM users GROUP BY country", con)
df.to_sql("country_counts", con, if_exists="replace", index=False)
con.close()
# PostgreSQL:  pip install "psycopg[binary]" sqlalchemy
#   import psycopg; conn = psycopg.connect("postgresql://user:pass@localhost:5432/shop"); conn.execute("SELECT 1").fetchone()
#   from sqlalchemy import create_engine, text; engine = create_engine("postgresql+psycopg://user:pass@localhost/shop")
#   pd.read_sql(text("SELECT * FROM orders WHERE amount > :min"), engine, params={"min": 50})
# DuckDB (no server; SQL straight on files and DataFrames):
#   import duckdb; duckdb.sql("SELECT status, SUM(amount) FROM 'orders.parquet' GROUP BY status").df()
#   duckdb.sql("SELECT * FROM df WHERE amount > 50").df()      # df = a pandas DataFrame in scope
```

```bash
psql -h localhost -U ada -d shop                 # inside psql: \l databases · \c db · \dt tables · \d orders · \x expanded · \timing · \i file.sql · \q
psql -d shop -c "SELECT COUNT(*) FROM orders"; psql -d shop -f schema.sql
psql -d shop -c "\copy orders TO 'orders.csv' CSV HEADER"   # export a table to CSV
sqlite3 shop.db                                  # inside: .tables · .schema users · .headers on · .mode column · .mode csv · .import data.csv users · .quit
duckdb -c "SELECT * FROM 'data.csv' LIMIT 5"
```

**Drills:** (1) Load a CSV into SQLite with pandas and query it with SQL. (2) Write a parameterized query function that is safe from injection. (3) Query a Parquet file with DuckDB without loading it into pandas.

### 10.16 Analytics & ML patterns

```sql
-- 1) deduplicate: keep the latest row per key
WITH ranked AS (SELECT *, ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY created_at DESC) AS rn FROM orders)
SELECT * FROM ranked WHERE rn = 1;
-- 2) deterministic train / val / test split by hashing an id (same split every run)
SELECT *, CASE WHEN ABS(HASHTEXT(id::TEXT)) % 10 < 8 THEN 'train'
               WHEN ABS(HASHTEXT(id::TEXT)) % 10 = 8 THEN 'val'
               ELSE 'test' END AS split
FROM users;
-- 3) features from PAST rows only (no leakage)
SELECT id, user_id, created_at, amount,
       COUNT(*) OVER (PARTITION BY user_id ORDER BY created_at ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prior_orders,
       AVG(amount) OVER (PARTITION BY user_id ORDER BY created_at ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) AS prev3_avg_amount,
       created_at - LAG(created_at) OVER (PARTITION BY user_id ORDER BY created_at) AS time_since_prev
FROM orders;
-- 4) a label: did the user order again within 30 days?
SELECT o.id,
       EXISTS (SELECT 1 FROM orders o2
               WHERE o2.user_id = o.user_id AND o2.created_at > o.created_at
                 AND o2.created_at <= o.created_at + INTERVAL '30 days') AS reordered_30d
FROM orders o;
-- 5) class balance, and long -> wide pivot with conditional aggregation
SELECT status, COUNT(*) AS n, ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct FROM orders GROUP BY status;
SELECT user_id,
       SUM(amount) FILTER (WHERE status = 'paid') AS paid,
       SUM(amount) FILTER (WHERE status = 'pending') AS pending
FROM orders GROUP BY user_id;
-- 6) random and stratified samples
SELECT * FROM orders ORDER BY RANDOM() LIMIT 3;
SELECT * FROM orders TABLESAMPLE BERNOULLI (50);      -- ~50% of rows, cheap on big tables
WITH r AS (SELECT *, ROW_NUMBER() OVER (PARTITION BY status ORDER BY RANDOM()) AS rn FROM orders)
SELECT * FROM r WHERE rn <= 1;                        -- one random row per status
-- 7) monthly cohort activity
WITH firsts AS (SELECT user_id, DATE_TRUNC('month', MIN(created_at)) AS cohort FROM orders GROUP BY user_id)
SELECT f.cohort, DATE_TRUNC('month', o.created_at) AS month, COUNT(DISTINCT o.user_id) AS active_users
FROM orders o JOIN firsts f USING (user_id)
GROUP BY 1, 2 ORDER BY 1, 2;
-- 8) fill missing days with zeros via a date spine
SELECT d::DATE AS day, COALESCE(SUM(o.amount), 0) AS revenue
FROM GENERATE_SERIES(DATE '2024-03-01', DATE '2024-03-07', INTERVAL '1 day') AS d
LEFT JOIN orders o ON o.created_at::DATE = d::DATE
GROUP BY 1 ORDER BY 1;
```

**Drills:** (1) Build a leakage-free feature table (prior order count, prior average spend) and a 30-day reorder label in one query. (2) Produce a reproducible 80/10/10 split. (3) Compute weekly revenue with zero-filled gaps.

Full reference: https://www.postgresql.org/docs/current/

## 11. LaTeX

LaTeX is the standard for ML papers, theses and math-heavy notes. You write markup in a `.tex` file and compile it to a PDF. The quickest start is Overleaf in a browser. Locally, install TeX Live (Linux/Windows) or MacTeX (macOS) and run `latexmk -pdf main.tex`. Blocks starting with `% preamble` go before `\begin{document}`; everything else goes in the body.

### 11.01 Document skeleton & compiling

```latex
\documentclass[11pt, a4paper]{article}   % also: report, book, beamer (slides), memoir
\usepackage[margin=1in]{geometry}
\usepackage{amsmath, amssymb, amsthm, mathtools}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{hyperref}
\title{Attention Is All I Practiced}
\author{Ada Lovelace \and Alan Turing}
\date{\today}
\begin{document}
\maketitle
\begin{abstract}
A one-paragraph summary of the paper.
\end{abstract}
\tableofcontents
\section{Introduction}\label{sec:intro}
Hello, \LaTeX!
\end{document}
```

```bash
pdflatex main.tex && pdflatex main.tex           # run twice so references and the table of contents resolve
latexmk -pdf main.tex                            # reruns everything (including BibTeX) as often as needed
latexmk -pdf -pvc main.tex                       # watch mode: recompile on every save
latexmk -c                                       # delete auxiliary files (.aux .log .out ...)
xelatex main.tex; lualatex main.tex              # Unicode input and system fonts (fontspec)
```

**Drills:** (1) Compile the skeleton, then add an abstract and a second section. (2) Set up `latexmk -pvc` and watch it recompile. (3) Recreate it in Overleaf.

### 11.02 Structure

```latex
\section{Method}\label{sec:method}
\subsection{Data}
\subsubsection{Cleaning}
\paragraph{Note.} Run-in heading for a short paragraph.
\section*{Acknowledgments}   % starred = unnumbered and not in the table of contents
\appendix
\section{Extra results}      % numbered A, B, ... after \appendix
% \chapter{...} exists in the report/book classes · \input{sections/method} pastes a file here · \include{ch1} starts a new page
% \newpage starts a new page · \clearpage also flushes pending figures and tables
```

**Drills:** (1) Split a paper into `sections/*.tex` files pulled in with `\input`. (2) Add an unnumbered acknowledgments section. (3) Add an appendix with two sections.

### 11.03 Text formatting, special characters & spacing

```latex
\textbf{bold} \textit{italic} \emph{emphasis} \underline{underline} \texttt{monospace} \textsc{Small Caps} \textsf{sans serif}
{\small small} {\large large} {\Large Larger} {\huge huge}   % also \tiny \scriptsize \footnotesize \normalsize \LARGE \Huge
Reserved characters must be escaped: \% \$ \& \# \_ \{ \} \textbackslash{} \textasciitilde{} \textasciicircum{}
``double quotes'' and `single quotes' --- hyphen -, en dash 1--10, em dash ---, ellipsis \ldots
Non-breaking space: Figure~1, Dr.~Smith. Spaces: a\,b a\quad b a\qquad b
Line break\\ A blank line starts a new paragraph.

\noindent No indent here. \hfill Right-aligned. \vspace{1em}
\begin{center} Centered text. \end{center}
\begin{flushright} Right-aligned block. \end{flushright}
A footnote.\footnote{Footnote text.} A link: \url{https://pytorch.org} or \href{https://numpy.org}{NumPy}.
Inline code: \verb|x = np.zeros(3)|
\begin{verbatim}
for i in range(3):
    print(i)
\end{verbatim}
```

**Drills:** (1) Write a sentence containing `50%`, `$5`, `R&D` and `snake_case` that compiles. (2) Use correct quotes and dashes in a page range. (3) Add a footnote with a link.

### 11.04 Lists

```latex
\begin{itemize}
  \item First point
  \item Second point
  \begin{itemize}
    \item Nested point
  \end{itemize}
\end{itemize}
\begin{enumerate}
  \item Load data
  \item Train
  \item Evaluate
\end{enumerate}
\begin{description}
  \item[Epoch] One pass over the training data.
  \item[Batch] The samples processed together in one step.
\end{description}
\begin{enumerate}[label=(\alph*), noitemsep]   % enumitem: custom labels and spacing
  \item first
  \item second
\end{enumerate}
\begin{itemize}[leftmargin=*, itemsep=2pt]
  \item compact list
\end{itemize}
```

**Drills:** (1) Make a numbered list with (a), (b), (c) labels. (2) Nest a bullet list inside a numbered list. (3) Write a glossary with `description`.

### 11.05 Math mode basics

```latex
Inline math: $f(x) = x^2$ or \(a + b\). Display math:
\[ \hat{y} = \sigma(Wx + b) \]
\begin{equation}\label{eq:nll}
  \mathcal{L}(\theta) = -\frac{1}{N} \sum_{i=1}^{N} \log p_\theta(y_i \mid x_i)
\end{equation}
Equation~\eqref{eq:nll} is the negative log-likelihood.
\begin{equation*} E = mc^2 \end{equation*}
$x^2$, $x_i$, $x_i^2$, $x^{10}$, $x_{i,j}$, $e^{-x^2/2}$, $\frac{a}{b}$, $\dfrac{a}{b}$, $\tfrac{1}{2}$, $\sqrt{x}$, $\sqrt[3]{x}$
$\sum_{i=1}^{n} x_i$, $\prod_{k} p_k$, $\int_0^\infty f(x)\,dx$, $\lim_{x \to 0} f(x)$, $\max_{\theta} J(\theta)$, $\frac{\partial L}{\partial w}$, $\nabla_\theta L$
$\text{if } x > 0$, $\mathrm{d}x$, $\operatorname{softmax}(z)$, $\mathbb{E}_{x \sim p}[f(x)]$, $\|x\|_2$, $|x|$, $\langle u, v \rangle$
$\hat{y}$ $\bar{x}$ $\tilde{x}$ $\vec{v}$ $\dot{x}$ $\mathbf{W}$ $\boldsymbol{\theta}$ $\mathbb{R}^{d}$ $\mathcal{D}$ $x^\top$
```

**Drills:** (1) Typeset the MSE loss inline and as a numbered equation. (2) Reference the equation in a sentence. (3) Write an expectation over a data distribution.

### 11.06 Symbols reference

```latex
$\alpha \beta \gamma \delta \epsilon \varepsilon \zeta \eta \theta \kappa \lambda \mu \nu \xi \pi \rho \sigma \tau \phi \varphi \chi \psi \omega$
$\Gamma \Delta \Theta \Lambda \Xi \Pi \Sigma \Phi \Psi \Omega$
$\le \ge \ne \approx \equiv \sim \simeq \propto \ll \gg \pm \mp \times \cdot \div \circ \odot \otimes \oplus$
$\in \notin \subset \subseteq \cup \cap \emptyset \forall \exists \neg \land \lor \implies \iff$
$\to \leftarrow \gets \Rightarrow \Leftrightarrow \mapsto \uparrow \downarrow$
$\infty \partial \nabla \top \perp \ell \prime \dagger \ldots \cdots \vdots \ddots$
$\sin x \cos x \log x \ln x \exp x \det A \min \max \sup \inf \arg \operatorname{tr}(A)$
```

**Drills:** (1) Type every lowercase Greek letter from memory. (2) Write "for all x in R, there exists y such that y > x". (3) Write the Hadamard product of two matrices.

### 11.07 Multi-line equations & numbering

```latex
\begin{align}
  \nabla_W L &= X^\top (\hat{Y} - Y) \label{eq:grad} \\
  W &\gets W - \eta \nabla_W L \nonumber
\end{align}
\begin{align*}
  f(x) &= (x + 1)^2 \\
       &= x^2 + 2x + 1 && \text{expand}
\end{align*}
\begin{gather*}
  a = b + c \\
  d = e
\end{gather*}
\begin{multline*}
  p(x) = a_0 + a_1 x + a_2 x^2 + a_3 x^3 + a_4 x^4 \\
  + a_5 x^5 + a_6 x^6
\end{multline*}
\begin{equation}
  \begin{split}
    (a + b)^2 &= a^2 + 2ab + b^2 \\
              &\ge 4ab
  \end{split}
\end{equation}
```

**Drills:** (1) Derive the gradient of MSE in 3 aligned steps with one equation number. (2) Annotate a step with `&& \text{...}`. (3) Break a long equation with `multline`.

### 11.08 Matrices, cases, brackets & custom operators

```latex
% preamble
\DeclareMathOperator*{\argmax}{arg\,max}
\DeclareMathOperator*{\argmin}{arg\,min}
\DeclareMathOperator{\softmax}{softmax}
\newcommand{\R}{\mathbb{R}}
\newcommand{\E}{\mathbb{E}}
\newcommand{\norm}[1]{\left\lVert #1 \right\rVert}
```

```latex
\[ W = \begin{bmatrix} w_{11} & w_{12} \\ w_{21} & w_{22} \end{bmatrix}, \quad
   \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}, \quad
   \begin{vmatrix} a & b \\ c & d \end{vmatrix}, \quad
   \begin{bmatrix} x_{11} & \cdots & x_{1n} \\ \vdots & \ddots & \vdots \\ x_{m1} & \cdots & x_{mn} \end{bmatrix} \]
\[ \operatorname{ReLU}(x) = \begin{cases} x & \text{if } x > 0 \\ 0 & \text{otherwise} \end{cases} \]
\[ \left( \frac{a}{b} \right), \ \left[ \sum_i x_i \right], \ \left\{ x \mid x > 0 \right\}, \ \bigl( x \bigr), \ \Bigl[ x \Bigr], \ \left. \frac{df}{dx} \right|_{x=0} \]
\[ \underbrace{a + \cdots + a}_{n \text{ times}} \quad \overbrace{x_1, \dots, x_k}^{k} \quad \overset{\text{def}}{=} \quad \xrightarrow{\text{train}} \]
\[ \hat{c} = \argmax_{c} p(c \mid x), \quad \norm{x}_2, \quad \E[X], \quad x \in \R^d \]
```

**Drills:** (1) Write a 3×3 matrix with dots for a general n×n matrix. (2) Define the sign function with `cases`. (3) Define `\argmin` and use it with a subscript under it.

### 11.09 ML equations gallery

```latex
\begin{align}
  \softmax(z)_i &= \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}} \\
  \mathcal{L}_{\text{CE}} &= -\sum_{k=1}^{K} y_k \log \hat{y}_k \\
  \operatorname{Attention}(Q, K, V) &= \softmax\!\left( \frac{Q K^\top}{\sqrt{d_k}} \right) V \\
  \theta_{t+1} &= \theta_t - \eta \, \nabla_\theta \mathcal{L}(\theta_t) \\
  m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t, \qquad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
  \theta_{t+1} &= \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}, \qquad \hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \ \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \\
  D_{\mathrm{KL}}(P \,\|\, Q) &= \sum_x P(x) \log \frac{P(x)}{Q(x)} \\
  \operatorname{LayerNorm}(x) &= \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta \\
  h &= W_0 x + \frac{\alpha}{r} B A x, \qquad B \in \R^{d \times r},\ A \in \R^{r \times k} \\
  \mathcal{L}_{\text{DPO}} &= -\E_{(x, y_w, y_l)} \left[ \log \sigma\!\left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right] \\
  \delta^{(l)} &= \bigl( W^{(l+1)\top} \delta^{(l+1)} \bigr) \odot f'\bigl(z^{(l)}\bigr), \qquad \frac{\partial \mathcal{L}}{\partial W^{(l)}} = \delta^{(l)} \bigl(a^{(l-1)}\bigr)^\top
\end{align}
```

**Drills:** Typeset each line from memory, then compare: softmax, cross-entropy, attention, SGD, Adam, KL, LayerNorm, LoRA, DPO, backprop. These mirror the PyTorch modules, so you are practicing the same ideas in both notations.

### 11.10 Figures

```latex
% preamble: \usepackage{graphicx} \usepackage{subcaption} \usepackage{float}
\begin{figure}[htbp]          % placement: h here, t top, b bottom, p own page, ! try harder; [H] (float package) = exactly here
  \centering
  \includegraphics[width=0.8\linewidth]{example-image}   % your plot: fig.savefig("loss.pdf") -> {loss.pdf}
  \caption{Training and validation loss.}
  \label{fig:loss}            % \label AFTER \caption
\end{figure}
\begin{figure}[t]
  \centering
  \begin{subfigure}{0.48\linewidth}
    \includegraphics[width=\linewidth]{example-image-a}
    \caption{Loss}\label{fig:sub-loss}
  \end{subfigure}\hfill
  \begin{subfigure}{0.48\linewidth}
    \includegraphics[width=\linewidth]{example-image-b}
    \caption{Accuracy}\label{fig:sub-acc}
  \end{subfigure}
  \caption{Two panels side by side.}\label{fig:pair}
\end{figure}
% \includegraphics options: width= height= scale=0.5 angle=90 trim=1cm 0 1cm 0 clip keepaspectratio
```

**Drills:** (1) Insert a Matplotlib PDF at 80% text width with a caption and label. (2) Put two plots side by side with subcaptions. (3) Reference the figure from the text.

### 11.11 Tables

```latex
% preamble: \usepackage{booktabs} \usepackage{multirow} \usepackage{siunitx}
\begin{table}[t]
  \centering
  \caption{Test accuracy (\%) on two benchmarks.}   % table captions go ABOVE the table
  \label{tab:results}
  \begin{tabular}{l c c r}
    \toprule
    Model & CIFAR-10 & ImageNet & Params \\
    \midrule
    ResNet-18 & 94.1 & 69.8 & 11.7M \\
    ViT-B/16  & \textbf{98.1} & \textbf{81.8} & 86M \\
    \bottomrule
  \end{tabular}
\end{table}
\begin{tabular}{|l|p{4cm}|}   % | = vertical rule (avoid with booktabs), p{width} = wrapping column
  \hline
  \multicolumn{2}{|c|}{Header spanning two columns} \\ \hline
  \multirow{2}{*}{Merged} & row 1 \\
                          & row 2 \\ \hline
\end{tabular}
\begin{tabular}{l S[table-format=2.2] S[table-format=1.3]}   % siunitx: align numbers on the decimal point
  \toprule
  Run & {Loss} & {Acc} \\
  \midrule
  a & 12.34 & 0.912 \\
  b & 3.5 & 0.87 \\
  \bottomrule
\end{tabular}
% \cmidrule(lr){2-3} partial rule · \addlinespace · tabularx for full-width tables · \resizebox{\linewidth}{!}{...} shrinks a wide table
% from pandas: df.to_latex(index=False, float_format="%.3f") prints a booktabs-ready table
```

**Drills:** (1) Typeset a results table with booktabs and the best number in bold. (2) Add a column group header with `\multicolumn` and `\cmidrule`. (3) Generate a table from a DataFrame with `to_latex` and paste it in.

### 11.12 Cross-references, links & citations

```latex
% preamble: \usepackage{hyperref} \usepackage{cleveref} (cleveref last) \usepackage{natbib}
See Section~\ref{sec:intro}, Figure~\ref{fig:loss}, Table~\ref{tab:results} and Equation~\eqref{eq:nll} on page~\pageref{fig:loss}.
\Cref{fig:loss} and \cref{tab:results} insert ``Figure''/``table'' automatically.
Transformers~\cite{vaswani2017attention} changed NLP.
\citet{vaswani2017attention} introduced attention-only models \citep{vaswani2017attention}.   % natbib: textual vs parenthetical
\bibliographystyle{plainnat}
\bibliography{references}     % reads references.bib
```

```bibtex
@inproceedings{vaswani2017attention,
  title     = {Attention Is All You Need},
  author    = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N. and Kaiser, {\L}ukasz and Polosukhin, Illia},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2017}
}
@misc{lastname2024arxiv,
  title         = {Paper Title},
  author        = {Lastname, Firstname and Other, Author},
  year          = {2024},
  eprint        = {2401.00001},
  archivePrefix = {arXiv}
}
```

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main   # what latexmk -pdf automates
# biblatex alternative: \usepackage[style=authoryear]{biblatex} \addbibresource{references.bib} ... \printbibliography   (compile with biber)
```

**Drills:** (1) Cite two papers with `\citet` and `\citep`. (2) Copy a BibTeX entry from Google Scholar or arXiv into `references.bib`. (3) Switch to `\cref` everywhere.

### 11.13 Algorithms & code listings

```latex
% preamble: \usepackage{algorithm} \usepackage{algpseudocode} \usepackage{listings}
\begin{algorithm}[t]
\caption{Mini-batch SGD with early stopping}\label{alg:sgd}
\begin{algorithmic}[1]
  \Require dataset $\mathcal{D}$, learning rate $\eta$, epochs $E$
  \Ensure trained parameters $\theta$
  \State initialize $\theta$
  \For{$e = 1$ \textbf{to} $E$}
    \For{each mini-batch $B \subset \mathcal{D}$}
      \State $g \gets \frac{1}{|B|} \sum_{(x, y) \in B} \nabla_\theta \ell(f_\theta(x), y)$
      \State $\theta \gets \theta - \eta g$ \Comment{gradient step}
    \EndFor
    \If{validation loss stopped improving}
      \State \textbf{break}
    \EndIf
  \EndFor
  \State \Return $\theta$
\end{algorithmic}
\end{algorithm}
% also: \While{c} ... \EndWhile · \ForAll{x} · \Else · \ElsIf{c} · \Function{Name}{args} ... \EndFunction
\begin{lstlisting}[language=Python, basicstyle=\ttfamily\small, frame=single, numbers=left]
loss = criterion(model(x), y)
loss.backward()
optimizer.step()
\end{lstlisting}
% minted gives nicer highlighting but needs: pdflatex -shell-escape main.tex
```

**Drills:** (1) Write Adam as an `algorithmic` block. (2) Add a Python listing with line numbers. (3) Reference the algorithm with `\cref`.

### 11.14 Theorems & custom macros

```latex
% preamble
\newtheorem{theorem}{Theorem}[section]
\newtheorem{lemma}[theorem]{Lemma}
\theoremstyle{definition}
\newtheorem{definition}{Definition}
\newenvironment{note}{\par\noindent\textbf{Note.}\ }{\par}
\newcommand{\vect}[1]{\mathbf{#1}}
```

```latex
\begin{definition}[Convexity]
A function $f$ is convex if $f(\lambda x + (1 - \lambda) y) \le \lambda f(x) + (1 - \lambda) f(y)$ for all $\lambda \in [0, 1]$.
\end{definition}
\begin{theorem}\label{thm:gd}
For convex, $L$-smooth $f$ and step size $\eta \le 1/L$, gradient descent converges at rate $O(1/t)$.
\end{theorem}
\begin{proof}
Combine smoothness with convexity and telescope the per-step decrease.
\end{proof}
\begin{note} Macros keep notation consistent: $\vect{x} \in \R^d$, $\norm{\vect{x}}_2$, $\E[X]$. \end{note}
```

**Drills:** (1) Define a `lemma` that shares numbering with theorems. (2) Make a macro `\loss` that prints $\mathcal{L}$. (3) Write a short proof environment.

### 11.15 Page layout, colors & boxes

```latex
% preamble
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\lhead{COMP Guide}
\rhead{\thepage}
\usepackage{xcolor}
\definecolor{brand}{HTML}{1F77B4}
\usepackage{tcolorbox}
\usepackage{setspace}
\setlength{\parskip}{4pt}
```

```latex
\textcolor{brand}{Colored text}, \colorbox{yellow}{highlighted}, {\color{red} a red span}.
\begin{tcolorbox}[colback=blue!5, colframe=brand, title=Key idea]
Gradients accumulate: call \texttt{zero\_grad()} every step.
\end{tcolorbox}
\begin{minipage}{0.45\linewidth} Left column text. \end{minipage}\hfill
\begin{minipage}{0.45\linewidth} Right column text. \end{minipage}
\thispagestyle{empty}
% margins: \usepackage[margin=2.5cm]{geometry} · spacing: \onehalfspacing (setspace) · two columns: \documentclass[twocolumn]{article}
% or \usepackage{multicol} with \begin{multicols}{2} ... \end{multicols} · landscape: \documentclass[landscape]{article}
```

**Drills:** (1) Add a header with your name and page numbers. (2) Make a colored "Key idea" box. (3) Lay out text and a figure side by side with minipages.

### 11.16 TikZ & pgfplots: diagrams and plots

```latex
% preamble
\usepackage{tikz}
\usetikzlibrary{arrows.meta, positioning}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
```

```latex
\begin{tikzpicture}[neuron/.style={circle, draw, minimum size=7mm}, >={Stealth}]
  \foreach \i in {1,2,3} \node[neuron] (x\i) at (0, -\i) {$x_\i$};
  \foreach \j in {1,2} \node[neuron, fill=blue!10] (h\j) at (2, -\j-0.5) {$h_\j$};
  \node[neuron, fill=red!10] (y) at (4, -2) {$\hat{y}$};
  \foreach \i in {1,2,3} \foreach \j in {1,2} \draw[->] (x\i) -- (h\j);
  \foreach \j in {1,2} \draw[->] (h\j) -- (y);
\end{tikzpicture}
\begin{tikzpicture}[box/.style={draw, rounded corners, minimum width=2cm, minimum height=8mm}, >={Stealth}]
  \node[box] (enc) {Encoder};
  \node[box, right=1.5cm of enc] (dec) {Decoder};
  \draw[->, thick] (enc) -- node[above] {$z$} (dec);
  \draw[thick, blue] (0, -1) rectangle (1, -2);
  \fill[red] (2, -1.5) circle (2pt);
\end{tikzpicture}
\begin{tikzpicture}
  \begin{axis}[xlabel=epoch, ylabel=loss, ymode=log, legend pos=north east, width=8cm, height=5cm, grid=major]
    \addplot[blue, mark=*] coordinates {(1, 2.3) (2, 1.1) (3, 0.6) (4, 0.4) (5, 0.35)};
    \addlegendentry{train}
    \addplot[red, dashed, domain=1:5, samples=50] {2.5 * exp(-0.5 * x) + 0.3};
    \addlegendentry{val}
  \end{axis}
\end{tikzpicture}
% plot straight from a CSV written by pandas: \addplot table[x=epoch, y=loss, col sep=comma] {log.csv};
```

**Drills:** (1) Draw a 3-4-2 MLP diagram with labeled layers. (2) Draw an encoder → decoder block diagram. (3) Plot train/val loss from a CSV with pgfplots.

### 11.17 Beamer slides

```latex
\documentclass{beamer}
\usetheme{Madrid}
\title{Muscle-Memory ML}
\author{Ada Lovelace}
\begin{document}
\frame{\titlepage}
\begin{frame}{Outline}
  \tableofcontents
\end{frame}
\section{Tensors}
\begin{frame}{Tensors}
  \begin{itemize}
    \item<1-> Creation
    \item<2-> Broadcasting
    \item<3-> Autograd
  \end{itemize}
\end{frame}
\begin{frame}[fragile]{Code needs the fragile option}
\begin{verbatim}
x = torch.randn(3, requires_grad=True)
\end{verbatim}
\end{frame}
\begin{frame}{Two columns}
  \begin{columns}
    \column{0.5\textwidth} Text on the left. \pause
    \column{0.5\textwidth} \begin{block}{Tip} Reveal step by step with \texttt{\textbackslash pause}. \end{block}
  \end{columns}
\end{frame}
\end{document}
```

**Drills:** (1) Make a 5-slide talk with a title slide and an outline. (2) Reveal bullets one at a time. (3) Put a plot and its takeaway side by side.

### 11.18 Troubleshooting

| Error | Usual cause → fix |
|---|---|
| `Undefined control sequence` | typo in a command, or the package that defines it isn't loaded |
| `Missing $ inserted` | math (`_`, `^`, `\alpha`) used outside `$...$`, or an unescaped `_` in text |
| `Missing \begin{document}` | stray text in the preamble |
| `File 'x.sty' not found` | install the package (`tlmgr install x`) or use Overleaf |
| `Option clash for package x` | the same package loaded twice with different options |
| `??` in the PDF / `Citation undefined` | run BibTeX and compile again (`latexmk` does it) |
| `Overfull \hbox` (warning) | a line is too wide: rephrase, `\allowbreak`, or shrink the table |
| Figure appears pages away | use `[htbp]`, `\FloatBarrier` (placeins), or `[H]` (float) |

Full reference: https://www.overleaf.com/learn

## 12. Docker

Docker packages code together with its exact environment (OS libraries, Python, CUDA, packages), so it runs the same on your laptop, a GPU server and the cloud. The key terms:
- **Image:** a read-only template built from a `Dockerfile`, layer by layer.
- **Container:** a running instance of an image.
- **Registry:** where images are stored and shared (Docker Hub, GHCR).
- **Volume:** storage that persists outside the container.

### 12.01 Setup & first commands

```bash
docker version; docker info                      # client/server versions; is the daemon running?
docker run hello-world                           # pulls a tiny image and runs it
docker --help; docker run --help                 # every subcommand has --help
```

### 12.02 Running containers

```bash
docker run ubuntu:24.04 echo "hi"                # pull if needed, run one command, exit
docker run -it --rm python:3.12-slim bash        # -it interactive terminal · --rm delete the container on exit
docker run -d --name web -p 8080:80 nginx        # -d detached · host port 8080 -> container port 80
docker run --rm -v "$(pwd)":/workspace -w /workspace python:3.12-slim python script.py   # mount the current folder and run in it
docker run --rm -e LOG_LEVEL=debug --env-file .env my-image   # environment variables
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi   # GPUs (needs the NVIDIA Container Toolkit)
docker run --rm --gpus '"device=0,1"' --shm-size=8g my-train-image   # pick GPUs; more shared memory for DataLoader workers
docker run --rm --user "$(id -u):$(id -g)" -v "$(pwd)":/w -w /w python:3.12-slim touch out.txt   # created files owned by you, not root
docker run --rm -it --entrypoint bash my-image   # override the image's entrypoint
# more flags: --cpus 4 · --memory 8g · --restart unless-stopped · --network my-net · --name my-container · --ipc=host
```

**Drills:** (1) Run Python 3.12 interactively without installing it. (2) Run a script from your current folder inside a container. (3) Serve a folder with nginx on port 8080.

### 12.03 Managing containers

```bash
docker ps; docker ps -a; docker ps -q            # running / all / only ids
docker logs web; docker logs -f --tail 100 web   # follow the last 100 lines
docker exec -it web bash                         # a shell inside a RUNNING container (use sh if there's no bash)
docker stop web; docker start web; docker restart web; docker kill web
docker rm web; docker rm -f web                  # remove (force: also stops it)
docker cp web:/etc/nginx/nginx.conf ./; docker cp ./model.pt web:/models/
docker inspect web; docker inspect -f '{{.State.Status}}' web
docker stats; docker top web; docker port web; docker container prune
```

**Drills:** (1) Tail the logs of a detached container. (2) Open a shell inside it and look around. (3) Copy a file out of a container.

### 12.04 Images

```bash
docker images; docker pull python:3.12-slim; docker pull --platform linux/amd64 python:3.12-slim
docker tag my-app:latest ghcr.io/ada/my-app:1.0  # give an image another name (needed before pushing)
docker push ghcr.io/ada/my-app:1.0
docker rmi my-app:old; docker image prune; docker image prune -a
docker history my-app; docker image inspect my-app   # layers and their sizes / full metadata
docker save my-app:1.0 -o my-app.tar; docker load -i my-app.tar   # move an image without a registry
# docker commit <container> <image> exists, but it isn't reproducible: write a Dockerfile instead
```

### 12.05 Dockerfile instructions

```dockerfile
# Base image: pin a tag (or a digest) so builds are reproducible. "AS" names the stage.
FROM python:3.12-slim AS base
# Build-time variable: docker build --build-arg APP_VERSION=1.0 .
ARG APP_VERSION=dev
# Runtime environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
LABEL org.opencontainers.image.source="https://github.com/ada/comp-guide"
# Working directory (created if missing); later relative paths start here
WORKDIR /app
# One RUN = one layer: install and clean up in the same layer
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
# Copy the dependency list FIRST so this layer stays cached when only your code changes
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
# Don't run as root
RUN useradd --create-home appuser
USER appuser
# Documentation only: publish with docker run -p 8000:8000
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
# ENTRYPOINT = the fixed executable; CMD = default arguments (replaced by anything after the image name in docker run)
ENTRYPOINT ["python"]
CMD ["train.py", "--epochs", "10"]
# Also available: ADD (COPY plus URLs/auto-extracting tars) · VOLUME · SHELL · STOPSIGNAL · ONBUILD
```

**Watch out:** In a Dockerfile, `#` only starts a comment at the beginning of a line. Trailing text after an instruction becomes part of its arguments.

**Drills:** (1) Write a Dockerfile for a script with a `requirements.txt`. (2) Reorder its lines so a code change doesn't reinstall dependencies. (3) Make it run as a non-root user.

### 12.06 Building images & .dockerignore

```bash
docker build -t my-app:1.0 .                     # "." is the build context: everything in it is sent to the builder
docker build -t my-app:dev -f docker/Dockerfile.dev .
docker build --build-arg APP_VERSION=1.0 --target base -t my-app:base .   # stop at a named stage
docker build --no-cache --progress=plain -t my-app .   # rebuild from scratch, show full output
docker build --platform linux/amd64 -t my-app .  # build for x86 servers from an Apple Silicon Mac
docker buildx build --platform linux/amd64,linux/arm64 -t ghcr.io/ada/my-app:1.0 --push .   # multi-architecture
docker builder prune                             # clear the build cache
```

```text
# .dockerignore: keep the build context small and secrets out of images
.git
.venv
venv
__pycache__
*.pyc
data/
checkpoints/
*.pt
wandb/
.env
```

### 12.07 Best practices & multi-stage builds

```dockerfile
# syntax=docker/dockerfile:1
# Stage 1: build the environment with all the tools
FROM python:3.12-slim AS builder
WORKDIR /app
RUN pip install uv
COPY pyproject.toml uv.lock ./
# BuildKit cache mount: dependency downloads are reused across builds
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev
COPY src ./src

# Stage 2: a slim runtime image that only receives the finished environment
FROM python:3.12-slim AS runtime
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src ./src
ENV PATH="/app/.venv/bin:$PATH"
RUN useradd -m app
USER app
CMD ["python", "-m", "src.serve"]
```

Best practices:
- Order layers from least to most frequently changed.
- Pin base-image tags.
- Prefer `-slim` bases.
- Clean package caches in the same `RUN`.
- Never bake secrets into images; use `RUN --mount=type=secret,id=hf_token ...` and `docker build --secret id=hf_token,env=HF_TOKEN` instead.
- Keep a `.dockerignore`.
- Run as a non-root user, one process per container.
- Scan images with `docker scout cves my-app`.

**Drills:** (1) Convert a single-stage Dockerfile to multi-stage and compare image sizes. (2) Add a cache mount for pip or uv. (3) Pass a token at build time without it appearing in `docker history`.

### 12.08 Volumes & bind mounts

```bash
docker volume create hf-cache; docker volume ls; docker volume inspect hf-cache; docker volume rm hf-cache; docker volume prune
docker run --rm -v hf-cache:/root/.cache/huggingface my-app   # named volume: managed by Docker, outlives containers
docker run --rm -v "$(pwd)/data":/data:ro my-app              # bind mount a host folder, read-only
docker run --rm --mount type=bind,source="$(pwd)",target=/workspace my-app   # explicit long form
docker run --rm --tmpfs /tmp my-app                           # in-memory scratch space
```

**Drills:** (1) Share a Hugging Face cache volume between two containers. (2) Mount data read-only and outputs read-write. (3) Find where a named volume lives with `inspect`.

### 12.09 Networking

```bash
docker network create ml-net; docker network ls; docker network inspect ml-net
docker run -d --name db --network ml-net -e POSTGRES_PASSWORD=secret postgres:16
docker run --rm --network ml-net -e PGPASSWORD=secret postgres:16 psql -h db -U postgres -c "SELECT 1"   # containers find each other BY NAME
docker run -d -p 127.0.0.1:8000:8000 my-api     # publish only on localhost (not to your whole network)
# container -> a service on your host: host.docker.internal (Docker Desktop; on Linux add --add-host=host.docker.internal:host-gateway)
```

### 12.10 Docker Compose

```yaml
# compose.yaml : the whole stack in one file -> docker compose up
services:
  api:
    build: .
    image: ghcr.io/ada/ml-api:latest
    ports:
      - "8000:8000"
    environment:
      - MODEL_NAME=distilbert/distilbert-base-uncased-finetuned-sst-2-english
    env_file: .env
    volumes:
      - hf-cache:/root/.cache/huggingface
      - ./src:/app/src
    depends_on:
      db:
        condition: service_healthy
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
  db:
    image: postgres:16
    environment:
      POSTGRES_PASSWORD: secret
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      retries: 5
volumes:
  hf-cache:
  pgdata:
```

```bash
docker compose up -d; docker compose up --build  # start everything (rebuilding images)
docker compose ps; docker compose logs -f api; docker compose exec api bash
docker compose run --rm api python -m pytest     # one-off command in a fresh container
docker compose restart api; docker compose pull; docker compose config   # validate and print the merged config
docker compose down; docker compose down -v      # stop and remove (-v also deletes volumes!)
```

**Drills:** (1) Run an API and Postgres together with Compose. (2) Make the API wait for the database health check. (3) Run your test suite with `compose run`.

### 12.11 GPUs & ML images

```bash
# host setup (Linux): NVIDIA driver + NVIDIA Container Toolkit, then check:
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
docker run --rm --gpus all pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime python -c "import torch; print(torch.cuda.is_available())"
```

```dockerfile
# Training image. Check Docker Hub for current tags and match the CUDA version to your driver.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV HF_HOME=/cache/hf
CMD ["torchrun", "--standalone", "--nproc_per_node=gpu", "train.py"]
```

```bash
docker build -t trainer . && docker run --rm --gpus all --shm-size=16g -v hf-cache:/cache/hf -v "$(pwd)/outputs":/workspace/outputs trainer
```

**Drills:** (1) Verify GPU access from a container. (2) Containerize your 3.05 training script and run it on a GPU. (3) Persist checkpoints to the host with a bind mount.

### 12.12 Recipes: Jupyter & a model API

```bash
docker run --rm -p 8888:8888 -v "$(pwd)":/home/jovyan/work quay.io/jupyter/scipy-notebook   # Jupyter Lab; the login URL is in the logs
```

```python
# serve.py : a minimal model API
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()
clf = pipeline("text-classification", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

class Request(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: Request):
    return clf(req.text)[0]
```

```dockerfile
FROM python:3.12-slim
WORKDIR /app
# CPU-only PyTorch wheels keep the image much smaller
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" transformers torch --extra-index-url https://download.pytorch.org/whl/cpu
COPY serve.py .
EXPOSE 8000
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t sentiment-api . && docker run -d -p 8000:8000 --name api sentiment-api
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{"text": "I love Docker"}'
```

**Drills:** (1) Build and call the sentiment API. (2) Add a `/batch` endpoint that takes a list of texts. (3) Bake the model weights into the image at build time so containers start offline.

### 12.13 Registries

```bash
docker login; docker tag my-app ada/my-app:1.0; docker push ada/my-app:1.0   # Docker Hub
echo "$GITHUB_TOKEN" | docker login ghcr.io -u ada --password-stdin           # GitHub Container Registry
docker tag my-app ghcr.io/ada/my-app:1.0; docker push ghcr.io/ada/my-app:1.0; docker pull ghcr.io/ada/my-app:1.0
docker logout ghcr.io
```

### 12.14 Debugging & cleanup

```bash
docker logs --tail 50 api; docker exec -it api sh
docker run --rm -it --entrypoint sh my-app       # start a shell instead of the default command
docker inspect api --format '{{.State.ExitCode}} {{.State.OOMKilled}}'
# exit codes: 125 docker itself failed · 126 not executable · 127 command not found · 137 killed (often out of memory) · 139 segfault · 143 SIGTERM
docker events; docker system df                  # live events / disk used by images, containers, volumes, cache
docker system prune                              # stopped containers, dangling images, unused networks, build cache
docker system prune -a --volumes                 # EVERYTHING unused, including volumes (careful!)
```

**Drills:** (1) Debug a container that exits immediately by overriding its entrypoint. (2) Find out why a container was killed. (3) Reclaim disk space safely.

Full reference: https://docs.docker.com/reference/

---

# Part III — Data Structures & Algorithms

## 13. Data Structures & Algorithms

Part III teaches differently from Parts I and II. Each module is **one guided tutorial** that walks a beginner through building a single data structure or algorithm **from a blank file**, in **Python, C++, Java or Rust**. There is no problem counter or medal. The home page shows a status instead: **Not Started → In Progress → Almost! → Completed**.

**How a tutorial works.**

1. **Options.** The learner chooses the language, the element type (one concrete type, or generic) and the tutorial's own options, such as singly or doubly linked. Complexity boxes (space, insert, lookup, minimum, …) update as each choice changes.
2. **Steps.** The top of each step is a short overview of what it builds. The literal micro-steps ("make `n` point at `L.head`: `n.next = L.head`", in the chosen language) appear **one at a time in a bubble beside the code, as the learner types**:
   * Each typed line is compared with the expected one, in order. Nothing is compiled. Spacing, comments and the usual equivalent spellings don't matter.
   * When a micro-step is typed, its bubble fades into the next one; deleting code brings earlier bubbles back.
   * The full list is folded under "All micro-steps".

   The learner types everything; there is no class or function stub. **Check** compiles and runs that step's tests plus the earlier steps' tests, and the next step opens only when they pass.
3. **Almost!** Once every step passes, the learner builds the whole structure again in a blank editor. The page shows only the problem: interface, contracts and complexity requirements. Passing gives **Completed**.
4. **Reset.** Reset (with a confirmation) is available at any point and starts over with new options.

**Conventions (all tutorials).**

* **Half-open segments [lo, hi).** `lo` is included and `hi` is excluded, for array ranges (`A[lo..hi)`), list segments (`[start, end)`, the dummy node or null excluded) and every recursive range (merge sort and quicksort sort `A[lo, hi)`). An empty range is `lo == hi`, a range's length is `hi - lo`, and the midpoint is `lo + (hi - lo) / 2`, which can't overflow.
* **Variable names.** A structure is named with its capital first letter:

  | Structure | Name |
  |---|---|
  | array, unbounded array | `A` |
  | list | `L` |
  | stack | `S` |
  | queue | `Q` |
  | hash table, dictionary | `H` |
  | tree | `T` |
  | heap | `H` |
  | priority queue | `P` |
  | set | `S` |

  Functions take the structure as their first argument, as in CMU 15-122 or CLRS pseudocode, so every step reads the same in all four languages:
  * Python, C++ and Rust: `insert_front(L, x)`.
  * Java: `insertFront(L, x)`, as `static` methods inside the class.
  * Rust silences its warning about the capital `L` with `#![allow(non_snake_case)]`.
* **Contracts: an option, off by default.** With contracts off, a tutorial is just the structure or algorithm and its operations:
  * no invariant functions;
  * no checks inside functions;
  * no contract boxes or contract tests.

  A step that held only contracts disappears, and its declarations move into the next step. Binary search goes from 4 steps to 2.

  The checker still inspects the structure with its own invariant, so a broken list is still reported.

  With contracts **on**, it's "contracts before code": step 2 of every data structure writes its **representation invariant** (`is_list(L)`, `is_heap(H)`, `is_bst(T)`, …), plus helpers such as `is_segment`. Every operation then begins with `assert is_X(S)`, the precondition (**requires**), and checks it again before returning, the postcondition (**ensures**). The assert syntax per language:
  * Python: `assert`, removed by `python -O`.
  * C++: `assert`, removed by `-DNDEBUG`.
  * Java: `assert`; the checker runs with assertions on.
  * Rust: `debug_assert!`, gone in release builds.

  Because the invariant checks are O(n), contracts make every call O(n) while testing. Release builds drop them.
* **Generic elements.** The generic option uses each language's native mechanism:
  * C++: `template <typename T>`.
  * Java: `<T>`, comparing with `.equals`.
  * Rust: `<T>` with only the bounds a function needs (`PartialEq`, `Display`, `Default` for dummy nodes).
  * Python: type parameters, `class Node[T]`.

  C's `void*` with client-supplied comparison functions (15-122's C1 style) is the pre-template way to do the same thing. The tutorials explain it but use templates, which are type-checked and cost nothing at run time.
* **Rust and pointers.** A structure where one node is reachable from two places uses raw pointers (`*mut Node<T>`) inside `unsafe` blocks. That covers tail pointers, `prev` pointers, dummies and cycles; `std::collections::LinkedList` is written the same way. The linked-list, stack and queue tutorials all use raw pointers, so the queue can build on the stack. Strictly tree-shaped ownership will use `Option<Box<…>>`, as in BST children.
* **Running code.** Python runs in the browser (Pyodide). C++ (GCC 15.2), Java (JDK 25) and Rust (rustc 1.94, edition 2021) are compiled and run on **Compiler Explorer** (godbolt.org), which accepts requests from any web page. No in-browser compiler covers all three, and none exists at all for Rust.

**How the tests read a structure.** Each test program adds its own `main` with its own helpers, built on the learner's types: an independent walker that prints the contents and the first broken invariant, and builders for corrupted structures. Every printed line is compared with the expected output, and the first line that differs is shown. Each step's tests are mutation-tested: deleting any one statement of the reference makes them fail (`scripts/dsa_selftest.mjs --mutants`).

### 13.01 Binary search

**Interface.** The arguments go array, segment, value. The array is `A`; its segments are `A[lo, hi)`, so an empty segment is `lo == hi`.

| Function | Behaviour | Time |
|---|---|---|
| `is_sorted(A, lo, hi)` | is `A[lo, hi)` in order? | O(n) |
| `is_in(A, lo, hi, x)` | linear search, for contracts | O(n) |
| `all_smaller(A, lo, hi, x)` | every element < x | O(n) |
| `all_bigger(A, lo, hi, x)` | every element > x | O(n) |
| `all_at_least(A, lo, hi, x)` | every element ≥ x | O(n) |
| `binsearch(A, n, x)` | an index of `x`, or -1 | O(log n) |
| `lower_bound(A, n, x)` | the first index with `A[i] >= x` (`n` if none), which is also the insertion point | O(log n) |

`binsearch` requires `is_sorted(A, 0, n)`; when it returns -1, it ensures `!is_in(A, 0, n, x)`. With the recursive option the steps add `binsearch_rec(A, lo, hi, x)` and `lower_bound_rec(A, lo, hi, x)`.

**Invariant.** `A[0, lo) < x < A[hi, n)`, so `x` can only be in [lo, hi). It is checked with `all_smaller` and `all_bigger` at the top of every round.

* The loop runs `while lo < hi`, with `mid = lo + (hi - lo) / 2` (the same value as `(lo + hi) / 2`, but no overflow).
* Found: return `mid`. Smaller: `lo = mid + 1`. Bigger: `hi = mid`.
* Each round shrinks `hi - lo`, which proves termination.
* Lower bound uses `A[0, lo) < x <= A[hi, n)` and returns `lo` once `lo == hi`.

**Options:**

* **Iterative:** O(1) extra space.
* **Recursive:** `binsearch_rec` with the base case `lo == hi`, using O(log n) stack space.

Plus int or generic elements. Generic needs ordering:
* Python: `<`.
* C++: templates.
* Java: `<T extends Comparable<T>>` and `compareTo`.
* Rust: `T: PartialOrd`, with `x: &T`.

**Complexity:** O(log n) time. Space is O(1) iterative or O(log n) recursive. With contracts on it's O(n log n), because the invariant checks scan the array every round. Comparison boxes:
* linear search: O(n);
* minimum and maximum of a sorted array: O(1);
* inserting while keeping it sorted: O(n).

**Steps:**

1. Sorted arrays and their contracts: `is_sorted`, which compares `A[i - 1]` with `A[i]` for `i` in [lo + 1, hi), so an empty segment can't underflow an index; and `is_in`.
2. The loop invariant: `all_smaller`, `all_bigger`.
3. `binsearch`, or `binsearch_rec` plus `binsearch`.
4. Lower bound: `all_at_least`, then `lower_bound`, or `lower_bound_rec` plus `lower_bound`.

**Tests:**
* distinct sorted arrays, every element and absent values below, between and above;
* empty, one- and two-element arrays;
* `n` smaller than the array;
* duplicates for the lower bound;
* random arrays in the final.

Results are checked against a linear scan.

### 13.02 Mergesort

**Interface:**

| Function | Behaviour |
|---|---|
| `is_sorted(A, lo, hi)` | |
| `merge(A, lo, mid, hi)` | requires `A[lo, mid)` and `A[mid, hi)` sorted; ensures `A[lo, hi)` sorted |
| `mergesort(A, lo, hi)` | sorts `A[lo, hi)` |

The array is changed in place: Python `list`, C++ `std::vector<T>&`, Java `T[]`, Rust `&mut [T]`.

**Implementation (one option, as requested).**

`merge` walks the halves with `i` (over `[lo, mid)`) and `j` (over `[mid, hi)`):
1. It appends the smaller front element to a temporary growable array `B`: a list, `std::vector`, `ArrayList` (so Java imports it) or `Vec`. On a tie it takes the left one, which makes the sort **stable**.
2. Then it appends whatever is left of either half.
3. Finally it copies `B` back into `A[lo, hi)`.

Rust's generic version clones elements, since it can't move them out of a slice, so `T: PartialOrd + Clone`.

`mergesort` returns when `hi - lo <= 1`. Otherwise it splits at `mid = lo + (hi - lo) / 2`, sorts both halves and merges them.

**Complexity:** O(n log n) time in every case, sorted input included; `merge` is O(n). O(n) extra space for `B`, plus O(log n) stack. Stable. Quicksort is shown for comparison.

**Steps:**

1. Sorted segments (`is_sorted`).
2. `merge`.
3. `mergesort`.

**Tests.** The checker prints the whole array after every `merge` and `mergesort`.
* `merge` covers empty halves, ties and a sub-segment, leaving the rest of the array untouched.
* `mergesort` covers mixed, reversed, sorted, empty, one-element and duplicate arrays, plus a sub-segment.
* The final adds random arrays.

### 13.03 Quicksort

**Interface:**

| Function | Behaviour |
|---|---|
| `is_sorted(A, lo, hi)` | |
| `all_at_most(A, lo, hi, x)` | |
| `all_bigger(A, lo, hi, x)` | |
| `swap(A, i, j)` | |
| `partition(A, lo, pi, hi) -> mid` | requires `lo <= pi < hi`; ensures `A[lo, mid)` ≤ `A[mid]` < `A[mid + 1, hi)`, with the pivot at `mid` |
| `quicksort(A, lo, hi)` | |

**Implementation (one option, as requested).** The 15-122 partition:
1. Swap the pivot to `A[lo]`.
2. Grow `A[lo + 1, left)` (≤ pivot) and `A[right, hi)` (> pivot) until `left == right`. A small element advances `left`; a big one is swapped to `right - 1`, which then moves left. The loop invariant checks both sides every round.
3. Swap the pivot to `left - 1` and return that index.

`swap` is written each language's own way:
* Python: tuple assignment.
* C++ and Java: a temporary variable.
* Rust: `A.swap(i, j)`, because it can't move a generic element into a temporary.

**Option: the pivot.**
* **Middle element** (`lo + (hi - lo) / 2`): sorted input is O(n log n).
* **First element:** sorted input is O(n²), and the complexity boxes show it.

A random pivot makes bad inputs unlikely; the boxes mention it.

**Complexity:**
* Time: O(n log n) on typical input; O(n²) worst case.
* `partition`: O(n).
* Space: in place, with an O(log n) stack (O(n) worst case).
* Not stable.

**Steps:**

1. Contracts: `is_sorted`, `all_at_most`, `all_bigger`.
2. `swap`.
3. `partition`.
4. `quicksort`.

**Tests.** After a `partition` the checker prints:
* the returned `mid`;
* the untouched outside of `A[lo, hi)`;
* each side **sorted** (by a small insertion sort of its own);
* the pivot.

So any correct partition passes, not only the tutorial's exact arrangement. Duplicates of the pivot belong on its left. Full sorts print the array, and the final adds random arrays.

### 13.04 Linked lists

**Interface:**

| Function | Behaviour | Time |
|---|---|---|
| `Node(data)` | `next` (and `prev`) start null | O(1) |
| `LinkedList()` | an empty list | O(1) |
| `is_segment(start, end, n)` | | O(n) |
| `is_list(L)` | | O(n) |
| `is_empty(L)` | | O(1) |
| `insert_front(L, x)` | | O(1) |
| `insert_back(L, x)` | | O(1) |
| `delete_head(L) -> x` | requires a non-empty list | O(1) |
| `delete_node(L, x) -> bool` | removes the first node holding x | O(n) |
| `lookup(L, x) -> bool` | | O(n) |
| `print_list(L)` | prints `[a, b, c]` | O(n) |

The list keeps `size`, so its length is O(1).

**Options.** End: NULL-terminated or dummy node. Links: singly or doubly. Shape: linear or circular. That makes 8 layouts × int/generic × 4 languages. Each layout uses the fewest pointers that keep both ends O(1):

| Layout | Fields | Elements | Empty | Insert at end |
|---|---|---|---|---|
| NULL, linear | `head`, `tail` | [head, null) | both null | via `tail` |
| NULL, circular, singly | `tail` | `tail.next` is the first | `tail` null | |
| NULL, circular, doubly | `head` | `head.prev` is the last | `head` null | |
| dummy, linear | `head`, `dummy` | [head, dummy); the dummy is the exclusive end | `head == dummy` | write `x` into the dummy, then append a fresh dummy (15-122) |
| dummy, circular | `dummy` | [dummy.next, dummy) around the circle | dummy points to itself | singly: the same fill-the-dummy trick; doubly (CLRS sentinel): via `dummy.prev` |

**Invariant.** The building block is `is_segment(start, end, n)`: following `next` from `start` reaches `end` in exactly `n` steps, never meeting null or `end` early. Doubly linked lists also check `p.next.prev == p` at every step. Each layout's `is_list` is a few checks around one call:

| Layout | `is_list(L)` |
|---|---|
| NULL, singly, linear | empty ⇒ both ends null; otherwise `L.tail.next` is null and `is_segment(L.head, L.tail, L.size - 1)` |
| dummy, singly, linear | `L.dummy.next` is null and `is_segment(L.head, L.dummy, L.size)` |
| NULL, singly, circular | `is_segment(L.tail.next, L.tail, L.size - 1)` |
| dummy, circular | `is_segment(L.dummy.next, L.dummy, L.size)` |

**Complexity:**

| Operation | Singly | Doubly |
|---|---|---|
| insert front / back, delete head | O(1) | O(1) |
| delete last | O(n): needs the node before it | O(1) |
| unlink a node you already hold | O(n) | O(1) |
| lookup, delete by value, minimum, traverse | O(n) | O(n) |
| k-th element | O(k) | O(k) |

Space is n nodes × (element + 1 or 2 pointers), plus 1 dummy.

**Steps.**

1. **The node:** `data`, `next` (and `prev`), null by default.
2. **The list and its contracts:** the header fields and the constructor (it allocates the dummy), then `is_segment`, `is_list` and `is_empty`.
3. **Insert at the start.**
4. **Insert at the end.**
5. **Delete the head:** add the extra precondition, non-empty.
6. **Delete a node:** search with a trailing `prev`; a doubly linked list unlinks in O(1).
7. **Lookup.**
8. **Traverse and print.**

**Pitfalls the tests catch:**

* Forgetting the tail on the first insert or the last delete.
* A circular walk with no step count; a circle has no null to stop at.
* Reading `p.next` after freeing `p`.
* Java `==` on generic elements, which compares references instead of calling `.equals`.
* A function that falls off its end without returning.

### 13.05 Stacks

**Interface:**

| Function | Behaviour | Time |
|---|---|---|
| `Stack()` | an empty stack | O(1) |
| `is_segment(start, end, n)` | | O(n) |
| `is_stack(S)` | true when [`S.top`, null) holds exactly `S.size` nodes | O(n) |
| `stack_empty(S)` | | O(1) |
| `push(S, x)` | | O(1) |
| `pop(S) -> x` | requires non-empty | O(1) |
| `peek(S) -> x` | requires non-empty; Rust's generic version returns `&T`, since it can't move the element out of the node | O(1) |
| `stack_size(S)` | | O(1) |

**Implementation (the requested option).** A singly linked list whose first node is the top: `S.top` and `S.size`. Push and pop both happen at the top, so every operation is O(1) worst case, with no resizing.

**Comparison box (not built here).** An unbounded array: amortized O(1), but better locality.

**Steps:**

1. The node.
2. The stack and its contracts: `Stack`, `is_segment`, `is_stack`, `stack_empty`.
3. `push`.
4. `pop`.
5. `peek` and `stack_size`.

**Tests.** The broken stacks the contract must reject:
* the size one too big or one too small;
* the bottom node pointing back to the top;
* a null top.

`is_segment` is also called past the end.

**Rust.** Raw pointers here too. A stack alone could own its nodes with `Option<Box<…>>`, but the queue in 13.06 can't, and 13.06's two-stack queue builds on this stack. So the tutorials keep one style.

### 13.06 Queues

**Interface:**

| Function | Behaviour |
|---|---|
| `Queue()` | |
| `is_queue(Q)` | |
| `queue_empty(Q)` | |
| `enq(Q, x)` | |
| `deq(Q) -> x` | requires non-empty |
| `queue_peek(Q)` | requires non-empty |
| `queue_size(Q)` | |

**Options:**

* **Linked list with a dummy node (15-122).**
  * Fields: `Q.front`, `Q.back` (the dummy) and `Q.size`. The queue is the segment [front, back), and `is_queue` is `is_segment(Q.front, Q.back, Q.size)` with a dummy whose `next` is null.
  * `enq` writes into the dummy and adds a new one after it.
  * `deq` advances `front`.
  * Every operation is O(1) worst case.
  * Steps:
    1. Node.
    2. Queue and contracts.
    3. `enq`.
    4. `deq`.
    5. `queue_peek` and `queue_size`.
* **Two stacks.**
  * `Q.inbox` and `Q.outbox` are 13.05 stacks.
  * `enq` pushes onto the inbox.
  * `refill(Q)` moves the whole inbox onto the outbox, but only when the outbox is empty. Popping reverses the order, so the oldest element lands on top.
  * `deq` and `queue_peek` refill first, then pop or peek the outbox.
  * Each element moves at most once, so `deq` is **amortized O(1)**; a single `deq` is O(n) in the worst case. The complexity boxes show both.
  * `queue_peek` takes `Q` mutably, because it may refill.
  * Steps:
    1. Node.
    2. Stack and contracts.
    3. `push`, `pop`, `peek`.
    4. Queue and contracts.
    5. `enq`.
    6. `refill` and `deq`.
    7. `queue_peek` and `queue_size`.

**Tests.** The two-stack tests print both stacks (`out=[…] in=[…]`), so the expected output pins down exactly when the transfer happens.

### 13.07 Unbounded arrays

**Interface:**

| Function | Behaviour | Time |
|---|---|---|
| `UnboundedArray()` | empty, with `limit` 1 | |
| `is_uba(A)` | | O(1) |
| `uba_len(A)` | | |
| `uba_get(A, i)` | Rust's generic version returns `&T` | |
| `uba_set(A, i, x)` | | |
| `uba_resize(A, new_limit)` | | |
| `uba_add(A, x)` | push back | |
| `uba_rem(A) -> x` | pop back | |

**Representation (15-122).**
* `A.size` elements live in `A.data[0, A.size)`, inside a fixed block of `A.limit` slots.
* The block per language:
  * Python: a list made with `[0] * limit`.
  * C++: a `std::vector` used only as a fixed block.
  * Java: `int[]`, or `(T[]) new Object[n]`.
  * Rust: a `Vec`, filled with `T::default()`, so `T: Clone + Default`.
* **Invariant** (`is_uba`): `size < limit`, so one slot is always free, and `len(data) == limit`.
* `uba_resize` requires only `size <= limit`, because `uba_add` calls it exactly when the block is full.

**Options: the resizing mechanism.**

| Growth | Rule | Amortized add | Copies for n adds |
|---|---|---|---|
| ×2 | | O(1) | < 2n |
| ×1.5 | `limit + (limit + 1) / 2` | O(1) | < 3n |
| +4 | constant growth | **O(n)** | ≈ n²/8 |

The +4 row is there so the learner sees why geometric growth matters.

| Shrinking | Rule | Amortized remove | Space |
|---|---|---|---|
| at ¼ full | halve when `limit >= 4` and `size <= limit / 4` | O(1) | at most about 4× the elements |
| below ½ full | halve when `limit >= 2` and `size < limit / 2` | O(n): thrashes, since every add then remove at the boundary resizes | |
| never | | | O(peak) |

**Complexity:** get, set and length are O(1). One add that resizes is O(n). Inserting in the middle is O(n), shown for comparison. The contract is O(1).

**Steps:**

1. The struct and its contracts (`is_uba`, `uba_len`).
2. `uba_get` and `uba_set`, which require `0 <= i < size`.
3. `uba_resize`: allocate, copy `[0, size)`, switch.
4. `uba_add`, with the chosen growth.
5. `uba_rem`, with the chosen shrink rule.

**Tests.** The checker prints `size` and `limit` after every operation, so the chosen policy itself is checked. A JavaScript model replays it exactly.
* Long runs: 20 or 33 adds, then removes.
* Broken arrays: size equal to limit, a wrong limit, a wrong block length, a negative size.

### 13.08 Hash tables

**Interface (key → int value):**

| Function | Behaviour |
|---|---|
| `ht_hash(k, m)` | |
| `is_ht(H)` | |
| `ht_size(H)` | |
| `ht_contains(H, k)` | |
| `ht_get(H, k)` | requires the key |
| `ht_insert(H, k, v)` | inserts or updates |
| `ht_delete(H, k) -> bool` | |
| `ht_resize(H, new_capacity)` | |

Chaining adds `ht_find(H, k) -> entry`. Open addressing adds `ht_probe(h, t, m)` and `ht_find_slot(H, k) -> slot or -1`.

**Keys: int or string, instead of generic.** A hash table needs a concrete hash function, and writing it is the first step.
* int: the key reduced into [0, m), never negative. Python `%`, C++ `((k % m) + m) % m`, Java `Math.floorMod`, Rust `rem_euclid`.
* string: `h = (h * 31 + code) % m`, character by character. It's reduced at every step, so there's no overflow and the same numbers come out in every language.

Because the hash gives the same numbers everywhere, the tests can print the exact table layout.

**Options: collision resolution.**
* **Separate chaining:**
  * Each bucket is a linked list of entries `{key, value, next}`, and new entries go at the front.
  * The table doubles when `size > capacity`.
  * `is_ht` checks that every entry sits in the bucket its key hashes to, and that there are exactly `size` of them.
* **Open addressing, linear probing:** slot `(h + t) % m`.
* **Open addressing, quadratic probing:** slot `(h + t(t + 1)/2) % m`. These triangular steps reach every slot when the capacity is a power of two, and every capacity here is (it starts at 8 and doubles).

How both open-addressing versions work:
* One entry per slot.
* `delete` leaves a **tombstone**: a special entry, `H.tomb`. That way, keys that probed past the slot are still found.
* `used` counts entries plus tombstones. The table doubles when `used * 2 > capacity`.
* Resizing rehashes the entries, drops the tombstones and resets `used`.
* `is_ht` checks that searching for each entry's key finds it at its own slot (reachable, and no earlier copy of the key), that the counts match, and that at most half the slots are in use.

**Complexity:**
* insert, lookup and delete: expected O(1); worst case O(n), all keys colliding.
* resize: O(n), so O(1) amortized per insert.
* min or sorted order: O(n), because there's no order.
* The boxes explain primary clustering for linear probing, and why quadratic probing clusters less.

**Steps:**

1. The table and the hash function. Open addressing adds `ht_probe`.
2. The contract: `is_ht` and `ht_size`. Open addressing writes `ht_find_slot` first, because the invariant uses it.
3. Lookup: chaining's `ht_find`, then `ht_contains` and `ht_get`.
4. `ht_resize`.
5. `ht_insert`.
6. `ht_delete`.

**Tests.** The checker prints the whole table after every change: the non-empty buckets in chain order, or every slot (`_` for empty, `X` for a tombstone), with `size`, `used` and `capacity`.

The broken tables the contract must reject:
* the size one too big;
* a capacity that doesn't match the table;
* a chain moved to the wrong bucket;
* a chain that loops back on itself;
* an entry cut off from its home slot;
* an uncounted tombstone.

A run that creates tombstones and then forces a resize checks that `used` resets.

### 13.09 Binary search trees

**Interface:** `bst_new()`, `is_bst(T)`, `bst_lookup(T, k)`, `bst_insert(T, k)`, `bst_delete(T, k)`, `bst_min` / `bst_max`, `bst_inorder(T)`, `height(T)`.

**Invariant.** Every key in the left subtree < node < every key in the right subtree. This is checked with bounds passed down (`is_ordered(node, lo, hi)`), not just parent-child pairs.

**Options:**

* **Plain BST.** Every operation costs O(h), which is O(n) worst case on sorted input.
* **AVL tree.**
  * Each node stores its height, and every node's balance factor is in {-1, 0, 1}.
  * Single and double rotations repair the balance on the way back up from an insert or delete.
  * Worst case O(log n), with height < 1.44 log₂ n.
* **Splay tree.**
  * Each access splays the node to the root with zig, zig-zig and zig-zag steps.
  * Amortized O(log n), with no balance information stored.
  * Recently used keys are fast.

**Complexity:**

| Operation | Plain | AVL | Splay |
|---|---|---|---|
| lookup, insert, delete | O(h), O(n) worst | O(log n) | O(log n) amortized |
| min, max, successor | O(h) | O(log n) | O(log n) amortized |
| in-order traversal | O(n) | O(n) | O(n) |

Space is O(n).

**Steps:**

1. Node and tree.
2. Contracts: ordering with bounds, heights for AVL.
3. `lookup`.
4. `insert`.
5. min/max and in-order traversal.
6. `delete`: the leaf, one-child and two-child cases; the last swaps with the in-order successor.
7. AVL only: `height`/`fix_height`, `rotate_left`/`rotate_right`, `rebalance`.
8. Splay only: `rotate` and `splay`.

Children use `Option<Box<Node>>` in Rust.

### 13.10 Dictionaries and sets

**Interface:**

* Dictionary `H`: `dict_insert(H, k, v)`, `dict_lookup(H, k) -> v or null`, `dict_delete(H, k)`, `dict_size`.
* Set `S`: `set_add(S, x)`, `set_contains(S, x)`, `set_remove(S, x)`.
* Set operations: `union`, `intersection`, `difference`.

**Options: what it's built on.**

* **Hash table** (13.08): expected O(1) operations, no order.
* **Binary search tree** (13.09, plain or AVL): O(log n) operations with order. This adds `min`, `successor`, range queries and sorted iteration.

The complexity boxes show this trade-off directly. The tutorial wraps the earlier structure behind a client interface (the abstraction boundary), so the client code is the same for both options.

**Steps:**

1. The interface and its contracts, written first.
2. Implement it on the chosen structure.
3. Sets as dictionaries without values.
4. Set operations: union, intersection, difference.
5. A client: word counts.

### 13.11 Heaps

**Interface:**

| Function | Behaviour | Time |
|---|---|---|
| `heap_new(capacity)` | | |
| `is_heap(H)` | | |
| `heap_add(H, x)` | | O(log n) |
| `heap_peek(H)` | | O(1) |
| `heap_rem(H) -> x` | | O(log n) |
| `heapify(A)` | | O(n) |
| `heapsort(A)` | | O(n log n) |

**Representation.** An array-based complete binary tree. With 1-indexing (15-122), the children of i are `2i` and `2i+1` and its parent is `i/2`. With 0-indexing, the children are `2i+1` and `2i+2` and the parent is `(i-1)/2`.

**Invariant:**

* The shape: the elements occupy `[1, next)`.
* The **ordering invariant:** the parent ≤ the child (a min-heap).

**Options:**

* Min-heap or max-heap, or generic with a comparison function.
* 0- or 1-indexed.
* An unbounded backing array (13.07) or a fixed capacity.

**Steps:**

1. Struct and contracts. `is_heap` checks every parent/child pair.
2. `sift_up`.
3. `add`.
4. `sift_down`, comparing with the smaller child.
5. `rem`: move the last element to the root, then sift down.
6. Bottom-up `heapify`: sift down from n/2 to 1, O(n) in total.
7. `heapsort`.

### 13.12 Priority queues

**Interface:** `pq_new()`, `pq_add(P, x, priority)`, `pq_peek(P)`, `pq_rem(P)` (highest priority first), `pq_size(P)`, and optionally `pq_decrease_key`.

**Options: the implementation.**

| Implementation | `pq_add` | `pq_rem` | Notes |
|---|---|---|---|
| unsorted array | O(1) | O(n) | |
| sorted linked list | O(n) | O(1) | |
| binary heap (13.11) | O(log n) | O(log n) | the optimal general-purpose choice |
| heap + index map | O(log n) | O(log n) | `decrease_key` also O(log n), as Dijkstra needs |

The complexity boxes compare all of them.

**Steps:**

1. The interface and contracts.
2. Elements carry priorities.
3. Wrap the heap.
4. Ties: stable order by insertion counter.
5. A client that merges k sorted lists.
