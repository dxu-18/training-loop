"""Helpers shared by the scikit-learn templates (COMP_GUIDE §4).

Problems run in Pyodide with scikit-learn 1.8 (the same version as the local tests). The setup imports NumPy and
pandas and builds the data; the learner writes the sklearn imports (learning where things live is part of it, and
the Instructions show each name's full path). The grader reruns the reference solution on the same setup, so
answers only need to be deterministic: every model gets a `random_state` and the prompt says which.

Keep problems fast: Pyodide is a few times slower than CPython and grading runs the reference and the answer
(and hidden tests), so a reference should take well under half a second here (`scripts/sklearn_templates.py`
measures it). Use the bundled datasets (iris, wine, breast cancer, digits, diabetes), `make_*` generators or a
themed `table()`; never `fetch_*` (no network) and no `n_jobs`.
"""

from __future__ import annotations

import os
import random
import tempfile
from dataclasses import dataclass, field

from ...core import Problem, pick, template as _template
from ..s01_numpy._common import an, arr, floats, lit, mat, seeded  # noqa: F401  (re-exported)

S = "sklearn"
HEADER = "import numpy as np\nimport pandas as pd\n"
STDOUT = "<stdout>"


def template(module: str, tier: str):
    return _template(S, module, tier)


def P(prompt: str, setup: str, solution: str, **kw) -> Problem:
    kw.setdefault("timeout", 15.0)
    return Problem(prompt=prompt.strip(), setup=(HEADER + setup.strip()).strip(), solution=solution.strip(), **kw)


def solve(setup: str, solution: str, var: str = "result"):
    """Run a reference solution (to show a sample answer in a prompt), in a throwaway directory."""
    import warnings
    from ...grader import numpy_32bit_compat
    numpy_32bit_compat()            # Pyodide is 32-bit: np.repeat / np.bincount with int64 counts
    old = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="tl-solve-") as tmp, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        os.chdir(tmp)
        try:
            ns: dict = {}
            exec(HEADER + setup, ns)
            exec(solution, ns)
            return ns[var]
        finally:
            os.chdir(old)


def fmt(v, nd: int = 4) -> str:
    """A number for a prompt: 0.8125 -> '0.8125', 3.0 -> '3'."""
    if isinstance(v, float):
        s = f"{v:.{nd}f}".rstrip("0").rstrip(".")
        return s if s not in ("-0", "") else "0"
    return str(v)


# ---------------------------------------------------------------- catalogs
# One template, many related tasks on shared random data: ONE item per problem, chosen with core.pick() (the tests
# force every index). Item: (api, prompt, solution[, {options}]); options may add `setup` lines, `require` (default
# [api]; [] for none), `tol`, `answer`, `post`, `timeout`.

def catalog(rng: random.Random, items: list, setup: str, intro: str = "", **kw) -> Problem:
    item = pick(rng, items)
    api, prompt, sol = item[:3]
    opts = dict(item[3]) if len(item) > 3 else {}
    req = opts.pop("require", kw.pop("require", None) or ([api] if api else []))
    extra = opts.pop("setup", "")
    return P(f"{intro} {prompt}".strip(), (setup + ("\n" + extra if extra else "")).strip(), sol, require=req,
             **{**kw, **opts})


# ---------------------------------------------------------------- challenge problems
# Application problems (~Codeforces 1800) about how ML algorithms work: implement a piece of scikit-learn (or a
# step around it) efficiently. The visible data is small; hidden tests are large, so quadratic code times out. Any
# Python is allowed (sklearn too, when it helps). Each one has a brute force in tests/sklearn_brutes/.

def challenge(title: str, story: str, setup: str, solution: str, tests: list[str], constraints: str,
              answer_desc: str = "", speed: str = "O(n log n) or better", **kw) -> Problem:
    sample = solve(setup, solution)
    shown = sample.tolist() if hasattr(sample, "tolist") else sample
    if isinstance(shown, float):
        shown = round(shown, 6)
    kw.setdefault("timeout", 20.0)
    return P(f"**{title}.** {story}\n\n**Constraints:** {constraints}. Your code also runs on hidden tests up to these "
             f"limits (time limit a few seconds), so aim for {speed}.\n\nFor the data shown, the answer is "
             f"`{shown!r}`. Store your answer in `result`{answer_desc}.",
             setup, solution, tests=[HEADER + t for t in tests], **kw)


# ---------------------------------------------------------------- bundled and generated datasets

@dataclass
class Data:
    setup: str          # code defining X, y (and maybe feature_names / target_names)
    desc: str           # "`X` holds 150 iris flowers (4 measurements) and `y` their species (0, 1, 2)"
    kind: str           # "clf" | "reg"
    n_classes: int = 0
    n: int = 0
    d: int = 0
    name: str = ""


BUILTIN = {
    "iris": Data("from sklearn.datasets import load_iris\nX, y = load_iris(return_X_y=True)",
                 "`X` holds 150 iris flowers (4 measurements in cm) and `y` their species (0, 1, 2)", "clf", 3, 150, 4,
                 "iris"),
    "wine": Data("from sklearn.datasets import load_wine\nX, y = load_wine(return_X_y=True)",
                 "`X` holds 178 wines (13 chemical measurements) and `y` their cultivar (0, 1, 2)", "clf", 3, 178, 13,
                 "wine"),
    "cancer": Data("from sklearn.datasets import load_breast_cancer\nX, y = load_breast_cancer(return_X_y=True)",
                   "`X` holds 569 breast-tumour samples (30 cell measurements) and `y` the diagnosis (0 malignant, 1 "
                   "benign)", "clf", 2, 569, 30, "cancer"),
    "digits": Data("from sklearn.datasets import load_digits\nX, y = load_digits(return_X_y=True)",
                   "`X` holds 1797 handwritten digits (8×8 pixels flattened to 64 values from 0 to 16) and `y` the digit "
                   "(0-9)", "clf", 10, 1797, 64, "digits"),
    "diabetes": Data("from sklearn.datasets import load_diabetes\nX, y = load_diabetes(return_X_y=True)",
                     "`X` holds 442 diabetes patients (10 standardized measurements) and `y` disease progression one "
                     "year later", "reg", 0, 442, 10, "diabetes"),
}


def builtin(rng: random.Random, kind: str = "clf", allow=None, small: bool = False) -> Data:
    """A bundled dataset. small: skip digits (1797 rows)."""
    names = [k for k, v in BUILTIN.items() if v.kind == kind and (allow is None or k in allow)
             and not (small and k == "digits")]
    return BUILTIN[rng.choice(names)]


def synth_clf(rng: random.Random, n=(120, 400), d=(4, 10), classes=(2, 3), informative=None, imbalance=None,
              sep: float | None = None) -> Data:
    """make_classification with random sizes and a seeded random_state."""
    n_, d_ = rng.randint(*n), rng.randint(*d)
    k = rng.randint(*classes) if isinstance(classes, tuple) else classes
    inf = informative or min(d_, max(2, k, rng.randint(2, max(2, d_ - 2))))
    red = rng.randint(0, max(0, min(2, d_ - inf)))
    args = [f"n_samples={n_}", f"n_features={d_}", f"n_informative={inf}", f"n_redundant={red}"]
    if k != 2:
        args.append(f"n_classes={k}")
    if imbalance:
        w = [round(1 - imbalance, 3), imbalance] if k == 2 else None
        if w:
            args.append(f"weights={w}")
    if sep:
        args.append(f"class_sep={sep}")
    args.append(f"random_state={seeded(rng)}")
    return Data(f"from sklearn.datasets import make_classification\nX, y = make_classification({', '.join(args)})",
                f"`X` ({n_} × {d_}) and `y` ({k} classes) come from `make_classification`", "clf", k, n_, d_, "synthetic")


def synth_reg(rng: random.Random, n=(100, 300), d=(3, 8), noise=(5, 20)) -> Data:
    n_, d_ = rng.randint(*n), rng.randint(*d)
    inf = rng.randint(2, d_)
    nz = rng.randint(*noise)
    return Data(f"from sklearn.datasets import make_regression\nX, y = make_regression(n_samples={n_}, n_features={d_}, "
                f"n_informative={inf}, noise={nz}, random_state={seeded(rng)})",
                f"`X` ({n_} × {d_}) and the continuous target `y` come from `make_regression`", "reg", 0, n_, d_,
                "synthetic")


def clf_data(rng: random.Random, small: bool = True, synth: float = 0.5, **kw) -> Data:
    """A classification dataset: bundled or make_classification."""
    return synth_clf(rng, **kw) if rng.random() < synth else builtin(rng, "clf", small=small)


def reg_data(rng: random.Random, synth: float = 0.6, **kw) -> Data:
    return synth_reg(rng, **kw) if rng.random() < synth else builtin(rng, "reg")


# ---------------------------------------------------------------- themed tables (DataFrames with mixed columns)

@dataclass
class Table:
    setup: str                  # code defining `df` (features) and `y` (target)
    desc: str                   # one sentence for the prompt
    kind: str                   # clf | reg
    num: list[str]              # numeric columns
    cat: list[str]              # categorical (string) columns
    target: str                 # what y is called in words
    n: int
    missing: list[str] = field(default_factory=list)   # numeric columns with NaNs
    levels: dict = field(default_factory=dict)         # categorical column -> its levels
    noun: str = ""


# key: (noun, plural, kind, target words, numeric {col: generator code}, categorical {col: levels}, target code)
# Generators use `rng` (np.random.default_rng) and `n`; the target code may use `df` and `rng`.
SCENARIOS = {
    "churn": ("customer", "customers", "clf", "whether the customer churned (1) or stayed (0)",
              {"tenure_months": "rng.integers(1, 73, n)", "monthly_spend": "rng.normal(70, 25, n).clip(5).round(2)",
               "support_calls": "rng.poisson(2, n)"},
              {"plan": ["basic", "standard", "premium"], "region": ["north", "south", "east", "west"]},
              "(0.8 * df['support_calls'] - 0.05 * df['tenure_months'] + 0.02 * df['monthly_spend'] "
              "+ (df['plan'] == 'basic') + rng.normal(0, 1, n) > 1.2).astype(int)"),
    "houses": ("house", "houses", "reg", "the sale price in thousands of dollars",
               {"area_m2": "rng.uniform(35, 240, n).round(1)", "rooms": "rng.integers(1, 8, n)",
                "age_years": "rng.integers(0, 91, n)"},
               {"neighborhood": ["riverside", "old town", "hills", "center", "harbor"], "garage": ["yes", "no"]},
               "(1.8 * df['area_m2'] + 15 * df['rooms'] - 0.6 * df['age_years'] + df['neighborhood'].map({'riverside': "
               "40, 'old town': 10, 'hills': 60, 'center': 80, 'harbor': 25}) + 20 * (df['garage'] == 'yes') "
               "+ rng.normal(0, 25, n)).round(1)"),
    "loans": ("loan applicant", "loan applicants", "clf", "whether the applicant defaulted (1) or repaid (0)",
              {"income_k": "rng.normal(60, 20, n).clip(10).round(1)", "loan_k": "rng.uniform(2, 80, n).round(1)",
               "credit_score": "rng.integers(300, 851, n)"},
              {"employment": ["salaried", "self-employed", "student", "retired"],
               "purpose": ["car", "home", "education", "travel"]},
              "(0.05 * df['loan_k'] - 0.03 * df['income_k'] - 0.01 * (df['credit_score'] - 600) "
              "+ 0.8 * (df['employment'] == 'student') + rng.normal(0, 1, n) > 0).astype(int)"),
    "students": ("student", "students", "clf", "whether the student passed the final exam (1) or not (0)",
                 {"study_hours": "rng.uniform(0, 30, n).round(1)", "sleep_hours": "rng.normal(7, 1.2, n).round(1)",
                  "attendance_pct": "rng.uniform(50, 100, n).round(1)"},
                 {"track": ["science", "arts", "business"], "school": ["north high", "lakeside", "westview"]},
                 "(0.12 * df['study_hours'] + 0.3 * df['sleep_hours'] + 0.04 * df['attendance_pct'] "
                 "+ rng.normal(0, 1, n) > 7.4).astype(int)"),
    "experiments": ("training run", "training runs", "reg", "the validation accuracy (%)",
                    {"log10_lr": "rng.uniform(-4, -1, n).round(2)", "batch_size": "rng.choice([16, 32, 64, 128], n)",
                     "dropout": "rng.uniform(0, 0.5, n).round(2)", "epochs": "rng.integers(5, 51, n)"},
                    {"optimizer": ["adam", "sgd", "adamw"], "scheduler": ["cosine", "step", "none"]},
                    "(80 - 6 * (df['log10_lr'] + 2.5) ** 2 + 0.1 * df['epochs'] - 8 * (df['dropout'] - 0.2) ** 2 "
                    "+ 2 * (df['optimizer'] == 'adamw') + rng.normal(0, 1.5, n)).round(2)"),
    "servers": ("server", "servers", "reg", "the p95 request latency in ms",
                {"cpu_util": "rng.uniform(5, 95, n).round(1)", "mem_gb": "rng.choice([8, 16, 32, 64], n)",
                 "requests_per_s": "rng.normal(300, 90, n).clip(20).round()"},
                {"gpu": ["none", "T4", "A10", "A100"], "cloud_region": ["us-east", "eu-west", "ap-south"]},
                "(20 + 0.9 * df['cpu_util'] + 0.08 * df['requests_per_s'] - 0.3 * df['mem_gb'] + df['cloud_region']"
                ".map({'us-east': 0, 'eu-west': 12, 'ap-south': 30}) + rng.normal(0, 6, n)).round(1)"),
    "patients": ("patient", "patients", "clf", "whether the patient is at cardiac risk (1) or not (0)",
                 {"age": "rng.integers(20, 86, n)", "bmi": "rng.normal(27, 5, n).round(1)",
                  "systolic_bp": "rng.normal(128, 15, n).round()"},
                 {"smoker": ["yes", "no"], "activity": ["low", "moderate", "high"]},
                 "(0.05 * df['age'] + 0.1 * df['bmi'] + 0.03 * df['systolic_bp'] + 1.2 * (df['smoker'] == 'yes') "
                 "- 0.8 * (df['activity'] == 'high') + rng.normal(0, 1, n) > 10.4).astype(int)"),
    "flights": ("flight", "flights", "clf", "whether the flight was delayed by over 15 minutes (1) or not (0)",
                {"distance_km": "rng.uniform(200, 5000, n).round()", "dep_hour": "rng.integers(5, 24, n)",
                 "load_factor": "rng.uniform(0.4, 1.0, n).round(2)"},
                {"carrier": ["skyway", "aerolux", "bluejet"], "weather": ["clear", "rain", "snow", "fog"]},
                "(0.15 * df['dep_hour'] + 2 * df['load_factor'] + df['weather'].map({'clear': 0, 'rain': 1, 'snow': 2.5, "
                "'fog': 1.5}) + rng.normal(0, 1, n) > 4.6).astype(int)"),
    "bikes": ("day of bike rentals", "days of bike rentals", "reg", "the number of bikes rented",
              {"temp_c": "rng.normal(18, 8, n).round(1)", "humidity": "rng.uniform(20, 95, n).round()",
               "windspeed": "rng.gamma(2, 5, n).round(1)"},
              {"season": ["spring", "summer", "autumn", "winter"], "workday": ["yes", "no"]},
              "(400 + 25 * df['temp_c'] - 3 * df['humidity'] - 6 * df['windspeed'] + 150 * (df['workday'] == 'yes') "
              "+ rng.normal(0, 60, n)).clip(0).round()"),
    "phones": ("phone", "phones", "reg", "the resale price in dollars",
               {"age_months": "rng.integers(1, 61, n)", "storage_gb": "rng.choice([64, 128, 256, 512], n)",
                "battery_pct": "rng.uniform(60, 100, n).round()"},
               {"brand": ["pear", "galaxy", "pixelon", "nokio"], "condition": ["mint", "good", "fair"]},
               "(700 - 9 * df['age_months'] + 0.5 * df['storage_gb'] + 3 * (df['battery_pct'] - 80) + df['condition']"
               ".map({'mint': 60, 'good': 0, 'fair': -90}) + rng.normal(0, 30, n)).clip(20).round()"),
}


def table(rng: random.Random, kind: str | None = None, n=(120, 300), missing: int = 0, only=None,
          numeric_only: bool = False, seed: int | None = None) -> Table:
    """A themed DataFrame `df` and target `y`, generated in the setup from a seed.
    missing: how many numeric columns get ~10% NaN (after the target is computed)."""
    keys = [k for k, v in SCENARIOS.items() if (kind is None or v[2] == kind) and (only is None or k in only)]
    key = rng.choice(keys)
    noun, plural, kd, target, num, cat, target_code = SCENARIOS[key]
    n_ = rng.randint(*n) if isinstance(n, tuple) else n
    s = seeded(rng) if seed is None else seed
    cols = dict(num)
    if not numeric_only:
        cols.update({c: f"rng.choice({lit(v)}, n)" for c, v in cat.items()})
    body = "".join(f"    {c!r}: {g},\n" for c, g in cols.items())
    lines = [f"rng = np.random.default_rng({s})", f"n = {n_}", "df = pd.DataFrame({\n" + body + "})"]
    tgt = target_code
    if numeric_only:
        for c, v in cat.items():   # the target formula may use a categorical column: draw it, then don't keep it
            if f"df['{c}']" in tgt:
                tgt = tgt.replace(f"df['{c}']", f"pd.Series(rng.choice({lit(v)}, n))")
    lines.append(f"y = {tgt}")
    miss = rng.sample(list(num), min(missing, len(num))) if missing else []
    for c in miss:
        lines.append(f"df.loc[rng.random(n) < 0.1, {c!r}] = np.nan")
    lines.append("y = y.to_numpy()")
    desc = f"`df` describes {n_} {plural} and `y` holds {target}"
    return Table("\n".join(lines), desc, kd, list(num), [] if numeric_only else list(cat), target, n_, miss,
                 {} if numeric_only else {c: list(v) for c, v in cat.items()}, noun)
