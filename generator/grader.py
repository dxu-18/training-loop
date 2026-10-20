"""Run and grade user code against a problem dict (as produced by core.generate).

The graded value is, in order of precedence:
  * the value of ``problem["post"]`` (check code run after the user's code), else
  * the variables in ``problem["answer"]`` ("<stdout>" = everything the code printed), else
  * a variable named ``result``, else the value of the last expression.
The reference solution is run the same way on the same setup, and the two values
are compared with ``compare``. Every run gets a fresh temporary working directory,
and NumPy's global print/error options are restored afterwards.
"""

from __future__ import annotations

import ast
import contextlib
import io
import linecache
import math
import os
import shutil
import sys
import tempfile
import traceback
import warnings

os.environ.setdefault("MPLBACKEND", "Agg")  # headless plotting (Pyodide runs us in a Web Worker)

USER_FILE = "<answer>"
STDOUT = "<stdout>"
MAX_REPR = 4000
_MISSING = object()


class NoAnswer(Exception):
    """The code ran but did not produce the value we grade."""


# ---------------------------------------------------------------- execution

def _register_source(filename: str, code: str) -> None:
    linecache.cache[filename] = (len(code), None, code.splitlines(True), filename)


def _exec(code: str, ns: dict, filename: str):
    """Exec ``code`` in ``ns``; return the value of a trailing expression (or _MISSING)."""
    _register_source(filename, code)
    tree = ast.parse(code, filename)
    last = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = ast.Expression(tree.body.pop().value)
    exec(compile(tree, filename, "exec"), ns)
    if last is None:
        return _MISSING
    return eval(compile(last, filename, "eval"), ns)


def _answer(problem: dict, ns: dict, last, stdout: str):
    names = problem.get("answer") or []
    if names == [STDOUT]:
        return stdout
    if names:
        missing = [n for n in names if n not in ns]
        if missing:
            raise NoAnswer(f"Your code must define {', '.join('`' + n + '`' for n in missing)}.")
        return {n: ns[n] for n in names} if len(names) > 1 else ns[names[0]]
    if "result" in ns:
        return ns["result"]
    if last is _MISSING or last is None:
        raise NoAnswer("Your code didn't produce an answer: end with an expression "
                       "(not a print) or assign it to `result`.")
    return last


@contextlib.contextmanager
def _sandbox():
    """Run in a fresh temp directory; afterwards restore NumPy/pandas options and close figures."""
    try:
        import numpy as np
        saved = (np.get_printoptions(), np.geterr())
    except ImportError:
        np = saved = None
    old = os.getcwd()
    tmp = tempfile.mkdtemp(prefix="tl-")
    os.chdir(tmp)
    try:
        yield
    finally:
        os.chdir(old)
        shutil.rmtree(tmp, ignore_errors=True)
        if saved:
            np.set_printoptions(**saved[0])
            np.seterr(**saved[1])
        if (pd := sys.modules.get("pandas")) is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # only display.* (what user code may set): resetting "all" also resets the plotting
                # options, whose callback imports matplotlib, which pages without plots don't load
                pd.reset_option("^display\\.")
        if (plt := sys.modules.get("matplotlib.pyplot")) is not None:
            plt.close("all")


def _detach(value):
    """Copy memory-mapped arrays so they survive the temp directory being removed."""
    np = sys.modules.get("numpy")
    if np is not None and isinstance(value, np.memmap):
        return np.array(value)
    if isinstance(value, dict):
        return {k: _detach(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_detach(v) for v in value)
    return value


def format_error(exc: BaseException) -> str:
    if isinstance(exc, SyntaxError):
        out = [f"  line {exc.lineno}"] if exc.lineno else []
        if exc.text:
            out.append("    " + exc.text.rstrip("\n"))
            if exc.offset:
                out.append("    " + " " * (exc.offset - 1) + "^")
        out.append(f"{type(exc).__name__}: {exc.msg}")
        return "\n".join(out)
    frames = [f for f in traceback.extract_tb(exc.__traceback__) if f.filename == USER_FILE]
    lines = ["Traceback (most recent call last):"] if frames else []
    for f in frames:
        lines.append(f"  line {f.lineno}" + (f", in {f.name}" if f.name != "<module>" else ""))
        if f.line:
            lines.append("    " + f.line)
    lines.extend(traceback.format_exception_only(type(exc), exc))
    return "\n".join(l.rstrip("\n") for l in lines)


def short_repr(value) -> str:
    try:
        r = repr(value)
    except Exception as e:  # noqa: BLE001
        r = f"<unprintable {type(value).__name__}: {e}>"
    return r if len(r) <= MAX_REPR else r[:MAX_REPR] + " ..."


def _run_once(problem: dict, code: str, setup: str, filename: str):
    """-> (answer, stdout). Raises the user's exception unchanged, NoAnswer if nothing to grade."""
    ns: dict = {"__name__": "__main__"}
    buf = io.StringIO()
    with _sandbox():
        with contextlib.redirect_stdout(io.StringIO()):
            _exec(setup, ns, "<setup>")
        with contextlib.redirect_stdout(buf):
            last = _exec(code, ns, filename)
        if problem.get("post"):
            ns["_"] = ns.get("result") if last is _MISSING or last is None else last  # the answer, for the check
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    value = _exec(problem["post"], ns, "<check>")
            except Exception as e:  # noqa: BLE001
                raise NoAnswer(f"Checking your result failed: {type(e).__name__}: {e}") from None
        else:
            value = _answer(problem, ns, last, buf.getvalue())
        return _detach(value), buf.getvalue()


# ---------------------------------------------------------------- comparison

def _plain(x):
    """numpy scalars / 0-d arrays -> Python scalars; leave everything else alone."""
    np = sys.modules.get("numpy")
    if np is not None:
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, np.ndarray) and x.ndim == 0:
            return x.item()
    return x


def compare(exp, got, tol: float = 1e-6, check_dtype: bool = False, path: str = "answer"):
    """None if ``got`` matches ``exp``, else a short message saying what differs."""
    np = sys.modules.get("numpy")
    pd = sys.modules.get("pandas")

    if pd is not None:
        if isinstance(exp, pd.api.extensions.ExtensionArray):  # e.g. pd.array(..., dtype="Int64"), Categorical
            if not isinstance(got, pd.api.extensions.ExtensionArray) and not hasattr(got, "__len__"):
                return f"{path} should be an array, got {type(got).__name__}"
            exp, got = pd.Series(exp), pd.Series(got)
        if isinstance(exp, (pd.DataFrame, pd.Series, pd.Index)):
            kind = "DataFrame" if isinstance(exp, pd.DataFrame) else "Series" if isinstance(exp, pd.Series) else "Index"
            if not isinstance(got, (pd.DataFrame if kind == "DataFrame" else pd.Series if kind == "Series" else pd.Index)):
                return f"{path} should be a {kind}, got {type(got).__name__}"
            loose = dict(check_exact=False, rtol=tol, atol=tol, check_names=False)
            try:
                if kind == "Index":
                    pd.testing.assert_index_equal(exp, got, exact=False, **loose)
                else:
                    fn = pd.testing.assert_frame_equal if kind == "DataFrame" else pd.testing.assert_series_equal
                    extra = dict(check_column_type=False) if kind == "DataFrame" else {}
                    fn(exp, got, check_dtype=check_dtype, check_index_type=False, check_freq=False,
                       **extra, **loose)
            except AssertionError as e:
                lines = [ln for ln in str(e).strip().splitlines() if ln.strip()]
                return f"{path} differs: " + " ".join(lines[:2])[:300]
            return None
        if exp is pd.NA or exp is pd.NaT:
            return None if got is exp or (got is not None and pd.isna(got) is True) else f"{path} should be missing ({exp})"

    if np is not None and isinstance(exp, np.ndarray) and exp.ndim > 0:
        if not isinstance(got, np.ndarray):
            if check_dtype or not isinstance(got, (list, tuple)):
                return f"{path} should be a NumPy array, got {type(got).__name__}"
            got = np.asarray(got)
        if got.shape != exp.shape:
            return f"{path} has shape {got.shape}; expected shape {exp.shape}"
        if check_dtype and got.dtype != exp.dtype:
            return f"{path} has dtype {got.dtype}; expected dtype {exp.dtype}"
        try:
            if exp.dtype.kind in "fc" or got.dtype.kind in "fc":
                same = np.isclose(got, exp, rtol=tol, atol=tol, equal_nan=True)
            else:
                same = got == exp
            same = np.asarray(same, dtype=bool)
        except (TypeError, ValueError):
            return f"{path} has the right shape but values of the wrong type ({got.dtype})"
        if same.shape != exp.shape:
            return f"{path} values differ"
        bad = int((~same).sum())
        return None if bad == 0 else f"{path}: {bad} of {exp.size} values differ"

    if check_dtype and np is not None and isinstance(exp, np.generic):
        g = got.dtype if hasattr(got, "dtype") else type(got).__name__
        if g != exp.dtype:
            return f"{path} has dtype {g}; expected dtype {exp.dtype}"

    exp, got = _plain(exp), _plain(got)

    if isinstance(exp, bool):
        if not isinstance(got, bool):
            return f"{path} should be a bool, got {type(got).__name__}"
        return None if exp == got else f"{path} is {got}, which is wrong"
    if isinstance(exp, (int, float)):
        if isinstance(got, bool) or not isinstance(got, (int, float)):
            return f"{path} should be a number, got {type(got).__name__}"
        if isinstance(exp, int) and isinstance(got, int):
            return None if exp == got else f"{path} is {got}, which is wrong"
        if math.isnan(exp) and math.isnan(got):
            return None
        return None if math.isclose(exp, got, rel_tol=tol, abs_tol=tol) else f"{path} is {got!r}, which is wrong"
    if isinstance(exp, str):
        if not isinstance(got, str):
            return f"{path} should be a str, got {type(got).__name__}"
        return None if exp == got else f"{path} is {got!r}, which is wrong"
    if exp is None:
        return None if got is None else f"{path} should be None"
    if isinstance(exp, (list, tuple)):
        if np is not None and isinstance(got, np.ndarray):
            got = got.tolist()
        if not isinstance(got, (list, tuple)):
            return f"{path} should be a {type(exp).__name__}, got {type(got).__name__}"
        if len(got) != len(exp):
            return f"{path} has length {len(got)}; expected length {len(exp)}"
        for i, (e, g) in enumerate(zip(exp, got)):
            msg = compare(e, g, tol, check_dtype, f"{path}[{i}]")
            if msg:
                return msg
        return None
    if isinstance(exp, dict):
        if not isinstance(got, dict):
            return f"{path} should be a dict, got {type(got).__name__}"
        if set(map(_plain, got)) != set(map(_plain, exp)):
            return f"{path} has the wrong keys"
        for k in exp:
            msg = compare(exp[k], got[k], tol, check_dtype, f"{path}[{k!r}]")
            if msg:
                return msg
        return None
    if isinstance(exp, (set, frozenset)):
        if not isinstance(got, (set, frozenset)):
            return f"{path} should be a set, got {type(got).__name__}"
        return None if set(map(_plain, exp)) == set(map(_plain, got)) else f"{path} has the wrong elements"
    try:
        return None if exp == got else f"{path} is wrong"
    except Exception:  # noqa: BLE001
        return f"{path} could not be compared"


# ---------------------------------------------------------------- static checks

_LOOPS = (ast.For, ast.While, ast.AsyncFor, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


def _names(tree) -> set[str]:
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            out.add(node.id)
        elif isinstance(node, ast.Attribute):
            out.add(node.attr)
        elif isinstance(node, ast.keyword) and node.arg:
            out.add(node.arg)
        elif isinstance(node, ast.alias):
            out.add(node.name.split(".")[-1])
            if node.asname:
                out.add(node.asname)
    return out


def static_check(problem: dict, tree) -> str | None:
    names = _names(tree)
    for req in problem.get("require") or []:
        alts = req.split("|")
        if not any(a in names for a in alts):
            return "This problem asks you to use " + " or ".join(f"`{a}`" for a in alts) + "."
    if problem.get("no_loops") and any(isinstance(n, _LOOPS) for n in ast.walk(tree)):
        return "Solve this one without Python loops or comprehensions: use vectorized NumPy."
    return None


# ---------------------------------------------------------------- public API

def run(problem: dict, code: str) -> dict:
    """Run setup + code like a REPL cell. -> {ok, stdout, value, error}"""
    if problem.get("kind", "python") != "python":
        return _CHECKERS[problem["kind"]](problem, code, grade=False)
    ns: dict = {"__name__": "__main__"}
    buf = io.StringIO()
    value, error = None, None
    try:
        with _sandbox():
            with contextlib.redirect_stdout(io.StringIO()):
                _exec(problem.get("setup", ""), ns, "<setup>")
            with contextlib.redirect_stdout(buf):
                last = _exec(code, ns, USER_FILE)
            if last is not _MISSING and last is not None:
                value = short_repr(last)
            elif "result" in ns:
                value = "result = " + short_repr(ns["result"])
    except BaseException as e:  # noqa: BLE001  (includes SystemExit from user code)
        error = format_error(e)
    return {"ok": error is None, "stdout": buf.getvalue(), "value": value, "error": error}


def grade(problem: dict, code: str) -> dict:
    """-> {correct, compiled, stdout, value, error, message}"""
    if problem.get("kind", "python") != "python":
        return _CHECKERS[problem["kind"]](problem, code, grade=True)
    res = {"correct": False, "compiled": True, "stdout": "", "value": None, "error": None, "message": ""}
    try:
        tree = ast.parse(code, USER_FILE)
    except SyntaxError as e:
        res["compiled"] = False
        res["error"] = format_error(e)
        return res
    msg = static_check(problem, tree)
    if msg:
        res["message"] = msg
        return res

    setups = [problem.get("setup", "")] + list(problem.get("tests") or [])
    for i, setup in enumerate(setups):
        expected, _ = _run_once(problem, problem["solution"], setup, "<solution>")
        try:
            got, stdout = _run_once(problem, code, setup, USER_FILE)
        except NoAnswer as e:
            res["message"] = str(e)
            return res
        except BaseException as e:  # noqa: BLE001
            res["error"] = format_error(e)
            if i:
                res["message"] = f"Hidden test {i} crashed."
            return res
        if i == 0:
            res["stdout"] = stdout
            res["value"] = None if problem.get("answer") == [STDOUT] else short_repr(got)
        diff = compare(expected, got, problem.get("tol", 1e-6), problem.get("check_dtype", False))
        if diff:
            res["message"] = diff if i == 0 else f"Wrong answer on hidden test {i}."
            return res
    res["correct"] = True
    res["message"] = "" if len(setups) == 1 else f"Passed all {len(setups)} tests, including the hidden ones."
    return res


# kind -> checker(problem, code, grade: bool) -> dict ; filled in by the non-Python sections
_CHECKERS: dict = {}
