import numpy as np

from generator.grader import compare, grade, run

P = {"setup": "import numpy as np\nx = np.array([3, 1, 2])", "solution": "np.sort(x)", "answer": []}


def test_last_expression_and_result_var():
    assert grade(P, "np.sort(x)")["correct"]
    assert grade(P, "result = np.sort(x)\nprint(result)")["correct"]
    assert not grade(P, "print(np.sort(x))")["correct"]


def test_errors_are_reported():
    r = grade(P, "x.sortt()")
    assert not r["correct"] and "AttributeError" in r["error"]
    r = grade(P, "np.sort(x")
    assert "SyntaxError" in r["error"]


def test_user_lookup_errors_are_errors_not_missing_answers():
    r = grade(P, "x[10]")
    assert "IndexError" in r["error"]


def test_require_and_no_loops():
    p = dict(P, require=["sort"], no_loops=True)
    assert "use `sort`" in grade(p, "np.array(sorted(x))")["message"]
    assert "loops" in grade(p, "np.sort(np.array([v for v in x]))")["message"]


def test_hidden_tests():
    p = dict(P, solution="x.max()", tests=["import numpy as np\nx = np.array([9, 4])"])
    assert grade(p, "x.max()")["correct"]
    assert "hidden test 1" in grade(p, "3")["message"]


def test_compare():
    assert compare(np.array([1.0, 2.0]), np.array([1.0, 2.0 + 1e-9])) is None
    assert "shape" in compare(np.zeros((2, 3)), np.zeros((3, 2)))
    assert compare(np.float64(2.5), 2.5) is None
    assert compare((1, 2), [1, 2]) is None
    assert compare(np.array(["a", "b"]), ["a", "b"]) is None
    assert compare({"a": 1}, {"a": 2})
    assert compare(np.array([np.nan, 1.0]), np.array([np.nan, 1.0])) is None
    assert "dtype" in compare(np.zeros(2, dtype=np.float32), np.zeros(2), check_dtype=True)


def test_run_shows_value_and_stdout():
    r = run(P, "print('hi')\nx * 2")
    assert r["stdout"] == "hi\n" and r["value"] == "array([6, 2, 4])"
    assert "ZeroDivisionError" in run(P, "1/0")["error"]


def test_stdout_answers_and_post_checks():
    p = dict(P, solution="print(np.sort(x))", answer=["<stdout>"])
    assert grade(p, "print(np.sort(x))")["correct"]
    assert not grade(p, "np.sort(x)")["correct"]
    p = dict(P, solution="np.save('s.npy', np.sort(x))", post="np.load('s.npy')")
    assert grade(p, "np.save('s.npy', np.sort(x))")["correct"]
    r = grade(p, "pass")
    assert not r["correct"] and "FileNotFoundError" in r["message"]


def test_runs_are_sandboxed():
    p = dict(P, solution="np.save('s.npy', x)", post="np.load('s.npy')")
    grade(p, "np.save('s.npy', x)")
    assert not grade(p, "pass")["correct"]  # the reference's file must not leak into the user's run
    p = dict(P, solution="repr(np.pi * np.ones(2))")
    grade(p, "np.set_printoptions(precision=1)\nrepr(np.pi * np.ones(2))")
    assert grade(p, "repr(np.pi * np.ones(2))")["correct"]  # global print options were restored
