"""Grading SQL problems (COMP_GUIDE §10, generator/sections/s10_sql).

The session's keys are replayed on the problem's machine (its PostgreSQL databases rebuilt from the `sql` spec) and
the reference solution is typed into a fresh copy; then, for each hidden database (same schema, other data built to
catch the classic mistakes: NULLs, ties, users without orders, duplicates, boundary values ...), both again.

sql["check"]:
  query   some statement of the session returns the target (the reference's last result) on the shown database,
          and the same statement returns the hidden databases' targets. Rows compare as a bag of typed values
          (1 = 1.0, NULL = NULL, floats within 1e-9); with sql["order"] the order counts too (ties in any order).
          Right values under other column names is "Not yet" with the name to use.
  state   the tables the reference changes (or sql["tables"]) end up with the same columns, constraints and rows;
          sql["probes"] are statements run afterwards on both (an INSERT that must fail, a SELECT ...).
  explain the probes' plans (EXPLAIN (COSTS OFF)) must use / avoid plan nodes; plus the state / query checks named in
          sql["also"].
  psql    psql's own state: settings (\\x, \\timing, \\pset null ...) and files the session wrote.
`require` / `forbid` name SQL keywords or psql commands that must / must not appear in what was run.
"""

from __future__ import annotations

import datetime as dt
import math
import re
from decimal import Decimal

from ..postgreslite import types as T


# ---------------------------------------------------------------- running sessions

def _term(problem, override=None):
    from .live import LiveTerm
    p = dict(problem)
    if override is not None:
        p["_sql_setup"] = override
    return LiveTerm(p, problem.get("setup", ""))


def user_run(problem, log, end, override=None):
    t = _term(problem, override)
    t.replay(log, end)
    return t


def reference_run(problem, override=None):
    """Type the reference solution: lines into psql (or the shell, for problems that start there)."""
    t = _term(problem, override)
    type_lines(t, problem["solution"])
    return t


def type_lines(t, text):
    from .live import PROMPT_WAIT
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    for line in lines:
        if line:
            t.feed("text", line)
        t.feed("key", "Enter")
        t.advance(t.k.now + PROMPT_WAIT, cond=lambda: t.where() != "r")
        while t.where() == "e":
            t.edit("close")
    t.finish()
    return t


def sql_log(t):
    return list(getattr(t.world, "sql_log", None) or [])


def results(t):
    """[(statement index, result record)] for every result with rows, in order."""
    out = []
    for i, rec in enumerate(sql_log(t)):
        for r in rec["results"]:
            if r.get("names") is not None:
                out.append((i, r))
    return out


def target_of(t, spec):
    res = results(t)
    if not res:
        return None
    which = spec.get("target", -1)
    return res[which][1]


# ---------------------------------------------------------------- comparing results

def _norm(v, tname):
    """A comparable form: numbers by value, date / timestamp by instant, json canonical, arrays element-wise."""
    if v is None:
        return None
    if isinstance(v, bool):
        return ("b", v)
    if isinstance(v, (int, Decimal)) and not isinstance(v, bool):
        d = Decimal(v)
        if d.is_nan():
            return ("nan",)
        return ("n", d)
    if isinstance(v, float):
        if math.isnan(v):
            return ("nan",)
        return ("f", v)
    if isinstance(v, dt.datetime):
        if v.tzinfo is not None:
            v = v.astimezone(T.UTC).replace(tzinfo=None)
        return ("t", v)
    if isinstance(v, dt.date):
        return ("t", dt.datetime(v.year, v.month, v.day))
    if isinstance(v, str):
        if tname == "bpchar":
            v = v.rstrip(" ")
        return ("s", v)
    if isinstance(v, T.Interval):
        return ("i", v.key())
    if isinstance(v, T.Json):
        return ("j", T._freeze(v.v))
    if isinstance(v, list):
        return ("a", tuple(_norm(x, None) for x in v))
    if isinstance(v, tuple):
        return ("r", tuple(_norm(x, None) for x in v))
    return ("x", repr(v))


def _eq(a, b):
    if a is None or b is None:
        return a is None and b is None
    if a[0] in ("n", "f") and b[0] in ("n", "f"):
        x, y = float(a[1]), float(b[1])
        if a[0] == "n" and b[0] == "n":
            return a[1] == b[1]
        return x == y or abs(x - y) <= 1e-9 * max(1.0, abs(x), abs(y))
    return a == b


def _row_key(row, types):
    out = []
    for v, t in zip(row, types):
        n = _norm(v, t)
        if n is not None and n[0] == "f":
            n = ("n", Decimal(repr(round(n[1], 9))))
        elif n is not None and n[0] == "n":
            n = ("n", n[1].normalize() if n[1] != 0 else Decimal(0))
        out.append(n)
    return tuple(out)


def _show(v, t):
    if v is None:
        return "NULL"
    from ..postgreslite.types import output, TYPES
    tt = T.TYPES.get(t) if isinstance(t, str) else t
    try:
        s = output(v, tt) if tt is not None else str(v)
    except Exception:  # noqa: BLE001
        s = str(v)
    return s


def show_row(row, types):
    return "(" + ", ".join(_show(v, t) for v, t in zip(row, types)) + ")"


def result_matches(spec, exp, got):
    """None when got matches exp, else what differs (one sentence, plus details)."""
    en, gn = exp["names"], got["names"]
    if len(en) != len(gn):
        return (f"it returns {len(gn)} column{'s' if len(gn) != 1 else ''} ({', '.join(gn)}); expected {len(en)} "
                f"({', '.join(en)})")
    et, gt = exp["types"], got["types"]
    erows, grows = exp["rows"], got["rows"]
    if len(erows) != len(grows):
        return f"it returns {len(grows)} row{'s' if len(grows) != 1 else ''}; expected {len(erows)}"
    ek = [_row_key(r, et) for r in erows]
    gk = [_row_key(r, gt) for r in grows]
    # bag comparison (floats within tolerance)
    if not _bag_equal(ek, gk):
        missing, extra = _bag_diff(ek, gk, erows, grows)
        parts = []
        if missing:
            parts.append("missing " + ", ".join(show_row(r, et) for r in missing[:3]) +
                         (" …" if len(missing) > 3 else ""))
        if extra:
            parts.append("unexpected " + ", ".join(show_row(r, gt) for r in extra[:3]) +
                         (" …" if len(extra) > 3 else ""))
        return "the values differ: " + "; ".join(parts)
    if spec.get("order"):
        msg = _order_ok(exp, ek, gk)
        if msg:
            return msg
    return None


def names_match(exp, got):
    """-> None, or the first column whose name differs (1-based, expected name)."""
    for i, (a, b) in enumerate(zip(exp["names"], got["names"])):
        if a.lower() != b.lower():
            return i + 1, a, b
    return None


def _bag_equal(ek, gk):
    if sorted(map(repr, ek)) == sorted(map(repr, gk)):
        return True
    rest = list(gk)
    for row in ek:
        hit = None
        for j, g in enumerate(rest):
            if all(_eq(a, b) for a, b in zip(row, g)):
                hit = j
                break
        if hit is None:
            return False
        rest.pop(hit)
    return not rest


def _bag_diff(ek, gk, erows, grows):
    rest = list(range(len(gk)))
    missing = []
    for i, row in enumerate(ek):
        hit = None
        for j in rest:
            if all(_eq(a, b) for a, b in zip(row, gk[j])):
                hit = j
                break
        if hit is None:
            missing.append(erows[i])
        else:
            rest.remove(hit)
    extra = [grows[j] for j in rest]
    return missing, extra


def _order_ok(exp, ek, gk):
    """The rows must come in the reference's order; rows the reference's ORDER BY can't tell apart (ties) may come
    in any order among themselves."""
    order = exp.get("order")
    if order and exp.get("order_visible"):
        cols = [i for i, *_ in order]

        def tie(k):
            return tuple(repr(k[i]) for i in cols)
        groups = []
        for k in ek:
            if groups and tie(groups[-1][0]) == tie(k):
                groups[-1].append(k)
            else:
                groups.append([k])
        pos = 0
        for g in groups:
            chunk = gk[pos:pos + len(g)]
            if sorted(map(repr, chunk)) != sorted(map(repr, g)) and not _bag_equal(g, chunk):
                return "the rows are right but not in the order asked for"
            pos += len(g)
        return None
    for a, b in zip(ek, gk):
        if not all(_eq(x, y) for x, y in zip(a, b)):
            return "the rows are right but not in the order asked for"
    return None


# ---------------------------------------------------------------- state

def table_state(t, dbname, table):
    cl = t.world.local.extra.get("pg")
    if cl is None or dbname not in cl.databases:
        return None
    db = cl.databases[dbname]
    schema = "public"
    if "." in table:
        schema, table = table.split(".", 1)
    rel = db.find_rel(table, schema, None)
    if rel is None:
        return None
    from ..postgreslite.catalog import Table, View
    cols = [(c.name, c.type.format(), bool(getattr(c, "not_null", False))) for c in rel.columns]
    cons = []
    if isinstance(rel, Table):
        for k in rel.constraints:
            if k.kind in ("p", "u"):
                cons.append((k.kind, tuple(sorted(k.cols))))
            elif k.kind == "f":
                cons.append(("f", tuple(k.cols), k.ref_table, tuple(k.ref_cols), k.on_delete))
            elif k.kind == "c":
                cons.append(("c", tuple(sorted(k.cols))))
        idx = sorted((tuple(el.col or el.expr_sql for el in ix.elems), ix.unique, ix.method, bool(ix.pred_sql))
                     for ix in db.indexes_of(rel) if ix.constraint is None)
    else:
        idx = []
    kind = "view" if isinstance(rel, View) and not rel.materialized else (
        "matview" if isinstance(rel, View) else "table")
    rows = rel.rows if kind != "view" else None
    types = [c.type.name for c in rel.columns]
    return {"kind": kind, "cols": cols, "cons": sorted(map(repr, cons)), "idx": idx, "rows": rows, "types": types,
            "defaults": [c.default_sql or (("identity " + c.identity) if c.identity else None)
                         for c in rel.columns] if hasattr(rel.columns[0] if rel.columns else None, "default_sql")
            else None}


def compare_state(spec, ref, got, name):
    if ref is None and got is None:
        return None
    if ref is None:
        return f"`{name}` shouldn't exist"
    if got is None:
        return f"there is no {ref['kind']} `{name}`"
    if ref["kind"] != got["kind"]:
        return f"`{name}` should be a {ref['kind']}, not a {got['kind']}"
    rc = [(c[0], c[1]) for c in ref["cols"]]
    gc = [(c[0], c[1]) for c in got["cols"]]
    if [c[0] for c in rc] != [c[0] for c in gc]:
        return f"`{name}` should have the columns ({', '.join(c[0] for c in rc)}), not ({', '.join(c[0] for c in gc)})"
    want = spec.get("compare", ("types", "nullable", "constraints", "rows", "defaults"))
    if "types" in want:
        for (n, a), (_, b) in zip(rc, gc):
            if a != b:
                return f"column `{name}.{n}` should be {a}, not {b}"
    if "nullable" in want:
        for a, b in zip(ref["cols"], got["cols"]):
            if a[2] != b[2]:
                return f"column `{name}.{a[0]}` should be {'NOT NULL' if a[2] else 'nullable'}"
    if "defaults" in want and ref.get("defaults") is not None and got.get("defaults") is not None:
        for c, a, b in zip(ref["cols"], ref["defaults"], got["defaults"]):
            if (a is None) != (b is None):
                return f"column `{name}.{c[0]}` should {'have a default' if a else 'not have a default'}"
    if "constraints" in want and ref["cons"] != got["cons"]:
        miss = sorted(set(ref["cons"]) - set(got["cons"]))
        extra = sorted(set(got["cons"]) - set(ref["cons"]))
        if miss:
            return f"`{name}` is missing a constraint: {_cons_text(miss[0])}"
        return f"`{name}` has a constraint it shouldn't: {_cons_text(extra[0])}"
    if "indexes" in want and ref["idx"] != got["idx"]:
        return f"the indexes on `{name}` aren't the ones asked for"
    if "rows" in want and ref["rows"] is not None:
        e = {"names": [c[0] for c in ref["cols"]], "types": ref["types"], "rows": ref["rows"]}
        g = {"names": [c[0] for c in got["cols"]], "types": got["types"], "rows": got["rows"]}
        msg = result_matches({}, e, g)
        if msg:
            return f"the rows of `{name}` differ: {msg}"
    return None


def _cons_text(c):
    c = eval(c) if isinstance(c, str) else c   # noqa: S307  (our own repr of tuples of names)
    kind = c[0]
    if kind == "p":
        return f"PRIMARY KEY ({', '.join(c[1])})"
    if kind == "u":
        return f"UNIQUE ({', '.join(c[1])})"
    if kind == "f":
        act = {"c": " ON DELETE CASCADE", "n": " ON DELETE SET NULL", "r": " ON DELETE RESTRICT",
               "d": " ON DELETE SET DEFAULT"}.get(c[4], "")
        return f"FOREIGN KEY ({', '.join(c[1])}) REFERENCES {c[2]}({', '.join(c[3])}){act}"
    return f"a CHECK on ({', '.join(c[1])})"


def changed_tables(t_ref, spec):
    if spec.get("tables"):
        return list(spec["tables"])
    names = []
    for rec in sql_log(t_ref):
        m = re.match(r"\s*(?:insert\s+into|update|delete\s+from|create\s+(?:temp\s+|temporary\s+|unlogged\s+)?"
                     r"(?:table|(?:materialized\s+)?view|unique\s+index|index)(?:\s+if\s+not\s+exists)?|"
                     r"alter\s+table|truncate(?:\s+table)?|merge\s+into|drop\s+(?:table|view)(?:\s+if\s+exists)?)"
                     r"\s+(?:only\s+)?([A-Za-z_][\w.]*)", rec["sql"], re.I)
        if m:
            n = m.group(1).lower()
            if n not in names:
                names.append(n)
    return names


def run_probes(t, spec, dbname):
    """Statements run after the session: -> [(ok, columns, rows or SQLSTATE class)]."""
    from ..postgreslite.errors import PgError
    cl = t.world.local.extra.get("pg")
    out = []
    for sql in spec.get("probes") or []:
        s = cl.connect(dbname, "ada")
        try:
            res = s.execute(sql)
            r = res[-1]
            out.append(("ok", r.names, [tuple(x) for x in (r.rows or [])], [tt.name for tt in (r.types or [])]))
        except PgError as e:
            out.append(("error", e.code[:2], e.message, None))
        finally:
            s.close()
    return out


def compare_probes(spec, ref, got):
    for k, (a, b) in enumerate(zip(ref, got)):
        sql = spec["probes"][k]
        if a[0] != b[0]:
            if a[0] == "error":
                return f"`{sql}` should fail ({_why(a)}) but it runs"
            return f"`{sql}` should work but fails: {b[2]}"
        if a[0] == "error":
            if a[1] != b[1]:
                return f"`{sql}` should fail with a {_why(a)}, not: {b[2]}"
            continue
        e = {"names": a[1] or [], "types": a[3] or [], "rows": a[2]}
        g = {"names": b[1] or [], "types": b[3] or [], "rows": b[2]}
        if e["names"] and result_matches({}, e, g):
            return f"after your session, `{sql}` gives a different result"
    return None


def _why(a):
    return {"23": "constraint violation", "22": "data error", "42": "an error in the statement",
            "25": "transaction error"}.get(a[1], "error") + f" ({a[2]})"


# ---------------------------------------------------------------- plans

def plan_nodes(t, dbname, query):
    from ..postgreslite.errors import PgError
    cl = t.world.local.extra.get("pg")
    s = cl.connect(dbname, "ada")
    try:
        r = s.execute("EXPLAIN (COSTS OFF) " + query)[-1]
        return [row[0] for row in r.rows]
    except PgError as e:
        return ["ERROR: " + e.message]
    finally:
        s.close()


def check_plans(spec, t):
    for item in spec.get("plans") or []:
        lines = plan_nodes(t, spec["db"], item["query"])
        text = "\n".join(lines)
        if lines and lines[0].startswith("ERROR"):
            return f"the plan of `{item['query']}` can't be made: {lines[0][7:]}"
        for want in item.get("uses", []):
            if not re.search(want, text):
                return item.get("why") or f"the plan of `{item['query']}` should use {want.replace('|', ' or ')}"
        for bad in item.get("avoids", []):
            if re.search(bad, text):
                return item.get("why_avoid") or f"the plan of `{item['query']}` still uses {bad}"
    return None


# ---------------------------------------------------------------- features (require / forbid)

def session_text(t):
    """Everything psql was sent and the meta-commands typed, for require / forbid."""
    parts = [rec["sql"] for rec in sql_log(t)]
    term = getattr(t.world, "psql_last", None)
    if term is not None:
        parts += term.psql.history
    parts += [m["raw"] for m in t.marks]
    return "\n".join(parts)


def _strip_strings(sql):
    return re.sub(r"'(?:[^']|'')*'", "''", sql)


def feature_used(text, name):
    """A keyword / function / psql command in the session: "LEFT JOIN", "COALESCE(", "\\x", "OVER" ..."""
    s = _strip_strings(text)
    if name.startswith("\\"):
        return re.search(re.escape(name) + r"(?![A-Za-z])", s) is not None
    if name.endswith("("):
        return re.search(r"\b" + re.escape(name[:-1]) + r"\s*\(", s, re.I) is not None
    if not re.match(r"\w", name):
        return name in s
    pat = r"\b" + r"\s+".join(re.escape(w) for w in name.split()) + r"\b"
    return re.search(pat, s, re.I) is not None


def _alts(req):
    return [x.strip() for x in req.split("|")]


def missing_feature(problem, text):
    for req in problem.get("require") or []:
        if not any(feature_used(text, a) for a in _alts(req)):
            names = " or ".join(f"`{a}`" for a in _alts(req))
            return f"This exercise practises {names}: use it in your answer."
    for bad in problem.get("forbid") or []:
        if feature_used(text, bad):
            return f"Solve it without `{bad}`."
    return None


# ---------------------------------------------------------------- the checker

def _result(**kw):
    out = {"correct": False, "compiled": True, "stdout": "", "value": None, "error": None, "message": "",
           "console": []}
    out.update(kw)
    return out


def checker(problem, log, end):
    spec = problem["sql"]
    check = spec.get("check", "query")
    t = user_run(problem, log, end)
    if not sql_log(t) and check in ("query", "state", "explain", "psql") and not t.marks and \
            not (getattr(t.world, "psql_last", None) and t.world.psql_last.psql.history):
        return _result(compiled=False, message="Nothing to check yet: run your SQL in the terminal, then Submit.")
    ref = reference_run(problem)
    text = session_text(t)
    msg = missing_feature(problem, text)
    if msg:
        return _result(compiled=False, message="Not yet: " + msg)
    hidden = list(spec.get("hidden") or [])
    tags = list(spec.get("hidden_tags") or [])
    if check == "query":
        return _query_check(problem, spec, t, ref, hidden, tags, log, end)
    if check in ("state", "explain"):
        return _state_check(problem, spec, t, ref, hidden, tags, log, end, check)
    if check == "psql":
        return _psql_check(problem, spec, t, ref)
    raise ValueError(check)


def _hidden_label(tags, i):
    tag = tags[i] if i < len(tags) else None
    return f"a hidden test database ({tag})" if tag else f"hidden test database {i + 1}"


def _query_check(problem, spec, t, ref, hidden, tags, log, end):
    target = target_of(ref, spec)
    if target is None:
        raise AssertionError("the reference solution returned no rows")
    mine = results(t)
    if not mine:
        errs = [rec for rec in sql_log(t) if rec["error"]]
        if errs:
            return _result(message=f"Your last query failed: {errs[-1]['error'][1]}. Fix it and run it again.")
        return _result(compiled=False, message="Nothing to check yet: run a query that returns the answer, then "
                                               "Submit.")
    hits = [(i, r) for i, r in mine if result_matches(spec, target, r) is None]
    if not hits:
        i, r = mine[-1]
        why = result_matches(spec, target, r)
        return _result(message=f"Not the expected result yet. Your last query: {why}.")
    # right values: names?
    good = [(i, r) for i, r in hits if names_match(target, r) is None]
    if not good:
        k, want, have = names_match(target, hits[-1][1])
        return _result(compiled=False, message=f"Not yet: the values are right; name column {k} `{want}` "
                                               f"(it's `{have}`), e.g. with AS {want}.")
    cands = [i for i, _ in good]
    for h, setup in enumerate(hidden):
        th = user_run(problem, log, end, override=setup)
        rh = reference_run(problem, override=setup)
        tgt = target_of(rh, spec)
        logh = sql_log(th)
        ok = []
        why = None
        for i in cands:
            if i >= len(logh):
                continue
            rr = [r for r in logh[i]["results"] if r.get("names") is not None]
            if not rr:
                why = why or (f"your query fails there: {logh[i]['error'][1]}" if logh[i]["error"] else
                              "your query returns no rows there")
                continue
            m = result_matches(spec, tgt, rr[-1])
            if m is None:
                ok.append(i)
            else:
                why = why or m
        if not ok:
            return _result(message=f"Right on this database, wrong on {_hidden_label(tags, h)}: {why}. "
                                   "Make the query correct for any data, not just the rows shown.")
        cands = ok
    n = len(hidden)
    return _result(correct=True, message="" if not n else f"Correct on this database and on {n} hidden one"
                                                         f"{'s' if n != 1 else ''}.")


def _state_check(problem, spec, t, ref, hidden, tags, log, end, check):
    db = spec["db"]
    tables = changed_tables(ref, spec)
    for k in range(len(hidden) + 1):
        if k:
            setup = hidden[k - 1]
            tu = user_run(problem, log, end, override=setup)
            tr = reference_run(problem, override=setup)
            where = " on " + _hidden_label(tags, k - 1)
        else:
            tu, tr, where = t, ref, ""
        status_msg = _txn_status(spec, tu, tr)
        if status_msg:
            return _result(message=status_msg + where + ".")
        for name in tables:
            m = compare_state(spec, table_state(tr, db, name), table_state(tu, db, name), name)
            if m:
                return _result(message=m[0].upper() + m[1:] + where + ".")
        pm = compare_probes(spec, run_probes(tr, spec, db), run_probes(tu, spec, db))
        if pm:
            return _result(message=pm[0].upper() + pm[1:] + where + ".")
        if check == "explain" or spec.get("plans"):
            m = check_plans(spec, tu)
            if m:
                return _result(message=m[0].upper() + m[1:] + where + ".")
        if spec.get("also") == "query":
            target = target_of(tr, spec)
            if target is not None and not any(result_matches(spec, target, r) is None for _, r in results(tu)):
                return _result(message="Your session doesn't show the result asked for" + where + ".")
    return _result(correct=True, message="" if not hidden else
                   f"Correct on this database and on {len(hidden)} hidden one{'s' if len(hidden) != 1 else ''}.")


def _txn_status(spec, tu, tr):
    """A transaction left open (or aborted) at the end isn't committed work."""
    term = getattr(tu.world, "psql_last", None)
    if term is None:
        return None
    st = term.psql.session.status if term.psql.session is not None else "I"
    want = "I"
    rterm = getattr(tr.world, "psql_last", None)
    if rterm is not None and rterm.psql.session is not None:
        want = rterm.psql.session.status
    if st != want:
        if st == "T":
            return "Your transaction is still open: finish it with COMMIT (or ROLLBACK)"
        if st == "E":
            return "Your transaction is aborted (an error happened inside it): ROLLBACK, or ROLLBACK TO a savepoint"
    return None


def _psql_check(problem, spec, t, ref):
    term = getattr(t.world, "psql_last", None)
    rterm = getattr(ref.world, "psql_last", None)
    for name in spec.get("settings") or []:
        a = _setting(rterm, name)
        b = _setting(term, name)
        if a != b:
            return _result(message=f"psql's `{name}` should be {a!r}, not {b!r}.")
    for path in spec.get("files") or []:
        a = _file(ref, path)
        b = _file(t, path)
        if b is None:
            return _result(message=f"There is no file `{path}`.")
        if a != b:
            return _result(message=f"`{path}` doesn't contain what was asked for.")
    for path, sql in spec.get("sqlite") or []:
        a = _sqlite_rows(ref, path, sql)
        b = _sqlite_rows(t, path, sql)
        if b is None:
            return _result(message=f"There is no SQLite database `{path}` (or `{sql}` fails on it).")
        if a != b:
            return _result(message=f"In `{path}`, `{sql}` doesn't give what it should.")
    if spec.get("also") == "query":
        target = target_of(ref, spec)
        if target is not None and not any(result_matches(spec, target, r) is None for _, r in results(t)):
            return _result(message="Your session doesn't show the result asked for.")
    return _result(correct=True)


def _sqlite_rows(t, path, sql):
    import sqlite3
    full = path if path.startswith("/") else t.home + "/" + path
    try:
        data = bytes(t.fs.lookup(full).data)
    except Exception:  # noqa: BLE001
        return None
    con = sqlite3.connect(":memory:")
    try:
        if data:
            con.deserialize(data)
        rows = con.execute(sql).fetchall()
    except sqlite3.Error:
        return None
    finally:
        con.close()
    return sorted(rows, key=repr)


def _setting(term, name):
    if term is None:
        return None
    p = term.psql
    o = p.opts
    return {"expanded": o.expanded, "timing": "on" if p.timing else "off", "null": o.null, "format": o.format,
            "border": o.border, "tuples_only": o.tuples_only, "footer": o.footer}.get(name, p.vars.get(name))


def _file(t, path):
    home = t.home
    full = path if path.startswith("/") else home + "/" + path
    try:
        node = t.fs.lookup(full)
    except Exception:  # noqa: BLE001
        return None
    return bytes(node.data).decode("utf-8", "replace")
