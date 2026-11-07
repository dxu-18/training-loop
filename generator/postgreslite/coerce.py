"""Overload resolution and type unification, after parse_func.c, parse_oper.c and parse_coerce.c:
which function or operator a call means, polymorphic types (anyelement, anycompatible ...), and the common type of
CASE / COALESCE / UNION / VALUES / ARRAY[] branches. Errors use PostgreSQL's wording."""

from __future__ import annotations

from .errors import PgError
from . import types as T
from .types import Type

HINT_FUNC = "No function matches the given name and argument types. You might need to add explicit type casts."
HINT_OP = "No operator matches the given name and argument types. You might need to add explicit type casts."


def _is_poly(t: Type) -> bool:
    return t.name in T.POLY


def _preferred(t: Type) -> bool:
    return t.pref


def _consistent(params: list[Type], args: list[Type]) -> bool:
    """check_generic_type_consistency: anyelement / anyarray arguments must agree."""
    elem = None
    for p, a in zip(params, args):
        if a.name == "unknown":
            continue
        if p.name in ("anyelement", "anynonarray", "anyenum"):
            e = a
        elif p.name == "anyarray":
            if not a.is_array:
                return False
            e = a.elem
        else:
            continue
        if p.name == "anynonarray" and a.is_array:
            return False
        if elem is None:
            elem = e
        elif elem != e:
            return False
    # anycompatible: a common type must exist
    comp = []
    for p, a in zip(params, args):
        if a.name == "unknown":
            continue
        if p.name in ("anycompatible", "anycompatiblenonarray"):
            comp.append(a)
        elif p.name == "anycompatiblearray":
            if not a.is_array:
                return False
            comp.append(a.elem)
    if comp:
        cats = {c.cat for c in comp}
        if len(cats) > 1:
            return False
    return True


def resolve_poly(params: list[Type], args: list[Type], ret: Type, what: str = "") -> tuple[list[Type], Type]:
    """enforce_generic_type_consistency: the concrete parameter types and result type."""
    elem = None
    for p, a in zip(params, args):
        if a.name == "unknown":
            continue
        if p.name in ("anyelement", "anynonarray", "anyenum"):
            elem = elem or a
        elif p.name == "anyarray" and a.is_array:
            elem = elem or a.elem
    comp_in = []
    for p, a in zip(params, args):
        if a.name == "unknown":
            continue
        if p.name in ("anycompatible", "anycompatiblenonarray"):
            comp_in.append(a)
        elif p.name == "anycompatiblearray" and a.is_array:
            comp_in.append(a.elem)
    comp = None
    if comp_in:
        comp = common_type(comp_in, None, None, silent=True) or comp_in[0]
    elif any(p.name.startswith("anycompatible") for p in params):
        if any(a.name == "unknown" for a, p in zip(args, params) if p.name.startswith("anycompatible")):
            comp = T.TEXT
    has_any = any(p.name in ("anyelement", "anyarray", "anynonarray", "anyenum") for p in params)
    if has_any and elem is None:
        if any(a.name == "unknown" for a in args):
            if ret.name in ("anyelement", "anyarray", "anynonarray") or any(
                    p.name in ("anyarray",) for p in params):
                raise PgError("could not determine polymorphic type because input has type unknown", "42804")
            elem = T.TEXT
        else:
            elem = T.TEXT
    out = []
    for p, a in zip(params, args):
        n = p.name
        if n in ("anyelement", "anynonarray", "anyenum"):
            out.append(elem)
        elif n == "anyarray":
            out.append(elem.array())
        elif n in ("anycompatible", "anycompatiblenonarray"):
            out.append(comp)
        elif n == "anycompatiblearray":
            out.append(comp.array())
        elif n == "any":
            out.append(a if a.name != "unknown" else T.TEXT)
        else:
            out.append(p)
    r = ret
    if ret.name in ("anyelement", "anynonarray", "anyenum"):
        r = elem
    elif ret.name == "anyarray":
        r = elem.array() if elem is not None else ret
    elif ret.name in ("anycompatible", "anycompatiblenonarray"):
        r = comp
    elif ret.name == "anycompatiblearray":
        r = comp.array()
    return out, r


def _expand_variadic(p: T.Proc, nargs: int):
    """The parameter list for a call with nargs arguments (VARIADIC expands), or None if it can't fit."""
    if p.variadic is not None:
        fixed = p.args[:-1]
        if nargs < len(fixed):
            return None
        if nargs == len(fixed) and p.variadic.name != "any":
            return None if nargs != len(p.args) else None
        return fixed + [p.variadic] * (nargs - len(fixed))
    if len(p.args) == nargs:
        return list(p.args)
    nd = sum(1 for x in (getattr(p, "defaults", None) or []) if x is not None)
    if nd and len(p.args) > nargs >= len(p.args) - nd:
        return list(p.args[:nargs])
    return None


# functions whose trailing arguments have defaults
DEFAULTS = {"jsonb_set": 1, "jsonb_insert": 1, "date_bin": 0, "make_interval": 7, "make_timestamp": 0,
            "regexp_replace": 0, "jsonb_path_query": 2, "to_timestamp": 0, "array_fill": 1}


def select_candidate(cands: list, args: list[Type]):
    """func_select_candidate / oper_select_candidate over [(params, payload)]. -> payload, or "ambiguous", or None."""
    if not cands:
        return None
    nargs = len(args)
    # exact match
    for params, payload in cands:
        if all(p == a for p, a in zip(params, args)) and not any(_is_poly(p) for p in params):
            return payload
    # binary operators: an unknown operand is assumed to be the other operand's type
    if nargs == 2 and (args[0].name == "unknown") != (args[1].name == "unknown"):
        known = args[1] if args[0].name == "unknown" else args[0]
        for params, payload in cands:
            if params[0] == known and params[1] == known:
                return payload
    viable = [(p, x) for p, x in cands if all(T.can_coerce(a, q) for a, q in zip(args, p)) and _consistent(p, args)]
    if not viable:
        return None
    if len(viable) == 1:
        return viable[0][1]
    base = list(args)
    nunknown = sum(1 for a in base if a.name == "unknown")
    # most exact matches
    best, keep = -1, []
    for params, payload in viable:
        n = sum(1 for a, p in zip(base, params) if a.name != "unknown" and a == p)
        if n > best:
            best, keep = n, [(params, payload)]
        elif n == best:
            keep.append((params, payload))
    viable = keep
    if len(viable) == 1:
        return viable[0][1]
    # exact or preferred types where coercion is needed
    best, keep = -1, []
    for params, payload in viable:
        n = sum(1 for a, p in zip(base, params) if a.name != "unknown" and (a == p or _preferred(p)))
        if n > best:
            best, keep = n, [(params, payload)]
        elif n == best:
            keep.append((params, payload))
    viable = keep
    if len(viable) == 1:
        return viable[0][1]
    if nunknown == 0:
        return "ambiguous"
    # unknown arguments: prefer the string category, else a category every candidate agrees on
    resolved = True
    slot_cat = [None] * nargs
    slot_pref = [False] * nargs
    for i, a in enumerate(base):
        if a.name != "unknown":
            continue
        cat = None
        pref = False
        conflict = False
        for params, _ in viable:
            p = params[i]
            pc = p.cat
            if cat is None:
                cat, pref = pc, p.pref
            elif pc == cat:
                pref = pref or p.pref
            elif pc == "S":
                cat, pref = "S", p.pref
            else:
                conflict = True
        if conflict and cat != "S":
            resolved = False
            break
        slot_cat[i], slot_pref[i] = cat, pref
    if resolved:
        keep = []
        for params, payload in viable:
            ok = True
            for i, a in enumerate(base):
                if a.name != "unknown":
                    continue
                p = params[i]
                if p.cat != slot_cat[i] or (slot_pref[i] and not p.pref):
                    ok = False
                    break
            if ok:
                keep.append((params, payload))
        if len(keep) == 1:
            return keep[0][1]
        if keep:
            viable = keep
    # unknowns take the known arguments' type when they all share one
    if nunknown < nargs:
        known = {a.name for a in base if a.name != "unknown"}
        if len(known) == 1:
            kt = next(a for a in base if a.name != "unknown")
            keep = [(params, payload) for params, payload in viable
                    if all(T.can_coerce(kt, params[i]) for i, a in enumerate(base) if a.name == "unknown")]
            if len(keep) == 1:
                return keep[0][1]
    return "ambiguous"


def fmt_args(types: list[Type]) -> str:
    return ", ".join(t.format() if t.mod is None else t.base.format() for t in types)


def find_function(name: str, args: list[Type], pos=None, *, star=False, kinds=None, qualified=None):
    """-> (Proc, param types (concrete), result type). Raises PgError like the real server."""
    procs = T.PROCS.get(name, [])
    if kinds is not None:
        procs = [p for p in procs if p.kind in kinds]
    cands = []
    for p in procs:
        params = _expand_variadic(p, len(args))
        if params is not None:
            cands.append((params, p))
    choice = select_candidate(cands, args)
    shown = (qualified + "." if qualified else "") + name
    if choice is None:
        sig = "*" if star else fmt_args(args)
        raise PgError(f"function {shown}({sig}) does not exist", "42883", pos=pos, hint=HINT_FUNC)
    if choice == "ambiguous":
        raise PgError(f"function {shown}({fmt_args(args)}) is not unique", "42725", pos=pos,
                      hint="Could not choose a best candidate function. You might need to add explicit type casts.")
    params = _expand_variadic(choice, len(args))
    concrete, ret = resolve_poly(params, args, choice.ret)
    return choice, concrete, ret


def find_operator(op: str, left: Type | None, right: Type, pos=None):
    """-> (param types, result type)."""
    cands = []
    for lt, rt, res in T.OPS.get(op, []):
        if (lt is None) != (left is None):
            continue
        params = [rt] if lt is None else [lt, rt]
        cands.append((params, (params, res)))
    args = [right] if left is None else [left, right]
    choice = select_candidate(cands, args)
    if choice is None or choice == "ambiguous":
        shown = f"{op} {right.format()}" if left is None else f"{left.format()} {op} {right.format()}"
        if choice == "ambiguous":
            raise PgError(f"operator is not unique: {shown}", "42725", pos=pos,
                          hint="Could not choose a best candidate operator. You might need to add explicit type "
                               "casts.")
        raise PgError(f"operator does not exist: {shown}", "42883", pos=pos, hint=HINT_OP)
    params, res = choice
    concrete, ret = resolve_poly(params, args, res)
    return concrete, ret


def common_type(types: list[Type], context: str | None, positions=None, silent=False):
    """select_common_type: the type CASE / UNION / COALESCE ... branches are converted to."""
    known = [(i, t) for i, t in enumerate(types) if t.name != "unknown"]
    if not known:
        return T.TEXT
    first_i, ptype = known[0]
    if all(t == ptype for _, t in known):
        return ptype.base if ptype.mod is not None and not all(t.mod == ptype.mod for _, t in known) else ptype
    pcat = ptype.cat
    ppref = ptype.pref
    for i, t in known[1:]:
        if t == ptype:
            continue
        if t.cat != pcat:
            if silent:
                return None
            pos = positions[i] if positions else None
            raise PgError(f"{context} types {ptype.format()} and {t.format()} cannot be matched", "42804", pos=pos)
        if not ppref and T.can_coerce(ptype, t) and not T.can_coerce(t, ptype):
            ptype, ppref = t, t.pref
    return ptype.base


def check_common(types: list[Type], target: Type, context: str, positions=None):
    """After choosing a common type, every branch must convert to it implicitly."""
    for i, t in enumerate(types):
        if t.name == "unknown" or t == target:
            continue
        if not T.can_coerce(t, target):
            pos = positions[i] if positions else None
            raise PgError(f"{context} could not convert type {t.format()} to {target.format()}", "42846", pos=pos)
