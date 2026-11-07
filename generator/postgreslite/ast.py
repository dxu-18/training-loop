"""The parse tree (after PostgreSQL's parsenodes.h, simplified). Every node keeps `pos`, the 0-based offset of its
first character in the statement, for error positions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(eq=False)
class Node:
    pass


# ------------------------------------------------------------------ expressions

@dataclass(eq=False)
class Const(Node):
    value: Any            # int / str (numeric text) / str / bool / None
    kind: str             # int numeric str bitstr bool null
    pos: int = -1


@dataclass(eq=False)
class TypeName(Node):
    names: list           # ["int4"] or ["public", "mytype"]
    mods: list = field(default_factory=list)      # typmod expressions (Const)
    array: int = 0        # number of [] suffixes
    pos: int = -1
    setof: bool = False
    tz: bool | None = None          # TIMESTAMP/TIME WITH (True) / WITHOUT (False) TIME ZONE
    interval_fields: str | None = None


@dataclass(eq=False)
class TypedLiteral(Node):          # DATE '2024-01-01', INTERVAL '1 day'
    typ: TypeName
    value: str
    pos: int = -1


@dataclass(eq=False)
class ColumnRef(Node):
    names: list           # ["u", "name"], ["name"], ["u", "*"], ["*"]
    pos: int = -1


@dataclass(eq=False)
class Param(Node):
    n: int
    pos: int = -1


@dataclass(eq=False)
class Op(Node):            # binary (left, right) or prefix (left None)
    op: str
    left: Any
    right: Any
    pos: int = -1


@dataclass(eq=False)
class BoolOp(Node):
    op: str               # and or not
    args: list
    pos: int = -1


@dataclass(eq=False)
class FuncCall(Node):
    name: list            # ["count"], ["pg_catalog", "lower"]
    args: list
    pos: int = -1
    star: bool = False
    distinct: bool = False
    order_by: list = field(default_factory=list)
    within_group: bool = False
    filter: Any = None
    over: Any = None      # WindowDef
    variadic: bool = False
    argnames: list = field(default_factory=list)  # named args (None for positional)
    special: str | None = None     # how it was written: "extract", "substring_from", "trim", "position" ...


@dataclass(eq=False)
class SQLValue(Node):      # CURRENT_DATE, CURRENT_TIMESTAMP, CURRENT_USER ...
    name: str
    precision: int | None = None
    pos: int = -1


@dataclass(eq=False)
class Cast(Node):
    arg: Any
    typ: TypeName
    pos: int = -1
    style: str = "::"     # :: or CAST


@dataclass(eq=False)
class Case(Node):
    arg: Any
    whens: list           # [(cond, result)]
    default: Any
    pos: int = -1


@dataclass(eq=False)
class NullTest(Node):
    arg: Any
    negated: bool
    pos: int = -1


@dataclass(eq=False)
class BoolTest(Node):      # IS [NOT] TRUE / FALSE / UNKNOWN
    arg: Any
    what: str             # true false unknown
    negated: bool
    pos: int = -1


@dataclass(eq=False)
class Distinct(Node):      # IS [NOT] DISTINCT FROM
    left: Any
    right: Any
    negated: bool         # IS NOT DISTINCT FROM
    pos: int = -1


@dataclass(eq=False)
class InList(Node):
    arg: Any
    items: list
    negated: bool
    pos: int = -1


@dataclass(eq=False)
class Between(Node):
    arg: Any
    lo: Any
    hi: Any
    negated: bool
    symmetric: bool = False
    pos: int = -1


@dataclass(eq=False)
class Like(Node):
    arg: Any
    pattern: Any
    escape: Any
    negated: bool
    kind: str             # like ilike similar
    pos: int = -1


@dataclass(eq=False)
class SubLink(Node):
    kind: str             # exists expr any all in array
    query: Any
    test: Any = None      # left operand for any/all/in
    op: str | None = None
    pos: int = -1


@dataclass(eq=False)
class AnyAll(Node):        # x op ANY (array expr)
    kind: str             # any all
    op: str
    left: Any
    right: Any
    pos: int = -1


@dataclass(eq=False)
class ArrayExpr(Node):
    items: list
    pos: int = -1


@dataclass(eq=False)
class RowExpr(Node):
    items: list
    pos: int = -1
    explicit: bool = False


@dataclass(eq=False)
class Subscript(Node):     # arg[idx] / arg[lo:hi]
    arg: Any
    lo: Any
    hi: Any
    slice: bool
    pos: int = -1


@dataclass(eq=False)
class FieldSel(Node):      # (expr).field / (expr).*
    arg: Any
    name: str
    pos: int = -1


@dataclass(eq=False)
class Collate(Node):
    arg: Any
    collation: list
    pos: int = -1


@dataclass(eq=False)
class AtTimeZone(Node):
    arg: Any
    zone: Any
    pos: int = -1


@dataclass(eq=False)
class Grouping(Node):      # GROUPING(a, b)
    args: list
    pos: int = -1


@dataclass(eq=False)
class SetToDefault(Node):  # DEFAULT in INSERT VALUES / UPDATE SET
    pos: int = -1


@dataclass(eq=False)
class Star(Node):          # a bare * in a target list
    pos: int = -1


# ------------------------------------------------------------------ query parts

@dataclass(eq=False)
class ResTarget(Node):
    expr: Any
    name: str | None
    pos: int = -1


@dataclass(eq=False)
class SortBy(Node):
    expr: Any
    desc: bool = False
    nulls: str | None = None     # first last
    using: str | None = None
    pos: int = -1


@dataclass(eq=False)
class FrameBound(Node):
    kind: str             # unbounded_preceding preceding current_row following unbounded_following
    offset: Any = None
    pos: int = -1


@dataclass(eq=False)
class WindowDef(Node):
    name: str | None = None          # WINDOW w AS (...) : its name
    ref: str | None = None           # OVER (w ORDER BY ...) / OVER w
    partition: list = field(default_factory=list)
    order: list = field(default_factory=list)
    frame_mode: str | None = None    # rows range groups
    start: Any = None
    end: Any = None
    exclude: str | None = None       # current_row group ties no_others
    pos: int = -1
    bare_ref: bool = False           # OVER w (no parentheses)


@dataclass(eq=False)
class GroupingSet(Node):
    kind: str             # rollup cube sets empty
    items: list
    pos: int = -1


@dataclass(eq=False)
class Alias(Node):
    name: str
    cols: list = field(default_factory=list)
    pos: int = -1


@dataclass(eq=False)
class RangeVar(Node):
    schema: str | None
    name: str
    alias: Alias | None = None
    only: bool = False
    pos: int = -1
    sample: Any = None    # (method, [args], repeatable)


@dataclass(eq=False)
class RangeSubselect(Node):
    query: Any
    alias: Alias | None
    lateral: bool = False
    pos: int = -1


@dataclass(eq=False)
class RangeFunction(Node):
    funcs: list           # [FuncCall]
    alias: Alias | None
    lateral: bool = False
    ordinality: bool = False
    pos: int = -1
    coldefs: list = field(default_factory=list)


@dataclass(eq=False)
class JoinExpr(Node):
    kind: str             # inner left right full cross
    left: Any
    right: Any
    quals: Any = None
    using: list | None = None
    natural: bool = False
    alias: Alias | None = None
    pos: int = -1
    using_alias: str | None = None


@dataclass(eq=False)
class CTE(Node):
    name: str
    cols: list
    query: Any
    materialized: bool | None = None
    pos: int = -1
    search: Any = None
    cycle: Any = None


@dataclass(eq=False)
class With(Node):
    ctes: list
    recursive: bool
    pos: int = -1


@dataclass(eq=False)
class Select(Node):
    targets: list = field(default_factory=list)
    from_: list = field(default_factory=list)
    where: Any = None
    group_by: list = field(default_factory=list)
    group_distinct: bool = False
    having: Any = None
    windows: list = field(default_factory=list)
    distinct: Any = None  # None, True (DISTINCT), or [exprs] (DISTINCT ON)
    into: Any = None
    # wrapping clauses (also on SetOp / Values)
    with_: With | None = None
    order_by: list = field(default_factory=list)
    limit: Any = None
    offset: Any = None
    with_ties: bool = False
    locking: list = field(default_factory=list)
    pos: int = -1
    fetch_first: bool = False


@dataclass(eq=False)
class Values(Node):
    rows: list
    with_: With | None = None
    order_by: list = field(default_factory=list)
    limit: Any = None
    offset: Any = None
    with_ties: bool = False
    locking: list = field(default_factory=list)
    pos: int = -1
    fetch_first: bool = False


@dataclass(eq=False)
class SetOp(Node):
    op: str               # union intersect except
    all: bool
    left: Any
    right: Any
    with_: With | None = None
    order_by: list = field(default_factory=list)
    limit: Any = None
    offset: Any = None
    with_ties: bool = False
    locking: list = field(default_factory=list)
    pos: int = -1
    fetch_first: bool = False


# ------------------------------------------------------------------ statements

@dataclass(eq=False)
class OnConflict(Node):
    cols: list            # inference columns (exprs)
    constraint: str | None
    where: Any            # inference WHERE
    action: str           # nothing update
    sets: list = field(default_factory=list)
    update_where: Any = None
    pos: int = -1


@dataclass(eq=False)
class SetClause(Node):
    cols: list            # [(name, [subscripts], pos)]
    value: Any            # expr, or for (a, b) = (...): RowExpr / SubLink
    multi: bool = False
    pos: int = -1


@dataclass(eq=False)
class Insert(Node):
    table: RangeVar
    cols: list            # [(name, pos)]
    source: Any           # Select / Values / None (DEFAULT VALUES)
    on_conflict: OnConflict | None = None
    returning: list = field(default_factory=list)
    with_: With | None = None
    overriding: str | None = None
    pos: int = -1


@dataclass(eq=False)
class Update(Node):
    table: RangeVar
    sets: list
    from_: list
    where: Any
    returning: list = field(default_factory=list)
    with_: With | None = None
    pos: int = -1


@dataclass(eq=False)
class Delete(Node):
    table: RangeVar
    using: list
    where: Any
    returning: list = field(default_factory=list)
    with_: With | None = None
    pos: int = -1


@dataclass(eq=False)
class MergeWhen(Node):
    matched: bool
    cond: Any
    action: str           # update delete insert nothing
    sets: list = field(default_factory=list)
    cols: list = field(default_factory=list)
    values: list | None = None
    by_source: bool = False
    pos: int = -1


@dataclass(eq=False)
class Merge(Node):
    target: RangeVar
    source: Any
    on: Any
    whens: list
    with_: With | None = None
    pos: int = -1


@dataclass(eq=False)
class Constraint(Node):
    kind: str             # notnull null default check primary unique foreign identity generated
    name: str | None = None
    expr: Any = None      # check / default / generated expression
    cols: list = field(default_factory=list)          # table constraint columns
    ref_table: RangeVar | None = None
    ref_cols: list = field(default_factory=list)
    on_delete: str = "a"  # a no action, r restrict, c cascade, n set null, d set default
    on_update: str = "a"
    identity: str | None = None   # always / by default
    pos: int = -1
    expr_text: str = ""
    deferrable: bool = False
    nulls_not_distinct: bool = False
    include: list = field(default_factory=list)
    seq_opts: list = field(default_factory=list)
    not_valid: bool = False


@dataclass(eq=False)
class ColumnDef(Node):
    name: str
    typ: TypeName
    constraints: list
    pos: int = -1
    collate: Any = None


@dataclass(eq=False)
class CreateTable(Node):
    table: RangeVar
    elements: list        # ColumnDef / Constraint / ("like", RangeVar)
    if_not_exists: bool = False
    temp: bool = False
    unlogged: bool = False
    as_query: Any = None
    as_cols: list = field(default_factory=list)
    with_data: bool = True
    pos: int = -1
    on_commit: str | None = None
    partition_by: Any = None


@dataclass(eq=False)
class AlterCmd(Node):
    kind: str             # add_column drop_column rename_column rename_table alter_type set_default drop_default
                          # set_not_null drop_not_null add_constraint drop_constraint rename_constraint set_schema
                          # add_identity validate_constraint
    name: str | None = None
    newname: str | None = None
    coldef: ColumnDef | None = None
    typ: TypeName | None = None
    using: Any = None
    expr: Any = None
    constraint: Constraint | None = None
    if_exists: bool = False
    if_not_exists: bool = False
    cascade: bool = False
    pos: int = -1


@dataclass(eq=False)
class AlterTable(Node):
    table: RangeVar
    cmds: list
    if_exists: bool = False
    kind: str = "table"   # table view materialized view index sequence
    pos: int = -1


@dataclass(eq=False)
class Drop(Node):
    kind: str             # table view materialized_view index schema sequence database type function
    names: list           # [RangeVar]
    if_exists: bool = False
    cascade: bool = False
    pos: int = -1
    concurrently: bool = False


@dataclass(eq=False)
class IndexElem(Node):
    expr: Any             # ColumnRef-ish name (str) or expression
    name: str | None      # plain column name
    desc: bool = False
    nulls: str | None = None
    opclass: list = field(default_factory=list)
    pos: int = -1
    expr_text: str = ""


@dataclass(eq=False)
class CreateIndex(Node):
    name: str | None
    table: RangeVar
    elems: list
    unique: bool = False
    method: str = "btree"
    where: Any = None
    if_not_exists: bool = False
    concurrently: bool = False
    include: list = field(default_factory=list)
    pos: int = -1
    where_text: str = ""
    nulls_not_distinct: bool = False


@dataclass(eq=False)
class CreateView(Node):
    view: RangeVar
    cols: list
    query: Any
    query_text: str
    replace: bool = False
    materialized: bool = False
    with_data: bool = True
    temp: bool = False
    if_not_exists: bool = False
    pos: int = -1
    check_option: str | None = None


@dataclass(eq=False)
class RefreshMatView(Node):
    view: RangeVar
    concurrently: bool = False
    with_data: bool = True
    pos: int = -1


@dataclass(eq=False)
class Truncate(Node):
    tables: list
    restart_identity: bool = False
    cascade: bool = False
    pos: int = -1


@dataclass(eq=False)
class CreateSchema(Node):
    name: str
    if_not_exists: bool = False
    authorization: str | None = None
    pos: int = -1


@dataclass(eq=False)
class CreateSequence(Node):
    seq: RangeVar
    options: list
    if_not_exists: bool = False
    temp: bool = False
    pos: int = -1


@dataclass(eq=False)
class CreateDatabase(Node):
    name: str
    options: list = field(default_factory=list)
    pos: int = -1


@dataclass(eq=False)
class CreateType(Node):
    name: RangeVar
    labels: list          # enum labels
    pos: int = -1


@dataclass(eq=False)
class CreateExtension(Node):
    name: str
    if_not_exists: bool = False
    pos: int = -1


@dataclass(eq=False)
class Transaction(Node):
    kind: str             # begin start commit rollback savepoint release rollback_to prepare
    name: str | None = None
    options: list = field(default_factory=list)
    chain: bool = False
    pos: int = -1
    word: str = ""        # the keyword used (BEGIN / START / END / ABORT ...) for the command tag


@dataclass(eq=False)
class Explain(Node):
    stmt: Any
    options: dict
    pos: int = -1


@dataclass(eq=False)
class Analyze(Node):
    tables: list
    verbose: bool = False
    vacuum: bool = False
    full: bool = False
    pos: int = -1


@dataclass(eq=False)
class VarSet(Node):
    name: str
    values: list          # raw values (str) ; [] -> DEFAULT
    local: bool = False
    reset: bool = False
    pos: int = -1


@dataclass(eq=False)
class VarShow(Node):
    name: str
    pos: int = -1


@dataclass(eq=False)
class Copy(Node):
    table: RangeVar | None
    cols: list
    query: Any
    to: bool               # COPY ... TO (True) / FROM (False)
    target: str | None     # file name; None = STDIN/STDOUT
    options: dict = field(default_factory=dict)
    pos: int = -1


@dataclass(eq=False)
class Comment(Node):
    kind: str
    target: list
    text: str | None
    pos: int = -1


@dataclass(eq=False)
class Prepare(Node):
    name: str
    types: list
    query: Any
    pos: int = -1


@dataclass(eq=False)
class Execute(Node):
    name: str
    params: list
    pos: int = -1


@dataclass(eq=False)
class Deallocate(Node):
    name: str | None
    pos: int = -1


@dataclass(eq=False)
class Grant(Node):
    text: str
    pos: int = -1


@dataclass(eq=False)
class Noop(Node):          # statements accepted and ignored (LISTEN, DISCARD ...), with their command tag
    tag: str
    pos: int = -1


@dataclass(eq=False)
class AlterSimple(Node):   # ALTER VIEW/INDEX/SEQUENCE/SCHEMA ... RENAME TO / ALTER SEQUENCE ... RESTART
    kind: str
    name: RangeVar
    action: str
    newname: str | None = None
    options: list = field(default_factory=list)
    if_exists: bool = False
    pos: int = -1
